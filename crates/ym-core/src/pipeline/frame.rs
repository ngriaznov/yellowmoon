//! Frames: geometry-sampled camera RGB, the per-pixel colour pass (CPU or GPU), viewport and fit
//! frames, full-resolution renders and pixel readouts.
//!
//! Every frame is made the same way (`docs/01` section Render graph, "Geometry ... applied when
//! sampling for display and export"):
//!
//! 1. choose the raw group's resolution from the zoom (never upscaled: the image has at least as
//!    many pixels as the viewport shows; 1/4 below 25 % zoom, 1/2 up to 50 %, full above),
//! 2. build the [`Grid`] that maps output pixels to positions in that image
//!    through orientation, straighten and crop,
//! 3. sample the raw group's camera RGB (copy, bicubic, or box-filtered bilinear when shrinking),
//! 4. run the colour chain over the sampled pixels: C1 to C13 on the CPU reference path, or the same
//!    chain as one compute pass when a GPU is attached (the two agree within the parity tolerances
//!    of `ym-gpu`), and
//! 5. paint the letterbox where the grid found no photo.
//!
//! Because geometry is applied *before* the pointwise colour chain, a crop or a rotation never
//! re-runs a raw or colour group, and the colour chain only touches the pixels that are shown.

use std::sync::Arc;

use rayon::prelude::*;
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, Frame, Layout, OutputTarget, View};
use ym_gpu::cpu::resample::sample_catmull_rom;
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::kernels::color::{ColorOutput, PreparedChain};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

use super::RenderError;
use super::colour::{GradePlan, PhotoSource};
use super::geometry::{Affine, Grid, oriented_to_sensor, sensor_to_oriented};
use super::graph::{ComputeDevice, Job, RenderReport, Renderer};
use super::viewport::ViewportState;

/// A finished 8-bit frame: RGBA, Display P3 for the viewport, sRGB for thumbnails.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedFrame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `4 * width * height` bytes, row-major. Alpha is 255 everywhere (the letterbox is painted).
    pub rgba: Vec<u8>,
    /// False when the frame was made from a coarser image than the zoom asks for because the
    /// full-resolution tiles were not ready ([`FrameOptions::coarse_only`]).
    pub sharp: bool,
    /// What the request did.
    pub report: RenderReport,
}

/// The frame of a viewport state (an alias that names the use).
pub type ViewportFrame = RenderedFrame;
/// The frame of a photo at Fit (an alias that names the use).
pub type FitFrame = RenderedFrame;

/// Options of [`Renderer::viewport_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameOptions {
    /// Never compute full-resolution tiles: when the zoom needs them and they are not cached yet,
    /// serve the frame from the reduced image (blurry but immediate) and mark it not sharp. The app
    /// then asks for the tiles at `TileNeural` priority and renders again.
    pub coarse_only: bool,
    /// Do not read or write the frame cache.
    pub no_cache: bool,
}

/// Display values under a point, 0 to 255 per channel (`sample_pixel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PixelSample {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
}

/// What a full-resolution render produces.
///
/// The first target is the picture of a JPEG, TIFF or PNG: orientation, straighten and crop are
/// baked into the pixels. The other two are the two images a DNG holds (`docs/09`, "Pixels" and
/// `ym_export::dng::DngParams`): the raw pixels stay as the sensor wrote them, the user's turns go
/// into the `Orientation` tag and a crop without a straighten angle into the default crop tags, so
/// that other editors can undo them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullTarget {
    /// Display-linear RGB in the output's primaries (C1 to C12): what the JPEG, TIFF and PNG
    /// writers resize in linear light and encode. Orientation (the camera's and the user's),
    /// straighten and crop are applied.
    DisplayLinear(OutputTarget),
    /// The DNG's raw pixels: C1 to C11 decoded back to linear AWG4, in the buffer's own
    /// orientation (after the lens stage's auto crop), neither turned nor, without a straighten
    /// angle, cropped: the user's crop is then [`FullImage::window`] for the writer to put in the
    /// default crop tags. With a straighten angle the angle and the crop are baked in (a DNG has no
    /// rotated crop), still unturned, and `window` is `None`. Not available with a look that
    /// includes its own display rendering.
    DngLinearAwg4,
    /// The DNG's preview: the exact render in sRGB (display-linear, Rec.709 primaries) in the
    /// buffer's orientation, cropped and straightened as the user sees it, so it has the size of
    /// [`FullImage::window`] when the raw pixels carry one.
    DngPreview,
}

/// A request for [`Renderer::render_full`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullRequest {
    /// What to produce.
    pub target: FullTarget,
}

/// A rectangle of whole pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelWindow {
    /// Left edge.
    pub x: usize,
    /// Top edge.
    pub y: usize,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

/// A full-resolution image (see [`FullTarget`] for what is applied), interleaved f32.
#[derive(Debug, Clone, PartialEq)]
pub struct FullImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `3 * width * height` samples, `r, g, b` per pixel, row-major. Pixels outside the straightened
    /// photo are zero.
    pub rgb: Vec<f32>,
    /// What the samples are.
    pub target: FullTarget,
    /// For [`FullTarget::DngLinearAwg4`]: the user's crop in pixels of this image, for the DNG's
    /// default crop tags (`None`: the whole image, or the crop was baked in with a straighten
    /// angle). Always `None` for the other targets.
    pub window: Option<PixelWindow>,
    /// What the render did.
    pub report: RenderReport,
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

/// Camera RGB sampled through a grid, with a mask of the pixels that fell outside the photo.
pub(crate) struct Sampled {
    pub img: RgbImage,
    /// 1 where the pixel is outside the photo; `None` when every pixel is inside.
    pub outside: Option<Vec<u8>>,
}

/// The sampling kernel the grid asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    /// Read the pixel the grid points at.
    Copy,
    /// Catmull-Rom bicubic.
    Bicubic,
    /// Bilinear averaged over `n x n` sub-samples per output pixel (shrinking).
    Box(usize),
}

fn kernel_for(grid: &Grid) -> Kernel {
    if grid.is_pixel_copy() {
        return Kernel::Copy;
    }
    let (sx, sy) = grid.step_lengths();
    let ratio = sx.max(sy);
    if ratio <= 1.25 {
        Kernel::Bicubic
    } else {
        Kernel::Box((ratio - 0.25).ceil().clamp(2.0, 4.0) as usize)
    }
}

#[inline]
fn bilinear(plane: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> f32 {
    let (ux, uy) = (
        (sx - 0.5).clamp(0.0, w as f64 - 1.0),
        (sy - 0.5).clamp(0.0, h as f64 - 1.0),
    );
    let (x0, y0) = (ux.floor() as usize, uy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((ux - x0 as f64) as f32, (uy - y0 as f64) as f32);
    let a = plane[y0 * w + x0] * (1.0 - fx) + plane[y0 * w + x1] * fx;
    let b = plane[y1 * w + x0] * (1.0 - fx) + plane[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

/// Samples the `w x h` output pixels at offset `off` of `grid` from `src`, an image whose pixel
/// `(0, 0)` is buffer pixel `origin`.
pub(crate) fn sample_to_camera(
    src: &RgbImage,
    origin: (usize, usize),
    grid: &Grid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Sampled {
    let kernel = kernel_for(grid);
    let mut img = RgbImage::new(w, h);
    let mut outside = vec![0u8; w * h];
    let n = w * h;
    let (rp, rest) = img.data.split_at_mut(n);
    let (gp, bp) = rest.split_at_mut(n);
    let (sw, sh) = (src.width, src.height);
    let planes = [src.plane(0), src.plane(1), src.plane(2)];
    let (ox, oy) = (origin.0 as f64, origin.1 as f64);
    (
        rp.par_chunks_mut(w.max(1)),
        gp.par_chunks_mut(w.max(1)),
        bp.par_chunks_mut(w.max(1)),
        outside.par_chunks_mut(w.max(1)),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(y, (r, g, b, out))| {
            let rows = [r, g, b];
            for x in 0..w {
                let (gx, gy) = (off.0 + x, off.1 + y);
                match kernel {
                    Kernel::Copy | Kernel::Bicubic => {
                        let Some((sx, sy)) = grid.source(gx, gy) else {
                            out[x] = 1;
                            continue;
                        };
                        let (px, py) = (sx - ox, sy - oy);
                        for c in 0..3 {
                            rows[c][x] = if kernel == Kernel::Copy {
                                let ix = (px.floor() as isize).clamp(0, sw as isize - 1) as usize;
                                let iy = (py.floor() as isize).clamp(0, sh as isize - 1) as usize;
                                planes[c][iy * sw + ix]
                            } else {
                                sample_catmull_rom(planes[c], sw, sh, px, py)
                            };
                        }
                    }
                    Kernel::Box(taps) => {
                        let mut acc = [0.0f32; 3];
                        let mut count = 0u32;
                        for j in 0..taps {
                            for i in 0..taps {
                                let (fx, fy) = (
                                    gx as f64 + (i as f64 + 0.5) / taps as f64,
                                    gy as f64 + (j as f64 + 0.5) / taps as f64,
                                );
                                if let Some((sx, sy)) = grid.source_at(fx, fy) {
                                    for c in 0..3 {
                                        acc[c] += bilinear(planes[c], sw, sh, sx - ox, sy - oy);
                                    }
                                    count += 1;
                                }
                            }
                        }
                        if count == 0 {
                            out[x] = 1;
                        } else {
                            for c in 0..3 {
                                rows[c][x] = acc[c] / count as f32;
                            }
                        }
                    }
                }
            }
        });
    let any_outside = outside.iter().any(|&v| v != 0);
    Sampled {
        img,
        outside: any_outside.then_some(outside),
    }
}

// ---------------------------------------------------------------------------------------------
// The colour pass
// ---------------------------------------------------------------------------------------------

/// Runs the colour chain over `sampled` and returns RGBA8 with the letterbox painted, and where it
/// ran. The GPU is used for the final view when there is one; on any GPU error the CPU path runs
/// instead and the note says why.
pub(crate) fn colour_pass(
    job: &mut Job<'_>,
    sampled: Sampled,
    grade: &GradePlan,
    before: bool,
    background: [u8; 3],
    priority: Priority,
) -> Result<(Vec<u8>, ComputeDevice), RenderError> {
    job.check()?;
    let started = std::time::Instant::now();
    let (w, h) = (sampled.img.width, sampled.img.height);
    let n = w * h;
    let mut rgba = vec![255u8; 4 * n];
    let mut device = ComputeDevice::Cpu;
    let mut done = false;
    if let (Some(gpu), false) = (job.r.options().gpu.clone(), before) {
        let chain = Arc::clone(&grade.chain);
        let img = Arc::new(sampled.img.clone());
        let id = u64::from(job.src.id.0);
        let result = gpu.run(JobSpec::new(priority).owned_by(id), move |ctx, cancel| {
            cancel.check()?;
            let prepared = PreparedChain::new(ctx, &chain)?;
            prepared.render(ctx, &img, (0, 0), ColorOutput::Rgba8 { dither: true })
        });
        match result {
            Ok(out) => {
                if let ym_gpu::kernels::PixelData::Rgba8(bytes) = out.data
                    && bytes.len() == 4 * n
                {
                    rgba = bytes;
                    device = ComputeDevice::Gpu;
                    done = true;
                }
            }
            Err(e) => job.notes.push(format!(
                "The graphics card could not render this frame, so the CPU did: {e}."
            )),
        }
    }
    if !done {
        let frame = Frame::planar(w, h, &sampled.img.data)
            .map_err(|e| RenderError::Colour(e.to_string()))?;
        let mut rgb = vec![0u8; 3 * n];
        let view = if before { View::Before } else { View::Final };
        grade
            .chain
            .render_u8(view, &frame, (0, 0), &mut rgb, job.r.options().threads)
            .map_err(|e| RenderError::Colour(e.to_string()))?;
        rgba.par_chunks_mut(4)
            .zip(rgb.par_chunks(3))
            .for_each(|(d, s)| {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = 255;
            });
    }
    job.runs.colour_passes += 1;
    job.timings.pass_ms += started.elapsed().as_secs_f64() * 1000.0;
    if let Some(outside) = &sampled.outside {
        rgba.par_chunks_mut(4)
            .zip(outside.par_iter())
            .for_each(|(d, &o)| {
                if o != 0 {
                    d[0] = background[0];
                    d[1] = background[1];
                    d[2] = background[2];
                    d[3] = 255;
                }
            });
    }
    Ok((rgba, device))
}

// ---------------------------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------------------------

/// The reduction of the raw group a zoom needs (device pixels per full-resolution image pixel):
/// the image must have at least as many pixels as it is shown with, and is never upscaled.
#[must_use]
pub fn scale_for_zoom(zoom: f64) -> usize {
    if zoom <= 0.25 {
        4
    } else if zoom <= 0.5 {
        2
    } else {
        1
    }
}

/// One view of a photo: an output grid and the map from it to the final image.
#[derive(Debug, Clone)]
pub(crate) struct ViewSpec {
    pub out_w: usize,
    pub out_h: usize,
    /// Output pixels to pixels of the final image at full resolution.
    pub view_to_final: Affine,
    /// The reduction of the raw group wanted (before the plan's limits).
    pub k: usize,
    pub target: OutputTarget,
    pub before: bool,
    pub coarse_only: bool,
    pub cache: bool,
    pub background: [u8; 3],
    /// The queue priority of the colour pass when it runs on the GPU: a thumbnail's pass must not
    /// jump ahead of the viewport's.
    pub gpu_priority: Priority,
}

/// The sampled camera RGB of a view and how it was found.
pub(crate) struct SampledView {
    pub sampled: Sampled,
    pub scale: usize,
    pub sharp: bool,
}

/// Steps 1 to 3 of the module documentation.
pub(crate) fn sample_view(job: &mut Job<'_>, spec: &ViewSpec) -> Result<SampledView, RenderError> {
    let plan = job.plan()?;
    let geom = job.geometry();
    // A camera-processed file has no mosaic, so any scale works; the limit that keeps a reduced
    // image at least a few pixels across applies to both.
    let mut k = plan.effective_scale(spec.k);
    let mut sharp = true;
    // Full-resolution tiles of a mosaic: only when they are ready or may be computed.
    if !plan.reduced && k == 1 {
        let grid = geom
            .at_full_resolution()
            .grid_from((plan.out_w, plan.out_h), spec.view_to_final);
        if let Some((x0, y0, x1, y1)) = grid.source_bounds(0, 0, spec.out_w, spec.out_h, 3) {
            let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
            if spec.coarse_only && !job.tiles_ready(rect)? {
                k = plan.effective_scale(2).max(2);
                sharp = false;
            } else {
                let region = job.region(rect, true)?;
                let t = std::time::Instant::now();
                let sampled =
                    sample_to_camera(&region, (x0, y0), &grid, (0, 0), spec.out_w, spec.out_h);
                job.timings.sample_ms += t.elapsed().as_secs_f64() * 1000.0;
                return Ok(SampledView {
                    sampled,
                    scale: 1,
                    sharp,
                });
            }
        } else {
            // Nothing of the photo is in view.
            return Ok(SampledView {
                sampled: Sampled {
                    img: RgbImage::new(spec.out_w, spec.out_h),
                    outside: Some(vec![1; spec.out_w * spec.out_h]),
                },
                scale: 1,
                sharp,
            });
        }
    }
    let img = job.group(k)?;
    let inv = 1.0 / k as f64;
    let v2f = spec
        .view_to_final
        .then(&Affine::scale_translate(inv, inv, 0.0, 0.0));
    let grid = geom.grid_from((img.width, img.height), v2f);
    let t = std::time::Instant::now();
    let sampled = sample_to_camera(&img, (0, 0), &grid, (0, 0), spec.out_w, spec.out_h);
    job.timings.sample_ms += t.elapsed().as_secs_f64() * 1000.0;
    Ok(SampledView {
        sampled,
        scale: k,
        sharp,
    })
}

/// A whole view: sample, colour pass, report. Frames of a viewport go through the frame cache.
pub(crate) fn render_view(mut job: Job<'_>, spec: &ViewSpec) -> Result<RenderedFrame, RenderError> {
    let grade = job.grade(spec.target, true)?;
    let cache_key = spec
        .cache
        .then(|| frame_key(&mut job, spec, &grade))
        .transpose()?;
    if let Some(key) = &cache_key
        && let Some(hit) = job.g.cached_frame(key)
        && !(spec.coarse_only && !hit.sharp)
    {
        job.runs.frame_hits += 1;
        let (scale, device) = (hit.report.scale, hit.report.device);
        let mut frame = (*hit).clone();
        frame.report = job.finish(scale, device, spec.target);
        return Ok(frame);
    }
    let view = sample_view(&mut job, spec)?;
    let (rgba, device) = colour_pass(
        &mut job,
        view.sampled,
        &grade,
        spec.before,
        spec.background,
        spec.gpu_priority,
    )?;
    let (scale, sharp) = (view.scale, view.sharp);
    let graph = Arc::clone(&job.g);
    let report = job.finish(scale, device, spec.target);
    let frame = RenderedFrame {
        width: spec.out_w,
        height: spec.out_h,
        rgba,
        sharp,
        report,
    };
    if let Some(key) = cache_key {
        graph.store_frame(key, Arc::new(frame.clone()));
    }
    Ok(frame)
}

fn frame_key(job: &mut Job<'_>, spec: &ViewSpec, grade: &GradePlan) -> Result<String, RenderError> {
    let params = job.params()?;
    Ok(format!(
        "{:?}|{:?}|{:?}|{:?}|{}x{}|{:?}|{}|{}",
        params.key(),
        grade.key,
        job.geometry(),
        spec.view_to_final,
        spec.out_w,
        spec.out_h,
        spec.k,
        spec.before,
        spec.coarse_only
    ))
}

// ---------------------------------------------------------------------------------------------
// Public API on the renderer
// ---------------------------------------------------------------------------------------------

impl Renderer {
    /// The size of the final image (after the lens crop, orientation, straighten and crop) at full
    /// resolution: what the viewport's zoom and the export size are measured against.
    ///
    /// # Errors
    ///
    /// [`RenderError`] when the photo's data cannot be prepared.
    pub fn final_size(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<(usize, usize), RenderError> {
        Job::new(self, src, cancel).final_size()
    }

    /// The frame of a viewport state: the photo in the cell at the zoom and pan of `state`, in
    /// Display P3, with the letterbox painted (`docs/01` section Native viewport).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn viewport_frame(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        options: &FrameOptions,
        cancel: &CancelToken,
    ) -> Result<RenderedFrame, RenderError> {
        let mut job = Job::new(self, src, cancel);
        let image = job.final_size()?;
        let (dw, dh) = state.device_size();
        let zoom = state.zoom(image);
        let spec = ViewSpec {
            out_w: dw,
            out_h: dh,
            view_to_final: state.view_to_final(image),
            k: scale_for_zoom(zoom),
            target: OutputTarget::Viewport,
            before: state.before,
            coarse_only: options.coarse_only,
            cache: !options.no_cache,
            background: self.options().background,
            gpu_priority: Priority::Viewport,
        };
        let frame = render_view(job, &spec)?;
        // The header's status text is the last *viewport* render: a thumbnail, a look tile, a pixel
        // readout or an export of the same photo must not replace it.
        self.note_stats(src.id, frame.report.stats());
        Ok(frame)
    }

    /// Computes the full-resolution tiles a viewport state shows, so that the next
    /// [`Renderer::viewport_frame`] is sharp. This is the `TileNeural`-priority job of the preview
    /// strategy; today the tiles are classical.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn prepare_tiles(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        cancel: &CancelToken,
    ) -> Result<usize, RenderError> {
        let mut job = Job::new(self, src, cancel);
        let plan = job.plan()?;
        let image = job.final_size()?;
        if plan.reduced || scale_for_zoom(state.zoom(image)) != 1 {
            return Ok(0);
        }
        let (dw, dh) = state.device_size();
        let grid = job
            .geometry()
            .at_full_resolution()
            .grid_from((plan.out_w, plan.out_h), state.view_to_final(image));
        let Some((x0, y0, x1, y1)) = grid.source_bounds(0, 0, dw, dh, 3) else {
            return Ok(0);
        };
        let _ = job.region(Rect::new(x0, y0, x1 - x0, y1 - y0), true)?;
        let made = job.runs.tiles;
        job.finish(1, ComputeDevice::Cpu, OutputTarget::Viewport);
        Ok(made as usize)
    }

    /// Display values (0 to 255) under a point of the cell, as the viewport shows them (the Before
    /// view included), without the dither. `None` outside the photo (the letterbox).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn sample_pixel(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        css: (f64, f64),
        cancel: &CancelToken,
    ) -> Result<Option<PixelSample>, RenderError> {
        let mut job = Job::new(self, src, cancel);
        let image = job.final_size()?;
        let Some((fx, fy)) = state.final_point_at(image, css) else {
            return Ok(None);
        };
        let zoom = state.zoom(image);
        // A 1 x 1 view of exactly that point, at the scale the viewport uses.
        let spec = ViewSpec {
            out_w: 1,
            out_h: 1,
            view_to_final: Affine::scale_translate(1.0, 1.0, fx - 0.5, fy - 0.5),
            k: scale_for_zoom(zoom),
            target: OutputTarget::Viewport,
            before: state.before,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Viewport,
        };
        let grade = job.grade(OutputTarget::Viewport, false)?;
        let view = sample_view(&mut job, &spec)?;
        if view.sampled.outside.is_some() {
            return Ok(None);
        }
        let rgb = pixel_rgb(&view.sampled.img);
        let kind = if state.before {
            View::Before
        } else {
            View::Final
        };
        let v = grade.chain.apply_view_pixel(kind, rgb);
        job.finish(view.scale, ComputeDevice::Cpu, OutputTarget::Viewport);
        let q = |c: f64| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        Ok(Some(PixelSample {
            r: q(v[0]),
            g: q(v[1]),
            b: q(v[2]),
        }))
    }

    /// Where a click in the cell lands in the space `DevelopSettings::white_pick` uses: fractions
    /// of the photo as the camera oriented it, over the image area before the lens crop. `None`
    /// outside the photo.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data.
    pub fn pick_point_at(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        css: (f64, f64),
        cancel: &CancelToken,
    ) -> Result<Option<[f64; 2]>, RenderError> {
        let mut job = Job::new(self, src, cancel);
        let plan = job.plan()?;
        let image = job.final_size()?;
        let Some((fx, fy)) = state.final_point_at(image, css) else {
            return Ok(None);
        };
        let geom = job.geometry();
        let Some(o) = geom.final_to_oriented((plan.out_w, plan.out_h), fx, fy) else {
            return Ok(None);
        };
        // Oriented fractions of the lens-cropped image -> sensor pixel of the corrected image.
        let s = oriented_to_sensor(geom.exif_orientation, o);
        let (bx, by) = (s[0] * plan.out_w as f64, s[1] * plan.out_h as f64);
        let (ax, ay) = match &plan.lens {
            Some(lens) => {
                let c = lens.crop();
                let (px, py) = (bx + c.x as f64, by + c.y as f64);
                lens.source_position(1, px, py)
            }
            None => (bx, by),
        };
        let area = [
            (ax / plan.width as f64).clamp(0.0, 1.0),
            (ay / plan.height as f64).clamp(0.0, 1.0),
        ];
        Ok(Some(sensor_to_oriented(geom.exif_orientation, area)))
    }

    /// The 5 x 5 median of the demosaiced, `g0`-balanced camera RGB at a Pick white point (the
    /// space of `white_pick`), the sample `wb::picked_scene_white` takes, **without the vignetting gain**
    /// (the contract of the session's own sample, `auto::pick_white_sample`: a gain of 1.3 at the
    /// edge of the frame would make an unclipped grey look clipped to the "too bright" check, and
    /// it scales all three channels alike, so it carries no colour). Unlike the session's sample,
    /// which reads the raw data directly, this one comes from the classical render, so it sees
    /// what the viewport shows; the session's is the one the chain uses.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data, a photo smaller than 5 x 5 pixels, or a spot with no
    /// usable colour.
    pub fn pick_white_sample(
        &self,
        src: &PhotoSource,
        point: [f64; 2],
        cancel: &CancelToken,
    ) -> Result<Vec3, RenderError> {
        let mut job = Job::new(self, src, cancel);
        let plan = job.plan()?;
        if plan.out_w < 5 || plan.out_h < 5 {
            return Err(RenderError::Image(
                "This photo is too small to pick a white from.".to_owned(),
            ));
        }
        let geom = job.geometry();
        let s = oriented_to_sensor(
            geom.exif_orientation,
            [point[0].clamp(0.0, 1.0), point[1].clamp(0.0, 1.0)],
        );
        let target = (s[0] * plan.width as f64, s[1] * plan.height as f64);
        // Find the pixel of the corrected image whose source is that sensor position.
        let (mut px, mut py) = target;
        let crop = plan
            .lens
            .as_ref()
            .map_or(Rect::whole(plan.out_w, plan.out_h), |l| l.crop());
        if let Some(lens) = &plan.lens {
            for _ in 0..4 {
                let (sx, sy) = lens.source_position(1, px, py);
                px += target.0 - sx;
                py += target.1 - sy;
            }
        }
        let cx = (px - crop.x as f64)
            .floor()
            .clamp(2.0, plan.out_w as f64 - 3.0) as usize;
        let cy = (py - crop.y as f64)
            .floor()
            .clamp(2.0, plan.out_h as f64 - 3.0) as usize;
        let region = job.region(Rect::new(cx - 2, cy - 2, 5, 5), false)?;
        // Undo R3 where it was applied (the gain of the radius of the sensor position).
        let gain = plan.vignette.as_ref().map_or(1.0, |v| {
            let (w, h) = (plan.width as f64, plan.height as f64);
            let r = (target.0 - w * 0.5).hypot(target.1 - h * 0.5) / (0.5 * w.hypot(h));
            v.gain_at(r).max(1e-4)
        });
        let pixels: Vec<Vec3> = (0..25)
            .map(|i| {
                let p = region.pixel(i % 5, i / 5);
                [
                    f64::from(p[0]) / gain,
                    f64::from(p[1]) / gain,
                    f64::from(p[2]) / gain,
                ]
            })
            .collect();
        job.finish(1, ComputeDevice::Cpu, OutputTarget::Viewport);
        ym_color::wb::median_rgb(&pixels).ok_or_else(|| {
            RenderError::Image("That spot has no usable colour. Pick another one.".to_owned())
        })
    }

    /// Renders the photo at full resolution (`docs/01` "Export: full resolution"), tile by tile: the
    /// raw group runs on one tile of the lens-cropped image at a time, is sampled through the
    /// geometry and goes through the colour chain, so only one tile of intermediates is alive.
    ///
    /// The reservation in the memory ledger is the output image plus the working set of one tile;
    /// a photo that does not fit the budget is refused with [`RenderError::Memory`].
    ///
    /// Called directly this runs flat out on the calling thread; [`Renderer::spawn_full`] runs the
    /// same render on the export worker, which gives way to viewport frames between tiles.
    ///
    /// # Errors
    ///
    /// [`RenderError::Memory`], [`RenderError::Cancelled`] and the errors of the stages.
    pub fn render_full(
        &self,
        src: &PhotoSource,
        request: &FullRequest,
        cancel: &CancelToken,
    ) -> Result<FullImage, RenderError> {
        self.render_full_with(src, request, cancel, false, None)
    }

    /// [`Renderer::render_full`], optionally waiting for the foreground (viewport frames, tile
    /// passes, thumbnails) to be idle before it starts and before every tile, and reporting the
    /// fraction of tiles done (0 to 1) after each one.
    pub(crate) fn render_full_with(
        &self,
        src: &PhotoSource,
        request: &FullRequest,
        cancel: &CancelToken,
        yield_to_foreground: bool,
        progress: Option<&(dyn Fn(f32) + Sync)>,
    ) -> Result<FullImage, RenderError> {
        let gate = || -> Result<(), RenderError> {
            if yield_to_foreground {
                self.scheduler()
                    .wait_for_foreground(cancel)
                    .map_err(|_| RenderError::Cancelled)?;
            }
            Ok(())
        };
        gate()?;
        let mut job = Job::new(self, src, cancel);
        let plan = job.plan()?;
        // Which geometry the target asks for (see [`FullTarget`]).
        let baked = job.geometry().at_full_resolution();
        let (geom, window) = match request.target {
            FullTarget::DisplayLinear(_) => (baked, None),
            FullTarget::DngLinearAwg4 if baked.is_unrotated() => {
                let (x, y, width, height) = baked
                    .upright(plan.out_w, plan.out_h, true)
                    .crop_window(plan.out_w, plan.out_h);
                let whole = (x, y, width, height) == (0, 0, plan.out_w, plan.out_h);
                (
                    baked.upright(plan.out_w, plan.out_h, false),
                    (!whole).then_some(PixelWindow {
                        x,
                        y,
                        width,
                        height,
                    }),
                )
            }
            FullTarget::DngLinearAwg4 | FullTarget::DngPreview => {
                (baked.upright(plan.out_w, plan.out_h, true), None)
            }
        };
        let (fw, fh) = geom.output_size(plan.out_w, plan.out_h);
        let tile = self.options().full_tile_px.max(64);
        let halo_allowance = 160usize;
        let per_px = 12u64 * 12; // twelve f32 planes of working set per tile pixel, a generous bound
        let working = ((tile + halo_allowance) as u64).pow(2) * per_px;
        let frame_extra = job.frame()?.extra_bytes() as u64;
        let sigma_map = if plan.reduced {
            0
        } else {
            (plan.width * plan.height * 4) as u64
        };
        let output = (fw * fh * 12) as u64;
        let _reservation = self
            .ledger()
            .try_reserve(output + working + frame_extra + sigma_map)
            .map_err(|_| RenderError::Memory)?;
        let target_out = match request.target {
            FullTarget::DisplayLinear(t) => t,
            FullTarget::DngLinearAwg4 => OutputTarget::DisplayP3,
            FullTarget::DngPreview => OutputTarget::Srgb,
        };
        let grade = job.grade(target_out, false)?;
        if matches!(request.target, FullTarget::DngLinearAwg4) && !grade.chain.dng_available() {
            return Err(RenderError::Colour(
                "This look includes its own display rendering, so it can't go into a DNG; pick another look or export TIFF."
                    .to_owned(),
            ));
        }
        let grid = geom.grid((plan.out_w, plan.out_h), (fw, fh));
        let mut rgb = vec![0.0f32; fw * fh * 3];
        let threads = self.options().threads;
        let tiles_total = (fw.div_ceil(tile) * fh.div_ceil(tile)).max(1);
        let mut tiles_done = 0usize;
        let mut oy = 0;
        while oy < fh {
            let th = tile.min(fh - oy);
            let mut ox = 0;
            while ox < fw {
                job.check()?;
                gate()?;
                let tw = tile.min(fw - ox);
                if let Some((x0, y0, x1, y1)) = grid.source_bounds(ox, oy, tw, th, 3) {
                    let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
                    let region = job.region(rect, false)?;
                    let sampled = sample_to_camera(&region, (x0, y0), &grid, (ox, oy), tw, th);
                    let frame = Frame::planar(tw, th, &sampled.img.data)
                        .map_err(|e| RenderError::Colour(e.to_string()))?;
                    let mut out = vec![0.0f32; 3 * tw * th];
                    match request.target {
                        FullTarget::DisplayLinear(_) | FullTarget::DngPreview => grade
                            .chain
                            .render_display_linear_f32(
                                &frame,
                                Layout::Interleaved,
                                &mut out,
                                threads,
                            )
                            .map_err(|e| RenderError::Colour(e.to_string()))?,
                        FullTarget::DngLinearAwg4 => grade
                            .chain
                            .render_dng_linear_awg4_f32(
                                &frame,
                                Layout::Interleaved,
                                &mut out,
                                threads,
                            )
                            .map_err(|e| RenderError::Colour(e.to_string()))?,
                    }
                    if let Some(outside) = &sampled.outside {
                        for (i, &o) in outside.iter().enumerate() {
                            if o != 0 {
                                out[3 * i..3 * i + 3].fill(0.0);
                            }
                        }
                    }
                    for y in 0..th {
                        let d = ((oy + y) * fw + ox) * 3;
                        rgb[d..d + 3 * tw].copy_from_slice(&out[y * tw * 3..(y + 1) * tw * 3]);
                    }
                }
                tiles_done += 1;
                if let Some(report) = progress {
                    report(tiles_done as f32 / tiles_total as f32);
                }
                ox += tw;
            }
            oy += th;
        }
        let report = job.finish(1, ComputeDevice::Cpu, target_out);
        Ok(FullImage {
            width: fw,
            height: fh,
            rgb,
            target: request.target,
            window,
            report,
        })
    }
}

/// The only pixel of a 1 x 1 image as camera RGB.
fn pixel_rgb(img: &RgbImage) -> Vec3 {
    let p = img.pixel(0, 0);
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

/// `ColorChain` is referenced in doc links above.
#[allow(dead_code)]
fn _link(_: &ColorChain) {}
