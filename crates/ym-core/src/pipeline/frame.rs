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
//! # Output sharpening (the last stage)
//!
//! With `DevelopSettings::output_sharpening` above 0 (the default is 100) a view takes another
//! road between steps 4 and 5 (`docs/01` Render graph, `docs/algorithms/04-output-sharpening.md`):
//! the colour chain stops at C12 and produces display-linear f32 for the view **grown by the
//! stage's receptive field** on every side (`ViewSharpening::halo`, private); that frame is cached
//! (`DisplayFrame`, private, so moving the slider samples and colours nothing again); the stage then
//! sharpens the photo's own pixels in it at the scale `s = (the long edge shown) / (the long
//! edge rendered at full resolution)`, the border is cut off, and C13 (the transfer function and
//! the dither, at the pixel's position in the view) makes the 8-bit frame. The border makes a
//! view at 100 % or a pan of it equal the corresponding part of the whole picture; where the
//! photo itself ends (the letterbox) the stage mirrors at the photo's edge, as it does at the
//! edge of an export. Thumbnails and look tiles take the same road at their own `s`. The Before
//! view, the histogram and pixel readouts do not (they are not the finished picture). A view
//! magnified above 100 % runs the stage on the 100 % picture it magnifies (`s = 1`, the pixels an
//! export holds, over the part of the picture the view covers with a margin) and magnifies the
//! sharpened picture with the same Catmull-Rom filter, so the stage always sees the pixel grid it
//! was made for (`ViewSharpening::frame_geometry`, private).
//!
//! Because geometry is applied *before* the pointwise colour chain, a crop or a rotation never
//! re-runs a raw or colour group, and the colour chain only touches the pixels that are shown.

use std::sync::Arc;

use rayon::prelude::*;
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, Frame, Layout, OutputTarget, View};
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::resample::sample_catmull_rom;
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

use super::color::{GradePlan, PhotoSource};
use super::geometry::{Affine, Grid, oriented_to_sensor, sensor_to_oriented};
use super::graph::{ComputeDevice, Job, RenderReport, Renderer};
use super::viewport::ViewportState;
use super::{FrameZoom, RenderError};

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
pub(crate) fn color_pass(
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
            .map_err(|e| RenderError::Color(e.to_string()))?;
        let mut rgb = vec![0u8; 3 * n];
        let view = if before { View::Before } else { View::Final };
        grade
            .chain
            .render_u8(view, &frame, (0, 0), &mut rgb, job.r.options().threads)
            .map_err(|e| RenderError::Color(e.to_string()))?;
        rgba.par_chunks_mut(4)
            .zip(rgb.par_chunks(3))
            .for_each(|(d, s)| {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = 255;
            });
    }
    job.runs.color_passes += 1;
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
// Output sharpening
// ---------------------------------------------------------------------------------------------

/// The output sharpening in effect for one view: the stage, the scale of the view and the
/// receptive field the view is grown by (see the module documentation).
#[derive(Debug, Clone)]
pub(crate) struct ViewSharpening {
    /// The screen preset at the photo's amount.
    pub stage: OutputSharpen,
    /// `s = long edge shown / long edge rendered`, at most 1.
    pub scale: f32,
    /// The receptive field at that scale, in pixels of the view.
    pub halo: usize,
}

impl ViewSharpening {
    /// The stage for a view that maps device pixels to the final image with `view_to_final`
    /// (`fw x fh` pixels at full resolution), `None` when it creates no stage: the amount is 0 (or
    /// below 0.1 %), or the view is the Before view.
    pub(crate) fn of_view(
        amount_pct: f64,
        view_to_final: &Affine,
        (fw, fh): (usize, usize),
        before: bool,
    ) -> Result<Option<Self>, RenderError> {
        if before || amount_pct.is_nan() || amount_pct <= 0.0 || fw == 0 || fh == 0 {
            return Ok(None);
        }
        // The long edge of the photo as shown: its full-resolution long edge times the zoom (device
        // pixels per final pixel), measured on the long side (a thumbnail's two sides round
        // differently).
        let long = fw.max(fh);
        let step = if fw >= fh {
            view_to_final.m[0][0]
        } else {
            view_to_final.m[1][1]
        };
        let shown = if step.is_finite() && step > 0.0 {
            ((long as f64 / step).round().max(1.0)) as usize
        } else {
            long
        };
        let scale = scale_for_edges(shown, long);
        let stage = OutputSharpen::screen().with_amount_pct(amount_pct as f32);
        let plan = stage
            .plan(scale)
            .map_err(|e| RenderError::Image(format!("This photo can't be rendered: {e}.")))?;
        if plan.stages().is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            stage,
            scale,
            halo: plan.halo_px(),
        }))
    }

    /// The frame the stage reads for a view of `(w, h)` device pixels that maps to the final image
    /// (`fw x fh` pixels) with `view_to_final`; `None` when nothing of the photo is in view.
    ///
    /// * A view at 100 % or below is read **grown by the halo** on every side: the frame has
    ///   `w + 2 halo` by `h + 2 halo` pixels and the stage's receptive field is real picture.
    /// * A view **magnified above 100 %** shows the picture at 100 % made bigger, so the stage runs
    ///   on the 100 % picture (`s = 1`, the pixels an export holds) over the part of the final image
    ///   the view covers, with a margin of the halo and the 2 px of the magnifying filter, clipped
    ///   to the final image (where the photo ends the stage mirrors, as it does in an export); the
    ///   sharpened picture is magnified to the view afterwards ([`magnify_and_encode`]).
    fn frame_geometry(
        &self,
        view_to_final: &Affine,
        (w, h): (usize, usize),
        (fw, fh): (usize, usize),
    ) -> Option<FrameGeometry> {
        let step = view_to_final.m[0][0];
        if step.is_finite() && step < 1.0 - 1e-9 {
            let (ax, ay) = view_to_final.apply(0.0, 0.0);
            let (bx, by) = view_to_final.apply(w as f64, h as f64);
            let margin = (2 + self.halo) as f64;
            let x0 = (ax.min(bx) - margin).floor().max(0.0);
            let y0 = (ay.min(by) - margin).floor().max(0.0);
            let x1 = (ax.max(bx) + margin).ceil().min(fw as f64);
            let y1 = (ay.max(by) + margin).ceil().min(fh as f64);
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            let (x0, y0) = (x0 as usize, y0 as usize);
            return Some(FrameGeometry {
                view_to_final: Affine::scale_translate(1.0, 1.0, x0 as f64, y0 as f64),
                width: x1 as usize - x0,
                height: y1 as usize - y0,
                origin: Some((x0, y0)),
            });
        }
        let h_px = self.halo;
        Some(FrameGeometry {
            view_to_final: Affine::scale_translate(1.0, 1.0, -(h_px as f64), -(h_px as f64))
                .then(view_to_final),
            width: w + 2 * h_px,
            height: h + 2 * h_px,
            origin: None,
        })
    }
}

/// The frame a sharpened view reads ([`ViewSharpening::frame_geometry`]).
#[derive(Debug, Clone)]
struct FrameGeometry {
    /// Pixels of the frame to pixels of the final image at full resolution.
    view_to_final: Affine,
    /// Width of the frame.
    width: usize,
    /// Height of the frame.
    height: usize,
    /// For a magnified view: the pixel of the final image the frame starts at (its pixels are those
    /// of the final image). `None` for the view grown by the halo.
    origin: Option<(usize, usize)>,
}

/// A view before output sharpening: display-linear f32 (C1 to C12), interleaved, over the frame the
/// stage reads ([`ViewSharpening::frame_geometry`]: the view grown by the stage's receptive field,
/// or the 100 % picture under a magnified view). Cached per photo, so that a change of the amount
/// sharpens the same pixels again and nothing else runs.
#[derive(Debug)]
pub(crate) struct DisplayFrame {
    /// Width of the frame.
    pub width: usize,
    /// Height of the frame.
    pub height: usize,
    /// `3 * width * height` samples; pixels outside the photo are zero.
    pub rgb: Vec<f32>,
    /// 1 where the pixel is outside the photo; `None` when every pixel is inside.
    pub outside: Option<Vec<u8>>,
    /// The reduction of the raw group the pixels came from.
    pub scale: usize,
    /// False when the view was made from a coarser image than its zoom asks for.
    pub sharp: bool,
    /// Where the colour chain ran.
    pub device: ComputeDevice,
}

impl DisplayFrame {
    /// Bytes the frame holds.
    pub(crate) fn bytes(&self) -> usize {
        self.rgb.len() * 4 + self.outside.as_ref().map_or(0, Vec::len)
    }
}

/// What [`display_linear_pass`] made.
struct DisplayPass {
    /// Display-linear f32, interleaved, zero outside the photo.
    rgb: Vec<f32>,
    /// 1 where the pixel is outside the photo; `None` when every pixel is inside.
    outside: Option<Vec<u8>>,
    /// Where the colour chain ran.
    device: ComputeDevice,
}

/// Runs the colour chain to C12 over `sampled`: display-linear f32, interleaved, zero outside the
/// photo. The GPU does it when there is one (on any GPU error the CPU path runs and the report's
/// notes say why).
fn display_linear_pass(
    job: &mut Job<'_>,
    sampled: Sampled,
    grade: &GradePlan,
    priority: Priority,
) -> Result<DisplayPass, RenderError> {
    job.check()?;
    let started = std::time::Instant::now();
    let (w, h) = (sampled.img.width, sampled.img.height);
    let n = w * h;
    let mut rgb: Option<Vec<f32>> = None;
    let mut device = ComputeDevice::Cpu;
    if let Some(gpu) = job.r.options().gpu.clone() {
        let chain = Arc::clone(&grade.chain);
        let img = Arc::new(sampled.img.clone());
        let id = u64::from(job.src.id.0);
        let result = gpu.run(JobSpec::new(priority).owned_by(id), move |ctx, cancel| {
            cancel.check()?;
            let prepared = PreparedChain::new(ctx, &chain)?;
            prepared.render(ctx, &img, (0, 0), ColorOutput::Stage(Stop::C12))
        });
        match result {
            Ok(out) => {
                if let PixelData::F32Planar(planar) = out.data
                    && planar.len() == 3 * n
                {
                    let mut inter = vec![0.0f32; 3 * n];
                    let (r, rest) = planar.split_at(n);
                    let (g, b) = rest.split_at(n);
                    inter.par_chunks_mut(3).enumerate().for_each(|(i, d)| {
                        d[0] = r[i];
                        d[1] = g[i];
                        d[2] = b[i];
                    });
                    rgb = Some(inter);
                    device = ComputeDevice::Gpu;
                }
            }
            Err(e) => job.notes.push(format!(
                "The graphics card could not render this frame, so the CPU did: {e}."
            )),
        }
    }
    let mut rgb = match rgb {
        Some(v) => v,
        None => {
            let frame = Frame::planar(w, h, &sampled.img.data)
                .map_err(|e| RenderError::Color(e.to_string()))?;
            let mut out = vec![0.0f32; 3 * n];
            grade
                .chain
                .render_display_linear_f32(
                    &frame,
                    Layout::Interleaved,
                    &mut out,
                    job.r.options().threads,
                )
                .map_err(|e| RenderError::Color(e.to_string()))?;
            out
        }
    };
    if let Some(outside) = &sampled.outside {
        rgb.par_chunks_mut(3)
            .zip(outside.par_iter())
            .for_each(|(d, &o)| {
                if o != 0 {
                    d.fill(0.0);
                }
            });
    }
    job.runs.color_passes += 1;
    job.timings.pass_ms += started.elapsed().as_secs_f64() * 1000.0;
    Ok(DisplayPass {
        rgb,
        outside: sampled.outside,
        device,
    })
}

/// The smallest rectangle `(x0, y0, x1, y1)` (exclusive ends) that holds every pixel inside the
/// photo, or `None` when none is.
fn inside_bounds(
    outside: Option<&[u8]>,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    let Some(mask) = outside else {
        return (w > 0 && h > 0).then_some((0, 0, w, h));
    };
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (y, row) in mask.chunks_exact(w.max(1)).enumerate() {
        let Some(first) = row.iter().position(|&o| o == 0) else {
            continue;
        };
        // `rposition` exists because `first` does.
        let last = row.iter().rposition(|&o| o == 0).unwrap_or(first);
        x0 = x0.min(first);
        x1 = x1.max(last + 1);
        y0 = y0.min(y);
        y1 = y1.max(y + 1);
    }
    (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
}

/// The part of a display-linear frame the stage sharpened: the bounding box of the pixels inside the
/// photo (so the stage mirrors at the photo's edge and never reads the letterbox), at `(x0, y0)` of
/// the frame, `w x h`, interleaved.
struct Sharpened {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    rgb: Vec<f32>,
}

/// Runs the stage on the photo's pixels of `display`; `None` when no pixel is inside the photo.
fn sharpen_photo(
    display: &DisplayFrame,
    sharpening: &ViewSharpening,
) -> Result<Option<Sharpened>, RenderError> {
    let Some((x0, y0, x1, y1)) =
        inside_bounds(display.outside.as_deref(), display.width, display.height)
    else {
        return Ok(None);
    };
    let (w, h) = (x1 - x0, y1 - y0);
    let ew = display.width;
    let mut rgb = vec![0.0f32; 3 * w * h];
    rgb.par_chunks_mut(3 * w).enumerate().for_each(|(y, row)| {
        let at = ((y0 + y) * ew + x0) * 3;
        row.copy_from_slice(&display.rgb[at..at + 3 * w]);
    });
    sharpening
        .stage
        .apply_interleaved_in_place(w, h, &mut rgb, sharpening.scale)
        .map_err(|e| RenderError::Image(format!("This photo can't be rendered: {e}.")))?;
    Ok(Some(Sharpened { x0, y0, w, h, rgb }))
}

/// C13 and the letterbox: encodes a display-linear `w x h` view to RGBA8 (the dither at the pixel's
/// place in the view) and paints the paper where `letterbox` is 1.
fn encode_view(
    job: &mut Job<'_>,
    view: &[f32],
    (w, h): (usize, usize),
    letterbox: Option<&[u8]>,
    grade: &GradePlan,
    background: [u8; 3],
) -> Result<Vec<u8>, RenderError> {
    let frame = Frame::interleaved(w, h, view).map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgb8 = vec![0u8; 3 * w * h];
    grade
        .chain
        .encode_display_linear_u8(&frame, (0, 0), &mut rgb8, job.r.options().threads)
        .map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgba = vec![255u8; 4 * w * h];
    rgba.par_chunks_mut(4)
        .zip(rgb8.par_chunks(3))
        .for_each(|(d, s)| d[..3].copy_from_slice(s));
    if let Some(mask) = letterbox {
        rgba.par_chunks_mut(4)
            .zip(mask.par_iter())
            .for_each(|(d, &o)| {
                if o != 0 {
                    d[..3].copy_from_slice(&background);
                }
            });
    }
    Ok(rgba)
}

/// The output-sharpening stage and C13 over a display-linear frame of the grown view: sharpens
/// the photo's pixels, cuts the border off and encodes the view to RGBA8 with the letterbox
/// painted.
fn sharpen_and_encode(
    job: &mut Job<'_>,
    display: &DisplayFrame,
    sharpening: &ViewSharpening,
    grade: &GradePlan,
    (w, h): (usize, usize),
    background: [u8; 3],
) -> Result<Vec<u8>, RenderError> {
    job.check()?;
    let started = std::time::Instant::now();
    let halo = sharpening.halo;
    let ew = display.width;
    debug_assert_eq!(
        (display.width, display.height),
        (w + 2 * halo, h + 2 * halo)
    );
    // The view's own rows (without the border), unsharpened.
    let mut view = vec![0.0f32; 3 * w * h];
    view.par_chunks_mut(3 * w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            let at = ((y + halo) * ew + halo) * 3;
            row.copy_from_slice(&display.rgb[at..at + 3 * w]);
        });
    if let Some(s) = sharpen_photo(display, sharpening)? {
        // Put the sharpened pixels back where they fall inside the view.
        let (vx0, vx1) = (s.x0.max(halo), (s.x0 + s.w).min(halo + w));
        if vx0 < vx1 {
            view.par_chunks_mut(3 * w.max(1))
                .enumerate()
                .for_each(|(y, row)| {
                    let ey = y + halo;
                    if ey < s.y0 || ey >= s.y0 + s.h {
                        return;
                    }
                    let src = ((ey - s.y0) * s.w + (vx0 - s.x0)) * 3;
                    let dst = (vx0 - halo) * 3;
                    let len = (vx1 - vx0) * 3;
                    row[dst..dst + len].copy_from_slice(&s.rgb[src..src + len]);
                });
        }
    }
    let letterbox = display.outside.as_ref().map(|outside| {
        let mut mask = vec![0u8; w * h];
        mask.par_chunks_mut(w.max(1))
            .enumerate()
            .for_each(|(y, row)| row.copy_from_slice(&outside[(y + halo) * ew + halo..][..w]));
        mask
    });
    let rgba = encode_view(job, &view, (w, h), letterbox.as_deref(), grade, background)?;
    job.runs.output_passes += 1;
    job.timings.output_ms += started.elapsed().as_secs_f64() * 1000.0;
    Ok(rgba)
}

/// A view magnified above 100 %: the stage has run on the 100 % picture `display` (the final
/// image's pixels from `origin`, `s = 1`), and that sharpened picture is magnified to the view with
/// the Catmull-Rom filter every magnified view uses, then encoded. Device pixels that fall outside
/// the final image, or on pixels of it that are outside the straightened photo, are the paper.
fn magnify_and_encode(
    job: &mut Job<'_>,
    display: &DisplayFrame,
    origin: (usize, usize),
    sharpening: &ViewSharpening,
    grade: &GradePlan,
    spec: &ViewSpec,
    (fw, fh): (usize, usize),
) -> Result<Vec<u8>, RenderError> {
    job.check()?;
    let started = std::time::Instant::now();
    let (nw, nh) = (display.width, display.height);
    let n = nw * nh;
    let (w, h) = (spec.out_w, spec.out_h);
    // The sharpened 100 % picture, as three planes (what the filter reads).
    let mut native = display.rgb.clone();
    if let Some(s) = sharpen_photo(display, sharpening)? {
        for y in 0..s.h {
            let dst = ((s.y0 + y) * nw + s.x0) * 3;
            native[dst..dst + 3 * s.w].copy_from_slice(&s.rgb[y * s.w * 3..(y + 1) * s.w * 3]);
        }
    }
    let mut planes = vec![0.0f32; 3 * n];
    {
        let (r, rest) = planes.split_at_mut(n);
        let (g, b) = rest.split_at_mut(n);
        r.par_iter_mut()
            .zip(g.par_iter_mut())
            .zip(b.par_iter_mut())
            .zip(native.par_chunks(3))
            .for_each(|(((r, g), b), p)| {
                *r = p[0];
                *g = p[1];
                *b = p[2];
            });
    }
    drop(native);
    let mut view = vec![0.0f32; 3 * w * h];
    let mut letterbox = vec![0u8; w * h];
    let (ox, oy) = (origin.0 as f64, origin.1 as f64);
    view.par_chunks_mut(3 * w.max(1))
        .zip(letterbox.par_chunks_mut(w.max(1)))
        .enumerate()
        .for_each(|(y, (row, mask))| {
            for x in 0..w {
                let (fx, fy) = spec.view_to_final.apply(x as f64 + 0.5, y as f64 + 0.5);
                let inside = fx >= 0.0 && fy >= 0.0 && fx < fw as f64 && fy < fh as f64;
                let (px, py) = (fx - ox, fy - oy);
                let empty = !inside
                    || display.outside.as_ref().is_some_and(|o| {
                        let (ix, iy) = (
                            (px.max(0.0) as usize).min(nw - 1),
                            (py.max(0.0) as usize).min(nh - 1),
                        );
                        o[iy * nw + ix] != 0
                    });
                if empty {
                    mask[x] = 1;
                    continue;
                }
                for c in 0..3 {
                    row[3 * x + c] =
                        sample_catmull_rom(&planes[c * n..(c + 1) * n], nw, nh, px, py);
                }
            }
        });
    let rgba = encode_view(job, &view, (w, h), Some(&letterbox), grade, spec.background)?;
    job.runs.output_passes += 1;
    job.timings.output_ms += started.elapsed().as_secs_f64() * 1000.0;
    Ok(rgba)
}

/// The road of a view with output sharpening (see the module documentation): the display-linear
/// frame the stage reads (from the display cache when `spec.cache`, else made), then the stage and
/// C13. Returns the RGBA8 frame, where the colour chain ran, the raw scale and whether the frame is
/// sharp.
fn sharpened_pass(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: &ViewSharpening,
) -> Result<(Vec<u8>, ComputeDevice, usize, bool), RenderError> {
    let image = job.final_size()?;
    let Some(geometry) =
        sharpening.frame_geometry(&spec.view_to_final, (spec.out_w, spec.out_h), image)
    else {
        // Nothing of the photo is in view: all paper.
        let mut rgba = vec![255u8; 4 * spec.out_w * spec.out_h];
        rgba.par_chunks_mut(4)
            .for_each(|d| d[..3].copy_from_slice(&spec.background));
        return Ok((rgba, ComputeDevice::Cpu, 1, true));
    };
    let read = ViewSpec {
        out_w: geometry.width,
        out_h: geometry.height,
        view_to_final: geometry.view_to_final,
        ..spec.clone()
    };
    let key = spec
        .cache
        .then(|| display_key(job, &read, grade, sharpening))
        .transpose()?;
    let cached = key
        .as_ref()
        .and_then(|k| job.g.cached_display(k))
        .filter(|d| {
            // A frame made from a coarser image than asked for is not served to a request that wants
            // the tiles (same rule as the finished frames).
            !(spec.coarse_only && !d.sharp)
        });
    let display = match cached {
        Some(d) => {
            job.runs.display_hits += 1;
            d
        }
        None => {
            let SampledView {
                sampled,
                scale,
                sharp,
            } = sample_view(job, &read)?;
            let DisplayPass {
                rgb,
                outside,
                device,
            } = display_linear_pass(job, sampled, grade, spec.gpu_priority)?;
            let d = Arc::new(DisplayFrame {
                width: read.out_w,
                height: read.out_h,
                rgb,
                outside,
                scale,
                sharp,
                device,
            });
            if let Some(k) = key {
                job.g.store_display(k, Arc::clone(&d));
            }
            d
        }
    };
    let rgba = match geometry.origin {
        None => sharpen_and_encode(
            job,
            &display,
            sharpening,
            grade,
            (spec.out_w, spec.out_h),
            spec.background,
        )?,
        Some(origin) => magnify_and_encode(job, &display, origin, sharpening, grade, spec, image)?,
    };
    Ok((rgba, display.device, display.scale, display.sharp))
}

fn display_key(
    job: &mut Job<'_>,
    grown: &ViewSpec,
    grade: &GradePlan,
    sharpening: &ViewSharpening,
) -> Result<String, RenderError> {
    let params = job.params()?;
    // The sharpening amount is not part of it: the frame is what the amount is applied to. The
    // halo is (it is what the view was grown by).
    Ok(format!(
        "display|{:?}|{:?}|{:?}|{:?}|{}x{}|{:?}|{}|{}",
        params.key(),
        grade.key,
        job.geometry(),
        grown.view_to_final,
        grown.out_w,
        grown.out_h,
        grown.k,
        grown.coarse_only,
        sharpening.halo
    ))
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
    let sharpening = view_sharpening(&mut job, spec)?;
    let cache_key = spec
        .cache
        .then(|| frame_key(&mut job, spec, &grade, sharpening.as_ref()))
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
    let (rgba, device, scale, sharp) = if let Some(sh) = &sharpening {
        sharpened_pass(&mut job, spec, &grade, sh)?
    } else {
        let view = sample_view(&mut job, spec)?;
        let (rgba, device) = color_pass(
            &mut job,
            view.sampled,
            &grade,
            spec.before,
            spec.background,
            spec.gpu_priority,
        )?;
        (rgba, device, view.scale, view.sharp)
    };
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

/// The output sharpening of a view, if it has any (see [`ViewSharpening::of_view`]).
fn view_sharpening(
    job: &mut Job<'_>,
    spec: &ViewSpec,
) -> Result<Option<ViewSharpening>, RenderError> {
    let amount = job.src.settings.output_sharpening;
    if spec.before || amount.is_nan() || amount <= 0.0 {
        return Ok(None);
    }
    let image = job.final_size()?;
    ViewSharpening::of_view(amount, &spec.view_to_final, image, spec.before)
}

fn frame_key(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: Option<&ViewSharpening>,
) -> Result<String, RenderError> {
    let params = job.params()?;
    let mut key = format!(
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
    );
    // The amount belongs to the finished frame, not to the display-linear one under it.
    if sharpening.is_some() {
        key.push_str(&format!("|os{:?}", job.src.settings.output_sharpening));
    }
    Ok(key)
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
        Job::new(self, src, cancel)?.final_size()
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
        let mut job = Job::new(self, src, cancel)?;
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
        let mut frame = render_view(job, &spec)?;
        // The zoom belongs to the request, not to the pixels (a cached frame is served for the
        // same view): say what this request asked for.
        frame.report.zoom = Some(FrameZoom {
            zoom,
            fit_zoom: state.fit_zoom(image),
        });
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
        let mut job = Job::new(self, src, cancel)?;
        let plan = job.plan()?;
        let image = job.final_size()?;
        if plan.reduced || scale_for_zoom(state.zoom(image)) != 1 {
            return Ok(0);
        }
        // The tiles the frame will read. With output sharpening the frame is the view grown by the
        // stage's receptive field, or, magnified above 100 %, the part of the 100 % picture it
        // covers with its margin ([`ViewSharpening::frame_geometry`]): the tiles under those are
        // needed too, or the frame would stay coarse for want of tiles that nothing computes.
        let to_final = state.view_to_final(image);
        let (dw, dh) = state.device_size();
        let (to_final, dw, dh) = match ViewSharpening::of_view(
            src.settings.output_sharpening,
            &to_final,
            image,
            state.before,
        )? {
            None => (to_final, dw, dh),
            Some(sh) => match sh.frame_geometry(&to_final, (dw, dh), image) {
                Some(g) => (g.view_to_final, g.width, g.height),
                None => return Ok(0),
            },
        };
        let grid = job
            .geometry()
            .at_full_resolution()
            .grid_from((plan.out_w, plan.out_h), to_final);
        let Some((x0, y0, x1, y1)) = grid.source_bounds(0, 0, dw, dh, 3) else {
            return Ok(0);
        };
        let _ = job.region(Rect::new(x0, y0, x1 - x0, y1 - y0), true)?;
        let made = job.runs.tiles;
        job.finish(1, ComputeDevice::Cpu, OutputTarget::Viewport);
        Ok(made as usize)
    }

    /// Display values (0 to 255) under a point of the cell, as the viewport shows them (the Before
    /// view included), without the dither and **before output sharpening** (a readout is of the
    /// colour at a point; the stage needs the neighbourhood of a whole view). `None` outside the
    /// photo (the letterbox).
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
        let mut job = Job::new(self, src, cancel)?;
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
        let mut job = Job::new(self, src, cancel)?;
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
        let mut job = Job::for_export(self, src, cancel);
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
            return Err(RenderError::Color(
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
                        .map_err(|e| RenderError::Color(e.to_string()))?;
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
                            .map_err(|e| RenderError::Color(e.to_string()))?,
                        FullTarget::DngLinearAwg4 => grade
                            .chain
                            .render_dng_linear_awg4_f32(
                                &frame,
                                Layout::Interleaved,
                                &mut out,
                                threads,
                            )
                            .map_err(|e| RenderError::Color(e.to_string()))?,
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

#[cfg(test)]
mod sharpening_tests {
    use super::*;

    fn to_final(step: f64, tx: f64, ty: f64) -> Affine {
        Affine::scale_translate(step, step, tx, ty)
    }

    fn of(amount: f64, step: f64, image: (usize, usize), before: bool) -> Option<ViewSharpening> {
        ViewSharpening::of_view(amount, &to_final(step, 0.0, 0.0), image, before).expect("a plan")
    }

    #[test]
    fn the_scale_is_the_long_edge_shown_over_the_long_edge_rendered() {
        // 6000 x 4000 shown 3 final pixels to a device pixel: 2000 px on the long edge.
        let sh = of(100.0, 3.0, (6000, 4000), false).expect("a stage");
        assert_eq!(sh.scale, scale_for_edges(2000, 6000));
        // The same for a portrait photo (the long side is the vertical one).
        let sh = of(100.0, 3.0, (4000, 6000), false).expect("a stage");
        assert_eq!(sh.scale, scale_for_edges(2000, 6000));
        // 100 % and every magnification sharpen the picture an export holds: s = 1.
        for step in [1.0, 0.999, 0.5, 0.125] {
            let sh = of(100.0, step, (6000, 4000), false).expect("a stage");
            assert_eq!(sh.scale, 1.0, "step {step}");
        }
        // At s = 1 the border is the stage's own receptive field, never more than its bound.
        let sh = of(100.0, 1.0, (6000, 4000), false).expect("a stage");
        let plan = OutputSharpen::screen().plan(1.0).expect("plan");
        assert_eq!(sh.halo, plan.halo_px());
        assert!(sh.halo <= OutputSharpen::screen().max_halo_px());
        // A smaller picture needs a smaller border.
        let small = of(100.0, 6.0, (6000, 4000), false).expect("a stage");
        assert!(small.halo < sh.halo, "{} against {}", small.halo, sh.halo);
    }

    #[test]
    fn no_amount_the_before_view_and_a_collapsed_stage_make_no_stage() {
        assert!(of(0.0, 1.0, (600, 400), false).is_none());
        // Below a tenth of a percent the stage has no amount (spec 04: A < 0.001).
        assert!(of(0.05, 1.0, (600, 400), false).is_none());
        assert!(of(f64::NAN, 1.0, (600, 400), false).is_none());
        assert!(of(-3.0, 1.0, (600, 400), false).is_none());
        assert!(of(100.0, 1.0, (600, 400), true).is_none());
        assert!(of(100.0, 1.0, (0, 0), false).is_none());
        // A step that is not a number falls back to the full size, not to a panic.
        assert_eq!(
            of(100.0, f64::NAN, (600, 400), false)
                .expect("a stage")
                .scale,
            1.0
        );
        // The smallest thumbnails still plan: a 1 px long edge of a 6000 px photo.
        assert!(of(200.0, 6000.0, (6000, 4000), false).is_some());
    }

    #[test]
    fn a_view_at_100_percent_or_below_is_grown_by_the_halo_on_every_side() {
        let image = (6000, 4000);
        let map = to_final(2.0, 300.0, 200.0);
        let sh = ViewSharpening::of_view(100.0, &map, image, false)
            .expect("plan")
            .expect("a stage");
        let g = sh.frame_geometry(&map, (200, 100), image).expect("a frame");
        let halo = sh.halo;
        assert_eq!((g.width, g.height), (200 + 2 * halo, 100 + 2 * halo));
        assert!(g.origin.is_none());
        // The frame's pixel (halo, halo) is the view's pixel (0, 0): the same place in the picture.
        let h = halo as f64;
        assert_eq!(g.view_to_final.apply(h, h), map.apply(0.0, 0.0));
        assert_eq!(
            g.view_to_final.apply(h + 200.0, h + 100.0),
            map.apply(200.0, 100.0)
        );
        // Exactly 100 % is this road too.
        let map = to_final(1.0, 10.0, 20.0);
        let sh = ViewSharpening::of_view(100.0, &map, image, false)
            .expect("plan")
            .expect("a stage");
        let g = sh.frame_geometry(&map, (64, 48), image).expect("a frame");
        assert!(g.origin.is_none());
        assert_eq!((g.width, g.height), (64 + 2 * sh.halo, 48 + 2 * sh.halo));
    }

    #[test]
    fn a_magnified_view_reads_the_100_percent_picture_it_covers_with_a_margin() {
        let image = (800, 600);
        // 200 %: a 200 x 100 view covers 100 x 50 pixels of the picture from (300, 200).
        let map = to_final(0.5, 300.0, 200.0);
        let sh = ViewSharpening::of_view(100.0, &map, image, false)
            .expect("plan")
            .expect("a stage");
        let g = sh.frame_geometry(&map, (200, 100), image).expect("a frame");
        let (x0, y0) = g.origin.expect("the 100 % picture");
        let margin = 2 + sh.halo;
        // The frame is the picture's own pixels from its origin.
        assert_eq!(
            g.view_to_final,
            Affine::scale_translate(1.0, 1.0, x0 as f64, y0 as f64)
        );
        assert!(x0 + margin <= 300 && y0 + margin <= 200, "({x0}, {y0})");
        assert!(x0 + g.width >= 400 + margin && y0 + g.height >= 250 + margin);
        // Inside the picture.
        assert!(x0 + g.width <= image.0 && y0 + g.height <= image.1);
        // Over the edge: clipped to the picture, where the stage mirrors as it does in an export.
        let map = to_final(0.5, -50.0, -25.0);
        let g = sh.frame_geometry(&map, (200, 100), image).expect("a frame");
        assert_eq!(g.origin, Some((0, 0)));
        assert_eq!((g.width, g.height), (50 + margin, 25 + margin));
        let map = to_final(0.5, 700.0, 560.0);
        let g = sh.frame_geometry(&map, (200, 100), image).expect("a frame");
        let (x0, y0) = g.origin.expect("origin");
        assert_eq!((x0 + g.width, y0 + g.height), image);
        // Nothing of the photo in view: no frame at all.
        assert!(
            sh.frame_geometry(&to_final(0.5, 5000.0, 5000.0), (200, 100), image)
                .is_none()
        );
        assert!(
            sh.frame_geometry(&to_final(0.5, -5000.0, 0.0), (200, 100), image)
                .is_none()
        );
    }

    #[test]
    fn the_bounding_box_of_the_photo_ignores_the_paper() {
        // No mask: the whole frame.
        assert_eq!(inside_bounds(None, 5, 4), Some((0, 0, 5, 4)));
        assert_eq!(inside_bounds(None, 0, 4), None);
        // Paper (1) all round a 2 x 2 photo.
        #[rustfmt::skip]
        let mask = [
            1, 1, 1, 1, 1,
            1, 0, 0, 1, 1,
            1, 0, 0, 1, 1,
            1, 1, 1, 1, 1,
        ];
        assert_eq!(inside_bounds(Some(&mask), 5, 4), Some((1, 1, 3, 3)));
        // A tilted photo: the box holds its corners (the empty corners of the box are in it).
        #[rustfmt::skip]
        let tilted = [
            1, 1, 0, 1,
            1, 0, 0, 0,
            0, 0, 0, 1,
            1, 0, 1, 1,
        ];
        assert_eq!(inside_bounds(Some(&tilted), 4, 4), Some((0, 0, 4, 4)));
        // Nothing of the photo in view.
        assert_eq!(inside_bounds(Some(&[1; 12]), 4, 3), None);
    }
}
