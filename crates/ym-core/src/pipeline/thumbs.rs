//! Thumbnails, look tiles, the histogram and the scheduled entry points.
//!
//! * **Thumbnails** (`thumbnail_ready`): the raw group at 1/8 (or the largest reduction that leaves
//!   at least twice the thumbnail size), geometry applied, the colour pass over at most 240 px,
//!   sRGB, PNG. They are rendered once after open (the session's [`ThumbnailRenderer`] hook,
//!   [`PhotoThumbnails`]) and again on look changes.
//! * **Look tiles** (`look_thumbnail`, an addition to `docs/01`): the same render with one look at
//!   100 %, for the tiles of the Look tab. Only the grade group differs between looks, so the raw
//!   group and the colour plan are shared by every tile.
//! * **Histogram** (`histogram`, an addition to `docs/01`): 64 bins of the LogC4 signal at the
//!   input of the curve (C10), per channel, scaled so that the tallest bin is 1 (`docs/02` C11:
//!   "The histogram in the Curve tab is of LogC4 values").
//! * **Chroma scatter** (`chroma_scatter`, for the plane of the colour warp, `docs/11`): the photo's
//!   colours at the input of the warp (the C9 output), projected onto its chroma plane
//!   (`ym_color::warp::chroma_of`), from the same small view as the histogram, at most
//!   [`CHROMA_SCATTER_MAX_POINTS`] of them.
//! * **Scheduled calls**: `spawn_*` put a request on the render queue at the priority `docs/01`
//!   gives it and return a handle.
//! * **Apple Auto's picture** (`docs/19` section 3): the photo at its default settings through the
//!   Signature look at 100 % and ACES 2, at 768 px on the long edge, 8-bit sRGB, on the CPU reference
//!   path ([`AdvisorPictures`], the session's [`PictureRenderer`] hook).

use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ym_color::display::{ColorChain, OutputTarget};
use ym_color::warp::chroma_of;
use ym_export::DetailStat;
use ym_gpu::queue::{CancelToken, JobError, JobHandle, Priority};
use ym_looks::library::LookId;

use super::RenderError;
use super::color::{
    LookProblem, LookProvider, PhotoSource, ResolvedLook, raw_fingerprint, with_look,
};
use super::frame::{
    FrameOptions, FrameOutput, FullImage, FullRequest, RenderedFrame, Sampled, ViewSpec,
    render_view, sample_view_routed,
};
use super::geometry::Affine;
use super::graph::{Job, RenderOptions, Renderer};
use super::viewport::ViewportState;
use crate::apple_auto::{AdvisorPicture, PICTURE_LONG_EDGE_PX};
use crate::session::{PhotoId, PictureInput, PictureRenderer, ThumbnailInput, ThumbnailRenderer};
use crate::settings::DevelopSettings;
use ym_color::display::DisplayTransform;
use ym_color::look::LookTransform;

/// Bins of the histogram.
pub const HISTOGRAM_BINS: usize = 64;
/// Longest edge of the sampled image behind the histogram, in pixels.
const HISTOGRAM_LONG_EDGE_PX: usize = 192;

/// The `histogram` response of `src/api/types.ts`: per channel, [`HISTOGRAM_BINS`] values from black
/// to white, scaled so that the tallest bin of any channel is 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramData {
    /// The red channel.
    pub red: Vec<f64>,
    /// The green channel.
    pub green: Vec<f64>,
    /// The blue channel.
    pub blue: Vec<f64>,
}

/// The most points a [`ChromaScatter`] holds.
pub const CHROMA_SCATTER_MAX_POINTS: usize = 2048;
/// The scatter's coordinates are rounded to this step (the plane is drawn 352 px wide for 0.6 units,
/// so this is far below a pixel and keeps the answer small).
pub const CHROMA_SCATTER_STEP: f64 = 1e-4;

/// The `chroma_scatter` response of `src/api/types.ts`: the photo's colours at the input of the colour
/// warp (C9b), as points `(a, b)` of its chroma plane, flat: `[a0, b0, a1, b1, ...]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChromaScatter {
    /// At most [`CHROMA_SCATTER_MAX_POINTS`] points, each coordinate rounded to
    /// [`CHROMA_SCATTER_STEP`].
    pub points: Vec<f64>,
}

/// The pixels of a sampled view that are inside the photo, every `stride`-th of them so that at most
/// `max` are left: a fixed stride over the pixels in reading order, so the same view gives the same
/// points.
fn every_nth_inside(sampled: &Sampled, max: usize) -> Vec<usize> {
    let n = sampled.img.width * sampled.img.height;
    let outside = sampled.outside.as_deref();
    let inside: Vec<usize> = (0..n)
        .filter(|&i| !outside.is_some_and(|o| o[i] != 0))
        .collect();
    let stride = inside.len().div_ceil(max.max(1)).max(1);
    inside.into_iter().step_by(stride).collect()
}

/// A coordinate of the scatter: rounded to [`CHROMA_SCATTER_STEP`], never `-0`. Dividing the count
/// of steps by their number per unit gives the double nearest to the decimal, so the JSON says
/// `0.1234` and not `0.12340000000000001`.
fn scatter_coordinate(v: f64) -> f64 {
    const STEPS_PER_UNIT: f64 = 1e4;
    (v * STEPS_PER_UNIT).round() / STEPS_PER_UNIT + 0.0
}

/// Encodes 8-bit RGB as a PNG.
///
/// # Errors
///
/// [`RenderError::Image`] when the encoder refuses the size.
pub fn encode_png_rgb8(width: usize, height: usize, rgb: &[u8]) -> Result<Vec<u8>, RenderError> {
    let fail = |e: png::EncodingError| {
        RenderError::Image(format!("The thumbnail could not be written: {e}."))
    };
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width as u32, height as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(fail)?;
        w.write_image_data(rgb).map_err(fail)?;
    }
    Ok(out)
}

/// The size of a thumbnail of an image: the long edge at most `max_px`, never upscaled.
#[must_use]
pub fn thumbnail_size(image: (usize, usize), max_px: usize) -> (usize, usize) {
    let long = image.0.max(image.1).max(1);
    let s = (max_px as f64 / long as f64).min(1.0);
    (
        ((image.0 as f64 * s).round() as usize).max(1),
        ((image.1 as f64 * s).round() as usize).max(1),
    )
}

/// The reduction a thumbnail of `max_px` is made from: the largest of 8, 4 and 2 that leaves the
/// long edge at least twice the thumbnail, else full resolution.
fn thumbnail_scale(out: (usize, usize), max_px: usize) -> usize {
    let long = out.0.max(out.1);
    for k in [8usize, 4, 2] {
        if long / k >= 2 * max_px {
            return k;
        }
    }
    1
}

impl Renderer {
    fn thumbnail_spec(&self, job: &mut Job<'_>) -> Result<(ViewSpec, (usize, usize)), RenderError> {
        let image = job.final_size()?;
        let max_px = self.options().thumbnail_px;
        let (tw, th) = thumbnail_size(image, max_px);
        // The reduction follows the size of what is shown (a crop to a tenth of the photo needs a
        // finer image than the whole photo does).
        let k = thumbnail_scale(image, max_px);
        let spec = ViewSpec {
            out_w: tw,
            out_h: th,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / tw as f64,
                image.1 as f64 / th as f64,
                0.0,
                0.0,
            ),
            k,
            target: OutputTarget::Srgb,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            output: FrameOutput::Host,
        };
        Ok((spec, (tw, th)))
    }

    /// The thumbnail's pixels: sRGB, 8 bits, `width * height * 3` interleaved, with the crop,
    /// straighten and rotation of the settings applied.
    fn thumbnail_rgb(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<(usize, usize, Vec<u8>), RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let (spec, (tw, th)) = self.thumbnail_spec(&mut job)?;
        let frame = render_view(job, &spec)?;
        let mut rgb = vec![0u8; 3 * tw * th];
        rgb.par_chunks_mut(3)
            .zip(frame.rgba.par_chunks(4))
            .for_each(|(d, s)| {
                d.copy_from_slice(&s[..3]);
            });
        Ok((tw, th, rgb))
    }

    /// The photo as a thumbnail: a PNG of at most 240 px on its long edge (`thumbnail_ready`), in
    /// sRGB, with the crop, straighten and rotation of the settings applied.
    ///
    /// The picture's detail statistic is kept for the size estimate of the Export panel
    /// ([`Renderer::thumbnail_detail`]); it always describes the newest thumbnail rendered here,
    /// so it follows the look, crop and turn the filmstrip follows.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn thumbnail_png(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RenderError> {
        let (tw, th, rgb) = self.thumbnail_rgb(src, cancel)?;
        self.note_thumbnail_detail(src.id, DetailStat::measure_rgb8(tw as u32, th as u32, &rgb));
        encode_png_rgb8(tw, th, &rgb)
    }

    /// Apple Auto's picture of a photo (`docs/19` section 3): `src` as it is (its settings, look and
    /// estimates), whole, at [`PICTURE_LONG_EDGE_PX`] on the long edge (a smaller photo is not
    /// enlarged), from the reduced raw group as a thumbnail is, in sRGB, 8 bits, `width * height * 3`
    /// bytes, rows top to bottom. It runs on the calling thread on whatever device this renderer has;
    /// [`AdvisorPictures`] keeps one without a device, so the picture is the CPU reference path's.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn advisor_picture(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<AdvisorPicture, RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let image = job.final_size()?;
        let (width, height) = thumbnail_size(image, PICTURE_LONG_EDGE_PX);
        let spec = ViewSpec {
            out_w: width,
            out_h: height,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / width as f64,
                image.1 as f64 / height as f64,
                0.0,
                0.0,
            ),
            k: thumbnail_scale(image, PICTURE_LONG_EDGE_PX),
            target: OutputTarget::Srgb,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            // The model gets bytes in memory; this renderer has no device anyway.
            output: FrameOutput::Host,
        };
        let frame = render_view(job, &spec)?;
        let mut rgb = vec![0u8; 3 * width * height];
        rgb.par_chunks_mut(3)
            .zip(frame.rgba.par_chunks(4))
            .for_each(|(d, s)| {
                d.copy_from_slice(&s[..3]);
            });
        Ok(AdvisorPicture { width, height, rgb })
    }

    /// The photo as it is developed now, through one look at 100 % (the Look tab's tiles). It does
    /// not touch the photo's detail statistic: a tile is the photo through another look.
    ///
    /// # Errors
    ///
    /// [`RenderError::Look`] with "File missing or changed. Import again." when the look cannot be
    /// loaded, and the errors of [`Renderer::thumbnail_png`].
    pub fn look_thumbnail_png(
        &self,
        src: &PhotoSource,
        look: &LookId,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RenderError> {
        self.looks()
            .look(look)
            .map_err(|e| RenderError::Look(e.to_string()))?;
        let tile = src.with_settings(with_look(&src.settings, look));
        let (tw, th, rgb) = self.thumbnail_rgb(&tile, cancel)?;
        encode_png_rgb8(tw, th, &rgb)
    }

    /// The small view of the whole photo that the histogram and the chroma scatter read, and the
    /// grade chain of the photo as it is rendered now: `read` gets the sampled camera RGB (with the
    /// letterbox mask) and the chain.
    fn panel_view<T>(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
        read: impl FnOnce(&Sampled, &ColorChain) -> T,
    ) -> Result<T, RenderError> {
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::Thumbnail);
        let image = job.final_size()?;
        let (tw, th) = thumbnail_size(image, HISTOGRAM_LONG_EDGE_PX);
        let spec = ViewSpec {
            out_w: tw,
            out_h: th,
            view_to_final: Affine::scale_translate(
                image.0 as f64 / tw as f64,
                image.1 as f64 / th as f64,
                0.0,
                0.0,
            ),
            k: thumbnail_scale(image, HISTOGRAM_LONG_EDGE_PX),
            target: OutputTarget::Viewport,
            before: false,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Thumbnail,
            dither_origin: (0, 0),
            output: FrameOutput::Host,
        };
        let grade = job.grade(OutputTarget::Viewport, false)?;
        // The sample comes from the device group when the sampler is there; the reading stays on
        // the CPU (a few thousand pixels through the f64 trace: `docs/17-gpu.md` section 11).
        let view = sample_view_routed(&mut job, &spec)?;
        let out = read(&view.sampled, &grade.chain);
        job.finish(
            view.scale,
            super::graph::ComputeDevice::Cpu,
            OutputTarget::Viewport,
        );
        Ok(out)
    }

    /// The histogram of the photo as rendered now (see the module documentation).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn histogram(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<HistogramData, RenderError> {
        let counts = self.panel_view(src, cancel, |sampled, chain| {
            let img = &sampled.img;
            let n = img.width * img.height;
            let outside = sampled.outside.as_deref();
            (0..n)
                .into_par_iter()
                .fold(
                    || [[0u64; HISTOGRAM_BINS]; 3],
                    |mut acc, i| {
                        if outside.is_some_and(|o| o[i] != 0) {
                            return acc;
                        }
                        let p = img.pixel(i % img.width, i / img.width);
                        let t = chain.trace([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                        for (c, acc_c) in acc.iter_mut().enumerate() {
                            let v = t.c10_logc4.0[c].clamp(0.0, 1.0);
                            let bin =
                                ((v * HISTOGRAM_BINS as f64) as usize).min(HISTOGRAM_BINS - 1);
                            acc_c[bin] += 1;
                        }
                        acc
                    },
                )
                .reduce(
                    || [[0u64; HISTOGRAM_BINS]; 3],
                    |mut a, b| {
                        for c in 0..3 {
                            for i in 0..HISTOGRAM_BINS {
                                a[c][i] += b[c][i];
                            }
                        }
                        a
                    },
                )
        })?;
        let peak = counts.iter().flatten().copied().max().unwrap_or(0).max(1) as f64;
        let norm = |c: usize| -> Vec<f64> {
            counts[c]
                .iter()
                .map(|&v| ((v as f64 / peak) * 1000.0).round() / 1000.0)
                .collect()
        };
        Ok(HistogramData {
            red: norm(0),
            green: norm(1),
            blue: norm(2),
        })
    }

    /// The photo's colours at the input of the colour warp (the C9 output, `StageTrace::c9_logc4`),
    /// projected onto the warp's chroma plane with `ym_color::warp::chroma_of`: the faint scatter
    /// behind the pins (`docs/11` Colour warp). It reads the histogram's view (the whole photo at
    /// most 192 px on the long edge, the letterbox left out) and keeps every `n`-th pixel in reading
    /// order so that at most [`CHROMA_SCATTER_MAX_POINTS`] are left, so the same photo and settings
    /// give the same points; each coordinate is rounded to [`CHROMA_SCATTER_STEP`]. The warp itself
    /// does not move them (they are what the pins take hold of); everything before it does.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn chroma_scatter(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<ChromaScatter, RenderError> {
        let points = self.panel_view(src, cancel, |sampled, chain| {
            let img = &sampled.img;
            every_nth_inside(sampled, CHROMA_SCATTER_MAX_POINTS)
                .par_iter()
                .flat_map_iter(|&i| {
                    let p = img.pixel(i % img.width, i / img.width);
                    let t = chain.trace([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                    chroma_of(t.c9_logc4).map(scatter_coordinate)
                })
                .collect::<Vec<f64>>()
        })?;
        Ok(ChromaScatter { points })
    }

    // --- Scheduled calls -------------------------------------------------------------------

    /// Records that the user moved to `photo`: the viewport and tile jobs of the photo they left
    /// are cancelled; thumbnails and exports are not.
    pub fn navigate(&self, photo: PhotoId) -> usize {
        let left = self.note_current(photo);
        if let Some(old) = left {
            // Full-resolution intermediates (tiles, the prepared chain with its sigma map) live
            // only while the photo is being looked at; the reduced images stay for the filmstrip.
            if let Some(g) = self.existing_graph(old) {
                g.release_full_resolution();
            }
            // The device jobs of the viewport and tile renders of the photo left are cancelled
            // too, queued ones at once (they hold the photo's mosaic).
            self.cancel_gpu_navigation(old);
        }
        self.scheduler().navigate(photo)
    }

    /// Queues a viewport frame at [`Priority::Viewport`].
    #[must_use]
    pub fn spawn_viewport_frame(
        &self,
        src: PhotoSource,
        state: ViewportState,
        options: FrameOptions,
    ) -> JobHandle<Result<RenderedFrame, RenderError>> {
        self.spawn_viewport_frame_to(src, state, options, FrameOutput::Host)
    }

    /// [`Renderer::spawn_viewport_frame`] with the place the frame's bytes go
    /// ([`Renderer::viewport_frame_to`]): the native viewport asks for [`FrameOutput::Device`].
    #[must_use]
    pub fn spawn_viewport_frame_to(
        &self,
        src: PhotoSource,
        state: ViewportState,
        options: FrameOptions,
        output: FrameOutput,
    ) -> JobHandle<Result<RenderedFrame, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Viewport, Some(id), move |token| {
                r.viewport_frame_to(&src, &state, &options, output, token)
            })
    }

    /// Queues the full-resolution tiles a viewport shows at [`Priority::TileNeural`].
    #[must_use]
    pub fn spawn_tiles(
        &self,
        src: PhotoSource,
        state: ViewportState,
    ) -> JobHandle<Result<usize, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::TileNeural, Some(id), move |token| {
                r.prepare_tiles(&src, &state, token)
            })
    }

    /// Queues a thumbnail at [`Priority::Thumbnail`].
    #[must_use]
    pub fn spawn_thumbnail(&self, src: PhotoSource) -> JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Thumbnail, Some(id), move |token| {
                r.thumbnail_png(&src, token)
            })
    }

    /// Queues a look tile at [`Priority::Thumbnail`]. A tile is a request of the Look tab for the
    /// photo on screen: moving to another photo ([`Renderer::navigate`]) cancels it, pending or
    /// running, and a pending one lets go of the photo's decoded image at once; the handle then
    /// resolves with [`JobError::Cancelled`] (a running tile ends with
    /// [`RenderError::Cancelled`]; see
    /// [`JobScheduler::submit_superseded`](crate::scheduler::JobScheduler::submit_superseded)).
    #[must_use]
    pub fn spawn_look_thumbnail(
        &self,
        src: PhotoSource,
        look: LookId,
    ) -> JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.look_thumbnail_png(&src, &look, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues the histogram of a photo at [`Priority::Thumbnail`]. Like a look tile it is a request
    /// of a panel for the photo on screen, and moving to another photo cancels it (see
    /// [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_histogram(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<HistogramData, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.histogram(&src, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues the chroma scatter of a photo at [`Priority::Thumbnail`]. Like the histogram it is a
    /// request of a panel for the photo on screen, and moving to another photo cancels it (see
    /// [`Renderer::spawn_look_thumbnail`]).
    #[must_use]
    pub fn spawn_chroma_scatter(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<ChromaScatter, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler().submit_superseded(
            Priority::Thumbnail,
            id,
            move |token| r.chroma_scatter(&src, token),
            || Err(RenderError::Cancelled),
        )
    }

    /// Queues a full-resolution render at [`Priority::Export`]; navigation never cancels it. It runs
    /// on the export worker, which starts it when no viewport frame, tile pass or thumbnail is
    /// waiting and looks again before every tile, so browsing is not held up by an export
    /// ([`JobScheduler`](crate::scheduler::JobScheduler)).
    #[must_use]
    pub fn spawn_full(
        &self,
        src: PhotoSource,
        request: FullRequest,
    ) -> JobHandle<Result<FullImage, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                r.render_full_with(&src, &request, token, true, None)
            })
    }

    /// [`Renderer::spawn_full`] that also calls `progress` with the fraction of tiles done (0 to 1)
    /// after each tile, from the export worker's thread. The export queue shows it as the
    /// per-file percentage.
    #[must_use]
    pub fn spawn_full_reporting(
        &self,
        src: PhotoSource,
        request: FullRequest,
        progress: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> JobHandle<Result<FullImage, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                r.render_full_with(&src, &request, token, true, Some(&*progress))
            })
    }

    /// The hook the session calls to render a freshly developed photo's thumbnail: it queues the
    /// render at thumbnail priority and waits for it, so a viewport frame asked for meanwhile runs
    /// first.
    #[must_use]
    pub fn thumbnail_renderer(&self) -> Arc<dyn ThumbnailRenderer> {
        Arc::new(PhotoThumbnails {
            renderer: self.clone(),
        })
    }
}

/// The looks Apple Auto's picture is rendered with: the Signature look the white lift is sized for
/// (`<data>/looks/builtin/signature.json`, `SessionEnv::reference_look`), and Neutral. No display LUT:
/// the picture's display rendering is ours, ACES 2.
struct PictureLooks {
    signature: Option<Arc<LookTransform>>,
}

impl LookProvider for PictureLooks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        match (id.as_str(), &self.signature) {
            ("builtin:signature", Some(signature)) => Ok(ResolvedLook {
                transform: Arc::clone(signature),
                fingerprint: "builtin:signature".to_owned(),
            }),
            ("builtin:neutral", _) => Ok(ResolvedLook {
                transform: Arc::new(LookTransform::neutral()),
                fingerprint: "builtin:neutral".to_owned(),
            }),
            _ => Err(LookProblem(format!("The look {id} is not in the library."))),
        }
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        Err(LookProblem(format!(
            "The display LUT {id} is not in the library."
        )))
    }
}

/// The session's [`PictureRenderer`] hook: Apple Auto's pictures (`docs/19` section 3), made by a
/// renderer of their own that has no graphics device, no display LUT and only the Signature and
/// Neutral looks, so the app and `ym` give the model the same bytes on every Mac whatever the GPU
/// policy, the display rendering and the look library say. Each picture is the photo at its default
/// settings (every Auto the rule's: the photo's own estimates), no crop, no straightening, no turn by
/// the user and the camera's orientation applied, through the Signature look at 100 % (Neutral when
/// the data folder has none) and ACES 2. Its caches are dropped after each picture.
pub struct AdvisorPictures {
    renderer: Renderer,
    look: LookId,
}

impl std::fmt::Debug for AdvisorPictures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdvisorPictures")
            .field("look", &self.look)
            .finish_non_exhaustive()
    }
}

impl AdvisorPictures {
    /// A picture renderer with the Signature look `signature` (`None`: Neutral) and the options of
    /// the app's renderer without its device (`options.gpu` is ignored).
    ///
    /// # Errors
    ///
    /// [`RenderError::Thread`] when the operating system cannot start the renderer's thread.
    pub fn new(
        signature: Option<Arc<LookTransform>>,
        options: &RenderOptions,
    ) -> Result<Self, RenderError> {
        let look = if signature.is_some() {
            LookId::builtin("signature")
        } else {
            LookId::neutral()
        };
        let options = RenderOptions {
            gpu: None,
            ..options.clone()
        };
        let renderer = Renderer::new(Arc::new(PictureLooks { signature }), options)?;
        Ok(Self { renderer, look })
    }

    /// The picture of a photo (see the type's documentation).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn picture(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, RenderError> {
        if (input.cancelled)() {
            return Err(RenderError::Cancelled);
        }
        let src = PhotoSource {
            id: input.id,
            raw: Arc::clone(input.raw),
            profile: Arc::clone(input.profile),
            estimates: Arc::clone(input.estimates),
            settings: DevelopSettings::with_look(self.look.clone()),
            pick: None,
            fingerprint: raw_fingerprint(input.raw),
            sensor: input.camera.matched.sensor,
            capture_method: super::raw::DEFAULT_CAPTURE_METHOD,
            before_view: false,
        };
        let picture = self.renderer.advisor_picture(&src, &CancelToken::new());
        // One picture per photo: nothing of it is worth keeping.
        self.renderer.forget(input.id);
        picture
    }
}

impl PictureRenderer for AdvisorPictures {
    fn render(&self, input: &PictureInput<'_>) -> Result<AdvisorPicture, String> {
        self.picture(input).map_err(|e| e.to_string())
    }
}

/// The session's thumbnail hook backed by a [`Renderer`] (see [`Renderer::thumbnail_renderer`]).
#[derive(Debug, Clone)]
pub struct PhotoThumbnails {
    renderer: Renderer,
}

impl ThumbnailRenderer for PhotoThumbnails {
    fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
        let src = PhotoSource::from_thumbnail_input(input);
        let handle = self.renderer.spawn_thumbnail(src);
        loop {
            if (input.cancelled)() {
                handle.cancel();
                return Err("The session was closed.".to_owned());
            }
            match handle.wait_timeout(Duration::from_millis(25)) {
                None => continue,
                Some(Ok(Ok(png))) => return Ok(png),
                Some(Ok(Err(e))) => return Err(e.to_string()),
                Some(Err(JobError::Cancelled)) => {
                    return Err("The render was cancelled.".to_owned());
                }
                Some(Err(e)) => return Err(e.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod scatter_tests {
    use ym_gpu::cpu::RgbImage;

    use super::*;

    #[test]
    fn the_scatter_keeps_every_nth_pixel_inside_the_photo_and_at_most_the_limit() {
        // 10 x 10 with the first row on the paper: 90 pixels inside.
        let mut outside = vec![0u8; 100];
        outside[..10].fill(1);
        let sampled = Sampled {
            img: RgbImage::new(10, 10),
            outside: Some(outside),
        };
        let all = every_nth_inside(&sampled, 1000);
        assert_eq!(all, (10..100).collect::<Vec<_>>());
        // 90 pixels for at most 40: every third, 30 of them, from the first inside.
        let some = every_nth_inside(&sampled, 40);
        assert_eq!(some.len(), 30);
        assert_eq!(&some[..3], &[10, 13, 16]);
        assert_eq!(every_nth_inside(&sampled, 45).len(), 45);
        // No mask: every pixel is inside.
        let whole = Sampled {
            img: RgbImage::new(10, 10),
            outside: None,
        };
        assert_eq!(every_nth_inside(&whole, 2048).len(), 100);
    }

    #[test]
    fn scatter_coordinates_are_the_nearest_decimals_of_four_places() {
        assert_eq!(scatter_coordinate(0.123_449), 0.1234);
        assert_eq!(scatter_coordinate(-0.123_46), -0.1235);
        assert_eq!(
            serde_json::to_string(&[scatter_coordinate(0.123_4), scatter_coordinate(0.3)]).unwrap(),
            "[0.1234,0.3]"
        );
        // Never `-0`.
        assert!(scatter_coordinate(-0.000_01).is_sign_positive());
    }
}
