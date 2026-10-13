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
//! * **Scheduled calls**: `spawn_*` put a request on the render queue at the priority `docs/01`
//!   gives it and return a handle.

use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ym_color::display::OutputTarget;
use ym_gpu::queue::{CancelToken, JobError, JobHandle, Priority};
use ym_looks::library::LookId;

use super::RenderError;
use super::color::{PhotoSource, with_look};
use super::frame::{
    FrameOptions, FullImage, FullRequest, RenderedFrame, ViewSpec, render_view, sample_view,
};
use super::geometry::Affine;
use super::graph::{Job, Renderer};
use super::viewport::ViewportState;
use crate::session::{PhotoId, ThumbnailInput, ThumbnailRenderer};

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
        };
        Ok((spec, (tw, th)))
    }

    /// The photo as a thumbnail: a PNG of at most 240 px on its long edge (`thumbnail_ready`), in
    /// sRGB, with the crop, straighten and rotation of the settings applied.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn thumbnail_png(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<Vec<u8>, RenderError> {
        let mut job = Job::new(self, src, cancel)?;
        let (spec, (tw, th)) = self.thumbnail_spec(&mut job)?;
        let frame = render_view(job, &spec)?;
        let mut rgb = vec![0u8; 3 * tw * th];
        rgb.par_chunks_mut(3)
            .zip(frame.rgba.par_chunks(4))
            .for_each(|(d, s)| {
                d.copy_from_slice(&s[..3]);
            });
        encode_png_rgb8(tw, th, &rgb)
    }

    /// The photo as it is developed now, through one look at 100 % (the Look tab's tiles).
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
        self.thumbnail_png(&tile, cancel)
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
        let mut job = Job::new(self, src, cancel)?;
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
        };
        let grade = job.grade(OutputTarget::Viewport, false)?;
        let view = sample_view(&mut job, &spec)?;
        let img = &view.sampled.img;
        let n = img.width * img.height;
        let outside = view.sampled.outside.as_deref();
        let counts: [[u64; HISTOGRAM_BINS]; 3] = (0..n)
            .into_par_iter()
            .fold(
                || [[0u64; HISTOGRAM_BINS]; 3],
                |mut acc, i| {
                    if outside.is_some_and(|o| o[i] != 0) {
                        return acc;
                    }
                    let p = img.pixel(i % img.width, i / img.width);
                    let t = grade
                        .chain
                        .trace([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                    for (c, acc_c) in acc.iter_mut().enumerate() {
                        let v = t.c10_logc4.0[c].clamp(0.0, 1.0);
                        let bin = ((v * HISTOGRAM_BINS as f64) as usize).min(HISTOGRAM_BINS - 1);
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
            );
        let peak = counts.iter().flatten().copied().max().unwrap_or(0).max(1) as f64;
        let norm = |c: usize| -> Vec<f64> {
            counts[c]
                .iter()
                .map(|&v| ((v as f64 / peak) * 1000.0).round() / 1000.0)
                .collect()
        };
        job.finish(
            view.scale,
            super::graph::ComputeDevice::Cpu,
            OutputTarget::Viewport,
        );
        Ok(HistogramData {
            red: norm(0),
            green: norm(1),
            blue: norm(2),
        })
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
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Viewport, Some(id), move |token| {
                r.viewport_frame(&src, &state, &options, token)
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

    /// Queues a look tile at [`Priority::Thumbnail`].
    #[must_use]
    pub fn spawn_look_thumbnail(
        &self,
        src: PhotoSource,
        look: LookId,
    ) -> JobHandle<Result<Vec<u8>, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Thumbnail, Some(id), move |token| {
                r.look_thumbnail_png(&src, &look, token)
            })
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
