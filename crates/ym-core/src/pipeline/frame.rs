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
//!
//! # On the device
//!
//! With a GPU and the view stages on (`RenderOptions::gpu_stages`), steps 3 to 5 and the output
//! sharpening are one device job per frame: sampler, C1 to C12, stage and C13 fused, one readback
//! of the RGBA8 frame (`device` module). A member of the stage set that is off keeps its host
//! boundary; a device error makes the whole frame again on the CPU with a note; a view too big for
//! one storage binding is cut into bands with the same pixels (`banded`). The display-linear frame
//! under the stage stays resident on the device, so a move of the Output slider is one small job.
//! A view magnified above 100 % is a device job too: the device sharpens the 100 % picture under the
//! view and the magnification kernel (`ym_gpu::kernels::magnify`, the CPU reference's taps and sums
//! bit for bit, C13 fused) makes the RGBA8 frame; a magnified view whose 100 % picture or RGBA8 frame
//! does not fit one binding is cut into bands of rows like any other view (each band reads the 100 %
//! picture under its rows). Where the device does not run the tail the CPU sharpens and magnifies
//! (`magnify_and_encode`). The CPU stage, where it runs, is cut into strips that each look at the
//! cancel token (`sharpen_photo`).
//!
//! # The device frame
//!
//! The native viewport asks for its frames with [`FrameOutput::Device`]
//! ([`Renderer::viewport_frame_to`]): a frame the device makes is then not read back but handed over
//! as a [`DeviceFrame`] (the pool buffer the kernels wrote), which the viewport draws straight into
//! its surface on the same queue (`docs/17-gpu.md` section 16). The bytes are those of the readback.
//! A frame the CPU makes (no device, a device error, a stage that is off) is a host frame whatever was
//! asked; every other request reads back as before.

mod device;
mod export_tail;

use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};

use rayon::prelude::*;
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, Frame, Layout, OutputTarget, View};
use ym_color::warp::chroma_of;
use ym_gpu::cpu::magnify::{MagnifyMap, magnify_with};
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::device::{Gpu, GpuError};
use ym_gpu::kernels::copy::Piece;
use ym_gpu::pool::PooledBuffer;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tail::{TailRoute, colour_device_tile_to_host, colour_tile_to_host};

use self::device::{
    Bounds, CameraSource, DeviceDisplay, DeviceInput, DisplayJob, GatherJob, PlainJob, Rgba8,
    TailJob, TailWork, Upload, ViewRoute,
};
pub use self::export_tail::{BAND_DEPTH, GpuBands, TailRequest};
use super::color::{GradePlan, PhotoSource};
use super::geometry::{Affine, Grid, oriented_to_sensor, sensor_to_oriented};
use super::graph::{
    Backing, ComputeDevice, Job, PartSource, Pixels, RenderReport, Renderer, ResidentPart,
    TileImage,
};
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
    /// Empty when the frame stayed on the device ([`RenderedFrame::on_device`]).
    pub rgba: Vec<u8>,
    /// The frame on the device, for a request that asked for one ([`FrameOutput::Device`]) and
    /// that the device made; [`RenderedFrame::rgba`] is then empty. `None` for every other frame.
    pub on_device: Option<DeviceFrame>,
    /// False when the frame was made from a coarser image than the zoom asks for because the
    /// full-resolution tiles were not ready ([`FrameOptions::coarse_only`]).
    pub sharp: bool,
    /// What the request did.
    pub report: RenderReport,
}

impl RenderedFrame {
    /// The bytes the frame holds, on the host or on the device (what a cache of frames keeps
    /// alive; the device pool's ledger counts the device's).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rgba.len()
            + self
                .on_device
                .as_ref()
                .map_or(0, |d| usize::try_from(d.byte_len()).unwrap_or(usize::MAX))
    }

    /// The RGBA8 bytes on the host: [`RenderedFrame::rgba`], or the device frame read back (one
    /// device job; tests, and a viewport that cannot draw the device's buffer).
    ///
    /// # Errors
    ///
    /// The device's error when the frame is on a device that is gone or fails.
    pub fn host_rgba(&self) -> Result<Cow<'_, [u8]>, GpuError> {
        match &self.on_device {
            Some(frame) => frame.read().map(Cow::Owned),
            None => Ok(Cow::Borrowed(&self.rgba)),
        }
    }
}

/// Where a finished frame's bytes go (`docs/17-gpu.md` section 16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameOutput {
    /// Read back into [`RenderedFrame::rgba`]: every consumer but the native viewport.
    #[default]
    Host,
    /// Kept on the device when the device makes the frame ([`RenderedFrame::on_device`]): the
    /// native viewport, which draws it from there. A frame the CPU makes is a host frame anyway.
    Device,
}

/// A finished RGBA8 frame that stayed on the device: the pool buffer the view job's kernels wrote,
/// not read back (`docs/17-gpu.md` section 16). Its bytes are those [`RenderedFrame::rgba`] would
/// hold: `4 * width * height` of them from offset 0, row-major, R, G, B, A per pixel (a
/// little-endian `u32` per pixel with R in the low byte), Display P3 sRGB-encoded for a viewport.
///
/// The buffer is on loan from the device's pool: it counts against the device's memory budget while
/// any clone of the frame lives, and goes back to the pool when the last one is dropped. Work that
/// reads it (a present on the same queue) is submitted before the frame is dropped, and the queue
/// runs it before any later job that is given the same buffer.
#[derive(Clone)]
pub struct DeviceFrame {
    buffer: Arc<PooledBuffer>,
    width: usize,
    height: usize,
    gpu: Weak<Gpu>,
}

impl std::fmt::Debug for DeviceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("buffer", &*self.buffer)
            .finish_non_exhaustive()
    }
}

impl PartialEq for DeviceFrame {
    /// The same buffer (a frame served again from the cache is the same frame).
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
            && (self.width, self.height) == (other.width, other.height)
    }
}

impl DeviceFrame {
    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The size of the frame in bytes: `4 * width * height`.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        4 * self.width as u64 * self.height as u64
    }

    /// The storage buffer that holds the frame: [`DeviceFrame::byte_len`] bytes from offset 0
    /// (the wgpu buffer may be longer, its pool size class). [`PooledBuffer::binding`] binds exactly
    /// the frame's bytes.
    #[must_use]
    pub fn buffer(&self) -> &PooledBuffer {
        &self.buffer
    }

    /// Whether the frame lives on `gpu` (a [`Weak`] of the device handle: two handles are the same
    /// device exactly when they point to the same allocation, which a live `Weak` keeps from being
    /// reused).
    #[must_use]
    pub fn is_on(&self, gpu: &Weak<Gpu>) -> bool {
        Weak::ptr_eq(&self.gpu, gpu)
    }

    /// The device the frame lives on, while it is there.
    #[must_use]
    pub fn device(&self) -> Option<Arc<Gpu>> {
        self.gpu.upgrade()
    }

    /// Whether a request on `gpu` (the renderer's device now) may serve this frame from a cache:
    /// the same device, and not lost.
    fn usable_with(&self, gpu: Option<&Arc<Gpu>>) -> bool {
        gpu.is_some_and(|g| self.is_on(&Arc::downgrade(g)) && !g.is_lost())
    }

    /// Reads the frame back to the host: one job of viewport priority on its device.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shutdown`] when the device is gone, else the job's error.
    pub fn read(&self) -> Result<Vec<u8>, GpuError> {
        let gpu = self.device().ok_or(GpuError::Shutdown)?;
        let buffer = Arc::clone(&self.buffer);
        let bytes = usize::try_from(self.byte_len()).unwrap_or(usize::MAX);
        let mut rgba = gpu.run(JobSpec::new(Priority::Viewport), move |ctx, token| {
            device::read_job(ctx, token, &buffer)
        })?;
        rgba.truncate(bytes);
        Ok(rgba)
    }
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

/// The chroma of the colour under a point at the input of the colour warp (`sample_chroma`): the
/// point of the warp's plane, `(e1·v, e2·v)` of the C9 output (`ym_color::warp::chroma_of`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChromaSample {
    /// Along `e1`, red against cyan.
    pub a: f64,
    /// Along `e2`, green against blue.
    pub b: f64,
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
    /// rotated crop), still unturned, and `window` is `None`. Made without R8 capture sharpening (a
    /// DNG carries none). With a look that includes its own display rendering, or an imported display
    /// LUT, the pixels are instead the scene values that ACES 2 turns into that picture
    /// ([`ym_color::display::ColorChain::apply_dng_pixel`]).
    DngLinearAwg4,
    /// The DNG's preview: the exact render in sRGB (display-linear, Rec.709 primaries) in the
    /// buffer's orientation, cropped and straightened as the user sees it, so it has the size of
    /// [`FullImage::window`] when the raw pixels carry one. Made without R8 capture sharpening, like
    /// the raw pixels, and never with output sharpening.
    DngPreview,
}

impl FullTarget {
    /// Whether this is one of the two images of a DNG, which carries no sharpening: the renders
    /// of both run with R8 capture sharpening off (`docs/09`, "What a DNG carries").
    #[must_use]
    pub fn is_dng(self) -> bool {
        matches!(self, Self::DngLinearAwg4 | Self::DngPreview)
    }
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
    /// The device failed during this render and the CPU made the picture again (the note in
    /// [`RenderReport::notes`] says why): the picture is the CPU reference's. The typed signal for an
    /// export that is more than one render (a DNG is its raw pixels and its preview), so that the
    /// file is wholly device or wholly CPU (`docs/17-gpu.md` section 9): the render that follows
    /// a fallen-back one runs on the CPU too, and one that fell back after the first went on the
    /// device is made again on the CPU.
    pub fell_back_to_cpu: bool,
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

pub(crate) use ym_gpu::cpu::sampler::Sampled;

/// Samples the `w x h` output pixels at offset `off` of `grid` from `src`, an image whose pixel
/// `(0, 0)` is buffer pixel `origin`: camera RGB through the geometry, with a mask of the pixels
/// that fell outside the photo.
///
/// The arithmetic is `ym-gpu`'s CPU port of the device sampler ([`ym_gpu::cpu::sampler`], the
/// reference of the kernel and the one implementation): a [`Grid`] only hands it the
/// numbers of its position chain ([`Grid::sampler_grid`]). The port is a verbatim move of what this
/// function was, and `tests::the_sampler_of_ym_gpu_is_the_function_this_was` keeps the old code and
/// requires the same bits.
pub(crate) fn sample_to_camera(
    src: &RgbImage,
    origin: (usize, usize),
    grid: &Grid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Sampled {
    ym_gpu::cpu::sampler::sample_to_camera(src, origin, &grid.sampler_grid(), off, w, h)
}

/// Where the pixels of a view come from once the raw group is there.
enum ViewSource {
    /// Camera RGB: `pixels` is a window of the raw group whose pixel (0, 0) is buffer pixel `origin`
    /// (on the device, the images that cover the rectangle that starts there), sampled through `grid`.
    Image {
        pixels: Pixels,
        origin: (usize, usize),
        grid: Grid,
    },
    /// Nothing of the photo is in view.
    Empty,
}

/// A view ready to be sampled: the raw group's pixels and the geometry that maps the view onto them
/// (steps 1 and 2 of the module documentation). Sampling itself is the next step, on the CPU
/// ([`PreparedView::sample_host`]) or on the device (`frame::device`).
pub(crate) struct PreparedView {
    source: ViewSource,
    /// The reduction of the raw group the pixels come from.
    pub scale: usize,
    /// False when the view was made from a coarser image than its zoom asks for.
    pub sharp: bool,
    width: usize,
    height: usize,
}

impl PreparedView {
    /// Samples the view on the CPU ([`sample_to_camera`]); the time goes to `sample_ms`. Pixels that
    /// are on the device are read back first ([`Job::host_pixels`]; the read is not sampling time).
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`], and what reading the pixels back or making them on the CPU says.
    fn sample_host(&self, job: &mut Job<'_>) -> Result<Arc<Sampled>, RenderError> {
        let (w, h) = (self.width, self.height);
        let sampled = match &self.source {
            // A view that reads nothing of the photo is all paper whatever is in the buffer: no need
            // to bring a picture back from the device for that.
            ViewSource::Image {
                pixels: Pixels::Resident { .. },
                ..
            } if self.window_rect().is_none() => Sampled {
                img: RgbImage::new(w, h),
                outside: Some(vec![1; w * h]),
            },
            ViewSource::Image {
                pixels,
                origin,
                grid,
            } => {
                let image = job.host_pixels(pixels)?;
                let t = std::time::Instant::now();
                let sampled = sample_to_camera(&image, *origin, grid, (0, 0), w, h);
                job.timings.sample_ms += t.elapsed().as_secs_f64() * 1000.0;
                sampled
            }
            ViewSource::Empty => Sampled {
                img: RgbImage::new(w, h),
                outside: Some(vec![1; w * h]),
            },
        };
        Ok(Arc::new(sampled))
    }

    /// The rectangle the view reads from the buffer, and the rectangle of the pixels it reads it
    /// from; the two are equal when the view reads all of it (no copy then). The read rectangle is
    /// where the pixels sample from, widened by 3 px for the filters (as the 100 % path reads its
    /// region) and cut by the image: sampling from the window gives the same pixels as from the
    /// whole image (the filters never reach beyond it, and where the window is cut by the image's
    /// own edge the clamp is the same). The window is what the device job takes, so a view of part
    /// of a large group (a band of a cut view, a zoomed fit view) takes only that part. No pixel is
    /// copied. `None` when the view reads nothing of the photo (a band of a letterbox that is all
    /// paper): there is nothing to upload, and a window of the whole image would count against the
    /// binding (and go up to the device) for a frame the sampler leaves empty.
    fn window_rect(&self) -> Option<(Rect, Rect)> {
        let ViewSource::Image {
            pixels,
            origin,
            grid,
        } = &self.source
        else {
            return None;
        };
        let whole = pixels.rect(*origin);
        let (x0, y0, x1, y1) = grid.source_bounds(0, 0, self.width, self.height, 3)?;
        let want = Rect::new(x0, y0, x1 - x0, y1 - y0).intersect(&whole);
        Some((if want.is_empty() { whole } else { want }, whole))
    }

    /// What a device job starts from for this view, and the bounding box of the pixels inside the
    /// photo in the sampled picture. With the sampler on the device the window of the pixels is the
    /// job's input: uploaded as camera RGB when it is on the host, gathered in place when it is on
    /// the device `route` runs on; otherwise (no sampler on the device, or the pixels are on another
    /// device) the CPU samples and the sampled pixels are uploaded.
    ///
    /// # Errors
    ///
    /// As [`PreparedView::sample_host`].
    fn device_input(
        &self,
        job: &mut Job<'_>,
        route: &ViewRoute,
    ) -> Result<(DeviceInput, Option<Bounds>), RenderError> {
        let (w, h) = (self.width, self.height);
        if route.stages.sampler
            && let ViewSource::Image {
                pixels,
                origin,
                grid,
            } = &self.source
            && let Some((want, whole)) = self.window_rect()
        {
            let camera = match pixels {
                Pixels::Host(image) => {
                    let (image, origin) = if want == whole {
                        (Arc::clone(image), *origin)
                    } else {
                        (
                            Arc::new(image.extract(want.relative_to(&whole))),
                            (want.x, want.y),
                        )
                    };
                    Some((CameraSource::Host(image), origin))
                }
                Pixels::Resident { parts, .. }
                    if parts.iter().all(|p| p.image.is_on(&route.gpu_ref())) =>
                {
                    let pieces = parts
                        .iter()
                        .flat_map(|p| {
                            p.image.pieces().iter().map(|q| Piece {
                                at: (p.at.0 + q.at.0, p.at.1 + q.at.1),
                                image: Arc::clone(&q.image),
                            })
                        })
                        .collect();
                    Some((
                        CameraSource::Resident {
                            pieces,
                            window: want,
                        },
                        (want.x, want.y),
                    ))
                }
                Pixels::Resident { .. } => None,
            };
            if let Some((source, origin)) = camera {
                return Ok((
                    DeviceInput::Camera {
                        source,
                        origin,
                        grid: grid.sampler_grid(),
                        width: w,
                        height: h,
                    },
                    self.inside_bounds(),
                ));
            }
        }
        let sampled = self.sample_host(job)?;
        let bounds = inside_bounds(sampled.outside.as_deref(), w, h);
        Ok((DeviceInput::Sampled(Upload::of(&sampled)), bounds))
    }

    /// The bounding box `(x0, y0, x1, y1)` (exclusive ends) of the pixels of the view that are inside
    /// the photo, found from the geometry alone (no pixel is touched); `None` when none is. It is
    /// what the CPU finds in the mask of a sampled view ([`inside_bounds`]).
    fn inside_bounds(&self) -> Option<Bounds> {
        match &self.source {
            ViewSource::Image { grid, .. } => {
                grid.sampler_grid()
                    .inside_bounds((0, 0), self.width, self.height)
            }
            ViewSource::Empty => None,
        }
    }

    /// Bytes of the camera RGB window an upload of this view would need (3 planes of f32), when the
    /// view has pixels to sample.
    fn source_window_bytes(&self) -> Option<u64> {
        let (want, _) = self.window_rect()?;
        Some(12 * want.w.next_multiple_of(2) as u64 * want.h as u64)
    }
}

// ---------------------------------------------------------------------------------------------
// The colour pass on the CPU
// ---------------------------------------------------------------------------------------------

/// Paints the paper over the pixels the mask marks as outside the photo.
fn paint_letterbox(rgba: &mut [u8], outside: Option<&[u8]>, background: [u8; 3]) {
    let Some(outside) = outside else { return };
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

/// The colour chain C1 to C13 over `sampled` on the CPU reference path: RGBA8 with the letterbox
/// painted (a frame that is not sharpened: amount 0, or the Before view).
fn color_pass_cpu(
    job: &mut Job<'_>,
    sampled: &Sampled,
    grade: &GradePlan,
    before: bool,
    background: [u8; 3],
) -> Result<Vec<u8>, RenderError> {
    job.check()?;
    let (w, h) = (sampled.img.width, sampled.img.height);
    let n = w * h;
    let frame =
        Frame::planar(w, h, &sampled.img.data).map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgb = vec![0u8; 3 * n];
    let view = if before { View::Before } else { View::Final };
    grade
        .chain
        .render_u8(view, &frame, (0, 0), &mut rgb, job.options().threads)
        .map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgba = vec![255u8; 4 * n];
    rgba.par_chunks_mut(4)
        .zip(rgb.par_chunks(3))
        .for_each(|(d, s)| {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 255;
        });
    paint_letterbox(&mut rgba, sampled.outside.as_deref(), background);
    Ok(rgba)
}

/// The colour chain C1 to C12 over `sampled` on the CPU reference path: display-linear f32,
/// interleaved, zero outside the photo, and the mask of the pixels outside.
fn display_linear_cpu(
    job: &mut Job<'_>,
    sampled: &Sampled,
    grade: &GradePlan,
) -> Result<(Vec<f32>, Option<Vec<u8>>), RenderError> {
    job.check()?;
    let (w, h) = (sampled.img.width, sampled.img.height);
    let frame =
        Frame::planar(w, h, &sampled.img.data).map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgb = vec![0.0f32; 3 * w * h];
    grade
        .chain
        .render_display_linear_f32(&frame, Layout::Interleaved, &mut rgb, job.options().threads)
        .map_err(|e| RenderError::Color(e.to_string()))?;
    if let Some(outside) = &sampled.outside {
        rgb.par_chunks_mut(3)
            .zip(outside.par_iter())
            .for_each(|(d, &o)| {
                if o != 0 {
                    d.fill(0.0);
                }
            });
    }
    Ok((rgb, sampled.outside.clone()))
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
    ///   sharpened picture is magnified to the view afterwards (on the device by the magnification
    ///   kernel, on the CPU by [`magnify_and_encode`]).
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

/// The pixels of a [`DisplayFrame`]: on the host, or resident on the device.
pub(crate) enum DisplayData {
    /// Display-linear f32, interleaved, zero outside the photo, and the mask of the pixels outside
    /// (`None` when every pixel is inside).
    Host {
        rgb: Vec<f32>,
        outside: Option<Vec<u8>>,
    },
    /// Four f32 planes on the device (`frame::device`): the tail of the view runs on them where they
    /// are.
    Device(Arc<DeviceDisplay>),
}

/// A view before output sharpening: display-linear f32 (C1 to C12) over the frame the stage reads
/// ([`ViewSharpening::frame_geometry`]: the view grown by the stage's receptive field, or the
/// 100 % picture under a magnified view). Cached per photo, so that a change of the amount
/// sharpens the same pixels again and nothing else runs. On the device the frame stays on the
/// device ([`DisplayData::Device`]) and a move of the slider runs the tail there.
pub(crate) struct DisplayFrame {
    /// Width of the frame.
    pub width: usize,
    /// Height of the frame.
    pub height: usize,
    /// The pixels.
    pub data: DisplayData,
    /// The reduction of the raw group the pixels came from.
    pub scale: usize,
    /// False when the view was made from a coarser image than its zoom asks for.
    pub sharp: bool,
    /// Where the colour chain ran.
    pub device: ComputeDevice,
}

/// The host pixels of a [`DisplayFrame`], borrowed.
struct HostDisplay<'a> {
    width: usize,
    height: usize,
    /// `3 * width * height` samples, interleaved.
    rgb: &'a [f32],
    outside: Option<&'a [u8]>,
}

impl DisplayFrame {
    /// Bytes the frame holds (on the host, or on the device: either way memory the cache keeps
    /// alive, and the pool's ledger counts the device's).
    pub(crate) fn bytes(&self) -> usize {
        match &self.data {
            DisplayData::Host { rgb, outside } => {
                rgb.len() * 4 + outside.as_ref().map_or(0, Vec::len)
            }
            DisplayData::Device(d) => d.bytes(),
        }
    }

    /// The frame's host pixels, `None` for a resident one.
    fn host(&self) -> Option<HostDisplay<'_>> {
        match &self.data {
            DisplayData::Host { rgb, outside } => Some(HostDisplay {
                width: self.width,
                height: self.height,
                rgb,
                outside: outside.as_deref(),
            }),
            DisplayData::Device(_) => None,
        }
    }

    /// Whether a request that wants its tail on the device `gpu` (`None`: on the CPU) can use this
    /// frame as it is: a resident frame only for the very device it lives on and a tail that runs
    /// there, a host frame only for a tail on the CPU (the other pairs would read the frame back or
    /// upload it, and a frame made again on the right side costs about the same as the colour pass it
    /// replaces).
    fn usable(&self, gpu: Option<&Weak<Gpu>>) -> bool {
        match (&self.data, gpu) {
            (DisplayData::Device(d), Some(device)) => Weak::ptr_eq(&d.gpu, device),
            (DisplayData::Host { .. }, None) => true,
            _ => false,
        }
    }
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

/// Core rows of one strip of the CPU stage (`sharpen_photo`): few enough that a cancel is noticed
/// within a fraction of a second (the stage costs 220 to 450 ms per megapixel on four shared cores,
/// `docs/algorithms/04` section 9; a strip of a retina frame is about 0.75 Mpixel), many enough that
/// the receptive field read above and below (17 to 21 rows) stays a small part of the work.
const CPU_SHARPEN_STRIP_ROWS: usize = 256;

/// The cancel check of the CPU stage: the request's own, callable from the threads the strips run on.
type CancelCheck<'a> = &'a (dyn Fn() -> Result<(), RenderError> + Sync);

/// Runs the stage on the photo's pixels of `display` on the CPU, in strips of rows with the stage's
/// own receptive field of context above and below (the tiling contract of
/// `docs/algorithms/04` section 5.2: the whole-frame result bit for bit, as in the export's
/// `sharpen_output`), calling `check` (the request's cancel check) before every strip, so that a
/// frame that is no longer wanted stops within one strip instead of after seconds. `None` when no
/// pixel is inside the photo.
///
/// The strips run side by side on the rayon pool (the render thread's, never the device thread's):
/// one after the other, a strip of a few hundred rows gives the stage's own bands of 128 rows too
/// little to share among the cores, and the frame took 1.8 times as long as in one piece (measured on
/// a 1764 x 1189 and a 2948 x 1988 frame at the fit scale, four shared cores); side by side it takes
/// 1.02 to 1.17 times as long (the receptive field each strip reads again, and the load of the
/// machine). A cancel stops the strips that have not started; the ones that run finish theirs, one
/// strip each.
fn sharpen_photo(
    check: CancelCheck<'_>,
    display: &HostDisplay<'_>,
    sharpening: &ViewSharpening,
) -> Result<Option<Sharpened>, RenderError> {
    let Some((x0, y0, x1, y1)) = inside_bounds(display.outside, display.width, display.height)
    else {
        return Ok(None);
    };
    let (w, h) = (x1 - x0, y1 - y0);
    let row = 3 * w;
    let halo = sharpening.halo;
    let starts: Vec<usize> = (0..h).step_by(CPU_SHARPEN_STRIP_ROWS).collect();
    let strips: Vec<Vec<f32>> = starts
        .par_iter()
        .map(|&start| -> Result<Vec<f32>, RenderError> {
            check()?;
            let end = (start + CPU_SHARPEN_STRIP_ROWS).min(h);
            let (top, bottom) = (start.saturating_sub(halo), (end + halo).min(h));
            let mut strip = Vec::with_capacity((bottom - top) * row);
            for y in top..bottom {
                let at = ((y0 + y) * display.width + x0) * 3;
                strip.extend_from_slice(&display.rgb[at..at + row]);
            }
            sharpening
                .stage
                .apply_interleaved_in_place(w, bottom - top, &mut strip, sharpening.scale)
                .map_err(|e| RenderError::Image(format!("This photo can't be rendered: {e}.")))?;
            strip.drain(..(start - top) * row);
            strip.truncate((end - start) * row);
            Ok(strip)
        })
        .collect::<Result<_, _>>()?;
    check()?;
    Ok(Some(Sharpened {
        x0,
        y0,
        w,
        h,
        rgb: strips.concat(),
    }))
}

/// C13 and the letterbox: encodes a display-linear `w x h` view to RGBA8 (the dither at the pixel's
/// place in the view, counted from `origin`: (0, 0) for a whole view, `(0, row)` for a band of one)
/// and paints the paper where `letterbox` is 1.
fn encode_view(
    job: &mut Job<'_>,
    view: &[f32],
    (w, h): (usize, usize),
    letterbox: Option<&[u8]>,
    grade: &GradePlan,
    (background, origin): ([u8; 3], (usize, usize)),
) -> Result<Vec<u8>, RenderError> {
    let frame = Frame::interleaved(w, h, view).map_err(|e| RenderError::Color(e.to_string()))?;
    let mut rgb8 = vec![0u8; 3 * w * h];
    grade
        .chain
        .encode_display_linear_u8(&frame, origin, &mut rgb8, job.r.options().threads)
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
    display: &HostDisplay<'_>,
    sharpening: &ViewSharpening,
    grade: &GradePlan,
    (w, h): (usize, usize),
    (background, origin): ([u8; 3], (usize, usize)),
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
    let cancel = job.cancel;
    let check = move || cancel.check().map_err(|_| RenderError::Cancelled);
    if let Some(s) = sharpen_photo(&check, display, sharpening)? {
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
    let letterbox = display.outside.map(|outside| {
        let mut mask = vec![0u8; w * h];
        mask.par_chunks_mut(w.max(1))
            .enumerate()
            .for_each(|(y, row)| row.copy_from_slice(&outside[(y + halo) * ew + halo..][..w]));
        mask
    });
    let rgba = encode_view(
        job,
        &view,
        (w, h),
        letterbox.as_deref(),
        grade,
        (background, origin),
    )?;
    job.runs.output_passes += 1;
    job.timings.output_ms += started.elapsed().as_secs_f64() * 1000.0;
    Ok(rgba)
}

/// A view magnified above 100 %: the stage has run on the 100 % picture `display` (the final
/// image's pixels from `origin`, `s = 1`), and that sharpened picture is magnified to the view with
/// the Catmull-Rom filter every magnified view uses, then encoded. Device pixels that fall outside
/// the final image, or on pixels of it that are outside the straightened photo, are the paper. The
/// magnification is the CPU reference of the device's (`ym_gpu::cpu::magnify`, the taps made per
/// pixel from this view's map); the device runs it from the same taps ([`magnify_map`],
/// `frame::device`'s magnified tail).
fn magnify_and_encode(
    job: &mut Job<'_>,
    display: &HostDisplay<'_>,
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
    let mut native = display.rgb.to_vec();
    let cancel = job.cancel;
    let check = move || cancel.check().map_err(|_| RenderError::Cancelled);
    if let Some(s) = sharpen_photo(&check, display, sharpening)? {
        for y in 0..s.h {
            let dst = ((s.y0 + y) * nw + s.x0) * 3;
            native[dst..dst + 3 * s.w].copy_from_slice(&s.rgb[y * s.w * 3..(y + 1) * s.w * 3]);
        }
    }
    let mut planes = RgbImage::new(nw, nh);
    {
        let (r, rest) = planes.data.split_at_mut(n);
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
    let (view, letterbox) = magnify_with(
        &planes,
        display.outside,
        (w, h),
        (fw, fh),
        origin,
        Some(job.cancel),
        |x, y| spec.view_to_final.apply(x as f64 + 0.5, y as f64 + 0.5),
    )
    .map_err(|e| RenderError::Image(format!("This view can't be magnified: {e}.")))?;
    job.check()?;
    let rgba = encode_view(
        job,
        &view,
        (w, h),
        Some(&letterbox),
        grade,
        (spec.background, spec.dither_origin),
    )?;
    job.runs.output_passes += 1;
    job.timings.output_ms += started.elapsed().as_secs_f64() * 1000.0;
    Ok(rgba)
}

/// The taps of a view magnified above 100 % ([`MagnifyMap`]: one per column and one per row) over
/// the `src` pixels of the final image (`image` at full resolution) from `origin`, the 100 % picture
/// the stage reads ([`ViewSharpening::frame_geometry`]). `None` when the view's map is not a scale and
/// a translation (no viewport makes one: the straightening and the crop are in the sampler's grid):
/// the CPU then magnifies the view. For a map with no cross terms a column's centre is the same at
/// every row, so these taps are the ones [`magnify_and_encode`] makes per pixel, bit for bit.
fn magnify_map(
    spec: &ViewSpec,
    origin: (usize, usize),
    image: (usize, usize),
    src: (usize, usize),
) -> Option<MagnifyMap> {
    let m = &spec.view_to_final.m;
    if m[0][1] != 0.0 || m[1][0] != 0.0 {
        return None;
    }
    let cols: Vec<f64> = (0..spec.out_w)
        .map(|x| spec.view_to_final.apply(x as f64 + 0.5, 0.5).0)
        .collect();
    let rows: Vec<f64> = (0..spec.out_h)
        .map(|y| spec.view_to_final.apply(0.5, y as f64 + 0.5).1)
        .collect();
    MagnifyMap::new(&cols, &rows, image, origin, src).ok()
}

// ---------------------------------------------------------------------------------------------
// The passes of a view
// ---------------------------------------------------------------------------------------------

/// A finished frame and where it came from.
struct Done {
    rgba: Rgba8,
    /// Where the colour chain ran.
    device: ComputeDevice,
    /// The reduction of the raw group.
    scale: usize,
    /// False when the frame was made from a coarser image than its zoom asks for.
    sharp: bool,
    /// The frame is paper only (nothing of the photo is in view): no stage ran, on either device. A
    /// band of a cut view that is nothing but paper is as much the device's as any other band.
    paper: bool,
}

/// What a pass of a view came to.
enum Flow {
    /// The frame.
    Done(Done),
    /// A device error ended the pass (it is noted, and the photo's flags are set,
    /// `Job::view_gpu_failed`): make the whole frame again on the CPU, never a frame of both.
    Cpu,
}

/// Answers the result of a device job: its value; the render's own cancellation; or `None` after a
/// device error, which the request answers on the CPU (a strict policy fails the request instead).
fn attempt<T>(job: &mut Job<'_>, result: Result<T, GpuError>) -> Result<Option<T>, RenderError> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(GpuError::Cancelled) => Err(RenderError::Cancelled),
        Err(e) => {
            job.view_gpu_failed(&e)?;
            Ok(None)
        }
    }
}

fn millis(since: std::time::Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// The sampled pixels of a view: on the device when the sampler is there (the colour pass is on the
/// CPU, or the caller wants the pixels, as the histogram does), else on the CPU. A device error falls
/// back to the CPU sampler for this call, with the usual note.
fn sample_routed(
    job: &mut Job<'_>,
    prepared: &PreparedView,
    route: Option<&ViewRoute>,
) -> Result<Arc<Sampled>, RenderError> {
    if let Some(r) = route.filter(|r| r.stages.sampler)
        && matches!(prepared.source, ViewSource::Image { .. })
    {
        let (input, _) = prepared.device_input(job, r)?;
        if input.fits(r) {
            let t = std::time::Instant::now();
            if let Some((w, h, planar)) = attempt(job, r.run(input, device::sample_job))? {
                job.timings.sample_ms += millis(t);
                return Ok(Arc::new(device::sampled_of(w, h, planar)));
            }
        }
    }
    prepared.sample_host(job)
}

/// A frame that is not sharpened (amount 0, or the Before view): the colour chain to C13 in one
/// pass, on the device when it is there.
fn plain_pass(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    route: Option<&ViewRoute>,
) -> Result<Flow, RenderError> {
    job.check()?;
    let prepared = prepare_view(job, spec)?;
    job.check()?;
    let (scale, sharp) = (prepared.scale, prepared.sharp);
    let started = std::time::Instant::now();
    if let Some(r) = route.filter(|r| r.stages.color) {
        let (input, _) = prepared.device_input(job, r)?;
        // A frame that does not fit one binding is made on the CPU, quietly (a view cut into bands
        // never gets here: `banded`).
        if input.fits(r) {
            let data = PlainJob {
                input,
                chain: Arc::clone(&grade.chain),
                before: spec.before,
                background: spec.background,
                origin: spec.dither_origin,
                output: spec.output,
            };
            let Some(rgba) = attempt(job, r.run(data, device::plain_job))? else {
                return Ok(Flow::Cpu);
            };
            job.runs.color_passes += 1;
            job.timings.pass_ms += millis(started);
            return Ok(Flow::Done(Done {
                rgba,
                device: ComputeDevice::Gpu,
                scale,
                sharp,
                paper: false,
            }));
        }
    }
    let sampled = sample_routed(job, &prepared, route)?;
    let started = std::time::Instant::now();
    let rgba = color_pass_cpu(job, &sampled, grade, spec.before, spec.background)?;
    job.runs.color_passes += 1;
    job.timings.pass_ms += millis(started);
    Ok(Flow::Done(Done {
        rgba: Rgba8::Host(rgba),
        device: ComputeDevice::Cpu,
        scale,
        sharp,
        paper: false,
    }))
}

/// The window output sharpening works on in the frame a pass makes: the bounding box of the pixels
/// inside the photo ("the stage works on the bounding box of the pixels inside the photo and mirrors at
/// its edge", `docs/01`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageWindow {
    /// The box of the frame the pass makes (a view in one piece).
    Own,
    /// A box found from the whole view, in the frame of this pass: what a band of a cut view hands
    /// to the stage, so that the stage sees what it sees in the view in one piece. A band's own box
    /// is narrower where the photo's edge is slanted, and a pixel at the edge would then mirror
    /// where the view in one piece sees the black of the empty corner. `None`: nothing of the photo.
    Of(Option<Bounds>),
}

impl StageWindow {
    /// The box to give the stage, `own` being the one found from the pass's own frame.
    fn or(self, own: Option<Bounds>) -> Option<Bounds> {
        match self {
            Self::Own => own,
            Self::Of(b) => b,
        }
    }
}

/// What a display pass made: the display-linear frame (kept, when anything wants it again), the
/// RGBA8 frame when the tail ran in the same job, and where things ran.
struct Built {
    display: Option<DisplayFrame>,
    rgba: Option<Rgba8>,
    scale: usize,
    sharp: bool,
    device: ComputeDevice,
}

/// What a display pass does beyond the colour chain.
struct TailPlan<'a> {
    /// The tail of the view (the stage and C13) in the same device job as the frame; `None` reads the
    /// display-linear frame back for the CPU to finish.
    work: Option<&'a TailWork>,
    /// Keep the display-linear frame resident (a tail job can sharpen it again): asked for when the
    /// frame is cached.
    keep: bool,
    /// The window the stage works on.
    window: StageWindow,
}

/// The display-linear frame of `read` (the view grown by the stage's receptive field): the sampler
/// (the CPU's, or the device's) and the colour chain C1 to C12, with the tail of the view in the same
/// device job when `plan` says the device runs it. `None` after a device error (answered on the CPU by
/// the caller).
fn build_display(
    job: &mut Job<'_>,
    route: Option<&ViewRoute>,
    prepared: &PreparedView,
    read: &ViewSpec,
    grade: &GradePlan,
    plan: &TailPlan<'_>,
) -> Result<Option<Built>, RenderError> {
    let (scale, sharp) = (prepared.scale, prepared.sharp);
    let started = std::time::Instant::now();
    if let Some(r) = route.filter(|r| r.stages.color) {
        let (input, own) = prepared.device_input(job, r)?;
        let bounds = plan.window.or(own);
        if input.fits(r) {
            let data = DisplayJob {
                input,
                bounds,
                chain: Arc::clone(&grade.chain),
                tail: plan.work.cloned(),
                keep: plan.keep && plan.work.is_some(),
                gpu: r.gpu_ref(),
                output: read.output,
            };
            let Some(out) = attempt(job, r.run(data, device::display_job))? else {
                return Ok(None);
            };
            job.runs.color_passes += 1;
            if plan.work.is_some() {
                job.runs.output_passes += 1;
            }
            // One job made the frame, and the tail when it was there: its time is the pass's.
            job.timings.pass_ms += millis(started);
            let frame = |data| DisplayFrame {
                width: read.out_w,
                height: read.out_h,
                data,
                scale,
                sharp,
                device: ComputeDevice::Gpu,
            };
            let display = if let Some(resident) = out.resident {
                Some(frame(DisplayData::Device(Arc::new(resident))))
            } else {
                out.planar.map(|(planes, planar)| {
                    let (rgb, outside) =
                        device::interleave_display(read.out_w, read.out_h, planes, &planar);
                    frame(DisplayData::Host { rgb, outside })
                })
            };
            return Ok(Some(Built {
                display,
                rgba: out.rgba,
                scale,
                sharp,
                device: ComputeDevice::Gpu,
            }));
        }
    }
    let sampled = sample_routed(job, prepared, route)?;
    let started = std::time::Instant::now();
    let (rgb, outside) = display_linear_cpu(job, &sampled, grade)?;
    job.runs.color_passes += 1;
    job.timings.pass_ms += millis(started);
    Ok(Some(Built {
        display: Some(DisplayFrame {
            width: read.out_w,
            height: read.out_h,
            data: DisplayData::Host { rgb, outside },
            scale,
            sharp,
            device: ComputeDevice::Cpu,
        }),
        rgba: None,
        scale,
        sharp,
        device: ComputeDevice::Cpu,
    }))
}

/// The display-linear frame of `read` made by the device in bands of `rows` rows: for a frame that
/// does not fit one storage binding and whose tail the CPU runs (the colour pass alone is on, as under
/// `Auto`). The colour chain is pointwise, so a band is a view of its own and the bands put together
/// are the frame in one piece, which the CPU stage then works on whole. `None` after a device error
/// (answered on the CPU by the caller).
fn build_display_in_bands(
    job: &mut Job<'_>,
    route: &ViewRoute,
    read: &ViewSpec,
    grade: &GradePlan,
    rows: usize,
) -> Result<Option<Built>, RenderError> {
    let started = std::time::Instant::now();
    let (w, h) = (read.out_w, read.out_h);
    let mut rgb: Vec<f32> = Vec::with_capacity(3 * w * h);
    let mut outside: Option<Vec<u8>> = None;
    let (mut scale, mut sharp) = (1, true);
    let mut done_rows = 0;
    for band in device::bands_of(read, rows) {
        job.check()?;
        let prepared = prepare_view(job, &band)?;
        let (input, _) = prepared.device_input(job, route)?;
        if !input.fits(route) {
            // The cut is made so that every band fits (`band_split`): a band that does not is the
            // frame's buffers not fitting, answered as such.
            let requested = 16 * band.out_w.next_multiple_of(2) as u64 * band.out_h as u64;
            job.view_gpu_failed(&GpuError::BufferTooLarge {
                requested,
                limit: route.band_bytes,
            })?;
            return Ok(None);
        }
        let data = DisplayJob {
            input,
            bounds: None,
            chain: Arc::clone(&grade.chain),
            tail: None,
            keep: false,
            gpu: route.gpu_ref(),
            // No tail: the job reads the display-linear frame back for the CPU, and makes no RGBA8.
            output: FrameOutput::Host,
        };
        let Some(out) = attempt(job, route.run(data, device::display_job))? else {
            return Ok(None);
        };
        job.runs.color_passes += 1;
        // A job without a tail reads the frame back: the invariant of `display_job`.
        let Some((planes, planar)) = out.planar else {
            return Ok(None);
        };
        let (band_rgb, band_outside) =
            device::interleave_display(band.out_w, band.out_h, planes, &planar);
        let band_px = band.out_w * band.out_h;
        match (band_outside, outside.as_mut()) {
            (Some(mask), Some(all)) => all.extend_from_slice(&mask),
            (Some(mask), None) => {
                let mut all = vec![0u8; w * done_rows];
                all.extend_from_slice(&mask);
                outside = Some(all);
            }
            (None, Some(all)) => all.resize(all.len() + band_px, 0),
            (None, None) => {}
        }
        rgb.extend_from_slice(&band_rgb);
        done_rows += band.out_h;
        (scale, sharp) = (prepared.scale, sharp && prepared.sharp);
    }
    job.timings.pass_ms += millis(started);
    Ok(Some(Built {
        display: Some(DisplayFrame {
            width: w,
            height: h,
            data: DisplayData::Host { rgb, outside },
            scale,
            sharp,
            device: ComputeDevice::Gpu,
        }),
        rgba: None,
        scale,
        sharp,
        device: ComputeDevice::Gpu,
    }))
}

/// The tail of a view as the device runs it: the stage's inputs and the encode's, and for a view
/// magnified above 100 % the taps of its magnification.
fn tail_work(
    grade: &GradePlan,
    sharpening: &ViewSharpening,
    spec: &ViewSpec,
    magnify: Option<Arc<MagnifyMap>>,
) -> TailWork {
    TailWork {
        sharpen: sharpening.stage.clone(),
        scale: sharpening.scale,
        halo: sharpening.halo,
        view: (spec.out_w, spec.out_h),
        space: grade.chain.space(),
        dither: grade.chain.dither(),
        background: spec.background,
        origin: spec.dither_origin,
        magnify,
    }
}

/// The taps of a magnified view whose tail the device can run (`None` for a view at 100 % or below,
/// and for a magnified view the device leaves to the CPU): the map is a scale and a translation, the
/// 100 % picture under the view fits one storage binding as four planes, and so does the RGBA8 view
/// (both against the renderer's constant, [`ViewRoute::band_bytes`], so the choice is the same on
/// every machine). A view that does not fit is cut into bands first ([`band_split`]), and this is
/// asked again for each band.
fn device_magnify(
    route: &ViewRoute,
    spec: &ViewSpec,
    geometry: &FrameGeometry,
    image: (usize, usize),
) -> Option<Arc<MagnifyMap>> {
    let origin = geometry.origin?;
    let frame = 16 * geometry.width.next_multiple_of(2) as u64 * geometry.height as u64;
    let view = 4 * spec.out_w as u64 * spec.out_h as u64;
    if frame > route.band_bytes || view > route.band_bytes {
        return None;
    }
    magnify_map(spec, origin, image, (geometry.width, geometry.height)).map(Arc::new)
}

/// The road of a view with output sharpening (see the module documentation): the display-linear
/// frame the stage reads (from the display cache when `spec.cache`, else made), then the stage and
/// C13. On the device that is one job when the frame is made (sampler, colour pass, stage, C13, one
/// readback) and one small job when it is cached (stage and C13), so that a move of the Output slider
/// runs no sampler and no colour pass.
fn sharpened_pass(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: &ViewSharpening,
    route: Option<&ViewRoute>,
    window: StageWindow,
) -> Result<Flow, RenderError> {
    let image = job.final_size()?;
    let Some(geometry) =
        sharpening.frame_geometry(&spec.view_to_final, (spec.out_w, spec.out_h), image)
    else {
        // Nothing of the photo is in view: all paper.
        let mut rgba = vec![255u8; 4 * spec.out_w * spec.out_h];
        rgba.par_chunks_mut(4)
            .for_each(|d| d[..3].copy_from_slice(&spec.background));
        return Ok(Flow::Done(Done {
            rgba: Rgba8::Host(rgba),
            device: ComputeDevice::Cpu,
            scale: 1,
            sharp: true,
            paper: true,
        }));
    };
    let read = ViewSpec {
        out_w: geometry.width,
        out_h: geometry.height,
        view_to_final: geometry.view_to_final,
        ..spec.clone()
    };
    let key = spec
        .cache
        .then(|| display_key(job, &read, grade, sharpening, window))
        .transpose()?;
    // The tail runs on the device when the route has it: over the view grown by the halo, or, for a
    // view magnified above 100 %, over the 100 % picture under it, which the device sharpens and
    // magnifies to the view (`device_magnify`; a magnified view the device cannot take keeps the host
    // road, the stage and the magnification on the CPU).
    let magnify = route
        .filter(|r| r.stages.tail)
        .and_then(|r| device_magnify(r, spec, &geometry, image));
    let tail = route
        .filter(|r| r.stages.tail && (geometry.origin.is_none() || magnify.is_some()))
        .map(|_| tail_work(grade, sharpening, spec, magnify));
    let tail_gpu = tail.as_ref().and(route).map(ViewRoute::gpu_ref);
    let cached = key
        .as_ref()
        .and_then(|k| job.g.cached_display(k))
        // A frame made from a coarser image than asked for is not served to a request that wants
        // the tiles (same rule as the finished frames).
        .filter(|d| !(spec.coarse_only && !d.sharp))
        .filter(|d| d.usable(tail_gpu.as_ref()));
    let (scale, sharp, device);
    let mut rgba: Option<Rgba8> = None;
    let display: Option<Arc<DisplayFrame>>;
    match cached {
        Some(d) => {
            job.runs.display_hits += 1;
            (scale, sharp, device) = (d.scale, d.sharp, d.device);
            display = Some(d);
        }
        None => {
            // A frame the device colours that does not fit one binding, and whose tail the CPU runs
            // (the colour pass alone is on, as under `Auto`): the colour pass is pointwise, so the
            // device makes it in bands of rows and the CPU gets the whole frame to sharpen.
            let in_bands = match route {
                Some(r) if r.stages.color && tail.is_none() => band_split(job, &read, None, r)?,
                _ => None,
            };
            let built = match (in_bands, route) {
                (Some(rows), Some(r)) => build_display_in_bands(job, r, &read, grade, rows)?,
                _ => {
                    let prepared = prepare_view(job, &read)?;
                    let plan = TailPlan {
                        work: tail.as_ref(),
                        keep: key.is_some(),
                        window,
                    };
                    build_display(job, route, &prepared, &read, grade, &plan)?
                }
            };
            let Some(built) = built else {
                return Ok(Flow::Cpu);
            };
            (scale, sharp, device) = (built.scale, built.sharp, built.device);
            rgba = built.rgba;
            display = built.display.map(|d| {
                let d = Arc::new(d);
                if let Some(k) = key {
                    job.g.store_display(k, Arc::clone(&d));
                }
                d
            });
        }
    }
    let rgba = match (rgba, display) {
        (Some(rgba), _) => rgba,
        // Made without the tail (or served from the cache): the tail now, where the frame is.
        (None, Some(display)) => match &display.data {
            DisplayData::Device(d) => {
                let (Some(r), Some(tail)) = (route, tail) else {
                    return Ok(Flow::Cpu);
                };
                let started = std::time::Instant::now();
                let data = TailJob {
                    display: Arc::clone(d),
                    tail,
                    output: spec.output,
                };
                let Some(rgba) = attempt(job, r.run(data, device::tail_job))? else {
                    return Ok(Flow::Cpu);
                };
                job.runs.output_passes += 1;
                job.timings.output_ms += millis(started);
                rgba
            }
            DisplayData::Host { .. } => {
                let Some(host) = display.host() else {
                    return Ok(Flow::Cpu);
                };
                Rgba8::Host(match geometry.origin {
                    None => sharpen_and_encode(
                        job,
                        &host,
                        sharpening,
                        grade,
                        (spec.out_w, spec.out_h),
                        (spec.background, spec.dither_origin),
                    )?,
                    Some(origin) => {
                        magnify_and_encode(job, &host, origin, sharpening, grade, spec, image)?
                    }
                })
            }
        },
        // A pass that made neither a frame nor a display has nothing to finish: the CPU road.
        (None, None) => return Ok(Flow::Cpu),
    };
    Ok(Flow::Done(Done {
        rgba,
        device,
        scale,
        sharp,
        paper: false,
    }))
}

/// The view, or the bands it is cut into, made by one pass each.
fn render_pass(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: Option<&ViewSharpening>,
    mut route: Option<ViewRoute>,
) -> Result<Done, RenderError> {
    loop {
        let flow = match &route {
            Some(r) => banded(job, spec, grade, sharpening, r)?,
            None => whole(job, spec, grade, sharpening, None, StageWindow::Own)?,
        };
        match flow {
            Flow::Done(done) => return Ok(done),
            // A device error ended the pass: the frame again, on the CPU only.
            Flow::Cpu if route.is_some() => route = None,
            // The CPU road never asks for the CPU road (an invariant of the passes); a request that
            // did would loop, so it is refused instead.
            Flow::Cpu => {
                return Err(RenderError::Image(
                    "This frame can't be rendered: the CPU path gave up.".to_owned(),
                ));
            }
        }
    }
}

/// One pass over `spec` as a whole; `window` is the window the stage works on (a frame without the
/// stage has none).
fn whole(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: Option<&ViewSharpening>,
    route: Option<&ViewRoute>,
    window: StageWindow,
) -> Result<Flow, RenderError> {
    match sharpening {
        Some(sh) => sharpened_pass(job, spec, grade, sh, route, window),
        None => plain_pass(job, spec, grade, route),
    }
}

/// The window the stage works on in the frame of the band of a cut view that starts at row `row` of
/// the view and is `rows` rows tall, from `whole`, the box of the frame of the view in one piece
/// (`None`: nothing of the photo is in view). The band's frame is the band grown by `halo` rows
/// above and below, which are rows `row ..< row + rows + 2 * halo` of the view's frame; the box keeps
/// its columns and is cut to those rows.
fn band_window(whole: Option<Bounds>, row: usize, rows: usize, halo: usize) -> Option<Bounds> {
    let (x0, y0, x1, y1) = whole?;
    let (top, bottom) = (y0.max(row), y1.min(row + rows + 2 * halo));
    (top < bottom).then(|| (x0, top - row, x1, bottom - row))
}

/// The window the stage works on in the frame a band of a magnified view reads (the 100 % picture
/// under the band's rows, `band`), from `whole`, the box in the frame of the view in one piece, which
/// starts at pixel `origin` of the final image: the same pixels of the final image, cut to the band's
/// frame. A band's own box would end where its frame ends, and the stage would mirror there; with the
/// whole view's box it mirrors only at the photo's edge, as the view in one piece does, and the rows
/// a band's taps read are at least the stage's receptive field away from the frame's cut
/// (`ViewSharpening::frame_geometry`'s margin), so they are the whole view's sharpened rows.
fn magnified_band_window(
    whole: Option<Bounds>,
    origin: (usize, usize),
    band: Option<&FrameGeometry>,
) -> Option<Bounds> {
    let (x0, y0, x1, y1) = whole?;
    let g = band?;
    let (bx, by) = g.origin?;
    let (left, top) = ((x0 + origin.0).max(bx), (y0 + origin.1).max(by));
    let (right, bottom) = (
        (x1 + origin.0).min(bx + g.width),
        (y1 + origin.1).min(by + g.height),
    );
    (left < right && top < bottom).then(|| (left - bx, top - by, right - bx, bottom - by))
}

/// The bounding box of the pixels inside the photo in the frame the stage reads for the whole of
/// `spec` (the view grown by the stage's receptive field, or the 100 % picture under a magnified
/// view), found from the geometry alone: no tile is assembled and no pixel is sampled. `None` when
/// nothing of the photo is in the frame.
fn whole_window(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    sharpening: &ViewSharpening,
) -> Result<Option<Bounds>, RenderError> {
    let image = job.final_size()?;
    let Some(g) = sharpening.frame_geometry(&spec.view_to_final, (spec.out_w, spec.out_h), image)
    else {
        return Ok(None);
    };
    let read = ViewSpec {
        out_w: g.width,
        out_h: g.height,
        view_to_final: g.view_to_final,
        ..spec.clone()
    };
    Ok(prepare_view_with(job, &read, false)?.inside_bounds())
}

/// The view as one pass, or as one pass per band when the device cannot hold its frame in one
/// binding ([`band_split`]): a band is a view of its own, the pixels are those of the whole view bit
/// for bit (every device stage of a view is tile-exact, and the dither counts its texels from the
/// band's place in the view), and a device error in any band, or a band the device could not take
/// whole (its buffers would not fit, so the CPU made it), makes the whole frame again on the CPU:
/// never a frame of both.
fn banded(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    grade: &GradePlan,
    sharpening: Option<&ViewSharpening>,
    route: &ViewRoute,
) -> Result<Flow, RenderError> {
    let Some(rows) = band_split(job, spec, sharpening, route)? else {
        return whole(job, spec, grade, sharpening, Some(route), StageWindow::Own);
    };
    // The stage works on the bounding box of the photo in the view's frame: every band is given
    // the box of the view in one piece, cut to its rows (a band's own box is narrower where the
    // photo's edge is slanted, and the stage would mirror there instead of seeing the black of the
    // empty corner, as it does in the view in one piece).
    let whole_box = sharpening
        .map(|sh| whole_window(job, spec, sh))
        .transpose()?;
    let halo = sharpening.map_or(0, |s| s.halo);
    // A band of a magnified view reads the 100 % picture under its own rows: its window is the
    // whole view's box moved to where that picture starts (`magnified_band_window`).
    let image = job.final_size()?;
    let frame_of = |v: &ViewSpec| {
        sharpening.and_then(|sh| sh.frame_geometry(&v.view_to_final, (v.out_w, v.out_h), image))
    };
    let whole_origin = frame_of(spec).and_then(|g| g.origin);
    // The bands of a frame that stays on the device are gathered there, into one buffer the device
    // can bind whole; a frame larger than that is read back band by band, as every frame is.
    let frame_bytes = 4 * spec.out_w as u64 * spec.out_h as u64;
    let host_bands;
    let spec = if spec.output == FrameOutput::Device && frame_bytes > route.frame_bytes {
        host_bands = ViewSpec {
            output: FrameOutput::Host,
            ..spec.clone()
        };
        &host_bands
    } else {
        spec
    };
    let mut parts = Vec::new();
    let (mut scale, mut sharp) = (1, true);
    let mut row = 0;
    for band in device::bands_of(spec, rows) {
        job.check()?;
        let window = match (whole_box, whole_origin) {
            (Some(b), Some(origin)) => {
                StageWindow::Of(magnified_band_window(b, origin, frame_of(&band).as_ref()))
            }
            (Some(b), None) => StageWindow::Of(band_window(b, row, band.out_h, halo)),
            (None, _) => StageWindow::Own,
        };
        row += band.out_h;
        match whole(job, &band, grade, sharpening, Some(route), window)? {
            Flow::Done(done) if done.device == ComputeDevice::Gpu || done.paper => {
                parts.push(done.rgba);
                // A band of paper says nothing of the raw group; of the others the coarsest counts,
                // as `sharp` says that one of them was made from less than the zoom asks for.
                if !done.paper {
                    scale = scale.max(done.scale);
                    sharp &= done.sharp;
                }
            }
            Flow::Done(_) | Flow::Cpu => return Ok(Flow::Cpu),
        }
    }
    let rgba = if parts.iter().any(|p| matches!(p, Rgba8::Device(_))) {
        let data = GatherJob {
            parts,
            bytes: frame_bytes,
        };
        let Some(rgba) = attempt(job, route.run(data, device::gather_job))? else {
            return Ok(Flow::Cpu);
        };
        rgba
    } else {
        let mut rgba = Vec::with_capacity(4 * spec.out_w * spec.out_h);
        for part in parts {
            if let Rgba8::Host(bytes) = part {
                rgba.extend_from_slice(&bytes);
            }
        }
        Rgba8::Host(rgba)
    };
    Ok(Flow::Done(Done {
        rgba,
        device: ComputeDevice::Gpu,
        scale,
        sharp,
        paper: false,
    }))
}

/// How many rows per band a view is cut into, or `None` for one pass: only a view the device makes
/// whole, from the sampler to the RGBA8 frame, is cut, and only when its frame (four planes of f32
/// over the view grown by the stage's receptive field) or the camera RGB it reads does not fit one
/// storage binding of [`RenderOptions::view_band_bytes`] ([`device::band_rows`]; a constant of the
/// renderer, so the cut is the same on every machine). A sharpened view magnified above 100 % reads
/// the 100 % picture under it: it is cut when that picture (four planes of f32) or its RGBA8 frame
/// does not fit, each band reading the 100 % picture under its own rows (`device_magnify`). The
/// bands give the view in one piece: the stage's window is the whole view's, the dither counts from
/// the band's row, and a row's taps come from the band's map, which is the view's with the band's
/// offset folded in (one `f64` rounding, as for the sampler's bands; the tests find the bands
/// identical to the view in one piece).
///
/// The camera RGB a view reads can be larger than its frame (the whole group at 1/2 is 150 MB for a
/// 50 MP photo; full-resolution tiles under a view between 50 and 100 % hold up to four pixels per
/// pixel shown): the bands share it out, and the cut is made finer until the window of every band,
/// grown by the stage's receptive field, fits. The window of full-resolution tiles is found from the
/// geometry alone, without computing a tile.
///
/// [`RenderOptions::view_band_bytes`]: crate::pipeline::RenderOptions::view_band_bytes
fn band_split(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    sharpening: Option<&ViewSharpening>,
    route: &ViewRoute,
) -> Result<Option<usize>, RenderError> {
    let stages = route.stages;
    // Only a view the device colours is cut, and with output sharpening only when the device also
    // runs the stage (the CPU stage takes the whole frame: `build_display_in_bands`).
    if !(stages.color && (sharpening.is_none() || stages.tail)) {
        return Ok(None);
    }
    let step = spec.view_to_final.m[0][0];
    let magnified = sharpening.is_some() && step.is_finite() && step < 1.0 - 1e-9;
    let halo = sharpening.map_or(0, |s| s.halo);
    // A magnified view's bands are sized by the loop below, from the pictures each one reads.
    let by_frame = if magnified {
        None
    } else {
        device::band_rows(spec.out_w, spec.out_h, halo, route.band_bytes, 1)
    };
    let plan = job.plan()?;
    let image = job.final_size()?;
    let geometry = job.geometry().at_full_resolution();
    // The camera RGB one band reads: the view's own rows, or with output sharpening the band grown
    // by the stage's receptive field (the frame the band's colour pass makes), or the 100 % picture
    // under a magnified band, which must fit one binding itself, with the band's RGBA8.
    let source_fits = |job: &mut Job<'_>, band: &ViewSpec| -> Result<bool, RenderError> {
        let frame = sharpening
            .and_then(|sh| sh.frame_geometry(&band.view_to_final, (band.out_w, band.out_h), image));
        if magnified && let Some(g) = &frame {
            let planes = 16 * g.width.next_multiple_of(2) as u64 * g.height as u64;
            let rgba8 = 4 * band.out_w as u64 * band.out_h as u64;
            if planes > route.band_bytes || rgba8 > route.band_bytes {
                return Ok(false);
            }
        }
        // The CPU samples: what goes up is the band's frame, which the cut already holds.
        if !stages.sampler {
            return Ok(true);
        }
        let read = match frame {
            Some(g) => ViewSpec {
                out_w: g.width,
                out_h: g.height,
                view_to_final: g.view_to_final,
                ..band.clone()
            },
            None => band.clone(),
        };
        let bytes = if !plan.reduced && plan.effective_scale(read.k) == 1 {
            // Full-resolution tiles: the region `prepare_view` would read (a coarse frame reads
            // less, the 1/2 image, which this bounds too).
            geometry
                .grid_from((plan.out_w, plan.out_h), read.view_to_final)
                .source_bounds(0, 0, read.out_w, read.out_h, 3)
                .map(|(x0, y0, x1, y1)| {
                    12 * (x1 - x0).next_multiple_of(2) as u64 * (y1 - y0) as u64
                })
        } else {
            prepare_view(job, &read)?.source_window_bytes()
        };
        Ok(bytes.is_none_or(|b| b <= route.source_bytes))
    };
    let mut rows = by_frame.unwrap_or(spec.out_h).max(1);
    loop {
        let bands = device::bands_of(spec, rows);
        let mut fits = true;
        for band in &bands {
            if !source_fits(job, band)? {
                fits = false;
                break;
            }
        }
        if fits {
            return Ok((rows < spec.out_h).then_some(rows));
        }
        if rows == 1 {
            // Not even one row fits: the device cannot take the view, the CPU makes it.
            return Ok(None);
        }
        // One more band each time: the same rows for every machine and the fewest bands that fit.
        rows = spec.out_h.div_ceil(bands.len() + 1).min(rows - 1);
    }
}

fn display_key(
    job: &mut Job<'_>,
    grown: &ViewSpec,
    grade: &GradePlan,
    sharpening: &ViewSharpening,
    window: StageWindow,
) -> Result<String, RenderError> {
    let params = job.params()?;
    // The sharpening amount is not part of it: the frame is what the amount is applied to. The
    // halo is (it is what the view was grown by).
    let mut key = format!(
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
    );
    // A band of a cut view keeps the box of the whole view with its frame: the same pixels in the
    // frame of another view have another box, and a frame resident with one must not serve the other.
    if let StageWindow::Of(b) = window {
        key.push_str(&format!("|window{b:?}"));
    }
    Ok(key)
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
    /// The queue priority of the device jobs of the view: a thumbnail's must not jump ahead of the
    /// viewport's.
    pub gpu_priority: Priority,
    /// The place of the view's pixel (0, 0) in the picture the dither texels are counted in: (0, 0)
    /// for a whole view, `(0, row)` for a band of one ([`device::bands_of`]), so that a band
    /// dithers as the same rows of the whole view do.
    pub dither_origin: (usize, usize),
    /// Where the finished frame goes: [`FrameOutput::Device`] for the native viewport's frames
    /// only.
    pub output: FrameOutput,
}

/// The sampled camera RGB of a view and how it was found.
pub(crate) struct SampledView {
    pub sampled: Arc<Sampled>,
    pub scale: usize,
}

/// Steps 1 and 2 of the module documentation: the raw group's pixels (the reduced image, or the
/// full-resolution tiles the view reads) and the geometry that maps the view onto them.
pub(crate) fn prepare_view(
    job: &mut Job<'_>,
    spec: &ViewSpec,
) -> Result<PreparedView, RenderError> {
    prepare_view_with(job, spec, true)
}

/// [`prepare_view`]; with `pixels` false the full-resolution tiles a view reads are not assembled
/// (the view then holds an empty image: it is for its grid and its scale, the bounding box of the
/// photo in it, not for sampling).
fn prepare_view_with(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    pixels: bool,
) -> Result<PreparedView, RenderError> {
    let plan = job.plan()?;
    let geom = job.geometry();
    let (w, h) = (spec.out_w, spec.out_h);
    // A camera-processed file has no mosaic, so any scale works; the limit that keeps a reduced
    // image at least a few pixels across applies to both.
    let mut k = plan.effective_scale(spec.k);
    let mut sharp = true;
    // Full-resolution tiles of a mosaic: only when they are ready or may be computed.
    if !plan.reduced && k == 1 {
        let grid = geom
            .at_full_resolution()
            .grid_from((plan.out_w, plan.out_h), spec.view_to_final);
        if let Some((x0, y0, x1, y1)) = grid.source_bounds(0, 0, w, h, 3) {
            let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
            if spec.coarse_only && !job.tiles_ready(rect)? {
                k = plan.effective_scale(2).max(2);
                sharp = false;
            } else {
                let region = if pixels {
                    job.region_pixels(rect)?
                } else {
                    Pixels::Host(Arc::new(RgbImage::new(0, 0)))
                };
                return Ok(PreparedView {
                    source: ViewSource::Image {
                        pixels: region,
                        origin: (x0, y0),
                        grid,
                    },
                    scale: 1,
                    sharp,
                    width: w,
                    height: h,
                });
            }
        } else {
            // Nothing of the photo is in view.
            return Ok(PreparedView {
                source: ViewSource::Empty,
                scale: 1,
                sharp,
                width: w,
                height: h,
            });
        }
    }
    let backing = job.group_backing(k)?;
    let (group_w, group_h) = backing.size();
    let pixels = match backing {
        Backing::Cpu(image) => Pixels::Host(image),
        Backing::Gpu(image) => Pixels::Resident {
            parts: vec![ResidentPart {
                image,
                at: (0, 0),
                source: PartSource::Group(k),
            }],
            rect: Rect::new(0, 0, group_w, group_h),
        },
    };
    let inv = 1.0 / k as f64;
    let v2f = spec
        .view_to_final
        .then(&Affine::scale_translate(inv, inv, 0.0, 0.0));
    let grid = geom.grid_from((group_w, group_h), v2f);
    Ok(PreparedView {
        source: ViewSource::Image {
            pixels,
            origin: (0, 0),
            grid,
        },
        scale: k,
        sharp,
        width: w,
        height: h,
    })
}

/// Steps 1 to 3 of the module documentation, on the device when the sampler is there (the
/// histogram's sample: the pixels come back to the host, where the counting stays), else on the CPU.
/// A device error samples on the CPU for this call, with the usual note.
pub(crate) fn sample_view_routed(
    job: &mut Job<'_>,
    spec: &ViewSpec,
) -> Result<SampledView, RenderError> {
    let prepared = prepare_view(job, spec)?;
    let route = job
        .view_gpu()?
        .map(|gpu| ViewRoute::new(job, gpu, spec.gpu_priority));
    let sampled = sample_routed(job, &prepared, route.as_ref())?;
    Ok(SampledView {
        sampled,
        scale: prepared.scale,
    })
}

/// Steps 1 to 3 of the module documentation for a view of one pixel, the readout under the pointer,
/// which is asked for on every move of the mouse over the photo. Pixels that are on the device (the
/// caches keep the group and the tiles there,
/// [`RenderOptions::resident_caches`](super::graph::RenderOptions::resident_caches)) are sampled
/// there and one pixel comes back: a host read of them would bring the whole group or the tiles back
/// and replace the cache entries with host images ([`Job::host_pixels`]), and the next frame would
/// upload its window again, so the residency would be gone as soon as the pointer moved. Pixels that
/// are on the host are sampled on the host (no device job for a pixel). A device error samples on the
/// CPU for this call, with the usual note.
pub(crate) fn sample_view_in_place(
    job: &mut Job<'_>,
    spec: &ViewSpec,
) -> Result<SampledView, RenderError> {
    let prepared = prepare_view(job, spec)?;
    let on_device = matches!(
        prepared.source,
        ViewSource::Image {
            pixels: Pixels::Resident { .. },
            ..
        }
    );
    let sampled = if on_device {
        let route = job
            .view_gpu()?
            .map(|gpu| ViewRoute::new(job, gpu, spec.gpu_priority));
        sample_routed(job, &prepared, route.as_ref())?
    } else {
        prepared.sample_host(job)?
    };
    Ok(SampledView {
        sampled,
        scale: prepared.scale,
    })
}

/// A whole view: sample, colour pass, report. Frames of a viewport go through the frame cache.
pub(crate) fn render_view(mut job: Job<'_>, spec: &ViewSpec) -> Result<RenderedFrame, RenderError> {
    // The Before view reads a source made for it (`PhotoSource::for_before_view`: no R3), and nothing
    // else does: the raw front and the colour pass must agree on which picture this is.
    debug_assert_eq!(
        job.src.before_view, spec.before,
        "a Before view is rendered from the source made for it, and only that"
    );
    let grade = job.grade(spec.target, true)?;
    let sharpening = view_sharpening(&mut job, spec)?;
    let cache_key = spec
        .cache
        .then(|| frame_key(&mut job, spec, &grade, sharpening.as_ref()))
        .transpose()?;
    if let Some(key) = &cache_key
        && let Some(hit) = job.g.cached_frame(key)
        && !(spec.coarse_only && !hit.sharp)
        // A frame kept on the device serves a request on that device only, while it works.
        && hit
            .on_device
            .as_ref()
            .is_none_or(|d| d.usable_with(job.options().gpu.as_ref()))
    {
        job.runs.frame_hits += 1;
        let (scale, device) = (hit.report.scale, hit.report.device);
        let mut frame = (*hit).clone();
        frame.report = job.finish(scale, device, spec.target);
        return Ok(frame);
    }
    let route = job
        .view_gpu()?
        .map(|gpu| ViewRoute::new(&job, gpu, spec.gpu_priority));
    let gpu = route.as_ref().map(ViewRoute::gpu_ref);
    let done = render_pass(&mut job, spec, &grade, sharpening.as_ref(), route)?;
    let (rgba, on_device) = match (done.rgba, gpu) {
        (Rgba8::Host(rgba), _) => (rgba, None),
        (Rgba8::Device(buffer), Some(gpu)) => (
            Vec::new(),
            Some(DeviceFrame {
                buffer,
                width: spec.out_w,
                height: spec.out_h,
                gpu,
            }),
        ),
        // Only a device job keeps a frame on the device, and a request without a route runs none.
        (Rgba8::Device(_), None) => {
            return Err(RenderError::Image(
                "This frame can't be shown: it was made on a graphics card the request did not use."
                    .to_owned(),
            ));
        }
    };
    let graph = Arc::clone(&job.g);
    let report = job.finish(done.scale, done.device, spec.target);
    let frame = RenderedFrame {
        width: spec.out_w,
        height: spec.out_h,
        rgba,
        on_device,
        sharp: done.sharp,
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
    // A frame kept on the device and one read back are the same pixels in different places; each
    // request is served the kind it asked for.
    if spec.output == FrameOutput::Device {
        key.push_str("|device");
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
        self.viewport_frame_to(src, state, options, FrameOutput::Host, cancel)
    }

    /// [`Renderer::viewport_frame`] with the place the frame's bytes go: with
    /// [`FrameOutput::Device`] a frame the device makes stays there ([`RenderedFrame::on_device`],
    /// the same bytes the readback gives), for the native viewport to draw straight from the
    /// device's memory (`docs/17-gpu.md` section 16).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn viewport_frame_to(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        options: &FrameOptions,
        output: FrameOutput,
        cancel: &CancelToken,
    ) -> Result<RenderedFrame, RenderError> {
        let before_src;
        let src = if state.before {
            before_src = src.for_before_view();
            &before_src
        } else {
            src
        };
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
            dither_origin: (0, 0),
            output,
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
        let before_src;
        let src = if state.before {
            before_src = src.for_before_view();
            &before_src
        } else {
            src
        };
        let mut job = Job::new(self, src, cancel)?.with_gpu_priority(Priority::TileNeural);
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
        job.warm_tiles(Rect::new(x0, y0, x1 - x0, y1 - y0))?;
        let made = job.runs.tiles;
        job.finish(1, ComputeDevice::Cpu, OutputTarget::Viewport);
        Ok(made as usize)
    }

    /// The camera RGB under a point of the cell (the raw group's output, before C1, sampled at the
    /// scale the viewport uses) and the grade plan of the photo: what [`Renderer::sample_pixel`] and
    /// [`Renderer::sample_chroma`] read. `None` outside the photo (the letterbox).
    fn point_reading(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        css: (f64, f64),
        before: bool,
        cancel: &CancelToken,
    ) -> Result<Option<(Vec3, Arc<GradePlan>)>, RenderError> {
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
            before,
            coarse_only: false,
            cache: false,
            background: self.options().background,
            gpu_priority: Priority::Viewport,
            dither_origin: (0, 0),
            output: FrameOutput::Host,
        };
        let grade = job.grade(OutputTarget::Viewport, false)?;
        let view = sample_view_in_place(&mut job, &spec)?;
        if view.sampled.outside.is_some() {
            return Ok(None);
        }
        let rgb = pixel_rgb(&view.sampled.img);
        job.finish(view.scale, ComputeDevice::Cpu, OutputTarget::Viewport);
        Ok(Some((rgb, grade)))
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
        let before_src;
        let src = if state.before {
            before_src = src.for_before_view();
            &before_src
        } else {
            src
        };
        let Some((rgb, grade)) = self.point_reading(src, state, css, state.before, cancel)? else {
            return Ok(None);
        };
        let kind = if state.before {
            View::Before
        } else {
            View::Final
        };
        let v = grade.chain.apply_view_pixel(kind, rgb);
        let q = |c: f64| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        Ok(Some(PixelSample {
            r: q(v[0]),
            g: q(v[1]),
            b: q(v[2]),
        }))
    }

    /// The camera RGB under a point of the cell that [`Renderer::sample_chroma`] reads (and
    /// [`Renderer::sample_pixel`] outside the Before view): the raw group's output at the point, before
    /// C1. `None` outside the photo. For checks and diagnostics.
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn camera_rgb_at(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        css: (f64, f64),
        cancel: &CancelToken,
    ) -> Result<Option<Vec3>, RenderError> {
        Ok(self
            .point_reading(src, state, css, false, cancel)?
            .map(|(rgb, _)| rgb))
    }

    /// The chroma of the colour under a point of the cell at the input of the colour warp: the pixel
    /// [`Renderer::sample_pixel`] reads, through the photo's chain to C9
    /// (`StageTrace::c9_logc4`), projected with `ym_color::warp::chroma_of` (Pick colour on the
    /// Colour warp card, `docs/11`). Always the final picture's colour, also while the Before view is
    /// shown (the pins act on the final picture; Before leaves the warp out). `None` outside the
    /// photo (the letterbox).
    ///
    /// # Errors
    ///
    /// [`RenderError`] for unusable data or a cancelled job.
    pub fn sample_chroma(
        &self,
        src: &PhotoSource,
        state: &ViewportState,
        css: (f64, f64),
        cancel: &CancelToken,
    ) -> Result<Option<ChromaSample>, RenderError> {
        let Some((rgb, grade)) = self.point_reading(src, state, css, false, cancel)? else {
            return Ok(None);
        };
        let [a, b] = chroma_of(grade.chain.trace(rgb).c9_logc4);
        Ok(Some(ChromaSample { a, b }))
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
    ///
    /// The colour stage of a DNG's renders runs on the device when [`GpuStages::EXPORT_TAIL`] is on
    /// (the frame itself stays on the host: the DNG's white point is found in it). A device error
    /// there makes the whole render again with no device at all, with a note, as for every export;
    /// a strict policy fails the render instead.
    pub(crate) fn render_full_with(
        &self,
        src: &PhotoSource,
        request: &FullRequest,
        cancel: &CancelToken,
        yield_to_foreground: bool,
        progress: Option<&(dyn Fn(f32) + Sync)>,
    ) -> Result<FullImage, RenderError> {
        // Monotone across the two attempts.
        let best = AtomicU32::new(0);
        let mono = |f: f32| {
            if f > f32::from_bits(best.load(Ordering::Relaxed)) {
                best.store(f.to_bits(), Ordering::Relaxed);
                if let Some(p) = progress {
                    p(f);
                }
            }
        };
        match self.render_full_on(
            src,
            request,
            cancel,
            yield_to_foreground,
            Some(&mono),
            false,
        )? {
            Attempt::Done(image) => Ok(*image),
            Attempt::DeviceFailed(reason) => {
                let attempt = self.render_full_on(
                    src,
                    request,
                    cancel,
                    yield_to_foreground,
                    Some(&mono),
                    true,
                )?;
                let Attempt::Done(mut image) = attempt else {
                    return Err(RenderError::Gpu(reason));
                };
                image.report.notes.insert(
                    0,
                    format!(
                        "The graphics card could not render this photo, so the CPU did: {reason}."
                    ),
                );
                image.fell_back_to_cpu = true;
                Ok(*image)
            }
        }
    }

    /// One attempt of [`Renderer::render_full_with`]. With `cpu_reference` the attempt runs with no
    /// device at all (the options of the renderer without its device and stages).
    fn render_full_on(
        &self,
        src: &PhotoSource,
        request: &FullRequest,
        cancel: &CancelToken,
        yield_to_foreground: bool,
        progress: Option<&(dyn Fn(f32) + Sync)>,
        cpu_reference: bool,
    ) -> Result<Attempt, RenderError> {
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
        if cpu_reference {
            job = job.with_options(self.cpu_reference_options());
        }
        if request.target.is_dng() {
            // A DNG carries no sharpening (docs/09): its pixels and its preview are made without R8.
            job = job.without_capture_sharpening();
        }
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
        // The options of this render are the ones of the moment it began (a device attached while
        // an export runs cannot move half of it onto another device).
        let tile = job.options().full_tile_px.max(64);
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
        // The DNG's pixels and preview are made for sRGB, the display its fitted profile emulates: a
        // chain whose display rendering is not ours (a look with its own, an imported display LUT)
        // inverts it from sRGB for the pixels.
        let target_out = match request.target {
            FullTarget::DisplayLinear(t) => t,
            FullTarget::DngLinearAwg4 | FullTarget::DngPreview => OutputTarget::Srgb,
        };
        let grade = job.grade(target_out, false)?;
        let grid = geom.grid((plan.out_w, plan.out_h), (fw, fh));
        let mut rgb = vec![0.0f32; fw * fh * 3];
        let threads = job.options().threads;
        let strict = job.options().gpu_strict;
        let tiles_total = (fw.div_ceil(tile) * fh.div_ceil(tile)).max(1);
        // The colour stage of a DNG's renders runs on the device when the export tail is on; the
        // frame stays on the host. A device error there, or in the raw chain, makes the whole render
        // again on the CPU: one file is wholly device or wholly CPU, never a mix of tiles.
        let mut device_colour = if request.target.is_dng() {
            export_tail::dng_colour_gpu(&mut job)?
        } else {
            None
        };
        let colour_on_device = device_colour.is_some();
        let spec = JobSpec::new(Priority::Export).owned_by(u64::from(src.id.0));
        let mut best_fraction = 0.0f32;
        'render: loop {
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
                        // With the colour stage on the device the raw chain's tile may stay there
                        // (`Job::region_resident`); with the CPU colouring it the host's is wanted.
                        let region = if device_colour.is_some() {
                            job.region_resident(rect)
                        } else {
                            job.region(rect, false).map(TileImage::Host)
                        };
                        let region = match region {
                            Ok(region) => region,
                            Err(RenderError::Gpu(_)) if job.take_gpu_restart() => {
                                rgb.fill(0.0);
                                // The raw chain's route is gone for this photo; its colour stage
                                // follows it to the CPU.
                                device_colour = None;
                                continue 'render;
                            }
                            Err(e) => return Err(e),
                        };
                        let out = if let Some(gpu) = &device_colour {
                            let route = TailRoute::new(gpu, spec, cancel.clone());
                            let (grid_s, stop) =
                                (grid.sampler_grid(), export_tail::host_stop(request.target));
                            let colour = match region {
                                TileImage::Device(image) => colour_device_tile_to_host(
                                    &route,
                                    image,
                                    (x0, y0),
                                    grid_s,
                                    (ox, oy),
                                    tw,
                                    th,
                                    Arc::clone(&grade.chain),
                                    stop,
                                ),
                                TileImage::Host(image) => colour_tile_to_host(
                                    &route,
                                    export_tail::shared(image),
                                    (x0, y0),
                                    grid_s,
                                    (ox, oy),
                                    tw,
                                    th,
                                    Arc::clone(&grade.chain),
                                    stop,
                                ),
                            };
                            match colour {
                                Ok(out) => out,
                                Err(GpuError::Cancelled) => return Err(RenderError::Cancelled),
                                Err(e) => {
                                    let reason = e.to_string();
                                    tracing::warn!(error = %reason, "the device failed on a tile; the CPU takes over");
                                    self.note_export_tail_error(&e);
                                    if strict {
                                        return Err(RenderError::Gpu(reason));
                                    }
                                    return Ok(Attempt::DeviceFailed(reason));
                                }
                            }
                        } else {
                            // Only the colour stage on the device asks for a tile that stays there.
                            let TileImage::Host(region) = region else {
                                return Err(RenderError::Image(
                                    "A tile was left on the graphics card for the CPU's colour stage."
                                        .to_owned(),
                                ));
                            };
                            let sampled =
                                sample_to_camera(&region, (x0, y0), &grid, (ox, oy), tw, th);
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
                            out
                        };
                        for y in 0..th {
                            let d = ((oy + y) * fw + ox) * 3;
                            rgb[d..d + 3 * tw].copy_from_slice(&out[y * tw * 3..(y + 1) * tw * 3]);
                        }
                    }
                    tiles_done += 1;
                    if let Some(report) = progress {
                        // Monotone also across a restart on the CPU.
                        let fraction = tiles_done as f32 / tiles_total as f32;
                        if fraction > best_fraction {
                            best_fraction = fraction;
                            report(fraction);
                        }
                    }
                    ox += tw;
                }
                oy += th;
            }
            break;
        }
        let device = if colour_on_device && device_colour.is_some() {
            ComputeDevice::Gpu
        } else {
            ComputeDevice::Cpu
        };
        let fell_back_to_cpu = job.gpu_restarted();
        let report = job.finish(1, device, target_out);
        Ok(Attempt::Done(Box::new(FullImage {
            width: fw,
            height: fh,
            rgb,
            target: request.target,
            window,
            report,
            fell_back_to_cpu,
        })))
    }
}

/// How one attempt of a full-resolution render ended.
enum Attempt {
    /// The picture.
    Done(Box<FullImage>),
    /// The device failed in a stage the attempt had put on it (and the policy is not strict): the
    /// render is made again with no device, and says why.
    DeviceFailed(String),
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

    /// A display-linear picture of `w x h` with edges and texture everywhere, and a border of paper
    /// (the mask) `border` pixels wide on the left and right and `border` rows on the top and bottom.
    fn picture_with_paper(w: usize, h: usize, border: usize) -> (Vec<f32>, Vec<u8>) {
        let mut rgb = vec![0.0f32; 3 * w * h];
        let mut mask = vec![0u8; w * h];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x < border || y < border || x >= w - border || y >= h - border {
                    mask[i] = 1;
                    continue;
                }
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 40) as f32) / 16_777_216.0;
                let edge = if (x / 23 + y / 17) % 2 == 0 { 0.8 } else { 0.1 };
                for c in 0..3 {
                    rgb[3 * i + c] = edge * (0.6 + 0.2 * c as f32) + 0.05 * noise;
                }
            }
        }
        (rgb, mask)
    }

    fn sharpening_at(step: f64, image: (usize, usize)) -> ViewSharpening {
        ViewSharpening::of_view(100.0, &to_final(step, 0.0, 0.0), image, false)
            .expect("plan")
            .expect("a stage")
    }

    #[test]
    fn the_cpu_stage_in_strips_is_the_stage_on_the_whole_frame_bit_for_bit() {
        // Frames shorter than a strip, as tall as one, a little taller, and three strips; two
        // receptive fields (full size and shrunk); with paper round the photo, so that the box is
        // not the frame.
        for (w, h) in [
            (64usize, 5usize),
            (90, 255),
            (90, 256),
            (90, 300),
            (130, 700),
        ] {
            for (step, border) in [(1.0, 0usize), (3.0, 2)] {
                let (rgb, mask) = picture_with_paper(w, h, border);
                let outside = (border > 0).then_some(mask.as_slice());
                let host = HostDisplay {
                    width: w,
                    height: h,
                    rgb: &rgb,
                    outside,
                };
                let sh = sharpening_at(step, (6000, 4000));
                let strips = sharpen_photo(&|| Ok(()), &host, &sh)
                    .expect("strips")
                    .expect("a photo");
                // The same stage over the box in one call.
                let (x0, y0, x1, y1) = inside_bounds(outside, w, h).expect("a box");
                let (bw, bh) = (x1 - x0, y1 - y0);
                let mut whole = Vec::with_capacity(3 * bw * bh);
                for y in y0..y1 {
                    whole.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1) * 3]);
                }
                sh.stage
                    .apply_interleaved_in_place(bw, bh, &mut whole, sh.scale)
                    .expect("whole");
                assert_eq!((strips.x0, strips.y0, strips.w, strips.h), (x0, y0, bw, bh));
                assert!(
                    strips.rgb == whole,
                    "{w} x {h} at step {step}: strips differ from the whole frame"
                );
            }
        }
    }

    #[test]
    fn the_cpu_stage_looks_at_the_cancel_check_before_every_strip() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (w, h) = (80usize, 700usize);
        let (rgb, _) = picture_with_paper(w, h, 0);
        let host = HostDisplay {
            width: w,
            height: h,
            rgb: &rgb,
            outside: None,
        };
        let sh = sharpening_at(1.0, (6000, 4000));
        let strips = 700usize.div_ceil(CPU_SHARPEN_STRIP_ROWS);
        let calls = AtomicUsize::new(0);
        // Never cancelled: one look per strip (700 rows in strips of 256: three) and one at the end.
        let _ = sharpen_photo(
            &|| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            &host,
            &sh,
        )
        .expect("strips");
        assert_eq!(calls.load(Ordering::SeqCst), strips + 1);
        // Cancelled before the stage starts: no strip is computed, the stage says so.
        let computed = AtomicUsize::new(0);
        let result = sharpen_photo(
            &|| {
                computed.fetch_add(1, Ordering::SeqCst);
                Err(RenderError::Cancelled)
            },
            &host,
            &sh,
        );
        assert!(matches!(result, Err(RenderError::Cancelled)));
        assert!(computed.load(Ordering::SeqCst) <= strips);
        // Cancelled after the first strip has looked: the stage stops with the cancel.
        calls.store(0, Ordering::SeqCst);
        let result = sharpen_photo(
            &|| {
                if calls.fetch_add(1, Ordering::SeqCst) >= 1 {
                    Err(RenderError::Cancelled)
                } else {
                    Ok(())
                }
            },
            &host,
            &sh,
        );
        assert!(matches!(result, Err(RenderError::Cancelled)));
    }

    #[test]
    fn a_band_gets_the_box_of_the_whole_view_cut_to_its_rows() {
        // The view's frame has the box columns 10 to 70 and rows 5 to 95; the halo is 4 rows, so a
        // band of 20 view rows at row `r` has the frame rows `r ..< r + 28` of the view's frame.
        let whole = Some((10, 5, 70, 95));
        // The first band: the box starts inside its frame, at row 5.
        assert_eq!(band_window(whole, 0, 20, 4), Some((10, 5, 70, 28)));
        // One in the middle: the frame is inside the box.
        assert_eq!(band_window(whole, 40, 20, 4), Some((10, 0, 70, 28)));
        // The last: the box ends inside its frame, at row 95 of the view's frame.
        assert_eq!(band_window(whole, 80, 20, 4), Some((10, 0, 70, 15)));
        // A frame the box does not touch has no window (and nothing underflows).
        assert_eq!(band_window(Some((0, 50, 8, 60)), 0, 10, 2), None);
        assert_eq!(band_window(Some((0, 5, 8, 10)), 80, 10, 2), None);
        assert_eq!(band_window(None, 0, 10, 2), None);
        // The window the pass is given: its own box, or the one handed down.
        let own = Some((1, 2, 3, 4));
        assert_eq!(StageWindow::Own.or(own), own);
        assert_eq!(StageWindow::Of(None).or(own), None);
        assert_eq!(StageWindow::Of(whole).or(own), whole);
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

/// The sampler as it was before it moved into `ym-gpu` (`ym_gpu::cpu::sampler`), kept verbatim as
/// the reference of the bit-equality test below.
#[cfg(test)]
mod reference_sampler {
    use rayon::prelude::*;
    use ym_gpu::cpu::RgbImage;
    use ym_gpu::cpu::resample::sample_catmull_rom;
    use ym_gpu::cpu::sampler::Sampled;

    use crate::pipeline::geometry::Grid;

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
    pub(super) fn reference_sample_to_camera(
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
                                    let ix =
                                        (px.floor() as isize).clamp(0, sw as isize - 1) as usize;
                                    let iy =
                                        (py.floor() as isize).clamp(0, sh as isize - 1) as usize;
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
}

#[cfg(test)]
mod sampler_tests {
    use super::reference_sampler::reference_sample_to_camera;
    use super::*;
    use crate::pipeline::geometry::Geometry;
    use crate::settings::{Crop, CropAspect, DevelopSettings, NormRect, Orientation};

    /// A picture with structure at every scale (so that every filter tap matters) and a different
    /// value in every plane.
    fn picture(w: usize, h: usize) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for y in 0..h {
            for x in 0..w {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 40) as f32) / 16_777_216.0;
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                img.set_pixel(
                    x,
                    y,
                    [
                        fx * 0.7 + noise * 0.3,
                        fy * 0.6 + noise * 0.4 + ((x / 7 + y / 5) % 2) as f32 * 0.25,
                        (fx * fy).sqrt() + noise * 0.2,
                    ],
                );
            }
        }
        img
    }

    #[test]
    fn the_sampler_of_ym_gpu_is_the_function_this_was() {
        // Every orientation and quarter turn, three straighten angles (0 is the exact grid: a copy;
        // the others rotate), and output sizes that shrink (the box), magnify (Catmull-Rom) and
        // match: both of the grid's kernels and the copy, over windows of the output with an
        // offset, as the tiles of an export are.
        let (bw, bh) = (96usize, 64usize);
        let img = picture(bw, bh);
        let mut cases = 0;
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                for angle in [0.0, 4.0, -17.5] {
                    let settings = DevelopSettings {
                        orientation: match turns {
                            0 => Orientation::Deg0,
                            1 => Orientation::Deg90,
                            2 => Orientation::Deg180,
                            _ => Orientation::Deg270,
                        },
                        crop: Some(Crop {
                            rect: NormRect {
                                x: 0.04,
                                y: 0.07,
                                width: 0.9,
                                height: 0.8,
                            },
                            angle_deg: angle,
                            aspect: CropAspect::Free,
                        }),
                        ..DevelopSettings::default()
                    };
                    let g = Geometry::new(exif, &settings);
                    let own = g.at_full_resolution().output_size(bw, bh);
                    for out in [own, (own.0 / 3 + 1, own.1 / 3 + 1), (own.0 * 2, own.1 * 2)] {
                        let grid = g.grid((bw, bh), out);
                        let off = (out.0 / 5, out.1 / 7);
                        let (w, h) = (out.0 - off.0, out.1 - off.1);
                        let a = reference_sample_to_camera(&img, (0, 0), &grid, off, w, h);
                        let b = sample_to_camera(&img, (0, 0), &grid, off, w, h);
                        assert!(
                            a.img.data == b.img.data && a.outside == b.outside,
                            "exif {exif} turns {turns} angle {angle} out {out:?}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 8 * 4 * 3 * 3);
        // The pixel-exact grid of the 100 % view, sampling a window of the buffer.
        let g = Geometry::camera_only(1).at_full_resolution();
        let grid = g.grid((bw, bh), (bw, bh));
        let window = img.extract(Rect::new(10, 6, 40, 30));
        let a = reference_sample_to_camera(&window, (10, 6), &grid, (12, 9), 30, 20);
        let b = sample_to_camera(&window, (10, 6), &grid, (12, 9), 30, 20);
        assert!(a.img.data == b.img.data && a.outside == b.outside);
    }
}
