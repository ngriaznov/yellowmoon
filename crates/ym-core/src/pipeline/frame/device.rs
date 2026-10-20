//! The view stages on the device (`docs/17-gpu.md` section 11): the geometry sampler, the colour chain
//! C1 to C12, output sharpening and C13 of a viewport frame, a thumbnail or a look tile, as device jobs.
//!
//! # What a job is
//!
//! One job is one closure on the device thread ([`ViewRoute::run`]), submitted with the render's own
//! cancel token and the request's priority and owner (so that going to another photo drops the
//! queued ones). It records the stages it covers into one command list, submits it once and reads
//! back once: the group (or the window of 100 % tiles) is uploaded as camera RGB, the sampler maps it
//! through the geometry into a four-plane image (red, green, blue and the letterbox mask), the colour
//! pass makes display-linear C12 of it, and, when the tail is on the device, output sharpening and C13
//! make the RGBA8 frame, which is the one thing read back ([`display_job`]). The colour chain's tables
//! are persistent (`ym_gpu::kernels::color::ChainTables`), so an unchanged chain uploads nothing.
//!
//! The display-linear frame (the picture under the output sharpening) stays **resident** as a
//! [`DeviceDisplay`] when the tail is on the device and the request caches it, so that a move of the
//! Output slider is one small job that sharpens and encodes it again ([`tail_job`]) and runs no
//! sampler and no colour pass.
//!
//! # The device frame
//!
//! A request for the native viewport (`ViewSpec::output` is [`FrameOutput::Device`]) asks the jobs to
//! keep the RGBA8 frame where the kernels wrote it: the job records the same passes, submits them and
//! waits for them as before, but reads nothing back, and hands back the pool buffer itself
//! ([`Rgba8::Device`]). The viewport draws that buffer into its surface on the same queue
//! (`docs/17-gpu.md` section 16). The bands of a view cut into bands are gathered into one buffer on
//! the device ([`gather_job`]); every other request reads back as before. A view magnified above
//! 100 % has its tail on the device as well: the display-linear frame is then the 100 % picture under
//! the view, which the tail sharpens and the magnification kernel makes bigger, with C13 fused
//! ([`record_magnified_tail`]), so the native viewport gets device frames at every zoom.
//!
//! # Which stage runs where
//!
//! [`ViewStages`] says what the device runs for a request, from the [`GpuStages`] the renderer was
//! given. A member that is off keeps a host boundary at its edges: the sampler off means
//! the CPU samples and the device gets the sampled pixels uploaded; the colour pass off means the CPU
//! colours. In a view the output sharpening and C13 are **one device pass** (the fused tail of
//! `ym_gpu::kernels::output_sharpen`, bit for bit the chained pair): the device runs them when both
//! members are on (and the colour pass is on, because the frame has to be on the device already); with
//! one of them off both stay on the CPU, which then reads the display-linear frame back.
//!
//! # Fall back
//!
//! A device error in any job is the caller's to answer: the whole request is made on the CPU, with a
//! note (never a frame of both). A buffer that does not fit ([`GpuError::OutOfBudget`] and
//! [`GpuError::BufferTooLarge`]) is tried once more after the pool and the persistent tables were let
//! go ([`ViewRoute::run`]; `docs/17-gpu.md` section 9).
//!
//! # No rayon on the device thread
//!
//! A job waits for nothing the rayon pool does: everything a job reads is prepared on the render
//! thread before it is submitted (the upload planes, the sampler's grid, the bounding box of the
//! photo), and everything it returns is converted on the render thread afterwards.

use std::sync::{Arc, Weak};
use std::time::Duration;

use rayon::prelude::*;
use ym_color::display::ColorChain;
use ym_color::encode::OutputSpace;
use ym_gpu::cpu::magnify::MagnifyMap;
use ym_gpu::cpu::output_sharpen::OutputSharpen;
use ym_gpu::cpu::sampler::{Sampled, SamplerGrid};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Placement, RgbImage};
use ym_gpu::device::{Gpu, GpuContext, GpuError};
use ym_gpu::image::{GpuImage, IMAGE_USAGE};
use ym_gpu::kernels::color::{PreparedChain, Stop, release_tables};
use ym_gpu::kernels::copy::{Piece, encode_gather};
use ym_gpu::kernels::encode::{C13, Encoding, encode_c13};
use ym_gpu::kernels::magnify::{MagnifyTarget, encode_magnify};
use ym_gpu::kernels::output_sharpen::{FusedC13, PreparedSharpen};
use ym_gpu::kernels::sampler::encode_sample;
use ym_gpu::pool::PooledBuffer;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::stream::{PassList, ReadId, upload_bytes};

use super::{FrameOutput, Job, ViewSpec};
use crate::app::gpu::GpuStages;
use crate::pipeline::graph::ViewGpu;

/// How often a render that waits for its device job looks at its cancel token.
const POLL: Duration = Duration::from_millis(5);

/// The bounding box `(x0, y0, x1, y1)`, exclusive ends, of the pixels inside the photo.
pub(super) type Bounds = (usize, usize, usize, usize);

/// Which view stages run on the device for one request (module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ViewStages {
    /// The geometry sampler: camera RGB is uploaded and sampled on the device.
    pub sampler: bool,
    /// The colour chain: C1 to C12 (to C13 on the frames that are not sharpened).
    pub color: bool,
    /// Output sharpening and C13 as the fused tail of the device pass.
    pub tail: bool,
}

impl ViewStages {
    /// The stages a renderer with `stages` runs on the device in a view.
    pub(super) fn of(stages: GpuStages) -> Self {
        let color = stages.contains(GpuStages::COLOR);
        Self {
            sampler: stages.contains(GpuStages::SAMPLER),
            color,
            tail: color
                && stages.contains(GpuStages::OUTPUT_SHARPEN)
                && stages.contains(GpuStages::ENCODE),
        }
    }
}

/// The device of a request and what every one of its jobs carries: the request's priority and owner,
/// the render's own cancel token (setting it skips the jobs that have not started and is seen by the
/// one that runs).
#[derive(Clone)]
pub(super) struct ViewRoute {
    gpu: Arc<Gpu>,
    /// What runs on the device.
    pub stages: ViewStages,
    spec: JobSpec,
    token: CancelToken,
    /// The largest storage binding of one device pass over a frame
    /// ([`RenderOptions::view_band_bytes`]).
    ///
    /// [`RenderOptions::view_band_bytes`]: crate::pipeline::RenderOptions::view_band_bytes
    pub band_bytes: u64,
    /// The largest uploaded camera RGB of one device pass: the same constant as the frame's (a
    /// renderer constant, not the device's limit, so that the cut does not depend on the machine:
    /// `docs/17-gpu.md` section 4).
    pub source_bytes: u64,
    /// The largest RGBA8 frame the device can hold in one buffer and bind whole: what a frame cut
    /// into bands may be to be gathered on the device ([`gather_job`]). The device's own limit: it
    /// decides only where the finished bytes are, never the pixels.
    pub frame_bytes: u64,
}

impl ViewRoute {
    /// The route of `job` for `gpu`, at the queue priority `priority`.
    pub(super) fn new(job: &Job<'_>, gpu: ViewGpu, priority: Priority) -> Self {
        Self {
            stages: ViewStages::of(gpu.stages),
            spec: JobSpec::new(priority).owned_by(u64::from(job.src.id.0)),
            token: job.cancel.clone(),
            band_bytes: job.options().view_band_bytes,
            source_bytes: job.options().view_band_bytes,
            frame_bytes: {
                let limits = &gpu.gpu.info().limits;
                limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size)
            },
            gpu: gpu.gpu,
        }
    }

    /// Identifies the device a resident frame belongs to. A [`Weak`] and not the address: while any
    /// resident frame holds it the allocation of the device cannot be given to the next one, so
    /// `Weak::ptr_eq` is true for the device that made the frame and for no other (an address alone
    /// is reused by the device that replaces a dropped one, and a frame of the old device would then
    /// be bound in the new device's passes: a validation error that loses a healthy device).
    pub(super) fn gpu_ref(&self) -> Weak<Gpu> {
        Arc::downgrade(&self.gpu)
    }

    /// Runs `work` over `data` as one job of the route and waits for it.
    ///
    /// A device that is out of memory or refuses a buffer is given one more try after the pool let go
    /// of what it holds unused and the persistent tables were dropped. A render that is cancelled
    /// while its job is still queued behind others (navigation, a closed session) stops waiting at
    /// once: the job is abandoned, skipped when it is popped (its token is set) and its buffers go
    /// back to the pool when it is dropped.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when the render was cancelled, the job's own error otherwise.
    pub(super) fn run<D, R>(
        &self,
        data: D,
        work: fn(&GpuContext, &CancelToken, &D) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>
    where
        D: Send + 'static,
        R: Send + 'static,
    {
        let handle =
            self.gpu
                .submit_with_token(self.spec, self.token.clone(), move |ctx, token| {
                    ctx.check_alive()?;
                    match work(ctx, token, &data) {
                        Err(GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }) => {
                            ctx.pool().trim();
                            release_tables(ctx);
                            work(ctx, token, &data)
                        }
                        other => other,
                    }
                });
        loop {
            if let Some(done) = handle.wait_timeout(POLL) {
                return done?;
            }
            if self.token.is_cancelled() {
                return Err(GpuError::Cancelled);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// What a job reads
// ---------------------------------------------------------------------------------------------

/// Sampled pixels ready to upload: three planes, or four when some pixel is outside the photo (the
/// fourth is 1.0 there). Built on the render thread.
pub(super) enum Upload {
    /// Camera RGB with no pixel outside the photo: the three planes of the sampled image as they are.
    Rgb(Arc<Sampled>),
    /// Four planes: the sampled image and its letterbox mask as a plane.
    Masked {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// `4 * width * height` samples, planar.
        data: Vec<f32>,
    },
}

impl Upload {
    /// The upload of a host-sampled picture.
    pub(super) fn of(sampled: &Arc<Sampled>) -> Self {
        let Some(mask) = &sampled.outside else {
            return Self::Rgb(Arc::clone(sampled));
        };
        let (width, height) = (sampled.img.width, sampled.img.height);
        let n = width * height;
        let mut data = Vec::with_capacity(4 * n);
        data.extend_from_slice(&sampled.img.data);
        data.extend(mask.iter().map(|&o| if o != 0 { 1.0f32 } else { 0.0 }));
        Self::Masked {
            width,
            height,
            data,
        }
    }
}

/// The camera RGB a device job samples from.
pub(super) enum CameraSource {
    /// A host image (a window of the raw group), uploaded for the job.
    Host(Arc<RgbImage>),
    /// Images that are on the device already (the raw group the caches kept there,
    /// [`crate::pipeline::RenderOptions::resident_caches`]): the job gathers `window` out of them
    /// with the copy kernel, which moves the samples and changes none, so the sampler reads the bits
    /// the uploaded window would have.
    Resident {
        /// The pieces of the picture, in buffer pixels.
        pieces: Vec<Piece>,
        /// The part of the picture to sample from, in buffer pixels.
        window: Rect,
    },
}

impl CameraSource {
    /// Width and height of the window in pixels.
    fn size(&self) -> (usize, usize) {
        match self {
            Self::Host(image) => (image.width, image.height),
            Self::Resident { window, .. } => (window.w, window.h),
        }
    }
}

/// What a device job starts from.
pub(super) enum DeviceInput {
    /// Camera RGB (a window of the raw group) that the device samples through `grid` into
    /// `width x height` pixels.
    Camera {
        /// The window of the group; its pixel (0, 0) is buffer pixel `origin`.
        source: CameraSource,
        /// Where the window starts in the buffer the grid maps into.
        origin: (usize, usize),
        /// The geometry.
        grid: SamplerGrid,
        /// Width of the sampled picture.
        width: usize,
        /// Height of the sampled picture.
        height: usize,
    },
    /// Pixels the CPU sampled.
    Sampled(Upload),
}

impl DeviceInput {
    /// Whether the buffers of a job over this input fit one storage binding: the frame (four
    /// planes of f32, which the colour pass, the sharpening and C13 read and write) and the uploaded
    /// camera RGB within the route's band size.
    pub(super) fn fits(&self, route: &ViewRoute) -> bool {
        let even = |n: usize| n.next_multiple_of(2) as u64;
        let frame = |w: usize, h: usize| 16 * even(w) * h as u64 <= route.band_bytes;
        match self {
            Self::Camera {
                source,
                width,
                height,
                ..
            } => {
                let (sw, sh) = source.size();
                12 * even(sw) * sh as u64 <= route.source_bytes && frame(*width, *height)
            }
            Self::Sampled(Upload::Rgb(s)) => frame(s.img.width, s.img.height),
            Self::Sampled(Upload::Masked { width, height, .. }) => frame(*width, *height),
        }
    }
}

/// The output sharpening and C13 of a view as the device runs them.
#[derive(Clone)]
pub(super) struct TailWork {
    /// The screen preset at the photo's amount.
    pub sharpen: OutputSharpen,
    /// The scale of the view (`s = long edge shown / long edge rendered`).
    pub scale: f32,
    /// The stage's receptive field: the view is grown by it on every side.
    pub halo: usize,
    /// The size of the view (without the border).
    pub view: (usize, usize),
    /// The colour space C13 encodes for.
    pub space: OutputSpace,
    /// Whether the 8-bit codes are dithered.
    pub dither: bool,
    /// The paper colour.
    pub background: [u8; 3],
    /// The place of the view's pixel (0, 0) in the picture the dither texels are counted in: (0, 0)
    /// for a whole view, `(0, row)` for a band of one.
    pub origin: (usize, usize),
    /// A view magnified above 100 %: the taps of its magnification. The display-linear frame is
    /// then the 100 % picture under the view (not the view grown by the halo); the tail sharpens it
    /// and magnifies it to the view with C13 fused ([`record_magnified_tail`]).
    pub magnify: Option<Arc<MagnifyMap>>,
}

/// A display-linear frame (C12) of a view on the device, before output sharpening: four planes (the
/// last is the letterbox mask) over the view grown by the stage's receptive field.
#[derive(Debug)]
pub(crate) struct DeviceDisplay {
    /// The planes.
    pub image: GpuImage,
    /// The bounding box of the pixels inside the photo, where the stage works.
    pub bounds: Option<Bounds>,
    /// The device it lives on ([`ViewRoute::gpu_ref`]).
    pub gpu: Weak<Gpu>,
}

impl DeviceDisplay {
    /// Bytes the frame holds on the device.
    pub(crate) fn bytes(&self) -> usize {
        self.image.byte_len() as usize
    }
}

/// The RGBA8 frame a job made (`4 * width * height` bytes, row-major, the letterbox painted):
/// read back, or kept on the device where the kernels wrote it (module notes, "The device frame").
pub(crate) enum Rgba8 {
    /// The bytes on the host.
    Host(Vec<u8>),
    /// The pool buffer the kernels wrote: the same bytes from offset 0 ([`PooledBuffer::size`] of
    /// them).
    Device(Arc<PooledBuffer>),
}

impl Rgba8 {
    /// The size of the frame in bytes.
    pub(crate) fn byte_len(&self) -> u64 {
        match self {
            Self::Host(bytes) => bytes.len() as u64,
            Self::Device(buffer) => buffer.size(),
        }
    }
}

/// Whether a request keeps its RGBA8 frame on the device.
fn keeps(output: FrameOutput) -> bool {
    output == FrameOutput::Device
}

/// Submits the passes of a job that made the RGBA8 frame `dst`, waits for them and returns the
/// frame: the readback `read` when there is one, else `dst` itself.
fn finish_rgba(
    passes: PassList<'_>,
    dst: Arc<PooledBuffer>,
    read: Option<ReadId>,
) -> Result<Rgba8, GpuError> {
    let mut done = passes.submit()?.finish()?;
    Ok(match read {
        Some(read) => Rgba8::Host(done.take(read)),
        None => Rgba8::Device(dst),
    })
}

// ---------------------------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------------------------

/// Uploads or samples the input: the picture the colour pass reads, three planes or four.
fn input_image(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    input: &DeviceInput,
) -> Result<GpuImage, GpuError> {
    match input {
        DeviceInput::Camera {
            source,
            origin,
            grid,
            width,
            height,
        } => {
            let src = match source {
                CameraSource::Host(image) => {
                    Arc::new(GpuImage::upload_rgb(ctx, "view source", image)?)
                }
                CameraSource::Resident { pieces, window } => {
                    encode_gather(passes, pieces, *window)?
                }
            };
            let sampled = encode_sample(passes, &src, *origin, grid, (0, 0), *width, *height)?;
            // The source must outlive the submission: a buffer made here (the upload, a gathered
            // copy) is kept by the pass list; a cached one is held by the input.
            if let Ok(owned) = Arc::try_unwrap(src) {
                let _ = passes.keep(owned.buf);
            }
            Ok(sampled)
        }
        DeviceInput::Sampled(Upload::Rgb(s)) => GpuImage::upload(
            ctx,
            "view sampled",
            s.img.width,
            s.img.height,
            3,
            &s.img.data,
            Placement::whole(s.img.width, s.img.height),
        ),
        DeviceInput::Sampled(Upload::Masked {
            width,
            height,
            data,
        }) => GpuImage::upload(
            ctx,
            "view sampled",
            *width,
            *height,
            4,
            data,
            Placement::whole(*width, *height),
        ),
    }
}

/// Records the tail over a resident display: C13 of the whole view first when the photo does not fill
/// it (the pixels outside the stage's window are the letterbox, which the mask paints), then output
/// sharpening fused with C13 over the photo's bounding box, writing its core into the same RGBA8
/// buffer. A magnified view takes [`record_magnified_tail`] instead. Returns the buffer and, when
/// `read`, the readback of it.
fn record_tail(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    display: &GpuImage,
    bounds: Option<Bounds>,
    tail: &TailWork,
    read: bool,
) -> Result<(Arc<PooledBuffer>, Option<ReadId>), GpuError> {
    if let Some(map) = &tail.magnify {
        return record_magnified_tail(ctx, passes, display, bounds, tail, map, read);
    }
    let (w, h) = tail.view;
    let halo = tail.halo;
    let view_rect = Rect::new(halo, halo, w, h);
    let bytes = (w * h * 4) as u64;
    let dst = passes.keep(ctx.pool().acquire("view rgba8", bytes, IMAGE_USAGE)?);
    let window = bounds.map(|(x0, y0, x1, y1)| Rect::new(x0, y0, x1 - x0, y1 - y0));
    let core = window
        .map(|b| b.intersect(&view_rect))
        .filter(|c| !c.is_empty());
    if core != Some(view_rect) {
        encode_c13(
            passes,
            display,
            view_rect,
            Encoding::Rgba8 {
                background: tail.background,
            },
            C13 {
                space: tail.space,
                dither: tail.dither,
                origin: tail.origin,
            },
            &dst,
            0,
            0,
        )?;
    }
    if let (Some(window), Some(core)) = (window, core) {
        let sharpen = PreparedSharpen::new(ctx, &tail.sharpen, tail.scale)?;
        let at = (core.x - halo, core.y - halo);
        sharpen.encode_rgba8(
            passes,
            display,
            window,
            core,
            &dst,
            w,
            at,
            FusedC13 {
                space: tail.space,
                dither: tail.dither,
                origin: (tail.origin.0 + at.0, tail.origin.1 + at.1),
                background: tail.background,
            },
        )?;
    }
    let read = read.then(|| passes.read(&dst, 0, bytes)).transpose()?;
    Ok((dst, read))
}

/// Records the tail of a view magnified above 100 % (`docs/17-gpu.md` section 16): the display is
/// the 100 % picture under the view; output sharpening runs over the photo's bounding box in it at
/// `s = 1` into a copy of it (the pixels outside the box stay as they are, as on the CPU), and the
/// magnification kernel samples that sharpened picture for every device pixel of the view, paints the
/// paper and encodes (C13 fused) into the RGBA8 buffer. The bytes `frame::magnify_and_encode` makes
/// from the same display-linear frame, but for C13's own rounding (`ym_gpu::kernels::magnify`).
#[allow(clippy::too_many_arguments)] // the tail's own inputs and the magnification
fn record_magnified_tail(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    display: &GpuImage,
    bounds: Option<Bounds>,
    tail: &TailWork,
    map: &MagnifyMap,
    read: bool,
) -> Result<(Arc<PooledBuffer>, Option<ReadId>), GpuError> {
    let (w, h) = (map.width(), map.height());
    let bytes = (w * h * 4) as u64;
    let dst = passes.keep(ctx.pool().acquire("view rgba8", bytes, IMAGE_USAGE)?);
    // The sharpened 100 % picture: a copy of the display (its mask plane included), the stage's
    // result written over the photo's bounding box.
    let sharp = GpuImage::alloc(
        ctx,
        "view magnified source",
        display.width,
        display.height,
        display.planes,
        display.place,
    )?;
    passes
        .encoder()
        .copy_buffer_to_buffer(&display.buf, 0, &sharp.buf, 0, display.byte_len());
    if let Some((x0, y0, x1, y1)) = bounds {
        let window = Rect::new(x0, y0, x1 - x0, y1 - y0);
        let sharpen = PreparedSharpen::new(ctx, &tail.sharpen, tail.scale)?;
        sharpen.encode(passes, display, window, window, &sharp, (x0, y0))?;
    }
    encode_magnify(
        passes,
        &sharp,
        map,
        MagnifyTarget::Rgba8 {
            dst: &dst,
            c13: C13 {
                space: tail.space,
                dither: tail.dither,
                origin: tail.origin,
            },
            background: tail.background,
        },
    )?;
    // The copy must outlive the submission.
    let _ = passes.keep(sharp.buf);
    let read = read.then(|| passes.read(&dst, 0, bytes)).transpose()?;
    Ok((dst, read))
}

// ---------------------------------------------------------------------------------------------
// The jobs
// ---------------------------------------------------------------------------------------------

/// A frame that is not sharpened (the Before view, amount 0): sampler (or an upload), the colour pass
/// to C13 in one dispatch, RGBA8 with the letterbox painted.
pub(super) struct PlainJob {
    /// What the job starts from.
    pub input: DeviceInput,
    /// The colour chain.
    pub chain: Arc<ColorChain>,
    /// The Before view (the camera default rendering) instead of the picture.
    pub before: bool,
    /// The paper colour.
    pub background: [u8; 3],
    /// The place of the frame's pixel (0, 0) in the picture the dither is counted in.
    pub origin: (usize, usize),
    /// Where the finished frame goes.
    pub output: FrameOutput,
}

/// The RGBA8 frame of a [`PlainJob`].
pub(super) fn plain_job(
    ctx: &GpuContext,
    token: &CancelToken,
    job: &PlainJob,
) -> Result<Rgba8, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "view plain");
    let sampled = input_image(ctx, &mut passes, &job.input)?;
    let prepared = PreparedChain::new(ctx, &job.chain)?;
    let (w, h) = (sampled.width, sampled.height);
    let bytes = (w * h * 4) as u64;
    let dst = passes.keep(ctx.pool().acquire("view rgba8", bytes, IMAGE_USAGE)?);
    prepared.encode_rgba8(
        &mut passes,
        &sampled,
        Rect::whole(w, h),
        job.before,
        prepared.dither(),
        job.origin,
        job.background,
        &dst,
        w,
        (0, 0),
    )?;
    let read = (!keeps(job.output))
        .then(|| passes.read(&dst, 0, bytes))
        .transpose()?;
    token.check()?;
    finish_rgba(passes, dst, read)
}

/// A sharpened frame: sampler (or an upload), the colour pass to C12, then the tail on the device or
/// a readback of the display-linear frame.
pub(super) struct DisplayJob {
    /// What the job starts from.
    pub input: DeviceInput,
    /// The bounding box of the photo's pixels in the picture the colour pass makes.
    pub bounds: Option<Bounds>,
    /// The colour chain.
    pub chain: Arc<ColorChain>,
    /// Output sharpening and C13 in the same job; `None` reads the display-linear frame back.
    pub tail: Option<TailWork>,
    /// Keep the display-linear frame resident (a tail job can sharpen it again).
    pub keep: bool,
    /// The device's identity, for the resident frame ([`ViewRoute::gpu_ref`]).
    pub gpu: Weak<Gpu>,
    /// Where the RGBA8 frame of the tail goes.
    pub output: FrameOutput,
}

/// What a [`DisplayJob`] made.
#[derive(Default)]
pub(super) struct DisplayOut {
    /// The RGBA8 frame, when the job ran the tail.
    pub rgba: Option<Rgba8>,
    /// The resident display-linear frame, when the job was asked to keep it.
    pub resident: Option<DeviceDisplay>,
    /// The display-linear frame read back (dense planar, `planes` planes), when the job had no tail.
    pub planar: Option<(usize, Vec<f32>)>,
}

/// Runs a [`DisplayJob`].
pub(super) fn display_job(
    ctx: &GpuContext,
    token: &CancelToken,
    job: &DisplayJob,
) -> Result<DisplayOut, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "view display");
    let sampled = input_image(ctx, &mut passes, &job.input)?;
    let prepared = PreparedChain::new(ctx, &job.chain)?;
    let (w, h) = (sampled.width, sampled.height);
    let display = GpuImage::alloc(
        ctx,
        "view display",
        w,
        h,
        sampled.planes,
        Placement::whole(w, h),
    )?;
    prepared.encode_stage(
        &mut passes,
        &sampled,
        Rect::whole(w, h),
        Stop::C12,
        &display,
        (0, 0),
    )?;
    let tail_read = job
        .tail
        .as_ref()
        .map(|tail| {
            record_tail(
                ctx,
                &mut passes,
                &display,
                job.bounds,
                tail,
                !keeps(job.output),
            )
        })
        .transpose()?;
    let planar_read = if job.tail.is_none() {
        Some(display.encode_read(&mut passes)?)
    } else {
        None
    };
    token.check()?;
    let mut done = passes.submit()?.finish()?;
    let mut out = DisplayOut::default();
    if let Some((dst, read)) = tail_read {
        out.rgba = Some(match read {
            Some(read) => Rgba8::Host(done.take(read)),
            None => Rgba8::Device(dst),
        });
    }
    if let Some(read) = planar_read {
        let planes = display.planes;
        out.planar = Some((planes, display.take(&done, read)?));
    }
    if job.keep {
        out.resident = Some(DeviceDisplay {
            image: display,
            bounds: job.bounds,
            gpu: job.gpu.clone(),
        });
    }
    Ok(out)
}

/// The tail over a resident display: the slider's job.
pub(super) struct TailJob {
    /// The resident display-linear frame.
    pub display: Arc<DeviceDisplay>,
    /// The tail.
    pub tail: TailWork,
    /// Where the RGBA8 frame goes.
    pub output: FrameOutput,
}

/// Runs a [`TailJob`].
pub(super) fn tail_job(
    ctx: &GpuContext,
    token: &CancelToken,
    job: &TailJob,
) -> Result<Rgba8, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "view tail");
    let (dst, read) = record_tail(
        ctx,
        &mut passes,
        &job.display.image,
        job.display.bounds,
        &job.tail,
        !keeps(job.output),
    )?;
    token.check()?;
    finish_rgba(passes, dst, read)
}

/// The bands of a view cut into bands, put together on the device: one RGBA8 buffer of `bytes`
/// bytes, the bands' rows one after the other (a band is whole rows of the view, so it is one
/// contiguous piece of the frame).
pub(super) struct GatherJob {
    /// The bands, top to bottom: the device's own, and the host's (a band that is nothing but
    /// paper is painted on the host).
    pub parts: Vec<Rgba8>,
    /// The size of the frame.
    pub bytes: u64,
}

/// Runs a [`GatherJob`]: a buffer copy per band the device made, an upload per band the host made.
/// The copies move the bytes and change none.
pub(super) fn gather_job(
    ctx: &GpuContext,
    token: &CancelToken,
    job: &GatherJob,
) -> Result<Rgba8, GpuError> {
    token.check()?;
    let total: u64 = job.parts.iter().map(Rgba8::byte_len).sum();
    if total != job.bytes {
        return Err(GpuError::InvalidInput(format!(
            "the bands hold {total} bytes, the frame {}",
            job.bytes
        )));
    }
    let mut passes = PassList::new(ctx, "view bands");
    let dst = passes.keep(ctx.pool().acquire("view rgba8", job.bytes, IMAGE_USAGE)?);
    let mut at = 0;
    for part in &job.parts {
        match part {
            Rgba8::Device(band) => {
                passes
                    .encoder()
                    .copy_buffer_to_buffer(band, 0, &dst, at, band.size());
            }
            Rgba8::Host(bytes) => upload_bytes(ctx, &dst, at, bytes)?,
        }
        at += part.byte_len();
    }
    token.check()?;
    finish_rgba(passes, dst, None)
}

/// Reads an RGBA8 frame the device kept back to the host (`DeviceFrame::read`).
pub(crate) fn read_job(
    ctx: &GpuContext,
    token: &CancelToken,
    buffer: &Arc<PooledBuffer>,
) -> Result<Vec<u8>, GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "device frame readback");
    let read = passes.read(buffer, 0, buffer.size())?;
    let mut done = passes.submit()?.finish()?;
    Ok(done.take(read))
}

/// The sampler alone: the sampled picture back on the host (the colour pass is on the CPU, or the
/// histogram wants the pixels).
pub(super) fn sample_job(
    ctx: &GpuContext,
    token: &CancelToken,
    input: &DeviceInput,
) -> Result<(usize, usize, Vec<f32>), GpuError> {
    token.check()?;
    let mut passes = PassList::new(ctx, "view sample");
    let sampled = input_image(ctx, &mut passes, input)?;
    let read = sampled.encode_read(&mut passes)?;
    token.check()?;
    let done = passes.submit()?.finish()?;
    Ok((sampled.width, sampled.height, sampled.take(&done, read)?))
}

// ---------------------------------------------------------------------------------------------
// Host conversions (render thread)
// ---------------------------------------------------------------------------------------------

/// The picture a sampler job read back as a host-sampled image: three planes and the mask plane.
pub(super) fn sampled_of(width: usize, height: usize, mut planar: Vec<f32>) -> Sampled {
    let n = width * height;
    let mask = planar.split_off(3 * n);
    let outside: Vec<u8> = mask.iter().map(|&m| u8::from(m != 0.0)).collect();
    let any = outside.iter().any(|&o| o != 0);
    Sampled {
        // A job that read back the sampler's output has `3 * n` colour samples by construction.
        img: RgbImage {
            width,
            height,
            place: Placement::whole(width, height),
            data: planar,
        },
        outside: any.then_some(outside),
    }
}

/// A display-linear frame read back as interleaved f32 (zero outside the photo, the colour kernel
/// writes zeros there) and the mask of the pixels outside.
pub(super) fn interleave_display(
    width: usize,
    height: usize,
    planes: usize,
    planar: &[f32],
) -> (Vec<f32>, Option<Vec<u8>>) {
    let n = width * height;
    let (r, g, b) = (&planar[..n], &planar[n..2 * n], &planar[2 * n..3 * n]);
    let mut rgb = vec![0.0f32; 3 * n];
    rgb.par_chunks_mut(3).enumerate().for_each(|(i, d)| {
        d[0] = r[i];
        d[1] = g[i];
        d[2] = b[i];
    });
    let outside = (planes == 4).then(|| {
        let mask = &planar[3 * n..4 * n];
        mask.iter()
            .map(|&m| u8::from(m != 0.0))
            .collect::<Vec<u8>>()
    });
    let any = outside.as_ref().is_some_and(|o| o.iter().any(|&v| v != 0));
    (rgb, outside.filter(|_| any))
}

/// Rows of the view per band so that the four-plane frame of a band, grown by `halo` rows above and
/// below, fits `band_bytes` of one storage binding, and how many bands that makes for `view_h` rows;
/// `None` when the whole frame fits (no cut). The rule depends on the sizes alone, never on the device.
pub(super) fn band_rows(
    view_w: usize,
    view_h: usize,
    halo: usize,
    band_bytes: u64,
    min_bands: usize,
) -> Option<usize> {
    let row_bytes = 16 * (view_w + 2 * halo).next_multiple_of(2) as u64;
    let fit_rows = (band_bytes / row_bytes.max(1)) as usize;
    let core_rows = fit_rows.saturating_sub(2 * halo).max(1);
    let by_size = view_h.div_ceil(core_rows);
    let bands = by_size.max(min_bands).min(view_h.max(1));
    (bands > 1).then(|| view_h.div_ceil(bands))
}

/// The view `spec` cut into horizontal bands of `rows` view rows: each band is a view of its own
/// (its own pixels and its own grown frame), and the dither of a band counts its texels from the
/// band's first row in the whole view.
pub(super) fn bands_of(spec: &ViewSpec, rows: usize) -> Vec<ViewSpec> {
    use crate::pipeline::geometry::Affine;
    let mut bands = Vec::new();
    let mut y = 0;
    while y < spec.out_h {
        let h = rows.min(spec.out_h - y);
        bands.push(ViewSpec {
            out_h: h,
            view_to_final: Affine::scale_translate(1.0, 1.0, 0.0, y as f64)
                .then(&spec.view_to_final),
            dither_origin: (spec.dither_origin.0, spec.dither_origin.1 + y),
            ..spec.clone()
        });
        y += h;
    }
    bands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_stages_follow_the_members_and_the_tail_needs_all_three() {
        let none = ViewStages::of(GpuStages::NONE);
        assert_eq!(
            none,
            ViewStages {
                sampler: false,
                color: false,
                tail: false
            }
        );
        let colour = ViewStages::of(GpuStages::COLOR);
        assert!(colour.color && !colour.sampler && !colour.tail);
        let sampler = ViewStages::of(GpuStages::SAMPLER);
        assert!(sampler.sampler && !sampler.color);
        // Sharpening and C13 are one device pass: one of them alone does nothing, and without the
        // colour pass the frame is not on the device for them to read.
        for stages in [
            GpuStages::COLOR.union(GpuStages::OUTPUT_SHARPEN),
            GpuStages::COLOR.union(GpuStages::ENCODE),
            GpuStages::OUTPUT_SHARPEN.union(GpuStages::ENCODE),
        ] {
            assert!(!ViewStages::of(stages).tail, "{stages}");
        }
        let all = ViewStages::of(GpuStages::ALL);
        assert!(all.sampler && all.color && all.tail);
        // The raw chain and the export tail are not view stages.
        assert_eq!(
            ViewStages::of(GpuStages::RAW_CHAIN.union(GpuStages::EXPORT_TAIL)),
            none
        );
    }

    #[test]
    fn a_frame_that_fits_one_binding_is_not_cut() {
        // 128 MiB holds 8,388,608 pixels of four f32 planes.
        assert_eq!(band_rows(2880, 1920, 17, 128 << 20, 1), None);
        assert_eq!(band_rows(4000, 2000, 0, 128 << 20, 1), None);
        // A 5K frame does not.
        let rows = band_rows(5120, 2880, 17, 128 << 20, 1).expect("cut");
        assert!(rows < 2880);
        // Every band's grown frame fits.
        let grown_bytes = 16 * (5120u64 + 34) * (rows as u64 + 34);
        assert!(grown_bytes <= 128 << 20, "{grown_bytes}");
        // The source can ask for more bands than the frame does.
        assert_eq!(band_rows(100, 100, 0, 128 << 20, 4), Some(25));
        // The cut never makes a band of no rows, however small the limit.
        assert_eq!(band_rows(64, 5, 3, 1, 1), Some(1));
    }

    #[test]
    fn bands_cover_the_view_once_and_keep_its_pixels_and_dither_places() {
        use crate::pipeline::geometry::Affine;
        use ym_color::display::OutputTarget;
        let spec = ViewSpec {
            out_w: 40,
            out_h: 23,
            view_to_final: Affine::scale_translate(2.0, 2.0, 10.0, 20.0),
            k: 2,
            target: OutputTarget::Viewport,
            before: false,
            coarse_only: false,
            cache: true,
            background: [255, 255, 255],
            gpu_priority: Priority::Viewport,
            dither_origin: (0, 0),
            output: FrameOutput::Device,
        };
        let bands = bands_of(&spec, 8);
        assert_eq!(bands.iter().map(|b| b.out_h).collect::<Vec<_>>(), [8, 8, 7]);
        let mut row = 0;
        for b in &bands {
            assert_eq!((b.out_w, b.k, b.cache), (40, 2, true));
            assert_eq!(
                b.output,
                FrameOutput::Device,
                "a band keeps the view's output"
            );
            assert_eq!(b.dither_origin, (0, row));
            // The band's pixel (x, y) is the view's pixel (x, y + row): same place in the picture.
            for (x, y) in [(0.5, 0.5), (39.5, b.out_h as f64 - 0.5)] {
                assert_eq!(
                    b.view_to_final.apply(x, y),
                    spec.view_to_final.apply(x, y + row as f64)
                );
            }
            row += b.out_h;
        }
        assert_eq!(row, spec.out_h);
    }

    #[test]
    fn a_readback_becomes_a_sampled_picture_and_a_host_display() {
        // 2 x 2: red 1..4, green 5..8, blue 9..12, the mask marks pixel 2.
        let mut planar: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        planar.extend([0.0, 0.0, 1.0, 0.0]);
        let s = sampled_of(2, 2, planar.clone());
        assert_eq!(s.img.data, (1..=12).map(|v| v as f32).collect::<Vec<_>>());
        assert_eq!(s.outside, Some(vec![0, 0, 1, 0]));
        let (rgb, outside) = interleave_display(2, 2, 4, &planar);
        assert_eq!(&rgb[..6], &[1.0, 5.0, 9.0, 2.0, 6.0, 10.0]);
        assert_eq!(outside, Some(vec![0, 0, 1, 0]));
        // No pixel outside: no mask.
        planar[12..].fill(0.0);
        assert_eq!(sampled_of(2, 2, planar.clone()).outside, None);
        assert_eq!(interleave_display(2, 2, 4, &planar).1, None);
        // Three planes: no mask either.
        assert_eq!(interleave_display(2, 2, 3, &planar[..12]).1, None);
    }
}
