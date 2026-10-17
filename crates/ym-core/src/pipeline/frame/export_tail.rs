//! The export tail on the device, in the render pipeline (`docs/17-gpu.md` section 10).
//!
//! # What it is
//!
//! A JPEG, TIFF or PNG export used to render the whole picture into one host frame of f32 (600 MB at
//! 50 MP), colour it tile by tile on the CPU and hand it to a writer that resized, sharpened and
//! encoded it. With [`GpuStages::EXPORT_TAIL`] on, the render never makes that frame: the raw chain
//! runs tile by tile as before (on the device too, when its stages are on), and each tile's camera
//! RGB goes to the device, through the sampler and the colour chain into a slab, and from the slabs
//! through the resize, output sharpening and C13 as encoded **bands** that the CPU writers pull
//! ([`GpuBands`] is a [`BandSource`]). The device work is `ym_gpu::tail::ExportTail`; this module
//! drives it from the render, hands its bands to the writer through a bounded channel, and decides
//! what happens when the device fails.
//!
//! # Threads
//!
//! [`Renderer::start_export_tail`] queues the render on the export worker, as `spawn_full` does. The
//! job renders tile rows and sends the bands to a channel of [`BAND_DEPTH`] places; the caller's
//! thread (the export queue's, or `ym`'s) pulls them through [`GpuBands`] while the writer
//! compresses and writes, so the render of the rows that follow overlaps the encoding of the rows
//! before. A writer that stops (an error, a cancel) drops the bands and the job ends at its next
//! tile; [`GpuBands::finish`] waits for it, so nothing outlives the export call.
//!
//! # Fall back
//!
//! A device error, in the tail or in the raw chain of a tile, ends the attempt with
//! [`RenderError::Gpu`]: the writer has seen some of the rows, so the attempt cannot go on in
//! another way, and the export (`crate::export`) makes the whole photo again on the CPU reference
//! path ([`Renderer::spawn_full_cpu`]: the same render with no device at all), with a note. A
//! strict policy fails the photo instead. The DNG's renders take the same road for their colour
//! stage ([`Renderer::render_full_with`]).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::time::Duration;

use ym_color::display::OutputTarget;
use ym_export::pixels::BandSource;
use ym_export::{BitDepth, ColorSpace, ExportError as WriteError, Size};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::{Gpu, GpuError};
use ym_gpu::queue::{CancelToken, JobError, JobHandle, JobSpec, Priority};
use ym_gpu::tail::{Band, BandData, BandFormat, ExportTail, TailRoute, TailSharpen, TailSpec};

use super::{FullImage, FullRequest, FullTarget};
use crate::app::gpu::GpuStages;
use crate::pipeline::color::PhotoSource;
use crate::pipeline::graph::{ComputeDevice, Job, RenderReport, Renderer, TileImage};
use crate::pipeline::{RenderError, graph::RenderOptions};

/// Bands the render may have made ahead of the writer: the depth of the channel between them. Each
/// is at most a few million pixels of codes (a megabyte or ten), so the host holds a few tens of
/// megabytes of finished picture at the most.
pub const BAND_DEPTH: usize = 3;

/// How often a waiting side looks at its cancel token.
const POLL: Duration = Duration::from_millis(5);

/// What the export tail makes of the picture: the settings of the file, as the writers take them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailRequest {
    /// The primaries C12 produces (the export's colour space).
    pub target: OutputTarget,
    /// The colour space of the file (its transfer function is C13's).
    pub space: ColorSpace,
    /// The size of the written picture (`Size::Full` or a long edge; never upscaled).
    pub size: Size,
    /// Output sharpening, percent (`DevelopSettings::output_sharpening`); 0 is none.
    pub output_sharpening_pct: f32,
    /// 8 bits (dithered) or 16.
    pub bits: BitDepth,
}

impl TailRequest {
    fn format(&self) -> BandFormat {
        match self.bits {
            BitDepth::Eight => BandFormat::Eight { dither: true },
            BitDepth::Sixteen => BandFormat::Sixteen,
        }
    }
}

/// What the render tells the writer before the first band.
struct Start {
    rendered: (usize, usize),
    out: (usize, usize),
}

enum Msg {
    Start(Start),
    Band(Band),
}

/// The bands of an export tail, as a [`BandSource`] (see the module notes).
///
/// Rows are handed out in order. The call that takes the last rows first runs the check given with
/// [`GpuBands::check_before_last_rows`], so that nothing is written under a look that changed while
/// the picture was rendering.
pub struct GpuBands<'a> {
    width: u32,
    height: u32,
    rendered: (u32, u32),
    sixteen: bool,
    rx: Option<Receiver<Msg>>,
    handle: Option<JobHandle<Result<RenderReport, RenderError>>>,
    /// Bands received and not yet handed out, the front one from row `offset` of it.
    pending: VecDeque<Band>,
    offset: usize,
    /// Rows handed out.
    next_row: usize,
    /// Rows received.
    received: usize,
    cancel: CancelToken,
    check: Option<Box<dyn FnMut() -> Result<(), RenderError> + 'a>>,
    failure: Option<RenderError>,
}

impl std::fmt::Debug for GpuBands<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBands")
            .field("size", &(self.width, self.height))
            .field("handed_out", &self.next_row)
            .field("received", &self.received)
            .finish_non_exhaustive()
    }
}

impl<'a> GpuBands<'a> {
    /// The size of the rendered picture (before the resize).
    #[must_use]
    pub fn rendered(&self) -> (u32, u32) {
        self.rendered
    }

    /// Runs `check` once, when the last rows are asked for and all of them have been made. An error
    /// of it ends the export (nothing is written) and is what [`GpuBands::take_failure`] returns.
    pub fn check_before_last_rows(&mut self, check: impl FnMut() -> Result<(), RenderError> + 'a) {
        self.check = Some(Box::new(check));
    }

    /// Why the last call of the [`BandSource`] failed, when the render (or the check) is the reason
    /// and not the writer: the error to report in place of the one the writer saw.
    pub fn take_failure(&mut self) -> Option<RenderError> {
        self.failure.take()
    }

    /// Waits for the render to end and returns its report. Cancels it first when it is still
    /// producing (the writer stopped early); drops the bands that were not taken.
    ///
    /// # Errors
    ///
    /// The render's error, [`RenderError::Cancelled`] for a render that was cancelled.
    pub fn finish(mut self) -> Result<RenderReport, RenderError> {
        self.join()
    }

    fn join(&mut self) -> Result<RenderReport, RenderError> {
        let done = self.received >= self.height as usize;
        // The channel's receiver goes first: a render blocked on a full channel then fails its send.
        self.rx = None;
        self.pending.clear();
        let Some(handle) = self.handle.take() else {
            return Err(RenderError::Image(
                "The export's render has already been collected.".to_owned(),
            ));
        };
        if !done {
            handle.cancel();
        }
        match handle.wait() {
            Ok(result) => result,
            Err(JobError::Cancelled) => Err(RenderError::Cancelled),
            Err(JobError::Shutdown) => Err(RenderError::Shutdown),
            Err(e) => Err(RenderError::Image(format!(
                "This photo can't be rendered: {e}."
            ))),
        }
    }

    /// Takes the end of a render that stopped sending: its error, or the complaint that it ended
    /// before the last row.
    fn render_ended(&mut self) -> RenderError {
        match self.join() {
            Ok(_) => RenderError::Image(
                "This photo can't be rendered: the render ended before the last row.".to_owned(),
            ),
            Err(e) => e,
        }
    }

    /// Brings `rows` rows from row `y0` into `pending`, waiting for the render when it is behind.
    fn make_available(&mut self, y0: usize, rows: usize, sixteen: bool) -> Result<(), WriteError> {
        if self.sixteen != sixteen {
            return Err(WriteError::invalid(if sixteen {
                "these bands are 8 bit, a 16-bit writer asked for them"
            } else {
                "these bands are 16 bit, an 8-bit writer asked for them"
            }));
        }
        if y0 != self.next_row {
            return Err(WriteError::invalid(format!(
                "the export bands are handed out in order: rows from {} were next, {y0} was asked for",
                self.next_row
            )));
        }
        let end = y0.checked_add(rows).filter(|&e| e <= self.height as usize);
        let Some(end) = end else {
            return Err(WriteError::invalid(format!(
                "rows {y0} to {} are outside the {} rows of the picture",
                y0.saturating_add(rows),
                self.height
            )));
        };
        while self.received < end {
            let Some(rx) = &self.rx else {
                return Err(WriteError::Cancelled);
            };
            match rx.recv_timeout(POLL) {
                Ok(Msg::Band(band)) => {
                    if band.y0 != self.received {
                        return Err(WriteError::invalid("the export bands came out of order"));
                    }
                    self.received += band.rows;
                    self.pending.push_back(band);
                }
                Ok(Msg::Start(_)) => {
                    return Err(WriteError::invalid("the export render started twice"));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.cancel.is_cancelled() {
                        if let Some(h) = &self.handle {
                            h.cancel();
                        }
                        return Err(WriteError::Cancelled);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // The render ended without all the rows: its error is the reason.
                    let e = self.render_ended();
                    let cancelled = matches!(e, RenderError::Cancelled);
                    self.failure = Some(e);
                    return Err(if cancelled {
                        WriteError::Cancelled
                    } else {
                        WriteError::encode("export", "the picture could not be rendered")
                    });
                }
            }
        }
        if end == self.height as usize
            && let Some(mut check) = self.check.take()
            && let Err(e) = check()
        {
            self.failure = Some(e);
            return Err(WriteError::encode(
                "export",
                "the picture changed while it was rendering",
            ));
        }
        Ok(())
    }

    /// Takes `rows` rows off the front of `pending` into `out` (the caller made them available).
    fn take<T: Copy>(
        &mut self,
        rows: usize,
        out: &mut Vec<T>,
        samples: impl Fn(&BandData) -> Option<&[T]>,
    ) -> Result<(), WriteError> {
        out.clear();
        let row_len = self.width as usize * 3;
        let mut left = rows;
        while left > 0 {
            let Some(band) = self.pending.front() else {
                return Err(WriteError::invalid("the export bands ran out of rows"));
            };
            let Some(data) = samples(&band.data) else {
                return Err(WriteError::invalid("the export bands have another depth"));
            };
            let n = left.min(band.rows - self.offset);
            out.extend_from_slice(&data[self.offset * row_len..(self.offset + n) * row_len]);
            self.offset += n;
            left -= n;
            if self.offset == band.rows {
                self.pending.pop_front();
                self.offset = 0;
            }
        }
        self.next_row += rows;
        Ok(())
    }
}

impl BandSource for GpuBands<'_> {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), WriteError> {
        self.make_available(y0, rows, false)?;
        self.take(rows, out, |d| match d {
            BandData::Eight(v) => Some(v.as_slice()),
            BandData::Sixteen(_) => None,
        })
    }

    fn rows_u16(&mut self, y0: usize, rows: usize, out: &mut Vec<u16>) -> Result<(), WriteError> {
        self.make_available(y0, rows, true)?;
        self.take(rows, out, |d| match d {
            BandData::Sixteen(v) => Some(v.as_slice()),
            BandData::Eight(_) => None,
        })
    }
}

impl Drop for GpuBands<'_> {
    fn drop(&mut self) {
        // An export that ends early drops its bands: the render stops at its next tile, and the
        // thread waits for it so that nothing it holds outlives the call.
        if self.handle.is_some() {
            let _ = self.join();
        }
    }
}

impl Renderer {
    /// Whether an export would run its tail on the device now: a device is attached, it is not
    /// lost, [`GpuStages::EXPORT_TAIL`] is on, and no shader of the tail has failed on this machine
    /// ([`Renderer::export_tail_disabled`]: the tail is then off for the rest of the process).
    #[must_use]
    pub fn export_tail_enabled(&self) -> bool {
        let o = self.options();
        o.gpu_stages.contains(GpuStages::EXPORT_TAIL)
            && o.gpu.as_ref().is_some_and(|g| !g.is_lost())
            && !self.export_tail_disabled()
    }

    /// The reason an export may not go on without the tail, when the policy is strict and an earlier
    /// shader failure switched the tail off: a strict policy never turns that into a quiet CPU
    /// export of the next photo (as for the raw chain, `Job::raw_device`). `None` otherwise.
    #[must_use]
    pub fn export_tail_refused(&self) -> Option<String> {
        let o = self.options();
        (o.gpu_strict
            && o.gpu.is_some()
            && o.gpu_stages.contains(GpuStages::EXPORT_TAIL)
            && self.export_tail_disabled())
        .then(|| {
            "an earlier error ended the use of the graphics card for exports, and the strict setting does not let the CPU take over"
                .to_owned()
        })
    }

    /// Queues the render of the written picture of a raster export on the export worker and waits
    /// until it knows the size of the picture; the bands then arrive as the writer asks for them
    /// ([`GpuBands`]). `cancel` is the export's own token: setting it stops the wait and the render.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` was set; [`RenderError::Gpu`] when the device failed
    /// before the first row (the caller makes the photo on the CPU); the errors of the render
    /// (memory, an unusable photo).
    pub fn start_export_tail<'a>(
        &self,
        src: PhotoSource,
        request: TailRequest,
        cancel: &CancelToken,
    ) -> Result<GpuBands<'a>, RenderError> {
        let (tx, rx) = mpsc::sync_channel(BAND_DEPTH);
        let r = self.clone();
        let id = src.id;
        let handle = self
            .scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                r.render_export_tail(&src, &request, token, &tx)
            });
        let mut bands = GpuBands {
            width: 0,
            height: 0,
            rendered: (0, 0),
            sixteen: request.bits == BitDepth::Sixteen,
            rx: Some(rx),
            handle: Some(handle),
            pending: VecDeque::new(),
            offset: 0,
            next_row: 0,
            received: 0,
            cancel: cancel.clone(),
            check: None,
            failure: None,
        };
        loop {
            if cancel.is_cancelled() {
                bands.failure = None;
                let _ = bands.join();
                return Err(RenderError::Cancelled);
            }
            let Some(rx) = &bands.rx else {
                return Err(RenderError::Cancelled);
            };
            match rx.recv_timeout(POLL) {
                Ok(Msg::Start(s)) => {
                    bands.width = s.out.0 as u32;
                    bands.height = s.out.1 as u32;
                    bands.rendered = (s.rendered.0 as u32, s.rendered.1 as u32);
                    return Ok(bands);
                }
                Ok(Msg::Band(_)) => {
                    return Err(RenderError::Image(
                        "This photo can't be rendered: the render sent rows before its size."
                            .to_owned(),
                    ));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(bands.render_ended()),
            }
        }
    }

    /// Queues the CPU reference render of an export: `Renderer::render_full_with` with no device
    /// at all, on the export worker. What an export falls back to after a device error.
    #[must_use]
    pub fn spawn_full_cpu(
        &self,
        src: PhotoSource,
        request: FullRequest,
        progress: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> JobHandle<Result<FullImage, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Export, Some(id), move |token| {
                match r.render_full_on(&src, &request, token, true, Some(&*progress), true)? {
                    super::Attempt::Done(image) => Ok(*image),
                    super::Attempt::DeviceFailed(reason) => Err(RenderError::Gpu(reason)),
                }
            })
    }

    /// The options of the CPU reference render: these, with no device and no stage on one.
    pub(super) fn cpu_reference_options(&self) -> Arc<RenderOptions> {
        let mut o = (*self.options()).clone();
        o.gpu = None;
        o.gpu_stages = GpuStages::NONE;
        Arc::new(o)
    }

    /// The render of the export tail, on the export worker: tile rows through the raw chain, tiles
    /// to the device, bands to the writer.
    fn render_export_tail(
        &self,
        src: &PhotoSource,
        request: &TailRequest,
        cancel: &CancelToken,
        tx: &SyncSender<Msg>,
    ) -> Result<RenderReport, RenderError> {
        let gate = || -> Result<(), RenderError> {
            self.scheduler()
                .wait_for_foreground(cancel)
                .map_err(|_| RenderError::Cancelled)
        };
        gate()?;
        let mut job = Job::for_export(self, src, cancel);
        let opts = job.options();
        let (Some(gpu), true) = (
            opts.gpu.clone(),
            opts.gpu_stages.contains(GpuStages::EXPORT_TAIL),
        ) else {
            return Err(RenderError::Gpu(
                "the export tail is not on the graphics card".to_owned(),
            ));
        };
        if gpu.is_lost() {
            return Err(RenderError::Gpu(
                gpu.lost_reason()
                    .unwrap_or_else(|| "the device was lost".to_owned()),
            ));
        }
        let tile = opts.full_tile_px.max(64);
        let plan = job.plan()?;
        let baked = job.geometry().at_full_resolution();
        let (fw, fh) = baked.output_size(plan.out_w, plan.out_h);
        let (ow, oh) = request.size.output_dimensions(fw as u32, fh as u32);
        let out = (ow as usize, oh as usize);
        // The reservation is the working set of one tile, not the picture: the frame is on the
        // device, a slab at a time (the CPU path reserves `fw * fh * 12` bytes for its host frame).
        let halo_allowance = 160usize;
        let per_px = 12u64 * 12; // twelve f32 planes of working set per tile pixel, a generous bound
        let working = ((tile + halo_allowance) as u64).pow(2) * per_px;
        let frame_extra = job.frame()?.extra_bytes() as u64;
        let sigma_map = if plan.reduced {
            0
        } else {
            (plan.width * plan.height * 4) as u64
        };
        let _reservation = self
            .ledger()
            .try_reserve(working + frame_extra + sigma_map)
            .map_err(|_| RenderError::Memory)?;
        let grade = job.grade(request.target, false)?;
        let grid = baked.grid((plan.out_w, plan.out_h), (fw, fh));
        let sharpen = (request.output_sharpening_pct > 0.0).then(|| TailSharpen {
            params: OutputSharpen::screen().with_amount_pct(request.output_sharpening_pct),
            scale: scale_for_edges(out.0.max(out.1), fw.max(fh)),
        });
        let route = TailRoute::new(
            &gpu,
            JobSpec::new(Priority::Export).owned_by(u64::from(src.id.0)),
            cancel.clone(),
        );
        let mut tail = ExportTail::new(
            route,
            TailSpec {
                rendered: (fw, fh),
                out,
                chain: Arc::clone(&grade.chain),
                grid: grid.sampler_grid(),
                sharpen,
                space: request.space.output_space(),
                format: request.format(),
            },
        )
        .map_err(|e| self.tail_error(&e))?;
        send(
            tx,
            Msg::Start(Start {
                rendered: (fw, fh),
                out,
            }),
            cancel,
        )?;
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
                    let region = match job.region_resident(rect) {
                        Ok(region) => region,
                        Err(RenderError::Gpu(reason)) => {
                            // The raw chain failed on the device; the photo goes to the CPU, all of
                            // it (the writer has seen rows already).
                            job.take_gpu_restart();
                            return Err(RenderError::Gpu(reason));
                        }
                        Err(e) => return Err(e),
                    };
                    // A tile the raw chain made on the device goes to the sampler where it is; one
                    // the CPU made (or the device made behind the CPU's front) is uploaded.
                    match region {
                        TileImage::Device(image) => {
                            tail.add_tile_device(image, (x0, y0), (ox, oy), tw, th)
                        }
                        TileImage::Host(image) => {
                            tail.add_tile(Arc::new(image), (x0, y0), (ox, oy), tw, th)
                        }
                    }
                    .map_err(|e| self.tail_error(&e))?;
                }
                ox += tw;
            }
            let bands = tail.advance(oy + th).map_err(|e| self.tail_error(&e))?;
            for band in bands {
                send(tx, Msg::Band(band), cancel)?;
            }
            oy += th;
        }
        for band in tail.finish().map_err(|e| self.tail_error(&e))? {
            send(tx, Msg::Band(band), cancel)?;
        }
        Ok(job.finish(1, ComputeDevice::Gpu, request.target))
    }
}

impl Renderer {
    /// A device error of the tail as the render's: a cancel stays a cancel; anything else is
    /// [`RenderError::Gpu`], which the export answers with the CPU (or, for a strict policy, with the
    /// failure of the photo). A shader that does not compile also switches the tail off for the
    /// process ([`Renderer::note_export_tail_error`]).
    pub(super) fn tail_error(&self, e: &GpuError) -> RenderError {
        match e {
            GpuError::Cancelled => RenderError::Cancelled,
            other => {
                tracing::warn!(error = %other, "the device failed in the export tail");
                self.note_export_tail_error(other);
                RenderError::Gpu(other.to_string())
            }
        }
    }
}

/// Sends a message to the writer's side, giving up when the writer is gone or the render was
/// cancelled.
fn send(tx: &SyncSender<Msg>, mut msg: Msg, cancel: &CancelToken) -> Result<(), RenderError> {
    loop {
        match tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Disconnected(_)) => return Err(RenderError::Cancelled),
            Err(TrySendError::Full(back)) => {
                if cancel.is_cancelled() {
                    return Err(RenderError::Cancelled);
                }
                msg = back;
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

/// The route of the export tail for the colour stage of a DNG render, or `None` when the CPU makes
/// it: no device, the stage is off, or the device is lost.
///
/// # Errors
///
/// [`RenderError::Gpu`] when the device is lost and the policy is strict.
pub(super) fn dng_colour_gpu(job: &mut Job<'_>) -> Result<Option<Arc<Gpu>>, RenderError> {
    let opts = job.options();
    if !opts.gpu_stages.contains(GpuStages::EXPORT_TAIL) {
        return Ok(None);
    }
    let Some(gpu) = opts.gpu.clone() else {
        return Ok(None);
    };
    if job.r.export_tail_disabled() {
        // A shader of the tail failed on this machine earlier: the CPU makes the colour stage, and
        // a strict policy fails the render instead of doing it quietly.
        if opts.gpu_strict {
            return Err(RenderError::Gpu(
                "an earlier error ended the use of the graphics card for exports, and the strict setting does not let the CPU take over"
                    .to_owned(),
            ));
        }
        return Ok(None);
    }
    if gpu.is_lost() {
        let why = gpu
            .lost_reason()
            .unwrap_or_else(|| "the device was lost".to_owned());
        if opts.gpu_strict {
            return Err(RenderError::Gpu(why));
        }
        job.notes.push(format!(
            "The graphics card stopped working, so the CPU did: {why}."
        ));
        return Ok(None);
    }
    Ok(Some(gpu))
}

/// Whether the FullTarget is one of the colour stages the tail runs for a host frame.
pub(super) fn host_stop(target: FullTarget) -> ym_gpu::kernels::color::Stop {
    match target {
        FullTarget::DngLinearAwg4 => ym_gpu::kernels::color::Stop::DngLinearAwg4,
        FullTarget::DisplayLinear(_) | FullTarget::DngPreview => ym_gpu::kernels::color::Stop::C12,
    }
}

/// An image in an `Arc` for the jobs, from a region the raw chain made.
pub(super) fn shared(region: RgbImage) -> Arc<RgbImage> {
    Arc::new(region)
}
