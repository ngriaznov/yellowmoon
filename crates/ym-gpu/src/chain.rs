//! The device chain: stages of the raw-processing chain (`docs/05`) run on the wgpu device, the rest
//! on the CPU stepper, for one tile at a time (`docs/gpu-plan.md` sections 4.3 and 11, task G-03).
//!
//! [`GpuChain`] is built from the very `Chain` and `Prepared` the CPU uses, and walks the same
//! stages with the same rectangles ([`TileRun`](crate::cpu::chain::TileRun)), so the settings, the stage choice and the halos are
//! single-sourced. A [`StageRange`] says which stages run on the device; every stage outside it runs
//! on the CPU stepper and the boundary is a host copy. There are two ranges so far:
//!
//! * R6 ([`StageRange::DEMOSAIC`], G-03): the CPU runs R1 to R5b exactly as today and hands the
//!   balanced mosaic over, the device runs the adaptive ratio demosaic (including the R4 undo, the CA
//!   path and the shadow colour matching of the kernel), the planes come back as an [`RgbImage`] and
//!   the CPU continues with R6's bilateral denoise, R7, R8 and R9.
//! * R6 to R8 ([`StageRange::DEMOSAIC_TO_SHARPEN`], G-06): the same hand-off, and the planes stay on
//!   the device through the bilateral denoise, the R7 colour guided filter and detail re-injection
//!   and R8 (Richardson-Lucy, or diffraction deconvolution) as one job with one readback; the CPU
//!   runs R9 on what comes back. The noise sigma of the stages is expanded on the device from the
//!   photo's half-resolution plane ([`GpuChain::with_sigma`]), so the 200 MB full-resolution map of the
//!   CPU chain is never built for this range.
//!
//! The later tasks widen the range (G-07 to R1..R9) by adding variants of [`DeviceWork`].
//!
//! # Where the work runs
//!
//! A tile is stepped on the calling thread (the render thread), which does the CPU stages; only the
//! device work is submitted, as one self-contained [`DeviceWork`] value to a [`DeviceRunner`]. The
//! runner of a [`GpuContext`] runs it at once (tests, tools); the application's runner submits it as
//! one job to the device thread with the render's own cancel token and priority
//! (`Gpu::run_with_token`), so the CPU stages of a tile never occupy the device thread and a viewport
//! frame can run between two tiles of an export.
//!
//! # Splitting a big rectangle
//!
//! The adaptive ratio demosaic needs 48 bytes of scratch per padded pixel in one storage binding, and
//! the design binding is 128 MiB (the size lavapipe reports; `docs/gpu-plan.md` N-1). A core whose
//! padded plane ([`adaptive_ratio::padded_pixels`]) is above [`MAX_PHYS_AREA`] is cut in halves
//! along its longer side, recursively, each half computed from its own rectangle with the full halo
//! ([`split_core`]). The rule reads the rectangle only, never the device, so the application and `ym`
//! cut alike. The kernel is tile-exact, so a split changes no bit.

use crate::cpu::chain::{Chain, Prepared, RawFrame};
use crate::cpu::demosaic::Method;
use crate::cpu::demosaic::adaptive_ratio::Params;
use crate::cpu::denoise::BilateralParams;
use crate::cpu::detail::network_sigma_scale;
use crate::cpu::sharpen::CaptureSharpen;
use crate::cpu::sharpen::diffraction::{DiffractionPlan, SPEC_LUMA};
use crate::cpu::tiling::{Halo, Rect, Stage};
use crate::cpu::{Mosaic, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::adaptive_ratio::{self, Balance, PreparedDemosaic};
use crate::kernels::bilateral::{self, SigmaHalf, SigmaInput, SigmaTile};
use crate::kernels::detail::{encode_classical_luma, encode_inject_detail};
use crate::kernels::diffraction::{PreparedDiffraction, encode_sharpen};
use crate::kernels::guided::encode_guided;
use crate::kernels::richardson_lucy::{RlPlan, encode_richardson_lucy};
use crate::queue::CancelToken;
use crate::stream::PassList;

/// The most padded pixels one device rectangle has: 2.5 Mpixel (`docs/gpu-plan.md` section 4.4). The
/// adaptive ratio scratch is 12 planes of 4 bytes with shadow colour matching, 48 bytes per pixel, so
/// 2.5 Mpixel are 120 MB in one binding, under the 128 MiB design binding
/// ([`crate::tiles::DESIGN_BINDING_BYTES`]) of lavapipe. A constant and not a device limit, so that
/// the application and `ym` cut the same way on every machine.
pub const MAX_PHYS_AREA: u64 = 2_500_000;

/// A stage of the raw-processing chain, named as `docs/05` numbers them. The order is the order of
/// processing, so a [`StageRange`] is a slice of this list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RawStage {
    /// R1 normalise.
    Normalize,
    /// R3 vignetting.
    Vignette,
    /// R4 gains.
    Gains,
    /// R5 highlight reconstruction.
    Highlights,
    /// R5b lateral CA on the mosaic.
    MosaicCa,
    /// R6 demosaic.
    Demosaic,
    /// R6 bilateral denoise.
    Denoise,
    /// R7 colour guided filter and detail re-injection.
    Detail,
    /// R8 capture sharpening.
    Sharpen,
    /// R9 distortion resample.
    Lens,
}

impl RawStage {
    /// Every stage, in processing order.
    pub const ALL: [RawStage; 10] = [
        RawStage::Normalize,
        RawStage::Vignette,
        RawStage::Gains,
        RawStage::Highlights,
        RawStage::MosaicCa,
        RawStage::Demosaic,
        RawStage::Denoise,
        RawStage::Detail,
        RawStage::Sharpen,
        RawStage::Lens,
    ];

    /// The stage of `docs/05` a stage of [`Chain::stages`] is.
    #[must_use]
    pub fn of(stage: &Stage<'_>) -> RawStage {
        match stage {
            Stage::Normalize => RawStage::Normalize,
            Stage::Vignette => RawStage::Vignette,
            Stage::Gains => RawStage::Gains,
            Stage::Highlights => RawStage::Highlights,
            Stage::MosaicCa { .. } => RawStage::MosaicCa,
            Stage::Demosaic(_) => RawStage::Demosaic,
            Stage::Denoise { .. } => RawStage::Denoise,
            Stage::Detail { .. } => RawStage::Detail,
            Stage::Sharpen(_) => RawStage::Sharpen,
            Stage::Lens(_) => RawStage::Lens,
        }
    }

    /// The name used in reports and notes (`R6`, `R5b`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            RawStage::Normalize => "R1",
            RawStage::Vignette => "R3",
            RawStage::Gains => "R4",
            RawStage::Highlights => "R5",
            RawStage::MosaicCa => "R5b",
            RawStage::Demosaic => "R6",
            RawStage::Denoise => "R6 denoise",
            RawStage::Detail => "R7",
            RawStage::Sharpen => "R8",
            RawStage::Lens => "R9",
        }
    }
}

/// The stages that run on the device, first to last (both included). The test and gating lever of
/// the plan: stages outside the range run on the CPU stepper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StageRange {
    /// The first stage on the device.
    pub first: RawStage,
    /// The last stage on the device.
    pub last: RawStage,
}

impl StageRange {
    /// R6, the adaptive ratio demosaic: the one range the device runs in this version.
    pub const DEMOSAIC: StageRange = StageRange {
        first: RawStage::Demosaic,
        last: RawStage::Demosaic,
    };

    /// R6 to R8 (G-06): the adaptive ratio demosaic, the bilateral denoise, R7 and capture sharpening,
    /// one device job with one readback.
    pub const DEMOSAIC_TO_SHARPEN: StageRange = StageRange {
        first: RawStage::Demosaic,
        last: RawStage::Sharpen,
    };

    /// Whether the stages of this range read the noise sigma on the device only: the denoise, R7 and
    /// R8 are all in it (or none of them is needed), so the chain's full-resolution sigma map is not
    /// read by any CPU stage and need not exist (`GpuChain::with_sigma`).
    #[must_use]
    pub fn covers_sigma_stages(&self) -> bool {
        self.first <= RawStage::Denoise && self.last >= RawStage::Sharpen
    }

    /// Whether `stage` is on the device.
    #[must_use]
    pub fn contains(&self, stage: RawStage) -> bool {
        self.first <= stage && stage <= self.last
    }

    /// `R6` or `R6..R8`, for reports.
    #[must_use]
    pub fn label(&self) -> String {
        if self.first == self.last {
            self.first.name().to_owned()
        } else {
            format!("{}..{}", self.first.name(), self.last.name())
        }
    }
}

/// Splits the core of a demosaic into rectangles the device can take: halves along the longer side,
/// recursively, until the padded plane of each is at most [`MAX_PHYS_AREA`]. A function of `core`
/// alone. The rectangles tile `core` exactly, in raster order of the splits (left or top half first).
#[must_use]
pub fn split_core(core: Rect) -> Vec<Rect> {
    split_core_above(core, MAX_PHYS_AREA)
}

/// [`split_core`] with another limit, for tests that exercise the rule on small frames.
#[must_use]
pub fn split_core_above(core: Rect, max_padded_pixels: u64) -> Vec<Rect> {
    let mut out = Vec::new();
    split_into(core, max_padded_pixels, &mut out);
    out
}

fn split_into(core: Rect, max_padded_pixels: u64, out: &mut Vec<Rect>) {
    if adaptive_ratio::padded_pixels(core) <= max_padded_pixels || (core.w < 2 && core.h < 2) {
        out.push(core);
        return;
    }
    // Cut at an even offset when there is room for one, so that the halves keep the Bayer phase of
    // the parent (the kernel does not need it; the planner of the CPU does the same).
    let half = |len: usize| {
        if len / 2 >= 2 {
            (len / 2) & !1
        } else {
            len / 2
        }
    };
    if core.w >= core.h {
        let a = half(core.w);
        split_into(Rect::new(core.x, core.y, a, core.h), max_padded_pixels, out);
        split_into(
            Rect::new(core.x + a, core.y, core.w - a, core.h),
            max_padded_pixels,
            out,
        );
    } else {
        let a = half(core.h);
        split_into(Rect::new(core.x, core.y, core.w, a), max_padded_pixels, out);
        split_into(
            Rect::new(core.x, core.y + a, core.w, core.h - a),
            max_padded_pixels,
            out,
        );
    }
}

/// The device half of a tile, as plain owned data: a job closure can hold it, move it to the device
/// thread and run it there. Built on the render thread by [`GpuChain::run_tile`].
#[derive(Debug)]
pub enum DeviceWork {
    /// R6: the adaptive ratio demosaic of a host mosaic.
    Demosaic(DemosaicWork),
    /// R6 to R8: the demosaic and the stages after it, on the device without a readback between
    /// them.
    Raw(RawWork),
}

impl DeviceWork {
    /// Runs the work on the device behind `ctx`. The token is checked at the start and between the
    /// rectangles of a split; a submitted rectangle cannot be stopped (cancel latency is one job).
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`], and whatever the kernel reports (device errors, a binding that does
    /// not fit, an injected fault of the `fault-injection` feature).
    pub fn run(self, ctx: &GpuContext, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        self.run_ref(ctx, cancel)
    }

    /// The same as [`run`](Self::run) for work held by reference (tests that run one piece twice).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_ref(&self, ctx: &GpuContext, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        match self {
            DeviceWork::Demosaic(w) => w.run(ctx, cancel),
            DeviceWork::Raw(w) => w.run(ctx, cancel),
        }
    }
}

/// R6 of one tile: the balanced mosaic of the compute rectangle, the part of it to demosaic and the
/// frame constants.
#[derive(Debug)]
pub struct DemosaicWork {
    mosaic: Mosaic,
    core: Rect,
    frame: PreparedDemosaic,
}

impl DemosaicWork {
    /// Checks the samples of `mosaic` (once, on the calling thread: the device thread should not
    /// spend its time on a scan) and wraps the work.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a mosaic the kernel refuses (below 2 x 2, a value that is not
    /// finite or beyond the kernel's range) or a core outside it.
    pub fn new(
        mosaic: Mosaic,
        core: Rect,
        frame: PreparedDemosaic,
    ) -> Result<DemosaicWork, GpuError> {
        frame.check_mosaic(&mosaic)?;
        if core.is_empty() || core.x1() > mosaic.width || core.y1() > mosaic.height {
            return Err(GpuError::InvalidInput(format!(
                "the core {core:?} is empty or outside the {} x {} mosaic",
                mosaic.width, mosaic.height
            )));
        }
        Ok(DemosaicWork {
            mosaic,
            core,
            frame,
        })
    }

    /// The part of the mosaic the work demosaics (the rest is context), in the mosaic's coordinates.
    #[must_use]
    pub fn core(&self) -> Rect {
        self.core
    }

    /// The rectangles the work is cut into ([`split_core`] of its core).
    #[must_use]
    pub fn parts(&self) -> Vec<Rect> {
        split_core(self.core)
    }

    /// Runs the work; a device that is out of memory or refuses a buffer is given one more try after
    /// the pool let go of what it holds unused (`docs/gpu-plan.md` 4.9: "trim the pool, retry once,
    /// then the CPU"; the integrator answers a second failure with the CPU chain).
    fn run(&self, ctx: &GpuContext, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        let parts = split_core(self.core);
        match self.run_split(ctx, cancel, &parts) {
            Err(GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }) => {
                ctx.pool().trim();
                self.run_split(ctx, cancel, &parts)
            }
            other => other,
        }
    }

    /// Runs the work cut into `parts`, which must tile the core ([`split_core_above`] gives such a
    /// list).
    fn run_split(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        parts: &[Rect],
    ) -> Result<RgbImage, GpuError> {
        cancel.check()?;
        ctx.inject_tile_fault()?;
        let DemosaicWork {
            mosaic,
            core,
            frame,
        } = self;
        let core = *core;
        if let [only] = parts
            && *only == core
        {
            return adaptive_ratio::demosaic_core(ctx, mosaic, core, frame);
        }
        let whole = Rect::whole(mosaic.width, mosaic.height);
        let halo = Halo::uniform(frame.halo_px());
        let mut out = RgbImage::new(core.w, core.h);
        out.place = mosaic.place.offset(core.x, core.y);
        for part in parts {
            cancel.check()?;
            let sub_rect = part.grow(halo, whole);
            let sub = mosaic.extract(sub_rect);
            let img = adaptive_ratio::demosaic_core(ctx, &sub, part.relative_to(&sub_rect), frame)?;
            out.paste(
                &img,
                Rect::whole(img.width, img.height),
                part.x - core.x,
                part.y - core.y,
            );
        }
        Ok(out)
    }

    /// Runs the work with the core cut at `max_padded_pixels` instead of [`MAX_PHYS_AREA`] (tests).
    ///
    /// # Errors
    ///
    /// As [`DeviceWork::run`].
    pub fn run_with_limit(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        max_padded_pixels: u64,
    ) -> Result<RgbImage, GpuError> {
        let parts = split_core_above(self.core, max_padded_pixels);
        self.run_split(ctx, cancel, &parts)
    }
}

/// Where the noise sigma of a tile comes from (the device makes the plane the stages read from it, see
/// [`bilateral::encode_sigma`]).
#[derive(Debug)]
pub enum SigmaWork {
    /// One value for the whole tile (the chain's scalar sigma).
    Scalar(f32),
    /// The tile's rectangle of a full-resolution map the host has (`width * height` of the rectangle).
    Plane(Vec<f32>),
    /// The cells of the half-resolution plane that the rectangle reads, expanded on the device.
    Cells(SigmaTile),
}

/// What the stages after the demosaic receive.
#[derive(Debug)]
enum RawInput {
    /// The balanced mosaic: the device runs the demosaic first.
    Mosaic(DemosaicWork),
    /// The demosaiced planes (a mosaic below 2 x 2 photosites was demosaiced on the CPU).
    Image(RgbImage),
}

/// The capture-sharpening method of a [`PostStep`].
#[derive(Debug)]
enum SharpenKind {
    /// Richardson-Lucy.
    RichardsonLucy(RlPlan),
    /// Diffraction deconvolution (an active plan only).
    Diffraction(DiffractionPlan),
}

/// One stage after the demosaic, with the rectangle (frame coordinates) it must leave.
#[derive(Debug)]
enum PostStep {
    /// R6 bilateral denoise; with `keep_classical` the luma of its input is kept for R7.
    Denoise {
        out: Rect,
        scale: f32,
        params: BilateralParams,
        keep_classical: bool,
    },
    /// R7: the colour guided filter (`color > 0`) and the detail re-injection (`beta`, when the
    /// denoise kept the classical luma).
    Detail {
        out: Rect,
        color: f32,
        beta: Option<f32>,
    },
    /// R8.
    Sharpen { out: Rect, kind: SharpenKind },
}

impl PostStep {
    fn out(&self) -> Rect {
        match self {
            PostStep::Denoise { out, .. }
            | PostStep::Detail { out, .. }
            | PostStep::Sharpen { out, .. } => *out,
        }
    }
}

/// R6 to R8 of one tile: the balanced mosaic (or the demosaiced planes), the part of it the stages
/// leave, the frame constants of the demosaic and the stages after it, with the noise sigma they read.
///
/// One job: the device demosaics, denoises, filters the colour, re-injects the detail and sharpens
/// without a readback between the stages, and the planes come back once. Every stage is the CPU
/// stage's kernel with the CPU's own rectangle for it (`TileRun::need`), so the result is the CPU
/// chain's within the tolerances of the kernels.
#[derive(Debug)]
pub struct RawWork {
    input: RawInput,
    /// The frame rectangle of the demosaic's output (the input of the first stage after it).
    start: Rect,
    /// The frame rectangle the last stage leaves (the input of R9, or the output of the tile).
    end: Rect,
    sigma: SigmaWork,
    steps: Vec<PostStep>,
}

impl RawWork {
    /// The dispatches the work records at most (informational: the demosaic's up to 14, the sigma
    /// plane, the denoise, the guided filter's 4 and 2 for the injection, Richardson-Lucy's `2 + 4 n`).
    #[must_use]
    pub fn max_dispatches(&self) -> usize {
        let mut n = match &self.input {
            RawInput::Mosaic(_) => 14,
            RawInput::Image(_) => 0,
        } + 1;
        for step in &self.steps {
            n += match step {
                PostStep::Denoise { keep_classical, .. } => 1 + usize::from(*keep_classical),
                PostStep::Detail { color, beta, .. } => {
                    4 * usize::from(*color > 0.0) + usize::from(beta.is_some())
                }
                PostStep::Sharpen { kind, .. } => match kind {
                    SharpenKind::RichardsonLucy(p) => p.dispatches(),
                    SharpenKind::Diffraction(_) => 11,
                },
            };
        }
        n
    }

    /// Runs the work; a device that is out of memory or refuses a buffer is given one more try after
    /// the pool let go of what it holds unused (`docs/gpu-plan.md` 4.9).
    fn run(&self, ctx: &GpuContext, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        match self.run_once(ctx, cancel, MAX_PHYS_AREA) {
            Err(GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }) => {
                ctx.pool().trim();
                self.run_once(ctx, cancel, MAX_PHYS_AREA)
            }
            other => other,
        }
    }

    /// Runs the work with the demosaic core cut at `max_padded_pixels` instead of [`MAX_PHYS_AREA`]
    /// (tests).
    ///
    /// # Errors
    ///
    /// As [`DeviceWork::run`].
    pub fn run_with_limit(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        max_padded_pixels: u64,
    ) -> Result<RgbImage, GpuError> {
        self.run_once(ctx, cancel, max_padded_pixels)
    }

    /// The rectangles the demosaic is cut into at `max_padded_pixels` (one for a work that starts
    /// from demosaiced planes).
    #[must_use]
    pub fn demosaic_parts(&self, max_padded_pixels: u64) -> usize {
        match &self.input {
            RawInput::Mosaic(d) => split_core_above(d.core, max_padded_pixels).len(),
            RawInput::Image(_) => 1,
        }
    }

    fn run_once(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        max_padded_pixels: u64,
    ) -> Result<RgbImage, GpuError> {
        cancel.check()?;
        ctx.inject_tile_fault()?;
        // Every buffer the list reads or writes stays alive until it has finished: a buffer that went
        // back to the pool could be handed out again by the next `acquire` of the same list, and the
        // poison fill (a queue write) of that second use runs before the whole submission.
        let mut keep: Vec<GpuImage> = Vec::new();
        let mut passes;
        let mut cur = match &self.input {
            RawInput::Mosaic(d) => {
                let parts = split_core_above(d.core, max_padded_pixels);
                if let [only] = parts.as_slice()
                    && *only == d.core
                {
                    let m = &d.mosaic;
                    let input = GpuImage::upload(
                        ctx,
                        "adaptive ratio mosaic",
                        m.width,
                        m.height,
                        1,
                        &m.data,
                        m.place,
                    )?;
                    passes = PassList::new(ctx, "raw R6 to R8");
                    let out =
                        adaptive_ratio::encode_demosaic(&mut passes, &input, m.cfa, d.core, &d.frame)?;
                    keep.push(input);
                    out
                } else {
                    // A core above MAX_PHYS_AREA is cut in rectangles whose planes are put together on
                    // the host and uploaded once more (a corner tile of a strong lens; the stages that
                    // follow need the whole core in one image).
                    let rgb = d.run_split(ctx, cancel, &parts)?;
                    passes = PassList::new(ctx, "raw R6 to R8");
                    GpuImage::upload_rgb(ctx, "adaptive ratio rgb", &rgb)?
                }
            }
            RawInput::Image(img) => {
                passes = PassList::new(ctx, "raw R6 to R8");
                GpuImage::upload_rgb(ctx, "demosaiced planes", img)?
            }
        };
        let mut cur_rect = self.start;
        if (cur.width, cur.height) != (cur_rect.w, cur_rect.h) {
            return Err(GpuError::InvalidInput(format!(
                "the demosaic left {} x {} pixels, the stages after it expect {:?}",
                cur.width, cur.height, cur_rect
            )));
        }
        cancel.check()?;
        let sigma_input = match &self.sigma {
            SigmaWork::Scalar(v) => SigmaInput::Scalar(*v),
            SigmaWork::Plane(p) => SigmaInput::Plane(p),
            SigmaWork::Cells(t) => SigmaInput::Tile(t),
        };
        let sigma = bilateral::encode_sigma(&mut passes, &sigma_input, self.start)?;
        // The luma of the image that went into the denoise, for R7's re-injection, with the frame
        // rectangle it covers.
        let mut classical: Option<(GpuImage, Rect)> = None;
        let origin_of = |inner: Rect, outer: Rect| -> Result<(usize, usize), GpuError> {
            if outer.contains(&inner) {
                Ok((inner.x - outer.x, inner.y - outer.y))
            } else {
                Err(GpuError::InvalidInput(format!(
                    "the stage's rectangle {inner:?} is not inside {outer:?}"
                )))
            }
        };
        for step in &self.steps {
            let out = step.out();
            if !cur_rect.contains(&out) {
                return Err(GpuError::InvalidInput(format!(
                    "a stage must leave {out:?}, the image it gets covers {cur_rect:?}"
                )));
            }
            // The part of the current image the stage leaves, and where the image sits in the sigma
            // plane. A stage that is an identity (no iteration, a zero colour strength with nothing to
            // re-inject) is not recorded at all and leaves the image as it was, larger than `out`: the
            // next stage names its core relative to the image it really gets.
            let core = out.relative_to(&cur_rect);
            let sigma_origin = origin_of(cur_rect, self.start)?;
            match step {
                PostStep::Denoise {
                    scale,
                    params,
                    keep_classical,
                    ..
                } => {
                    if *keep_classical {
                        let luma = encode_classical_luma(&mut passes, &cur, core)?;
                        classical = Some((luma, out));
                    }
                    let next = bilateral::encode_bilateral(
                        &mut passes,
                        &cur,
                        core,
                        &sigma,
                        sigma_origin,
                        *scale,
                        params,
                    )?;
                    keep.push(std::mem::replace(&mut cur, next));
                    cur_rect = out;
                }
                PostStep::Detail { color, beta, .. } => {
                    if *color > 0.0 {
                        let next =
                            encode_guided(&mut passes, &cur, core, &sigma, sigma_origin, *color)?;
                        keep.push(std::mem::replace(&mut cur, next));
                        cur_rect = out;
                    }
                    if let Some(beta) = beta {
                        let Some((luma, luma_rect)) = classical.as_ref() else {
                            return Err(GpuError::InvalidInput(
                                "the detail re-injection has no classical luma".into(),
                            ));
                        };
                        let inner = out.relative_to(&cur_rect);
                        let origin = origin_of(cur_rect, *luma_rect)?;
                        let next =
                            encode_inject_detail(&mut passes, &cur, inner, luma, origin, *beta)?;
                        keep.push(std::mem::replace(&mut cur, next));
                        cur_rect = out;
                    }
                }
                PostStep::Sharpen { kind, .. } => {
                    let next = match kind {
                        SharpenKind::RichardsonLucy(plan) => encode_richardson_lucy(
                            &mut passes,
                            plan,
                            &cur,
                            core,
                            &sigma,
                            sigma_origin,
                        )?,
                        SharpenKind::Diffraction(plan) => {
                            let prepared = PreparedDiffraction::new(ctx, plan)?;
                            encode_sharpen(&mut passes, &prepared, &cur, core)?
                        }
                    };
                    keep.push(std::mem::replace(&mut cur, next));
                    cur_rect = out;
                }
            }
        }
        cancel.check()?;
        let read = cur.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let data = cur.take(&done, read)?;
        drop((keep, sigma, classical));
        let mut rgb = RgbImage {
            width: cur.width,
            height: cur.height,
            place: cur.place,
            data,
        };
        if cur_rect != self.end {
            // Stages that were identities leave a larger image than the tile needs: cut it.
            if !cur_rect.contains(&self.end) {
                return Err(GpuError::InvalidInput(format!(
                    "the stages left {cur_rect:?}, the tile needs {:?}",
                    self.end
                )));
            }
            rgb = rgb.extract(self.end.relative_to(&cur_rect));
        }
        Ok(rgb)
    }
}

/// Where the device work of a tile runs. The context of a device runs it on the calling thread; the
/// application submits it to the device thread as one job.
pub trait DeviceRunner {
    /// Runs `work` and returns the host image it produces.
    ///
    /// # Errors
    ///
    /// As [`DeviceWork::run`], plus the scheduler's errors for a runner that submits a job.
    fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError>;
}

impl DeviceRunner for GpuContext {
    fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        work.run(self, cancel)
    }
}

/// The classical chain with some stages on the device (see the module notes).
#[derive(Debug)]
pub struct GpuChain<'a> {
    chain: &'a Chain,
    prepared: &'a Prepared,
    range: StageRange,
    sigma: Option<SigmaHalf<'a>>,
}

impl<'a> GpuChain<'a> {
    /// The device chain for `chain` and the whole-frame reductions `prepared` the CPU also uses.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a range the device does not run yet (only
    /// [`StageRange::DEMOSAIC`] and [`StageRange::DEMOSAIC_TO_SHARPEN`] in this version) or a chain
    /// whose R6 method is not the adaptive ratio demosaic (the other methods run on the CPU only).
    pub fn new(
        chain: &'a Chain,
        prepared: &'a Prepared,
        range: StageRange,
    ) -> Result<GpuChain<'a>, GpuError> {
        if range != StageRange::DEMOSAIC && range != StageRange::DEMOSAIC_TO_SHARPEN {
            return Err(GpuError::InvalidInput(format!(
                "the device chain runs the ranges {} and {}, not {}",
                StageRange::DEMOSAIC.label(),
                StageRange::DEMOSAIC_TO_SHARPEN.label(),
                range.label()
            )));
        }
        if chain.method != Method::AdaptiveRatio {
            return Err(GpuError::InvalidInput(format!(
                "the device runs the adaptive ratio demosaic only, the chain asks for {}",
                chain.method.name()
            )));
        }
        Ok(GpuChain {
            chain,
            prepared,
            range,
            sigma: None,
        })
    }

    /// Gives the device chain the photo's noise sigma as the half-resolution plane and the cell gains
    /// (`ym-core`'s `SigmaPlane` and `gain_plane`), instead of the chain's own full-resolution map
    /// (`Chain::sigma_map`) or scalar. The device expands the cells a tile reads with the rule of the
    /// CPU's `full_sigma_map`, so the chain's map need not exist when every stage that reads the sigma
    /// is on the device ([`StageRange::covers_sigma_stages`]); a range that leaves one on the CPU
    /// needs the map the CPU stepper reads.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a half plane of the wrong length, or a range that leaves a stage
    /// that reads the sigma on the CPU while the chain has no sigma map to give it.
    pub fn with_sigma(mut self, sigma: SigmaHalf<'a>) -> Result<GpuChain<'a>, GpuError> {
        sigma.validate()?;
        if !self.range.covers_sigma_stages() && self.chain.sigma_map.is_none() {
            return Err(GpuError::InvalidInput(format!(
                "the range {} leaves stages that read the noise sigma on the CPU, and the chain has no sigma map",
                self.range.label()
            )));
        }
        self.sigma = Some(sigma);
        Ok(self)
    }

    /// The stages on the device.
    #[must_use]
    pub fn range(&self) -> StageRange {
        self.range
    }

    /// Runs the part of the output covered by `out` (output coordinates, as
    /// [`Chain::run_tile`]) and returns exactly that rectangle: the same contract as the CPU chain.
    ///
    /// The CPU stepper runs the stages before the range, the device work goes to `runner`, the CPU
    /// stepper runs the stages after it. `cancel` is checked between the steps; the runner gets it
    /// for the device job.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`]; [`GpuError::Cpu`] for inputs a CPU stage refuses (the CPU chain
    /// would refuse them too); the runner's errors, which the integrator answers with the CPU chain.
    pub fn run_tile<R: DeviceRunner + ?Sized>(
        &self,
        runner: &R,
        raw: &RawFrame<'_>,
        out: Rect,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
        cancel.check()?;
        let mut run = self.chain.begin_tile(raw, self.prepared, out)?;
        let first = run
            .stages()
            .iter()
            .position(|s| RawStage::of(s) == RawStage::Demosaic)
            .ok_or_else(|| GpuError::InvalidInput("the chain has no demosaic stage".into()))?;
        run.run_until(first)?;
        cancel.check()?;
        let degenerate = run
            .demosaic_input()
            .is_some_and(|i| i.mosaic.width < 2 || i.mosaic.height < 2);
        if self.range == StageRange::DEMOSAIC_TO_SHARPEN {
            return self.run_raw_group(runner, raw, run, first, degenerate, cancel);
        }
        if degenerate {
            // A mosaic below 2 x 2 photosites is a bilinear demosaic on the CPU (no kernel): the
            // CPU stage is the same stage, and costs nothing.
            run.step()?;
        } else {
            let ca = self.chain.ca_curves();
            let inputs = self.chain.demosaic_inputs(ca.as_ref());
            let frame = PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default())?;
            let (mosaic, core) = run.take_demosaic_input().ok_or_else(|| {
                GpuError::InvalidInput("the demosaic is not the next stage".into())
            })?;
            let work = DeviceWork::Demosaic(DemosaicWork::new(mosaic, core, frame)?);
            let rgb = runner.run_device(work, cancel)?;
            run.supply_demosaic(rgb)?;
        }
        cancel.check()?;
        run.run_to_end()?;
        Ok(run.finish()?)
    }

    /// The range R6 to R8: the demosaic and the stages after it as one device job, then R9 (the lens
    /// resample) on the CPU on what comes back.
    fn run_raw_group<R: DeviceRunner + ?Sized>(
        &self,
        runner: &R,
        raw: &RawFrame<'_>,
        mut run: crate::cpu::chain::TileRun<'_, '_>,
        first: usize,
        degenerate: bool,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
        let chain = self.chain;
        let stages_len = run.stages().len();
        let start = run.need(first + 1);
        let input = if degenerate {
            // The CPU demosaics a mosaic below 2 x 2 photosites (bilinear); the stages after it still
            // run on the device.
            run.step()?;
            let img = run
                .image()
                .ok_or_else(|| GpuError::InvalidInput("the demosaic left no image".into()))?
                .clone();
            RawInput::Image(img)
        } else {
            let ca = chain.ca_curves();
            let inputs = chain.demosaic_inputs(ca.as_ref());
            let frame = PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default())?;
            let (mosaic, core) = run.take_demosaic_input().ok_or_else(|| {
                GpuError::InvalidInput("the demosaic is not the next stage".into())
            })?;
            RawInput::Mosaic(DemosaicWork::new(mosaic, core, frame)?)
        };
        // The stages after the demosaic, up to R9: each with the rectangle the CPU stepper cuts for
        // it. A stage that does nothing for these controls is left out; the next one then names its
        // core relative to the image it really gets (`RawWork::run_once`).
        let nr = &chain.noise_reduction;
        let mut steps = Vec::new();
        let mut classical = false;
        let mut lens: Option<usize> = None;
        for i in first + 1..stages_len {
            let out = run.need(i + 1);
            match run.stages()[i] {
                Stage::Denoise { .. } => {
                    let keep_classical = nr.needs_classical();
                    classical = keep_classical;
                    steps.push(PostStep::Denoise {
                        out,
                        scale: network_sigma_scale(nr.luminance, chain.iso),
                        params: chain.denoise.unwrap_or_default(),
                        keep_classical,
                    });
                }
                Stage::Detail { .. } => {
                    let beta = classical.then(|| nr.beta());
                    if nr.color > 0.0 || beta.is_some() {
                        steps.push(PostStep::Detail {
                            out,
                            color: nr.color,
                            beta,
                        });
                    }
                }
                Stage::Sharpen(CaptureSharpen::RichardsonLucy(p)) => {
                    if let Some(plan) = RlPlan::new(&p) {
                        steps.push(PostStep::Sharpen {
                            out,
                            kind: SharpenKind::RichardsonLucy(plan),
                        });
                    }
                }
                Stage::Sharpen(CaptureSharpen::Diffraction(p)) => {
                    // The plan comes from the controls, the gains and the noise model, none of which
                    // depends on the tile; an inactive plan is the identity, as on the CPU.
                    if let Some(plan) =
                        DiffractionPlan::new(&p, chain.gains, &chain.noise_dn(), SPEC_LUMA)
                    {
                        steps.push(PostStep::Sharpen {
                            out,
                            kind: SharpenKind::Diffraction(plan),
                        });
                    }
                }
                Stage::Lens(_) => {
                    lens = Some(i);
                    break;
                }
                other => {
                    return Err(GpuError::InvalidInput(format!(
                        "the stage {:?} cannot follow the demosaic on the device",
                        RawStage::of(&other)
                    )));
                }
            }
        }
        let end = run.need(lens.unwrap_or(stages_len));
        let sigma = self.sigma_work(raw, start, chain)?;
        let work = DeviceWork::Raw(RawWork {
            input,
            start,
            end,
            sigma,
            steps,
        });
        let rgb = runner.run_device(work, cancel)?;
        cancel.check()?;
        match lens {
            Some(i) => {
                let Stage::Lens(geo) = run.stages()[i] else {
                    return Err(GpuError::InvalidInput("the lens stage moved".into()));
                };
                Ok(crate::cpu::resample::resample_lens(&rgb, geo, run.out())
                    .map_err(|e| GpuError::Cpu(e.to_string()))?)
            }
            None => Ok(rgb),
        }
    }

    /// The noise sigma of the tile's first rectangle `start` (frame coordinates), from the half plane
    /// given to [`with_sigma`](Self::with_sigma), else the chain's map, else its scalar.
    fn sigma_work(
        &self,
        raw: &RawFrame<'_>,
        start: Rect,
        chain: &Chain,
    ) -> Result<SigmaWork, GpuError> {
        if let Some(half) = &self.sigma {
            return Ok(SigmaWork::Cells(SigmaTile::cut(half, start)?));
        }
        if let Some(map) = &chain.sigma_map {
            let mut plane = Vec::with_capacity(start.area());
            for y in start.y..start.y1() {
                plane.extend_from_slice(&map[y * raw.width + start.x..y * raw.width + start.x1()]);
            }
            return Ok(SigmaWork::Plane(plane));
        }
        Ok(SigmaWork::Scalar(chain.sigma))
    }
}
