//! The device chain: stages of the raw-processing chain (`docs/05`) run on the wgpu device, the rest
//! on the CPU stepper, for one tile at a time (`docs/gpu-plan.md` sections 4.3 and 11, task G-03).
//!
//! [`GpuChain`] is built from the very `Chain` and `Prepared` the CPU uses, and walks the same
//! stages with the same rectangles ([`TileRun`]), so the settings, the stage choice and the halos are
//! single-sourced. A [`StageRange`] says which stages run on the device; every stage outside it runs
//! on the CPU stepper and the boundary is a host copy. This task (G-03) has one range, R6 (the adaptive
//! ratio demosaic, [`StageRange::DEMOSAIC`]): the CPU runs R1 to R5b exactly as today and hands the
//! balanced mosaic over, the device runs the demosaic (including the R4 undo, the CA path and the
//! shadow colour matching of the kernel), the planes come back as an [`RgbImage`] and the CPU continues
//! with R6's bilateral denoise, R7, R8 and R9. The later tasks widen the range (G-06 to R8, G-07 to
//! R1..R9) by adding variants of [`DeviceWork`].
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
use crate::cpu::tiling::{Halo, Rect, Stage};
use crate::cpu::{Mosaic, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::kernels::adaptive_ratio::{self, Balance, PreparedDemosaic};
use crate::queue::CancelToken;

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
        match self {
            DeviceWork::Demosaic(w) => w.run(ctx, cancel),
        }
    }

    /// The same as [`run`](Self::run) for work held by reference (tests that run one piece twice).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_ref(&self, ctx: &GpuContext, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        match self {
            DeviceWork::Demosaic(w) => w.run(ctx, cancel),
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
}

impl<'a> GpuChain<'a> {
    /// The device chain for `chain` and the whole-frame reductions `prepared` the CPU also uses.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a range the device does not run yet (only
    /// [`StageRange::DEMOSAIC`] in this version) or a chain whose R6 method is not the adaptive
    /// ratio demosaic (the other methods run on the CPU only).
    pub fn new(
        chain: &'a Chain,
        prepared: &'a Prepared,
        range: StageRange,
    ) -> Result<GpuChain<'a>, GpuError> {
        if range != StageRange::DEMOSAIC {
            return Err(GpuError::InvalidInput(format!(
                "the device chain runs the range {} only, not {}",
                StageRange::DEMOSAIC.label(),
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
        })
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
}
