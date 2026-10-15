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
//! * R1 to R9 ([`StageRange::RAW_TO_LENS`], G-07): the whole classical chain from the photo's raw
//!   words, resident on the device ([`DeviceRaw`], uploaded once per photo): R1 normalise, R3
//!   vignetting and R4 gains in one pass, R5 highlights with the CPU's chroma offsets, R5b lateral CA
//!   on the mosaic, the demosaic and the stages after it as above, and R9 on the device too, one job
//!   per tile with one readback ([`GpuChain::run_tile_front`]). A demosaic core above
//!   [`MAX_PHYS_AREA`] is cut, and the front of each part is made from its own rectangle (the front is
//!   tile-exact).
//!
//! The reduced fit-view path has its own runner, [`GpuReduced`], which never builds a
//! full-resolution frame on the host.
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

use std::sync::Arc;

use ym_noise::sigma_map::SigmaParams;

use crate::cpu::chain::{Chain, Prepared, RawFrame};
use crate::cpu::demosaic::Method;
use crate::cpu::demosaic::adaptive_ratio::Params;
use crate::cpu::denoise::BilateralParams;
use crate::cpu::detail::network_sigma_scale;
use crate::cpu::sharpen::CaptureSharpen;
use crate::cpu::sharpen::diffraction::{DiffractionPlan, SPEC_LUMA};
use crate::cpu::normalize::Levels;
use crate::cpu::tiling::{Halo, Rect, Stage};
use crate::cpu::{Mosaic, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::adaptive_ratio::{self, Balance, PreparedDemosaic};
use crate::kernels::bilateral::{self, SigmaHalf, SigmaInput, SigmaTile};
use crate::kernels::bin::{self, BinParams};
use crate::kernels::highlights::{OffsetReducer, Offsets, encode_highlights};
use crate::kernels::lens::{LensTable, encode_lens};
use crate::kernels::mosaic_ca::{CaTable, encode_mosaic_ca};
use crate::kernels::prep::{DeviceRaw, PrepParams, VignetteTable, encode_prep};
use crate::kernels::sigma::encode_sigma_plane;
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

    /// R1 to R9 (G-07): the whole chain from the resident raw words, R9 included.
    pub const RAW_TO_LENS: StageRange = StageRange {
        first: RawStage::Normalize,
        last: RawStage::Lens,
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
        self.run_parts(ctx, cancel, parts)
    }

    /// [`run_split`](Self::run_split) without the start-of-tile token check and fault hook: for a
    /// caller that has made both for the tile it belongs to (the raw group's tile, which is one
    /// tile for the fault counter whether or not its demosaic core is cut).
    fn run_parts(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        parts: &[Rect],
    ) -> Result<RgbImage, GpuError> {
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

/// The stages in front of the demosaic that look at neighbours, as the device chain needs to know
/// them to cut a rectangle (`Chain::stages`): R5 (halo 4 px) and R5b (its own halo), both on the
/// mosaic. R1, R3 and R4 are pointwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontShape {
    /// R5 runs.
    pub highlights: bool,
    /// R5b runs, with this halo (`lens::mosaic_ca_halo_px`).
    pub ca_halo_px: Option<usize>,
    /// The frame.
    pub frame: Rect,
}

/// The rectangles of the raw front for one target: the mosaic that R6 reads (frame coordinates).
/// Walks the stages backwards with the halos of `Stage::input_rect`, so it gives the CPU stepper's
/// rectangles (`TileRun::need`) for the target it is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontPlan {
    /// The rectangle R1, R3 and R4 make, and R5 reads.
    pub prep: Rect,
    /// R5: the core it writes, relative to `prep`.
    pub highlights: Option<Rect>,
    /// R5b: the rectangle it reads (frame coordinates) and the core it writes, relative to it.
    pub mosaic_ca: Option<(Rect, Rect)>,
    /// The rectangle the front leaves.
    pub target: Rect,
}

impl FrontShape {
    /// The rectangles for `target`.
    #[must_use]
    pub fn plan(&self, target: Rect) -> FrontPlan {
        let mut r = target;
        let mut mosaic_ca = None;
        let mut highlights = None;
        if let Some(h) = self.ca_halo_px {
            let input = r.grow_aligned(Halo::uniform(h), self.frame, 2);
            mosaic_ca = Some((input, r.relative_to(&input)));
            r = input;
        }
        if self.highlights {
            let input = r.grow_aligned(Halo::uniform(4), self.frame, 2);
            highlights = Some(r.relative_to(&input));
            r = input;
        }
        FrontPlan {
            prep: r.grow_aligned(Halo::NONE, self.frame, 2),
            highlights,
            mosaic_ca,
            target,
        }
    }
}

/// The raw front of one tile on the device: the resident photo, what R1 to R5b need, and the
/// demosaic after it (G-07).
#[derive(Debug)]
pub struct FrontWork {
    raw: Arc<DeviceRaw>,
    prep: PrepParams,
    /// The CPU's R5 chroma offsets (`Prepared::offsets`), rounded to `f32` where they are added.
    offsets: [f32; 3],
    ca: Option<CaTable>,
    shape: FrontShape,
    /// The mosaic R6 reads, in frame coordinates (the CPU stepper's `need[demosaic]`).
    target: Rect,
    /// The part of it R6 demosaics, relative to `target`.
    core: Rect,
    frame: PreparedDemosaic,
}

impl FrontWork {
    /// The pattern at the first photosite of a rectangle of the frame.
    fn cfa_at(&self, x: usize, y: usize) -> crate::cpu::Cfa {
        self.raw.cfa().shifted(x, y)
    }

    /// Records the front for `target` (frame coordinates) and returns the mosaic. Buffers the
    /// recorded passes read stay in `keep` until the list has finished.
    fn encode_front(
        &self,
        passes: &mut PassList<'_>,
        target: Rect,
        keep: &mut Vec<GpuImage>,
    ) -> Result<GpuImage, GpuError> {
        let plan = self.shape.plan(target);
        let out = encode_prep(
            passes,
            &self.raw,
            plan.prep,
            &self.prep,
            plan.highlights.is_some(),
        )?;
        let mut cur = out.mosaic;
        if let (Some(core), Some(mask)) = (plan.highlights, out.mask.as_ref()) {
            let offsets = Offsets::from_host(passes, self.offsets)?;
            let cfa = self.cfa_at(cur.place.x0, cur.place.y0);
            let next = encode_highlights(passes, &cur, mask, cfa, core, &offsets)?;
            keep.push(std::mem::replace(&mut cur, next));
        }
        if let (Some((_, core)), Some(table)) = (plan.mosaic_ca, self.ca.as_ref()) {
            let cfa = self.cfa_at(cur.place.x0, cur.place.y0);
            let next = encode_mosaic_ca(passes, &cur, cfa, core, table)?;
            keep.push(std::mem::replace(&mut cur, next));
        }
        if (cur.width, cur.height) != (target.w, target.h) {
            return Err(GpuError::InvalidInput(format!(
                "the raw front left {} x {} photosites, the demosaic reads {:?}",
                cur.width, cur.height, target
            )));
        }
        Ok(cur)
    }

    /// The demosaic core cut into `parts` (rectangles relative to `target`), each from the front of
    /// its own rectangle, the planes put together on the host.
    fn run_parts(
        &self,
        ctx: &GpuContext,
        cancel: &CancelToken,
        parts: &[Rect],
    ) -> Result<RgbImage, GpuError> {
        let whole = Rect::whole(self.target.w, self.target.h);
        let halo = Halo::uniform(self.frame.halo_px());
        let mut out = RgbImage::new(self.core.w, self.core.h);
        out.place = crate::cpu::Placement::whole(self.raw.width(), self.raw.height())
            .offset(self.target.x + self.core.x, self.target.y + self.core.y);
        for part in parts {
            cancel.check()?;
            let sub_rect = part.grow(halo, whole);
            let target = Rect::new(
                self.target.x + sub_rect.x,
                self.target.y + sub_rect.y,
                sub_rect.w,
                sub_rect.h,
            );
            let mut keep = Vec::new();
            let mut passes = PassList::new(ctx, "raw front part");
            let mosaic = self.encode_front(&mut passes, target, &mut keep)?;
            let cfa = self.cfa_at(target.x, target.y);
            let img = adaptive_ratio::encode_demosaic(
                &mut passes,
                &mosaic,
                cfa,
                part.relative_to(&sub_rect),
                &self.frame,
            )?;
            let read = img.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = img.take(&done, read)?;
            drop((keep, mosaic));
            let host = RgbImage {
                width: img.width,
                height: img.height,
                place: img.place,
                data,
            };
            out.paste(
                &host,
                Rect::whole(host.width, host.height),
                part.x - self.core.x,
                part.y - self.core.y,
            );
        }
        Ok(out)
    }
}

/// R9 on the device after the stages: the geometry and the output rectangle (G-07).
#[derive(Debug)]
struct LensWork {
    table: LensTable,
    out: Rect,
}

/// What the stages after the demosaic receive.
#[derive(Debug)]
enum RawInput {
    /// The balanced mosaic: the device runs the demosaic first.
    Mosaic(Box<DemosaicWork>),
    /// The demosaiced planes (a mosaic below 2 x 2 photosites was demosaiced on the CPU).
    Image(RgbImage),
    /// The photo's raw words on the device: R1 to R5b run first, then the demosaic (G-07).
    Front(Box<FrontWork>),
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
    /// R9 on the device after the steps, when the range holds it.
    lens: Option<LensWork>,
}

impl RawWork {
    /// The dispatches the work records at most (informational: the demosaic's up to 14, the sigma
    /// plane, the denoise, the guided filter's 4 and 2 for the injection, Richardson-Lucy's `2 + 4 n`).
    #[must_use]
    pub fn max_dispatches(&self) -> usize {
        let mut n = match &self.input {
            RawInput::Mosaic(_) => 14,
            RawInput::Image(_) => 0,
            // The prep and its bits, the superpixels and the apply pass, the CA pass, the demosaic.
            RawInput::Front(_) => 5 + 14,
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
            RawInput::Front(f) => split_core_above(f.core, max_padded_pixels).len(),
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
        // The diffraction plan's taps (a pool buffer) are read by the recorded passes, so the plan lives
        // as long as the images do.
        let mut plans: Vec<PreparedDiffraction> = Vec::new();
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
                    let out = adaptive_ratio::encode_demosaic(
                        &mut passes,
                        &input,
                        m.cfa,
                        d.core,
                        &d.frame,
                    )?;
                    keep.push(input);
                    out
                } else {
                    // A core above MAX_PHYS_AREA is cut in rectangles whose planes are put together on
                    // the host and uploaded once more (a corner tile of a strong lens; the stages that
                    // follow need the whole core in one image).
                    let rgb = d.run_parts(ctx, cancel, &parts)?;
                    passes = PassList::new(ctx, "raw R6 to R8");
                    GpuImage::upload_rgb(ctx, "adaptive ratio rgb", &rgb)?
                }
            }
            RawInput::Image(img) => {
                passes = PassList::new(ctx, "raw R6 to R8");
                GpuImage::upload_rgb(ctx, "demosaiced planes", img)?
            }
            RawInput::Front(f) => {
                let parts = split_core_above(f.core, max_padded_pixels);
                if let [only] = parts.as_slice()
                    && *only == f.core
                {
                    passes = PassList::new(ctx, "raw R1 to R9");
                    let mosaic = f.encode_front(&mut passes, f.target, &mut keep)?;
                    let cfa = f.cfa_at(f.target.x, f.target.y);
                    let out = adaptive_ratio::encode_demosaic(
                        &mut passes,
                        &mosaic,
                        cfa,
                        f.core,
                        &f.frame,
                    )?;
                    keep.push(mosaic);
                    out
                } else {
                    // A core above MAX_PHYS_AREA: each rectangle from the front of its own
                    // rectangle, the planes put together on the host (`FrontWork::run_parts`).
                    let rgb = f.run_parts(ctx, cancel, &parts)?;
                    passes = PassList::new(ctx, "raw R1 to R9");
                    GpuImage::upload_rgb(ctx, "adaptive ratio rgb", &rgb)?
                }
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
                            let out = encode_sharpen(&mut passes, &prepared, &cur, core)?;
                            plans.push(prepared);
                            out
                        }
                    };
                    keep.push(std::mem::replace(&mut cur, next));
                    cur_rect = out;
                }
            }
        }
        cancel.check()?;
        if let Some(lens) = &self.lens {
            // R9 on the device: the window of the current image that holds the stage's input
            // rectangle (`end`), resampled to the tile.
            if !cur_rect.contains(&self.end) {
                return Err(GpuError::InvalidInput(format!(
                    "the stages left {cur_rect:?}, the lens stage needs {:?}",
                    self.end
                )));
            }
            let window = self.end.relative_to(&cur_rect);
            let next = encode_lens(&mut passes, &cur, window, &lens.table, lens.out)?;
            keep.push(std::mem::replace(&mut cur, next));
            cur_rect = self.end;
        }
        let read = cur.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let data = cur.take(&done, read)?;
        drop((keep, plans, sigma, classical));
        let mut rgb = RgbImage {
            width: cur.width,
            height: cur.height,
            place: cur.place,
            data,
        };
        if self.lens.is_none() && cur_rect != self.end {
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
    /// [`GpuError::InvalidInput`] for a range the device does not run (only [`StageRange::DEMOSAIC`],
    /// [`StageRange::DEMOSAIC_TO_SHARPEN`] and [`StageRange::RAW_TO_LENS`]) or a chain
    /// whose R6 method is not the adaptive ratio demosaic (the other methods run on the CPU only).
    pub fn new(
        chain: &'a Chain,
        prepared: &'a Prepared,
        range: StageRange,
    ) -> Result<GpuChain<'a>, GpuError> {
        if range != StageRange::DEMOSAIC
            && range != StageRange::DEMOSAIC_TO_SHARPEN
            && range != StageRange::RAW_TO_LENS
        {
            return Err(GpuError::InvalidInput(format!(
                "the device chain runs the ranges {}, {} and {}, not {}",
                StageRange::DEMOSAIC.label(),
                StageRange::DEMOSAIC_TO_SHARPEN.label(),
                StageRange::RAW_TO_LENS.label(),
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
        if self.range == StageRange::RAW_TO_LENS {
            return Err(GpuError::InvalidInput(
                "the range R1..R9 reads the resident raw words: use run_tile_front".into(),
            ));
        }
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
            RawInput::Mosaic(Box::new(DemosaicWork::new(mosaic, core, frame)?))
        };
        // The stages after the demosaic, up to R9: each with the rectangle the CPU stepper cuts for
        // it (`post_steps`).
        let needs: Vec<Rect> = (0..=stages_len).map(|i| run.need(i)).collect();
        let (steps, lens) = self.post_steps(run.stages(), &needs, first)?;
        let end = run.need(lens.unwrap_or(stages_len));
        let sigma = self.sigma_work(raw.width, start, chain)?;
        let work = DeviceWork::Raw(RawWork {
            input,
            start,
            end,
            sigma,
            steps,
            lens: None,
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

    /// The stages after the demosaic (index `first`), up to R9: each with the rectangle the CPU
    /// stepper cuts for it (`needs[i + 1]` is the rectangle stage `i` leaves). A stage that does
    /// nothing for these controls is left out; the next one then names its core relative to the image
    /// it really gets (`RawWork::run_once`). Returns the steps and the index of the lens stage.
    fn post_steps(
        &self,
        stages: &[Stage<'_>],
        needs: &[Rect],
        first: usize,
    ) -> Result<(Vec<PostStep>, Option<usize>), GpuError> {
        let chain = self.chain;
        let nr = &chain.noise_reduction;
        let mut steps = Vec::new();
        let mut classical = false;
        let mut lens: Option<usize> = None;
        for i in first + 1..stages.len() {
            let out = needs[i + 1];
            match stages[i] {
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
        Ok((steps, lens))
    }

    /// Runs the part of the output covered by `out` (output coordinates, as [`Chain::run_tile`]) of
    /// the whole chain R1 to R9 from the photo's resident raw words, and returns exactly that
    /// rectangle (the range must be [`StageRange::RAW_TO_LENS`]).
    ///
    /// Nothing runs on the CPU: the front (R1, R3, R4 in one pass, R5 with the CPU's chroma offsets of
    /// [`Prepared`], R5b), the demosaic, the denoise, R7, R8 and the lens resample are one device job
    /// with one readback. The rectangles are the CPU stepper's: `needs[i]` is walked backwards from
    /// `out` with `Stage::input_rect` as `Chain::begin_tile` does.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for another range, a raw frame whose size is not the chain's lens
    /// geometry's, an `out` outside the output, a mosaic below 2 x 2 photosites, or a vignetting
    /// profile the kernel cannot take; [`GpuError::Cancelled`]; the runner's errors, which the
    /// integrator answers with the CPU chain.
    pub fn run_tile_front<R: DeviceRunner + ?Sized>(
        &self,
        runner: &R,
        raw: &Arc<DeviceRaw>,
        out: Rect,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
        cancel.check()?;
        if self.range != StageRange::RAW_TO_LENS {
            return Err(GpuError::InvalidInput(format!(
                "run_tile_front runs the range {}, not {}",
                StageRange::RAW_TO_LENS.label(),
                self.range.label()
            )));
        }
        let chain = self.chain;
        let frame = Rect::whole(raw.width(), raw.height());
        if let Some(l) = &chain.lens
            && (l.frame_width(), l.frame_height()) != (frame.w, frame.h)
        {
            return Err(GpuError::InvalidInput(format!(
                "the lens geometry is for a {} x {} frame, the resident raw frame is {} x {}",
                l.frame_width(),
                l.frame_height(),
                frame.w,
                frame.h
            )));
        }
        let (ow, oh) = chain.output_size(frame.w, frame.h);
        if out.is_empty() || out.x1() > ow || out.y1() > oh {
            return Err(GpuError::InvalidInput(format!(
                "the tile {out:?} is empty or outside the {ow} x {oh} output"
            )));
        }
        let stages = chain.stages();
        let mut needs = vec![out; stages.len() + 1];
        for i in (0..stages.len()).rev() {
            needs[i] = stages[i].input_rect(needs[i + 1], frame);
        }
        let d = stages
            .iter()
            .position(|s| RawStage::of(s) == RawStage::Demosaic)
            .ok_or_else(|| GpuError::InvalidInput("the chain has no demosaic stage".into()))?;
        let target = needs[d];
        if target.w < 2 || target.h < 2 {
            return Err(GpuError::InvalidInput(
                "a mosaic below 2 x 2 photosites is a CPU stage".into(),
            ));
        }
        let shape = FrontShape {
            highlights: stages[..d]
                .iter()
                .any(|s| RawStage::of(s) == RawStage::Highlights),
            ca_halo_px: stages[..d].iter().find_map(|s| match s {
                Stage::MosaicCa { halo_px } => Some(*halo_px),
                _ => None,
            }),
            frame,
        };
        let ca = match (shape.ca_halo_px, chain.mosaic_ca_curves()) {
            (Some(_), Some(curves)) => Some(CaTable::from_curves(&curves)?),
            (Some(_), None) => {
                return Err(GpuError::InvalidInput(
                    "the chain has a mosaic CA stage and no curves for it".into(),
                ));
            }
            _ => None,
        };
        let vignette = chain
            .vignette
            .as_ref()
            .map(VignetteTable::from_profile)
            .transpose()?;
        let off = &self.prepared.offsets.offset;
        let ca_curves = chain.ca_curves();
        let inputs = chain.demosaic_inputs(ca_curves.as_ref());
        let frame_demosaic = PreparedDemosaic::new(Balance::Balanced, &inputs, &Params::default())?;
        let core = needs[d + 1].relative_to(&target);
        let input = RawInput::Front(Box::new(FrontWork {
            raw: Arc::clone(raw),
            prep: PrepParams {
                levels: chain.levels,
                gains: chain.gains,
                apply_gains: true,
                vignette,
            },
            offsets: [off[0] as f32, off[1] as f32, off[2] as f32],
            ca,
            shape,
            target,
            core,
            frame: frame_demosaic,
        }));
        let start = needs[d + 1];
        let (steps, lens_at) = self.post_steps(&stages, &needs, d)?;
        let lens = match lens_at {
            Some(i) => {
                let Stage::Lens(geo) = stages[i] else {
                    return Err(GpuError::InvalidInput("the lens stage moved".into()));
                };
                Some(LensWork {
                    table: LensTable::from_geometry(geo)?,
                    out,
                })
            }
            None => None,
        };
        let end = needs[lens_at.unwrap_or(stages.len())];
        let sigma = self.sigma_work(frame.w, start, chain)?;
        let work = DeviceWork::Raw(RawWork {
            input,
            start,
            end,
            sigma,
            steps,
            lens,
        });
        let rgb = runner.run_device(work, cancel)?;
        cancel.check()?;
        Ok(rgb)
    }

    /// The noise sigma of the tile's first rectangle `start` (frame coordinates), from the half plane
    /// given to [`with_sigma`](Self::with_sigma), else the chain's map, else its scalar.
    fn sigma_work(
        &self,
        frame_width: usize,
        start: Rect,
        chain: &Chain,
    ) -> Result<SigmaWork, GpuError> {
        if let Some(half) = &self.sigma {
            return Ok(SigmaWork::Cells(SigmaTile::cut(half, start)?));
        }
        if let Some(map) = &chain.sigma_map {
            let mut plane = Vec::with_capacity(start.area());
            for y in start.y..start.y1() {
                plane.extend_from_slice(&map[y * frame_width + start.x..y * frame_width + start.x1()]);
            }
            return Ok(SigmaWork::Plane(plane));
        }
        Ok(SigmaWork::Scalar(chain.sigma))
    }
}

// ---------------------------------------------------------------------------------------------
// The reduced fit-view path
// ---------------------------------------------------------------------------------------------

/// Photosite rows of one slab of the reduced path (a multiple of 2): the rows of the normalised
/// mosaic that exist at one time, 38 MB at 9500 photosites a row.
pub const REDUCED_SLAB_ROWS: usize = 1024;

/// The denoise of the reduced path: the sigma scale and the filter.
#[derive(Clone, Copy, Debug)]
pub struct ReducedDenoise {
    /// `network_sigma_scale(L, ISO)`; the denoise runs only for a scale above zero.
    pub scale: f32,
    /// The filter's parameters.
    pub params: BilateralParams,
}

/// What the reduced fit-view path needs besides the photo's raw words, as plain data (`ym-core` builds
/// it from the `RawPlan` and the `RawParams`, `docs/gpu-plan.md` section 4.3).
#[derive(Clone, Debug)]
pub struct ReducedParams {
    /// The binning factor: even, 2 to 16 (`RawPlan::effective_scale`).
    pub k: usize,
    /// The R1 levels at the frame's parity.
    pub levels: Levels,
    /// The R4 gains `g0`.
    pub g0: [f32; 3],
    /// Whether R5 runs (`decode.highlight_recovery` and the frame has a clipped photosite).
    pub highlight_recovery: bool,
    /// The vignetting profile, applied at the binned pixels, or `None`.
    pub vignette: Option<VignetteTable>,
    /// The noise model and the gains per CFA position, for the sigma plane.
    pub noise: SigmaParams,
    /// The noise left in the luminance of a bin (`binning_noise_factor(k)`).
    pub noise_factor: f32,
    /// The light denoise, or `None` below a scale of zero.
    pub denoise: Option<ReducedDenoise>,
    /// R9 at `1/k` (`RawPlan::lens_at`), or `None`.
    pub lens: Option<LensTable>,
}

/// The reduced fit-view path on the device: R1 to R9 at `1/k` of the resolution, from the resident raw
/// words, never building a full-resolution frame on the host (`docs/gpu-plan.md` N-8). The CPU
/// reference is `ym-core`'s `reduced_bayer`.
///
/// The frame goes through in slabs of [`REDUCED_SLAB_ROWS`] rows. With R5 on, a first sweep reduces
/// the global chroma offsets (a fixed-order reduction, [`OffsetReducer`]); a second sweep makes each
/// slab's normalised mosaic, applies R5 and bins it into the output; then the sigma plane (from the
/// raw words), the binned sigma, the light bilateral denoise and R9 at `1/k` run on the small image.
#[derive(Debug)]
pub struct GpuReduced;

impl GpuReduced {
    /// Runs the path and returns the image (`out_w x out_h` after R9, three planes) on the device.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] between slabs; [`GpuError::InvalidInput`] for parameters the kernels
    /// refuse (an odd `k`, a frame smaller than one block, a vignetting profile with too many knots);
    /// the pool's errors; shader errors.
    pub fn run(
        ctx: &GpuContext,
        raw: &DeviceRaw,
        p: &ReducedParams,
        cancel: &CancelToken,
    ) -> Result<GpuImage, GpuError> {
        Self::run_with_slabs(ctx, raw, p, cancel, REDUCED_SLAB_ROWS)
    }

    /// [`run`](Self::run) with slabs of `slab_rows` photosite rows (tests make them small).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub fn run_with_slabs(
        ctx: &GpuContext,
        raw: &DeviceRaw,
        p: &ReducedParams,
        cancel: &CancelToken,
        slab_rows: usize,
    ) -> Result<GpuImage, GpuError> {
        cancel.check()?;
        ctx.inject_tile_fault()?;
        let (w, h) = (raw.width(), raw.height());
        let slab_rows = (slab_rows.max(8)) & !1;
        let bp = BinParams {
            k: p.k,
            cfa: raw.cfa(),
            frame_w: w,
            frame_h: h,
            vignette: p.vignette.clone(),
        };
        bp.check()?;
        let (ow, oh) = (bp.out_w(), bp.out_h());
        // R1 and R4, no vignetting: the vignetting gain is applied at the binned pixels.
        let prep = PrepParams {
            levels: p.levels,
            gains: p.g0,
            apply_gains: true,
            vignette: None,
        };
        let cfa_of = |r: Rect| raw.cfa().shifted(r.x, r.y);

        // 1. The global chroma offsets, over the whole frame in slabs.
        let offsets = if p.highlight_recovery && w >= 2 && h >= 2 {
            let sw = w / 2;
            let rows = h / 2 * 2;
            let mut slabs = Vec::new();
            let mut r0 = 0;
            while r0 < rows {
                slabs.push((r0, (r0 + slab_rows).min(rows)));
                r0 += slab_rows;
            }
            let capacity: usize = slabs
                .iter()
                .map(|&(a, b)| OffsetReducer::partials_for(sw, b - a))
                .sum();
            let mut reducer = OffsetReducer::new(ctx, capacity)?;
            for (r0, r1) in slabs {
                cancel.check()?;
                let top = r0.saturating_sub(4) & !1;
                let bottom = (r1 + 4).min(h);
                let rect = Rect::new(0, top, w, bottom - top);
                let mut passes = PassList::new(ctx, "reduced offsets slab");
                let out = encode_prep(&mut passes, raw, rect, &prep, true)?;
                let Some(mask) = out.mask.as_ref() else {
                    return Err(GpuError::InvalidInput("prep made no clip bits".into()));
                };
                reducer.encode_slab(&mut passes, &out.mosaic, mask, cfa_of(rect), (r0 - top, r1 - top))?;
                passes.submit()?.finish()?;
            }
            let mut passes = PassList::new(ctx, "reduced offsets fold");
            let off = reducer.encode_fold(&mut passes)?;
            passes.submit()?.finish()?;
            Some(off)
        } else {
            None
        };

        // 2. The binned image, a slab of output rows at a time.
        let binned = GpuImage::alloc(ctx, "binned rgb", ow, oh, 3, crate::cpu::Placement::whole(ow, oh))?;
        let step = (slab_rows / p.k).max(1);
        let mut r = 0;
        while r < oh {
            cancel.check()?;
            let rows = r..(r + step).min(oh);
            let need = bin::slab_rows(p.k, &rows, h);
            let mut passes = PassList::new(ctx, "reduced bin slab");
            let mut keep: Vec<GpuImage> = Vec::new();
            let slab = match &offsets {
                Some(off) => {
                    let top = need.start.saturating_sub(4) & !1;
                    let bottom = (need.end + 4).min(h);
                    let rect = Rect::new(0, top, w, bottom - top);
                    let out = encode_prep(&mut passes, raw, rect, &prep, true)?;
                    let Some(mask) = out.mask.as_ref() else {
                        return Err(GpuError::InvalidInput("prep made no clip bits".into()));
                    };
                    let core = Rect::new(0, need.start - top, w, need.end - need.start);
                    let slab = encode_highlights(&mut passes, &out.mosaic, mask, cfa_of(rect), core, off)?;
                    keep.push(out.mosaic);
                    slab
                }
                None => {
                    let rect = Rect::new(0, need.start, w, need.end - need.start);
                    encode_prep(&mut passes, raw, rect, &prep, false)?.mosaic
                }
            };
            bin::encode_bin(&mut passes, &slab, &bp, rows.clone(), &binned)?;
            passes.submit()?.finish()?;
            drop((keep, slab));
            r = rows.end;
        }

        // 3. The sigma, the light denoise and R9 on the small image.
        cancel.check()?;
        let mut passes = PassList::new(ctx, "reduced finish");
        let mut keep: Vec<GpuImage> = Vec::new();
        let mut cur = binned;
        if let Some(dn) = &p.denoise {
            let sigma = encode_sigma_plane(&mut passes, raw, &p.levels, &p.noise)?;
            let bsigma = bin::encode_bin_sigma(&mut passes, &sigma, &bp, p.noise_factor)?;
            let next = bilateral::encode_bilateral(
                &mut passes,
                &cur,
                Rect::whole(ow, oh),
                &bsigma,
                (0, 0),
                dn.scale,
                &dn.params,
            )?;
            keep.push(sigma);
            keep.push(bsigma);
            keep.push(std::mem::replace(&mut cur, next));
        }
        if let Some(lens) = &p.lens {
            let (cw, ch) = lens.output_size();
            let next = encode_lens(&mut passes, &cur, Rect::whole(ow, oh), lens, Rect::whole(cw, ch))?;
            keep.push(std::mem::replace(&mut cur, next));
        }
        passes.submit()?.finish()?;
        drop(keep);
        Ok(cur)
    }

    /// [`run`](Self::run) and the image read back as a host image (the group the renderer caches).
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run), plus the readback's.
    pub fn run_rgb(
        ctx: &GpuContext,
        raw: &DeviceRaw,
        p: &ReducedParams,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
        Self::run(ctx, raw, p, cancel)?.read_rgb(ctx)
    }
}
