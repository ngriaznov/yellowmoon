//! Output sharpening on the GPU (RA-09): the WGSL port of [`crate::cpu::output_sharpen`], specified in
//! `docs/algorithms/04-output-sharpening.md`.
//!
//! [`PreparedSharpen::new`] derives the stages of an [`OutputSharpen`] for an output scale with the
//! CPU reference's own plan code, rounds the CPU's `f64` kernel taps to `f32` and uploads them once;
//! [`run`](PreparedSharpen::run) sharpens a whole planar image and
//! [`run_tiled`](PreparedSharpen::run_tiled) does it tile by tile (core plus the exact receptive field
//! of the plan, `docs/algorithms/04-output-sharpening.md` section 1.4) with the cores handed to a sink
//! in raster order, like the colour kernel. Nothing here is wired into the render or export path: the
//! raw stages and the export run the CPU reference, and wiring this one is a later decision.
//!
//! # Passes
//!
//! One compute dispatch per pass over `output_sharpen.wgsl`, each with its own pipeline (the passes
//! are separate [`KernelId`]s that share one source file). For each tile:
//!
//! 1. luma `L` (per pixel, `f32`, the exact operation order of spec 3.1);
//! 2. per stage: the high-pass in difference form (two passes), for methods 1 and 2 with a threshold
//!    the gain terms `P` and `Q` and their separable blur with the gain kernel (three passes), and
//!    the accumulation `D += a_free s`, `Hacc += a_halo s`;
//! 3. when any stage has a limited share: the 5 x 5 envelope window (25 reads per pixel), the
//!    separable blur of the four maps with the halo kernel (two passes);
//! 4. the limiter's clamp and the ratio transfer, over the core only.
//!
//! No pass reads a pixel that another thread of the same dispatch writes, there are no atomics and no
//! reductions, and every sum has a fixed order, so a run is deterministic and a tile gives exactly
//! the bits of the same pixels in a whole-frame run.
//!
//! # Precision decisions
//!
//! * **Storage is `f32` whatever the device's storage mode.** The colour kernels keep image
//!   intermediates in f16, but the halo limiter compares luma values for equality and order (spec
//!   5.4), and an f16 round trip would turn most of its ties into different counts. Every plane (`L`,
//!   `t`, `h`, `D`, `Hacc`, gain terms, envelope maps) and the input and output pixels are `f32`
//!   buffers; the sharpened pixels go back to the caller as `f32` planes.
//! * **Accumulation is `f32`.** WGSL has no `f64`, so convolutions, window sums and the envelope
//!   formula accumulate in `f32` (taps left to right, window row by row), the model of the C++
//!   reference's `Accumulation::F32` mode, where the CPU port uses `f64`. The taps are the CPU's `f64`
//!   kernels rounded to `f32`. The difference form of the high-pass and of the window sums is kept, so
//!   a flat region gives exact zeros, as on the CPU.
//! * **No `erf` on the GPU.** The kernels, the threshold rescale and the gradient constants of method
//!   2 are computed on the CPU in `f64` (the same code paths as the CPU reference) and passed as taps
//!   and uniforms.
//! * **Rank counts.** The limiter's counts depend on exact `f32` values of `L` and `S = L + Hacc`.
//!   `L` has the CPU's bits when the driver does not contract the luma into fused multiply-adds
//!   (checked in `tests/gpu_output_sharpen.rs`); `S` differs from the CPU's wherever `f32` and `f64`
//!   accumulation round the high-pass differently, which flips a count where the content has ties in
//!   exact arithmetic (noiseless bars, checkerboards, diagonals: spec 5.4). On noisy and
//!   photographic content no count flips at frame sizes up to 150 x 113 and 2 to 4 of 2.8 million
//!   pixels flip at 2048 x 1365; on noiseless periodic content up to 19 % of the pixels do. The
//!   numbers, and the proof that the kernel is bit-identical to the C++ reference's
//!   `Accumulation::F32` mode (so every difference from the CPU is the accumulation precision), are in
//!   `docs/algorithms/04-output-sharpening.md` section 10.

use bytemuck::{Pod, Zeroable};

use super::color::ColorImage;
use super::shader::KernelId;
use super::util::{
    OutKind, PixelData, Readback, STAGING, STORAGE_IN, STORAGE_OUT, STORAGE_SCRATCH, bind,
    image_samples, storage_buffer, submit_with_readback, uniform_buffer, unpack,
};
use crate::cpu::RgbImage;
use crate::cpu::output_sharpen::{Method, OutputSharpen, SharpenPlan};
use crate::cpu::tiling::{Halo, Rect};
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::tiles::{Tile, TilePlan};

/// Device bytes one pixel of a compute rectangle needs at most: the input (3 planes), `L`, `t`, `h`,
/// `D`, `Hacc`, two working buffers of 4 planes and the sharpened core (3 planes), all `f32`. Pass it
/// to [`crate::kernels::raw::recommended_tile_px`] when choosing a tile size. That function checks
/// the binding limit for three planes of the device's colour storage mode, while the largest binding
/// here is a working buffer of four `f32` planes (16 bytes per pixel): the sizes in
/// [`crate::tiles::TILE_SIZES`] (at most 2048 px plus the halo) fit the default 128 MiB binding, a
/// device with a smaller binding limit must be checked by the caller (an oversized tile is refused
/// with [`GpuError::BufferTooLarge`], not run).
pub const BYTES_PER_PIXEL: u64 = 4 * (3 + 1 + 1 + 1 + 1 + 1 + 4 + 4 + 3);

/// Threads per workgroup side (the shader's `@workgroup_size(16, 16, 1)`).
const GROUP: u32 = 16;

/// Usage of the planes the host clears (the accumulators) or copies out (the traced planes).
const STORAGE_PLANE: wgpu::BufferUsages = STORAGE_IN.union(wgpu::BufferUsages::COPY_SRC);

/// The bindings of `output_sharpen.wgsl`: the discriminant is the `@binding` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Slot {
    Params = 0,
    Taps = 1,
    Img = 2,
    Lum = 3,
    Tmp = 4,
    Hp = 5,
    Dacc = 6,
    Hacc = 7,
    WorkA = 8,
    WorkB = 9,
    Out = 10,
}

/// The WGSL `main` that runs `function`, appended after the shared source.
fn entry(function: &str) -> String {
    format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {function}(gid); }}\n"
    )
}

/// The pass of `id`: its WGSL function and the bindings its entry point uses (the auto-generated
/// bind group layout holds exactly these).
fn pass_of(id: KernelId) -> Option<(&'static str, &'static [Slot])> {
    use Slot::{Dacc, Hacc, Hp, Img, Lum, Out, Params, Taps, Tmp, WorkA, WorkB};
    Some(match id {
        KernelId::OutputSharpenLuma => ("sharpen_luma", &[Params, Img, Lum]),
        KernelId::OutputSharpenHighpassRows => ("sharpen_highpass_rows", &[Params, Taps, Lum, Tmp]),
        KernelId::OutputSharpenHighpassCols => {
            ("sharpen_highpass_cols", &[Params, Taps, Lum, Tmp, Hp])
        }
        KernelId::OutputSharpenGainTerms => ("sharpen_gain_terms", &[Params, Lum, Hp, WorkA]),
        KernelId::OutputSharpenBlur => ("sharpen_blur", &[Params, Taps, WorkA, WorkB]),
        KernelId::OutputSharpenAccumulate => {
            ("sharpen_accumulate", &[Params, Hp, Dacc, Hacc, WorkA])
        }
        KernelId::OutputSharpenWindow => ("sharpen_window", &[Params, Lum, Hacc, WorkA]),
        KernelId::OutputSharpenTransfer => (
            "sharpen_transfer",
            &[Params, Img, Lum, Dacc, Hacc, WorkA, Out],
        ),
        _ => return None,
    })
}

/// The kernel body of an output sharpening pass: the shared WGSL plus the pass's `main`. Called by
/// [`super::shader::shader_source`]; any other id is a programming error and gets an empty body.
pub(super) fn body(id: KernelId) -> String {
    let Some((function, _)) = pass_of(id) else {
        return String::new();
    };
    let mut s = String::from(include_str!("../shaders/output_sharpen.wgsl"));
    s.push_str(&entry(function));
    s
}

/// The uniform block of `output_sharpen.wgsl` (`SharpenParams`): the same fields in the same order,
/// 24 scalars of 4 bytes. Every pass reads the fields it needs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ParamsU {
    width: u32,
    height: u32,
    plane: u32,
    direction: u32,
    core_x: u32,
    core_y: u32,
    core_w: u32,
    core_h: u32,
    tap_off: u32,
    tap_n: u32,
    planes: u32,
    method: u32,
    luma_r: f32,
    luma_g: f32,
    luma_b: f32,
    threshold: f32,
    a_free: f32,
    a_halo: f32,
    g0: f32,
    inv: f32,
    use_gain: u32,
    use_ramp: u32,
    halo_acc: u32,
    limiter: u32,
}

/// A kernel in the tap table.
#[derive(Clone, Copy, Debug)]
struct TapRange {
    offset: u32,
    count: u32,
}

/// One stage of the plan with everything the passes need, derived on the CPU in the same `f32`/`f64`
/// operations as `cpu::output_sharpen`.
#[derive(Clone, Copy, Debug)]
struct StageGpu {
    method: u32,
    threshold: f32,
    kernel: TapRange,
    /// The gain kernel; present for methods 1 and 2 with a threshold above 0.
    gain: Option<TapRange>,
    a_free: f32,
    a_halo: f32,
    /// Method 2: `g0` and `0.25 / g0` as `f32`, and whether the ramp exists (the kernel has not
    /// collapsed to one tap, so `left > 0`).
    g0: f32,
    inv: f32,
    ramp: bool,
}

/// An output sharpening setup uploaded to the GPU: immutable, reusable for any number of tiles and
/// images of the same scale.
#[derive(Debug)]
pub struct PreparedSharpen {
    plan: SharpenPlan,
    stages: Vec<StageGpu>,
    halo_kernel: Option<TapRange>,
    halo_px: usize,
    taps: PooledBuffer,
}

/// Intermediate planes of a run over a whole compute rectangle, read back for diagnostics (see
/// [`PreparedSharpen::run_traced`]). The three planes are row-major `width x height`, `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct SharpenTrace {
    /// Width of the traced rectangle.
    pub width: usize,
    /// Height of the traced rectangle.
    pub height: usize,
    /// The luma `L`.
    pub luma: Vec<f32>,
    /// `D`, the free delta after all stages, before the halo limiter.
    pub free_delta: Vec<f32>,
    /// `Hacc`, the halo-limited delta (all zeros when no stage has a limited share).
    pub halo_delta: Vec<f32>,
}

fn invalid(e: impl std::fmt::Display) -> GpuError {
    GpuError::InvalidInput(e.to_string())
}

/// Appends `kernel` rounded to `f32` to `taps` and returns where it sits.
fn push_taps(taps: &mut Vec<f32>, kernel: &[f64]) -> TapRange {
    let range = TapRange {
        offset: taps.len() as u32,
        count: kernel.len() as u32,
    };
    taps.extend(kernel.iter().map(|&t| t as f32));
    range
}

/// Method 2's constants (spec 3.3): `g0 = T ((k_c + k_{c+1}) / 2) / left` in `f64` then `f32`,
/// `inv = 0.25 / g0` in `f64` then `f32`, with `left` the sum of the taps left of the centre. A
/// kernel that has collapsed to one tap (`left == 0`) has no ramp. These are the preamble of the CPU
/// reference's `gradient_gain`, repeated here because that function is private.
fn ramp_constants(kernel: &[f64], threshold: f32) -> (bool, f32, f32) {
    let c = kernel.len() / 2;
    let left: f64 = kernel[..c].iter().sum();
    if left > 0.0 {
        let g0 = f64::from(threshold) * ((kernel[c] + kernel[c + 1]) * 0.5) / left;
        (true, g0 as f32, (0.25 / g0) as f32)
    } else {
        (false, 0.0, 0.0)
    }
}

impl PreparedSharpen {
    /// Derives the stages of `params` for an output `scale` times the rendered size (use
    /// [`crate::cpu::output_sharpen::scale_for_edges`]) and uploads the kernels.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for parameters or a scale the CPU reference's `plan` rejects, and
    /// allocation errors from the pool.
    pub fn new(ctx: &GpuContext, params: &OutputSharpen, scale: f32) -> Result<Self, GpuError> {
        let plan = params.plan(scale).map_err(invalid)?;
        let mut taps: Vec<f32> = Vec::new();
        let mut stages = Vec::with_capacity(plan.stages().len());
        for (i, st) in plan.stages().iter().enumerate() {
            let k = plan.kernel(i);
            let kernel = push_taps(&mut taps, &k);
            let has_gain = st.method != Method::HardThreshold && st.threshold > 0.0;
            let gain = has_gain.then(|| push_taps(&mut taps, &plan.gain_kernel(i)));
            // Method 0 compensates the amount for the threshold; the limited share is H of it
            // (spec 3.4). `f32` operations as in the CPU reference's `band_ratio`.
            let a = match st.method {
                Method::HardThreshold => st.amount / (1.0 - st.threshold),
                _ => st.amount,
            };
            let (a_free, a_halo) = if st.halo > 0.0 {
                let ah = st.halo * a;
                (a - ah, ah)
            } else {
                (a, 0.0)
            };
            let (ramp, g0, inv) = if has_gain && st.method == Method::GradientRamp {
                ramp_constants(&k, st.threshold)
            } else {
                (false, 0.0, 0.0)
            };
            stages.push(StageGpu {
                method: u32::from(st.method.id()),
                threshold: st.threshold,
                kernel,
                gain,
                a_free,
                a_halo,
                g0,
                inv,
                ramp,
            });
        }
        let halo_kernel = plan
            .halo_enabled()
            .then(|| push_taps(&mut taps, &plan.halo_kernel()));
        if taps.is_empty() {
            taps.push(0.0);
        }
        let taps_buf = storage_buffer(ctx, "sharpen taps", bytemuck::cast_slice(&taps))?;
        let halo_px = plan.halo_px();
        Ok(Self {
            plan,
            stages,
            halo_kernel,
            halo_px,
            taps: taps_buf,
        })
    }

    /// The derived plan (stages, kernels, receptive field) the kernel runs.
    pub fn plan(&self) -> &SharpenPlan {
        &self.plan
    }

    /// Whether no stage exists (amount or radius too small): a run copies its input.
    pub fn is_noop(&self) -> bool {
        self.stages.is_empty()
    }

    /// The exact receptive field in pixels (spec 1.4): the context a tile needs on every side.
    pub fn halo_px(&self) -> usize {
        self.halo_px
    }

    /// A tile plan for a `width x height` frame with cores of `tile_px` and this setup's halo.
    pub fn tile_plan(&self, width: usize, height: usize, tile_px: usize) -> TilePlan {
        TilePlan::with_halo(width, height, tile_px, self.halo_px)
    }

    /// Sharpens `image` in one set of dispatches over the whole frame. A frame needs
    /// [`BYTES_PER_PIXEL`] bytes of device memory, and the four-plane working buffers of the halo
    /// limiter must fit one storage binding (128 MiB by default: about 8 million pixels); larger
    /// frames go through [`run_tiled`](Self::run_tiled).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty image, a buffer that does not match its size or a side
    /// of more than `16 x max_compute_workgroups_per_dimension` pixels (1,048,560 on the default
    /// limits), [`GpuError::BufferTooLarge`] or [`GpuError::OutOfBudget`] for a frame that does not
    /// fit, and shader or readback errors.
    pub fn run(&self, ctx: &GpuContext, image: &RgbImage) -> Result<RgbImage, GpuError> {
        check_image(image)?;
        if self.is_noop() {
            return Ok(image.clone());
        }
        let whole = Rect::whole(image.width, image.height);
        let done = self
            .dispatch(ctx, image, whole, whole, false)?
            .finish(ctx)?;
        into_rgb(done.0)
    }

    /// Like [`run`](Self::run), and also returns the intermediate planes `L`, `D` and `Hacc` of the
    /// whole frame, for diagnostics and the parity tests. A no-op setup returns zero deltas.
    ///
    /// # Errors
    ///
    /// The errors of [`run`](Self::run).
    pub fn run_traced(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
    ) -> Result<(RgbImage, SharpenTrace), GpuError> {
        check_image(image)?;
        let (w, h) = (image.width, image.height);
        if self.is_noop() {
            let [r, g, b] = [image.plane(0), image.plane(1), image.plane(2)];
            let luma = (0..w * h)
                .map(|i| {
                    let wt = self.plan.luma_weights();
                    (((wt[0] * r[i]) * r[i] + (wt[1] * g[i]) * g[i]) + (wt[2] * b[i]) * b[i]).sqrt()
                })
                .collect();
            let trace = SharpenTrace {
                width: w,
                height: h,
                luma,
                free_delta: vec![0.0; w * h],
                halo_delta: vec![0.0; w * h],
            };
            return Ok((image.clone(), trace));
        }
        let whole = Rect::whole(w, h);
        let (image_out, trace) = self.dispatch(ctx, image, whole, whole, true)?.finish(ctx)?;
        let trace =
            trace.ok_or_else(|| GpuError::Readback("the trace was not read back".into()))?;
        Ok((into_rgb(image_out)?, trace))
    }

    /// Sharpens `image` tile by tile over `plan` with up to `depth` tiles in flight and hands each
    /// finished core (planar `f32`, dense) to `sink` in raster order. The result equals
    /// [`run`](Self::run) bit for bit when every tile's compute rectangle holds its core plus
    /// [`halo_px`](Self::halo_px) pixels (clipped to the frame); [`tile_plan`](Self::tile_plan) builds
    /// such a plan, and a plan with less halo is refused.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when `cancel` fires, [`GpuError::InvalidInput`] when the plan is for
    /// another frame size or its tiles lack the halo, plus the errors of [`run`](Self::run).
    pub fn run_tiled(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        plan: &TilePlan,
        depth: usize,
        cancel: &CancelToken,
        sink: &mut dyn FnMut(Tile, ColorImage) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        check_image(image)?;
        let frame = plan.frame();
        if (frame.w, frame.h) != (image.width, image.height) {
            return Err(GpuError::InvalidInput(format!(
                "tile plan is for {}x{}, the image is {}x{}",
                frame.w, frame.h, image.width, image.height
            )));
        }
        for tile in plan.tiles() {
            let needed = tile.core.grow(Halo::uniform(self.halo_px), frame);
            if !tile.compute.contains(&needed) {
                return Err(GpuError::InvalidInput(format!(
                    "tile {} computes {:?}, but its core {:?} needs {} px of halo ({needed:?}); \
                     build the plan with `PreparedSharpen::tile_plan`",
                    tile.index, tile.compute, tile.core, self.halo_px
                )));
            }
        }
        let depth = depth.max(1);
        let mut queue: std::collections::VecDeque<(Tile, Pending)> =
            std::collections::VecDeque::new();
        for &tile in plan.tiles() {
            cancel.check()?;
            if self.is_noop() {
                sink(tile, copy_core(image, tile.core))?;
                continue;
            }
            let pending = self.dispatch(ctx, image, tile.compute, tile.core, false)?;
            queue.push_back((tile, pending));
            if queue.len() >= depth
                && let Some((t, p)) = queue.pop_front()
            {
                sink(t, p.finish(ctx)?.0)?;
            }
        }
        while let Some((t, p)) = queue.pop_front() {
            sink(t, p.finish(ctx)?.0)?;
        }
        Ok(())
    }

    /// Encodes every pass for the `compute` rectangle of `image` and the `core` inside it, submits
    /// the work and returns the pending readback.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        compute: Rect,
        core: Rect,
        trace: bool,
    ) -> Result<Pending, GpuError> {
        let (w, h) = (compute.w, compute.h);
        // One workgroup covers GROUP x GROUP pixels; a side of more than `limit` workgroups is a
        // validation error that wgpu raises as a panic, so refuse it here (a 1 x N frame with
        // N > 65535 * 16 on the default limits).
        let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
        let groups = |side: usize| side.div_ceil(GROUP as usize);
        if groups(w) > limit || groups(h) > limit {
            return Err(GpuError::InvalidInput(format!(
                "output sharpening: a {w} x {h} compute rectangle needs more than {limit} \
                 workgroups of {GROUP} px on one side; use run_tiled with smaller tiles"
            )));
        }
        let n = w * h;
        let plane_bytes = (n * 4) as u64;
        let limiter = self.halo_kernel.is_some();
        let gain_planes = if self.stages.iter().any(|s| s.gain.is_some()) {
            2
        } else {
            0
        };
        let work_planes = if limiter { 4 } else { gain_planes };

        // Input: the compute rectangle's three planes, dense.
        let mut samples: Vec<f32> = Vec::with_capacity(3 * n);
        for c in 0..3 {
            let src = image.plane(c);
            for y in 0..h {
                let s = (compute.y + y) * image.width + compute.x;
                samples.extend_from_slice(&src[s..s + w]);
            }
        }
        let pool = ctx.pool();
        let img = pool.acquire("sharpen input", 3 * plane_bytes, STORAGE_IN)?;
        ctx.queue
            .write_buffer(&img, 0, bytemuck::cast_slice(&samples));
        drop(samples);

        let tiny = |label: &str| pool.acquire(label, 16, STORAGE_SCRATCH);
        let scratch = |label: &str, planes: usize| {
            if planes == 0 {
                tiny(label)
            } else {
                pool.acquire(label, planes as u64 * plane_bytes, STORAGE_SCRATCH)
            }
        };
        let (core_w, core_h) = (core.w, core.h);
        let out_bytes = (3 * core_w * core_h * 4) as u64;
        let bufs = Bufs {
            taps: &self.taps,
            img,
            lum: pool.acquire("sharpen luma", plane_bytes, STORAGE_PLANE)?,
            tmp: scratch("sharpen t", 1)?,
            hp: scratch("sharpen h", 1)?,
            dacc: pool.acquire("sharpen D", plane_bytes, STORAGE_PLANE)?,
            // A traced run reads Hacc back, so it exists (as zeros) even without a limiter.
            hacc: if limiter || trace {
                pool.acquire("sharpen Hacc", plane_bytes, STORAGE_PLANE)?
            } else {
                pool.acquire("sharpen Hacc (unused)", 16, STORAGE_PLANE)?
            },
            work_a: scratch("sharpen work A", work_planes)?,
            work_b: scratch("sharpen work B", work_planes)?,
            out: pool.acquire("sharpen out", out_bytes, STORAGE_OUT)?,
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("output sharpening"),
            });
        // Pooled buffers keep their old contents: the accumulators start from zero.
        encoder.clear_buffer(&bufs.dacc, 0, Some(plane_bytes));
        if limiter || trace {
            encoder.clear_buffer(&bufs.hacc, 0, Some(plane_bytes));
        }

        let weights = self.plan.luma_weights();
        let base = ParamsU {
            width: w as u32,
            height: h as u32,
            plane: n as u32,
            luma_r: weights[0],
            luma_g: weights[1],
            luma_b: weights[2],
            ..ParamsU::default()
        };
        let mut held: Vec<PooledBuffer> = Vec::new();
        let mut run = |id: KernelId, p: &ParamsU, z: u32| -> Result<(), GpuError> {
            let ub = uniform_buffer(ctx, "sharpen params", p)?;
            let pipeline = ctx.kernels().pipeline(ctx, id)?;
            let slots = pass_of(id).map_or(&[][..], |(_, s)| s);
            let entries: Vec<(u32, wgpu::BindingResource<'_>)> = slots
                .iter()
                .map(|&s| {
                    let resource = match s {
                        Slot::Params => ub.binding(),
                        Slot::Taps => bufs.taps.binding(),
                        Slot::Img => bufs.img.binding(),
                        Slot::Lum => bufs.lum.binding(),
                        Slot::Tmp => bufs.tmp.binding(),
                        Slot::Hp => bufs.hp.binding(),
                        Slot::Dacc => bufs.dacc.binding(),
                        Slot::Hacc => bufs.hacc.binding(),
                        Slot::WorkA => bufs.work_a.binding(),
                        Slot::WorkB => bufs.work_b.binding(),
                        Slot::Out => bufs.out.binding(),
                    };
                    (s as u32, resource)
                })
                .collect();
            let group = bind(ctx, &pipeline, id.name(), &entries);
            let (gw, gh) = if id == KernelId::OutputSharpenTransfer {
                (p.core_w, p.core_h)
            } else {
                (p.width, p.height)
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(id.name()),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gw.div_ceil(GROUP), gh.div_ceil(GROUP), z);
            drop(pass);
            held.push(ub);
            Ok(())
        };

        run(KernelId::OutputSharpenLuma, &base, 1)?;
        for st in &self.stages {
            let mut p = ParamsU {
                method: st.method,
                threshold: st.threshold,
                tap_off: st.kernel.offset,
                tap_n: st.kernel.count,
                ..base
            };
            run(KernelId::OutputSharpenHighpassRows, &p, 1)?;
            run(KernelId::OutputSharpenHighpassCols, &p, 1)?;
            if let Some(gain) = st.gain {
                p.use_ramp = u32::from(st.ramp);
                p.g0 = st.g0;
                p.inv = st.inv;
                run(KernelId::OutputSharpenGainTerms, &p, 1)?;
                let blur = ParamsU {
                    tap_off: gain.offset,
                    tap_n: gain.count,
                    planes: 2,
                    ..base
                };
                run(KernelId::OutputSharpenBlur, &blur, 2)?;
                let cols = ParamsU {
                    direction: 1,
                    ..blur
                };
                run(KernelId::OutputSharpenBlur, &cols, 2)?;
                p.use_gain = 1;
            }
            p.a_free = st.a_free;
            p.a_halo = st.a_halo;
            p.halo_acc = u32::from(st.a_halo > 0.0);
            run(KernelId::OutputSharpenAccumulate, &p, 1)?;
        }
        if let Some(halo) = self.halo_kernel {
            run(KernelId::OutputSharpenWindow, &base, 1)?;
            let blur = ParamsU {
                tap_off: halo.offset,
                tap_n: halo.count,
                planes: 4,
                ..base
            };
            run(KernelId::OutputSharpenBlur, &blur, 4)?;
            run(
                KernelId::OutputSharpenBlur,
                &ParamsU {
                    direction: 1,
                    ..blur
                },
                4,
            )?;
        }
        let (dx, dy) = (core.x - compute.x, core.y - compute.y);
        let transfer = ParamsU {
            core_x: dx as u32,
            core_y: dy as u32,
            core_w: core_w as u32,
            core_h: core_h as u32,
            limiter: u32::from(limiter),
            ..base
        };
        run(KernelId::OutputSharpenTransfer, &transfer, 1)?;

        let (readback, trace_readbacks) = if trace {
            // The output and the three diagnostic planes share one submission.
            let mut copies: Vec<(PooledBuffer, u64)> = Vec::new();
            for (src, bytes) in [
                (&bufs.out, out_bytes),
                (&bufs.lum, plane_bytes),
                (&bufs.dacc, plane_bytes),
                (&bufs.hacc, plane_bytes),
            ] {
                let staging = pool.acquire("sharpen readback", bytes, STAGING)?;
                encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
                copies.push((staging, bytes));
            }
            let index = ctx.queue.submit([encoder.finish()]);
            let mut it = copies
                .into_iter()
                .map(|(staging, bytes)| Readback::new(staging, bytes, index.clone()));
            let main = it
                .next()
                .ok_or_else(|| GpuError::Readback("no output readback".into()))?;
            (main, Some(it.collect::<Vec<_>>()))
        } else {
            (
                submit_with_readback(ctx, encoder, &bufs.out, out_bytes)?,
                None,
            )
        };
        let Bufs {
            img,
            lum,
            tmp,
            hp,
            dacc,
            hacc,
            work_a,
            work_b,
            out,
            ..
        } = bufs;
        held.extend([img, lum, tmp, hp, dacc, hacc, work_a, work_b, out]);
        Ok(Pending {
            readback,
            trace: trace_readbacks,
            width: core_w,
            height: core_h,
            trace_size: (w, h),
            _held: held,
        })
    }
}

/// The buffers of one tile's run. `taps` is shared by every tile.
struct Bufs<'a> {
    taps: &'a PooledBuffer,
    img: PooledBuffer,
    lum: PooledBuffer,
    tmp: PooledBuffer,
    hp: PooledBuffer,
    dacc: PooledBuffer,
    hacc: PooledBuffer,
    work_a: PooledBuffer,
    work_b: PooledBuffer,
    out: PooledBuffer,
}

/// A dispatch whose readback has not been collected yet.
struct Pending {
    readback: Readback,
    trace: Option<Vec<Readback>>,
    width: usize,
    height: usize,
    trace_size: (usize, usize),
    _held: Vec<PooledBuffer>,
}

impl Pending {
    fn finish(self, ctx: &GpuContext) -> Result<(ColorImage, Option<SharpenTrace>), GpuError> {
        let bytes = self.readback.finish(ctx)?;
        let data = unpack(
            OutKind::F32Planar,
            &bytes,
            self.width,
            self.height,
            self.width,
        );
        let image = ColorImage {
            width: self.width,
            height: self.height,
            data,
        };
        let trace = match self.trace {
            Some(readbacks) => {
                let mut planes = Vec::with_capacity(3);
                for r in readbacks {
                    planes.push(image_samples(StorageMode::F32, &r.finish(ctx)?));
                }
                let mut it = planes.into_iter();
                let (luma, free_delta, halo_delta) = match (it.next(), it.next(), it.next()) {
                    (Some(a), Some(b), Some(c)) => (a, b, c),
                    _ => return Err(GpuError::Readback("missing trace plane".into())),
                };
                Some(SharpenTrace {
                    width: self.trace_size.0,
                    height: self.trace_size.1,
                    luma,
                    free_delta,
                    halo_delta,
                })
            }
            None => None,
        };
        Ok((image, trace))
    }
}

fn check_image(image: &RgbImage) -> Result<(), GpuError> {
    if image.width == 0 || image.height == 0 {
        return Err(GpuError::InvalidInput(
            "output sharpening: the image has no pixels".into(),
        ));
    }
    if image.data.len() != 3 * image.width * image.height {
        return Err(GpuError::InvalidInput(format!(
            "output sharpening: the buffer holds {} samples but {} x {} pixels need {}",
            image.data.len(),
            image.width,
            image.height,
            3 * image.width * image.height
        )));
    }
    Ok(())
}

/// A planar `f32` result as an [`RgbImage`] (a whole frame).
fn into_rgb(image: ColorImage) -> Result<RgbImage, GpuError> {
    match image.data {
        PixelData::F32Planar(v) => {
            RgbImage::from_planar(image.width, image.height, v).map_err(Into::into)
        }
        other => Err(GpuError::Readback(format!(
            "expected f32 planes, got {other:?}"
        ))),
    }
}

/// The `core` of `image` as planar `f32` (what a no-op run hands the sink).
fn copy_core(image: &RgbImage, core: Rect) -> ColorImage {
    let mut v = Vec::with_capacity(3 * core.w * core.h);
    for c in 0..3 {
        let plane = image.plane(c);
        for y in 0..core.h {
            let s = (core.y + y) * image.width + core.x;
            v.extend_from_slice(&plane[s..s + core.w]);
        }
    }
    ColorImage {
        width: core.w,
        height: core.h,
        data: PixelData::F32Planar(v),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::kernels::shader_source;

    /// Every output sharpening pass, as registered in [`KernelId`].
    fn passes() -> Vec<KernelId> {
        let all: Vec<KernelId> = KernelId::ALL
            .into_iter()
            .filter(|id| id.name().starts_with("output_sharpen"))
            .collect();
        assert_eq!(all.len(), 8, "eight passes are registered");
        all
    }

    fn module_of(id: KernelId) -> (naga::Module, naga::valid::ModuleInfo) {
        let source = shader_source(StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        (module, info)
    }

    #[test]
    fn each_pass_binds_exactly_the_buffers_the_host_binds() {
        // The pipelines use an auto-generated layout holding the bindings the entry point touches; a
        // bind group with another set would be rejected, but only at run time on a GPU. This pins
        // the table `pass_of` against the shader, offline.
        for id in passes() {
            let (module, info) = module_of(id);
            assert_eq!(module.entry_points.len(), 1, "{id:?}: one entry point");
            assert_eq!(module.entry_points[0].name, "main");
            assert_eq!(module.entry_points[0].workgroup_size, [GROUP, GROUP, 1]);
            let used = info.get_entry_point(0);
            let mut bindings = BTreeSet::new();
            for (handle, var) in module.global_variables.iter() {
                if let Some(b) = &var.binding
                    && !used[handle].is_empty()
                {
                    assert_eq!(b.group, 0);
                    bindings.insert(b.binding);
                }
            }
            let (_, slots) = pass_of(id).expect("registered pass");
            let expected: BTreeSet<u32> = slots.iter().map(|&s| s as u32).collect();
            assert_eq!(bindings, expected, "{id:?}: bindings the shader uses");
        }
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_params_u() {
        let (module, _) = module_of(KernelId::OutputSharpenLuma);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("sp"))
            .expect("the uniform `sp`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`sp` is not a struct");
        };
        let fields: [(&str, usize); 24] = [
            ("width", offset_of!(ParamsU, width)),
            ("height", offset_of!(ParamsU, height)),
            ("plane", offset_of!(ParamsU, plane)),
            ("direction", offset_of!(ParamsU, direction)),
            ("core_x", offset_of!(ParamsU, core_x)),
            ("core_y", offset_of!(ParamsU, core_y)),
            ("core_w", offset_of!(ParamsU, core_w)),
            ("core_h", offset_of!(ParamsU, core_h)),
            ("tap_off", offset_of!(ParamsU, tap_off)),
            ("tap_n", offset_of!(ParamsU, tap_n)),
            ("planes", offset_of!(ParamsU, planes)),
            ("method", offset_of!(ParamsU, method)),
            ("luma_r", offset_of!(ParamsU, luma_r)),
            ("luma_g", offset_of!(ParamsU, luma_g)),
            ("luma_b", offset_of!(ParamsU, luma_b)),
            ("threshold", offset_of!(ParamsU, threshold)),
            ("a_free", offset_of!(ParamsU, a_free)),
            ("a_halo", offset_of!(ParamsU, a_halo)),
            ("g0", offset_of!(ParamsU, g0)),
            ("inv", offset_of!(ParamsU, inv)),
            ("use_gain", offset_of!(ParamsU, use_gain)),
            ("use_ramp", offset_of!(ParamsU, use_ramp)),
            ("halo_acc", offset_of!(ParamsU, halo_acc)),
            ("limiter", offset_of!(ParamsU, limiter)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<ParamsU>());
    }

    #[test]
    fn the_shader_is_f32_only_in_every_storage_mode() {
        // Planes are f32 whatever the storage mode (spec 5.4: the limiter compares luma values), so
        // the text has no f16 and no accessor expansion beyond the generated prelude.
        let body = include_str!("../shaders/output_sharpen.wgsl");
        for token in ["<f16>", "f16(", "enable f16", "ST_ELEM", "//@ACCESSORS"] {
            assert!(!body.contains(token), "the shader text contains {token}");
        }
        for mode in [
            StorageMode::F32,
            StorageMode::F16Packed,
            StorageMode::F16Native,
        ] {
            for id in passes() {
                let source = shader_source(mode, id);
                assert!(source.contains("fn main("), "{id:?} {mode:?}");
                assert!(!source.contains("array<f16>") && !source.contains("2x16float"));
            }
        }
    }

    #[test]
    fn ramp_constants_follow_the_cpu_reference() {
        // `g0 = T ((k_c + k_{c+1}) / 2) / left` and `inv = 0.25 / g0`; a collapsed kernel has no ramp.
        let k = crate::cpu::output_sharpen::gaussian_kernel(1.0, 13);
        let c = k.len() / 2;
        let left: f64 = k[..c].iter().sum();
        let t = 4.0f32 / 255.0;
        let g0 = f64::from(t) * ((k[c] + k[c + 1]) * 0.5) / left;
        assert_eq!(ramp_constants(&k, t), (true, g0 as f32, (0.25 / g0) as f32));
        let collapsed = [0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(ramp_constants(&collapsed, t), (false, 0.0, 0.0));
    }
}
