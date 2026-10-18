//! Output sharpening on the GPU: the WGSL port of
//! [`crate::cpu::output_sharpen`], specified in `docs/algorithms/04-output-sharpening.md`.
//!
//! [`PreparedSharpen::new`] derives the stages of an [`OutputSharpen`] for an output scale with the
//! CPU reference's own plan code, rounds the CPU's `f64` kernel taps to `f32` and uploads them once.
//! Two layers use it:
//!
//! * **Device in, device out** ([`encode`](PreparedSharpen::encode),
//!   [`encode_rgba8`](PreparedSharpen::encode_rgba8)): record the passes into a
//!   [`PassList`]. The input is a [`GpuImage`] (three planes, optionally a
//!   fourth one that marks the letterbox) and a `window` of it, the rectangle the stage reads: the
//!   photo's bounding box, where the stage mirrors at its edge, exactly as the CPU does. The `core`
//!   inside the window is what is written, either as three f32 planes of a destination image or, fused
//!   with C13, the dither and the letterbox, as RGBA8 words.
//! * **Host wrappers** ([`run`](PreparedSharpen::run), [`run_traced`](PreparedSharpen::run_traced),
//!   [`run_tiled`](PreparedSharpen::run_tiled)): upload a planar host image, call the device entry
//!   point, read the core back. They are thin layers on the first, so the two cannot differ
//!   (`tests/gpu_output_sharpen.rs` checks the fused tail against the chained pair as well).
//!
//! # Passes
//!
//! One compute dispatch per pass over `output_sharpen.wgsl`, each with its own pipeline (the passes
//! are separate [`KernelId`]s that share one source file). For each compute rectangle:
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
//!   intermediates in f16 in the legacy modes, but the halo limiter compares luma values for equality
//!   and order (spec 5.4), and an f16 round trip would turn most of its ties into different counts.
//!   Every plane (`L`, `t`, `h`, `D`, `Hacc`, gain terms, envelope maps) and the input and output
//!   pixels are `f32` buffers.
//! * **Accumulation is `f32`.** WGSL has no `f64`, so convolutions, window sums and the envelope
//!   formula accumulate in `f32` (taps left to right, window row by row), the model of the C++
//!   reference's `Accumulation::F32` mode, where the CPU port uses `f64`. The taps are the CPU's `f64`
//!   kernels rounded to `f32`. The difference form of the high-pass and of the window sums is kept, so
//!   a flat region gives exact zeros, as on the CPU.
//! * **No `erf` on the GPU.** The kernels, the threshold rescale and the gradient constants of method
//!   2 are computed on the CPU in `f64` (the same code paths as the CPU reference) and passed as taps
//!   and uniforms.
//! * **Rank counts.** The limiter's counts depend on exact `f32` values of `L` and `S = L + Hacc`.
//!   `L` has the CPU's bits on every device (the products are pinned and the square root is the
//!   shader's own, see below; checked in `tests/gpu_output_sharpen.rs`); `S` differs from the CPU's
//!   wherever `f32` and `f64` accumulation round the high-pass differently, which flips a count where the content has ties in
//!   exact arithmetic (noiseless bars, checkerboards, diagonals: spec 5.4). On noisy and
//!   photographic content no count flips at frame sizes up to 150 x 113 and 2 to 4 of 2.8 million
//!   pixels flip at 2048 x 1365; on noiseless periodic content up to 19 % of the pixels do. The
//!   numbers, and the proof that the kernel is bit-identical to the C++ reference's
//!   `Accumulation::F32` mode (so every difference from the CPU is the accumulation precision), are in
//!   `docs/algorithms/04-output-sharpening.md` section 10.
//!
//! # Fast-math (Metal, K8)
//!
//! Metal compiles with fast-math by default and `wgpu` 30 has no switch for it (`wgpu-hal` sets only
//! the language version), so the kernel is written so that fast-math cannot change what matters:
//!
//! * **Identities are exact by construction.** A stage whose kernel has collapsed to one tap has an
//!   exactly zero delta (every side tap is 0, the centre term is `L - L`), so [`PreparedSharpen::new`]
//!   drops it; a setup with no stage left copies its input, bit for bit, on every device. The
//!   transfer takes the ratio `(L + D) / max(L, floor)` as exactly 1 when `D == 0`.
//! * **Nothing the limiter compares can move by an ulp.** The limiter counts window samples of `L`
//!   and `S = L + Hacc` by exact comparisons, so one ulp at a near-tie flips a count and moves a pixel
//!   by up to 2e-3. On Metal at b6f2e4b three things did that: its `sqrt` left 27 to 37 % of the lumas
//!   1 or 2 ulp off, its division is the dividend times an approximate reciprocal, and it fused the
//!   blurs' products into multiply-adds. So every product that meets a sum, every partial sum and
//!   every difference whose rounding matters is ORed with a uniform that is 0 at run time and unknown
//!   to the compiler (`opaque` in `output_sharpen.wgsl`): one operation per barrier, nothing left to
//!   fuse or re-associate. An OR and not an XOR: an XOR applied twice cancels, and fast-math folds the
//!   `0 + x` that starts a running sum, which put two barriers in a row and left a product bare. The
//!   square roots are integer square roots and every quotient is the device's estimate made exact
//!   with an integer remainder (`sharpen_sqrt`, `sharpen_div`; both correctly rounded whatever the
//!   device's `sqrt` and `/` do; the device's own only for zero, subnormal or non-finite operands).
//!   IEEE arithmetic is unchanged, so lavapipe gives the bits it gave before.
//! * **Evidence without a Mac.** The MSL that naga writes for these passes (the text wgpu hands to
//!   Apple's compiler), compiled with clang `-O2` and `-O3 -ffast-math -ffp-contract=fast -mfma`, also
//!   with `-mrecip=all` (an approximate reciprocal and square root, as on a GPU), gives lavapipe's bits
//!   on all 67 cases of the four failing tests (the 16 vectors, 28 noisy and photographic frames, 21
//!   flip and transpose runs, the two 2048 x 1365 limiter cases); the MSL of the shader before the
//!   change reproduces Metal's 1.959e-3 on the 2048 px export exactly. The tests therefore hold
//!   hardware adapters to the lavapipe bounds (`tests/gpu_output_sharpen.rs`), and
//!   `docs/17-gpu.md` section 13 has the numbers.
//!
//! The cost: the luma and the gradient take a 26-step integer square root, and the window, the
//! accumulation and the transfer take up to eleven exact divisions per pixel (about 25 operations
//! each); on lavapipe the tiled 2048 x 1365 frame took 4.2 s instead of 3.3 s.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ym_color::encode::OutputSpace;

use super::color::{ColorImage, dither_table};
use super::shader::KernelId;
use super::util::{OutKind, PixelData, STORAGE_IN, even, unpack};
use crate::cpu::output_sharpen::{Method, OutputSharpen, SharpenPlan};
use crate::cpu::tiling::{Halo, Rect};
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::stream::{PassList, Pending, ReadId};
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

/// The size of the largest storage binding a run over the `compute` rectangle makes (K4 of
/// `docs/17-gpu.md`): four `f32` planes, either the limiter's working buffers or a source image
/// with a mask plane.
pub fn max_binding_bytes(compute: Rect) -> u64 {
    16 * even(compute.w) as u64 * compute.h as u64
}

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
    Dither = 11,
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
    use Slot::{Dacc, Dither, Hacc, Hp, Img, Lum, Out, Params, Taps, Tmp, WorkA, WorkB};
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
            &[Params, Img, Lum, Dacc, Hacc, WorkA, Out, Dither],
        ),
        _ => return None,
    })
}

/// The kernel body of an output sharpening pass: the shared WGSL, the C13 helpers and the pass's
/// `main`. Called by [`super::shader::shader_source`]; any other id is a programming error and gets
/// an empty body.
pub(super) fn body(id: KernelId) -> String {
    let Some((function, _)) = pass_of(id) else {
        return String::new();
    };
    let mut s = String::from(include_str!("../shaders/output_sharpen.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/c13.wgsl"));
    s.push_str(&entry(function));
    s
}

/// The uniform block of `output_sharpen.wgsl` (`SharpenParams`): the same fields in the same order,
/// 44 scalars of 4 bytes (a multiple of 16 bytes). Every pass reads the fields it needs.
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
    in_stride: u32,
    in_plane: u32,
    in_x: u32,
    in_y: u32,
    out_stride: u32,
    out_plane: u32,
    out_x: u32,
    out_y: u32,
    out_kind: u32,
    space: u32,
    dither: u32,
    has_mask: u32,
    org_x: u32,
    org_y: u32,
    bg: u32,
    copy_only: u32,
    /// Always 0: the shader ORs the terms whose rounding matters with it so that no compiler can
    /// re-associate or fuse them (`opaque` in `output_sharpen.wgsl`). Nothing sets it.
    zero: u32,
    /// The halo limiter's relative tie tolerance (`SharpenPlan::halo_tie_tolerance`).
    tie_rel: f32,
    pad: [u32; 2],
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

/// The C13 tail fused into the last pass of the sharpening ([`PreparedSharpen::encode_rgba8`]): the
/// transfer function of the output, the blue-noise dither at the pixel's place in the picture, and
/// the colour of the letterbox where the source's mask plane says the pixel is outside the photo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusedC13 {
    /// The output colour space (its transfer function).
    pub space: OutputSpace,
    /// Whether the 8-bit codes are dithered; otherwise they are rounded.
    pub dither: bool,
    /// The absolute position of the core's pixel (0, 0) in the picture, which selects the dither
    /// texel, so tiles and the whole picture agree.
    pub origin: (usize, usize),
    /// The letterbox colour (RGB), painted where the mask is set.
    pub background: [u8; 3],
}

impl FusedC13 {
    /// `1` for the BT.1886 transfer of Rec.2020, `0` for sRGB (also Display P3).
    fn transfer_code(self) -> u32 {
        u32::from(self.space == OutputSpace::Rec2020)
    }

    fn background_word(self) -> u32 {
        u32::from(self.background[0])
            | (u32::from(self.background[1]) << 8)
            | (u32::from(self.background[2]) << 16)
            | 0xff00_0000
    }
}

/// An output sharpening setup uploaded to the GPU: immutable, reusable for any number of tiles and
/// images of the same scale.
#[derive(Debug)]
pub struct PreparedSharpen {
    plan: SharpenPlan,
    /// The stages that change something (a collapsed kernel has an exactly zero delta and is
    /// dropped; see the module notes).
    stages: Vec<StageGpu>,
    /// The halo limiter's kernel, present when a remaining stage sends part of its delta to it.
    halo_kernel: Option<TapRange>,
    halo_px: usize,
    taps: Arc<PooledBuffer>,
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

/// Whether a kernel has collapsed to its centre tap: every other tap is exactly 0.
fn is_collapsed(kernel: &[f64]) -> bool {
    let c = kernel.len() / 2;
    kernel.iter().enumerate().all(|(j, &t)| j == c || t == 0.0)
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

/// Where a run writes its core.
enum Target<'a> {
    /// Three f32 planes of an image, the core's (0, 0) at `at`.
    Planes {
        dst: &'a GpuImage,
        at: (usize, usize),
    },
    /// RGBA8 words, `stride` per row, the core's (0, 0) at `at`, C13 fused.
    Rgba8 {
        dst: &'a PooledBuffer,
        stride: usize,
        at: (usize, usize),
        tail: FusedC13,
    },
}

/// The scratch planes of a run that tests read back: `(luma, free delta, halo delta)`.
type TraceIds = (ReadId, ReadId, ReadId);

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
            if is_collapsed(&k) {
                // High-pass 0 exactly, so d = 0, s = 0 and both accumulators gain +0: the stage
                // is the identity in every arithmetic, and dropping it keeps that exact under
                // fast-math (module notes).
                continue;
            }
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
        // The limiter exists when a stage that remains sends part of its delta to it. (The plan's
        // `halo_enabled` also counts a dropped stage, whose limited share is exactly zero and which
        // therefore leaves the limiter with nothing to do.)
        let halo_kernel = stages
            .iter()
            .any(|s| s.a_halo > 0.0)
            .then(|| push_taps(&mut taps, &plan.halo_kernel()));
        if taps.is_empty() {
            taps.push(0.0);
        }
        let taps_buf = Arc::new(crate::kernels::util::storage_buffer(
            ctx,
            "sharpen taps",
            bytemuck::cast_slice(&taps),
        )?);
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

    /// Whether no stage changes anything (amount or radius too small, or every kernel collapsed): a
    /// run copies its input, bit for bit.
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

    // ---- device in, device out ----------------------------------------------------------------

    /// Records the passes that sharpen the `window` of `src` and write its `core` as three f32
    /// planes into `dst`, the core's pixel (0, 0) at `dst_at`.
    ///
    /// `src` has three planes (a fourth one, a letterbox mask, is ignored here). The stage reads
    /// the whole `window` and mirrors at its edge, so the window is the photo's own pixels: the
    /// caller passes the bounding box of the pixels inside the photo, grown by nothing (a tile of a
    /// larger picture passes core plus [`halo_px`](Self::halo_px)). `core` lies inside `window`, both
    /// in pixels of `src`. Nothing is read back.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when a rectangle is empty or outside its image, when `src` has
    /// fewer than three planes or `dst` is not a three- or four-plane image, or when a side needs
    /// more workgroups than the device allows; the pool's errors for the scratch planes.
    pub fn encode(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        core: Rect,
        dst: &GpuImage,
        dst_at: (usize, usize),
    ) -> Result<(), GpuError> {
        if dst.planes < 3 {
            return Err(GpuError::InvalidInput(format!(
                "output sharpening writes three planes, the destination has {}",
                dst.planes
            )));
        }
        if dst_at.0 + core.w > dst.width || dst_at.1 + core.h > dst.height {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} core at {dst_at:?} does not fit the {} x {} destination",
                core.w, core.h, dst.width, dst.height
            )));
        }
        self.encode_to(
            passes,
            src,
            window,
            core,
            &Target::Planes { dst, at: dst_at },
            None,
        )
        .map(|_| ())
    }

    /// Like [`encode`](Self::encode), with C13, the dither and the letterbox fused into the last
    /// pass: the core is written as RGBA8 words (`stride` words per row, `dst` of at least
    /// `stride * (at.1 + core.h)` words), pixel (0, 0) of the core at `dst_at`. Where `src` has a
    /// fourth plane and it is not 0 the pixel is `tail.background` instead. Equal to
    /// [`encode`](Self::encode) followed by the encode kernel, bit for bit.
    ///
    /// # Errors
    ///
    /// As [`encode`](Self::encode); [`GpuError::InvalidInput`] when `dst` is too small.
    #[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
    pub fn encode_rgba8(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        core: Rect,
        dst: &PooledBuffer,
        stride: usize,
        dst_at: (usize, usize),
        tail: FusedC13,
    ) -> Result<(), GpuError> {
        let words = (dst_at.1 + core.h) as u64 * stride as u64;
        if dst_at.0 + core.w > stride || dst.size() < words * 4 {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} core at {dst_at:?} does not fit {stride} words per row in a {} byte buffer",
                core.w,
                core.h,
                dst.size()
            )));
        }
        self.encode_to(
            passes,
            src,
            window,
            core,
            &Target::Rgba8 {
                dst,
                stride,
                at: dst_at,
                tail,
            },
            None,
        )
        .map(|_| ())
    }

    /// The common recording: validates the rectangles, allocates the scratch planes, records the
    /// passes. With `trace` the three diagnostic planes are asked for from the list.
    fn encode_to(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        core: Rect,
        target: &Target<'_>,
        trace: Option<()>,
    ) -> Result<Option<TraceIds>, GpuError> {
        let ctx = passes.context();
        if src.planes < 3 {
            return Err(GpuError::InvalidInput(format!(
                "output sharpening reads three planes, the source has {}",
                src.planes
            )));
        }
        if window.is_empty() || core.is_empty() {
            return Err(GpuError::InvalidInput(
                "output sharpening: the window and the core must not be empty".into(),
            ));
        }
        if window.x1() > src.width || window.y1() > src.height {
            return Err(GpuError::InvalidInput(format!(
                "the window {window:?} is outside the {} x {} source",
                src.width, src.height
            )));
        }
        if !window.contains(&core) {
            return Err(GpuError::InvalidInput(format!(
                "the core {core:?} is not inside the window {window:?}"
            )));
        }
        let (w, h) = (window.w, window.h);
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
        let noop = self.is_noop();
        let limiter = self.halo_kernel.is_some();
        let tracing = trace.is_some() && !noop;
        let gain_planes = if self.stages.iter().any(|s| s.gain.is_some()) {
            2
        } else {
            0
        };
        let work_planes = if limiter { 4 } else { gain_planes };

        let pool = ctx.pool();
        let dither = dither_table(ctx)?;
        let (lum, tmp, hp, dacc, hacc, work_a, work_b);
        if noop {
            // The transfer alone copies the core: it binds these, nothing reads them.
            let t = passes.scratch("sharpen unused", 16)?;
            (lum, tmp, hp, dacc, hacc, work_a, work_b) = (
                Arc::clone(&t),
                Arc::clone(&t),
                Arc::clone(&t),
                Arc::clone(&t),
                Arc::clone(&t),
                Arc::clone(&t),
                t,
            );
        } else {
            // The planes the host clears or reads back carry COPY_DST / COPY_SRC.
            lum = passes.keep(pool.acquire("sharpen luma", plane_bytes, STORAGE_PLANE)?);
            tmp = passes.scratch("sharpen t", plane_bytes)?;
            hp = passes.scratch("sharpen h", plane_bytes)?;
            dacc = passes.keep(pool.acquire("sharpen D", plane_bytes, STORAGE_PLANE)?);
            // A traced run reads Hacc back, so it exists (as zeros) even without a limiter.
            let hacc_bytes = if limiter || tracing { plane_bytes } else { 16 };
            hacc = passes.keep(pool.acquire("sharpen Hacc", hacc_bytes, STORAGE_PLANE)?);
            work_a =
                passes.scratch("sharpen work A", (work_planes as u64 * plane_bytes).max(16))?;
            work_b =
                passes.scratch("sharpen work B", (work_planes as u64 * plane_bytes).max(16))?;
        }

        let weights = self.plan.luma_weights();
        let (core_w, core_h) = (core.w, core.h);
        let (dx, dy) = (core.x - window.x, core.y - window.y);
        let mut base = ParamsU {
            width: w as u32,
            height: h as u32,
            plane: n as u32,
            luma_r: weights[0],
            luma_g: weights[1],
            luma_b: weights[2],
            tie_rel: self.plan.halo_tie_tolerance(),
            in_stride: src.stride as u32,
            in_plane: src.plane_len() as u32,
            in_x: window.x as u32,
            in_y: window.y as u32,
            has_mask: u32::from(src.planes == 4),
            ..ParamsU::default()
        };
        match target {
            Target::Planes { dst, at } => {
                base.out_stride = dst.stride as u32;
                base.out_plane = dst.plane_len() as u32;
                base.out_x = at.0 as u32;
                base.out_y = at.1 as u32;
                base.out_kind = 0;
            }
            Target::Rgba8 {
                stride, at, tail, ..
            } => {
                base.out_stride = *stride as u32;
                base.out_x = at.0 as u32;
                base.out_y = at.1 as u32;
                base.out_kind = 1;
                base.space = tail.transfer_code();
                base.dither = u32::from(tail.dither);
                base.org_x = tail.origin.0 as u32;
                base.org_y = tail.origin.1 as u32;
                base.bg = tail.background_word();
            }
        }
        let out_binding: &PooledBuffer = match target {
            Target::Planes { dst, .. } => &dst.buf,
            Target::Rgba8 { dst, .. } => dst,
        };
        let src_buf = &src.buf;
        let taps = &self.taps;
        let dither_buf = &dither;

        let run = |passes: &mut PassList<'_>,
                   id: KernelId,
                   p: &ParamsU,
                   z: u32|
         -> Result<(), GpuError> {
            let ub = passes.uniform("sharpen params", p)?;
            let slots = pass_of(id).map_or(&[][..], |(_, s)| s);
            let entries: Vec<(u32, wgpu::BindingResource<'_>)> = slots
                .iter()
                .map(|&s| {
                    let resource = match s {
                        Slot::Params => ub.binding(),
                        Slot::Taps => taps.binding(),
                        Slot::Img => src_buf.binding(),
                        Slot::Lum => lum.binding(),
                        Slot::Tmp => tmp.binding(),
                        Slot::Hp => hp.binding(),
                        Slot::Dacc => dacc.binding(),
                        Slot::Hacc => hacc.binding(),
                        Slot::WorkA => work_a.binding(),
                        Slot::WorkB => work_b.binding(),
                        Slot::Out => out_binding.binding(),
                        Slot::Dither => dither_buf.binding(),
                    };
                    (s as u32, resource)
                })
                .collect();
            let (gw, gh) = if id == KernelId::OutputSharpenTransfer {
                (p.core_w, p.core_h)
            } else {
                (p.width, p.height)
            };
            passes.dispatch(
                id,
                id.name(),
                &entries,
                (gw.div_ceil(GROUP), gh.div_ceil(GROUP), z),
            )
        };

        if !noop {
            // Pooled buffers keep their old contents: the accumulators start from zero.
            passes.encoder().clear_buffer(&dacc, 0, Some(plane_bytes));
            if limiter || tracing {
                passes.encoder().clear_buffer(&hacc, 0, Some(plane_bytes));
            }
            run(passes, KernelId::OutputSharpenLuma, &base, 1)?;
            for st in &self.stages {
                let mut p = ParamsU {
                    method: st.method,
                    threshold: st.threshold,
                    tap_off: st.kernel.offset,
                    tap_n: st.kernel.count,
                    ..base
                };
                run(passes, KernelId::OutputSharpenHighpassRows, &p, 1)?;
                run(passes, KernelId::OutputSharpenHighpassCols, &p, 1)?;
                if let Some(gain) = st.gain {
                    p.use_ramp = u32::from(st.ramp);
                    p.g0 = st.g0;
                    p.inv = st.inv;
                    run(passes, KernelId::OutputSharpenGainTerms, &p, 1)?;
                    let blur = ParamsU {
                        tap_off: gain.offset,
                        tap_n: gain.count,
                        planes: 2,
                        ..base
                    };
                    run(passes, KernelId::OutputSharpenBlur, &blur, 2)?;
                    let cols = ParamsU {
                        direction: 1,
                        ..blur
                    };
                    run(passes, KernelId::OutputSharpenBlur, &cols, 2)?;
                    p.use_gain = 1;
                }
                p.a_free = st.a_free;
                p.a_halo = st.a_halo;
                p.halo_acc = u32::from(st.a_halo > 0.0);
                run(passes, KernelId::OutputSharpenAccumulate, &p, 1)?;
            }
            if let Some(halo) = self.halo_kernel {
                run(passes, KernelId::OutputSharpenWindow, &base, 1)?;
                let blur = ParamsU {
                    tap_off: halo.offset,
                    tap_n: halo.count,
                    planes: 4,
                    ..base
                };
                run(passes, KernelId::OutputSharpenBlur, &blur, 4)?;
                run(
                    passes,
                    KernelId::OutputSharpenBlur,
                    &ParamsU {
                        direction: 1,
                        ..blur
                    },
                    4,
                )?;
            }
        }
        let transfer = ParamsU {
            core_x: dx as u32,
            core_y: dy as u32,
            core_w: core_w as u32,
            core_h: core_h as u32,
            limiter: u32::from(limiter),
            copy_only: u32::from(noop),
            ..base
        };
        run(passes, KernelId::OutputSharpenTransfer, &transfer, 1)?;

        if tracing {
            let l = passes.read(&lum, 0, plane_bytes)?;
            let d = passes.read(&dacc, 0, plane_bytes)?;
            let hh = passes.read(&hacc, 0, plane_bytes)?;
            return Ok(Some((l, d, hh)));
        }
        Ok(None)
    }
}

// ---- host wrappers ----------------------------------------------------------------------------

/// A host run in flight: the submission, what to read back, and the destination image (its
/// metadata decodes the readback).
struct PendingRun {
    pending: Pending,
    out: ReadId,
    dst: GpuImage,
    trace: Option<(TraceIds, (usize, usize))>,
}

impl PendingRun {
    fn finish(self) -> Result<(ColorImage, Option<SharpenTrace>), GpuError> {
        let done = self.pending.finish()?;
        let data = self.dst.take(&done, self.out)?;
        let image = ColorImage {
            width: self.dst.width,
            height: self.dst.height,
            data: PixelData::F32Planar(data),
        };
        let trace = self.trace.map(|((l, d, h), (w, ht))| {
            let plane = |id: ReadId| -> Vec<f32> {
                done.read(id)
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            SharpenTrace {
                width: w,
                height: ht,
                luma: plane(l),
                free_delta: plane(d),
                halo_delta: plane(h),
            }
        });
        Ok((image, trace))
    }
}

impl PreparedSharpen {
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
        let done = self.start(ctx, image, whole, whole, false)?.finish()?;
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
        let (image_out, trace) = self.start(ctx, image, whole, whole, true)?.finish()?;
        let trace =
            trace.ok_or_else(|| GpuError::Readback("the trace was not read back".into()))?;
        Ok((into_rgb(image_out)?, trace))
    }

    /// Sharpens `image` and encodes it with the fused C13 tail ([`FusedC13`]) into RGBA8, one set of
    /// dispatches over the whole frame (the letterbox mask is absent in a host image, so no pixel is
    /// painted). The test of the fused tail against the chained pair uses it.
    ///
    /// # Errors
    ///
    /// The errors of [`run`](Self::run).
    pub fn run_rgba8(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        tail: FusedC13,
    ) -> Result<ColorImage, GpuError> {
        check_image(image)?;
        let (w, h) = (image.width, image.height);
        let src = GpuImage::upload_rgb(ctx, "sharpen input", image)?;
        let bytes = (w * h * 4) as u64;
        let dst = ctx
            .pool()
            .acquire("sharpen rgba8", bytes, super::util::STORAGE_OUT)?;
        let mut passes = PassList::new(ctx, "output sharpening (fused C13)");
        let whole = Rect::whole(w, h);
        self.encode_rgba8(&mut passes, &src, whole, whole, &dst, w, (0, 0), tail)?;
        let id = passes.read(&dst, 0, bytes)?;
        let _src = passes.keep(src.buf);
        let done = passes.submit()?.finish()?;
        Ok(ColorImage {
            width: w,
            height: h,
            data: unpack(OutKind::Rgba8, done.read(id), w, h, w),
        })
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
        let mut queue: std::collections::VecDeque<(Tile, PendingRun)> =
            std::collections::VecDeque::new();
        for &tile in plan.tiles() {
            cancel.check()?;
            if self.is_noop() {
                sink(tile, copy_core(image, tile.core))?;
                continue;
            }
            let pending = self.start(ctx, image, tile.compute, tile.core, false)?;
            queue.push_back((tile, pending));
            if queue.len() >= depth
                && let Some((t, p)) = queue.pop_front()
            {
                sink(t, p.finish()?.0)?;
            }
        }
        while let Some((t, p)) = queue.pop_front() {
            sink(t, p.finish()?.0)?;
        }
        Ok(())
    }

    /// Uploads the `compute` rectangle of `image`, records the passes for the `core` inside it
    /// (both in pixels of `image`), submits them and returns the run in flight.
    fn start(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        compute: Rect,
        core: Rect,
        trace: bool,
    ) -> Result<PendingRun, GpuError> {
        let part;
        let upload: &RgbImage = if compute == Rect::whole(image.width, image.height) {
            image
        } else {
            part = image.extract(compute);
            &part
        };
        let src = GpuImage::upload_rgb(ctx, "sharpen input", upload)?;
        let dst = GpuImage::alloc(
            ctx,
            "sharpen out",
            core.w,
            core.h,
            3,
            Placement::whole(core.w, core.h),
        )?;
        let mut passes = PassList::new(ctx, "output sharpening");
        let window = Rect::whole(compute.w, compute.h);
        let ids = self.encode_to(
            &mut passes,
            &src,
            window,
            core.relative_to(&compute),
            &Target::Planes {
                dst: &dst,
                at: (0, 0),
            },
            trace.then_some(()),
        )?;
        let out = dst.encode_read(&mut passes)?;
        let _src = passes.keep(src.buf);
        let pending = passes.submit()?;
        Ok(PendingRun {
            pending,
            out,
            dst,
            trace: ids.map(|i| (i, (compute.w, compute.h))),
        })
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
    use crate::device::StorageMode;
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
        let fields: [(&str, usize); 44] = [
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
            ("in_stride", offset_of!(ParamsU, in_stride)),
            ("in_plane", offset_of!(ParamsU, in_plane)),
            ("in_x", offset_of!(ParamsU, in_x)),
            ("in_y", offset_of!(ParamsU, in_y)),
            ("out_stride", offset_of!(ParamsU, out_stride)),
            ("out_plane", offset_of!(ParamsU, out_plane)),
            ("out_x", offset_of!(ParamsU, out_x)),
            ("out_y", offset_of!(ParamsU, out_y)),
            ("out_kind", offset_of!(ParamsU, out_kind)),
            ("space", offset_of!(ParamsU, space)),
            ("dither", offset_of!(ParamsU, dither)),
            ("has_mask", offset_of!(ParamsU, has_mask)),
            ("org_x", offset_of!(ParamsU, org_x)),
            ("org_y", offset_of!(ParamsU, org_y)),
            ("bg", offset_of!(ParamsU, bg)),
            ("copy_only", offset_of!(ParamsU, copy_only)),
            ("zero", offset_of!(ParamsU, zero)),
            ("tie_rel", offset_of!(ParamsU, tie_rel)),
            ("pad1", offset_of!(ParamsU, pad)),
            ("pad2", offset_of!(ParamsU, pad) + 4),
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
