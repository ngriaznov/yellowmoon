//! R8 diffraction deconvolution on the GPU (RA-09): the WGSL port of
//! [`crate::cpu::sharpen::diffraction`], specified in
//! `docs/algorithms/03-diffraction-deconvolution.md` (sections 3.4 to 3.12; section 5 lists the GPU
//! notes).
//!
//! [`PreparedDiffraction::new`] takes the CPU planner's result (a [`DiffractionPlan`]: the PSF
//! kernel, `sigma_G`, the radii, the noise constants, the gains and the luma weights) and uploads the
//! kernel and the Gaussian taps once. [`run`](PreparedDiffraction::run) sharpens a whole planar image and
//! [`run_tiled`](PreparedDiffraction::run_tiled) does it tile by tile with the exact halo of the plan
//! (`10 + R + r_G`, at most 25 px). Nothing here is wired into the render path: the raw stages still
//! run the CPU reference, and wiring this one is a later decision.
//!
//! # What runs where
//!
//! The planner (159 x 499 Bessel evaluations, the splat, the radius rule, the line-spread fit) stays
//! on the CPU: it runs once per `(pitch, f-number)` and is cached there. Everything per pixel runs on
//! the GPU, as eleven dispatches of `diffraction.wgsl`, each with its own pipeline (the passes are
//! separate [`KernelId`]s that share one source file). No pass reads a pixel that another thread of
//! the same dispatch writes, there are no atomics, no barriers and no reductions, and every sum has a
//! fixed order, so a run is deterministic and a tile gives exactly the bits of the same pixels in a
//! whole-frame run.
//!
//! | # | Pass | Reads | Writes |
//! |---|---|---|---|
//! | 1 | `DiffractionLuma` | the RGB planes | `Y`, `w`, clip penalty `Pc` (spec 3.4) |
//! | 2 | `DiffractionEnvelope` | `Y` (5 x 5) | lower and upper envelope (3.5) |
//! | 3 | `DiffractionLayers` | `Y`, envelopes (5 x 5 means) | dark `D` and bright `B` layer (3.6) |
//! | 4 | `DiffractionGaussRows` (`src 0`) | `Y` | the horizontal half of `G(Y)` |
//! | 5 | `DiffractionGaussCols` (`src 0`) | that, `Y` | `Hp = Y - G(Y)` (3.7) |
//! | 6 | `DiffractionPsf` | `D`, `B`, `Y` | `F1 = K * D`, `F2 = K * B`, `eps = Y - (F1 + F2)` |
//! | 7 | `DiffractionGaussRows` (`src 1`) | `Z = F1 + F2` | the horizontal half of `G(Z)` |
//! | 8 | `DiffractionGaussCols` (`src 1`) | that, `Z` | `F3 = Z - G(Z)` |
//! | 9 | `DiffractionMomentRows` | `w`, `F1`..`F3`, `eps`, `Pc` (9 along x) | the 11 row sums of the moments (3.8) |
//! | 10 | `DiffractionSolve` | the row sums (9 along y) | `uD`, `uB`, `uH`, `omega` (3.9) |
//! | 11 | `DiffractionOutput` | the four coefficient planes (5 x 5), `Y`, `D`, `B`, `Hp`, the RGB planes | the core's RGB (3.10 to 3.12) |
//!
//! The ten product planes of the moments are never stored: the row pass forms `(w u) v` at each tap.
//!
//! # Precision decisions
//!
//! * **Storage is `f32` whatever the device's storage mode.** The colour kernels keep image
//!   intermediates in `f16`, but the envelope estimate compares luma values for equality and order
//!   (spec 3.5) and the moments subtract nearly equal sums, so an `f16` plane would change counts and
//!   erase the signal. Every plane and the input and output pixels are `f32` buffers. (Spec 3.13
//!   keeps the planes `f32` as well; `f16` storage is not an option for this stage.)
//! * **Accumulation is `f32`.** WGSL has no `f64`, so the envelopes, the 5 x 5 and 9 x 9
//!   means, the Gaussians, the PSF convolutions, the products and the solve run in `f32`, where the
//!   CPU port accumulates in `f64` and rounds once per stored plane. This is the "all-`f32`"
//!   implementation of spec 3.13 (which measured 1e-5 against the reference). The order of the sums
//!   is the CPU's: windows along x first and the row sums along y second, PSF taps row by row,
//!   Gaussian taps from the left; the solve is the same Cramer solve on the cofactors with the same
//!   branches, written as ordinary `if`s.
//! * **The planner's numbers are the CPU's.** The kernel taps are the plan's `f32` taps; the Gaussian
//!   taps are computed here in `f64` with the CPU's own expression (`exp(-i^2 / (2 sigma^2))`,
//!   normalised) and rounded to `f32`; `sigma0^2` is formed in `f64` and rounded; the clip threshold
//!   is `0.8 S` as `f32`. The luma weights, the gains and the strength reach the shader as `f32`.
//! * **Transcendentals.** The shader uses none: the only `exp` is the Gaussian taps (host side), so
//!   there is no `exp2`/`log2` accuracy question.
//! * **The luma is exact.** The one place where `f32` is not enough. The CPU forms `Y` in `f64` from
//!   the exact products `a_c S x_c` and rounds once; the envelope counts of spec 3.5 then compare
//!   neighbouring lumas for order and equality. The plain `f32` sum
//!   `a_R (S R) + a_G (S G) + a_B (S B)` is 1 to 2 ulp (0.004 DN at 30000 DN) off in a third of the
//!   pixels, and a count that flips moves a pixel's envelope by up to 8 %: on a 2048 x 1365 frame
//!   that alone gave a worst error of 9.1e-5 (feeding the CPU port the `f32` luma and nothing else
//!   reproduced the GPU's worst pixels to 0.5 %). `exact_luma` in the shader forms the sum as an
//!   exact expansion (Dekker's product, Knuth's sum, `f32` only) and returns the correctly rounded
//!   `f32` of the exact value, which is what the CPU's `f64` sum rounds to except where the exact
//!   value lies within about 2^-24 ulp of a rounding boundary (a rate of order one pixel in 10^7):
//!   the luma plane has the CPU's bits (`tests/gpu_diffraction.rs` counts 0 differences in 240000
//!   pixels; the RA-09 review counted 0 in a further 17.3 million random, 14-bit and balanced
//!   lumas) and the large frame drops to 4.8e-7. Two things are needed to make it work, and
//!   both were found by measurement on lavapipe: every rounded intermediate goes through `opaque`
//!   (an xor with the uniform `zero`, which the compiler cannot fold), because the optimiser
//!   otherwise fuses and cancels the error terms of the transformations, and `fma` is not used,
//!   because lavapipe lowers it to a multiply and an add (every error term came out 0). The
//!   transformations assume round-to-nearest `f32` and no re-association across the barrier, which
//!   desktop and Apple GPUs give; a driver that does not would leave the luma 1 to 2 ulp off and the
//!   parity at the 1e-4 level of the plain sum, which `the_luma_plane_has_the_cpu_bits` reports.
//! * **A noise-free smooth ramp has no stable answer.** Every window is evenly spread, so the
//!   envelopes collapse to `lower = upper = mid`, `Y` sits within a rounding of both layer thresholds
//!   (`Lm`, `Hm`) and the dark/bright split is decided by rounding. The CPU port itself moves by up
//!   to 3.5e-3 when such an input changes by 0.6 ulp, so no port can match it there; real frames
//!   carry sensor noise (`sigma0` is at least 1 DN) and the tests use scenes with noise.
//! * **Strength and the output.** The CPU keeps `L` and `L_s` in `f64`; the shader forms the correction
//!   `delta = (D uD + B uB + Hp uH) / omega` directly and `L_s = Y + strength * delta` (the same value
//!   without the `Y + delta - Y` round trip). When `L_s == Y` (a gated pixel, `strength` 0, a flat
//!   field) the shader writes `in * 1 + 0` without dividing, so those are exact identities whatever
//!   the accuracy of the driver's divide.
//! * **Range.** The cofactors multiply three moments, each at most `w F^2`. With `w <= 1` and balanced
//!   values up to about 40 times white they stay below the `f32` maximum; a frame beyond that is out
//!   of contract (R7 never produces it). A degenerate solve (a non-finite coefficient) leaves the
//!   pixel alone, as on the CPU.
//!
//! # Parity with the CPU reference
//!
//! `tests/gpu_diffraction.rs` asserts the maximum absolute difference in normalised units on the
//! lavapipe software adapter (Mesa 25.2, LLVM 20.1): 8.9e-7 on the 13 image vectors (their own
//! tolerance is 2e-5), 5.1e-6 on random frames of 11 optics (R 2 to 6, `r_G` 1 to 6, halos 13 to 24)
//! and six scene kinds with noise, 3.6e-6 for the other strengths, scales, gates, luma weights and
//! gains, 4.3e-6 for values from -0.05 to 4 with strength 3, 9.2e-7 for sizes 1 x 1 to 7 x 7, and
//! 4.8e-7 on a 2048 x 1365 frame through 512 px tiles (nothing above 1e-6 in 8.4 million samples).
//! Tiles with the plan's exact halo equal the whole-frame run bit for bit, and two runs are
//! bit-identical in every storage mode. [`PreparedDiffraction::run_traced`] returns the
//! intermediate planes ([`DiffractionTrace`]) for diagnosing a difference.
//!
//! # Memory
//!
//! For a compute rectangle of `n` pixels: 3 input planes, 10 planes of `Y, w, Pc, D, B, Hp, F1, F2, F3,
//! eps`, 11 scratch planes (envelopes, then the Gaussian row, then the 11 moment row sums), 4
//! coefficient planes and the core's 3 output planes, all `f32`: [`BYTES_PER_PIXEL`] = 124 bytes per
//! pixel. The 11-plane scratch is one storage binding, so it must fit the device's
//! `max_storage_buffer_binding_size` (128 MiB by default: 3 million pixels); use
//! [`PreparedDiffraction::recommended_tile_px`] and [`PreparedDiffraction::run_tiled`] for frames.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{GROUP, STORAGE_SCRATCH, storage_buffer};
use crate::cpu::RgbImage;
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::sharpen::diffraction::DiffractionPlan;
use crate::cpu::tiling::{Halo, Rect};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::stream::PassList;
use crate::tiles::{TILE_SIZES, TilePlan, tile_size_for_budget};
use std::sync::Arc;

/// Planes of the `base` buffer: `Y, w, Pc, D, B, Hp, F1, F2, F3, eps`.
const BASE_PLANES: usize = 10;
/// Planes of the `work` buffer: the 11 row sums of the moments (the envelopes and the Gaussian row
/// use the first two and the first).
const WORK_PLANES: usize = 11;
/// Planes of the `coef` buffer: `uD`, `uB`, `uH`, `omega`.
const COEF_PLANES: usize = 4;

/// Device bytes one pixel of a compute rectangle needs: the input (3 planes), `base` (10), `work`
/// (11), `coef` (4) and the core's output (3, at most the rectangle), all `f32`. Pass it to
/// [`crate::tiles::tile_size_for_budget`] when choosing a tile size
/// ([`PreparedDiffraction::recommended_tile_px`] does).
pub const BYTES_PER_PIXEL: u64 = 4 * (3 + BASE_PLANES + WORK_PLANES + COEF_PLANES + 3) as u64;

/// Fraction of white above which a channel counts as near clipping (spec 2.4): `T = 0.8 S`.
const CLIP_FRACTION: f64 = 0.8;

/// The bindings of `diffraction.wgsl`: the discriminant is the `@binding` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Slot {
    Params = 0,
    Taps = 1,
    Img = 2,
    Base = 3,
    Work = 4,
    Coef = 5,
    Out = 6,
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
    use Slot::{Base, Coef, Img, Out, Params, Taps, Work};
    Some(match id {
        KernelId::DiffractionLuma => ("diffraction_luma", &[Params, Img, Base]),
        KernelId::DiffractionEnvelope => ("diffraction_envelope", &[Params, Base, Work]),
        KernelId::DiffractionLayers => ("diffraction_layers", &[Params, Base, Work]),
        KernelId::DiffractionGaussRows => ("diffraction_gauss_rows", &[Params, Taps, Base, Work]),
        KernelId::DiffractionGaussCols => ("diffraction_gauss_cols", &[Params, Taps, Base, Work]),
        KernelId::DiffractionPsf => ("diffraction_psf", &[Params, Taps, Base]),
        KernelId::DiffractionMomentRows => ("diffraction_moment_rows", &[Params, Base, Work]),
        KernelId::DiffractionSolve => ("diffraction_solve", &[Params, Work, Coef]),
        KernelId::DiffractionOutput => ("diffraction_output", &[Params, Img, Base, Coef, Out]),
        _ => return None,
    })
}

/// The kernel body of a diffraction pass: the shared WGSL plus the pass's `main`. Called by
/// [`super::shader::shader_source`]; any other id is a programming error and gets an empty body.
pub(super) fn body(id: KernelId) -> String {
    let Some((function, _)) = pass_of(id) else {
        return String::new();
    };
    let mut s = String::from(include_str!("../shaders/diffraction.wgsl"));
    s.push_str(&entry(function));
    s
}

/// The uniform block of `diffraction.wgsl` (`DiffParams`): the same fields in the same order, 25
/// scalars of 4 bytes. Every pass reads the fields it needs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ParamsU {
    width: u32,
    height: u32,
    plane: u32,
    src: u32,
    core_x: u32,
    core_y: u32,
    core_w: u32,
    core_h: u32,
    kernel_radius: u32,
    gauss_radius: u32,
    kernel_off: u32,
    gauss_off: u32,
    /// Always 0. The luma expansion xors its rounded intermediates with it, which hides them from the
    /// shader compiler (it cannot know the value), so that it cannot fuse or cancel the error terms.
    zero: u32,
    luma_r: f32,
    luma_g: f32,
    luma_b: f32,
    scale: f32,
    g0_r: f32,
    g0_g: f32,
    g0_b: f32,
    clip_t: f32,
    sigma0_sq: f32,
    sigma1: f32,
    gate_factor: f32,
    strength: f32,
    /// Row stride and plane length of the input image (a device image has an even stride).
    in_stride: u32,
    in_plane: u32,
    /// Row stride and plane length of the output image.
    out_stride: u32,
    out_plane: u32,
}

/// The per-frame constants of a plan as the shader reads them.
#[derive(Clone, Copy, Debug)]
struct Constants {
    luma: [f32; 3],
    g0: [f32; 3],
    clip_t: f32,
    sigma0_sq: f32,
    sigma1: f32,
    gate_factor: f32,
    strength: f32,
}

/// A diffraction deconvolution plan uploaded to the GPU: immutable, reusable for any number of tiles
/// and images with the same plan.
#[derive(Debug)]
pub struct PreparedDiffraction {
    constants: Constants,
    kernel_radius: usize,
    gauss_radius: usize,
    /// Offset of the Gaussian taps in `taps` (the PSF kernel comes first, at 0).
    gauss_off: u32,
    halo_px: usize,
    taps: PooledBuffer,
}

/// Intermediate planes of a run over a whole compute rectangle, read back for diagnostics (see
/// [`PreparedDiffraction::run_traced`]). Every plane is row-major `width x height`, `f32`, in the
/// units of the stage (luma in DN on the 65535 scale).
#[derive(Clone, Debug, PartialEq)]
pub struct DiffractionTrace {
    /// Width of the traced rectangle.
    pub width: usize,
    /// Height of the traced rectangle.
    pub height: usize,
    /// Luma `Y` in DN (spec 3.4).
    pub luma: Vec<f32>,
    /// Inverse noise variance `w`.
    pub weight: Vec<f32>,
    /// Clip penalty `Pc`.
    pub penalty: Vec<f32>,
    /// Dark layer `D` (3.6).
    pub dark: Vec<f32>,
    /// Bright layer `B`.
    pub bright: Vec<f32>,
    /// High-pass of luma `Hp = Y - G(Y)` (3.7).
    pub highpass: Vec<f32>,
    /// `F1 = K * D`.
    pub f1: Vec<f32>,
    /// `F2 = K * B`.
    pub f2: Vec<f32>,
    /// `F3 = Z - G(Z)`, the high-pass of the reblurred luma.
    pub f3: Vec<f32>,
    /// The residual `eps = Y - Z`.
    pub residual: Vec<f32>,
    /// The solve's output `uD = cD * gain * omega` per pixel, before the 5 x 5 smoothing (3.9).
    pub coef_dark: Vec<f32>,
    /// `uB`.
    pub coef_bright: Vec<f32>,
    /// `uH`.
    pub coef_high: Vec<f32>,
    /// The confidence weight `omega`.
    pub omega: Vec<f32>,
}

/// The separable Gaussian of the high-pass: taps `exp(-i^2 / (2 sigma^2))` for `|i| <= r`, normalised
/// to sum 1, in `f64` with the CPU's own expression (`cpu::sharpen::diffraction::gauss_blur`), then
/// rounded to `f32`.
fn gauss_taps(sigma_g: f32, radius: usize) -> Vec<f32> {
    let sigma = f64::from(sigma_g);
    let mut taps = vec![0.0f64; 2 * radius + 1];
    let mut sum = 0.0f64;
    for (k, t) in taps.iter_mut().enumerate() {
        let ix = k as i64 - radius as i64;
        *t = (-((ix * ix) as f64) / (2.0 * sigma * sigma)).exp();
        sum += *t;
    }
    taps.iter().map(|&t| (t / sum) as f32).collect()
}

fn check_image(image: &RgbImage) -> Result<(), GpuError> {
    if image.width == 0 || image.height == 0 {
        return Err(GpuError::InvalidInput(
            "diffraction deconvolution: the image has no pixels".into(),
        ));
    }
    if image.data.len() != 3 * image.width * image.height {
        return Err(GpuError::InvalidInput(format!(
            "diffraction deconvolution: the buffer holds {} samples but {} x {} pixels need {}",
            image.data.len(),
            image.width,
            image.height,
            3 * image.width * image.height
        )));
    }
    Ok(())
}

impl PreparedDiffraction {
    /// Uploads `plan` (its PSF kernel and the Gaussian taps of its `sigma_G`).
    ///
    /// Everything the correction needs comes from the plan: its noise constants, strength, noise-gate
    /// factor, kernel and radii, and the gains and luma weights it was made with
    /// ([`DiffractionPlan::g0`], [`DiffractionPlan::luma`]). There is nothing to pass again, so the
    /// kernel cannot be run with values other than the CPU reference's.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for gains that are not positive and finite or luma weights that are
    /// not finite (a plan from [`DiffractionPlan::new`] never has them, the planner refuses them; the
    /// check is kept because the values go straight into the shader), and allocation errors from the
    /// pool.
    pub fn new(ctx: &GpuContext, plan: &DiffractionPlan) -> Result<Self, GpuError> {
        let (g0, luma) = (plan.g0(), plan.luma());
        if g0.iter().any(|&g| !g.is_finite() || g <= 0.0) || luma.iter().any(|a| !a.is_finite()) {
            return Err(GpuError::InvalidInput(format!(
                "diffraction deconvolution: the plan's gains {g0:?} must be positive and its luma weights {luma:?} finite"
            )));
        }
        let kernel = plan.kernel();
        let mut taps: Vec<f32> = kernel.to_vec();
        let gauss_off = taps.len() as u32;
        taps.extend(gauss_taps(plan.gauss_sigma(), plan.gauss_radius()));
        let taps = storage_buffer(ctx, "diffraction taps", bytemuck::cast_slice(&taps))?;
        let s0 = plan.sigma0_dn();
        Ok(Self {
            constants: Constants {
                luma,
                g0,
                clip_t: (CLIP_FRACTION * WORKING_SCALE) as f32,
                sigma0_sq: (s0 * s0) as f32,
                sigma1: plan.sigma1_dn() as f32,
                gate_factor: plan.gate_factor() as f32,
                strength: plan.strength() as f32,
            },
            kernel_radius: plan.radius(),
            gauss_radius: plan.gauss_radius(),
            gauss_off,
            halo_px: plan.halo_px(),
            taps,
        })
    }

    /// The exact receptive field in pixels (`10 + R + r_G`, spec 1.5): the context a tile needs on
    /// every side.
    #[must_use]
    pub fn halo_px(&self) -> usize {
        self.halo_px
    }

    /// A tile plan for a `width x height` frame with cores of `tile_px` and this plan's halo.
    #[must_use]
    pub fn tile_plan(&self, width: usize, height: usize, tile_px: usize) -> TilePlan {
        TilePlan::with_halo(width, height, tile_px, self.halo_px)
    }

    /// The tile size to use on this device: the largest of [`TILE_SIZES`] whose working set (one
    /// tile in flight, [`BYTES_PER_PIXEL`]) fits in half the memory budget and whose 11-plane scratch
    /// buffer fits one storage binding.
    #[must_use]
    pub fn recommended_tile_px(&self, ctx: &GpuContext) -> usize {
        let info = ctx.info();
        let mut tile = tile_size_for_budget(
            info.memory_budget_bytes,
            BYTES_PER_PIXEL,
            self.halo_px,
            1,
            &TILE_SIZES,
        );
        let largest_binding = (4 * WORK_PLANES) as u64;
        while tile > 256 {
            let side = (tile + 2 * self.halo_px) as u64;
            if side * side * largest_binding <= info.limits.max_storage_buffer_binding_size {
                break;
            }
            tile /= 2;
        }
        tile
    }

    /// Sharpens `image` in one set of dispatches over the whole frame: the whole buffer is one
    /// rectangle, mirrored at its edges. The output has the input's size and
    /// [`Placement`](crate::cpu::Placement).
    ///
    /// This is the GPU counterpart of [`DiffractionPlan::apply_whole`]; see the module documentation
    /// for how they differ. For a big frame use [`run_tiled`](Self::run_tiled): the scratch planes
    /// need [`BYTES_PER_PIXEL`] bytes per pixel and 44 bytes per pixel in one storage binding.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty image or a buffer that does not match its size,
    /// [`GpuError::BufferTooLarge`] and [`GpuError::OutOfBudget`] when the buffers do not fit the
    /// device, shader and readback errors.
    pub fn run(&self, ctx: &GpuContext, image: &RgbImage) -> Result<RgbImage, GpuError> {
        check_image(image)?;
        let whole = Rect::whole(image.width, image.height);
        Ok(self.run_host(ctx, image, whole, whole, false)?.0)
    }

    /// Like [`run`](Self::run), and also returns the intermediate planes of the whole frame
    /// ([`DiffractionTrace`]), for diagnostics and the parity tests.
    ///
    /// # Errors
    ///
    /// The errors of [`run`](Self::run).
    pub fn run_traced(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
    ) -> Result<(RgbImage, DiffractionTrace), GpuError> {
        check_image(image)?;
        let whole = Rect::whole(image.width, image.height);
        let (out, trace) = self.run_host(ctx, image, whole, whole, true)?;
        let trace = trace.ok_or_else(|| GpuError::Readback("no trace planes".into()))?;
        Ok((out, trace))
    }

    /// Sharpens `image` tile by tile: cores of at most `tile_px` (rounded down to an even number),
    /// each computed from a rectangle that carries [`halo_px`](Self::halo_px) pixels of context
    /// (clamped to the frame, left and top edges even) and pasted into the result.
    ///
    /// Gives exactly the pixels of [`run`](Self::run) on the whole frame, bit for bit (the tests
    /// assert it). Tiles are processed one after the other; `cancel` is checked between them.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when `cancel` fires, plus the errors of [`run`](Self::run).
    pub fn run_tiled(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        tile_px: usize,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
        check_image(image)?;
        let plan = self.tile_plan(image.width, image.height, tile_px);
        self.run_planned(ctx, image, &plan, cancel)
    }

    /// Like [`run_tiled`](Self::run_tiled) with a plan made by the caller, which must be for the size
    /// of `image` and give every tile at least [`halo_px`](Self::halo_px) pixels of context around
    /// its core ([`tile_plan`](Self::tile_plan) builds such a plan, and so does a
    /// [`TilePlan::new`] over `Stage::Sharpen(CaptureSharpen::Diffraction(..))`).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the plan is for another size or a tile lacks context (the
    /// kernel mirrors at the edge of the tile's rectangle, so missing context would show as seams,
    /// not as an error), [`GpuError::Cancelled`] when `cancel` fires, plus the errors of
    /// [`run`](Self::run).
    pub fn run_planned(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        plan: &TilePlan,
        cancel: &CancelToken,
    ) -> Result<RgbImage, GpuError> {
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
                    "tile {} computes {:?}, but its core {:?} needs {} px of context ({needed:?}); \
                     build the plan with `PreparedDiffraction::tile_plan`",
                    tile.index, tile.compute, tile.core, self.halo_px
                )));
            }
        }
        let mut out = RgbImage::new(image.width, image.height);
        out.place = image.place;
        for &tile in plan.tiles() {
            cancel.check()?;
            let core = self.run_host(ctx, image, tile.compute, tile.core, false)?.0;
            let at = Rect::new(0, 0, core.width, core.height);
            out.paste(&core, at, tile.core.x, tile.core.y);
        }
        Ok(out)
    }

    /// Uploads the `compute` rectangle of `image`, runs [`encode`](Self::encode) for the `core`
    /// inside it, submits and reads the core back (and the traced planes when `trace`).
    fn run_host(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        compute: Rect,
        core: Rect,
        trace: bool,
    ) -> Result<(RgbImage, Option<DiffractionTrace>), GpuError> {
        let sub = if compute == Rect::whole(image.width, image.height) {
            None
        } else {
            Some(image.extract(compute))
        };
        let src = sub.as_ref().unwrap_or(image);
        let input = GpuImage::upload_rgb(ctx, "diffraction input", src)?;
        let mut passes = PassList::new(ctx, "diffraction deconvolution");
        let core_in = Rect::new(core.x - compute.x, core.y - compute.y, core.w, core.h);
        let (out, traced) = self.record(&mut passes, &input, core_in, trace)?;
        let read = out.encode_read(&mut passes)?;
        let trace_reads = match &traced {
            Some(t) => Some((
                passes.read(&t.base, 0, BASE_PLANES as u64 * t.plane_bytes)?,
                passes.read(&t.coef, 0, COEF_PLANES as u64 * t.plane_bytes)?,
            )),
            None => None,
        };
        let done = passes.submit()?.finish()?;
        let data = out.take(&done, read)?;
        let mut image_out = RgbImage {
            width: out.width,
            height: out.height,
            place: out.place,
            data,
        };
        image_out.place = image.place.offset(core.x, core.y);
        let trace = match (traced, trace_reads) {
            (Some(t), Some((base, coef))) => Some(unpack_trace(
                (t.width, t.height),
                &bytes_to_f32(done.read(base)),
                &bytes_to_f32(done.read(coef)),
            )?),
            _ => None,
        };
        Ok((image_out, trace))
    }

    /// Records every pass for the `core` rectangle of `input` (a device image of the compute
    /// rectangle, three planes) and returns the core's three planes as a new [`GpuImage`]: device in,
    /// device out. Nothing is submitted.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an input that is not three planes, a core outside it, or a
    /// compute rectangle that needs more workgroups on one side than the device allows;
    /// [`GpuError::BufferTooLarge`] and [`GpuError::OutOfBudget`] when the buffers do not fit the
    /// device; shader errors.
    pub fn encode(
        &self,
        passes: &mut PassList<'_>,
        input: &GpuImage,
        core: Rect,
    ) -> Result<GpuImage, GpuError> {
        Ok(self.record(passes, input, core, false)?.0)
    }

    fn record(
        &self,
        passes: &mut PassList<'_>,
        input: &GpuImage,
        core: Rect,
        trace: bool,
    ) -> Result<(GpuImage, Option<TraceBuffers>), GpuError> {
        let ctx = passes.context();
        let (w, h) = (input.width, input.height);
        if input.planes != 3 || core.is_empty() || core.x1() > w || core.y1() > h {
            return Err(GpuError::InvalidInput(format!(
                "diffraction deconvolution: the input must be three planes and the core {core:?} must lie inside the {w} x {h} rectangle (got {} planes)",
                input.planes
            )));
        }
        // One workgroup covers GROUP x GROUP pixels; a side of more than `limit` workgroups is a
        // validation error that wgpu raises as a panic, so refuse it here (a 1 x N frame with
        // N > 65535 * 16 on the default limits).
        let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
        let groups = |side: usize| side.div_ceil(GROUP as usize);
        if groups(w) > limit || groups(h) > limit {
            return Err(GpuError::InvalidInput(format!(
                "diffraction deconvolution: a {w} x {h} compute rectangle needs more than {limit} \
                 workgroups of {GROUP} px on one side; use run_tiled with smaller tiles"
            )));
        }
        let n = w * h;
        if (WORK_PLANES.max(BASE_PLANES) * n) as u64 > u64::from(u32::MAX) {
            return Err(GpuError::BufferTooLarge {
                requested: (WORK_PLANES.max(BASE_PLANES) * n) as u64 * 4,
                limit: u64::from(u32::MAX) * 4,
            });
        }
        let plane_bytes = (n * 4) as u64;
        let pool = ctx.pool();

        // Every pass writes every pixel of the planes it produces before a later pass reads them, so
        // the pooled buffers need no clearing. A traced run copies two of the buffers out.
        let traced_usage = if trace {
            STORAGE_SCRATCH.union(wgpu::BufferUsages::COPY_SRC)
        } else {
            STORAGE_SCRATCH
        };
        let base_buf = passes.keep(pool.acquire(
            "diffraction planes",
            BASE_PLANES as u64 * plane_bytes,
            traced_usage,
        )?);
        let work = passes.scratch("diffraction work", WORK_PLANES as u64 * plane_bytes)?;
        let coef = passes.keep(pool.acquire(
            "diffraction coefficients",
            COEF_PLANES as u64 * plane_bytes,
            traced_usage,
        )?);
        let out = GpuImage::alloc(
            ctx,
            "diffraction out",
            core.w,
            core.h,
            3,
            input.place.offset(core.x, core.y),
        )?;

        let c = &self.constants;
        let base = ParamsU {
            width: w as u32,
            height: h as u32,
            plane: n as u32,
            kernel_radius: self.kernel_radius as u32,
            gauss_radius: self.gauss_radius as u32,
            kernel_off: 0,
            gauss_off: self.gauss_off,
            luma_r: c.luma[0],
            luma_g: c.luma[1],
            luma_b: c.luma[2],
            scale: WORKING_SCALE as f32,
            g0_r: c.g0[0],
            g0_g: c.g0[1],
            g0_b: c.g0[2],
            clip_t: c.clip_t,
            sigma0_sq: c.sigma0_sq,
            sigma1: c.sigma1,
            gate_factor: c.gate_factor,
            strength: c.strength,
            in_stride: input.stride as u32,
            in_plane: input.plane_len() as u32,
            out_stride: out.stride as u32,
            out_plane: out.plane_len() as u32,
            ..ParamsU::default()
        };

        let mut run = |id: KernelId, p: &ParamsU, size: (usize, usize)| -> Result<(), GpuError> {
            let ub = passes.uniform("diffraction params", p)?;
            let slots = pass_of(id).map_or(&[][..], |(_, s)| s);
            let entries: Vec<(u32, wgpu::BindingResource<'_>)> = slots
                .iter()
                .map(|&s| {
                    let resource = match s {
                        Slot::Params => ub.binding(),
                        Slot::Taps => self.taps.binding(),
                        Slot::Img => input.buf.binding(),
                        Slot::Base => base_buf.binding(),
                        Slot::Work => work.binding(),
                        Slot::Coef => coef.binding(),
                        Slot::Out => out.buf.binding(),
                    };
                    (s as u32, resource)
                })
                .collect();
            passes.dispatch(
                id,
                id.name(),
                &entries,
                (
                    size.0.div_ceil(GROUP as usize) as u32,
                    size.1.div_ceil(GROUP as usize) as u32,
                    1,
                ),
            )
        };

        let all = (w, h);
        let blur_y = ParamsU { src: 0, ..base };
        let blur_z = ParamsU { src: 1, ..base };
        run(KernelId::DiffractionLuma, &base, all)?;
        run(KernelId::DiffractionEnvelope, &base, all)?;
        run(KernelId::DiffractionLayers, &base, all)?;
        run(KernelId::DiffractionGaussRows, &blur_y, all)?;
        run(KernelId::DiffractionGaussCols, &blur_y, all)?;
        run(KernelId::DiffractionPsf, &base, all)?;
        run(KernelId::DiffractionGaussRows, &blur_z, all)?;
        run(KernelId::DiffractionGaussCols, &blur_z, all)?;
        run(KernelId::DiffractionMomentRows, &base, all)?;
        run(KernelId::DiffractionSolve, &base, all)?;
        let output = ParamsU {
            core_x: core.x as u32,
            core_y: core.y as u32,
            core_w: core.w as u32,
            core_h: core.h as u32,
            ..base
        };
        run(KernelId::DiffractionOutput, &output, (core.w, core.h))?;
        let traced = trace.then(|| TraceBuffers {
            base: base_buf,
            coef,
            plane_bytes,
            width: w,
            height: h,
        });
        Ok((out, traced))
    }
}

/// The two buffers a traced run copies out.
struct TraceBuffers {
    base: Arc<PooledBuffer>,
    coef: Arc<PooledBuffer>,
    plane_bytes: u64,
    width: usize,
    height: usize,
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Splits the copied-out `base` and `coef` planes of a traced run into a [`DiffractionTrace`].
fn unpack_trace(
    (w, h): (usize, usize),
    base: &[f32],
    coef: &[f32],
) -> Result<DiffractionTrace, GpuError> {
    let n = w * h;
    if base.len() != BASE_PLANES * n || coef.len() != COEF_PLANES * n {
        return Err(GpuError::Readback("missing trace planes".into()));
    }
    let mut it = base.chunks_exact(n).chain(coef.chunks_exact(n)).map(<[f32]>::to_vec);
    let mut next = || {
        it.next()
            .ok_or_else(|| GpuError::Readback("missing trace plane".into()))
    };
    Ok(DiffractionTrace {
        width: w,
        height: h,
        luma: next()?,
        weight: next()?,
        penalty: next()?,
        dark: next()?,
        bright: next()?,
        highpass: next()?,
        f1: next()?,
        f2: next()?,
        f3: next()?,
        residual: next()?,
        coef_dark: next()?,
        coef_bright: next()?,
        coef_high: next()?,
        omega: next()?,
    })
}

/// Records the diffraction deconvolution of a device image (device in, device out); see
/// [`PreparedDiffraction::encode`].
///
/// # Errors
///
/// As [`PreparedDiffraction::encode`].
pub fn encode_sharpen(
    passes: &mut PassList<'_>,
    plan: &PreparedDiffraction,
    input: &GpuImage,
    core: Rect,
) -> Result<GpuImage, GpuError> {
    plan.encode(passes, input, core)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::kernels::shader_source;

    /// Every diffraction pass, as registered in [`KernelId`].
    fn passes() -> Vec<KernelId> {
        let all: Vec<KernelId> = KernelId::ALL
            .into_iter()
            .filter(|id| id.name().starts_with("diffraction"))
            .collect();
        assert_eq!(all.len(), 9, "nine passes are registered");
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
            assert!(
                expected.len() <= 8,
                "{id:?}: more bindings than the default per-stage limit"
            );
        }
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_params_u() {
        let (module, _) = module_of(KernelId::DiffractionLuma);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("dp"))
            .expect("the uniform `dp`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`dp` is not a struct");
        };
        let fields: [(&str, usize); 29] = [
            ("width", offset_of!(ParamsU, width)),
            ("height", offset_of!(ParamsU, height)),
            ("plane", offset_of!(ParamsU, plane)),
            ("src", offset_of!(ParamsU, src)),
            ("core_x", offset_of!(ParamsU, core_x)),
            ("core_y", offset_of!(ParamsU, core_y)),
            ("core_w", offset_of!(ParamsU, core_w)),
            ("core_h", offset_of!(ParamsU, core_h)),
            ("kernel_radius", offset_of!(ParamsU, kernel_radius)),
            ("gauss_radius", offset_of!(ParamsU, gauss_radius)),
            ("kernel_off", offset_of!(ParamsU, kernel_off)),
            ("gauss_off", offset_of!(ParamsU, gauss_off)),
            ("zero", offset_of!(ParamsU, zero)),
            ("luma_r", offset_of!(ParamsU, luma_r)),
            ("luma_g", offset_of!(ParamsU, luma_g)),
            ("luma_b", offset_of!(ParamsU, luma_b)),
            ("scale", offset_of!(ParamsU, scale)),
            ("g0_r", offset_of!(ParamsU, g0_r)),
            ("g0_g", offset_of!(ParamsU, g0_g)),
            ("g0_b", offset_of!(ParamsU, g0_b)),
            ("clip_t", offset_of!(ParamsU, clip_t)),
            ("sigma0_sq", offset_of!(ParamsU, sigma0_sq)),
            ("sigma1", offset_of!(ParamsU, sigma1)),
            ("gate_factor", offset_of!(ParamsU, gate_factor)),
            ("strength", offset_of!(ParamsU, strength)),
            ("in_stride", offset_of!(ParamsU, in_stride)),
            ("in_plane", offset_of!(ParamsU, in_plane)),
            ("out_stride", offset_of!(ParamsU, out_stride)),
            ("out_plane", offset_of!(ParamsU, out_plane)),
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
        // Planes are f32 whatever the storage mode (the envelope compares luma values), so the text
        // has no f16 and no accessor expansion beyond the generated prelude.
        let body = include_str!("../shaders/diffraction.wgsl");
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
    fn the_shader_has_no_transcendental_functions() {
        // The only exp of the stage is the Gaussian taps, computed on the host in f64.
        let body = include_str!("../shaders/diffraction.wgsl");
        for call in [
            "exp(", "exp2(", "log(", "log2(", "pow(", "sin(", "cos(", "sqrt(",
        ] {
            assert!(!body.contains(call), "the shader calls {call}");
        }
    }

    #[test]
    fn the_gaussian_taps_sum_to_one_and_are_symmetric() {
        for (sigma, radius) in [(0.2f32, 1usize), (0.5, 2), (1.61, 6), (3.0, 8)] {
            let taps = gauss_taps(sigma, radius);
            assert_eq!(taps.len(), 2 * radius + 1);
            let sum: f64 = taps.iter().map(|&t| f64::from(t)).sum();
            assert!((sum - 1.0).abs() < 1e-6, "sigma {sigma}: sum {sum}");
            for k in 0..radius {
                assert_eq!(taps[k].to_bits(), taps[2 * radius - k].to_bits());
                assert!(taps[k] <= taps[k + 1]);
            }
        }
    }

    #[test]
    fn bytes_per_pixel_counts_the_planes() {
        assert_eq!(BYTES_PER_PIXEL, 124);
    }
}
