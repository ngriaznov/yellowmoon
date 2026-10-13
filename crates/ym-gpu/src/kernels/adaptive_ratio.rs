//! The adaptive ratio demosaic on the GPU (RA-09): the WGSL port of
//! [`crate::cpu::demosaic::adaptive_ratio`], specified in
//! `docs/algorithms/01-adaptive-ratio-demosaic.md` (section 5 lists the GPU notes, section 11 the CA
//! path). It is R6's classical demosaic, `Method::AdaptiveRatio`, including the CA-aware green
//! refinement, the shadow sharpening and the shadow colour matching, and the entry that undoes R4 on
//! the balanced mosaic the chain hands to R6 ([`Balance::Balanced`]).
//!
//! Nothing here is wired into the render path: the raw stages still run the CPU reference, and wiring
//! this one is a later decision. The entry points upload a mosaic, run the passes and read the planes
//! back, for the parity tests and for that later decision.
//!
//! # What runs where
//!
//! The CPU computes the frame constants once, with the CPU reference's own code
//! ([`Derived`]: `K`, `T`, `k`, `rho`, the shadow-matching switch, the halo) and passes them in the
//! uniform; the CA curves go in a small storage buffer of knots. Everything per pixel runs on the GPU as
//! up to 14 dispatches of `adaptive_ratio.wgsl`, each with its own pipeline (the passes are separate
//! [`KernelId`]s that share one source file). No pass reads a pixel that another thread of the same
//! dispatch writes, there are no atomics, no barriers and no reductions, and every sum has a fixed
//! order, so a run is deterministic and a tile gives exactly the bits of the same pixels in a
//! whole-frame run.
//!
//! | # | Pass | Reads | Writes |
//! |---|---|---|---|
//! | 0 | `AdaptiveRatioPrepare` | the input (mirrored), the gains | `M = S max(m_u, 0)` over the padded plane, and the native red and blue (spec 3.1, 3.4.1) |
//! | 1 | `AdaptiveRatioGreen` | `M` (up to 5 px), the CA curves | pass-1 green `G1` (3.2) |
//! | 2 | `AdaptiveRatioRefine` | `G1`, `M` (2 px), the CA curves | the white-balanced green `g_G G2` (3.3, 3.4.1) |
//! | 3 | `AdaptiveRatioStep1` | the native planes, green | B at R sites, R at B sites (3.4.2) |
//! | 4 | `AdaptiveRatioStep2` | green, red, blue at the four axis neighbours | R and B at G sites (3.4.3) |
//! | 5 | `AdaptiveRatioLuma` | red, green, blue | the luma proxy (3.4.4; `shadow_sharpening` only) |
//! | 6 | `AdaptiveRatioSharpen` | the luma, the native greens | red, green, blue in place (3.4.4) |
//! | 7 | `AdaptiveRatioSimple` x 3 | `M` | `Gauss3(sparse_c) m_c` of one channel (3.4.5; matching only) |
//! | 8 | `AdaptiveRatioLowErr` x 3 | that, the pass-3 planes | `L_c = Gauss3(simple_c)` and `(S3_c - L_c)^2` |
//! | 9 | `AdaptiveRatioMatch` | `L`, the errors (3 x 3), the pass-3 planes (5 x 5) | the output (3.4.5, 3.5) |
//! | 9' | `AdaptiveRatioExit` | the pass-3 planes | the output (3.5), instead of 7 to 9 without matching |
//!
//! The planes are `pw x ph` where the padded plane holds the core and the margin of the receptive
//! field on every side ([`HALO_PLAIN_PX`](crate::cpu::demosaic::adaptive_ratio::HALO_PLAIN_PX) and
//! [`HALO_MAX_PX`](crate::cpu::demosaic::adaptive_ratio::HALO_MAX_PX) are the spec's safe bounds; the
//! exact reach is 10 px, 12 with shadow matching, and that is what the plane holds). The padding is made
//! once at the entry, mirrored at the edge of the uploaded rectangle without repeating the edge sample,
//! with the left and top pad chosen so that red sits at (even, even): every pass assumes RGGB. Each pass
//! is evaluated on the rectangle of the plane that the core depends on, from the output backwards,
//! exactly as the CPU port does, so every value a pass reads was written by an earlier pass and the
//! frame-edge pixels are the CPU's (mirroring each pass's input separately would differ in the outer
//! ring; spec 5).
//!
//! # Precision decisions
//!
//! * **`f32` everywhere, `f32` storage in every storage mode.** WGSL has no `f64`, so every formula
//!   the CPU runs in `f64` runs in `f32`, in the operation order of the spec; the planes are stored as
//!   `f32` exactly where the CPU stores `f32` (`M`, `G1`, the pass-3 planes, the luma, the matching
//!   planes). All planes are `f32` whatever the device's storage mode, like the other RA kernels: the
//!   decisions compare sums of absolute differences for order, which `f16` planes would destroy. The
//!   input and the output are `f32` buffers too.
//! * **Decisions are exact up to ties.** The pass-1 direction tests, the pass-2 argmin, the step-1
//!   diagonal and the step-3 diagonal compare `f32` sums where the CPU compares `f64` sums, so a
//!   comparison whose two sides differ by less than the `f32` rounding noise (relative 1e-7) can go the
//!   other way (spec 5: "expect isolated decision flips at near-ties"). A flipped decision changes the
//!   pixel by the difference between two estimates that were equally plausible. `tests/gpu_adaptive_ratio.rs`
//!   counts the pixels and states the bounds.
//! * **Inverse-cost weights.** Pass 1 and step 1 weigh the samples by `1 / w_j` instead of the product
//!   of the other costs (identical in exact arithmetic, the form spec 5 recommends for `f32`): a product
//!   of three costs reaches 1e18 for 1e6 DN data and overflows `f32` for 1e13; the inverse form stays
//!   below the largest sample.
//! * **Constants.** `K`, `T`, `k`, `rho`, the matching constants `g g_c` and `(r g_c)^2` and the
//!   multipliers `(4 g_R, 2 g_G, 4 g_B)` are computed in `f64` with the CPU reference's code and rounded
//!   to `f32` once. The working scale `S = 65535` and the noise model arrive the same way.
//! * **The CA field is `f32`.** The misregistration `|shift_R(r) - shift_B(r)|` of a site is a pure
//!   function of its position (spec 5): the shader forms the distance to the optical centre from the
//!   frame position (an exact integer difference plus a half-integer for the usual centre, so only the
//!   square root and the curve interpolation round), looks the two curves up in the `f32` knots and
//!   subtracts. The CPU does the same in `f64`; the difference is a few `1e-7` px.
//! * **Division.** `1 / w` and the final divisions are `f32` divisions, correctly rounded on lavapipe;
//!   a GPU that approximates them (up to 2.5 ulp is allowed by WGSL) moves the result by that much.
//! * **No fused multiply-add** is requested; WGSL cannot forbid one, so a driver that contracts
//!   `a * b + c` can move the last bit.
//!
//! # Inputs the kernel accepts
//!
//! The same as the CPU reference, plus two limits of the `f32` range. The host checks the whole
//! mosaic once (a scan per frame, not per tile): both sides at least 2 photosites ([`Error::BadSize`]
//! of the CPU), no NaN or infinity, gains positive and finite, a valid noise model, finite
//! parameters and a finite optical centre; every error is [`GpuError::InvalidInput`]. A sample whose
//! balanced magnitude `S |m_u| max(g)` exceeds [`MAX_BALANCED_DN`] (1e17 DN, about 1.5e12 times white)
//! is refused too, where the CPU reference accepts up to about 5e33: the squares of the matching step
//! would overflow `f32` and give NaN. So are gains or a noise model whose per-frame constants
//! (`rho`, `4 g_c`, `g g_c`, `(r g_c)^2`, `K`, `T`, `r`) do not fit `f32`, which the CPU accepts in
//! `f64` (a shot-noise gain of 1e50 DN per DN, a read noise of 1e40 DN): rounded to `f32` they would
//! be infinities and the image NaN. Negative samples are handled as the spec says (clamped at 0, the
//! residual restored in the photosite's own channel at the exit).
//!
//! # Memory and tiles
//!
//! For a core of `w x h` pixels with the margin `mg` (10, or 12 with shadow matching) a run needs 5
//! planes (12 with shadow matching) of `(w + 2 mg + 1) (h + 2 mg + 1)` `f32` in one scratch binding,
//! the input rectangle (4 bytes per photosite) and the output (12 bytes per core pixel):
//! [`BYTES_PER_PIXEL`] per pixel at most. The scratch must fit the device's
//! `max_storage_buffer_binding_size` (128 MiB by default: 2.8 million padded pixels with matching, 6.7
//! million without; a larger one is refused with [`GpuError::BufferTooLarge`]): use [`demosaic_tiled`]
//! for frames. A tile of 1024 px needs 53 MB of planes with matching. Tiles carry
//! [`Derived::halo_px`] pixels of real context (11, 13 with matching) and give the whole-frame pixels
//! bit for bit.
//!
//! [`Error::BadSize`]: crate::cpu::demosaic::adaptive_ratio::Error::BadSize

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{
    GROUP, STORAGE_IN, STORAGE_OUT, STORAGE_SCRATCH, bind, image_samples, submit_with_readback,
    uniform_buffer,
};
use crate::cpu::demosaic::adaptive_ratio::{Derived, Params};
use crate::cpu::demosaic::{CaCurves, DemosaicInputs, Method};
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::tiling::{Halo, Rect, Stage};
use crate::cpu::{Cfa, CfaColor, Mosaic, RgbImage};
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::tiles::TilePlan;

/// The largest balanced sample `S |m_u| max(g)` in DN (working scale 65535) the kernel accepts: the
/// squares of the shadow colour matching (`(S3 - L)^2`, summed over 27 samples) stay below the `f32`
/// range with a factor of 100 to spare. A sample above it is refused with
/// [`GpuError::InvalidInput`]; the CPU reference accepts values up to about 5e33.
pub const MAX_BALANCED_DN: f64 = 1.0e17;

/// Device bytes one padded pixel needs at most: the 12 planes of shadow colour matching, the input and
/// the three output planes, all `f32`. For choosing a tile size; the binding limit applies to the
/// 12 planes alone (48 bytes per padded pixel).
pub const BYTES_PER_PIXEL: u64 = 4 * (12 + 1 + 3);

/// Planes of the scratch buffer without shadow colour matching: `M`, `G1` (later the luma), red,
/// green, blue.
const PLANES_PLAIN: usize = 5;

/// Planes with shadow colour matching: the five above, `simple`, three `L` and three squared errors.
const PLANES_MATCH: usize = 12;

/// Whether the mosaic handed to the kernel has had the R4 gains applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Balance {
    /// The mosaic as the algorithm's contract has it (spec 1.2): black-subtracted, white = 1.0, *not*
    /// multiplied by the R4 gains. The output is white-balanced (white comes out as the gains).
    Unbalanced,
    /// The mosaic R5 hands to R6: R4 multiplied every photosite by the gain of its colour. The kernel
    /// divides each photosite by that gain (`m_u = m_bal / g0[colour]`, the exact inverse, spec 1.3)
    /// and runs on the result; the output is balanced, as every method's is.
    Balanced,
}

/// One slot of the bindings of `adaptive_ratio.wgsl`: the discriminant is the `@binding` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Slot {
    Params = 0,
    Input = 1,
    Scratch = 2,
    Out = 3,
    Curves = 4,
}

/// The WGSL `main` that runs `function`, appended after the shared source.
fn entry(function: &str) -> String {
    format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {function}(gid); }}\n"
    )
}

/// The pass of `id`: its WGSL function and the bindings its entry point uses (the auto-generated bind
/// group layout holds exactly these).
fn pass_of(id: KernelId) -> Option<(&'static str, &'static [Slot])> {
    use Slot::{Curves, Input, Out, Params, Scratch};
    Some(match id {
        KernelId::AdaptiveRatioPrepare => ("ar_prepare", &[Params, Input, Scratch]),
        KernelId::AdaptiveRatioGreen => ("ar_green", &[Params, Scratch, Curves]),
        KernelId::AdaptiveRatioRefine => ("ar_refine", &[Params, Scratch, Curves]),
        KernelId::AdaptiveRatioStep1 => ("ar_step1", &[Params, Scratch]),
        KernelId::AdaptiveRatioStep2 => ("ar_step2", &[Params, Scratch]),
        KernelId::AdaptiveRatioLuma => ("ar_luma", &[Params, Scratch]),
        KernelId::AdaptiveRatioSharpen => ("ar_sharpen", &[Params, Scratch]),
        KernelId::AdaptiveRatioSimple => ("ar_simple", &[Params, Scratch]),
        KernelId::AdaptiveRatioLowErr => ("ar_low_err", &[Params, Scratch]),
        KernelId::AdaptiveRatioMatch => ("ar_match", &[Params, Input, Scratch, Out]),
        KernelId::AdaptiveRatioExit => ("ar_exit", &[Params, Input, Scratch, Out]),
        _ => return None,
    })
}

/// The kernel body of an adaptive ratio pass: the shared WGSL plus the pass's `main`. Called by
/// [`super::shader::shader_source`]; any other id is a programming error and gets an empty body.
pub(super) fn body(id: KernelId) -> String {
    let Some((function, _)) = pass_of(id) else {
        return String::new();
    };
    let mut s = String::from(include_str!("../shaders/adaptive_ratio.wgsl"));
    s.push_str(&entry(function));
    s
}

/// The uniform block of `adaptive_ratio.wgsl` (`ArParams`): the same fields in the same order, thirteen
/// `vec4`s.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ArU {
    /// Padded width (the row stride of every plane), padded height, plane length, input width.
    dims: [u32; 4],
    /// Plane column and row of core column and row 0, core width, core height.
    geom: [u32; 4],
    /// Dispatch region `x0, y0, x1, y1` in plane coordinates.
    region: [u32; 4],
    /// Balanced input (1), matching channel, CA path (1).
    ctl: [u32; 4],
    /// Input column and row of plane column and row 0, input height.
    org: [i32; 4],
    /// `g_R, g_G, g_B` and the working scale `S`.
    gains: [f32; 4],
    /// `rho_R, rho_B, K, k`.
    cst: [f32; 4],
    /// `T`, `r`.
    cst2: [f32; 4],
    /// `(4 g_R, 2 g_G, 4 g_B)`.
    mult: [f32; 4],
    /// `g g_c`.
    vg: [f32; 4],
    /// `(r g_c)^2`.
    vr2: [f32; 4],
    /// CA: frame column and row of plane column and row 0 minus the integer part of the centre, knots
    /// of the red curve, knots of the blue curve.
    cai: [i32; 4],
    /// CA: `0.5 - fract(centre_x)`, `0.5 - fract(centre_y)`.
    caf: [f32; 4],
}

/// The margins around the core at which each stage must be evaluated (the receptive field, from the
/// output backwards): the same table as the CPU port's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Margins {
    /// The pass-3 result (2 px with shadow matching: the 5 x 5 Gaussian).
    s3: usize,
    /// Red, green and blue before the sharpening (the luma proxy reads +-1 around the result).
    rgb: usize,
    /// Step 1 of pass 3 (step 2 reads its results at +-1).
    step1: usize,
    /// Pass-2 green (read at +-2 by step 2).
    green: usize,
    /// Pass-1 green (read at +-2 by pass 2).
    green1: usize,
    /// The mosaic (read at +-5 by pass 1): 10, or 12 with shadow matching.
    mosaic: usize,
}

impl Margins {
    fn new(shadow_match: bool) -> Margins {
        let s3 = if shadow_match { 2 } else { 0 };
        let rgb = s3 + 1;
        let green = rgb + 2;
        let green1 = green + 2;
        Margins {
            s3,
            rgb,
            step1: rgb + 1,
            green,
            green1,
            mosaic: green1 + 5,
        }
    }
}

/// Where the core sits in the padded plane, and how big the plane is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    /// Plane width (the row stride).
    pw: usize,
    /// Plane height.
    ph: usize,
    /// Plane column of core column 0.
    pad_x: usize,
    /// Plane row of core row 0.
    pad_y: usize,
    /// The core in the coordinates of the uploaded rectangle.
    core: Rect,
}

impl Layout {
    /// The padded plane around `core` with `margin` pixels on the right and bottom and as many on
    /// the left and top as make red land on (even, even) of the plane (spec 3.1).
    fn new(cfa: Cfa, core: Rect, margin: usize) -> Layout {
        let rx = usize::from(
            !(cfa.color_at(0, 0) == CfaColor::Red || cfa.color_at(0, 1) == CfaColor::Red),
        );
        let ry = cfa.red_row_parity();
        let pad_x = margin + ((margin + core.x + rx) & 1);
        let pad_y = margin + ((margin + core.y + ry) & 1);
        Layout {
            pw: pad_x + core.w + margin,
            ph: pad_y + core.h + margin,
            pad_x,
            pad_y,
            core,
        }
    }

    /// The core grown by `e` pixels on every side, as a dispatch region (`x0, y0, x1, y1`). The
    /// margins keep every region inside the plane.
    fn around(&self, e: usize) -> [u32; 4] {
        [
            (self.pad_x - e) as u32,
            (self.pad_y - e) as u32,
            (self.pad_x + self.core.w + e) as u32,
            (self.pad_y + self.core.h + e) as u32,
        ]
    }
}

/// The CA curves as the shader reads them.
struct CaUpload {
    /// Red knots, then blue knots, each `(radius_px, shift_px)` rounded to `f32`.
    knots: Vec<[f32; 2]>,
    n_red: u32,
    n_blue: u32,
    /// `floor(centre_x)`, `floor(centre_y)`.
    centre_int: [i32; 2],
    /// `0.5 - fract(centre)` per axis.
    centre_frac: [f32; 2],
}

impl CaUpload {
    fn new(curves: &CaCurves) -> Result<CaUpload, GpuError> {
        let refuse = |what: &str| {
            GpuError::InvalidInput(format!(
                "adaptive ratio demosaic: the CA {what} do not fit f32"
            ))
        };
        let cx = curves.centre_x.floor();
        let cy = curves.centre_y.floor();
        // The shader forms the frame position in `i32`; a centre further out than half its range is
        // not a lens centre.
        let limit = f64::from(i32::MAX / 2);
        if !(cx.abs() < limit && cy.abs() < limit) {
            return Err(GpuError::InvalidInput(format!(
                "adaptive ratio demosaic: the optical centre ({}, {}) is out of range",
                curves.centre_x, curves.centre_y
            )));
        }
        let mut knots = Vec::with_capacity(curves.red.len() + curves.blue.len());
        for curve in [&curves.red, &curves.blue] {
            for (&r, &s) in curve.radius_px().iter().zip(curve.shift_px()) {
                let k = [r as f32, s as f32];
                if !(k[0].is_finite() && k[1].is_finite()) {
                    return Err(refuse("knots"));
                }
                knots.push(k);
            }
        }
        Ok(CaUpload {
            knots,
            n_red: curves.red.len() as u32,
            n_blue: curves.blue.len() as u32,
            centre_int: [cx as i32, cy as i32],
            centre_frac: [
                (0.5 - (curves.centre_x - cx)) as f32,
                (0.5 - (curves.centre_y - cy)) as f32,
            ],
        })
    }
}

/// The per-frame constants of the uniform block, rounded to `f32` once.
#[derive(Clone, Copy)]
struct Constants {
    /// `g_R, g_G, g_B` and the working scale `S`.
    gains: [f32; 4],
    /// `rho_R, rho_B, K, k`.
    cst: [f32; 4],
    /// `T`, `r`.
    cst2: [f32; 4],
    /// `(4 g_R, 2 g_G, 4 g_B)`.
    mult: [f32; 4],
    /// `g g_c`.
    vg: [f32; 4],
    /// `(r g_c)^2`.
    vr2: [f32; 4],
}

impl Constants {
    /// Rounds the CPU reference's `f64` constants to `f32`.
    ///
    /// The CPU reference runs in `f64` and accepts a noise model or gains whose products leave the
    /// `f32` range (a shot-noise gain of 1e50 DN per DN, a read noise of 1e40 DN, a gain of 1e30
    /// against a read noise of a few DN). Rounded to `f32` those become infinities, and the shader
    /// turns the whole image into NaN; so they are refused here, once per frame, like a sample beyond
    /// [`MAX_BALANCED_DN`].
    fn new(d: &Derived) -> Result<Constants, GpuError> {
        let g = d.gains;
        let per_channel =
            |f: &dyn Fn(usize) -> f64| -> [f32; 4] { [f(0) as f32, f(1) as f32, f(2) as f32, 0.0] };
        let c = Constants {
            gains: [g[0] as f32, g[1] as f32, g[2] as f32, WORKING_SCALE as f32],
            cst: [
                d.rho_r as f32,
                d.rho_b as f32,
                d.offset1_dn as f32,
                d.offset3_dn as f32,
            ],
            cst2: [d.gate_dn as f32, d.r_dn as f32, 0.0, 0.0],
            mult: [
                (4.0 * g[0]) as f32,
                (2.0 * g[1]) as f32,
                (4.0 * g[2]) as f32,
                0.0,
            ],
            vg: per_channel(&|c| d.g_dn * g[c]),
            vr2: per_channel(&|c| {
                let rc = d.r_dn * g[c];
                rc * rc
            }),
        };
        let fits = [c.gains, c.cst, c.cst2, c.mult, c.vg, c.vr2]
            .iter()
            .flatten()
            .all(|v| v.is_finite());
        if fits {
            Ok(c)
        } else {
            Err(GpuError::InvalidInput(format!(
                "adaptive ratio demosaic: the gains {:?} and the noise model (g {:e}, r {:e} DN) give \
                 constants that do not fit f32",
                d.gains, d.g_dn, d.r_dn
            )))
        }
    }
}

/// What the host checks and derives once per frame.
struct Frame {
    derived: Derived,
    constants: Constants,
    ca: Option<CaUpload>,
}

/// Checks the mosaic, the gains, the noise model, the parameters and the CA curves once for a whole
/// frame and derives the frame constants (spec 2.2) with the CPU reference's own code.
fn prepare(
    m: &Mosaic,
    balance: Balance,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<Frame, GpuError> {
    let invalid = |what: String| GpuError::InvalidInput(format!("adaptive ratio demosaic: {what}"));
    if m.width < 2 || m.height < 2 || m.data.len() != m.width * m.height {
        return Err(invalid(format!(
            "needs a mosaic of at least 2 x 2 photosites with one sample each, got {} x {} with {} samples",
            m.width,
            m.height,
            m.data.len()
        )));
    }
    let derived =
        Derived::new(inputs.gains, inputs.noise, params).map_err(|e| invalid(e.to_string()))?;
    let constants = Constants::new(&derived)?;
    let ca = inputs.ca.map(CaUpload::new).transpose()?;
    let gain_max = derived.gains.iter().copied().fold(0.0f64, f64::max);
    for (y, row) in m.data.chunks_exact(m.width).enumerate() {
        for (x, &v) in row.iter().enumerate() {
            let unbalanced = match balance {
                Balance::Unbalanced => v,
                Balance::Balanced => {
                    (f64::from(v) / derived.gains[m.cfa.color_at(x, y).plane()]) as f32
                }
            };
            let scaled = WORKING_SCALE * f64::from(unbalanced).abs() * gain_max;
            // The negated comparison also refuses NaN.
            if !(v.is_finite() && scaled <= MAX_BALANCED_DN) {
                return Err(invalid(format!(
                    "the photosite ({x}, {y}) holds {v}, which is not finite or exceeds the \
                     {MAX_BALANCED_DN:e} DN the kernel accepts"
                )));
            }
        }
    }
    Ok(Frame {
        derived,
        constants,
        ca,
    })
}

/// Everything a dispatch of one rectangle needs besides the buffers.
struct RectJob<'a> {
    m: &'a Mosaic,
    core: Rect,
    balance: Balance,
    frame: &'a Frame,
    params: &'a Params,
}

/// The uniform block shared by the passes of one rectangle; the region and the channel are set per
/// pass.
fn base_uniform(job: &RectJob<'_>, lay: &Layout) -> ArU {
    let k = job.frame.constants;
    let (cai, caf) = match &job.frame.ca {
        Some(ca) => {
            let place = job.m.place;
            (
                [
                    place.x0 as i32 + job.core.x as i32 - lay.pad_x as i32 - ca.centre_int[0],
                    place.y0 as i32 + job.core.y as i32 - lay.pad_y as i32 - ca.centre_int[1],
                    ca.n_red as i32,
                    ca.n_blue as i32,
                ],
                [ca.centre_frac[0], ca.centre_frac[1], 0.0, 0.0],
            )
        }
        None => ([0; 4], [0.0; 4]),
    };
    ArU {
        dims: [
            lay.pw as u32,
            lay.ph as u32,
            (lay.pw * lay.ph) as u32,
            job.m.width as u32,
        ],
        geom: [
            lay.pad_x as u32,
            lay.pad_y as u32,
            job.core.w as u32,
            job.core.h as u32,
        ],
        region: [0; 4],
        ctl: [
            u32::from(job.balance == Balance::Balanced),
            0,
            u32::from(job.frame.ca.is_some()),
            0,
        ],
        org: [
            job.core.x as i32 - lay.pad_x as i32,
            job.core.y as i32 - lay.pad_y as i32,
            job.m.height as i32,
            0,
        ],
        gains: k.gains,
        cst: k.cst,
        cst2: k.cst2,
        mult: k.mult,
        vg: k.vg,
        vr2: k.vr2,
        cai,
        caf,
    }
}

/// The command encoder and the buffers of one rectangle's passes.
struct Passes<'a> {
    ctx: &'a GpuContext,
    encoder: wgpu::CommandEncoder,
    base: ArU,
    input: &'a PooledBuffer,
    scratch: &'a PooledBuffer,
    out: &'a PooledBuffer,
    curves: &'a PooledBuffer,
    /// The uniform buffers the submitted work reads; kept alive until the readback finished.
    held: Vec<PooledBuffer>,
}

impl Passes<'_> {
    /// Encodes one dispatch of the pass `id` over `region` (plane coordinates, half open) for the
    /// matching channel `channel`.
    fn run(&mut self, id: KernelId, region: [u32; 4], channel: u32) -> Result<(), GpuError> {
        let Some((_, slots)) = pass_of(id) else {
            return Err(GpuError::InvalidInput(format!(
                "{id:?} is not an adaptive ratio pass"
            )));
        };
        let pipeline = self.ctx.kernels().pipeline(self.ctx, id)?;
        let mut u = self.base;
        u.region = region;
        u.ctl[1] = channel;
        let ub = uniform_buffer(self.ctx, "adaptive ratio uniform", &u)?;
        let group = {
            let entries: Vec<(u32, wgpu::BindingResource<'_>)> = slots
                .iter()
                .map(|&slot| {
                    let resource = match slot {
                        Slot::Params => ub.binding(),
                        Slot::Input => self.input.binding(),
                        Slot::Scratch => self.scratch.binding(),
                        Slot::Out => self.out.binding(),
                        Slot::Curves => self.curves.binding(),
                    };
                    (slot as u32, resource)
                })
                .collect();
            bind(self.ctx, &pipeline, id.name(), &entries)
        };
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(id.name()),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                (region[2] - region[0]).div_ceil(GROUP),
                (region[3] - region[1]).div_ceil(GROUP),
                1,
            );
        }
        self.held.push(ub);
        Ok(())
    }
}

/// Runs the kernel on the `job.core` rectangle of `job.m` and reads the core back. The rectangle
/// around the core supplies the context (real rows and columns, or the buffer edge, where it
/// mirrors).
fn run_rect(ctx: &GpuContext, job: &RectJob<'_>) -> Result<RgbImage, GpuError> {
    let d = &job.frame.derived;
    let mg = Margins::new(d.shadow_match);
    let lay = Layout::new(job.m.cfa, job.core, mg.mosaic);

    // One workgroup covers GROUP x GROUP positions; a side of more than `limit` workgroups is a
    // validation error that wgpu raises as a panic, so refuse it here (a 2 x N frame with N beyond
    // 16 x 65535 on the default limits).
    let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
    let groups = |side: usize| side.div_ceil(GROUP as usize);
    if groups(lay.pw) > limit || groups(lay.ph) > limit {
        return Err(GpuError::InvalidInput(format!(
            "adaptive ratio demosaic: a {} x {} padded plane needs more than {limit} workgroups of \
             {GROUP} positions on one side; use demosaic_tiled with smaller tiles",
            lay.pw, lay.ph
        )));
    }
    let planes = if d.shadow_match {
        PLANES_MATCH
    } else {
        PLANES_PLAIN
    };
    let plane_len = lay.pw as u64 * lay.ph as u64;
    // The shader indexes the scratch buffer with u32.
    if plane_len * planes as u64 > u64::from(u32::MAX) {
        return Err(GpuError::BufferTooLarge {
            requested: plane_len * planes as u64 * 4,
            limit: u64::from(u32::MAX) * 4,
        });
    }

    let m = job.m;
    let input = ctx
        .pool()
        .acquire("adaptive ratio mosaic", m.data.len() as u64 * 4, STORAGE_IN)?;
    ctx.queue
        .write_buffer(&input, 0, bytemuck::cast_slice(&m.data));
    let scratch = ctx.pool().acquire(
        "adaptive ratio planes",
        plane_len * planes as u64 * 4,
        STORAGE_SCRATCH,
    )?;
    let core_px = (job.core.w * job.core.h) as u64;
    let out = ctx
        .pool()
        .acquire("adaptive ratio rgb", core_px * 3 * 4, STORAGE_OUT)?;
    // The knots; a binding must exist even without CA, and holds one unused knot then.
    let dummy = [[0.0f32; 2]];
    let knots: &[[f32; 2]] = job.frame.ca.as_ref().map_or(&dummy, |c| &c.knots);
    let curves = ctx
        .pool()
        .acquire("adaptive ratio curves", knots.len() as u64 * 8, STORAGE_IN)?;
    ctx.queue
        .write_buffer(&curves, 0, bytemuck::cast_slice(knots));

    let mut passes = Passes {
        ctx,
        encoder: ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adaptive ratio demosaic"),
            }),
        base: base_uniform(job, &lay),
        input: &input,
        scratch: &scratch,
        out: &out,
        curves: &curves,
        held: Vec::with_capacity(20),
    };
    passes.run(
        KernelId::AdaptiveRatioPrepare,
        [0, 0, lay.pw as u32, lay.ph as u32],
        0,
    )?;
    passes.run(KernelId::AdaptiveRatioGreen, lay.around(mg.green1), 0)?;
    passes.run(KernelId::AdaptiveRatioRefine, lay.around(mg.green), 0)?;
    passes.run(KernelId::AdaptiveRatioStep1, lay.around(mg.step1), 0)?;
    passes.run(KernelId::AdaptiveRatioStep2, lay.around(mg.rgb), 0)?;
    if job.params.shadow_sharpening {
        passes.run(KernelId::AdaptiveRatioLuma, lay.around(mg.rgb), 0)?;
        passes.run(KernelId::AdaptiveRatioSharpen, lay.around(mg.s3), 0)?;
    }
    if d.shadow_match {
        for channel in 0..3 {
            passes.run(KernelId::AdaptiveRatioSimple, lay.around(2), channel)?;
            passes.run(KernelId::AdaptiveRatioLowErr, lay.around(1), channel)?;
        }
        passes.run(KernelId::AdaptiveRatioMatch, lay.around(0), 0)?;
    } else {
        passes.run(KernelId::AdaptiveRatioExit, lay.around(0), 0)?;
    }
    let Passes { encoder, held, .. } = passes;

    let bytes = core_px * 3 * 4;
    let readback = submit_with_readback(ctx, encoder, &out, bytes)?;
    let samples = image_samples(StorageMode::F32, &readback.finish(ctx)?);
    drop(held);
    let mut img = RgbImage::new(job.core.w, job.core.h);
    if samples.len() != img.data.len() {
        return Err(GpuError::Readback(format!(
            "adaptive ratio demosaic: read back {} samples, expected {}",
            samples.len(),
            img.data.len()
        )));
    }
    img.data.copy_from_slice(&samples);
    img.place = m.place.offset(job.core.x, job.core.y);
    Ok(img)
}

/// Demosaics `m` in one go: the whole buffer is the core, mirrored at its edges (spec 3.1).
///
/// `inputs.gains` are the R4 gains `g0` (the output is multiplied by them), `inputs.noise` the noise
/// model on the 16-bit working scale and `inputs.ca` the CA curves of the lateral CA path, whose
/// positions come from `m.place` (`None` is the plain path). `params` are the tunables of spec 2.1. The
/// output has the mosaic's size and [`Placement`](crate::cpu::Placement) and is white-balanced
/// (`balance` says whether the input was). The padded planes need 20 to 48 bytes per pixel in one
/// storage binding: for a big frame use [`demosaic_tiled`].
///
/// This is the GPU counterpart of
/// [`adaptive_ratio::demosaic_whole`](crate::cpu::demosaic::adaptive_ratio::demosaic_whole)
/// ([`Balance::Unbalanced`]) and
/// [`demosaic_balanced_whole`](crate::cpu::demosaic::adaptive_ratio::demosaic_balanced_whole)
/// ([`Balance::Balanced`]); see the module documentation for how they differ.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a mosaic below 2 x 2 photosites, a value that is not finite or above
/// [`MAX_BALANCED_DN`], unusable gains, an invalid noise model, gains or a noise model whose constants
/// do not fit `f32`, non-finite parameters, an unusable CA centre or knots, or a padded plane that
/// needs more workgroups on one side than the device allows;
/// [`GpuError::BufferTooLarge`] and [`GpuError::OutOfBudget`] when the buffers do not fit the device;
/// shader and readback errors.
pub fn demosaic(
    ctx: &GpuContext,
    m: &Mosaic,
    balance: Balance,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<RgbImage, GpuError> {
    let frame = prepare(m, balance, inputs, params)?;
    run_rect(
        ctx,
        &RectJob {
            m,
            core: Rect::whole(m.width, m.height),
            balance,
            frame: &frame,
            params,
        },
    )
}

/// Demosaics `m` tile by tile: cores of at most `tile_px` (rounded down to an even number), each
/// computed from a rectangle that carries [`Method::AdaptiveRatio`]'s halo of context (13 px, clamped
/// to the frame, left and top edges even so the Bayer phase is unchanged) and pasted into the result.
///
/// Gives the same pixels as [`demosaic`] on the whole frame, bit for bit. Tiles are processed one
/// after the other; `cancel` is checked between them. The mosaic is checked once, not per tile.
///
/// # Errors
///
/// [`GpuError::Cancelled`] when `cancel` fires, plus the errors of [`demosaic`].
pub fn demosaic_tiled(
    ctx: &GpuContext,
    m: &Mosaic,
    balance: Balance,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
    tile_px: usize,
    cancel: &CancelToken,
) -> Result<RgbImage, GpuError> {
    // A mosaic below 2 x 2 has no meaningful plan; `prepare` reports it.
    let plan = TilePlan::new(
        m.width.max(2),
        m.height.max(2),
        tile_px,
        &[Stage::Demosaic(Method::AdaptiveRatio)],
    );
    demosaic_planned(ctx, m, balance, inputs, params, &plan, cancel)
}

/// Like [`demosaic_tiled`] with a plan made by the caller, which must be for the size of `m` and
/// give every tile at least [`Derived::halo_px`] pixels of context around its core (11 px, 13 with
/// shadow matching; a plan built with `Stage::Demosaic(Method::AdaptiveRatio)` has 13).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the plan is for another size or a tile lacks context (the kernel
/// mirrors at the edge of the tile's rectangle, so missing context would show as seams, not as an
/// error); [`GpuError::Cancelled`] when `cancel` fires; plus the errors of [`demosaic`].
pub fn demosaic_planned(
    ctx: &GpuContext,
    m: &Mosaic,
    balance: Balance,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
    plan: &TilePlan,
    cancel: &CancelToken,
) -> Result<RgbImage, GpuError> {
    let frame = prepare(m, balance, inputs, params)?;
    let rect = plan.frame();
    if (rect.w, rect.h) != (m.width, m.height) {
        return Err(GpuError::InvalidInput(format!(
            "tile plan is for {}x{}, the mosaic is {}x{}",
            rect.w, rect.h, m.width, m.height
        )));
    }
    let halo_px = frame.derived.halo_px;
    for tile in plan.tiles() {
        let needed = tile.core.grow(Halo::uniform(halo_px), rect);
        if !tile.compute.contains(&needed) {
            return Err(GpuError::InvalidInput(format!(
                "tile {} computes {:?}, but its core {:?} needs {halo_px} px of context ({needed:?})",
                tile.index, tile.compute, tile.core
            )));
        }
    }
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    for &tile in plan.tiles() {
        cancel.check()?;
        let sub = m.extract(tile.compute);
        let core = tile.core_in_compute();
        let img = run_rect(
            ctx,
            &RectJob {
                m: &sub,
                core,
                balance,
                frame: &frame,
                params,
            },
        )?;
        out.paste(&img, Rect::whole(core.w, core.h), tile.core.x, tile.core.y);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::kernels::shader_source;

    /// Every adaptive ratio pass, as registered in [`KernelId`].
    fn passes() -> Vec<KernelId> {
        let all: Vec<KernelId> = KernelId::ALL
            .into_iter()
            .filter(|id| id.name().starts_with("adaptive_ratio"))
            .collect();
        assert_eq!(all.len(), 11, "eleven passes are registered");
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
    fn the_uniform_block_has_the_layout_of_ar_u() {
        let (module, _) = module_of(KernelId::AdaptiveRatioPrepare);
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("u"))
            .expect("the uniform `u`");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`u` is not a struct");
        };
        let fields: [(&str, usize); 13] = [
            ("dims", offset_of!(ArU, dims)),
            ("geom", offset_of!(ArU, geom)),
            ("region", offset_of!(ArU, region)),
            ("ctl", offset_of!(ArU, ctl)),
            ("org", offset_of!(ArU, org)),
            ("gains", offset_of!(ArU, gains)),
            ("cst", offset_of!(ArU, cst)),
            ("cst2", offset_of!(ArU, cst2)),
            ("mult", offset_of!(ArU, mult)),
            ("vg", offset_of!(ArU, vg)),
            ("vr2", offset_of!(ArU, vr2)),
            ("cai", offset_of!(ArU, cai)),
            ("caf", offset_of!(ArU, caf)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<ArU>());
    }

    #[test]
    fn the_shader_is_f32_only_in_every_storage_mode() {
        // Planes are f32 whatever the storage mode (the decisions compare sums for order), so the
        // text has no f16 and no accessor expansion beyond the generated prelude.
        let body = include_str!("../shaders/adaptive_ratio.wgsl");
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
    fn margins_and_the_padded_layout_follow_the_cpu_port() {
        // The reach is the spec's sum of halos minus one (11 and 13 are safe bounds).
        let plain = Margins::new(false);
        let matching = Margins::new(true);
        assert_eq!((plain.mosaic, matching.mosaic), (10, 12));
        assert_eq!((plain.s3, plain.rgb, plain.step1), (0, 1, 2));
        assert_eq!((plain.green, plain.green1), (3, 5));
        assert_eq!((matching.s3, matching.rgb, matching.green1), (2, 3, 7));
        // Red lands on (even, even) of the plane for every pattern and core origin.
        for cfa in Cfa::ALL {
            for core in [Rect::new(0, 0, 9, 11), Rect::new(1, 3, 6, 5)] {
                for margin in [10, 12] {
                    let lay = Layout::new(cfa, core, margin);
                    for dy in 0..core.h {
                        for dx in 0..core.w {
                            let want = cfa.color_at(core.x + dx, core.y + dy);
                            let (px, py) = (lay.pad_x + dx, lay.pad_y + dy);
                            let got = match (px & 1, py & 1) {
                                (0, 0) => CfaColor::Red,
                                (1, 1) => CfaColor::Blue,
                                _ => CfaColor::Green,
                            };
                            assert_eq!(got, want, "{} core {core:?} ({dx},{dy})", cfa.name());
                        }
                    }
                    assert!(lay.pad_x >= margin && lay.pw - lay.pad_x - core.w >= margin);
                    assert!(lay.pad_y >= margin && lay.ph - lay.pad_y - core.h >= margin);
                    let r = lay.around(margin);
                    assert!(
                        r[0] <= 1
                            && r[1] <= 1
                            && r[2] as usize <= lay.pw
                            && r[3] as usize <= lay.ph
                    );
                }
            }
        }
    }
}
