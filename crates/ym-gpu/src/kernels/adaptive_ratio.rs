//! The adaptive ratio demosaic on the GPU: the WGSL port of
//! [`crate::cpu::demosaic::adaptive_ratio`], specified in
//! `docs/algorithms/01-adaptive-ratio-demosaic.md` (section 5 lists the GPU notes, section 11 the CA
//! path). It is R6's classical demosaic, `Method::AdaptiveRatio`, including the CA-aware green
//! refinement, the shadow sharpening and the shadow colour matching, and the entry that undoes R4 on
//! the balanced mosaic the chain hands to R6 ([`Balance::Balanced`]).
//!
//! The render runs it through [`crate::chain::GpuChain`]: [`encode_demosaic`] is the device-in,
//! device-out entry point (`docs/17-gpu.md` section 3), and the host wrappers ([`demosaic`],
//! [`demosaic_tiled`], [`demosaic_planned`]) upload a mosaic, run the passes and read the planes back,
//! for the parity tests and for the chain's hand-off.
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
//! * **Fast-math (Metal, K8).** Metal compiles with fast-math: it fuses `a * b + c`, re-associates
//!   sums and products, and divides and takes square roots to a few ulp (WGSL allows 2.5 ulp for a
//!   division). Every one of those moved a near-tie of the decisions above, and every later value is
//!   computed from what a decision chose: at b6f2e4b the Apple runners counted 40 samples of the
//!   `edge_knight` vector over 1e-4 (lavapipe 10) and the device chain's strong-lens corner 921 (lavapipe
//!   293), up to 2.07e-2. So every product that meets a sum, every partial sum and every difference
//!   passes through a barrier (`ar_op`, an OR with the always-zero third word of `cst2`, which the
//!   compiler cannot know; an OR because an XOR applied twice cancels), and every quotient and the
//!   CA square root are made correctly rounded with integer arithmetic (`ar_div`: the device's estimate
//!   corrected with the exact integer remainder; `ar_sqrt`: a digit-by-digit integer root). IEEE
//!   arithmetic is unchanged. Evidence without a Mac: naga's MSL of the eleven passes, compiled with
//!   clang `-O2` and `-O3 -ffast-math -ffp-contract=fast -mfma -mrecip=all`, gives lavapipe's bits on
//!   all 18 vectors, 240 random frames and the three demosaic calls of the strong-lens corner test
//!   (24.6 million samples); the MSL of the kernel before the change reproduces Metal's worst
//!   `edge_knight` and `ca_weighting` differences (6.595e-4, 1.078e-3) to the digit, and with the
//!   barriers but the device's division the strong-lens counts come back (3606 samples off by more
//!   than 1e-4). The cost: about 60 exact divisions per pixel (about 25 integer operations each); the
//!   scalar CPU model of the passes runs 3.9 times as long as before. Mac timing is an M0-H4 item.
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
use super::util::{GROUP, even, storage_buffer};
use crate::cpu::demosaic::adaptive_ratio::{Derived, Params};
use crate::cpu::demosaic::{CaCurves, DemosaicInputs, Method};
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::tiling::{Halo, Rect, Stage};
use crate::cpu::{Cfa, CfaColor, Mosaic, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::queue::CancelToken;
use crate::stream::PassList;
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
    /// `T`, `r`, then `0.0`: the shader's fast-math barrier ORs the bits of the third word in
    /// (`ar_op` in `adaptive_ratio.wgsl`), which the compiler cannot know; it must stay `+0.0`.
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Copy, Debug)]
struct Constants {
    /// `g_R, g_G, g_B` and the working scale `S`.
    gains: [f32; 4],
    /// `rho_R, rho_B, K, k`.
    cst: [f32; 4],
    /// `T`, `r`, then `0.0`: the shader's fast-math barrier ORs the bits of the third word in
    /// (`ar_op` in `adaptive_ratio.wgsl`), which the compiler cannot know; it must stay `+0.0`.
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

/// The per-frame state of the demosaic, made once on the host: the constants of spec 2.2 derived with
/// the CPU reference's own code, the same constants rounded to `f32` for the uniform, the CA curves as
/// the shader reads them, and whether the input is balanced. Every rectangle of the frame (every tile,
/// every half of a split tile) uses the same value, so tiles agree with the whole frame.
///
/// Owned and `Send`: a device job can hold it.
#[derive(Clone, Debug)]
pub struct PreparedDemosaic {
    derived: Derived,
    constants: Constants,
    ca: Option<CaUpload>,
    balance: Balance,
    shadow_sharpening: bool,
}

impl PreparedDemosaic {
    /// Derives the frame constants from the gains, the noise model and the CA curves in `inputs` and
    /// the tunables in `params`.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for unusable gains, an invalid noise model, non-finite parameters,
    /// gains or a noise model whose constants do not fit `f32`, or an unusable CA centre or knots.
    pub fn new(
        balance: Balance,
        inputs: &DemosaicInputs<'_>,
        params: &Params,
    ) -> Result<PreparedDemosaic, GpuError> {
        let invalid =
            |what: String| GpuError::InvalidInput(format!("adaptive ratio demosaic: {what}"));
        let derived =
            Derived::new(inputs.gains, inputs.noise, params).map_err(|e| invalid(e.to_string()))?;
        let constants = Constants::new(&derived)?;
        let ca = inputs.ca.map(CaUpload::new).transpose()?;
        Ok(PreparedDemosaic {
            derived,
            constants,
            ca,
            balance,
            shadow_sharpening: params.shadow_sharpening,
        })
    }

    /// The context in pixels a tile needs around its core for the same pixels as the whole frame
    /// (11, or 13 with shadow colour matching).
    #[must_use]
    pub fn halo_px(&self) -> usize {
        self.derived.halo_px
    }

    /// Whether shadow colour matching runs for this frame (it needs 12 planes of scratch instead of 5).
    #[must_use]
    pub fn shadow_match(&self) -> bool {
        self.derived.shadow_match
    }

    /// Whether the input is the balanced mosaic of R4/R5 or the unbalanced one.
    #[must_use]
    pub fn balance(&self) -> Balance {
        self.balance
    }

    /// Checks the samples of a host mosaic once per frame or tile: the shape, no NaN or infinity, and
    /// no balanced magnitude above [`MAX_BALANCED_DN`]. A device image cannot be scanned here; the
    /// host that uploaded it did (this function on its mosaic).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`].
    pub fn check_mosaic(&self, m: &Mosaic) -> Result<(), GpuError> {
        check_shape(m)?;
        let invalid =
            |what: String| GpuError::InvalidInput(format!("adaptive ratio demosaic: {what}"));
        let gain_max = self.derived.gains.iter().copied().fold(0.0f64, f64::max);
        for (y, row) in m.data.chunks_exact(m.width).enumerate() {
            for (x, &v) in row.iter().enumerate() {
                let unbalanced = match self.balance {
                    Balance::Unbalanced => v,
                    Balance::Balanced => {
                        (f64::from(v) / self.derived.gains[m.cfa.color_at(x, y).plane()]) as f32
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
        Ok(())
    }
}

/// The shape check of a host mosaic: at least 2 x 2 photosites, one sample each.
fn check_shape(m: &Mosaic) -> Result<(), GpuError> {
    if m.width < 2 || m.height < 2 || m.data.len() != m.width * m.height {
        return Err(GpuError::InvalidInput(format!(
            "adaptive ratio demosaic: needs a mosaic of at least 2 x 2 photosites with one sample each, got {} x {} with {} samples",
            m.width,
            m.height,
            m.data.len()
        )));
    }
    Ok(())
}

/// Checks the mosaic, the gains, the noise model, the parameters and the CA curves once for a whole
/// frame and derives the frame constants (spec 2.2) with the CPU reference's own code.
fn prepare(
    m: &Mosaic,
    balance: Balance,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<PreparedDemosaic, GpuError> {
    check_shape(m)?;
    let frame = PreparedDemosaic::new(balance, inputs, params)?;
    frame.check_mosaic(m)?;
    Ok(frame)
}

/// The padded plane of the scratch buffer for a core of `core`: its footprint in the binding the
/// 128 MiB design limit bounds. At most `(w + 25) (h + 25)` pixels for any configuration (the margin
/// is 10, or 12 with shadow colour matching, and the left and top pad may add one more).
#[must_use]
pub fn padded_pixels(core: Rect) -> u64 {
    (core.w as u64 + 2 * MAX_MARGIN_PX as u64 + 1) * (core.h as u64 + 2 * MAX_MARGIN_PX as u64 + 1)
}

/// The largest storage binding (bytes) a run for a core of `core` needs: the 12 planes of the padded
/// scratch with shadow colour matching (the input, 4 bytes per photosite of the rectangle that
/// carries [`Derived::halo_px`] pixels of context, and the three output planes of the core are
/// smaller). A test checks it against the 128 MiB design binding at
/// [`crate::chain::MAX_PHYS_AREA`] (`docs/17-gpu.md` K4).
#[must_use]
pub fn max_binding_bytes(core: Rect) -> u64 {
    let scratch = padded_pixels(core) * PLANES_MATCH as u64 * 4;
    let compute = Rect::new(0, 0, core.w + 2 * HALO_MAX_PX, core.h + 2 * HALO_MAX_PX);
    let input = even(compute.w) as u64 * compute.h as u64 * 4;
    let output = 3 * even(core.w) as u64 * core.h as u64 * 4;
    scratch.max(input).max(output)
}

/// The most margin around a core the planes ever hold (shadow colour matching).
const MAX_MARGIN_PX: usize = 12;

/// The halo of a tile with shadow colour matching.
const HALO_MAX_PX: usize = crate::cpu::demosaic::adaptive_ratio::HALO_MAX_PX;

/// The uniform block shared by the passes of one rectangle; the region and the channel are set per
/// pass.
fn base_uniform(
    input: &GpuImage,
    core: Rect,
    frame: &PreparedDemosaic,
    lay: &Layout,
    out_stride: usize,
) -> ArU {
    let k = frame.constants;
    let (cai, caf) = match &frame.ca {
        Some(ca) => {
            let place = input.place;
            (
                [
                    place.x0 as i32 + core.x as i32 - lay.pad_x as i32 - ca.centre_int[0],
                    place.y0 as i32 + core.y as i32 - lay.pad_y as i32 - ca.centre_int[1],
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
            input.width as u32,
        ],
        geom: [
            lay.pad_x as u32,
            lay.pad_y as u32,
            core.w as u32,
            core.h as u32,
        ],
        region: [0; 4],
        ctl: [
            u32::from(frame.balance == Balance::Balanced),
            0,
            u32::from(frame.ca.is_some()),
            out_stride as u32,
        ],
        org: [
            core.x as i32 - lay.pad_x as i32,
            core.y as i32 - lay.pad_y as i32,
            input.height as i32,
            input.stride as i32,
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

/// Records the passes of the adaptive ratio demosaic for the `core` rectangle of a mosaic that is on
/// the device, and returns the three planes of the core as a new [`GpuImage`] (device in, device out:
/// the host wrappers [`demosaic`], [`demosaic_tiled`] and [`demosaic_planned`] are a thin layer on
/// this).
///
/// `input` is the one-plane mosaic of the compute rectangle in `f32` (an even row stride is fine),
/// `cfa` the pattern at its first photosite, `core` the part of it to compute, in its coordinates:
/// the rest is context, mirrored at the edge of the rectangle, so a core that is not at the edge of
/// the frame needs [`PreparedDemosaic::halo_px`] real pixels around it. The output keeps the
/// mosaic's [`Placement`](crate::cpu::Placement) moved to the core, is balanced like the input
/// (`balance` of `frame`) and has the even stride of every [`GpuImage`].
///
/// Nothing is submitted: the caller adds readbacks or further passes and submits the list. The
/// samples are not checked here (they are on the device); check the host mosaic with
/// [`PreparedDemosaic::check_mosaic`] before uploading it.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not one plane, below 2 x 2, a core outside it or
/// empty, or a padded plane that needs more workgroups on one side than the device allows;
/// [`GpuError::BufferTooLarge`] and [`GpuError::OutOfBudget`] when the buffers do not fit the device;
/// shader errors.
pub fn encode_demosaic(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    cfa: Cfa,
    core: Rect,
    frame: &PreparedDemosaic,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if input.planes != 1 || input.width < 2 || input.height < 2 {
        return Err(GpuError::InvalidInput(format!(
            "adaptive ratio demosaic: the input must be one plane of at least 2 x 2 photosites, got {} plane(s) of {} x {}",
            input.planes, input.width, input.height
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "adaptive ratio demosaic: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    let d = &frame.derived;
    let mg = Margins::new(d.shadow_match);
    let lay = Layout::new(cfa, core, mg.mosaic);

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

    let scratch = passes.scratch("adaptive ratio planes", plane_len * planes as u64 * 4)?;
    let out = GpuImage::alloc(
        ctx,
        "adaptive ratio rgb",
        core.w,
        core.h,
        3,
        input.place.offset(core.x, core.y),
    )?;
    // The knots; a binding must exist even without CA, and holds one unused knot then.
    let dummy = [[0.0f32; 2]];
    let knots: &[[f32; 2]] = frame.ca.as_ref().map_or(&dummy, |c| &c.knots);
    let curves = passes.keep(storage_buffer(
        ctx,
        "adaptive ratio curves",
        bytemuck::cast_slice(knots),
    )?);
    let base = base_uniform(input, core, frame, &lay, out.stride);

    let mut run = |id: KernelId, region: [u32; 4], channel: u32| -> Result<(), GpuError> {
        let Some((_, slots)) = pass_of(id) else {
            return Err(GpuError::InvalidInput(format!(
                "{id:?} is not an adaptive ratio pass"
            )));
        };
        let mut u = base;
        u.region = region;
        u.ctl[1] = channel;
        let ub = passes.uniform("adaptive ratio uniform", &u)?;
        let entries: Vec<(u32, wgpu::BindingResource<'_>)> = slots
            .iter()
            .map(|&slot| {
                let resource = match slot {
                    Slot::Params => ub.binding(),
                    Slot::Input => input.buf.binding(),
                    Slot::Scratch => scratch.binding(),
                    Slot::Out => out.buf.binding(),
                    Slot::Curves => curves.binding(),
                };
                (slot as u32, resource)
            })
            .collect();
        passes.dispatch(
            id,
            id.name(),
            &entries,
            (
                (region[2] - region[0]).div_ceil(GROUP),
                (region[3] - region[1]).div_ceil(GROUP),
                1,
            ),
        )
    };
    run(
        KernelId::AdaptiveRatioPrepare,
        [0, 0, lay.pw as u32, lay.ph as u32],
        0,
    )?;
    run(KernelId::AdaptiveRatioGreen, lay.around(mg.green1), 0)?;
    run(KernelId::AdaptiveRatioRefine, lay.around(mg.green), 0)?;
    run(KernelId::AdaptiveRatioStep1, lay.around(mg.step1), 0)?;
    run(KernelId::AdaptiveRatioStep2, lay.around(mg.rgb), 0)?;
    if frame.shadow_sharpening {
        run(KernelId::AdaptiveRatioLuma, lay.around(mg.rgb), 0)?;
        run(KernelId::AdaptiveRatioSharpen, lay.around(mg.s3), 0)?;
    }
    if d.shadow_match {
        for channel in 0..3 {
            run(KernelId::AdaptiveRatioSimple, lay.around(2), channel)?;
            run(KernelId::AdaptiveRatioLowErr, lay.around(1), channel)?;
        }
        run(KernelId::AdaptiveRatioMatch, lay.around(0), 0)?;
    } else {
        run(KernelId::AdaptiveRatioExit, lay.around(0), 0)?;
    }
    Ok(out)
}

/// Runs the kernel on the `core` rectangle of the host mosaic `m` and reads the core back: upload,
/// [`encode_demosaic`], submit, read. The host path every wrapper and the device chain use; the
/// samples were checked by the caller ([`PreparedDemosaic::check_mosaic`]). The rectangle around the core supplies the context (real rows
/// and columns, or the buffer edge, where it mirrors).
pub(crate) fn demosaic_core(
    ctx: &GpuContext,
    m: &Mosaic,
    core: Rect,
    frame: &PreparedDemosaic,
) -> Result<RgbImage, GpuError> {
    let input = GpuImage::upload(
        ctx,
        "adaptive ratio mosaic",
        m.width,
        m.height,
        1,
        &m.data,
        m.place,
    )?;
    let mut passes = PassList::new(ctx, "adaptive ratio demosaic");
    let out = encode_demosaic(&mut passes, &input, m.cfa, core, frame)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = out.take(&done, read)?;
    Ok(RgbImage {
        width: out.width,
        height: out.height,
        place: out.place,
        data,
    })
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
    demosaic_core(ctx, m, Rect::whole(m.width, m.height), &frame)
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
    let halo_px = frame.halo_px();
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
        let img = demosaic_core(ctx, &sub, core, &frame)?;
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
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    /// The text of the WGSL function `name` in `src`, up to its closing brace.
    fn wgsl_function<'a>(src: &'a str, name: &str) -> &'a str {
        let start = src
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("fn {name}"));
        let end = src[start..]
            .find("\n}\n")
            .map(|e| start + e + 3)
            .expect("end");
        &src[start..end]
    }

    /// K8 (`docs/17-gpu.md`): every fast-math barrier of every kernel is an OR with a run-time
    /// zero, never an XOR. An XOR applied twice cancels (`x ^ z ^ z == x`, which a compiler folds)
    /// and fast-math folds the `0 + x` that starts a running sum, so a product lost its barrier and
    /// was fused into the next sum (the Metal model of the adaptive ratio: 617 935 samples of 2 638
    /// demosaic calls off the written order, against 0 with the OR). A barrier is a function whose
    /// body is `bitcast<f32>(bitcast<u32>(x) OP word)`; the test scans the text of every kernel,
    /// and it must find the six kernel families that have one (so that it cannot pass by finding
    /// none).
    #[test]
    fn every_fast_math_barrier_is_an_or_never_an_xor() {
        let mut families = BTreeSet::new();
        for id in KernelId::ALL {
            let src = shader_source(StorageMode::F32, id);
            for (at, _) in src.match_indices("bitcast<f32>(bitcast<u32>(x)") {
                let rest = &src[at + "bitcast<f32>(bitcast<u32>(x)".len()..];
                let op = rest.trim_start().chars().next().expect("an operator");
                assert_eq!(
                    op,
                    '|',
                    "{id:?}: a barrier must OR the run-time zero, found `{}`",
                    &src[at..at + 60.min(src.len() - at)]
                );
                let name = id.name();
                families.insert(
                    [
                        "adaptive_ratio",
                        "output_sharpen",
                        "log_guided",
                        "detail",
                        "guided",
                        "diffraction",
                        "prep",
                        "highlights",
                        "reduce",
                        "mosaic_ca",
                        "lens",
                        "sigma",
                        "bin",
                        "magnify",
                    ]
                    .into_iter()
                    .find(|f| name.starts_with(f))
                    .unwrap_or_else(|| panic!("{name} has a barrier but is in no known family")),
                );
            }
            assert!(
                !src.contains("bitcast<u32>(x) ^"),
                "{id:?}: an XOR barrier (x ^ z ^ z cancels)"
            );
        }
        assert_eq!(
            families.into_iter().collect::<Vec<_>>(),
            [
                "adaptive_ratio",
                "bin",
                "detail",
                "diffraction",
                "guided",
                "highlights",
                "lens",
                "log_guided",
                "magnify",
                "mosaic_ca",
                "output_sharpen",
                "prep",
                "reduce",
                "sigma"
            ],
            "the kernels with a fast-math barrier"
        );
    }

    /// The barrier of the adaptive ratio shader (`ar_op`) ORs the bits of `cst2.z` into every value it
    /// hides. The host must send `+0.0` there for every frame: any other bit pattern would corrupt
    /// every pinned value, and `-0.0` would set every sign bit. Nothing but this test reads the word
    /// back, and lavapipe cannot tell a wrong word from a right one until the shader is changed.
    #[test]
    fn the_barrier_word_of_the_uniform_is_plus_zero_for_every_frame() {
        use crate::cpu::NoiseDn;
        use crate::cpu::demosaic::adaptive_ratio::ShadowMatch;
        let noises = [
            NoiseDn::ZERO,
            NoiseDn::new(0.07, 3.1),
            NoiseDn::new(120.0, 17_202.0),
        ];
        let gains = [[1.0f32, 1.0, 1.0], [2.1, 1.0, 1.7], [0.5, 0.4, 3.0]];
        let mut frames = 0;
        for noise in noises {
            for g in gains {
                for shadow_match in [ShadowMatch::Off, ShadowMatch::On] {
                    let params = Params {
                        shadow_match,
                        ..Params::default()
                    };
                    let derived = Derived::new(g, noise, &params).expect("derived");
                    let c = Constants::new(&derived).expect("constants");
                    assert_eq!(
                        c.cst2[2].to_bits(),
                        0,
                        "the barrier word for {noise:?}, gains {g:?}"
                    );
                    frames += 1;
                }
            }
        }
        assert_eq!(frames, 18);
    }

    /// K8 (module notes, "Fast-math"): lavapipe neither fuses nor re-associates, so only the text can
    /// show that a barrier or an exact routine went missing. The barrier is an OR (an XOR applied
    /// twice cancels), no pass divides or takes a square root with the device's operations except
    /// inside the two routines' fallbacks (and the exact `/ 16.0`), and the routines are the output
    /// sharpening kernel's, which `tests/gpu_output_sharpen.rs` runs against IEEE on the device.
    #[test]
    fn the_shader_pins_every_rounding_and_divides_exactly() {
        let src = include_str!("../shaders/adaptive_ratio.wgsl");
        assert!(src.contains("return bitcast<f32>(bitcast<u32>(x) | bitcast<u32>(u.cst2.z));"));
        let sharpen = include_str!("../shaders/output_sharpen.wgsl");
        assert_eq!(
            wgsl_function(src, "ar_div").replacen("fn ar_div(", "", 1),
            wgsl_function(sharpen, "sharpen_div").replacen("fn sharpen_div(", "", 1),
            "ar_div is the output sharpening kernel's sharpen_div"
        );
        assert_eq!(
            wgsl_function(src, "ar_sqrt").replacen("fn ar_sqrt(", "", 1),
            wgsl_function(sharpen, "sharpen_sqrt").replacen("fn sharpen_sqrt(", "", 1),
            "ar_sqrt is the output sharpening kernel's sharpen_sqrt"
        );
        let passes = &src[src.find("fn ar_prepare(").expect("the passes")..];
        for (n, line) in passes.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains(" / ") || code.contains("/ 2u") || code.contains("acc / 16.0"),
                "line {n} of the passes divides with the device's `/`: {line}"
            );
            assert!(
                !code.contains("sqrt(") || code.contains("ar_sqrt("),
                "line {n} of the passes uses the device's `sqrt`: {line}"
            );
        }
        for needle in [
            "let w_u = ar_op(ar_op(ar_op(ar_op(ar_op(ar_op(rho * abs(",
            "return ar_op(ar_op(c + k) * ar_div(num, den)) - k;",
            "let s2 = ar_op(ar_op(ar_op(ar_op(abs(s - e) + abs(n - w))",
            "gsum = ar_op(gsum + ar_op(wj * gn[j]));",
            "let wt = ar_div(b2, ar_op(a2 + b2));",
            "outp[o] = ar_div(v.x, u.gains.w);",
        ] {
            assert!(src.contains(needle), "the shader lost `{needle}`");
        }
    }

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
