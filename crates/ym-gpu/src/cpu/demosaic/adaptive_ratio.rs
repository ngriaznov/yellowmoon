//! Adaptive ratio demosaic (`Method::AdaptiveRatio`), specified in
//! `docs/algorithms/01-adaptive-ratio-demosaic.md` (RA-03; the CA-aware path of passes 1 and 2 is
//! RA-03b).
//!
//! A three-pass Bayer demosaic:
//!
//! 1. **Green.** At every R and B site the local edge direction (horizontal, vertical or none) is
//!    classified with white-balance-aware colour-difference tests, then green is estimated as a
//!    noise-regularised colour ratio from the two or four adjacent greens.
//! 2. **Green refinement.** Where an oblique or knight-move direction is smoother than the axes,
//!    green is re-estimated along it.
//! 3. **Chroma.** The missing R and B are filled by cost-weighted offset ratios (first at R/B
//!    sites along the smoother diagonal, then at G sites from the four axis neighbours), followed
//!    by a diagonal luma sharpening that works in shadows only and, for low-SNR images, a
//!    noise-adaptive shadow colour matching.
//!
//! # Contract
//!
//! The input is the **unbalanced** mosaic (black-subtracted, white = 1.0, *not* multiplied by the
//! R4 gains) and the gains `g0` in [`DemosaicInputs::gains`]; the output is **white-balanced**
//! planar camera RGB (white comes out as `g0`), normalised, with no clamp at 0 or 1. The chain
//! hands R6 the *balanced* mosaic (R5 works on it), so R6 first divides each photosite by the gain
//! of its colour, the exact inverse of R4: [`demosaic_balanced`] does that. The noise model is the
//! unbalanced one on the 16-bit working scale ([`NoiseDn`]). Every absolute constant of the
//! algorithm lives at `S = 65535`; see the spec, section 1.4.
//!
//! Entry points: [`demosaic`] (parallel bands) and [`demosaic_whole`] (one piece, no banding) take
//! the unbalanced mosaic, as the oracle's `demosaic` does. [`demosaic_balanced`] and
//! [`demosaic_balanced_whole`] take the balanced mosaic the chain has, undo R4 and call those; they
//! are what [`demosaic_with`](super::demosaic_with) and
//! [`demosaic_whole_with`](super::demosaic_whole_with) run for `Method::AdaptiveRatio`
//! (`docs/algorithms/INTEGRATION.md` section 4), so the chain, the tile scheduler and R7's
//! `Y_classical` all get the division without doing it themselves. All of them return [`Error`] for
//! input the spec rejects (`BadSize`, `BadInput`, `BadGains`, `BadNoise`, `BadParams`, `BadCurve`); the
//! infallible `demosaic_with` falls back to Menon on the same balanced mosaic in that case.
//!
//! # How this port follows the spec
//!
//! * **Arithmetic.** Every per-pixel formula runs in `f64`, in the evaluation order of the spec
//!   (left to right, no `mul_add`); every plane between steps is stored as `f32` DN with one
//!   rounding at each write (`round32`). Only `+ - * /`, `sqrt`, `min`, `max` and `abs` are on the
//!   data path, so the result does not depend on the platform's libm. Decisions (pass 1 types,
//!   pass 2 argmin, pass 3 diagonal) are exact under these rules; the vectors match the C++
//!   oracle bit for bit apart from the `f32` rounding of the gains (see below).
//! * **Gains** arrive as `f32` ([`DemosaicInputs::gains`]) and are widened to `f64`. The vectors
//!   carry `f64` gains such as 1.9; their `f32` roundings differ by up to 6e-8 relative, far
//!   inside the 1e-5 tolerance.
//! * **Borders** are mirrored once at entry ([`mirror`], parity-preserving, repeated for pads
//!   larger than the buffer) into a padded plane whose red photosites sit at (even, even), so
//!   every formula assumes RGGB. Reads inside the plane are direct (no per-read mirroring).
//! * **Regions.** The spec evaluates every pass on the whole padded buffer and sums the per-pass
//!   halos (11 px, or 13 with shadow matching). This port evaluates each stage only on the
//!   rectangle the output core needs (core grown by 0 to 12 px, from the output backwards), which
//!   is the exact receptive field (10 px, or 12 with shadow matching; 1 px less than the sum
//!   because step 1 of pass 3 only reads *native* samples at +-2). The values inside the core are
//!   the same as the spec's, because every value the core depends on lies inside those rectangles.
//!   [`Method::halo`](super::Method::halo) still returns the spec's 13, so scheduler tiles
//!   carry at least the context that either configuration needs.
//! * **Banding.** [`demosaic`] cuts the frame into bands of [`BAND_ROWS`] rows with
//!   [`Derived::halo_px`] rows of real context (the shared band layout of `filters`), and runs the
//!   whole-piece routine on each band's *core* rows only (no wasted work on the context rows).
//!   The band layout depends on the image height alone, so the result is identical for every
//!   thread count and equal to [`demosaic_whole`] bit for bit.
//! * **CA-aware path (RA-03b).** With `DemosaicInputs::ca = Some(curves)` the two lateral CA
//!   curves ([`CaCurves`]) drive the pass-1 relaxation factor `f` and the pass-2 weighting `kappa`
//!   (spec 3.6). The misregistration `|shift_R(r) - shift_B(r)|` of every R and B site is computed
//!   once per band in `f64` from the site's own pixel centre, `sqrt` and the curves' linear
//!   interpolation, exactly as the oracle's `CaField::diff` does (padded positions outside the
//!   buffer use their geometric frame position, not the mirrored one), and read by both passes.
//!   `None` (no CA tags, `decode.chromatic_aberration` off, a binned mosaic) is the plain path,
//!   unchanged bit for bit. The CA path does not change the halo.
//!
//! Memory: about 9 padded `f32` planes (M, G1, G2, R, G, B and the luma proxy, freed as early as
//! possible) per band, plus about 9 transient planes with shadow matching.

use rayon::prelude::*;

use super::{CaCurves, DemosaicInputs};
use crate::cpu::filters::{self, BAND_ROWS};
use crate::cpu::noise::WORKING_SCALE;
use crate::cpu::{Cfa, CfaColor, Mosaic, NoiseDn, Rect, RgbImage, mirror};

/// The working scale `S` (16-bit white): every absolute constant is defined at `x_dn = S * x`.
const S: f64 = WORKING_SCALE;

/// Halo of the plain configuration in pixels: pass 1 (5) + pass 2 (2) + pass 3 (4).
pub const HALO_PLAIN_PX: usize = 11;

/// Halo with shadow colour matching in pixels (pass 3 grows to 6): the maximum, and what
/// [`Method::halo`](super::Method::halo) reports.
pub const HALO_MAX_PX: usize = 13;

/// Why the adaptive ratio demosaic refused its input (`BadSize`, `BadInput`, `BadGains`, `BadCurve`,
/// `BadNoise` and `BadParams` of the spec's section 4).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum Error {
    /// The mosaic is smaller than 2 x 2 photosites (mirroring needs two samples to keep the
    /// Bayer parity).
    #[error("the adaptive ratio demosaic needs at least 2 x 2 photosites, got {width} x {height}")]
    BadSize {
        /// Width in photosites.
        width: usize,
        /// Height in photosites.
        height: usize,
    },
    /// A photosite is NaN or infinite, or overflows the 16-bit working scale.
    #[error("the mosaic holds a value that is not finite on the 16-bit working scale")]
    BadInput,
    /// A gain is not positive and finite.
    #[error("white-balance gains {0:?} are not usable (need positive, finite values)")]
    BadGains([f32; 3]),
    /// The noise model has a negative or non-finite number.
    #[error("noise model {0:?} is not usable (need finite, non-negative values)")]
    BadNoise(NoiseDn),
    /// `detail` or `strength` is NaN or infinite.
    #[error("detail {detail} and strength {strength} must be finite numbers")]
    BadParams {
        /// The `detail` given.
        detail: f32,
        /// The `strength` given.
        strength: f32,
    },
    /// The optical centre of the CA curves is NaN or infinite. (The curves themselves are checked
    /// when a [`RadialCurve`](super::RadialCurve) is built; the fields of [`CaCurves`] are public,
    /// so its centre can still be set to anything.)
    #[error("the optical centre ({x}, {y}) of the CA curves is not finite")]
    BadCurve {
        /// Centre x in frame pixels.
        x: f64,
        /// Centre y in frame pixels.
        y: f64,
    },
}

/// Whether the shadow colour matching step (spec 3.4.5) runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShadowMatch {
    /// On when the SNR at 18 % grey is below 25 dB (spec 2.2).
    #[default]
    Auto,
    /// Always on (halo 13).
    On,
    /// Always off (halo 11).
    Off,
}

/// The tunable parameters of the algorithm besides gains and noise (spec 2.1). The defaults are
/// the reference behaviour; the chain passes them unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// 0 to 100. With `strength` it lowers the pass-3 SNR target `s` (default 0 gives s = 20).
    pub detail: f32,
    /// 0 to 100. Enters only through `detail + strength`.
    pub strength: f32,
    /// Shadow colour matching (default [`ShadowMatch::Auto`]).
    pub shadow_match: ShadowMatch,
    /// The diagonal shadow luma sharpening of pass 3 (default on; the switch is for A/B tests).
    pub shadow_sharpening: bool,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            detail: 0.0,
            strength: 0.0,
            shadow_match: ShadowMatch::Auto,
            shadow_sharpening: true,
        }
    }
}

/// The per-frame constants derived once from gains, noise model and parameters (spec 2.2). All
/// `_dn` quantities are on the 16-bit working scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Derived {
    /// The gains `g0` as `f64`.
    pub gains: [f64; 3],
    /// `g_R / g_G`: scales red samples to green in the pass-1 direction tests.
    pub rho_r: f64,
    /// `g_B / g_G`: scales blue samples to green in the pass-1 direction tests.
    pub rho_b: f64,
    /// Shot-noise gain `g` in DN per DN ([`NoiseDn::g`]).
    pub g_dn: f64,
    /// Signal-independent noise sigma `r` in DN ([`NoiseDn::r`]).
    pub r_dn: f64,
    /// Pass-1 offset `K`: the signal level with SNR 10, clamped to `[1, 65536]` DN.
    pub offset1_dn: f64,
    /// Pass-3 SNR target `s = max((100 - (detail + strength)) * 0.2, 1)`.
    pub snr_target: f64,
    /// Pass-3 shadow gate `T`: the signal level with SNR `s`, clamped to `[1, 65535]` DN.
    pub gate_dn: f64,
    /// Pass-3 ratio offset `k` (equal to `T`).
    pub offset3_dn: f64,
    /// SNR at 18 % of the working scale in dB (infinite for a zero noise model).
    pub snr18_db: f64,
    /// Whether shadow colour matching runs for this frame.
    pub shadow_match: bool,
    /// The halo of this configuration in pixels: 13 with shadow matching, else 11.
    pub halo_px: usize,
}

impl Derived {
    /// Derives the constants of one frame.
    ///
    /// # Errors
    ///
    /// [`Error::BadGains`] for a gain that is not positive and finite, [`Error::BadNoise`] for an
    /// invalid [`NoiseDn`] ([`NoiseDn::is_valid`]), [`Error::BadParams`] for a non-finite
    /// `detail` or `strength`.
    pub fn new(gains: [f32; 3], noise: NoiseDn, params: &Params) -> Result<Derived, Error> {
        Derived::with_f64_gains(gains.map(f64::from), noise, params)
    }

    /// [`Derived::new`] for gains held as `f64`. The demosaic itself takes `f32` gains
    /// ([`DemosaicInputs::gains`]); this is for callers that hold more precision (the oracle's
    /// vectors carry `f64` gains such as 1.9) and for checking the derived values against it.
    ///
    /// # Errors
    ///
    /// As [`Derived::new`]; the gains in [`Error::BadGains`] are rounded to `f32`.
    pub fn with_f64_gains(
        gains: [f64; 3],
        noise: NoiseDn,
        params: &Params,
    ) -> Result<Derived, Error> {
        if !gains.iter().all(|g| g.is_finite() && *g > 0.0) {
            return Err(Error::BadGains(gains.map(|g| g as f32)));
        }
        if !noise.is_valid() {
            return Err(Error::BadNoise(noise));
        }
        if !params.detail.is_finite() || !params.strength.is_finite() {
            return Err(Error::BadParams {
                detail: params.detail,
                strength: params.strength,
            });
        }
        let (g, r) = (noise.g, noise.r);
        // Signal level with SNR 10 for var = g x + r^2: x = 50 (g + sqrt(g^2 + 0.04 r^2)).
        let offset1_dn = (50.0 * (g + (g * g + 0.04 * r * r).sqrt())).clamp(1.0, 65536.0);
        let s = ((100.0 - (f64::from(params.detail) + f64::from(params.strength))) * 0.2).max(1.0);
        let s2 = s * s;
        // Signal level with SNR s: x = s^2 / 2 (g + sqrt(g^2 + 4 r^2 / s^2)).
        let gate_dn = (0.5 * s2 * (g + (g * g + 4.0 * r * r / s2).sqrt())).clamp(1.0, 65535.0);
        let x18 = 0.18 * S;
        let var18 = g * x18 + r * r;
        let snr18_db = if var18 > 0.0 {
            10.0 * (x18 * x18 / var18).log10()
        } else {
            f64::INFINITY
        };
        // SNR at 18 % grey below 25 dB  <=>  g * x18 + r^2 > x18^2 / 10^2.5 (no logarithm).
        let low_snr = var18 * 316.227_766_016_837_96 > x18 * x18;
        let shadow_match = match params.shadow_match {
            ShadowMatch::On => true,
            ShadowMatch::Off => false,
            ShadowMatch::Auto => low_snr,
        };
        Ok(Derived {
            gains,
            rho_r: gains[0] / gains[1],
            rho_b: gains[2] / gains[1],
            g_dn: g,
            r_dn: r,
            offset1_dn,
            snr_target: s,
            gate_dn,
            offset3_dn: gate_dn,
            snr18_db,
            shadow_match,
            halo_px: if shadow_match {
                HALO_MAX_PX
            } else {
                HALO_PLAIN_PX
            },
        })
    }
}

/// Checks the input once for a whole frame and derives its constants.
fn prepare(m: &Mosaic, inputs: &DemosaicInputs<'_>, params: &Params) -> Result<Derived, Error> {
    if m.width < 2 || m.height < 2 {
        return Err(Error::BadSize {
            width: m.width,
            height: m.height,
        });
    }
    let derived = Derived::new(inputs.gains, inputs.noise, params)?;
    if let Some(ca) = inputs.ca
        && !(ca.centre_x.is_finite() && ca.centre_y.is_finite())
    {
        return Err(Error::BadCurve {
            x: ca.centre_x,
            y: ca.centre_y,
        });
    }
    // `f64::max` ignores a NaN, so the input itself is checked as well as its scaled value.
    if !m
        .data
        .iter()
        .all(|&v| v.is_finite() && scale_in(v).is_finite())
    {
        return Err(Error::BadInput);
    }
    Ok(derived)
}

/// Demosaics `m` (the unbalanced mosaic) in parallel bands. Bit-identical to [`demosaic_whole`]
/// and independent of the thread count.
///
/// `inputs.ca` selects the CA-aware path (spec 3.6); the position of every photosite in the frame
/// comes from `m.place`.
///
/// # Errors
///
/// [`Error`] for a mosaic below 2 x 2, a non-finite photosite, unusable gains, an invalid noise
/// model, non-finite parameters or a CA centre that is not finite.
pub fn demosaic(
    m: &Mosaic,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<RgbImage, Error> {
    let d = prepare(m, inputs, params)?;
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    let bands = filters::bands(m.height, BAND_ROWS);
    let w = m.width;
    // Per-band mutable row slices of the three output planes.
    let mut slices = Vec::with_capacity(bands.len());
    let [mut pr, mut pg, mut pb] = out.planes_mut();
    for band in &bands {
        let len = band.len() * w;
        let (r1, r2) = pr.split_at_mut(len);
        let (g1, g2) = pg.split_at_mut(len);
        let (b1, b2) = pb.split_at_mut(len);
        slices.push((band.clone(), [r1, g1, b1]));
        (pr, pg, pb) = (r2, g2, b2);
    }
    slices.into_par_iter().for_each(|(band, dst)| {
        let ext = filters::extended_rows(&band, d.halo_px, m.height);
        let local = m.extract(Rect::new(0, ext.start, m.width, ext.len()));
        let core = Rect::new(0, band.start - ext.start, m.width, band.len());
        compute_core(&local, core, &d, params, inputs.ca, dst);
    });
    Ok(out)
}

/// Demosaics `m` as one piece, without banding or threads. The reference [`demosaic`] must match.
///
/// # Errors
///
/// As [`demosaic`].
pub fn demosaic_whole(
    m: &Mosaic,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<RgbImage, Error> {
    let d = prepare(m, inputs, params)?;
    let mut out = RgbImage::new(m.width, m.height);
    out.place = m.place;
    compute_core(
        m,
        Rect::whole(m.width, m.height),
        &d,
        params,
        inputs.ca,
        out.planes_mut(),
    );
    Ok(out)
}

/// The unbalanced mosaic behind a balanced one: every photosite divided by the gain of its colour,
/// `m_u = m_bal / g0[colour]` (spec 1.3), the exact inverse of R4. The division is `f64`, rounded
/// to `f32` once, as the oracle's `demosaic_balanced` does it.
///
/// `inputs` must already have been checked (positive finite gains): a zero gain would turn into
/// infinities here.
fn unbalance(balanced: &Mosaic, inputs: &DemosaicInputs<'_>) -> Mosaic {
    let gains = inputs.gains.map(f64::from);
    let mut unbalanced = balanced.clone();
    let w = balanced.width;
    for (y, row) in unbalanced.data.chunks_exact_mut(w).enumerate() {
        for (x, v) in row.iter_mut().enumerate() {
            let g = gains[balanced.cfa.color_at(x, y).plane()];
            *v = (f64::from(*v) / g) as f32;
        }
    }
    unbalanced
}

/// Checks what [`unbalance`] needs before it divides: the size, the gains and the noise model.
fn check_balanced(
    balanced: &Mosaic,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<(), Error> {
    if balanced.width < 2 || balanced.height < 2 {
        return Err(Error::BadSize {
            width: balanced.width,
            height: balanced.height,
        });
    }
    // The gains are checked by `demosaic` as well; checking first keeps a zero gain from
    // turning into infinities in the division.
    Derived::new(inputs.gains, inputs.noise, params)?;
    Ok(())
}

/// Demosaics the *balanced* mosaic R5 hands to R6: divides every photosite by the gain of its
/// colour (the exact inverse of R4, `m_u = m_bal / g0[colour]`) and calls [`demosaic`] with the
/// same gains. The output is balanced. This is what [`Method::AdaptiveRatio`](super::Method) does
/// in [`demosaic_with`](super::demosaic_with).
///
/// # Errors
///
/// As [`demosaic`]; bad gains are reported before the division.
pub fn demosaic_balanced(
    balanced: &Mosaic,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<RgbImage, Error> {
    check_balanced(balanced, inputs, params)?;
    demosaic(&unbalance(balanced, inputs), inputs, params)
}

/// [`demosaic_balanced`] as one piece, without banding or threads: the reference the banded one
/// must match bit for bit.
///
/// # Errors
///
/// As [`demosaic_balanced`].
pub fn demosaic_balanced_whole(
    balanced: &Mosaic,
    inputs: &DemosaicInputs<'_>,
    params: &Params,
) -> Result<RgbImage, Error> {
    check_balanced(balanced, inputs, params)?;
    demosaic_whole(&unbalance(balanced, inputs), inputs, params)
}

// ---------------------------------------------------------------------------------------------
// Geometry: the padded plane and the regions each stage is evaluated on
// ---------------------------------------------------------------------------------------------

/// `S * max(v, 0)` rounded to `f32`: the entry scale and clamp (spec 3.1).
#[inline]
fn scale_in(v: f32) -> f32 {
    (f64::from(v).max(0.0) * S) as f32
}

/// Colour of a photosite of the padded plane, where red sits at (even, even).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Site {
    R,
    G,
    B,
}

#[inline(always)]
fn site(x: usize, y: usize) -> Site {
    match (x & 1, y & 1) {
        (0, 0) => Site::R,
        (1, 1) => Site::B,
        _ => Site::G,
    }
}

/// A half-open rectangle of the padded plane.
#[derive(Clone, Copy, Debug)]
struct Region {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

/// Where the output core sits in the padded plane, and how big the plane is.
#[derive(Clone, Copy, Debug)]
struct Layout {
    pw: usize,
    ph: usize,
    /// The core in padded coordinates.
    core: Region,
}

impl Layout {
    /// The core grown by `e` pixels on every side.
    fn around(&self, e: usize) -> Region {
        Region {
            x0: self.core.x0 - e,
            x1: self.core.x1 + e,
            y0: self.core.y0 - e,
            y1: self.core.y1 + e,
        }
    }
}

/// How far around the output core each stage must be evaluated (the receptive field, derived from
/// the output backwards).
#[derive(Clone, Copy, Debug)]
struct Margins {
    /// The pass-3 result `S3` (2 px with shadow matching: the 5 x 5 Gaussian).
    s3: usize,
    /// Red, green and blue before the sharpening (the luma proxy reads +-1 around `S3`).
    rgb: usize,
    /// Step 1 of pass 3 (R/B sites; step 2 reads their results at +-1).
    step1: usize,
    /// Pass-2 green and the white-balanced green plane (read at +-2 by step 2).
    green: usize,
    /// Native samples scaled by the gains (read at +-2 by step 1).
    native: usize,
    /// Pass-1 green (read at +-2 by pass 2).
    green1: usize,
    /// The mosaic (read at +-5 by pass 1). 10 plain, 12 with shadow matching.
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
            native: rgb + 3,
            green1,
            mosaic: green1 + 5,
        }
    }
}

/// A plane of `n` samples. Debug builds fill it with NaN so that a read outside the rectangle a
/// stage wrote shows up in the output (the tests check every output for finiteness).
fn plane(n: usize) -> Vec<f32> {
    if cfg!(debug_assertions) {
        vec![f32::NAN; n]
    } else {
        vec![0.0; n]
    }
}

/// Column parity of the red photosites of a pattern.
fn red_column_parity(cfa: Cfa) -> usize {
    if cfa.color_at(0, 0) == CfaColor::Red || cfa.color_at(0, 1) == CfaColor::Red {
        0
    } else {
        1
    }
}

/// Builds the padded, scaled and clamped mosaic plane `M` around `core` (spec 3.1): `margin`
/// pixels on the right and bottom, and on the left and top as many as make red land on
/// (even, even) of the plane. Padded samples are `M(mirror(x), mirror(y))` of the input.
fn padded_plane(m: &Mosaic, core: Rect, margin: usize) -> (Vec<f32>, Layout) {
    let rx = red_column_parity(m.cfa);
    let ry = m.cfa.red_row_parity();
    let pad_x = margin + ((margin + core.x + rx) & 1);
    let pad_y = margin + ((margin + core.y + ry) & 1);
    let pw = pad_x + core.w + margin;
    let ph = pad_y + core.h + margin;
    let xs: Vec<usize> = (0..pw)
        .map(|px| mirror(core.x as isize - pad_x as isize + px as isize, m.width))
        .collect();
    let mut data = vec![0.0f32; pw * ph];
    for (py, row) in data.chunks_exact_mut(pw).enumerate() {
        let sy = mirror(core.y as isize - pad_y as isize + py as isize, m.height);
        let src = &m.data[sy * m.width..(sy + 1) * m.width];
        for (o, &sx) in row.iter_mut().zip(&xs) {
            *o = scale_in(src[sx]);
        }
    }
    let layout = Layout {
        pw,
        ph,
        core: Region {
            x0: pad_x,
            x1: pad_x + core.w,
            y0: pad_y,
            y1: pad_y + core.h,
        },
    };
    (data, layout)
}

/// The sample at `base + off` as `f64`.
#[inline(always)]
fn at(p: &[f32], base: isize, off: isize) -> f64 {
    f64::from(p[(base + off) as usize])
}

/// A read-only `N x N` view of a plane around one photosite (`N` odd): `p(dx, dy)` is the sample
/// at that offset from the centre as `f64` (x to the right, y down, `|dx|` and `|dy|` at most
/// `(N - 1) / 2`).
///
/// It holds one slice of `N` samples per row, so a lookup with constant offsets is a plain load
/// with no bounds check, and the index arithmetic is paid once per row per photosite instead of
/// once per lookup. A struct with `inline(always)` methods rather than a closure, so that
/// unoptimised builds inline it too.
#[derive(Clone, Copy)]
struct Window<'a, const N: usize> {
    rows: [&'a [f32]; N],
}

impl<'a, const N: usize> Window<'a, N> {
    /// Offset of the centre from the first row and column.
    const RADIUS: isize = ((N - 1) / 2) as isize;

    /// The window around flat index `centre` of a plane with `stride` samples per row. The window
    /// must lie inside the plane.
    #[inline(always)]
    fn new(data: &'a [f32], centre: usize, stride: usize) -> Window<'a, N> {
        let radius = N / 2;
        let top_left = centre - radius * stride - radius;
        Window {
            rows: std::array::from_fn(|k| &data[top_left + k * stride..top_left + k * stride + N]),
        }
    }

    #[inline(always)]
    fn p(&self, dx: isize, dy: isize) -> f64 {
        f64::from(self.rows[(dy + Self::RADIUS) as usize][(dx + Self::RADIUS) as usize])
    }
}

// ---------------------------------------------------------------------------------------------
// The core routine
// ---------------------------------------------------------------------------------------------

/// Demosaics the `core` rectangle of `m` (buffer coordinates) into `dst`, three planes of
/// `core.w * core.h` samples. `m` supplies the context around the core (real rows and columns, or
/// the buffer edge, where it mirrors).
fn compute_core(
    m: &Mosaic,
    core: Rect,
    d: &Derived,
    params: &Params,
    ca: Option<&CaCurves>,
    dst: [&mut [f32]; 3],
) {
    let mg = Margins::new(d.shadow_match);
    let (mp, lay) = padded_plane(m, core, mg.mosaic);
    let n = lay.pw * lay.ph;

    // The CA path (spec 3.6): the R/B misregistration at every R and B site both passes read.
    let ca_plane =
        ca.map(|curves| misregistration_plane(m, core, &lay, curves, lay.around(mg.green1)));
    let ca_plane = ca_plane.as_deref();

    // Pass 1: green at R and B sites.
    let mut g1 = plane(n);
    pass_green(&mp, &lay, lay.around(mg.green1), d, ca_plane, &mut g1);
    // Pass 2: refinement along oblique and knight-move directions.
    let mut g2 = plane(n);
    pass_refine(&mp, &g1, &lay, lay.around(mg.green), ca_plane, &mut g2);
    drop(g1);

    // Pass 3: chroma.
    let (mut pr, mut pg, mut pb) = (plane(n), plane(n), plane(n));
    apply_gains(&mp, &g2, &lay, &mg, d, [&mut pr, &mut pg, &mut pb]);
    drop(g2);
    step1_diagonals(&lay, lay.around(mg.step1), d, &mut pr, &pg, &mut pb);
    step2_green_sites(&lay, lay.around(mg.rgb), d, &mut pr, &pg, &mut pb);
    if params.shadow_sharpening {
        shadow_sharpen(&mp, &lay, &mg, d, [&mut pr, &mut pg, &mut pb]);
    }
    if d.shadow_match {
        let matched = shadow_colour_match(&mp, &lay, d, [&pr, &pg, &pb]);
        let w = core.w;
        let views = matched.each_ref().map(|p| View::new(p, w, 0, 0));
        write_core(m, core, &lay, d, views, dst);
    } else {
        let views = [&pr, &pg, &pb].map(|p| View::new(p, lay.pw, lay.core.x0, lay.core.y0));
        write_core(m, core, &lay, d, views, dst);
    }
}

/// The R/B misregistration `|shift_R(r) - shift_B(r)|` in pixels at every R and B site of `r`
/// (spec 3.6), as a plane indexed like the padded plane; G sites are not written.
///
/// The photosite at padded `(x, y)` sits at frame position `(x0 + x + 0.5, y0 + y + 0.5)`, where
/// `(x0, y0)` is the frame position of padded `(0, 0)`: the placement of `m` plus the core's offset
/// in `m` minus the padding. Padded positions outside the buffer (which hold mirrored data) use
/// their geometric position, like the oracle's `CaField`, so a tile and the whole frame agree at
/// the frame edge. `f64` throughout: the same `+`, `-`, `sqrt` and piecewise-linear evaluation as
/// the oracle, so the value is bit for bit its `diff`.
fn misregistration_plane(
    m: &Mosaic,
    core: Rect,
    lay: &Layout,
    curves: &CaCurves,
    r: Region,
) -> Vec<f64> {
    let pw = lay.pw;
    let x0 = m.place.x0 as i64 + core.x as i64 - lay.core.x0 as i64;
    let y0 = m.place.y0 as i64 + core.y as i64 - lay.core.y0 as i64;
    let mut out = if cfg!(debug_assertions) {
        vec![UNWRITTEN; pw * lay.ph]
    } else {
        vec![0.0f64; pw * lay.ph]
    };
    for y in r.y0..r.y1 {
        let dy = (y0 + y as i64) as f64 + 0.5 - curves.centre_y;
        for x in r.x0..r.x1 {
            if site(x, y) == Site::G {
                continue;
            }
            let dx = (x0 + x as i64) as f64 + 0.5 - curves.centre_x;
            out[y * pw + x] = curves.misregistration_px((dx * dx + dy * dy).sqrt());
        }
    }
    out
}

/// Pass 1: `G1 = M` at G sites; at R and B sites the regularised colour-ratio estimate along the
/// classified direction (spec 3.2). `ca` is the misregistration plane of the CA path (`None`: plain
/// path, read as 0 everywhere).
fn pass_green(m: &[f32], lay: &Layout, r: Region, d: &Derived, ca: Option<&[f64]>, g1: &mut [f32]) {
    let pw = lay.pw;
    for y in r.y0..r.y1 {
        let row = y * pw;
        for x in r.x0..r.x1 {
            let i = row + x;
            g1[i] = match site(x, y) {
                Site::G => m[i],
                Site::R => green_estimate(m, i, pw, d.rho_r, d.offset1_dn, ca_at(ca, i)),
                Site::B => green_estimate(m, i, pw, d.rho_b, d.offset1_dn, ca_at(ca, i)),
            };
        }
    }
}

/// What a debug build puts into the entries of the misregistration plane it has not computed, so
/// that a read outside the region a stage wrote is caught. A computed entry is `|a - b|`: never
/// negative, though it can be NaN when curves with finite knots overflow, so unlike NaN this marker
/// cannot be mistaken for a computed value.
const UNWRITTEN: f64 = -1.0;

/// The misregistration at flat index `i` of an R or B site: 0 on the plain path. A NaN (curves that
/// overflow to infinities) is no relaxation in pass 1 (`NaN >= 0.1` is false) and full weight in pass
/// 2 (`f64::min` returns the other argument), so the output stays finite.
#[inline(always)]
fn ca_at(ca: Option<&[f64]>, i: usize) -> f64 {
    ca.map_or(0.0, |p| {
        debug_assert!(
            p[i] != UNWRITTEN,
            "CA read outside the region it was computed on"
        );
        p[i]
    })
}

/// The pass-1 direction type: which greens the estimate uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Type 1: the left and right greens.
    Horizontal,
    /// Type 2: the upper and lower greens.
    Vertical,
    /// Type 3: all four.
    Isotropic,
}

/// The green estimate at the R or B site with flat index `i` (spec 3.2): classification, costs and
/// the regularised ratio. `rho` is `g_c / g_G` of the site's colour, `k` the pass-1 offset `K`,
/// `ca` the R/B misregistration at the site in pixels (0 on the plain path).
fn green_estimate(m: &[f32], i: usize, pw: usize, rho: f64, k: f64, ca: f64) -> f32 {
    let px = Window::<11>::new(m, i, pw);
    let c = px.p(0, 0);
    let cs = rho * c;
    let (gu, gd, gl, gr) = (px.p(0, -1), px.p(0, 1), px.p(-1, 0), px.p(1, 0));
    // Centre colour differences.
    let (du, dd, dl, dr) = (cs - gu, cs - gd, cs - gl, cs - gr);
    let w2 = 1.6 * rho;
    let vbase = (gl - px.p(-1, -2)).abs()
        + (gl - px.p(-1, 2)).abs()
        + (gr - px.p(1, -2)).abs()
        + (gr - px.p(1, 2)).abs()
        + w2 * (px.p(0, -2) - 2.0 * c + px.p(0, 2)).abs();
    let hbase = (gu - px.p(-2, -1)).abs()
        + (gu - px.p(2, -1)).abs()
        + (gd - px.p(-2, 1)).abs()
        + (gd - px.p(2, 1)).abs()
        + w2 * (px.p(-2, 0) - 2.0 * c + px.p(2, 0)).abs();
    let q_n = rho * px.p(0, -4);
    let q_s = rho * px.p(0, 4);
    let q_w = rho * px.p(-4, 0);
    let q_e = rho * px.p(4, 0);
    let mut su = vbase + (du + px.p(0, -5) - q_n).abs() + (du + px.p(0, 3) - q_s).abs();
    let mut sd = vbase + (dd + px.p(0, -3) - q_n).abs() + (dd + px.p(0, 5) - q_s).abs();
    let mut sv = vbase + (px.p(0, -5) - px.p(0, -3)).abs() + (px.p(0, 3) - px.p(0, 5)).abs();
    let mut sl = hbase + (dl + px.p(-5, 0) - q_w).abs() + (dl + px.p(3, 0) - q_e).abs();
    let mut sr = hbase + (dr + px.p(-3, 0) - q_w).abs() + (dr + px.p(5, 0) - q_e).abs();
    let mut sh = hbase + (px.p(-5, 0) - px.p(-3, 0)).abs() + (px.p(3, 0) - px.p(5, 0)).abs();
    for dy in [-2isize, 0, 2] {
        for dx in [-2isize, 0, 2] {
            let q = rho * px.p(dx, dy);
            let ql = px.p(dx - 1, dy);
            let qr = px.p(dx + 1, dy);
            let qu = px.p(dx, dy - 1);
            let qd = px.p(dx, dy + 1);
            sl += (dl + ql - q).abs();
            sr += (dr + qr - q).abs();
            su += (du + qu - q).abs();
            sd += (dd + qd - q).abs();
            sh += (ql - qr).abs();
            sv += (qu - qd).abs();
        }
    }
    // Decision. The CA relaxation factor f is exactly 1 on the plain path (ca = 0) and for a
    // misregistration under 0.1 px; it is unbounded above, as the spec has it (evaluation note 3).
    let f = if ca >= 0.1 {
        1.0 + 2.5 * (ca * ca * ca)
    } else {
        1.0
    };
    let dir = if sl.max(sr) < f * su.min(sd) && sh < sv {
        Direction::Horizontal
    } else if f * sl.min(sr) > su.max(sd) && sv < sh {
        Direction::Vertical
    } else {
        // Flat test: the two green-pair means differ by less than about two sigma of unit-gain
        // Poisson noise (DN), so no direction is trusted.
        let s_h = gl + gr;
        let s_v = gu + gd;
        let half_diff = 0.5 * (s_h - s_v).abs();
        if half_diff * half_diff < s_h + s_v {
            Direction::Isotropic
        } else {
            wide_direction(&px, rho, [dl, dr, du, dd])
        }
    };
    // Costs of the four directions (always >= 1).
    let w_u = rho * (px.p(0, -2) - c).abs()
        + (px.p(0, -3) - gu).abs()
        + (px.p(-1, -2) - gl).abs()
        + (px.p(1, -2) - gr).abs()
        + (gu - gd).abs()
        + 1.0;
    let w_d = rho * (px.p(0, 2) - c).abs()
        + (px.p(0, 3) - gd).abs()
        + (px.p(-1, 2) - gl).abs()
        + (px.p(1, 2) - gr).abs()
        + (gu - gd).abs()
        + 1.0;
    let w_l = rho * (px.p(-2, 0) - c).abs()
        + (px.p(-3, 0) - gl).abs()
        + (px.p(-2, -1) - gu).abs()
        + (px.p(-2, 1) - gd).abs()
        + (gl - gr).abs()
        + 1.0;
    let w_r = rho * (px.p(2, 0) - c).abs()
        + (px.p(3, 0) - gr).abs()
        + (px.p(2, -1) - gu).abs()
        + (px.p(2, 1) - gd).abs()
        + (gl - gr).abs()
        + 1.0;
    let c_u = 0.5 * (px.p(0, -2) + c);
    let c_d = 0.5 * (px.p(0, 2) + c);
    let c_l = 0.5 * (px.p(-2, 0) + c);
    let c_r = 0.5 * (px.p(2, 0) + c);
    // The weight of each green is the product of the costs of the other directions in use.
    let (a_u, a_d, a_l, a_r) = match dir {
        Direction::Horizontal => (0.0, 0.0, w_r, w_l),
        Direction::Vertical => (w_d, w_u, 0.0, 0.0),
        Direction::Isotropic => (
            w_d * w_l * w_r,
            w_u * w_l * w_r,
            w_u * w_d * w_r,
            w_u * w_d * w_l,
        ),
    };
    let sum = a_u + a_d + a_l + a_r;
    let ng = a_u * gu + a_d * gd + a_l * gl + a_r * gr;
    let nc = a_u * c_u + a_d * c_d + a_l * c_l + a_r * c_r;
    let ks = k * sum;
    let num = (ks + ng).max(ks);
    let den = (ks + nc).max(ks);
    ((c + k) * (num / den) - k) as f32
}

/// The 5 x 5 fallback of the pass-1 decision (spec 3.2), reached when the first test and the flat
/// test did not decide. `d` holds the centre colour differences `[dl, dr, du, dd]`.
fn wide_direction(px: &Window<'_, 11>, rho: f64, d: [f64; 4]) -> Direction {
    let [dl, dr, du, dd] = d;
    let (mut l5, mut r5, mut u5, mut d5) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for dy in [-4isize, -2, 0, 2, 4] {
        for dx in [-4isize, -2, 0, 2, 4] {
            let q = rho * px.p(dx, dy);
            l5 += (dl + px.p(dx - 1, dy) - q).abs();
            r5 += (dr + px.p(dx + 1, dy) - q).abs();
            u5 += (du + px.p(dx, dy - 1) - q).abs();
            d5 += (dd + px.p(dx, dy + 1) - q).abs();
        }
    }
    if l5.max(r5) < u5.min(d5) {
        Direction::Horizontal
    } else if l5.min(r5) > u5.max(d5) {
        Direction::Vertical
    } else {
        Direction::Isotropic
    }
}

/// Pass 2 (spec 3.3): at G sites `G2 = G1`; at R and B sites the vertical and horizontal scores are
/// compared with two diagonals and four knight-move directions. `ca` is the misregistration plane
/// of the CA path (spec 3.6), whose pass-2 weight is `min(ca, 1)`; `None` is the plain path.
fn pass_refine(m: &[f32], g1: &[f32], lay: &Layout, r: Region, ca: Option<&[f64]>, g2: &mut [f32]) {
    let pw = lay.pw;
    for y in r.y0..r.y1 {
        let row = y * pw;
        for x in r.x0..r.x1 {
            let i = row + x;
            g2[i] = if site(x, y) == Site::G {
                g1[i]
            } else {
                refine_green(m, g1, i, pw, ca.map(|_| ca_at(ca, i).min(1.0)))
            };
        }
    }
}

/// The pass-2 value at the R or B site with flat index `i` (spec 3.3, and 3.6 for `kappa`). `m`
/// holds the native samples, `g` the pass-1 green, `kappa` the CA weight `min(ca, 1)` of the site
/// (`None`: plain path).
fn refine_green(m: &[f32], g: &[f32], i: usize, pw: usize, kappa: Option<f64>) -> f32 {
    let gw = Window::<5>::new(g, i, pw);
    let cw = Window::<5>::new(m, i, pw);
    let g0 = gw.p(0, 0);
    let (n, so, w, e) = (gw.p(0, -1), gw.p(0, 1), gw.p(-1, 0), gw.p(1, 0));
    let (nw, ne, sw, se) = (gw.p(-1, -1), gw.p(1, -1), gw.p(-1, 1), gw.p(1, 1));
    let c0 = cw.p(0, 0);
    let mut v = (n - g0).abs() + (so - g0).abs();
    let mut h = (w - g0).abs() + (e - g0).abs();
    if let Some(k) = kappa {
        // The axis scores also see the same-colour gradients two photosites out, weighted by kappa.
        v += k * ((cw.p(0, -2) - c0).abs() + (cw.p(0, 2) - c0).abs());
        h += k * ((cw.p(-2, 0) - c0).abs() + (cw.p(2, 0) - c0).abs());
    }
    // Scores of the six oblique directions (indices 2 to 7).
    let score = [
        (so - e).abs()
            + (n - w).abs()
            + (cw.p(2, -2) - c0).abs()
            + (cw.p(-2, 2) - c0).abs()
            + (cw.p(-1, 1) - cw.p(1, -1)).abs(),
        (so - w).abs()
            + (n - e).abs()
            + (cw.p(-2, -2) - c0).abs()
            + (cw.p(2, 2) - c0).abs()
            + (cw.p(-1, -1) - cw.p(1, 1)).abs(),
        (so - ne).abs() + (n - sw).abs() + (gw.p(-1, 2) - g0).abs() + (gw.p(1, -2) - g0).abs(),
        (e - sw).abs() + (w - ne).abs() + (gw.p(-2, 1) - g0).abs() + (gw.p(2, -1) - g0).abs(),
        (n - se).abs() + (so - nw).abs() + (gw.p(-1, -2) - g0).abs() + (gw.p(1, 2) - g0).abs(),
        (w - se).abs() + (e - nw).abs() + (gw.p(-2, -1) - g0).abs() + (gw.p(2, 1) - g0).abs(),
    ];
    let mut best = usize::from(h < v);
    let mut lowest = if h < v { h } else { v };
    for (j, &sc) in score.iter().enumerate() {
        if sc < lowest {
            lowest = sc;
            best = j + 2;
        }
    }
    if best < 2 && kappa.is_none() {
        return g[i];
    }
    let mean = |dir: usize| -> f64 {
        match dir {
            0 => (g0 + n + so) / 3.0,
            1 => (g0 + w + e) / 3.0,
            2 => (2.0 * g0 + ne + sw) / 4.0,
            3 => (2.0 * g0 + nw + se) / 4.0,
            4 => (g0 + gw.p(-1, 2) + gw.p(1, -2)) / 3.0,
            5 => (g0 + gw.p(-2, 1) + gw.p(2, -1)) / 3.0,
            6 => (g0 + gw.p(-1, -2) + gw.p(1, 2)) / 3.0,
            _ => (g0 + gw.p(-2, -1) + gw.p(2, 1)) / 3.0,
        }
    };
    // Note: V = H selects the horizontal mean.
    let axis = if v < h { mean(0) } else { mean(1) };
    let Some(k) = kappa else {
        return (0.5 * (axis + mean(best))) as f32;
    };
    // CA path (spec 3.6).
    if best < 2 {
        // A 3-tap mean along the winning axis, which takes over from G0 as kappa grows (plain
        // 3-tap from kappa = 0.5); the weight of G0 is 1 + 18 t + 102 t^3, t = (1 - min(1, 2 kappa))^2.
        let t = 1.0 - (2.0 * k).min(1.0);
        let t = t * t;
        let wt = 1.0 + 18.0 * t + 102.0 * (t * t * t);
        let num = if best == 0 {
            wt * g0 + n + so
        } else {
            wt * g0 + w + e
        };
        return (num / (wt + 2.0)) as f32;
    }
    ((axis + (2.0 * k + 1.0) * mean(best)) / (2.0 * k + 2.0)) as f32
}

/// Pass 3, gains (spec 3.4.1): `G = round32(g_G * G2)` everywhere, `R = round32(g_R * M)` at R
/// sites and `B = round32(g_B * M)` at B sites. From here on the planes are white-balanced.
fn apply_gains(
    m: &[f32],
    g2: &[f32],
    lay: &Layout,
    mg: &Margins,
    d: &Derived,
    planes: [&mut [f32]; 3],
) {
    let pw = lay.pw;
    let [pr, pg, pb] = planes;
    let nat = lay.around(mg.native);
    for y in nat.y0..nat.y1 {
        for x in nat.x0..nat.x1 {
            let i = y * pw + x;
            match site(x, y) {
                Site::R => pr[i] = (f64::from(m[i]) * d.gains[0]) as f32,
                Site::B => pb[i] = (f64::from(m[i]) * d.gains[2]) as f32,
                Site::G => {}
            }
        }
    }
    let gr = lay.around(mg.green);
    for y in gr.y0..gr.y1 {
        for x in gr.x0..gr.x1 {
            let i = y * pw + x;
            pg[i] = (f64::from(g2[i]) * d.gains[1]) as f32;
        }
    }
}

/// Step 1 (spec 3.4.2): the missing colour at every R and B site of `r` from the two diagonal
/// samples of that colour along the smoother diagonal. The step reads only native samples of the
/// colour it writes, so the order of the sites does not matter.
fn step1_diagonals(
    lay: &Layout,
    r: Region,
    d: &Derived,
    pr: &mut [f32],
    pg: &[f32],
    pb: &mut [f32],
) {
    let pw = lay.pw;
    let s = pw as isize;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let i = y * pw + x;
            match site(x, y) {
                Site::R => {
                    pb[i] = diagonal_ratio(pb, pg, pr, i, s, d.offset3_dn);
                }
                Site::B => {
                    pr[i] = diagonal_ratio(pr, pg, pb, i, s, d.offset3_dn);
                }
                Site::G => {}
            }
        }
    }
}

/// The missing colour X at an R or B site (spec 3.4.2). `xp` is the plane of the missing colour
/// (native at the four diagonal neighbours), `gp` the green plane, `yp` the plane of the site's
/// own colour, `k` the pass-3 offset.
fn diagonal_ratio(xp: &[f32], gp: &[f32], yp: &[f32], i: usize, s: isize, k: f64) -> f32 {
    let base = i as isize;
    let (ul, ur, dl, dr) = (-s - 1, -s + 1, s - 1, s + 1);
    let (x_ul, x_ur, x_dl, x_dr) = (
        at(xp, base, ul),
        at(xp, base, ur),
        at(xp, base, dl),
        at(xp, base, dr),
    );
    let (g_ul, g_ur, g_dl, g_dr) = (
        at(gp, base, ul),
        at(gp, base, ur),
        at(gp, base, dl),
        at(gp, base, dr),
    );
    let g0 = at(gp, base, 0);
    let c0 = at(yp, base, 0);
    // "\" is smoother: A = up-left, B = down-right; else A = down-left, B = up-right.
    let main_diag = (x_ul - x_dr).abs() < (x_dl - x_ur).abs();
    let (x_a, g_a, c_a) = if main_diag {
        (x_ul, g_ul, at(yp, base, -2 * s - 2))
    } else {
        (x_dl, g_dl, at(yp, base, 2 * s - 2))
    };
    let (x_b, g_b, c_b) = if main_diag {
        (x_dr, g_dr, at(yp, base, 2 * s + 2))
    } else {
        (x_ur, g_ur, at(yp, base, -2 * s + 2))
    };
    let mut w_a = 1.0 + (g_a - g0).abs() + (c_a - c0).abs();
    let mut w_b = 1.0 + (g_b - g0).abs() + (c_b - c0).abs();
    // X not dominant over G: penalise high-chroma samples.
    if 0.75 * (x_ul + x_ur + x_dl + x_dr) < g_ul + g_ur + g_dl + g_dr {
        w_a *= (x_a - g_a).abs() / (x_a + g_a + 1.0).max(1.0) + 0.03;
        w_b *= (x_b - g_b).abs() / (x_b + g_b + 1.0).max(1.0) + 0.03;
    }
    // Each sample is weighted by the other sample's cost.
    let t = (w_a + w_b) * k;
    let den = (g_b * w_a + g_a * w_b + t).max(t);
    let num = x_b * w_a + x_a * w_b + t;
    ((g0 + k) * (num / den) - k) as f32
}

/// Step 2 (spec 3.4.3): R and B at every G site of `r` from the four axis neighbours (R/B sites
/// that carry both colours after step 1). Both colours are computed before either is written;
/// the step reads R and B only at R/B sites, so the order of the sites does not matter.
fn step2_green_sites(
    lay: &Layout,
    r: Region,
    d: &Derived,
    pr: &mut [f32],
    pg: &[f32],
    pb: &mut [f32],
) {
    let pw = lay.pw;
    let k = d.offset3_dn;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            if site(x, y) != Site::G {
                continue;
            }
            let i = y * pw + x;
            let g0 = f64::from(pg[i]);
            // Neighbour order: up, down, left, right.
            let nb = [i - pw, i + pw, i - 1, i + 1];
            let gn = nb.map(|j| f64::from(pg[j]));
            let d_v = (gn[0] - gn[1]).abs();
            let d_h = (gn[2] - gn[3]).abs();
            let cost0 = [
                1.0 + d_v + (f64::from(pg[i - 2 * pw]) - g0).abs(),
                1.0 + d_v + (f64::from(pg[i + 2 * pw]) - g0).abs(),
                1.0 + d_h + (f64::from(pg[i - 2]) - g0).abs(),
                1.0 + d_h + (f64::from(pg[i + 2]) - g0).abs(),
            ];
            let sum_g = gn[0] + gn[1] + gn[2] + gn[3];
            let mut result = [0.0f64; 2];
            for (res, chan) in result.iter_mut().zip([&*pr, &*pb]) {
                let xn = nb.map(|j| f64::from(chan[j]));
                let mut cost = cost0;
                if 0.75 * (xn[0] + xn[1] + xn[2] + xn[3]) < sum_g {
                    for j in 0..4 {
                        cost[j] *= (gn[j] - xn[j]).abs() / (gn[j] + xn[j] + 1.0).max(1.0) + 0.03;
                    }
                }
                let (mut wsum, mut gsum, mut xsum) = (0.0f64, 0.0f64, 0.0f64);
                for j in 0..4 {
                    let wj = 1.0 / cost[j];
                    wsum += wj;
                    gsum += wj * gn[j];
                    xsum += wj * xn[j];
                }
                let t = k * wsum;
                let den = (gsum + t).max(t);
                *res = (g0 + k) * ((xsum + t) / den) - k;
            }
            pr[i] = result[0] as f32;
            pb[i] = result[1] as f32;
        }
    }
}

/// Step 3 (spec 3.4.4): a soft-limited sharpening of the luma proxy along the smoother diagonal,
/// where every native green of the 3 x 3 neighbourhood is below `T`, fading out at `4T`. All three
/// channels get the same factor. Writes the result over `lay.around(mg.s3)`; the luma proxy is
/// computed over the larger rectangle it reads first.
fn shadow_sharpen(m: &[f32], lay: &Layout, mg: &Margins, d: &Derived, planes: [&mut [f32]; 3]) {
    let pw = lay.pw;
    let s = pw;
    let [pr, pg, pb] = planes;
    // Unbalanced luma proxy (x4 scale), floored at 1, stored as f32.
    let lr = lay.around(mg.rgb);
    let mut luma = plane(pw * lay.ph);
    for y in lr.y0..lr.y1 {
        for x in lr.x0..lr.x1 {
            let i = y * pw + x;
            let yl = f64::from(pr[i]) / d.gains[0]
                + 2.0 * f64::from(pg[i]) / d.gains[1]
                + f64::from(pb[i]) / d.gains[2]
                + 1.0;
            luma[i] = yl.max(1.0) as f32;
        }
    }
    let t = d.gate_dn;
    let out = lay.around(mg.s3);
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let i = y * pw + x;
            // Largest native green of the 3 x 3 window (>= 0: M is clamped).
            let gmax = match site(x, y) {
                Site::G => [m[i], m[i - s - 1], m[i - s + 1], m[i + s - 1], m[i + s + 1]],
                _ => [m[i - s], m[i + s], m[i - 1], m[i + 1], 0.0],
            }
            .into_iter()
            .fold(0.0f64, |top, v| top.max(f64::from(v)));
            let f = ((4.0 * t - gmax) / (3.0 * t)).clamp(0.0, 1.0);
            if f <= 0.0 {
                continue;
            }
            let y0 = f64::from(luma[i]);
            let pm = f64::from(luma[i - s - 1]) + f64::from(luma[i + s + 1]);
            let pa = f64::from(luma[i - s + 1]) + f64::from(luma[i + s - 1]);
            let pair = if (2.0 * y0 - pm).abs() < (2.0 * y0 - pa).abs() {
                pm
            } else {
                pa
            };
            let lap = 5.0 * (2.0 * y0 - pair);
            let lim = y0.min(d.r_dn);
            let den = lim + lap.abs();
            let add = if den > 0.0 { lim * lap / den } else { 0.0 };
            let gain = (y0 + 0.75 * f * add) / y0;
            pr[i] = (f64::from(pr[i]) * gain) as f32;
            pg[i] = (f64::from(pg[i]) * gain) as f32;
            pb[i] = (f64::from(pb[i]) * gain) as f32;
        }
    }
}

/// 3 x 3 Gaussian `[1 2 1]^T [1 2 1] / 16` times `scale`, over `r`, from a plane valid one pixel
/// beyond `r` (spec 3.4.5).
///
/// The spec sums three row sums `src(x-1) + 2 src(x) + src(x+1)` in `f64`, top to bottom. Each row
/// sum is the same expression for the three outputs that use it, so it is computed once per row
/// and reused: the values are bit for bit the spec's.
fn gauss3(src: &[f32], pw: usize, r: Region, scale: f64, dst: &mut [f32]) {
    let w = r.x1 - r.x0;
    let row_sums = |y: usize, out: &mut [f64]| {
        let row = &src[y * pw + r.x0 - 1..y * pw + r.x1 + 1];
        for (x, o) in out.iter_mut().enumerate() {
            *o = f64::from(row[x]) + 2.0 * f64::from(row[x + 1]) + f64::from(row[x + 2]);
        }
    };
    let (mut above, mut here, mut below) = (vec![0.0f64; w], vec![0.0f64; w], vec![0.0f64; w]);
    row_sums(r.y0 - 1, &mut above);
    row_sums(r.y0, &mut here);
    for y in r.y0..r.y1 {
        row_sums(y + 1, &mut below);
        let out = &mut dst[y * pw + r.x0..y * pw + r.x1];
        for (x, o) in out.iter_mut().enumerate() {
            // acc = 1 * row(-1) + 2 * row(0) + 1 * row(1), left to right.
            let mut acc = 0.0f64;
            acc += above[x];
            acc += 2.0 * here[x];
            acc += below[x];
            *o = (acc / 16.0 * scale) as f32;
        }
        std::mem::swap(&mut above, &mut here);
        std::mem::swap(&mut here, &mut below);
    }
}

/// The 5 x 5 Gaussian with integer weights over 571 (not separable) of the three planes `src` at
/// flat index `i` (each valid two pixels around it), summed row by row from the top left like the
/// spec. The three sums are independent chains evaluated side by side, which hides the latency of
/// the sequential additions without changing any of them.
#[inline]
fn gauss5_at(src: [&[f32]; 3], i: usize, pw: usize) -> [f32; 3] {
    const W: [[f64; 5]; 5] = [
        [2.0, 7.0, 12.0, 7.0, 2.0],
        [7.0, 31.0, 52.0, 31.0, 7.0],
        [12.0, 52.0, 127.0, 52.0, 12.0],
        [7.0, 31.0, 52.0, 31.0, 7.0],
        [2.0, 7.0, 12.0, 7.0, 2.0],
    ];
    let top_left = i - 2 * pw - 2;
    let mut acc = [0.0f64; 3];
    for (dy, weights) in W.iter().enumerate() {
        let at = top_left + dy * pw;
        let rows = src.map(|p| &p[at..at + 5]);
        for (dx, w) in weights.iter().enumerate() {
            for c in 0..3 {
                acc[c] += w * f64::from(rows[c][dx]);
            }
        }
    }
    acc.map(|a| (a / 571.0) as f32)
}

/// 3 x 3 box means of the three planes `src` at flat index `i` (each valid one pixel around it),
/// each summed row by row from the top left like the spec.
#[inline]
fn box3_at(src: [&[f32]; 3], i: usize, pw: usize) -> [f64; 3] {
    let top_left = i - pw - 1;
    let mut acc = [0.0f64; 3];
    for dy in 0..3 {
        let at = top_left + dy * pw;
        for (a, p) in acc.iter_mut().zip(src) {
            for &v in &p[at..at + 3] {
                *a += f64::from(v);
            }
        }
    }
    acc.map(|a| a / 9.0)
}

/// Step 4 (spec 3.4.5): replaces the low-pass part of the pass-3 result `s3` with the low-pass of
/// a bilinear demosaic of the native samples, to the degree the difference between the two can be
/// explained by noise. Returns the three matched planes, each `core.w * core.h` samples.
fn shadow_colour_match(m: &[f32], lay: &Layout, d: &Derived, s3: [&[f32]; 3]) -> [Vec<f32>; 3] {
    let pw = lay.pw;
    let n = pw * lay.ph;
    let (e1, e2, e3) = (lay.around(1), lay.around(2), lay.around(3));
    let mult = [4.0 * d.gains[0], 2.0 * d.gains[1], 4.0 * d.gains[2]];
    // low[c] = Gauss3(Gauss3(sparse_c) * m_c); err[c] = (S3_c - low[c])^2, both over core +- 1.
    let mut low: [Vec<f32>; 3] = std::array::from_fn(|_| plane(n));
    let mut err: [Vec<f32>; 3] = std::array::from_fn(|_| plane(n));
    let mut sparse = plane(n);
    let mut simple = plane(n);
    for c in 0..3 {
        // Native samples of colour c, zero elsewhere.
        for y in e3.y0..e3.y1 {
            for x in e3.x0..e3.x1 {
                let i = y * pw + x;
                sparse[i] = if site(x, y) == [Site::R, Site::G, Site::B][c] {
                    m[i]
                } else {
                    0.0
                };
            }
        }
        gauss3(&sparse, pw, e2, mult[c], &mut simple);
        gauss3(&simple, pw, e1, 1.0, &mut low[c]);
        for y in e1.y0..e1.y1 {
            for x in e1.x0..e1.x1 {
                let i = y * pw + x;
                let e = f64::from(s3[c][i]) - f64::from(low[c][i]);
                err[c][i] = (e * e) as f32;
            }
        }
    }
    drop((sparse, simple));
    let core = lay.core;
    let (cw, ch) = (core.x1 - core.x0, core.y1 - core.y0);
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; cw * ch]);
    let err_planes = err.each_ref().map(|p| p.as_slice());
    let low_planes = low.each_ref().map(|p| p.as_slice());
    for y in core.y0..core.y1 {
        for x in core.x0..core.x1 {
            let i = y * pw + x;
            let boxes = box3_at(err_planes, i, pw);
            let mut e = 0.0f64;
            let mut var = 0.0f64;
            for c in 0..3 {
                e += boxes[c];
                let rc = d.r_dn * d.gains[c];
                var += d.g_dn * d.gains[c] * f64::from(low_planes[c][i]).max(0.0) + rc * rc;
            }
            let excess = (e - 0.5 * var).max(0.0);
            let a = 3.0 * excess + 1.0;
            let b = var + 1.0;
            let top = a.max(b);
            let a2 = (a - 0.5 * top).max(0.0);
            let b2 = (b - 0.5 * top).max(0.0);
            let wt = b2 / (a2 + b2);
            let low_s3 = gauss5_at(s3, i, pw);
            let o = (y - core.y0) * cw + (x - core.x0);
            for c in 0..3 {
                out[c][o] = (f64::from(s3[c][i])
                    + wt * (f64::from(low_planes[c][i]) - f64::from(low_s3[c])))
                    as f32;
            }
        }
    }
    out
}

/// A read-only window onto the core of a plane.
#[derive(Clone, Copy)]
struct View<'a> {
    data: &'a [f32],
    stride: usize,
    x0: usize,
    y0: usize,
}

impl<'a> View<'a> {
    fn new(data: &'a [f32], stride: usize, x0: usize, y0: usize) -> View<'a> {
        View {
            data,
            stride,
            x0,
            y0,
        }
    }

    /// The sample at core coordinates `(x, y)`.
    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> f64 {
        f64::from(self.data[(self.y0 + y) * self.stride + self.x0 + x])
    }
}

/// Exit (spec 3.5): `out_c = round32((P3_c + [c = colour(x, y)] * g_c * S * min(m_u, 0)) / S)`.
/// The part of a negative measurement that the entry clamp removed goes back into the photosite's
/// own channel, white-balanced. `views` hold the three pass-3 results over the core.
fn write_core(
    m: &Mosaic,
    core: Rect,
    lay: &Layout,
    d: &Derived,
    views: [View<'_>; 3],
    dst: [&mut [f32]; 3],
) {
    for (c, (view, out)) in views.iter().zip(dst).enumerate() {
        for y in 0..core.h {
            let src_row = &m.data[(core.y + y) * m.width + core.x..][..core.w];
            let out_row = &mut out[y * core.w..(y + 1) * core.w];
            for (x, (o, &input)) in out_row.iter_mut().zip(src_row).enumerate() {
                let native = match site(lay.core.x0 + x, lay.core.y0 + y) {
                    Site::R => 0,
                    Site::G => 1,
                    Site::B => 2,
                };
                let mut v = view.get(x, y);
                if native == c {
                    let removed = if input < 0.0 { f64::from(input) } else { 0.0 } * S;
                    v += d.gains[c] * removed;
                }
                *o = (v / S) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cfa;

    fn nz(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseDn {
        let var = b + sigma_row * sigma_row + q * q / 12.0;
        NoiseDn::new(S * a, S * var.sqrt())
    }

    fn low_noise() -> NoiseDn {
        nz(2e-5, 1e-8, 0.0, 1.0 / 65535.0)
    }

    fn high_noise() -> NoiseDn {
        nz(5e-4, 4e-5, 1e-3, 0.0)
    }

    #[test]
    fn derived_values_follow_the_spec_examples() {
        // Spec 2.2 table: low model g 1.311, r 6.56, K 158.3, T 555.3, 39.5 dB, matching off.
        let d = Derived::new([2.0, 1.0, 1.5], low_noise(), &Params::default()).unwrap();
        assert!((d.g_dn - 1.311).abs() < 1e-3 && (d.r_dn - 6.56).abs() < 1e-2);
        assert!((d.offset1_dn - 158.3).abs() < 0.05, "{}", d.offset1_dn);
        assert!((d.gate_dn - 555.3).abs() < 0.05, "{}", d.gate_dn);
        assert_eq!(d.snr_target, 20.0);
        assert_eq!(d.offset3_dn, d.gate_dn);
        assert!((d.snr18_db - 39.5).abs() < 0.05);
        assert!(!d.shadow_match);
        assert_eq!(d.halo_px, HALO_PLAIN_PX);
        assert_eq!((d.rho_r, d.rho_b), (2.0, 1.5));
        // High model: g 32.77, r 419.6, K 6143, T 17202, 23.9 dB, matching on.
        let d = Derived::new([2.0, 1.0, 1.5], high_noise(), &Params::default()).unwrap();
        assert!((d.offset1_dn - 6143.0).abs() < 1.0, "{}", d.offset1_dn);
        assert!((d.gate_dn - 17202.0).abs() < 2.0, "{}", d.gate_dn);
        assert!((d.snr18_db - 23.9).abs() < 0.05);
        assert!(d.shadow_match);
        assert_eq!(d.halo_px, HALO_MAX_PX);
    }

    #[test]
    fn the_snr_targets_and_overrides_behave() {
        let low = low_noise();
        let p = Params {
            detail: 40.0,
            strength: 20.0,
            ..Params::default()
        };
        assert_eq!(Derived::new([1.0; 3], low, &p).unwrap().snr_target, 8.0);
        // A sum of 95 or more floors s at 1.
        let p = Params {
            detail: 100.0,
            strength: 100.0,
            ..Params::default()
        };
        assert_eq!(Derived::new([1.0; 3], low, &p).unwrap().snr_target, 1.0);
        // Overrides of the automatic switch.
        let off = Params {
            shadow_match: ShadowMatch::Off,
            ..Params::default()
        };
        let on = Params {
            shadow_match: ShadowMatch::On,
            ..Params::default()
        };
        assert!(
            !Derived::new([1.0; 3], high_noise(), &off)
                .unwrap()
                .shadow_match
        );
        assert!(Derived::new([1.0; 3], low, &on).unwrap().shadow_match);
        // The zero model floors K and T at 1 DN, and its SNR is infinite.
        let z = Derived::new([1.0; 3], NoiseDn::ZERO, &Params::default()).unwrap();
        assert_eq!((z.offset1_dn, z.gate_dn), (1.0, 1.0));
        assert!(z.snr18_db.is_infinite() && !z.shadow_match);
    }

    #[test]
    fn the_automatic_switch_flips_at_25_db() {
        // 0.18 a + b_total = 0.0324 / 10^2.5 = 1.0246e-4 (spec 2.2); with a = 0 only b counts.
        let edge = 0.0324 / 316.227_766_016_837_96;
        for (scale, want) in [(0.999, false), (1.001, true)] {
            let d = Derived::new(
                [1.0; 3],
                nz(0.0, edge * scale, 0.0, 0.0),
                &Params::default(),
            )
            .unwrap();
            assert_eq!(d.shadow_match, want, "{scale}");
        }
    }

    #[test]
    fn bad_input_is_refused() {
        let ok = Params::default();
        for gains in [
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, f32::NAN],
            [f32::INFINITY, 1.0, 1.0],
        ] {
            assert!(matches!(
                Derived::new(gains, NoiseDn::ZERO, &ok),
                Err(Error::BadGains(_))
            ));
        }
        for noise in [
            NoiseDn::new(-1.0, 0.0),
            NoiseDn::new(0.0, f64::NAN),
            NoiseDn::new(f64::INFINITY, 0.0),
        ] {
            assert!(matches!(
                Derived::new([1.0; 3], noise, &ok),
                Err(Error::BadNoise(_))
            ));
        }
        let bad = Params {
            detail: f32::NAN,
            ..ok
        };
        assert!(matches!(
            Derived::new([1.0; 3], NoiseDn::ZERO, &bad),
            Err(Error::BadParams { .. })
        ));
        let inputs = DemosaicInputs::default();
        // Size and values.
        let tiny = Mosaic::new(1, 4, Cfa::RGGB);
        assert_eq!(
            demosaic(&tiny, &inputs, &ok),
            Err(Error::BadSize {
                width: 1,
                height: 4
            })
        );
        let mut nan = Mosaic::new(4, 4, Cfa::RGGB);
        nan.data[5] = f32::NAN;
        assert_eq!(demosaic(&nan, &inputs, &ok), Err(Error::BadInput));
        nan.data[5] = f32::INFINITY;
        assert_eq!(demosaic_whole(&nan, &inputs, &ok), Err(Error::BadInput));
        // A finite value that overflows the working scale is refused too.
        nan.data[5] = 1.0e38;
        assert_eq!(demosaic(&nan, &inputs, &ok), Err(Error::BadInput));
    }

    #[test]
    fn padding_puts_red_at_even_even_for_every_phase_and_core() {
        // The padded plane of a ramp whose value is the buffer column: red sites must land on
        // even x and even y of the plane for all four patterns and cores at odd origins.
        for cfa in Cfa::ALL {
            let (w, h) = (9, 11);
            let m = Mosaic::from_data(w, h, cfa, vec![0.5; w * h]).unwrap();
            for core in [Rect::new(0, 0, 9, 11), Rect::new(1, 3, 6, 5)] {
                let (_, lay) = padded_plane(&m, core, 10);
                for dy in 0..core.h {
                    for dx in 0..core.w {
                        let want = match m.cfa.color_at(core.x + dx, core.y + dy) {
                            CfaColor::Red => Site::R,
                            CfaColor::Green => Site::G,
                            CfaColor::Blue => Site::B,
                        };
                        let got = site(lay.core.x0 + dx, lay.core.y0 + dy);
                        assert!(got == want, "{} core {core:?} ({dx},{dy})", cfa.name());
                    }
                }
                assert!(lay.core.x0 >= 10 && lay.pw - lay.core.x1 >= 10);
                assert!(lay.core.y0 >= 10 && lay.ph - lay.core.y1 >= 10);
            }
        }
    }

    #[test]
    fn margins_match_the_spec_reach() {
        let plain = Margins::new(false);
        let matching = Margins::new(true);
        // The reach is the spec's sum of halos minus one (11 and 13 are safe bounds).
        assert_eq!((plain.mosaic, matching.mosaic), (10, 12));
        assert!(plain.mosaic <= HALO_PLAIN_PX && matching.mosaic <= HALO_MAX_PX);
    }
}
