//! C12 step 1c: the red chroma step, a smooth hue-preserving chroma gain on the reds and, on its up side,
//! the ochres and yellows, applied to linear AWG4 in front of ACES 2 (`docs/02-colour-pipeline.md`, "Red
//! chroma (C12 step 1c)"; owner decisions of 2026-10-24, "R1 fix looks better than what we have currently",
//! and 2026-10-25, "Yellows keep colour").
//!
//! # Why
//!
//! ACES 2 SDR takes chroma out of mid-tone and light reds at the same display lightness (10 to 22 %, the
//! reds "go coral") and adds 30 to 40 % to dark reds. It takes about a third out of ochres and yellows at
//! +1 stop over grey (display C* over colourimetric C* 0.66 to 0.70 at hue 70 to 100 degrees, 0.43 to 0.46
//! at +2 stops), so an ochre wall reads as peach. The step gives back what ACES 2 removes above about -0.5
//! stop, caps the result below the output gamut's edge, and (per look) can take back part of ACES 2's dark
//! boost of reds. Grey, skin and other hues are untouched.
//!
//! # The step
//!
//! On a linear AWG4 triple `rgb` (after the white lift, before the display placement and ACES 2):
//!
//! ```text
//! xyz  = AWG4_TO_XYZ * rgb ;  Y = xyz.y            return rgb if Y <= 1e-7 (or not finite)
//! s    = log2(Y / 0.18)                            stops over middle grey (held at 8 for the tables)
//! fade = smoothstep(-7.5, -5.5, s)
//! yd   = YD(s)                                     display luminance of the neutral axis through ACES 2
//! lab  = Lab_D65(xyz / Y * yd) ; Ld = L* of yd ; C = |ab| ; h = atan2(b, a)
//! kap  = clamp((20 / max(C, 8))^0.65, 0.35, 1.25)
//! W    = raised-cosine hue window: the red window (1 on [3, 31] deg, 0 below -27 and above 43 deg); on the
//!        up side also the yellow window (1 on [66, 95] deg, 0 below 58 and above 110 deg), whichever is larger;
//!        the dark side has the red window only
//! P(s) = 0.34 (s + 0.5) for s >= -0.5 ; -0.43 tanh((-0.5 - s) / 1.65) for s < -0.5
//! lnf  = kap * P
//! if P > 0:                                        the up side, strength f_up
//!     k0   = 1 + (kmax - 1) tanh((exp(lnf) - 1) / (kmax - 1)),   kmax = 1.45
//!     Cin  = C k0 ;  Cmax = CMAX_out(Ld, h)       (41 x 72 table per limiting gamut)
//!     Lim  = Cmax exp(kap(0.9 Cmax) P) ;  kn = 0.8 Lim
//!     Csoft = Cin if Cin <= kn else kn + (Lim - kn) tanh((Cin - kn) / (Lim - kn))
//!     x    = Csoft - C ;  Cb = C + (x > 0 ? x^2 / (x + 0.015 C) : 0)
//!     lnk  = f_up * W * ln(Cb / C)                 (ln k0 if C < 1e-4)
//! else:                                            the dark side, strength f_dn
//!     lnk  = f_dn * W * lnf
//! lnk *= fade ;  lab.ab *= exp(lnk) ;  Y kept ;  back to AWG4
//! ```
//!
//! `f_up` and `f_dn` are per-look strengths ([`RedChroma`]): `f_up` is 1 and `f_dn` ("dark relief") is 0 for
//! Neutral and for every look that does not set them; Signature sets both to 0, which skips the step.
//!
//! # Properties (all tested)
//!
//! * Grey is bit-exact: the Lab hue of the neutral axis (the offset of AWG4's white from the Lab reference
//!   white) is about 260 degrees, far outside the window, so the step returns its input.
//! * Hue is kept (the Lab hue at the display lightness changes by less than 1e-9 degrees), Y is kept.
//! * Skin hues (CIELAB 45 to 57 degrees, between the red and the yellow window) keep a weight of 0: the step
//!   returns them bit for bit.
//! * Bit-exact where the window is 0, where the fade is 0, where `f_up` or `f_dn` is 0 for the side the
//!   pixel is on, and for `Y <= 1e-7`, a negative `Y` or a non-finite component.
//! * Monotone in chroma (never folds): the output chroma grows with the input chroma at every stop and hue,
//!   up to 1.6 times the gamut edge, with a minimum slope of 0.09 (Rec.709) and 0.07 (P3) for the reds and
//!   0.066 and 0.056 for the yellows.
//! * `s` is held at [`S_MAX`] (the end of the neutral-axis table, +8 stops: display white) before `P` and
//!   the table lookups, so the step stays finite for the brightest scene values.
//!
//! # In a DNG
//!
//! The step is a scene-linear AWG4 to AWG4 map, so a DNG carries it in its pixels like the white lift
//! (`ColorChain::apply_dng_pixel`, `docs/09`). A DNG has no output gamut of its own: the step there uses the
//! P3-D65 tables whatever the chain's output ([`RedGamut::P3D65`]). A look with its own display rendering
//! above 0 % and an imported display LUT are inverted from their picture through plain ACES 2 and have no
//! step.
//!
//! # Tables
//!
//! Both tables are computed here, once per limiting gamut, from `ym-color`'s own ACES 2 and gamut code
//! ([`RedChromaTables`]): `YD(s)` is the display luminance of the neutral axis through ACES 2 with the
//! display placement, on a 0.25 stop grid from -10 to +8 stops (73 entries); `CMAX(L*, h)` is the largest
//! CIELAB chroma inside the gamut, a 41 (L* 0 to 100, step 2.5) by 72 (hue, step 5 degrees) grid found by
//! 40 bisection steps. They match the investigation's tables (`tests/data/red_chroma/*.csv`). The GPU
//! kernel reads the same numbers (`ym_gpu`).

use std::sync::LazyLock;

use super::{OutputTarget, aces2_input_matrix};
use crate::aces2::{Aces2Output, Aces2Transform, Primaries};
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::deltae::{lab_to_xyz, xyz_to_lab};
use crate::gamut::matrices;

/// The white of the CIELAB the step works in: the D65 white as tabulated by the CIE (`X 0.95047`,
/// `Z 1.08883`), which is what the investigation used. (AWG4's own white is `0.950456, 1, 1.089058`:
/// the 2e-4 gap puts the neutral axis at `C` 0.014 and hue 260 degrees, outside the window.)
pub const LAB_WHITE: Vec3 = [0.95047, 1.0, 1.08883];
/// CIE epsilon `216/24389` of the CIELAB the step works in (the cube of 6/29).
pub const LAB_EPSILON: f64 = 216.0 / 24389.0;
/// CIE kappa `24389/27` of the CIELAB the step works in.
pub const LAB_KAPPA: f64 = 24389.0 / 27.0;
/// Scene luminance of middle grey, the reference of `s`.
pub const GREY: f64 = 0.18;
/// Below this luminance (AWG4 `Y`) a pixel is returned unchanged.
pub const Y_MIN: f64 = 1e-7;
/// `s` is held at this many stops over grey: the end of the neutral-axis table.
pub const S_MAX: f64 = 8.0;

/// First stop of the neutral-axis table `YD`.
pub const YD_S_MIN: f64 = -10.0;
/// Step of the neutral-axis table, in stops.
pub const YD_STEP: f64 = 0.25;
/// Entries of the neutral-axis table: -10 to +8 stops.
pub const YD_LEN: usize = 73;
/// Smallest display luminance the step works with.
pub const YD_FLOOR: f64 = 1e-7;
/// Step of the `CMAX` table in L*.
pub const CMAX_L_STEP: f64 = 2.5;
/// Rows of the `CMAX` table (L* 0 to 100).
pub const CMAX_L_ROWS: usize = 41;
/// Step of the `CMAX` table in hue, degrees.
pub const CMAX_H_STEP: f64 = 5.0;
/// Columns of the `CMAX` table (hue 0 to 355 degrees).
pub const CMAX_H_COLS: usize = 72;
/// Bisection steps and upper bound of the chroma search that builds `CMAX`.
const CMAX_BISECT_STEPS: usize = 40;
const CMAX_BISECT_TOP: f64 = 250.0;
/// Tolerance of the in-gamut test of the search (linear display light).
const CMAX_GAMUT_TOL: f64 = 1e-9;
/// The `CMAX` row index is held just below the last row so the interpolation has a row above.
pub const CMAX_ROW_LIMIT: f64 = 40.0 - 1e-6;

/// Fade-out of the step in the dark: 0 below `FADE_LO` stops, 1 above `FADE_HI`.
pub const FADE_LO: f64 = -7.5;
/// See [`FADE_LO`].
pub const FADE_HI: f64 = -5.5;
/// The hue window: 1 on `WINDOW_A..=WINDOW_B` degrees, a raised cosine to 0 over `WINDOW_E_LO` below and
/// `WINDOW_E_HI` above.
pub const WINDOW_A: f64 = 3.0;
/// See [`WINDOW_A`].
pub const WINDOW_B: f64 = 31.0;
/// See [`WINDOW_A`].
pub const WINDOW_E_LO: f64 = 30.0;
/// See [`WINDOW_A`].
pub const WINDOW_E_HI: f64 = 12.0;
/// The yellow window of the up side (ochres, yellows, warm oranges): 1 on `YELLOW_WINDOW_A..=YELLOW_WINDOW_B`
/// degrees, a raised cosine to 0 over `YELLOW_WINDOW_E_LO` below and `YELLOW_WINDOW_E_HI` above. It starts at 58
/// degrees, after skin (CIELAB hue 45 to 62 degrees in the photos tested) and well after the red window's
/// end at 43 degrees, so the two never overlap and a band of hues between them is left alone.
pub const YELLOW_WINDOW_A: f64 = 66.0;
/// See [`YELLOW_WINDOW_A`].
pub const YELLOW_WINDOW_B: f64 = 95.0;
/// See [`YELLOW_WINDOW_A`].
pub const YELLOW_WINDOW_E_LO: f64 = 8.0;
/// See [`YELLOW_WINDOW_A`].
pub const YELLOW_WINDOW_E_HI: f64 = 15.0;
/// Where the profile `P` crosses 0: -0.5 stop.
pub const PROFILE_S0: f64 = -0.5;
/// Slope of `P` above `PROFILE_S0`, per stop.
pub const PROFILE_UP: f64 = 0.34;
/// Depth of the dark side of `P` (the limit of `-P` far below `PROFILE_S0`).
pub const PROFILE_DARK: f64 = 0.43;
/// Width of the dark side of `P`, in stops.
pub const PROFILE_TAU: f64 = 1.65;
/// The ceiling of the up-side gain at `kappa` 1.
pub const K_MAX: f64 = 1.45;
/// The soft cap starts at this fraction of its limit.
pub const KNEE: f64 = 0.8;
/// `kappa(C) = clamp((KAPPA_C0 / max(C, KAPPA_CMIN))^KAPPA_POW, KAPPA_LO, KAPPA_HI)`.
pub const KAPPA_C0: f64 = 20.0;
/// See [`KAPPA_C0`].
pub const KAPPA_CMIN: f64 = 8.0;
/// See [`KAPPA_C0`].
pub const KAPPA_POW: f64 = 0.65;
/// See [`KAPPA_C0`].
pub const KAPPA_LO: f64 = 0.35;
/// See [`KAPPA_C0`].
pub const KAPPA_HI: f64 = 1.25;
/// The gamut cap uses `kappa` at this fraction of `CMAX`.
pub const CAP_KAPPA_AT: f64 = 0.9;
/// Smallest gamut edge the cap uses (chroma), so the cap never closes on a grey.
pub const CMAX_FLOOR: f64 = 1.0;
/// The ramp that keeps the output chroma from falling below the input: `x^2 / (x + RAMP_EPS C)`.
pub const RAMP_EPS: f64 = 0.015;
/// Below this chroma the gain is `k0`, not the capped ratio.
pub const C_TINY: f64 = 1e-4;

/// The output gamut the cap and the neutral axis are for: ACES 2 limits to it. The sRGB output is limited
/// to Rec.709; the viewport, Display P3 and Rec.2020 outputs to P3-D65.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RedGamut {
    /// Rec.709 primaries (the sRGB output).
    Rec709,
    /// P3-D65 primaries (the viewport, Display P3 and Rec.2020 outputs).
    P3D65,
}

impl RedGamut {
    /// The gamut ACES 2 limits `output` to.
    #[must_use]
    pub fn of(output: OutputTarget) -> Self {
        if output.aces2().limiting_primaries() == Primaries::REC709 {
            Self::Rec709
        } else {
            Self::P3D65
        }
    }

    /// The gamut's place in the pair of table sets the kernel holds: 0 for Rec.709, 1 for P3-D65.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Rec709 => 0,
            Self::P3D65 => 1,
        }
    }

    /// The ACES 2 output whose neutral axis defines `YD` (any output with this limiting gamut gives the
    /// same axis).
    fn aces2_output(self) -> Aces2Output {
        match self {
            Self::Rec709 => Aces2Output::Srgb,
            Self::P3D65 => Aces2Output::Viewport,
        }
    }

    fn to_xyz(self) -> Mat3 {
        match self {
            Self::Rec709 => matrices().rec709_to_xyz,
            Self::P3D65 => matrices().p3_to_xyz,
        }
    }

    fn xyz_to_rgb(self) -> Mat3 {
        match self {
            Self::Rec709 => matrices().xyz_to_rec709,
            Self::P3D65 => matrices().xyz_to_p3,
        }
    }
}

/// The two tables of the step for one limiting gamut, built once per process (module notes).
#[derive(Debug, Clone, PartialEq)]
pub struct RedChromaTables {
    gamut: RedGamut,
    yd: [f64; YD_LEN],
    cmax: Vec<f64>,
}

static TABLES_REC709: LazyLock<RedChromaTables> =
    LazyLock::new(|| RedChromaTables::build(RedGamut::Rec709));
static TABLES_P3: LazyLock<RedChromaTables> =
    LazyLock::new(|| RedChromaTables::build(RedGamut::P3D65));

impl RedChromaTables {
    /// The shared tables of `gamut` (built on first use, a few milliseconds).
    #[must_use]
    pub fn shared(gamut: RedGamut) -> &'static RedChromaTables {
        match gamut {
            RedGamut::Rec709 => &TABLES_REC709,
            RedGamut::P3D65 => &TABLES_P3,
        }
    }

    /// Builds the tables from ACES 2 and the gamut matrices (deterministic).
    fn build(gamut: RedGamut) -> Self {
        Self {
            gamut,
            yd: build_yd(gamut),
            cmax: build_cmax(gamut),
        }
    }

    /// The gamut the tables are for.
    #[must_use]
    pub fn gamut(&self) -> RedGamut {
        self.gamut
    }

    /// `YD`: the display luminance (1.0 is the 100 nit white) of the neutral axis through ACES 2 with the
    /// display placement, at -10 to +8 stops over grey in steps of 0.25.
    #[must_use]
    pub fn yd(&self) -> &[f64; YD_LEN] {
        &self.yd
    }

    /// `CMAX`: the largest CIELAB chroma inside the gamut, `CMAX_L_ROWS` rows (L* 0 to 100 in steps of
    /// 2.5) of `CMAX_H_COLS` columns (hue 0 to 355 degrees in steps of 5), row-major.
    #[must_use]
    pub fn cmax(&self) -> &[f64] {
        &self.cmax
    }

    /// `YD` at `s` stops over grey, linear between the entries and held at the ends (`np.interp`).
    #[must_use]
    pub fn yd_at(&self, s: f64) -> f64 {
        let f = ((s - YD_S_MIN) / YD_STEP).clamp(0.0, (YD_LEN - 1) as f64);
        let i = (f as usize).min(YD_LEN - 2);
        let t = f - i as f64;
        self.yd[i] + t * (self.yd[i + 1] - self.yd[i])
    }

    /// `CMAX` at lightness `l` (L*) and hue `h` (degrees): bilinear, periodic in hue, held at L* 0 and 100.
    #[must_use]
    pub fn cmax_at(&self, l: f64, h: f64) -> f64 {
        let fl = (l / CMAX_L_STEP).clamp(0.0, CMAX_ROW_LIMIT);
        let i = fl as usize;
        let a = fl - i as f64;
        let fh = h.rem_euclid(360.0) / CMAX_H_STEP;
        let j = (fh as usize) % CMAX_H_COLS;
        let j1 = (j + 1) % CMAX_H_COLS;
        let b = fh - fh.floor();
        let at = |r: usize, c: usize| self.cmax[r * CMAX_H_COLS + c];
        (1.0 - a) * ((1.0 - b) * at(i, j) + b * at(i, j1))
            + a * ((1.0 - b) * at(i + 1, j) + b * at(i + 1, j1))
    }
}

/// `YD` from the ACES 2 transform of the gamut's limiting primaries: the neutral axis (the Lab reference
/// white at `0.18 * 2^s`) through the placement and ACES 2, as the luminance of the limited display light.
fn build_yd(gamut: RedGamut) -> [f64; YD_LEN] {
    let transform = Aces2Transform::shared(gamut.aces2_output());
    let to_ap0 = aces2_input_matrix();
    let xyz_to_awg4 = matrices().xyz_to_awg4;
    let y_row = gamut.to_xyz()[1];
    std::array::from_fn(|i| {
        let s = YD_S_MIN + YD_STEP * i as f64;
        let y = GREY * s.exp2();
        let awg4 = linalg::mul_vec(&xyz_to_awg4, LAB_WHITE.map(|c| y * c));
        let lin = transform.transform_ap0_limiting_linear(linalg::mul_vec(&to_ap0, awg4));
        y_row[0] * lin[0] + y_row[1] * lin[1] + y_row[2] * lin[2]
    })
}

/// `CMAX`: for every (L*, hue) the largest chroma whose colour is inside the gamut (bisection).
fn build_cmax(gamut: RedGamut) -> Vec<f64> {
    let m = gamut.xyz_to_rgb();
    let mut table = vec![0.0; CMAX_L_ROWS * CMAX_H_COLS];
    for i in 0..CMAX_L_ROWS {
        let l = CMAX_L_STEP * i as f64;
        for j in 0..CMAX_H_COLS {
            let h = (CMAX_H_STEP * j as f64).to_radians();
            let (sin_h, cos_h) = h.sin_cos();
            let (mut lo, mut hi) = (0.0_f64, CMAX_BISECT_TOP);
            for _ in 0..CMAX_BISECT_STEPS {
                let mid = 0.5 * (lo + hi);
                let xyz = lab_to_xyz([l, mid * cos_h, mid * sin_h], LAB_WHITE);
                let rgb = linalg::mul_vec(&m, xyz);
                let lowest = rgb[0].min(rgb[1]).min(rgb[2]);
                let highest = rgb[0].max(rgb[1]).max(rgb[2]);
                if lowest >= -CMAX_GAMUT_TOL && highest <= 1.0 + CMAX_GAMUT_TOL {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            table[i * CMAX_H_COLS + j] = lo;
        }
    }
    table
}

/// Why a [`RedChroma`] was refused.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum RedChromaError {
    /// A strength must be a number from 0 to 1.
    #[error("the red chroma strength {name} must be a number from 0 to 1, got {value}")]
    Strength {
        /// `up` or `dark_relief`.
        name: &'static str,
        /// The value given.
        value: f64,
    },
}

/// The two strengths of the red chroma step, per look.
///
/// * `up` (`f_up`): the strength of the up side, which gives back the chroma ACES 2 takes out of the
///   mid-tone and light reds (0 to 1; 1 is the full step).
/// * `dark_relief` (`f_dn`): the strength of the dark side, which takes back part of the boost ACES 2
///   gives to dark reds (0 to 1; 0 leaves ACES 2's dark reds alone).
///
/// [`RedChroma::DEFAULT`] (`up` 1, `dark_relief` 0) is Neutral's and every look's that does not set them;
/// [`RedChroma::OFF`] (both 0) skips the step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedChroma {
    up: f64,
    dark_relief: f64,
}

impl Default for RedChroma {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl RedChroma {
    /// Neutral's strengths: the full up side, no dark relief.
    pub const DEFAULT: Self = Self {
        up: 1.0,
        dark_relief: 0.0,
    };
    /// Both sides off: the step is skipped and the picture is plain ACES 2's.
    pub const OFF: Self = Self {
        up: 0.0,
        dark_relief: 0.0,
    };

    /// Strengths `up` and `dark_relief`.
    ///
    /// # Errors
    ///
    /// [`RedChromaError::Strength`] for a value that is not a number from 0 to 1.
    pub fn new(up: f64, dark_relief: f64) -> Result<Self, RedChromaError> {
        for (name, value) in [("up", up), ("dark_relief", dark_relief)] {
            if !(0.0..=1.0).contains(&value) {
                return Err(RedChromaError::Strength { name, value });
            }
        }
        Ok(Self { up, dark_relief })
    }

    /// The strength of the up side.
    #[must_use]
    pub fn up(&self) -> f64 {
        self.up
    }

    /// The strength of the dark side (the look's dark relief).
    #[must_use]
    pub fn dark_relief(&self) -> f64 {
        self.dark_relief
    }

    /// Whether both sides are off, so the step is skipped.
    #[must_use]
    pub fn is_off(&self) -> bool {
        self.up == 0.0 && self.dark_relief == 0.0
    }

    /// `self + k (other - self)`: exactly `self` at `k` 0 and exactly `other` at `k` 1. A look at
    /// intensity `k` blends from Neutral's strengths to its own, as its colours blend from Neutral's.
    #[must_use]
    pub fn blend(self, other: Self, k: f64) -> Self {
        if k <= 0.0 {
            self
        } else if k >= 1.0 {
            other
        } else {
            Self {
                up: self.up + k * (other.up - self.up),
                dark_relief: self.dark_relief + k * (other.dark_relief - self.dark_relief),
            }
        }
    }
}

/// `smoothstep(e0, e1, x)`.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The signed hue difference `a - b` in degrees, in `[-180, 180)`.
fn hue_diff(a: f64, b: f64) -> f64 {
    (a - b + 180.0).rem_euclid(360.0) - 180.0
}

/// A raised-cosine window on hue (`h` in degrees): 1 on `a..=b`, 0 more than `e_lo` below `a` and more than
/// `e_hi` above `b`, C1 at both ends.
fn raised_window(h: f64, a: f64, b: f64, e_lo: f64, e_hi: f64) -> f64 {
    let d_lo = hue_diff(h, a);
    let d_hi = hue_diff(h, b);
    let mut w = 1.0;
    if d_lo < 0.0 {
        w = 0.5 * (1.0 + (std::f64::consts::PI * (-d_lo / e_lo).clamp(0.0, 1.0)).cos());
    }
    if d_hi > 0.0 {
        w = 0.5 * (1.0 + (std::f64::consts::PI * (d_hi / e_hi).clamp(0.0, 1.0)).cos());
    }
    w
}

/// The red hue window `W(h)` (`h` in degrees): the window of both sides of the step.
#[must_use]
pub fn hue_window(h: f64) -> f64 {
    raised_window(h, WINDOW_A, WINDOW_B, WINDOW_E_LO, WINDOW_E_HI)
}

/// The yellow hue window (`h` in degrees): 1 on 66 to 95 degrees, 0 below 58 and above 110. Only the up side
/// of the step uses it.
#[must_use]
pub fn yellow_window(h: f64) -> f64 {
    raised_window(
        h,
        YELLOW_WINDOW_A,
        YELLOW_WINDOW_B,
        YELLOW_WINDOW_E_LO,
        YELLOW_WINDOW_E_HI,
    )
}

/// The hue window of the up side: the red window and the yellow window (they never overlap).
#[must_use]
pub fn hue_window_up(h: f64) -> f64 {
    hue_window(h).max(yellow_window(h))
}

/// `kappa(C)`: how much of the flat gain a colour of CIELAB chroma `c` takes (more for muted colours).
#[must_use]
pub fn kappa(c: f64) -> f64 {
    (KAPPA_C0 / c.max(KAPPA_CMIN))
        .powf(KAPPA_POW)
        .clamp(KAPPA_LO, KAPPA_HI)
}

/// `P(s)`: the log of the flat gain at `kappa` 1, `s` stops over grey.
#[must_use]
pub fn profile(s: f64) -> f64 {
    if s >= PROFILE_S0 {
        PROFILE_UP * (s - PROFILE_S0)
    } else {
        -PROFILE_DARK * ((PROFILE_S0 - s) / PROFILE_TAU).tanh()
    }
}

/// The step prepared for one output gamut and one pair of strengths (immutable, `Send` and `Sync`).
#[derive(Debug, Clone, Copy)]
pub struct RedChromaStep {
    tables: &'static RedChromaTables,
    strength: RedChroma,
}

/// Everything [`RedChromaStep::lch`] reports about a pixel, for tests and tools.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedChromaLch {
    /// Stops over middle grey of the pixel's luminance (not held at `S_MAX`).
    pub s: f64,
    /// L* of the display luminance of the neutral axis at that `s`.
    pub l: f64,
    /// CIELAB chroma of the pixel re-lit at that luminance.
    pub c: f64,
    /// Its CIELAB hue in degrees, `[0, 360)`.
    pub h: f64,
}

impl RedChromaStep {
    /// The step for the gamut ACES 2 limits `output` to, with `strength`.
    #[must_use]
    pub fn new(output: OutputTarget, strength: RedChroma) -> Self {
        Self::for_gamut(RedGamut::of(output), strength)
    }

    /// The step for `gamut` with `strength`.
    #[must_use]
    pub fn for_gamut(gamut: RedGamut, strength: RedChroma) -> Self {
        Self {
            tables: RedChromaTables::shared(gamut),
            strength,
        }
    }

    /// The tables the step reads.
    #[must_use]
    pub fn tables(&self) -> &'static RedChromaTables {
        self.tables
    }

    /// The strengths the step was made with.
    #[must_use]
    pub fn strength(&self) -> RedChroma {
        self.strength
    }

    /// The luminance, stops over grey, display lightness, chroma and hue the step sees in `rgb`
    /// (linear AWG4): `None` for a pixel the step returns unchanged because of its luminance or because
    /// a component is not a finite number.
    #[must_use]
    pub fn lch(&self, rgb: Vec3) -> Option<RedChromaLch> {
        let (xyz, s) = self.luminance(rgb)?;
        let yd = self.tables.yd_at(s.min(S_MAX)).clamp(YD_FLOOR, 1.0);
        let lab = xyz_to_lab(relit(xyz, yd), LAB_WHITE);
        Some(RedChromaLch {
            s,
            l: lab[0],
            c: lab[1].hypot(lab[2]),
            h: lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0),
        })
    }

    /// XYZ and `s` of `rgb`, `None` when the step leaves it alone for its luminance or a non-finite value.
    fn luminance(&self, rgb: Vec3) -> Option<(Vec3, f64)> {
        if !rgb.iter().all(|c| c.is_finite()) {
            return None;
        }
        let xyz = linalg::mul_vec(&matrices().awg4_to_xyz, rgb);
        let y = xyz[1];
        if y.is_nan() || y <= Y_MIN || !y.is_finite() {
            return None;
        }
        Some((xyz, (y / GREY).log2()))
    }

    /// The natural log of the chroma gain the step applies to `rgb` (linear AWG4), 0 where it returns
    /// the input unchanged.
    #[must_use]
    pub fn ln_gain(&self, rgb: Vec3) -> f64 {
        self.gain(rgb).map_or(0.0, |g| g.lnk)
    }

    /// What the step does to `rgb`, or `None` when it returns the input unchanged.
    fn gain(&self, rgb: Vec3) -> Option<Gain> {
        let (xyz, s_raw) = self.luminance(rgb)?;
        let fade = smoothstep(FADE_LO, FADE_HI, s_raw);
        if fade <= 0.0 {
            return None;
        }
        let s = s_raw.min(S_MAX);
        let p = profile(s);
        let RedChroma { up, dark_relief } = self.strength;
        // The side the pixel is on must have a strength, or nothing happens.
        if (p > 0.0 && up == 0.0) || (p <= 0.0 && dark_relief == 0.0) {
            return None;
        }
        let yd = self.tables.yd_at(s).clamp(YD_FLOOR, 1.0);
        let lab = xyz_to_lab(relit(xyz, yd), LAB_WHITE);
        let c = lab[1].hypot(lab[2]);
        let h = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
        // The dark side is for reds only: ACES 2's boost of dark yellows is left alone.
        let w = if p > 0.0 {
            hue_window_up(h)
        } else {
            hue_window(h)
        };
        if w <= 0.0 {
            return None;
        }
        let kap = kappa(c);
        let lnf = kap * p;
        let lnk = if p > 0.0 {
            let raw = lnf.exp();
            let k0 = if raw > 1.0 {
                1.0 + (K_MAX - 1.0) * ((raw - 1.0) / (K_MAX - 1.0)).tanh()
            } else {
                raw
            };
            let c_in = c * k0;
            let c_max = self.tables.cmax_at(lab[0], h).max(CMAX_FLOOR);
            let limit = c_max * (kappa(CAP_KAPPA_AT * c_max) * p).exp();
            let knee = KNEE * limit;
            let c_soft = if c_in <= knee {
                c_in
            } else {
                knee + (limit - knee) * ((c_in - knee) / (limit - knee).max(1e-6)).tanh()
            };
            let x = c_soft - c;
            let c_boost = c + if x > 0.0 {
                x * x / (x + RAMP_EPS * c + 1e-9)
            } else {
                0.0
            };
            let ln_ratio = if c > C_TINY {
                (c_boost.max(1e-9) / c.max(C_TINY)).ln()
            } else {
                k0.ln()
            };
            up * w * ln_ratio
        } else {
            dark_relief * w * lnf
        };
        let lnk = lnk * fade;
        if lnk == 0.0 {
            return None;
        }
        Some(Gain { xyz, lab, yd, lnk })
    }

    /// The red chroma step on a linear AWG4 triple (in front of the display placement and ACES 2).
    ///
    /// Returns `rgb` itself, bit for bit, when the step does nothing (module notes). Never returns a NaN
    /// for a finite input; a non-finite input comes back unchanged.
    #[must_use]
    pub fn apply(&self, rgb: Vec3) -> Vec3 {
        let Some(Gain { xyz, lab, yd, lnk }) = self.gain(rgb) else {
            return rgb;
        };
        let k = lnk.exp();
        let scaled = lab_to_xyz([lab[0], lab[1] * k, lab[2] * k], LAB_WHITE);
        let y = xyz[1];
        let out = linalg::mul_vec(&matrices().xyz_to_awg4, scaled.map(|c| c / yd * y));
        if out.iter().all(|c| c.is_finite()) {
            out
        } else {
            rgb
        }
    }
}

/// A pixel the step changes: its XYZ, its Lab at the display luminance `yd`, and `ln k` (not 0).
struct Gain {
    xyz: Vec3,
    lab: Vec3,
    yd: f64,
    lnk: f64,
}

/// The XYZ of the pixel re-lit at the display luminance `yd` of the neutral axis: its chromaticity at `Y = yd`.
fn relit(xyz: Vec3, yd: f64) -> Vec3 {
    let y = xyz[1];
    [xyz[0] / y * yd, yd, xyz[2] / y * yd]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(gamut: RedGamut, up: f64, dn: f64) -> RedChromaStep {
        RedChromaStep::for_gamut(gamut, RedChroma::new(up, dn).unwrap())
    }

    /// A linear AWG4 pixel at `s` stops over grey whose Lab at the display luminance has chroma `c` and hue `h`.
    fn pixel(t: &RedChromaTables, s: f64, h: f64, c: f64) -> Vec3 {
        let yd = t.yd_at(s.min(S_MAX)).clamp(YD_FLOOR, 1.0);
        let l = xyz_to_lab([LAB_WHITE[0] * yd, yd, LAB_WHITE[2] * yd], LAB_WHITE)[0];
        let hr = h.to_radians();
        let rel = lab_to_xyz([l, c * hr.cos(), c * hr.sin()], LAB_WHITE);
        let y = GREY * s.exp2();
        linalg::mul_vec(&matrices().xyz_to_awg4, rel.map(|v| v / yd * y))
    }

    /// The window the step uses at `s` stops over grey: the up side (above -0.5 stop) has the red and the
    /// yellow window, the dark side the red one.
    fn window_of(s: f64, h: f64) -> f64 {
        if s > PROFILE_S0 {
            hue_window_up(h)
        } else {
            hue_window(h)
        }
    }

    #[test]
    fn grey_is_bit_exact_at_every_stop() {
        for gamut in [RedGamut::Rec709, RedGamut::P3D65] {
            let st = step(gamut, 1.0, 1.0);
            let mut s = -14.0;
            while s <= 12.0 {
                let v = GREY * f64::exp2(s);
                assert_eq!(st.apply([v, v, v]), [v, v, v], "s {s}");
                s += 0.0625;
            }
        }
    }

    #[test]
    fn dark_non_finite_and_negative_luminance_come_back_unchanged() {
        let st = step(RedGamut::P3D65, 1.0, 1.0);
        for rgb in [
            [0.0, 0.0, 0.0],
            [1e-9, 2e-9, 1e-9],
            [-0.1, -0.2, -0.3],
            [0.3, -0.5, -0.9],
            [f64::NAN, 0.2, 0.1],
            [0.2, f64::INFINITY, 0.1],
            [0.2, 0.1, f64::NEG_INFINITY],
        ] {
            let out = st.apply(rgb);
            for (a, b) in out.iter().zip(rgb) {
                assert!(a.to_bits() == b.to_bits(), "{rgb:?} gave {out:?}");
            }
        }
    }

    #[test]
    fn extreme_finite_values_stay_finite() {
        let st = step(RedGamut::P3D65, 1.0, 1.0);
        for rgb in [
            [1e30, 3e29, 1e29],
            [1e30, -1e30, 1e-3],
            [-1e30, 1e30, 1e30],
            [1e-7, 4e-8, 3e-8],
            [1e6, 1e5, 1e4],
        ] {
            let out = st.apply(rgb);
            assert!(out.iter().all(|c| c.is_finite()), "{rgb:?} gave {out:?}");
        }
    }

    #[test]
    fn hue_and_luminance_are_kept_and_the_window_is_respected() {
        for gamut in [RedGamut::Rec709, RedGamut::P3D65] {
            let st = step(gamut, 1.0, 1.0);
            let t = st.tables();
            let mut worst_hue = 0.0_f64;
            let mut worst_y = 0.0_f64;
            let mut changed = 0;
            for s in [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 7.0] {
                for h in (0..360).step_by(3) {
                    for c in [3.0, 10.0, 25.0, 45.0, 70.0] {
                        let rgb = pixel(t, s, f64::from(h), c);
                        let out = st.apply(rgb);
                        let (a, b) = (st.lch(rgb).unwrap(), st.lch(out).unwrap());
                        if out != rgb {
                            changed += 1;
                            let dh = hue_diff(a.h, b.h).abs();
                            worst_hue = worst_hue.max(dh);
                            let ya = linalg::mul_vec(&matrices().awg4_to_xyz, rgb)[1];
                            let yb = linalg::mul_vec(&matrices().awg4_to_xyz, out)[1];
                            worst_y = worst_y.max(((yb - ya) / ya).abs());
                            // Only the windows (with their tails: reds -27 to 43 degrees, and on the up
                            // side yellows 58 to 110 degrees) are touched.
                            assert!(window_of(s, a.h) > 0.0, "s {s} h {} moved", a.h);
                        } else if window_of(s, a.h) > 0.01 && s >= -5.4 && s != -0.5 {
                            // Inside the window the step does something (the up side, or the dark side
                            // at strength 1), unless the colour is already near the gamut cap.
                            let edge = t.cmax_at(a.l, a.h).max(CMAX_FLOOR);
                            assert!(
                                a.c < 0.2 || a.c > 0.8 * edge,
                                "no change at s {s} h {} c {} edge {edge}",
                                a.h,
                                a.c
                            );
                        }
                    }
                }
            }
            assert!(changed > 100, "the step did nothing ({changed})");
            assert!(worst_hue < 1e-9, "hue moved {worst_hue} degrees");
            assert!(worst_y < 1e-12, "luminance moved {worst_y}");
        }
    }

    #[test]
    fn dark_relief_zero_leaves_everything_below_the_cross_untouched() {
        for gamut in [RedGamut::Rec709, RedGamut::P3D65] {
            let st = step(gamut, 1.0, 0.0);
            let t = st.tables();
            for s in [-9.0, -6.0, -5.0, -3.0, -1.5, -0.8, -0.5] {
                for h in (0..360).step_by(2) {
                    for c in [2.0, 12.0, 30.0, 60.0] {
                        let rgb = pixel(t, s, f64::from(h), c);
                        assert_eq!(st.apply(rgb), rgb, "s {s} h {h} c {c}");
                    }
                }
            }
            // And dark relief does act below the cross.
            let rgb = pixel(t, -3.0, 20.0, 30.0);
            assert_ne!(step(gamut, 1.0, 0.5).apply(rgb), rgb);
        }
    }

    #[test]
    fn up_zero_leaves_everything_above_the_cross_untouched() {
        for gamut in [RedGamut::Rec709, RedGamut::P3D65] {
            let st = step(gamut, 0.0, 1.0);
            let t = st.tables();
            for s in [-0.4, 0.0, 1.0, 3.0, 7.0, 8.5] {
                for h in (0..360).step_by(2) {
                    for c in [2.0, 12.0, 30.0, 60.0] {
                        let rgb = pixel(t, s, f64::from(h), c);
                        assert_eq!(st.apply(rgb), rgb, "s {s} h {h} c {c}");
                    }
                }
            }
        }
    }

    #[test]
    fn outside_the_window_and_in_the_deep_dark_nothing_changes() {
        let st = step(RedGamut::P3D65, 1.0, 1.0);
        let t = st.tables();
        for s in [-9.0, -8.0, -7.6] {
            for h in 0..360 {
                let rgb = pixel(t, s, f64::from(h), 30.0);
                assert_eq!(st.apply(rgb), rgb, "fade 0 at s {s}");
            }
        }
        for s in [0.5, 3.0] {
            // The up side: the red window is 0 from 43 degrees to 333, the yellow window is 0 below 58 and
            // above 110. Skin hues (44 to 57) and everything from 111 to 333 are untouched.
            for h in (44..58).chain(111..333) {
                let rgb = pixel(t, s, f64::from(h), 30.0);
                assert_eq!(st.apply(rgb), rgb, "window 0 at s {s} h {h}");
            }
        }
        for s in [-4.0, -1.0] {
            // The dark side is for reds only: yellows are untouched there, whatever the strength.
            for h in 44..333 {
                let rgb = pixel(t, s, f64::from(h), 30.0);
                assert_eq!(st.apply(rgb), rgb, "dark side, window 0 at s {s} h {h}");
            }
        }
    }

    #[test]
    fn yellows_get_the_up_side_only() {
        for gamut in [RedGamut::Rec709, RedGamut::P3D65] {
            let up = step(gamut, 1.0, 0.0);
            let dn = step(gamut, 0.0, 1.0);
            let t = up.tables();
            for h in [66.0, 80.0, 95.0] {
                // Up side: the chroma grows (a mid-tone ochre at +1 stop).
                let rgb = pixel(t, 1.0, h, 25.0);
                let (a, b) = (up.lch(rgb).unwrap(), up.lch(up.apply(rgb)).unwrap());
                assert!(b.c > 1.2 * a.c, "h {h}: chroma {} to {}", a.c, b.c);
                // A strength of 0 on the up side leaves it alone; the dark side never touches yellows.
                assert_eq!(dn.apply(rgb), rgb, "h {h} with f_up 0");
                let dark = pixel(t, -3.0, h, 25.0);
                assert_eq!(step(gamut, 1.0, 1.0).apply(dark), dark, "dark yellow h {h}");
            }
        }
    }

    #[test]
    fn the_window_has_the_stated_shape() {
        assert_eq!(hue_window(3.0), 1.0);
        assert_eq!(hue_window(17.0), 1.0);
        assert_eq!(hue_window(31.0), 1.0);
        assert!((hue_window(-27.0)).abs() < 1e-15 && (hue_window(333.0)).abs() < 1e-15);
        assert!((hue_window(43.0)).abs() < 1e-15);
        assert!((hue_window(-12.0) - 0.5).abs() < 1e-15);
        assert!((hue_window(37.0) - 0.5).abs() < 1e-15);
        assert_eq!(hue_window(180.0), 0.0);
        assert_eq!(hue_window(100.0), 0.0);
        // Continuous across 0/360.
        assert!((hue_window(359.999) - hue_window(-0.001)).abs() < 1e-12);
    }

    #[test]
    fn the_yellow_window_has_the_stated_shape() {
        // 1 on 66 to 95 degrees, a raised cosine to 0 over 8 degrees below and 15 above.
        for h in [66.0, 70.0, 80.0, 95.0] {
            assert_eq!(yellow_window(h), 1.0, "{h}");
        }
        assert!(yellow_window(58.0).abs() < 1e-15 && yellow_window(110.0).abs() < 1e-15);
        assert!((yellow_window(62.0) - 0.5).abs() < 1e-15);
        assert!((yellow_window(102.5) - 0.5).abs() < 1e-15);
        for h in [0.0, 30.0, 45.0, 57.9, 110.1, 150.0, 260.0, 359.9] {
            assert_eq!(yellow_window(h), 0.0, "{h}");
        }
        // C1: the slope is 0 where the window meets 0 and 1, and the window is continuous with a bounded
        // slope everywhere (pi / 2 / 8 per degree at most on the short side).
        let eps = 1e-6;
        for edge in [58.0, 66.0, 95.0, 110.0] {
            let slope = (yellow_window(edge + eps) - yellow_window(edge - eps)) / (2.0 * eps);
            assert!(slope.abs() < 1e-4, "slope {slope} at {edge}");
        }
        let mut prev = yellow_window(40.0);
        let mut h = 40.0;
        while h < 130.0 {
            h += 0.01;
            let w = yellow_window(h);
            assert!((w - prev).abs() < 0.0035, "jump at {h}");
            prev = w;
        }
        // The two windows never overlap, so the up side's window is the larger of them.
        for i in 0..3600 {
            let h = f64::from(i) * 0.1;
            assert!(hue_window(h) == 0.0 || yellow_window(h) == 0.0, "{h}");
            assert_eq!(hue_window_up(h), hue_window(h).max(yellow_window(h)));
        }
        // Skin hues (45 to 57) keep weight 0, and the band between the windows is 43 to 58 degrees.
        assert_eq!(hue_window_up(50.0), 0.0);
        assert_eq!(hue_window_up(57.0), 0.0);
    }

    /// Output chroma against input chroma, over stops (-6 to +6), hues (-40 to +50 degrees for the reds, 56 to
    /// 115 for the yellows) and chroma up to a multiple of the gamut edge: never falls (no fold). Minimum slope
    /// measured (Python and Rust agree), reds: Rec.709 0.501 up to 1.0 times the edge, 0.158 up to 1.25, 0.095
    /// up to 1.4 and 0.0905 up to 1.6; P3 0.493, 0.171, 0.108 and 0.0700 (at s +2, hue 11, chroma 69, edge
    /// 43). Yellows: see `YELLOW_SLOPE_FLOORS`.
    #[test]
    fn chroma_is_monotone_up_to_one_point_six_times_the_gamut_edge() {
        for (hues, range_name) in [(-40..=50, "reds"), (56..=115, "yellows")] {
            for (extent, floors) in [(1.25, [0.15, 0.15]), (1.6, [0.09, 0.069])] {
                let floors = if range_name == "yellows" {
                    YELLOW_SLOPE_FLOORS[usize::from(extent > 1.3)]
                } else {
                    floors
                };
                for (gamut, floor) in [RedGamut::Rec709, RedGamut::P3D65].into_iter().zip(floors) {
                    let mut slopes = Vec::new();
                    for dn in [0.0, 1.0] {
                        let st = step(gamut, 1.0, dn);
                        let t = st.tables();
                        let mut min_slope = f64::INFINITY;
                        for s in (-12..=12).map(|i| f64::from(i) * 0.5) {
                            for h in hues.clone().map(f64::from) {
                                let h = h.rem_euclid(360.0);
                                let Some(probe) = st.lch(pixel(t, s, h, 20.0)) else {
                                    continue;
                                };
                                let edge = t.cmax_at(probe.l, h).max(CMAX_FLOOR);
                                let n = 120;
                                let mut prev: Option<(f64, f64)> = None;
                                for i in 1..=n {
                                    let c = extent * edge * f64::from(i) / f64::from(n);
                                    let rgb = pixel(t, s, h, c);
                                    let (Some(a), Some(b)) = (st.lch(rgb), st.lch(st.apply(rgb)))
                                    else {
                                        continue;
                                    };
                                    if let Some((c0, o0)) = prev {
                                        min_slope = min_slope.min((b.c - o0) / (a.c - c0));
                                    }
                                    prev = Some((a.c, b.c));
                                }
                            }
                        }
                        assert!(
                            min_slope >= floor,
                            "{range_name} {gamut:?} dn {dn} to {extent} x edge: minimum slope {min_slope} under {floor}"
                        );
                        slopes.push(min_slope);
                    }
                    println!(
                        "{range_name} {gamut:?} up to {extent} x edge: minimum slope {slopes:?}"
                    );
                }
            }
        }
    }

    /// The smallest slope of output against input chroma allowed in the yellow window, up to 1.25 and to 1.6
    /// times the gamut edge: `[[Rec.709, P3-D65]; 2]`.
    const YELLOW_SLOPE_FLOORS: [[f64; 2]; 2] = [[0.14, 0.11], [0.065, 0.055]];

    /// The tables ym-color builds from its own ACES 2 and gamut code are the investigation's.
    #[test]
    fn tables_match_the_investigation_csvs() {
        fn rows(text: &str) -> Vec<Vec<f64>> {
            text.lines()
                .skip(1)
                .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
                .collect()
        }
        let yd = rows(include_str!("../../tests/data/red_chroma/yd_of_s.csv"));
        assert_eq!(yd.len(), YD_LEN);
        for (gamut, col, name, text) in [
            (
                RedGamut::Rec709,
                1,
                "srgb",
                include_str!("../../tests/data/red_chroma/cmax_srgb.csv"),
            ),
            (
                RedGamut::P3D65,
                2,
                "p3",
                include_str!("../../tests/data/red_chroma/cmax_p3.csv"),
            ),
        ] {
            let t = RedChromaTables::shared(gamut);
            let mut worst_yd = 0.0_f64;
            for (i, row) in yd.iter().enumerate() {
                assert!((row[0] - (YD_S_MIN + YD_STEP * i as f64)).abs() < 1e-12);
                worst_yd = worst_yd.max(((t.yd()[i] - row[col]) / row[col]).abs());
            }
            let cmax = rows(text);
            assert_eq!((cmax.len(), cmax[0].len()), (CMAX_L_ROWS, CMAX_H_COLS));
            let mut worst_cmax = 0.0_f64;
            for (i, row) in cmax.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    worst_cmax = worst_cmax.max((t.cmax()[i * CMAX_H_COLS + j] - v).abs());
                }
            }
            println!(
                "{name}: YD max relative difference {worst_yd:.2e}, CMAX max difference {worst_cmax:.2e}"
            );
            assert!(worst_yd < 1e-8, "{name}: YD differs by {worst_yd:e}");
            // The CSVs carry three decimals.
            assert!(
                worst_cmax <= 5.5e-4,
                "{name}: CMAX differs by {worst_cmax:e}"
            );
        }
    }

    /// The numpy implementation of `tools/vectors/red_chroma_vectors.py` (the investigation's maths).
    #[test]
    fn matches_the_python_reference() {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/red_chroma/vectors.json")).unwrap();
        let tolerance = doc["tolerance"].as_f64().unwrap();
        let samples = doc["samples"].as_array().unwrap();
        assert!(samples.len() >= 600);
        let triple = |v: &serde_json::Value| -> Vec3 {
            let a = v.as_array().unwrap();
            [
                a[0].as_f64().unwrap(),
                a[1].as_f64().unwrap(),
                a[2].as_f64().unwrap(),
            ]
        };
        let (mut worst, mut changed, mut worst_rel) = (0.0_f64, 0, 0.0_f64);
        for smp in samples {
            let gamut = match smp["gamut"].as_str().unwrap() {
                "rec709" => RedGamut::Rec709,
                _ => RedGamut::P3D65,
            };
            let st = step(
                gamut,
                smp["f_up"].as_f64().unwrap(),
                smp["f_dn"].as_f64().unwrap(),
            );
            let (input, want) = (triple(&smp["in"]), triple(&smp["out"]));
            let got = st.apply(input);
            if got != input {
                changed += 1;
            } else {
                assert_eq!(smp["ln_k"].as_f64().unwrap(), 0.0, "{smp}");
            }
            for ch in 0..3 {
                let d = (got[ch] - want[ch]).abs();
                worst = worst.max(d);
                worst_rel = worst_rel.max(d / want[ch].abs().max(1e-3));
            }
        }
        println!(
            "Python to Rust: max abs difference {worst:.2e} (relative to max(|v|, 1e-3) {worst_rel:.2e}) over {} samples, {changed} changed",
            samples.len()
        );
        assert!(changed > 150, "only {changed} samples changed");
        assert!(
            worst < tolerance && worst_rel < tolerance,
            "{worst:e} {worst_rel:e}"
        );
    }

    #[test]
    fn strengths_validate_and_blend() {
        assert!(RedChroma::new(1.0, 0.0).is_ok());
        assert!(RedChroma::new(0.0, 1.0).is_ok());
        assert!(RedChroma::new(-0.1, 0.0).is_err());
        assert!(RedChroma::new(1.1, 0.0).is_err());
        assert!(RedChroma::new(0.5, f64::NAN).is_err());
        assert!(RedChroma::OFF.is_off() && !RedChroma::DEFAULT.is_off());
        let look = RedChroma::new(0.0, 0.5).unwrap();
        assert_eq!(RedChroma::DEFAULT.blend(look, 0.0), RedChroma::DEFAULT);
        assert_eq!(RedChroma::DEFAULT.blend(look, 1.0), look);
        let half = RedChroma::DEFAULT.blend(look, 0.5);
        assert_eq!((half.up(), half.dark_relief()), (0.5, 0.25));
    }
}
