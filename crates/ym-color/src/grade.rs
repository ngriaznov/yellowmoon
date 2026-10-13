//! Primary grade in LogC4: contrast, brightness, zones, colour balance, saturation (stage C9) (`docs/02-colour-pipeline.md` section C9).
//!
//! `v` is the LogC4 RGB triple. `L = encode(Y_lin)` is the LogC4 code value of the luminance
//! (the AWG4 `Y` row) of the **linear** values before C8, so it is a property of the pixel that
//! the grade does not change. Grey is `g = encode(0.18) = 0.27840`; one stop in LogC4 is
//! `S = b / 14 = 0.06480` code values. The controls are applied in this order:
//!
//! | Control | Range | Operation |
//! |---|---|---|
//! | Contrast `c` | -100..100 | `v = g + (v - g) 2^(0.6 c / 100)` |
//! | Brightness `b` | -100..100 | `v += (b / 100) S bump(L; g, 2.5 S)` |
//! | Shadows, Highlights, Black, White | -100..100 | `v += (x / 100) 1.5 S bump(L; centre, 1.5 S)`, centres at grey -3, +3, -6, +6 stops |
//! | Colour balance (shadows, midtones, highlights) | `(a, b)` in the unit disc | `v += 0.03 (a e1 + b e2) zone(L)` |
//! | Saturation `s` | -100..100 | `v = M + (v - M) (1 + s / 100)`, `M = max(r, g, b)` of the pixel at this point |
//!
//! with `bump(L; mu, sigma) = exp(-(L - mu)^2 / (2 sigma^2))`, `e1 = (1, -1/2, -1/2) / sqrt(1.5)` and
//! `e2 = (0, 1, -1) / sqrt(2)`. All constants are starting values (`docs/02`).
//!
//! # Decisions where the spec is silent
//!
//! * **Stops are scene-referred.** "Grey -3 stops" is the LogC4 code value of `0.18 * 2^-3` (and so
//!   on), not `g - 3 S`: LogC4 is only linear in stops far above the toe, and `g - 6 S` would fall
//!   below the code value of black. The widths (`1.5 S`, `2.5 S`) are code-value distances.
//! * **Zone shape.** `zone(L)` is `bump(L; centre, ZONE_SIGMA)` with `ZONE_SIGMA = 1.5 S`, centred at
//!   -3, 0 and +3 stops. The zones are Gaussian bumps, not a partition of unity, so colour balance
//!   leaves the extremes of the range alone.
//! * **Saturation is subtractive.** The pivot is `M`, the brightest channel of the pixel after the
//!   controls above it (owner's decision of 2026-10-13, `docs/16`), not the luma `L`. More
//!   saturation makes a colour denser and darker, less makes it paler and lighter, and the brightest
//!   channel and every neutral pixel stay where they are. At `s = -100` every channel equals `M`
//!   (a neutral at the pixel's brightest channel). Saturation does not clamp, as nothing else in C9
//!   does: a channel that goes below LogC4 black at `s > 0` is passed on as it is (C8 and the DRT
//!   take negatives). The ASC CDL saturation inside looks (C10, Rec.709 weights) is a standard and
//!   is not changed.
//! * **Sign.** For Shadows, Highlights, Black and White a positive value raises the pixel (brighter),
//!   a negative value lowers it. Positive brightness brightens.
//! * Out-of-range control values are clamped to their range; a colour-balance vector longer than 1
//!   is scaled back to the unit circle.
//!
//! # Monotonicity (what the tests check)
//!
//! * *Per pixel, in the control value:* contrast moves `v` away from `g` more the higher the value
//!   (for `v > g` the output is non-decreasing in `c`, for `v < g` non-increasing). Brightness,
//!   Shadows, Highlights, Black and White add a non-negative bump times the control, so every
//!   output channel is non-decreasing in the control. Saturation scales `v - M` by a
//!   non-decreasing factor, so `|v - M|` is non-decreasing in `s` for every channel; `M` and neutral
//!   pixels are unchanged, and for `s > 0` each non-maximum channel goes down (for `s < 0`, up
//!   toward `M`). Colour balance displaces `v` by
//!   `0.03 r zone(L)` along a fixed direction, so the displacement length is non-decreasing in the
//!   radius `r`.
//! * *As a tone map on a neutral ramp:* with one control moved over its full range, the output for
//!   neutral input is strictly increasing in the input (the slope stays above 0.39).
//! * *Not guaranteed:* several tonal controls stacked at their extremes (for example Shadows, Black
//!   and Brightness all at -100 with Contrast at -100) can fold the neutral ramp, because each bump
//!   contributes a slope of up to 0.4 to 0.6 and the slopes add. `tests/grade.rs` prints the worst
//!   slopes. A UI that allows the full stack should limit or re-fit the combination.

use crate::consts::{LOGC4_GRAY, LOGC4_STOP, MIDDLE_GRAY, Vec3};
use crate::gamut::LinearAwg4;
use crate::logc::{LogC4, logc4_encode};

/// Width of the brightness bump, `2.5 S`, in LogC4 code values.
pub const BRIGHTNESS_SIGMA: f64 = 2.5 * LOGC4_STOP;
/// Width and amplitude unit of the Shadows, Highlights, Black and White bumps, `1.5 S`.
pub const ZONE_CONTROL_SIGMA: f64 = 1.5 * LOGC4_STOP;
/// Width of the colour-balance zones, `1.5 S` (a starting value; the spec leaves it open).
pub const ZONE_SIGMA: f64 = 1.5 * LOGC4_STOP;
/// Peak displacement of colour balance at radius 1, in LogC4 code values.
pub const COLOR_BALANCE_STRENGTH: f64 = 0.03;
/// Contrast gain exponent: `2^(0.6 c / 100)`.
pub const CONTRAST_EXPONENT: f64 = 0.6;

/// Stops from middle grey of the Black, Shadows, Highlights and White controls.
pub const ZONE_CENTER_STOPS: [f64; 4] = [-3.0, 3.0, -6.0, 6.0];
/// Stops from middle grey of the colour-balance zones (shadows, midtones, highlights).
pub const BALANCE_CENTER_STOPS: [f64; 3] = [-3.0, 0.0, 3.0];

/// First colour-balance axis `e1 = (1, -1/2, -1/2) / sqrt(1.5)`.
pub const E1: Vec3 = [
    0.816_496_580_927_726_1,
    -0.408_248_290_463_863_1,
    -0.408_248_290_463_863_1,
];
/// Second colour-balance axis `e2 = (0, 1, -1) / sqrt(2)`.
pub const E2: Vec3 = [
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
];

/// A colour-balance control: a point `(a, b)` in the unit disc along the axes `e1`, `e2`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorBalance {
    /// Coordinate along `e1` (the red-versus-cyan axis).
    pub a: f64,
    /// Coordinate along `e2` (the green-versus-blue axis).
    pub b: f64,
}

impl ColorBalance {
    /// The point scaled back into the unit disc (a NaN becomes 0, an infinity points along its axis).
    #[must_use]
    pub fn clamped(self) -> Self {
        // Finite stand-ins keep `hypot` and the division below from producing inf / inf.
        let tame = |x: f64| {
            if x.is_nan() {
                0.0
            } else {
                x.clamp(-1e300, 1e300)
            }
        };
        let (a, b) = (tame(self.a), tame(self.b));
        let r = a.hypot(b);
        if r > 1.0 {
            Self { a: a / r, b: b / r }
        } else {
            Self { a, b }
        }
    }
}

/// The user-facing controls of stage C9. Ranges are the UI's -100..100.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradeParams {
    /// Contrast, -100..100.
    pub contrast: f64,
    /// Brightness, -100..100.
    pub brightness: f64,
    /// Shadows, -100..100 (centre grey -3 stops).
    pub shadows: f64,
    /// Highlights, -100..100 (centre grey +3 stops).
    pub highlights: f64,
    /// Black, -100..100 (centre grey -6 stops).
    pub black: f64,
    /// White, -100..100 (centre grey +6 stops).
    pub white: f64,
    /// Colour balance in the shadows.
    pub balance_shadows: ColorBalance,
    /// Colour balance in the midtones.
    pub balance_midtones: ColorBalance,
    /// Colour balance in the highlights.
    pub balance_highlights: ColorBalance,
    /// Saturation, -100..100.
    pub saturation: f64,
}

impl GradeParams {
    /// Whether every control is at its neutral value (the grade is then the identity).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
}

/// `bump(L; mu, sigma) = exp(-(L - mu)^2 / (2 sigma^2))`.
#[must_use]
pub fn bump(l: f64, mu: f64, sigma: f64) -> f64 {
    let z = (l - mu) / sigma;
    (-0.5 * z * z).exp()
}

/// LogC4 code value of the scene-linear value `0.18 * 2^stops`.
#[must_use]
pub fn logc4_at_stops(stops: f64) -> f64 {
    logc4_encode(MIDDLE_GRAY * stops.exp2())
}

/// The `L` of stage C9 for a linear AWG4 pixel: `encode(Y)` with the AWG4 `Y` row.
#[must_use]
pub fn grade_luma(linear: LinearAwg4) -> f64 {
    crate::logc::logc4_luma_of_linear(linear)
}

fn clamp_control(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-100.0, 100.0)
    }
}

/// Stage C9 with its zone centres precomputed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grade {
    contrast_gain: f64,
    brightness: f64,
    zone_amounts: [f64; 4],
    zone_centers: [f64; 4],
    balance: [ColorBalance; 3],
    balance_centers: [f64; 3],
    saturation_gain: f64,
}

impl Grade {
    /// Precomputes the gains and zone centres for `params`.
    #[must_use]
    pub fn new(params: &GradeParams) -> Self {
        let c = clamp_control(params.contrast);
        let zone_amounts = [
            clamp_control(params.shadows),
            clamp_control(params.highlights),
            clamp_control(params.black),
            clamp_control(params.white),
        ]
        .map(|x| x / 100.0);
        Self {
            contrast_gain: (CONTRAST_EXPONENT * c / 100.0).exp2(),
            brightness: clamp_control(params.brightness) / 100.0,
            zone_amounts,
            zone_centers: ZONE_CENTER_STOPS.map(logc4_at_stops),
            balance: [
                params.balance_shadows.clamped(),
                params.balance_midtones.clamped(),
                params.balance_highlights.clamped(),
            ],
            balance_centers: BALANCE_CENTER_STOPS.map(logc4_at_stops),
            saturation_gain: 1.0 + clamp_control(params.saturation) / 100.0,
        }
    }

    /// Applies C9 to LogC4 `v` for a pixel whose luma is `l` (see [`grade_luma`]).
    #[must_use]
    pub fn apply(&self, v: LogC4, l: f64) -> LogC4 {
        let mut out = v.0;
        // 1. Contrast about grey.
        if self.contrast_gain != 1.0 {
            for c in &mut out {
                *c = LOGC4_GRAY + (*c - LOGC4_GRAY) * self.contrast_gain;
            }
        }
        // 2. Brightness: a midtone lift.
        let lift = self.brightness * LOGC4_STOP * bump(l, LOGC4_GRAY, BRIGHTNESS_SIGMA);
        // 3. Shadows, Highlights, Black, White.
        let mut shift = lift;
        for (amount, center) in self.zone_amounts.iter().zip(&self.zone_centers) {
            if *amount != 0.0 {
                shift += amount * ZONE_CONTROL_SIGMA * bump(l, *center, ZONE_CONTROL_SIGMA);
            }
        }
        for c in &mut out {
            *c += shift;
        }
        // 4. Colour balance in three zones.
        for (bal, center) in self.balance.iter().zip(&self.balance_centers) {
            if bal.a != 0.0 || bal.b != 0.0 {
                let w = COLOR_BALANCE_STRENGTH * bump(l, *center, ZONE_SIGMA);
                for k in 0..3 {
                    out[k] += w * (bal.a * E1[k] + bal.b * E2[k]);
                }
            }
        }
        // 5. Saturation, subtractive: about the brightest channel M, so M and neutrals stay put.
        if self.saturation_gain != 1.0 {
            let m = out[0].max(out[1]).max(out[2]);
            for c in &mut out {
                *c = m + (*c - m) * self.saturation_gain;
            }
        }
        LogC4(out)
    }

    /// Stages C8 and C9 together: encodes the linear pixel to LogC4 and grades it, taking `L` from
    /// the linear values (before C8) as the spec requires.
    #[must_use]
    pub fn apply_linear(&self, linear: LinearAwg4) -> LogC4 {
        self.apply(linear.to_logc4(), grade_luma(linear))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_params_are_exact_identity() {
        let g = Grade::new(&GradeParams::default());
        for v in [[0.1, 0.5, 0.9], [0.2784, 0.2784, 0.2784], [-0.05, 0.3, 1.2]] {
            assert_eq!(g.apply(LogC4(v), 0.4).0, v);
        }
        assert!(GradeParams::default().is_identity());
    }

    #[test]
    fn axes_are_orthonormal_and_sum_to_zero() {
        let dot = |a: Vec3, b: Vec3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!((dot(E1, E1) - 1.0).abs() < 1e-15);
        assert!((dot(E2, E2) - 1.0).abs() < 1e-15);
        assert!(dot(E1, E2).abs() < 1e-15);
        assert!(E1.iter().sum::<f64>().abs() < 1e-15);
        assert!(E2.iter().sum::<f64>().abs() < 1e-15);
        assert!((E1[0] - 1.0 / 1.5_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn zone_centers_are_scene_referred_stops() {
        assert!((logc4_at_stops(0.0) - LOGC4_GRAY).abs() < 1e-15);
        // +6 stops is 11.52 linear.
        assert!((logc4_at_stops(6.0) - logc4_encode(0.18 * 64.0)).abs() < 1e-15);
        assert!(logc4_at_stops(-6.0) > logc4_encode(0.0));
    }
}
