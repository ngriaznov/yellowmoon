//! Primary grade in LogC4: contrast, brightness, zones, colour balance, saturation (stage C9) (`docs/02-colour-pipeline.md` section C9).
//!
//! `v` is the LogC4 RGB triple. `L = encode(Y_lin)` is the LogC4 code value of the luminance
//! (the AWG4 `Y` row) of the **linear** values before C8, so it is a property of the pixel that
//! the grade does not change. Grey is `g = encode(0.18) = 0.27840`; one stop in LogC4 is
//! `S = b / 14 = 0.06480` code values. The controls are applied in this order:
//!
//! | Control | Range | Operation |
//! |---|---|---|
//! | Contrast `c` | -100..100 | `u = v + T_k(L) - L`, the tone curve [`contrast_curve`] with slope `k = 2^(0.6 c / 100)` at grey, then `u`'s chroma moved part of the way toward that of `T_k` on each channel ([`contrast_pixel`]) |
//! | Brightness `b` | -100..100 | `v += (b / 100) S bump(L; g, 2.5 S)` |
//! | Shadows, Highlights, Black, White | -100..100 | `v += (x / 100) 1.5 S bump(L; centre, 1.5 S)`, centres at grey -3, +3, -6, +6 stops |
//! | Colour balance (shadows, midtones, highlights) | `(a, b)` in the unit disc | `v += 0.03 (a e1 + b e2) zone(L)` |
//! | Saturation `s` | -100..100 | in linear AWG4, the chroma `rgb - Y` scaled about the luminance `Y` ([`saturate`]) |
//!
//! with `bump(L; mu, sigma) = exp(-(L - mu)^2 / (2 sigma^2))`, `e1 = (1, -1/2, -1/2) / sqrt(1.5)` and
//! `e2 = (0, 1, -1) / sqrt(2)`. All constants are starting values (`docs/02`).
//!
//! # Contrast (the design of 2026-10-19)
//!
//! A tone curve `T` on the pixel's luma `L`, fixed at three points: black `a` (the code of linear 0,
//! [`CONTRAST_BLACK`]), grey `g` (the pivot) and the top code value `b = 1` ([`CONTRAST_TOP`]). With
//! `h_k(t) = k t / (1 + (k - 1) t)`, which runs from `h(0) = 0` to `h(1) = 1` with the slope `k` at 0
//! and `1 / k` at 1 and is increasing for every `k > 0`:
//!
//! * `g <= x <= b`: `T(x) = g + (b - g) h_k((x - g) / (b - g))`;
//! * `a <= x < g`: `T(x) = g - (g - a) h_k((g - x) / (g - a))`;
//! * above `b` and below `a`: the straight line through the end with the slope `1 / k`.
//!
//! `T` is continuous, has a continuous slope (`k` at grey from both sides, `1 / k` at both ends) and
//! is increasing for all `x`. The tonal change is one common offset `T(L) - L` added to all three
//! channels, `u = v + (T(L) - L)`: on LogC4's log segment a common offset multiplies
//! `x + 64 / a_LogC4` (`x + 0.029`) by a common gain, which is a common gain in linear light wherever
//! `x` is well above 0.029, so the channel ratios, and with them the hue, are kept, while the luma
//! follows `T`. The slope at grey is the one the old control had (`2^(0.6 c / 100)` about grey), so
//! the slider feels the same around mid grey; unlike the old control, black and the top code value
//! stay where they are, and the slope toward them flattens (`c > 0`) or steepens (`c < 0`) instead
//! of the whole range stretching about grey. `c = 0` skips the step (bit-exact identity). The rows
//! after Contrast keep using the original `L`.
//!
//! **The colour step** (the owner's answer of 2026-10-19: contrast adds part of the colour a curve on
//! each channel would). `u` keeps the chroma of `v`, where the same curve on each channel,
//! `p = (T(v_r), T(v_g), T(v_b))`, would spread the channels apart about grey (`c > 0`) or draw them
//! together. With the chroma of a LogC4 triple in the colour-balance plane, `c_x = (e1 . x, e2 . x)`:
//!
//! * `f = 1 + beta (|c_p| / |c_u| - 1)` with `beta =` [`CONTRAST_COLOR`] (0.5);
//! * `v' = u + (f - 1) (c_u1 e1 + c_u2 e2)`: `u`'s chroma scaled by `f`, its direction (the hue in the
//!   plane) kept;
//! * `f = 1` (`v' = u`) where `|c_u|` is below [`CONTRAST_NEUTRAL_CHROMA`]: neutrals stay neutral.
//!
//! `e1` and `e2` sum to 0, so the mean of the three channels (the position along the neutral axis) is
//! `u`'s. `T`'s slope lies between `min(k, 1 / k)` and `max(k, 1 / k)`, so (mean value theorem on each
//! channel difference) `|c_p| / |c_u|` does too; the ratio is limited to that interval, which only
//! guards rounding, and `f` lies between `1 - beta (1 - min)` and `1 + beta (max - 1)`: 0.83 to 1.26
//! at the ends of the range. `beta = 0` is the hue-keeping offset alone, `beta = 1` gives `u` the
//! chroma length of the per-channel curve.
//!
//! # Saturation (the design of 2026-10-19, confirmed by the owner the same day)
//!
//! It replaces the subtractive saturation of 2026-10-13 (about the brightest channel, which made
//! colours denser and darker). In linear AWG4: `rgb = decode(v)` and `Y` its luminance (the AWG4
//! `Y` row, the weights of `L`). The chroma `rgb - Y` is scaled about `Y` by a factor `lambda`,
//! `rgb' = Y + lambda (rgb - Y)`, and the result is encoded to LogC4 again. As the `Y` row sums to
//! 1, `Y` is unchanged: more saturation does not make a colour darker or lighter.
//!
//! * `s < 0`: `lambda = 1 + s / 100`, an even blend toward grey (`-100` is the grey of the same `Y`).
//! * `s > 0`: `F = 1 + SATURATION_BOOST s / 100` ([`SATURATION_BOOST`] 1.5, measured so that +60 adds
//!   about as much colour to the owner's photos as the old control did). With
//!   `r = (Y - min(rgb)) / Y` (0 for a neutral, 1 when the darkest channel is at 0: scaling the
//!   chroma by `1 / r` would take that channel to zero), `r' = 1 - (1 - r)^F` and `lambda = r' / r`. A muted colour (small `r`) has its chroma
//!   scaled by nearly `F`, a strong one (r near 1) by nearly 1, and the darkest channel ends at
//!   `Y (1 - r') >= 0`: positive saturation never pushes a channel below zero. A pixel with `Y <= 0`
//!   or a channel at or below 0 (`r >= 1`) is left as it is.
//!
//! `s = 0` skips the step (bit-exact identity). The ASC CDL saturation inside looks (C10, Rec.709
//! weights) is a standard and is not changed.
//!
//! # Decisions where the spec is silent
//!
//! * **Stops are scene-referred.** "Grey -3 stops" is the LogC4 code value of `0.18 * 2^-3` (and so
//!   on), not `g - 3 S`: LogC4 is only linear in stops far above the toe, and `g - 6 S` would fall
//!   below the code value of black. The widths (`1.5 S`, `2.5 S`) are code-value distances.
//! * **Zone shape.** `zone(L)` is `bump(L; centre, ZONE_SIGMA)` with `ZONE_SIGMA = 1.5 S`, centred at
//!   -3, 0 and +3 stops. The zones are Gaussian bumps, not a partition of unity, so colour balance
//!   leaves the extremes of the range alone.
//! * **Saturation where `r` is small or negative.** Below [`SATURATION_SMALL_R`] (1e-9), `lambda` is
//!   its limit `F`. That covers `r < 0` too: the blue weight of the AWG4 `Y` row is negative, so a
//!   blue or violet pixel whose red and green are nearly equal can have every channel above `Y`. No
//!   channel can then reach zero as the chroma grows, and the formula `r' / r` would exceed `F`
//!   without bound as `Y` goes to 0 (at `s = 100` it would scale the chroma of the linear blue
//!   `(0.02, 0.02, 0.57)`, `Y = 0.0002`, by 101); such pixels get the plain factor `F`. `r'` is computed as `-expm1(F ln(1 - r))`, the same number without the
//!   cancellation of `1 - (1 - r)^F` when `r` is small.
//! * **Saturation of absurd codes.** A pixel with a code above [`SATURATION_CODE_LIMIT`] (7.5, linear
//!   about 7e32, 112 stops over grey; no photo comes near it) is passed on unchanged, so that the f32
//!   kernel's decode and encode stay finite; real pixels after C9 stay below 3.
//! * Nothing in C9 clamps: Colour balance and the zone controls can take a channel below the LogC4
//!   code of black, and C8's inverse, C10 and the DRT take it as it is.
//! * **Sign.** For Shadows, Highlights, Black and White a positive value raises the pixel (brighter),
//!   a negative value lowers it. Positive brightness brightens.
//! * Out-of-range control values are clamped to their range; a colour-balance vector longer than 1
//!   is scaled back to the unit circle.
//!
//! # Monotonicity (what the tests check)
//!
//! * *Per pixel, in the control value:* for a luma between black and the top code value, Contrast
//!   moves the mean of the three channels (and every channel of a neutral) away from grey more the
//!   higher the value (up for `L > g`, down for `L < g`; `h_k(t)` grows with `k` for `0 < t < 1`);
//!   beyond the ends the line of slope `1 / k` pulls the other way. The colour step scales a
//!   coloured pixel's chroma by `f` (between 0.83 and 1.26) without moving that mean, so a single
//!   channel of a coloured pixel is not promised to move with the mean. Brightness, Shadows, Highlights, Black and White add a non-negative bump times the
//!   control, so every output channel is non-decreasing in the control. Saturation scales the
//!   chroma about `Y` by a factor that is non-decreasing in `s` (`(1 - r)^F` falls as `F` grows).
//!   Colour balance displaces `v` by `0.03 r zone(L)` along a fixed direction, so the displacement
//!   length is non-decreasing in the radius `r`.
//! * *As a tone map on a neutral ramp:* with one control moved over its full range, the output for
//!   neutral input is strictly increasing in the input (the slope stays above 0.39; for Contrast it
//!   is between `2^-0.6 = 0.66` and `2^0.6 = 1.52`).
//! * *Not guaranteed:* several tonal controls stacked at their extremes (for example Shadows, Black
//!   and Brightness all at -100 with Contrast at -100) can fold the neutral ramp, because each bump
//!   contributes a slope of up to 0.4 to 0.6 and the slopes add. `tests/grade.rs` prints the worst
//!   slopes. A UI that allows the full stack should limit or re-fit the combination.

use crate::consts::{LOGC4, LOGC4_GRAY, LOGC4_STOP, MIDDLE_GRAY, Vec3};
use crate::gamut::{LinearAwg4, awg4_luminance};
use crate::logc::{LogC4, logc4_decode_rgb, logc4_encode, logc4_encode_rgb};
use crate::warp::chroma_of;

/// Width of the brightness bump, `2.5 S`, in LogC4 code values.
pub const BRIGHTNESS_SIGMA: f64 = 2.5 * LOGC4_STOP;
/// Width and amplitude unit of the Shadows, Highlights, Black and White bumps, `1.5 S`.
pub const ZONE_CONTROL_SIGMA: f64 = 1.5 * LOGC4_STOP;
/// Width of the colour-balance zones, `1.5 S` (a starting value; the spec leaves it open).
pub const ZONE_SIGMA: f64 = 1.5 * LOGC4_STOP;
/// Peak displacement of colour balance at radius 1, in LogC4 code values.
pub const COLOR_BALANCE_STRENGTH: f64 = 0.03;
/// Contrast slope exponent: the slope of the tone curve at grey is `k = 2^(0.6 c / 100)`.
pub const CONTRAST_EXPONENT: f64 = 0.6;
/// The lower fixed point of the contrast curve: the LogC4 code of linear 0, `encode(0) = c =
/// 0.09286` (the constant `c` of LogC4, as `log2(64) - 6 = 0`).
pub const CONTRAST_BLACK: f64 = LOGC4.c;
/// The upper fixed point of the contrast curve: the top code value 1.0 (linear 469.8, about 11.35
/// stops over grey).
pub const CONTRAST_TOP: f64 = 1.0;
/// The colour step of Contrast, `beta`: the fraction of the way from the hue-keeping result's chroma
/// length `|c_u|` to the length `|c_p|` the same curve on each channel would give (owner's answer of
/// 2026-10-19; a tunable starting value). 0 is the hue-keeping offset alone, 1 the per-channel
/// curve's chroma length; the direction of the chroma is kept at every value.
pub const CONTRAST_COLOR: f64 = 0.5;
/// Below this chroma length `|c_u|` (LogC4 code values in the C9 plane) Contrast leaves a pixel at
/// `u`, its colour step off: neutrals stay neutral, and nothing divides by a vanishing length.
pub const CONTRAST_NEUTRAL_CHROMA: f64 = 1e-9;
/// How strongly positive saturation scales chroma: `F = 1 + SATURATION_BOOST s / 100` (`F = 2.5` at
/// `s = 100`). Set so that Saturation +60 raises the mean CIELAB chroma of the owner's three photos by
/// about as much as the subtractive control of 0.4.2 did (+73.6 % against +72.9 %; owner's answer of
/// 2026-10-19, the sweep in `docs/02` C9). Tunable.
pub const SATURATION_BOOST: f64 = 1.5;
/// Below this distance `r` of the darkest channel under the luminance (as a fraction of the
/// luminance), positive saturation uses its limit, the full factor `F` (see [`saturation_chroma_scale`]).
pub const SATURATION_SMALL_R: f64 = 1e-9;
/// Saturation passes on a pixel with a code value above this unchanged (linear about 7e32, 112 stops
/// over grey), so the f32 kernel's decode and encode stay finite. No photo comes near it.
pub const SATURATION_CODE_LIMIT: f64 = 7.5;

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

/// The slope at grey of the contrast curve for the control value `contrast` (clamped to
/// -100..100): `k = 2^(0.6 c / 100)`, 1 at 0, `2^0.6 = 1.52` at 100 and `2^-0.6 = 0.66` at -100.
#[must_use]
pub fn contrast_slope(contrast: f64) -> f64 {
    (CONTRAST_EXPONENT * clamp_control(contrast) / 100.0).exp2()
}

/// `h_k(t) = k t / (1 + (k - 1) t)`: `h(0) = 0`, `h(1) = 1`, slope `k` at 0 and `1 / k` at 1, increasing
/// on `[0, 1]` for every `k > 0` (the denominator runs from 1 to `k`).
fn contrast_h(t: f64, k: f64) -> f64 {
    k * t / (1.0 + (k - 1.0) * t)
}

/// The contrast tone curve `T_k` of stage C9 on a LogC4 code `x`, for the slope `k > 0` at grey
/// ([`contrast_slope`]). It fixes black ([`CONTRAST_BLACK`]), grey and the top code value
/// ([`CONTRAST_TOP`]); between them it is `h_k` scaled into each half (`g + (b - g) h_k((x - g) /
/// (b - g))` above grey, `g - (g - a) h_k((g - x) / (g - a))` below), and beyond them the straight
/// line of slope `1 / k` through the end. Continuous, with a continuous slope (`k` at grey, `1 / k`
/// at both ends), increasing for all `x`; `T(a) = a`, `T(g) = g` and `T(b) = b` exactly.
#[must_use]
pub fn contrast_curve(x: f64, k: f64) -> f64 {
    let (a, g, b) = (CONTRAST_BLACK, LOGC4_GRAY, CONTRAST_TOP);
    if x >= b {
        b + (x - b) / k
    } else if x >= g {
        g + (b - g) * contrast_h((x - g) / (b - g), k)
    } else if x > a {
        g - (g - a) * contrast_h((g - x) / (g - a), k)
    } else {
        a + (x - a) / k
    }
}

/// Contrast of stage C9 on the LogC4 pixel `v` whose luma is `l`, for the slope `k > 0` at grey
/// ([`contrast_slope`]) and the colour fraction `color` (`beta`; the grade uses [`CONTRAST_COLOR`]).
///
/// `u = v + (T_k(l) - l)` on all three channels ([`contrast_curve`]); then, unless `u`'s chroma
/// `c_u = (e1 . u, e2 . u)` is shorter than [`CONTRAST_NEUTRAL_CHROMA`], `u`'s chroma is scaled by
/// `f = 1 + color (|c_p| / |c_u| - 1)`, where `c_p` is the chroma of `T_k` on each channel of `v`:
/// `u + (f - 1) (c_u1 e1 + c_u2 e2)`. The ratio is limited to `[min(k, 1 / k), max(k, 1 / k)]`, the
/// range of `T_k`'s slope, which it lies in exactly (this only guards rounding). Keeps the mean of
/// the channels (`u`'s) and the direction of the chroma; `color = 0` is `u`.
#[must_use]
pub fn contrast_pixel(v: Vec3, l: f64, k: f64, color: f64) -> Vec3 {
    let offset = contrast_curve(l, k) - l;
    let u = v.map(|x| x + offset);
    // `u`'s chroma is `v`'s (e1 and e2 sum to 0); taken from `v`, it carries no rounding of the offset.
    let [cu1, cu2] = chroma_of(LogC4(v));
    let length_u = cu1.hypot(cu2);
    if length_u < CONTRAST_NEUTRAL_CHROMA {
        return u;
    }
    let [cp1, cp2] = chroma_of(LogC4(v.map(|x| contrast_curve(x, k))));
    // `max` then `min`, not `clamp`: no panic whatever `k` is.
    let ratio = (cp1.hypot(cp2) / length_u)
        .max(k.min(1.0 / k))
        .min(k.max(1.0 / k));
    let g = color * (ratio - 1.0);
    let mut out = u;
    for (i, o) in out.iter_mut().enumerate() {
        *o += g * (cu1 * E1[i] + cu2 * E2[i]);
    }
    out
}

/// The factor `lambda` by which positive saturation scales the chroma of a pixel whose darkest
/// channel lies the fraction `r = (Y - min) / Y` of its luminance below the luminance, for
/// `F = 1 + SATURATION_BOOST s / 100`: `r' / r` with `r' = 1 - (1 - r)^F`, computed as
/// `-expm1(F ln(1 - r)) / r`. It falls from `F` (a muted colour, `r` near 0) to 1 (`r` near 1, a
/// channel near 0). Below [`SATURATION_SMALL_R`], negative `r` included, it is the limit `F`; at
/// `r >= 1` (a channel at or below 0) it is 1, as the pixel is left.
#[must_use]
pub fn saturation_chroma_scale(r: f64, f: f64) -> f64 {
    if r < SATURATION_SMALL_R {
        f
    } else if r >= 1.0 {
        1.0
    } else {
        -(f * (-r).ln_1p()).exp_m1() / r
    }
}

/// The saturation factor for the control value `saturation` (clamped to -100..100): `1 + s / 100`
/// for `s < 0` (the blend toward grey), `F = 1 + SATURATION_BOOST s / 100` for `s > 0`, and 1 (off)
/// at 0. A factor below 1 is a blend, above 1 a boost; the GPU kernel reads it the same way.
#[must_use]
pub fn saturation_factor(saturation: f64) -> f64 {
    let s = clamp_control(saturation) / 100.0;
    if s < 0.0 {
        1.0 + s
    } else {
        1.0 + SATURATION_BOOST * s
    }
}

/// The saturation of stage C9 on a linear AWG4 triple for a prepared factor (see
/// [`saturation_factor`]), or `None` where the pixel is left as it is (`s > 0` with `Y <= 0` or a
/// channel at or below 0).
fn saturate_rgb(rgb: Vec3, factor: f64) -> Option<Vec3> {
    let y = awg4_luminance(rgb);
    let lambda = if factor < 1.0 {
        factor
    } else {
        if y <= 0.0 || y.is_nan() {
            return None;
        }
        let m = rgb[0].min(rgb[1]).min(rgb[2]);
        let r = (y - m) / y;
        if r >= 1.0 {
            return None;
        }
        saturation_chroma_scale(r, factor)
    };
    Some(rgb.map(|c| y + lambda * (c - y)))
}

/// Saturation `s` (-100..100, clamped) of stage C9 on a linear AWG4 pixel: its chroma `rgb - Y` scaled
/// about its luminance `Y`, by `1 + s / 100` below 0 and by [`saturation_chroma_scale`] above. `Y` is
/// kept, so are neutrals and the direction of the chroma (the hue); for `s > 0` no channel goes below
/// 0, and a pixel with `Y <= 0` or a channel at or below 0 is returned unchanged.
#[must_use]
pub fn saturate(rgb: LinearAwg4, saturation: f64) -> LinearAwg4 {
    let factor = saturation_factor(saturation);
    if factor == 1.0 {
        return rgb;
    }
    LinearAwg4(saturate_rgb(rgb.0, factor).unwrap_or(rgb.0))
}

/// Stage C9 with its zone centres precomputed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grade {
    contrast_slope: f64,
    brightness: f64,
    zone_amounts: [f64; 4],
    zone_centers: [f64; 4],
    balance: [ColorBalance; 3],
    balance_centers: [f64; 3],
    saturation_factor: f64,
}

impl Grade {
    /// Precomputes the slopes, factors and zone centres for `params`.
    #[must_use]
    pub fn new(params: &GradeParams) -> Self {
        let zone_amounts = [
            clamp_control(params.shadows),
            clamp_control(params.highlights),
            clamp_control(params.black),
            clamp_control(params.white),
        ]
        .map(|x| x / 100.0);
        Self {
            contrast_slope: contrast_slope(params.contrast),
            brightness: clamp_control(params.brightness) / 100.0,
            zone_amounts,
            zone_centers: ZONE_CENTER_STOPS.map(logc4_at_stops),
            balance: [
                params.balance_shadows.clamped(),
                params.balance_midtones.clamped(),
                params.balance_highlights.clamped(),
            ],
            balance_centers: BALANCE_CENTER_STOPS.map(logc4_at_stops),
            saturation_factor: saturation_factor(params.saturation),
        }
    }

    /// Applies C9 to LogC4 `v` for a pixel whose luma is `l` (see [`grade_luma`]).
    #[must_use]
    pub fn apply(&self, v: LogC4, l: f64) -> LogC4 {
        let mut out = v.0;
        // 1. Contrast: the tone curve on the luma, as one offset for all three channels, then part of
        //    the colour the same curve on each channel would add.
        if self.contrast_slope != 1.0 {
            out = contrast_pixel(out, l, self.contrast_slope, CONTRAST_COLOR);
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
        // 5. Saturation: the chroma scaled about the luminance, in linear AWG4.
        if self.saturation_factor != 1.0
            && out.iter().all(|&c| c <= SATURATION_CODE_LIMIT)
            && let Some(rgb) = saturate_rgb(logc4_decode_rgb(out), self.saturation_factor)
        {
            out = logc4_encode_rgb(rgb);
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
