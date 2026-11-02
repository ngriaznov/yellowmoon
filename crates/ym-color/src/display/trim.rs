//! A corrector node's two trims (`docs/algorithms/06-masks.md` section 2, `docs/23-nodes.md`
//! section 1.3): an exposure trim in stops and a temperature and tint trim, both multiplications in
//! linear AWG4 that act first in a node, so the grade after them sees the corrected picture.
//!
//! The two trims make one gain, `g = 2^ev · d(temperature, tint)`, three numbers per node worked out
//! once in f64 ([`Trim::gain`]); the node then does `encode(decode(v) · g)` and no more.
//!
//! # The temperature and tint gain
//!
//! The Input makes the photo's white `[1, 1, 1]` in AWG4, which is the gamut's own white, D65. A node's
//! trim therefore starts from D65 whatever the Input says (`docs/23` section 14, decision 4), so the same
//! node pasted on another photo renders the same change. On the CIE 1960 `(u, v)` diagram:
//!
//! 1. *Temperature* `t` moves the white that the picture is balanced for along the CIE daylight locus
//!    to `m₁ = 1e6/6504 − t/2` mired (`t = ±100` is ∓50 mired): `t > 0` is a bluer light, so the
//!    picture comes out warmer. The step is the displacement of the locus between 6504 K and
//!    `T₁ = 1e6/m₁`, added to D65's own `(u, v)`, so that `t = 0` gives D65 exactly.
//! 2. *Tint* `n` moves it by `0.0001·n` across the locus, towards green (positive Duv), at `T₁`.
//! 3. The gain is the colour of that light seen by a picture balanced for D65, inverted:
//!    `d = R(W₀) / R(W₂)` per channel with `R(W) = M_XYZ→AWG4 · XYZ(W)`, then divided by `d_g` so green
//!    is 1: a von Kries adaptation done in AWG4 itself.

use crate::consts::linalg;
use crate::consts::{D65_XY, Vec3, xy_to_xyz};
use crate::gamut::matrices;

/// Range of the exposure trim, in stops.
pub const TRIM_EXPOSURE_RANGE_EV: (f64, f64) = (-4.0, 4.0);
/// Range of the temperature trim and of the tint trim.
pub const TRIM_RANGE: (f64, f64) = (-100.0, 100.0);
/// Mired per unit of the temperature trim: ±100 is ±50 mired.
pub const TRIM_MIRED_PER_UNIT: f64 = 0.5;
/// `(u, v)` distance per unit of the tint trim: ±100 is ±0.01.
pub const TRIM_DUV_PER_UNIT: f64 = 0.0001;
/// D65's correlated colour temperature, K.
pub const D65_CCT_K: f64 = 6504.0;

/// A node's trims. All zero is the identity and runs no code.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Trim {
    /// Exposure in stops, -4 to 4.
    pub exposure_ev: f64,
    /// Temperature, -100 to 100: positive is warmer.
    pub temperature: f64,
    /// Tint, -100 to 100: positive is more magenta.
    pub tint: f64,
}

impl Trim {
    /// Whether every trim is exactly 0 (the node's first step is then skipped).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.exposure_ev == 0.0 && self.temperature == 0.0 && self.tint == 0.0
    }

    /// The gain in linear AWG4, or `None` for the identity. Values outside their ranges are held
    /// to them and a value that is not a number counts as 0.
    #[must_use]
    pub fn gain(&self) -> Option<Vec3> {
        let held = |x: f64, (lo, hi): (f64, f64)| if x.is_nan() { 0.0 } else { x.clamp(lo, hi) };
        let ev = held(self.exposure_ev, TRIM_EXPOSURE_RANGE_EV);
        let t = held(self.temperature, TRIM_RANGE);
        let n = held(self.tint, TRIM_RANGE);
        if ev == 0.0 && t == 0.0 && n == 0.0 {
            return None;
        }
        let scale = ev.exp2();
        let d = if t == 0.0 && n == 0.0 {
            [1.0; 3]
        } else {
            von_kries(t, n)
        };
        Some(d.map(|c| c * scale))
    }
}

/// The CIE daylight locus: `xy` of the daylight illuminant at `t_k` kelvin (4000 K to 25000 K).
#[must_use]
pub fn daylight_xy(t_k: f64) -> [f64; 2] {
    let x = if t_k <= 7000.0 {
        -4.6070e9 / (t_k * t_k * t_k) + 2.9678e6 / (t_k * t_k) + 0.09911e3 / t_k + 0.244063
    } else {
        -2.0064e9 / (t_k * t_k * t_k) + 1.9018e6 / (t_k * t_k) + 0.24748e3 / t_k + 0.237040
    };
    [x, -3.0 * x * x + 2.870 * x - 0.275]
}

/// CIE 1960 `(u, v)` of a chromaticity `xy`.
fn uv_of_xy(xy: [f64; 2]) -> [f64; 2] {
    let d = -2.0 * xy[0] + 12.0 * xy[1] + 3.0;
    [4.0 * xy[0] / d, 6.0 * xy[1] / d]
}

/// The chromaticity `xy` of CIE 1960 `(u, v)`.
fn xy_of_uv(uv: [f64; 2]) -> [f64; 2] {
    let d = 2.0 * uv[0] - 8.0 * uv[1] + 4.0;
    [3.0 * uv[0] / d, 2.0 * uv[1] / d]
}

fn uv_daylight(t_k: f64) -> [f64; 2] {
    uv_of_xy(daylight_xy(t_k))
}

/// The `(u, v)` of the white a node's trim moves the picture's white to (section 2.2 of the maths
/// file): D65 moved along the daylight locus by `temperature` and across it by `tint`.
#[must_use]
pub fn trim_white_uv(temperature: f64, tint: f64) -> [f64; 2] {
    let w0 = uv_of_xy(D65_XY);
    let m1 = 1e6 / D65_CCT_K - TRIM_MIRED_PER_UNIT * temperature;
    let t1 = 1e6 / m1;
    let (u0, u1) = (uv_daylight(D65_CCT_K), uv_daylight(t1));
    let mut w = [w0[0] + (u1[0] - u0[0]), w0[1] + (u1[1] - u0[1])];
    if tint != 0.0 {
        // The unit normal of the locus at T₁, from the polyline at ±1 mired, towards green (+v).
        let (a, b) = (uv_daylight(1e6 / (m1 - 1.0)), uv_daylight(1e6 / (m1 + 1.0)));
        let tangent = [b[0] - a[0], b[1] - a[1]];
        let len = tangent[0].hypot(tangent[1]);
        let normal = if tangent[0] > 0.0 {
            [-tangent[1] / len, tangent[0] / len]
        } else {
            [tangent[1] / len, -tangent[0] / len]
        };
        w[0] += TRIM_DUV_PER_UNIT * tint * normal[0];
        w[1] += TRIM_DUV_PER_UNIT * tint * normal[1];
    }
    w
}

/// AWG4 of a white `(u, v)` at `Y = 1`.
fn awg4_of_uv(uv: [f64; 2]) -> Vec3 {
    linalg::mul_vec(&matrices().xyz_to_awg4, xy_to_xyz(xy_of_uv(uv)))
}

/// The von Kries gain of a temperature and a tint: green 1.
#[must_use]
pub fn von_kries(temperature: f64, tint: f64) -> Vec3 {
    let r0 = awg4_of_uv(uv_of_xy(D65_XY));
    let r2 = awg4_of_uv(trim_white_uv(temperature, tint));
    let d = [r0[0] / r2[0], r0[1] / r2[1], r0[2] / r2[2]];
    let green = d[1];
    d.map(|c| c / green)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_the_identity() {
        assert!(Trim::default().is_identity());
        assert_eq!(Trim::default().gain(), None);
        let only_tint = Trim {
            tint: 1.0,
            ..Trim::default()
        };
        assert!(!only_tint.is_identity());
        assert!(only_tint.gain().is_some());
    }

    #[test]
    fn an_exposure_trim_is_two_to_the_ev() {
        let g = Trim {
            exposure_ev: 1.0,
            ..Trim::default()
        }
        .gain();
        assert_eq!(g, Some([2.0, 2.0, 2.0]));
    }

    #[test]
    fn green_is_one_and_warmer_raises_red() {
        let warm = von_kries(100.0, 0.0);
        assert_eq!(warm[1], 1.0);
        assert!(warm[0] > 1.0 && warm[2] < 1.0, "{warm:?}");
        let cool = von_kries(-100.0, 0.0);
        assert!(cool[0] < 1.0 && cool[2] > 1.0, "{cool:?}");
    }
}
