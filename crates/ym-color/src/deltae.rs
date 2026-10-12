//! CIEDE2000 and the other colour-difference metrics used by tests and gates (`docs/02-colour-pipeline.md` section Tests).
//!
//! * [`xyz_to_lab`] and [`lab_to_xyz`]: CIE 1931 XYZ and CIE L*a*b* for an explicit white, with the
//!   exact rational constants `epsilon = 216/24389` and `kappa = 24389/27`.
//! * [`delta_e_2000`]: CIEDE2000 (Sharma, Wu and Dalal, 2005), validated against all 34 published
//!   test pairs to 1e-4 in `tests/deltae.rs`.
//! * [`delta_e_76`]: Euclidean distance in L*a*b*.
//! * [`Summary`]: median, 95th percentile, mean and maximum of a list of differences, the way the
//!   gates in `docs/13-testing.md` report them.

use crate::consts::{ACES_WHITE_XY, D65_XY, Vec3, xy_to_xyz};

/// CIE epsilon `216/24389` (exact).
const EPSILON: f64 = 216.0 / 24389.0;
/// CIE kappa `24389/27` (exact).
const KAPPA: f64 = 24389.0 / 27.0;

/// The D65 white as XYZ at `Y = 1`.
#[must_use]
pub fn d65_white_xyz() -> Vec3 {
    xy_to_xyz(D65_XY)
}

/// The ACES white as XYZ at `Y = 1`.
#[must_use]
pub fn aces_white_xyz() -> Vec3 {
    xy_to_xyz(ACES_WHITE_XY)
}

fn lab_f(t: f64) -> f64 {
    if t > EPSILON {
        t.cbrt()
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}

fn lab_f_inv(f: f64) -> f64 {
    let cubed = f * f * f;
    if cubed > EPSILON {
        cubed
    } else {
        (116.0 * f - 16.0) / KAPPA
    }
}

/// CIE XYZ to CIE L*a*b* relative to `white` (XYZ of the reference white).
#[must_use]
pub fn xyz_to_lab(xyz: Vec3, white: Vec3) -> Vec3 {
    let fx = lab_f(xyz[0] / white[0]);
    let fy = lab_f(xyz[1] / white[1]);
    let fz = lab_f(xyz[2] / white[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIE L*a*b* to CIE XYZ relative to `white`.
#[must_use]
pub fn lab_to_xyz(lab: Vec3, white: Vec3) -> Vec3 {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [
        lab_f_inv(fx) * white[0],
        lab_f_inv(fy) * white[1],
        lab_f_inv(fz) * white[2],
    ]
}

/// CIE 1976 colour difference: Euclidean distance in L*a*b*.
#[must_use]
pub fn delta_e_76(lab1: Vec3, lab2: Vec3) -> f64 {
    let d = [lab1[0] - lab2[0], lab1[1] - lab2[1], lab1[2] - lab2[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Hue angle in degrees in `[0, 360)`; zero when both components are zero.
fn hue_deg(b: f64, a: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        return 0.0;
    }
    let h = b.atan2(a).to_degrees();
    if h < 0.0 { h + 360.0 } else { h }
}

/// Tolerance in degrees for the `|h1 - h2| <= 180` tests: the published tie-break for a difference
/// of exactly 180 degrees must not depend on the last bit of `atan2` (Sharma pair 14).
const HUE_TIE_DEG: f64 = 1e-9;

/// CIEDE2000 colour difference of two L*a*b* colours with `kL = kC = kH = 1`.
#[must_use]
pub fn delta_e_2000(lab1: Vec3, lab2: Vec3) -> f64 {
    delta_e_2000_weighted(lab1, lab2, 1.0, 1.0, 1.0)
}

/// CIEDE2000 with explicit parametric weights `kL`, `kC` and `kH`.
#[must_use]
pub fn delta_e_2000_weighted(lab1: Vec3, lab2: Vec3, k_l: f64, k_c: f64, k_h: f64) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let c_bar = 0.5 * (c1 + c2);
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);
    let h1p = hue_deg(b1, a1p);
    let h2p = hue_deg(b2, a2p);

    let dl = l2 - l1;
    let dc = c2p - c1p;
    let chroma_product = c1p * c2p;
    let dhp = if chroma_product == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d.abs() <= 180.0 + HUE_TIE_DEG {
            d
        } else if d > 180.0 {
            d - 360.0
        } else {
            d + 360.0
        }
    };
    let dh = 2.0 * chroma_product.sqrt() * (0.5 * dhp.to_radians()).sin();

    let l_bar = 0.5 * (l1 + l2);
    let c_bar_p = 0.5 * (c1p + c2p);
    let h_bar_p = if chroma_product == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 + HUE_TIE_DEG {
        0.5 * (h1p + h2p)
    } else if h1p + h2p < 360.0 {
        0.5 * (h1p + h2p + 360.0)
    } else {
        0.5 * (h1p + h2p - 360.0)
    };

    let t = 1.0 - 0.17 * (h_bar_p - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar_p).to_radians().cos()
        + 0.32 * (3.0 * h_bar_p + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar_p - 63.0).to_radians().cos();
    let d_theta = 30.0 * (-((h_bar_p - 275.0) / 25.0).powi(2)).exp();
    let c_bar_p7 = c_bar_p.powi(7);
    let r_c = 2.0 * (c_bar_p7 / (c_bar_p7 + 25.0_f64.powi(7))).sqrt();
    let l50 = (l_bar - 50.0).powi(2);
    let s_l = 1.0 + 0.015 * l50 / (20.0 + l50).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;
    let r_t = -(2.0 * d_theta).to_radians().sin() * r_c;

    let tl = dl / (k_l * s_l);
    let tc = dc / (k_c * s_c);
    let th = dh / (k_h * s_h);
    (tl * tl + tc * tc + th * th + r_t * tc * th)
        .max(0.0)
        .sqrt()
}

/// Median, 95th percentile, mean and maximum of a set of colour differences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Number of values summarised.
    pub count: usize,
    /// Median (linear interpolation between the two middle values for an even count).
    pub median: f64,
    /// 95th percentile (linear interpolation between order statistics, as numpy's default).
    pub p95: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Maximum.
    pub max: f64,
}

/// Percentile `q` in `[0, 100]` of `values` by linear interpolation between order statistics
/// (numpy's default method). Returns `None` for an empty slice or any NaN.
#[must_use]
pub fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 100.0) / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64))
}

impl Summary {
    /// Summarises `values`, or `None` when empty or containing NaN.
    #[must_use]
    pub fn of(values: &[f64]) -> Option<Self> {
        let median = percentile(values, 50.0)?;
        let p95 = percentile(values, 95.0)?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self {
            count: values.len(),
            median,
            p95,
            mean,
            max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_colors_have_zero_difference() {
        let lab = [53.2, 80.1, 67.2];
        assert_eq!(delta_e_2000(lab, lab), 0.0);
        assert_eq!(delta_e_76(lab, lab), 0.0);
    }

    #[test]
    fn white_is_lab_100_0_0() {
        let w = d65_white_xyz();
        let lab = xyz_to_lab(w, w);
        assert!((lab[0] - 100.0).abs() < 1e-12 && lab[1].abs() < 1e-12 && lab[2].abs() < 1e-12);
    }

    #[test]
    fn summary_of_known_values() {
        let s = Summary::of(&[1.0, 2.0, 3.0, 4.0, 100.0]).unwrap();
        assert_eq!(s.median, 3.0);
        assert_eq!(s.max, 100.0);
        assert!((s.p95 - 80.8).abs() < 1e-12);
        assert!(Summary::of(&[]).is_none());
    }
}
