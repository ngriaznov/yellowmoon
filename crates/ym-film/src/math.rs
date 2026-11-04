//! Small numeric helpers: the error function, Planck's law, a 3 x 3 solve.
//!
//! Written from the standard series and continued fraction of the error function (Abramowitz and Stegun,
//! *Handbook of Mathematical Functions*, 7.1.5 and 7.1.14) and from Planck's law with the second radiation
//! constant `c2 = 1.438776877e-2 m K` (CODATA 2018). Only `+ - * /`, `exp`, `sqrt`, `ln` and `powf` are used,
//! never `mul_add`.

use crate::{N_LAMBDA, Spectrum, lambda_nm};

/// The error function, accurate to about 1e-13.
#[must_use]
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let v = if ax < 2.5 {
        // Maclaurin series: 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1)).
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        for n in 1..200 {
            let nf = f64::from(n);
            term *= -x2 / nf;
            let add = term / (2.0 * nf + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        1.0 - erfc_large(ax)
    };
    if x < 0.0 { -v } else { v }
}

/// The complementary error function.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    if x >= 2.5 {
        erfc_large(x)
    } else {
        1.0 - erf(x)
    }
}

/// `erfc(x)` for `x >= 2.5` by the continued fraction
/// `exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`.
fn erfc_large(x: f64) -> f64 {
    if x > 27.0 {
        return 0.0;
    }
    let mut f = x;
    for k in (1..80).rev() {
        f = x + (f64::from(k) * 0.5) / f;
    }
    (-x * x).exp() / (std::f64::consts::PI.sqrt() * f)
}

/// A smooth step from 1 (far below `centre`) to 0 (far above), `0.5 * erfc((x - centre) / width)`.
#[must_use]
pub fn falling_edge(x: f64, centre: f64, width: f64) -> f64 {
    0.5 * erfc((x - centre) / width)
}

/// A smooth step from 0 (far below `centre`) to 1 (far above).
#[must_use]
pub fn rising_edge(x: f64, centre: f64, width: f64) -> f64 {
    0.5 * erfc((centre - x) / width)
}

/// The spectral radiance of a black body at `temperature_k`, on the 380 to 780 nm grid, scaled so that its
/// largest sample is 1.
#[must_use]
pub fn blackbody(temperature_k: f64) -> Spectrum {
    const C2: f64 = 1.438_776_877e-2;
    let mut out = [0.0; N_LAMBDA];
    for (i, o) in out.iter_mut().enumerate() {
        let lam = lambda_nm(i) * 1e-9;
        *o = lam.powi(-5) / ((C2 / (lam * temperature_k)).exp() - 1.0);
    }
    let peak = out.iter().copied().fold(0.0, f64::max);
    for o in &mut out {
        *o /= peak;
    }
    out
}

/// `10^x`, written as `exp(x ln 10)` so every platform takes the same path.
#[must_use]
pub fn pow10(x: f64) -> f64 {
    (x * std::f64::consts::LN_10).exp()
}

/// Solves the 2 x 2 system `a x = b`; `None` when singular.
#[must_use]
pub fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-300 || !det.is_finite() {
        return None;
    }
    Some([
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ])
}

/// Solves the 3 x 3 system `a x = b` by Cramer's rule; `None` when singular.
#[must_use]
pub fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-300 || !d.is_finite() {
        return None;
    }
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        *o = det(&m) / d;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-13);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-13);
        assert!((erf(2.0) - 0.995_322_265_018_952_7).abs() < 1e-13);
        assert!((erf(3.0) - 0.999_977_909_503_001_4).abs() < 1e-13);
        assert!((erfc(4.0) - 1.541_725_790_028_002e-8).abs() < 1e-20);
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-15);
    }

    #[test]
    fn planck_peaks_where_wien_says() {
        let s = blackbody(3400.0);
        let peak = s.iter().position(|v| *v == 1.0).unwrap_or(0);
        // Wien: 2.898e-3 / 3400 = 852 nm, so on the grid the radiance rises to 780 nm.
        assert_eq!(peak, N_LAMBDA - 1);
        let s = blackbody(6500.0);
        let peak = s.iter().position(|v| *v == 1.0).unwrap_or(0);
        assert!((lambda_nm(peak) - 445.0).abs() <= 10.0);
    }
}
