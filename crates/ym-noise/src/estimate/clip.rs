//! The clipped Gaussian: what clipping at the white level does to the mean and variance of a
//! noisy flat region, and the inverse map (`docs/06-noise.md` step 4, after Foi et al. 2008 section IV).
//!
//! Model: a photosite records `z = min(y, c)` with `y ~ N(mu, sigma^2)` and `c` the clip level.
//! With `u = (c - mu) / sigma` and `Z` a standard normal, `c - z = (sigma (u - Z))^+`, so
//!
//! ```text
//! E[c - z]  = sigma * h(u),   h(u) = u Phi(u) + phi(u)
//! Var[z]    = sigma^2 * g(u), g(u) = (u^2 + 1) Phi(u) + u phi(u) - h(u)^2
//! ```
//!
//! (the first and second moments of the positive part of a shifted normal). Both depend on the
//! pair `(mu, sigma)` only through `u` and the scale `sigma`, so an observed `(mean, variance)` of
//! clipped data maps back to the unclipped pair with a one-dimensional root find: the ratio
//! `r(u) = h(u) / sqrt(g(u))` equals `(c - mean) / sqrt(variance)` and is monotone in `u`.
//!
//! The Haar detail coefficient `(z1 - z2 - z3 + z4) / 2` of four independent identically clipped
//! samples has exactly the variance of one sample, and the mean of the four has the mean of one
//! sample, so the same relations hold for the coefficients and block means the estimator uses.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Standard normal density.
pub fn pdf(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
}

/// Complementary error function, accurate to about 1e-14 relative error.
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.5 {
        // Maclaurin series of erf: sum_n (-1)^n x^(2n+1) / (n! (2n+1)).
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        1.0 - 2.0 / PI.sqrt() * sum
    } else {
        // Continued fraction (modified Lentz) for erfc(x) exp(x^2) sqrt(pi).
        // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
        let tiny = 1e-300;
        let mut f = x;
        let mut c = x;
        let mut d = 0.0;
        for k in 1..200 {
            let a = k as f64 * 0.5;
            d = x + a * d;
            if d == 0.0 {
                d = tiny;
            }
            c = x + a / c;
            if c == 0.0 {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-x * x).exp() / PI.sqrt() / f
    }
}

/// Standard normal distribution function.
pub fn cdf(u: f64) -> f64 {
    0.5 * erfc(-u * FRAC_1_SQRT_2)
}

/// Above this `u` the clipped fraction is below 1e-30 and the data are treated as unclipped.
const U_MAX: f64 = 12.0;

/// Below this `u` (clipped fraction above 1 - 1e-9) the variance carries no information.
const U_MIN: f64 = -6.0;

/// `h(u) = E[(u - Z)^+]`.
pub fn h(u: f64) -> f64 {
    if u >= U_MAX {
        return u;
    }
    u * cdf(u) + pdf(u)
}

/// `g(u) = Var[(u - Z)^+]`.
pub fn g(u: f64) -> f64 {
    if u >= U_MAX {
        return 1.0;
    }
    let (p, f) = (cdf(u), pdf(u));
    let hu = u * p + f;
    (u * u + 1.0) * p + u * f - hu * hu
}

/// `r(u) = h(u) / sqrt(g(u))`, monotone increasing; `r(u) -> u` for large `u`.
pub fn ratio(u: f64) -> f64 {
    h(u) / g(u).sqrt()
}

/// Mean and variance of `min(N(mu, sigma^2), clip)`.
pub fn forward(mu: f64, sigma: f64, clip: f64) -> (f64, f64) {
    let u = (clip - mu) / sigma;
    (clip - sigma * h(u), sigma * sigma * g(u))
}

/// Maps the observed mean and variance of clipped data back to the unclipped `(mu, sigma^2)`.
///
/// Returns `None` when the data are clipped so heavily that the variance cannot be inverted
/// (`u < -6`), or when the inputs are not usable (`variance <= 0`, `mean >= clip`).
pub fn invert(mean: f64, variance: f64, clip: f64) -> Option<(f64, f64)> {
    if variance.is_nan() || variance <= 0.0 || mean.is_nan() || mean >= clip || !mean.is_finite() {
        return None;
    }
    let target = (clip - mean) / variance.sqrt();
    if target >= ratio(U_MAX) {
        // Not clipped at all: the observation already is the unclipped pair.
        return Some((mean, variance));
    }
    if target <= ratio(U_MIN) {
        return None;
    }
    let (mut lo, mut hi) = (U_MIN, U_MAX);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if ratio(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let u = 0.5 * (lo + hi);
    let sigma2 = variance / g(u);
    let sigma = sigma2.sqrt();
    Some((clip - u * sigma, sigma2))
}

/// Fraction of samples that sit at the clip level, `1 - Phi(u)`.
pub fn clipped_fraction(mu: f64, sigma: f64, clip: f64) -> f64 {
    1.0 - cdf((clip - mu) / sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::synth::SynthRng;

    #[test]
    fn erfc_matches_known_values() {
        // Reference values to 15 digits.
        let table = [
            (0.0, 1.0),
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-5),
            (5.0, 1.537459794428035e-12),
            (-1.0, 1.8427007929497148),
        ];
        for (x, want) in table {
            let got = erfc(x);
            assert!(
                (got / want - 1.0).abs() < 1e-12,
                "erfc({x}) = {got}, want {want}"
            );
        }
        // Continuity across the switch from the series to the continued fraction.
        assert!((erfc(2.5) - erfc(2.5 - 1e-9)).abs() < 5e-12);
        assert!((cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((cdf(1.959_963_984_540_054) - 0.975).abs() < 1e-12);
    }

    #[test]
    fn ratio_is_monotone() {
        let mut prev = ratio(U_MIN);
        let mut u = U_MIN;
        while u < 11.0 {
            u += 0.05;
            let r = ratio(u);
            assert!(r > prev, "ratio not increasing at u = {u}");
            prev = r;
        }
        assert!((ratio(10.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn forward_matches_monte_carlo() {
        let mut rng = SynthRng::new(99);
        for &(mu, sigma) in &[
            (0.95, 0.03),
            (1.0, 0.02),
            (1.05, 0.04),
            (0.9, 0.05),
            (1.2, 0.1),
        ] {
            let n = 400_000;
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let z = (mu + sigma * rng.normal()).min(1.0);
                s += z;
                s2 += z * z;
            }
            let mean = s / n as f64;
            let var = s2 / n as f64 - mean * mean;
            let (fm, fv) = forward(mu, sigma, 1.0);
            assert!((mean - fm).abs() < 0.01 * sigma, "mean {mean} vs {fm}");
            assert!(
                (var / fv - 1.0).abs() < 0.02,
                "var {var} vs {fv} at {mu} {sigma}"
            );
        }
    }

    #[test]
    fn invert_round_trips() {
        for &(mu, sigma) in &[
            (0.8, 0.02),
            (0.97, 0.03),
            (1.0, 0.02),
            (1.04, 0.05),
            (1.15, 0.08),
            (0.5, 0.01),
        ] {
            let (m, v) = forward(mu, sigma, 1.0);
            let (mu2, s2) = invert(m, v, 1.0).expect("invertible");
            assert!((mu2 - mu).abs() < 1e-7, "mu {mu2} vs {mu}");
            assert!(
                (s2.sqrt() / sigma - 1.0).abs() < 1e-6,
                "sigma {} vs {sigma}",
                s2.sqrt()
            );
        }
        // Hopelessly clipped and invalid inputs.
        let (m, v) = forward(1.6, 0.02, 1.0);
        assert!(invert(m, v.max(1e-300), 1.0).is_none() || v < 1e-100);
        assert!(invert(0.5, 0.0, 1.0).is_none());
        assert!(invert(1.0, 1e-4, 1.0).is_none());
    }

    #[test]
    fn unclipped_data_pass_through() {
        let (mu, s2) = invert(0.4, 1e-4, 1.0).unwrap();
        assert_eq!(mu, 0.4);
        assert_eq!(s2, 1e-4);
    }
}
