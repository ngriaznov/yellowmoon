//! The noise model `var(mu) = a(mu - delta_c) + b + sigma_row^2 + q^2/12` and its parameter types
//! (`docs/06-noise.md` section Model).
//!
//! All quantities are in normalised units: the mosaic after R1 (black subtracted, white
//! normalised, `docs/05-raw-processing.md`) and before white balance, so white is 1.0 and the
//! shot-noise gain `a` is the same for the four CFA planes.
//!
//! Plane indices follow the position inside the 2x2 CFA cell, `p = 2 * (y & 1) + (x & 1)`, with
//! `(x, y)` counted from the origin of the active area. For an RGGB sensor that is R, G1, G2, B.
//! The model never needs to know which colour a position carries.

use serde::{Deserialize, Serialize};

pub mod synth;

/// Number of CFA planes (positions in the 2x2 cell).
pub const NUM_PLANES: usize = 4;

/// Per-image Poisson-Gaussian noise model in normalised units (`docs/06-noise.md` section Model).
///
/// For CFA plane `c` and a recorded plane mean `mu`:
///
/// ```text
/// var(mu) = a * (mu - delta_c) + b + sigma_row^2 + q^2 / 12
/// ```
///
/// * `a`: shot-noise gain, shared by the four planes;
/// * `delta`: per-plane offsets of the black level (the signal term is `mu - delta_c`);
/// * `b`: read-noise variance of one photosite;
/// * `sigma_row`: standard deviation of the row-to-row offset;
/// * `q`: quantisation step (1 DN normalised for lossless files; the decoder's `qstep` for cRAW).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseModel {
    /// Shot-noise gain (variance per unit of normalised signal), shared by the four planes.
    pub a: f64,
    /// Read-noise variance of one photosite, in normalised units squared.
    pub b: f64,
    /// Per-plane black-level offsets (the "colour bias"), in normalised units.
    pub delta: [f64; NUM_PLANES],
    /// Row-noise standard deviation, in normalised units.
    pub sigma_row: f64,
    /// Quantisation step, in normalised units.
    pub q: f64,
}

impl NoiseModel {
    /// A model with only shot and read noise: no offsets, no row noise, no quantisation.
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            a,
            b,
            delta: [0.0; NUM_PLANES],
            sigma_row: 0.0,
            q: 0.0,
        }
    }

    /// The same model with the quantisation step set (`q`, normalised units).
    pub fn with_quantisation(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// Variance of one photosite whose plane mean is `mu` in plane `plane`, including the row term
    /// and the quantisation term. The signal term is clamped at zero.
    pub fn variance(&self, mu: f64, plane: usize) -> f64 {
        self.pixel_variance(mu, plane) + self.sigma_row * self.sigma_row
    }

    /// Variance without the row-noise term: what a within-row difference such as the Haar detail
    /// coefficients sees (`a(mu - delta_c) + b + q^2/12`).
    pub fn pixel_variance(&self, mu: f64, plane: usize) -> f64 {
        let signal = (mu - self.delta[plane & 3]).max(0.0);
        self.a * signal + self.b + self.q * self.q / 12.0
    }

    /// Standard deviation at plane mean `mu`: `sqrt(variance)`.
    pub fn sigma(&self, mu: f64, plane: usize) -> f64 {
        self.variance(mu, plane).sqrt()
    }

    /// The model with the standard deviation scaled by `factor` (`a`, `b` and `sigma_row` scale
    /// by `factor^2`, `factor^2` and `factor`; the quantisation step is a known constant and stays).
    pub fn with_sigma_scaled(&self, factor: f64) -> Self {
        Self {
            a: self.a * factor * factor,
            b: self.b * factor * factor,
            delta: self.delta,
            sigma_row: self.sigma_row * factor,
            q: self.q,
        }
    }

    /// DNG `NoiseProfile` values per CFA position (`docs/06-noise.md` section DNG NoiseProfile).
    ///
    /// DNG models the noise of a linear value `x` in [0, 1] as `sqrt(S x + O)`. Our `x` already is
    /// the normalised mosaic, so `S = a` and `O = max(0, b - a delta_c + sigma_row^2 + q^2/12)`.
    /// (The research note writes the same thing for a model fitted in DN: `S = a / (W - B)`,
    /// `O = (...) / (W - B)^2`.) V1 does not write the tag; the helper keeps the mapping in one place.
    pub fn dng_noise_profile(&self) -> [(f64, f64); NUM_PLANES] {
        let mut out = [(0.0, 0.0); NUM_PLANES];
        for (c, slot) in out.iter_mut().enumerate() {
            let o = self.b - self.a * self.delta[c]
                + self.sigma_row * self.sigma_row
                + self.q * self.q / 12.0;
            *slot = (self.a, o.max(0.0));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn variance_follows_the_formula() {
        let m = NoiseModel {
            a: 2e-4,
            b: 1e-7,
            delta: [1e-4, 0.0, -1e-4, 2e-4],
            sigma_row: 1e-3,
            q: 1.0 / 16383.0,
        };
        let mu = 0.25;
        let q2 = m.q * m.q / 12.0;
        assert_relative_eq!(
            m.variance(mu, 0),
            2e-4 * (0.25 - 1e-4) + 1e-7 + 1e-6 + q2,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            m.pixel_variance(mu, 2),
            2e-4 * (0.25 + 1e-4) + 1e-7 + q2,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            m.sigma(mu, 1),
            m.variance(mu, 1).sqrt(),
            max_relative = 1e-15
        );
    }

    #[test]
    fn signal_term_is_clamped_at_zero() {
        let m = NoiseModel {
            delta: [1e-3; 4],
            ..NoiseModel::new(1e-3, 1e-8)
        };
        assert_relative_eq!(m.variance(0.0, 0), 1e-8, max_relative = 1e-12);
        assert_relative_eq!(m.variance(-0.01, 3), 1e-8, max_relative = 1e-12);
    }

    #[test]
    fn scaling_sigma_scales_every_term_but_q() {
        let m = NoiseModel {
            a: 1e-4,
            b: 4e-8,
            delta: [0.0; 4],
            sigma_row: 1e-4,
            q: 0.0,
        };
        let s = m.with_sigma_scaled(1.5);
        assert_relative_eq!(s.sigma(0.3, 0), 1.5 * m.sigma(0.3, 0), max_relative = 1e-12);
    }

    #[test]
    fn dng_profile_is_non_negative_and_uses_the_plane_offset() {
        let m = NoiseModel {
            a: 1e-3,
            b: 1e-8,
            delta: [0.0, 0.0, 0.0, 1e-3],
            sigma_row: 0.0,
            q: 0.0,
        };
        let p = m.dng_noise_profile();
        assert_relative_eq!(p[0].0, 1e-3);
        assert_relative_eq!(p[0].1, 1e-8, max_relative = 1e-12);
        // b - a delta = 1e-8 - 1e-6 < 0 clamps to zero.
        assert_eq!(p[3].1, 0.0);
    }
}
