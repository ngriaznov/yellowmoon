//! The noise model on the 16-bit working scale (`docs/algorithms/INTEGRATION.md` section 3).
//!
//! The raw-algorithm package (`docs/algorithms/`) defines its absolute constants (offsets, noise
//! gates, floors of "1 DN") on a working scale of **S = 65535** relative to white. [`NoiseDn`]
//! holds the two numbers every such constant is built from, converted once per frame from the
//! normalised [`NoiseModel`] of the sigma map (`docs/06-noise.md`):
//!
//! ```text
//! x_dn = S * x                                   (mosaic, planes)
//! g    = S * a                                   (shot-noise gain)
//! r    = S * sqrt(b + sigma_row^2 + q^2 / 12)    (signal-independent sigma)
//! sigma_dn^2(x_dn) = g * x_dn + r^2
//! ```
//!
//! * `S` is fixed relative to white ([`WORKING_SCALE`]), even if the normalisation of the mosaic
//!   ever changes. White is 1.0 after R1, so `x_dn = 65535` at white.
//! * The model's per-plane offsets `delta` are not used (the specs' formulas have no
//!   per-channel black offset).
//! * The model is the *unbalanced* one: it describes the mosaic before the R4 gains. The
//!   algorithms that work on balanced data multiply by the gains themselves
//!   (`docs/algorithms/03-diffraction-deconvolution.md` section 2.3).
//! * The arithmetic is `f64` and follows the order of the formula above (`b + sigma_row^2 +
//!   q^2 / 12`, left to right, no fused multiply-add), the order of the development oracle, so
//!   the values agree with the `derived` fields of the vectors to the digits those carry.

use ym_noise::model::NoiseModel;

/// The working scale `S` of the raw algorithms: 16-bit white, `65535`. `x_dn = S * x` for a
/// normalised value `x` (white = 1.0).
pub const WORKING_SCALE: f64 = 65535.0;

/// Poisson-Gaussian noise on the working scale: `sigma_dn^2(x_dn) = g * x_dn + r^2`.
///
/// Built from a normalised model with [`NoiseDn::from_model`], or [`NoiseDn::ZERO`] when no
/// model is known. A zero model is valid input for every algorithm of the package: constants that
/// derive from the noise then sit at their documented floors (for example K = T = 1 DN in the
/// adaptive ratio demosaic).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoiseDn {
    /// Shot-noise gain `g = S * a`: variance in DN squared per DN of signal.
    pub g: f64,
    /// Signal-independent standard deviation `r = S * sqrt(b + sigma_row^2 + q^2 / 12)`, in DN.
    pub r: f64,
}

impl NoiseDn {
    /// No noise: `g = 0`, `r = 0`.
    pub const ZERO: NoiseDn = NoiseDn { g: 0.0, r: 0.0 };

    /// Builds a value from the two numbers on the working scale.
    #[must_use]
    pub const fn new(g: f64, r: f64) -> NoiseDn {
        NoiseDn { g, r }
    }

    /// The zero-noise model (same as [`NoiseDn::ZERO`] and [`Default::default`]).
    #[must_use]
    pub const fn zero() -> NoiseDn {
        NoiseDn::ZERO
    }

    /// Converts the normalised, unbalanced model of the sigma map:
    /// `g = S * a`, `r = S * sqrt(b + sigma_row^2 + q^2 / 12)`.
    ///
    /// `NoiseModel::delta` is not used. The conversion is the plain formula: a model with a
    /// negative or non-finite field gives a negative or `NaN` result, which
    /// [`is_valid`](Self::is_valid) reports; nothing is clamped here.
    #[must_use]
    pub fn from_model(model: &NoiseModel) -> NoiseDn {
        let variance = model.b + model.sigma_row * model.sigma_row + model.q * model.q / 12.0;
        NoiseDn {
            g: WORKING_SCALE * model.a,
            r: WORKING_SCALE * variance.sqrt(),
        }
    }

    /// Whether both numbers are finite and not negative, which is what the algorithms accept
    /// (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 4: a negative or non-finite noise
    /// field is `BadNoise`).
    ///
    /// This checks the two numbers, not the four fields of the model they came from: a negative
    /// `b` that a positive `sigma_row` or `q` outweighs, or a negative `sigma_row`, still gives a
    /// valid pair (the oracle's `derive()` of the adaptive ratio demosaic refuses those per field).
    /// The estimator of `ym-noise` never produces them. A negative variance (`NaN` in `r`), a
    /// negative `a` and non-finite fields are all caught here. The log-guided and diffraction
    /// references read an invalid `r` as 0 instead of refusing it (spec 02 section 2.2); a stage
    /// that follows them maps a `NaN` or infinite `r` to 0 itself.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.g.is_finite() && self.g >= 0.0 && self.r.is_finite() && self.r >= 0.0
    }

    /// Whether this is the zero-noise model (`g == 0` and `r == 0`).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.g == 0.0 && self.r == 0.0
    }
}

impl From<&NoiseModel> for NoiseDn {
    fn from(model: &NoiseModel) -> NoiseDn {
        NoiseDn::from_model(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseModel {
        NoiseModel {
            a,
            b,
            delta: [0.0; 4],
            sigma_row,
            q,
        }
    }

    #[test]
    fn working_scale_is_16_bit_white() {
        assert_eq!(WORKING_SCALE, f64::from(u16::MAX));
    }

    #[test]
    fn the_conversion_is_the_formula() {
        // A model that uses every field.
        let m = model(2.0e-5, 1.0e-8, 3.0e-5, 1.5e-5);
        let n = NoiseDn::from_model(&m);
        let g = 65535.0 * 2.0e-5;
        let r = 65535.0 * (1.0e-8 + 3.0e-5 * 3.0e-5 + 1.5e-5 * 1.5e-5 / 12.0_f64).sqrt();
        assert_eq!(n.g, g);
        assert_eq!(n.r, r);
        assert_eq!(NoiseDn::from(&m), n);
    }

    #[test]
    fn each_field_enters_as_the_formula_says() {
        // Only a: g scales, r stays 0.
        let n = NoiseDn::from_model(&model(1.0e-4, 0.0, 0.0, 0.0));
        assert!((n.g - 6.5535).abs() < 1e-12);
        assert_eq!(n.r, 0.0);
        // Only b: r = S * sqrt(b).
        let n = NoiseDn::from_model(&model(0.0, 4.0e-8, 0.0, 0.0));
        assert_eq!(n.g, 0.0);
        assert!((n.r - 65535.0 * 2.0e-4).abs() < 1e-9);
        // Only sigma_row: r = S * sigma_row (a standard deviation, not a variance).
        let n = NoiseDn::from_model(&model(0.0, 0.0, 1.0e-3, 0.0));
        assert!((n.r - 65.535).abs() < 1e-9);
        // Only q: the uniform quantisation error, q / sqrt(12).
        let n = NoiseDn::from_model(&model(0.0, 0.0, 0.0, 1.0e-3));
        assert!((n.r - 65.535 / 12.0_f64.sqrt()).abs() < 1e-9);
        // The terms add as variances.
        let n = NoiseDn::from_model(&model(0.0, 9.0e-6, 4.0e-3, 0.0));
        assert!((n.r - 65535.0 * (9.0e-6_f64 + 1.6e-5).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn the_black_offsets_are_not_used() {
        let mut m = model(2.0e-5, 1.0e-8, 0.0, 1.5e-5);
        let plain = NoiseDn::from_model(&m);
        m.delta = [0.01, -0.02, 0.03, 0.5];
        assert_eq!(NoiseDn::from_model(&m), plain);
    }

    #[test]
    fn the_zero_model_converts_to_zero() {
        assert_eq!(
            NoiseDn::from_model(&model(0.0, 0.0, 0.0, 0.0)),
            NoiseDn::ZERO
        );
        assert_eq!(NoiseDn::zero(), NoiseDn::ZERO);
        assert_eq!(NoiseDn::default(), NoiseDn::ZERO);
        assert!(NoiseDn::ZERO.is_zero());
        assert!(NoiseDn::ZERO.is_valid());
        assert!(!NoiseDn::new(1.0, 0.0).is_zero());
        assert!(!NoiseDn::new(0.0, 1.0).is_zero());
    }

    #[test]
    fn invalid_values_are_reported_not_clamped() {
        assert!(NoiseDn::new(1.3, 6.5).is_valid());
        for bad in [
            NoiseDn::new(-1.0, 1.0),
            NoiseDn::new(1.0, -1.0),
            NoiseDn::new(f64::NAN, 1.0),
            NoiseDn::new(1.0, f64::NAN),
            NoiseDn::new(f64::INFINITY, 1.0),
            NoiseDn::new(1.0, f64::INFINITY),
        ] {
            assert!(!bad.is_valid(), "{bad:?}");
        }
        // A negative variance (b < 0) shows up as NaN in r.
        let n = NoiseDn::from_model(&model(1.0e-5, -1.0e-6, 0.0, 0.0));
        assert!(n.r.is_nan());
        assert!(!n.is_valid());
        // A negative gain stays negative.
        assert!(!NoiseDn::from_model(&model(-1.0e-5, 0.0, 0.0, 0.0)).is_valid());
    }
}
