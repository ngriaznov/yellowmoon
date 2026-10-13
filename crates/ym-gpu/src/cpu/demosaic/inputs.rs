//! Per-frame inputs of the demosaic methods that need more than the mosaic (RA-02,
//! `docs/algorithms/INTEGRATION.md` sections 2 to 4 and 6).
//!
//! [`DemosaicInputs`] is what [`demosaic_with`](super::demosaic_with) takes besides the mosaic and
//! the method: the R4 gains, the noise model on the 16-bit working scale ([`NoiseDn`]) and, for
//! the CA-aware green refinement of the adaptive ratio demosaic, the lateral CA curves
//! ([`CaCurves`]). Bilinear, Menon and LMMSE work on a balanced mosaic and ignore all three.
//!
//! [`CaCurves`] is the *type* of `docs/algorithms/01-adaptive-ratio-demosaic.md` section 3.6 /
//! `INTEGRATION.md` section 6: two radial shift curves (red and blue against green) and the
//! optical centre. This module defines the type, checks it (strictly increasing radii, at least
//! two knots, finite numbers: the oracle's `curve_valid`) and evaluates a curve (linear between
//! knots, linear extrapolation of the first and last segment); `LensGeometry::ca_curves`
//! (`cpu::lens`) builds them from the R9 lens knots (RA-03b).

use super::super::noise::NoiseDn;

/// Why a [`RadialCurve`] or [`CaCurves`] was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CaCurveError {
    /// Fewer than two knots.
    #[error("a CA curve needs at least 2 knots, got {0}")]
    TooFewKnots(usize),
    /// The radius and shift lists differ in length.
    #[error("a CA curve has {radius} radii but {shift} shifts")]
    LengthMismatch {
        /// Number of radii.
        radius: usize,
        /// Number of shifts.
        shift: usize,
    },
    /// A radius or shift is NaN or infinite.
    #[error("knot {index} of the CA curve is not finite")]
    NotFinite {
        /// Index of the first offending knot.
        index: usize,
    },
    /// A radius does not exceed the one before it.
    #[error("the radii of a CA curve must increase strictly, but knot {index} does not")]
    NotIncreasing {
        /// Index of the first knot whose radius is not above its predecessor's.
        index: usize,
    },
    /// The optical centre is NaN or infinite.
    #[error("the optical centre ({x}, {y}) is not finite")]
    CentreNotFinite {
        /// Centre x in frame pixels.
        x: f64,
        /// Centre y in frame pixels.
        y: f64,
    },
}

/// The lateral shift of one colour plane relative to green as a function of the radius from the
/// optical centre: knots `(radius_px, shift_px)` with strictly increasing radius, at least two of
/// them. Linear between knots; the first and last segments extend linearly beyond the ends.
///
/// Radii and shifts are in pixels of the mosaic being demosaiced. The sign convention does not
/// matter to the demosaic (only `|shift_R - shift_B|` is used).
#[derive(Clone, Debug, PartialEq)]
pub struct RadialCurve {
    // Invariant (checked by `new`): same length, at least 2, all finite, radius strictly increasing.
    radius_px: Vec<f64>,
    shift_px: Vec<f64>,
}

impl RadialCurve {
    /// Builds a curve from the radius and shift columns (the layout of the vectors' `red` and
    /// `blue` objects).
    ///
    /// # Errors
    ///
    /// [`CaCurveError`] when the lists differ in length, hold fewer than two knots, contain a
    /// non-finite number or the radii do not increase strictly.
    pub fn new(radius_px: Vec<f64>, shift_px: Vec<f64>) -> Result<RadialCurve, CaCurveError> {
        if radius_px.len() != shift_px.len() {
            return Err(CaCurveError::LengthMismatch {
                radius: radius_px.len(),
                shift: shift_px.len(),
            });
        }
        if radius_px.len() < 2 {
            return Err(CaCurveError::TooFewKnots(radius_px.len()));
        }
        for (index, (r, s)) in radius_px.iter().zip(&shift_px).enumerate() {
            if !r.is_finite() || !s.is_finite() {
                return Err(CaCurveError::NotFinite { index });
            }
            if index > 0
                && r.partial_cmp(&radius_px[index - 1]) != Some(std::cmp::Ordering::Greater)
            {
                return Err(CaCurveError::NotIncreasing { index });
            }
        }
        Ok(RadialCurve {
            radius_px,
            shift_px,
        })
    }

    /// Builds a curve from `(radius_px, shift_px)` knots.
    ///
    /// # Errors
    ///
    /// As [`RadialCurve::new`].
    pub fn from_knots(knots: &[(f64, f64)]) -> Result<RadialCurve, CaCurveError> {
        RadialCurve::new(
            knots.iter().map(|k| k.0).collect(),
            knots.iter().map(|k| k.1).collect(),
        )
    }

    /// The knot radii in pixels, strictly increasing.
    #[must_use]
    pub fn radius_px(&self) -> &[f64] {
        &self.radius_px
    }

    /// The knot shifts in pixels, one per radius.
    #[must_use]
    pub fn shift_px(&self) -> &[f64] {
        &self.shift_px
    }

    /// Number of knots (at least 2).
    #[must_use]
    pub fn len(&self) -> usize {
        self.radius_px.len()
    }

    /// Always `false`: a curve has at least two knots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The shift at `radius_px`: linear interpolation between the two knots around it, linear
    /// extrapolation of the first segment below the first knot and of the last segment above the
    /// last. A NaN radius gives NaN.
    #[must_use]
    pub fn shift_at(&self, radius_px: f64) -> f64 {
        let r = &self.radius_px;
        let n = r.len(); // invariant: n >= 2
        let j = if radius_px >= r[n - 1] {
            n - 2
        } else if radius_px > r[0] {
            // r[lo] <= radius_px < r[hi]
            let (mut lo, mut hi) = (0, n - 1);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if r[mid] <= radius_px {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        } else {
            0
        };
        let t = (radius_px - r[j]) / (r[j + 1] - r[j]);
        self.shift_px[j] + (self.shift_px[j + 1] - self.shift_px[j]) * t
    }
}

/// The lateral CA of the lens as two radial curves around the optical centre
/// (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 3.6).
///
/// Coordinates are *frame* pixels (pixel `n` covers `[n, n + 1)`); the position of a mosaic buffer
/// in the frame comes from its [`Placement`](super::super::Placement). `None` instead of a value
/// of this type means the plain path: no CA tags, `decode.chromatic_aberration = false`, or a
/// binned or downscaled mosaic.
#[derive(Clone, Debug, PartialEq)]
pub struct CaCurves {
    /// Optical centre x in frame pixels (the R9 centre: half the frame width).
    pub centre_x: f64,
    /// Optical centre y in frame pixels (half the frame height).
    pub centre_y: f64,
    /// Shift of the red plane relative to green against the radius from the centre.
    pub red: RadialCurve,
    /// Shift of the blue plane relative to green against the radius from the centre.
    pub blue: RadialCurve,
}

impl CaCurves {
    /// Builds the CA input from its parts.
    ///
    /// # Errors
    ///
    /// [`CaCurveError::CentreNotFinite`] for a NaN or infinite centre (the curves are already
    /// checked by [`RadialCurve::new`]).
    pub fn new(
        centre_x: f64,
        centre_y: f64,
        red: RadialCurve,
        blue: RadialCurve,
    ) -> Result<CaCurves, CaCurveError> {
        if !centre_x.is_finite() || !centre_y.is_finite() {
            return Err(CaCurveError::CentreNotFinite {
                x: centre_x,
                y: centre_y,
            });
        }
        Ok(CaCurves {
            centre_x,
            centre_y,
            red,
            blue,
        })
    }

    /// `|shift_R(r) - shift_B(r)|`: the red/blue misregistration in pixels at radius `radius_px`
    /// from the centre (the `ca` of section 3.6, before the clamp to 1 that pass 2 applies).
    #[must_use]
    pub fn misregistration_px(&self, radius_px: f64) -> f64 {
        (self.red.shift_at(radius_px) - self.blue.shift_at(radius_px)).abs()
    }
}

/// Why [`DemosaicInputs::validate`] refused its input.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DemosaicInputError {
    /// A gain is not positive and finite.
    #[error("white-balance gains {0:?} are not usable (need positive, finite values)")]
    BadGains([f32; 3]),
    /// The noise model has a negative or non-finite field.
    #[error("noise model {0:?} is not usable (need finite, non-negative values)")]
    BadNoise(NoiseDn),
}

/// What a demosaic needs besides the mosaic: gains, noise and optional CA curves.
///
/// Only the adaptive ratio demosaic (`Method::AdaptiveRatio`, which divides the balanced mosaic it
/// is given by the R4 gains `g0` in `gains` to work on the unbalanced one) and the log-guided
/// demosaic read these; Bilinear, Menon and LMMSE ignore them. [`DemosaicInputs::default`] is the input of
/// [`demosaic`](super::demosaic): unit gains, zero noise, no CA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemosaicInputs<'a> {
    /// Absolute white-balance gains `g0` (red, green, blue): finite and positive.
    pub gains: [f32; 3],
    /// The noise model on the 16-bit working scale ([`NoiseDn::ZERO`] when unknown).
    pub noise: NoiseDn,
    /// Lateral CA curves for the CA-aware green refinement; `None` is the plain path.
    pub ca: Option<&'a CaCurves>,
}

impl DemosaicInputs<'_> {
    /// Unit gains, zero noise, no CA curves: what [`demosaic`](super::demosaic) passes.
    pub const UNIT: DemosaicInputs<'static> = DemosaicInputs {
        gains: [1.0; 3],
        noise: NoiseDn::ZERO,
        ca: None,
    };

    /// Checks the numbers the algorithms rely on: every gain finite and above zero, the noise
    /// valid ([`NoiseDn::is_valid`]). The CA curves are checked when they are built.
    ///
    /// [`demosaic_with`](super::demosaic_with) does not call this (it cannot fail); callers that
    /// take gains or a noise model from outside, such as the chain, should.
    ///
    /// # Errors
    ///
    /// [`DemosaicInputError::BadGains`] or [`DemosaicInputError::BadNoise`].
    pub fn validate(&self) -> Result<(), DemosaicInputError> {
        if !self.gains.iter().all(|g| g.is_finite() && *g > 0.0) {
            return Err(DemosaicInputError::BadGains(self.gains));
        }
        if !self.noise.is_valid() {
            return Err(DemosaicInputError::BadNoise(self.noise));
        }
        Ok(())
    }
}

impl Default for DemosaicInputs<'_> {
    fn default() -> Self {
        DemosaicInputs::UNIT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(knots: &[(f64, f64)]) -> RadialCurve {
        RadialCurve::from_knots(knots).expect("valid curve")
    }

    #[test]
    fn a_curve_needs_two_finite_knots_with_strictly_increasing_radius() {
        assert_eq!(
            RadialCurve::new(vec![], vec![]),
            Err(CaCurveError::TooFewKnots(0))
        );
        assert_eq!(
            RadialCurve::new(vec![1.0], vec![0.5]),
            Err(CaCurveError::TooFewKnots(1))
        );
        assert_eq!(
            RadialCurve::new(vec![0.0, 1.0], vec![0.0]),
            Err(CaCurveError::LengthMismatch {
                radius: 2,
                shift: 1
            })
        );
        assert_eq!(
            RadialCurve::new(vec![0.0, 1.0, 1.0], vec![0.0; 3]),
            Err(CaCurveError::NotIncreasing { index: 2 })
        );
        assert_eq!(
            RadialCurve::new(vec![0.0, 2.0, 1.0], vec![0.0; 3]),
            Err(CaCurveError::NotIncreasing { index: 2 })
        );
        assert_eq!(
            RadialCurve::new(vec![0.0, f64::NAN], vec![0.0; 2]),
            Err(CaCurveError::NotFinite { index: 1 })
        );
        assert_eq!(
            RadialCurve::new(vec![0.0, 1.0], vec![0.0, f64::INFINITY]),
            Err(CaCurveError::NotFinite { index: 1 })
        );
        assert!(RadialCurve::new(vec![-3.0, 0.0, 8.0], vec![0.0; 3]).is_ok());
    }

    #[test]
    fn evaluation_is_linear_between_knots_and_extrapolates_the_end_segments() {
        let c = curve(&[(0.0, 0.0), (10.0, 1.0), (20.0, 4.0)]);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        // At the knots.
        for (r, s) in [(0.0, 0.0), (10.0, 1.0), (20.0, 4.0)] {
            assert_eq!(c.shift_at(r), s, "knot at {r}");
        }
        // Between them.
        assert!((c.shift_at(5.0) - 0.5).abs() < 1e-15);
        assert!((c.shift_at(15.0) - 2.5).abs() < 1e-15);
        // Beyond the ends: the first segment slope 0.1, the last 0.3.
        assert!((c.shift_at(-10.0) + 1.0).abs() < 1e-15);
        assert!((c.shift_at(30.0) - 7.0).abs() < 1e-15);
        assert!(c.shift_at(f64::NAN).is_nan());
    }

    #[test]
    fn a_two_knot_curve_is_a_line_everywhere() {
        let c = curve(&[(1.0, 2.0), (3.0, 6.0)]);
        for r in [-5.0, 0.0, 1.0, 2.0, 3.0, 100.0] {
            assert!(
                (c.shift_at(r) - (2.0 + 2.0 * (r - 1.0))).abs() < 1e-12,
                "r = {r}"
            );
        }
    }

    #[test]
    fn misregistration_is_the_absolute_red_blue_difference() {
        let ca = CaCurves::new(
            3024.0,
            2012.0,
            curve(&[(0.0, 0.0), (100.0, 1.0)]),
            curve(&[(0.0, 0.0), (100.0, -0.5)]),
        )
        .expect("ca");
        assert!((ca.misregistration_px(50.0) - 0.75).abs() < 1e-15);
        // The order of the colours does not matter.
        let swapped = CaCurves::new(0.0, 0.0, ca.blue.clone(), ca.red.clone()).expect("ca");
        assert_eq!(
            swapped.misregistration_px(50.0),
            ca.misregistration_px(50.0)
        );
    }

    #[test]
    fn the_centre_must_be_finite() {
        let c = curve(&[(0.0, 0.0), (1.0, 0.0)]);
        assert!(CaCurves::new(1.0, 2.0, c.clone(), c.clone()).is_ok());
        for (x, y) in [(f64::NAN, 0.0), (0.0, f64::INFINITY)] {
            assert!(matches!(
                CaCurves::new(x, y, c.clone(), c.clone()),
                Err(CaCurveError::CentreNotFinite { .. })
            ));
        }
    }

    #[test]
    fn the_default_input_has_unit_gains_zero_noise_and_no_ca() {
        let d = DemosaicInputs::default();
        assert_eq!(d.gains, [1.0, 1.0, 1.0]);
        assert_eq!(d.noise, NoiseDn::ZERO);
        assert!(d.ca.is_none());
        assert_eq!(d, DemosaicInputs::UNIT);
        assert_eq!(d.validate(), Ok(()));
    }

    #[test]
    fn validation_rejects_bad_gains_and_bad_noise() {
        let ok = DemosaicInputs {
            gains: [2.0, 1.0, 1.5],
            noise: NoiseDn::new(1.3, 6.5),
            ca: None,
        };
        assert_eq!(ok.validate(), Ok(()));
        for gains in [
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, f32::NAN],
            [f32::INFINITY, 1.0, 1.0],
        ] {
            let bad = DemosaicInputs { gains, ..ok };
            // (A NaN gain is never equal to itself, so compare the variant, not the payload.)
            assert!(
                matches!(bad.validate(), Err(DemosaicInputError::BadGains(_))),
                "{gains:?}"
            );
        }
        let noise = NoiseDn::new(-1.0, 6.5);
        let bad = DemosaicInputs { noise, ..ok };
        assert_eq!(bad.validate(), Err(DemosaicInputError::BadNoise(noise)));
    }
}
