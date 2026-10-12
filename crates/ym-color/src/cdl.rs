//! ASC CDL: slope, offset, power and saturation (`docs/02-colour-pipeline.md` section C10).
//!
//! ASC Color Decision List, release 1.2 ("ASC CDL Transfer Functions and Interchange Syntax"),
//! applied per channel and then across channels:
//!
//! ```text
//! x   = max(in * slope + offset, 0) ^ power          (per channel)
//! L   = 0.2126 R + 0.7152 G + 0.0722 B                (Rec.709 weights, on x)
//! out = L + sat * (x - L)
//! ```
//!
//! This is the *unclamped-above* form: the value after the slope and offset step is clamped at 0
//! (so the power never sees a negative base) but there is no clamp at 1, because the CDL here runs
//! on LogC code values and on look output that may leave `[0, 1]`. ASC's own "clamp" style (as in
//! OpenColorIO's `CDL_ASC`) also clamps at 1 before and after saturation; it agrees with this
//! function wherever every intermediate value stays in `[0, 1]`.

use crate::consts::{REC709_LUMA_WEIGHTS, Vec3};

/// Why a CDL was rejected.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum CdlError {
    /// A value is NaN or infinite.
    #[error("CDL values must be finite")]
    NotFinite,
    /// Slope must not be negative (ASC CDL: slope >= 0).
    #[error("CDL slope must not be negative, got {0}")]
    NegativeSlope(f64),
    /// Power must be greater than zero (ASC CDL: power > 0).
    #[error("CDL power must be greater than zero, got {0}")]
    NonPositivePower(f64),
    /// Saturation must not be negative (ASC CDL: saturation >= 0).
    #[error("CDL saturation must not be negative, got {0}")]
    NegativeSaturation(f64),
}

/// An ASC CDL: per-channel slope, offset and power, and a saturation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cdl {
    /// Slope (gain) per channel.
    pub slope: Vec3,
    /// Offset per channel.
    pub offset: Vec3,
    /// Power (gamma) per channel.
    pub power: Vec3,
    /// Saturation; 1 leaves colour unchanged, 0 is Rec.709-weighted luma.
    pub saturation: f64,
}

impl Default for Cdl {
    fn default() -> Self {
        Self::identity()
    }
}

impl Cdl {
    /// The identity CDL: slope 1, offset 0, power 1, saturation 1.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        }
    }

    /// Builds a CDL, checking the ASC ranges (finite; slope >= 0; power > 0; saturation >= 0).
    ///
    /// # Errors
    ///
    /// A [`CdlError`] naming the first violated range.
    pub fn new(slope: Vec3, offset: Vec3, power: Vec3, saturation: f64) -> Result<Self, CdlError> {
        let finite = slope
            .iter()
            .chain(&offset)
            .chain(&power)
            .all(|v| v.is_finite())
            && saturation.is_finite();
        if !finite {
            return Err(CdlError::NotFinite);
        }
        if let Some(&s) = slope.iter().find(|&&s| s < 0.0) {
            return Err(CdlError::NegativeSlope(s));
        }
        if let Some(&p) = power.iter().find(|&&p| p <= 0.0) {
            return Err(CdlError::NonPositivePower(p));
        }
        if saturation < 0.0 {
            return Err(CdlError::NegativeSaturation(saturation));
        }
        Ok(Self {
            slope,
            offset,
            power,
            saturation,
        })
    }

    /// Whether this CDL leaves every input unchanged.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::identity()
    }

    /// Applies slope, offset and power, then saturation, to one RGB triple.
    #[must_use]
    pub fn apply(&self, rgb: Vec3) -> Vec3 {
        let mut x = [0.0; 3];
        for k in 0..3 {
            let sop = (rgb[k] * self.slope[k] + self.offset[k]).max(0.0);
            // powf(x, 1.0) is exact in IEEE 754 libm, but skip it anyway for the common case.
            x[k] = if self.power[k] == 1.0 {
                sop
            } else {
                sop.powf(self.power[k])
            };
        }
        if self.saturation == 1.0 {
            return x;
        }
        let l = REC709_LUMA_WEIGHTS[0] * x[0]
            + REC709_LUMA_WEIGHTS[1] * x[1]
            + REC709_LUMA_WEIGHTS[2] * x[2];
        [
            l + self.saturation * (x[0] - l),
            l + self.saturation * (x[1] - l),
            l + self.saturation * (x[2] - l),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_exact() {
        let c = Cdl::identity();
        for v in [[0.0, 0.5, 1.0], [0.2, 0.3, 0.4], [1.7, 0.0, 0.9]] {
            assert_eq!(c.apply(v), v);
        }
        assert!(c.is_identity());
    }

    #[test]
    fn validates_ranges() {
        assert_eq!(
            Cdl::new([1.0; 3], [0.0; 3], [1.0; 3], f64::NAN),
            Err(CdlError::NotFinite)
        );
        assert_eq!(
            Cdl::new([-1.0, 1.0, 1.0], [0.0; 3], [1.0; 3], 1.0),
            Err(CdlError::NegativeSlope(-1.0))
        );
        assert_eq!(
            Cdl::new([1.0; 3], [0.0; 3], [1.0, 0.0, 1.0], 1.0),
            Err(CdlError::NonPositivePower(0.0))
        );
        assert_eq!(
            Cdl::new([1.0; 3], [0.0; 3], [1.0; 3], -0.1),
            Err(CdlError::NegativeSaturation(-0.1))
        );
        assert!(Cdl::new([0.0; 3], [-5.0; 3], [0.5; 3], 0.0).is_ok());
    }
}
