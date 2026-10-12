//! Highlight roll-off and the low-light toe (stage C7) (`docs/02-colour-pipeline.md` section C7).
//!
//! Both act on the norm `n = max(R, G, B)` of a linear AWG4 pixel and scale all three channels by
//! `f(n) / n`, which keeps the channel ratios (hue and saturation) unchanged.
//!
//! * **Highlights** (`r_h` in `[0, 1]`, default 0.5). With `c` the clip level after C1 to C6, the
//!   knee is `k = c * 2^-(0.5 + 2 r_h)`. Up to the knee `f(n) = n`; above it
//!   `f(n) = k + (c' - k) tanh((n - k) / (c' - k))` with `c' = 1.5 c`. The two pieces meet with
//!   equal value and equal slope (1) at the knee, and `f` rises towards `c'` without reaching it.
//! * **Low light** (`r_l` in `[0, 1]`, default 0.5). `tau = 4 r_l sigma_floor` and
//!   `f(n) = n^3 / (n^2 + tau^2)`: monotonic, close to `n` for `n >> tau`, and crushing the noise
//!   floor smoothly below `tau`. `r_l = 0` disables it exactly.
//!
//! The two are composed as `f = toe(shoulder(n))`. The shoulder only changes values above
//! `k >= 0.17 c` and the toe only values of a few `tau`, so for any sensible noise floor the order
//! does not matter; the composition of two monotone functions is monotone, which is what the tests
//! check over the whole parameter range.
//!
//! Negative or zero `n` (all channels at or below zero) skips the shoulder; the toe's scale
//! `n^2 / (n^2 + tau^2)` is still applied, so a pixel with `n = 0` and `tau > 0` maps to black.
//! Non-finite inputs pass through unchanged.

use crate::consts::Vec3;
use crate::gamut::LinearAwg4;

/// Why a [`ToneParams`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ToneError {
    /// The clip level must be finite and greater than zero.
    #[error("clip level must be finite and greater than zero, got {0}")]
    InvalidClipLevel(f64),
    /// The noise floor must be finite and not negative.
    #[error("noise floor must be finite and not negative, got {0}")]
    InvalidNoiseFloor(f64),
}

/// Parameters of stage C7.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneParams {
    /// `rolloff.highlights`, `r_h`, clamped to `[0, 1]`.
    pub highlights: f64,
    /// `rolloff.low_light`, `r_l`, clamped to `[0, 1]`. Zero disables the toe.
    pub low_light: f64,
    /// Clip level `c` in linear AWG4 after C1 to C6 (smallest per-channel clip, mapped to AWG4, times exposure).
    pub clip_level: f64,
    /// `sigma_floor`: read noise of the noise model mapped to linear AWG4 at this exposure.
    pub noise_floor: f64,
}

impl ToneParams {
    /// Builds parameters, clamping the two roll-off amounts to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`ToneError::InvalidClipLevel`] unless `clip_level` is finite and positive;
    /// [`ToneError::InvalidNoiseFloor`] unless `noise_floor` is finite and not negative.
    pub fn new(
        highlights: f64,
        low_light: f64,
        clip_level: f64,
        noise_floor: f64,
    ) -> Result<Self, ToneError> {
        if !clip_level.is_finite() || clip_level <= 0.0 {
            return Err(ToneError::InvalidClipLevel(clip_level));
        }
        if !noise_floor.is_finite() || noise_floor < 0.0 {
            return Err(ToneError::InvalidNoiseFloor(noise_floor));
        }
        Ok(Self {
            highlights: clamp_unit(highlights),
            low_light: clamp_unit(low_light),
            clip_level,
            noise_floor,
        })
    }

    /// The shoulder knee `k = c * 2^-(0.5 + 2 r_h)`.
    #[must_use]
    pub fn knee(&self) -> f64 {
        self.clip_level * (-(0.5 + 2.0 * clamp_unit(self.highlights))).exp2()
    }

    /// The shoulder's asymptote `c' = 1.5 c`.
    #[must_use]
    pub fn shoulder_limit(&self) -> f64 {
        1.5 * self.clip_level
    }

    /// The toe width `tau = 4 r_l sigma_floor`.
    #[must_use]
    pub fn toe_tau(&self) -> f64 {
        4.0 * clamp_unit(self.low_light) * self.noise_floor
    }

    /// The highlight shoulder alone: `n` up to the knee, `k + (c' - k) tanh((n - k) / (c' - k))` above.
    #[must_use]
    pub fn shoulder(&self, n: f64) -> f64 {
        let k = self.knee();
        if n <= k {
            n
        } else {
            let span = self.shoulder_limit() - k;
            k + span * ((n - k) / span).tanh()
        }
    }

    /// The low-light toe alone: `n^3 / (n^2 + tau^2)`, or `n` when `tau` is zero.
    #[must_use]
    pub fn toe(&self, n: f64) -> f64 {
        n * toe_scale(n, self.toe_tau())
    }

    /// The full norm curve `f(n) = toe(shoulder(n))`.
    #[must_use]
    pub fn curve(&self, n: f64) -> f64 {
        self.toe(self.shoulder(n))
    }

    /// The factor `f(n) / n` that multiplies all three channels.
    #[must_use]
    pub fn scale(&self, n: f64) -> f64 {
        let m = self.shoulder(n);
        let toe = toe_scale(m, self.toe_tau());
        if n == 0.0 {
            // Limit of (m / n) as n -> 0 is 1 (the shoulder is the identity there); only the toe acts.
            return toe;
        }
        (m / n) * toe
    }

    /// Applies stage C7 to a linear AWG4 pixel.
    #[must_use]
    pub fn apply(&self, rgb: LinearAwg4) -> LinearAwg4 {
        LinearAwg4(self.apply_rgb(rgb.0))
    }

    /// Applies stage C7 to a raw linear AWG4 triple.
    #[must_use]
    pub fn apply_rgb(&self, rgb: Vec3) -> Vec3 {
        let n = rgb[0].max(rgb[1]).max(rgb[2]);
        if !n.is_finite() || rgb.iter().any(|c| !c.is_finite()) {
            return rgb;
        }
        let s = self.scale(n);
        [rgb[0] * s, rgb[1] * s, rgb[2] * s]
    }
}

/// `m^2 / (m^2 + tau^2)`, the factor the toe applies to `m`, written as `1 / (1 + (tau / m)^2)` so
/// that nothing underflows to `0 / 0` when both `m` and `tau` are tiny.
fn toe_scale(m: f64, tau: f64) -> f64 {
    if tau == 0.0 {
        1.0
    } else if m == 0.0 {
        0.0
    } else {
        let ratio = tau / m;
        1.0 / (1.0 + ratio * ratio)
    }
}

fn clamp_unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_parameters() {
        assert!(matches!(
            ToneParams::new(0.5, 0.5, 0.0, 0.0),
            Err(ToneError::InvalidClipLevel(_))
        ));
        assert!(matches!(
            ToneParams::new(0.5, 0.5, f64::NAN, 0.0),
            Err(ToneError::InvalidClipLevel(_))
        ));
        assert!(matches!(
            ToneParams::new(0.5, 0.5, 1.0, -1.0),
            Err(ToneError::InvalidNoiseFloor(_))
        ));
        let p = ToneParams::new(2.0, -1.0, 1.0, 0.0).unwrap();
        assert_eq!((p.highlights, p.low_light), (1.0, 0.0));
    }

    #[test]
    fn knee_positions_match_the_spec() {
        let c = 16.0;
        let lo = ToneParams::new(0.0, 0.0, c, 0.0).unwrap();
        let hi = ToneParams::new(1.0, 0.0, c, 0.0).unwrap();
        assert!((lo.knee() - c * 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((hi.knee() - c * (-2.5_f64).exp2()).abs() < 1e-12);
    }
}
