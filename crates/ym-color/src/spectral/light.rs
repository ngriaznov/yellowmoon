//! Light classes and class probabilities (`docs/02-colour-pipeline.md` sections C2 and C3-C4).
//!
//! The camera profile carries tables for four classes of reference light: daylight, tungsten,
//! fluorescent and LED. The illuminant estimator (`docs/02`, "Illuminant estimate") reports its own
//! richer set of classes (shade/overcast, tungsten/halogen, mixed, ...); folding those into these four
//! belongs to the estimator. Everything in this module works on the four profile classes.

use serde::{Deserialize, Serialize};

/// The four classes of reference light a profile has tables for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightClass {
    /// CIE daylight (D series), 4000 K and above in the reference set.
    Daylight,
    /// Blackbody radiators (tungsten, halogen), 2000 to 4000 K in the reference set.
    Tungsten,
    /// Fluorescent tubes (CIE F1 to F12 in the reference set).
    Fluorescent,
    /// LEDs (CIE LED-B1 to B5, BH1, RGB1, V1, V2 in the reference set).
    Led,
}

impl LightClass {
    /// The classes in profile order (`daylight`, `tungsten`, `fluorescent`, `led`).
    pub const ALL: [LightClass; 4] = [
        LightClass::Daylight,
        LightClass::Tungsten,
        LightClass::Fluorescent,
        LightClass::Led,
    ];

    /// The class name as it appears in `profile.json`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            LightClass::Daylight => "daylight",
            LightClass::Tungsten => "tungsten",
            LightClass::Fluorescent => "fluorescent",
            LightClass::Led => "led",
        }
    }

    /// Position in [`LightClass::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            LightClass::Daylight => 0,
            LightClass::Tungsten => 1,
            LightClass::Fluorescent => 2,
            LightClass::Led => 3,
        }
    }
}

/// Why a set of class probabilities was refused.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ClassProbabilityError {
    /// A probability is NaN, infinite or negative.
    #[error("class probability for {class:?} must be finite and not negative, found {value}")]
    Invalid {
        /// The offending class.
        class: LightClass,
        /// The offending value.
        value: f64,
    },
    /// All four probabilities are zero, so there is nothing to normalise.
    #[error("the class probabilities are all zero")]
    AllZero,
}

/// Probabilities of the four light classes, summing to 1.
///
/// [`ClassProbabilities::new`] normalises its input, so estimator outputs that sum to 1 only within
/// rounding (or unnormalised scores) are accepted; the stored values always sum to 1 within rounding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ClassProbabilities {
    daylight: f64,
    tungsten: f64,
    fluorescent: f64,
    led: f64,
}

impl ClassProbabilities {
    /// Builds probabilities from four non-negative scores, dividing by their sum.
    ///
    /// # Errors
    ///
    /// Returns a [`ClassProbabilityError`] for a negative or non-finite value or four zeros.
    pub fn new(
        daylight: f64,
        tungsten: f64,
        fluorescent: f64,
        led: f64,
    ) -> Result<Self, ClassProbabilityError> {
        let raw = [daylight, tungsten, fluorescent, led];
        for (class, &value) in LightClass::ALL.iter().zip(&raw) {
            if !value.is_finite() || value < 0.0 {
                return Err(ClassProbabilityError::Invalid {
                    class: *class,
                    value,
                });
            }
        }
        let total: f64 = raw.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(ClassProbabilityError::AllZero);
        }
        Ok(Self {
            daylight: daylight / total,
            tungsten: tungsten / total,
            fluorescent: fluorescent / total,
            led: led / total,
        })
    }

    /// All the probability on one class.
    #[must_use]
    pub fn only(class: LightClass) -> Self {
        let mut p = [0.0; 4];
        p[class.index()] = 1.0;
        Self {
            daylight: p[0],
            tungsten: p[1],
            fluorescent: p[2],
            led: p[3],
        }
    }

    /// The probability of `class`.
    #[must_use]
    pub fn get(&self, class: LightClass) -> f64 {
        match class {
            LightClass::Daylight => self.daylight,
            LightClass::Tungsten => self.tungsten,
            LightClass::Fluorescent => self.fluorescent,
            LightClass::Led => self.led,
        }
    }

    /// The four probabilities in [`LightClass::ALL`] order.
    #[must_use]
    pub fn to_array(&self) -> [f64; 4] {
        [self.daylight, self.tungsten, self.fluorescent, self.led]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normalises_and_validates() {
        let p = ClassProbabilities::new(2.0, 1.0, 1.0, 0.0).unwrap();
        assert_eq!(p.to_array(), [0.5, 0.25, 0.25, 0.0]);
        assert_eq!(p.get(LightClass::Tungsten), 0.25);
        assert!(matches!(
            ClassProbabilities::new(0.0, 0.0, 0.0, 0.0),
            Err(ClassProbabilityError::AllZero)
        ));
        assert!(matches!(
            ClassProbabilities::new(-0.1, 1.0, 0.0, 0.0),
            Err(ClassProbabilityError::Invalid {
                class: LightClass::Daylight,
                ..
            })
        ));
        assert!(ClassProbabilities::new(f64::NAN, 1.0, 0.0, 0.0).is_err());
        assert!(ClassProbabilities::new(f64::INFINITY, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn only_puts_everything_on_one_class() {
        for class in LightClass::ALL {
            let p = ClassProbabilities::only(class);
            assert_eq!(p.get(class), 1.0);
            assert_eq!(p.to_array().iter().sum::<f64>(), 1.0);
            assert_eq!(LightClass::ALL[class.index()], class);
        }
    }

    #[test]
    fn class_names_match_profile_json() {
        let json = serde_json::to_string(&LightClass::Fluorescent).unwrap();
        assert_eq!(json, "\"fluorescent\"");
        for class in LightClass::ALL {
            assert_eq!(
                format!("\"{}\"", class.name()),
                serde_json::to_string(&class).unwrap()
            );
        }
    }
}
