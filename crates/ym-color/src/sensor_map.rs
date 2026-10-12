//! The sensor map: class-blended, mired-interpolated 3x3 from white-balanced Sony RGB to white-balanced
//! ALEXA native RGB (stage C2, `docs/02-colour-pipeline.md` section C2).
//!
//! A profile lists, for each of four light classes (daylight, tungsten, fluorescent, LED), the
//! white-preserving 3x3 maps fitted at the class's reference lights, sorted by strictly increasing mired
//! (`1e6 / CCT`). At run time
//!
//! ```text
//! M_map = sum_k p_k * M_k(mired_scene)
//! ```
//!
//! where `p_k` are the class probabilities of the illuminant estimate and `M_k(mired)` interpolates
//! linearly in mired between the two bracketing entries of class `k`, clamped at the class's ends. Every
//! stored matrix has rows summing to 1 (`M * [1, 1, 1]^T = [1, 1, 1]^T`), and so does any convex
//! combination of them: the map preserves white by construction.
//!
//! The arithmetic is `runtime_ref.sensor_map` / `fit.blend_map` (`training/sensor_map/`); the tests
//! compare against the reference file to 1e-12.
//!
//! # Empty classes
//!
//! A profile may leave a class without entries (the schema asks for two or more, but tables built by
//! other tools may not). [`SensorMapTables::blend`] then drops that class and renormalises the others:
//! probability that sat on an empty class moves to the classes that have a table, in proportion to their
//! own probabilities; if those have none either, to the non-empty classes equally; with no table at all
//! the result is the identity map. The function never fails and the result stays white preserving.

use serde::{Deserialize, Serialize};

use crate::consts::linalg::{self, IDENTITY, Mat3};
use crate::spectral::{ClassProbabilities, LightClass, bracket_by_mired};

/// Largest allowed `|row sum - 1|` of a stored matrix (white preservation).
pub const WHITE_PRESERVATION_TOL: f64 = 1e-12;

/// One fitted map: the reference light it was fitted at, that light's mired coordinate and the 3x3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapEntry {
    /// Name of the reference light (for traceability), for example `daylight-6500K` or `FL7`.
    pub light: String,
    /// `1e6 / CCT` of the reference light.
    pub mired: f64,
    /// White-preserving 3x3, row-major, acting on white-balanced camera RGB.
    pub m: Mat3,
}

/// Why a set of map tables was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SensorMapError {
    /// A mired coordinate is not a positive number above the previous entry's.
    #[error(
        "{class}: entry {light:?} has mired {mired}, which is not positive and strictly above the previous entry"
    )]
    MiredOrder {
        /// The class name.
        class: &'static str,
        /// The reference light.
        light: String,
        /// The offending mired value.
        mired: f64,
    },
    /// A matrix entry is NaN or infinite.
    #[error("{class}: the matrix of {light:?} is not finite")]
    NonFinite {
        /// The class name.
        class: &'static str,
        /// The reference light.
        light: String,
    },
    /// A matrix does not map white to white.
    #[error(
        "{class}: the matrix of {light:?} does not preserve white (largest |row sum - 1| is {error:e})"
    )]
    NotWhitePreserving {
        /// The class name.
        class: &'static str,
        /// The reference light.
        light: String,
        /// Largest `|row sum - 1|`.
        error: f64,
    },
}

/// The `sensor_map.classes` object of a camera profile: the fitted maps of each light class.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorMapTables {
    /// CIE daylight reference lights, sorted by increasing mired.
    pub daylight: Vec<MapEntry>,
    /// Blackbody reference lights, sorted by increasing mired.
    pub tungsten: Vec<MapEntry>,
    /// Fluorescent reference lights, sorted by increasing mired.
    pub fluorescent: Vec<MapEntry>,
    /// LED reference lights, sorted by increasing mired.
    pub led: Vec<MapEntry>,
}

/// Largest `|row sum - 1|` of a matrix: how far it is from preserving white.
#[must_use]
pub fn max_white_error(m: &Mat3) -> f64 {
    linalg::row_sums(m)
        .iter()
        .fold(0.0_f64, |worst, s| worst.max((s - 1.0).abs()))
}

impl SensorMapTables {
    /// The entries of one class, sorted by increasing mired.
    #[must_use]
    pub fn class(&self, class: LightClass) -> &[MapEntry] {
        match class {
            LightClass::Daylight => &self.daylight,
            LightClass::Tungsten => &self.tungsten,
            LightClass::Fluorescent => &self.fluorescent,
            LightClass::Led => &self.led,
        }
    }

    /// `M_k(mired)`: the class's maps interpolated linearly in mired between the two bracketing
    /// entries, clamped to the first or last entry outside their range (a NaN mired gives the first).
    /// `None` when the class has no entries.
    #[must_use]
    pub fn class_map(&self, class: LightClass, mired: f64) -> Option<Mat3> {
        let entries = self.class(class);
        let (lo, hi, a) = bracket_by_mired(entries, mired, |e| e.mired)?;
        Some(linalg::lerp(&entries[lo].m, &entries[hi].m, a))
    }

    /// The effective class weights: `probs` with empty classes dropped and the rest renormalised (see
    /// the module notes). `None` when every class is empty.
    fn weights(&self, probs: &ClassProbabilities) -> Option<[f64; 4]> {
        let p = probs.to_array();
        let has_table: [bool; 4] = LightClass::ALL.map(|c| !self.class(c).is_empty());
        let count = has_table.iter().filter(|&&h| h).count();
        if count == 0 {
            return None;
        }
        let mass: f64 = p
            .iter()
            .zip(&has_table)
            .filter(|(_, h)| **h)
            .map(|(p, _)| *p)
            .sum();
        let mut w = [0.0; 4];
        for k in 0..4 {
            if has_table[k] {
                w[k] = if mass > 0.0 {
                    p[k] / mass
                } else {
                    1.0 / count as f64
                };
            }
        }
        Some(w)
    }

    /// The sensor map for a scene light: `M_map = sum_k p_k * M_k(mired)`.
    ///
    /// `mired` is `1e6 / CCT` of the scene light. White preserving: the rows of the result sum to 1
    /// (to the 1e-12 of [`WHITE_PRESERVATION_TOL`] and, for tables that pass
    /// [`SensorMapTables::validate`], to a few 1e-16). Empty classes are handled as described in the
    /// module notes; with no tables at all the identity is returned.
    #[must_use]
    pub fn blend(&self, probs: &ClassProbabilities, mired: f64) -> Mat3 {
        let Some(weights) = self.weights(probs) else {
            return IDENTITY;
        };
        let mut out = [[0.0; 3]; 3];
        for (class, w) in LightClass::ALL.into_iter().zip(weights) {
            if w == 0.0 {
                continue;
            }
            // `weights` only gives weight to classes with entries.
            if let Some(m) = self.class_map(class, mired) {
                for (out_row, row) in out.iter_mut().zip(&m) {
                    for (acc, v) in out_row.iter_mut().zip(row) {
                        *acc += w * v;
                    }
                }
            }
        }
        out
    }

    /// Checks every entry: finite matrices, white preservation within [`WHITE_PRESERVATION_TOL`], mired
    /// positive and strictly increasing inside each class.
    ///
    /// # Errors
    ///
    /// Returns the first problem found as a [`SensorMapError`].
    pub fn validate(&self) -> Result<(), SensorMapError> {
        for class in LightClass::ALL {
            let name = class.name();
            let mut previous = 0.0_f64;
            for e in self.class(class) {
                if !(e.mired.is_finite() && e.mired > previous) {
                    return Err(SensorMapError::MiredOrder {
                        class: name,
                        light: e.light.clone(),
                        mired: e.mired,
                    });
                }
                previous = e.mired;
                if !e.m.iter().flatten().all(|v| v.is_finite()) {
                    return Err(SensorMapError::NonFinite {
                        class: name,
                        light: e.light.clone(),
                    });
                }
                let error = max_white_error(&e.m);
                if error > WHITE_PRESERVATION_TOL {
                    return Err(SensorMapError::NotWhitePreserving {
                        class: name,
                        light: e.light.clone(),
                        error,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(light: &str, mired: f64, m: Mat3) -> MapEntry {
        MapEntry {
            light: light.to_string(),
            mired,
            m,
        }
    }

    /// A white-preserving matrix with distinguishable entries.
    fn wp(a: f64, b: f64) -> Mat3 {
        [
            [1.0 + a, -a, 0.0],
            [b, 1.0 - b - 0.5 * a, 0.5 * a],
            [0.0, 0.25 * b, 1.0 - 0.25 * b],
        ]
    }

    fn tables() -> SensorMapTables {
        SensorMapTables {
            daylight: vec![
                entry("d-a", 100.0, wp(0.10, 0.02)),
                entry("d-b", 200.0, wp(0.20, 0.04)),
            ],
            tungsten: vec![
                entry("t-a", 250.0, wp(0.30, 0.06)),
                entry("t-b", 300.0, wp(0.40, 0.08)),
                entry("t-c", 500.0, wp(0.60, 0.12)),
            ],
            fluorescent: vec![
                entry("f-a", 150.0, wp(-0.10, 0.01)),
                entry("f-b", 400.0, wp(-0.20, 0.03)),
            ],
            led: vec![
                entry("l-a", 150.0, wp(0.05, -0.02)),
                entry("l-b", 360.0, wp(0.07, -0.04)),
            ],
        }
    }

    fn close(a: &Mat3, b: &Mat3, tol: f64) -> bool {
        linalg::max_abs_diff(a, b) <= tol
    }

    #[test]
    fn class_maps_interpolate_linearly_in_mired() {
        let t = tables();
        t.validate().unwrap();
        // On an entry: the entry itself, bit for bit.
        assert_eq!(
            t.class_map(LightClass::Tungsten, 300.0),
            Some(t.tungsten[1].m)
        );
        // Halfway between two entries.
        let mid = t.class_map(LightClass::Tungsten, 400.0).unwrap();
        assert!(close(
            &mid,
            &linalg::lerp(&t.tungsten[1].m, &t.tungsten[2].m, 0.5),
            1e-15
        ));
        // A quarter of the way, between the first two.
        let q = t.class_map(LightClass::Tungsten, 262.5).unwrap();
        assert!(close(
            &q,
            &linalg::lerp(&t.tungsten[0].m, &t.tungsten[1].m, 0.25),
            1e-15
        ));
    }

    #[test]
    fn class_maps_clamp_at_the_ends() {
        let t = tables();
        assert_eq!(
            t.class_map(LightClass::Daylight, 10.0),
            Some(t.daylight[0].m)
        );
        assert_eq!(
            t.class_map(LightClass::Daylight, 100.0),
            Some(t.daylight[0].m)
        );
        assert_eq!(
            t.class_map(LightClass::Daylight, 200.0),
            Some(t.daylight[1].m)
        );
        assert_eq!(
            t.class_map(LightClass::Daylight, 1e6),
            Some(t.daylight[1].m)
        );
        assert_eq!(t.class_map(LightClass::Led, f64::NAN), Some(t.led[0].m));
        assert_eq!(
            t.class_map(LightClass::Led, f64::INFINITY),
            Some(t.led[1].m)
        );
    }

    #[test]
    fn blend_is_the_probability_weighted_sum() {
        let t = tables();
        let p = ClassProbabilities::new(0.5, 0.25, 0.125, 0.125).unwrap();
        let mired = 240.0;
        let m = t.blend(&p, mired);
        let mut expect = [[0.0; 3]; 3];
        for class in LightClass::ALL {
            let mk = t.class_map(class, mired).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    expect[i][j] += p.get(class) * mk[i][j];
                }
            }
        }
        assert!(close(&m, &expect, 1e-15));
        // One class with probability 1 is that class's map.
        for class in LightClass::ALL {
            assert_eq!(
                t.blend(&ClassProbabilities::only(class), mired),
                t.class_map(class, mired).unwrap()
            );
        }
    }

    #[test]
    fn empty_classes_are_dropped_and_the_rest_renormalised() {
        let mut t = tables();
        t.led.clear();
        // Mass on the empty class moves to the others in proportion to their probabilities.
        let p = ClassProbabilities::new(0.3, 0.3, 0.2, 0.2).unwrap();
        let m = t.blend(&p, 280.0);
        let expect = {
            let q = ClassProbabilities::new(0.3, 0.3, 0.2, 0.0).unwrap();
            t.blend(&q, 280.0)
        };
        assert!(close(&m, &expect, 1e-15));
        assert!(max_white_error(&m) < 1e-15);
        // All the mass on the empty class: the non-empty classes share equally.
        let only_led = t.blend(&ClassProbabilities::only(LightClass::Led), 280.0);
        let mut equal = [[0.0; 3]; 3];
        for class in [
            LightClass::Daylight,
            LightClass::Tungsten,
            LightClass::Fluorescent,
        ] {
            let mk = t.class_map(class, 280.0).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    equal[i][j] += mk[i][j] / 3.0;
                }
            }
        }
        assert!(close(&only_led, &equal, 1e-15));
        // No tables at all: identity.
        let none = SensorMapTables::default();
        assert_eq!(
            none.blend(&ClassProbabilities::only(LightClass::Daylight), 200.0),
            IDENTITY
        );
        assert_eq!(none.class_map(LightClass::Daylight, 200.0), None);
        none.validate().unwrap();
    }

    #[test]
    fn white_is_preserved_by_the_blend() {
        let t = tables();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            // SplitMix64: deterministic, no dependencies.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        };
        for _ in 0..2000 {
            let p = ClassProbabilities::new(next(), next(), next(), next())
                .unwrap_or(ClassProbabilities::only(LightClass::Daylight));
            let mired = next() * 700.0;
            let m = t.blend(&p, mired);
            assert!(max_white_error(&m) < 1e-14, "{p:?} {mired}");
        }
    }

    #[test]
    fn validate_rejects_bad_tables() {
        let mut t = tables();
        t.tungsten[1].mired = 250.0;
        assert!(matches!(
            t.validate(),
            Err(SensorMapError::MiredOrder {
                class: "tungsten",
                ..
            })
        ));
        let mut t = tables();
        t.led[0].mired = 0.0;
        assert!(matches!(
            t.validate(),
            Err(SensorMapError::MiredOrder { class: "led", .. })
        ));
        let mut t = tables();
        t.daylight[0].m[1][1] += 1e-9;
        assert!(matches!(
            t.validate(),
            Err(SensorMapError::NotWhitePreserving {
                class: "daylight",
                ..
            })
        ));
        let mut t = tables();
        t.fluorescent[1].m[0][0] = f64::NAN;
        assert!(matches!(
            t.validate(),
            Err(SensorMapError::NonFinite {
                class: "fluorescent",
                ..
            })
        ));
        // Just inside the tolerance passes.
        let mut t = tables();
        t.daylight[0].m[1][1] += 5e-13;
        t.validate().unwrap();
    }

    #[test]
    fn json_round_trip_rejects_unknown_fields() {
        let t = tables();
        let text = serde_json::to_string(&t).unwrap();
        let back: SensorMapTables = serde_json::from_str(&text).unwrap();
        // serde_json's default float parser can be one ulp off, so compare with a tolerance.
        for class in LightClass::ALL {
            assert_eq!(back.class(class).len(), t.class(class).len());
            for (a, b) in back.class(class).iter().zip(t.class(class)) {
                assert_eq!((&a.light, a.mired), (&b.light, b.mired));
                assert!(close(&a.m, &b.m, 1e-15));
            }
        }
        assert!(
            serde_json::from_str::<SensorMapTables>(
                r#"{"daylight":[],"tungsten":[],"fluorescent":[],"led":[],"mixed":[]}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<SensorMapTables>(
                r#"{"daylight":[],"tungsten":[],"fluorescent":[]}"#
            )
            .is_err()
        );
    }
}
