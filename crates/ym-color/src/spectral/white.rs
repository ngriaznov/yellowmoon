//! `w_A`: the ALEXA estimate's white for a light (`docs/02-colour-pipeline.md` sections C3 and C4).
//!
//! `w_A(E) = S_A^T E`, normalised to green = 1, as `[r, 1, b]`. The profile precomputes it, so the app
//! needs no CIE data:
//!
//! * blackbody 1500 to 25000 K and CIE daylight 4000 to 25000 K every 10 K, interpolated
//!   **geometrically in mired** (`w_i^(1-a) * w_(i+1)^a`, green stays exactly 1). The scene-light rule
//!   (the rawtoaces convention) uses the blackbody table below 4000 K and the daylight table from
//!   4000 K ([`AlexaWhite::locus_white`]);
//! * the fluorescent and LED reference lights as `(mired, w)` lists, interpolated linearly in mired
//!   between the bracketing entries and clamped at the ends.
//!
//! [`AlexaWhite::scene_white`] blends the two by class probability (C3): the daylight and tungsten
//! classes follow the CCT rule above, the fluorescent and LED classes their lists.

use serde::{Deserialize, Serialize};

use super::grid::{Grid, bracket_by_mired};
use super::light::{ClassProbabilities, LightClass};
use super::{
    BLACKBODY_START_K, DAYLIGHT_START_K, GRID_STEP_K, KELVIN_SPLIT_K, SpectralError, TABLE_END_K,
    check_grid_range,
};

/// One reference light of a class list: its name, mired coordinate and `w_A`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteEntry {
    /// Name of the reference light (for traceability), for example `FL7` or `LED-B5`.
    pub light: String,
    /// `1e6 / CCT` of the light.
    pub mired: f64,
    /// `w_A` as `[r, 1, b]`.
    pub w: [f64; 3],
}

/// The `alexa_white` section of a camera profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlexaWhite {
    /// Free-text description of the interpolation rules.
    pub interpolation: String,
    /// Blackbody `w_A`, 1500 to 25000 K, 10 K step.
    pub blackbody: Grid<[f64; 3]>,
    /// CIE daylight `w_A`, 4000 to 25000 K, 10 K step.
    pub daylight: Grid<[f64; 3]>,
    /// Fluorescent reference lights, sorted by increasing mired.
    pub fluorescent: Vec<WhiteEntry>,
    /// LED reference lights, sorted by increasing mired.
    pub led: Vec<WhiteEntry>,
}

/// Interpolates a class list linearly in mired, clamped at the ends. `None` for an empty list.
fn interp_entries(entries: &[WhiteEntry], mired: f64) -> Option<[f64; 3]> {
    let (lo, hi, a) = bracket_by_mired(entries, mired, |e| e.mired)?;
    let (x, y) = (entries[lo].w, entries[hi].w);
    Some(std::array::from_fn(|k| (1.0 - a) * x[k] + a * y[k]))
}

impl AlexaWhite {
    /// `w_A` of the blackbody table at `cct_k` (geometric in mired, clamped to 1500 to 25000 K).
    #[must_use]
    pub fn blackbody_white(&self, cct_k: f64) -> [f64; 3] {
        self.blackbody.geometric_mired(cct_k)
    }

    /// `w_A` of the CIE daylight table at `cct_k` (geometric in mired, clamped to 4000 to 25000 K).
    #[must_use]
    pub fn daylight_white(&self, cct_k: f64) -> [f64; 3] {
        self.daylight.geometric_mired(cct_k)
    }

    /// `w_A` of the scene-light model at `cct_k`: blackbody below 4000 K, CIE daylight from 4000 K.
    /// This is also `E_user` of stage C4 (the user's CCT, same rule).
    #[must_use]
    pub fn locus_white(&self, cct_k: f64) -> [f64; 3] {
        if cct_k < KELVIN_SPLIT_K {
            self.blackbody_white(cct_k)
        } else {
            self.daylight_white(cct_k)
        }
    }

    /// `w_A` of one light class at a mired coordinate. The fluorescent and LED classes interpolate
    /// their reference lights (clamped at the ends); the daylight and tungsten classes are the
    /// blackbody/daylight model at `1e6 / mired` K ([`AlexaWhite::locus_white`]). A fluorescent or LED
    /// class with no reference lights falls back to that model as well.
    #[must_use]
    pub fn class_white(&self, class: LightClass, mired: f64) -> [f64; 3] {
        let list = match class {
            LightClass::Fluorescent => interp_entries(&self.fluorescent, mired),
            LightClass::Led => interp_entries(&self.led, mired),
            LightClass::Daylight | LightClass::Tungsten => None,
        };
        list.unwrap_or_else(|| self.locus_white(1e6 / mired))
    }

    /// `w_A(E_scene)` for stage C3: the class whites blended by class probability,
    /// `sum_k p_k * w_k(mired)` with `mired = 1e6 / cct_k`, and green set to exactly 1.
    #[must_use]
    pub fn scene_white(&self, cct_k: f64, probs: &ClassProbabilities) -> [f64; 3] {
        let mired = 1e6 / cct_k;
        let mut w = [0.0; 3];
        for class in LightClass::ALL {
            let p = probs.get(class);
            if p == 0.0 {
                continue;
            }
            let wk = self.class_white(class, mired);
            for (acc, v) in w.iter_mut().zip(wk) {
                *acc += p * v;
            }
        }
        [w[0], 1.0, w[2]]
    }

    /// Checks the structure the schema cannot express: grid ranges, positive whites with green exactly
    /// 1, class lists sorted by strictly increasing mired.
    ///
    /// # Errors
    ///
    /// Returns the first problem found as a [`SpectralError::InvalidTable`].
    pub fn validate(&self) -> Result<(), SpectralError> {
        check_grid_range(
            "alexa_white.blackbody",
            &self.blackbody,
            BLACKBODY_START_K,
            TABLE_END_K,
            GRID_STEP_K,
        )?;
        check_grid_range(
            "alexa_white.daylight",
            &self.daylight,
            DAYLIGHT_START_K,
            TABLE_END_K,
            GRID_STEP_K,
        )?;
        let bad = |table: &str, reason: String| SpectralError::InvalidTable {
            table: table.to_string(),
            reason,
        };
        for (name, grid) in [
            ("alexa_white.blackbody", &self.blackbody),
            ("alexa_white.daylight", &self.daylight),
        ] {
            for (i, w) in grid.values().iter().enumerate() {
                if w[1] != 1.0 || w[0] <= 0.0 || w[2] <= 0.0 {
                    return Err(bad(
                        name,
                        format!("entry {i} is {w:?}: w_A must be positive with green exactly 1"),
                    ));
                }
            }
        }
        for (name, list) in [
            ("alexa_white.fluorescent", &self.fluorescent),
            ("alexa_white.led", &self.led),
        ] {
            let mut previous = f64::NEG_INFINITY;
            for e in list {
                if !(e.mired.is_finite() && e.mired > previous) {
                    return Err(bad(
                        name,
                        format!("{}: mired {} is not strictly increasing", e.light, e.mired),
                    ));
                }
                previous = e.mired;
                if e.w[1] != 1.0
                    || !(e.w[0] > 0.0 && e.w[2] > 0.0 && e.w[0].is_finite() && e.w[2].is_finite())
                {
                    return Err(bad(
                        name,
                        format!(
                            "{}: w {:?} must be positive with green exactly 1",
                            e.light, e.w
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> AlexaWhite {
        // w_A = [exp(a*mired), 1, exp(-b*mired)] is geometric-exact in mired.
        let make = |t: f64| {
            let m = 1e6 / t;
            [(0.004 * m).exp(), 1.0, (-0.003 * m).exp()]
        };
        let bb: Vec<[f64; 3]> = (0..2351)
            .map(|i| make(1500.0 + 10.0 * f64::from(i)))
            .collect();
        let dy: Vec<[f64; 3]> = (0..2101)
            .map(|i| make(4000.0 + 10.0 * f64::from(i)))
            .collect();
        let entry = |light: &str, m: f64, shift: f64| WhiteEntry {
            light: light.to_string(),
            mired: m,
            w: [(0.004 * m + shift).exp(), 1.0, (-0.003 * m).exp()],
        };
        AlexaWhite {
            interpolation: "test".to_string(),
            blackbody: Grid::new(1500.0, 10.0, bb).unwrap(),
            daylight: Grid::new(4000.0, 10.0, dy).unwrap(),
            fluorescent: vec![entry("FL-a", 150.0, 0.1), entry("FL-b", 300.0, 0.2)],
            led: vec![entry("LED-a", 150.0, -0.1), entry("LED-b", 360.0, -0.2)],
        }
    }

    #[test]
    fn rule_switches_at_4000_k() {
        let t = tables();
        t.validate().unwrap();
        assert_eq!(t.locus_white(3999.9), t.blackbody_white(3999.9));
        assert_eq!(t.locus_white(4000.0), t.daylight_white(4000.0));
        // Geometric-exact tables are reproduced between nodes (to rounding).
        let m: f64 = 1e6 / 3333.3;
        let w = t.locus_white(3333.3);
        assert!((w[0] - (0.004 * m).exp()).abs() < 1e-9 * w[0]);
        assert!((w[2] - (-0.003 * m).exp()).abs() < 1e-9);
        assert_eq!(w[1], 1.0);
    }

    #[test]
    fn class_lists_interpolate_linearly_in_mired_and_clamp() {
        let t = tables();
        let at = |class, m| t.class_white(class, m);
        assert_eq!(at(LightClass::Fluorescent, 100.0), t.fluorescent[0].w);
        assert_eq!(at(LightClass::Fluorescent, 400.0), t.fluorescent[1].w);
        let mid = at(LightClass::Fluorescent, 225.0);
        for (k, v) in mid.iter().enumerate() {
            let expect = 0.5 * (t.fluorescent[0].w[k] + t.fluorescent[1].w[k]);
            assert!((v - expect).abs() < 1e-15);
        }
        assert_eq!(at(LightClass::Led, 150.0), t.led[0].w);
        assert_eq!(at(LightClass::Led, f64::NAN), t.led[0].w);
        // The daylight and tungsten classes follow the CCT rule.
        assert_eq!(
            at(LightClass::Daylight, 1e6 / 6500.0),
            t.locus_white(1e6 / (1e6 / 6500.0))
        );
        assert_eq!(
            at(LightClass::Tungsten, 1e6 / 3000.0),
            t.locus_white(1e6 / (1e6 / 3000.0))
        );
    }

    #[test]
    fn scene_white_blends_by_class_probability() {
        let t = tables();
        let cct = 4100.0;
        let m = 1e6 / cct;
        // All daylight or tungsten: exactly the CCT rule.
        for class in [LightClass::Daylight, LightClass::Tungsten] {
            assert_eq!(
                t.scene_white(cct, &ClassProbabilities::only(class)),
                t.locus_white(1e6 / m)
            );
        }
        // All LED: the LED list at that mired.
        assert_eq!(
            t.scene_white(cct, &ClassProbabilities::only(LightClass::Led)),
            t.class_white(LightClass::Led, m)
        );
        // A mix is the convex combination, green exactly 1.
        let p = ClassProbabilities::new(0.4, 0.1, 0.2, 0.3).unwrap();
        let w = t.scene_white(cct, &p);
        assert_eq!(w[1], 1.0);
        let expect: Vec<f64> = (0..3)
            .map(|k| {
                p.get(LightClass::Daylight) * t.locus_white(1e6 / m)[k]
                    + p.get(LightClass::Tungsten) * t.locus_white(1e6 / m)[k]
                    + p.get(LightClass::Fluorescent) * t.class_white(LightClass::Fluorescent, m)[k]
                    + p.get(LightClass::Led) * t.class_white(LightClass::Led, m)[k]
            })
            .collect();
        assert!((w[0] - expect[0]).abs() < 1e-14 && (w[2] - expect[2]).abs() < 1e-14);
    }

    #[test]
    fn empty_class_lists_fall_back_to_the_model() {
        let mut t = tables();
        t.led.clear();
        assert_eq!(t.class_white(LightClass::Led, 250.0), t.locus_white(4000.0));
    }

    #[test]
    fn validate_catches_structural_errors() {
        let mut t = tables();
        t.fluorescent[1].mired = t.fluorescent[0].mired;
        assert!(t.validate().is_err());
        let mut t = tables();
        t.led[0].w[1] = 0.999;
        assert!(t.validate().is_err());
        let mut t = tables();
        t.daylight = Grid::new(4000.0, 10.0, vec![[1.0, 1.0, 1.0]; 2100]).unwrap();
        assert!(t.validate().is_err());
    }
}
