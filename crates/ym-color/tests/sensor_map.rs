//! M1-05 runtime: the class-blended, mired-interpolated sensor map on the shipped profiles.
//!
//! * `M_map = sum_k p_k M_k(mired)` against `runtime_ref.sensor_map` / `fit.blend_map` through the
//!   reference file (1e-12);
//! * white preservation to 1e-12 over random class probabilities and mireds, on all 11 profiles;
//! * entries are reproduced exactly at their own mired, the blend is continuous and stays inside the
//!   range of the stored matrices;
//! * empty classes.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::consts::linalg::{self, IDENTITY, Mat3};
use ym_color::sensor_map::{SensorMapTables, WHITE_PRESERVATION_TOL, max_white_error};
use ym_color::spectral::{ClassProbabilities, LightClass};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn profile_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

fn tables(id: &str) -> SensorMapTables {
    let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap()
}

fn as_mat(v: &Value) -> Mat3 {
    let rows = v.as_array().unwrap();
    std::array::from_fn(|i| {
        let row = rows[i].as_array().unwrap();
        std::array::from_fn(|j| row[j].as_f64().unwrap())
    })
}

/// Deterministic SplitMix64 in [0, 1).
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn random_probs(rng: &mut Rng) -> ClassProbabilities {
    // Mix of dense draws and sparse ones (zeros and one-hot), the shapes an estimator produces.
    let mut p = [
        rng.next_f64(),
        rng.next_f64(),
        rng.next_f64(),
        rng.next_f64(),
    ];
    let zeros = (rng.next_f64() * 4.0) as usize; // 0..=3 classes zeroed
    for slot in p.iter_mut().take(zeros) {
        *slot = 0.0;
    }
    if p.iter().all(|&v| v == 0.0) {
        p[3] = 1.0;
    }
    p.rotate_left((rng.next_f64() * 4.0) as usize);
    ClassProbabilities::new(p[0], p[1], p[2], p[3]).unwrap()
}

#[test]
fn blend_reproduces_the_reference_cases() {
    let r: Value = serde_json::from_str(include_str!("data/spectral_reference.json")).unwrap();
    let t = tables("sony-ilce-7rm4");
    let mut worst = 0.0_f64;
    for case in r["sensor_map_blend"].as_array().unwrap() {
        let probs = case["probs"].as_object().unwrap();
        let get = |k: &str| probs.get(k).map_or(0.0, |v| v.as_f64().unwrap());
        let p = ClassProbabilities::new(
            get("daylight"),
            get("tungsten"),
            get("fluorescent"),
            get("led"),
        )
        .unwrap();
        let mired = case["mired"].as_f64().unwrap();
        let got = t.blend(&p, mired);
        let expect = as_mat(&case["m"]);
        let d = linalg::max_abs_diff(&got, &expect);
        assert!(d <= 1e-12, "{probs:?} at {mired} mired: {d:e}");
        worst = worst.max(d);
        // The reference rows sum to 1 as stored.
        let sums = as_mat(&case["m"]).map(|row| row.iter().sum::<f64>());
        assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-12));
    }
    eprintln!(
        "sensor-map blend vs runtime_ref: max difference {worst:.2e} over {} cases",
        r["sensor_map_blend"].as_array().unwrap().len()
    );
}

#[test]
fn white_is_preserved_on_every_profile_over_random_inputs() {
    let mut rng = Rng(0x5EED_5301);
    let mut worst = 0.0_f64;
    let mut cases = 0;
    for id in profile_ids() {
        let t = tables(&id);
        t.validate().unwrap_or_else(|e| panic!("{id}: {e}"));
        for _ in 0..4000 {
            let p = random_probs(&mut rng);
            // Beyond both ends of every class (mired 90..367 for the reference lights) and inside.
            let mired = rng.next_f64() * 600.0;
            let m = t.blend(&p, mired);
            let err = max_white_error(&m);
            assert!(
                err <= WHITE_PRESERVATION_TOL,
                "{id}: {p:?} at {mired} mired: row sums off by {err:e}"
            );
            worst = worst.max(err);
            // Applied to white it returns white.
            let w = linalg::mul_vec(&m, [1.0, 1.0, 1.0]);
            assert!(w.iter().all(|c| (c - 1.0).abs() <= 1e-12), "{id}: {w:?}");
            cases += 1;
        }
    }
    eprintln!(
        "white preservation: max |row sum - 1| = {worst:.2e} over {cases} random (profile, probabilities, mired) cases"
    );
}

#[test]
fn every_reference_light_is_reproduced_exactly() {
    for id in profile_ids() {
        let t = tables(&id);
        for class in LightClass::ALL {
            let entries = t.class(class);
            assert!(entries.len() >= 2, "{id} {class:?}");
            for e in entries {
                let m = t.blend(&ClassProbabilities::only(class), e.mired);
                assert_eq!(m, e.m, "{id} {class:?} {}", e.light);
            }
            // Clamped beyond the ends.
            let (first, last) = (entries.first().unwrap(), entries.last().unwrap());
            assert_eq!(
                t.blend(&ClassProbabilities::only(class), first.mired - 50.0),
                first.m
            );
            assert_eq!(
                t.blend(&ClassProbabilities::only(class), last.mired + 50.0),
                last.m
            );
        }
    }
}

#[test]
fn blend_is_continuous_and_stays_inside_the_stored_range() {
    let mut rng = Rng(0x5EED_5302);
    for id in profile_ids() {
        let t = tables(&id);
        // Per-element range over every stored matrix.
        let mut lo = [[f64::INFINITY; 3]; 3];
        let mut hi = [[f64::NEG_INFINITY; 3]; 3];
        for class in LightClass::ALL {
            for e in t.class(class) {
                for i in 0..3 {
                    for j in 0..3 {
                        lo[i][j] = lo[i][j].min(e.m[i][j]);
                        hi[i][j] = hi[i][j].max(e.m[i][j]);
                    }
                }
            }
        }
        // Largest slope of any class's piecewise linear map, per unit mired.
        let mut slope = 0.0_f64;
        for class in LightClass::ALL {
            for w in t.class(class).windows(2) {
                slope =
                    slope.max(linalg::max_abs_diff(&w[0].m, &w[1].m) / (w[1].mired - w[0].mired));
            }
        }
        for _ in 0..500 {
            let p = random_probs(&mut rng);
            let mired = rng.next_f64() * 500.0;
            let m = t.blend(&p, mired);
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        m[i][j] >= lo[i][j] - 1e-12 && m[i][j] <= hi[i][j] + 1e-12,
                        "{id}: element {i}{j} = {} outside [{}, {}]",
                        m[i][j],
                        lo[i][j],
                        hi[i][j]
                    );
                }
            }
            let delta = 1e-4;
            let m2 = t.blend(&p, mired + delta);
            assert!(
                linalg::max_abs_diff(&m, &m2) <= slope * delta * (1.0 + 1e-9) + 1e-15,
                "{id}: jump at {mired} mired"
            );
        }
    }
}

#[test]
fn a_convex_mix_of_classes_lies_between_the_pure_classes() {
    let t = tables("sony-ilce-7rm4");
    for mired in [100.0, 180.0, 250.0, 310.0, 360.0] {
        let a = t.blend(&ClassProbabilities::only(LightClass::Daylight), mired);
        let b = t.blend(&ClassProbabilities::only(LightClass::Led), mired);
        let mix = t.blend(&ClassProbabilities::new(3.0, 0.0, 0.0, 1.0).unwrap(), mired);
        let expect = linalg::lerp(&a, &b, 0.25);
        assert!(linalg::max_abs_diff(&mix, &expect) < 1e-14);
    }
}

#[test]
fn empty_classes_on_a_real_profile() {
    let full = tables("sony-ilce-7rm4");
    let mut t = full.clone();
    t.led.clear();
    t.fluorescent.clear();
    let mired = 200.0;
    // Probability on the empty classes moves to the others, in proportion.
    let p = ClassProbabilities::new(0.2, 0.2, 0.3, 0.3).unwrap();
    let got = t.blend(&p, mired);
    let expect = full.blend(&ClassProbabilities::new(0.5, 0.5, 0.0, 0.0).unwrap(), mired);
    assert!(linalg::max_abs_diff(&got, &expect) < 1e-14);
    assert!(max_white_error(&got) < WHITE_PRESERVATION_TOL);
    // All the probability on an empty class: the others share it equally.
    let got = t.blend(&ClassProbabilities::only(LightClass::Led), mired);
    let expect = full.blend(&ClassProbabilities::new(1.0, 1.0, 0.0, 0.0).unwrap(), mired);
    assert!(linalg::max_abs_diff(&got, &expect) < 1e-14);
    // Nothing at all: identity.
    let none = SensorMapTables::default();
    assert_eq!(
        none.blend(&ClassProbabilities::only(LightClass::Daylight), mired),
        IDENTITY
    );
}

#[test]
fn stored_maps_are_near_the_identity() {
    // Sanity on the content: white-balanced Sony RGB is already close to white-balanced ALEXA RGB, so the
    // maps are near the identity (largest deviation of any element well below 1).
    for id in profile_ids() {
        let t = tables(&id);
        for class in LightClass::ALL {
            for e in t.class(class) {
                assert!(
                    linalg::max_abs_diff(&e.m, &IDENTITY) < 1.5,
                    "{id} {}",
                    e.light
                );
            }
        }
    }
}
