//! The exposure label of `ym_core::autoset::labels` against the training code (`docs/18` section 6.1).
//!
//! `tests/data/autoset/labels.json` was written by `python3 -m training.autoset.export --label-fixtures
//! crates/ym-core/tests/data/autoset/labels.json`: 18 cases (synthetic tables, the shoulder, and the edge cases of the
//! band rule, ties, the cumulative maximum, the range check and the extrapolation), each with the level Python's
//! `labels.exposure_label` gives (null where it is NaN). The grid must be numpy's `linspace` bit for bit, and the
//! quantile table must agree with numpy's default `linear` quantiles on the same values.

use serde_json::Value;
use ym_core::autoset::labels::{exposure_label, quantile_grid, quantile_table};

const FIXTURES: &str = include_str!("data/autoset/labels.json");

/// The tolerance of the level (EV), the one of `training/autoset/dataset.py`.
const TOLERANCE_EV: f64 = 1e-9;

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_f64().expect("a number"))
        .collect()
}

#[test]
fn the_grid_is_numpys_linspace_bit_for_bit() {
    let doc: Value = serde_json::from_str(FIXTURES).expect("the fixture is JSON");
    let grid = numbers(&doc["grid"]);
    let ours = quantile_grid();
    assert_eq!(ours.len(), grid.len());
    for (i, (a, b)) in ours.iter().zip(&grid).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "grid point {i}: {a} vs numpy {b}");
    }
}

#[test]
fn the_exposure_label_reproduces_the_training_code_to_1e_9() {
    let doc: Value = serde_json::from_str(FIXTURES).expect("the fixture is JSON");
    assert_eq!(doc["format"], "yellowmoon-autoset-label-fixtures");
    let cases = doc["cases"].as_array().expect("cases");
    let mut with_level = 0;
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let scene = numbers(&case["scene_q"]);
        let jpeg = numbers(&case["jpeg_q"]);
        let got = exposure_label(&scene, &jpeg);
        match (got, case["level_ev"].as_f64()) {
            (None, None) => {}
            (Some(ours), Some(python)) => {
                with_level += 1;
                assert!(
                    (ours - python).abs() <= TOLERANCE_EV,
                    "{name}: {ours} vs Python {python}"
                );
            }
            (ours, python) => panic!("{name}: {ours:?} vs Python {python:?}"),
        }
    }
    // Both sides of the range check and the band rule are exercised: several cases have a level, several do not.
    assert!(with_level >= 10, "{with_level} cases with a level");
    // The dark scene, the too-bright one and the flat table have no level.
    assert_eq!(cases.len() - with_level, 3, "the cases without a level");
}

#[test]
fn the_quantiles_agree_with_numpys_default_method() {
    let doc: Value = serde_json::from_str(FIXTURES).expect("the fixture is JSON");
    let mut values = numbers(&doc["quantiles"]["values"]);
    let expected = numbers(&doc["quantiles"]["expected"]);
    let grid = quantile_grid();
    let ours = quantile_table(&mut values, &grid);
    assert_eq!(ours.len(), expected.len());
    for (i, (a, b)) in ours.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "point {i}: {a} vs numpy {b}"
        );
    }
}
