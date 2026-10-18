//! Parity of the Rust `autoset` model with the training code (`docs/18` §6.4): every vector that
//! `python3 -m training.autoset.export` wrote from the stored f32 weights (numpy f64) is reproduced to 1e-9:
//! the raw member outputs, the ensemble mean and mixture variance, and the §4.4 application.
//!
//! The vectors and the small test model live in `tests/data/autoset/`; the shipped model gets its own
//! vectors file from the same exporter once it is trained.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_color::autoset::{AutosetModel, FEATURE_COUNT, Prediction, apply};

const TOLERANCE: f64 = 1e-9;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/autoset")
}

fn number(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what} is not a number: {v}"))
}

fn close(got: f64, want: f64, what: &str) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= TOLERANCE * scale,
        "{what}: {got} vs {want} (difference {:e})",
        got - want
    );
}

fn vectors(file: &str) -> Value {
    let text = std::fs::read_to_string(data_dir().join(file)).expect("the vectors file");
    let doc: Value = serde_json::from_str(&text).expect("the vectors are JSON");
    assert_eq!(doc["format"], "yellowmoon-autoset-vectors");
    assert_eq!(doc["format_version"], 1);
    doc
}

fn check(doc: &Value) -> usize {
    let model_file = doc["model_file"].as_str().expect("model_file");
    let name = Path::new(model_file)
        .file_name()
        .expect("a file name")
        .to_owned();
    let model = AutosetModel::load(&data_dir().join(name)).expect("the model loads");
    assert_eq!(model.version(), doc["model_version"].as_str().unwrap());
    assert_eq!(model.sha256(), doc["model_sha256"].as_str().unwrap());
    let constants = &doc["constants"];
    close(
        model.contrast_limit(),
        number(&constants["contrast_limit"], "contrast_limit"),
        "contrast_limit",
    );
    close(
        ym_color::exposure::CAMERA_GAIN_EV,
        number(&constants["camera_gain_ev"], "camera_gain_ev"),
        "camera_gain_ev",
    );
    close(
        ym_color::exposure::EV_LIMIT,
        number(&constants["ev_limit"], "ev_limit"),
        "ev_limit",
    );

    let cases = doc["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let label = case["name"].as_str().unwrap_or("?");
        let input: Vec<f64> = case["input"]
            .as_array()
            .expect("input")
            .iter()
            .map(|v| number(v, "input"))
            .collect();
        let x: [f64; FEATURE_COUNT] = input.try_into().expect("39 inputs");
        let key = number(&case["key"], "key");

        let members = model.member_outputs(&x);
        let want_members = case["members"].as_array().expect("members");
        assert_eq!(members.len(), want_members.len(), "{label}: member count");
        for (i, (got, want)) in members.iter().zip(want_members).enumerate() {
            for (j, w) in want.as_array().expect("four outputs").iter().enumerate() {
                close(
                    got[j],
                    number(w, "member output"),
                    &format!("{label} member {i} output {j}"),
                );
            }
        }

        let p: Prediction = model.predict(&x);
        let c = &case["combined"];
        close(
            p.key_target,
            number(&c["key_mean"], "key_mean"),
            &format!("{label} key_mean"),
        );
        close(
            p.key_var,
            number(&c["key_var"], "key_var"),
            &format!("{label} key_var"),
        );
        close(
            p.contrast,
            number(&c["contrast_mean"], "contrast_mean"),
            &format!("{label} contrast_mean"),
        );
        close(
            p.contrast_var,
            number(&c["contrast_var"], "contrast_var"),
            &format!("{label} contrast_var"),
        );

        let a = apply(&p, key, &model);
        let w = &case["apply"];
        close(
            a.ev_before_guard,
            number(&w["ev"], "ev"),
            &format!("{label} ev"),
        );
        close(
            a.ev_weight,
            number(&w["ev_confidence"], "ev_confidence"),
            &format!("{label} ev_confidence"),
        );
        close(
            a.contrast,
            number(&w["contrast"], "contrast"),
            &format!("{label} contrast"),
        );
        close(
            a.contrast_weight,
            number(&w["contrast_confidence"], "contrast_confidence"),
            &format!("{label} contrast_confidence"),
        );
    }
    cases.len()
}

#[test]
fn the_test_model_reproduces_the_training_code() {
    let n = check(&vectors("vectors.json"));
    assert!(n >= 20, "only {n} vectors");
}
