//! The data files of the Before view (camera default rendering, `docs/02` section The Before view):
//! `data/cameras/color-matrices.json` and `data/cameras/camera-default-tone.json` validate against
//! their JSON schemas, the matrix table is what the embedded copy and the loader say, every body's
//! as-shot neutral renders neutral, and the tone file is pinned to the constants in `ym_color`.

mod common;

use common::{data_root, read_json, repo_root};
use jsonschema::Validator;
use serde_json::Value;
use ym_color::camera_default::{
    CameraDefault, CameraTone, TONE_K, TONE_N, TONE_S, TONE_SIGMA, TONE_WHITE,
};
use ym_color::consts::linalg;
use ym_color::encode::srgb_encode;
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_profiles::color_matrices::{ColorMatrices, EMBEDDED_JSON, MatrixMatch};
use ym_profiles::registry::Registry;

fn validator(schema_file: &str) -> Validator {
    let schema = read_json(
        &repo_root()
            .join("crates/ym-profiles/schema")
            .join(schema_file),
    );
    jsonschema::validator_for(&schema)
        .unwrap_or_else(|e| panic!("{schema_file} is not a valid schema: {e}"))
}

fn assert_valid(v: &Validator, instance: &Value, label: &str) {
    let errs: Vec<String> = v
        .iter_errors(instance)
        .take(8)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect();
    assert!(
        errs.is_empty(),
        "{label} does not validate:\n{}",
        errs.join("\n")
    );
}

fn matrices_file() -> Value {
    read_json(&data_root().join("cameras/color-matrices.json"))
}

fn tone_file() -> Value {
    read_json(&data_root().join("cameras/camera-default-tone.json"))
}

#[test]
fn both_files_validate_against_their_schemas() {
    assert_valid(
        &validator("color-matrices.schema.json"),
        &matrices_file(),
        "color-matrices.json",
    );
    assert_valid(
        &validator("camera-default-tone.schema.json"),
        &tone_file(),
        "camera-default-tone.json",
    );
}

#[test]
fn the_embedded_copy_is_the_file_and_it_loads() {
    let on_disk = std::fs::read_to_string(ColorMatrices::data_path(data_root())).unwrap();
    assert_eq!(EMBEDDED_JSON, on_disk);
    let t = ColorMatrices::from_json(&on_disk).unwrap();
    assert_eq!(t.bodies().len(), 42);
    assert_eq!(t.fallback().model, "ILCE-7RM4");
}

#[test]
fn the_matrix_table_agrees_with_the_registry_and_the_exclusions() {
    let table = ColorMatrices::shipped();
    let registry: Registry = {
        let text = std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap();
        Registry::parse(&text).unwrap()
    };
    let doc = matrices_file();
    let excluded: Vec<&str> = doc["excluded"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["model"].as_str().unwrap())
        .collect();
    assert_eq!(
        excluded,
        ["ILCE-7M5", "ILCE-7RM6", "ILCE-1M2", "ILCE-3000"],
        "the four draft entries that are prefix matches or have no real entry"
    );
    for body in &registry.bodies {
        let hit = table.lookup("SONY", &body.model, Some(body.model_id));
        if excluded.contains(&body.model.as_str()) {
            // No matrix of its own: the A7R IV's.
            assert_eq!(hit.how, MatrixMatch::Fallback, "{}", body.model);
            assert_eq!(hit.body.model, "ILCE-7RM4");
        } else {
            // Every other registry body has its own entry, found by ID, and the model string agrees.
            assert_eq!(hit.how, MatrixMatch::ModelId, "{}", body.model);
            assert_eq!(hit.body.model, body.model);
            assert_eq!(hit.body.model_id, body.model_id);
            assert_eq!(
                table.lookup("SONY", &body.model, None).body.model,
                body.model
            );
        }
    }
    // A body that is not in the table at all uses the A7R IV matrix.
    let unknown = table.lookup("SONY", "ILCE-9999", Some(65000));
    assert_eq!(unknown.how, MatrixMatch::Fallback);
    assert_eq!(unknown.body, table.fallback());
}

#[test]
fn an_as_shot_neutral_renders_neutral_for_every_body_in_the_table() {
    let table = ColorMatrices::shipped();
    // Several as-shot whites (daylight-like, tungsten-like, strongly green), each with initial gains
    // that differ from them, as they do when our estimate is not the camera's.
    let whites = [[2.05, 1.0, 1.48], [1.2, 1.0, 2.6], [2.9, 1.0, 1.05]];
    let initial = [[1.9, 1.0, 1.6], [1.0, 1.0, 1.0], [2.2, 1.0, 1.9]];
    let gain = CAMERA_GAIN_EV.exp2();
    for body in table.bodies() {
        for (gs, g0) in whites.iter().zip(&initial) {
            let cd = CameraDefault::new(&body.xyz_d65_to_camera, *gs, *g0).unwrap();
            // A white surface records 1 / gs in the raw data; the raw stages multiply by g0.
            let cam_in = [g0[0] / gs[0], g0[1] / gs[1], g0[2] / gs[2]];
            let lin = cd.scene_linear_rec709(cam_in);
            for c in lin {
                assert!((c - gain).abs() < 1e-12, "{}: {lin:?}", body.model);
            }
            // Any neutral level of that white stays neutral through the curve and the encode.
            for level in [0.004, 0.03, 0.18, 0.5, 0.9] {
                let out = cd.display_linear_rec709(cam_in.map(|c| c * level));
                assert!(
                    (out[0] - out[1]).abs() < 1e-12 && (out[1] - out[2]).abs() < 1e-12,
                    "{} level {level}: {out:?}",
                    body.model
                );
                let codes = out.map(|y| (srgb_encode(y) * 255.0).round());
                assert!(codes[0] == codes[1] && codes[1] == codes[2], "{codes:?}");
            }
            // Middle grey after the gain lands on code 119.7.
            let grey = cd.display_linear_rec709(cam_in.map(|c| c * 0.18 / gain));
            let code = srgb_encode(grey[1]) * 255.0;
            assert!((code - 119.7).abs() < 0.5, "{}: {code}", body.model);
        }
    }
}

#[test]
fn every_matrix_has_the_shape_of_a_colour_matrix() {
    for body in ColorMatrices::shipped().bodies() {
        let m = &body.xyz_d65_to_camera;
        // Normalised, the rows map D65 white to camera 1 and the inverse is a real, well-conditioned
        // camera-to-XYZ matrix: positive determinant, no huge entries.
        assert!(
            linalg::det(m) > 0.0,
            "{}: det {}",
            body.model,
            linalg::det(m)
        );
        let cd = CameraDefault::new(m, [1.0; 3], [1.0; 3]).unwrap();
        for v in cd.matrix().iter().flatten() {
            assert!(v.abs() < 40.0, "{}: {v}", body.model);
        }
    }
}

#[test]
fn the_tone_file_is_pinned_to_the_constants_in_code() {
    let doc = tone_file();
    let curve = &doc["curve"];
    assert_eq!(curve["n"].as_f64().unwrap(), TONE_N);
    assert_eq!(curve["sigma"].as_f64().unwrap(), TONE_SIGMA);
    assert_eq!(curve["white"].as_f64().unwrap(), TONE_WHITE);
    assert_eq!(curve["s"].as_f64().unwrap(), TONE_S);
    assert_eq!(curve["k"].as_f64().unwrap(), TONE_K);
    // The constants in code are the formula of the file.
    let t = CameraTone::new(TONE_N, TONE_SIGMA, TONE_WHITE).unwrap();
    assert!((t.s() - TONE_S).abs() < 1e-15);
    assert!((t.k() - TONE_K).abs() < 1e-14);
    assert_eq!(doc["anchors"]["white_x"].as_f64().unwrap(), TONE_WHITE);
    let anchor = &doc["anchors"];
    let y = CameraTone::FITTED.apply(anchor["middle_grey_x"].as_f64().unwrap());
    assert!((y - anchor["middle_grey_y"].as_f64().unwrap()).abs() < 1e-7);
    let code = srgb_encode(y) * 255.0;
    assert!((code - anchor["middle_grey_srgb_code"].as_f64().unwrap()).abs() < 0.05);
    assert_eq!(doc["fit"]["photos"], 12);
    assert_eq!(doc["fit"]["bodies"], 10);
}

#[test]
fn the_golden_table_is_the_curve() {
    let doc = tone_file();
    let points = doc["table"]["points_x_y"].as_array().unwrap();
    assert_eq!(points.len(), 28);
    let mut last_y = -1.0;
    for p in points {
        let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        let got = CameraTone::FITTED.apply(x);
        // The table is stored to 8 significant digits.
        assert!(
            (got - y).abs() <= 1e-7 * y.max(1e-3) + 1e-9,
            "x {x}: {got} vs {y}"
        );
        assert!(y > last_y, "the table is increasing");
        last_y = y;
    }
    let (last_x, last_y) = (
        points.last().unwrap()[0].as_f64().unwrap(),
        points.last().unwrap()[1].as_f64().unwrap(),
    );
    assert_eq!((last_x, last_y), (TONE_WHITE, 1.0));
}

#[test]
fn the_files_carry_only_numbers_and_body_names() {
    // docs/14 rule 8 and the owner's files: no file names, no owner photo names, and no mention of
    // the tools or projects the numbers were read with.
    for (name, text) in [
        ("color-matrices.json", EMBEDDED_JSON.to_string()),
        (
            "camera-default-tone.json",
            std::fs::read_to_string(data_root().join("cameras/camera-default-tone.json")).unwrap(),
        ),
    ] {
        let lower = text.to_ascii_lowercase();
        for banned in [
            "libraw",
            "rawpy",
            "dcraw",
            "darktable",
            "rawtherapee",
            ".arw",
            ".jpg",
            ".jpeg",
            "owner",
            "dsc0",
            "a1m0",
        ] {
            assert!(!lower.contains(banned), "{name} mentions {banned:?}");
        }
    }
}
