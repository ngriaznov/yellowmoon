//! M1-06, the illuminant estimator in Rust (`docs/02-colour-pipeline.md`, section Illuminant estimate).
//!
//! * Parity with the Python reference (`training/illum/`) on the committed vectors (`tests/data/illum/`): every stage of a
//!   small random-weight model (histograms, filter scores, the histogram-only and fused Gaussians, the class features,
//!   probabilities, confidence, `w_scene`) and the final outputs of the shipped A7R IV model.
//! * The shipped models (`data/models/illum/`): format, SHA-256, size, histogram window against the profile's locus.
//! * Behaviour: bad files and previews are errors, fallbacks, the as-shot prior, the hand-over to `SceneLight`.
//!
//! Needs no git-ignored data.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::illum::{
    CLASS_NAMES, HistogramChannel, IllumError, IlluminantModel, PRIOR_SIGMA, RawPreview, ShotInfo,
};
use ym_color::spectral::Locus;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn vector_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/illum")
}

fn read_json(path: &PathBuf) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

fn b64(text: &str) -> Vec<u8> {
    // Independent of the crate's own decoder: the standard alphabet, `=` padding.
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes().filter(|c| *c != b'=') {
        let v = ALPHABET
            .iter()
            .position(|a| *a == c)
            .expect("base64 character") as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

struct Case {
    name: String,
    width: usize,
    height: usize,
    rgb: Vec<f32>,
    mask: Option<Vec<u8>>,
    shot: ShotInfo,
    as_shot_uv: Option<[f64; 2]>,
    expected: Value,
}

impl Case {
    fn preview(&self) -> RawPreview<'_> {
        RawPreview {
            width: self.width,
            height: self.height,
            rgb: &self.rgb,
            clip_mask: self.mask.as_deref(),
        }
    }
}

/// Gains that make a white surface under the light `(u, v)` neutral: `g = (exp(u), 1, exp(v))`.
fn gains_of_uv(uv: [f64; 2]) -> [f64; 3] {
    [uv[0].exp(), 1.0, uv[1].exp()]
}

/// Bv = log2(N^2 / t) - log2(ISO / 100) realised with N = 4, ISO = 100: t = 16 / 2^bv.
fn shot_for(as_shot_uv: Option<[f64; 2]>, bv: Option<f64>) -> ShotInfo {
    ShotInfo {
        as_shot_gains: as_shot_uv.map(gains_of_uv),
        f_number: bv.map(|_| 4.0),
        exposure_time_s: bv.map(|b| 16.0 / b.exp2()),
        iso: bv.map(|_| 100.0),
    }
}

fn load_cases(file: &str) -> (Value, Vec<Case>) {
    let doc = read_json(&vector_dir().join(file));
    let cases = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let rgb: Vec<f32> = b64(c["rgb_f32_b64"].as_str().unwrap())
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let mask = c["clip_mask_u8_b64"].as_str().map(b64);
            let as_shot_uv = c["as_shot_uv"]
                .as_array()
                .map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()]);
            let bv = c["bv"].as_f64();
            Case {
                name: c["name"].as_str().unwrap().to_string(),
                width: c["width"].as_u64().unwrap() as usize,
                height: c["height"].as_u64().unwrap() as usize,
                rgb,
                mask,
                shot: shot_for(as_shot_uv, bv),
                as_shot_uv,
                expected: c["expected"].clone(),
            }
        })
        .collect();
    (doc, cases)
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst <= tol,
        "{what}: worst difference {worst:e} above {tol:e}"
    );
}

fn flat(m: [[f64; 2]; 2]) -> Vec<f64> {
    vec![m[0][0], m[0][1], m[1][0], m[1][1]]
}

// ---------------------------------------------------------------------------------------------------------
// Parity with the Python reference
// ---------------------------------------------------------------------------------------------------------

/// Largest difference tolerated against the Python reference for outputs: 1e-9 (docs/02, M1-06).
const PARITY_TOL: f64 = 1e-9;

#[test]
fn every_stage_of_the_test_model_matches_the_python_reference() {
    let model = IlluminantModel::from_path(vector_dir().join("test_model.json")).unwrap();
    let (doc, cases) = load_cases("vectors.json");
    assert_eq!(
        model.sha256(),
        doc["model_sha256"].as_str().unwrap(),
        "vectors are stale: rerun export_vectors"
    );
    assert!(
        cases.len() >= 10,
        "expected the scene crops and the edge cases"
    );
    let mut worst = 0.0f64;
    let mut full_cases = 0;
    for case in &cases {
        let d = model
            .estimate_detailed(&case.preview(), &case.shot)
            .unwrap();
        let e = &case.expected;
        let name = &case.name;
        assert_eq!(
            d.estimate.fallback,
            e["fallback"].as_bool().unwrap(),
            "{name}: fallback"
        );
        assert_eq!(
            d.estimate.prior_used,
            case.as_shot_uv.is_some(),
            "{name}: prior_used"
        );
        let mut check = |what: &str, got: &[f64], want: &[f64], tol: f64| {
            let w = got
                .iter()
                .zip(want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max);
            worst = worst.max(w);
            assert_close(&format!("{name}: {what}"), got, want, tol);
        };
        check("uv", &d.estimate.chroma_uv, &floats(&e["uv"]), PARITY_TOL);
        check(
            "covariance",
            &flat(d.estimate.covariance_uv),
            &floats(&e["cov"]),
            PARITY_TOL,
        );
        check(
            "class probabilities",
            &d.estimate.class_probabilities(),
            &floats(&e["probs"]),
            PARITY_TOL,
        );
        check(
            "confidence",
            &[d.estimate.confidence],
            &[e["confidence"].as_f64().unwrap()],
            PARITY_TOL,
        );
        check(
            "w_scene",
            &d.estimate.w_scene,
            &floats(&e["w_scene"]),
            PARITY_TOL,
        );
        if d.estimate.fallback {
            continue;
        }
        let hist = d.histograms.as_ref().unwrap();
        check(
            "valid fraction",
            &hist.valid_fraction,
            &floats(&e["valid_fraction"]),
            1e-14,
        );
        check("in range", &hist.in_range, &floats(&e["in_range"]), 1e-12);
        check("mass", &hist.mass, &floats(&e["mass"]), 1e-10);
        let sums: Vec<f64> = hist.data.chunks(16 * 16).map(|c| c.iter().sum()).collect();
        let squares: Vec<f64> = hist
            .data
            .chunks(16 * 16)
            .map(|c| c.iter().map(|v| v * v).sum())
            .collect();
        check("histogram sums", &sums, &floats(&e["hist_sum"]), 1e-12);
        check(
            "histogram squares",
            &squares,
            &floats(&e["hist_sq_sum"]),
            1e-12,
        );
        check(
            "histogram-only uv",
            &d.estimate.histogram_chroma_uv,
            &floats(&e["uv_obs"]),
            PARITY_TOL,
        );
        if e.get("hist").is_some() {
            full_cases += 1;
            check("histograms", &hist.data, &floats(&e["hist"]), 1e-12);
            check("filter scores", &d.scores, &floats(&e["scores"]), 1e-10);
            check(
                "histogram-only covariance",
                &flat(d.covariance_obs),
                &floats(&e["cov_obs"]),
                PARITY_TOL,
            );
            check(
                "class features",
                &d.features,
                &floats(&e["features"]),
                PARITY_TOL,
            );
        }
    }
    assert!(full_cases >= 4, "the first cases carry every stage");
    eprintln!("test model: worst difference to the Python reference {worst:.2e}");
}

#[test]
fn the_shipped_a7r4_model_matches_the_python_reference() {
    let path = repo_root().join("data/models/illum/sony-ilce-7rm4.json");
    let Ok(model) = IlluminantModel::from_path(&path) else {
        panic!("{} must load", path.display());
    };
    let (doc, cases) = load_cases("shipped_vectors.json");
    assert_eq!(doc["profile_id"].as_str().unwrap(), "sony-ilce-7rm4");
    assert_eq!(
        model.sha256(),
        doc["model_sha256"].as_str().unwrap(),
        "data/models/illum/sony-ilce-7rm4.json changed: rerun `python3 -m training.illum.export_vectors`"
    );
    let mut worst = 0.0f64;
    for case in &cases {
        let est = model.estimate(&case.preview(), &case.shot).unwrap();
        let e = &case.expected;
        let name = &case.name;
        assert_eq!(
            est.fallback,
            e["fallback"].as_bool().unwrap(),
            "{name}: fallback"
        );
        let mut check = |what: &str, got: &[f64], want: &[f64]| {
            let w = got
                .iter()
                .zip(want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max);
            worst = worst.max(w);
            assert_close(&format!("{name}: {what}"), got, want, PARITY_TOL);
        };
        check("uv", &est.chroma_uv, &floats(&e["uv"]));
        check("covariance", &flat(est.covariance_uv), &floats(&e["cov"]));
        check(
            "class probabilities",
            &est.class_probabilities(),
            &floats(&e["probs"]),
        );
        check(
            "confidence",
            &[est.confidence],
            &[e["confidence"].as_f64().unwrap()],
        );
        check("w_scene", &est.w_scene, &floats(&e["w_scene"]));
        if !est.fallback {
            check(
                "histogram-only uv",
                &est.histogram_chroma_uv,
                &floats(&e["uv_obs"]),
            );
        }
    }
    eprintln!("shipped A7R IV model: worst difference to the Python reference {worst:.2e}");
}

/// A preview of `all_profiles_vectors.json`, shared by the 11 models (each scales it to its own body).
struct SharedPreview {
    width: usize,
    height: usize,
    rgb: Vec<f32>,
    mask: Vec<u8>,
    as_shot_uv: Option<[f64; 2]>,
    bv: Option<f64>,
}

/// Every shipped model, not only the A7R IV's, against the Python reference: the 11 histogram windows differ, so a loader or
/// estimator fault that depends on `lo_u`, `lo_v` or the other constants of a model file would pass the A7R IV test alone.
/// The previews are scaled per channel to look like each body's raw data (`export_vectors.channel_gain`).
#[test]
fn every_shipped_model_matches_the_python_reference() {
    let doc = read_json(&vector_dir().join("all_profiles_vectors.json"));
    let previews: Vec<SharedPreview> = doc["previews"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| SharedPreview {
            width: p["width"].as_u64().unwrap() as usize,
            height: p["height"].as_u64().unwrap() as usize,
            rgb: b64(p["rgb_f32_b64"].as_str().unwrap())
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            mask: b64(p["clip_mask_u8_b64"].as_str().unwrap()),
            as_shot_uv: p["as_shot_uv"]
                .as_array()
                .map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()]),
            bv: p["bv"].as_f64(),
        })
        .collect();
    let profiles = doc["profiles"].as_object().unwrap();
    assert_eq!(profiles.len(), 11, "one entry per shipped profile");
    let mut worst = 0.0f64;
    let mut cases_run = 0;
    for (id, entry) in profiles {
        let path = repo_root().join(format!("data/models/illum/{id}.json"));
        let model = IlluminantModel::from_path(&path).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(
            model.sha256(),
            entry["model_sha256"].as_str().unwrap(),
            "{id}: the model changed, rerun `python3 -m training.illum.export_vectors`"
        );
        let gain: Vec<f32> = floats(&entry["channel_gain"])
            .iter()
            .map(|g| *g as f32)
            .collect();
        for case in entry["cases"].as_array().unwrap() {
            let shared = &previews[case["preview"].as_u64().unwrap() as usize];
            // The same float32 multiplication as numpy's, so both sides read identical pixels.
            let scaled: Vec<f32> = shared
                .rgb
                .iter()
                .enumerate()
                .map(|(i, v)| v * gain[i % 3])
                .collect();
            let mask = if case["all_masked"].as_bool().unwrap() {
                vec![1u8; shared.mask.len()]
            } else {
                shared.mask.clone()
            };
            let as_shot = shared
                .as_shot_uv
                .filter(|_| case["use_as_shot"].as_bool().unwrap());
            let bv = shared.bv.filter(|_| case["use_bv"].as_bool().unwrap());
            let est = model
                .estimate(
                    &RawPreview {
                        width: shared.width,
                        height: shared.height,
                        rgb: &scaled,
                        clip_mask: Some(&mask),
                    },
                    &shot_for(as_shot, bv),
                )
                .unwrap();
            let what = format!("{id} preview {} {case}", case["preview"]);
            assert_eq!(
                est.fallback,
                case["fallback"].as_bool().unwrap(),
                "{what}: fallback"
            );
            assert_eq!(est.prior_used, as_shot.is_some(), "{what}: prior_used");
            let mut check = |name: &str, got: &[f64], want: &[f64]| {
                let w = got
                    .iter()
                    .zip(want)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0.0f64, f64::max);
                worst = worst.max(w);
                assert_close(&format!("{what}: {name}"), got, want, PARITY_TOL);
            };
            check("uv", &est.chroma_uv, &floats(&case["uv"]));
            check(
                "covariance",
                &flat(est.covariance_uv),
                &floats(&case["cov"]),
            );
            check(
                "class probabilities",
                &est.class_probabilities(),
                &floats(&case["probs"]),
            );
            check(
                "confidence",
                &[est.confidence],
                &[case["confidence"].as_f64().unwrap()],
            );
            check("w_scene", &est.w_scene, &floats(&case["w_scene"]));
            cases_run += 1;
        }
    }
    assert_eq!(cases_run, 11 * 8);
    eprintln!("all 11 shipped models: worst difference to the Python reference {worst:.2e}");
}

// ---------------------------------------------------------------------------------------------------------
// The shipped models
// ---------------------------------------------------------------------------------------------------------

fn profile_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

fn load_locus(id: &str) -> Locus {
    let v = read_json(&repo_root().join(format!("data/cameras/profiles/{id}/profile.json")));
    serde_json::from_value(v["locus"].clone()).unwrap()
}

#[test]
fn every_profile_has_a_valid_model_under_the_size_limit() {
    let ids = profile_ids();
    assert_eq!(ids.len(), 11);
    for id in &ids {
        let path = repo_root().join(format!("data/models/illum/{id}.json"));
        let model = IlluminantModel::from_path(&path).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(model.profile_id(), id);
        assert!(!model.version().is_empty());
        assert!(
            model.param_count() < 50_000,
            "{id}: {} parameters",
            model.param_count()
        );
        let spec = model.histogram_spec();
        assert_eq!(spec.n_bins, 64);
        assert_eq!(
            spec.channels,
            [HistogramChannel::Intensity, HistogramChannel::Edge]
        );
        // The histogram window has to contain the profile's whole light locus (1500-25000 K), with room for scenes.
        let locus = load_locus(id);
        let hi_u = spec.lo_u + (spec.n_bins - 1) as f64 * spec.bin_size;
        let hi_v = spec.lo_v + (spec.n_bins - 1) as f64 * spec.bin_size;
        for t in (1500..=25000).step_by(500) {
            let rgb = locus.rgb_at(f64::from(t));
            let (u, v) = (-rgb[0].ln(), -rgb[2].ln());
            assert!(
                u > spec.lo_u + 0.5 && u < hi_u - 0.5 && v > spec.lo_v + 0.5 && v < hi_v - 0.5,
                "{id}: the locus at {t} K ({u:.3}, {v:.3}) is not well inside the window"
            );
        }
    }
}

#[test]
fn a_model_that_was_edited_is_refused() {
    let text = std::fs::read_to_string(vector_dir().join("test_model.json")).unwrap();
    assert!(IlluminantModel::from_json_str(&text).is_ok());
    // Flip one base64 character of the filter: the SHA-256 no longer matches.
    let at = text.find("\"data_b64\": \"").unwrap() + "\"data_b64\": \"".len() + 8;
    let mut bytes = text.clone().into_bytes();
    bytes[at] = if bytes[at] == b'A' { b'B' } else { b'A' };
    let err = IlluminantModel::from_json_str(&String::from_utf8(bytes).unwrap()).unwrap_err();
    assert!(matches!(err, IllumError::HashMismatch { .. }), "{err}");
}

#[test]
fn a_model_is_found_by_profile_id_in_the_data_folder() {
    let model =
        IlluminantModel::load_for_profile(repo_root().join("data"), "sony-ilce-7rm4").unwrap();
    assert_eq!(model.profile_id(), "sony-ilce-7rm4");
    for bad in ["", "../x", "a/b", "sony ilce", "sony-ilce-7rm4.json"] {
        assert!(
            matches!(
                IlluminantModel::load_for_profile(repo_root().join("data"), bad),
                Err(IllumError::InvalidModel(_))
            ),
            "{bad:?}"
        );
    }
    assert!(matches!(
        IlluminantModel::load_for_profile(repo_root().join("data"), "sony-nonexistent"),
        Err(IllumError::Io { .. })
    ));
}

#[test]
fn bad_model_files_are_errors_not_panics() {
    let good: Value = read_json(&vector_dir().join("test_model.json"));
    let mutate = |f: &dyn Fn(&mut Value)| {
        let mut v = good.clone();
        f(&mut v);
        IlluminantModel::from_json_str(&v.to_string())
    };
    assert!(matches!(
        IlluminantModel::from_json_str("not json"),
        Err(IllumError::Json(_))
    ));
    assert!(matches!(
        IlluminantModel::from_json_str("{}"),
        Err(IllumError::Json(_))
    ));
    assert!(matches!(
        mutate(&|v| v["format"] = "other".into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["format_version"] = 2.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["histogram"]["n_bins"] = 17.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["histogram"]["n_bins"] = 1024.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["histogram"]["bin_size"] = 0.0.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["histogram"]["channels"] = Value::Array(vec![])),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["histogram"]["channels"] = Value::Array(vec!["blue".into()])),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["fit"]["pad_variance"] = (-1.0).into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["fit"]["prior_sigma"] = 0.0.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["confidence"]["kind"] = "linear".into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["confidence"]["n_features"] = 5.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["head"]["temperature"] = 0.0.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["head"]["hidden"] = 5.into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["head"]["classes"] = Value::Array(vec![])),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["head"]["class_prior"] = serde_json::json!([1, 0, 0, 0, 0, 1])),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["tensors"]["filter"]["shape"] = serde_json::json!([2, 16, 8])),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["tensors"]["bias"]["data_b64"] = "AAAA".into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["tensors"]["bias"]["data_b64"] = "@@@@".into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["tensors"]["w1"]["dtype"] = "f64le".into()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("b2")
            .map(drop)
            .unwrap()),
        Err(IllumError::InvalidModel(_))
    ));
    assert!(matches!(
        mutate(&|v| v["sha256"] = "00".into()),
        Err(IllumError::HashMismatch { .. })
    ));
    assert!(matches!(
        IlluminantModel::from_path("/nonexistent/illum.json"),
        Err(IllumError::Io { .. })
    ));
}

// ---------------------------------------------------------------------------------------------------------
// Behaviour
// ---------------------------------------------------------------------------------------------------------

fn a7r4() -> IlluminantModel {
    IlluminantModel::from_path(repo_root().join("data/models/illum/sony-ilce-7rm4.json")).unwrap()
}

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    (dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// A simple RGB-reflectance scene under the camera white `w`: `n` colour patches (a few saturated colours and greys)
/// with shading, as a 1/8-resolution preview.
fn scene_under(w: [f64; 3], seed: u64) -> (Vec<f32>, usize, usize) {
    const REFL: [[f64; 3]; 8] = [
        [0.55, 0.55, 0.55],
        [0.80, 0.30, 0.22],
        [0.28, 0.60, 0.30],
        [0.22, 0.30, 0.70],
        [0.62, 0.52, 0.12],
        [0.30, 0.30, 0.30],
        [0.75, 0.70, 0.62],
        [0.45, 0.25, 0.40],
    ];
    let (width, height) = (96usize, 64usize);
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let cell = ((x / 12) + 8 * (y / 16)) % REFL.len();
            let shade = 0.05 * (0.7 + 0.6 * next());
            for c in 0..3 {
                rgb.push((w[c] * REFL[cell][c] * shade) as f32);
            }
        }
    }
    (rgb, width, height)
}

#[test]
fn a_scene_under_a_locus_light_is_estimated_within_a_few_degrees() {
    let model = a7r4();
    let locus = load_locus("sony-ilce-7rm4");
    let mut worst: f64 = 0.0;
    for (i, t) in [2400.0, 3000.0, 4000.0, 5000.0, 5500.0, 6500.0, 8000.0]
        .into_iter()
        .enumerate()
    {
        let w = locus.rgb_at(t);
        let (rgb, width, height) = scene_under(w, 7 + i as u64);
        let preview = RawPreview {
            width,
            height,
            rgb: &rgb,
            clip_mask: None,
        };
        let est = model.estimate(&preview, &ShotInfo::default()).unwrap();
        let err = angle_deg(est.w_scene, w);
        worst = worst.max(err);
        assert!(!est.fallback && !est.prior_used);
        assert!((est.w_scene[1] - 1.0).abs() < 1e-15, "green must be 1");
        assert!(
            err < 4.0,
            "{t} K: {err:.2} degrees (estimate {:?}, truth {w:?})",
            est.w_scene
        );
    }
    eprintln!("worst error on the RGB-reflectance scenes: {worst:.2} degrees");
}

#[test]
fn the_as_shot_prior_pulls_the_estimate_but_a_confident_histogram_resists() {
    let model = a7r4();
    let locus = load_locus("sony-ilce-7rm4");
    let w = locus.rgb_at(5000.0);
    let (rgb, width, height) = scene_under(w, 3);
    let preview = RawPreview {
        width,
        height,
        rgb: &rgb,
        clip_mask: None,
    };
    let truth_uv = [-w[0].ln(), -w[2].ln()];
    let alone = model.estimate(&preview, &ShotInfo::default()).unwrap();
    // As-shot exactly the truth: the result does not move away from it.
    let shot = ShotInfo {
        as_shot_gains: Some(gains_of_uv(truth_uv)),
        ..ShotInfo::default()
    };
    let good = model.estimate(&preview, &shot).unwrap();
    assert!(good.prior_used);
    assert!(angle_deg(good.w_scene, w) <= angle_deg(alone.w_scene, w) + 0.5);
    // The prior shrinks the covariance (a product of Gaussians is narrower than either).
    assert!(
        good.covariance_uv[0][0] < alone.covariance_uv[0][0]
            && good.covariance_uv[0][0] < PRIOR_SIGMA * PRIOR_SIGMA
    );
    // As-shot far off (a preset white balance): the estimate moves towards it, but stays between.
    let off_uv = [truth_uv[0] + 0.3, truth_uv[1] - 0.3];
    let shot = ShotInfo {
        as_shot_gains: Some(gains_of_uv(off_uv)),
        ..ShotInfo::default()
    };
    let pulled = model.estimate(&preview, &shot).unwrap();
    let d_alone = (alone.chroma_uv[0] - off_uv[0]).hypot(alone.chroma_uv[1] - off_uv[1]);
    let d_pulled = (pulled.chroma_uv[0] - off_uv[0]).hypot(pulled.chroma_uv[1] - off_uv[1]);
    assert!(
        d_pulled < d_alone,
        "the prior must pull towards the as-shot value"
    );
    assert!(d_pulled > 0.0);
    // An as-shot value that disagrees with the histogram lowers the confidence, one that agrees does not.
    assert!(
        pulled.confidence < good.confidence,
        "{} vs {}",
        pulled.confidence,
        good.confidence
    );
    // Unusable gains are ignored like missing ones.
    for bad in [
        [0.0, 1.0, 1.0],
        [f64::NAN, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, f64::INFINITY, 1.0],
    ] {
        let shot = ShotInfo {
            as_shot_gains: Some(bad),
            ..ShotInfo::default()
        };
        let est = model.estimate(&preview, &shot).unwrap();
        assert!(!est.prior_used);
        assert_eq!(est.w_scene, alone.w_scene);
    }
}

#[test]
fn a_preview_without_usable_pixels_falls_back_to_the_as_shot_white_or_the_cameras_d55() {
    let model = a7r4();
    let rgb = vec![0.2f32; 3 * 20 * 10];
    let mask = vec![1u8; 20 * 10];
    let preview = RawPreview {
        width: 20,
        height: 10,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let none = model.estimate(&preview, &ShotInfo::default()).unwrap();
    assert!(none.fallback && none.confidence == 0.0 && !none.prior_used);
    let sum: f64 = none.class_probabilities().iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    // D55 for this camera: the A7R IV daylight locus at 5500 K.
    let d55 = load_locus("sony-ilce-7rm4").rgb_at(5500.0);
    assert!(
        angle_deg(none.w_scene, d55) < 0.5,
        "{:?} vs {d55:?}",
        none.w_scene
    );
    let uv = [0.9f64, 0.45];
    let shot = ShotInfo {
        as_shot_gains: Some(gains_of_uv(uv)),
        ..ShotInfo::default()
    };
    let with = model.estimate(&preview, &shot).unwrap();
    assert!(with.fallback && with.prior_used);
    assert!((with.chroma_uv[0] - uv[0]).abs() < 1e-12 && (with.chroma_uv[1] - uv[1]).abs() < 1e-12);
    // All pixels at or below the noise floor are as good as masked.
    let dark = vec![1e-4f32; 3 * 20 * 10];
    let est = model
        .estimate(
            &RawPreview {
                width: 20,
                height: 10,
                rgb: &dark,
                clip_mask: None,
            },
            &ShotInfo::default(),
        )
        .unwrap();
    assert!(est.fallback);
}

#[test]
fn preview_buffers_of_the_wrong_size_are_errors() {
    let model = a7r4();
    let rgb = vec![0.1f32; 3 * 10 * 10];
    let short = RawPreview {
        width: 11,
        height: 10,
        rgb: &rgb,
        clip_mask: None,
    };
    assert!(matches!(
        model.estimate(&short, &ShotInfo::default()),
        Err(IllumError::PreviewSize { .. })
    ));
    let mask = vec![0u8; 99];
    let bad_mask = RawPreview {
        width: 10,
        height: 10,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    assert!(matches!(
        model.estimate(&bad_mask, &ShotInfo::default()),
        Err(IllumError::PreviewSize { .. })
    ));
    let huge = RawPreview {
        width: usize::MAX,
        height: 3,
        rgb: &rgb,
        clip_mask: None,
    };
    assert!(matches!(
        model.estimate(&huge, &ShotInfo::default()),
        Err(IllumError::PreviewSize { .. })
    ));
    // An empty preview is a fallback, not an error.
    let empty = RawPreview {
        width: 0,
        height: 0,
        rgb: &[],
        clip_mask: None,
    };
    assert!(
        model
            .estimate(&empty, &ShotInfo::default())
            .unwrap()
            .fallback
    );
}

#[test]
fn shot_info_converts_gains_and_exposure_settings() {
    let shot = ShotInfo {
        as_shot_gains: Some([2.0, 1.0, 1.5]),
        ..ShotInfo::default()
    };
    let uv = shot.as_shot_uv().unwrap();
    assert!((uv[0] - 2.0f64.ln()).abs() < 1e-15 && (uv[1] - 1.5f64.ln()).abs() < 1e-15);
    assert_eq!(ShotInfo::default().as_shot_uv(), None);
    assert_eq!(ShotInfo::default().bv(), None);
    // Bv = log2(N^2/t) - log2(ISO/100): f/8 at 1/125 s and ISO 100 is 6 + 7 - 0 = 12.97
    let shot = ShotInfo {
        f_number: Some(8.0),
        exposure_time_s: Some(1.0 / 125.0),
        iso: Some(100.0),
        ..ShotInfo::default()
    };
    assert!((shot.bv().unwrap() - (64.0f64 * 125.0).log2()).abs() < 1e-12);
    let shot = ShotInfo {
        f_number: Some(8.0),
        exposure_time_s: Some(1.0 / 125.0),
        iso: Some(800.0),
        ..ShotInfo::default()
    };
    assert!((shot.bv().unwrap() - ((64.0f64 * 125.0).log2() - 3.0)).abs() < 1e-12);
    for (n, t, iso) in [
        (0.0, 1.0, 100.0),
        (4.0, 0.0, 100.0),
        (4.0, 1.0, -1.0),
        (f64::NAN, 1.0, 100.0),
    ] {
        let shot = ShotInfo {
            f_number: Some(n),
            exposure_time_s: Some(t),
            iso: Some(iso),
            ..ShotInfo::default()
        };
        assert_eq!(shot.bv(), None);
    }
}

#[test]
fn the_estimate_hands_over_to_the_scene_light() {
    let model = a7r4();
    let locus = load_locus("sony-ilce-7rm4");
    let w = locus.rgb_at(3000.0);
    let (rgb, width, height) = scene_under(w, 11);
    let preview = RawPreview {
        width,
        height,
        rgb: &rgb,
        clip_mask: None,
    };
    let est = model.estimate(&preview, &ShotInfo::default()).unwrap();
    let probs = est.class_probabilities();
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(CLASS_NAMES.len(), probs.len());
    assert!((0.0..=1.0).contains(&est.confidence));
    let light = est.scene_light(&locus).unwrap();
    assert_eq!(light.w_scene, est.w_scene);
    assert!((light.confidence - est.confidence).abs() < 1e-15);
    // A tungsten scene reads a warm CCT from the profile's locus (the estimator and the locus share the raw scale).
    assert!(
        light.cct_k > 2400.0 && light.cct_k < 3800.0,
        "CCT {}",
        light.cct_k
    );
    let total = light.classes.to_array().iter().sum::<f64>();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn clipped_pixels_are_ignored() {
    let model = a7r4();
    let locus = load_locus("sony-ilce-7rm4");
    let w = locus.rgb_at(5500.0);
    let (mut rgb, width, height) = scene_under(w, 5);
    let clean = model
        .estimate(
            &RawPreview {
                width,
                height,
                rgb: &rgb,
                clip_mask: None,
            },
            &ShotInfo::default(),
        )
        .unwrap();
    // Corrupt the left third with an absurd colour and mask it: the estimate must not change.
    let mut mask = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width / 3 {
            let i = y * width + x;
            rgb[3 * i] = 0.9;
            rgb[3 * i + 1] = 0.05;
            rgb[3 * i + 2] = 0.9;
            mask[i] = 1;
        }
    }
    let masked = model
        .estimate(
            &RawPreview {
                width,
                height,
                rgb: &rgb,
                clip_mask: Some(&mask),
            },
            &ShotInfo::default(),
        )
        .unwrap();
    // The histogram of the remaining two thirds differs from the clean one (fewer pixels), so allow a small change.
    assert!(
        angle_deg(masked.w_scene, clean.w_scene) < 3.0,
        "{:?} vs {:?}",
        masked.w_scene,
        clean.w_scene
    );
    // What is under the mask does not matter at all: other garbage under the same mask gives the very same estimate,
    // and the masked pixels carry no weight (about two thirds of the pixels are valid).
    let mut other = rgb.clone();
    for (i, m) in mask.iter().enumerate() {
        if *m != 0 {
            other[3 * i] = 0.01;
            other[3 * i + 1] = 0.9;
            other[3 * i + 2] = 0.3;
        }
    }
    let again = model
        .estimate(
            &RawPreview {
                width,
                height,
                rgb: &other,
                clip_mask: Some(&mask),
            },
            &ShotInfo::default(),
        )
        .unwrap();
    assert_eq!(again, masked);
    let hist = ym_color::illum::histogram::compute(
        model.histogram_spec(),
        &RawPreview {
            width,
            height,
            rgb: &rgb,
            clip_mask: Some(&mask),
        },
    )
    .unwrap();
    assert!(
        hist.valid_fraction[0] > 0.6 && hist.valid_fraction[0] < 0.7,
        "{:?}",
        hist.valid_fraction
    );
}

#[test]
fn an_estimated_light_drives_the_fused_chain_to_a_neutral_scene_white() {
    use ym_color::chain::{ChainSettings, FusedChain, ProfileTables};
    use ym_color::sensor_map::SensorMapTables;
    use ym_color::spectral::AlexaWhite;

    let model = a7r4();
    let v = read_json(&repo_root().join("data/cameras/profiles/sony-ilce-7rm4/profile.json"));
    let locus: Locus = serde_json::from_value(v["locus"].clone()).unwrap();
    let white: AlexaWhite = serde_json::from_value(v["alexa_white"].clone()).unwrap();
    let maps: SensorMapTables = serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap();
    let tables = ProfileTables {
        sensor_map: &maps,
        alexa_white: &white,
        locus: &locus,
    };
    for (i, t) in [2700.0, 3200.0, 5200.0, 7500.0].into_iter().enumerate() {
        let w = locus.rgb_at(t);
        let (rgb, width, height) = scene_under(w, 20 + i as u64);
        let preview = RawPreview {
            width,
            height,
            rgb: &rgb,
            clip_mask: None,
        };
        let est = model.estimate(&preview, &ShotInfo::default()).unwrap();
        let light = est.scene_light(&locus).unwrap();
        // Raw processing balanced the mosaic with the estimate itself, so the chain input is `raw / w_scene`;
        // the scene white then is [1, 1, 1] and Auto must keep it neutral: AWG4 [1, 1, 1].
        let chain =
            FusedChain::build(tables, &light, est.w_scene, &ChainSettings::default()).unwrap();
        let out = chain.apply([1.0, 1.0, 1.0]).0;
        for c in out {
            assert!((c - 1.0).abs() < 1e-9, "{t} K: {out:?}");
        }
        // That was true by construction. What the estimator contributes: a white surface of the *real* light, balanced
        // with the estimate (`w / w_scene`), lands near neutral. The cast is the estimate's own error, amplified by the
        // sensor map and ARRI's matrix: about 1.2 degrees of angular error read 3 to 6 per cent off in a channel here.
        let off = [w[0] / est.w_scene[0], 1.0, w[2] / est.w_scene[2]];
        let out = chain.apply(off).0;
        eprintln!(
            "{t} K: {:.2} deg, true white through the estimate {out:?}",
            angle_deg(est.w_scene, w)
        );
        for c in out {
            assert!(
                (c - 1.0).abs() < 0.10,
                "{t} K: the true white reads {out:?}"
            );
        }
    }
}

/// Timing of one estimate on the 1/8-resolution preview of a 50 MP frame (8640 x 5760 -> 1080 x 720). Run with
/// `cargo test --release -p ym-color --test illum -- --ignored --nocapture timing`.
#[test]
#[ignore = "timing; run in release mode with --nocapture"]
fn timing_of_one_estimate_on_a_50_mp_preview() {
    let model = a7r4();
    let locus = load_locus("sony-ilce-7rm4");
    let w = locus.rgb_at(5000.0);
    let (width, height) = (1080usize, 720usize);
    let mut s = 1u64;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let shade = 0.02 + 0.1 * ((s >> 40) as f64 / (1u64 << 24) as f64);
            let cell = ((x / 90) + 12 * (y / 90)) % 7;
            let tint = 0.3 + 0.1 * cell as f64;
            for (c, wc) in w.iter().enumerate() {
                let r = if c == cell % 3 { tint * 1.6 } else { tint };
                rgb.push((wc * shade * r) as f32);
            }
        }
    }
    let preview = RawPreview {
        width,
        height,
        rgb: &rgb,
        clip_mask: None,
    };
    let shot = ShotInfo::default();
    let _ = model.estimate(&preview, &shot).unwrap(); // warm up
    let runs = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        std::hint::black_box(
            model
                .estimate(std::hint::black_box(&preview), &shot)
                .unwrap(),
        );
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(runs);
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        std::hint::black_box(
            ym_color::illum::histogram::compute(model.histogram_spec(), &preview).unwrap(),
        );
    }
    let hist_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(runs);
    eprintln!(
        "estimate on {width} x {height} ({:.2} Mpx, stride {}): {ms:.2} ms, of which the histograms {hist_ms:.2} ms",
        (width * height) as f64 / 1e6,
        model.histogram_spec().stride(width, height)
    );
    assert!(ms < 200.0, "the estimate takes {ms:.1} ms");
}
