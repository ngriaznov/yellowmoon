//! The trained illuminant estimator inside the automatic estimates. These tests need a model file
//! for the A7R IV profile; they look in `data/models/illum` and then in the training output
//! `testdata/out/illum/models`, and say so (and pass) when neither exists.

mod common;

use std::path::PathBuf;

use common::{Layout, Scene, base_env, data_root, light, wb_levels};
use ym_color::illum::IlluminantModel;
use ym_core::auto::{AutoEstimates, AutoInputs, IlluminantSource};
use ym_core::camera::PhotoCamera;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn model_path() -> Option<PathBuf> {
    let shipped = data_root()
        .join("models/illum")
        .join(format!("{PROFILE}.json"));
    let training = data_root()
        .parent()?
        .join("testdata/out/illum/models")
        .join(format!("{PROFILE}.json"));
    [shipped, training].into_iter().find(|p| p.is_file())
}

/// Angle in degrees between two camera whites.
fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    (dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

fn run(
    model: &IlluminantModel,
    truth_cct: f64,
    as_shot_cct: Option<f64>,
    dir: &std::path::Path,
) -> (f64, f64, bool) {
    let env = base_env();
    let truth = light(PROFILE, truth_cct);
    let scene = Scene::new(512, 384, truth, Layout::Patchwork, 0.5).noise(2e-4, 1e-6);
    let path = dir.join("m.ARW");
    let wb = as_shot_cct.map(|k| wb_levels(light(PROFILE, k)));
    scene
        .builder(MODEL)
        .wb(wb)
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let raw = ym_raw::decode::decode(&path).unwrap();
    let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    let est = AutoEstimates::compute(
        &raw,
        &AutoInputs {
            profile,
            illuminant_model: Some(model),
            priors: &env.priors,
            autoset: None,
            embedded_jpeg: None,
        },
    )
    .unwrap();
    let used_prior = matches!(
        est.illuminant.source,
        IlluminantSource::Model {
            prior_used: true,
            ..
        }
    );
    assert!(matches!(
        est.illuminant.source,
        IlluminantSource::Model { .. }
    ));
    (
        angle_deg(est.color.w_scene_initial, truth),
        // The scene light the estimator found (the Auto white is the preferred one, `docs/18` section 5, and keeps
        // part of a warm light's warmth, so it is not the light itself).
        est.illuminant.scene.cct_k,
        used_prior,
    )
}

#[test]
fn the_trained_estimator_finds_the_light_with_and_without_the_cameras_white_balance() {
    let Some(path) = model_path() else {
        eprintln!(
            "no illuminant model for {PROFILE} in data/models/illum or testdata/out; skipping"
        );
        return;
    };
    let model = IlluminantModel::from_path(&path).unwrap();
    eprintln!("using {}", path.display());
    let dir = tempfile::tempdir().unwrap();
    let (mut worst_right, mut worst_none, mut worst_wrong) = (0.0f64, 0.0f64, 0.0f64);
    for truth in [2800.0, 3200.0, 4300.0, 5500.0, 6500.0] {
        // The camera got it right: the prior and the pixels agree.
        let (err, kelvin, prior) = run(&model, truth, Some(truth), dir.path());
        assert!(prior, "the as-shot white balance is the prior");
        worst_right = worst_right.max(err);
        assert!(
            err < 1.0,
            "{truth} K with the right prior: {err:.2} degrees"
        );
        assert!(
            (kelvin - truth).abs() <= 150.0,
            "{truth} K read as {kelvin}"
        );
        // No white balance in the file: the pixels alone.
        let (err, _, prior) = run(&model, truth, None, dir.path());
        assert!(!prior);
        worst_none = worst_none.max(err);
        assert!(
            err < 1.5,
            "{truth} K from the pixels alone: {err:.2} degrees"
        );
        // The camera got it badly wrong (daylight white balance under any other light): the
        // estimate stays close to the truth and far from the camera's guess.
        let wrong = angle_deg(light(PROFILE, 5500.0), light(PROFILE, truth));
        let (err, _, _) = run(&model, truth, Some(5500.0), dir.path());
        worst_wrong = worst_wrong.max(err);
        assert!(err < 4.0, "{truth} K with a wrong prior: {err:.2} degrees");
        if wrong > 6.0 {
            assert!(
                err < 0.5 * wrong,
                "{truth} K: {err:.2} degrees against {wrong:.2} for the camera's own white balance"
            );
        }
    }
    eprintln!(
        "worst angular error: right prior {worst_right:.2}, no prior {worst_none:.2}, wrong prior {worst_wrong:.2} degrees"
    );
}

#[test]
fn a_model_that_belongs_to_another_profile_is_not_loaded_for_this_one() {
    // The loader checks the profile id in the file against the file name.
    let Some(path) = model_path() else {
        eprintln!("no illuminant model for {PROFILE}; skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models/illum");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::copy(&path, models.join("sony-ilce-9.json")).unwrap();
    let found = ym_core::auto::IlluminantModels::load(dir.path(), &common::profiles());
    assert!(found.get("sony-ilce-9").is_none());
    assert!(found.failed().iter().any(|(id, _)| id == "sony-ilce-9"));
    assert!(found.missing().iter().any(|id| id == PROFILE));
}
