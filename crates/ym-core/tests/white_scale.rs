//! The white scale of a stand-in body in the automatic estimates (`docs/03`, "White scale of stand-in
//! bodies"; `docs/02` C1): the as-shot white, the illuminant estimator's input, the initial gains and Pick white
//! work on the profile's scale, so a body that borrows a profile reads its whites the way the profile's curves
//! would.
//!
//! The model of a stand-in used throughout: a body whose raw is the profile's raw divided by the scale `s`
//! (red and blue), with as-shot gains multiplied by `s`: exactly what `training/synth` writes for the A1.

#![allow(clippy::needless_range_loop)]

mod common;

use std::path::PathBuf;

use common::{Layout, Scene, base_env, data_root, light, wb_levels};
use ym_color::chain::ChainError;
use ym_color::exposure::ExposureSetting;
use ym_color::illum::IlluminantModel;
use ym_color::wb::{LightSource, PickWhiteError, WbError, WhiteBalance};
use ym_core::auto::{AutoError, AutoEstimates, AutoInputs, IlluminantSource, pick_white_sample};
use ym_core::camera::PhotoCamera;

/// The A1 borrows the A7R IV.
const STAND_IN: &str = "ILCE-1";
const OWN: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn model_path() -> Option<PathBuf> {
    [
        data_root()
            .join("models/illum")
            .join(format!("{PROFILE}.json")),
        data_root()
            .parent()?
            .join("testdata/out/illum/models")
            .join(format!("{PROFILE}.json")),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    (dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// The registry's scale of the stand-in body (it must be a real correction for these tests to mean anything).
fn registry_scale() -> [f64; 3] {
    let m = base_env().profiles.detect("SONY", STAND_IN, Some(384));
    assert_eq!(m.profile.as_str(), PROFILE);
    let s = m.white_scale;
    assert!(
        (s[0] - 1.0).abs() > 0.05 && (s[2] - 1.0).abs() > 0.05 && s[1] == 1.0,
        "{s:?}"
    );
    s
}

struct Written {
    raw: ym_raw::decode::RawImage,
    scale: [f64; 3],
    truth: [f64; 3],
}

/// A scene lit by `truth_cct` as seen by the stand-in body: the profile's white of that light divided by the
/// scale is the body's raw white; the camera's as-shot white balance is `as_shot_cct`'s white, in the body's raw.
fn stand_in_scene(
    dir: &std::path::Path,
    truth_cct: f64,
    as_shot_cct: Option<f64>,
    layout: Layout,
    noise: bool,
) -> Written {
    let scale = registry_scale();
    let profile_white = |cct: f64| {
        let w = light(PROFILE, cct);
        [w[0] / scale[0], w[1] / scale[1], w[2] / scale[2]]
    };
    let truth = light(PROFILE, truth_cct);
    let mut scene = Scene::new(512, 384, profile_white(truth_cct), layout, 0.5);
    if noise {
        scene = scene.noise(2e-4, 1e-6);
    }
    let path = dir.join("a1.ARW");
    scene
        .builder(STAND_IN)
        .wb(as_shot_cct.map(|k| wb_levels(profile_white(k))))
        .write_to(&path, &scene.mosaic())
        .unwrap();
    Written {
        raw: ym_raw::decode::decode(&path).unwrap(),
        scale,
        truth,
    }
}

fn compute(
    raw: &ym_raw::decode::RawImage,
    model: Option<&IlluminantModel>,
    scale: Option<[f64; 3]>,
) -> AutoEstimates {
    let env = base_env();
    let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    let inputs = AutoInputs {
        profile,
        illuminant_model: model,
        priors: &env.priors,
    };
    match scale {
        Some(s) => AutoEstimates::compute_scaled(raw, &inputs, s).unwrap(),
        None => AutoEstimates::compute(raw, &inputs).unwrap(),
    }
}

#[test]
fn the_camera_match_of_a_session_carries_the_scale_the_estimates_need() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(64, 48, light(PROFILE, 5500.0), Layout::Uniform, 0.5);
    let a1 = ym_raw::decode::decode(&scene.write(dir.path(), "a1.ARW", STAND_IN)).unwrap();
    let own = ym_raw::decode::decode(&scene.write(dir.path(), "own.ARW", OWN)).unwrap();
    let a1_camera = PhotoCamera::new(&env.profiles, &a1.meta, None);
    let own_camera = PhotoCamera::new(&env.profiles, &own.meta, None);
    assert!(a1_camera.stand_in && a1_camera.matched.white_scale == registry_scale());
    assert!(!own_camera.stand_in && own_camera.matched.white_scale == [1.0; 3]);
    // `--profile` another one: the scale does not apply to it.
    let forced = PhotoCamera::new(&env.profiles, &a1.meta, Some("sony-ilce-7m3"));
    assert_eq!(forced.matched.white_scale, [1.0; 3]);
}

#[test]
fn the_identity_scale_is_the_unscaled_computation_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let truth = light(PROFILE, 4800.0);
    let scene = Scene::new(512, 384, truth, Layout::Patchwork, 0.5).noise(2e-4, 1e-6);
    let path = scene.write(dir.path(), "own.ARW", OWN);
    let raw = ym_raw::decode::decode(&path).unwrap();
    let model = model_path().map(|p| IlluminantModel::from_path(&p).unwrap());
    for model in [None, model.as_ref()] {
        let plain = compute(&raw, model, None);
        let scaled = compute(&raw, model, Some([1.0; 3]));
        assert_eq!(plain.color, scaled.color, "model: {}", model.is_some());
        assert_eq!(plain.readout, scaled.readout);
        assert_eq!(plain.summary(), scaled.summary());
        assert_eq!(plain.exposure.ev, scaled.exposure.ev);
        assert_eq!(
            plain.initial_gains().unwrap().map(f64::to_bits),
            scaled.initial_gains().unwrap().map(f64::to_bits)
        );
    }
    // And a body with its own curves gets exactly that from the session path.
    let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
    assert_eq!(camera.matched.white_scale, [1.0; 3]);
}

#[test]
fn an_a1_photo_reads_its_light_on_the_profiles_scale_without_a_model() {
    let dir = tempfile::tempdir().unwrap();
    for cct in [2800.0, 3200.0, 5000.0, 5500.0, 6500.0, 7500.0] {
        let w = stand_in_scene(dir.path(), cct, Some(cct), Layout::Patchwork, false);
        // The as-shot fallback: the camera's white times the scale is the profile's white of the light.
        let est = compute(&w.raw, None, Some(w.scale));
        assert_eq!(est.illuminant.source, IlluminantSource::AsShotFallback);
        let got = est.color.w_scene_initial;
        for c in 0..3 {
            assert!(
                (got[c] / w.truth[c] - 1.0).abs() < 2e-3,
                "{cct} K channel {c}: {} against {}",
                got[c],
                w.truth[c]
            );
        }
        assert!(
            (est.summary().kelvin - cct).abs() <= 50.0,
            "{cct} K read as {}",
            est.summary().kelvin
        );
        // Without the scale the same file reads as another light: the error this exists to remove.
        let unscaled = compute(&w.raw, None, None);
        assert!(
            (unscaled.summary().kelvin - cct).abs() > 150.0,
            "{cct} K read as {} without the scale",
            unscaled.summary().kelvin
        );
        // The mosaic is balanced with the body's own white: g0 = 1 / w_body, whatever the profile.
        let g0 = est.initial_gains().unwrap();
        for c in 0..3 {
            let want = w.scale[c] / w.truth[c];
            assert!((g0[c] / want - 1.0).abs() < 2e-3, "{cct} K g0[{c}]");
        }
    }
}

#[test]
fn the_estimator_sees_the_body_on_the_profiles_scale() {
    let Some(path) = model_path() else {
        eprintln!("no illuminant model for {PROFILE}; skipping");
        return;
    };
    let model = IlluminantModel::from_path(&path).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (mut worst_scaled, mut best_unscaled) = (0.0f64, f64::INFINITY);
    for truth in [2800.0, 3200.0, 4300.0, 5500.0, 6500.0] {
        for as_shot in [Some(truth), None] {
            let w = stand_in_scene(dir.path(), truth, as_shot, Layout::Patchwork, true);
            let est = compute(&w.raw, Some(&model), Some(w.scale));
            assert!(matches!(
                est.illuminant.source,
                IlluminantSource::Model { .. }
            ));
            let err = angle_deg(est.color.w_scene_initial, w.truth);
            worst_scaled = worst_scaled.max(err);
            assert!(
                err < 1.5,
                "{truth} K (as shot {as_shot:?}) on the profile's scale: {err:.2} degrees"
            );
            assert!(
                (est.summary().kelvin - truth).abs() <= 250.0,
                "{truth} K read as {}",
                est.summary().kelvin
            );
            // The same file without the scale: the estimator is trained on the profile's raw, so the error is the
            // body's offset (the angle between the two whites) or more.
            let unscaled = compute(&w.raw, Some(&model), None);
            best_unscaled = best_unscaled.min(angle_deg(unscaled.color.w_scene_initial, w.truth));
        }
    }
    eprintln!("worst with the scale {worst_scaled:.2} degrees, best without {best_unscaled:.2}");
    assert!(best_unscaled > worst_scaled);
}

#[test]
fn pick_white_on_an_a1_photo_returns_the_light_on_the_profiles_scale() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    // The camera guessed 5500 K; the light is 3200 K and the picked patch is a neutral at 40 percent.
    let w = stand_in_scene(dir.path(), 3200.0, Some(5500.0), Layout::Uniform, false);
    let est = compute(&w.raw, None, Some(w.scale));
    let g0 = est.initial_gains().unwrap();
    let sample = pick_white_sample(&w.raw, [0.5, 0.5], g0).unwrap();
    let camera = PhotoCamera::new(&env.profiles, &w.raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    let chain = est
        .color
        .chain(
            ym_core::auto::tables_of(profile),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Auto,
            Some(sample),
        )
        .unwrap();
    let picked = chain.scene().w_scene;
    assert!(
        angle_deg(picked, w.truth) < 0.3,
        "picked {picked:?} against {:?}",
        w.truth
    );
    assert!(
        (chain.scene().cct_k - 3200.0).abs() < 60.0,
        "{}",
        chain.scene().cct_k
    );
    // The Auto balance then lands the picked patch on neutral: a neutral grey of the scene comes out equal in
    // the three channels (the sample is the g0-balanced raw; the chain maps it to 2^EV [1, 1, 1] * level).
    let out = chain.apply(sample).0;
    for c in 1..3 {
        assert!((out[c] / out[0] - 1.0).abs() < 2e-3, "{out:?}");
    }
    // A sample at the body's own clip level is refused (the body clips at g0 per channel).
    let at_clip = [0.95 * g0[0], 0.5 * g0[1], 0.5 * g0[2]];
    let err = est
        .color
        .chain(
            ym_core::auto::tables_of(profile),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Auto,
            Some(at_clip),
        )
        .unwrap_err();
    assert!(
        matches!(err, ChainError::Pick(PickWhiteError::TooBright)),
        "{err:?}"
    );
}

#[test]
fn an_unusable_scale_is_refused_before_anything_is_computed() {
    let dir = tempfile::tempdir().unwrap();
    let w = stand_in_scene(dir.path(), 5500.0, Some(5500.0), Layout::Uniform, false);
    let env = base_env();
    let camera = PhotoCamera::new(&env.profiles, &w.raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    for bad in [
        [0.0, 1.0, 1.0],
        [f64::NAN, 1.0, 1.0],
        [0.9, 1.1, 1.0],
        [1.0, 1.0, -2.0],
    ] {
        let err = AutoEstimates::compute_scaled(
            &w.raw,
            &AutoInputs {
                profile,
                illuminant_model: None,
                priors: &env.priors,
            },
            bad,
        )
        .unwrap_err();
        assert!(
            matches!(err, AutoError::White(WbError::InvalidWhite(_))),
            "{bad:?}: {err:?}"
        );
    }
}
