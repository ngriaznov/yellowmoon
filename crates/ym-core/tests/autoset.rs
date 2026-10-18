//! The automatic settings of the core (`docs/18-auto-settings.md`): the contrast setting and its JSON, the
//! preferred-white strength, the Auto contrast of a photo with and without the model, the grade's use of the effective
//! value, the burst median and the reset of the Automatic card.
//!
//! The model is the small test model of `crates/ym-color/tests/data/autoset/test_model.json`, injected into the
//! session's environment (it is never copied into `data/`).

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, base_env, light, settle};
use serde_json::json;
use ym_color::autoset::AutosetModel;
use ym_color::exposure::{BurstFrame, CAMERA_GAIN_EV, ShotSettings, burst_groups, share_burst_ev};
use ym_core::auto::AutosetEstimates;
use ym_core::pipeline::PhotoSource;
use ym_core::pipeline::color::grade_params;
use ym_core::session::{OpenOptions, PhotoId, Session, SessionEnv};
use ym_core::settings::{
    AutoOr, ContrastSetting, DevelopSettings, ExposureSetting, SettingsError, SettingsField,
    SettingsPatch, WHITE_PREFERENCE_DEFAULT, WhiteBalanceValue,
};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// The test model of `crates/ym-color/tests/data/autoset`.
fn test_model() -> AutosetModel {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../ym-color/tests/data/autoset/test_model.json");
    let text = std::fs::read_to_string(&path).expect("the test model");
    AutosetModel::from_json(&text).expect("the test model is valid")
}

/// Reads a settings document with the neutral look as the default look.
fn read(value: serde_json::Value) -> Result<(DevelopSettings, Vec<SettingsField>), SettingsError> {
    DevelopSettings::from_value(value, &LookId::neutral())
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

#[test]
fn contrast_reads_the_auto_or_form_and_a_plain_number_of_an_older_document() {
    assert_eq!(DevelopSettings::default().contrast, AutoOr::Auto);
    let auto = read(json!({ "contrast": { "kind": "auto" } })).unwrap().0;
    assert_eq!(auto.contrast, AutoOr::Auto);
    let manual = read(json!({ "contrast": { "kind": "manual", "value": 12 } }))
        .unwrap()
        .0;
    assert_eq!(manual.contrast, ContrastSetting::manual_value(12.0));
    // Documents written before autoset: a plain number is Manual with that value ...
    let old = read(json!({ "contrast": 12 })).unwrap().0;
    assert_eq!(old.contrast, ContrastSetting::manual_value(12.0));
    assert_eq!(
        read(json!({ "contrast": -7.5 })).unwrap().0.contrast,
        ContrastSetting::manual_value(-7.5)
    );
    // ... except 0, which was the default and reads as Auto.
    assert_eq!(
        read(json!({ "contrast": 0 })).unwrap().0.contrast,
        AutoOr::Auto
    );
    assert_eq!(
        read(json!({ "contrast": 0.0 })).unwrap().0.contrast,
        AutoOr::Auto
    );
    // A value that is not a number is refused, and so is an unknown field of the object.
    assert!(matches!(
        read(json!({ "contrast": "lots" })),
        Err(SettingsError::Parse(_))
    ));
    assert!(read(json!({ "contrast": { "kind": "manual", "value": 1, "extra": 2 } })).is_err());
    // Out of range is clamped to the range and reported.
    let (clamped, fields) =
        read(json!({ "contrast": { "kind": "manual", "value": 400 } })).unwrap();
    assert_eq!(clamped.contrast, ContrastSetting::manual_value(100.0));
    assert!(fields.contains(&SettingsField::Contrast));
}

#[test]
fn contrast_writes_the_auto_or_form_and_reads_it_back() {
    assert_eq!(
        serde_json::to_value(ContrastSetting::manual_value(12.0)).unwrap(),
        json!({ "kind": "manual", "value": 12.0 })
    );
    assert_eq!(
        serde_json::to_value(ContrastSetting::Auto).unwrap(),
        json!({ "kind": "auto" })
    );
    for contrast in [
        AutoOr::Auto,
        ContrastSetting::manual_value(12.0),
        ContrastSetting::manual_value(-100.0),
    ] {
        let s = DevelopSettings {
            contrast,
            ..DevelopSettings::default()
        };
        let value = serde_json::to_value(&s).unwrap();
        let (back, _) = DevelopSettings::from_value(value, &LookId::neutral()).unwrap();
        assert_eq!(back, s, "{contrast:?}");
    }
}

#[test]
fn white_preference_defaults_to_the_preferred_white_and_is_clamped_to_its_range() {
    assert_eq!(WHITE_PREFERENCE_DEFAULT, 1.0);
    assert_eq!(DevelopSettings::default().white_preference, 1.0);
    // A document without the field reads as the default.
    assert_eq!(
        read(json!({ "look": "builtin:neutral" }))
            .unwrap()
            .0
            .white_preference,
        1.0
    );
    assert_eq!(
        read(json!({ "white_preference": 0.25 }))
            .unwrap()
            .0
            .white_preference,
        0.25
    );
    let (high, fields) = read(json!({ "white_preference": 3 })).unwrap();
    assert_eq!(high.white_preference, 1.0);
    assert!(fields.contains(&SettingsField::WhitePreference));
    assert_eq!(
        read(json!({ "white_preference": -1 }))
            .unwrap()
            .0
            .white_preference,
        0.0
    );
}

#[test]
fn a_patch_takes_either_form_of_contrast_and_the_preference() {
    let number: SettingsPatch = serde_json::from_value(json!({ "contrast": 12 })).unwrap();
    assert_eq!(number.contrast, Some(ContrastSetting::manual_value(12.0)));
    let auto: SettingsPatch =
        serde_json::from_value(json!({ "contrast": { "kind": "auto" } })).unwrap();
    assert_eq!(auto.contrast, Some(AutoOr::Auto));
    let absent: SettingsPatch = serde_json::from_value(json!({ "contrast": null })).unwrap();
    assert_eq!(absent.contrast, None);
    let preference: SettingsPatch =
        serde_json::from_value(json!({ "white_preference": 0.5 })).unwrap();
    assert_eq!(preference.white_preference, Some(0.5));

    let mut s = DevelopSettings::default();
    let patch = SettingsPatch {
        contrast: Some(ContrastSetting::manual_value(40.0)),
        white_preference: Some(0.5),
        ..SettingsPatch::default()
    };
    let report = s.apply_patch(&patch, None).unwrap();
    assert_eq!(s.contrast, ContrastSetting::manual_value(40.0));
    assert_eq!(s.white_preference, 0.5);
    assert!(report.changed.contains(&SettingsField::Contrast));
    assert!(report.changed.contains(&SettingsField::WhitePreference));
    s.apply_patch(
        &SettingsPatch {
            contrast: Some(AutoOr::Auto),
            ..SettingsPatch::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(s.contrast, AutoOr::Auto, "a patch of Auto is the Reset");
}

#[test]
fn re_run_on_the_automatic_card_resets_contrast_with_the_other_automatic_values() {
    let mut s = DevelopSettings {
        contrast: ContrastSetting::manual_value(-20.0),
        exposure: ExposureSetting::manual_ev(0.5),
        white_balance: AutoOr::Manual(WhiteBalanceValue {
            kelvin: 4200.0,
            tint: 3.0,
        }),
        white_offset_ev: 1.0,
        white_preference: 0.25,
        ..DevelopSettings::default()
    };
    s.reset_automatic();
    assert_eq!(s.contrast, AutoOr::Auto);
    assert_eq!(s.exposure, AutoOr::Auto);
    assert_eq!(s.white_balance, AutoOr::Auto);
    assert_eq!(s.white_offset_ev, 0.0);
    // The preference is not an automatic value, so Re-run keeps it.
    assert_eq!(s.white_preference, 0.25);

    // The field's own Reset (the Light tab's Reset and the Tone card's) also returns contrast to Auto.
    let mut t = DevelopSettings {
        contrast: ContrastSetting::manual_value(7.0),
        ..DevelopSettings::default()
    };
    let reset = SettingsPatch::reset(SettingsField::Contrast, &LookId::neutral());
    t.apply_patch(&reset, None).unwrap();
    assert_eq!(t.contrast, AutoOr::Auto);
    assert!(
        SettingsPatch::reset(SettingsField::Contrast, &LookId::neutral())
            .contrast
            .is_some()
    );
}

// ---------------------------------------------------------------------------------------------
// Estimates and the session
// ---------------------------------------------------------------------------------------------

/// Writes a synthetic photo of brightness `level` taken with the shot of `burst_settings`, at `capture_time`.
fn write_photo(dir: &Path, name: &str, level: f64, capture_time: Option<&str>) -> PathBuf {
    let scene = Scene::new(256, 192, light(PROFILE, 5200.0), Layout::Patchwork, level);
    let mut builder = scene.builder(MODEL);
    if let Some(time) = capture_time {
        builder = builder
            .capture_time(time, "00", None)
            .exposure_time(1, 250)
            .f_number(8, 1)
            .iso(100);
    }
    let path = dir.join(name);
    builder
        .write_to(&path, &scene.mosaic())
        .expect("write the synthetic ARW");
    path
}

/// A session over `dir` with `env`, every photo estimated.
fn develop(env: SessionEnv, dir: &Path) -> Session {
    let session = Session::new(env);
    let outcome = session.open(&[dir.to_path_buf()], OpenOptions::default());
    assert!(outcome.message().is_none(), "{:?}", outcome.message());
    settle(&session);
    session
}

#[test]
fn with_the_test_model_a_photo_takes_the_model_values_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    write_photo(dir.path(), "DSC00001.ARW", 0.3, None);
    let env = base_env().with_autoset(test_model());
    let session = develop(env.clone(), dir.path());
    let id: PhotoId = session.photo_ids()[0];
    let est = session.estimates(id).expect("estimates");

    let version = test_model().version().to_owned();
    assert_eq!(est.autoset.version, version);
    assert!(est.autoset.ev_confidence > 0.0 && est.autoset.ev_confidence <= 1.0);
    assert!(est.autoset.contrast_confidence > 0.0 && est.autoset.contrast_confidence <= 1.0);
    assert!(
        est.autoset.contrast.abs() <= 30.0,
        "bounded to the contrast limit"
    );
    assert!(
        (est.autoset.contrast * 10.0 - (est.autoset.contrast * 10.0).round()).abs() < 1e-9,
        "a multiple of 0.1"
    );
    // The exposure before the highlight guard is the model's, not the camera gain.
    assert_ne!(est.exposure.meter.ev_before_guard, CAMERA_GAIN_EV);
    // The summary the bridge shows carries the same numbers.
    let summary = session.settings(id).unwrap().estimates;
    assert_eq!(summary.autoset_version, version);
    assert_eq!(summary.contrast, est.autoset.contrast);
    assert_eq!(summary.ev_confidence, est.autoset.ev_confidence);
    assert_eq!(summary.contrast_confidence, est.autoset.contrast_confidence);

    // The same file with the same model gives the same bits, in another session.
    let again = develop(base_env().with_autoset(test_model()), dir.path());
    let twice = again.estimates(again.photo_ids()[0]).unwrap();
    assert_eq!(twice.autoset, est.autoset);
    assert_eq!(twice.exposure.meter, est.exposure.meter);
}

#[test]
fn without_a_model_a_photo_gets_exactly_the_rule() {
    let dir = tempfile::tempdir().unwrap();
    write_photo(dir.path(), "DSC00001.ARW", 0.3, None);
    let session = develop(base_env(), dir.path());
    let id = session.photo_ids()[0];
    let est = session.estimates(id).expect("estimates");

    assert_eq!(est.autoset, AutosetEstimates::rule());
    assert_eq!(est.autoset.version, "none");
    assert_eq!(est.autoset.contrast, 0.0);
    assert_eq!(est.autoset.ev_confidence, 0.0);
    assert_eq!(est.autoset.contrast_confidence, 0.0);
    assert_eq!(est.exposure.meter.ev_before_guard, CAMERA_GAIN_EV);
    let summary = session.settings(id).unwrap().estimates;
    assert_eq!(summary.autoset_version, "none");
    assert_eq!(summary.contrast, 0.0);
}

#[test]
fn an_auto_contrast_is_the_effective_contrast_and_a_manual_one_overrides_it() {
    let dir = tempfile::tempdir().unwrap();
    write_photo(dir.path(), "DSC00001.ARW", 0.3, None);
    let session = develop(base_env().with_autoset(test_model()), dir.path());
    let id = session.photo_ids()[0];
    let auto = session.settings(id).unwrap();
    assert!(auto.auto.contrast);
    assert_eq!(auto.values.contrast, auto.estimates.contrast);

    // The grade uses the same effective value: the render, the thumbnails and the exports all read it.
    let source = PhotoSource::from_session(&session, id).unwrap();
    assert_eq!(
        grade_params(&source.settings, source.estimates.autoset.contrast).contrast,
        auto.values.contrast
    );

    session
        .update_settings(
            id,
            &SettingsPatch {
                contrast: Some(ContrastSetting::manual_value(12.0)),
                ..SettingsPatch::default()
            },
        )
        .unwrap();
    let manual = session.settings(id).unwrap();
    assert!(!manual.auto.contrast);
    assert_eq!(manual.values.contrast, 12.0);
    assert_eq!(
        manual.estimates.contrast, auto.estimates.contrast,
        "the estimate is still reported"
    );
    let source = PhotoSource::from_session(&session, id).unwrap();
    assert_eq!(
        grade_params(&source.settings, source.estimates.autoset.contrast).contrast,
        12.0
    );

    // Re-run returns the contrast to Auto with the other automatic values.
    session.rerun_auto(id).unwrap();
    let rerun = session.settings(id).unwrap();
    assert!(rerun.auto.contrast);
    assert_eq!(rerun.values.contrast, auto.values.contrast);
}

#[test]
fn a_burst_shares_the_median_auto_contrast_rounded_to_a_tenth() {
    // The median of the frames' contrasts, rounded to a tenth (docs/18 section 4.4), as the session does it.
    let frame = |t: f64| BurstFrame {
        camera: "SONY|ILCE-7RM4".to_owned(),
        capture_time_s: Some(t),
        settings: ShotSettings {
            shutter_s: 0.004,
            f_number: 8.0,
            iso: 100.0,
            exposure_compensation_ev: 0.0,
        },
    };
    let frames = [frame(0.0), frame(1.0), frame(2.0), frame(60.0)];
    let groups = burst_groups(&frames);
    assert_eq!(
        groups,
        [0, 0, 0, 1],
        "three frames within 2 s, one a minute later"
    );
    let shared = share_burst_ev(&frames, &[10.0, 11.0, 30.0, -4.0]);
    assert_eq!(
        shared,
        [11.0, 11.0, 11.0, -4.0],
        "the median of a burst, and a lone frame is its own"
    );
    // An even burst: the mean of the middle two, rounded to a tenth.
    let even = share_burst_ev(&frames[..2], &[10.0, 10.04]);
    assert_eq!(even, [10.02, 10.02]);
    assert_eq!(ym_color::autoset::round_contrast(even[0]), 10.0);
    assert_eq!(ym_color::autoset::round_contrast(11.46), 11.5);
}

#[test]
fn photos_of_one_burst_share_the_rounded_median_contrast() {
    // Two frames of one shot (same time and settings) with different brightness, so their own contrasts differ.
    let dir = tempfile::tempdir().unwrap();
    let time = "2026:10:18 10:00:00";
    write_photo(dir.path(), "DSC00001.ARW", 0.2, Some(time));
    write_photo(dir.path(), "DSC00002.ARW", 0.5, Some(time));
    let session = develop(base_env().with_autoset(test_model()), dir.path());
    let ids = session.photo_ids();
    assert_eq!(ids.len(), 2);
    let own: Vec<f64> = ids
        .iter()
        .map(|id| session.estimates(*id).unwrap().autoset.own_contrast)
        .collect();
    let median = ym_color::autoset::round_contrast((own[0] + own[1]) / 2.0);
    for id in ids {
        let served = session.estimates(id).unwrap();
        assert_eq!(
            served.autoset.contrast, median,
            "the burst's median contrast"
        );
    }
}
