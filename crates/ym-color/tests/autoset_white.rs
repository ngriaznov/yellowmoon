//! The preferred white in the chain's Auto arm (`docs/18-auto-settings.md` section 5): what the `white_preference`
//! strength changes, and what it leaves exactly as it was (the chain before `autoset`, Manual, Pick white, the light
//! presets, a daylight scene).
//!
//! Uses the real `data/cameras/profiles/*/profile.json` tables, like `tests/chain.rs`.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::chain::{ChainSettings, ColorEstimates, FusedChain, ProfileTables};
use ym_color::exposure::ExposureSetting;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus};
use ym_color::wb::{self, LightSource, SceneLight, WhiteBalance};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Profile {
    locus: Locus,
    white: AlexaWhite,
    maps: SensorMapTables,
}

impl Profile {
    fn tables(&self) -> ProfileTables<'_> {
        ProfileTables {
            sensor_map: &self.maps,
            alexa_white: &self.white,
            locus: &self.locus,
        }
    }
}

fn all_profiles() -> Vec<Profile> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    assert_eq!(ids.len(), 11, "the eleven shipped profiles");
    ids.iter()
        .map(|id| {
            let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
            let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            Profile {
                locus: serde_json::from_value(v["locus"].clone()).unwrap(),
                white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
                maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
            }
        })
        .collect()
}

/// A scene light with exactly known CCT and Duv, on the profile's locus.
fn scene(p: &Profile, cct_k: f64, duv: f64) -> SceneLight {
    let class = if cct_k < 4000.0 {
        LightClass::Tungsten
    } else {
        LightClass::Daylight
    };
    SceneLight::from_parts(
        p.locus.cct_duv_to_rgb(cct_k, duv),
        ClassProbabilities::only(class),
        0.8,
        cct_k,
        duv,
    )
    .unwrap()
}

/// The chain of an Auto white balance and EV 0 with the given preference and brightness.
fn auto_chain(
    p: &Profile,
    scene: &SceneLight,
    white_preference: f64,
    key_bv: Option<f64>,
) -> FusedChain {
    let w_init = scene.w_scene;
    FusedChain::build(
        p.tables(),
        scene,
        w_init,
        &ChainSettings {
            white_preference,
            key_bv,
            ..ChainSettings::default()
        },
    )
    .unwrap()
}

fn max_relative_difference(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut worst: f64 = 0.0;
    for (ra, rb) in a.iter().zip(b) {
        for (x, y) in ra.iter().zip(rb) {
            let scale = x.abs().max(y.abs()).max(1e-12);
            worst = worst.max((x - y).abs() / scale);
        }
    }
    worst
}

#[test]
fn a_zero_preference_is_the_neutral_auto_white_of_the_chain_before_autoset() {
    for p in all_profiles() {
        for (cct, duv) in [(3000.0, 0.0), (2850.0, 0.004), (4200.0, -0.002), (6500.0, 0.0)] {
            let s = scene(&p, cct, duv);
            let chain = auto_chain(&p, &s, 0.0, Some(8.0));
            // The Auto arm of the chain before `autoset`: C3 and C4 cancel exactly, the user's CCT is the scene's,
            // and the readout is the scene's own.
            assert_eq!(chain.stages().c3_c4, [1.0; 3], "cct {cct} K duv {duv}");
            assert_eq!(chain.stages().user_cct_k, s.cct_k, "cct {cct} K");
            assert_eq!(
                chain.readout(),
                wb::auto_readout(&p.white, &p.locus, &s),
                "cct {cct} K"
            );
        }
    }
}

#[test]
fn a_daylight_scene_is_unchanged_at_full_strength() {
    // Above the reference (6000 K) the preferred white is the scene's own light: the chain is the same bit for
    // bit at every strength, and with or without a brightness.
    for p in all_profiles() {
        // 5500 K is warmer than the 6000 K reference, so the rule applies to it; the test starts above the reference.
        for (cct, duv) in [(6500.0, 0.0), (6000.0, 0.003), (7500.0, -0.004), (10000.0, 0.0)] {
            let s = scene(&p, cct, duv);
            let neutral = auto_chain(&p, &s, 0.0, Some(9.0));
            assert_eq!(
                auto_chain(&p, &s, 1.0, Some(9.0)),
                neutral,
                "cct {cct} K duv {duv}"
            );
            assert_eq!(auto_chain(&p, &s, 1.0, None), neutral, "cct {cct} K, no brightness");
        }
    }
}

#[test]
fn a_warm_scene_keeps_part_of_its_warmth_at_full_strength() {
    // `docs/18` section 5's worked number: a 3000 K scene in a bright scene has D = 0.70, so the user's white is
    // the light at mired 283 (about 3530 K); the scene is balanced for that light, not for 3000 K.
    for p in all_profiles() {
        let s = scene(&p, 3000.0, 0.0);
        let chain = auto_chain(&p, &s, 1.0, Some(8.0));
        let c3_c4 = chain.stages().c3_c4;
        assert!(
            c3_c4 != [1.0; 3],
            "the warm scene's C4 product is not neutral (a profile gave {c3_c4:?})"
        );
        // Red keeps more of the light than blue: the photo stays warm.
        assert!(
            c3_c4[0] > c3_c4[2],
            "warmth is kept: red gain {} over blue {}",
            c3_c4[0],
            c3_c4[2]
        );
        let user_k = chain.stages().user_cct_k;
        assert!(
            (user_k - 1e6 / 283.333_333).abs() < 1.0,
            "the user's CCT is {user_k} K, expected about 3529 K"
        );
        let readout = chain.readout();
        assert!(
            (readout.kelvin_k - 3550.0).abs() <= 50.0,
            "the Auto readout shows {} K, expected about 3500 K",
            readout.kelvin_k
        );
    }
}

#[test]
fn the_readout_as_manual_gives_nearly_the_auto_picture() {
    // The Auto readout is the user's light, so Manual at the shown Kelvin and tint renders nearly the same picture
    // (the Kelvin is rounded to 50 K and the tint to a whole unit).
    let p = all_profiles().into_iter().next().unwrap();
    let s = scene(&p, 3000.0, 0.0);
    let auto = auto_chain(&p, &s, 1.0, Some(8.0));
    let readout = auto.readout();
    let manual = FusedChain::build(
        p.tables(),
        &s,
        s.w_scene,
        &ChainSettings {
            white_balance: WhiteBalance::Manual {
                kelvin: readout.kelvin_k,
                tint: readout.tint,
            },
            ..ChainSettings::default()
        },
    )
    .unwrap();
    // The colour the two render: a grey patch and a coloured one, balanced with the initial gains (each channel of
    // the output relative to the output's largest channel, so a brightness difference is not counted).
    let mut worst: f64 = 0.0;
    for rgb in [[0.3, 0.3, 0.3], [0.2, 0.05, 0.1], [0.05, 0.25, 0.1]] {
        let a = auto.apply(rgb).0;
        let m = manual.apply(rgb).0;
        let (na, nm) = (a.iter().fold(0.0_f64, |x, v| x.max(v.abs())), m.iter().fold(0.0_f64, |x, v| x.max(v.abs())));
        for c in 0..3 {
            worst = worst.max((a[c] / na - m[c] / nm).abs());
        }
    }
    assert!(
        worst < 0.025,
        "Manual at the readout moves the picture by {:.2} % of its largest channel",
        worst * 100.0
    );
}

#[test]
fn a_picked_white_is_neutral_whatever_the_preference() {
    // Pick white renders the spot neutral (docs/11): the preferred white does not apply to a picked white.
    let p = all_profiles().into_iter().next().unwrap();
    let s = scene(&p, 3000.0, 0.0);
    let w_init = s.w_scene;
    // A 30 % grey patch of the scene light, balanced with the initial gains (the sample Pick white takes; the white
    // itself would be clipped and refused).
    let g0 = wb::initial_gains(w_init).unwrap();
    let sample = [0.3 * g0[0], 0.3 * g0[1], 0.3 * g0[2]];
    let build = |white_preference: f64| {
        FusedChain::build(
            p.tables(),
            &s,
            w_init,
            &ChainSettings {
                picked_white: Some(sample),
                white_preference,
                key_bv: Some(8.0),
                ..ChainSettings::default()
            },
        )
        .unwrap()
    };
    assert_eq!(build(1.0), build(0.0));
}

#[test]
fn a_light_preset_is_unchanged_at_full_strength() {
    // A light preset sets the user's white to the preset (docs/02, C1): the preferred white is for Auto light only.
    let p = all_profiles().into_iter().next().unwrap();
    let s = scene(&p, 3000.0, 0.0);
    for light_source in [LightSource::Tungsten, LightSource::Day, LightSource::Led] {
        let build = |white_preference: f64| {
            FusedChain::build(
                p.tables(),
                &s,
                s.w_scene,
                &ChainSettings {
                    light_source,
                    white_preference,
                    key_bv: Some(8.0),
                    ..ChainSettings::default()
                },
            )
            .unwrap()
        };
        assert_eq!(build(1.0), build(0.0), "{light_source:?}");
    }
}

#[test]
fn a_manual_white_balance_ignores_the_preference() {
    let p = all_profiles().into_iter().next().unwrap();
    let s = scene(&p, 3000.0, 0.0);
    let build = |white_preference: f64| {
        FusedChain::build(
            p.tables(),
            &s,
            s.w_scene,
            &ChainSettings {
                white_balance: WhiteBalance::Manual {
                    kelvin: 4200.0,
                    tint: 5.0,
                },
                white_preference,
                key_bv: Some(8.0),
                ..ChainSettings::default()
            },
        )
        .unwrap()
    };
    assert_eq!(build(1.0), build(0.0));
}

#[test]
fn color_estimates_pass_the_preference_and_the_brightness_to_the_chain() {
    let p = all_profiles().into_iter().next().unwrap();
    let s = scene(&p, 3000.0, 0.0);
    let estimates = ColorEstimates {
        scene: s,
        w_scene_initial: s.w_scene,
        white_scale: [1.0; 3],
        ev_auto: 0.0,
        key_bv: Some(8.0),
    };
    let through_estimates = estimates
        .chain(
            p.tables(),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Auto,
            1.0,
            None,
        )
        .unwrap();
    let direct = auto_chain(&p, &s, 1.0, Some(8.0));
    assert_eq!(through_estimates.matrix(), direct.matrix());
    // The metering chain is the neutral one, whatever the estimates say (the model's features are metered on it).
    let metering = estimates.metering_chain(p.tables()).unwrap();
    assert_eq!(metering.stages().c3_c4, [1.0; 3]);
}
