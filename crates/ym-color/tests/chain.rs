//! M1-07, the fused colour chain C1 to C6 on the shipped profiles (`docs/02-colour-pipeline.md`,
//! section "Stage order"): Auto white balance, Manual Kelvin and tint, the light presets, Pick white,
//! exposure, the clip level.
//!
//! Uses the real `data/cameras/profiles/*/profile.json` tables and nothing from the git-ignored fixtures,
//! so it always runs. The independent spectral simulation of the done-when criterion is in
//! `tests/chain_spectral.rs`.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::arri::ArriTable;
use ym_color::chain::{
    ChainError, ChainSettings, ColorEstimates, FusedChain, ProfileTables, clip_level,
};
use ym_color::consts::linalg::{self, Mat3};
use ym_color::exposure::ExposureSetting;
use ym_color::gamut;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus};
use ym_color::wb::{
    self, DAY_PRESET_K, LightSource, PickWhiteError, SHADE_PRESET_K, SceneLight, TUNGSTEN_PRESET_K,
    WbError, WhiteBalance,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Profile {
    id: String,
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

fn profile_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

fn load(id: &str) -> Profile {
    let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Profile {
        id: id.to_string(),
        locus: serde_json::from_value(v["locus"].clone()).unwrap(),
        white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
        maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
    }
}

fn all_profiles() -> Vec<Profile> {
    let profiles: Vec<Profile> = profile_ids().iter().map(|id| load(id)).collect();
    assert_eq!(profiles.len(), 11);
    profiles
}

fn class_for(cct_k: f64) -> LightClass {
    if cct_k < 4000.0 {
        LightClass::Tungsten
    } else {
        LightClass::Daylight
    }
}

/// An estimate `duv` off the locus at `cct_k`, read back through the profile's locus.
fn estimate_at(p: &Profile, cct_k: f64, duv: f64, classes: ClassProbabilities) -> SceneLight {
    SceneLight::new(p.locus.cct_duv_to_rgb(cct_k, duv), classes, 0.8, &p.locus).unwrap()
}

/// An estimate with exactly known CCT and Duv.
fn exact_estimate(p: &Profile, cct_k: f64, duv: f64, classes: ClassProbabilities) -> SceneLight {
    SceneLight::from_parts(p.locus.cct_duv_to_rgb(cct_k, duv), classes, 0.8, cct_k, duv).unwrap()
}

fn auto_settings(ev: f64) -> ChainSettings {
    ChainSettings {
        exposure_ev: ev,
        ..ChainSettings::default()
    }
}

fn manual(kelvin: f64, tint: f64, ev: f64) -> ChainSettings {
    ChainSettings {
        white_balance: WhiteBalance::Manual { kelvin, tint },
        exposure_ev: ev,
        ..ChainSettings::default()
    }
}

/// The g0-balanced camera RGB of the scene white: `w_scene / w_scene_initial`.
fn balanced_white(scene: &SceneLight, w_init: [f64; 3]) -> [f64; 3] {
    [
        scene.w_scene[0] / w_init[0],
        scene.w_scene[1] / w_init[1],
        scene.w_scene[2] / w_init[2],
    ]
}

fn max_abs_diff(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).fold(0.0, |m, k| m.max((a[k] - b[k]).abs()))
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

#[test]
fn auto_white_balance_puts_the_scene_white_on_an_awg4_neutral() {
    // The done-when of the Auto mode: C3 and C4 cancel, the sensor map preserves white, the ARRI
    // matrices and AWG3 to AWG4 keep it balanced, so the scene white lands on 2^EV [1, 1, 1].
    let mut worst = 0.0_f64;
    for p in all_profiles() {
        let mixes = [
            ClassProbabilities::only(LightClass::Daylight),
            ClassProbabilities::only(LightClass::Tungsten),
            ClassProbabilities::only(LightClass::Fluorescent),
            ClassProbabilities::only(LightClass::Led),
            ClassProbabilities::new(0.4, 0.3, 0.2, 0.1).unwrap(),
            ClassProbabilities::new(0.0, 0.5, 0.0, 0.5).unwrap(),
        ];
        for cct_k in [
            1900.0, 2400.0, 3100.0, 3900.0, 4100.0, 5000.0, 6500.0, 9000.0, 12_000.0,
        ] {
            for (m, classes) in mixes.iter().enumerate() {
                for duv in [-0.01, 0.0, 0.007] {
                    let estimate = estimate_at(&p, cct_k, duv, *classes);
                    // The mosaic was balanced for a different light than the scene.
                    let w_init = p.locus.rgb_at(if cct_k < 5000.0 { 6200.0 } else { 3300.0 });
                    for ev in [0.0, 1.3, -2.0] {
                        let chain =
                            FusedChain::build(p.tables(), &estimate, w_init, &auto_settings(ev))
                                .unwrap();
                        let white = balanced_white(chain.scene(), w_init);
                        let out = chain.apply(white).0;
                        let gain = ev.exp2();
                        let err = max_abs_diff(out, [gain; 3]) / gain;
                        worst = worst.max(err);
                        assert!(
                            err < 1e-12,
                            "{} {cct_k} K mix {m} duv {duv} EV {ev}: {out:?}",
                            p.id
                        );
                    }
                }
            }
        }
    }
    eprintln!("Auto: scene white vs AWG4 neutral, worst relative error {worst:.2e}");
}

#[test]
fn c3_and_c4_cancel_exactly_in_auto() {
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        4600.0,
        0.004,
        ClassProbabilities::new(0.6, 0.0, 0.1, 0.3).unwrap(),
    );
    let chain =
        FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(0.0)).unwrap();
    let s = chain.stages();
    assert_eq!(s.c3_c4, [1.0, 1.0, 1.0]);
    // g_user is the inverse of the class-blended, tinted w_A(E_scene), not of the model at the CCT.
    for k in 0..3 {
        assert!((s.g_user[k] * s.w_a_scene[k] - 1.0).abs() < 1e-15);
    }
    assert_eq!(s.user_cct_k, chain.scene().cct_k);
    assert_eq!(
        s.g_res,
        [1.0, 1.0, 1.0],
        "the mosaic was balanced for the estimate"
    );
    // With nothing to undo, the matrix is the ARRI step and the sensor map: 2^0 AWG4 * ARRI(CCT) * M_map.
    let arri = ArriTable::builtin().native_to_awg4(chain.scene().cct_k);
    let expect = linalg::mul(&arri, &s.sensor_map);
    assert!(linalg::max_abs_diff(chain.matrix(), &expect) < 1e-14);
}

#[test]
fn an_led_scene_is_neutralised_by_its_own_class_white() {
    // docs/03: Auto must take g_user = 1 / scene_white (the LED list), not the model at the CCT, or the
    // LED scene keeps a residual cast.
    let p = load("sony-ilce-7rm4");
    let classes = ClassProbabilities::only(LightClass::Led);
    let estimate = exact_estimate(&p, 4100.0, 0.0, classes);
    let chain =
        FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(0.0)).unwrap();
    let out = chain.apply([1.0; 3]).0;
    assert!(max_abs_diff(out, [1.0; 3]) < 1e-12, "{out:?}");
    // The model's white at 4100 K would not have done it.
    let model = p.white.locus_white(4100.0);
    let led = p.white.scene_white(4100.0, &classes);
    assert!((model[0] / led[0] - 1.0).abs() > 1e-3 || (model[2] / led[2] - 1.0).abs() > 1e-3);
    assert_eq!(chain.stages().w_a_scene, led);
}

#[test]
fn the_matrix_is_the_documented_product_for_random_pixels() {
    // M = 2^EV * M_awg3->awg4 * M_arri(CCT_user) * diag(g_user) * diag(w_A(E_scene)) * M_map * diag(g_res),
    // applied stage by stage.
    let mut rng = Rng(0x5EED_C0DE);
    let arri = ArriTable::builtin();
    for p in all_profiles() {
        let classes = ClassProbabilities::new(0.5, 0.1, 0.1, 0.3).unwrap();
        let estimate = estimate_at(&p, 4300.0, -0.003, classes);
        let w_init = p.locus.rgb_at(5600.0);
        for settings in [
            auto_settings(0.4),
            manual(3200.0, 0.0, 0.4),
            manual(7777.0, -23.0, -1.0),
            ChainSettings {
                light_source: LightSource::Tungsten,
                white_balance: WhiteBalance::Manual {
                    kelvin: 5600.0,
                    tint: 9.0,
                },
                exposure_ev: 0.0,
                picked_white: None,
                white_scale: [1.0; 3],
                white_shift_mired: 0.0,
                tint_shift: 0.0,
            },
        ] {
            let chain = FusedChain::build(p.tables(), &estimate, w_init, &settings).unwrap();
            let s = chain.stages();
            for _ in 0..50 {
                let x = [
                    rng.next_f64() * 2.0,
                    rng.next_f64() * 2.0,
                    rng.next_f64() * 2.0,
                ];
                let c1 = [x[0] * s.g_res[0], x[1] * s.g_res[1], x[2] * s.g_res[2]];
                let c2 = linalg::mul_vec(&s.sensor_map, c1);
                let c3 = [
                    c2[0] * s.w_a_scene[0],
                    c2[1] * s.w_a_scene[1],
                    c2[2] * s.w_a_scene[2],
                ];
                let c4 = [
                    c3[0] * s.g_user[0],
                    c3[1] * s.g_user[1],
                    c3[2] * s.g_user[2],
                ];
                let c5 = linalg::mul_vec(&arri.native_to_awg3(s.user_cct_k), c4);
                let c6 = linalg::mul_vec(&gamut::matrices().awg3_to_awg4, c5);
                let expect = c6.map(|v| v * settings.exposure_ev.exp2());
                let got = chain.apply(x).0;
                for k in 0..3 {
                    assert!(
                        (got[k] - expect[k]).abs() < 1e-13 * (1.0 + expect[k].abs()),
                        "{} {settings:?}: {got:?} vs {expect:?}",
                        p.id
                    );
                }
            }
        }
    }
}

#[test]
fn manual_white_balance_uses_the_users_white_and_the_users_arri_matrix() {
    let p = load("sony-ilce-7rm4");
    let estimate = exact_estimate(
        &p,
        5200.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w_init = estimate.w_scene;
    let arri = ArriTable::builtin();
    let auto = FusedChain::build(p.tables(), &estimate, w_init, &auto_settings(0.0)).unwrap();
    for (kelvin, tint) in [
        (2000.0, -50.0),
        (3200.0, 0.0),
        (3999.0, 0.0),
        (4000.0, 12.0),
        (5600.0, 50.0),
        (11000.0, 0.0),
    ] {
        let chain =
            FusedChain::build(p.tables(), &estimate, w_init, &manual(kelvin, tint, 0.0)).unwrap();
        let s = chain.stages();
        // The sensor map and the scene white depend on the scene light only.
        assert_eq!(s.sensor_map, auto.stages().sensor_map);
        assert_eq!(s.w_a_scene, auto.stages().w_a_scene);
        assert_eq!(s.g_res, [1.0; 3]);
        // The user's side.
        let w_u = p.white.locus_white(kelvin);
        assert_eq!(
            s.g_user,
            [1.0 / w_u[0], 2.0_f64.powf(-tint / 100.0), 1.0 / w_u[2]]
        );
        assert_eq!(s.user_cct_k, kelvin);
        assert_eq!(s.arri_native_to_awg3, arri.native_to_awg3(kelvin));
        assert_eq!(chain.readout().kelvin_k, kelvin);
        assert_eq!(chain.readout().tint, tint);
        // The scene white goes where the ALEXA balanced for `kelvin` would put it.
        let expect = {
            let native = linalg::mul_vec(&linalg::diag(s.g_user), s.w_a_scene);
            linalg::mul_vec(&arri.native_to_awg4(kelvin), native)
        };
        assert!(
            max_abs_diff(chain.apply([1.0; 3]).0, expect) < 1e-13,
            "{kelvin} K tint {tint}"
        );
    }
}

#[test]
fn manual_at_the_shown_values_is_auto_up_to_the_tint_brightness() {
    // docs/02: the tint shown in Auto reproduces the scene's distance from the locus. For a daylight or
    // tungsten scene with an exactly known CCT that is a multiple of 50 K, `Manual` at the shown pair
    // is Auto times the green gain 2^(-t/100) (the tint scales the green gain, not red and blue).
    for p in all_profiles() {
        for cct_k in [2750.0, 3350.0, 4300.0, 5250.0, 6500.0, 9000.0] {
            for duv in [-0.009, -0.003, 0.0, 0.004, 0.01] {
                let estimate =
                    exact_estimate(&p, cct_k, duv, ClassProbabilities::only(class_for(cct_k)));
                let w_init = p.locus.rgb_at(5000.0);
                let auto =
                    FusedChain::build(p.tables(), &estimate, w_init, &auto_settings(0.0)).unwrap();
                let shown = auto.readout();
                assert_eq!(shown.kelvin_k, cct_k);
                let chain = FusedChain::build(
                    p.tables(),
                    &estimate,
                    w_init,
                    &manual(shown.kelvin_k, shown.tint, 0.0),
                )
                .unwrap();
                let g = wb::tint_green_gain(shown.tint);
                let expect = auto.matrix().map(|row| row.map(|v| v * g));
                let d = linalg::max_abs_diff(chain.matrix(), &expect);
                assert!(
                    d < 1e-12,
                    "{} {cct_k} K duv {duv}: tint {:.2}, {d:e}",
                    p.id,
                    shown.tint
                );
            }
        }
    }
}

#[test]
fn manual_at_the_shown_values_stays_close_to_auto_for_led_light() {
    // The LED class white sits off the locus in a way the green gain alone cannot cancel (red against
    // blue), so the pair shown for an LED scene is an approximation: bound it.
    let p = load("sony-ilce-7rm4");
    let mut worst = 0.0_f64;
    for cct_k in [2750.0, 3500.0, 4100.0, 5000.0, 6500.0] {
        let estimate = exact_estimate(&p, cct_k, 0.0, ClassProbabilities::only(LightClass::Led));
        let w_init = estimate.w_scene;
        let auto = FusedChain::build(p.tables(), &estimate, w_init, &auto_settings(0.0)).unwrap();
        let shown = auto.readout();
        let chain = FusedChain::build(
            p.tables(),
            &estimate,
            w_init,
            &manual(shown.kelvin_k, shown.tint, 0.0),
        )
        .unwrap();
        let g = wb::tint_green_gain(shown.tint);
        let white = chain.apply([1.0; 3]).0.map(|v| v / g);
        worst = worst.max(max_abs_diff(white, [1.0; 3]));
    }
    eprintln!("LED: Manual at the shown pair, scene white off neutral by {worst:.4}");
    assert!(worst < 0.05, "{worst}");
}

#[test]
fn presets_choose_the_sensor_map_class_and_the_user_cct() {
    // docs/02 C1 table. The estimate is a 4600 K light with a cast and mixed classes; the presets ignore it.
    for p in all_profiles() {
        let estimate = estimate_at(
            &p,
            4600.0,
            0.004,
            ClassProbabilities::new(0.5, 0.1, 0.3, 0.1).unwrap(),
        );
        let w_init = estimate.w_scene;
        for (preset, cct_k, class) in [
            (LightSource::Day, DAY_PRESET_K, LightClass::Daylight),
            (LightSource::Shade, SHADE_PRESET_K, LightClass::Daylight),
            (
                LightSource::Tungsten,
                TUNGSTEN_PRESET_K,
                LightClass::Tungsten,
            ),
        ] {
            let settings = ChainSettings {
                light_source: preset,
                ..ChainSettings::default()
            };
            let chain = FusedChain::build(p.tables(), &estimate, w_init, &settings).unwrap();
            let s = chain.stages();
            // The map is the class's table at that mired, weight 1.
            assert_eq!(
                s.sensor_map,
                p.maps.class_map(class, 1e6 / cct_k).unwrap(),
                "{} {preset:?}",
                p.id
            );
            assert_eq!(s.user_cct_k, cct_k);
            assert_eq!(chain.readout().kelvin_k, cct_k);
            assert!(chain.readout().tint.abs() < 1e-12);
            assert_eq!(chain.scene().classes, ClassProbabilities::only(class));
            assert_eq!(
                s.w_a_scene,
                p.white.locus_white(cct_k),
                "{} {preset:?}",
                p.id
            );
            // The preset's camera white is what lands on neutral, whatever the estimate said.
            let white = balanced_white(chain.scene(), w_init);
            assert!(max_abs_diff(chain.apply(white).0, [1.0; 3]) < 1e-12);
            assert!(
                max_abs_diff(chain.apply([1.0; 3]).0, [1.0; 3]) > 1e-3,
                "the estimate's white is no longer neutral"
            );
        }
        for (preset, class) in [
            (LightSource::Led, LightClass::Led),
            (LightSource::Fluorescent, LightClass::Fluorescent),
        ] {
            let settings = ChainSettings {
                light_source: preset,
                ..ChainSettings::default()
            };
            let chain = FusedChain::build(p.tables(), &estimate, w_init, &settings).unwrap();
            let s = chain.stages();
            // The estimated chromaticity and CCT with all the probability on the class.
            assert_eq!(chain.scene().w_scene, estimate.w_scene);
            assert_eq!(chain.scene().cct_k, estimate.cct_k);
            assert_eq!(s.user_cct_k, estimate.cct_k);
            assert_eq!(
                s.sensor_map,
                p.maps.class_map(class, 1e6 / estimate.cct_k).unwrap()
            );
            assert_eq!(
                s.w_a_scene,
                p.white
                    .scene_white(estimate.cct_k, &ClassProbabilities::only(class))
            );
            let out = chain.apply(balanced_white(chain.scene(), w_init)).0;
            assert!(
                max_abs_diff(out, [1.0; 3]) < 1e-12,
                "{} {preset:?}: {out:?}",
                p.id
            );
        }
    }
}

#[test]
fn a_preset_and_a_manual_white_balance_are_independent() {
    // The preset sets the scene light (sensor map, C3); Manual sets the user's side (C4, C5).
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        4600.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w_init = estimate.w_scene;
    let day = FusedChain::build(
        p.tables(),
        &estimate,
        w_init,
        &ChainSettings {
            light_source: LightSource::Day,
            white_balance: WhiteBalance::Manual {
                kelvin: 3200.0,
                tint: 0.0,
            },
            ..ChainSettings::default()
        },
    )
    .unwrap();
    let auto_light =
        FusedChain::build(p.tables(), &estimate, w_init, &manual(3200.0, 0.0, 0.0)).unwrap();
    assert_eq!(
        day.stages().sensor_map,
        p.maps
            .class_map(LightClass::Daylight, 1e6 / 5500.0)
            .unwrap()
    );
    assert_eq!(day.stages().user_cct_k, 3200.0);
    assert_eq!(day.stages().g_user, auto_light.stages().g_user);
    assert_ne!(day.stages().g_res, auto_light.stages().g_res);
    assert_eq!(day.readout().kelvin_k, 3200.0);
}

#[test]
fn pick_white_makes_the_picked_sample_neutral() {
    for p in all_profiles() {
        let estimate = estimate_at(
            &p,
            5600.0,
            0.0,
            ClassProbabilities::new(0.7, 0.0, 0.1, 0.2).unwrap(),
        );
        let w_init = estimate.w_scene;
        let g0 = wb::initial_gains(w_init).unwrap();
        // A grey card under a 3100 K light with a magenta cast, 35 % of the way to clip.
        let light = p.locus.cct_duv_to_rgb(3100.0, 0.006);
        let sample = [
            0.35 * light[0] * g0[0],
            0.35 * light[1] * g0[1],
            0.35 * light[2] * g0[2],
        ];
        let settings = ChainSettings {
            picked_white: Some(sample),
            exposure_ev: 0.5,
            ..ChainSettings::default()
        };
        let chain = FusedChain::build(p.tables(), &estimate, w_init, &settings).unwrap();
        let out = chain.apply(sample).0;
        let neutral = 0.35 * 0.5_f64.exp2();
        assert!(
            max_abs_diff(out, [neutral; 3]) < 1e-12,
            "{}: {out:?} vs {neutral}",
            p.id
        );
        // The classes stay as estimated and the CCT is the picked light's.
        assert_eq!(chain.scene().classes, estimate.classes);
        assert!(
            (chain.scene().cct_k - 3100.0).abs() < 5.0,
            "{}",
            chain.scene().cct_k
        );
        // The residual gains are the ones of docs/02: w_scene_initial / w_scene.
        let g_res = chain.stages().g_res;
        let w = chain.scene().w_scene;
        for k in 0..3 {
            assert!((g_res[k] - w_init[k] / w[k]).abs() < 1e-14);
        }
        assert!((chain.readout().kelvin_k - 3100.0).abs() <= 50.0);
        assert!(
            chain.readout().tint < 0.0,
            "a magenta cast reads a negative tint"
        );
    }
}

#[test]
fn a_picked_white_that_is_blown_out_is_refused_with_the_documented_message() {
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        5600.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w_init = estimate.w_scene;
    let g0 = wb::initial_gains(w_init).unwrap();
    let blown = [0.95 * g0[0], 0.4 * g0[1], 0.4 * g0[2]];
    let err = FusedChain::build(
        p.tables(),
        &estimate,
        w_init,
        &ChainSettings {
            picked_white: Some(blown),
            ..ChainSettings::default()
        },
    )
    .unwrap_err();
    assert_eq!(err, ChainError::Pick(PickWhiteError::TooBright));
    assert_eq!(
        err.to_string(),
        "That spot is too bright to use. Pick a grey or white area that isn't blown out."
    );
}

#[test]
fn a_pick_is_replaced_by_day_shade_and_tungsten_but_kept_by_led_and_fluorescent() {
    // docs/02 C1, order of the scene light: estimate, then Pick white, then the preset. The settings do not
    // record which came last, so the chain applies the preset on top of the pick: a model light (Day, Shade,
    // Tungsten) wins, LED and Fluorescent keep the picked chromaticity and only change the class.
    for p in all_profiles() {
        let estimate = estimate_at(
            &p,
            5600.0,
            0.0,
            ClassProbabilities::new(0.7, 0.0, 0.1, 0.2).unwrap(),
        );
        let w_init = estimate.w_scene;
        let g0 = wb::initial_gains(w_init).unwrap();
        let light = p.locus.cct_duv_to_rgb(3100.0, 0.004);
        let sample = [
            0.35 * light[0] * g0[0],
            0.35 * light[1] * g0[1],
            0.35 * light[2] * g0[2],
        ];
        let build = |light_source: LightSource, white_balance: WhiteBalance| {
            FusedChain::build(
                p.tables(),
                &estimate,
                w_init,
                &ChainSettings {
                    light_source,
                    white_balance,
                    exposure_ev: 0.0,
                    picked_white: Some(sample),
                    white_scale: [1.0; 3],
                    white_shift_mired: 0.0,
                    tint_shift: 0.0,
                },
            )
            .unwrap()
        };
        let picked = build(LightSource::Auto, WhiteBalance::Auto);
        assert!((picked.scene().cct_k - 3100.0).abs() < 5.0);

        for (preset, class) in [
            (LightSource::Led, LightClass::Led),
            (LightSource::Fluorescent, LightClass::Fluorescent),
        ] {
            let chain = build(preset, WhiteBalance::Auto);
            // The picked chromaticity, CCT and Duv; all the probability on the class.
            assert_eq!(chain.scene().w_scene, picked.scene().w_scene);
            assert_eq!(chain.scene().cct_k, picked.scene().cct_k);
            assert_eq!(chain.scene().duv, picked.scene().duv);
            assert_eq!(chain.scene().classes, ClassProbabilities::only(class));
            assert_eq!(chain.stages().g_res, picked.stages().g_res);
            // Auto white balance: the picked sample is still neutral.
            let out = chain.apply(sample).0;
            let mean = (out[0] + out[1] + out[2]) / 3.0;
            assert!(
                max_abs_diff(out, [mean; 3]) < 1e-12,
                "{} {preset:?}: {out:?}",
                p.id
            );
        }
        for (preset, cct_k, class) in [
            (LightSource::Day, DAY_PRESET_K, LightClass::Daylight),
            (LightSource::Shade, SHADE_PRESET_K, LightClass::Daylight),
            (
                LightSource::Tungsten,
                TUNGSTEN_PRESET_K,
                LightClass::Tungsten,
            ),
        ] {
            let chain = build(preset, WhiteBalance::Auto);
            // The model light, not the pick.
            assert_eq!(chain.scene().w_scene, p.locus.rgb_at(cct_k));
            assert_eq!(chain.scene().cct_k, cct_k);
            assert_eq!(chain.scene().classes, ClassProbabilities::only(class));
            let same = FusedChain::build(
                p.tables(),
                &estimate,
                w_init,
                &ChainSettings {
                    light_source: preset,
                    ..ChainSettings::default()
                },
            )
            .unwrap();
            assert_eq!(chain.matrix(), same.matrix(), "{} {preset:?}", p.id);
        }

        // A pick under a Manual white balance: the scene light is the picked one, the user's side is the
        // user's own.
        let manual_chain = build(
            LightSource::Auto,
            WhiteBalance::Manual {
                kelvin: 6500.0,
                tint: 4.0,
            },
        );
        assert_eq!(manual_chain.scene().w_scene, picked.scene().w_scene);
        assert_eq!(manual_chain.stages().sensor_map, picked.stages().sensor_map);
        assert_eq!(manual_chain.stages().user_cct_k, 6500.0);
        assert_eq!(manual_chain.readout().kelvin_k, 6500.0);
        assert_eq!(manual_chain.readout().tint, 4.0);
    }
}

#[test]
fn an_exposure_whose_gain_is_not_a_number_is_refused() {
    // 2^EV overflows above about +1024 and underflows to zero below about -1075: finite stops, useless gain.
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        5200.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    for ev in [1024.0, 1e4, -1100.0, -1e9, f64::MAX, f64::MIN] {
        let err = FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(ev))
            .unwrap_err();
        assert_eq!(err, ChainError::InvalidExposure(ev), "{ev}");
        assert!(err.to_string().contains("exposure"));
    }
    // The range of the controls and far beyond it builds a finite matrix.
    for ev in [-4.0, 4.0, -60.0, 60.0, 1000.0] {
        let chain =
            FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(ev)).unwrap();
        assert!(
            chain.matrix().iter().flatten().all(|v| v.is_finite()),
            "{ev}"
        );
    }
}

#[test]
fn exposure_scales_the_matrix_by_two_to_the_ev() {
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        6500.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let base =
        FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(0.0)).unwrap();
    for ev in [-4.0, -1.0, 0.01, 2.5, 4.0] {
        let chain =
            FusedChain::build(p.tables(), &estimate, estimate.w_scene, &auto_settings(ev)).unwrap();
        let gain = ev.exp2();
        assert_eq!(chain.stages().exposure_gain, gain);
        assert_eq!(chain.exposure_ev(), ev);
        for i in 0..3 {
            for j in 0..3 {
                let expect = base.matrix()[i][j] * gain;
                assert!((chain.matrix()[i][j] - expect).abs() <= 1e-15 * expect.abs().max(1.0));
            }
        }
    }
}

#[test]
fn the_clip_level_is_the_first_neutral_clip_mapped_through_the_matrix() {
    for p in all_profiles() {
        for (cct_k, duv) in [
            (2200.0, 0.0),
            (3200.0, 0.004),
            (5600.0, -0.006),
            (9000.0, 0.0),
        ] {
            let estimate = estimate_at(&p, cct_k, duv, ClassProbabilities::only(class_for(cct_k)));
            for settings in [
                auto_settings(0.0),
                auto_settings(-1.5),
                manual(3200.0, 8.0, 0.7),
            ] {
                let chain =
                    FusedChain::build(p.tables(), &estimate, estimate.w_scene, &settings).unwrap();
                let clip = chain.clip();
                let w = chain.scene().w_scene;
                // The first channel of a neutral surface to clip, in the balanced domain.
                let balanced = (1.0 / w[0]).min(1.0 / w[1]).min(1.0 / w[2]);
                assert_eq!(clip.balanced, balanced);
                let awg4 = linalg::mul_vec(chain.matrix(), [balanced; 3]);
                assert_eq!(clip.awg4, awg4);
                assert_eq!(clip.n, awg4[0].max(awg4[1]).max(awg4[2]));
                if settings.white_balance == WhiteBalance::Auto {
                    // Auto: the clip point is neutral, n = 2^EV * balanced.
                    let expect = settings.exposure_ev.exp2() * balanced;
                    assert!(
                        (clip.n - expect).abs() < 1e-12 * expect,
                        "{} {cct_k} K",
                        p.id
                    );
                }
                assert_eq!(*clip, clip_level(chain.matrix(), w));
            }
        }
    }
    // Daylight on a Sony sensor clips green first. A light so red that red clips first has to be warmer than
    // about 1800 K (the A7R IV raw white is `[0.88, 1, 0.24]` at 2000 K, `[1.19, 1, 0.17]` at 1600 K; the
    // locus is in the raw scale, docs/02 C1; the red channel of a Sony body is the least sensitive).
    let p = load("sony-ilce-7rm4");
    let day = estimate_at(
        &p,
        6500.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    assert_eq!(
        FusedChain::build(p.tables(), &day, day.w_scene, &auto_settings(0.0))
            .unwrap()
            .clip()
            .balanced,
        1.0
    );
    let tungsten = estimate_at(
        &p,
        1600.0,
        0.0,
        ClassProbabilities::only(LightClass::Tungsten),
    );
    let red = tungsten.w_scene[0];
    assert!(red > 1.0, "{red}");
    let chain =
        FusedChain::build(p.tables(), &tungsten, tungsten.w_scene, &auto_settings(0.0)).unwrap();
    assert!((chain.clip().balanced - 1.0 / red).abs() < 1e-15);
}

#[test]
fn the_initial_white_may_be_given_at_any_scale() {
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        4900.0,
        0.002,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w_init = p.locus.rgb_at(7000.0);
    let a = FusedChain::build(p.tables(), &estimate, w_init, &auto_settings(0.0)).unwrap();
    let b = FusedChain::build(
        p.tables(),
        &estimate,
        w_init.map(|c| c * 7.5),
        &auto_settings(0.0),
    )
    .unwrap();
    assert!(linalg::max_abs_diff(a.matrix(), b.matrix()) < 1e-15);
}

#[test]
fn bad_input_is_refused() {
    let p = load("sony-ilce-7rm4");
    let estimate = estimate_at(
        &p,
        4900.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w = estimate.w_scene;
    let build =
        |w_init: [f64; 3], s: ChainSettings| FusedChain::build(p.tables(), &estimate, w_init, &s);
    assert!(matches!(
        build(w, auto_settings(f64::NAN)),
        Err(ChainError::InvalidExposure(_))
    ));
    assert!(matches!(
        build(w, auto_settings(f64::INFINITY)),
        Err(ChainError::InvalidExposure(_))
    ));
    assert!(matches!(
        build([0.5, 0.0, 0.5], auto_settings(0.0)),
        Err(ChainError::Wb(WbError::InvalidWhite(_)))
    ));
    assert!(matches!(
        build(w, manual(1900.0, 0.0, 0.0)),
        Err(ChainError::Wb(WbError::KelvinOutOfRange(_)))
    ));
    assert!(matches!(
        build(w, manual(5000.0, 51.0, 0.0)),
        Err(ChainError::Wb(WbError::TintOutOfRange(_)))
    ));
    // Every error has a sentence.
    for e in [
        build(w, auto_settings(f64::NAN)).unwrap_err(),
        build(w, manual(1900.0, 0.0, 0.0)).unwrap_err(),
    ] {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn the_four_thousand_kelvin_step_of_w_a_shows_in_manual_white_balance() {
    // docs/03: the user white is a blackbody below 4000 K and CIE daylight from there, so w_A steps by
    // about 1.2 % (red) and 3.4 % (blue) at 4000 K and a Kelvin slider dragged across it shows a small
    // jump (about 0.05 stop in blue). Pin that the step is as small as documented.
    let p = load("sony-ilce-7rm4");
    let estimate = exact_estimate(
        &p,
        5000.0,
        0.0,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let w_init = estimate.w_scene;
    let below =
        FusedChain::build(p.tables(), &estimate, w_init, &manual(3999.0, 0.0, 0.0)).unwrap();
    let above =
        FusedChain::build(p.tables(), &estimate, w_init, &manual(4000.0, 0.0, 0.0)).unwrap();
    let white_step: Vec<f64> = (0..3)
        .map(|c| above.apply([1.0; 3]).0[c] / below.apply([1.0; 3]).0[c] - 1.0)
        .collect();
    eprintln!("3999 K -> 4000 K: scene white steps by {white_step:?}");
    assert!(
        white_step.iter().any(|s| s.abs() > 0.005),
        "the step is the documented one"
    );
    assert!(white_step.iter().all(|s| s.abs() < 0.06), "{white_step:?}");
    // Red up, green down, blue up by the amounts docs/02 gives (1.1 %, -0.4 %, 4.4 % for the A7R IV): the
    // user's white at 4000 K is the daylight table, a little less red and much less blue than the 3999 K
    // blackbody, so dividing by it lifts both.
    assert!(white_step[0] > 0.005 && white_step[1] < 0.0 && white_step[2] > 0.03);
}

#[test]
fn color_estimates_resolve_exposure_and_build_the_metering_chain() {
    let p = load("sony-ilce-7rm4");
    let scene = estimate_at(
        &p,
        5200.0,
        0.002,
        ClassProbabilities::only(LightClass::Daylight),
    );
    let estimates = ColorEstimates {
        scene,
        w_scene_initial: scene.w_scene,
        white_scale: [1.0; 3],
        ev_auto: 0.75,
        white_shift_mired: 0.0,
        tint_shift: 0.0,
    };
    let auto = estimates
        .chain(
            p.tables(),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Auto,
            None,
        )
        .unwrap();
    assert_eq!(auto.exposure_ev(), 0.75);
    let manual_ev = estimates
        .chain(
            p.tables(),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Manual { ev: -1.25 },
            None,
        )
        .unwrap();
    assert_eq!(manual_ev.exposure_ev(), -1.25);
    // A manual exposure outside the control's range is refused (the automatic value is not checked: the
    // guard can take it to -4).
    let err = estimates
        .chain(
            p.tables(),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Manual { ev: 4.5 },
            None,
        )
        .unwrap_err();
    assert!(matches!(err, ChainError::Exposure(_)), "{err:?}");
    let low = ColorEstimates {
        ev_auto: -4.0,
        ..estimates
    };
    assert!(
        low.chain(
            p.tables(),
            LightSource::Auto,
            WhiteBalance::Auto,
            ExposureSetting::Auto,
            None
        )
        .is_ok()
    );
    let metering = estimates.metering_chain(p.tables()).unwrap();
    assert_eq!(metering.exposure_ev(), 0.0);
    assert_eq!(metering.stages().c3_c4, [1.0; 3]);
    // Same colour, different exposure.
    let scale = 0.75_f64.exp2();
    for i in 0..3 {
        for j in 0..3 {
            let e = metering.matrix()[i][j] * scale;
            assert!((auto.matrix()[i][j] - e).abs() < 1e-14);
        }
    }
    let mat: Mat3 = *metering.matrix();
    assert!(linalg::det(&mat).abs() > 1e-3, "the chain is invertible");
}
