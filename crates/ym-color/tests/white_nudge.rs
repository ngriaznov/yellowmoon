//! Apple Auto's nudge of the Auto white (`docs/19` section 5, `docs/02` C3–C4): the user's white moves
//! along the locus by `m` mired, keeping the scene's classes and tint, and the ARRI matrix and the
//! readout follow it. With `m = 0` the chain is today's Auto chain, bit for bit. A light preset, a
//! picked white and a manual white balance get no nudge.
//!
//! Uses the shipped `data/cameras/profiles/*/profile.json` tables, so it always runs.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::chain::{ChainSettings, ColorEstimates, FusedChain, ProfileTables};
use ym_color::consts::linalg;
use ym_color::exposure::ExposureSetting;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus};
use ym_color::wb::{self, LightSource, SceneLight, WhiteBalance};

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

fn all_profiles() -> Vec<Profile> {
    let mut ids: Vec<String> = std::fs::read_dir(repo_root().join("data/cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    let profiles: Vec<Profile> = ids
        .iter()
        .map(|id| {
            let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
            let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            Profile {
                id: id.clone(),
                locus: serde_json::from_value(v["locus"].clone()).unwrap(),
                white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
                maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
            }
        })
        .collect();
    assert_eq!(profiles.len(), 11);
    profiles
}

/// An estimate `duv` off the locus at `cct_k`, read back through the profile's locus.
fn estimate_at(p: &Profile, cct_k: f64, duv: f64, classes: ClassProbabilities) -> SceneLight {
    SceneLight::new(p.locus.cct_duv_to_rgb(cct_k, duv), classes, 0.8, &p.locus).unwrap()
}

/// An estimate with exactly known CCT and Duv.
fn exact_estimate(p: &Profile, cct_k: f64, duv: f64, classes: ClassProbabilities) -> SceneLight {
    SceneLight::from_parts(p.locus.cct_duv_to_rgb(cct_k, duv), classes, 0.8, cct_k, duv).unwrap()
}

/// A range of scenes: daylight and tungsten on and off the locus, LED, fluorescent and mixed light.
fn scenes(p: &Profile) -> Vec<SceneLight> {
    let day = ClassProbabilities::only(LightClass::Daylight);
    let tungsten = ClassProbabilities::only(LightClass::Tungsten);
    vec![
        estimate_at(p, 2900.0, 0.0, tungsten),
        estimate_at(p, 3400.0, 0.002, tungsten),
        estimate_at(p, 5200.0, 0.0, day),
        estimate_at(p, 6500.0, -0.003, day),
        estimate_at(p, 4300.0, 0.004, ClassProbabilities::only(LightClass::Led)),
        estimate_at(
            p,
            3900.0,
            0.003,
            ClassProbabilities::only(LightClass::Fluorescent),
        ),
        estimate_at(
            p,
            4700.0,
            -0.001,
            ClassProbabilities::new(0.5, 0.1, 0.1, 0.3).unwrap(),
        ),
    ]
}

fn auto(ev: f64, shift: f64) -> ChainSettings {
    ChainSettings {
        exposure_ev: ev,
        white_shift_mired: shift,
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

/// Red over blue of the scene white in AWG4: above 1 is warmer than neutral.
fn warmth(p: &Profile, scene: &SceneLight, w_init: [f64; 3], shift: f64) -> f64 {
    let chain = FusedChain::build(p.tables(), scene, w_init, &auto(0.0, shift)).unwrap();
    let v = chain.apply(balanced_white(scene, w_init)).0;
    v[0] / v[2]
}

#[test]
fn no_nudge_is_the_auto_chain_bit_for_bit() {
    // A shift of 0 (or -0) takes the Auto path untouched, on every profile and kind of light; the
    // estimates' own field defaults to it, and the metering chain is the same chain.
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = p.locus.rgb_at(5600.0);
            for ev in [-1.25, 0.0, 0.6] {
                let plain = FusedChain::build(
                    p.tables(),
                    &scene,
                    w_init,
                    &ChainSettings {
                        exposure_ev: ev,
                        ..ChainSettings::default()
                    },
                )
                .unwrap();
                for zero in [0.0, -0.0] {
                    let explicit =
                        FusedChain::build(p.tables(), &scene, w_init, &auto(ev, zero)).unwrap();
                    assert_eq!(plain, explicit, "{} {:?}", p.id, scene.cct_k);
                }
                assert_eq!(plain.stages().c3_c4, [1.0; 3], "{}", p.id);
                let estimates = ColorEstimates {
                    scene,
                    w_scene_initial: w_init,
                    white_scale: wb::IDENTITY_WHITE_SCALE,
                    ev_auto: ev,
                    white_shift_mired: 0.0,
                    tint_shift: 0.0,
                };
                let via = estimates
                    .chain(
                        p.tables(),
                        LightSource::Auto,
                        WhiteBalance::Auto,
                        ExposureSetting::Auto,
                        None,
                    )
                    .unwrap();
                assert_eq!(plain, via, "{}", p.id);
                assert_eq!(
                    estimates.metering_chain(p.tables()).unwrap(),
                    FusedChain::build(p.tables(), &scene, w_init, &auto(0.0, 0.0)).unwrap(),
                    "{}",
                    p.id
                );
            }
        }
    }
}

#[test]
fn a_warmer_nudge_warms_the_scene_white_and_a_cooler_one_cools_it_monotonically() {
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = scene.w_scene;
            let rb: Vec<f64> = [-20.0, -10.0, 0.0, 10.0, 20.0]
                .iter()
                .map(|&m| warmth(&p, &scene, w_init, m))
                .collect();
            // Without a nudge the scene white is neutral.
            assert!((rb[2] - 1.0).abs() < 1e-9, "{} {:?}", p.id, rb);
            // Warmer (a lower mired, a higher Kelvin) raises R/B, cooler lowers it, and more is more.
            for pair in rb.windows(2) {
                assert!(pair[0] > pair[1], "{} at {} K: {rb:?}", p.id, scene.cct_k);
            }
            // A step is a visible but small change: 10 mired moves R/B by a few percent.
            assert!(
                rb[1] > 1.005 && rb[1] < 1.2,
                "{} at {} K: {rb:?}",
                p.id,
                scene.cct_k
            );
        }
    }
}

#[test]
fn the_kelvin_readout_and_the_arri_matrix_follow_the_nudge() {
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = scene.w_scene;
            let build =
                |m: f64| FusedChain::build(p.tables(), &scene, w_init, &auto(0.0, m)).unwrap();
            let (cool2, cool, plain, warm, warm2) = (
                build(20.0),
                build(10.0),
                build(0.0),
                build(-10.0),
                build(-20.0),
            );
            let k = |c: &FusedChain| c.readout().kelvin_k;
            assert!(
                k(&cool2) < k(&cool)
                    && k(&cool) < k(&plain)
                    && k(&plain) < k(&warm)
                    && k(&warm) < k(&warm2),
                "{}: {} {} {} {} {}",
                p.id,
                k(&cool2),
                k(&cool),
                k(&plain),
                k(&warm),
                k(&warm2)
            );
            for c in [&cool2, &cool, &warm, &warm2] {
                // A multiple of 50 inside the control's range.
                assert_eq!(c.readout().kelvin_k % 50.0, 0.0);
                assert!((2000.0..=11000.0).contains(&c.readout().kelvin_k));
            }
            // The ARRI matrix takes the nudged CCT.
            let cct_warm = wb::nudged_cct(scene.cct_k, -10.0);
            assert_eq!(warm.stages().user_cct_k, cct_warm.clamp(2000.0, 11000.0));
            assert!((1e6 / cct_warm - (1e6 / scene.cct_k - 10.0)).abs() < 1e-9);
            // The tint stays near the scene's (Auto's readout is the reference).
            assert!(
                (warm.readout().tint - plain.readout().tint).abs() < 3.0,
                "{}: {} vs {}",
                p.id,
                warm.readout().tint,
                plain.readout().tint
            );
        }
    }
}

#[test]
fn a_nudged_daylight_white_renders_like_a_manual_white_balance_at_the_nudged_kelvin() {
    // On the locus, in daylight, the nudged user white is the daylight white at the nudged CCT: the
    // chain is Manual at that Kelvin and no tint, to rounding.
    for p in all_profiles() {
        let scene = exact_estimate(
            &p,
            5000.0,
            0.0,
            ClassProbabilities::only(LightClass::Daylight),
        );
        let w_init = scene.w_scene;
        for m in [-20.0, -10.0, 10.0, 20.0] {
            let nudged = FusedChain::build(p.tables(), &scene, w_init, &auto(0.3, m)).unwrap();
            let kelvin = wb::nudged_cct(5000.0, m);
            let manual = FusedChain::build(
                p.tables(),
                &scene,
                w_init,
                &ChainSettings {
                    white_balance: WhiteBalance::Manual { kelvin, tint: 0.0 },
                    exposure_ev: 0.3,
                    ..ChainSettings::default()
                },
            )
            .unwrap();
            let (a, b) = (nudged.matrix(), manual.matrix());
            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        (a[r][c] - b[r][c]).abs() < 1e-9,
                        "{} m={m}: {a:?} vs {b:?}",
                        p.id
                    );
                }
            }
            assert_eq!(
                nudged.readout().kelvin_k,
                (kelvin / 50.0).round() * 50.0,
                "{}",
                p.id
            );
            assert!(nudged.readout().tint.abs() < 0.5, "{}", p.id);
        }
    }
}

#[test]
fn a_preset_a_picked_white_and_a_manual_white_balance_get_no_nudge() {
    for p in all_profiles() {
        let scene = estimate_at(
            &p,
            4300.0,
            0.002,
            ClassProbabilities::new(0.6, 0.2, 0.1, 0.1).unwrap(),
        );
        let w_init = p.locus.rgb_at(5600.0);
        let g0 = wb::initial_gains(w_init).unwrap();
        let sample = [0.3 * g0[0] * 0.9, 0.3, 0.3 * g0[2] * 1.1];
        let cases: Vec<ChainSettings> = [
            LightSource::Day,
            LightSource::Shade,
            LightSource::Tungsten,
            LightSource::Led,
            LightSource::Fluorescent,
        ]
        .into_iter()
        .map(|light_source| ChainSettings {
            light_source,
            ..ChainSettings::default()
        })
        .chain([
            ChainSettings {
                picked_white: Some(sample),
                ..ChainSettings::default()
            },
            ChainSettings {
                white_balance: WhiteBalance::Manual {
                    kelvin: 3600.0,
                    tint: 7.0,
                },
                ..ChainSettings::default()
            },
            ChainSettings {
                light_source: LightSource::Tungsten,
                white_balance: WhiteBalance::Manual {
                    kelvin: 6100.0,
                    tint: -3.0,
                },
                ..ChainSettings::default()
            },
        ])
        .collect();
        for settings in cases {
            let plain = FusedChain::build(p.tables(), &scene, w_init, &settings).unwrap();
            for m in [-20.0, 10.0] {
                let shifted = FusedChain::build(
                    p.tables(),
                    &scene,
                    w_init,
                    &ChainSettings {
                        white_shift_mired: m,
                        ..settings
                    },
                )
                .unwrap();
                assert_eq!(plain, shifted, "{} {settings:?}", p.id);
            }
        }
    }
}

#[test]
fn the_nudge_keeps_the_scene_white_on_its_green_and_the_matrix_balanced_in_the_rest() {
    // The nudge changes only the red and blue gains of C4: a neutral that is not the scene white keeps
    // its luminance close, and the product of C3 and C4 has green exactly 1.
    for p in all_profiles() {
        for scene in scenes(&p) {
            let chain =
                FusedChain::build(p.tables(), &scene, scene.w_scene, &auto(0.0, -20.0)).unwrap();
            let s = chain.stages();
            assert_eq!(s.c3_c4[1], 1.0, "{}", p.id);
            assert_ne!(s.c3_c4, [1.0; 3], "{}", p.id);
            let m = linalg::mul_vec(chain.matrix(), [1.0; 3]);
            assert!(m.iter().all(|v| v.is_finite() && *v > 0.0), "{}", p.id);
        }
    }
}

#[test]
fn the_nudged_cct_stays_inside_the_white_tables() {
    assert!((wb::nudged_cct(5000.0, -10.0) - 1e6 / 190.0).abs() < 1e-9);
    assert!((wb::nudged_cct(5000.0, 10.0) - 1e6 / 210.0).abs() < 1e-9);
    assert_eq!(wb::nudged_cct(5000.0, 0.0), 5000.0);
    // A light far beyond the tables is held at their ends.
    assert_eq!(wb::nudged_cct(40_000.0, -30.0), wb::NUDGE_MAX_K);
    assert_eq!(wb::nudged_cct(1200.0, 20.0), wb::NUDGE_MIN_K);
}

#[test]
fn every_kind_of_light_gets_the_nudge_in_the_right_direction_across_the_range() {
    // The fluorescent and LED class whites are lists of real lights, so the nudge moves the scene's
    // own white as the locus moves (`wb::nudged_alexa_white`): every class, on and off the locus and
    // across the 4000 K step of the white model, is warmer for a negative shift and cooler for a
    // positive one, more so for a larger one.
    for p in all_profiles() {
        for class in LightClass::ALL {
            for cct in [
                2200.0, 2700.0, 3000.0, 3500.0, 3900.0, 4000.0, 4100.0, 4300.0, 5000.0, 5600.0,
                6500.0, 8000.0, 10000.0,
            ] {
                for duv in [-0.004, 0.0, 0.004] {
                    let scene = estimate_at(&p, cct, duv, ClassProbabilities::only(class));
                    let rb: Vec<f64> = [-20.0, -10.0, 0.0, 10.0, 20.0]
                        .iter()
                        .map(|&m| warmth(&p, &scene, scene.w_scene, m))
                        .collect();
                    assert!(
                        rb.windows(2).all(|w| w[0] > w[1]) && rb[1] > 1.0 && rb[3] < 1.0,
                        "{} {class:?} {cct} K {duv}: {rb:?}",
                        p.id
                    );
                }
            }
        }
    }
}

fn tinted(ev: f64, shift: f64, tint: f64) -> ChainSettings {
    ChainSettings {
        tint_shift: tint,
        ..auto(ev, shift)
    }
}

/// Green over the mean of red and blue of the scene white in AWG4: above 1 is greener than neutral.
fn greenness(p: &Profile, scene: &SceneLight, w_init: [f64; 3], shift: f64, tint: f64) -> f64 {
    let chain = FusedChain::build(p.tables(), scene, w_init, &tinted(0.0, shift, tint)).unwrap();
    let v = chain.apply(balanced_white(scene, w_init)).0;
    v[1] / (0.5 * (v[0] + v[2]))
}

#[test]
fn no_tint_nudge_is_the_chain_without_it_bit_for_bit() {
    // A tint shift of 0 (or -0) changes nothing: alone it is the Auto chain, and with a white shift
    // it is the white nudge as it was before the tint nudge existed.
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = p.locus.rgb_at(5600.0);
            for m in [0.0, -10.0, 20.0] {
                let plain = FusedChain::build(p.tables(), &scene, w_init, &auto(0.3, m)).unwrap();
                for zero in [0.0, -0.0] {
                    let explicit =
                        FusedChain::build(p.tables(), &scene, w_init, &tinted(0.3, m, zero))
                            .unwrap();
                    assert_eq!(plain, explicit, "{} {m}", p.id);
                }
            }
        }
    }
}

#[test]
fn a_magenta_tint_nudge_takes_green_out_of_the_scene_white_and_a_green_one_adds_it() {
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = scene.w_scene;
            for m in [0.0, -10.0] {
                let g: Vec<f64> = [-10.0, -5.0, 0.0, 5.0, 10.0]
                    .iter()
                    .map(|&t| greenness(&p, &scene, w_init, m, t))
                    .collect();
                // More magenta (a positive tint) is less green, monotonically.
                assert!(
                    g.windows(2).all(|w| w[0] > w[1]),
                    "{} at {} K, {m} mired: {g:?}",
                    p.id,
                    scene.cct_k
                );
                // A step is small: 5 tint units move G/((R+B)/2) by a few percent.
                assert!(g[3] / g[2] > 0.95 && g[3] / g[2] < 0.995, "{}: {g:?}", p.id);
            }
        }
    }
}

#[test]
fn the_tint_readout_follows_the_tint_nudge_and_the_kelvin_stays() {
    for p in all_profiles() {
        for scene in scenes(&p) {
            let w_init = scene.w_scene;
            let plain = FusedChain::build(p.tables(), &scene, w_init, &auto(0.0, 0.0)).unwrap();
            for t in [-10.0, -5.0, 5.0, 10.0] {
                let nudged =
                    FusedChain::build(p.tables(), &scene, w_init, &tinted(0.0, 0.0, t)).unwrap();
                assert_eq!(
                    nudged.readout().kelvin_k,
                    plain.readout().kelvin_k,
                    "{}",
                    p.id
                );
                let expected = (plain.readout().tint + t).clamp(-50.0, 50.0);
                assert!(
                    (nudged.readout().tint - expected).abs() < 1e-6,
                    "{} {t}: {} vs {expected}",
                    p.id,
                    nudged.readout().tint
                );
                // The ARRI matrix keeps the scene's CCT.
                assert_eq!(nudged.stages().user_cct_k, plain.stages().user_cct_k);
                // Only red and blue of C3·C4 move; green stays exactly 1.
                assert_eq!(nudged.stages().c3_c4[1], 1.0);
            }
        }
    }
}

#[test]
fn the_tint_nudge_holds_the_tint_in_its_range() {
    let p = &all_profiles()[0];
    // A scene well on the green side: its own tint is near the top of the range.
    let day = ClassProbabilities::only(LightClass::Daylight);
    let scene = exact_estimate(p, 5200.0, -0.03, day);
    let own = wb::scene_tint(&p.locus, &scene);
    assert!(own > 30.0, "{own}");
    assert_eq!(wb::nudged_tint(&p.locus, &scene, 0.0), 0.0);
    assert_eq!(
        wb::nudged_tint(&p.locus, &scene, 20.0),
        (50.0 - own).max(0.0)
    );
    assert_eq!(wb::nudged_tint(&p.locus, &scene, -10.0), -10.0);
    assert_eq!(wb::nudged_tint(&p.locus, &scene, f64::NAN), 0.0);
}

#[test]
fn a_preset_a_picked_white_and_a_manual_white_balance_get_no_tint_nudge() {
    for p in all_profiles() {
        let scene = estimate_at(
            &p,
            4300.0,
            0.002,
            ClassProbabilities::new(0.6, 0.2, 0.1, 0.1).unwrap(),
        );
        let w_init = p.locus.rgb_at(5600.0);
        let g0 = wb::initial_gains(w_init).unwrap();
        let sample = [0.3 * g0[0] * 0.9, 0.3, 0.3 * g0[2] * 1.1];
        for settings in [
            ChainSettings {
                light_source: LightSource::Day,
                ..ChainSettings::default()
            },
            ChainSettings {
                picked_white: Some(sample),
                ..ChainSettings::default()
            },
            ChainSettings {
                white_balance: WhiteBalance::Manual {
                    kelvin: 3600.0,
                    tint: 7.0,
                },
                ..ChainSettings::default()
            },
        ] {
            let plain = FusedChain::build(p.tables(), &scene, w_init, &settings).unwrap();
            let shifted = FusedChain::build(
                p.tables(),
                &scene,
                w_init,
                &ChainSettings {
                    tint_shift: 10.0,
                    white_shift_mired: -10.0,
                    ..settings
                },
            )
            .unwrap();
            assert_eq!(plain, shifted, "{} {settings:?}", p.id);
        }
    }
}
