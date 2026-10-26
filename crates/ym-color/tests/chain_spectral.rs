//! M1-07 done-when: "3200 K under daylight renders [differently from] Auto by the amount the ALEXA maths
//! predicts", against an independent CPU spectral simulation.
//!
//! The simulation never touches the profile tables the chain reads. It integrates spectra:
//!
//! * the scene is a light SPD (the CIE daylight and reference lights of the exporter's fixtures, or
//!   Planck's law computed here for blackbody light) times a reflectance (a flat white, or the 24
//!   ColorChecker patches);
//! * the ALEXA native response is `S_A . (E o R)` with the ALEXA estimate curves, divided by the green
//!   response to the white (so a white has native green 1);
//! * ARRI's white balance at `K` is the inverse of the native white of a blackbody (below 4000 K) or CIE
//!   daylight (from 4000 K) at `K`, then ARRI's matrix at `K` (the 17-CCT table), then AWG3 to AWG4.
//!
//! The chain gets the Sony camera RGB of the same patch under the same light (A7R IV, A7S III and RX100 IV
//! curves), balanced with the scene white, as the raw stages would deliver it. For a white patch the
//! sensor map plays no part (it preserves white), so the chain has to equal the simulation to the accuracy
//! of the `w_A` tables (1e-7 on their grid). For coloured patches the two differ by the sensor map's error
//! (the gate of M1-05), which is reported as a colour difference.
//!
//! Needs the git-ignored fixtures of `python3 -m training.sensor_map.export_fixtures` (CIE light SPDs are
//! CC BY-SA data) and skips with a message when they are missing; `YM_REQUIRE_FIXTURES=1` makes that a
//! failure.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::arri::ArriTable;
use ym_color::chain::{ChainSettings, FusedChain, ProfileTables};
use ym_color::consts::linalg;
use ym_color::deltae::{self, Summary};
use ym_color::gamut;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::fixtures::{Fixtures, Light};
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus, SpectralCurves};
use ym_color::wb::{SceneLight, WhiteBalance};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Profile {
    sony: SpectralCurves,
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

fn load(id: &str) -> Profile {
    let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let curves = repo_root()
        .join("data/cameras")
        .join(v["spectral_curves"].as_str().unwrap());
    Profile {
        // The camera in the raw scale of its data (docs/02 C1): the curves as published, noise below zero
        // clipped, not normalised per channel. The scene white read from them is compared with the profile
        // locus, which is in that scale; the g0-balanced patches below divide by the white of the same
        // curves, so they do not depend on the scale.
        sony: SpectralCurves::from_path(curves).unwrap().clipped_at_zero(),
        locus: serde_json::from_value(v["locus"].clone()).unwrap(),
        white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
        maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
    }
}

fn alexa() -> SpectralCurves {
    SpectralCurves::from_path(repo_root().join("data/alexa/alexa_alev3_estimate_380_780_5.json"))
        .unwrap()
        .peak_normalized()
        .unwrap()
}

/// Planck's law on the fixtures' 380 to 780 nm grid (5 nm), peak-normalised, with colour-science's
/// `c2 = 0.014388` (the constant the fit scripts use).
fn planck(t_k: f64) -> Vec<f64> {
    let spd: Vec<f64> = (0..81)
        .map(|i| {
            let l = (380.0 + 5.0 * f64::from(i)) * 1e-9;
            1.0 / (l.powi(5) * ((0.014388 / (l * t_k)).exp() - 1.0))
        })
        .collect();
    let peak = spd.iter().copied().fold(0.0, f64::max);
    spd.into_iter().map(|v| v / peak).collect()
}

/// `[r, 1, b]` of the response to a light.
fn white_of(curves: &SpectralCurves, spd: &[f64]) -> [f64; 3] {
    let r = curves.response(spd).unwrap();
    [r[0] / r[1], 1.0, r[2] / r[1]]
}

fn times(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// The light a user setting of `kelvin` assumes: a blackbody below 4000 K (Planck here), CIE daylight
/// from 4000 K (the exporter's reference lights, every 500 K).
fn user_light(fx: &Fixtures, kelvin: f64) -> Vec<f64> {
    if kelvin < 4000.0 {
        planck(kelvin)
    } else {
        fx.light(&format!("ref:daylight:daylight-{kelvin:.0}K"))
            .unwrap_or_else(|| panic!("no reference daylight at {kelvin} K"))
            .spd
            .clone()
    }
}

/// The ALEXA simulation of a patch of reflectance `r` under `light`, with ARRI's white balance at
/// `kelvin` (tint 0), as linear AWG4. A white has native green 1 before the white balance gains.
fn alexa_sim(
    s_a: &SpectralCurves,
    fx: &Fixtures,
    light: &Light,
    r: &[f64],
    kelvin: Option<f64>,
) -> [f64; 3] {
    let native = s_a.response(&times(&light.spd, r)).unwrap();
    let white_g = s_a.response(&light.spd).unwrap()[1];
    let native = native.map(|v| v / white_g);
    let arri = ArriTable::builtin();
    match kelvin {
        // Auto: the light's own white is balanced.
        None => {
            let u = white_of(s_a, &light.spd);
            let balanced = [native[0] / u[0], native[1] / u[1], native[2] / u[2]];
            linalg::mul_vec(&arri.native_to_awg4(light.cct_k), balanced)
        }
        Some(k) => {
            let u = white_of(s_a, &user_light(fx, k));
            let balanced = [native[0] / u[0], native[1] / u[1], native[2] / u[2]];
            linalg::mul_vec(&arri.native_to_awg4(k), balanced)
        }
    }
}

/// The g0-balanced Sony RGB of a patch: its response divided by the response to the white.
fn camera_balanced(p: &Profile, light: &Light, r: &[f64]) -> [f64; 3] {
    let x = p.sony.response(&times(&light.spd, r)).unwrap();
    let w = p.sony.response(&light.spd).unwrap();
    [x[0] / w[0], x[1] / w[1], x[2] / w[2]]
}

/// The scene light for a fixture light: its camera white, the class of the light and its CCT.
fn scene_of(p: &Profile, light: &Light, exact_cct: bool) -> SceneLight {
    let class = match light.class.as_str() {
        "daylight" => LightClass::Daylight,
        "tungsten" => LightClass::Tungsten,
        "fluorescent" => LightClass::Fluorescent,
        "led" => LightClass::Led,
        other => panic!("unexpected class {other}"),
    };
    let classes = ClassProbabilities::only(class);
    let w = white_of(&p.sony, &light.spd);
    if exact_cct {
        // The CCT the light is defined by (generating temperature, or Ohno's for F and LED lights).
        SceneLight::from_parts(w, classes, 1.0, light.cct_k, 0.0).unwrap()
    } else {
        SceneLight::new(w, classes, 1.0, &p.locus).unwrap()
    }
}

fn lab_of(awg4: [f64; 3]) -> [f64; 3] {
    let xyz = linalg::mul_vec(&gamut::matrices().awg4_to_xyz, awg4);
    deltae::xyz_to_lab(xyz, deltae::d65_white_xyz())
}

fn max_abs(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).fold(0.0, |m, k| m.max((a[k] - b[k]).abs()))
}

const PROFILES: [&str; 3] = ["sony-ilce-7rm4", "sony-ilce-7sm3", "sony-dsc-rx100m4"];

#[test]
fn the_test_blackbody_is_the_fixtures_blackbody() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    for t in [2000.0, 2500.0, 3000.0, 3500.0, 4000.0] {
        let fixture = fx
            .light(&format!("ref:tungsten:blackbody-{t:.0}K"))
            .unwrap();
        let mine = planck(t);
        let d = fixture
            .spd
            .iter()
            .zip(&mine)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(d < 1e-9, "{t} K: {d:e}");
    }
}

#[test]
fn white_light_under_a_manual_white_balance_is_what_the_alexa_maths_predicts() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    let white = vec![1.0; 81];
    let mut worst = 0.0_f64;
    let mut cases = 0;
    for id in PROFILES {
        let p = load(id);
        let scenes = [
            "ref:daylight:daylight-5500K",
            "ref:daylight:daylight-6500K",
            "ref:daylight:daylight-7500K",
            "gate:daylight-5250K",
            "gate:daylight-6750K",
            "ref:daylight:daylight-4500K",
            "ref:tungsten:blackbody-3000K",
            "ref:tungsten:blackbody-2500K",
            "gate:blackbody-2850K",
            "ref:led:LED-B3",
            "ref:led:LED-RGB1",
            "ref:fluorescent:FL2",
            "ref:fluorescent:FL7",
        ];
        for name in scenes {
            let light = fx
                .light(name)
                .unwrap_or_else(|| panic!("no fixture {name}"));
            let scene = scene_of(&p, light, true);
            let w_init = scene.w_scene;
            let r_cam = camera_balanced(&p, light, &white);
            for k in [
                None,
                Some(2500.0),
                Some(3000.0),
                Some(3500.0),
                Some(4500.0),
                Some(5500.0),
                Some(6500.0),
                Some(9000.0),
            ] {
                let settings = ChainSettings {
                    white_balance: match k {
                        None => WhiteBalance::Auto,
                        Some(kelvin) => WhiteBalance::Manual { kelvin, tint: 0.0 },
                    },
                    ..ChainSettings::default()
                };
                let chain = FusedChain::build(p.tables(), &scene, w_init, &settings).unwrap();
                let got = chain.apply(r_cam).0;
                let want = alexa_sim(&s_a, &fx, light, &white, k);
                let d = max_abs(got, want);
                worst = worst.max(d);
                cases += 1;
                assert!(
                    d < 1e-6,
                    "{id} {name} balanced for {k:?}: chain {got:?} simulation {want:?}"
                );
            }
        }
    }
    eprintln!(
        "white patch: chain vs ALEXA simulation, worst abs difference {worst:.2e} over {cases} cases"
    );
}

#[test]
fn three_thousand_two_hundred_kelvin_under_daylight_differs_from_auto_by_the_predicted_amount() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    let white = vec![1.0; 81];
    let daylight = fx.light("ref:daylight:daylight-5500K").unwrap();
    let mut reported = false;
    for id in PROFILES {
        let p = load(id);
        let scene = scene_of(&p, daylight, true);
        let r_cam = camera_balanced(&p, daylight, &white);
        let render = |wb: WhiteBalance| {
            let settings = ChainSettings {
                white_balance: wb,
                ..ChainSettings::default()
            };
            FusedChain::build(p.tables(), &scene, scene.w_scene, &settings)
                .unwrap()
                .apply(r_cam)
                .0
        };
        let auto = render(WhiteBalance::Auto);
        let tungsten = render(WhiteBalance::Manual {
            kelvin: 3200.0,
            tint: 0.0,
        });
        let sim_auto = alexa_sim(&s_a, &fx, daylight, &white, None);
        let sim_tungsten = alexa_sim(&s_a, &fx, daylight, &white, Some(3200.0));
        // Auto: a neutral.
        assert!(max_abs(auto, [1.0; 3]) < 1e-12, "{id}: {auto:?}");
        assert!(max_abs(sim_auto, [1.0; 3]) < 1e-9, "{id}: {sim_auto:?}");

        // The amount, in stops: log2 of blue over red and of green over red, Auto against 3200 K.
        let stops = |rgb: [f64; 3]| [(rgb[2] / rgb[0]).log2(), (rgb[1] / rgb[0]).log2()];
        let (got, want) = (stops(tungsten), stops(sim_tungsten));
        assert!(
            (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6,
            "{id}: chain {got:?} vs simulation {want:?}"
        );
        assert!(
            max_abs(tungsten, sim_tungsten) < 1e-6,
            "{id}: {tungsten:?} vs {sim_tungsten:?}"
        );
        // Balancing an ALEXA for 3200 K tungsten and shooting daylight is blue: red is the weakest
        // channel and blue the strongest, by about 1.3 stops (the sign the maths predicts).
        assert!(
            tungsten[2] > tungsten[1] && tungsten[1] > tungsten[0],
            "{id}: {tungsten:?}"
        );
        assert!(
            got[0] > 1.0 && got[0] < 2.5,
            "{id}: blue over red {:.3} stops",
            got[0]
        );
        if !reported {
            eprintln!(
                "daylight 5500 K at WB 3200 K: AWG4 chain {tungsten:?}, simulation {sim_tungsten:?}; blue over red {:.4} stops, green over red {:.4} stops (Auto: 0)",
                got[0], got[1]
            );
            reported = true;
        }
    }
}

#[test]
fn a_manual_white_balance_below_the_scene_light_cools_and_above_it_warms() {
    // Setting the user's Kelvin above the scene's CCT (daylight 5500 K set to 9000 K) renders warm
    // (red above blue), below it (3200 K) cool (blue above red): the direction of an ALEXA white
    // balance, as in any raw converter.
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    let white = vec![1.0; 81];
    let p = load("sony-ilce-7rm4");
    let light = fx.light("ref:daylight:daylight-5500K").unwrap();
    let scene = scene_of(&p, light, true);
    let r_cam = camera_balanced(&p, light, &white);
    let mut previous_ratio = f64::INFINITY;
    for k in [2500.0, 3000.0, 3500.0, 4500.0, 5500.0, 6500.0, 9000.0] {
        let settings = ChainSettings {
            white_balance: WhiteBalance::Manual {
                kelvin: k,
                tint: 0.0,
            },
            ..ChainSettings::default()
        };
        let out = FusedChain::build(p.tables(), &scene, scene.w_scene, &settings)
            .unwrap()
            .apply(r_cam)
            .0;
        let sim = alexa_sim(&s_a, &fx, light, &white, Some(k));
        assert!(max_abs(out, sim) < 1e-6, "{k} K");
        let ratio = out[2] / out[0];
        assert!(
            ratio < previous_ratio,
            "blue over red must fall as the Kelvin rises: {k} K"
        );
        previous_ratio = ratio;
        if k < 5500.0 {
            assert!(ratio > 1.0, "{k} K should be cool");
        }
        if k > 5500.0 {
            assert!(ratio < 1.0, "{k} K should be warm");
        }
    }
}

#[test]
fn colored_patches_follow_the_simulation_up_to_the_sensor_map_error() {
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    for id in PROFILES {
        let p = load(id);
        for name in [
            "ref:daylight:daylight-5500K",
            "gate:daylight-6750K",
            "gate:blackbody-2850K",
        ] {
            let light = fx.light(name).unwrap();
            let scene = scene_of(&p, light, true);
            for k in [None, Some(3200.0), Some(5500.0), Some(2800.0)] {
                let settings = ChainSettings {
                    white_balance: match k {
                        None => WhiteBalance::Auto,
                        Some(kelvin) => WhiteBalance::Manual { kelvin, tint: 0.0 },
                    },
                    ..ChainSettings::default()
                };
                let chain =
                    FusedChain::build(p.tables(), &scene, scene.w_scene, &settings).unwrap();
                let mut diffs = Vec::new();
                for r in &fx.colorchecker.spectra {
                    let got = chain.apply(camera_balanced(&p, light, r)).0;
                    let want = alexa_sim(&s_a, &fx, light, r, k);
                    diffs.push(deltae::delta_e_2000(lab_of(got), lab_of(want)));
                }
                let s = Summary::of(&diffs).unwrap();
                eprintln!(
                    "{id} {name} WB {k:?}: ColorChecker chain vs simulation, dE00 median {:.3} p95 {:.3} max {:.3}",
                    s.median, s.p95, s.max
                );
                // The sensor map's gate is 0.5 median / 1.2 p95 on the CES. Measured here (24 patches, three
                // bodies, twelve lights and settings): median at most 0.42, p95 at most 1.33, max at most 1.80.
                // Before the ALEXA estimate moved (owner decision 2026-10-25, docs/03) the median was at most 0.47
                // and the max at most 1.4, and the p95 bound was 1.3; it is 1.4 now, for the same cause as the
                // sensor-map gate (the hue-pushed target is harder for a white-preserving 3x3 to reach).
                assert!(
                    s.median < 0.6 && s.p95 < 1.4 && s.max < 1.8,
                    "{id} {name} WB {k:?}: median {:.3} p95 {:.3} max {:.3}",
                    s.median,
                    s.p95,
                    s.max
                );
            }
        }
    }
}

#[test]
fn the_camera_locus_readout_gives_the_same_picture_as_the_exact_cct() {
    // The estimator passes a camera white, not a CCT: the CCT comes from the profile locus. For a CIE
    // daylight scene that readout is within a kelvin or two, and the 3200 K render agrees with the
    // simulation to the w_A slope times that (a few 1e-4).
    let Some(fx) = Fixtures::load_or_skip() else {
        return;
    };
    let s_a = alexa();
    let white = vec![1.0; 81];
    for id in PROFILES {
        let p = load(id);
        for name in [
            "ref:daylight:daylight-5500K",
            "ref:daylight:daylight-7000K",
            "ref:tungsten:blackbody-3000K",
        ] {
            let light = fx.light(name).unwrap();
            let scene = scene_of(&p, light, false);
            assert!(
                (scene.cct_k - light.cct_k).abs() < 25.0,
                "{id} {name}: read {} K",
                scene.cct_k
            );
            let settings = ChainSettings {
                white_balance: WhiteBalance::Manual {
                    kelvin: 3200.0,
                    tint: 0.0,
                },
                ..ChainSettings::default()
            };
            let chain = FusedChain::build(p.tables(), &scene, scene.w_scene, &settings).unwrap();
            let got = chain.apply(camera_balanced(&p, light, &white)).0;
            let want = alexa_sim(&s_a, &fx, light, &white, Some(3200.0));
            let d = max_abs(got, want) / want[2];
            assert!(d < 2e-3, "{id} {name}: relative difference {d:e}");
        }
    }
}
