//! M1-07, white balance maths on the shipped profiles (`docs/02-colour-pipeline.md`, sections C1 and
//! C3-C4): the light presets, the Auto Kelvin and tint readouts, Pick white and the residual gains.
//!
//! Uses the real `data/cameras/profiles/*/profile.json` tables and nothing from the git-ignored
//! fixtures, so it always runs.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus};
use ym_color::wb::{
    self, DAY_PRESET_K, EstimatorClasses, LightSource, PickWhiteError, SHADE_PRESET_K, SceneLight,
    TUNGSTEN_PRESET_K,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Profile {
    id: String,
    locus: Locus,
    white: AlexaWhite,
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

/// A scene light `duv` away from the locus at `cct_k` (camera Duv, positive towards magenta), read
/// back through the profile's locus.
fn light_at(p: &Profile, cct_k: f64, duv: f64) -> SceneLight {
    let rgb = p.locus.cct_duv_to_rgb(cct_k, duv);
    SceneLight::new(
        rgb,
        ClassProbabilities::only(class_for(cct_k)),
        0.9,
        &p.locus,
    )
    .unwrap()
}

/// The ALEXA white a manual pair `(kelvin, tint)` neutralises, applied to `w_s`: the channels the gains
/// leave after balancing, normalised to their mean.
fn balanced_spread(p: &Profile, w_s: [f64; 3], kelvin: f64, tint: f64) -> [f64; 3] {
    let g = wb::manual_user_gains(&p.white, kelvin, tint);
    let out = [w_s[0] * g[0], w_s[1] * g[1], w_s[2] * g[2]];
    let mean = (out[0] + out[1] + out[2]) / 3.0;
    out.map(|v| v / mean)
}

#[test]
fn presets_follow_the_table_on_every_profile() {
    for p in all_profiles() {
        // An estimate that is nothing like any preset: a 4600 K light with a magenta cast, mixed classes.
        let classes = ClassProbabilities::new(0.5, 0.1, 0.3, 0.1).unwrap();
        let rgb = p.locus.cct_duv_to_rgb(4600.0, 0.004);
        let estimate = SceneLight::new(rgb, classes, 0.7, &p.locus).unwrap();

        let auto = LightSource::Auto.scene_light(&estimate, &p.locus);
        assert_eq!(auto, estimate, "{}", p.id);

        for (preset, cct_k, class) in [
            (LightSource::Day, DAY_PRESET_K, LightClass::Daylight),
            (LightSource::Shade, SHADE_PRESET_K, LightClass::Daylight),
            (
                LightSource::Tungsten,
                TUNGSTEN_PRESET_K,
                LightClass::Tungsten,
            ),
        ] {
            let s = preset.scene_light(&estimate, &p.locus);
            assert_eq!(s.cct_k, cct_k, "{} {preset:?}", p.id);
            assert_eq!(
                s.classes,
                ClassProbabilities::only(class),
                "{} {preset:?}",
                p.id
            );
            // The camera white of the model light on this profile's locus: D 5500 K and D 7500 K from the
            // daylight table, a 3200 K blackbody from the blackbody table.
            assert_eq!(s.w_scene, p.locus.rgb_at(cct_k), "{} {preset:?}", p.id);
            assert_eq!(s.duv, 0.0);
            assert_eq!(s.confidence, 1.0);
        }

        for (preset, class) in [
            (LightSource::Led, LightClass::Led),
            (LightSource::Fluorescent, LightClass::Fluorescent),
        ] {
            let s = preset.scene_light(&estimate, &p.locus);
            // The estimated chromaticity, all the probability on the class.
            assert_eq!(s.w_scene, estimate.w_scene);
            assert_eq!(s.cct_k, estimate.cct_k);
            assert_eq!(s.duv, estimate.duv);
            assert_eq!(s.classes, ClassProbabilities::only(class));
        }
    }
}

#[test]
fn preset_whites_use_the_right_branch_of_the_locus() {
    let p = load("sony-ilce-7rm4");
    // Day and Shade are CIE daylight (the D series branch), Tungsten a blackbody (below 4000 K).
    let day = p.locus.rgb_at(5500.0);
    let tungsten = p.locus.rgb_at(3200.0);
    assert!(
        tungsten[0] > day[0] && tungsten[2] < day[2],
        "{tungsten:?} {day:?}"
    );
    assert_eq!(
        p.locus.chroma_at(3200.0),
        p.locus.blackbody.lerp_mired(3200.0)
    );
    assert_eq!(
        p.locus.chroma_at(5500.0),
        p.locus.daylight.lerp_mired(5500.0)
    );
}

#[test]
fn kelvin_readout_rounds_to_50_and_stays_in_the_control_range() {
    let p = load("sony-ilce-7rm4");
    let classes = ClassProbabilities::only(LightClass::Daylight);
    let read = |cct_k: f64| {
        let s = SceneLight::from_parts(p.locus.rgb_at(cct_k), classes, 1.0, cct_k, 0.0).unwrap();
        wb::auto_readout(&p.white, &p.locus, &s).kelvin_k
    };
    assert_eq!(read(5224.9), 5200.0);
    assert_eq!(read(5225.0), 5250.0);
    assert_eq!(read(5250.0), 5250.0);
    assert_eq!(read(6474.0), 6450.0);
    assert_eq!(read(1500.0), 2000.0);
    assert_eq!(read(1999.0), 2000.0);
    assert_eq!(read(11024.0), 11000.0);
    assert_eq!(read(25000.0), 11000.0);
    for cct_k in (2000..=11000).step_by(37) {
        let k = read(f64::from(cct_k));
        assert_eq!(k % 50.0, 0.0);
        assert!((k - f64::from(cct_k)).abs() <= 25.0);
    }
}

#[test]
fn tint_readout_is_zero_on_the_locus_and_signed_by_the_cast() {
    for p in all_profiles() {
        for cct_k in [
            2400.0, 2900.0, 3500.0, 4600.0, 5500.0, 6500.0, 8000.0, 10_000.0,
        ] {
            let on = wb::auto_readout(&p.white, &p.locus, &light_at(&p, cct_k, 0.0));
            assert!(on.tint.abs() < 0.1, "{} {cct_k} K: tint {}", p.id, on.tint);
            let mut previous = f64::INFINITY;
            // Duv from green (negative) to magenta: the tint falls from positive to negative.
            for duv in [-0.01, -0.005, -0.002, 0.002, 0.005, 0.01] {
                let s = light_at(&p, cct_k, duv);
                let r = wb::auto_readout(&p.white, &p.locus, &s);
                assert_eq!(
                    r.tint.signum(),
                    -duv.signum(),
                    "{} {cct_k} K duv {duv}: {r:?}",
                    p.id
                );
                assert!(r.tint < previous, "{} {cct_k} K duv {duv}", p.id);
                previous = r.tint;
            }
        }
    }
}

#[test]
fn tint_readout_cancels_the_cast_in_camera_space() {
    // Independent of the formula: apply the tint's green gain to the camera white and read the
    // locus distance again. The cast must be cancelled to within 6 % of its size (the Duv scale and the
    // chord geometry of the readout are not the same arithmetic as the tint's).
    let mut worst = 0.0_f64;
    for p in all_profiles() {
        for cct_k in [2600.0, 3200.0, 5000.0, 6500.0, 9000.0] {
            for duv in [-0.01, -0.004, 0.004, 0.01] {
                let s = light_at(&p, cct_k, duv);
                let t = wb::auto_readout(&p.white, &p.locus, &s).tint;
                let g = wb::tint_green_gain(t);
                // A green gain g on the camera white: R/G and B/G both divide by g.
                let corrected = [s.w_scene[0], g * s.w_scene[1], s.w_scene[2]];
                let after = p.locus.rgb_to_cct_duv(corrected).unwrap();
                let ratio = after.duv.abs() / s.duv.abs();
                worst = worst.max(ratio);
                assert!(
                    ratio < 0.06,
                    "{} {cct_k} K duv {duv}: tint {t:.2} leaves {:.4} of {:.4}",
                    p.id,
                    after.duv,
                    s.duv
                );
            }
        }
    }
    eprintln!(
        "tint readout: worst residual Duv {:.2} % of the cast",
        worst * 100.0
    );
}

#[test]
fn the_shown_kelvin_and_tint_balance_the_scene_white() {
    // With a CCT that is a multiple of 50 K the shown pair neutralises the scene's ALEXA white exactly:
    // the tint is the green gain that cancels the scene's own tint.
    for p in all_profiles() {
        for cct_k in [
            2300.0, 2950.0, 3500.0, 4150.0, 4750.0, 5250.0, 6150.0, 7350.0, 9900.0,
        ] {
            for duv in [-0.008, 0.0, 0.006] {
                let rgb = p.locus.cct_duv_to_rgb(cct_k, duv);
                let s = SceneLight::from_parts(
                    rgb,
                    ClassProbabilities::only(class_for(cct_k)),
                    1.0,
                    cct_k,
                    duv,
                )
                .unwrap();
                let r = wb::auto_readout(&p.white, &p.locus, &s);
                assert_eq!(r.kelvin_k, cct_k);
                let w_s = wb::scene_alexa_white(&p.white, &p.locus, &s);
                let spread = balanced_spread(&p, w_s, r.kelvin_k, r.tint);
                let off = spread.iter().fold(0.0_f64, |m, v| m.max((v - 1.0).abs()));
                // The readout holds the tint to +-50; none of these casts gets near that.
                assert!(r.tint.abs() < 40.0, "{} {cct_k} K: {r:?}", p.id);
                assert!(
                    off < 1e-12,
                    "{} {cct_k} K duv {duv}: {r:?} leaves {spread:?}",
                    p.id
                );
            }
        }
    }
}

#[test]
fn kelvin_rounding_costs_at_most_a_percent_and_a_half_of_red_blue_balance() {
    // The Kelvin shown is rounded to 50 K (up to 25 K away from the scene CCT): the red/blue balance of
    // `Manual` at the shown pair is off by the colour of that difference. A fixed number of kelvin is a
    // bigger step in colour at the warm end (6 mired at 2000 K, 1.6 mired at 4000 K), so the bound is
    // stated per band, over the whole range (every 3 K, all 11 profiles, lights on the locus):
    // 2000-2300 K 1.6 %, 2300-4000 K 1.3 % (the 4000 K junction included), above 4000 K 0.5 %.
    let bands = [
        (2000.0, 2300.0, 0.017),
        (2300.0, 4000.0, 0.014),
        (4000.0, 11_000.0, 0.006),
    ];
    let mut worst = [0.0_f64; 3];
    for p in all_profiles() {
        for cct_k in (2000..=11_000).step_by(3) {
            let cct_k = f64::from(cct_k);
            let classes = ClassProbabilities::only(class_for(cct_k));
            let s =
                SceneLight::from_parts(p.locus.rgb_at(cct_k), classes, 1.0, cct_k, 0.0).unwrap();
            let r = wb::auto_readout(&p.white, &p.locus, &s);
            let w_s = wb::scene_alexa_white(&p.white, &p.locus, &s);
            let spread = balanced_spread(&p, w_s, r.kelvin_k, r.tint);
            let off = spread.iter().fold(0.0_f64, |m, v| m.max((v - 1.0).abs()));
            for (k, (lo, hi, limit)) in bands.iter().enumerate() {
                if cct_k >= *lo && cct_k < *hi || (k == 2 && cct_k == 11_000.0) {
                    worst[k] = worst[k].max(off);
                    assert!(
                        off < *limit,
                        "{} {cct_k} K: {r:?} leaves {spread:?} (limit {limit})",
                        p.id
                    );
                }
            }
        }
    }
    eprintln!(
        "Manual at the rounded Auto readout: worst departure from neutral {:.3} % (2000-2300 K), {:.3} % (2300-4000 K), {:.3} % (above 4000 K)",
        worst[0] * 100.0,
        worst[1] * 100.0,
        worst[2] * 100.0
    );
}

#[test]
fn led_and_fluorescent_scenes_keep_the_class_white_and_ignore_the_camera_duv() {
    let p = load("sony-ilce-7rm4");
    for class in [LightClass::Led, LightClass::Fluorescent] {
        let classes = ClassProbabilities::only(class);
        // The class white at 5600 K, with three different camera casts.
        let reads: Vec<f64> = [-0.01, 0.0, 0.01]
            .iter()
            .map(|&duv| {
                let rgb = p.locus.cct_duv_to_rgb(5600.0, duv);
                let s = SceneLight::new(rgb, classes, 1.0, &p.locus).unwrap();
                assert_eq!(wb::scene_tint(&p.locus, &s), 0.0);
                wb::auto_readout(&p.white, &p.locus, &s).tint
            })
            .collect();
        // Duv does not enter (the CCT moves a little with the cast, hence the small tolerance), the
        // class list's own offset from the locus does.
        assert!((reads[0] - reads[2]).abs() < 0.3, "{class:?}: {reads:?}");
        let s = SceneLight::from_parts(p.locus.rgb_at(5600.0), classes, 1.0, 5600.0, 0.0).unwrap();
        let w_s = p.white.scene_white(5600.0, &classes);
        let w_u = p.white.locus_white(5600.0);
        let expect = -50.0 * ((w_s[0] / w_u[0]) * (w_s[2] / w_u[2])).log2();
        assert!((wb::auto_readout(&p.white, &p.locus, &s).tint - expect).abs() < 1e-12);
        assert!(
            expect.abs() > 0.05,
            "{class:?} should sit off the locus: {expect}"
        );
    }
}

#[test]
fn a_mixed_estimate_blends_the_scene_tint_by_the_model_classes() {
    let p = load("sony-ilce-7rm4");
    let rgb = p.locus.cct_duv_to_rgb(5000.0, -0.006);
    let full = SceneLight::new(
        rgb,
        ClassProbabilities::only(LightClass::Daylight),
        1.0,
        &p.locus,
    )
    .unwrap();
    let t_full = wb::scene_tint(&p.locus, &full);
    assert!(t_full > 5.0, "{t_full}");
    let half = SceneLight::new(
        rgb,
        ClassProbabilities::new(0.25, 0.25, 0.5, 0.0).unwrap(),
        1.0,
        &p.locus,
    )
    .unwrap();
    assert!((wb::scene_tint(&p.locus, &half) - 0.5 * t_full).abs() < 1e-9);
}

#[test]
fn tint_scale_is_one_green_gain_per_log_chroma() {
    // 100 / (sqrt 2 ln 2) tint units per unit of log-chroma distance, from the definition of the gain.
    let d = 0.05; // log-chroma distance towards magenta
    let t = -wb::TINT_PER_LOG_CHROMA * d;
    let g = wb::tint_green_gain(t); // > 1: more green cancels a magenta cast
    // g moves (ln R/G, ln B/G) by ln g in both coordinates: sqrt(2) ln g = d.
    assert!((2.0_f64.sqrt() * g.ln() - d).abs() < 1e-12);
    assert!((wb::TINT_PER_LOG_CHROMA - 102.0).abs() < 0.1);
}

#[test]
fn manual_gains_are_the_inverse_white_with_the_tint_on_green() {
    let p = load("sony-ilce-7rm4");
    for (kelvin, tint) in [
        (2000.0, -50.0),
        (3200.0, 0.0),
        (3999.0, 12.5),
        (4000.0, -7.0),
        (6500.0, 50.0),
        (11000.0, 3.0),
    ] {
        let w = p.white.locus_white(kelvin);
        let g = wb::manual_user_gains(&p.white, kelvin, tint);
        assert_eq!(g[0], 1.0 / w[0]);
        assert_eq!(g[2], 1.0 / w[2]);
        assert_eq!(g[1], 2.0_f64.powf(-tint / 100.0));
        // The gains neutralise the white of the tinted light, up to the overall factor g_green.
        let tinted = wb::tinted_white(w, tint);
        let out = [tinted[0] * g[0], tinted[1] * g[1], tinted[2] * g[2]];
        assert!(
            out.iter().all(|v| (v - g[1]).abs() < 1e-14),
            "{out:?} {g:?}"
        );
    }
    // Out-of-range input is held at the control's ends.
    assert_eq!(
        wb::manual_user_gains(&p.white, 500.0, 99.0),
        wb::manual_user_gains(&p.white, 2000.0, 50.0)
    );
    assert_eq!(
        wb::manual_user_gains(&p.white, 40_000.0, -99.0),
        wb::manual_user_gains(&p.white, 11000.0, -50.0)
    );
}

#[test]
fn user_white_follows_the_four_thousand_kelvin_rule() {
    let p = load("sony-ilce-7rm4");
    // Blackbody below 4000 K, daylight from there: the documented step of w_A at the junction.
    let below = p.white.locus_white(3999.99);
    let above = p.white.locus_white(4000.0);
    assert_eq!(below, p.white.blackbody_white(3999.99));
    assert_eq!(above, p.white.daylight_white(4000.0));
    let red_step = above[0] / below[0] - 1.0;
    let blue_step = above[2] / below[2] - 1.0;
    assert!((red_step + 0.012).abs() < 0.002, "red {red_step}");
    assert!((blue_step + 0.034).abs() < 0.003, "blue {blue_step}");
}

#[test]
fn pick_white_recovers_the_picked_light_and_keeps_the_classes() {
    for p in all_profiles() {
        // The mosaic was balanced for a 6000 K estimate.
        let w_init = p.locus.rgb_at(6000.0);
        let g0 = wb::initial_gains(w_init).unwrap();
        assert_eq!(g0[1], 1.0);
        let classes = ClassProbabilities::new(0.3, 0.1, 0.2, 0.4).unwrap();
        let estimate = SceneLight::new(w_init, classes, 0.6, &p.locus).unwrap();
        for cct_k in [2700.0, 3600.0, 4800.0, 7000.0] {
            let light = p.locus.rgb_at(cct_k);
            // A grey card in that light at 30 % of the green clip level, balanced with g0.
            let sample = [
                0.3 * light[0] * g0[0],
                0.3 * light[1] * g0[1],
                0.3 * light[2] * g0[2],
            ];
            let picked = estimate
                .with_picked_white(sample, w_init, &p.locus)
                .unwrap();
            for (got, want) in picked.w_scene.iter().zip(light) {
                assert!((got - want).abs() < 1e-14, "{} {cct_k} K", p.id);
            }
            assert_eq!(picked.w_scene[1], 1.0);
            assert_eq!(
                picked.classes, classes,
                "class probabilities stay as estimated"
            );
            assert_eq!(picked.confidence, 1.0);
            assert!(
                (picked.cct_k - cct_k).abs() < 2.0,
                "{} {cct_k} K read {}",
                p.id,
                picked.cct_k
            );
            assert!(picked.duv.abs() < 1e-4);

            // g_res = w_initial / w_scene turns the sample into a neutral.
            let g_res = wb::residual_gains(w_init, picked.w_scene).unwrap();
            let out = [
                sample[0] * g_res[0],
                sample[1] * g_res[1],
                sample[2] * g_res[2],
            ];
            assert!(
                (out[0] - out[1]).abs() < 1e-14 && (out[2] - out[1]).abs() < 1e-14,
                "{out:?}"
            );
        }
    }
}

#[test]
fn pick_white_messages_and_thresholds() {
    let p = load("sony-ilce-7rm4");
    let w_init = p.locus.rgb_at(5000.0);
    let estimate = SceneLight::new(
        w_init,
        ClassProbabilities::only(LightClass::Daylight),
        1.0,
        &p.locus,
    )
    .unwrap();
    let g0 = wb::initial_gains(w_init).unwrap();
    // Each channel clips at 1.0 raw, i.e. at g0[c] after the gains.
    let near = |frac: f64| g0.map(|g| frac * g);
    assert!(
        estimate
            .with_picked_white(near(0.89), w_init, &p.locus)
            .is_ok()
    );
    assert!(
        estimate
            .with_picked_white(near(0.5), w_init, &p.locus)
            .is_ok()
    );
    for channel in 0..3 {
        let mut sample = near(0.5);
        sample[channel] = 0.91 * g0[channel];
        let err = estimate
            .with_picked_white(sample, w_init, &p.locus)
            .unwrap_err();
        assert_eq!(err, PickWhiteError::TooBright, "channel {channel}");
        assert_eq!(
            err.to_string(),
            "That spot is too bright to use. Pick a grey or white area that isn't blown out."
        );
    }
    let dark = near(0.0005);
    assert_eq!(
        estimate
            .with_picked_white(dark, w_init, &p.locus)
            .unwrap_err(),
        PickWhiteError::TooDark
    );
}

#[test]
fn the_five_by_five_median_ignores_hot_pixels() {
    // A 5x5 patch of a grey card with three hot pixels and a dead one: the median is the card.
    let card = [0.2, 0.3, 0.25];
    let mut patch = vec![card; 25];
    patch[3] = [1.0, 1.0, 1.0];
    patch[11] = [0.9, 0.0, 0.7];
    patch[17] = [0.0, 0.0, 0.0];
    patch[24] = [0.99, 0.95, 0.9];
    assert_eq!(wb::median_rgb(&patch), Some(card));
}

#[test]
fn an_estimator_output_is_folded_and_read_out() {
    let p = load("sony-ilce-7rm4");
    let rgb = p.locus.rgb_at(3000.0);
    let classes = EstimatorClasses {
        daylight: 0.1,
        shade: 0.1,
        tungsten: 0.4,
        fluorescent: 0.1,
        led: 0.1,
        mixed: 0.2,
    };
    let s = SceneLight::from_estimate(rgb, &classes, 0.8, &p.locus).unwrap();
    assert!((s.cct_k - 3000.0).abs() < 1.0);
    // Mixed light at 3000 K counts as tungsten.
    let probs = s.classes.to_array();
    let expect = [0.2, 0.6, 0.1, 0.1];
    for (a, b) in probs.iter().zip(expect) {
        assert!((a - b).abs() < 1e-15, "{probs:?}");
    }
    assert_eq!(s.confidence, 0.8);
    // Green is normalised, bad whites are refused.
    let scaled = SceneLight::from_estimate(rgb.map(|c| c * 9.0), &classes, 0.8, &p.locus).unwrap();
    assert!((scaled.w_scene[0] - rgb[0]).abs() < 1e-15);
    assert!(SceneLight::from_estimate([0.5, 0.0, 0.5], &classes, 0.8, &p.locus).is_err());
    assert!(SceneLight::from_estimate(rgb, &classes, 1.5, &p.locus).is_err());
}

/// Planck's law on the 380 to 780 nm / 5 nm grid of the curves, `c2 = 0.014388` as in the fit scripts.
fn planck_380_780(t_k: f64) -> Vec<f64> {
    (0..81)
        .map(|i| {
            let l = (380.0 + 5.0 * f64::from(i)) * 1e-9;
            1.0 / (l.powi(5) * ((0.014388 / (l * t_k)).exp() - 1.0))
        })
        .collect()
}

#[test]
fn the_profile_locus_is_the_camera_white_in_the_cameras_own_raw_scale() {
    // The scene light is a camera white in the scale of the raw data (the mosaic after black subtraction and
    // normalisation, balanced with the initial gains). Pick white, the as-shot gains and the illuminant
    // estimate all deliver whites in that scale, and the presets (`locus.rgb_at`) and the CCT and Duv readout
    // (`Locus::rgb_to_cct_duv`) must agree with it. Weta's published curves keep the relative sensitivity of
    // the three channels (the red peak is 0.60 of the green peak on the A7 III), which fixes the raw white
    // balance. A locus tabulated from curves that were normalised to a peak of 1 per channel is off by those
    // factors: on an A7 IV a real D65 white reads about 5 % green, 6100 K and Duv -0.047 instead of
    // neutral, and the Day preset renders a real daylight white green.
    //
    // This test integrates the published curves (clipped at zero, not normalised) against Planck's law
    // and compares with the profile's blackbody locus: they must agree to the accuracy of the tables.
    let mut worst = 0.0_f64;
    let mut failures = Vec::new();
    for id in profile_ids() {
        let path = repo_root().join(format!("data/cameras/profiles/{id}/profile.json"));
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let curves = ym_color::spectral::SpectralCurves::from_path(
            repo_root()
                .join("data/cameras")
                .join(v["spectral_curves"].as_str().unwrap()),
        )
        .unwrap()
        .clipped_at_zero();
        let p = load(&id);
        for t_k in [2000.0, 2500.0, 3200.0, 3900.0] {
            let r = curves.response(&planck_380_780(t_k)).unwrap();
            let own = [r[0] / r[1], 1.0, r[2] / r[1]];
            let table = p.locus.rgb_at(t_k);
            let d = (0..3).fold(0.0_f64, |m, k| m.max((own[k] / table[k] - 1.0).abs()));
            worst = worst.max(d);
            if d > 2e-3 {
                failures.push(format!("{id} {t_k} K: curves {own:.4?} locus {table:.4?}"));
            }
        }
    }
    eprintln!(
        "camera white from the published curves against the profile locus: worst {worst:.3e}"
    );
    assert!(
        failures.is_empty(),
        "{} mismatches, first: {:?}",
        failures.len(),
        failures.first()
    );
}
