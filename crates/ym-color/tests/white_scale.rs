//! The white scale of a stand-in body (`docs/03`, "White scale of stand-in bodies"; `docs/02` C1): the
//! factor that takes the body's raw white to the white of the profile it borrows, `w_profile = w_body (.) s`.
//!
//! What the colour crate must get right, on the shipped profile tables:
//!
//! * with the identity scale every function agrees **to the last bit** with the one it grew from, so a body
//!   with its own curves renders exactly as before;
//! * a body whose raw is the profile's raw divided by `s` (the model of a stand-in, and what the synthetic
//!   scenes write), balanced with `g0 = s / w_init`, is the profile's balanced mosaic: Pick white returns the
//!   profile-scale white and refuses a sample by the body's own clip level;
//! * the clip level of a neutral surface follows the body's channel that clips first.

use std::path::PathBuf;

use serde_json::Value;
use ym_color::chain::{
    ChainSettings, ColorEstimates, FusedChain, ProfileTables, clip_level, clip_level_scaled,
};
use ym_color::exposure::ExposureSetting;
use ym_color::sensor_map::SensorMapTables;
use ym_color::spectral::{AlexaWhite, ClassProbabilities, LightClass, Locus};
use ym_color::wb::{
    self, IDENTITY_WHITE_SCALE, LightSource, PICK_CLIP_FRACTION, PickWhiteError, SceneLight,
    WbError, WhiteBalance,
};

/// A white scale of the size the A1 needs on the A7R IV profile (red low, blue high).
const SCALE: [f64; 3] = [0.8675, 1.0, 1.0861];

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

fn load(id: &str) -> Profile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../data/cameras/profiles/{id}/profile.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Profile {
        locus: serde_json::from_value(v["locus"].clone()).unwrap(),
        white: serde_json::from_value(v["alexa_white"].clone()).unwrap(),
        maps: serde_json::from_value(v["sensor_map"]["classes"].clone()).unwrap(),
    }
}

fn bits(v: [f64; 3]) -> [u64; 3] {
    v.map(f64::to_bits)
}

fn estimate(p: &Profile, cct_k: f64) -> SceneLight {
    SceneLight::new(
        p.locus.rgb_at(cct_k),
        ClassProbabilities::only(LightClass::Daylight),
        0.8,
        &p.locus,
    )
    .unwrap()
}

#[test]
fn the_identity_scale_changes_nothing_to_the_last_bit() {
    let p = load("sony-ilce-7rm4");
    for cct in [2300.0, 3200.0, 4100.0, 5500.0, 6500.0, 9000.0] {
        let w = p.locus.rgb_at(cct);
        assert_eq!(
            bits(wb::initial_gains(w).unwrap()),
            bits(wb::initial_gains_scaled(w, IDENTITY_WHITE_SCALE).unwrap()),
            "{cct} K"
        );
        // The same for a white that is not normalised.
        let raw = [w[0] * 3.7, w[1] * 3.7, w[2] * 3.7];
        assert_eq!(
            bits(wb::initial_gains(raw).unwrap()),
            bits(wb::initial_gains_scaled(raw, IDENTITY_WHITE_SCALE).unwrap())
        );
        // Pick white: a sample at 0.4 of the clip level, with a tint.
        let g0 = wb::initial_gains(w).unwrap();
        let sample = [0.4 * g0[0], 0.38 * g0[1], 0.41 * g0[2]];
        assert_eq!(
            bits(wb::picked_scene_white(sample, w).unwrap()),
            bits(wb::picked_scene_white_scaled(sample, w, IDENTITY_WHITE_SCALE).unwrap())
        );
        assert_eq!(
            wb::picked_scene_white(sample, w),
            wb::picked_scene_white_scaled(sample, w, IDENTITY_WHITE_SCALE)
        );
        assert_eq!(
            bits(wb::scale_white(w, IDENTITY_WHITE_SCALE).unwrap()),
            bits(wb::normalise_green(w).unwrap())
        );
    }
}

#[test]
fn the_identity_scale_builds_the_same_chain_and_clip_level_to_the_last_bit() {
    for id in ["sony-ilce-7rm4", "sony-ilce-7m3", "sony-dsc-rx100m4"] {
        let p = load(id);
        for cct in [2800.0, 5200.0, 7500.0] {
            let scene = estimate(&p, cct);
            let w_init = scene.w_scene;
            let g0 = wb::initial_gains(w_init).unwrap();
            let sample = [0.3 * g0[0], 0.3 * g0[1], 0.3 * g0[2]];
            for picked in [None, Some(sample)] {
                let plain = ChainSettings {
                    picked_white: picked,
                    exposure_ev: 0.5,
                    ..ChainSettings::default()
                };
                let explicit = ChainSettings {
                    white_scale: IDENTITY_WHITE_SCALE,
                    ..plain
                };
                let a = FusedChain::build(p.tables(), &scene, w_init, &plain).unwrap();
                let b = FusedChain::build(p.tables(), &scene, w_init, &explicit).unwrap();
                assert_eq!(a, b, "{id} {cct} K");
                let m = a.matrix();
                assert_eq!(
                    clip_level(m, w_init),
                    clip_level_scaled(m, w_init, IDENTITY_WHITE_SCALE)
                );
                assert_eq!(a.clip(), &clip_level(m, a.scene().w_scene));
            }
        }
    }
}

#[test]
fn a_body_balanced_with_the_scaled_gains_is_the_profiles_balanced_mosaic() {
    let p = load("sony-ilce-7rm4");
    for cct in [2700.0, 3800.0, 5200.0, 6500.0, 8000.0] {
        // The light on the profile's scale, and the body's own raw white of it.
        let w_profile = p.locus.rgb_at(cct);
        let w_body = [
            w_profile[0] / SCALE[0],
            w_profile[1] / SCALE[1],
            w_profile[2] / SCALE[2],
        ];
        // The as-shot white of the body, put on the profile's scale, is the profile's white.
        let on_profile = wb::scale_white(w_body, SCALE).unwrap();
        for c in 0..3 {
            assert!(
                (on_profile[c] / w_profile[c] - 1.0).abs() < 1e-14,
                "{cct} K {c}"
            );
        }
        // g0 = s / w_init balances the body's raw to what the profile's curves would have recorded.
        let g0 = wb::initial_gains_scaled(w_profile, SCALE).unwrap();
        for c in 0..3 {
            let balanced_body = w_body[c] * g0[c];
            let balanced_profile = w_profile[c] * wb::initial_gains(w_profile).unwrap()[c];
            assert!(
                (balanced_body - balanced_profile).abs() < 1e-14,
                "{cct} K channel {c}: {balanced_body} against {balanced_profile}"
            );
            assert!((balanced_body - 1.0).abs() < 1e-14);
        }
        // Pick white on a neutral patch of another light, 40 percent of the way to clipping: the body's sample
        // is `raw_body * g0`, and the picked white is that light on the profile's scale.
        let other = p.locus.rgb_at(cct * 0.8);
        let other_body = [
            other[0] / SCALE[0],
            other[1] / SCALE[1],
            other[2] / SCALE[2],
        ];
        let sample = [
            0.4 * other_body[0] * g0[0],
            0.4 * other_body[1] * g0[1],
            0.4 * other_body[2] * g0[2],
        ];
        let picked = wb::picked_scene_white_scaled(sample, w_profile, SCALE).unwrap();
        let want = wb::normalise_green(other).unwrap();
        for c in 0..3 {
            assert!(
                (picked[c] / want[c] - 1.0).abs() < 1e-12,
                "{cct} K channel {c}"
            );
        }
        // The sample times `w_init` is already on the profile's scale whatever the scale is: it only moves the
        // clip level the sample is judged by (next test).
        assert_eq!(
            bits(wb::picked_scene_white(sample, w_profile).unwrap()),
            bits(picked)
        );
    }
}

#[test]
fn pick_white_refuses_a_sample_by_the_bodys_own_clip_level() {
    let p = load("sony-ilce-7rm4");
    let w = p.locus.rgb_at(5200.0);
    let g0 = wb::initial_gains_scaled(w, SCALE).unwrap();
    // The raw clips at 1.0 per channel before the gains, so after them channel c clips at g0[c].
    let at = |fraction: f64| [fraction * g0[0], 0.5 * g0[1], 0.5 * g0[2]];
    assert!(wb::picked_scene_white_scaled(at(PICK_CLIP_FRACTION * 0.999), w, SCALE).is_ok());
    assert_eq!(
        wb::picked_scene_white_scaled(at(PICK_CLIP_FRACTION * 1.001), w, SCALE),
        Err(PickWhiteError::TooBright)
    );
    // The same red sample is fine for a body with the identity scale when its own clip is higher.
    let loose = at(PICK_CLIP_FRACTION * 0.999);
    let plain_clip = wb::initial_gains(w).unwrap();
    assert!(
        loose[0] < PICK_CLIP_FRACTION * plain_clip[0],
        "the scaled red clips lower than the plain red does (scale {})",
        SCALE[0]
    );
    // Too dark by the same level.
    let dark = [1e-6 * g0[0], 0.5 * g0[1], 0.5 * g0[2]];
    assert_eq!(
        wb::picked_scene_white_scaled(dark, w, SCALE),
        Err(PickWhiteError::TooDark)
    );
}

#[test]
fn the_clip_level_follows_the_channel_that_clips_first_in_the_bodys_raw() {
    let p = load("sony-ilce-7rm4");
    for cct in [2500.0, 4000.0, 5500.0, 7500.0] {
        let scene = estimate(&p, cct);
        let chain = FusedChain::build(
            p.tables(),
            &scene,
            scene.w_scene,
            &ChainSettings {
                white_scale: SCALE,
                ..ChainSettings::default()
            },
        )
        .unwrap();
        // A neutral surface of brightness b has the body raw b * w_body; the first channel to reach 1.0.
        let w = chain.scene().w_scene;
        let first = (0..3)
            .map(|c| SCALE[c] / w[c])
            .fold(f64::INFINITY, f64::min);
        assert_eq!(chain.clip().balanced, first, "{cct} K");
        let direct = clip_level_scaled(chain.matrix(), w, SCALE);
        assert_eq!(chain.clip(), &direct);
        // Brute force: raise the brightness of the body's white until a channel hits the sensor's clip.
        let w_body = [w[0] / SCALE[0], w[1] / SCALE[1], w[2] / SCALE[2]];
        let brute = (1..400_000)
            .map(|i| f64::from(i) * 1e-5)
            .find(|b| w_body.iter().any(|c| c * b >= 1.0))
            .unwrap();
        assert!(
            (brute - first).abs() < 2e-5,
            "{cct} K: {brute} against {first}"
        );
    }
}

#[test]
fn a_scaled_chain_still_lands_the_scene_white_on_neutral() {
    // Everything but Pick white and the clip level works on the profile's scale already, so the Auto
    // balance of a stand-in body is exact: the scene white (profile scale) is mapped to 2^EV [1, 1, 1].
    let p = load("sony-ilce-7rm4");
    for cct in [2800.0, 4300.0, 5600.0, 7200.0] {
        let scene = estimate(&p, cct);
        let estimates = ColorEstimates {
            scene,
            w_scene_initial: scene.w_scene,
            white_scale: SCALE,
            ev_auto: 0.6,
            white_shift_mired: 0.0,
            tint_shift: 0.0,
        };
        let chain = estimates
            .chain(
                p.tables(),
                LightSource::Auto,
                WhiteBalance::Auto,
                ExposureSetting::Auto,
                None,
            )
            .unwrap();
        // The balanced white: the body's raw white times g0 is [1, 1, 1] (to rounding).
        let g0 = estimates.initial_gains().unwrap();
        let w_body = [
            scene.w_scene[0] / SCALE[0],
            scene.w_scene[1] / SCALE[1],
            scene.w_scene[2] / SCALE[2],
        ];
        let balanced = [w_body[0] * g0[0], w_body[1] * g0[1], w_body[2] * g0[2]];
        let out = chain.apply(balanced).0;
        let gain = 0.6f64.exp2();
        for (c, v) in out.iter().enumerate() {
            assert!((v / gain - 1.0).abs() < 1e-12, "{cct} K channel {c}: {v}");
        }
        // The metering chain carries the scale too.
        let metering = estimates.metering_chain(p.tables()).unwrap();
        assert_eq!(metering.clip().balanced, chain.clip().balanced);
    }
}

#[test]
fn unusable_scales_are_refused() {
    let w = [0.4, 1.0, 0.7];
    for bad in [
        [0.0, 1.0, 1.0],
        [-0.9, 1.0, 1.0],
        [f64::NAN, 1.0, 1.0],
        [1.0, 1.0, f64::INFINITY],
        [1.0, 2.0, 1.0],
        [0.9, 0.5, 1.1],
    ] {
        assert!(
            matches!(
                wb::initial_gains_scaled(w, bad),
                Err(WbError::InvalidWhite(_))
            ),
            "{bad:?}"
        );
        assert!(matches!(
            wb::scale_white(w, bad),
            Err(WbError::InvalidWhite(_))
        ));
        assert!(matches!(
            wb::picked_scene_white_scaled([0.1, 0.2, 0.1], w, bad),
            Err(PickWhiteError::Wb(_))
        ));
    }
}
