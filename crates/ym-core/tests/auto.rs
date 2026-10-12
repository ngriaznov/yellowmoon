//! Automatic estimates on synthetic ARW scenes with a known light, brightness and noise model.

#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

mod common;

use std::path::Path;

use common::{Layout, Scene, base_env, light};
use ym_color::spectral::LightClass;
use ym_core::auto::{
    AutoEstimates, AutoInputs, IlluminantSource, NoiseSource, as_shot_white, auto_detail_values,
    build_preview, burst_frame, display_to_sensor, image_area, meta_image_size, pick_white_sample,
};
use ym_core::camera::PhotoCamera;
use ym_core::session::SessionEnv;
use ym_raw::decode::RawImage;
use ym_raw::layout::Rect;
use ym_raw::testgen::{ArwBuilder, ContainerOnly, LosslessMs};

fn decode(path: &Path) -> RawImage {
    ym_raw::decode::decode(path).expect("decode the synthetic file")
}

fn estimates_of(raw: &RawImage, env: &SessionEnv) -> AutoEstimates {
    let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    AutoEstimates::compute(
        raw,
        &AutoInputs {
            profile,
            illuminant_model: None,
            priors: &env.priors,
        },
    )
    .expect("estimates")
}

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

#[test]
fn the_as_shot_fallback_reads_the_light_back_from_the_white_balance() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    for (cct, class) in [
        (2800.0, LightClass::Tungsten),
        (3200.0, LightClass::Tungsten),
        (5500.0, LightClass::Daylight),
        (6500.0, LightClass::Daylight),
        (7500.0, LightClass::Daylight),
    ] {
        let scene = Scene::new(512, 384, light(PROFILE, cct), Layout::Patchwork, 0.5);
        let raw = decode(&scene.write(dir.path(), "a.ARW", MODEL));
        let est = estimates_of(&raw, &env);
        assert_eq!(est.illuminant.source, IlluminantSource::AsShotFallback);
        let s = est.summary();
        // The as-shot levels are stored as integers, the readout is rounded to 50 K.
        assert!(
            (s.kelvin - cct).abs() <= 50.0,
            "{cct} K read back as {} K",
            s.kelvin
        );
        assert_eq!(s.light_class, class, "{cct} K");
        assert_eq!(s.light_confidence, 0.5);
        assert!(s.tint.abs() < 3.0, "tint {}", s.tint);
        // The white the mosaic is balanced with is the true light (green = 1).
        let w = est.colour.w_scene_initial;
        for c in 0..3 {
            let want = scene.white[c];
            assert!(
                (w[c] / want - 1.0).abs() < 2e-3,
                "channel {c}: {} against {want}",
                w[c]
            );
        }
    }
}

#[test]
fn fallback_classes_blend_between_tungsten_and_daylight() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(512, 384, light(PROFILE, 4000.0), Layout::Patchwork, 0.5);
    let raw = decode(&scene.write(dir.path(), "a.ARW", MODEL));
    let est = estimates_of(&raw, &env);
    let classes = est.illuminant.scene.classes;
    // 4000 K sits half way between the 3500 K and 4500 K ends of the blend.
    assert!((classes.get(LightClass::Tungsten) - 0.5).abs() < 0.05);
    assert!((classes.get(LightClass::Daylight) - 0.5).abs() < 0.05);
    assert_eq!(classes.get(LightClass::Led), 0.0);
    assert_eq!(classes.get(LightClass::Fluorescent), 0.0);
}

#[test]
fn a_file_with_no_white_balance_gets_daylight_with_no_confidence() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(512, 384, light(PROFILE, 6000.0), Layout::Patchwork, 0.5);
    let path = dir.path().join("nowb.ARW");
    scene
        .builder(MODEL)
        .wb(None)
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let raw = decode(&path);
    assert!(
        raw.wb_as_shot.is_none(),
        "the file really has no white balance"
    );
    assert!(as_shot_white(raw.wb_as_shot).is_none());
    let est = estimates_of(&raw, &env);
    assert_eq!(est.illuminant.source, IlluminantSource::DefaultDaylight);
    assert_eq!(est.summary().light_confidence, 0.0);
    assert!((est.summary().kelvin - 5500.0).abs() <= 50.0);
}

#[test]
fn auto_exposure_lands_a_neutral_scene_on_middle_grey() {
    // A neutral pixel of the scene has the camera RGB 0.9 L w / max(w); the initial gains 1 / w
    // balance it to 0.9 L / max(w) in all three channels, which the colour chain keeps as
    // Y = 0.9 L / max(w) (white-preserving map, balanced ARRI matrices). EV_auto = log2(0.18 / Y)
    // for a flat scene (docs/02 Auto exposure).
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let white = light(PROFILE, 5200.0);
    let peak = white.iter().copied().fold(0.0, f64::max);
    for level in [0.18, 0.045, 0.72, 0.09, 0.3] {
        let scene = Scene::new(512, 384, white, Layout::Uniform, level).noise(2e-4, 1e-6);
        let raw = decode(&scene.write(dir.path(), "e.ARW", MODEL));
        let ev = estimates_of(&raw, &env).summary().exposure_ev;
        let want = (0.18 / (0.9 * level / peak)).log2();
        assert!(
            (ev - want).abs() < 0.08,
            "level {level}: EV {ev}, want {want}"
        );
    }
}

#[test]
fn auto_exposure_follows_the_cameras_exposure_compensation_at_half_weight() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(512, 384, light(PROFILE, 5200.0), Layout::Uniform, 0.18);
    let path = dir.path().join("ec.ARW");
    scene
        .builder(MODEL)
        .exposure_comp(10, 10)
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let ev = estimates_of(&decode(&path), &env).summary().exposure_ev;
    let white = scene.white;
    let peak = white.iter().copied().fold(0.0, f64::max);
    let want = (0.18 / (0.9 * 0.18 / peak)).log2() + 0.5;
    assert!((ev - want).abs() < 0.08, "EV {ev}, want {want}");
}

#[test]
fn a_ramp_is_metered_in_the_middle_of_its_range() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(512, 384, light(PROFILE, 5200.0), Layout::Ramp, 0.5).noise(2e-4, 1e-6);
    let est = estimates_of(&decode(&scene.write(dir.path(), "r.ARW", MODEL)), &env);
    // The geometric mean of a ramp from 0.05 to 0.5 with a centre weight is between its ends.
    let m = 0.18 / est.exposure.own_ev.exp2();
    assert!((0.05..=0.5).contains(&m), "metered luminance {m}");
    assert!(est.exposure.meter.metered);
}

#[test]
fn the_noise_model_is_recovered_from_the_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    // One stop above the physical prior for ISO 800 on this body class, so the data has to win.
    let (a, b) = (6e-4, 2e-6);
    let mut worst = 0.0f64;
    for seed in 1..=3 {
        let scene = Scene::new(1536, 1024, light(PROFILE, 5200.0), Layout::Ramp, 0.9)
            .noise(a, b)
            .seed(seed);
        let path = dir.path().join(format!("n{seed}.ARW"));
        scene
            .builder(MODEL)
            .iso(800)
            .write_to(&path, &scene.mosaic())
            .unwrap();
        let est = estimates_of(&decode(&path), &env);
        assert_eq!(est.noise.source, NoiseSource::Image, "seed {seed}");
        let model = est.noise.model;
        // The noise level at three signal levels, which is what the sigma map uses.
        for mu in [0.05, 0.25, 0.6] {
            let truth = (a * mu + b).sqrt();
            let got = model.sigma(mu, 0);
            let err = (got / truth - 1.0).abs();
            worst = worst.max(err);
            assert!(
                err < 0.12,
                "seed {seed}, signal {mu}: sigma {got} against {truth}"
            );
        }
        let a_err = (model.a / a - 1.0).abs();
        assert!(a_err < 0.15, "seed {seed}: a {} against {a}", model.a);
    }
    eprintln!("worst sigma error over seeds and levels: {worst:.3}");
}

#[test]
fn a_camera_processed_file_has_no_mosaic_to_measure_and_gets_the_prior() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let (w, h) = (128u32, 96u32);
    let white = light(PROFILE, 5200.0);
    // Camera RGB, not balanced, of a neutral grey scene.
    let mut rgb = Vec::new();
    for _ in 0..w * h {
        for c in 0..3 {
            rgb.push((1024.0 + 0.4 * white[c] / white[1] * 14000.0).round() as u16);
        }
    }
    let path = dir.path().join("ms.ARW");
    ArwBuilder::new(MODEL, w, h)
        .lossless_ms(LosslessMs::new(
            ym_raw::layout::ChromaSubsampling::Yuv420,
            rgb,
            w,
            h,
            [1.0 / white[0], 1.0, 1.0 / white[2]].map(|g| g as f32),
        ))
        .write_to(&path, &[])
        .unwrap();
    let raw = decode(&path);
    assert!(raw.is_reduced());
    let est = estimates_of(&raw, &env);
    assert_eq!(est.noise.source, NoiseSource::PriorOnly);
    assert!(est.noise.model.a > 0.0 && est.noise.model.a.is_finite());
    assert!(est.noise.estimate.is_none());
}

#[test]
fn auto_sharpening_and_noise_reduction_follow_the_iso() {
    // docs/05 R7: L_auto = clamp(10 (1 + log2(ISO / 100)), 10, 90); C and D are 50.
    let (_, nr) = auto_detail_values(Some(100), 100, Some(5.6));
    assert_eq!((nr.luminance, nr.color, nr.detail), (10.0, 50.0, 50.0));
    let (_, nr) = auto_detail_values(Some(3200), 100, Some(5.6));
    assert!((nr.luminance - 60.0).abs() < 1e-4);
    let (_, nr) = auto_detail_values(Some(25600), 100, Some(5.6));
    assert!((nr.luminance - 90.0).abs() < 1e-4);
    // docs/05 R8: A = 120 at base ISO falling to 60 at ISO 12800; rho 0.7 from f/8, 0.9 below f/2.
    let (s, _) = auto_detail_values(Some(100), 100, Some(5.6));
    assert!((s.amount - 120.0).abs() < 1e-4 && (s.radius_px - 0.8).abs() < 1e-6);
    let (s, _) = auto_detail_values(Some(12800), 100, Some(8.0));
    assert!((s.amount - 60.0).abs() < 1e-3 && (s.radius_px - 0.7).abs() < 1e-6);
    let (s, _) = auto_detail_values(Some(400), 100, Some(1.4));
    assert!((s.radius_px - 0.9).abs() < 1e-6);
    // A body whose base ISO is 250 starts the sharpening ramp there.
    let (s, _) = auto_detail_values(Some(250), 250, None);
    assert!((s.amount - 120.0).abs() < 1e-4);
    // Without an ISO the base ISO is assumed.
    let (s, nr) = auto_detail_values(None, 100, None);
    assert!((s.amount - 120.0).abs() < 1e-4 && nr.luminance == 10.0);
}

#[test]
fn the_preview_bins_eight_by_eight_and_masks_clipped_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let mut scene = Scene::new(512, 384, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let path = dir.path().join("p.ARW");
    let mut mosaic = scene.mosaic();
    // One clipped photosite in block (3, 2).
    mosaic[(2 * 8 + 1) * 512 + 3 * 8 + 1] = 16300;
    scene.layout = Layout::Uniform;
    scene.builder(MODEL).write_to(&path, &mosaic).unwrap();
    let raw = decode(&path);
    let preview = build_preview(&raw).unwrap();
    assert_eq!((preview.width, preview.height), (64, 48));
    assert_eq!(preview.rgb.len(), 64 * 48 * 3);
    let clipped: Vec<usize> = (0..preview.clip.len())
        .filter(|&i| preview.clip[i] != 0)
        .collect();
    assert_eq!(clipped, vec![2 * 64 + 3]);
    // A neutral pixel of luminance L is 0.9 L w / max(w) per channel before any gain.
    let w = scene.white;
    let peak = w.iter().copied().fold(0.0, f64::max);
    let p = &preview.rgb[3 * (10 * 64 + 10)..][..3];
    for c in 0..3 {
        let want = 0.9 * 0.3 * w[c] / peak;
        assert!(
            (f64::from(p[c]) - want).abs() < 2e-3,
            "channel {c}: {} against {want}",
            p[c]
        );
    }
}

#[test]
fn the_image_area_is_the_crop_then_the_active_area_then_the_readout() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(512, 384, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let write = |name: &str, active: Option<(u32, u32)>, crop: Option<Rect>| {
        let mut b = scene.builder(MODEL);
        if let Some((w, h)) = active {
            b = b.active_area(w, h);
        }
        if let Some(c) = crop {
            b = b.crop(c);
        }
        let path = dir.path().join(name);
        b.write_to(&path, &scene.mosaic()).unwrap();
        decode(&path)
    };
    let whole = write("whole.ARW", None, None);
    let a = image_area(&whole);
    assert_eq!((a.x, a.y, a.width, a.height), (0, 0, 512, 384));
    let active = write("active.ARW", Some((504, 376)), None);
    let a = image_area(&active);
    assert_eq!((a.width, a.height), (504, 376));
    let crop = write(
        "crop.ARW",
        Some((504, 376)),
        Some(Rect {
            x: 8,
            y: 6,
            width: 400,
            height: 300,
        }),
    );
    let a = image_area(&crop);
    assert_eq!((a.x, a.y, a.width, a.height), (8, 6, 400, 300));
    // The probe alone gives the same size, swapped for a rotated orientation.
    assert_eq!(meta_image_size(&crop.meta), (400, 300));
    let rotated = scene
        .builder(MODEL)
        .orientation(6)
        .write_to(&dir.path().join("rot.ARW"), &scene.mosaic())
        .map(|()| decode(&dir.path().join("rot.ARW")))
        .unwrap();
    assert_eq!(meta_image_size(&rotated.meta), (384, 512));
}

#[test]
fn exif_orientations_map_display_points_back_to_the_sensor() {
    // Corner (0, 0) of the displayed photo, for orientations 1 to 8.
    let corners: Vec<[f64; 2]> = (1..=8).map(|o| display_to_sensor(o, [0.0, 0.0])).collect();
    assert_eq!(
        corners,
        vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    );
    // Every orientation is a bijection of the unit square that maps the centre to the centre.
    for o in 1..=8 {
        let c = display_to_sensor(o, [0.5, 0.5]);
        assert!(
            (c[0] - 0.5).abs() < 1e-12 && (c[1] - 0.5).abs() < 1e-12,
            "{o}"
        );
        let p = display_to_sensor(o, [0.25, 0.75]);
        let q = display_to_sensor(o, [0.75, 0.25]);
        assert!(((p[0] + q[0]) - 1.0).abs() < 1e-12 && ((p[1] + q[1]) - 1.0).abs() < 1e-12);
    }
    // 6: the top-left of the display is the bottom-left of the sensor (rotated 90 degrees clockwise).
    assert_eq!(display_to_sensor(6, [0.2, 0.0]), [0.0, 0.8]);
    assert_eq!(display_to_sensor(0, [0.3, 0.4]), [0.3, 0.4]);
}

#[test]
fn pick_white_samples_a_neutral_patch_balanced_with_the_initial_gains() {
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let white = light(PROFILE, 3000.0);
    let scene = Scene::new(512, 384, white, Layout::Patchwork, 0.5).noise(1e-4, 1e-6);
    let raw = decode(&scene.write(dir.path(), "w.ARW", MODEL));
    let est = estimates_of(&raw, &env);
    let g0 = est.initial_gains().unwrap();
    // Patch k = 2 (row 0, column 2) is neutral; its luminance multiplier comes from the scene.
    let (x, y) = (512 * 5 / 12, 384 / 8);
    let lum = scene.level * (0.12 + 0.88 * (14.0 / 23.0));
    let sample = pick_white_sample(&raw, [0.4167, 0.125], g0).expect("sample");
    let peak = white.iter().copied().fold(0.0, f64::max);
    // A neutral patch is balanced by g0 to equal channels: 0.9 L / (max w) / w_green... = grey.
    let grey = 0.9 * lum / peak;
    for c in 0..3 {
        assert!(
            (sample[c] / grey - 1.0).abs() < 0.03,
            "channel {c}: {} against {grey} at pixel ({x}, {y})",
            sample[c]
        );
    }
    // A point on a coloured patch (k = 3) is not neutral.
    let tint = pick_white_sample(&raw, [0.58, 0.125], g0).expect("sample");
    assert!((tint[0] / tint[2] - 1.0).abs() > 0.1, "{tint:?}");
    // Points outside the photo are clamped to its edge; NaN is refused.
    assert!(pick_white_sample(&raw, [2.0, -1.0], g0).is_some());
    assert!(pick_white_sample(&raw, [f64::NAN, 0.5], g0).is_none());
}

#[test]
fn bursts_are_grouped_by_camera_time_and_settings() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(64, 64, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let frame = |name: &str, sec: &str, subsec: &str, shutter: u32| {
        let path = dir.path().join(name);
        scene
            .builder(MODEL)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), subsec, Some("+02:00"))
            .exposure_time(1, shutter)
            .write_to(&path, &scene.mosaic())
            .unwrap();
        burst_frame(&decode(&path).meta)
    };
    let a = frame("a.ARW", "44", "50", 250);
    let b = frame("b.ARW", "45", "10", 250);
    let c = frame("c.ARW", "45", "10", 500);
    let (ta, tb) = (a.capture_time_s.unwrap(), b.capture_time_s.unwrap());
    assert!((tb - ta - 0.6).abs() < 1e-6, "{}", tb - ta);
    assert!(a.settings.same_as(&b.settings));
    assert!(!a.settings.same_as(&c.settings));
    assert_eq!(a.camera, b.camera);
    let groups = ym_color::exposure::burst_groups(&[a, b, c]);
    assert_eq!(groups, vec![0, 0, 1]);
}

#[test]
fn photos_that_cannot_be_analysed_say_so_in_plain_words() {
    // White at or below black: no range to normalise.
    let dir = tempfile::tempdir().unwrap();
    let env = base_env();
    let scene = Scene::new(64, 64, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let path = dir.path().join("bad.ARW");
    scene
        .builder(MODEL)
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let mut raw = decode(&path);
    raw.white = raw.black;
    let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
    let profile = env.profiles.resolve(&camera.matched);
    let err = AutoEstimates::compute(
        &raw,
        &AutoInputs {
            profile,
            illuminant_model: None,
            priors: &env.priors,
        },
    )
    .expect_err("no level range");
    assert!(err.to_string().contains("can't be analysed"), "{err}");
}

#[test]
fn the_container_only_arw6_file_is_still_probed_for_its_camera() {
    // A probe works on ARW 6.0 files; only the decode is refused (docs/04).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a7v.ARW");
    ArwBuilder::new("ILCE-7M5", 64, 64)
        .container_only(ContainerOnly::arw6(12))
        .write_to(&path, &[])
        .unwrap();
    let meta = ym_raw::metadata::probe(&path).unwrap();
    let camera = PhotoCamera::new(&base_env().profiles, &meta, None);
    assert!(!camera.is_supported());
    assert!(
        camera
            .unsupported_message()
            .unwrap()
            .starts_with("ARW 6.0 compressed files from the A7 V and A7R VI")
    );
}
