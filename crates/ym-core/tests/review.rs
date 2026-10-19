//! Regression tests from the review of ym-core part 1: integration details between the decoder's
//! conventions (CFA positions, levels per position, orientation, crop) and the estimates, and the
//! session's behaviour under concurrent commands.

#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

mod common;

use common::{Layout, Scene, estimates_of_file, light};
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_core::auto::{build_preview, pick_white_sample};
use ym_raw::layout::{CfaPattern, Layout as Layout2, Rect};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A mosaic whose raw values are shifted per CFA position by `offsets` (so black levels that
/// differ per position, declared as `black + offset`, leave the normalised signal unchanged).
fn shifted(scene: &Scene, offsets: [u16; 4]) -> Vec<u16> {
    let mut m = scene.mosaic();
    let w = scene.width as usize;
    for (i, v) in m.iter_mut().enumerate() {
        let (x, y) = (i % w, i / w);
        *v += offsets[((y & 1) << 1) | (x & 1)];
    }
    m
}

#[test]
fn levels_per_cfa_position_follow_the_pixel_when_the_image_area_starts_on_an_odd_pixel() {
    // The readout starts with a margin that is an odd number of pixels, so the first pixel of the
    // photo sits on a blue-row/green-column position of the readout. Black levels differ per
    // position, so any mix-up of positions shows as a colour cast in the preview and as a wrong
    // noise level.
    let dir = tempfile::tempdir().unwrap();
    let white = light(PROFILE, 5200.0);
    let offsets = [0u16, 30, 60, 90];
    let black: [u16; 4] = std::array::from_fn(|p| 512 + offsets[p]);
    let write = |scene: &Scene, name: &str| {
        let path = dir.path().join(name);
        scene
            .builder(MODEL)
            .black(black)
            .active_area(632, 472)
            .crop(Rect {
                x: 3,
                y: 5,
                width: 600,
                height: 440,
            })
            .write_to(&path, &shifted(scene, offsets))
            .unwrap();
        estimates_of_file(&path)
    };
    // Noiseless: the preview of a neutral ramp has the camera white at every block.
    let clean = Scene::new(640, 480, white, Layout::Ramp, 0.4);
    let (raw, _, _) = write(&clean, "odd-clean.ARW");
    let preview = build_preview(&raw).unwrap();
    let peak = white.iter().copied().fold(0.0, f64::max);
    let mut worst = 0.0f64;
    for by in 0..preview.height {
        for bx in 0..preview.width {
            let p = &preview.rgb[3 * (by * preview.width + bx)..][..3];
            // The block's centre column in readout coordinates.
            let x = 3.0 + (bx as f64 + 0.5) * 600.0 / preview.width as f64;
            let lum = 0.4 * (0.1 + 0.9 * x / 639.0);
            for c in 0..3 {
                let want = 0.9 * lum * white[c] / peak;
                worst = worst.max((f64::from(p[c]) - want).abs());
            }
        }
    }
    assert!(worst < 1.5e-3, "preview colours are off by {worst}");
    // Noisy: the gain is recovered.
    let noisy = Scene::new(640, 480, white, Layout::Ramp, 0.4).noise(4e-4, 4e-6);
    let (_, est, _) = write(&noisy, "odd-noisy.ARW");
    let a = est.noise.model.a;
    assert!(
        (a / 4e-4 - 1.0).abs() < 0.3,
        "noise gain {a} for a true 4e-4"
    );
}

#[test]
fn the_as_shot_white_balance_is_read_by_color_not_by_position() {
    // A BGGR sensor (not a Sony layout, but the decoder reads it): `wb_as_shot` is stored per CFA
    // position, so the first value belongs to blue.
    let dir = tempfile::tempdir().unwrap();
    let white = light(PROFILE, 3200.0);
    let scene = Scene::new(256, 192, white, Layout::Uniform, 0.3);
    let path = dir.path().join("bggr.ARW");
    scene
        .builder(MODEL)
        .cfa(CfaPattern::BGGR)
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let (raw, est, _) = estimates_of_file(&path);
    assert!(matches!(
        raw.layout,
        Layout2::Bayer {
            pattern: CfaPattern::BGGR,
            ..
        }
    ));
    let w = est.color.w_scene_initial;
    for c in 0..3 {
        assert!(
            (w[c] / white[c] - 1.0).abs() < 2e-3,
            "channel {c}: {} against {}",
            w[c],
            white[c]
        );
    }
}

/// Where a point of the sensor (fractions of the image area) appears in the photo as the camera
/// oriented it, from the EXIF definitions of the orientation values (rotate 90 degrees clockwise
/// sends the sensor's top-left corner to the display's top-right corner, and so on). Written in
/// the direction opposite to `display_to_sensor`, so the two check each other.
fn sensor_to_display(orientation: u16, [sx, sy]: [f64; 2]) -> [f64; 2] {
    match orientation {
        2 => [1.0 - sx, sy],
        3 => [1.0 - sx, 1.0 - sy],
        4 => [sx, 1.0 - sy],
        5 => [sy, sx],
        6 => [1.0 - sy, sx],
        7 => [1.0 - sy, 1.0 - sx],
        8 => [sy, 1.0 - sx],
        _ => [sx, sy],
    }
}

#[test]
fn pick_white_follows_the_exif_orientation_of_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let white = light(PROFILE, 4500.0);
    let scene = Scene::new(384, 288, white, Layout::Patchwork, 0.5);
    let mosaic = scene.mosaic();
    let write = |name: &str, orientation: u16| {
        let path = dir.path().join(name);
        scene
            .builder(MODEL)
            .orientation(orientation)
            .write_to(&path, &mosaic)
            .unwrap();
        estimates_of_file(&path)
    };
    let (raw1, est1, _) = write("o1.ARW", 1);
    let g0 = est1.initial_gains().unwrap();
    // Centres of patches with different luminance and colour (6 x 4 patches).
    let sensor_points = [
        [0.5 / 6.0, 0.5 / 4.0],
        [5.5 / 6.0, 3.5 / 4.0],
        [3.5 / 6.0, 0.5 / 4.0],
        [0.5 / 6.0, 2.5 / 4.0],
    ];
    for orientation in 1..=8u16 {
        let (raw, _, _) = write(&format!("o{orientation}.ARW"), orientation);
        for sensor in sensor_points {
            let display = sensor_to_display(orientation, sensor);
            let via_display = pick_white_sample(&raw, display, g0).unwrap();
            let direct = pick_white_sample(&raw1, sensor, g0).unwrap();
            for c in 0..3 {
                assert!(
                    (via_display[c] - direct[c]).abs() < 1e-9,
                    "orientation {orientation}, sensor {sensor:?} shown at {display:?}: {via_display:?} against {direct:?}"
                );
            }
        }
    }
}

mod concurrency {
    use super::*;
    use common::{session_with_events, settle};
    use std::sync::Arc;
    use ym_core::session::{OpenOptions, Session};
    use ym_core::settings::{AutoOr, SettingsPatch};

    /// A folder of five photos; the last is large and, with the first as the current photo, outside
    /// the window of cached mosaics, so Pick white has to decode it again (slow enough for another
    /// command to run in the meantime).
    fn session_with_a_slow_photo() -> (tempfile::TempDir, Arc<Session>, ym_core::session::PhotoId) {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for i in 0..4 {
            Scene::new(64, 48, white, Layout::Uniform, 0.3).write(
                dir.path(),
                &format!("a{i}.ARW"),
                MODEL,
            );
        }
        Scene::new(3000, 2000, white, Layout::Uniform, 0.3).write(dir.path(), "z-big.ARW", MODEL);
        let (session, _rx) = session_with_events();
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        let session = Arc::new(session);
        let big = *session.photo_ids().last().unwrap();
        assert!(
            !session.cached_ids().contains(&big),
            "the big photo is outside the window"
        );
        (dir, session, big)
    }

    #[test]
    fn commands_on_one_photo_do_not_overwrite_each_other() {
        let (_dir, session, big) = session_with_a_slow_photo();
        for round in 0..4u32 {
            let contrast = 10.0 + f64::from(round);
            let point = [0.3 + 0.1 * f64::from(round), 0.5];
            let pick = {
                let session = Arc::clone(&session);
                std::thread::spawn(move || {
                    session
                        .update_settings(
                            big,
                            &SettingsPatch {
                                white_pick: Some(Some(point)),
                                ..SettingsPatch::default()
                            },
                        )
                        .unwrap();
                })
            };
            // Let the pick start decoding, then move a slider on the same photo.
            std::thread::sleep(std::time::Duration::from_millis(3));
            let slider = {
                let session = Arc::clone(&session);
                std::thread::spawn(move || {
                    session
                        .update_settings(
                            big,
                            &SettingsPatch {
                                contrast: Some(ym_core::settings::PercentSetting::manual_value(
                                    contrast,
                                )),
                                exposure: Some(AutoOr::Manual(ym_core::settings::ExposureValue {
                                    ev: 0.5,
                                })),
                                ..SettingsPatch::default()
                            },
                        )
                        .unwrap();
                })
            };
            pick.join().unwrap();
            slider.join().unwrap();
            let settings = session.settings(big).unwrap();
            assert_eq!(settings.settings.white_pick, Some(point), "round {round}");
            assert_eq!(
                settings.settings.contrast,
                ym_core::settings::PercentSetting::manual_value(contrast),
                "round {round}"
            );
            assert_eq!(
                settings.settings.exposure,
                AutoOr::Manual(ym_core::settings::ExposureValue { ev: 0.5 }),
                "round {round}"
            );
        }
    }
}

mod extreme_frames {
    use super::*;
    use ym_core::auto::{AutoEstimates, AutoInputs};
    use ym_core::camera::PhotoCamera;

    fn run(
        width: u32,
        height: u32,
        fill: impl Fn(u32, u32) -> u16,
        tweak: impl Fn(ym_raw::testgen::ArwBuilder) -> ym_raw::testgen::ArwBuilder,
    ) -> Result<AutoEstimates, String> {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        let scene = Scene::new(width, height, white, Layout::Uniform, 0.3);
        let mosaic: Vec<u16> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| fill(x, y))
            .collect();
        let path = dir.path().join("x.ARW");
        tweak(scene.builder(MODEL))
            .write_to(&path, &mosaic)
            .map_err(|e| e.to_string())?;
        let raw = ym_raw::decode::decode(&path).map_err(|e| e.to_string())?;
        let env = common::base_env();
        let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
        let profile = env.profiles.resolve(&camera.matched);
        AutoEstimates::compute(
            &raw,
            &AutoInputs {
                profile,
                illuminant_model: None,
                priors: &env.priors,
            },
        )
        .map_err(|e| e.to_string())
    }

    #[test]
    fn tiny_black_white_and_odd_shaped_frames_give_estimates_or_a_plain_error() {
        let id = |b: ym_raw::testgen::ArwBuilder| b;
        let mut outcomes = Vec::new();
        for (w, h) in [
            (2, 2),
            (3, 3),
            (4, 4),
            (8, 8),
            (15, 9),
            (16, 16),
            (33, 17),
            (4000, 2),
            (2, 3000),
        ] {
            for (name, level) in [
                ("black", 512u16),
                ("grey", 4000),
                ("white", 16383),
                ("noise", 0),
            ] {
                let r = run(
                    w,
                    h,
                    |x, y| {
                        if level == 0 {
                            512 + ((x * 7 + y * 13) % 900) as u16
                        } else {
                            level
                        }
                    },
                    id,
                );
                match &r {
                    Ok(est) => {
                        let s = est.summary();
                        assert!(
                            s.exposure_ev.is_finite() && s.kelvin.is_finite() && s.tint.is_finite(),
                            "{w}x{h} {name}"
                        );
                        assert!(
                            s.noise.a.is_finite() && s.noise.b.is_finite(),
                            "{w}x{h} {name}"
                        );
                    }
                    Err(e) => assert!(!e.is_empty(), "{w}x{h} {name}"),
                }
                outcomes.push(format!(
                    "{w}x{h} {name}: {}",
                    r.map(|_| "ok".to_owned()).unwrap_or_else(|e| e)
                ));
            }
        }
        eprintln!("{}", outcomes.join("\n"));
    }
}

mod session_lifecycle {
    use super::*;
    use common::{session_with_events, settle};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;
    use ym_core::auto::NoiseSource;
    use ym_core::session::{
        EventSink, OpenOptions, PhotoId, PhotoProgress, PhotoStage, SessionError, SessionEvent,
        ThumbnailInput, ThumbnailRenderer,
    };
    use ym_core::settings::SettingsPatch;
    use ym_looks::library::LookId;

    #[test]
    fn photo_ids_are_never_reused_so_a_late_command_cannot_reach_another_photo() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        Scene::new(64, 48, white, Layout::Uniform, 0.3).write(dir.path(), "a.ARW", MODEL);
        Scene::new(64, 48, white, Layout::Uniform, 0.4).write(dir.path(), "b.ARW", MODEL);
        let (session, _rx) = session_with_events();
        let first = session.open(&[dir.path().join("a.ARW")], OpenOptions::default());
        settle(&session);
        let old = first.added[0];
        session.close();
        let second = session.open(&[dir.path().join("b.ARW")], OpenOptions::default());
        settle(&session);
        let new = second.added[0];
        assert_ne!(old, new, "the id of a closed session's photo came back");
        // A command that names the old photo does not touch the new one.
        let err = session
            .update_settings(old, &SettingsPatch::exposure_ev(1.0))
            .expect_err("the old photo is gone");
        assert!(
            matches!(err, SessionError::UnknownPhoto(id) if id == old),
            "{err}"
        );
        assert!(session.settings(new).unwrap().settings.exposure.is_auto());
        assert!(matches!(
            session.rerun_auto(old),
            Err(SessionError::UnknownPhoto(_))
        ));
    }

    #[test]
    fn photos_that_do_not_say_their_iso_are_not_pooled() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for (name, iso) in [
            ("n0.ARW", 0),
            ("n1.ARW", 0),
            ("i0.ARW", 400),
            ("i1.ARW", 400),
        ] {
            let scene = Scene::new(512, 384, white, Layout::Ramp, 0.6)
                .noise(2e-4, 1e-6)
                .seed(u64::from(name.as_bytes()[1]));
            scene
                .builder(MODEL)
                .iso(iso)
                .write_to(&dir.path().join(name), &scene.mosaic())
                .unwrap();
        }
        let (session, _rx) = session_with_events();
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        let state = session.state();
        let source = |name: &str| {
            let p = state.photos.iter().find(|p| p.file_name == name).unwrap();
            session.estimates(p.id).unwrap().noise.source
        };
        // Control: two files with the same ISO share their estimate.
        assert!(
            matches!(source("i0.ARW"), NoiseSource::Pooled { files: 2 }),
            "{:?}",
            source("i0.ARW")
        );
        // Files that do not know their ISO cannot be told apart by it: each keeps its own.
        let p = state
            .photos
            .iter()
            .find(|p| p.file_name == "n0.ARW")
            .unwrap();
        assert_eq!(p.iso, 0);
        assert_eq!(source("n0.ARW"), NoiseSource::Image);
        assert_eq!(source("n1.ARW"), NoiseSource::Image);
    }

    #[test]
    fn the_estimates_a_photo_is_served_do_not_depend_on_the_number_of_decode_threads() {
        // A burst (shared exposure) and a group of one ISO (pooled noise), developed by one thread
        // and by four: which job finishes first must not show in any number.
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for i in 0..6u32 {
            let level = 0.1 + 0.12 * f64::from(i);
            let scene = Scene::new(256, 192, white, Layout::Ramp, level)
                .noise(2e-4, 1e-6)
                .seed(u64::from(i) + 7);
            let burst = i < 3;
            let mut b = scene
                .builder(MODEL)
                .iso(if i < 3 || i % 2 == 0 { 400 } else { 800 });
            if burst {
                b = b
                    .capture_time(&format!("2026:10:04 12:03:4{i}"), "10", Some("+02:00"))
                    .exposure_time(1, 250);
            } else {
                b = b
                    .capture_time(&format!("2026:10:04 13:0{i}:00"), "00", Some("+02:00"))
                    .exposure_time(1, 125 + i);
            }
            b.write_to(&dir.path().join(format!("f{i}.ARW")), &scene.mosaic())
                .unwrap();
        }
        let run = |threads: usize| {
            let session =
                ym_core::session::Session::new(common::base_env().with_decode_threads(threads));
            session.open(&[dir.path().to_path_buf()], OpenOptions::default());
            settle(&session);
            session
                .photo_ids()
                .into_iter()
                .map(|id| {
                    let e = session.estimates(id).unwrap();
                    let s = e.summary();
                    (
                        s.exposure_ev.to_bits(),
                        s.kelvin.to_bits(),
                        s.tint.to_bits(),
                        s.noise.a.to_bits(),
                        s.noise.b.to_bits(),
                        e.noise.source,
                    )
                })
                .collect::<Vec<_>>()
        };
        let one = run(1);
        for threads in [4, 3, 4] {
            assert_eq!(run(threads), one, "{threads} threads");
        }
        // The group really did share: the three burst frames have one exposure.
        assert_eq!(one[0].0, one[1].0);
        assert_eq!(one[1].0, one[2].0);
        assert!(
            one.iter()
                .any(|r| matches!(r.5, NoiseSource::Pooled { .. }))
        );
    }

    /// Panics for one photo (by file name), renders the others.
    struct PanicsFor(&'static str, Mutex<Vec<PhotoId>>);

    impl ThumbnailRenderer for PanicsFor {
        fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
            self.1.lock().unwrap().push(input.id);
            if input.id.0 == self.0.parse::<u32>().unwrap() {
                panic!("the renderer broke on purpose");
            }
            Ok(vec![0x89, b'P', b'N', b'G'])
        }
    }

    #[test]
    fn a_panic_in_a_job_skips_that_photo_and_leaves_the_pool_working() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for i in 0..4 {
            Scene::new(64, 48, white, Layout::Uniform, 0.3).write(
                dir.path(),
                &format!("p{i}.ARW"),
                MODEL,
            );
        }
        let renderer = Arc::new(PanicsFor("2", Mutex::new(Vec::new())));
        // One decode thread: if the panic killed it, nothing after it would ever be developed.
        let session = ym_core::session::Session::new(
            common::base_env()
                .with_decode_threads(1)
                .with_thumbnails(renderer.clone()),
        );
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        let state = session.state();
        let kinds: Vec<_> = state.photos.iter().map(|p| p.status.clone()).collect();
        use ym_core::session::PhotoStatus;
        assert_eq!(kinds[0], PhotoStatus::Ready);
        assert!(
            matches!(&kinds[1], PhotoStatus::Skipped { reason } if reason.contains("didn't expect")),
            "{:?}",
            kinds[1]
        );
        assert_eq!(kinds[2], PhotoStatus::Ready);
        assert_eq!(kinds[3], PhotoStatus::Ready);
        assert_eq!(session.pending_jobs(), 0);
        // The pool still works afterwards.
        Scene::new(64, 48, white, Layout::Uniform, 0.5).write(dir.path(), "later.ARW", MODEL);
        let again = session.open(&[dir.path().join("later.ARW")], OpenOptions::default());
        settle(&session);
        assert_eq!(session.status(again.added[0]), Some(PhotoStatus::Ready));
    }

    /// Records the look each thumbnail was rendered with.
    #[derive(Default)]
    struct LookRecorder(Mutex<Vec<(PhotoId, LookId, f64)>>);

    impl ThumbnailRenderer for LookRecorder {
        fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
            self.0.lock().unwrap().push((
                input.id,
                input.settings.look.clone(),
                input.settings.look_intensity,
            ));
            Ok(vec![0x89, b'P', b'N', b'G'])
        }
    }

    /// Opens one photo, holds its development job at the first event `at` accepts, applies a look
    /// to the photo there and lets the job go on. Returns what the thumbnail hook was asked to
    /// render.
    ///
    /// The job's thread is held inside the event sink (a sink may be slow: the app's forwards to
    /// the webview), after the photo's settings were first read and before the thumbnail stage
    /// reads them again. The look is therefore applied at a point of the job that is fixed, not at
    /// a time after `open` that a faster or busier machine would put somewhere else. (The test used
    /// to sleep 10 ms and assume the decode and the analysis were still running; on a fast Mac in
    /// a release build they are not, and the thumbnail was already rendered with the old look.)
    fn thumbnails_after_a_look_applied_at(
        at: impl Fn(&PhotoProgress) -> bool + Send + Sync + 'static,
    ) -> Vec<(PhotoId, LookId, f64)> {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        Scene::new(300, 200, white, Layout::Uniform, 0.3).write(dir.path(), "big.ARW", MODEL);
        let recorder = Arc::new(LookRecorder::default());

        let (reached_tx, reached_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (reached_tx, go_rx) = (Mutex::new(reached_tx), Mutex::new(go_rx));
        let held = AtomicBool::new(false);
        let sink = EventSink::new(move |event| {
            if let SessionEvent::PhotoProgress(p) = &event
                && at(p)
                && !held.swap(true, Ordering::SeqCst)
            {
                reached_tx.lock().unwrap().send(()).unwrap();
                // Only the test thread's `apply_look` has to happen now; it takes the state lock,
                // which the job does not hold while it sends an event. The timeout keeps a failing
                // test from hanging.
                let _ = go_rx.lock().unwrap().recv_timeout(Duration::from_secs(30));
            }
        });
        let session = ym_core::session::Session::new(
            common::base_env()
                .with_events(sink)
                .with_thumbnails(recorder.clone()),
        );
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        reached_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the photo's job reached the point where the look is applied");
        session
            .apply_look(
                ym_core::session::ApplyScope::All,
                &LookId::builtin("signature"),
                80.0,
            )
            .unwrap();
        go_tx.send(()).unwrap();
        settle(&session);
        recorder.0.lock().unwrap().clone()
    }

    #[test]
    fn a_thumbnail_is_rendered_with_the_look_the_photo_has_by_then() {
        // The look is applied once the photo's estimates are in, the moment before its thumbnail
        // is started.
        let seen = thumbnails_after_a_look_applied_at(|p| {
            p.stage == PhotoStage::Estimate && p.done && p.error.is_none()
        });
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].1,
            LookId::builtin("signature"),
            "the thumbnail shows the old look"
        );
        assert_eq!(seen[0].2, 80.0);
    }

    #[test]
    fn a_look_applied_before_the_photo_is_decoded_is_in_its_thumbnail() {
        // The job took its copy of the settings when it was started; the look comes after that.
        let seen = thumbnails_after_a_look_applied_at(|p| p.stage == PhotoStage::Decode && !p.done);
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].1,
            LookId::builtin("signature"),
            "the thumbnail shows the look the photo was opened with"
        );
        assert_eq!(seen[0].2, 80.0);
    }
}

mod reading_settings {
    use ym_core::settings::{DevelopSettings, SettingsError, SettingsField};
    use ym_looks::library::LookId;

    #[test]
    fn a_settings_file_without_a_look_gets_the_users_default_look() {
        let signature = LookId::builtin("signature");
        let (s, clamped) =
            DevelopSettings::from_json(r#"{ "contrast": 20, "saturation": 250 }"#, &signature)
                .unwrap();
        assert_eq!(s.look, signature);
        assert_eq!(
            s.contrast,
            ym_core::settings::PercentSetting::manual_value(20.0)
        );
        assert_eq!(
            s.saturation,
            ym_core::settings::PercentSetting::manual_value(100.0)
        );
        assert_eq!(clamped, vec![SettingsField::Saturation]);
        // A look the document names wins, and so does an empty document's default.
        let (s, _) =
            DevelopSettings::from_json(r#"{ "look": "builtin:neutral" }"#, &signature).unwrap();
        assert_eq!(s.look, LookId::builtin("neutral"));
        let (s, _) = DevelopSettings::from_json("{}", &signature).unwrap();
        assert_eq!(s, DevelopSettings::with_look(signature.clone()));
        // Typos and broken JSON are errors in plain words.
        let err = DevelopSettings::from_json(r#"{ "exposur": { "kind": "auto" } }"#, &signature)
            .unwrap_err();
        assert!(
            matches!(err, SettingsError::Parse(ref m) if m.contains("exposur")),
            "{err}"
        );
        assert!(matches!(
            DevelopSettings::from_json("{ nope", &signature),
            Err(SettingsError::Parse(_))
        ));
        assert!(matches!(
            DevelopSettings::from_json(r#"{ "contrast": "high" }"#, &signature),
            Err(SettingsError::Parse(_))
        ));
    }
}

mod stress {
    use super::*;
    use common::{session_with_events, settle};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use ym_core::session::{ApplyScope, OpenOptions, Session};
    use ym_core::settings::SettingsPatch;
    use ym_looks::library::LookId;
    use ym_raw::testgen::SplitMix64;

    /// Every kind of command from several threads at once, including closing and reopening the
    /// session under them. Nothing may deadlock or panic, errors are the plain "gone / not ready"
    /// kind, and the session works afterwards.
    #[test]
    fn commands_from_many_threads_never_deadlock_or_corrupt_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for i in 0..8 {
            Scene::new(
                160,
                120,
                white,
                Layout::Patchwork,
                0.2 + 0.05 * f64::from(i),
            )
            .noise(2e-4, 1e-6)
            .seed(i as u64)
            .write(dir.path(), &format!("s{i}.ARW"), MODEL);
        }
        let (session, rx) = session_with_events();
        let session = Arc::new(session);
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        let deadline = Instant::now() + Duration::from_millis(1500);
        let handles: Vec<_> = (0..6u64)
            .map(|t| {
                let session = Arc::clone(&session);
                let folder = dir.path().to_path_buf();
                std::thread::spawn(move || {
                    let mut rng = SplitMix64(t + 1);
                    let mut ops = 0u32;
                    while Instant::now() < deadline {
                        let ids = session.photo_ids();
                        let id = ids
                            .get(rng.below(ids.len().max(1) as u64) as usize)
                            .copied();
                        let Some(id) = id else {
                            session.open(std::slice::from_ref(&folder), OpenOptions::default());
                            continue;
                        };
                        let v = (rng.below(200) as f64 - 100.0).clamp(-100.0, 100.0);
                        match rng.below(12) {
                            0 => {
                                let _ = session.update_settings(
                                    id,
                                    &SettingsPatch {
                                        contrast: Some(
                                            ym_core::settings::PercentSetting::manual_value(v),
                                        ),
                                        ..SettingsPatch::default()
                                    },
                                );
                            }
                            1 => {
                                let _ = session
                                    .update_settings(id, &SettingsPatch::kelvin(3000.0 + v * 20.0));
                            }
                            2 => {
                                let _ = session.update_settings(
                                    id,
                                    &SettingsPatch {
                                        white_pick: Some(Some([0.1 + 0.004 * (v + 100.0), 0.3])),
                                        ..SettingsPatch::default()
                                    },
                                );
                            }
                            3 => {
                                let _ = session.rerun_auto(id);
                            }
                            4 => {
                                let _ = session.set_current(id);
                            }
                            5 => {
                                let _ = session.apply_look(
                                    ApplyScope::All,
                                    &LookId::builtin("neutral"),
                                    v.abs(),
                                );
                            }
                            6 => {
                                let _ = session.settings(id);
                                let _ = session.state();
                            }
                            7 => {
                                let _ = session.set_picked(id, v > 0.0);
                                session.pick_all();
                            }
                            8 => {
                                let _ = session.raw_image(id);
                            }
                            9 if t == 0 && rng.below(40) == 0 => {
                                session.close();
                                session.open(std::slice::from_ref(&folder), OpenOptions::default());
                            }
                            10 => {
                                let _ = session.view(id).map(|view| view.resolved());
                            }
                            _ => {
                                let _ = session.scope_ids(ApplyScope::Picked);
                            }
                        }
                        ops += 1;
                    }
                    ops
                })
            })
            .collect();
        let total: u32 = handles
            .into_iter()
            .map(|h| h.join().expect("no thread panicked"))
            .sum();
        assert!(total > 100, "only {total} commands ran");
        // Events keep their shape: drain them without a panic.
        let _ = rx.try_iter().count();
        // Whatever state the session is in, it settles and every photo is Ready or Skipped.
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        for p in session.state().photos {
            assert!(
                matches!(
                    p.status,
                    ym_core::session::PhotoStatus::Ready
                        | ym_core::session::PhotoStatus::Skipped { .. }
                ),
                "{:?}",
                p.status
            );
        }
        assert!(Arc::strong_count(&session) == 1);
        let _: &Session = &session;
    }
}

mod other_layouts {
    use super::*;
    use common::{session_with_events, settle};
    use ym_core::session::OpenOptions;
    use ym_core::settings::SettingsPatch;
    use ym_raw::layout::ChromaSubsampling;
    use ym_raw::testgen::{ArwBuilder, CrawEncoder, LosslessMs};

    #[test]
    fn a_craw_file_with_an_odd_crop_has_a_neutral_preview_and_a_noise_model() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 4300.0);
        let scene = Scene::new(640, 480, white, Layout::Ramp, 0.5).noise(3e-4, 2e-6);
        let path = dir.path().join("craw.ARW");
        scene
            .builder(MODEL)
            .active_area(632, 472)
            .crop(Rect {
                x: 5,
                y: 3,
                width: 600,
                height: 440,
            })
            .craw(CrawEncoder::new())
            .write_to(&path, &scene.mosaic())
            .unwrap();
        let (raw, est, _) = estimates_of_file(&path);
        assert!(raw.qstep.is_some(), "cRAW carries its quantisation map");
        let preview = build_preview(&raw).unwrap();
        let peak = white.iter().copied().fold(0.0, f64::max);
        // The middle of the ramp (the block columns away from the edges) is neutral after the
        // camera white: all three channels agree with the scene to the cRAW rounding.
        let mut worst = 0.0f64;
        for by in 4..preview.height - 4 {
            for bx in 4..preview.width - 4 {
                let p = &preview.rgb[3 * (by * preview.width + bx)..][..3];
                let x = 5.0 + (bx as f64 + 0.5) * 600.0 / preview.width as f64;
                let lum = 0.5 * (0.1 + 0.9 * x / 639.0);
                for c in 0..3 {
                    let want = 0.9 * lum * white[c] / peak;
                    worst = worst.max((f64::from(p[c]) - want).abs());
                }
            }
        }
        // Shot noise of a block of 16 to 32 photosites (0.3 * 0.45 -> sigma 0.012 / 4) plus the
        // cRAW rounding: well under a percent.
        assert!(worst < 0.012, "preview colours are off by {worst}");
        assert!(est.noise.model.a > 0.0 && est.noise.model.a.is_finite());
    }

    #[test]
    fn pick_white_and_rerun_work_on_a_camera_processed_file() {
        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (128u32, 96u32);
        let white = light(PROFILE, 5200.0);
        let mut rgb = Vec::new();
        for _ in 0..w * h {
            for c in 0..3 {
                rgb.push((1024.0 + 0.4 * white[c] / white[1] * 14000.0).round() as u16);
            }
        }
        ArwBuilder::new(MODEL, w, h)
            .exposure_comp(0, 10)
            .lossless_ms(LosslessMs::new(
                ChromaSubsampling::Yuv420,
                rgb,
                w,
                h,
                [1.0 / white[0], 1.0, 1.0 / white[2]].map(|g| g as f32),
            ))
            .write_to(&dir.path().join("ms.ARW"), &[])
            .unwrap();
        let (session, _rx) = session_with_events();
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        let id = session.photo_ids()[0];
        assert!(session.summary(id).unwrap().reduced);
        // The file's white balance was divided out again, so red and blue top out lower than green
        // (docs/04). The data keeps the raw scale of a mosaic: a neutral grey of 0.4 of the green
        // range balanced by the initial gains is 0.4 * 14000 / 16512 = 0.339 in every channel. The
        // exposure is the camera's gain whatever the brightness (docs/02, interim rule of 2026-10-14);
        // the key statistic that is still reported meters middle grey / 0.339 (it used to meter the
        // camera's own balance on top of its own, and the colour estimate saw a strong cast).
        let estimates = session.estimates(id).unwrap();
        assert_eq!(estimates.summary().exposure_ev, CAMERA_GAIN_EV);
        assert_eq!(estimates.exposure.own_ev, CAMERA_GAIN_EV);
        let key = (0.18 / estimates.exposure.meter.mean_luminance).log2();
        let want = (0.18f64 / (0.4 * 14000.0 / 16512.0)).log2();
        assert!((key - want).abs() < 0.1, "key statistic {key}, want {want}");
        let picked = session
            .update_settings(
                id,
                &SettingsPatch {
                    white_pick: Some(Some([0.5, 0.5])),
                    ..SettingsPatch::default()
                },
            )
            .expect("a grey spot of a camera-processed file can be picked");
        // The scene is neutral under a 5200 K light: the picked light reads about that.
        assert!(
            (picked.values.kelvin - 5200.0).abs() < 100.0,
            "{} K",
            picked.values.kelvin
        );
        let again = session.rerun_auto(id).unwrap();
        assert!(again.settings.white_pick.is_none());
    }
}

mod concurrent_open {
    use super::*;
    use common::{session_with_events, settle};
    use std::sync::{Arc, Barrier};
    use ym_core::session::OpenOptions;

    #[test]
    fn the_same_folder_opened_twice_at_once_adds_each_photo_once() {
        let dir = tempfile::tempdir().unwrap();
        let white = light(PROFILE, 5200.0);
        for i in 0..40 {
            Scene::new(32, 32, white, Layout::Uniform, 0.3).write(
                dir.path(),
                &format!("c{i:02}.ARW"),
                MODEL,
            );
        }
        let (session, _rx) = session_with_events();
        let session = Arc::new(session);
        let start = Arc::new(Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let (session, start) = (Arc::clone(&session), Arc::clone(&start));
                let folder = dir.path().to_path_buf();
                std::thread::spawn(move || {
                    start.wait();
                    session.open(&[folder], OpenOptions::default())
                })
            })
            .collect();
        let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        settle(&session);
        assert_eq!(session.state().photos.len(), 40, "every file once");
        let added: usize = outcomes.iter().map(|o| o.added.len()).sum();
        let already: usize = outcomes.iter().map(|o| o.already_open).sum();
        assert_eq!(added, 40);
        assert_eq!(already, 40, "the other call saw each file as already open");
    }
}
