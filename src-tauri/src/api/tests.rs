//! The command handlers against `ym-core`, with synthetic ARW files and no webview
//! (`docs/01` §IPC). The harness is in `crate::testing`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::json;

use super::*;
use crate::testing::{Harness, WAIT, jpeg_job, png_size, wait_until};
use crate::wire::*;

// --- Session ----------------------------------------------------------------------------------

#[test]
fn opening_a_folder_lists_the_photos_and_develops_them_with_events() {
    let h = Harness::new();
    std::fs::write(h.photos.path().join("notes.txt"), b"not a raw file").expect("write");
    let opened = h.open(3);
    assert_eq!(opened.photos.len(), 3, "the text file is ignored");
    assert_eq!(opened.current_index, Some(0));
    assert!(opened.folder_name.is_some());
    assert!(
        opened
            .default_export_folder
            .as_deref()
            .is_some_and(|f| f.ends_with(" export"))
    );
    let first = &opened.photos[0];
    assert_eq!(first.file_name, "DSC00001.ARW");
    assert_eq!(first.camera.profile_id, "sony-ilce-7rm4");
    assert_eq!(first.camera.match_kind, MatchKind::Exact);
    assert_eq!(
        (first.iso, first.width_px, first.height_px),
        (400, 192, 128)
    );
    assert!((first.shutter_s.0 - 0.008).abs() < 1e-9 && (first.aperture_f.0 - 2.8).abs() < 1e-9);

    let done = h.develop();
    assert!(done.photos.iter().all(|p| !p.reduced && !p.picked));

    // photo_progress arrives per photo, ending with the thumbnail stage done; thumbnail_ready
    // carries a PNG of at most 240 px.
    wait_until("three thumbnails", || {
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::ThumbnailReady(_)))
            .count()
            >= 3
    });
    let events = h.events.events();
    for photo in &done.photos {
        let progress: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AppEvent::PhotoProgress(p) if p.id == photo.id => Some(p.clone()),
                _ => None,
            })
            .collect();
        assert!(
            progress
                .iter()
                .any(|p| p.stage == PhotoStage::Thumbnail && p.done && p.error.is_none()),
            "{progress:?}"
        );
    }
    for e in &events {
        if let AppEvent::ThumbnailReady(t) = e {
            let (w, hh) = png_size(&t.png_bytes);
            assert!(w.max(hh) <= 240 && w.max(hh) > 0, "{w} x {hh}");
        }
    }
}

#[test]
fn a_drop_without_arw_files_says_so_and_a_missing_file_is_named() {
    let h = Harness::new();
    let txt = h.photos.path().join("notes.txt");
    std::fs::write(&txt, b"x").expect("write");
    let err = h
        .api
        .open_paths(&[txt.to_string_lossy().into_owned()])
        .expect_err("no ARW");
    assert_eq!(err.scope, ErrorScope::Open);
    assert_eq!(err.message, "No ARW files in what you dropped.");
    let err = h
        .api
        .open_paths(&[h
            .photos
            .path()
            .join("gone.ARW")
            .to_string_lossy()
            .into_owned()])
        .expect_err("missing");
    assert!(err.message.contains("isn't there"), "{err}");
    // An empty session is still a session.
    assert!(h.api.session_state().expect("state").photos.is_empty());
}

#[test]
fn navigation_picking_and_closing() {
    let h = Harness::new();
    h.open(3);
    let state = h.develop();
    let ids: Vec<_> = state.photos.iter().map(|p| p.id).collect();
    h.api.set_current(ids[2]).expect("current");
    h.api.set_picked(ids[0], true).expect("pick");
    h.api.set_picked(ids[2], true).expect("pick");
    let state = h.api.session_state().expect("state");
    assert_eq!(state.current_index, Some(2));
    assert_eq!(state.picked, vec![ids[0], ids[2]]);
    assert!(state.photos[0].picked && !state.photos[1].picked);
    let err = h.api.set_current(PhotoId(9_999)).expect_err("unknown");
    assert!(err.message.contains("isn't in the session"), "{err}");

    h.api.close_session();
    let state = h.api.session_state().expect("state");
    assert!(state.photos.is_empty() && state.current_index.is_none());
    // A photo of the closed session is gone for good, also for the commands that name it.
    assert!(h.api.get_settings(ids[0]).is_err());
}

// --- Settings ---------------------------------------------------------------------------------

#[test]
fn settings_round_trip_with_the_json_the_frontend_sends() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let id = h.first();
    let before = h.api.get_settings(id).expect("settings");
    assert!(before.auto.exposure && before.auto.white_balance && before.auto.sharpen);
    assert!(matches!(before.settings.exposure, ExposureSetting::Auto));

    let patch: SettingsPatch = serde_json::from_value(json!({
        "exposure": { "kind": "manual", "ev": 0.5 },
        "contrast": 20,
        "color_balance": { "shadows": [0.2, -0.1] },
        "curve": { "luma": [[0, 0], [0.5, 0.6], [1, 1]] },
        "crop": {
            "rect": { "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5 },
            "angle_deg": 2.5,
            "aspect": "free"
        },
        "orientation": 90,
        "white_pick": [0.5, 0.5]
    }))
    .expect("a patch as the frontend writes it");
    let after = h.api.update_settings(id, &patch).expect("update");
    assert!((after.values.exposure_ev.0 - 0.5).abs() < 1e-9);
    assert!(!after.auto.exposure);
    assert!((after.settings.contrast.0 - 20.0).abs() < 1e-9);
    assert_eq!(after.settings.color_balance.shadows, [Num(0.2), Num(-0.1)]);
    assert_eq!(after.settings.color_balance.midtones, [Num(0.0), Num(0.0)]);
    assert_eq!(after.settings.curve.luma.len(), 3);
    assert_eq!(after.settings.orientation, crate::wire::Orientation(90));
    assert!(after.settings.crop.is_some());
    assert!(after.settings.white_pick.is_some());
    // The same answer comes back from get_settings.
    assert_eq!(h.api.get_settings(id).expect("settings"), after);

    // `null` clears the crop and the pick; Auto resets the exposure; nothing else changes.
    let clear: SettingsPatch = serde_json::from_value(json!({
        "exposure": { "kind": "auto" },
        "crop": null,
        "white_pick": null
    }))
    .expect("clearing patch");
    let cleared = h.api.update_settings(id, &clear).expect("clear");
    assert!(cleared.auto.exposure);
    assert!(cleared.settings.crop.is_none() && cleared.settings.white_pick.is_none());
    assert!((cleared.settings.contrast.0 - 20.0).abs() < 1e-9);

    // A patch without the key leaves the crop alone.
    let set: SettingsPatch = serde_json::from_value(json!({
        "crop": { "rect": { "x": 0, "y": 0, "width": 1, "height": 0.5 }, "angle_deg": 0, "aspect": "free" }
    }))
    .expect("crop patch");
    h.api.update_settings(id, &set).expect("crop");
    let leave: SettingsPatch = serde_json::from_value(json!({ "saturation": 10 })).expect("patch");
    assert!(
        h.api
            .update_settings(id, &leave)
            .expect("update")
            .settings
            .crop
            .is_some()
    );
}

#[test]
fn settings_errors_are_plain_sentences_and_change_nothing() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let id = h.first();
    // A curve with one point is not a curve.
    let bad: SettingsPatch =
        serde_json::from_value(json!({ "contrast": 5, "curve": { "luma": [[0.5, 0.5]] } }))
            .expect("parses");
    let err = h.api.update_settings(id, &bad).expect_err("refused");
    assert_eq!(err.scope, ErrorScope::Render);
    assert!(!err.message.is_empty());
    assert!(
        (h.api
            .get_settings(id)
            .expect("settings")
            .settings
            .contrast
            .0)
            .abs()
            < 1e-9
    );
    // A misspelt field is an error, not a patch that silently does nothing (`docs/01`
    // DevelopSettings), also inside the nested objects and in the preferences.
    for typo in [
        json!({ "constrast": 1 }),
        json!({ "decode": { "neural_demosiac": false } }),
        json!({ "rolloff": { "highlight": 0.2 } }),
        json!({ "color_balance": { "shadow": [0.1, 0.1] } }),
        json!({ "curve": { "lumna": [[0, 0], [1, 1]] } }),
    ] {
        assert!(
            serde_json::from_value::<SettingsPatch>(typo.clone()).is_err(),
            "{typo} was accepted"
        );
    }
    assert!(
        serde_json::from_value::<PreferencesPatch>(json!({ "remove_location": true })).is_err()
    );
    // The same patch spelt right is fine.
    assert!(serde_json::from_value::<SettingsPatch>(json!({ "contrast": 1 })).is_ok());
}

#[test]
fn apply_look_reaches_this_picked_and_all_photos() {
    let h = Harness::new();
    h.open(3);
    let state = h.develop();
    let ids: Vec<_> = state.photos.iter().map(|p| p.id).collect();
    let neutral = LookId("builtin:neutral".to_owned());
    h.api
        .apply_look(ApplyScope::This, &neutral, 40.0)
        .expect("this");
    let intensity = |id| {
        h.api
            .get_settings(id)
            .expect("settings")
            .values
            .look_intensity
            .0
    };
    assert!((intensity(ids[0]) - 40.0).abs() < 1e-9);
    assert!((intensity(ids[1]) - 100.0).abs() < 1e-9);
    h.api.set_picked(ids[1], true).expect("pick");
    h.api
        .apply_look(ApplyScope::Picked, &neutral, 70.0)
        .expect("picked");
    assert!((intensity(ids[1]) - 70.0).abs() < 1e-9 && (intensity(ids[2]) - 100.0).abs() < 1e-9);
    h.api
        .apply_look(ApplyScope::All, &neutral, 55.0)
        .expect("all");
    assert!(ids.iter().all(|&id| (intensity(id) - 55.0).abs() < 1e-9));

    let err = h
        .api
        .apply_look(
            ApplyScope::All,
            &LookId("arri:3000:nothing".to_owned()),
            100.0,
        )
        .expect_err("no such look");
    assert_eq!(err.scope, ErrorScope::Looks);
    let err = h
        .api
        .apply_look(ApplyScope::All, &LookId("not a look id".to_owned()), 100.0)
        .expect_err("bad id");
    assert!(err.message.contains("isn't in the library"), "{err}");
}

#[test]
fn rerun_histogram_and_look_thumbnails() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let id = h.first();
    let patch: SettingsPatch =
        serde_json::from_value(json!({ "exposure": { "kind": "manual", "ev": 1.5 } }))
            .expect("patch");
    h.api.update_settings(id, &patch).expect("update");
    let rerun = h.api.rerun_auto(id).expect("rerun");
    assert!(rerun.auto.exposure, "Re-run sets exposure back to Auto");

    let histogram = h.api.histogram(id).expect("histogram");
    for channel in [&histogram.red, &histogram.green, &histogram.blue] {
        assert_eq!(channel.len(), 64);
    }
    let peak = histogram
        .red
        .iter()
        .chain(&histogram.green)
        .chain(&histogram.blue)
        .map(|v| v.0)
        .fold(0.0, f64::max);
    assert!(
        (peak - 1.0).abs() < 1e-9,
        "the tallest bin is 1, not {peak}"
    );

    let png = h
        .api
        .look_thumbnail(id, &LookId("builtin:neutral".to_owned()))
        .expect("look tile");
    let (w, hh) = png_size(&png);
    assert!(w.max(hh) <= 240);
    let err = h
        .api
        .look_thumbnail(id, &LookId("arri:3000:film-warm".to_owned()))
        .expect_err("not imported");
    assert_eq!(err.scope, ErrorScope::Looks);
    assert!(!err.message.is_empty());
}

// --- Viewport ---------------------------------------------------------------------------------

#[test]
fn the_fallback_viewport_sends_frames_for_the_cell_and_follows_the_input() {
    let h = Harness::new();
    assert_eq!(h.api.viewport_info().backend, ViewportBackendKind::Fallback);
    h.open(1);
    h.develop();
    // 100 x 100 device pixels: the photo is 192 x 128, so zooming to 100 % is a real zoom.
    h.rect(100.0, 100.0);
    let first = h.frame_of_size(100, 100);
    assert!(
        first.url.ends_with(&format!("/frame-{}.png", first.seq)),
        "{}",
        first.url
    );
    let png = h.frames.get(first.seq).expect("the frame is served");
    assert_eq!(png_size(&png), (100, 100));
    // The status text follows every frame.
    let stats = h
        .events
        .wait_for(Duration::from_secs(5), |e| match e {
            AppEvent::RenderStats(s) => Some(*s),
            _ => None,
        })
        .expect("render_stats");
    assert_eq!(stats.path, crate::wire::RenderPath::Classical);
    assert_eq!(stats.space, crate::wire::ColorSpace::DisplayP3);

    // Before shows another picture.
    assert!(h.api.viewport().wait_idle(WAIT));
    let n = h.api.viewport().frames_presented();
    h.api.set_before(true);
    let before = h.frame_after(n);
    assert_ne!(
        h.frames.get(before.seq).expect("frame"),
        png,
        "the Before view differs from the finished photo"
    );

    // A double click zooms to 100 %, a pan moves the picture, a mode change goes back to Fit.
    assert!(h.api.viewport().wait_idle(WAIT));
    let n = h.api.viewport().frames_presented();
    h.api
        .viewport_input(&ViewportInput::ToggleZoom {
            x: Num(50.0),
            y: Num(50.0),
        })
        .expect("toggle");
    let zoomed = h.frame_after(n);
    assert_eq!((zoomed.width, zoomed.height), (100, 100));
    assert!(h.api.viewport().wait_idle(WAIT));
    let (_, state) = h.api.viewport().snapshot_synced();
    assert!(
        matches!(state.mode, ym_core::pipeline::ViewportMode::Percent { value } if (value - 100.0).abs() < 1e-9)
    );
    let n = h.api.viewport().frames_presented();
    h.api
        .viewport_input(&ViewportInput::Pan {
            dx: Num(-20.0),
            dy: Num(0.0),
        })
        .expect("pan");
    h.frame_after(n);
    assert!(h.api.viewport().wait_idle(WAIT));
    let n = h.api.viewport().frames_presented();
    h.api.viewport_set_mode(&ViewportMode::Fit).expect("fit");
    h.frame_after(n);
    assert!(h.api.viewport().wait_idle(WAIT));
    assert!(matches!(
        h.api.viewport().snapshot_synced().1.mode,
        ym_core::pipeline::ViewportMode::Fit
    ));
}

#[test]
fn a_burst_of_input_is_coalesced_and_never_stalls_the_viewport() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    h.rect(100.0, 100.0);
    h.frame_of_size(100, 100);
    assert!(h.api.viewport().wait_idle(WAIT));
    h.api
        .viewport_input(&ViewportInput::ToggleZoom {
            x: Num(50.0),
            y: Num(50.0),
        })
        .expect("zoom in");
    assert!(h.api.viewport().wait_idle(WAIT));
    let before = h.api.viewport().frames_presented();

    // Two threads drag and wheel as fast as they can, with a resize in the middle.
    let api = Arc::clone(&h.api);
    let wheel = std::thread::spawn(move || {
        for i in 0..150 {
            let factor = if i % 2 == 0 { 1.05 } else { 0.95 };
            api.viewport_input(&ViewportInput::Zoom {
                factor: Num(factor),
                x: Num(40.0),
                y: Num(60.0),
            })
            .expect("zoom");
        }
    });
    for i in 0..300 {
        let dx = if i % 2 == 0 { 3.0 } else { -2.0 };
        h.api
            .viewport_input(&ViewportInput::Pan {
                dx: Num(dx),
                dy: Num(1.0),
            })
            .expect("pan");
        if i == 150 {
            h.rect(120.0, 90.0);
        }
    }
    wheel.join().expect("the wheel thread");

    assert!(
        h.api.viewport().wait_idle(WAIT),
        "the viewport settles after a burst"
    );
    let shown = h.api.viewport().frames_presented() - before;
    assert!(shown >= 1, "the last input is shown");
    assert!(
        shown < 450,
        "{shown} frames for 450 inputs: the pans were not coalesced"
    );
    // The picture that stays is the one for the last cell size.
    let last = h
        .events
        .events()
        .into_iter()
        .rev()
        .find_map(|e| match e {
            AppEvent::ViewportFrame(f) => Some(f),
            _ => None,
        })
        .expect("a frame");
    assert_eq!((last.width, last.height), (120, 90));
}

#[test]
fn choosing_photos_while_everything_else_changes_never_stalls_the_viewport() {
    let h = Harness::new();
    h.open(5);
    let state = h.develop();
    let ids: Vec<PhotoId> = state.photos.iter().map(|p| p.id).collect();
    h.rect(100.0, 70.0);
    h.frame_of_size(100, 70);

    // Three threads do what a busy person and a busy webview do at once.
    let workers: Vec<_> = (0..3_usize)
        .map(|t| {
            let api = Arc::clone(&h.api);
            let ids = ids.clone();
            std::thread::spawn(move || {
                for i in 0..80_usize {
                    let photo = ids[(i * 7 + t) % ids.len()];
                    match (i + t) % 7 {
                        0 => api.set_current(photo).expect("current"),
                        1 => api
                            .viewport_input(&ViewportInput::Zoom {
                                factor: Num(1.1),
                                x: Num(30.0),
                                y: Num(20.0),
                            })
                            .expect("zoom"),
                        2 => api.set_before(i % 2 == 0),
                        3 => {
                            let patch: SettingsPatch =
                                serde_json::from_value(json!({ "contrast": (i % 40) as i32 - 20 }))
                                    .expect("patch");
                            api.update_settings(photo, &patch).expect("update");
                        }
                        4 => api
                            .viewport_set_mode(&ViewportMode::Percent {
                                value: Num(if i % 2 == 0 { 100.0 } else { 50.0 }),
                            })
                            .expect("mode"),
                        5 => api
                            .viewport_set_rect(&ViewportRect {
                                x: Num(0.0),
                                y: Num(0.0),
                                width: Num(90.0 + (i % 5) as f64 * 10.0),
                                height: Num(60.0),
                                scale: Num(1.0),
                            })
                            .expect("rect"),
                        _ => {
                            // The readout and the pick race with all of it; either answer is fine.
                            let _ = api.sample_pixel(40.0, 30.0);
                            let _ = api.histogram(photo);
                        }
                    }
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("a worker thread");
    }
    assert!(
        h.api.viewport().wait_idle(WAIT),
        "the viewport settles once the input stops"
    );
    // The cell shows a picture of one of the sizes reported, and the session still answers.
    let last = h
        .events
        .events()
        .into_iter()
        .rev()
        .find_map(|e| match e {
            AppEvent::ViewportFrame(f) if !f.url.is_empty() => Some(f),
            _ => None,
        })
        .expect("a frame");
    assert_eq!(last.height, 60);
    assert!(h.api.session_state().is_ok());
    h.api.close_session();
    assert!(h.api.viewport().wait_idle(WAIT));
}

#[test]
fn the_pixel_readout_finds_the_photo_and_not_the_paper() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    // A square cell: a 3:2 photo leaves paper above and below.
    h.rect(300.0, 300.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    let inside = h
        .api
        .sample_pixel(150.0, 150.0)
        .expect("a pixel of the photo");
    assert!(u16::from(inside.r) + u16::from(inside.g) + u16::from(inside.b) > 0);
    let err = h.api.sample_pixel(150.0, 5.0).expect_err("paper");
    assert_eq!(err.scope, ErrorScope::Viewport);
    assert!(h.api.sample_pixel(f64::NAN, 3.0).is_err());
}

#[test]
fn a_pick_white_click_becomes_the_photos_white_pick() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let id = h.first();
    h.rect(300.0, 300.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    // On the paper: nothing happens.
    h.api
        .viewport_input(&ViewportInput::PickWhite {
            x: Num(150.0),
            y: Num(4.0),
        })
        .expect("a click on the paper is ignored");
    assert!(
        h.api
            .get_settings(id)
            .expect("settings")
            .settings
            .white_pick
            .is_none()
    );

    h.api
        .viewport_input(&ViewportInput::PickWhite {
            x: Num(150.0),
            y: Num(150.0),
        })
        .expect("a click on the photo");
    let settings = h.api.get_settings(id).expect("settings");
    let [x, y] = settings.settings.white_pick.expect("the pick is set");
    assert!(
        (x.0 - 0.5).abs() < 0.05 && (y.0 - 0.5).abs() < 0.05,
        "{x:?} {y:?}"
    );
    assert!(
        h.events.events().contains(&AppEvent::SettingsChanged(id)),
        "settings_changed tells the webview to ask again"
    );
}

/// Writes a file that is named like a raw file and is not one.
fn write_broken_arw(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, b"this is not a raw file at all").expect("write");
    path
}

#[test]
fn a_photo_that_cannot_be_opened_leaves_no_picture_behind() {
    let h = Harness::new();
    crate::testing::write_arw(h.photos.path(), "DSC00001.ARW", 1);
    write_broken_arw(h.photos.path(), "DSC00002.ARW");
    let opened = h
        .api
        .open_paths(&[h.photos.path().to_string_lossy().into_owned()])
        .expect("open");
    assert_eq!(opened.photos.len(), 2);
    wait_until("the first photo and the verdict on the second", || {
        h.api.session_state().is_ok_and(|s| {
            s.photos[0].status == PhotoStatus::Ready
                && matches!(s.photos[1].status, PhotoStatus::Skipped { .. })
        })
    });
    h.rect(100.0, 100.0);
    let shown = h.frame_of_size(100, 100);
    assert!(!shown.url.is_empty());

    // The broken file is chosen: the picture of the other photo must not stay in the cell.
    h.api
        .set_current(opened.photos[1].id)
        .expect("a photo that can't be developed can still be selected");
    let gone = h
        .events
        .wait_for(Duration::from_secs(20), |e| match e {
            AppEvent::ViewportFrame(f) if f.url.is_empty() => Some(f.clone()),
            _ => None,
        })
        .expect("the cell is told there is no picture");
    assert!(gone.seq > shown.seq);
    assert_eq!((gone.width, gone.height), (0, 0));

    // And the good photo draws again when it is chosen.
    let n = h.api.viewport().frames_presented();
    h.api.set_current(opened.photos[0].id).expect("back");
    let again = h.frame_after(n);
    assert!(!again.url.is_empty() && again.seq > gone.seq);

    // Closing the session takes the picture away too.
    h.api.close_session();
    wait_until("the second clear", || {
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::ViewportFrame(f) if f.url.is_empty()))
            .count()
            >= 2
    });
}

#[test]
fn a_cell_reported_while_the_photo_develops_is_drawn_when_it_is_ready() {
    let h = Harness::new();
    // The cell is reported at once, before the photo is developed, as the webview does when the
    // photo screen mounts.
    h.open(1);
    h.rect(90.0, 60.0);
    let frame = h.frame_of_size(90, 60);
    assert!(!frame.url.is_empty());
    h.develop();
    assert!(h.api.viewport().wait_idle(WAIT));
}

#[test]
fn a_new_photo_goes_back_to_fit() {
    let h = Harness::new();
    h.open(2);
    let state = h.develop();
    h.rect(96.0, 64.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    h.api
        .viewport_input(&ViewportInput::ToggleZoom {
            x: Num(40.0),
            y: Num(30.0),
        })
        .expect("zoom");
    assert!(h.api.viewport().wait_idle(WAIT));
    assert!(!matches!(
        h.api.viewport().snapshot_synced().1.mode,
        ym_core::pipeline::ViewportMode::Fit
    ));
    h.api.set_current(state.photos[1].id).expect("next");
    assert!(h.api.viewport().wait_idle(WAIT));
    assert!(matches!(
        h.api.viewport().snapshot_synced().1.mode,
        ym_core::pipeline::ViewportMode::Fit
    ));
}

// --- Thumbnails -------------------------------------------------------------------------------

#[test]
fn the_filmstrip_thumbnail_follows_a_look_change() {
    let h = Harness::new();
    h.open(2);
    let state = h.develop();
    wait_until("the first thumbnails", || {
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::ThumbnailReady(_)))
            .count()
            >= 2
    });
    let count_for = |id: PhotoId| {
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::ThumbnailReady(t) if t.id == id))
            .count()
    };
    let (a, b) = (state.photos[0].id, state.photos[1].id);
    let (a0, b0) = (count_for(a), count_for(b));
    h.api
        .apply_look(
            ApplyScope::This,
            &LookId("builtin:neutral".to_owned()),
            30.0,
        )
        .expect("look");
    wait_until("a fresh thumbnail", || count_for(a) > a0);
    assert!(h.api.thumbnails().wait_idle(WAIT));
    assert_eq!(count_for(b), b0, "the other photo keeps its thumbnail");

    // A crop shows in the thumbnail, so it refreshes it; an exposure change does not.
    let a1 = count_for(a);
    let crop: SettingsPatch = serde_json::from_value(json!({
        "crop": { "rect": { "x": 0, "y": 0, "width": 0.5, "height": 0.5 }, "angle_deg": 0, "aspect": "free" }
    }))
    .expect("crop");
    h.api.update_settings(a, &crop).expect("crop");
    wait_until("a thumbnail of the crop", || count_for(a) > a1);
    assert!(h.api.thumbnails().wait_idle(WAIT));

    // An exposure change is not asked for again: wait longer than the thumbnails collect requests
    // before looking.
    let a2 = count_for(a);
    let exposure: SettingsPatch =
        serde_json::from_value(json!({ "exposure": { "kind": "manual", "ev": 1.0 } }))
            .expect("exposure");
    h.api.update_settings(a, &exposure).expect("exposure");
    std::thread::sleep(Duration::from_millis(700));
    assert!(h.api.thumbnails().wait_idle(WAIT));
    assert_eq!(
        count_for(a),
        a2,
        "an exposure change does not refresh the thumbnail"
    );
}

// --- Looks ------------------------------------------------------------------------------------

fn write_cube(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let text = ym_color::lut::Lut3d::identity(5)
        .expect("identity cube")
        .to_cube_string(name);
    std::fs::write(&path, text).expect("write the cube");
    path
}

#[test]
fn importing_a_cube_needs_the_answer_and_announces_the_new_look() {
    let h = Harness::new();
    let cube = write_cube(h.photos.path(), "Warm Film.cube");
    let path = cube.to_string_lossy().into_owned();

    let seen = h.api.looks_recognise(&path);
    assert_eq!(seen.source, SourceKind::File);
    assert_eq!(seen.name, "Warm Film.cube");
    assert!(seen.size_bytes.is_some());
    match &seen.recognition {
        Recognition::NeedsAnswer {
            file_count,
            lut_size,
            ..
        } => {
            assert_eq!((*file_count, *lut_size), (1, Some(5)));
        }
        other => panic!("expected a question, got {other:?}"),
    }
    // Without the answer the import is refused with the importer's sentence.
    let err = h.api.looks_import(&path, None).expect_err("needs input");
    assert_eq!(err.scope, ErrorScope::Looks);

    let result = h
        .api
        .looks_import(&path, Some(LookInput::Logc4))
        .expect("import");
    assert_eq!(result.added.len(), 1);
    assert_eq!(result.added[0].group, LookGroup::User);
    assert_eq!(result.added[0].input, LookInput::Logc4);
    wait_until("looks_changed", || {
        h.events.events().contains(&AppEvent::LooksChanged)
    });
    let listed = h.api.looks_list().expect("list");
    assert!(listed.iter().any(|l| l.id == result.added[0].id));
    assert!(
        listed
            .iter()
            .any(|l| l.id.0 == "builtin:neutral" && l.is_default)
    );

    // The same file again changes nothing; the folder it came from is remembered.
    let again = h
        .api
        .looks_import(&path, Some(LookInput::Logc4))
        .expect("again");
    assert!(again.added.is_empty() && again.unchanged == 1);
    let prefs = h.api.preferences_get().expect("prefs");
    assert_eq!(
        prefs.last_used_folders.looks_import.as_deref(),
        h.photos.path().to_str()
    );

    // It can be the default look, and can be removed again.
    h.api
        .looks_set_default(&result.added[0].id)
        .expect("default");
    assert_eq!(
        h.api.preferences_get().expect("prefs").default_look,
        result.added[0].id
    );
    h.api.looks_remove(&result.added[0].id).expect("remove");
    assert_eq!(
        h.api.preferences_get().expect("prefs").default_look.0,
        "builtin:neutral",
        "removing the default look puts Neutral back"
    );
}

#[test]
fn recognition_never_fails_and_explains_what_is_wrong() {
    let h = Harness::new();
    let missing = h
        .api
        .looks_recognise(&h.photos.path().join("Gone.zip").to_string_lossy());
    assert!(
        matches!(missing.recognition, Recognition::Missing { .. }),
        "{missing:?}"
    );
    assert_eq!(missing.source, SourceKind::Zip);
    assert!(missing.size_bytes.is_none());

    // A zip that only holds ARRI Look Files.
    let zip_path = h.photos.path().join("ARRI_LookFiles_ALF4.zip");
    {
        let file = std::fs::File::create(&zip_path).expect("zip");
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file("Looks/Film.alf4", zip::write::SimpleFileOptions::default())
            .expect("entry");
        std::io::Write::write_all(&mut zip, b"not a cube").expect("bytes");
        zip.finish().expect("finish");
    }
    let seen = h.api.looks_recognise(&zip_path.to_string_lossy());
    assert_eq!(seen.source, SourceKind::Zip);
    match seen.recognition {
        Recognition::Unusable { reason, .. } => {
            assert!(reason.to_lowercase().contains("alf4"), "{reason}");
        }
        other => panic!("expected unusable, got {other:?}"),
    }

    // Not a zip at all.
    let junk = h.photos.path().join("Holiday.zip");
    std::fs::write(&junk, b"this is not a zip").expect("write");
    assert!(matches!(
        h.api.looks_recognise(&junk.to_string_lossy()).recognition,
        Recognition::Unusable { .. }
    ));
}

#[test]
fn the_downloads_watcher_reports_a_zip_once_it_stops_growing() {
    let watch = WatchConfig {
        poll_interval: Duration::from_millis(40),
        stable_for: Duration::from_millis(200),
        report_existing: false,
    };
    let h = Harness::with_watch(watch);
    let downloads = h.home.path().join("Downloads");
    std::fs::create_dir_all(&downloads).expect("mkdir");
    let status = h.api.looks_watch_downloads(true);
    assert!(status.watching && status.problem.is_none(), "{status:?}");
    std::fs::write(
        downloads.join("ARRI Look Library LogC4.zip"),
        vec![0u8; 4096],
    )
    .expect("zip");
    std::fs::write(downloads.join("holiday photos.zip"), b"zzzz").expect("other zip");
    let found = h
        .events
        .wait_for(Duration::from_secs(10), |e| match e {
            AppEvent::DownloadFound(d) => Some(d.clone()),
            _ => None,
        })
        .expect("download_found");
    assert_eq!(found.file_name, "ARRI Look Library LogC4.zip");
    assert_eq!(found.package_hint, Some(PackageId::LookLibraryLogc4));
    assert!((found.size_bytes.0 - 4096.0).abs() < 1e-9);
    assert!(
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::DownloadFound(_)))
            .count()
            == 1,
        "only the zip that matches a package is reported"
    );
    let off = h.api.looks_watch_downloads(false);
    assert!(!off.watching);
}

#[test]
fn a_downloads_folder_that_cannot_be_read_is_explained() {
    let h = Harness::new();
    // The home folder of the harness has no Downloads folder.
    let status = h.api.looks_watch_downloads(true);
    assert!(!status.watching);
    assert!(status.problem.is_some_and(|p| p.contains("Downloads")));
}

// --- Preferences ------------------------------------------------------------------------------

#[test]
fn preferences_round_trip_and_nothing_else_is_written() {
    let h = Harness::new();
    let prefs = h.api.preferences_get().expect("prefs");
    assert_eq!(prefs.default_look.0, "builtin:neutral");
    assert_eq!(prefs.display_rendering, DisplayRendering::Aces2);
    assert!(prefs.check_for_updates_automatically && !prefs.remove_location_from_exports);
    // Reading creates nothing.
    assert_eq!(
        std::fs::read_dir(h.app_root.path()).expect("dir").count(),
        0
    );

    let patch: PreferencesPatch = serde_json::from_value(json!({
        "remove_location_from_exports": true,
        "check_for_updates_automatically": false
    }))
    .expect("patch");
    let next = h.api.preferences_set(&patch).expect("set");
    assert!(next.remove_location_from_exports && !next.check_for_updates_automatically);
    assert_eq!(h.api.preferences_get().expect("prefs"), next);
    let mut names: Vec<_> = std::fs::read_dir(h.app_root.path())
        .expect("dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["preferences.json"]);

    let bad: PreferencesPatch =
        serde_json::from_value(json!({ "default_look": "arri:3000:nothing" })).expect("patch");
    let err = h.api.preferences_set(&bad).expect_err("no such look");
    assert_eq!(err.scope, ErrorScope::Looks);
    assert_eq!(
        h.api.preferences_get().expect("prefs"),
        next,
        "nothing changed"
    );
}

#[test]
fn opening_remembers_the_folder() {
    let h = Harness::new();
    h.open(1);
    let prefs = h.api.preferences_get().expect("prefs");
    assert_eq!(
        prefs.last_used_folders.open.as_deref(),
        h.photos.path().to_str()
    );
}

// --- Export -----------------------------------------------------------------------------------

#[test]
fn an_export_runs_in_the_background_and_reports_its_progress() {
    let h = Harness::new();
    h.open(2);
    h.develop();
    let out = tempfile::tempdir().expect("out");
    let job = jpeg_job(out.path());

    let estimate = h.api.export_estimate(&job).expect("estimate");
    assert!(estimate.bytes.0 > 0.0 && estimate.seconds.0 > 0.0);
    assert_eq!(estimate.adjusted_photos, 0);
    assert!(!estimate.dng_unavailable);
    let command = h.api.cli_preview(&job).expect("preview");
    assert!(command.starts_with("ym render "), "{command}");

    let id = h.api.export_start(&job).expect("start");
    wait_until("the export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total && p.total == 2)
        })
    });
    wait_until("the queue to go idle", || !h.api.export_busy());
    let events = h.events.events();
    let busy: Vec<bool> = events
        .iter()
        .filter_map(|e| match e {
            AppEvent::ExportBusy(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(
        busy.first(),
        Some(&true),
        "busy comes before the first progress event"
    );
    assert_eq!(busy.last(), Some(&false));
    let files: Vec<_> = std::fs::read_dir(out.path())
        .expect("dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(
        files
            .iter()
            .all(|f| f.ends_with(".jpg") || f.ends_with(".jpeg")),
        "{files:?}"
    );
    let reveal = h
        .api
        .export_reveal_path(id)
        .expect("a path to show in Finder");
    assert!(reveal.starts_with(out.path()), "{}", reveal.display());
}

/// The command-line tool of this workspace, if it has been built (`cargo build -p ym-cli`, or
/// `YM_BIN`); the tests that compare the two front ends skip without it.
fn command_line_tool() -> Option<PathBuf> {
    let path = std::env::var_os("YM_BIN").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/debug/ym"),
        PathBuf::from,
    );
    path.is_file().then_some(path)
}

/// The files of a folder with their bytes, by name.
fn folder_contents(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .expect("dir")
        .map(|e| {
            let e = e.expect("entry");
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("read"),
            )
        })
        .collect()
}

/// Exports `files` once through the app's API and once through `ym render`, both with their
/// defaults, and checks that the folders hold the same files with the same bytes.
fn assert_app_and_command_line_agree(h: &Harness, ym: &Path, files: &[PathBuf]) {
    let paths: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().into_owned())
        .collect();
    h.api.open_paths(&paths).expect("open");
    h.develop();

    // The app, with the export defaults it ships with.
    let by_app = tempfile::tempdir().expect("out");
    let mut job = ExportJob {
        scope: ApplyScope::All,
        options: h.api.preferences_get().expect("prefs").export_defaults,
    };
    job.options.destination = Some(by_app.path().to_string_lossy().into_owned());
    let id = h.api.export_start(&job).expect("start");
    wait_until("the export to finish", || {
        h.events.events().iter().any(
            |e| matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total),
        )
    });
    wait_until("the queue to go idle", || !h.api.export_busy());

    // The command line, on the same files, with an app folder of its own.
    let by_cli = tempfile::tempdir().expect("out");
    let cli_home = tempfile::tempdir().expect("cli home");
    let output = std::process::Command::new(ym)
        .arg("render")
        .args(files)
        .arg("--out")
        .arg(by_cli.path())
        .arg("--quiet")
        .env("YELLOWMOON_HOME", cli_home.path())
        .env("HOME", cli_home.path())
        .output()
        .expect("run ym");
    assert!(
        output.status.success(),
        "ym render failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (app, cli) = (
        folder_contents(by_app.path()),
        folder_contents(by_cli.path()),
    );
    assert_eq!(
        app.len(),
        files.len(),
        "{:?}",
        app.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        app.keys().collect::<Vec<_>>(),
        cli.keys().collect::<Vec<_>>(),
        "the same file names"
    );
    for (name, bytes) in &app {
        assert!(bytes == &cli[name], "{name} differs between the app and ym");
    }
}

#[test]
fn the_app_and_the_command_line_write_the_same_bytes() {
    let Some(ym) = command_line_tool() else {
        eprintln!("skipped: no `ym` binary (cargo build -p ym-cli, or set YM_BIN)");
        return;
    };
    let h = Harness::new();
    let files: Vec<PathBuf> = (1..=2)
        .map(|i| crate::testing::write_arw(h.photos.path(), &format!("DSC{i:05}.ARW"), i))
        .collect();
    assert_app_and_command_line_agree(&h, &ym, &files);
}

/// Local evaluation files (`testdata/real`, not committed): an uncompressed 14-bit file and a cRAW
/// one, with their lens tables. `cargo test -p yellowmoon-app -- --ignored real_files`.
#[test]
#[ignore = "needs testdata/real/files and the ym binary; exports two 24 MP photos twice (about a minute)"]
fn real_files_are_exported_with_the_same_bytes_by_the_app_and_the_command_line() {
    let real = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/real/files");
    let files: Vec<PathBuf> = ["ILCE-7M3.ARW", "ILCE-6300_craw.ARW"]
        .iter()
        .map(|n| real.join(n))
        .filter(|f| f.is_file())
        .collect();
    let Some(ym) = command_line_tool() else {
        eprintln!("skipped: no `ym` binary");
        return;
    };
    if files.is_empty() {
        eprintln!("skipped: no files in testdata/real/files");
        return;
    }
    let h = Harness::new();
    assert_app_and_command_line_agree(&h, &ym, &files);
}

/// Every real file through the viewport, the readouts and the Look tab's tiles. Prints what it
/// measured (`--nocapture`).
#[test]
#[ignore = "needs testdata/real/files; develops nine real photos and draws each (about a minute)"]
fn real_files_open_draw_and_answer_the_readouts() {
    let real = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/real/files");
    if !real.is_dir() {
        eprintln!("skipped: no testdata/real/files");
        return;
    }
    let h = Harness::new();
    let started = std::time::Instant::now();
    let opened = h
        .api
        .open_paths(&[real.to_string_lossy().into_owned()])
        .expect("open the folder");
    assert!(opened.photos.len() >= 2, "{} photos", opened.photos.len());
    let state = h.develop();
    eprintln!(
        "{} real photos developed in {:?}",
        state.photos.len(),
        started.elapsed()
    );
    // The cell as the Photos screen reports it on a retina display.
    h.api
        .viewport_set_rect(&ViewportRect {
            x: Num(0.0),
            y: Num(0.0),
            width: Num(720.0),
            height: Num(480.0),
            scale: Num(2.0),
        })
        .expect("rect");
    let last_seq = |h: &Harness| {
        h.events
            .events()
            .iter()
            .filter_map(|e| match e {
                AppEvent::ViewportFrame(f) => Some(f.seq),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    };
    for (i, photo) in state.photos.iter().enumerate() {
        assert!(h.api.viewport().wait_idle(WAIT));
        let seen = last_seq(&h);
        let t = std::time::Instant::now();
        if i > 0 {
            h.api.set_current(photo.id).expect("current");
        }
        // The first frame of this photo: the one the cell asked for, for the first photo (it is
        // current from the start and may be drawn already), else newer than anything shown so far.
        let frame = if i == 0 {
            h.frame_of_size(1440, 960)
        } else {
            h.events
                .wait_for(WAIT, |e| match e {
                    AppEvent::ViewportFrame(f) if f.seq > seen && !f.url.is_empty() => {
                        Some(f.clone())
                    }
                    _ => None,
                })
                .expect("a frame of the photo")
        };
        eprintln!("{}: first frame {:?}", photo.file_name, t.elapsed());
        assert_eq!(
            (frame.width, frame.height),
            (1440, 960),
            "{}",
            photo.file_name
        );
        let png = h.frames.get(frame.seq).expect("the frame is served");
        assert_eq!(png_size(&png), (1440, 960));
        assert!(h.api.viewport().wait_idle(WAIT));

        // The centre of the cell is on the photo, landscape or portrait.
        let centre = h.api.sample_pixel(360.0, 240.0).expect("a pixel");
        assert!(u16::from(centre.r) + u16::from(centre.g) + u16::from(centre.b) > 0);
        h.api
            .histogram(photo.id)
            .unwrap_or_else(|e| panic!("{}: {e}", photo.file_name));
        h.api
            .look_thumbnail(photo.id, &LookId("builtin:neutral".to_owned()))
            .unwrap_or_else(|e| panic!("{}: {e}", photo.file_name));

        // 100 %: a coarse frame first when the full-resolution tiles are not there, the sharp one
        // when they are.
        let seen = last_seq(&h);
        let t = std::time::Instant::now();
        h.api
            .viewport_set_mode(&ViewportMode::Percent { value: Num(100.0) })
            .expect("100 %");
        let sharp = h
            .events
            .wait_for(WAIT, |e| match e {
                AppEvent::ViewportFrame(f) if f.seq > seen && f.sharp => Some(f.clone()),
                _ => None,
            })
            .expect("a sharp frame at 100 %");
        eprintln!(
            "{}: sharp 100 % frame {:?} (frame {})",
            photo.file_name,
            t.elapsed(),
            sharp.seq - seen
        );
        h.api.viewport_set_mode(&ViewportMode::Fit).expect("fit");
        assert!(h.api.viewport().wait_idle(WAIT));
    }
}

/// M3-03: "opening 500 files keeps the UI responsive". The shell's share of that: the commands the
/// webview sends while 500 photos develop answer quickly (the pool is busy with decoding, the
/// commands are not), and the events stay a bounded stream. Prints the numbers (`--nocapture`).
#[test]
#[ignore = "writes 500 synthetic photos and develops them (about half a minute)"]
fn five_hundred_files_leave_the_commands_responsive() {
    let h = Harness::new();
    for i in 1..=500_u64 {
        crate::testing::write_arw(h.photos.path(), &format!("DSC{i:05}.ARW"), i % 7 + 1);
    }
    let t = std::time::Instant::now();
    let opened = h
        .api
        .open_paths(&[h.photos.path().to_string_lossy().into_owned()])
        .expect("open 500 files");
    let open_time = t.elapsed();
    assert_eq!(opened.photos.len(), 500);
    eprintln!("open_paths: {open_time:?}");

    let ids: Vec<PhotoId> = opened.photos.iter().map(|p| p.id).collect();
    let mut slowest = Duration::ZERO;
    let mut calls = 0_u32;
    let mut i = 0;
    while !h
        .api
        .session_state()
        .is_ok_and(|s| s.photos.iter().all(|p| p.status == PhotoStatus::Ready))
    {
        // What a person does while the photos develop: look around, pick, open the tools.
        for call in 0..4 {
            let t = std::time::Instant::now();
            match call {
                0 => {
                    let _ = h.api.session_state();
                }
                1 => {
                    let _ = h.api.set_current(ids[i % ids.len()]);
                }
                2 => {
                    let _ = h.api.set_picked(ids[(i * 3) % ids.len()], i % 2 == 0);
                }
                _ => {
                    let _ = h.api.preferences_get();
                }
            }
            slowest = slowest.max(t.elapsed());
            calls += 1;
        }
        i += 1;
        std::thread::sleep(Duration::from_millis(5));
    }
    eprintln!(
        "developed after {:?}; {calls} commands, the slowest took {slowest:?}",
        t.elapsed()
    );
    let events = h.events.events();
    let progress = events
        .iter()
        .filter(|e| matches!(e, AppEvent::PhotoProgress(_)))
        .count();
    let thumbs = events
        .iter()
        .filter(|e| matches!(e, AppEvent::ThumbnailReady(_)))
        .count();
    eprintln!("{progress} photo_progress events, {thumbs} thumbnails");
    assert!(
        slowest < Duration::from_millis(100),
        "a command took {slowest:?} while photos were developing"
    );
}

#[test]
fn closing_the_session_does_not_stop_an_export() {
    let h = Harness::new();
    h.open(4);
    h.develop();
    let out = tempfile::tempdir().expect("out");
    let id = h.api.export_start(&jpeg_job(out.path())).expect("start");
    h.api.close_session();
    assert!(h.api.session_state().expect("state").photos.is_empty());

    wait_until("the export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total && p.total == 4)
        })
    });
    wait_until("the queue to go idle", || !h.api.export_busy());
    let written = std::fs::read_dir(out.path()).expect("dir").count();
    assert_eq!(written, 4, "every photo of the job was written");
    let busy: Vec<bool> = h
        .events
        .events()
        .iter()
        .filter_map(|e| match e {
            AppEvent::ExportBusy(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(
        busy.first(),
        Some(&true),
        "the app is told the queue is busy"
    );
    assert_eq!(busy.last(), Some(&false), "and told when it is free");
}

#[test]
fn a_cancelled_export_frees_the_queue() {
    let h = Harness::new();
    h.open(6);
    h.develop();
    let out = tempfile::tempdir().expect("out");
    let id = h.api.export_start(&jpeg_job(out.path())).expect("start");
    h.api.export_cancel(id);
    wait_until("the queue to go idle", || !h.api.export_busy());
    // Whatever was written is a whole file, and no temporary file of a photo that was cut short is
    // left behind.
    for entry in std::fs::read_dir(out.path()).expect("dir") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(!name.starts_with('.') && !name.ends_with(".tmp"), "{name}");
        assert!(entry.metadata().expect("meta").len() > 0, "{name} is empty");
    }
    let busy: Vec<bool> = h
        .events
        .events()
        .iter()
        .filter_map(|e| match e {
            AppEvent::ExportBusy(b) => Some(*b),
            _ => None,
        })
        .collect();
    assert_eq!(busy.last(), Some(&false));
}

fn summary(id: u32, status: PhotoStatus) -> PhotoSummary {
    PhotoSummary {
        id: PhotoId(id),
        file_name: format!("DSC{id:05}.ARW"),
        camera: CameraInfo {
            body: "Sony A7R IV".to_owned(),
            model: "ILCE-7RM4".to_owned(),
            model_id: None,
            profile: "A7R IV".to_owned(),
            profile_id: "sony-ilce-7rm4".to_owned(),
            profile_version: "1.0.0".to_owned(),
            match_kind: MatchKind::Exact,
        },
        iso: 100,
        shutter_s: Num(0.01),
        aperture_f: Num(2.8),
        captured_at: "2026-10-04T17:42:10".to_owned(),
        width_px: 192,
        height_px: 128,
        status,
        picked: false,
        reduced: false,
    }
}

#[test]
fn only_waiting_and_developing_photos_hold_an_export_back() {
    let photos = [
        summary(1, PhotoStatus::Ready),
        summary(2, PhotoStatus::Waiting),
        summary(
            3,
            PhotoStatus::Developing {
                stage: PhotoStage::Decode,
            },
        ),
        summary(
            4,
            PhotoStatus::Skipped {
                reason: "Can't open".to_owned(),
            },
        ),
    ];
    assert_eq!(developing_photos(&photos), 2);
    assert_eq!(developing_photos(&photos[..1]), 0);
    assert_eq!(
        developing_photos(&photos[3..]),
        0,
        "a skipped photo never finishes"
    );
    assert_eq!(developing_photos(&[]), 0);
}

#[test]
fn an_export_waits_for_every_photo_to_be_developed_like_the_command_line_does() {
    let h = Harness::new();
    h.open(12);
    let out = tempfile::tempdir().expect("out");
    let mut job = jpeg_job(out.path());
    job.scope = ApplyScope::This;
    // Straight after opening, most photos are still being developed: the answer is the sentence
    // that says so, or (if this machine is fast enough to have finished already) a job.
    match h.api.export_start(&job) {
        Err(e) => {
            assert_eq!(e.scope, ErrorScope::Export);
            assert!(e.message.contains("still being developed"), "{e}");
            assert!(!h.api.export_busy(), "nothing was queued");
        }
        Ok(id) => {
            wait_until("the early job", || {
                h.events.events().iter().any(|e| {
                    matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total)
                })
            });
        }
    }
    h.develop();
    let id = h
        .api
        .export_start(&job)
        .expect("everything is developed now");
    wait_until("the export to finish", || {
        h.events.events().iter().any(
            |e| matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total),
        )
    });
    wait_until("the queue to go idle", || !h.api.export_busy());
}

#[test]
fn an_export_with_nothing_picked_says_what_to_do() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let mut job = jpeg_job(h.photos.path());
    job.scope = ApplyScope::Picked;
    let err = h.api.export_start(&job).expect_err("nothing picked");
    assert_eq!(err.scope, ErrorScope::Export);
    assert!(err.message.contains("Pick some photos"), "{err}");
    assert!(!h.api.export_busy());
}

// --- The system -------------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn the_command_line_tool_is_linked_into_the_home_folder() {
    let h = Harness::new();
    assert!(!h.api.cli_link_status().installed);
    let made = h.api.install_cli_link().expect("install");
    assert!(made.path.ends_with(".local/bin/ym"), "{}", made.path);
    let status = h.api.cli_link_status();
    assert!(status.installed);
    assert_eq!(status.path.as_deref(), Some(made.path.as_str()));
    assert_eq!(
        h.api
            .cli_source()
            .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())),
        Some(Some("ym".to_owned()))
    );
}

#[test]
fn app_info_and_the_looks_folder() {
    let h = Harness::new();
    let info = h.api.app_info();
    assert_eq!(info.version, ym_core::version::app_version());
    assert_eq!(info.pipeline_version, ym_core::version::PIPELINE_VERSION);
    let folder = h.api.looks_folder().expect("looks folder");
    assert!(folder.is_dir() && folder.starts_with(h.app_root.path()));
}

// --- The mirror types lose nothing ------------------------------------------------------------

/// JSON equality that does not mind whether a whole number was written as an integer or a float.
fn same_json(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value::{Array, Bool, Null, Number, Object, String};
    match (a, b) {
        (Null, Null) => true,
        (Bool(x), Bool(y)) => x == y,
        (Number(x), Number(y)) => x.as_f64() == y.as_f64(),
        (String(x), String(y)) => x == y,
        (Array(x), Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q)),
        (Object(x), Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
        }
        _ => false,
    }
}

fn assert_same<C: serde::Serialize, W: serde::Serialize>(what: &str, core: &C, wire: &W) {
    let (c, w) = (
        serde_json::to_value(core).expect("core json"),
        serde_json::to_value(wire).expect("wire json"),
    );
    assert!(
        same_json(&c, &w),
        "{what} lost something on the bridge:\ncore {c}\nwire {w}"
    );
}

#[test]
fn core_types_convert_without_loss() {
    let h = Harness::new();
    let cube = {
        let path = h.photos.path().join("Warm.cube");
        let text = ym_color::lut::Lut3d::identity(5)
            .expect("cube")
            .to_cube_string("Warm");
        std::fs::write(&path, text).expect("write");
        path
    };
    h.api
        .looks_import(&cube.to_string_lossy(), Some(LookInput::Logc3Ei800))
        .expect("import a look");
    h.open(2);
    let state = h.develop();
    let id = state.photos[0].id;
    // Something in every optional field.
    let patch: SettingsPatch = serde_json::from_value(json!({
        "exposure": { "kind": "manual", "ev": -0.7 },
        "white_balance": { "kind": "manual", "kelvin": 4300, "tint": 5 },
        "sharpen": { "kind": "manual", "amount": 120, "radius_px": 1.1 },
        "noise_reduction": { "kind": "manual", "luminance": 30, "color": 40, "detail": 50 },
        "light_source": "tungsten",
        "color_balance": { "midtones": [0.3, 0.1] },
        "crop": { "rect": { "x": 0.1, "y": 0.2, "width": 0.6, "height": 0.5 }, "angle_deg": -3, "aspect": "four_by_five" },
        "orientation": 270,
        "decode": { "neural_demosaic": false },
        "rolloff": { "highlights": 0.2 }
    }))
    .expect("patch");
    h.api.update_settings(id, &patch).expect("update");

    let core = h.api.core();
    assert_same(
        "session_state",
        &core.session.state(),
        &h.api.session_state().expect("state"),
    );
    assert_same(
        "get_settings",
        &core
            .session
            .settings(ym_core::session::PhotoId(id.0))
            .expect("settings"),
        &h.api.get_settings(id).expect("settings"),
    );
    assert_same(
        "looks_list",
        &core.looks_list(),
        &h.api.looks_list().expect("looks"),
    );
    assert_same(
        "looks_display_luts",
        &core.looks_display_luts(),
        &h.api.looks_display_luts().expect("luts"),
    );
    assert_same(
        "preferences",
        &core.preferences_get(),
        &h.api.preferences_get().expect("prefs"),
    );

    let out = tempfile::tempdir().expect("out");
    let job = jpeg_job(out.path());
    let core_job: ym_core::export_queue::ExportJob = convert(&job).expect("job");
    assert_same("export_job", &core_job, &job);
    assert_same(
        "export_estimate",
        &core.export_estimate(&core_job),
        &h.api.export_estimate(&job).expect("estimate"),
    );

    let src =
        PhotoSource::from_session(&core.session, ym_core::session::PhotoId(id.0)).expect("source");
    let histogram = core
        .renderer
        .histogram(&src, &CancelToken::new())
        .expect("histogram");
    assert_same(
        "histogram",
        &histogram,
        &h.api.histogram(id).expect("histogram"),
    );
}

#[test]
fn a_patch_survives_the_bridge_and_a_missing_key_stays_missing() {
    // `crop: null` is a present key and no `crop` is none: the manual `Serialize` keeps them apart.
    let clear: SettingsPatch =
        serde_json::from_value(json!({ "crop": null, "white_pick": null })).expect("patch");
    assert_eq!(clear.crop, Some(None));
    assert_eq!(clear.white_pick, Some(None));
    let json = serde_json::to_value(&clear).expect("json");
    assert_eq!(json, json!({ "crop": null, "white_pick": null }));
    let core: ym_core::settings::SettingsPatch = convert(&clear).expect("core patch");
    assert_eq!(core.crop, Some(None));
    assert_eq!(core.white_pick, Some(None));
    assert_eq!(core.look, None);

    let none: SettingsPatch = serde_json::from_value(json!({})).expect("empty");
    assert_eq!(serde_json::to_value(&none).expect("json"), json!({}));
    let core: ym_core::settings::SettingsPatch = convert(&none).expect("core");
    assert_eq!(core, ym_core::settings::SettingsPatch::default());

    // Parts of a manual white balance may be left out; the core fills them in.
    let part: SettingsPatch =
        serde_json::from_value(json!({ "white_balance": { "kind": "manual", "kelvin": 5000 } }))
            .expect("patch");
    let core: ym_core::settings::SettingsPatch = convert(&part).expect("core");
    assert!(core.white_balance.is_some());
}
