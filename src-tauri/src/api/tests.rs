//! The command handlers against `ym-core`, with synthetic ARW files and no webview
//! (`docs/01` §IPC). The harness is in `crate::testing`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use super::*;
use crate::testing::{Harness, WAIT, jpeg_job, png_size, wait_until};
use crate::wire::*;

// --- The paper --------------------------------------------------------------------------------

/// The paper of the dark theme in `src/styles/tokens.css` (`--paper`), as 8-bit codes.
fn dark_paper_of_tokens() -> [u8; 3] {
    let tokens = include_str!("../../../src/styles/tokens.css");
    let dark = tokens
        .split(r#":root[data-theme="dark"]"#)
        .nth(1)
        .expect("tokens.css has a dark theme block");
    let hex = dark
        .split("--paper:")
        .nth(1)
        .expect("the dark theme sets --paper")
        .trim_start();
    let hex = hex.strip_prefix('#').expect("--paper is a hex colour");
    let code = |i: usize| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("two hex digits");
    [code(0), code(1), code(2)]
}

#[test]
fn the_letterbox_the_app_paints_is_the_paper_of_the_interface() {
    assert_eq!(
        PAPER_RGB8,
        dark_paper_of_tokens(),
        "PAPER_RGB8 is --paper of the dark theme"
    );
    let h = Harness::new();
    assert_eq!(
        h.api.core.renderer.options().background,
        PAPER_RGB8,
        "the core's frames, thumbnails and look tiles are letterboxed with the paper"
    );
}

#[test]
fn the_window_is_the_paper_from_its_first_moment() {
    // `backgroundColor` of the window in tauri.conf.json is what macOS shows before `native.rs` sets the window's own colour (a
    // transparent window is clear until then, and the desktop would show through).
    let conf: serde_json::Value =
        serde_json::from_str(include_str!("../../tauri.conf.json")).expect("tauri.conf.json");
    let window = &conf["app"]["windows"][0];
    let [r, g, b] = PAPER_RGB8;
    assert_eq!(
        window["backgroundColor"].as_str(),
        Some(format!("#{r:02x}{g:02x}{b:02x}").as_str()),
        "the main window's backgroundColor is the paper"
    );
}

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
            // The event carries an address, not the picture; the picture is served at it.
            let (w, hh) = png_size(&served(&h, &t.url));
            assert!(w.max(hh) <= 240 && w.max(hh) > 0, "{w} x {hh}");
        }
    }
}

/// What the `ym-frame` scheme answers for an address the shell announced (200 and the PNG).
fn served(h: &Harness, url: &str) -> Vec<u8> {
    let (status, mime, body) = h.frames.respond(path_of_url(url));
    assert_eq!((status, mime), (200, "image/png"), "{url}");
    body.into_owned()
}

/// `/<launch>/<file>`: what follows the host of an address of the scheme.
fn path_of_url(url: &str) -> &str {
    assert!(url.contains("ym-frame"), "{url}");
    let host = url.find("localhost/").expect("a localhost address");
    &url[host + "localhost".len()..]
}

#[test]
fn thumbnails_cross_the_bridge_as_addresses_and_can_be_asked_for_again() {
    let h = Harness::new();
    h.open(4);
    let state = h.develop();
    let ids: Vec<_> = state.photos.iter().map(|p| p.id).collect();
    wait_until("four thumbnails", || {
        h.events
            .events()
            .iter()
            .filter(|e| matches!(e, AppEvent::ThumbnailReady(_)))
            .count()
            >= 4
    });
    // The event is small: an id and an address, whatever the size of the picture.
    let announced: Vec<ThumbnailReady> = h
        .events
        .events()
        .into_iter()
        .filter_map(|e| match e {
            AppEvent::ThumbnailReady(t) => Some(t),
            _ => None,
        })
        .collect();
    for t in &announced {
        assert!(t.url.len() < 100, "{}", t.url);
        assert!(serde_json::to_string(t).expect("json").len() < 150);
    }
    // A webview that reloaded asks again and gets every photo's address, the newest ones.
    let again = h.api.thumbnails_now();
    assert_eq!(again.iter().map(|t| t.id).collect::<Vec<_>>(), ids);
    for t in &again {
        let newest = announced
            .iter()
            .rev()
            .find(|a| a.id == t.id)
            .expect("announced");
        assert_eq!(t.url, newest.url);
        assert!(png_size(&served(&h, &t.url)).0 > 0);
    }

    // A new look makes a new thumbnail at a new address; the one before still loads for a moment.
    let before = again[0].clone();
    h.api
        .apply_look(ApplyScope::All, &LookId("builtin:neutral".to_owned()), 40.0)
        .expect("apply");
    h.api
        .thumbnails()
        .refresh(ids.iter().map(|id| ym_core::session::PhotoId(id.0)));
    assert!(h.api.thumbnails().wait_idle(WAIT));
    let after = h.api.thumbnails_now();
    assert_ne!(after[0].url, before.url, "a new picture is a new address");
    assert!(png_size(&served(&h, &after[0].url)).0 > 0);
    assert!(
        png_size(&served(&h, &before.url)).0 > 0,
        "the older one is still served"
    );

    // Closing the session forgets them: nothing to get again, and the addresses are gone.
    h.api.close_session();
    assert!(h.api.thumbnails_now().is_empty());
    assert_eq!(h.frames.respond(path_of_url(&after[0].url)).0, 404);
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

#[test]
fn closing_the_session_cancels_its_queued_renders_and_forgets_what_they_cached() {
    let h = Harness::new();
    h.open(8);
    let state = h.develop();
    let ids: Vec<_> = state.photos.iter().map(|p| p.id).collect();
    h.rect(300.0, 200.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    let core = Arc::clone(h.api.core());
    assert!(
        core.renderer.cached_bytes() > 0,
        "the viewport and the thumbnails cached something"
    );

    // A pile of renders behind each other: look tiles and histograms of every photo (each call
    // waits for its job, as the webview's requests do), thumbnails refreshes and viewport frames.
    let look = LookId("builtin:neutral".to_owned());
    let calls: Vec<_> = ids
        .iter()
        .flat_map(|id| {
            let (a, b) = (Arc::clone(&h.api), Arc::clone(&h.api));
            let (id, look) = (*id, look.clone());
            [
                std::thread::spawn(move || a.look_thumbnail(id, &look).map(|png| png.len())),
                std::thread::spawn(move || b.histogram(id).map(|_| 0)),
            ]
        })
        .collect();
    h.api
        .thumbnails()
        .refresh(ids.iter().map(|id| ym_core::session::PhotoId(id.0)));
    h.api.close_session();

    // Every call returns (an answer or a plain error), none hangs behind a queue of renders that
    // are no longer wanted.
    for call in calls {
        let _ = call.join().expect("the call did not panic");
    }
    // Nothing is queued or running in the render scheduler, nothing was kept, nothing is asked
    // for by the thumbnail refresher, and the cell shows no picture.
    wait_until("the render queue to drain", || {
        core.renderer.scheduler().foreground_load() == 0 && core.renderer.scheduler().pending() == 0
    });
    assert_eq!(core.renderer.cached_bytes(), 0, "the caches were forgotten");
    assert!(h.api.thumbnails().wait_idle(WAIT));
    assert!(h.api.viewport().wait_idle(WAIT));
    let (photo, _) = h.api.viewport().snapshot_synced();
    assert_eq!(photo, None);
    let last_frame = h.events.events().into_iter().rev().find_map(|e| match e {
        AppEvent::ViewportFrame(f) => Some(f),
        _ => None,
    });
    assert!(
        last_frame.is_some_and(|f| f.url.is_empty() && f.width == 0),
        "the last thing the cell was told is that there is no picture"
    );
    // The closed session's photos stay closed: no event for them follows, whatever was queued.
    let seen = h.events.events().len();
    std::thread::sleep(Duration::from_millis(500));
    let later: Vec<_> = h.events.events().split_off(seen);
    assert!(
        later
            .iter()
            .all(|e| !matches!(e, AppEvent::ThumbnailReady(_) | AppEvent::PhotoProgress(_))),
        "events after the close: {later:?}"
    );

    // A new session starts from nothing: its photos have their own ids and render.
    h.api
        .open_paths(&[h.photos.path().to_string_lossy().into_owned()])
        .expect("open again");
    let state = h.develop();
    assert!(state.photos.iter().all(|p| !ids.contains(&p.id)));
    h.rect(310.0, 210.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    assert!(core.renderer.cached_bytes() > 0);
}

// --- Removing photos --------------------------------------------------------------------------

/// The bytes and modification time of every file in the photo folder, by name.
fn disk_state(dir: &Path) -> Vec<(String, Vec<u8>, std::time::SystemTime)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("dir")
        .map(|e| {
            let path = e.expect("entry").path();
            (
                path.file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned(),
                std::fs::read(&path).expect("read"),
                std::fs::metadata(&path)
                    .expect("metadata")
                    .modified()
                    .expect("modified"),
            )
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

#[test]
fn removing_photos_takes_them_out_hands_the_photo_on_screen_over_and_leaves_the_files() {
    let h = Harness::new();
    h.open(5);
    let done = h.develop();
    let ids: Vec<PhotoId> = done.photos.iter().map(|p| p.id).collect();
    wait_until("five thumbnails", || h.frames.thumbnail_urls().len() == 5);
    let urls: std::collections::HashMap<u32, String> =
        h.frames.thumbnail_urls().into_iter().collect();
    let on_disk = disk_state(h.photos.path());
    std::thread::sleep(Duration::from_millis(30));

    h.api.set_current(ids[1]).expect("current");
    h.api.set_picked(ids[2], true).expect("pick");
    h.api.set_picked(ids[4], true).expect("pick");
    let change = h
        .api
        .remove_photos(&[ids[1], ids[2], PhotoId(9999)])
        .expect("remove");
    // The ids that were in the session, in filmstrip order; the unknown one is ignored.
    assert_eq!(change.photo_ids, vec![ids[1], ids[2]]);
    assert_eq!(change.already_open, 0);
    let names: Vec<_> = change
        .session
        .photos
        .iter()
        .map(|p| p.file_name.as_str())
        .collect();
    assert_eq!(names, ["DSC00001.ARW", "DSC00004.ARW", "DSC00005.ARW"]);
    // The photo that was on screen went: the next one is, in the core and in the viewport.
    assert_eq!(change.session.current_index, Some(1));
    assert_eq!(change.session.picked, vec![ids[4]]);
    assert_eq!(h.api.core().session.current(), Some(core_id(ids[3])));
    assert_eq!(h.api.viewport().snapshot_synced().0, Some(core_id(ids[3])));
    // The removed photos are gone for every command, and their pictures are not served.
    assert!(h.api.get_settings(ids[1]).is_err());
    assert!(h.api.set_picked(ids[2], true).is_err());
    for gone in [ids[1], ids[2]] {
        let url = urls.get(&gone.0).expect("it had a thumbnail");
        let (status, _, _) = h.frames.respond(path_of_url(url));
        assert_eq!(status, 404, "{url}");
    }
    let kept: Vec<u32> = h
        .frames
        .thumbnail_urls()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(kept, vec![ids[0].0, ids[3].0, ids[4].0]);
    // The files on disk are as they were (hard rule 5).
    assert_eq!(disk_state(h.photos.path()), on_disk);

    // Nothing is announced for a removed photo from now on.
    let seen = h.events.events().len();
    std::thread::sleep(Duration::from_millis(200));
    for e in &h.events.events()[seen..] {
        match e {
            AppEvent::PhotoProgress(p) => assert!(p.id != ids[1] && p.id != ids[2], "{p:?}"),
            AppEvent::ThumbnailReady(t) => assert!(t.id != ids[1] && t.id != ids[2], "{t:?}"),
            _ => {}
        }
    }
    // The session as a fresh `session_state` says it.
    assert_eq!(h.api.session_state().expect("state"), change.session);
}

#[test]
fn undo_brings_the_photos_their_settings_and_their_thumbnails_back() {
    let h = Harness::new();
    h.open(4);
    let done = h.develop();
    let ids: Vec<PhotoId> = done.photos.iter().map(|p| p.id).collect();
    wait_until("four thumbnails", || h.frames.thumbnail_urls().len() == 4);
    let urls: std::collections::HashMap<u32, String> =
        h.frames.thumbnail_urls().into_iter().collect();
    let patch: SettingsPatch = serde_json::from_value(
        json!({ "contrast": 25, "exposure": { "kind": "manual", "ev": 0.5 } }),
    )
    .expect("patch");
    h.api.update_settings(ids[1], &patch).expect("edit");
    h.api.set_picked(ids[1], true).expect("pick");
    let edited = h.api.get_settings(ids[1]).expect("settings");
    h.api.set_current(ids[1]).expect("current");

    h.api.remove_photos(&[ids[1], ids[3]]).expect("remove");
    let seen = h.events.events().len();
    let back = h.api.restore_photos().expect("restore");
    assert_eq!(back.photo_ids, vec![ids[1], ids[3]]);
    assert_eq!(back.session.photos.len(), 4);
    assert_eq!(
        back.session.photos.iter().map(|p| p.id).collect::<Vec<_>>(),
        ids
    );
    // On screen again, picked again, with its edits.
    assert_eq!(back.session.current_index, Some(1));
    assert_eq!(back.session.picked, vec![ids[1]]);
    assert_eq!(h.api.get_settings(ids[1]).expect("settings"), edited);
    assert_eq!(h.api.viewport().snapshot_synced().0, Some(core_id(ids[1])));
    // The thumbnails are announced again at the addresses they had, and served.
    let announced: Vec<(PhotoId, String)> = h.events.events()[seen..]
        .iter()
        .filter_map(|e| match e {
            AppEvent::ThumbnailReady(t) => Some((t.id, t.url.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(announced.len(), 2, "{announced:?}");
    for (id, url) in &announced {
        assert_eq!(url, urls.get(&id.0).expect("it had one"));
        served(&h, url);
    }
    // There is nothing more to put back.
    let again = h.api.restore_photos().expect("restore");
    assert!(again.photo_ids.is_empty());
    assert_eq!(again.session.photos.len(), 4);
}

#[test]
fn removing_the_last_photos_empties_the_session_and_undo_fills_it_again() {
    let h = Harness::new();
    h.open(2);
    let done = h.develop();
    let ids: Vec<PhotoId> = done.photos.iter().map(|p| p.id).collect();
    let change = h.api.remove_photos(&ids).expect("remove");
    assert!(change.session.photos.is_empty());
    assert_eq!(change.session.current_index, None);
    assert_eq!(change.session.folder_name, None);
    assert_eq!(h.api.core().session.current(), None);
    assert_eq!(h.api.viewport().snapshot_synced().0, None);
    assert!(h.frames.thumbnail_urls().is_empty());
    // The files can be opened again: they are not "already open" any more.
    let reopened = h.open(2);
    assert_eq!(reopened.photos.len(), 2);
    assert_eq!(
        reopened.opened,
        Some(OpenSummary {
            added: 2,
            already_open: 0
        })
    );
    h.api.close_session();
    assert!(
        h.api
            .restore_photos()
            .expect("restore")
            .photo_ids
            .is_empty(),
        "the session was closed"
    );

    // Remove everything and put it back.
    let opened = h.open(2);
    h.develop();
    let ids: Vec<PhotoId> = opened.photos.iter().map(|p| p.id).collect();
    h.api.remove_photos(&ids).expect("remove");
    let back = h.api.restore_photos().expect("restore");
    assert_eq!(back.photo_ids, ids);
    assert_eq!(back.session.current_index, Some(0));
    assert!(back.session.folder_name.is_some());
    h.develop();
}

#[test]
fn an_export_that_was_queued_keeps_the_photos_it_was_started_with() {
    let h = Harness::new();
    h.open(4);
    let done = h.develop();
    let ids: Vec<PhotoId> = done.photos.iter().map(|p| p.id).collect();
    let out = tempfile::tempdir().expect("out");
    let id = h.api.export_start(&jpeg_job(out.path())).expect("start");
    // The person removes three of the four photos while the export runs.
    h.api.remove_photos(&ids[1..]).expect("remove");
    assert_eq!(h.api.session_state().expect("state").photos.len(), 1);

    wait_until("the export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total && p.total == 4)
        })
    });
    wait_until("the queue to go idle", || !h.api.export_busy());
    assert_eq!(
        std::fs::read_dir(out.path()).expect("dir").count(),
        4,
        "every photo of the job was written"
    );
    // The removed photos left no render cache behind.
    for gone in &ids[1..] {
        assert_eq!(
            h.api.core().renderer.counters(core_id(*gone)),
            Default::default()
        );
    }
}

// --- Settings ---------------------------------------------------------------------------------

#[test]
fn the_output_sharpening_crosses_the_bridge_as_a_plain_number() {
    let h = Harness::new();
    h.open(1);
    h.develop();
    let id = h.first();
    // The default is 100, in the stored settings and in the resolved values.
    let before = h.api.get_settings(id).expect("settings");
    assert_eq!(before.settings.output_sharpening, Num(100.0));
    assert_eq!(before.values.output_sharpening, Num(100.0));
    // Moving the slider sends a number; out of range is clamped by the core; 0 is a value.
    for (sent, shown) in [
        (35.0, 35.0),
        (0.0, 0.0),
        (250.0, 200.0),
        (-3.0, 0.0),
        (100.0, 100.0),
    ] {
        let patch: SettingsPatch =
            serde_json::from_value(json!({ "output_sharpening": sent })).expect("patch");
        let after = h.api.update_settings(id, &patch).expect("update");
        assert_eq!(after.settings.output_sharpening, Num(shown), "sent {sent}");
        assert_eq!(after.values.output_sharpening, Num(shown), "sent {sent}");
    }
    // A string is not a number: the bridge refuses it before the core sees it.
    assert!(
        serde_json::from_value::<SettingsPatch>(json!({ "output_sharpening": "a lot" })).is_err()
    );
    // The patch of one field leaves the others alone.
    let patch: SettingsPatch =
        serde_json::from_value(json!({ "output_sharpening": 60, "contrast": 5 })).expect("patch");
    let after = h.api.update_settings(id, &patch).expect("update");
    assert_eq!(after.settings.output_sharpening, Num(60.0));
    assert_eq!(
        after.settings.contrast,
        PercentSetting::Manual { value: Num(5.0) }
    );
}

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
    assert_eq!(
        after.settings.contrast,
        PercentSetting::Manual { value: Num(20.0) }
    );
    assert_eq!(
        after.settings.color_balance.shadows,
        crate::wire::BalanceSetting::Manual {
            point: [Num(0.2), Num(-0.1)]
        }
    );
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
    assert_eq!(
        cleared.settings.contrast,
        PercentSetting::Manual { value: Num(20.0) }
    );

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
    assert_eq!(
        h.api.get_settings(id).expect("settings").settings.contrast,
        PercentSetting::Auto
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
fn opening_more_files_adds_to_the_session_and_says_what_was_new() {
    let h = Harness::new();
    let first = h.open(2);
    assert_eq!(
        first.opened,
        Some(OpenSummary {
            added: 2,
            already_open: 0
        })
    );
    let ids: Vec<_> = first.photos.iter().map(|p| p.id).collect();
    h.develop();
    // The same folder again: every file is open already, nothing is added, nothing is an error.
    let folder = h.photos.path().to_string_lossy().into_owned();
    let again = h
        .api
        .open_paths(std::slice::from_ref(&folder))
        .expect("open again");
    assert_eq!(
        again.opened,
        Some(OpenSummary {
            added: 0,
            already_open: 2
        })
    );
    assert_eq!(again.photos.len(), 2);
    // One new file and the two known ones: the new photo goes after the others, and the current
    // photo and the photos before it stay as they were.
    h.api.set_current(ids[1]).expect("current");
    crate::testing::write_arw(h.photos.path(), "DSC00003.ARW", 3);
    let more = h.api.open_paths(&[folder]).expect("open more");
    assert_eq!(
        more.opened,
        Some(OpenSummary {
            added: 1,
            already_open: 2
        })
    );
    assert_eq!(
        more.photos.iter().map(|p| p.id).take(2).collect::<Vec<_>>(),
        ids
    );
    assert_eq!(more.photos.len(), 3);
    assert_eq!(more.current_index, Some(1), "the current photo stays");
    // Only an open says what it did.
    assert_eq!(h.api.session_state().expect("state").opened, None);
}

#[test]
fn copy_and_paste_settings_move_a_photos_settings_onto_others() {
    let h = Harness::new();
    h.open(3);
    let state = h.develop();
    let ids: Vec<_> = state.photos.iter().map(|p| p.id).collect();
    let edit: SettingsPatch = serde_json::from_value(json!({
        "exposure": { "kind": "manual", "ev": 0.6 },
        "contrast": 25,
        "look": "builtin:neutral",
        "look_intensity": 45,
        "crop": { "rect": { "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5 }, "angle_deg": 2, "aspect": "free" },
        "orientation": 90
    }))
    .expect("patch");
    h.api.update_settings(ids[0], &edit).expect("edit");

    let copied = h.api.copy_settings(ids[0]).expect("copy");
    assert!(copied.crop.is_none() && copied.orientation.is_none() && copied.white_pick.is_none());
    assert_eq!(
        copied.contrast,
        Some(PercentPatch::Setting(PercentSetting::Manual {
            value: Num(25.0)
        }))
    );
    let before_events = h.events.events().len();

    let report = h
        .api
        .paste_settings(&[ids[1], ids[2], PhotoId(9999)], &copied)
        .expect("paste");
    assert_eq!(report.skipped, [PhotoId(9999)]);
    assert_eq!(
        report.pasted.iter().map(|p| p.id).collect::<Vec<_>>(),
        [ids[1], ids[2]]
    );
    for id in [ids[1], ids[2]] {
        let r = h.api.get_settings(id).expect("settings");
        assert_eq!(r.values.contrast.0, 25.0);
        assert_eq!(r.values.exposure_ev.0, 0.6);
        assert_eq!(r.values.look_intensity.0, 45.0);
        assert!(r.settings.crop.is_none(), "the crop is not pasted");
    }
    // Every photo that changed is announced, once.
    let events = h.events.events();
    for id in [ids[1], ids[2]] {
        let n = events[before_events..]
            .iter()
            .filter(|e| **e == AppEvent::SettingsChanged(id))
            .count();
        assert_eq!(n, 1, "settings_changed for {id:?}");
    }
    // The same paste again changes nothing, so announces nothing and is an empty step.
    let after = events.len();
    let again = h.api.paste_settings(&[ids[1]], &copied).expect("again");
    assert_eq!(again.pasted.len(), 1);
    assert!(again.pasted[0].before == SettingsPatch::default());
    assert_eq!(h.events.events().len(), after);

    // A look that is not in the library is refused as `apply_look` refuses it, and nothing changes.
    let gone: SettingsPatch =
        serde_json::from_value(json!({ "look": "arri:3000:nothing", "contrast": -5 }))
            .expect("patch");
    let err = h
        .api
        .paste_settings(&[ids[1]], &gone)
        .expect_err("no such look");
    assert_eq!(err.scope, ErrorScope::Looks);
    assert_eq!(
        h.api
            .get_settings(ids[1])
            .expect("settings")
            .values
            .contrast
            .0,
        25.0
    );
    assert!(h.api.copy_settings(PhotoId(9999)).is_err());
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

    // The tile does not cross the bridge as bytes: the answer is the address of a PNG that the
    // shell serves on the `ym-frame` scheme, a new address for every request.
    let neutral = LookId("builtin:neutral".to_owned());
    let url = h.api.look_thumbnail(id, &neutral).expect("look tile");
    assert!(url.contains("/look-"), "{url}");
    let (w, hh) = png_size(&served(&h, &url));
    assert!(w.max(hh) <= 240 && w.max(hh) > 0, "{w} x {hh}");
    let again = h.api.look_thumbnail(id, &neutral).expect("look tile");
    assert_ne!(again, url);
    // The tile the webview may still be loading stays served for a moment; the one before it is
    // replaced, and the session closing drops them all.
    assert_eq!(h.frames.respond(path_of_url(&url)).0, 200);
    let third = h.api.look_thumbnail(id, &neutral).expect("look tile");
    assert_eq!(h.frames.respond(path_of_url(&url)).0, 404);
    assert_eq!(h.frames.respond(path_of_url(&again)).0, 200);
    assert_eq!(h.frames.respond(path_of_url(&third)).0, 200);
    let err = h
        .api
        .look_thumbnail(id, &LookId("arri:3000:film-warm".to_owned()))
        .expect_err("not imported");
    assert_eq!(err.scope, ErrorScope::Looks);
    assert!(!err.message.is_empty());
    h.api.close_session();
    assert_eq!(h.frames.respond(path_of_url(&third)).0, 404);
    assert_eq!(h.frames.respond(path_of_url(&again)).0, 404);
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
fn nothing_is_drawn_before_the_webview_has_said_where_the_photo_cell_is() {
    // The controller starts with a 1 x 1 cell. A photo that develops before the first
    // `viewport_set_rect` used to be drawn into it: a frame of one pixel, sent to the webview and
    // to the native layer, which the screen then had to throw away.
    let h = Harness::new();
    h.open(1);
    h.develop();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(h.api.viewport().frames_presented(), 0);
    assert!(
        !h.events
            .events()
            .iter()
            .any(|e| matches!(e, AppEvent::ViewportFrame(_) | AppEvent::RenderStats(_))),
        "a frame was drawn without a cell"
    );
    // A cell with no area (a screen that is hidden, a layout that has not run) is not the cell.
    h.rect(0.0, 0.0);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(h.api.viewport().frames_presented(), 0);
    // Waiting for the idle viewport does not wait for a rectangle that has not come.
    assert!(h.api.viewport().wait_idle(Duration::from_secs(1)));

    // The first real rectangle draws the photo, and the first frame is the size of the cell.
    h.rect(300.0, 200.0);
    let first = h.frame_of_size(300, 200);
    let frames: Vec<_> = h
        .events
        .events()
        .into_iter()
        .filter_map(|e| match e {
            AppEvent::ViewportFrame(f) if !f.url.is_empty() => Some((f.width, f.height)),
            _ => None,
        })
        .collect();
    assert_eq!(frames.first(), Some(&(300, 200)), "{frames:?}");
    assert!(first.width == 300 && first.height == 200);
}

#[test]
fn the_render_stats_of_a_frame_carry_its_zoom_and_the_fit_zoom() {
    // The webview used to estimate these; the core knows the lens crop, the straighten angle and
    // the crop between the photo's size and the picture.
    let h = Harness::new();
    h.open(1);
    h.develop();
    h.rect(100.0, 100.0);
    h.frame_of_size(100, 100);
    assert!(h.api.viewport().wait_idle(WAIT));
    let last_stats = || {
        h.events
            .events()
            .into_iter()
            .rev()
            .find_map(|e| match e {
                AppEvent::RenderStats(s) => Some(s),
                _ => None,
            })
            .expect("render_stats")
    };
    let at_fit = last_stats();
    let fit = at_fit.fit_percent.expect("the Fit zoom").0;
    // The photo is about 192 x 128 pixels in a 100 x 100 cell: Fit is around 52 %, and the
    // frame is drawn at it.
    assert!(fit > 40.0 && fit < 60.0, "Fit {fit}");
    assert!((at_fit.zoom_percent.expect("the zoom").0 - fit).abs() < 1e-9);

    let n = h.api.viewport().frames_presented();
    h.api
        .viewport_set_mode(&ViewportMode::Percent { value: Num(100.0) })
        .expect("100 %");
    h.frame_after(n);
    assert!(h.api.viewport().wait_idle(WAIT));
    let at_100 = last_stats();
    assert!(
        (at_100.zoom_percent.expect("the zoom").0 - 100.0).abs() < 1e-9,
        "{at_100:?}"
    );
    assert!(
        (at_100.fit_percent.expect("the Fit zoom").0 - fit).abs() < 1e-9,
        "{at_100:?}"
    );
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
fn pick_colour_and_the_chroma_scatter_cross_the_bridge() {
    let h = Harness::new();
    // Asked for while photos are still developing, the scatter is empty, never an error.
    let opened = h.open(6);
    let last = opened.photos.last().expect("photos").id;
    let early = h
        .api
        .chroma_scatter(last)
        .expect("no error while developing");
    assert!(early.points.len().is_multiple_of(2));
    h.develop();
    let id = h.first();
    let scatter = h.api.chroma_scatter(id).expect("scatter");
    assert!(!scatter.points.is_empty() && scatter.points.len() <= 2 * 2048);
    assert_eq!(scatter.points.len() % 2, 0);
    assert_eq!(h.api.chroma_scatter(id).expect("scatter"), scatter);
    assert!(h.api.chroma_scatter(PhotoId(9999)).is_err());
    // Pick colour: the photo's chroma under the pointer, `null` on the paper and for a point that
    // is not a number.
    h.rect(300.0, 300.0);
    assert!(h.api.viewport().wait_idle(WAIT));
    let inside = h
        .api
        .sample_chroma(150.0, 150.0)
        .expect("chroma")
        .expect("a point of the photo");
    assert!(inside.a.0.is_finite() && inside.b.0.is_finite());
    assert_eq!(
        serde_json::to_value(inside).expect("json"),
        json!({ "a": inside.a.0, "b": inside.b.0 })
    );
    assert_eq!(h.api.sample_chroma(150.0, 5.0).expect("paper"), None);
    assert_eq!(
        h.api.sample_chroma(f64::NAN, 3.0).expect("not a number"),
        None
    );
    // A pin on that colour moves the picture there but not what Pick colour reads (the warp's
    // input), and the answer is the core's.
    let pin = json!({ "source": [inside.a.0, inside.b.0], "target": [inside.a.0 + 0.05, inside.b.0],
        "locked": false, "range": 0.04, "shadows": 1, "highlights": 1, "pivot_ev": 0, "exposure_ev": 0 });
    let patch: SettingsPatch =
        serde_json::from_value(json!({ "color_warp": { "pins": [pin] } })).expect("patch");
    let resolved = h.api.update_settings(id, &patch).expect("update");
    assert_eq!(resolved.settings.color_warp.pins.len(), 1);
    assert_eq!(resolved.values.color_warp, resolved.settings.color_warp);
    assert_eq!(
        h.api.sample_chroma(150.0, 150.0).expect("chroma"),
        Some(inside)
    );
    // The settings travel with Copy settings and come back in `get_settings`.
    let copied = h.api.copy_settings(id).expect("copy");
    assert_eq!(
        copied.color_warp,
        Some(resolved.settings.color_warp.clone())
    );
    let cleared: SettingsPatch =
        serde_json::from_value(json!({ "color_warp": { "pins": [] } })).expect("patch");
    let resolved = h.api.update_settings(id, &cleared).expect("update");
    assert!(resolved.settings.color_warp.pins.is_empty());
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
    // The library, the kind of file and the table size cross the bridge with the look.
    assert_eq!(result.added[0].library, LookLibrary::User);
    assert_eq!(result.added[0].format, Some(LookFormat::Cube));
    assert_eq!(result.added[0].lut_size, Some(5));
    wait_until("looks_changed", || {
        h.events.events().contains(&AppEvent::LooksChanged)
    });
    let listed = h.api.looks_list().expect("list");
    assert!(listed.iter().any(|l| l.id == result.added[0].id));
    // A built-in look has no file: no format and no size.
    let neutral = listed
        .iter()
        .find(|l| l.id.0 == "builtin:neutral")
        .expect("Neutral is listed");
    assert_eq!(neutral.library, LookLibrary::BuiltIn);
    assert_eq!((neutral.format, neutral.lut_size), (None, None));
    assert!(
        listed
            .iter()
            .any(|l| l.id.0 == "builtin:signature" && l.is_default)
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
        "builtin:signature",
        "removing the default look puts Signature back"
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

    // A zip of `.alf4` files that are not look files at all.
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
            assert!(reason.contains(".alf4"), "{reason}");
            assert!(reason.contains("It is not an ARRI Look File"), "{reason}");
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

/// A look file written by our own writer from an identity LUT (and a CDL when asked).
fn write_look_file(dir: &Path, name: &str, kind: &str, cdl: bool) -> PathBuf {
    use ym_color::alf::AlfFile;
    let lut = ym_color::lut::Lut3d::identity(5).expect("identity cube");
    let cdl = cdl
        .then(|| ym_color::cdl::Cdl::new([1.2, 1.0, 0.9], [0.0; 3], [1.0; 3], 1.1).expect("cdl"));
    let file = match kind {
        "alf4" => AlfFile::alf4(&lut, cdl.as_ref(), "", "x.cube"),
        "aml" => AlfFile::aml(&lut, cdl.as_ref(), "x.cube"),
        "alf4c" => AlfFile::alf4c(Some(&lut), cdl.as_ref(), Some(&lut), Some(&lut), ""),
        other => panic!("{other}"),
    };
    let path = dir.join(name);
    std::fs::write(&path, file.to_bytes().expect("write")).expect("write the look file");
    path
}

#[test]
fn look_files_are_recognised_and_imported_without_a_question() {
    let h = Harness::new();
    let dir = h.photos.path();

    // An .alf4 with a CDL: the person's own look, with the input read from the file.
    let alf4 = write_look_file(dir, "Warm Pull.alf4", "alf4", true);
    let path = alf4.to_string_lossy().into_owned();
    let seen = h.api.looks_recognise(&path);
    assert_eq!(seen.source, SourceKind::File);
    match &seen.recognition {
        Recognition::LookFiles {
            what,
            contents,
            file_count,
            input,
            note,
        } => {
            assert_eq!(what, "ARRI Look Files · LogC4");
            assert_eq!(contents, "1 .alf4 · 5-point");
            assert_eq!((*file_count, *input), (1, Some(LookInput::Logc4)));
            assert!(
                note.as_deref()
                    .is_some_and(|n| n.contains("your own looks"))
            );
        }
        other => panic!("expected look files, got {other:?}"),
    }
    let result = h
        .api
        .looks_import(&path, None)
        .expect("import without an answer");
    assert_eq!(result.added.len(), 1);
    assert_eq!(result.added[0].group, LookGroup::User);
    assert_eq!(result.added[0].input, LookInput::Logc4);
    assert_eq!(result.added[0].name, "Warm Pull");
    // The person's own look file is in Your LUTs, and says what it is.
    assert_eq!(result.added[0].library, LookLibrary::User);
    assert_eq!(result.added[0].format, Some(LookFormat::Alf4));
    assert_eq!(result.added[0].lut_size, Some(5));
    // The same file again changes nothing.
    let again = h.api.looks_import(&path, None).expect("again");
    assert!(again.added.is_empty() && again.unchanged == 1);

    // An .aml is a LogC3-to-Rec.709 look.
    let aml = write_look_file(dir, "Night.aml", "aml", false);
    let seen = h.api.looks_recognise(&aml.to_string_lossy());
    assert!(matches!(
        &seen.recognition,
        Recognition::LookFiles { what, input: Some(LookInput::Logc3ToRec709), .. }
            if what == "ARRI Look Files · LogC3 to Rec.709"
    ));
    let result = h
        .api
        .looks_import(&aml.to_string_lossy(), None)
        .expect("aml");
    assert_eq!(result.added[0].input, LookInput::Logc3ToRec709);
    // An .aml of the person's own stays in Your LUTs (ARRI's ALF2 section is for ARRI's library).
    assert_eq!(result.added[0].library, LookLibrary::User);
    assert_eq!(result.added[0].format, Some(LookFormat::Aml));

    // An .alf4c says its display transforms are not used.
    let alf4c = write_look_file(dir, "Custom.alf4c", "alf4c", false);
    let seen = h.api.looks_recognise(&alf4c.to_string_lossy());
    match &seen.recognition {
        Recognition::LookFiles { note, contents, .. } => {
            let note = note.as_deref().expect("a note");
            assert!(note.contains("display transforms are not used"), "{note}");
            assert_eq!(contents, "1 .alf4c · 5-point");
        }
        other => panic!("expected look files, got {other:?}"),
    }

    // A folder of both kinds: one row, the input differs, so none is claimed for the row.
    let folder = dir.join("Mine");
    std::fs::create_dir_all(&folder).expect("mkdir");
    write_look_file(&folder, "A.alf4", "alf4", false);
    write_look_file(&folder, "B.aml", "aml", false);
    let seen = h.api.looks_recognise(&folder.to_string_lossy());
    assert_eq!(seen.source, SourceKind::Folder);
    assert!(matches!(
        &seen.recognition,
        Recognition::LookFiles { what, contents, file_count: 2, input: None, .. }
            if what == "ARRI Look Files" && contents == "1 .alf4 · 1 .aml · 5-point"
    ));

    // A damaged file is explained, not imported.
    let mut bytes = std::fs::read(&alf4).expect("read");
    let at = bytes.len() - 3;
    bytes[at] ^= 0x20;
    let damaged = dir.join("Damaged.alf4");
    std::fs::write(&damaged, bytes).expect("write");
    match h
        .api
        .looks_recognise(&damaged.to_string_lossy())
        .recognition
    {
        Recognition::Unusable { reason, .. } => {
            assert!(reason.contains("damaged"), "{reason}");
        }
        other => panic!("expected unusable, got {other:?}"),
    }
}

#[test]
fn the_look_file_libraries_are_the_packages_they_are_the_looks_of() {
    assert_eq!(
        PackageId::from_data_id("look-library-logc4-alf4"),
        Some(PackageId::LookLibraryLogc4)
    );
    assert_eq!(
        PackageId::from_data_id("look-library-logc3-rec709-alf2"),
        Some(PackageId::LookLibraryLogc3ToRec709)
    );
    // The ids of the 3D-LUT packages are as before, and an unknown id is none.
    assert_eq!(
        PackageId::from_data_id("look-library-logc4"),
        Some(PackageId::LookLibraryLogc4)
    );
    assert_eq!(PackageId::from_data_id("look-library-logc5"), None);
    for id in [
        PackageId::LookLibraryLogc4,
        PackageId::DisplayLutsLogc4,
        PackageId::LookLibraryLogc3,
        PackageId::LookLibraryLogc3ToRec709,
        PackageId::DisplayLutsLogc3,
    ] {
        assert_eq!(PackageId::from_data_id(id.data_id()), Some(id));
    }
}

#[test]
fn the_alf4_library_reads_as_the_logc4_look_library_with_its_own_numbers() {
    use std::io::Write as _;
    // The shipped package list, shrunk to a library of three looks of 5 points (as the tests of
    // `ym-looks` do), and a zip laid out and named as ARRI's ALF4 download is.
    let mut db = ym_looks::recognize::PackageDb::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/looks/arri-packages.json"),
    )
    .expect("arri-packages.json");
    for package in &mut db.packages {
        package.fingerprints.clear();
    }
    let package = db
        .packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc4-alf4")
        .expect("the ALF4 package");
    package.expected.cube_files = Some(3);
    package.expected.lut_sizes = vec![5];
    let work = tempfile::tempdir().expect("tempdir");
    let zip_path = work
        .path()
        .join("arri-look-library-logc4-log-to-log-alf4-data.zip");
    {
        let lut = ym_color::lut::Lut3d::identity(5).expect("identity cube");
        let file = std::fs::File::create(&zip_path).expect("zip");
        let mut zip = zip::ZipWriter::new(file);
        for (i, (group, code)) in [
            ("1000 Black-and-White", 1010),
            ("3000 Film", 3010),
            ("4000 Tinted", 4010),
        ]
        .into_iter()
        .enumerate()
        {
            zip.start_file(
                format!(
                    "ARRI Look Library LogC4 (log-to-log) ALF4/{group}/LL-{code}-Look-{i}.alf4"
                ),
                zip::write::SimpleFileOptions::default(),
            )
            .expect("entry");
            let bytes = ym_color::alf::AlfFile::alf4(&lut, None, "", "x.cube")
                .to_bytes()
                .expect("write");
            zip.write_all(&bytes).expect("bytes");
        }
        zip.finish().expect("finish");
    }
    let seen = crate::wizard::recognise(&db, &zip_path);
    match seen.recognition {
        Recognition::Package {
            package_id,
            title,
            what,
            contents,
            file_count,
            ..
        } => {
            assert_eq!(package_id, PackageId::LookLibraryLogc4);
            assert_eq!(title, "Look Library · LogC4");
            assert_eq!(what, "Look Library · LogC4 · log to log");
            assert_eq!(contents, "3 .alf4 · 5-point");
            assert_eq!(file_count, 3);
        }
        other => panic!("expected the package, got {other:?}"),
    }
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
    assert_eq!(prefs.default_look.0, "builtin:signature");
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

/// An advisor that is always available and gives the same answer, or the same failure, for every
/// picture (`docs/19` section 12).
struct Scripted(Result<ym_core::apple_auto::AdvisorAnswer, ym_core::apple_auto::AdvisorError>);

impl ym_core::apple_auto::AutoAdvisor for Scripted {
    fn availability(&self) -> ym_core::apple_auto::Availability {
        ym_core::apple_auto::Availability::Available(ym_core::apple_auto::AdvisorInfo::new(
            "Scripted", "test",
        ))
    }

    fn ask(
        &self,
        _: &ym_core::apple_auto::AdvisorPicture,
        _: &str,
    ) -> Result<ym_core::apple_auto::AdvisorAnswer, ym_core::apple_auto::AdvisorError> {
        self.0.clone()
    }
}

fn apple_state(h: &Harness, id: PhotoId) -> AppleAuto {
    h.api.get_settings(id).expect("settings").estimates.apple
}

fn prefs_patch(value: serde_json::Value) -> PreferencesPatch {
    serde_json::from_value(value).expect("preferences patch")
}

#[test]
fn apple_auto_reaches_the_webview_through_the_settings_the_status_and_settings_changed() {
    use ym_core::apple_auto::{AdvisorAnswer, AutoSteps, ShadowColor};
    let answer = AdvisorAnswer {
        steps: AutoSteps {
            exposure: Some(0),
            white: Some(0),
            contrast: Some(2),
            saturation: Some(-1),
            highlights: Some(-1),
            shadow_color: Some(ShadowColor::Teal),
            shadow_color_strength: Some(2),
            ..AutoSteps::default()
        },
        scene: "A grey gradient".into(),
        plan: "Teal shadows, more contrast".into(),
    };
    let h = Harness::with_advisor(Arc::new(Scripted(Ok(answer))));
    let status = h.api.apple_auto_status().expect("status");
    assert!(status.available && status.active && status.preference);
    assert_eq!(status.reason, None);
    let sha = ym_core::apple_auto::InstructionsId::default().instructions_sha256;
    assert_eq!(
        status.info,
        Some(AppleAutoInfo {
            model: "Scripted".into(),
            os: "test".into(),
            instructions: crate::wire::AppleAutoInstructions::Default,
            instructions_sha256: sha.clone(),
        })
    );
    assert_eq!(status.mode, AppleAutoMode::FromPreference);
    assert_eq!((status.prompt, status.table), (3, 2));
    assert_eq!(status.own_instructions, "");
    assert_eq!(
        status.default_instructions,
        ym_core::apple_auto::DEFAULT_INSTRUCTIONS
    );
    assert_eq!(status.instructions_max_chars, 2000);

    h.open(1);
    h.develop();
    let id = h.first();
    // The viewer does not wait for the answer; when it arrives the webview is told to read the
    // photo's settings again.
    wait_until("the answer", || {
        matches!(apple_state(&h, id), AppleAuto::Answered { .. })
    });
    wait_until("settings_changed for the answer", || {
        h.events.events().contains(&AppEvent::SettingsChanged(id))
    });
    let rs = h.api.get_settings(id).expect("settings");
    let AppleAuto::Answered { answer, info } = &rs.estimates.apple else {
        panic!("answered: {:?}", rs.estimates.apple);
    };
    assert_eq!(
        answer.steps,
        AppleAutoSteps {
            exposure: Some(0),
            highlights: Some(-1),
            shadows: None,
            blacks: None,
            contrast: Some(2),
            white: Some(0),
            tint: None,
            saturation: Some(-1),
            shadow_color: Some(crate::wire::AppleShadowColor::Teal),
            shadow_color_strength: Some(2),
            highlight_color: None,
            highlight_color_strength: None,
        }
    );
    assert_eq!(answer.scene, "A grey gradient");
    assert_eq!(answer.plan, "Teal shadows, more contrast");
    assert_eq!(rs.estimates.apple_steps, Some(answer.steps));
    assert_eq!(rs.estimates.highlights, Num(-20.0));
    assert_eq!(rs.values.highlights, Num(-20.0));
    assert_eq!(rs.settings.highlights, PercentSetting::Auto);
    assert!(rs.auto.highlights && rs.auto.balance_shadows);
    assert_eq!(
        rs.values.color_balance.shadows,
        [Num(-0.866 * 0.55), Num(-0.5 * 0.55)]
    );
    assert_eq!(info.model, "Scripted");
    assert_eq!(rs.estimates.source, AutoSource::Apple);
    assert_eq!(
        (rs.estimates.contrast, rs.estimates.saturation),
        (Num(30.0), Num(-15.0))
    );
    assert_eq!(rs.settings.contrast, PercentSetting::Auto);
    assert!(rs.auto.contrast && rs.auto.saturation);
    assert_eq!(
        (rs.values.contrast, rs.values.saturation),
        (Num(30.0), Num(-15.0))
    );

    // A manual value wins over the answer; Re-run puts contrast and saturation back to Auto.
    let manual: SettingsPatch = serde_json::from_value(json!({
        "contrast": { "kind": "manual", "value": -10 },
        "saturation": { "kind": "manual", "value": 5 }
    }))
    .expect("patch");
    let edited = h.api.update_settings(id, &manual).expect("update");
    assert!(!edited.auto.contrast && !edited.auto.saturation);
    assert_eq!(
        (edited.values.contrast, edited.values.saturation),
        (Num(-10.0), Num(5.0))
    );
    assert_eq!(edited.estimates.contrast, Num(30.0), "the Auto value stays");
    let rerun = h.api.rerun_auto(id).expect("rerun");
    assert!(rerun.auto.contrast && rerun.auto.saturation);
    assert_eq!(rerun.settings.saturation, PercentSetting::Auto);
    wait_until("the answer after Re-run", || {
        matches!(apple_state(&h, id), AppleAuto::Answered { .. })
    });

    // The preference off: every photo back to the rule at once, and the webview is told.
    let asks = h.api.core().session.apple_auto_asks();
    let before = h.events.events().len();
    let prefs = h
        .api
        .preferences_set(&prefs_patch(json!({ "apple_auto": false })))
        .expect("off");
    assert!(!prefs.apple_auto);
    let off = h.api.get_settings(id).expect("settings");
    assert_eq!(off.estimates.apple, AppleAuto::Off);
    assert_eq!(off.estimates.source, AutoSource::Rule);
    assert_eq!(
        (off.values.contrast, off.values.saturation),
        (Num(0.0), Num(0.0))
    );
    wait_until("settings_changed for the preference", || {
        h.events.events()[before..].contains(&AppEvent::SettingsChanged(id))
    });
    let status = h.api.apple_auto_status().expect("status");
    assert!(status.available && !status.active && !status.preference);

    // On again: the answer was kept, so nothing is asked again.
    h.api
        .preferences_set(&prefs_patch(json!({ "apple_auto": true })))
        .expect("on");
    assert!(matches!(apple_state(&h, id), AppleAuto::Answered { .. }));
    assert_eq!(h.api.core().session.apple_auto_asks(), asks);
    assert!(h.api.preferences_get().expect("prefs").apple_auto);

    // New instructions: saved, every photo asked again; too long: refused with a plain message.
    let asks = h.api.core().session.apple_auto_asks();
    let err = h
        .api
        .preferences_set(&prefs_patch(
            json!({ "apple_auto_instructions": "x".repeat(2001) }),
        ))
        .expect_err("too long");
    assert!(format!("{err:?}").contains("At most 2000"), "{err:?}");
    let prefs = h
        .api
        .preferences_set(&prefs_patch(
            json!({ "apple_auto_instructions": "Moody and cool." }),
        ))
        .expect("instructions");
    assert_eq!(prefs.apple_auto_instructions, "Moody and cool.");
    wait_until("the answer to the new instructions", || {
        h.api.core().session.apple_auto_asks() > asks
            && matches!(apple_state(&h, id), AppleAuto::Answered { .. })
    });
    let status = h.api.apple_auto_status().expect("status");
    assert_eq!(
        status.instructions,
        crate::wire::AppleAutoInstructions::Edited
    );
    assert_eq!(status.own_instructions, "Moody and cool.");
    let AppleAuto::Answered { info, .. } = apple_state(&h, id) else {
        panic!("answered");
    };
    assert_eq!(
        info.instructions,
        crate::wire::AppleAutoInstructions::Edited
    );
    // Back to the default: the preference is cleared.
    let prefs = h
        .api
        .preferences_set(&prefs_patch(json!({ "apple_auto_instructions": "" })))
        .expect("default");
    assert_eq!(prefs.apple_auto_instructions, "");
}

#[test]
fn a_failed_question_crosses_the_bridge_and_auto_stays_the_rule() {
    use ym_core::apple_auto::{AdvisorError, FailureKind};
    let h = Harness::with_advisor(Arc::new(Scripted(Err(AdvisorError::new(
        FailureKind::Refused,
        "Not this one.",
    )))));
    h.open(1);
    h.develop();
    let id = h.first();
    wait_until("the failure", || {
        matches!(apple_state(&h, id), AppleAuto::Failed { .. })
    });
    let rs = h.api.get_settings(id).expect("settings");
    assert_eq!(
        rs.estimates.apple,
        AppleAuto::Failed {
            kind: AppleAutoFailure::Refused,
            message: "Not this one.".into(),
            info: AppleAutoInfo {
                model: "Scripted".into(),
                os: "test".into(),
                instructions: crate::wire::AppleAutoInstructions::Default,
                instructions_sha256: ym_core::apple_auto::InstructionsId::default()
                    .instructions_sha256,
            }
        }
    );
    assert_eq!(rs.estimates.source, AutoSource::Rule);
    assert_eq!(
        (rs.estimates.contrast, rs.estimates.saturation),
        (Num(0.0), Num(0.0))
    );
}

#[test]
fn without_an_advisor_apple_auto_is_off_and_its_row_stays_hidden() {
    let h = Harness::new();
    let status = h.api.apple_auto_status().expect("status");
    assert!(!status.available && !status.active);
    assert_eq!(status.reason, None);
    assert_eq!(status.info, None);
    h.open(1);
    h.develop();
    let rs = h.api.get_settings(h.first()).expect("settings");
    assert_eq!(rs.estimates.apple, AppleAuto::Off);
    assert_eq!(rs.estimates.source, AutoSource::Rule);
    assert_eq!(rs.settings.contrast, PercentSetting::Auto);
    assert!(rs.auto.contrast && rs.auto.saturation);
}

#[test]
fn the_system_advisor_off_apple_silicon_says_why_it_is_not_available() {
    // What `AppConfig::from_environment` gives the app. Off Apple silicon the reason is the first
    // of `docs/19` section 1; on Apple silicon the answer depends on the Mac and the SDK.
    if std::env::var_os("YELLOWMOON_TEST_APPLE_AUTO").is_some()
        || cfg!(all(target_os = "macos", target_arch = "aarch64"))
    {
        return;
    }
    let h = Harness::with_advisor(ym_apple::system_advisor());
    let status = h.api.apple_auto_status().expect("status");
    assert!(!status.available && !status.active);
    assert!(
        matches!(
            status.reason,
            Some(AppleAutoReason::NotAMac | AppleAutoReason::IntelMac)
        ),
        "{:?}",
        status.reason
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
fn quitting_while_an_export_is_writing_deletes_the_partial_file_before_the_process_ends() {
    // `RunEvent::Exit` calls `AppApi::shutdown`, and the process ends as soon as that returns: a
    // worker that was still writing would never delete its hidden `.yellowmoon-*.tmp`.
    let h = Harness::new();
    let (w, hh) = (2400_u32, 1600_u32);
    ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", w, hh)
        .black([512; 4])
        .white([16_300; 3])
        .wb(Some([2_100, 1_024, 1_024, 1_600]))
        .iso(400)
        .write_to(
            &h.photos.path().join("DSC00001.ARW"),
            &ym_raw::testgen::gradient_mosaic(w, hh, 512, 12_000, 7),
        )
        .expect("write the ARW");
    h.api
        .open_paths(&[h.photos.path().to_string_lossy().into_owned()])
        .expect("open");
    h.develop();
    let out = tempfile::tempdir().expect("out");
    let mut job = jpeg_job(out.path());
    job.options.format = ExportFormat::Png;
    job.options.bit_depth = BitDepth(16);
    h.api.export_start(&job).expect("start");

    let names = || -> Vec<String> {
        std::fs::read_dir(out.path())
            .expect("dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect()
    };
    wait_until("the encoder to start writing", || {
        names().iter().any(|n| n.starts_with(".yellowmoon-"))
    });
    h.api.shutdown();
    assert_eq!(names(), Vec::<String>::new(), "nothing is left behind");
    assert!(!h.api.export_busy());
}

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
    assert!(!estimate.dng_approximate);
    let command = h.api.cli_preview(&job).expect("preview");
    assert!(command.starts_with("ym render "), "{command}");

    let id = h.api.export_start(&job).expect("start");
    wait_until("the export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == id && p.done == p.total && p.total == 2)
        })
    });
    wait_until("the queue to go idle", || !h.api.export_busy());
    wait_for_the_free_event(&h);
    let busy = busy_events(&h);
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
        // The app under test has no advisor, so Apple Auto is off there; off for `ym` too, which
        // would otherwise follow the preference on a Mac with Apple Intelligence (`docs/19` section 7).
        .args(["--apple-auto", "off"])
        .arg("--out")
        .arg(by_cli.path())
        .arg("--quiet")
        .env("YELLOWMOON_HOME", cli_home.path())
        .env("HOME", cli_home.path())
        // The app under test opens no device (`open_app_gpu(false, ..)`), so `ym` runs on the CPU
        // too: on a machine with a GPU, `auto` would put the raw chain on it. App and `ym` on the
        // same device are compared by `crates/ym-cli/tests/parity_gpu.rs`.
        .env("YELLOWMOON_GPU", "off")
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

/// The bound on the slowest command of the 500-photo test when nothing says otherwise: 100 ms, the
/// figure of M3-03.
const LATENCY_BOUND_DEFAULT: Duration = Duration::from_millis(100);

/// The environment variable that moves that bound, in whole milliseconds. A developer's machine
/// keeps the 100 ms; the `latency-500` job of `weekly.yml` sets a larger value because its runner
/// is shared with other jobs and the margin is thin (the slowest command took 92.6 ms in one run
/// and 61 ms in another, against 100). The numbers are printed either way, so the trend stays
/// visible.
const LATENCY_BOUND_VAR: &str = "YM_PERF_SHELL_MAX_MS";

/// The bound for the value of [`LATENCY_BOUND_VAR`] (`None` when it is not set). An empty value is
/// not set; one that is not a positive whole number of milliseconds is a mistake in the job, and
/// stops the test instead of silently measuring against the default.
fn latency_bound(setting: Option<&str>) -> Duration {
    match setting.map(str::trim).filter(|s| !s.is_empty()) {
        None => LATENCY_BOUND_DEFAULT,
        Some(text) => match text.parse::<u64>() {
            Ok(ms) if ms > 0 => Duration::from_millis(ms),
            _ => panic!(
                "{LATENCY_BOUND_VAR} is {text:?}: give a positive whole number of milliseconds"
            ),
        },
    }
}

#[test]
fn the_latency_bound_is_100_ms_unless_the_environment_relaxes_it() {
    assert_eq!(latency_bound(None), Duration::from_millis(100));
    assert_eq!(latency_bound(Some("")), Duration::from_millis(100));
    assert_eq!(latency_bound(Some("  ")), Duration::from_millis(100));
    assert_eq!(latency_bound(Some("250")), Duration::from_millis(250));
    assert_eq!(latency_bound(Some(" 80 ")), Duration::from_millis(80));
}

#[test]
fn a_latency_bound_that_is_not_a_number_of_milliseconds_stops_the_test() {
    for bad in ["fast", "0", "-5", "2.5", "100ms"] {
        let caught = std::panic::catch_unwind(|| latency_bound(Some(bad)));
        assert!(caught.is_err(), "{bad:?} was accepted");
    }
}

/// M3-03: "opening 500 files keeps the UI responsive". The shell's share of that: the commands the
/// webview sends while 500 photos develop answer quickly (the pool is busy with decoding, the
/// commands are not), and the events stay a bounded stream. Prints the numbers (`--nocapture`).
///
/// The slowest command must take less than 100 ms. `YM_PERF_SHELL_MAX_MS=<milliseconds>` moves
/// that bound for a runner that is shared (the weekly job sets 250); without it the bound is the
/// 100 ms of the task. Run it with `npm run perf:latency:shell` (or `npm run perf:latency` for
/// this half and the web view's half).
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
    let bound = latency_bound(std::env::var(LATENCY_BOUND_VAR).ok().as_deref());
    eprintln!("bound on the slowest command: {bound:?}");
    assert!(
        slowest < bound,
        "a command took {slowest:?} while photos were developing (the bound is {bound:?}; \
         {LATENCY_BOUND_VAR} moves it)"
    );
}

/// The `ExportBusy` events seen so far, in order.
fn busy_events(h: &Harness) -> Vec<bool> {
    h.events
        .events()
        .iter()
        .filter_map(|e| match e {
            AppEvent::ExportBusy(b) => Some(*b),
            _ => None,
        })
        .collect()
}

/// Waits until the app has been told that the queue is free: the worker clears its state first and sends
/// `ExportBusy(false)` just after, so a test that saw `export_busy()` turn false may still be a moment
/// ahead of the event.
fn wait_for_the_free_event(h: &Harness) {
    wait_until("the app to be told the queue is free", || {
        busy_events(h).last() != Some(&true)
    });
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
    wait_for_the_free_event(&h);
    let written = std::fs::read_dir(out.path()).expect("dir").count();
    assert_eq!(written, 4, "every photo of the job was written");
    let busy = busy_events(&h);
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
    wait_for_the_free_event(&h);
    // Whatever was written is a whole file, and no temporary file of a photo that was cut short is
    // left behind.
    for entry in std::fs::read_dir(out.path()).expect("dir") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(!name.starts_with('.') && !name.ends_with(".tmp"), "{name}");
        assert!(entry.metadata().expect("meta").len() > 0, "{name} is empty");
    }
    // A job cancelled before the worker took it never made the queue busy, so there may be no event
    // at all; if the app was told it is busy, it has been told it is free again.
    assert_ne!(busy_events(&h).last(), Some(&true));
}

#[test]
fn an_export_started_straight_after_opening_writes_what_one_after_developing_writes() {
    // The burst median and the pooled noise of a photo depend on the others. Straight after
    // opening the session is still developing; the export of the current photo is queued as soon
    // as that photo has estimates and its turn waits for the ones it depends on (here: all twelve
    // share a body and an ISO), so the file is the one an export after everything is developed
    // writes (and `ym render`, which waits for everything first).
    let h = Harness::new();
    h.open(12);
    let early = tempfile::tempdir().expect("early");
    let late = tempfile::tempdir().expect("late");
    let mut job = jpeg_job(early.path());
    job.scope = ApplyScope::This;
    // The first attempts can say that this very photo is not ready; once it is, the job is
    // queued although eleven others may still be developing.
    let first = h.first();
    let mut early_id = None;
    wait_until("the current photo to be exportable", || {
        match h.api.export_start(&job) {
            Ok(id) => {
                early_id = Some(id);
                true
            }
            Err(e) => {
                assert_eq!(e.scope, ErrorScope::Export);
                assert!(e.message.contains("still being developed"), "{e}");
                false
            }
        }
    });
    let early_id = early_id.expect("queued");
    wait_until("the early export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == early_id && p.done == p.total)
        })
    });
    h.develop();
    wait_until("the queue to go idle", || !h.api.export_busy());
    job.options.destination = Some(late.path().to_string_lossy().into_owned());
    assert_eq!(h.api.session_state().expect("state").photos[0].id, first);
    let late_id = h
        .api
        .export_start(&job)
        .expect("everything is developed now");
    wait_until("the late export to finish", || {
        h.events.events().iter().any(|e| {
            matches!(e, AppEvent::ExportProgress(p) if p.job_id == late_id && p.done == p.total)
        })
    });
    let bytes_of = |dir: &Path| {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|e| std::fs::read(e.expect("entry").path()).expect("read"))
            .collect();
        assert_eq!(files.len(), 1);
        files.remove(0)
    };
    assert!(
        bytes_of(early.path()) == bytes_of(late.path()),
        "an early export and a late one write the same bytes"
    );
}

#[test]
fn photos_without_estimates_are_refused_in_plain_words_and_nothing_is_queued() {
    let h = Harness::new();
    h.open(12);
    let out = tempfile::tempdir().expect("out");
    // All twelve: straight after opening some have no estimates yet. The answer is the sentence
    // that says so (or, on a machine that is quick enough, a job).
    match h.api.export_start(&jpeg_job(out.path())) {
        Err(e) => {
            assert_eq!(e.scope, ErrorScope::Export);
            assert!(
                e.message.contains("still being developed") && e.message.contains("export again"),
                "{e}"
            );
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
    wait_until("the queue to go idle", || !h.api.export_busy());
}

#[test]
fn the_app_does_not_restart_while_an_export_is_queued_or_running() {
    // "Update ready" restarts through `restart_app`, which asks `begin_restart` first. The
    // restart Tauri makes cannot be vetoed by the exit handler (`prevent_exit` is ignored for it),
    // so this is the only guard: an export that is queued or running refuses it.
    let h = Harness::new();
    h.open(12);
    h.develop();
    let mut refused = 0;
    let mut began = false;
    for round in 0..40 {
        let out = tempfile::tempdir().expect("out");
        h.api.export_start(&jpeg_job(out.path())).expect("queued");
        let busy_before = h.api.export_busy();
        let answer = h.api.begin_restart();
        let busy_after = h.api.export_busy();
        match answer {
            Err(e) => {
                assert!(
                    busy_before,
                    "round {round}: refused although nothing ran: {e}"
                );
                assert_eq!(e.scope, ErrorScope::Export);
                assert_eq!(
                    e.message,
                    "An export is still running. Restart after it finishes."
                );
                refused += 1;
                wait_until("the export to finish", || !h.api.export_busy());
            }
            Ok(()) => {
                // The export had finished by the time the question was asked (a very fast machine):
                // then nothing was running and the restart may begin.
                assert!(!busy_after, "round {round}: allowed while an export ran");
                began = true;
                break;
            }
        }
        if refused >= 3 {
            break;
        }
    }
    assert!(
        refused >= 1,
        "the restart was never refused while an export ran"
    );
    if !began {
        wait_until("the queue to go idle", || !h.api.export_busy());
        h.api
            .begin_restart()
            .expect("nothing runs: the restart may begin");
    }
    // Once a restart has begun no export is queued, so none is cut off by it.
    let out = tempfile::tempdir().expect("out");
    let err = h
        .api
        .export_start(&jpeg_job(out.path()))
        .expect_err("restarting");
    assert_eq!(err.scope, ErrorScope::Export);
    assert!(err.message.contains("restarting"), "{err}");
    assert!(!h.api.export_busy());
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
fn every_look_library_and_format_of_the_core_crosses_the_bridge_by_the_same_word() {
    use ym_looks::library::{LookFormat as CoreFormat, LookLibrary as CoreLibrary};
    let wire = [
        LookLibrary::BuiltIn,
        LookLibrary::ArriLogC4,
        LookLibrary::ArriLogC3,
        LookLibrary::ArriLogC3ToRec709,
        LookLibrary::ArriAlf2,
        LookLibrary::User,
    ];
    assert_eq!(wire.len(), CoreLibrary::ALL.len());
    for (core, wire) in CoreLibrary::ALL.into_iter().zip(wire) {
        let converted: LookLibrary = convert(&core).expect("library");
        assert_eq!(converted, wire, "{core:?}");
        assert_eq!(
            serde_json::to_value(wire).expect("json"),
            json!(core.slug()),
            "{core:?}"
        );
    }
    let formats = [
        (CoreFormat::Cube, LookFormat::Cube),
        (CoreFormat::Alf4, LookFormat::Alf4),
        (CoreFormat::Alf4c, LookFormat::Alf4c),
        (CoreFormat::Aml, LookFormat::Aml),
    ];
    for (core, wire) in formats {
        let converted: LookFormat = convert(&core).expect("format");
        assert_eq!(converted, wire, "{core:?}");
        assert_eq!(
            serde_json::to_value(wire).expect("json"),
            json!(core.extension()),
            "{core:?}"
        );
    }
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
        "output_sharpening": 35,
        "noise_reduction": { "kind": "manual", "luminance": 30, "color": 40, "detail": 50 },
        "light_source": "tungsten",
        "color_balance": { "midtones": [0.3, 0.1] },
        "crop": { "rect": { "x": 0.1, "y": 0.2, "width": 0.6, "height": 0.5 }, "angle_deg": -3, "aspect": "four_by_five" },
        "orientation": 270,
        "decode": { "neural_demosaic": false },
        "rolloff": { "highlights": 0.2 },
        "color_warp": { "pins": [
            { "source": [0.12, 0.03], "target": [0.08, 0.1], "locked": false, "range": 0.06,
              "shadows": 0.5, "highlights": 1, "pivot_ev": -1.5, "exposure_ev": 0.25 },
            { "source": [-0.05, -0.1], "target": [-0.05, -0.1], "locked": true, "range": 0.04,
              "shadows": 1, "highlights": 1, "pivot_ev": 0, "exposure_ev": 0 }
        ] }
    }))
    .expect("patch");
    h.api.update_settings(id, &patch).expect("update");

    let core = h.api.core();
    // The wire state also says what the last open did (`opened`); a plain `session_state` says nothing.
    let mut core_state = serde_json::to_value(core.session.state()).expect("core json");
    core_state["opened"] = serde_json::Value::Null;
    assert_same(
        "session_state",
        &core_state,
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
    let scatter = core
        .renderer
        .chroma_scatter(&src, &CancelToken::new())
        .expect("scatter");
    assert_same(
        "chroma_scatter",
        &scatter,
        &h.api.chroma_scatter(id).expect("scatter"),
    );
    let sample = ym_core::pipeline::ChromaSample {
        a: 0.125,
        b: -0.0625,
    };
    let wire: crate::wire::ChromaSample = convert(&sample).expect("sample");
    assert_same("sample_chroma", &sample, &wire);
    // A patch's warp crosses as the core's (the nested partial patches of the wire write their
    // absent parts as `null`, so only the warp is compared).
    let core_patch: ym_core::settings::SettingsPatch = convert(&patch).expect("core patch");
    assert_same(
        "settings_patch.color_warp",
        &core_patch.color_warp,
        &patch.color_warp,
    );
    assert_eq!(core_patch.color_warp.map(|w| w.pins.len()), Some(2));
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

/// An advisor whose exposure answers go +1, +3, +3 in the order it is asked about the photos (the rest keep).
struct Sequence(std::sync::atomic::AtomicUsize);

impl ym_core::apple_auto::AutoAdvisor for Sequence {
    fn availability(&self) -> ym_core::apple_auto::Availability {
        ym_core::apple_auto::Availability::Available(ym_core::apple_auto::AdvisorInfo::new(
            "Sequence", "test",
        ))
    }

    fn ask(
        &self,
        _: &ym_core::apple_auto::AdvisorPicture,
        _: &str,
    ) -> Result<ym_core::apple_auto::AdvisorAnswer, ym_core::apple_auto::AdvisorError> {
        let n = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ym_core::apple_auto::AdvisorAnswer {
            steps: ym_core::apple_auto::AutoSteps {
                exposure: Some([1, 3, 3][n % 3]),
                ..ym_core::apple_auto::AutoSteps::keep()
            },
            scene: format!("Photo {n}"),
            plan: String::new(),
        })
    }
}

#[test]
fn removing_a_frame_with_apple_auto_makes_the_burst_steps_again_tells_the_webview_once_and_undo_keeps_the_answers()
 {
    // The three photos have one capture time, so they are one burst; the answers are +1, +3 and +3 EV steps, whose middle is +3.
    let h = Harness::with_advisor(Arc::new(Sequence(std::sync::atomic::AtomicUsize::new(0))));
    h.open(3);
    let ids: Vec<PhotoId> = h.develop().photos.iter().map(|p| p.id).collect();
    wait_until("the three answers", || {
        ids.iter()
            .all(|id| matches!(apple_state(&h, *id), AppleAuto::Answered { .. }))
    });
    let step = |id: PhotoId| {
        h.api
            .get_settings(id)
            .expect("settings")
            .estimates
            .apple_steps
            .and_then(|s| s.exposure)
    };
    assert!(
        ids.iter().all(|id| step(*id) == Some(3)),
        "the burst's middle"
    );
    let own: Vec<PhotoId> = ids
        .iter()
        .copied()
        .filter(|id| match apple_state(&h, *id) {
            AppleAuto::Answered { answer, .. } => answer.steps.exposure == Some(3),
            _ => false,
        })
        .collect();
    assert_eq!(own.len(), 2, "two frames were answered +3");
    let victim = own[0];
    let stays: Vec<PhotoId> = ids.iter().copied().filter(|id| *id != victim).collect();
    let asks = h.api.core().session.apple_auto_asks();
    let before = h.events.events().len();

    let change = h.api.remove_photos(&[victim]).expect("remove");
    assert_eq!(change.photo_ids, vec![victim]);
    // The frames that stay share what is left: +1 and +3, whose lower middle is +1.
    assert!(stays.iter().all(|id| step(*id) == Some(1)));
    assert!(h.api.get_settings(victim).is_err());
    // Each is told once: the core's event reaches the webview as `settings_changed` and the command does not say it again.
    let told = |id: PhotoId, from: usize| {
        h.events.events()[from..]
            .iter()
            .filter(|e| **e == AppEvent::SettingsChanged(id))
            .count()
    };
    for id in &stays {
        assert_eq!(told(*id, before), 1, "{id:?}");
    }
    assert_eq!(told(victim, before), 0);

    // Undo: the photo is back with its answer and nobody is asked again; the burst's middle is +3 again.
    let before = h.events.events().len();
    let back = h.api.restore_photos().expect("restore");
    assert_eq!(back.photo_ids, vec![victim]);
    assert!(h.api.core().session.wait_idle(WAIT));
    assert!(matches!(
        apple_state(&h, victim),
        AppleAuto::Answered { .. }
    ));
    assert_eq!(h.api.core().session.apple_auto_asks(), asks);
    assert!(ids.iter().all(|id| step(*id) == Some(3)));
    for id in &stays {
        assert_eq!(told(*id, before), 1, "{id:?}");
    }
}
