//! Removing photos from the session (Remove from filmstrip) and putting them back (Undo): the filmstrip, the current
//! photo, the jobs and events of the photos that left, the caches, the picks and settings that come back exactly, the
//! estimates the remaining photos share, Apple Auto's answers (kept with a removed photo, not asked for while it is out),
//! and the files on disk, which are never touched.

mod common;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, SystemTime};

use common::{GateThumbnails, Layout, Scene, base_env, light, session_with_events, settle};
use ym_core::apple_auto::{AdvisorPicture, AppleAuto, AppleAutoMode, FakeAdvisor};
use ym_core::auto::NoiseSource;
use ym_core::session::{
    AppleAutoEnv, EventSink, OpenOptions, PhotoId, PhotoStatus, PictureInput, PictureRenderer,
    Session, SessionError, SessionEvent,
};
use ym_core::settings::{PercentSetting, SettingsPatch};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn write_plain(dir: &Path, name: &str, level: f64) -> PathBuf {
    Scene::new(256, 192, light(PROFILE, 5200.0), Layout::Patchwork, level)
        .noise(2e-4, 1e-6)
        .write(dir, name, MODEL)
}

/// `n` photos `p0.ARW`, `p1.ARW`, … opened and developed.
fn developed(n: usize) -> (tempfile::TempDir, Session, Vec<PhotoId>) {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..n {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.2 + 0.05 * i as f64);
    }
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    assert_eq!(ids.len(), n);
    (dir, session, ids)
}

fn names(session: &Session) -> Vec<String> {
    session
        .state()
        .photos
        .iter()
        .map(|p| p.file_name.clone())
        .collect()
}

fn photo_events(rx: &mpsc::Receiver<SessionEvent>) -> Vec<PhotoId> {
    rx.try_iter()
        .map(|e| match e {
            SessionEvent::PhotoProgress(p) => p.id,
            SessionEvent::ThumbnailReady(t) => t.id,
            SessionEvent::EstimatesChanged { id } => id,
        })
        .collect()
}

#[test]
fn the_current_photo_hands_over_to_the_next_one_then_to_the_previous_one_then_to_none() {
    let (dir, session, ids) = developed(4);
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
    session.set_current(b).unwrap();

    // The current photo is removed: the next one is shown.
    let out = session.remove(&[b]);
    assert_eq!(out.removed, vec![b]);
    assert_eq!(out.current, Some(c));
    assert_eq!(names(&session), ["p0.ARW", "p2.ARW", "p3.ARW"]);
    assert_eq!(session.current(), Some(c));
    assert_eq!(session.state().current_index, Some(1));
    // The removed photo is gone for every command.
    assert!(matches!(
        session.set_current(b),
        Err(SessionError::UnknownPhoto(_))
    ));
    assert!(matches!(
        session.view(b),
        Err(SessionError::UnknownPhoto(_))
    ));
    assert!(matches!(
        session.set_picked(b, true),
        Err(SessionError::UnknownPhoto(_))
    ));
    assert!(session.status(b).is_none() && session.summary(b).is_none());

    // The last photo is removed while it is current: the previous one is shown.
    session.set_current(d).unwrap();
    let out = session.remove(&[d]);
    assert_eq!(out.current, Some(c));
    assert_eq!(session.state().current_index, Some(1));

    // Another photo than the current one: the current photo stays, whatever its place becomes.
    let out = session.remove(&[a]);
    assert_eq!(out.current, Some(c));
    assert_eq!(session.state().current_index, Some(0));

    // The last photo: nothing is left, and the session is empty at once.
    let out = session.remove(&[c]);
    assert_eq!(out.current, None);
    let state = session.state();
    assert!(state.photos.is_empty() && state.picked.is_empty());
    assert_eq!((state.current_index, state.folder_name), (None, None));
    assert_eq!(session.current(), None);
    assert!(session.cached_ids().is_empty());
    // Removing nothing, or what is not there, is not an error and changes nothing.
    assert_eq!(session.remove(&[c, PhotoId(999)]), Default::default());
    assert_eq!(session.remove(&[]), Default::default());

    // The session is usable: photos opened now are current and develop.
    let again = session.open(&[dir.path().join("p1.ARW")], OpenOptions::default());
    settle(&session);
    assert_eq!(again.added.len(), 1);
    assert_eq!(session.current(), Some(again.added[0]));
    assert_eq!(session.status(again.added[0]), Some(PhotoStatus::Ready));
}

#[test]
fn several_photos_go_at_once_and_the_picks_go_with_them() {
    let (_dir, session, ids) = developed(6);
    for &id in &[ids[0], ids[2], ids[3], ids[5]] {
        session.set_picked(id, true).unwrap();
    }
    session.set_current(ids[2]).unwrap();
    // Ids may repeat, may not exist and need not be in order.
    let out = session.remove(&[ids[3], ids[2], ids[3], PhotoId(777), ids[0]]);
    assert_eq!(
        out.removed,
        vec![ids[0], ids[2], ids[3]],
        "in filmstrip order"
    );
    assert_eq!(names(&session), ["p1.ARW", "p4.ARW", "p5.ARW"]);
    // The next photo that stays after the current one (p2) is p4.
    assert_eq!(out.current, Some(ids[4]));
    assert_eq!(session.picked(), vec![ids[5]]);
    assert_eq!(session.state().picked, vec![ids[5]]);
}

#[test]
fn removing_leaves_the_files_on_disk_as_they_were() {
    let (dir, session, ids) = developed(3);
    let files: Vec<PathBuf> = (0..3)
        .map(|i| dir.path().join(format!("p{i}.ARW")))
        .collect();
    let before: Vec<(Vec<u8>, SystemTime)> = files
        .iter()
        .map(|f| {
            (
                std::fs::read(f).unwrap(),
                std::fs::metadata(f).unwrap().modified().unwrap(),
            )
        })
        .collect();
    let listing = common::names_in(dir.path());
    // Let the clock move on, so that a write would show in the modification time.
    std::thread::sleep(Duration::from_millis(30));

    session.remove(&[ids[1]]);
    session.remove(&[ids[0], ids[2]]);
    assert!(session.state().photos.is_empty());
    session.close();

    assert_eq!(
        common::names_in(dir.path()),
        listing,
        "nothing was deleted or renamed"
    );
    for (file, (bytes, modified)) in files.iter().zip(&before) {
        assert_eq!(&std::fs::read(file).unwrap(), bytes, "{}", file.display());
        assert_eq!(
            &std::fs::metadata(file).unwrap().modified().unwrap(),
            modified,
            "{}",
            file.display()
        );
    }
}

#[test]
fn undo_puts_the_photos_back_where_they_were_with_their_settings_picks_and_estimates() {
    let (_dir, session, ids) = developed(5);
    // Edits on the photos that will go, a pick, and a different current photo.
    session
        .update_settings(ids[1], &SettingsPatch::exposure_ev(0.7))
        .unwrap();
    session
        .update_settings(ids[3], &SettingsPatch::kelvin(4300.0))
        .unwrap();
    session
        .update_settings(
            ids[3],
            &SettingsPatch {
                contrast: Some(PercentSetting::manual_value(25.0)),
                look_intensity: Some(60.0),
                ..Default::default()
            },
        )
        .unwrap();
    session.set_picked(ids[1], true).unwrap();
    session.set_picked(ids[4], true).unwrap();
    session.set_current(ids[3]).unwrap();
    let json = |s: &Session, id: PhotoId| serde_json::to_value(s.settings(id).unwrap()).unwrap();
    let keep: Vec<_> = ids
        .iter()
        .map(|&id| (json(&session, id), session.estimates(id).unwrap()))
        .collect();
    let state_before = session.state();

    let out = session.remove(&[ids[1], ids[3]]);
    assert_eq!(out.current, Some(ids[4]), "the next photo that stays");
    assert_eq!(session.restorable(), vec![ids[1], ids[3]]);
    assert_eq!(names(&session), ["p0.ARW", "p2.ARW", "p4.ARW"]);
    assert_eq!(session.picked(), vec![ids[4]]);

    let back = session.restore_removed();
    assert_eq!(back.restored, vec![ids[1], ids[3]]);
    assert_eq!(back.already_open, 0);
    // Exactly where they were, the one on screen on screen again, the picks as before.
    assert_eq!(back.current, Some(ids[3]));
    assert_eq!(session.state(), state_before);
    assert_eq!(session.photo_ids(), ids);
    // Their settings come back as they were, resolved against the same estimates: the photos that stayed have the pooled
    // noise and the shared exposure of the whole set again, and the photos that were away have the very ones they took.
    for (id, (settings, estimates)) in ids.iter().zip(&keep) {
        assert_eq!(&json(&session, *id), settings, "{id}");
        let now = session.estimates(*id).unwrap();
        assert_eq!(now.noise.model, estimates.noise.model, "{id}");
        assert_eq!(now.exposure.ev, estimates.exposure.ev, "{id}");
    }
    for id in [ids[1], ids[3]] {
        let kept = &keep[ids.iter().position(|i| *i == id).unwrap()].1;
        assert!(
            std::sync::Arc::ptr_eq(&session.estimates(id).unwrap(), kept),
            "{id}"
        );
    }
    let r = session.settings(ids[3]).unwrap();
    assert_eq!((r.values.kelvin, r.values.contrast), (4300.0, 25.0));
    assert!(!r.auto.white_balance);
    assert_eq!(session.status(ids[1]), Some(PhotoStatus::Ready));
    // Undo is for the last removal, once.
    assert!(session.restorable().is_empty());
    assert_eq!(session.restore_removed(), Default::default());
}

#[test]
fn undo_comes_back_to_the_photo_that_was_on_screen_only_if_the_person_has_not_moved() {
    let (_dir, session, ids) = developed(4);
    session.set_current(ids[1]).unwrap();
    session.remove(&[ids[1]]);
    assert_eq!(session.current(), Some(ids[2]));
    // Moved on: the photo on screen stays.
    session.set_current(ids[3]).unwrap();
    let back = session.restore_removed();
    assert_eq!(back.restored, vec![ids[1]]);
    assert_eq!(back.current, Some(ids[3]));
    assert_eq!(
        session.state().current_index,
        Some(3),
        "p3 has its old place again"
    );

    // Not moved: it comes back to the photo it took away.
    session.remove(&[ids[1]]);
    assert_eq!(
        session.current(),
        Some(ids[3]),
        "the current photo was not removed"
    );
    let back = session.restore_removed();
    assert_eq!(back.current, Some(ids[3]));
    session.set_current(ids[0]).unwrap();
    session.remove(&[ids[0]]);
    let back = session.restore_removed();
    assert_eq!(back.current, Some(ids[0]));
}

#[test]
fn removing_every_photo_and_undoing_it_brings_back_the_folder_and_the_first_photo() {
    let (dir, session, ids) = developed(3);
    session.set_current(ids[2]).unwrap();
    let before = session.state();
    session.remove(&ids);
    assert!(session.state().photos.is_empty());
    let back = session.restore_removed();
    assert_eq!(back.restored, ids);
    assert_eq!(back.current, Some(ids[2]));
    assert_eq!(session.state(), before);
    assert_eq!(
        before.folder_name.as_deref(),
        dir.path().file_name().unwrap().to_str()
    );
}

#[test]
fn a_second_removal_replaces_the_first_and_closing_forgets_it() {
    let (_dir, session, ids) = developed(4);
    session.remove(&[ids[0]]);
    session.remove(&[ids[1]]);
    assert_eq!(session.restorable(), vec![ids[1]]);
    let back = session.restore_removed();
    assert_eq!(back.restored, vec![ids[1]]);
    assert_eq!(names(&session), ["p1.ARW", "p2.ARW", "p3.ARW"]);

    session.remove(&[ids[2]]);
    session.close();
    assert!(session.restorable().is_empty());
    assert_eq!(session.restore_removed(), Default::default());
    // A removal that took nothing out does not replace the one before it.
    let (_dir2, other, other_ids) = developed(2);
    other.remove(&[other_ids[0]]);
    assert_eq!(other.remove(&[PhotoId(5000)]), Default::default());
    assert_eq!(other.restorable(), vec![other_ids[0]]);
}

#[test]
fn files_opened_into_an_emptied_session_start_a_new_one_and_the_removal_is_forgotten() {
    let (dir, session, ids) = developed(3);
    session.remove(&ids);
    assert_eq!(session.restorable(), ids);
    // Opening files into the emptied session (the Start screen shows) is a new session: nothing is left to put back, and the
    // removed files open as new photos.
    let out = session.open(&[dir.path().join("p1.ARW")], OpenOptions::default());
    assert_eq!((out.added.len(), out.already_open), (1, 0));
    assert!(session.restorable().is_empty());
    assert_eq!(session.restore_removed(), Default::default());
    settle(&session);
    assert_eq!(names(&session), ["p1.ARW"]);
    // Files opened into a session that has photos do not end a removal.
    session.remove(&[out.added[0]]);
    assert_eq!(session.restorable(), out.added);
}

#[test]
fn a_file_that_was_added_again_stays_out_when_the_removal_is_undone() {
    let (dir, session, ids) = developed(3);
    session.remove(&[ids[0], ids[1]]);
    // The person adds one of the removed files again (it is not in the session, so it is added).
    let again = session.open(&[dir.path().join("p1.ARW")], OpenOptions::default());
    assert_eq!(again.added.len(), 1);
    assert_eq!(again.already_open, 0);
    settle(&session);
    let back = session.restore_removed();
    assert_eq!(back.restored, vec![ids[0]]);
    assert_eq!(back.already_open, 1);
    // Every file once.
    let mut files = names(&session);
    files.sort();
    assert_eq!(files, ["p0.ARW", "p1.ARW", "p2.ARW"]);
    // And a removed file can be opened again (it is not "already open").
    session.remove(&[ids[2]]);
    let reopened = session.open(&[dir.path().join("p2.ARW")], OpenOptions::default());
    assert_eq!(reopened.added.len(), 1);
    assert_eq!(reopened.already_open, 0);
}

#[test]
fn removing_frees_the_mosaics_and_decodes_the_neighbours_of_the_new_current_photo() {
    let (_dir, session, ids) = developed(5);
    session.set_current(ids[1]).unwrap();
    settle(&session);
    let cached: HashSet<PhotoId> = session.cached_ids().into_iter().collect();
    assert_eq!(cached, HashSet::from([ids[0], ids[1], ids[2]]));
    session.remove(&[ids[1], ids[2]]);
    // At once: nothing of the removed photos is held. The current photo is p3 now.
    assert!(
        session
            .cached_ids()
            .iter()
            .all(|id| *id != ids[1] && *id != ids[2])
    );
    settle(&session);
    let cached: HashSet<PhotoId> = session.cached_ids().into_iter().collect();
    assert_eq!(cached, HashSet::from([ids[0], ids[3], ids[4]]));
    // The photo of an ended render is not coming back into the cache.
    assert!(matches!(
        session.raw_image(ids[1]),
        Err(SessionError::UnknownPhoto(_))
    ));
    assert_eq!(session.cached_ids().len(), 3);
}

#[test]
fn removed_photos_are_not_developed_and_not_announced() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..6 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.3);
    }
    // One decode thread, and the first thumbnail parked: photo 1 is in its thumbnail stage, the other five wait.
    let gate = GateThumbnails::new(1);
    let (sink, rx) = EventSink::channel();
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_events(sink)
            .with_thumbnails(gate.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    assert_eq!(session.pending_jobs(), 6, "one running, five queued");
    let _ = photo_events(&rx);

    // Photo 1 (running), photos 3 and 5 (queued).
    let out = session.remove(&[ids[0], ids[2], ids[4]]);
    assert_eq!(out.removed.len(), 3);
    assert_eq!(
        session.pending_jobs(),
        1 + 3,
        "the queued jobs of the removed photos are dropped"
    );
    gate.release();
    settle(&session);

    // Photos 2, 4 and 6 are developed; the three that left were not, and nothing was announced for them, also not the
    // thumbnail that was being drawn when photo 1 left.
    let left: HashSet<PhotoId> = [ids[0], ids[2], ids[4]].into();
    let events = photo_events(&rx);
    assert!(!events.is_empty());
    assert!(
        events.iter().all(|id| !left.contains(id)),
        "events for removed photos: {events:?}"
    );
    assert_eq!(
        gate.calls(),
        4,
        "photo 1's thumbnail and the three that stayed"
    );
    assert_eq!(session.jobs_started(), 4);
    for id in [ids[1], ids[3], ids[5]] {
        assert_eq!(session.status(id), Some(PhotoStatus::Ready), "{id}");
    }
}

#[test]
fn a_photo_removed_while_it_develops_and_put_back_develops_once_more() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..3 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.3);
    }
    let gate = GateThumbnails::new(1);
    let (sink, rx) = EventSink::channel();
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_events(sink)
            .with_thumbnails(gate.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    let _ = photo_events(&rx);

    // Photo 1 is in its thumbnail stage; photo 2 waits. Both go and both come back at once.
    session.remove(&[ids[0], ids[1]]);
    let back = session.restore_removed();
    assert_eq!(back.restored, vec![ids[0], ids[1]]);
    // The photos that had not finished start again (the status says so at once).
    assert_eq!(session.status(ids[0]), Some(PhotoStatus::Waiting));
    assert_eq!(session.status(ids[1]), Some(PhotoStatus::Waiting));
    gate.release();
    settle(&session);

    for id in &ids {
        assert_eq!(session.status(*id), Some(PhotoStatus::Ready), "{id}");
    }
    // The job that was running when photo 1 left stopped when it came back: every photo announces its thumbnail once.
    let announced: Vec<PhotoId> = rx
        .try_iter()
        .filter_map(|e| match e {
            SessionEvent::ThumbnailReady(t) => Some(t.id),
            _ => None,
        })
        .collect();
    let mut sorted = announced.clone();
    sorted.sort();
    assert_eq!(sorted, ids, "one thumbnail each: {announced:?}");
}

/// Writes `n` ISO 800 frames of one camera that cover different parts of the intensity range, so that they pool their noise.
fn write_pool(dir: &Path, n: usize) {
    for i in 0..n {
        let layout = if i % 2 == 0 {
            Layout::Ramp
        } else {
            Layout::Patchwork
        };
        let level = 0.3 + 0.2 * (i % 3) as f64;
        let s = Scene::new(768, 512, light(PROFILE, 5200.0), layout, level)
            .noise(5e-4, 2e-6)
            .seed(i as u64 + 1);
        s.builder(MODEL)
            .iso(800)
            .write_to(&dir.join(format!("n{i}.ARW")), &s.mosaic())
            .unwrap();
    }
}

#[test]
fn the_pooled_noise_of_the_photos_that_stay_is_that_of_a_session_without_the_others() {
    let dir = tempfile::tempdir().unwrap();
    write_pool(dir.path(), 4);
    let (session, rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    for id in &ids {
        assert_eq!(
            session.estimates(*id).unwrap().noise.source,
            NoiseSource::Pooled { files: 4 }
        );
    }
    // The current photo is p0; the pool of the others changes with it gone.
    session.set_current(ids[2]).unwrap();
    let _ = rx.try_iter().count();
    let out = session.remove(&[ids[0], ids[3]]);
    assert_eq!(out.removed, vec![ids[0], ids[3]]);

    let (fresh, _rx) = session_with_events();
    fresh.open(
        &[dir.path().join("n1.ARW"), dir.path().join("n2.ARW")],
        OpenOptions::default(),
    );
    settle(&fresh);
    let fresh_ids = fresh.photo_ids();
    for (kept, want) in [ids[1], ids[2]].into_iter().zip(fresh_ids) {
        let got = session.estimates(kept).unwrap();
        let want = fresh.estimates(want).unwrap();
        assert_eq!(got.noise.source, NoiseSource::Pooled { files: 2 });
        assert_eq!(got.noise.model, want.noise.model, "{kept}");
    }
    // The current photo's noise model moved, so it is told (the others re-read it when they are shown).
    assert!(out.estimates_changed.contains(&ids[2]));

    // Undo: the pool of four is back.
    session.restore_removed();
    for id in &ids {
        assert_eq!(
            session.estimates(*id).unwrap().noise.source,
            NoiseSource::Pooled { files: 4 },
            "{id}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Apple Auto (`docs/19`), with the deterministic fake advisor
// ---------------------------------------------------------------------------------------------

/// A picture for every photo: flat mid grey, which the fake advisor answers (highlights -1, contrast +1, subtle teal shadows).
struct GreyPictures;

impl PictureRenderer for GreyPictures {
    fn render(&self, _input: &PictureInput<'_>) -> Result<AdvisorPicture, String> {
        Ok(AdvisorPicture {
            width: 2,
            height: 1,
            rgb: vec![128; 6],
        })
    }
}

/// `n` photos `p0.ARW`, … opened into a session with the fake advisor, in `mode`, and developed.
fn developed_with_apple(
    n: usize,
    mode: AppleAutoMode,
) -> (tempfile::TempDir, Session, Vec<PhotoId>) {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..n {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.2 + 0.05 * i as f64);
    }
    let (sink, _rx) = EventSink::channel();
    let session = Session::new(base_env().with_events(sink).with_apple_auto(AppleAutoEnv {
        mode,
        ..AppleAutoEnv::new(Arc::new(FakeAdvisor), Arc::new(GreyPictures))
    }));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    assert_eq!(ids.len(), n);
    (dir, session, ids)
}

fn answered(session: &Session, id: PhotoId) -> bool {
    matches!(
        session.estimates(id).map(|e| e.apple.clone()),
        Some(AppleAuto::Answered { .. })
    )
}

#[test]
fn a_removed_photo_keeps_its_answer_for_undo_and_nobody_is_asked_again() {
    let (_dir, session, ids) = developed_with_apple(3, AppleAutoMode::On);
    assert_eq!(session.apple_auto_asks(), 3);
    let before: Vec<_> = ids
        .iter()
        .map(|&id| {
            (
                session.estimates(id).unwrap(),
                serde_json::to_value(session.settings(id).unwrap()).unwrap(),
            )
        })
        .collect();
    assert!(before.iter().all(|(e, _)| e.apple_steps.is_some()));

    session.remove(&[ids[0], ids[2]]);
    assert!(session.estimates(ids[0]).is_none() && session.estimates(ids[2]).is_none());
    // Turning Apple Auto off and on while they are out asks only about the photos that are in the session.
    session.set_apple_auto_mode(AppleAutoMode::Off);
    session.set_apple_auto_mode(AppleAutoMode::On);
    settle(&session);
    assert_eq!(session.apple_auto_asks(), 3);

    session.restore_removed();
    settle(&session);
    assert_eq!(
        session.apple_auto_asks(),
        3,
        "the answers came back with the photos"
    );
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            session.estimates(id).unwrap().apple,
            before[i].0.apple,
            "{id}"
        );
        assert_eq!(
            serde_json::to_value(session.settings(id).unwrap()).unwrap(),
            before[i].1,
            "{id}"
        );
    }
}

#[test]
fn a_photo_removed_before_it_was_asked_is_asked_once_it_is_back() {
    // Apple Auto is off while the photos develop: nobody has an answer, and the question waits for it to be on.
    let (_dir, session, ids) = developed_with_apple(3, AppleAutoMode::Off);
    assert_eq!(session.apple_auto_asks(), 0);
    session.remove(&[ids[1]]);

    session.set_apple_auto_mode(AppleAutoMode::On);
    settle(&session);
    assert_eq!(
        session.apple_auto_asks(),
        2,
        "the photo that is out is not asked about"
    );
    assert!(answered(&session, ids[0]) && answered(&session, ids[2]));

    session.restore_removed();
    settle(&session);
    assert_eq!(session.apple_auto_asks(), 3, "back in the queue, and asked");
    assert!(answered(&session, ids[1]));
}
