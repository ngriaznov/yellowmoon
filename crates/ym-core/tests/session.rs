//! The in-memory session on synthetic ARW files: opening, progress, caches, estimates shared
//! between photos, settings commands and cancellation.

#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use common::{Layout, Scene, base_env, light, session_with_events, settle};
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_color::wb::LightSource;
use ym_core::auto::NoiseSource;
use ym_core::session::{
    ApplyScope, EventSink, OpenOptions, PhotoId, PhotoProgress, PhotoStage, PhotoStatus, Session,
    SessionError, SessionEvent, ThumbnailInput, ThumbnailRenderer, default_decode_threads,
};
use ym_core::settings::{
    AutoOr, ColorBalancePatch, Crop, CropAspect, CurvesPatch, DecodePatch, ExposureSetting,
    NoiseReductionManualPatch, NormRect, Orientation, RolloffPatch, SettingsPatch,
    SharpenManualPatch, WhiteBalanceManualPatch, WhiteBalanceValue,
};
use ym_looks::library::LookId;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{ArwBuilder, ContainerOnly, CrawEncoder, LosslessL, LosslessMs};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn scene(level: f64) -> Scene {
    Scene::new(384, 288, light(PROFILE, 5200.0), Layout::Patchwork, level).noise(2e-4, 1e-6)
}

fn write_plain(dir: &Path, name: &str, level: f64) -> PathBuf {
    scene(level).write(dir, name, MODEL)
}

fn write_ms(dir: &Path, name: &str) -> PathBuf {
    let (w, h) = (128u32, 96u32);
    let white = light(PROFILE, 5200.0);
    let mut rgb = Vec::new();
    for _ in 0..w * h {
        for c in 0..3 {
            rgb.push((1024.0 + 0.4 * white[c] / white[1] * 14000.0).round() as u16);
        }
    }
    let path = dir.join(name);
    ArwBuilder::new(MODEL, w, h)
        .lossless_ms(LosslessMs::new(
            ChromaSubsampling::Yuv420,
            rgb,
            w,
            h,
            [1.0 / white[0], 1.0, 1.0 / white[2]].map(|g| g as f32),
        ))
        .write_to(&path, &[])
        .unwrap();
    path
}

/// Per photo, the progress events in the order they arrived.
fn progress_by_photo(rx: &mpsc::Receiver<SessionEvent>) -> HashMap<PhotoId, Vec<PhotoProgress>> {
    let mut out: HashMap<PhotoId, Vec<PhotoProgress>> = HashMap::new();
    for e in rx.try_iter() {
        if let SessionEvent::PhotoProgress(p) = e {
            out.entry(p.id).or_default().push(p);
        }
    }
    out
}

fn ready_ids(session: &Session) -> Vec<PhotoId> {
    session
        .state()
        .photos
        .iter()
        .filter(|p| p.status == PhotoStatus::Ready)
        .map(|p| p.id)
        .collect()
}

#[test]
fn a_mixed_folder_opens_with_per_file_errors_that_block_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Good files, in several raw modes, with ARW in three spellings.
    write_plain(d, "a_plain.ARW", 0.3);
    write_plain(d, "b_lower.arw", 0.35);
    write_plain(d, "c_mixed.Arw", 0.4);
    scene(0.3)
        .builder(MODEL)
        .craw(CrawEncoder::new())
        .write_to(&d.join("d_craw.ARW"), &scene(0.3).mosaic())
        .unwrap();
    scene(0.3)
        .builder(MODEL)
        .payload(LosslessL::new().tile(128, 128))
        .write_to(&d.join("e_lossless.ARW"), &scene(0.3).mosaic())
        .unwrap();
    write_ms(d, "f_ms.ARW");
    // Files that cannot be developed.
    ArwBuilder::new("ILCE-7M5", 64, 64)
        .container_only(ContainerOnly::arw6(12))
        .write_to(&d.join("g_arw6.ARW"), &[])
        .unwrap();
    std::fs::write(
        d.join("h_corrupt.ARW"),
        b"this is not a raw file at all, only text",
    )
    .unwrap();
    let valid = std::fs::read(d.join("a_plain.ARW")).unwrap();
    std::fs::write(d.join("i_truncated.ARW"), &valid[..valid.len() / 3]).unwrap();
    // Files that are ignored: other formats, a macOS resource fork, a folder with one more ARW.
    std::fs::write(d.join("notes.txt"), "hello").unwrap();
    std::fs::write(d.join("photo.jpg"), b"\xff\xd8\xff\xe0").unwrap();
    std::fs::write(d.join("._a_plain.ARW"), b"\0\x05\x16\x07").unwrap();
    std::fs::create_dir(d.join("sub")).unwrap();
    write_plain(&d.join("sub"), "deep.ARW", 0.3);

    let (session, rx) = session_with_events();
    let outcome = session.open(&[d.to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 9, "{outcome:?}");
    assert_eq!(
        outcome.ignored, 3,
        "notes.txt, photo.jpg and the resource fork"
    );
    assert!(outcome.missing.is_empty() && outcome.message().is_none());
    settle(&session);

    let state = session.state();
    let by_name: HashMap<&str, &ym_core::session::PhotoSummary> = state
        .photos
        .iter()
        .map(|p| (p.file_name.as_str(), p))
        .collect();
    // Filmstrip order: by name, case-insensitive.
    let order: Vec<&str> = state.photos.iter().map(|p| p.file_name.as_str()).collect();
    let mut sorted = order.clone();
    sorted.sort_by_key(|n| n.to_lowercase());
    assert_eq!(order, sorted);
    assert_eq!(state.current_index, Some(0));
    assert_eq!(
        state.folder_name.as_deref(),
        d.file_name().unwrap().to_str()
    );

    for name in [
        "a_plain.ARW",
        "b_lower.arw",
        "c_mixed.Arw",
        "d_craw.ARW",
        "e_lossless.ARW",
        "f_ms.ARW",
    ] {
        assert_eq!(by_name[name].status, PhotoStatus::Ready, "{name}");
        assert_eq!(by_name[name].camera.profile_id, PROFILE, "{name}");
    }
    // Lossless M/S is flagged REDUCED; the others are not.
    assert!(by_name["f_ms.ARW"].reduced);
    assert!(!by_name["a_plain.ARW"].reduced);
    let card = session.camera_card(by_name["f_ms.ARW"].id).unwrap();
    assert!(
        card.reduced
            && card
                .reduced_note
                .unwrap()
                .starts_with("This file was saved in the camera as Lossless M or S.")
    );
    assert!(card.raw_mode.contains("Lossless M or S"));
    // The summary carries what the filmstrip and the export panel show.
    let a = by_name["a_plain.ARW"];
    assert_eq!((a.width_px, a.height_px), (384, 288));
    assert_eq!(a.iso, 400);
    assert!((a.shutter_s - 0.004).abs() < 1e-9 && (a.aperture_f - 4.0).abs() < 1e-9);
    assert_eq!(a.captured_at, "2026-10-04T12:03:44");
    assert_eq!(
        state.default_export_folder.as_deref(),
        Some(
            d.parent()
                .unwrap()
                .join(format!(
                    "{} export",
                    d.file_name().unwrap().to_str().unwrap()
                ))
                .to_str()
                .unwrap()
        )
    );

    // Failures carry the plain messages of docs/04 and stay in the filmstrip.
    let reason = |name: &str| match &by_name[name].status {
        PhotoStatus::Skipped { reason } => reason.clone(),
        other => panic!("{name}: {other:?}"),
    };
    assert_eq!(
        reason("g_arw6.ARW"),
        "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet. Lossless files from these cameras work. We're working on it."
    );
    assert_eq!(
        reason("h_corrupt.ARW"),
        "This isn't an ARW file. Yellowmoon opens Sony ARW files only."
    );
    assert!(
        reason("i_truncated.ARW").starts_with("This file is damaged or incomplete"),
        "{}",
        reason("i_truncated.ARW")
    );
    // A skipped photo has no estimates and says why.
    let id = by_name["h_corrupt.ARW"].id;
    assert!(matches!(session.view(id), Err(SessionError::Skipped(_))));
    assert!(matches!(
        session.settings(id),
        Err(SessionError::Skipped(_))
    ));

    // Failure events carry the reason; the others end Ready.
    let events = progress_by_photo(&rx);
    for p in &state.photos {
        let last = events[&p.id].last().unwrap();
        match &p.status {
            PhotoStatus::Skipped { reason } => assert_eq!(last.error.as_ref(), Some(reason)),
            _ => assert!(last.stage == PhotoStage::Thumbnail && last.done && last.error.is_none()),
        }
    }

    // Subfolders only with the recursive option, and the same files are not added twice.
    let again = session.open(&[d.to_path_buf()], OpenOptions::default());
    assert!(again.added.is_empty() && again.already_open == 9 && !again.no_arw());
    let deeper = session.open(&[d.to_path_buf()], OpenOptions { recursive: true });
    assert_eq!(deeper.added.len(), 1);
    assert_eq!(session.state().photos.last().unwrap().file_name, "deep.ARW");
    settle(&session);
    assert_eq!(session.status(deeper.added[0]), Some(PhotoStatus::Ready));
}

#[test]
fn a_file_listed_twice_in_one_call_is_not_counted_as_already_open() {
    let dir = tempfile::tempdir().unwrap();
    for (i, name) in ["a.ARW", "b.ARW", "c.ARW"].into_iter().enumerate() {
        write_plain(dir.path(), name, 0.2 + 0.1 * i as f64);
    }
    let (session, _rx) = session_with_events();
    // A folder and a file inside it, as Finder's list view hands over when both are selected.
    let both = [dir.path().to_path_buf(), dir.path().join("a.ARW")];
    let first = session.open(&both, OpenOptions::default());
    assert_eq!(first.added.len(), 3);
    assert_eq!(first.already_open, 0, "nothing was open before the drop");
    settle(&session);
    // The same drop again: each file counts once, however often it was listed.
    let again = session.open(&both, OpenOptions::default());
    assert!(again.added.is_empty());
    assert_eq!(again.already_open, 3);
}

#[test]
fn a_drop_without_arw_files_says_so() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.jpg"), b"x").unwrap();
    std::fs::write(dir.path().join("b.png"), b"x").unwrap();
    let (session, _rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert!(outcome.no_arw());
    assert_eq!(outcome.message(), Some("No ARW files in what you dropped."));
    assert_eq!(outcome.ignored, 2);
    assert!(session.state().photos.is_empty());
    assert_eq!(session.state().current_index, None);
    // Explicit non-ARW files, and paths that do not exist.
    let outcome = session.open(
        &[dir.path().join("a.jpg"), dir.path().join("nope.ARW")],
        OpenOptions::default(),
    );
    assert!(outcome.no_arw());
    assert_eq!(outcome.ignored, 1);
    assert_eq!(outcome.missing, vec![dir.path().join("nope.ARW")]);
}

#[test]
fn photos_report_their_progress_in_order() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..6 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.2 + 0.05 * f64::from(i));
    }
    std::fs::write(dir.path().join("p9.ARW"), b"junk").unwrap();
    let (session, rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 7);
    settle(&session);
    let events = progress_by_photo(&rx);
    let step = |stage, done| (stage, done);
    let healthy = vec![
        step(PhotoStage::Probe, true),
        step(PhotoStage::Decode, false),
        step(PhotoStage::Decode, true),
        step(PhotoStage::Estimate, false),
        step(PhotoStage::Estimate, true),
        step(PhotoStage::Thumbnail, false),
        step(PhotoStage::Thumbnail, true),
    ];
    let mut seen_ready = 0;
    for id in &outcome.added {
        let got: Vec<_> = events[id].iter().map(|p| (p.stage, p.done)).collect();
        match session.status(*id).unwrap() {
            PhotoStatus::Ready => {
                assert_eq!(got, healthy, "photo {id}");
                assert!(events[id].iter().all(|p| p.error.is_none()));
                seen_ready += 1;
            }
            PhotoStatus::Skipped { reason } => {
                // The probe already found it unusable: one event, with the reason.
                assert_eq!(got, vec![step(PhotoStage::Probe, true)]);
                assert_eq!(events[id][0].error, Some(reason));
            }
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(seen_ready, 6);
}

#[test]
fn a_decode_failure_after_a_good_probe_ends_the_sequence_with_the_reason() {
    // ARW 6.0: the probe reads the camera, the decode refuses.
    let dir = tempfile::tempdir().unwrap();
    ArwBuilder::new("ILCE-7M5", 64, 64)
        .container_only(ContainerOnly::arw6(12))
        .write_to(&dir.path().join("a7v.ARW"), &[])
        .unwrap();
    let (session, rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let events = &progress_by_photo(&rx)[&outcome.added[0]];
    let got: Vec<_> = events
        .iter()
        .map(|p| (p.stage, p.done, p.error.is_some()))
        .collect();
    assert_eq!(
        got,
        vec![
            (PhotoStage::Probe, true, false),
            (PhotoStage::Decode, false, false),
            (PhotoStage::Decode, true, true),
        ]
    );
    let camera = session.camera(outcome.added[0]).unwrap();
    assert!(!camera.is_supported());
    assert_eq!(camera.card().unsupported, camera.unsupported_message());
    // The default profile stands in for an A7 V, and the card says so.
    assert!(camera.stand_in);
    assert!(camera.card().note.unwrap().contains("borrows the A7R IV"));
}

struct Recorder {
    calls: AtomicUsize,
    started: AtomicBool,
    delay: Duration,
    seen: Mutex<Vec<(PhotoId, f64)>>,
    fail_for: Option<PhotoId>,
}

impl Recorder {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            started: AtomicBool::new(false),
            delay,
            seen: Mutex::new(Vec::new()),
            fail_for: None,
        })
    }
}

impl ThumbnailRenderer for Recorder {
    fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String> {
        self.started.store(true, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen
            .lock()
            .unwrap()
            .push((input.id, input.estimates.summary().exposure_ev));
        std::thread::sleep(self.delay);
        if (input.cancelled)() {
            return Err("cancelled".into());
        }
        if Some(input.id) == self.fail_for {
            return Err("The thumbnail could not be drawn.".into());
        }
        Ok(vec![0x89, b'P', b'N', b'G', input.id.0 as u8])
    }
}

#[test]
fn the_thumbnail_hook_runs_after_the_estimates_and_its_picture_is_announced() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..3 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.25);
    }
    let renderer = Recorder::new(Duration::ZERO);
    let (sink, rx) = EventSink::channel();
    let session = Session::new(
        base_env()
            .with_events(sink)
            .with_thumbnails(renderer.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    assert_eq!(renderer.calls.load(Ordering::SeqCst), 3);
    let events: Vec<SessionEvent> = rx.try_iter().collect();
    for id in session.photo_ids() {
        // thumbnail(start), the picture, thumbnail(done), in that order.
        let at = |f: &dyn Fn(&SessionEvent) -> bool| events.iter().position(f).unwrap();
        let start = at(
            &|e| matches!(e, SessionEvent::PhotoProgress(p) if p.id == id && p.stage == PhotoStage::Thumbnail && !p.done),
        );
        let picture = at(
            &|e| matches!(e, SessionEvent::ThumbnailReady(t) if t.id == id && t.png_bytes[..4] == [0x89, b'P', b'N', b'G']),
        );
        let done = at(
            &|e| matches!(e, SessionEvent::PhotoProgress(p) if p.id == id && p.stage == PhotoStage::Thumbnail && p.done),
        );
        assert!(start < picture && picture < done, "photo {id}");
    }
    // The renderer saw real estimates (the exposure of the scene).
    for (_, ev) in renderer.seen.lock().unwrap().iter() {
        assert!(ev.is_finite() && ev.abs() < 3.5);
    }
}

#[test]
fn a_thumbnail_that_fails_skips_that_photo_only() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    write_plain(dir.path(), "b.ARW", 0.25);
    let mut recorder = Recorder::new(Duration::ZERO);
    Arc::get_mut(&mut recorder).unwrap().fail_for = Some(PhotoId(1));
    let session = Session::new(base_env().with_thumbnails(recorder));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    assert_eq!(
        session.status(PhotoId(1)),
        Some(PhotoStatus::Skipped {
            reason: "The thumbnail could not be drawn.".into()
        })
    );
    assert_eq!(session.status(PhotoId(2)), Some(PhotoStatus::Ready));
}

#[test]
fn only_the_current_photo_and_its_neighbours_keep_their_mosaics() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..8 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.25 + 0.01 * f64::from(i));
    }
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    // The first photo is current: it and its next neighbour are cached, nothing else.
    assert_eq!(session.current(), Some(ids[0]));
    let mut cached = session.cached_ids();
    cached.sort();
    assert_eq!(cached, vec![ids[0], ids[1]]);
    assert!(session.max_cached() <= 3);

    // Moving the current photo queues the new window for decoding.
    session.set_current(ids[4]).unwrap();
    settle(&session);
    let mut cached = session.cached_ids();
    cached.sort();
    assert_eq!(cached, vec![ids[3], ids[4], ids[5]]);
    session.set_current(ids[7]).unwrap();
    settle(&session);
    let mut cached = session.cached_ids();
    cached.sort();
    assert_eq!(cached, vec![ids[6], ids[7]]);
    assert!(session.max_cached() <= 3, "never more than three at once");

    // A mosaic outside the window is decoded on demand and not kept.
    let raw = session.raw_image(ids[0]).unwrap();
    assert_eq!(raw.meta.model, MODEL);
    assert!(!session.cached_ids().contains(&ids[0]));
    // One inside the window is served from the cache (the same allocation).
    let a = session.raw_image(ids[7]).unwrap();
    let b = session.raw_image(ids[7]).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    assert!(session.set_current(PhotoId(9999)).is_err());
}

#[test]
fn callers_that_ask_for_one_uncached_mosaic_at_once_share_one_decode() {
    // The app asks for a photo's pixels from many commands at once (the viewport frame, the
    // histogram, a tile for each look of the Look tab, the prefetch); each used to decode the file
    // for itself, a transient mosaic apiece (122 MB at 61 MP), against the budget of three.
    let dir = tempfile::tempdir().unwrap();
    for i in 0..3 {
        Scene::new(
            4000,
            3000,
            light(PROFILE, 5200.0),
            Layout::Patchwork,
            0.25 + 0.01 * f64::from(i),
        )
        .noise(2e-4, 1e-6)
        .write(dir.path(), &format!("p{i}.ARW"), MODEL);
    }
    let (session, _rx) = session_with_events();
    let session = Arc::new(session);
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    assert!(!session.cached_ids().contains(&ids[2]), "not decoded yet");

    // Moving to the third photo is the jump: its window is not cached, and every command asks.
    let third = ids[2];
    session.set_current(third).unwrap();
    const CALLERS: usize = 12;
    let barrier = Arc::new(std::sync::Barrier::new(CALLERS));
    let threads: Vec<_> = (0..CALLERS)
        .map(|_| {
            let (session, barrier) = (Arc::clone(&session), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                session.raw_image(third).unwrap()
            })
        })
        .collect();
    let images: Vec<Arc<ym_raw::decode::RawImage>> =
        threads.into_iter().map(|t| t.join().unwrap()).collect();
    settle(&session);

    let cached = session.raw_image(third).unwrap();
    let distinct = images.iter().filter(|i| !Arc::ptr_eq(i, &cached)).count();
    assert_eq!(
        distinct, 0,
        "{CALLERS} callers, {distinct} got an image of their own"
    );
    assert_eq!(
        session.raw_decodes(),
        1,
        "the file was read once (the first photo's window was cached by its development)"
    );
    assert!(session.max_cached() <= 3);
}

#[test]
fn a_prefetch_queued_for_a_photo_the_person_has_left_decodes_nothing() {
    // Stepping through photos faster than they decode used to decode every photo passed, once
    // each, to keep two or three mosaics at the end. One decode thread, parked on the development
    // of a photo opened last, so that the steps are taken while the prefetches wait in the queue.
    let dir = tempfile::tempdir().unwrap();
    for i in 0..12 {
        write_plain(
            dir.path(),
            &format!("p{i:02}.ARW"),
            0.25 + 0.01 * f64::from(i),
        );
    }
    let later = tempfile::tempdir().unwrap();
    write_plain(later.path(), "z.ARW", 0.3);
    let gate = common::GateThumbnails::new(13);
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(gate.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    session.open(&[later.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    assert_eq!(ids.len(), 13);

    let before = session.raw_decodes();
    for id in &ids[..12] {
        session.set_current(*id).unwrap();
    }
    gate.release();
    settle(&session);

    let decoded = session.raw_decodes() - before;
    assert!(
        decoded <= 3,
        "12 steps decoded {decoded} photos; only the last window is worth decoding"
    );
    let mut cached = session.cached_ids();
    cached.sort();
    assert_eq!(cached, vec![ids[10], ids[11], ids[12]], "the last window");
}

#[test]
fn closing_cancels_the_jobs_and_nothing_is_announced_afterwards() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..12 {
        write_plain(dir.path(), &format!("p{i:02}.ARW"), 0.25);
    }
    // One decode thread and a slow thumbnail: the queue cannot drain before the close.
    let renderer = Recorder::new(Duration::from_millis(150));
    let (sink, rx) = EventSink::channel();
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_events(sink)
            .with_thumbnails(renderer.clone()),
    );
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 12);
    let deadline = Instant::now() + Duration::from_secs(30);
    while !renderer.started.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "no job ever started");
        std::thread::sleep(Duration::from_millis(5));
    }
    let closing = Instant::now();
    session.close();
    assert!(
        closing.elapsed() < Duration::from_millis(100),
        "close does not wait for the queue"
    );
    let before: Vec<SessionEvent> = rx.try_iter().collect();
    assert!(!before.is_empty());
    // The session is empty at once, and nothing queued runs afterwards.
    assert!(session.state().photos.is_empty());
    assert_eq!(session.state().current_index, None);
    assert!(session.cached_ids().is_empty());
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(rx.try_iter().count(), 0, "no event after close");
    assert!(
        renderer.calls.load(Ordering::SeqCst) <= 2,
        "{} thumbnails",
        renderer.calls.load(Ordering::SeqCst)
    );
    assert!(session.jobs_started() < 12);
    assert_eq!(session.pending_jobs(), 0);

    // The session can be used again; the old photos' ids do not leak into it.
    let again = session.open(&[dir.path().join("p03.ARW")], OpenOptions::default());
    assert_eq!(again.added.len(), 1);
    settle(&session);
    assert_eq!(session.status(again.added[0]), Some(PhotoStatus::Ready));
    assert_eq!(session.state().photos.len(), 1);
}

#[test]
fn dropping_a_busy_session_stops_its_threads() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..10 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.25);
    }
    let renderer = Recorder::new(Duration::from_millis(50));
    let session = Session::new(
        base_env()
            .with_decode_threads(2)
            .with_thumbnails(renderer.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    let t = Instant::now();
    drop(session);
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "drop took {:?}",
        t.elapsed()
    );
}

#[test]
fn the_decode_pool_has_min_of_cores_minus_two_and_six_threads() {
    let cpus = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    let want = cpus.saturating_sub(2).clamp(1, 6);
    assert_eq!(default_decode_threads(), want);
    assert!((1..=6).contains(&default_decode_threads()));
}

#[test]
fn a_burst_shares_one_exposure_and_a_lone_frame_keeps_its_own() {
    let dir = tempfile::tempdir().unwrap();
    // Three frames half a second apart at identical settings, but of different brightness (a
    // subject moving through the frame), and one frame ten seconds later at the same brightness
    // as the darkest.
    let frame = |name: &str, sec: &str, level: f64| {
        let s = scene(level);
        s.builder(MODEL)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(&dir.path().join(name), &s.mosaic())
            .unwrap();
    };
    frame("a.ARW", "10", 0.10);
    frame("b.ARW", "10", 0.20);
    frame("c.ARW", "11", 0.40);
    frame("d.ARW", "40", 0.10);
    let (session, rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    let own = |i: usize| session.estimates(ids[i]).unwrap().exposure.own_ev;
    let shared = |i: usize| session.estimates(ids[i]).unwrap().exposure.ev;
    // The exposure is the camera's gain whatever the brightness (docs/02, interim rule of 2026-10-14), so the own
    // exposures of the four frames are one constant. The key statistic that is still reported follows the brightness
    // (one stop per doubling), which shows that the frames are not alike. (Own exposures that differ, and the median
    // of them, are tested on the pure functions in `ym-color`: on a Sony frame only the highlight guard can lower the
    // constant and it does not fire on a synthetic scene that stays within the raw range.)
    for i in 0..4 {
        assert_eq!(own(i), CAMERA_GAIN_EV, "photo {i}");
    }
    let key = |i: usize| {
        session
            .estimates(ids[i])
            .unwrap()
            .exposure
            .meter
            .mean_luminance
    };
    assert!((key(1) / key(0) - 2.0).abs() < 0.1 && (key(2) / key(1) - 2.0).abs() < 0.1);
    assert!((key(3) / key(0) - 1.0).abs() < 0.05);
    // The burst a, b, c (b to c is 1 s, a to b 0 s) shares the median, which is b's.
    for i in 0..3 {
        assert!((shared(i) - own(1)).abs() < 1e-9, "photo {i}");
    }
    // The photo ten seconds later is alone.
    assert!((shared(3) - own(3)).abs() < 1e-9);
    // The colour estimates the render uses carry the shared value.
    let est = session.estimates(ids[0]).unwrap();
    assert!((est.color.ev_auto - own(1)).abs() < 1e-9);
    // Whoever finished first was told when the median moved.
    let changed: Vec<PhotoId> = rx
        .try_iter()
        .filter_map(|e| match e {
            SessionEvent::EstimatesChanged { id } => Some(id),
            _ => None,
        })
        .collect();
    assert!(
        changed.is_empty() || changed.iter().all(|id| ids[..3].contains(id)),
        "only burst members can change: {changed:?}"
    );
    // The value does not depend on the order of completion: a second session with one thread
    // (strict order) gives the same numbers.
    let (single, _rx) = {
        let (sink, rx) = EventSink::channel();
        (
            Session::new(base_env().with_decode_threads(1).with_events(sink)),
            rx,
        )
    };
    single.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&single);
    for (i, id) in single.photo_ids().into_iter().enumerate() {
        let e = single.estimates(id).unwrap();
        assert!((e.exposure.ev - shared(i)).abs() < 1e-9, "photo {i}");
    }
}

/// Writes `layouts` as ISO 800 files of the same camera with true noise `(a, b)`.
fn write_noise_group(dir: &Path, layouts: &[(Layout, f64)], a: f64, b: f64) {
    for (i, (layout, level)) in layouts.iter().enumerate() {
        let s = Scene::new(768, 512, light(PROFILE, 5200.0), *layout, *level)
            .noise(a, b)
            .seed(i as u64 + 1);
        s.builder(MODEL)
            .iso(800)
            .write_to(&dir.join(format!("n{i}.ARW")), &s.mosaic())
            .unwrap();
    }
}

#[test]
fn photos_of_one_body_and_iso_pool_their_noise_estimate() {
    let dir = tempfile::tempdir().unwrap();
    // Three frames that cover different parts of the intensity range, as a shoot does, and a
    // fourth at another ISO that does not join the group.
    write_noise_group(
        dir.path(),
        &[
            (Layout::Ramp, 0.25),
            (Layout::Ramp, 0.9),
            (Layout::Patchwork, 0.6),
        ],
        5e-4,
        2e-6,
    );
    let other = Scene::new(768, 512, light(PROFILE, 5200.0), Layout::Ramp, 0.8)
        .noise(1e-4, 2e-6)
        .seed(9);
    other
        .builder(MODEL)
        .iso(200)
        .write_to(&dir.path().join("z.ARW"), &other.mosaic())
        .unwrap();
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    let first = session.estimates(ids[0]).unwrap().noise.model;
    for id in &ids[..3] {
        let e = session.estimates(*id).unwrap();
        assert_eq!(
            e.noise.source,
            NoiseSource::Pooled { files: 3 },
            "photo {id}"
        );
        assert!(
            (e.noise.model.a / 5e-4 - 1.0).abs() < 0.06,
            "a {}",
            e.noise.model.a
        );
        // The members share one model (none of them is filtered).
        assert_eq!(e.noise.model, first);
        // Pooling brought the read noise within reach: three bright-only or dark-only frames
        // cannot see it alone.
        assert!(
            (e.noise.model.b / 2e-6 - 1.0).abs() < 0.5,
            "b {}",
            e.noise.model.b
        );
        // Each file keeps its own estimate next to the pooled one.
        assert!(e.noise.estimate.is_some());
    }
    let lone = session.estimates(ids[3]).unwrap();
    assert_eq!(lone.noise.source, NoiseSource::Image);
    assert!(
        (lone.noise.model.a / 1e-4 - 1.0).abs() < 0.2,
        "a {}",
        lone.noise.model.a
    );

    // The pooled result does not depend on which photo finished first.
    let (sink, _r2) = EventSink::channel();
    let single = Session::new(base_env().with_decode_threads(1).with_events(sink));
    single.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&single);
    for (i, id) in single.photo_ids().into_iter().enumerate() {
        let want = session.estimates(ids[i]).unwrap().noise.model;
        let got = single.estimates(id).unwrap().noise.model;
        assert_eq!(got, want, "photo {i}");
    }
}

#[test]
fn three_frames_of_one_scene_pool_to_their_own_fit() {
    // Three frames of the same scene have the same bins up to independent noise. The first
    // version of `ym_noise::BatchPool` read that as overwhelming evidence and slid to 1.77e-4 for
    // a true 5e-4 (each file's own fit: 5.0 to 5.1e-4), and the agreement guard of the session
    // (`POOL_AGREEMENT_FACTOR`) was what kept the files' own estimates. The pooling maths is fixed:
    // the pooled fit is accepted by the guard and is the files' own to a few percent.
    let dir = tempfile::tempdir().unwrap();
    write_noise_group(
        dir.path(),
        &[
            (Layout::Ramp, 0.8),
            (Layout::Ramp, 0.8),
            (Layout::Ramp, 0.8),
        ],
        5e-4,
        2e-6,
    );
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    for id in session.photo_ids() {
        let e = session.estimates(id).unwrap();
        assert!(
            (e.noise.model.a / 5e-4 - 1.0).abs() < 0.06,
            "photo {id}: a {}",
            e.noise.model.a
        );
        assert_eq!(
            e.noise.source,
            NoiseSource::Pooled { files: 3 },
            "photo {id}"
        );
    }
}

/// Four frames: a, b, c a burst (the same second, identical settings), d ten seconds later.
/// `d_iso` is the ISO of d: the same as the others puts d in their noise pool, another does not.
fn write_burst_and_a_lone_frame(dir: &Path, d_iso: u32) {
    let frame = |name: &str, sec: &str, level: f64, iso: u32| {
        let s = scene(level);
        s.builder(MODEL)
            .iso(iso)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(&dir.join(name), &s.mosaic())
            .unwrap();
    };
    frame("a.ARW", "10", 0.10, 400);
    frame("b.ARW", "10", 0.20, 400);
    frame("c.ARW", "11", 0.40, 400);
    frame("d.ARW", "40", 0.10, d_iso);
}

#[test]
fn estimates_are_final_when_the_photos_they_depend_on_are_estimated() {
    // One decode thread and a thumbnail hook that parks the third thumbnail: a, b and c are
    // estimated, d is still waiting. a's burst is {a, b, c}; whether d is in its noise pool
    // depends on d's ISO.
    for (d_iso, d_in_pool) in [(800, false), (400, true)] {
        let dir = tempfile::tempdir().unwrap();
        write_burst_and_a_lone_frame(dir.path(), d_iso);
        let gate = common::GateThumbnails::new(3);
        let session = Session::new(
            base_env()
                .with_decode_threads(1)
                .with_thumbnails(gate.clone()),
        );
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        gate.wait_parked();
        let ids = session.photo_ids();
        let est = |i: usize| session.estimates(ids[i]).is_some();
        assert!(est(0) && est(1) && est(2) && !est(3), "a, b, c done; d not");
        // A photo that has no estimates of its own yet is not final.
        assert!(!session.estimates_final(ids[3]).unwrap());
        // The burst is complete; the pool is complete only without d.
        for i in 0..3 {
            assert_eq!(
                session.estimates_final(ids[i]).unwrap(),
                !d_in_pool,
                "photo {i}, d at ISO {d_iso}"
            );
        }
        // Unknown photos are an error, not "not final".
        assert!(matches!(
            session.estimates_final(PhotoId(999)),
            Err(SessionError::UnknownPhoto(_))
        ));
        gate.release();
        settle(&session);
        for id in &ids {
            assert!(
                session.estimates_final(*id).unwrap(),
                "after everything is developed"
            );
        }
    }
}

#[test]
fn a_burst_member_that_is_still_waiting_keeps_the_whole_burst_open() {
    // Same setup, but the parked photo is the first one (a): b and c have not been estimated. The
    // burst {a, b, c} is incomplete although a has its estimates, and d (alone, another ISO) is
    // not final either since it has none.
    let dir = tempfile::tempdir().unwrap();
    write_burst_and_a_lone_frame(dir.path(), 800);
    let gate = common::GateThumbnails::new(1);
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(gate.clone()),
    );
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    assert!(session.estimates(ids[0]).is_some() && session.estimates(ids[1]).is_none());
    assert!(!session.estimates_final(ids[0]).unwrap());
    gate.release();
    settle(&session);
    assert!(session.estimates_final(ids[0]).unwrap());
}

#[test]
fn settings_commands_resolve_clamp_and_stay_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    write_plain(dir.path(), "b.ARW", 0.25);
    write_plain(dir.path(), "c.ARW", 0.25);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    // Not ready photos refuse settings commands in plain words (or are already done).
    settle(&session);
    let ids = session.photo_ids();

    let r = session.settings(ids[0]).unwrap();
    assert!(r.auto.exposure && r.auto.white_balance && r.auto.noise_reduction);
    assert_eq!(r.photo_id, ids[0]);
    assert_eq!(r.settings.look.as_str(), "builtin:neutral");

    // Moving the exposure slider makes it Manual and clamps.
    let r = session
        .update_settings(ids[0], &SettingsPatch::exposure_ev(0.4))
        .unwrap();
    assert!(!r.auto.exposure && r.values.exposure_ev == 0.4);
    let r = session
        .update_settings(ids[0], &SettingsPatch::exposure_ev(9.0))
        .unwrap();
    assert_eq!(r.values.exposure_ev, 4.0);
    // A rejected patch leaves the settings as they were.
    let bad = SettingsPatch {
        contrast: Some(ym_core::settings::PercentSetting::manual_value(f64::NAN)),
        ..Default::default()
    };
    assert!(matches!(
        session.update_settings(ids[0], &bad),
        Err(SessionError::Settings(_))
    ));
    assert_eq!(session.settings(ids[0]).unwrap().values.exposure_ev, 4.0);
    // Only Kelvin moved: manual at that Kelvin and the automatic tint.
    let auto_tint = session.settings(ids[1]).unwrap().values.tint;
    let r = session
        .update_settings(ids[1], &SettingsPatch::kelvin(4100.0))
        .unwrap();
    assert_eq!((r.values.kelvin, r.values.tint), (4100.0, auto_tint));
    assert!(!r.auto.white_balance);
    // Other photos are untouched.
    assert!(session.settings(ids[2]).unwrap().auto.exposure);

    // Apply to: this photo (the current one), picked, all.
    let look = LookId::builtin("signature");
    assert_eq!(
        session.apply_look(ApplyScope::This, &look, 70.0).unwrap(),
        vec![ids[0]]
    );
    session.set_picked(ids[2], true).unwrap();
    assert_eq!(
        session.apply_look(ApplyScope::Picked, &look, 70.0).unwrap(),
        vec![ids[2]]
    );
    assert_eq!(
        session
            .apply_look(ApplyScope::All, &look, 40.0)
            .unwrap()
            .len(),
        3
    );
    for id in &ids {
        let r = session.settings(*id).unwrap();
        assert_eq!(
            (r.values.look.as_str(), r.values.look_intensity),
            ("builtin:signature", 40.0)
        );
    }
    // Look intensity is clamped; the other settings were not touched by Apply to.
    session.apply_look(ApplyScope::All, &look, 400.0).unwrap();
    assert_eq!(
        session.settings(ids[1]).unwrap().values.look_intensity,
        100.0
    );
    assert_eq!(session.settings(ids[0]).unwrap().values.exposure_ev, 4.0);
    assert!(
        session
            .apply_look(ApplyScope::All, &look, f64::NAN)
            .is_err()
    );

    // Picking.
    assert_eq!(session.picked(), vec![ids[2]]);
    session.pick_all();
    assert_eq!(session.picked(), ids);
    assert!(session.state().photos.iter().all(|p| p.picked));
    session.set_picked(ids[1], false).unwrap();
    assert_eq!(session.picked(), vec![ids[0], ids[2]]);
    assert_eq!(session.scope_ids(ApplyScope::Picked), vec![ids[0], ids[2]]);
    session.clear_picks();
    assert!(session.picked().is_empty());
    assert!(session.set_picked(PhotoId(77), true).is_err());

    // Unknown photos.
    assert!(matches!(
        session.settings(PhotoId(77)),
        Err(SessionError::UnknownPhoto(_))
    ));

    // New photos take the default look.
    session.set_default_look(LookId::builtin("signature"));
    write_plain(dir.path(), "d.ARW", 0.25);
    let added = session.open(&[dir.path().join("d.ARW")], OpenOptions::default());
    settle(&session);
    assert_eq!(
        session
            .settings(added.added[0])
            .unwrap()
            .values
            .look
            .as_str(),
        "builtin:signature"
    );
}

#[test]
fn replacing_the_settings_clamps_them_like_a_patch() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let id = session.photo_ids()[0];
    let mut s = ym_core::settings::DevelopSettings::default();
    s.contrast = ym_core::settings::PercentSetting::manual_value(500.0);
    s.exposure = ym_core::settings::ExposureSetting::manual_ev(-9.0);
    let r = session.replace_settings(id, s).unwrap();
    assert_eq!(r.values.exposure_ev, -4.0);
    assert_eq!(
        r.settings.contrast,
        ym_core::settings::PercentSetting::manual_value(100.0)
    );
    let mut bad = ym_core::settings::DevelopSettings::default();
    bad.curve.luma = vec![[0.5, 0.5]];
    assert!(session.replace_settings(id, bad).is_err());
}

#[test]
fn rerun_recomputes_the_estimates_and_resets_exposure_white_balance_and_noise() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let id = session.photo_ids()[0];
    let before = session.view(id).unwrap();
    let patch = SettingsPatch {
        exposure: Some(ym_core::settings::ExposureSetting::manual_ev(1.0)),
        white_balance: Some(AutoOr::Manual(ym_core::settings::WhiteBalanceManualPatch {
            kelvin: Some(3000.0),
            tint: Some(5.0),
        })),
        noise_reduction: Some(AutoOr::Manual(
            ym_core::settings::NoiseReductionManualPatch {
                luminance: Some(80.0),
                color: Some(80.0),
                detail: Some(80.0),
            },
        )),
        sharpen: Some(AutoOr::Manual(ym_core::settings::SharpenManualPatch {
            amount: Some(200.0),
            radius_px: Some(1.0),
        })),
        contrast: Some(ym_core::settings::PercentSetting::manual_value(20.0)),
        ..Default::default()
    };
    session.update_settings(id, &patch).unwrap();
    let r = session.rerun_auto(id).unwrap();
    assert!(
        r.auto.exposure && r.auto.white_balance && r.auto.noise_reduction && r.auto.light_source
    );
    assert!(r.settings.white_pick.is_none());
    // Sharpening and the other controls are not part of Re-run, nor are contrast and saturation
    // without Apple Auto (they are while Apple Auto acts on the photo, docs/19 section 5).
    assert!(!r.auto.sharpen);
    assert!(!r.auto.contrast);
    assert_eq!(r.values.contrast, 20.0);
    // The estimates were recomputed (same pixels, same answer) and the version moved on.
    let after = session.view(id).unwrap();
    assert!(after.estimates_version > before.estimates_version);
    assert_eq!(
        after.estimates.summary().exposure_ev,
        before.estimates.summary().exposure_ev
    );
    assert_eq!(
        after.estimates.color.w_scene_initial,
        before.estimates.color.w_scene_initial
    );
}

#[test]
fn a_white_pick_samples_the_photo_and_a_blown_out_spot_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    // Lit at 3000 K, but the camera's white balance says daylight.
    let truth = light(PROFILE, 3000.0);
    let s = Scene::new(512, 384, truth, Layout::Patchwork, 0.5).noise(1e-4, 1e-6);
    let mut mosaic = s.mosaic();
    // Blow out the neighbourhood of one point (patch k = 0, around x = 40, y = 40).
    for y in 30..52 {
        for x in 30..52 {
            mosaic[y * 512 + x] = 16300;
        }
    }
    s.builder(MODEL)
        .wb(Some(common::wb_levels(light(PROFILE, 6500.0))))
        .write_to(&dir.path().join("a.ARW"), &mosaic)
        .unwrap();
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let id = session.photo_ids()[0];
    let before = session.settings(id).unwrap();
    assert!((before.values.kelvin - 6500.0).abs() <= 50.0);
    // A neutral patch (k = 2) finds the true light.
    let patch = SettingsPatch {
        white_pick: Some(Some([0.4167, 0.125])),
        ..Default::default()
    };
    let r = session.update_settings(id, &patch).unwrap();
    assert!(
        r.auto.white_balance,
        "a pick leaves the white balance on Auto"
    );
    assert!(
        (r.values.kelvin - 3000.0).abs() <= 100.0,
        "{}",
        r.values.kelvin
    );
    assert_eq!(r.settings.white_pick, Some([0.4167, 0.125]));
    // The sample is remembered with the photo.
    let view = session.view(id).unwrap();
    assert!(view.picked_white().is_some());
    assert!((session.settings(id).unwrap().values.kelvin - r.values.kelvin).abs() < 1e-9);
    // Moving Kelvin switches to Manual and keeps the picked scene light for the sensor map.
    let r2 = session
        .update_settings(id, &SettingsPatch::kelvin(4000.0))
        .unwrap();
    assert_eq!(r2.values.kelvin, 4000.0);
    assert!(r2.settings.white_pick.is_some());

    // The blown-out spot is refused with the spec's message and changes nothing.
    let blown = SettingsPatch {
        white_pick: Some(Some([41.0 / 512.0, 41.0 / 384.0])),
        ..Default::default()
    };
    let err = session.update_settings(id, &blown).unwrap_err();
    assert_eq!(
        err.to_string(),
        "That spot is too bright to use. Pick a grey or white area that isn't blown out."
    );
    assert_eq!(
        session.settings(id).unwrap().settings.white_pick,
        Some([0.4167, 0.125])
    );
    // Clearing the pick drops the sample.
    let cleared = SettingsPatch {
        white_pick: Some(None),
        ..Default::default()
    };
    session.update_settings(id, &cleared).unwrap();
    assert!(session.view(id).unwrap().pick.is_none());
    // Re-run also forgets a pick.
    session.update_settings(id, &patch).unwrap();
    session.rerun_auto(id).unwrap();
    assert!(session.settings(id).unwrap().settings.white_pick.is_none());
}

/// Three developed photos of one scene at three brightnesses.
fn three_photos() -> (tempfile::TempDir, Session, Vec<PhotoId>) {
    let dir = tempfile::tempdir().unwrap();
    for (name, level) in [("a.ARW", 0.3), ("b.ARW", 0.2), ("c.ARW", 0.4)] {
        write_plain(dir.path(), name, level);
    }
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    assert_eq!(ids.len(), 3);
    (dir, session, ids)
}

/// A patch that sets every kind of setting, the crop, the turn and the white pick included.
fn edit_everything() -> SettingsPatch {
    SettingsPatch {
        look: Some(LookId::builtin("signature")),
        look_intensity: Some(60.0),
        exposure: Some(ExposureSetting::manual_ev(0.7)),
        contrast: Some(ym_core::settings::PercentSetting::manual_value(12.0)),
        brightness: Some(-8.0),
        highlights: Some(ym_core::settings::PercentSetting::manual_value(-30.0)),
        shadows: Some(ym_core::settings::PercentSetting::manual_value(25.0)),
        white: Some(5.0),
        white_offset_ev: Some(0.5),
        black: Some(ym_core::settings::PercentSetting::manual_value(-4.0)),
        light_source: Some(LightSource::Day),
        white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
            kelvin: Some(4300.0),
            tint: Some(7.0),
        })),
        color_balance: Some(ColorBalancePatch {
            shadows: Some(ym_core::settings::BalanceSetting::manual_point([0.2, -0.1])),
            midtones: None,
            highlights: Some(ym_core::settings::BalanceSetting::manual_point([-0.1, 0.1])),
        }),
        saturation: Some(ym_core::settings::PercentSetting::manual_value(15.0)),
        color_warp: Some(ym_core::settings::ColorWarp {
            pins: vec![
                ym_color::warp::WarpPin::moving([0.12, 0.03], [0.08, 0.1]).into(),
                ym_color::warp::WarpPin::locked_at([-0.05, -0.1]).into(),
            ],
        }),
        sharpen: Some(AutoOr::Manual(SharpenManualPatch {
            amount: Some(80.0),
            radius_px: Some(1.2),
        })),
        output_sharpening: Some(60.0),
        noise_reduction: Some(AutoOr::Manual(NoiseReductionManualPatch {
            luminance: Some(30.0),
            color: Some(40.0),
            detail: Some(50.0),
        })),
        curve: Some(CurvesPatch {
            luma: Some(vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]),
            ..Default::default()
        }),
        decode: Some(DecodePatch {
            neural_demosaic: Some(false),
            lens_corrections: Some(false),
            ..Default::default()
        }),
        rolloff: Some(RolloffPatch {
            highlights: Some(0.7),
            low_light: Some(0.3),
        }),
        crop: Some(Some(Crop {
            rect: NormRect {
                x: 0.1,
                y: 0.1,
                width: 0.5,
                height: 0.5,
            },
            angle_deg: 3.0,
            aspect: CropAspect::Free,
        })),
        orientation: Some(Orientation::Deg90),
        white_pick: Some(Some([0.4167, 0.125])),
    }
}

#[test]
fn pasting_takes_every_setting_but_the_crop_the_turn_and_the_white_pick() {
    let (_dir, session, ids) = three_photos();
    let (a, b, c) = (ids[0], ids[1], ids[2]);
    // `white_pick` last in the patch: a pick clears the light preset and the white balance unless
    // the same patch sets them, and this one does.
    session.update_settings(a, &edit_everything()).unwrap();
    // The targets have a crop and a turn of their own; c has only a turn.
    let own_crop = SettingsPatch {
        crop: Some(Some(Crop {
            rect: NormRect {
                x: 0.0,
                y: 0.2,
                width: 0.9,
                height: 0.6,
            },
            angle_deg: -2.0,
            aspect: CropAspect::Original,
        })),
        orientation: Some(Orientation::Deg180),
        ..Default::default()
    };
    session.update_settings(b, &own_crop).unwrap();
    session
        .update_settings(
            c,
            &SettingsPatch {
                orientation: Some(Orientation::Deg270),
                ..Default::default()
            },
        )
        .unwrap();

    let source = session.settings(a).unwrap().settings;
    assert!(source.crop.is_some() && source.white_pick.is_some());
    assert_eq!(source.orientation, Orientation::Deg90);

    let patch = session.copy_settings(a).unwrap();
    assert!(
        patch.crop.is_none() && patch.orientation.is_none() && patch.white_pick.is_none(),
        "the copy leaves out the crop, the turn and the white pick"
    );
    assert_eq!(
        patch.color_warp.as_ref().map(|w| w.pins.len()),
        Some(2),
        "the copy takes the colour warp's pins"
    );
    let own: Vec<_> = [b, c]
        .iter()
        .map(|id| session.settings(*id).unwrap().settings)
        .collect();
    assert_ne!(own[0], source);
    let report = session.paste_settings(&[b, c], &patch).unwrap();
    assert_eq!(
        report.pasted.iter().map(|p| p.id).collect::<Vec<_>>(),
        [b, c]
    );
    assert!(report.skipped.is_empty());
    for (id, before) in [b, c].into_iter().zip(own) {
        let mut expected = source.clone();
        expected.crop = before.crop;
        expected.orientation = before.orientation;
        expected.white_pick = before.white_pick;
        assert_eq!(session.settings(id).unwrap().settings, expected);
    }
    // The source itself was not touched.
    assert_eq!(session.settings(a).unwrap().settings, source);
}

#[test]
fn auto_stays_auto_and_each_photo_keeps_its_own_estimates() {
    let (_dir, session, ids) = three_photos();
    let (a, b) = (ids[0], ids[1]);
    session
        .update_settings(
            a,
            &SettingsPatch {
                contrast: Some(ym_core::settings::PercentSetting::manual_value(20.0)),
                ..Default::default()
            },
        )
        .unwrap();
    // b is manual wherever a is automatic.
    session
        .update_settings(
            b,
            &SettingsPatch {
                exposure: Some(ExposureSetting::manual_ev(-1.5)),
                light_source: Some(LightSource::Tungsten),
                white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                    kelvin: Some(3300.0),
                    tint: Some(-4.0),
                })),
                sharpen: Some(AutoOr::Manual(SharpenManualPatch {
                    amount: Some(120.0),
                    radius_px: Some(1.5),
                })),
                noise_reduction: Some(AutoOr::Manual(NoiseReductionManualPatch {
                    luminance: Some(70.0),
                    color: Some(70.0),
                    detail: Some(70.0),
                })),
                ..Default::default()
            },
        )
        .unwrap();
    let estimates_before = session.settings(b).unwrap().estimates;
    assert!(!session.settings(b).unwrap().auto.exposure);

    let patch = session.copy_settings(a).unwrap();
    assert_eq!(patch.exposure, Some(AutoOr::Auto));
    assert_eq!(patch.white_balance, Some(AutoOr::Auto));
    assert_eq!(patch.light_source, Some(LightSource::Auto));
    assert_eq!(patch.sharpen, Some(AutoOr::Auto));
    assert_eq!(patch.noise_reduction, Some(AutoOr::Auto));
    session.paste_settings(&[b], &patch).unwrap();

    let r = session.settings(b).unwrap();
    assert!(r.auto.exposure && r.auto.light_source && r.auto.white_balance);
    assert!(r.auto.sharpen && r.auto.noise_reduction);
    assert_eq!(
        r.settings.contrast,
        ym_core::settings::PercentSetting::manual_value(20.0)
    );
    // The automatic values are b's own, from b's pixels, and nothing changed them.
    assert_eq!(r.estimates, estimates_before);
    assert_eq!(r.values.exposure_ev, r.estimates.exposure_ev);
    assert_eq!(r.values.sharpen, r.estimates.sharpen);
    assert_eq!(r.values.noise_reduction, r.estimates.noise_reduction);

    // Manual values are pasted as they are, onto a photo that is automatic.
    session.update_settings(a, &edit_everything()).unwrap();
    let patch = session.copy_settings(a).unwrap();
    session.paste_settings(&[b], &patch).unwrap();
    let r = session.settings(b).unwrap();
    assert!(
        !r.auto.exposure && !r.auto.white_balance && !r.auto.sharpen && !r.auto.noise_reduction
    );
    assert_eq!(r.values.exposure_ev, 0.7);
    assert_eq!((r.values.kelvin, r.values.tint), (4300.0, 7.0));
    assert_eq!(r.settings.white_offset_ev, 0.5);
    assert_eq!(
        (r.settings.rolloff.highlights, r.settings.rolloff.low_light),
        (0.7, 0.3),
        "the highlight range is pasted as it is"
    );
}

#[test]
fn a_white_balance_from_a_picked_white_is_pasted_as_manual_kelvin_and_tint() {
    let dir = tempfile::tempdir().unwrap();
    // Lit at 3000 K, but the camera's white balance says daylight, as in the Pick white test.
    let s = Scene::new(512, 384, light(PROFILE, 3000.0), Layout::Patchwork, 0.5).noise(1e-4, 1e-6);
    s.builder(MODEL)
        .wb(Some(common::wb_levels(light(PROFILE, 6500.0))))
        .write_to(&dir.path().join("a.ARW"), &s.mosaic())
        .unwrap();
    write_plain(dir.path(), "b.ARW", 0.3);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let (a, b) = (session.photo_ids()[0], session.photo_ids()[1]);

    let pick = SettingsPatch {
        white_pick: Some(Some([0.4167, 0.125])),
        ..Default::default()
    };
    let picked = session.update_settings(a, &pick).unwrap();
    assert!((picked.values.kelvin - 3000.0).abs() <= 100.0);
    assert!(picked.auto.white_balance && picked.auto.light_source);

    let patch = session.copy_settings(a).unwrap();
    assert!(patch.white_pick.is_none(), "the point stays with its photo");
    assert_eq!(patch.light_source, Some(LightSource::Auto));
    assert_eq!(
        patch.white_balance,
        Some(AutoOr::Manual(WhiteBalanceManualPatch {
            kelvin: Some(picked.values.kelvin),
            tint: Some(picked.values.tint),
        }))
    );
    session.paste_settings(&[b], &patch).unwrap();
    let r = session.settings(b).unwrap();
    assert_eq!(
        r.settings.white_balance,
        AutoOr::Manual(WhiteBalanceValue {
            kelvin: picked.values.kelvin,
            tint: picked.values.tint
        })
    );
    assert!(r.settings.white_pick.is_none() && !r.auto.white_balance);

    // A light preset chosen after the pick wins over it: the preset is copied, the white balance
    // stays Auto.
    session
        .update_settings(
            a,
            &SettingsPatch {
                light_source: Some(LightSource::Day),
                ..Default::default()
            },
        )
        .unwrap();
    let patch = session.copy_settings(a).unwrap();
    assert_eq!(patch.light_source, Some(LightSource::Day));
    assert_eq!(patch.white_balance, Some(AutoOr::Auto));

    // A manual white balance next to a pick is copied as it is.
    session
        .update_settings(
            a,
            &SettingsPatch {
                light_source: Some(LightSource::Auto),
                white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                    kelvin: Some(4000.0),
                    tint: Some(3.0),
                })),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(session.settings(a).unwrap().settings.white_pick.is_some());
    let patch = session.copy_settings(a).unwrap();
    assert_eq!(
        patch.white_balance,
        Some(AutoOr::Manual(WhiteBalanceManualPatch {
            kelvin: Some(4000.0),
            tint: Some(3.0)
        }))
    );
}

#[test]
fn a_photo_that_cannot_be_adjusted_is_skipped_and_a_bad_patch_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.3);
    std::fs::write(dir.path().join("b.ARW"), b"this is not a raw file at all").unwrap();
    write_plain(dir.path(), "c.ARW", 0.4);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let ids = session.photo_ids();
    let (a, failed, c) = (ids[0], ids[1], ids[2]);
    assert!(matches!(
        session.status(failed),
        Some(PhotoStatus::Skipped { .. })
    ));
    assert!(session.copy_settings(failed).is_err());
    assert!(session.copy_settings(PhotoId(99)).is_err());

    session.update_settings(a, &edit_everything()).unwrap();
    let patch = session.copy_settings(a).unwrap();
    // A photo that failed or is not there is skipped; the others are still done, each once.
    let report = session
        .paste_settings(&[failed, PhotoId(99), c, c], &patch)
        .unwrap();
    assert_eq!(report.skipped, [failed, PhotoId(99)]);
    assert_eq!(report.pasted.len(), 1);
    assert_eq!(report.pasted[0].id, c);
    assert_eq!(session.settings(c).unwrap().values.exposure_ev, 0.7);

    // A patch that names the crop, the turn or the white pick does not touch them.
    let (_dir2, session2, ids2) = three_photos();
    session2
        .paste_settings(&[ids2[1]], &edit_everything())
        .unwrap();
    let r = session2.settings(ids2[1]).unwrap().settings;
    assert!(r.crop.is_none() && r.white_pick.is_none());
    assert_eq!(r.orientation, Orientation::Deg0);
    assert_eq!(
        r.contrast,
        ym_core::settings::PercentSetting::manual_value(12.0)
    );

    // A patch that cannot be applied changes no photo, not even the ones before the bad value.
    let before = session.settings(a).unwrap().settings;
    let bad = SettingsPatch {
        contrast: Some(ym_core::settings::PercentSetting::manual_value(f64::NAN)),
        ..Default::default()
    };
    assert!(matches!(
        session.paste_settings(&[a, c], &bad),
        Err(SessionError::Settings(_))
    ));
    assert_eq!(session.settings(a).unwrap().settings, before);
    assert_eq!(
        session.settings(c).unwrap().settings.contrast,
        ym_core::settings::PercentSetting::manual_value(12.0)
    );
}

#[test]
fn the_patches_of_a_paste_undo_and_redo_it_and_a_second_paste_changes_nothing() {
    let (_dir, session, ids) = three_photos();
    let (a, b) = (ids[0], ids[1]);
    session.update_settings(a, &edit_everything()).unwrap();
    session
        .update_settings(
            b,
            &SettingsPatch {
                contrast: Some(ym_core::settings::PercentSetting::manual_value(-10.0)),
                ..Default::default()
            },
        )
        .unwrap();
    let patch = session.copy_settings(a).unwrap();
    let was = session.settings(b).unwrap().settings;
    let report = session.paste_settings(&[b], &patch).unwrap();
    let now = session.settings(b).unwrap().settings;
    assert_ne!(was, now);
    let step = &report.pasted[0];
    assert!(!step.before.is_empty() && !step.after.is_empty());
    // Only what changed is in the step: the crop and the turn never are.
    assert!(step.before.crop.is_none() && step.after.orientation.is_none());
    session.update_settings(b, &step.before).unwrap();
    assert_eq!(session.settings(b).unwrap().settings, was);
    session.update_settings(b, &step.after).unwrap();
    assert_eq!(session.settings(b).unwrap().settings, now);
    // The same paste again: the photo already has these settings.
    let again = session.paste_settings(&[b], &patch).unwrap();
    assert_eq!(again.pasted.len(), 1);
    assert!(again.pasted[0].before.is_empty() && again.pasted[0].after.is_empty());
}

#[test]
fn the_current_photo_is_developed_first() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..10 {
        write_plain(dir.path(), &format!("p{i}.ARW"), 0.25);
    }
    // One decode thread and a slow thumbnail make the order visible.
    let renderer = Recorder::new(Duration::from_millis(120));
    let session = Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(renderer.clone()),
    );
    let files: Vec<PathBuf> = (0..10)
        .map(|i| dir.path().join(format!("p{i}.ARW")))
        .collect();
    // Open the first one so the pool is busy, then the rest, then move to the last.
    session.open(&files[..1], OpenOptions::default());
    session.open(&files[1..], OpenOptions::default());
    let last = *session.photo_ids().last().unwrap();
    session.set_current(last).unwrap();
    settle(&session);
    let seen = renderer.seen.lock().unwrap();
    let position = seen.iter().position(|(id, _)| *id == last).unwrap();
    // Photo 0 was already running when the others arrived; the last one is next, not ninth.
    assert!(position <= 3, "the last photo was thumbnailed {position}th");
}

#[test]
fn a_forced_profile_is_a_stand_in_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    let env = base_env().with_forced_profile("sony-ilce-9").unwrap();
    let session = Session::new(env);
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let s = session.state().photos[0].clone();
    assert_eq!(s.camera.profile_id, "sony-ilce-9");
    assert!(matches!(
        s.camera.match_kind,
        ym_profiles::detect::MatchKind::Forced { .. }
    ));
    assert!(session.camera(s.id).unwrap().stand_in);
    // An unknown id is refused with the valid ones listed.
    let err = base_env().with_forced_profile("nope").err().unwrap();
    assert!(err.to_string().contains("sony-ilce-7rm4"), "{err}");
}

#[test]
fn files_added_later_join_the_session_and_keep_the_current_photo() {
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    write_plain(dir.path(), "b.ARW", 0.25);
    let other = tempfile::tempdir().unwrap();
    write_plain(other.path(), "c.ARW", 0.25);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().join("b.ARW")], OpenOptions::default());
    session.open(&[dir.path().join("a.ARW")], OpenOptions::default());
    settle(&session);
    let state = session.state();
    assert_eq!(state.photos.len(), 2);
    // Opened order, then sorted within each open: b then a.
    assert_eq!(state.photos[0].file_name, "b.ARW");
    assert_eq!(state.current_index, Some(0));
    assert_eq!(
        state.folder_name.as_deref(),
        dir.path().file_name().unwrap().to_str()
    );
    // Photos from another folder: the session has no single folder any more.
    session.open(&[other.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let state = session.state();
    assert_eq!(state.photos.len(), 3);
    assert_eq!(state.folder_name, None);
    assert_eq!(state.default_export_folder, None);
    assert_eq!(ready_ids(&session).len(), 3);
}

#[test]
fn a_few_hundred_photos_open_without_the_shared_estimates_getting_expensive() {
    // 240 small frames of one camera and ISO: one noise pool, bursts of ten, one decode pool.
    // The shared estimates are recomputed as photos finish, so this guards against a cost that
    // grows with the square of the session.
    let dir = tempfile::tempdir().unwrap();
    for i in 0..240u32 {
        let level = 0.15 + 0.02 * f64::from(i % 10);
        let s = Scene::new(256, 192, light(PROFILE, 5200.0), Layout::Patchwork, level)
            .noise(2e-4, 1e-6)
            .seed(u64::from(i) + 1);
        s.builder(MODEL)
            .capture_time(
                &format!("2026:10:04 12:{:02}:{:02}", i / 100, (i / 10) % 60),
                &format!("{:02}", (i % 10) * 10),
                None,
            )
            .write_to(&dir.path().join(format!("p{i:03}.ARW")), &s.mosaic())
            .unwrap();
    }
    let (session, rx) = session_with_events();
    let started = Instant::now();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    let opened = started.elapsed();
    settle(&session);
    let total = started.elapsed();
    eprintln!("240 photos: open returned after {opened:?}, all developed after {total:?}");
    assert_eq!(ready_ids(&session).len(), 240);
    assert!(session.max_cached() <= 3);
    assert!(opened < Duration::from_secs(10), "open took {opened:?}");
    assert!(
        total < Duration::from_secs(120),
        "development took {total:?}"
    );
    // Bursts of ten share one exposure; the pool of the group is shared.
    let ids = session.photo_ids();
    let ev = |i: usize| session.estimates(ids[i]).unwrap().exposure.ev;
    for burst in 0..24 {
        for k in 1..10 {
            assert!(
                (ev(burst * 10 + k) - ev(burst * 10)).abs() < 1e-9,
                "burst {burst}"
            );
        }
    }
    let first = session.estimates(ids[0]).unwrap().noise.source;
    assert!(
        matches!(first, NoiseSource::Pooled { .. } | NoiseSource::Image),
        "{first:?}"
    );
    // Every photo got exactly one Ready, and the event count is linear in the photos.
    let events = progress_by_photo(&rx);
    assert_eq!(events.len(), 240);
    assert!(events.values().all(|e| e.len() == 7));
}

#[test]
fn the_session_and_its_views_can_be_shared_between_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<Session>();
    shareable::<ym_core::session::PhotoView>();
    shareable::<ym_core::settings::ResolvedSettings>();
    shareable::<ym_core::auto::AutoEstimates>();
    shareable::<SessionEvent>();
}

#[test]
fn statuses_scopes_and_events_have_the_wire_shapes_of_the_typescript_types() {
    use serde_json::json;
    let to = |v: &dyn erased::Ser| v.value();
    mod erased {
        pub trait Ser {
            fn value(&self) -> serde_json::Value;
        }
        impl<T: serde::Serialize> Ser for T {
            fn value(&self) -> serde_json::Value {
                serde_json::to_value(self).unwrap()
            }
        }
    }
    assert_eq!(to(&PhotoStatus::Waiting), json!({ "kind": "waiting" }));
    assert_eq!(
        to(&PhotoStatus::Developing {
            stage: PhotoStage::Estimate
        }),
        json!({ "kind": "developing", "stage": "estimate" })
    );
    assert_eq!(to(&PhotoStatus::Ready), json!({ "kind": "ready" }));
    assert_eq!(
        to(&PhotoStatus::Skipped { reason: "x".into() }),
        json!({ "kind": "skipped", "reason": "x" })
    );
    assert_eq!(to(&ApplyScope::Picked), json!("picked"));
    assert_eq!(to(&PhotoId(12)), json!(12));
    let progress = PhotoProgress {
        id: PhotoId(3),
        stage: PhotoStage::Thumbnail,
        done: true,
        error: None,
    };
    assert_eq!(
        to(&progress),
        json!({ "id": 3, "stage": "thumbnail", "done": true, "error": null })
    );
    assert_eq!(
        SessionEvent::PhotoProgress(progress).name(),
        "photo_progress"
    );
    // The camera object: match kinds as in src/api/types.ts.
    let dir = tempfile::tempdir().unwrap();
    write_plain(dir.path(), "a.ARW", 0.25);
    let (session, _rx) = session_with_events();
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let camera = to(&session.state().photos[0].camera);
    assert_eq!(camera["match_kind"], json!({ "kind": "exact" }));
    assert_eq!(camera["body"], json!("Sony A7R IV"));
    assert_eq!(camera["model"], json!("ILCE-7RM4"));
    assert_eq!(camera["model_id"], json!(375));
    assert_eq!(camera["profile"], json!("A7R IV"));
    assert_eq!(camera["profile_id"], json!("sony-ilce-7rm4"));
}
