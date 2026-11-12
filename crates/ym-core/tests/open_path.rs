//! The open path under slow and hung media (`docs/26-hardening.md` P3-3 and P1-7, tasks H-10 and
//! H-15): paths are made canonical outside the session lock, progress is reported, an open can be
//! cancelled, and a file that does not answer is `Skipped` instead of hanging `open`.
//!
//! No slow drive is used: `Media` is the seam, and a call that "hangs" waits on a gate the test
//! opens at the end.
mod common;

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use common::*;
use ym_core::media::{MAX_HUNG_PER_FOLDER, Media, NO_ANSWER_MESSAGE};
use ym_core::session::{OPEN_PROGRESS_STEP, OpenOptions, OpenProgress, PhotoStatus, Session};

/// A gate that holds calls until the test opens it (a safety cap of 60 s, so a failed test does not
/// leave threads for ever), and counts the calls that reached it.
#[derive(Clone, Default)]
struct Gate {
    open: Arc<(Mutex<bool>, Condvar)>,
    entered: Arc<AtomicUsize>,
}

impl Gate {
    fn wait(&self) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let (flag, cv) = &*self.open;
        let mut open = flag.lock().unwrap_or_else(PoisonError::into_inner);
        let deadline = Instant::now() + Duration::from_secs(60);
        while !*open && Instant::now() < deadline {
            open = cv
                .wait_timeout(open, Duration::from_millis(50))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn release(&self) {
        let (flag, cv) = &*self.open;
        *flag.lock().unwrap_or_else(PoisonError::into_inner) = true;
        cv.notify_all();
    }

    fn entered(&self) -> usize {
        self.entered.load(Ordering::SeqCst)
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A small valid ARW.
fn write_tiny(dir: &Path, name: &str, k: u64) {
    Scene::new(
        96,
        64,
        light("sony-ilce-7rm4", 5000.0),
        Layout::Uniform,
        0.3,
    )
    .seed(k + 1)
    .write(dir, name, "ILCE-7RM4");
}

/// Files with no content: they are listed like any ARW, and a test that replaces the probe never
/// reads them.
fn write_empty(dir: &Path, count: usize, prefix: &str) {
    for k in 0..count {
        std::fs::write(dir.join(format!("{prefix}{k:03}.ARW")), b"").expect("write");
    }
}

fn session_with(media: Media) -> Session {
    Session::new(base_env().with_media(media))
}

fn status_of(session: &Session, name: &str) -> PhotoStatus {
    session
        .state()
        .photos
        .into_iter()
        .find(|p| p.file_name == name)
        .unwrap_or_else(|| panic!("{name} is not in the session"))
        .status
}

#[test]
fn paths_are_made_canonical_outside_the_session_lock() {
    let first = tempfile::tempdir().expect("dir");
    let second = tempfile::tempdir().expect("dir");
    for k in 0..2 {
        write_tiny(first.path(), &format!("A{k}.ARW"), k);
    }
    for k in 0..3 {
        write_tiny(second.path(), &format!("B{k}.ARW"), 10 + k);
    }
    let gate = Gate::default();
    let slow_root = second.path().to_path_buf();
    let slow = gate.clone();
    let mut media = Media::new();
    media.canonicalize = Arc::new(move |path| {
        // The slow volume: every path of the second folder takes as long as the test wants.
        if path.starts_with(&slow_root) {
            slow.wait();
        }
        std::fs::canonicalize(path)
    });
    let session = Arc::new(session_with(media));
    let opened = session.open(&[first.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 2);
    let id = opened.added[1];

    let opening = {
        let session = Arc::clone(&session);
        let folder = second.path().to_path_buf();
        std::thread::spawn(move || session.open(&[folder], OpenOptions::default()))
    };
    wait_for("a path to reach the slow volume", || gate.entered() > 0);

    // The slow call is running. Commands that need the session lock must not wait for it.
    let (sent, received) = std::sync::mpsc::channel();
    {
        let session = Arc::clone(&session);
        std::thread::spawn(move || {
            let current = session.set_current(id);
            let listed = session.state().photos.len();
            let _ = sent.send((current.is_ok(), listed));
        });
    }
    let answered = received.recv_timeout(Duration::from_secs(20));
    gate.release();
    let outcome = opening.join().expect("the open");
    let (set, listed) = answered.expect("set_current and state waited for the slow paths");
    assert!(set);
    assert_eq!(listed, 2, "the second folder was not registered yet");
    assert_eq!(outcome.added.len(), 3);
    assert_eq!(session.state().photos.len(), 5);
}

#[test]
fn a_probe_that_sleeps_past_the_limit_skips_the_photo_and_open_returns() {
    let dir = tempfile::tempdir().expect("dir");
    for k in 0..3 {
        write_tiny(dir.path(), &format!("GOOD{k}.ARW"), k);
    }
    write_tiny(dir.path(), "HANG.ARW", 9);
    let gate = Gate::default();
    let held = gate.clone();
    let mut media = Media::new();
    media.probe_limit = Duration::from_millis(150);
    media.probe = Arc::new(move |path| {
        if path.file_name().is_some_and(|n| n == "HANG.ARW") {
            held.wait();
        }
        ym_raw::metadata::probe(path)
    });
    let session = session_with(media);
    let started = Instant::now();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(30),
        "open waited for the hung probe: {took:?}"
    );
    assert_eq!(outcome.added.len(), 4);
    assert!(!outcome.cancelled);
    assert_eq!(
        status_of(&session, "HANG.ARW"),
        PhotoStatus::Skipped {
            reason: NO_ANSWER_MESSAGE.to_owned()
        }
    );
    assert_eq!(
        NO_ANSWER_MESSAGE,
        "This file didn't answer. Check the drive or the network connection, then open it again."
    );
    settle(&session);
    for k in 0..3 {
        assert_eq!(
            status_of(&session, &format!("GOOD{k}.ARW")),
            PhotoStatus::Ready
        );
    }
    gate.release();
}

#[test]
fn a_hundred_hung_files_start_a_bounded_number_of_threads() {
    let dir = tempfile::tempdir().expect("dir");
    write_empty(dir.path(), 100, "HUNG");
    let gate = Gate::default();
    let held = gate.clone();
    let mut media = Media::new();
    media.probe_limit = Duration::from_millis(50);
    media.probe = Arc::new(move |_| {
        held.wait();
        Err(ym_raw::error::RawError::NotArw)
    });
    let watch = media.clone();
    let session = session_with(media);
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 100);
    for id in &outcome.added {
        assert_eq!(
            session.status(*id),
            Some(PhotoStatus::Skipped {
                reason: NO_ANSWER_MESSAGE.to_owned()
            })
        );
    }
    // The bound: MAX_HUNG_PER_FOLDER hung threads, plus the callers that were waiting when the
    // limit was reached (one per rayon thread). The other files got the sentence without a thread.
    let bound = MAX_HUNG_PER_FOLDER + rayon::current_num_threads();
    let started = usize::try_from(watch.threads_started()).expect("count");
    assert!(
        started <= bound,
        "{started} threads were started for 100 hung files; the bound is {bound}"
    );
    assert!(started >= MAX_HUNG_PER_FOLDER, "{started}");
    eprintln!("100 hung files: {started} threads started (bound {bound}), none for the rest");
    assert!(watch.hung_threads() <= bound);
    // Hung threads that return are taken out of the count.
    gate.release();
    wait_for("the hung threads to return", || watch.hung_threads() == 0);
}

#[test]
fn a_folder_that_answers_is_not_refused_because_another_one_hung() {
    let dead = tempfile::tempdir().expect("dir");
    let live = tempfile::tempdir().expect("dir");
    write_empty(dead.path(), 20, "DEAD");
    for k in 0..3 {
        write_tiny(live.path(), &format!("LIVE{k}.ARW"), k);
    }
    let gate = Gate::default();
    let held = gate.clone();
    let dead_root = dead.path().to_path_buf();
    let mut media = Media::new();
    media.probe_limit = Duration::from_millis(50);
    media.probe = Arc::new(move |path| {
        if path.starts_with(&dead_root) {
            held.wait();
        }
        ym_raw::metadata::probe(path)
    });
    let session = session_with(media);
    session.open(&[dead.path().to_path_buf()], OpenOptions::default());
    let outcome = session.open(&[live.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 3);
    settle(&session);
    for id in outcome.added {
        assert_eq!(session.status(id), Some(PhotoStatus::Ready));
    }
    gate.release();
}

#[test]
fn a_develop_read_that_does_not_answer_fails_the_photo_and_frees_the_pool() {
    let dir = tempfile::tempdir().expect("dir");
    write_tiny(dir.path(), "SLOW.ARW", 1);
    write_tiny(dir.path(), "FAST.ARW", 2);
    let gate = Gate::default();
    let held = gate.clone();
    let mut media = Media::new();
    media.read_limit = Duration::from_millis(200);
    media.decode = Arc::new(move |path| {
        if path.file_name().is_some_and(|n| n == "SLOW.ARW") {
            held.wait();
        }
        ym_raw::decode::decode(path)
    });
    let session = session_with(media);
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    assert_eq!(
        status_of(&session, "SLOW.ARW"),
        PhotoStatus::Skipped {
            reason: NO_ANSWER_MESSAGE.to_owned()
        }
    );
    assert_eq!(status_of(&session, "FAST.ARW"), PhotoStatus::Ready);
    gate.release();
}

#[test]
fn progress_is_reported_at_the_start_every_fifty_files_and_at_the_end() {
    let dir = tempfile::tempdir().expect("dir");
    write_empty(dir.path(), 120, "P");
    let mut media = Media::new();
    media.probe = Arc::new(|_| Err(ym_raw::error::RawError::NotArw));
    let session = session_with(media);
    let seen = Mutex::new(Vec::new());
    let outcome = session.open_with_progress(
        &[dir.path().to_path_buf()],
        OpenOptions::default(),
        &|p: OpenProgress| seen.lock().unwrap().push((p.done, p.total)),
    );
    assert_eq!(outcome.added.len(), 120);
    let mut seen = seen.into_inner().unwrap();
    seen.sort_unstable();
    assert_eq!(OPEN_PROGRESS_STEP, 50);
    assert_eq!(seen, [(0, 120), (50, 120), (100, 120), (120, 120)]);
}

#[test]
fn cancelling_an_open_keeps_the_files_read_so_far() {
    let dir = tempfile::tempdir().expect("dir");
    write_empty(dir.path(), 70, "F");
    // Sorts last, and never answers.
    write_empty(dir.path(), 1, "zzz");
    let gate = Gate::default();
    let held = gate.clone();
    let mut media = Media::new();
    media.probe = Arc::new(move |path| {
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("zzz"))
        {
            held.wait();
        }
        Err(ym_raw::error::RawError::NotArw)
    });
    let session = Arc::new(session_with(media));
    let reported = Arc::new(AtomicUsize::new(0));
    let opening = {
        let session = Arc::clone(&session);
        let folder = dir.path().to_path_buf();
        let reported = Arc::clone(&reported);
        std::thread::spawn(move || {
            session.open_with_progress(&[folder], OpenOptions::default(), &|p: OpenProgress| {
                reported.fetch_max(p.done, Ordering::SeqCst);
            })
        })
    };
    wait_for("the hung file and fifty others", || {
        gate.entered() > 0 && reported.load(Ordering::SeqCst) >= OPEN_PROGRESS_STEP
    });
    session.cancel_open();
    let outcome = opening.join().expect("the open");
    gate.release();
    assert!(outcome.cancelled);
    assert!(
        (OPEN_PROGRESS_STEP..71).contains(&outcome.added.len()),
        "{} files were kept",
        outcome.added.len()
    );
    assert_eq!(session.state().photos.len(), outcome.added.len());
    assert!(
        session
            .state()
            .photos
            .iter()
            .all(|p| !p.file_name.starts_with("zzz")),
        "the file that was being waited for is dropped"
    );
    // The next open is not cancelled.
    let again = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert!(!again.cancelled);
}
