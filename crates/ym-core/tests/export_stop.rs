//! A job stops at a failure of the destination (`docs/26-hardening.md` P2-2, task H-06): the disk is
//! full, the folder is gone or read-only. The photos after the failing one are neither decoded nor
//! rendered, no temporary file is left, and the summary and the events say what happened.
mod common;

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use common::*;
use ym_core::export::ExportContext;
use ym_core::export_queue::{
    ExportEvent, ExportJob, ExportProgress, ExportQueue, ExportSink, FileDone, JobState,
};
use ym_core::pipeline::Renderer;
use ym_core::session::{ApplyScope, OpenOptions, PhotoId, Session};
use ym_export::ExportOptions;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

struct Folder {
    _dir: tempfile::TempDir,
    session: Arc<Session>,
    ids: Vec<PhotoId>,
}

fn folder(n: usize) -> Folder {
    let dir = tempfile::tempdir().expect("dir");
    for k in 0..n {
        let scene = Scene::new(
            300,
            200,
            light(PROFILE, 4500.0 + 500.0 * k as f64),
            Layout::Patchwork,
            0.3 + 0.1 * k as f64,
        )
        .noise(2e-4, 1e-5)
        .seed(k as u64 + 1);
        scene.write(dir.path(), &format!("DSC{:05}.ARW", k + 1), MODEL);
    }
    let session = Arc::new(Session::new(base_env()));
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(out.added.len(), n);
    settle(&session);
    Folder {
        _dir: dir,
        session,
        ids: out.added,
    }
}

fn queue(home: &Path) -> (Arc<ExportContext>, ExportQueue, Receiver<ExportEvent>) {
    let renderer = Renderer::neutral_only().expect("render thread");
    let ctx = Arc::new(ExportContext::new(renderer).with_home(Some(home.to_path_buf())));
    let (sink, rx) = ExportSink::channel();
    let queue = ExportQueue::new(Arc::clone(&ctx), sink);
    (ctx, queue, rx)
}

fn job(out: &Path) -> ExportJob {
    ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            destination: Some(out.to_path_buf()),
            ..ExportOptions::default()
        },
    }
}

fn until_idle(rx: &Receiver<ExportEvent>) -> Vec<ExportEvent> {
    let mut events = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let e = rx.recv_timeout(left).expect("the queue went idle in time");
        let idle = e == ExportEvent::Busy(false);
        events.push(e);
        if idle {
            return events;
        }
    }
}

fn progress(events: &[ExportEvent]) -> Vec<&ExportProgress> {
    events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::Progress(p) => Some(p),
            _ => None,
        })
        .collect()
}

fn files(events: &[ExportEvent]) -> Vec<&FileDone> {
    events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::File(f) => Some(f),
            _ => None,
        })
        .collect()
}

/// The write fault is one switch for the whole process: tests that arm it run one at a time.
#[cfg(feature = "fault-injection")]
static FAULT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// How many photos of `ids` the renderer made a full-resolution chain for.
fn rendered(ctx: &ExportContext, ids: &[PhotoId]) -> Vec<bool> {
    ids.iter()
        .map(|id| ctx.renderer().counters(*id).full_chains > 0)
        .collect()
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_full_disk_stops_the_job_after_the_photo_it_hit() {
    use ym_export::fault::{WriteFault, arm};

    let _one_at_a_time = FAULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let f = folder(5);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, rx) = queue(home.path());
    let decodes_before = f.session.raw_decodes();
    // The first photo is written; the second one hits a full disk.
    arm(Some(WriteFault {
        at_bytes: 0,
        after_files: 1,
    }));
    let id = queue
        .start(&f.session, &job(out.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    arm(None);

    let message = "The disk is full, so the export stopped. Free some space, then export the photos that are left.";
    let p = progress(&events);
    let stopped: Vec<_> = p.iter().filter(|e| e.stopped).collect();
    assert_eq!(stopped.len(), 1, "{p:?}");
    assert_eq!(stopped[0].file, "DSC00002_neutral.jpg");
    assert_eq!((stopped[0].done, stopped[0].total), (2, 5));
    assert_eq!(stopped[0].error.as_deref(), Some(message));
    // It is the job's last progress event, and nothing follows for the photos after it.
    assert!(p.last().is_some_and(|e| e.stopped), "{p:?}");
    assert!(p.iter().all(|e| e.done <= 2), "{p:?}");

    let fd = files(&events);
    assert_eq!(
        fd.len(),
        2,
        "a File event for the saved and the failed photo only"
    );
    assert_eq!(fd[0].not_tried, 0);
    let failure = fd[1].outcome.as_ref().expect_err("the second photo failed");
    assert_eq!(
        (failure.kind, failure.stopped, fd[1].not_tried),
        ("disk_full", true, 3)
    );
    assert_eq!(failure.exit_code, 1);
    assert!(
        failure
            .cli_message
            .starts_with("Stopped: the disk holding ")
    );

    let s = queue.summary(id).expect("summary");
    assert_eq!(
        (s.state, s.saved, s.failed, s.skipped, s.done, s.not_tried),
        (JobState::Finished, 1, 1, 0, 2, 3)
    );
    assert_eq!(s.stopped.as_deref(), Some(message));
    assert_eq!(s.errors.len(), 1);

    // The saved photo is there; no temporary file is left behind.
    assert_eq!(names_in(out.path()), vec!["DSC00001_neutral.jpg"]);

    // Nothing was rendered for a photo after the failing one, and at most the one decode that
    // overlapped the failing photo's render was started.
    assert!(
        rendered(&ctx, &f.ids)[..2].iter().all(|r| *r),
        "the counter sees the photos that were rendered"
    );
    assert_eq!(rendered(&ctx, &f.ids)[2..], [false, false, false]);
    let decoded = f.session.raw_decodes() - decodes_before;
    assert!(
        decoded <= 3,
        "decoded {decoded} photos for a job that stopped at the second"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_disk_that_is_full_from_the_start_renders_one_photo_and_stops() {
    use ym_export::fault::{WriteFault, arm};

    let _one_at_a_time = FAULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let f = folder(5);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, rx) = queue(home.path());
    arm(Some(WriteFault {
        at_bytes: 0,
        after_files: 0,
    }));
    let id = queue
        .start(&f.session, &job(out.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    arm(None);

    let s = queue.summary(id).expect("summary");
    assert_eq!(
        (s.state, s.saved, s.failed, s.done, s.not_tried),
        (JobState::Finished, 0, 1, 1, 4)
    );
    assert!(s.stopped.is_some());
    assert_eq!(progress(&events).iter().filter(|e| e.stopped).count(), 1);
    assert!(
        names_in(out.path()).is_empty(),
        "no file and no temporary file"
    );
    assert!(
        rendered(&ctx, &f.ids)[0],
        "the counter sees the first photo"
    );
    assert_eq!(rendered(&ctx, &f.ids)[1..], [false, false, false, false]);
}

/// A folder that is not there and cannot be made is found before the first photo is decoded.
#[cfg(target_os = "linux")]
#[test]
fn a_destination_that_is_gone_is_found_before_any_photo_is_decoded() {
    let f = folder(4);
    let home = tempfile::tempdir().expect("h");
    let (ctx, queue, rx) = queue(home.path());
    let decodes_before = f.session.raw_decodes();
    // Nothing can be made under /proc.
    let gone = Path::new("/proc/yellowmoon-no-such-folder/out");
    let id = queue.start(&f.session, &job(gone), false).expect("queued");
    let events = until_idle(&rx);

    let s = queue.summary(id).expect("summary");
    assert_eq!(
        (s.state, s.saved, s.failed, s.done, s.not_tried),
        (JobState::Finished, 0, 1, 1, 3)
    );
    assert!(
        s.stopped
            .as_deref()
            .is_some_and(|m| m.starts_with("Yellowmoon can't reach "))
    );
    let fd = files(&events);
    assert_eq!(fd.len(), 1);
    let failure = fd[0].outcome.as_ref().expect_err("failed");
    assert_eq!(
        (failure.kind, failure.stopped, fd[0].not_tried),
        ("destination", true, 3)
    );
    assert_eq!(
        f.session.raw_decodes(),
        decodes_before,
        "no photo was decoded"
    );
    assert_eq!(rendered(&ctx, &f.ids), [false; 4], "no photo was rendered");
    assert!(progress(&events).last().is_some_and(|e| e.stopped));
}

/// The snapshot of the first photo, for the calls that work on one photo.
fn first_snapshot(f: &Folder) -> ym_core::export::PhotoSnapshot {
    ym_core::export::PhotoSnapshot::take(&f.session, f.ids[0]).expect("snapshot")
}

fn set_age(path: &Path, hours: u64) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open");
    file.set_modified(std::time::SystemTime::now() - Duration::from_secs(hours * 3600))
        .expect("set mtime");
}

/// The start of a job removes the writer's temporary files in the destination that are older than 24 hours,
/// and nothing else (`docs/26-hardening.md` P2-4).
#[test]
fn the_start_of_a_job_removes_stale_temporary_files_in_the_destination() {
    let f = folder(1);
    let home = tempfile::tempdir().expect("h");
    let out = tempfile::tempdir().expect("o");
    let (ctx, _queue, _rx) = queue(home.path());
    let write = |name: &str, hours: u64| {
        let p = out.path().join(name);
        std::fs::write(&p, b"half a file").expect("write");
        set_age(&p, hours);
    };
    write(".yellowmoon-4242-0.tmp", 25);
    write(".yellowmoon-4242-1.tmp", 23);
    write(".yellowmoon-4242-2.tmp.old", 200);
    write("holiday.jpg", 200);
    let options = job(out.path()).options;
    ym_core::export::check_destination(&ctx, &first_snapshot(&f), &options, 1).expect("ready");
    assert_eq!(
        names_in(out.path()),
        [
            ".yellowmoon-4242-1.tmp",
            ".yellowmoon-4242-2.tmp.old",
            "holiday.jpg"
        ]
    );
}

/// Only the first photo of a job makes the folder; after that a folder that has gone is "Folder gone" and is
/// not made again.
#[test]
fn a_folder_deleted_after_the_first_photo_is_not_made_again() {
    let f = folder(2);
    let home = tempfile::tempdir().expect("h");
    let parent = tempfile::tempdir().expect("o");
    let out = parent.path().join("out");
    let (ctx, _queue, _rx) = queue(home.path());
    let options = job(&out).options;
    let snap = first_snapshot(&f);

    // The first photo makes it.
    ym_core::export::check_destination(&ctx, &snap, &options, 1).expect("made");
    assert!(out.is_dir());
    // The person deletes it; the second photo finds it gone and does not make it.
    std::fs::remove_dir_all(&out).expect("deleted");
    let err = ym_core::export::check_destination(&ctx, &snap, &options, 2).expect_err("gone");
    assert_eq!(
        err.destination_problem(),
        Some(ym_export::DestinationProblem::Gone)
    );
    assert!(!out.exists(), "not made again");
}

/// The same through the queue: the folder is removed while a job is running and the job stops with the
/// "Folder gone" sentence.
#[cfg(target_os = "linux")]
#[test]
fn a_job_whose_folder_is_deleted_while_it_runs_stops_with_folder_gone() {
    // A fault armed by another test of this file must not reach this job's writes.
    #[cfg(feature = "fault-injection")]
    let _one_at_a_time = FAULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let f = folder(6);
    let home = tempfile::tempdir().expect("h");
    let parent = tempfile::tempdir().expect("o");
    let out = parent.path().join("out");
    let (_ctx, queue, rx) = queue(home.path());
    let id = queue.start(&f.session, &job(&out), false).expect("queued");
    let mut events = Vec::new();
    let mut removed = false;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let e = rx.recv_timeout(left).expect("the queue went idle in time");
        // The first photo is done: the folder goes. (A write can race with the removal; try again.)
        if !removed && matches!(e, ExportEvent::File(_)) {
            for _ in 0..50 {
                if std::fs::remove_dir_all(&out).is_ok() {
                    removed = true;
                    break;
                }
            }
        }
        let idle = e == ExportEvent::Busy(false);
        events.push(e);
        if idle {
            break;
        }
    }
    assert!(removed);
    let s = queue.summary(id).expect("summary");
    assert!(s.saved >= 1 && s.saved < 6, "saved {}", s.saved);
    assert!(
        s.stopped
            .as_deref()
            .is_some_and(|m| m.starts_with("Yellowmoon can't reach out")),
        "{:?}",
        s.stopped
    );
    assert!(!out.exists(), "the folder was not made again");
}
