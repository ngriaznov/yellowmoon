//! The export queue: snapshots at queue time, one photo at a time, progress events in the order
//! the panel expects, cancel, errors per file, the busy flag for the updater, estimates.
mod common;

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use common::*;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::export_queue::{
    ExportEvent, ExportJob, ExportProgress, ExportQueue, ExportSink, FileDone, JobState, QueueError,
};
use ym_core::pipeline::{
    FrameOptions, LookProblem, LookProvider, NeutralOnly, RenderOptions, Renderer, ResolvedLook,
    ViewportRect, ViewportState,
};
use ym_core::session::{ApplyScope, OpenOptions, PhotoId, Session};
use ym_core::settings::SettingsPatch;
use ym_export::{ExportOptions, IfExists};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

struct Folder {
    _dir: tempfile::TempDir,
    session: Arc<Session>,
    ids: Vec<PhotoId>,
}

/// `n` photos, `DSC00001.ARW` and on, of `w` x `h`, in one folder, opened and developed.
fn folder(n: usize, w: u32, h: u32) -> Folder {
    let dir = tempfile::tempdir().expect("dir");
    for k in 0..n {
        let scene = Scene::new(
            w,
            h,
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

fn queue_over(
    renderer: Renderer,
    home: &Path,
) -> (Arc<ExportContext>, ExportQueue, Receiver<ExportEvent>) {
    let ctx = Arc::new(ExportContext::new(renderer).with_home(Some(home.to_path_buf())));
    let (sink, rx) = ExportSink::channel();
    let queue = ExportQueue::new(Arc::clone(&ctx), sink);
    (ctx, queue, rx)
}

fn neutral_queue(home: &Path) -> (Arc<ExportContext>, ExportQueue, Receiver<ExportEvent>) {
    queue_over(Renderer::neutral_only().expect("render thread"), home)
}

fn job(scope: ApplyScope, out: &Path) -> ExportJob {
    ExportJob {
        scope,
        options: ExportOptions {
            destination: Some(out.to_path_buf()),
            ..ExportOptions::default()
        },
    }
}

/// Every event until the queue reports idle (fails after two minutes).
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

fn files_done(events: &[ExportEvent]) -> Vec<&FileDone> {
    events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::File(f) => Some(f),
            _ => None,
        })
        .collect()
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read")
}

// ---------------------------------------------------------------------------------------------

#[test]
fn a_job_exports_every_photo_in_order_and_reports_what_the_panel_expects() {
    let f = folder(3, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    assert!(!queue.is_busy());

    let id = queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    assert!(queue.is_busy(), "busy from the moment the job is queued");
    let events = until_idle(&rx);
    assert!(!queue.is_busy());

    // Busy first, idle last, nothing after.
    assert_eq!(events.first(), Some(&ExportEvent::Busy(true)));
    assert_eq!(events.last(), Some(&ExportEvent::Busy(false)));
    assert!(rx.try_recv().is_err(), "no event after idle");

    let p = progress(&events);
    assert!(p.iter().all(|e| e.job_id == id && e.total == 3));
    // Each photo starts at 0 % under its own name with the number of files finished before it.
    for (i, name) in [
        "DSC00001_neutral.jpg",
        "DSC00002_neutral.jpg",
        "DSC00003_neutral.jpg",
    ]
    .into_iter()
    .enumerate()
    {
        let first = p
            .iter()
            .find(|e| e.file == name)
            .unwrap_or_else(|| panic!("an event for {name}"));
        assert_eq!((first.done, first.file_percent), (i, Some(0.0)), "{name}");
        assert!(first.error.is_none() && !first.skipped);
    }
    // Counts and percentages never go backwards.
    for pair in p.windows(2) {
        assert!(pair[1].done >= pair[0].done, "{pair:?}");
        if pair[1].done == pair[0].done && pair[1].file == pair[0].file {
            assert!(pair[1].file_percent >= pair[0].file_percent, "{pair:?}");
        }
    }
    // The last event says everything is done.
    let last = p.last().expect("progress");
    assert_eq!(
        (last.done, last.total, last.file_percent),
        (3, 3, Some(100.0))
    );
    assert_eq!(last.file, "DSC00003_neutral.jpg");

    assert_eq!(
        names_in(out.path()),
        vec![
            "DSC00001_neutral.jpg",
            "DSC00002_neutral.jpg",
            "DSC00003_neutral.jpg"
        ]
    );
    let s = queue.summary(id).expect("summary");
    assert_eq!(s.state, JobState::Finished);
    assert_eq!(
        (s.total, s.done, s.saved, s.skipped, s.failed),
        (3, 3, 3, 0, 0)
    );
    assert_eq!(s.files.len(), 3);
    assert_eq!(s.folders, vec![out.path().to_path_buf()]);
    assert_eq!(
        queue.reveal_path(id),
        Some(out.path().to_path_buf()),
        "several files: the folder"
    );
}

#[test]
fn a_job_of_one_photo_reveals_the_file() {
    let f = folder(2, 200, 150);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    f.session.set_current(f.ids[1]).expect("current");
    let id = queue
        .start(&f.session, &job(ApplyScope::This, out.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    let p = progress(&events);
    assert_eq!(p.last().map(|e| (e.done, e.total)), Some((1, 1)));
    assert_eq!(
        queue.reveal_path(id),
        Some(out.path().join("DSC00002_neutral.jpg"))
    );
    assert_eq!(names_in(out.path()), vec!["DSC00002_neutral.jpg"]);
}

#[test]
fn settings_are_snapshotted_when_the_job_is_queued() {
    let f = folder(3, 300, 200);
    let (home, out, expected_dir) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
        tempfile::tempdir().expect("e"),
    );
    let (ctx, queue, rx) = neutral_queue(home.path());

    // What the three photos export as they are now, made directly (the CLI's path).
    let options = job(ApplyScope::All, out.path()).options;
    let mut expected = Vec::new();
    for (i, id) in f.ids.iter().enumerate() {
        let photo = ExportPhoto::from_session(&f.session, *id).expect("photo");
        let report = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(ExportOptions {
                destination: Some(expected_dir.path().to_path_buf()),
                ..options.clone()
            })
            .seq(i as u32 + 1),
            &ExportControl::none(),
        )
        .expect("direct export");
        expected.push(read(&report.path));
    }

    queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    // Edit every photo at once: none of it may reach the job.
    for id in &f.ids {
        f.session
            .update_settings(
                *id,
                &SettingsPatch {
                    saturation: Some(60.0),
                    contrast: Some(-30.0),
                    ..SettingsPatch::exposure_ev(1.5)
                },
            )
            .expect("edit");
    }
    until_idle(&rx);
    for (i, bytes) in expected.iter().enumerate() {
        let name = format!("DSC{:05}_neutral.jpg", i + 1);
        assert_eq!(
            &read(&out.path().join(&name)),
            bytes,
            "{name}: the queue and a direct call differ"
        );
    }
    // And the edit was real: a new job now exports something else.
    let again = tempfile::tempdir().expect("again");
    queue
        .start(&f.session, &job(ApplyScope::This, again.path()), false)
        .expect("queued");
    until_idle(&rx);
    assert_ne!(
        read(&again.path().join("DSC00001_neutral.jpg")),
        expected[0]
    );
}

#[test]
fn the_job_goes_on_when_the_session_is_closed() {
    let f = folder(3, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    f.session.close();
    let events = until_idle(&rx);
    assert_eq!(progress(&events).last().map(|e| e.done), Some(3));
    assert_eq!(names_in(out.path()).len(), 3);
}

#[test]
fn cancel_stops_the_job_and_deletes_the_partial_file() {
    let f = folder(3, 1100, 700);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    let id = queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    // Wait until the first photo is under way, then cancel.
    loop {
        match rx.recv_timeout(Duration::from_secs(60)).expect("progress") {
            ExportEvent::Progress(p) if p.file_percent.is_some_and(|x| x > 0.0) => break,
            _ => {}
        }
    }
    queue.cancel(id);
    let rest = until_idle(&rx);
    assert!(!queue.is_busy());
    // A cancelled job sends no "done" and nothing for the photos it never reached.
    assert!(progress(&rest).iter().all(|p| p.done < 3), "{rest:?}");
    let names = names_in(out.path());
    assert!(
        names
            .iter()
            .all(|n| n.ends_with("_neutral.jpg") && !n.contains(".tmp")),
        "no temporary file is left: {names:?}"
    );
    assert!(names.len() < 3, "{names:?}");
    let s = queue.summary(id).expect("summary");
    assert_eq!(s.state, JobState::Cancelled);
    // The queue is not stuck: a new job runs.
    let again = tempfile::tempdir().expect("again");
    queue
        .start(&f.session, &job(ApplyScope::This, again.path()), false)
        .expect("queued");
    until_idle(&rx);
    assert_eq!(names_in(again.path()).len(), 1);
}

#[test]
fn cancelling_a_job_that_has_not_started_drops_it() {
    let f = folder(2, 1100, 700);
    let (home, out1, out2) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o1"),
        tempfile::tempdir().expect("o2"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    let first = queue
        .start(&f.session, &job(ApplyScope::All, out1.path()), false)
        .expect("queued");
    let second = queue
        .start(&f.session, &job(ApplyScope::All, out2.path()), false)
        .expect("queued");
    queue.cancel(second);
    assert_eq!(
        queue.summary(second).map(|s| s.state),
        Some(JobState::Cancelled)
    );
    let events = until_idle(&rx);
    assert!(progress(&events).iter().all(|p| p.job_id == first));
    assert_eq!(names_in(out1.path()).len(), 2);
    assert!(names_in(out2.path()).is_empty());
    // Exactly one busy period.
    assert_eq!(
        events
            .iter()
            .filter(|e| **e == ExportEvent::Busy(true))
            .count(),
        1
    );
}

#[test]
fn jobs_run_one_after_another() {
    let f = folder(2, 300, 200);
    let (home, out1, out2) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o1"),
        tempfile::tempdir().expect("o2"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    let a = queue
        .start(&f.session, &job(ApplyScope::All, out1.path()), false)
        .expect("queued");
    let b = queue
        .start(&f.session, &job(ApplyScope::All, out2.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    let order: Vec<u64> = progress(&events).iter().map(|p| p.job_id).collect();
    let switch = order.iter().position(|j| *j == b).expect("job b reported");
    assert!(order[..switch].iter().all(|j| *j == a));
    assert!(order[switch..].iter().all(|j| *j == b));
    assert_eq!(
        events
            .iter()
            .filter(|e| **e == ExportEvent::Busy(true))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| **e == ExportEvent::Busy(false))
            .count(),
        1
    );
    assert_eq!(names_in(out1.path()).len(), 2);
    assert_eq!(names_in(out2.path()).len(), 2);
    assert!(queue.wait_idle(Duration::from_secs(1)));
}

// ---------------------------------------------------------------------------------------------
// Errors and skips per file
// ---------------------------------------------------------------------------------------------

struct Looks;

impl LookProvider for Looks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        match id.as_str() {
            "builtin:neutral" => NeutralOnly.look(id),
            _ => Err(LookProblem(
                "File missing or changed. Import again.".to_owned(),
            )),
        }
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

#[test]
fn one_photo_that_fails_does_not_stop_the_job() {
    let f = folder(3, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let renderer = Renderer::new(Arc::new(Looks), RenderOptions::default()).expect("render thread");
    let (_ctx, queue, rx) = queue_over(renderer, home.path());
    // The second photo got a look whose file has since gone.
    f.session
        .update_settings(
            f.ids[1],
            &SettingsPatch {
                look: Some(LookId::parse("arri:3000:gone").expect("id")),
                ..SettingsPatch::default()
            },
        )
        .expect("edit");
    let id = queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    let p = progress(&events);
    let failed = p
        .iter()
        .find(|e| e.error.is_some())
        .expect("an error event");
    // `{look}` is the slug of the look the photo has, gone or not.
    assert_eq!(failed.file, "DSC00002_gone.jpg");
    assert_eq!(failed.done, 2, "the failed file counts as finished");
    assert_eq!(
        failed.error.as_deref(),
        Some("File missing or changed. Import again.")
    );
    assert_eq!(p.last().map(|e| (e.done, e.total)), Some((3, 3)));
    assert_eq!(
        names_in(out.path()),
        vec!["DSC00001_neutral.jpg", "DSC00003_neutral.jpg"]
    );
    let s = queue.summary(id).expect("summary");
    assert_eq!((s.saved, s.failed, s.skipped, s.done), (2, 1, 0, 3));
    assert_eq!(s.state, JobState::Finished);
    assert_eq!(s.errors.len(), 1);
    assert_eq!(s.errors[0].file, "DSC00002_gone.jpg");
}

#[test]
fn a_file_that_exists_with_skip_is_reported_as_skipped() {
    let f = folder(2, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    std::fs::write(out.path().join("DSC00001_neutral.jpg"), b"already there").expect("existing");
    let mut j = job(ApplyScope::All, out.path());
    j.options.if_exists = IfExists::Skip;
    let id = queue.start(&f.session, &j, false).expect("queued");
    let events = until_idle(&rx);
    let p = progress(&events);
    let skipped = p.iter().find(|e| e.skipped).expect("a skipped event");
    assert_eq!(
        (skipped.file.as_str(), skipped.done),
        ("DSC00001_neutral.jpg", 1)
    );
    assert!(skipped.error.is_none());
    assert_eq!(
        read(&out.path().join("DSC00001_neutral.jpg")),
        b"already there"
    );
    let s = queue.summary(id).expect("summary");
    assert_eq!((s.saved, s.skipped, s.failed), (1, 1, 0));
}

#[test]
fn keep_both_names_the_file_with_its_number_before_it_starts() {
    let f = folder(1, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    std::fs::write(out.path().join("DSC00001_neutral.jpg"), b"x").expect("existing");
    queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    let p = progress(&events);
    assert!(
        p.iter().all(|e| e.file == "DSC00001_neutral 2.jpg"),
        "{p:?}"
    );
    assert!(out.path().join("DSC00001_neutral 2.jpg").is_file());
}

#[test]
fn a_job_that_cannot_be_queued_says_why() {
    let f = folder(2, 200, 150);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, _rx) = neutral_queue(home.path());
    let err = queue
        .start(&f.session, &job(ApplyScope::Picked, out.path()), false)
        .expect_err("nothing picked");
    assert!(matches!(err, QueueError::NoPhotos(_)));
    assert_eq!(
        err.to_string(),
        "No photos are picked. Pick some photos, or choose All, then export again."
    );
    let mut bad = job(ApplyScope::All, out.path());
    bad.options.size = ym_export::Size::LongEdge { px: 3 };
    assert!(matches!(
        queue.start(&f.session, &bad, false),
        Err(QueueError::Options(_))
    ));
    assert!(!queue.is_busy());
    // Picking works as a scope.
    f.session.set_picked(f.ids[1], true).expect("pick");
    let id = queue
        .start(&f.session, &job(ApplyScope::Picked, out.path()), false)
        .expect("queued");
    assert_eq!(queue.summary(id).map(|s| s.total), Some(1));
    assert!(queue.wait_idle(Duration::from_secs(60)));
    assert_eq!(names_in(out.path()), vec!["DSC00002_neutral.jpg"]);
}

// ---------------------------------------------------------------------------------------------
// Browsing while exporting, estimates
// ---------------------------------------------------------------------------------------------

#[test]
fn the_viewport_stays_responsive_while_a_job_runs() {
    let f = folder(3, 1100, 700);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, rx) = neutral_queue(home.path());
    let id = queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    // Wait until a photo is rendering.
    loop {
        if let ExportEvent::Progress(_) =
            rx.recv_timeout(Duration::from_secs(60)).expect("progress")
        {
            break;
        }
    }
    let view = f.session.view(f.ids[2]).expect("view");
    let raw = f.session.raw_image(f.ids[2]).expect("raw");
    let src = ym_core::pipeline::PhotoSource::from_view(&view, raw);
    let started = Instant::now();
    let frame = ctx
        .renderer()
        .spawn_viewport_frame(
            src,
            ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0)),
            FrameOptions::default(),
        )
        .wait()
        .expect("scheduled")
        .expect("frame");
    let took = started.elapsed();
    assert_eq!((frame.width, frame.height), (300, 200));
    assert_eq!(
        queue.summary(id).map(|s| s.state),
        Some(JobState::Running),
        "the frame came back while the export was still going (took {took:?})"
    );
    assert!(queue.wait_idle(Duration::from_secs(120)));
}

#[test]
fn the_estimate_counts_bytes_seconds_and_what_a_command_cannot_carry() {
    let f = folder(3, 600, 400);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, rx) = neutral_queue(home.path());

    let jpeg = job(ApplyScope::All, out.path());
    let est = queue.estimate(&f.session, &jpeg);
    assert_eq!(est.adjusted_photos, 0);
    assert!(!est.dng_unavailable);
    assert!(est.bytes > 3 * 4096 && est.seconds > 0.0, "{est:?}");

    // A bigger file for a higher quality, a smaller one for a smaller size.
    let mut hq = jpeg.clone();
    hq.options.jpeg_quality = ym_export::JpegQuality::new(100).expect("q");
    assert!(queue.estimate(&f.session, &hq).bytes > est.bytes);
    let mut small = jpeg.clone();
    small.options.size = ym_export::Size::LongEdge { px: 300 };
    assert!(queue.estimate(&f.session, &small).bytes < est.bytes);
    // Scopes count their photos.
    let one = queue.estimate(
        &f.session,
        &ExportJob {
            scope: ApplyScope::This,
            ..jpeg.clone()
        },
    );
    assert!(one.bytes < est.bytes && one.seconds < est.seconds);

    // Adjustments beyond the look are what a command line cannot carry.
    f.session
        .update_settings(
            f.ids[0],
            &SettingsPatch {
                saturation: Some(10.0),
                ..SettingsPatch::default()
            },
        )
        .expect("edit");
    assert_eq!(queue.estimate(&f.session, &jpeg).adjusted_photos, 1);
    f.session
        .update_settings(f.ids[1], &SettingsPatch::exposure_ev(0.4))
        .expect("edit");
    assert_eq!(queue.estimate(&f.session, &jpeg).adjusted_photos, 2);

    // The estimate is the right order of magnitude. It comes from film scans; a flat synthetic
    // patchwork compresses three times better than that, so the bound is wide.
    assert!(
        !ctx.speed().is_measured(),
        "nothing exported yet: the starting speed"
    );
    queue.start(&f.session, &jpeg, false).expect("queued");
    until_idle(&rx);
    let actual: u64 = names_in(out.path())
        .iter()
        .map(|n| std::fs::metadata(out.path().join(n)).expect("meta").len())
        .sum();
    let ratio = est.bytes as f64 / actual as f64;
    eprintln!(
        "estimate {} B, actual {actual} B, ratio {ratio:.2}; {:.2} s",
        est.bytes, est.seconds
    );
    assert!((0.25..6.0).contains(&ratio), "ratio {ratio}");
    // The speed of this machine replaced the starting numbers.
    assert!(ctx.speed().is_measured());
    let measured = queue.estimate(&f.session, &jpeg).seconds;
    assert!(measured > 0.0 && (measured - est.seconds).abs() > 1e-9);
}

#[test]
fn a_destination_that_cannot_be_written_fails_each_photo_with_a_plain_message() {
    let f = folder(2, 200, 150);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    // A file where the folder should be.
    let blocker = out.path().join("blocker");
    std::fs::write(&blocker, b"x").expect("blocker");
    let id = queue
        .start(
            &f.session,
            &job(ApplyScope::All, &blocker.join("exports")),
            false,
        )
        .expect("queued");
    let events = until_idle(&rx);
    let p = progress(&events);
    let errors: Vec<_> = p.iter().filter_map(|e| e.error.as_deref()).collect();
    assert_eq!(errors.len(), 2, "{p:?}");
    assert!(
        errors
            .iter()
            .all(|e| e.starts_with("Couldn't write to") && e.contains("Check that the folder")),
        "{errors:?}"
    );
    assert_eq!(p.last().map(|e| (e.done, e.total)), Some((2, 2)));
    let s = queue.summary(id).expect("summary");
    assert_eq!((s.saved, s.failed, s.state), (0, 2, JobState::Finished));
    assert_eq!(queue.reveal_path(id), None, "nothing was written");
}

#[test]
fn photos_with_the_same_name_from_two_folders_keep_both() {
    let a = folder(1, 200, 150);
    let b = folder(1, 200, 150);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    // One session over both folders.
    let session = Arc::new(Session::new(base_env()));
    session.open(&[a._dir.path().to_path_buf()], OpenOptions::default());
    session.open(&[b._dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    assert_eq!(session.state().photos.len(), 2);
    let id = queue
        .start(&session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");
    until_idle(&rx);
    assert_eq!(
        names_in(out.path()),
        vec!["DSC00001_neutral 2.jpg", "DSC00001_neutral.jpg"]
    );
    assert_eq!(
        queue.summary(id).map(|s| (s.saved, s.files.len())),
        Some((2, 2))
    );
    // Without a chosen folder each goes next to its own source folder.
    let id = queue
        .start(&session, &jpeg_default_destination(), false)
        .expect("queued");
    until_idle(&rx);
    let s = queue.summary(id).expect("summary");
    assert_eq!(s.folders.len(), 2, "{:?}", s.folders);
    assert!(
        s.folders
            .iter()
            .all(|f| f.to_string_lossy().ends_with(" export"))
    );
    assert_eq!(queue.reveal_path(id), s.folders.first().cloned());
    for f in &s.folders {
        let _ = std::fs::remove_dir_all(f);
    }
}

fn jpeg_default_destination() -> ExportJob {
    ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions::default(),
    }
}

#[test]
fn each_photo_gets_a_file_event_with_the_report_for_the_clis_json() {
    let f = folder(3, 300, 200);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let renderer = Renderer::new(Arc::new(Looks), RenderOptions::default()).expect("render thread");
    let (_ctx, queue, rx) = queue_over(renderer, home.path());
    std::fs::write(out.path().join("DSC00003_neutral.jpg"), b"there").expect("existing");
    // The second photo's look has gone; the third's file exists and the job says Skip.
    f.session
        .update_settings(
            f.ids[1],
            &SettingsPatch {
                look: Some(LookId::parse("arri:3000:gone").expect("id")),
                ..SettingsPatch::default()
            },
        )
        .expect("edit");
    let mut j = job(ApplyScope::All, out.path());
    j.options.if_exists = IfExists::Skip;
    let id = queue.start(&f.session, &j, false).expect("queued");
    let events = until_idle(&rx);
    let files = files_done(&events);
    assert_eq!(files.len(), 3);
    for (i, d) in files.iter().enumerate() {
        assert_eq!((d.job_id, d.index, d.total, d.photo), (id, i, 3, f.ids[i]));
    }
    let first = files[0].outcome.as_ref().expect("exported");
    assert_eq!(first.path, out.path().join("DSC00001_neutral.jpg"));
    assert!(!first.skipped && first.bytes > 0 && (first.width, first.height) == (300, 200));
    assert!(first.timings.render_ms > 0.0 && first.timings.encode_ms > 0.0);
    assert!(files[0].decode_ms > 0.0);
    let failure = files[1].outcome.as_ref().expect_err("the look is gone");
    assert_eq!(failure.kind, "look");
    assert_eq!(failure.exit_code, 3);
    assert_eq!(failure.message, "File missing or changed. Import again.");
    let skipped = files[2].outcome.as_ref().expect("skipped is not a failure");
    assert!(skipped.skipped && skipped.bytes == 0);
    // Each File event comes just before the progress event that counts the photo.
    let positions = |pred: &dyn Fn(&ExportEvent) -> bool| -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter(|(_, e)| pred(e))
            .map(|(i, _)| i)
            .collect()
    };
    let file_at = positions(&|e| matches!(e, ExportEvent::File(_)));
    for (n, at) in file_at.iter().enumerate() {
        let ExportEvent::Progress(p) = &events[at + 1] else {
            panic!("a progress event follows the file event");
        };
        assert_eq!(
            p.done,
            n + 1,
            "the progress event counts the photo just reported"
        );
    }
}
