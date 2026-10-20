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
                    saturation: Some(ym_core::settings::PercentSetting::manual_value(60.0)),
                    contrast: Some(ym_core::settings::PercentSetting::manual_value(-30.0)),
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
    // `files` lists what was written; the skipped file is counted, not listed, and its folder is
    // still one of the job's folders.
    assert_eq!(s.files.len(), 1);
    assert!(
        s.files[0].ends_with("DSC00002_neutral.jpg"),
        "{:?}",
        s.files
    );
    assert_eq!(s.folders, vec![out.path().to_path_buf()]);
    assert_eq!(s.reveal_path(), Some(s.files[0].clone()));

    // A job that wrote nothing because every file is there: "Show in Finder" opens the folder.
    let again = queue.start(&f.session, &j, false).expect("queued");
    until_idle(&rx);
    let s = queue.summary(again).expect("summary");
    assert_eq!((s.saved, s.skipped, s.failed), (0, 2, 0));
    assert!(s.files.is_empty());
    assert_eq!(s.reveal_path(), Some(out.path().to_path_buf()));
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
    assert!(!est.dng_approximate);
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
                saturation: Some(ym_core::settings::PercentSetting::manual_value(10.0)),
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
fn a_failed_export_logs_its_kind_and_job_but_no_folder_or_file_name() {
    captured_log(); // installs the capture before the work starts
    let f = folder(1, 200, 150);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    // A file where the folder should be: the error message names the path.
    let blocker = out.path().join("blocker-of-the-folder-to-find-in-the-log");
    std::fs::write(&blocker, b"x").expect("blocker");
    let id = queue
        .start(
            &f.session,
            &job(ApplyScope::All, &blocker.join("Lisbon export")),
            false,
        )
        .expect("queued");
    let events = until_idle(&rx);
    let message = progress(&events)
        .iter()
        .find_map(|e| e.error.clone())
        .expect("the photo failed");
    assert!(
        message.contains("blocker-of-the-folder-to-find-in-the-log"),
        "the person is told where: {message}"
    );

    // The application's log can end up in a bug report: it says what happened, not where. (Tests
    // that run at the same time log into the same capture, so every such line is checked.)
    let log = captured_log();
    let lines: Vec<&str> = log
        .lines()
        .filter(|l| l.contains("export failed"))
        .collect();
    assert!(
        lines.iter().any(|l| l.contains(&format!("job={id}"))),
        "a line for the failed export in:\n{log}"
    );
    for line in lines {
        assert!(
            line.contains("kind="),
            "the line says what kind of failure it was: {line}"
        );
        for private in ["/", "\\", ".jpg", ".ARW", "DSC0", "Lisbon", "blocker"] {
            assert!(
                !line.contains(private),
                "{private:?} reached the log: {line}"
            );
        }
    }
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

/// One session over two folders that each hold a `DSC00001.ARW`: two different pictures with one
/// name (two cards of one body, two bodies that both count from 1).
fn two_pictures_with_one_name() -> (Vec<tempfile::TempDir>, Arc<Session>) {
    let dirs: Vec<tempfile::TempDir> = (0..2u32)
        .map(|k| {
            let dir = tempfile::tempdir().expect("dir");
            Scene::new(
                200,
                150,
                light(PROFILE, 4500.0 + 1500.0 * f64::from(k)),
                Layout::Patchwork,
                0.3 + 0.3 * f64::from(k),
            )
            .noise(2e-4, 1e-5)
            .seed(u64::from(k) + 1)
            .write(dir.path(), "DSC00001.ARW", MODEL);
            dir
        })
        .collect();
    let session = Arc::new(Session::new(base_env()));
    for dir in &dirs {
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    }
    settle(&session);
    assert_eq!(session.state().photos.len(), 2);
    (dirs, session)
}

#[test]
fn two_photos_of_one_job_with_one_name_never_overwrite_each_other_whatever_if_exists_says() {
    let (_dirs, session) = two_pictures_with_one_name();
    let home = tempfile::tempdir().expect("h");
    let (_ctx, queue, rx) = neutral_queue(home.path());
    let both = ["DSC00001_neutral 2.jpg", "DSC00001_neutral.jpg"];

    // Keep both is the reference: the second photo is numbered.
    let reference_dir = tempfile::tempdir().expect("o");
    queue
        .start(&session, &job(ApplyScope::All, reference_dir.path()), false)
        .expect("queued");
    until_idle(&rx);
    assert_eq!(names_in(reference_dir.path()), both);
    let first = read(&reference_dir.path().join("DSC00001_neutral.jpg"));
    let second = read(&reference_dir.path().join("DSC00001_neutral 2.jpg"));
    assert_ne!(first, second, "the two photos are different pictures");

    for if_exists in [IfExists::Replace, IfExists::Skip] {
        let out = tempfile::tempdir().expect("o");
        let mut j = job(ApplyScope::All, out.path());
        j.options.if_exists = if_exists;
        let id = queue.start(&session, &j, false).expect("queued");
        let events = until_idle(&rx);

        assert_eq!(names_in(out.path()), both, "{if_exists:?}");
        assert_eq!(
            read(&out.path().join("DSC00001_neutral.jpg")),
            first,
            "{if_exists:?}: the first photo's file"
        );
        assert_eq!(
            read(&out.path().join("DSC00001_neutral 2.jpg")),
            second,
            "{if_exists:?}: the second photo's file"
        );
        let s = queue.summary(id).expect("summary");
        assert_eq!(
            (s.saved, s.skipped, s.failed),
            (2, 0, 0),
            "{if_exists:?}: both were written"
        );
        assert_eq!(s.files.len(), 2);
        assert_ne!(
            s.files[0], s.files[1],
            "{if_exists:?}: two files, not one twice"
        );
        // The panel is told the name the second photo will have.
        let started: Vec<&str> = progress(&events)
            .iter()
            .filter(|e| e.file_percent == Some(0.0))
            .map(|e| e.file.as_str())
            .collect();
        assert_eq!(
            started,
            ["DSC00001_neutral.jpg", "DSC00001_neutral 2.jpg"],
            "{if_exists:?}"
        );
        assert!(
            progress(&events).iter().all(|e| !e.skipped),
            "{if_exists:?}"
        );
    }
}

#[test]
fn replace_still_replaces_a_file_from_before_the_job_and_skip_still_skips_it() {
    let (_dirs, session) = two_pictures_with_one_name();
    let home = tempfile::tempdir().expect("h");
    let (_ctx, queue, rx) = neutral_queue(home.path());

    // Replace: the file that was there is replaced by the first photo; the second is numbered.
    let out = tempfile::tempdir().expect("o");
    std::fs::write(out.path().join("DSC00001_neutral.jpg"), b"from yesterday").expect("old");
    let mut j = job(ApplyScope::All, out.path());
    j.options.if_exists = IfExists::Replace;
    let id = queue.start(&session, &j, false).expect("queued");
    until_idle(&rx);
    assert_eq!(
        names_in(out.path()),
        ["DSC00001_neutral 2.jpg", "DSC00001_neutral.jpg"]
    );
    assert_ne!(
        read(&out.path().join("DSC00001_neutral.jpg")),
        b"from yesterday"
    );
    let s = queue.summary(id).expect("summary");
    assert_eq!((s.saved, s.skipped, s.failed), (2, 0, 0));

    // Skip: the file from before the job stays, and both photos are skipped because of it (the job
    // wrote nothing, so no file of this job is in the way of anything).
    let out = tempfile::tempdir().expect("o");
    std::fs::write(out.path().join("DSC00001_neutral.jpg"), b"from yesterday").expect("old");
    let mut j = job(ApplyScope::All, out.path());
    j.options.if_exists = IfExists::Skip;
    let id = queue.start(&session, &j, false).expect("queued");
    until_idle(&rx);
    assert_eq!(
        read(&out.path().join("DSC00001_neutral.jpg")),
        b"from yesterday"
    );
    let s = queue.summary(id).expect("summary");
    assert_eq!(
        (s.saved, s.skipped, s.failed),
        (0, 2, 0),
        "both names are taken by the file from before the job"
    );
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

// ---------------------------------------------------------------------------------------------
// Exports queued while the photos they depend on are still developing

/// Four frames of `DSC0000n.ARW`: 1 to 3 a burst of different brightness (their own exposures
/// differ, the burst shares the median), 4 ten seconds later at another ISO (alone in both
/// respects).
fn burst_folder() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("dir");
    for (k, (sec, level, iso)) in [
        ("10", 0.10, 400),
        ("10", 0.20, 400),
        ("11", 0.40, 400),
        ("40", 0.10, 800),
    ]
    .into_iter()
    .enumerate()
    {
        let s = Scene::new(320, 240, light(PROFILE, 5200.0), Layout::Patchwork, level)
            .noise(2e-4, 1e-6)
            .seed(k as u64 + 1);
        s.builder(MODEL)
            .iso(iso)
            .capture_time(&format!("2026:10:04 12:03:{sec}"), "00", None)
            .write_to(
                &dir.path().join(format!("DSC{:05}.ARW", k + 1)),
                &s.mosaic(),
            )
            .expect("write");
    }
    dir
}

/// The bytes the command line's path writes for each photo: open, wait for everything, export.
fn bytes_after_waiting(dir: &Path, scope: ApplyScope) -> Vec<(String, Vec<u8>)> {
    let session = Arc::new(Session::new(base_env()));
    session.open(&[dir.to_path_buf()], OpenOptions::default());
    settle(&session);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    queue
        .start(&session, &job(scope, out.path()), false)
        .expect("queued");
    until_idle(&rx);
    written_files(out.path())
}

fn written_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| {
            let p = e.expect("entry").path();
            (
                p.file_name().expect("name").to_string_lossy().into_owned(),
                read(&p),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn a_job_queued_while_its_burst_is_developing_waits_for_it_and_writes_what_the_command_line_writes()
{
    let dir = burst_folder();
    // The reference: what `ym render` does (wait until everything is developed, then export).
    let want = bytes_after_waiting(dir.path(), ApplyScope::This);
    assert_eq!(want.len(), 1);

    // The app: one decode thread, and the first photo's thumbnail is parked. Photo 1 is estimated;
    // the rest of its burst and its noise pool are still waiting.
    let gate = GateThumbnails::new(1);
    let session = Arc::new(Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(gate.clone()),
    ));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    assert!(session.estimates(ids[0]).is_some() && session.estimates(ids[1]).is_none());
    assert!(!session.estimates_final(ids[0]).unwrap());

    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    let id = queue
        .start(&session, &job(ApplyScope::This, out.path()), false)
        .expect("queued: the photo has estimates, they may still move");
    // The photo's turn has come and it waits: it was announced, nothing is written, the queue is
    // busy (the updater must not restart now).
    let first = rx.recv_timeout(Duration::from_secs(30)).expect("busy");
    assert_eq!(first, ExportEvent::Busy(true));
    std::thread::sleep(Duration::from_millis(400));
    assert!(queue.is_busy());
    assert!(
        written_files(out.path()).is_empty(),
        "nothing is written before the estimates are final"
    );
    assert_eq!(queue.summary(id).expect("summary").state, JobState::Running);

    // The photos it depends on develop; the parked thumbnail goes. The job finishes.
    gate.release();
    let mut events = vec![first];
    events.extend(until_idle(&rx));
    let got = written_files(out.path());
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, want[0].0, "same file name");
    assert!(
        got[0].1 == want[0].1,
        "same bytes as after waiting for idle"
    );
    // The panel saw the usual sequence: the photo starts at 0 % (while it waits), then finishes.
    let p = progress(&events);
    assert_eq!((p[0].done, p[0].total), (0, 1));
    assert_eq!(p.last().map(|p| (p.done, p.total)), Some((1, 1)));
}

#[test]
fn an_export_does_not_wait_for_photos_it_does_not_depend_on() {
    // Photo 4 (another ISO, ten seconds apart) is neither in the burst of photo 1 nor in its noise
    // pool: with photos 1 to 3 estimated and 4 still waiting, an export of photo 1 goes ahead.
    let dir = burst_folder();
    let want = bytes_after_waiting(dir.path(), ApplyScope::This);
    let gate = GateThumbnails::new(3);
    let session = Arc::new(Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(gate.clone()),
    ));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let ids = session.photo_ids();
    assert!(session.estimates(ids[2]).is_some() && session.estimates(ids[3]).is_none());
    assert!(session.estimates_final(ids[0]).unwrap());

    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());
    queue
        .start(&session, &job(ApplyScope::This, out.path()), false)
        .expect("queued");
    until_idle(&rx);
    // Done while the decode thread is still stuck on the parked thumbnail.
    assert_eq!(gate.calls(), 3);
    assert!(session.estimates(ids[3]).is_none());
    let got = written_files(out.path());
    assert_eq!(got.len(), 1);
    assert!(
        got[0].1 == want[0].1,
        "same bytes as after waiting for idle"
    );
    gate.release();
    settle(&session);
}

#[test]
fn a_job_waiting_for_estimates_can_be_cancelled_and_survives_the_session_closing() {
    let dir = burst_folder();
    let gate = GateThumbnails::new(1);
    let session = Arc::new(Session::new(
        base_env()
            .with_decode_threads(1)
            .with_thumbnails(gate.clone()),
    ));
    session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    gate.wait_parked();
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (_ctx, queue, rx) = neutral_queue(home.path());

    // Cancelled while it waits: stops within a moment, writes nothing.
    let id = queue
        .start(&session, &job(ApplyScope::This, out.path()), false)
        .expect("queued");
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(30)).expect("busy"),
        ExportEvent::Busy(true)
    );
    queue.cancel(id);
    assert!(
        queue.wait_idle(Duration::from_secs(10)),
        "the cancel is noticed"
    );
    assert!(written_files(out.path()).is_empty());
    assert_eq!(
        queue.summary(id).expect("summary").state,
        JobState::Cancelled
    );
    while rx.try_recv().is_ok() {}

    // The session closes while a job waits: development stops for good, the job exports its photo
    // with the estimates it was queued with.
    queue
        .start(&session, &job(ApplyScope::This, out.path()), false)
        .expect("queued");
    session.close();
    gate.release();
    assert!(queue.wait_idle(Duration::from_secs(60)), "the job goes on");
    assert_eq!(written_files(out.path()).len(), 1);
}

#[test]
fn the_estimate_takes_the_detail_of_each_photos_thumbnail_and_a_typical_picture_before_there_is_one()
 {
    use ym_core::export::estimated_output_size;
    use ym_core::pipeline::PhotoSource;
    use ym_export::{DetailStat, estimate_bytes, estimate_bytes_with_detail};

    let f = folder(2, 600, 400);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, _rx) = neutral_queue(home.path());
    let jpeg = job(ApplyScope::All, out.path());
    let format = jpeg.options.format_spec();
    let sizes: Vec<(u32, u32)> = f
        .ids
        .iter()
        .map(|id| {
            let v = f.session.view(*id).expect("view");
            estimated_output_size(&v.meta, &v.settings)
        })
        .collect();
    let sum = |details: &[Option<DetailStat>]| -> u64 {
        sizes
            .iter()
            .zip(details)
            .map(|(&(w, h), d)| estimate_bytes_with_detail(w, h, format, jpeg.options.size, *d))
            .sum()
    };

    // No thumbnail yet: every photo is a picture of typical detail, what `estimate_bytes` says.
    let typical: u64 = sizes
        .iter()
        .map(|&(w, h)| estimate_bytes(w, h, format, jpeg.options.size))
        .sum();
    assert_eq!(queue.estimate(&f.session, &jpeg).bytes, typical);

    // One thumbnail: that photo uses its statistic, the other stays typical.
    let thumbnail = |id: PhotoId| -> DetailStat {
        let src = PhotoSource::from_session(&f.session, id).expect("source");
        let png = ctx
            .renderer()
            .spawn_thumbnail(src)
            .wait()
            .expect("job")
            .expect("thumbnail");
        let mut reader = png::Decoder::new(std::io::Cursor::new(png))
            .read_info()
            .expect("a PNG");
        let mut rgb = vec![0u8; reader.output_buffer_size().expect("size")];
        let info = reader.next_frame(&mut rgb).expect("frame");
        DetailStat::measure_rgb8(info.width, info.height, &rgb[..info.buffer_size()])
            .expect("measurable")
    };
    let first = thumbnail(f.ids[0]);
    let est = queue.estimate(&f.session, &jpeg);
    assert_eq!(est.bytes, sum(&[Some(first), None]));
    assert_ne!(est.bytes, typical, "a flat patchwork is not typical detail");

    // Both: the sum over the photos, each with its own.
    let second = thumbnail(f.ids[1]);
    assert_eq!(
        queue.estimate(&f.session, &jpeg).bytes,
        sum(&[Some(first), Some(second)])
    );
    // A job over one photo uses that photo's statistic only.
    let this = ExportJob {
        scope: ApplyScope::This,
        ..jpeg.clone()
    };
    f.session.set_current(f.ids[1]).expect("current");
    let (w, h) = sizes[1];
    assert_eq!(
        queue.estimate(&f.session, &this).bytes,
        estimate_bytes_with_detail(w, h, format, jpeg.options.size, Some(second))
    );

    // The session closes: its photos' figures go, a photo of the next session starts typical.
    ctx.renderer().retire_through(*f.ids.last().expect("ids"));
    assert_eq!(ctx.renderer().thumbnail_detail(f.ids[0]), None);
    assert_eq!(queue.estimate(&f.session, &jpeg).bytes, typical);
}

#[test]
fn photos_removed_from_the_filmstrip_while_a_job_waits_are_still_exported_as_queued() {
    // Remove from filmstrip takes photos out of the session; an export that was queued before keeps the photos it was
    // started with (their snapshots) and writes the files it would have written.
    let f = folder(3, 300, 200);
    f.session
        .update_settings(f.ids[1], &SettingsPatch::exposure_ev(0.6))
        .expect("edit");
    let (home, reference, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("r"),
        tempfile::tempdir().expect("o"),
    );
    let (ctx, queue, rx) = neutral_queue(home.path());
    queue
        .start(&f.session, &job(ApplyScope::All, reference.path()), false)
        .expect("queued");
    until_idle(&rx);
    assert_eq!(names_in(reference.path()).len(), 3);

    // Hold the render worker: the job below is queued, its snapshots taken, and nothing has rendered yet.
    let (open, parked) = std::sync::mpsc::channel::<()>();
    let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
    let gate = ctx.renderer().scheduler().submit(
        ym_core::scheduler::Priority::Viewport,
        None,
        move |_| {
            started_tx.send(()).expect("started");
            let _ = parked.recv();
        },
    );
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("gate");
    queue
        .start(&f.session, &job(ApplyScope::All, out.path()), false)
        .expect("queued");

    // The person removes two of the photos, and edits the one that stays.
    f.session.remove(&[f.ids[1], f.ids[2]]);
    ctx.renderer().retire_photos(&[f.ids[1], f.ids[2]]);
    f.session
        .update_settings(f.ids[0], &SettingsPatch::exposure_ev(-1.0))
        .expect("edit after the queueing");
    assert_eq!(f.session.photo_ids(), vec![f.ids[0]]);

    open.send(()).expect("open");
    gate.wait().expect("gate");
    let events = until_idle(&rx);
    assert_eq!(files_done(&events).len(), 3);
    assert!(files_done(&events).iter().all(|d| d.outcome.is_ok()));
    // The same three files, byte for byte, as when nothing was removed or edited.
    let written = names_in(out.path());
    assert_eq!(written, names_in(reference.path()));
    for name in &written {
        assert_eq!(
            read(&out.path().join(name)),
            read(&reference.path().join(name)),
            "{name}"
        );
    }
    // The photos that had left stay out of the renderer's caches.
    assert_eq!(ctx.renderer().counters(f.ids[1]), Default::default());
    assert_eq!(ctx.renderer().counters(f.ids[2]), Default::default());
}
