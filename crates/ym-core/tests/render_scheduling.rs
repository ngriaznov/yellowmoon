//! Priorities, navigation and cancellation on the real render jobs: viewport frames before tile
//! passes before thumbnails before exports; navigation cancels the viewport and tile jobs of the
//! photo it leaves and nothing else; a cancelled render leaves the caches usable.
mod common;

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderError, RenderOptions,
    Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::scheduler::{CancelToken, JobError, Priority};
use ym_core::session::{OpenOptions, Session};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";

struct Two {
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    session: Session,
    a: PhotoSource,
    b: PhotoSource,
}

fn two() -> Two {
    let dir = tempfile::tempdir().expect("dir");
    for (i, level) in [0.2, 0.4].into_iter().enumerate() {
        Scene::new(
            1200,
            800,
            light("sony-ilce-7rm4", 5500.0),
            Layout::Patchwork,
            level,
        )
        .noise(2e-4, 1e-6)
        .write(dir.path(), &format!("DSC0000{}.ARW", i + 1), MODEL);
    }
    let session = Session::new(base_env());
    let out = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let src = |i: usize| {
        let view = session.view(out.added[i]).expect("view");
        PhotoSource::from_view(&view, session.raw_image(out.added[i]).expect("raw"))
    };
    let (a, b) = (src(0), src(1));
    Two { dir, session, a, b }
}

/// Holds the render worker until the returned sender is used.
fn block(r: &Renderer) -> (mpsc::Sender<()>, ym_core::scheduler::JobHandle<()>) {
    let (tx, rx) = mpsc::channel::<()>();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    let h = r.scheduler().submit(Priority::Viewport, None, move |_| {
        ready_tx.send(()).expect("ready");
        rx.recv().expect("gate");
    });
    ready_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the worker took the gate job");
    (tx, h)
}

fn cell() -> ViewportState {
    ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0))
}

#[test]
fn a_viewport_frame_overtakes_queued_thumbnails_and_exports() {
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let (gate, gate_job) = block(&r);
    let export = r.spawn_full(
        t.a.clone(),
        FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
    );
    let thumbs: Vec<_> = (0..3).map(|_| r.spawn_thumbnail(t.a.clone())).collect();
    let tiles = {
        let mut st = cell();
        st.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
        r.spawn_tiles(t.a.clone(), st)
    };
    let frame = r.spawn_viewport_frame(t.a.clone(), cell(), FrameOptions::default());
    // The export sits on its own worker, waiting for the foreground to go idle (so it is queued or
    // already picked up); the other five are behind the gate.
    assert!(r.scheduler().pending() >= 5, "{}", r.scheduler().pending());
    gate.send(()).expect("open");
    gate_job.wait().expect("gate");
    // The viewport frame is done first: nothing queued behind it has finished.
    let f = frame.wait().expect("job").expect("frame");
    assert_eq!((f.width, f.height), (300, 200));
    assert!(
        tiles.try_wait().is_none(),
        "tiles run after the viewport frame"
    );
    assert!(thumbs[0].try_wait().is_none() && export.try_wait().is_none());
    // Tiles, then thumbnails, then the export, in that order.
    assert!(tiles.wait().expect("job").is_ok());
    assert!(export.try_wait().is_none(), "the export is last");
    for t in thumbs {
        assert!(t.wait().expect("job").is_ok());
    }
    let full = export.wait().expect("job").expect("export");
    assert_eq!((full.width, full.height), (1200, 800));
}

#[test]
fn navigating_away_cancels_viewport_and_tile_jobs_but_not_thumbnails_or_exports() {
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    r.navigate(t.a.id);
    let (gate, gate_job) = block(&r);
    let mut st = cell();
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
    let frame_a = r.spawn_viewport_frame(t.a.clone(), cell(), FrameOptions::default());
    let tiles_a = r.spawn_tiles(t.a.clone(), st);
    let thumb_a = r.spawn_thumbnail(t.a.clone());
    let export_a = r.spawn_full(
        t.a.clone(),
        FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
    );
    let frame_b = r.spawn_viewport_frame(t.b.clone(), cell(), FrameOptions::default());
    assert_eq!(
        r.navigate(t.b.id),
        2,
        "the viewport and tile jobs of photo A"
    );
    gate.send(()).expect("open");
    gate_job.wait().expect("gate");
    assert!(matches!(frame_a.wait(), Err(JobError::Cancelled)));
    assert!(matches!(tiles_a.wait(), Err(JobError::Cancelled)));
    assert!(
        frame_b.wait().expect("job").is_ok(),
        "the new photo's frame runs"
    );
    assert!(
        thumb_a.wait().expect("job").is_ok(),
        "thumbnails are not cancelled by navigation"
    );
    assert!(
        export_a.wait().expect("job").is_ok(),
        "exports are never cancelled by navigation"
    );
}

#[test]
fn navigating_away_supersedes_the_look_tiles_and_histogram_of_the_photo_left_and_lets_go_of_its_mosaic()
 {
    // The Look tab asks for a tile per look and the Light tab for a histogram, for every photo the
    // person steps to. Each queued job holds the photo's decoded mosaic; run after run for photos
    // that have been left, they used to pile up (about 120 ms a tile at 24 MP, 122 MB a mosaic at
    // 61 MP) and the current photo's own tiles waited behind them.
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    r.navigate(t.a.id);
    let (gate, gate_job) = block(&r);
    let neutral = LookId::builtin("neutral");
    let holders_before = Arc::strong_count(&t.a.raw);

    let tiles_a: Vec<_> = (0..10)
        .map(|_| r.spawn_look_thumbnail(t.a.clone(), neutral.clone()))
        .collect();
    let histogram_a = r.spawn_histogram(t.a.clone());
    let thumbnail_a = r.spawn_thumbnail(t.a.clone());
    let tile_b = r.spawn_look_thumbnail(t.b.clone(), neutral.clone());
    assert_eq!(
        Arc::strong_count(&t.a.raw),
        holders_before + 12,
        "each queued job holds the mosaic"
    );

    // The person steps to photo B: what was asked of photo A for the panels is no longer wanted.
    assert_eq!(r.navigate(t.b.id), 11, "ten tiles and the histogram");
    assert_eq!(
        Arc::strong_count(&t.a.raw),
        holders_before + 1,
        "only the filmstrip thumbnail still holds it; the cancelled jobs let go at once"
    );

    gate.send(()).expect("open");
    gate_job.wait().expect("gate");
    for tile in tiles_a {
        assert!(
            matches!(tile.wait(), Err(JobError::Cancelled)),
            "a superseded tile is a cancelled job, like a viewport frame of the photo left"
        );
    }
    assert!(matches!(histogram_a.wait(), Err(JobError::Cancelled)));
    assert!(
        tile_b.wait().expect("job").is_ok(),
        "the new photo's tile runs"
    );
    assert!(
        thumbnail_a.wait().expect("job").is_ok(),
        "the filmstrip thumbnail is not a panel request and is kept"
    );
    assert_eq!(Arc::strong_count(&t.a.raw), holders_before);
}

#[test]
fn a_tile_that_is_already_running_when_the_person_moves_on_is_told_to_stop() {
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    r.navigate(t.a.id);
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let running = r.scheduler().submit_superseded(
        Priority::Thumbnail,
        t.a.id,
        move |token| {
            started_tx.send(()).expect("started");
            stop_rx.recv().expect("stop");
            token.is_cancelled()
        },
        || false,
    );
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("running");
    assert_eq!(r.navigate(t.b.id), 1);
    stop_tx.send(()).expect("stop");
    assert_eq!(running.wait(), Ok(true), "the job saw its token set");
    // Moving on to the same photo again, or to a third, finds nothing left to cancel.
    assert_eq!(r.navigate(t.b.id), 0);
}

#[test]
fn a_panel_request_for_the_photo_being_moved_to_is_never_cancelled_whatever_the_order_of_arrival() {
    // The webview asks for the new photo's tiles and tells the core about the move in two commands
    // that may be handled in either order.
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    r.navigate(t.a.id);
    let (gate, gate_job) = block(&r);
    let tile_b = r.spawn_look_thumbnail(t.b.clone(), LookId::builtin("neutral"));
    let histogram_b = r.spawn_histogram(t.b.clone());
    assert_eq!(r.navigate(t.b.id), 0, "nothing of A is waiting");
    assert_eq!(
        r.navigate(t.b.id),
        0,
        "and the same photo again changes nothing"
    );
    gate.send(()).expect("open");
    gate_job.wait().expect("gate");
    assert!(tile_b.wait().expect("job").is_ok());
    assert!(histogram_b.wait().expect("job").is_ok());
}

#[test]
fn a_render_stopped_in_the_middle_leaves_the_caches_usable() {
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let mut st = cell();
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
    // Cancel from another thread shortly after the job starts. Whether it lands before the end is
    // a race; both outcomes are valid, and neither may poison the next request.
    let token = CancelToken::new();
    let canceller = {
        let token = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            token.cancel();
        })
    };
    let first = r.viewport_frame(&t.a, &st, &FrameOptions::default(), &token);
    canceller.join().expect("thread");
    match &first {
        Ok(_) | Err(RenderError::Cancelled) => {}
        Err(e) => panic!("{e}"),
    }
    let again = r
        .viewport_frame(&t.a, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("the next request works");
    assert_eq!((again.width, again.height), (300, 200));
    let reference = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default())
        .expect("render thread")
        .viewport_frame(&t.a, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("reference");
    assert_eq!(
        again.rgba, reference.rgba,
        "no half-made cache entry reached the picture"
    );
}

#[test]
fn photos_that_share_an_id_across_sessions_never_share_pixels() {
    // Two sessions both call their first photo id 1. One renderer serves both: the graph notices
    // that the decoded image behind the id changed and starts over.
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let cancel = CancelToken::new();
    let st = cell();
    let mut means = Vec::new();
    for level in [0.1, 0.5] {
        let d = Developed::new(
            &Scene::new(
                600,
                400,
                light("sony-ilce-7rm4", 5500.0),
                Layout::Uniform,
                level,
            ),
            MODEL,
        );
        // Auto exposure would put both fields at mid-grey: fix the exposure.
        let mut settings = d.source().settings.clone();
        settings.exposure =
            ym_core::settings::AutoOr::Manual(ym_core::settings::ExposureValue { ev: 0.0 });
        let src = d.source().with_settings(settings);
        assert_eq!(src.id.0, 1);
        let f = r
            .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .expect("frame");
        means.push(mean_rgba(&f, 100, 70, 100, 60)[1]);
    }
    assert!(means[1] > means[0] + 20.0, "{means:?}");
}

#[test]
fn threads_that_render_the_same_photo_together_get_the_same_picture() {
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let frames: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    r.viewport_frame(&t.a, &cell(), &FrameOptions::default(), &CancelToken::new())
                        .expect("frame")
                        .rgba
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    assert!(frames.windows(2).all(|w| w[0] == w[1]));
    let _ = &t.session;
}

#[test]
fn removed_photos_get_no_more_render_work_the_others_and_the_exports_go_on() {
    // Taking a photo out of the filmstrip cancels its queued and running renders, forgets its caches and refuses the
    // requests that were on their way; an export, which holds its own snapshot, finishes the photo and leaves no cache
    // behind; the photo put back by Undo is served again.
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    for src in [&t.a, &t.b] {
        r.spawn_viewport_frame(src.clone(), cell(), FrameOptions::default())
            .wait()
            .expect("job")
            .expect("frame");
    }
    let both = r.cached_bytes();
    assert!(both > 0);
    assert_ne!(r.counters(t.a.id), Default::default());

    let (gate, gate_job) = block(&r);
    let thumbnail_a = r.spawn_thumbnail(t.a.clone());
    let frame_a = r.spawn_viewport_frame(t.a.clone(), cell(), FrameOptions::default());
    let tile_a =
        r.spawn_look_thumbnail(t.a.clone(), LookId::parse("builtin:neutral").expect("look"));
    let thumbnail_b = r.spawn_thumbnail(t.b.clone());
    let export_a = r.spawn_full(
        t.a.clone(),
        FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
    );
    assert!(!r.is_retired(t.a.id));
    r.retire_photos(&[t.a.id]);
    assert!(r.is_retired(t.a.id) && !r.is_retired(t.b.id));
    assert!(
        r.cached_bytes() < both,
        "the caches of the photo were forgotten"
    );
    assert_eq!(r.counters(t.a.id), Default::default());
    gate.send(()).expect("open the gate");
    gate_job.wait().expect("gate job");

    // The photo's queued renders never ran; the other photo's did; the export finished.
    assert_eq!(thumbnail_a.wait().map(|_| ()), Err(JobError::Cancelled));
    assert_eq!(frame_a.wait().map(|_| ()), Err(JobError::Cancelled));
    assert_eq!(tile_a.wait().map(|_| ()), Err(JobError::Cancelled));
    assert!(thumbnail_b.wait().expect("job").is_ok());
    let full = export_a
        .wait()
        .expect("job")
        .expect("the export renders the photo it was started with");
    assert_eq!((full.width, full.height), (1200, 800));
    // The export left nothing of the removed photo behind.
    assert_eq!(
        r.counters(t.a.id),
        Default::default(),
        "no graph for the removed photo"
    );

    // A request that was on its way is refused as cancelled and builds nothing.
    let never = CancelToken::new();
    assert!(matches!(
        r.viewport_frame(&t.a, &cell(), &FrameOptions::default(), &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.thumbnail_png(&t.a, &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.histogram(&t.a, &never),
        Err(RenderError::Cancelled)
    ));
    assert_eq!(r.counters(t.a.id), Default::default());
    // The other photo renders as before.
    assert!(r.thumbnail_png(&t.b, &never).is_ok());

    // Undo: the photo is served again.
    r.unretire_photos(&[t.a.id]);
    assert!(!r.is_retired(t.a.id));
    assert!(r.thumbnail_png(&t.a, &never).is_ok());
    // A closed session retires everything, the removed photos' marks included.
    r.retire_photos(&[t.b.id]);
    r.retire_through(t.session.last_photo_id().expect("photos were opened"));
    assert!(r.is_retired(t.a.id) && r.is_retired(t.b.id));
}

#[test]
fn a_retired_session_gets_no_more_viewport_work_but_its_exports_go_on() {
    // Closing the session cancels the queued foreground jobs, forgets the caches and retires the
    // ids: a request that was on its way (a command that read its photo before the close and
    // queued its job after the cancel) is refused instead of making caches for a photo that is
    // gone, while an export, which holds its own snapshot, still renders.
    let t = two();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let frame = r
        .spawn_viewport_frame(t.a.clone(), cell(), FrameOptions::default())
        .wait()
        .expect("job");
    assert!(frame.is_ok());
    assert!(r.cached_bytes() > 0);

    let last = t.session.last_photo_id().expect("photos were opened");
    assert_eq!(last, t.b.id.max(t.a.id));
    assert!(!r.is_retired(t.a.id));
    r.retire_through(last);
    assert!(r.is_retired(t.a.id) && r.is_retired(t.b.id));
    assert_eq!(r.cached_bytes(), 0, "the caches were forgotten");

    // Every foreground entry point is refused as cancelled, and builds nothing.
    let never = CancelToken::new();
    assert!(matches!(
        r.viewport_frame(&t.a, &cell(), &FrameOptions::default(), &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.thumbnail_png(&t.a, &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.histogram(&t.a, &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.final_size(&t.b, &never),
        Err(RenderError::Cancelled)
    ));
    assert!(matches!(
        r.spawn_thumbnail(t.b.clone()).wait().expect("job"),
        Err(RenderError::Cancelled)
    ));
    assert_eq!(r.cached_bytes(), 0, "refused requests cache nothing");

    // An export goes on.
    let full = r
        .spawn_full(
            t.a.clone(),
            FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
        )
        .wait()
        .expect("job")
        .expect("the export renders after the close");
    assert_eq!((full.width, full.height), (1200, 800));

    // Photos of the next session (ids above the closed one's) render normally.
    let next = {
        let mut src = t.a.clone();
        src.id = ym_core::session::PhotoId(last.0 + 1);
        src
    };
    assert!(!r.is_retired(next.id));
    assert!(r.thumbnail_png(&next, &never).is_ok());
}
