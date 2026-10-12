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
