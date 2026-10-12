//! First end-to-end render of a synthetic photo.
mod common;

use common::*;
use ym_core::pipeline::{FrameOptions, Renderer, ViewportRect, ViewportState};
use ym_gpu::queue::CancelToken;

#[test]
fn a_synthetic_photo_renders_a_fit_frame() {
    let scene = Scene::new(
        1200,
        800,
        light("sony-ilce-7rm4", 5500.0),
        Layout::Patchwork,
        0.4,
    )
    .noise(2e-4, 1e-6);
    let d = Developed::new(&scene, "ILCE-7RM4");
    let src = d.source();
    let renderer = Renderer::neutral_only().expect("render thread");
    let state = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    let t = std::time::Instant::now();
    let frame = renderer
        .viewport_frame(&src, &state, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    eprintln!(
        "fit frame {}x{} in {:?}: {:?}",
        frame.width,
        frame.height,
        t.elapsed(),
        frame.report
    );
    assert_eq!((frame.width, frame.height), (600, 400));
    let m = mean_rgba(&frame, 0, 0, 600, 400);
    eprintln!("mean {m:?}");
    assert!(m.iter().all(|v| *v > 5.0 && *v < 250.0));
}
