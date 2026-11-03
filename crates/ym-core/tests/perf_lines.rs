//! The `ym::perf` lines of a viewport frame (`ym_core::perf`): off, a frame writes nothing; on, it writes one
//! `frame` line with the stages of the mask work, whether the frame was made on the CPU or the device.
mod common;

use std::sync::Arc;

use common::*;
use serde_json::json;
use ym_core::pipeline::{
    FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;

fn settings(ev: f64) -> DevelopSettings {
    let brush = json!({
        "kind": "brush", "mode": "add", "feather": 0.3, "refine": 0.5,
        "strokes": [{ "size": 0.1, "hardness": 0.5, "erase": false,
                      "points": [[0.2, 0.3], [0.5, 0.5], [0.8, 0.6]] }],
    });
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": ev },
          "masks": { "invert": false, "items": [brush] } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn frame(r: &Renderer, src: &PhotoSource, ev: f64) {
    let st = ViewportState::new(ViewportRect::cell(240.0, 160.0, 1.0));
    r.viewport_frame(
        &src.with_settings(settings(ev)),
        &st,
        &FrameOptions::default(),
        &CancelToken::new(),
    )
    .expect("frame");
}

#[test]
fn a_frame_writes_its_perf_line_only_when_perf_is_on() {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let src = Developed::new(&scene, "ILCE-7RM4").source();
    let r = Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let lines = || {
        captured_log()
            .lines()
            .filter(|l| l.contains("ym::perf") && l.contains("frame 240x160"))
            .count()
    };
    let before = lines();
    ym_core::perf::set_enabled(false);
    frame(&r, &src, 0.5);
    assert_eq!(lines(), before, "perf off: no line");
    ym_core::perf::set_enabled(true);
    frame(&r, &src, 0.7);
    ym_core::perf::set_enabled(false);
    let log = captured_log();
    let line = log
        .lines()
        .rfind(|l| l.contains("ym::perf") && l.contains("frame 240x160"))
        .expect("a frame line");
    assert_eq!(lines(), before + 1, "perf on: one line for the frame");
    for field in [
        "cache=miss",
        "rasters=yes",
        "mask_view=none",
        "plans=",
        "pass=",
        "prepare=",
        "cpu_raster=",
        "grade_plans=1",
        "color_passes=1",
    ] {
        assert!(line.contains(field), "{field} in {line}");
    }
    // The Brush row's planes were made on the CPU path, and took time.
    let cpu_raster: f64 = line
        .split("cpu_raster=")
        .nth(1)
        .and_then(|s| s.split('x').next())
        .and_then(|s| s.parse().ok())
        .expect("a number");
    assert!(cpu_raster > 0.0, "{line}");
}
