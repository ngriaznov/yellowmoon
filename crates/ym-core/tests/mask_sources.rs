//! The source planes of raster rows stay cached when a plan has more than eight of them (`docs/23-nodes.md`
//! section 15, speed): the CPU's Brush cache and the device's copies hold one plane for every live row, so that
//! the second frame of a plan with ten Brush rows draws no stroke and uploads no source.
//!
//! This file has one test and so one process: the Brush cache is process-wide, and a test that draws other brushes
//! at the same time would trim it.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in mask_sources: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// Input, Main, node 2 with these mask rows, Output.
fn graph(rows: Vec<Value>) -> DevelopSettings {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": rows } },
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

#[test]
fn ten_brush_rows_draw_and_upload_their_sources_once() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            gpu_stages: GpuStages::ALL,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    // The planes of the rows are kept on the device too, which would make the second frame skip the sources
    // altogether: this test is about the caches of the sources, so they are off.
    gpu.set_planes_kept(false);
    let rows: Vec<Value> = (0..10)
        .map(|i| {
            let y = 0.1 + 0.08 * f64::from(i);
            json!({
                "kind": "brush", "mode": "add",
                "strokes": [{ "size": 0.05, "hardness": 0.5, "erase": false, "points": [[0.2, y], [0.8, y]] }],
            })
        })
        .collect();
    let s = src.with_settings(graph(rows));
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let frame = || {
        let before = gpu.counters();
        let f = r
            .viewport_frame(
                &s,
                &st,
                &FrameOptions {
                    no_cache: true,
                    ..FrameOptions::default()
                },
                &CancelToken::new(),
            )
            .expect("frame");
        assert_eq!(f.report.device, ComputeDevice::Gpu);
        gpu.counters().since(before)
    };
    let first = frame();
    assert_eq!(first.plane_builds, 10);
    // Ten sources of 2048 x 1365 u16 codes (5.6 MB each) went up.
    assert!(
        first.uploaded_bytes > 50 << 20,
        "{} bytes uploaded",
        first.uploaded_bytes
    );
    assert!(ym_color::mask_raster::brush_cache_len() >= 10);
    let second = frame();
    assert_eq!(second.plane_builds, 10);
    // Only the passes' small uniforms went up this time.
    assert!(
        second.uploaded_bytes < 1 << 20,
        "{} bytes uploaded by the second frame",
        second.uploaded_bytes
    );
    assert!(ym_color::mask_raster::brush_cache_len() >= 10);
}
