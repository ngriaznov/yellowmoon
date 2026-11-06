//! A mask plane that arrives after the settings is a new key for the whole-image level (G-13, `pipeline::frame::level`).
//!
//! A Subject row selects nothing until the Vision request has made its mask; the grade plan's key names the plane
//! (`vision:<kind>:<plane key>`, `none` before it is there), so the level made before the plane arrived is freed at the next
//! request and never served after it: the settings are the same, the picture is not.
//!
//! On lavapipe in `gpu-lavapipe` (`YM_REQUIRE_GPU=1`); skipped, loudly, without an adapter.
mod common;

use std::io::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    FrameOptions, FrameOutput, LevelStatus, NeutralOnly, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_core::vision::{FakeVision, VisionMasks};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in view_level_planes: {e}\n"
            );
            None
        }
    }
}

/// Input, Main, node 2 with a Subject row and a stop of exposure, Output.
fn subject_graph() -> DevelopSettings {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": [ { "kind": "subject", "mode": "add" } ] } },
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
fn a_subject_mask_that_arrives_after_the_level_frees_it() {
    let Some(gpu) = gpu() else { return };
    let scene =
        Scene::new(1200, 800, light(PROFILE, 5500.0), Layout::Uniform, 0.08).noise(2e-4, 1e-6);
    let dir = tempfile::tempdir().expect("temp dir");
    scene.write(dir.path(), "DSC00001.ARW", MODEL);
    let d = Developed::open(dir);
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            data_root: Some(data_root()),
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL
                .difference(GpuStages::RAW_CHAIN)
                .difference(GpuStages::RAW_FRONT),
            levels: true,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let src = d.source().with_settings(subject_graph());
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let request = |no_cache: bool| {
        r.viewport_frame_to(
            &src,
            &st,
            &FrameOptions {
                no_cache,
                ..FrameOptions::default()
            },
            FrameOutput::Device,
            &CancelToken::new(),
        )
        .expect("frame")
    };
    // The level is made while the row selects nothing (no plane yet).
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    assert_eq!(request(false).report.runs.level_hits, 1);
    let before = r.levels_held(src.id);
    // The plane arrives: the settings are the same, the key is not.
    r.vision_prepare(&src);
    let end = Instant::now() + Duration::from_secs(60);
    while r.vision_worker_alive(src.id) {
        assert!(Instant::now() < end, "the Vision worker never ended");
        std::thread::sleep(Duration::from_millis(20));
    }
    let after = request(false);
    assert_eq!(
        after.report.runs.level_hits, 0,
        "the level made before the plane is not served after it"
    );
    assert_eq!(r.levels_held(src.id), (0, 0), "{before:?}: freed");
    // And a level made now is the picture with the mask.
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    let served = request(false);
    assert_eq!(served.report.runs.level_hits, 1);
    assert_eq!(
        served.host_rgba().expect("read").into_owned(),
        request(true).host_rgba().expect("read").into_owned()
    );
}
