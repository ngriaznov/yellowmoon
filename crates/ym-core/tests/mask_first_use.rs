//! The first frame of every mask view builds no pipeline: the warm-up the application starts when it opens the
//! device (`GpuSlot::warm_up_for_viewer`, `docs/17-gpu.md` section 7) compiled them all. On the owner's Mac the
//! first frame after the eye of a mask row was switched on took 245 ms, 210 of them recording the tail, where
//! the overlay's kernel was compiled on first use. A frame's builds are `CounterSnapshot::pipelines`, counted
//! by the kernel cache, so nothing here depends on timing.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_core::app::gpu::{GpuPolicy, open_gpu, warm_up_each};
use ym_core::scheduler::MemoryBudget;
use ym_gpu::kernels::KernelId;

use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_core::vision::{FakeVision, VisionMasks, VisionProvider};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

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
                "\n* GPU TESTS SKIPPED in mask_device_route: {e}\n"
            );
            None
        }
    }
}

fn renderer(gpu: &Arc<Gpu>, vision: Option<Arc<dyn VisionProvider>>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::ALL,
            vision: vision.map(VisionMasks::new),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn settings(nodes: Vec<Value>, links: Vec<Value>) -> DevelopSettings {
    let mut items = vec![json!({ "id": 1 })];
    items.extend(nodes);
    let doc = json!({ "nodes": { "items": items, "links": links } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn picture(from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } })
}

/// Input, Main, node 2, Output.
fn line() -> Vec<Value> {
    vec![
        picture(json!("input"), json!(1)),
        picture(json!(1), json!(2)),
        picture(json!(2), json!("output")),
    ]
}

fn node(id: u32, x: f64, rows: Vec<Value>, extra: Value) -> Value {
    let mut n = json!({
        "id": id, "kind": "corrector", "pos": [x, 0.0],
        "trim": { "exposure_ev": 1.0 },
        "masks": { "invert": false, "items": rows },
    });
    if let (Some(a), Some(b)) = (n.as_object_mut(), extra.as_object()) {
        a.extend(b.clone());
    }
    n
}

fn row(kind: &str, extra: Value) -> Value {
    let mut r = json!({ "kind": kind, "mode": "add" });
    if let (Some(a), Some(b)) = (r.as_object_mut(), extra.as_object()) {
        a.extend(b.clone());
    }
    r
}

fn linear() -> Value {
    row(
        "linear",
        json!({ "center": [0.5, 0.5], "angle_deg": 20.0, "feather": 0.3 }),
    )
}
fn brush_painted() -> Value {
    row(
        "brush",
        json!({ "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.35, 0.45], [0.6, 0.55]] }],
                "feather": 0.3, "refine": 0.5 }),
    )
}
fn background() -> Value {
    row("background", json!({}))
}
fn object_empty() -> Value {
    row("object", json!({ "points": [] }))
}

/// The photo with node 2 behind these rows, in a line.
fn one_node(rows: Vec<Value>) -> DevelopSettings {
    settings(vec![node(2, 1.0, rows, json!({}))], line())
}

fn view(zoom_100: bool, mask: Option<(MaskStyle, Option<u32>)>) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st.mask_view = mask.map(|(style, row)| MaskView {
        node: 2,
        style,
        row,
        depth: false,
        face: None,
    });
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(
        src,
        st,
        &FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
        &CancelToken::new(),
    )
    .expect("frame")
}

/// The views the owner can switch on, and a photo with every kind of row the viewer draws.
/// A named state of the mask view: nothing, or a style and a row.
type State = (&'static str, Option<(MaskStyle, Option<u32>)>);

fn shown() -> Vec<State> {
    vec![
        ("no mask view", None),
        ("row, overlay", Some((MaskStyle::Overlay, Some(0)))),
        ("row, mask only", Some((MaskStyle::MaskOnly, Some(0)))),
        ("brush row, overlay", Some((MaskStyle::Overlay, Some(1)))),
        ("object row, overlay", Some((MaskStyle::Overlay, Some(2)))),
        (
            "background row, overlay",
            Some((MaskStyle::Overlay, Some(3))),
        ),
        ("node, overlay", Some((MaskStyle::Overlay, None))),
        ("node, mask only", Some((MaskStyle::MaskOnly, None))),
    ]
}

fn all_rows() -> DevelopSettings {
    one_node(vec![
        linear(),
        brush_painted(),
        object_empty(),
        background(),
    ])
}

/// Renders the photo in each state at Fit and at 100 %, and says which frames built a pipeline.
fn builds(gpu: &Arc<Gpu>, zoom_100: &[bool]) -> Vec<String> {
    let r = renderer(gpu, Some(Arc::new(FakeVision::new())));
    let d = photo();
    let src = d.source().with_settings(all_rows());
    let mut built = Vec::new();
    for &z in zoom_100 {
        for (name, mask) in shown() {
            let before = gpu.counters();
            let f = frame(&r, &src, &view(z, mask));
            let now = gpu.counters().since(before);
            assert_eq!(f.report.device, ComputeDevice::Gpu, "{name}: on the device");
            if now.pipelines > 0 {
                built.push(format!("{name} (100 % {z}): {} pipelines", now.pipelines));
            }
        }
    }
    built
}

fn warmed(gpu: &Arc<Gpu>) {
    for handle in warm_up_each(gpu) {
        handle.wait().expect("job").expect("a kernel compiled");
    }
}

#[test]
fn after_the_warm_up_no_frame_of_any_mask_view_builds_a_pipeline() {
    let Some(gpu) = gpu() else { return };
    warmed(&gpu);
    let built = builds(&gpu, &[false, true]);
    assert!(built.is_empty(), "frames that compiled a shader: {built:?}");
}

#[test]
fn without_it_the_first_frames_compile_shaders_which_is_what_the_owner_waited_for() {
    // The control of the test above: the same frames on a device nobody warmed build pipelines, the eye's
    // first frame among them (its tint kernel).
    let Some(gpu) = gpu() else { return };
    let built = builds(&gpu, &[false]);
    assert!(
        built.iter().any(|b| b.starts_with("row, overlay")),
        "the first eye frame builds something when cold: {built:?}"
    );
}

#[test]
fn the_slots_warm_up_is_one_job_a_kernel_and_compiles_them_all() {
    let slot = open_gpu(GpuPolicy::On, MemoryBudget::default());
    let Some(gpu) = slot.gpu().cloned() else {
        return;
    };
    if !slot.is_using() {
        return;
    }
    let jobs = slot.warm_up_for_viewer();
    assert_eq!(jobs.len(), KernelId::ALL.len());
    for job in jobs {
        assert_eq!(job.wait().expect("job").expect("compiled"), 1);
    }
    let before = gpu.counters();
    let again: usize = warm_up_each(&gpu)
        .into_iter()
        .map(|j| j.wait().expect("job").expect("compiled"))
        .sum();
    assert_eq!(again, KernelId::ALL.len());
    assert_eq!(gpu.counters().since(before).pipelines, 0, "already built");
    assert!(gpu.counters().pipelines >= KernelId::ALL.len() as u64);
}
