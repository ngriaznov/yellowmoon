//! No mask state ends the device route of a photo's views. On the owner's Mac every masked viewer frame of a 50 MP
//! photo ran on the CPU after one device error (`Job::view_gpu_failed` is sticky per photo), so each state a mask
//! row can be in is checked here on the device: the frame is the device's, carries no "graphics card" note, and the
//! photo's views are still on the device afterwards (`Renderer::raw_on_cpu` says CPU once the route has ended).
//! The second half checks the fallback rule itself (`docs/17-gpu.md` section 9): a refused check of ours
//! (`invalid kernel input`) costs one frame and leaves the route open.
mod common;

use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex};

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, MaskKind, MaskUnavailable, mask_kinds};
use ym_core::vision::{
    FakeVision, VisionError, VisionKind, VisionMask, VisionMasks, VisionPicture, VisionProvider,
    VisionStatus, VisionSupport,
};
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

fn key(from: u32, to: u32) -> Value {
    json!({ "from": { "node": from, "port": "key" }, "to": { "node": to, "port": "key" } })
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
fn radial() -> Value {
    row(
        "radial",
        json!({ "center": [0.5, 0.5], "radii": [0.3, 0.2], "angle_deg": 0.0, "feather": 0.3 }),
    )
}
fn colour_range() -> Value {
    row(
        "color_range",
        json!({ "center": [0.05, 0.0], "width": 0.2, "softness": 0.5 }),
    )
}
fn brightness_range() -> Value {
    row(
        "brightness_range",
        json!({ "low": 0.2, "high": 0.8, "softness": 0.2 }),
    )
}
fn brush_painted() -> Value {
    row(
        "brush",
        json!({ "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.35, 0.45], [0.6, 0.55]] }],
                "feather": 0.3, "refine": 0.5 }),
    )
}
fn brush_empty() -> Value {
    row("brush", json!({ "strokes": [] }))
}
fn background() -> Value {
    row("background", json!({}))
}
fn object_empty() -> Value {
    row("object", json!({ "points": [] }))
}
fn subject() -> Value {
    row("subject", json!({}))
}
fn people() -> Value {
    row("people", json!({}))
}
fn off(mut r: Value) -> Value {
    r["enabled"] = json!(false);
    r
}

/// The photo with node 2 behind these rows, in a line.
fn one_node(rows: Vec<Value>) -> DevelopSettings {
    settings(vec![node(2, 1.0, rows, json!({}))], line())
}

/// What the mask view shows: nothing, or a style and a row.
type Shown = Option<(MaskStyle, Option<u32>)>;

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

/// Renders the photo in every state of the list, each at Fit and at 100 % with a pan between two frames, and says
/// what went wrong: a frame the CPU made, a "graphics card" note, or an ended route.
fn route_problems(r: &Renderer, src: &PhotoSource, states: &[(&str, Shown)]) -> Vec<String> {
    let mut problems = Vec::new();
    for (what, mask) in states {
        for zoom_100 in [false, true] {
            let mut st = view(zoom_100, *mask);
            for pan in 0..2 {
                if pan == 1 {
                    st.centre = [0.45, 0.52];
                }
                let f = frame(r, src, &st);
                let at = format!("{what}, 100 % {zoom_100}, pan {pan}");
                if f.report.device != ComputeDevice::Gpu {
                    problems.push(format!("{at}: the frame is {:?}", f.report.device));
                }
                for n in f
                    .report
                    .notes
                    .iter()
                    .filter(|n| n.contains("graphics card"))
                {
                    problems.push(format!("{at}: note {n}"));
                }
                if r.raw_on_cpu(src.id) {
                    problems.push(format!("{at}: the photo's views are on the CPU now"));
                    return problems;
                }
            }
        }
    }
    problems
}

/// Every view the viewer can ask for with the mask row(s) in node 2.
fn all_states() -> Vec<(&'static str, Shown)> {
    vec![
        ("photo", None),
        ("overlay", Some((MaskStyle::Overlay, None))),
        ("mask only", Some((MaskStyle::MaskOnly, None))),
        ("overlay, eye on row 0", Some((MaskStyle::Overlay, Some(0)))),
        (
            "mask only, eye on row 0",
            Some((MaskStyle::MaskOnly, Some(0))),
        ),
    ]
}

fn check(r: &Renderer, src: &PhotoSource, states: &[(&str, Shown)]) {
    let problems = route_problems(r, src, states);
    assert!(problems.is_empty(), "{problems:#?}");
}

// ---- providers -------------------------------------------------------------------------------------------

/// A Mac whose Vision request is held shut until `open`: the rows have no plane meanwhile.
#[derive(Debug)]
struct Gate {
    inner: FakeVision,
    open: Mutex<bool>,
    cv: Condvar,
}

impl Gate {
    fn shut() -> Arc<Self> {
        Arc::new(Self {
            inner: FakeVision::new(),
            open: Mutex::new(false),
            cv: Condvar::new(),
        })
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

impl VisionProvider for Gate {
    fn probe(&self) -> VisionSupport {
        self.inner.probe()
    }

    fn mask(&self, kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.cv.wait(open).unwrap();
        }
        drop(open);
        self.inner.mask(kind, picture)
    }
}

fn unsupported() -> Arc<dyn VisionProvider> {
    let no = VisionStatus::Unsupported(MaskUnavailable::NotSupported);
    Arc::new(FakeVision::new().with_support(VisionSupport {
        subject: no,
        people: no,
    }))
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !done() {
        assert!(
            std::time::Instant::now() < end,
            "timed out waiting for {what}"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

// ---- the sweep ---------------------------------------------------------------------------------------------

#[test]
fn every_mask_kind_is_made_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let r = renderer(&gpu, Some(Arc::new(FakeVision::new())));
    // The kinds this computer lists (the others need a model or a Mac), and the rest by hand below.
    let listed = mask_kinds();
    for (kind, rows) in [
        (MaskKind::Linear, vec![linear()]),
        (MaskKind::Radial, vec![radial()]),
        (MaskKind::ColorRange, vec![colour_range()]),
        (MaskKind::BrightnessRange, vec![brightness_range()]),
        (MaskKind::Brush, vec![brush_painted()]),
        (MaskKind::Background, vec![background()]),
        (MaskKind::Object, vec![object_empty()]),
        (MaskKind::Subject, vec![subject()]),
        (MaskKind::People, vec![people()]),
    ] {
        let s = src.with_settings(one_node(rows));
        eprintln!("kind {kind:?} (listed {})", listed.contains(&kind));
        check(&r, &s, &all_states());
    }
}

#[test]
fn rows_with_no_plane_yet_are_skipped_not_refused() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let r = renderer(&gpu, Some(Arc::new(FakeVision::new())));
    for (name, rows) in [
        ("brush, no stroke", vec![brush_empty()]),
        ("object, no point", vec![object_empty()]),
        ("background, before its map", vec![background()]),
        (
            "all of them",
            vec![brush_empty(), object_empty(), background()],
        ),
        (
            "empty ones beside a painted brush",
            vec![brush_empty(), brush_painted(), object_empty()],
        ),
        (
            "empty ones beside a linear",
            vec![object_empty(), linear(), background()],
        ),
    ] {
        eprintln!("{name}");
        let s = src.with_settings(one_node(rows));
        check(&r, &s, &all_states());
    }
}

#[test]
fn subject_and_people_before_the_worker_answers_and_when_unsupported() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let s = src.with_settings(one_node(vec![subject(), people()]));
    // Before the worker answers.
    let gate = Gate::shut();
    let r = renderer(&gpu, Some(gate.clone()));
    check(&r, &s, &all_states());
    // After it has answered.
    gate.open();
    r.vision_prepare(&s);
    wait_until("the Vision worker", || !r.vision_worker_alive(s.id));
    check(&r, &s, &all_states());
    // A Mac that cannot make them, and no Mac at all.
    check(&renderer(&gpu, Some(unsupported())), &s, &all_states());
    check(&renderer(&gpu, None), &s, &all_states());
}

#[test]
fn switches_bypass_and_links() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let r = renderer(&gpu, Some(Arc::new(FakeVision::new())));
    let planes = || vec![brush_painted(), background(), object_empty(), subject()];
    // A raster row whose switch is off (with and without others).
    let off_rows = settings(
        vec![node(
            2,
            1.0,
            vec![off(brush_painted()), off(background()), off(object_empty())],
            json!({}),
        )],
        line(),
    );
    // A node with Bypass on.
    let bypassed = settings(
        vec![node(2, 1.0, planes(), json!({ "bypass": true }))],
        line(),
    );
    // A dead node: it reads Main and goes nowhere.
    let mut links = line();
    links.push(picture(json!(1), json!(3)));
    let dead = settings(
        vec![
            node(2, 1.0, vec![linear()], json!({})),
            node(3, 1.0, planes(), json!({})),
        ],
        links,
    );
    // A node whose input link is cut.
    let cut = settings(
        vec![node(2, 1.0, planes(), json!({}))],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(2), json!("output")),
        ],
    );
    // A key link from a masked node.
    let keyed = settings(
        vec![
            node(2, 1.0, planes(), json!({})),
            node(3, 2.0, vec![], json!({})),
        ],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!(3)),
            picture(json!(3), json!("output")),
            key(2, 3),
        ],
    );
    for (name, s) in [
        ("switches off", off_rows),
        ("bypassed", bypassed),
        ("dead node", dead),
        ("cut input", cut),
        ("key link", keyed),
    ] {
        eprintln!("{name}");
        check(&r, &src.with_settings(s), &all_states());
    }
}

#[test]
fn two_chains_where_one_has_planes_and_one_has_none() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let r = renderer(&gpu, Some(Arc::new(FakeVision::new())));
    // The photo's chain has a painted brush in node 2; the mask view shows node 3, which has only empty rows (and the
    // other way round).
    let both = |a: Vec<Value>, b: Vec<Value>| {
        settings(
            vec![node(2, 1.0, a, json!({})), node(3, 2.0, b, json!({}))],
            vec![
                picture(json!("input"), json!(1)),
                picture(json!(1), json!(2)),
                picture(json!(2), json!(3)),
                picture(json!(3), json!("output")),
            ],
        )
    };
    for (name, s) in [
        (
            "planes then none",
            both(vec![brush_painted()], vec![object_empty()]),
        ),
        (
            "none then planes",
            both(vec![object_empty()], vec![brush_painted()]),
        ),
    ] {
        eprintln!("{name}");
        check(&r, &src.with_settings(s), &all_states());
    }
}

#[test]
fn an_object_row_with_a_point_but_no_model_and_a_retina_cell() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let r = renderer(&gpu, Some(Arc::new(FakeVision::new())));
    // Points, and nobody to turn them into a plane (no model is loaded in this renderer).
    let pointed = row(
        "object",
        json!({ "points": [[0.4, 0.5, true], [0.7, 0.4, false]] }),
    );
    let s = src.with_settings(one_node(vec![pointed]));
    check(&r, &s, &all_states());
    // A retina-size cell with every kind at once.
    let everything = src.with_settings(one_node(vec![
        linear(),
        radial(),
        colour_range(),
        brightness_range(),
        brush_painted(),
        background(),
        object_empty(),
        subject(),
        people(),
    ]));
    let mut problems = Vec::new();
    for (zoom_100, style) in [
        (false, None),
        (true, None),
        (false, Some(MaskStyle::Overlay)),
        (true, Some(MaskStyle::MaskOnly)),
    ] {
        let mut st = ViewportState::new(ViewportRect::cell(1100.0, 760.0, 2.0));
        if zoom_100 {
            st.mode = ViewportMode::Percent { value: 100.0 };
        }
        st.mask_view = style.map(|style| MaskView {
            node: 2,
            style,
            row: None,
            depth: false,
        });
        for pan in 0..2 {
            if pan == 1 {
                st.centre = [0.4, 0.55];
            }
            let f = frame(&r, &everything, &st);
            if f.report.device != ComputeDevice::Gpu || r.raw_on_cpu(everything.id) {
                problems.push(format!(
                    "100 % {zoom_100}, {style:?}, pan {pan}: {:?} {:?}",
                    f.report.device, f.report.notes
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

// ---- our own checks do not end the route -------------------------------------------------------------------

/// `invalid kernel input` is our validation, raised before any device work (`docs/17-gpu.md` section 9): the frame
/// it refuses is the CPU's, with the usual note, and the next frame is the device's. Needs
/// `--features fault-injection` (without it the test is empty).
#[cfg(feature = "fault-injection")]
#[test]
fn a_refused_check_costs_one_cpu_frame_and_leaves_the_route_open() {
    use ym_gpu::device::GpuFault;
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source().with_settings(one_node(vec![brush_painted()]));
    let r = renderer(&gpu, None);
    for style in [None, Some(MaskStyle::Overlay)] {
        let st = view(false, style.map(|s| (s, None)));
        let _ = frame(&r, &src, &st);
        assert!(!r.raw_on_cpu(src.id));
        gpu.set_fault(Some(GpuFault::FailInvalid(1)));
        let refused = frame(&r, &src, &st);
        assert_eq!(refused.report.device, ComputeDevice::Cpu, "{style:?}");
        assert!(
            refused
                .report
                .notes
                .iter()
                .any(|n| n.contains("invalid kernel input")),
            "{:?}",
            refused.report.notes
        );
        assert!(!r.raw_on_cpu(src.id), "{style:?}: the route is still open");
        // The fault fired once: the next frame, and the one after it, are the device's.
        for _ in 0..2 {
            let next = frame(&r, &src, &st);
            assert_eq!(next.report.device, ComputeDevice::Gpu, "{style:?}");
            assert!(!next.report.device_refused, "{style:?}");
            assert!(
                !next
                    .report
                    .notes
                    .iter()
                    .any(|n| n.contains("graphics card")),
                "{:?}",
                next.report.notes
            );
        }
        // And the CPU's frame is the device's within the kernels' tolerance.
        let worst = refused
            .rgba
            .iter()
            .zip(&frame(&r, &src, &st).rgba)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(worst <= 3, "{style:?}: {worst} codes");
        gpu.set_fault(None);
    }
}

/// A true device fault still ends the photo's views for good, and the viewport says so (`raw_on_cpu` is what its
/// "Rendering on the CPU" asks): a shader that does not compile ends the views for the process, a lost device for
/// every renderer on it.
#[cfg(feature = "fault-injection")]
#[test]
fn a_true_device_fault_still_ends_the_route_and_the_viewport_says_the_cpu() {
    use ym_gpu::device::GpuFault;
    for fault in [GpuFault::Shader, GpuFault::Lost] {
        // A device of its own: nothing has compiled on it yet, so the fault meets the first frame.
        let Some(gpu) = self::gpu() else { return };
        let d = photo();
        let src = d.source().with_settings(one_node(vec![brush_painted()]));
        let r = renderer(&gpu, None);
        assert!(!r.raw_on_cpu(src.id));
        gpu.set_fault(Some(fault));
        let st = view(false, None);
        let _ = frame(&r, &src, &st);
        gpu.set_fault(None);
        assert!(r.raw_on_cpu(src.id), "{fault:?}: the route ended");
        assert_eq!(
            frame(&r, &src, &st).report.device,
            ComputeDevice::Cpu,
            "{fault:?}"
        );
    }
}

/// Frames that never asked the device do not say it refused: one served from the cache (the report of the frame that
/// was refused is not repeated), and one of paper only.
#[cfg(feature = "fault-injection")]
#[test]
fn a_cache_hit_and_a_paper_frame_are_not_refusals() {
    use ym_gpu::device::GpuFault;
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source().with_settings(one_node(vec![brush_painted()]));
    let r = renderer(&gpu, None);
    let st = view(false, None);
    let cached = FrameOptions::default();
    gpu.set_fault(Some(GpuFault::FailInvalid(1)));
    let refused = r
        .viewport_frame(&src, &st, &cached, &CancelToken::new())
        .expect("frame");
    gpu.set_fault(None);
    assert!(refused.report.device_refused);
    // The same view again is the cache's.
    let hit = r
        .viewport_frame(&src, &st, &cached, &CancelToken::new())
        .expect("frame");
    assert!(!hit.report.device_refused, "{:?}", hit.report.notes);
    // A cell the photo is not in (panned far off at 100 %) is paper only.
    let mut off = view(true, None);
    off.centre = [40.0, 40.0];
    let paper = r
        .viewport_frame(&src, &off, &cached, &CancelToken::new())
        .expect("frame");
    assert!(!paper.report.device_refused);
}
