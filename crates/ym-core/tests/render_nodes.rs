//! The nodes on the CPU reference through the render graph (`docs/23-nodes.md` sections 1.3, 5.1 to 5.3
//! and 5.10): a graph of correctors renders as its plan says, a node at its identity, a bypassed node and
//! a dead branch change no bit, an exposure trim is a stop of exposure, the nodes are deterministic whatever
//! the thread count and the tile size, a graph with nodes beyond Main renders on the CPU when a device is
//! attached (with a note) and the grade cache key sees the plan, every node and the look.
//!
//! Masks are not rendered until N-10: a document with mask rows renders as if they were off.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::color::{
    GradeRequest, NODES_ON_CPU_NOTE, color_plan, grade_key, grade_plan, node_plan,
    nodes_run_on_device,
};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource,
    RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn photo(layout: Layout, level: f64) -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, layout, level)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

/// A node-first document: the correctors given (each a JSON object with an `id`), and the picture links
/// `chain` joins in a line from the Input to the Output (`[1, 2, 3]` is Input, 1, 2, 3, Output).
fn graph(nodes: Vec<Value>, links: Vec<Value>) -> DevelopSettings {
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

/// Links in a line: the Input, then these ids, then the Output.
fn line(ids: &[u32]) -> Vec<Value> {
    let mut links = Vec::new();
    let mut from = json!("input");
    for id in ids {
        links.push(picture(from, json!(id)));
        from = json!(id);
    }
    links.push(picture(from, json!("output")));
    links
}

fn corrector(id: u32, x: f64, body: Value) -> Value {
    let mut node = json!({ "id": id, "kind": "corrector", "pos": [x, 0.0] });
    if let (Some(n), Some(b)) = (node.as_object_mut(), body.as_object()) {
        n.extend(b.clone());
    }
    node
}

fn trim(ev: f64) -> Value {
    json!({ "trim": { "exposure_ev": ev } })
}

fn s_curve() -> Value {
    json!({ "grade": { "curve": { "luma": [[0.0, 0.0], [0.25, 0.18], [0.6, 0.66], [1.0, 1.0]] } } })
}

fn warm() -> Value {
    json!({ "trim": { "temperature": 60.0, "tint": -25.0 }, "grade": { "contrast": { "kind": "manual", "value": 30.0 } } })
}

fn three_nodes() -> DevelopSettings {
    graph(
        vec![
            corrector(2, 1.0, trim(0.5)),
            corrector(3, 2.0, s_curve()),
            corrector(4, 3.0, warm()),
        ],
        line(&[1, 2, 3, 4]),
    )
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn full(r: &Renderer, src: &PhotoSource, target: FullTarget) -> FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("render")
}

fn display(r: &Renderer, src: &PhotoSource) -> FullImage {
    full(r, src, FullTarget::DisplayLinear(OutputTarget::Srgb))
}

fn frame(r: &Renderer, src: &PhotoSource) -> Vec<u8> {
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    r.viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
        .rgba
}

fn renderer() -> Renderer {
    Renderer::neutral_only().expect("render thread")
}

fn centre(img: &FullImage) -> [f64; 3] {
    let (w, h) = (img.width, img.height);
    let (x0, y0, cw, ch) = (w * 3 / 8, h * 3 / 8, w / 4, h / 4);
    let mut sum = [0.0f64; 3];
    for y in y0..y0 + ch {
        for x in x0..x0 + cw {
            for (c, s) in sum.iter_mut().enumerate() {
                *s += f64::from(img.rgb[(y * w + x) * 3 + c]);
            }
        }
    }
    let n = (cw * ch) as f64;
    sum.map(|v| v / n)
}

// ---- the plan ----------------------------------------------------------------------------------

#[test]
fn the_default_graph_is_main_in_one_pass_and_a_chain_is_its_nodes_in_order() {
    let plan = node_plan(
        &DevelopSettings::default().nodes,
        ym_color::plan::Stop::Output,
    )
    .unwrap();
    assert_eq!(plan, ym_color::plan::Plan::main_only());
    let plan = node_plan(&three_nodes().nodes, ym_color::plan::Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 2, 3, 4]);
    assert_eq!(plan.passes.len(), 1);
    assert_eq!(
        plan.cards.iter().map(|c| c.number).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn the_plan_follows_the_links_not_the_ids_or_the_order_of_the_file() {
    // Node 2 sits at the end of the line and node 4 at the start: the plan is Main, 4, 3, 2.
    let s = graph(
        vec![
            corrector(2, 3.0, json!({})),
            corrector(3, 2.0, json!({})),
            corrector(4, 1.0, json!({})),
        ],
        line(&[1, 4, 3, 2]),
    );
    let plan = node_plan(&s.nodes, ym_color::plan::Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 4, 3, 2]);
    // The same graph with its items listed the other way round is the same plan.
    let doc = serde_json::to_value(&s).unwrap();
    let mut items = doc["nodes"]["items"].as_array().unwrap().clone();
    items.reverse();
    let mut links = doc["nodes"]["links"].as_array().unwrap().clone();
    links.reverse();
    let reversed: DevelopSettings =
        serde_json::from_value(json!({ "nodes": { "items": items, "links": links } })).unwrap();
    assert_eq!(
        node_plan(&reversed.nodes, ym_color::plan::Stop::Output).unwrap(),
        plan
    );
}

#[test]
fn a_dead_node_is_numbered_and_left_out_of_the_plan() {
    let s = graph(
        vec![corrector(2, 1.0, trim(1.0)), corrector(5, 1.0, trim(-2.0))],
        {
            let mut l = line(&[1, 2]);
            l.push(picture(json!(1), json!(5)));
            l
        },
    );
    let plan = node_plan(&s.nodes, ym_color::plan::Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 2]);
    let dead = plan.card(5).unwrap();
    assert!(!dead.live);
    assert_eq!(dead.number, 3);
}

// ---- pixels --------------------------------------------------------------------------------------

#[test]
fn an_exposure_trim_is_a_stop_of_exposure() {
    // The DNG's pixels are the scene-linear AWG4 values after the whole LogC4 chain: a trim of +1 EV in a
    // node doubles them, to the f32 output (relative 1e-5), and the Input's exposure of +1 EV, whose
    // highlight range and toe come first, agrees within 3 % (as render_pixels does for the Input alone).
    for level in [0.05, 0.12] {
        let d = photo(Layout::Uniform, level);
        let src = d.source();
        let r = renderer();
        let dng = |s: &PhotoSource| centre(&full(&r, s, FullTarget::DngLinearAwg4));
        let base = dng(&src);
        for (ev, scale) in [(1.0, 2.0), (-1.0, 0.5), (0.5, 0.5f64.exp2())] {
            let trimmed = dng(&with(
                &src,
                graph(vec![corrector(2, 1.0, trim(ev))], line(&[1, 2])),
            ));
            for c in 0..3 {
                let ratio = trimmed[c] / base[c];
                assert!(
                    (ratio / scale - 1.0).abs() < 1e-5,
                    "level {level}, trim {ev:+} EV, channel {c}: {ratio} against {scale}"
                );
            }
        }
        // The Input's exposure +1 stop (from the photo's Auto value) against the node's trim +1.
        let mut s = src.settings.clone();
        let auto = src.estimates.exposure.ev;
        s.input_mut().exposure =
            ym_core::settings::AutoOr::Manual(ym_core::settings::ExposureValue { ev: auto + 1.0 });
        let exposed = dng(&with(&src, s));
        for c in 0..3 {
            assert!(
                (exposed[c] / base[c] / 2.0 - 1.0).abs() < 0.03,
                "level {level}: the Input's stop is {}",
                exposed[c] / base[c]
            );
        }
    }
}

#[test]
fn nodes_change_the_picture_and_their_order_matters() {
    let d = photo(Layout::Patchwork, 0.5);
    let src = d.source();
    let r = renderer();
    let plain = display(&r, &src);
    let a = with(
        &src,
        graph(
            vec![corrector(2, 1.0, trim(1.0)), corrector(3, 2.0, s_curve())],
            line(&[1, 2, 3]),
        ),
    );
    // The same two nodes, the curve first.
    let b = with(
        &src,
        graph(
            vec![corrector(2, 2.0, trim(1.0)), corrector(3, 1.0, s_curve())],
            line(&[1, 3, 2]),
        ),
    );
    let (ia, ib) = (display(&r, &a), display(&r, &b));
    assert_ne!(ia.rgb, plain.rgb);
    assert_ne!(ib.rgb, plain.rgb);
    assert_ne!(ia.rgb, ib.rgb, "the two orders gave one picture");
    // The frames and the DNG follow the same plan.
    assert_ne!(frame(&r, &a), frame(&r, &b));
    assert_ne!(
        full(&r, &a, FullTarget::DngLinearAwg4).rgb,
        full(&r, &b, FullTarget::DngLinearAwg4).rgb
    );
}

#[test]
fn a_bypassed_node_a_default_node_and_a_dead_branch_change_no_bit() {
    let d = photo(Layout::Patchwork, 0.5);
    let src = d.source();
    let r = renderer();
    let clean = with(
        &src,
        graph(vec![corrector(2, 1.0, trim(0.5))], line(&[1, 2])),
    );
    let reference = display(&r, &clean);
    let reference_frame = frame(&r, &clean);
    let reference_dng = full(&r, &clean, FullTarget::DngLinearAwg4);

    // A bypassed node with strong values in the middle.
    let bypassed = with(
        &src,
        graph(
            vec![
                corrector(2, 1.0, trim(0.5)),
                corrector(
                    3,
                    2.0,
                    json!({ "bypass": true, "trim": { "exposure_ev": 3.0, "temperature": 80.0 } }),
                ),
            ],
            line(&[1, 3, 2]),
        ),
    );
    // Nodes at their defaults, before and after.
    let defaults = with(
        &src,
        graph(
            vec![
                corrector(2, 2.0, trim(0.5)),
                corrector(3, 1.0, json!({})),
                corrector(
                    4,
                    3.0,
                    json!({ "trim": { "exposure_ev": 0.0, "temperature": 0.0, "tint": 0.0 } }),
                ),
            ],
            line(&[1, 3, 2, 4]),
        ),
    );
    // A dead branch: nodes that read Main and go nowhere, and a mask row that nothing renders yet.
    let dead = with(&src, {
        let mut links = line(&[1, 2]);
        links.push(picture(json!(1), json!(5)));
        links.push(picture(json!(5), json!(6)));
        graph(
            vec![
                corrector(2, 1.0, trim(0.5)),
                corrector(5, 1.0, trim(-3.0)),
                corrector(6, 2.0, s_curve()),
            ],
            links,
        )
    });
    for (name, s) in [
        ("bypassed", bypassed),
        ("defaults", defaults),
        ("dead", dead),
    ] {
        assert_eq!(display(&r, &s).rgb, reference.rgb, "{name}");
        assert_eq!(frame(&r, &s), reference_frame, "{name}");
        assert_eq!(
            full(&r, &s, FullTarget::DngLinearAwg4).rgb,
            reference_dng.rgb,
            "{name}"
        );
    }
}

#[test]
fn masks_are_not_rendered_until_n_10() {
    // A mask row on a node renders as if it were off: m is 1 everywhere.
    let d = photo(Layout::Patchwork, 0.5);
    let src = d.source();
    let r = renderer();
    let without = with(
        &src,
        graph(vec![corrector(2, 1.0, trim(1.0))], line(&[1, 2])),
    );
    let masked = with(
        &src,
        graph(
            vec![corrector(
                2,
                1.0,
                json!({ "trim": { "exposure_ev": 1.0 }, "masks": { "invert": false, "items": [
                    { "kind": "radial", "mode": "add", "center": [0.5, 0.5], "radii": [0.1, 0.1], "feather": 0.2 }
                ] } }),
            )],
            line(&[1, 2]),
        ),
    );
    assert_eq!(display(&r, &masked).rgb, display(&r, &without).rgb);
}

#[test]
fn a_node_that_only_makes_a_mask_is_live_and_changes_no_pixel_yet() {
    let d = photo(Layout::Patchwork, 0.5);
    let src = d.source();
    let r = renderer();
    let plain = with(
        &src,
        graph(vec![corrector(3, 2.0, trim(1.0))], line(&[1, 3])),
    );
    // Node 5 reads Main, goes nowhere with its picture, and its key feeds node 3's key input.
    let mut links = line(&[1, 3]);
    links.push(picture(json!(1), json!(5)));
    links.push(key(5, 3));
    let keyed = with(
        &src,
        graph(
            vec![
                corrector(3, 2.0, trim(1.0)),
                corrector(5, 1.0, json!({ "trim": { "exposure_ev": 2.0 } })),
            ],
            links,
        ),
    );
    let plan = node_plan(&keyed.settings.nodes, ym_color::plan::Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 5, 3]);
    assert_eq!(display(&r, &keyed).rgb, display(&r, &plain).rgb);
}

#[test]
fn three_nodes_are_deterministic_whatever_the_threads_and_the_tiles() {
    let d = photo(Layout::Patchwork, 0.5);
    let src = with(&d.source(), three_nodes());
    let render = |threads: usize, tile: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let r = Renderer::new(
                Arc::new(NeutralOnly),
                RenderOptions {
                    threads,
                    full_tile_px: tile,
                    ..RenderOptions::default()
                },
            )
            .unwrap();
            (
                display(&r, &src).rgb,
                full(&r, &src, FullTarget::DngLinearAwg4).rgb,
                frame(&r, &src),
            )
        })
    };
    let one = render(1, 512);
    // A repeat is the same, other thread counts and tile sizes are the same.
    assert!(render(1, 512) == one, "a repeat differs");
    for (threads, tile) in [(2, 512), (4, 512), (1, 160), (4, 160), (2, 96)] {
        let other = render(threads, tile);
        assert!(
            other.0 == one.0,
            "the display picture with {threads} threads and {tile} px tiles"
        );
        assert!(
            other.1 == one.1,
            "the DNG with {threads} threads and {tile} px tiles"
        );
        assert!(
            other.2 == one.2,
            "the frame with {threads} threads and {tile} px tiles"
        );
    }
    // And the nodes did something.
    let plain = renderer();
    assert_ne!(display(&plain, &d.source()).rgb, one.0);
}

// ---- the grade cache ---------------------------------------------------------------------------

#[test]
fn the_grade_key_sees_the_plan_every_node_and_the_look_and_ignores_a_dead_branch() {
    let d = photo(Layout::Uniform, 0.12);
    let src = d.source();
    let r = renderer();
    let color = color_plan(&src).unwrap();
    let req = GradeRequest {
        target: OutputTarget::Srgb,
        dither: false,
        looks: r.looks().as_ref(),
    };
    let key_of = |s: DevelopSettings| grade_key(&with(&src, s), &color, &req);
    let base = key_of(three_nodes());
    assert_eq!(base, key_of(three_nodes()));
    assert_ne!(base, key_of(src.settings.clone()), "the default graph");
    // Every node's values.
    for (what, body) in [
        (
            "a trim",
            vec![
                corrector(2, 1.0, trim(0.6)),
                corrector(3, 2.0, s_curve()),
                corrector(4, 3.0, warm()),
            ],
        ),
        (
            "a curve",
            vec![
                corrector(2, 1.0, trim(0.5)),
                corrector(
                    3,
                    2.0,
                    json!({ "grade": { "curve": { "luma": [[0.0, 0.0], [0.25, 0.2], [0.6, 0.66], [1.0, 1.0]] } } }),
                ),
                corrector(4, 3.0, warm()),
            ],
        ),
        (
            "a bypass",
            vec![
                corrector(2, 1.0, trim(0.5)),
                corrector(3, 2.0, s_curve()),
                corrector(4, 3.0, {
                    let mut w = warm();
                    w["bypass"] = json!(true);
                    w
                }),
            ],
        ),
    ] {
        assert_ne!(base, key_of(graph(body, line(&[1, 2, 3, 4]))), "{what}");
    }
    // The order (the plan).
    assert_ne!(
        base,
        key_of(graph(
            vec![
                corrector(2, 1.0, trim(0.5)),
                corrector(3, 2.0, s_curve()),
                corrector(4, 3.0, warm())
            ],
            line(&[1, 3, 2, 4]),
        ))
    );
    // A dead branch is not in it.
    let mut links = line(&[1, 2, 3, 4]);
    links.push(picture(json!(1), json!(9)));
    let with_dead = graph(
        vec![
            corrector(2, 1.0, trim(0.5)),
            corrector(3, 2.0, s_curve()),
            corrector(4, 3.0, warm()),
            corrector(9, 5.0, trim(2.0)),
        ],
        links,
    );
    assert_eq!(base, key_of(with_dead.clone()));
    // So the second render of the same photo with a dead node added builds no grade plan.
    let a = with(&src, three_nodes());
    let b = with(&src, with_dead);
    let first = r
        .render_full(
            &a,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(first.report.runs.grade_plans, 1);
    let second = r
        .render_full(
            &b,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .unwrap();
    assert_eq!(
        second.report.runs.grade_plans, 0,
        "a dead node rebuilt the grade plan"
    );
    assert_eq!(first.rgb, second.rgb);
    // The look is in it too.
    let mut s = three_nodes();
    s.output_mut().look_intensity = 40.0;
    assert_ne!(base, key_of(s));
    // grade_plan builds the chain the key describes.
    let plan = grade_plan(&with(&src, three_nodes()), &color, &req).unwrap();
    assert_eq!(plan.key, base);
    assert_eq!(plan.chain.params().nodes.plan.node_ids(), vec![1, 2, 3, 4]);
    assert!(!plan.chain.nodes_run_on_device());
}

// ---- the device -----------------------------------------------------------------------------------

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
                "\n* GPU TESTS SKIPPED in render_nodes: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

#[test]
fn the_device_runs_main_alone_and_the_cpu_runs_the_rest_with_a_note() {
    let Some(gpu) = gpu() else { return };
    let d = photo(Layout::Patchwork, 0.5);
    let src = d.source();
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            ..RenderOptions::default()
        },
    )
    .unwrap();
    let no_colour = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL.difference(GpuStages::COLOR.union(GpuStages::EXPORT_TAIL)),
            ..RenderOptions::default()
        },
    )
    .unwrap();
    let cpu = renderer();
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let frame_of = |r: &Renderer, s: &PhotoSource| {
        r.viewport_frame(s, &st, &FrameOptions::default(), &CancelToken::new())
            .unwrap()
    };

    // The default graph: Main alone runs on the device, with no note.
    assert!(nodes_run_on_device(&src.settings));
    let main_only = frame_of(&on_gpu, &src);
    assert_eq!(main_only.report.device, ComputeDevice::Gpu);
    assert!(
        !main_only
            .report
            .notes
            .iter()
            .any(|n| n == NODES_ON_CPU_NOTE)
    );

    // Graphs the device does not run yet: nodes after Main, a trim in Main, a bypassed Main.
    let mut trimmed_main = src.settings.clone();
    trimmed_main.main_mut().trim.exposure_ev = 0.5;
    let mut bypassed_main = src.settings.clone();
    bypassed_main.main_mut().bypass = true;
    for (what, s) in [
        ("three nodes", three_nodes()),
        ("a trim in Main", trimmed_main),
        ("Main bypassed", bypassed_main),
    ] {
        assert!(!nodes_run_on_device(&s), "{what}");
        let photo = with(&src, s);
        let device = frame_of(&on_gpu, &photo);
        assert_eq!(device.report.device, ComputeDevice::Cpu, "{what}");
        assert!(
            device.report.notes.iter().any(|n| n == NODES_ON_CPU_NOTE),
            "{what}: {:?}",
            device.report.notes
        );
        // The colour pass left the device and nothing else did: the bytes of a renderer whose
        // colour stages are switched off, and the CPU renderer's within what the raw chain of the
        // device differs by (render_gpu.rs: two codes, almost always zero). The CPU renderer says
        // nothing about nodes.
        let colour_off = frame_of(&no_colour, &photo);
        assert_eq!(device.rgba, colour_off.rgba, "{what}: the frame");
        let reference = frame_of(&cpu, &photo);
        assert!(
            !reference
                .report
                .notes
                .iter()
                .any(|n| n == NODES_ON_CPU_NOTE)
        );
        let (mut worst, mut sum) = (0u8, 0u64);
        for (x, y) in device.rgba.iter().zip(&reference.rgba) {
            let d = x.abs_diff(*y);
            worst = worst.max(d);
            sum += u64::from(d);
        }
        let mean = sum as f64 / device.rgba.len() as f64;
        eprintln!(
            "{what}: the CPU colour pass behind the device's raw chain differs from the CPU by {worst} codes, mean {mean:.4}"
        );
        assert!(
            worst <= 2 && mean < 0.25,
            "{what}: {worst} codes, mean {mean}"
        );
        // The full-resolution renders follow the same rule.
        let full_device = display(&on_gpu, &photo);
        assert!(
            full_device
                .report
                .notes
                .iter()
                .any(|n| n == NODES_ON_CPU_NOTE),
            "{what}"
        );
        assert_eq!(full_device.report.device, ComputeDevice::Cpu, "{what}");
    }
}
