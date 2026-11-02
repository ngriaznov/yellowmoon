//! Node thumbnails (`docs/23-nodes.md` section 5.7, task N-05): a picture for every corrector, each the
//! plan stopped after that node's picture and then run through the Output, kept in memory under a key
//! that holds the plan up to the node.
//!
//! * Each node's thumbnail equals a truncated render within one 8-bit code, on the CPU and on the
//!   device. The reference is the same photo with the graph **built to end at that node** (the nodes
//!   after it are not there, the Output reads the node) rendered by the ordinary thumbnail path at the
//!   same size, so it does not share the code that stops the plan; the plan the thumbnail runs is also
//!   compared with the plan compiled with `Stop::Picture`.
//! * A change late in the graph makes no new thumbnail for the nodes before it; a change to the Input
//!   or to the Output makes new ones for all.
//! * A dead node has a thumbnail, and its key is the plan up to it.
//! * The pictures are the same bytes on every run.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::plan::Stop;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::color::node_plan;
use ym_core::pipeline::{
    NODE_THUMBNAIL_PX, NeutralOnly, NodeThumbnail, PhotoSource, RenderOptions, Renderer,
    settings_ending_at,
};
use ym_core::settings::{DevelopSettings, NodeId};
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

/// A node-first document: the correctors given (each a JSON object with an `id`) and these links.
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

/// The corrector nodes of the line Input, 1, 2, 3, 4, Output.
fn line_nodes() -> Vec<Value> {
    vec![
        corrector(2, 1.0, trim(0.5)),
        corrector(3, 2.0, s_curve()),
        corrector(4, 3.0, warm()),
    ]
}

fn three_nodes() -> DevelopSettings {
    graph(line_nodes(), line(&[1, 2, 3, 4]))
}

/// The same graph with a dead branch: node 4 reads Main and nothing reads it.
fn dead_branch() -> DevelopSettings {
    let mut links = line(&[1, 2, 3]);
    links.push(picture(json!(1), json!(4)));
    graph(line_nodes(), links)
}

fn options() -> RenderOptions {
    RenderOptions {
        // A thumbnail of the ordinary path (the reference) is as large as a node's.
        thumbnail_px: NODE_THUMBNAIL_PX,
        ..RenderOptions::default()
    }
}

fn cpu() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options()).expect("render thread")
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
                "\n* GPU TESTS SKIPPED in node_thumbs: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn on_device(gpu: Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            ..options()
        },
    )
    .expect("render thread")
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

/// A PNG as (width, height, RGB8).
fn decode(png_bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes))
        .read_info()
        .expect("a PNG");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("a size")];
    let info = reader.next_frame(&mut buf).expect("a frame");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    buf.truncate(info.buffer_size());
    (info.width as usize, info.height as usize, buf)
}

/// The largest difference of two RGB pictures in 8-bit codes, and the mean.
fn differ(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len());
    let (mut worst, mut sum) = (0u8, 0u64);
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    (worst, sum as f64 / a.len() as f64)
}

fn thumbnails(r: &Renderer, src: &PhotoSource) -> Vec<NodeThumbnail> {
    r.node_thumbnails(src, &CancelToken::new())
        .expect("node thumbnails")
}

fn fresh(t: &[NodeThumbnail]) -> Vec<u32> {
    t.iter().filter(|t| t.fresh).map(|t| t.node).collect()
}

fn ids(t: &[NodeThumbnail]) -> Vec<u32> {
    t.iter().map(|t| t.node).collect()
}

/// The ordinary thumbnail of a photo, decoded.
fn ordinary(r: &Renderer, src: &PhotoSource) -> (usize, usize, Vec<u8>) {
    decode(
        &r.thumbnail_png(src, &CancelToken::new())
            .expect("a thumbnail"),
    )
}

/// What the plan the thumbnail of `node` runs and the plan compiled with `Stop::Picture` say.
fn plans_agree(settings: &DevelopSettings, node: u32) {
    let stopped = node_plan(&settings.nodes, Stop::Picture(node)).unwrap();
    let ours = node_plan(&settings_ending_at(settings, node).nodes, Stop::Output).unwrap();
    assert_eq!(
        format!(
            "{:?}|{:?}|{:?}",
            stopped.passes, stopped.buffers, stopped.result
        ),
        format!("{:?}|{:?}|{:?}", ours.passes, ours.buffers, ours.result),
        "node {node}: the thumbnail runs the plan stopped at the node's picture"
    );
}

// ---- the plan ---------------------------------------------------------------------------------------

#[test]
fn the_thumbnail_runs_the_plan_stopped_at_the_node() {
    for s in [three_nodes(), dead_branch()] {
        for id in [1, 2, 3, 4] {
            plans_agree(&s, id);
        }
    }
    // The Output's other links are replaced, and a graph that already ends there is unchanged.
    let s = three_nodes();
    assert_eq!(settings_ending_at(&s, 4), s);
    let at_two = settings_ending_at(&s, 2);
    assert_ne!(at_two, s);
    assert_eq!(
        node_plan(&at_two.nodes, Stop::Output).unwrap().node_ids(),
        vec![1, 2]
    );
    assert!(at_two.nodes.has(NodeId::Corrector(4)), "the nodes stay");
}

// ---- the pictures -----------------------------------------------------------------------------------

/// Every node's thumbnail against the graph built to end at that node, rendered by the ordinary
/// thumbnail path of `reference` at the same size.
fn check_against_truncated_renders(r: &Renderer, reference: &Renderer, label: &str) -> u8 {
    let d = photo();
    let src = d.source();
    let mut worst_overall = 0u8;
    // The line: the truncated graphs are built from the start, not by cutting the full one.
    /// A case: what it is, the graph, and for each node the graph built to end there.
    type Case = (&'static str, DevelopSettings, Vec<(u32, DevelopSettings)>);
    let cases: [Case; 2] = [
        (
            "a line of four",
            three_nodes(),
            (1..=4u32)
                .map(|k| {
                    let ids: Vec<u32> = (1..=k).collect();
                    let nodes = line_nodes().into_iter().take(k as usize - 1).collect();
                    (k, graph(nodes, line(&ids)))
                })
                .collect(),
        ),
        (
            "a dead branch",
            dead_branch(),
            vec![
                (1, graph(vec![], line(&[1]))),
                (2, graph(vec![corrector(2, 1.0, trim(0.5))], line(&[1, 2]))),
                (
                    3,
                    graph(
                        vec![corrector(2, 1.0, trim(0.5)), corrector(3, 2.0, s_curve())],
                        line(&[1, 2, 3]),
                    ),
                ),
                // Node 4 is dead: its truncated graph is Main, then node 4.
                (4, graph(vec![corrector(4, 3.0, warm())], line(&[1, 4]))),
            ],
        ),
    ];
    for (what, settings, truncated) in cases {
        let t = thumbnails(r, &with(&src, settings));
        assert_eq!(ids(&t), vec![1, 2, 3, 4], "{what}");
        for (node, ending_there) in truncated {
            let got = t.iter().find(|t| t.node == node).expect("a thumbnail");
            let (w, h, rgb) = decode(&got.png);
            let (rw, rh, expected) = ordinary(reference, &with(&src, ending_there));
            assert_eq!((w, h), (rw, rh), "{label} {what} node {node}: the size");
            assert_eq!(w.max(h), NODE_THUMBNAIL_PX.min(w.max(h)));
            let (worst, mean) = differ(&rgb, &expected);
            eprintln!(
                "{label}, {what}, node {node}: {w}x{h}, thumbnail against the truncated render: largest {worst} codes, mean {mean:.5}"
            );
            assert!(worst <= 1, "{label} {what} node {node}: {worst} codes");
            worst_overall = worst_overall.max(worst);
        }
    }
    worst_overall
}

#[test]
fn each_nodes_thumbnail_is_the_truncated_render_on_the_cpu() {
    let r = cpu();
    let reference = cpu();
    check_against_truncated_renders(&r, &reference, "CPU");
}

#[test]
fn each_nodes_thumbnail_is_the_truncated_render_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let r = on_device(Arc::clone(&gpu));
    let reference = on_device(gpu);
    let d = photo();
    // The renders are on the device, not behind a note: the frame says so.
    let t = thumbnails(&r, &with(&d.source(), three_nodes()));
    assert_eq!(t.len(), 4);
    check_against_truncated_renders(&r, &reference, "device");
}

#[test]
fn the_device_and_the_cpu_make_the_same_thumbnails_within_the_device_tolerance() {
    let Some(gpu) = gpu() else { return };
    let device = on_device(gpu);
    let host = cpu();
    let d = photo();
    let src = d.source();
    for (what, s) in [
        ("a line of four", three_nodes()),
        ("a dead branch", dead_branch()),
    ] {
        let a = thumbnails(&device, &with(&src, s.clone()));
        let b = thumbnails(&host, &with(&src, s));
        for (x, y) in a.iter().zip(&b) {
            let (worst, mean) = differ(&decode(&x.png).2, &decode(&y.png).2);
            eprintln!(
                "{what}, node {}: the device against the CPU: largest {worst} codes, mean {mean:.5}",
                x.node
            );
            // render_gpu.rs: the device's frame is the CPU's within two codes.
            assert!(worst <= 2 && mean < 0.25, "{what} node {}", x.node);
        }
    }
}

#[test]
fn the_thumbnails_are_not_the_photo_and_differ_from_node_to_node() {
    let r = cpu();
    let d = photo();
    let t = thumbnails(&r, &with(&d.source(), three_nodes()));
    let pictures: Vec<Vec<u8>> = t.iter().map(|t| decode(&t.png).2).collect();
    for i in 0..pictures.len() {
        for j in i + 1..pictures.len() {
            assert_ne!(
                pictures[i], pictures[j],
                "nodes {} and {} look alike",
                t[i].node, t[j].node
            );
        }
    }
    // They are at most 160 px on the long edge, whole photo (480 x 320 is 3:2).
    let (w, h, _) = decode(&t[0].png);
    assert_eq!((w, h), (160, 107));
}

// ---- the cache --------------------------------------------------------------------------------------

#[test]
fn a_change_to_node_three_makes_no_new_thumbnail_for_nodes_one_and_two() {
    let r = cpu();
    let d = photo();
    let src = d.source();
    let base = with(&src, three_nodes());

    let before = r.counters(base.id).grade_plans;
    let first = thumbnails(&r, &base);
    assert_eq!(ids(&first), vec![1, 2, 3, 4]);
    assert_eq!(
        fresh(&first),
        vec![1, 2, 3, 4],
        "a first call makes them all"
    );
    assert_eq!(r.counters(base.id).grade_plans - before, 4);

    // Asking again makes nothing: every picture is in memory, the same one.
    let again = thumbnails(&r, &base);
    assert_eq!(fresh(&again), Vec::<u32>::new());
    for (a, b) in first.iter().zip(&again) {
        assert_eq!(a.version, b.version);
        assert!(Arc::ptr_eq(&a.png, &b.png));
    }
    assert_eq!(
        r.counters(base.id).grade_plans - before,
        4,
        "a hit builds no plan"
    );

    // Node 3 changes: nodes 1 and 2 are cache hits, 3 and 4 (which read it) are made again.
    let mut changed = base.settings.clone();
    changed.nodes.corrector_mut(3).unwrap().trim.exposure_ev = -1.0;
    let after = thumbnails(&r, &with(&src, changed.clone()));
    assert_eq!(fresh(&after), vec![3, 4]);
    for node in [1, 2] {
        let (a, b) = (&first[node as usize - 1], &after[node as usize - 1]);
        assert_eq!((a.node, b.node), (node, node));
        assert_eq!(a.version, b.version, "node {node} is the picture it was");
        assert!(Arc::ptr_eq(&a.png, &b.png));
    }
    for node in [3, 4] {
        let (a, b) = (&first[node as usize - 1], &after[node as usize - 1]);
        assert_ne!(a.version, b.version, "node {node} is a new picture");
        assert_ne!(a.png, b.png, "and a different one");
    }
    assert_eq!(r.counters(base.id).grade_plans - before, 6);

    // Node 4 changes: only node 4.
    let mut last = changed.clone();
    last.nodes.corrector_mut(4).unwrap().trim.exposure_ev = 1.0;
    assert_eq!(fresh(&thumbnails(&r, &with(&src, last))), vec![4]);

    // Going back to the first settings finds nothing kept for 3 and 4 (the cache holds one picture per
    // node), and nodes 1 and 2 are still hits.
    assert_eq!(fresh(&thumbnails(&r, &base)), vec![3, 4]);
}

#[test]
fn a_change_before_the_nodes_or_in_the_output_makes_new_thumbnails_for_all() {
    let r = cpu();
    let d = photo();
    let src = d.source();
    let base = with(&src, three_nodes());
    thumbnails(&r, &base);

    // The Input: a setting of the white balance, the exposure, the crop.
    let mut exposed = base.settings.clone();
    exposed.input_mut().exposure =
        ym_core::settings::AutoOr::Manual(ym_core::settings::ExposureValue { ev: 0.7 });
    assert_eq!(
        fresh(&thumbnails(&r, &with(&src, exposed))),
        vec![1, 2, 3, 4]
    );
    thumbnails(&r, &base);

    let mut cropped = base.settings.clone();
    cropped.input_mut().orientation = ym_core::settings::Orientation::Deg90;
    let turned = thumbnails(&r, &with(&src, cropped));
    assert_eq!(fresh(&turned), vec![1, 2, 3, 4]);
    let (w, h, _) = decode(&turned[0].png);
    assert!(h > w, "turned: {w}x{h}");
    thumbnails(&r, &base);

    // The Output: every thumbnail goes through it.
    let mut intensity = base.settings.clone();
    intensity.output_mut().look_intensity = 40.0;
    assert_eq!(
        fresh(&thumbnails(&r, &with(&src, intensity))),
        vec![1, 2, 3, 4]
    );
    thumbnails(&r, &base);
    let mut sharp = base.settings.clone();
    sharp.output_mut().output_sharpening = 40.0;
    assert_eq!(fresh(&thumbnails(&r, &with(&src, sharp))), vec![1, 2, 3, 4]);
    thumbnails(&r, &base);

    // Nothing that is not in a picture makes one: a name, a place on the canvas.
    let mut moved = base.settings.clone();
    moved.nodes.corrector_mut(3).unwrap().name = "Contrast".to_owned();
    moved.nodes.corrector_mut(3).unwrap().pos = [2.0, 1.5];
    assert_eq!(
        fresh(&thumbnails(&r, &with(&src, moved))),
        Vec::<u32>::new()
    );
}

#[test]
fn a_node_that_is_added_or_removed_changes_only_the_pictures_it_reaches() {
    let r = cpu();
    let d = photo();
    let src = d.source();
    thumbnails(
        &r,
        &with(&src, graph(line_nodes()[..2].to_vec(), line(&[1, 2, 3]))),
    );
    // Node 4 is added at the end: one new picture.
    let four = thumbnails(&r, &with(&src, three_nodes()));
    assert_eq!(fresh(&four), vec![4]);
    // Node 2 is taken out of the line (Main, 3, 4): the pictures of 3 and 4 change, Main's does not,
    // and node 2 has no picture any more.
    let without_two = graph(
        vec![corrector(3, 2.0, s_curve()), corrector(4, 3.0, warm())],
        line(&[1, 3, 4]),
    );
    let t = thumbnails(&r, &with(&src, without_two));
    assert_eq!(ids(&t), vec![1, 3, 4]);
    assert_eq!(fresh(&t), vec![3, 4]);
}

#[test]
fn a_dead_node_has_a_thumbnail_made_from_the_plan_up_to_it() {
    let r = cpu();
    let d = photo();
    let src = d.source();
    let base = with(&src, dead_branch());
    let first = thumbnails(&r, &base);
    // Every node of the photo, the dead one too.
    assert_eq!(ids(&first), vec![1, 2, 3, 4]);
    assert_eq!(fresh(&first), vec![1, 2, 3, 4]);
    let plan = node_plan(&base.settings.nodes, Stop::Output).unwrap();
    assert!(
        plan.cards.iter().any(|c| c.id == 4 && !c.live),
        "node 4 is dead in the plan"
    );
    // It shows the photo after Main and node 4 (it is not the live picture, and not Main's).
    let live = first.iter().find(|t| t.node == 3).unwrap();
    let dead = first.iter().find(|t| t.node == 4).unwrap();
    let main = first.iter().find(|t| t.node == 1).unwrap();
    assert_ne!(dead.png, live.png);
    assert_ne!(dead.png, main.png);

    // A change to node 3 (live, after Main) leaves the dead node's picture alone: its plan is Main
    // and itself.
    let mut changed = base.settings.clone();
    changed.nodes.corrector_mut(3).unwrap().trim.exposure_ev = -1.0;
    let t = thumbnails(&r, &with(&src, changed.clone()));
    assert_eq!(fresh(&t), vec![3]);
    // A change to the dead node changes its own picture and no other.
    let mut warmer = changed.clone();
    warmer.nodes.corrector_mut(4).unwrap().trim.temperature = -40.0;
    let t = thumbnails(&r, &with(&src, warmer.clone()));
    assert_eq!(fresh(&t), vec![4]);
    // A change to Main reaches everything that reads it, the dead node included.
    let mut main_changed = warmer;
    main_changed.main_mut().trim.exposure_ev = 0.4;
    assert_eq!(
        fresh(&thumbnails(&r, &with(&src, main_changed))),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn a_bypassed_node_has_a_thumbnail_like_the_one_before_it() {
    let r = cpu();
    let d = photo();
    let src = d.source();
    let mut s = three_nodes();
    s.nodes.corrector_mut(3).unwrap().bypass = true;
    let t = thumbnails(&r, &with(&src, s));
    // Node 3 is bypassed: its picture is its input's, node 2's, to the bit.
    assert_eq!(decode(&t[2].png).2, decode(&t[1].png).2);
    assert_ne!(decode(&t[3].png).2, decode(&t[2].png).2);
}

// ---- determinism ------------------------------------------------------------------------------------

#[test]
fn the_thumbnails_are_the_same_bytes_every_time() {
    let d = photo();
    let src = with(&d.source(), three_nodes());
    let a = thumbnails(&cpu(), &src);
    // A renderer of its own, nothing kept.
    let b = thumbnails(&cpu(), &src);
    assert_eq!(fresh(&a), vec![1, 2, 3, 4]);
    assert_eq!(fresh(&b), vec![1, 2, 3, 4]);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.png, y.png, "node {}", x.node);
    }
    // And a picture served from memory is the picture that was made.
    let r = cpu();
    let made = thumbnails(&r, &src);
    let kept = thumbnails(&r, &src);
    for (x, y) in made.iter().zip(&kept) {
        assert_eq!(x.png, y.png);
        assert_eq!(x.png, a[x.node as usize - 1].png);
    }
}

#[test]
fn the_thumbnails_are_the_same_bytes_on_the_device_every_time() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = with(&d.source(), three_nodes());
    let a = thumbnails(&on_device(Arc::clone(&gpu)), &src);
    let b = thumbnails(&on_device(gpu), &src);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.png, y.png, "node {}", x.node);
    }
}

// ---- the scheduled call -----------------------------------------------------------------------------

#[test]
fn the_spawned_call_gives_what_the_direct_call_gives_and_a_closed_session_cancels_it() {
    let r = cpu();
    let d = photo();
    let src = with(&d.source(), three_nodes());
    let queued = r
        .spawn_node_thumbnails(src.clone())
        .wait()
        .expect("the job ran")
        .expect("thumbnails");
    assert_eq!(fresh(&queued), vec![1, 2, 3, 4]);
    let direct = thumbnails(&r, &src);
    for (a, b) in queued.iter().zip(&direct) {
        assert_eq!(a.version, b.version);
    }
    // A photo of a closed session is refused as cancelled.
    r.retire_photos(&[src.id]);
    assert!(matches!(
        r.node_thumbnails(&src, &CancelToken::new()),
        Err(ym_core::pipeline::RenderError::Cancelled)
    ));
}
