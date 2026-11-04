//! The Film node through the render graph on the CPU reference (`docs/24-film.md` sections 2.6, 3.9, 8 and
//! `docs/23` section 5): a graph with a Film node renders, differs from the graph without it, and its pixels are the
//! same for any tile size and any thread count, with halation and grain on; a bypassed node, a node at Strength 0 and
//! a graph without a Film node change no bit.
mod common;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions,
    Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Uniform, 0.12)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn graph(film: Value) -> DevelopSettings {
    let mut node = json!({ "id": 2, "kind": "film", "pos": [1.0, 0.0] });
    node.as_object_mut()
        .unwrap()
        .extend(film.as_object().unwrap().clone());
    let doc = json!({ "nodes": { "items": [ { "id": 1 }, node ], "links": [
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

fn renderer(tile: usize, threads: usize) -> Renderer {
    Renderer::new(
        std::sync::Arc::new(NeutralOnly),
        RenderOptions {
            threads,
            full_tile_px: tile,
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn full(r: &Renderer, src: &PhotoSource, target: FullTarget) -> FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("render")
}

fn srgb(r: &Renderer, src: &PhotoSource) -> FullImage {
    full(r, src, FullTarget::DisplayLinear(OutputTarget::Srgb))
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn bytes(img: &FullImage) -> Vec<u32> {
    img.rgb.iter().map(|v| v.to_bits()).collect()
}

fn busy() -> Value {
    json!({
        "grain": { "amount": 150, "size": 100, "seed": 7 },
        "halation": { "amount": 200, "size": 120 },
        "strength": 80,
        "masks": { "items": [
            { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 }
        ] },
    })
}

#[test]
fn a_film_node_changes_the_picture_and_a_graph_without_one_is_as_it_was() {
    let d = photo();
    let src = d.source();
    let r = renderer(1024, 4);
    let plain = srgb(&r, &src);
    let filmed = srgb(&r, &with(&src, graph(json!({}))));
    assert_ne!(bytes(&plain), bytes(&filmed));
    // The default graph through the same renderer twice is the same.
    assert_eq!(bytes(&plain), bytes(&srgb(&r, &src)));
}

#[test]
fn bypass_and_strength_zero_change_no_bit() {
    let d = photo();
    let src = d.source();
    let r = renderer(1024, 4);
    let plain = bytes(&srgb(&r, &src));
    for film in [json!({ "bypass": true }), json!({ "strength": 0 })] {
        let got = bytes(&srgb(&r, &with(&src, graph(film.clone()))));
        // A node that only passes the picture through is the graph without it, bit for bit (the node
        // stands between Main and the Output and acts as the identity there).
        assert_eq!(got, plain, "{film}");
    }
}

#[test]
fn tiles_and_threads_do_not_change_a_bit_with_halation_grain_and_a_mask() {
    let d = photo();
    let src = with(&d.source(), graph(busy()));
    let whole = bytes(&srgb(&renderer(1024, 4), &src));
    for (tile, threads) in [(160, 1), (97, 3), (256, 4)] {
        let got = bytes(&srgb(&renderer(tile, threads), &src));
        assert_eq!(got, whole, "tile {tile}, {threads} threads");
    }
}

#[test]
fn a_repeat_export_is_identical_and_the_dng_bakes_the_node_in() {
    let d = photo();
    let src = d.source();
    let filmed = with(&src, graph(busy()));
    let r = renderer(1024, 4);
    let a = full(&r, &filmed, FullTarget::DngLinearAwg4);
    let b = full(&r, &filmed, FullTarget::DngLinearAwg4);
    assert_eq!(bytes(&a), bytes(&b));
    let plain = full(&r, &src, FullTarget::DngLinearAwg4);
    assert_ne!(bytes(&a), bytes(&plain));
}

#[test]
fn a_view_shows_the_film_node_and_stays_on_the_cpu() {
    let d = photo();
    let src = d.source();
    let r = renderer(1024, 4);
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let frame = |s: &PhotoSource| {
        r.viewport_frame(s, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
    };
    let plain = frame(&src);
    let filmed = frame(&with(&src, graph(json!({}))));
    assert_ne!(plain.rgba, filmed.rgba);
    assert_eq!(filmed.report.device, ym_core::pipeline::ComputeDevice::Cpu);
}
