//! The Film node through the render graph on the CPU reference (`docs/24-film.md` sections 2.6, 3.9, 8 and
//! `docs/23` section 5): a graph with a Film node renders, differs from the graph without it, and its pixels are the
//! same for any tile size and any thread count, with halation and grain on; a bypassed node, a node at Strength 0 and
//! a graph without a Film node change no bit.
mod common;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullImage, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly,
    PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
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

/// Two Film nodes in a line, then a corrector: the halos add up, and tiles still equal the whole frame.
#[test]
fn two_film_nodes_in_a_line_and_a_corrector_after_them_are_tile_exact() {
    let d = photo();
    let doc = json!({ "nodes": { "items": [
        { "id": 2, "kind": "film", "pos": [1, 0], "film": { "stock": "kodak_portra_400" },
          "grain": { "amount": 120, "seed": 2 }, "halation": { "amount": 250 } },
        { "id": 3, "kind": "film", "pos": [2, 0], "film": { "stock": "fujifilm_velvia_100" },
          "grain": { "amount": 80, "seed": 3 }, "strength": 60,
          "masks": { "items": [ { "kind": "radial", "center": [0.4, 0.5], "radii": [0.3, 0.2], "feather": 0.5 } ] } },
        { "id": 4, "kind": "corrector", "pos": [3, 0], "trim": { "exposure_ev": 0.4 } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": 4, "port": "picture" } },
        { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = with(&d.source(), settings);
    let whole = bytes(&srgb(&renderer(1024, 4), &src));
    assert_ne!(whole, bytes(&srgb(&renderer(1024, 4), &d.source())));
    for (tile, threads) in [(130, 2), (200, 4)] {
        assert_eq!(
            bytes(&srgb(&renderer(tile, threads), &src)),
            whole,
            "tile {tile}"
        );
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

#[test]
fn show_mask_and_node_thumbnails_work_for_a_film_node() {
    let d = photo();
    let src = with(&d.source(), graph(busy()));
    let r = renderer(1024, 4);
    // Mask only: the code value is the node's mask, here a Linear band times Strength 80 %; the Film node's picture
    // is not made at all for it.
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
    st.mask_view = Some(MaskView {
        node: 2,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
    });
    let frame = r
        .viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("mask view");
    let row = |y: usize| -> Vec<u8> { (0..300).map(|x| frame.rgba[(y * 300 + x) * 4]).collect() };
    let mid = row(150);
    assert!(
        mid.first() != mid.last(),
        "the Linear mask is a ramp across the photo: {mid:?}"
    );
    assert!(
        mid.iter().all(|v| *v <= 204),
        "the mask is at most Strength, 80 %: {}",
        mid.iter().max().unwrap()
    );
    // The photo's cards each get a picture, the Film node's included.
    let thumbs = r
        .node_thumbnails(&src, &CancelToken::new())
        .expect("node thumbnails");
    assert_eq!(
        thumbs.iter().map(|t| t.node).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_ne!(
        thumbs[0].png, thumbs[1].png,
        "the Film node's card shows the film"
    );
}

/// Times of the CPU path (`cargo test --release -p ym-core --test render_film -- --ignored --nocapture timings`):
/// the first tables, a view of the fit size, and a full-resolution export of 24 MP, with a Film node at its
/// defaults (halation and grain on).
#[test]
#[ignore = "a measurement, not a check"]
fn timings() {
    use std::time::Instant;
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(6000, 4000, white, Layout::Uniform, 0.12)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let r = renderer(1024, 0);
    let st = ViewportState::new(ViewportRect::cell(1600.0, 1066.0, 1.0));
    let frame = |s: &PhotoSource| {
        let t = Instant::now();
        let f = r
            .viewport_frame(s, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame");
        (t.elapsed().as_secs_f64(), f.width, f.height)
    };
    let (plain, w, h) = frame(&src);
    let filmed = with(&src, graph(json!({})));
    let (cold, ..) = frame(&filmed);
    let (grain_only, ..) = frame(&with(&src, graph(json!({ "grain": { "seed": 1 } }))));
    let (warm, ..) = frame(&with(
        &src,
        graph(json!({ "film": { "exposure_ev": 0.5 } })),
    ));
    eprintln!(
        "view {w}x{h}: without the node {plain:.2} s, first with it (tables baked) {cold:.2} s, new seed {grain_only:.2} s, new exposure {warm:.2} s"
    );
    let t = Instant::now();
    let a = srgb(&r, &filmed);
    eprintln!(
        "export {}x{} ({:.1} MP) with a Film node: {:.1} s",
        a.width,
        a.height,
        (a.width * a.height) as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let _ = srgb(&r, &src);
    eprintln!(
        "the same export without it: {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let (built_chains, built_models) = ym_core::film::FilmEngine::shared(&data_root())
        .map(|e| e.built())
        .unwrap_or_default();
    eprintln!(
        "built: {built_chains} chains, {built_models} models, {:?}",
        ym_core::film::FilmEngine::shared(&data_root()).map(|e| e.baker().built())
    );
}
