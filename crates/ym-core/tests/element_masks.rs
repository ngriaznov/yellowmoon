//! The element masks of Reference Grade through the render graph (`docs/34-reference-grade.md` sections 4a.3, 7.1 and 10, RG-04):
//! a node's element grades act through soft masks made from the photo's scene, which a background worker makes once per photo from
//! the Input's picture (never on disk, dropped with the photo), only for the elements whose effective grade is not the identity;
//! an export makes its own scene; a copy with no scene model draws the view without the element step, says so and refuses an
//! export.
//!
//! These tests run on a **fake scene model** (`SceneModel::fake`), so they need no weights. The real networks are checked in
//! `crates/ym-ml/tests/eomt.rs` and `tests/scene_real.rs` (which skips without them), and the app against `ym` in
//! `crates/ym-cli/tests/parity.rs`.
mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NO_SCENE_MODEL_SENTENCE, NeutralOnly, PhotoSource,
    RenderError, RenderOptions, Renderer, SceneModel, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;
use ym_ml::scene::{GROUND, SKIN, SKY, SUBJECT};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
const WIDTH: usize = 640;
const HEIGHT: usize = 420;

/// A fake scene that counts its runs: sky in the top third, ground below, a subject disc with skin in the middle.
fn counting_scene(runs: &Arc<AtomicUsize>) -> Arc<SceneModel> {
    let runs = Arc::clone(runs);
    SceneModel::fake(
        "count",
        true,
        Arc::new(move |w, h, _rgb, _| {
            runs.fetch_add(1, Ordering::SeqCst);
            let mut bits = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                    let mut b = if v < 0.33 { SKY } else { GROUND };
                    if (u - 0.5).hypot(v - 0.6) < 0.15 {
                        b |= SUBJECT | SKIN;
                    }
                    bits[y * w + x] = b;
                }
            }
            Ok(bits)
        }),
    )
}

fn renderer(model: Option<Arc<SceneModel>>, tile: usize, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            scene_model: model,
            data_root: Some(data_root()),
            full_tile_px: tile,
            threads,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn photo() -> Developed {
    let scene = Scene::new(
        WIDTH as u32,
        HEIGHT as u32,
        light(PROFILE, 4800.0),
        Layout::Patchwork,
        0.5,
    )
    .noise(2e-4, 2e-6)
    .seed(11);
    Developed::new(&scene, MODEL)
}

/// Main, then a Reference Grade node (id 2) with `body`'s fields.
fn with_node(body: Value) -> DevelopSettings {
    let mut n = json!({ "id": 2, "kind": "reference_grade", "pos": [2, 0.0] });
    n.as_object_mut()
        .expect("an object")
        .extend(body.as_object().expect("an object").clone());
    let doc = json!({ "nodes": { "items": [ { "id": 1 }, n ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn full(r: &Renderer, src: &PhotoSource, threads: usize) -> Result<Vec<f32>, RenderError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    pool.install(|| {
        r.render_full(
            src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .map(|i| i.rgb)
    })
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn fit(r: &Renderer, src: &PhotoSource) -> ym_core::pipeline::RenderedFrame {
    r.viewport_frame(
        src,
        &ViewportState::new(ViewportRect::cell(960.0, 640.0, 1.0)),
        &FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
        &CancelToken::new(),
    )
    .expect("a frame")
}

/// A sky grade that is plainly visible: a turn of the hue and a lift of a stop.
fn sky_grade() -> Value {
    json!({ "elements": { "sky": { "exposure_ev": 0.6, "hue_deg": 25, "chroma": 1.4 } } })
}

/// An export with a live sky grade changes the pixels of the sky and those far from it by nothing: the mask is soft at the edge only.
/// The same bits at tiles of 97, 160 and 256 pixels and on 1, 3 and 4 threads, and on a second run.
#[test]
fn a_sky_grade_changes_the_sky_only_and_the_same_bits_at_any_tile_and_thread_count() {
    let d = photo();
    let runs = Arc::new(AtomicUsize::new(0));
    let plain_src = d.source();
    let r0 = renderer(Some(counting_scene(&runs)), 256, 1);
    let plain = full(&r0, &plain_src, 1).expect("plain");
    assert_eq!(runs.load(Ordering::SeqCst), 0, "no node, no scene");
    let src = d.source().with_settings(with_node(sky_grade()));
    let first = full(&r0, &src, 1).expect("render");
    assert_eq!(runs.load(Ordering::SeqCst), 1, "the export made its scene");
    let (mut sky_changed, mut ground_changed) = (0usize, 0usize);
    let (w, h) = (WIDTH, HEIGHT);
    for y in 0..h {
        for x in 0..w {
            let i = 3 * (y * w + x);
            let moved = first[i..i + 3] != plain[i..i + 3];
            // 0.33 of the height is the edge; 2 % of the short side (8 px) of soft mask, either side, plus the filter's own reach.
            if y < (0.33 * h as f64) as usize - 24 {
                sky_changed += usize::from(moved);
            } else if y > (0.33 * h as f64) as usize + 24 {
                ground_changed += usize::from(moved);
            }
        }
    }
    assert!(sky_changed > w * 80, "the sky is graded: {sky_changed}");
    assert_eq!(ground_changed, 0, "nothing below the sky moves");
    for (tile, threads) in [(97usize, 1usize), (160, 3), (256, 4)] {
        let r = renderer(Some(counting_scene(&runs)), tile, threads);
        let got = full(&r, &src, threads).expect("render");
        assert!(
            bits(&got) == bits(&first),
            "tile {tile} on {threads} threads is other bits"
        );
    }
    assert!(
        bits(&full(&r0, &src, 1).expect("again")) == bits(&first),
        "a second run is other bits"
    );
    // The scene was kept for the photo by the first export: the second render made none.
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1 + 3,
        "one for each renderer, none for a repeat"
    );
}

/// A node with every element at the identity makes no scene, no mask and changes no bit; a node whose only live element is the
/// subject makes the scene once and a slider that moves another node does not make it again.
#[test]
fn only_live_elements_cost_anything_and_an_identity_node_changes_no_bit() {
    let d = photo();
    let runs = Arc::new(AtomicUsize::new(0));
    let r = renderer(Some(counting_scene(&runs)), 256, 2);
    let plain = full(&r, &d.source(), 2).expect("plain");
    for (name, body) in [
        ("default", json!({})),
        (
            "elements spelled at the identity",
            json!({ "elements": { "sky": { "exposure_ev": 0, "hue_deg": 0, "chroma": 1 } } }),
        ),
        ("strength 0", {
            let mut b = sky_grade();
            b["strength"] = json!(0);
            b
        }),
        ("bypassed", {
            let mut b = sky_grade();
            b["bypass"] = json!(true);
            b
        }),
    ] {
        let src = d.source().with_settings(with_node(body));
        let got = full(&r, &src, 2).expect("render");
        assert!(bits(&got) == bits(&plain), "{name}: a bit moved");
        assert_eq!(runs.load(Ordering::SeqCst), 0, "{name}: made a scene");
        assert!(r.unmakeable_mask(&src.settings).is_none(), "{name}");
    }
    // The subject alone.
    let src = d.source().with_settings(with_node(
        json!({ "elements": { "subject": { "exposure_ev": 0.5 } } }),
    ));
    let got = full(&r, &src, 2).expect("render");
    assert!(bits(&got) != bits(&plain));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    // A node's own slider is no new scene: the scene does not depend on a grade.
    let src = d.source().with_settings(with_node(
        json!({ "elements": { "subject": { "exposure_ev": 0.9 } }, "trim": { "tint": 4 } }),
    ));
    let _ = full(&r, &src, 2).expect("render");
    assert_eq!(runs.load(Ordering::SeqCst), 1, "a slider made a scene");
}

/// The viewer never waits: the first frame has no masks and asks the worker, the listener hears it start and end, and the next frame
/// has the masks and equals another renderer's. A failed scene is told in the frame and not asked for again by the viewer.
#[test]
fn a_viewer_never_waits_the_worker_makes_the_scene_and_the_photo_is_drawn_again() {
    let d = photo();
    let runs = Arc::new(AtomicUsize::new(0));
    let r = renderer(Some(counting_scene(&runs)), 256, 2);
    let heard = Arc::new(std::sync::Mutex::new(Vec::new()));
    {
        let heard = Arc::clone(&heard);
        r.set_scene_listener(Arc::new(move |id, phase| {
            heard.lock().unwrap().push((id, phase));
        }));
    }
    let src = d.source().with_settings(with_node(sky_grade()));
    r.navigate(src.id);
    let plain = fit(&r, &d.source());
    let first = fit(&r, &src);
    assert!(
        first.rgba == plain.rgba,
        "no scene yet: the step does nothing"
    );
    wait_until("the worker", 60, || {
        r.scene_ready(&src) && !r.scene_worker_alive(src.id)
    });
    assert!(!r.scene_working(src.id));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(r.scene_workers(src.id), 1);
    let heard = heard.lock().unwrap().clone();
    assert_eq!(heard.len(), 2, "{heard:?}");
    assert_eq!(heard[0].1, ym_core::pipeline::ScenePhase::Started);
    assert_eq!(
        heard[1].1,
        ym_core::pipeline::ScenePhase::Finished { failed: None }
    );
    let second = fit(&r, &src);
    assert!(second.rgba != plain.rgba, "the masks act");
    // Another renderer with its own scene gives the same frame, bit for bit.
    let other = renderer(Some(counting_scene(&runs)), 256, 3);
    other.navigate(src.id);
    let _ = fit(&other, &src);
    wait_until("the other worker", 60, || {
        other.scene_ready(&src) && !other.scene_worker_alive(src.id)
    });
    assert!(fit(&other, &src).rgba == second.rgba);
    // The scene is the photo's, in memory: dropped with the photo.
    assert!(r.scene_ready(&src));
    r.forget(src.id);
    assert!(!r.scene_ready(&src));
}

/// Grading a node, the look or another node's sliders never makes the scene again; the Input's exposure does, as it moves the
/// picture (`docs/23` section 4.7).
#[test]
fn the_scene_is_keyed_by_the_input_alone() {
    let d = photo();
    let runs = Arc::new(AtomicUsize::new(0));
    let r = renderer(Some(counting_scene(&runs)), 256, 2);
    let a = d.source().with_settings(with_node(sky_grade()));
    let key = |s: &PhotoSource| r.scene_key(s).expect("a key").text;
    let k = key(&a);
    // Another strength, another value, a bypassed node: the same key.
    let mut body = sky_grade();
    body["strength"] = json!(40);
    assert_eq!(key(&d.source().with_settings(with_node(body))), k);
    // The Input's exposure moves the picture: another key.
    let mut s = with_node(sky_grade());
    s.input_mut().exposure = ym_core::settings::ExposureSetting::manual_ev(0.7);
    assert_ne!(key(&d.source().with_settings(s)), k);
}

/// A copy of Yellowmoon with no scene model: the viewer draws the photo without the element step and says so; an export is refused
/// with the `mask` error and a plain sentence; a node that does nothing needs no model.
#[test]
fn without_the_scene_model_the_view_says_so_and_the_export_is_refused() {
    let d = photo();
    let r = renderer(None, 256, 2);
    let src = d.source().with_settings(with_node(sky_grade()));
    let plain = fit(&r, &d.source());
    let got = fit(&r, &src);
    assert!(got.rgba == plain.rgba, "the element step is skipped");
    assert!(
        got.report
            .notes
            .iter()
            .any(|n| n == ym_core::pipeline::scene::SCENE_MODEL_MISSING_NOTE),
        "{:?}",
        got.report.notes
    );
    assert_eq!(
        r.unmakeable_mask(&src.settings).as_deref(),
        Some(NO_SCENE_MODEL_SENTENCE)
    );
    match full(&r, &src, 2) {
        Err(RenderError::Mask(sentence)) => assert_eq!(sentence, NO_SCENE_MODEL_SENTENCE),
        other => panic!("expected the mask error, got {:?}", other.map(|v| v.len())),
    }
    // A node at the identity needs no model.
    let idle = d.source().with_settings(with_node(json!({})));
    assert!(r.unmakeable_mask(&idle.settings).is_none());
    assert!(full(&r, &idle, 2).is_ok());
    // A model whose skin network is missing: the sky is fine, a Skin grade is refused.
    let no_skin = SceneModel::fake(
        "no-skin",
        false,
        Arc::new(|w, h, _, _| Ok(vec![SKY; w * h])),
    );
    let r = renderer(Some(no_skin), 256, 2);
    assert!(r.unmakeable_mask(&src.settings).is_none());
    assert!(full(&r, &src, 2).is_ok());
    let skin = d.source().with_settings(with_node(
        json!({ "elements": { "skin": { "chroma": 1.2 } } }),
    ));
    let sentence = r.unmakeable_mask(&skin.settings).expect("refused");
    assert!(sentence.contains("skin model"), "{sentence}");
    assert!(matches!(full(&r, &skin, 2), Err(RenderError::Mask(_))));
}

/// A scene that cannot be made is the `mask` error in an export, with the text of what went wrong, and a note in the viewer.
#[test]
fn a_scene_that_fails_is_an_error_in_an_export_and_a_note_in_the_view() {
    let d = photo();
    let model = SceneModel::fake(
        "fails",
        true,
        Arc::new(|_, _, _, _| Err("the network stopped".to_owned())),
    );
    let r = renderer(Some(model), 256, 2);
    let src = d.source().with_settings(with_node(sky_grade()));
    // The viewer asks once, is told, and does not ask again.
    r.navigate(src.id);
    let _ = fit(&r, &src);
    wait_until("the worker", 60, || !r.scene_worker_alive(src.id));
    let got = fit(&r, &src);
    assert!(
        got.report
            .notes
            .iter()
            .any(|n| n.contains("The Elements can't be made")),
        "{:?}",
        got.report.notes
    );
    // An export asks again itself, and fails with the same plain sentence.
    match full(&r, &src, 2) {
        Err(RenderError::Mask(s)) => {
            assert_eq!(s, "The Elements can't be made: the network stopped.");
        }
        other => panic!("expected the mask error, got {:?}", other.map(|v| v.len())),
    }
    assert_eq!(r.scene_workers(src.id), 1);
}
