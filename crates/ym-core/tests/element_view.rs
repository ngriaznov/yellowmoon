//! The element eye of Reference Grade (`docs/34-reference-grade.md` section 9.6, RG-12b): the mask view of the viewport shows one
//! element's soft mask of a Reference Grade node, the mask the node grades with, at the viewport's size.
//!
//! * **Mask only** shows the element's mask as grey (the code value after C13 is `m`), **Overlay** tints the photo with it.
//! * The elements are numbered as the node's settings number them (0 sky ... 5 skin, 6 subject).
//! * It is a view and nothing else: no setting moves, an export is the export it was, the Before view and the pixel readout
//!   never show it, and the same frame comes at any thread count.
//! * What cannot be shown leaves the photo as it is: an element past the end, a node that is not a Reference Grade node, a photo
//!   whose scene is not made yet and a computer with no scene model.
//!
//! The CPU reference of the plane is `ym-color`'s `tests/element_view.rs` (exact); the mask's own tiling, threads and the guided
//! filter are `tests/element_masks.rs` and `ym-color`'s tests. These tests run on a **fake scene model**, so they need no weights.
mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly, PhotoSource,
    RenderOptions, RenderedFrame, Renderer, SceneModel, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;
use ym_ml::scene::{GROUND, SKIN, SKY, SUBJECT};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The paper: nothing in the scene or in a mask is this colour.
const PAPER: [u8; 3] = [40, 60, 90];
const CELL: f64 = 300.0;
/// The photo's rows in the square cell (3:2, centred).
const ROWS: std::ops::Range<usize> = 50..250;

/// A fake scene: sky in the top third, ground below, a subject disc with skin in the middle.
fn scene_model() -> Arc<SceneModel> {
    SceneModel::fake(
        "view",
        true,
        Arc::new(move |w, h, _rgb, _| {
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

fn renderer(model: Option<Arc<SceneModel>>, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            scene_model: model,
            data_root: Some(data_root()),
            background: PAPER,
            threads,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn photo() -> Developed {
    let scene = Scene::new(480, 320, light(PROFILE, 4800.0), Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(11);
    Developed::new(&scene, MODEL)
}

/// Main, a corrector (id 3) and a Reference Grade node (id 2) with `body`'s fields, in a line: Main, 2, 3.
fn with_node(body: Value) -> DevelopSettings {
    let mut n = json!({ "id": 2, "kind": "reference_grade", "pos": [2, 0.0] });
    n.as_object_mut()
        .expect("an object")
        .extend(body.as_object().expect("an object").clone());
    let doc = json!({ "nodes": { "items": [
        { "id": 1 }, n, { "id": 3, "kind": "corrector", "pos": [3, 0.0] } ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// A node whose elements are all at the identity.
fn identity_node() -> DevelopSettings {
    with_node(json!({}))
}

fn sky_grade() -> Value {
    json!({ "elements": { "sky": { "exposure_ev": 1.0, "hue_deg": 25, "chroma": 1.4 } } })
}

fn view(style: Option<MaskStyle>, node: u32, element: Option<u32>) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = style.map(|style| MaskView {
        node,
        style,
        row: None,
        depth: false,
        face: None,
        element,
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

fn px(f: &RenderedFrame, x: usize, y: usize) -> [u8; 3] {
    let i = (y * f.width + x) * 4;
    [f.rgba[i], f.rgba[i + 1], f.rgba[i + 2]]
}

/// Makes the photo's scene: asks for it with a view, waits for the worker.
fn ready(r: &Renderer, src: &PhotoSource) {
    r.navigate(src.id);
    let _ = frame(r, src, &view(Some(MaskStyle::MaskOnly), 2, Some(0)));
    let end = Instant::now() + Duration::from_secs(60);
    while !(r.scene_ready(src) && !r.scene_worker_alive(src.id)) {
        assert!(Instant::now() < end, "timed out waiting for the scene");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The mask of element `e` at the centre of the top of the sky, of the ground and of the subject: the cell is 300 square and
/// the 3:2 photo lies in rows 50 to 250, so the sky is rows 50 to 116, the subject disc is centred at row 50 + 0.6 * 200 = 170.
const SKY_AT: (usize, usize) = (150, 70);
const GROUND_AT: (usize, usize) = (40, 235);
const SUBJECT_AT: (usize, usize) = (150, 170);

#[test]
fn an_old_mask_view_without_element_still_parses() {
    let old = r#"{"node":2,"style":"overlay","row":null,"depth":false,"face":null}"#;
    let v: MaskView = serde_json::from_str(old).expect("an old view");
    assert_eq!(v.element, None);
    let older = r#"{"node":2,"style":"mask_only"}"#;
    let v: MaskView = serde_json::from_str(older).expect("an older view");
    assert_eq!((v.node, v.element), (2, None));
    let new = r#"{"node":2,"style":"mask_only","element":5}"#;
    let v: MaskView = serde_json::from_str(new).expect("a new view");
    assert_eq!(v.element, Some(5));
}

#[test]
fn mask_only_draws_the_numbered_elements_mask_as_grey() {
    let d = photo();
    let src = d.source().with_settings(identity_node());
    let r = renderer(Some(scene_model()), 2);
    ready(&r, &src);
    let letterbox =
        |f: &RenderedFrame| (0..f.width).all(|x| px(f, x, 10) == PAPER && px(f, x, 290) == PAPER);
    // Sky is element 0: white in the sky, black on the ground.
    let sky = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(0)));
    assert!(letterbox(&sky), "the letterbox is the paper");
    assert_eq!(px(&sky, SKY_AT.0, SKY_AT.1), [255; 3]);
    assert_eq!(px(&sky, GROUND_AT.0, GROUND_AT.1), [0; 3]);
    assert_eq!(px(&sky, SUBJECT_AT.0, SUBJECT_AT.1), [0; 3]);
    // The edge is soft: some pixel in the sky's rows is neither black nor white.
    let soft = (0..300).any(|y| {
        let p = px(&sky, 40, y);
        ROWS.contains(&y) && p[0] != 0 && p[0] != 255 && p[0] != PAPER[0]
    });
    assert!(soft, "the mask is soft at its edge");
    // Ground is element 3, the subject 6 and skin 5; vegetation (1) is empty in this scene.
    let ground = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(3)));
    assert_eq!(px(&ground, GROUND_AT.0, GROUND_AT.1), [255; 3]);
    assert_eq!(px(&ground, SKY_AT.0, SKY_AT.1), [0; 3]);
    let subject = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(6)));
    assert_eq!(px(&subject, SUBJECT_AT.0, SUBJECT_AT.1), [255; 3]);
    assert_eq!(px(&subject, GROUND_AT.0, GROUND_AT.1), [0; 3]);
    let skin = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(5)));
    assert_eq!(px(&skin, SUBJECT_AT.0, SUBJECT_AT.1), [255; 3]);
    let vegetation = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(1)));
    assert!(
        ROWS.clone()
            .all(|y| (0..300).all(|x| px(&vegetation, x, y) == [0; 3])),
        "an element the photo has none of is black"
    );
    // Grey: the three channels agree wherever the photo is.
    assert!(ROWS.clone().all(|y| (0..300).all(|x| {
        let p = px(&sky, x, y);
        p[0] == p[1] && p[1] == p[2]
    })));
}

/// Output sharpening (on by default) makes a grade's edge ring a little beyond its mask: a pixel there moves by a few codes with
/// the mask at 0. A move above this many codes is the grade's.
const SHARPENING_HALO: u8 = 6;

/// The view's mask is the mask the node grades with: where the graded frame moved against the same photo with the node at the
/// identity, the view has a mask; where the view is 0, nothing moved; and the view is the same whatever the node's grade is.
#[test]
fn it_is_the_mask_the_node_grades_with() {
    let d = photo();
    let r = renderer(Some(scene_model()), 2);
    let graded = d.source().with_settings(with_node(sky_grade()));
    let plain = d.source().with_settings(identity_node());
    ready(&r, &graded);
    let st = view(None, 2, None);
    let after = frame(&r, &graded, &st);
    let before = frame(&r, &plain, &st);
    let mask = frame(&r, &graded, &view(Some(MaskStyle::MaskOnly), 2, Some(0)));
    let (mut moved_inside, mut inside, mut moved_outside) = (0usize, 0usize, 0usize);
    for y in ROWS.clone() {
        for x in 0..300 {
            let m = px(&mask, x, y)[0];
            let moved = px(&after, x, y)
                .iter()
                .zip(px(&before, x, y))
                .any(|(a, b)| a.abs_diff(b) > SHARPENING_HALO);
            if m == 0 {
                moved_outside += usize::from(moved);
            } else if m == 255 {
                inside += 1;
                moved_inside += usize::from(moved);
            }
        }
    }
    assert!(inside > 300 * 30, "the sky is in view: {inside}");
    assert_eq!(moved_outside, 0, "nothing moves where the mask is 0");
    assert!(
        moved_inside * 10 > inside * 9,
        "the grade acts where the mask is 1: {moved_inside} of {inside}"
    );
    // The mask does not depend on the grade: the node with the sky at the identity, bypassed, or at Strength 0 shows the same.
    let mut bypassed = sky_grade();
    bypassed["bypass"] = json!(true);
    let mut weak = sky_grade();
    weak["strength"] = json!(0);
    for body in [json!({}), bypassed, weak] {
        let other = d.source().with_settings(with_node(body.clone()));
        let got = frame(&r, &other, &view(Some(MaskStyle::MaskOnly), 2, Some(0)));
        assert_eq!(got.rgba, mask.rgba, "{body}");
    }
}

#[test]
fn overlay_tints_the_photo_where_the_mask_is() {
    let d = photo();
    let src = d.source().with_settings(identity_node());
    let r = renderer(Some(scene_model()), 2);
    ready(&r, &src);
    let plain = frame(&r, &src, &view(None, 2, None));
    let mask = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 2, Some(0)));
    let tinted = frame(&r, &src, &view(Some(MaskStyle::Overlay), 2, Some(0)));
    let mut changed = 0usize;
    for y in 0..300 {
        for x in 0..300 {
            let m = px(&mask, x, y);
            if m == PAPER {
                assert_eq!(px(&tinted, x, y), px(&plain, x, y), "letterbox ({x},{y})");
            } else if m[0] == 0 {
                assert_eq!(px(&tinted, x, y), px(&plain, x, y), "outside ({x},{y})");
            } else if px(&tinted, x, y) != px(&plain, x, y) {
                changed += 1;
            }
        }
    }
    assert!(changed > 300 * 40, "the sky is tinted: {changed}");
    // The tint is the mask accent of the other mask views (`#ff6fd2`): the blue and red rise, the green falls.
    let (a, b) = (
        px(&tinted, SKY_AT.0, SKY_AT.1),
        px(&plain, SKY_AT.0, SKY_AT.1),
    );
    assert!(a[0] > b[0] && a[2] > b[2], "{a:?} against {b:?}");
}

#[test]
fn what_cannot_be_shown_leaves_the_photo() {
    let d = photo();
    let src = d.source().with_settings(identity_node());
    let r = renderer(Some(scene_model()), 2);
    // The scene is not there yet: the photo as it is, and the worker is asked.
    r.navigate(src.id);
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        let plain = frame(&r, &src, &view(None, 2, None));
        let first = frame(&r, &src, &view(Some(style), 2, Some(0)));
        assert_eq!(first.rgba, plain.rgba, "{style:?}: no scene yet");
    }
    ready(&r, &src);
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        let plain = frame(&r, &src, &view(None, 2, None));
        assert_ne!(
            frame(&r, &src, &view(Some(style), 2, Some(0))).rgba,
            plain.rgba,
            "{style:?}: with the scene it draws"
        );
        // An element past the end, a node that is not a Reference Grade node, a node the photo has not.
        for (node, element) in [(2, 7), (2, 99), (3, 0), (1, 0), (9, 0)] {
            assert_eq!(
                frame(&r, &src, &view(Some(style), node, Some(element))).rgba,
                plain.rgba,
                "{style:?}: node {node} element {element}"
            );
        }
    }
    // A computer with no scene model.
    let bare = renderer(None, 2);
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        let plain = frame(&bare, &src, &view(None, 2, None));
        assert_eq!(
            frame(&bare, &src, &view(Some(style), 2, Some(0))).rgba,
            plain.rgba,
            "{style:?}: no model"
        );
    }
}

#[test]
fn it_is_a_view_and_nothing_else() {
    let d = photo();
    let src = d.source().with_settings(with_node(sky_grade()));
    let settings = src.settings.clone();
    let r = renderer(Some(scene_model()), 2);
    ready(&r, &src);
    let export = |r: &Renderer| {
        r.render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("render")
        .rgb
    };
    let exported = export(&r);
    let plain = frame(&r, &src, &view(None, 2, None));
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        let _ = frame(&r, &src, &view(Some(style), 2, Some(0)));
        // Before: with or without the view, the same frame.
        let mut with_mask = view(Some(style), 2, Some(0));
        with_mask.before = true;
        let mut without = view(None, 2, None);
        without.before = true;
        assert_eq!(
            frame(&r, &src, &with_mask).rgba,
            frame(&r, &src, &without).rgba,
            "{style:?}: the Before view"
        );
        // The readout reads the photo.
        for css in [(150.0, 80.0), (60.0, 200.0)] {
            let a = r
                .sample_pixel(
                    &src,
                    &view(Some(style), 2, Some(0)),
                    css,
                    &CancelToken::new(),
                )
                .expect("a reading");
            let b = r
                .sample_pixel(&src, &view(None, 2, None), css, &CancelToken::new())
                .expect("a reading");
            assert_eq!(a, b, "{style:?}: the readout at {css:?}");
        }
    }
    assert_eq!(src.settings, settings, "no setting moved");
    assert_eq!(export(&r), exported, "the export is the export it was");
    // The cached frames of the photo are its own: the next plain frame is the one before.
    let cached = r
        .viewport_frame(
            &src,
            &view(None, 2, None),
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame");
    assert_eq!(cached.rgba, plain.rgba);
    let thumb = r.thumbnail_png(&src, &CancelToken::new()).expect("thumb");
    let _ = frame(&r, &src, &view(Some(MaskStyle::Overlay), 2, Some(0)));
    assert_eq!(
        r.thumbnail_png(&src, &CancelToken::new()).expect("thumb"),
        thumb,
        "thumbnails never show it"
    );
}

#[test]
fn the_same_frame_at_any_thread_count_and_on_a_second_run() {
    let d = photo();
    let src = d.source().with_settings(identity_node());
    let one = renderer(Some(scene_model()), 1);
    let four = renderer(Some(scene_model()), 4);
    ready(&one, &src);
    ready(&four, &src);
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        for e in [0u32, 3, 5, 6] {
            let a = frame(&one, &src, &view(Some(style), 2, Some(e)));
            let b = frame(&four, &src, &view(Some(style), 2, Some(e)));
            assert_eq!(a.rgba, b.rgba, "{style:?} element {e}: 1 and 4 threads");
            let again = frame(&one, &src, &view(Some(style), 2, Some(e)));
            assert_eq!(a.rgba, again.rgba, "{style:?} element {e}: a second run");
        }
    }
    // The cache keeps the elements apart.
    let cached = |e: u32| {
        one.viewport_frame(
            &src,
            &view(Some(MaskStyle::MaskOnly), 2, Some(e)),
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame")
        .rgba
    };
    let (sky, ground) = (cached(0), cached(3));
    assert_ne!(sky, ground);
    assert_eq!(cached(0), sky);
}
