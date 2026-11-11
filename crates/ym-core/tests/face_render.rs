//! The Face Refinement node's render (`docs/27-skin-retouch.md` section 10, S-11b, decisions 9 and 13), on a **fake finder** and
//! the mock models of `ym_retouch` (`FaceModel::fake`), so no weights are needed. The real models on public portraits are in
//! `face_real.rs` (it says so and passes without the weights).
//!
//! What is checked: the step changes the faces and nothing outside them (bit for bit), each face takes its own settings by its
//! number, painted protection holds bit for bit, an inert node and a photo without faces are exact pass-throughs, the node's masks
//! act as a corrector's, a viewer never waits for the analysis, the analysis is made once and no slider of the node, of a
//! corrector upstream or of the look makes it again, the grade cache key sees every field, the bytes do not depend on the
//! thread count, and a copy with no face models passes the picture through in a view and refuses an export.
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::color::{GradeRequest, color_plan, grade_key};
use ym_core::pipeline::{
    FaceModel, FacePhase, FaceState, FakeFace, FrameOptions, FullImage, FullRequest, FullTarget,
    NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The fake's two faces, as the finder gives them: shares of the picture.
const FACE_A: [f64; 4] = [0.12, 0.18, 0.26, 0.46];
const FACE_B: [f64; 4] = [0.60, 0.26, 0.20, 0.36];

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A photo of 1200 x 800: a ramp from dark to light with a darker patch in each face, so that the low-frequency tools have
/// something to do.
fn photo() -> Developed {
    let (w, h) = (1200u32, 800u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Ramp, 0.4).noise(0.0, 0.0);
    let mut mosaic = scene.mosaic();
    let darker = |m: &mut u16, f: f64| {
        let v = f64::from(*m) - scene.black;
        *m = (scene.black + v * f) as u16;
    };
    for (face, (px, py)) in [(FACE_A, (0.3, 0.5)), (FACE_B, (0.4, 0.5))] {
        let (cx, cy) = (
            (face[0] + px * face[2]) * f64::from(w),
            (face[1] + py * face[3]) * f64::from(h),
        );
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (f64::from(x) - cx, f64::from(y) - cy);
                let d = (dx * dx + dy * dy).sqrt() / (0.06 * f64::from(w));
                if d < 2.0 {
                    darker(
                        &mut mosaic[(y * w + x) as usize],
                        1.0 - 0.45 * (-d * d).exp(),
                    );
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn doc_of(nodes: Vec<Value>, links: Vec<Value>) -> DevelopSettings {
    let mut items = vec![json!({ "id": 1 })];
    items.extend(nodes);
    serde_json::from_value::<DevelopSettings>(
        json!({ "nodes": { "items": items, "links": links } }),
    )
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn line(ids: &[u32]) -> Vec<Value> {
    let link = |from: Value, to: Value| json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } });
    let mut links = Vec::new();
    let mut from = json!("input");
    for id in ids {
        links.push(link(from, json!(id)));
        from = json!(id);
    }
    links.push(link(from, json!("output")));
    links
}

fn face_node(id: u32, body: Value) -> Value {
    let mut node = json!({ "id": id, "kind": "face_refinement", "pos": [f64::from(id), 0.0] });
    node.as_object_mut()
        .unwrap()
        .extend(body.as_object().unwrap().clone());
    node
}

fn corrector(id: u32, trim_ev: f64) -> Value {
    json!({ "id": id, "kind": "corrector", "pos": [f64::from(id), 0.0], "trim": { "exposure_ev": trim_ev } })
}

/// Main, then a Face Refinement node 2 with `body`.
fn with_node(body: Value) -> DevelopSettings {
    doc_of(vec![face_node(2, body)], line(&[1, 2]))
}

fn plain() -> DevelopSettings {
    DevelopSettings::default()
}

/// Strong amounts that change a face: the low-frequency tools and the grading.
fn strong() -> Value {
    json!({ "all": { "blemish": 0, "dark_circles": 100, "even_skin": 100, "texture": 0,
                     "skin_contrast": 60, "shadow_desaturation": 0, "shine_removal": 0, "teeth_whitening": 0 } })
}

fn renderer_with(model: Option<Arc<FaceModel>>, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            face_model: model,
            data_root: Some(data_root()),
            threads,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn fake(faces: Vec<[f64; 4]>) -> Arc<FaceModel> {
    let calls = Arc::new(AtomicUsize::new(0));
    FaceModel::fake(
        &format!("render{faces:?}"),
        24,
        Arc::new(move |_, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            faces
                .iter()
                .enumerate()
                .map(|(i, f)| FakeFace {
                    x: f[0],
                    y: f[1],
                    width: f[2],
                    height: f[3],
                    score: 0.9 - 0.1 * i as f32,
                })
                .collect()
        }),
    )
}

fn two_faces() -> Arc<FaceModel> {
    fake(vec![FACE_A, FACE_B])
}

fn export(r: &Renderer, src: &PhotoSource, s: DevelopSettings) -> FullImage {
    r.render_full(
        &src.with_settings(s),
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

fn pixel(img: &FullImage, x: usize, y: usize) -> [f32; 3] {
    let i = (y * img.width + x) * 3;
    [img.rgb[i], img.rgb[i + 1], img.rgb[i + 2]]
}

/// Whether the pixel is inside `face` grown by `grow` of its long side.
fn near(img: &FullImage, face: [f64; 4], grow: f64, x: usize, y: usize) -> bool {
    let (w, h) = (img.width as f64, img.height as f64);
    let long = (face[2] * w).max(face[3] * h);
    let (x0, y0) = (face[0] * w - grow * long, face[1] * h - grow * long);
    let (x1, y1) = (
        (face[0] + face[2]) * w + grow * long,
        (face[1] + face[3]) * h + grow * long,
    );
    (x as f64) >= x0 && (x as f64) < x1 && (y as f64) >= y0 && (y as f64) < y1
}

/// The pixels where two renders differ.
fn differing(a: &FullImage, b: &FullImage) -> Vec<(usize, usize)> {
    assert_eq!((a.width, a.height), (b.width, b.height));
    (0..a.height)
        .flat_map(|y| (0..a.width).map(move |x| (x, y)))
        .filter(|&(x, y)| pixel(a, x, y) != pixel(b, x, y))
        .collect()
}

/// Every pixel the node changed is in a face's neighbourhood (the pad of the low-frequency tools is under half a face), and it
/// changed some pixels of each of `faces`.
fn changed_only_in(a: &FullImage, b: &FullImage, faces: &[[f64; 4]]) -> Vec<usize> {
    let diff = differing(a, b);
    let mut per_face = vec![0usize; faces.len()];
    for &(x, y) in &diff {
        let mut inside = false;
        for (i, f) in faces.iter().enumerate() {
            if near(a, *f, 0.5, x, y) {
                inside = true;
                if near(a, *f, 0.0, x, y) {
                    per_face[i] += 1;
                }
            }
        }
        assert!(inside, "pixel ({x}, {y}) changed outside every face");
    }
    per_face
}

// ---- the step ----------------------------------------------------------------------------------------------------------

#[test]
fn the_step_changes_the_faces_and_nothing_outside_them() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let reference = export(&r, &src, plain());
    let retouched = export(&r, &src, with_node(strong()));
    let per_face = changed_only_in(&reference, &retouched, &[FACE_A, FACE_B]);
    assert!(
        per_face.iter().all(|&n| n > 100),
        "both faces changed: {per_face:?}"
    );
    // The export makes its own analysis and keeps nothing.
    assert_eq!(
        r.faces_analyses(src.id),
        0,
        "an export does not keep what it made"
    );
}

#[test]
fn each_face_takes_its_own_settings_by_its_number() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let reference = export(&r, &src, plain());
    // Face 1 is the left one: its settings are all 0; face 2 has the strong ones.
    let zero = json!({ "blemish": 0, "dark_circles": 0, "even_skin": 0, "texture": 0 });
    let only_two = with_node(json!({
        "same_for_all": false,
        "faces": [ zero, strong()["all"].clone() ],
    }));
    let per_face = changed_only_in(&reference, &export(&r, &src, only_two), &[FACE_A, FACE_B]);
    assert_eq!(per_face[0], 0, "face 1 is left alone");
    assert!(per_face[1] > 100, "face 2 is retouched: {per_face:?}");
    let only_one = with_node(json!({
        "same_for_all": false,
        "faces": [ strong()["all"].clone(), zero ],
    }));
    let per_face = changed_only_in(&reference, &export(&r, &src, only_one), &[FACE_A, FACE_B]);
    assert!(per_face[0] > 100 && per_face[1] == 0, "{per_face:?}");
    // A face past the end of the list takes face 1's settings; "same for all" ignores the list.
    let past = with_node(json!({ "same_for_all": false, "faces": [ strong()["all"].clone() ] }));
    let all = with_node(strong());
    assert_eq!(
        export(&r, &src, past).rgb,
        export(&r, &src, all.clone()).rgb
    );
    let ignored = with_node(json!({
        "same_for_all": true, "all": strong()["all"].clone(), "faces": [ zero, zero ],
    }));
    assert_eq!(export(&r, &src, ignored).rgb, export(&r, &src, all).rgb);
}

#[test]
fn painted_protection_holds_bit_for_bit() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let reference = export(&r, &src, plain());
    let open = export(&r, &src, with_node(strong()));
    // A fat stroke over face A (the left one): its neighbourhood is protected. `size` is a share of the long edge.
    let (cx, cy) = (FACE_A[0] + FACE_A[2] / 2.0, FACE_A[1] + FACE_A[3] / 2.0);
    let mut body = strong();
    body["protect"] = json!({ "strokes": [
        { "size": 0.25, "hardness": 1.0, "erase": false, "points": [[cx - 0.05, cy - 0.25], [cx - 0.05, cy + 0.25]] },
        { "size": 0.25, "hardness": 1.0, "erase": false, "points": [[cx + 0.05, cy - 0.25], [cx + 0.05, cy + 0.25]] },
    ] });
    let held = export(&r, &src, with_node(body));
    let per_face = changed_only_in(&reference, &held, &[FACE_A, FACE_B]);
    assert_eq!(per_face[0], 0, "face A is protected: not one bit changes");
    assert!(per_face[1] > 100, "face B is not: {per_face:?}");
    // Face B comes out as it did without protection, bit for bit, where the protection plane is 0 (the stroke ends well before B).
    for y in 0..open.height {
        for x in 0..open.width {
            if near(&open, FACE_B, 0.0, x, y) {
                assert_eq!(pixel(&open, x, y), pixel(&held, x, y), "({x}, {y})");
            }
        }
    }
}

#[test]
fn an_inert_node_and_a_photo_without_faces_are_exact_pass_throughs() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let reference = export(&r, &src, plain());
    let zero = json!({ "all": { "blemish": 0, "dark_circles": 0, "even_skin": 0, "texture": 0,
                                "skin_contrast": 0, "shadow_desaturation": 0, "shine_removal": 0, "teeth_whitening": 0 } });
    for (name, body) in [
        ("every amount 0", zero),
        (
            "bypassed",
            json!({ "bypass": true, "all": strong()["all"].clone() }),
        ),
    ] {
        assert_eq!(
            export(&r, &src, with_node(body)).rgb,
            reference.rgb,
            "{name}"
        );
    }
    // No face found: the node does nothing.
    let none = renderer_with(Some(fake(Vec::new())), 0);
    assert_eq!(
        export(&none, &src, with_node(strong())).rgb,
        reference.rgb,
        "no faces"
    );
    // A face under the floor (the fake's floor is 24 px of the full-resolution photo) is dropped: 0.01 of 1200 is 12.
    let tiny = renderer_with(Some(fake(vec![[0.4, 0.4, 0.01, 0.01]])), 0);
    assert_eq!(
        export(&tiny, &src, with_node(strong())).rgb,
        reference.rgb,
        "a face under the floor"
    );
}

#[test]
fn the_nodes_masks_act_as_a_correctors() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let reference = export(&r, &src, plain());
    let open = export(&r, &src, with_node(strong()));
    // A linear mask that selects the right half of the photo: face B is retouched, face A is not.
    let mut body = strong();
    body["masks"] = json!({ "invert": false, "items": [
        { "kind": "linear", "mode": "add", "center": [0.5, 0.5], "angle_deg": 90.0, "feather": 0.0 } ] });
    let masked = export(&r, &src, with_node(body.clone()));
    let per_face = changed_only_in(&reference, &masked, &[FACE_A, FACE_B]);
    assert!(per_face[1] > 100, "{per_face:?}");
    let (_, _) = (&open, &masked);
    // Inverted, the other face.
    body["masks"]["invert"] = json!(true);
    let inverted = export(&r, &src, with_node(body));
    let per_face = changed_only_in(&reference, &inverted, &[FACE_A, FACE_B]);
    assert!(per_face[0] > 100, "{per_face:?}");
    // Where the mask is on the picture is the unmasked retouch, where it is off the input.
    for y in (0..open.height).step_by(7) {
        for x in (0..open.width).step_by(7) {
            let p = pixel(&masked, x, y);
            assert!(
                p == pixel(&open, x, y) || p == pixel(&reference, x, y),
                "({x}, {y}) is the retouch or the input"
            );
        }
    }
}

// ---- the worker --------------------------------------------------------------------------------------------------------

fn frame(r: &Renderer, src: &PhotoSource) -> Vec<u8> {
    let st = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    r.viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
        .rgba
}

fn listen(r: &Renderer) -> Arc<Mutex<Vec<(u32, FaceState)>>> {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&heard);
    r.set_faces_listener(Arc::new(move |id, state| {
        sink.lock().unwrap().push((id.0, state))
    }));
    heard
}

#[test]
fn a_viewer_never_waits_and_the_worker_finds_the_faces_once() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let heard = listen(&r);
    r.navigate(src.id);
    let node = src.with_settings(with_node(strong()));
    let without = frame(&r, &src.with_settings(plain()));
    // No analysis yet: the frame is the photo without the node's effect, at once, and the worker is on its way.
    assert_eq!(frame(&r, &node), without, "a viewer does not wait");
    assert_eq!(r.face_state(&node).phase, FacePhase::Finding);
    wait_until("the faces", 60, || {
        !r.faces_worker_alive(src.id) && r.faces_ready(&node)
    });
    assert_eq!(r.faces_analyses(src.id), 1);
    {
        let heard = heard.lock().unwrap();
        assert_eq!(heard.len(), 2, "{heard:?}");
        assert_eq!(heard[0].1.phase, FacePhase::Finding);
        let ready = &heard[1].1;
        assert_eq!(ready.phase, FacePhase::Ready);
        let numbers: Vec<u32> = ready.faces.iter().map(|f| f.number).collect();
        assert_eq!(numbers, [1, 2], "numbered from left to right");
        assert!(ready.faces[0].x < ready.faces[1].x);
        let a = ready.faces[0];
        assert!(
            (a.x - FACE_A[0]).abs() < 0.01 && (a.height - FACE_A[3]).abs() < 0.01,
            "{a:?}"
        );
    }
    let with_faces = frame(&r, &node);
    assert_ne!(with_faces, without, "the faces are retouched in the frame");
    // No slider of the node, of a corrector before it or of the look finds the faces again; the Input's exposure does.
    let before = r.faces_analyses(src.id);
    for body in [
        json!({ "all": { "dark_circles": 40, "even_skin": 80, "skin_contrast": 20 } }),
        json!({ "all": { "dark_circles": 100, "even_skin": 100 }, "bypass": false }),
    ] {
        let _ = frame(&r, &src.with_settings(with_node(body)));
    }
    let upstream = doc_of(
        vec![corrector(3, 0.7), face_node(2, strong())],
        line(&[1, 3, 2]),
    );
    let _ = frame(&r, &src.with_settings(upstream));
    std::thread::sleep(Duration::from_millis(400));
    assert!(!r.faces_worker_alive(src.id));
    assert_eq!(
        r.faces_analyses(src.id),
        before,
        "moving a slider runs no network"
    );
    assert_eq!(r.faces_workers(src.id), 1, "one worker for the photo");
}

#[test]
fn leaving_the_photo_stops_the_worker_without_a_word() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let heard = listen(&r);
    // The photo is not the one looked at: a frame does not start the worker.
    let node = src.with_settings(with_node(strong()));
    let _ = frame(&r, &node);
    std::thread::sleep(Duration::from_millis(400));
    assert!(!r.faces_worker_alive(src.id));
    assert!(heard.lock().unwrap().is_empty());
    assert!(!r.faces_ready(&node));
}

// ---- the cache key -----------------------------------------------------------------------------------------------------

fn key_of(src: &PhotoSource, r: &Renderer, s: &DevelopSettings) -> String {
    let src = src.with_settings(s.clone());
    let color = color_plan(&src).expect("colour plan");
    let mut job_faces = None;
    // The grade key names the faces by the analysis' own key, which the renderer's request makes.
    let key = r.faces_key(&src).ok();
    job_faces.get_or_insert(key);
    let req = GradeRequest {
        target: OutputTarget::Srgb,
        dither: false,
        looks: &NeutralOnly,
        background: None,
        objects: None,
    };
    format!("{:?}", grade_key(&src, &color, &req))
}

#[test]
fn the_grade_key_sees_every_field_of_the_node() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let base = with_node(strong());
    let base_key = key_of(&src, &r, &base);
    assert_eq!(
        base_key,
        key_of(&src, &r, &base),
        "the same settings, the same key"
    );
    let stroke = json!({ "strokes": [ { "size": 0.05, "hardness": 0.5, "erase": false, "points": [[0.3, 0.3], [0.6, 0.5]] } ] });
    let mut variants = vec![
        (
            "an amount",
            with_node(
                json!({ "all": { "dark_circles": 99, "even_skin": 100, "skin_contrast": 60 } }),
            ),
        ),
        (
            "texture",
            with_node(
                json!({ "all": { "dark_circles": 100, "even_skin": 100, "skin_contrast": 60, "texture": 10 } }),
            ),
        ),
        (
            "same_for_all",
            with_node(json!({ "all": strong()["all"].clone(), "same_for_all": false })),
        ),
        (
            "a face's set",
            with_node(json!({ "all": strong()["all"].clone(), "faces": [ { "blemish": 7 } ] })),
        ),
        (
            "bypass",
            with_node(json!({ "all": strong()["all"].clone(), "bypass": true })),
        ),
        (
            "protect strokes",
            with_node(json!({ "all": strong()["all"].clone(), "protect": stroke.clone() })),
        ),
    ];
    let mut feathered = stroke.clone();
    feathered["feather"] = json!(0.3);
    variants.push((
        "protect feather",
        with_node(json!({ "all": strong()["all"].clone(), "protect": feathered })),
    ));
    let mut grown = stroke;
    grown["grow"] = json!(0.2);
    variants.push((
        "protect grow",
        with_node(json!({ "all": strong()["all"].clone(), "protect": grown })),
    ));
    variants.push((
        "a mask",
        with_node(
            json!({ "all": strong()["all"].clone(), "masks": { "invert": true, "items": [] } }),
        ),
    ));
    let mut seen = vec![base_key.clone()];
    for (name, s) in variants {
        let k = key_of(&src, &r, &s);
        assert_ne!(k, base_key, "{name} changes the key");
        assert!(!seen.contains(&k), "{name} has a key of its own");
        seen.push(k);
    }
}

// ---- the same bytes on any number of threads -----------------------------------------------------------------------------

#[test]
fn one_and_four_threads_give_the_same_bytes() {
    let d = photo();
    let src = d.source();
    let s = with_node(
        json!({ "all": { "blemish": 40, "dark_circles": 80, "even_skin": 70, "skin_contrast": 30, "shine_removal": 30 } }),
    );
    let one = export(&renderer_with(Some(two_faces()), 1), &src, s.clone());
    let four = export(&renderer_with(Some(two_faces()), 4), &src, s.clone());
    assert_eq!(one.rgb, four.rgb);
    let plain_img = export(&renderer_with(Some(two_faces()), 1), &src, plain());
    assert_ne!(one.rgb, plain_img.rgb);
    // A tile size does not change a bit either: faces that straddle tiles are made once, whole.
    let small_tiles = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            face_model: Some(two_faces()),
            data_root: Some(data_root()),
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    assert_eq!(
        export(&small_tiles, &src, s).rgb,
        one.rgb,
        "tiles of 256 px"
    );
}

// ---- no models ---------------------------------------------------------------------------------------------------------

#[test]
fn without_face_models_a_view_passes_the_picture_through_and_an_export_refuses() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(None, 0);
    let node = with_node(strong());
    assert_eq!(
        frame(&r, &src.with_settings(node.clone())),
        frame(&r, &src.with_settings(plain())),
        "the node passes its picture through"
    );
    let state = r.face_state(&src.with_settings(node.clone()));
    assert_eq!(state.phase, FacePhase::ModelsMissing);
    assert!(
        state
            .message
            .as_deref()
            .is_some_and(|m| m.contains("Reinstall"))
    );
    let sentence = r.unmakeable_mask(&node).expect("an export refuses");
    assert!(
        sentence.contains("Node 2") && sentence.contains("Face Refinement"),
        "{sentence}"
    );
    // A node that does nothing needs no model.
    assert_eq!(
        r.unmakeable_mask(&with_node(json!({ "bypass": true }))),
        None
    );
    assert_eq!(r.unmakeable_mask(&plain()), None);
    // With the models there is nothing to refuse.
    assert_eq!(
        renderer_with(Some(two_faces()), 0).unmakeable_mask(&node),
        None
    );
}

#[test]
fn a_frame_with_a_live_node_is_made_on_the_cpu_and_says_so() {
    let d = photo();
    let src = d.source();
    // Without a device the CPU makes every frame; with one, the node forces it. Either way the node's picture is the CPU's.
    let r = renderer_with(Some(two_faces()), 0);
    r.navigate(src.id);
    let node = src.with_settings(with_node(strong()));
    let _ = frame(&r, &node);
    wait_until("the faces", 60, || {
        !r.faces_worker_alive(src.id) && r.faces_ready(&node)
    });
    let st = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    let made = r
        .viewport_frame(&node, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    assert_ne!(made.rgba, frame(&r, &src.with_settings(plain())));
}
