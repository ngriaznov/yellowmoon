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
    FaceEvent, FaceModel, FacePhase, FakeFace, FrameOptions, FullImage, FullRequest, FullTarget,
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
    photo_sized(1200, 800)
}

/// [`photo`] at another size (the faces are shares of it).
fn photo_sized(w: u32, h: u32) -> Developed {
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

/// What a Layer fed by the node's picture gets (`docs/27` decision 13): what it gets from a corrector, the node's picture before its
/// own mask blend with its `m` as the alpha. A Layer at Normal and 100 % over the node's input, with the node's picture above it, is the
/// node alone, with or without a mask on the node.
#[test]
fn a_layer_fed_by_the_nodes_picture_gets_what_it_gets_from_a_corrector() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    let link = |from: Value, port: &str, to: u32, to_port: &str| json!({ "from": { "node": from, "port": port }, "to": { "node": to, "port": to_port } });
    let layered = |node: Value| {
        doc_of(
            vec![
                node,
                json!({ "id": 3, "kind": "layer", "pos": [3.0, 0.0], "mode": "normal", "opacity": 1.0 }),
            ],
            vec![
                link(json!("input"), "picture", 1, "picture"),
                link(json!(1), "picture", 2, "picture"),
                link(json!(1), "picture", 3, "picture.1"),
                link(json!(2), "picture", 3, "picture.2"),
                json!({ "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } }),
            ],
        )
    };
    let mut half = strong();
    half["masks"] = json!({ "invert": false, "items": [
        { "kind": "linear", "mode": "add", "center": [0.5, 0.5], "angle_deg": 90.0, "feather": 0.0 } ] });
    for (name, body) in [("no mask", strong()), ("a mask on the node", half)] {
        let alone = export(&r, &src, with_node(body.clone()));
        let through = export(&r, &src, layered(face_node(2, body)));
        let worst = alone
            .rgb
            .iter()
            .zip(&through.rgb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "{name}: the Layer gives the node's picture (worst {worst})"
        );
        let reference = export(&r, &src, plain());
        assert!(
            !differing(&reference, &through).is_empty(),
            "{name}: the faces are in it"
        );
    }
}

// ---- the worker --------------------------------------------------------------------------------------------------------

fn frame(r: &Renderer, src: &PhotoSource) -> Vec<u8> {
    let st = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    r.viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
        .rgba
}

fn listen(r: &Renderer) -> Arc<Mutex<Vec<(u32, FaceEvent)>>> {
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
        assert_eq!(heard[0].1, FaceEvent::Started);
        let FaceEvent::Finished(ready) = &heard[1].1 else {
            panic!("{heard:?}");
        };
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
fn a_frame_shows_the_retouch_once_the_faces_are_ready() {
    let d = photo();
    let src = d.source();
    // (That a device renderer makes this frame on the CPU, refused with its reason, is `any_graph.rs`'s, where a device opens.)
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

// ---- a Face Refinement node among Film nodes and other nodes (S-11b2) ---------------------------------------------------------

/// A Film node with a halation wide enough to cross a face region from outside it.
fn film_node(id: u32) -> Value {
    json!({ "id": id, "kind": "film", "pos": [f64::from(id), 0.0],
            "film": { "exposure_ev": 0.3 },
            "halation": { "amount": 200, "size": 200 },
            "strength": 90 })
}

fn link(from: Value, port: &str, to: u32, to_port: &str) -> Value {
    json!({ "from": { "node": from, "port": port }, "to": { "node": to, "port": to_port } })
}

/// A weaker Face Refinement body than [`strong`], so that two nodes in a line do different things.
fn gentle() -> Value {
    json!({ "all": { "blemish": 0, "dark_circles": 60, "even_skin": 0, "texture": 0,
                     "skin_contrast": 0, "shadow_desaturation": 0, "shine_removal": 50, "teeth_whitening": 0 } })
}

/// Every graph order the pass has to get right: the Face Refinement node before and after a Film node, two Face Refinement
/// nodes (with a Film node between and after), and the node on a branch of a Parallel and under a Layer.
fn orders() -> Vec<(&'static str, DevelopSettings)> {
    let to_output = |from: u32| json!({ "from": { "node": from, "port": "picture" }, "to": { "node": "output", "port": "picture" } });
    let parallel = doc_of(
        vec![
            face_node(2, strong()),
            film_node(3),
            json!({ "id": 4, "kind": "parallel", "pos": [4.0, 0.0] }),
        ],
        vec![
            link(json!("input"), "picture", 1, "picture"),
            link(json!(1), "picture", 2, "picture"),
            link(json!(1), "picture", 3, "picture"),
            link(json!(2), "picture", 4, "picture.1"),
            link(json!(3), "picture", 4, "picture.2"),
            to_output(4),
        ],
    );
    let layer = doc_of(
        vec![
            face_node(2, strong()),
            film_node(3),
            json!({ "id": 4, "kind": "layer", "pos": [4.0, 0.0], "mode": "normal", "opacity": 0.8 }),
        ],
        vec![
            link(json!("input"), "picture", 1, "picture"),
            link(json!(1), "picture", 2, "picture"),
            link(json!(1), "picture", 3, "picture"),
            link(json!(3), "picture", 4, "picture.1"),
            link(json!(2), "picture", 4, "picture.2"),
            to_output(4),
        ],
    );
    vec![
        (
            "Face then Film",
            doc_of(vec![face_node(2, strong()), film_node(3)], line(&[1, 2, 3])),
        ),
        (
            // The halos add: the reach of the two Film nodes is wider than the pad of a face's region.
            "Face then two Films",
            doc_of(
                vec![face_node(2, strong()), film_node(3), film_node(4)],
                line(&[1, 2, 3, 4]),
            ),
        ),
        (
            "Film then Face",
            doc_of(vec![film_node(2), face_node(3, strong())], line(&[1, 2, 3])),
        ),
        (
            "Film, Face, Film",
            doc_of(
                vec![film_node(2), face_node(3, strong()), film_node(4)],
                line(&[1, 2, 3, 4]),
            ),
        ),
        (
            "two Face nodes",
            doc_of(
                vec![face_node(2, strong()), face_node(3, gentle())],
                line(&[1, 2, 3]),
            ),
        ),
        (
            "Face, Film, Face",
            doc_of(
                vec![face_node(2, strong()), film_node(3), face_node(4, gentle())],
                line(&[1, 2, 3, 4]),
            ),
        ),
        ("Face and Film in a Parallel", parallel),
        ("Face and Film under a Layer", layer),
    ]
}

fn tiled(tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            face_model: Some(two_faces()),
            data_root: Some(data_root()),
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// A tiled export gives the same pixels as the export in one piece, for every graph order, and both faces are retouched whole.
#[test]
fn a_tiled_export_equals_the_whole_frame_in_every_graph_order() {
    let d = photo_sized(600, 400);
    let src = d.source();
    let whole = tiled(4096);
    let mut cuts = 0;
    for (name, graph) in orders() {
        let reference = export(&whole, &src, graph.clone());
        // The faces are in it: the graph is not the same without the Face Refinement nodes (its Film nodes only).
        assert_ne!(
            reference.rgb,
            export(&whole, &src, plain()).rgb,
            "{name}: the graph does something"
        );
        for tile in [97usize, 160] {
            let renderer = tiled(tile);
            let img = export(&renderer, &src, graph.clone());
            cuts += renderer.face_regions_cut();
            let diff = differing(&reference, &img);
            assert!(
                diff.is_empty(),
                "{name}: tiles of {tile} px differ from the whole frame at {} pixels, the first at {:?}",
                diff.len(),
                diff.first()
            );
        }
    }
    // No face was left partly unretouched: the halo of a Film node never cut a region (`docs/27` S-11b2).
    assert_eq!(
        cuts + whole.face_regions_cut(),
        0,
        "regions left unretouched"
    );
}

/// A viewport frame equals the same pixels of a frame of the whole photo at 100 % (within the dither's one code), for every graph
/// order, and no region of a face is left unretouched.
#[test]
fn a_viewport_frame_equals_the_whole_frame_in_every_graph_order() {
    let r = renderer_with(Some(two_faces()), 0);
    cells_equal_the_whole_frame(&r, "CPU", false);
}

/// The cells of a viewport at 100 % against the frame of the whole photo, in every graph order. With `device`, the frames are the
/// device's (and say so), the kernels of the face step make whole regions whatever the cell, and the bound is the same.
fn cells_equal_the_whole_frame(r: &Renderer, route: &str, device: bool) {
    let d = photo_sized(600, 400);
    let src = d.source();
    r.navigate(src.id);
    let opts = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let view = |s: &PhotoSource, w: f64, h: f64, x0: f64, y0: f64| {
        let mut st = ViewportState::new(ViewportRect::cell(w, h, 1.0));
        st.mode = ym_core::pipeline::ViewportMode::Percent { value: 100.0 };
        st.centre = [(x0 + w / 2.0) / 600.0, (y0 + h / 2.0) / 400.0];
        let frame = r
            .viewport_frame(s, &st, &opts, &CancelToken::new())
            .expect("frame");
        if device {
            assert_eq!(
                frame.report.device,
                ym_core::pipeline::ComputeDevice::Gpu,
                "{route}: cell at ({x0}, {y0}): {:?}",
                frame.report.cpu_reason
            );
            assert!(!frame.report.device_refused, "{route}: refused");
        }
        frame
    };
    let mut graphs = orders();
    graphs.insert(0, ("no nodes", plain()));
    for (name, graph) in graphs {
        let node = src.with_settings(graph);
        let _ = view(&node, 200.0, 200.0, 0.0, 0.0);
        if name != "no nodes" {
            wait_until("the faces", 60, || {
                !r.faces_worker_alive(src.id) && r.faces_ready(&node)
            });
        }
        let whole = view(&node, 600.0, 400.0, 0.0, 0.0);
        assert_eq!((whole.width, whole.height), (600, 400));
        // Cells whose edges cut the faces' regions and the halation's reach.
        let mut checked = 0;
        for (x0, y0, w, h) in [
            (0.0, 0.0, 150.0, 150.0),
            (75.0, 50.0, 150.0, 150.0),
            (116.0, 75.0, 151.0, 129.0),
            (20.0, 165.0, 200.0, 110.0),
            (250.0, 100.0, 167.0, 151.0),
            (300.0, 60.0, 125.0, 200.0),
            (400.0, 250.0, 200.0, 150.0),
        ] {
            let part = view(&node, w, h, x0, y0);
            assert_eq!((part.width, part.height), (w as usize, h as usize));
            for y in 0..part.height {
                for x in 0..part.width {
                    let a = &part.rgba[(y * part.width + x) * 4..][..4];
                    let b = &whole.rgba[((y + y0 as usize) * 600 + x + x0 as usize) * 4..][..4];
                    // The 8-bit frame is dithered by the cell's own pixel grid, so a cell and the crop of the whole differ by one
                    // code in places even for a photo with no node (the first graph of the list); a region cut would show more.
                    let worst = a
                        .iter()
                        .zip(b)
                        .map(|(p, q)| p.abs_diff(*q))
                        .max()
                        .unwrap_or(0);
                    assert!(
                        worst <= 1,
                        "{route}, {name}: cell at ({x0}, {y0}), pixel ({x}, {y}): {a:?} against {b:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }
    assert_eq!(r.face_regions_cut(), 0, "regions left unretouched");
}

// ---- Show skin areas and the faces' pictures (S-11c) -------------------------------------------------------------------------

fn areas_view(face: Option<u32>, node: u32) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    st.mask_view = Some(ym_core::pipeline::MaskView {
        node,
        style: ym_core::pipeline::MaskStyle::Overlay,
        row: None,
        depth: false,
        face,
    });
    st
}

fn view_frame(
    r: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
) -> ym_core::pipeline::RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

/// The photo with the node and its faces ready, on a renderer that has them.
fn ready(body: Value) -> (Developed, Renderer, PhotoSource) {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    r.navigate(src.id);
    let node = src.with_settings(with_node(body));
    let _ = frame(&r, &node);
    wait_until("the faces", 60, || {
        !r.faces_worker_alive(src.id) && r.faces_ready(&node)
    });
    (d, r, node)
}

#[test]
fn show_skin_areas_tints_the_selected_face_and_leaves_the_settings_alone() {
    let (_d, r, node) = ready(strong());
    let before = serde_json::to_value(&node.settings).expect("settings");
    let plain_view = ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0));
    let photo_frame = view_frame(&r, &node, &plain_view);
    let one = view_frame(&r, &node, &areas_view(Some(1), 2));
    let two = view_frame(&r, &node, &areas_view(Some(2), 2));
    assert_eq!(
        (one.width, one.height),
        (photo_frame.width, photo_frame.height)
    );
    // The cell is 600 x 400 for a 1200 x 800 photo: a face's box is half its share of the cell.
    let near = |f: [f64; 4], x: usize, y: usize| {
        let (cx, cy) = ((f[0] + f[2] / 2.0) * 600.0, (f[1] + f[3] / 2.0) * 400.0);
        let (hw, hh) = (f[2] * 600.0 * 0.7, f[3] * 400.0 * 0.7);
        (x as f64 - cx).abs() <= hw && (y as f64 - cy).abs() <= hh
    };
    let changed = |a: &ym_core::pipeline::RenderedFrame, b: &ym_core::pipeline::RenderedFrame| {
        let mut out = Vec::new();
        for y in 0..a.height {
            for x in 0..a.width {
                let i = (y * a.width + x) * 4;
                if a.rgba[i..i + 4] != b.rgba[i..i + 4] {
                    out.push((x, y));
                }
            }
        }
        out
    };
    let c1 = changed(&photo_frame, &one);
    let c2 = changed(&photo_frame, &two);
    assert!(
        c1.len() > 500 && c2.len() > 500,
        "{} and {}",
        c1.len(),
        c2.len()
    );
    assert!(
        c1.iter().all(|&(x, y)| near(FACE_A, x, y)),
        "face 1's tint is on face 1"
    );
    assert!(
        c2.iter().all(|&(x, y)| near(FACE_B, x, y)),
        "face 2's tint is on face 2"
    );
    // The centre of the skin is tinted pink: red up, blue up.
    let (cx, cy) = (
        ((FACE_A[0] + FACE_A[2] * 0.3) * 600.0) as usize,
        ((FACE_A[1] + FACE_A[3] * 0.25) * 400.0) as usize,
    );
    let i = (cy * one.width + cx) * 4;
    assert!(
        one.rgba[i] > photo_frame.rgba[i] && one.rgba[i + 2] > photo_frame.rgba[i + 2],
        "pink on the skin"
    );
    // It is a view: the settings are the ones it had, and the photo's own frame is the same as before it.
    assert_eq!(
        serde_json::to_value(&node.settings).expect("settings"),
        before
    );
    assert_eq!(view_frame(&r, &node, &plain_view).rgba, photo_frame.rgba);
    // Whatever the style says.
    let mut mask_only = areas_view(Some(1), 2);
    if let Some(v) = mask_only.mask_view.as_mut() {
        v.style = ym_core::pipeline::MaskStyle::MaskOnly;
    }
    assert_eq!(view_frame(&r, &node, &mask_only).rgba, one.rgba);
}

#[test]
fn show_skin_areas_with_no_such_face_is_the_photo() {
    let (_d, r, node) = ready(strong());
    let photo_frame = view_frame(
        &r,
        &node,
        &ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0)),
    );
    for face in [Some(0), Some(3), Some(99)] {
        assert_eq!(
            view_frame(&r, &node, &areas_view(face, 2)).rgba,
            photo_frame.rgba,
            "face {face:?}"
        );
    }
    // No faces known yet: the photo.
    let d = photo();
    let src = d.source();
    let fresh = renderer_with(Some(two_faces()), 0);
    let node = src.with_settings(with_node(strong()));
    let plain_frame = view_frame(
        &fresh,
        &node,
        &ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0)),
    );
    let early = view_frame(&fresh, &node, &areas_view(Some(1), 2));
    assert_eq!(early.rgba, plain_frame.rgba);
}

#[test]
fn a_node_that_is_not_a_face_refinement_node_ignores_the_face_field() {
    let (_d, r, src) = ready(strong());
    let graph = doc_of(
        vec![corrector(2, 0.5), face_node(3, strong())],
        line(&[1, 2, 3]),
    );
    let node = src.with_settings(graph);
    // Node 2 is a corrector: its mask view is the same with or without a face number.
    let without = view_frame(&r, &node, &areas_view(None, 2));
    let with = view_frame(&r, &node, &areas_view(Some(1), 2));
    assert_eq!(with.rgba, without.rgba);
    // A node the photo does not have is the photo.
    let missing = view_frame(&r, &node, &areas_view(Some(1), 77));
    let photo_frame = view_frame(
        &r,
        &node,
        &ViewportState::new(ViewportRect::cell(600.0, 400.0, 1.0)),
    );
    assert_eq!(missing.rgba, photo_frame.rgba);
}

#[test]
fn a_face_picture_is_a_small_png_of_the_input_without_the_retouch() {
    let d = photo();
    let src = d.source();
    let r = renderer_with(Some(two_faces()), 0);
    r.navigate(src.id);
    // Before the faces are known: an error with a plain sentence.
    let early = r
        .face_crop_png(&src.with_settings(with_node(strong())), 1)
        .expect_err("no faces yet");
    assert!(early.to_string().contains("not found yet"), "{early}");
    let node = src.with_settings(with_node(strong()));
    let _ = frame(&r, &node);
    wait_until("the faces", 60, || {
        !r.faces_worker_alive(src.id) && r.faces_ready(&node)
    });
    let png = r.face_crop_png(&node, 2).expect("a crop");
    let decoded = png::Decoder::new(std::io::Cursor::new(&png))
        .read_info()
        .expect("a PNG");
    let info = decoded.info();
    assert_eq!((info.width, info.height), (104, 104));
    // The Input's colour, not the retouch: the node's amounts do not change it.
    let other = src.with_settings(with_node(
        json!({ "all": { "blemish": 100, "dark_circles": 100, "even_skin": 100 } }),
    ));
    assert_eq!(r.face_crop_png(&other, 2).expect("a crop"), png);
    assert_ne!(r.face_crop_png(&node, 1).expect("a crop"), png);
    // A number past the end, and 0.
    for n in [0, 3] {
        let e = r.face_crop_png(&node, n).expect_err("no such face");
        assert!(e.to_string().contains("no face"), "{e}");
    }
}

// ---- the device (S-13) -------------------------------------------------------------------------------------------------------
//
// The face step runs on the device: the per-pixel apply of the cached maps and masks and the low-frequency operator are the
// kernels of `ym_gpu::kernels::face`, held to the CPU's `ym_retouch` in `ym-gpu/tests/gpu_face.rs`. Here the whole render is on
// the device (lavapipe in CI; skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` makes that a failure): the frames
// and exports are the device's and agree with the CPU's within the bound of the other device-against-CPU tests, the regions are
// made whole whatever the tile or the cell (so the device's parts equal its whole, to the dither), and a fault in a face kernel
// is answered on the CPU.

use std::io::Write as _;
use std::path::Path;

use ym_core::app::gpu::GpuStages;
use ym_core::export::{
    ExportContext, ExportControl, ExportParams, ExportPhoto, ExportReport, export_photo,
};
use ym_core::pipeline::{ComputeDevice, RenderedFrame, ViewportMode};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};

fn gpu(poison: bool) -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        poison,
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
                "\n* GPU TESTS SKIPPED in face_render: {e}\n"
            );
            None
        }
    }
}

fn device_renderer(gpu: Option<Arc<Gpu>>, tile: usize) -> Renderer {
    let stages = if gpu.is_some() {
        GpuStages::ALL
    } else {
        GpuStages::NONE
    };
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            face_model: Some(two_faces()),
            data_root: Some(data_root()),
            full_tile_px: tile,
            gpu,
            gpu_stages: stages,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// Every tool at a clear amount, every face.
fn everything() -> Value {
    json!({ "all": { "blemish": 80, "dark_circles": 70, "even_skin": 70, "texture": 20,
                     "skin_contrast": 40, "shadow_desaturation": 40, "shine_removal": 50, "teeth_whitening": 60 } })
}

fn device_view(zoom_100: bool) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(500.0, 330.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st
}

/// A frame of `src` as it is now.
fn frame_of(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
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

/// A frame of `src` once its faces are found.
fn frame_when_ready(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    let _ = frame_of(r, src, st);
    wait_until("the faces", 60, || {
        !r.faces_worker_alive(src.id) && r.faces_ready(src)
    });
    frame_of(r, src, st)
}

/// The largest difference of two 8-bit frames, and the pixel values more than one code apart.
fn frame_diff(a: &RenderedFrame, b: &RenderedFrame) -> (u8, usize) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut max, mut over) = (0u8, 0usize);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        over += usize::from(d > 1);
    }
    (max, over)
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[face step on the device] {text}");
}

/// The frame of a live node is the device's (not refused, no note about the CPU), says so to the viewport, agrees with the CPU's
/// within the view bound at Fit and at 100 %, and is not the frame without the node.
#[test]
fn the_face_step_runs_on_the_device_and_a_frame_agrees_with_the_cpu() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let device = device_renderer(Some(gpu), 256);
    let cpu = device_renderer(None, 256);
    device.navigate(d.source().id);
    cpu.navigate(d.source().id);
    for (what, body) in [("grading", strong()), ("every tool", everything())] {
        let src = d.source().with_settings(with_node(body));
        for zoom_100 in [false, true] {
            let st = device_view(zoom_100);
            let at = format!("{what}, 100 % {zoom_100}");
            let got = frame_when_ready(&device, &src, &st);
            let want = frame_when_ready(&cpu, &src, &st);
            assert_eq!(got.report.device, ComputeDevice::Gpu, "{at}");
            assert!(!got.report.device_refused, "{at}");
            assert!(
                got.report.cpu_reason.is_none(),
                "{at}: {:?}",
                got.report.cpu_reason
            );
            assert!(!device.views_on_cpu(src.id), "{at}: views_on_cpu");
            assert_eq!(want.report.device, ComputeDevice::Cpu, "{at}");
            let (max, over) = frame_diff(&got, &want);
            say(&format!(
                "view, {at}: largest {max} codes, {over} values over 1"
            ));
            assert!(
                max <= 2 && over < got.rgba.len() / 2000,
                "{at}: {max}, {over}"
            );
            // The node is in the picture: the same photo without it is not this frame, on the device and by a clear amount.
            let none = frame_of(&device, &d.source(), &st);
            let (moved, _) = frame_diff(&got, &none);
            assert!(
                moved >= 1,
                "{at}: the node moves the device frame by {moved} codes"
            );
        }
    }
}

/// A PNG's samples, 16 bit.
fn decode_png16(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read the export");
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
        .collect()
}

fn png_options(out: &Path) -> ExportOptions {
    ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        color_space: ColorSpace::Srgb,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    }
}

fn export_png(
    d: &Developed,
    settings: &DevelopSettings,
    r: Renderer,
    out: &Path,
) -> Result<ExportReport, ym_core::export::ExportError> {
    let ctx = ExportContext::new(r).with_home(Some(out.to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = photo.source.with_settings(settings.clone());
    export_photo(
        &ctx,
        &photo,
        &ExportParams::new(png_options(out)),
        &ExportControl::none(),
    )
}

fn code_diff(a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// One 10-bit code after C13, in the codes of a 16-bit file.
const TEN_BIT_CODE: u32 = 64;

/// The export tail makes the faces on the device and agrees with the CPU's export within one 10-bit code, and for every tile
/// size the device's export is the same bytes: a region is made whole in whichever tile it is met.
#[test]
fn the_device_export_agrees_with_the_cpu_and_is_the_same_for_every_tile() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let settings = with_node(everything());
    let home = tempfile::tempdir().expect("dir");
    let cpu = export_png(&d, &settings, device_renderer(None, 256), home.path()).expect("cpu");
    assert_eq!(cpu.device, ComputeDevice::Cpu);
    let cpu_codes = decode_png16(&cpu.path);
    // The photo's own difference between the device and the CPU, without the node. With a node the plan is split into passes (a
    // step that reads a picture made before the colour pass, as a Film node's), and the device's colour pass differs from the CPU's
    // by up to 37 of 65535 in this photo whatever the node does (`each_tool_alone_and_together_agree_with_the_cpu_in_an_export`:
    // the same 37, at the same pixel, for every tool), under the 10-bit code the other nodes of this kind are held to.
    let plain_cpu =
        export_png(&d, &plain(), device_renderer(None, 256), home.path()).expect("plain");
    let plain_dev = export_png(
        &d,
        &plain(),
        device_renderer(Some(Arc::clone(&gpu)), 400),
        home.path(),
    )
    .expect("plain on the device");
    let baseline = code_diff(
        &decode_png16(&plain_dev.path),
        &decode_png16(&plain_cpu.path),
    );
    say(&format!(
        "export without the node: largest {baseline} of 65535 against the CPU"
    ));
    let mut first: Option<Vec<u32>> = None;
    for tile in [160usize, 400, 1024] {
        let out = tempfile::tempdir().expect("dir");
        let done = export_png(
            &d,
            &settings,
            device_renderer(Some(Arc::clone(&gpu)), tile),
            out.path(),
        )
        .expect("device export");
        assert_eq!(
            done.device,
            ComputeDevice::Gpu,
            "tile {tile}: {:?}",
            done.notes
        );
        assert!(
            !done.notes.iter().any(|n| n.contains("graphics card")),
            "tile {tile}: {:?}",
            done.notes
        );
        let codes = decode_png16(&done.path);
        let worst = code_diff(&codes, &cpu_codes);
        assert!(
            worst <= TEN_BIT_CODE,
            "tile {tile}: {worst}, the photo alone differs by {baseline}"
        );
        say(&format!(
            "export, tile {tile}: largest {worst} of 65535 against the CPU (a 10-bit code is {TEN_BIT_CODE})"
        ));
        let n = codes.iter().zip(&cpu_codes).filter(|(a, b)| a != b).count();
        say(&format!(
            "  tile {tile}: {n} of {} codes differ from the CPU's; notes {:?}",
            codes.len(),
            done.notes
        ));
        // Where the device differs from the CPU: near a face (the tools' reach) and away from the faces, where the node does
        // nothing and only the colour pass, whose plan the node splits into passes, can differ.
        let (w, h) = (1200usize, 800usize);
        let near_face = |x: usize, y: usize| {
            [FACE_A, FACE_B].iter().any(|f| {
                let long = (f[2] * w as f64).max(f[3] * h as f64);
                let (x0, y0) = (f[0] * w as f64 - 0.5 * long, f[1] * h as f64 - 0.5 * long);
                let (x1, y1) = (
                    (f[0] + f[2]) * w as f64 + 0.5 * long,
                    (f[1] + f[3]) * h as f64 + 0.5 * long,
                );
                (x as f64) >= x0 && (x as f64) < x1 && (y as f64) >= y0 && (y as f64) < y1
            })
        };
        let (mut worst_near, mut worst_away) = (0u32, 0u32);
        for (i, (a, b)) in codes.iter().zip(&cpu_codes).enumerate() {
            let (x, y) = ((i / 3) % w, (i / 3) / w);
            let d = a.abs_diff(*b);
            if near_face(x, y) {
                worst_near = worst_near.max(d);
            } else {
                worst_away = worst_away.max(d);
            }
        }
        say(&format!(
            "  tile {tile}: the largest difference is {worst_near} near the faces and {worst_away} away from them"
        ));
        match &first {
            None => first = Some(codes),
            Some(f) => assert_eq!(f, &codes, "tile {tile}: the bytes depend on the tile"),
        }
    }
    // The node is in the picture.
    let none = export_png(&d, &plain(), device_renderer(None, 256), home.path()).expect("plain");
    assert!(
        code_diff(
            first.as_ref().expect("an export"),
            &decode_png16(&none.path)
        ) > 300
    );
}

/// The device's cells equal its whole frame in every graph order, to the dither: the regions the cell cuts are made whole.
#[test]
fn the_devices_viewport_cells_equal_its_whole_frame_in_every_graph_order() {
    let Some(gpu) = gpu(true) else { return };
    let r = device_renderer(Some(gpu), 256);
    cells_equal_the_whole_frame(&r, "device", true);
}

/// Painted protection holds on the device as it does on the CPU: not one bit of a protected face changes, and the other is made.
#[test]
fn painted_protection_holds_on_the_device() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let device = |s: &DevelopSettings| {
        let out = tempfile::tempdir().expect("dir");
        let done = export_png(
            &d,
            s,
            device_renderer(Some(Arc::clone(&gpu)), 400),
            out.path(),
        )
        .expect("device export");
        assert_eq!(done.device, ComputeDevice::Gpu, "{:?}", done.notes);
        let codes = decode_png16(&done.path);
        drop(out);
        codes
    };
    let reference = device(&plain());
    let (cx, cy) = (FACE_A[0] + FACE_A[2] / 2.0, FACE_A[1] + FACE_A[3] / 2.0);
    let mut body = strong();
    body["protect"] = json!({ "strokes": [
        { "size": 0.25, "hardness": 1.0, "erase": false, "points": [[cx - 0.05, cy - 0.25], [cx - 0.05, cy + 0.25]] },
        { "size": 0.25, "hardness": 1.0, "erase": false, "points": [[cx + 0.05, cy - 0.25], [cx + 0.05, cy + 0.25]] },
    ] });
    let open = device(&with_node(strong()));
    let held = device(&with_node(body));
    let (w, h) = (1200usize, 800usize);
    let inside = |face: [f64; 4], x: usize, y: usize| {
        let (x0, y0) = (face[0] * w as f64, face[1] * h as f64);
        let (x1, y1) = (
            (face[0] + face[2]) * w as f64,
            (face[1] + face[3]) * h as f64,
        );
        (x as f64) >= x0 && (x as f64) < x1 && (y as f64) >= y0 && (y as f64) < y1
    };
    let (mut a_changed, mut b_changed, mut b_differs) = (0usize, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                if inside(FACE_A, x, y) && held[i] != reference[i] {
                    a_changed += 1;
                }
                if inside(FACE_B, x, y) {
                    b_changed += usize::from(held[i] != reference[i]);
                    b_differs += usize::from(held[i] != open[i]);
                }
            }
        }
    }
    assert_eq!(a_changed, 0, "face A is protected: not one code changes");
    assert!(b_changed > 100, "face B is not protected: {b_changed}");
    assert_eq!(
        b_differs, 0,
        "face B is as it is without protection, where the plane is 0"
    );
}

/// The protection plane's Grow, Feather and Refine edges (the raster passes of the protection row, with the Input's luma as the
/// guide) give the CPU's picture on the device, within one 10-bit code, and the protection matters in it.
#[test]
fn the_protection_planes_edges_agree_with_the_cpu_on_the_device() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let (cx, cy) = (FACE_A[0] + FACE_A[2] / 2.0, FACE_A[1] + FACE_A[3] / 2.0);
    let protected = |edges: Value| {
        let mut body = everything();
        let mut protect = json!({ "strokes": [
            { "size": 0.12, "hardness": 0.6, "erase": false, "points": [[cx + 0.02, cy - 0.2], [cx + 0.06, cy + 0.2]] },
        ] });
        protect
            .as_object_mut()
            .unwrap()
            .extend(edges.as_object().unwrap().clone());
        body["protect"] = protect;
        with_node(body)
    };
    let render = |s: &DevelopSettings, on_device: bool| {
        let out = tempfile::tempdir().expect("dir");
        let r = device_renderer(on_device.then(|| Arc::clone(&gpu)), 400);
        let done = export_png(&d, s, r, out.path()).expect("export");
        assert_eq!(
            done.device == ComputeDevice::Gpu,
            on_device,
            "{:?}",
            done.notes
        );
        decode_png16(&done.path)
    };
    let open = render(&with_node(everything()), false);
    for (what, edges) in [
        ("feather", json!({ "feather": 0.4 })),
        ("grow", json!({ "grow": 0.3 })),
        ("refine", json!({ "refine": 0.7 })),
        (
            "all three",
            json!({ "feather": 0.3, "grow": -0.2, "refine": 0.5 }),
        ),
    ] {
        let s = protected(edges);
        let (cpu, dev) = (render(&s, false), render(&s, true));
        let worst = code_diff(&cpu, &dev);
        let held = code_diff(&cpu, &open);
        say(&format!(
            "protection with {what}: the device differs from the CPU by {worst} of 65535 (the protection moves the picture by {held})"
        ));
        assert!(held > 300, "{what}: the protection does something ({held})");
        assert!(worst <= TEN_BIT_CODE, "{what}: {worst}");
    }
}

/// A fault in a face kernel is answered on the CPU (`--features fault-injection`; without it this module is empty).
#[cfg(feature = "fault-injection")]
mod fallback {
    use super::*;
    use ym_gpu::device::GpuFault;

    /// A frame our own check refuses is the CPU's, with the note, and the next frame is the device's.
    #[test]
    fn a_refused_face_frame_costs_one_cpu_frame_and_leaves_the_route_open() {
        let Some(gpu) = gpu(true) else { return };
        let d = photo();
        let src = d.source().with_settings(with_node(everything()));
        let r = device_renderer(Some(Arc::clone(&gpu)), 256);
        r.navigate(src.id);
        let st = device_view(false);
        let device = frame_when_ready(&r, &src, &st);
        assert_eq!(device.report.device, ComputeDevice::Gpu);
        // The faces are ready; the fault fires once, on the next frame's first view job.
        gpu.set_fault(Some(GpuFault::FailInvalid(1)));
        let refused = frame_of(&r, &src, &st);
        gpu.set_fault(None);
        assert_eq!(refused.report.device, ComputeDevice::Cpu);
        assert!(refused.report.device_refused);
        assert!(
            refused
                .report
                .notes
                .iter()
                .any(|n| n.contains("invalid kernel input")),
            "{:?}",
            refused.report.notes
        );
        assert!(!r.raw_on_cpu(src.id), "the route is still open");
        let (max, _) = frame_diff(&refused, &device);
        assert!(max <= 3, "{max} codes");
        let next = frame_of(&r, &src, &st);
        assert_eq!(next.report.device, ComputeDevice::Gpu);
        assert!(!next.report.device_refused);
    }

    /// The face kernel's pipeline fails to compile (every other kernel is compiled already): the frame is the CPU's, with the
    /// note, the route stays open for the frames without the node, and an export makes the whole photo again on the CPU,
    /// bit for bit the CPU export.
    #[test]
    fn a_face_kernel_that_fails_is_answered_on_the_cpu() {
        let Some(gpu) = gpu(true) else { return };
        let d = photo();
        let with = d.source().with_settings(with_node(everything()));
        let without = d.source();
        let r = device_renderer(Some(Arc::clone(&gpu)), 256);
        r.navigate(with.id);
        // Everything the node's frame uses but the face kernel is compiled by a frame of the same photo without the node's work.
        let st = device_view(false);
        let warm = frame_of(&r, &without, &st);
        assert_eq!(warm.report.device, ComputeDevice::Gpu);
        gpu.set_fault(Some(GpuFault::Shader));
        let refused = frame_when_ready(&r, &with, &st);
        gpu.set_fault(None);
        assert_eq!(
            refused.report.device,
            ComputeDevice::Cpu,
            "{:?}",
            refused.report.notes
        );
        assert!(refused.report.device_refused);

        let cpu_dir = tempfile::tempdir().expect("dir");
        let settings = with_node(everything());
        let cpu =
            export_png(&d, &settings, device_renderer(None, 256), cpu_dir.path()).expect("cpu");
        for fault in [GpuFault::Shader, GpuFault::OutOfBudget, GpuFault::Lost] {
            // A device that is lost stays lost: each case gets a device of its own.
            let own = Arc::new(Gpu::start(GpuConfig::default()).expect("device"));
            own.set_fault(Some(fault));
            let out = tempfile::tempdir().expect("dir");
            let done = export_png(
                &d,
                &settings,
                device_renderer(Some(Arc::clone(&own)), 256),
                out.path(),
            );
            own.set_fault(None);
            let done = done.unwrap_or_else(|e| panic!("{fault:?}: {e}"));
            assert_eq!(done.device, ComputeDevice::Cpu, "{fault:?}");
            assert!(
                done.notes
                    .iter()
                    .any(|n| n
                        .contains("The graphics card could not render this photo, so the CPU did")),
                "{fault:?}: {:?}",
                done.notes
            );
            assert_eq!(
                std::fs::read(&done.path).expect("read"),
                std::fs::read(&cpu.path).expect("read"),
                "{fault:?}: the fallback is not the CPU export"
            );
        }
    }
}

/// Each tool on its own and some together, on the device against the CPU, in an export: the number for each (and where the worst
/// pixel is), so that a combination that differs is found at once.
#[test]
fn each_tool_alone_and_together_agree_with_the_cpu_in_an_export() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let amounts = |pairs: &[(&str, i64)]| {
        let mut all = serde_json::Map::new();
        for (k, v) in pairs {
            all.insert((*k).to_owned(), json!(v));
        }
        json!({ "all": all })
    };
    let mut cases: Vec<(String, Value)> = [
        "blemish",
        "dark_circles",
        "even_skin",
        "skin_contrast",
        "shadow_desaturation",
        "shine_removal",
        "teeth_whitening",
    ]
    .iter()
    .map(|t| ((*t).to_owned(), amounts(&[(t, 70)])))
    .collect();
    cases.push((
        "texture with dark circles".to_owned(),
        amounts(&[("dark_circles", 70), ("texture", 20)]),
    ));
    cases.push((
        "blemish with even skin".to_owned(),
        amounts(&[("blemish", 80), ("even_skin", 70)]),
    ));
    cases.push((
        "grading".to_owned(),
        amounts(&[
            ("skin_contrast", 40),
            ("shadow_desaturation", 40),
            ("shine_removal", 50),
        ]),
    ));
    cases.push(("every tool".to_owned(), everything()));
    for (name, body) in cases {
        let settings = with_node(body);
        let (cpu_dir, dev_dir) = (
            tempfile::tempdir().expect("dir"),
            tempfile::tempdir().expect("dir"),
        );
        let cpu =
            export_png(&d, &settings, device_renderer(None, 256), cpu_dir.path()).expect("cpu");
        let dev = export_png(
            &d,
            &settings,
            device_renderer(Some(Arc::clone(&gpu)), 400),
            dev_dir.path(),
        )
        .expect("device");
        assert_eq!(dev.device, ComputeDevice::Gpu, "{name}: {:?}", dev.notes);
        let (a, b) = (decode_png16(&dev.path), decode_png16(&cpu.path));
        let worst = code_diff(&a, &b);
        let at = a
            .iter()
            .zip(&b)
            .enumerate()
            .max_by_key(|(_, (x, y))| x.abs_diff(**y))
            .map_or(0, |(i, _)| i / 3);
        let n = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        say(&format!(
            "{name}: the device differs from the CPU by {worst} of 65535 at ({}, {}), {n} of {} codes differ",
            at % 1200,
            at / 1200,
            a.len()
        ));
        assert!(worst <= TEN_BIT_CODE, "{name}: {worst} at pixel {at}");
    }
}
