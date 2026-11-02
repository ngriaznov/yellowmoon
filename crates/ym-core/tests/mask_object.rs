//! Object masks through the render graph (`docs/23-nodes.md` sections 4.4 and 9, `docs/algorithms/06-masks.md` section 6,
//! task N-18): SAM 2.1 small's encoder runs in the background for the photo being looked at, `mask_find` adds a click or a
//! box to an Object row and answers the row's new points and box (or that nothing was found), and a row's raster `S0` is
//! the decoder's soft mask on its stored points and box.
//!
//! These tests need the shipped weights (`data/models/sam2/`, fetched by `tools/fetch/sam2.sh`) and say so and pass
//! without them (`YM_REQUIRE_SAM2=1` makes that a failure). SAM's encoder is 180 billion multiply-adds, so run them in an
//! optimised build: `cargo test -p ym-core --release --test mask_object`.
mod common;

use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ym_color::display::OutputTarget;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{FullRequest, FullTarget};
use ym_core::session::PhotoId;
use ym_core::settings::{
    MaskKind, NO_OBJECT_MODEL_SENTENCE, SettingsPatch, mask_kinds_with, mask_unavailable_with,
};
use ym_ml::sam2::{Decoding, Picture, Prompt};

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::ComputeDevice;
use ym_core::pipeline::{
    FrameOptions, MaskFound, MaskStyle, MaskView, NeutralOnly, ObjectModel, ObjectPhase,
    ObjectPrompt, PhotoSource, RenderOptions, RenderedFrame, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The paper around a photo in a Mask only frame.
const PAPER: [u8; 3] = [40, 60, 90];
const CELL: f64 = 300.0;

fn model() -> Option<Arc<ObjectModel>> {
    static CELL: OnceLock<Option<Arc<ObjectModel>>> = OnceLock::new();
    let found = CELL
        .get_or_init(|| ObjectModel::probe(&data_root()))
        .clone();
    if found.is_none() {
        assert!(
            std::env::var_os("YM_REQUIRE_SAM2").is_none(),
            "YM_REQUIRE_SAM2 is set and data/models/sam2 is missing"
        );
        eprintln!("[object] data/models/sam2 is absent: skipped");
    }
    found
}

fn renderer_with(model: Option<Arc<ObjectModel>>, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            object_model: model,
            data_root: Some(data_root()),
            background: PAPER,
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn renderer(model: Arc<ObjectModel>) -> Renderer {
    renderer_with(Some(model), RenderOptions::default().full_tile_px)
}

/// Input, Main, node 2 (with these rows and an exposure trim of a stop), Output.
fn graph(rows: Vec<Value>) -> DevelopSettings {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": rows } },
    ], "links": [
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

fn object_row(extra: Value) -> Value {
    let mut row = json!({ "kind": "object", "mode": "add" });
    if let (Some(r), Some(e)) = (row.as_object_mut(), extra.as_object()) {
        r.extend(e.clone());
    }
    row
}

fn mask_view(node: u32) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node,
        style: MaskStyle::MaskOnly,
    });
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

fn coverage(f: &RenderedFrame) -> f64 {
    let v: Vec<f64> = f
        .rgba
        .chunks(4)
        .filter(|p| p[..3] != PAPER)
        .map(|p| f64::from(p[0]) / 255.0)
        .collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// A photo with a bright disc on a darker ground: an object SAM may find.
fn disc_photo() -> Developed {
    let (w, h) = (1200u32, 800u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.08).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.6 * (scene.white_level - scene.black)) as u16;
    let (cx, cy, r) = (0.45 * f64::from(w), 0.5 * f64::from(h), 0.2 * f64::from(h));
    for y in 0..h {
        for x in 0..w {
            if (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy) <= r {
                mosaic[(y * w + x) as usize] = code;
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

/// The clicks of an Object row, `[x, y, positive]`.
type Points = Vec<(f64, f64, bool)>;

fn point(x: f64, y: f64, positive: bool) -> ObjectPrompt {
    ObjectPrompt::Point { x, y, positive }
}

fn found(r: &Renderer, src: &PhotoSource, row: usize, prompt: ObjectPrompt) -> MaskFound {
    r.mask_find(src, 2, row, prompt, &CancelToken::new())
        .expect("an answer")
}

/// A source whose node 2 has one Object row with these points and box.
fn with_row(src: &PhotoSource, points: &[(f64, f64, bool)], bbox: Option<[f64; 4]>) -> PhotoSource {
    src.with_settings(graph(vec![object_row(
        json!({ "points": points, "box": bbox }),
    )]))
}

fn plain(r: &Renderer, src: &PhotoSource) -> RenderedFrame {
    r.viewport_frame(
        src,
        &ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0)),
        &FrameOptions::default(),
        &CancelToken::new(),
    )
    .expect("frame")
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ---- the kind and the sentence ---------------------------------------------------------------------

#[test]
fn object_is_a_kind_where_the_model_is_and_a_mask_error_where_it_is_not() {
    let Some(model) = model() else { return };
    // The lists: Object first where the weights are, `weights_missing` where they are not.
    let with = mask_kinds_with(true);
    assert_eq!(with[0], MaskKind::Object);
    assert!(
        mask_unavailable_with(true)
            .iter()
            .all(|(k, _)| *k != MaskKind::Object)
    );
    let without = mask_unavailable_with(false);
    assert_eq!(without[0].0, MaskKind::Object);
    assert_eq!(without[0].1.wire(), "weights_missing");
    assert!(!mask_kinds_with(false).contains(&MaskKind::Object));
    // The renderer says the same, and refuses the row without the model with the sentence of `docs/23` section 9.
    let has = renderer(model);
    let lacks = renderer_with(None, 1024);
    assert_eq!(has.mask_kinds(), with);
    assert!(!lacks.mask_kinds().contains(&MaskKind::Object));
    let row = graph(vec![object_row(json!({}))]);
    assert_eq!(has.unmakeable_mask(&row), None);
    assert_eq!(
        lacks.unmakeable_mask(&row).as_deref(),
        Some("This copy of Yellowmoon has no Object model. Reinstall the app.")
    );
    assert_eq!(
        NO_OBJECT_MODEL_SENTENCE,
        lacks.unmakeable_mask(&row).unwrap()
    );
    // A row with the eye off is not in the picture.
    let off = graph(vec![object_row(json!({ "enabled": false }))]);
    assert_eq!(lacks.unmakeable_mask(&off), None);
    // `mask_find` without the model is the same sentence, with the `mask` scope.
    let d = disc_photo();
    let src = with_row(&d.source(), &[], None);
    let err = lacks
        .mask_find(&src, 2, 0, point(0.5, 0.5, true), &CancelToken::new())
        .unwrap_err();
    assert!(
        matches!(err, ym_core::pipeline::RenderError::Mask(ref m) if m == NO_OBJECT_MODEL_SENTENCE)
    );
    // An export of such a photo is refused before a pixel is made; a view shows the row selecting nothing.
    let none = plain(&lacks, &with_row(&d.source(), &[(0.45, 0.5, true)], None));
    let empty = plain(&lacks, &with_row(&d.source(), &[], None));
    assert_eq!(none.rgba, empty.rgba);
    // …and the node that selects nothing changes nothing: the photo as the session has it.
    assert_eq!(empty.rgba, plain(&lacks, &d.source()).rgba);
}

// ---- S0 is the decoder's soft mask -------------------------------------------------------------------

#[test]
fn s0_from_stored_points_is_the_decoders_soft_mask_on_the_picture_the_input_renders() {
    let Some(model) = model() else { return };
    let r = renderer(Arc::clone(&model));
    let d = disc_photo();
    let src = with_row(&d.source(), &[], None);
    let cancel = CancelToken::new();
    // The picture: made by hand here the way the core makes it, then encoded and decoded by the model itself.
    let (w, h, rgb) = r.object_picture_of(&src, &cancel).expect("picture");
    assert_eq!(
        (w, h),
        (1024, 683),
        "the photo frame at 1024 px on the long edge"
    );
    let sam = model.sam().expect("model");
    let features = sam
        .encode(&Picture::new(w, h, &rgb).expect("picture"))
        .expect("encode");
    let cases: [(Points, Option<[f64; 4]>); 3] = [
        (vec![(0.45, 0.5, true)], None),
        (
            vec![(0.45, 0.5, true), (0.3, 0.45, false), (0.55, 0.55, true)],
            None,
        ),
        (vec![(0.45, 0.5, true)], Some([0.3, 0.2, 0.6, 0.8])),
    ];
    for (points, bbox) in cases {
        let prompt = Prompt {
            points: points
                .iter()
                .map(|&(x, y, positive)| ym_ml::sam2::Point {
                    x: x as f32,
                    y: y as f32,
                    positive,
                })
                .collect(),
            bbox: bbox.map(|b| b.map(|v| v as f32)),
        };
        let Decoding::Found(decoded) = sam.decode(&features, &prompt).expect("decode") else {
            panic!("nothing found for {points:?}");
        };
        let soft = decoded.soft_mask(w, h);
        let plane = r
            .object_plane(&src, &points, bbox, &cancel)
            .expect("plane")
            .expect("a plane");
        assert_eq!((plane.width, plane.height), (soft.width, soft.height));
        let want: Vec<u16> = soft
            .data
            .iter()
            .map(|&v| ym_color::mask_raster::quantise(f64::from(v)))
            .collect();
        assert!(
            plane.data == want,
            "the plane is the decoder's soft mask for {points:?} {bbox:?}"
        );
        assert!(plane.data.iter().any(|&c| c > 60_000) && plane.data.iter().any(|&c| c < 5_000));
    }
    // A row with no point and no box selects nothing: its node changes nothing, the frame is the photo's own.
    let empty = plain(&r, &with_row(&d.source(), &[], None));
    assert_eq!(empty.rgba, plain(&r, &d.source()).rgba);
}

#[test]
fn the_row_acts_in_a_frame_and_its_mask_follows_the_points_not_the_grade() {
    let Some(model) = model() else { return };
    let r = renderer(model);
    let d = disc_photo();
    let at = |points: &[(f64, f64, bool)]| {
        let src = with_row(&d.source(), points, None);
        // A viewer does not wait for the encoder: the first frame has no features and the row selects nothing, then they are there.
        r.object_prepare(&src);
        wait_until("the features", 120, || r.object_ready(&src));
        coverage(&frame(&r, &src, &mask_view(2)))
    };
    let disc = at(&[(0.45, 0.5, true)]);
    assert!(disc > 0.01 && disc < 0.6, "the mask covers {disc}");
    // Grading the node moves nothing of its mask (docs/23 section 4.7): another trim, the same coverage and the same features.
    let src = with_row(&d.source(), &[(0.45, 0.5, true)], None);
    let mut s = src.settings.clone();
    s.nodes.corrector_mut(2).expect("node 2").trim.exposure_ev = -2.0;
    let graded = src.with_settings(s);
    assert!(
        r.object_ready(&graded),
        "the same features serve a graded node"
    );
    let again = coverage(&frame(&r, &graded, &mask_view(2)));
    assert_eq!(again, disc);
    assert_eq!(r.object_encodes(src.id), 1, "the encoder ran once");
}

// ---- the background encoder, the cache, mask_find ---------------------------------------------------------

#[test]
fn the_encoder_starts_in_the_background_a_find_waits_for_it_and_the_features_are_kept() {
    let Some(model) = model() else { return };
    let r = renderer(model);
    let d = disc_photo();
    let heard = Arc::new(Mutex::new(Vec::<(PhotoId, ObjectPhase)>::new()));
    {
        let heard = Arc::clone(&heard);
        r.set_object_listener(Arc::new(move |id, phase| {
            heard.lock().unwrap().push((id, phase))
        }));
    }
    let src = with_row(&d.source(), &[], None);
    let id = src.id;
    // The row appears: the encoder is wanted at once, and "Getting ready…" is true.
    r.object_prepare(&src);
    assert!(r.object_preparing(id));
    assert!(!r.object_ready(&src));
    // A click that comes meanwhile waits for it.
    let t = Instant::now();
    let first = found(&r, &src, 0, point(0.45, 0.5, true));
    let waited = t.elapsed();
    let MaskFound::Found {
        points,
        box_,
        took_ms,
    } = first
    else {
        panic!("nothing found on the disc");
    };
    assert_eq!(points, [(0.45, 0.5, true)]);
    assert_eq!(box_, None);
    assert!(
        took_ms > 0.0 && took_ms < 5_000.0,
        "the decoder took {took_ms} ms"
    );
    assert!(
        waited > Duration::from_millis(500),
        "the click waited for the encoder ({waited:?})"
    );
    wait_until("the worker to end", 30, || !r.object_preparing(id));
    assert_eq!(r.object_encodes(id), 1);
    assert_eq!(
        *heard.lock().unwrap(),
        [(id, ObjectPhase::Preparing), (id, ObjectPhase::Idle)]
    );
    // The cache is used: the next clicks (and the same photo with other nodes' settings) run no encoder.
    let t = Instant::now();
    let src2 = with_row(&d.source(), &points, None);
    let second = found(&r, &src2, 0, point(0.3, 0.45, false));
    assert!(
        matches!(second, MaskFound::Found { ref points, .. } if points.len() == 2 && !points[1].2)
    );
    assert!(
        t.elapsed() < Duration::from_secs(3),
        "a second find is a decode ({:?})",
        t.elapsed()
    );
    assert_eq!(r.object_encodes(id), 1, "the features were kept");
    // A prompt that was found is in the cache: the frame that follows decodes nothing, and still shows it.
    let MaskFound::Found { points: both, .. } = second else {
        unreachable!()
    };
    let src3 = with_row(&d.source(), &both, None);
    assert!(r.object_ready(&src3));
    assert!(coverage(&frame(&r, &src3, &mask_view(2))) > 0.0);
    assert_eq!(r.object_encodes(id), 1);
}

#[test]
fn a_box_is_one_prompt_and_a_row_with_256_points_refuses_one_more() {
    let Some(model) = model() else { return };
    let r = renderer(model);
    let d = disc_photo();
    let src = with_row(&d.source(), &[(0.45, 0.5, true)], None);
    let MaskFound::Found { points, box_, .. } = found(
        &r,
        &src,
        0,
        ObjectPrompt::Box {
            x0: 0.6,
            y0: 0.8,
            x1: 0.3,
            y1: 0.2,
        },
    ) else {
        panic!("nothing found in the box");
    };
    // The corners come back in order, and the points the row had stay.
    assert_eq!(box_, Some([0.3, 0.2, 0.6, 0.8]));
    assert_eq!(points, [(0.45, 0.5, true)]);
    // 256 points: one more is refused with the core's sentence, and the row is not asked.
    let full: Vec<(f64, f64, bool)> = (0..256)
        .map(|i| {
            (
                0.4 + f64::from(i % 16) * 0.001,
                0.45 + f64::from(i / 16) * 0.001,
                true,
            )
        })
        .collect();
    let src = with_row(&d.source(), &full, None);
    let err = r
        .mask_find(&src, 2, 0, point(0.5, 0.5, true), &CancelToken::new())
        .unwrap_err();
    assert!(
        matches!(err, ym_core::pipeline::RenderError::Mask(ref m) if m == "An Object mask can have up to 256 points."),
        "{err}"
    );
    // A mask that is not an Object row, and a row that is not there, are the `mask` scope too.
    let radial = src.with_settings(graph(vec![json!({ "kind": "radial" })]));
    assert!(
        r.mask_find(&radial, 2, 0, point(0.5, 0.5, true), &CancelToken::new())
            .is_err()
    );
    assert!(
        r.mask_find(&radial, 2, 5, point(0.5, 0.5, true), &CancelToken::new())
            .is_err()
    );
    assert!(
        r.mask_find(&radial, 9, 0, point(0.5, 0.5, true), &CancelToken::new())
            .is_err()
    );
    // Coordinates are stored to four decimals, so the decoder was given exactly what the settings keep.
    let MaskFound::Found { points, .. } = found(
        &r,
        &with_row(&d.source(), &[], None),
        0,
        point(0.123_456_789, 0.5, true),
    ) else {
        panic!("found");
    };
    assert_eq!(points[0].0, 0.1235);
}

#[test]
fn a_photo_change_stops_an_encoder_that_started_on_its_own_and_a_wish_the_person_left() {
    let Some(model) = model() else { return };
    let r = renderer(model);
    let d = disc_photo();
    let src = with_row(&d.source(), &[], None);
    let id = src.id;
    r.navigate(id);
    r.object_prepare(&src);
    // The picture and the encoder have started; the person moves to another photo.
    std::thread::sleep(Duration::from_millis(900));
    assert!(r.object_preparing(id));
    r.navigate(PhotoId(id.0 + 100));
    wait_until("the worker to stop", 30, || !r.object_preparing(id));
    assert_eq!(
        r.object_encodes(id),
        0,
        "the stale encode was stopped, not finished"
    );
    assert!(!r.object_ready(&src));
    // Back on the photo a click makes the features itself and finds the object.
    r.navigate(id);
    assert!(matches!(
        found(&r, &src, 0, point(0.45, 0.5, true)),
        MaskFound::Found { .. }
    ));
    assert_eq!(r.object_encodes(id), 1);
}

// ---- the app's path: Core --------------------------------------------------------------------------------

#[test]
fn the_core_offers_object_asks_for_the_features_when_the_row_appears_and_a_find_changes_no_settings()
 {
    if model().is_none() {
        return;
    }
    let app = tempfile::tempdir().expect("app root");
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.decode_threads = 2;
    options.thumbnails = false;
    let core = Core::open(options).expect("core");
    // `mask_kinds()` gains Object when the weights are there (and is first of the list).
    assert_eq!(core.mask_kinds().first(), Some(&MaskKind::Object));
    assert!(
        core.mask_unavailable()
            .iter()
            .all(|(k, _)| *k != MaskKind::Object)
    );

    let d = disc_photo();
    let opened = core.session.open(
        &[d.dir.path().to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    let id = opened.added[0];
    core.session.set_current(id).expect("current");
    core.renderer.navigate(id);

    // No Object row, no encoder.
    assert!(!core.renderer.object_preparing(id));
    // The patch that makes the row starts the encoder in the background.
    let patch: SettingsPatch = serde_json::from_value(
        json!({ "nodes": { "1": { "masks": { "items": [ { "kind": "object" } ] } } } }),
    )
    .expect("a patch");
    core.update_settings(id, &patch).expect("the row is added");
    assert!(
        core.renderer.object_preparing(id),
        "the core asked for the features"
    );
    let before = core.session.settings(id).expect("settings").settings;
    // A click waits for it and answers the row's new points; the settings are the person's to change, not the find's.
    let found = core
        .mask_find(id, 1, 0, point(0.45, 0.5, true), &CancelToken::new())
        .expect("answer");
    assert!(matches!(found, MaskFound::Found { ref points, .. } if points == &[(0.45, 0.5, true)]));
    assert_eq!(
        core.session.settings(id).expect("settings").settings,
        before
    );
    // The page applies the answer as one patch; the row then has its point.
    let apply: SettingsPatch = serde_json::from_value(json!({ "nodes": { "1": { "masks": { "edit": { "0": { "points": [[0.45, 0.5, true]] } } } } } }))
        .expect("a patch");
    core.update_settings(id, &apply).expect("applied");
    assert_eq!(core.renderer.object_encodes(id), 1);
    // Nothing found changes nothing either: a prompt the decoder refuses leaves the settings as they are.
    let tiny = core
        .mask_find(
            id,
            1,
            0,
            ObjectPrompt::Box {
                x0: 0.5,
                y0: 0.5,
                x1: 0.5005,
                y1: 0.5005,
            },
            &CancelToken::new(),
        )
        .expect("answer");
    eprintln!("[object] a box of a hundredth of a percent: {tiny:?}");
    let after = core.session.settings(id).expect("settings").settings;
    let MaskFound::NothingFound(_) = tiny else {
        // The model may find something in a box that small; the settings are untouched either way.
        assert_eq!(after.main().masks.items.len(), 1);
        return;
    };
    assert_eq!(after.main().masks.items.len(), 1);
}

// ---- exports ---------------------------------------------------------------------------------------------

fn export_image(r: &Renderer, src: &PhotoSource) -> ym_core::pipeline::FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

#[test]
fn an_export_with_an_object_row_is_the_same_bytes_in_two_renderers_and_in_tiles_and_acts() {
    let Some(model) = model() else { return };
    let d = disc_photo();
    let row = object_row(
        json!({ "points": [[0.45, 0.5, true], [0.3, 0.3, false]], "grow": 0.2, "feather": 0.2, "refine": 0.6 }),
    );
    let src = d.source().with_settings(graph(vec![row]));
    let whole = export_image(&renderer_with(Some(Arc::clone(&model)), 4096), &src);
    // Another renderer: the same bytes (the encoder, the decoder and the picture are deterministic, and the export makes its own).
    let again = export_image(&renderer_with(Some(Arc::clone(&model)), 4096), &src);
    assert!(
        again.rgb == whole.rgb,
        "a second renderer gives other bytes"
    );
    // Tiles of an export equal the whole frame bit for bit, the halos of Grow, Feather and Refine edges included.
    let tiled = export_image(&renderer_with(Some(Arc::clone(&model)), 192), &src);
    assert!(
        tiled.rgb == whole.rgb,
        "tiles of 192 px differ from the whole frame"
    );
    // The row acts: the same node with no point is another picture (the trim of a stop reaches only the object).
    let bare = d.source().with_settings(graph(vec![object_row(json!({}))]));
    let none = export_image(&renderer_with(Some(model), 4096), &bare);
    assert!(none.rgb != whole.rgb);
}

#[test]
fn an_export_of_an_object_row_without_the_model_is_the_mask_error() {
    let d = disc_photo();
    let src = d.source().with_settings(graph(vec![object_row(
        json!({ "points": [[0.45, 0.5, true]] }),
    )]));
    let r = renderer_with(None, 4096);
    assert_eq!(
        r.unmakeable_mask(&src.settings).as_deref(),
        Some("This copy of Yellowmoon has no Object model. Reinstall the app.")
    );
}

// ---- the device ------------------------------------------------------------------------------------------

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
                "\n* GPU TESTS SKIPPED in mask_object: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// A frame and a Mask only frame on the device and on the CPU, with an Object row in them: the plane is the model's, so what is compared is the
/// raster pass, the colour pass and the encode on the device against the CPU reference, within the bounds of N-13.
#[test]
fn the_device_makes_the_same_frames_and_masks_as_the_cpu_with_an_object_row() {
    let Some(model) = model() else { return };
    let Some(gpu) = gpu() else { return };
    let d = disc_photo();
    let src = d.source();
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL
                .difference(GpuStages::RAW_CHAIN)
                .difference(GpuStages::RAW_FRONT),
            object_model: Some(Arc::clone(&model)),
            background: PAPER,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let cpu = renderer(model);
    let picture = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    for (what, extra) in [
        ("a point", json!({ "points": [[0.45, 0.5, true]] })),
        (
            "a point and a part taken away",
            json!({ "points": [[0.45, 0.5, true], [0.4, 0.45, false]] }),
        ),
        ("a box", json!({ "box": [0.25, 0.2, 0.65, 0.8] })),
        (
            "all of it",
            json!({ "points": [[0.45, 0.5, true]], "grow": 0.3, "feather": 0.3, "refine": 0.7 }),
        ),
    ] {
        let s = with_row_json(&src, extra);
        for r in [&cpu, &on_gpu] {
            r.object_prepare(&s);
            wait_until("the features", 180, || r.object_ready(&s));
        }
        for (name, st) in [("picture", &picture), ("mask only", &mask_view(2))] {
            let device = frame(&on_gpu, &s, st);
            assert_eq!(device.report.device, ComputeDevice::Gpu, "{what}, {name}");
            let reference = frame(&cpu, &s, st);
            if name == "mask only" {
                let m = coverage(&reference);
                assert!(m > 0.005 && m < 0.9, "{what}: the mask is flat ({m})");
            }
            let (mut worst, mut sum) = (0u8, 0u64);
            for (x, y) in device.rgba.iter().zip(&reference.rgba) {
                let d = x.abs_diff(*y);
                worst = worst.max(d);
                sum += u64::from(d);
            }
            let mean = sum as f64 / device.rgba.len() as f64;
            eprintln!(
                "[object] {what}, {name}: the device differs from the CPU by {worst} codes, mean {mean:.4}"
            );
            assert!(
                worst <= 3 && mean < 0.3,
                "{what}, {name}: {worst} codes, mean {mean}"
            );
        }
    }
}

fn with_row_json(src: &PhotoSource, extra: Value) -> PhotoSource {
    src.with_settings(graph(vec![object_row(extra)]))
}

// ---- the encoder's stop ---------------------------------------------------------------------------------

#[test]
fn the_encoder_stops_between_its_blocks_when_asked_and_gives_nothing() {
    let Some(model) = model() else { return };
    let sam = model.sam().expect("model");
    let grey = vec![128u8; 3 * 8 * 8];
    let picture = Picture::new(8, 8, &grey).expect("a picture");
    let t = Instant::now();
    let stopped = sam.encode_cancellable(&picture, &|| true);
    assert!(
        matches!(stopped, Err(ym_ml::model::MlError::Cancelled)),
        "{stopped:?}"
    );
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "the stop was noticed before the first block finished ({:?})",
        t.elapsed()
    );
}
