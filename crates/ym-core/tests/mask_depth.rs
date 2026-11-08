//! Depth masks through the render graph (`docs/23-nodes.md` sections 3.10a, 4.8 and 9, `docs/algorithms/06-masks.md`
//! sections 5.8 and 8, task D-04): a Depth row's raster `S0` is its gradient cut from the photo's depth map, which a background
//! worker makes once per photo from the Input's picture (debounced, cancellable, never on disk); Pick depth and Focus plane
//! read the cached map; the plane goes through the geometry and the raster steps as Background's does; an export makes its
//! own map on the CPU.
//!
//! These tests run on a **fake model** (`DepthModel::fake`), so they need no weights. The real network is checked against
//! its oracle in `crates/ym-ml/tests/depth.rs`; the app-and-`ym` parity with the real weights is in
//! `crates/ym-cli/tests/parity.rs` (it says so and passes without them).
mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    DepthModel, DepthPhase, FocusPlane, FocusReason, FrameOptions, FullRequest, FullTarget,
    MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderError, RenderOptions, RenderedFrame,
    Renderer, ViewportRect, ViewportState,
};
use ym_core::session::PhotoId;
use ym_core::settings::{
    DEPTH_ROW_NO_MODEL_SENTENCE, DevelopSettings, MaskItem, MaskKind, MaskRowPatch, MaskShape,
    MaskSupport, MaskUnavailable, NO_DEPTH_MODEL_SENTENCE, Orientation, mask_kinds_with,
    mask_unavailable_with,
};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The paper around a photo in a Mask only frame: nothing in a mask is this colour.
const PAPER: [u8; 3] = [40, 60, 90];
const CELL: f64 = 300.0;

// ---- fakes ---------------------------------------------------------------------------------------------------------

/// What a fake model did: a count of its runs and the pictures it was given.
#[derive(Default)]
struct Spy {
    runs: AtomicUsize,
    cancelled: AtomicBool,
    started: AtomicBool,
    pictures: Mutex<Vec<(usize, usize, Vec<u8>)>>,
    /// The fake works this long (a ms at a time, asking for a cancel), before it answers.
    delay_ms: AtomicUsize,
    /// The fake fails with this text.
    fail: Mutex<Option<String>>,
    /// The fake answers a NaN.
    nan: AtomicBool,
    /// The fake's depth is 1 for the bright parts of the picture and 0 for the rest (two groups of equal cells).
    binary: AtomicBool,
}

/// One cell for every 14 pixels of the network's input, as the real map is: the depth of a cell is the luma of the picture
/// at its centre (brighter is nearer), plus `row_slope` toward the bottom of the photo.
fn cells(w: usize, h: usize, rgb: &[u8], row_slope: f64) -> (usize, usize, Vec<f32>) {
    let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
    let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
    let mut d = Vec::with_capacity(cw * ch);
    for cy in 0..ch {
        for cx in 0..cw {
            let x = (((cx as f64 + 0.5) / cw as f64) * w as f64) as usize;
            let y = (((cy as f64 + 0.5) / ch as f64) * h as f64) as usize;
            let p = &rgb[3 * (y.min(h - 1) * w + x.min(w - 1))..][..3];
            let luma =
                (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
                    / 255.0;
            d.push((luma + row_slope * (cy as f64 + 0.5) / ch as f64) as f32);
        }
    }
    (cw, ch, d)
}

/// A fake model that spies on what it is asked, with the picture's luma (and a slope toward the bottom) as its depth.
fn spy_model(spy: &Arc<Spy>, row_slope: f64) -> Arc<DepthModel> {
    let spy = Arc::clone(spy);
    DepthModel::fake(
        &format!("spy{row_slope}"),
        Arc::new(move |w, h, rgb, cancel| {
            spy.runs.fetch_add(1, Ordering::SeqCst);
            spy.started.store(true, Ordering::SeqCst);
            spy.pictures.lock().unwrap().push((w, h, rgb.to_vec()));
            let until =
                Instant::now() + Duration::from_millis(spy.delay_ms.load(Ordering::SeqCst) as u64);
            while Instant::now() < until {
                if cancel() {
                    spy.cancelled.store(true, Ordering::SeqCst);
                    return Err("cancelled".to_owned());
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            if let Some(why) = spy.fail.lock().unwrap().clone() {
                return Err(why);
            }
            let (cw, ch, mut d) = cells(w, h, rgb, row_slope);
            if spy.binary.load(Ordering::SeqCst) {
                let (lo, hi) = d
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                for v in &mut d {
                    *v = if *v > 0.5 * (lo + hi) { 1.0 } else { 0.0 };
                }
            }
            if spy.nan.load(Ordering::SeqCst) {
                d[0] = f32::NAN;
            }
            Ok((cw, ch, d))
        }),
    )
}

/// A fake whose depth is a ramp from the left (near) to the right (far) of the photo, to see the geometry in the mask.
fn ramp_model() -> Arc<DepthModel> {
    DepthModel::fake(
        "ramp",
        Arc::new(|w, h, _, _| {
            let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
            let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
            let d = (0..ch)
                .flat_map(|_| (0..cw).map(move |x| 1.0 - (x as f32 + 0.5) / cw as f32))
                .collect();
            Ok((cw, ch, d))
        }),
    )
}

fn options(model: Option<Arc<DepthModel>>, tile: usize) -> RenderOptions {
    RenderOptions {
        depth_model: model,
        data_root: Some(data_root()),
        background: PAPER,
        full_tile_px: tile,
        ..RenderOptions::default()
    }
}

fn renderer(model: Arc<DepthModel>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        options(Some(model), RenderOptions::default().full_tile_px),
    )
    .expect("render thread")
}

fn renderer_without_model() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options(None, 1024)).expect("render thread")
}

/// Everything the renderer's listener hears, in order.
fn listen(r: &Renderer) -> Arc<Mutex<Vec<DepthPhase>>> {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&heard);
    r.set_depth_listener(Arc::new(move |_, phase| sink.lock().unwrap().push(phase)));
    heard
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Waits for the worker of the photo to end its work.
fn settled(r: &Renderer, src: &PhotoSource) {
    wait_until("the depth worker", 60, || {
        !r.depth_worker_alive(src.id) && r.depth_ready(src)
    });
}

/// A spy whose depth is two groups of equal cells: the bright square, which is the nearest, and the rest.
fn binary_spy() -> Arc<Spy> {
    let spy = Arc::new(Spy::default());
    spy.binary.store(true, Ordering::SeqCst);
    spy
}

// ---- photos and documents -------------------------------------------------------------------------------------------

/// A photo of 1200 x 800 with a bright square at (0.3, 0.7) of the sensor: `half` is its half side as a share of the width
/// and height. The camera's focus tags and orientation are the file's.
fn photo(orientation: u16, focus: Option<([u16; 4], Option<[u16; 3]>)>) -> Developed {
    let (w, h) = (1200u32, 800u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.08).noise(0.0, 0.0);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.6 * (scene.white_level - scene.black)) as u16;
    let (cx, cy, half) = (0.3 * f64::from(w), 0.7 * f64::from(h), 0.07);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (f64::from(x) + 0.5 - cx, f64::from(y) + 0.5 - cy);
            if dx.abs() <= half * f64::from(w) && dy.abs() <= half * f64::from(h) {
                mosaic[(y * w + x) as usize] = code;
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let mut builder = scene.builder(MODEL).orientation(orientation);
    if let Some((location, size)) = focus {
        builder = builder.focus_location(location);
        if let Some(size) = size {
            builder = builder.focus_frame_size(size);
        }
    }
    builder
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
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

fn depth_row(extra: Value) -> Value {
    let mut row = json!({ "kind": "depth", "mode": "add" });
    if let (Some(r), Some(e)) = (row.as_object_mut(), extra.as_object()) {
        r.extend(e.clone());
    }
    row
}

fn with_rows(src: &PhotoSource, rows: Vec<Value>) -> PhotoSource {
    src.with_settings(graph(rows))
}

fn edit_depth(src: &PhotoSource, f: impl FnOnce(&mut ym_core::settings::DepthMask)) -> PhotoSource {
    let mut s = src.settings.clone();
    let row = s
        .nodes
        .corrector_mut(2)
        .expect("node 2")
        .masks
        .items
        .iter_mut()
        .find_map(|r| match &mut r.shape {
            MaskShape::Depth(d) => Some(d),
            _ => None,
        })
        .expect("a Depth row");
    f(row);
    src.with_settings(s)
}

fn mask_view(node: u32) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
    });
    st
}

fn picture_view() -> ViewportState {
    ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0))
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

/// The mean of the mask of a Mask only frame over the photo's pixels (the paper is left out).
fn coverage(f: &RenderedFrame) -> f64 {
    let v: Vec<f64> = f
        .rgba
        .chunks(4)
        .filter(|p| p[..3] != PAPER)
        .map(|p| f64::from(p[0]) / 255.0)
        .collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// The mean of the mask over the photo's pixels in a region of the cell (`x0..x1`, `y0..y1`, shares of the cell).
fn coverage_in(f: &RenderedFrame, (x0, x1): (f64, f64), (y0, y1): (f64, f64)) -> f64 {
    let w = CELL as usize;
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in (y0 * CELL) as usize..(y1 * CELL) as usize {
        for x in (x0 * CELL) as usize..(x1 * CELL) as usize {
            let p = &f.rgba[4 * (y * w + x)..][..4];
            if p[..3] != PAPER {
                sum += f64::from(p[0]) / 255.0;
                n += 1.0;
            }
        }
    }
    sum / f64::max(n, 1.0)
}

// ---- the kind and the sentence -----------------------------------------------------------------------------------

#[test]
fn depth_is_a_kind_where_the_model_is_and_a_mask_error_where_it_is_not() {
    // The lists: Depth right after Background where the weights are, `weights_missing` where they are not.
    let has = MaskSupport::bare().with_depth(true);
    let lacks = MaskSupport::bare();
    let with = mask_kinds_with(has);
    let at = with
        .iter()
        .position(|k| *k == MaskKind::Background)
        .expect("Background");
    assert_eq!(with[at + 1], MaskKind::Depth);
    assert!(!mask_kinds_with(lacks).contains(&MaskKind::Depth));
    let why = mask_unavailable_with(lacks)
        .into_iter()
        .find(|(k, _)| *k == MaskKind::Depth)
        .expect("Depth is listed")
        .1;
    assert_eq!(why, MaskUnavailable::DepthWeightsMissing);
    assert_eq!(why.wire(), "weights_missing");
    assert_eq!(why.doctor_text(false), "the depth weights are missing");
    assert!(
        mask_unavailable_with(has)
            .iter()
            .all(|(k, _)| *k != MaskKind::Depth)
    );
    // The renderer says the same, and refuses the row without the model with the sentence of `docs/23` section 3.10a.
    let r = renderer(DepthModel::fake_default());
    let none = renderer_without_model();
    assert_eq!(
        r.mask_kinds(),
        mask_kinds_with(
            MaskSupport::bare()
                .with_depth(true)
                .with_vision(r.vision_support())
        )
    );
    assert!(!none.mask_kinds().contains(&MaskKind::Depth));
    let row = graph(vec![depth_row(json!({}))]);
    assert_eq!(r.unmakeable_mask(&row), None);
    assert_eq!(
        none.unmakeable_mask(&row).as_deref(),
        Some("This copy of Yellowmoon has no Depth model. Reinstall the app.")
    );
    assert_eq!(NO_DEPTH_MODEL_SENTENCE, none.unmakeable_mask(&row).unwrap());
    assert_eq!(
        DEPTH_ROW_NO_MODEL_SENTENCE,
        "Depth masks need the depth model. Reinstall the app."
    );
    // A row with the eye off is not in the picture; neither is a row of a node the Output does not need.
    let off = graph(vec![depth_row(json!({ "enabled": false }))]);
    assert_eq!(none.unmakeable_mask(&off), None);
    // The bare machine of the free function agrees.
    assert_eq!(
        ym_core::pipeline::unmakeable_mask(&row).as_deref(),
        Some(NO_DEPTH_MODEL_SENTENCE)
    );
    // Sample and Focus without the model are the same sentence, with the `mask` scope.
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    for err in [
        none.sample_depth(&src, 0.5, 0.5, &CancelToken::new())
            .unwrap_err(),
        none.focus_plane(&src, &CancelToken::new()).unwrap_err(),
    ] {
        assert!(matches!(err, RenderError::Mask(ref m) if m == NO_DEPTH_MODEL_SENTENCE));
    }
    // A view shows the row selecting nothing, with no worker: the photo as the session has it.
    let plain = frame(&none, &d.source(), &picture_view());
    let empty = frame(&none, &src, &picture_view());
    assert_eq!(empty.rgba, plain.rgba);
    assert!(!none.depth_working(src.id) && !none.depth_worker_alive(src.id));
}

// ---- the row in the settings -------------------------------------------------------------------------------------

#[test]
fn a_depth_row_round_trips_with_its_defaults_and_is_checked() {
    // `{"kind": "depth"}` is the defaults of the spec.
    let item: MaskItem = serde_json::from_value(json!({ "kind": "depth" })).expect("a row");
    let MaskShape::Depth(d) = &item.shape else {
        panic!("not a Depth row")
    };
    assert_eq!((d.far, d.feather, d.grow, d.refine), (1.0, 0.0, 0.0, 0.5));
    let written = serde_json::to_value(&item).expect("json");
    assert_eq!(
        written,
        json!({ "kind": "depth", "far": 1.0, "feather": 0.0, "grow": 0.0,
                "refine": 0.5, "mode": "add", "invert": false, "enabled": true, "opacity": 1.0 })
    );
    let again: MaskItem = serde_json::from_value(written).expect("round trip");
    assert_eq!(again, item);
    // The document form of `docs/10`.
    let full: MaskItem = serde_json::from_value(json!({ "kind": "depth", "far": 0.35,
        "feather": 0.3, "grow": -0.2, "refine": 0.9, "invert": true, "mode": "subtract" }))
    .expect("a row");
    assert_eq!(
        serde_json::from_value::<MaskItem>(serde_json::to_value(&full).unwrap()).unwrap(),
        full
    );
    // A field another kind has, or none the kind has, is refused with the field's place.
    for bad in [
        json!({ "kind": "depth", "level": 0.5 }),
        json!({ "kind": "depth", "points": [] }),
        json!({ "kind": "depth", "nera": 0.5 }),
        // The band is gone: a document that still has `near` or `softness` on a Depth row is refused like any unknown field.
        json!({ "kind": "depth", "near": 0.1 }),
        json!({ "kind": "depth", "softness": 0.5 }),
    ] {
        assert!(serde_json::from_value::<MaskItem>(bad).is_err());
    }
    // The kind is one of the ten.
    assert_eq!(ym_core::settings::MASK_KINDS.len(), 10);
    assert_eq!(ym_core::settings::MASK_KINDS[4], "depth");
    assert_eq!(MaskKind::ALL[4], MaskKind::Depth);
    assert_eq!(MaskKind::Depth.name(), "depth");
    assert_eq!(MaskKind::Depth.label(), "Depth");
    assert_eq!(MaskKind::of(&item.shape), MaskKind::Depth);
    assert_eq!(
        MaskShape::default_of("depth"),
        Some(MaskShape::Depth(Default::default()))
    );
    // The numbers are clamped to their ranges: the far limit to 0.02 to 1.
    let (s, changed) = serde_json::from_value::<DevelopSettings>(json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "masks": { "invert": false, "items": [
            { "kind": "depth", "far": 1.5, "feather": 3.0, "grow": -4.0, "refine": 9.0 },
            { "kind": "depth", "far": 0.0 },
        ] } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } }))
    .expect("a document")
    .validated()
    .expect("valid");
    assert!(!changed.is_empty());
    let rows = &s.nodes.corrector(2).expect("node 2").masks.items;
    let (MaskShape::Depth(a), MaskShape::Depth(b)) = (&rows[0].shape, &rows[1].shape) else {
        panic!("rows")
    };
    assert_eq!((a.far, a.feather, a.grow, a.refine), (1.0, 1.0, -1.0, 1.0));
    assert_eq!(b.far, 0.02);
}

#[test]
fn a_row_patch_sets_far_and_refuses_the_other_kinds_fields_and_the_old_band() {
    let row = MaskItem::new(MaskShape::Depth(Default::default()));
    let doc = graph(vec![serde_json::to_value(&row).unwrap()]);
    // The patch `masks.edit` of the app: a row by its index.
    let mut s = doc.clone();
    let patch: ym_core::settings::SettingsPatch = serde_json::from_value(json!({ "nodes": { "2": {
        "masks": { "edit": { "0": { "far": 0.4, "feather": 0.1, "invert": true } } } } } }))
    .expect("a settings patch");
    s.apply_patch(&patch, None).expect("applies");
    let row = &s.nodes.corrector(2).unwrap().masks.items[0];
    let MaskShape::Depth(d) = &row.shape else {
        panic!("a Depth row")
    };
    assert_eq!((d.far, d.feather), (0.4, 0.1));
    assert!(row.invert);
    // A far limit under the least is held to it (no band, so no swap).
    let low: ym_core::settings::SettingsPatch = serde_json::from_value(json!({ "nodes": { "2": {
        "masks": { "edit": { "0": { "far": 0.001 } } } } } }))
    .expect("a settings patch");
    s.apply_patch(&low, None).expect("applies");
    let MaskShape::Depth(d) = &s.nodes.corrector(2).unwrap().masks.items[0].shape else {
        panic!("a Depth row")
    };
    assert_eq!(d.far, 0.02);
    // `near` is not a field of any row any more (the patch is refused as it is read); `softness` is still the field of other
    // kinds, and a Depth row refuses it with the plain sentence.
    assert!(
        serde_json::from_value::<ym_core::settings::SettingsPatch>(json!({ "nodes": { "2": {
            "masks": { "edit": { "0": { "near": 0.3 } } } } } }))
        .is_err()
    );
    let soft: ym_core::settings::SettingsPatch = serde_json::from_value(json!({ "nodes": { "2": {
        "masks": { "edit": { "0": { "softness": 0.3 } } } } } }))
    .expect("a settings patch");
    let err = doc.clone().apply_patch(&soft, None).unwrap_err();
    assert_eq!(
        err.to_string(),
        "A Depth mask has no field called “softness”."
    );
    // Another kind's field is refused with the plain sentence.
    let bad: ym_core::settings::SettingsPatch = serde_json::from_value(json!({ "nodes": { "2": {
        "masks": { "edit": { "0": { "level": 0.3 } } } } } }))
    .expect("a settings patch");
    let err = doc.clone().apply_patch(&bad, None).unwrap_err();
    assert_eq!(err.to_string(), "A Depth mask has no field called “level”.");
    // And a Background row has no `far`.
    let mut bg = graph(vec![json!({ "kind": "background" })]);
    let bad: ym_core::settings::SettingsPatch = serde_json::from_value(json!({ "nodes": { "2": {
        "masks": { "edit": { "0": { "far": 0.3 } } } } } }))
    .expect("a settings patch");
    assert!(bg.apply_patch(&bad, None).is_err());
    // The row patch type reads the same fields.
    let row_patch: MaskRowPatch =
        serde_json::from_value(json!({ "far": 0.4 })).expect("a row patch");
    assert_eq!(row_patch.far, Some(0.4));
    // The old band's fields are not in it.
    assert!(serde_json::from_value::<MaskRowPatch>(json!({ "near": 0.1 })).is_err());
}

// ---- the key follows the Input, not a node ------------------------------------------------------------------------

#[test]
fn the_maps_key_follows_the_inputs_picture_and_the_model_and_not_a_nodes_grade() {
    let r = renderer(DepthModel::fake_default());
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    let base = r.depth_key(&src).expect("a key");
    let same = |s: &PhotoSource, what: &str| {
        assert_eq!(
            r.depth_key(s).expect("a key"),
            base,
            "{what} must not change the key"
        );
    };
    let moves = |s: &PhotoSource, what: &str| {
        assert_ne!(
            r.depth_key(s).expect("a key"),
            base,
            "{what} must change the key"
        );
    };
    // Does not change: a node's trim, grade, curve and masks (the Depth row's own fields included), the Output's look and
    // output sharpening, the crop, the straightening, the quarter turn, the lens character, Feather, Grow, Refine, Opacity,
    // Invert and Mode.
    let edit = |f: &dyn Fn(&mut DevelopSettings)| {
        let mut s = src.settings.clone();
        f(&mut s);
        src.with_settings(s)
    };
    same(
        &edit(&|s| s.nodes.corrector_mut(2).unwrap().trim.exposure_ev = -2.0),
        "a node's trim",
    );
    same(&edit(&|s| s.output_mut().look_intensity = 40.0), "the look");
    same(
        &edit(&|s| s.output_mut().output_sharpening = 80.0),
        "the output sharpening",
    );
    same(
        &edit(&|s| s.input_mut().orientation = Orientation::Deg90),
        "the quarter turn",
    );
    same(
        &edit(&|s| {
            s.input_mut().crop = Some(ym_core::settings::Crop {
                rect: ym_core::settings::NormRect {
                    x: 0.1,
                    y: 0.1,
                    width: 0.5,
                    height: 0.5,
                },
                angle_deg: 3.0,
                aspect: ym_core::settings::CropAspect::Free,
                auto: false,
            })
        }),
        "the crop and the straightening",
    );
    same(
        &edit_depth(&src, |d| {
            d.far = 0.9;
            d.feather = 0.4;
            d.grow = 0.2;
            d.refine = 0.1;
        }),
        "the row's own fields",
    );
    {
        let mut s = src.settings.clone();
        let row = &mut s.nodes.corrector_mut(2).unwrap().masks.items[0];
        row.invert = true;
        row.opacity = 0.4;
        row.mode = ym_core::settings::MaskMode::Subtract;
        same(&src.with_settings(s), "Invert, Opacity and Mode");
    }
    // Changes: the Input's exposure, the model.
    moves(
        &edit(&|s| s.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(1.0)),
        "the Input's exposure",
    );
    let other_model = renderer(ramp_model());
    assert_ne!(other_model.depth_key(&src).unwrap(), base, "the weights");
    // A different photo (here, one the camera turned another way) is a different map.
    let e = photo(6, None);
    let other = with_rows(&e.source(), vec![depth_row(json!({}))]);
    moves(&other, "a different photo");
}

// ---- the worker ---------------------------------------------------------------------------------------------------

#[test]
fn a_viewer_never_waits_the_row_selects_nothing_until_the_map_is_made() {
    let spy = Arc::new(Spy::default());
    spy.delay_ms.store(300, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    let heard = listen(&r);
    let d = photo(1, None);
    let plain = frame(&r, &d.source(), &mask_view(1));
    let src = with_rows(&d.source(), vec![depth_row(json!({ "far": 1.0 }))]);
    let t = Instant::now();
    let first = frame(&r, &src, &picture_view());
    assert!(
        t.elapsed() < Duration::from_secs(20),
        "the frame waited ({:?})",
        t.elapsed()
    );
    // The row selected nothing: the photo as the session has it (the trim of node 2 is under the empty mask).
    let none_yet = frame(&r, &src, &mask_view(2));
    assert_eq!(coverage(&none_yet), 0.0, "an empty mask");
    assert_eq!(first.rgba, frame(&r, &d.source(), &picture_view()).rgba);
    // The worker was asked: Waiting (debounce), Running ("Finding…"), Ready.
    assert!(
        r.depth_worker_alive(src.id),
        "the worker was asked for the map"
    );
    assert!(!r.depth_ready(&src));
    wait_until("the worker to start", 20, || r.depth_working(src.id));
    assert!(
        r.finding_masks(src.id, &src.settings),
        "the app says Finding…"
    );
    assert_eq!(heard.lock().unwrap().first(), Some(&DepthPhase::Started));
    settled(&r, &src);
    assert!(!r.depth_working(src.id));
    assert!(!r.finding_masks(src.id, &src.settings));
    assert_eq!(
        *heard.lock().unwrap(),
        [DepthPhase::Started, DepthPhase::Finished { failed: None }]
    );
    // With the map there, the row selects something: the gradient over the whole depth, a mean of about one half.
    let made = frame(&r, &src, &mask_view(2));
    let c = coverage(&made);
    assert!(c > 0.3 && c < 0.7, "about half the photo: {c}");
    assert_ne!(
        first.rgba,
        frame(&r, &src, &picture_view()).rgba,
        "the picture is graded where the mask is"
    );
    assert_eq!(spy.runs.load(Ordering::SeqCst), 1);
    assert_eq!(r.depth_networks(src.id), 1);
    let _ = plain;
}

#[test]
fn a_stream_of_wishes_makes_one_map_of_the_last_picture() {
    let spy = Arc::new(Spy::default());
    let r = renderer(spy_model(&spy, 0.0));
    let heard = listen(&r);
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    // A drag of the Input's exposure asks for a new map on every tick.
    let mut last = src.clone();
    for i in 0..6 {
        let mut s = src.settings.clone();
        s.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(0.1 * f64::from(i));
        last = src.with_settings(s);
        r.depth_prepare(&last);
    }
    assert_eq!(r.depth_workers(src.id), 1, "one worker");
    settled(&r, &last);
    assert_eq!(
        spy.runs.load(Ordering::SeqCst),
        1,
        "one network run, for the last picture"
    );
    assert_eq!(r.depth_networks(src.id), 1);
    assert_eq!(heard.lock().unwrap().len(), 2, "Started and Finished once");
    // A wish for a map the photo has starts nothing.
    r.depth_prepare(&last);
    assert!(!r.depth_worker_alive(src.id));
    assert_eq!(r.depth_workers(src.id), 1);
}

#[test]
fn leaving_the_photo_stops_a_run_nobody_waits_for() {
    let spy = Arc::new(Spy::default());
    spy.delay_ms.store(20_000, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    let heard = listen(&r);
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    r.navigate(src.id);
    r.depth_prepare(&src);
    wait_until("the network to start", 30, || {
        spy.started.load(Ordering::SeqCst)
    });
    assert!(r.depth_working(src.id));
    // The person goes to another photo.
    r.navigate(PhotoId(src.id.0 + 1000));
    wait_until("the worker to end", 30, || !r.depth_worker_alive(src.id));
    assert!(
        spy.cancelled.load(Ordering::SeqCst),
        "the network was asked to stop"
    );
    assert!(!r.depth_ready(&src), "no map");
    assert!(!r.depth_working(src.id));
    assert_eq!(r.depth_networks(src.id), 0);
    assert_eq!(
        *heard.lock().unwrap(),
        [DepthPhase::Started, DepthPhase::Finished { failed: None }]
    );
    // Nothing is wanted for the photo any more: coming back asks again.
    spy.delay_ms.store(0, Ordering::SeqCst);
    r.navigate(src.id);
    r.depth_prepare(&src);
    settled(&r, &src);
    assert_eq!(r.depth_networks(src.id), 1);
}

#[test]
fn a_newer_wish_stops_the_run_for_the_older_picture() {
    let spy = Arc::new(Spy::default());
    spy.delay_ms.store(20_000, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    let d = photo(1, None);
    let first = with_rows(&d.source(), vec![depth_row(json!({}))]);
    r.depth_prepare(&first);
    wait_until("the network to start", 30, || {
        spy.started.load(Ordering::SeqCst)
    });
    spy.delay_ms.store(0, Ordering::SeqCst);
    let mut s = first.settings.clone();
    s.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(1.0);
    let second = first.with_settings(s);
    r.depth_prepare(&second);
    settled(&r, &second);
    assert!(
        spy.cancelled.load(Ordering::SeqCst),
        "the first run was stopped"
    );
    assert_eq!(spy.runs.load(Ordering::SeqCst), 2);
    assert!(r.depth_ready(&second));
    assert!(
        !r.depth_ready(&first),
        "the photo has the newer picture's map"
    );
}

#[test]
fn a_map_that_cannot_be_made_is_kept_as_a_failure_a_viewer_does_not_ask_again_and_an_export_fails_with_the_mask_error()
 {
    let spy = Arc::new(Spy::default());
    *spy.fail.lock().unwrap() = Some("the network fell over".to_owned());
    let r = renderer(spy_model(&spy, 0.0));
    let heard = listen(&r);
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    let _ = frame(&r, &src, &picture_view());
    wait_until("the worker to end", 30, || {
        !r.depth_worker_alive(src.id)
            && !heard.lock().unwrap().is_empty()
            && r.depth_failure(&src).is_some()
    });
    wait_until("Finished", 30, || heard.lock().unwrap().len() == 2);
    let sentence = "The Depth mask can't be made: the network fell over.";
    assert_eq!(r.depth_failure(&src).as_deref(), Some(sentence));
    assert_eq!(
        *heard.lock().unwrap(),
        [
            DepthPhase::Started,
            DepthPhase::Finished {
                failed: Some(sentence.to_owned())
            }
        ]
    );
    assert!(!r.depth_ready(&src));
    // The row selects nothing, and further frames ask for nothing more.
    for _ in 0..3 {
        assert_eq!(coverage(&frame(&r, &src, &mask_view(2))), 0.0);
    }
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(spy.runs.load(Ordering::SeqCst), 1, "not asked again");
    assert!(!r.depth_worker_alive(src.id));
    // Pick depth and Focus plane say there is no map (`null`, `map_failed`).
    assert_eq!(
        r.sample_depth(&src, 0.5, 0.5, &CancelToken::new()).unwrap(),
        None
    );
    // An export fails with the `mask` error and the same sentence.
    let err = r
        .render_full(&src, &export_request(), &CancelToken::new())
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(err, RenderError::Mask(ref m) if m == sentence),
        "{err:?}"
    );
    // A key that changes is asked again: the Input's exposure.
    *spy.fail.lock().unwrap() = None;
    let mut s = src.settings.clone();
    s.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(1.0);
    let brighter = src.with_settings(s);
    r.depth_prepare(&brighter);
    settled(&r, &brighter);
    assert_eq!(r.depth_failure(&brighter), None);
}

/// The request of an export to sRGB display values.
fn export_request() -> FullRequest {
    FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    }
}

#[test]
fn a_map_with_a_value_that_is_not_a_number_is_refused() {
    let spy = Arc::new(Spy::default());
    spy.nan.store(true, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    let err = r.depth_map(&src, &CancelToken::new()).unwrap_err();
    assert!(
        matches!(err, RenderError::Mask(ref m) if m.starts_with("The Depth mask can't be made:")),
        "{err:?}"
    );
    assert!(r.depth_failure(&src).is_some());
}

#[test]
fn one_map_serves_two_depth_rows_and_a_second_node_and_the_picture_is_made_once() {
    let spy = Arc::new(Spy::default());
    let r = renderer(spy_model(&spy, 0.0));
    let d = photo(1, None);
    // Node 2 has two Depth rows; node 3 has one of its own, with another far limit.
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": [
            depth_row(json!({ "far": 0.3 })), depth_row(json!({ "far": 0.6 })) ] } },
        { "id": 3, "kind": "corrector", "pos": [2.0, 0.0], "trim": { "exposure_ev": -1.0 },
          "masks": { "invert": false, "items": [ depth_row(json!({ "far": 0.9 })) ] } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("valid")
        .0;
    let src = d.source().with_settings(settings);
    let _ = frame(&r, &src, &picture_view());
    settled(&r, &src);
    let before = r.object_pictures_made();
    let a = frame(&r, &src, &picture_view());
    let b2 = coverage(&frame(&r, &src, &mask_view(2)));
    let b3 = coverage(&frame(&r, &src, &mask_view(3)));
    assert!(
        b2 > 0.2 && b3 > 0.2,
        "both nodes have their masks: {b2} {b3}"
    );
    assert!((b2 - b3).abs() > 1e-3, "with their own limits: {b2} {b3}");
    assert_eq!(
        spy.runs.load(Ordering::SeqCst),
        1,
        "one network run for the photo"
    );
    assert_eq!(r.depth_networks(src.id), 1);
    assert_eq!(
        r.object_pictures_made(),
        before,
        "the frames made no picture of their own"
    );
    assert_eq!(r.object_pictures_made(), 1, "the picture was made once");
    assert_eq!(spy.pictures.lock().unwrap().len(), 1);
    drop(a);
}

#[test]
fn the_sliders_cut_a_new_plane_from_the_cached_ranks_and_never_run_the_network() {
    let spy = Arc::new(Spy::default());
    let r = renderer(spy_model(&spy, 0.0));
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({ "far": 0.4 }))]);
    let _ = frame(&r, &src, &picture_view());
    settled(&r, &src);
    let base = coverage(&frame(&r, &src, &mask_view(2)));
    let mut seen = vec![base];
    for tweak in 0..5 {
        let s = edit_depth(&src, |d| match tweak {
            0 => d.far = 0.2,
            1 => d.far = 0.9,
            2 => d.feather = 0.5,
            3 => d.grow = 0.5,
            _ => d.refine = 1.0,
        });
        r.depth_prepare(&s);
        assert!(
            !r.depth_worker_alive(s.id),
            "a slider is no new map (tweak {tweak})"
        );
        seen.push(coverage(&frame(&r, &s, &mask_view(2))));
    }
    assert_eq!(
        spy.runs.load(Ordering::SeqCst),
        1,
        "one network run for all the sliders"
    );
    assert_eq!(r.depth_networks(src.id), 1);
    // The far limit did act: a farther limit covers more.
    assert!(
        seen[2] > seen[0],
        "far 0.9 covers more than far 0.4: {seen:?}"
    );
    // Grading a node, the look, the output sharpening and the crop leave the map where it is.
    let mut s = src.settings.clone();
    s.nodes.corrector_mut(2).unwrap().trim.exposure_ev = -2.0;
    s.output_mut().look_intensity = 30.0;
    s.output_mut().output_sharpening = 50.0;
    s.input_mut().orientation = Orientation::Deg180;
    let graded = src.with_settings(s);
    r.depth_prepare(&graded);
    assert!(!r.depth_worker_alive(src.id));
    let _ = frame(&r, &graded, &mask_view(2));
    assert_eq!(spy.runs.load(Ordering::SeqCst), 1);
    // The Input's exposure moves the picture: a new map, and until it is made the old one stands in.
    let mut s = src.settings.clone();
    s.input_mut().exposure = ym_core::settings::AutoOr::manual_ev(1.0);
    let brighter = src.with_settings(s);
    let during = coverage(&frame(&r, &brighter, &mask_view(2)));
    // (Refine edges reads the Input's picture as its guide, which the exposure moved: the mask is the same to a hair.)
    assert!(
        (during - base).abs() < 5e-3,
        "the old map stood in: {during} against {base}"
    );
    wait_until("the new map", 60, || {
        spy.runs.load(Ordering::SeqCst) == 2 && r.depth_ready(&brighter)
    });
    assert!(coverage(&frame(&r, &brighter, &mask_view(2))) > 0.1);
}

#[test]
fn a_document_without_a_depth_row_renders_the_bytes_it_did_and_starts_no_worker() {
    let spy = Arc::new(Spy::default());
    let with_model = renderer(spy_model(&spy, 0.0));
    let without = renderer_without_model();
    let d = photo(1, None);
    let src = with_rows(
        &d.source(),
        vec![json!({ "kind": "radial", "mode": "add" })],
    );
    for st in [picture_view(), mask_view(2)] {
        let a = frame(&with_model, &src, &st);
        let b = frame(&without, &src, &st);
        assert_eq!(a.rgba, b.rgba);
    }
    assert!(!with_model.depth_worker_alive(src.id));
    assert_eq!(spy.runs.load(Ordering::SeqCst), 0, "the network never ran");
    assert_eq!(
        with_model.object_pictures_made(),
        0,
        "and no picture was made"
    );
}

// ---- exports ------------------------------------------------------------------------------------------------------

#[test]
fn an_export_makes_its_own_map_and_equals_the_export_of_a_second_renderer_and_of_tiles() {
    let d = photo(1, None);
    let src = with_rows(
        &d.source(),
        vec![depth_row(json!({ "far": 0.5, "feather": 0.2 }))],
    );
    let export = |r: &Renderer| {
        r.render_full(&src, &export_request(), &CancelToken::new())
            .expect("an export")
    };
    let spy = Arc::new(Spy::default());
    let a = renderer(spy_model(&spy, 0.0));
    let first = export(&a);
    assert_eq!(
        spy.runs.load(Ordering::SeqCst),
        1,
        "the export made its map"
    );
    assert!(
        !a.depth_ready(&src),
        "and kept none: the viewer's slot is the viewer's"
    );
    // A second renderer, another tile size: the same bytes.
    let spy2 = Arc::new(Spy::default());
    let b = Renderer::new(
        Arc::new(NeutralOnly),
        options(Some(spy_model(&spy2, 0.0)), 256),
    )
    .expect("render thread");
    assert!(first.rgb == export(&b).rgb, "tiles equal the whole frame");
    // The viewer's map, made on the same picture, is the map an export would make.
    let _ = frame(&a, &src, &picture_view());
    settled(&a, &src);
    let viewer = coverage(&frame(&a, &src, &mask_view(2)));
    assert!(viewer > 0.2);
    assert!(
        first.rgb == export(&a).rgb,
        "an export reads nothing of the viewer's map but its key"
    );
    // The row acts: the same node without it is another picture.
    let plain = a
        .render_full(
            &d.source().with_settings(graph(vec![])),
            &export_request(),
            &CancelToken::new(),
        )
        .expect("an export");
    assert!(plain.rgb != first.rgb);
}

// ---- the plane through the geometry --------------------------------------------------------------------------------

#[test]
fn the_plane_follows_the_photos_geometry_as_backgrounds_does() {
    let r = renderer(ramp_model());
    let d = photo(1, None);
    // The ramp: nearest at the left of the photo, farthest at the right. The gradient to 0.5 is full at the left edge and
    // none from the middle on: a strip at the left edge is nearly full, the right half has none.
    let base = with_rows(&d.source(), vec![depth_row(json!({ "far": 0.5 }))]);
    let _ = frame(&r, &base, &picture_view());
    settled(&r, &base);
    let plain = frame(&r, &base, &mask_view(2));
    let (left, right) = (
        coverage_in(&plain, (0.02, 0.15), (0.3, 0.7)),
        coverage_in(&plain, (0.55, 1.0), (0.3, 0.7)),
    );
    assert!(left > 0.7 && right < 0.02, "left {left} right {right}");
    // A quarter turn clockwise: the left of the photo is the top of the cell.
    let mut turned = base.settings.clone();
    turned.input_mut().orientation = Orientation::Deg90;
    let turned = base.with_settings(turned);
    let t = frame(&r, &turned, &mask_view(2));
    let (top, bottom) = (
        coverage_in(&t, (0.3, 0.7), (0.02, 0.15)),
        coverage_in(&t, (0.3, 0.7), (0.55, 1.0)),
    );
    assert!(top > 0.7 && bottom < 0.02, "top {top} bottom {bottom}");
    assert_eq!(r.depth_networks(base.id), 1, "the turn made no map");
    // Half a turn: the left of the photo is the right of the cell.
    let mut flipped = base.settings.clone();
    flipped.input_mut().orientation = Orientation::Deg180;
    let f = frame(&r, &base.with_settings(flipped), &mask_view(2));
    let (l, rr) = (
        coverage_in(&f, (0.0, 0.45), (0.3, 0.7)),
        coverage_in(&f, (0.85, 0.98), (0.3, 0.7)),
    );
    assert!(l < 0.02 && rr > 0.7, "left {l} right {rr}");
    // A crop to the left half of the photo shows the near part only; to the right half, the far part.
    let crop = |x: f64| ym_core::settings::Crop {
        rect: ym_core::settings::NormRect {
            x,
            y: 0.0,
            width: 0.4,
            height: 1.0,
        },
        angle_deg: 0.0,
        aspect: ym_core::settings::CropAspect::Free,
        auto: false,
    };
    for (x, covered) in [(0.0, true), (0.6, false)] {
        let mut s = base.settings.clone();
        s.input_mut().crop = Some(crop(x));
        let c = frame(&r, &base.with_settings(s), &mask_view(2));
        // The cropped photo is a portrait in the middle of the square cell: look at the middle.
        let m = coverage_in(&c, (0.3, 0.7), (0.3, 0.7));
        assert_eq!(m > 0.4, covered, "crop at {x}: {m}");
        assert_eq!(m < 0.02, !covered, "crop at {x}: {m}");
    }
    // A straightened photo has the plane rotate with the picture: the empty corners are not selected.
    let mut s = base.settings.clone();
    s.input_mut().crop = Some(ym_core::settings::Crop {
        rect: ym_core::settings::NormRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        },
        angle_deg: 10.0,
        aspect: ym_core::settings::CropAspect::Free,
        auto: false,
    });
    let straight = frame(&r, &base.with_settings(s), &mask_view(2));
    let (l, rr) = (
        coverage_in(&straight, (0.1, 0.4), (0.4, 0.6)),
        coverage_in(&straight, (0.6, 0.9), (0.4, 0.6)),
    );
    assert!(l > 0.3 && rr < 0.05, "left {l} right {rr}");
    assert_eq!(r.depth_networks(base.id), 1);
}

// ---- Pick depth ---------------------------------------------------------------------------------------------------

#[test]
fn pick_depth_waits_for_the_worker_reads_the_cached_map_and_runs_no_network_of_its_own() {
    let spy = binary_spy();
    spy.delay_ms.store(150, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    r.depth_prepare(&src);
    // The click comes while the worker is waiting or running: it waits for the map.
    let on_patch = r
        .sample_depth(&src, 0.3, 0.7, &CancelToken::new())
        .expect("an answer")
        .expect("a far limit");
    let on_ground = r
        .sample_depth(&src, 0.8, 0.2, &CancelToken::new())
        .expect("an answer")
        .expect("a far limit");
    // The bright patch is the nearest, about 2 % of the photo: its depth position is under 0.1, so the limit is the margin
    // above it. The dark ground is the far end (z = 1): the limit is the whole depth.
    assert!((0.1..0.2).contains(&on_patch), "{on_patch}");
    assert_eq!(on_ground, 1.0);
    assert_eq!(spy.runs.load(Ordering::SeqCst), 1, "one run, the worker's");
    assert_eq!(r.depth_networks(src.id), 1);
    // The answer is the map's: the limit of `pick_depth` at that place.
    let map = r.depth_map(&src, &CancelToken::new()).expect("the map");
    assert_eq!(ym_color::mask_depth::pick_depth(&map, 0.8, 0.2), on_ground);
    // Never under the margin (the least depth position is 0) and never over 1.
    for (x, y) in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.31, 0.69)] {
        let f = r
            .sample_depth(&src, x, y, &CancelToken::new())
            .unwrap()
            .unwrap();
        assert!((0.1..=1.0).contains(&f), "{f}");
    }
    // A click that is not a number is refused.
    assert!(
        r.sample_depth(&src, f64::NAN, 0.5, &CancelToken::new())
            .is_err()
    );
    // A click when nothing has asked for the map makes it (as `mask_find` makes the features).
    let spy2 = binary_spy();
    let r2 = renderer(spy_model(&spy2, 0.0));
    let far = r2
        .sample_depth(&src, 0.3, 0.7, &CancelToken::new())
        .unwrap();
    assert_eq!(far, Some(on_patch));
    assert_eq!(spy2.runs.load(Ordering::SeqCst), 1);
}

// ---- Focus plane --------------------------------------------------------------------------------------------------

/// The camera's frame of the synthetic files, in the units of the maker note.
const CAMERA: [u16; 2] = [6000, 4000];

/// The maker note's point for the sensor fraction `(x, y)`.
fn at(x: f64, y: f64) -> [u16; 4] {
    [
        CAMERA[0],
        CAMERA[1],
        (x * f64::from(CAMERA[0])) as u16,
        (y * f64::from(CAMERA[1])) as u16,
    ]
}

fn focus_of(r: &Renderer, d: &Developed) -> FocusPlane {
    let src = with_rows(&d.source(), vec![depth_row(json!({}))]);
    r.focus_plane(&src, &CancelToken::new()).expect("an answer")
}

#[test]
fn focus_plane_takes_the_median_depth_in_the_af_box_in_all_eight_orientations() {
    // The bright square is the nearest thing in the photo, at (0.3, 0.7) of the sensor, wherever the camera turned it.
    let spy = binary_spy();
    let r = renderer(spy_model(&spy, 0.0));
    for orientation in 1..=8u16 {
        let d = photo(orientation, Some((at(0.3, 0.7), Some([300, 300, 257]))));
        match focus_of(&r, &d) {
            FocusPlane::Far { far, exact_centre } => {
                assert!(
                    (0.1..0.2).contains(&far),
                    "orientation {orientation}: the box lands on the square: {far}"
                );
                assert!(!exact_centre);
            }
            other => panic!("orientation {orientation}: {other:?}"),
        }
        // A point on the dark ground is far: the median says so.
        let ground = photo(orientation, Some((at(0.8, 0.2), Some([300, 300, 257]))));
        match focus_of(&r, &ground) {
            FocusPlane::Far { far, .. } => {
                assert_eq!(far, 1.0, "orientation {orientation}: the ground is far")
            }
            other => panic!("orientation {orientation}: {other:?}"),
        }
    }
    // One map for each picture: the two photos of an orientation have the same pixels and only differ in the camera's focus tag.
    assert_eq!(spy.runs.load(Ordering::SeqCst), 8);
}

#[test]
fn focus_plane_without_a_box_uses_a_small_square_and_with_a_box_the_cameras() {
    let r = renderer(spy_model(&binary_spy(), 0.0));
    // No FocusFrameSize: a square of 4 % of the long edge at the point.
    let small = photo(1, Some((at(0.3, 0.7), None)));
    match focus_of(&r, &small) {
        FocusPlane::Far { far, exact_centre } => {
            assert!((0.1..0.2).contains(&far), "{far}");
            assert!(!exact_centre);
        }
        other => panic!("{other:?}"),
    }
    // A box that is much bigger than the square takes the side that holds most of it: a box of 3000 x 2000 camera units
    // (half the photo) centred on the square holds mostly ground.
    let big = photo(1, Some((at(0.3, 0.7), Some([3000, 2000, 257]))));
    match focus_of(&r, &big) {
        FocusPlane::Far { far, .. } => assert_eq!(far, 1.0, "the median is the ground's: {far}"),
        other => panic!("{other:?}"),
    }
    // FocusLocation2 wins over FocusLocation on newer bodies (the file reader's rule): the same answer through either tag.
    let dir = tempfile::tempdir().expect("dir");
    let scene = Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Uniform, 0.08);
    scene
        .builder(MODEL)
        .focus_location2(at(0.3, 0.7))
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let two = Developed::open(dir);
    assert!(matches!(focus_of(&r, &two), FocusPlane::Far { .. }));
}

#[test]
fn focus_plane_has_no_answer_without_a_point_and_says_so_without_waiting_for_the_map() {
    let spy = Arc::new(Spy::default());
    spy.delay_ms.store(20_000, Ordering::SeqCst);
    let r = renderer(spy_model(&spy, 0.0));
    // No tags at all (an older body, a file that is not from a Sony camera).
    let none = photo(1, None);
    let t = Instant::now();
    assert_eq!(
        focus_of(&r, &none),
        FocusPlane::None(FocusReason::NoFocusPoint)
    );
    // A panorama: the maker note's point is 0 0.
    let pano = photo(1, Some(([6000, 4000, 0, 0], None)));
    assert_eq!(
        focus_of(&r, &pano),
        FocusPlane::None(FocusReason::NoFocusPoint)
    );
    assert!(
        t.elapsed() < Duration::from_secs(15),
        "answered at once, not after the 20 s network"
    );
    assert_eq!(spy.runs.load(Ordering::SeqCst), 0, "no map was asked for");
    assert_eq!(FocusReason::NoFocusPoint.wire(), "no_focus_point");
    assert_eq!(FocusReason::MapFailed.wire(), "map_failed");
}

#[test]
fn focus_plane_at_the_exact_centre_says_so_and_at_the_edge_of_the_grid_still_answers() {
    let r = renderer(spy_model(&binary_spy(), 0.0));
    // Sony writes the centre when it does not know: the limit is made all the same, and the answer says so.
    let centre = photo(1, Some(([6000, 4000, 3000, 2000], Some([300, 300, 257]))));
    match focus_of(&r, &centre) {
        FocusPlane::Far { far, exact_centre } => {
            assert!(exact_centre);
            assert!((0.1..=1.0).contains(&far), "{far}");
        }
        other => panic!("{other:?}"),
    }
    // One unit off the centre is not the centre.
    let off = photo(1, Some(([6000, 4000, 3001, 2000], None)));
    assert!(matches!(
        focus_of(&r, &off),
        FocusPlane::Far {
            exact_centre: false,
            ..
        }
    ));
    // The corner: the box is cut to the photo and still gives an answer in 0 to 1.
    for point in [
        [6000, 4000, 1, 1],
        [6000, 4000, 5999, 3999],
        [6000, 4000, 5999, 1],
    ] {
        let corner = photo(1, Some((point, Some([2000, 2000, 257]))));
        match focus_of(&r, &corner) {
            FocusPlane::Far { far, .. } => {
                assert!((0.1..=1.0).contains(&far), "{point:?}: {far}")
            }
            other => panic!("{point:?}: {other:?}"),
        }
    }
}

// ---- the mask views -----------------------------------------------------------------------------------------------

#[test]
fn the_mask_view_of_a_depth_row_shows_its_gradient_and_invert_is_its_complement() {
    let r = renderer(ramp_model());
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![depth_row(json!({ "far": 0.5 }))]);
    let _ = frame(&r, &src, &picture_view());
    settled(&r, &src);
    let band = frame(&r, &src, &mask_view(2));
    let mut s = src.settings.clone();
    s.nodes.corrector_mut(2).unwrap().masks.items[0].invert = true;
    let inverted = frame(&r, &src.with_settings(s), &mask_view(2));
    let (c, ci) = (coverage(&band), coverage(&inverted));
    assert!((c + ci - 1.0).abs() < 0.02, "{c} + {ci}");
    // The eye off: the row is absent and the node's mask is 1 everywhere.
    let mut s = src.settings.clone();
    s.nodes.corrector_mut(2).unwrap().masks.items[0].enabled = false;
    let off = frame(&r, &src.with_settings(s), &mask_view(2));
    assert!(coverage(&off) > 0.99);
}

// ---- Show depth ----------------------------------------------------------------------------------------------------

/// The mask view of row `row` of node 2, with Show depth on or off.
fn show_view(row: Option<u32>, style: MaskStyle, depth: bool) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node: 2,
        style,
        row,
        depth,
    });
    st
}

/// A Depth row whose gradient ends near the front, inverted, faint and with every raster step on: the mask view of it is
/// nothing like the depth picture.
fn fussy_row() -> Value {
    depth_row(json!({
        "far": 0.1, "invert": true, "opacity": 0.3,
        "grow": 0.5, "feather": 0.3, "refine": 0.9
    }))
}

#[test]
fn show_depth_is_the_whole_map_as_grey_white_near_and_the_row_s_band_steps_invert_and_opacity_do_not_touch_it()
 {
    // The ramp model: near on the left, far on the right.
    let r = renderer(ramp_model());
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![fussy_row()]);
    // A viewer never waits: until the map is made the view selects nothing (black), and the page draws again when it is.
    let early = frame(&r, &src, &show_view(Some(0), MaskStyle::MaskOnly, true));
    assert_eq!(coverage(&early), 0.0);
    settled(&r, &src);
    let shown = frame(&r, &src, &show_view(Some(0), MaskStyle::MaskOnly, true));
    let (left, middle, right) = (
        coverage_in(&shown, (0.1, 0.25), (0.3, 0.7)),
        coverage_in(&shown, (0.45, 0.55), (0.3, 0.7)),
        coverage_in(&shown, (0.75, 0.9), (0.3, 0.7)),
    );
    assert!(
        left > 0.8 && left > middle + 0.3 && middle > right + 0.3 && right < 0.2,
        "{left} {middle} {right}"
    );
    // It is the whole map and not the gradient: the same bytes for another limit, another Invert, another Opacity and no steps.
    let plain = with_rows(&d.source(), vec![depth_row(json!({}))]);
    let same = frame(&r, &plain, &show_view(Some(0), MaskStyle::MaskOnly, true));
    assert!(shown.rgba == same.rgba, "shown and same differ");
    // The band view of the same row is something else.
    let band = frame(&r, &src, &show_view(Some(0), MaskStyle::MaskOnly, false));
    assert!(band.rgba != shown.rgba, "band and shown are the same");
    // Show depth is grey whatever the style says (it is not a tint over the photo).
    let overlay = frame(&r, &src, &show_view(Some(0), MaskStyle::Overlay, true));
    assert!(overlay.rgba == shown.rgba, "overlay and shown differ");
    // The row's eye off makes no difference to a view that is about that row.
    let mut off = src.settings.clone();
    off.nodes.corrector_mut(2).unwrap().masks.items[0].enabled = false;
    let off = frame(
        &r,
        &src.with_settings(off),
        &show_view(Some(0), MaskStyle::MaskOnly, true),
    );
    assert!(off.rgba == shown.rgba, "off and shown differ");
    // The photo is not drawn under it: no pixel of the grey picture is the paper's colour but the paper itself.
    assert!(coverage(&shown) > 0.2 && coverage(&shown) < 0.8);
}

#[test]
fn show_depth_means_nothing_for_a_row_that_is_not_a_depth_row_and_for_the_node_s_own_mask() {
    let r = renderer(ramp_model());
    let d = photo(1, None);
    let linear = json!({ "kind": "linear", "mode": "add", "center": [0.5, 0.5], "angle_deg": 0.0, "feather": 0.2 });
    let src = with_rows(
        &d.source(),
        vec![linear.clone(), depth_row(json!({ "far": 0.5 }))],
    );
    let _ = frame(&r, &src, &picture_view());
    settled(&r, &src);
    // The Linear row (row 0) with the flag on is the Linear row's own view.
    let with_flag = frame(&r, &src, &show_view(Some(0), MaskStyle::MaskOnly, true));
    let without = frame(&r, &src, &show_view(Some(0), MaskStyle::MaskOnly, false));
    assert!(
        with_flag.rgba == without.rgba,
        "with_flag and without differ"
    );
    // No row: the node's own mask, whatever the flag says.
    let with_flag = frame(&r, &src, &show_view(None, MaskStyle::MaskOnly, true));
    let without = frame(&r, &src, &show_view(None, MaskStyle::MaskOnly, false));
    assert!(
        with_flag.rgba == without.rgba,
        "with_flag and without differ"
    );
    // The Depth row (row 1) shows the map: it differs from its band.
    let shown = frame(&r, &src, &show_view(Some(1), MaskStyle::MaskOnly, true));
    let band = frame(&r, &src, &show_view(Some(1), MaskStyle::MaskOnly, false));
    assert!(shown.rgba != band.rgba, "shown and band are the same");
    // A row the node does not have is the node's own view.
    let past = frame(&r, &src, &show_view(Some(9), MaskStyle::MaskOnly, true));
    let node = frame(&r, &src, &show_view(None, MaskStyle::MaskOnly, false));
    assert!(past.rgba == node.rgba, "past and node differ");
}

#[test]
fn show_depth_takes_the_depth_rows_of_the_nodes_upstream_at_their_limits() {
    // Node 2 has the Depth row that is shown; node 3, after it, has a Depth row of its own that must not become a grey picture.
    let r = renderer(ramp_model());
    let d = photo(1, None);
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "masks": { "invert": false, "items": [ depth_row(json!({})) ] } },
        { "id": 3, "kind": "corrector", "pos": [2.0, 0.0], "masks": { "invert": false, "items": [ depth_row(json!({ "far": 0.1 })) ] } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    let settings = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0;
    let src = d.source().with_settings(settings);
    let _ = frame(&r, &src, &picture_view());
    settled(&r, &src);
    // Showing node 3's row: node 2's row stays its gradient, node 3's is the map.
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = Some(MaskView {
        node: 3,
        style: MaskStyle::MaskOnly,
        row: Some(0),
        depth: true,
    });
    let shown = frame(&r, &src, &st);
    let (left, right) = (
        coverage_in(&shown, (0.1, 0.25), (0.3, 0.7)),
        coverage_in(&shown, (0.75, 0.9), (0.3, 0.7)),
    );
    assert!(left > 0.8 && right < 0.2, "{left} {right}");
}

/// Mean absolute difference of two frames' colour channels, in code values.
fn mean_difference(a: &RenderedFrame, b: &RenderedFrame) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let total: u64 = a
        .rgba
        .iter()
        .zip(&b.rgba)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.rgba.len() as f64
}

#[test]
fn show_depth_is_made_on_the_device_as_on_the_cpu() {
    let gpu = match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            eprintln!("\n* GPU TESTS SKIPPED in mask_depth (Show depth): {e}\n");
            return;
        }
    };
    let device = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: ym_core::app::gpu::GpuStages::ALL,
            ..options(Some(ramp_model()), RenderOptions::default().full_tile_px)
        },
    )
    .expect("render thread");
    let cpu = renderer(ramp_model());
    let d = photo(1, None);
    let src = with_rows(&d.source(), vec![fussy_row()]);
    for r in [&device, &cpu] {
        let _ = frame(r, &src, &picture_view());
        settled(r, &src);
    }
    for style in [MaskStyle::MaskOnly, MaskStyle::Overlay] {
        let st = show_view(Some(0), style, true);
        let (a, b) = (frame(&device, &src, &st), frame(&cpu, &src, &st));
        assert_eq!(
            a.report.device,
            ym_core::pipeline::ComputeDevice::Gpu,
            "the Show depth frame is the device's ({style:?})"
        );
        assert!(
            !a.report.notes.iter().any(|n| n.contains("graphics card")),
            "{:?}",
            a.report.notes
        );
        let difference = mean_difference(&a, &b);
        assert!(
            difference < 1.0,
            "device and CPU differ by {difference} codes ({style:?})"
        );
        // And it is the grey ramp on the device too.
        let (left, right) = (
            coverage_in(&a, (0.1, 0.25), (0.3, 0.7)),
            coverage_in(&a, (0.75, 0.9), (0.3, 0.7)),
        );
        assert!(left > 0.8 && right < 0.2, "{left} {right}");
    }
    // The device is still the route afterwards.
    assert!(!device.raw_on_cpu(src.id));
}
