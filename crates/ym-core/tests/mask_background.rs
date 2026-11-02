//! Background masks through the render graph (`docs/23-nodes.md` sections 4.6 and 7.6, `docs/algorithms/06` section 5.4,
//! task N-15): a Background row is cut from the photo's blur map, which is the map the lens's bokeh reshaping uses, made
//! once per photo from the raw group (so the lens sliders do not move it) by the learned model, or by the classical
//! detector when the model cannot run. Level and Softness cut a new `S0` from the cached map; Feather, Grow and Refine
//! edges run the raster steps only. Tiles of an export equal the whole frame bit for bit, the device agrees with the
//! CPU (`crates/ym-gpu/tests/gpu_raster_masks.rs`, `crates/ym-cli/tests/parity.rs`), and a photo with no out-of-focus
//! area gets an empty mask.
//!
//! The photos are the synthetic ones the other render tests use and, for what the learned model says about real
//! blur, the lens photos of `testdata/lens-photos` (git-ignored; the tests that need them say so and pass without
//! them), the same photos `docs/21` section 7 judged the model on.
mod common;

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly, PhotoSource,
    RenderOptions, RenderedFrame, Renderer, ViewportRect, ViewportState, unmakeable_mask,
};
use ym_core::session::{OpenOptions, Session, SessionEnv};
use ym_core::settings::{
    DevelopSettings, LensKind, LensSettings, MaskKind, Orientation, mask_kinds, mask_unavailable,
};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;
use ym_lens::LensLibrary;
use ym_ml::bokeh_net::BokehNet;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
/// The paper around a photo in a Mask only frame: nothing in a mask is this colour.
const PAPER: [u8; 3] = [40, 60, 90];

fn bokeh_model() -> Arc<BokehNet> {
    static CELL: OnceLock<Option<Arc<BokehNet>>> = OnceLock::new();
    CELL.get_or_init(|| BokehNet::load_bundled(&data_root()).ok().map(Arc::new))
        .clone()
        .expect("the shipped bokeh model loads (data/models/bokeh)")
}

fn library() -> Arc<LensLibrary> {
    static LIBRARY: OnceLock<Arc<LensLibrary>> = OnceLock::new();
    Arc::clone(LIBRARY.get_or_init(|| {
        Arc::new(LensLibrary::load(&data_root()).expect("the shipped lens data load"))
    }))
}

/// A renderer with the learned model (`learned`) or without it (the classical detector), the shipped lenses, a
/// tile size for exports and the paper colour of the Mask only view.
fn renderer_with(learned: bool, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            bokeh_model: learned.then(bokeh_model),
            lens: Some(library()),
            data_root: Some(data_root()),
            background: PAPER,
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn renderer(learned: bool) -> Renderer {
    renderer_with(learned, RenderOptions::default().full_tile_px)
}

/// Input, Main, node 2 (with these mask rows and an exposure trim of a stop), Output.
fn graph(rows: Vec<Value>) -> DevelopSettings {
    graph_with_trim(1.0, rows)
}

/// [`graph`] with a trim of `ev` stops on node 2: 0 is a node that changes nothing whatever its mask is.
fn graph_with_trim(ev: f64, rows: Vec<Value>) -> DevelopSettings {
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": ev },
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

fn background(extra: Value) -> Value {
    let mut row = json!({ "kind": "background", "mode": "add" });
    if let (Some(r), Some(e)) = (row.as_object_mut(), extra.as_object()) {
        r.extend(e.clone());
    }
    row
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

/// A square cell, so that the 3:2 photos have paper above and below them.
const CELL: f64 = 300.0;

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

/// The mask of a Mask only frame as a list of `m` (the grey codes over 255) at the photo's pixels (the paper is left
/// out).
fn mask_values(f: &RenderedFrame) -> Vec<f64> {
    f.rgba
        .chunks(4)
        .filter(|p| p[..3] != PAPER)
        .map(|p| {
            assert!(p[0] == p[1] && p[1] == p[2], "the mask is grey: {p:?}");
            f64::from(p[0]) / 255.0
        })
        .collect()
}

fn coverage(f: &RenderedFrame) -> f64 {
    let m = mask_values(f);
    m.iter().sum::<f64>() / m.len().max(1) as f64
}

/// The number of times the photo's blur map was made so far (`StageRuns::blur_maps`).
fn maps_made(r: &Renderer, src: &PhotoSource) -> u64 {
    r.counters(src.id).blur_maps
}

// ---- synthetic photos --------------------------------------------------------------------------------

/// A night scene of defocused lights (discs of 1.5 % of the width on a dark ground), `cols` x `rows` of them in the
/// part of the frame `region` (fractions `x0, x1`), as an ARW with this EXIF orientation.
fn lights(w: u32, h: u32, exif: u16, region: (f64, f64), cols: usize, rows: usize) -> Developed {
    lights_of(w, h, exif, region, cols, rows, 0.45)
}

/// [`lights`] whose discs are `level` of the sensor's range (0.45 is the other tests'; a dimmer scene has edges the raw
/// stages cut into tiles without a difference in the last bit).
fn lights_of(w: u32, h: u32, exif: u16, region: (f64, f64), cols: usize, rows: usize, level: f64) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.02).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + level * (scene.white_level - scene.black)) as u16;
    let radius = 0.015 * f64::from(w);
    for r in 0..rows {
        for c in 0..cols {
            let fx = region.0 + (region.1 - region.0) * (c as f64 + 0.5) / cols as f64;
            let fy = 0.1 + 0.8 * (r as f64 + 0.5) / rows as f64;
            let (cx, cy) = (fx * f64::from(w), fy * f64::from(h));
            for y in 0..h {
                for x in 0..w {
                    let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                    if d <= radius {
                        mosaic[(y * w + x) as usize] = code;
                    }
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .orientation(exif)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

/// A photo of sharp patches of colour (nothing out of focus).
fn sharp() -> Developed {
    let scene = Scene::new(480, 320, light(PROFILE, 4800.0), Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

// ---- the kind --------------------------------------------------------------------------------------

#[test]
fn background_is_a_kind_this_computer_can_make_first_in_the_menu() {
    let kinds = mask_kinds();
    assert_eq!(kinds[0], MaskKind::Background, "the FIND group comes first");
    let missing: Vec<MaskKind> = mask_unavailable().into_iter().map(|(k, _)| k).collect();
    assert!(!missing.contains(&MaskKind::Background), "{missing:?}");
    // A Background row is no `mask` error, a row of a kind that is not built still is.
    let s = graph(vec![background(json!({}))]);
    assert_eq!(unmakeable_mask(&s), None);
    let s = graph(vec![json!({ "kind": "object", "points": [[0.5, 0.5, true]] })]);
    assert!(unmakeable_mask(&s).is_some());
}

#[test]
fn background_rows_count_among_the_48_planes_a_render_holds() {
    let rows = |n: usize| vec![background(json!({})); n];
    let doc = json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "masks": { "invert": false, "items": rows(32) } },
        { "id": 3, "kind": "corrector", "pos": [2.0, 0.0], "masks": { "invert": false, "items": rows(17) } },
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    let s = serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("valid")
        .0;
    assert_eq!(
        unmakeable_mask(&s).as_deref(),
        Some("This photo has more than 48 masks that are drawn or found. Remove some of them.")
    );
}

// ---- real photos: what the learned model says ---------------------------------------------------------

fn lens_photo(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/lens-photos")
        .join(name);
    if !path.exists() {
        eprintln!("[background] testdata/lens-photos/{name} is absent: skipped");
    }
    path.exists().then_some(path)
}

fn develop(path: &std::path::Path) -> (Session, PhotoSource) {
    let env = SessionEnv::load(&data_root()).expect("data folder");
    let session = Session::new(env);
    let out = session.open(&[path.to_path_buf()], OpenOptions::default());
    assert!(session.wait_idle(Duration::from_secs(300)));
    let src = PhotoSource::from_session(&session, out.added[0]).expect("source");
    (session, src)
}

/// The mean of the Background mask at the default Level and Softness over a real photo, with the learned model.
fn real_coverage(r: &Renderer, name: &str) -> Option<f64> {
    let (_session, src) = develop(&lens_photo(name)?);
    let masked = with(&src, graph(vec![background(json!({}))]));
    let c = coverage(&frame(r, &masked, &mask_view(2)));
    eprintln!("[background] {name}: the mask covers {c:.3} of the photo");
    Some(c)
}

/// The fixtures of `docs/21` section 7: photos the lens judged to have bokeh (a share of the frame at `P >= 0.9`:
/// 50585 77.7 %, 56893 64.5 %, 26276 50.0 %, 27308 35.5 %) and photos with none (0.0 %). The learned map cut at the
/// default Level covers most of the first and none of the second.
#[test]
fn the_learned_map_makes_the_background_of_real_photos_with_bokeh_and_nothing_of_sharp_ones() {
    let r = renderer(true);
    for (name, at_least) in [
        ("50585_DSC02988.ARW", 0.5),
        ("26276_DSC03483.ARW", 0.4),
        ("27308_DSC08793.ARW", 0.3),
    ] {
        if let Some(c) = real_coverage(&r, name) {
            assert!(c >= at_least, "{name}: the mask covers {c:.3}, wanted {at_least}");
        }
    }
    for name in ["18469_DSC02868.ARW", "26551_DSC05356.ARW", "31909_DSC09550.ARW"] {
        if let Some(c) = real_coverage(&r, name) {
            assert!(c <= 0.02, "{name}: a sharp photo's mask covers {c:.3}");
        }
    }
}

// ---- the map -----------------------------------------------------------------------------------------

#[test]
fn an_evenly_sharp_photo_gives_an_empty_mask_with_the_model_and_without_it() {
    let d = sharp();
    let src = d.source();
    for learned in [true, false] {
        let r = renderer(learned);
        let map = r.blur_map(&src, &CancelToken::new()).expect("a map");
        assert_eq!((map.width, map.height), (240, 160), "a cell is 2 x 2 pixels of the lens plane");
        let top = map.p.iter().copied().fold(0.0f32, f32::max);
        assert!(top < 0.05, "learned {learned}: the sharpest photo has P up to {top}");
        for level in [0.5, 0.7] {
            let masked = with(&src, graph(vec![background(json!({ "level": level }))]));
            let f = frame(&r, &masked, &mask_view(2));
            let m = mask_values(&f);
            assert!(!m.is_empty());
            assert!(
                m.iter().all(|v| *v == 0.0),
                "learned {learned}, level {level}: an empty mask, found up to {}",
                m.iter().copied().fold(0.0, f64::max)
            );
        }
        // And the node changes nothing: the photo is the photo (the same graph with a node that has no trim).
        let none = with(&src, graph_with_trim(0.0, vec![]));
        let render = |s: &PhotoSource| {
            r.viewport_frame(s, &ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0)), &FrameOptions::default(), &CancelToken::new())
                .expect("frame")
                .rgba
        };
        let masked = with(&src, graph(vec![background(json!({}))]));
        assert!(render(&masked) == render(&none), "learned {learned}: the empty mask changed a pixel");
    }
}

#[test]
fn the_classical_detector_makes_the_map_when_the_model_cannot_run() {
    // Defocused lights on a dark ground: the detector finds the lights' blur. (The learned model's verdict on a
    // synthetic scene is its own business, `docs/21` section 7.)
    let d = lights(1200, 800, 1, (0.0, 1.0), 6, 4);
    let src = d.source();
    let r = renderer(false);
    let map = r.blur_map(&src, &CancelToken::new()).expect("a map");
    let n = map.p.len() as f64;
    let strong = map.p.iter().filter(|p| **p > 0.9).count() as f64 / n;
    eprintln!("[background] the classical map of the lights: {}x{} cells, {strong:.3} at P > 0.9", map.width, map.height);
    assert!(strong > 0.02, "the lights are out of focus for the detector: {strong}");
    assert!(map.p.iter().all(|p| (0.0..=1.0).contains(p)));
    let masked = with(&src, graph(vec![background(json!({}))]));
    let c = coverage(&frame(&r, &masked, &mask_view(2)));
    assert!(c > 0.03 && c < 0.6, "the mask covers {c:.3} of the photo");
    // The mask is where the map is: the cells that are out of focus (P over one half) are lit, the others are not.
    let lit = mask_values(&frame(&r, &masked, &mask_view(2))).iter().filter(|m| **m > 0.5).count() as f64;
    let photo = mask_values(&frame(&r, &masked, &mask_view(2))).len() as f64;
    let cells = map.p.iter().filter(|p| **p > 0.5).count() as f64 / n;
    assert!(
        (lit / photo - cells).abs() < 0.05,
        "{:.3} of the photo is lit, {cells:.3} of the cells are out of focus",
        lit / photo
    );
}

#[test]
fn the_exif_orientation_turns_the_map_to_the_photo_as_the_camera_oriented_it() {
    // Lights in the left 40 % of the sensor image. Where that lands in the photo as the camera oriented it depends on
    // the orientation: 1 left, 3 right, 6 (a quarter turn clockwise) top, 8 bottom. The map has the photo's proportions.
    let r = renderer(false);
    for (exif, w, h, side) in [(1u16, 300usize, 200usize, "left"), (3, 300, 200, "right"), (6, 200, 300, "top"), (8, 200, 300, "bottom")] {
        let d = lights(1200, 800, exif, (0.02, 0.4), 3, 4);
        let map = r.blur_map(&d.source(), &CancelToken::new()).expect("a map");
        assert_eq!((map.width, map.height), (w, h), "orientation {exif}");
        let mass = |keep: &dyn Fn(usize, usize) -> bool| -> f64 {
            (0..map.height)
                .flat_map(|y| (0..map.width).map(move |x| (x, y)))
                .filter(|&(x, y)| keep(x, y))
                .map(|(x, y)| f64::from(map.p[y * map.width + x]))
                .sum()
        };
        let (a, b) = match side {
            "left" => (mass(&|x, _| x < w / 2), mass(&|x, _| x >= w / 2)),
            "right" => (mass(&|x, _| x >= w / 2), mass(&|x, _| x < w / 2)),
            "top" => (mass(&|_, y| y < h / 2), mass(&|_, y| y >= h / 2)),
            _ => (mass(&|_, y| y >= h / 2), mass(&|_, y| y < h / 2)),
        };
        eprintln!("[background] orientation {exif}: P mass {a:.0} on the {side}, {b:.0} on the other side");
        assert!(a > 5.0 * b.max(1.0), "orientation {exif}: the blur should be at the {side} ({a} against {b})");
    }
}

#[test]
fn the_map_is_made_once_per_photo_and_shared_with_the_lens_in_both_orders() {
    // The learned model on both sides: whichever of the lens's Anamorphic and a Background row comes first runs the
    // network, the other reads its map. The lens sliders do not move the map (it is made from the raw group).
    let vintage = |anamorphic: f64, edges: f64| LensSettings {
        kind: LensKind::Vintage,
        flares: 0.0,
        edges,
        anamorphic,
    };
    let d = lights(1200, 800, 1, (0.0, 1.0), 6, 4);
    let base = d.source();
    let row = graph(vec![background(json!({ "level": 0.8 }))]);
    let lensed = |src: &PhotoSource, l: LensSettings| edit(src, |s| s.input_mut().lens = l);
    let view = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let render = |r: &Renderer, s: &PhotoSource| r.viewport_frame(s, &view, &FrameOptions::default(), &CancelToken::new()).expect("frame");

    // The lens first.
    let r = renderer(true);
    let with_lens = lensed(&with(&base, row.clone()), vintage(50.0, 0.0));
    let f = render(&r, &with_lens);
    assert_eq!(f.report.runs.lens_planes, 1, "{:?}", f.report.runs);
    assert_eq!(maps_made(&r, &base), 1, "the lens and the Background row ran the network once between them");
    let map = r.blur_map(&base, &CancelToken::new()).expect("map");
    // The lens sliders do not move it: no new map, the very same cells.
    for l in [vintage(80.0, 0.0), vintage(30.0, 60.0), vintage(0.0, 40.0)] {
        let f = render(&r, &lensed(&with(&base, row.clone()), l));
        assert_eq!(f.report.runs.blur_maps, 0, "{l:?}");
        assert!(Arc::ptr_eq(&r.blur_map(&base, &CancelToken::new()).expect("map"), &map), "{l:?}");
    }
    assert_eq!(maps_made(&r, &base), 1);

    // The Background row first, then the lens (a fresh renderer).
    let r = renderer(true);
    let f = render(&r, &with(&base, row.clone()));
    assert_eq!(f.report.runs.blur_maps, 1);
    assert_eq!(f.report.runs.lens_planes, 0);
    let f = render(&r, &lensed(&with(&base, row), vintage(50.0, 0.0)));
    assert_eq!(f.report.runs.lens_planes, 1, "{:?}", f.report.runs);
    assert_eq!(f.report.runs.blur_maps, 0, "the lens read the map the Background row made");
    assert_eq!(maps_made(&r, &base), 1);
}

#[test]
fn level_and_softness_cut_the_cached_map_and_feather_grow_and_refine_edges_do_not_touch_it() {
    let d = lights(1200, 800, 1, (0.0, 1.0), 6, 4);
    let base = d.source();
    let r = renderer(false);
    let st = mask_view(2);
    let masked = |extra: Value| with(&base, graph(vec![background(extra)]));
    let first = frame(&r, &masked(json!({})), &st);
    assert_eq!(first.report.runs.blur_maps, 1);
    assert_eq!(first.report.runs.grade_plans, 1);
    // Nothing changed: the frame is served as it is.
    let again = frame(&r, &masked(json!({})), &st);
    assert_eq!((again.report.runs.blur_maps, again.report.runs.grade_plans), (0, 0));
    assert!(again.rgba == first.rgba);
    // Each of the five fields makes a new plan from the same map: the network never runs again, the picture moves.
    for (what, extra) in [
        ("Level", json!({ "level": 0.9 })),
        ("Softness", json!({ "softness": 0.9 })),
        ("Feather", json!({ "feather": 0.5 })),
        ("Grow", json!({ "grow": 0.6 })),
        ("Refine edges", json!({ "refine": 1.0 })),
    ] {
        let f = frame(&r, &masked(extra), &st);
        assert_eq!(f.report.runs.blur_maps, 0, "{what} makes no map");
        assert_eq!(f.report.runs.grade_plans, 1, "{what} makes a plan");
        assert!(f.rgba != first.rgba, "{what} moves the mask");
    }
    assert_eq!(maps_made(&r, &base), 1);
}

#[test]
fn the_app_says_it_is_finding_the_mask_until_the_photo_has_its_map() {
    let d = lights(1200, 800, 1, (0.0, 1.0), 6, 4);
    let base = d.source();
    let r = renderer(false);
    let row = with(&base, graph(vec![background(json!({}))]));
    let off = with(&base, graph(vec![background(json!({ "enabled": false }))]));
    let plain = with(&base, graph(vec![]));
    // Before the first frame: a Background row with the eye on is to be found; a row with the eye off or no row is not.
    assert!(r.finding_masks(base.id, &row.settings));
    assert!(!r.finding_masks(base.id, &off.settings));
    assert!(!r.finding_masks(base.id, &plain.settings));
    // A frame without the row makes no map and the question stays.
    let _ = frame(&r, &plain, &mask_view(1));
    assert!(r.finding_masks(base.id, &row.settings));
    // The first frame with the row makes it: from then on the photo has its map.
    let _ = frame(&r, &row, &mask_view(2));
    assert!(!r.finding_masks(base.id, &row.settings));
    // Another photo of the session has to find its own.
    let other = lights(1200, 800, 1, (0.0, 1.0), 6, 4);
    assert!(r.finding_masks(other.source().id, &row.settings) || other.source().id == base.id);
}

// ---- exports, tiles and determinism -----------------------------------------------------------------

fn export_image(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

/// Tiles of an export equal the whole frame bit for bit with the halos of Grow, Feather and Refine edges (the planes of the
/// raster pass are the ones of `crates/ym-color/tests/mask_raster.rs`; a Background plane is one more source for them). The
/// discs are dim (0.15 of the range): the raw stages of a bright hard-edged disc (0.45) are not tile-independent to the last
/// bit at a disc's edge, with no mask at all (tiles of 100, 128 and 256 px; 8 pixels of 1e-8), and a Refine edges guide reads
/// them through its halo.
#[test]
fn tiles_of_an_export_equal_the_whole_frame_with_the_halos_of_grow_feather_and_refine_edges() {
    let d = lights_of(640, 480, 1, (0.0, 1.0), 5, 4, 0.15);
    let src = d.source();
    let rows = vec![
        background(json!({ "level": 0.7, "softness": 0.6, "grow": 0.4, "feather": 0.3, "refine": 0.6 })),
        json!({ "kind": "background", "mode": "subtract", "level": 0.9, "feather": 0.1 }),
    ];
    let s = with(&src, graph(rows));
    let whole = export_image(&renderer_with(false, 4096), &s);
    for tile in [64, 100, 192] {
        let tiled = export_image(&renderer_with(false, tile), &s);
        assert_eq!((tiled.width, tiled.height), (whole.width, whole.height));
        assert!(tiled.rgb == whole.rgb, "tiles of {tile} px differ from the whole frame");
    }
    // The rows act: the same node without them is another picture.
    let plain = export_image(&renderer_with(false, 4096), &with(&src, graph(vec![])));
    assert!(plain.rgb != whole.rgb);
}

#[test]
fn the_same_photo_gives_the_same_map_and_the_same_bytes_in_two_renderers() {
    let d = lights(640, 480, 1, (0.0, 1.0), 5, 4);
    let src = d.source();
    let s = with(&src, graph(vec![background(json!({ "level": 0.7, "feather": 0.2, "refine": 0.5 }))]));
    let (a, b) = (renderer(false), renderer(false));
    let (ma, mb) = (
        a.blur_map(&src, &CancelToken::new()).expect("map"),
        b.blur_map(&src, &CancelToken::new()).expect("map"),
    );
    assert_eq!((ma.width, ma.height), (mb.width, mb.height));
    assert!(ma.p == mb.p, "the cells are the same bits");
    assert!(export_image(&a, &s).rgb == export_image(&b, &s).rgb, "the exports differ");
    assert!(frame(&a, &s, &mask_view(2)).rgba == frame(&b, &s, &mask_view(2)).rgba, "the mask frames differ");
    // The learned model too.
    let (a, b) = (renderer(true), renderer(true));
    assert!(
        a.blur_map(&src, &CancelToken::new()).expect("map").p
            == b.blur_map(&src, &CancelToken::new()).expect("map").p,
        "the learned cells differ"
    );
}

#[test]
fn a_turned_photo_is_the_upright_photo_turned_by_hand_with_a_background_row_on_it() {
    let d = lights(640, 480, 1, (0.0, 0.5), 3, 4);
    let src = d.source();
    let r = renderer(false);
    let upright = with(&src, graph(vec![background(json!({ "level": 0.8, "feather": 0.2 }))]));
    let mut turned = upright.settings.clone();
    turned.input_mut().orientation = Orientation::Deg90;
    let a = export_image(&r, &upright);
    let b = export_image(&r, &with(&src, turned));
    assert_eq!((b.width, b.height), (a.height, a.width));
    let (w, h) = (a.width, a.height);
    let mut worst = 0.0f32;
    for y in 0..b.height {
        for x in 0..b.width {
            for c in 0..3 {
                let t = b.rgb[(y * b.width + x) * 3 + c];
                let u = a.rgb[((h - 1 - x) * w + y) * 3 + c];
                worst = worst.max((t - u).abs());
            }
        }
    }
    eprintln!("[background] a turned photo against the upright one turned by hand: {worst:e}");
    assert!(worst < 2e-4, "{worst}");
    // The mask has something to follow: the node's effect is not the same everywhere.
    let none = export_image(&r, &with(&src, graph(vec![])));
    assert!(a.rgb != none.rgb);
}

// ---- the device ----------------------------------------------------------------------------------------

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
                "\n* GPU TESTS SKIPPED in mask_background: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// The picture of a viewport frame and the mask of a Mask only frame on the device and on the CPU. The raw stages stay on
/// the CPU on both, so that the blur map (the classical detector decides on exact values) is the same bits: what is compared
/// is the raster pass, the colour pass and the encode on the device against the CPU reference, with a Background plane in them.
#[test]
fn the_device_makes_the_same_frames_and_masks_as_the_cpu_with_a_background_row() {
    let Some(gpu) = gpu() else { return };
    let d = lights_of(640, 480, 1, (0.0, 1.0), 5, 4, 0.15);
    let src = d.source();
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL
                .difference(GpuStages::RAW_CHAIN)
                .difference(GpuStages::RAW_FRONT),
            background: PAPER,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let cpu = renderer(false);
    let picture = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    for (what, extra) in [
        ("the default", json!({})),
        ("a high level, a narrow ramp", json!({ "level": 0.8, "softness": 0.1 })),
        ("grown", json!({ "grow": 0.5 })),
        ("feathered", json!({ "feather": 0.4 })),
        ("refined", json!({ "refine": 0.8 })),
        ("all of them", json!({ "level": 0.7, "softness": 0.4, "grow": 0.3, "feather": 0.3, "refine": 0.7 })),
    ] {
        let s = with(&src, graph(vec![background(extra)]));
        for (name, st) in [("picture", &picture), ("mask only", &mask_view(2))] {
            let device = frame(&on_gpu, &s, st);
            assert_eq!(device.report.device, ComputeDevice::Gpu, "{what}, {name}");
            let reference = frame(&cpu, &s, st);
            if name == "mask only" {
                // The test has a mask to compare: lit where the lights are out of focus, dark between them.
                let m = mask_values(&reference);
                assert!(m.iter().any(|v| *v > 0.5) && m.iter().any(|v| *v < 0.5), "{what}: the mask is flat");
            }
            let (mut worst, mut sum) = (0u8, 0u64);
            for (x, y) in device.rgba.iter().zip(&reference.rgba) {
                let d = x.abs_diff(*y);
                worst = worst.max(d);
                sum += u64::from(d);
            }
            let mean = sum as f64 / device.rgba.len() as f64;
            eprintln!("[background] {what}, {name}: the device differs from the CPU by {worst} codes, mean {mean:.4}");
            assert!(worst <= 3 && mean < 0.3, "{what}, {name}: {worst} codes, mean {mean}");
        }
    }
}
