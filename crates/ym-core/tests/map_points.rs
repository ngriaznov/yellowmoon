//! `viewport_map_points` and `mask_kinds` (`docs/23-nodes.md` sections 3.6 and 9, task N-12).
//!
//! * the two directions are each other's inverse through the user's turn, the straightening, the crop, the
//!   lens's bend and the view's zoom and pan;
//! * a point on the paper (outside the picture, or in the empty corner a straightening leaves) has no place in
//!   the photo, and a photo point outside the cell is still answered;
//! * **the mapping agrees with the pixels**: a Radial's centre is drawn where `map_points` says the photo point
//!   is (the centroid of a small Radial in a Mask only frame of N-11), in a photo as shot, cropped,
//!   straightened, turned, bent, and all of them at once, at Fit and zoomed in, at one and two device pixels
//!   per CSS pixel.
mod common;

use std::sync::{Arc, OnceLock};

use common::*;
use serde_json::{Value, json};
use ym_core::pipeline::{
    FrameOptions, MapFrom, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportInput, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{
    Crop, CropAspect, DevelopSettings, LensKind, LensSettings, MASK_KINDS, MaskKind, MaskShape,
    NormRect, Orientation, mask_kinds,
};
use ym_gpu::queue::CancelToken;
use ym_lens::LensLibrary;

const MODEL: &str = "ILCE-7RM4";
/// The paper: nothing in a mask is this colour (a mask is grey).
const PAPER: [u8; 3] = [40, 60, 90];
/// The photo-frame point the Radials are at, in the cases whose crop keeps it in view.
const CENTRE: [f64; 2] = [0.45, 0.55];

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn library() -> Arc<LensLibrary> {
    static LIBRARY: OnceLock<Arc<LensLibrary>> = OnceLock::new();
    Arc::clone(LIBRARY.get_or_init(|| {
        Arc::new(LensLibrary::load(&data_root()).expect("the shipped lens data load"))
    }))
}

fn renderer() -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            lens: Some(library()),
            data_root: Some(data_root()),
            background: PAPER,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// The shapes of the cases: what changes the way the photo is drawn.
#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    crop: Option<(NormRect, f64)>,
    turn: Orientation,
    bend: bool,
}

const FULL: NormRect = NormRect::FULL;

fn cases() -> Vec<Case> {
    let rect = |x, y, width, height| NormRect {
        x,
        y,
        width,
        height,
    };
    vec![
        Case {
            name: "as shot",
            crop: None,
            turn: Orientation::Deg0,
            bend: false,
        },
        Case {
            name: "cropped",
            crop: Some((rect(0.2, 0.1, 0.6, 0.7), 0.0)),
            turn: Orientation::Deg0,
            bend: false,
        },
        Case {
            name: "straightened",
            crop: Some((rect(0.15, 0.15, 0.7, 0.7), 9.0)),
            turn: Orientation::Deg0,
            bend: false,
        },
        Case {
            name: "turned",
            crop: None,
            turn: Orientation::Deg90,
            bend: false,
        },
        Case {
            name: "bent",
            crop: None,
            turn: Orientation::Deg0,
            bend: true,
        },
        Case {
            name: "cropped, straightened, turned and bent",
            crop: Some((rect(0.1, 0.1, 0.8, 0.8), -8.0)),
            turn: Orientation::Deg270,
            bend: true,
        },
    ]
}

/// A node-first document: Main with these mask rows.
fn doc(rows: Vec<Value>) -> DevelopSettings {
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": { "items": [
        { "id": 1, "kind": "corrector", "pos": [0.0, 0.0], "masks": { "invert": false, "items": rows } }
    ] } }))
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn radial(center: [f64; 2], radius: f64) -> Value {
    json!({ "kind": "radial", "mode": "add", "center": center, "radii": [radius, radius], "feather": 0.6 })
}

fn source_of(d: &Developed, case: Case, rows: Vec<Value>) -> PhotoSource {
    let mut s = doc(rows);
    let input = s.input_mut();
    input.crop = case.crop.map(|(rect, angle_deg)| Crop {
        rect,
        angle_deg,
        aspect: CropAspect::Free,
        auto: false,
    });
    input.orientation = case.turn;
    if case.bend {
        input.lens = LensSettings {
            kind: LensKind::Vintage,
            flares: 0.0,
            edges: 50.0,
            anamorphic: 60.0,
        };
    }
    d.source().with_settings(s)
}

/// A viewport on a cell of `width x height` CSS pixels at `scale`, at Fit.
fn fit(width: f64, height: f64, scale: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(width, height, scale))
}

/// The same, at a fixed zoom about the middle, moved by `pan` CSS pixels.
fn zoomed(
    r: &Renderer,
    src: &PhotoSource,
    mut state: ViewportState,
    percent: f64,
    pan: (f64, f64),
) -> ViewportState {
    let image = r.final_size(src, &CancelToken::new()).expect("final size");
    state.set_mode(ViewportMode::Percent { value: percent }, image);
    state.input(
        &ViewportInput::Pan {
            dx: pan.0,
            dy: pan.1,
        },
        image,
    );
    state
}

fn map(
    r: &Renderer,
    src: &PhotoSource,
    state: &ViewportState,
    from: MapFrom,
    points: &[[f64; 2]],
) -> Vec<Option<[f64; 2]>> {
    r.map_points(src, state, from, points, &CancelToken::new())
        .expect("mapped")
}

/// A Mask only frame of Main.
fn mask_frame(r: &Renderer, src: &PhotoSource, state: &ViewportState) -> RenderedFrame {
    let mut state = state.clone();
    state.mask_view = Some(MaskView {
        node: 1,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
        face: None,
    });
    r.viewport_frame(
        src,
        &state,
        &FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
        &CancelToken::new(),
    )
    .expect("frame")
}

/// The centroid of the mask in a Mask only frame, in CSS pixels (pixel centres at `i + 0.5`), with the mask's
/// total weight in pixels. Only grey pixels count: the paper is not grey.
fn centroid(frame: &RenderedFrame, scale: f64) -> ([f64; 2], f64) {
    let (mut sum, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = (y * frame.width + x) * 4;
            let [r, g, b] = [frame.rgba[i], frame.rgba[i + 1], frame.rgba[i + 2]];
            if r != g || g != b {
                continue;
            }
            let m = f64::from(r) / 255.0;
            sum += m;
            sx += m * (x as f64 + 0.5);
            sy += m * (y as f64 + 0.5);
        }
    }
    assert!(
        sum > 20.0,
        "the mask has {sum} pixels of weight in the frame"
    );
    ([sx / sum / scale, sy / sum / scale], sum)
}

/// A point `margin` CSS pixels inside the cell.
fn well_inside(p: [f64; 2], cell: (f64, f64), margin: f64) -> bool {
    p[0] > margin && p[1] > margin && p[0] < cell.0 - margin && p[1] < cell.1 - margin
}

// ---- The two directions ---------------------------------------------------------------------------------

#[test]
fn photo_and_cell_are_each_others_inverse_in_every_geometry() {
    let d = photo();
    let r = renderer();
    // A grid of photo points; the ones that land on the paper have no way back.
    let points: Vec<[f64; 2]> = (1..10)
        .flat_map(|i| (1..10).map(move |j| [f64::from(i) / 10.0, f64::from(j) / 10.0]))
        .collect();
    let mut worst: f64 = 0.0;
    let mut round_trips = 0;
    for case in cases() {
        let src = source_of(&d, case, vec![]);
        for (view, state) in [
            ("fit", fit(300.0, 300.0, 1.0)),
            ("fit at 2x", fit(200.0, 150.0, 2.0)),
            (
                "zoomed and panned",
                zoomed(&r, &src, fit(300.0, 200.0, 1.0), 200.0, (-40.0, 25.0)),
            ),
        ] {
            let css = map(&r, &src, &state, MapFrom::Photo, &points);
            assert_eq!(css.len(), points.len());
            let css_points: Vec<[f64; 2]> = css
                .iter()
                .map(|p| p.expect("every photo point has a place"))
                .collect();
            let back = map(&r, &src, &state, MapFrom::Cell, &css_points);
            for ((photo, css), back) in points.iter().zip(&css_points).zip(&back) {
                // A point the straightening, the crop or the view puts outside the picture has no way back.
                let Some(back) = back else { continue };
                let e = (back[0] - photo[0]).hypot(back[1] - photo[1]);
                worst = worst.max(e);
                round_trips += 1;
                assert!(
                    e < 1e-5,
                    "{} / {view}: {photo:?} -> {css:?} -> {back:?}",
                    case.name
                );
            }
        }
    }
    eprintln!(
        "map_points: {round_trips} round trips, the largest is {worst:.2e} of the photo's side"
    );
    assert!(round_trips > 300, "the cases keep most points in view");
}

#[test]
fn a_cell_point_on_the_paper_has_no_place_in_the_photo_and_a_photo_point_outside_the_cell_has_one()
{
    let d = photo();
    let r = renderer();
    // A square cell: the 3:2 photo has paper above and below it.
    let src = source_of(&d, cases()[0], vec![]);
    let state = fit(300.0, 300.0, 1.0);
    let got = map(
        &r,
        &src,
        &state,
        MapFrom::Cell,
        &[
            [150.0, 10.0],
            [150.0, 290.0],
            [150.0, 150.0],
            [f64::NAN, 5.0],
        ],
    );
    assert_eq!(got[0], None, "the paper above");
    assert_eq!(got[1], None, "the paper below");
    let centre = got[2].expect("the middle of the photo");
    assert!((centre[0] - 0.5).abs() < 1e-9 && (centre[1] - 0.5).abs() < 1e-9);
    assert_eq!(got[3], None, "not a number");
    // The photo's corners are at the picture's corners; a point outside the photo is drawn outside it.
    let got = map(
        &r,
        &src,
        &state,
        MapFrom::Photo,
        &[[0.0, 0.0], [1.0, 1.0], [-0.2, 0.5], [0.5, f64::NAN]],
    );
    let near = |p: Option<[f64; 2]>, x: f64, y: f64| {
        let p = p.expect("a place");
        assert!((p[0] - x).abs() < 1e-6 && (p[1] - y).abs() < 1e-6, "{p:?}");
    };
    near(got[0], 0.0, 50.0);
    near(got[1], 300.0, 250.0);
    near(got[2], -60.0, 150.0);
    assert_eq!(got[3], None, "not a number");

    // A straightened photo leaves an empty corner in the picture: no place in the photo there.
    let tilted = Case {
        name: "tilted",
        crop: Some((FULL, 20.0)),
        turn: Orientation::Deg0,
        bend: false,
    };
    let src = source_of(&d, tilted, vec![]);
    let got = map(
        &r,
        &src,
        &state,
        MapFrom::Cell,
        &[[2.0, 52.0], [298.0, 248.0], [150.0, 150.0]],
    );
    assert_eq!(got[0], None, "the empty corner at the top left");
    assert_eq!(got[1], None, "the empty corner at the bottom right");
    assert!(got[2].is_some());
}

#[test]
fn a_zoomed_view_keeps_a_point_that_is_out_of_the_cell_where_it_is() {
    let d = photo();
    let r = renderer();
    let src = source_of(&d, cases()[0], vec![]);
    let state = zoomed(&r, &src, fit(300.0, 200.0, 1.0), 200.0, (0.0, 0.0));
    // At 200 % the 480 x 320 photo is 960 px wide in a 300 px cell: the photo's left edge is 330 px left of
    // the cell's, and a point at 40 % is 0.4 * 960 - 330 = 54 px in.
    let got = map(&r, &src, &state, MapFrom::Photo, &[[0.0, 0.5], [0.4, 0.5]]);
    let (a, b) = (got[0].expect("a place"), got[1].expect("a place"));
    assert!((a[0] + 330.0).abs() < 1e-6, "{a:?}");
    assert!(
        (b[0] - 54.0).abs() < 1e-6 && (b[1] - 100.0).abs() < 1e-6,
        "{b:?}"
    );
}

// ---- The mapping agrees with the pixels -----------------------------------------------------------------

/// The core draws a Radial's centre where `map_points` says its photo point is. The bound is one CSS pixel
/// (`docs/23` section 15, N-10: the centroid of a Radial is 0.000 to 0.05 px from the photo feature).
const CENTRE_BOUND_PX: f64 = 1.0;

#[test]
fn a_radial_is_drawn_where_the_core_maps_its_centre() {
    let d = photo();
    let r = renderer();
    let mut worst: f64 = 0.0;
    let mut compared = 0;
    for case in cases() {
        let src = source_of(&d, case, vec![radial(CENTRE, 0.04)]);
        for (view, state, cell) in [
            ("fit", fit(300.0, 300.0, 1.0), (300.0, 300.0)),
            ("fit at 2x", fit(200.0, 150.0, 2.0), (200.0, 150.0)),
        ] {
            let at = map(&r, &src, &state, MapFrom::Photo, &[CENTRE])[0].expect("a place");
            if !well_inside(at, cell, 25.0) {
                eprintln!(
                    "{} / {view}: the centre is at {at:?}, out of the test",
                    case.name
                );
                continue;
            }
            let frame = mask_frame(&r, &src, &state);
            let (found, weight) = centroid(&frame, state.rect.scale);
            let e = (found[0] - at[0]).hypot(found[1] - at[1]);
            worst = worst.max(e);
            compared += 1;
            eprintln!(
                "{} / {view}: mapped {at:.2?}, drawn {found:.2?} ({weight:.0} px of mask), {e:.3} px apart",
                case.name
            );
            assert!(
                e < CENTRE_BOUND_PX,
                "{} / {view}: {at:?} against {found:?}",
                case.name
            );
        }
    }
    eprintln!("the largest distance between a mapped centre and the drawn one is {worst:.3} px");
    // A case that is left out is not a case that passed: the six geometries at two device scales are all in view.
    assert_eq!(compared, 12, "every case is compared");
}

#[test]
fn a_radial_is_drawn_where_the_core_maps_its_centre_in_a_zoomed_and_panned_view() {
    let d = photo();
    let r = renderer();
    let mut compared = 0;
    for case in [cases()[0], cases()[5]] {
        let src = source_of(&d, case, vec![radial(CENTRE, 0.04)]);
        for pan in [(0.0, 0.0), (-60.0, 20.0)] {
            let state = zoomed(&r, &src, fit(300.0, 200.0, 1.0), 200.0, pan);
            let at = map(&r, &src, &state, MapFrom::Photo, &[CENTRE])[0].expect("a place");
            if !well_inside(at, (300.0, 200.0), 30.0) {
                eprintln!(
                    "{} / pan {pan:?}: the centre is at {at:?}, out of the test",
                    case.name
                );
                continue;
            }
            let (found, _) = centroid(&mask_frame(&r, &src, &state), 1.0);
            let e = (found[0] - at[0]).hypot(found[1] - at[1]);
            compared += 1;
            eprintln!("{} / zoomed, pan {pan:?}: {e:.3} px apart", case.name);
            assert!(
                e < CENTRE_BOUND_PX,
                "{} / pan {pan:?}: {at:?} against {found:?}",
                case.name
            );
        }
    }
    assert_eq!(compared, 4, "every case is compared");
}

#[test]
fn a_dragged_point_comes_back_to_the_photo_where_the_pixel_is() {
    // What a handle's drag does: the pointer's CSS point, mapped to the photo frame, is where a Radial put
    // there is drawn under the pointer.
    let d = photo();
    let r = renderer();
    let case = cases()[5];
    let state = fit(300.0, 300.0, 1.0);
    let pointer = [140.0, 170.0];
    let src = source_of(&d, case, vec![]);
    let placed = map(&r, &src, &state, MapFrom::Cell, &[pointer])[0].expect("on the photo");
    let src = source_of(&d, case, vec![radial(placed, 0.04)]);
    let (found, _) = centroid(&mask_frame(&r, &src, &state), 1.0);
    let e = (found[0] - pointer[0]).hypot(found[1] - pointer[1]);
    eprintln!("a Radial put at the pointer {pointer:?} is drawn {e:.3} px from it");
    assert!(e < CENTRE_BOUND_PX, "{found:?} against {pointer:?}");
}

// ---- mask_kinds -----------------------------------------------------------------------------------------

#[test]
fn this_machine_makes_the_shapes_brush_and_the_ranges_that_are_built() {
    // What needs no model (`mask_kinds()`: Background since N-15); Object is first where the SAM weights are (`mask_kinds_with`, N-18) and
    // Subject and People come after it where the Vision probe says yes (N-19).
    assert_eq!(
        mask_kinds(),
        [
            MaskKind::Background,
            MaskKind::Linear,
            MaskKind::Radial,
            MaskKind::Brush,
            MaskKind::ColorRange,
            MaskKind::BrightnessRange
        ],
        "Object needs its model, Subject and People the Vision probe"
    );
    assert_eq!(
        ym_core::settings::mask_kinds_with(
            ym_core::settings::MaskSupport::bare().with_object(true)
        )[0],
        MaskKind::Object
    );
}

#[test]
fn a_kind_is_written_as_the_rows_kind_is() {
    assert_eq!(MaskKind::ALL.map(MaskKind::name), MASK_KINDS);
    for kind in MaskKind::ALL {
        assert_eq!(
            serde_json::to_value(kind).expect("json"),
            json!(kind.name())
        );
        assert_eq!(
            serde_json::from_value::<MaskKind>(json!(kind.name())).expect("read"),
            kind
        );
    }
    assert!(serde_json::from_value::<MaskKind>(json!("lasso")).is_err());
    // Every built kind is a kind a row can have, and `of` names a row's shape.
    let s = doc(vec![radial([0.5, 0.5], 0.1)]);
    let row = &s.nodes.main.masks.items[0];
    assert_eq!(MaskKind::of(&row.shape), MaskKind::Radial);
    assert!(matches!(row.shape, MaskShape::Radial(_)));
}
