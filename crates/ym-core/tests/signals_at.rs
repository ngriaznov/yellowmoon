//! The `at` argument of the panel signals (`docs/01`, `docs/23` sections 7.5 and 9, task N-09): the
//! histogram, the chroma scatter and Pick colour read the signal at a node. Left out it is Main's,
//! as before nodes; `input` is the Input's output; a corrector is read at the input of its curve
//! (the histogram) or of its colour warp (the scatter and Pick colour), through the graph rewrite
//! of the node thumbnails, so no kernel is involved.
//!
//! The references are **independent of the rewrite**: a signal at node 2 of `Input → 1 → 2` equals
//! the signal at Main of a graph whose Main has node 2's trim and grade, and a neutral node reads
//! what the graph that ends before it reads.
mod common;

use common::*;
use serde_json::{Value, json};
use ym_core::pipeline::{
    ChromaSample, ChromaScatter, ChromaTap, HistogramData, PhotoSource, RenderError, Renderer,
    ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, NodeId};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn photo() -> Developed {
    let scene = Scene::new(300, 200, light(PROFILE, 5500.0), Layout::Patchwork, 0.4);
    Developed::new(&scene, MODEL)
}

fn renderer() -> Renderer {
    Renderer::neutral_only().expect("render thread")
}

/// A node-first document read over the default graph.
fn settings(items: Vec<Value>, links: Vec<Value>) -> DevelopSettings {
    let doc = json!({ "nodes": { "items": items, "links": links } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

fn picture(from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } })
}

/// Main with a grade of manual zeros: a node that changes nothing (an Auto Main resolves to the
/// photo's own numbers).
fn flat_main() -> Value {
    json!({ "id": 1, "grade": flat_grade() })
}

fn flat_grade() -> Value {
    json!({
        "contrast": { "kind": "manual", "value": 0.0 },
        "highlights": { "kind": "manual", "value": 0.0 },
        "shadows": { "kind": "manual", "value": 0.0 },
        "black": { "kind": "manual", "value": 0.0 },
        "saturation": { "kind": "manual", "value": 0.0 },
        "color_balance": {
            "shadows": { "kind": "manual", "point": [0.0, 0.0] },
            "highlights": { "kind": "manual", "point": [0.0, 0.0] }
        }
    })
}

/// The fields of the node under test: a trim, a primary grade, a colour warp and a curve.
fn busy() -> (Value, Value) {
    let trim = json!({ "exposure_ev": 0.8, "temperature": 40.0, "tint": -20.0 });
    let mut grade = flat_grade();
    let extra = json!({
        "contrast": { "kind": "manual", "value": 25.0 },
        "saturation": { "kind": "manual", "value": 30.0 },
        "color_warp": { "pins": [{
            "source": [0.05, -0.03], "target": [0.0, 0.08], "range": 0.1, "shadows": 1.0,
            "highlights": 1.0, "pivot_ev": 0.0, "exposure_ev": 0.0, "saturation": 0.0, "locked": false
        }] },
        "curve": { "luma": [[0.0, 0.0], [0.25, 0.15], [0.6, 0.7], [1.0, 1.0]] }
    });
    grade
        .as_object_mut()
        .expect("an object")
        .extend(extra.as_object().expect("an object").clone());
    (trim, grade)
}

/// `Input → 1 → 2 → Output`, node 2 with these fields.
fn line(trim: &Value, grade: &Value) -> DevelopSettings {
    settings(
        vec![
            flat_main(),
            json!({ "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": trim, "grade": grade }),
        ],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!("output")),
        ],
    )
}

/// `Input → 1 → Output` and a dead branch `1 → 2`: nothing reads node 2.
fn dead_branch(trim: &Value, grade: &Value) -> DevelopSettings {
    settings(
        vec![
            flat_main(),
            json!({ "id": 2, "kind": "corrector", "pos": [1.0, 1.0], "trim": trim, "grade": grade }),
        ],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!("output")),
            picture(json!(1), json!(2)),
        ],
    )
}

/// The graph whose Main has node 2's fields, which is what the signal at node 2 has to equal.
fn main_with(trim: &Value, grade: &Value) -> DevelopSettings {
    settings(
        vec![json!({ "id": 1, "trim": trim, "grade": grade })],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!("output")),
        ],
    )
}

fn at(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn histogram(r: &Renderer, src: &PhotoSource, at: Option<NodeId>) -> HistogramData {
    r.histogram_at(src, at, &CancelToken::new())
        .expect("a histogram")
}

fn scatter(r: &Renderer, src: &PhotoSource, at: Option<NodeId>) -> ChromaScatter {
    r.chroma_scatter_at(src, at, &CancelToken::new())
        .expect("a scatter")
}

const NODE_2: Option<NodeId> = Some(NodeId::Corrector(2));

#[test]
fn without_at_and_at_main_the_signal_is_mains_whatever_follows() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    for s in [line(&trim, &grade), dead_branch(&trim, &grade)] {
        let s = at(&src, s);
        let plain = r.histogram(&s, &CancelToken::new()).expect("histogram");
        assert_eq!(histogram(&r, &s, None), plain);
        assert_eq!(histogram(&r, &s, Some(NodeId::MAIN)), plain);
        let dots = r.chroma_scatter(&s, &CancelToken::new()).expect("scatter");
        assert_eq!(scatter(&r, &s, None), dots);
        assert_eq!(scatter(&r, &s, Some(NodeId::MAIN)), dots);
        // Main is flat, node 2 is not: what is read at node 2 is another signal.
        assert_ne!(histogram(&r, &s, NODE_2), plain);
    }
}

#[test]
fn the_histogram_at_a_node_is_the_signal_at_the_input_of_its_curve() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let reference = histogram(&r, &at(&src, main_with(&trim, &grade)), None);
    // The curve is in the node and not in what is read: a graph with and without it read the same.
    let mut straight = grade.clone();
    straight["curve"] = json!({});
    let uncurved = histogram(&r, &at(&src, main_with(&trim, &straight)), None);
    assert_eq!(
        reference, uncurved,
        "Main's own reading stops before its curve"
    );
    for (what, s) in [
        ("a line", line(&trim, &grade)),
        ("a dead branch", dead_branch(&trim, &grade)),
    ] {
        assert_eq!(
            histogram(&r, &at(&src, s), NODE_2),
            reference,
            "{what}: node 2 reads what Main reads when it has node 2's fields"
        );
    }
}

#[test]
fn a_neutral_node_reads_the_graph_that_ends_before_it() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let before = at(
        &src,
        settings(
            vec![flat_main()],
            vec![
                picture(json!("input"), json!(1)),
                picture(json!(1), json!("output")),
            ],
        ),
    );
    let with_node = at(&src, line(&json!({}), &json!({})));
    // The picture that enters node 2 is Main's; node 2 changes nothing, so its curve input is that.
    assert_eq!(
        histogram(&r, &with_node, NODE_2),
        histogram(&r, &before, None)
    );
    assert_eq!(scatter(&r, &with_node, NODE_2), scatter(&r, &before, None));
}

#[test]
fn the_scatter_at_a_node_is_the_colours_at_the_input_of_its_warp() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let mut unwarped = grade.clone();
    unwarped["color_warp"] = json!({ "pins": [] });
    let reference = scatter(&r, &at(&src, main_with(&trim, &unwarped)), None);
    assert!(!reference.points.is_empty());
    // The pins do not move the colours they take hold of (Main's own reading is the same).
    assert_eq!(
        scatter(&r, &at(&src, main_with(&trim, &grade)), None),
        reference
    );
    for s in [line(&trim, &grade), dead_branch(&trim, &grade)] {
        assert_eq!(scatter(&r, &at(&src, s), NODE_2), reference);
    }
    // The trim is in the colours it reads: another temperature, another scatter.
    let cooler = json!({ "exposure_ev": 0.8, "temperature": -40.0, "tint": -20.0 });
    assert_ne!(
        scatter(&r, &at(&src, line(&cooler, &grade)), NODE_2),
        reference
    );
}

#[test]
fn at_the_input_the_signal_is_the_inputs_output() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let input = Some(NodeId::Input);
    let plain = histogram(&r, &at(&src, line(&json!({}), &json!({}))), input);
    // Whatever the correctors do, the Input's output is the same...
    assert_eq!(histogram(&r, &at(&src, line(&trim, &grade)), input), plain);
    assert_eq!(
        scatter(&r, &at(&src, line(&trim, &grade)), input),
        scatter(&r, &at(&src, line(&json!({}), &json!({}))), input)
    );
    // ...and it is not Main's reading once Main grades.
    let graded = at(&src, main_with(&trim, &grade));
    assert_ne!(histogram(&r, &graded, input), histogram(&r, &graded, None));
    // The Input's own settings are in it.
    let mut brighter = src.settings.clone();
    brighter.input_mut().exposure =
        ym_core::settings::AutoOr::Manual(ym_core::settings::ExposureValue { ev: 1.5 });
    assert_ne!(histogram(&r, &src.with_settings(brighter), input), plain);
}

#[test]
fn a_node_the_photo_does_not_have_and_the_output_are_refused_in_words() {
    let d = photo();
    let src = at(&d.source(), line(&json!({}), &json!({})));
    let r = renderer();
    let missing = r
        .histogram_at(&src, Some(NodeId::Corrector(9)), &CancelToken::new())
        .expect_err("no node 9");
    assert!(
        matches!(&missing, RenderError::Color(m) if m == "This photo has no node 9."),
        "{missing}"
    );
    let output = r
        .chroma_scatter_at(&src, Some(NodeId::Output), &CancelToken::new())
        .expect_err("the output");
    assert!(
        matches!(&output, RenderError::Color(m) if m == "The output has no picture to read here."),
        "{output}"
    );
}

#[test]
fn pick_colour_at_a_node_reads_the_chroma_at_the_input_of_its_warp() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let mut unwarped = grade.clone();
    unwarped["color_warp"] = json!({ "pins": [] });
    // A 300 x 300 cell: the 3:2 photo is letterboxed above and below.
    let cell = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
    let chroma = |s: DevelopSettings, node: Option<NodeId>, point: (f64, f64)| {
        r.sample_chroma_at(
            &at(&src, s),
            node,
            ChromaTap::WarpInput,
            &cell,
            point,
            &CancelToken::new(),
        )
        .expect("a sample")
    };
    for point in [(75.5, 131.5), (225.5, 94.5), (25.5, 94.5), (125.5, 169.5)] {
        let reference = chroma(main_with(&trim, &unwarped), None, point).expect("inside");
        // The same colour; a flat node 1 in front costs a rounding of the last digits.
        let near = |got: Option<ChromaSample>| {
            let got = got.expect("inside");
            assert!(
                (got.a - reference.a).abs() < 1e-9 && (got.b - reference.b).abs() < 1e-9,
                "{point:?}: {got:?} against {reference:?}"
            );
        };
        near(chroma(line(&trim, &grade), NODE_2, point));
        near(chroma(dead_branch(&trim, &grade), NODE_2, point));
        // Left out is Main, as ever.
        let mains = chroma(line(&trim, &grade), None, point).expect("inside");
        assert_eq!(
            Some(mains),
            r.sample_chroma(
                &at(&src, line(&trim, &grade)),
                &cell,
                point,
                &CancelToken::new()
            )
            .expect("a sample")
        );
        assert_ne!(mains, reference, "{point:?}: Main is flat, node 2 is not");
    }
    // The paper around the photo has no colour, at any node.
    assert_eq!(chroma(line(&trim, &grade), NODE_2, (150.0, 5.0)), None);
    assert!(
        r.sample_chroma_at(
            &at(&src, line(&trim, &grade)),
            Some(NodeId::Corrector(7)),
            ChromaTap::WarpInput,
            &cell,
            (75.5, 131.5),
            &CancelToken::new()
        )
        .is_err()
    );
}

// ---- The node's own signal, whatever its mask does (N-13) -------------------------------------------

/// A Radial row that covers part of the photo with a soft edge: `m` is neither 0 nor 1 over most
/// of it.
fn radial_row() -> Value {
    json!({
        "kind": "radial", "center": [0.4, 0.5], "radii": [0.2, 0.15], "angle_deg": 20.0,
        "feather": 0.5
    })
}

/// `Input -> 1 -> 2 -> Output`, node 2 with these fields and a mask, and (with `key`) node 3 with a
/// Linear row feeding node 2's key input.
fn masked_line(trim: &Value, grade: &Value, invert: bool, key: bool) -> DevelopSettings {
    let mut items = vec![
        flat_main(),
        json!({
            "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": trim, "grade": grade,
            "masks": { "invert": invert, "items": [radial_row()] }
        }),
    ];
    let mut links = vec![
        picture(json!("input"), json!(1)),
        picture(json!(1), json!(2)),
        picture(json!(2), json!("output")),
    ];
    if key {
        items.push(json!({
            "id": 3, "kind": "corrector", "pos": [1.0, 1.0],
            "masks": { "items": [{ "kind": "linear", "center": [0.6, 0.5], "angle_deg": 70.0, "feather": 0.4 }] }
        }));
        links.push(json!({
            "from": { "node": 3, "port": "key" }, "to": { "node": 2, "port": "key" }
        }));
    }
    settings(items, links)
}

#[test]
fn a_masked_nodes_histogram_and_scatter_are_its_own_signal_not_the_blend() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let plain = at(&src, line(&trim, &grade));
    let histogram_plain = histogram(&r, &plain, NODE_2);
    let scatter_plain = scatter(&r, &plain, NODE_2);
    assert!(!scatter_plain.points.is_empty());
    for (what, s) in [
        ("a Radial row", masked_line(&trim, &grade, false, false)),
        ("a Radial row and the node's Invert", masked_line(&trim, &grade, true, false)),
        ("a Radial row and a key input", masked_line(&trim, &grade, false, true)),
    ] {
        let s = at(&src, s);
        assert_eq!(
            histogram(&r, &s, NODE_2),
            histogram_plain,
            "{what}: the histogram is of the node's own C9b"
        );
        assert_eq!(
            scatter(&r, &s, NODE_2),
            scatter_plain,
            "{what}: the scatter is of the node's own C9"
        );
    }
}

#[test]
fn a_masked_nodes_pick_colour_at_its_warp_is_its_own_signal() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    for point in POINTS {
        let want = tapped(&r, &src, line(&trim, &grade), NODE_2, ChromaTap::WarpInput, point);
        for (what, s) in [
            ("a Radial row", masked_line(&trim, &grade, false, false)),
            ("a Radial row and a key input", masked_line(&trim, &grade, false, true)),
        ] {
            let got = tapped(&r, &src, s, NODE_2, ChromaTap::WarpInput, point);
            assert_eq!(got, want, "{what} at {point:?}");
        }
    }
}

// ---- Pick colour at the node's input (`ChromaTap::NodeInput`, for a Colour range) -----------------

/// The chroma under `point` of a 300 x 300 cell, read at `tap` of `node`.
fn tapped(
    r: &Renderer,
    src: &PhotoSource,
    s: DevelopSettings,
    node: Option<NodeId>,
    tap: ChromaTap,
    point: (f64, f64),
) -> ChromaSample {
    let cell = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
    r.sample_chroma_at(&at(src, s), node, tap, &cell, point, &CancelToken::new())
        .expect("a sample")
        .expect("inside the photo")
}

/// Two neutral patches and three coloured ones of the 3:2 photo in a 300 x 300 cell.
const POINTS: [(f64, f64); 4] = [(75.5, 131.5), (225.5, 94.5), (25.5, 94.5), (125.5, 131.5)];

fn near(got: ChromaSample, want: ChromaSample, what: &str) {
    assert!(
        (got.a - want.a).abs() < 1e-9 && (got.b - want.b).abs() < 1e-9,
        "{what}: {got:?} against {want:?}"
    );
}

fn apart(a: ChromaSample, b: ChromaSample) -> f64 {
    (a.a - b.a).hypot(a.b - b.b)
}

#[test]
fn a_node_at_its_identity_has_one_chroma_at_both_taps() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    for point in POINTS {
        for s in [
            line(&json!({}), &json!({})),
            dead_branch(&json!({}), &json!({})),
        ] {
            let warp = tapped(&r, &src, s.clone(), NODE_2, ChromaTap::WarpInput, point);
            let node = tapped(&r, &src, s, NODE_2, ChromaTap::NodeInput, point);
            assert_eq!(node, warp, "{point:?}: nothing lies between the two");
        }
    }
}

#[test]
fn at_a_graded_node_the_input_is_the_picture_it_reads_and_the_warp_input_is_not() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    // Main has its own trim and grade (no warp, no curve), so that what enters node 2 is not the Input's output.
    let mut main_grade = grade.clone();
    main_grade["color_warp"] = json!({ "pins": [] });
    main_grade["curve"] = json!({});
    let main_trim = json!({ "temperature": -30.0, "tint": 25.0 });
    let main = json!({ "id": 1, "trim": main_trim, "grade": main_grade });
    let node_2 =
        json!({ "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": trim, "grade": grade });
    let line_graph = settings(
        vec![main.clone(), node_2.clone()],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!("output")),
        ],
    );
    let dead_graph = settings(
        vec![main.clone(), node_2],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!("output")),
            picture(json!(1), json!(2)),
        ],
    );
    // The reference does not use node 2: Main alone, read at the input of its warp (C9: its trim and grade, which is
    // its output here, since the warp and the curve are empty).
    let alone = settings(
        vec![main],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!("output")),
        ],
    );
    for point in POINTS {
        let want = tapped(&r, &src, alone.clone(), None, ChromaTap::WarpInput, point);
        for (what, s) in [("a line", &line_graph), ("a dead branch", &dead_graph)] {
            let node = tapped(&r, &src, s.clone(), NODE_2, ChromaTap::NodeInput, point);
            near(
                node,
                want,
                &format!("{point:?}, {what}: node 2's input is Main's output"),
            );
            let warp = tapped(&r, &src, s.clone(), NODE_2, ChromaTap::WarpInput, point);
            assert!(
                apart(warp, node) > 0.005,
                "{point:?}, {what}: the trim and the grade of node 2 lie between the two ({warp:?}, {node:?})"
            );
        }
    }
}

#[test]
fn at_main_and_the_input_the_node_input_is_the_inputs_output() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let graded = main_with(&trim, &grade);
    let flat = settings(
        vec![flat_main()],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!("output")),
        ],
    );
    for point in POINTS {
        let input = tapped(
            &r,
            &src,
            graded.clone(),
            Some(NodeId::Input),
            ChromaTap::NodeInput,
            point,
        );
        // Main always takes its picture from the Input: its input is the Input's output, whatever Main does.
        for main in [None, Some(NodeId::MAIN)] {
            let got = tapped(&r, &src, graded.clone(), main, ChromaTap::NodeInput, point);
            assert_eq!(got, input, "{point:?}: {main:?}");
        }
        // A flat Main passes the Input's output through, so its own warp input is that too.
        near(
            tapped(&r, &src, flat.clone(), None, ChromaTap::WarpInput, point),
            input,
            "a flat Main",
        );
        // A graded one does not: its warp input is after its trim and its grade.
        let warp = tapped(&r, &src, graded.clone(), None, ChromaTap::WarpInput, point);
        assert!(apart(warp, input) > 0.005, "{point:?}: {warp:?} {input:?}");
        // At the Input both taps are the Input's output.
        assert_eq!(
            tapped(
                &r,
                &src,
                graded.clone(),
                Some(NodeId::Input),
                ChromaTap::WarpInput,
                point
            ),
            input
        );
    }
}

#[test]
fn the_node_input_tap_refuses_the_output_and_a_missing_node_in_words() {
    let d = photo();
    let src = at(&d.source(), line(&json!({}), &json!({})));
    let r = renderer();
    let cell = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
    let ask = |node| {
        r.sample_chroma_at(
            &src,
            Some(node),
            ChromaTap::NodeInput,
            &cell,
            (75.5, 131.5),
            &CancelToken::new(),
        )
    };
    let missing = ask(NodeId::Corrector(9)).expect_err("no node 9");
    assert!(
        matches!(&missing, RenderError::Color(m) if m == "This photo has no node 9."),
        "{missing}"
    );
    let output = ask(NodeId::Output).expect_err("the output");
    assert!(
        matches!(&output, RenderError::Color(m) if m == "The output has no picture to read here."),
        "{output}"
    );
    // The paper around the photo has no colour at this tap either.
    let none = r
        .sample_chroma_at(
            &src,
            NODE_2,
            ChromaTap::NodeInput,
            &cell,
            (150.0, 5.0),
            &CancelToken::new(),
        )
        .expect("a sample");
    assert_eq!(none, None);
}

#[test]
fn the_node_input_does_not_depend_on_the_nodes_masks_or_bypass() {
    let d = photo();
    let src = d.source();
    let r = renderer();
    let (trim, grade) = busy();
    let masked = |bypass: bool, rows: Value| {
        settings(
            vec![
                flat_main(),
                json!({
                    "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "bypass": bypass,
                    "trim": trim, "grade": grade, "masks": { "invert": false, "items": rows }
                }),
            ],
            vec![
                picture(json!("input"), json!(1)),
                picture(json!(1), json!(2)),
                picture(json!(2), json!("output")),
            ],
        )
    };
    let radial = json!([{ "kind": "radial", "mode": "add", "center": [0.5, 0.5], "radii": [0.2, 0.2], "feather": 0.2 }]);
    let range = json!([{ "kind": "color_range", "mode": "add", "center": [0.0, 0.0], "width": 0.01, "softness": 0.0 }]);
    for point in POINTS {
        let plain = tapped(
            &r,
            &src,
            line(&json!({}), &json!({})),
            NODE_2,
            ChromaTap::NodeInput,
            point,
        );
        for (what, s) in [
            ("a Radial", masked(false, radial.clone())),
            ("a Colour range", masked(false, range.clone())),
            ("a bypass", masked(true, json!([]))),
        ] {
            near(
                tapped(&r, &src, s, NODE_2, ChromaTap::NodeInput, point),
                plain,
                &format!("{point:?}, {what}"),
            );
        }
    }
}
