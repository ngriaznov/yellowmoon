//! The Reference Grade node's settings (`docs/34-reference-grade.md` section 3): the format, the defaults and clamping, every
//! refusal and its text, the hue points, the element members, Strength, patches (the fit among them as one patch), Reset, undo,
//! Copy and Paste, the XMP text, the JSON schema and the kind.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::session::PhotoId;
use ym_core::settings::{
    DevelopSettings, ELEMENT_CHROMA_RANGE, ELEMENT_EXPOSURE_RANGE_EV, ELEMENT_HUE_CAPS_DEG,
    ELEMENT_NAMES, FieldRef, GraphError, HUE_VS_HUE_RANGE_DEG, HUE_VS_SAT_RANGE, NodeId, NodeKind,
    NodePatch, Port, ReferenceGradeNode, ResolvedSettings, SettingsError, SettingsField,
    SettingsPatch, ports_in, ports_out,
};
use ym_looks::library::LookId;

fn rid(n: u32) -> NodeId {
    NodeId::Corrector(n)
}

fn link(from: Value, fp: &str, to: Value, tp: &str) -> Value {
    json!({ "from": { "node": from, "port": fp }, "to": { "node": to, "port": tp } })
}

fn line(ids: &[u32]) -> Vec<Value> {
    let mut links = Vec::new();
    let mut from = json!("input");
    for id in ids {
        links.push(link(from, "picture", json!(id), "picture"));
        from = json!(id);
    }
    links.push(link(from, "picture", json!("output"), "picture"));
    links
}

fn doc(items: Vec<Value>, links: Vec<Value>) -> Value {
    json!({ "nodes": { "items": items, "links": links } })
}

fn read(doc: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    DevelopSettings::from_value(doc, &LookId::neutral())
}

fn with_node(body: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    let mut node = json!({ "id": 2, "kind": "reference_grade" });
    if let (Some(n), Some(b)) = (node.as_object_mut(), body.as_object()) {
        n.extend(b.clone());
    }
    read(doc(vec![node], line(&[1, 2])))
}

fn node_of(s: &DevelopSettings) -> &ReferenceGradeNode {
    s.nodes.reference_grade(2).expect("reference grade node 2")
}

fn base() -> DevelopSettings {
    with_node(json!({})).expect("a default node").0
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

fn reported(clamped: &[FieldRef], field: SettingsField) -> bool {
    clamped.contains(&FieldRef {
        node: rid(2),
        field,
    })
}

/// The node of `docs/34` section 3.1, the example.
fn example() -> Value {
    json!({ "id": 2, "kind": "reference_grade", "name": "Match", "bypass": false, "pos": [2, 0],
      "trim": { "exposure_ev": 0, "temperature": 0, "tint": 0 },
      "grade": {
        "contrast": { "kind": "manual", "value": 0 }, "brightness": 0,
        "highlights": { "kind": "manual", "value": 0 }, "shadows": { "kind": "manual", "value": 0 },
        "white": 0, "black": { "kind": "manual", "value": 0 },
        "color_balance": { "shadows": { "kind": "manual", "point": [0, 0] }, "midtones": [0, 0],
                           "highlights": { "kind": "manual", "point": [0, 0] } },
        "saturation": { "kind": "manual", "value": -12 },
        "color_warp": { "pins": [] },
        "curve": { "luma": [[0, 0], [0.2136, 0.2050], [0.2784, 0.2784], [0.4080, 0.4310], [1, 1]],
                   "red": [[0, 0], [0.2784, 0.2840], [1, 1]],
                   "green": [[0, 0], [0.2784, 0.2784], [1, 1]],
                   "blue": [[0, 0], [0.2784, 0.2700], [1, 1]] } },
      "hue_curves": {
        "hue_vs_hue": [ { "hue_deg": 0, "value": -4 }, { "hue_deg": 120, "value": 6 }, { "hue_deg": 240, "value": 0 } ],
        "hue_vs_sat": [ { "hue_deg": 0, "value": 1.05 }, { "hue_deg": 120, "value": 0.85 }, { "hue_deg": 240, "value": 1 } ] },
      "elements": {
        "sky":        { "exposure_ev": -0.30, "hue_deg": 8,  "chroma": 1.10 },
        "vegetation": { "exposure_ev": 0,     "hue_deg": -6, "chroma": 0.90 },
        "water":      { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
        "ground":     { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
        "built":      { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
        "skin":       { "exposure_ev": 0,     "hue_deg": 3,  "chroma": 1 },
        "subject":    { "exposure_ev": 0.20 } },
      "strength": 100, "skin_protection": true,
      "masks": { "invert": false, "items": [] } })
}

#[test]
fn a_new_node_has_the_defaults_of_the_spec_writes_every_field_and_changes_no_pixel() {
    let (s, clamped) = with_node(json!({})).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    assert_eq!(n.name, "Match");
    assert!(!n.bypass);
    assert_eq!(n.strength, 100.0);
    assert!(n.skin_protection);
    assert!(n.hue_curves.is_default() && n.elements.is_default());
    assert!(n.masks.items.is_empty() && !n.masks.invert);
    assert!(n.is_inert());
    // Every field is written (a new kind), in the spec's names and no others.
    let v = serde_json::to_value(&s).unwrap();
    let item = v["nodes"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"] == json!(2))
        .unwrap()
        .clone();
    let mut keys: Vec<&str> = item
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "bypass",
            "elements",
            "grade",
            "hue_curves",
            "id",
            "kind",
            "masks",
            "name",
            "pos",
            "skin_protection",
            "strength",
            "trim"
        ]
    );
    assert_eq!(item["kind"], json!("reference_grade"));
    assert_eq!(
        item["hue_curves"],
        json!({ "hue_vs_hue": [], "hue_vs_sat": [] })
    );
    assert_eq!(
        item["elements"]["sky"],
        json!({ "exposure_ev": 0.0, "hue_deg": 0.0, "chroma": 1.0 })
    );
    assert_eq!(item["elements"]["subject"], json!({ "exposure_ev": 0.0 }));
    assert_eq!(item["strength"], json!(100.0));
    assert_eq!(item["skin_protection"], json!(true));
    // A default node's effective values are its stored values, at any Strength of 100.
    let e = n.effective();
    assert!(e.hue_and_elements_are_identity());
}

#[test]
fn the_example_of_the_spec_reads_and_writes_back_the_same() {
    let (s, clamped) = read(doc(vec![example()], line(&[1, 2]))).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    assert_eq!(n.hue_curves.hue_vs_hue.len(), 3);
    assert_eq!(n.hue_curves.hue_vs_sat[1].value, 0.85);
    assert_eq!(n.elements.sky.exposure_ev, -0.30);
    assert_eq!(n.elements.skin.hue_deg, 3.0);
    assert_eq!(n.elements.subject.exposure_ev, 0.20);
    assert_eq!(n.grade.curve.luma.len(), 5);
    assert!(!n.is_inert());
    let text = serde_json::to_string(&s).unwrap();
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, s);
    // A member or a field left out reads as its default.
    let (s, _) = with_node(json!({ "elements": { "sky": { "hue_deg": 5 } } })).unwrap();
    let e = node_of(&s).elements;
    assert_eq!(
        (e.sky.exposure_ev, e.sky.hue_deg, e.sky.chroma),
        (0.0, 5.0, 1.0)
    );
    assert_eq!(e.water.chroma, 1.0);
}

#[test]
fn the_ports_are_a_correctors_and_ids_are_shared() {
    assert_eq!(
        ports_in(NodeKind::ReferenceGrade),
        [Port::Picture, Port::Key]
    );
    assert_eq!(
        ports_out(NodeKind::ReferenceGrade),
        [Port::Picture, Port::Key]
    );
    // A key link into and out of the node is valid.
    let mut links = line(&[1, 2]);
    links.push(link(json!(1), "key", json!(2), "key"));
    read(doc(
        vec![json!({ "id": 2, "kind": "reference_grade" })],
        links,
    ))
    .unwrap();
    // Two nodes of one id.
    let e = read(doc(
        vec![
            json!({ "id": 2, "kind": "reference_grade" }),
            json!({ "id": 2, "kind": "color_uniform" }),
        ],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert!(matches!(
        e,
        SettingsError::Graph(GraphError::DuplicateId(_))
    ));
    // The kind needs its own id and its `kind`.
    let e = read(doc(vec![json!({ "id": 2 })], line(&[1, 2]))).unwrap_err();
    assert!(matches!(e, SettingsError::Graph(GraphError::NeedsKind(_))));
    // The 64-node limit counts this kind.
    let mut items: Vec<Value> = (2..=64)
        .map(|id| json!({ "id": id, "kind": "reference_grade" }))
        .collect();
    let ids: Vec<u32> = (1..=64).collect();
    read(doc(items.clone(), line(&ids))).expect("64 nodes");
    items.push(json!({ "id": 65, "kind": "reference_grade" }));
    let ids: Vec<u32> = (1..=65).collect();
    let e = read(doc(items, line(&ids))).unwrap_err();
    assert_eq!(e.to_string(), "A photo can have up to 64 nodes.");
}

#[test]
fn an_unknown_field_or_element_is_refused_with_its_text() {
    let e = with_node(json!({ "reference": "a.jpg" })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Reference Grade node has no field called “reference”."
    );
    // A field of another kind is told from a typo by the same sentence.
    let e = with_node(json!({ "target": null })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Reference Grade node has no field called “target”."
    );
    let e = with_node(json!({ "elements": { "ceiling": {} } })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Reference Grade node has no element called “ceiling”."
    );
    // The subject has an exposure alone: a hue or a chroma on it is an unknown field of that member.
    for field in ["hue_deg", "chroma"] {
        let e = with_node(json!({ "elements": { "subject": { field: 1 } } })).unwrap_err();
        assert!(matches!(e, SettingsError::Parse(_)), "{field}: {e}");
    }
    let e = with_node(json!({ "elements": { "sky": { "saturation": 1 } } })).unwrap_err();
    assert!(matches!(e, SettingsError::Parse(_)));
    // Other kinds refuse the node's fields.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "corrector", "hue_curves": {} })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "A corrector has no field called “hue_curves”."
    );
}

#[test]
fn a_group_or_a_point_of_the_wrong_shape_is_a_parse_error() {
    for body in [
        json!({ "hue_curves": [] }),
        json!({ "hue_curves": 3 }),
        json!({ "hue_curves": { "hue_vs_hue": 3 } }),
        json!({ "hue_curves": { "hue_vs_hue": [[0, 1]] } }),
        json!({ "hue_curves": { "hue_vs_hue": [{ "hue_deg": "red", "value": 1 }] } }),
        json!({ "hue_curves": { "hue_vs_hue": [{ "hue_deg": 1 }] } }),
        json!({ "hue_curves": { "hue_vs_luma": [] } }),
        json!({ "elements": [] }),
        json!({ "elements": { "sky": [] } }),
        json!({ "elements": { "sky": { "chroma": "high" } } }),
        json!({ "skin_protection": "yes" }),
        json!({ "strength": "all" }),
    ] {
        let e = with_node(body.clone()).unwrap_err();
        assert!(matches!(e, SettingsError::Parse(_)), "{body}: {e}");
    }
}

#[test]
fn hue_points_are_read_modulo_360_sorted_limited_and_never_duplicated() {
    // The hue is read modulo 360 into [0, 360), not reported; the list is sorted by hue.
    let (s, clamped) = with_node(json!({ "hue_curves": { "hue_vs_hue": [
        { "hue_deg": 725, "value": 3 }, { "hue_deg": -10, "value": 4 }, { "hue_deg": 100, "value": 5 } ] } }))
    .unwrap();
    assert!(!reported(&clamped, SettingsField::HueCurves));
    let h: Vec<(f64, f64)> = node_of(&s)
        .hue_curves
        .hue_vs_hue
        .iter()
        .map(|p| (p.hue_deg, p.value))
        .collect();
    assert_eq!(h, [(5.0, 3.0), (100.0, 5.0), (350.0, 4.0)]);
    // Values are clamped and reported, at the parametric look's limits.
    let (s, clamped) = with_node(json!({ "hue_curves": {
        "hue_vs_hue": [ { "hue_deg": 0, "value": 200 }, { "hue_deg": 90, "value": -300 } ],
        "hue_vs_sat": [ { "hue_deg": 0, "value": 9 }, { "hue_deg": 90, "value": -1 } ] } }))
    .unwrap();
    assert!(reported(&clamped, SettingsField::HueCurves));
    let c = &node_of(&s).hue_curves;
    assert_eq!(
        (c.hue_vs_hue[0].value, c.hue_vs_hue[1].value),
        (HUE_VS_HUE_RANGE_DEG.1, HUE_VS_HUE_RANGE_DEG.0)
    );
    assert_eq!(
        (c.hue_vs_sat[0].value, c.hue_vs_sat[1].value),
        (HUE_VS_SAT_RANGE.1, HUE_VS_SAT_RANGE.0)
    );
    // Two points at one hue after the reduction are refused, whichever order they come in.
    let e = with_node(json!({ "hue_curves": { "hue_vs_sat": [
        { "hue_deg": 0, "value": 1 }, { "hue_deg": 360, "value": 2 } ] } }))
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Two points of a hue curve are at the same hue."
    );
    // 16 points are fine and 17 are refused.
    let points = |n: usize| -> Vec<Value> {
        (0..n)
            .map(|i| json!({ "hue_deg": 20 * i, "value": 1 }))
            .collect()
    };
    with_node(json!({ "hue_curves": { "hue_vs_sat": points(16) } })).unwrap();
    let e = with_node(json!({ "hue_curves": { "hue_vs_sat": points(17) } })).unwrap_err();
    assert_eq!(e.to_string(), "A hue curve can have up to 16 points.");
}

#[test]
fn element_values_are_clamped_per_element_and_reported_by_field() {
    let doc = json!({ "elements": {
        "sky":        { "exposure_ev": 5, "hue_deg": 90, "chroma": 3 },
        "vegetation": { "exposure_ev": -5, "hue_deg": -90, "chroma": 0.1 },
        "water":      { "hue_deg": 36 },
        "ground":     { "hue_deg": 31 },
        "built":      { "hue_deg": -26 },
        "skin":       { "hue_deg": 16 },
        "subject":    { "exposure_ev": 4 } } });
    let (s, clamped) = with_node(doc).unwrap();
    assert!(reported(&clamped, SettingsField::Elements));
    let e = node_of(&s).elements;
    assert_eq!(
        (e.sky.exposure_ev, e.sky.hue_deg, e.sky.chroma),
        (1.0, 40.0, 1.6)
    );
    assert_eq!(
        (
            e.vegetation.exposure_ev,
            e.vegetation.hue_deg,
            e.vegetation.chroma
        ),
        (-1.0, -35.0, 0.5)
    );
    assert_eq!(
        [
            e.water.hue_deg,
            e.ground.hue_deg,
            e.built.hue_deg,
            e.skin.hue_deg
        ],
        [35.0, 30.0, -25.0, 15.0]
    );
    assert_eq!(e.subject.exposure_ev, 1.0);
    // The caps are the spec's, and a value on the cap is in range.
    assert_eq!(ELEMENT_HUE_CAPS_DEG, [40.0, 35.0, 35.0, 30.0, 25.0, 15.0]);
    assert_eq!(ELEMENT_EXPOSURE_RANGE_EV, (-1.0, 1.0));
    assert_eq!(ELEMENT_CHROMA_RANGE, (0.5, 1.6));
    assert_eq!(
        ELEMENT_NAMES,
        [
            "sky",
            "vegetation",
            "water",
            "ground",
            "built",
            "skin",
            "subject"
        ]
    );
    let (_, clamped) = with_node(json!({ "elements": { "skin": { "hue_deg": 15 } } })).unwrap();
    assert!(!reported(&clamped, SettingsField::Elements));
    // Strength.
    let (s, clamped) = with_node(json!({ "strength": 150 })).unwrap();
    assert!(reported(&clamped, SettingsField::Strength));
    assert_eq!(node_of(&s).strength, 100.0);
    let (s, _) = with_node(json!({ "strength": -4 })).unwrap();
    assert_eq!(node_of(&s).strength, 0.0);
}

#[test]
fn a_number_that_is_not_finite_is_refused_and_nothing_is_applied() {
    let mut s = base();
    let before = s.clone();
    for patch in [
        NodePatch {
            strength: Some(f64::NAN),
            ..NodePatch::default()
        },
        NodePatch {
            elements: Some(ym_core::settings::ElementsPatch {
                sky: Some(ym_core::settings::ElementPatch {
                    hue_deg: Some(f64::INFINITY),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..NodePatch::default()
        },
        NodePatch {
            hue_curves: Some(ym_core::settings::HueCurvesPatch {
                hue_vs_sat: Some(vec![ym_core::settings::HuePoint {
                    hue_deg: f64::NAN,
                    value: 1.0,
                }]),
                ..Default::default()
            }),
            ..NodePatch::default()
        },
    ] {
        let e = s
            .apply_patch(&SettingsPatch::node(rid(2), patch), None)
            .unwrap_err();
        assert!(matches!(e, SettingsError::NotFinite(_)), "{e}");
        assert_eq!(s, before, "atomic");
    }
}

#[test]
fn the_name_is_trimmed_cut_and_an_empty_one_reads_as_the_default() {
    assert_eq!(
        node_of(&with_node(json!({ "name": "  " })).unwrap().0).name,
        "Match"
    );
    let long = "x".repeat(40);
    assert_eq!(
        node_of(&with_node(json!({ "name": long })).unwrap().0)
            .name
            .chars()
            .count(),
        32
    );
}

#[test]
fn strength_scales_each_fitted_field_by_the_formulas_of_the_spec() {
    let (s, _) = read(doc(vec![example()], line(&[1, 2]))).unwrap();
    let mut n = node_of(&s).clone();
    n.hue_curves.hue_vs_hue[0].value = -40.0;
    n.hue_curves.hue_vs_sat[1].value = 0.5;
    n.elements.sky.chroma = 1.6;
    n.elements.vegetation.chroma = 0.5;
    n.elements.sky.hue_deg = 20.0;
    n.elements.subject.exposure_ev = 0.8;
    for pct in [0.0, 1.0, 25.0, 50.0, 73.0, 99.0, 100.0] {
        n.strength = pct;
        let k = pct / 100.0;
        let e = n.effective();
        // A curve's knot (x, y) is (x, x + k (y - x)); monotone stays monotone.
        for (stored, eff) in [
            (&n.grade.curve.luma, &e.grade.curve.luma),
            (&n.grade.curve.red, &e.grade.curve.red),
            (&n.grade.curve.blue, &e.grade.curve.blue),
        ] {
            for (a, b) in stored.iter().zip(eff) {
                assert_eq!(a[0], b[0]);
                assert!((b[1] - (a[0] + k * (a[1] - a[0]))).abs() <= 1e-12);
            }
            assert!(eff.windows(2).all(|w| w[1][1] >= w[0][1]), "{pct}");
        }
        // Saturation is linear in s.
        let want = k * (-12.0);
        match e.grade.saturation {
            ym_core::settings::AutoOr::Manual(m) => assert!((m.value - want).abs() <= 1e-12),
            ym_core::settings::AutoOr::Auto => panic!("auto"),
        }
        assert!((e.hue_curves.hue_vs_hue[0].value - k * -40.0).abs() <= 1e-12);
        assert!((e.hue_curves.hue_vs_sat[1].value - (1.0 + k * (0.5 - 1.0))).abs() <= 1e-12);
        assert!((e.elements.sky.chroma - 1.6f64.powf(k)).abs() <= 1e-12);
        assert!((e.elements.vegetation.chroma - 0.5f64.powf(k)).abs() <= 1e-12);
        assert!((e.elements.sky.hue_deg - k * 20.0).abs() <= 1e-12);
        assert!((e.elements.sky.exposure_ev - k * -0.30).abs() <= 1e-12);
        assert!((e.elements.subject.exposure_ev - k * 0.8).abs() <= 1e-12);
        // The trims and the grade fields the fit does not write are the person's: Strength does not scale them.
        assert_eq!(e.trim, n.trim);
        assert_eq!(e.grade.contrast, n.grade.contrast);
        assert_eq!(e.grade.color_warp, n.grade.color_warp);
    }
    // Half of a ratio of 1.6 is the geometric half, and the chroma power is exact at the ends.
    n.strength = 50.0;
    assert!((n.effective().elements.sky.chroma - 1.6f64.sqrt()).abs() <= 1e-12);
    // Strength 100 is the stored values, bit for bit; Strength 0 is the identity of every fitted field, bit for bit.
    n.strength = 100.0;
    let e = n.effective();
    assert_eq!(e.hue_curves, n.hue_curves);
    assert_eq!(e.elements, n.elements);
    assert_eq!(e.grade.curve, n.grade.curve);
    n.strength = 0.0;
    let e = n.effective();
    assert!(e.hue_and_elements_are_identity());
    assert!(
        e.grade
            .curve
            .luma
            .iter()
            .all(|p| p[0].to_bits() == p[1].to_bits())
    );
    assert!(
        e.grade
            .curve
            .red
            .iter()
            .all(|p| p[0].to_bits() == p[1].to_bits())
    );
    assert!(n.effective().grade.curve.green.iter().all(|p| p[0] == p[1]));
    match e.grade.saturation {
        ym_core::settings::AutoOr::Manual(m) => assert_eq!(m.value, 0.0),
        ym_core::settings::AutoOr::Auto => panic!("auto"),
    }
    // Only the fitted fields: a node with its other fields at their defaults is inert at Strength 0.
    let mut only_fit = ReferenceGradeNode::default();
    only_fit.hue_curves = n.hue_curves.clone();
    only_fit.elements = n.elements;
    only_fit.grade.curve = n.grade.curve.clone();
    only_fit.grade.saturation = ym_core::settings::PercentSetting::manual_value(-12.0);
    only_fit.strength = 0.0;
    assert!(only_fit.is_inert());
    only_fit.strength = 100.0;
    assert!(!only_fit.is_inert());
    // A node with Auto in its grade reads it as a manual 0.
    let mut auto = ReferenceGradeNode::default();
    auto.grade = ym_core::settings::NodeGrade::default();
    assert!(auto.effective().grade == ym_core::settings::NodeGrade::neutral());
}

#[test]
fn patches_change_fields_report_them_and_partials_keep_the_rest() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "hue_curves": { "hue_vs_hue": [ { "hue_deg": 30, "value": 12 } ] },
                "elements": { "sky": { "hue_deg": 10 }, "subject": { "exposure_ev": 0.25 } },
                "skin_protection": false, "strength": 60 } } })),
            None,
        )
        .unwrap();
    for field in [
        SettingsField::HueCurves,
        SettingsField::Elements,
        SettingsField::SkinProtection,
        SettingsField::Strength,
    ] {
        assert!(report.changed.contains(&FieldRef {
            node: rid(2),
            field
        }));
    }
    assert_eq!(
        report.first_group(),
        Some(ym_core::settings::StageGroup::Grade)
    );
    let n = node_of(&s).clone();
    assert_eq!(n.hue_curves.hue_vs_hue.len(), 1);
    assert!(n.hue_curves.hue_vs_sat.is_empty());
    assert_eq!(n.elements.sky.hue_deg, 10.0);
    assert_eq!(n.elements.sky.chroma, 1.0, "a part left out stays");
    assert_eq!(n.elements.subject.exposure_ev, 0.25);
    assert!(!n.skin_protection);
    assert_eq!(n.strength, 60.0);
    // A partial hue_curves: the curve given replaces that curve, the one left out stays.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "hue_curves": {
            "hue_vs_sat": [ { "hue_deg": 200, "value": 1.5 } ] } } } })),
        None,
    )
    .unwrap();
    let c = &node_of(&s).hue_curves;
    assert_eq!((c.hue_vs_hue.len(), c.hue_vs_sat.len()), (1, 1));
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "hue_curves": { "hue_vs_hue": [] } } } })),
        None,
    )
    .unwrap();
    assert!(node_of(&s).hue_curves.hue_vs_hue.is_empty());
    assert_eq!(node_of(&s).hue_curves.hue_vs_sat.len(), 1);
    // A partial elements: a member given takes the parts it names, the others stay.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "elements": { "sky": { "chroma": 1.2 } } } } })),
        None,
    )
    .unwrap();
    let e = node_of(&s).elements;
    assert_eq!((e.sky.hue_deg, e.sky.chroma), (10.0, 1.2));
    assert_eq!(e.subject.exposure_ev, 0.25);
    // Nothing changes when a patch sets what is there.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "elements": { "sky": { "chroma": 1.2 } } } } })),
            None,
        )
        .unwrap();
    assert!(report.changed.is_empty());
    // Name and place re-run nothing.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "name": "Look", "pos": [3, 1] } } })),
            None,
        )
        .unwrap();
    assert_eq!(report.first_group(), None);
    // The trims and grade go through the same patch of a corrector's.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "trim": { "exposure_ev": 0.5 },
                "grade": { "contrast": 12, "curve": { "luma": [[0, 0], [0.5, 0.55], [1, 1]] } } } } })),
            None,
        )
        .unwrap();
    for field in [
        SettingsField::Trim,
        SettingsField::Contrast,
        SettingsField::Curve,
    ] {
        assert!(report.changed.contains(&FieldRef {
            node: rid(2),
            field
        }));
    }
    assert_eq!(node_of(&s).trim.exposure_ev, 0.5);
    // A field the kind has not, and the unknown member of a patch.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "compress": { "hue": 1 } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Reference Grade node has no field called “compress”."
    );
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "1": { "hue_curves": { "hue_vs_hue": [] } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "A corrector has no field called “hue_curves”."
    );
    let e = serde_json::from_value::<SettingsPatch>(
        json!({ "nodes": { "2": { "elements": { "ceiling": {} } } } }),
    )
    .unwrap_err();
    assert!(e.to_string().contains("no element called “ceiling”"), "{e}");
    assert!(
        serde_json::from_value::<SettingsPatch>(
            json!({ "nodes": { "2": { "elements": { "subject": { "chroma": 1 } } } } })
        )
        .is_err()
    );
}

#[test]
fn a_patch_reports_what_it_clamped_in_the_fields_it_touched() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "hue_curves": { "hue_vs_sat": [ { "hue_deg": 725, "value": 9 } ] },
                "elements": { "skin": { "hue_deg": 40 } }, "strength": 400 } } })),
            None,
        )
        .unwrap();
    assert!(reported(&report.clamped, SettingsField::HueCurves));
    assert!(reported(&report.clamped, SettingsField::Elements));
    assert!(reported(&report.clamped, SettingsField::Strength));
    let n = node_of(&s);
    assert_eq!(n.hue_curves.hue_vs_sat[0].value, 8.0);
    assert_eq!(n.hue_curves.hue_vs_sat[0].hue_deg, 5.0);
    assert_eq!(n.elements.skin.hue_deg, 15.0);
    assert_eq!(n.strength, 100.0);
}

#[test]
fn a_node_patch_adds_the_node_removes_it_and_refuses_the_wrong_kind() {
    let mut s = DevelopSettings::default();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "reference_grade" } },
                "links": serde_json::to_value(line(&[1, 2])).unwrap() })),
            None,
        )
        .unwrap();
    assert_eq!(report.added, vec![rid(2)]);
    assert_eq!(
        node_of(&s),
        &ReferenceGradeNode {
            pos: [1.0, 0.0],
            ..ReferenceGradeNode::default()
        }
    );
    for field in [
        SettingsField::HueCurves,
        SettingsField::Elements,
        SettingsField::SkinProtection,
    ] {
        assert!(report.changed.contains(&FieldRef {
            node: rid(2),
            field
        }));
    }
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "corrector" } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Node 2 is a Reference Grade node, not a corrector."
    );
    let report = s
        .apply_patch(
            &SettingsPatch {
                remove: vec![rid(2)],
                links: Some(
                    line(&[1])
                        .into_iter()
                        .map(|v| serde_json::from_value(v).unwrap())
                        .collect(),
                ),
                ..SettingsPatch::default()
            },
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![rid(2)]);
    assert!(s.nodes.reference_grade(2).is_none());
    assert_eq!(
        NodeKind::ReferenceGrade.with_article(),
        "a Reference Grade node"
    );
}

/// The fit is one patch of its fields (`docs/34` section 3.5) and one undo step; it leaves the person's fields alone.
#[test]
fn the_fit_is_one_patch_and_one_undo_step_and_leaves_the_other_fields_alone() {
    let mut a = base();
    a.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "name": "Mine", "trim": { "exposure_ev": 0.2 }, "grade": { "contrast": 10 },
            "strength": 70, "skin_protection": false,
            "masks": { "items": [ { "kind": "radial" } ] } } } })),
        None,
    )
    .unwrap();
    let fit = patch_of(json!({ "nodes": { "2": {
        "grade": { "saturation": -15,
                   "curve": { "luma": [[0, 0], [0.2784, 0.27], [1, 1]], "red": [[0, 0], [1, 1]],
                              "green": [[0, 0], [1, 1]], "blue": [[0, 0], [0.5, 0.48], [1, 1]] } },
        "hue_curves": { "hue_vs_hue": [ { "hue_deg": 0, "value": 5 }, { "hue_deg": 180, "value": -5 } ],
                        "hue_vs_sat": [ { "hue_deg": 0, "value": 1.1 }, { "hue_deg": 180, "value": 0.9 } ] },
        "elements": { "sky": { "exposure_ev": -0.2, "hue_deg": 9, "chroma": 1.1 },
                      "vegetation": { "exposure_ev": 0, "hue_deg": -4, "chroma": 0.9 },
                      "water": { "exposure_ev": 0, "hue_deg": 0, "chroma": 1 },
                      "ground": { "exposure_ev": 0, "hue_deg": 0, "chroma": 1 },
                      "built": { "exposure_ev": 0, "hue_deg": 0, "chroma": 1 },
                      "skin": { "exposure_ev": 0, "hue_deg": 2, "chroma": 1 },
                      "subject": { "exposure_ev": 0.1 } } } } }));
    let mut b = a.clone();
    let report = b.apply_patch(&fit, None).unwrap();
    for field in [
        SettingsField::Saturation,
        SettingsField::Curve,
        SettingsField::HueCurves,
        SettingsField::Elements,
    ] {
        assert!(report.changed.contains(&FieldRef {
            node: rid(2),
            field
        }));
    }
    let (na, nb) = (node_of(&a), node_of(&b));
    assert_eq!(nb.name, "Mine");
    assert_eq!(nb.trim, na.trim);
    assert_eq!(nb.grade.contrast, na.grade.contrast);
    assert_eq!((nb.strength, nb.skin_protection), (70.0, false));
    assert_eq!(nb.masks, na.masks);
    // One undo step: the patch between the two documents takes b back to a, and forward again.
    let undo = SettingsPatch::between(&b, &a);
    assert_eq!(undo.nodes.len(), 1);
    let mut back = b.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, a);
    let redo = SettingsPatch::between(&a, &b);
    let mut fwd = a.clone();
    fwd.apply_patch(&redo, None).unwrap();
    assert_eq!(fwd, b);
    // A node that was added or removed is carried whole, with its kind.
    let empty = DevelopSettings::default();
    let mut c = empty.clone();
    c.apply_patch(&SettingsPatch::between(&empty, &b), None)
        .unwrap();
    assert_eq!(c.nodes.reference_grade(2), b.nodes.reference_grade(2));
    let mut d = b.clone();
    d.apply_patch(&SettingsPatch::between(&b, &empty), None)
        .unwrap();
    assert!(d.nodes.reference_grade(2).is_none());
}

#[test]
fn reset_puts_the_defaults_back_and_keeps_name_bypass_place_masks_and_links() {
    let mut b = base();
    b.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "name": "Sky", "bypass": true, "pos": [4, 1],
            "trim": { "tint": 9 }, "grade": { "brightness": 7, "saturation": -20 },
            "hue_curves": { "hue_vs_hue": [ { "hue_deg": 10, "value": 5 } ] },
            "elements": { "sky": { "hue_deg": 12 }, "subject": { "exposure_ev": 0.3 } },
            "strength": 40, "skin_protection": false,
            "masks": { "items": [ { "kind": "radial" } ] } } } })),
        None,
    )
    .unwrap();
    let reset = SettingsPatch::reset_node_in(&b, rid(2), &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&reset, None).unwrap();
    let n = node_of(&r);
    let d = ReferenceGradeNode::default();
    assert_eq!(n.trim, d.trim);
    assert_eq!(n.grade, d.grade);
    assert_eq!(n.hue_curves, d.hue_curves);
    assert_eq!(n.elements, d.elements);
    assert_eq!((n.strength, n.skin_protection), (100.0, true));
    assert!(n.bypass);
    assert_eq!(n.name, "Sky");
    assert_eq!(n.pos, [4.0, 1.0]);
    assert_eq!(n.masks.items.len(), 1);
    assert_eq!(r.nodes.links, b.nodes.links);
    // Reset of one field (a card's Reset).
    let one = SettingsPatch::reset_in(&b, rid(2), SettingsField::Elements, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert!(node_of(&r).elements.is_default());
    assert_eq!(node_of(&r).hue_curves.hue_vs_hue.len(), 1);
    assert_eq!(node_of(&r).strength, 40.0);
}

fn resolved_of(settings: &DevelopSettings) -> ResolvedSettings {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(
        256,
        192,
        light("sony-ilce-7rm4", 5200.0),
        Layout::Patchwork,
        0.5,
    );
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", "ILCE-7RM4"));
    ResolvedSettings::resolve(PhotoId(3), settings, &est, &profile, None).unwrap()
}

#[test]
fn copy_carries_the_node_paste_replaces_the_targets_graph_and_resolved_values_have_its_grade() {
    let mut src = base();
    src.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "name": "Match",
            "grade": { "saturation": -14, "contrast": 6 },
            "hue_curves": { "hue_vs_hue": [ { "hue_deg": 30, "value": 7 } ] },
            "elements": { "sky": { "hue_deg": 8 } }, "strength": 80,
            "masks": { "items": [ { "kind": "object", "points": [[0.5, 0.5, true]] } ] } } } })),
        None,
    )
    .unwrap();
    let resolved = resolved_of(&src);
    // The node has no Auto: the resolved settings hold it as it is, and values.nodes has an entry with its trim and grade.
    assert_eq!(
        resolved.settings.nodes.reference_grade(2),
        src.nodes.reference_grade(2)
    );
    let entry = resolved
        .values
        .nodes
        .iter()
        .find(|n| n.id == 2)
        .expect("a resolved entry");
    assert_eq!(entry.grade.saturation, -14.0);
    assert_eq!(entry.grade.contrast, 6.0);
    let copy = SettingsPatch::copied_from(&resolved);
    let node = copy.nodes.get(&rid(2)).expect("the node is in the copy");
    assert_eq!(node.kind, Some(NodeKind::ReferenceGrade));
    let mut target = DevelopSettings::default();
    target
        .apply_patch(
            &patch_of(json!({ "nodes": { "7": { "kind": "reference_grade" } },
                "links": serde_json::to_value(line(&[1, 7])).unwrap() })),
            None,
        )
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(paste.remove.contains(&rid(7)), "the target's own node goes");
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.reference_grade(7).is_none());
    let n = target.nodes.reference_grade(2).expect("pasted");
    let s = node_of(&src);
    assert_eq!(n.hue_curves, s.hue_curves);
    assert_eq!(n.elements, s.elements);
    assert_eq!(n.grade, s.grade);
    assert_eq!(n.strength, 80.0);
    match &n.masks.items[0].shape {
        ym_core::settings::MaskShape::Object(o) => assert!(o.points.is_empty() && o.box_.is_none()),
        other => panic!("{other:?}"),
    }
    assert_eq!(target.nodes.links, src.nodes.links);
}

#[test]
fn the_xmp_text_holds_the_node_with_its_fitted_numbers_and_reads_back() {
    let (s, _) = read(doc(vec![example()], line(&[1, 2]))).unwrap();
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"kind\":\"reference_grade\""));
    assert!(text.contains("\"hue_curves\""));
    // The reference is never in the file: no name, path or statistic.
    for word in [
        "reference\"",
        "\"path\"",
        "\"file\"",
        "statistics",
        "fit_status",
    ] {
        assert!(!text.contains(word), "{word}");
    }
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, s);
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

#[test]
fn the_schema_validates_what_the_core_writes_and_refuses_what_it_refuses() {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("crates/ym-core/schema/settings.schema.json"))
            .unwrap(),
    )
    .unwrap();
    let v = jsonschema::validator_for(&schema).expect("a schema");
    for s in [base(), read(doc(vec![example()], line(&[1, 2]))).unwrap().0] {
        let written = serde_json::to_value(&s).unwrap();
        let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
        assert!(problems.is_empty(), "{problems:?}");
    }
    let bad = |body: Value| {
        let mut node = json!({ "id": 2, "kind": "reference_grade" });
        node.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        json!({ "nodes": { "items": [node] } })
    };
    for body in [
        json!({ "reference": "x" }),
        json!({ "range": {} }),
        json!({ "strength": 101 }),
        json!({ "strength": -1 }),
        json!({ "skin_protection": 1 }),
        json!({ "hue_curves": { "x": [] } }),
        json!({ "hue_curves": { "hue_vs_hue": [ { "hue_deg": 1 } ] } }),
        json!({ "hue_curves": { "hue_vs_hue": [ { "hue_deg": 1, "value": 181 } ] } }),
        json!({ "hue_curves": { "hue_vs_sat": [ { "hue_deg": 1, "value": 8.1 } ] } }),
        json!({ "elements": { "ceiling": {} } }),
        json!({ "elements": { "skin": { "hue_deg": 16 } } }),
        json!({ "elements": { "sky": { "chroma": 1.7 } } }),
        json!({ "elements": { "sky": { "exposure_ev": 1.1 } } }),
        json!({ "elements": { "subject": { "chroma": 1 } } }),
    ] {
        assert!(!v.is_valid(&bad(body.clone())), "{body}");
    }
    for body in [
        json!({}),
        json!({ "strength": 0 }),
        json!({ "hue_curves": { "hue_vs_hue": [ { "hue_deg": 725, "value": -180 } ] } }),
        json!({ "elements": { "skin": { "hue_deg": 15 }, "subject": { "exposure_ev": -1 } } }),
    ] {
        assert!(v.is_valid(&bad(body.clone())), "{body}");
    }
    // The ranges are the ones the core clamps to, and the defaults are the core's.
    let range = |p: &Value| {
        (
            p["minimum"].as_f64().unwrap(),
            p["maximum"].as_f64().unwrap(),
        )
    };
    assert_eq!(
        range(&schema["$defs"]["hue_point_turn"]["properties"]["value"]),
        HUE_VS_HUE_RANGE_DEG
    );
    assert_eq!(
        range(&schema["$defs"]["hue_point_chroma"]["properties"]["value"]),
        HUE_VS_SAT_RANGE
    );
    for (name, cap) in ELEMENT_NAMES.iter().zip(ELEMENT_HUE_CAPS_DEG) {
        let p = &schema["$defs"][format!("element_{name}")]["properties"];
        assert_eq!(range(&p["hue_deg"]), (-cap, cap), "{name}");
        assert_eq!(range(&p["chroma"]), ELEMENT_CHROMA_RANGE, "{name}");
        assert_eq!(
            range(&p["exposure_ev"]),
            ELEMENT_EXPOSURE_RANGE_EV,
            "{name}"
        );
    }
    let n = &schema["$defs"]["reference_grade_node"]["properties"];
    assert_eq!(
        n["name"]["default"],
        json!(ReferenceGradeNode::default().name)
    );
    assert_eq!(n["kind"]["const"], json!("reference_grade"));
    assert_eq!(n["strength"]["default"], json!(100));
    assert_eq!(n["skin_protection"]["default"], json!(true));
}

#[test]
fn the_kind_is_in_all_and_its_names_and_fields_are_the_specs() {
    assert!(NodeKind::ALL.contains(&NodeKind::ReferenceGrade));
    assert_eq!(NodeKind::ReferenceGrade.name(), "reference_grade");
    assert_eq!(
        NodeKind::ReferenceGrade.sentence(),
        "A Reference Grade node"
    );
    assert_eq!(
        NodeKind::parse("reference_grade").unwrap(),
        NodeKind::ReferenceGrade
    );
    assert!(NodeKind::ReferenceGrade.is_numbered_extra());
    assert!(!NodeKind::ReferenceGrade.is_mixer_family());
    for f in SettingsField::ALL {
        let has = f.belongs_to(NodeKind::ReferenceGrade);
        let want = matches!(
            f,
            SettingsField::Name
                | SettingsField::Bypass
                | SettingsField::Pos
                | SettingsField::Masks
                | SettingsField::Trim
                | SettingsField::HueCurves
                | SettingsField::Elements
                | SettingsField::Strength
                | SettingsField::SkinProtection
        ) || f.is_grade();
        assert_eq!(has, want, "{f:?}");
    }
    for f in [
        SettingsField::HueCurves,
        SettingsField::Elements,
        SettingsField::SkinProtection,
    ] {
        assert_eq!(f.group(), Some(ym_core::settings::StageGroup::Grade));
        assert_eq!(
            f.name(),
            serde_json::from_str::<String>(&serde_json::to_string(&f).unwrap()).unwrap()
        );
    }
}

/// The pipeline version names this kind (an older build refuses the node with the unknown-kind sentence).
#[test]
fn an_older_build_refuses_the_kind_with_the_unknown_kind_sentence() {
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "a_kind_from_a_later_build" })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert!(matches!(
        e,
        SettingsError::Graph(GraphError::UnknownKind(_))
    ));
    assert_eq!(ym_core::version::PIPELINE_VERSION, "0.17.0");
}
