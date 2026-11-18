//! The Blur node's settings (`docs/33-blur.md` section 5, `docs/23` section 3): the format, the defaults and clamping, every refusal
//! and its text, patches (Pick centre as one patch among them), Reset, undo, Copy and Paste, the XMP text, the JSON schema, the
//! plan's keys and the render as the identity until BL-04.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::session::PhotoId;
use ym_core::settings::{
    BLUR_AMOUNT_RANGE, BLUR_BLADES_MAX, BLUR_BLADES_MIN, BLUR_CENTER_RANGE, BlurNode, BlurType,
    DevelopSettings, FieldRef, GraphError, LensPatch, NodeId, NodeKind, NodePatch, Port,
    ResolvedSettings, SettingsError, SettingsField, SettingsPatch, ports_in, ports_out,
};
use ym_looks::library::LookId;

fn uid(n: u32) -> NodeId {
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
    let mut node = json!({ "id": 2, "kind": "blur" });
    if let (Some(n), Some(b)) = (node.as_object_mut(), body.as_object()) {
        n.extend(b.clone());
    }
    read(doc(vec![node], line(&[1, 2])))
}

fn node_of(s: &DevelopSettings) -> &BlurNode {
    s.nodes.blur(2).expect("blur node 2")
}

fn base() -> DevelopSettings {
    with_node(json!({})).expect("a default node").0
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

/// The node of `docs/33` section 5.1, the example, with a Radial row (it needs no Vision).
fn example() -> Value {
    json!({ "id": 2, "kind": "blur", "name": "Blur", "bypass": false, "pos": [2, 0],
        "type": "lens",
        "soft": { "radius": 20 },
        "lens": { "radius": 50, "blades": 6, "rotation_deg": 15, "highlights": 60 },
        "motion": { "length": 20, "angle_deg": 0 },
        "zoom": { "amount": 20, "center": [0.5, 0.5] },
        "respect_mask_edges": true,
        "masks": { "invert": false, "items": [
            { "kind": "radial", "mode": "add", "invert": true, "enabled": true, "opacity": 1, "feather": 0.08 } ] } })
}

fn reported(clamped: &[FieldRef], field: SettingsField) -> bool {
    clamped.contains(&FieldRef {
        node: uid(2),
        field,
    })
}

#[test]
fn a_new_node_has_the_defaults_of_the_spec() {
    let (s, clamped) = with_node(json!({})).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    assert_eq!(n.name, "Blur");
    assert!(!n.bypass);
    assert_eq!(n.blur_type, BlurType::Soft);
    assert_eq!(n.soft.radius, 20.0);
    assert_eq!(
        (
            n.lens.radius,
            n.lens.blades,
            n.lens.rotation_deg,
            n.lens.highlights
        ),
        (20.0, 0, 0.0, 0.0)
    );
    assert_eq!((n.motion.length, n.motion.angle_deg), (20.0, 0.0));
    assert_eq!((n.zoom.amount, n.zoom.center), (20.0, [0.5, 0.5]));
    assert!(n.respect_mask_edges, "Respect mask edges is on by default");
    assert!(n.masks.items.is_empty() && !n.masks.invert);
    assert!(!n.is_inert(), "a new node blurs a little at once");
    // The written form: the fields of section 5.1 and no others, every group present.
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
            "id",
            "kind",
            "lens",
            "masks",
            "motion",
            "name",
            "pos",
            "respect_mask_edges",
            "soft",
            "type",
            "zoom"
        ]
    );
    assert_eq!(item["kind"], json!("blur"));
    assert_eq!(item["type"], json!("soft"));
    assert_eq!(item["zoom"]["center"], json!([0.5, 0.5]));
}

#[test]
fn the_inert_node_is_judged_by_its_own_types_amount_only() {
    let (s, _) = with_node(json!({ "soft": { "radius": 0 } })).unwrap();
    assert!(node_of(&s).is_inert(), "Soft at Radius 0");
    // Another type chosen: Soft's 0 does not make it inert, its own amount does.
    let (s, _) = with_node(json!({ "type": "lens", "soft": { "radius": 0 } })).unwrap();
    assert!(!node_of(&s).is_inert());
    let (s, _) = with_node(json!({ "type": "motion", "motion": { "length": 0 } })).unwrap();
    assert!(node_of(&s).is_inert());
    let (s, _) = with_node(json!({ "type": "zoom", "zoom": { "amount": 0 } })).unwrap();
    assert!(node_of(&s).is_inert());
    let (s, _) = with_node(json!({ "bypass": true })).unwrap();
    assert!(node_of(&s).is_inert());
}

#[test]
fn the_example_of_the_spec_reads_and_writes_back_the_same() {
    let (s, clamped) = read(doc(vec![example()], line(&[1, 2]))).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    assert_eq!(n.blur_type, BlurType::Lens);
    assert_eq!(
        (
            n.lens.radius,
            n.lens.blades,
            n.lens.rotation_deg,
            n.lens.highlights
        ),
        (50.0, 6, 15.0, 60.0)
    );
    assert_eq!(n.masks.items.len(), 1);
    // Written and read again, the same; every type's group is in the file.
    let written = serde_json::to_value(&s).unwrap();
    let item = written["nodes"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"] == json!(2))
        .unwrap();
    for group in ["soft", "lens", "motion", "zoom"] {
        assert!(item[group].is_object(), "{group}");
    }
    let (again, clamped) = read(written).unwrap();
    assert!(clamped.is_empty());
    assert_eq!(again, s);
}

#[test]
fn a_partial_group_is_read_over_the_defaults_and_every_type_keeps_its_values() {
    let (s, _) =
        with_node(json!({ "lens": { "radius": 60 }, "zoom": { "center": [0.25, 0.75] } })).unwrap();
    let n = node_of(&s);
    assert_eq!(
        (n.lens.radius, n.lens.blades, n.lens.highlights),
        (60.0, 0, 0.0)
    );
    assert_eq!((n.zoom.amount, n.zoom.center), (20.0, [0.25, 0.75]));
    assert_eq!(n.soft.radius, 20.0);
    // Switching the type with a patch keeps every group's values.
    let mut s = s;
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "type": "zoom" } } })),
        None,
    )
    .unwrap();
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "type": "lens" } } })),
        None,
    )
    .unwrap();
    assert_eq!(node_of(&s).lens.radius, 60.0);
    assert_eq!(node_of(&s).zoom.center, [0.25, 0.75]);
}

#[test]
fn the_ports_are_a_correctors_and_ids_are_shared() {
    assert_eq!(ports_in(NodeKind::Blur), ports_in(NodeKind::Corrector));
    assert_eq!(
        ports_out(NodeKind::Blur),
        [Port::Picture, Port::Key].as_slice()
    );
    let items = vec![
        json!({ "id": 2, "kind": "blur" }),
        json!({ "id": 3, "kind": "corrector" }),
    ];
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "key", json!(3), "key"));
    assert!(read(doc(items.clone(), links)).is_ok());
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "picture", json!(3), "key"));
    assert_eq!(
        read(doc(items, links)).unwrap_err().to_string(),
        GraphError::LinkPortTypes.to_string()
    );
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "blur" })],
        vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "red"),
            link(json!(2), "picture", json!("output"), "picture"),
        ],
    ))
    .unwrap_err();
    assert!(e.to_string().contains("no port"), "{e}");
    let e = read(doc(
        vec![
            json!({ "id": 2, "kind": "blur" }),
            json!({ "id": 2, "kind": "color_uniform" }),
        ],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "Two nodes have the id 2.");
    assert!(read(doc(vec![json!({ "id": 1, "kind": "blur" })], line(&[1]))).is_err());
}

#[test]
fn the_limit_of_64_nodes_counts_this_kind() {
    let items: Vec<Value> = (2..=64)
        .map(|i| json!({ "id": i, "kind": "blur" }))
        .collect();
    assert!(
        read(doc(items, line(&[1]))).is_ok(),
        "Main and 63 nodes are 64"
    );
    let items: Vec<Value> = (2..=65)
        .map(|i| json!({ "id": i, "kind": "blur" }))
        .collect();
    let e = read(doc(items, line(&[1]))).unwrap_err();
    assert_eq!(e.to_string(), "A photo can have up to 64 nodes.");
    let mut items: Vec<Value> = (2..=33)
        .map(|i| json!({ "id": i, "kind": "blur" }))
        .collect();
    items.extend((34..=65).map(|i| json!({ "id": i, "kind": "film" })));
    assert!(read(doc(items, line(&[1]))).is_err());
}

#[test]
fn an_unknown_field_is_refused_with_its_text() {
    for (field, body) in [
        ("range", json!({ "range": { "width": 0.05 } })),
        ("trim", json!({ "trim": {} })),
        ("strength", json!({ "strength": 3 })),
        ("target", json!({ "target": null })),
    ] {
        let e = with_node(body).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("A Blur node has no field called “{field}”.")
        );
    }
    // Another kind has no `type`, `soft` or `respect_mask_edges`.
    for (kind, name, field, value) in [
        ("corrector", "A corrector", "type", json!("soft")),
        ("film", "A film node", "soft", json!({})),
        (
            "color_uniform",
            "A Colour Uniform node",
            "respect_mask_edges",
            json!(true),
        ),
        ("corrector", "A corrector", "zoom", json!({})),
    ] {
        let mut node = json!({ "id": 2, "kind": kind });
        node[field] = value;
        let e = read(doc(vec![node], line(&[1, 2]))).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("{name} has no field called “{field}”.")
        );
    }
    // A typo inside a group is serde's refusal.
    for body in [
        json!({ "soft": { "radious": 3 } }),
        json!({ "lens": { "kind": "vintage" } }),
        json!({ "motion": { "angle": 3 } }),
        json!({ "zoom": { "centre": [0.5, 0.5] } }),
    ] {
        assert!(
            matches!(
                with_node(body.clone()).unwrap_err(),
                SettingsError::Parse(_)
            ),
            "{body}"
        );
    }
}

#[test]
fn an_unknown_type_is_a_parse_error_with_its_text() {
    for word in ["spin", "Soft", "gaussian", ""] {
        let e = with_node(json!({ "type": word })).unwrap_err();
        assert!(matches!(e, SettingsError::Parse(_)), "{e}");
        assert!(
            e.to_string()
                .contains(&format!("A Blur node has no type called “{word}”.")),
            "{e}"
        );
    }
    for t in [
        json!(3),
        json!(null),
        json!(["soft"]),
        json!({ "name": "soft" }),
    ] {
        let e = with_node(json!({ "type": t.clone() })).unwrap_err();
        assert!(matches!(e, SettingsError::Parse(_)), "{t}: {e}");
    }
    for t in BlurType::ALL {
        let (s, _) = with_node(json!({ "type": t.name() })).unwrap();
        assert_eq!(node_of(&s).blur_type, t);
    }
}

#[test]
fn a_value_of_the_wrong_shape_is_a_parse_error() {
    for body in [
        json!({ "soft": 5 }),
        json!({ "soft": [20] }),
        json!({ "soft": { "radius": "big" } }),
        json!({ "zoom": { "center": [0.5] } }),
        json!({ "zoom": { "center": [0.5, 0.5, 0.5] } }),
        json!({ "zoom": { "center": "middle" } }),
        json!({ "zoom": { "center": { "x": 0.5, "y": 0.5 } } }),
        json!({ "zoom": { "center": [0.5, "x"] } }),
        json!({ "lens": { "blades": 6.5 } }),
        json!({ "lens": { "blades": -1 } }),
        json!({ "lens": { "blades": "six" } }),
        json!({ "respect_mask_edges": "yes" }),
        json!({ "motion": null }),
    ] {
        assert!(
            matches!(
                with_node(body.clone()).unwrap_err(),
                SettingsError::Parse(_)
            ),
            "{body}"
        );
    }
}

#[test]
fn numbers_are_clamped_and_reported_and_the_angles_are_read_modulo_silently() {
    let (s, clamped) = with_node(json!({
        "soft": { "radius": 140 },
        "lens": { "radius": -5, "blades": 40, "highlights": 101 },
        "motion": { "length": 200 },
        "zoom": { "amount": -1, "center": [1.5, -0.25] }
    }))
    .unwrap();
    let n = node_of(&s);
    assert_eq!(n.soft.radius, 100.0);
    assert_eq!(
        (n.lens.radius, n.lens.blades, n.lens.highlights),
        (0.0, BLUR_BLADES_MAX, 100.0)
    );
    assert_eq!(n.motion.length, 100.0);
    assert_eq!((n.zoom.amount, n.zoom.center), (0.0, [1.0, 0.0]));
    for f in [
        SettingsField::Soft,
        SettingsField::BlurLens,
        SettingsField::Motion,
        SettingsField::Zoom,
    ] {
        assert!(reported(&clamped, f), "{f:?}");
    }
    // The blades 1 and 2 are held at 3 and reported; 0 and 3 to 12 are what they are.
    for (given, want, report) in [
        (0, 0, false),
        (1, 3, true),
        (2, 3, true),
        (3, 3, false),
        (9, 9, false),
        (12, 12, false),
        (13, 12, true),
    ] {
        let (s, clamped) = with_node(json!({ "lens": { "blades": given } })).unwrap();
        assert_eq!(node_of(&s).lens.blades, want, "{given}");
        assert_eq!(
            reported(&clamped, SettingsField::BlurLens),
            report,
            "{given}"
        );
    }
    assert_eq!(BLUR_BLADES_MIN, 3);
    // The rotation is any finite number read modulo 360, the angle modulo 180: neither is reported.
    for (rot, want) in [
        (15.0, 15.0),
        (375.0, 15.0),
        (-15.0, 345.0),
        (360.0, 0.0),
        (720.5, 0.5),
    ] {
        let (s, clamped) = with_node(json!({ "lens": { "rotation_deg": rot } })).unwrap();
        assert_eq!(node_of(&s).lens.rotation_deg, want, "{rot}");
        assert!(clamped.is_empty(), "{rot}");
    }
    for (angle, want) in [
        (20.0, 20.0),
        (180.0, 0.0),
        (200.0, 20.0),
        (-20.0, 160.0),
        (359.0, 179.0),
    ] {
        let (s, clamped) = with_node(json!({ "motion": { "angle_deg": angle } })).unwrap();
        assert_eq!(node_of(&s).motion.angle_deg, want, "{angle}");
        assert!(clamped.is_empty(), "{angle}");
    }
    // A value in range is not reported, and the ends are in range.
    let (_, clamped) = with_node(json!({
        "soft": { "radius": 100 }, "lens": { "radius": 0, "highlights": 100 },
        "zoom": { "amount": 100, "center": [0, 1] } }))
    .unwrap();
    assert!(clamped.is_empty());
}

#[test]
fn a_number_that_is_not_finite_is_refused_and_nothing_is_applied() {
    let mut s = base();
    let before = s.clone();
    let bad = |f: &dyn Fn(&mut NodePatch)| {
        let mut p = NodePatch::default();
        f(&mut p);
        p
    };
    for patch in [
        bad(&|p| {
            p.soft = Some(ym_core::settings::BlurSoftPatch {
                radius: Some(f64::NAN),
            });
        }),
        bad(&|p| {
            p.lens = Some(LensPatch {
                rotation_deg: Some(f64::INFINITY),
                ..LensPatch::default()
            });
        }),
        bad(&|p| {
            p.lens = Some(LensPatch {
                highlights: Some(f64::NEG_INFINITY),
                ..LensPatch::default()
            });
        }),
        bad(&|p| {
            p.motion = Some(ym_core::settings::BlurMotionPatch {
                angle_deg: Some(f64::NAN),
                ..Default::default()
            });
        }),
        bad(&|p| {
            p.zoom = Some(ym_core::settings::BlurZoomPatch {
                center: Some([0.5, f64::NAN]),
                ..Default::default()
            });
        }),
    ] {
        let e = s
            .apply_patch(&SettingsPatch::node(uid(2), patch), None)
            .unwrap_err();
        assert!(matches!(e, SettingsError::NotFinite(_)), "{e}");
        assert_eq!(s, before, "atomic");
    }
    // The path names the field.
    let e = s
        .apply_patch(
            &SettingsPatch::node(
                uid(2),
                bad(&|p| {
                    p.zoom = Some(ym_core::settings::BlurZoomPatch {
                        center: Some([0.5, f64::INFINITY]),
                        ..Default::default()
                    });
                }),
            ),
            None,
        )
        .unwrap_err();
    assert!(e.to_string().contains("zoom.center[1]"), "{e}");
}

#[test]
fn the_name_is_trimmed_cut_and_an_empty_one_reads_as_the_default() {
    let (s, _) = with_node(json!({ "name": "   " })).unwrap();
    assert_eq!(node_of(&s).name, "Blur");
    let (s, _) = with_node(json!({ "name": format!("  {}  ", "x".repeat(40)) })).unwrap();
    assert_eq!(node_of(&s).name.chars().count(), 32);
    let (s, _) = with_node(json!({ "name": "Background" })).unwrap();
    assert_eq!(node_of(&s).name, "Background");
}

#[test]
fn patches_change_fields_report_them_and_a_partial_group_keeps_the_rest() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "type": "lens",
                "soft": { "radius": 33 },
                "lens": { "blades": 6, "highlights": 40 },
                "motion": { "angle_deg": 45 },
                "zoom": { "amount": 80 },
                "respect_mask_edges": false } } })),
            None,
        )
        .unwrap();
    let n = node_of(&s);
    assert_eq!(n.blur_type, BlurType::Lens);
    assert_eq!(n.soft.radius, 33.0);
    assert_eq!(
        (
            n.lens.radius,
            n.lens.blades,
            n.lens.rotation_deg,
            n.lens.highlights
        ),
        (20.0, 6, 0.0, 40.0)
    );
    assert_eq!((n.motion.length, n.motion.angle_deg), (20.0, 45.0));
    assert_eq!((n.zoom.amount, n.zoom.center), (80.0, [0.5, 0.5]));
    assert!(!n.respect_mask_edges);
    for field in [
        SettingsField::BlurType,
        SettingsField::Soft,
        SettingsField::BlurLens,
        SettingsField::Motion,
        SettingsField::Zoom,
        SettingsField::RespectMaskEdges,
    ] {
        assert!(
            report.changed.contains(&FieldRef {
                node: uid(2),
                field
            }),
            "{field:?}"
        );
        // The nodes' group: a change runs the colour pass again, and `name` and `pos` run nothing.
        assert!(field.group().is_some(), "{field:?}");
    }
    assert!(SettingsField::Name.group().is_none());
    // A patch that changes nothing reports nothing.
    let again = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "soft": { "radius": 33 }, "type": "lens" } } })),
            None,
        )
        .unwrap();
    assert!(again.changed.is_empty(), "{:?}", again.changed);
    // The two `lens` fields are told apart: the Input's has its own parts.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "lens": { "kind": "vintage" } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Blur node has no field called “lens.kind”."
    );
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "input": { "lens": { "radius": 30 } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(e.to_string(), "An input has no field called “lens.radius”.");
    // ... and the Input's lens still works.
    let r = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "input": { "lens": { "flares": 40 } } } })),
            None,
        )
        .unwrap();
    assert!(r.changed.contains(&FieldRef {
        node: NodeId::Input,
        field: SettingsField::Lens
    }));
    assert!(!r.changed.iter().any(|f| f.field == SettingsField::BlurLens));
    // Another field of another kind is refused with the kind's sentence.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "trim": { "exposure_ev": 1 } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(e.to_string(), "A Blur node has no field called “trim”.");
}

#[test]
fn a_patch_reports_what_it_clamped_in_the_fields_it_touched() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "soft": { "radius": 500 },
                "lens": { "blades": 2 },
                "zoom": { "center": [2, 0.5] } } } })),
            None,
        )
        .unwrap();
    assert!(reported(&report.clamped, SettingsField::Soft));
    assert!(reported(&report.clamped, SettingsField::BlurLens));
    assert!(reported(&report.clamped, SettingsField::Zoom));
    assert!(!reported(&report.clamped, SettingsField::Motion));
    assert_eq!(node_of(&s).soft.radius, BLUR_AMOUNT_RANGE.1);
    assert_eq!(node_of(&s).lens.blades, 3);
    assert_eq!(node_of(&s).zoom.center, [BLUR_CENTER_RANGE.1, 0.5]);
}

#[test]
fn a_node_patch_adds_the_node_removes_it_and_refuses_the_wrong_kind() {
    let mut s = DevelopSettings::default();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "blur" } },
                "links": serde_json::to_value(line(&[1, 2])).unwrap() })),
            None,
        )
        .unwrap();
    assert_eq!(report.added, vec![uid(2)]);
    assert_eq!(
        node_of(&s),
        &BlurNode {
            pos: [1.0, 0.0],
            ..BlurNode::default()
        }
    );
    for field in [
        SettingsField::BlurType,
        SettingsField::Soft,
        SettingsField::RespectMaskEdges,
    ] {
        assert!(
            report.changed.contains(&FieldRef {
                node: uid(2),
                field
            }),
            "{field:?}"
        );
    }
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "corrector" } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(e.to_string(), "Node 2 is a Blur node, not a corrector.");
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "1": { "kind": "blur" } } })),
            None,
        )
        .unwrap_err();
    assert!(e.to_string().contains("not a Blur node"), "{e}");
    let report = s
        .apply_patch(
            &patch_of(json!({ "remove": [2], "links": serde_json::to_value(line(&[1])).unwrap() })),
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![uid(2)]);
    assert!(s.nodes.blur(2).is_none());
    // Adding with its fields in the one patch.
    s.apply_patch(
        &patch_of(
            json!({ "nodes": { "5": { "kind": "blur", "name": "Background", "type": "motion",
            "motion": { "length": 45, "angle_deg": 20 } } } }),
        ),
        None,
    )
    .unwrap();
    let n = s.nodes.blur(5).unwrap();
    assert_eq!(n.name, "Background");
    assert_eq!(n.blur_type, BlurType::Motion);
    assert_eq!((n.motion.length, n.motion.angle_deg), (45.0, 20.0));
    // The 64-node limit counts the node in a patch too.
    let mut full = DevelopSettings::default();
    let nodes: serde_json::Map<String, Value> = (2..=65)
        .map(|i| (i.to_string(), json!({ "kind": "blur" })))
        .collect();
    let e = full
        .apply_patch(&patch_of(json!({ "nodes": nodes })), None)
        .unwrap_err();
    assert_eq!(e.to_string(), "A photo can have up to 64 nodes.");
    assert_eq!(full, DevelopSettings::default(), "atomic");
}

#[test]
fn pick_centre_is_one_atomic_patch_and_one_undo_step() {
    let mut s = base();
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "type": "zoom" } } })),
        None,
    )
    .unwrap();
    let before = s.clone();
    let pick = patch_of(json!({ "nodes": { "2": { "zoom": { "center": [0.3, 0.6] } } } }));
    let report = s.apply_patch(&pick, None).unwrap();
    assert_eq!(report.changed.len(), 1, "{:?}", report.changed);
    assert!(report.changed.contains(&FieldRef {
        node: uid(2),
        field: SettingsField::Zoom
    }));
    assert_eq!(node_of(&s).zoom.center, [0.3, 0.6]);
    assert_eq!(node_of(&s).zoom.amount, 20.0, "the amount stays");
    // One undo step: the patch between the two documents undoes it in one application, and redoes it.
    let undo = SettingsPatch::between(&s, &before);
    let mut back = s.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, before);
    let redo = SettingsPatch::between(&before, &s);
    let mut fwd = before.clone();
    fwd.apply_patch(&redo, None).unwrap();
    assert_eq!(fwd, s);
    // A drag of the handle is a patch per frame; each is the centre alone, and the last one is what is kept.
    let mut drag = before.clone();
    for (x, y) in [(0.4, 0.5), (0.45, 0.55), (0.5, 0.6)] {
        drag.apply_patch(
            &patch_of(json!({ "nodes": { "2": { "zoom": { "center": [x, y] } } } })),
            None,
        )
        .unwrap();
    }
    assert_eq!(node_of(&drag).zoom.center, [0.5, 0.6]);
    let undo = SettingsPatch::between(&drag, &before);
    let mut back = drag.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, before);
    // A centre outside the frame is clamped and reported; one that is not finite changes nothing.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "zoom": { "center": [1.2, 0.5] } } } })),
            None,
        )
        .unwrap();
    assert!(reported(&report.clamped, SettingsField::Zoom));
    assert_eq!(node_of(&s).zoom.center, [1.0, 0.5]);
}

#[test]
fn undo_is_the_patch_between_two_settings_and_reset_puts_the_defaults_back() {
    let a = base();
    let mut b = a.clone();
    b.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "type": "zoom", "soft": { "radius": 70 },
            "lens": { "radius": 30, "blades": 7, "rotation_deg": 10, "highlights": 20 },
            "motion": { "length": 60, "angle_deg": 90 },
            "zoom": { "amount": 50, "center": [0.1, 0.9] },
            "respect_mask_edges": false,
            "bypass": true, "name": "Sky", "masks": { "items": [ { "kind": "radial" } ] } } } })),
        None,
    )
    .unwrap();
    let undo = SettingsPatch::between(&b, &a);
    let mut back = b.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, a);
    let redo = SettingsPatch::between(&a, &b);
    let mut fwd = a.clone();
    fwd.apply_patch(&redo, None).unwrap();
    assert_eq!(fwd, b);
    // A node that was added or removed is carried whole, with its kind.
    let empty = DevelopSettings::default();
    let add = SettingsPatch::between(&empty, &b);
    let mut c = empty.clone();
    c.apply_patch(&add, None).unwrap();
    assert_eq!(c.nodes.blur(2), b.nodes.blur(2));
    let drop = SettingsPatch::between(&b, &empty);
    let mut d = b.clone();
    d.apply_patch(&drop, None).unwrap();
    assert!(d.nodes.blur(2).is_none());
    // Reset of the head: the type and every group and the switch to the defaults; name, bypass, place, masks and links stay.
    let reset = SettingsPatch::reset_node_in(&b, uid(2), &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&reset, None).unwrap();
    let n = node_of(&r);
    let d = BlurNode::default();
    assert_eq!(n.blur_type, BlurType::Soft);
    assert_eq!(
        (n.soft, n.lens, n.motion, n.zoom),
        (d.soft, d.lens, d.motion, d.zoom)
    );
    assert!(n.respect_mask_edges);
    assert!(n.bypass);
    assert_eq!(n.name, "Sky");
    assert_eq!(n.masks.items.len(), 1);
    assert_eq!(r.nodes.links, b.nodes.links);
    // Reset of one field: the others stay.
    let one = SettingsPatch::reset_in(&b, uid(2), SettingsField::Zoom, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(node_of(&r).zoom, d.zoom);
    assert_eq!(node_of(&r).blur_type, BlurType::Zoom);
    assert_eq!(node_of(&r).lens.blades, 7);
    let one = SettingsPatch::reset_in(&b, uid(2), SettingsField::BlurLens, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(node_of(&r).lens, d.lens);
    assert_eq!(node_of(&r).zoom.center, [0.1, 0.9]);
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
fn copy_carries_the_node_and_paste_replaces_the_targets_graph() {
    let mut src = base();
    src.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "type": "zoom", "name": "Zoomed",
            "zoom": { "amount": 70, "center": [0.2, 0.8] },
            "lens": { "blades": 6 },
            "masks": { "items": [ { "kind": "radial" },
                                  { "kind": "object", "points": [[0.5, 0.5, true]] } ] } } } })),
        None,
    )
    .unwrap();
    let resolved = resolved_of(&src);
    // The node has no Auto: the resolved settings hold it as it is.
    assert_eq!(resolved.settings.nodes.blur(2), src.nodes.blur(2));
    let copy = SettingsPatch::copied_from(&resolved);
    let node = copy.nodes.get(&uid(2)).expect("the node is in the copy");
    assert_eq!(node.kind, Some(NodeKind::Blur));
    let mut target = DevelopSettings::default();
    target
        .apply_patch(
            &patch_of(json!({ "nodes": { "7": { "kind": "blur" } },
                "links": serde_json::to_value(line(&[1, 7])).unwrap() })),
            None,
        )
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(paste.remove.contains(&uid(7)), "the target's own node goes");
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.blur(7).is_none());
    let n = target.nodes.blur(2).expect("pasted");
    assert_eq!(n.blur_type, BlurType::Zoom);
    assert_eq!(n.zoom, node_of(&src).zoom);
    assert_eq!(n.lens, node_of(&src).lens);
    assert_eq!(n.name, "Zoomed");
    assert_eq!(n.masks.items.len(), 2);
    match &n.masks.items[1].shape {
        ym_core::settings::MaskShape::Object(o) => assert!(o.points.is_empty() && o.box_.is_none()),
        other => panic!("{other:?}"),
    }
    assert_eq!(target.nodes.links, src.nodes.links);
}

#[test]
fn the_xmp_holds_the_node_and_reads_back() {
    let mut s = base();
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "type": "lens", "lens": { "radius": 50, "blades": 6, "rotation_deg": 15, "highlights": 60 } } } })),
        None,
    )
    .unwrap();
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"kind\":\"blur\""));
    assert!(text.contains("\"type\":\"lens\""));
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, s);
    let resolved = resolved_of(&s);
    assert_eq!(resolved.settings.nodes.blur(2).unwrap().lens.blades, 6);
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
        let mut node = json!({ "id": 2, "kind": "blur" });
        node.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        json!({ "nodes": { "items": [node] } })
    };
    for body in [
        json!({ "range": {} }),
        json!({ "trim": {} }),
        json!({ "type": "spin" }),
        json!({ "type": 3 }),
        json!({ "soft": { "radius": 101 } }),
        json!({ "soft": { "radius": -1 } }),
        json!({ "soft": { "x": 1 } }),
        json!({ "lens": { "blades": 13 } }),
        json!({ "lens": { "blades": 6.5 } }),
        json!({ "lens": { "highlights": 101 } }),
        json!({ "motion": { "length": 101 } }),
        json!({ "zoom": { "amount": 101 } }),
        json!({ "zoom": { "center": [0.5] } }),
        json!({ "zoom": { "center": [0.5, 1.5] } }),
        json!({ "respect_mask_edges": "yes" }),
    ] {
        assert!(!v.is_valid(&bad(body.clone())), "{body}");
    }
    for body in [
        json!({}),
        json!({ "type": "zoom" }),
        json!({ "lens": { "rotation_deg": 725, "blades": 1 } }),
        json!({ "motion": { "angle_deg": -30 } }),
        json!({ "zoom": { "center": [0, 1] } }),
        json!({ "lens": { "radius": 50 } }),
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
    let d = BlurNode::default();
    let defs = &schema["$defs"];
    assert_eq!(
        range(&defs["blur_soft"]["properties"]["radius"]),
        BLUR_AMOUNT_RANGE
    );
    assert_eq!(
        defs["blur_soft"]["properties"]["radius"]["default"],
        json!(d.soft.radius as i64)
    );
    let l = &defs["blur_lens"]["properties"];
    assert_eq!(range(&l["radius"]), BLUR_AMOUNT_RANGE);
    assert_eq!(range(&l["highlights"]), BLUR_AMOUNT_RANGE);
    assert_eq!(l["blades"]["maximum"], json!(BLUR_BLADES_MAX));
    assert_eq!(l["radius"]["default"], json!(d.lens.radius as i64));
    assert_eq!(l["blades"]["default"], json!(d.lens.blades));
    assert_eq!(
        range(&defs["blur_motion"]["properties"]["length"]),
        BLUR_AMOUNT_RANGE
    );
    assert_eq!(
        defs["blur_motion"]["properties"]["length"]["default"],
        json!(d.motion.length as i64)
    );
    let z = &defs["blur_zoom"]["properties"];
    assert_eq!(range(&z["amount"]), BLUR_AMOUNT_RANGE);
    assert_eq!(range(&z["center"]["items"]), BLUR_CENTER_RANGE);
    assert_eq!(z["center"]["default"], json!(d.zoom.center));
    let n = &defs["blur_node"]["properties"];
    assert_eq!(n["name"]["default"], json!(d.name));
    assert_eq!(n["kind"]["const"], json!("blur"));
    assert_eq!(n["respect_mask_edges"]["default"], json!(true));
    // The types of the schema are the core's, in the order of the control.
    let words: Vec<&str> = BlurType::ALL.iter().map(|t| t.name()).collect();
    assert_eq!(defs["blur_type"]["enum"], json!(words));
    assert_eq!(defs["blur_type"]["default"], json!("soft"));
}

#[test]
fn the_kind_is_in_all_and_its_names_are_the_specs() {
    assert!(NodeKind::ALL.contains(&NodeKind::Blur));
    assert_eq!(NodeKind::Blur.name(), "blur");
    assert_eq!(NodeKind::Blur.with_article(), "a Blur node");
    assert_eq!(NodeKind::Blur.sentence(), "A Blur node");
    assert_eq!(NodeKind::parse("blur").unwrap(), NodeKind::Blur);
    assert!(NodeKind::Blur.is_numbered_extra());
    assert!(!NodeKind::Blur.is_mixer_family());
    assert!(NodeKind::parse("outside").is_err());
    for f in SettingsField::ALL {
        let has = f.belongs_to(NodeKind::Blur);
        let want = matches!(
            f,
            SettingsField::Name
                | SettingsField::Bypass
                | SettingsField::Pos
                | SettingsField::Masks
                | SettingsField::BlurType
                | SettingsField::Soft
                | SettingsField::BlurLens
                | SettingsField::Motion
                | SettingsField::Zoom
                | SettingsField::RespectMaskEdges
        );
        assert_eq!(has, want, "{f:?}");
    }
    // The JSON names of the fields are the spec's (`lens` is the Input's too).
    assert_eq!(SettingsField::BlurType.name(), "type");
    assert_eq!(SettingsField::BlurLens.name(), "lens");
    assert!(!SettingsField::Lens.belongs_to(NodeKind::Blur));
    assert!(!SettingsField::BlurLens.belongs_to(NodeKind::Input));
}

#[test]
fn a_document_that_holds_the_node_is_still_refused_where_the_kind_is_unknown() {
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "outside" })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert!(e.to_string().contains("outside"), "{e}");
}
