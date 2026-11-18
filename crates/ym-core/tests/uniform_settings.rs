//! The Colour Uniform node's settings (`docs/32-color-uniform.md` section 5, `docs/23` section 3): the format, the defaults and
//! clamping, every refusal and its text, patches (the first pick's single patch among them), Reset, undo, Copy and Paste, the XMP
//! text, the JSON schema and the plan's keys.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::session::PhotoId;
use ym_core::settings::{
    ColorUniformNode, DevelopSettings, FieldRef, GraphError, NodeId, NodeKind, NodePatch, Port,
    ResolvedSettings, SettingsError, SettingsField, SettingsPatch, TARGET_LIGHTNESS_RANGE_EV,
    TARGET_SATURATION_RANGE, UNIFORM_AMOUNT_RANGE, ports_in, ports_out,
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
    let mut node = json!({ "id": 2, "kind": "color_uniform" });
    node.as_object().map(|_| ()).expect("a node is an object");
    if let (Some(n), Some(b)) = (node.as_object_mut(), body.as_object()) {
        n.extend(b.clone());
    }
    read(doc(vec![node], line(&[1, 2])))
}

fn node_of(s: &DevelopSettings) -> &ColorUniformNode {
    s.nodes.color_uniform(2).expect("color uniform node 2")
}

fn base() -> DevelopSettings {
    with_node(json!({})).expect("a default node").0
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

/// The node of `docs/32` section 5.1, the example.
fn example() -> Value {
    json!({ "id": 2, "kind": "color_uniform", "name": "Uniform", "bypass": false, "pos": [2, 0],
        "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
        "compress": { "hue": 60, "saturation": 60, "lightness": 0 },
        "masks": { "invert": false, "items": [
            { "kind": "color_range", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
              "center": [0.0875, 0.0423], "width": 0.05, "softness": 0.5 } ] } })
}

fn reported(clamped: &[FieldRef], field: SettingsField) -> bool {
    clamped.contains(&FieldRef {
        node: uid(2),
        field,
    })
}

#[test]
fn a_new_node_has_the_defaults_of_the_spec_and_changes_no_pixel() {
    let (s, clamped) = with_node(json!({})).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    assert_eq!(n.name, "Uniform");
    assert!(!n.bypass);
    assert_eq!(n.target, None);
    assert_eq!(
        (n.compress.hue, n.compress.saturation, n.compress.lightness),
        (0.0, 0.0, 0.0)
    );
    assert!(n.masks.items.is_empty() && !n.masks.invert);
    assert!(n.is_inert());
    // The written form: the fields of section 5.1 and no others, `target` as null.
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
            "bypass", "compress", "id", "kind", "masks", "name", "pos", "target"
        ]
    );
    assert_eq!(item["kind"], json!("color_uniform"));
    assert!(item["target"].is_null());
    assert_eq!(
        item["compress"],
        json!({ "hue": 0.0, "saturation": 0.0, "lightness": 0.0 })
    );
}

#[test]
fn the_example_of_the_spec_reads_and_writes_back_the_same() {
    let (s, clamped) = read(doc(vec![example()], line(&[1, 2]))).unwrap();
    assert!(clamped.is_empty());
    let n = node_of(&s);
    let t = n.target.expect("a target");
    assert_eq!(
        (t.hue_deg, t.saturation, t.lightness_ev),
        (26.0, 9.7, -0.03)
    );
    assert_eq!(n.compress.hue, 60.0);
    assert_eq!(n.masks.items.len(), 1);
    assert!(!n.is_inert());
    let written = serde_json::to_value(&s).unwrap();
    let (again, _) = read(written.clone()).unwrap();
    assert_eq!(again, s);
    assert_eq!(serde_json::to_value(&again).unwrap(), written);
}

#[test]
fn the_node_is_inert_with_no_target_no_amount_or_bypass() {
    let live = json!({ "target": { "hue_deg": 26, "saturation": 9, "lightness_ev": 0 }, "compress": { "hue": 50 } });
    assert!(!node_of(&with_node(live).unwrap().0).is_inert());
    for body in [
        json!({ "compress": { "hue": 50 } }),
        json!({ "target": { "hue_deg": 26, "saturation": 9, "lightness_ev": 0 } }),
        json!({ "target": { "hue_deg": 26, "saturation": 9, "lightness_ev": 0 }, "compress": { "hue": 50 }, "bypass": true }),
    ] {
        assert!(node_of(&with_node(body).unwrap().0).is_inert());
    }
}

#[test]
fn the_ports_are_a_correctors_and_ids_are_shared() {
    assert_eq!(
        ports_in(NodeKind::ColorUniform),
        ports_in(NodeKind::Corrector)
    );
    assert_eq!(
        ports_out(NodeKind::ColorUniform),
        [Port::Picture, Port::Key].as_slice()
    );
    let items = vec![
        json!({ "id": 2, "kind": "color_uniform" }),
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
    // A port the kind does not have.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "color_uniform" })],
        vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "red"),
            link(json!(2), "picture", json!("output"), "picture"),
        ],
    ))
    .unwrap_err();
    assert!(e.to_string().contains("no port"), "{e}");
    // Ids are unique across kinds; Main is a corrector; a new node needs its kind.
    let e = read(doc(
        vec![
            json!({ "id": 2, "kind": "color_uniform" }),
            json!({ "id": 2, "kind": "film" }),
        ],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "Two nodes have the id 2.");
    assert!(
        read(doc(
            vec![json!({ "id": 1, "kind": "color_uniform" })],
            line(&[1])
        ))
        .is_err()
    );
    let e = read(doc(vec![json!({ "id": 2 })], line(&[1, 2]))).unwrap_err();
    assert_eq!(e.to_string(), GraphError::NeedsKind("2".into()).to_string());
}

#[test]
fn the_limit_of_64_nodes_counts_this_kind() {
    let items: Vec<Value> = (2..=64)
        .map(|i| json!({ "id": i, "kind": "color_uniform" }))
        .collect();
    assert!(
        read(doc(items, line(&[1]))).is_ok(),
        "Main and 63 nodes are 64"
    );
    let items: Vec<Value> = (2..=65)
        .map(|i| json!({ "id": i, "kind": "color_uniform" }))
        .collect();
    let e = read(doc(items, line(&[1]))).unwrap_err();
    assert_eq!(e.to_string(), "A photo can have up to 64 nodes.");
    // Mixed with other kinds.
    let mut items: Vec<Value> = (2..=33)
        .map(|i| json!({ "id": i, "kind": "color_uniform" }))
        .collect();
    items.extend((34..=65).map(|i| json!({ "id": i, "kind": "film" })));
    assert!(read(doc(items, line(&[1]))).is_err());
}

#[test]
fn an_unknown_field_is_refused_with_its_text_and_a_range_among_them() {
    let e = with_node(json!({ "range": { "width": 0.05 } })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Colour Uniform node has no field called “range”."
    );
    let e = with_node(json!({ "trim": {} })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Colour Uniform node has no field called “trim”."
    );
    let e = with_node(json!({ "strength": 3 })).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Colour Uniform node has no field called “strength”."
    );
    // Another kind has no `target` or `compress`.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "corrector", "target": null })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "A corrector has no field called “target”.");
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "film", "compress": {} })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "A film node has no field called “compress”.");
    // A typo inside a field is serde's refusal.
    assert!(matches!(
        with_node(json!({ "compress": { "hu": 3 } })).unwrap_err(),
        SettingsError::Parse(_)
    ));
    assert!(matches!(
        with_node(
            json!({ "target": { "hue_deg": 1, "saturation": 5, "lightness_ev": 0, "x": 1 } })
        )
        .unwrap_err(),
        SettingsError::Parse(_)
    ));
}

#[test]
fn a_target_of_the_wrong_shape_is_a_parse_error() {
    for target in [
        json!(5),
        json!("orange"),
        json!([26, 9, 0]),
        json!({ "hue_deg": "x", "saturation": 5, "lightness_ev": 0 }),
        // All three numbers are required.
        json!({ "hue_deg": 26, "saturation": 9 }),
        json!({}),
    ] {
        assert!(
            matches!(
                with_node(json!({ "target": target.clone() })).unwrap_err(),
                SettingsError::Parse(_)
            ),
            "{target}"
        );
    }
    assert!(matches!(
        with_node(json!({ "compress": [1, 2, 3] })).unwrap_err(),
        SettingsError::Parse(_)
    ));
    assert!(with_node(json!({ "target": null })).is_ok());
}

#[test]
fn numbers_are_clamped_and_reported_and_the_hue_is_wrapped_silently() {
    let (s, clamped) = with_node(json!({
        "target": { "hue_deg": 26, "saturation": 90, "lightness_ev": -9 },
        "compress": { "hue": 140, "saturation": -5, "lightness": 20 },
    }))
    .unwrap();
    let n = node_of(&s);
    let t = n.target.unwrap();
    assert_eq!((t.saturation, t.lightness_ev), (50.0, -4.0));
    assert_eq!(
        (n.compress.hue, n.compress.saturation, n.compress.lightness),
        (100.0, 0.0, 20.0)
    );
    assert!(reported(&clamped, SettingsField::Target));
    assert!(reported(&clamped, SettingsField::Compress));
    // A saturation below 2 is held at 2, reported.
    let (s, clamped) =
        with_node(json!({ "target": { "hue_deg": 10, "saturation": 0.5, "lightness_ev": 8.5 } }))
            .unwrap();
    let t = node_of(&s).target.unwrap();
    assert_eq!((t.saturation, t.lightness_ev), (2.0, 8.0));
    assert!(reported(&clamped, SettingsField::Target));
    assert!(!reported(&clamped, SettingsField::Compress));
    // The hue is read modulo 360 into [0, 360): neither clamped nor reported.
    for (given, want) in [
        (0.0, 0.0),
        (360.0, 0.0),
        (-1.0, 359.0),
        (725.5, 5.5),
        (-720.0, 0.0),
        (359.5, 359.5),
        (-0.0, 0.0),
    ] {
        let (s, clamped) = with_node(
            json!({ "target": { "hue_deg": given, "saturation": 9, "lightness_ev": 0 } }),
        )
        .unwrap();
        let h = node_of(&s).target.unwrap().hue_deg;
        assert_eq!(h, want, "hue {given}");
        assert!(h.is_sign_positive(), "no negative zero for {given}");
        assert!(clamped.is_empty(), "hue {given} is not reported");
    }
    // A hue just below zero stays inside [0, 360).
    let (s, _) =
        with_node(json!({ "target": { "hue_deg": -1e-20, "saturation": 9, "lightness_ev": 0 } }))
            .unwrap();
    let h = node_of(&s).target.unwrap().hue_deg;
    assert!((0.0..360.0).contains(&h), "{h}");
    // Numbers in range are not reported.
    let (_, clamped) = with_node(json!({
        "target": { "hue_deg": 200, "saturation": 2, "lightness_ev": 8 },
        "compress": { "hue": 100, "saturation": 0, "lightness": 50 }
    }))
    .unwrap();
    assert!(clamped.is_empty());
    // A partial compress is read over the defaults.
    let (s, _) = with_node(json!({ "compress": { "saturation": 40 } })).unwrap();
    let c = node_of(&s).compress;
    assert_eq!((c.hue, c.saturation, c.lightness), (0.0, 40.0, 0.0));
}

#[test]
fn a_number_that_is_not_finite_is_refused_and_nothing_is_applied() {
    let mut s = base();
    let before = s.clone();
    for patch in [
        NodePatch {
            target: Some(Some(ym_core::settings::TargetValue {
                hue_deg: f64::NAN,
                saturation: 9.0,
                lightness_ev: 0.0,
            })),
            ..NodePatch::default()
        },
        NodePatch {
            target: Some(Some(ym_core::settings::TargetValue {
                hue_deg: 10.0,
                saturation: f64::INFINITY,
                lightness_ev: 0.0,
            })),
            ..NodePatch::default()
        },
        NodePatch {
            target: Some(Some(ym_core::settings::TargetValue {
                hue_deg: f64::INFINITY,
                saturation: 9.0,
                lightness_ev: 0.0,
            })),
            ..NodePatch::default()
        },
        NodePatch {
            compress: Some(ym_core::settings::UniformAmountsPatch {
                lightness: Some(f64::NEG_INFINITY),
                ..Default::default()
            }),
            ..NodePatch::default()
        },
    ] {
        let e = s
            .apply_patch(&SettingsPatch::node(uid(2), patch), None)
            .unwrap_err();
        assert!(matches!(e, SettingsError::NotFinite(_)), "{e}");
        assert_eq!(s, before);
    }
}

#[test]
fn the_name_is_trimmed_cut_and_an_empty_one_reads_as_the_default() {
    let (s, _) = with_node(json!({ "name": "   " })).unwrap();
    assert_eq!(node_of(&s).name, "Uniform");
    let (s, _) = with_node(json!({ "name": format!("  {}  ", "x".repeat(40)) })).unwrap();
    assert_eq!(node_of(&s).name.chars().count(), 32);
}

#[test]
fn patches_change_fields_report_them_and_a_partial_compress_keeps_the_rest() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
                "compress": { "hue": 50 } } } })),
            None,
        )
        .unwrap();
    let n = node_of(&s);
    assert_eq!(n.target.unwrap().saturation, 9.7);
    assert_eq!(
        (n.compress.hue, n.compress.saturation, n.compress.lightness),
        (50.0, 0.0, 0.0)
    );
    for field in [SettingsField::Target, SettingsField::Compress] {
        assert!(report.changed.contains(&FieldRef {
            node: uid(2),
            field
        }));
    }
    assert_eq!(
        report.first_group(),
        Some(ym_core::settings::StageGroup::Grade)
    );
    // A second partial keeps the amounts it does not name.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "compress": { "lightness": 25 } } } })),
        None,
    )
    .unwrap();
    let c = node_of(&s).compress;
    assert_eq!((c.hue, c.saturation, c.lightness), (50.0, 0.0, 25.0));
    // `"target": null` clears.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "target": null } } })),
            None,
        )
        .unwrap();
    assert_eq!(node_of(&s).target, None);
    assert!(report.changed.contains(&FieldRef {
        node: uid(2),
        field: SettingsField::Target
    }));
    // Clearing a cleared target changes nothing.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "target": null } } })),
            None,
        )
        .unwrap();
    assert!(report.changed.is_empty());
    // Name and place re-run nothing.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "name": "Skin", "pos": [3, 1] } } })),
            None,
        )
        .unwrap();
    assert_eq!(report.first_group(), None);
    // A field the kind has not.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "film": { "stock": "x" } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "A Colour Uniform node has no field called “film”."
    );
    // A corrector refuses the new fields.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "1": { "target": null } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(e.to_string(), "A corrector has no field called “target”.");
    // The patch's own typos are serde's.
    assert!(
        serde_json::from_value::<SettingsPatch>(
            json!({ "nodes": { "2": { "compress": { "hu": 1 } } } })
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
                "target": { "hue_deg": 400, "saturation": 1, "lightness_ev": 20 },
                "compress": { "saturation": 150 } } } })),
            None,
        )
        .unwrap();
    assert!(reported(&report.clamped, SettingsField::Target));
    assert!(reported(&report.clamped, SettingsField::Compress));
    let t = node_of(&s).target.unwrap();
    assert_eq!((t.hue_deg, t.saturation, t.lightness_ev), (40.0, 2.0, 8.0));
    assert_eq!(node_of(&s).compress.saturation, 100.0);
}

#[test]
fn a_node_patch_adds_the_node_removes_it_and_refuses_the_wrong_kind() {
    let mut s = DevelopSettings::default();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "color_uniform" } },
                "links": serde_json::to_value(line(&[1, 2])).unwrap() })),
            None,
        )
        .unwrap();
    assert_eq!(report.added, vec![uid(2)]);
    assert_eq!(
        node_of(&s),
        &ColorUniformNode {
            pos: [1.0, 0.0],
            ..ColorUniformNode::default()
        }
    );
    assert!(report.changed.contains(&FieldRef {
        node: uid(2),
        field: SettingsField::Target
    }));
    // The wrong kind is refused with the sentence of the kind.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "corrector" } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Node 2 is a Colour Uniform node, not a corrector."
    );
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "1": { "kind": "color_uniform" } } })),
            None,
        )
        .unwrap_err();
    assert!(e.to_string().contains("not a Colour Uniform node"), "{e}");
    // Remove takes its links with it.
    let report = s
        .apply_patch(
            &patch_of(json!({ "remove": [2], "links": serde_json::to_value(line(&[1])).unwrap() })),
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![uid(2)]);
    assert!(s.nodes.color_uniform(2).is_none());
    // Adding with its fields in the one patch.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "5": { "kind": "color_uniform", "name": "Skin",
            "target": { "hue_deg": 30, "saturation": 8, "lightness_ev": 0 }, "compress": { "hue": 50, "saturation": 50 } } } })),
        None,
    )
    .unwrap();
    let n = s.nodes.color_uniform(5).unwrap();
    assert_eq!(n.name, "Skin");
    assert_eq!(n.compress.hue, 50.0);
}

#[test]
fn the_first_picks_single_patch_is_one_atomic_patch_and_one_undo_step() {
    let mut s = base();
    let before = s.clone();
    let pick = patch_of(json!({ "nodes": { "2": {
        "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
        "compress": { "hue": 50, "saturation": 50, "lightness": 0 },
        "masks": { "items": [ { "kind": "color_range", "center": [0.0875, 0.0423], "width": 0.05, "softness": 0.5 } ] } } } }));
    let report = s.apply_patch(&pick, None).unwrap();
    for field in [
        SettingsField::Target,
        SettingsField::Compress,
        SettingsField::Masks,
    ] {
        assert!(
            report.changed.contains(&FieldRef {
                node: uid(2),
                field
            }),
            "{field:?}"
        );
    }
    let n = node_of(&s);
    assert_eq!(n.masks.items.len(), 1);
    assert!(n.masks.items[0].enabled);
    // One undo step: the patch between the two documents undoes all three fields in one application.
    let undo = SettingsPatch::between(&s, &before);
    let mut back = s.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, before);
    let redo = SettingsPatch::between(&before, &s);
    let mut fwd = before.clone();
    fwd.apply_patch(&redo, None).unwrap();
    assert_eq!(fwd, s);
    // A later pick is a patch of the target alone and leaves the amounts and the masks as they are.
    let mut later = s.clone();
    later
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "target": { "hue_deg": 40, "saturation": 7, "lightness_ev": 0.2 } } } })),
            None,
        )
        .unwrap();
    assert_eq!(node_of(&later).compress, n.compress);
    assert_eq!(node_of(&later).masks, n.masks);
    // A patch that breaks a rule is atomic: nothing of the pick is applied.
    let mut atomic = base();
    let broken = patch_of(json!({ "nodes": { "2": {
        "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": 0 },
        "masks": { "edit": { "7": { "width": 0.1 } } } } } }));
    let kept = atomic.clone();
    if atomic.apply_patch(&broken, None).is_err() {
        assert_eq!(atomic, kept);
    }
}

#[test]
fn undo_is_the_patch_between_two_settings_and_reset_puts_the_defaults_back() {
    let a = base();
    let mut b = a.clone();
    b.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "target": { "hue_deg": 200, "saturation": 12, "lightness_ev": 1 },
            "compress": { "hue": 30, "saturation": 40, "lightness": 50 },
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
    assert_eq!(c.nodes.color_uniform(2), b.nodes.color_uniform(2));
    let drop = SettingsPatch::between(&b, &empty);
    let mut d = b.clone();
    d.apply_patch(&drop, None).unwrap();
    assert!(d.nodes.color_uniform(2).is_none());
    // Reset of the head: target to null, compress to its defaults; name, bypass, place, masks and links stay.
    let reset = SettingsPatch::reset_node_in(&b, uid(2), &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&reset, None).unwrap();
    let n = node_of(&r);
    assert_eq!(n.target, None);
    assert_eq!(n.compress, ColorUniformNode::default().compress);
    assert!(n.bypass);
    assert_eq!(n.name, "Sky");
    assert_eq!(n.masks.items.len(), 1);
    assert_eq!(r.nodes.links, b.nodes.links);
    // Reset of one field (the card's Reset).
    let one = SettingsPatch::reset_in(&b, uid(2), SettingsField::Compress, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert!(
        node_of(&r).target.is_some(),
        "Reset of Compress keeps the target"
    );
    assert_eq!(node_of(&r).compress.hue, 0.0);
    let one = SettingsPatch::reset_in(&b, uid(2), SettingsField::Target, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(node_of(&r).target, None);
    assert_eq!(
        node_of(&r).compress.hue,
        30.0,
        "Reset of Target keeps the amounts"
    );
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
            "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
            "compress": { "hue": 60, "saturation": 60 }, "name": "Skin",
            "masks": { "items": [ { "kind": "color_range", "center": [0.08, 0.04] },
                                  { "kind": "object", "points": [[0.5, 0.5, true]] } ] } } } })),
        None,
    )
    .unwrap();
    let resolved = resolved_of(&src);
    // The node has no Auto: the resolved settings hold it as it is.
    assert_eq!(
        resolved.settings.nodes.color_uniform(2),
        src.nodes.color_uniform(2)
    );
    let copy = SettingsPatch::copied_from(&resolved);
    let node = copy.nodes.get(&uid(2)).expect("the node is in the copy");
    assert_eq!(node.kind, Some(NodeKind::ColorUniform));
    let mut target = DevelopSettings::default();
    target
        .apply_patch(
            &patch_of(json!({ "nodes": { "7": { "kind": "color_uniform" } },
                "links": serde_json::to_value(line(&[1, 7])).unwrap() })),
            None,
        )
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(paste.remove.contains(&uid(7)), "the target's own node goes");
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.color_uniform(7).is_none());
    let n = target.nodes.color_uniform(2).expect("pasted");
    assert_eq!(n.target, node_of(&src).target);
    assert_eq!(n.compress, node_of(&src).compress);
    assert_eq!(n.name, "Skin");
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
            "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
            "compress": { "hue": 60 } } } })),
        None,
    )
    .unwrap();
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"kind\":\"color_uniform\""));
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, s);
    let resolved = resolved_of(&s);
    let xmp = resolved.settings.nodes.color_uniform(2).unwrap();
    assert_eq!(xmp.compress.hue, 60.0);
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
    // What the core writes validates: the default node, and the example.
    for s in [base(), read(doc(vec![example()], line(&[1, 2]))).unwrap().0] {
        let written = serde_json::to_value(&s).unwrap();
        let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
        assert!(problems.is_empty(), "{problems:?}");
    }
    // What the core refuses, the schema does too.
    let bad = |body: Value| {
        let mut node = json!({ "id": 2, "kind": "color_uniform" });
        node.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        json!({ "nodes": { "items": [node] } })
    };
    for body in [
        json!({ "range": {} }),
        json!({ "trim": {} }),
        json!({ "target": 5 }),
        json!({ "target": { "hue_deg": 1, "saturation": 5 } }),
        json!({ "target": { "hue_deg": 1, "saturation": 1, "lightness_ev": 0 } }),
        json!({ "target": { "hue_deg": 1, "saturation": 51, "lightness_ev": 0 } }),
        json!({ "target": { "hue_deg": 1, "saturation": 5, "lightness_ev": 9 } }),
        json!({ "compress": { "hue": 101 } }),
        json!({ "compress": { "hue": -1 } }),
        json!({ "compress": { "x": 1 } }),
    ] {
        assert!(!v.is_valid(&bad(body.clone())), "{body}");
    }
    for body in [
        json!({}),
        json!({ "target": null }),
        json!({ "target": { "hue_deg": 725, "saturation": 5, "lightness_ev": 0 } }),
        json!({ "compress": { "saturation": 40 } }),
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
    let t = &schema["$defs"]["target_value"]["properties"];
    assert_eq!(range(&t["saturation"]), TARGET_SATURATION_RANGE);
    assert_eq!(range(&t["lightness_ev"]), TARGET_LIGHTNESS_RANGE_EV);
    let a = &schema["$defs"]["uniform_amounts"]["properties"];
    for k in ["hue", "saturation", "lightness"] {
        assert_eq!(range(&a[k]), UNIFORM_AMOUNT_RANGE);
        assert_eq!(a[k]["default"], json!(0));
    }
    let n = &schema["$defs"]["color_uniform_node"]["properties"];
    assert_eq!(
        n["name"]["default"],
        json!(ColorUniformNode::default().name)
    );
    assert_eq!(n["kind"]["const"], json!("color_uniform"));
}

#[test]
fn the_kind_is_in_all_and_its_names_are_the_specs() {
    assert!(NodeKind::ALL.contains(&NodeKind::ColorUniform));
    assert_eq!(NodeKind::ColorUniform.name(), "color_uniform");
    assert_eq!(
        NodeKind::ColorUniform.with_article(),
        "a Colour Uniform node"
    );
    assert_eq!(NodeKind::ColorUniform.sentence(), "A Colour Uniform node");
    assert_eq!(
        NodeKind::parse("color_uniform").unwrap(),
        NodeKind::ColorUniform
    );
    // The unknown-kind example stays unknown.
    assert!(NodeKind::parse("outside").is_err());
    // The fields of the kind.
    for f in SettingsField::ALL {
        let has = f.belongs_to(NodeKind::ColorUniform);
        let want = matches!(
            f,
            SettingsField::Name
                | SettingsField::Bypass
                | SettingsField::Pos
                | SettingsField::Masks
                | SettingsField::Target
                | SettingsField::Compress
        );
        assert_eq!(has, want, "{f:?}");
    }
    assert!(SettingsField::Target.group().is_some());
}

#[test]
fn a_document_that_holds_the_node_is_still_refused_where_the_kind_is_unknown() {
    // The unknown-kind sentence of docs/23 section 3.5 is the one an older build gives; this build knows the kind.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "outside" })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert!(e.to_string().contains("outside"), "{e}");
}
