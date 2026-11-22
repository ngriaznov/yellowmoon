//! The Colour Uniform node's `trim` and `grade` (`docs/32` section 12a.3, task CG-02): written only when they differ from the
//! defaults, read as the defaults when left out, Auto read as a manual 0, clamping and reports, every patch builder, presets, an
//! older build's refusal and the JSON schema.
#![allow(clippy::field_reassign_with_default)]

mod common;

use common::rich::resolved_of;
use serde_json::{Value, json};
use ym_core::presets::{Preset, PresetGroup};
use ym_core::settings::{
    DevelopSettings, FieldRef, GraphError, NodeId, NodeKind, SettingsError, SettingsField,
    SettingsPatch,
};
use ym_looks::library::LookId;

fn read(doc: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    DevelopSettings::from_value(doc, &LookId::neutral())
}

fn node(extra: Value) -> Value {
    let mut n = json!({ "id": 2, "kind": "color_uniform",
        "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
        "compress": { "hue": 60, "saturation": 60, "lightness": 0 } });
    if let (Some(m), Some(e)) = (n.as_object_mut(), extra.as_object()) {
        m.extend(e.clone());
    }
    n
}

fn doc_of(extra: Value) -> Value {
    json!({ "nodes": { "items": [node(extra)], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } })
}

fn settings_of(extra: Value) -> DevelopSettings {
    read(doc_of(extra)).expect("a document").0
}

fn item(s: &DevelopSettings) -> Value {
    let v = serde_json::to_value(s).expect("settings are JSON");
    v["nodes"]["items"]
        .as_array()
        .and_then(|a| a.iter().find(|n| n["id"] == 2))
        .cloned()
        .expect("node 2")
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

fn uid() -> NodeId {
    NodeId::Corrector(2)
}

const NEUTRAL_GRADE: fn() -> Value = || {
    json!({
        "contrast": { "kind": "manual", "value": 0 }, "brightness": 0,
        "highlights": { "kind": "manual", "value": 0 }, "shadows": { "kind": "manual", "value": 0 },
        "white": 0, "black": { "kind": "manual", "value": 0 },
        "color_balance": { "shadows": { "kind": "manual", "point": [0, 0] }, "midtones": [0, 0],
                           "highlights": { "kind": "manual", "point": [0, 0] } },
        "saturation": { "kind": "manual", "value": 0 },
        "color_warp": { "pins": [] },
        "curve": { "luma": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]],
                   "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]] } })
};

#[test]
fn a_node_with_neither_field_writes_no_trim_and_no_grade_and_the_same_bytes() {
    let s = settings_of(json!({}));
    let n = item(&s);
    assert!(n.get("trim").is_none() && n.get("grade").is_none(), "{n}");
    // The defaults written out by hand are the same document: the same bytes.
    let explicit = settings_of(json!({
        "trim": { "exposure_ev": 0, "temperature": 0, "tint": 0 }, "grade": NEUTRAL_GRADE() }));
    assert_eq!(
        serde_json::to_string(&s).unwrap(),
        serde_json::to_string(&explicit).unwrap()
    );
    assert_eq!(s, explicit);
    // A partial grade of Autos reads as the neutral grade, so it writes nothing either.
    let autos = settings_of(json!({ "grade": { "contrast": { "kind": "auto" } } }));
    assert!(item(&autos).get("grade").is_none());
    assert_eq!(autos, s);
}

#[test]
fn a_graded_node_round_trips_and_writes_both_fields_once() {
    let s = settings_of(json!({
        "trim": { "exposure_ev": 0.5 },
        "grade": { "contrast": { "kind": "manual", "value": 15 },
                   "curve": { "luma": [[0, 0], [0.5, 0.55], [1, 1]] } } }));
    let n = item(&s);
    assert_eq!(n["trim"]["exposure_ev"], 0.5);
    assert_eq!(n["grade"]["contrast"]["value"], 15.0);
    assert_eq!(n["grade"]["curve"]["luma"][1][1], 0.55);
    // A grade is written whole, as a corrector's is.
    assert_eq!(n["grade"]["saturation"], json!({ "kind": "manual", "value": 0.0 }));
    let text = serde_json::to_string(&s).unwrap();
    assert_eq!(text.matches("\"trim\"").count(), 2, "Main's and the node's: {text}");
    let back = read(serde_json::from_str(&text).unwrap()).unwrap().0;
    assert_eq!(back, s);
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
    // A trim alone is written too.
    let t = settings_of(json!({ "trim": { "tint": -20 } }));
    assert!(item(&t).get("trim").is_some() && item(&t).get("grade").is_none());
}

#[test]
fn auto_reads_as_a_manual_zero_in_a_document_a_patch_and_a_paste() {
    // A document.
    let s = settings_of(json!({ "grade": { "contrast": { "kind": "auto" }, "brightness": 4,
        "color_balance": { "shadows": { "kind": "auto" } } } }));
    let g = &s.nodes.color_uniform(2).unwrap().grade;
    assert!(!g.contrast.is_auto() && !g.color_balance.shadows.is_auto());
    // A patch that sets Auto stores a manual 0.
    let mut s = settings_of(json!({}));
    let p = patch_of(json!({ "nodes": { "2": { "grade": {
        "contrast": { "kind": "auto" }, "saturation": { "kind": "auto" } } } } }));
    s.apply_patch(&p, None).unwrap();
    let g = &s.nodes.color_uniform(2).unwrap().grade;
    assert!(!g.contrast.is_auto() && !g.saturation.is_auto());
    assert!(item(&s).get("grade").is_none(), "a manual 0 is the neutral grade");
    // A paste: Main keeps its Auto in the target's Main and the node gets a manual 0.
    let source = settings_of(json!({ "grade": { "contrast": { "kind": "manual", "value": 9 } } }));
    let whole = SettingsPatch::copied_from(&resolved_of(&source));
    let mut onto = DevelopSettings::default();
    onto.apply_patch(&whole.pasted_onto(&onto.clone()), None).unwrap();
    assert!(onto.main().grade.contrast.is_auto());
    assert_eq!(onto.nodes.color_uniform(2), source.nodes.color_uniform(2));
}

#[test]
fn clamping_is_reported_under_the_nodes_path() {
    let (s, clamped) = read(doc_of(json!({
        "trim": { "tint": 500 }, "grade": { "contrast": { "kind": "manual", "value": 900 } } })))
    .unwrap();
    let n = s.nodes.color_uniform(2).unwrap();
    assert_eq!(n.trim.tint, 100.0);
    assert!(clamped.contains(&FieldRef { node: uid(), field: SettingsField::Trim }));
    assert!(clamped.contains(&FieldRef { node: uid(), field: SettingsField::Contrast }));
    // A patch reports it too, and applies nothing for a field it does not clamp.
    let mut s = settings_of(json!({}));
    let rep = s
        .apply_patch(&patch_of(json!({ "nodes": { "2": { "trim": { "exposure_ev": 40 } } } })), None)
        .unwrap();
    assert!(rep.clamped.contains(&FieldRef { node: uid(), field: SettingsField::Trim }));
    assert_eq!(s.nodes.color_uniform(2).unwrap().trim.exposure_ev, 4.0);
}

#[test]
fn an_unknown_field_is_still_refused_and_the_older_builds_sentence_is_pinned() {
    let e = read(doc_of(json!({ "strength": 3 }))).unwrap_err();
    assert_eq!(e.to_string(), "A Colour Uniform node has no field called “strength”.");
    // What a build without the fields says for `grade` (and for `trim`): the sentence of `docs/23` section 3.5.
    for field in ["grade", "trim"] {
        let old = GraphError::UnknownField {
            kind: NodeKind::ColorUniform.sentence(),
            field: field.to_owned(),
        };
        assert_eq!(
            old.to_string(),
            format!("A Colour Uniform node has no field called “{field}”.")
        );
    }
    // A kind that never had them still refuses them.
    let e = read(json!({ "nodes": { "items": [{ "id": 2, "kind": "blur", "grade": {} }] } })).unwrap_err();
    assert_eq!(e.to_string(), "A Blur node has no field called “grade”.");
    // A patch to a Colour Uniform node takes them; a patch with a field it has not is refused.
    let mut s = settings_of(json!({}));
    s.apply_patch(&patch_of(json!({ "nodes": { "2": { "trim": { "tint": 3 } } } })), None).unwrap();
    assert!(
        s.apply_patch(&patch_of(json!({ "nodes": { "2": { "film": { "exposure_ev": 1 } } } })), None)
            .is_err()
    );
}

#[test]
fn every_patch_builder_carries_the_fields() {
    let plain = settings_of(json!({}));
    let graded = settings_of(json!({
        "trim": { "exposure_ev": 0.5, "temperature": 12 },
        "grade": { "contrast": { "kind": "manual", "value": 15 }, "brightness": -7, "white": 3,
                   "saturation": { "kind": "manual", "value": -20 },
                   "color_balance": { "midtones": [0.1, -0.2] },
                   "color_warp": { "pins": [ { "source": [0.05, 0.02], "target": [0.07, 0.03] } ] },
                   "curve": { "red": [[0, 0], [0.4, 0.45], [1, 1]] } } }));
    // Undo's diff and its inverse, both ways.
    for (from, to) in [(&plain, &graded), (&graded, &plain)] {
        let patch = SettingsPatch::between(from, to);
        let mut s = from.clone();
        let rep = s.apply_patch(&patch, None).unwrap();
        assert_eq!(&s, to);
        assert!(!rep.is_empty());
    }
    // Copy and paste to another photo (the whole-node patch).
    let whole = SettingsPatch::copied_from(&resolved_of(&graded));
    let mut onto = DevelopSettings::default();
    onto.apply_patch(&whole.pasted_onto(&onto.clone()), None).unwrap();
    assert_eq!(onto.nodes.color_uniform(2), graded.nodes.color_uniform(2));
    // One field changed is one field reported.
    let mut s = graded.clone();
    let rep = s
        .apply_patch(&patch_of(json!({ "nodes": { "2": { "grade": { "brightness": 9 } } } })), None)
        .unwrap();
    assert_eq!(rep.changed, vec![FieldRef { node: uid(), field: SettingsField::Brightness }]);
    // The same value is not a change.
    let rep = s
        .apply_patch(&patch_of(json!({ "nodes": { "2": { "grade": { "brightness": 9 } } } })), None)
        .unwrap();
    assert!(rep.changed.is_empty());
    // The grade fields belong to the kind, and to the nodes' group.
    for f in [SettingsField::Trim, SettingsField::Contrast, SettingsField::ColorWarp, SettingsField::Curve] {
        assert!(f.belongs_to(NodeKind::ColorUniform), "{f:?}");
        assert_eq!(f.group(), SettingsField::Target.group());
    }
}

#[test]
fn reset_this_node_puts_target_compress_trim_and_grade_back_and_keeps_the_masks() {
    let mut s = settings_of(json!({ "trim": { "tint": 9 },
        "grade": { "brightness": 5 },
        "masks": { "invert": false, "items": [ { "kind": "radial", "mode": "add" } ] } }));
    let reset = SettingsPatch::reset_node_in(&s, uid(), &LookId::neutral());
    s.apply_patch(&reset, None).unwrap();
    let n = s.nodes.color_uniform(2).unwrap();
    assert!(n.target.is_none() && n.compress.is_zero());
    assert_eq!(n.trim, Default::default());
    assert_eq!(n.grade, ym_core::settings::NodeGrade::neutral());
    assert_eq!(n.masks.items.len(), 1);
}

#[test]
fn a_preset_and_an_album_carry_the_fields() {
    let graded = settings_of(json!({ "trim": { "exposure_ev": 0.5 }, "grade": { "brightness": 5 } }));
    let preset = Preset::from_settings("Graded", &graded, &[PresetGroup::Nodes], None).unwrap();
    assert_eq!(preset.settings.nodes.color_uniform(2), graded.nodes.color_uniform(2));
    let applied = preset.applied_to(&DevelopSettings::default(), true).unwrap();
    assert_eq!(applied.nodes.color_uniform(2), graded.nodes.color_uniform(2));
    let back = Preset::from_json(&preset.to_json().unwrap()).unwrap();
    assert_eq!(back.settings, preset.settings);
    // A preset of a node with no grade writes neither field.
    let plain = Preset::from_settings("Plain", &settings_of(json!({})), &[PresetGroup::Nodes], None).unwrap();
    let text = plain.to_json().unwrap();
    assert!(!text.contains("\"grade\""), "{text}");
    // An album holds the settings as stored: the same round trip.
    let text = serde_json::to_string(&graded).unwrap();
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, graded);
}

#[test]
fn resolved_values_hold_the_nodes_trim_and_grade() {
    let graded = settings_of(json!({ "trim": { "tint": 9 }, "grade": { "brightness": 5 } }));
    let r = resolved_of(&graded);
    let v = r.values.nodes.iter().find(|n| n.id == 2).expect("node 2 in values");
    assert_eq!(v.trim.tint, 9.0);
    assert_eq!(v.grade.brightness, 5.0);
}

#[test]
fn the_schema_validates_both_forms_and_refuses_a_field_it_does_not_have() {
    let schema: Value = serde_json::from_str(include_str!("../schema/settings.schema.json")).unwrap();
    let node_def = &schema["$defs"]["color_uniform_node"]["properties"];
    assert_eq!(node_def["trim"]["$ref"], "#/$defs/trim");
    assert_eq!(node_def["grade"]["$ref"], "#/$defs/grade");
    assert_eq!(schema["$defs"]["color_uniform_node"]["additionalProperties"], false);
}
