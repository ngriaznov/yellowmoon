//! Mask Intersect, the setting (`docs/23` sections 3.3, 3.5, 3.7 and 4.2; task MI-02): `masks.combine` reads as Add when it is
//! missing and is written only at Intersect, a bad value gives the sentence, patches, undo, copy and paste, presets, the JSON
//! schema.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::rich::{resolved_of, rich};
use serde_json::{Value, json};
use ym_core::presets::{Preset, PresetGroup};
use ym_core::settings::{
    DevelopSettings, MaskCombine, NodeId, SettingsError, SettingsField, SettingsPatch,
};
use ym_looks::library::LookId;

fn read(doc: Value) -> Result<DevelopSettings, SettingsError> {
    DevelopSettings::from_value(doc, &LookId::neutral()).map(|(s, _)| s)
}

fn patch_of(v: Value) -> Result<SettingsPatch, serde_json::Error> {
    serde_json::from_value(v)
}

const KINDS: [&str; 5] = [
    "corrector",
    "film",
    "face_refinement",
    "color_uniform",
    "blur",
];

/// A graph with a node of each kind that has masks (ids 2 to 6) and Main, each with two rows and `combine` as given.
fn every_kind(combine: Option<&str>) -> Value {
    let mut items = vec![json!({ "id": 1, "masks": masks(combine) })];
    for (i, kind) in KINDS.iter().enumerate() {
        items.push(json!({ "id": i + 2, "kind": kind, "masks": masks(combine) }));
    }
    json!({ "nodes": { "items": items } })
}

fn masks(combine: Option<&str>) -> Value {
    let mut m = json!({ "invert": false, "items": [
        { "kind": "radial", "mode": "add" },
        { "kind": "brightness_range", "mode": "subtract" } ] });
    if let Some(c) = combine {
        m["combine"] = json!(c);
    }
    m
}

fn all_masks(s: &DevelopSettings) -> Vec<(u32, MaskCombine)> {
    s.nodes.all_masks().map(|(id, m)| (id, m.combine)).collect()
}

#[test]
fn a_document_without_the_field_reads_as_add_and_writes_the_same_bytes() {
    let doc = every_kind(None);
    let s = read(doc).unwrap();
    assert_eq!(all_masks(&s).len(), 6);
    assert!(all_masks(&s).iter().all(|(_, c)| *c == MaskCombine::Add));
    let written = serde_json::to_string(&s).unwrap();
    assert!(!written.contains("combine"), "Add is left out: {written}");
    // The written form reads back to the same settings and the same bytes.
    let again = read(serde_json::from_str(&written).unwrap()).unwrap();
    assert_eq!(again, s);
    assert_eq!(serde_json::to_string(&again).unwrap(), written);
    // A document that says "add" reads as the same settings and is written without it.
    let said = read(every_kind(Some("add"))).unwrap();
    assert_eq!(said, s);
    assert_eq!(serde_json::to_string(&said).unwrap(), written);
    // The rich document of the other tests carries no field either.
    assert!(!serde_json::to_string(&rich()).unwrap().contains("combine"));
}

#[test]
fn an_intersect_document_round_trips_and_writes_the_field_once_a_node() {
    let s = read(every_kind(Some("intersect"))).unwrap();
    assert!(
        all_masks(&s)
            .iter()
            .all(|(_, c)| *c == MaskCombine::Intersect)
    );
    let written = serde_json::to_string(&s).unwrap();
    assert_eq!(written.matches("\"combine\":\"intersect\"").count(), 6);
    assert_eq!(read(serde_json::from_str(&written).unwrap()).unwrap(), s);
}

#[test]
fn a_bad_value_is_refused_with_the_sentence_and_nothing_is_read() {
    for (value, shown) in [
        (json!("merge"), "merge"),
        (json!(7), "7"),
        (json!(null), "null"),
        (json!("Add"), "Add"),
    ] {
        let mut doc = every_kind(None);
        doc["nodes"]["items"][2]["masks"]["combine"] = value;
        let err = read(doc).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("The masks of node 3 combine as “add” or “intersect”, not “{shown}”."),
        );
    }
    // Main and a patch say it the same way.
    let mut doc = every_kind(None);
    doc["nodes"]["items"][0]["masks"]["combine"] = json!("merge");
    assert!(
        read(doc)
            .unwrap_err()
            .to_string()
            .starts_with("The masks of node 1 combine as")
    );
    let s = read(every_kind(None)).unwrap();
    let before = s.clone();
    let err = patch_of(json!({ "nodes": { "4": { "masks": { "combine": "merge" } } } }))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("The masks of node 4 combine as “add” or “intersect”, not “merge”."),
        "{err}"
    );
    assert_eq!(s, before);
}

#[test]
fn combine_alone_is_one_patch_one_field_and_the_same_value_is_no_change() {
    let mut s = read(every_kind(None)).unwrap();
    let before = s.clone();
    let p = patch_of(json!({ "nodes": { "2": { "masks": { "combine": "intersect" } } } })).unwrap();
    let report = s.apply_patch(&p, None).unwrap();
    assert_eq!(report.changed.len(), 1, "{:?}", report.changed);
    assert_eq!(report.changed[0].field, SettingsField::Masks);
    assert_eq!(report.changed[0].node, NodeId::Corrector(2));
    assert_eq!(s.nodes.masks_of(2).unwrap().combine, MaskCombine::Intersect);
    assert_eq!(s.nodes.masks_of(2).unwrap().items.len(), 2, "the rows stay");
    assert_eq!(s.nodes.masks_of(3).unwrap().combine, MaskCombine::Add);
    let again = s.apply_patch(&p, None).unwrap();
    assert!(again.changed.is_empty(), "{:?}", again.changed);
    // Undo is the patch between the two documents, in one application; redo likewise.
    let undo = SettingsPatch::between(&s, &before);
    let mut back = s.clone();
    back.apply_patch(&undo, None).unwrap();
    assert_eq!(back, before);
    let redo = SettingsPatch::between(&before, &s);
    let mut fwd = before.clone();
    fwd.apply_patch(&redo, None).unwrap();
    assert_eq!(fwd, s);
}

#[test]
fn every_patch_builder_carries_combine_with_invert() {
    // Corrector, Film, Face Refinement, Colour Uniform and Blur: the patch that is built from the node (undo, copy, presets)
    // puts the mode back, in both directions.
    let add = read(every_kind(None)).unwrap();
    let int = read(every_kind(Some("intersect"))).unwrap();
    for (from, to) in [(&add, &int), (&int, &add)] {
        let patch = SettingsPatch::between(from, to);
        let mut s = from.clone();
        s.apply_patch(&patch, None).unwrap();
        assert_eq!(&s, to);
        for id in 1..=6 {
            assert_eq!(
                s.nodes.masks_of(id).unwrap().combine,
                to.nodes.masks_of(id).unwrap().combine,
                "node {id}"
            );
        }
        // And the whole-node patches of every kind.
        let whole = SettingsPatch::copied_from(&resolved_of(to));
        let mut onto = DevelopSettings::default();
        onto.apply_patch(&whole.pasted_onto(&onto.clone()), None)
            .unwrap();
        for id in 1..=6 {
            assert_eq!(
                onto.nodes.masks_of(id).unwrap().combine,
                to.nodes.masks_of(id).unwrap().combine,
                "pasted node {id}"
            );
        }
    }
}

#[test]
fn a_preset_saves_and_applies_the_mode() {
    let int = read(every_kind(Some("intersect"))).unwrap();
    let preset = Preset::from_settings("Overlap", &int, &[PresetGroup::Nodes], None).unwrap();
    for id in 2..=6 {
        assert_eq!(
            preset.settings.nodes.masks_of(id).unwrap().combine,
            MaskCombine::Intersect,
            "node {id}"
        );
    }
    let applied = preset
        .applied_to(&DevelopSettings::default(), true)
        .unwrap();
    for id in 2..=6 {
        assert_eq!(
            applied.nodes.masks_of(id).unwrap().combine,
            MaskCombine::Intersect
        );
    }
    // The file reads back with the mode.
    let back = Preset::from_json(&preset.to_json().unwrap()).unwrap();
    assert_eq!(back.settings, preset.settings);
    // A preset of Add nodes writes no field: its bytes are those of a preset from before.
    let add = read(every_kind(None)).unwrap();
    let p = Preset::from_settings("Plain", &add, &[PresetGroup::Nodes], None).unwrap();
    assert!(!p.to_json().unwrap().contains("combine"));
}

#[test]
fn an_older_build_refuses_the_field_like_any_unknown_one() {
    // `MaskSettings` denies unknown fields: this is the text an older build gives, produced here by the field's neighbour.
    let mut doc = every_kind(None);
    doc["nodes"]["items"][0]["masks"]["nope"] = json!("x");
    let err = read(doc).unwrap_err().to_string();
    assert!(
        err.starts_with("The settings can't be read: unknown field `nope`, expected "),
        "{err}"
    );
    assert!(
        err.contains("`combine`") && err.contains("`invert`") && err.contains("`items`"),
        "{err}"
    );
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

#[test]
fn the_schema_validates_both_forms_and_refuses_a_bad_value() {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("crates/ym-core/schema/settings.schema.json"))
            .unwrap(),
    )
    .unwrap();
    let v = jsonschema::validator_for(&schema).expect("a schema");
    for combine in [None, Some("intersect")] {
        let s = read(every_kind(combine)).unwrap();
        let written = serde_json::to_value(&s).unwrap();
        let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
        assert!(problems.is_empty(), "{problems:?}");
    }
    let mut written = serde_json::to_value(read(every_kind(None)).unwrap()).unwrap();
    written["nodes"]["items"][0]["masks"]["combine"] = json!("x");
    assert!(v.iter_errors(&written).next().is_some());
}
