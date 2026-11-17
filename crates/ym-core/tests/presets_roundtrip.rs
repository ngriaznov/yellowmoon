//! A saved preset always reads back (`docs/30` section 5 and 17): for settings that cover what a real session holds (an imported
//! look, Face Refinement with Protect strokes, Film, mixers, correctors with every mask kind, Auto fields, a crop, an
//! orientation, a picked white), every subset of groups written by `to_json` is read by `from_json` to the same preset, and
//! the store lists no damaged file after a save.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::time::Duration;

use common::rich::{resolved_of, rich};
use serde_json::{Value, json};
use ym_core::presets::{Preset, PresetGroup, PresetStore};
use ym_core::settings::DevelopSettings;
use ym_looks::library::LookId;

fn link(from: Value, fp: &str, to: Value, tp: &str) -> Value {
    json!({ "from": { "node": from, "port": fp }, "to": { "node": to, "port": tp } })
}

/// A graph with every node kind: Main, a corrector with masks (2), Face Refinement (3), Film (4), Splitter (5), Combiner (6),
/// Layer (7) and Parallel (8).
fn every_kind(look: &str) -> DevelopSettings {
    let masks = serde_json::to_value(common::rich::every_mask()).unwrap();
    let doc = json!({ "nodes": {
        "items": [
            { "id": "input", "crop": { "rect": { "x": 0.1, "y": 0.1, "width": 0.8, "height": 0.7 }, "angle_deg": 1.5,
                "aspect": "original", "auto": false }, "orientation": 90, "white_pick": [0.3, 0.6] },
            { "id": "output", "look": look, "look_intensity": 70 },
            { "id": 1, "kind": "corrector", "masks": masks },
            { "id": 2, "kind": "corrector", "name": "Sky", "pos": [1, 0], "masks": masks,
              "trim": { "temperature": -10 } },
            { "id": 3, "kind": "face_refinement", "name": "Portraits", "pos": [2, 0], "same_for_all": false,
              "all": { "blemish": 10, "texture": -40, "teeth_whitening": 80 },
              "faces": [ { "blemish": 90 }, { "texture": 100 } ],
              "protect": { "strokes": [ { "size": 0.02, "hardness": 0.7, "erase": true,
                  "points": [[0.1, 0.2], [0.3, 0.4]] } ], "feather": 0.1, "grow": -0.2, "refine": 0.3 },
              "masks": { "invert": true, "items": [ { "kind": "radial" } ] } },
            { "id": 4, "kind": "film", "name": "Stock", "pos": [3, 0] },
            { "id": 5, "kind": "splitter", "pos": [4, 0] },
            { "id": 6, "kind": "combiner", "pos": [5, 0] },
            { "id": 7, "kind": "layer", "mode": "multiply", "opacity": 0.4, "pos": [6, 0] },
            { "id": 8, "kind": "parallel", "pos": [7, 0] }
        ],
        "links": [
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "picture"),
            link(json!(2), "picture", json!(3), "picture"),
            link(json!(3), "picture", json!(4), "picture"),
            link(json!(4), "picture", json!(5), "picture"),
            link(json!(5), "red", json!(6), "red"),
            link(json!(5), "green", json!(6), "green"),
            link(json!(5), "blue", json!(6), "blue"),
            link(json!(4), "picture", json!(7), "picture.1"),
            link(json!(6), "picture", json!(7), "picture.2"),
            link(json!(7), "picture", json!(8), "picture.1"),
            link(json!(8), "picture", json!("output"), "picture")
        ] } });
    DevelopSettings::from_value(doc, &LookId::neutral())
        .expect("a graph with every kind")
        .0
}

/// A Face Refinement node with per-face settings ("Same for all faces" off) and a Protect stroke, nothing else.
fn face_only() -> DevelopSettings {
    let doc = json!({ "nodes": {
        "items": [ { "id": 2, "kind": "face_refinement", "same_for_all": false,
            "faces": [ { "blemish": 90 }, { "texture": 100, "teeth_whitening": 5 } ],
            "protect": { "strokes": [ { "size": 0.02, "points": [[0.1, 0.2]] } ] } } ],
        "links": [
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "picture"),
            link(json!(2), "picture", json!("output"), "picture") ] } });
    DevelopSettings::from_value(doc, &LookId::neutral())
        .expect("a face graph")
        .0
}

fn cases() -> Vec<(&'static str, DevelopSettings)> {
    let mut out = vec![
        ("rich", rich()),
        ("every kind", every_kind("builtin:signature")),
        ("every kind, imported look", every_kind("user:warm-film-3")),
        (
            "every kind, imported-package id",
            every_kind("arri:3000:film-warm"),
        ),
        (
            "every kind, odd slug",
            every_kind(LookId::user("Warm Film – (v2) é").as_str()),
        ),
        ("face only", face_only()),
    ];
    let mut auto = DevelopSettings::default();
    auto.input_mut().exposure = ym_core::settings::AutoOr::Auto;
    out.push(("auto", auto));
    out.push(("default", DevelopSettings::default()));
    out
}

fn subsets() -> Vec<Vec<PresetGroup>> {
    let all = PresetGroup::ALL;
    (1u32..(1 << all.len()))
        .map(|bits| {
            all.iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, g)| *g)
                .collect()
        })
        .collect()
}

fn check(case: &str, p: &Preset) {
    let text = p.to_json().expect("writes");
    let back = Preset::from_json(&text).unwrap_or_else(|e| {
        panic!(
            "{case} {:?}: written but not read back: {e}\n{text}",
            p.groups
        )
    });
    assert_eq!(&back, p, "{case} {:?}", p.groups);
}

#[test]
fn every_subset_of_groups_of_every_case_reads_back_from_settings() {
    for (case, s) in cases() {
        for groups in subsets() {
            let p = Preset::from_settings("Mine", &s, &groups, Some("Warm film"))
                .unwrap_or_else(|e| panic!("{case} {groups:?}: {e}"));
            check(case, &p);
        }
    }
}

#[test]
fn every_subset_of_groups_of_every_case_reads_back_from_a_photo() {
    for (case, s) in cases() {
        let photo = resolved_of(&s);
        for groups in subsets() {
            let p = Preset::from_photo("Mine", &photo, &groups, Some("Warm film"))
                .unwrap_or_else(|e| panic!("{case} {groups:?}: {e}"));
            check(case, &p);
        }
    }
}

#[test]
fn a_saved_preset_is_never_listed_as_damaged() {
    for (case, s) in cases() {
        let root = tempfile::tempdir().unwrap();
        let store = PresetStore::open(root.path()).with_lock_timeout(Duration::from_millis(300));
        let photo = resolved_of(&s);
        let p = Preset::from_photo("Mine", &photo, &PresetGroup::ALL, Some("Warm film")).unwrap();
        store
            .save(&p, false)
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        let listing = store.list();
        assert!(listing.damaged.is_empty(), "{case}: {:?}", listing.damaged);
        assert_eq!(listing.presets.len(), 1, "{case}");
    }
}

#[test]
fn a_preset_that_would_not_read_back_is_refused_and_nothing_is_written() {
    let root = tempfile::tempdir().unwrap();
    let store = PresetStore::open(root.path()).with_lock_timeout(Duration::from_millis(300));
    let mut p = Preset::from_settings("Mine", &rich(), &[PresetGroup::Look], Some("Warm")).unwrap();
    // A look group without its look is written without a `look` object, which the reader refuses.
    p.look = None;
    let err = store.save(&p, false).expect_err("refused");
    assert!(
        matches!(err, ym_core::presets::PresetError::WouldNotReadBack(_)),
        "{err:?}"
    );
    let text = err.to_string();
    assert!(text.contains("Nothing was saved."), "{text}");
    assert!(!text.contains("isn't a Yellowmoon preset"), "{text}");
    assert!(store.list().presets.is_empty() && store.list().damaged.is_empty());
    assert!(
        !root.path().join("presets").exists()
            || std::fs::read_dir(root.path().join("presets"))
                .unwrap()
                .all(|e| e.unwrap().file_name().to_string_lossy().starts_with('.'))
    );
}

#[test]
fn a_preset_holds_no_protect_strokes_and_no_clicks_in_a_face_node_and_the_rest_is_kept() {
    let s = every_kind("builtin:signature");
    let mut face_masks = serde_json::to_value(common::rich::every_mask()).unwrap();
    face_masks["items"].as_array_mut().unwrap().truncate(9);
    let mut doc = serde_json::to_value(&s).unwrap();
    for item in doc["nodes"]["items"].as_array_mut().unwrap() {
        if item["kind"] == "face_refinement" {
            item["masks"] = face_masks.clone();
        }
    }
    let s = DevelopSettings::from_value(doc, &LookId::neutral())
        .unwrap()
        .0;
    let before = s.nodes.faces.values().next().expect("a face node").clone();
    assert!(!before.protect.strokes.is_empty());
    let kinds = |m: &ym_core::settings::MaskSettings| m.items.len();
    for p in [
        Preset::from_settings("Mine", &s, &PresetGroup::ALL, None).unwrap(),
        Preset::from_photo("Mine", &resolved_of(&s), &PresetGroup::ALL, None).unwrap(),
    ] {
        let back = Preset::from_json(&p.to_json().unwrap()).unwrap();
        let face = back
            .settings
            .nodes
            .faces
            .values()
            .next()
            .expect("a face node");
        assert!(face.protect.strokes.is_empty());
        assert_eq!(
            (face.protect.feather, face.protect.grow, face.protect.refine),
            (
                before.protect.feather,
                before.protect.grow,
                before.protect.refine
            )
        );
        assert_eq!(
            kinds(&face.masks),
            kinds(&before.masks),
            "the rows are there"
        );
        let json = serde_json::to_string(&face.masks).unwrap();
        assert!(!json.contains("\"strokes\":[{"), "no Brush stroke: {json}");
        assert!(
            !json.contains("\"points\":[["),
            "no Object click or stroke point: {json}"
        );
        let mut a = face.clone();
        let mut b = before.clone();
        for n in [&mut a, &mut b] {
            n.protect.strokes.clear();
            n.masks = Default::default();
        }
        assert_eq!(a, b, "the rest of the node is equal");
        assert_eq!(face.same_for_all, before.same_for_all);
        assert_eq!(face.faces, before.faces);
    }
}
