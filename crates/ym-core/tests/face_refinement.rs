//! The Face Refinement node's settings (`docs/27-skin-retouch.md` sections 3 and 10, task S-11a, and `docs/23` section 3):
//! the format and its defaults, the unknown-field rule, the ranges, the face-number rule, the ports, patches, Reset, undo,
//! Copy and Paste, the written form, the JSON schema, the conversion to `ym_retouch::SkinSettings` and the text of the
//! kind. The render (a pass-through until S-11b) is tested in `render_nodes.rs`.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::session::PhotoId;
use ym_core::settings::{
    DevelopSettings, FACE_AMOUNT_RANGE, FACE_BIPOLAR_RANGE, FaceRefinementNode, FaceSettings,
    FieldRef, GraphError, MAX_FACES, NodeId, NodeKind, NodePatch, Port, ResolvedSettings,
    SettingsError, SettingsField, SettingsPatch, ports_in, ports_out,
};
use ym_looks::library::LookId;

fn id(n: u32) -> NodeId {
    NodeId::Corrector(n)
}

fn link(from: Value, fp: &str, to: Value, tp: &str) -> Value {
    json!({ "from": { "node": from, "port": fp }, "to": { "node": to, "port": tp } })
}

fn line(ids: &[u32]) -> Vec<Value> {
    let mut links = Vec::new();
    let mut from = json!("input");
    for n in ids {
        links.push(link(from, "picture", json!(n), "picture"));
        from = json!(n);
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

/// A photo with a Face Refinement node 2 after Main, with `body` merged into the item.
fn with_face(body: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    let mut node = json!({ "id": 2, "kind": "face_refinement" });
    node.as_object_mut()
        .unwrap()
        .extend(body.as_object().unwrap().clone());
    read(doc(vec![node], line(&[1, 2])))
}

fn face_of(s: &DevelopSettings) -> &FaceRefinementNode {
    s.nodes.face_refinement(2).expect("Face Refinement node 2")
}

fn base() -> DevelopSettings {
    with_face(json!({})).unwrap().0
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

fn text_of<T>(r: Result<T, SettingsError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

const DEFAULTS: FaceSettings = FaceSettings {
    blemish: 30.0,
    dark_circles: 30.0,
    even_skin: 25.0,
    texture: 0.0,
    skin_contrast: 0.0,
    shadow_desaturation: 0.0,
    shine_removal: 0.0,
    teeth_whitening: 0.0,
};

#[test]
fn a_minimal_item_is_a_valid_node_with_the_approved_defaults() {
    let (s, clamped) = with_face(json!({})).unwrap();
    assert!(clamped.is_empty());
    let f = face_of(&s);
    assert_eq!(f.name, "Face Refinement");
    assert!(!f.bypass);
    assert!(f.same_for_all, "Same for all faces is on by default");
    assert_eq!(f.all, DEFAULTS);
    assert_eq!(FaceSettings::default(), DEFAULTS);
    assert!(f.faces.is_empty());
    assert!(f.protect.strokes.is_empty());
    assert_eq!(
        (f.protect.feather, f.protect.grow, f.protect.refine),
        (0.0, 0.0, 0.0)
    );
    assert!(f.masks.items.is_empty() && !f.masks.invert);
    assert_eq!(s.nodes.kind_of(id(2)), Some(NodeKind::FaceRefinement));
    assert!(s.nodes.corrector(2).is_none() && s.nodes.film(2).is_none());
    // The node counts as a numbered node: Main and node 2.
    assert_eq!(s.nodes.node_count(), 2);
    assert_eq!(f.faces, FaceRefinementNode::default().faces);
}

#[test]
fn the_written_form_is_the_documented_one_and_a_full_item_round_trips() {
    let (s, _) = with_face(json!({})).unwrap();
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
            "all",
            "bypass",
            "faces",
            "id",
            "kind",
            "masks",
            "name",
            "pos",
            "protect",
            "same_for_all"
        ]
    );
    assert_eq!(item["kind"], "face_refinement");
    assert_eq!(
        item["all"],
        json!({ "blemish": 30.0, "dark_circles": 30.0, "even_skin": 25.0, "texture": 0.0,
            "skin_contrast": 0.0, "shadow_desaturation": 0.0, "shine_removal": 0.0, "teeth_whitening": 0.0 })
    );
    assert_eq!(item["faces"], json!([]));
    assert_eq!(
        item["protect"],
        json!({ "strokes": [], "feather": 0.0, "grow": 0.0, "refine": 0.0 })
    );
    // A full item (every field set) reads, writes back the same and reads again to the same settings.
    let full = json!({
        "name": "Portraits", "bypass": true, "pos": [3, 1], "same_for_all": false,
        "all": { "blemish": 10, "dark_circles": 20, "even_skin": 30, "texture": -40,
                 "skin_contrast": 50, "shadow_desaturation": 60, "shine_removal": 70, "teeth_whitening": 80 },
        "faces": [ { "blemish": 90 }, { "texture": 100, "teeth_whitening": 5 } ],
        "protect": { "strokes": [ { "size": 0.02, "hardness": 0.7, "erase": true, "points": [[0.1, 0.2], [0.3, 0.4]] } ],
                     "feather": 0.1, "grow": -0.2, "refine": 0.3 },
        "masks": { "invert": true, "items": [ { "kind": "radial" } ] },
    });
    let (s, clamped) = with_face(full).unwrap();
    assert!(clamped.is_empty());
    let f = face_of(&s);
    assert_eq!(
        (f.name.as_str(), f.bypass, f.pos),
        ("Portraits", true, [3.0, 1.0])
    );
    assert!(!f.same_for_all);
    assert_eq!(f.all.teeth_whitening, 80.0);
    assert_eq!(f.faces.len(), 2);
    // A face entry that leaves an amount out takes its default (30, 30, 25 ...).
    assert_eq!(f.faces[0].blemish, 90.0);
    assert_eq!(f.faces[0].dark_circles, 30.0);
    assert_eq!(f.faces[0].even_skin, 25.0);
    assert_eq!(f.faces[1].texture, 100.0);
    assert_eq!(f.protect.strokes.len(), 1);
    assert!(f.protect.strokes[0].erase);
    assert_eq!(f.protect.grow, -0.2);
    let written = serde_json::to_value(&s).unwrap();
    let (again, clamped) = read(written.clone()).unwrap();
    assert!(clamped.is_empty());
    assert_eq!(again, s);
    assert_eq!(serde_json::to_value(&again).unwrap(), written);
}

#[test]
fn an_unknown_field_is_refused_and_other_kinds_fields_are_not_a_face_refinement_nodes() {
    // A field of another kind: the graph's sentence.
    for (field, body) in [
        ("trim", json!({ "trim": { "tint": 3 } })),
        ("grade", json!({ "grade": {} })),
        ("film", json!({ "film": {} })),
        ("strength", json!({ "strength": 50 })),
        ("mode", json!({ "mode": "add" })),
    ] {
        assert_eq!(
            text_of(with_face(body)),
            format!("A face refinement node has no field called “{field}”.")
        );
    }
    // And a face refinement node's field on a corrector.
    for (field, body) in [
        ("faces", json!({ "faces": [] })),
        ("same_for_all", json!({ "same_for_all": true })),
        ("protect", json!({ "protect": {} })),
    ] {
        let mut item = json!({ "id": 2, "kind": "corrector" });
        item.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        assert_eq!(
            text_of(read(doc(vec![item], line(&[1, 2])))),
            format!("A corrector has no field called “{field}”.")
        );
    }
    // A typo inside a nested object, in a face entry or in the protection plane is serde's refusal.
    for body in [
        json!({ "all": { "blemsh": 3 } }),
        json!({ "faces": [ { "blur": 3 } ] }),
        json!({ "protect": { "stroke": [] } }),
        json!({ "protect": { "strokes": [ { "sise": 1 } ] } }),
        json!({ "same_for_all": "yes" }),
        json!({ "all": { "blemish": "a lot" } }),
    ] {
        assert!(
            matches!(
                with_face(body.clone()).unwrap_err(),
                SettingsError::Parse(_)
            ),
            "{body}"
        );
    }
    // The node needs its kind, like any new node, and Main is a corrector.
    assert_eq!(
        text_of(read(doc(
            vec![json!({ "id": 2, "all": {} })],
            line(&[1, 2])
        ))),
        GraphError::NeedsKind("2".into()).to_string()
    );
    assert!(
        read(doc(
            vec![json!({ "id": 1, "kind": "face_refinement" })],
            line(&[1])
        ))
        .is_err()
    );
}

#[test]
fn every_amount_is_clamped_and_reported() {
    let (s, clamped) = with_face(json!({
        "all": { "blemish": 150, "dark_circles": -5, "even_skin": 101, "texture": -300,
                 "skin_contrast": 120, "shadow_desaturation": -1, "shine_removal": 1000, "teeth_whitening": 100.5 },
        "faces": [ { "blemish": 99, "texture": 500 }, { "teeth_whitening": -9 } ],
        "protect": { "feather": 4, "grow": -4, "refine": 9, "strokes": [ { "size": 9, "hardness": 3, "points": [[2, -1]] } ] },
    }))
    .unwrap();
    let f = face_of(&s);
    assert_eq!(
        f.all,
        FaceSettings {
            blemish: 100.0,
            dark_circles: 0.0,
            even_skin: 100.0,
            texture: -100.0,
            skin_contrast: 100.0,
            shadow_desaturation: 0.0,
            shine_removal: 100.0,
            teeth_whitening: 100.0,
        }
    );
    assert_eq!((f.faces[0].blemish, f.faces[0].texture), (99.0, 100.0));
    assert_eq!(f.faces[1].teeth_whitening, 0.0);
    assert_eq!(
        (f.protect.feather, f.protect.grow, f.protect.refine),
        (1.0, -1.0, 1.0)
    );
    assert_eq!(f.protect.strokes[0].size, 0.25);
    assert_eq!(f.protect.strokes[0].hardness, 1.0);
    assert_eq!(f.protect.strokes[0].points[0], [1.0, 0.0]);
    for field in [
        SettingsField::AllFaces,
        SettingsField::Faces,
        SettingsField::Protect,
    ] {
        assert!(
            clamped.contains(&FieldRef { node: id(2), field }),
            "{field:?} is reported"
        );
    }
    // The ranges are the ones the spec gives.
    assert_eq!(FACE_AMOUNT_RANGE, (0.0, 100.0));
    assert_eq!(FACE_BIPOLAR_RANGE, (-100.0, 100.0));
    // In range: nothing is reported, and the end points are kept.
    let (s, clamped) = with_face(json!({
        "all": { "blemish": 0, "texture": -100, "skin_contrast": 100, "teeth_whitening": 100 }
    }))
    .unwrap();
    assert!(clamped.is_empty());
    assert_eq!(face_of(&s).all.texture, -100.0);
    // A name is trimmed, cut at 32 characters and an empty one reads as the default.
    let (s, _) = with_face(json!({ "name": "   " })).unwrap();
    assert_eq!(face_of(&s).name, "Face Refinement");
    let (s, _) =
        with_face(json!({ "name": "  A rather long name for a face refinement node  " })).unwrap();
    assert_eq!(face_of(&s).name.chars().count(), 32);
    // A number that is not a number is refused and nothing is applied.
    let mut s = base();
    let before = s.clone();
    let mut np = NodePatch::default();
    np.all = Some(ym_core::settings::FaceSettingsPatch {
        blemish: Some(f64::NAN),
        ..Default::default()
    });
    let e = s
        .apply_patch(&SettingsPatch::node(id(2), np), None)
        .unwrap_err();
    assert!(matches!(e, SettingsError::NotFinite(_)), "{e}");
    assert_eq!(s, before);
}

#[test]
fn more_than_32_faces_are_refused() {
    let faces = |n: usize| json!({ "faces": vec![json!({}); n] });
    assert_eq!(MAX_FACES, 32);
    assert!(with_face(faces(32)).is_ok());
    assert_eq!(
        text_of(with_face(faces(33))),
        "A Face Refinement node can hold settings for up to 32 faces."
    );
}

#[test]
fn faces_are_numbered_from_1_and_a_face_past_the_list_takes_face_1s_settings() {
    let (s, _) = with_face(json!({
        "same_for_all": false,
        "all": { "blemish": 11 },
        "faces": [ { "blemish": 40 }, { "blemish": 50 }, { "blemish": 60 } ],
    }))
    .unwrap();
    let f = face_of(&s);
    // Per-face settings follow the face numbers.
    assert_eq!(f.settings_for_face(1).blemish, 40.0);
    assert_eq!(f.settings_for_face(2).blemish, 50.0);
    assert_eq!(f.settings_for_face(3).blemish, 60.0);
    // Faces beyond them take face 1's settings; `all` is not used.
    assert_eq!(f.settings_for_face(4), f.settings_for_face(1));
    assert_eq!(f.settings_for_face(32), f.settings_for_face(1));
    // Face 0 does not exist; it reads as face 1.
    assert_eq!(f.settings_for_face(0), f.settings_for_face(1));
    // Same for all: `all` serves every face, the list is kept.
    let mut same = f.clone();
    same.same_for_all = true;
    for n in [1, 2, 3, 9] {
        assert_eq!(same.settings_for_face(n).blemish, 11.0);
    }
    assert_eq!(same.faces.len(), 3);
    // Per face with an empty list: `all`.
    let mut empty = f.clone();
    empty.faces.clear();
    assert_eq!(empty.settings_for_face(1).blemish, 11.0);
    assert_eq!(empty.settings_for_face(5).blemish, 11.0);
}

#[test]
fn the_settings_convert_to_the_retouch_libraries_in_f32() {
    let (s, _) = with_face(json!({
        "same_for_all": false,
        "faces": [ { "blemish": 10, "dark_circles": 20, "even_skin": 30, "texture": -40,
                     "skin_contrast": 50, "shadow_desaturation": 60, "shine_removal": 70, "teeth_whitening": 80 } ],
    }))
    .unwrap();
    let k = face_of(&s).skin_settings_for_face(3);
    assert_eq!(
        (
            k.blemish,
            k.dark_circles,
            k.even_skin,
            k.texture,
            k.skin_contrast,
            k.shadow_desaturation,
            k.shine_removal,
            k.teeth_whitening
        ),
        (10.0, 20.0, 30.0, -40.0, 50.0, 60.0, 70.0, 80.0)
    );
    // The defaults are the approved starting point, not the library's all-zero default.
    let d = FaceSettings::default().to_skin_settings();
    assert_eq!(
        (d.blemish, d.dark_circles, d.even_skin, d.texture),
        (30.0, 30.0, 25.0, 0.0)
    );
    assert!(!d.grading_active());
    assert_eq!(d.clamped(), d, "the defaults are in range for the library");
}

#[test]
fn a_node_is_inert_when_bypassed_or_every_amount_is_zero() {
    let mut n = FaceRefinementNode::default();
    assert!(!n.is_inert());
    n.bypass = true;
    assert!(n.is_inert());
    n.bypass = false;
    n.all = FaceSettings {
        blemish: 0.0,
        dark_circles: 0.0,
        even_skin: 0.0,
        ..FaceSettings::default()
    };
    assert!(n.is_inert());
    // Per face: a node is inert only if every face that can be used is.
    n.same_for_all = false;
    n.faces = vec![n.all, FaceSettings::default()];
    assert!(!n.is_inert());
    n.faces = vec![n.all, n.all];
    assert!(n.is_inert());
}

#[test]
fn the_ports_and_links_are_a_correctors_and_ids_are_shared() {
    assert_eq!(
        ports_in(NodeKind::FaceRefinement),
        &[Port::Picture, Port::Key]
    );
    assert_eq!(
        ports_out(NodeKind::FaceRefinement),
        &[Port::Picture, Port::Key]
    );
    assert_eq!(ports_in(NodeKind::FaceRefinement), ports_in(NodeKind::Film));
    // Key output to a corrector's key input is a link a corrector could have.
    let items = vec![
        json!({ "id": 2, "kind": "face_refinement" }),
        json!({ "id": 3, "kind": "corrector" }),
    ];
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "key", json!(3), "key"));
    assert!(read(doc(items.clone(), links)).is_ok());
    // A key into the Face Refinement node's key input.
    let mut links = line(&[1, 3, 2]);
    links.push(link(json!(3), "key", json!(2), "key"));
    assert!(read(doc(items.clone(), links)).is_ok());
    // A picture to a key is refused as for correctors.
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "picture", json!(3), "key"));
    assert_eq!(
        text_of(read(doc(items.clone(), links))),
        GraphError::LinkPortTypes.to_string()
    );
    // A mixer's port is not its.
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(1), "picture", json!(2), "picture.1"));
    assert_eq!(
        text_of(read(doc(items.clone(), links))),
        "Node 2 has no port called “picture.1”."
    );
    // Two nodes with one id are refused, whatever their kinds.
    for other in ["corrector", "film", "layer", "face_refinement"] {
        let e = read(doc(
            vec![
                json!({ "id": 2, "kind": "face_refinement" }),
                json!({ "id": 2, "kind": other }),
            ],
            line(&[1, 2]),
        ))
        .unwrap_err();
        assert_eq!(e.to_string(), "Two nodes have the id 2.", "{other}");
    }
    // The 64-node limit counts these nodes.
    let items: Vec<Value> = (2..=65)
        .map(|i| json!({ "id": i, "kind": "face_refinement" }))
        .collect();
    assert_eq!(
        text_of(read(doc(items, line(&[1])))),
        "A photo can have up to 64 nodes."
    );
    // The node's own Mask tab: rows are kept and a bypassed node keeps them.
    let (s, _) = with_face(json!({ "bypass": true, "masks": { "items": [ { "kind": "radial" }, { "kind": "linear" } ] } })).unwrap();
    assert_eq!(s.nodes.masks_of(2).map(|m| m.items.len()), Some(2));
    assert_eq!(s.nodes.all_masks().count(), 2);
}

#[test]
fn the_kind_has_its_names_and_a_field_belongs_to_the_kinds_that_have_it() {
    let k = NodeKind::FaceRefinement;
    assert_eq!(k.name(), "face_refinement");
    assert_eq!(NodeKind::parse("face_refinement").unwrap(), k);
    assert_eq!(k.with_article(), "a face refinement node");
    assert_eq!(k.sentence(), "A face refinement node");
    assert!(k.is_numbered_extra() && !k.is_mixer() && !k.is_mixer_family());
    assert!(NodeKind::ALL.contains(&k));
    assert_eq!(serde_json::to_value(k).unwrap(), json!("face_refinement"));
    let own: Vec<SettingsField> = SettingsField::ALL
        .into_iter()
        .filter(|f| f.belongs_to(k))
        .collect();
    assert_eq!(
        own,
        [
            SettingsField::Name,
            SettingsField::Bypass,
            SettingsField::Pos,
            SettingsField::Masks,
            SettingsField::SameForAll,
            SettingsField::AllFaces,
            SettingsField::Faces,
            SettingsField::Protect,
        ]
    );
    assert_eq!(SettingsField::AllFaces.name(), "all");
    assert_eq!(
        serde_json::to_value(SettingsField::AllFaces).unwrap(),
        json!("all")
    );
    for f in [
        SettingsField::SameForAll,
        SettingsField::AllFaces,
        SettingsField::Faces,
        SettingsField::Protect,
    ] {
        // Only this kind has them; every field has a serde name equal to its name.
        for other in NodeKind::ALL {
            assert_eq!(f.belongs_to(other), other == k, "{f:?} {other:?}");
        }
        assert_eq!(serde_json::to_value(f).unwrap(), json!(f.name()));
    }
    // The first group of render stages a change affects is the nodes' (the grade group).
    assert_eq!(
        SettingsField::Faces.group(),
        SettingsField::Strength.group()
    );
}

#[test]
fn patches_change_fields_report_them_and_a_partial_keeps_the_rest() {
    let mut s = base();
    // `all` merges amount by amount.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "all": { "blemish": 70, "texture": -20 } } } })),
            None,
        )
        .unwrap();
    assert_eq!(
        report.changed,
        vec![FieldRef {
            node: id(2),
            field: SettingsField::AllFaces
        }]
    );
    let f = face_of(&s);
    assert_eq!(
        (f.all.blemish, f.all.texture, f.all.dark_circles),
        (70.0, -20.0, 30.0)
    );
    // The switch, the per-face list (replaced whole), the name and Bypass.
    let report = s
        .apply_patch(
            &patch_of(
                json!({ "nodes": { "2": { "same_for_all": false, "name": "Faces", "bypass": true,
                "faces": [ { "blemish": 5 }, { "blemish": 6 } ] } } }),
            ),
            None,
        )
        .unwrap();
    for field in [
        SettingsField::SameForAll,
        SettingsField::Name,
        SettingsField::Bypass,
        SettingsField::Faces,
    ] {
        assert!(
            report.changed.contains(&FieldRef { node: id(2), field }),
            "{field:?}"
        );
    }
    assert!(!report.changed.contains(&FieldRef {
        node: id(2),
        field: SettingsField::AllFaces
    }));
    let f = face_of(&s);
    assert!(!f.same_for_all && f.bypass && f.name == "Faces");
    assert_eq!(f.faces.len(), 2);
    assert_eq!(f.settings_for_face(2).blemish, 6.0);
    // The same patch again changes nothing.
    let again = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "same_for_all": false, "faces": [ { "blemish": 5 }, { "blemish": 6 } ] } } })),
            None,
        )
        .unwrap();
    assert!(again.is_empty(), "{again:?}");
    // The protection: strokes appended, then kept to the first n, then replaced; edge amounts alone.
    let stroke = |x: f64| json!({ "size": 0.02, "points": [[x, 0.5]] });
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "protect": { "strokes_append": [stroke(0.1)] } } } })),
        None,
    )
    .unwrap();
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "protect": { "strokes_append": [stroke(0.2), stroke(0.3)] } } } })),
        None,
    )
    .unwrap();
    assert_eq!(face_of(&s).protect.strokes.len(), 3);
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "protect": { "strokes_keep": 1 } } } })),
        None,
    )
    .unwrap();
    assert_eq!(face_of(&s).protect.strokes.len(), 1);
    assert_eq!(face_of(&s).protect.strokes[0].points[0], [0.1, 0.5]);
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "protect": { "feather": 0.4 } } } })),
            None,
        )
        .unwrap();
    assert_eq!(
        report.changed,
        vec![FieldRef {
            node: id(2),
            field: SettingsField::Protect
        }]
    );
    assert_eq!(face_of(&s).protect.strokes.len(), 1);
    assert_eq!(face_of(&s).protect.feather, 0.4);
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "protect": { "strokes": [] } } } })),
        None,
    )
    .unwrap();
    assert!(face_of(&s).protect.strokes.is_empty());
    // Out of range numbers are clamped and reported.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "all": { "even_skin": 400 }, "protect": { "grow": 3 } } } })),
            None,
        )
        .unwrap();
    assert_eq!(face_of(&s).all.even_skin, 100.0);
    assert_eq!(face_of(&s).protect.grow, 1.0);
    assert!(report.clamped.contains(&FieldRef {
        node: id(2),
        field: SettingsField::AllFaces
    }));
    assert!(report.clamped.contains(&FieldRef {
        node: id(2),
        field: SettingsField::Protect
    }));
    // The masks of the node (its Mask tab) are patched like a corrector's.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "masks": { "items": [ { "kind": "linear" } ] } } } })),
        None,
    )
    .unwrap();
    assert_eq!(face_of(&s).masks.items.len(), 1);
    // 33 faces in a patch are refused whole.
    let before = s.clone();
    let too_many = json!({ "nodes": { "2": { "faces": vec![json!({}); 33] } } });
    assert_eq!(
        text_of(s.apply_patch(&patch_of(too_many), None)),
        "A Face Refinement node can hold settings for up to 32 faces."
    );
    assert_eq!(s, before);
}

#[test]
fn a_node_patch_adds_the_node_removes_it_and_refuses_the_wrong_kind() {
    let mut s = DevelopSettings::default();
    let add = patch_of(json!({
        "nodes": { "5": { "kind": "face_refinement", "all": { "blemish": 80 } } },
        "links": [
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(5), "picture"),
            link(json!(5), "picture", json!("output"), "picture")
        ]
    }));
    let report = s.apply_patch(&add, None).unwrap();
    assert_eq!(report.added, vec![id(5)]);
    assert!(s.nodes.face_refinement(5).is_some() && s.nodes.corrector(5).is_none());
    assert_eq!(s.nodes.kind_of(id(5)), Some(NodeKind::FaceRefinement));
    let f = s.nodes.face_refinement(5).unwrap();
    // The new node has the defaults but for what the patch says, and the name of the kind.
    assert_eq!(f.name, "Face Refinement");
    assert!(f.same_for_all);
    assert_eq!(f.all.blemish, 80.0);
    assert_eq!(f.all.dark_circles, 30.0);
    // Every field of the new node is reported as changed.
    for field in [
        SettingsField::SameForAll,
        SettingsField::AllFaces,
        SettingsField::Faces,
        SettingsField::Protect,
        SettingsField::Masks,
    ] {
        assert!(
            report.changed.contains(&FieldRef { node: id(5), field }),
            "{field:?}"
        );
    }
    assert!(!report.changed.contains(&FieldRef {
        node: id(5),
        field: SettingsField::Trim
    }));
    // A kind other than the node's is refused, and so is a field of another kind.
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "5": { "kind": "corrector" } } })),
            None
        )),
        "Node 5 is a face refinement node, not a corrector."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "5": { "trim": { "tint": 1 } } } })),
            None
        )),
        "A face refinement node has no field called “trim”."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "1": { "faces": [] } } })),
            None
        )),
        "A corrector has no field called “faces”."
    );
    // Without its links the Output has no picture; with them the node goes.
    assert_eq!(
        text_of(s.apply_patch(&patch_of(json!({ "remove": [5] })), None)),
        GraphError::OutputLink.to_string()
    );
    let report = s
        .apply_patch(
            &patch_of(json!({ "remove": [5], "links": [
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!("output"), "picture")] })),
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![id(5)]);
    assert!(s.nodes.face_refinement(5).is_none());
}

#[test]
fn undo_is_the_patch_between_two_settings_and_reset_puts_the_defaults_back() {
    let a = base();
    let mut b = a.clone();
    b.apply_patch(
        &patch_of(json!({ "nodes": { "2": {
            "same_for_all": false, "name": "Faces", "bypass": true,
            "all": { "blemish": 77 }, "faces": [ { "blemish": 1 }, { "teeth_whitening": 50 } ],
            "protect": { "strokes": [ { "size": 0.03, "points": [[0.2, 0.2], [0.3, 0.3]] } ], "feather": 0.5 },
            "masks": { "items": [ { "kind": "radial" } ] } } } })),
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
    assert_eq!(
        add.nodes.get(&id(2)).and_then(|n| n.kind),
        Some(NodeKind::FaceRefinement)
    );
    let mut c = empty.clone();
    c.apply_patch(&add, None).unwrap();
    assert_eq!(c.nodes.face_refinement(2), b.nodes.face_refinement(2));
    let drop = SettingsPatch::between(&b, &empty);
    let mut d = b.clone();
    d.apply_patch(&drop, None).unwrap();
    assert!(d.nodes.face_refinement(2).is_none());
    // Reset of the head: the switch, `all` and `faces` go back to the defaults (Blemish 30, Dark circles 30, Even skin 25 ...);
    // the name, Bypass, place, masks and protection stay ("Masks and protection stay").
    let reset = SettingsPatch::reset_node_in(&b, id(2), &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&reset, None).unwrap();
    let f = face_of(&r);
    assert!(f.same_for_all);
    assert_eq!(f.all, DEFAULTS);
    assert!(f.faces.is_empty());
    assert!(
        f.bypass && f.name == "Faces",
        "Reset leaves Bypass and the name alone"
    );
    assert_eq!(f.masks, face_of(&b).masks);
    assert_eq!(
        f.protect,
        face_of(&b).protect,
        "Reset leaves the protection alone"
    );
    // Reset of one field.
    let one = SettingsPatch::reset_in(&b, id(2), SettingsField::AllFaces, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(face_of(&r).all, DEFAULTS);
    assert!(!face_of(&r).same_for_all);
    assert_eq!(face_of(&r).faces.len(), 2);
    // Reset of the protection alone clears it ("Clear").
    let clear = SettingsPatch::reset_in(&b, id(2), SettingsField::Protect, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&clear, None).unwrap();
    assert!(face_of(&r).protect.strokes.is_empty());
    assert_eq!(face_of(&r).protect.feather, 0.0);
    // The diff names the fields that differ.
    let diff: Vec<SettingsField> = a.diff(&b).into_iter().map(|f| f.field).collect();
    for f in [
        SettingsField::SameForAll,
        SettingsField::AllFaces,
        SettingsField::Faces,
        SettingsField::Protect,
        SettingsField::Masks,
        SettingsField::Name,
        SettingsField::Bypass,
    ] {
        assert!(diff.contains(&f), "{f:?}");
    }
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
        &patch_of(json!({ "nodes": { "2": { "same_for_all": false, "name": "Faces",
            "faces": [ { "blemish": 12 }, { "blemish": 34 } ],
            "protect": { "strokes": [ { "size": 0.03, "points": [[0.5, 0.5]] } ] },
            "masks": { "items": [ { "kind": "radial" }, { "kind": "object", "points": [[0.5, 0.5, true]] } ] } } } })),
        None,
    )
    .unwrap();
    let resolved = resolved_of(&src);
    assert_eq!(
        resolved.settings.nodes.face_refinement(2),
        src.nodes.face_refinement(2)
    );
    let copy = SettingsPatch::copied_from(&resolved);
    let node = copy.nodes.get(&id(2)).expect("the node is in the copy");
    assert_eq!(node.kind, Some(NodeKind::FaceRefinement));
    // Per-face settings are copied by face number.
    assert_eq!(node.faces.as_ref().map(Vec::len), Some(2));
    let mut target = DevelopSettings::default();
    target
        .apply_patch(
            &patch_of(
                json!({ "nodes": { "7": { "kind": "face_refinement", "all": { "blemish": 99 } } },
            "links": [
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!(7), "picture"),
                link(json!(7), "picture", json!("output"), "picture")] }),
            ),
            None,
        )
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(paste.remove.contains(&id(7)), "the target's own node goes");
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.face_refinement(7).is_none());
    let f = target.nodes.face_refinement(2).expect("pasted");
    assert!(!f.same_for_all && f.name == "Faces");
    // A pasted list follows the face numbers; a face beyond it takes face 1's settings.
    assert_eq!(f.settings_for_face(1).blemish, 12.0);
    assert_eq!(f.settings_for_face(2).blemish, 34.0);
    assert_eq!(f.settings_for_face(5).blemish, 12.0);
    assert_eq!(f.protect, face_of(&src).protect);
    // Object clicks are not copied: the pasted row arrives empty, as for correctors.
    assert_eq!(f.masks.items.len(), 2);
    match &f.masks.items[1].shape {
        ym_core::settings::MaskShape::Object(o) => assert!(o.points.is_empty() && o.box_.is_none()),
        other => panic!("{other:?}"),
    }
    assert_eq!(target.nodes.links, src.nodes.links);
    // Pasting onto a photo with fewer faces than the list is the same settings: nothing about faces is stored per photo.
    let mut other = DevelopSettings::default();
    other.apply_patch(&copy.pasted_onto(&other), None).unwrap();
    assert_eq!(
        other.nodes.face_refinement(2),
        target.nodes.face_refinement(2)
    );
}

#[test]
fn the_settings_survive_the_xmp_text() {
    // `ym:Settings` is the settings as JSON text; the node is in it and reads back whole.
    let (s, _) = with_face(
        json!({ "same_for_all": false, "faces": [ { "blemish": 12 } ],
        "protect": { "strokes": [ { "size": 0.03, "points": [[0.5, 0.5]] } ] } }),
    )
    .unwrap();
    let text = serde_json::to_string(&s).unwrap();
    for needle in [
        r#""kind":"face_refinement""#,
        r#""same_for_all":false"#,
        r#""blemish":12.0"#,
        r#""protect":{"feather":0.0"#,
        r#""size":0.03"#,
    ] {
        assert!(text.contains(needle), "{needle} in {text}");
    }
    let back = read(serde_json::from_str(&text).unwrap()).unwrap().0;
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
    let mut s = base();
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "same_for_all": false,
            "faces": [ { "blemish": 40 }, { "texture": -10 } ],
            "protect": { "strokes": [ { "size": 0.03, "points": [[0.5, 0.5]] } ], "grow": 0.1 },
            "masks": { "items": [ { "kind": "linear" } ] } } } })),
        None,
    )
    .unwrap();
    let written = serde_json::to_value(&s).unwrap();
    let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
    assert!(problems.is_empty(), "{problems:?}");
    // The minimal item is valid.
    let bad = |body: Value| {
        let mut node = json!({ "id": 2, "kind": "face_refinement" });
        node.as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        json!({ "nodes": { "items": [node] } })
    };
    assert!(v.is_valid(&bad(json!({}))));
    // What the core refuses, the schema does too.
    for body in [
        json!({ "trim": {} }),
        json!({ "strength": 50 }),
        json!({ "all": { "blemish": 101 } }),
        json!({ "all": { "blemish": -1 } }),
        json!({ "all": { "texture": -101 } }),
        json!({ "all": { "skin_contrast": 101 } }),
        json!({ "all": { "teeth_whitening": 120 } }),
        json!({ "all": { "blur": 1 } }),
        json!({ "faces": [ { "even_skin": 101 } ] }),
        json!({ "faces": vec![json!({}); 33] }),
        json!({ "same_for_all": 1 }),
        json!({ "protect": { "strokes": [ { "size": 0.5 } ] } }),
        json!({ "protect": { "grow": 2 } }),
        json!({ "protect": { "mode": "add" } }),
    ] {
        assert!(!v.is_valid(&bad(body.clone())), "{body}");
    }
    // The ranges are the ones the core clamps to, and the defaults are the ones it fills in.
    let defs = &schema["$defs"]["face_settings"]["properties"];
    let range = |p: &Value| {
        (
            p["minimum"].as_f64().unwrap(),
            p["maximum"].as_f64().unwrap(),
        )
    };
    for name in [
        "blemish",
        "dark_circles",
        "even_skin",
        "shadow_desaturation",
        "shine_removal",
        "teeth_whitening",
    ] {
        assert_eq!(range(&defs[name]), FACE_AMOUNT_RANGE, "{name}");
    }
    for name in ["texture", "skin_contrast"] {
        assert_eq!(range(&defs[name]), FACE_BIPOLAR_RANGE, "{name}");
    }
    let d = FaceSettings::default();
    for (name, value) in [
        ("blemish", d.blemish),
        ("dark_circles", d.dark_circles),
        ("even_skin", d.even_skin),
        ("texture", d.texture),
        ("skin_contrast", d.skin_contrast),
        ("shadow_desaturation", d.shadow_desaturation),
        ("shine_removal", d.shine_removal),
        ("teeth_whitening", d.teeth_whitening),
    ] {
        assert_eq!(defs[name]["default"].as_f64(), Some(value), "{name}");
    }
    let node = &schema["$defs"]["face_refinement_node"]["properties"];
    assert_eq!(node["same_for_all"]["default"], json!(true));
    assert_eq!(node["faces"]["maxItems"], json!(MAX_FACES));
}
