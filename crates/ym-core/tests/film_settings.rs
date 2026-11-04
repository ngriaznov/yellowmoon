//! The Film node's settings (`docs/24-film.md` section 4.5, `docs/23` section 3): the format, the defaults, every
//! refusal and every clamp, patches, Reset, undo, Copy and Paste, the JSON schema and the XMP text.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::film::{catalogue, film_data_of};
use ym_core::session::PhotoId;
use ym_core::settings::{
    DevelopSettings, FieldRef, FilmNode, GraphError, NodeId, NodeKind, NodePatch, ResolvedSettings,
    SettingsError, SettingsField, SettingsPatch,
};
use ym_looks::library::LookId;

fn film_id(n: u32) -> NodeId {
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

fn with_film(body: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    let mut node = json!({ "id": 2, "kind": "film" });
    node.as_object_mut()
        .unwrap()
        .extend(body.as_object().unwrap().clone());
    read(doc(vec![node], line(&[1, 2])))
}

fn film_of(s: &DevelopSettings) -> &FilmNode {
    s.nodes.film(2).expect("film node 2")
}

fn text_of(e: SettingsError) -> String {
    e.to_string()
}

#[test]
fn a_new_film_node_is_portra_400_on_its_matching_paper_with_every_number_at_its_default() {
    let (s, clamped) = with_film(json!({})).unwrap();
    assert!(clamped.is_empty());
    let f = film_of(&s);
    assert_eq!(f.name, "Film");
    assert!(!f.bypass);
    assert_eq!(f.film.stock, "kodak_portra_400");
    assert_eq!(f.film.exposure_ev, 0.0);
    assert_eq!(f.print.paper, None);
    assert_eq!(
        (
            f.print.exposure_ev,
            f.print.lights.c,
            f.print.lights.m,
            f.print.lights.y
        ),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!((f.halation.amount, f.halation.size), (100.0, 100.0));
    assert_eq!(
        (f.grain.amount, f.grain.size, f.grain.seed),
        (100.0, 100.0, 0)
    );
    assert_eq!(f.strength, 100.0);
    assert!(f.masks.items.is_empty() && !f.masks.invert);
    // The written form is the one of docs/24 section 4.5.
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
            "bypass", "film", "grain", "halation", "id", "kind", "masks", "name", "pos", "print",
            "strength"
        ]
    );
    assert_eq!(
        item["film"],
        json!({ "stock": "kodak_portra_400", "exposure_ev": 0.0 })
    );
    assert_eq!(
        item["print"],
        json!({ "paper": null, "exposure_ev": 0.0, "lights": { "c": 0.0, "m": 0.0, "y": 0.0 } })
    );
    assert_eq!(
        item["grain"],
        json!({ "amount": 100.0, "size": 100.0, "seed": 0 })
    );
    // It reads back as itself.
    let (again, _) = read(v).unwrap();
    assert_eq!(again, s);
}

#[test]
fn the_ports_and_links_are_a_correctors_and_ids_are_shared() {
    // Film key output to a corrector's key input is a link a corrector could have.
    let items = vec![
        json!({ "id": 2, "kind": "film" }),
        json!({ "id": 3, "kind": "corrector" }),
    ];
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "key", json!(3), "key"));
    assert!(read(doc(items.clone(), links)).is_ok());
    // A picture to a key is refused as for correctors.
    let mut links = line(&[1, 2, 3]);
    links.push(link(json!(2), "picture", json!(3), "key"));
    let e = read(doc(items.clone(), links)).unwrap_err();
    assert_eq!(e.to_string(), GraphError::LinkPortTypes.to_string());
    // Two nodes with one id are refused, whatever their kinds.
    let e = read(doc(
        vec![
            json!({ "id": 2, "kind": "film" }),
            json!({ "id": 2, "kind": "corrector" }),
        ],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "Two nodes have the id 2.");
    // Main is a corrector.
    assert!(read(doc(vec![json!({ "id": 1, "kind": "film" })], line(&[1]))).is_err());
    // A film node needs its kind, like any new node.
    let e = read(doc(vec![json!({ "id": 2 })], line(&[1, 2]))).unwrap_err();
    assert_eq!(e.to_string(), GraphError::NeedsKind("2".into()).to_string());
    // The 64-node limit counts Film nodes.
    let items: Vec<Value> = (2..=65)
        .map(|i| json!({ "id": i, "kind": "film" }))
        .collect();
    let e = read(doc(items, line(&[1]))).unwrap_err();
    assert_eq!(e.to_string(), "A photo can have up to 64 nodes.");
}

#[test]
fn an_unknown_field_is_refused_and_a_correctors_fields_are_not_a_film_nodes() {
    let e = with_film(json!({ "trim": { "tint": 3 } })).unwrap_err();
    assert_eq!(e.to_string(), "A film node has no field called “trim”.");
    let e = with_film(json!({ "grade": {} })).unwrap_err();
    assert_eq!(e.to_string(), "A film node has no field called “grade”.");
    // A corrector has none of a film node's.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "corrector", "film": {} })],
        line(&[1, 2]),
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "A corrector has no field called “film”.");
    // A typo inside a field is serde's refusal.
    assert!(matches!(
        with_film(json!({ "grain": { "amout": 3 } })).unwrap_err(),
        SettingsError::Parse(_)
    ));
    assert!(matches!(
        with_film(json!({ "film": { "stok": "x" } })).unwrap_err(),
        SettingsError::Parse(_)
    ));
}

#[test]
fn an_unknown_stock_or_paper_is_refused_in_plain_words() {
    let e = with_film(json!({ "film": { "stock": "kodak_kodachrome_9000" } })).unwrap_err();
    assert_eq!(
        text_of(e),
        "Unknown film stock \"kodak_kodachrome_9000\". Run \"ym film list\" to see the stocks."
    );
    // A paper is not a film, a cine print film is hidden.
    for stock in ["kodak_portra_endura", "kodak_2383", ""] {
        let e = with_film(json!({ "film": { "stock": stock } })).unwrap_err();
        assert!(
            text_of(e).starts_with(&format!("Unknown film stock \"{stock}\".")),
            "{stock}"
        );
    }
    let e = with_film(json!({ "print": { "paper": "nope" } })).unwrap_err();
    assert_eq!(
        text_of(e),
        "Unknown film paper \"nope\". Run \"ym film list\" to see the papers."
    );
    // A film is not a paper; a hidden print film is not offered.
    for paper in ["kodak_portra_400", "kodak_2393"] {
        let e = with_film(json!({ "print": { "paper": paper } })).unwrap_err();
        assert!(text_of(e).starts_with("Unknown film paper"), "{paper}");
    }
    // The real ones are fine, every stock of the picker with its matching paper.
    let c = catalogue().expect("the catalogue");
    let mut films = 0;
    for row in c.rows().iter().filter(|r| c.is_film(&r.id)) {
        films += 1;
        let (s, _) = with_film(json!({ "film": { "stock": row.id } })).unwrap();
        assert_eq!(film_of(&s).film.stock, row.id);
        assert_eq!(c.matching_paper(&row.id).is_none(), c.is_slide(&row.id));
    }
    assert_eq!(films, 20);
    for row in c.rows().iter().filter(|r| c.is_paper(&r.id)) {
        assert!(with_film(json!({ "print": { "paper": row.id } })).is_ok());
    }
}

#[test]
fn a_slide_takes_no_paper() {
    let (s, clamped) =
        with_film(json!({ "film": { "stock": "fujifilm_velvia_100" }, "print": { "paper": "kodak_ultra_endura" } }))
            .unwrap();
    assert_eq!(film_of(&s).print.paper, None);
    assert!(
        clamped.is_empty(),
        "reading a slide's paper as null is not clamping"
    );
    let c = catalogue().unwrap();
    assert_eq!(c.matching_paper("fujifilm_velvia_100"), None);
    // A cine negative's matching print is a hidden print film: a still paper stands in (docs/24 section 4.4).
    assert_eq!(
        c.matching_paper("kodak_vision3_500t").as_deref(),
        Some("kodak_portra_endura")
    );
}

#[test]
fn every_number_is_clamped_and_reported() {
    let (s, clamped) = with_film(json!({
        "film": { "exposure_ev": 9 },
        "print": { "exposure_ev": -9, "lights": { "c": 99, "m": -99, "y": 60 } },
        "halation": { "amount": 999, "size": 1 },
        "grain": { "amount": -5, "size": 900 },
        "strength": 140,
    }))
    .unwrap();
    let f = film_of(&s);
    assert_eq!(f.film.exposure_ev, 4.0);
    assert_eq!(f.print.exposure_ev, -3.0);
    assert_eq!(
        (f.print.lights.c, f.print.lights.m, f.print.lights.y),
        (50.0, -50.0, 50.0)
    );
    assert_eq!((f.halation.amount, f.halation.size), (300.0, 50.0));
    assert_eq!((f.grain.amount, f.grain.size), (0.0, 200.0));
    assert_eq!(f.strength, 100.0);
    for field in [
        SettingsField::Film,
        SettingsField::Print,
        SettingsField::Halation,
        SettingsField::Grain,
        SettingsField::Strength,
    ] {
        assert!(
            clamped.contains(&FieldRef {
                node: film_id(2),
                field
            }),
            "{field:?} is reported"
        );
    }
    // Numbers in range are not.
    let (_, clamped) = with_film(json!({ "strength": 40, "halation": { "amount": 0 } })).unwrap();
    assert!(clamped.is_empty());
    // A number that is not a number, or a seed that is not one, is refused and nothing is applied.
    assert!(matches!(
        with_film(json!({ "strength": "high" })).unwrap_err(),
        SettingsError::Parse(_)
    ));
    assert!(matches!(
        with_film(json!({ "grain": { "seed": -1 } })).unwrap_err(),
        SettingsError::Parse(_)
    ));
    let mut s = with_film(json!({})).unwrap().0;
    let patch = SettingsPatch::node(
        film_id(2),
        NodePatch {
            strength: Some(f64::NAN),
            ..NodePatch::default()
        },
    );
    let before = s.clone();
    assert!(matches!(
        s.apply_patch(&patch, None).unwrap_err(),
        SettingsError::NotFinite(_)
    ));
    assert_eq!(s, before);
}

#[test]
fn the_name_is_trimmed_cut_and_an_empty_one_reads_as_the_default() {
    let (s, _) = with_film(json!({ "name": "   " })).unwrap();
    assert_eq!(film_of(&s).name, "Film");
    let (s, _) =
        with_film(json!({ "name": "  A very long name for a film node indeed  " })).unwrap();
    assert_eq!(film_of(&s).name.chars().count(), 32);
}

fn base() -> DevelopSettings {
    with_film(json!({})).unwrap().0
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

#[test]
fn patches_change_fields_report_them_and_a_partial_keeps_the_rest() {
    let mut s = base();
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": {
                "film": { "stock": "kodak_ektar_100" },
                "print": { "lights": { "y": 12.5 } },
                "grain": { "seed": 9 },
                "strength": 70
            } } })),
            None,
        )
        .unwrap();
    let f = film_of(&s);
    assert_eq!(f.film.stock, "kodak_ektar_100");
    assert_eq!(f.print.lights.y, 12.5);
    assert_eq!((f.print.lights.m, f.print.lights.c), (0.0, 0.0));
    assert_eq!((f.grain.seed, f.grain.amount), (9, 100.0));
    assert_eq!(f.strength, 70.0);
    for field in [
        SettingsField::Film,
        SettingsField::Print,
        SettingsField::Grain,
        SettingsField::Strength,
    ] {
        assert!(
            report.changed.contains(&FieldRef {
                node: film_id(2),
                field
            }),
            "{field:?}"
        );
    }
    assert_eq!(report.changed.len(), 4);
    // The nodes' group re-runs; name and place re-run nothing.
    assert_eq!(
        report.first_group(),
        Some(ym_core::settings::StageGroup::Grade)
    );
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "name": "Print", "pos": [3, 1] } } })),
            None,
        )
        .unwrap();
    assert_eq!(report.first_group(), None);
    // `paper: null` puts the matching paper back; a paper is set by id.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "print": { "paper": "kodak_ultra_endura" } } } })),
        None,
    )
    .unwrap();
    assert_eq!(
        film_of(&s).print.paper.as_deref(),
        Some("kodak_ultra_endura")
    );
    s.apply_patch(
        &patch_of(json!({ "nodes": { "2": { "print": { "paper": null } } } })),
        None,
    )
    .unwrap();
    assert_eq!(film_of(&s).print.paper, None);
    // A stock that is not there is refused and the photo keeps what it had.
    let before = s.clone();
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "film": { "stock": "x" } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(
        text_of(e),
        "Unknown film stock \"x\". Run \"ym film list\" to see the stocks."
    );
    assert_eq!(s, before);
    // Clamping in a patch is reported.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "strength": 400 } } })),
            None,
        )
        .unwrap();
    assert!(report.clamped.contains(&FieldRef {
        node: film_id(2),
        field: SettingsField::Strength
    }));
}

#[test]
fn a_node_patch_adds_a_film_node_removes_it_and_refuses_the_wrong_kind() {
    let mut s = DevelopSettings::default();
    let add = patch_of(json!({
        "nodes": { "5": { "kind": "film", "film": { "stock": "fujifilm_velvia_100" } } },
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 5, "port": "picture" } },
            { "from": { "node": 5, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
        ]
    }));
    let report = s.apply_patch(&add, None).unwrap();
    assert_eq!(report.added, vec![film_id(5)]);
    assert!(s.nodes.film(5).is_some() && s.nodes.corrector(5).is_none());
    assert_eq!(s.nodes.kind_of(film_id(5)), Some(NodeKind::Film));
    // Every field of the new node is reported as changed.
    for field in [
        SettingsField::Film,
        SettingsField::Print,
        SettingsField::Strength,
        SettingsField::Masks,
    ] {
        assert!(
            report.changed.contains(&FieldRef {
                node: film_id(5),
                field
            }),
            "{field:?}"
        );
    }
    assert!(!report.changed.contains(&FieldRef {
        node: film_id(5),
        field: SettingsField::Trim
    }));
    // A kind other than the node's is refused.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "5": { "kind": "corrector" } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(text_of(e), "Node 5 is a film node, not a corrector.");
    // A corrector's field on it is refused.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "5": { "trim": { "tint": 1 } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(text_of(e), "A film node has no field called “trim”.");
    // And a film node's field on a corrector.
    let e = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "1": { "film": { "stock": "kodak_portra_400" } } } })),
            None,
        )
        .unwrap_err();
    assert_eq!(text_of(e), "A corrector has no field called “film”.");
    // Removing it takes its links with it.
    let report = s
        .apply_patch(&patch_of(json!({ "remove": [5] })), None)
        .unwrap_err();
    // (the Output needed its link: the patch alone leaves the Output without a picture)
    assert_eq!(text_of(report), GraphError::OutputLink.to_string());
    let report = s
        .apply_patch(
            &patch_of(json!({ "remove": [5], "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
            ] })),
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![film_id(5)]);
    assert!(s.nodes.film(5).is_none());
}

#[test]
fn undo_is_the_patch_between_two_settings_and_reset_puts_the_defaults_back() {
    let a = base();
    let mut b = a.clone();
    b.apply_patch(
        &patch_of(
            json!({ "nodes": { "2": { "film": { "stock": "kodak_gold_200", "exposure_ev": 1.5 },
            "halation": { "amount": 20 }, "bypass": true } } }),
        ),
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
    assert_eq!(c.nodes.film(2), b.nodes.film(2));
    let drop = SettingsPatch::between(&b, &empty);
    let mut d = b.clone();
    d.apply_patch(&drop, None).unwrap();
    assert!(d.nodes.film(2).is_none());
    // Reset of the head (stock, print, halation, grain, Strength; not name, Bypass, place or masks).
    let reset = SettingsPatch::reset_node_in(&b, film_id(2), &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&reset, None).unwrap();
    let f = film_of(&r);
    assert_eq!(f.film, FilmNode::default().film);
    assert_eq!(f.halation, FilmNode::default().halation);
    assert!(f.bypass, "Reset leaves Bypass alone");
    // Reset of one field.
    let one = SettingsPatch::reset_in(&b, film_id(2), SettingsField::Halation, &LookId::neutral());
    let mut r = b.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(film_of(&r).halation.amount, 100.0);
    assert_eq!(film_of(&r).film.stock, "kodak_gold_200");
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
        &patch_of(json!({ "nodes": { "2": { "film": { "stock": "fujifilm_pro_400h" },
            "grain": { "amount": 33, "seed": 4 }, "name": "Pro",
            "masks": { "items": [ { "kind": "radial" }, { "kind": "object", "points": [[0.5, 0.5, true]] } ] } } } })),
        None,
    )
    .unwrap();
    let resolved = resolved_of(&src);
    // The resolved settings hold the node (it has no Auto to resolve).
    assert_eq!(resolved.settings.nodes.film(2), src.nodes.film(2));
    let copy = SettingsPatch::copied_from(&resolved);
    let node = copy
        .nodes
        .get(&film_id(2))
        .expect("the film node is in the copy");
    assert_eq!(node.kind, Some(NodeKind::Film));
    // Object clicks are not copied: the pasted row arrives empty, as for correctors.
    let mut target = DevelopSettings::default();
    target
        .apply_patch(&patch_of(json!({ "nodes": { "7": { "kind": "film" } },
            "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 7, "port": "picture" } },
                { "from": { "node": 7, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] })), None)
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(
        paste.remove.contains(&film_id(7)),
        "the target's own film node goes"
    );
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.film(7).is_none());
    let f = target.nodes.film(2).expect("pasted");
    assert_eq!(f.film.stock, "fujifilm_pro_400h");
    assert_eq!(
        (f.grain.amount, f.grain.seed, f.name.as_str()),
        (33.0, 4, "Pro")
    );
    assert_eq!(f.masks.items.len(), 2);
    match &f.masks.items[1].shape {
        ym_core::settings::MaskShape::Object(o) => assert!(o.points.is_empty() && o.box_.is_none()),
        other => panic!("{other:?}"),
    }
    assert_eq!(target.nodes.links, src.nodes.links);
}

#[test]
fn the_xmp_says_which_film_data_made_the_picture() {
    // Nothing without a live Film node: not for the default graph, a bypassed node or Strength 0.
    assert_eq!(film_data_of(&DevelopSettings::default()), None);
    for body in [json!({ "bypass": true }), json!({ "strength": 0 })] {
        assert_eq!(film_data_of(&with_film(body).unwrap().0), None);
    }
    let text = film_data_of(&base()).expect("FilmData");
    assert!(
        text.starts_with("kodak_portra_400 on kodak_portra_endura; profiles 0.3.2; engine "),
        "{text}"
    );
    assert!(text.contains("Andrea Volpato"), "{text}");
    let slide = film_data_of(
        &with_film(json!({ "film": { "stock": "fujifilm_velvia_100" } }))
            .unwrap()
            .0,
    )
    .unwrap();
    assert!(
        slide.starts_with("fujifilm_velvia_100; profiles"),
        "{slide}"
    );
    let paper = film_data_of(
        &with_film(json!({ "print": { "paper": "kodak_ultra_endura" } }))
            .unwrap()
            .0,
    )
    .unwrap();
    assert!(
        paper.starts_with("kodak_portra_400 on kodak_ultra_endura;"),
        "{paper}"
    );
    // A film node that is not on the path to the Output is not in the picture.
    let (dead, _) = read(json!({ "nodes": { "items": [ { "id": 2, "kind": "film" } ] } })).unwrap();
    assert_eq!(film_data_of(&dead), None);
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
        &patch_of(json!({ "nodes": { "2": { "masks": { "items": [ { "kind": "linear" } ] } } } })),
        None,
    )
    .unwrap();
    let written = serde_json::to_value(&s).unwrap();
    let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
    assert!(problems.is_empty(), "{problems:?}");
    // What the core refuses, the schema does too.
    let bad = |film: Value| {
        let mut node = json!({ "id": 2, "kind": "film" });
        node.as_object_mut()
            .unwrap()
            .extend(film.as_object().unwrap().clone());
        json!({ "nodes": { "items": [node] } })
    };
    for film in [
        json!({ "film": { "exposure_ev": 9 } }),
        json!({ "strength": 101 }),
        json!({ "halation": { "size": 10 } }),
        json!({ "grain": { "seed": -1 } }),
        json!({ "trim": {} }),
        json!({ "print": { "lights": { "c": 51 } } }),
    ] {
        assert!(!v.is_valid(&bad(film.clone())), "{film}");
    }
    assert!(v.is_valid(&bad(json!({}))));
    // The ranges are the ones the core clamps to.
    use ym_core::settings::*;
    let p = &schema["$defs"]["film_node"]["properties"];
    let range = |p: &Value| {
        (
            p["minimum"].as_f64().unwrap(),
            p["maximum"].as_f64().unwrap(),
        )
    };
    assert_eq!(
        range(&p["film"]["properties"]["exposure_ev"]),
        FILM_EXPOSURE_RANGE_EV
    );
    assert_eq!(
        range(&p["print"]["properties"]["exposure_ev"]),
        PRINT_EXPOSURE_RANGE_EV
    );
    assert_eq!(
        range(&p["print"]["properties"]["lights"]["properties"]["c"]),
        LIGHTS_RANGE
    );
    assert_eq!(
        range(&p["halation"]["properties"]["amount"]),
        HALATION_AMOUNT_RANGE
    );
    assert_eq!(range(&p["halation"]["properties"]["size"]), SIZE_RANGE);
    assert_eq!(
        range(&p["grain"]["properties"]["amount"]),
        GRAIN_AMOUNT_RANGE
    );
    assert_eq!(range(&p["grain"]["properties"]["size"]), SIZE_RANGE);
    assert_eq!(range(&p["strength"]), STRENGTH_RANGE);
    assert_eq!(
        p["film"]["properties"]["stock"]["default"],
        json!(DEFAULT_STOCK)
    );
}

#[test]
fn the_built_in_catalogue_is_the_shipped_names_file() {
    let shipped = ym_core::film::Catalogue::load(&repo_root().join("data")).unwrap();
    let built_in = catalogue().unwrap();
    assert_eq!(shipped.profile_count(), 28);
    assert_eq!(built_in.profile_count(), shipped.profile_count());
    assert_eq!(built_in.profile_version(), shipped.profile_version());
    assert_eq!(built_in.profile_version(), "0.3.2");
}
