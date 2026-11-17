//! The mixer nodes' settings (`docs/25-mixers.md` section 2): the four kinds, their ports and their types, the
//! fields and their defaults, every rule of section 2.3 with its text, patches, Reset, undo, Copy and Paste, the
//! written form, the JSON schema and the sentences that name the kinds.
#![allow(clippy::field_reassign_with_default)]

mod common;

use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_core::session::PhotoId;
use ym_core::settings::{
    BlendMode, DevelopSettings, FieldRef, MixerNode, NodeId, NodeKind, NodePatch, Port,
    ResolvedSettings, SettingsError, SettingsField, SettingsPatch, StageGroup, ports_in, ports_out,
};
use ym_looks::library::LookId;

fn id(n: u32) -> NodeId {
    NodeId::Corrector(n)
}

fn link(from: Value, fp: &str, to: Value, tp: &str) -> Value {
    json!({ "from": { "node": from, "port": fp }, "to": { "node": to, "port": tp } })
}

fn default_links() -> Vec<Value> {
    vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!("output"), "picture"),
    ]
}

fn doc(items: Vec<Value>, links: Vec<Value>) -> Value {
    json!({ "nodes": { "items": items, "links": links } })
}

fn read(doc: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    DevelopSettings::from_value(doc, &LookId::neutral())
}

fn text_of<T>(r: Result<T, SettingsError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

fn patch_of(v: Value) -> SettingsPatch {
    serde_json::from_value(v).expect("a patch")
}

/// The links of Input -> Main -> `mixer` (input 1) -> Output, plus anything in `more`.
fn through(mixer: u32, more: Vec<Value>) -> Vec<Value> {
    let mut links = vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!(mixer), "picture.1"),
        link(json!(mixer), "picture", json!("output"), "picture"),
    ];
    links.extend(more);
    links
}

fn with_layer(body: Value) -> Result<(DevelopSettings, Vec<FieldRef>), SettingsError> {
    let mut node = json!({ "id": 2, "kind": "layer" });
    node.as_object_mut()
        .unwrap()
        .extend(body.as_object().unwrap().clone());
    read(doc(vec![node], through(2, vec![])))
}

/// The first example of `docs/25` section 2.4 (a Parallel of two correctors).
fn parallel_example() -> Value {
    json!({ "nodes": {
      "items": [
        { "id": 2, "kind": "corrector", "name": "Skin", "pos": [1, -0.5],
          "grade": { "saturation": { "kind": "manual", "value": 15 } },
          "masks": { "invert": false, "items": [
            { "kind": "color_range", "center": [0.06, 0.04], "width": 0.05, "softness": 0.5 } ] } },
        { "id": 3, "kind": "corrector", "name": "Sky", "pos": [1, 0.5],
          "trim": { "temperature": -18 },
          "masks": { "invert": false, "items": [
            { "kind": "linear", "center": [0.5, 0.2], "angle_deg": 4, "feather": 0.25 } ] } },
        { "id": 4, "kind": "parallel", "pos": [2, 0] }
      ],
      "links": [
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!(2), "picture"),
        link(json!(1), "picture", json!(3), "picture"),
        link(json!(2), "picture", json!(4), "picture.1"),
        link(json!(3), "picture", json!(4), "picture.2"),
        link(json!(4), "picture", json!("output"), "picture")
      ] } })
}

/// The second example of `docs/25` section 2.4 (a Splitter and a Combiner around three correctors).
fn trio_example() -> Value {
    json!({ "nodes": {
      "items": [
        { "id": 2, "kind": "splitter", "pos": [1, 0] },
        { "id": 3, "kind": "corrector", "name": "Red", "pos": [2, -1], "trim": { "exposure_ev": 0.2 } },
        { "id": 4, "kind": "corrector", "name": "Green", "pos": [2, 0] },
        { "id": 5, "kind": "corrector", "name": "Blue", "pos": [2, 1], "trim": { "exposure_ev": -0.2 } },
        { "id": 6, "kind": "combiner", "pos": [3, 0] }
      ],
      "links": [
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!(2), "picture"),
        link(json!(2), "red", json!(3), "picture"),
        link(json!(2), "green", json!(4), "picture"),
        link(json!(2), "blue", json!(5), "picture"),
        link(json!(3), "picture", json!(6), "red"),
        link(json!(4), "picture", json!(6), "green"),
        link(json!(5), "picture", json!(6), "blue"),
        link(json!(6), "picture", json!("output"), "picture")
      ] } })
}

/// All four kinds in one graph: Main -> Splitter -> (red, green, blue) -> Combiner -> Layer.2, Main -> Layer.1;
/// Layer -> Parallel.1 -> Output.
fn all_four() -> DevelopSettings {
    read(doc(
        vec![
            json!({ "id": 2, "kind": "splitter", "name": "Split", "pos": [1, 0] }),
            json!({ "id": 3, "kind": "combiner", "pos": [2, 0] }),
            json!({ "id": 4, "kind": "layer", "mode": "multiply", "opacity": 0.4, "bypass": true,
                "name": "Stack", "pos": [3, 0] }),
            json!({ "id": 5, "kind": "parallel", "name": "Sum", "pos": [4, 0] }),
        ],
        vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "picture"),
            link(json!(2), "red", json!(3), "red"),
            link(json!(2), "green", json!(3), "green"),
            link(json!(2), "blue", json!(3), "blue"),
            link(json!(1), "picture", json!(4), "picture.1"),
            link(json!(3), "picture", json!(4), "picture.2"),
            link(json!(4), "picture", json!(5), "picture.1"),
            link(json!(5), "picture", json!("output"), "picture"),
        ],
    ))
    .unwrap()
    .0
}

fn layer_of(s: &DevelopSettings, n: u32) -> &ym_core::settings::LayerNode {
    match s.nodes.mixer(n) {
        Some(MixerNode::Layer(l)) => l,
        other => panic!("node {n} is not a layer: {other:?}"),
    }
}

// ---- the kinds, the sentences and the ports ----------------------------------------------------------------------

#[test]
fn the_four_kinds_have_names_and_sentences() {
    let kinds = [
        (
            NodeKind::Parallel,
            "parallel",
            "a parallel mixer",
            "A parallel mixer",
        ),
        (NodeKind::Layer, "layer", "a layer mixer", "A layer mixer"),
        (NodeKind::Splitter, "splitter", "a splitter", "A splitter"),
        (NodeKind::Combiner, "combiner", "a combiner", "A combiner"),
    ];
    for (kind, name, article, sentence) in kinds {
        assert!(NodeKind::ALL.contains(&kind));
        assert_eq!(kind.name(), name);
        assert_eq!(kind.with_article(), article);
        assert_eq!(kind.sentence(), sentence);
        assert_eq!(NodeKind::parse(name).unwrap(), kind);
    }
    assert_eq!(NodeKind::ALL.len(), 9);
    assert_eq!(
        NodeKind::parse("outside").unwrap_err().to_string(),
        "This settings file has a node of a kind (“outside”) that this version of Yellowmoon does not have. Update Yellowmoon."
    );
}

#[test]
fn ports_have_names_types_and_the_written_order() {
    // The port names round trip, and the numbered inputs are exactly picture.1 to picture.8.
    for (name, port) in [
        ("picture", Port::Picture),
        ("key", Port::Key),
        ("picture.1", Port::PictureN(1)),
        ("picture.8", Port::PictureN(8)),
        ("red", Port::Red),
        ("green", Port::Green),
        ("blue", Port::Blue),
    ] {
        assert_eq!(Port::parse(name), port);
        assert_eq!(port.name(), name);
        let json = serde_json::to_value(&port).unwrap();
        assert_eq!(json, json!(name));
        assert_eq!(serde_json::from_value::<Port>(json).unwrap(), port);
    }
    for odd in [
        "picture.0",
        "picture.01",
        "picture.9",
        "picture.10",
        "picture.",
        "Picture",
        "alpha",
    ] {
        assert_eq!(Port::parse(odd), Port::Other(odd.to_owned()), "{odd}");
    }
    // The type of each.
    use ym_core::settings::PortType;
    for p in [
        Port::Picture,
        Port::PictureN(3),
        Port::Red,
        Port::Green,
        Port::Blue,
    ] {
        assert_eq!(p.port_type(), Some(PortType::Picture), "{p}");
    }
    assert_eq!(Port::Key.port_type(), Some(PortType::Key));
    assert_eq!(Port::Other("x".into()).port_type(), None);
    // The written order: picture, key, picture.1 to picture.8, red, green, blue.
    let mut ports = [
        Port::Blue,
        Port::PictureN(2),
        Port::Red,
        Port::Key,
        Port::PictureN(1),
        Port::Green,
        Port::Picture,
        Port::PictureN(8),
    ];
    ports.sort();
    let names: Vec<String> = ports.iter().map(Port::name).collect();
    assert_eq!(
        names,
        [
            "picture",
            "key",
            "picture.1",
            "picture.2",
            "picture.8",
            "red",
            "green",
            "blue"
        ]
    );
    // The ports of every kind.
    let mixer_inputs: Vec<String> = (1..=8).map(|k| format!("picture.{k}")).collect();
    for kind in [NodeKind::Parallel, NodeKind::Layer] {
        assert_eq!(
            ports_in(kind).iter().map(Port::name).collect::<Vec<_>>(),
            mixer_inputs
        );
        assert_eq!(ports_out(kind), &[Port::Picture]);
    }
    assert_eq!(ports_in(NodeKind::Splitter), &[Port::Picture]);
    assert_eq!(
        ports_out(NodeKind::Splitter),
        &[Port::Red, Port::Green, Port::Blue]
    );
    assert_eq!(
        ports_in(NodeKind::Combiner),
        &[Port::Red, Port::Green, Port::Blue]
    );
    assert_eq!(ports_out(NodeKind::Combiner), &[Port::Picture]);
    assert_eq!(ports_in(NodeKind::Corrector), &[Port::Picture, Port::Key]);
    assert_eq!(ports_out(NodeKind::Film), &[Port::Picture, Port::Key]);
    // None of the four has a key port.
    for kind in [
        NodeKind::Parallel,
        NodeKind::Layer,
        NodeKind::Splitter,
        NodeKind::Combiner,
    ] {
        assert!(!ports_in(kind).contains(&Port::Key) && !ports_out(kind).contains(&Port::Key));
    }
}

// ---- the fields and their defaults -------------------------------------------------------------------------------

fn keys_of(item: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = item
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

fn item(s: &DevelopSettings, n: u32) -> Value {
    serde_json::to_value(s).unwrap()["nodes"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"] == json!(n))
        .unwrap()
        .clone()
}

#[test]
fn a_new_node_of_each_kind_has_its_defaults_and_is_written_with_its_fields_only() {
    let (s, clamped) = read(doc(
        vec![
            json!({ "id": 2, "kind": "parallel" }),
            json!({ "id": 3, "kind": "layer" }),
            json!({ "id": 4, "kind": "splitter" }),
            json!({ "id": 5, "kind": "combiner" }),
        ],
        through(2, vec![]),
    ))
    .unwrap();
    assert!(clamped.is_empty());
    let names: Vec<&str> = (2..=5).map(|n| s.nodes.mixer(n).unwrap().name()).collect();
    assert_eq!(names, ["Parallel", "Layer", "Splitter", "Combiner"]);
    let l = layer_of(&s, 3);
    assert_eq!(
        (l.mode, l.opacity, l.bypass),
        (BlendMode::Normal, 1.0, false)
    );
    assert_eq!(s.nodes.kind_of(id(2)), Some(NodeKind::Parallel));
    assert_eq!(s.nodes.kind_of(id(3)), Some(NodeKind::Layer));
    assert_eq!(s.nodes.kind_of(id(4)), Some(NodeKind::Splitter));
    assert_eq!(s.nodes.kind_of(id(5)), Some(NodeKind::Combiner));
    // No Auto, no masks: not even a masks accessor.
    for n in 2..=5 {
        assert!(s.nodes.masks_of(n).is_none());
    }
    assert_eq!(
        keys_of(&item(&s, 2)),
        ["bypass", "id", "kind", "name", "pos"]
    );
    assert_eq!(
        keys_of(&item(&s, 3)),
        ["bypass", "id", "kind", "mode", "name", "opacity", "pos"]
    );
    assert_eq!(keys_of(&item(&s, 4)), ["id", "kind", "name", "pos"]);
    assert_eq!(keys_of(&item(&s, 5)), ["id", "kind", "name", "pos"]);
    // A Layer as docs/25 section 2.4 writes it.
    assert_eq!(
        item(&s, 3),
        json!({ "id": 3, "kind": "layer", "name": "Layer", "bypass": false, "pos": [2.0, 0.0],
            "mode": "normal", "opacity": 1.0 })
    );
    // The nodes with no place are placed in a row by number, as any node is.
    assert_eq!(s.nodes.pos(id(2)), Some([1.0, 0.0]));
    assert_eq!(s.nodes.pos(id(5)), Some([4.0, 0.0]));
}

#[test]
fn name_is_trimmed_cut_and_defaults_for_every_kind() {
    for (kind, default) in [
        ("parallel", "Parallel"),
        ("layer", "Layer"),
        ("splitter", "Splitter"),
        ("combiner", "Combiner"),
    ] {
        let node = |name: &str| {
            let links = if kind == "layer" || kind == "parallel" {
                through(2, vec![])
            } else {
                vec![
                    link(json!("input"), "picture", json!(1), "picture"),
                    link(json!(1), "picture", json!("output"), "picture"),
                ]
            };
            read(doc(
                vec![json!({ "id": 2, "kind": kind, "name": name })],
                links,
            ))
            .unwrap()
            .0
        };
        assert_eq!(node("   ").nodes.mixer(2).unwrap().name(), default);
        assert_eq!(node("  Mix  ").nodes.mixer(2).unwrap().name(), "Mix");
        let long = "x".repeat(40);
        assert_eq!(node(&long).nodes.mixer(2).unwrap().name(), "x".repeat(32));
    }
}

#[test]
fn opacity_is_clamped_and_reported_and_a_bad_value_is_refused() {
    let (s, clamped) = with_layer(json!({ "opacity": 1.5 })).unwrap();
    assert_eq!(layer_of(&s, 2).opacity, 1.0);
    assert_eq!(
        clamped,
        vec![FieldRef {
            node: id(2),
            field: SettingsField::Opacity
        }]
    );
    assert_eq!(clamped[0].to_string(), "2.opacity");
    let (s, clamped) = with_layer(json!({ "opacity": -3 })).unwrap();
    assert_eq!(layer_of(&s, 2).opacity, 0.0);
    assert_eq!(clamped.len(), 1);
    assert!(with_layer(json!({ "opacity": 0.25 })).unwrap().1.is_empty());
    // Not finite: nothing is applied.
    let mut s = with_layer(json!({})).unwrap().0;
    let before = s.clone();
    let e = s
        .apply_patch(
            &SettingsPatch::node(
                id(2),
                NodePatch {
                    opacity: Some(f64::NAN),
                    ..NodePatch::default()
                },
            ),
            None,
        )
        .unwrap_err();
    assert!(
        matches!(e, SettingsError::NotFinite(ref p) if p.ends_with(".opacity")),
        "{e}"
    );
    assert_eq!(s, before);
    // A mode that is not in the list is a parse error.
    let e = with_layer(json!({ "mode": "screen" })).unwrap_err();
    assert!(matches!(e, SettingsError::Parse(_)), "{e}");
    assert!(
        e.to_string().starts_with("The settings can't be read:"),
        "{e}"
    );
    for mode in BlendMode::ALL {
        let s = with_layer(json!({ "mode": mode.name() })).unwrap().0;
        assert_eq!(layer_of(&s, 2).mode, mode);
    }
}

#[test]
fn a_field_of_another_kind_is_refused_with_the_kinds_name() {
    let one = |kind: &str, field: Value| {
        let mut node = json!({ "id": 2, "kind": kind });
        node.as_object_mut()
            .unwrap()
            .extend(field.as_object().unwrap().clone());
        let links = if kind == "parallel" || kind == "layer" {
            through(2, vec![])
        } else {
            vec![
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!("output"), "picture"),
            ]
        };
        text_of(read(doc(vec![node], links)))
    };
    assert_eq!(
        one("parallel", json!({ "masks": { "items": [] } })),
        "A parallel mixer has no field called “masks”."
    );
    assert_eq!(
        one("layer", json!({ "masks": { "invert": false } })),
        "A layer mixer has no field called “masks”."
    );
    assert_eq!(
        one("parallel", json!({ "mode": "add" })),
        "A parallel mixer has no field called “mode”."
    );
    assert_eq!(
        one("splitter", json!({ "bypass": true })),
        "A splitter has no field called “bypass”."
    );
    assert_eq!(
        one("combiner", json!({ "trim": {} })),
        "A combiner has no field called “trim”."
    );
    // And a field of theirs on a corrector.
    let e = read(doc(
        vec![json!({ "id": 2, "kind": "corrector", "opacity": 0.5 })],
        vec![],
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "A corrector has no field called “opacity”.");
}

// ---- the rules of section 2.3, each with its text -----------------------------------------------------------------

#[test]
fn a_ninth_input_is_refused() {
    let items = vec![json!({ "id": 2, "kind": "layer" })];
    for port in ["picture.9", "picture.10", "picture.99999999999999999999"] {
        let links = vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), port),
            link(json!(2), "picture", json!("output"), "picture"),
        ];
        assert_eq!(
            text_of(read(doc(items.clone(), links))),
            "A mixer can have up to 8 inputs.",
            "{port}"
        );
    }
    // Eight are fine.
    let mut links = vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(2), "picture", json!("output"), "picture"),
    ];
    for k in 1..=8 {
        links.push(link(json!(1), "picture", json!(2), &format!("picture.{k}")));
    }
    assert!(read(doc(items, links)).is_ok());
    // On a node that is not a mixer it is an unknown port.
    let links = vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!("output"), "picture.9"),
    ];
    assert_eq!(
        text_of(read(doc(vec![], links))),
        "The output has no port called “picture.9”."
    );
}

#[test]
fn a_gap_in_a_mixers_inputs_is_refused() {
    for kind in ["parallel", "layer"] {
        let items = vec![json!({ "id": 7, "kind": kind })];
        let links = vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(7), "picture.1"),
            link(json!(1), "picture", json!(7), "picture.3"),
            link(json!(7), "picture", json!("output"), "picture"),
        ];
        assert_eq!(
            text_of(read(doc(items.clone(), links))),
            "Node 7 has a link on input 3 and none on input 2. A mixer's inputs are numbered from 1 with no gap."
        );
        // A first input missing.
        let links = vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(7), "picture.2"),
            link(json!(7), "picture", json!("output"), "picture"),
        ];
        assert_eq!(
            text_of(read(doc(items.clone(), links))),
            "Node 7 has a link on input 2 and none on input 1. A mixer's inputs are numbered from 1 with no gap."
        );
        // No linked input at all is not a gap.
        let links = vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!("output"), "picture"),
        ];
        assert!(read(doc(items, links)).is_ok());
    }
}

#[test]
fn a_port_the_kind_does_not_have_is_refused_with_the_existing_text() {
    let nodes = || {
        vec![
            json!({ "id": 7, "kind": "layer" }),
            json!({ "id": 8, "kind": "splitter" }),
            json!({ "id": 9, "kind": "combiner" }),
        ]
    };
    let base_links = || {
        vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!("output"), "picture"),
        ]
    };
    let with = |extra: Value| {
        let mut links = base_links();
        links.push(extra);
        text_of(read(doc(nodes(), links)))
    };
    // A bare `picture` on a mixer's input side, and `picture.0`, `picture.01`.
    for port in ["picture", "picture.0", "picture.01", "key", "red"] {
        assert_eq!(
            with(link(json!(1), "picture", json!(7), port)),
            format!("Node 7 has no port called “{port}”."),
            "{port}"
        );
    }
    // `key` on any of the four, in either direction.
    assert_eq!(
        with(link(json!(1), "key", json!(7), "key")),
        "Node 7 has no port called “key”."
    );
    assert_eq!(
        with(link(json!(7), "key", json!(1), "key")),
        "Node 7 has no port called “key”."
    );
    assert_eq!(
        with(link(json!(8), "key", json!(1), "key")),
        "Node 8 has no port called “key”."
    );
    assert_eq!(
        with(link(json!(1), "key", json!(9), "key")),
        "Node 9 has no port called “key”."
    );
    // `red` on a mixer's output, `picture` on a Splitter's outputs, a Combiner's `picture` input.
    assert_eq!(
        with(link(json!(7), "red", json!(1), "picture")),
        "Node 7 has no port called “red”."
    );
    assert_eq!(
        with(link(json!(8), "picture", json!(9), "red")),
        "Node 8 has no port called “picture”."
    );
    assert_eq!(
        with(link(json!(1), "picture", json!(9), "picture")),
        "Node 9 has no port called “picture”."
    );
    assert_eq!(
        with(link(json!(1), "picture", json!(8), "red")),
        "Node 8 has no port called “red”."
    );
}

#[test]
fn a_link_joins_two_ports_of_one_type() {
    let items = vec![
        json!({ "id": 2, "kind": "splitter" }),
        json!({ "id": 3, "kind": "combiner" }),
        json!({ "id": 4, "kind": "layer" }),
        json!({ "id": 5, "kind": "corrector" }),
    ];
    let links = |extra: Vec<Value>| {
        let mut l = vec![
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "picture"),
            link(json!(3), "picture", json!("output"), "picture"),
        ];
        l.extend(extra);
        l
    };
    // A Splitter's red may feed a Combiner's green (a channel swap), a corrector's picture or a mixer's input.
    assert!(
        read(doc(
            items.clone(),
            links(vec![
                link(json!(2), "red", json!(3), "green"),
                link(json!(2), "green", json!(3), "blue"),
                link(json!(2), "blue", json!(3), "red"),
                link(json!(2), "red", json!(5), "picture"),
                link(json!(2), "blue", json!(4), "picture.1"),
                link(json!(5), "picture", json!(4), "picture.2"),
            ])
        ))
        .is_ok()
    );
    // A corrector's picture may feed a Combiner's red.
    assert!(
        read(doc(
            items.clone(),
            links(vec![link(json!(5), "picture", json!(3), "red")])
        ))
        .is_ok()
    );
    // A Splitter's output may feed the Output.
    assert!(
        read(doc(
            items.clone(),
            vec![
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!(2), "picture"),
                link(json!(2), "green", json!("output"), "picture"),
            ]
        ))
        .is_ok()
    );
    // A key into a mixer is refused: a picture port and a key.
    let sentence = "A link joins a picture to a picture, or a key to a key.";
    assert_eq!(
        text_of(read(doc(
            items.clone(),
            links(vec![link(json!(5), "key", json!(4), "picture.1")])
        ))),
        sentence
    );
    assert_eq!(
        text_of(read(doc(
            items,
            links(vec![
                link(json!(5), "picture", json!(3), "red"),
                link(json!(2), "red", json!(5), "key")
            ])
        ))),
        sentence
    );
}

#[test]
fn an_input_has_one_link_and_no_circle() {
    let items = vec![json!({ "id": 2, "kind": "parallel" })];
    let links = vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!(2), "picture.1"),
        link(json!("input"), "picture", json!(2), "picture.1"),
        link(json!(2), "picture", json!("output"), "picture"),
    ];
    assert_eq!(
        text_of(read(doc(items, links))),
        "An input can have one link. Node 2 has two on its picture.1 input."
    );
    // A circle through a Splitter and a Combiner.
    let items = vec![
        json!({ "id": 2, "kind": "splitter" }),
        json!({ "id": 3, "kind": "combiner" }),
    ];
    let links = vec![
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!("output"), "picture"),
        link(json!(2), "red", json!(3), "red"),
        link(json!(3), "picture", json!(2), "picture"),
    ];
    assert_eq!(
        text_of(read(doc(items, links))),
        "The nodes of this photo are linked in a circle. Remove a link."
    );
}

#[test]
fn the_new_kinds_count_toward_the_64_nodes() {
    let kinds = [
        "corrector",
        "film",
        "parallel",
        "layer",
        "splitter",
        "combiner",
    ];
    // Main and 63 others: allowed. The others are in no link (dead nodes are fine).
    let items = |n: u32| -> Vec<Value> {
        (0..n)
            .map(|i| json!({ "id": 2 + i, "kind": kinds[i as usize % kinds.len()] }))
            .collect()
    };
    read(doc(items(63), default_links())).unwrap();
    assert_eq!(
        text_of(read(doc(items(64), default_links()))),
        "A photo can have up to 64 nodes."
    );
    // Only the new kinds also count.
    let only = |n: u32| -> Vec<Value> {
        (0..n)
            .map(|i| json!({ "id": 2 + i, "kind": "splitter" }))
            .collect()
    };
    assert!(read(doc(only(63), default_links())).is_ok());
    assert_eq!(
        text_of(read(doc(only(64), default_links()))),
        "A photo can have up to 64 nodes."
    );
}

#[test]
fn an_id_belongs_to_one_node() {
    let e = read(doc(
        vec![
            json!({ "id": 2, "kind": "layer" }),
            json!({ "id": 2, "kind": "splitter" }),
        ],
        vec![],
    ))
    .unwrap_err();
    assert_eq!(e.to_string(), "Two nodes have the id 2.");
    // Main is a corrector.
    let e = read(doc(vec![json!({ "id": 1, "kind": "layer" })], vec![])).unwrap_err();
    assert_eq!(
        e.to_string(),
        "A photo needs its input, its output and node 1, Main."
    );
    // A new node says what it is.
    let e = read(doc(vec![json!({ "id": 2 })], vec![])).unwrap_err();
    assert_eq!(
        e.to_string(),
        "Node 2 is not in the default graph, so it needs its “kind”."
    );
}

// ---- the examples, the written form and the schema ---------------------------------------------------------------

#[test]
fn the_two_examples_read_and_write_the_same_bytes_whatever_the_order_of_the_file() {
    for example in [parallel_example(), trio_example()] {
        let (s, clamped) = read(example.clone()).unwrap();
        assert!(clamped.is_empty());
        let written = serde_json::to_string(&s).unwrap();
        // A round trip is the same bytes.
        let (again, _) = read(serde_json::from_str(&written).unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&again).unwrap(), written);
        assert_eq!(again, s);
        // Items and links in another order are the same graph, the same bytes.
        let mut shuffled = example.clone();
        shuffled["nodes"]["items"].as_array_mut().unwrap().reverse();
        shuffled["nodes"]["links"].as_array_mut().unwrap().reverse();
        let (other, _) = read(shuffled).unwrap();
        assert_eq!(serde_json::to_string(&other).unwrap(), written);
    }
    // The written order of the links follows the order of ports.
    let (s, _) = read(trio_example()).unwrap();
    let v = serde_json::to_value(&s).unwrap();
    let ports: Vec<String> = v["nodes"]["links"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| l["from"]["node"] == json!(2))
        .map(|l| l["from"]["port"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(ports, ["red", "green", "blue"]);
    let (s, _) = read(parallel_example()).unwrap();
    let v = serde_json::to_value(&s).unwrap();
    let ports: Vec<String> = v["nodes"]["links"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| l["to"]["node"] == json!(4))
        .map(|l| l["to"]["port"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(ports, ["picture.1", "picture.2"]);
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
    for s in [
        read(parallel_example()).unwrap().0,
        read(trio_example()).unwrap().0,
        all_four(),
        with_layer(json!({})).unwrap().0,
    ] {
        let written = serde_json::to_value(&s).unwrap();
        let problems: Vec<String> = v.iter_errors(&written).map(|e| e.to_string()).collect();
        assert!(problems.is_empty(), "{problems:?}");
    }
    // What the core refuses, the schema does too.
    let bad = |node: Value, links: Vec<Value>| doc(vec![node], links);
    let one = through(2, vec![]);
    for node in [
        json!({ "id": 2, "kind": "layer", "mode": "screen" }),
        json!({ "id": 2, "kind": "layer", "opacity": 2 }),
        json!({ "id": 2, "kind": "layer", "masks": {} }),
        json!({ "id": 2, "kind": "parallel", "mode": "add" }),
        json!({ "id": 2, "kind": "splitter", "bypass": true }),
        json!({ "id": 2, "kind": "combiner", "bypass": false }),
        json!({ "id": 2, "kind": "outside" }),
    ] {
        assert!(!v.is_valid(&bad(node.clone(), one.clone())), "{node}");
    }
    for node in [
        json!({ "id": 2, "kind": "layer", "mode": "difference", "opacity": 0.5, "bypass": true }),
        json!({ "id": 2, "kind": "parallel" }),
        json!({ "id": 2, "kind": "splitter", "name": "S" }),
        json!({ "id": 2, "kind": "combiner" }),
    ] {
        assert!(v.is_valid(&bad(node.clone(), one.clone())), "{node}");
    }
    // The port names.
    for port in ["picture.1", "picture.8", "red", "green", "blue"] {
        let l = vec![link(json!(1), "picture", json!(2), port)];
        assert!(v.is_valid(&doc(vec![], l)), "{port}");
    }
    for port in ["picture.0", "picture.9", "picture.01", "alpha"] {
        let l = vec![link(json!(1), "picture", json!(2), port)];
        assert!(!v.is_valid(&doc(vec![], l)), "{port}");
    }
    // The opacity range is the one the core clamps to.
    let p = &schema["$defs"]["layer_node"]["properties"]["opacity"];
    assert_eq!(
        (
            p["minimum"].as_f64().unwrap(),
            p["maximum"].as_f64().unwrap()
        ),
        ym_core::settings::OPACITY_RANGE
    );
}

// ---- patches, Reset, undo, Copy and Paste ------------------------------------------------------------------------

#[test]
fn a_patch_adds_each_kind_changes_a_layer_and_removes_a_node() {
    let mut s = DevelopSettings::default();
    let add = patch_of(json!({
        "nodes": {
            "2": { "kind": "layer", "mode": "add", "opacity": 0.5, "name": "Glow" },
            "3": { "kind": "parallel" },
            "4": { "kind": "splitter" },
            "5": { "kind": "combiner" }
        },
        "links": [
            link(json!("input"), "picture", json!(1), "picture"),
            link(json!(1), "picture", json!(2), "picture.1"),
            link(json!(2), "picture", json!("output"), "picture"),
        ]
    }));
    let report = s.apply_patch(&add, None).unwrap();
    assert_eq!(report.added, vec![id(2), id(3), id(4), id(5)]);
    let l = layer_of(&s, 2);
    assert_eq!(
        (l.mode, l.opacity, l.name.as_str()),
        (BlendMode::Add, 0.5, "Glow")
    );
    assert_eq!(s.nodes.mixer(3).unwrap().kind(), NodeKind::Parallel);
    assert_eq!(s.nodes.kind_of(id(4)), Some(NodeKind::Splitter));
    assert_eq!(s.nodes.node_count(), 5);
    // Every field of a new Layer is reported, and not another kind's.
    for field in [
        SettingsField::Mode,
        SettingsField::Opacity,
        SettingsField::Bypass,
        SettingsField::Name,
    ] {
        assert!(
            report.changed.contains(&FieldRef { node: id(2), field }),
            "{field:?}"
        );
    }
    assert!(!report.changed.contains(&FieldRef {
        node: id(3),
        field: SettingsField::Mode
    }));
    assert!(!report.changed.contains(&FieldRef {
        node: id(2),
        field: SettingsField::Masks
    }));
    // The nodes' group: a change runs the colour pass again and nothing else.
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    // `mode` and `opacity` patches, their report and the clamp.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "mode": "darken", "opacity": 4 } } })),
            None,
        )
        .unwrap();
    assert_eq!(layer_of(&s, 2).mode, BlendMode::Darken);
    assert_eq!(layer_of(&s, 2).opacity, 1.0);
    assert_eq!(
        report.changed,
        vec![
            FieldRef::new(id(2), SettingsField::Mode),
            FieldRef::new(id(2), SettingsField::Opacity)
        ]
    );
    assert_eq!(
        report.clamped,
        vec![FieldRef::new(id(2), SettingsField::Opacity)]
    );
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "2": { "opacity": 0.5 } } })),
            None,
        )
        .unwrap();
    assert_eq!(
        report.changed,
        vec![FieldRef::new(id(2), SettingsField::Opacity)]
    );
    // A name and a place re-run nothing.
    let report = s
        .apply_patch(
            &patch_of(json!({ "nodes": { "3": { "name": "Sum", "pos": [5, 1] } } })),
            None,
        )
        .unwrap();
    assert_eq!(report.first_group(), None);
    assert_eq!(s.nodes.mixer(3).unwrap().name(), "Sum");
    assert_eq!(s.nodes.pos(id(3)), Some([5.0, 1.0]));
    // Bypass: a Parallel and a Layer have it, a Splitter has not.
    s.apply_patch(
        &patch_of(json!({ "nodes": { "3": { "bypass": true } } })),
        None,
    )
    .unwrap();
    assert!(s.nodes.mixer(3).unwrap().bypass());
    let before = s.clone();
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "4": { "bypass": true } } })),
            None
        )),
        "A splitter has no field called “bypass”."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "3": { "mode": "add" } } })),
            None
        )),
        "A parallel mixer has no field called “mode”."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "2": { "masks": { "invert": true } } } })),
            None
        )),
        "A layer mixer has no field called “masks”."
    );
    assert_eq!(s, before, "a refused patch changes nothing");
    // Removing a node takes its links with it.
    let report = s
        .apply_patch(&patch_of(json!({ "remove": [3, 4, 5] })), None)
        .unwrap();
    assert_eq!(report.removed, vec![id(3), id(4), id(5)]);
    assert_eq!(s.nodes.node_count(), 2);
    let report = s
        .apply_patch(
            &patch_of(json!({ "remove": [2], "links": [
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!("output"), "picture")] })),
            None,
        )
        .unwrap();
    assert_eq!(report.removed, vec![id(2)]);
    assert!(s.nodes.mixer(2).is_none());
    assert_eq!(
        text_of(s.apply_patch(&patch_of(json!({ "remove": [2] })), None)),
        "The patch names node 2, which this photo does not have."
    );
}

#[test]
fn a_patch_with_the_wrong_kind_or_a_gap_is_refused_whole() {
    let mut s = all_four();
    let before = s.clone();
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "4": { "kind": "corrector" } } })),
            None
        )),
        "Node 4 is a layer mixer, not a corrector."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "2": { "kind": "layer" } } })),
            None
        )),
        "Node 2 is a splitter, not a layer mixer."
    );
    assert_eq!(
        text_of(s.apply_patch(
            &patch_of(json!({ "nodes": { "1": { "kind": "combiner" } } })),
            None
        )),
        "Node 1 is a corrector, not a combiner."
    );
    // A ninth input and a gap, in a `links` patch.
    let mut links = s.nodes.links.clone();
    links.retain(|l| !(l.to.node == id(4) && l.to.port == Port::PictureN(1)));
    let gap = SettingsPatch {
        links: Some(links),
        ..SettingsPatch::default()
    };
    assert_eq!(
        text_of(s.apply_patch(&gap, None)),
        "Node 4 has a link on input 2 and none on input 1. A mixer's inputs are numbered from 1 with no gap."
    );
    let nine = patch_of(json!({ "links": [
        link(json!("input"), "picture", json!(1), "picture"),
        link(json!(1), "picture", json!(5), "picture.9"),
        link(json!(5), "picture", json!("output"), "picture")] }));
    assert_eq!(
        text_of(s.apply_patch(&nine, None)),
        "A mixer can have up to 8 inputs."
    );
    assert_eq!(s, before);
}

#[test]
fn undo_is_the_patch_between_two_settings_and_reset_puts_a_layer_back() {
    let a = all_four();
    let mut b = a.clone();
    b.apply_patch(
        &patch_of(json!({ "nodes": {
            "4": { "mode": "lighten", "opacity": 0.9, "bypass": false, "name": "Top" },
            "5": { "bypass": true, "pos": [8, 2] },
            "2": { "name": "Channels" } } })),
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
    // A node that was added or removed is carried whole, with its kind, and the links.
    let empty = DevelopSettings::default();
    let add = SettingsPatch::between(&empty, &b);
    assert_eq!(
        add.nodes.get(&id(4)).and_then(|n| n.kind),
        Some(NodeKind::Layer)
    );
    let mut c = empty.clone();
    c.apply_patch(&add, None).unwrap();
    assert_eq!(c.nodes.mixers, b.nodes.mixers);
    assert_eq!(c.nodes.links, b.nodes.links);
    let drop = SettingsPatch::between(&b, &empty);
    let mut d = b.clone();
    d.apply_patch(&drop, None).unwrap();
    assert_eq!(d, empty);

    // Reset of a Layer: Mode to Normal and Opacity to 1; not the name, Bypass or the place.
    let reset = SettingsPatch::reset_node_in(&a, id(4), &LookId::neutral());
    let mut r = a.clone();
    r.apply_patch(&reset, None).unwrap();
    let l = layer_of(&r, 4);
    assert_eq!((l.mode, l.opacity), (BlendMode::Normal, 1.0));
    assert!(l.bypass && l.name == "Stack" && l.pos == layer_of(&a, 4).pos);
    // Reset of one field.
    let one = SettingsPatch::reset_in(&a, id(4), SettingsField::Opacity, &LookId::neutral());
    let mut r = a.clone();
    r.apply_patch(&one, None).unwrap();
    assert_eq!(layer_of(&r, 4).opacity, 1.0);
    assert_eq!(layer_of(&r, 4).mode, BlendMode::Multiply);
    // A Parallel, a Splitter and a Combiner have nothing to reset: the patch is empty.
    for n in [2, 3, 5] {
        let p = SettingsPatch::reset_node_in(&a, id(n), &LookId::neutral());
        assert!(p.is_empty(), "node {n}: {p:?}");
        assert_eq!(p, SettingsPatch::default(), "node {n}");
    }
}

#[test]
fn a_field_belongs_to_the_kinds_that_have_it() {
    use SettingsField as F;
    for f in F::ALL {
        let layer = f.belongs_to(NodeKind::Layer);
        assert_eq!(
            layer,
            matches!(f, F::Name | F::Bypass | F::Pos | F::Mode | F::Opacity),
            "{f:?}"
        );
        assert_eq!(
            f.belongs_to(NodeKind::Parallel),
            matches!(f, F::Name | F::Bypass | F::Pos),
            "{f:?}"
        );
        for kind in [NodeKind::Splitter, NodeKind::Combiner] {
            assert_eq!(f.belongs_to(kind), matches!(f, F::Name | F::Pos), "{f:?}");
        }
        // The new fields belong to no other kind.
        if matches!(f, F::Mode | F::Opacity) {
            for kind in [
                NodeKind::Input,
                NodeKind::Output,
                NodeKind::Corrector,
                NodeKind::Film,
                NodeKind::Parallel,
                NodeKind::Splitter,
                NodeKind::Combiner,
            ] {
                assert!(!f.belongs_to(kind), "{f:?} {kind:?}");
            }
        }
    }
    assert_eq!(F::Mode.name(), "mode");
    assert_eq!(F::Opacity.path(), "opacity");
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
fn copy_carries_the_nodes_and_paste_replaces_the_targets_graph() {
    let src = all_four();
    let resolved = resolved_of(&src);
    // The resolved settings hold the nodes as they are (they have no Auto).
    assert_eq!(resolved.settings.nodes.mixers, src.nodes.mixers);
    let copy = SettingsPatch::copied_from(&resolved);
    for (n, kind) in [
        (2, NodeKind::Splitter),
        (3, NodeKind::Combiner),
        (4, NodeKind::Layer),
        (5, NodeKind::Parallel),
    ] {
        assert_eq!(
            copy.nodes.get(&id(n)).and_then(|p| p.kind),
            Some(kind),
            "{n}"
        );
    }
    let layer = copy.nodes.get(&id(4)).unwrap();
    assert_eq!(layer.mode, Some(BlendMode::Multiply));
    assert_eq!(layer.opacity, Some(0.4));
    // A target with a node of its own: its node goes, the pasted graph is the source's.
    let mut target = DevelopSettings::default();
    target
        .apply_patch(
            &patch_of(json!({ "nodes": { "7": { "kind": "parallel" } },
            "links": [
                link(json!("input"), "picture", json!(1), "picture"),
                link(json!(1), "picture", json!(7), "picture.1"),
                link(json!(7), "picture", json!("output"), "picture")] })),
            None,
        )
        .unwrap();
    let paste = copy.pasted_onto(&target);
    assert!(paste.remove.contains(&id(7)), "the target's own node goes");
    target.apply_patch(&paste, None).unwrap();
    assert!(target.nodes.mixer(7).is_none());
    assert_eq!(target.nodes.mixers, src.nodes.mixers);
    assert_eq!(target.nodes.links, src.nodes.links);
    assert_eq!(target.nodes, src.nodes);
}

#[test]
fn the_settings_survive_the_xmp_text() {
    // `ym:Settings` is the settings as JSON text; the new kinds are in it and read back whole.
    let s = all_four();
    let text = serde_json::to_string(&s).unwrap();
    for needle in [
        r#""kind":"splitter""#,
        r#""kind":"combiner""#,
        r#""kind":"layer""#,
        r#""kind":"parallel""#,
        r#""mode":"multiply""#,
        r#""opacity":0.4"#,
        r#""port":"picture.2""#,
        r#""port":"green""#,
    ] {
        assert!(text.contains(needle), "{needle} in {text}");
    }
    let back = read(serde_json::from_str(&text).unwrap()).unwrap().0;
    assert_eq!(back, s);
}
