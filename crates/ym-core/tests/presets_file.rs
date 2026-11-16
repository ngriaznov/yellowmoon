//! The preset file (`docs/30` section 5): the same bytes for the same settings, what is refused and with which sentence, what an
//! older file reads as, and the JSON Schema against what the core writes and against the documents it refuses.

mod common;

use std::path::{Path, PathBuf};

use common::rich::rich;
use serde_json::{Value, json};
use ym_core::presets::{
    MAX_PRESET_BYTES, Preset, PresetError, PresetGroup, TOO_LARGE_SENTENCE, TOO_NEW_SENTENCE,
};
use ym_core::settings::{AutoOr, DevelopSettings, ExposureSetting, PercentSetting};
use ym_looks::library::LookId;

const NOT_A_PRESET: &str = "This file isn't a Yellowmoon preset.";

fn preset(groups: &[PresetGroup]) -> Preset {
    Preset::from_settings("Warm evening", &rich(), groups, Some("Film warm")).expect("a preset")
}

fn all() -> Preset {
    preset(&PresetGroup::ALL)
}

fn doc(p: &Preset) -> Value {
    serde_json::from_str(&p.to_json().expect("json")).expect("valid json")
}

fn refused(text: &str) -> String {
    Preset::from_json(text).expect_err("refused").to_string()
}

#[test]
fn the_file_is_compact_json_with_sorted_keys_and_one_newline() {
    let text = all().to_json().expect("json");
    assert!(text.ends_with("}\n") && !text[..text.len() - 1].contains('\n'));
    assert!(!text.contains(": ") && !text.contains(", "), "compact");
    assert!(
        text.starts_with(r#"{"groups":["look","lens","light","color","detail","curve","raw","nodes"],"look":{"id":"user:film-warm","name":"Film warm"},"name":"Warm evening","settings":{"nodes":{"#),
        "{}",
        &text[..200]
    );
    assert!(text.contains(r#""yellowmoon_preset":1"#));
    // Sorted at every level: re-serialising with sorted maps changes nothing.
    fn sorted(v: &Value) -> bool {
        match v {
            Value::Object(m) => {
                let keys: Vec<_> = m.keys().collect();
                let mut s = keys.clone();
                s.sort();
                keys == s && m.values().all(sorted)
            }
            Value::Array(a) => a.iter().all(sorted),
            _ => true,
        }
    }
    let v: Value = serde_json::from_str(&text).expect("json");
    // `serde_json` keeps maps sorted unless a dependency turns on `preserve_order`; the check is on the bytes.
    let mut positions = Vec::new();
    for key in ["groups", "look", "name", "settings", "yellowmoon_preset"] {
        positions.push(text.find(&format!("\"{key}\":")).expect("key"));
    }
    assert!(positions.windows(2).all(|w| w[0] < w[1]));
    assert!(sorted(&v) || !v.is_object());
}

#[test]
fn a_preset_saved_twice_from_the_same_settings_is_the_same_bytes() {
    for groups in [
        &PresetGroup::ALL[..],
        &[PresetGroup::Light],
        &[PresetGroup::Look, PresetGroup::Nodes],
    ] {
        assert_eq!(
            preset(groups).to_json().unwrap(),
            preset(groups).to_json().unwrap()
        );
    }
}

#[test]
fn what_is_read_is_written_again_byte_for_byte() {
    for groups in [
        &PresetGroup::ALL[..],
        &[PresetGroup::Color, PresetGroup::Curve],
        &[PresetGroup::Raw],
        &[PresetGroup::Detail, PresetGroup::Lens, PresetGroup::Look],
    ] {
        let p = preset(groups);
        let text = p.to_json().unwrap();
        let back = Preset::from_json(&text).expect("reads");
        assert_eq!(back.name, p.name);
        assert_eq!(back.groups, p.groups);
        assert_eq!(back.look, p.look);
        assert_eq!(back.to_json().unwrap(), text, "groups {groups:?}");
    }
}

#[test]
fn a_file_holds_only_the_fields_of_its_groups() {
    let d = doc(&preset(&[PresetGroup::Light]));
    let items = d["settings"]["nodes"]["items"].as_array().unwrap();
    assert_eq!(items.len(), 2, "the Input and Main only: {items:?}");
    assert_eq!(items[0]["id"], "input");
    let mut input: Vec<_> = items[0].as_object().unwrap().keys().cloned().collect();
    input.sort();
    assert_eq!(input, ["exposure", "id", "kind"]);
    assert_eq!(items[1]["id"], 1);
    let grade = items[1]["grade"].as_object().unwrap();
    let mut keys: Vec<_> = grade.keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "black",
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "white"
        ]
    );
    assert!(d["settings"]["nodes"].get("links").is_none());
    assert!(
        d.get("look").is_none(),
        "the look is there only with the look group"
    );

    // Nodes brings the other nodes whole, the links and Main's own fields.
    let d = doc(&preset(&[PresetGroup::Nodes]));
    let nodes = &d["settings"]["nodes"];
    assert_eq!(nodes["links"].as_array().unwrap().len(), 4);
    let ids: Vec<_> = nodes["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["id"].clone())
        .collect();
    assert_eq!(
        ids,
        [
            json!("input"),
            json!(1),
            json!(2),
            json!(3),
            json!("output")
        ]
    );
    let main = &nodes["items"][1];
    for key in ["name", "bypass", "trim", "masks", "pos"] {
        assert!(main.get(key).is_some(), "Main's {key} goes with nodes");
    }
    assert!(main.get("grade").is_none());
    // The Input and the Output keep only their position.
    assert_eq!(nodes["items"][0].as_object().unwrap().len(), 3);
}

#[test]
fn a_hand_written_field_outside_the_groups_is_ignored_when_applying_and_left_out_when_saved_again()
{
    let text = r#"{"yellowmoon_preset":1,"name":"Hand","groups":["light"],"settings":{"nodes":{"items":[
        {"id":"input","exposure":{"kind":"manual","ev":0.5},"sharpen":{"kind":"manual","amount":200,"radius_px":1.0}},
        {"id":1,"grade":{"contrast":{"kind":"manual","value":20},"saturation":{"kind":"manual","value":50}}}]}}}"#;
    let p = Preset::from_json(text).expect("reads");
    assert_eq!(
        p.settings.input().sharpen,
        AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 200.0,
            radius_px: 1.0
        })
    );
    let patch = p.patch_for(&DevelopSettings::default(), true).patch;
    let mut target = DevelopSettings::default();
    target.apply_patch(&patch, None).unwrap();
    assert_eq!(target.input().exposure, ExposureSetting::manual_ev(0.5));
    assert_eq!(
        target.main_grade().contrast,
        PercentSetting::manual_value(20.0)
    );
    assert!(target.input().sharpen.is_auto(), "not in the groups");
    assert!(target.main_grade().saturation.is_auto());
    let saved = p.to_json().unwrap();
    assert!(
        !saved.contains("sharpen") && !saved.contains("saturation"),
        "{saved}"
    );
}

#[test]
fn a_newer_version_an_unknown_field_and_an_unknown_settings_field_are_refused_with_the_sentence() {
    let ok = doc(&preset(&[PresetGroup::Light]));
    let with = |f: &dyn Fn(&mut Value)| {
        let mut d = ok.clone();
        f(&mut d);
        d.to_string()
    };
    assert_eq!(
        TOO_NEW_SENTENCE,
        "This preset was made by a newer version of Yellowmoon. Update to use it."
    );
    assert_eq!(
        refused(&with(&|d| d["yellowmoon_preset"] = json!(2))),
        TOO_NEW_SENTENCE
    );
    // The version is read before anything else: a version 2 file of another shape is still "newer".
    assert_eq!(
        refused(r#"{"yellowmoon_preset":2,"name":7}"#),
        TOO_NEW_SENTENCE
    );
    assert_eq!(
        refused(&with(&|d| d["colour"] = json!("warm"))),
        TOO_NEW_SENTENCE
    );
    assert_eq!(
        refused(&with(&|d| {
            d["settings"]["nodes"]["items"][0]["teleport"] = json!(true);
        })),
        TOO_NEW_SENTENCE
    );
    assert!(matches!(
        Preset::from_json(&with(&|d| d["yellowmoon_preset"] = json!(7))),
        Err(PresetError::TooNew)
    ));
}

#[test]
fn an_older_preset_with_fields_missing_reads_with_the_defaults() {
    let text = r#"{"yellowmoon_preset":1,"name":"Old","groups":["light","color"],"settings":{"nodes":{"items":[
        {"id":"input","exposure":{"kind":"manual","ev":-0.3}}]}}}"#;
    let p = Preset::from_json(text).expect("reads");
    assert_eq!(
        p.settings.input().exposure,
        ExposureSetting::manual_ev(-0.3)
    );
    // Everything the file does not have is the default graph's.
    let mut expect = DevelopSettings::default();
    expect.input_mut().exposure = ExposureSetting::manual_ev(-0.3);
    assert_eq!(p.settings, expect);
    assert_eq!(p.groups, [PresetGroup::Light, PresetGroup::Color]);
    // No `look` object is needed without the look group.
    assert!(p.look.is_none());
}

#[test]
fn a_file_that_is_not_a_preset_says_so() {
    let ok = doc(&preset(&[PresetGroup::Light]));
    let with = |f: &dyn Fn(&mut Value)| {
        let mut d = ok.clone();
        f(&mut d);
        d.to_string()
    };
    for text in [
        "".to_owned(),
        "not json".to_owned(),
        "[]".to_owned(),
        "{}".to_owned(),
        r#"{"name":"x","groups":["light"],"settings":{}}"#.to_owned(),
        with(&|d| {
            d.as_object_mut().unwrap().remove("settings");
        }),
        with(&|d| d["name"] = json!("")),
        with(&|d| d["name"] = json!("   ")),
        with(&|d| d["name"] = json!("x".repeat(65))),
        with(&|d| d["name"] = json!(7)),
        with(&|d| d["groups"] = json!([])),
        with(&|d| d["groups"] = json!(["light", "light"])),
        with(&|d| d["groups"] = json!(["warmth"])),
        with(&|d| d["groups"] = json!("light")),
        with(&|d| d["settings"] = json!(5)),
        with(&|d| d["yellowmoon_preset"] = json!(0)),
        with(&|d| d["yellowmoon_preset"] = json!("1")),
        // A look object without the look group, and the look group without its object.
        with(&|d| d["look"] = json!({ "id": "builtin:neutral", "name": "Neutral" })),
        with(&|d| d["groups"] = json!(["look", "light"])),
    ] {
        assert_eq!(refused(&text), NOT_A_PRESET, "{text}");
    }
    // A flat settings document of Yellowmoon 0.1 inside `settings` (no preset was ever saved in that form).
    let flat = with(&|d| {
        d["settings"] = json!({ "exposure": { "kind": "auto" }, "contrast": { "kind": "auto" } })
    });
    assert_eq!(refused(&flat), NOT_A_PRESET);
}

#[test]
fn a_file_above_4_mib_is_too_large() {
    let big = format!(
        r#"{{"yellowmoon_preset":1,"name":"Big","groups":["light"],"settings":{{}},"pad":"{}"}}"#,
        "x".repeat(MAX_PRESET_BYTES)
    );
    assert_eq!(refused(&big), TOO_LARGE_SENTENCE);
    assert_eq!(TOO_LARGE_SENTENCE, "This file is too large to be a preset.");
}

#[test]
fn numbers_out_of_range_are_clamped_like_any_settings_document() {
    let text = r#"{"yellowmoon_preset":1,"name":"Loud","groups":["light"],"settings":{"nodes":{"items":[
        {"id":"input","exposure":{"kind":"manual","ev":40}},
        {"id":1,"grade":{"brightness":900}}]}}}"#;
    let p = Preset::from_json(text).expect("reads");
    assert_eq!(p.settings.input().exposure, ExposureSetting::manual_ev(4.0));
    assert_eq!(p.settings.main_grade().brightness, 100.0);
}

#[test]
fn the_looks_id_is_the_settings_own_and_the_name_is_kept() {
    let p = preset(&[PresetGroup::Look]);
    let look = p.look.as_ref().expect("a look");
    assert_eq!(look.id, LookId::parse("user:film-warm").unwrap());
    assert_eq!(look.name, "Film warm");
    assert_eq!(p.settings.output().look, look.id);
    // With no name given the id stands in.
    let q = Preset::from_settings("Q", &rich(), &[PresetGroup::Look], None).unwrap();
    assert_eq!(q.look.unwrap().name, "user:film-warm");
}

#[test]
fn names_are_trimmed_and_checked_and_a_preset_needs_a_group() {
    let q =
        Preset::from_settings("  Warm evening \t", &rich(), &[PresetGroup::Light], None).unwrap();
    assert_eq!(q.name, "Warm evening");
    let e = |name: &str, groups: &[PresetGroup]| {
        Preset::from_settings(name, &rich(), groups, None)
            .expect_err("refused")
            .to_string()
    };
    assert_eq!(e("  ", &[PresetGroup::Light]), "Give the preset a name.");
    assert_eq!(
        e(&"é".repeat(65), &[PresetGroup::Light]),
        "Use 64 characters or fewer."
    );
    assert!(Preset::from_settings(&"é".repeat(64), &rich(), &[PresetGroup::Light], None).is_ok());
    assert!(e("a\u{7}b", &[PresetGroup::Light]).contains("control"));
    assert_eq!(e("Fine", &[]), "Tick at least one group.");
    // The groups are a sorted set.
    let s = Preset::from_settings(
        "S",
        &rich(),
        &[PresetGroup::Nodes, PresetGroup::Look, PresetGroup::Nodes],
        None,
    )
    .unwrap();
    assert_eq!(s.groups, [PresetGroup::Look, PresetGroup::Nodes]);
}

// ---------------------------------------------------------------------------------------------
// The JSON Schema
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

/// The preset schema with the settings schema registered under the address its `$ref` resolves to.
fn validator() -> jsonschema::Validator {
    let read = |name: &str| -> Value {
        let path = repo_root().join("crates/ym-core/schema").join(name);
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };
    let registry = jsonschema::Registry::new()
        .add(
            "https://yellowmoon.app/schema/settings.schema.json",
            read("settings.schema.json"),
        )
        .expect("a uri")
        .prepare()
        .expect("a registry");
    jsonschema::options()
        .with_registry(&registry)
        .build(&read("preset.schema.json"))
        .expect("the schema is a valid schema")
}

fn problems(v: &jsonschema::Validator, doc: &Value) -> Vec<String> {
    v.iter_errors(doc)
        .take(5)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect()
}

#[test]
fn what_the_core_writes_validates_against_the_schema() {
    let v = validator();
    for groups in [
        &PresetGroup::ALL[..],
        &[PresetGroup::Look],
        &[PresetGroup::Lens, PresetGroup::Light],
        &[PresetGroup::Color],
        &[PresetGroup::Detail, PresetGroup::Curve, PresetGroup::Raw],
        &[PresetGroup::Nodes],
    ] {
        let d = doc(&preset(groups));
        assert!(
            problems(&v, &d).is_empty(),
            "{groups:?}: {:?}",
            problems(&v, &d)
        );
    }
}

#[test]
fn the_schema_refuses_what_the_core_refuses() {
    let v = validator();
    let ok = doc(&preset(&[PresetGroup::Look, PresetGroup::Light]));
    assert!(problems(&v, &ok).is_empty());
    let with = |f: &dyn Fn(&mut Value)| {
        let mut d = ok.clone();
        f(&mut d);
        d
    };
    let bad = [
        with(&|d| d["yellowmoon_preset"] = json!(2)),
        with(&|d| d["colour"] = json!(1)),
        with(&|d| {
            d.as_object_mut().unwrap().remove("settings");
        }),
        with(&|d| {
            d.as_object_mut().unwrap().remove("look");
        }),
        with(&|d| d["name"] = json!("")),
        with(&|d| d["name"] = json!("x".repeat(65))),
        with(&|d| d["groups"] = json!([])),
        with(&|d| d["groups"] = json!(["look", "look"])),
        with(&|d| d["groups"] = json!(["warmth"])),
        with(&|d| {
            d["groups"] = json!([
                "look", "light", "color", "lens", "detail", "curve", "raw", "nodes", "look"
            ])
        }),
        with(&|d| d["groups"] = json!(["light"])),
        with(&|d| d["look"] = json!({ "id": "", "name": "x" })),
        with(&|d| d["look"] = json!({ "id": "a:b" })),
        with(&|d| d["settings"]["exposure"] = json!(1)),
        with(&|d| d["settings"]["nodes"]["items"][0]["teleport"] = json!(1)),
    ];
    for (i, d) in bad.iter().enumerate() {
        assert!(!problems(&v, d).is_empty(), "document {i} validates: {d}");
    }
}
