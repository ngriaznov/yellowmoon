//! M1-04: the shipped data files. Every `profile.json`, the registry and the target validate against their
//! JSON schemas (`crates/ym-profiles/schema/`) and load into the Rust types; the registry round-trips
//! (every model ID maps to exactly one profile, every referenced profile exists and loads, the default is
//! present); the registry equals the table in `docs/03`.

mod common;

use std::path::Path;

use common::{data_root, docs_table, read_json, repo_root, store};
use jsonschema::Validator;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use ym_profiles::profile::{Profile, ProfileError, Target};
use ym_profiles::registry::{
    BodyMatch, IDENTITY_WHITE_SCALE, ProfileStore, Registry, RegistryError, WhiteScaleKind,
};

fn validator(schema_file: &str) -> Validator {
    let path = repo_root()
        .join("crates/ym-profiles/schema")
        .join(schema_file);
    let schema = read_json(&path);
    jsonschema::validator_for(&schema)
        .unwrap_or_else(|e| panic!("{schema_file} is not a valid schema: {e}"))
}

fn errors(v: &Validator, instance: &Value) -> Vec<String> {
    v.iter_errors(instance)
        .take(8)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect()
}

fn assert_valid(v: &Validator, instance: &Value, label: &str) {
    let errs = errors(v, instance);
    assert!(
        errs.is_empty(),
        "{label} does not validate:\n{}",
        errs.join("\n")
    );
}

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn profile_files() -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(data_root().join("cameras/profiles"))
        .unwrap()
        .map(|e| e.unwrap().path().join("profile.json"))
        .collect();
    files.sort();
    files
}

#[test]
fn every_shipped_profile_validates_against_the_schema() {
    let v = validator("profile.schema.json");
    let files = profile_files();
    assert_eq!(files.len(), 11);
    for file in files {
        let doc = read_json(&file);
        assert_valid(&v, &doc, &file.display().to_string());
    }
}

#[test]
fn the_registry_and_the_target_validate_against_their_schemas() {
    assert_valid(
        &validator("registry.schema.json"),
        &read_json(&data_root().join("cameras/registry.json")),
        "registry.json",
    );
    assert_valid(
        &validator("target.schema.json"),
        &read_json(&data_root().join("alexa/target.json")),
        "target.json",
    );
}

#[test]
fn the_schemas_reject_broken_documents() {
    // Proves the validator is live: each mutation of a shipped file must be refused.
    let pv = validator("profile.schema.json");
    let profile = read_json(&data_root().join("cameras/profiles/sony-ilce-7rm4/profile.json"));
    type Mutation = Box<dyn Fn(&mut Value)>;
    let mutations: Vec<(&str, Mutation)> = vec![
        (
            "missing locus",
            Box::new(|d| {
                d.as_object_mut().unwrap().remove("locus");
            }),
        ),
        (
            "unknown top-level field",
            Box::new(|d| {
                d["extra"] = json!(1);
            }),
        ),
        (
            "wrong method",
            Box::new(|d| {
                d["sensor_map"]["method"] = json!("mlp");
            }),
        ),
        (
            "green not 1 in w_A",
            Box::new(|d| {
                d["alexa_white"]["led"][0]["w"][1] = json!(0.99);
            }),
        ),
        (
            "short matrix",
            Box::new(|d| {
                d["sensor_map"]["classes"]["led"][0]["m"] = json!([[1, 0, 0], [0, 1, 0]]);
            }),
        ),
        (
            "one-entry class",
            Box::new(|d| {
                let c = d["sensor_map"]["classes"]["tungsten"]
                    .as_array_mut()
                    .unwrap();
                c.truncate(1);
            }),
        ),
        (
            "bad version",
            Box::new(|d| {
                d["version"] = json!("1.0");
            }),
        ),
        (
            "bad id",
            Box::new(|d| {
                d["id"] = json!("Sony ILCE");
            }),
        ),
        (
            "sign convention",
            Box::new(|d| {
                d["locus"]["duv_sign"] = json!("positive toward green");
            }),
        ),
        (
            "negative Duv scale",
            Box::new(|d| {
                d["locus"]["duv_scale"]["daylight"]["values"][0] = json!(-0.1);
            }),
        ),
        (
            "curves outside sony/",
            Box::new(|d| {
                d["spectral_curves"] = json!("../etc/passwd");
            }),
        ),
        (
            "decode quirks not strings",
            Box::new(|d| {
                d["decode"]["quirks"] = json!([1]);
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let mut doc = profile.clone();
        mutate(&mut doc);
        assert!(!pv.is_valid(&doc), "the profile schema accepted: {name}");
    }
    let rv = validator("registry.schema.json");
    let registry = read_json(&data_root().join("cameras/registry.json"));
    let mut doc = registry.clone();
    doc["bodies"][0]["match"] = json!("similar");
    assert!(!rv.is_valid(&doc));
    let mut doc = registry.clone();
    doc["bodies"][1]["via"] = Value::Null; // a sensor_family body needs `via`
    assert!(!rv.is_valid(&doc));
    let mut doc = registry;
    doc["bodies"][0]["via"] = json!("Sony A7 III"); // an exact body must have null `via`
    assert!(!rv.is_valid(&doc));
    let tv = validator("target.schema.json");
    let mut doc = read_json(&data_root().join("alexa/target.json"));
    doc["kind"] = json!("guess");
    assert!(!tv.is_valid(&doc));
}

#[test]
fn the_rust_types_write_back_documents_the_schemas_accept() {
    // Typed round trip: parse -> serialise -> validate. Catches a field the types drop or rename.
    let pv = validator("profile.schema.json");
    for file in profile_files() {
        let profile = Profile::from_path(&file).unwrap_or_else(|e| panic!("{e}"));
        let value = serde_json::to_value(&profile).unwrap();
        assert_valid(&pv, &value, &format!("{} (re-serialised)", profile.id));
        // And the typed form of the written document is the same profile (floats may differ by an ulp).
        let again: Profile = serde_json::from_value(value).unwrap();
        assert_eq!(again.id, profile.id);
        assert_eq!(again.version, profile.version);
        assert_eq!(
            again.sensor_map.classes.led.len(),
            profile.sensor_map.classes.led.len()
        );
    }
    let registry = Registry::parse(
        &std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap(),
    )
    .unwrap();
    assert_valid(
        &validator("registry.schema.json"),
        &serde_json::to_value(&registry).unwrap(),
        "registry (re-serialised)",
    );
    let target = Target::from_path(data_root().join("alexa/target.json")).unwrap();
    assert_valid(
        &validator("target.schema.json"),
        &serde_json::to_value(&target).unwrap(),
        "target (re-serialised)",
    );
}

#[test]
fn the_store_loads_everything_once() {
    let started = std::time::Instant::now();
    let s = ProfileStore::load(data_root()).unwrap();
    eprintln!(
        "ProfileStore::load: {} profiles in {:.1} ms",
        s.profiles().count(),
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(s.profiles().count(), 11);
    assert_eq!(s.own_profile_count(), 11);
    assert_eq!(s.default_profile().id.as_str(), "sony-ilce-7rm4");
    assert_eq!(s.default_profile().display_name, "A7R IV");
    assert_eq!(s.registry().default_profile, s.default_profile().id);
    assert!(s.target().is_estimate());
    assert_eq!(s.target().id, "alexa-alev3-estimate");
    assert_eq!(s.data_root(), data_root());
    // Ordered by id; lookups by id and by name agree.
    let ids: Vec<&str> = s.profile_ids().map(|i| i.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    for p in s.profiles() {
        assert!(std::sync::Arc::ptr_eq(
            s.profile(&p.id).unwrap(),
            s.profile_by_name(p.id.as_str()).unwrap()
        ));
    }
    assert!(s.profile_by_name("sony-ilce-1").is_none());
}

#[test]
fn registry_round_trip() {
    let s = store();
    let reg = s.registry();
    // Every model ID maps to exactly one profile, and model IDs are unique.
    let mut ids: Vec<u16> = reg.bodies.iter().map(|b| b.model_id).collect();
    ids.sort_unstable();
    let n = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), n, "duplicate model IDs");
    assert_eq!(n, 28);
    for b in &reg.bodies {
        let hits = reg
            .bodies
            .iter()
            .filter(|o| o.model_id == b.model_id)
            .count();
        assert_eq!(hits, 1);
        // The referenced profile exists, is listed, and loads with the id the registry gives it.
        let p = s
            .profile(&b.profile)
            .unwrap_or_else(|| panic!("{} uses missing profile {}", b.model, b.profile));
        assert_eq!(p.id, b.profile);
        assert!(reg.profile_entry(&b.profile).is_some());
        // The model string resolves back to the same ID, and so does the ID to the same body.
        assert_eq!(reg.model_strings[&b.model], b.model_id);
        assert_eq!(reg.body_by_model_id(b.model_id).unwrap().model, b.model);
        assert_eq!(
            reg.body_by_model_string(&b.model).unwrap().model_id,
            b.model_id
        );
    }
    // Every listed profile has a file whose folder is its id, exactly one exact body, and is used.
    for entry in &reg.profiles {
        assert!(
            data_root().join("cameras").join(&entry.path).is_file(),
            "{}",
            entry.path
        );
        assert_eq!(
            Path::new(&entry.path)
                .parent()
                .unwrap()
                .file_name()
                .unwrap(),
            entry.id.as_str()
        );
        let exact = reg
            .bodies
            .iter()
            .filter(|b| b.profile == entry.id && b.match_kind == BodyMatch::Exact)
            .count();
        assert_eq!(exact, 1, "{}", entry.id);
    }
    // Defaults are present: the default profile is listed and loaded, and every `default` body uses it.
    assert!(s.profile(&reg.default_profile).is_some());
    for b in reg
        .bodies
        .iter()
        .filter(|b| b.match_kind == BodyMatch::Default)
    {
        assert_eq!(b.profile, reg.default_profile, "{}", b.model);
    }
    // The default bodies of docs/03: the five without a measured sibling, and the A7R V (390) and
    // A7CR (396), whose same-sensor pairing with the A7R IV failed the ColorMatrix check.
    let defaults: Vec<u16> = reg
        .bodies
        .iter()
        .filter(|b| b.match_kind == BodyMatch::Default)
        .map(|b| b.model_id)
        .collect();
    assert_eq!(defaults, vec![390, 396, 384, 400, 392, 407, 410]);
}

#[test]
fn the_registry_equals_the_table_in_docs_03() {
    let table = docs_table();
    assert!(
        table.len() >= 25,
        "parsed only {} rows of docs/03",
        table.len()
    );
    let reg = store().registry();
    let ours: std::collections::BTreeMap<u16, (String, String)> = reg
        .bodies
        .iter()
        .map(|b| {
            let kind = match b.match_kind {
                BodyMatch::Exact => "exact",
                BodyMatch::SensorFamily => "sensor_family",
                BodyMatch::Default => "default",
            };
            (b.model_id, (b.profile.to_string(), kind.to_string()))
        })
        .collect();
    assert_eq!(ours, table);
}

#[test]
fn files_named_by_the_profiles_exist_and_match_their_provenance() {
    let s = store();
    let root = data_root();
    for p in s.profiles() {
        let curves = p.curves_path(&root);
        assert_eq!(
            sha256_file(&curves),
            p.provenance.curves_source.sha256,
            "{}: curves hash",
            p.id
        );
        assert_eq!(
            sha256_file(&root.join(&p.provenance.target.file)),
            p.provenance.target.sha256,
            "{}: target hash",
            p.id
        );
        assert_eq!(p.sensor_map.target, "alexa/target.json");
        assert!(root.join(&p.sensor_map.target).is_file());
        // The curve file is the body's own measurement: its model matches the profile.
        let curves = p.load_curves(&root).unwrap();
        assert_eq!(curves.len(), 81);
        let exact = s.registry().exact_body(&p.id).unwrap();
        assert_eq!(curves.model(), exact.model, "{}", p.id);
        assert_eq!(p.version, semver::Version::new(1, 0, 0));
    }
    // The target's files exist and match the hashes it records.
    let t = s.target();
    assert_eq!(t.version, semver::Version::new(1, 0, 0));
    for (name, digest) in &t.provenance.sha256 {
        assert_eq!(
            &sha256_file(&root.join("alexa").join(name)),
            digest,
            "{name}"
        );
    }
    assert!(t.curves_path(&root).is_file());
    assert!(root.join(&t.ensemble_csv).is_file() && root.join(&t.variants_csv).is_file());
    assert!(!t.provenance.caveats.is_empty());
}

/// The largest per-coefficient difference a recorded check states: `pass: max |diff| 0.0000 per
/// coefficient (limit 0.02); ...` gives 0.0.
fn recorded_diff(result: &str) -> (bool, f64) {
    let (verdict, rest) = result
        .split_once(": max |diff| ")
        .unwrap_or_else(|| panic!("no recorded value in {result:?}"));
    let value = rest
        .split_whitespace()
        .next()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no number after max |diff| in {result:?}"));
    (verdict == "pass", value)
}

#[test]
fn every_same_sensor_pairing_has_a_recorded_check() {
    // docs/03: "Record the check in the profile's `provenance`." V1 runs it on the bodies' published
    // colour matrices (`tests/py/sibling_matrices.py`, which `--verify`s these numbers against a fresh run):
    // every same-sensor body needs a numeric result within 0.02 and the record of what was compared.
    let s = store();
    let mut pairings = 0;
    for b in s
        .registry()
        .bodies
        .iter()
        .filter(|b| b.match_kind == BodyMatch::SensorFamily)
    {
        let p = s.profile(&b.profile).unwrap();
        let needle = format!("{} ({})", b.model, b.model_id);
        let check = p
            .provenance
            .checks
            .iter()
            .find(|c| c.pairing.starts_with(&needle))
            .unwrap_or_else(|| panic!("{}: no provenance check for {needle}", p.id));
        assert!(
            !check.result.starts_with("pending"),
            "{needle}: {}",
            check.result
        );
        let (passed, diff) = recorded_diff(&check.result);
        assert!(passed, "{needle}: {}", check.result);
        assert!((0.0..=0.02).contains(&diff), "{needle}: max |diff| {diff}");
        assert!(
            check.result.contains("published D65 colour matrices"),
            "{needle}: what was compared is recorded: {}",
            check.result
        );
        assert!(
            check.check.contains("0.02"),
            "{needle}: the limit is recorded: {}",
            check.check
        );
        pairings += 1;
    }
    assert_eq!(pairings, 10);
    // The two pairings that failed (A7R V and A7CR against the A7R IV) do not ship as siblings: the
    // body has match kind default, and the failed measurement stays in the A7R IV profile.
    let a7r4 = s.profile_by_name("sony-ilce-7rm4").unwrap();
    for (model, id) in [("ILCE-7RM5", 390u16), ("ILCE-7CR", 396)] {
        let body = s.registry().body_by_model_id(id).unwrap();
        assert_eq!(body.model, model);
        assert_eq!(body.match_kind, BodyMatch::Default, "{model}");
        assert_eq!(
            body.profile, a7r4.id,
            "{model} still renders with the A7R IV profile"
        );
        assert!(
            body.notes.contains("did not pass") && body.notes.contains("0.0934"),
            "{}",
            body.notes
        );
        let check = a7r4
            .provenance
            .checks
            .iter()
            .find(|c| {
                c.pairing
                    .starts_with(&format!("{model} ({id}) was considered"))
            })
            .unwrap_or_else(|| panic!("no failed check recorded for {model}"));
        let (passed, diff) = recorded_diff(&check.result);
        assert!(!passed && diff > 0.02, "{}", check.result);
        assert!(diff > 0.09 && diff < 0.095, "{}", check.result);
    }
}

#[test]
fn bodies_carry_published_sensor_geometry_with_its_source() {
    // docs/algorithms/03 section 2.2: `sensor_width_mm` and `active_width_px` per body, the source in
    // the entry. 27 of 28 bodies have them; the FX3 has none (Sony gives its stills only as "12M").
    let s = store();
    let reg = s.registry();
    let mut with = 0;
    for b in &reg.bodies {
        match b.sensor() {
            None => {
                assert_eq!(b.model, "ILME-FX3");
                assert!(b.sensor_source.is_none() && b.sensor_source_kind.is_none());
            }
            Some(g) => {
                with += 1;
                assert!(
                    b.sensor_source.as_deref().is_some_and(|t| t.len() > 20),
                    "{}: the source is recorded",
                    b.model
                );
                assert!(b.sensor_source_kind.is_some());
                let pitch = 1000.0 * g.sensor_width_mm / f64::from(g.active_width_px);
                assert_eq!(g.pixel_pitch_um, pitch, "{}", b.model);
                assert!((0.5..=20.0).contains(&pitch), "{}: {pitch}", b.model);
            }
        }
    }
    assert_eq!(with, 27);
    // Known pitches (the maker's figure for the A7R IV is 3.76 um, the A7 III 5.93 um, the A7S III
    // 8.4 um, the A1 4.16 um, the RX100 IV 2.4 um).
    let pitch = |id: u16| {
        reg.body_by_model_id(id)
            .unwrap()
            .sensor()
            .unwrap()
            .pixel_pitch_um
    };
    for (id, want) in [
        (375u16, 3.76),
        (363, 5.93),
        (383, 8.4),
        (384, 4.16),
        (341, 2.41),
        (371, 3.92),
    ] {
        assert!(
            (pitch(id) - want).abs() < 0.01,
            "{id}: {} vs {want}",
            pitch(id)
        );
    }
    // Bodies of one sensor agree on its pitch within 2 %.
    for b in reg
        .bodies
        .iter()
        .filter(|b| b.match_kind == BodyMatch::SensorFamily)
    {
        let (Some(own), Some(base)) = (
            b.sensor(),
            reg.exact_body(&b.profile).and_then(|e| e.sensor()),
        ) else {
            continue;
        };
        assert!(
            (own.pixel_pitch_um / base.pixel_pitch_um - 1.0).abs() < 0.02,
            "{}",
            b.model
        );
    }
    // The detection result carries it (and a body without it has none).
    let m = s.detect("SONY", "ILCE-7RM4", Some(375));
    assert_eq!(m.sensor.unwrap().active_width_px, 9504);
    assert!(s.detect("SONY", "ILME-FX3", Some(385)).sensor.is_none());
    assert!(s.detect("SONY", "ILCE-7M6", Some(60000)).sensor.is_none());
    // A forced profile keeps the sensor of the body.
    let forced = s
        .detect_forced("SONY", "ILCE-7RM4", Some(375), "sony-ilce-7m3")
        .unwrap();
    assert_eq!(forced.sensor, m.sensor);
}

#[test]
fn the_registry_schema_and_rules_refuse_broken_sensor_fields() {
    let rv = validator("registry.schema.json");
    let text = std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap();
    let registry: Value = serde_json::from_str(&text).unwrap();
    let a7r4 = registry["bodies"]
        .as_array()
        .unwrap()
        .iter()
        .position(|b| b["model_id"] == 375)
        .unwrap();
    type Mutation = Box<dyn Fn(&mut Value)>;
    let broken: Vec<(&str, Mutation)> = vec![
        (
            "width without the pixel count",
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("active_width_px");
            }),
        ),
        (
            "pixel count without the width",
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("sensor_width_mm");
            }),
        ),
        (
            "no source",
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("sensor_source");
            }),
        ),
        (
            "no source kind",
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("sensor_source_kind");
            }),
        ),
        (
            "unknown source kind",
            Box::new(|b| {
                b["sensor_source_kind"] = json!("guess");
            }),
        ),
        (
            "empty source",
            Box::new(|b| {
                b["sensor_source"] = json!("");
            }),
        ),
        (
            "negative width",
            Box::new(|b| {
                b["sensor_width_mm"] = json!(-35.7);
            }),
        ),
        (
            "fractional pixel count",
            Box::new(|b| {
                b["active_width_px"] = json!(9504.5);
            }),
        ),
        (
            "pitch override outside 0.5 to 20",
            Box::new(|b| {
                b["pixel_pitch_um"] = json!(25.0);
            }),
        ),
    ];
    for (name, mutate) in broken {
        let mut doc = registry.clone();
        mutate(&mut doc["bodies"][a7r4]);
        assert!(!rv.is_valid(&doc), "the registry schema accepted: {name}");
    }
    // The pitch override is allowed and the typed value uses it.
    let mut doc = registry.clone();
    doc["bodies"][a7r4]["pixel_pitch_um"] = json!(3.8);
    assert!(rv.is_valid(&doc));
    let parsed = Registry::parse(&doc.to_string()).unwrap();
    assert_eq!(parsed.bodies[a7r4].sensor().unwrap().pixel_pitch_um, 3.8);
    // A width over the schema's maximum of 100 mm is refused by the schema.
    let mut doc = registry.clone();
    doc["bodies"][a7r4]["sensor_width_mm"] = json!(400.0);
    assert!(!rv.is_valid(&doc), "400 mm");
    // What the schema cannot say: the pitch the width and pixel count give must be plausible
    // (100 mm over 2000 px is 50 um).
    doc["bodies"][a7r4]["sensor_width_mm"] = json!(100.0);
    doc["bodies"][a7r4]["active_width_px"] = json!(2000);
    assert!(rv.is_valid(&doc));
    let err = Registry::parse(&doc.to_string()).unwrap_err().to_string();
    assert!(
        err.contains("ILCE-7RM4") && err.contains("pixel pitch"),
        "{err}"
    );
    let mut doc = registry;
    doc["bodies"][a7r4]
        .as_object_mut()
        .unwrap()
        .remove("active_width_px");
    let err = Registry::parse(&doc.to_string()).unwrap_err().to_string();
    assert!(err.contains("come together"), "{err}");
}

#[test]
fn white_scales_are_recorded_with_their_source() {
    // docs/03, White scale of stand-in bodies: a body with a profile of its own or of its sensor has [1, 1, 1];
    // a stand-in body has the scale fitted on the presets of its own files, a sibling's, the scale of its own
    // published colour matrix, or [1, 1, 1] with the statement that nothing was measured. Every body says which.
    let reg = store().registry();
    let (mut own, mut same, mut presets, mut sibling, mut matrix) = (0, 0, 0, 0, 0);
    for b in &reg.bodies {
        let src = b
            .white_scale_source
            .as_ref()
            .unwrap_or_else(|| panic!("{}: no white_scale_source", b.model));
        assert!(src.note.len() > 20, "{}: a note", b.model);
        assert_eq!(b.white_scale[1], 1.0, "{}", b.model);
        match (b.match_kind, src.kind) {
            (BodyMatch::Exact, WhiteScaleKind::OwnCurves) => {
                own += 1;
                assert_eq!(b.white_scale, IDENTITY_WHITE_SCALE, "{}", b.model);
            }
            (BodyMatch::SensorFamily, WhiteScaleKind::SameSensor) => {
                same += 1;
                assert_eq!(b.white_scale, IDENTITY_WHITE_SCALE, "{}", b.model);
            }
            (BodyMatch::Default, WhiteScaleKind::Presets) => presets += 1,
            (BodyMatch::Default, WhiteScaleKind::Sibling) => sibling += 1,
            (BodyMatch::Default, WhiteScaleKind::Matrix) => {
                matrix += 1;
                assert_ne!(b.white_scale, IDENTITY_WHITE_SCALE, "{}", b.model);
                assert!(
                    src.files.is_empty() && src.residual.is_none(),
                    "{}",
                    b.model
                );
                assert!(
                    src.note.contains("published colour matrix at D65"),
                    "{}",
                    b.model
                );
            }
            (m, k) => panic!("{}: match {m:?} with white scale source {k:?}", b.model),
        }
    }
    assert_eq!((own, same), (11, 10));
    assert_eq!((presets, sibling), (1, 1), "the A1, and the A1 II after it");
    assert_eq!(
        matrix, 5,
        "the A7R V, A7CR, A9 III, A7 V and A7R VI: no file, so their published matrices"
    );

    // The A1: fitted on the presets stored in the owner's three photos.
    let a1 = reg.body_by_model_id(384).unwrap();
    let src = a1.white_scale_source.as_ref().unwrap();
    assert_eq!(src.kind, WhiteScaleKind::Presets);
    let mut files = src.files.clone();
    files.sort();
    assert_eq!(files, ["A1M01454.ARW", "A1M01739.ARW", "A1M01749.ARW"]);
    let [r, g, b] = a1.white_scale;
    assert!((0.80..0.95).contains(&r) && g == 1.0 && (1.0..1.2).contains(&b));
    // The limits of docs/03: the presets read within 2 degrees of their labels on average and the Daylight preset
    // reads between 5000 and 5600 K; the matrix reference does not do either, which is why it did not decide.
    let res = src.residual.as_ref().unwrap();
    assert!(res.mean_preset_angle_before_deg > 2.0 * res.mean_preset_angle_after_deg);
    assert!(res.mean_preset_angle_after_deg <= 2.0);
    assert!((5000.0..=5600.0).contains(&res.daylight_k_after));
    assert!(res.daylight_k_before < 4600.0);
    assert!(res.matrix_mean_preset_angle_deg > 2.0 || res.matrix_daylight_k < 5000.0);
    assert_eq!(res.matrix_scale[1], 1.0);

    // The A1 II takes the A1's scale by the owner's decision of 2026-10-19 (the two share a sensor), although
    // its own published colour matrix is 0.1672 from the A1's, over the 0.02 of the matrix rule; its note says
    // both.
    let a1m2 = reg.body_by_model_id(400).unwrap();
    let src = a1m2.white_scale_source.as_ref().unwrap();
    assert_eq!(src.kind, WhiteScaleKind::Sibling);
    assert_eq!(a1m2.white_scale, a1.white_scale);
    for word in ["Sony A1:", "owner's decision of 2026-10-19", "0.1672"] {
        assert!(src.note.contains(word), "{word}: {}", src.note);
    }
}

#[test]
fn the_registry_schema_and_rules_refuse_broken_white_scales() {
    let rv = validator("registry.schema.json");
    let text = std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap();
    let registry: Value = serde_json::from_str(&text).unwrap();
    let index = |id: u64| {
        registry["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .position(|b| b["model_id"] == id)
            .unwrap()
    };
    let (a7r4, a1, a7c, rm5, a1m2) = (index(375), index(384), index(381), index(390), index(400));
    type Mutation = Box<dyn Fn(&mut Value)>;
    // (name, body index, mutation)
    let broken: Vec<(&str, usize, Mutation)> = vec![
        (
            "no white scale",
            a1,
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("white_scale");
            }),
        ),
        (
            "no source",
            a1,
            Box::new(|b| {
                b.as_object_mut().unwrap().remove("white_scale_source");
            }),
        ),
        (
            "green not 1",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([0.87, 1.02, 1.09]);
            }),
        ),
        (
            "two numbers",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([0.87, 1.0]);
            }),
        ),
        (
            "a string",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([0.87, 1.0, "1.09"]);
            }),
        ),
        (
            "zero",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([0.0, 1.0, 1.09]);
            }),
        ),
        (
            "negative",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([0.87, 1.0, -1.09]);
            }),
        ),
        (
            "five times",
            a1,
            Box::new(|b| {
                b["white_scale"] = json!([5.0, 1.0, 1.09]);
            }),
        ),
        (
            "an exact body with a scale",
            a7r4,
            Box::new(|b| {
                b["white_scale"] = json!([0.9, 1.0, 1.1]);
            }),
        ),
        (
            "a same-sensor body with a scale",
            a7c,
            Box::new(|b| {
                b["white_scale"] = json!([0.9, 1.0, 1.1]);
            }),
        ),
        (
            "an exact body that says presets",
            a7r4,
            Box::new(|b| {
                b["white_scale_source"]["kind"] = json!("presets");
            }),
        ),
        (
            "a stand-in that says own curves",
            a1,
            Box::new(|b| {
                b["white_scale_source"]["kind"] = json!("own_curves");
            }),
        ),
        (
            "unknown kind",
            a1,
            Box::new(|b| {
                b["white_scale_source"]["kind"] = json!("guess");
            }),
        ),
        (
            "none with a scale",
            rm5,
            Box::new(|b| {
                b["white_scale_source"]["kind"] = json!("none");
            }),
        ),
        (
            "a matrix scale of [1, 1, 1]",
            rm5,
            Box::new(|b| {
                b["white_scale"] = json!([1.0, 1.0, 1.0]);
            }),
        ),
        (
            "a sibling of [1, 1, 1]",
            a1m2,
            Box::new(|b| {
                b["white_scale"] = json!([1.0, 1.0, 1.0]);
            }),
        ),
        (
            "presets without residuals",
            a1,
            Box::new(|b| {
                b["white_scale_source"]
                    .as_object_mut()
                    .unwrap()
                    .remove("residual");
            }),
        ),
        (
            "presets without files",
            a1,
            Box::new(|b| {
                b["white_scale_source"]["files"] = json!([]);
            }),
        ),
        (
            "a residual on a matrix scale",
            rm5,
            Box::new(|b| {
                b["white_scale_source"]["residual"] = json!({
                    "mean_preset_angle_before_deg": 1.0, "mean_preset_angle_after_deg": 1.0,
                    "daylight_k_before": 5000, "daylight_k_after": 5000, "matrix_scale": [1, 1, 1],
                    "matrix_mean_preset_angle_deg": 1.0, "matrix_daylight_k": 5000
                });
            }),
        ),
        (
            "an unknown field in the source",
            a1,
            Box::new(|b| {
                b["white_scale_source"]["tool"] = json!("x");
            }),
        ),
        (
            "an empty note",
            a1,
            Box::new(|b| {
                b["white_scale_source"]["note"] = json!("");
            }),
        ),
    ];
    for (name, body, mutate) in &broken {
        let mut doc = registry.clone();
        mutate(&mut doc["bodies"][*body]);
        assert!(!rv.is_valid(&doc), "the registry schema accepted: {name}");
        // The Rust rules refuse these too (or the types fail to parse them).
        assert!(
            Registry::parse(&doc.to_string()).is_err(),
            "the typed registry accepted: {name}"
        );
    }
    // A registry written before the scale existed (no fields at all) still loads: every body is the identity.
    let mut old = registry.clone();
    for b in old["bodies"].as_array_mut().unwrap() {
        let o = b.as_object_mut().unwrap();
        o.remove("white_scale");
        o.remove("white_scale_source");
    }
    let parsed = Registry::parse(&old.to_string()).unwrap();
    assert!(
        parsed
            .bodies
            .iter()
            .all(|b| b.white_scale == IDENTITY_WHITE_SCALE && b.white_scale_source.is_none())
    );
    // ... and the schema asks for the fields, so the shipped file is never in that form.
    assert!(!rv.is_valid(&old));
}

/// Copies the shipped registry, target and profile files into a temporary data root.
fn copy_data(dest: &Path) {
    let src = data_root();
    std::fs::create_dir_all(dest.join("cameras/profiles")).unwrap();
    std::fs::create_dir_all(dest.join("alexa")).unwrap();
    std::fs::copy(
        src.join("cameras/registry.json"),
        dest.join("cameras/registry.json"),
    )
    .unwrap();
    std::fs::copy(
        src.join("alexa/target.json"),
        dest.join("alexa/target.json"),
    )
    .unwrap();
    for entry in std::fs::read_dir(src.join("cameras/profiles")).unwrap() {
        let entry = entry.unwrap();
        let to = dest.join("cameras/profiles").join(entry.file_name());
        std::fs::create_dir_all(&to).unwrap();
        std::fs::copy(entry.path().join("profile.json"), to.join("profile.json")).unwrap();
    }
}

#[test]
fn loading_reports_what_is_wrong() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    copy_data(root);
    // The copy loads.
    ProfileStore::load(root).unwrap();

    // A missing data root and a missing profile file are I/O errors that name the file.
    let err = ProfileStore::load(root.join("nowhere")).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Io { path, .. } if path.ends_with("registry.json")),
        "{err}"
    );
    let victim = root.join("cameras/profiles/sony-ilce-9/profile.json");
    let original = std::fs::read_to_string(&victim).unwrap();
    std::fs::remove_file(&victim).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Io { path, .. } if path.ends_with("sony-ilce-9/profile.json")),
        "{err}"
    );

    // Truncated JSON is a JSON error with the path.
    std::fs::write(&victim, &original[..original.len() / 2]).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Json { path, .. } if path.ends_with("sony-ilce-9/profile.json")),
        "{err}"
    );

    // A profile that is not white preserving is refused by validation, naming the profile.
    let mut doc: Value = serde_json::from_str(&original).unwrap();
    doc["sensor_map"]["classes"]["led"][0]["m"][1][1] = json!(1.001);
    std::fs::write(&victim, serde_json::to_string(&doc).unwrap()).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Invalid { id, reason } if id == "sony-ilce-9" && reason.contains("white")),
        "{err}"
    );

    // A profile file holding another profile's id disagrees with the registry.
    let mut doc: Value = serde_json::from_str(&original).unwrap();
    doc["id"] = json!("sony-ilce-6400");
    std::fs::write(&victim, serde_json::to_string(&doc).unwrap()).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(
            &err,
            ProfileError::Registry(RegistryError::IdMismatch { .. })
        ),
        "{err}"
    );

    // Restoring the file restores the store.
    std::fs::write(&victim, &original).unwrap();
    ProfileStore::load(root).unwrap();

    // A registry that lists a body twice is refused.
    let reg_path = root.join("cameras/registry.json");
    let mut reg: Value = read_json(&reg_path);
    let first = reg["bodies"][0].clone();
    reg["bodies"].as_array_mut().unwrap().push(first);
    std::fs::write(&reg_path, serde_json::to_string(&reg).unwrap()).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(
            &err,
            ProfileError::Registry(RegistryError::DuplicateModelId(363))
        ),
        "{err}"
    );

    // A truncated registry is a JSON error that names the file; a missing target is an I/O error that
    // names its file too.
    let original_registry =
        std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap();
    std::fs::write(&reg_path, &original_registry[..original_registry.len() / 2]).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Json { path, .. } if path.ends_with("registry.json")),
        "{err}"
    );
    std::fs::write(&reg_path, &original_registry).unwrap();
    ProfileStore::load(root).unwrap();
    std::fs::remove_file(root.join("alexa/target.json")).unwrap();
    let err = ProfileStore::load(root).unwrap_err();
    assert!(
        matches!(&err, ProfileError::Io { path, .. } if path.ends_with("target.json")),
        "{err}"
    );
}

#[test]
fn the_target_parser_refuses_malformed_documents() {
    let text = std::fs::read_to_string(data_root().join("alexa/target.json")).unwrap();
    Target::parse(&text).unwrap();
    assert!(matches!(
        Target::parse("{}"),
        Err(ProfileError::Json { .. })
    ));
    type Mutation = Box<dyn Fn(&mut Value)>;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("unknown field", Box::new(|d| d["extra"] = json!(1))),
        (
            "kind outside the enum",
            Box::new(|d| d["kind"] = json!("guess")),
        ),
        (
            "version not semver",
            Box::new(|d| d["version"] = json!("1.0")),
        ),
        (
            "missing provenance",
            Box::new(|d| {
                d.as_object_mut().unwrap().remove("provenance");
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let mut doc: Value = serde_json::from_str(&text).unwrap();
        mutate(&mut doc);
        assert!(
            matches!(
                Target::parse(&doc.to_string()),
                Err(ProfileError::Json { .. })
            ),
            "{name}"
        );
    }
    assert!(matches!(
        Target::from_path("/nonexistent/target.json"),
        Err(ProfileError::Io { .. })
    ));
}

#[test]
fn profile_tables_work_through_the_profile_type() {
    use ym_color::spectral::{ClassProbabilities, LightClass};
    let p = store().default_profile();
    // The sensor map keeps white white.
    let probs = ClassProbabilities::new(0.2, 0.3, 0.1, 0.4).unwrap();
    let m = p.sensor_map(&probs, 1e6 / 4300.0);
    for row in m {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    // w_A of the scene and user whites have green exactly 1; daylight class at 6500 K equals the user white.
    let w = p.scene_white(6500.0, &ClassProbabilities::only(LightClass::Daylight));
    assert_eq!(w, p.user_white(6500.0));
    assert_eq!(w[1], 1.0);
    // CCT/Duv round trip through the profile.
    let rgb = p.cct_duv_to_rgb(5200.0, 0.002);
    let back = p.rgb_to_cct_duv(rgb).unwrap();
    // (The foot of the perpendicular moves a little along the curved locus: 2e-7 Duv at 0.002.)
    assert!(
        (back.cct_k - 5200.0).abs() < 1e-3 && (back.duv - 0.002).abs() < 1e-6,
        "{back:?}"
    );
    assert!(p.rgb_to_cct_duv([0.0, 1.0, 1.0]).is_err());
}

#[test]
fn profile_validation_rejects_rule_violations() {
    // Each mutation parses (the shape is right) but breaks a rule the schema cannot express.
    let original =
        std::fs::read_to_string(data_root().join("cameras/profiles/sony-ilce-6400/profile.json"))
            .unwrap();
    Profile::parse(&original).unwrap();
    let base: Value = serde_json::from_str(&original).unwrap();
    type Mutation = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, &str, Mutation)> = vec![
        (
            "rows do not sum to 1",
            "preserve white",
            Box::new(|d| d["sensor_map"]["classes"]["daylight"][3]["m"][2][0] = json!(0.5)),
        ),
        (
            "mired out of order",
            "mired",
            Box::new(|d| {
                let c = d["sensor_map"]["classes"]["fluorescent"]
                    .as_array_mut()
                    .unwrap();
                c.swap(0, 1);
            }),
        ),
        (
            "green not 1 in a fluorescent w_A",
            "green",
            Box::new(|d| d["alexa_white"]["fluorescent"][2]["w"][1] = json!(1.0001)),
        ),
        (
            "daylight w_A starts at the wrong temperature",
            "4000",
            Box::new(|d| d["alexa_white"]["daylight"]["start_k"] = json!(4010)),
        ),
        (
            "locus runs the wrong way",
            "locus.blackbody",
            Box::new(|d| {
                let v = d["locus"]["blackbody"]["values"].as_array_mut().unwrap();
                v.reverse();
            }),
        ),
        (
            "a reference light is renamed",
            "reference_lights",
            Box::new(|d| d["provenance"]["reference_lights"]["led"][0] = json!("LED-X")),
        ),
        (
            "a formula light's mired is off",
            "1e6/",
            Box::new(|d| d["sensor_map"]["classes"]["tungsten"][0]["mired"] = json!(251.0)),
        ),
        (
            "the fluorescent w_A list drifts from the maps",
            "alexa_white.fluorescent",
            Box::new(|d| d["alexa_white"]["fluorescent"][0]["mired"] = json!(153.0)),
        ),
        (
            "method changed",
            "method",
            Box::new(|d| d["sensor_map"]["method"] = json!("mlp")),
        ),
        (
            "quirks in V1",
            "quirks",
            Box::new(|d| d["decode"]["quirks"] = json!(["q"])),
        ),
    ];
    for (name, expect, mutate) in cases {
        let mut doc = base.clone();
        mutate(&mut doc);
        match Profile::parse(&serde_json::to_string(&doc).unwrap()) {
            Err(ProfileError::Invalid { id, reason }) => {
                assert_eq!(id, "sony-ilce-6400", "{name}");
                assert!(reason.contains(expect), "{name}: {reason}");
            }
            other => panic!(
                "{name}: expected a validation error, got {:?}",
                other.map(|p| p.id)
            ),
        }
    }
    // Shape errors are JSON errors, not validation errors: an unknown field, a bad id, a bad version.
    for mutate in [
        Box::new(|d: &mut Value| d["extra"] = json!(1)) as Mutation,
        Box::new(|d| d["id"] = json!("Not An Id")),
        Box::new(|d| d["version"] = json!("one")),
        Box::new(|d| d["locus"]["blackbody"]["values"] = json!([[0.0, 0.0]])),
    ] {
        let mut doc = base.clone();
        mutate(&mut doc);
        assert!(matches!(
            Profile::parse(&serde_json::to_string(&doc).unwrap()),
            Err(ProfileError::Json { .. })
        ));
    }
}

#[test]
fn shipped_camera_data_names_no_project_we_cannot_license_as_a_source() {
    // docs/14 rule 8 (2026-10-14): colour matrices, white-balance coefficients and model IDs may be taken as
    // numbers from any source, but nothing in the repository or the app refers to a project whose licence we
    // cannot accept as the source. The registry and the profiles ship in the app bundle, so their notes,
    // sources and provenance say "the published colour matrix" and "the values the cameras write", not the
    // tool that read them.
    const BANNED: [&str; 8] = [
        "libraw",
        "rawpy",
        "exiftool",
        "dcraw",
        "rawtherapee",
        "darktable",
        "rawler",
        "rawspeed",
    ];
    let cameras = data_root().join("cameras");
    let mut files = vec![cameras.join("registry.json")];
    for entry in std::fs::read_dir(cameras.join("profiles")).expect("profiles folder") {
        files.push(entry.expect("entry").path().join("profile.json"));
    }
    assert!(files.len() > 11, "the registry and the 11 profiles");
    for file in files {
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("{}: {e}", file.display()))
            .to_lowercase();
        for word in BANNED {
            assert!(
                !text.contains(word),
                "{} names {word}: docs/14 rule 8 (say \"the published colour matrix\" instead)",
                file.display()
            );
        }
    }
}
