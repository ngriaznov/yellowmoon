//! DevelopSettings: JSON shape (against `src/api/types.ts` and the JSON Schema), patches,
//! clamping and resolution against automatic estimates.

#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use common::{Layout, Scene, estimates_of_file, light};
use serde_json::{Value, json};
use ym_color::warp::WarpPin;
use ym_color::wb::LightSource;
use ym_core::auto::pick_white_sample;
use ym_core::camera::{CameraCard, CameraInfo};
use ym_core::session::{
    PhotoId, PhotoProgress, PhotoStage, PhotoStatus, PhotoSummary, SessionState, ThumbnailReady,
};
use ym_core::settings::{
    AutoOr, AutoSeeds, COPY_LEAVES_OUT, ColorBalance, ColorWarp, ColorWarpPin, Crop, CropAspect,
    Curves, DevelopSettings, NoiseReductionValue, NormRect, Orientation, ResolvedSettings, Rolloff,
    SettingsError, SettingsField, SettingsPatch, SharpenValue, StageGroup,
};
use ym_looks::library::LookId;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

/// Every number as f64, so `100` and `100.0` compare equal.
fn norm(v: Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Array(a) => Value::Array(a.into_iter().map(norm).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, norm(v))).collect()),
        other => other,
    }
}

fn rich() -> DevelopSettings {
    let mut s = DevelopSettings::with_look(LookId::parse("arri:3000:film-warm").unwrap());
    s.look_intensity = 62.5;
    s.exposure = AutoOr::Manual(ym_core::settings::ExposureValue { ev: -1.25 });
    s.contrast = ym_core::settings::PercentSetting::manual_value(12.0);
    s.brightness = -7.5;
    s.highlights = ym_core::settings::PercentSetting::manual_value(-30.0);
    s.shadows = ym_core::settings::PercentSetting::manual_value(25.0);
    s.white = 4.0;
    s.white_offset_ev = 0.4;
    s.black = ym_core::settings::PercentSetting::manual_value(-9.0);
    s.light_source = LightSource::Tungsten;
    s.white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 3250.0,
        tint: -4.5,
    });
    s.color_balance = ColorBalance {
        shadows: ym_core::settings::BalanceSetting::manual_point([0.2, -0.3]),
        midtones: [0.0, 0.0],
        highlights: ym_core::settings::BalanceSetting::manual_point([-0.5, 0.5]),
    };
    s.saturation = ym_core::settings::PercentSetting::manual_value(15.0);
    s.color_warp = ColorWarp {
        pins: vec![
            ColorWarpPin {
                source: [0.12, 0.03],
                target: [0.08, 0.1],
                locked: false,
                range: 0.06,
                shadows: 0.5,
                highlights: 1.0,
                pivot_ev: -1.5,
                exposure_ev: 0.25,
            },
            ColorWarpPin::from(WarpPin::locked_at([-0.05, -0.1])),
        ],
    };
    s.sharpen = AutoOr::Manual(SharpenValue {
        amount: 150.0,
        radius_px: 1.1,
    });
    s.output_sharpening = 140.0;
    s.noise_reduction = AutoOr::Manual(NoiseReductionValue {
        luminance: 35.0,
        color: 60.0,
        detail: 20.0,
    });
    s.curve = Curves {
        luma: vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.82], [1.0, 1.0]],
        red: vec![[0.0, 0.02], [1.0, 1.0]],
        green: vec![[0.0, 0.0], [1.0, 1.0]],
        blue: vec![[0.0, 0.0], [0.5, 0.45], [1.0, 0.98]],
    };
    s.decode.neural_demosaic = false;
    s.decode.chromatic_aberration = false;
    s.rolloff = Rolloff {
        highlights: 0.8,
        low_light: 0.1,
    };
    s.crop = Some(Crop {
        rect: NormRect {
            x: 0.1,
            y: 0.2,
            width: 0.7,
            height: 0.6,
        },
        angle_deg: -2.5,
        aspect: CropAspect::FourByFive,
    });
    s.orientation = Orientation::Deg90;
    s.white_pick = Some([0.25, 0.75]);
    s
}

// ---------------------------------------------------------------------------------------------
// JSON shape
// ---------------------------------------------------------------------------------------------

/// What `defaultSettings()` in `src/api.mock.ts` produces: the frontend's idea of the defaults (with
/// Auto contrast, highlights, shadows, black, saturation and the shadows and highlights wheels,
/// `docs/19` section 5).
fn typescript_defaults() -> Value {
    json!({
        "look": "builtin:neutral",
        "look_intensity": 100,
        "exposure": { "kind": "auto" },
        "contrast": { "kind": "auto" },
        "brightness": 0,
        "highlights": { "kind": "auto" },
        "shadows": { "kind": "auto" },
        "white": 0,
        "black": { "kind": "auto" },
        "white_offset_ev": 0,
        "light_source": "auto",
        "white_balance": { "kind": "auto" },
        "color_balance": {
            "shadows": { "kind": "auto" },
            "midtones": [0, 0],
            "highlights": { "kind": "auto" }
        },
        "saturation": { "kind": "auto" },
        "color_warp": { "pins": [] },
        "sharpen": { "kind": "auto" },
        "output_sharpening": 100,
        "noise_reduction": { "kind": "auto" },
        "curve": {
            "luma": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]],
            "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]]
        },
        "decode": {
            "neural_demosaic": true, "highlight_recovery": true,
            "lens_corrections": true, "chromatic_aberration": true
        },
        "rolloff": { "highlights": 1, "low_light": 0.5 },
        "crop": null,
        "orientation": 0,
        "white_pick": null
    })
}

#[test]
fn the_default_settings_are_the_frontends_defaults() {
    let ours = norm(serde_json::to_value(DevelopSettings::default()).unwrap());
    assert_eq!(ours, norm(typescript_defaults()));
    // And the frontend's JSON reads back as the Rust defaults.
    let back: DevelopSettings = serde_json::from_value(typescript_defaults()).unwrap();
    assert_eq!(back, DevelopSettings::default());
    // An empty document is the defaults too: every missing field takes its default.
    let empty: DevelopSettings = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, DevelopSettings::default());
}

/// The field names of `export interface <name> { ... }` in `src/api/types.ts`.
fn ts_interface_keys(source: &str, name: &str) -> BTreeSet<String> {
    let start = source
        .find(&format!("export interface {name} {{"))
        .unwrap_or_else(|| panic!("interface {name} in types.ts"));
    let body = &source[start..];
    let end = body.find("\n}").expect("end of the interface");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.strip_prefix("  ")?;
            if line.starts_with(' ') || line.starts_with('/') || line.starts_with('*') {
                return None;
            }
            let (key, _) = line.split_once(':')?;
            Some(key.trim_end_matches('?').to_owned())
        })
        .collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn the_rust_types_have_the_fields_of_the_typescript_interfaces() {
    // A drift guard: the serde field names of every wire type that crosses the bridge equal the
    // interface in `src/api/types.ts`, so generated bindings line up.
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let check = |name: &str, value: Value| {
        assert_eq!(
            ts_interface_keys(&source, name),
            keys(&value),
            "interface {name} against its Rust type"
        );
    };
    let settings = rich();
    check("DevelopSettings", serde_json::to_value(&settings).unwrap());
    check(
        "ColorBalance",
        serde_json::to_value(settings.color_balance).unwrap(),
    );
    check(
        "ColorWarp",
        serde_json::to_value(&settings.color_warp).unwrap(),
    );
    check(
        "ColorWarpPin",
        serde_json::to_value(settings.color_warp.pins[0]).unwrap(),
    );
    check("Curves", serde_json::to_value(&settings.curve).unwrap());
    check(
        "DecodeOptions",
        serde_json::to_value(settings.decode).unwrap(),
    );
    check("Rolloff", serde_json::to_value(settings.rolloff).unwrap());
    check(
        "Crop",
        serde_json::to_value(settings.crop.unwrap()).unwrap(),
    );
    check(
        "NormRect",
        serde_json::to_value(settings.crop.unwrap().rect).unwrap(),
    );
    let summary = PhotoSummary {
        id: PhotoId(1),
        file_name: "A.ARW".into(),
        camera: CameraInfo::unknown(),
        iso: 100,
        shutter_s: 0.01,
        aperture_f: 2.8,
        captured_at: String::new(),
        width_px: 1,
        height_px: 1,
        status: PhotoStatus::Waiting,
        picked: false,
        reduced: false,
    };
    check("PhotoSummary", serde_json::to_value(&summary).unwrap());
    check(
        "CameraInfo",
        serde_json::to_value(CameraInfo::unknown()).unwrap(),
    );
    // The interface also has `opened`, what an `open_paths` or `open_dialog` call did: the shell's answer adds it
    // (`src-tauri/src/wire.rs`, `OpenSummary`) and the core's `state()` never has it.
    let mut state = serde_json::to_value(SessionState {
        folder_name: None,
        photos: vec![],
        current_index: None,
        picked: vec![],
        default_export_folder: None,
    })
    .unwrap();
    state["opened"] = Value::Null;
    check("SessionState", state);
    check(
        "PhotoProgress",
        serde_json::to_value(PhotoProgress {
            id: PhotoId(1),
            stage: PhotoStage::Decode,
            done: true,
            error: None,
        })
        .unwrap(),
    );
    // `ThumbnailReady` is not here: the core's event still carries the PNG (`png_bytes`), and the
    // shell turns it into the `{ id, url }` of `types.ts` (the picture is served on the `ym-frame`
    // scheme, `docs/01` IPC); the shell's `bindings` tests compare that wire type with the interface.
    let _ = ThumbnailReady {
        id: PhotoId(1),
        png_bytes: vec![1],
    };
    // The status text's payload goes to the webview as it is, zoom included.
    check(
        "RenderStats",
        serde_json::to_value(ym_core::pipeline::RenderStats {
            ms: 1.0,
            path: ym_core::pipeline::RenderPath::Classical,
            space: ym_export::ColorSpace::DisplayP3,
            zoom_percent: Some(50.0),
            fit_percent: None,
        })
        .unwrap(),
    );
    // The colour warp's plane and Pick colour.
    check(
        "ChromaScatter",
        serde_json::to_value(ym_core::pipeline::ChromaScatter {
            points: vec![0.12, -0.03],
        })
        .unwrap(),
    );
    check(
        "ChromaSample",
        serde_json::to_value(ym_core::pipeline::ChromaSample { a: 0.12, b: -0.03 }).unwrap(),
    );
    // The camera card has no TypeScript twin yet; it must at least be plain data.
    let _ = serde_json::to_value(CameraCard {
        detected: String::new(),
        profile: String::new(),
        stand_in: false,
        note: None,
        reduced: false,
        reduced_note: None,
        raw_mode: String::new(),
        unsupported: None,
    })
    .unwrap();
}

#[test]
fn resolved_settings_have_the_shape_of_the_typescript_type() {
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(
        256,
        192,
        light("sony-ilce-7rm4", 5200.0),
        Layout::Patchwork,
        0.5,
    );
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", "ILCE-7RM4"));
    let resolved = ResolvedSettings::resolve(
        PhotoId(7),
        &DevelopSettings::default(),
        &est,
        &profile,
        None,
    )
    .unwrap();
    let v = serde_json::to_value(&resolved).unwrap();
    assert_eq!(ts_interface_keys(&source, "ResolvedSettings"), keys(&v));
    // Apple Auto (`docs/19` sections 10 and 12): the flags of contrast and saturation, and the Auto values,
    // the source and the photo's state with Apple Auto, in the core and in `types.ts` alike.
    assert_eq!(ts_interface_keys(&source, "AutoFlags"), keys(&v["auto"]));
    assert_eq!(
        ts_interface_keys(&source, "AutoEstimates"),
        keys(&v["estimates"])
    );
    for key in ["contrast", "saturation"] {
        assert!(
            ts_interface_keys(&source, "AutoFlags").contains(key),
            "{key}"
        );
    }
    for key in ["contrast", "saturation", "source", "apple"] {
        assert!(
            ts_interface_keys(&source, "AutoEstimates").contains(key),
            "{key}"
        );
    }
    assert_eq!(v["auto"]["contrast"], json!(true));
    assert_eq!(v["estimates"]["source"], json!("rule"));
    assert_eq!(v["estimates"]["apple"], json!({ "state": "off" }));
    assert_eq!(
        (
            v["estimates"]["contrast"].clone(),
            v["estimates"]["saturation"].clone()
        ),
        (json!(0.0), json!(0.0))
    );
    // ResolvedValues = DevelopSettings without the six automatic fields, plus their numbers.
    let mut want = ts_interface_keys(&source, "DevelopSettings");
    for gone in [
        "exposure",
        "contrast",
        "white_balance",
        "saturation",
        "sharpen",
        "noise_reduction",
    ] {
        want.remove(gone);
    }
    want.extend(
        [
            "exposure_ev",
            "contrast",
            "saturation",
            "white_ev",
            "kelvin",
            "tint",
            "sharpen",
            "noise_reduction",
        ]
        .into_iter()
        .map(String::from),
    );
    assert_eq!(want, keys(&v["values"]));
    assert_eq!(v["photo_id"], json!(7));
    let class = v["estimates"]["light_class"].as_str().unwrap();
    assert!(
        ["daylight", "tungsten", "fluorescent", "led"].contains(&class),
        "{class}"
    );
}

#[test]
fn settings_round_trip_through_json() {
    let s = rich();
    let text = serde_json::to_string(&s).unwrap();
    let back: DevelopSettings = serde_json::from_str(&text).unwrap();
    assert_eq!(back, s);
    let pretty = serde_json::to_string_pretty(&s).unwrap();
    assert_eq!(serde_json::from_str::<DevelopSettings>(&pretty).unwrap(), s);
    // Tagged unions carry their kind; the orientation is a number of degrees.
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["exposure"], json!({ "kind": "manual", "ev": -1.25 }));
    assert_eq!(
        v["white_balance"],
        json!({ "kind": "manual", "kelvin": 3250.0, "tint": -4.5 })
    );
    assert_eq!(v["orientation"], json!(90));
    assert_eq!(v["light_source"], json!("tungsten"));
    assert_eq!(v["crop"]["aspect"], json!("four_by_five"));
    assert_eq!(v["look"], json!("arri:3000:film-warm"));
}

#[test]
fn unknown_fields_and_malformed_values_are_refused() {
    for bad in [
        r#"{"exposur": {"kind": "auto"}}"#,
        r#"{"exposure": {"kind": "sometimes"}}"#,
        r#"{"exposure": {"kind": "manual"}}"#,
        r#"{"exposure": {"kind": "manual", "ev": 1, "extra": 2}}"#,
        r#"{"look": "Neutral"}"#,
        r#"{"light_source": "moonlight"}"#,
        r#"{"orientation": 45}"#,
        r#"{"crop": {"rect": {"x": 0, "y": 0, "width": 1}, "angle_deg": 0, "aspect": "free"}}"#,
        r#"{"decode": {"neural": true}}"#,
        r#"{"curve": {"luma": [[0, 0, 0], [1, 1]]}}"#,
    ] {
        assert!(
            serde_json::from_str::<DevelopSettings>(bad).is_err(),
            "{bad} should be refused"
        );
    }
    // A partial document is fine.
    let partial: DevelopSettings =
        serde_json::from_str(r#"{"contrast": 10, "decode": {"lens_corrections": false}}"#).unwrap();
    assert_eq!(
        partial.contrast,
        ym_core::settings::PercentSetting::manual_value(10.0)
    );
    assert!(!partial.decode.lens_corrections && partial.decode.neural_demosaic);
}

// ---------------------------------------------------------------------------------------------
// JSON Schema
// ---------------------------------------------------------------------------------------------

fn schema() -> (Value, jsonschema::Validator) {
    let path = repo_root().join("crates/ym-core/schema/settings.schema.json");
    let value: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&value).expect("the schema is a valid schema");
    (value, validator)
}

fn problems(v: &jsonschema::Validator, doc: &Value) -> Vec<String> {
    v.iter_errors(doc)
        .take(5)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect()
}

#[test]
fn what_the_core_writes_validates_against_the_schema() {
    let (_, v) = schema();
    for (label, s) in [("default", DevelopSettings::default()), ("rich", rich())] {
        let doc = serde_json::to_value(&s).unwrap();
        assert!(
            problems(&v, &doc).is_empty(),
            "{label}: {:?}",
            problems(&v, &doc)
        );
    }
    // Every light preset, aspect and rotation the core can write.
    for light_source in [
        LightSource::Auto,
        LightSource::Day,
        LightSource::Shade,
        LightSource::Tungsten,
        LightSource::Led,
        LightSource::Fluorescent,
    ] {
        let mut s = rich();
        s.light_source = light_source;
        let doc = serde_json::to_value(&s).unwrap();
        assert!(problems(&v, &doc).is_empty(), "{light_source:?}");
    }
    for aspect in [
        CropAspect::Free,
        CropAspect::Original,
        CropAspect::Square,
        CropAspect::FourByFive,
        CropAspect::SixteenByNine,
    ] {
        let mut s = rich();
        s.crop.as_mut().unwrap().aspect = aspect;
        assert!(
            problems(&v, &serde_json::to_value(&s).unwrap()).is_empty(),
            "{aspect:?}"
        );
    }
    for o in [
        Orientation::Deg0,
        Orientation::Deg90,
        Orientation::Deg180,
        Orientation::Deg270,
    ] {
        let mut s = rich();
        s.orientation = o;
        assert!(
            problems(&v, &serde_json::to_value(&s).unwrap()).is_empty(),
            "{o:?}"
        );
    }
    // Partial documents and the typescript defaults.
    assert!(problems(&v, &json!({})).is_empty());
    assert!(problems(&v, &typescript_defaults()).is_empty());
    assert!(
        problems(
            &v,
            &json!({ "contrast": 10, "decode": { "lens_corrections": false } })
        )
        .is_empty()
    );
}

#[test]
fn the_schema_refuses_what_the_core_refuses() {
    let (_, v) = schema();
    let mut cases: Vec<(&str, Value)> = vec![
        ("unknown field", json!({ "exposur": { "kind": "auto" } })),
        (
            "manual without ev",
            json!({ "exposure": { "kind": "manual" } }),
        ),
        (
            "exposure above range",
            json!({ "exposure": { "kind": "manual", "ev": 4.5 } }),
        ),
        (
            "kelvin below range",
            json!({ "white_balance": { "kind": "manual", "kelvin": 1500, "tint": 0 } }),
        ),
        (
            "tint above range",
            json!({ "white_balance": { "kind": "manual", "kelvin": 5000, "tint": 80 } }),
        ),
        (
            "sharpen radius below range",
            json!({ "sharpen": { "kind": "manual", "amount": 100, "radius_px": 0.1 } }),
        ),
        (
            "noise reduction above range",
            json!({ "noise_reduction": { "kind": "manual", "luminance": 101, "color": 0, "detail": 0 } }),
        ),
        ("contrast above range", json!({ "contrast": 101 })),
        (
            "manual contrast above range",
            json!({ "contrast": { "kind": "manual", "value": 101 } }),
        ),
        (
            "manual saturation without a value",
            json!({ "saturation": { "kind": "manual" } }),
        ),
        (
            "manual saturation with an unknown field",
            json!({ "saturation": { "kind": "manual", "value": 1, "ev": 2 } }),
        ),
        ("intensity above range", json!({ "look_intensity": 120 })),
        (
            "output sharpening above range",
            json!({ "output_sharpening": 201 }),
        ),
        (
            "output sharpening below range",
            json!({ "output_sharpening": -1 }),
        ),
        (
            "output sharpening is not Auto",
            json!({ "output_sharpening": { "kind": "auto" } }),
        ),
        (
            "rolloff above range",
            json!({ "rolloff": { "highlights": 1.5 } }),
        ),
        ("bad look id", json!({ "look": "Neutral" })),
        ("bad preset", json!({ "light_source": "moonlight" })),
        ("rotation of 45", json!({ "orientation": 45 })),
        (
            "pick outside the photo",
            json!({ "white_pick": [1.5, 0.5] }),
        ),
        (
            "pick with three numbers",
            json!({ "white_pick": [0.5, 0.5, 0.5] }),
        ),
        (
            "crop without aspect",
            json!({ "crop": { "rect": { "x": 0, "y": 0, "width": 1, "height": 1 }, "angle_deg": 0 } }),
        ),
        (
            "crop angle above range",
            json!({ "crop": { "rect": { "x": 0, "y": 0, "width": 1, "height": 1 }, "angle_deg": 60, "aspect": "free" } }),
        ),
        (
            "colour balance outside the square",
            json!({ "color_balance": { "shadows": [1.5, 0] } }),
        ),
        (
            "warp pin without a source",
            json!({ "color_warp": { "pins": [{ "target": [0.1, 0.0] }] } }),
        ),
        (
            "warp pin with an unknown field",
            json!({ "color_warp": { "pins": [{ "source": [0.1, 0.0], "strength": 1 }] } }),
        ),
        (
            "warp point outside the plane",
            json!({ "color_warp": { "pins": [{ "source": [0.7, 0.0] }] } }),
        ),
        (
            "warp target of null",
            json!({ "color_warp": { "pins": [{ "source": [0.1, 0.0], "target": null }] } }),
        ),
        (
            "warp range above range",
            json!({ "color_warp": { "pins": [{ "source": [0.1, 0.0], "range": 0.5 }] } }),
        ),
        (
            "warp exposure above range",
            json!({ "color_warp": { "pins": [{ "source": [0.1, 0.0], "exposure_ev": 3 }] } }),
        ),
        (
            "warp with an unknown field",
            json!({ "color_warp": { "pins": [], "strength": 1 } }),
        ),
        (
            "curve with one point",
            json!({ "curve": { "luma": [[0, 0]] } }),
        ),
        (
            "curve point outside the square",
            json!({ "curve": { "luma": [[0, 0], [1, 2]] } }),
        ),
    ];
    let too_many: Vec<Value> = (0..17)
        .map(|i| json!([f64::from(i) / 16.0, f64::from(i) / 16.0]))
        .collect();
    cases.push((
        "curve with 17 points",
        json!({ "curve": { "red": too_many } }),
    ));
    let seventeen_pins: Vec<Value> = (0..17)
        .map(|i| json!({ "source": [0.01 * f64::from(i), 0.1] }))
        .collect();
    cases.push((
        "warp with 17 pins",
        json!({ "color_warp": { "pins": seventeen_pins } }),
    ));
    for (label, doc) in cases {
        assert!(
            !problems(&v, &doc).is_empty(),
            "{label} should be refused: {doc}"
        );
    }
}

#[test]
fn the_schema_ranges_are_the_ranges_the_core_clamps_to() {
    use ym_core::settings::*;
    let (s, _) = schema();
    let range = |p: &Value| {
        (
            p["minimum"].as_f64().unwrap(),
            p["maximum"].as_f64().unwrap(),
        )
    };
    let props = &s["properties"];
    let defs = &s["$defs"];
    assert_eq!(range(&props["look_intensity"]), LOOK_INTENSITY_RANGE);
    assert_eq!(range(&props["output_sharpening"]), OUTPUT_SHARPENING_RANGE);
    assert_eq!(
        props["output_sharpening"]["default"].as_f64(),
        Some(OUTPUT_SHARPENING_DEFAULT)
    );
    assert_eq!(range(&defs["percent"]), PERCENT_RANGE);
    let manual = |def: &str, field: &str| range(&defs[def]["oneOf"][1]["properties"][field]);
    assert_eq!(manual("auto_percent", "value"), PERCENT_RANGE);
    assert_eq!(range(&defs["auto_percent"]["oneOf"][2]), PERCENT_RANGE);
    assert_eq!(manual("exposure", "ev"), EXPOSURE_RANGE_EV);
    assert_eq!(manual("white_balance", "kelvin"), KELVIN_RANGE_K);
    assert_eq!(manual("white_balance", "tint"), TINT_RANGE);
    assert_eq!(manual("sharpen", "amount"), SHARPEN_AMOUNT_RANGE);
    assert_eq!(manual("sharpen", "radius_px"), SHARPEN_RADIUS_RANGE_PX);
    for f in ["luminance", "color", "detail"] {
        assert_eq!(manual("noise_reduction", f), NOISE_REDUCTION_RANGE);
    }
    assert_eq!(
        range(&props["rolloff"]["properties"]["highlights"]),
        ROLLOFF_RANGE
    );
    assert_eq!(
        range(&props["crop"]["oneOf"][1]["properties"]["angle_deg"]),
        CROP_ANGLE_RANGE_DEG
    );
    assert_eq!(defs["curve"]["maxItems"], json!(MAX_CURVE_POINTS));
    assert_eq!(defs["curve"]["minItems"], json!(MIN_CURVE_POINTS));
    {
        use ym_color::warp::*;
        let pin = &defs["warp_pin"]["properties"];
        assert_eq!(
            props["color_warp"]["properties"]["pins"]["maxItems"],
            json!(WARP_MAX_PINS)
        );
        assert_eq!(range(&pin["range"]), (WARP_RANGE_MIN, WARP_RANGE_MAX));
        assert_eq!(pin["range"]["default"].as_f64(), Some(WARP_RANGE_DEFAULT));
        assert_eq!(range(&pin["shadows"]), (0.0, 1.0));
        assert_eq!(range(&pin["highlights"]), (0.0, 1.0));
        assert_eq!(
            range(&pin["pivot_ev"]),
            (-WARP_PIVOT_LIMIT_EV, WARP_PIVOT_LIMIT_EV)
        );
        assert_eq!(
            range(&pin["exposure_ev"]),
            (-WARP_EXPOSURE_LIMIT_EV, WARP_EXPOSURE_LIMIT_EV)
        );
        for item in defs["chroma_point"]["prefixItems"].as_array().unwrap() {
            assert_eq!(range(item), (-WARP_COORD_LIMIT, WARP_COORD_LIMIT));
        }
        // The pin's defaults are what a pin that names only its source reads as.
        let read: ColorWarpPin = serde_json::from_value(json!({ "source": [0.1, 0.2] })).unwrap();
        let written = serde_json::to_value(read).unwrap();
        for key in [
            "locked",
            "range",
            "shadows",
            "highlights",
            "pivot_ev",
            "exposure_ev",
        ] {
            assert_eq!(
                norm(pin[key]["default"].clone()),
                norm(written[key].clone()),
                "{key}"
            );
        }
        assert_eq!(
            props["color_warp"]["default"],
            serde_json::to_value(ColorWarp::default()).unwrap()
        );
    }
    // The schema's defaults are the core's defaults.
    let d = serde_json::to_value(DevelopSettings::default()).unwrap();
    assert_eq!(
        norm(props["look"]["default"].clone()),
        norm(d["look"].clone())
    );
    assert_eq!(props["look_intensity"]["default"], json!(100));
    assert_eq!(
        props["rolloff"]["properties"]["highlights"]["default"],
        json!(1)
    );
}

// ---------------------------------------------------------------------------------------------
// Patches
// ---------------------------------------------------------------------------------------------

fn apply(s: &mut DevelopSettings, patch: SettingsPatch) -> ym_core::settings::PatchReport {
    s.apply_patch(&patch, None).expect("patch applies")
}

fn parse_patch(text: &str) -> SettingsPatch {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

#[test]
fn an_empty_patch_changes_nothing() {
    let mut s = rich();
    let before = s.clone();
    let report = apply(&mut s, SettingsPatch::default());
    assert_eq!(s, before);
    assert!(report.changed.is_empty() && report.clamped.is_empty());
    assert_eq!(report.first_group(), None);
    assert!(SettingsPatch::default().is_empty());
    assert!(parse_patch("{}").is_empty());
}

#[test]
fn moving_a_slider_sends_a_manual_value_and_reset_sends_auto() {
    let mut s = DevelopSettings::default();
    let report = apply(&mut s, SettingsPatch::exposure_ev(0.3));
    assert_eq!(
        s.exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.3)
    );
    assert_eq!(report.changed, vec![SettingsField::Exposure]);
    assert_eq!(report.first_group(), Some(StageGroup::Color));
    // The same patch again changes nothing.
    assert!(
        apply(&mut s, SettingsPatch::exposure_ev(0.3))
            .changed
            .is_empty()
    );
    // Reset.
    apply(
        &mut s,
        SettingsPatch::reset(SettingsField::Exposure, &LookId::builtin("neutral")),
    );
    assert!(s.exposure.is_auto());
}

#[test]
fn values_out_of_range_are_clamped_and_reported() {
    let mut s = DevelopSettings::default();
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"exposure": {"kind":"manual","ev": 9}, "contrast": -250, "look_intensity": 140,
                "rolloff": {"highlights": 1.5, "low_light": -1},
                "white_balance": {"kind":"manual","kelvin": 20000,"tint": -90},
                "sharpen": {"kind":"manual","amount": 900,"radius_px": 0.01},
                "noise_reduction": {"kind":"manual","luminance": 120,"color": -5,"detail": 50}}"#,
        ),
    );
    assert_eq!(
        s.exposure,
        ym_core::settings::ExposureSetting::manual_ev(4.0)
    );
    assert_eq!(
        s.contrast,
        ym_core::settings::PercentSetting::manual_value(-100.0)
    );
    assert_eq!(s.look_intensity, 100.0);
    assert_eq!(
        s.rolloff,
        Rolloff {
            highlights: 1.0,
            low_light: 0.0
        }
    );
    assert_eq!(
        s.white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 11000.0,
            tint: -50.0
        })
    );
    assert_eq!(
        s.sharpen,
        AutoOr::Manual(SharpenValue {
            amount: 300.0,
            radius_px: 0.3
        })
    );
    assert_eq!(
        s.noise_reduction,
        AutoOr::Manual(NoiseReductionValue {
            luminance: 100.0,
            color: 0.0,
            detail: 50.0
        })
    );
    for f in [
        SettingsField::Exposure,
        SettingsField::Contrast,
        SettingsField::Rolloff,
        SettingsField::WhiteBalance,
        SettingsField::Sharpen,
        SettingsField::NoiseReduction,
    ] {
        assert!(report.clamped.contains(&f), "{f:?} was clamped");
    }
    // look_intensity 140 -> 100 equals the default 100, so it changed nothing and is not a change
    // (but it was clamped).
    assert!(report.clamped.contains(&SettingsField::LookIntensity));
    assert!(!report.changed.contains(&SettingsField::LookIntensity));
}

#[test]
fn a_patch_that_cannot_be_applied_changes_nothing() {
    let mut s = rich();
    let before = s.clone();
    // A valid change next to a value that is not a number.
    let patch = SettingsPatch {
        contrast: Some(ym_core::settings::PercentSetting::manual_value(10.0)),
        saturation: Some(ym_core::settings::PercentSetting::manual_value(f64::NAN)),
        ..SettingsPatch::default()
    };
    let err = s.apply_patch(&patch, None).unwrap_err();
    assert_eq!(err, SettingsError::NotFinite("saturation"));
    assert_eq!(s, before, "atomic: contrast must not have changed");
    for (label, patch) in [
        (
            "infinite kelvin",
            SettingsPatch::white_balance(f64::INFINITY, 0.0),
        ),
        ("NaN exposure", SettingsPatch::exposure_ev(f64::NAN)),
        (
            "NaN colour balance",
            SettingsPatch {
                color_balance: Some(ym_core::settings::ColorBalancePatch {
                    shadows: Some(ym_core::settings::BalanceSetting::manual_point([
                        f64::NAN,
                        0.0,
                    ])),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ),
    ] {
        assert!(s.apply_patch(&patch, None).is_err(), "{label}");
        assert_eq!(s, before, "{label}");
    }
}

#[test]
fn color_balance_points_are_scaled_back_into_the_unit_disc() {
    let mut s = DevelopSettings::default();
    let report = apply(
        &mut s,
        parse_patch(r#"{"color_balance": {"shadows": [3, 4], "highlights": [0.6, 0.8]}}"#),
    );
    let point = |w: &ym_core::settings::BalanceSetting| w.manual().expect("manual").point;
    assert!((point(&s.color_balance.shadows)[0] - 0.6).abs() < 1e-12);
    assert!((point(&s.color_balance.shadows)[1] - 0.8).abs() < 1e-12);
    // Exactly on the circle stays as it is; the untouched wheel stays zero.
    assert_eq!(point(&s.color_balance.highlights), [0.6, 0.8]);
    assert_eq!(s.color_balance.midtones, [0.0, 0.0]);
    assert!(report.clamped.contains(&SettingsField::ColorBalance));
}

#[test]
fn curves_are_checked_and_sorted() {
    let mut s = DevelopSettings::default();
    // Sorted by x, clamped into the unit square.
    let report = apply(
        &mut s,
        parse_patch(r#"{"curve": {"luma": [[1, 1], [0.5, 0.4], [0, 0], [0.7, 1.4]]}}"#),
    );
    assert_eq!(
        s.curve.luma,
        vec![[0.0, 0.0], [0.5, 0.4], [0.7, 1.0], [1.0, 1.0]]
    );
    assert!(report.changed.contains(&SettingsField::Curve));
    assert!(report.clamped.contains(&SettingsField::Curve));
    // The other channels were not named and are untouched.
    assert_eq!(s.curve.red, vec![[0.0, 0.0], [1.0, 1.0]]);
    // Up to 16 points; 17 is refused; fewer than 2 and duplicates are refused.
    let points = |n: usize| -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| [i as f64 / (n - 1) as f64, i as f64 / (n - 1) as f64])
            .collect()
    };
    let with = |p: Vec<[f64; 2]>| SettingsPatch {
        curve: Some(ym_core::settings::CurvesPatch {
            green: Some(p),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(s.apply_patch(&with(points(16)), None).is_ok());
    let before = s.clone();
    let err = s.apply_patch(&with(points(17)), None).unwrap_err();
    assert!(err.to_string().contains("17 points"), "{err}");
    assert!(err.to_string().starts_with("The green curve"), "{err}");
    assert!(s.apply_patch(&with(vec![[0.0, 0.0]]), None).is_err());
    assert!(
        s.apply_patch(
            &with(vec![[0.0, 0.0], [0.5, 0.2], [0.5, 0.3], [1.0, 1.0]]),
            None
        )
        .is_err()
    );
    assert_eq!(s, before);
}

#[test]
fn crop_and_white_pick_distinguish_absent_from_null() {
    let mut s = rich();
    // Absent: untouched.
    apply(&mut s, parse_patch(r#"{"contrast": 1}"#));
    assert!(s.crop.is_some() && s.white_pick.is_some());
    // null clears.
    let report = apply(&mut s, parse_patch(r#"{"crop": null, "white_pick": null}"#));
    assert!(s.crop.is_none() && s.white_pick.is_none());
    assert!(report.changed.contains(&SettingsField::Crop));
    assert!(report.changed.contains(&SettingsField::WhitePick));
    // A value sets.
    apply(
        &mut s,
        parse_patch(
            r#"{"crop": {"rect": {"x": 0.9, "y": 0, "width": 0.5, "height": 2},
                          "angle_deg": 80, "aspect": "free"}}"#,
        ),
    );
    let crop = s.crop.unwrap();
    // The rectangle is moved and shrunk to fit the photo, the angle clamped.
    assert!((crop.rect.x - 0.5).abs() < 1e-12 && (crop.rect.width - 0.5).abs() < 1e-12);
    assert_eq!((crop.rect.y, crop.rect.height), (0.0, 1.0));
    assert_eq!(crop.angle_deg, 45.0);
    // A crop with no area is refused.
    let empty = parse_patch(
        r#"{"crop": {"rect": {"x": 0.1, "y": 0.1, "width": 0, "height": 0.5},
                      "angle_deg": 0, "aspect": "free"}}"#,
    );
    assert_eq!(
        s.apply_patch(&empty, None).unwrap_err(),
        SettingsError::EmptyCrop
    );
    // The patch serialises with the same distinction.
    let cleared = serde_json::to_value(parse_patch(r#"{"crop": null}"#)).unwrap();
    assert_eq!(cleared, json!({ "crop": null }));
    assert_eq!(serde_json::to_value(parse_patch("{}")).unwrap(), json!({}));
}

#[test]
fn nested_objects_may_carry_only_the_parts_to_change() {
    let mut s = DevelopSettings::default();
    apply(
        &mut s,
        parse_patch(r#"{"decode": {"neural_demosaic": false}}"#),
    );
    assert!(!s.decode.neural_demosaic && s.decode.highlight_recovery);
    apply(&mut s, parse_patch(r#"{"rolloff": {"low_light": 0.9}}"#));
    assert_eq!(
        s.rolloff,
        Rolloff {
            highlights: 1.0,
            low_light: 0.9
        }
    );
    // A complete object, as the frontend sends it, works the same.
    apply(
        &mut s,
        parse_patch(
            r#"{"decode": {"neural_demosaic": true, "highlight_recovery": false,
                           "lens_corrections": true, "chromatic_aberration": true}}"#,
        ),
    );
    assert!(s.decode.neural_demosaic && !s.decode.highlight_recovery);
    let report = apply(
        &mut s,
        parse_patch(r#"{"decode": {"lens_corrections": false}}"#),
    );
    assert_eq!(report.first_group(), Some(StageGroup::Raw));
}

#[test]
fn a_manual_value_with_parts_left_out_keeps_the_rest_of_what_the_panel_shows() {
    let seeds = AutoSeeds {
        exposure_ev: 0.3,
        kelvin: 5450.0,
        tint: 3.0,
        sharpen: SharpenValue {
            amount: 120.0,
            radius_px: 0.8,
        },
        noise_reduction: NoiseReductionValue {
            luminance: 30.0,
            color: 50.0,
            detail: 50.0,
        },
    };
    // Auto white balance, Kelvin moved: manual at that Kelvin and the automatic tint.
    let mut s = DevelopSettings::default();
    s.apply_patch(&SettingsPatch::kelvin(4200.0), Some(&seeds))
        .unwrap();
    assert_eq!(
        s.white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 4200.0,
            tint: 3.0
        })
    );
    // Now manual: only the tint moves.
    s.apply_patch(
        &parse_patch(r#"{"white_balance": {"kind":"manual","tint": -8}}"#),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 4200.0,
            tint: -8.0
        })
    );
    // Sharpening: the amount moves, the radius is the automatic one.
    s.apply_patch(
        &parse_patch(r#"{"sharpen": {"kind":"manual","amount": 200}}"#),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.sharpen,
        AutoOr::Manual(SharpenValue {
            amount: 200.0,
            radius_px: 0.8
        })
    );
    s.apply_patch(
        &parse_patch(r#"{"noise_reduction": {"kind":"manual","color": 10}}"#),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.noise_reduction,
        AutoOr::Manual(NoiseReductionValue {
            luminance: 30.0,
            color: 10.0,
            detail: 50.0
        })
    );
    // Without seeds, a partial manual value on an Auto field cannot be filled in.
    let mut fresh = DevelopSettings::default();
    assert_eq!(
        fresh
            .apply_patch(&SettingsPatch::kelvin(4200.0), None)
            .unwrap_err(),
        SettingsError::NeedsEstimates("white_balance")
    );
    // A complete value never needs seeds.
    fresh
        .apply_patch(&SettingsPatch::white_balance(4200.0, 1.0), None)
        .unwrap();
}

#[test]
fn a_white_pick_resets_the_white_balance_unless_the_patch_says_otherwise() {
    let mut s = DevelopSettings::default();
    s.light_source = LightSource::Shade;
    s.white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 4000.0,
        tint: 2.0,
    });
    let report = apply(&mut s, parse_patch(r#"{"white_pick": [2, -1]}"#));
    assert_eq!(s.white_pick, Some([1.0, 0.0]), "clamped into the photo");
    assert_eq!(s.light_source, LightSource::Auto);
    assert!(s.white_balance.is_auto());
    for f in [
        SettingsField::WhitePick,
        SettingsField::LightSource,
        SettingsField::WhiteBalance,
    ] {
        assert!(report.changed.contains(&f), "{f:?}");
    }
    // The patch may choose a preset together with the pick.
    let mut s = DevelopSettings::default();
    apply(
        &mut s,
        parse_patch(r#"{"white_pick": [0.5, 0.5], "light_source": "led"}"#),
    );
    assert_eq!(s.light_source, LightSource::Led);
    // Clearing the pick leaves the white balance alone.
    s.white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 4000.0,
        tint: 2.0,
    });
    apply(&mut s, parse_patch(r#"{"white_pick": null}"#));
    assert!(!s.white_balance.is_auto());
}

#[test]
fn orientation_patches_normalise_quarter_turns() {
    let mut s = DevelopSettings::default();
    apply(&mut s, parse_patch(r#"{"orientation": -90}"#));
    assert_eq!(s.orientation, Orientation::Deg270);
    apply(&mut s, parse_patch(r#"{"orientation": 450}"#));
    assert_eq!(s.orientation, Orientation::Deg90);
    assert!(serde_json::from_str::<SettingsPatch>(r#"{"orientation": 45}"#).is_err());
    let report = apply(&mut s, parse_patch(r#"{"orientation": 180}"#));
    assert_eq!(report.first_group(), Some(StageGroup::Geometry));
}

#[test]
fn a_patch_with_an_unknown_field_is_refused() {
    assert!(serde_json::from_str::<SettingsPatch>(r#"{"exposur": {"kind":"auto"}}"#).is_err());
    assert!(serde_json::from_str::<SettingsPatch>(r#"{"decode": {"neural": false}}"#).is_err());
}

#[test]
fn the_frontends_patches_deserialise_and_apply() {
    // Shapes the screens send (src/screens/tools/*.tsx).
    let mut s = DevelopSettings::default();
    for text in [
        r#"{"exposure": {"kind": "manual", "ev": 0.45}}"#,
        r#"{"white_balance": {"kind": "manual", "kelvin": 4800, "tint": 6}}"#,
        r#"{"white_balance": {"kind": "auto"}, "light_source": "auto", "white_pick": null}"#,
        r#"{"color_balance": {"shadows": [0.1, 0.2], "midtones": [0, 0], "highlights": [0, 0]}}"#,
        r#"{"curve": {"luma": [[0, 0], [0.5, 0.6], [1, 1]], "red": [[0, 0], [1, 1]], "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]]}}"#,
        r#"{"crop": {"rect": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}, "angle_deg": 1.5, "aspect": "sixteen_by_nine"}, "orientation": 90}"#,
        r#"{"look": "builtin:signature", "look_intensity": 80}"#,
        r#"{"sharpen": {"kind": "manual", "amount": 90, "radius_px": 1}}"#,
    ] {
        let patch = parse_patch(text);
        s.apply_patch(&patch, None)
            .unwrap_or_else(|e| panic!("{text}: {e}"));
    }
    assert_eq!(s.look.as_str(), "builtin:signature");
    assert_eq!(s.crop.unwrap().aspect, CropAspect::SixteenByNine);
}

#[test]
fn reset_restores_every_field_to_its_default() {
    let look = LookId::builtin("signature");
    let d = DevelopSettings::with_look(look.clone());
    for field in SettingsField::ALL {
        let mut s = rich();
        // Reset through the patch, as the UI's Reset does.
        s.apply_patch(&SettingsPatch::reset(field, &look), None)
            .unwrap();
        let mut want = rich();
        want.reset_field(field, &look);
        assert_eq!(s, want, "{field:?}");
        // Only that field is back at its default.
        let diff = s.diff(&rich());
        assert!(
            diff.contains(&field),
            "{field:?} must differ from the edited value: {diff:?}"
        );
        let rest = s.diff(&d);
        assert!(
            !rest.contains(&field),
            "{field:?} must now equal its default: {rest:?}"
        );
    }
    // Re-run resets exposure, white balance (with preset and pick) and noise reduction only.
    let mut s = rich();
    s.reset_automatic(false);
    assert!(s.exposure.is_auto() && s.white_balance.is_auto() && s.noise_reduction.is_auto());
    assert_eq!(s.light_source, LightSource::Auto);
    assert!(s.white_pick.is_none());
    assert_eq!(
        s.white_offset_ev, 0.0,
        "the White row of the Automatic card is Auto again"
    );
    assert_eq!(
        s.rolloff,
        rich().rolloff,
        "the highlight range is not part of Re-run"
    );
    assert!(!s.sharpen.is_auto(), "sharpening is not part of Re-run");
    // Without Apple Auto acting on the photo, contrast and saturation are not part of Re-run.
    assert_eq!(s.contrast, rich().contrast);
    assert_eq!(s.saturation, rich().saturation);
    assert!(!rich().contrast.is_auto(), "the test edits contrast");
    // While Apple Auto acts on it they are on the Automatic card, and Re-run returns them to Auto
    // too (`docs/19` section 5).
    let mut s = rich();
    s.reset_automatic(true);
    assert_eq!(s.contrast, AutoOr::Auto);
    assert_eq!(s.saturation, AutoOr::Auto);
    // So do the other values Apple Auto sets (`docs/19` prompt version 2).
    assert_eq!(
        (s.highlights, s.shadows, s.black),
        (AutoOr::Auto, AutoOr::Auto, AutoOr::Auto)
    );
    assert!(s.color_balance.shadows.is_auto() && s.color_balance.highlights.is_auto());
    assert!(s.exposure.is_auto() && s.white_balance.is_auto() && s.noise_reduction.is_auto());
    assert!(!s.sharpen.is_auto(), "sharpening is not part of Re-run");
    // Without Apple Auto acting, they stay as they were.
    let mut s = rich();
    s.reset_automatic(false);
    assert_eq!(
        (s.highlights, s.shadows, s.black, s.color_balance),
        (
            rich().highlights,
            rich().shadows,
            rich().black,
            rich().color_balance
        )
    );
}

#[test]
fn highlights_shadows_black_and_the_wheels_are_auto_or_manual_and_read_old_values() {
    use ym_core::settings::{BalanceSetting, PercentSetting};
    let neutral = LookId::builtin("neutral");
    // The defaults are Auto, written in the tagged form.
    let d = DevelopSettings::default();
    let v = serde_json::to_value(&d).unwrap();
    for key in ["highlights", "shadows", "black"] {
        assert_eq!(v[key], json!({ "kind": "auto" }), "{key}");
    }
    assert_eq!(v["color_balance"]["shadows"], json!({ "kind": "auto" }));
    assert_eq!(v["color_balance"]["midtones"], json!([0.0, 0.0]));
    assert_eq!(v["color_balance"]["highlights"], json!({ "kind": "auto" }));
    // Old documents: a number is Manual, 0 is Auto; a point is Manual, [0, 0] is Auto; clamped.
    let (s, clamped) = DevelopSettings::from_json(
        r#"{ "highlights": -30, "shadows": 0, "black": 250,
             "color_balance": { "shadows": [0.2, -0.1], "highlights": [0, 0], "midtones": [0.1, 0] } }"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(s.highlights, PercentSetting::manual_value(-30.0));
    assert_eq!(s.shadows, AutoOr::Auto);
    assert_eq!(s.black, PercentSetting::manual_value(100.0));
    assert!(clamped.contains(&SettingsField::Black));
    assert_eq!(
        s.color_balance.shadows,
        BalanceSetting::manual_point([0.2, -0.1])
    );
    assert_eq!(s.color_balance.highlights, AutoOr::Auto);
    assert_eq!(s.color_balance.midtones, [0.1, 0.0]);
    // The new forms.
    let (s, _) = DevelopSettings::from_json(
        r#"{ "highlights": { "kind": "manual", "value": 12 },
             "color_balance": { "highlights": { "kind": "manual", "point": [3, 4] } } }"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(s.highlights, PercentSetting::manual_value(12.0));
    let p = s.color_balance.highlights.manual().unwrap().point;
    assert!((p[0] - 0.6).abs() < 1e-12 && (p[1] - 0.8).abs() < 1e-12);
    assert_eq!(
        s.color_balance.shadows,
        AutoOr::Auto,
        "a wheel left out is Auto"
    );
    assert_eq!(
        serde_json::to_value(BalanceSetting::manual_point([0.1, 0.2])).unwrap(),
        json!({ "kind": "manual", "point": [0.1, 0.2] })
    );
    // Wrong shapes are refused.
    for bad in [
        r#"{ "black": "deep" }"#,
        r#"{ "color_balance": { "shadows": { "kind": "manual" } } }"#,
        r#"{ "color_balance": { "shadows": { "kind": "manual", "value": 1 } } }"#,
        r#"{ "color_balance": { "shadows": [1, 2, 3] } }"#,
    ] {
        assert!(DevelopSettings::from_json(bad, &neutral).is_err(), "{bad}");
    }
    // Patches take either form; Auto is Reset.
    let mut s = DevelopSettings::default();
    for (text, highlights, wheel) in [
        (
            r#"{ "highlights": 20, "color_balance": { "shadows": [0.3, 0.1] } }"#,
            PercentSetting::manual_value(20.0),
            BalanceSetting::manual_point([0.3, 0.1]),
        ),
        (
            r#"{ "highlights": { "kind": "auto" } }"#,
            AutoOr::Auto,
            BalanceSetting::manual_point([0.3, 0.1]),
        ),
        (
            r#"{ "color_balance": { "shadows": { "kind": "auto" } } }"#,
            AutoOr::Auto,
            AutoOr::Auto,
        ),
        (
            r#"{ "color_balance": { "shadows": { "kind": "manual", "point": [0, -0.5] } } }"#,
            AutoOr::Auto,
            BalanceSetting::manual_point([0.0, -0.5]),
        ),
        (
            r#"{ "color_balance": { "shadows": [0, 0] } }"#,
            AutoOr::Auto,
            AutoOr::Auto,
        ),
    ] {
        s.apply_patch(&parse_patch(text), None).unwrap();
        assert_eq!(
            (s.highlights, s.color_balance.shadows),
            (highlights, wheel),
            "{text}"
        );
    }
    // Copy and paste carries Auto as Auto, and a manual value as it is.
    let mut from = DevelopSettings::default();
    from.shadows = PercentSetting::manual_value(-15.0);
    let copied = SettingsPatch::of_fields(
        &from,
        &[
            SettingsField::Highlights,
            SettingsField::Shadows,
            SettingsField::ColorBalance,
        ],
    );
    let v = serde_json::to_value(&copied).unwrap();
    assert_eq!(v["highlights"], json!({ "kind": "auto" }));
    assert_eq!(v["shadows"], json!({ "kind": "manual", "value": -15.0 }));
    assert_eq!(v["color_balance"]["shadows"], json!({ "kind": "auto" }));
    let mut onto = DevelopSettings::default();
    onto.highlights = PercentSetting::manual_value(30.0);
    onto.apply_patch(&copied, None).unwrap();
    assert_eq!(
        (onto.highlights, onto.shadows),
        (AutoOr::Auto, from.shadows)
    );
}

#[test]
fn output_sharpening_is_a_plain_number_in_a_patch_with_its_own_stage_group() {
    let look = LookId::builtin("neutral");
    let mut s = DevelopSettings::default();
    assert_eq!(s.output_sharpening, 100.0);
    // Moving the slider sends a number; only that field changes, and only the last stage group
    // is affected, so nothing before it re-runs.
    let report = apply(&mut s, parse_patch(r#"{"output_sharpening": 35}"#));
    assert_eq!(s.output_sharpening, 35.0);
    assert_eq!(report.changed, vec![SettingsField::OutputSharpening]);
    assert_eq!(report.first_group(), Some(StageGroup::Output));
    // The same patch again changes nothing.
    assert!(
        apply(&mut s, parse_patch(r#"{"output_sharpening": 35}"#))
            .changed
            .is_empty()
    );
    // 0 is a value (no stage), not a reset.
    apply(&mut s, parse_patch(r#"{"output_sharpening": 0}"#));
    assert_eq!(s.output_sharpening, 0.0);
    // Out of range is clamped and reported, in both directions.
    let report = apply(&mut s, parse_patch(r#"{"output_sharpening": 900}"#));
    assert_eq!(s.output_sharpening, 200.0);
    assert!(report.clamped.contains(&SettingsField::OutputSharpening));
    apply(&mut s, parse_patch(r#"{"output_sharpening": -20}"#));
    assert_eq!(s.output_sharpening, 0.0);
    // NaN is refused and changes nothing.
    let before = s.clone();
    let nan = SettingsPatch {
        output_sharpening: Some(f64::NAN),
        ..SettingsPatch::default()
    };
    assert_eq!(
        s.apply_patch(&nan, None).unwrap_err(),
        SettingsError::NotFinite("output_sharpening")
    );
    assert_eq!(s, before);
    // Reset returns to 100, not to Auto.
    let report = apply(
        &mut s,
        SettingsPatch::reset(SettingsField::OutputSharpening, &look),
    );
    assert_eq!(s.output_sharpening, 100.0);
    assert_eq!(report.changed, vec![SettingsField::OutputSharpening]);
    // It changes with a Grade field: the first group is the earliest of them.
    let report = apply(
        &mut s,
        parse_patch(r#"{"output_sharpening": 50, "contrast": 10}"#),
    );
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    // The patch that takes the field of another photo carries it (copy and paste of one field).
    let mut other = DevelopSettings::default();
    apply(
        &mut other,
        SettingsPatch::of_field(&s, SettingsField::OutputSharpening),
    );
    assert_eq!(other.output_sharpening, 50.0);
    assert_eq!(other.contrast, AutoOr::Auto);
}

#[test]
fn an_older_document_without_output_sharpening_reads_as_100() {
    // A settings file, a golden manifest or the ym:Settings of an export made before the field.
    let mut doc = serde_json::to_value(rich()).unwrap();
    doc.as_object_mut().unwrap().remove("output_sharpening");
    let (s, clamped) =
        DevelopSettings::from_value(doc, &LookId::builtin("neutral")).expect("reads");
    assert_eq!(s.output_sharpening, 100.0);
    assert!(!clamped.contains(&SettingsField::OutputSharpening));
    // And the full document writes it back.
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"output_sharpening\":100"), "{text}");
}

#[test]
fn settings_read_from_a_file_are_clamped_and_checked() {
    let mut s = rich();
    s.contrast = ym_core::settings::PercentSetting::manual_value(400.0);
    s.rolloff.highlights = -3.0;
    s.curve.red = vec![[0.9, 0.1], [0.1, 0.9]];
    let (clean, clamped) = s.validated().unwrap();
    assert_eq!(
        clean.contrast,
        ym_core::settings::PercentSetting::manual_value(100.0)
    );
    assert_eq!(clean.rolloff.highlights, 0.0);
    assert_eq!(clean.curve.red, vec![[0.1, 0.9], [0.9, 0.1]]);
    for f in [
        SettingsField::Contrast,
        SettingsField::Rolloff,
        SettingsField::Curve,
    ] {
        assert!(clamped.contains(&f), "{f:?}");
    }
    let mut bad = rich();
    bad.curve.luma = vec![[0.0, 0.0]];
    assert!(bad.validated().is_err());
}

// ---------------------------------------------------------------------------------------------
// Resolution against estimates
// ---------------------------------------------------------------------------------------------

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn resolve_with(
    settings: &DevelopSettings,
    est: &ym_core::auto::AutoEstimates,
    profile: &ym_profiles::profile::Profile,
    pick: Option<[f64; 3]>,
) -> ResolvedSettings {
    ResolvedSettings::resolve(PhotoId(1), settings, est, profile, pick).unwrap()
}

#[test]
fn auto_values_resolve_against_the_estimates_and_say_so() {
    let dir = tempfile::tempdir().unwrap();
    let scene =
        Scene::new(512, 384, light(PROFILE, 4300.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let summary = est.summary();
    let r = resolve_with(&DevelopSettings::default(), &est, &profile, None);
    assert!(r.auto.exposure && r.auto.light_source && r.auto.white_balance);
    assert!(r.auto.sharpen && r.auto.noise_reduction);
    assert_eq!(r.values.exposure_ev, summary.exposure_ev);
    assert_eq!(
        (r.values.kelvin, r.values.tint),
        (summary.kelvin, summary.tint)
    );
    assert!(
        (r.values.kelvin - 4300.0).abs() <= 50.0,
        "{}",
        r.values.kelvin
    );
    assert_eq!(r.values.sharpen, est.sharpen);
    assert_eq!(r.values.noise_reduction, est.noise_reduction);
    assert_eq!(r.estimates, summary);
    // ISO 400: L_auto = 10 (1 + 2) = 30, as the Automatic card's "Luminance 30".
    assert!((r.values.noise_reduction.luminance - 30.0).abs() < 1e-4);
    assert_eq!(
        r.settings,
        DevelopSettings::default(),
        "stored settings keep their Auto"
    );

    // Editing a value flips its flag and shows the user's number.
    let mut s = DevelopSettings::default();
    s.apply_patch(&SettingsPatch::exposure_ev(-0.5), None)
        .unwrap();
    s.apply_patch(&SettingsPatch::noise_reduction(70.0, 10.0, 5.0), None)
        .unwrap();
    s.apply_patch(&SettingsPatch::white_balance(3300.0, 4.0), None)
        .unwrap();
    let r = resolve_with(&s, &est, &profile, None);
    assert!(!r.auto.exposure && !r.auto.noise_reduction && !r.auto.white_balance);
    assert!(r.auto.sharpen && r.auto.light_source);
    assert_eq!(r.values.exposure_ev, -0.5);
    assert_eq!((r.values.kelvin, r.values.tint), (3300.0, 4.0));
    assert_eq!(r.values.noise_reduction.luminance, 70.0);
}

#[test]
fn a_light_preset_shows_its_kelvin_while_white_balance_is_auto() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(512, 384, light(PROFILE, 4300.0), Layout::Patchwork, 0.4);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let with = |l: LightSource| {
        let mut s = DevelopSettings::default();
        s.light_source = l;
        resolve_with(&s, &est, &profile, None)
    };
    for (preset, kelvin) in [
        (LightSource::Day, 5500.0),
        (LightSource::Shade, 7500.0),
        (LightSource::Tungsten, 3200.0),
    ] {
        let r = with(preset);
        assert_eq!(r.values.kelvin, kelvin, "{preset:?}");
        assert_eq!(r.values.tint, 0.0, "{preset:?}");
        assert!(!r.auto.light_source && r.auto.white_balance);
    }
    // LED and fluorescent keep the estimated chromaticity.
    let auto = with(LightSource::Auto);
    for preset in [LightSource::Led, LightSource::Fluorescent] {
        let r = with(preset);
        assert!(
            (r.values.kelvin - auto.values.kelvin).abs() < 1e-9,
            "{preset:?}"
        );
    }
    // A manual white balance wins over the readout of any preset.
    let mut s = DevelopSettings::default();
    s.light_source = LightSource::Tungsten;
    s.apply_patch(&SettingsPatch::white_balance(6000.0, -3.0), None)
        .unwrap();
    let r = resolve_with(&s, &est, &profile, None);
    assert_eq!((r.values.kelvin, r.values.tint), (6000.0, -3.0));
}

#[test]
fn a_picked_white_shows_the_picked_light_while_white_balance_is_auto() {
    // The camera says daylight, the scene is lit at 3000 K; picking a neutral patch finds it.
    let dir = tempfile::tempdir().unwrap();
    let truth = light(PROFILE, 3000.0);
    let scene = Scene::new(512, 384, truth, Layout::Patchwork, 0.5).noise(1e-4, 1e-6);
    let path = dir.path().join("p.ARW");
    scene
        .builder(MODEL)
        .wb(Some(common::wb_levels(light(PROFILE, 6500.0))))
        .write_to(&path, &scene.mosaic())
        .unwrap();
    let (raw, est, profile) = estimates_of_file(&path);
    let before = resolve_with(&DevelopSettings::default(), &est, &profile, None);
    assert!(
        (before.values.kelvin - 6500.0).abs() <= 50.0,
        "{}",
        before.values.kelvin
    );
    let point = [0.4167, 0.125];
    let sample = pick_white_sample(&raw, point, est.initial_gains().unwrap()).unwrap();
    let mut s = DevelopSettings::default();
    s.apply_patch(
        &parse_patch(&format!(
            r#"{{"white_pick": [{}, {}]}}"#,
            point[0], point[1]
        )),
        None,
    )
    .unwrap();
    let after = resolve_with(&s, &est, &profile, Some(sample));
    assert!(
        after.auto.white_balance,
        "the pick leaves white_balance on Auto"
    );
    assert!(
        (after.values.kelvin - 3000.0).abs() <= 100.0,
        "{}",
        after.values.kelvin
    );
    // Without the sample the pick has no effect on the readout.
    let ignored = resolve_with(&s, &est, &profile, None);
    assert_eq!(ignored.values.kelvin, before.values.kelvin);
}

#[test]
fn a_pick_on_a_blown_out_spot_is_refused_with_the_message_of_the_spec() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(256, 192, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let mut s = DevelopSettings::default();
    s.white_pick = Some([0.5, 0.5]);
    // A sample at 95 percent of the clip level of green.
    let err = ResolvedSettings::resolve(PhotoId(1), &s, &est, &profile, Some([0.2, 0.95, 0.3]))
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "That spot is too bright to use. Pick a grey or white area that isn't blown out."
    );
}

#[test]
fn pinned_settings_record_the_numbers_without_changing_the_picture() {
    let dir = tempfile::tempdir().unwrap();
    let scene =
        Scene::new(512, 384, light(PROFILE, 5200.0), Layout::Patchwork, 0.3).noise(2e-4, 1e-6);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let s = DevelopSettings::default();
    let r = resolve_with(&s, &est, &profile, None);
    let pinned = r.pinned_settings();
    assert_eq!(
        pinned.exposure,
        ym_core::settings::ExposureSetting::manual_ev(r.values.exposure_ev)
    );
    assert_eq!(pinned.sharpen, AutoOr::Manual(r.values.sharpen));
    assert_eq!(
        pinned.noise_reduction,
        AutoOr::Manual(r.values.noise_reduction)
    );
    assert!(
        pinned.white_balance.is_auto(),
        "a manual white balance would not render like Auto"
    );
    // Same fused chain, same resolved values.
    let a = est.fused_chain(&profile, &s, None).unwrap();
    let b = est.fused_chain(&profile, &pinned, None).unwrap();
    assert_eq!(a.matrix(), b.matrix());
    let r2 = resolve_with(&pinned, &est, &profile, None);
    assert_eq!(r2.values, r.values);
    // The pinned document is a valid settings document.
    let (_, v) = schema();
    let doc = serde_json::to_value(&pinned).unwrap();
    assert!(problems(&v, &doc).is_empty(), "{:?}", problems(&v, &doc));
    // And the manual white balance at the shown Kelvin and tint is *not* the Auto picture: it
    // differs by the tint's green gain (so Auto stays Auto in the pinned document).
    let mut manual = s.clone();
    manual
        .apply_patch(&SettingsPatch::white_balance(r.values.kelvin, 25.0), None)
        .unwrap();
    let m = est.fused_chain(&profile, &manual, None).unwrap();
    assert_ne!(a.matrix(), m.matrix());
}

// ---------------------------------------------------------------------------------------------
// Colour warp (docs/02 C9b)
// ---------------------------------------------------------------------------------------------

fn pin_at(source: [f64; 2], target: [f64; 2]) -> ColorWarpPin {
    ColorWarpPin::from(WarpPin::moving(source, target))
}

#[test]
fn a_warp_pin_may_leave_out_everything_but_its_source_and_is_written_whole() {
    // Only the source: the rest are the defaults, and the target is the source.
    let warp: ColorWarp =
        serde_json::from_value(json!({ "pins": [{ "source": [0.12, 0.03] }] })).unwrap();
    assert_eq!(
        warp.pins,
        vec![ColorWarpPin {
            source: [0.12, 0.03],
            target: [0.12, 0.03],
            locked: false,
            range: 0.04,
            shadows: 1.0,
            highlights: 1.0,
            pivot_ev: 0.0,
            exposure_ev: 0.0,
        }]
    );
    // The core writes every field, and what it writes reads back as it was.
    let written = serde_json::to_value(&warp).unwrap();
    assert_eq!(
        keys(&written["pins"][0]),
        [
            "exposure_ev",
            "highlights",
            "locked",
            "pivot_ev",
            "range",
            "shadows",
            "source",
            "target"
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );
    assert_eq!(serde_json::from_value::<ColorWarp>(written).unwrap(), warp);
    // Every field given: kept as given.
    let full = json!({ "source": [0.12, 0.03], "target": [0.08, 0.1], "locked": false,
        "range": 0.06, "shadows": 0.5, "highlights": 0.75, "pivot_ev": -1, "exposure_ev": 0.5 });
    let pin: ColorWarpPin = serde_json::from_value(full.clone()).unwrap();
    assert_eq!(norm(serde_json::to_value(pin).unwrap()), norm(full));
    // A settings document round-trips with and without the optional fields.
    let s = rich();
    let back: DevelopSettings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.color_warp, s.color_warp);
    let (short, clamped) = DevelopSettings::from_json(
        r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "target": [0.1, 0.05] }, { "source": [0.0, -0.1], "locked": true }] } }"#,
        &LookId::builtin("neutral"),
    )
    .unwrap();
    assert!(clamped.is_empty(), "{clamped:?}");
    assert_eq!(
        short.color_warp.pins,
        vec![
            pin_at([0.1, 0.0], [0.1, 0.05]),
            ColorWarpPin::from(WarpPin::locked_at([0.0, -0.1]))
        ]
    );
    // A document without the field, or with an empty object, has no pins.
    let (none, _) = DevelopSettings::from_json("{}", &LookId::builtin("neutral")).unwrap();
    assert!(none.color_warp.pins.is_empty());
    assert_eq!(
        serde_json::from_str::<DevelopSettings>(r#"{ "color_warp": {} }"#)
            .unwrap()
            .color_warp,
        ColorWarp::default()
    );
    // Strict about what is not a pin.
    for bad in [
        r#"{ "color_warp": { "pins": [{ "target": [0.1, 0.0] }] } }"#,
        r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "strength": 2 }] } }"#,
        r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "target": null }] } }"#,
        r#"{ "color_warp": { "pins": [{ "source": [0.1] }] } }"#,
        r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "locked": "yes" }] } }"#,
        r#"{ "color_warp": { "pins": [], "strength": 2 } }"#,
    ] {
        assert!(
            serde_json::from_str::<DevelopSettings>(bad).is_err(),
            "{bad} should be refused"
        );
    }
}

#[test]
fn the_warp_keeps_16_pins_and_clamps_each_like_the_stage_does() {
    let neutral = LookId::builtin("neutral");
    // 17 pins: the 17th is cut, and the field is reported.
    let mut s = DevelopSettings::default();
    s.color_warp.pins = (0..17)
        .map(|i| pin_at([0.01 * f64::from(i), 0.1], [0.01 * f64::from(i), 0.12]))
        .collect();
    let (v, clamped) = s.clone().validated().unwrap();
    assert_eq!(v.color_warp.pins.len(), 16);
    assert_eq!(v.color_warp.pins[..], s.color_warp.pins[..16]);
    assert_eq!(clamped, vec![SettingsField::ColorWarp]);
    // Points scaled back to |c| <= 0.5, ranges and weights clamped, a locked pin's target is its
    // source and its weights, pivot and exposure the defaults: what `WarpPin::clamped` does.
    let wild = ColorWarpPin {
        source: [3.0, -4.0],
        target: [0.0, 0.9],
        locked: false,
        range: 2.0,
        shadows: -1.0,
        highlights: 7.0,
        pivot_ev: -20.0,
        exposure_ev: 9.0,
    };
    let locked = ColorWarpPin {
        source: [0.1, 0.0],
        target: [0.3, 0.3],
        locked: true,
        range: 0.001,
        shadows: 0.2,
        highlights: 0.3,
        pivot_ev: 2.0,
        exposure_ev: 1.0,
    };
    let (v, clamped) = DevelopSettings::from_value(
        json!({ "color_warp": { "pins": [wild, locked] } }),
        &neutral,
    )
    .unwrap();
    assert_eq!(clamped, vec![SettingsField::ColorWarp]);
    let [a, b] = [v.color_warp.pins[0], v.color_warp.pins[1]];
    assert!((a.source[0] - 0.3).abs() < 1e-12 && (a.source[1] + 0.4).abs() < 1e-12);
    assert!(a.source[0].hypot(a.source[1]) <= 0.5);
    assert!((a.target[1] - 0.5).abs() < 1e-12 && a.target[0] == 0.0);
    assert_eq!(
        (a.range, a.shadows, a.highlights, a.pivot_ev, a.exposure_ev),
        (0.25, 0.0, 1.0, -6.0, 2.0)
    );
    assert_eq!(
        b,
        ColorWarpPin {
            source: [0.1, 0.0],
            target: [0.1, 0.0],
            locked: true,
            range: 0.01,
            shadows: 1.0,
            highlights: 1.0,
            pivot_ev: 0.0,
            exposure_ev: 0.0,
        }
    );
    // The settings are what the stage renders: clamping them again is the stage's own clamp.
    for p in &v.color_warp.pins {
        let c = WarpPin::from(*p).clamped();
        assert_eq!(c.locked, p.locked);
        assert_eq!(
            (c.range, c.shadows, c.highlights, c.pivot_ev, c.exposure_ev),
            (p.range, p.shadows, p.highlights, p.pivot_ev, p.exposure_ev)
        );
        for (x, y) in c
            .source
            .iter()
            .chain(&c.target)
            .zip(p.source.iter().chain(&p.target))
        {
            assert!((x - y).abs() <= 1e-15, "{x} {y}");
        }
    }
    // Checking again changes nothing, also for points that land just outside the disc when they
    // are scaled back (about one in fifty does): a later patch must not report the warp.
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0
    };
    let mut s = DevelopSettings::default();
    s.color_warp.pins = (0..16)
        .map(|_| pin_at([next(), next()], [next(), next()]))
        .collect();
    for _ in 0..200 {
        let (once, _) = s.clone().validated().unwrap();
        let (twice, again) = once.clone().validated().unwrap();
        assert_eq!(twice, once);
        assert!(again.is_empty(), "{again:?}");
        for p in &once.color_warp.pins {
            assert!(p.source[0].hypot(p.source[1]) <= 0.5);
            assert!(p.target[0].hypot(p.target[1]) <= 0.5);
        }
        for p in &mut s.color_warp.pins {
            *p = pin_at([next(), next()], [next(), next()]);
        }
    }
    // A number that is not finite is an error, as everywhere else; nothing is applied.
    let breaks: [fn(&mut ColorWarpPin); 4] = [
        |p: &mut ColorWarpPin| p.source[1] = f64::NAN,
        |p: &mut ColorWarpPin| p.target[0] = f64::INFINITY,
        |p: &mut ColorWarpPin| p.range = f64::NAN,
        |p: &mut ColorWarpPin| p.exposure_ev = f64::NEG_INFINITY,
    ];
    for broken in breaks {
        let mut s = DevelopSettings::default();
        let mut p = pin_at([0.1, 0.0], [0.1, 0.1]);
        broken(&mut p);
        s.color_warp.pins.push(p);
        assert!(
            matches!(s.clone().validated(), Err(SettingsError::NotFinite(n)) if n.starts_with("color_warp")),
            "{p:?}"
        );
        let mut t = DevelopSettings::default();
        let patch = SettingsPatch {
            color_warp: Some(s.color_warp.clone()),
            ..SettingsPatch::default()
        };
        assert!(t.apply_patch(&patch, None).is_err());
        assert_eq!(t, DevelopSettings::default());
    }
}

#[test]
fn a_warp_patch_replaces_the_whole_list_and_reset_removes_the_pins() {
    let neutral = LookId::builtin("neutral");
    assert_eq!(SettingsField::ColorWarp.name(), "color_warp");
    assert_eq!(SettingsField::ColorWarp.group(), StageGroup::Grade);
    assert!(SettingsField::ALL.contains(&SettingsField::ColorWarp));
    let mut s = DevelopSettings::default();
    let two = parse_patch(
        r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "target": [0.1, 0.05] }, { "source": [0.0, 0.1], "locked": true }] } }"#,
    );
    let report = apply(&mut s, two);
    assert_eq!(report.changed, vec![SettingsField::ColorWarp]);
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    assert_eq!(s.color_warp.pins.len(), 2);
    // A patch is the whole list: one pin replaces the two.
    let one = parse_patch(
        r#"{ "color_warp": { "pins": [{ "source": [-0.1, 0.0], "target": [-0.12, 0.0] }] } }"#,
    );
    apply(&mut s, one);
    assert_eq!(s.color_warp.pins, vec![pin_at([-0.1, 0.0], [-0.12, 0.0])]);
    // A patch that leaves the warp out keeps it, and reports nothing about it.
    let report = apply(&mut s, parse_patch(r#"{ "contrast": 10 }"#));
    assert_eq!(report.changed, vec![SettingsField::Contrast]);
    assert_eq!(s.color_warp.pins.len(), 1);
    // A value out of range in the patch is clamped and reported.
    let report = apply(
        &mut s,
        parse_patch(r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "range": 9 }] } }"#),
    );
    assert_eq!(report.clamped, vec![SettingsField::ColorWarp]);
    assert_eq!(s.color_warp.pins[0].range, 0.25);
    // The same list again changes nothing.
    let same = SettingsPatch::of_field(&s, SettingsField::ColorWarp);
    assert!(apply(&mut s, same.clone()).changed.is_empty());
    assert_eq!(
        serde_json::to_value(&same).unwrap(),
        json!({ "color_warp": serde_json::to_value(&s.color_warp).unwrap() })
    );
    // The card's Reset sends no pins; `reset_field` and the Reset patch agree.
    let mut by_patch = s.clone();
    apply(
        &mut by_patch,
        parse_patch(r#"{ "color_warp": { "pins": [] } }"#),
    );
    let mut by_reset = s.clone();
    by_reset.reset_field(SettingsField::ColorWarp, &neutral);
    let mut by_reset_patch = s.clone();
    apply(
        &mut by_reset_patch,
        SettingsPatch::reset(SettingsField::ColorWarp, &neutral),
    );
    for t in [&by_patch, &by_reset, &by_reset_patch] {
        assert!(t.color_warp.pins.is_empty());
        assert_eq!(t.diff(&s), vec![SettingsField::ColorWarp]);
    }
}

#[test]
fn copy_settings_takes_the_warp_and_resolved_values_carry_it() {
    // Copy settings takes every field but the photo's own geometry and Pick white point.
    assert!(!COPY_LEAVES_OUT.contains(&SettingsField::ColorWarp));
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(
        256,
        192,
        light("sony-ilce-7rm4", 5200.0),
        Layout::Patchwork,
        0.5,
    );
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", "ILCE-7RM4"));
    let s = rich();
    let resolved = ResolvedSettings::resolve(PhotoId(3), &s, &est, &profile, None).unwrap();
    assert_eq!(resolved.values.color_warp, s.color_warp);
    assert_eq!(resolved.pinned_settings().color_warp, s.color_warp);
    let patch = SettingsPatch::copied_from(&resolved);
    assert_eq!(patch.color_warp.as_ref(), Some(&s.color_warp));
    // Pasted onto a photo with pins of its own, the copied list replaces them.
    let mut other = DevelopSettings::default();
    other.color_warp.pins = vec![pin_at([0.0, -0.2], [0.05, -0.2])];
    other.apply_patch(&patch, None).unwrap();
    assert_eq!(other.color_warp, s.color_warp);
}
