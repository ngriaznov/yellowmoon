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
    Curves, DevelopSettings, FieldRef, LENS_AMOUNT_RANGE, LensKind, LensPatch, LensSettings,
    NoiseReductionValue, NormRect, Orientation, ResolvedSettings, Rolloff, SettingsError,
    SettingsField, SettingsPatch, SharpenValue, StageGroup,
};
use ym_core::settings::{
    BrushMask, BrushStroke, CorrectorNode, GradePatch, MaskItem, MaskSettings, MaskShape, NodeId,
    NodeLink, NodePatch, NodeTrim, ObjectMask,
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
    s.output_mut().look_intensity = 62.5;
    s.input_mut().exposure = AutoOr::Manual(ym_core::settings::ExposureValue { ev: -1.25 });
    s.main_grade_mut().contrast = ym_core::settings::PercentSetting::manual_value(12.0);
    s.main_grade_mut().brightness = -7.5;
    s.main_grade_mut().highlights = ym_core::settings::PercentSetting::manual_value(-30.0);
    s.main_grade_mut().shadows = ym_core::settings::PercentSetting::manual_value(25.0);
    s.main_grade_mut().white = 4.0;
    s.main_grade_mut().black = ym_core::settings::PercentSetting::manual_value(-9.0);
    s.input_mut().light_source = LightSource::Tungsten;
    s.input_mut().white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 3250.0,
        tint: -4.5,
    });
    s.main_grade_mut().color_balance = ColorBalance {
        shadows: ym_core::settings::BalanceSetting::manual_point([0.2, -0.3]),
        midtones: [0.0, 0.0],
        highlights: ym_core::settings::BalanceSetting::manual_point([-0.5, 0.5]),
    };
    s.main_grade_mut().saturation = ym_core::settings::PercentSetting::manual_value(15.0);
    s.main_grade_mut().color_warp = ColorWarp {
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
                saturation: -0.4,
            },
            ColorWarpPin::from(WarpPin::locked_at([-0.05, -0.1])),
        ],
    };
    s.input_mut().lens = LensSettings {
        kind: LensKind::Vintage,
        flares: 60.0,
        edges: 45.0,
        anamorphic: 30.0,
    };
    s.input_mut().sharpen = AutoOr::Manual(SharpenValue {
        amount: 150.0,
        radius_px: 1.1,
    });
    s.output_mut().output_sharpening = 140.0;
    s.input_mut().noise_reduction = AutoOr::Manual(NoiseReductionValue {
        luminance: 35.0,
        color: 60.0,
        detail: 20.0,
    });
    s.main_grade_mut().curve = Curves {
        luma: vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.82], [1.0, 1.0]],
        red: vec![[0.0, 0.02], [1.0, 1.0]],
        green: vec![[0.0, 0.0], [1.0, 1.0]],
        blue: vec![[0.0, 0.0], [0.5, 0.45], [1.0, 0.98]],
    };
    s.input_mut().decode.neural_demosaic = false;
    s.input_mut().decode.chromatic_aberration = false;
    s.input_mut().rolloff = Rolloff {
        highlights: 0.8,
        low_light: 0.1,
    };
    s.input_mut().crop = Some(Crop {
        rect: NormRect {
            x: 0.1,
            y: 0.2,
            width: 0.7,
            height: 0.6,
        },
        angle_deg: -2.5,
        aspect: CropAspect::FourByFive,
        // On, so the JSON carries the field (`false` is left out).
        auto: true,
    });
    s.input_mut().orientation = Orientation::Deg90;
    s.input_mut().white_pick = Some([0.25, 0.75]);
    // Main's own corrector fields.
    s.main_mut().trim = NodeTrim {
        exposure_ev: 0.25,
        temperature: -10.0,
        tint: 5.0,
    };
    // Node 2: a corrector with a trim and a row of every kind of mask; node 3: a plain one, in a
    // chain Input -> 1 -> 2 -> 3 -> Output.
    let mut mountain = CorrectorNode::default();
    mountain.name = "Mountain".to_owned();
    mountain.pos = [1.0, 0.0];
    mountain.trim.exposure_ev = 0.35;
    mountain.grade.highlights = ym_core::settings::PercentSetting::manual_value(-12.0);
    mountain.masks = every_mask();
    s.nodes.extra.insert(2, mountain);
    let mut sky = CorrectorNode::default();
    sky.name = "Sky".to_owned();
    sky.bypass = true;
    sky.pos = [2.0, 0.0];
    sky.trim.temperature = -18.0;
    s.nodes.extra.insert(3, sky);
    s.output_mut().pos = [3.0, 0.0];
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Corrector(3)),
        NodeLink::picture(NodeId::Corrector(3), NodeId::Output),
    ];
    s
}

/// A node's masks with one row of each of the nine kinds.
fn every_mask() -> MaskSettings {
    let row = |shape: MaskShape| MaskItem::new(shape);
    let mut items = vec![
        row(MaskShape::Object(ObjectMask {
            points: vec![(0.41, 0.33, true), (0.52, 0.31, false)],
            box_: Some([0.3, 0.2, 0.6, 0.5]),
            feather: 0.1,
            grow: -0.2,
            refine: 0.7,
        })),
        row(MaskShape::Subject(Default::default())),
        row(MaskShape::People(Default::default())),
        row(MaskShape::Background(Default::default())),
        row(MaskShape::Linear(Default::default())),
        row(MaskShape::Radial(Default::default())),
        row(MaskShape::Brush(BrushMask {
            strokes: vec![BrushStroke {
                size: 0.05,
                hardness: 0.8,
                erase: true,
                points: vec![[0.4, 0.5], [0.45, 0.52]],
            }],
            ..Default::default()
        })),
        row(MaskShape::ColorRange(Default::default())),
        row(MaskShape::BrightnessRange(Default::default())),
    ];
    items[1].mode = ym_core::settings::MaskMode::Subtract;
    items[2].invert = true;
    items[3].enabled = false;
    items[4].opacity = 0.5;
    MaskSettings {
        invert: true,
        items,
    }
}

// ---------------------------------------------------------------------------------------------
// JSON shape
// ---------------------------------------------------------------------------------------------

/// What `defaultSettings()` in `src/api.mock.ts` produces: the frontend's idea of the defaults (with
/// Auto contrast, highlights, shadows, black, saturation and the shadows and highlights wheels,
/// `docs/19` section 5).
fn typescript_defaults() -> Value {
    let curve = json!({
        "luma": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]],
        "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]]
    });
    json!({ "nodes": {
        "items": [
            { "id": "input", "kind": "input", "pos": [-1, 0],
              "exposure": { "kind": "auto" },
              "light_source": "auto",
              "white_balance": { "kind": "auto" },
              "white_pick": null,
              "rolloff": { "highlights": 0.6, "low_light": 0.5 },
              "decode": {
                  "neural_demosaic": true, "highlight_recovery": true,
                  "lens_corrections": true, "chromatic_aberration": true
              },
              "noise_reduction": { "kind": "auto" },
              "sharpen": { "kind": "auto" },
              "lens": { "kind": "default", "flares": 0, "edges": 0, "anamorphic": 0 },
              "crop": null,
              "orientation": 0 },
            { "id": 1, "kind": "corrector", "name": "Main", "bypass": false, "pos": [0, 0],
              "trim": { "exposure_ev": 0, "temperature": 0, "tint": 0 },
              "grade": {
                  "contrast": { "kind": "auto" },
                  "brightness": 0,
                  "highlights": { "kind": "auto" },
                  "shadows": { "kind": "auto" },
                  "white": 0,
                  "black": { "kind": "auto" },
                  "color_balance": {
                      "shadows": { "kind": "auto" },
                      "midtones": [0, 0],
                      "highlights": { "kind": "auto" }
                  },
                  "saturation": { "kind": "auto" },
                  "color_warp": { "pins": [] },
                  "curve": curve },
              "masks": { "invert": false, "items": [] } },
            { "id": "output", "kind": "output", "pos": [1, 0],
              "look": "builtin:neutral", "look_intensity": 100, "output_sharpening": 100 }
        ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
        ]
    } })
}

/// A node-first document that holds one field, given by the field's name and value: it goes to
/// the node the field belongs to (the Input's, the Output's or Main's grade).
fn field_doc(field: Value) -> Value {
    let (mut input, mut output, mut grade) = (
        json!({ "id": "input" }),
        json!({ "id": "output" }),
        json!({}),
    );
    for (k, v) in field.as_object().expect("a field object") {
        match k.as_str() {
            "look" | "look_intensity" | "output_sharpening" => output[k] = v.clone(),
            "contrast" | "brightness" | "highlights" | "shadows" | "white" | "black"
            | "color_balance" | "saturation" | "color_warp" | "curve" => grade[k] = v.clone(),
            _ => input[k] = v.clone(),
        }
    }
    json!({ "nodes": { "items": [input, { "id": 1, "grade": grade }, output] } })
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
    // So is a document that lists the three nodes and nothing in them.
    let bare: DevelopSettings = serde_json::from_value(json!({ "nodes": { "items": [
        { "id": "input" }, { "id": 1 }, { "id": "output" } ] } }))
    .unwrap();
    assert_eq!(bare, DevelopSettings::default());
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
    let doc = serde_json::to_value(&settings).unwrap();
    let items = &doc["nodes"]["items"];
    check("DevelopSettings", doc.clone());
    check("NodesSettings", doc["nodes"].clone());
    // A node of the document is `{ kind } & <its struct>`: `NodeItem` carries the tag, the interfaces the fields
    // (the generated bindings say the same, `src/api/types.ts`).
    let untagged = |item: &Value| {
        let mut item = item.clone();
        assert!(item.as_object_mut().unwrap().remove("kind").is_some());
        item
    };
    check("InputNode", untagged(&items[0]));
    check("CorrectorNode", untagged(&items[1]));
    check("OutputNode", untagged(&items[4]));
    check("NodeTrim", items[1]["trim"].clone());
    check("NodeGrade", items[1]["grade"].clone());
    check("MaskSettings", items[2]["masks"].clone());
    check("NodeLink", doc["nodes"]["links"][0].clone());
    check("NodeRef", doc["nodes"]["links"][0]["from"].clone());
    check(
        "ColorBalance",
        serde_json::to_value(settings.main_grade().color_balance).unwrap(),
    );
    check(
        "ColorWarp",
        serde_json::to_value(&settings.main_grade().color_warp).unwrap(),
    );
    check(
        "ColorWarpPin",
        serde_json::to_value(settings.main_grade().color_warp.pins[0]).unwrap(),
    );
    check(
        "LensSettings",
        serde_json::to_value(settings.input().lens).unwrap(),
    );
    check(
        "Curves",
        serde_json::to_value(&settings.main_grade().curve).unwrap(),
    );
    check(
        "DecodeOptions",
        serde_json::to_value(settings.input().decode).unwrap(),
    );
    check(
        "Rolloff",
        serde_json::to_value(settings.input().rolloff).unwrap(),
    );
    check(
        "Crop",
        serde_json::to_value(settings.input().crop.unwrap()).unwrap(),
    );
    check(
        "NormRect",
        serde_json::to_value(settings.input().crop.unwrap().rect).unwrap(),
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
        apple_pending: false,
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
            lens: None,
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
    // Grouped by node (`docs/23` section 3.9): the flags and the estimates for the Input and for Main,
    // and beside them what belongs to the photo (Apple Auto, `docs/19` sections 10 and 12).
    assert_eq!(ts_interface_keys(&source, "AutoFlags"), keys(&v["auto"]));
    assert_eq!(
        ts_interface_keys(&source, "InputAutoFlags"),
        keys(&v["auto"]["input"])
    );
    assert_eq!(
        ts_interface_keys(&source, "MainAutoFlags"),
        keys(&v["auto"]["main"])
    );
    assert_eq!(
        ts_interface_keys(&source, "AutoEstimates"),
        keys(&v["estimates"])
    );
    assert_eq!(
        ts_interface_keys(&source, "InputEstimates"),
        keys(&v["estimates"]["input"])
    );
    assert_eq!(
        ts_interface_keys(&source, "MainEstimates"),
        keys(&v["estimates"]["main"])
    );
    assert_eq!(v["auto"]["main"]["contrast"], json!(true));
    assert_eq!(v["estimates"]["source"], json!("rule"));
    assert_eq!(v["estimates"]["apple"], json!({ "state": "off" }));
    assert_eq!(
        (
            v["estimates"]["main"]["contrast"].clone(),
            v["estimates"]["main"]["saturation"].clone()
        ),
        (json!(0.0), json!(0.0))
    );
    // ResolvedValues: the Input, Main, the other correctors and the Output with their Auto resolved.
    assert_eq!(
        ts_interface_keys(&source, "ResolvedValues"),
        keys(&v["values"])
    );
    assert_eq!(
        ts_interface_keys(&source, "ResolvedInput"),
        keys(&v["values"]["input"])
    );
    assert_eq!(
        ts_interface_keys(&source, "ResolvedCorrector"),
        keys(&v["values"]["main"])
    );
    assert_eq!(
        ts_interface_keys(&source, "GradeValues"),
        keys(&v["values"]["main"]["grade"])
    );
    assert_eq!(
        ts_interface_keys(&source, "ResolvedOutput"),
        keys(&v["values"]["output"])
    );
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
    let items = v["nodes"]["items"].as_array().unwrap();
    let kinds: Vec<_> = items
        .iter()
        .map(|i| (i["id"].clone(), i["kind"].clone()))
        .collect();
    assert_eq!(
        kinds,
        [
            (json!("input"), json!("input")),
            (json!(1), json!("corrector")),
            (json!(2), json!("corrector")),
            (json!(3), json!("corrector")),
            (json!("output"), json!("output")),
        ],
        "the Input, the correctors by id, the Output"
    );
    let input = &items[0];
    assert_eq!(input["exposure"], json!({ "kind": "manual", "ev": -1.25 }));
    assert_eq!(
        input["white_balance"],
        json!({ "kind": "manual", "kelvin": 3250.0, "tint": -4.5 })
    );
    assert_eq!(input["orientation"], json!(90));
    assert_eq!(input["light_source"], json!("tungsten"));
    assert_eq!(input["crop"]["aspect"], json!("four_by_five"));
    assert_eq!(items[4]["look"], json!("arri:3000:film-warm"));
    // The masks: every kind is one flat object with its `kind`.
    let rows = items[2]["masks"]["items"].as_array().unwrap();
    assert_eq!(rows.len(), 9);
    assert_eq!(rows[0]["kind"], json!("object"));
    assert_eq!(
        rows[0]["points"],
        json!([[0.41, 0.33, true], [0.52, 0.31, false]])
    );
    assert_eq!(rows[8]["kind"], json!("brightness_range"));
    assert_eq!(rows[1]["mode"], json!("subtract"));
    assert_eq!(v["nodes"]["links"].as_array().unwrap().len(), 4);
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
        r#"{"lens": {"kind": "leica"}}"#,
        r#"{"lens": {"kind": "Vintage"}}"#,
        r#"{"lens": {"grain": 3}}"#,
        r#"{"lens": {"flares": "a lot"}}"#,
        r#"{"lens": "vintage"}"#,
    ] {
        let doc = field_doc(serde_json::from_str(bad).unwrap());
        assert!(
            serde_json::from_value::<DevelopSettings>(doc).is_err(),
            "{bad} should be refused"
        );
    }
    // A key at the top that is not `nodes` is an unknown field; so is one in a node.
    for bad in [
        r#"{"nodez": {}}"#,
        r#"{"nodes": {}, "extra": 1}"#,
        r#"{"nodes": {"items": [], "linkz": []}}"#,
        r#"{"nodes": {"items": [{"id": "input", "zoom": 2}]}}"#,
        r#"{"nodes": {"items": [{"id": 1, "grade": {"contrsat": 2}}]}}"#,
        r#"{"nodes": {"items": [{"id": 2, "kind": "corrector", "masks": {"items": [{"kind": "radial", "colour": 2}]}}]}}"#,
        r#"{"nodes": {"items": [{"id": 2, "kind": "corrector", "masks": {"items": [{"kind": "blob"}]}}]}}"#,
        r#"{"nodes": {"items": [{"id": 2, "kind": "corrector", "masks": {"items": [{"mode": "add"}]}}]}}"#,
    ] {
        assert!(
            DevelopSettings::from_json(bad, &LookId::neutral()).is_err(),
            "{bad} should be refused"
        );
    }
    // A partial document is fine.
    let partial: DevelopSettings = serde_json::from_value(field_doc(
        json!({ "contrast": 10, "decode": { "lens_corrections": false } }),
    ))
    .unwrap();
    assert_eq!(
        partial.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(10.0)
    );
    assert!(!partial.input().decode.lens_corrections && partial.input().decode.neural_demosaic);
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
        s.input_mut().light_source = light_source;
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
        s.input_mut().crop.as_mut().unwrap().aspect = aspect;
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
        s.input_mut().orientation = o;
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
            &field_doc(json!({ "contrast": 10, "decode": { "lens_corrections": false } }))
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
            "lens of an unknown kind",
            json!({ "lens": { "kind": "leica" } }),
        ),
        (
            "lens kind in capitals",
            json!({ "lens": { "kind": "Vintage" } }),
        ),
        (
            "lens with an unknown field",
            json!({ "lens": { "kind": "vintage", "grain": 3 } }),
        ),
        ("flares above range", json!({ "lens": { "flares": 101 } })),
        ("edges below range", json!({ "lens": { "edges": -1 } })),
        (
            "anamorphic above range",
            json!({ "lens": { "anamorphic": 250 } }),
        ),
        ("flares as text", json!({ "lens": { "flares": "40" } })),
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
        let doc = field_doc(doc);
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
    let defs = &s["$defs"];
    let input = &defs["input_node"]["properties"];
    let output = &defs["output_node"]["properties"];
    let grade = &defs["grade"]["properties"];
    assert_eq!(range(&output["look_intensity"]), LOOK_INTENSITY_RANGE);
    assert_eq!(range(&output["output_sharpening"]), OUTPUT_SHARPENING_RANGE);
    assert_eq!(
        output["output_sharpening"]["default"].as_f64(),
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
        range(&input["rolloff"]["properties"]["highlights"]),
        ROLLOFF_RANGE
    );
    for amount in ["flares", "edges", "anamorphic"] {
        assert_eq!(
            range(&defs["lens"]["properties"][amount]),
            LENS_AMOUNT_RANGE,
            "{amount}"
        );
    }
    // The schema names the seven lenses of `ym_lens::LensKind`, in the menu order, and its default.
    let kinds: Vec<&str> = defs["lens"]["properties"]["kind"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    let wanted: Vec<&str> = LensKind::ALL.iter().map(|k| k.as_str()).collect();
    assert_eq!(kinds, wanted);
    assert_eq!(
        norm(defs["lens"]["default"].clone()),
        norm(serde_json::to_value(LensSettings::default()).unwrap())
    );
    assert_eq!(input["lens"], json!({ "$ref": "#/$defs/lens" }));
    assert_eq!(
        range(&input["crop"]["oneOf"][1]["properties"]["angle_deg"]),
        CROP_ANGLE_RANGE_DEG
    );
    assert_eq!(defs["curve"]["maxItems"], json!(MAX_CURVE_POINTS));
    assert_eq!(defs["curve"]["minItems"], json!(MIN_CURVE_POINTS));
    {
        use ym_color::warp::*;
        let pin = &defs["warp_pin"]["properties"];
        assert_eq!(
            grade["color_warp"]["properties"]["pins"]["maxItems"],
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
        assert_eq!(
            range(&pin["saturation"]),
            (-WARP_SATURATION_LIMIT, WARP_SATURATION_LIMIT)
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
            "saturation",
        ] {
            assert_eq!(
                norm(pin[key]["default"].clone()),
                norm(written[key].clone()),
                "{key}"
            );
        }
        assert_eq!(
            grade["color_warp"]["default"],
            serde_json::to_value(ColorWarp::default()).unwrap()
        );
    }
    // The schema's defaults are the core's defaults.
    let d = serde_json::to_value(DevelopSettings::default()).unwrap();
    let d_output = &d["nodes"]["items"][2];
    assert_eq!(
        norm(output["look"]["default"].clone()),
        norm(d_output["look"].clone())
    );
    assert_eq!(output["look_intensity"]["default"], json!(100));
    assert_eq!(
        input["rolloff"]["properties"]["highlights"]["default"],
        json!(0.6)
    );
    assert_eq!(
        input["rolloff"]["properties"]["low_light"]["default"],
        json!(0.5)
    );
}

/// The owner's "gentler range" (2026-10-30, `docs/02` C7): the default highlight range is 0.6. A document
/// that leaves the field out, or the whole `rolloff` object, reads as the new default; an explicit 1 (a
/// document written before the change, which always carried it) keeps its 1.
#[test]
fn the_highlight_range_defaults_to_0_6_and_an_explicit_value_is_kept() {
    assert_eq!(ym_core::settings::ROLLOFF_HIGHLIGHTS_DEFAULT, 0.6);
    assert_eq!(DevelopSettings::default().input().rolloff.highlights, 0.6);
    let neutral = LookId::builtin("neutral");
    let read = |text: &str| {
        read_fields(text, &neutral)
            .expect("reads")
            .0
            .input()
            .rolloff
    };
    // Absent: the new default, with the toe at its own.
    assert_eq!(
        read("{}"),
        Rolloff {
            highlights: 0.6,
            low_light: 0.5
        }
    );
    assert_eq!(
        read(r#"{ "rolloff": { "low_light": 0.2 } }"#),
        Rolloff {
            highlights: 0.6,
            low_light: 0.2
        }
    );
    assert_eq!(read(r#"{ "rolloff": {} }"#), Rolloff::default());
    // Explicit: kept, 1 included.
    assert_eq!(
        read(r#"{ "rolloff": { "highlights": 1.0, "low_light": 0.5 } }"#),
        Rolloff {
            highlights: 1.0,
            low_light: 0.5
        }
    );
    assert_eq!(
        read(r#"{ "rolloff": { "highlights": 0 } }"#).highlights,
        0.0
    );
    // A patch that leaves the field out does not move it, and the Reset patch sends the new default.
    let mut s = DevelopSettings::default();
    s.input_mut().rolloff.highlights = 1.0;
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"rolloff": {"low_light": 0.9}}}}"#),
    );
    assert_eq!(
        s.input().rolloff,
        Rolloff {
            highlights: 1.0,
            low_light: 0.9
        }
    );
}

// ---------------------------------------------------------------------------------------------
// Patches
// ---------------------------------------------------------------------------------------------

fn apply(s: &mut DevelopSettings, patch: SettingsPatch) -> ym_core::settings::PatchReport {
    s.apply_patch(&patch, None).expect("patch applies")
}

/// The fields of a report, without their nodes.
fn fields(refs: &[ym_core::settings::FieldRef]) -> Vec<SettingsField> {
    refs.iter().map(|r| r.field).collect()
}

/// The node a field of the simple view lives on.
fn home(field: SettingsField) -> NodeId {
    use SettingsField as F;
    if F::Look == field || F::LookIntensity == field || F::OutputSharpening == field {
        NodeId::Output
    } else if field.belongs_to(ym_core::settings::NodeKind::Input) && field != F::Pos {
        NodeId::Input
    } else {
        NodeId::MAIN
    }
}

/// A patch that holds the given fields, by name: each goes to the node the field belongs to (the
/// Input's, the Output's or Main's grade).
fn field_patch(fields: Value) -> Value {
    let (mut input, mut output, mut grade) = (json!({}), json!({}), json!({}));
    for (k, v) in fields.as_object().expect("a field object") {
        match k.as_str() {
            "look" | "look_intensity" | "output_sharpening" => output[k] = v.clone(),
            "contrast" | "brightness" | "highlights" | "shadows" | "white" | "black"
            | "color_balance" | "saturation" | "color_warp" | "curve" => grade[k] = v.clone(),
            _ => input[k] = v.clone(),
        }
    }
    let mut nodes = serde_json::Map::new();
    for (id, node) in [("input", input), ("output", output)] {
        if node.as_object().is_some_and(|o| !o.is_empty()) {
            nodes.insert(id.to_owned(), node);
        }
    }
    if grade.as_object().is_some_and(|o| !o.is_empty()) {
        nodes.insert("1".to_owned(), json!({ "grade": grade }));
    }
    json!({ "nodes": nodes })
}

/// [`field_patch`] of fields written as JSON text.
fn parse_fields(text: &str) -> SettingsPatch {
    serde_json::from_value(field_patch(serde_json::from_str(text).unwrap()))
        .unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// [`field_doc`] of fields written as JSON text, read as a document.
fn read_fields(
    text: &str,
    look: &LookId,
) -> Result<(DevelopSettings, Vec<ym_core::settings::FieldRef>), SettingsError> {
    DevelopSettings::from_value(field_doc(serde_json::from_str(text).unwrap()), look)
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
    assert!(fields(&report.changed).is_empty() && fields(&report.clamped).is_empty());
    assert_eq!(report.first_group(), None);
    assert!(SettingsPatch::default().is_empty());
    assert!(parse_patch("{}").is_empty());
}

#[test]
fn moving_a_slider_sends_a_manual_value_and_reset_sends_auto() {
    let mut s = DevelopSettings::default();
    let report = apply(&mut s, SettingsPatch::exposure_ev(0.3));
    assert_eq!(
        s.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.3)
    );
    assert_eq!(fields(&report.changed), vec![SettingsField::Exposure]);
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
        SettingsPatch::reset(
            home(SettingsField::Exposure),
            SettingsField::Exposure,
            &LookId::builtin("neutral"),
        ),
    );
    assert!(s.input().exposure.is_auto());
}

#[test]
fn values_out_of_range_are_clamped_and_reported() {
    let mut s = DevelopSettings::default();
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"input": {"exposure": {"kind": "manual", "ev": 9}, "rolloff": {"highlights": 1.5, "low_light": -1}, "white_balance": {"kind": "manual", "kelvin": 20000, "tint": -90}, "sharpen": {"kind": "manual", "amount": 900, "radius_px": 0.01}, "noise_reduction": {"kind": "manual", "luminance": 120, "color": -5, "detail": 50}}, "1": {"grade": {"contrast": -250}}, "output": {"look_intensity": 140}}}"#,
        ),
    );
    assert_eq!(
        s.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(4.0)
    );
    assert_eq!(
        s.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(-100.0)
    );
    assert_eq!(s.output().look_intensity, 100.0);
    assert_eq!(
        s.input().rolloff,
        Rolloff {
            highlights: 1.0,
            low_light: 0.0
        }
    );
    assert_eq!(
        s.input().white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 11000.0,
            tint: -50.0
        })
    );
    assert_eq!(
        s.input().sharpen,
        AutoOr::Manual(SharpenValue {
            amount: 300.0,
            radius_px: 0.3
        })
    );
    assert_eq!(
        s.input().noise_reduction,
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
        assert!(fields(&report.clamped).contains(&f), "{f:?} was clamped");
    }
    // look_intensity 140 -> 100 equals the default 100, so it changed nothing and is not a change
    // (but it was clamped).
    assert!(fields(&report.clamped).contains(&SettingsField::LookIntensity));
    assert!(!fields(&report.changed).contains(&SettingsField::LookIntensity));
}

#[test]
fn a_patch_that_cannot_be_applied_changes_nothing() {
    let mut s = rich();
    let before = s.clone();
    // A valid change next to a value that is not a number.
    let patch = SettingsPatch::default().and(
        NodeId::MAIN,
        NodePatch {
            grade: Some(GradePatch {
                contrast: Some(ym_core::settings::PercentSetting::manual_value(10.0)),
                saturation: Some(ym_core::settings::PercentSetting::manual_value(f64::NAN)),
                ..GradePatch::default()
            }),
            ..NodePatch::default()
        },
    );
    let err = s.apply_patch(&patch, None).unwrap_err();
    assert_eq!(
        err,
        SettingsError::NotFinite("nodes.items[1].grade.saturation".to_owned())
    );
    assert_eq!(s, before, "atomic: contrast must not have changed");
    for (label, patch) in [
        (
            "infinite kelvin",
            SettingsPatch::white_balance(f64::INFINITY, 0.0),
        ),
        ("NaN exposure", SettingsPatch::exposure_ev(f64::NAN)),
        (
            "NaN colour balance",
            SettingsPatch::default().and(
                NodeId::MAIN,
                NodePatch {
                    grade: Some(GradePatch {
                        color_balance: Some(ym_core::settings::ColorBalancePatch {
                            shadows: Some(ym_core::settings::BalanceSetting::manual_point([
                                f64::NAN,
                                0.0,
                            ])),
                            ..Default::default()
                        }),
                        ..GradePatch::default()
                    }),
                    ..NodePatch::default()
                },
            ),
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
        parse_patch(
            r#"{"nodes": {"1": {"grade": {"color_balance": {"shadows": [3, 4], "highlights": [0.6, 0.8]}}}}}"#,
        ),
    );
    let point = |w: &ym_core::settings::BalanceSetting| w.manual().expect("manual").point;
    assert!((point(&s.main_grade().color_balance.shadows)[0] - 0.6).abs() < 1e-12);
    assert!((point(&s.main_grade().color_balance.shadows)[1] - 0.8).abs() < 1e-12);
    // Exactly on the circle stays as it is; the untouched wheel stays zero.
    assert_eq!(point(&s.main_grade().color_balance.highlights), [0.6, 0.8]);
    assert_eq!(s.main_grade().color_balance.midtones, [0.0, 0.0]);
    assert!(fields(&report.clamped).contains(&SettingsField::ColorBalance));
}

#[test]
fn curves_are_checked_and_sorted() {
    let mut s = DevelopSettings::default();
    // Sorted by x, clamped into the unit square.
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"1": {"grade": {"curve": {"luma": [[1, 1], [0.5, 0.4], [0, 0], [0.7, 1.4]]}}}}}"#,
        ),
    );
    assert_eq!(
        s.main_grade().curve.luma,
        vec![[0.0, 0.0], [0.5, 0.4], [0.7, 1.0], [1.0, 1.0]]
    );
    assert!(fields(&report.changed).contains(&SettingsField::Curve));
    assert!(fields(&report.clamped).contains(&SettingsField::Curve));
    // The other channels were not named and are untouched.
    assert_eq!(s.main_grade().curve.red, vec![[0.0, 0.0], [1.0, 1.0]]);
    // Up to 16 points; 17 is refused; fewer than 2 and duplicates are refused.
    let points = |n: usize| -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| [i as f64 / (n - 1) as f64, i as f64 / (n - 1) as f64])
            .collect()
    };
    let with = |p: Vec<[f64; 2]>| {
        SettingsPatch::default().and(
            NodeId::MAIN,
            NodePatch {
                grade: Some(GradePatch {
                    curve: Some(ym_core::settings::CurvesPatch {
                        green: Some(p),
                        ..Default::default()
                    }),
                    ..GradePatch::default()
                }),
                ..NodePatch::default()
            },
        )
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
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"1": {"grade": {"contrast": 1}}}}"#),
    );
    assert!(s.input().crop.is_some() && s.input().white_pick.is_some());
    // null clears.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"crop": null, "white_pick": null}}}"#),
    );
    assert!(s.input().crop.is_none() && s.input().white_pick.is_none());
    assert!(fields(&report.changed).contains(&SettingsField::Crop));
    assert!(fields(&report.changed).contains(&SettingsField::WhitePick));
    // A value sets.
    apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"input": {"crop": {"rect": {"x": 0.9, "y": 0, "width": 0.5, "height": 2}, "angle_deg": 80, "aspect": "free"}}}}"#,
        ),
    );
    let crop = s.input().crop.unwrap();
    // The rectangle is moved and shrunk to fit the photo, the angle clamped.
    assert!((crop.rect.x - 0.5).abs() < 1e-12 && (crop.rect.width - 0.5).abs() < 1e-12);
    assert_eq!((crop.rect.y, crop.rect.height), (0.0, 1.0));
    assert_eq!(crop.angle_deg, 45.0);
    // A crop with no area is refused.
    let empty = parse_patch(
        r#"{"nodes": {"input": {"crop": {"rect": {"x": 0.1, "y": 0.1, "width": 0, "height": 0.5}, "angle_deg": 0, "aspect": "free"}}}}"#,
    );
    assert_eq!(
        s.apply_patch(&empty, None).unwrap_err(),
        SettingsError::EmptyCrop
    );
    // The patch serialises with the same distinction.
    let cleared =
        serde_json::to_value(parse_patch(r#"{"nodes": {"input": {"crop": null}}}"#)).unwrap();
    assert_eq!(cleared, json!({ "nodes": { "input": { "crop": null } } }));
    assert_eq!(serde_json::to_value(parse_patch("{}")).unwrap(), json!({}));
    // `null` and absent are told apart for the white pick too.
    assert_eq!(
        serde_json::to_value(parse_patch(r#"{"nodes": {"input": {"white_pick": null}}}"#)).unwrap(),
        json!({ "nodes": { "input": { "white_pick": null } } })
    );
}

#[test]
fn nested_objects_may_carry_only_the_parts_to_change() {
    let mut s = DevelopSettings::default();
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"decode": {"neural_demosaic": false}}}}"#),
    );
    assert!(!s.input().decode.neural_demosaic && s.input().decode.highlight_recovery);
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"rolloff": {"low_light": 0.9}}}}"#),
    );
    assert_eq!(
        s.input().rolloff,
        Rolloff {
            highlights: 0.6,
            low_light: 0.9
        }
    );
    // A complete object, as the frontend sends it, works the same.
    apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"input": {"decode": {"neural_demosaic": true, "highlight_recovery": false, "lens_corrections": true, "chromatic_aberration": true}}}}"#,
        ),
    );
    assert!(s.input().decode.neural_demosaic && !s.input().decode.highlight_recovery);
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"decode": {"lens_corrections": false}}}}"#),
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
        s.input().white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 4200.0,
            tint: 3.0
        })
    );
    // Now manual: only the tint moves.
    s.apply_patch(
        &parse_patch(r#"{"nodes": {"input": {"white_balance": {"kind": "manual", "tint": -8}}}}"#),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.input().white_balance,
        AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
            kelvin: 4200.0,
            tint: -8.0
        })
    );
    // Sharpening: the amount moves, the radius is the automatic one.
    s.apply_patch(
        &parse_patch(r#"{"nodes": {"input": {"sharpen": {"kind": "manual", "amount": 200}}}}"#),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.input().sharpen,
        AutoOr::Manual(SharpenValue {
            amount: 200.0,
            radius_px: 0.8
        })
    );
    s.apply_patch(
        &parse_patch(
            r#"{"nodes": {"input": {"noise_reduction": {"kind": "manual", "color": 10}}}}"#,
        ),
        Some(&seeds),
    )
    .unwrap();
    assert_eq!(
        s.input().noise_reduction,
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
    s.input_mut().light_source = LightSource::Shade;
    s.input_mut().white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 4000.0,
        tint: 2.0,
    });
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"white_pick": [2, -1]}}}"#),
    );
    assert_eq!(
        s.input().white_pick,
        Some([1.0, 0.0]),
        "clamped into the photo"
    );
    assert_eq!(s.input().light_source, LightSource::Auto);
    assert!(s.input().white_balance.is_auto());
    for f in [
        SettingsField::WhitePick,
        SettingsField::LightSource,
        SettingsField::WhiteBalance,
    ] {
        assert!(fields(&report.changed).contains(&f), "{f:?}");
    }
    // The patch may choose a preset together with the pick.
    let mut s = DevelopSettings::default();
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"white_pick": [0.5, 0.5], "light_source": "led"}}}"#),
    );
    assert_eq!(s.input().light_source, LightSource::Led);
    // Clearing the pick leaves the white balance alone.
    s.input_mut().white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 4000.0,
        tint: 2.0,
    });
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"white_pick": null}}}"#),
    );
    assert!(!s.input().white_balance.is_auto());
}

#[test]
fn orientation_patches_normalise_quarter_turns() {
    let mut s = DevelopSettings::default();
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"orientation": -90}}}"#),
    );
    assert_eq!(s.input().orientation, Orientation::Deg270);
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"orientation": 450}}}"#),
    );
    assert_eq!(s.input().orientation, Orientation::Deg90);
    assert!(
        serde_json::from_value::<SettingsPatch>(field_patch(json!({ "orientation": 45 }))).is_err()
    );
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"orientation": 180}}}"#),
    );
    assert_eq!(report.first_group(), Some(StageGroup::Geometry));
}

#[test]
fn a_patch_with_an_unknown_field_is_refused() {
    for bad in [
        json!({ "exposur": { "kind": "auto" } }),
        json!({ "decode": { "neural": false } }),
    ] {
        assert!(serde_json::from_value::<SettingsPatch>(field_patch(bad)).is_err());
    }
    // A top-level field of the flat form is an unknown field of the patch.
    assert!(serde_json::from_str::<SettingsPatch>(r#"{"exposure": {"kind":"auto"}}"#).is_err());
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
        let patch = parse_fields(text);
        s.apply_patch(&patch, None)
            .unwrap_or_else(|e| panic!("{text}: {e}"));
    }
    assert_eq!(s.output().look.as_str(), "builtin:signature");
    assert_eq!(s.input().crop.unwrap().aspect, CropAspect::SixteenByNine);
}

#[test]
fn reset_restores_every_field_to_its_default() {
    let look = LookId::builtin("signature");
    let d = DevelopSettings::with_look(look.clone());
    // The fields the settings had before they were nodes; `rich()` sets every one of them.
    for field in SettingsField::ALL.into_iter().take(24) {
        let node = home(field);
        let at = FieldRef { node, field };
        let mut s = rich();
        // Reset through the patch, as the UI's Reset does.
        s.apply_patch(&SettingsPatch::reset(node, field, &look), None)
            .unwrap();
        // Only that field is back at its default.
        let diff = s.diff(&rich());
        assert_eq!(
            diff,
            vec![at],
            "{field:?} must differ from the edited value, and nothing else"
        );
        let rest = s.diff(&d);
        assert!(
            !rest.contains(&at),
            "{field:?} must now equal its default: {rest:?}"
        );
    }
    // Re-run resets exposure, white balance (with preset and pick) and noise reduction only.
    let mut s = rich();
    s.reset_automatic(false);
    assert!(
        s.input().exposure.is_auto()
            && s.input().white_balance.is_auto()
            && s.input().noise_reduction.is_auto()
    );
    assert_eq!(s.input().light_source, LightSource::Auto);
    assert!(s.input().white_pick.is_none());
    assert_eq!(
        s.input().rolloff,
        rich().input().rolloff,
        "the highlight range is not part of Re-run"
    );
    assert!(
        !s.input().sharpen.is_auto(),
        "sharpening is not part of Re-run"
    );
    // Without Apple Auto acting on the photo, contrast and saturation are not part of Re-run.
    assert_eq!(s.main_grade().contrast, rich().main_grade().contrast);
    assert_eq!(s.main_grade().saturation, rich().main_grade().saturation);
    assert!(
        !rich().main_grade().contrast.is_auto(),
        "the test edits contrast"
    );
    // While Apple Auto acts on it they are on the Automatic card, and Re-run returns them to Auto
    // too (`docs/19` section 5).
    let mut s = rich();
    s.reset_automatic(true);
    assert_eq!(s.main_grade().contrast, AutoOr::Auto);
    assert_eq!(s.main_grade().saturation, AutoOr::Auto);
    // So do the other values Apple Auto sets (`docs/19` prompt version 2).
    assert_eq!(
        (
            s.main_grade().highlights,
            s.main_grade().shadows,
            s.main_grade().black
        ),
        (AutoOr::Auto, AutoOr::Auto, AutoOr::Auto)
    );
    assert!(
        s.main_grade().color_balance.shadows.is_auto()
            && s.main_grade().color_balance.highlights.is_auto()
    );
    assert!(
        s.input().exposure.is_auto()
            && s.input().white_balance.is_auto()
            && s.input().noise_reduction.is_auto()
    );
    assert!(
        !s.input().sharpen.is_auto(),
        "sharpening is not part of Re-run"
    );
    // Without Apple Auto acting, they stay as they were.
    let mut s = rich();
    s.reset_automatic(false);
    assert_eq!(
        (
            s.main_grade().highlights,
            s.main_grade().shadows,
            s.main_grade().black,
            s.main_grade().color_balance
        ),
        (
            rich().main_grade().highlights,
            rich().main_grade().shadows,
            rich().main_grade().black,
            rich().main_grade().color_balance
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
    let g = &v["nodes"]["items"][1]["grade"];
    for key in ["highlights", "shadows", "black"] {
        assert_eq!(g[key], json!({ "kind": "auto" }), "{key}");
    }
    assert_eq!(g["color_balance"]["shadows"], json!({ "kind": "auto" }));
    assert_eq!(g["color_balance"]["midtones"], json!([0.0, 0.0]));
    assert_eq!(g["color_balance"]["highlights"], json!({ "kind": "auto" }));
    // Old documents: a number is Manual, 0 is Auto; a point is Manual, [0, 0] is Auto; clamped.
    let (s, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"highlights": -30, "shadows": 0, "black": 250, "color_balance": {"shadows": [0.2, -0.1], "highlights": [0, 0], "midtones": [0.1, 0]}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(
        s.main_grade().highlights,
        PercentSetting::manual_value(-30.0)
    );
    assert_eq!(s.main_grade().shadows, AutoOr::Auto);
    assert_eq!(s.main_grade().black, PercentSetting::manual_value(100.0));
    assert!(fields(&clamped).contains(&SettingsField::Black));
    assert_eq!(
        s.main_grade().color_balance.shadows,
        BalanceSetting::manual_point([0.2, -0.1])
    );
    assert_eq!(s.main_grade().color_balance.highlights, AutoOr::Auto);
    assert_eq!(s.main_grade().color_balance.midtones, [0.1, 0.0]);
    // The new forms.
    let (s, _) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"highlights": {"kind": "manual", "value": 12}, "color_balance": {"highlights": {"kind": "manual", "point": [3, 4]}}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(
        s.main_grade().highlights,
        PercentSetting::manual_value(12.0)
    );
    let p = s
        .main_grade()
        .color_balance
        .highlights
        .manual()
        .unwrap()
        .point;
    assert!((p[0] - 0.6).abs() < 1e-12 && (p[1] - 0.8).abs() < 1e-12);
    assert_eq!(
        s.main_grade().color_balance.shadows,
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
        assert!(read_fields(bad, &neutral).is_err(), "{bad}");
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
        s.apply_patch(&parse_fields(text), None).unwrap();
        assert_eq!(
            (
                s.main_grade().highlights,
                s.main_grade().color_balance.shadows
            ),
            (highlights, wheel),
            "{text}"
        );
    }
    // Copy and paste carries Auto as Auto, and a manual value as it is.
    let mut from = DevelopSettings::default();
    from.main_grade_mut().shadows = PercentSetting::manual_value(-15.0);
    let copied = SettingsPatch::of_fields(
        &from,
        NodeId::MAIN,
        &[
            SettingsField::Highlights,
            SettingsField::Shadows,
            SettingsField::ColorBalance,
        ],
    );
    let v = serde_json::to_value(&copied).unwrap();
    let grade = &v["nodes"]["1"]["grade"];
    assert_eq!(grade["highlights"], json!({ "kind": "auto" }));
    assert_eq!(
        grade["shadows"],
        json!({ "kind": "manual", "value": -15.0 })
    );
    assert_eq!(grade["color_balance"]["shadows"], json!({ "kind": "auto" }));
    let mut onto = DevelopSettings::default();
    onto.main_grade_mut().highlights = PercentSetting::manual_value(30.0);
    onto.apply_patch(&copied, None).unwrap();
    assert_eq!(
        (onto.main_grade().highlights, onto.main_grade().shadows),
        (AutoOr::Auto, from.main_grade().shadows)
    );
}

#[test]
fn output_sharpening_is_a_plain_number_in_a_patch_with_its_own_stage_group() {
    let look = LookId::builtin("neutral");
    let mut s = DevelopSettings::default();
    assert_eq!(s.output().output_sharpening, 100.0);
    // Moving the slider sends a number; only that field changes, and only the last stage group
    // is affected, so nothing before it re-runs.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"output": {"output_sharpening": 35}}}"#),
    );
    assert_eq!(s.output().output_sharpening, 35.0);
    assert_eq!(
        fields(&report.changed),
        vec![SettingsField::OutputSharpening]
    );
    assert_eq!(report.first_group(), Some(StageGroup::Output));
    // The same patch again changes nothing.
    assert!(
        apply(
            &mut s,
            parse_patch(r#"{"nodes": {"output": {"output_sharpening": 35}}}"#)
        )
        .changed
        .is_empty()
    );
    // 0 is a value (no stage), not a reset.
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"output": {"output_sharpening": 0}}}"#),
    );
    assert_eq!(s.output().output_sharpening, 0.0);
    // Out of range is clamped and reported, in both directions.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"output": {"output_sharpening": 900}}}"#),
    );
    assert_eq!(s.output().output_sharpening, 200.0);
    assert!(fields(&report.clamped).contains(&SettingsField::OutputSharpening));
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"output": {"output_sharpening": -20}}}"#),
    );
    assert_eq!(s.output().output_sharpening, 0.0);
    // NaN is refused and changes nothing.
    let before = s.clone();
    let nan = SettingsPatch::default().and(
        NodeId::Output,
        NodePatch {
            output_sharpening: Some(f64::NAN),
            ..NodePatch::default()
        },
    );
    assert_eq!(
        s.apply_patch(&nan, None).unwrap_err(),
        SettingsError::NotFinite("nodes.items[2].output_sharpening".to_owned())
    );
    assert_eq!(s, before);
    // Reset returns to 100, not to Auto.
    let report = apply(
        &mut s,
        SettingsPatch::reset(
            home(SettingsField::OutputSharpening),
            SettingsField::OutputSharpening,
            &look,
        ),
    );
    assert_eq!(s.output().output_sharpening, 100.0);
    assert_eq!(
        fields(&report.changed),
        vec![SettingsField::OutputSharpening]
    );
    // It changes with a Grade field: the first group is the earliest of them.
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"1": {"grade": {"contrast": 10}}, "output": {"output_sharpening": 50}}}"#,
        ),
    );
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    // The patch that takes the field of another photo carries it (copy and paste of one field).
    let mut other = DevelopSettings::default();
    apply(
        &mut other,
        SettingsPatch::of_field(
            &s,
            home(SettingsField::OutputSharpening),
            SettingsField::OutputSharpening,
        ),
    );
    assert_eq!(other.output().output_sharpening, 50.0);
    assert_eq!(other.main_grade().contrast, AutoOr::Auto);
}

#[test]
fn an_older_document_without_output_sharpening_reads_as_100() {
    // A settings file, a golden manifest or the ym:Settings of an export made before the field.
    let mut doc = serde_json::to_value(rich()).unwrap();
    for item in doc["nodes"]["items"].as_array_mut().unwrap() {
        if item["id"] == "output" {
            item.as_object_mut().unwrap().remove("output_sharpening");
        }
    }
    let (s, clamped) =
        DevelopSettings::from_value(doc, &LookId::builtin("neutral")).expect("reads");
    assert_eq!(s.output().output_sharpening, 100.0);
    assert!(!fields(&clamped).contains(&SettingsField::OutputSharpening));
    // And the full document writes it back.
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"output_sharpening\":100"), "{text}");
}

#[test]
fn settings_read_from_a_file_are_clamped_and_checked() {
    let mut s = rich();
    s.main_grade_mut().contrast = ym_core::settings::PercentSetting::manual_value(400.0);
    s.input_mut().rolloff.highlights = -3.0;
    s.main_grade_mut().curve.red = vec![[0.9, 0.1], [0.1, 0.9]];
    let (clean, clamped) = s.validated().unwrap();
    assert_eq!(
        clean.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(100.0)
    );
    assert_eq!(clean.input().rolloff.highlights, 0.0);
    assert_eq!(clean.main_grade().curve.red, vec![[0.1, 0.9], [0.9, 0.1]]);
    for f in [
        SettingsField::Contrast,
        SettingsField::Rolloff,
        SettingsField::Curve,
    ] {
        assert!(fields(&clamped).contains(&f), "{f:?}");
    }
    let mut bad = rich();
    bad.main_grade_mut().curve.luma = vec![[0.0, 0.0]];
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
    assert!(r.auto.input.exposure && r.auto.input.light_source && r.auto.input.white_balance);
    assert!(r.auto.input.sharpen && r.auto.input.noise_reduction);
    assert_eq!(r.values.input.exposure_ev, summary.input.exposure_ev);
    assert_eq!(
        (r.values.input.kelvin, r.values.input.tint),
        (summary.input.kelvin, summary.input.tint)
    );
    assert!(
        (r.values.input.kelvin - 4300.0).abs() <= 50.0,
        "{}",
        r.values.input.kelvin
    );
    assert_eq!(r.values.input.sharpen, est.sharpen);
    assert_eq!(r.values.input.noise_reduction, est.noise_reduction);
    assert_eq!(r.estimates, summary);
    // ISO 400: L_auto = 10 (1 + 2) = 30, as the Automatic card's "Luminance 30".
    assert!((r.values.input.noise_reduction.luminance - 30.0).abs() < 1e-4);
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
    assert!(!r.auto.input.exposure && !r.auto.input.noise_reduction && !r.auto.input.white_balance);
    assert!(r.auto.input.sharpen && r.auto.input.light_source);
    assert_eq!(r.values.input.exposure_ev, -0.5);
    assert_eq!((r.values.input.kelvin, r.values.input.tint), (3300.0, 4.0));
    assert_eq!(r.values.input.noise_reduction.luminance, 70.0);
}

#[test]
fn a_light_preset_shows_its_kelvin_while_white_balance_is_auto() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(512, 384, light(PROFILE, 4300.0), Layout::Patchwork, 0.4);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let with = |l: LightSource| {
        let mut s = DevelopSettings::default();
        s.input_mut().light_source = l;
        resolve_with(&s, &est, &profile, None)
    };
    for (preset, kelvin) in [
        (LightSource::Day, 5500.0),
        (LightSource::Shade, 7500.0),
        (LightSource::Tungsten, 3200.0),
    ] {
        let r = with(preset);
        assert_eq!(r.values.input.kelvin, kelvin, "{preset:?}");
        assert_eq!(r.values.input.tint, 0.0, "{preset:?}");
        assert!(!r.auto.input.light_source && r.auto.input.white_balance);
    }
    // LED and fluorescent keep the estimated chromaticity.
    let auto = with(LightSource::Auto);
    for preset in [LightSource::Led, LightSource::Fluorescent] {
        let r = with(preset);
        assert!(
            (r.values.input.kelvin - auto.values.input.kelvin).abs() < 1e-9,
            "{preset:?}"
        );
    }
    // A manual white balance wins over the readout of any preset.
    let mut s = DevelopSettings::default();
    s.input_mut().light_source = LightSource::Tungsten;
    s.apply_patch(&SettingsPatch::white_balance(6000.0, -3.0), None)
        .unwrap();
    let r = resolve_with(&s, &est, &profile, None);
    assert_eq!((r.values.input.kelvin, r.values.input.tint), (6000.0, -3.0));
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
        (before.values.input.kelvin - 6500.0).abs() <= 50.0,
        "{}",
        before.values.input.kelvin
    );
    let point = [0.4167, 0.125];
    let sample = pick_white_sample(&raw, point, est.initial_gains().unwrap()).unwrap();
    let mut s = DevelopSettings::default();
    s.apply_patch(
        &parse_patch(&format!(
            r#"{{"nodes": {{"input": {{"white_pick": [{}, {}]}}}}}}"#,
            point[0], point[1]
        )),
        None,
    )
    .unwrap();
    let after = resolve_with(&s, &est, &profile, Some(sample));
    assert!(
        after.auto.input.white_balance,
        "the pick leaves white_balance on Auto"
    );
    assert!(
        (after.values.input.kelvin - 3000.0).abs() <= 100.0,
        "{}",
        after.values.input.kelvin
    );
    // Without the sample the pick has no effect on the readout.
    let ignored = resolve_with(&s, &est, &profile, None);
    assert_eq!(ignored.values.input.kelvin, before.values.input.kelvin);
}

#[test]
fn a_pick_on_a_blown_out_spot_is_refused_with_the_message_of_the_spec() {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(256, 192, light(PROFILE, 5200.0), Layout::Uniform, 0.3);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let mut s = DevelopSettings::default();
    s.input_mut().white_pick = Some([0.5, 0.5]);
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
        pinned.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(r.values.input.exposure_ev)
    );
    assert_eq!(
        pinned.input().sharpen,
        AutoOr::Manual(r.values.input.sharpen)
    );
    assert_eq!(
        pinned.input().noise_reduction,
        AutoOr::Manual(r.values.input.noise_reduction)
    );
    assert!(
        pinned.input().white_balance.is_auto(),
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
        .apply_patch(
            &SettingsPatch::white_balance(r.values.input.kelvin, 25.0),
            None,
        )
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
            range: 0.02,
            shadows: 1.0,
            highlights: 1.0,
            pivot_ev: 0.0,
            exposure_ev: 0.0,
            saturation: 0.0,
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
            "saturation",
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
        "range": 0.06, "shadows": 0.5, "highlights": 0.75, "pivot_ev": -1, "exposure_ev": 0.5,
        "saturation": -0.35 });
    let pin: ColorWarpPin = serde_json::from_value(full.clone()).unwrap();
    assert_eq!(norm(serde_json::to_value(pin).unwrap()), norm(full));
    // A settings document round-trips with and without the optional fields.
    let s = rich();
    let back: DevelopSettings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.main_grade().color_warp, s.main_grade().color_warp);
    let (short, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "target": [0.1, 0.05]}, {"source": [0.0, -0.1], "locked": true}]}}}]}}"#,
        &LookId::builtin("neutral"),
    )
    .unwrap();
    assert!(fields(&clamped).is_empty(), "{clamped:?}");
    assert_eq!(
        short.main_grade().color_warp.pins,
        vec![
            pin_at([0.1, 0.0], [0.1, 0.05]),
            ColorWarpPin::from(WarpPin::locked_at([0.0, -0.1]))
        ]
    );
    // A document without the field, or with an empty object, has no pins.
    let (none, _) = DevelopSettings::from_json("{}", &LookId::builtin("neutral")).unwrap();
    assert!(none.main_grade().color_warp.pins.is_empty());
    assert_eq!(
        read_fields(r#"{ "color_warp": {} }"#, &LookId::builtin("neutral"))
            .unwrap()
            .0
            .main_grade()
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
            read_fields(bad, &LookId::builtin("neutral")).is_err(),
            "{bad} should be refused"
        );
    }
}

#[test]
fn the_warp_keeps_16_pins_and_clamps_each_like_the_stage_does() {
    let neutral = LookId::builtin("neutral");
    // 17 pins: the 17th is cut, and the field is reported.
    let mut s = DevelopSettings::default();
    s.main_grade_mut().color_warp.pins = (0..17)
        .map(|i| pin_at([0.01 * f64::from(i), 0.1], [0.01 * f64::from(i), 0.12]))
        .collect();
    let (v, clamped) = s.clone().validated().unwrap();
    assert_eq!(v.main_grade().color_warp.pins.len(), 16);
    assert_eq!(
        v.main_grade().color_warp.pins[..],
        s.main_grade().color_warp.pins[..16]
    );
    assert_eq!(fields(&clamped), vec![SettingsField::ColorWarp]);
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
        saturation: -4.0,
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
        saturation: 0.5,
    };
    let (v, clamped) = DevelopSettings::from_value(
        field_doc(json!({ "color_warp": { "pins": [wild, locked] } })),
        &neutral,
    )
    .unwrap();
    assert_eq!(fields(&clamped), vec![SettingsField::ColorWarp]);
    let [a, b] = [
        v.main_grade().color_warp.pins[0],
        v.main_grade().color_warp.pins[1],
    ];
    assert!((a.source[0] - 0.3).abs() < 1e-12 && (a.source[1] + 0.4).abs() < 1e-12);
    assert!(a.source[0].hypot(a.source[1]) <= 0.5);
    assert!((a.target[1] - 0.5).abs() < 1e-12 && a.target[0] == 0.0);
    assert_eq!(
        (
            a.range,
            a.shadows,
            a.highlights,
            a.pivot_ev,
            a.exposure_ev,
            a.saturation
        ),
        (0.25, 0.0, 1.0, -6.0, 2.0, -1.0)
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
            saturation: 0.0,
        }
    );
    // The settings are what the stage renders: clamping them again is the stage's own clamp.
    for p in &v.main_grade().color_warp.pins {
        let c = WarpPin::from(*p).clamped();
        assert_eq!(c.locked, p.locked);
        assert_eq!(
            (
                c.range,
                c.shadows,
                c.highlights,
                c.pivot_ev,
                c.exposure_ev,
                c.saturation
            ),
            (
                p.range,
                p.shadows,
                p.highlights,
                p.pivot_ev,
                p.exposure_ev,
                p.saturation
            )
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
    s.main_grade_mut().color_warp.pins = (0..16)
        .map(|_| pin_at([next(), next()], [next(), next()]))
        .collect();
    for _ in 0..200 {
        let (once, _) = s.clone().validated().unwrap();
        let (twice, again) = once.clone().validated().unwrap();
        assert_eq!(twice, once);
        assert!(again.is_empty(), "{again:?}");
        for p in &once.main_grade().color_warp.pins {
            assert!(p.source[0].hypot(p.source[1]) <= 0.5);
            assert!(p.target[0].hypot(p.target[1]) <= 0.5);
        }
        for p in &mut s.main_grade_mut().color_warp.pins {
            *p = pin_at([next(), next()], [next(), next()]);
        }
    }
    // A number that is not finite is an error, as everywhere else; nothing is applied.
    let breaks: [fn(&mut ColorWarpPin); 5] = [
        |p: &mut ColorWarpPin| p.source[1] = f64::NAN,
        |p: &mut ColorWarpPin| p.target[0] = f64::INFINITY,
        |p: &mut ColorWarpPin| p.range = f64::NAN,
        |p: &mut ColorWarpPin| p.exposure_ev = f64::NEG_INFINITY,
        |p: &mut ColorWarpPin| p.saturation = f64::NAN,
    ];
    for broken in breaks {
        let mut s = DevelopSettings::default();
        let mut p = pin_at([0.1, 0.0], [0.1, 0.1]);
        broken(&mut p);
        s.main_grade_mut().color_warp.pins.push(p);
        assert!(
            matches!(s.clone().validated(), Err(SettingsError::NotFinite(n)) if n.starts_with("nodes.items[1].grade.color_warp")),
            "{p:?}"
        );
        let mut t = DevelopSettings::default();
        let patch = SettingsPatch::default().and(
            NodeId::MAIN,
            NodePatch {
                grade: Some(GradePatch {
                    color_warp: Some(s.main_grade().color_warp.clone()),
                    ..GradePatch::default()
                }),
                ..NodePatch::default()
            },
        );
        assert!(t.apply_patch(&patch, None).is_err());
        assert_eq!(t, DevelopSettings::default());
    }
}

#[test]
fn a_pins_saturation_reads_as_zero_when_absent_and_is_clamped_and_kept() {
    let neutral = LookId::builtin("neutral");
    // A document from before the saturation existed: every older field, no `saturation`. It reads as
    // 0, is not reported as clamped, and renders as it did (the pin does not act on saturation).
    let (old, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "target": [0.1, 0.05], "locked": false, "range": 0.04, "shadows": 0.5, "highlights": 1, "pivot_ev": 0, "exposure_ev": 0.5}]}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert!(fields(&clamped).is_empty(), "{clamped:?}");
    assert_eq!(old.main_grade().color_warp.pins[0].saturation, 0.0);
    // `null` is not a number.
    assert!(
        read_fields(
            r#"{ "color_warp": { "pins": [{ "source": [0.1, 0.0], "saturation": null }] } }"#,
            &neutral
        )
        .is_err()
    );
    // A value in range is kept as it is and written back; one beyond is clamped and reported.
    let (kept, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "saturation": -0.5}]}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert!(fields(&clamped).is_empty(), "{clamped:?}");
    assert_eq!(kept.main_grade().color_warp.pins[0].saturation, -0.5);
    assert_eq!(
        serde_json::to_value(&kept).unwrap()["nodes"]["items"][1]["grade"]["color_warp"]["pins"][0]
            ["saturation"],
        json!(-0.5)
    );
    let (wild, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "saturation": 3}]}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(fields(&clamped), vec![SettingsField::ColorWarp]);
    assert_eq!(wild.main_grade().color_warp.pins[0].saturation, 1.0);
    // It reaches the stage's own parameters, and a pin that only has saturation acts there.
    let params = kept.main_grade().color_warp.to_color();
    assert_eq!(params.pins[0].saturation, -0.5);
    assert!(!params.is_identity());
    // A locked pin has none.
    let (locked, _) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": 1, "grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "locked": true, "saturation": 0.8}]}}}]}}"#,
        &neutral,
    )
    .unwrap();
    assert_eq!(locked.main_grade().color_warp.pins[0].saturation, 0.0);
    // A copy of the colour warp's settings carries it (the patch holds the whole list).
    let mut s = DevelopSettings::default();
    s.main_grade_mut().color_warp = kept.main_grade().color_warp.clone();
    let patch = SettingsPatch::default().and(
        NodeId::MAIN,
        NodePatch {
            grade: Some(GradePatch {
                color_warp: Some(s.main_grade().color_warp.clone()),
                ..GradePatch::default()
            }),
            ..NodePatch::default()
        },
    );
    let json = serde_json::to_string(&patch).unwrap();
    let back: SettingsPatch = serde_json::from_str(&json).unwrap();
    assert_eq!(
        back.main_patch()
            .and_then(|p| p.grade.as_ref())
            .and_then(|g| g.color_warp.clone()),
        Some(kept.main_grade().color_warp.clone())
    );
}

#[test]
fn a_warp_patch_replaces_the_whole_list_and_reset_removes_the_pins() {
    let neutral = LookId::builtin("neutral");
    assert_eq!(SettingsField::ColorWarp.name(), "color_warp");
    assert_eq!(SettingsField::ColorWarp.group(), Some(StageGroup::Grade));
    assert!(SettingsField::ALL.contains(&SettingsField::ColorWarp));
    let mut s = DevelopSettings::default();
    let two = parse_patch(
        r#"{"nodes": {"1": {"grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "target": [0.1, 0.05]}, {"source": [0.0, 0.1], "locked": true}]}}}}}"#,
    );
    let report = apply(&mut s, two);
    assert_eq!(fields(&report.changed), vec![SettingsField::ColorWarp]);
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    assert_eq!(s.main_grade().color_warp.pins.len(), 2);
    // A patch is the whole list: one pin replaces the two.
    let one = parse_patch(
        r#"{"nodes": {"1": {"grade": {"color_warp": {"pins": [{"source": [-0.1, 0.0], "target": [-0.12, 0.0]}]}}}}}"#,
    );
    apply(&mut s, one);
    assert_eq!(
        s.main_grade().color_warp.pins,
        vec![pin_at([-0.1, 0.0], [-0.12, 0.0])]
    );
    // A patch that leaves the warp out keeps it, and reports nothing about it.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"1": {"grade": {"contrast": 10}}}}"#),
    );
    assert_eq!(fields(&report.changed), vec![SettingsField::Contrast]);
    assert_eq!(s.main_grade().color_warp.pins.len(), 1);
    // A value out of range in the patch is clamped and reported.
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"1": {"grade": {"color_warp": {"pins": [{"source": [0.1, 0.0], "range": 9}]}}}}}"#,
        ),
    );
    assert_eq!(fields(&report.clamped), vec![SettingsField::ColorWarp]);
    assert_eq!(s.main_grade().color_warp.pins[0].range, 0.25);
    // The same list again changes nothing.
    let same =
        SettingsPatch::of_field(&s, home(SettingsField::ColorWarp), SettingsField::ColorWarp);
    assert!(apply(&mut s, same.clone()).is_empty());
    assert_eq!(
        serde_json::to_value(&same).unwrap(),
        json!({ "nodes": { "1": { "grade": {
            "color_warp": serde_json::to_value(&s.main_grade().color_warp).unwrap() } } } })
    );
    // The card's Reset sends no pins; the patch of no pins and the Reset patch agree.
    let mut by_patch = s.clone();
    apply(
        &mut by_patch,
        parse_patch(r#"{"nodes": {"1": {"grade": {"color_warp": {"pins": []}}}}}"#),
    );
    let mut by_reset_patch = s.clone();
    apply(
        &mut by_reset_patch,
        SettingsPatch::reset(
            home(SettingsField::ColorWarp),
            SettingsField::ColorWarp,
            &neutral,
        ),
    );
    for t in [&by_patch, &by_reset_patch] {
        assert!(t.main_grade().color_warp.pins.is_empty());
        assert_eq!(
            t.diff(&s),
            vec![FieldRef::new(NodeId::MAIN, SettingsField::ColorWarp)]
        );
    }
}

#[test]
fn copy_settings_takes_the_warp_and_resolved_values_carry_it() {
    // Copy settings takes every field but the photo's own geometry and Pick white point.
    assert!(
        !COPY_LEAVES_OUT
            .iter()
            .any(|r| r.field == SettingsField::ColorWarp)
    );
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
    assert_eq!(
        resolved.values.main.grade.color_warp,
        s.main_grade().color_warp
    );
    assert_eq!(
        resolved.pinned_settings().main_grade().color_warp,
        s.main_grade().color_warp
    );
    let patch = SettingsPatch::copied_from(&resolved);
    assert_eq!(
        patch
            .main_patch()
            .and_then(|p| p.grade.as_ref())
            .and_then(|g| g.color_warp.as_ref()),
        Some(&s.main_grade().color_warp)
    );
    // Pasted onto a photo with pins of its own, the copied list replaces them.
    let mut other = DevelopSettings::default();
    other.main_grade_mut().color_warp.pins = vec![pin_at([0.0, -0.2], [0.05, -0.2])];
    other.apply_patch(&patch, None).unwrap();
    assert_eq!(other.main_grade().color_warp, s.main_grade().color_warp);
}

// ---------------------------------------------------------------------------------------------
// Lens (docs/21 section 2)
// ---------------------------------------------------------------------------------------------

fn lens(kind: LensKind, flares: f64, edges: f64, anamorphic: f64) -> LensSettings {
    LensSettings {
        kind,
        flares,
        edges,
        anamorphic,
    }
}

#[test]
fn a_document_reads_with_and_without_a_lens() {
    let neutral = LookId::builtin("neutral");
    // Without the field: the default lens, nothing clamped (older settings, older XMP).
    let (s, clamped) = DevelopSettings::from_json("{}", &neutral).expect("reads");
    assert_eq!(s.input().lens, lens(LensKind::Default, 0.0, 0.0, 0.0));
    assert!(s.input().lens.is_identity() && fields(&clamped).is_empty());
    // With every field, with some of them (the others take the defaults), and as written.
    let (s, _) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": "input", "lens": {"kind": "petzval", "flares": 40, "edges": 55, "anamorphic": 20}}]}}"#,
        &neutral,
    )
    .expect("reads");
    assert_eq!(s.input().lens, lens(LensKind::Petzval, 40.0, 55.0, 20.0));
    let (s, _) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": "input", "lens": {"flares": 40}}]}}"#,
        &neutral,
    )
    .expect("reads");
    assert_eq!(s.input().lens, lens(LensKind::Default, 40.0, 0.0, 0.0));
    assert!(!s.input().lens.is_identity());
    let written = serde_json::to_value(s.input().lens).unwrap();
    assert_eq!(
        written,
        json!({ "kind": "default", "flares": 40.0, "edges": 0.0, "anamorphic": 0.0 }),
        "writing writes every field"
    );
    // All seven ids are read and written as the same words.
    for kind in LensKind::ALL {
        let doc = field_doc(json!({ "lens": { "kind": kind.as_str() } }));
        let (s, _) = DevelopSettings::from_value(doc, &neutral).expect("reads");
        assert_eq!(s.input().lens.kind, kind);
        assert_eq!(
            serde_json::to_value(s.input().lens).unwrap()["kind"],
            json!(kind.as_str())
        );
    }
    assert_eq!(
        LensKind::ALL.map(LensKind::as_str),
        [
            "default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"
        ]
    );
    // An unknown kind or field is a parse error, not a silent default.
    for bad in [
        r#"{ "lens": { "kind": "leica" } }"#,
        r#"{ "lens": { "kind": "Vintage" } }"#,
        r#"{ "lens": { "grain": 3 } }"#,
        r#"{ "lens": { "flares": "lots" } }"#,
    ] {
        assert!(
            matches!(read_fields(bad, &neutral), Err(SettingsError::Parse(_))),
            "{bad}"
        );
    }
}

#[test]
fn the_amounts_clamp_and_a_value_that_is_not_a_number_is_refused() {
    let neutral = LookId::builtin("neutral");
    assert_eq!(LENS_AMOUNT_RANGE, (0.0, 100.0));
    // From a document.
    let (s, clamped) = DevelopSettings::from_json(
        r#"{"nodes": {"items": [{"id": "input", "lens": {"kind": "sonnar", "flares": 140, "edges": -5, "anamorphic": 100}}]}}"#,
        &neutral,
    )
    .expect("reads");
    assert_eq!(s.input().lens, lens(LensKind::Sonnar, 100.0, 0.0, 100.0));
    assert_eq!(fields(&clamped), vec![SettingsField::Lens]);
    // From a patch: reported only in the fields the patch touched.
    let mut s = DevelopSettings::default();
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"lens": {"edges": 250}}}}"#),
    );
    assert_eq!(s.input().lens.edges, 100.0);
    assert_eq!(fields(&report.changed), vec![SettingsField::Lens]);
    assert_eq!(fields(&report.clamped), vec![SettingsField::Lens]);
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"1": {"grade": {"contrast": 10}}}}"#),
    );
    assert!(fields(&report.clamped).is_empty(), "{report:?}");
    // Negative zero is zero.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"lens": {"flares": -0.0}}}}"#),
    );
    assert!(fields(&report.changed).is_empty(), "{report:?}");
    assert!(s.input().lens.flares.is_sign_positive());
    // Not finite: an error that names the field, and nothing is applied.
    for (field, patch) in [
        (
            "lens.flares",
            LensPatch {
                flares: Some(f64::NAN),
                ..LensPatch::default()
            },
        ),
        (
            "lens.edges",
            LensPatch {
                edges: Some(f64::INFINITY),
                ..LensPatch::default()
            },
        ),
        (
            "lens.anamorphic",
            LensPatch {
                anamorphic: Some(f64::NEG_INFINITY),
                ..LensPatch::default()
            },
        ),
    ] {
        let mut t = s.clone();
        let err = t
            .apply_patch(
                &SettingsPatch::default()
                    .and(
                        NodeId::Input,
                        NodePatch {
                            lens: Some(patch),
                            ..NodePatch::default()
                        },
                    )
                    .and(
                        NodeId::MAIN,
                        NodePatch {
                            grade: Some(GradePatch {
                                contrast: Some(ym_core::settings::PercentSetting::manual_value(
                                    30.0,
                                )),
                                ..GradePatch::default()
                            }),
                            ..NodePatch::default()
                        },
                    ),
                None,
            )
            .unwrap_err();
        assert_eq!(
            err,
            SettingsError::NotFinite(format!("nodes.items[0].{field}"))
        );
        assert_eq!(t, s, "{field}: nothing is applied");
    }
    let doc = serde_json::from_value::<DevelopSettings>(
        json!({ "nodes": { "items": [ { "id": "input", "lens": { "flares": 1 } } ] } }),
    )
    .map(|mut d| {
        d.input_mut().lens.edges = f64::NAN;
        d
    })
    .unwrap();
    assert_eq!(
        doc.validated().unwrap_err(),
        SettingsError::NotFinite("nodes.items[0].lens.edges".to_owned())
    );
}

#[test]
fn a_lens_patch_may_carry_only_the_parts_it_changes() {
    let mut s = DevelopSettings::default();
    // Choosing a lens does not move the amounts by itself.
    let report = apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"lens": {"kind": "triplet"}}}}"#),
    );
    assert_eq!(s.input().lens, lens(LensKind::Triplet, 0.0, 0.0, 0.0));
    assert_eq!(fields(&report.changed), vec![SettingsField::Lens]);
    assert_eq!(
        report.first_group(),
        Some(StageGroup::Lens),
        "a lens change invalidates the lens group and the ones after it"
    );
    // One amount at a time, the rest as they were.
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"lens": {"flares": 40}}}}"#),
    );
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"lens": {"anamorphic": 25.5}}}}"#),
    );
    assert_eq!(s.input().lens, lens(LensKind::Triplet, 40.0, 0.0, 25.5));
    // What the card sends on choosing a lens: the kind and its starting amounts, in one patch.
    let report = apply(
        &mut s,
        parse_patch(
            r#"{"nodes": {"input": {"lens": {"kind": "vintage", "flares": 35, "edges": 50}}}}"#,
        ),
    );
    assert_eq!(s.input().lens, lens(LensKind::Vintage, 35.0, 50.0, 25.5));
    assert_eq!(fields(&report.changed), vec![SettingsField::Lens]);
    // The whole object, as the frontend sends it, works the same; the same value again is no change.
    let same = parse_patch(
        r#"{"nodes": {"input": {"lens": {"kind": "vintage", "flares": 35, "edges": 50, "anamorphic": 25.5}}}}"#,
    );
    assert!(apply(&mut s, same).changed.is_empty());
    // An empty object changes nothing; a patch that leaves the lens out keeps it.
    assert!(
        apply(&mut s, parse_patch(r#"{"nodes": {"input": {"lens": {}}}}"#))
            .changed
            .is_empty()
    );
    apply(
        &mut s,
        parse_patch(r#"{"nodes": {"input": {"exposure": {"kind": "auto"}}}}"#),
    );
    assert_eq!(s.input().lens.kind, LensKind::Vintage);
    // The patch serialises with only the parts it carries, and an unknown part is refused.
    let p = parse_patch(r#"{"nodes": {"input": {"lens": {"edges": 10}}}}"#);
    assert_eq!(
        serde_json::to_value(&p).unwrap(),
        json!({ "nodes": { "input": { "lens": { "edges": 10.0 } } } })
    );
    for bad in [
        r#"{ "lens": { "kind": "leica" } }"#,
        r#"{ "lens": { "grain": 1 } }"#,
        r#"{ "lens": { "flares": "lots" } }"#,
    ] {
        assert!(
            serde_json::from_value::<SettingsPatch>(field_patch(
                serde_json::from_str(bad).unwrap()
            ))
            .is_err(),
            "{bad}"
        );
    }
}

#[test]
fn reset_gives_the_default_lens_and_re_run_leaves_it() {
    let look = LookId::builtin("neutral");
    assert_eq!(SettingsField::Lens.name(), "lens");
    assert_eq!(SettingsField::Lens.group(), Some(StageGroup::Lens));
    assert!(SettingsField::ALL.contains(&SettingsField::Lens));
    let mut s = rich();
    assert_ne!(s.input().lens, LensSettings::default());
    // The Reset patch gives the default lens and changes nothing else.
    let before = s.clone();
    apply(
        &mut s,
        SettingsPatch::reset(NodeId::Input, SettingsField::Lens, &look),
    );
    assert_eq!(s.input().lens, lens(LensKind::Default, 0.0, 0.0, 0.0));
    assert_eq!(
        s.diff(&before),
        vec![FieldRef::new(NodeId::Input, SettingsField::Lens)]
    );
    // Re-run (the Automatic card) does not touch the lens.
    let mut s = rich();
    s.reset_automatic(true);
    assert_eq!(s.input().lens, rich().input().lens);
    // A fresh photo (`with_look`) has no lens.
    assert_eq!(
        DevelopSettings::with_look(look.clone()).input().lens,
        LensSettings::default()
    );
}

#[test]
fn copy_settings_takes_the_lens_and_resolved_values_and_exports_carry_it() {
    assert!(
        !COPY_LEAVES_OUT
            .iter()
            .any(|r| r.field == SettingsField::Lens)
    );
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
    assert_eq!(resolved.values.input.lens, s.input().lens);
    // `ym:Settings` of an export is the pinned settings: it gains the lens object, as stored.
    let pinned = resolved.pinned_settings();
    assert_eq!(pinned.input().lens, s.input().lens);
    let doc = serde_json::to_value(&pinned).unwrap();
    assert_eq!(
        doc["nodes"]["items"][0]["lens"],
        json!({ "kind": "vintage", "flares": 60.0, "edges": 45.0, "anamorphic": 30.0 })
    );
    // Copy takes it, with every part, so Paste onto another photo gives that photo this lens.
    let patch = SettingsPatch::copied_from(&resolved);
    assert_eq!(
        patch.input_patch().and_then(|p| p.lens),
        Some(LensPatch {
            kind: Some(LensKind::Vintage),
            flares: Some(60.0),
            edges: Some(45.0),
            anamorphic: Some(30.0),
        })
    );
    let mut other = DevelopSettings::default();
    other.input_mut().lens = lens(LensKind::Detuned, 5.0, 6.0, 7.0);
    let report = other.apply_patch(&patch, None).unwrap();
    assert_eq!(other.input().lens, s.input().lens);
    assert!(fields(&report.changed).contains(&SettingsField::Lens));
    // A photo with the default lens copies the default, which clears the lens on the target.
    let plain = ResolvedSettings::resolve(
        PhotoId(4),
        &DevelopSettings::default(),
        &est,
        &profile,
        None,
    )
    .unwrap();
    other
        .apply_patch(&SettingsPatch::copied_from(&plain), None)
        .unwrap();
    assert_eq!(other.input().lens, LensSettings::default());
}

#[test]
fn the_lens_converts_to_and_from_the_lens_crates_amounts() {
    let s = lens(LensKind::Classic, 12.0, 34.0, 56.0);
    let a: ym_lens::LensAmounts = s.into();
    assert_eq!(
        (a.kind, a.flares, a.edges, a.anamorphic),
        (LensKind::Classic, 12.0, 34.0, 56.0)
    );
    assert_eq!(LensSettings::from(a), s);
    assert_eq!(a.is_identity(), s.is_identity());
    assert_eq!(
        ym_lens::LensAmounts::from(LensSettings::default()).is_identity(),
        LensSettings::default().is_identity()
    );
}

// ---------------------------------------------------------------------------------------------
// The graph (`docs/23` section 3)
// ---------------------------------------------------------------------------------------------

fn neutral() -> LookId {
    LookId::builtin("neutral")
}

fn read_doc(doc: Value) -> (DevelopSettings, Vec<ym_core::settings::FieldRef>) {
    DevelopSettings::from_value(doc.clone(), &neutral()).unwrap_or_else(|e| panic!("{doc}: {e}"))
}

/// The plain text a document is refused with.
fn refused(doc: Value) -> String {
    match DevelopSettings::from_value(doc.clone(), &neutral()) {
        Ok(_) => panic!("{doc} should be refused"),
        Err(e) => e.to_string(),
    }
}

/// The plain text a patch is refused with.
fn patch_refused(settings: &DevelopSettings, patch: Value) -> String {
    let applied: SettingsPatch = serde_json::from_value(patch.clone()).expect("a patch");
    let mut s = settings.clone();
    let err = s
        .apply_patch(&applied, None)
        .expect_err(&format!("{patch} should be refused"));
    assert_eq!(&s, settings, "a refused patch changes nothing");
    err.to_string()
}

fn corrector(id: u32) -> Value {
    json!({ "id": id, "kind": "corrector" })
}

fn link(port: &str, from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": port }, "to": { "node": to, "port": port } })
}

fn picture(from: Value, to: Value) -> Value {
    link("picture", from, to)
}

fn graph(items: Vec<Value>, links: Vec<Value>) -> Value {
    json!({ "nodes": { "items": items, "links": links } })
}

/// Input -> 1 -> ... -> Output, the way the default graph is linked.
fn chain(ids: &[u32]) -> Vec<Value> {
    let mut nodes = vec![json!("input")];
    nodes.extend(ids.iter().map(|id| json!(id)));
    nodes.push(json!("output"));
    nodes
        .windows(2)
        .map(|w| picture(w[0].clone(), w[1].clone()))
        .collect()
}

#[test]
fn an_empty_document_is_the_default_graph_and_a_document_is_read_over_it() {
    let default = DevelopSettings::with_look(neutral());
    for doc in [
        json!({}),
        json!({ "nodes": {} }),
        json!({ "nodes": { "items": [] } }),
        json!({ "nodes": { "items": [], "links": [] } })
            .as_object()
            .map(|_| json!({ "nodes": { "links": ym_core::settings::NodeGraph::default_links() } }))
            .unwrap(),
    ] {
        assert_eq!(read_doc(doc).0, default);
    }
    // The default graph is Input -> Main -> Output, at x = -1, 0 and 1.
    assert_eq!(
        default.nodes.ids(),
        [NodeId::Input, NodeId::MAIN, NodeId::Output]
    );
    assert_eq!(
        (
            default.input().pos,
            default.main().pos,
            default.output().pos
        ),
        ([-1.0, 0.0], [0.0, 0.0], [1.0, 0.0])
    );
    assert_eq!(
        default.nodes.links,
        vec![
            NodeLink::picture(NodeId::Input, NodeId::MAIN),
            NodeLink::picture(NodeId::MAIN, NodeId::Output)
        ]
    );
    // A document that lists only what it changes: the others keep their defaults.
    let (s, clamped) = read_doc(json!({ "nodes": { "items": [
        { "id": "input", "exposure": { "kind": "manual", "ev": 0.3 } },
        { "id": 1, "grade": { "contrast": { "kind": "manual", "value": 20 } } },
        { "id": "output", "look": "builtin:neutral" } ] } }));
    assert!(clamped.is_empty());
    assert_eq!(
        s.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.3)
    );
    assert_eq!(
        s.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(20.0)
    );
    assert!(s.main_grade().saturation.is_auto() && s.input().sharpen.is_auto());
    assert_eq!(s.main().name, "Main");
    assert_eq!(s.nodes.links, default.nodes.links);
    // `links` given replace the default links whole; a node left unlinked is kept.
    let (s, _) = read_doc(graph(vec![corrector(2)], chain(&[1])));
    assert_eq!(s.nodes.extra.len(), 1);
    assert_eq!(s.nodes.links.len(), 2);
    // A new node without its kind is told so; Parallel and the like are not known.
    assert_eq!(
        refused(json!({ "nodes": { "items": [ { "id": 5 } ] } })),
        "Node 5 is not in the default graph, so it needs its “kind”."
    );
    // A node that gives no `pos` is placed in a row by its number, the Output last; one that gives it keeps it.
    let (s, _) = read_doc(graph(
        vec![
            corrector(2),
            corrector(3),
            json!({ "id": 4, "kind": "corrector", "pos": [7, 2] }),
        ],
        chain(&[1, 2, 3, 4]),
    ));
    assert_eq!(s.main().pos, [0.0, 0.0]);
    assert_eq!(s.nodes.corrector(2).unwrap().pos, [1.0, 0.0]);
    assert_eq!(s.nodes.corrector(3).unwrap().pos, [2.0, 0.0]);
    assert_eq!(s.nodes.corrector(4).unwrap().pos, [7.0, 2.0]);
    assert_eq!(s.output().pos, [4.0, 0.0]);
}

#[test]
fn the_same_graph_is_the_same_bytes_whatever_order_a_document_lists_it_in() {
    let a = graph(
        vec![
            corrector(3),
            corrector(2),
            json!({ "id": "output" }),
            json!({ "id": "input" }),
        ],
        vec![
            picture(json!(3), json!("output")),
            picture(json!(2), json!(3)),
            picture(json!(1), json!(2)),
            picture(json!("input"), json!(1)),
            link("key", json!(2), json!(3)),
        ],
    );
    let b = graph(
        vec![
            json!({ "id": "input" }),
            corrector(2),
            json!({ "id": "output" }),
            corrector(3),
        ],
        vec![
            link("key", json!(2), json!(3)),
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!(3)),
            picture(json!(3), json!("output")),
        ],
    );
    // A key link next to the picture chain is fine: node 3 takes node 2's mask as a region.
    let (sa, _) = read_doc(a);
    let (sb, _) = read_doc(b);
    assert_eq!(sa, sb);
    let (ta, tb) = (
        serde_json::to_string(&sa).unwrap(),
        serde_json::to_string(&sb).unwrap(),
    );
    assert_eq!(ta, tb, "the same graph is the same bytes");
    // The items are written Input, correctors by id, Output; the links in the order of their ends.
    let v: Value = serde_json::from_str(&ta).unwrap();
    let ids: Vec<Value> = v["nodes"]["items"]
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
    let ends: Vec<(Value, Value, Value)> = v["nodes"]["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            (
                l["from"]["node"].clone(),
                l["from"]["port"].clone(),
                l["to"]["node"].clone(),
            )
        })
        .collect();
    assert_eq!(
        ends,
        [
            (json!("input"), json!("picture"), json!(1)),
            (json!(1), json!("picture"), json!(2)),
            (json!(2), json!("picture"), json!(3)),
            (json!(2), json!("key"), json!(3)),
            (json!(3), json!("picture"), json!("output")),
        ]
    );
    // And what is written reads back to itself, byte for byte.
    let (again, _) = read_doc(v);
    assert_eq!(serde_json::to_string(&again).unwrap(), ta);
}

#[test]
fn every_rule_of_the_graph_has_its_plain_text() {
    let two_on_an_input = {
        let mut links = chain(&[1, 2, 3]);
        links.push(picture(json!("input"), json!(3)));
        graph(vec![corrector(2), corrector(3)], links)
    };
    let cases: Vec<(&str, Value, &str)> = vec![
        (
            "an input that is a corrector",
            json!({ "nodes": { "items": [ { "id": "input", "kind": "corrector" } ] } }),
            "A photo needs its input, its output and node 1, Main.",
        ),
        (
            "Main that is an output",
            json!({ "nodes": { "items": [ { "id": 1, "kind": "output" } ] } }),
            "A photo needs its input, its output and node 1, Main.",
        ),
        (
            "a field of the Input in a corrector",
            json!({ "nodes": { "items": [ { "id": 2, "kind": "corrector", "exposure": { "kind": "auto" } } ] } }),
            "A corrector has no field called “exposure”.",
        ),
        (
            "a grade in the Output",
            json!({ "nodes": { "items": [ { "id": "output", "grade": {} } ] } }),
            "An output has no field called “grade”.",
        ),
        (
            "a link from the output",
            graph(
                vec![],
                vec![
                    picture(json!("input"), json!(1)),
                    picture(json!("output"), json!(1)),
                    picture(json!(1), json!("output")),
                ],
            ),
            "A link goes from the output of one node to the input of another.",
        ),
        (
            "a link into the input",
            graph(
                vec![],
                vec![
                    picture(json!(1), json!("input")),
                    picture(json!("input"), json!(1)),
                    picture(json!(1), json!("output")),
                ],
            ),
            "A link goes from the output of one node to the input of another.",
        ),
        (
            "a picture joined to a key",
            graph(vec![corrector(2)], {
                let mut l = chain(&[1, 2]);
                l.push(json!({ "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "key" } }));
                l
            }),
            "A link joins a picture to a picture, or a key to a key.",
        ),
        (
            "a link to a node the photo does not have",
            graph(
                vec![],
                vec![
                    picture(json!("input"), json!(1)),
                    picture(json!(1), json!(7)),
                    picture(json!(1), json!("output")),
                ],
            ),
            "A link names node 7, which this photo does not have.",
        ),
        (
            "a port that does not exist",
            graph(
                vec![],
                vec![
                    picture(json!("input"), json!(1)),
                    picture(json!(1), json!("output")),
                    link("mask", json!(1), json!("output")),
                ],
            ),
            "Node 1 has no port called “mask”.",
        ),
        (
            "a key into the output",
            graph(
                vec![],
                vec![
                    picture(json!("input"), json!(1)),
                    picture(json!(1), json!("output")),
                    link("key", json!(1), json!("output")),
                ],
            ),
            "The output has no port called “key”.",
        ),
        (
            "a second link on an input",
            two_on_an_input,
            "An input can have one link. Node 3 has two on its picture input.",
        ),
        (
            "no link to the output",
            graph(vec![], vec![picture(json!("input"), json!(1))]),
            "The output needs one picture link.",
        ),
        (
            "Main not linked to the input",
            graph(
                vec![corrector(2)],
                vec![
                    picture(json!("input"), json!(2)),
                    picture(json!(2), json!(1)),
                    picture(json!(1), json!("output")),
                ],
            ),
            "Node 1, Main, must take its picture from the input.",
        ),
        (
            "Main with no link at all",
            graph(vec![], vec![picture(json!("input"), json!("output"))]),
            "Node 1, Main, must take its picture from the input.",
        ),
        (
            "two nodes with one id",
            graph(vec![corrector(3), corrector(3)], chain(&[1, 3])),
            "Two nodes have the id 3.",
        ),
        (
            "a node of a kind this version does not have",
            json!({ "nodes": { "items": [ { "id": 2, "kind": "parallel" } ] } }),
            "This settings file has a node of a kind (“parallel”) that this version of Yellowmoon does not have. Update Yellowmoon.",
        ),
    ];
    for (label, doc, text) in cases {
        assert_eq!(refused(doc), text, "{label}");
    }
    // The text of a rule is the text of its error, whichever way it is reached.
    assert_eq!(
        ym_core::settings::GraphError::Cycle.to_string(),
        "The nodes of this photo are linked in a circle. Remove a link."
    );
}

#[test]
fn a_graph_with_a_circle_is_refused_through_pictures_keys_or_both() {
    let text = "The nodes of this photo are linked in a circle. Remove a link.";
    // Through pictures: 2 -> 3 -> 2.
    let pictures = graph(
        vec![corrector(2), corrector(3)],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!(3)),
            picture(json!(3), json!(2)),
            picture(json!(3), json!("output")),
        ],
    );
    // `3 -> 2` makes a second link on 2's picture input; the circle is the point of this case, so
    // the graph below has nothing else wrong: 2 feeds 3's picture, and 3 feeds 2's key.
    assert_ne!(refused(pictures), "");
    let via_picture_then_key = graph(
        vec![corrector(2), corrector(3)],
        vec![
            picture(json!("input"), json!(1)),
            picture(json!(1), json!(2)),
            picture(json!(2), json!(3)),
            link("key", json!(3), json!(2)),
            picture(json!(3), json!("output")),
        ],
    );
    assert_eq!(
        refused(via_picture_then_key),
        text,
        "picture and key links together"
    );
    // Through keys alone: two nodes that take each other's masks.
    let keys = graph(vec![corrector(2), corrector(3)], {
        let mut l = chain(&[1, 2, 3]);
        l.push(link("key", json!(2), json!(3)));
        l.push(link("key", json!(3), json!(2)));
        l
    });
    assert_eq!(refused(keys), text, "key links alone");
    // A node whose key feeds itself.
    let own = graph(vec![corrector(2)], {
        let mut l = chain(&[1, 2]);
        l.push(link("key", json!(2), json!(2)));
        l
    });
    assert_eq!(refused(own), text, "a node that feeds itself");
    // The same nodes in a line, with a key link that goes forward, are fine.
    let ok = graph(vec![corrector(2), corrector(3)], {
        let mut l = chain(&[1, 2, 3]);
        l.push(link("key", json!(2), json!(3)));
        l
    });
    assert!(DevelopSettings::from_value(ok, &neutral()).is_ok());
    // A patch that would close a circle is refused and changes nothing.
    let (s, _) = read_doc(graph(vec![corrector(2), corrector(3)], {
        let mut l = chain(&[1, 2, 3]);
        l.push(link("key", json!(2), json!(3)));
        l
    }));
    let mut links = serde_json::to_value(&s.nodes.links).unwrap();
    links
        .as_array_mut()
        .unwrap()
        .push(link("key", json!(3), json!(2)));
    assert_eq!(patch_refused(&s, json!({ "links": links })), text);
}

#[test]
fn the_limits_are_sanity_bounds_and_every_refusal_is_atomic() {
    // 64 correctors, Main included, are fine; a 65th is not (in a document, and in a patch).
    let mut s = DevelopSettings::default();
    let mut nodes = serde_json::Map::new();
    for id in 2..=64u32 {
        nodes.insert(id.to_string(), json!({ "kind": "corrector" }));
    }
    s.apply_patch(
        &serde_json::from_value(json!({ "nodes": nodes })).unwrap(),
        None,
    )
    .expect("64 correctors");
    assert_eq!(s.nodes.corrector_count(), 64);
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "65": { "kind": "corrector" } } })),
        "A photo can have up to 64 nodes."
    );
    let items: Vec<Value> = (2..=65).map(corrector).collect();
    assert_eq!(
        refused(graph(items, chain(&[1]))),
        "A photo can have up to 64 nodes."
    );
    // 64 masks in a node, 65 are not.
    let rows = |n: usize| -> Vec<Value> { (0..n).map(|_| json!({ "kind": "linear" })).collect() };
    let mut s = DevelopSettings::default();
    s.apply_patch(
        &serde_json::from_value(json!({ "nodes": { "1": { "masks": { "items": rows(64) } } } }))
            .unwrap(),
        None,
    )
    .expect("64 masks");
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "1": { "masks": { "items": rows(65) } } } })
        ),
        "A node can have up to 64 masks."
    );
    assert_eq!(
        refused(json!({ "nodes": { "items": [ { "id": 1, "masks": { "items": rows(65) } } ] } })),
        "A node can have up to 64 masks."
    );
    // 100,000 brush points in a mask, 100,001 are not; 256 Object points, 257 are not.
    let brush = |points: usize| json!({ "kind": "brush", "strokes": [ { "points": vec![[0.5, 0.5]; points] } ] });
    let object = |points: usize| json!({ "kind": "object", "points": vec![json!([0.5, 0.5, true]); points] });
    let with = |row: Value| json!({ "nodes": { "1": { "masks": { "items": [row] } } } });
    let mut s = DevelopSettings::default();
    s.apply_patch(&serde_json::from_value(with(brush(100_000))).unwrap(), None)
        .expect("100,000 brush points");
    s.apply_patch(&serde_json::from_value(with(object(256))).unwrap(), None)
        .expect("256 Object points");
    assert_eq!(
        patch_refused(&s, with(brush(100_001))),
        "A brush mask can have up to 100,000 points."
    );
    assert_eq!(
        patch_refused(&s, with(object(257))),
        "An Object mask can have up to 256 points."
    );
    // The points of several strokes count together.
    let two_strokes = json!({ "kind": "brush", "strokes": [
        { "points": vec![[0.5, 0.5]; 60_000] }, { "points": vec![[0.5, 0.5]; 40_001] } ] });
    assert_eq!(
        patch_refused(&s, with(two_strokes)),
        "A brush mask can have up to 100,000 points."
    );
}

#[test]
fn auto_belongs_to_main() {
    let auto = json!({ "kind": "auto" });
    let manual0 = json!({ "kind": "manual", "value": 0.0 });
    let wheel0 = json!({ "kind": "manual", "point": [0.0, 0.0] });
    // A document with Auto in node 2: it reads as a manual 0, field by field and wheel by wheel, and
    // it is written as one. A field left out of such a node is a manual 0 too. Main keeps its Auto.
    let (s, clamped) = read_doc(graph(
        vec![
            json!({ "id": 2, "kind": "corrector", "grade": {
                "contrast": auto, "highlights": auto, "shadows": auto, "black": auto,
                "saturation": auto,
                "color_balance": { "shadows": auto, "highlights": auto } } }),
            corrector(3),
        ],
        chain(&[1, 2, 3]),
    ));
    assert!(
        clamped.is_empty(),
        "reading Auto as 0 is not clamping: {clamped:?}"
    );
    for id in [2, 3] {
        let g = &s.nodes.corrector(id).unwrap().grade;
        for setting in [g.contrast, g.highlights, g.shadows, g.black, g.saturation] {
            assert_eq!(
                setting,
                ym_core::settings::PercentSetting::manual_value(0.0),
                "node {id}"
            );
        }
        assert_eq!(
            g.color_balance.shadows,
            ym_core::settings::BalanceSetting::manual_point([0.0, 0.0])
        );
        assert_eq!(
            g.color_balance.highlights,
            ym_core::settings::BalanceSetting::manual_point([0.0, 0.0])
        );
    }
    assert!(s.main_grade().contrast.is_auto() && s.main_grade().color_balance.shadows.is_auto());
    let v = serde_json::to_value(&s).unwrap();
    let node2 = &v["nodes"]["items"][2]["grade"];
    assert_eq!(node2["contrast"], manual0);
    assert_eq!(node2["color_balance"]["shadows"], wheel0);
    assert_eq!(v["nodes"]["items"][3]["grade"]["saturation"], manual0);
    assert_eq!(
        v["nodes"]["items"][1]["grade"]["contrast"], auto,
        "Main keeps its Auto"
    );
    // Through a patch: Auto on node 3 stores a manual 0, and says nothing about clamping.
    let mut t = s.clone();
    t.apply_patch(
        &serde_json::from_value(json!({ "nodes": { "3": { "grade": { "contrast": { "kind": "manual", "value": 30 } } } } })).unwrap(),
        None,
    )
    .unwrap();
    let report = t
        .apply_patch(
            &serde_json::from_value(json!({ "nodes": { "3": { "grade": { "contrast": auto } } } }))
                .unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(
        t.nodes.corrector(3).unwrap().grade.contrast,
        ym_core::settings::PercentSetting::manual_value(0.0)
    );
    assert!(report.clamped.is_empty());
    assert_eq!(
        report.changed,
        vec![FieldRef::new(NodeId::Corrector(3), SettingsField::Contrast)]
    );
    // A new node is a manual 0 everywhere (the grade of `CorrectorNode::default`).
    let mut t = DevelopSettings::default();
    t.apply_patch(
        &serde_json::from_value(json!({ "nodes": { "2": { "kind": "corrector" } } })).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        t.nodes.corrector(2).unwrap().grade,
        ym_core::settings::NodeGrade::neutral()
    );
    assert_eq!(t.nodes.corrector(2).unwrap().name, "Corrector");
    // The resolved values of such a node are its numbers (nothing resolves against the estimates).
    let (r, _) = rich_resolved();
    for c in &r.values.nodes {
        assert!(
            !r.settings
                .nodes
                .corrector(c.id)
                .unwrap()
                .grade
                .contrast
                .is_auto()
        );
    }
    assert_eq!(r.values.nodes.len(), 2);
    // Main keeps its Auto through copy: the copied patch gives Main's Auto as Auto.
    let (copied, _) = default_resolved();
    let patch = SettingsPatch::copied_from(&copied);
    let grade = patch.main_patch().and_then(|p| p.grade.clone()).unwrap();
    assert_eq!(grade.contrast, Some(AutoOr::Auto));
    // ... and a pasted graph keeps Main's Auto in the target's Main and gives node 3 its manual 0.
    let mut source = DevelopSettings::default();
    source
        .apply_patch(
            &serde_json::from_value(
                json!({ "nodes": { "3": { "kind": "corrector", "grade": { "contrast": auto } } } }),
            )
            .unwrap(),
            None,
        )
        .unwrap();
    let mut target = DevelopSettings::default();
    target.main_grade_mut().contrast = ym_core::settings::PercentSetting::manual_value(40.0);
    let pasted = SettingsPatch::between(&DevelopSettings::default(), &source);
    target.apply_patch(&pasted, None).unwrap();
    assert_eq!(
        target.main_grade().contrast,
        ym_core::settings::PercentSetting::manual_value(40.0),
        "the target's own Main is not touched by a patch that says nothing about it"
    );
    assert_eq!(
        target.nodes.corrector(3).unwrap().grade.contrast,
        ym_core::settings::PercentSetting::manual_value(0.0)
    );
}

/// The resolved settings of a default photo and of the rich graph, against the estimates of a small
/// synthetic scene.
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

fn rich_resolved() -> (ResolvedSettings, DevelopSettings) {
    let s = rich();
    (resolved_of(&s), s)
}

fn default_resolved() -> (ResolvedSettings, DevelopSettings) {
    let s = DevelopSettings::default();
    (resolved_of(&s), s)
}

// ---------------------------------------------------------------------------------------------
// Node patches
// ---------------------------------------------------------------------------------------------

fn node_patch(v: Value) -> SettingsPatch {
    serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{v}: {e}"))
}

#[test]
fn a_patch_of_one_nodes_partial_changes_only_that_nodes_fields() {
    let mut s = rich();
    let before = s.clone();
    let report = apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "trim": { "tint": 12 }, "name": "Peak" } } })),
    );
    let node = s.nodes.corrector(2).unwrap();
    assert_eq!(
        (node.trim.tint, node.trim.exposure_ev),
        (12.0, 0.35),
        "a partial trim"
    );
    assert_eq!(node.name, "Peak");
    assert_eq!(
        report.changed,
        vec![
            FieldRef::new(NodeId::Corrector(2), SettingsField::Name),
            FieldRef::new(NodeId::Corrector(2), SettingsField::Trim),
        ]
    );
    // Nothing else moved: the Input, Main, the Output and the other corrector are as they were.
    assert_eq!(s.input(), before.input());
    assert_eq!(s.main(), before.main());
    assert_eq!(s.output(), before.output());
    assert_eq!(s.nodes.corrector(3), before.nodes.corrector(3));
    assert_eq!(s.nodes.links, before.nodes.links);
    // A slider on Main is a partial of Main's grade.
    let report = apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "1": { "grade": { "contrast": { "kind": "manual", "value": 13 } } } } }),
        ),
    );
    assert_eq!(
        report.changed,
        vec![FieldRef::new(NodeId::MAIN, SettingsField::Contrast)]
    );
    assert_eq!(
        report.changed[0].to_string(),
        "1.grade.contrast",
        "a field is named <node>.<field path>"
    );
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
}

#[test]
fn name_and_pos_re_run_nothing_and_the_rest_of_a_nodes_fields_have_their_groups() {
    let mut s = rich();
    let report = apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "name": "Peak", "pos": [4, 1] } } })),
    );
    assert_eq!(report.first_group(), None, "{report:?}");
    assert!(report.clamped.is_empty());
    // Moving a card is a change (undo holds it) that renders nothing.
    assert_eq!(report.changed.len(), 2);
    for (patch, group) in [
        (
            json!({ "nodes": { "input": { "decode": { "neural_demosaic": false } } } }),
            None,
        ),
        (
            json!({ "nodes": { "input": { "decode": { "highlight_recovery": false } } } }),
            Some(StageGroup::Raw),
        ),
        (
            json!({ "nodes": { "input": { "lens": { "flares": 5 } } } }),
            Some(StageGroup::Lens),
        ),
        (
            json!({ "nodes": { "input": { "exposure": { "kind": "manual", "ev": 1 } } } }),
            Some(StageGroup::Color),
        ),
        (
            json!({ "nodes": { "input": { "orientation": 180 } } }),
            Some(StageGroup::Geometry),
        ),
        (json!({ "nodes": { "2": { "bypass": false } } }), None),
        (
            json!({ "nodes": { "2": { "bypass": true } } }),
            Some(StageGroup::Grade),
        ),
        (
            json!({ "nodes": { "2": { "trim": { "exposure_ev": 1 } } } }),
            Some(StageGroup::Grade),
        ),
        (
            json!({ "nodes": { "2": { "masks": { "invert": false } } } }),
            Some(StageGroup::Grade),
        ),
        (
            json!({ "nodes": { "output": { "look_intensity": 40 } } }),
            Some(StageGroup::Grade),
        ),
        (
            json!({ "nodes": { "output": { "output_sharpening": 40 } } }),
            Some(StageGroup::Output),
        ),
    ] {
        let mut t = rich();
        let report = apply(&mut t, node_patch(patch.clone()));
        assert_eq!(report.first_group(), group, "{patch}");
    }
}

#[test]
fn a_new_id_with_a_kind_adds_a_node_remove_drops_it_and_links_replace_whole() {
    let mut s = DevelopSettings::default();
    // Add node 2 and link it in, in one patch.
    let report = apply(
        &mut s,
        node_patch(json!({
            "nodes": { "2": { "kind": "corrector", "name": "Warmth", "trim": { "temperature": 30 } } },
            "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
                { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
            ]
        })),
    );
    assert_eq!(report.added, vec![NodeId::Corrector(2)]);
    assert!(report.links_changed);
    assert_eq!(report.first_group(), Some(StageGroup::Grade));
    let node = s.nodes.corrector(2).unwrap();
    assert_eq!(
        (node.name.as_str(), node.trim.temperature),
        ("Warmth", 30.0)
    );
    assert_eq!(
        node.pos,
        [1.0, 0.0],
        "a node that gives no place goes at the end of the row"
    );
    assert_eq!(s.nodes.links.len(), 3);
    // A patch of a node that exists changes it and adds nothing.
    let report = apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "bypass": true } } })),
    );
    assert!(report.added.is_empty() && s.nodes.corrector(2).unwrap().bypass);
    // Remove takes the node and the links that touch it; the others stay. Node 3 hangs off Main and
    // takes node 2's mask, so removing node 2 drops its picture link and the key link.
    apply(
        &mut s,
        node_patch(json!({
            "nodes": { "3": { "kind": "corrector" } },
            "links": [
                { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
                { "from": { "node": 1, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
                { "from": { "node": 2, "port": "key" }, "to": { "node": 3, "port": "key" } }
            ]
        })),
    );
    let report = apply(&mut s, node_patch(json!({ "remove": [2] })));
    assert_eq!(report.removed, vec![NodeId::Corrector(2)]);
    assert!(report.links_changed);
    assert!(s.nodes.corrector(2).is_none() && s.nodes.corrector(3).is_some());
    assert_eq!(
        s.nodes.links,
        vec![
            NodeLink::picture(NodeId::Input, NodeId::MAIN),
            NodeLink::picture(NodeId::MAIN, NodeId::Corrector(3)),
            NodeLink::picture(NodeId::MAIN, NodeId::Output),
        ],
        "the links that touched it went with it"
    );
    // Taking node 2 out of the chain leaves node 3 without a picture: a dead node, which is kept. The
    // patch that removes a node says how to link the rest.
    let mut t = rich();
    let mut dead = t.clone();
    apply(&mut dead, node_patch(json!({ "remove": [2] })));
    assert!(
        !dead
            .nodes
            .links
            .iter()
            .any(|l| l.to == ym_core::settings::NodeRef::picture(NodeId::Corrector(3)))
    );
    apply(
        &mut t,
        node_patch(json!({ "remove": [2], "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
            { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
        ] })),
    );
    assert_eq!(t.nodes.extra.keys().copied().collect::<Vec<_>>(), [3]);
    // The same links again change nothing.
    let links = serde_json::to_value(&t.nodes.links).unwrap();
    let again = apply(&mut t, node_patch(json!({ "links": links })));
    assert!(again.is_empty());
}

#[test]
fn the_input_the_output_and_main_cannot_be_removed_and_a_bad_patch_changes_nothing() {
    let s = rich();
    for id in [json!("input"), json!("output"), json!(1)] {
        assert_eq!(
            patch_refused(&s, json!({ "remove": [id] })),
            "The input, the output and node 1, Main, can't be removed."
        );
    }
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "7": { "trim": { "tint": 1 } } } })),
        "The patch names node 7, which this photo does not have."
    );
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "7": { "kind": "input" } } })),
        "Node 7 is a corrector, not an input."
    );
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "2": { "kind": "output" } } })),
        "Node 2 is a corrector, not an output."
    );
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "input": { "kind": "corrector" } } })),
        "The input is an input, not a corrector."
    );
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "2": { "exposure": { "kind": "auto" } } } })
        ),
        "A corrector has no field called “exposure”."
    );
    assert_eq!(
        patch_refused(&s, json!({ "nodes": { "output": { "bypass": true } } })),
        "An output has no field called “bypass”."
    );
    // A valid change next to a refused one: nothing is applied.
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": {
                "1": { "grade": { "brightness": 5 } },
                "2": { "trim": { "tint": 1 } },
                "9": { "trim": { "tint": 1 } } } })
        ),
        "The patch names node 9, which this photo does not have."
    );
    // A number that is not a number is named by its place in the document.
    let nan = SettingsPatch::main(NodePatch {
        trim: Some(ym_core::settings::TrimPatch {
            tint: Some(f64::NAN),
            ..Default::default()
        }),
        ..Default::default()
    });
    let mut t = s.clone();
    assert_eq!(
        t.apply_patch(&nan, None).unwrap_err(),
        SettingsError::NotFinite("nodes.items[1].trim.tint".to_owned())
    );
    assert_eq!(t, s);
    // Ids are numbers from 1: 0 and negative numbers are not ids.
    assert!(serde_json::from_value::<SettingsPatch>(json!({ "nodes": { "0": {} } })).is_err());
    assert!(serde_json::from_value::<SettingsPatch>(json!({ "nodes": { "-1": {} } })).is_err());
    assert!(serde_json::from_value::<SettingsPatch>(json!({ "nodes": { "middle": {} } })).is_err());
}

#[test]
fn clamping_names_the_node_and_the_field_and_a_pos_off_the_canvas_is_pulled_back() {
    let mut s = rich();
    let report = apply(
        &mut s,
        node_patch(json!({ "nodes": {
            "1": { "grade": { "contrast": { "kind": "manual", "value": 400 } }, "trim": { "exposure_ev": 9 } },
            "2": { "pos": [5000, -5000], "trim": { "temperature": 150 },
                   "masks": { "edit": { "5": { "opacity": 3 }, "0": { "feather": 2, "grow": -4, "refine": 9 } } } },
            "output": { "look_intensity": 900 } } })),
    );
    let at = |n, f| FieldRef::new(n, f);
    for r in [
        at(NodeId::MAIN, SettingsField::Contrast),
        at(NodeId::MAIN, SettingsField::Trim),
        at(NodeId::Corrector(2), SettingsField::Pos),
        at(NodeId::Corrector(2), SettingsField::Trim),
        at(NodeId::Corrector(2), SettingsField::Masks),
        at(NodeId::Output, SettingsField::LookIntensity),
    ] {
        assert!(
            report.clamped.contains(&r),
            "{r} was clamped: {:?}",
            report.clamped
        );
    }
    let node = s.nodes.corrector(2).unwrap();
    assert_eq!(node.pos, [1000.0, -1000.0]);
    assert_eq!(node.trim.temperature, 100.0);
    assert_eq!(s.main().trim.exposure_ev, 4.0);
    let ym_core::settings::MaskShape::Object(o) = &node.masks.items[0].shape else {
        panic!("an Object row")
    };
    assert_eq!((o.feather, o.grow, o.refine), (1.0, -1.0, 1.0));
    assert_eq!(node.masks.items[5].opacity, 1.0);
    // The same values already in range report no clamping at all.
    let again = apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "pos": [1000, -1000] } } })),
    );
    assert!(again.clamped.is_empty() && again.is_empty());
    // Names are trimmed and cut at 32 characters; an empty one reads as the default.
    apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "name": "  Peak  " } } })),
    );
    assert_eq!(s.nodes.corrector(2).unwrap().name, "Peak");
    apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "name": "x".repeat(40) } } })),
    );
    assert_eq!(s.nodes.corrector(2).unwrap().name.chars().count(), 32);
    apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "name": "   " } } })),
    );
    assert_eq!(s.nodes.corrector(2).unwrap().name, "Corrector");
    apply(
        &mut s,
        node_patch(json!({ "nodes": { "1": { "name": "" } } })),
    );
    assert_eq!(s.main().name, "Main");
}

#[test]
fn mask_rows_are_edited_by_index_and_a_stroke_is_an_append_with_an_exact_inverse() {
    let mut s = rich();
    let before = s.nodes.corrector(2).unwrap().masks.clone();
    // A row patch: a partial of that row's fields.
    let report = apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "2": { "masks": { "edit": { "4": { "feather": 0.4 }, "0": { "points": [[0.1, 0.2, true]], "box": null } } } } } }),
        ),
    );
    assert_eq!(
        report.changed,
        vec![FieldRef::new(NodeId::Corrector(2), SettingsField::Masks)]
    );
    let masks = &s.nodes.corrector(2).unwrap().masks;
    let ym_core::settings::MaskShape::Linear(l) = &masks.items[4].shape else {
        panic!("a Linear row")
    };
    assert_eq!((l.feather, l.angle_deg), (0.4, 0.0));
    let ym_core::settings::MaskShape::Object(o) = &masks.items[0].shape else {
        panic!("an Object row")
    };
    assert_eq!((o.points.clone(), o.box_), (vec![(0.1, 0.2, true)], None));
    // A row patch cannot change the row's kind, nor name a field the kind does not have, nor a row
    // that is not there.
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "2": { "masks": { "edit": { "4": { "kind": "radial" } } } } } })
        ),
        "A mask can't change its kind. Add a new mask instead."
    );
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "2": { "masks": { "edit": { "4": { "points": [] } } } } } })
        ),
        "A Linear mask has no field called “points”."
    );
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "2": { "masks": { "edit": { "12": { "feather": 0.1 } } } } } })
        ),
        "Node 2 has no mask number 12."
    );
    assert_eq!(
        patch_refused(
            &s,
            json!({ "nodes": { "2": { "masks": { "edit": { "0": { "strokes_append": [] } } } } } })
        ),
        "An Object mask has no field called “strokes_append”."
    );
    // The same kind named again is no change of kind.
    apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "2": { "masks": { "edit": { "4": { "kind": "linear", "opacity": 0.25 } } } } } }),
        ),
    );
    // A stroke is an append; its inverse is `strokes_keep` with the old count.
    let stroke = json!({ "size": 0.03, "hardness": 0.5, "erase": false, "points": [[0.4, 0.5], [0.45, 0.52]] });
    let strokes_of = |s: &DevelopSettings| -> usize {
        match &s.nodes.corrector(2).unwrap().masks.items[6].shape {
            ym_core::settings::MaskShape::Brush(b) => b.strokes.len(),
            _ => panic!("a Brush row"),
        }
    };
    let (kept_before, snapshot) = (strokes_of(&s), s.clone());
    apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "2": { "masks": { "edit": { "6": { "strokes_append": [stroke] } } } } } }),
        ),
    );
    assert_eq!(strokes_of(&s), kept_before + 1);
    apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "2": { "masks": { "edit": { "6": { "strokes_keep": kept_before } } } } } }),
        ),
    );
    assert_eq!(
        s, snapshot,
        "an append and its inverse restore the strokes exactly"
    );
    // The whole list is replaced by a patch that gives it (add, remove or reorder rows), and the
    // node's Invert is its own field.
    apply(
        &mut s,
        node_patch(
            json!({ "nodes": { "2": { "masks": { "invert": false, "items": [ { "kind": "radial", "mode": "subtract" } ] } } } }),
        ),
    );
    let masks = &s.nodes.corrector(2).unwrap().masks;
    assert_eq!((masks.invert, masks.items.len()), (false, 1));
    assert_eq!(masks.items[0].mode, ym_core::settings::MaskMode::Subtract);
    assert_ne!(masks, &before);
    // An Object row with no point and no box selects nothing: it is a row like any other.
    apply(
        &mut s,
        node_patch(json!({ "nodes": { "2": { "masks": { "items": [ { "kind": "object" } ] } } } })),
    );
    assert!(matches!(
        &s.nodes.corrector(2).unwrap().masks.items[0].shape,
        ym_core::settings::MaskShape::Object(o) if o.points.is_empty() && o.box_.is_none()
    ));
}

#[test]
fn reset_puts_a_nodes_fields_at_their_defaults() {
    let look = LookId::builtin("signature");
    // Main: Auto where there is Auto. Another corrector: a manual 0. The Output: the default look.
    let mut s = rich();
    let mut patch = SettingsPatch::reset(NodeId::MAIN, SettingsField::Contrast, &look);
    patch = patch
        .and(
            NodeId::Corrector(2),
            SettingsPatch::reset(NodeId::Corrector(2), SettingsField::Highlights, &look)
                .nodes
                .remove(&NodeId::Corrector(2))
                .unwrap(),
        )
        .and(
            NodeId::Output,
            SettingsPatch::reset(NodeId::Output, SettingsField::Look, &look)
                .nodes
                .remove(&NodeId::Output)
                .unwrap(),
        );
    apply(&mut s, patch);
    assert!(s.main_grade().contrast.is_auto());
    assert_eq!(
        s.nodes.corrector(2).unwrap().grade.highlights,
        ym_core::settings::PercentSetting::manual_value(0.0)
    );
    assert_eq!(s.output().look, look);
    // The node head's Reset: a corrector's trim and grade (not its name, bypass, place or masks); the
    // Input's and the Output's: all their fields but the place.
    let mut s = rich();
    let name = s.nodes.corrector(2).unwrap().name.clone();
    let masks = s.nodes.corrector(2).unwrap().masks.clone();
    apply(
        &mut s,
        SettingsPatch::reset_node(NodeId::Corrector(2), &look),
    );
    let node = s.nodes.corrector(2).unwrap();
    assert_eq!(node.trim, NodeTrim::default());
    assert_eq!(node.grade, ym_core::settings::NodeGrade::neutral());
    assert_eq!(
        (node.name.clone(), node.masks.clone(), node.pos),
        (name, masks, [1.0, 0.0])
    );
    apply(&mut s, SettingsPatch::reset_node(NodeId::Input, &look));
    let mut want = ym_core::settings::InputNode::default();
    want.pos = s.input().pos;
    assert_eq!(s.input(), &want);
    apply(&mut s, SettingsPatch::reset_node(NodeId::Output, &look));
    assert_eq!(s.output().look, look);
    assert_eq!(s.output().output_sharpening, 100.0);
    // Re-run (the Automatic card) resets the Auto fields of the Input and Main and leaves the rest of the graph alone.
    let mut s = rich();
    s.reset_automatic(true);
    assert!(s.main_grade().contrast.is_auto() && s.input().exposure.is_auto());
    assert_eq!(s.nodes.extra, rich().nodes.extra);
    assert_eq!(s.nodes.links, rich().nodes.links);
    assert_eq!(s.main().trim, rich().main().trim);
}

#[test]
fn the_patch_between_two_settings_is_what_an_undo_step_holds() {
    let a = rich();
    let mut b = a.clone();
    // Touch one node's fields, add a node, remove another and change the links.
    b.nodes.extra.remove(&3);
    b.nodes.extra.get_mut(&2).unwrap().trim.tint = 9.0;
    let mut warm = CorrectorNode::default();
    warm.name = "Warmth".to_owned();
    warm.trim.temperature = 30.0;
    b.nodes.extra.insert(4, warm);
    b.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Corrector(4)),
        NodeLink::picture(NodeId::Corrector(4), NodeId::Output),
    ];
    b.main_grade_mut().brightness = 3.0;
    let forward = SettingsPatch::between(&a, &b);
    // It holds only what changed: node 2's trim, Main's brightness, node 4 whole with its kind,
    // node 3 in `remove` and the links.
    assert_eq!(forward.remove, vec![NodeId::Corrector(3)]);
    let mut ids: Vec<NodeId> = forward.nodes.keys().copied().collect();
    ids.sort();
    assert_eq!(
        ids,
        [NodeId::MAIN, NodeId::Corrector(2), NodeId::Corrector(4)]
    );
    assert_eq!(
        forward.nodes[&NodeId::Corrector(2)].trim.map(|t| t.tint),
        Some(Some(9.0))
    );
    assert_eq!(
        forward.nodes[&NodeId::Corrector(2)].name,
        None,
        "unchanged fields are not copied"
    );
    assert_eq!(
        forward.nodes[&NodeId::Corrector(4)].kind,
        Some(ym_core::settings::NodeKind::Corrector)
    );
    assert!(forward.links.is_some());
    // Applying it to `a` gives `b`; the inverse gives `a` back; both are exact.
    let mut x = a.clone();
    apply(&mut x, forward);
    assert_eq!(x, b);
    let backward = SettingsPatch::between(&b, &a);
    apply(&mut x, backward);
    assert_eq!(x, a);
    // Two equal settings have an empty patch, and a patch of one field holds just that field.
    assert!(SettingsPatch::between(&a, &a).is_empty());
    let mut c = a.clone();
    c.output_mut().look_intensity = 70.0;
    let one = SettingsPatch::between(&a, &c);
    assert_eq!(one.nodes.len(), 1);
    assert_eq!(one.nodes[&NodeId::Output].look_intensity, Some(70.0));
    assert!(one.remove.is_empty() && one.links.is_none());
}

// ---------------------------------------------------------------------------------------------
// Copy and paste
// ---------------------------------------------------------------------------------------------

#[test]
fn copy_takes_every_node_and_link_but_three_fields_of_the_input_and_the_object_clicks() {
    let (resolved, source) = rich_resolved();
    let patch = SettingsPatch::copied_from(&resolved);
    // Every node and every link, and nothing to remove.
    let mut ids: Vec<NodeId> = patch.nodes.keys().copied().collect();
    ids.sort();
    assert_eq!(ids, source.nodes.ids());
    assert_eq!(patch.links.as_ref(), Some(&source.nodes.links));
    assert!(patch.remove.is_empty());
    // The Input's `crop`, `orientation` and `white_pick` stay with the photo; the rest is copied.
    let input = patch.input_patch().unwrap();
    assert_eq!(
        (input.crop, input.orientation, input.white_pick),
        (None, None, None)
    );
    assert!(input.exposure.is_some() && input.lens.is_some() && input.pos.is_some());
    assert_eq!(
        COPY_LEAVES_OUT.map(|r| (r.node, r.field)),
        [
            (NodeId::Input, SettingsField::Crop),
            (NodeId::Input, SettingsField::Orientation),
            (NodeId::Input, SettingsField::WhitePick)
        ]
    );
    // Names, places, trims, grades and masks are copied; a node comes with its kind, so that a
    // photo that lacks it gets it.
    let node2 = &patch.nodes[&NodeId::Corrector(2)];
    assert_eq!(node2.kind, Some(ym_core::settings::NodeKind::Corrector));
    assert_eq!(node2.name.as_deref(), Some("Mountain"));
    assert_eq!(node2.pos, Some([1.0, 0.0]));
    assert!(node2.trim.is_some() && node2.grade.is_some());
    // The Object row arrives empty: no points and no box, so it selects nothing; the others are
    // copied as they are, the Subject, People and Background rows too (their rasters are made again
    // on the new photo).
    let rows = node2.masks.as_ref().unwrap().items.as_ref().unwrap();
    assert_eq!(rows.len(), 9);
    assert!(matches!(
        &rows[0].shape,
        ym_core::settings::MaskShape::Object(o) if o.points.is_empty() && o.box_.is_none() && o.refine == 0.7
    ));
    assert_eq!(rows[1], source.nodes.corrector(2).unwrap().masks.items[1]);
    assert_eq!(rows[6], source.nodes.corrector(2).unwrap().masks.items[6]);
    // Pasted onto a photo with other settings: its graph is replaced by the copy, its own crop,
    // turn and picked white are kept.
    let mut target = DevelopSettings::default();
    target.input_mut().crop = Some(Crop {
        rect: NormRect::FULL,
        angle_deg: 3.0,
        aspect: CropAspect::Free,
        auto: false,
    });
    target.input_mut().orientation = Orientation::Deg180;
    target.input_mut().white_pick = Some([0.1, 0.2]);
    let mut own = CorrectorNode::default();
    own.name = "Mine".to_owned();
    target.nodes.extra.insert(9, own);
    target.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(9)),
        NodeLink::picture(NodeId::Corrector(9), NodeId::Output),
    ];
    let for_target = patch.pasted_onto(&target);
    assert_eq!(
        for_target.remove,
        vec![NodeId::Corrector(9)],
        "the target's node the copy lacks"
    );
    let report = target.apply_patch(&for_target, None).unwrap();
    assert!(!report.is_empty());
    assert!(target.nodes.corrector(9).is_none());
    assert_eq!(target.nodes.links, source.nodes.links);
    assert_eq!(target.input().crop.unwrap().angle_deg, 3.0);
    assert_eq!(target.input().orientation, Orientation::Deg180);
    assert_eq!(target.input().white_pick, Some([0.1, 0.2]));
    assert_eq!(target.input().lens, source.input().lens);
    assert_eq!(target.output().look, source.output().look);
    // The Object row is empty on the target; everything else of the graph is the source's.
    let mut expected = source.clone();
    expected.input_mut().crop = target.input().crop;
    expected.input_mut().orientation = target.input().orientation;
    expected.input_mut().white_pick = target.input().white_pick;
    for row in &mut expected.nodes.extra.get_mut(&2).unwrap().masks.items {
        *row = row.emptied();
    }
    assert_eq!(target, expected);
    // A paste of a patch that carries no links replaces nothing: it only changes what it names.
    let look_only = SettingsPatch::look_at(LookId::builtin("signature"), 60.0);
    assert!(look_only.pasted_onto(&target).remove.is_empty());
    // Apply to copies the Output's look and its intensity only.
    let mut other = rich();
    let before = other.clone();
    apply(&mut other, look_only);
    assert_eq!(other.output().look.as_str(), "builtin:signature");
    assert_eq!(other.output().look_intensity, 60.0);
    assert_eq!(other.input(), before.input());
    assert_eq!(other.nodes.extra, before.nodes.extra);
}

#[test]
fn a_white_balance_that_came_from_a_picked_white_is_copied_as_the_manual_kelvin_and_tint() {
    let mut s = DevelopSettings::default();
    s.input_mut().white_pick = Some([0.4, 0.4]);
    let resolved = resolved_of(&s);
    let patch = SettingsPatch::copied_from(&resolved);
    let input = patch.input_patch().unwrap();
    assert_eq!(
        input.white_balance,
        Some(AutoOr::Manual(ym_core::settings::WhiteBalanceManualPatch {
            kelvin: Some(resolved.values.input.kelvin),
            tint: Some(resolved.values.input.tint),
        }))
    );
    assert!(input.white_pick.is_none());
    // Auto stays Auto in the Input and in Main when no white was picked.
    let plain = SettingsPatch::copied_from(&resolved_of(&DevelopSettings::default()));
    let input = plain.input_patch().unwrap();
    assert_eq!(input.white_balance, Some(AutoOr::Auto));
    assert_eq!(input.exposure, Some(AutoOr::Auto));
    assert_eq!(input.sharpen, Some(AutoOr::Auto));
    assert_eq!(input.noise_reduction, Some(AutoOr::Auto));
    let grade = plain.main_patch().and_then(|p| p.grade.clone()).unwrap();
    assert_eq!(grade.contrast, Some(AutoOr::Auto));
    assert_eq!(grade.saturation, Some(AutoOr::Auto));
}

// ---------------------------------------------------------------------------------------------
// The form of Yellowmoon 0.1, resolved settings and the pinned document
// ---------------------------------------------------------------------------------------------

#[test]
fn the_form_of_yellowmoon_0_1_is_refused_with_its_sentence() {
    // The only test that holds a flat document (`docs/23` section 3.12).
    const SENTENCE: &str = "This settings file is in the old form of Yellowmoon 0.1. Settings are now a graph of nodes: `ym render --help` shows the new form, and `ym render --json` prints a photo's settings in it.";
    assert_eq!(ym_core::settings::OLD_FORM_SENTENCE, SENTENCE);
    assert_eq!(SettingsError::OldForm.to_string(), SENTENCE);
    for flat in [
        r#"{ "look": "builtin:neutral", "contrast": { "kind": "auto" } }"#,
        r#"{ "exposure": { "kind": "manual", "ev": 0.3 } }"#,
        r#"{ "white_offset_ev": 0.5 }"#,
        r#"{ "crop": null, "orientation": 90, "white_pick": null }"#,
    ] {
        assert_eq!(
            DevelopSettings::from_json(flat, &neutral()).unwrap_err(),
            SettingsError::OldForm,
            "{flat}"
        );
        // A flat patch is refused with the same sentence.
        let err = serde_json::from_str::<SettingsPatch>(flat).unwrap_err();
        assert!(err.to_string().contains(SENTENCE), "{flat}: {err}");
    }
    // A document that has `nodes` and another key is an unknown field, not the old form; and a key
    // that is not a field of the old form is a typo.
    for mixed in [
        r#"{ "nodes": {}, "contrast": 1 }"#,
        r#"{ "contrast": 1, "contrsat": 1 }"#,
    ] {
        let err = DevelopSettings::from_json(mixed, &neutral()).unwrap_err();
        assert!(
            matches!(err, SettingsError::Parse(ref m) if m.contains("unknown field")),
            "{mixed}: {err}"
        );
    }
    // An empty document is the default graph, not the old form.
    assert!(DevelopSettings::from_json("{}", &neutral()).is_ok());
}

#[test]
fn resolved_settings_are_node_first_and_agree_with_the_numbers_they_replace() {
    let (r, s) = rich_resolved();
    assert_eq!(
        r.settings, s,
        "the stored settings keep their Auto and everything else"
    );
    // `values`: the Input, Main, the other correctors and the Output with their Auto resolved.
    assert_eq!(r.values.input.lens, s.input().lens);
    assert_eq!(r.values.input.orientation, Orientation::Deg90);
    assert_eq!(
        r.values.input.exposure_ev, -1.25,
        "a manual exposure is its own number"
    );
    assert_eq!(r.values.main.grade.contrast, 12.0);
    assert_eq!(r.values.main.trim, s.main().trim);
    assert_eq!(r.values.main.id, 1);
    assert_eq!(
        r.values
            .nodes
            .iter()
            .map(|c| (c.id, c.name.as_str()))
            .collect::<Vec<_>>(),
        [(2, "Mountain"), (3, "Sky")]
    );
    assert_eq!(r.values.nodes[0].masks, s.nodes.corrector(2).unwrap().masks);
    assert_eq!(r.values.output.look, s.output().look);
    // `auto` says which Auto fields are still automatic, by node.
    assert!(!r.auto.input.exposure && !r.auto.input.light_source && !r.auto.input.white_balance);
    assert!(!r.auto.main.contrast && !r.auto.main.highlights && !r.auto.main.balance_shadows);
    let (d, _) = default_resolved();
    assert!(d.auto.input.exposure && d.auto.input.sharpen && d.auto.input.noise_reduction);
    assert!(d.auto.main.contrast && d.auto.main.black && d.auto.main.balance_highlights);
    // `estimates` groups the numbers by node and keeps what belongs to the photo beside them.
    let v = serde_json::to_value(&d).unwrap();
    let est = &v["estimates"];
    for key in [
        "exposure_ev",
        "kelvin",
        "tint",
        "sharpen",
        "noise_reduction",
    ] {
        assert!(est["input"].get(key).is_some(), "estimates.input.{key}");
    }
    for key in [
        "contrast",
        "saturation",
        "highlights",
        "shadows",
        "black",
        "balance_shadows",
        "balance_highlights",
    ] {
        assert!(est["main"].get(key).is_some(), "estimates.main.{key}");
    }
    for key in [
        "source",
        "provider",
        "apple",
        "apple_steps",
        "light_class",
        "light_confidence",
        "noise",
    ] {
        assert!(est.get(key).is_some(), "estimates.{key}");
    }
    assert_eq!(d.estimates.input.exposure_ev, d.values.input.exposure_ev);
    // The seeds that fill a partial manual value are the Input's numbers.
    assert_eq!(d.seeds().exposure_ev, d.values.input.exposure_ev);
}

#[test]
fn the_pinned_document_pins_the_inputs_and_mains_auto_and_reproduces_the_photo() {
    let mut s = DevelopSettings::default();
    s.nodes.extra.insert(2, CorrectorNode::default());
    s.nodes.links = chain(&[1, 2])
        .into_iter()
        .map(|l| serde_json::from_value(l).unwrap())
        .collect();
    let r = resolved_of(&s);
    let pinned = r.pinned_settings();
    // The Input's exposure, sharpening and noise reduction, and Main's contrast, saturation,
    // highlights, shadows, black and wheels are numbers now; the white balance stays Auto.
    let i = pinned.input();
    assert_eq!(
        i.exposure,
        ym_core::settings::ExposureSetting::manual_ev(r.values.input.exposure_ev)
    );
    assert!(!i.sharpen.is_auto() && !i.noise_reduction.is_auto() && i.white_balance.is_auto());
    assert!(i.light_source == LightSource::Auto);
    let g = pinned.main_grade();
    for setting in [g.contrast, g.saturation, g.highlights, g.shadows, g.black] {
        assert!(!setting.is_auto());
    }
    assert!(!g.color_balance.shadows.is_auto() && !g.color_balance.highlights.is_auto());
    // Nothing else is pinned: the other corrector, the Output and the links are as they were.
    assert_eq!(pinned.nodes.extra, s.nodes.extra);
    assert_eq!(pinned.output(), s.output());
    assert_eq!(pinned.nodes.links, s.nodes.links);
    // Feeding the document back (`--settings`) reproduces it, byte for byte, and pinning it again
    // changes nothing.
    let text = serde_json::to_string(&pinned).unwrap();
    let (back, clamped) = DevelopSettings::from_json(&text, &neutral()).unwrap();
    assert!(clamped.is_empty());
    assert_eq!(back, pinned);
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
    assert_eq!(resolved_of(&back).pinned_settings(), back);
}
