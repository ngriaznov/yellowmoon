//! The data files of the Before view (camera default rendering, `docs/02` section The Before view):
//! `data/cameras/color-matrices.json` and `data/cameras/camera-default-tone.json` validate against
//! their JSON schemas, the matrix table is what the embedded copy and the loader say, every body's
//! as-shot neutral renders neutral, and the tone file is pinned to the constants in `ym_color`.

mod common;

use common::{data_root, read_json, repo_root};
use jsonschema::Validator;
use serde_json::Value;
use ym_color::camera_default::{
    CameraDefault, CameraTone, TONE_K, TONE_N, TONE_S, TONE_SIGMA, TONE_WHITE,
};
use ym_color::consts::linalg;
use ym_color::encode::srgb_encode;
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_profiles::color_matrices::{ColorMatrices, EMBEDDED_JSON, MatrixMatch};
use ym_profiles::registry::Registry;

fn validator(schema_file: &str) -> Validator {
    let schema = read_json(
        &repo_root()
            .join("crates/ym-profiles/schema")
            .join(schema_file),
    );
    jsonschema::validator_for(&schema)
        .unwrap_or_else(|e| panic!("{schema_file} is not a valid schema: {e}"))
}

fn assert_valid(v: &Validator, instance: &Value, label: &str) {
    let errs: Vec<String> = v
        .iter_errors(instance)
        .take(8)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect();
    assert!(
        errs.is_empty(),
        "{label} does not validate:\n{}",
        errs.join("\n")
    );
}

fn matrices_file() -> Value {
    read_json(&data_root().join("cameras/color-matrices.json"))
}

fn tone_file() -> Value {
    read_json(&data_root().join("cameras/camera-default-tone.json"))
}

#[test]
fn both_files_validate_against_their_schemas() {
    assert_valid(
        &validator("color-matrices.schema.json"),
        &matrices_file(),
        "color-matrices.json",
    );
    assert_valid(
        &validator("camera-default-tone.schema.json"),
        &tone_file(),
        "camera-default-tone.json",
    );
}

#[test]
fn the_embedded_copy_is_the_file_and_it_loads() {
    let on_disk = std::fs::read_to_string(ColorMatrices::data_path(data_root())).unwrap();
    assert_eq!(EMBEDDED_JSON, on_disk);
    let t = ColorMatrices::from_json(&on_disk).unwrap();
    assert_eq!(t.bodies().len(), 44);
    assert_eq!(t.fallback().model, "ILCE-7RM4");
}

#[test]
fn the_matrix_table_agrees_with_the_registry_and_the_exclusions() {
    let table = ColorMatrices::shipped();
    let registry: Registry = {
        let text = std::fs::read_to_string(data_root().join("cameras/registry.json")).unwrap();
        Registry::parse(&text).unwrap()
    };
    let doc = matrices_file();
    let excluded: Vec<&str> = doc["excluded"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["model"].as_str().unwrap())
        .collect();
    assert_eq!(
        excluded,
        ["ILCE-7M5", "ILCE-7RM6"],
        "the two bodies left out: no matrix of their own and no sibling to borrow from"
    );
    // The A3000 is in the table (by decision) although the profile registry does not list it.
    assert!(!registry.bodies.iter().any(|b| b.model == "ILCE-3000"));
    assert_eq!(
        table.lookup("SONY", "ILCE-3000", Some(302)).how,
        MatrixMatch::ModelId
    );
    // Registry 1.3.0 added fifteen stand-in bodies for their white scale (`docs/03`). Nine of them have a published
    // matrix and an entry here; these six have none, so the Before view gives them the A7R IV's, as it did when
    // the registry did not list them.
    let without_entry = [
        "ILCE-6700",
        "ZV-E10",
        "DSC-RX100",
        "DSC-RX100M2",
        "DSC-RX100M7",
        "DSC-RX10M4",
    ];
    let mut missing: Vec<&str> = registry
        .bodies
        .iter()
        .filter(|b| table.bodies().iter().all(|t| t.model_id != b.model_id))
        .map(|b| b.model.as_str())
        .collect();
    missing.sort_unstable();
    let mut expected: Vec<&str> = excluded
        .iter()
        .chain(without_entry.iter())
        .copied()
        .collect();
    expected.sort_unstable();
    assert_eq!(
        missing, expected,
        "the registry bodies without a matrix of their own are the two excluded and the six of registry 1.3.0"
    );
    for body in &registry.bodies {
        let hit = table.lookup("SONY", &body.model, Some(body.model_id));
        if excluded.contains(&body.model.as_str()) || without_entry.contains(&body.model.as_str()) {
            // No matrix of its own: the A7R IV's.
            assert_eq!(hit.how, MatrixMatch::Fallback, "{}", body.model);
            assert_eq!(hit.body.model, "ILCE-7RM4");
        } else {
            // Every other registry body has its own entry, found by ID, and the model string agrees.
            assert_eq!(hit.how, MatrixMatch::ModelId, "{}", body.model);
            assert_eq!(hit.body.model, body.model);
            assert_eq!(hit.body.model_id, body.model_id);
            assert_eq!(
                table.lookup("SONY", &body.model, None).body.model,
                body.model
            );
        }
    }
    // A body that is not in the table at all uses the A7R IV matrix.
    let unknown = table.lookup("SONY", "ILCE-9999", Some(65000));
    assert_eq!(unknown.how, MatrixMatch::Fallback);
    assert_eq!(unknown.body, table.fallback());
}

/// Every entry of the matrix table as Sony publishes it: (model ID, Model string), in file order, which is
/// the order of the IDs. The IDs are facts about the cameras (maker-note tag 0xB001), pinned here from an
/// independent public list so that a guessed ID cannot slip in:
/// `the_matrix_table_agrees_with_the_registry_and_the_exclusions` walks the 43 registry bodies.
/// Sony lists no 402 and 403 in older versions of that list; the profile registry (`docs/03`) has them.
/// ID 302 is the A3000's (Sony writes it for the A3500 too, under the Model string ILCE-3500).
const PUBLISHED_IDS: [(u16, &str); 44] = [
    (261, "DSLR-A300"),
    (274, "DSLR-A500"),
    (288, "NEX-5N"),
    (289, "NEX-7"),
    (295, "NEX-6"),
    (296, "NEX-5R"),
    (302, "ILCE-3000"),
    (305, "NEX-3N"),
    (306, "ILCE-7"),
    (307, "NEX-5T"),
    (311, "ILCE-7R"),
    (312, "ILCE-6000"),
    (313, "ILCE-5000"),
    (318, "ILCE-7S"),
    (339, "ILCE-5100"),
    (340, "ILCE-7M2"),
    (341, "DSC-RX100M4"),
    (347, "ILCE-7RM2"),
    (350, "ILCE-7SM2"),
    (357, "ILCE-6300"),
    (358, "ILCE-9"),
    (360, "ILCE-6500"),
    (362, "ILCE-7RM3"),
    (363, "ILCE-7M3"),
    (371, "ILCE-6400"),
    (375, "ILCE-7RM4"),
    (376, "ILCE-9M2"),
    (378, "ILCE-6600"),
    (379, "ILCE-6100"),
    (381, "ILCE-7C"),
    (383, "ILCE-7SM3"),
    (384, "ILCE-1"),
    (385, "ILME-FX3"),
    (386, "ILCE-7RM3A"),
    (387, "ILCE-7RM4A"),
    (388, "ILCE-7M4"),
    (390, "ILCE-7RM5"),
    (392, "ILCE-9M3"),
    (393, "ZV-E1"),
    (396, "ILCE-7CR"),
    (397, "ILCE-7CM2"),
    (400, "ILCE-1M2"),
    (402, "ILCE-6400A"),
    (403, "ILCE-6100A"),
];

#[test]
fn every_entry_has_the_model_id_sony_publishes_for_its_body() {
    let table = ColorMatrices::shipped();
    let have: Vec<(u16, &str)> = table
        .bodies()
        .iter()
        .map(|b| (b.model_id, b.model.as_str()))
        .collect();
    assert_eq!(
        have,
        PUBLISHED_IDS.to_vec(),
        "an entry has an ID that belongs to another body"
    );
    for (id, model) in PUBLISHED_IDS {
        // The file's own ID finds the body, and so does its Model string when the file has no ID.
        let by_id = table.lookup("SONY", model, Some(id));
        assert_eq!(by_id.how, MatrixMatch::ModelId, "{model} ({id})");
        assert_eq!(by_id.body.model, model, "ID {id}");
        let by_model = table.lookup("SONY", model, None);
        assert_eq!(by_model.how, MatrixMatch::ModelString, "{model}");
        assert_eq!(by_model.body.model_id, id, "{model}");
    }
    // The first draft guessed these IDs. 291 is the SLT-A37 and 300 the NEX-VG30E; 315 and 368 are
    // nobody's. A real file with one of them must not find an entry.
    for id in [291, 300, 315, 368] {
        let l = table.lookup("SONY", "ILCE-7", Some(id));
        assert_eq!(l.how, MatrixMatch::Fallback, "ID {id}");
        assert_eq!(l.body.model, "ILCE-7RM4", "ID {id}");
    }
    // The IDs those guesses were meant for find their own bodies, not the A7R IV's.
    for (id, model) in [
        (295, "NEX-6"),
        (307, "NEX-5T"),
        (318, "ILCE-7S"),
        (360, "ILCE-6500"),
    ] {
        let l = table.lookup("SONY", model, Some(id));
        assert_eq!(
            (l.how, l.body.model.as_str()),
            (MatrixMatch::ModelId, model)
        );
    }
}

#[test]
fn the_matrices_of_the_bodies_once_keyed_by_a_neighbouring_id_are_their_own() {
    // The numbers belong to the ID the camera writes, so when the draft's IDs were wrong the numbers
    // sat under the wrong names: the ILCE-7's under ILCE-6000, the ILCE-7R's under ILCE-7, the
    // ILCE-6000's under ILCE-7R and the SLT-A37's under NEX-6. Pinned here, rounded as published.
    let table = ColorMatrices::shipped();
    let pinned: [(u16, [[f64; 3]; 3]); 4] = [
        (
            295, // NEX-6
            [
                [0.6129, -0.1545, -0.0418],
                [-0.493, 1.249, 0.2743],
                [-0.0977, 0.1693, 0.6615],
            ],
        ),
        (
            306, // ILCE-7
            [
                [0.5271, -0.0712, -0.0347],
                [-0.6153, 1.3653, 0.2763],
                [-0.1601, 0.2366, 0.7242],
            ],
        ),
        (
            311, // ILCE-7R
            [
                [0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183],
            ],
        ),
        (
            312, // ILCE-6000
            [
                [0.5991, -0.1456, -0.0455],
                [-0.4764, 1.2135, 0.298],
                [-0.0707, 0.1425, 0.6701],
            ],
        ),
    ];
    for (id, want) in pinned {
        let body = table.lookup("SONY", "", Some(id));
        assert_eq!(body.how, MatrixMatch::ModelId, "ID {id}");
        assert_eq!(
            body.body.xyz_d65_to_camera, want,
            "{} ({id})",
            body.body.model
        );
    }
}

#[test]
fn the_two_borrowed_entries_carry_the_numbers_they_borrow_and_say_so() {
    // Owner's decision 2026-10-16 (`docs/STATUS.md`, question 29): the A1 II takes the A1's numbers (same
    // sensor), the A3000 the generic E-mount numbers (the NEX-5N's). Every other entry is a body's own.
    let table = ColorMatrices::shipped();
    let numbers = |id: u16| table.lookup("SONY", "", Some(id)).body.xyz_d65_to_camera;
    assert_eq!(numbers(400), numbers(384), "ILCE-1M2 = ILCE-1");
    assert_eq!(numbers(302), numbers(288), "ILCE-3000 = NEX-5N");
    let doc = matrices_file();
    let noted: Vec<(&str, &str)> = doc["bodies"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|b| Some((b["model"].as_str()?, b.get("note")?.as_str()?)))
        .collect();
    assert_eq!(
        noted.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
        ["ILCE-3000", "ILCE-1M2"],
        "only the two borrowed entries carry a note"
    );
    for (model, note) in noted {
        assert!(note.contains("2026-10-16"), "{model}: {note}");
    }
}

#[test]
fn an_as_shot_neutral_renders_neutral_for_every_body_in_the_table() {
    let table = ColorMatrices::shipped();
    // Several as-shot whites (daylight-like, tungsten-like, strongly green), each with initial gains
    // that differ from them, as they do when our estimate is not the camera's.
    let whites = [[2.05, 1.0, 1.48], [1.2, 1.0, 2.6], [2.9, 1.0, 1.05]];
    let initial = [[1.9, 1.0, 1.6], [1.0, 1.0, 1.0], [2.2, 1.0, 1.9]];
    let gain = CAMERA_GAIN_EV.exp2();
    for body in table.bodies() {
        for (gs, g0) in whites.iter().zip(&initial) {
            let cd = CameraDefault::new(&body.xyz_d65_to_camera, *gs, *g0).unwrap();
            // A white surface records 1 / gs in the raw data; the raw stages multiply by g0.
            let cam_in = [g0[0] / gs[0], g0[1] / gs[1], g0[2] / gs[2]];
            let lin = cd.scene_linear_rec709(cam_in);
            for c in lin {
                assert!((c - gain).abs() < 1e-12, "{}: {lin:?}", body.model);
            }
            // Any neutral level of that white stays neutral through the curve and the encode.
            for level in [0.004, 0.03, 0.18, 0.5, 0.9] {
                let out = cd.display_linear_rec709(cam_in.map(|c| c * level));
                assert!(
                    (out[0] - out[1]).abs() < 1e-12 && (out[1] - out[2]).abs() < 1e-12,
                    "{} level {level}: {out:?}",
                    body.model
                );
                let codes = out.map(|y| (srgb_encode(y) * 255.0).round());
                assert!(codes[0] == codes[1] && codes[1] == codes[2], "{codes:?}");
            }
            // Middle grey after the gain lands on code 119.7.
            let grey = cd.display_linear_rec709(cam_in.map(|c| c * 0.18 / gain));
            let code = srgb_encode(grey[1]) * 255.0;
            assert!((code - 119.7).abs() < 0.5, "{}: {code}", body.model);
        }
    }
}

#[test]
fn every_matrix_has_the_shape_of_a_colour_matrix() {
    for body in ColorMatrices::shipped().bodies() {
        let m = &body.xyz_d65_to_camera;
        // Normalised, the rows map D65 white to camera 1 and the inverse is a real, well-conditioned
        // camera-to-XYZ matrix: positive determinant, no huge entries.
        assert!(
            linalg::det(m) > 0.0,
            "{}: det {}",
            body.model,
            linalg::det(m)
        );
        let cd = CameraDefault::new(m, [1.0; 3], [1.0; 3]).unwrap();
        for v in cd.matrix().iter().flatten() {
            assert!(v.abs() < 40.0, "{}: {v}", body.model);
        }
    }
}

#[test]
fn the_tone_file_is_pinned_to_the_constants_in_code() {
    let doc = tone_file();
    let curve = &doc["curve"];
    assert_eq!(curve["n"].as_f64().unwrap(), TONE_N);
    assert_eq!(curve["sigma"].as_f64().unwrap(), TONE_SIGMA);
    assert_eq!(curve["white"].as_f64().unwrap(), TONE_WHITE);
    assert_eq!(curve["s"].as_f64().unwrap(), TONE_S);
    assert_eq!(curve["k"].as_f64().unwrap(), TONE_K);
    // The constants in code are the formula of the file.
    let t = CameraTone::new(TONE_N, TONE_SIGMA, TONE_WHITE).unwrap();
    assert!((t.s() - TONE_S).abs() < 1e-15);
    assert!((t.k() - TONE_K).abs() < 1e-14);
    assert_eq!(doc["anchors"]["white_x"].as_f64().unwrap(), TONE_WHITE);
    let anchor = &doc["anchors"];
    let y = CameraTone::FITTED.apply(anchor["middle_grey_x"].as_f64().unwrap());
    assert!((y - anchor["middle_grey_y"].as_f64().unwrap()).abs() < 1e-7);
    let code = srgb_encode(y) * 255.0;
    assert!((code - anchor["middle_grey_srgb_code"].as_f64().unwrap()).abs() < 0.05);
    assert_eq!(doc["fit"]["photos"], 12);
    assert_eq!(doc["fit"]["bodies"], 10);
}

#[test]
fn the_golden_table_is_the_curve() {
    let doc = tone_file();
    let points = doc["table"]["points_x_y"].as_array().unwrap();
    assert_eq!(points.len(), 28);
    let mut last_y = -1.0;
    for p in points {
        let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        let got = CameraTone::FITTED.apply(x);
        // The table is stored to 8 significant digits.
        assert!(
            (got - y).abs() <= 1e-7 * y.max(1e-3) + 1e-9,
            "x {x}: {got} vs {y}"
        );
        assert!(y > last_y, "the table is increasing");
        last_y = y;
    }
    let (last_x, last_y) = (
        points.last().unwrap()[0].as_f64().unwrap(),
        points.last().unwrap()[1].as_f64().unwrap(),
    );
    assert_eq!((last_x, last_y), (TONE_WHITE, 1.0));
}

#[test]
fn the_files_carry_only_numbers_and_body_names() {
    // docs/14 rule 8 and the owner's files: no file names, no owner photo names, and no mention of
    // the tools or projects the numbers were read with. The notes of the two borrowed entries name the
    // owner's decision, so the bare word is allowed; the owner's files and photos are not.
    for (name, text) in [
        ("color-matrices.json", EMBEDDED_JSON.to_string()),
        (
            "camera-default-tone.json",
            std::fs::read_to_string(data_root().join("cameras/camera-default-tone.json")).unwrap(),
        ),
    ] {
        let lower = text.to_ascii_lowercase();
        for banned in [
            "libraw",
            "rawpy",
            "dcraw",
            "darktable",
            "rawtherapee",
            ".arw",
            ".jpg",
            ".jpeg",
            "owner's file",
            "owner's photo",
            "owner photo",
            "owner file",
            "dsc0",
            "a1m0",
        ] {
            assert!(!lower.contains(banned), "{name} mentions {banned:?}");
        }
    }
}
