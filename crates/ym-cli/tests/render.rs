//! End-to-end tests of `ym render` on synthetic ARW files: inputs, options, `--json`, settings
//! files, exit codes and Ctrl-C.

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{Sandbox, head, json_lines, lines, names_in, shape, text};
use predicates::prelude::*;
use serde_json::{Value, json};

#[test]
fn one_file_goes_to_the_folder_next_to_its_folder_and_its_path_is_on_stdout() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb.ym().arg("render").arg(&arw).assert().success();
    let expected = sb.default_export().join("DSC00001_signature.jpg");
    assert_eq!(
        lines(&out.get_output().stdout),
        [expected.display().to_string()]
    );
    assert_eq!(head(&expected, 2), [0xFF, 0xD8], "a JPEG");
    let err = text(&out.get_output().stderr);
    assert!(err.contains("Exported 1 photo"), "{err}");
    assert!(
        err.contains("Note: No trained demosaic model"),
        "the classical path is said once: {err}"
    );
    // The original is untouched and nothing else was written next to it.
    assert_eq!(names_in(&sb.photos()), ["DSC00001.ARW"]);
    assert_eq!(names_in(&sb.default_export()), ["DSC00001_signature.jpg"]);
}

#[test]
fn a_folder_is_walked_in_name_order_and_recursion_is_asked_for() {
    let sb = Sandbox::new();
    sb.arw("b.ARW");
    sb.arw("A.arw");
    sb.arw("sub/c.ARW");
    std::fs::write(sb.photos().join("notes.txt"), "x").expect("write");
    std::fs::write(sb.photos().join("._b.ARW"), "x").expect("write");

    let out = sb.ym().arg("render").arg(sb.photos()).assert().success();
    let names: Vec<String> = lines(&out.get_output().stdout)
        .iter()
        .map(|l| l.rsplit('/').next().unwrap_or(l).to_owned())
        .collect();
    assert_eq!(names, ["A_signature.jpg", "b_signature.jpg"]);

    let sb2 = sb;
    let out = sb2
        .ym()
        .args(["render", "--recursive", "--overwrite"])
        .arg(sb2.photos())
        .assert()
        .success();
    assert_eq!(
        lines(&out.get_output().stdout).len(),
        3,
        "the sub folder too"
    );
}

#[test]
fn paths_are_read_from_stdin_with_a_dash() {
    let sb = Sandbox::new();
    let a = sb.arw("a.ARW");
    let b = sb.arw("b.ARW");
    let out = sb
        .ym()
        .args(["render", "-", "--json"])
        .write_stdin(format!("{}\n\n{}\r\n", a.display(), b.display()))
        .assert()
        .success();
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 2);
    let mut inputs: Vec<String> = objects
        .iter()
        .map(|o| o["input"].as_str().expect("input").to_owned())
        .collect();
    inputs.sort();
    assert_eq!(inputs, [a.display().to_string(), b.display().to_string()]);
}

#[test]
fn json_has_one_object_per_file_in_the_documented_shape_and_order() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb
        .ym()
        .args(["render", "--json", "--format", "png", "--bits", "8"])
        .arg(&arw)
        .assert()
        .success();
    assert_eq!(
        text(&out.get_output().stderr)
            .lines()
            .filter(|l| !l.starts_with("Warning"))
            .count(),
        0,
        "JSON mode keeps stderr to warnings: {}",
        text(&out.get_output().stderr)
    );
    let raw = text(&out.get_output().stdout);
    let line = raw.trim_end();
    assert_eq!(line.lines().count(), 1);
    let at = |key: &str| line.find(&format!("\"{key}\":")).expect(key);
    let order = [
        "input",
        "output",
        "status",
        "camera",
        "look",
        "auto",
        "settings",
        "timings_ms",
        "notes",
        "pipeline_version",
        "error",
    ];
    for pair in order.windows(2) {
        assert!(at(pair[0]) < at(pair[1]), "{} before {}", pair[0], pair[1]);
    }
    let v: Value = serde_json::from_str(line).expect("JSON");
    assert_eq!(v["input"], arw.display().to_string());
    assert_eq!(
        v["output"],
        sb.default_export()
            .join("DSC00001_signature.png")
            .display()
            .to_string()
    );
    assert_eq!(v["status"], "ok");
    assert_eq!(v["error"], Value::Null);
    assert_eq!(v["camera"]["body"], "Sony A7R IV");
    assert_eq!(v["camera"]["model_id"], 375);
    assert_eq!(v["camera"]["profile"], "sony-ilce-7rm4");
    assert_eq!(v["camera"]["profile_version"], "1.2.0");
    assert_eq!(v["camera"]["match"], "exact");
    assert_eq!(
        v["look"],
        json!({ "id": "builtin:signature", "intensity": 100 })
    );
    // The numbers Auto resolves to are grouped by the node they are for (`docs/23` section 3.9).
    for key in [
        "exposure_ev",
        "kelvin",
        "tint",
        "sharpen",
        "noise_reduction",
    ] {
        assert!(!v["auto"]["input"][key].is_null(), "auto.input.{key}");
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
        assert!(!v["auto"]["main"][key].is_null(), "auto.main.{key}");
    }
    for key in ["light_class", "light_confidence"] {
        assert!(!v["auto"][key].is_null(), "auto.{key}");
    }
    assert!(v["auto"]["noise"]["a"].is_number() && v["auto"]["noise"]["b"].is_number());
    for key in ["decode", "develop", "encode"] {
        assert!(v["timings_ms"][key].is_u64(), "timings_ms.{key}");
    }
    assert!(v["pipeline_version"].is_string());
    // The settings are what the file's `ym:Settings` says: Auto exposure written as its number.
    assert_eq!(common::node(&v, "input")["exposure"]["kind"], "manual");
    assert_eq!(
        common::node(&v, "input")["exposure"]["ev"],
        v["auto"]["input"]["exposure_ev"]
    );
    assert_eq!(common::node(&v, "input")["white_balance"]["kind"], "auto");
    assert_eq!(common::node(&v, "output")["look"], "builtin:signature");
}

#[test]
fn a_failed_file_has_status_error_no_output_and_the_exit_code_is_1() {
    let sb = Sandbox::new();
    sb.arw("good.ARW");
    std::fs::write(sb.photos().join("bad.ARW"), "this is not a raw file").expect("write");
    let out = sb
        .ym()
        .args(["render", "--json"])
        .arg(sb.photos())
        .assert()
        .code(1);
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 2);
    let bad = objects
        .iter()
        .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with("bad.ARW")))
        .expect("bad");
    assert_eq!(bad["status"], "error");
    assert_eq!(bad["output"], Value::Null);
    assert_eq!(bad["error"]["kind"], "not_arw");
    assert_eq!(
        bad["error"]["message"],
        "This isn't an ARW file. Yellowmoon opens Sony ARW files only."
    );
    let good = objects
        .iter()
        .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with("good.ARW")))
        .expect("good");
    assert_eq!(good["status"], "ok");
    assert!(
        sb.default_export().join("good_signature.jpg").exists(),
        "the good file is still exported"
    );
}

#[test]
fn text_mode_reports_a_failed_file_on_stderr_and_still_exports_the_rest() {
    let sb = Sandbox::new();
    sb.arw("good.ARW");
    std::fs::write(sb.photos().join("bad.ARW"), "x").expect("write");
    sb.ym()
        .arg("render")
        .arg(sb.photos())
        .assert()
        .code(1)
        .stdout(predicate::str::contains("good_signature.jpg"))
        .stderr(predicate::str::contains("bad.ARW: This isn't an ARW file."))
        .stderr(predicate::str::contains("1 photo failed."));
}

#[test]
fn every_format_writes_its_own_file() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let cases: [(&[&str], &str, &[u8]); 4] = [
        (&["jpeg"], "DSC00001_signature.jpg", &[0xFF, 0xD8]),
        (&["tiff"], "DSC00001_signature.tif", b"II*\0"),
        (
            &["png"],
            "DSC00001_signature.png",
            &[0x89, b'P', b'N', b'G'],
        ),
        (&["dng"], "DSC00001_signature.dng", b"II*\0"),
    ];
    for (format, file, magic) in cases {
        let out = sb.root().join(format[0]);
        sb.ym()
            .args(["render", "--format", format[0], "--out"])
            .arg(&out)
            .arg(&arw)
            .assert()
            .success();
        assert_eq!(names_in(&out), [file], "{}", format[0]);
        assert_eq!(head(&out.join(file), magic.len()), magic, "{}", format[0]);
    }
}

#[test]
fn out_name_and_the_if_exists_options_decide_the_file_names() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb.root().join("exports");
    let run = |extra: &[&str]| {
        sb.ym()
            .args(["render", "--name", "{name}-{seq}", "--out"])
            .arg(&out)
            .args(extra)
            .arg(&arw)
            .assert()
    };
    run(&[]).success();
    assert_eq!(names_in(&out), ["DSC00001-001.jpg"]);
    // Keep both is the default.
    run(&[]).success();
    assert_eq!(names_in(&out), ["DSC00001-001 2.jpg", "DSC00001-001.jpg"]);
    // Skip leaves the file alone, says nothing on stdout and reports it in the JSON.
    let skipped = run(&["--skip-existing", "--json"]).success();
    let objects = json_lines(&skipped.get_output().stdout);
    assert_eq!(objects[0]["status"], "skipped");
    assert_eq!(
        objects[0]["output"],
        out.join("DSC00001-001.jpg").display().to_string()
    );
    assert_eq!(names_in(&out).len(), 2);
    let quiet = run(&["--skip-existing"]).success();
    assert!(quiet.get_output().stdout.is_empty());
    // The summary agrees with the number: one photo existed and was skipped.
    let said = text(&quiet.get_output().stderr);
    assert!(
        said.contains("1 photo already existed and was skipped."),
        "{said}"
    );
    // Replace writes over the first file and adds nothing.
    let before = std::fs::metadata(out.join("DSC00001-001.jpg"))
        .expect("meta")
        .len();
    run(&["--overwrite", "--quality", "60"]).success();
    assert_eq!(names_in(&out).len(), 2);
    let after = std::fs::metadata(out.join("DSC00001-001.jpg"))
        .expect("meta")
        .len();
    assert!(after < before, "quality 60 is smaller: {after} < {before}");
}

#[test]
fn two_photos_that_map_to_one_name_are_both_written_whatever_the_exists_option_says() {
    let sb = Sandbox::new();
    let a = sb.arw_sized("card1/DSC00001.ARW", 160, 120, 11);
    let b = sb.arw_sized("card2/DSC00001.ARW", 160, 120, 12);
    // `--overwrite` and `--skip-existing` look at the disk; the file the first photo wrote a moment
    // ago is not "a file that existed", and must not be replaced or taken for one.
    for (n, flag) in ["--overwrite", "--skip-existing", "--keep-both"]
        .into_iter()
        .enumerate()
    {
        let out = sb.root().join(format!("out{n}"));
        let mut cmd = sb.ym();
        cmd.args(["render", "--json", "--format", "jpeg", "--out"])
            .arg(&out);
        if flag != "--keep-both" {
            cmd.arg(flag);
        }
        let run = cmd.arg(&a).arg(&b).assert().success();
        let objects = json_lines(&run.get_output().stdout);
        let statuses: Vec<&str> = objects
            .iter()
            .map(|o| o["status"].as_str().expect("status"))
            .collect();
        assert_eq!(statuses, ["ok", "ok"], "{flag}");
        let outputs: Vec<&str> = objects
            .iter()
            .map(|o| o["output"].as_str().expect("output"))
            .collect();
        assert_ne!(outputs[0], outputs[1], "{flag}: two files, not one twice");
        // The run says why the second photo has a number, once for the one photo; with "keep both"
        // numbering is what was asked for and there is nothing to explain.
        let noted: Vec<bool> = objects
            .iter()
            .map(|o| {
                o["notes"].as_array().expect("notes").iter().any(|n| {
                    n.as_str()
                        .is_some_and(|n| n.contains("has the same file name"))
                })
            })
            .collect();
        let expected: [bool; 2] = if flag == "--keep-both" {
            [false, false]
        } else {
            [false, true]
        };
        assert_eq!(noted, expected, "{flag}");
        assert_eq!(
            names_in(&out),
            ["DSC00001_signature 2.jpg", "DSC00001_signature.jpg"],
            "{flag}"
        );
        let said = text(&run.get_output().stderr);
        assert!(!said.contains("skipped"), "{flag}: {said}");
    }
    // A name template that two photos of a day share (no {name}, no {seq}) is the same case.
    let out = sb.root().join("by_date");
    let run = sb
        .ym()
        .args([
            "render",
            "--json",
            "--overwrite",
            "--name",
            "{date}",
            "--out",
        ])
        .arg(&out)
        .arg(&a)
        .arg(&b)
        .assert()
        .success();
    assert_eq!(json_lines(&run.get_output().stdout).len(), 2);
    assert_eq!(names_in(&out).len(), 2, "{:?}", names_in(&out));
}

#[test]
fn the_long_edge_scales_the_picture() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 320, 240, 3);
    let out = sb.root().join("small");
    sb.ym()
        .args(["render", "--format", "png", "--long-edge", "100", "--out"])
        .arg(&out)
        .arg(&arw)
        .assert()
        .success();
    let bytes = std::fs::read(out.join("big_signature.png")).expect("png");
    // IHDR: width and height are the first two big-endian words after the signature and chunk head.
    let w = u32::from_be_bytes(bytes[16..20].try_into().expect("w"));
    let h = u32::from_be_bytes(bytes[20..24].try_into().expect("h"));
    assert_eq!(w, 100);
    assert!(
        h < 100 && h > 60,
        "a 4:3 frame shrinks to about 100 x 75, not {h}"
    );
}

#[test]
fn flags_become_settings_and_the_exposure_offset_is_relative_to_each_photos_auto() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw_with_white("b.ARW", 4000);
    let out = sb
        .ym()
        .args([
            "render",
            "--json",
            "--exposure-offset",
            "-0.5",
            "--contrast",
            "20",
            "--saturation",
            "-15",
            "--wb",
            "5200,3",
            "--light",
            "day",
            "--sharpen",
            "120,1.1",
            "--nr",
            "10,20,30",
            "--no-ca",
            "--rolloff",
            "0.3,0.7",
            "--intensity",
            "80",
        ])
        .arg(sb.photos())
        .assert()
        .success();
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 2);
    for o in &objects {
        let auto_ev = o["auto"]["input"]["exposure_ev"].as_f64().expect("auto ev");
        let (input, grade) = (common::node(o, "input"), &common::node(o, 1)["grade"]);
        let ev = input["exposure"]["ev"].as_f64().expect("exposure");
        assert!(
            (ev - (auto_ev - 0.5)).abs() < 1e-9,
            "{ev} vs {auto_ev} - 0.5"
        );
        // Contrast and saturation are Auto or manual (docs/19 section 5): a number is manual.
        assert_eq!(
            grade["contrast"],
            json!({ "kind": "manual", "value": 20.0 })
        );
        assert_eq!(
            grade["saturation"],
            json!({ "kind": "manual", "value": -15.0 })
        );
        assert_eq!(
            input["white_balance"],
            json!({ "kind": "manual", "kelvin": 5200.0, "tint": 3.0 })
        );
        assert_eq!(input["light_source"], "day");
        assert_eq!(
            input["sharpen"],
            json!({ "kind": "manual", "amount": 120.0, "radius_px": 1.1 })
        );
        assert_eq!(
            input["noise_reduction"],
            json!({ "kind": "manual", "luminance": 10.0, "color": 20.0, "detail": 30.0 })
        );
        assert_eq!(input["decode"]["chromatic_aberration"], false);
        assert_eq!(input["decode"]["neural_demosaic"], true);
        assert_eq!(
            input["rolloff"],
            json!({ "highlights": 0.3, "low_light": 0.7 })
        );
        assert_eq!(o["look"]["intensity"], 80);
    }
    // Auto exposure is the camera's gain whatever the photo (docs/02, interim rule of 2026-10-14), so the two photos
    // have one automatic exposure and the offset lands on one number: the offset is relative to it, not absolute.
    let evs: Vec<f64> = objects
        .iter()
        .map(|o| {
            common::node(o, "input")["exposure"]["ev"]
                .as_f64()
                .expect("ev")
        })
        .collect();
    let want = ym_color::exposure::CAMERA_GAIN_EV - 0.5;
    for ev in &evs {
        assert!((ev - want).abs() < 1e-9, "{evs:?}, want {want}");
    }
}

#[test]
fn the_remaining_flags_of_docs_10_reach_the_settings_of_the_file() {
    // The flags above are checked in the test before; these seven are the rest of the table.
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args([
            "render",
            "--json",
            "--brightness",
            "11",
            "--shadows",
            "22",
            "--white",
            "33",
            "--no-neural",
            "--no-highlight-recovery",
            "--no-lens",
        ])
        .arg(&arw)
        .assert()
        .success();
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 1);
    let (input, grade) = (
        common::node(&objects[0], "input"),
        &common::node(&objects[0], 1)["grade"],
    );
    assert_eq!(grade["brightness"], 11.0);
    // Shadows is Auto or a manual value (`docs/19` section 5): a number is Manual.
    assert_eq!(grade["shadows"], json!({ "kind": "manual", "value": 22.0 }));
    assert_eq!(grade["white"], 33.0);
    assert_eq!(
        input["decode"],
        json!({
            "neural_demosaic": false,
            "highlight_recovery": false,
            "lens_corrections": false,
            "chromatic_aberration": true
        })
    );
    // Untouched, they are what the photo opens with: Auto highlights, black, contrast and
    // saturation are pinned to the 0 they resolve to without Apple Auto, and the shadows and
    // highlights wheels to the centre (docs/19 section 5).
    for key in ["highlights", "black", "contrast", "saturation"] {
        assert_eq!(
            grade[key],
            json!({ "kind": "manual", "value": 0.0 }),
            "{key}"
        );
    }
    for key in ["shadows", "highlights"] {
        assert_eq!(
            grade["color_balance"][key],
            json!({ "kind": "manual", "point": [0.0, 0.0] }),
            "{key}"
        );
    }
}

#[test]
fn an_absolute_exposure_and_a_kelvin_without_a_tint() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["render", "--json", "--exposure", "-1.25", "--wb", "6100"])
        .arg(&arw)
        .assert()
        .success();
    let o = &json_lines(&out.get_output().stdout)[0];
    assert_eq!(
        common::node(o, "input")["exposure"],
        json!({ "kind": "manual", "ev": -1.25 })
    );
    // The tint keeps the automatic value.
    assert_eq!(common::node(o, "input")["white_balance"]["kelvin"], 6100.0);
    let tint = common::node(o, "input")["white_balance"]["tint"]
        .as_f64()
        .expect("tint");
    assert!((tint - o["auto"]["input"]["tint"].as_f64().expect("auto tint")).abs() < 1e-9);
}

#[test]
fn a_settings_file_applies_to_every_photo_and_flags_override_it() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw("b.ARW");
    let file = sb.root().join("settings.json");
    std::fs::write(
        &file,
        r#"{ "nodes": { "items": [
            { "id": "input", "exposure": {"kind": "manual", "ev": 0.25} },
            { "id": 1, "grade": { "contrast": {"kind": "manual", "value": 30},
                                  "saturation": {"kind": "manual", "value": 10} } } ] } }"#,
    )
    .expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--saturation", "-20", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .success();
    for o in json_lines(&out.get_output().stdout) {
        assert_eq!(common::node(&o, 1)["grade"]["contrast"]["value"], 30.0);
        assert_eq!(
            common::node(&o, 1)["grade"]["saturation"]["value"],
            -20.0,
            "the flag wins"
        );
        assert_eq!(common::node(&o, "input")["exposure"]["ev"], 0.25);
    }
}

#[test]
fn a_settings_object_keyed_by_file_name_gives_each_photo_its_own() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw("b.ARW");
    sb.arw("c.ARW");
    let file = sb.root().join("per-file.json");
    std::fs::write(
        &file,
        r#"{ "a.ARW": { "nodes": { "items": [ { "id": 1, "grade": { "contrast": {"kind": "manual", "value": 40} } } ] } },
            "B.arw": { "nodes": { "items": [ { "id": 1, "grade": { "contrast": {"kind": "manual", "value": -40} } } ] } },
            "z.ARW": { "nodes": { "items": [ { "id": 1, "grade": { "contrast": {"kind": "manual", "value": 1} } } ] } } }"#,
    )
    .expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "z.ARW, which is not one of the photos",
        ));
    let objects = json_lines(&out.get_output().stdout);
    let contrast = |name: &str| {
        objects
            .iter()
            .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with(name)))
            .map(|o| {
                common::node(o, 1)["grade"]["contrast"]["value"]
                    .as_f64()
                    .expect("contrast")
            })
            .expect(name)
    };
    assert_eq!(contrast("a.ARW"), 40.0);
    assert_eq!(
        contrast("b.ARW"),
        -40.0,
        "names match without regard to case"
    );
    assert_eq!(
        contrast("c.ARW"),
        0.0,
        "a photo with no entry is developed automatically"
    );
}

/// A settings file with one corrector, node 2, with these mask rows.
fn masked_file(rows: &str) -> String {
    format!(
        r#"{{ "nodes": {{ "items": [ {{ "id": 2, "kind": "corrector", "pos": [1, 0], "trim": {{ "exposure_ev": 0.7 }},
            "masks": {{ "invert": false, "items": [ {rows} ] }} }} ],
            "links": [
              {{ "from": {{ "node": "input", "port": "picture" }}, "to": {{ "node": 1, "port": "picture" }} }},
              {{ "from": {{ "node": 1, "port": "picture" }}, "to": {{ "node": 2, "port": "picture" }} }},
              {{ "from": {{ "node": 2, "port": "picture" }}, "to": {{ "node": "output", "port": "picture" }} }}
            ] }} }}"#
    )
}

/// A raster mask this computer cannot make (`docs/10` *Settings*, `docs/23` section 3.10) stops that photo with
/// `error.kind` `mask`, a sentence that names the node and the kind, and exit 1; the other photos go on and the
/// photo is never written without its mask.
#[test]
fn a_photo_that_needs_a_mask_this_computer_cannot_make_fails_with_the_mask_error_and_the_rest_go_on()
 {
    let sb = Sandbox::new();
    sb.arw("a_plain.ARW");
    sb.arw("b_subject.ARW");
    sb.arw("c_people.ARW");
    sb.arw("d_background.ARW");
    sb.arw("e_off.ARW");
    let file = sb.root().join("per-file.json");
    let doc = |rows: &str| serde_json::from_str::<Value>(&masked_file(rows)).expect("a document");
    let doc_text = serde_json::json!({
        "b_subject.ARW": doc(r#"{ "kind": "subject" }"#),
        "c_people.ARW": doc(r#"{ "kind": "people" }"#),
        "d_background.ARW": doc(r#"{ "kind": "background" }"#),
        // The eye is off: the row is absent, the photo renders.
        "e_off.ARW": doc(r#"{ "kind": "subject", "enabled": false }"#),
    });
    std::fs::write(&file, doc_text.to_string()).expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .code(1);
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 5);
    let find = |name: &str| {
        objects
            .iter()
            .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with(name)))
            .unwrap_or_else(|| panic!("no record for {name}"))
    };
    // The Object row's own sentence, which depends on the weights, is `an_object_row_without_the_model_is_the_mask_error_and_with_it_the_photo_renders`.
    // Subject and People: the sentences of `docs/23` section 3.10 on every computer (the sandbox's `ym` says Vision is not supported).
    for (name, sentence) in [
        (
            "b_subject.ARW",
            "Node 2 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask.",
        ),
        (
            "c_people.ARW",
            "Node 2 has a People mask, and this computer cannot make one. Use a Mac that supports People masks, or remove the mask.",
        ),
    ] {
        let o = find(name);
        assert_eq!(o["status"], "error", "{name}");
        assert_eq!(o["output"], Value::Null, "{name}");
        assert_eq!(o["error"]["kind"], "mask", "{name}");
        assert_eq!(o["error"]["message"], sentence, "{name}");
        // The record still says what the photo was.
        assert!(
            o["camera"].is_object() && o["settings"].is_object(),
            "{name}"
        );
        assert!(
            !sb.default_export()
                .join(format!("{}_signature.jpg", name.trim_end_matches(".ARW")))
                .exists(),
            "{name} was written without its mask"
        );
    }
    assert_eq!(find("a_plain.ARW")["status"], "ok");
    assert_eq!(find("e_off.ARW")["status"], "ok");
    // A Background row is made from the photo's own blur map (N-15): it renders.
    assert_eq!(find("d_background.ARW")["status"], "ok");
    assert!(sb.default_export().join("a_plain_signature.jpg").exists());
    assert!(sb.default_export().join("e_off_signature.jpg").exists());
    assert!(
        sb.default_export()
            .join("d_background_signature.jpg")
            .exists()
    );
}

/// The same settings file on a computer that supports Subject and People (the fake Vision: `YELLOWMOON_TEST_VISION=fake`): both
/// photos render, the rows act (the picture is not the one of the node without them), and the report says what the settings were.
/// The same file twice gives the same bytes (`docs/23` section 5.10).
#[test]
fn subject_and_people_rows_render_where_this_computer_supports_them_and_act() {
    let sb = Sandbox::new();
    sb.arw("a_subject.ARW");
    sb.arw("b_people.ARW");
    sb.arw("c_none.ARW");
    let doc = |rows: &str| serde_json::from_str::<Value>(&masked_file(rows)).expect("a document");
    let file = sb.root().join("per-file.json");
    let doc_text = serde_json::json!({
        "a_subject.ARW": doc(r#"{ "kind": "subject", "feather": 0.2 }"#),
        "b_people.ARW": doc(r#"{ "kind": "people" }"#),
        "c_none.ARW": doc(""),
    });
    std::fs::write(&file, doc_text.to_string()).expect("write");
    let run = |out: &str| {
        let out = sb
            .ym_fake_vision()
            .args([
                "render",
                "--cpu",
                "--json",
                "--format",
                "tiff",
                "--bits",
                "16",
                "--settings",
            ])
            .arg(&file)
            .args(["--out", out])
            .arg(sb.photos())
            .assert()
            .success();
        json_lines(&out.get_output().stdout)
    };
    let first = run("out-1");
    assert_eq!(first.len(), 3);
    for o in &first {
        assert_eq!(o["status"], "ok", "{o}");
    }
    let path = |dir: &str, name: &str| sb.root().join(dir).join(format!("{name}_signature.tif"));
    let bytes = |p: std::path::PathBuf| {
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    // The rows act: the mask is a part of the picture, the trim of 0.7 stop reaches that part only, so the same photo with an empty
    // node is another file.
    let none = bytes(path("out-1", "c_none"));
    assert_ne!(
        bytes(path("out-1", "a_subject")),
        none,
        "Subject changed no pixel"
    );
    assert_ne!(
        bytes(path("out-1", "b_people")),
        none,
        "People changed no pixel"
    );
    // The same file again: the same bytes.
    let second = run("out-2");
    assert_eq!(second.len(), 3);
    for name in ["a_subject", "b_people", "c_none"] {
        assert_eq!(
            bytes(path("out-1", name)),
            bytes(path("out-2", name)),
            "{name} is not the same file twice"
        );
    }
}

/// A copy of Yellowmoon with no Object model (`data/models/sam2` is not there, or does not load): an Object row stops that photo with
/// `error.kind` `mask` and "This copy of Yellowmoon has no Object model. Reinstall the app.", exit 1, nothing written; `ym doctor` says
/// `weights_missing`; and a photo with the eye of the row off, or no Object row, goes on (`docs/10`, `docs/23` section 9, N-18). With the
/// weights (the shipped data folder, where `tools/fetch/sam2.sh` has fetched them) the same file renders.
#[cfg(unix)]
#[test]
fn an_object_row_without_the_model_is_the_mask_error_and_with_it_the_photo_renders() {
    let sb = Sandbox::new();
    sb.arw("a_object.ARW");
    sb.arw("b_off.ARW");
    let doc = |rows: &str| serde_json::from_str::<Value>(&masked_file(rows)).expect("a document");
    let file = sb.root().join("per-file.json");
    let doc_text = serde_json::json!({
        "a_object.ARW": doc(r#"{ "kind": "object", "points": [[0.5, 0.5, true]] }"#),
        "b_off.ARW": doc(r#"{ "kind": "object", "enabled": false }"#),
    });
    std::fs::write(&file, doc_text.to_string()).expect("write");
    let without = sb.data_without_the_sam_weights();
    let out = sb
        .ym_data(&without)
        .args(["render", "--cpu", "--json", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .code(1);
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 2);
    let find = |name: &str| {
        objects
            .iter()
            .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with(name)))
            .unwrap_or_else(|| panic!("no record for {name}"))
    };
    let o = find("a_object.ARW");
    assert_eq!(o["status"], "error");
    assert_eq!(o["output"], Value::Null);
    assert_eq!(o["error"]["kind"], "mask");
    assert_eq!(
        o["error"]["message"],
        "This copy of Yellowmoon has no Object model. Reinstall the app."
    );
    assert!(o["camera"].is_object() && o["settings"].is_object());
    assert!(!sb.default_export().join("a_object_signature.jpg").exists());
    assert_eq!(find("b_off.ARW")["status"], "ok");
    // The doctor says so, and Object is not on offer.
    let out = sb
        .ym_data(&without)
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    assert_eq!(v["masks"]["unavailable"]["object"], "weights_missing");
    assert!(
        !v["masks"]["kinds"]
            .as_array()
            .expect("kinds")
            .iter()
            .any(|k| k == "object")
    );
    let text = sb.ym_data(&without).arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(
        text.contains("Object (the SAM weights are missing)"),
        "the Masks row names the reason:\n{text}"
    );
    // With the weights the same file renders, and the record's settings hold the row.
    if ym_core::pipeline::ObjectModel::probe(&common::data_dir()).is_some_and(|m| m.sam().is_ok()) {
        let out = sb
            .ym()
            .args(["render", "--cpu", "--json", "--settings"])
            .arg(&file)
            .arg(sb.photos())
            .assert()
            .success();
        let objects = json_lines(&out.get_output().stdout);
        let o = objects
            .iter()
            .find(|o| {
                o["input"]
                    .as_str()
                    .is_some_and(|i| i.ends_with("a_object.ARW"))
            })
            .expect("a record");
        assert_eq!(o["status"], "ok");
        assert!(sb.default_export().join("a_object_signature.jpg").exists());
    } else {
        assert!(
            std::env::var_os("YM_REQUIRE_SAM2").is_none(),
            "YM_REQUIRE_SAM2 is set and data/models/sam2 is missing"
        );
        eprintln!(
            "\n* the half of this test that renders with the weights was SKIPPED: no SAM weights\n"
        );
    }
}

/// A Background row renders from `--settings` (`docs/10`, N-15): the blur map is made from the photo itself, the
/// photo is written, and a Level that lets everything count (the formula gives at least one half at Level 100) is
/// not the photo without the row. The settings the record prints hold the row.
#[test]
fn a_background_row_renders_from_a_settings_file() {
    let sb = Sandbox::new();
    let arw = sb.arw_bokeh("lights.ARW", 640, 480, 4);
    let row =
        r#"{ "kind": "background", "mode": "add", "level": 1, "softness": 0.4, "feather": 0.1 }"#;
    let with = sb.root().join("background.json");
    std::fs::write(&with, masked_file(row)).expect("write");
    let without = sb.root().join("none.json");
    std::fs::write(&without, masked_file("")).expect("write");
    let render = |settings: &std::path::Path, dir: &str| {
        let out_dir = sb.root().join(dir);
        let out = sb
            .ym()
            .args([
                "render",
                "--cpu",
                "--json",
                "--format",
                "tiff",
                "--settings",
            ])
            .arg(settings)
            .arg("--out")
            .arg(&out_dir)
            .arg(&arw)
            .assert()
            .success();
        let o = json_lines(&out.get_output().stdout).remove(0);
        assert_eq!(o["status"], "ok");
        let path = o["output"].as_str().expect("output").to_owned();
        (o, std::fs::read(path).expect("the file"))
    };
    let (o, a) = render(&with, "with");
    let (_, b) = render(&without, "without");
    assert_ne!(a, b, "the Background row changed no pixel");
    let rows = &common::node(&o, 2)["masks"]["items"];
    assert_eq!(rows[0]["kind"], "background");
    assert_eq!(rows[0]["level"], 1.0);
    assert_eq!(rows[0]["softness"], 0.4);
    assert_eq!(rows[0]["refine"], 0.5);
}

/// A copy of Yellowmoon with no Depth model (`data/models/depth` is not there): a Depth row stops that photo with `error.kind` `mask` and
/// "This copy of Yellowmoon has no Depth model. Reinstall the app.", exit 1, nothing written; `ym doctor` says `weights_missing` and names
/// the depth weights; and a photo with the eye of the row off, or no Depth row, goes on (`docs/10`, `docs/23` section 3.10a, D-04).
#[cfg(unix)]
#[test]
fn a_depth_row_without_the_model_is_the_mask_error_and_the_other_photos_go_on() {
    let sb = Sandbox::new();
    sb.arw("a_depth.ARW");
    sb.arw("b_off.ARW");
    sb.arw("c_plain.ARW");
    let doc = |rows: &str| serde_json::from_str::<Value>(&masked_file(rows)).expect("a document");
    let file = sb.root().join("per-file.json");
    let doc_text = serde_json::json!({
        "a_depth.ARW": doc(r#"{ "kind": "depth", "near": 0, "far": 0.4 }"#),
        "b_off.ARW": doc(r#"{ "kind": "depth", "enabled": false }"#),
        "c_plain.ARW": doc(""),
    });
    std::fs::write(&file, doc_text.to_string()).expect("write");
    let without = sb.data_without_the_depth_weights();
    let out = sb
        .ym_data(&without)
        .args(["render", "--cpu", "--json", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .code(1);
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 3);
    let find = |name: &str| {
        objects
            .iter()
            .find(|o| o["input"].as_str().is_some_and(|i| i.ends_with(name)))
            .unwrap_or_else(|| panic!("no record for {name}"))
    };
    let o = find("a_depth.ARW");
    assert_eq!(o["status"], "error");
    assert_eq!(o["output"], Value::Null);
    assert_eq!(o["error"]["kind"], "mask");
    assert_eq!(
        o["error"]["message"],
        "This copy of Yellowmoon has no Depth model. Reinstall the app."
    );
    assert!(o["camera"].is_object() && o["settings"].is_object());
    assert!(!sb.default_export().join("a_depth_signature.jpg").exists());
    assert_eq!(find("b_off.ARW")["status"], "ok");
    assert_eq!(find("c_plain.ARW")["status"], "ok");
    // The doctor says so, and Depth is not on offer.
    let out = sb
        .ym_data(&without)
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    assert_eq!(v["masks"]["unavailable"]["depth"], "weights_missing");
    assert!(
        !v["masks"]["kinds"]
            .as_array()
            .expect("kinds")
            .iter()
            .any(|k| k == "depth")
    );
    let text = sb.ym_data(&without).arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(
        text.contains("Depth (the depth weights are missing)"),
        "the Masks row names the reason:\n{text}"
    );
}

/// A Depth row renders from `--settings` (`docs/10`, D-04) with the fake depth model (`YELLOWMOON_TEST_DEPTH=fake`): the map is made from
/// the photo's own picture, the photo is written, and the row changes pixels. The same file twice gives the same bytes, a `near` above
/// `far` is both at their mean, and the settings the record prints hold the row with its defaults and its numbers.
#[test]
fn a_depth_row_renders_from_a_settings_file_with_the_fake_model() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let row = r#"{ "kind": "depth", "mode": "add", "near": 0.1, "far": 0.7, "softness": 0.2, "feather": 0.1 }"#;
    let with = sb.root().join("depth.json");
    std::fs::write(&with, masked_file(row)).expect("write");
    let swapped = sb.root().join("swapped.json");
    std::fs::write(
        &swapped,
        masked_file(r#"{ "kind": "depth", "near": 0.9, "far": 0.1 }"#),
    )
    .expect("write");
    let mean = sb.root().join("mean.json");
    std::fs::write(
        &mean,
        masked_file(r#"{ "kind": "depth", "near": 0.5, "far": 0.5 }"#),
    )
    .expect("write");
    let defaults = sb.root().join("defaults.json");
    std::fs::write(&defaults, masked_file(r#"{ "kind": "depth" }"#)).expect("write");
    let without = sb.root().join("none.json");
    std::fs::write(&without, masked_file("")).expect("write");
    let render = |settings: &std::path::Path, dir: &str| {
        let out_dir = sb.root().join(dir);
        let out = sb
            .ym_fake_depth()
            .args([
                "render",
                "--cpu",
                "--json",
                "--format",
                "tiff",
                "--settings",
            ])
            .arg(settings)
            .arg("--out")
            .arg(&out_dir)
            .arg(&arw)
            .assert()
            .success();
        let o = json_lines(&out.get_output().stdout).remove(0);
        assert_eq!(o["status"], "ok", "{o}");
        let path = o["output"].as_str().expect("output").to_owned();
        (o, std::fs::read(path).expect("the file"))
    };
    let (o, a) = render(&with, "with");
    let (_, again) = render(&with, "again");
    let (_, b) = render(&without, "without");
    assert_ne!(a, b, "the Depth row changed no pixel");
    assert_eq!(a, again, "the same file twice is the same bytes");
    let rows = &common::node(&o, 2)["masks"]["items"];
    assert_eq!(rows[0]["kind"], "depth");
    assert_eq!(rows[0]["near"], 0.1);
    assert_eq!(rows[0]["far"], 0.7);
    assert_eq!(rows[0]["softness"], 0.2);
    assert_eq!(rows[0]["feather"], 0.1);
    assert_eq!(rows[0]["refine"], 0.5);
    // The defaults of `docs/10`.
    let (d, _) = render(&defaults, "defaults");
    let rows = &common::node(&d, 2)["masks"]["items"];
    assert_eq!(
        (
            &rows[0]["near"],
            &rows[0]["far"],
            &rows[0]["softness"],
            &rows[0]["grow"]
        ),
        (
            &Value::from(0.0),
            &Value::from(0.5),
            &Value::from(0.5),
            &Value::from(0.0)
        )
    );
    // `near` above `far` is both at their mean: the file of 0.9 and 0.1 is the file of 0.5 and 0.5.
    let (_, s) = render(&swapped, "swapped");
    let (_, m) = render(&mean, "mean");
    assert_eq!(s, m, "0.9 and 0.1 are both 0.5");
}

/// A Brush row renders from `--settings` (`docs/10`): the photo is written, and it is not the photo without the
/// brush.
#[test]
fn a_brush_row_renders_from_a_settings_file() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let brush = r#"{ "kind": "brush", "mode": "add", "feather": 0.2, "grow": 0.2, "refine": 0.4,
        "strokes": [ { "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.3, 0.4], [0.6, 0.55]] } ] }"#;
    let with = sb.root().join("brush.json");
    std::fs::write(&with, masked_file(brush)).expect("write");
    let without = sb.root().join("none.json");
    std::fs::write(&without, masked_file("")).expect("write");
    let render = |settings: &std::path::Path, dir: &str| {
        let out_dir = sb.root().join(dir);
        let out = sb
            .ym()
            .args([
                "render",
                "--cpu",
                "--json",
                "--format",
                "tiff",
                "--settings",
            ])
            .arg(settings)
            .arg("--out")
            .arg(&out_dir)
            .arg(&arw)
            .assert()
            .success();
        let o = json_lines(&out.get_output().stdout).remove(0);
        assert_eq!(o["status"], "ok");
        let path = o["output"].as_str().expect("output").to_owned();
        (o, std::fs::read(path).expect("the file"))
    };
    let (o, a) = render(&with, "with");
    let (_, b) = render(&without, "without");
    assert_ne!(a, b, "the brush changed no pixel");
    // The settings the record prints hold the strokes.
    let rows = &common::node(&o, 2)["masks"]["items"];
    assert_eq!(rows[0]["kind"], "brush");
    assert_eq!(rows[0]["strokes"][0]["points"][1][0], 0.6);
    assert_eq!(rows[0]["refine"], 0.4);
}

#[test]
fn settings_can_come_from_stdin_and_values_out_of_range_are_limited_with_a_warning() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["render", "--json", "--settings", "-"])
        .arg(&arw)
        .write_stdin(r#"{ "nodes": { "items": [ { "id": 1, "grade": { "contrast": { "kind": "manual", "value": 500 } } } ] } }"#)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "out-of-range values for 1.grade.contrast",
        ));
    assert_eq!(
        common::node(&json_lines(&out.get_output().stdout)[0], 1)["grade"]["contrast"]["value"],
        100.0
    );
}

#[test]
fn a_settings_file_that_does_not_parse_is_a_usage_error_and_nothing_is_rendered() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let file = sb.root().join("bad.json");
    for (content, needle) in [
        ("{ \"contrsat\": 3 }", "contrsat"),
        ("not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        (r#"{ "nodes": {}, "a.ARW": {} }"#, "mixes"),
        // A rule of the graph is refused with its plain sentence.
        (
            r#"{ "nodes": { "links": [] } }"#,
            "The output needs one picture link.",
        ),
        (
            r#"{ "nodes": { "items": [ { "id": 2, "kind": "parallel" } ] } }"#,
            "This settings file has a node of a kind (“parallel”)",
        ),
        // A typo whose value is an object must not pass for a photo's file name.
        (
            r#"{ "color_ballance": { "shadows": [0.1, 0.0] } }"#,
            "color_ballance",
        ),
        (
            r#"{ "curve2": {} }"#,
            "not `nodes` or the name of an ARW file",
        ),
    ] {
        std::fs::write(&file, content).expect("write");
        sb.ym()
            .args(["render", "--settings"])
            .arg(&file)
            .arg(&arw)
            .assert()
            .code(2)
            .stderr(predicate::str::contains(needle));
    }
    assert!(!sb.default_export().exists(), "nothing was rendered");
    sb.ym()
        .args(["render", "--settings", "/nonexistent/settings.json"])
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Couldn't read the settings file"));
}

#[test]
fn exit_code_2_for_usage_errors() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let cases: Vec<(Vec<&str>, &str)> = vec![
        (vec!["--contrast", "500"], "between -100 and 100"),
        (vec!["--exposure", "9"], "between -4 and 4"),
        (
            vec!["--exposure", "1", "--exposure-offset", "1"],
            "cannot be used with",
        ),
        (vec!["--wb", "1500"], "between 2000 and 11000"),
        (vec!["--nr", "1,2"], "LUMINANCE,COLOUR,DETAIL"),
        (vec!["--format", "bmp"], "possible values"),
        (vec!["--quality", "10"], "50..=100"),
        (vec!["--bits", "12"], "8 or 16"),
        (
            vec!["--skip-existing", "--overwrite"],
            "cannot be used with",
        ),
        (vec!["--quiet", "--verbose"], "cannot be used with"),
        (
            vec!["--profile", "no-such-profile"],
            "There is no camera profile",
        ),
        (vec!["--long-edge", "5"], "16..=65535"),
        (vec!["--output-sharpening", "201"], "between 0 and 200"),
        (vec!["--output-sharpening", "-5"], "between 0 and 200"),
        (vec!["--output-sharpening", "a lot"], "is not a number"),
        (vec!["--no-such-flag"], "unexpected argument"),
        (vec!["--white-offset", "0.5"], "unexpected argument"),
        (vec!["--rolloff", "1.5,0.5"], "the highlight range"),
        (vec!["--rolloff", "0.5,-1"], "the low-light toe"),
    ];
    for (flags, needle) in cases {
        sb.ym()
            .arg("render")
            .args(&flags)
            .arg(&arw)
            .assert()
            .code(2)
            .stderr(predicate::str::contains(needle));
    }
    // Both lists want stdin.
    sb.ym()
        .args(["render", "-", "--settings", "-"])
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("both read stdin").or(predicate::str::contains("read stdin")),
        );
    assert!(!sb.default_export().exists(), "nothing was rendered");
}

#[test]
fn options_that_do_not_apply_to_the_format_are_warned_about() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    sb.ym()
        .args([
            "render",
            "--format",
            "dng",
            "--quality",
            "70",
            "--long-edge",
            "50",
            "--output-sharpening",
            "40",
        ])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "--quality (JPEG only): ignored for DNG files.",
        ))
        .stderr(predicate::str::contains(
            "--long-edge (not for DNG): ignored for DNG files.",
        ))
        .stderr(predicate::str::contains(
            "--output-sharpening (not for DNG): ignored for DNG files.",
        ));
}

#[test]
fn the_output_sharpening_option_is_in_the_settings_the_json_reports() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let run = |flags: &[&str], dir: &str| {
        let out = sb
            .ym()
            .args([
                "render", "--json", "--format", "png", "--bits", "8", "--out",
            ])
            .arg(sb.root().join(dir))
            .args(flags)
            .arg(&arw)
            .assert()
            .success();
        json_lines(&out.get_output().stdout).remove(0)
    };
    // The default is 100, and it is in the settings an export records.
    let default = run(&[], "default");
    assert_eq!(
        common::node(&default, "output")["output_sharpening"],
        json!(100.0)
    );
    let set = run(&["--output-sharpening", "35"], "set");
    assert_eq!(
        common::node(&set, "output")["output_sharpening"],
        json!(35.0)
    );
    let off = run(&["--output-sharpening", "0"], "off");
    assert_eq!(
        common::node(&off, "output")["output_sharpening"],
        json!(0.0)
    );
    // The three files have three different pictures.
    let file = |dir: &str| std::fs::read(sb.root().join(dir).join("a_signature.png")).expect("png");
    let (a, b, c) = (file("default"), file("set"), file("off"));
    assert!(a != b && a != c && b != c);
}

/// The image data of a PNG (its IDAT chunks, concatenated): the pixels as the encoder wrote them,
/// without the XMP packet, which records the settings and so differs between the files compared.
fn png_image_data(path: &std::path::Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("png");
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "a PNG");
    let mut at = 8;
    let mut data = Vec::new();
    while at + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().expect("length")) as usize;
        if &bytes[at + 4..at + 8] == b"IDAT" {
            data.extend_from_slice(&bytes[at + 8..at + 8 + len]);
        }
        at += 12 + len;
    }
    assert!(!data.is_empty(), "image data");
    data
}

#[test]
fn a_colour_warp_in_a_settings_file_moves_colours_and_locked_pins_alone_do_not() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    // A white balance far from the light, so that the photo has colour for the pins to move; the
    // same in every file.
    let render = |warp: Value, dir: &str| {
        let file = sb.root().join(format!("{dir}.json"));
        let doc = common::field_doc(json!({
            "white_balance": { "kind": "manual", "kelvin": 2500, "tint": 30 },
            "color_warp": warp,
        }));
        std::fs::write(&file, doc.to_string()).expect("write");
        let out = sb
            .ym()
            .args([
                "render", "--json", "--format", "png", "--bits", "8", "--out",
            ])
            .arg(sb.root().join(dir))
            .arg("--settings")
            .arg(&file)
            .arg(&arw)
            .assert()
            .success();
        let o = json_lines(&out.get_output().stdout).remove(0);
        let pixels = png_image_data(std::path::Path::new(o["output"].as_str().expect("output")));
        (o, pixels)
    };
    let (none, plain) = render(json!({ "pins": [] }), "none");
    assert_eq!(
        common::node(&none, 1)["grade"]["color_warp"],
        json!({ "pins": [] })
    );
    // One pin on the neutral axis with the largest range moves every colour of the photo (neutrals
    // themselves never move). The file names only the source, target and range; the settings the
    // JSON reports (and the file records) have every field.
    let (moved, warped) = render(
        json!({ "pins": [{ "source": [0.0, 0.0], "target": [0.06, 0.04], "range": 0.25 }] }),
        "moved",
    );
    assert_eq!(
        common::node(&moved, 1)["grade"]["color_warp"]["pins"][0],
        json!({ "source": [0.0, 0.0], "target": [0.06, 0.04], "locked": false, "range": 0.25,
            "shadows": 1.0, "highlights": 1.0, "pivot_ev": 0.0, "exposure_ev": 0.0,
            "saturation": 0.0 })
    );
    assert_ne!(warped, plain, "the warp moves the photo's colours");
    // A pin that stays and only changes saturation is not skipped as a still pin: it changes the
    // photo, and the settings the JSON reports keep the number. At saturation 0 (or left out) it is
    // a still pin and the picture is the plain one, byte for byte.
    let (saturated, more) = render(
        json!({ "pins": [{ "source": [0.0, 0.0], "range": 0.25, "saturation": 0.6 }] }),
        "saturated",
    );
    assert_eq!(
        common::node(&saturated, 1)["grade"]["color_warp"]["pins"][0]["saturation"],
        json!(0.6)
    );
    assert_ne!(more, plain, "saturation alone changes the photo's colours");
    let (_, zero) = render(
        json!({ "pins": [{ "source": [0.0, 0.0], "range": 0.25, "saturation": 0 }] }),
        "zero",
    );
    assert_eq!(zero, plain, "a pin that changes nothing is no warp");
    // Locked pins alone change nothing, byte for byte.
    let (_, locked) = render(
        json!({ "pins": [{ "source": [0.1, 0.0], "locked": true }, { "source": [-0.05, -0.1], "locked": true, "range": 0.2 }] }),
        "locked",
    );
    assert_eq!(locked, plain, "locked pins alone are no warp");
    // Out of range: limited, with a warning that names the field.
    let file = sb.root().join("wild.json");
    std::fs::write(
        &file,
        common::field_doc(
            json!({ "color_warp": { "pins": [{ "source": [0.9, 0.0], "range": 3 }] } }),
        )
        .to_string(),
    )
    .expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--out"])
        .arg(sb.root().join("wild"))
        .arg("--settings")
        .arg(&file)
        .arg(&arw)
        .assert()
        .success();
    let o = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(
        common::node(&o, 1)["grade"]["color_warp"]["pins"][0]["source"],
        json!([0.5, 0.0])
    );
    assert_eq!(
        common::node(&o, 1)["grade"]["color_warp"]["pins"][0]["range"],
        json!(0.25)
    );
    let err = text(&out.get_output().stderr);
    assert!(err.contains("1.grade.color_warp"), "{err}");
}

#[test]
fn exit_code_3_when_a_look_is_missing() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    sb.ym()
        .args(["render", "--look", "no-such-look"])
        .arg(&arw)
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "No look called \"no-such-look\". Run `ym looks list`",
        ));
    assert!(!sb.default_export().exists());
    // A look named in the settings file that is not there fails that photo with the look code.
    let file = sb.root().join("s.json");
    std::fs::write(
        &file,
        r#"{ "nodes": { "items": [ { "id": "output", "look": "user:gone" } ] } }"#,
    )
    .expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--settings"])
        .arg(&file)
        .arg(&arw)
        .assert()
        .code(3);
    let o = &json_lines(&out.get_output().stdout)[0];
    assert_eq!(o["status"], "error");
    assert_eq!(o["error"]["kind"], "look");
    assert_eq!(o["output"], Value::Null);
    // The photo's own estimates are in the failure: it was developed, the export failed.
    assert_eq!(o["camera"]["body"], "Sony A7R IV");
}

#[test]
fn exit_code_5_when_there_is_nothing_to_render() {
    let sb = Sandbox::new();
    let empty = sb.root().join("empty");
    std::fs::create_dir_all(&empty).expect("folder");
    std::fs::write(empty.join("a.jpg"), "x").expect("write");
    sb.ym()
        .arg("render")
        .arg(&empty)
        .assert()
        .code(5)
        .stderr(predicate::str::contains("No ARW files were found"));
    sb.ym()
        .args(["render", "does-not-exist.ARW"])
        .assert()
        .code(5)
        .stderr(predicate::str::contains("does-not-exist.ARW"));
    // An empty list on stdin.
    sb.ym()
        .args(["render", "-"])
        .write_stdin("\n\n")
        .assert()
        .code(5);
    // With --json the named paths that failed are still reported, as objects.
    let out = sb
        .ym()
        .args(["render", "--json", "does-not-exist.ARW"])
        .assert()
        .code(5);
    let o = &json_lines(&out.get_output().stdout)[0];
    assert_eq!(o["status"], "error");
    assert_eq!(o["error"]["kind"], "not_found");
}

#[test]
fn a_named_file_that_is_not_an_arw_is_a_failure_next_to_the_photos_that_worked() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let notes = sb.root().join("notes.txt");
    std::fs::write(&notes, "hi").expect("write");
    sb.ym()
        .arg("render")
        .arg(&arw)
        .arg(&notes)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("a_signature.jpg"))
        .stderr(predicate::str::contains(
            "notes.txt: This isn't an ARW file.",
        ));
}

#[test]
fn require_gpu_exits_4_when_no_adapter_opens() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    // No Direct3D 12 on this machine whatever it is: the backend list is empty.
    let out = sb
        .ym()
        .env("WGPU_BACKEND", "dx12")
        .args(["render", "--require-gpu"])
        .arg(&arw)
        .assert()
        .code(4);
    assert!(
        text(&out.get_output().stderr).contains("--require-gpu was given, so nothing was rendered"),
        "{}",
        text(&out.get_output().stderr)
    );
    assert!(!sb.default_export().exists());
    // Without the flag the same machine renders on the CPU and says so.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .arg("render")
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "No usable GPU was found. Rendering on the CPU",
        ));
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .args(["render", "--quiet"])
        .arg(&arw)
        .arg("--overwrite")
        .assert()
        .success()
        .stderr(predicate::str::is_empty());
}

// --- the graphics policy (`docs/17-gpu.md` section 1) ----------------------------------------

/// Whether a graphics adapter opens on this machine (lavapipe counts), asked of `ym doctor`.
fn adapter_opens(sb: &Sandbox) -> bool {
    let out = sb
        .ym()
        .env_remove("YELLOWMOON_GPU")
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    v["gpu"]["available"] == true
}

/// Whether the default policy, `auto`, puts the device to work on this machine, asked of `ym doctor`: a hardware
/// adapter opens (a software one, Mesa lavapipe on a Linux runner, is left alone under `auto`). Since the stage
/// sign-off of 2026-10-15 `auto` runs every stage, so where this is true an export is made on the device and is
/// not the CPU reference to the byte (a Mac); where it is false the default is the CPU reference (Linux).
fn auto_uses_the_device(sb: &Sandbox) -> bool {
    let out = sb
        .ym()
        .env_remove("YELLOWMOON_GPU")
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    v["gpu_policy"]["uses_gpu"] == true
}

#[test]
fn cpu_and_require_gpu_together_are_a_usage_error() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["render", "--cpu", "--require-gpu"])
        .arg(&arw)
        .assert()
        .code(2);
    let err = text(&out.get_output().stderr);
    assert!(
        err.contains("--cpu") && err.contains("--require-gpu"),
        "{err}"
    );
    assert!(!sb.default_export().exists(), "nothing was rendered");
}

#[test]
fn require_gpu_exits_4_only_when_no_device_opens_at_the_start() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    // No adapter: exit 4 and nothing rendered.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .args(["render", "--require-gpu"])
        .arg(&arw)
        .assert()
        .code(4);
    assert!(!sb.default_export().exists());
    // An adapter that opens: the run goes on and writes the file.
    if adapter_opens(&sb) {
        sb.ym()
            .args(["render", "--require-gpu"])
            .arg(&arw)
            .assert()
            .success();
        assert_eq!(names_in(&sb.default_export()), ["a_signature.jpg"]);
    }
}

#[test]
fn the_environment_asks_for_a_device_as_the_flag_does_and_a_flag_beats_it() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    // YELLOWMOON_GPU=require is --require-gpu, and the message names what was asked.
    let out = sb
        .ym()
        .env("WGPU_BACKEND", "dx12")
        .env("YELLOWMOON_GPU", "require")
        .arg("render")
        .arg(&arw)
        .assert()
        .code(4);
    let err = text(&out.get_output().stderr);
    assert!(
        err.contains("YELLOWMOON_GPU=require is set, so nothing was rendered"),
        "{err}"
    );
    // --cpu beats it: no device is asked for, so none is missing.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .env("YELLOWMOON_GPU", "require")
        .args(["render", "--cpu"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains("No usable GPU").not());
    // --require-gpu beats YELLOWMOON_GPU=off.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .env("YELLOWMOON_GPU", "off")
        .args(["render", "--require-gpu", "--overwrite"])
        .arg(&arw)
        .assert()
        .code(4);
    // `off` in the environment alone is the CPU: nothing to say about a missing device.
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .env("YELLOWMOON_GPU", "off")
        .args(["render", "--overwrite"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains("No usable GPU").not());
}

#[test]
fn a_setting_that_is_not_a_policy_is_a_warning_and_the_default() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    sb.ym()
        .env("YELLOWMOON_GPU", "maybe")
        .arg("render")
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "YELLOWMOON_GPU=maybe is not a graphics setting. Use auto, on, off or require.",
        ));
    // `--quiet` leaves errors only.
    sb.ym()
        .env("YELLOWMOON_GPU", "maybe")
        .args(["render", "--quiet", "--overwrite"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::is_empty());
}

#[test]
fn the_json_of_a_file_has_this_shape() {
    // The additive `device` key is in the snapshot, between `settings` and `timings_ms`. Apple Auto
    // is off so that the shape is the same on every machine (with Apple Auto: `tests/apple_auto.rs`).
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .env_remove("YELLOWMOON_GPU")
        .args([
            "render",
            "--json",
            "--quiet",
            "--cpu",
            "--apple-auto",
            "off",
        ])
        .arg(&arw)
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    insta::assert_json_snapshot!("render_json_shape", shape(&v));
}

/// The bytes of the JPEG `ym render` writes for `a.ARW` under the given policy.
fn export_under(sb: &Sandbox, arw: &std::path::Path, flags: &[&str], gpu: Option<&str>) -> Vec<u8> {
    let mut cmd = sb.ym();
    cmd.env_remove("YELLOWMOON_GPU");
    if let Some(gpu) = gpu {
        cmd.env("YELLOWMOON_GPU", gpu);
    }
    cmd.args(["render", "--quiet", "--overwrite"])
        .args(flags)
        .arg(arw)
        .assert()
        .success();
    std::fs::read(sb.default_export().join("a_signature.jpg")).expect("the export")
}

#[test]
fn every_policy_that_keeps_the_cpu_writes_the_bytes_of_the_cpu_reference() {
    // `--cpu` and `YELLOWMOON_GPU=off` open no device: the CPU reference, the same bytes. So does `auto`
    // (the default) where it finds no hardware adapter, as on a Linux runner that has only a software one.
    // Where it does (a Mac), `auto` runs every stage, which since 2026-10-15 are all signed off, and writes
    // what a policy that puts the raw chain on the device (`on`, `--require-gpu`) writes: a picture that
    // agrees with the reference within the kernels' tolerances (`docs/05`), not necessarily to the byte. That
    // one is checked to be the same kind of file of about the same size, and `tests/parity_gpu.rs` pins that
    // the app and `ym` agree with each other on the device.
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let reference = export_under(&sb, &arw, &["--cpu"], None);
    assert_eq!(&reference[..2], [0xFF, 0xD8]);
    let near = |bytes: Vec<u8>, what: &str| {
        assert_eq!(&bytes[..2], [0xFF, 0xD8], "{what}");
        let (a, b) = (bytes.len() as f64, reference.len() as f64);
        assert!((a - b).abs() / b < 0.05, "{what}: {a} bytes against {b}");
    };
    assert_eq!(
        export_under(&sb, &arw, &[], Some("off")),
        reference,
        "YELLOWMOON_GPU=off"
    );
    let auto_on_the_device = auto_uses_the_device(&sb);
    for (bytes, what) in [
        (export_under(&sb, &arw, &[], None), "default (auto)"),
        (
            export_under(&sb, &arw, &[], Some("auto")),
            "YELLOWMOON_GPU=auto",
        ),
    ] {
        if auto_on_the_device {
            near(bytes, what);
        } else {
            assert_eq!(bytes, reference, "{what}");
        }
    }
    near(
        export_under(&sb, &arw, &[], Some("on")),
        "YELLOWMOON_GPU=on",
    );
    if adapter_opens(&sb) {
        near(
            export_under(&sb, &arw, &["--require-gpu"], None),
            "--require-gpu",
        );
    }
}

#[test]
fn json_says_where_the_pictures_were_made_after_the_settings() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    for (flags, gpu) in [(vec!["--cpu"], None), (vec![], Some("on")), (vec![], None)] {
        let mut cmd = sb.ym();
        cmd.env_remove("YELLOWMOON_GPU");
        if let Some(gpu) = gpu {
            cmd.env("YELLOWMOON_GPU", gpu);
        }
        let out = cmd
            .args(["render", "--json", "--quiet", "--overwrite"])
            .args(&flags)
            .arg(&arw)
            .assert()
            .success();
        let line = String::from_utf8(out.get_output().stdout.clone()).expect("utf-8");
        let v: Value = serde_json::from_str(line.trim()).expect("one JSON object");
        // The raw chain of an export runs on the device when the policy lets it and an adapter opens:
        // `on` with any adapter, `auto` (no flag, no variable) with a hardware one (every stage is signed
        // off since 2026-10-15); `--cpu`, and a machine without that adapter, is the CPU.
        let on_the_device = match (flags.is_empty(), gpu) {
            (true, Some("on")) => adapter_opens(&sb),
            (true, None) => auto_uses_the_device(&sb),
            _ => false,
        };
        let expected = if on_the_device { "gpu" } else { "cpu" };
        assert_eq!(v["device"], expected, "{flags:?} {gpu:?}");
        // Additive: `device` sits between `settings` and `timings_ms`, the rest keep their order.
        let keys: Vec<&str> = v
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        let at = |key: &str| line.find(&format!("\"{key}\":")).unwrap_or(usize::MAX);
        let order = [
            "input",
            "output",
            "status",
            "camera",
            "look",
            "auto",
            "settings",
            "device",
            "timings_ms",
            "notes",
            "pipeline_version",
            "error",
        ];
        for pair in order.windows(2) {
            assert!(
                at(pair[0]) < at(pair[1]),
                "{} before {}: {line}",
                pair[0],
                pair[1]
            );
        }
        for key in order {
            assert!(keys.contains(&key), "{key} in {keys:?}");
        }
    }
    // A file that was skipped was not rendered: its device is null.
    let out = sb
        .ym()
        .args(["render", "--json", "--quiet", "--skip-existing"])
        .arg(&arw)
        .assert()
        .success();
    let skipped = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(skipped["status"], "skipped");
    assert_eq!(skipped["device"], Value::Null);
    // A failure has no device at all: nothing was rendered.
    let out = sb
        .ym()
        .args(["render", "--json", "does-not-exist.ARW"])
        .assert()
        .code(5);
    assert!(
        json_lines(&out.get_output().stdout)[0]
            .get("device")
            .is_none()
    );
}

#[test]
fn quiet_prints_nothing_and_verbose_adds_timings() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    sb.ym()
        .args(["render", "--quiet"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::is_empty())
        .stdout(predicate::str::contains("a_signature.jpg"));
    sb.ym()
        .args(["render", "--verbose", "--overwrite"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains("render"))
        .stderr(predicate::str::contains("encode"));
}

#[test]
fn a_reader_that_goes_away_does_not_stop_the_export() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw("b.ARW");
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .args(["render", "--json"])
        .arg(sb.photos())
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env(common::NO_AUTO_PROVIDERS.0, common::NO_AUTO_PROVIDERS.1)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    drop(child.stdout.take());
    let output = child.wait_with_output().expect("wait");
    assert_eq!(output.status.code(), Some(0), "{:?}", text(&output.stderr));
    assert_eq!(
        names_in(&sb.default_export()).len(),
        2,
        "the files are the product"
    );
}

/// Ctrl-C: the run ends with 130, says so, and leaves no partial file behind.
#[cfg(unix)]
#[test]
fn ctrl_c_ends_the_run_with_130_and_deletes_the_partial_file() {
    let sb = Sandbox::new();
    // Big enough that the export is still running when the signal arrives.
    let arw = sb.arw_sized("big.ARW", 3000, 2000, 5);
    let child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .arg("render")
        .arg(&arw)
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env(common::NO_AUTO_PROVIDERS.0, common::NO_AUTO_PROVIDERS.1)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    // Let the handler get installed and the work begin, then press Ctrl-C.
    std::thread::sleep(Duration::from_millis(1500));
    let started = Instant::now();
    let kill = std::process::Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill");
    assert!(kill.success());
    let output = child.wait_with_output().expect("wait");
    assert_eq!(
        output.status.code(),
        Some(130),
        "stderr: {}",
        text(&output.stderr)
    );
    assert!(
        started.elapsed() < Duration::from_secs(60),
        "the run stops promptly after Ctrl-C"
    );
    let err = text(&output.stderr);
    assert!(err.contains("Interrupted"), "{err}");
    assert!(output.stdout.is_empty(), "no file was finished");
    let leftovers = names_in(&sb.default_export());
    assert!(
        leftovers.is_empty(),
        "no output and no partial file are left: {leftovers:?}"
    );
}

#[test]
fn a_photo_whose_settings_are_refused_is_reported_and_not_exported_with_its_defaults() {
    let sb = Sandbox::new();
    sb.arw("good.ARW");
    sb.arw_clipped("white.ARW");
    let file = sb.root().join("pick.json");
    // A white picked on a clipped patch is refused by the colour chain.
    std::fs::write(&file, r#"{ "white.ARW": { "nodes": { "items": [ { "id": "input", "white_pick": [0.5, 0.5] } ] } } }"#).expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--settings"])
        .arg(&file)
        .arg(sb.photos())
        .assert()
        .code(1);
    let objects = json_lines(&out.get_output().stdout);
    let white = objects
        .iter()
        .find(|o| {
            o["input"]
                .as_str()
                .is_some_and(|i| i.ends_with("white.ARW"))
        })
        .expect("white");
    assert_eq!(white["status"], "error");
    assert_eq!(white["error"]["kind"], "settings");
    assert_eq!(names_in(&sb.default_export()), ["good_signature.jpg"]);
}

/// Owner's decision of 2026-10-29: ARRI's display LUT is not a rendering Yellowmoon offers, so
/// `--display` is gone. It is a usage error that says what to do, not clap's "unexpected argument",
/// whatever it was given and before any photo is read.
#[test]
fn the_display_option_is_gone_and_the_error_says_what_to_do() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    for words in [
        vec!["--display", "aces2"],
        vec!["--display", "arri"],
        vec!["--display", "arri-display:logc4:rec709-g24"],
        vec!["--display=bogus"],
    ] {
        sb.ym()
            .arg("render")
            .args(&words)
            .arg(&arw)
            .assert()
            .code(2)
            .stdout(predicate::str::is_empty())
            .stderr(predicate::str::contains(
                "--display was removed. Yellowmoon always uses its own display rendering, so there is nothing to choose. Leave the option out. To change how photos look, use --look.",
            ));
    }
    // With no value at all, the photo that follows is still a photo, and it is not read either.
    sb.ym()
        .args(["render", "--display"])
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--display was removed"));
    assert!(!sb.default_export().exists(), "nothing was rendered");
    // The option is not in the help any more.
    sb.ym()
        .args(["render", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--display").not());
    // Without it the run is the same as ever.
    sb.ym()
        .args(["render", "--jobs", "1"])
        .arg(&arw)
        .assert()
        .success();
}

/// `render -` waiting for a list that never comes: nothing would look at the interrupt flag, so the
/// first Ctrl-C has to end the process by itself (it used to need a second one).
#[cfg(unix)]
#[test]
fn the_first_ctrl_c_ends_a_run_that_is_waiting_for_stdin() {
    let sb = Sandbox::new();
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .args(["render", "-"])
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env(common::NO_AUTO_PROVIDERS.0, common::NO_AUTO_PROVIDERS.1)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    // The handler is installed in the first moments; stdin stays open, so the read blocks.
    std::thread::sleep(Duration::from_millis(1500));
    let kill = std::process::Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill");
    assert!(kill.success());
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("one Ctrl-C did not end a run that is waiting for stdin");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(130));
    drop(child.stdin.take());
}

#[test]
fn an_exposure_offset_that_passes_the_limit_is_limited_and_said_so() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["render", "--json", "--exposure-offset", "9"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "the exposure is limited to +4.00 EV",
        ));
    let o = &json_lines(&out.get_output().stdout)[0];
    assert_eq!(common::node(o, "input")["exposure"]["ev"], 4.0);
    // An offset that fits says nothing.
    sb.ym()
        .args(["render", "--overwrite", "--exposure-offset", "0.25"])
        .arg(&arw)
        .assert()
        .success()
        .stderr(predicate::str::contains("limited").not());
}

#[test]
fn a_machine_with_no_home_folder_is_a_usage_error_that_names_the_way_out() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    for args in [
        vec!["render"],
        vec!["info"],
        vec!["doctor"],
        vec!["looks", "list"],
    ] {
        let mut cmd = sb.ym();
        cmd.env_remove("YELLOWMOON_HOME")
            .env_remove("HOME")
            .env_remove("XDG_DATA_HOME")
            .args(&args);
        if args[0] != "doctor" && args[0] != "looks" {
            cmd.arg(&arw);
        }
        cmd.assert()
            .code(2)
            .stderr(predicate::str::contains("Set YELLOWMOON_HOME"));
    }
}

#[test]
fn choices_are_accepted_in_any_case_like_looks_are() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args([
            "render",
            "--json",
            "--format",
            "TIFF",
            "--colour-space",
            "P3",
            "--compression",
            "Zip",
            "--light",
            "Day",
            "--wb",
            "AUTO",
        ])
        .arg(&arw)
        .assert()
        .success();
    let o = &json_lines(&out.get_output().stdout)[0];
    assert!(o["output"].as_str().expect("output").ends_with(".tif"));
    assert_eq!(common::node(o, "input")["light_source"], "day");
}

/// `--auto-crop` (`docs/20-auto-crop.md` section 8): the record carries the proposal, the photo's
/// crop is its first frame and angle with `auto`, and the same crop given as settings (what the
/// Crop panel's Done sends) writes the same bytes.
#[test]
fn auto_crop_applies_the_panels_answer_and_reports_it() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let by_flag = sb.root().join("by flag");
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-crop=1:1", "--out"])
        .arg(&by_flag)
        .arg(&arw)
        .assert()
        .success();
    let record = json_lines(&out.get_output().stdout)
        .into_iter()
        .find(|o| o["status"] == "ok")
        .expect("a record");
    let proposal = &record["auto_crop"];
    let frames = proposal["frames"].as_array().expect("frames");
    assert!(!frames.is_empty() && frames.len() <= 3, "{proposal}");
    // Square on a 4:3 photo is never the whole photo, so there is a crop.
    let crop = &common::node(&record, "input")["crop"];
    assert_eq!(crop["auto"], true, "{crop}");
    assert_eq!(crop["aspect"], "square");
    assert_eq!(crop["rect"], frames[0]["rect"]);
    assert_eq!(crop["angle_deg"], proposal["angle_deg"]);

    let settings = sb.root().join("crop.json");
    std::fs::write(
        &settings,
        common::field_doc(json!({ "crop": crop })).to_string(),
    )
    .expect("write");
    let by_settings = sb.root().join("by settings");
    sb.ym()
        .args(["render", "--cpu", "--settings"])
        .arg(&settings)
        .arg("--out")
        .arg(&by_settings)
        .arg(&arw)
        .assert()
        .success();
    let name = "a_signature.jpg";
    assert_eq!(
        std::fs::read(by_flag.join(name)).expect("by flag"),
        std::fs::read(by_settings.join(name)).expect("by settings"),
        "the same crop, the same bytes"
    );
}

#[test]
fn auto_crop_keeps_the_whole_photo_as_no_crop_and_replaces_a_crop_from_settings() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let settings = sb.root().join("crop.json");
    std::fs::write(
        &settings,
        r#"{ "nodes": { "items": [ { "id": "input", "crop": { "rect": { "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5 }, "angle_deg": 0, "aspect": "free" } } ] } }"#,
    )
    .expect("write");
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-crop", "--settings"])
        .arg(&settings)
        .arg(&arw)
        .assert()
        .success();
    let record = json_lines(&out.get_output().stdout)
        .into_iter()
        .find(|o| o["status"] == "ok")
        .expect("a record");
    let proposal = &record["auto_crop"];
    let first = &proposal["frames"][0]["rect"];
    let whole = first["width"].as_f64() == Some(1.0)
        && first["height"].as_f64() == Some(1.0)
        && proposal["angle_deg"].as_f64() == Some(0.0);
    if whole {
        assert_eq!(
            common::node(&record, "input")["crop"],
            Value::Null,
            "{record}"
        );
    } else {
        assert_eq!(common::node(&record, "input")["crop"]["rect"], *first);
    }
}

#[test]
fn auto_crop_takes_only_known_aspects() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    sb.ym()
        .args(["render", "--auto-crop=2:3"])
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("2:3 is not an aspect"));
}
