//! Apple Auto on the command line (`docs/19` section 8, `docs/10`): `--apple-auto on|off`, the
//! `auto_source` and `apple_auto` of `render --json`, the Apple Auto row and object of `ym info`, the
//! `apple_auto` of `ym doctor`, `--contrast auto` and `--saturation auto`, the XMP of an export and
//! determinism, with the test advisor (`YELLOWMOON_TEST_APPLE_AUTO=fake`) on every machine.
//!
//! The test advisor answers from the picture's statistics (`ym_core::apple_auto::FakeAdvisor`): on
//! the synthetic photos it asks for a darker picture with more contrast, so the answer changes the
//! exposure and the contrast and leaves the white and the saturation alone.

mod common;

use std::path::{Path, PathBuf};

use common::{Sandbox, json_lines, text};
use serde_json::{Value, json};
use ym_core::apple_auto::{AutoSteps, TABLE};

/// The photo of these tests: the 160 x 120 gradient most CLI tests use.
fn photo(sb: &Sandbox) -> PathBuf {
    sb.arw("a.ARW")
}

/// The steps of an `apple_auto` object.
fn steps_of(apple_auto: &Value) -> AutoSteps {
    let step = |key: &str| -> i8 {
        let v = apple_auto["steps"][key]
            .as_i64()
            .unwrap_or_else(|| panic!("steps.{key} in {apple_auto}"));
        i8::try_from(v).expect("a step")
    };
    AutoSteps {
        exposure: step("exposure"),
        white: step("white"),
        contrast: step("contrast"),
        saturation: step("saturation"),
    }
}

/// `ym render --json` of one photo with `flags`, through `cmd`, into `out`.
fn render_json(mut cmd: assert_cmd::Command, arw: &Path, flags: &[&str], out: &Path) -> Value {
    let assert = cmd
        .args(["render", "--json", "--quiet", "--cpu", "--overwrite"])
        .args(flags)
        .arg("--out")
        .arg(out)
        .arg(arw)
        .assert()
        .success();
    let mut lines = json_lines(&assert.get_output().stdout);
    assert_eq!(lines.len(), 1, "one object");
    lines.remove(0)
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn output_of(v: &Value) -> PathBuf {
    PathBuf::from(v["output"].as_str().expect("an output path"))
}

#[test]
fn render_json_says_apple_made_the_auto_values_and_what_it_answered() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let v = render_json(sb.ym_fake(), &arw, &[], &sb.root().join("out"));
    assert_eq!(v["auto_source"], "apple");
    assert_eq!(v["auto"]["source"], "apple");
    let a = &v["apple_auto"];
    assert_eq!(a["state"], "answered", "{a}");
    assert_eq!(a["reason"], Value::Null);
    assert_eq!(a["model"], "Test advisor");
    assert_eq!(a["os"], "test");
    assert_eq!(a["prompt"], 1);
    assert_eq!(a["table"], 1);
    assert_eq!(a["scene"], "Test picture");
    let steps = steps_of(a);
    // The test advisor's answer on this photo: darker, more contrast.
    assert_eq!(
        steps,
        AutoSteps {
            exposure: -1,
            white: 0,
            contrast: 1,
            saturation: 0
        },
        "{a}"
    );
    // The answer is in the Auto values and in the settings the file records, pinned to them.
    assert_eq!(v["auto"]["contrast"], TABLE.contrast[3]);
    assert_eq!(v["auto"]["saturation"], 0.0);
    assert_eq!(
        v["settings"]["contrast"],
        json!({ "kind": "manual", "value": 15.0 })
    );
    assert_eq!(v["settings"]["exposure"]["ev"], v["auto"]["exposure_ev"]);
    // The same photo with Yellowmoon's own Auto: the exposure is the table's step brighter.
    let rule = render_json(
        sb.ym_fake(),
        &arw,
        &["--apple-auto", "off"],
        &sb.root().join("rule"),
    );
    let (apple_ev, rule_ev) = (
        v["auto"]["exposure_ev"].as_f64().expect("ev"),
        rule["auto"]["exposure_ev"].as_f64().expect("ev"),
    );
    assert!(
        (apple_ev - (rule_ev + TABLE.exposure_ev[2])).abs() < 1e-9,
        "{apple_ev} against {rule_ev}"
    );
    // The white was kept, so the Kelvin and the tint are the rule's.
    assert_eq!(v["auto"]["kelvin"], rule["auto"]["kelvin"]);
    assert_eq!(v["auto"]["tint"], rule["auto"]["tint"]);
    // Keys: `auto_source` and `apple_auto` after `auto`, before `settings`.
    let line = serde_json::to_string(&v).expect("json");
    assert!(line.contains("\"auto_source\":\"apple\""));
}

#[test]
fn apple_auto_off_is_the_rule_and_render_json_has_no_apple_auto() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let assert = sb
        .ym_fake()
        .args(["render", "--json", "--cpu", "--apple-auto", "off", "--out"])
        .arg(sb.root().join("out"))
        .arg(&arw)
        .assert()
        .success();
    assert!(
        !text(&assert.get_output().stderr).contains("Apple Auto"),
        "{}",
        text(&assert.get_output().stderr)
    );
    let v = json_lines(&assert.get_output().stdout).remove(0);
    assert_eq!(v["auto_source"], "rule");
    assert_eq!(v["auto"]["apple"], json!({ "state": "off" }));
    assert!(v.get("apple_auto").is_none(), "{v}");
    assert_eq!(v["auto"]["contrast"], 0.0);
    assert_eq!(
        v["settings"]["contrast"],
        json!({ "kind": "manual", "value": 0.0 })
    );
    // The key order of docs/10, with `auto_source` after `auto`.
    let line = text(&assert.get_output().stdout);
    let at = |key: &str| line.find(&format!("\"{key}\":")).expect(key);
    assert!(at("auto") < at("auto_source") && at("auto_source") < at("settings"));
}

#[test]
fn apple_auto_on_where_it_is_not_available_says_so_once_and_uses_the_rule() {
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw_sized("b.ARW", 128, 96, 7);
    // What this computer says about Apple Auto (a Mac with Apple Intelligence has nothing to check).
    let doctor = sb.ym().args(["doctor", "--json"]).assert().success();
    let doctor: Value = serde_json::from_slice(&doctor.get_output().stdout).expect("JSON");
    if doctor["apple_auto"]["available"] == true {
        eprintln!("Apple Auto is available here; the notice is not printed");
        return;
    }
    let assert = sb
        .ym()
        .args(["render", "--json", "--cpu", "--apple-auto", "on", "--out"])
        .arg(sb.root().join("out"))
        .arg(sb.photos())
        .assert()
        .code(0);
    let stderr = text(&assert.get_output().stderr);
    let notice = "Apple Auto is not available on this computer (";
    assert_eq!(stderr.matches(notice).count(), 1, "once per run: {stderr}");
    assert!(
        stderr.contains("). Auto uses Yellowmoon's own settings."),
        "{stderr}"
    );
    if cfg!(not(target_os = "macos")) {
        assert!(
            stderr.contains(
                "Warning: Apple Auto is not available on this computer (this is not a Mac). Auto uses Yellowmoon's own settings."
            ),
            "{stderr}"
        );
    }
    for v in json_lines(&assert.get_output().stdout) {
        assert_eq!(v["status"], "ok");
        assert_eq!(v["auto_source"], "rule");
        assert_eq!(v["auto"]["apple"]["state"], "unavailable");
        assert!(v.get("apple_auto").is_none(), "{v}");
    }
    // `ym info` says it too, and its object says why.
    let info = sb
        .ym()
        .args(["info", "--json", "--apple-auto", "on"])
        .arg(sb.photos().join("a.ARW"))
        .assert()
        .success();
    assert_eq!(text(&info.get_output().stderr).matches(notice).count(), 1);
    let v = json_lines(&info.get_output().stdout).remove(0);
    assert_eq!(v["apple_auto"]["state"], "unavailable");
    assert_eq!(v["apple_auto"]["reason"], doctor["apple_auto"]["reason"]);
    // Without the option nothing is said: the app's setting is ignored where Apple Auto cannot run.
    let quiet = sb
        .ym()
        .args(["render", "--cpu", "--overwrite", "--out"])
        .arg(sb.root().join("out"))
        .arg(sb.photos().join("a.ARW"))
        .assert()
        .success();
    assert!(!text(&quiet.get_output().stderr).contains("Apple Auto"));
}

#[test]
fn the_apple_auto_preference_decides_unless_the_run_says_otherwise() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    sb.set_apple_auto_preference(false);
    let off = render_json(sb.ym_fake(), &arw, &[], &sb.root().join("pref-off"));
    assert_eq!(off["auto_source"], "rule");
    assert_eq!(off["auto"]["apple"]["state"], "off");
    assert!(off.get("apple_auto").is_none());
    let on = render_json(
        sb.ym_fake(),
        &arw,
        &["--apple-auto", "on"],
        &sb.root().join("run-on"),
    );
    assert_eq!(on["auto_source"], "apple");
    assert_eq!(on["apple_auto"]["state"], "answered");
    // And `ym doctor` shows the preference.
    let doctor = sb.ym_fake().args(["doctor", "--json"]).assert().success();
    let doctor: Value = serde_json::from_slice(&doctor.get_output().stdout).expect("JSON");
    assert_eq!(doctor["apple_auto"]["preference"], false);
    let text_report = sb.ym_fake().arg("doctor").assert().success();
    let text_report = text(&text_report.get_output().stdout);
    assert!(
        text_report.contains("table 1 · turned off in Preferences"),
        "{text_report}"
    );
}

#[test]
fn info_json_has_the_apple_auto_object_and_the_auto_values_in_effect() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let info = sb
        .ym_fake()
        .args(["info", "--json"])
        .arg(&arw)
        .assert()
        .success();
    let v = json_lines(&info.get_output().stdout).remove(0);
    let a = &v["apple_auto"];
    assert_eq!(
        a.as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "model", "os", "prompt", "reason", "scene", "state", "steps", "table"
        ],
        "{a}"
    );
    assert_eq!(a["state"], "answered");
    assert_eq!(a["model"], "Test advisor");
    assert_eq!(v["auto_source"], "apple");
    assert_eq!(v["auto"]["source"], "apple");
    assert_eq!(v["auto"]["contrast"], 15.0);
    assert_eq!(v["auto"]["saturation"], 0.0);
    assert_eq!(v["auto"]["apple"]["answer"]["steps"], a["steps"]);
    // `info` and `render` report the same photo the same way.
    let render = render_json(sb.ym_fake(), &arw, &[], &sb.root().join("out"));
    for key in ["auto", "auto_source", "apple_auto", "settings"] {
        assert_eq!(v[key], render[key], "{key}");
    }
    // Off, and where it is not available, the object says so with no steps and no scene.
    let off = sb
        .ym_fake()
        .args(["info", "--json", "--apple-auto", "off"])
        .arg(&arw)
        .assert()
        .success();
    let off = json_lines(&off.get_output().stdout).remove(0);
    assert_eq!(
        off["apple_auto"],
        json!({ "state": "off", "reason": null, "model": null, "os": null, "prompt": 1, "table": 1 })
    );
    assert_eq!(off["auto_source"], "rule");
    if cfg!(not(target_os = "macos")) {
        let here = sb
            .ym()
            .args(["info", "--json"])
            .arg(&arw)
            .assert()
            .success();
        let here = json_lines(&here.get_output().stdout).remove(0);
        assert_eq!(
            here["apple_auto"],
            json!({ "state": "unavailable", "reason": "not_a_mac", "model": null, "os": null, "prompt": 1, "table": 1 })
        );
        assert_eq!(
            here["auto"]["apple"],
            json!({ "state": "unavailable", "reason": "not_a_mac" })
        );
    }
}

#[test]
fn info_text_has_an_apple_auto_row() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let row = |cmd: assert_cmd::Command, flags: &[&str]| -> String {
        let mut cmd = cmd;
        let out = cmd.arg("info").args(flags).arg(&arw).assert().success();
        let out = text(&out.get_output().stdout);
        out.lines()
            .find(|l| l.trim_start().starts_with("Apple Auto"))
            .unwrap_or_else(|| panic!("no Apple Auto row:\n{out}"))
            .trim_start()
            .trim_start_matches("Apple Auto")
            .trim()
            .to_owned()
    };
    assert_eq!(
        row(sb.ym_fake(), &[]),
        "Darker -0.3 EV · contrast +15 · \"Test picture\""
    );
    assert_eq!(row(sb.ym_fake(), &["--apple-auto", "off"]), "Off");
    if cfg!(not(target_os = "macos")) {
        assert_eq!(row(sb.ym(), &[]), "Not available (this is not a Mac)");
    }
    // The row sits after the white balance and before the noise model, as on the Automatic card.
    let out = sb.ym_fake().arg("info").arg(&arw).assert().success();
    let out = text(&out.get_output().stdout);
    let at = |label: &str| out.find(label).unwrap_or_else(|| panic!("{label}:\n{out}"));
    assert!(at("Auto white") < at("Apple Auto") && at("Apple Auto") < at("Noise model"));
}

#[test]
fn doctor_reports_apple_auto_and_the_model() {
    let sb = Sandbox::new();
    let out = sb.ym_fake().args(["doctor", "--json"]).assert().success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    assert_eq!(
        v["apple_auto"],
        json!({
            "available": true, "reason": null, "model": "Test advisor", "os": "test",
            "preference": true, "prompt": 1, "table": 1
        })
    );
    let out = sb.ym_fake().arg("doctor").assert().success();
    let report = text(&out.get_output().stdout);
    assert!(
        report.contains(
            "Apple Auto         available · Apple Intelligence · Test advisor · macOS test · prompt 1 · table 1\n"
        ),
        "{report}"
    );
    assert!(
        report.contains("whether Apple Auto is available"),
        "{report}"
    );
    if cfg!(not(target_os = "macos")) {
        let out = sb.ym().args(["doctor", "--json"]).assert().success();
        let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
        assert_eq!(
            v["apple_auto"],
            json!({
                "available": false, "reason": "not_a_mac", "model": null, "os": null,
                "preference": true, "prompt": 1, "table": 1
            })
        );
        let out = sb.ym().arg("doctor").assert().success();
        let report = text(&out.get_output().stdout);
        assert!(
            report.contains("Apple Auto         not available (this is not a Mac)\n"),
            "{report}"
        );
    }
    // Reading the report writes nothing.
    assert!(!sb.home().join("preferences.json").exists());
}

#[test]
fn contrast_and_saturation_take_auto() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let file = sb.root().join("settings.json");
    std::fs::write(&file, r#"{ "contrast": 30, "saturation": 10 }"#).expect("write");
    let settings = file.to_str().expect("utf-8");
    // `auto` puts back what a settings file set by hand: the Auto value, pinned in the settings.
    let flags = [
        "--settings",
        settings,
        "--contrast",
        "auto",
        "--saturation",
        "AUTO",
    ];
    let apple = render_json(sb.ym_fake(), &arw, &flags, &sb.root().join("apple"));
    assert_eq!(
        apple["settings"]["contrast"],
        json!({ "kind": "manual", "value": 15.0 })
    );
    assert_eq!(
        apple["settings"]["saturation"],
        json!({ "kind": "manual", "value": 0.0 })
    );
    let mut rule_flags = flags.to_vec();
    rule_flags.extend(["--apple-auto", "off"]);
    let rule = render_json(sb.ym_fake(), &arw, &rule_flags, &sb.root().join("rule"));
    assert_eq!(
        rule["settings"]["contrast"],
        json!({ "kind": "manual", "value": 0.0 })
    );
    assert_eq!(
        rule["settings"]["saturation"],
        json!({ "kind": "manual", "value": 0.0 })
    );
    // A number stays a manual value, whatever Apple Auto says.
    let manual = render_json(
        sb.ym_fake(),
        &arw,
        &["--contrast", "-20", "--saturation", "12"],
        &sb.root().join("manual"),
    );
    assert_eq!(
        manual["settings"]["contrast"],
        json!({ "kind": "manual", "value": -20.0 })
    );
    assert_eq!(
        manual["settings"]["saturation"],
        json!({ "kind": "manual", "value": 12.0 })
    );
    assert_eq!(
        manual["auto"]["contrast"], 15.0,
        "the Auto value is still Apple's"
    );
    // Anything else is a usage error.
    for bad in ["automatic", "101", "x"] {
        sb.ym()
            .args(["render", "--contrast", bad])
            .arg(&arw)
            .assert()
            .code(2);
        sb.ym()
            .args(["render", "--saturation", bad])
            .arg(&arw)
            .assert()
            .code(2);
    }
}

#[test]
fn an_export_records_which_auto_made_it_in_its_xmp() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let xmp_of = |v: &Value| String::from_utf8_lossy(&bytes(&output_of(v))).into_owned();
    let apple = render_json(sb.ym_fake(), &arw, &[], &sb.root().join("apple"));
    let xmp = xmp_of(&apple);
    assert!(xmp.contains("<ym:Auto>apple</ym:Auto>"), "ym:Auto");
    assert!(
        xmp.contains(
            "<ym:AppleAuto>{&quot;model&quot;:&quot;Test advisor&quot;,&quot;os&quot;:&quot;test&quot;,&quot;prompt&quot;:1,&quot;result&quot;:&quot;answered&quot;,&quot;scene&quot;:&quot;Test picture&quot;,&quot;steps&quot;:{&quot;contrast&quot;:1,&quot;exposure&quot;:-1,&quot;saturation&quot;:0,&quot;white&quot;:0},&quot;table&quot;:1}</ym:AppleAuto>"
        ),
        "ym:AppleAuto"
    );
    let rule = render_json(
        sb.ym_fake(),
        &arw,
        &["--apple-auto", "off"],
        &sb.root().join("rule"),
    );
    let xmp = xmp_of(&rule);
    assert!(xmp.contains("<ym:Auto>rule</ym:Auto>"));
    assert!(!xmp.contains("ym:AppleAuto"));
    assert_ne!(bytes(&output_of(&apple)), bytes(&output_of(&rule)));
}

#[test]
fn two_renders_with_apple_auto_are_byte_identical_and_their_settings_reproduce_them() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let png = ["--format", "png", "--bits", "16"];
    let first = render_json(sb.ym_fake(), &arw, &png, &sb.root().join("first"));
    let second = render_json(sb.ym_fake(), &arw, &png, &sb.root().join("second"));
    assert_eq!(first["apple_auto"], second["apple_auto"]);
    assert_eq!(bytes(&output_of(&first)), bytes(&output_of(&second)));
    // The settings the file records, fed back with the same Apple Auto, make the same file (an Auto
    // white balance stays Auto in them, and its nudge comes from the answer again).
    let settings = sb.root().join("again.json");
    std::fs::write(&settings, first["settings"].to_string()).expect("write");
    let mut flags = png.to_vec();
    flags.extend(["--settings", settings.to_str().expect("utf-8")]);
    let again = render_json(sb.ym_fake(), &arw, &flags, &sb.root().join("again"));
    assert_eq!(again["settings"], first["settings"]);
    assert_eq!(bytes(&output_of(&again)), bytes(&output_of(&first)));
}

#[test]
fn the_json_of_a_file_with_apple_auto_has_this_shape() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let v = render_json(sb.ym_fake(), &arw, &[], &sb.root().join("out"));
    let keep = |key: &str| (key.to_owned(), common::shape(&v[key]));
    let picked: serde_json::Map<String, Value> = ["auto", "auto_source", "apple_auto"]
        .map(keep)
        .into_iter()
        .collect();
    insta::assert_json_snapshot!("render_json_shape_apple_auto", Value::Object(picked));
}
