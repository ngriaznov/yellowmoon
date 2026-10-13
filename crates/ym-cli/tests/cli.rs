//! End-to-end tests of the `ym` binary that are not about one command: the version, `--help` of
//! every command (snapshots), usage errors and `doctor`.

mod common;

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use common::Sandbox;
use predicates::prelude::*;
use serde_json::Value;

fn ym() -> Command {
    cargo_bin_cmd!("ym")
}

#[test]
fn version_prints_version_pipeline_and_git_sha() {
    ym().arg("--version")
        .assert()
        .success()
        .stdout(
            predicate::str::is_match(
                r"^ym \d+\.\d+\.\d+ \(pipeline \d+\.\d+\.\d+, ([0-9a-f]{7,40}|unknown)\)\n$",
            )
            .expect("valid regex"),
        )
        .stderr(predicate::str::is_empty());
}

#[test]
fn version_reports_the_workspace_and_pipeline_versions() {
    ym().arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("ym 0.0.1 (pipeline 0.1.0, "));
}

#[test]
fn no_subcommand_is_a_usage_error() {
    ym().assert()
        .code(2)
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    ym().arg("frobnicate").assert().code(2);
}

/// The text of `ym <args> --help`.
fn help(args: &[&str]) -> String {
    let out = ym().args(args).arg("--help").assert().success();
    String::from_utf8(out.get_output().stdout.clone()).expect("utf-8")
}

#[test]
fn help_lists_every_top_level_command() {
    let stdout = help(&[]);
    for command in ["render", "info", "looks", "doctor"] {
        assert!(stdout.contains(command), "{command}:\n{stdout}");
    }
    assert!(stdout.contains("Exit codes:"), "{stdout}");
}

#[test]
fn help_snapshots_of_every_command() {
    insta::assert_snapshot!("help_ym", help(&[]));
    insta::assert_snapshot!("help_render", help(&["render"]));
    insta::assert_snapshot!("help_info", help(&["info"]));
    insta::assert_snapshot!("help_doctor", help(&["doctor"]));
    insta::assert_snapshot!("help_looks", help(&["looks"]));
    insta::assert_snapshot!("help_looks_list", help(&["looks", "list"]));
    insta::assert_snapshot!("help_looks_import", help(&["looks", "import"]));
    insta::assert_snapshot!("help_looks_verify", help(&["looks", "verify"]));
    insta::assert_snapshot!("help_looks_default", help(&["looks", "default"]));
    insta::assert_snapshot!("help_looks_bake", help(&["looks", "bake"]));
}

#[test]
fn the_help_names_the_command_ym_and_never_the_old_name() {
    for args in [
        vec![],
        vec!["render"],
        vec!["info"],
        vec!["looks"],
        vec!["looks", "import"],
        vec!["doctor"],
    ] {
        let text = help(&args);
        assert!(text.contains("ym"), "{args:?}");
        assert!(
            !text.to_lowercase().contains("yellowmoon render"),
            "{args:?}:\n{text}"
        );
        assert!(!text.contains("yellowmoon info"), "{args:?}");
    }
}

#[test]
fn the_color_space_option_has_both_spellings() {
    let text = help(&["render"]);
    assert!(text.contains("--colour-space"), "{text}");
    assert!(text.contains("color-space"), "{text}");
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    for flag in ["--colour-space", "--color-space"] {
        sb.ym()
            .args(["render", "--format", "png", flag, "p3", "--overwrite"])
            .arg(&arw)
            .assert()
            .success();
    }
}

#[test]
fn doctor_says_what_it_includes_and_lists_the_parts() {
    let sb = Sandbox::new();
    let out = sb.ym().arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    for needle in [
        "does not include photos, file names, locations",
        "Version",
        "ym 0.0.1 (pipeline 0.1.0, ",
        "GPU",
        "Render path",
        "Model weights",
        "Camera profiles",
        "sony-ilce-7rm4 1.0.0",
        "Looks",
        "1 built in, 0 imported",
        "Preferences",
        "not created yet",
        "Data folder",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
    // Reading the report writes nothing.
    assert!(!sb.home().join("preferences.json").exists());
}

#[test]
fn doctor_json_has_the_same_parts() {
    let sb = Sandbox::new();
    let out = sb.ym().args(["doctor", "--json"]).assert().success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    assert!(
        v["includes"]
            .as_str()
            .expect("includes")
            .contains("does not include")
    );
    assert_eq!(v["pipeline_version"], "0.1.0");
    assert!(v["gpu"]["available"].is_boolean());
    if v["gpu"]["available"] == true {
        assert!(v["gpu"]["name"].is_string() && v["gpu"]["limits"]["max_buffer_size"].is_u64());
    } else {
        assert!(v["gpu"]["reason"].is_string());
    }
    assert_eq!(v["render_path"]["demosaic"], "classical");
    assert_eq!(v["model_weights"], serde_json::json!([]));
    assert!(v["illuminant_models"].as_array().expect("models").len() >= 10);
    for model in v["illuminant_models"].as_array().expect("models") {
        assert_eq!(model["sha256"].as_str().expect("hash").len(), 64);
    }
    assert_eq!(v["profiles"].as_array().expect("profiles").len(), 11);
    assert_eq!(v["looks"]["built_in"], 1);
    assert_eq!(v["looks"]["imported"], 0);
}

#[test]
fn doctor_without_a_gpu_says_so() {
    let sb = Sandbox::new();
    sb.ym()
        .env("WGPU_BACKEND", "dx12")
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("none, CPU fallback"));
}
