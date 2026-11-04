//! End-to-end tests of the `ym` binary that are not about one command: the version, `--help` of
//! every command (snapshots), usage errors and `doctor`.

mod common;

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use common::{Sandbox, data_dir, shape};
use predicates::prelude::*;
use serde_json::Value;

/// The binary as it is, but with Apple Auto not available, as for every test of this crate
/// (`common`).
fn ym() -> Command {
    let mut cmd = cargo_bin_cmd!("ym");
    cmd.env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env(common::NO_AUTO_PROVIDERS.0, common::NO_AUTO_PROVIDERS.1);
    cmd
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
        .stdout(predicate::str::starts_with(format!(
            "ym {} (pipeline 0.10.0, ",
            env!("CARGO_PKG_VERSION")
        )));
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
    for command in ["render", "info", "looks", "lenses", "doctor"] {
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
    insta::assert_snapshot!("help_lenses", help(&["lenses"]));
    insta::assert_snapshot!("help_lenses_list", help(&["lenses", "list"]));
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
    let version_line = format!("ym {} (pipeline 0.10.0, ", env!("CARGO_PKG_VERSION"));
    for needle in [
        "does not include photos, file names, locations",
        "Version",
        version_line.as_str(),
        "GPU",
        "Render path",
        "Model weights",
        "Masks",
        "Camera profiles",
        "sony-ilce-7rm4 1.2.0",
        "Looks",
        "2 built in, 0 imported",
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
    assert_eq!(v["pipeline_version"], "0.10.0");
    assert!(v["gpu"]["available"].is_boolean());
    if v["gpu"]["available"] == true {
        assert!(v["gpu"]["name"].is_string() && v["gpu"]["limits"]["max_buffer_size"].is_u64());
    } else {
        assert!(v["gpu"]["reason"].is_string());
    }
    assert_eq!(v["render_path"]["demosaic"], "classical");
    // The weight files are the bokeh map's and Auto crop's model, each with the hash its SHA256SUMS gives, in path
    // order, and SAM 2's when its file is there: it is too big to commit (`data/models/sam2/.gitignore`), so a
    // checkout has it only after `tools/fetch` or `tools/convert`, and the built app always does.
    let sha256_of = |model: &str| -> String {
        let sums = std::fs::read_to_string(data_dir().join(format!("models/{model}/SHA256SUMS")))
            .expect("sums");
        sums.split_whitespace().next().expect("hash").to_owned()
    };
    let mut weights = vec![
        serde_json::json!({ "file": "bokeh/bokehnet.safetensors", "sha256": sha256_of("bokeh") }),
        serde_json::json!({ "file": "crop/cropnet.safetensors", "sha256": sha256_of("crop") }),
    ];
    if data_dir()
        .join("models/sam2/sam2.1_hiera_small.f16.safetensors")
        .is_file()
    {
        weights.push(serde_json::json!({
            "file": "sam2/sam2.1_hiera_small.f16.safetensors",
            "sha256": sha256_of("sam2"),
        }));
    }
    assert_eq!(v["model_weights"], serde_json::Value::Array(weights));
    assert!(v["illuminant_models"].as_array().expect("models").len() >= 10);
    for model in v["illuminant_models"].as_array().expect("models") {
        assert_eq!(model["sha256"].as_str().expect("hash").len(), 64);
    }
    assert_eq!(v["profiles"].as_array().expect("profiles").len(), 11);
    assert_eq!(v["looks"]["built_in"], 2);
    assert_eq!(v["looks"]["imported"], 0);
}

/// The `Masks` row and the `masks` object of `ym doctor` (`docs/10` *doctor*): the kinds `mask_kinds()` gives the app,
/// in the order of the Add mask menu, and the reason for each that this computer cannot make. The sandbox's `ym` says
/// Vision is not supported (`YELLOWMOON_TEST_VISION=none`), so the line is the same on a Mac that supports it.
#[test]
fn doctor_lists_the_masks_this_computer_can_make_and_why_not_the_others() {
    let sb = Sandbox::new();
    let out = sb.ym().arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    let line = text
        .lines()
        .find(|l| l.starts_with("Masks"))
        .unwrap_or_else(|| panic!("no Masks row in:\n{text}"))
        .trim_start_matches("Masks")
        .trim()
        .to_owned();
    // Object is there where the SAM weights are (`data/models/sam2`, fetched by `tools/fetch/sam2.sh`; CI's other jobs have none): the doctor
    // reads them and checks them against `SHA256SUMS`, and so does this.
    let object =
        ym_core::pipeline::ObjectModel::probe(&data_dir()).is_some_and(|m| m.sam().is_ok());
    let rest = if cfg!(target_os = "macos") {
        "Subject and People (this computer can't make them)"
    } else {
        "Subject and People (this is not a Mac that supports them)"
    };
    let (made, not_made) = if object {
        (
            "Object, Background, Linear, Radial, Brush, Colour range, Brightness range",
            rest.to_owned(),
        )
    } else {
        (
            "Background, Linear, Radial, Brush, Colour range, Brightness range",
            format!("Object (the SAM weights are missing), {rest}"),
        )
    };
    assert_eq!(line, format!("{made} · not available: {not_made}"));
    // The row follows the model weights and comes before the illuminant models.
    let at = |label: &str| text.find(label).unwrap_or_else(|| panic!("no {label}"));
    assert!(at("Model weights") < at("Masks") && at("Masks") < at("Illuminant models"));

    let out = sb.ym().args(["doctor", "--json"]).assert().success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    let mut kinds = vec![];
    if object {
        kinds.push("object");
    }
    kinds.extend([
        "background",
        "linear",
        "radial",
        "brush",
        "color_range",
        "brightness_range",
    ]);
    assert_eq!(v["masks"]["kinds"], serde_json::json!(kinds));
    let reason = if cfg!(target_os = "macos") {
        "not_supported"
    } else {
        "not_a_mac"
    };
    let mut unavailable = serde_json::json!({ "subject": reason, "people": reason });
    if !object {
        unavailable["object"] = serde_json::json!("weights_missing");
    }
    assert_eq!(v["masks"]["unavailable"], unavailable);
}

/// A computer that supports the Vision requests (the fake: `YELLOWMOON_TEST_VISION=fake`) lists Subject and People after
/// Object in the Masks line and in `masks.kinds`, as the Add mask menu does, and gives them no reason.
#[test]
fn doctor_lists_subject_and_people_where_this_computer_supports_them() {
    let sb = Sandbox::new();
    let object =
        ym_core::pipeline::ObjectModel::probe(&data_dir()).is_some_and(|m| m.sam().is_ok());
    let out = sb.ym_fake_vision().arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    let line = text
        .lines()
        .find(|l| l.starts_with("Masks"))
        .unwrap_or_else(|| panic!("no Masks row in:\n{text}"))
        .trim_start_matches("Masks")
        .trim()
        .to_owned();
    let all = "Subject, People, Background, Linear, Radial, Brush, Colour range, Brightness range";
    let expected = if object {
        format!("Object, {all}")
    } else {
        format!("{all} · not available: Object (the SAM weights are missing)")
    };
    assert_eq!(line, expected);

    let out = sb
        .ym_fake_vision()
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    let mut kinds = vec![];
    if object {
        kinds.push("object");
    }
    kinds.extend([
        "subject",
        "people",
        "background",
        "linear",
        "radial",
        "brush",
        "color_range",
        "brightness_range",
    ]);
    assert_eq!(v["masks"]["kinds"], serde_json::json!(kinds));
    let unavailable = if object {
        serde_json::json!({})
    } else {
        serde_json::json!({ "object": "weights_missing" })
    };
    assert_eq!(v["masks"]["unavailable"], unavailable);
}

/// `ym doctor` reports Subject and People as this machine answers: on a Mac with Vision it asks Vision (the probe runs
/// each request on a 64 x 64 picture); the answer is one of the four wire names, and Subject and People each have
/// theirs. On a runner that is a virtual Mac the requests may not run: the test prints the answer plainly and passes.
#[test]
fn doctor_reports_subject_and_people_as_this_machine_answers() {
    let sb = Sandbox::new();
    let out = sb
        .ym_system_vision()
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    let kinds: Vec<&str> = v["masks"]["kinds"]
        .as_array()
        .expect("kinds")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for kind in ["subject", "people"] {
        let supported = kinds.contains(&kind);
        let why = v["masks"]["unavailable"][kind].as_str();
        eprintln!(
            "[vision] {kind} on this computer: {}",
            if supported {
                "supported"
            } else {
                why.unwrap_or("?")
            }
        );
        // Exactly one of the two lists has the kind, and a reason is a wire name of the spec.
        assert_eq!(supported, why.is_none(), "{kind}: {v}");
        if let Some(why) = why {
            assert!(
                ["not_a_mac", "macos_too_old", "not_supported"].contains(&why),
                "{kind}: {why}"
            );
        }
        if !cfg!(target_os = "macos") {
            assert_eq!(why, Some("not_a_mac"), "{kind}");
        }
    }
    // The supported ones are after Object and before Background, in the menu's order.
    if let (Some(s), Some(p), Some(b)) = (
        kinds.iter().position(|k| *k == "subject"),
        kinds.iter().position(|k| *k == "people"),
        kinds.iter().position(|k| *k == "background"),
    ) {
        assert!(s < p && p < b, "{kinds:?}");
    }
    if std::env::var_os("YM_REQUIRE_VISION").is_some() {
        assert!(
            kinds.contains(&"subject") && kinds.contains(&"people"),
            "YM_REQUIRE_VISION is set and this computer does not support the Vision requests: {v}"
        );
    }
}

#[test]
fn doctor_json_shape_snapshot_on_a_machine_with_no_adapter() {
    // No Direct3D 12 here whatever the machine is, so the GPU section is the same everywhere; the
    // test advisor of Apple Auto makes its section the same everywhere too.
    let sb = Sandbox::new();
    let out = sb
        .ym_fake()
        .env("WGPU_BACKEND", "dx12")
        .env_remove("YELLOWMOON_GPU")
        .args(["doctor", "--json"])
        .assert()
        .success();
    let mut v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    // Where the SAM weights are, Object is in `kinds` and not in `unavailable`; the shape is the one of a machine without them.
    if let Some(unavailable) = v["masks"]["unavailable"].as_object_mut() {
        unavailable
            .entry("object")
            .or_insert_with(|| serde_json::json!("weights_missing"));
    }
    insta::assert_json_snapshot!("doctor_json_shape_no_adapter", shape(&v));
}

#[test]
fn doctor_says_what_the_graphics_policy_comes_to() {
    let sb = Sandbox::new();
    let doctor = |gpu: Option<&str>, backend: &str| -> (Value, String) {
        let run = |json: bool| {
            let mut cmd = sb.ym();
            cmd.env("WGPU_BACKEND", backend)
                .env_remove("YELLOWMOON_GPU");
            if let Some(gpu) = gpu {
                cmd.env("YELLOWMOON_GPU", gpu);
            }
            cmd.arg("doctor");
            if json {
                cmd.arg("--json");
            }
            let out = cmd.assert().success();
            String::from_utf8(out.get_output().stdout.clone()).expect("utf-8")
        };
        (serde_json::from_str(&run(true)).expect("JSON"), run(false))
    };

    // Nothing set and no adapter: the default policy, the CPU, and why.
    let (v, text) = doctor(None, "dx12");
    assert_eq!(v["gpu_policy"]["setting"], "auto");
    assert_eq!(v["gpu_policy"]["source"], "default");
    assert_eq!(v["gpu_policy"]["uses_gpu"], false);
    assert!(
        v["gpu_policy"]["reason"]
            .as_str()
            .is_some_and(|r| r.contains("no adapter"))
    );
    assert_eq!(v["gpu_policy"]["stages"], serde_json::json!([]));
    assert_eq!(
        v["gpu_policy"]["auto_stages"],
        serde_json::json!([
            "raw",
            "sampler",
            "colour",
            "sharpening",
            "encode",
            "export",
            "raw front"
        ])
    );
    // The lens stage (L1 to L3, `docs/21`) is in `ALL` but not yet signed off, so `Auto` does not
    // list it while `On` and `Require` do.
    assert_eq!(
        v["gpu_policy"]["all_stages"],
        serde_json::json!([
            "raw",
            "sampler",
            "colour",
            "sharpening",
            "encode",
            "export",
            "raw front",
            "lens"
        ])
    );
    assert!(text.contains("GPU policy"), "{text}");
    assert!(
        text.contains("auto (default): uses the CPU, no adapter opened"),
        "{text}"
    );
    assert!(text.contains("GPU stages"), "{text}");
    assert!(
        text.contains(
            "raw, sampler, colour, sharpening, encode, export, raw front for Auto; On and Require run raw, sampler, colour, sharpening, encode, export, raw front, lens"
        ),
        "{text}"
    );
    assert!(text.contains("Memory budget"), "{text}");
    assert!(
        text.contains("CPU ") && text.contains(" GiB, GPU "),
        "{text}"
    );

    // The environment sets the policy; `off` still lists what is there.
    let (v, text) = doctor(Some("off"), "dx12");
    assert_eq!(v["gpu_policy"]["setting"], "off");
    assert_eq!(v["gpu_policy"]["source"], "environment");
    assert!(
        text.contains("off (YELLOWMOON_GPU): uses the CPU"),
        "{text}"
    );
    // A setting that is not a policy is read as auto, and the report says so.
    let (v, text) = doctor(Some("maybe"), "dx12");
    assert_eq!(v["gpu_policy"]["setting"], "auto");
    assert_eq!(
        v["gpu_policy"]["warning"],
        "YELLOWMOON_GPU=maybe is not a graphics setting. Use auto, on, off or require."
    );
    assert!(text.contains("is not a graphics setting"), "{text}");

    // Budgets: numbers, with the GPU's at most the default 4 GiB.
    let budget = &v["memory_budget"];
    assert!(budget["cpu_bytes"].as_u64().is_some_and(|b| b >= 512 << 20));
    assert!(budget["gpu_bytes"].as_u64().is_some_and(|b| b <= 4 << 30));
    assert!(budget["shared_memory"].is_boolean());
}

#[test]
fn doctor_with_an_adapter_reports_it_whatever_the_policy_does_with_it() {
    let sb = Sandbox::new();
    let run = |gpu: &str| -> Value {
        let out = sb
            .ym()
            .env("YELLOWMOON_GPU", gpu)
            .args(["doctor", "--json"])
            .assert()
            .success();
        serde_json::from_slice(&out.get_output().stdout).expect("JSON")
    };
    let on = run("on");
    if on["gpu"]["available"] != true {
        return; // no adapter on this machine (the GPU jobs set YM_REQUIRE_GPU and have one)
    }
    // The adapter is listed under every policy, `off` included.
    for gpu in ["auto", "on", "off", "require"] {
        let v = run(gpu);
        assert_eq!(v["gpu"]["available"], true, "{gpu}");
        assert_eq!(v["gpu"]["name"], on["gpu"]["name"], "{gpu}");
    }
    // The text row says what the policy lets the device run, not that it runs it all today.
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", "on")
        .arg("doctor")
        .assert()
        .success();
    let text = String::from_utf8(out.get_output().stdout.clone()).expect("utf-8");
    assert!(
        text.contains("on (YELLOWMOON_GPU): uses the GPU, which may run raw, sampler, colour"),
        "{text}"
    );
    // `on` and `require` use any adapter and all the stages; `off` none; `auto` only a hardware
    // adapter and only what is signed off.
    for gpu in ["on", "require"] {
        let v = run(gpu);
        assert_eq!(v["gpu_policy"]["uses_gpu"], true, "{gpu}");
        assert_eq!(
            v["gpu_policy"]["stages"], v["gpu_policy"]["all_stages"],
            "{gpu}"
        );
    }
    assert_eq!(run("off")["gpu_policy"]["uses_gpu"], false);
    let auto = run("auto");
    let hardware = on["gpu"]["software"] == false;
    assert_eq!(auto["gpu_policy"]["uses_gpu"], hardware);
    if hardware {
        assert_eq!(
            auto["gpu_policy"]["stages"],
            auto["gpu_policy"]["auto_stages"]
        );
    } else {
        assert!(
            auto["gpu_policy"]["reason"]
                .as_str()
                .is_some_and(|r| r.contains("software adapter"))
        );
    }
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
