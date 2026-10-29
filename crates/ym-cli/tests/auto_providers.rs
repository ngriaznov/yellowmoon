//! The Auto providers on the command line (`docs/22` sections 13, 14 and 16): `--auto-provider` and its
//! alias, the saved choice and its consent, what `ym` says on stderr, the objects and the row of `ym info`
//! and `render --json`, `ym doctor`, a render through the test double of Claude Code (two photos at once,
//! the answers in the file's XMP, Ctrl-C leaving no child running) and the breaker's sentence.
//!
//! Two ways to give `ym` a program: the in-process fakes of the hook `YELLOWMOON_TEST_AUTO_PROVIDERS`
//! (`claude=available`, `agy=signed_out`, ...; no process, deterministic answers) and `exec:<path>` over
//! the real advisor with the test double `ym-fake-claude` or `ym-fake-agy`, built by
//! `cargo test -p ym-agents` (`ym_agents::testing`: a missing double is a failure when `YM_REQUIRE_FAKE_CLI`
//! is set and a printed `SKIPPED` otherwise). Every other test of this crate runs the binary with
//! `YELLOWMOON_TEST_AUTO_PROVIDERS=none`, so a developer's own programs are never started.

mod common;

use std::path::{Path, PathBuf};

use common::{Sandbox, json_lines, text};
use serde_json::{Value, json};
use ym_agents::Tool;
use ym_core::apple_auto::UnavailableReason;
use ym_core::auto_provider::{AutoProvider, unavailable_hint};

/// The photo of these tests.
fn photo(sb: &Sandbox) -> PathBuf {
    sb.arw("a.ARW")
}

/// `ym render --json` of one photo with `flags`, through `cmd`: the object and what was said on stderr.
fn render(mut cmd: assert_cmd::Command, arw: &Path, flags: &[&str], out: &Path) -> (Value, String) {
    let assert = cmd
        .args(["render", "--json", "--cpu", "--overwrite"])
        .args(flags)
        .arg("--out")
        .arg(out)
        .arg(arw)
        .assert()
        .success();
    let stderr = text(&assert.get_output().stderr);
    let mut lines = json_lines(&assert.get_output().stdout);
    assert_eq!(lines.len(), 1, "one object");
    (lines.remove(0), stderr)
}

/// `ym info --json` of one photo with `flags`.
fn info(mut cmd: assert_cmd::Command, arw: &Path, flags: &[&str]) -> (Value, String) {
    let assert = cmd
        .args(["info", "--json"])
        .args(flags)
        .arg(arw)
        .assert()
        .success();
    let stderr = text(&assert.get_output().stderr);
    (json_lines(&assert.get_output().stdout).remove(0), stderr)
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn xmp_of(v: &Value) -> String {
    let output = PathBuf::from(v["output"].as_str().expect("an output path"));
    String::from_utf8_lossy(&bytes(&output)).into_owned()
}

/// The test double of `tool`, or `None` when it is not built (a printed `SKIPPED`).
fn double(tool: Tool) -> Option<PathBuf> {
    ym_agents::testing::require_fake_cli(tool)
}

fn exec(path: &Path) -> String {
    format!("exec:{}", path.display())
}

/// When each *request* the double saw started, from what it already writes: the time it wrote
/// `argv.txt` into its `run-NNNN` folder. A run is a request when its arguments name a `--json-schema`;
/// the probes (`--version`, `auth status`, `-p /usage`) do not.
fn request_starts(record: &Path) -> Vec<std::time::SystemTime> {
    let mut starts: Vec<_> = std::fs::read_dir(record)
        .expect("the record folder")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .filter_map(|e| {
            let argv = e.path().join("argv.txt");
            let text = std::fs::read_to_string(&argv).ok()?;
            text.lines()
                .any(|l| l == "--json-schema")
                .then(|| std::fs::metadata(&argv).and_then(|m| m.modified()).ok())
                .flatten()
        })
        .collect();
    starts.sort();
    starts
}

/// How many runs of the double were not requests: the checks (`--version`, `auth status`, `-p /usage`).
fn check_runs(record: &Path) -> usize {
    std::fs::read_dir(record)
        .expect("the record folder")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .filter_map(|e| std::fs::read_to_string(e.path().join("argv.txt")).ok())
        .filter(|text| !text.lines().any(|l| l == "--json-schema"))
        .count()
}

/// How far apart the first and the last of `starts` are.
fn gap(starts: &[std::time::SystemTime]) -> std::time::Duration {
    match (starts.first(), starts.last()) {
        (Some(first), Some(last)) => last.duration_since(*first).unwrap_or_default(),
        _ => std::time::Duration::ZERO,
    }
}

const DEFAULT_AUTO_USED: &str = " Default Auto is used.";

// ---------------------------------------------------------------------------------------------
// The options
// ---------------------------------------------------------------------------------------------

#[test]
fn the_provider_is_named_in_any_case_and_the_old_name_is_an_alias() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    for word in ["claude", "Claude", "CLAUDE"] {
        let (v, _) = render(
            sb.ym_provider("claude=available"),
            &arw,
            &["--auto-provider", word, "--quiet"],
            &sb.root().join("out"),
        );
        assert_eq!(v["auto_source"], "claude", "{word}");
    }
    for (flags, source) in [
        (["--auto-provider", "yellowmoon"], "rule"),
        (["--auto-provider", "YELLOWMOON"], "rule"),
        (["--apple-auto", "off"], "rule"),
    ] {
        let (v, _) = render(
            sb.ym_provider("claude=available"),
            &arw,
            &flags,
            &sb.root().join("out"),
        );
        assert_eq!(v["auto_source"], source, "{flags:?}");
    }
    // `--apple-auto on` is `--auto-provider apple`: with the test advisor it answers.
    let (v, _) = render(
        sb.ym_fake(),
        &arw,
        &["--apple-auto", "on"],
        &sb.root().join("out"),
    );
    assert_eq!(v["auto_source"], "apple");
    // Both names together are a usage error with the sentence of the spec; so is a name that is no provider.
    let out = sb
        .ym()
        .args(["render", "--apple-auto", "on", "--auto-provider", "claude"])
        .arg(&arw)
        .assert()
        .code(2);
    assert!(
        text(&out.get_output().stderr).contains("Use --auto-provider or --apple-auto, not both."),
        "{}",
        text(&out.get_output().stderr)
    );
    sb.ym()
        .args(["info", "--apple-auto", "off", "--auto-provider", "agy"])
        .arg(&arw)
        .assert()
        .code(2);
    sb.ym()
        .args(["render", "--auto-provider", "gemini"])
        .arg(&arw)
        .assert()
        .code(2);
    // The instructions file has two names, and not both.
    let file = sb.root().join("words.txt");
    std::fs::write(&file, "Cool and moody, please.").expect("write");
    for name in ["--auto-instructions", "--apple-auto-instructions"] {
        let (v, _) = render(
            sb.ym_fake(),
            &arw,
            &[name, file.to_str().expect("utf-8")],
            &sb.root().join("out"),
        );
        assert_eq!(v["apple_auto"]["plan"], "Test plan: Cool", "{name}");
    }
    sb.ym()
        .args(["render", "--auto-instructions"])
        .arg(&file)
        .arg("--apple-auto-instructions")
        .arg(&file)
        .arg(&arw)
        .assert()
        .code(2);
    // The help says the answers differ from run to run.
    let help = sb.ym().args(["render", "--help"]).assert().success();
    let help = text(&help.get_output().stdout);
    assert!(
        help.contains("Answers from Claude Code and Antigravity can differ from run to run"),
        "{help}"
    );
}

// ---------------------------------------------------------------------------------------------
// The saved choice and the consent
// ---------------------------------------------------------------------------------------------

#[test]
fn the_saved_choice_acts_only_with_its_consent_and_naming_it_is_the_consent_for_a_run() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let cloud = "Claude Code will send a 768 px picture of each photo and the instructions to";
    // Saved, with nothing agreed to: Default Auto, silently (the app's row is hidden too), and
    // the photo says why.
    sb.set_auto_provider("claude", &[]);
    let (v, stderr) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &[],
        &sb.root().join("a"),
    );
    assert_eq!(v["auto_source"], "rule");
    assert_eq!(
        v["auto"]["apple"],
        json!({ "state": "unavailable", "reason": "not_consented" }),
        "{v}"
    );
    assert!(
        !stderr.contains(cloud) && !stderr.contains("not available"),
        "{stderr}"
    );
    assert!(
        v.get("apple_auto").is_none() && v.get("auto_advisor").is_none(),
        "{v}"
    );
    let (i, _) = info(sb.ym_provider("claude=available"), &arw, &[]);
    assert_eq!(i["auto_advisor"]["state"], "unavailable");
    assert_eq!(i["auto_advisor"]["reason"], "not_consented");
    assert_eq!(i["auto_advisor"]["provider"], "claude");
    // The row says so, and `ym doctor` says how to use it all the same.
    let row = info_row(&sb, "claude=available", &[]);
    assert_eq!(
        row,
        "Not available (you haven't agreed to send photos to Claude Code yet)"
    );
    let doctor = sb
        .ym_provider("claude=available")
        .arg("doctor")
        .assert()
        .success();
    let doctor = text(&doctor.get_output().stdout);
    assert!(
        doctor.contains(
            "Claude Code (saved in Preferences) · not agreed to yet; naming it with --auto-provider is enough for ym"
        ),
        "{doctor}"
    );
    // Naming it is the consent for this run: it acts, says where the picture goes, and saves nothing.
    let before = std::fs::read(sb.home().join("preferences.json")).expect("preferences");
    let (v, stderr) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "claude"],
        &sb.root().join("b"),
    );
    assert_eq!(v["auto_source"], "claude");
    assert!(stderr.contains(cloud), "{stderr}");
    assert_eq!(
        std::fs::read(sb.home().join("preferences.json")).expect("preferences"),
        before,
        "nothing was saved"
    );
    let saved: Value = serde_json::from_slice(&before).expect("JSON");
    assert_eq!(
        saved["auto_consent"],
        json!({ "claude": false, "agy": false })
    );
    // With the consent saved, the saved choice acts without the option, and says the same once.
    sb.set_auto_provider("claude", &["claude"]);
    let (v, stderr) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &[],
        &sb.root().join("c"),
    );
    assert_eq!(v["auto_source"], "claude");
    assert_eq!(stderr.matches(cloud).count(), 1, "once per run: {stderr}");
    // `--quiet` hides it; the old switch off, or Default Auto for the run, does not call it.
    let (_, stderr) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--quiet"],
        &sb.root().join("d"),
    );
    assert!(!stderr.contains(cloud), "{stderr}");
    let (v, stderr) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "yellowmoon"],
        &sb.root().join("e"),
    );
    assert_eq!(v["auto_source"], "rule");
    assert!(!stderr.contains(cloud), "{stderr}");
}

/// A saved cloud choice that nothing agreed to (the file says `claude`, the consent is not saved) touches
/// no program, whatever the command: no `--version`, no `auth status`, no request (`docs/22` sections
/// 10.2 and 13.1). The double records every invocation, so a probe would show. The photo says
/// `not_consented` every time, not `checking` or `not_consented` by the luck of the timing.
#[test]
fn a_saved_cloud_choice_nobody_agreed_to_starts_no_program() {
    let Some(claude) = double(Tool::Claude) else {
        return;
    };
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let record = sb.root().join("record");
    sb.set_auto_provider("claude", &[]);
    let hook = format!("claude={}", exec(&claude));
    for round in 0..3 {
        let (i, _) = info(
            {
                let mut cmd = sb.ym_provider(&hook);
                cmd.env("YM_FAKE_RECORD", &record);
                cmd
            },
            &arw,
            &[],
        );
        assert_eq!(i["auto_advisor"]["state"], "unavailable", "round {round}");
        assert_eq!(
            i["auto_advisor"]["reason"], "not_consented",
            "round {round}"
        );
        assert_eq!(i["auto_advisor"]["provider"], "claude", "round {round}");
    }
    let (v, stderr) = render(
        {
            let mut cmd = sb.ym_provider(&hook);
            cmd.env("YM_FAKE_RECORD", &record);
            cmd
        },
        &arw,
        &[],
        &sb.root().join("out"),
    );
    assert_eq!(v["auto_source"], "rule");
    assert_eq!(
        v["auto"]["apple"],
        json!({ "state": "unavailable", "reason": "not_consented" }),
        "{v}"
    );
    assert!(!stderr.contains("Claude Code"), "{stderr}");
    // The double was never started, not even for `--version`.
    let runs = std::fs::read_dir(&record)
        .map(|d| d.flatten().count())
        .unwrap_or(0);
    assert_eq!(runs, 0, "a program that was not agreed to was started");
    // The control: naming the program is the consent, and the double is started (and recorded).
    let (v, _) = render(
        {
            let mut cmd = sb.ym_provider(&hook);
            cmd.env("YM_FAKE_RECORD", &record);
            cmd
        },
        &arw,
        &["--auto-provider", "claude"],
        &sb.root().join("named"),
    );
    assert_eq!(v["auto_source"], "claude", "{v}");
    assert!(
        std::fs::read_dir(record.join("claims"))
            .map(|d| d.flatten().count())
            .unwrap_or(0)
            > 0,
        "naming the program starts it"
    );
}

/// The text of the **Auto grading** row of `ym info` with `hook` and `flags`.
fn info_row(sb: &Sandbox, hook: &str, flags: &[&str]) -> String {
    let arw = photo(sb);
    let out = sb
        .ym_provider(hook)
        .arg("info")
        .args(flags)
        .arg(&arw)
        .assert()
        .success();
    let out = text(&out.get_output().stdout);
    out.lines()
        .find(|l| l.trim_start().starts_with("Auto grading"))
        .unwrap_or_else(|| panic!("no Auto grading row:\n{out}"))
        .trim_start()
        .trim_start_matches("Auto grading")
        .trim()
        .to_owned()
}

// ---------------------------------------------------------------------------------------------
// What `ym` says on stderr
// ---------------------------------------------------------------------------------------------

#[test]
fn a_provider_the_run_named_that_is_not_available_says_why_once_and_the_run_goes_on() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let os = std::env::consts::OS;
    let hint = |p, r, v: Option<&str>| unavailable_hint(p, r, v, os);
    let cases: [(&str, &str, String); 7] = [
        (
            "claude",
            "claude=not_installed",
            "Claude Code is not available on this computer (it isn't installed).".to_owned(),
        ),
        (
            "claude",
            "none",
            "Claude Code is not available on this computer (it isn't installed).".to_owned(),
        ),
        (
            "claude",
            "claude=signed_out",
            hint(AutoProvider::Claude, UnavailableReason::SignedOut, None),
        ),
        (
            "claude",
            "claude=too_old",
            hint(
                AutoProvider::Claude,
                UnavailableReason::TooOld,
                Some("0.0.0-test"),
            ),
        ),
        (
            "claude",
            "claude=unavailable",
            hint(AutoProvider::Claude, UnavailableReason::ProbeFailed, None),
        ),
        (
            "agy",
            "agy=signed_out",
            hint(AutoProvider::Agy, UnavailableReason::SignedOut, None),
        ),
        (
            "agy",
            "agy=not_installed",
            "Antigravity is not available on this computer (it isn't installed).".to_owned(),
        ),
    ];
    for (provider, hook, sentence) in cases {
        let want = format!("{sentence}{DEFAULT_AUTO_USED}");
        let (v, stderr) = render(
            sb.ym_provider(hook),
            &arw,
            &["--auto-provider", provider],
            &sb.root().join("out"),
        );
        assert_eq!(
            stderr.matches(&want).count(),
            1,
            "{hook}: once per run:\n{stderr}"
        );
        assert_eq!(v["auto_source"], "rule", "{hook}");
        assert_eq!(v["status"], "ok");
        // `ym info` says it too; `--quiet` says nothing.
        let (_, stderr) = info(sb.ym_provider(hook), &arw, &["--auto-provider", provider]);
        assert_eq!(stderr.matches(&want).count(), 1, "{hook}: info:\n{stderr}");
        let (_, stderr) = render(
            sb.ym_provider(hook),
            &arw,
            &["--auto-provider", provider, "--quiet"],
            &sb.root().join("out"),
        );
        assert!(stderr.is_empty(), "{hook}: --quiet:\n{stderr}");
    }
    // Where the program is there and the photo says so: the clause with the program's own words.
    assert_eq!(
        info_row(&sb, "claude=signed_out", &["--auto-provider", "claude"]),
        "Not available (Claude Code isn't signed in)"
    );
    assert_eq!(
        info_row(&sb, "agy=too_old", &["--auto-provider", "agy"]),
        "Not available (Antigravity 0.0.0-test is too old)"
    );
    assert_eq!(
        info_row(&sb, "none", &["--auto-provider", "claude"]),
        "Not available (Claude Code isn't installed)"
    );
    // Apple Intelligence keeps its sentence (`docs/19` section 8).
    if cfg!(not(target_os = "macos")) {
        let (_, stderr) = render(
            sb.ym(),
            &arw,
            &["--auto-provider", "apple"],
            &sb.root().join("out"),
        );
        assert!(
            stderr.contains(
                "Warning: Apple Auto is not available on this computer (this is not a Mac). Default Auto is used."
            ),
            "{stderr}"
        );
    }
    // A run that only follows the saved choice says nothing: the app hides its row too.
    sb.set_auto_provider("claude", &["claude"]);
    let (v, stderr) = render(
        sb.ym_provider("claude=signed_out"),
        &arw,
        &[],
        &sb.root().join("out"),
    );
    assert_eq!(v["auto_source"], "rule");
    assert!(stderr.is_empty(), "{stderr}");
}

#[test]
fn antigravity_says_where_the_picture_goes() {
    let sb = Sandbox::new();
    let (v, stderr) = render(
        sb.ym_provider("agy=available"),
        &photo(&sb),
        &["--auto-provider", "agy"],
        &sb.root().join("out"),
    );
    assert_eq!(v["auto_source"], "agy");
    assert_eq!(
        stderr
            .matches("Antigravity (agy) will send a 768 px picture of each photo and the instructions to Google.")
            .count(),
        1,
        "{stderr}"
    );
    assert_eq!(v["auto_advisor"]["provider"], "agy");
    assert_eq!(v["auto_advisor"]["model"], "Test Antigravity");
    assert_eq!(v["auto_advisor"]["prompt"], 1);
}

// ---------------------------------------------------------------------------------------------
// The objects and the row
// ---------------------------------------------------------------------------------------------

#[test]
fn the_json_names_the_provider_and_the_old_object_is_apples_view() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let (v, _) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "claude", "--quiet"],
        &sb.root().join("out"),
    );
    assert_eq!(v["auto_source"], "claude");
    assert_eq!(v["auto"]["source"], "claude");
    assert_eq!(v["auto"]["provider"], "claude");
    // The new object, every key, sorted; the answers of a program are not reproducible.
    let a = &v["auto_advisor"];
    let keys: Vec<&str> = a
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "api",
            "image",
            "instructions",
            "instructions_sha256",
            "model",
            "os",
            "plan",
            "prompt",
            "provider",
            "reason",
            "reproducible",
            "scene",
            "schema",
            "state",
            "steps",
            "table",
            "tool_version"
        ],
        "{a}"
    );
    assert_eq!(
        (
            a["provider"].clone(),
            a["state"].clone(),
            a["model"].clone(),
            a["tool_version"].clone(),
            a["os"].clone(),
            a["api"].clone()
        ),
        (
            json!("claude"),
            json!("answered"),
            json!("Test Claude Code"),
            json!("0.0.0-test"),
            Value::Null,
            Value::Null
        )
    );
    assert_eq!(
        (
            a["prompt"].clone(),
            a["schema"].clone(),
            a["image"].clone(),
            a["table"].clone(),
            a["reproducible"].clone(),
            a["scene"].clone()
        ),
        (
            json!(1),
            json!(1),
            json!("jpeg444q92@768"),
            json!(2),
            json!(false),
            json!("Test picture")
        )
    );
    assert_eq!(a["steps"].as_object().expect("steps").len(), 12);
    // The old object is Apple's view: for any other provider it says `off`, so a script that looks for
    // `answered` finds only what Apple graded.
    assert_eq!(
        v["apple_auto"],
        Value::Null,
        "render writes it only for Apple's photos: {v}"
    );
    let (i, _) = info(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "claude"],
    );
    assert_eq!(i["apple_auto"], json!({ "state": "off" }));
    assert_eq!(i["auto_advisor"], v["auto_advisor"]);
    for key in ["auto", "auto_source", "settings"] {
        assert_eq!(i[key], v[key], "{key}");
    }
    // The key order of docs/10: `auto`, `auto_source`, `apple_auto`, `auto_advisor`, `settings`.
    let out = sb
        .ym_provider("claude=available")
        .args(["info", "--json", "--auto-provider", "claude"])
        .arg(&arw)
        .assert()
        .success();
    let line = text(&out.get_output().stdout);
    let at = |key: &str| line.find(&format!("\"{key}\":")).expect(key);
    assert!(
        at("auto") < at("auto_source")
            && at("auto_source") < at("apple_auto")
            && at("apple_auto") < at("auto_advisor")
            && at("auto_advisor") < at("settings"),
        "{line}"
    );
    // The row names the provider first.
    assert_eq!(
        info_row(&sb, "claude=available", &["--auto-provider", "claude"]),
        "Claude Code · Darker -0.3 EV · highlights -20 · contrast +15 · teal shadows (subtle) · scene \"Test picture\" · plan \"Test plan: You\""
    );
    // Default Auto: the rule, and no object for a render.
    let (v, _) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "yellowmoon"],
        &sb.root().join("rule"),
    );
    assert_eq!(v["auto_source"], "rule");
    assert_eq!(v["auto"]["provider"], "yellowmoon");
    assert!(v.get("auto_advisor").is_none());
    let (i, _) = info(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "yellowmoon"],
    );
    assert_eq!(i["auto_advisor"]["provider"], "yellowmoon");
    assert_eq!(i["auto_advisor"]["state"], "off");
    assert_eq!(i["auto_advisor"]["prompt"], Value::Null);
    assert_eq!(
        info_row(&sb, "claude=available", &["--auto-provider", "yellowmoon"]),
        "Default Auto"
    );
}

#[test]
fn an_export_records_the_provider_in_its_xmp() {
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let (v, _) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "claude", "--quiet"],
        &sb.root().join("out"),
    );
    let xmp = xmp_of(&v);
    assert!(xmp.contains("<ym:Auto>claude</ym:Auto>"), "{xmp}");
    let sha = ym_core::apple_auto::InstructionsId::default().instructions_sha256;
    let want = format!(
        r#"{{"image":"jpeg444q92@768","instructions":"default","instructions_sha256":"{sha}","model":"Test Claude Code","plan":"Test plan: You","prompt":1,"provider":"claude","reproducible":false,"result":"answered","scene":"Test picture","schema":1,"steps":{{"blacks":null,"contrast":1,"exposure":-1,"highlight_color":null,"highlight_color_strength":null,"highlights":-1,"saturation":0,"shadow_color":"teal","shadow_color_strength":1,"shadows":0,"tint":null,"white":0}},"table":2,"tool_version":"0.0.0-test"}}"#
    )
    .replace('"', "&quot;");
    assert!(
        xmp.contains(&format!("<ym:AutoAdvisor>{want}</ym:AutoAdvisor>")),
        "ym:AutoAdvisor in {xmp}"
    );
    // Nothing of the program's place or messages, and no `ym:AppleAuto`.
    assert!(!xmp.contains("ym:AppleAuto"), "{xmp}");
    // Default Auto writes `ym:Auto` `rule` and no advisor.
    let (rule, _) = render(
        sb.ym_provider("claude=available"),
        &arw,
        &["--auto-provider", "yellowmoon"],
        &sb.root().join("rule"),
    );
    let xmp = xmp_of(&rule);
    assert!(xmp.contains("<ym:Auto>rule</ym:Auto>") && !xmp.contains("ym:AutoAdvisor"));
    assert_ne!(
        bytes(&PathBuf::from(v["output"].as_str().unwrap())),
        bytes(&PathBuf::from(rule["output"].as_str().unwrap()))
    );
}

// ---------------------------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------------------------

#[test]
fn doctor_reports_each_provider_in_each_state() {
    let sb = Sandbox::new();
    let json = |hook: &str| -> Value {
        let out = sb
            .ym_provider(hook)
            .args(["doctor", "--json"])
            .assert()
            .success();
        serde_json::from_slice(&out.get_output().stdout).expect("JSON")
    };
    let report = |hook: &str| -> String {
        let out = sb.ym_provider(hook).arg("doctor").assert().success();
        text(&out.get_output().stdout)
    };
    let row = |report: &str, label: &str| -> String {
        report
            .lines()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("no {label} row:\n{report}"))
            .trim_start_matches(label)
            .trim()
            .to_owned()
    };
    let v = json("claude=available,agy=signed_out");
    let auto = &v["auto"];
    assert_eq!(auto["preference"], "apple");
    assert_eq!(auto["consent"], json!({ "claude": false, "agy": false }));
    assert_eq!(auto["table"], 2);
    let providers = &auto["providers"];
    assert_eq!(
        providers["yellowmoon"],
        json!({ "state": "available", "reason": null })
    );
    assert_eq!(providers["claude"]["state"], "available");
    assert_eq!(providers["claude"]["model"], "Test Claude Code");
    assert_eq!(providers["claude"]["tool_version"], "0.0.0-test");
    assert_eq!(providers["claude"]["prompt"], 1);
    assert_eq!(providers["claude"]["schema"], 1);
    assert_eq!(providers["claude"]["image"], "jpeg444q92@768");
    assert_eq!(providers["agy"]["state"], "signed_out");
    assert_eq!(providers["agy"]["reason"], "signed_out");
    // The programs' alias or slug is the model until a request has answered.
    assert_eq!(providers["agy"]["model"], "default");
    assert_eq!(providers["apple"]["state"], "unavailable");
    // The old object stays, Apple only.
    assert!(v["apple_auto"]["preference"].as_bool().unwrap());
    let text_report = report("claude=available,agy=signed_out");
    assert_eq!(
        row(&text_report, "Claude Code"),
        "available · Claude Code 0.0.0-test · Test Claude Code · prompt 1 · schema 1 · table 2"
    );
    assert_eq!(
        row(&text_report, "Antigravity (agy)"),
        "installed but not signed in · 0.0.0-test"
    );
    assert_eq!(
        row(&text_report, "Auto grading"),
        "Apple Intelligence (saved in Preferences)"
    );
    // The rows of the other states.
    for (hook, label, want) in [
        ("claude=not_installed", "Claude Code", "not installed"),
        ("claude=checking", "Claude Code", "still checking"),
        (
            "claude=too_old",
            "Claude Code",
            "too old (0.0.0-test; 2.1.169 or later is needed)",
        ),
        (
            "agy=unavailable",
            "Antigravity (agy)",
            "didn't answer its check",
        ),
        (
            "agy=too_old",
            "Antigravity (agy)",
            "too old (0.0.0-test; 1.2.14 or later is needed)",
        ),
        ("none", "Antigravity (agy)", "not installed"),
    ] {
        assert_eq!(row(&report(hook), label), want, "{hook}");
    }
    assert_eq!(
        json("claude=not_installed")["auto"]["providers"]["claude"]["state"],
        "not_installed"
    );
    assert_eq!(
        json("claude=too_old")["auto"]["providers"]["claude"]["reason"],
        "too_old"
    );
    assert_eq!(
        json("agy=unavailable")["auto"]["providers"]["agy"]["reason"],
        "probe_failed"
    );
    assert_eq!(
        json("claude=checking")["auto"]["providers"]["claude"]["state"],
        "checking"
    );
    // The saved choice and its consent.
    sb.set_auto_provider("agy", &["agy"]);
    let v = json("agy=available");
    assert_eq!(v["auto"]["preference"], "agy");
    assert_eq!(v["auto"]["effective"], "agy");
    assert_eq!(
        v["auto"]["consent"],
        json!({ "claude": false, "agy": true })
    );
    assert_eq!(v["apple_auto"]["preference"], false);
    assert_eq!(
        row(&report("agy=available"), "Auto grading"),
        "Antigravity (saved in Preferences)"
    );
    // Reading the report writes nothing new.
    let saved = std::fs::read(sb.home().join("preferences.json")).expect("preferences");
    let _ = json("agy=available");
    assert_eq!(
        std::fs::read(sb.home().join("preferences.json")).expect("preferences"),
        saved
    );
}

// ---------------------------------------------------------------------------------------------
// The test doubles
// ---------------------------------------------------------------------------------------------

/// Two photos through the real advisor over `ym-fake-claude`: answered two at a time, the answers in the
/// files, the notice of where the picture goes (Anthropic: `apiProvider` is `firstParty`), the model that
/// answered in the file's metadata.
#[test]
fn a_render_through_the_test_double_of_claude_code_answers_both_photos() {
    let Some(claude) = double(Tool::Claude) else {
        return;
    };
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw_sized("b.ARW", 128, 96, 7);
    let record = sb.root().join("record");
    let out = sb
        .ym_provider(&format!("claude={}", exec(&claude)))
        .env("YM_FAKE_RECORD", &record)
        .env("YM_FAKE_DELAY_MS", "800")
        .args([
            "render",
            "--json",
            "--cpu",
            "--auto-provider",
            "claude",
            "--out",
        ])
        .arg(sb.root().join("out"))
        .arg(sb.photos())
        .assert()
        .success();
    let stderr = text(&out.get_output().stderr);
    assert_eq!(
        stderr
            .matches("Claude Code will send a 768 px picture of each photo and the instructions to Anthropic.")
            .count(),
        1,
        "{stderr}"
    );
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 2);
    for v in &objects {
        assert_eq!(v["status"], "ok", "{v}");
        assert_eq!(v["auto_source"], "claude");
        let a = &v["auto_advisor"];
        assert_eq!(a["state"], "answered", "{a}");
        assert_eq!(a["provider"], "claude");
        assert_eq!(
            a["model"], "claude-fake-1",
            "the model the program says answered"
        );
        assert_eq!(a["api"], "firstParty");
        assert_eq!(a["tool_version"], "2.1.310");
        let xmp = xmp_of(v);
        assert!(xmp.contains("<ym:Auto>claude</ym:Auto>"), "{xmp}");
        assert!(
            xmp.contains("&quot;model&quot;:&quot;claude-fake-1&quot;"),
            "{xmp}"
        );
        // The picture the double was given is the request of `docs/22` section 7: no path of ours.
        assert!(
            !xmp.contains(sb.root().to_str().expect("utf-8")),
            "no path in the file"
        );
    }
    // Two requests ran at the same time: the double saw two *requests* (not the probes, which it
    // records as runs too) and each sleeps 800 ms, so they overlap only if the second started within
    // that of the first.
    let starts = request_starts(&record);
    assert_eq!(starts.len(), 2, "two requests were made: {starts:?}");
    assert!(
        gap(&starts) < std::time::Duration::from_millis(400),
        "the two requests did not overlap: they started {:?} apart",
        gap(&starts)
    );
    // The program the run named was checked once, before the core opened (`--version`, then its sign-in
    // check), and opening the photos and starting the export asked for no second check while it read
    // available (`docs/22` sections 3.3 and 13.1, O7).
    assert_eq!(
        check_runs(&record),
        2,
        "one --version and one sign-in check, nothing more"
    );
    // The program's own state is used: `ym info` of a signed-in double is available, and `doctor` says
    // where it was found and how it is signed in.
    let out = sb
        .ym_provider(&format!("claude={}", exec(&claude)))
        .args(["doctor", "--json"])
        .assert()
        .success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    let c = &v["auto"]["providers"]["claude"];
    assert_eq!(c["state"], "available", "{c}");
    assert_eq!(c["tool_version"], "2.1.310");
    assert_eq!(c["api"], "firstParty");
    assert!(
        c["path"]
            .as_str()
            .is_some_and(|p| p.contains("ym-fake-claude")),
        "{c}"
    );
}

/// The helper above can tell a request that overlaps from one that does not: Antigravity takes one
/// request at a time (`docs/22` D6), so its two requests start a request's length apart.
#[test]
fn antigravity_asks_two_photos_one_after_the_other() {
    let Some(agy) = double(Tool::Agy) else {
        return;
    };
    let sb = Sandbox::new();
    sb.arw("a.ARW");
    sb.arw_sized("b.ARW", 128, 96, 7);
    let record = sb.root().join("record");
    sb.ym_provider(&format!("agy={}", exec(&agy)))
        .env("YM_FAKE_RECORD", &record)
        .env("YM_FAKE_DELAY_MS", "800")
        .args([
            "render",
            "--json",
            "--cpu",
            "--auto-provider",
            "agy",
            "--out",
        ])
        .arg(sb.root().join("out"))
        .arg(sb.photos())
        .assert()
        .success();
    let starts = request_starts(&record);
    assert_eq!(starts.len(), 2, "two requests were made: {starts:?}");
    assert!(
        gap(&starts) >= std::time::Duration::from_millis(700),
        "the requests overlapped: they started {:?} apart",
        gap(&starts)
    );
}

#[test]
fn a_signed_out_test_double_is_listed_and_a_run_that_names_it_says_so() {
    let Some(claude) = double(Tool::Claude) else {
        return;
    };
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let hook = format!("claude={}", exec(&claude));
    let scenario = |name: &str| {
        let mut cmd = sb.ym_provider(&hook);
        cmd.env("YM_FAKE_SCENARIO", name);
        cmd
    };
    let (v, stderr) = render(
        scenario("signed_out"),
        &arw,
        &["--auto-provider", "claude"],
        &sb.root().join("out"),
    );
    let want = format!(
        "{}{DEFAULT_AUTO_USED}",
        unavailable_hint(
            AutoProvider::Claude,
            UnavailableReason::SignedOut,
            None,
            std::env::consts::OS
        )
    );
    assert_eq!(stderr.matches(&want).count(), 1, "{stderr}");
    assert_eq!(v["auto_source"], "rule");
    // Too old: the version it printed is in the sentence.
    let (_, stderr) = render(
        scenario("too_old"),
        &arw,
        &["--auto-provider", "claude"],
        &sb.root().join("out"),
    );
    assert!(
        stderr.contains("Claude Code 2.1.100 is too old."),
        "{stderr}"
    );
}

/// Three failures in a row open the advisor's breaker: the photos it did not answer use the rule and `ym` says so,
/// once, after the waits.
#[test]
fn three_failures_in_a_row_open_the_breaker_and_ym_says_so_once() {
    let Some(claude) = double(Tool::Claude) else {
        return;
    };
    let sb = Sandbox::new();
    for k in 0..6u64 {
        // Taken a day apart: no two of them are a burst.
        let path = sb.photos().join(format!("p{k}.ARW"));
        ym_raw::testgen::ArwBuilder::new("ILCE-7RM4", 96, 64)
            .capture_time(
                &format!("2026:10:{:02} 18:30:00", 5 + k),
                "000",
                Some("+02:00"),
            )
            .write_to(
                &path,
                &ym_raw::testgen::gradient_mosaic(96, 64, 512, 16_000, 21 + k),
            )
            .expect("write synthetic ARW");
    }
    let out = sb
        .ym_provider(&format!("claude={}", exec(&claude)))
        .env("YM_FAKE_SCENARIO", "rate_limit")
        .args([
            "render",
            "--json",
            "--cpu",
            "--jobs",
            "1",
            "--auto-provider",
            "claude",
            "--out",
        ])
        .arg(sb.root().join("out"))
        .arg(sb.photos())
        .assert()
        .success();
    let stderr = text(&out.get_output().stderr);
    // The program's name and the failure are the advisor's words (`docs/22` section 9.2), and so is the name
    // of the Auto that answers the rest (`Default Auto`).
    let sentence = "Claude Code failed three times in a row (Claude Code reported a rate limit). Photos it doesn't answer use ";
    assert_eq!(stderr.matches(sentence).count(), 1, "{stderr}");
    assert!(
        stderr.contains("Default Auto."),
        "the sentence names Default Auto: {stderr}"
    );
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 6);
    for v in &objects {
        assert_eq!(v["status"], "ok");
        assert_eq!(v["auto_source"], "rule", "{v}");
        assert_eq!(v["auto_advisor"]["state"], "failed");
    }
    let reasons: Vec<_> = objects
        .iter()
        .map(|v| {
            v["auto_advisor"]["reason"]
                .as_str()
                .unwrap_or("")
                .to_owned()
        })
        .collect();
    assert!(
        reasons.iter().filter(|r| *r == "rate_limited").count() >= 3,
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r == "unavailable"),
        "the breaker refused the rest: {reasons:?}"
    );
    // `--quiet` hides the sentence.
    let out = sb
        .ym_provider(&format!("claude={}", exec(&claude)))
        .env("YM_FAKE_SCENARIO", "rate_limit")
        .args([
            "render",
            "--json",
            "--quiet",
            "--cpu",
            "--overwrite",
            "--jobs",
            "1",
            "--auto-provider",
            "claude",
            "--out",
        ])
        .arg(sb.root().join("out"))
        .arg(sb.photos())
        .assert()
        .success();
    assert!(!text(&out.get_output().stderr).contains("failed three times"));
}

/// Ctrl-C while a request runs: `ym` exits 130 and leaves no child running (a child is not stopped by its parent's
/// death on Unix: the handler kills it first).
#[cfg(unix)]
#[test]
fn ctrl_c_leaves_no_child_running() {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let Some(claude) = double(Tool::Claude) else {
        return;
    };
    let sb = Sandbox::new();
    let arw = photo(&sb);
    let pidfile = sb.root().join("pids");
    let ym = assert_cmd::cargo::cargo_bin_cmd!("ym");
    let mut child = Command::new(ym.get_program())
        .env("YELLOWMOON_HOME", sb.home())
        .env("YELLOWMOON_DATA", common::data_dir())
        .env("HOME", sb.root())
        .env_remove("RUST_LOG")
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .env(
            common::AUTO_PROVIDERS_ENV,
            format!("claude={}", exec(&claude)),
        )
        .env("YM_FAKE_SCENARIO", "slow")
        .env("YM_FAKE_PIDFILE", &pidfile)
        .args(["render", "--cpu", "--auto-provider", "claude", "--out"])
        .arg(sb.root().join("out"))
        .arg(&arw)
        .current_dir(sb.root())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start ym");
    // The double writes its pid and its grandchild's when the request starts.
    let deadline = Instant::now() + Duration::from_secs(120);
    let pids = loop {
        let text = std::fs::read_to_string(&pidfile).unwrap_or_default();
        let pids: Vec<i32> = text
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        if pids.len() >= 2 {
            break pids;
        }
        assert!(
            Instant::now() < deadline,
            "the request never started: {text:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    let alive = |pid: i32| {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    };
    assert!(
        pids.iter().all(|p| alive(*p)),
        "the double and its grandchild run: {pids:?}"
    );
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("send Ctrl-C");
    let status = child.wait().expect("ym ends");
    assert_eq!(status.code(), Some(130), "{status:?}");
    // Both are gone (a killed child may be a zombie for a moment until its parent reaps it).
    let deadline = Instant::now() + Duration::from_secs(10);
    while pids.iter().any(|p| alive(*p)) {
        assert!(
            Instant::now() < deadline,
            "a child of ym is still running: {pids:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------------------------
// Ctrl-C: the programs are gone, and their folders, when `ym` has ended
// ---------------------------------------------------------------------------------------------

#[cfg(unix)]
mod interrupt {
    //! `ym` over `ym-fake-claude` with its temporary folders in a folder of the test's own (`TMPDIR`), so
    //! that "no `ym-auto-*` folder of that run" can be told from another test's.

    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    use super::{Sandbox, common, double, exec, photo};
    use ym_agents::Tool;

    /// Whether the process `pid` is running. A zombie is not: it is dead and waits for whoever is its
    /// parent now to reap it (the grandchild of the double, whose parent is gone, waits for the
    /// container's init, which may be slow).
    pub fn alive(pid: i32) -> bool {
        let state = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .stderr(Stdio::null())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_default();
        !state.is_empty() && !state.starts_with('Z')
    }

    /// `ym render --auto-provider claude` over the double with `scenario`; the double writes its pids to
    /// `pidfile`, and `ym`'s temporary folders are in `tmp`.
    pub fn start(sb: &Sandbox, claude: &Path, scenario: &str, pidfile: &Path, tmp: &Path) -> Child {
        let args = ["render", "--cpu", "--auto-provider", "claude", "--out"];
        let out = sb.root().join("out");
        start_with(
            sb,
            claude,
            scenario,
            pidfile,
            tmp,
            &args,
            &[&out, &photo(sb)],
        )
    }

    /// `ym` with `args` and then `paths` over the double with `scenario` (as [`start`]).
    pub fn start_with(
        sb: &Sandbox,
        claude: &Path,
        scenario: &str,
        pidfile: &Path,
        tmp: &Path,
        args: &[&str],
        paths: &[&Path],
    ) -> Child {
        std::fs::create_dir_all(tmp).expect("tmp");
        let ym = assert_cmd::cargo::cargo_bin_cmd!("ym");
        Command::new(ym.get_program())
            .env("YELLOWMOON_HOME", sb.home())
            .env("YELLOWMOON_DATA", common::data_dir())
            .env("HOME", sb.root())
            .env("TMPDIR", tmp)
            .env_remove("RUST_LOG")
            .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
            .env(
                common::AUTO_PROVIDERS_ENV,
                format!("claude={}", exec(claude)),
            )
            .env("YM_FAKE_SCENARIO", scenario)
            .env("YM_FAKE_PIDFILE", pidfile)
            .args(args)
            .args(paths)
            .current_dir(sb.root())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(sb.root().join("ym.stderr")).expect("stderr file"))
            .spawn()
            .expect("start ym")
    }

    /// What `ym` said on stderr (for a failure's message).
    fn said(tmp: &Path) -> String {
        let file = tmp.parent().unwrap_or(tmp).join("ym.stderr");
        std::fs::read_to_string(file).unwrap_or_default()
    }

    /// The pids the double wrote, once there are `n` of them.
    pub fn wait_for_pids(pidfile: &Path, n: usize) -> Vec<i32> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let text = std::fs::read_to_string(pidfile).unwrap_or_default();
            let pids: Vec<i32> = text
                .split_whitespace()
                .filter_map(|p| p.parse().ok())
                .collect();
            if pids.len() >= n {
                return pids;
            }
            assert!(
                Instant::now() < deadline,
                "the double never started: {text:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// The `ym-auto-*` folders in `tmp`.
    pub fn folders(tmp: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(tmp)
            .map(|d| {
                d.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .is_some_and(|n| n.to_string_lossy().starts_with("ym-auto-"))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Sends Ctrl-C to `ym` and checks what the spec promises (`docs/22` section 9.4): it ends with 130
    /// at once, and **when it has ended** no program of the run is still running (a program that ignores
    /// the polite stop is killed by then, not left for ever) and no `ym-auto-*` folder is left.
    pub fn interrupt_and_check(mut child: Child, pids: &[i32], tmp: &Path) {
        let sent = Instant::now();
        Command::new("kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .expect("send Ctrl-C");
        let status = child.wait().expect("ym ends");
        assert_eq!(status.code(), Some(130), "{status:?}");
        assert!(
            sent.elapsed() < Duration::from_secs(10),
            "ym took {:?} to end",
            sent.elapsed()
        );
        // `ym` reaped what it started before it ended; only a short allowance for the process table of a
        // container whose init is slow to reap, nothing like the 2 s a program that ignores SIGTERM lives.
        let deadline = Instant::now() + Duration::from_millis(500);
        while pids.iter().any(|p| alive(*p)) {
            assert!(
                Instant::now() < deadline,
                "a program of ym is still running after ym ended ({:?} after Ctrl-C): {pids:?}\n{}",
                sent.elapsed(),
                said(tmp)
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            folders(tmp).is_empty(),
            "ym left a temporary folder: {:?}\n{}",
            folders(tmp),
            said(tmp)
        );
    }

    /// Ctrl-C during a request to a program that ignores `SIGTERM` (only `SIGKILL` ends it, and a
    /// grandchild of it): `ym` waits for the program to be killed and reaped and its folder removed.
    pub fn during_a_request_that_ignores_sigterm() {
        let Some(claude) = double(Tool::Claude) else {
            return;
        };
        let sb = Sandbox::new();
        let (pidfile, tmp) = (sb.root().join("pids"), sb.root().join("tmp"));
        let child = start(&sb, &claude, "slow_deaf", &pidfile, &tmp);
        let pids = wait_for_pids(&pidfile, 2);
        assert!(
            pids.iter().all(|p| alive(*p)),
            "the double and its grandchild run: {pids:?}"
        );
        assert!(
            !folders(&tmp).is_empty(),
            "the request has its temporary folder"
        );
        interrupt_and_check(child, &pids, &tmp);
    }

    /// Ctrl-C during `ym doctor`, which probes every program and waits for them: a probe that never ends
    /// is stopped too.
    pub fn during_doctor() {
        let Some(claude) = double(Tool::Claude) else {
            return;
        };
        let sb = Sandbox::new();
        let (pidfile, tmp) = (sb.root().join("pids"), sb.root().join("tmp"));
        let child = start_with(&sb, &claude, "probe_hang", &pidfile, &tmp, &["doctor"], &[]);
        let pids = wait_for_pids(&pidfile, 1);
        assert!(pids.iter().all(|p| alive(*p)), "the probe runs: {pids:?}");
        interrupt_and_check(child, &pids, &tmp);
    }

    /// Ctrl-C while `ym` waits for the start-up probe of the program it will use (a probe that never ends):
    /// the probe is stopped as well.
    pub fn during_the_startup_probe() {
        let Some(claude) = double(Tool::Claude) else {
            return;
        };
        let sb = Sandbox::new();
        let (pidfile, tmp) = (sb.root().join("pids"), sb.root().join("tmp"));
        let child = start(&sb, &claude, "probe_hang", &pidfile, &tmp);
        let pids = wait_for_pids(&pidfile, 1);
        assert!(pids.iter().all(|p| alive(*p)), "the probe runs: {pids:?}");
        interrupt_and_check(child, &pids, &tmp);
    }
}

/// Ctrl-C during a request to a program that ignores `SIGTERM` leaves no program and no folder behind,
/// at the moment `ym` has ended.
#[cfg(unix)]
#[test]
fn ctrl_c_during_a_request_waits_for_a_program_that_ignores_sigterm() {
    interrupt::during_a_request_that_ignores_sigterm();
}

/// Ctrl-C during the start-up probe (the wait before the core opens) stops the probe too.
#[cfg(unix)]
#[test]
fn ctrl_c_during_the_startup_probe_stops_the_probe() {
    interrupt::during_the_startup_probe();
}

/// `ym doctor` probes every program and waits; Ctrl-C stops the probes too.
#[cfg(unix)]
#[test]
fn ctrl_c_during_doctor_stops_the_probes() {
    interrupt::during_doctor();
}
