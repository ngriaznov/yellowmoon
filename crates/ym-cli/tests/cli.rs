//! End-to-end tests of the `yellowmoon` binary.

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

fn yellowmoon() -> Command {
    cargo_bin_cmd!("yellowmoon")
}

#[test]
fn version_prints_version_pipeline_and_git_sha() {
    yellowmoon()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::is_match(
            r"^yellowmoon \d+\.\d+\.\d+ \(pipeline \d+\.\d+\.\d+, ([0-9a-f]{7,40}|unknown)\)\n$",
        )
        .expect("valid regex"))
        .stderr(predicate::str::is_empty());
}

#[test]
fn version_reports_the_workspace_and_pipeline_versions() {
    yellowmoon()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with(
            "yellowmoon 0.0.1 (pipeline 0.1.0, ",
        ));
}

#[test]
fn stub_subcommands_exit_with_code_2_and_say_so_on_stderr() {
    let invocations: &[&[&str]] = &[
        &["render", "photo.ARW"],
        &["looks", "list"],
        &["looks", "import", "pack.zip"],
        &["looks", "verify"],
        &["looks", "default", "builtin:neutral"],
        &["looks", "bake", "signature.json"],
        &["doctor"],
    ];
    for args in invocations {
        yellowmoon()
            .args(*args)
            .assert()
            .code(2)
            .stdout(predicate::str::is_empty())
            .stderr("Not implemented yet.\n");
    }
}

#[test]
fn no_subcommand_is_a_usage_error() {
    yellowmoon()
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    yellowmoon().arg("frobnicate").assert().code(2);
}

#[test]
fn help_lists_every_top_level_command() {
    let assert = yellowmoon().arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    for command in ["render", "info", "looks", "doctor"] {
        assert!(
            stdout.contains(command),
            "--help does not mention {command}:\n{stdout}"
        );
    }
}
