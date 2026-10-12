//! End-to-end tests of the `oracle` binary's report handling.

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;

fn oracle() -> Command {
    cargo_bin_cmd!("oracle")
}

fn read_report(path: &std::path::Path) -> Value {
    let text = std::fs::read_to_string(path).expect("report was written");
    serde_json::from_str(&text).expect("report is valid JSON")
}

#[test]
fn empty_corpus_writes_an_empty_report_and_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    std::fs::create_dir(&corpus).expect("create corpus folder");
    let report_path = dir.path().join("out/report.json");

    oracle()
        .args(["--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .success();

    let report = read_report(&report_path);
    assert_eq!(report["files"], serde_json::json!([]));
    assert_eq!(report["summary"]["total"], 0);
}

#[test]
fn corpus_with_files_is_reported_as_not_compared_and_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    std::fs::create_dir(&corpus).expect("create corpus folder");
    std::fs::write(corpus.join("DSC00001.ARW"), b"").expect("create test file");
    let report_path = dir.path().join("report.json");

    oracle()
        .args(["--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .code(2)
        .stderr("Not implemented yet.\n");

    let report = read_report(&report_path);
    assert_eq!(report["summary"]["total"], 1);
    assert_eq!(report["summary"]["compared"], 0);
    assert_eq!(report["files"][0]["status"], "not_compared");
}

#[test]
fn missing_corpus_folder_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    oracle()
        .args(["--corpus"])
        .arg(dir.path().join("does-not-exist"))
        .args(["--report"])
        .arg(dir.path().join("report.json"))
        .assert()
        .failure();
}
