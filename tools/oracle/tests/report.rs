//! End-to-end tests of the `oracle` binary: report handling, the `dump` subcommand and the
//! comparison with rawler (always) and LibRaw (when `python3` has rawpy; `YM_REQUIRE_ORACLE=1`
//! turns a missing LibRaw into a failure).

use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;
use ym_raw::testgen::{ArwBuilder, CrawEncoder, LosslessL, Packed12, gradient_mosaic};

fn oracle() -> Command {
    cargo_bin_cmd!("oracle")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("file was written");
    serde_json::from_str(&text).expect("valid JSON")
}

fn rawpy_available() -> bool {
    let ok = StdCommand::new("python3")
        .args(["-c", "import rawpy, numpy"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok && std::env::var_os("YM_REQUIRE_ORACLE").is_some() {
        panic!("YM_REQUIRE_ORACLE is set but python3 cannot import rawpy");
    }
    ok
}

/// One small file per decoder variant that rawler understands.
fn write_corpus(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create corpus folder");
    let encoder = CrawEncoder::new();
    let files = [
        (
            "uncompressed14.ARW",
            ArwBuilder::new("ILCE-7RM4", 96, 64).build(&gradient_mosaic(96, 64, 512, 16000, 1)),
        ),
        (
            "packed12.ARW",
            ArwBuilder::new("DSLR-A900", 96, 64)
                .payload(Packed12)
                .build(&gradient_mosaic(96, 64, 128, 4000, 2)),
        ),
        (
            "craw.ARW",
            ArwBuilder::new("ILCE-7M3", 96, 64)
                .craw(encoder.clone())
                .build(&gradient_mosaic(96, 64, 512, encoder.max_value(), 3)),
        ),
        (
            "lossless_l.ARW",
            // Whole tiles (rawler needs them) and not exactly four of them (LibRaw takes a file
            // with four tile offsets for a Sinar 4-shot file).
            ArwBuilder::new("ILCE-1", 1024, 512)
                .payload(LosslessL::new())
                .build(&gradient_mosaic(1024, 512, 512, 16000, 4)),
        ),
    ];
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).expect("write corpus file");
    }
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

    let report = read_json(&report_path);
    assert_eq!(report["files"], serde_json::json!([]));
    assert_eq!(report["summary"]["total"], 0);
    assert_eq!(report["implemented"], true);
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

#[test]
fn a_corpus_without_a_report_argument_is_a_usage_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    oracle().args(["--corpus"]).arg(dir.path()).assert().code(2);
}

#[test]
fn synthetic_files_match_rawler() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    write_corpus(&corpus);
    let report_path = dir.path().join("report.json");
    oracle()
        .args(["--no-libraw", "--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .success();
    let report = read_json(&report_path);
    assert_eq!(report["summary"]["total"], 4);
    assert_eq!(report["summary"]["mismatched"], 0);
    for file in report["files"].as_array().expect("files") {
        let outcome = file["rawler"]["outcome"].as_str().expect("outcome");
        let variant = file["variant"].as_str().expect("variant");
        let want = if variant == "craw" {
            "within_tolerance"
        } else {
            "identical"
        };
        assert_eq!(outcome, want, "{variant}: {file}");
        assert_eq!(file["ours"]["status"], "ok");
    }
}

#[test]
fn synthetic_files_match_libraw_exactly() {
    if !rawpy_available() {
        eprintln!("SKIP: python3 with rawpy is not available");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    write_corpus(&corpus);
    let report_path = dir.path().join("report.json");
    oracle()
        .args(["--no-rawler", "--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .success();
    let report = read_json(&report_path);
    assert_eq!(report["summary"]["mismatched"], 0);
    for file in report["files"].as_array().expect("files") {
        assert_eq!(file["libraw"]["outcome"], "identical", "{file}");
        assert_eq!(file["libraw"]["pixels"]["mismatched"], 0);
        assert_eq!(file["libraw"]["levels"]["black"]["equal"], true, "{file}");
        assert_eq!(
            file["libraw"]["levels"]["white_balance"]["equal"], true,
            "{file}"
        );
    }
}

#[test]
fn a_file_that_cannot_be_read_fails_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    std::fs::create_dir(&corpus).expect("create corpus folder");
    std::fs::write(corpus.join("broken.ARW"), b"not a raw file at all").expect("write");
    let report_path = dir.path().join("report.json");
    oracle()
        .args(["--no-libraw", "--no-rawler", "--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .code(1);
    let report = read_json(&report_path);
    assert_eq!(report["files"][0]["ours"]["status"], "error");
    assert_eq!(report["files"][0]["ours"]["kind"], "not_arw");
}

#[test]
fn dump_writes_little_endian_samples_and_a_sidecar() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mosaic = gradient_mosaic(64, 48, 512, 16000, 9);
    let file = dir.path().join("a.ARW");
    std::fs::write(&file, ArwBuilder::new("ILCE-7RM4", 64, 48).build(&mosaic)).expect("write");
    let out = dir.path().join("raw.bin");
    oracle()
        .arg("dump")
        .arg(&file)
        .arg("--out")
        .arg(&out)
        .assert()
        .success();
    let bytes = std::fs::read(&out).expect("dump written");
    assert_eq!(bytes.len(), 64 * 48 * 2);
    let samples: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(samples, mosaic);
    let sidecar = read_json(&dir.path().join("raw.bin.json"));
    assert_eq!(sidecar["decode"], "uncompressed14");
    assert_eq!(sidecar["width"], 64);
    assert_eq!(sidecar["height"], 48);
    assert_eq!(
        sidecar["black_rggb"],
        serde_json::json!([512.0, 512.0, 512.0, 512.0])
    );
    assert_eq!(
        sidecar["wb_as_shot_rggb"],
        serde_json::json!([2043.0, 1024.0, 1024.0, 1755.0])
    );
}

#[test]
fn a_libraw_that_refuses_every_file_is_not_a_pass() {
    if !rawpy_available() {
        eprintln!("SKIP: python3 with rawpy is not available");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("arw");
    write_corpus(&corpus);
    // A stand-in for rawpy_compare.py that fails the way a broken install would.
    let script = dir.path().join("refuses.py");
    std::fs::write(
        &script,
        "import json\nprint(json.dumps({'kind': 'error', 'error': 'LibRaw says no'}))\n",
    )
    .expect("write the stub");
    let report_path = dir.path().join("report.json");
    oracle()
        .args(["--no-rawler", "--no-libjpeg", "--script"])
        .arg(&script)
        .args(["--corpus"])
        .arg(&corpus)
        .args(["--report"])
        .arg(&report_path)
        .assert()
        .code(2)
        .stderr(predicates::str::contains("nothing was compared"));
    // The report is still written, and says what happened to each file.
    let report = read_json(&report_path);
    assert_eq!(report["summary"]["total"], 4);
    for file in report["files"].as_array().expect("files") {
        assert_eq!(file["libraw"]["outcome"], "oracle_error", "{file}");
    }
}
