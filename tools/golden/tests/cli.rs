//! The `golden` binary: `synth` end to end, and the stub runner.

use std::fs;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use ym_raw::testgen::{gradient_mosaic, tiny_jpeg};

#[test]
fn golden_without_a_subcommand_is_the_stub() {
    Command::cargo_bin("golden")
        .unwrap()
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Not implemented yet"));
}

#[test]
fn golden_synth_writes_files_and_prints_a_table() {
    let dir = tempfile::tempdir().unwrap();
    let scenes = dir.path().join("scenes");
    let out = dir.path().join("arw");
    fs::create_dir_all(&scenes).unwrap();
    let mosaic = gradient_mosaic(64, 32, 512, 16383, 3);
    let raw: Vec<u8> = mosaic.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(scenes.join("s.bin"), &raw).unwrap();
    fs::write(scenes.join("s.jpg"), tiny_jpeg(64, 32, 9)).unwrap();
    let sidecar = json!({
        "id": "s", "tier": "synthetic",
        "mosaic": { "file": "s.bin", "width": 64, "height": 32, "cfa": "RGGB", "bits": 14, "black": 512,
                    "white": 16383, "sha256": hex::encode(Sha256::digest(&raw)) },
        "writer": {
            "make": "SONY", "model": "ILCE-7M3", "model_id": 363, "lens_model": null, "focal_length_mm": 50.0,
            "iso": 100, "exposure_time": [1, 125], "f_number": [80, 10], "exposure_comp_ev": 0.0,
            "capture_time": "2026:10:12 12:00:00",
            "sr2": { "black": [512, 512, 512, 512], "white": [16383, 16383, 16383], "wb_rggb": [2000, 1024, 1024, 1700] },
            "preview": { "file": "s.jpg", "width": 64, "height": 32 },
            "variants": ["uncompressed14"], "software": "synthetic"
        }
    });
    fs::write(scenes.join("s.json"), sidecar.to_string()).unwrap();
    Command::cargo_bin("golden")
        .unwrap()
        .args(["synth", "--scenes"])
        .arg(&scenes)
        .arg("--out")
        .arg(&out)
        .args(["--variants", "uncompressed14,craw,lossless-l"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("s__craw.ARW").and(predicate::str::contains("within qstep/2")),
        )
        .stdout(predicate::str::contains("3 files"));
    assert!(out.join("s__lossless-l.ARW").exists());
    assert!(out.join("index.json").exists());
}

#[test]
fn golden_synth_without_scenes_fails_with_a_hint() {
    let dir = tempfile::tempdir().unwrap();
    Command::cargo_bin("golden")
        .unwrap()
        .args(["synth", "--scenes"])
        .arg(dir.path())
        .arg("--out")
        .arg(dir.path().join("arw"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("make_scenes"));
}
