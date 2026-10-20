//! End-to-end tests of `ym info` on synthetic ARW files written with `ym_raw::testgen`.

mod common;

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serde_json::Value;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{ArwBuilder, ContainerOnly, GpsSpec, Sr2Variant, gradient_mosaic, tiny_jpeg};

/// The binary with the shipped data folder and an app folder that does not exist (`info` reads
/// the preferences for the default look and writes nothing), and Apple Auto not available, as for
/// every test of this crate (`common`).
fn ym() -> Command {
    let mut cmd = cargo_bin_cmd!("ym");
    cmd.env("YELLOWMOON_DATA", common::data_dir())
        .env(
            "YELLOWMOON_HOME",
            std::env::temp_dir().join("ym-cli-info-tests-no-such-home"),
        )
        .env_remove("RUST_LOG")
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1);
    cmd
}

fn write(dir: &Path, name: &str, builder: &ArwBuilder, w: u32, h: u32) -> PathBuf {
    let path = dir.join(name);
    builder
        .write_to(&path, &gradient_mosaic(w, h, 512, 16_000, 11))
        .expect("write synthetic ARW");
    path
}

fn a7r4(dir: &Path) -> PathBuf {
    write(
        dir,
        "DSC00001.ARW",
        &ArwBuilder::new("ILCE-7RM4", 128, 96),
        128,
        96,
    )
}

fn json_lines(stdout: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(stdout)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON ({e}): {l}")))
        .collect()
}

#[test]
fn text_output_names_the_camera_exposure_image_and_levels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = a7r4(dir.path());
    let assert = ym().arg("info").arg(&path).assert().success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    for expected in [
        "DSC00001.ARW",
        "SONY ILCE-7RM4 (model ID 375)",
        "ILCE-7RM4 v1.00, ARW 2.2",
        "FE 24-70mm F2.8 GM",
        "1/250 s  f/4  ISO 400  -0.7 EV  35 mm",
        "2026-10-04 12:03:44 +02:00",
        "128 x 96, 14 bits, 14-bit uncompressed (RGGB)",
        "512 512 512 512  (sr2)",
        "16300 16300 16300 16300  (sr2)",
        "2043 1024 1024 1755  (as shot, sr2)",
        "vignetting, chromatic aberration, distortion",
        "JPEG,",
    ] {
        assert!(out.contains(expected), "missing {expected:?} in:\n{out}");
    }
    assert!(assert.get_output().stderr.is_empty());
}

#[test]
fn json_output_has_one_object_per_file_in_the_documented_shape() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = a7r4(dir.path());
    let assert = ym().args(["info", "--json"]).arg(&path).assert().success();
    let lines = json_lines(&assert.get_output().stdout);
    assert_eq!(lines.len(), 1);
    let v = &lines[0];
    assert_eq!(v["status"], "ok");
    assert_eq!(v["error"], Value::Null);
    assert_eq!(v["input"], path.display().to_string());
    assert_eq!(v["camera"]["make"], "SONY");
    assert_eq!(v["camera"]["model"], "ILCE-7RM4");
    assert_eq!(v["camera"]["model_id"], 375);
    assert_eq!(v["camera"]["body"], "Sony A7R IV");
    assert_eq!(v["camera"]["profile"], "sony-ilce-7rm4");
    assert_eq!(v["camera"]["match"], "exact");
    assert_eq!(v["camera"]["stand_in"], false);
    assert_eq!(v["exposure"]["iso"], 400);
    assert_eq!(v["exposure"]["iso_source"], "sony_maker_note");
    assert_eq!(v["exposure"]["aperture"], 4.0);
    assert_eq!(v["exposure"]["captured"], "2026-10-04T12:03:44.070+02:00");
    assert_eq!(v["dimensions"]["width"], 128);
    assert_eq!(v["dimensions"]["height"], 96);
    assert_eq!(v["raw"]["decode"], "uncompressed14");
    assert_eq!(v["raw"]["cfa"], "RGGB");
    assert_eq!(v["black"], serde_json::json!([512.0, 512.0, 512.0, 512.0]));
    assert_eq!(
        v["white"],
        serde_json::json!([16300.0, 16300.0, 16300.0, 16300.0])
    );
    assert_eq!(
        v["wb_as_shot"],
        serde_json::json!([2043.0, 1024.0, 1024.0, 1755.0])
    );
    assert_eq!(v["levels_source"]["black"], "sr2");
    assert_eq!(v["lens_corrections"]["distortion"], true);
    assert!(v["preview"]["length"].as_u64().expect("preview length") > 100);
    assert!(v.get("output").is_none(), "info has no output");
    assert!(v.get("timings_ms").is_none(), "info has no timings");
    assert!(v["pipeline_version"].is_string());
}

/// Rounds every non-integer number to four significant digits, so the snapshot does not depend on
/// the last digits of the estimates (which differ between platforms' maths libraries).
fn round_floats(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().expect("f64");
            if x != 0.0 && x.is_finite() {
                let rounded: f64 = format!("{x:.3e}").parse().expect("number");
                *v = serde_json::json!(rounded);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(round_floats),
        Value::Object(map) => map.values_mut().for_each(round_floats),
        _ => {}
    }
}

#[test]
fn json_snapshot_of_a_synthetic_a7r_iv() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = a7r4(dir.path());
    // Apple Auto off: the same object on every machine (where it is not available the reason
    // differs between Linux, an Intel Mac and an Apple silicon Mac; `tests/apple_auto.rs`).
    let assert = ym()
        .args(["info", "--json", "--apple-auto", "off"])
        .arg(&path)
        .assert()
        .success();
    let mut v = json_lines(&assert.get_output().stdout).remove(0);
    // The path differs per run and the JPEG encoder's output size may change with its version.
    v["input"] = Value::from("[path]");
    v["preview"] = Value::from("[preview location]");
    round_floats(&mut v);
    insta::assert_json_snapshot!(v);
}

#[test]
fn the_estimates_are_the_ones_render_json_reports_for_the_same_file() {
    let sb = common::Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let info = sb
        .ym()
        .args(["info", "--json"])
        .arg(&arw)
        .assert()
        .success();
    let info = json_lines(&info.get_output().stdout).remove(0);
    // Reading the estimates writes nothing.
    assert!(!sb.default_export().exists());
    let render = sb
        .ym()
        .args(["render", "--json"])
        .arg(&arw)
        .assert()
        .success();
    let render = json_lines(&render.get_output().stdout).remove(0);
    for key in ["camera", "look", "auto", "settings"] {
        assert!(!info[key].is_null(), "{key}");
        // `info` has the metadata fields of the camera too; the rest is the same object.
        if key != "camera" {
            assert_eq!(info[key], render[key], "{key}");
        }
    }
    for (k, v) in render["camera"].as_object().expect("camera") {
        assert_eq!(&info["camera"][k], v, "camera.{k}");
    }
    assert_eq!(info["status"], "ok");
    assert_eq!(info["input"], render["input"]);
    assert_eq!(info["notes"], serde_json::json!([]));
    assert!(info.get("output").is_none() && info.get("timings_ms").is_none());
}

#[test]
fn a_forced_profile_is_reported_as_forced_and_a_stand_in() {
    let sb = common::Sandbox::new();
    let arw = sb.arw("a.ARW");
    let out = sb
        .ym()
        .args(["info", "--json", "--profile", "sony-ilce-7m3"])
        .arg(&arw)
        .assert()
        .success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["camera"]["profile"], "sony-ilce-7m3");
    assert_eq!(v["camera"]["match"], "forced");
    assert_eq!(v["camera"]["stand_in"], true);
    sb.ym()
        .args(["info", "--profile", "nope"])
        .arg(&arw)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "There is no camera profile called nope",
        ));
}

#[test]
fn a_body_without_a_profile_of_its_own_uses_the_default_one_and_says_so() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "A1.ARW",
        &ArwBuilder::new("ILCE-1", 128, 96),
        128,
        96,
    );
    let out = ym().args(["info", "--json"]).arg(&path).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["camera"]["body"], "Sony A1");
    assert_eq!(v["camera"]["profile"], "sony-ilce-7rm4");
    assert_ne!(v["camera"]["match"], "exact");
    assert_eq!(v["camera"]["stand_in"], true);
    let text = ym().arg("info").arg(&path).assert().success();
    let text = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(text.contains("stand-in"), "{text}");
}

#[test]
fn a_file_that_cannot_be_developed_says_why_and_has_no_estimates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "craw.ARW",
        &ArwBuilder::new("ILCE-7M3", 1024, 768).container_only(ContainerOnly::craw()),
        1024,
        768,
    );
    let out = ym().args(["info", "--json"]).arg(&path).assert().success();
    let v = json_lines(&out.get_output().stdout).remove(0);
    assert_eq!(v["status"], "ok");
    assert_eq!(v["auto"], Value::Null);
    assert_eq!(v["settings"], Value::Null);
    assert_eq!(v["look"], Value::Null);
    assert_eq!(v["camera"]["body"], "Sony A7 III");
    let notes = v["notes"].as_array().expect("notes");
    assert_eq!(notes.len(), 1, "{notes:?}");
    let text = ym().arg("info").arg(&path).assert().success();
    let text = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(text.contains("Development"), "{text}");
}

#[test]
fn several_files_with_one_failure_exit_1_and_still_report_every_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let good = a7r4(dir.path());
    let junk = dir.path().join("notes.txt");
    std::fs::write(&junk, "not a raw file").expect("write");
    let jpeg = dir.path().join("photo.jpg");
    std::fs::write(&jpeg, tiny_jpeg(16, 16, 1)).expect("write");
    let missing = dir.path().join("missing.ARW");

    let assert = ym()
        .args(["info", "--json"])
        .arg(&good)
        .arg(&junk)
        .arg(&jpeg)
        .arg(&missing)
        .assert()
        .code(1);
    let lines = json_lines(&assert.get_output().stdout);
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["status"], "ok");
    for line in &lines[1..3] {
        assert_eq!(line["status"], "error");
        assert_eq!(line["error"]["kind"], "not_arw");
        assert_eq!(
            line["error"]["message"],
            "This isn't an ARW file. Yellowmoon opens Sony ARW files only."
        );
        assert!(line.get("camera").is_none());
    }
    assert_eq!(lines[3]["status"], "error");
    assert_eq!(lines[3]["error"]["kind"], "io");
    assert!(
        assert.get_output().stderr.is_empty(),
        "JSON mode keeps stderr quiet"
    );
}

#[test]
fn text_mode_reports_failures_on_stderr_and_successes_on_stdout() {
    let dir = tempfile::tempdir().expect("tempdir");
    let good = a7r4(dir.path());
    let junk = dir.path().join("notes.txt");
    std::fs::write(&junk, "not a raw file").expect("write");
    ym().arg("info")
        .arg(&good)
        .arg(&junk)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("ILCE-7RM4"))
        .stderr(predicate::str::contains(
            "notes.txt: This isn't an ARW file.",
        ));
}

#[test]
fn a_folder_is_reported_as_an_error_not_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    ym().arg("info")
        .arg(dir.path())
        .assert()
        .code(1)
        .stderr(predicate::str::contains("folder"));
}

#[test]
fn arq_and_truncated_files_get_their_messages() {
    let dir = tempfile::tempdir().expect("tempdir");
    let arq = write(
        dir.path(),
        "pixel_shift.ARQ",
        &ArwBuilder::new("ILCE-7RM4", 64, 48).container_only(ContainerOnly::arq()),
        64,
        48,
    );
    let good = a7r4(dir.path());
    let bytes = std::fs::read(&good).expect("read");
    let cut = dir.path().join("cut.ARW");
    std::fs::write(&cut, &bytes[..bytes.len() - 500]).expect("write");

    let assert = ym()
        .args(["info", "--json"])
        .arg(&arq)
        .arg(&cut)
        .assert()
        .code(1);
    let lines = json_lines(&assert.get_output().stdout);
    assert_eq!(lines[0]["error"]["kind"], "arq");
    assert_eq!(
        lines[0]["error"]["message"],
        "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead."
    );
    assert_eq!(lines[1]["error"]["kind"], "corrupt");
}

#[test]
fn info_works_on_every_variant_even_those_that_cannot_be_decoded_yet() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cases: [(&str, ContainerOnly, &str); 5] = [
        ("craw", ContainerOnly::craw(), "craw"),
        ("lossless_l", ContainerOnly::lossless_l(), "lossless_l"),
        (
            "lossless_ms",
            ContainerOnly::lossless_ms(ChromaSubsampling::Yuv420, true),
            "lossless_ms",
        ),
        ("arw6", ContainerOnly::arw6(12), "arw6"),
        ("packed12", ContainerOnly::packed12(), "uncompressed12"),
    ];
    for (name, payload, decode) in cases {
        let path = write(
            dir.path(),
            &format!("{name}.ARW"),
            &ArwBuilder::new("ILCE-7RM5", 1024, 768).container_only(payload),
            1024,
            768,
        );
        let assert = ym().args(["info", "--json"]).arg(&path).assert().success();
        let v = json_lines(&assert.get_output().stdout).remove(0);
        assert_eq!(v["raw"]["decode"], decode, "{name}");
        // Only Lossless M and S are reduced; only ARW 6.0 has no decoder.
        assert_eq!(v["raw"]["reduced"], name == "lossless_ms", "{name}");
        assert_eq!(v["raw"]["supported"], name != "arw6", "{name}");
    }
}

#[test]
fn lossless_ms_is_shown_as_reduced_with_its_subsampling() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "ms.ARW",
        &ArwBuilder::new("ILCE-7RM5", 1024, 768)
            .container_only(ContainerOnly::lossless_ms(ChromaSubsampling::Yuv422, true)),
        1024,
        768,
    );
    let assert = ym().args(["info", "--json"]).arg(&path).assert().success();
    let v = json_lines(&assert.get_output().stdout).remove(0);
    assert_eq!(v["raw"]["reduced"], true);
    assert_eq!(v["raw"]["subsampling"], "4:2:2");
    let text = ym().arg("info").arg(&path).assert().success();
    let out = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(out.contains("Lossless M or S"), "{out}");
    assert!(out.contains("Reduced"), "{out}");
    assert!(out.contains("4:2:2 YCbCr"), "{out}");
    // A mosaic file has no such row.
    let plain = write(
        dir.path(),
        "plain.ARW",
        &ArwBuilder::new("ILCE-7M3", 64, 48),
        64,
        48,
    );
    let text = ym().arg("info").arg(&plain).assert().success();
    let out = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(!out.contains("Reduced"), "{out}");
}

#[test]
fn arw6_is_flagged_as_not_decodable_yet() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "a7v.ARW",
        &ArwBuilder::new("ILCE-7M5", 1024, 768).container_only(ContainerOnly::arw6(12)),
        1024,
        768,
    );
    let text = ym().arg("info").arg(&path).assert().success();
    let out = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(out.contains("ARW 6.0 compressed"), "{out}");
    assert!(out.contains("not available yet"), "{out}");
}

#[test]
fn a_craw_file_does_not_claim_a_bit_depth_the_data_does_not_have() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "craw.ARW",
        &ArwBuilder::new("ILCE-7M3", 1024, 768).container_only(ContainerOnly::craw()),
        1024,
        768,
    );
    let text = ym().arg("info").arg(&path).assert().success();
    let out = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(
        out.contains("1024 x 768, compressed (lossy cRAW) (RGGB)"),
        "{out}"
    );
    // The levels are on the 14-bit scale of the decoded data, as LibRaw reports them.
    assert!(out.contains("512 512 512 512  (sr2)"), "{out}");
    assert!(out.contains("16300 16300 16300 16300  (sr2)"), "{out}");
}

#[test]
fn optional_fields_are_null_or_absent_when_the_file_lacks_them() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "bare.ARW",
        &ArwBuilder::new("ILCE-7M3", 64, 48)
            .sr2(Sr2Variant::Absent)
            .wb(None)
            .lens_model(None)
            .thumbnail(None)
            .preview(None)
            .lens_tables(ym_raw::testgen::LensTables {
                vignetting: None,
                chromatic_aberration: None,
                distortion: None,
            }),
        64,
        48,
    );
    let assert = ym().args(["info", "--json"]).arg(&path).assert().success();
    let v = json_lines(&assert.get_output().stdout).remove(0);
    assert_eq!(v["wb_as_shot"], Value::Null);
    assert_eq!(v["preview"], Value::Null);
    assert_eq!(v["lens_corrections"], Value::Null);
    assert_eq!(v["exposure"]["lens_model"], Value::Null);
    assert_eq!(v["levels_source"]["black"], "default");

    let text = ym().arg("info").arg(&path).assert().success();
    let out = String::from_utf8_lossy(&text.get_output().stdout).into_owned();
    assert!(out.contains("Lens data"), "{out}");
    assert!(out.contains("none"), "{out}");
}

#[test]
fn gps_location_is_shown_when_the_file_has_a_fix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write(
        dir.path(),
        "gps.ARW",
        &ArwBuilder::new("ILCE-7M3", 64, 48).gps(GpsSpec {
            latitude_deg: 38.7223,
            longitude_deg: -9.1393,
            altitude_m: 12.0,
        }),
        64,
        48,
    );
    ym().arg("info")
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("38.72230, -9.13930"));
}

#[test]
fn no_files_is_a_usage_error() {
    ym().arg("info")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn two_files_in_text_mode_are_separated_by_a_blank_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = a7r4(dir.path());
    let b = write(
        dir.path(),
        "DSC00002.ARW",
        &ArwBuilder::new("ILCE-1", 128, 96),
        128,
        96,
    );
    let assert = ym().arg("info").arg(&a).arg(&b).assert().success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
    assert!(out.contains("\n\n"), "{out}");
    assert!(
        out.contains("ILCE-7RM4") && out.contains("ILCE-1 (model ID 384)"),
        "{out}"
    );
}

#[test]
fn shutter_speeds_between_a_quarter_second_and_one_second_are_decimals() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (numerator, denominator, shown) in [
        (8u32, 10u32, "0.8 s"),
        (4, 10, "0.4 s"),
        (1, 2, "0.5 s"),
        (13, 10, "1.3 s"),
        (1, 4, "1/4 s"),
        (1, 8000, "1/8000 s"),
    ] {
        let path = write(
            dir.path(),
            &format!("t{numerator}_{denominator}.ARW"),
            &ArwBuilder::new("ILCE-7M3", 64, 48).exposure_time(numerator, denominator),
            64,
            48,
        );
        let assert = ym().arg("info").arg(&path).assert().success();
        let out = String::from_utf8_lossy(&assert.get_output().stdout).into_owned();
        let line = out
            .lines()
            .find(|l| l.trim_start().starts_with("Exposure"))
            .unwrap_or_else(|| panic!("no Exposure row:\n{out}"));
        assert!(
            line.contains(&format!("{shown}  f/4")),
            "{numerator}/{denominator} should read {shown}: {line}"
        );
    }
}

/// `ym info *.ARW --json | head -n 1`: the reader goes away after the first line. The
/// tool stops without a panic message and without an error exit.
#[cfg(unix)]
#[test]
fn a_reader_that_goes_away_ends_the_run_quietly() {
    use std::process::Stdio;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = a7r4(dir.path());
    // Enough output that the writes after the close cannot all sit in a pipe buffer.
    let files: Vec<&Path> = std::iter::repeat_n(path.as_path(), 2000).collect();
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("ym"))
        .args(["info", "--json"])
        .args(&files)
        .env(common::NO_APPLE_AUTO.0, common::NO_APPLE_AUTO.1)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ym");
    // Close the read end without reading anything.
    drop(child.stdout.take());
    let output = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.is_empty(), "stderr: {stderr}");
    assert_eq!(output.status.code(), Some(0), "status {:?}", output.status);
}
