//! `golden synth` on small scenes written by the tests themselves (no Python, no fetched data): every variant
//! round-trips, the preview is embedded byte for byte, impossible requests are skipped with a reason, and the checks
//! catch a wrong file (negative controls).

use std::fs;
use std::path::Path;

use golden::preview::{jpeg_size, pad_exact};
use golden::scene::Scene;
use golden::synth::{self, SynthOptions, Variant, Verify};
use serde_json::json;
use sha2::{Digest, Sha256};
use ym_raw::metadata::{probe_bytes, read_preview};
use ym_raw::testgen::{gradient_mosaic, tiny_jpeg};

const PREVIEW: (u16, u16) = (192, 128);

/// Writes `<id>.bin`, `<id>.jpg` and `<id>.json` of a small scene into `dir`.
fn write_scene(
    dir: &Path,
    id: &str,
    (w, h): (u32, u32),
    bits: u16,
    cfa: &str,
    variants: &[&str],
) -> Vec<u16> {
    let (black, white) = if bits == 12 {
        (128u16, 4095u16)
    } else {
        (512, 16383)
    };
    let mosaic = gradient_mosaic(w, h, black, white, 7 + u64::from(bits));
    let raw: Vec<u8> = mosaic.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join(format!("{id}.bin")), &raw).unwrap();
    // A real JPEG that is not the builder's placeholder (another seed) and starts FF D8 FF DB.
    fs::write(
        dir.join(format!("{id}.jpg")),
        tiny_jpeg(PREVIEW.0, PREVIEW.1, 99),
    )
    .unwrap();
    let sidecar = json!({
        "id": id,
        "tier": "synthetic",
        "mosaic": {
            "file": format!("{id}.bin"), "width": w, "height": h, "cfa": cfa, "bits": bits,
            "black": black, "white": white, "sha256": hex::encode(Sha256::digest(&raw)),
        },
        "writer": {
            "make": "SONY", "model": "ILCE-7RM4", "model_id": 375, "lens_model": "FE 24-70mm F2.8 GM",
            "focal_length_mm": 35.0, "iso": 400, "exposure_time": [1, 500], "f_number": [80, 10],
            "exposure_comp_ev": 0.0, "capture_time": "2026:10:12 12:00:00",
            "sr2": { "black": [512, 512, 512, 512], "white": [16383, 16383, 16383], "wb_rggb": [3128, 1024, 1024, 1369] },
            "preview": { "file": format!("{id}.jpg"), "width": PREVIEW.0, "height": PREVIEW.1 },
            "variants": variants,
            "software": "ILCE-7RM4 v1.00 (Yellowmoon synthetic test file)",
        },
        "truth": { "ignored": true },
    });
    fs::write(
        dir.join(format!("{id}.json")),
        serde_json::to_string(&sidecar).unwrap(),
    )
    .unwrap();
    mosaic
}

/// The JPEG without the `COM` padding segments that follow its first segment.
fn strip_padding(preview: &[u8]) -> Vec<u8> {
    let first = 4 + usize::from(u16::from_be_bytes([preview[4], preview[5]]));
    let mut out = preview[..first].to_vec();
    let mut pos = first;
    while preview[pos..pos + 2] == [0xFF, 0xFE] {
        pos += 2 + usize::from(u16::from_be_bytes([preview[pos + 2], preview[pos + 3]]));
    }
    out.extend_from_slice(&preview[pos..]);
    out
}

fn options() -> SynthOptions {
    SynthOptions {
        jobs: 2,
        ..SynthOptions::default()
    }
}

#[test]
fn every_variant_round_trips_on_every_cfa_phase() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    for cfa in ["RGGB", "GRBG", "GBRG", "BGGR"] {
        write_scene(
            &scenes,
            &format!("s14-{}", cfa.to_lowercase()),
            (96, 64),
            14,
            cfa,
            &["uncompressed14", "craw", "lossless-l", "lossless-ms"],
        );
    }
    write_scene(&scenes, "s12-rggb", (96, 64), 12, "RGGB", &["packed12"]);
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert!(report.skipped.is_empty(), "{:?}", report.skipped);
    assert_eq!(report.written.len(), 4 * 4 + 1);
    for w in &report.written {
        assert!(out.join(&w.file).exists());
        assert_eq!(w.file, synth::arw_name(&w.scene, w.variant));
        assert!(w.decoded_sha256.is_some());
        match (w.variant, &w.verify) {
            (Variant::Uncompressed14 | Variant::Packed12 | Variant::LosslessL, Verify::Exact) => {}
            (Variant::Craw, Verify::WithinQstep { max_ratio }) => assert!(*max_ratio <= 1.0),
            (Variant::LosslessMs, Verify::Reduced { .. }) => {}
            other => panic!("{}: unexpected {other:?}", w.file),
        }
    }
    // the index lists them all
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("index.json")).unwrap()).unwrap();
    assert_eq!(index["files"].as_array().unwrap().len(), 17);
}

#[test]
fn the_scene_preview_is_embedded_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "p",
        (96, 64),
        14,
        "RGGB",
        &["uncompressed14", "lossless-l"],
    );
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    let jpeg = fs::read(scenes.join("p.jpg")).unwrap();
    for w in &report.written {
        let path = out.join(&w.file);
        let preview = read_preview(&path).unwrap().expect("a preview");
        assert!(
            preview.starts_with(&[0xFF, 0xD8, 0xFF, 0xDB]),
            "starts like a Sony preview"
        );
        assert_eq!(jpeg_size(&preview), Some(PREVIEW));
        assert_eq!(preview, pad_exact(&jpeg, preview.len()).unwrap());
        // padding is the only difference: dropping the COM segments after the first one gives the JPEG back
        assert_eq!(strip_padding(&preview), jpeg);
        let meta = probe_bytes(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(meta.make, "SONY");
        assert_eq!(meta.model, "ILCE-7RM4");
        assert_eq!(meta.model_id, Some(375));
        assert_eq!(meta.iso, Some(400));
        assert_eq!(meta.lens_model.as_deref(), Some("FE 24-70mm F2.8 GM"));
        assert!(
            meta.software
                .as_deref()
                .is_some_and(|s| s.contains("synthetic"))
        );
    }
}

#[test]
fn impossible_requests_are_skipped_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "narrow",
        (100, 64),
        14,
        "RGGB",
        &["craw", "lossless-l"],
    );
    write_scene(&scenes, "twelve", (96, 64), 12, "RGGB", &["packed12"]);
    // cRAW needs a width that is a multiple of 32
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(report.written.len(), 2);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].0, "narrow");
    assert!(
        report.skipped[0].2.contains("32"),
        "{}",
        report.skipped[0].2
    );
    // an explicit variant list applies where it fits: lossless-l does not fit a 12-bit mosaic
    let opts = SynthOptions {
        variants: Some(vec![Variant::LosslessL]),
        ..options()
    };
    let report = synth::run(&scenes, &out, &opts).unwrap();
    assert_eq!(report.written.len(), 1);
    assert_eq!(report.written[0].scene, "narrow");
    assert_eq!(report.skipped.len(), 1);
    assert!(
        report.skipped[0].2.contains("12-bit"),
        "{}",
        report.skipped[0].2
    );
}

#[test]
fn a_second_run_rewrites_nothing_and_gives_the_same_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "i",
        (96, 64),
        14,
        "RGGB",
        &["uncompressed14", "craw", "lossless-l", "lossless-ms"],
    );
    let first = synth::run(&scenes, &out, &options()).unwrap();
    assert!(first.written.iter().all(|w| !w.unchanged));
    let second = synth::run(&scenes, &out, &options()).unwrap();
    assert!(second.written.iter().all(|w| w.unchanged));
    for (a, b) in first.written.iter().zip(&second.written) {
        assert_eq!(
            (&a.file, &a.sha256, &a.decoded_sha256),
            (&b.file, &b.sha256, &b.decoded_sha256)
        );
    }
}

#[test]
fn a_mosaic_that_differs_from_its_sidecar_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(&scenes, "bad", (96, 64), 14, "RGGB", &["lossless-l"]);
    let bin = scenes.join("bad.bin");
    let mut raw = fs::read(&bin).unwrap();
    raw[100] ^= 1;
    fs::write(&bin, raw).unwrap();
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert!(report.written.is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(
        report.failures[0].1.contains("SHA-256"),
        "{}",
        report.failures[0].1
    );
    // a short file is refused before the hash
    fs::write(&bin, [0u8; 10]).unwrap();
    assert!(
        Scene::load(&scenes.join("bad.json"))
            .unwrap_err()
            .to_string()
            .contains("bytes")
    );
}

#[test]
fn a_preview_of_the_wrong_size_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(&scenes, "w", (96, 64), 14, "RGGB", &["lossless-l"]);
    fs::write(scenes.join("w.jpg"), tiny_jpeg(80, 48, 5)).unwrap(); // the sidecar says 96 x 64
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert_eq!(report.failures.len(), 1);
    assert!(
        report.failures[0].1.contains("192 x 128"),
        "{}",
        report.failures[0].1
    );
}

/// Negative controls: the round-trip checks must fail on a file that does not match the scene.
#[test]
fn the_round_trip_checks_detect_a_wrong_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path()).unwrap();
    write_scene(dir.path(), "n", (96, 64), 14, "RGGB", &["lossless-l"]);
    let scene = Scene::load(&dir.path().join("n.json")).unwrap();
    let mut other = scene.clone();
    for variant in [
        Variant::Uncompressed14,
        Variant::LosslessL,
        Variant::Craw,
        Variant::LosslessMs,
    ] {
        let file = synth::build_arw(&scene, variant).unwrap();
        assert!(
            synth::verify_round_trip(&scene, variant, &file).is_ok(),
            "{variant} should verify"
        );
        // one sample off by far more than any quantisation step
        other.mosaic = scene.mosaic.clone();
        other.mosaic[1000] = other.mosaic[1000].wrapping_add(3000).min(16383);
        let err = synth::verify_round_trip(&other, variant, &file);
        assert!(err.is_err(), "{variant}: a changed mosaic must not verify");
    }
    // a one-DN change is caught in the exact variants
    other.mosaic = scene.mosaic.clone();
    other.mosaic[5] ^= 1;
    let file = synth::build_arw(&scene, Variant::LosslessL).unwrap();
    let msg = synth::verify_round_trip(&other, Variant::LosslessL, &file)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("sample 5"), "{msg}");
    // and metadata that does not match the sidecar
    let mut renamed = scene.clone();
    renamed.sidecar.writer.iso = 800;
    assert!(
        synth::verify_round_trip(&renamed, Variant::LosslessL, &file)
            .unwrap_err()
            .to_string()
            .contains("ISO")
    );
    let mut wb = scene.clone();
    wb.sidecar.writer.sr2.wb_rggb[0] += 1;
    assert!(
        synth::verify_round_trip(&wb, Variant::LosslessL, &file)
            .unwrap_err()
            .to_string()
            .contains("white balance")
    );
}

#[test]
fn only_filters_scenes_and_names_are_stable() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "chart__D6500__ILCE-7RM4__iso400__rggb__14b",
        (96, 64),
        14,
        "RGGB",
        &["lossless-l"],
    );
    write_scene(
        &scenes,
        "chart__bb2850__ILCE-7RM4__iso400__rggb__14b",
        (96, 64),
        14,
        "RGGB",
        &["lossless-l"],
    );
    let opts = SynthOptions {
        only: vec!["bb2850".to_owned()],
        ..options()
    };
    let report = synth::run(&scenes, &out, &opts).unwrap();
    assert_eq!(report.written.len(), 1);
    assert_eq!(
        report.written[0].file,
        "chart__bb2850__ILCE-7RM4__iso400__rggb__14b__lossless-l.ARW"
    );
    assert!("lossless-l".parse::<Variant>().is_ok());
    assert!("lossless".parse::<Variant>().is_err());
    for v in Variant::ALL {
        assert_eq!(v.name().parse::<Variant>().unwrap(), v);
    }
}

#[test]
fn an_empty_folder_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = synth::run(dir.path(), &dir.path().join("arw"), &options()).unwrap_err();
    assert!(err.to_string().contains("make_scenes"), "{err}");
}

#[test]
fn a_variant_the_writer_does_not_know_is_a_failure_not_a_silent_omission() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "v",
        (96, 64),
        14,
        "RGGB",
        &["lossless-l", "lossless-xl"],
    );
    let report = synth::run(&scenes, &out, &options()).unwrap();
    assert_eq!(report.written.len(), 1);
    assert_eq!(report.failures.len(), 1, "{:?}", report.failures);
    assert_eq!(report.failures[0].0, "v__lossless-xl.ARW");
    assert!(
        report.failures[0].1.contains("lossless-xl"),
        "{}",
        report.failures[0].1
    );
}

#[test]
fn no_verify_keeps_the_decode_hash_of_identical_bytes_in_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let (scenes, out) = (dir.path().join("scenes"), dir.path().join("arw"));
    fs::create_dir_all(&scenes).unwrap();
    write_scene(
        &scenes,
        "k",
        (96, 64),
        14,
        "RGGB",
        &["uncompressed14", "lossless-ms"],
    );
    let read_index = || -> Vec<(String, Option<String>, String)> {
        let doc: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("index.json")).unwrap()).unwrap();
        doc["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["file"].as_str().unwrap().to_owned(),
                    f["decoded_sha256"].as_str().map(str::to_owned),
                    f["verify"]["check"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    };
    let verified = synth::run(&scenes, &out, &options()).unwrap();
    assert!(verified.failures.is_empty(), "{:?}", verified.failures);
    let before = read_index();
    assert!(before.iter().all(|(_, hash, _)| hash.is_some()));
    // the same bytes again, without decoding: the report says not checked, the index keeps what was verified
    let quick = synth::run(
        &scenes,
        &out,
        &SynthOptions {
            no_verify: true,
            ..options()
        },
    )
    .unwrap();
    assert!(
        quick
            .written
            .iter()
            .all(|w| w.verify == Verify::NotChecked && w.unchanged)
    );
    assert_eq!(read_index(), before);
    // a scene whose bytes changed has nothing verified to inherit
    let mosaic = gradient_mosaic(96, 64, 512, 16383, 1234);
    let raw: Vec<u8> = mosaic.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(scenes.join("k.bin"), &raw).unwrap();
    let mut sidecar: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(scenes.join("k.json")).unwrap()).unwrap();
    sidecar["mosaic"]["sha256"] = serde_json::Value::String(hex::encode(Sha256::digest(&raw)));
    fs::write(scenes.join("k.json"), sidecar.to_string()).unwrap();
    synth::run(
        &scenes,
        &out,
        &SynthOptions {
            no_verify: true,
            ..options()
        },
    )
    .unwrap();
    let after = read_index();
    let changed = after
        .iter()
        .find(|(f, _, _)| f == "k__uncompressed14.ARW")
        .unwrap();
    assert_eq!(changed.1, None);
    assert_eq!(changed.2, "not_checked");
}
