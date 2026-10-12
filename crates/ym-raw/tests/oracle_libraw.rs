//! LibRaw as an independent oracle for the container and metadata reader.
//!
//! There are no real ARW files in the test environment, so this test writes synthetic files for
//! several bodies with the test-only writer (`ym_raw::testgen`), asks LibRaw (through
//! `tools/oracle/rawpy_meta.py`, a separate process) what it sees, and compares every field that
//! both sides define with what `ym_raw::metadata::probe` reports. Both the writer and the reader
//! must agree with LibRaw's understanding of real files for the comparison to pass.
//!
//! The test runs only when `python3 -c "import rawpy"` succeeds. Otherwise it prints a skip
//! message and passes, unless `YM_REQUIRE_ORACLE=1` is set (the oracle CI job sets it).
//!
//! Where LibRaw legitimately differs, the test says so instead of comparing:
//!
//! - `white_level` in rawpy is LibRaw's `maximum` (`2^bits - 1`); Yellowmoon's `white` is the
//!   SR2 white level (tag 0x787F), which is LibRaw's `linear_max` (rawpy
//!   `camera_white_level_per_channel`). The latter is compared.
//! - LibRaw applies per-model margins to `width`, `height` and the left and top margins (a table
//!   keyed by sensor width). Yellowmoon reads the margins from the file (`SonyRawImageSize`) and
//!   from the quirks hook. Only `raw_width` and `raw_height` are compared.
//! - LibRaw's `timestamp` is the naive local time; Yellowmoon also keeps the sub-second and the
//!   UTC offset. The date and time of day are compared.
//! - LibRaw maps ISO from `ISOSpeedRatings` (and, when that saturates at 65535, from the
//!   recommended exposure index). Yellowmoon prefers the Sony maker-note ISO. For the cases here
//!   the sources agree, which the test asserts; the saturated case has its own expectation.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use ym_raw::exif::IsoSource;
use ym_raw::layout::CfaPattern;
use ym_raw::metadata::probe;
use ym_raw::testgen::{ArwBuilder, SonyIsoTable, Sr2Variant, gradient_mosaic};

fn script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/rawpy_meta.py")
}

fn rawpy_available() -> bool {
    Command::new("python3")
        .args(["-c", "import rawpy"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Returns false (after printing why) when the oracle cannot run.
fn oracle_ready() -> bool {
    if rawpy_available() {
        return true;
    }
    if std::env::var_os("YM_REQUIRE_ORACLE").is_some() {
        panic!("YM_REQUIRE_ORACLE is set but python3 cannot import rawpy");
    }
    eprintln!("SKIP oracle_libraw: python3 with rawpy is not available");
    false
}

fn libraw_view(path: &Path) -> Value {
    let output = Command::new("python3")
        .arg(script())
        .arg(path)
        .arg("--pixels")
        .output()
        .expect("run python3");
    let text = String::from_utf8_lossy(&output.stdout);
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("script output is not JSON ({e}): {text}"));
    assert!(
        value.get("error").is_none(),
        "LibRaw could not read {}: {}",
        path.display(),
        value["error"]
    );
    value
}

fn sha256_hex(bytes: &[u8]) -> String {
    // A tiny SHA-256 so the test needs no extra dependency. Standard FIPS 180-4.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array, got {value}"))
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

fn assert_close(label: &str, ours: f64, theirs: f64, relative: f64) {
    let tolerance = relative * theirs.abs().max(1e-12);
    assert!(
        (ours - theirs).abs() <= tolerance,
        "{label}: ours {ours} vs LibRaw {theirs}"
    );
}

/// One body to compare.
struct Case {
    name: &'static str,
    builder: ArwBuilder,
    width: u32,
    height: u32,
    black_floor: u16,
    /// Our expectation for the ISO and where it comes from, when LibRaw's value differs.
    expected_iso: Option<(u32, IsoSource)>,
    /// Orientation written to the file.
    orientation: u16,
}

fn cases() -> Vec<Case> {
    let case = |name, builder: ArwBuilder, width, height| Case {
        name,
        builder,
        width,
        height,
        black_floor: 512,
        expected_iso: None,
        orientation: 1,
    };
    vec![
        case(
            "ILCE-7RM4 defaults",
            ArwBuilder::new("ILCE-7RM4", 128, 96),
            128,
            96,
        ),
        Case {
            expected_iso: Some((12_800, IsoSource::SonyMakerNote)),
            ..case(
                "ILCE-1 with levels repeated in the raw IFD",
                ArwBuilder::new("ILCE-1", 192, 128)
                    .iso(12_800)
                    .raw_ifd_levels(true)
                    .black([510, 512, 514, 516])
                    .white([16200, 16250, 16300])
                    .wb(Some([2310, 1024, 1030, 1480]))
                    .exposure_time(1, 4000)
                    .f_number(28, 10)
                    .focal_length(850, 10),
                192,
                128,
            )
        },
        case(
            "ILCE-7M3 with legacy SR2 tags",
            ArwBuilder::new("ILCE-7M3", 128, 96)
                .sr2(Sr2Variant::Legacy)
                .wb(Some([2150, 1024, 1024, 1720]))
                .iso(3200)
                .exposure_time(1, 60),
            128,
            96,
        ),
        case(
            "ILCE-6400 APS-C, long exposure",
            ArwBuilder::new("ILCE-6400", 160, 112)
                .iso(100)
                .exposure_time(30, 1)
                .f_number(56, 10)
                .focal_length(160, 10)
                .black([512, 512, 512, 512])
                .white([16383, 16383, 16383]),
            160,
            112,
        ),
        case(
            "DSC-RX100M4 compact",
            ArwBuilder::new("DSC-RX100M4", 128, 96)
                .iso(125)
                .exposure_time(1, 1000)
                .f_number(18, 10)
                .focal_length(88, 10)
                .lens_model(None),
            128,
            96,
        ),
        Case {
            // EXIF ISOSpeedRatings is 16 bit and saturates at 65535. LibRaw falls back to the
            // recommended exposure index; Yellowmoon prefers the exact maker-note value.
            expected_iso: Some((102_400, IsoSource::SonyMakerNote)),
            ..case(
                "ILCE-1 ISO 102400",
                ArwBuilder::new("ILCE-1", 128, 96).iso(102_400),
                128,
                96,
            )
        },
        Case {
            orientation: 6,
            ..case(
                "ILCE-7M4 portrait",
                ArwBuilder::new("ILCE-7M4", 96, 128).orientation(6),
                96,
                128,
            )
        },
        case(
            "ILCE-7M3 with the SONY DSC maker-note header",
            ArwBuilder::new("ILCE-7M3", 128, 96)
                .maker_note_header(b"SONY DSC \0\0\0")
                .iso(800),
            128,
            96,
        ),
        Case {
            // Laid out like the real ILCE-9 file in github.com/lclevy/sony_raw: the measured
            // SonyISO word 2890 is ISO 2619 while the EXIF tags say 2500; the SR2 white level
            // (15360) differs from the raw IFD's DNG WhiteLevel (16383); 0x7313 is signed.
            expected_iso: Some((2619, IsoSource::SonyMakerNote)),
            ..case(
                "ILCE-9 as in the lclevy dump",
                ArwBuilder::new("ILCE-9", 96, 64)
                    .iso(2500)
                    .sony_iso_word(2890)
                    .white([15360; 3])
                    .wb(Some([2364, 1024, 1024, 1728]))
                    .raw_ifd_levels(true)
                    .raw_ifd_white_level(16383)
                    .file_format([3, 3, 3, 0])
                    .exposure_time(1, 60)
                    .f_number(56, 10)
                    .focal_length(240, 10),
                96,
                64,
            )
        },
        case(
            "ILCE-7RM3 without the maker-note ISO table",
            ArwBuilder::new("ILCE-7RM3", 128, 96)
                .iso(640)
                .iso_table(SonyIsoTable::None),
            128,
            96,
        ),
    ]
}

#[test]
fn libraw_agrees_with_the_reader_on_synthetic_files_of_several_bodies() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for (index, case) in cases().into_iter().enumerate() {
        let mosaic = gradient_mosaic(
            case.width,
            case.height,
            case.black_floor,
            16_000,
            100 + index as u64,
        );
        let path = dir.path().join(format!("case{index}.ARW"));
        case.builder
            .write_to(&path, &mosaic)
            .expect("write synthetic ARW");

        let ours = probe(&path).unwrap_or_else(|e| panic!("{}: probe failed: {e}", case.name));
        let theirs = libraw_view(&path);
        let name = case.name;

        // --- identity ---
        assert_eq!(
            theirs["make"].as_str(),
            Some(ours.make.as_str()),
            "{name}: make"
        );
        assert_eq!(
            theirs["model"].as_str(),
            Some(ours.model.as_str()),
            "{name}: model"
        );

        // --- geometry ---
        assert_eq!(
            theirs["raw_width"].as_u64(),
            Some(u64::from(ours.width)),
            "{name}: raw_width"
        );
        assert_eq!(
            theirs["raw_height"].as_u64(),
            Some(u64::from(ours.height)),
            "{name}: raw_height"
        );
        assert_eq!(
            theirs["cfa"].as_str(),
            Some(ours.cfa.name().as_str()),
            "{name}: CFA pattern"
        );
        assert_eq!(ours.cfa, CfaPattern::RGGB, "{name}");

        // --- exposure ---
        let expected_iso = case
            .expected_iso
            .map_or(theirs["iso"].as_f64().expect("iso") as u32, |(iso, _)| iso);
        assert_eq!(ours.iso, Some(expected_iso), "{name}: ISO");
        if let Some((_, source)) = case.expected_iso {
            assert_eq!(ours.iso_source, Some(source), "{name}: ISO source");
        }
        if case.expected_iso.is_none() {
            // Whenever LibRaw reads a plain ISO, ours is the same number.
            assert_eq!(
                f64::from(ours.iso.expect("iso")),
                theirs["iso"].as_f64().expect("iso"),
                "{name}: ISO vs LibRaw"
            );
        }
        assert_close(
            &format!("{name}: shutter"),
            ours.exposure_time_s.expect("exposure time"),
            theirs["shutter_s"].as_f64().expect("shutter"),
            1e-6,
        );
        assert_close(
            &format!("{name}: aperture"),
            ours.f_number.expect("f-number"),
            theirs["aperture"].as_f64().expect("aperture"),
            1e-6,
        );
        assert_close(
            &format!("{name}: focal length"),
            ours.focal_length_mm.expect("focal length"),
            theirs["focal_length_mm"].as_f64().expect("focal"),
            1e-6,
        );
        let t = ours.capture_time.expect("capture time");
        let ours_time = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            t.year, t.month, t.day, t.hour, t.minute, t.second
        );
        assert_eq!(
            theirs["timestamp"].as_str(),
            Some(ours_time.as_str()),
            "{name}: timestamp"
        );
        // LibRaw reports the EXIF orientation as dcraw's flip code.
        let flip = match case.orientation {
            1 => 0,
            2 => 1,
            3 => 3,
            4 => 2,
            5 => 4,
            6 => 6,
            7 => 7,
            _ => 5,
        };
        assert_eq!(
            theirs["flip"].as_i64(),
            Some(flip),
            "{name}: orientation / flip"
        );
        assert_eq!(ours.orientation, case.orientation, "{name}: orientation");

        // --- levels ---
        let black_theirs = floats(&theirs["black_level_rggb"]);
        for (i, (ours_v, theirs_v)) in ours.black.iter().zip(black_theirs.iter()).enumerate() {
            assert_eq!(
                f64::from(*ours_v),
                *theirs_v,
                "{name}: black level at CFA position {i}"
            );
        }
        let white_theirs = floats(&theirs["camera_white_level_rggb"]);
        for (i, (ours_v, theirs_v)) in ours.white.iter().zip(white_theirs.iter()).enumerate() {
            assert_eq!(
                f64::from(*ours_v),
                *theirs_v,
                "{name}: white level at CFA position {i}"
            );
        }
        let wb_theirs = floats(&theirs["wb_rggb"]);
        let wb_ours = ours.wb_as_shot.expect("white balance");
        for (i, (ours_v, theirs_v)) in wb_ours.iter().zip(wb_theirs.iter()).enumerate() {
            assert_eq!(
                f64::from(*ours_v),
                *theirs_v,
                "{name}: white balance at CFA position {i}"
            );
        }

        // --- the pixels the writer put in the file are the pixels LibRaw reads ---
        let bytes: Vec<u8> = mosaic.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            theirs["raw_image_sha256"].as_str(),
            Some(sha256_hex(&bytes).as_str()),
            "{name}: LibRaw's raw_image differs from the mosaic the writer was given"
        );
    }
}

#[test]
fn libraw_reads_what_the_reader_calls_the_model_id() {
    if !oracle_ready() {
        return;
    }
    // LibRaw's own model-ID table (setSonyBodyFeatures) recognises the IDs; the C++ API exposes
    // them but rawpy does not, so check indirectly: LibRaw names the camera body in
    // `lens.makernotes`, not available here. Compare the make/model strings instead and make sure
    // that an ID that disagrees with the Model string does not stop LibRaw from reading the file.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("odd.ARW");
    let mosaic = gradient_mosaic(96, 64, 512, 16_000, 1);
    ArwBuilder::new("ILCE-7M3", 96, 64)
        .model_id(384) // an A1 ID in an A7 III file
        .write_to(&path, &mosaic)
        .expect("write");
    let ours = probe(&path).expect("probe");
    assert_eq!(ours.model, "ILCE-7M3");
    assert_eq!(ours.model_id, Some(384));
    let theirs = libraw_view(&path);
    assert_eq!(theirs["model"].as_str(), Some("ILCE-7M3"));
}

/// The thumbnail LibRaw extracts is the JPEG the reader reports when the file has only a
/// thumbnail. LibRaw prefixes an Exif segment to what it returns, so the comparison is on the
/// tail of the data (everything after the SOI marker of the JPEG in the file).
#[test]
fn libraw_extracts_the_same_thumbnail_jpeg_the_reader_locates() {
    if !oracle_ready() {
        return;
    }
    use ym_raw::metadata::read_preview;
    use ym_raw::testgen::ThumbnailIfd;

    let dir = tempfile::tempdir().expect("tempdir");
    for (name, place) in [("ifd1", ThumbnailIfd::Ifd1), ("ifd0", ThumbnailIfd::Ifd0)] {
        let path = dir.path().join(format!("thumb_{name}.ARW"));
        let mosaic = gradient_mosaic(96, 64, 512, 16_000, 5);
        ArwBuilder::new("ILCE-7M3", 96, 64)
            .thumbnail(Some((200, 150)))
            .thumbnail_ifd(place)
            .preview(None)
            .write_to(&path, &mosaic)
            .expect("write");
        let ours = read_preview(&path).expect("preview").expect("a preview");
        assert_eq!(&ours[..3], &[0xFF, 0xD8, 0xFF]);
        let tail = &ours[2..];
        let output = Command::new("python3")
            .arg(script())
            .arg(&path)
            .args(["--thumb-tail", &tail.len().to_string()])
            .output()
            .expect("run python3");
        let theirs: Value = serde_json::from_slice(&output.stdout).expect("JSON");
        assert_eq!(
            theirs["thumb_format"].as_str(),
            Some("JPEG"),
            "{name}: {theirs}"
        );
        assert_eq!(
            theirs["thumb_tail_sha256"].as_str(),
            Some(sha256_hex(tail).as_str()),
            "{name}: LibRaw's thumbnail differs from the JPEG the reader located"
        );
    }
}

/// With both a thumbnail (IFD1) and the larger maker-note preview (tag 0x2001), LibRaw returns the
/// thumbnail while the reader reports the larger JPEG, because its job is a fast first preview.
/// This test records that difference so it cannot change silently.
#[test]
fn the_reader_prefers_the_larger_maker_note_preview_where_libraw_takes_the_thumbnail() {
    if !oracle_ready() {
        return;
    }
    use ym_raw::metadata::read_preview;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("both.ARW");
    let mosaic = gradient_mosaic(96, 64, 512, 16_000, 5);
    ArwBuilder::new("ILCE-7M3", 96, 64)
        .thumbnail(Some((80, 60)))
        .preview(Some((640, 427)))
        .write_to(&path, &mosaic)
        .expect("write");
    let ours = read_preview(&path).expect("preview").expect("a preview");
    let output = Command::new("python3")
        .arg(script())
        .arg(&path)
        .args(["--thumb-tail", "16"])
        .output()
        .expect("run python3");
    let theirs: Value = serde_json::from_slice(&output.stdout).expect("JSON");
    let libraw_len = theirs["thumb_len"].as_u64().expect("thumb length");
    assert!(
        ours.len() as u64 > libraw_len,
        "reader preview ({} bytes) should be larger than LibRaw's thumbnail ({libraw_len})",
        ours.len()
    );
}
