//! LibRaw (through rawpy) as an independent oracle for the pixel decoders.
//!
//! The synthetic files come from the test-only encoders in `ym_raw::testgen`; LibRaw decodes them
//! in a separate process (`tools/oracle/rawpy_compare.py`) and its `raw_image` must equal ours
//! sample for sample, with the same black level, white level and white balance. The decoded
//! values of every lossless variant are identical by construction; cRAW is exact in LibRaw too
//! (rawler is the one that dithers). For Lossless M and S the comparison is LibRaw's pseudo-raw
//! RGB against our decode with chroma replication and without the white-balance undo, which is
//! exactly what LibRaw's `sony_ycbcr_load_raw` produces.
//!
//! The tests skip (and print why) when `python3 -c "import rawpy, numpy"` fails, unless
//! `YM_REQUIRE_ORACLE=1` is set, as in the oracle CI job (`tools/oracle/ci.sh`).
//!
//! `real_files_match_libraw` is ignored by default: it needs ARW files in `testdata/real/files` or
//! `testdata/arw` (see `tools/oracle/README.md`). Run it with
//! `cargo test -p ym-raw --test oracle_decode --release -- --ignored`.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use ym_raw::decode::{ChromaUpsampling, DecodeOptions, RawImage, YcbcrOptions, decode_with};
use ym_raw::layout::{CfaColor, ChromaSubsampling, DecodeKind};
use ym_raw::testgen::{
    ArwBuilder, CrawEncoder, HuffmanChoice, LosslessL, LosslessMs, Packed12, gradient_mosaic,
    gradient_rgb,
};

fn script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/rawpy_compare.py")
}

fn oracle_ready() -> bool {
    let ok = Command::new("python3")
        .args(["-c", "import rawpy, numpy"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        if std::env::var_os("YM_REQUIRE_ORACLE").is_some() {
            panic!("YM_REQUIRE_ORACLE is set but python3 cannot import rawpy and numpy");
        }
        eprintln!("SKIP oracle_decode: python3 with rawpy and numpy is not available");
    }
    ok
}

/// LibRaw's metadata and pixels for a file.
fn libraw(path: &Path, scratch: &Path) -> (Value, Vec<u16>) {
    let out = scratch.join("libraw.bin");
    let output = Command::new("python3")
        .arg(script())
        .arg(path)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run python3");
    let text = String::from_utf8_lossy(&output.stdout);
    let meta: Value = serde_json::from_str(text.trim())
        .unwrap_or_else(|e| panic!("script output is not JSON ({e}): {text}"));
    assert!(
        meta["kind"] != "error",
        "LibRaw could not read {}: {}",
        path.display(),
        meta["error"]
    );
    let bytes = std::fs::read(&out).expect("read dump");
    let pixels = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    (meta, pixels)
}

/// Values in the order red, green 1, green 2, blue from the raster positions of the CFA pattern.
fn rggb(image: &RawImage, positions: [f32; 4]) -> [f32; 4] {
    let pattern = image.meta.cfa;
    let mut out = [0.0; 4];
    let mut greens = 0;
    for (color, value) in pattern.0.iter().zip(positions) {
        match color {
            CfaColor::Red => out[0] = value,
            CfaColor::Green => {
                greens += 1;
                out[greens] = value;
            }
            CfaColor::Blue => out[3] = value,
        }
    }
    out
}

fn libraw_levels(meta: &Value, key: &str) -> Option<[f32; 4]> {
    let list = meta[key].as_array()?;
    let v: Vec<f32> = list
        .iter()
        .filter_map(|x| x.as_f64())
        .map(|x| x as f32)
        .collect();
    // LibRaw lists red, green, blue, green 2.
    (v.len() >= 4).then(|| [v[0], v[1], v[3], v[2]])
}

/// Decodes `path` with the options LibRaw's behaviour needs and compares everything.
fn assert_matches_libraw(path: &Path, scratch: &Path) {
    let options = DecodeOptions {
        ycbcr: YcbcrOptions {
            chroma: ChromaUpsampling::Replicate,
            undo_white_balance: false,
        },
    };
    let ours = decode_with(path, &options).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (meta, theirs) = libraw(path, scratch);
    let name = path.file_name().unwrap_or_default().to_string_lossy();

    let channels = meta["channels"].as_u64().expect("channels") as usize;
    assert_eq!(channels, ours.channels(), "{name}: samples per pixel");
    assert_eq!(
        meta["width"].as_u64().unwrap() as usize,
        ours.pixel_width(),
        "{name}: width"
    );
    // ARW 1 files store 8 rows beyond `ImageLength`; LibRaw returns them, ym-raw drops them.
    let extra_rows = if ours.meta.decode == DecodeKind::Arw1 {
        ym_raw::decode::arw1::EXTRA_ROWS
    } else {
        0
    };
    assert_eq!(
        meta["height"].as_u64().unwrap() as usize,
        ours.pixel_height() + extra_rows,
        "{name}: height"
    );
    assert_eq!(
        theirs.len(),
        ours.data.data.len() + extra_rows * ours.data.width,
        "{name}: sample count"
    );
    let first = theirs.iter().zip(&ours.data.data).position(|(a, b)| a != b);
    assert!(
        first.is_none(),
        "{name}: first difference at sample {first:?} (ours {:?}, LibRaw {:?})",
        first.map(|i| ours.data.data[i]),
        first.map(|i| theirs[i])
    );

    // Levels. The black and white levels of camera-processed data are already ordered red, green,
    // green, blue; those of a mosaic follow the raster positions of the CFA pattern.
    let ordered = |v: [f32; 4]| if ours.is_reduced() { v } else { rggb(&ours, v) };
    let black = ordered(ours.black);
    let white = ordered(ours.white);
    // LibRaw does not rescale the 14-bit-scale levels of an SR2 block for 12-bit data (it reads
    // those files as 14-bit); ym-raw does (128 and 4075 instead of 512 and 16300), so the levels
    // of 12-bit files are not compared. The samples are.
    let twelve_bit = ours.meta.decode == DecodeKind::Uncompressed12;
    if !twelve_bit {
        if let Some(b) = libraw_levels(&meta, "black_level_per_channel") {
            assert_eq!(black, b, "{name}: black level");
        }
        if ours.is_reduced() {
            let maximum = meta["white_level"].as_f64().expect("maximum") as f32;
            assert_eq!(white, [maximum; 4], "{name}: white level (LibRaw maximum)");
        } else if let Some(w) = libraw_levels(&meta, "camera_white_level_per_channel") {
            assert_eq!(white, w, "{name}: white level");
        }
    }
    if let (Some(wb), Some(w)) = (ours.wb_as_shot, libraw_levels(&meta, "camera_whitebalance")) {
        let wb = rggb(&ours, wb);
        assert_eq!(
            [wb[0], wb[1], wb[3]],
            [w[0], w[1], w[3]],
            "{name}: white balance"
        );
    }
}

fn write(dir: &Path, name: &str, bytes: Vec<u8>) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write test file");
    path
}

#[test]
fn synthetic_mosaic_variants_match_libraw_sample_for_sample() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let encoder = CrawEncoder::new();
    let files = vec![
        write(
            dir.path(),
            "unc14.ARW",
            ArwBuilder::new("ILCE-7RM4", 160, 96).build(&gradient_mosaic(160, 96, 512, 16000, 1)),
        ),
        write(
            dir.path(),
            "unc12.ARW",
            ArwBuilder::new("ILCE-7M2", 128, 64)
                .uncompressed(12)
                .build(&gradient_mosaic(128, 64, 128, 4000, 2)),
        ),
        write(
            dir.path(),
            "packed12.ARW",
            ArwBuilder::new("DSLR-A900", 128, 64)
                .payload(Packed12)
                .build(&gradient_mosaic(128, 64, 128, 4000, 3)),
        ),
        write(
            dir.path(),
            "craw.ARW",
            ArwBuilder::new("ILCE-7M3", 192, 64)
                .craw(encoder.clone())
                .build(&gradient_mosaic(192, 64, 512, encoder.max_value(), 4)),
        ),
        write(
            dir.path(),
            "lossless_l.ARW",
            // 1100 x 520 makes 3 x 2 tiles with partial tiles at the right and bottom edges.
            // (LibRaw takes any file with exactly four tile offsets for a Sinar 4-shot file.)
            ArwBuilder::new("ILCE-1", 1100, 520)
                .payload(LosslessL::new())
                .build(&gradient_mosaic(1100, 520, 512, 16000, 5)),
        ),
    ];
    for file in files {
        assert_matches_libraw(&file, dir.path());
    }
}

#[test]
fn lossless_l_predictors_and_tables_match_libraw() {
    if !oracle_ready() {
        return;
    }
    // LibRaw's Sony loader handles every predictor and restart interval of one row; two-row
    // intervals with a two-dimensional predictor are not read the way T.81 says (libjpeg and
    // ym-raw agree with each other, see tools/oracle/README.md), so they are not compared here.
    let dir = tempfile::tempdir().expect("tempdir");
    let mosaic = gradient_mosaic(300, 260, 512, 16000, 6);
    for (predictor, restart_rows, huffman) in [
        (1u8, 0usize, HuffmanChoice::Optimal),
        (2, 0, HuffmanChoice::Optimal),
        (3, 0, HuffmanChoice::Skewed),
        (4, 0, HuffmanChoice::Optimal),
        (5, 0, HuffmanChoice::Optimal),
        (6, 0, HuffmanChoice::Skewed),
        (7, 0, HuffmanChoice::Optimal),
        (1, 1, HuffmanChoice::Optimal),
        (1, 2, HuffmanChoice::Skewed),
    ] {
        let file = ArwBuilder::new("ILCE-9M3", 300, 260)
            .payload(
                LosslessL::new()
                    .predictor(predictor)
                    .restart_rows(restart_rows)
                    .huffman(huffman)
                    .tile(128, 128),
            )
            .build(&mosaic);
        let path = write(
            dir.path(),
            &format!("l_p{predictor}_r{restart_rows}.ARW"),
            file,
        );
        assert_matches_libraw(&path, dir.path());
    }
}

#[test]
fn lossless_ms_matches_libraws_pseudo_raw_path() {
    if !oracle_ready() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for (sub, w, h) in [
        (ChromaSubsampling::Yuv420, 300u32, 260u32),
        (ChromaSubsampling::Yuv422, 300, 260),
        // LibRaw's loader reads 4:2:0 and 4:2:2 only (Sony's M and S sizes), not 4:4:4.
    ] {
        let rgb = gradient_rgb(w, h, 1100, 9000, 7);
        let payload = LosslessMs::new(sub, rgb, w, h, [2.0, 1.0, 1.6]).tile(128, 128);
        let file = ArwBuilder::new("ILCE-7RM5", w, h)
            .lossless_ms(payload)
            .build(&[]);
        let path = write(
            dir.path(),
            &format!("ms_{}.ARW", sub.name().replace(':', "")),
            file,
        );
        assert_matches_libraw(&path, dir.path());
    }
}

/// The real files this machine has, if any.
fn real_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    let mut found = Vec::new();
    for folder in ["real/files", "arw"] {
        let mut pending = vec![root.join(folder)];
        while let Some(dir) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    pending.push(path);
                } else if path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("arw"))
                {
                    found.push(path);
                }
            }
        }
    }
    found.sort();
    found
}

#[test]
#[ignore = "needs real ARW files in testdata/real/files or testdata/arw; see tools/oracle/README.md"]
fn real_files_match_libraw() {
    if !oracle_ready() {
        return;
    }
    let files = real_files();
    assert!(!files.is_empty(), "no real ARW files under testdata/");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut compared = 0;
    for file in files {
        // Layouts without a decoder (ARW 1 Huffman, ARW 6.0) are not compared.
        match ym_raw::metadata::probe(&file) {
            Ok(meta) if matches!(meta.decode, DecodeKind::Arw6 | DecodeKind::Unknown { .. }) => {
                eprintln!("skipping {}: {}", file.display(), meta.decode.describe());
                continue;
            }
            Err(e) => panic!("{}: {e}", file.display()),
            Ok(_) => {}
        }
        assert_matches_libraw(&file, dir.path());
        compared += 1;
    }
    assert!(compared > 0);
}
