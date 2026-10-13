//! Golden exports (`docs/09-export.md` section Tests): a fixed synthetic scene through every format and
//! every colour space, hashed with SHA-256. A change in a hash needs an explanation in the PR
//! (and usually a `pipeline_version` bump). The same bytes must come out whatever the number of threads.
//!
//! Update the table after an intended change with `YM_UPDATE_GOLDEN=1 cargo test -p ym-export --test golden`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::*;

const W: u32 = 64;
const H: u32 = 48;

fn table_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden_hashes.json")
}

/// Every golden case: a name and the encoded bytes.
fn cases() -> Vec<(String, Vec<u8>)> {
    let meta = common::metadata();
    let display = common::display_scene(W, H);
    let mut out = Vec::new();
    let formats: Vec<(&str, ExportFormat, Size)> = vec![
        (
            "jpeg-q92",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(92).unwrap(),
            },
            Size::Full,
        ),
        (
            "jpeg-q75",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(75).unwrap(),
            },
            Size::Full,
        ),
        (
            "tiff8-none",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::None,
            },
            Size::Full,
        ),
        (
            "tiff8-lzw",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Lzw,
            },
            Size::Full,
        ),
        (
            "tiff8-zip",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Zip,
            },
            Size::Full,
        ),
        (
            "tiff16-none",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
            Size::Full,
        ),
        (
            "tiff16-lzw",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Lzw,
            },
            Size::Full,
        ),
        (
            "tiff16-zip",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Zip,
            },
            Size::Full,
        ),
        (
            "png8",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            Size::Full,
        ),
        (
            "png16",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            Size::Full,
        ),
        (
            "jpeg-q92-long40",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(92).unwrap(),
            },
            Size::LongEdge { px: 40 },
        ),
        (
            "png16-long33",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            Size::LongEdge { px: 33 },
        ),
    ];
    for (name, format, size) in formats {
        for space in ColorSpace::ALL {
            let req = ExportRequest {
                pixels: ImageRef::new(W, H, &display, PixelEncoding::DisplayLinear).unwrap(),
                format,
                color_space: space,
                size,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
            let key = match space {
                ColorSpace::Srgb => "srgb",
                ColorSpace::DisplayP3 => "display-p3",
                ColorSpace::Rec2020 => "rec2020",
            };
            out.push((format!("{name}/{key}"), bytes));
        }
    }
    // Encoded input is a different path through the same writers.
    let encoded: Vec<f32> = display
        .iter()
        .map(|&v| ym_color::encode::srgb_encode(f64::from(v)) as f32)
        .collect();
    let req = ExportRequest {
        pixels: ImageRef::new(W, H, &encoded, PixelEncoding::DisplayEncoded).unwrap(),
        format: ExportFormat::Png {
            bits: BitDepth::Eight,
        },
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: true,
        dng: None,
    };
    out.push((
        "png8-encoded-input-no-gps/srgb".to_owned(),
        encode_to_vec(&req, &ExportHooks::none()).unwrap().0,
    ));

    // DNG: one per orientation/crop variant.
    let raw = common::awg4_scene(W, H);
    let profile = DngProfile::identity();
    for (name, orientation) in [("dng", 1_u16), ("dng-rot90", 6)] {
        let req = ExportRequest {
            pixels: ImageRef::new(W, H, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(W, H, &display, PixelEncoding::DisplayLinear).unwrap(),
                orientation,
                crop: None,
                baseline_exposure: BaselineExposure::default(),
                profile: &profile,
            }),
        };
        out.push((
            name.to_owned(),
            encode_to_vec(&req, &ExportHooks::none()).unwrap().0,
        ));
    }
    out
}

fn hashes() -> BTreeMap<String, String> {
    cases()
        .into_iter()
        .map(|(n, b)| (n, common::sha256_hex(&b)))
        .collect()
}

#[test]
fn golden_hashes_match() {
    let got = hashes();
    if std::env::var_os("YM_UPDATE_GOLDEN").is_some() {
        let mut json = serde_json::to_string_pretty(&got).unwrap();
        json.push('\n');
        std::fs::write(table_path(), json).unwrap();
        eprintln!(
            "wrote {} golden hashes to {}",
            got.len(),
            table_path().display()
        );
        return;
    }
    let want: BTreeMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(table_path())
            .expect("tests/data/golden_hashes.json (run with YM_UPDATE_GOLDEN=1 to create it)"),
    )
    .unwrap();
    let mut diffs = Vec::new();
    for (name, hash) in &got {
        match want.get(name) {
            Some(w) if w == hash => {}
            Some(w) => diffs.push(format!("{name}: expected {w}, got {hash}")),
            None => diffs.push(format!("{name}: not in the table")),
        }
    }
    for name in want.keys().filter(|n| !got.contains_key(*n)) {
        diffs.push(format!("{name}: in the table but no longer produced"));
    }
    assert!(
        diffs.is_empty(),
        "golden export hashes changed (explain it in the PR, bump pipeline_version if pixels changed; \
         update with YM_UPDATE_GOLDEN=1):\n{}",
        diffs.join("\n")
    );
}

#[test]
fn the_bytes_do_not_depend_on_the_number_of_threads() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(hashes);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap()
        .install(hashes);
    assert_eq!(one, many);
}

#[test]
fn every_color_space_and_format_gives_different_bytes() {
    let h = hashes();
    let values: std::collections::BTreeSet<&String> = h.values().collect();
    assert_eq!(
        values.len(),
        h.len(),
        "two golden cases produced identical files"
    );
}
