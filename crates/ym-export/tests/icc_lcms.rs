//! Every export decodes in a colour-managed reader to the expected values (`docs/09-export.md`
//! section Tests, ICC): the embedded profile of JPEG, PNG and TIFF files is read by LittleCMS
//! (through Pillow's ImageCms for 8-bit sRGB, and through the C library for 16-bit input and for
//! XYZ in doubles) and compared with `ym-color`'s own maths within one code value.
//!
//! Needs `python3` with numpy, Pillow and tifffile, and liblcms2. Skipped with a message when they
//! are missing, unless `YM_REQUIRE_EXTERNAL_TOOLS` is set.

mod common;

use std::collections::BTreeMap;

use serde::Deserialize;
use ym_color::consts::{Mat3, Vec3, linalg};
use ym_color::encode::{OutputSpace, srgb_encode};
use ym_color::gamut;
use ym_export::icc;
use ym_export::*;

const COLS: u32 = 17;
const ROWS: u32 = 16;
const CELL: u32 = 8;

/// The 272 patch codes: a 6x6x6 cube, a grey ramp in steps of 5, and four near-blacks.
fn patches() -> Vec<[u8; 3]> {
    let mut v = Vec::new();
    let levels = [0_u8, 51, 102, 153, 204, 255];
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                v.push([r, g, b]);
            }
        }
    }
    for g in (0..=255).step_by(5) {
        v.push([g as u8; 3]);
    }
    for g in 1..=4 {
        v.push([g; 3]);
    }
    assert_eq!(v.len(), (COLS * ROWS) as usize);
    v
}

/// The patch grid as encoded display values.
fn patch_image() -> Vec<f32> {
    let patches = patches();
    let (w, h) = (COLS * CELL, ROWS * CELL);
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = patches[((y / CELL) * COLS + x / CELL) as usize];
            v.extend(p.map(|c| f32::from(c) / 255.0));
        }
    }
    v
}

#[derive(Deserialize)]
struct Entry {
    bits: u32,
    codes: Vec<[u32; 3]>,
    xyz: Option<Vec<[f64; 3]>>,
    srgb_lcms: Option<Vec<[i32; 3]>>,
    srgb_lcms_default: Option<Vec<[i32; 3]>>,
    srgb_pillow: Option<Vec<[i32; 3]>>,
    srgb_pillow_default: Option<Vec<[i32; 3]>>,
    icc_sha256: String,
}

fn space_matrix(cs: ColorSpace) -> Mat3 {
    let m = gamut::matrices();
    match cs {
        ColorSpace::Srgb => m.rec709_to_xyz,
        ColorSpace::DisplayP3 => m.p3_to_xyz,
        ColorSpace::Rec2020 => m.rec2020_to_xyz,
    }
}

fn expected_xyz_d50(cs: ColorSpace, linear: Vec3) -> Vec3 {
    let d65 = linalg::mul_vec(&space_matrix(cs), linear);
    linalg::mul_vec(&icc::bradford_d65_to_d50(), d65)
}

fn expected_srgb_code(cs: ColorSpace, linear: Vec3) -> [i32; 3] {
    let d65 = linalg::mul_vec(&space_matrix(cs), linear);
    let rec709 = linalg::mul_vec(&gamut::matrices().xyz_to_rec709, d65);
    rec709.map(|c| (srgb_encode(c).clamp(0.0, 1.0) * 255.0).round() as i32)
}

struct Case {
    name: String,
    cs: ColorSpace,
    bytes: Vec<u8>,
}

fn cases() -> Vec<Case> {
    let image = patch_image();
    let meta = common::metadata();
    let (w, h) = (COLS * CELL, ROWS * CELL);
    let formats: [(&str, &str, ExportFormat); 5] = [
        (
            "jpeg100",
            "jpg",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(100).unwrap(),
            },
        ),
        (
            "png8",
            "png",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
        ),
        (
            "png16",
            "png",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
        ),
        (
            "tiff8lzw",
            "tif",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Lzw,
            },
        ),
        (
            "tiff16zip",
            "tif",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Zip,
            },
        ),
    ];
    let mut out = Vec::new();
    for cs in ColorSpace::ALL {
        for (name, ext, format) in formats {
            let req = ExportRequest {
                pixels: ImageRef::new(w, h, &image, PixelEncoding::DisplayEncoded).unwrap(),
                format,
                color_space: cs,
                size: Size::Full,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
            out.push(Case {
                name: format!("{name}_{cs:?}.{ext}"),
                cs,
                bytes,
            });
        }
    }
    out
}

#[test]
fn embedded_profiles_convert_to_the_expected_values_in_littlecms() {
    if !common::require_python_modules(&["numpy", "PIL", "tifffile"]) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cases = cases();
    let mut paths = Vec::new();
    for c in &cases {
        let p = dir.path().join(&c.name);
        std::fs::write(&p, &c.bytes).unwrap();
        paths.push(p);
    }
    let cols = COLS.to_string();
    let rows = ROWS.to_string();
    let cell = CELL.to_string();
    let mut args = vec![cols.as_str(), rows.as_str(), cell.as_str()];
    let path_strings: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
    args.extend(path_strings.iter().map(String::as_str));
    let json = common::run_python(&common::tests_py("icc_lcms.py"), &args);
    let results: BTreeMap<String, Entry> = serde_json::from_str(&json).unwrap();

    let mut worst_xyz = 0.0_f64;
    let mut worst_code_lcms = 0;
    let mut worst_code_pillow = 0;
    let mut worst_default = 0;
    let mut checked_with_c_library = 0;
    for (case, path) in cases.iter().zip(&path_strings) {
        let e = &results[path];
        let max = f64::from((1_u32 << e.bits) - 1);
        assert_eq!(
            e.icc_sha256,
            common::sha256_hex(icc::profile(case.cs)),
            "{}: the embedded profile",
            case.name
        );
        if e.xyz.is_none() {
            // Without the C library only Pillow's 8-bit path can be checked.
            assert!(
                std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
                "the LittleCMS C library (liblcms2) is required"
            );
        } else {
            checked_with_c_library += 1;
        }
        let space: OutputSpace = case.cs.output_space();
        for (i, code) in e.codes.iter().enumerate() {
            let linear = code.map(|c| space.decode_channel(f64::from(c) / max));
            let want = expected_srgb_code(case.cs, linear);
            if let Some(lcms_xyz) = &e.xyz {
                let xyz = expected_xyz_d50(case.cs, linear);
                for (k, ours) in xyz.iter().enumerate() {
                    let d = (lcms_xyz[i][k] - ours).abs();
                    worst_xyz = worst_xyz.max(d);
                    assert!(
                        d < 3e-4,
                        "{} patch {i} {code:?}: XYZ[{k}] lcms {} ours {ours}",
                        case.name,
                        lcms_xyz[i][k],
                    );
                }
            }
            for k in 0..3 {
                if let Some(lcms) = &e.srgb_lcms {
                    let d = (lcms[i][k] - want[k]).abs();
                    worst_code_lcms = worst_code_lcms.max(d);
                    assert!(
                        d <= 1,
                        "{} patch {i} {code:?}: sRGB[{k}] lcms {} ours {}",
                        case.name,
                        lcms[i][k],
                        want[k]
                    );
                }
                if let Some(d) = &e.srgb_lcms_default {
                    worst_default = worst_default.max((d[i][k] - want[k]).abs());
                }
                if let Some(p) = &e.srgb_pillow_default {
                    worst_default = worst_default.max((p[i][k] - want[k]).abs());
                }
                if let Some(p) = &e.srgb_pillow {
                    let d = (p[i][k] - want[k]).abs();
                    worst_code_pillow = worst_code_pillow.max(d);
                    assert!(
                        d <= 1,
                        "{} patch {i} {code:?}: sRGB[{k}] Pillow {} ours {}",
                        case.name,
                        p[i][k],
                        want[k]
                    );
                }
            }
        }
    }
    eprintln!(
        "ICC vs LittleCMS over {} files x {} patches ({checked_with_c_library} also through the C library): worst XYZ difference {worst_xyz:.2e}, worst sRGB code difference {worst_code_lcms} (C library) / {worst_code_pillow} (Pillow); with LittleCMS's default optimisations {worst_default}",
        cases.len(),
        COLS * ROWS
    );
}
