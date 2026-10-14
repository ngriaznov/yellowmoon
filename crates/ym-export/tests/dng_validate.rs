//! The DNG writer against independent readers (`docs/09-export.md` section Gates, M3-12):
//! LibRaw (rawpy) returns our pixels, exiftool lists every tag with the expected value and no
//! warnings, tifffile and Pillow parse the structure. The checks live in `tools/dng/check_dng.py`
//! (they are the same ones CI runs); this file also checks the structure with our own TIFF walker.
//!
//! Needs `python3` with numpy, Pillow, tifffile and rawpy, and `exiftool`. Skipped with a message when
//! they are missing, unless `YM_REQUIRE_EXTERNAL_TOOLS` is set.

mod common;

use ym_export::dng::{BaselineExposure, CropRect, DngProfile, FittedProfile, LookTable};
use ym_export::*;

struct Built {
    bytes: Vec<u8>,
    raw: Vec<f32>,
    report: ym_export::dng::DngReport,
}

#[allow(clippy::too_many_arguments)]
fn build(
    w: u32,
    h: u32,
    preview_size: (u32, u32),
    orientation: u16,
    crop: Option<CropRect>,
    baseline: BaselineExposure,
    profile: &DngProfile,
    remove_location: bool,
) -> Built {
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(preview_size.0, preview_size.1);
    let meta = common::metadata();
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location,
        dng: Some(DngInputs {
            preview: ImageRef::new(
                preview_size.0,
                preview_size.1,
                &preview,
                PixelEncoding::DisplayLinear,
            )
            .unwrap(),
            orientation,
            crop,
            baseline_exposure: baseline,
            profile,
        }),
    };
    let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    Built {
        bytes,
        raw,
        report: summary.dng.expect("a DNG report"),
    }
}

fn plain(w: u32, h: u32) -> Built {
    build(
        w,
        h,
        (w, h),
        1,
        None,
        BaselineExposure::default(),
        &DngProfile::identity(),
        false,
    )
}

#[test]
fn structure_matches_the_documented_layout() {
    let (w, h) = (96, 64);
    let built = build(
        w,
        h,
        (80, 50),
        6,
        Some(CropRect {
            x: 4,
            y: 2,
            width: 88,
            height: 58,
        }),
        BaselineExposure::default(),
        &DngProfile::identity(),
        false,
    );
    let f = &built.bytes;
    common::assert_values_word_aligned(f);
    let ifd0 = common::ifd0(f);
    // IFD0: the thumbnail.
    assert_eq!(ifd0.one(254), 1, "NewSubFileType 1");
    assert_eq!(ifd0.get(258).u32s(), vec![8, 8, 8]);
    assert_eq!(ifd0.one(259), 1);
    assert_eq!(ifd0.one(262), 2);
    assert_eq!(ifd0.one(274), 6, "orientation");
    assert!(ifd0.one(256) <= 256 && ifd0.one(257) <= 256);
    // The DNG tags and their TIFF types.
    for (tag, ty, count) in [
        (50706_u16, 1_u16, 4_u32), // DNGVersion BYTE x4
        (50707, 1, 4),             // DNGBackwardVersion
        (50708, 2, 16),            // UniqueCameraModel ASCII "Yellowmoon AWG4\0"
        (50721, 10, 9),            // ColorMatrix1 SRATIONAL x9
        (50778, 3, 1),             // CalibrationIlluminant1 SHORT
        (50728, 5, 3),             // AsShotNeutral RATIONAL x3
        (50730, 10, 1),            // BaselineExposure SRATIONAL
        (50732, 5, 1),             // BaselineSharpness RATIONAL
        (50781, 1, 16),            // RawDataUniqueID BYTE x16
        (50879, 3, 1),             // ColorimetricReference SHORT
        (50936, 2, 11),            // ProfileName
        (50940, 11, 4),            // ProfileToneCurve FLOAT x4
        (50941, 4, 1),             // ProfileEmbedPolicy LONG
        (50942, 2, 24),            // ProfileCopyright
        (50964, 10, 9),            // ForwardMatrix1
        (50970, 4, 1),             // PreviewColorSpace LONG
        (50981, 4, 3),             // ProfileLookTableDims LONG x3
        (50982, 11, 24),           // ProfileLookTableData FLOAT x24
        (51108, 4, 1),             // ProfileLookTableEncoding
        (51110, 4, 1),             // DefaultBlackRender
    ] {
        let e = ifd0.get(tag);
        assert_eq!((e.ty, e.count), (ty, count), "tag {tag}");
    }
    assert_eq!(ifd0.get(50706).data, vec![1, 4, 0, 0]);
    assert_eq!(ifd0.get(50707).data, vec![1, 1, 0, 0]);
    assert_eq!(ifd0.get(50708).ascii(), "Yellowmoon AWG4");
    assert_eq!(ifd0.one(50778), 21);
    assert_eq!(ifd0.get(50728).f64s(), vec![1.0, 1.0, 1.0]);
    assert_eq!(ifd0.get(50732).f64s(), vec![1.0]);
    assert_eq!(ifd0.one(50941), 3);
    assert_eq!(ifd0.one(50879), 0);
    assert_eq!(ifd0.one(50970), 2, "PreviewColorSpace sRGB");
    assert_eq!(ifd0.one(51110), 1);
    assert_eq!(ifd0.one(51108), 1);
    assert_eq!(ifd0.get(50940).f64s(), vec![0.0, 0.0, 1.0, 1.0]);
    let be = ifd0.get(50730).f64s()[0];
    assert!((be - built.report.baseline_exposure_ev).abs() < 1e-6);
    assert!(
        (be - f64::from(built.report.x_white).log2()).abs() < 1e-6,
        "BaselineExposure = log2(x_white)"
    );

    // SubIFDs: raw, then the preview.
    let subs = ifd0.get(330).u32s();
    assert_eq!(subs.len(), 2);
    let raw = common::read_ifd(f, subs[0] as usize);
    assert_eq!(raw.one(254), 0);
    assert_eq!(raw.one(262), 34892, "LinearRaw");
    assert_eq!(raw.one(277), 3);
    assert_eq!(raw.get(258).u32s(), vec![16, 16, 16]);
    assert_eq!(raw.one(259), 1, "uncompressed strips");
    assert_eq!(raw.get(50714).u32s(), vec![0, 0, 0], "BlackLevel");
    assert_eq!(
        raw.get(50717).u32s(),
        vec![65535, 65535, 65535],
        "WhiteLevel"
    );
    assert_eq!(raw.get(50935).f64s(), vec![1.0], "NoiseReductionApplied");
    assert_eq!(raw.get(50719).u32s(), vec![4, 2], "DefaultCropOrigin");
    assert_eq!(raw.get(50720).u32s(), vec![88, 58], "DefaultCropSize");
    assert_eq!(raw.get(50829).u32s(), vec![0, 0, 64, 96], "ActiveArea");
    assert_eq!((raw.one(256), raw.one(257)), (96, 64));
    let strip_bytes: u32 = raw.get(279).u32s().iter().sum();
    assert_eq!(strip_bytes, 96 * 64 * 6);
    let preview = common::read_ifd(f, subs[1] as usize);
    assert_eq!(preview.one(254), 1);
    assert_eq!(preview.one(259), 7, "JPEG");
    assert_eq!((preview.one(256), preview.one(257)), (80, 50));
    let (off, len) = (preview.one(273) as usize, preview.one(279) as usize);
    assert_eq!(
        &f[off..off + 2],
        &[0xFF, 0xD8],
        "the preview is a JPEG stream"
    );
    assert_eq!(&f[off + len - 2..off + len], &[0xFF, 0xD9]);

    // Exif IFD and GPS IFD hang off IFD0; the XMP is there.
    assert!(ifd0.has(34665) && ifd0.has(34853) && ifd0.has(700));
    // The Exif pixel dimensions are those of the meaningful image: the default crop.
    let exif = common::read_ifd(f, ifd0.one(34665) as usize);
    assert_eq!((exif.one(40962), exif.one(40963)), (88, 58));
    let xmp = &ifd0.get(700).data;
    let text = std::str::from_utf8(xmp).unwrap();
    assert!(text.contains("<xmp:CreatorTool>Yellowmoon 0.0.1</xmp:CreatorTool>"));
    assert!(text.contains("ym:Look"));
}

#[test]
fn remove_location_leaves_no_gps_directory() {
    let kept = plain(32, 24);
    let removed = build(
        32,
        24,
        (32, 24),
        1,
        None,
        BaselineExposure::default(),
        &DngProfile::identity(),
        true,
    );
    assert!(common::ifd0(&kept.bytes).has(34853));
    let ifd0 = common::ifd0(&removed.bytes);
    assert!(!ifd0.has(34853));
    assert!(ifd0.has(34665));
}

#[test]
fn report_and_scaling_follow_the_spec() {
    let built = plain(128, 96);
    let r = built.report;
    // The scene has a highlight patch of 6.0 (+-1 % noise) and a few out-of-gamut samples (0.5 % of the pixels).
    assert!((5.9..6.1).contains(&f64::from(r.x_white)), "{}", r.x_white);
    assert!(
        (r.negative_clipped_fraction - 0.005).abs() < 0.001,
        "{}",
        r.negative_clipped_fraction
    );
    assert!(r.highlight_clipped_fraction <= 1e-4);
    assert!(r.max_value >= 5.9);
    // Decode the stored raw samples and compare with the formula.
    let f = &built.bytes;
    let ifd0 = common::ifd0(f);
    let raw_ifd = common::read_ifd(f, ifd0.get(330).u32s()[0] as usize);
    let offsets = raw_ifd.get(273).u32s();
    let sizes = raw_ifd.get(279).u32s();
    let mut samples = Vec::new();
    for (o, s) in offsets.iter().zip(&sizes) {
        for b in f[*o as usize..(*o + *s) as usize].chunks_exact(2) {
            samples.push(u16::from_le_bytes([b[0], b[1]]));
        }
    }
    assert_eq!(samples.len(), built.raw.len());
    let inv = 1.0 / f64::from(r.x_white);
    for (i, (&s, &x)) in samples.iter().zip(&built.raw).enumerate() {
        let want = (f64::from(x) * inv).clamp(0.0, 1.0) * 65535.0;
        assert!(
            (f64::from(s) - want).abs() <= 0.5 + 1e-9,
            "sample {i}: stored {s}, x {x}, want {want}"
        );
        if x < 0.0 {
            assert_eq!(s, 0);
        }
    }
}

#[test]
fn output_is_deterministic_and_baseline_exposure_can_be_fixed() {
    let a = plain(64, 48);
    let b = plain(64, 48);
    assert_eq!(a.bytes, b.bytes);
    let fixed = build(
        64,
        48,
        (64, 48),
        1,
        None,
        BaselineExposure::Fixed(1.25),
        &DngProfile::identity(),
        false,
    );
    let be = common::ifd0(&fixed.bytes).get(50730).f64s()[0];
    assert!((be - 1.25).abs() < 1e-6);
    let offset = build(
        64,
        48,
        (64, 48),
        1,
        None,
        BaselineExposure::Auto {
            fit_offset_ev: -0.5,
            placement_ev: 0.0,
        },
        &DngProfile::identity(),
        false,
    );
    let be = common::ifd0(&offset.bytes).get(50730).f64s()[0];
    assert!((be - (f64::from(offset.report.x_white).log2() - 0.5)).abs() < 1e-6);
}

#[test]
fn fitted_profile_inputs_are_written_as_given() {
    let profile = DngProfile {
        tone_curve: vec![[0.0, 0.0], [0.25, 0.4], [0.5, 0.7], [1.0, 1.0]],
        look_table: LookTable {
            dims: [2, 3, 2],
            // Entries are (hue shift, saturation scale, value scale); the zero-saturation entries (every
            // third entry here) keep the value scale at 1, as the DNG specification requires.
            data: (0..36)
                .map(|i| match (i % 3, (i / 3) % 3) {
                    (0, _) => 1.5,
                    (2, 0) => 1.0,
                    _ => 1.0 + i as f32 * 0.01,
                })
                .collect(),
            encoding: 1,
        },
    };
    let built = build(
        48,
        32,
        (48, 32),
        1,
        None,
        BaselineExposure::default(),
        &profile,
        false,
    );
    let ifd0 = common::ifd0(&built.bytes);
    assert_eq!(ifd0.get(50940).f64s().len(), 8);
    assert_eq!(ifd0.get(50981).u32s(), vec![2, 3, 2]);
    let data = ifd0.get(50982).f64s();
    assert_eq!(data.len(), 36);
    assert!((data[0] - 1.5).abs() < 1e-6 && (data[4] - 1.04).abs() < 1e-6);
}

#[test]
fn bad_inputs_are_refused() {
    let raw = common::awg4_scene(16, 16);
    let preview = common::display_scene(16, 16);
    let meta = common::metadata();
    let profile = DngProfile::identity();
    let mk = |orientation: u16, crop: Option<CropRect>, enc: PixelEncoding| {
        let req = ExportRequest {
            pixels: ImageRef::new(16, 16, &raw, enc).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(16, 16, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation,
                crop,
                baseline_exposure: BaselineExposure::default(),
                profile: &profile,
            }),
        };
        encode_to_vec(&req, &ExportHooks::none())
    };
    assert!(mk(1, None, PixelEncoding::SceneLinearAwg4).is_ok());
    assert!(mk(0, None, PixelEncoding::SceneLinearAwg4).is_err());
    assert!(mk(9, None, PixelEncoding::SceneLinearAwg4).is_err());
    assert!(
        mk(
            1,
            Some(CropRect {
                x: 10,
                y: 0,
                width: 10,
                height: 4
            }),
            PixelEncoding::SceneLinearAwg4
        )
        .is_err()
    );
    assert!(
        mk(
            1,
            Some(CropRect {
                x: 0,
                y: 0,
                width: 0,
                height: 4
            }),
            PixelEncoding::SceneLinearAwg4
        )
        .is_err()
    );
    assert!(
        mk(1, None, PixelEncoding::DisplayLinear).is_err(),
        "display pixels are not a DNG"
    );
    // A look table that re-tones greys breaks the DNG specification's zero-saturation rule.
    let mut retoning = DngProfile::identity();
    retoning.look_table.data[2] = 1.2; // value scale at zero saturation
    let req = ExportRequest {
        pixels: ImageRef::new(16, 16, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(16, 16, &preview, PixelEncoding::DisplayLinear).unwrap(),
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile: &retoning,
        }),
    };
    let err = encode_to_vec(&req, &ExportHooks::none()).unwrap_err();
    assert!(err.to_string().contains("zero saturation"), "{err}");
    // A DNG without its inputs, and a JPEG with scene-linear pixels.
    let req = ExportRequest {
        pixels: ImageRef::new(16, 16, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: None,
    };
    assert!(encode_to_vec(&req, &ExportHooks::none()).is_err());
    let jpeg = ExportRequest {
        format: ExportFormat::Jpeg {
            quality: JpegQuality::default(),
        },
        ..req
    };
    assert!(encode_to_vec(&jpeg, &ExportHooks::none()).is_err());
}

fn run_check(name: &str, built: &Built, w: u32, h: u32, extra: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let dng = dir.path().join(format!("{name}.dng"));
    std::fs::write(&dng, &built.bytes).unwrap();
    let raw = dir.path().join("raw.f32");
    let bytes: Vec<u8> = built.raw.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&raw, bytes).unwrap();
    let script = common::repo_root().join("tools/dng/check_dng.py");
    let (ws, hs) = (w.to_string(), h.to_string());
    let mut args = vec![
        dng.to_str().unwrap(),
        "--raw-f32",
        raw.to_str().unwrap(),
        "--width",
        &ws,
        "--height",
        &hs,
        "--expect-make",
        "SONY",
        "--expect-model",
        "ILCE-1",
    ];
    args.extend_from_slice(extra);
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&script)
        .args(&args)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "check_dng.py gave no JSON ({e}):\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let failed: Vec<&serde_json::Value> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["ok"] != true)
        .collect();
    assert!(
        report["ok"] == true && failed.is_empty(),
        "{name}: failed checks: {failed:#?}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!(
        "{name}: {} independent checks passed (exiftool, tifffile, Pillow, LibRaw)",
        report["checks"].as_array().unwrap().len()
    );
}

#[test]
fn libraw_exiftool_and_tifffile_agree_with_the_writer() {
    if !common::require_python_modules(&["numpy", "PIL", "tifffile", "rawpy"]) {
        return;
    }
    if !common::tool_exists("exiftool") {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "exiftool is required (Debian/Ubuntu: libimage-exiftool-perl)"
        );
        eprintln!("SKIPPED: exiftool is not installed");
        return;
    }
    let (w, h) = (192, 128);
    run_check("plain", &plain(w, h), w, h, &["--preview-size", "192x128"]);
    let cropped = build(
        w,
        h,
        (170, 110),
        8,
        Some(CropRect {
            x: 6,
            y: 4,
            width: 170,
            height: 110,
        }),
        BaselineExposure::Auto {
            fit_offset_ev: 0.25,
            placement_ev: 0.0,
        },
        &DngProfile {
            tone_curve: vec![[0.0, 0.0], [0.3, 0.45], [1.0, 1.0]],
            look_table: LookTable::identity(),
        },
        true,
    );
    run_check(
        "cropped",
        &cropped,
        w,
        h,
        &[
            "--preview-size",
            "170x110",
            "--orientation",
            "8",
            "--crop",
            "6,4,170,110",
            "--baseline-offset-ev",
            "0.25",
            "--tone-curve",
            "0 0 0.3 0.45 1 1",
        ],
    );
}

/// The DNG the product writes: the embedded fitted profile with the `BaselineExposure` rule that goes with it.
fn with_embedded_profile(w: u32, h: u32, scene: Option<Vec<f32>>) -> Built {
    let fitted = FittedProfile::embedded();
    let raw = scene.unwrap_or_else(|| common::awg4_scene(w, h));
    let preview = common::display_scene(w, h);
    let meta = common::metadata();
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
            orientation: 1,
            crop: None,
            baseline_exposure: fitted.baseline_exposure(),
            profile: &fitted.profile,
        }),
    };
    let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    Built {
        bytes,
        raw,
        report: summary.dng.expect("a DNG report"),
    }
}

/// A DNG with the embedded profile and a photo that stays below the white floor: `x_white` is 16 and
/// `BaselineExposure` is the display placement (`log2(16) - 4 + 0.85`), positive, so Adobe's renderer applies a plain
/// gain of `2^0.85` and no highlight shoulder, and the DNG is as bright as the app.
#[test]
fn a_dim_photo_with_the_embedded_profile_has_a_baseline_exposure_of_the_placement() {
    let built = with_embedded_profile(128, 96, None);
    assert_eq!(built.report.x_white, 16.0, "the floor of the fitted offset");
    assert_eq!(
        built.report.baseline_exposure_ev,
        ym_color::display::DISPLAY_PLACEMENT_EV
    );
    let ifd0 = common::ifd0(&built.bytes);
    let be = ifd0.get(50730).f64s();
    assert_eq!(be.len(), 1);
    assert!(
        (be[0] - ym_color::display::DISPLAY_PLACEMENT_EV).abs() < 1e-6,
        "BaselineExposure {be:?}"
    );
    // The profile that went in is the one the file carries.
    let fitted = FittedProfile::embedded();
    assert_eq!(
        ifd0.get(50981).u32s(),
        fitted.profile.look_table.dims.to_vec(),
        "ProfileLookTableDims"
    );
    assert_eq!(
        ifd0.get(50940).f64s().len(),
        fitted.profile.tone_curve.len() * 2,
        "ProfileToneCurve"
    );
    // A stored code is the scene value over 16: mid grey 0.18 is code 737.
    let raw_ifd = common::read_ifd(&built.bytes, ifd0.get(330).u32s()[0] as usize);
    let first_strip = raw_ifd.get(273).u32s()[0] as usize;
    let sample = |i: usize| {
        u16::from_le_bytes([
            built.bytes[first_strip + 2 * i],
            built.bytes[first_strip + 2 * i + 1],
        ])
    };
    for i in 0..30 {
        let want = (f64::from(built.raw[i]) / 16.0).clamp(0.0, 1.0) * 65535.0;
        assert!(
            (f64::from(sample(i)) - want).abs() <= 0.5 + 1e-9,
            "sample {i}"
        );
    }
}

/// Renders a DNG with `dng_validate` to 32-bit float sRGB.
fn render_srgb(binary: &std::path::Path, bytes: &[u8], name: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let dng = dir.path().join(format!("{name}.dng"));
    std::fs::write(&dng, bytes).unwrap();
    let base = dir.path().join("render");
    let out = std::process::Command::new(binary)
        .args(["-v", "-32", "-cs1", "-tif"])
        .arg(&base)
        .arg(&dng)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{name}: dng_validate failed:\n{text}");
    assert!(
        text.contains("Validation complete")
            && !text.contains("Warning")
            && !text.contains("Error"),
        "{name}: dng_validate complained:\n{text}"
    );
    let file = std::fs::File::open(base.with_extension("tif")).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
    let tiff::decoder::DecodingResult::F32(pixels) = decoder.read_image().unwrap() else {
        panic!("expected a float render");
    };
    pixels
}

/// The product's DNG, rendered by Adobe's reference renderer, shows what Yellowmoon shows (ACES 2 with the display
/// placement): flat patches of a photo below the white floor (`BaselineExposure` the placement, 0.85) and of a photo above
/// it (`BaselineExposure` more) land within a CIEDE2000 of 1.5 of the placed ACES 2 output transform, greys up to scene 8.9
/// (`16 / 2^0.85`, where the tone curve's input range ends after the placement) within 0.2. This is the end-to-end check of
/// the white floor, the embedded profile and `BaselineExposure::Auto`; the full fit report is `dngfit eval`.
#[test]
fn adobes_renderer_shows_the_product_dng_as_aces_2_does() {
    use ym_color::consts::linalg;
    use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
    use ym_color::encode::srgb_decode;

    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!("SKIPPED: dng_validate not built and $DNG_VALIDATE not set");
        return;
    };
    let greys = [0.01_f32, 0.02, 0.05, 0.18, 0.5, 1.0, 4.0];
    let colors: [[f32; 3]; 8] = [
        [0.4, 0.1, 0.05],
        [0.05, 0.4, 0.1],
        [0.1, 0.05, 0.4],
        [0.3, 0.3, 0.05],
        [0.05, 0.3, 0.3],
        [0.3, 0.05, 0.3],
        [0.6, 0.3, 0.15],
        [0.15, 0.3, 0.6],
    ];
    let lab = |code: [f64; 3]| {
        let linear = code.map(|c| srgb_decode(c.clamp(0.0, 1.0)));
        xyz_to_lab(
            linalg::mul_vec(&ym_color::gamut::matrices().rec709_to_xyz, linear),
            d65_white_xyz(),
        )
    };
    let aces2 = ym_color::aces2::Aces2Transform::shared(ym_color::aces2::Aces2Output::Srgb);
    let placement = ym_color::display::DISPLAY_PLACEMENT_EV;
    // Scene value at which Adobe's renderer reaches the end of the tone curve's input: `x * 2^placement / 16 = 1`.
    let clip_scene = 16.0 / placement.exp2();
    // (name, the brightest patch, the BaselineExposure that follows)
    for (name, brightest, bright_baseline) in [
        ("below the floor", 10.0_f32, false),
        ("above it", 40.0, true),
    ] {
        let mut patches: Vec<[f32; 3]> = greys.iter().map(|&g| [g; 3]).collect();
        patches.push([brightest; 3]);
        let n_grey = patches.len();
        patches.extend(colors);
        patches.push([brightest, brightest * 0.9, brightest * 0.8]);
        assert_eq!(patches.len(), 17);
        let (w, h) = (32_u32, 40_u32); // 4 x 5 patches of 8 x 8 pixels, the last three repeat the brightest
        let mut raw = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = ((y / 8) * 4 + x / 8) as usize;
                raw.extend_from_slice(&patches[i.min(patches.len() - 1)]);
            }
        }
        let built = with_embedded_profile(w, h, Some(raw));
        if bright_baseline {
            assert!(
                built.report.baseline_exposure_ev > 1.0 + placement,
                "{name}: {}",
                built.report.baseline_exposure_ev
            );
        } else {
            assert_eq!(built.report.baseline_exposure_ev, placement, "{name}");
        }
        let pixels = render_srgb(&binary, &built.bytes, "product");
        let (mut worst_grey, mut worst_color) = (0.0_f64, 0.0_f64);
        for (i, p) in patches.iter().enumerate() {
            let (px, py) = ((i % 4) * 8 + 4, (i / 4) * 8 + 4);
            let got = &pixels[(py * w as usize + px) * 3..][..3];
            // What the app shows: the placed ACES 2 path (the scene times 2^0.85 before the transform).
            let ap0 = linalg::mul_vec(&ym_color::display::aces2_input_matrix(), p.map(f64::from));
            let want = aces2.transform_ap0(ap0);
            let d = delta_e_2000(lab([got[0], got[1], got[2]].map(f64::from)), lab(want));
            // The profile's tone curve covers the plain transform's input up to 16 (`2^-fit_offset_ev`), which
            // is scene `16 / 2^0.85` with the placement: a patch above that is clipped to the display value of
            // scene 16 / 2^0.85 placed (the plain transform's code at 16 is 0.975, against ACES 2's 0.992 at
            // 40), a dE00 of about 0.9 at most.
            let in_range = p.iter().all(|&v| f64::from(v) <= clip_scene);
            if i < n_grey && in_range {
                worst_grey = worst_grey.max(d);
                assert!(
                    d < 0.2,
                    "{name}: grey {p:?}: Adobe {got:?}, ACES 2 {want:?}, dE00 {d}"
                );
            } else {
                worst_color = worst_color.max(d);
                assert!(
                    d < 1.5,
                    "{name}: patch {p:?}: Adobe {got:?}, ACES 2 {want:?}, dE00 {d}"
                );
            }
        }
        eprintln!(
            "product DNG, {name} (x_white {:.3}, BaselineExposure {:.3} EV): worst dE00 against ACES 2 {worst_grey:.3} for greys, {worst_color:.3} for colours",
            built.report.x_white, built.report.baseline_exposure_ev
        );
    }
}

#[test]
fn independent_readers_accept_a_dng_with_the_embedded_profile() {
    if !common::require_python_modules(&["numpy", "PIL", "tifffile", "rawpy"]) {
        return;
    }
    if !common::tool_exists("exiftool") {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "exiftool is required (Debian/Ubuntu: libimage-exiftool-perl)"
        );
        eprintln!("SKIPPED: exiftool is not installed");
        return;
    }
    let fitted = FittedProfile::embedded();
    let (w, h) = (192, 128);
    let built = with_embedded_profile(w, h, None);
    let curve: Vec<String> = fitted
        .profile
        .tone_curve
        .iter()
        .flatten()
        .map(|v| format!("{v}"))
        .collect();
    let dims: Vec<String> = fitted
        .profile
        .look_table
        .dims
        .iter()
        .map(ToString::to_string)
        .collect();
    run_check(
        "embedded-profile",
        &built,
        w,
        h,
        &[
            "--preview-size",
            "192x128",
            "--baseline-offset-ev",
            &fitted.fit_offset_ev.to_string(),
            "--white-floor",
            &fitted.baseline_exposure().white_floor().to_string(),
            "--tone-curve",
            &curve.join(" "),
            "--look-dims",
            &dims.join(" "),
        ],
    );
}

/// Adobe's `dng_validate` (built by `tools/dng/build_dng_validate.sh` into `testdata/dng_sdk/`, or named
/// by `$DNG_VALIDATE`).
fn dng_validate_binary() -> Option<std::path::PathBuf> {
    let from_env = std::env::var_os("DNG_VALIDATE").map(std::path::PathBuf::from);
    let default = common::repo_root().join("testdata/dng_sdk/dng_validate");
    from_env.into_iter().chain([default]).find(|p| p.is_file())
}

/// Every DNG variant passes Adobe's `dng_validate` with no warnings and no errors (the M3-12 gate "Every
/// exported DNG passes dng_validate"), and its reference renderer can render it.
#[test]
fn adobes_dng_validate_accepts_every_variant() {
    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!(
            "SKIPPED: dng_validate not built (tools/dng/build_dng_validate.sh) and $DNG_VALIDATE not set"
        );
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let tone = DngProfile {
        tone_curve: vec![[0.0, 0.0], [0.3, 0.45], [1.0, 1.0]],
        look_table: LookTable {
            dims: [4, 3, 2],
            data: (0..72)
                .map(|i| match i % 3 {
                    0 => 0.0,
                    _ => 1.0,
                })
                .collect(),
            encoding: 1,
        },
    };
    let variants: Vec<(&str, Built)> = vec![
        ("plain", plain(192, 128)),
        ("tiny", plain(5, 3)),
        ("odd-size", plain(131, 77)),
        (
            "rotated-cropped",
            build(
                192,
                128,
                (170, 110),
                6,
                Some(CropRect {
                    x: 6,
                    y: 4,
                    width: 170,
                    height: 110,
                }),
                BaselineExposure::Auto {
                    fit_offset_ev: -0.3,
                    placement_ev: 0.0,
                },
                &tone,
                true,
            ),
        ),
        (
            "mirrored",
            build(
                64,
                48,
                (64, 48),
                5,
                None,
                BaselineExposure::Fixed(-1.5),
                &DngProfile::identity(),
                false,
            ),
        ),
    ];
    for (name, built) in &variants {
        let path = dir.path().join(format!("{name}.dng"));
        std::fs::write(&path, &built.bytes).unwrap();
        let render = dir.path().join(format!("{name}.tif"));
        let out = std::process::Command::new(&binary)
            .args(["-v", "-tif"])
            .arg(&render)
            .arg(&path)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "{name}: dng_validate failed:\n{text}");
        assert!(text.contains("Validation complete"), "{name}:\n{text}");
        assert!(
            !text.contains("Warning") && !text.contains("Error"),
            "{name}: dng_validate complained:\n{text}"
        );
        assert!(
            render.is_file() && std::fs::metadata(&render).unwrap().len() > 100,
            "{name}: no render"
        );
    }
    eprintln!(
        "dng_validate accepted {} DNG variants without warnings",
        variants.len()
    );
}

/// The matrices and the neutral tags are right: Adobe's reference renderer, given a DNG of flat scene-linear
/// AWG4 patches and an identity profile, returns the colours that `ym-color` computes for sRGB and
/// Display P3 (to 1e-3 in encoded values). This checks `ColorMatrix1`, `ForwardMatrix1`,
/// `AsShotNeutral`, the D50 white and the Bradford adaptation against Adobe's code instead of ours.
#[test]
fn adobes_renderer_reproduces_the_expected_colors() {
    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!(
            "SKIPPED: dng_validate not built (tools/dng/build_dng_validate.sh) and $DNG_VALIDATE not set"
        );
        return;
    };
    let patches: [[f32; 3]; 16] = [
        [0.18, 0.18, 0.18],
        [0.5, 0.5, 0.5],
        [0.02, 0.02, 0.02],
        [0.9, 0.9, 0.9],
        [0.4, 0.1, 0.05],
        [0.05, 0.4, 0.1],
        [0.1, 0.05, 0.4],
        [0.3, 0.3, 0.05],
        [0.05, 0.3, 0.3],
        [0.3, 0.05, 0.3],
        [0.6, 0.3, 0.15],
        [0.15, 0.3, 0.6],
        [0.25, 0.2, 0.1],
        [0.1, 0.2, 0.25],
        [0.7, 0.5, 0.4],
        [0.4, 0.5, 0.7],
    ];
    let (w, h) = (32_u32, 32_u32);
    let mut raw = Vec::new();
    for y in 0..h {
        for x in 0..w {
            raw.extend_from_slice(&patches[((y / 8) * 4 + x / 8) as usize]);
        }
    }
    let preview = common::display_scene(w, h);
    let meta = Metadata::default();
    // Identity look tables of several shapes must all render as the identity: the default 2 x 2 x 2 with
    // sRGB encoding, a 2.5D table (one value division) with linear encoding, and a finer table.
    let identity_table = |dims: [u32; 3], encoding: u32| LookTable {
        dims,
        data: [0.0, 1.0, 1.0].repeat((dims[0] * dims[1] * dims[2]) as usize),
        encoding,
    };
    let profiles = [
        ("default", DngProfile::identity()),
        (
            "2.5D linear",
            DngProfile {
                look_table: identity_table([3, 2, 1], 0),
                ..DngProfile::identity()
            },
        ),
        (
            "fine sRGB",
            DngProfile {
                look_table: identity_table([6, 4, 5], 1),
                ..DngProfile::identity()
            },
        ),
    ];
    for (profile_name, profile) in &profiles {
        let req = ExportRequest {
            pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation: 1,
                crop: None,
                baseline_exposure: BaselineExposure::Fixed(0.0),
                profile,
            }),
        };
        let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        assert_eq!(
            summary.dng.unwrap().x_white,
            1.0,
            "the patches stay below 1, so the scale is 1"
        );
        let dir = tempfile::tempdir().unwrap();
        let dng = dir.path().join("patches.dng");
        std::fs::write(&dng, bytes).unwrap();

        let m = ym_color::gamut::matrices();
        for (flag, name, matrix) in [
            ("-cs1", "srgb", m.awg4_to_rec709),
            ("-csP3", "p3", m.awg4_to_p3),
        ] {
            let base = dir.path().join(format!("render_{name}"));
            let out = std::process::Command::new(&binary)
                .args(["-32", flag, "-tif"])
                .arg(&base)
                .arg(&dng)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let file = std::fs::File::open(base.with_extension("tif")).unwrap();
            let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
            let tiff::decoder::DecodingResult::F32(pixels) = decoder.read_image().unwrap() else {
                panic!("expected a float render");
            };
            let mut worst = 0.0_f64;
            for (i, p) in patches.iter().enumerate() {
                let (px, py) = ((i % 4) * 8 + 4, (i / 4) * 8 + 4);
                let got = &pixels[(py * w as usize + px) * 3..][..3];
                let linear = ym_color::consts::linalg::mul_vec(&matrix, p.map(f64::from));
                for c in 0..3 {
                    let want = ym_color::encode::srgb_encode(linear[c].max(0.0)).clamp(0.0, 1.0);
                    let d = (f64::from(got[c]) - want).abs();
                    worst = worst.max(d);
                    assert!(
                        d < 1e-3,
                        "{name} patch {i} {p:?} channel {c}: Adobe {} ours {want}",
                        got[c]
                    );
                }
            }
            eprintln!(
                "Adobe's renderer vs ym-color, {name} ({profile_name}): worst difference {worst:.1e} in encoded values over 16 patches"
            );
        }
    }
}

/// `BaselineExposure` undoes the `1 / x_white` scaling in Adobe's renderer too: a scene with values well
/// above 1 (so `x_white` is about 3 and the stored values are a third of the scene values) renders to the
/// colours `ym-color` computes from the original scene-linear values, with the default `BaselineExposure`
/// (`log2(x_white)`) and with a fitted offset added on top (every stop of offset doubles the light).
#[test]
fn adobes_renderer_undoes_the_white_level_scaling_with_the_baseline_exposure() {
    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!("SKIPPED: dng_validate not built and $DNG_VALIDATE not set");
        return;
    };
    // Scene-linear AWG4 values between 0.02 and 2.6; the brightest sRGB channel stays below 1 after the
    // (offset) exposure, so no highlight clipping is involved.
    let patches: [[f32; 3]; 8] = [
        [0.18, 0.18, 0.18],
        [0.02, 0.02, 0.02],
        [1.0, 1.0, 1.0],
        [2.6, 2.4, 2.2],
        [0.5, 0.2, 0.1],
        [0.1, 0.5, 0.2],
        [0.2, 0.1, 0.5],
        [0.9, 0.7, 0.3],
    ];
    let (w, h) = (32_u32, 16_u32);
    let mut raw = Vec::new();
    for y in 0..h {
        for x in 0..w {
            raw.extend_from_slice(&patches[((y / 8) * 4 + x / 8) as usize]);
        }
    }
    let preview = common::display_scene(w, h);
    let meta = Metadata::default();
    let profile = DngProfile::identity();
    let m = ym_color::gamut::matrices();
    for offset_ev in [0.0_f64, -1.0] {
        let req = ExportRequest {
            pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation: 1,
                crop: None,
                baseline_exposure: BaselineExposure::Auto {
                    fit_offset_ev: offset_ev,
                    placement_ev: 0.0,
                },
                profile: &profile,
            }),
        };
        let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        let report = summary.dng.unwrap();
        assert!(
            (2.6..2.7).contains(&report.x_white),
            "x_white follows the brightest patches: {}",
            report.x_white
        );
        let dir = tempfile::tempdir().unwrap();
        let dng = dir.path().join("bright.dng");
        std::fs::write(&dng, bytes).unwrap();
        let base = dir.path().join("render");
        let out = std::process::Command::new(&binary)
            .args(["-32", "-cs1", "-tif"])
            .arg(&base)
            .arg(&dng)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let file = std::fs::File::open(base.with_extension("tif")).unwrap();
        let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
        let tiff::decoder::DecodingResult::F32(pixels) = decoder.read_image().unwrap() else {
            panic!("expected a float render");
        };
        let gain = 2.0_f64.powf(offset_ev);
        let mut worst = 0.0_f64;
        for (i, p) in patches.iter().enumerate() {
            let (px, py) = ((i % 4) * 8 + 4, (i / 4) * 8 + 4);
            let got = &pixels[(py * w as usize + px) * 3..][..3];
            let linear = ym_color::consts::linalg::mul_vec(&m.awg4_to_rec709, p.map(f64::from));
            for c in 0..3 {
                let want =
                    ym_color::encode::srgb_encode((linear[c] * gain).max(0.0)).clamp(0.0, 1.0);
                let d = (f64::from(got[c]) - want).abs();
                worst = worst.max(d);
                assert!(
                    d < 2e-3,
                    "offset {offset_ev} EV, patch {i} {p:?} channel {c}: Adobe {} ours {want}",
                    got[c]
                );
            }
        }
        eprintln!(
            "Adobe's renderer, x_white {:.3}, BaselineExposure {:.3} EV (offset {offset_ev}): worst difference {worst:.1e}",
            report.x_white, report.baseline_exposure_ev
        );
    }
}
