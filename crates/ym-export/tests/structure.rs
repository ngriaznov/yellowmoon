//! Structure checks added in review: every directory starts on a word boundary whatever the length
//! of the text values before it, multi-strip files decode to the right pixels, and the bytes do not
//! depend on the number of threads for images that span several bands.

mod common;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::*;

/// Every IFD offset reachable from the header (IFD0, SubIFDs, Exif and GPS IFDs, the chain) is even.
fn assert_ifds_word_aligned(file: &[u8], what: &str) {
    let first = u32::from_le_bytes(file[4..8].try_into().unwrap());
    assert_eq!(first % 2, 0, "{what}: IFD0 at odd offset {first}");
    let mut seen = vec![first];
    let mut next = Some(first as usize);
    while let Some(off) = next.take() {
        let ifd = common::read_ifd(file, off);
        for tag in [330_u16, 34665, 34853] {
            if let Some(e) = ifd.entries.get(&tag) {
                for sub in e.u32s() {
                    assert_eq!(
                        sub % 2,
                        0,
                        "{what}: sub-directory (tag {tag}) at odd offset {sub}"
                    );
                    seen.push(sub);
                }
            }
        }
        if ifd.next != 0 {
            assert_eq!(ifd.next % 2, 0, "{what}: chained IFD at odd offset");
            next = Some(ifd.next as usize);
        }
    }
    assert!(!seen.is_empty());
}

fn odd_text_metadata() -> Metadata {
    let mut meta = common::metadata();
    // "FE 35mm F1.8" is 12 characters: 13 bytes with the NUL, an odd length, and the highest Exif tag.
    meta.exif.lens_model = Some("FE 35mm F1.8".into());
    meta.exif.model = Some("ILCE-7C".into());
    meta.xmp.app_version = "0.1.10".into();
    meta
}

#[test]
fn tiff_directories_start_on_word_boundaries_with_odd_length_text() {
    let meta = odd_text_metadata();
    let image = common::display_scene(40, 30);
    for compression in [
        TiffCompression::None,
        TiffCompression::Lzw,
        TiffCompression::Zip,
    ] {
        for bits in [BitDepth::Eight, BitDepth::Sixteen] {
            let req = ExportRequest {
                pixels: ImageRef::new(40, 30, &image, PixelEncoding::DisplayLinear).unwrap(),
                format: ExportFormat::Tiff { bits, compression },
                color_space: ColorSpace::Srgb,
                size: Size::Full,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
            let what = format!("tiff {bits:?} {compression:?}");
            assert_ifds_word_aligned(&bytes, &what);
            common::assert_values_word_aligned(&bytes);
        }
    }
}

#[test]
fn dng_directories_start_on_word_boundaries_with_odd_length_text() {
    let meta = odd_text_metadata();
    let (w, h) = (37, 21);
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let profile = DngProfile::identity();
    for orientation in [1_u16, 6] {
        let req = ExportRequest {
            pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: Some(DngInputs {
                preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
                orientation,
                crop: None,
                baseline_exposure: BaselineExposure::default(),
                profile: &profile,
            }),
        };
        let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        assert_ifds_word_aligned(&bytes, "dng");
        common::assert_values_word_aligned(&bytes);
    }
}

// ---- multi-strip and multi-band files ------------------------------------------------------------

use std::io::Cursor;
use ym_color::encode::OutputSpace;
use ym_export::pixels::{encode_rows_u8, encode_rows_u16};

fn linear_request<'a>(
    image: &'a [f32],
    w: u32,
    h: u32,
    format: ExportFormat,
    meta: &'a Metadata,
) -> ExportRequest<'a> {
    ExportRequest {
        pixels: ImageRef::new(w, h, image, PixelEncoding::DisplayLinear).unwrap(),
        format,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: meta,
        remove_location: false,
        dng: None,
    }
}

/// A smooth, non-repeating image whose every row differs, so a misplaced strip or band shows up.
fn ramp_image(w: u32, h: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 7 + y * 13) % 1000) as f32 / 1000.0);
            v.push(y as f32 / h as f32);
            v.push(x as f32 / w as f32);
        }
    }
    v
}

/// TIFFs with many strips, written in several parallel batches, decode to exactly the encoded rows.
#[test]
fn multi_strip_tiffs_decode_to_the_encoded_pixels() {
    use tiff::decoder::{Decoder, DecodingResult};
    let meta = common::metadata();
    for (bits, w, h) in [
        (BitDepth::Eight, 3000_u32, 2000_u32),
        (BitDepth::Sixteen, 3000, 1000),
    ] {
        let image = ramp_image(w, h);
        let img = ImageRef::new(w, h, &image, PixelEncoding::DisplayLinear).unwrap();
        let mut want8 = Vec::new();
        let mut want16 = Vec::new();
        match bits {
            BitDepth::Eight => encode_rows_u8(&img, OutputSpace::Srgb, 0, h as usize, &mut want8),
            BitDepth::Sixteen => {
                encode_rows_u16(&img, OutputSpace::Srgb, 0, h as usize, &mut want16)
            }
        }
        for compression in [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Zip,
        ] {
            let req = linear_request(
                &image,
                w,
                h,
                ExportFormat::Tiff { bits, compression },
                &meta,
            );
            let (file, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
            let ifd0 = common::ifd0(&file);
            let strips = ifd0.get(273).u32s().len();
            assert!(
                strips > 16,
                "{bits:?}: the test image should span more than one batch of strips ({strips} strips)"
            );
            assert_eq!(
                ifd0.get(279).u32s().len(),
                strips,
                "one byte count per strip"
            );
            assert_ifds_word_aligned(&file, "multi-strip tiff");
            let mut dec = Decoder::new(Cursor::new(&file)).unwrap();
            match (bits, dec.read_image().unwrap()) {
                (BitDepth::Eight, DecodingResult::U8(got)) => {
                    assert!(got == want8, "{compression:?} 8-bit pixels differ")
                }
                (BitDepth::Sixteen, DecodingResult::U16(got)) => {
                    assert!(got == want16, "{compression:?} 16-bit pixels differ")
                }
                _ => panic!("unexpected sample type"),
            }
        }
    }
}

/// PNG bands (one million pixels each) and JPEG bands decode to the right rows.
#[test]
fn multi_band_png_and_jpeg_decode_to_the_encoded_rows() {
    let meta = common::metadata();
    let (w, h) = (900_u32, 1500_u32); // 900 px wide: 512-row bands, so three bands
    let image = ramp_image(w, h);
    let img = ImageRef::new(w, h, &image, PixelEncoding::DisplayLinear).unwrap();
    let mut want8 = Vec::new();
    encode_rows_u8(&img, OutputSpace::Srgb, 0, h as usize, &mut want8);
    let mut want16 = Vec::new();
    encode_rows_u16(&img, OutputSpace::Srgb, 0, h as usize, &mut want16);

    for bits in [BitDepth::Eight, BitDepth::Sixteen] {
        let req = linear_request(&image, w, h, ExportFormat::Png { bits }, &meta);
        let (file, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        let decoder = png::Decoder::new(Cursor::new(file));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        assert_eq!((info.width, info.height), (w, h));
        match bits {
            BitDepth::Eight => assert!(buf == want8, "PNG 8-bit pixels differ"),
            BitDepth::Sixteen => {
                let got: Vec<u16> = buf
                    .chunks_exact(2)
                    .map(|b| u16::from_be_bytes([b[0], b[1]]))
                    .collect();
                assert!(got == want16, "PNG 16-bit pixels differ");
            }
        }
    }

    let req = linear_request(
        &image,
        w,
        h,
        ExportFormat::Jpeg {
            quality: JpegQuality::new(100).unwrap(),
        },
        &meta,
    );
    let (file, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    let mut dec = zune_jpeg::JpegDecoder::new(Cursor::new(file));
    let got = dec.decode().unwrap();
    assert_eq!(got.len(), want8.len());
    // Quality 100 keeps every row close to the 8-bit codes: no band is shifted or repeated.
    for y in 0..h as usize {
        let row = y * w as usize * 3..(y + 1) * w as usize * 3;
        let sad: u64 = got[row.clone()]
            .iter()
            .zip(&want8[row])
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        let mean = sad as f64 / (w as f64 * 3.0);
        assert!(mean < 6.0, "JPEG row {y}: mean absolute error {mean}");
    }
}

fn hashes_for_big_images() -> Vec<(String, String)> {
    let meta = common::metadata();
    let (w, h) = (1000_u32, 1100_u32); // three 512-row bands for the encoders that work in bands
    let image = ramp_image(w, h);
    let mut out = Vec::new();
    let formats = [
        ExportFormat::Jpeg {
            quality: JpegQuality::new(92).unwrap(),
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::new(75).unwrap(),
        },
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::Zip,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Eight,
            compression: TiffCompression::Lzw,
        },
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
    ];
    for format in formats {
        for size in [Size::Full, Size::LongEdge { px: 700 }] {
            let mut req = linear_request(&image, w, h, format, &meta);
            req.size = size;
            let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
            out.push((format!("{format:?} {size:?}"), common::sha256_hex(&bytes)));
        }
    }
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let profile = DngProfile::identity();
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
            baseline_exposure: BaselineExposure::default(),
            profile: &profile,
        }),
    };
    let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    out.push(("dng".to_owned(), common::sha256_hex(&bytes)));
    out
}

/// The golden test uses a 64 x 48 image, which is a single band; this one spans several bands, several
/// batches of strips and a real resize.
#[test]
fn big_images_give_the_same_bytes_on_one_and_many_threads() {
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(hashes_for_big_images)
    };
    let one = run(1);
    let many = run(6);
    assert_eq!(one, many);
}

// ---- DNG through write_export ------------------------------------------------------------------------

fn dng_request<'a>(
    raw: &'a [f32],
    preview: &'a [f32],
    w: u32,
    h: u32,
    meta: &'a Metadata,
    profile: &'a DngProfile,
) -> ExportRequest<'a> {
    ExportRequest {
        pixels: ImageRef::new(w, h, raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, preview, PixelEncoding::DisplayLinear).unwrap(),
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile,
        }),
    }
}

/// A DNG goes through the same atomic write as the other formats: its progress never goes back,
/// a cancel removes the partial file, and the finished file has the `.dng` name.
#[test]
fn dng_write_export_reports_progress_cancels_and_lands_under_its_name() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    use ym_export::naming::Placement;

    let tmp = tempfile::tempdir().unwrap();
    let (w, h) = (700_u32, 500_u32);
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let meta = common::metadata();
    let profile = DngProfile::identity();
    let req = dng_request(&raw, &preview, w, h, &meta, &profile);
    let dest = ExportDestination {
        dir: tmp.path().to_path_buf(),
        file_stem: "DSC04204_neutral".into(),
        if_exists: IfExists::KeepBoth,
        source_path: None,
    };

    // Progress.
    let seen = Mutex::new(Vec::<f32>::new());
    let progress = |f: f32| seen.lock().unwrap().push(f);
    let out = write_export(&req, &dest, &ExportHooks::new(None, Some(&progress))).unwrap();
    assert_eq!(out.path(), tmp.path().join("DSC04204_neutral.dng"));
    assert!(matches!(out.placement, Placement::Written(_)));
    assert_eq!((out.width, out.height), (w, h));
    assert!(out.dng.is_some());
    let v = seen.into_inner().unwrap();
    assert!(v.iter().all(|f| (0.0..=1.0).contains(f)), "{v:?}");
    assert!(
        v.windows(2).all(|p| p[0] <= p[1] + 1e-6),
        "progress went back: {v:?}"
    );
    assert!((v.last().unwrap() - 1.0).abs() < 1e-6, "{v:?}");

    // Cancel at several points of the work leaves no partial file and keeps the finished one.
    for stop_at in [1_u32, 2, 3, 5, 8] {
        let polls = AtomicU32::new(0);
        let cancel = || polls.fetch_add(1, Ordering::Relaxed) + 1 >= stop_at;
        let r = write_export(&req, &dest, &ExportHooks::new(Some(&cancel), None));
        assert!(
            matches!(r, Err(ExportError::Cancelled)),
            "stop at poll {stop_at}: {r:?}"
        );
        let names: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["DSC04204_neutral.dng".to_owned()],
            "stop at poll {stop_at}"
        );
    }
}

// ---- hostile pixel values ------------------------------------------------------------------------------

/// NaN, infinities, huge and negative samples never panic any writer; the files decode, and a NaN
/// becomes black.
#[test]
fn non_finite_and_extreme_samples_are_survivable_in_every_format() {
    let special = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        -0.0,
        f32::MIN_POSITIVE / 4.0, // subnormal
        0.5,
    ];
    let (w, h) = (33_u32, 17_u32);
    let mut rng = common::Rng(99);
    let image: Vec<f32> = (0..w * h * 3)
        .map(|_| special[(rng.next_u32() % special.len() as u32) as usize])
        .collect();
    let meta = common::metadata();
    let formats = [
        ExportFormat::Jpeg {
            quality: JpegQuality::new(92).unwrap(),
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::new(60).unwrap(),
        },
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::Lzw,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Eight,
            compression: TiffCompression::Zip,
        },
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
        ExportFormat::Png {
            bits: BitDepth::Eight,
        },
    ];
    for encoding in [PixelEncoding::DisplayLinear, PixelEncoding::DisplayEncoded] {
        for format in formats {
            for space in ColorSpace::ALL {
                for size in [Size::Full, Size::LongEdge { px: 20 }] {
                    let req = ExportRequest {
                        pixels: ImageRef::new(w, h, &image, encoding).unwrap(),
                        format,
                        color_space: space,
                        size,
                        metadata: &meta,
                        remove_location: false,
                        dng: None,
                    };
                    let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none())
                        .unwrap_or_else(|e| panic!("{format:?} {space:?} {size:?}: {e}"));
                    assert!(bytes.len() > 100);
                    assert_eq!((summary.width, summary.height), req.output_dimensions());
                }
            }
        }
    }
    // A DNG, with the hostile values in the raw pixels and in the preview.
    let profile = DngProfile::identity();
    let preview: Vec<f32> = image
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    for scene in [
        &image,
        &vec![f32::NAN; image.len()],
        &vec![f32::INFINITY; image.len()],
    ] {
        let req = dng_request(scene, &preview, w, h, &meta, &profile);
        let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
        assert!(bytes.len() > 1000);
        assert!(summary.dng.unwrap().baseline_exposure_ev.is_finite());
        assert_ifds_word_aligned(&bytes, "dng with hostile samples");
    }
}

/// A NaN is black in a display export, and +infinity is white.
#[test]
fn nan_is_black_and_infinity_is_white() {
    let meta = common::metadata();
    let mut image = vec![0.5_f32; 4 * 4 * 3];
    image[0..3].copy_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    let req = linear_request(
        &image,
        4,
        4,
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
        &meta,
    );
    let (file, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    let mut reader = png::Decoder::new(Cursor::new(file)).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut buf).unwrap();
    let first: Vec<u16> = buf[..6]
        .chunks_exact(2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(first, vec![0, 65535, 0]);
}

/// One-pixel-wide, one-pixel-tall and one-pixel images, and extreme aspect ratios, in every format.
#[test]
fn tiny_and_extreme_aspect_images_work_in_every_format() {
    use tiff::decoder::Decoder;
    let meta = common::metadata();
    let formats = [
        ExportFormat::Jpeg {
            quality: JpegQuality::new(92).unwrap(),
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::new(50).unwrap(),
        },
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::Lzw,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Eight,
            compression: TiffCompression::None,
        },
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
    ];
    for (w, h) in [
        (1_u32, 1_u32),
        (1, 7),
        (9, 1),
        (2, 2),
        (3, 5),
        (1, 300),
        (300, 1),
    ] {
        let image = ramp_image(w, h);
        for format in formats {
            for size in [Size::Full, Size::LongEdge { px: 20 }] {
                let mut req = linear_request(&image, w, h, format, &meta);
                req.size = size;
                let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none())
                    .unwrap_or_else(|e| panic!("{w}x{h} {format:?} {size:?}: {e}"));
                let (ow, oh) = size.output_dimensions(w, h);
                assert_eq!((summary.width, summary.height), (ow, oh));
                match format {
                    ExportFormat::Jpeg { .. } => {
                        let mut dec = zune_jpeg::JpegDecoder::new(Cursor::new(bytes));
                        let px = dec.decode().unwrap();
                        assert_eq!(px.len(), (ow * oh * 3) as usize, "{w}x{h} {size:?}");
                    }
                    ExportFormat::Tiff { .. } => {
                        let mut dec = Decoder::new(Cursor::new(bytes)).unwrap();
                        assert_eq!(dec.dimensions().unwrap(), (ow, oh));
                        dec.read_image().unwrap();
                    }
                    ExportFormat::Png { .. } => {
                        let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
                        assert_eq!((reader.info().width, reader.info().height), (ow, oh));
                        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
                        reader.next_frame(&mut buf).unwrap();
                    }
                    ExportFormat::Dng => unreachable!(),
                }
            }
        }
        // And a DNG of the same size.
        let raw = common::awg4_scene(w, h);
        let preview = common::display_scene(w, h);
        let profile = DngProfile::identity();
        let req = dng_request(&raw, &preview, w, h, &meta, &profile);
        let (bytes, _) = encode_to_vec(&req, &ExportHooks::none())
            .unwrap_or_else(|e| panic!("DNG {w}x{h}: {e}"));
        assert_ifds_word_aligned(&bytes, "tiny dng");
    }
}

/// A DNG with odd-length text values (which used to put directories at odd offsets) is accepted by
/// exiftool's validator and by Adobe's `dng_validate` without a warning.
#[test]
fn odd_length_text_does_not_trouble_the_independent_validators() {
    let meta = odd_text_metadata();
    let (w, h) = (37_u32, 21_u32);
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let profile = DngProfile::identity();
    let req = dng_request(&raw, &preview, w, h, &meta, &profile);
    let (bytes, _) = encode_to_vec(&req, &ExportHooks::none()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("odd.dng");
    std::fs::write(&path, &bytes).unwrap();

    if common::tool_exists("exiftool") {
        let out = std::process::Command::new("exiftool")
            .args(["-validate", "-warning", "-error", "-a", "-s3"])
            .arg(&path)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert_eq!(text.trim(), "OK", "exiftool -validate:\n{text}");
    } else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "exiftool is required"
        );
        eprintln!("SKIPPED: exiftool is not installed");
    }

    let binary = std::env::var_os("DNG_VALIDATE")
        .map(std::path::PathBuf::from)
        .into_iter()
        .chain([common::repo_root().join("testdata/dng_sdk/dng_validate")])
        .find(|p| p.is_file());
    if let Some(binary) = binary {
        let out = std::process::Command::new(binary)
            .arg("-v")
            .arg(&path)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "dng_validate failed:\n{text}");
        assert!(
            !text.contains("Warning") && !text.contains("Error"),
            "dng_validate complained:\n{text}"
        );
    } else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required"
        );
        eprintln!("SKIPPED: dng_validate not built");
    }
}
