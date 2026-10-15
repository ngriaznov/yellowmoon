//! The writers pull their pixels through [`BandSource`] (`docs/gpu-plan.md` G-08): the bytes of a
//! file written from the rows of an [`ImageBands`] are those of the file written from the image,
//! a source that fails ends the export without a file, and the rows of a picture are only handed
//! out in order.

mod common;

use std::io::Cursor;
use std::sync::Mutex;

use ym_color::encode::OutputSpace;
use ym_export::naming::Placement;
use ym_export::pixels::{BandSource, ImageBands};
use ym_export::request::{
    BandRequest, ExportDestination, encode_bands_to_writer, write_export_bands,
};
use ym_export::*;

fn formats() -> Vec<ExportFormat> {
    let mut v = vec![
        ExportFormat::Jpeg {
            quality: JpegQuality::new(92).unwrap(),
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::new(80).unwrap(),
        },
    ];
    for bits in [BitDepth::Eight, BitDepth::Sixteen] {
        for compression in [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Zip,
        ] {
            v.push(ExportFormat::Tiff { bits, compression });
        }
        v.push(ExportFormat::Png { bits });
    }
    v
}

#[test]
fn bands_through_every_writer_are_the_bytes_of_the_image_path() {
    // 61 x 47: not a multiple of any band, strip or group size the writers use.
    let (w, h) = (61_u32, 47_u32);
    let data = common::display_scene(w, h);
    let meta = common::metadata();
    for space in [ColorSpace::Srgb, ColorSpace::DisplayP3, ColorSpace::Rec2020] {
        for format in formats() {
            let pixels = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
            let request = ExportRequest {
                pixels,
                format,
                color_space: space,
                size: Size::Full,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let (expected, summary) = encode_to_vec(&request, &ExportHooks::none()).unwrap();
            let mut bands = ImageBands::new(pixels, space.output_space());
            let mut out = Cursor::new(Vec::new());
            let got = encode_bands_to_writer(
                &BandRequest {
                    format,
                    color_space: space,
                    metadata: &meta,
                    remove_location: false,
                },
                &mut bands,
                &mut out,
                &ExportHooks::none(),
            )
            .unwrap();
            assert_eq!((got.width, got.height), (summary.width, summary.height));
            assert!(
                out.into_inner() == expected,
                "{format:?} in {space:?}: the bytes differ"
            );
        }
    }
}

#[test]
fn rows_are_checked_against_the_picture() {
    let data = common::display_scene(8, 6);
    let pixels = ImageRef::new(8, 6, &data, PixelEncoding::DisplayLinear).unwrap();
    let mut bands = ImageBands::new(pixels, OutputSpace::Srgb);
    assert_eq!((bands.width(), bands.height()), (8, 6));
    let mut out8 = Vec::new();
    let mut out16 = Vec::new();
    bands.rows_u8(0, 6, &mut out8).unwrap();
    assert_eq!(out8.len(), 8 * 6 * 3);
    bands.rows_u16(2, 3, &mut out16).unwrap();
    assert_eq!(out16.len(), 8 * 3 * 3);
    assert!(bands.rows_u8(4, 3, &mut out8).is_err(), "past the bottom");
    assert!(bands.rows_u16(7, 1, &mut out16).is_err(), "past the bottom");
    assert!(
        bands.rows_u8(usize::MAX, 2, &mut out8).is_err(),
        "no overflow"
    );
    // Rows of any size in any order are the same pixels as the whole.
    let mut whole = Vec::new();
    bands.rows_u8(0, 6, &mut whole).unwrap();
    let mut pieces = Vec::new();
    for (y0, n) in [(3, 3), (0, 1), (1, 2)] {
        bands.rows_u8(y0, n, &mut out8).unwrap();
        pieces.push((y0, out8.clone()));
    }
    pieces.sort_by_key(|p| p.0);
    let joined: Vec<u8> = pieces.into_iter().flat_map(|p| p.1).collect();
    assert_eq!(joined, whole);
}

/// A source that serves an image's rows in order and then fails at the `fail_at`th request, and
/// remembers the rows it was asked for.
struct Failing<'a> {
    inner: ImageBands<'a>,
    calls: usize,
    fail_at: usize,
    log: Mutex<Vec<(usize, usize)>>,
}

impl BandSource for Failing<'_> {
    fn width(&self) -> u32 {
        self.inner.width()
    }

    fn height(&self) -> u32 {
        self.inner.height()
    }

    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError> {
        self.calls += 1;
        self.log.lock().unwrap().push((y0, rows));
        if self.calls == self.fail_at {
            return Err(ExportError::Encode {
                format: "JPEG",
                reason: "the source failed".into(),
            });
        }
        self.inner.rows_u8(y0, rows, out)
    }

    fn rows_u16(&mut self, y0: usize, rows: usize, out: &mut Vec<u16>) -> Result<(), ExportError> {
        self.calls += 1;
        self.log.lock().unwrap().push((y0, rows));
        if self.calls == self.fail_at {
            return Err(ExportError::Cancelled);
        }
        self.inner.rows_u16(y0, rows, out)
    }
}

#[test]
fn a_source_that_fails_leaves_no_file_and_its_error_comes_through() {
    // Tall enough for several bands (the writers pull about a million pixels at a time).
    let (w, h) = (700_u32, 2000_u32);
    let data = vec![0.25_f32; (w * h * 3) as usize];
    let meta = common::metadata();
    let pixels = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
    // (format, whether the source says "cancelled", the request that fails): the JPEG and PNG
    // writers pull 1,498 rows at a time here, the uncompressed 8-bit TIFF writer all 2,000 at once.
    for (format, expect_cancel, fail_at) in [
        (
            ExportFormat::Jpeg {
                quality: JpegQuality::default(),
            },
            false,
            2,
        ),
        (
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            true,
            2,
        ),
        (
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::None,
            },
            false,
            1,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let dest = ExportDestination {
            dir: dir.path().to_path_buf(),
            file_stem: "out".into(),
            if_exists: IfExists::Replace,
            source_path: None,
        };
        let mut src = Failing {
            inner: ImageBands::new(pixels, OutputSpace::Srgb),
            calls: 0,
            fail_at,
            log: Mutex::new(Vec::new()),
        };
        let err = write_export_bands(
            &BandRequest {
                format,
                color_space: ColorSpace::Srgb,
                metadata: &meta,
                remove_location: false,
            },
            &mut src,
            &dest,
            &ExportHooks::none(),
        )
        .expect_err("the source failed");
        if expect_cancel {
            assert!(matches!(err, ExportError::Cancelled), "{err:?}");
        } else {
            assert!(err.to_string().contains("the source failed"), "{err}");
        }
        let left: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert!(left.is_empty(), "{format:?}: left {left:?}");
        // The rows were asked for in order and each once.
        let log = src.log.lock().unwrap();
        let mut next = 0;
        for &(y0, rows) in log.iter() {
            assert_eq!(y0, next, "{format:?}: {log:?}");
            next += rows;
        }
    }
}

#[test]
fn a_dng_is_not_written_from_bands() {
    let data = common::display_scene(8, 6);
    let meta = common::metadata();
    let pixels = ImageRef::new(8, 6, &data, PixelEncoding::DisplayLinear).unwrap();
    let mut bands = ImageBands::new(pixels, OutputSpace::Srgb);
    let dir = tempfile::tempdir().unwrap();
    let dest = ExportDestination {
        dir: dir.path().to_path_buf(),
        file_stem: "out".into(),
        if_exists: IfExists::Replace,
        source_path: None,
    };
    let err = write_export_bands(
        &BandRequest {
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            metadata: &meta,
            remove_location: false,
        },
        &mut bands,
        &dest,
        &ExportHooks::none(),
    )
    .expect_err("a DNG has its own writer");
    assert!(matches!(err, ExportError::InvalidInput(_)), "{err:?}");
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn write_export_bands_places_the_file_like_write_export() {
    let (w, h) = (40_u32, 30_u32);
    let data = common::display_scene(w, h);
    let meta = common::metadata();
    let pixels = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
    let format = ExportFormat::Png {
        bits: BitDepth::Sixteen,
    };
    let dir = tempfile::tempdir().unwrap();
    let dest = |stem: &str| ExportDestination {
        dir: dir.path().to_path_buf(),
        file_stem: stem.into(),
        if_exists: IfExists::Replace,
        source_path: None,
    };
    let progress = Mutex::new(Vec::new());
    let note = |f: f32| progress.lock().unwrap().push(f);
    let hooks = ExportHooks::new(None, Some(&note));
    let mut bands = ImageBands::new(pixels, OutputSpace::Srgb);
    let by_bands = write_export_bands(
        &BandRequest {
            format,
            color_space: ColorSpace::Srgb,
            metadata: &meta,
            remove_location: false,
        },
        &mut bands,
        &dest("bands"),
        &hooks,
    )
    .unwrap();
    let by_image = ym_export::write_export(
        &ExportRequest {
            pixels,
            format,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            metadata: &meta,
            remove_location: false,
            dng: None,
        },
        &dest("image"),
        &ExportHooks::none(),
    )
    .unwrap();
    assert!(matches!(by_bands.placement, Placement::Written(_)));
    assert_eq!((by_bands.width, by_bands.height), (w, h));
    assert_eq!(by_bands.bytes, by_image.bytes);
    assert_eq!(
        std::fs::read(by_bands.path()).unwrap(),
        std::fs::read(by_image.path()).unwrap()
    );
    let p = progress.lock().unwrap();
    assert!(p.last().is_some_and(|&l| l >= 0.999), "{p:?}");
    assert!(p.windows(2).all(|w| w[0] <= w[1]), "monotone: {p:?}");
}
