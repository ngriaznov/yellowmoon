//! Baseline JPEG export (`docs/09-export.md` section Pixel pipeline per format).
//!
//! * 8-bit RGB, dithered by the display encode (stage C13), converted to YCbCr by the encoder.
//! * 4:4:4 chroma at quality 90 and above, 4:2:0 (box-averaged chroma) below.
//! * Baseline (not progressive) with optimised Huffman tables.
//! * `APP0` JFIF (300 dpi), then `APP1` Exif, `APP1` XMP and the ICC profile in `APP2` chunks.
//!
//! The encoder is the `jpeg-encoder` crate (MIT or Apache-2.0, plus the IJG licence for its tables).

use std::io::Write;

mod interleave;

use jpeg_encoder::{ColorType, Encoder, PixelDensity, SamplingFactor};
use ym_color::encode::OutputSpace;

use crate::error::ExportError;
use crate::options::JpegQuality;
use crate::pixels::{BandSource, ExportHooks, ImageBands, ImageRef, band_rows, check_band_len};
use crate::xmp::XMP_APP1_HEADER;

/// The largest width or height a JPEG can have.
pub const MAX_JPEG_SIDE: u32 = 65_535;

/// What a JPEG carries besides pixels.
#[derive(Debug, Clone, Copy)]
pub struct JpegMetadata<'a> {
    /// The `APP1` Exif payload including the `Exif\0\0` header ([`crate::exif::build_exif_app1_payload`]).
    pub exif_app1: Option<&'a [u8]>,
    /// The XMP packet ([`crate::xmp::build_xmp`]).
    pub xmp: Option<&'a [u8]>,
    /// The ICC profile.
    pub icc: Option<&'a [u8]>,
}

/// Encodes `image` (display-linear or display-encoded) as a baseline JPEG.
pub fn write_jpeg<W: Write>(
    out: W,
    image: &ImageRef<'_>,
    space: OutputSpace,
    quality: JpegQuality,
    metadata: &JpegMetadata<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(), ExportError> {
    image.validate()?;
    write_jpeg_bands(
        out,
        &mut ImageBands::new(*image, space),
        quality,
        metadata,
        hooks,
    )
}

/// Encodes the rows of `bands` (8-bit, already through C13) as a baseline JPEG. The bands are pulled
/// in order, about a million pixels at a time, into one 8-bit buffer; the encoder then runs over it.
pub fn write_jpeg_bands<W: Write>(
    mut out: W,
    bands: &mut dyn BandSource,
    quality: JpegQuality,
    metadata: &JpegMetadata<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(), ExportError> {
    let (width, height) = (bands.width(), bands.height());
    if width == 0 || height == 0 {
        return Err(ExportError::invalid("the image has no pixels"));
    }
    if width > MAX_JPEG_SIDE || height > MAX_JPEG_SIDE {
        return Err(ExportError::Unsupported(format!(
            "JPEG can't be wider or taller than {MAX_JPEG_SIDE} pixels. Pick a smaller long edge, or export TIFF or PNG."
        )));
    }
    let (w, h) = (width as usize, height as usize);

    // Display encode (C13) band by band into one 8-bit buffer (a third of the f32 input).
    let mut pixels = Vec::with_capacity(w * h * 3);
    let mut band = Vec::new();
    let rows = band_rows(width);
    let mut y = 0;
    while y < h {
        hooks.check()?;
        let n = rows.min(h - y);
        bands.rows_u8(y, n, &mut band)?;
        check_band_len(band.len(), width, n)?;
        pixels.extend_from_slice(&band);
        y += n;
        hooks.report(0.5 * y as f32 / h as f32);
    }
    hooks.check()?;

    // The encoder writes its optimised-table output as one scan per component; the result is
    // rewritten as a single interleaved scan, which every decoder reads (see `interleave`).
    let mut scans = Vec::new();
    {
        let mut encoder = Encoder::new(&mut scans, quality.get());
        encoder.set_progressive(false);
        encoder.set_optimized_huffman_tables(true);
        encoder.set_density(PixelDensity::dpi(300));
        encoder.set_sampling_factor(if quality.is_full_chroma() {
            SamplingFactor::R_4_4_4
        } else {
            SamplingFactor::R_4_2_0
        });
        encoder.set_chroma_subsampling_method(jpeg_encoder::ChromaSubsamplingMethod::Average);
        if let Some(exif) = metadata.exif_app1 {
            encoder.add_app_segment(1, exif.to_vec())?;
        }
        if let Some(xmp) = metadata.xmp {
            let mut segment = XMP_APP1_HEADER.to_vec();
            segment.extend_from_slice(xmp);
            if segment.len() > 65_533 {
                return Err(ExportError::encode(
                    "JPEG",
                    format!(
                        "the XMP packet is {} bytes, more than a JPEG segment holds (65,533)",
                        segment.len()
                    ),
                ));
            }
            encoder.add_app_segment(1, segment)?;
        }
        if let Some(icc) = metadata.icc {
            // Splits into `ICC_PROFILE` chunks of at most 65,519 bytes.
            encoder.add_icc_profile(icc)?;
        }
        hooks.report(0.55);
        encoder.encode(&pixels, w as u16, h as u16, ColorType::Rgb)?;
    }
    drop(pixels);
    hooks.check()?;
    hooks.report(0.8);
    let interleaved = interleave::interleave_scans(&scans)?;
    drop(scans);
    out.write_all(&interleaved)
        .map_err(|e| ExportError::write("JPEG", e))?;
    hooks.report(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixels::{PixelEncoding, encode_rows_u8};

    /// The marker segments of a JPEG up to the start of scan: `(marker, payload)`.
    pub(crate) fn segments(jpeg: &[u8]) -> Vec<(u8, &[u8])> {
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "SOI");
        let mut i = 2;
        let mut out = Vec::new();
        while i + 4 <= jpeg.len() {
            assert_eq!(jpeg[i], 0xFF, "marker at {i}");
            let marker = jpeg[i + 1];
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            out.push((marker, &jpeg[i + 4..i + 2 + len]));
            i += 2 + len;
            if marker == 0xDA {
                break;
            }
        }
        out
    }

    fn gradient(w: u32, h: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push(x as f32 / w as f32);
                v.push(y as f32 / h as f32);
                v.push(0.25);
            }
        }
        v
    }

    fn encode(q: u8, meta: &JpegMetadata<'_>) -> Vec<u8> {
        let (w, h) = (48, 32);
        let data = gradient(w, h);
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let mut out = Vec::new();
        write_jpeg(
            &mut out,
            &img,
            OutputSpace::Srgb,
            JpegQuality::new(q).unwrap(),
            meta,
            &ExportHooks::none(),
        )
        .unwrap();
        out
    }

    #[test]
    fn baseline_with_requested_chroma_and_metadata_order() {
        let exif = crate::exif::build_exif_app1_payload(&crate::exif::ExifTables::default());
        let xmp = b"<x/>".to_vec();
        let icc = vec![7_u8; 100];
        let meta = JpegMetadata {
            exif_app1: Some(&exif),
            xmp: Some(&xmp),
            icc: Some(&icc),
        };
        for (q, sampling) in [
            (92, 0x11_u8),
            (90, 0x11),
            (89, 0x22),
            (50, 0x22),
            (100, 0x11),
        ] {
            let jpeg = encode(q, &meta);
            let segs = segments(&jpeg);
            let markers: Vec<u8> = segs.iter().map(|s| s.0).collect();
            assert_eq!(markers[0], 0xE0, "JFIF first");
            assert_eq!(&segs[0].1[0..5], b"JFIF\0");
            assert_eq!(markers[1], 0xE1);
            assert_eq!(&segs[1].1[0..6], b"Exif\0\0");
            assert_eq!(markers[2], 0xE1);
            assert_eq!(&segs[2].1[0..29], XMP_APP1_HEADER);
            assert_eq!(markers[3], 0xE2);
            assert_eq!(&segs[3].1[0..12], b"ICC_PROFILE\0");
            assert_eq!(&segs[3].1[12..14], &[1, 1], "chunk 1 of 1");
            assert_eq!(&segs[3].1[14..], &icc[..]);
            assert!(markers.contains(&0xC0), "baseline SOF0 at q{q}");
            assert!(!markers.contains(&0xC2), "no progressive SOF2");
            let sof = segs.iter().find(|s| s.0 == 0xC0).unwrap().1;
            assert_eq!(sof[5], 3, "three components");
            assert_eq!(sof[7], sampling, "luma sampling at q{q}");
            assert_eq!(sof[10], 0x11);
            assert_eq!(sof[13], 0x11);
            // JFIF density 300 dpi.
            let jfif = segs[0].1;
            assert_eq!(jfif[7], 1, "units: inches");
            assert_eq!(u16::from_be_bytes([jfif[8], jfif[9]]), 300);
        }
    }

    #[test]
    fn large_icc_profiles_are_chunked() {
        let icc: Vec<u8> = (0..150_000).map(|i| (i % 251) as u8).collect();
        let meta = JpegMetadata {
            exif_app1: None,
            xmp: None,
            icc: Some(&icc),
        };
        let jpeg = encode(92, &meta);
        let chunks: Vec<&[u8]> = segments(&jpeg)
            .into_iter()
            .filter(|s| s.0 == 0xE2)
            .map(|s| s.1)
            .collect();
        assert_eq!(chunks.len(), 3);
        let mut joined = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(&c[0..12], b"ICC_PROFILE\0");
            assert_eq!(c[12] as usize, i + 1);
            assert_eq!(c[13], 3);
            joined.extend_from_slice(&c[14..]);
        }
        assert_eq!(joined, icc);
    }

    #[test]
    fn optimised_huffman_tables_make_smaller_files() {
        let (w, h) = (96_u32, 64_u32);
        let data = gradient(w, h);
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let mut ours = Vec::new();
        write_jpeg(
            &mut ours,
            &img,
            OutputSpace::Srgb,
            JpegQuality::new(92).unwrap(),
            &JpegMetadata {
                exif_app1: None,
                xmp: None,
                icc: None,
            },
            &ExportHooks::none(),
        )
        .unwrap();
        // The same pixels with the standard tables.
        let mut pixels = Vec::new();
        encode_rows_u8(&img, OutputSpace::Srgb, 0, h as usize, &mut pixels);
        let mut standard = Vec::new();
        let mut e = Encoder::new(&mut standard, 92);
        e.set_density(PixelDensity::dpi(300));
        e.set_sampling_factor(SamplingFactor::R_4_4_4);
        e.encode(&pixels, w as u16, h as u16, ColorType::Rgb)
            .unwrap();
        assert!(
            ours.len() < standard.len(),
            "{} vs {}",
            ours.len(),
            standard.len()
        );
    }

    #[test]
    fn decodes_to_the_encoded_colors() {
        let (w, h) = (32_u32, 32_u32);
        // Left half linear 0.18 grey (sRGB code 118), right half linear 0.5 (code 188).
        let mut data = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0.18 } else { 0.5 };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let mut out = Vec::new();
        write_jpeg(
            &mut out,
            &img,
            OutputSpace::Srgb,
            JpegQuality::new(100).unwrap(),
            &JpegMetadata {
                exif_app1: None,
                xmp: None,
                icc: None,
            },
            &ExportHooks::none(),
        )
        .unwrap();
        let mut dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(out));
        let pixels = dec.decode().unwrap();
        let info = dec.info().unwrap();
        assert_eq!((info.width as u32, info.height as u32), (w, h));
        let at = |x: usize, y: usize| &pixels[(y * w as usize + x) * 3..][..3];
        for (x, want) in [(4, 118_i32), (28, 188)] {
            for c in at(x, 16) {
                assert!((i32::from(*c) - want).abs() <= 3, "x={x}: {c} vs {want}");
            }
        }
    }

    #[test]
    fn oversize_and_cancel_are_reported() {
        let data = vec![0.5_f32; 4 * 4 * 3];
        let img = ImageRef::new(4, 4, &data, PixelEncoding::DisplayLinear).unwrap();
        let always = || true;
        let hooks = ExportHooks::new(Some(&always), None);
        let r = write_jpeg(
            Vec::new(),
            &img,
            OutputSpace::Srgb,
            JpegQuality::default(),
            &JpegMetadata {
                exif_app1: None,
                xmp: None,
                icc: None,
            },
            &hooks,
        );
        assert!(matches!(r, Err(ExportError::Cancelled)));
        let huge_xmp = vec![b'x'; 70_000];
        let r = write_jpeg(
            Vec::new(),
            &img,
            OutputSpace::Srgb,
            JpegQuality::default(),
            &JpegMetadata {
                exif_app1: None,
                xmp: Some(&huge_xmp),
                icc: None,
            },
            &ExportHooks::none(),
        );
        assert!(matches!(r, Err(ExportError::Encode { .. })));
    }
}
