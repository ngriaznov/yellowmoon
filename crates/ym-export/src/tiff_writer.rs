//! TIFF export: 8 or 16 bits, RGB, strips (`docs/09-export.md`).
//!
//! The file is built on the `tiff` crate's directory encoder. Strips are compressed here, in
//! parallel batches, with the crate's own LZW and Deflate implementations; for the compressed
//! modes the horizontal predictor (TIFF `Predictor` 2) is applied first, as Photoshop and libtiff
//! do for photographic data. Tags: ICC profile 34675, XMP 700, Exif and GPS sub-IFDs (the same
//! entries as the JPEG's Exif), orientation 1 (upright pixels), 300 dpi. Values that do not fit in
//! the IFD entry start on a word boundary, as the TIFF specification requires.
//!
//! Standard TIFF cannot address more than 4 GiB; a bigger file is refused with a message that
//! suggests a smaller size.

use std::io::{Seek, Write};

use rayon::prelude::*;
use tiff::encoder::compression::{CompressionAlgorithm, Deflate, DeflateLevel, Lzw};
use tiff::encoder::{Rational, TiffEncoder};

use crate::error::ExportError;
use crate::exif::{self, EXIF_IFD_POINTER, ExifTables, GPS_IFD_POINTER, Undefined, put};
use crate::options::{BitDepth, ColorSpace, TiffCompression};
use crate::pixels::{ExportHooks, ImageRef, encode_rows_u8, encode_rows_u16};

/// What a TIFF carries besides pixels.
#[derive(Debug, Clone, Copy)]
pub struct TiffMetadata<'a> {
    /// The EXIF entries (descriptive IFD0 tags, Exif IFD, GPS IFD).
    pub exif: &'a ExifTables,
    /// The XMP packet.
    pub xmp: Option<&'a [u8]>,
    /// The ICC profile.
    pub icc: Option<&'a [u8]>,
}

/// Uncompressed bytes aimed at in each strip.
const STRIP_TARGET_BYTES: usize = 512 * 1024;
/// Strips compressed per parallel batch.
const STRIPS_PER_BATCH: usize = 16;

/// Applies the horizontal predictor to 8-bit RGB rows in place.
fn predict_u8(rows: &mut [u8], row_len: usize) {
    for row in rows.chunks_exact_mut(row_len) {
        for i in (3..row_len).rev() {
            row[i] = row[i].wrapping_sub(row[i - 3]);
        }
    }
}

/// Applies the horizontal predictor to 16-bit RGB rows in place.
fn predict_u16(rows: &mut [u16], row_len: usize) {
    for row in rows.chunks_exact_mut(row_len) {
        for i in (3..row_len).rev() {
            row[i] = row[i].wrapping_sub(row[i - 3]);
        }
    }
}

fn compress(method: TiffCompression, raw: Vec<u8>) -> Result<Vec<u8>, ExportError> {
    let mut out = Vec::with_capacity(raw.len() / 2 + 64);
    match method {
        TiffCompression::None => return Ok(raw),
        TiffCompression::Lzw => {
            Lzw.write_to(&mut out, &raw)
                .map_err(|e| ExportError::encode("TIFF", e))?;
        }
        TiffCompression::Zip => {
            Deflate::with_level(DeflateLevel::Balanced)
                .write_to(&mut out, &raw)
                .map_err(|e| ExportError::encode("TIFF", e))?;
        }
    }
    Ok(out)
}

fn compression_code(method: TiffCompression) -> u16 {
    match method {
        TiffCompression::None => 1,
        TiffCompression::Lzw => 5,
        TiffCompression::Zip => 8,
    }
}

fn sorted(entries: &[exif::Entry]) -> Vec<exif::Entry> {
    let mut v = entries.to_vec();
    v.sort_by_key(|e| e.tag);
    v
}

/// Encodes `image` as a TIFF.
pub fn write_tiff<W: Write + Seek>(
    out: W,
    image: &ImageRef<'_>,
    color_space: ColorSpace,
    bits: BitDepth,
    compression: TiffCompression,
    metadata: &TiffMetadata<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(), ExportError> {
    image.validate()?;
    let space = color_space.output_space();
    let (w, h) = (image.width as usize, image.height as usize);
    let bytes_per_sample = match bits {
        BitDepth::Eight => 1_usize,
        BitDepth::Sixteen => 2,
    };
    let row_bytes = w * 3 * bytes_per_sample;
    if (row_bytes as u64) * (h as u64) > 4_000_000_000 {
        return Err(ExportError::Unsupported(
            "A TIFF can't be larger than 4 GB. Pick a smaller long edge, a compressed TIFF or PNG."
                .to_owned(),
        ));
    }
    let rows_per_strip = (STRIP_TARGET_BYTES / row_bytes).clamp(1, h);
    let strip_count = h.div_ceil(rows_per_strip);

    let mut tiff = TiffEncoder::new(out)?;

    // Sub-IFDs are written first so their offsets can go into IFD0.
    let tables = metadata.exif;
    let exif_offset = if tables.exif.is_empty() {
        None
    } else {
        let mut dir = tiff.extra_directory()?;
        exif::write_directory(&mut dir, &sorted(&tables.exif))?;
        Some(exif::finish_with_offset(dir)?)
    };
    let gps_offset = if tables.gps.is_empty() {
        None
    } else {
        let mut dir = tiff.extra_directory()?;
        exif::write_directory(&mut dir, &sorted(&tables.gps))?;
        Some(exif::finish_with_offset(dir)?)
    };

    let mut dir = tiff.image_directory()?;
    let mut strip_offsets: Vec<u32> = Vec::with_capacity(strip_count);
    let mut strip_sizes: Vec<u32> = Vec::with_capacity(strip_count);

    let batch_rows = rows_per_strip * STRIPS_PER_BATCH;
    let mut y = 0;
    let mut band8 = Vec::new();
    let mut band16 = Vec::new();
    while y < h {
        hooks.check()?;
        let rows = batch_rows.min(h - y);
        // Raw bytes of each strip of this batch, predictor applied.
        let strips: Vec<Vec<u8>> = match bits {
            BitDepth::Eight => {
                encode_rows_u8(image, space, y, rows, &mut band8);
                band8
                    .chunks(rows_per_strip * row_bytes)
                    .map(<[u8]>::to_vec)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|mut s| {
                        if compression != TiffCompression::None {
                            predict_u8(&mut s, row_bytes);
                        }
                        s
                    })
                    .collect()
            }
            BitDepth::Sixteen => {
                encode_rows_u16(image, space, y, rows, &mut band16);
                band16
                    .chunks(rows_per_strip * w * 3)
                    .map(<[u16]>::to_vec)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|mut s| {
                        if compression != TiffCompression::None {
                            predict_u16(&mut s, w * 3);
                        }
                        s.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>()
                    })
                    .collect()
            }
        };
        let packed: Vec<Vec<u8>> = strips
            .into_par_iter()
            .map(|s| compress(compression, s))
            .collect::<Result<_, _>>()?;
        for strip in packed {
            let offset = dir.write_data(&strip[..])?;
            strip_offsets.push(u32::try_from(offset).map_err(|_| {
                ExportError::Unsupported("A TIFF can't be larger than 4 GB.".to_owned())
            })?);
            strip_sizes.push(strip.len() as u32);
        }
        y += rows;
        hooks.report(y as f32 / h as f32);
    }
    hooks.check()?;

    let sample_bits = u16::from(bits.bits());
    put(&mut dir, 256, image.width)?;
    put(&mut dir, 257, image.height)?;
    put(&mut dir, 258, &[sample_bits; 3][..])?;
    put(&mut dir, 259, compression_code(compression))?;
    put(&mut dir, 262, 2_u16)?; // RGB
    put(&mut dir, 273, &strip_offsets[..])?;
    put(&mut dir, 274, 1_u16)?; // upright
    put(&mut dir, 277, 3_u16)?;
    put(&mut dir, 278, rows_per_strip as u32)?;
    put(&mut dir, 279, &strip_sizes[..])?;
    put(&mut dir, 282, Rational { n: 300, d: 1 })?;
    put(&mut dir, 283, Rational { n: 300, d: 1 })?;
    put(&mut dir, 284, 1_u16)?; // chunky
    put(&mut dir, 296, 2_u16)?; // inches
    if compression != TiffCompression::None {
        put(&mut dir, 317, 2_u16)?; // horizontal predictor
    }
    put(&mut dir, 339, &[1_u16; 3][..])?; // unsigned integer samples
    exif::write_directory(&mut dir, &sorted(&tables.ifd0))?;
    if let Some(xmp) = metadata.xmp {
        put(&mut dir, 700, xmp)?;
    }
    if let Some(icc) = metadata.icc {
        put(&mut dir, 34675, Undefined(icc))?;
    }
    if let Some(offset) = exif_offset {
        put(&mut dir, EXIF_IFD_POINTER, offset)?;
    }
    if let Some(offset) = gps_offset {
        put(&mut dir, GPS_IFD_POINTER, offset)?;
    }
    exif::finish(dir)?;
    hooks.report(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixels::PixelEncoding;
    use std::io::Cursor;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;

    fn pattern(w: u32, h: u32) -> Vec<f32> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.push(x as f32 / (w - 1) as f32);
                v.push(y as f32 / (h - 1) as f32);
                v.push(((x * 7 + y * 3) % 17) as f32 / 16.0);
            }
        }
        v
    }

    fn meta<'a>(
        tables: &'a ExifTables,
        icc: Option<&'a [u8]>,
        xmp: Option<&'a [u8]>,
    ) -> TiffMetadata<'a> {
        TiffMetadata {
            exif: tables,
            xmp,
            icc,
        }
    }

    fn encode(bits: BitDepth, comp: TiffCompression, w: u32, h: u32) -> Vec<u8> {
        let data = pattern(w, h);
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let tables = ExifTables::default();
        let mut out = Cursor::new(Vec::new());
        write_tiff(
            &mut out,
            &img,
            ColorSpace::Srgb,
            bits,
            comp,
            &meta(&tables, None, None),
            &ExportHooks::none(),
        )
        .unwrap();
        out.into_inner()
    }

    #[test]
    fn every_mode_round_trips_to_the_same_pixels() {
        let (w, h) = (53_u32, 41_u32);
        let data = pattern(w, h);
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let mut want16 = Vec::new();
        encode_rows_u16(
            &img,
            ym_color::encode::OutputSpace::Srgb,
            0,
            h as usize,
            &mut want16,
        );
        let mut want8 = Vec::new();
        encode_rows_u8(
            &img,
            ym_color::encode::OutputSpace::Srgb,
            0,
            h as usize,
            &mut want8,
        );
        for comp in [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Zip,
        ] {
            for bits in [BitDepth::Eight, BitDepth::Sixteen] {
                let file = encode(bits, comp, w, h);
                let mut dec = Decoder::new(Cursor::new(file)).unwrap();
                assert_eq!(dec.dimensions().unwrap(), (w, h));
                match (bits, dec.read_image().unwrap()) {
                    (BitDepth::Eight, DecodingResult::U8(got)) => {
                        assert_eq!(got, want8, "{comp:?} 8")
                    }
                    (BitDepth::Sixteen, DecodingResult::U16(got)) => {
                        assert_eq!(got, want16, "{comp:?} 16")
                    }
                    (b, other) => panic!(
                        "{b:?} {comp:?}: unexpected {:?}",
                        std::mem::discriminant(&other)
                    ),
                }
            }
        }
    }

    #[test]
    fn compressed_modes_are_smaller_on_smooth_data() {
        let none = encode(BitDepth::Sixteen, TiffCompression::None, 256, 128).len();
        let lzw = encode(BitDepth::Sixteen, TiffCompression::Lzw, 256, 128).len();
        let zip = encode(BitDepth::Sixteen, TiffCompression::Zip, 256, 128).len();
        assert!(lzw < none && zip < none, "none {none} lzw {lzw} zip {zip}");
    }

    #[test]
    fn tags_icc_xmp_and_exif_are_present() {
        let data = pattern(32, 24);
        let img = ImageRef::new(32, 24, &data, PixelEncoding::DisplayLinear).unwrap();
        let info = exif::ExifInfo {
            make: Some("SONY".into()),
            model: Some("ILCE-1".into()),
            iso: Some(800),
            gps: Some(exif::Gps {
                latitude_deg: 10.0,
                longitude_deg: 20.0,
                altitude_m: None,
            }),
            ..exif::ExifInfo::default()
        };
        let facts = exif::OutputFacts {
            width: 32,
            height: 24,
            color_space: Some(ColorSpace::DisplayP3),
            ycbcr: false,
        };
        let tables = info.tables(&facts, "Yellowmoon 0.0.1", false);
        let icc = crate::icc::profile(ColorSpace::DisplayP3);
        let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
        let mut out = Cursor::new(Vec::new());
        write_tiff(
            &mut out,
            &img,
            ColorSpace::DisplayP3,
            BitDepth::Sixteen,
            TiffCompression::Lzw,
            &meta(&tables, Some(icc), Some(xmp)),
            &ExportHooks::none(),
        )
        .unwrap();
        let file = out.into_inner();
        let mut dec = Decoder::new(Cursor::new(file)).unwrap();
        let get_bytes = |dec: &mut Decoder<Cursor<Vec<u8>>>, n: u16| {
            dec.get_tag(Tag::Unknown(n)).unwrap().into_u8_vec().unwrap()
        };
        assert_eq!(get_bytes(&mut dec, 34675), icc);
        assert_eq!(get_bytes(&mut dec, 700), xmp);
        assert_eq!(
            dec.get_tag(Tag::Unknown(271))
                .unwrap()
                .into_string()
                .unwrap(),
            "SONY"
        );
        assert_eq!(
            dec.get_tag(Tag::Unknown(272))
                .unwrap()
                .into_string()
                .unwrap(),
            "ILCE-1"
        );
        assert_eq!(
            dec.get_tag(Tag::Unknown(274)).unwrap().into_u32().unwrap(),
            1
        );
        assert_eq!(
            dec.get_tag(Tag::Unknown(259)).unwrap().into_u32().unwrap(),
            5
        );
        assert_eq!(
            dec.get_tag(Tag::Unknown(317)).unwrap().into_u32().unwrap(),
            2
        );
        assert!(dec.get_tag(Tag::Unknown(34665)).is_ok(), "Exif IFD pointer");
        assert!(dec.get_tag(Tag::Unknown(34853)).is_ok(), "GPS IFD pointer");
    }

    #[test]
    fn output_is_deterministic_and_cancel_is_reported() {
        let a = encode(BitDepth::Sixteen, TiffCompression::Zip, 64, 48);
        let b = encode(BitDepth::Sixteen, TiffCompression::Zip, 64, 48);
        assert_eq!(a, b);
        let data = pattern(8, 8);
        let img = ImageRef::new(8, 8, &data, PixelEncoding::DisplayLinear).unwrap();
        let tables = ExifTables::default();
        let always = || true;
        let r = write_tiff(
            Cursor::new(Vec::new()),
            &img,
            ColorSpace::Srgb,
            BitDepth::Eight,
            TiffCompression::None,
            &meta(&tables, None, None),
            &ExportHooks::new(Some(&always), None),
        );
        assert!(matches!(r, Err(ExportError::Cancelled)));
    }

    #[test]
    fn predictor_is_the_inverse_of_cumulative_sum() {
        let mut row = vec![10_u8, 20, 30, 15, 25, 35, 250, 1, 2];
        let orig = row.clone();
        predict_u8(&mut row, 9);
        assert_eq!(&row[0..3], &orig[0..3]);
        assert_eq!(row[3], 5);
        assert_eq!(row[6], 235);
        // Undo with a running sum per channel.
        for i in 3..9 {
            row[i] = row[i].wrapping_add(row[i - 3]);
        }
        assert_eq!(row, orig);
    }
}
