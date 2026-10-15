//! PNG export: 8 or 16 bits, RGB (`docs/09-export.md`).
//!
//! Chunks written before the image data: `iCCP` (the ICC profile, zlib-compressed by the `png`
//! crate), `cICP` (colour primaries and transfer function as ITU-T H.273 code points, see
//! [`ColorSpace::cicp`]), `pHYs` (300 dpi), `eXIf` (the Exif TIFF structure) and an `iTXt` chunk with the keyword
//! `XML:com.adobe.xmp` holding the XMP packet. 16-bit samples are big-endian as PNG requires; 8-bit
//! output is dithered by the display encode.

use std::borrow::Cow;
use std::io::Write;

use png::{BitDepth as PngDepth, ColorType, Compression, Encoder, Info, PixelDimensions, Unit};
use ym_color::encode::OutputSpace;

use crate::error::ExportError;
use crate::options::{BitDepth, ColorSpace};
use crate::pixels::{BandSource, ExportHooks, ImageBands, ImageRef, band_rows, check_band_len};

/// The iTXt keyword Adobe's XMP specification reserves for an embedded packet.
pub const XMP_ITXT_KEYWORD: &str = "XML:com.adobe.xmp";

/// What a PNG carries besides pixels.
#[derive(Debug, Clone, Copy)]
pub struct PngMetadata<'a> {
    /// The Exif TIFF structure for the `eXIf` chunk ([`crate::exif::build_exif_blob`]).
    pub exif: Option<&'a [u8]>,
    /// The XMP packet.
    pub xmp: Option<&'a [u8]>,
    /// The ICC profile for `iCCP`.
    pub icc: Option<&'a [u8]>,
}

/// Encodes `image` as a PNG of `bits` per channel in `color_space`.
pub fn write_png<W: Write>(
    out: W,
    image: &ImageRef<'_>,
    color_space: ColorSpace,
    bits: BitDepth,
    metadata: &PngMetadata<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(), ExportError> {
    image.validate()?;
    let space: OutputSpace = color_space.output_space();
    write_png_bands(
        out,
        &mut ImageBands::new(*image, space),
        color_space,
        bits,
        metadata,
        hooks,
    )
}

/// Encodes the rows of `bands` (already through C13, 8 or 16 bit as `bits` says) as a PNG in
/// `color_space`. The bands are pulled in order while the encoder compresses the rows before them.
pub fn write_png_bands<W: Write>(
    out: W,
    bands: &mut dyn BandSource,
    color_space: ColorSpace,
    bits: BitDepth,
    metadata: &PngMetadata<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(), ExportError> {
    let (width, height) = (bands.width(), bands.height());
    if width == 0 || height == 0 {
        return Err(ExportError::invalid("the image has no pixels"));
    }
    let mut info = Info::with_size(width, height);
    info.color_type = ColorType::Rgb;
    info.bit_depth = match bits {
        BitDepth::Eight => PngDepth::Eight,
        BitDepth::Sixteen => PngDepth::Sixteen,
    };
    // 300 dpi = 11,811 pixels per metre.
    info.pixel_dims = Some(PixelDimensions {
        xppu: 11_811,
        yppu: 11_811,
        unit: Unit::Meter,
    });
    if let Some(icc) = metadata.icc {
        info.icc_profile = Some(Cow::Borrowed(icc));
    }
    if let Some(exif) = metadata.exif {
        info.exif_metadata = Some(Cow::Borrowed(exif));
    }
    if let Some(xmp) = metadata.xmp {
        let text = String::from_utf8(xmp.to_vec())
            .map_err(|_| ExportError::encode("PNG", "the XMP packet isn't valid UTF-8"))?;
        info.utf8_text
            .push(png::text_metadata::ITXtChunk::new(XMP_ITXT_KEYWORD, text));
    }

    let mut encoder = Encoder::with_info(out, info)?;
    encoder.set_compression(Compression::Balanced);
    let mut writer = encoder.write_header()?;
    // The png crate parses cICP but doesn't write it, so the chunk goes in by hand, before the image data.
    writer.write_chunk(png::chunk::cICP, &color_space.cicp())?;
    let mut stream = writer.stream_writer()?;

    let (w, h) = (width as usize, height as usize);
    let rows = band_rows(width);
    let mut y = 0;
    let mut band8 = Vec::new();
    let mut band16 = Vec::new();
    let mut bytes16 = Vec::new();
    while y < h {
        hooks.check()?;
        let n = rows.min(h - y);
        match bits {
            BitDepth::Eight => {
                bands.rows_u8(y, n, &mut band8)?;
                check_band_len(band8.len(), width, n)?;
                stream.write_all(&band8).map_err(png_io)?;
            }
            BitDepth::Sixteen => {
                bands.rows_u16(y, n, &mut band16)?;
                check_band_len(band16.len(), width, n)?;
                bytes16.clear();
                bytes16.reserve(n * w * 6);
                for s in &band16 {
                    bytes16.extend_from_slice(&s.to_be_bytes());
                }
                stream.write_all(&bytes16).map_err(png_io)?;
            }
        }
        y += n;
        hooks.report(y as f32 / h as f32);
    }
    stream.finish()?;
    writer.finish()?;
    Ok(())
}

fn png_io(e: std::io::Error) -> ExportError {
    ExportError::write("PNG", e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixels::PixelEncoding;

    fn decode(png: &[u8]) -> (png::OutputInfo, Vec<u8>, png::Info<'static>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(png.to_vec()));
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info().clone();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let out = reader.next_frame(&mut buf).unwrap();
        buf.truncate(out.buffer_size());
        (out, buf, info.to_owned())
    }

    fn encode(bits: BitDepth, cs: ColorSpace, value: f32) -> Vec<u8> {
        let (w, h) = (24_u32, 10_u32);
        let data = vec![value; (w * h * 3) as usize];
        let img = ImageRef::new(w, h, &data, PixelEncoding::DisplayLinear).unwrap();
        let icc = crate::icc::profile(cs);
        let meta = PngMetadata {
            exif: Some(b"II*\0\x08\0\0\0\0\0\0\0\0\0"),
            xmp: Some("<x:xmpmeta/>".as_bytes()),
            icc: Some(icc),
        };
        let mut out = Vec::new();
        write_png(&mut out, &img, cs, bits, &meta, &ExportHooks::none()).unwrap();
        out
    }

    #[test]
    fn eight_bit_round_trip_and_chunks() {
        let png = encode(BitDepth::Eight, ColorSpace::DisplayP3, 0.5);
        let (out, pixels, info) = decode(&png);
        assert_eq!((out.width, out.height), (24, 10));
        assert_eq!(out.bit_depth, PngDepth::Eight);
        assert_eq!(out.color_type, ColorType::Rgb);
        // Linear 0.5 is sRGB code 188 (187.5, dithered to 187 or 188).
        assert!(pixels.iter().all(|&v| v == 187 || v == 188));
        assert_eq!(
            info.icc_profile.as_deref(),
            Some(crate::icc::profile(ColorSpace::DisplayP3))
        );
        let cicp = info.coding_independent_code_points.unwrap();
        assert_eq!(
            (
                cicp.color_primaries,
                cicp.transfer_function,
                cicp.matrix_coefficients,
                cicp.is_video_full_range_image
            ),
            (12, 13, 0, true)
        );
        assert_eq!(info.utf8_text.len(), 1);
        assert_eq!(info.utf8_text[0].keyword, XMP_ITXT_KEYWORD);
        assert_eq!(info.utf8_text[0].get_text().unwrap(), "<x:xmpmeta/>");
        assert!(info.exif_metadata.is_some());
        let dims = info.pixel_dims.unwrap();
        assert_eq!((dims.xppu, dims.unit), (11_811, Unit::Meter));
        // Chunk order: IHDR, then the colour chunks and eXIf/iTXt before the first IDAT.
        let names = chunk_names(&png);
        let idat = names.iter().position(|n| n == "IDAT").unwrap();
        for wanted in ["iCCP", "cICP", "pHYs", "eXIf", "iTXt"] {
            let at = names
                .iter()
                .position(|n| n == wanted)
                .unwrap_or_else(|| panic!("{wanted} missing: {names:?}"));
            assert!(at < idat, "{wanted} before IDAT: {names:?}");
        }
        assert_eq!(names.first().unwrap(), "IHDR");
        assert_eq!(names.last().unwrap(), "IEND");
    }

    fn chunk_names(png: &[u8]) -> Vec<String> {
        let mut names = Vec::new();
        let mut i = 8;
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            names.push(String::from_utf8_lossy(&png[i + 4..i + 8]).into_owned());
            i += 12 + len;
        }
        names
    }

    #[test]
    fn sixteen_bit_values_are_exact() {
        let png = encode(BitDepth::Sixteen, ColorSpace::Srgb, 0.5);
        let (out, pixels, info) = decode(&png);
        assert_eq!(out.bit_depth, PngDepth::Sixteen);
        let want = ym_color::encode::quantize(ym_color::encode::srgb_encode(0.5), 16);
        for s in pixels.chunks_exact(2) {
            assert_eq!(u16::from_be_bytes([s[0], s[1]]), want);
        }
        let cicp = info.coding_independent_code_points.unwrap();
        assert_eq!((cicp.color_primaries, cicp.transfer_function), (1, 13));
    }

    #[test]
    fn rec2020_uses_the_bt1886_curve_and_its_cicp() {
        let png = encode(BitDepth::Sixteen, ColorSpace::Rec2020, 0.2);
        let (_, pixels, info) = decode(&png);
        let want = ym_color::encode::quantize(0.2_f64.powf(1.0 / 2.4), 16);
        assert_eq!(u16::from_be_bytes([pixels[0], pixels[1]]), want);
        let cicp = info.coding_independent_code_points.unwrap();
        assert_eq!((cicp.color_primaries, cicp.transfer_function), (9, 1));
    }

    #[test]
    fn output_is_deterministic() {
        assert_eq!(
            encode(BitDepth::Eight, ColorSpace::Srgb, 0.3),
            encode(BitDepth::Eight, ColorSpace::Srgb, 0.3)
        );
    }

    #[test]
    fn cancel_stops_before_the_image_is_finished() {
        let data = vec![0.5_f32; 8 * 8 * 3];
        let img = ImageRef::new(8, 8, &data, PixelEncoding::DisplayLinear).unwrap();
        let always = || true;
        let r = write_png(
            Vec::new(),
            &img,
            ColorSpace::Srgb,
            BitDepth::Eight,
            &PngMetadata {
                exif: None,
                xmp: None,
                icc: None,
            },
            &ExportHooks::new(Some(&always), None),
        );
        assert!(matches!(r, Err(ExportError::Cancelled)));
    }
}
