//! Reading a reference for Reference Grade (`docs/34-reference-grade.md` section 6, RG-05).
//!
//! A reference is a picture the person chooses to match: an image file (JPEG, PNG or TIFF, 8 or 16 bits, RGB or grey) or a photo of
//! the filmstrip ([`ReferencePicture::from_snapshot`], section 6.4). Reading it gives the two pictures the fit and the segmenter
//! work on, and a few notes for the card:
//!
//! 1. **Decode** (`zune-jpeg` with its SIMD paths off, `png`, `tiff`: the workspace's decoders), refusing anything else with the
//!    plain text of section 9.3 (HEIC and the others "not supported", a raw file too: its picture depends on its settings).
//! 2. **Orient** by the EXIF tag of a JPEG or a TIFF (values 1 to 8), because the scene model expects an upright picture
//!    (section 15, decision 18).
//! 3. **Trim black bars** (`ym_color::letterbox`: a step at the picture's edge, a ringing margin, night frames left alone).
//! 4. **Refuse a small picture** (its short side under [`MIN_SHORT_SIDE_PX`], section 16.6).
//! 5. **Convert to linear sRGB** in `f64` (`ym_color::icc`: an embedded matrix/TRC profile, PNG `cICP`, PNG `sRGB`, untagged as
//!    sRGB; a profile that cannot be read is taken as sRGB and the card says so), clamped to 0 to 1 as the fit's pictures are.
//! 6. **Reduce** by the deterministic area average in linear light (`ym_color::reduce`) to the **scene picture** (long edge at most
//!    [`SCENE_LONG_EDGE_PX`], what the segmenter and MediaPipe read) and the **analysis picture** (at most
//!    [`ANALYSIS_LONG_EDGE_PX`], what the fit measures). Neither is ever enlarged.
//!
//! **Nothing of a reference is ever kept** (section 6.6). The file is only read. A [`ReferencePicture`] lives in memory for the
//! session and has no way to be written: it implements neither `Serialize` nor any save, and the core's settings, presets, albums
//! and XMP have no field for it (`tests/reference_loading.rs` checks that no byte of one reaches a file the core writes).
//!
//! Everything here is `f64` or integer arithmetic in a fixed order: the same bytes give the same pictures on every machine and at
//! every thread count (`AGENTS.md` rule 8).

use std::io::Cursor;
use std::path::Path;

use ym_color::encode::srgb_encode;
use ym_color::icc::{RgbProfile, ToLinearSrgb};
use ym_color::letterbox::{self, Bars, Letterbox, Sample};
use ym_color::reduce::{area_average, reduced_size};

/// The long edge of the scene picture, in pixels (section 6.5): the picture the segmenter and MediaPipe read.
pub const SCENE_LONG_EDGE_PX: usize = 1024;

/// The long edge of the analysis picture, in pixels (section 6.5, 5.1): the picture the fit measures.
pub const ANALYSIS_LONG_EDGE_PX: usize = 512;

/// The shortest short side a reference may have after its black bars are removed (section 16.6).
pub const MIN_SHORT_SIDE_PX: usize = 256;

/// The most pixels a file may hold; a bigger one is refused as unreadable (a 16-bit RGB picture of this size is 1.2 GB decoded).
pub const MAX_PIXELS: usize = 200_000_000;

/// Why a picture cannot be a reference. The `Display` text is the sentence of section 9.3, shown in the card and, after "The
/// reference can't be used:", by `ym`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReferenceError {
    /// The file cannot be opened or decoded.
    #[error("This file can't be read. It may be damaged. Choose another picture.")]
    Unreadable,
    /// Not a JPEG, PNG or TIFF picture (HEIC, a raw file, a document, and so on).
    #[error(
        "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."
    )]
    Unsupported,
    /// The short side, after the bars are removed, is under [`MIN_SHORT_SIDE_PX`].
    #[error(
        "This picture is too small to match. Use one that is at least {} pixels on its short side.",
        MIN_SHORT_SIDE_PX
    )]
    TooSmall,
    /// Every line of the picture is a bar line.
    #[error("There's no picture inside the black bars. Choose another picture.")]
    Empty,
    /// The photo chosen from the strip is the photo being graded.
    #[error("That's the photo you're grading. Choose another.")]
    SamePhoto,
}

/// What the card says besides the status line (section 9.3, *Notes*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReferenceNote {
    /// Black bars were cut off the picture.
    BlackBarsRemoved,
    /// The embedded profile (or the `cICP` code points) could not be read or is not one this reader handles; the picture is taken
    /// as sRGB.
    ProfileUnreadable,
    /// The picture is grey: the photo is matched in black and white.
    NoColour,
}

impl ReferenceNote {
    /// The note's sentence.
    #[must_use]
    pub fn text(self) -> &'static str {
        match self {
            ReferenceNote::BlackBarsRemoved => "Black bars were removed from the reference.",
            ReferenceNote::ProfileUnreadable => {
                "The picture's colour profile couldn't be read, so it is taken as sRGB."
            }
            ReferenceNote::NoColour => {
                "This picture has no colour. The photo will be matched in black and white."
            }
        }
    }
}

/// A picture in linear sRGB, `f64`, 0 to 1 (the form the fit measures: CIELAB D65 is made from it).
#[derive(Clone, PartialEq)]
pub struct ReducedPicture {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` pixels, rows top to bottom, linear sRGB.
    pub linear: Vec<[f64; 3]>,
}

impl std::fmt::Debug for ReducedPicture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReducedPicture")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl ReducedPicture {
    /// The picture as sRGB display codes, `[y][x][rgb]`, 8 bits: the sRGB curve of each channel, rounded. The segmenter reads this.
    #[must_use]
    pub fn srgb8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.linear.len() * 3);
        for p in &self.linear {
            for &c in p {
                out.push((srgb_encode(c.clamp(0.0, 1.0)) * 255.0).round() as u8);
            }
        }
        out
    }
}

/// Where a reference came from. Never anything that can be opened again: a file is not remembered by its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSource {
    /// An image file the person chose.
    File,
    /// A snapshot of a photo of the strip, rendered through its own settings (section 6.4).
    Photo(crate::session::PhotoId),
}

/// A reference, read: the scene and analysis pictures and what to tell the person. Held in memory for the session and nowhere
/// else (section 6.6).
#[derive(Clone)]
pub struct ReferencePicture {
    scene: ReducedPicture,
    analysis: ReducedPicture,
    notes: Vec<ReferenceNote>,
    grey: bool,
    source: ReferenceSource,
    digest: u64,
}

impl std::fmt::Debug for ReferencePicture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferencePicture")
            .field("scene", &self.scene)
            .field("analysis", &self.analysis)
            .field("notes", &self.notes)
            .field("grey", &self.grey)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl ReferencePicture {
    /// Reads the image file at `path` (only reads it, rule 5).
    ///
    /// # Errors
    ///
    /// A [`ReferenceError`]: [`Unsupported`](ReferenceError::Unsupported) for a raw file by its extension and for anything that is
    /// not JPEG, PNG or TIFF by its first bytes, [`Unreadable`](ReferenceError::Unreadable) for a file that cannot be read or
    /// decoded, and the errors of [`ReferencePicture::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self, ReferenceError> {
        if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && RAW_EXTENSIONS.iter().any(|r| ext.eq_ignore_ascii_case(r))
        {
            return Err(ReferenceError::Unsupported);
        }
        let bytes = std::fs::read(path).map_err(|e| {
            tracing::debug!(error = %e, "a reference file could not be read");
            ReferenceError::Unreadable
        })?;
        Self::from_bytes(&bytes)
    }

    /// Reads a JPEG, PNG or TIFF picture from its bytes (the file's, which are let go of when this returns).
    ///
    /// # Errors
    ///
    /// [`Unsupported`](ReferenceError::Unsupported) for another format, [`Unreadable`](ReferenceError::Unreadable) for a damaged
    /// one, [`Empty`](ReferenceError::Empty) when every line is a bar line and [`TooSmall`](ReferenceError::TooSmall) when the
    /// picture inside the bars is under [`MIN_SHORT_SIDE_PX`] on its short side.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReferenceError> {
        let decoded = decode(bytes)?;
        let Decoded {
            width,
            height,
            channels,
            samples,
            icc,
            cicp,
            orientation,
        } = decoded;
        let grey = channels == 1;
        let mut notes = Vec::new();
        // The colour first: it only reads the tags.
        let colour = match (&icc, cicp) {
            (Some(Some(profile)), _) => match RgbProfile::parse(profile) {
                Ok(p) => p.to_linear_srgb(),
                Err(e) => {
                    tracing::debug!(error = %e, "a reference's profile was refused; sRGB");
                    notes.push(ReferenceNote::ProfileUnreadable);
                    ToLinearSrgb::srgb()
                }
            },
            // A profile the container held but could not give us.
            (Some(None), _) => {
                notes.push(ReferenceNote::ProfileUnreadable);
                ToLinearSrgb::srgb()
            }
            (None, Some(code_points)) => {
                ToLinearSrgb::from_cicp(code_points).unwrap_or_else(|| {
                    notes.push(ReferenceNote::ProfileUnreadable);
                    ToLinearSrgb::srgb()
                })
            }
            (None, None) => ToLinearSrgb::srgb(),
        };
        match samples {
            Samples::U8(data) => finish(
                data,
                width,
                height,
                channels,
                orientation,
                &colour,
                grey,
                notes,
            ),
            Samples::U16(data) => finish(
                data,
                width,
                height,
                channels,
                orientation,
                &colour,
                grey,
                notes,
            ),
        }
    }

    /// A reference made of a photo of the strip (section 6.4): `rgb` is the photo rendered through its own current settings to
    /// sRGB display codes (`width x height`, 8 bits, rows top to bottom, at most [`SCENE_LONG_EDGE_PX`] on the long edge: the same
    /// CPU reference path Auto's analysis picture uses, `Renderer::reference_snapshot`). It has no profile, no bars and no size
    /// limit; it is the photo as the person sees it.
    ///
    /// # Errors
    ///
    /// [`ReferenceError::SamePhoto`] when `photo` is `graded`, the photo being graded (section 6.4); [`ReferenceError::Unreadable`]
    /// when `rgb` is not `width * height * 3` bytes or is empty.
    pub fn from_snapshot(
        photo: crate::session::PhotoId,
        graded: crate::session::PhotoId,
        width: usize,
        height: usize,
        rgb: &[u8],
    ) -> Result<Self, ReferenceError> {
        if photo == graded {
            return Err(ReferenceError::SamePhoto);
        }
        if width == 0 || height == 0 || rgb.len() != width * height * 3 {
            return Err(ReferenceError::Unreadable);
        }
        let table = ToLinearSrgb::srgb().table(0, 256);
        let pixel = |x: usize, y: usize| {
            let at = (y * width + x) * 3;
            [
                table[usize::from(rgb[at])],
                table[usize::from(rgb[at + 1])],
                table[usize::from(rgb[at + 2])],
            ]
        };
        let (sw, sh) = reduced_size(width, height, SCENE_LONG_EDGE_PX);
        let (aw, ah) = reduced_size(width, height, ANALYSIS_LONG_EDGE_PX);
        let scene = ReducedPicture {
            width: sw,
            height: sh,
            linear: area_average(width, height, sw, sh, pixel),
        };
        let analysis = ReducedPicture {
            width: aw,
            height: ah,
            linear: area_average(width, height, aw, ah, pixel),
        };
        let digest = digest_of(&analysis);
        Ok(Self {
            scene,
            analysis,
            notes: Vec::new(),
            grey: false,
            source: ReferenceSource::Photo(photo),
            digest,
        })
    }

    /// The scene picture (long edge at most [`SCENE_LONG_EDGE_PX`]): what the segmenter and MediaPipe read, as
    /// [`ReducedPicture::srgb8`].
    #[must_use]
    pub fn scene(&self) -> &ReducedPicture {
        &self.scene
    }

    /// The analysis picture (long edge at most [`ANALYSIS_LONG_EDGE_PX`]): linear sRGB, `f64`, all of its pixels are the
    /// reference's pixels for the fit (section 5.1). The fit makes CIELAB D65 from it.
    #[must_use]
    pub fn analysis(&self) -> &ReducedPicture {
        &self.analysis
    }

    /// The notes for the card, in a fixed order.
    #[must_use]
    pub fn notes(&self) -> &[ReferenceNote] {
        &self.notes
    }

    /// Whether the picture is grey (one channel in the file): the photo is matched in black and white.
    #[must_use]
    pub fn is_grey(&self) -> bool {
        self.grey
    }

    /// Where the reference came from.
    #[must_use]
    pub fn source(&self) -> ReferenceSource {
        self.source
    }

    /// A number that names this picture's pixels (FNV-1a of the analysis picture's bits and size): equal pictures have equal
    /// digests. It keys what is made from the reference in memory (the scene), and is never written anywhere.
    #[must_use]
    pub fn digest(&self) -> u64 {
        self.digest
    }
}

fn digest_of(p: &ReducedPicture) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&(p.width as u64).to_le_bytes());
    eat(&(p.height as u64).to_le_bytes());
    for px in &p.linear {
        for c in px {
            eat(&c.to_bits().to_le_bytes());
        }
    }
    h
}

/// File extensions of raw files: a raw's picture depends on its settings, and several are TIFF underneath (section 6.1).
const RAW_EXTENSIONS: &[&str] = &[
    "arw", "sr2", "srf", "dng", "cr2", "cr3", "crw", "nef", "nrw", "raf", "orf", "rw2", "pef",
    "srw", "3fr", "fff", "erf", "kdc", "mrw", "raw", "rwl", "iiq", "x3f",
];

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// A decoded file: samples as stored (`channels` is 1 or 3), what it says about its colour, and its EXIF orientation.
struct Decoded {
    width: usize,
    height: usize,
    channels: usize,
    samples: Samples,
    /// `Some(Some(bytes))` an embedded profile; `Some(None)` the container holds one that could not be read out; `None` none.
    icc: Option<Option<Vec<u8>>>,
    /// PNG `cICP` code points.
    cicp: Option<[u8; 4]>,
    /// EXIF orientation, 1 to 8.
    orientation: u16,
}

fn decode(bytes: &[u8]) -> Result<Decoded, ReferenceError> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(bytes)
    } else if bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || bytes.starts_with(b"II+\0")
        || bytes.starts_with(b"MM\0+")
    {
        decode_tiff(bytes)
    } else {
        Err(ReferenceError::Unsupported)
    }
}

fn check_size(width: usize, height: usize) -> Result<(), ReferenceError> {
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_PIXELS {
        Err(ReferenceError::Unreadable)
    } else {
        Ok(())
    }
}

fn decode_jpeg(bytes: &[u8]) -> Result<Decoded, ReferenceError> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // The SIMD paths are off: the decoded bytes must not depend on which instructions the machine has (rule 8).
    let options = DecoderOptions::default().set_use_unsafe(false);
    let mut probe = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    probe.decode_headers().map_err(|e| {
        tracing::debug!(error = %e, "a reference JPEG's headers could not be read");
        ReferenceError::Unreadable
    })?;
    let info = probe.info().ok_or(ReferenceError::Unreadable)?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    check_size(width, height)?;
    let (channels, out) = match info.components {
        1 => (1, ColorSpace::Luma),
        3 => (3, ColorSpace::RGB),
        // CMYK and YCCK pictures need their own profile to mean anything.
        _ => return Err(ReferenceError::Unreadable),
    };
    let icc = probe.icc_profile();
    let orientation = probe.exif().map_or(1, |e| exif_orientation(e));
    let options = DecoderOptions::default()
        .set_use_unsafe(false)
        .jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let data = decoder.decode().map_err(|e| {
        tracing::debug!(error = %e, "a reference JPEG could not be decoded");
        ReferenceError::Unreadable
    })?;
    if data.len() != width * height * channels {
        return Err(ReferenceError::Unreadable);
    }
    Ok(Decoded {
        width,
        height,
        channels,
        samples: Samples::U8(data),
        icc: icc.map(Some),
        cicp: None,
        orientation,
    })
}

/// The orientation (1 to 8) in the Exif data of a JPEG (the TIFF structure, starting at its header); 1 when there is none or it
/// cannot be read.
fn exif_orientation(exif: &[u8]) -> u16 {
    // Some writers keep the "Exif\0\0" prefix.
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let big = match tiff.get(0..2) {
        Some(b"MM") => true,
        Some(b"II") => false,
        _ => return 1,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let b = tiff.get(at..at + 2)?;
        Some(if big {
            u16::from_be_bytes([b[0], b[1]])
        } else {
            u16::from_le_bytes([b[0], b[1]])
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b = tiff.get(at..at + 4)?;
        Some(if big {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    let read = || -> Option<u16> {
        if u16_at(2)? != 42 {
            return None;
        }
        let ifd = u32_at(4)? as usize;
        let count = usize::from(u16_at(ifd)?);
        for i in 0..count {
            let entry = ifd + 2 + 12 * i;
            if u16_at(entry)? == 0x0112 {
                // A SHORT of count 1 sits in the first two bytes of the value field.
                return u16_at(entry + 8);
            }
        }
        None
    };
    read().filter(|o| (1..=8).contains(o)).unwrap_or(1)
}

fn decode_png(bytes: &[u8]) -> Result<Decoded, ReferenceError> {
    let unreadable = |e: png::DecodingError| {
        tracing::debug!(error = %e, "a reference PNG could not be decoded");
        ReferenceError::Unreadable
    };
    let mut decoder =
        png::Decoder::new_with_limits(Cursor::new(bytes), png::Limits { bytes: 1 << 30 });
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(unreadable)?;
    let (icc, cicp) = {
        let info = reader.info();
        let icc = info.icc_profile.as_ref().map(|p| Some(p.to_vec()));
        let cicp = info.coding_independent_code_points.map(|c| {
            [
                c.color_primaries,
                c.transfer_function,
                c.matrix_coefficients,
                u8::from(c.is_video_full_range_image),
            ]
        });
        (icc, cicp)
    };
    let (width, height) = {
        let i = reader.info();
        (i.width as usize, i.height as usize)
    };
    check_size(width, height)?;
    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or(ReferenceError::Unreadable)?
    ];
    let frame = reader.next_frame(&mut buf).map_err(unreadable)?;
    let (color, depth) = reader.output_color_type();
    let (channels_in, keep) = match color {
        png::ColorType::Grayscale => (1, 1),
        png::ColorType::GrayscaleAlpha => (2, 1),
        png::ColorType::Rgb => (3, 3),
        png::ColorType::Rgba => (4, 3),
        png::ColorType::Indexed => return Err(ReferenceError::Unreadable),
    };
    let pixels = width * height;
    let data = &buf[..frame.buffer_size()];
    let samples = match depth {
        png::BitDepth::Eight => {
            if data.len() < pixels * channels_in {
                return Err(ReferenceError::Unreadable);
            }
            if channels_in == keep {
                Samples::U8(data[..pixels * keep].to_vec())
            } else {
                // The alpha channel is not part of the colour: it is left out.
                let mut v = Vec::with_capacity(pixels * keep);
                for px in data.chunks_exact(channels_in).take(pixels) {
                    v.extend_from_slice(&px[..keep]);
                }
                Samples::U8(v)
            }
        }
        png::BitDepth::Sixteen => {
            if data.len() < pixels * channels_in * 2 {
                return Err(ReferenceError::Unreadable);
            }
            let mut v = Vec::with_capacity(pixels * keep);
            for px in data.chunks_exact(channels_in * 2).take(pixels) {
                for c in 0..keep {
                    v.push(u16::from_be_bytes([px[2 * c], px[2 * c + 1]]));
                }
            }
            Samples::U16(v)
        }
        _ => return Err(ReferenceError::Unreadable),
    };
    Ok(Decoded {
        width,
        height,
        channels: keep,
        samples,
        icc,
        cicp,
        orientation: 1,
    })
}

fn decode_tiff(bytes: &[u8]) -> Result<Decoded, ReferenceError> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    use tiff::tags::Tag;
    let unreadable = |e: tiff::TiffError| {
        tracing::debug!(error = %e, "a reference TIFF could not be decoded");
        ReferenceError::Unreadable
    };
    let mut limits = Limits::default();
    limits.decoding_buffer_size = 1 << 30;
    limits.ifd_value_size = 16 << 20;
    limits.intermediate_buffer_size = 1 << 28;
    let mut decoder = Decoder::new(Cursor::new(bytes))
        .map_err(unreadable)?
        .with_limits(limits);
    // A DNG (a raw) is TIFF underneath.
    if decoder
        .find_tag(Tag::Unknown(50706))
        .map_err(unreadable)?
        .is_some()
    {
        return Err(ReferenceError::Unsupported);
    }
    let (w, h) = decoder.dimensions().map_err(unreadable)?;
    let (width, height) = (w as usize, h as usize);
    check_size(width, height)?;
    let color = decoder.colortype().map_err(unreadable)?;
    let (channels_in, keep, bits) = match color {
        tiff::ColorType::Gray(b) => (1, 1, b),
        tiff::ColorType::GrayA(b) => (2, 1, b),
        tiff::ColorType::RGB(b) => (3, 3, b),
        tiff::ColorType::RGBA(b) => (4, 3, b),
        _ => return Err(ReferenceError::Unreadable),
    };
    let orientation = decoder
        .find_tag_unsigned::<u16>(Tag::Orientation)
        .ok()
        .flatten()
        .filter(|o| (1..=8).contains(o))
        .unwrap_or(1);
    let icc = match decoder.find_tag(Tag::IccProfile) {
        Ok(None) => None,
        Ok(Some(_)) => Some(decoder.get_tag_u8_vec(Tag::IccProfile).ok()),
        Err(_) => Some(None),
    };
    let image = decoder.read_image().map_err(unreadable)?;
    let pixels = width * height;
    let samples = match (image, bits) {
        (DecodingResult::U8(d), 8) => {
            if d.len() < pixels * channels_in {
                return Err(ReferenceError::Unreadable);
            }
            if channels_in == keep {
                Samples::U8(d)
            } else {
                let mut v = Vec::with_capacity(pixels * keep);
                for px in d.chunks_exact(channels_in).take(pixels) {
                    v.extend_from_slice(&px[..keep]);
                }
                Samples::U8(v)
            }
        }
        (DecodingResult::U16(d), 16) => {
            if d.len() < pixels * channels_in {
                return Err(ReferenceError::Unreadable);
            }
            if channels_in == keep {
                Samples::U16(d)
            } else {
                let mut v = Vec::with_capacity(pixels * keep);
                for px in d.chunks_exact(channels_in).take(pixels) {
                    v.extend_from_slice(&px[..keep]);
                }
                Samples::U16(v)
            }
        }
        _ => return Err(ReferenceError::Unreadable),
    };
    Ok(Decoded {
        width,
        height,
        channels: keep,
        samples,
        icc,
        cicp: None,
        orientation,
    })
}

// ---------------------------------------------------------------------------------------------
// Orientation, bars, colour and reduction
// ---------------------------------------------------------------------------------------------

/// The picture turned upright by EXIF orientation `o` (1 to 8; 1 is returned as it is): the new samples and the new width and height.
fn orient<T: Copy>(
    data: Vec<T>,
    w: usize,
    h: usize,
    channels: usize,
    o: u16,
) -> (Vec<T>, usize, usize) {
    if o == 1 || data.is_empty() {
        return (data, w, h);
    }
    let (ow, oh) = if (5..=8).contains(&o) { (h, w) } else { (w, h) };
    let mut out = Vec::with_capacity(data.len());
    for y in 0..oh {
        for x in 0..ow {
            // The stored pixel that shows at (x, y).
            let (sx, sy) = match o {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                _ => (w - 1 - y, x),
            };
            let at = (sy * w + sx) * channels;
            out.extend_from_slice(&data[at..at + channels]);
        }
    }
    (out, ow, oh)
}

#[allow(clippy::too_many_arguments)]
fn finish<T: Sample + Into<u32>>(
    data: Vec<T>,
    width: usize,
    height: usize,
    channels: usize,
    orientation: u16,
    colour: &ToLinearSrgb,
    grey: bool,
    mut notes: Vec<ReferenceNote>,
) -> Result<ReferencePicture, ReferenceError> {
    let (data, width, height) = orient(data, width, height, channels, orientation);
    let bars = match letterbox::find_bars(&data, width, height, channels) {
        Letterbox::Empty => return Err(ReferenceError::Empty),
        Letterbox::Bars(b) => b,
    };
    let Bars {
        top,
        bottom,
        left,
        right,
    } = bars;
    let (cw, ch) = (width - left - right, height - top - bottom);
    if cw.min(ch) < MIN_SHORT_SIDE_PX {
        return Err(ReferenceError::TooSmall);
    }
    if bars.any() {
        notes.push(ReferenceNote::BlackBarsRemoved);
    }
    if grey {
        notes.push(ReferenceNote::NoColour);
    }
    notes.sort();
    // The curves, tabulated once for every code of the depth.
    let levels = if std::mem::size_of::<T>() == 1 {
        256
    } else {
        65_536
    };
    let tables = [
        colour.table(0, levels),
        colour.table(1, levels),
        colour.table(2, levels),
    ];
    let stride = width * channels;
    let pixel = |x: usize, y: usize| -> [f64; 3] {
        let at = (y + top) * stride + (x + left) * channels;
        let (r, g, b) = if channels == 1 {
            let v: u32 = data[at].into();
            (v as usize, v as usize, v as usize)
        } else {
            let r: u32 = data[at].into();
            let g: u32 = data[at + 1].into();
            let b: u32 = data[at + 2].into();
            (r as usize, g as usize, b as usize)
        };
        colour.convert([tables[0][r], tables[1][g], tables[2][b]])
    };
    let (sw, sh) = reduced_size(cw, ch, SCENE_LONG_EDGE_PX);
    let (aw, ah) = reduced_size(cw, ch, ANALYSIS_LONG_EDGE_PX);
    let scene = ReducedPicture {
        width: sw,
        height: sh,
        linear: area_average(cw, ch, sw, sh, pixel),
    };
    let analysis = ReducedPicture {
        width: aw,
        height: ah,
        linear: area_average(cw, ch, aw, ah, pixel),
    };
    let digest = digest_of(&analysis);
    Ok(ReferencePicture {
        scene,
        analysis,
        notes,
        grey,
        source: ReferenceSource::File,
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3 x 2 picture whose pixels are numbered 0 to 5 (one channel).
    fn numbered() -> Vec<u8> {
        vec![0, 1, 2, 3, 4, 5]
    }

    #[test]
    fn the_eight_orientations_turn_the_picture_as_exif_says() {
        // The stored picture (rows):  0 1 2 / 3 4 5. Each row below is what the viewer sees, top to bottom, per EXIF 2.32.
        let cases: [(u16, usize, usize, [u8; 6]); 8] = [
            (1, 3, 2, [0, 1, 2, 3, 4, 5]),
            (2, 3, 2, [2, 1, 0, 5, 4, 3]),
            (3, 3, 2, [5, 4, 3, 2, 1, 0]),
            (4, 3, 2, [3, 4, 5, 0, 1, 2]),
            (5, 2, 3, [0, 3, 1, 4, 2, 5]),
            (6, 2, 3, [3, 0, 4, 1, 5, 2]),
            (7, 2, 3, [5, 2, 4, 1, 3, 0]),
            (8, 2, 3, [2, 5, 1, 4, 0, 3]),
        ];
        for (o, w, h, want) in cases {
            let (got, gw, gh) = orient(numbered(), 3, 2, 1, o);
            assert_eq!((gw, gh), (w, h), "orientation {o}");
            assert_eq!(got, want, "orientation {o}");
        }
    }

    #[test]
    fn an_orientation_with_three_channels_moves_whole_pixels() {
        let rgb: Vec<u8> = (0..6u8).flat_map(|p| [p, p + 10, p + 20]).collect();
        let (got, w, h) = orient(rgb, 3, 2, 3, 6);
        assert_eq!((w, h), (2, 3));
        assert_eq!(&got[0..3], &[3, 13, 23]);
        assert_eq!(&got[3..6], &[0, 10, 20]);
    }

    #[test]
    fn exif_orientation_is_read_in_both_byte_orders_and_refuses_the_rest() {
        let little = |o: u16| -> Vec<u8> {
            let mut d = b"II*\0".to_vec();
            d.extend_from_slice(&8u32.to_le_bytes());
            d.extend_from_slice(&2u16.to_le_bytes());
            // A make-like entry, then the orientation.
            d.extend_from_slice(&[0x0F, 0x01, 2, 0, 1, 0, 0, 0, b'S', 0, 0, 0]);
            d.extend_from_slice(&[0x12, 0x01, 3, 0, 1, 0, 0, 0]);
            d.extend_from_slice(&o.to_le_bytes());
            d.extend_from_slice(&[0, 0]);
            d
        };
        let big = |o: u16| -> Vec<u8> {
            let mut d = b"MM\0*".to_vec();
            d.extend_from_slice(&8u32.to_be_bytes());
            d.extend_from_slice(&1u16.to_be_bytes());
            d.extend_from_slice(&[0x01, 0x12, 0, 3, 0, 0, 0, 1]);
            d.extend_from_slice(&o.to_be_bytes());
            d.extend_from_slice(&[0, 0]);
            d
        };
        for o in 1..=8 {
            assert_eq!(exif_orientation(&little(o)), o);
            assert_eq!(exif_orientation(&big(o)), o);
        }
        let mut prefixed = b"Exif\0\0".to_vec();
        prefixed.extend_from_slice(&big(6));
        assert_eq!(exif_orientation(&prefixed), 6);
        // Out of range, truncated, not TIFF.
        assert_eq!(exif_orientation(&little(0)), 1);
        assert_eq!(exif_orientation(&little(9)), 1);
        assert_eq!(exif_orientation(&little(6)[..14]), 1);
        assert_eq!(exif_orientation(b"nope"), 1);
        assert_eq!(exif_orientation(&[]), 1);
    }

    #[test]
    fn the_error_texts_are_the_specs() {
        assert_eq!(
            ReferenceError::Unreadable.to_string(),
            "This file can't be read. It may be damaged. Choose another picture."
        );
        assert_eq!(
            ReferenceError::Unsupported.to_string(),
            "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."
        );
        assert_eq!(
            ReferenceError::TooSmall.to_string(),
            format!(
                "This picture is too small to match. Use one that is at least {MIN_SHORT_SIDE_PX} pixels on its short side."
            )
        );
        assert_eq!(
            ReferenceError::Empty.to_string(),
            "There's no picture inside the black bars. Choose another picture."
        );
        assert_eq!(
            ReferenceError::SamePhoto.to_string(),
            "That's the photo you're grading. Choose another."
        );
        assert_eq!(
            ReferenceNote::BlackBarsRemoved.text(),
            "Black bars were removed from the reference."
        );
        assert_eq!(
            ReferenceNote::ProfileUnreadable.text(),
            "The picture's colour profile couldn't be read, so it is taken as sRGB."
        );
        assert_eq!(
            ReferenceNote::NoColour.text(),
            "This picture has no colour. The photo will be matched in black and white."
        );
        for e in [
            ReferenceError::Unreadable,
            ReferenceError::Unsupported,
            ReferenceError::TooSmall,
            ReferenceError::Empty,
            ReferenceError::SamePhoto,
        ] {
            assert!(!e.to_string().contains('!'));
        }
    }
}
