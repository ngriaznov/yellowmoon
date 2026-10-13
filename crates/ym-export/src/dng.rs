//! DNG export: LinearRaw, scene-linear AWG4 with the look baked in (`docs/09-export.md` section DNG
//! with the look baked in, `research/notes/dng-export.md`).
//!
//! **This product includes DNG technology under license by Adobe.**
//!
//! Adobe's DNG patent licence requires that notice in the source code and documentation of every
//! product that reads or writes DNG; it also goes in `NOTICE`, the README, the About window and
//! the DNG export help text. The writer is our own, on the `tiff` crate's directory encoder. The
//! `dng` crate (AGPL) and rawler's writer (LGPL) are not used.
//!
//! # File layout
//!
//! ```text
//! IFD0        thumbnail (NewSubFileType 1, 8-bit RGB, at most 256 px) + the DNG profile tags
//!   SubIFDs → raw IFD     NewSubFileType 0, PhotometricInterpretation 34892 (LinearRaw), 3 x uint16,
//!                         Compression 1, BlackLevel 0, WhiteLevel 65535, NoiseReductionApplied 1,
//!                         DefaultCropOrigin/Size, ActiveArea
//!           → preview IFD NewSubFileType 1, full-size JPEG (quality 92, sRGB, 4:4:4) of the exact render
//!   Exif IFD, GPS IFD (optional), XMP (tag 700)
//! ```
//!
//! The directories are written before IFD0 (the header's pointer to IFD0 is patched at the end), so
//! IFD0 is the last directory in the file.
//!
//! # Pixel scaling
//!
//! `stored = round(65535 * clamp(x / x_white, 0, 1))` with `x_white = max(p99.99 of max(R, G, B), floor)`,
//! where the floor is [`BaselineExposure::white_floor`]: 1.0, or `2^-fit_offset_ev` for the fitted profile
//! (16), so that `BaselineExposure` is never negative (Adobe's renderer shoulders the highlights instead of
//! scaling for a negative one). The percentile is taken over the photo's pixels with negative values counted as 0, as the upper
//! edge of the matching bin of a histogram with 2048 bins per octave (a relative resolution of
//! 0.05 %), so at least 99.99 % of the pixels are at or below it and the result does not depend on
//! the thread count. Negative values are clipped to 0 and the clipped fractions are reported in
//! [`DngReport`]. `BaselineExposure` is `log2(x_white)` plus a fitted offset
//! ([`BaselineExposure`]), so Adobe's renderer sees the original scene-linear values. A reader that
//! ignores `BaselineExposure` and the profile shows the stored image darker by the white level (4 EV for
//! the usual 16).
//!
//! # The fitted profile
//!
//! [`fitted::FittedProfile::embedded`] is `data/dng/profile.json`, compiled into the binary: the tone curve
//! and look table fitted to ACES 2 with Adobe's renderer, and the offset `-4`. The app and the CLI both
//! write it, with `BaselineExposure::Auto { fit_offset_ev }`.
//!
//! The 90-degree orientation steps go into the TIFF `Orientation` tag; pixels are not rotated, and
//! the preview and thumbnail must be in the same (unrotated) orientation as the raw pixels.

pub mod fitted;
pub mod profile;

use std::io::{Seek, Write};

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tiff::encoder::{Rational, SRational, TiffEncoder};
use ym_color::encode::OutputSpace;

use crate::error::ExportError;
use crate::exif::{self, Dir, EXIF_IFD_POINTER, ExifTables, GPS_IFD_POINTER, put};
use crate::jpeg::{JpegMetadata, write_jpeg};
use crate::options::JpegQuality;
use crate::pixels::{ExportHooks, ImageRef, PixelEncoding, band_rows, encode_rows_u8};
use crate::resize::resize_lanczos3;

pub use fitted::{FittedProfile, ProfileFile, ProfileFileError};
pub use profile::{
    BaselineExposure, DngProfile, FIT_OFFSET_RANGE_EV, LookTable, compose_orientation,
};

/// The notice Adobe's licence requires.
pub const ADOBE_NOTICE: &str = "This product includes DNG technology under license by Adobe.";

/// One sentence for the DNG export help text of the CLI, the export panel and `docs/09`.
///
/// The writer floors the white level at 16, so the stored 16-bit image is the scene divided by at least 16: Adobe's apps
/// undo that through `BaselineExposure` and the profile, other apps show the stored image as it is.
pub const OTHER_APPS_NOTE: &str = "Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied.";

/// `UniqueCameraModel`: keeps Lightroom from substituting Adobe's Sony profiles.
pub const UNIQUE_CAMERA_MODEL: &str = "Yellowmoon AWG4";

/// `ProfileName`.
pub const PROFILE_NAME: &str = "Yellowmoon";

/// `ProfileCopyright`.
pub const PROFILE_COPYRIGHT: &str = "Yellowmoon contributors";

/// The largest side of the IFD0 thumbnail in pixels.
pub const THUMBNAIL_LONG_EDGE_PX: u32 = 256;

/// The JPEG quality of the preview (`docs/09`).
pub const PREVIEW_QUALITY: u8 = 92;

/// `PreviewColorSpace` value for sRGB (0 unknown, 1 gray gamma 2.2, 2 sRGB, 3 Adobe RGB, 4 ProPhoto RGB).
pub const PREVIEW_COLOR_SPACE_SRGB: u32 = 2;

/// The DNG tags this writer uses (numbers from the DNG specification 1.4/1.6).
pub mod tags {
    /// DNG tag `DNGVersion`.
    pub const DNG_VERSION: u16 = 50706;
    /// DNG tag `DNGBackwardVersion`.
    pub const DNG_BACKWARD_VERSION: u16 = 50707;
    /// DNG tag `UniqueCameraModel`.
    pub const UNIQUE_CAMERA_MODEL: u16 = 50708;
    /// DNG tag `BlackLevel`.
    pub const BLACK_LEVEL: u16 = 50714;
    /// DNG tag `WhiteLevel`.
    pub const WHITE_LEVEL: u16 = 50717;
    /// DNG tag `DefaultCropOrigin`.
    pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
    /// DNG tag `DefaultCropSize`.
    pub const DEFAULT_CROP_SIZE: u16 = 50720;
    /// DNG tag `ColorMatrix1`.
    pub const COLOR_MATRIX_1: u16 = 50721;
    /// DNG tag `AsShotNeutral`.
    pub const AS_SHOT_NEUTRAL: u16 = 50728;
    /// DNG tag `BaselineExposure`.
    pub const BASELINE_EXPOSURE: u16 = 50730;
    /// DNG tag `BaselineSharpness`.
    pub const BASELINE_SHARPNESS: u16 = 50732;
    /// DNG tag `CalibrationIlluminant1`.
    pub const CALIBRATION_ILLUMINANT_1: u16 = 50778;
    /// DNG tag `RawDataUniqueID`.
    pub const RAW_DATA_UNIQUE_ID: u16 = 50781;
    /// DNG tag `ColorimetricReference`.
    pub const COLORIMETRIC_REFERENCE: u16 = 50879;
    /// DNG tag `ActiveArea`.
    pub const ACTIVE_AREA: u16 = 50829;
    /// DNG tag `NoiseReductionApplied`.
    pub const NOISE_REDUCTION_APPLIED: u16 = 50935;
    /// DNG tag `ProfileName`.
    pub const PROFILE_NAME: u16 = 50936;
    /// DNG tag `ProfileToneCurve`.
    pub const PROFILE_TONE_CURVE: u16 = 50940;
    /// DNG tag `ProfileEmbedPolicy`.
    pub const PROFILE_EMBED_POLICY: u16 = 50941;
    /// DNG tag `ProfileCopyright`.
    pub const PROFILE_COPYRIGHT: u16 = 50942;
    /// DNG tag `ForwardMatrix1`.
    pub const FORWARD_MATRIX_1: u16 = 50964;
    /// DNG tag `PreviewApplicationName`.
    pub const PREVIEW_APPLICATION_NAME: u16 = 50966;
    /// DNG tag `PreviewApplicationVersion`.
    pub const PREVIEW_APPLICATION_VERSION: u16 = 50967;
    /// DNG tag `PreviewColorSpace`.
    pub const PREVIEW_COLOR_SPACE: u16 = 50970;
    /// DNG tag `ProfileLookTableDims`.
    pub const PROFILE_LOOK_TABLE_DIMS: u16 = 50981;
    /// DNG tag `ProfileLookTableData`.
    pub const PROFILE_LOOK_TABLE_DATA: u16 = 50982;
    /// DNG tag `ProfileLookTableEncoding`.
    pub const PROFILE_LOOK_TABLE_ENCODING: u16 = 51108;
    /// DNG tag `DefaultBlackRender`.
    pub const DEFAULT_BLACK_RENDER: u16 = 51110;
}

/// The default crop, in raw pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// Everything the DNG writer needs.
#[derive(Debug, Clone, Copy)]
pub struct DngParams<'a> {
    /// The raw pixels: [`PixelEncoding::SceneLinearAwg4`], after the look and curve were applied in
    /// LogC4 and decoded back to linear.
    pub raw: ImageRef<'a>,
    /// The exact render in sRGB ([`PixelEncoding::DisplayLinear`] in Rec.709 primaries), in the same
    /// orientation as `raw` (not rotated), already cropped as the user sees it. Becomes the full-size
    /// JPEG preview and the thumbnail.
    pub preview: ImageRef<'a>,
    /// The TIFF orientation, 1 to 8 ([`compose_orientation`]).
    pub orientation: u16,
    /// The crop for other raw editors to undo; `None` for the whole frame.
    pub crop: Option<CropRect>,
    /// How `BaselineExposure` is chosen.
    pub baseline_exposure: BaselineExposure,
    /// The fitted tone curve and look table.
    pub profile: &'a DngProfile,
    /// Camera, lens and exposure data (descriptive IFD0 tags, Exif IFD, GPS IFD).
    pub exif: &'a ExifTables,
    /// The XMP packet.
    pub xmp: Option<&'a [u8]>,
    /// The application version for the preview tags (`0.0.1`).
    pub app_version: &'a str,
}

/// What the writer measured while writing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DngReport {
    /// The white level used for scaling: `max(p99.99, floor)`, with the floor of
    /// [`BaselineExposure::white_floor`].
    pub x_white: f32,
    /// The `BaselineExposure` written, in EV.
    pub baseline_exposure_ev: f64,
    /// The fraction of pixels with a negative channel (clipped to 0).
    pub negative_clipped_fraction: f64,
    /// The fraction of pixels with a channel above `x_white` (clipped to 65535).
    pub highlight_clipped_fraction: f64,
    /// The largest channel value in the image.
    pub max_value: f32,
}

/// The percentile of the per-pixel maximum used for `x_white`.
const WHITE_PERCENTILE: f64 = 0.9999;

/// Histogram bins: the top 19 bits of a non-negative f32 (8 exponent bits, 11 mantissa bits).
const BINS: usize = 1 << 19;
const BIN_SHIFT: u32 = 12;

fn bin_of(v: f32) -> usize {
    // NaN and negative values count as 0.
    if v.is_nan() || v <= 0.0 {
        0
    } else {
        (v.to_bits() >> BIN_SHIFT) as usize
    }
}

fn bin_upper_edge(bin: usize) -> f32 {
    if bin + 1 >= BINS {
        return f32::MAX;
    }
    let edge = f32::from_bits(((bin + 1) as u32) << BIN_SHIFT);
    if edge.is_finite() { edge } else { f32::MAX }
}

/// `x_white` (at least `white_floor`), the fraction of pixels with a negative channel, and the largest
/// channel value of a scene-linear image.
fn analyse(raw: &ImageRef<'_>, white_floor: f32) -> (f32, f64, f32) {
    let w = raw.width as usize;
    let (hist, negatives, max) = raw
        .rgb
        .par_chunks(w * 3 * 16)
        .fold(
            || (vec![0_u32; BINS], 0_u64, 0.0_f32),
            |(mut hist, mut neg, mut max), chunk| {
                for p in chunk.chunks_exact(3) {
                    let m = p[0].max(p[1]).max(p[2]);
                    hist[bin_of(m)] += 1;
                    if p.iter().any(|&v| v < 0.0) {
                        neg += 1;
                    }
                    if m > max {
                        max = m;
                    }
                }
                (hist, neg, max)
            },
        )
        .reduce(
            || (vec![0_u32; BINS], 0, 0.0),
            |(mut a, na, ma), (b, nb, mb)| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                (a, na + nb, ma.max(mb))
            },
        );
    let pixels = u64::from(raw.width) * u64::from(raw.height);
    let target = ((WHITE_PERCENTILE * pixels as f64).ceil() as u64).clamp(1, pixels);
    let mut cumulative = 0_u64;
    let mut bin = BINS - 1;
    for (b, &count) in hist.iter().enumerate() {
        cumulative += u64::from(count);
        if cumulative >= target {
            bin = b;
            break;
        }
    }
    let x_white = bin_upper_edge(bin).max(white_floor);
    (x_white, negatives as f64 / pixels as f64, max)
}

/// The number of pixels with a channel above `x_white`.
fn count_over(raw: &ImageRef<'_>, x_white: f32) -> u64 {
    raw.rgb
        .par_chunks_exact(3)
        .filter(|p| p[0] > x_white || p[1] > x_white || p[2] > x_white)
        .count() as u64
}

/// The stored 16-bit value of a scene-linear sample.
#[inline]
fn store(x: f32, inv_white: f64) -> u16 {
    // NaN fails both comparisons and becomes 0 through the `as` cast.
    let scaled = (f64::from(x) * inv_white).clamp(0.0, 1.0) * 65535.0;
    scaled.round() as u16
}

/// `x_white` for an image written with `baseline_exposure` (exposed for tests and for callers that want to
/// know it before writing).
pub fn white_level(
    raw: &ImageRef<'_>,
    baseline_exposure: BaselineExposure,
) -> Result<f32, ExportError> {
    raw.validate()?;
    check_baseline_exposure(baseline_exposure)?;
    Ok(analyse(raw, baseline_exposure.white_floor()).0)
}

fn check_baseline_exposure(b: BaselineExposure) -> Result<(), ExportError> {
    match b {
        BaselineExposure::Fixed(v) if !v.is_finite() => {
            Err(ExportError::invalid("the baseline exposure isn't a number"))
        }
        BaselineExposure::Auto { fit_offset_ev }
            if !fit_offset_ev.is_finite() || !FIT_OFFSET_RANGE_EV.contains(&fit_offset_ev) =>
        {
            Err(ExportError::invalid(format!(
                "the baseline exposure offset {fit_offset_ev} EV is outside {} to {}",
                FIT_OFFSET_RANGE_EV.start(),
                FIT_OFFSET_RANGE_EV.end()
            )))
        }
        _ => Ok(()),
    }
}

fn rational(v: f64) -> Rational {
    Rational {
        n: (v * 1e6).round() as u32,
        d: 1_000_000,
    }
}

fn srational(v: f64) -> SRational {
    SRational {
        n: (v * 1e6).round() as i32,
        d: 1_000_000,
    }
}

fn validate_params(p: &DngParams<'_>) -> Result<(), ExportError> {
    p.raw.validate()?;
    p.preview.validate()?;
    if p.raw.encoding != PixelEncoding::SceneLinearAwg4 {
        return Err(ExportError::invalid(
            "the DNG's raw pixels must be scene-linear AWG4",
        ));
    }
    if p.preview.encoding != PixelEncoding::DisplayLinear {
        return Err(ExportError::invalid(
            "the DNG preview must be display-linear sRGB pixels",
        ));
    }
    if !(1..=8).contains(&p.orientation) {
        return Err(ExportError::invalid(format!(
            "the orientation must be 1 to 8, not {}",
            p.orientation
        )));
    }
    if let Some(c) = p.crop {
        let fits = c.width > 0
            && c.height > 0
            && c.x.checked_add(c.width).is_some_and(|r| r <= p.raw.width)
            && c.y.checked_add(c.height).is_some_and(|b| b <= p.raw.height);
        if !fits {
            return Err(ExportError::invalid("the crop is outside the image"));
        }
    }
    p.profile.validate().map_err(ExportError::invalid)?;
    check_baseline_exposure(p.baseline_exposure)?;
    Ok(())
}

fn too_large() -> ExportError {
    ExportError::Unsupported("A DNG can't be larger than 4 GB.".to_owned())
}

/// Writes a DNG.
pub fn write_dng<W: Write + Seek>(
    out: W,
    p: &DngParams<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<DngReport, ExportError> {
    validate_params(p)?;
    let (w, h) = (p.raw.width as usize, p.raw.height as usize);
    let row_bytes = w * 6;
    if (row_bytes as u64) * (h as u64) > 3_800_000_000 {
        return Err(ExportError::Unsupported(
            "This image is too large for a DNG (4 GB limit of the TIFF structure).".to_owned(),
        ));
    }

    // Pass 1: the white level and the clipped fractions.
    let (x_white, negative_fraction, max_value) =
        analyse(&p.raw, p.baseline_exposure.white_floor());
    let pixels = (w * h) as f64;
    let over_count = count_over(&p.raw, x_white);
    let baseline_ev = p.baseline_exposure.resolve_ev(x_white);
    hooks.check()?;
    hooks.report(0.1);

    let mut tiff = TiffEncoder::new(out)?;

    // ---- raw IFD (SubIFD 1) -------------------------------------------------------------
    let inv_white = 1.0 / f64::from(x_white);
    let rows_per_strip = ((1 << 20) / row_bytes).clamp(1, h);
    let (raw_offset, unique_id) = {
        let mut dir = tiff.extra_directory()?;
        let mut hasher = Sha256::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut sizes: Vec<u32> = Vec::new();
        let batch = band_rows(p.raw.width).max(rows_per_strip) / rows_per_strip * rows_per_strip;
        let mut y = 0;
        while y < h {
            hooks.check()?;
            let rows = batch.min(h - y);
            let src = &p.raw.rgb[y * w * 3..(y + rows) * w * 3];
            let mut samples = vec![0_u16; src.len()];
            samples
                .par_chunks_mut(w * 3)
                .zip(src.par_chunks(w * 3))
                .for_each(|(dst, s)| {
                    for (d, &v) in dst.iter_mut().zip(s) {
                        *d = store(v, inv_white);
                    }
                });
            for strip in samples.chunks(rows_per_strip * w * 3) {
                let bytes: Vec<u8> = strip.iter().flat_map(|v| v.to_ne_bytes()).collect();
                hasher.update(&bytes);
                let offset = dir.write_data(&bytes[..])?;
                offsets.push(u32::try_from(offset).map_err(|_| too_large())?);
                sizes.push(bytes.len() as u32);
            }
            y += rows;
            hooks.report(0.1 + 0.6 * y as f32 / h as f32);
        }
        let digest = hasher.finalize();
        write_raw_ifd_tags(&mut dir, p, &offsets, &sizes, rows_per_strip)?;
        let id: [u8; 16] = digest[..16]
            .try_into()
            .expect("a SHA-256 digest has 32 bytes");
        (exif::finish_with_offset(dir)?, id)
    };
    hooks.check()?;

    // ---- preview IFD (SubIFD 2): the exact render as a JPEG ----------------------------------
    let mut preview_jpeg = Vec::new();
    write_jpeg(
        &mut preview_jpeg,
        &p.preview,
        OutputSpace::Srgb,
        JpegQuality::new(PREVIEW_QUALITY)?,
        &JpegMetadata {
            exif_app1: None,
            xmp: None,
            icc: Some(crate::icc::profile(crate::options::ColorSpace::Srgb)),
        },
        &hooks.stage(0.7, 0.9),
    )?;
    let preview_offset = {
        let mut dir = tiff.extra_directory()?;
        let strip_offset = dir.write_data(&preview_jpeg[..])?;
        put(&mut dir, 254, 1_u32)?;
        put(&mut dir, 256, p.preview.width)?;
        put(&mut dir, 257, p.preview.height)?;
        put(&mut dir, 258, &[8_u16; 3][..])?;
        put(&mut dir, 259, 7_u16)?; // JPEG
        put(&mut dir, 262, 6_u16)?; // YCbCr
        put(
            &mut dir,
            273,
            u32::try_from(strip_offset).map_err(|_| too_large())?,
        )?;
        put(&mut dir, 277, 3_u16)?;
        put(&mut dir, 278, p.preview.height)?;
        put(&mut dir, 279, preview_jpeg.len() as u32)?;
        put(&mut dir, 284, 1_u16)?;
        put(&mut dir, 530, &[1_u16, 1][..])?; // YCbCrSubSampling 4:4:4
        put(&mut dir, 531, 1_u16)?; // centred
        put(&mut dir, tags::PREVIEW_APPLICATION_NAME, "Yellowmoon")?;
        put(&mut dir, tags::PREVIEW_APPLICATION_VERSION, p.app_version)?;
        put(
            &mut dir,
            tags::PREVIEW_COLOR_SPACE,
            PREVIEW_COLOR_SPACE_SRGB,
        )?;
        exif::finish_with_offset(dir)?
    };
    drop(preview_jpeg);
    hooks.check()?;

    // ---- Exif and GPS IFDs -------------------------------------------------------------------
    let exif_offset = if p.exif.exif.is_empty() {
        None
    } else {
        let mut dir = tiff.extra_directory()?;
        let mut entries = p.exif.exif.clone();
        entries.sort_by_key(|e| e.tag);
        exif::write_directory(&mut dir, &entries)?;
        Some(exif::finish_with_offset(dir)?)
    };
    let gps_offset = if p.exif.gps.is_empty() {
        None
    } else {
        let mut dir = tiff.extra_directory()?;
        let mut entries = p.exif.gps.clone();
        entries.sort_by_key(|e| e.tag);
        exif::write_directory(&mut dir, &entries)?;
        Some(exif::finish_with_offset(dir)?)
    };

    // ---- thumbnail and IFD0 ------------------------------------------------------------------
    let (thumb_w, thumb_h, thumb) = thumbnail(&p.preview, hooks)?;
    hooks.report(0.95);
    {
        let mut dir = tiff.image_directory()?;
        let strip = dir.write_data(&thumb[..])?;
        put(&mut dir, 254, 1_u32)?; // reduced-resolution image
        put(&mut dir, 256, thumb_w)?;
        put(&mut dir, 257, thumb_h)?;
        put(&mut dir, 258, &[8_u16; 3][..])?;
        put(&mut dir, 259, 1_u16)?;
        put(&mut dir, 262, 2_u16)?;
        put(
            &mut dir,
            273,
            u32::try_from(strip).map_err(|_| too_large())?,
        )?;
        put(&mut dir, 274, p.orientation)?;
        put(&mut dir, 277, 3_u16)?;
        put(&mut dir, 278, thumb_h)?;
        put(&mut dir, 279, thumb.len() as u32)?;
        put(&mut dir, 282, Rational { n: 300, d: 1 })?;
        put(&mut dir, 283, Rational { n: 300, d: 1 })?;
        put(&mut dir, 284, 1_u16)?;
        put(&mut dir, 296, 2_u16)?;
        put(&mut dir, 330, &[raw_offset, preview_offset][..])?; // SubIFDs
        put(&mut dir, 339, &[1_u16; 3][..])?;
        let mut descriptive = p.exif.ifd0.clone();
        descriptive.sort_by_key(|e| e.tag);
        exif::write_directory(&mut dir, &descriptive)?;
        if let Some(xmp) = p.xmp {
            put(&mut dir, 700, xmp)?;
        }
        if let Some(offset) = exif_offset {
            put(&mut dir, EXIF_IFD_POINTER, offset)?;
        }
        if let Some(offset) = gps_offset {
            put(&mut dir, GPS_IFD_POINTER, offset)?;
        }
        write_profile_tags(&mut dir, p, baseline_ev, &unique_id)?;
        exif::finish(dir)?;
    }
    hooks.report(1.0);

    Ok(DngReport {
        x_white,
        baseline_exposure_ev: baseline_ev,
        negative_clipped_fraction: negative_fraction,
        highlight_clipped_fraction: over_count as f64 / pixels,
        max_value,
    })
}

/// The tags of the raw (LinearRaw) IFD, after its strips were written.
fn write_raw_ifd_tags<W: Write + Seek>(
    dir: &mut Dir<'_, W>,
    p: &DngParams<'_>,
    offsets: &[u32],
    sizes: &[u32],
    rows_per_strip: usize,
) -> Result<(), ExportError> {
    let (w, h) = (p.raw.width, p.raw.height);
    let crop = p.crop.unwrap_or(CropRect {
        x: 0,
        y: 0,
        width: w,
        height: h,
    });
    put(dir, 254, 0_u32)?; // the full-resolution image
    put(dir, 256, w)?;
    put(dir, 257, h)?;
    put(dir, 258, &[16_u16; 3][..])?;
    put(dir, 259, 1_u16)?; // uncompressed strips
    put(dir, 262, 34892_u16)?; // LinearRaw
    put(dir, 273, offsets)?;
    put(dir, 277, 3_u16)?;
    put(dir, 278, rows_per_strip as u32)?;
    put(dir, 279, sizes)?;
    put(dir, 284, 1_u16)?;
    put(dir, 339, &[1_u16; 3][..])?;
    put(dir, tags::BLACK_LEVEL, &[0_u16; 3][..])?;
    put(dir, tags::WHITE_LEVEL, &[65535_u32; 3][..])?;
    put(dir, tags::DEFAULT_CROP_ORIGIN, &[crop.x, crop.y][..])?;
    put(dir, tags::DEFAULT_CROP_SIZE, &[crop.width, crop.height][..])?;
    put(dir, tags::ACTIVE_AREA, &[0_u32, 0, h, w][..])?;
    put(dir, tags::NOISE_REDUCTION_APPLIED, Rational { n: 1, d: 1 })?;
    Ok(())
}

/// The IFD0 tags that describe the profile.
fn write_profile_tags<W: Write + Seek>(
    dir: &mut Dir<'_, W>,
    p: &DngParams<'_>,
    baseline_ev: f64,
    unique_id: &[u8; 16],
) -> Result<(), ExportError> {
    let one = || Rational { n: 1, d: 1 };
    put(dir, tags::DNG_VERSION, &[1_u8, 4, 0, 0][..])?;
    put(dir, tags::DNG_BACKWARD_VERSION, &[1_u8, 1, 0, 0][..])?;
    put(dir, tags::UNIQUE_CAMERA_MODEL, UNIQUE_CAMERA_MODEL)?;
    put(
        dir,
        tags::COLOR_MATRIX_1,
        &profile::srational_matrix(&profile::color_matrix_1())[..],
    )?;
    put(dir, tags::AS_SHOT_NEUTRAL, &[one(), one(), one()][..])?;
    put(dir, tags::BASELINE_EXPOSURE, srational(baseline_ev))?;
    put(dir, tags::BASELINE_SHARPNESS, rational(1.0))?;
    put(dir, tags::CALIBRATION_ILLUMINANT_1, 21_u16)?; // D65
    put(dir, tags::RAW_DATA_UNIQUE_ID, &unique_id[..])?;
    put(dir, tags::COLORIMETRIC_REFERENCE, 0_u16)?;
    put(dir, tags::PROFILE_NAME, PROFILE_NAME)?;
    let curve: Vec<f32> = p.profile.tone_curve.iter().flatten().copied().collect();
    put(dir, tags::PROFILE_TONE_CURVE, &curve[..])?;
    put(dir, tags::PROFILE_EMBED_POLICY, 3_u32)?; // no restrictions on embedding
    put(dir, tags::PROFILE_COPYRIGHT, PROFILE_COPYRIGHT)?;
    put(
        dir,
        tags::FORWARD_MATRIX_1,
        &profile::srational_matrix(&profile::forward_matrix_1())[..],
    )?;
    put(dir, tags::PREVIEW_COLOR_SPACE, PREVIEW_COLOR_SPACE_SRGB)?;
    put(
        dir,
        tags::PROFILE_LOOK_TABLE_DIMS,
        &p.profile.look_table.dims[..],
    )?;
    put(
        dir,
        tags::PROFILE_LOOK_TABLE_DATA,
        &p.profile.look_table.data[..],
    )?;
    put(
        dir,
        tags::PROFILE_LOOK_TABLE_ENCODING,
        p.profile.look_table.encoding,
    )?;
    put(dir, tags::DEFAULT_BLACK_RENDER, 1_u32)?; // no further black subtraction
    Ok(())
}

/// The IFD0 thumbnail: the preview shrunk to at most 256 px on the long edge, as 8-bit sRGB.
fn thumbnail(
    preview: &ImageRef<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(u32, u32, Vec<u8>), ExportError> {
    let size = crate::options::Size::LongEdge {
        px: THUMBNAIL_LONG_EDGE_PX,
    };
    let (tw, th) = size.output_dimensions(preview.width, preview.height);
    let mut bytes = Vec::new();
    if (tw, th) == (preview.width, preview.height) {
        encode_rows_u8(
            preview,
            OutputSpace::Srgb,
            0,
            preview.height as usize,
            &mut bytes,
        );
    } else {
        let small = resize_lanczos3(preview, tw, th, OutputSpace::Srgb, &hooks.stage(0.9, 0.95))?;
        encode_rows_u8(
            &small.as_ref(),
            OutputSpace::Srgb,
            0,
            th as usize,
            &mut bytes,
        );
    }
    Ok((tw, th, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_image(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> Vec<f32> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        v
    }

    #[test]
    fn white_level_is_one_for_dim_images_and_the_percentile_for_bright_ones() {
        let (w, h) = (100_u32, 100_u32);
        let none = BaselineExposure::default();
        let dim = raw_image(w, h, |x, _| [x as f32 / 200.0, 0.1, 0.0]);
        let img = ImageRef::new(w, h, &dim, PixelEncoding::SceneLinearAwg4).unwrap();
        assert_eq!(white_level(&img, none).unwrap(), 1.0);

        // 10,000 pixels: p99.99 is the 9,999th smallest, so one pixel may be above x_white.
        let mut bright = raw_image(w, h, |_, _| [2.0, 1.0, 0.5]);
        bright[0] = 50.0; // one outlier
        let img = ImageRef::new(w, h, &bright, PixelEncoding::SceneLinearAwg4).unwrap();
        let x = white_level(&img, none).unwrap();
        assert!(
            (2.0..=2.0 * (1.0 + 1.0 / 1024.0)).contains(&x),
            "outlier ignored: {x}"
        );

        // Two outliers out of 10,000 are 0.02 %: the percentile reaches the outlier's bin.
        bright[3] = 50.0;
        let img = ImageRef::new(w, h, &bright, PixelEncoding::SceneLinearAwg4).unwrap();
        assert!(white_level(&img, none).unwrap() >= 50.0);
    }

    #[test]
    fn the_white_level_is_floored_where_the_fitted_offset_cancels() {
        let (w, h) = (100_u32, 100_u32);
        let fitted = BaselineExposure::Auto {
            fit_offset_ev: -4.0,
        };
        // A dim photo: x_white is the floor, 16, and BaselineExposure log2(16) - 4 = 0.
        let dim = raw_image(w, h, |x, _| [x as f32 / 200.0, 0.1, 0.0]);
        let img = ImageRef::new(w, h, &dim, PixelEncoding::SceneLinearAwg4).unwrap();
        let x = white_level(&img, fitted).unwrap();
        assert_eq!(x, 16.0);
        assert_eq!(fitted.resolve_ev(x), 0.0);
        // The same photo with a fixed exposure keeps the old floor of 1.0.
        assert_eq!(
            white_level(&img, BaselineExposure::Fixed(0.0)).unwrap(),
            1.0
        );
        // A photo with highlights well above the floor keeps its own percentile and gets a positive exposure.
        let bright = raw_image(w, h, |_, y| [if y < 10 { 40.0 } else { 0.5 }, 0.2, 0.1]);
        let img = ImageRef::new(w, h, &bright, PixelEncoding::SceneLinearAwg4).unwrap();
        let x = white_level(&img, fitted).unwrap();
        assert!(
            (40.0..=40.0 * (1.0 + 1.0 / 1024.0)).contains(&x),
            "above the floor: {x}"
        );
        assert!(fitted.resolve_ev(x) > 1.3);
        // An offset the writer cannot use is refused.
        for bad in [f64::NAN, f64::INFINITY, -30.0, 9.0] {
            let b = BaselineExposure::Auto { fit_offset_ev: bad };
            assert!(white_level(&img, b).is_err(), "{bad}");
        }
    }

    #[test]
    fn storing_scales_clamps_and_rounds() {
        let inv = 1.0 / 4.0;
        assert_eq!(store(0.0, inv), 0);
        assert_eq!(store(4.0, inv), 65535);
        assert_eq!(store(8.0, inv), 65535);
        assert_eq!(store(-1.0, inv), 0);
        assert_eq!(store(2.0, inv), 32768); // 32767.5 rounds up
        assert_eq!(store(f32::NAN, inv), 0);
        assert_eq!(store(f32::INFINITY, inv), 65535);
        assert_eq!(store(f32::NEG_INFINITY, inv), 0);
    }

    #[test]
    fn bins_are_ordered_and_bracket_their_values() {
        for v in [1e-6_f32, 0.18, 1.0, 1.5, 100.0, 4000.0] {
            let b = bin_of(v);
            assert!(bin_upper_edge(b) >= v, "{v}");
            assert!(
                bin_upper_edge(b) <= v * (1.0 + 1.0 / 1024.0),
                "{v}: {}",
                bin_upper_edge(b)
            );
            assert!(bin_of(v * 1.01) >= b);
        }
        assert_eq!(bin_of(-5.0), 0);
        assert_eq!(bin_of(f32::NAN), 0);
        assert_eq!(bin_upper_edge(bin_of(f32::INFINITY)), f32::MAX);
    }

    #[test]
    fn analysis_does_not_depend_on_chunking() {
        let (w, h) = (37_u32, 91_u32);
        let data = raw_image(w, h, |x, y| {
            [(x * 3 + y) as f32 / 100.0, -((x % 5) as f32) * 0.01, 0.3]
        });
        let img = ImageRef::new(w, h, &data, PixelEncoding::SceneLinearAwg4).unwrap();
        let a = analyse(&img, 1.0);
        let b = analyse(&img, 1.0);
        assert_eq!(a, b);
        // Negative samples: x % 5 != 0 in 4 of 5 columns.
        assert!((a.1 - 0.8).abs() < 0.02, "{}", a.1);
    }
}
