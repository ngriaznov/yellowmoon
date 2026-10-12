//! The export entry points used by `ym-core` and the CLI (`docs/09-export.md`).
//!
//! * [`write_export`] writes one photo to a file: it plans the name (Skip, Replace, Keep both),
//!   encodes into a temporary file in the destination folder, fsyncs and renames it. On an error or a
//!   cancel the temporary file is deleted. Originals are never opened for writing.
//! * [`encode_to_writer`] / [`encode_to_vec`] do the encoding into any writer (tests, hashes, previews).
//! * [`estimate_bytes`] is the "≈ 120 MB" of the export panel.
//!
//! The same code path serves the app and the CLI, so the same pixels, options and metadata give
//! byte-identical files.

use std::io::{Cursor, Seek, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dng::{self, BaselineExposure, CropRect, DngParams, DngProfile, DngReport};
use crate::error::ExportError;
use crate::exif::{self, ExifInfo, OutputFacts};
use crate::icc;
use crate::jpeg::{self, JpegMetadata};
use crate::naming::{self, NameFields, Placement};
use crate::options::{
    BitDepth, ColorSpace, ExportFormat, ExportOptions, IfExists, Size, TiffCompression,
};
use crate::pixels::{ExportHooks, ImageRef, PixelEncoding};
use crate::png_writer::{self, PngMetadata};
use crate::resize::resize_lanczos3;
use crate::tiff_writer::{self, TiffMetadata};
use crate::xmp::{self, XmpInfo};

/// What an export may say about the photo: camera and exposure data for EXIF, and the processing
/// record for XMP.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Camera, lens, exposure, capture time and location.
    pub exif: ExifInfo,
    /// The creator tool, capture-time dates and the `ym:` properties.
    pub xmp: XmpInfo,
}

/// The inputs that only a DNG needs.
#[derive(Debug, Clone, Copy)]
pub struct DngInputs<'a> {
    /// The exact render in sRGB for the preview and thumbnail, in the raw pixels' orientation
    /// ([`DngParams::preview`]).
    pub preview: ImageRef<'a>,
    /// The TIFF orientation, 1 to 8 ([`dng::compose_orientation`]).
    pub orientation: u16,
    /// The crop other editors can undo (no straighten angle), or `None`.
    pub crop: Option<CropRect>,
    /// How `BaselineExposure` is chosen.
    pub baseline_exposure: BaselineExposure,
    /// The fitted tone curve and look table.
    pub profile: &'a DngProfile,
}

/// One photo to export.
#[derive(Debug, Clone, Copy)]
pub struct ExportRequest<'a> {
    /// The pixels: [`PixelEncoding::DisplayLinear`] or [`PixelEncoding::DisplayEncoded`] in the
    /// export colour space for JPEG, TIFF and PNG; [`PixelEncoding::SceneLinearAwg4`] for a DNG.
    pub pixels: ImageRef<'a>,
    /// The format and its options.
    pub format: ExportFormat,
    /// The colour space of `pixels` and of the file (not for DNG).
    pub color_space: ColorSpace,
    /// The output size (not for DNG).
    pub size: Size,
    /// EXIF and XMP content.
    pub metadata: &'a Metadata,
    /// The "Remove location from exports" preference.
    pub remove_location: bool,
    /// Required for [`ExportFormat::Dng`].
    pub dng: Option<DngInputs<'a>>,
}

impl<'a> ExportRequest<'a> {
    /// A request taking format, colour space and size from the panel's `options`.
    #[must_use]
    pub fn new(pixels: ImageRef<'a>, options: &ExportOptions, metadata: &'a Metadata) -> Self {
        Self {
            pixels,
            format: options.format_spec(),
            color_space: options.color_space,
            size: options.size,
            metadata,
            remove_location: false,
            dng: None,
        }
    }

    /// Sets the "Remove location" preference.
    #[must_use]
    pub fn remove_location(mut self, remove: bool) -> Self {
        self.remove_location = remove;
        self
    }

    /// Adds the DNG-only inputs.
    #[must_use]
    pub fn with_dng(mut self, dng: DngInputs<'a>) -> Self {
        self.dng = Some(dng);
        self
    }

    /// Checks the combination of format, pixels and options.
    pub fn validate(&self) -> Result<(), ExportError> {
        self.pixels.validate()?;
        match self.format {
            ExportFormat::Dng => {
                if self.pixels.encoding != PixelEncoding::SceneLinearAwg4 {
                    return Err(ExportError::invalid(
                        "a DNG needs scene-linear AWG4 pixels (the output of the look, decoded from LogC4)",
                    ));
                }
                if self.dng.is_none() {
                    return Err(ExportError::invalid(
                        "a DNG needs the preview and profile inputs",
                    ));
                }
            }
            _ => {
                if self.pixels.encoding == PixelEncoding::SceneLinearAwg4 {
                    return Err(ExportError::invalid(
                        "JPEG, TIFF and PNG need display pixels (after the display transform), not scene-linear AWG4",
                    ));
                }
                if let Size::LongEdge { px } = self.size
                    && !(crate::options::MIN_LONG_EDGE_PX..=crate::options::MAX_LONG_EDGE_PX)
                        .contains(&px)
                {
                    return Err(ExportError::invalid(format!(
                        "the long edge {px} px is out of range"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The size of the written image.
    #[must_use]
    pub fn output_dimensions(&self) -> (u32, u32) {
        if self.format == ExportFormat::Dng {
            (self.pixels.width, self.pixels.height)
        } else {
            self.size
                .output_dimensions(self.pixels.width, self.pixels.height)
        }
    }
}

/// What [`encode_to_writer`] reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeSummary {
    /// Width of the written image in pixels.
    pub width: u32,
    /// Height of the written image in pixels.
    pub height: u32,
    /// Scaling and clipping details, for a DNG.
    pub dng: Option<DngReport>,
}

/// Where and under which name to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDestination {
    /// The (existing, writable) destination folder; see [`naming::resolve_destination_dir`].
    pub dir: PathBuf,
    /// The file name without extension ([`naming::file_stem`]).
    pub file_stem: String,
    /// What to do if the file exists.
    pub if_exists: IfExists,
    /// The photo's original file, which must never be overwritten.
    pub source_path: Option<PathBuf>,
}

impl ExportDestination {
    /// Resolves the folder (creating it) and the file stem from the panel's options.
    pub fn resolve(
        options: &ExportOptions,
        fields: &NameFields,
        source_path: Option<&Path>,
        home: Option<&Path>,
    ) -> Result<Self, ExportError> {
        let dir =
            naming::resolve_destination_dir(options.destination.as_deref(), source_path, home)?;
        Ok(Self {
            dir,
            file_stem: naming::file_stem(&options.name_template, fields),
            if_exists: options.if_exists,
            source_path: source_path.map(Path::to_path_buf),
        })
    }
}

/// The result of [`write_export`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExportOutcome {
    /// Where the file is (for a skip, the existing file).
    pub placement: Placement,
    /// Bytes written (0 for a skip).
    pub bytes: u64,
    /// Width of the written image (0 for a skip).
    pub width: u32,
    /// Height of the written image (0 for a skip).
    pub height: u32,
    /// DNG scaling and clipping details.
    pub dng: Option<DngReport>,
}

impl ExportOutcome {
    /// The path of the file.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.placement.path()
    }
}

fn software_string(app_version: &str) -> String {
    if app_version.is_empty() {
        "Yellowmoon".to_owned()
    } else {
        format!("Yellowmoon {app_version}")
    }
}

fn mime(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Jpeg { .. } => "image/jpeg",
        ExportFormat::Tiff { .. } => "image/tiff",
        ExportFormat::Png { .. } => "image/png",
        ExportFormat::Dng => "image/dng",
    }
}

/// Encodes the request into `out`. Needs `Seek` because TIFF and DNG patch offsets.
pub fn encode_to_writer<W: Write + Seek>(
    req: &ExportRequest<'_>,
    out: &mut W,
    hooks: &ExportHooks<'_>,
) -> Result<EncodeSummary, ExportError> {
    req.validate()?;
    hooks.check()?;

    let mut xmp_info: XmpInfo = req.metadata.xmp.clone();
    if xmp_info.app_version.is_empty() {
        xmp_info.app_version = env!("CARGO_PKG_VERSION").to_owned();
    }
    if xmp_info.capture_time.is_none() {
        xmp_info.capture_time = req.metadata.exif.capture_time;
    }
    let software = software_string(&xmp_info.app_version);
    let xmp_packet = xmp::build_xmp(&xmp_info, mime(req.format))?;

    if req.format == ExportFormat::Dng {
        let Some(inputs) = req.dng else {
            return Err(ExportError::invalid(
                "a DNG needs the preview and profile inputs",
            ));
        };
        // The Exif pixel dimensions describe the meaningful image: the default crop when there is one.
        let (width, height) = inputs
            .crop
            .map_or((req.pixels.width, req.pixels.height), |c| {
                (c.width, c.height)
            });
        let facts = OutputFacts {
            width,
            height,
            color_space: None,
            ycbcr: false,
        };
        let tables = req
            .metadata
            .exif
            .tables(&facts, &software, req.remove_location);
        let report = dng::write_dng(
            out,
            &DngParams {
                raw: req.pixels,
                preview: inputs.preview,
                orientation: inputs.orientation,
                crop: inputs.crop,
                baseline_exposure: inputs.baseline_exposure,
                profile: inputs.profile,
                exif: &tables,
                xmp: Some(&xmp_packet),
                app_version: &xmp_info.app_version,
            },
            hooks,
        )?;
        return Ok(EncodeSummary {
            width: req.pixels.width,
            height: req.pixels.height,
            dng: Some(report),
        });
    }

    let space = req.color_space.output_space();
    let (out_w, out_h) = req.output_dimensions();
    let resized;
    let (image, encode_hooks) = if (out_w, out_h) == (req.pixels.width, req.pixels.height) {
        (req.pixels, *hooks)
    } else {
        resized = resize_lanczos3(&req.pixels, out_w, out_h, space, &hooks.stage(0.0, 0.3))?;
        (resized.as_ref(), hooks.stage(0.3, 1.0))
    };

    let facts = OutputFacts {
        width: out_w,
        height: out_h,
        color_space: Some(req.color_space),
        ycbcr: matches!(req.format, ExportFormat::Jpeg { .. }),
    };
    let tables = req
        .metadata
        .exif
        .tables(&facts, &software, req.remove_location);
    let icc_profile = icc::profile(req.color_space);
    match req.format {
        ExportFormat::Jpeg { quality } => {
            let app1 = exif::build_exif_app1_payload(&tables);
            jpeg::write_jpeg(
                out,
                &image,
                space,
                quality,
                &JpegMetadata {
                    exif_app1: Some(&app1),
                    xmp: Some(&xmp_packet),
                    icc: Some(icc_profile),
                },
                &encode_hooks,
            )?;
        }
        ExportFormat::Tiff { bits, compression } => {
            tiff_writer::write_tiff(
                out,
                &image,
                req.color_space,
                bits,
                compression,
                &TiffMetadata {
                    exif: &tables,
                    xmp: Some(&xmp_packet),
                    icc: Some(icc_profile),
                },
                &encode_hooks,
            )?;
        }
        ExportFormat::Png { bits } => {
            let blob = exif::build_exif_blob(&tables);
            png_writer::write_png(
                out,
                &image,
                req.color_space,
                bits,
                &PngMetadata {
                    exif: Some(&blob),
                    xmp: Some(&xmp_packet),
                    icc: Some(icc_profile),
                },
                &encode_hooks,
            )?;
        }
        ExportFormat::Dng => unreachable!("handled above"),
    }
    Ok(EncodeSummary {
        width: out_w,
        height: out_h,
        dng: None,
    })
}

/// Encodes the request into memory.
pub fn encode_to_vec(
    req: &ExportRequest<'_>,
    hooks: &ExportHooks<'_>,
) -> Result<(Vec<u8>, EncodeSummary), ExportError> {
    let mut cursor = Cursor::new(Vec::new());
    let summary = encode_to_writer(req, &mut cursor, hooks)?;
    Ok((cursor.into_inner(), summary))
}

/// Writes the photo to `dest`.
///
/// The file is encoded under a temporary name in the destination folder, then renamed according to
/// `dest.if_exists`. With "Skip" and an existing file nothing is encoded. A cancel
/// ([`ExportHooks`]) or an error deletes the partial file and leaves everything else as it was.
pub fn write_export(
    req: &ExportRequest<'_>,
    dest: &ExportDestination,
    hooks: &ExportHooks<'_>,
) -> Result<ExportOutcome, ExportError> {
    req.validate()?;
    let ext = req.format.extension();
    let source = dest.source_path.as_deref();
    if let Some(skipped) =
        naming::plan_output(&dest.dir, &dest.file_stem, ext, dest.if_exists, source)?
    {
        return Ok(ExportOutcome {
            placement: skipped,
            bytes: 0,
            width: 0,
            height: 0,
            dng: None,
        });
    }
    let mut summary = None;
    let (placement, bytes) = naming::write_atomically(
        &dest.dir,
        &dest.file_stem,
        ext,
        dest.if_exists,
        source,
        |writer| {
            summary = Some(encode_to_writer(req, writer, hooks)?);
            Ok(())
        },
    )?;
    let skipped = matches!(placement, Placement::Skipped(_));
    let summary = summary.filter(|_| !skipped);
    Ok(ExportOutcome {
        placement,
        bytes,
        width: summary.map_or(0, |s| s.width),
        height: summary.map_or(0, |s| s.height),
        dng: summary.and_then(|s| s.dng),
    })
}

// ---- size estimate --------------------------------------------------------------------

/// JPEG bits per pixel by quality: `(quality, bits per pixel)`. Measured on the 24 Kodak images
/// (`tests/calibrate_estimate.rs`) with this crate's encoder: 4:4:4 from quality 90, 4:2:0 below, optimised
/// tables. Film scans of 0.4 MP are not camera frames: a full-size, un-denoised 24 MP frame came out
/// 1.5 times larger than this table says at quality 92 (while the table is 25 % too high at quality 50),
/// so it is to be re-measured on Yellowmoon renders (`docs/09`, review notes).
const JPEG_BPP_TABLE: [(f64, f64); 9] = [
    (50.0, 0.99),
    (60.0, 1.13),
    (75.0, 1.48),
    (85.0, 2.00),
    (89.0, 2.35),
    (90.0, 2.95),
    (92.0, 3.28),
    (95.0, 4.25),
    (100.0, 10.2),
];

/// Compressed size over raw size for TIFF and PNG, `[8-bit, 16-bit]`. The 8-bit values are measured on the
/// Kodak images after the dither; the 16-bit values on the same images with a noise of sigma 1.2e-4
/// (about one LSB of 13 bits) added, because a rendered 16-bit frame is not 256 levels scaled up. LZW
/// does poorly on noisy 16-bit samples (the file can be as big as the uncompressed one); Deflate and PNG do better.
const LZW_RATIO: [f64; 2] = [0.70, 0.90];
const ZIP_RATIO: [f64; 2] = [0.63, 0.74];
const PNG_RATIO: [f64; 2] = [0.60, 0.72];

fn interpolate(table: &[(f64, f64)], x: f64) -> f64 {
    let first = table[0];
    let last = table[table.len() - 1];
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    for pair in table.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if x <= b.0 {
            return a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0);
        }
    }
    last.1
}

/// The expected file size in bytes of an export of a `width` x `height` frame: the number behind
/// "≈ 120 MB" in the export panel. It is an estimate from compression ratios measured on film scans
/// (within about 30 % on content like those, and 0.65 to 1.25 times the real size on a noisy camera
/// frame); the metadata and profile add under 4 KB.
#[must_use]
pub fn estimate_bytes(width: u32, height: u32, format: ExportFormat, size: Size) -> u64 {
    let (w, h) = if format == ExportFormat::Dng {
        (width, height)
    } else {
        size.output_dimensions(width, height)
    };
    let pixels = f64::from(w) * f64::from(h);
    let bits_index = |bits: BitDepth| usize::from(bits == BitDepth::Sixteen);
    let bytes = match format {
        ExportFormat::Jpeg { quality } => {
            pixels * interpolate(&JPEG_BPP_TABLE, f64::from(quality.get())) / 8.0
        }
        ExportFormat::Tiff { bits, compression } => {
            let raw = pixels * 3.0 * f64::from(bits.bits()) / 8.0;
            raw * match compression {
                TiffCompression::None => 1.0,
                TiffCompression::Lzw => LZW_RATIO[bits_index(bits)],
                TiffCompression::Zip => ZIP_RATIO[bits_index(bits)],
            }
        }
        ExportFormat::Png { bits } => {
            pixels * 3.0 * f64::from(bits.bits()) / 8.0 * PNG_RATIO[bits_index(bits)]
        }
        ExportFormat::Dng => {
            // 16-bit raw, plus the full-size q92 JPEG preview and a 256 px thumbnail.
            pixels * 6.0
                + pixels * interpolate(&JPEG_BPP_TABLE, f64::from(dng::PREVIEW_QUALITY)) / 8.0
                + 200_000.0
        }
    };
    (bytes + 4096.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::JpegQuality;

    #[test]
    fn estimate_is_monotone_and_sane() {
        let jpeg = |q: u8| {
            estimate_bytes(
                8640,
                5760,
                ExportFormat::Jpeg {
                    quality: JpegQuality::new(q).unwrap(),
                },
                Size::Full,
            )
        };
        let mut last = 0;
        for q in 50..=100 {
            let e = jpeg(q);
            assert!(e >= last, "q{q}");
            last = e;
        }
        // 50 MP at quality 92: tens of megabytes.
        assert!((15_000_000..40_000_000).contains(&jpeg(92)), "{}", jpeg(92));
        let tiff16 = estimate_bytes(
            8640,
            5760,
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
            Size::Full,
        );
        assert!((tiff16 as i64 - 8640 * 5760 * 6).abs() < 10_000);
        let lzw = estimate_bytes(
            8640,
            5760,
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Lzw,
            },
            Size::Full,
        );
        assert!(lzw < tiff16);
        // Smaller long edge, smaller file; DNG ignores the size.
        let small = estimate_bytes(
            8640,
            5760,
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            Size::LongEdge { px: 2000 },
        );
        let full = estimate_bytes(
            8640,
            5760,
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            Size::Full,
        );
        assert!(small * 10 < full);
        let dng_full = estimate_bytes(8640, 5760, ExportFormat::Dng, Size::Full);
        let dng_small = estimate_bytes(8640, 5760, ExportFormat::Dng, Size::LongEdge { px: 2000 });
        assert_eq!(dng_full, dng_small);
        assert!(dng_full > 8640 * 5760 * 6);
    }

    #[test]
    fn interpolation_is_clamped_and_linear() {
        let t = [(0.0, 1.0), (10.0, 3.0)];
        assert_eq!(interpolate(&t, -5.0), 1.0);
        assert_eq!(interpolate(&t, 5.0), 2.0);
        assert_eq!(interpolate(&t, 50.0), 3.0);
    }
}
