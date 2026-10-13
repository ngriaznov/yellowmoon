//! The export entry points used by `ym-core` and the CLI (`docs/09-export.md`).
//!
//! * [`write_export`] writes one photo to a file: it plans the name (Skip, Replace, Keep both),
//!   encodes into a temporary file in the destination folder, fsyncs and renames it. On an error or a
//!   cancel the temporary file is deleted. Originals are never opened for writing.
//! * [`encode_to_writer`] / [`encode_to_vec`] do the encoding into any writer (tests, hashes, previews).
//! * [`estimate_bytes`](crate::estimate::estimate_bytes) and
//!   [`estimate_bytes_with_detail`](crate::estimate::estimate_bytes_with_detail) (in [`crate::estimate`]) give the
//!   "≈ 120 MB" of the export panel.
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
use crate::options::{ColorSpace, ExportFormat, ExportOptions, IfExists, Size};
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
