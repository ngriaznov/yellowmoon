//! `RawMetadata` and the fast metadata probe that never decodes pixels (`docs/04-raw-decoding.md` sections Output and Performance targets).
//!
//! Entry points (re-exports from `lib.rs` are not allowed, so the full paths are):
//!
//! - [`probe`] (`ym_raw::metadata::probe`): metadata of a file on disk.
//! - [`probe_bytes`]: the same for bytes in memory.
//! - [`probe_source`]: the same for any [`ByteSource`].
//! - [`read_preview`]: the embedded preview JPEG of a file (about 1616 x 1080 pixels).
//!
//! A probe reads the TIFF header, the directories, the Sony maker note and the SR2 block with
//! positional reads, and never touches the pixel data, so it costs well under 20 ms for a 100 MB
//! file. Which tags it reads is documented on the `exif`, `makernote` and `sr2` modules and on
//! the fields of [`RawMetadata`].
//!
//! # Choosing the raw image
//!
//! The probe walks every directory it can reach (IFD0, the "next IFD" chain, `SubIFDs`, EXIF,
//! GPS) and keeps those that describe raw data: a CFA photometric interpretation (32803) with one
//! sample per pixel, or YCbCr (6) with 3 samples and at least 12 bits for compression 7, or the
//! four-sample layout of ARQ. Of these the one with the most pixels wins, the first one on a tie.
//! A file whose winner is not recognisably Sony (no `SONY` make, no Sony maker note, no Sony tags
//! in the raw directory) is [`RawError::NotArw`]. See
//! [`classify`] for how the compression, bit depth and photometric values map to a
//! [`DecodeKind`].
//!
//! # Black level, white level and white balance
//!
//! Each comes from the first of these that exists: the decrypted SR2 sub-IFD (`sr2` module), the
//! plain tags 0x7310 / 0x7313 in the raw IFD (newer bodies repeat them there), and finally a
//! documented default (black 512 for cRAW and 128 shifted by the bit depth otherwise, white
//! `2^bits - 1`; white balance has no default). [`RawMetadata::black_source`] and friends say
//! which one was used. Levels are on the scale of the decoded samples: when the data is 12 bits
//! wide and the SR2 levels are on Sony's 14-bit scale they are shifted down (see
//! `sr2::fit_levels`), and cRAW levels stay on the 14-bit scale of the decoder's curve output.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::RawError;
use crate::exif::{
    self, CaptureTime, ExposureProgram, FlashInfo, GpsInfo, IsoSource, MeteringMode,
    SceneCaptureType,
};
use crate::layout::{
    CfaPattern, ChromaSubsampling, DecodeKind, Layout, MAX_DIMENSION, MAX_PIXELS, Rect,
};
use crate::makernote::{self, CreativeStyle, DroMode, SceneMode};
use crate::quirks;
use crate::sr2;
use crate::tiff::{
    ByteSource, Entry, FileSource, Ifd, IfdKind, IfdNode, MAX_ARRAY_ELEMENTS, MemSource, Tiff,
    field_type, tag,
};

/// `SonyRawFileType`: 0 = uncompressed 14 bit, 1 = uncompressed 12 bit, 2 = cRAW, 3 = lossless,
/// 4 = lossless 2, 6 = Compressed RAW 2.
pub const TAG_SONY_RAW_FILE_TYPE: u16 = 0x7000;
/// `SonyToneCurve`: four 16-bit knots (cRAW).
pub const TAG_TONE_CURVE: u16 = 0x7010;
/// `VignettingCorrParams`.
pub const TAG_VIGNETTING_PARAMS: u16 = 0x7032;
/// `ChromaticAberrationCorrParams`.
pub const TAG_CA_PARAMS: u16 = 0x7035;
/// `DistortionCorrParams`.
pub const TAG_DISTORTION_PARAMS: u16 = 0x7037;
/// `SonyRawImageSize`: width and height of the actual image inside the readout.
pub const TAG_SONY_RAW_IMAGE_SIZE: u16 = 0x7038;
/// Black level repeated in the raw IFD of newer bodies (four values, R G G B).
pub const TAG_RAW_BLACK_LEVEL: u16 = 0x7310;
/// White balance repeated in the raw IFD of newer bodies (four values, R G G B).
pub const TAG_RAW_WB_LEVELS: u16 = 0x7313;
/// DNG `WhiteLevel`, which many Sony bodies repeat in the raw IFD (the ILCE-9 writes 16383 there
/// while its SR2 white level is 15360).
pub const TAG_RAW_WHITE_LEVEL: u16 = 0xC61D;
/// `SonyCropTopLeft`: left and top of the camera's crop.
pub const TAG_CROP_TOP_LEFT: u16 = 0x74C7;
/// `SonyCropSize`: width and height of the camera's crop.
pub const TAG_CROP_SIZE: u16 = 0x74C8;

/// TIFF compression value of ARW 6.0 "Compressed RAW 2".
pub const COMPRESSION_ARW6: u16 = 32766;
/// TIFF compression value Sony uses for ARW 2 (packed 12-bit and cRAW).
pub const COMPRESSION_ARW2: u16 = 32767;
/// TIFF compression value of lossless JPEG (Lossless L, M and S).
pub const COMPRESSION_LJPEG: u16 = 7;
/// TIFF compression value for no compression.
pub const COMPRESSION_NONE: u16 = 1;
/// TIFF photometric interpretation of a colour filter array.
pub const PHOTOMETRIC_CFA: u16 = 32803;
/// TIFF photometric interpretation YCbCr.
pub const PHOTOMETRIC_YCBCR: u16 = 6;
/// TIFF photometric interpretation of Sony ARQ (four samples per pixel).
pub const PHOTOMETRIC_ARQ: u16 = 32892;

/// The most strips or tiles accepted.
const MAX_CHUNKS: usize = 1 << 20;

/// How much of a candidate preview is read to find out how it is coded and how big it is (the
/// frame header sits at byte 556 in every Sony preview seen).
const JPEG_PROBE_BYTES: u64 = 8192;

/// The shortest long edge, in pixels, of a JPEG worth showing as the first preview.
pub const PREVIEW_MIN_LONG_EDGE_PX: u32 = 1024;

/// The largest embedded JPEG preview accepted (64 MiB; real previews are a few hundred kilobytes
/// to a few megabytes).
pub const MAX_PREVIEW_BYTES: u64 = 64 << 20;

/// A JPEG stored in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JpegRef {
    /// File offset of the first byte of the JPEG (`FF D8`).
    pub offset: u64,
    /// Length of the JPEG in bytes.
    pub length: u64,
}

/// The raw table of one lens-correction tag: the tag's signed 16-bit values exactly as stored.
///
/// Sony writes a count followed by the knots: tag 0x7032 (vignetting) and 0x7037 (distortion)
/// hold 17 values (the count `n` and `n` knots, zero padded to 16), tag 0x7035 (chromatic
/// aberration) holds 33 (the count `2n` and `2n` knots: `n` for red, then `n` for blue). `n` is
/// 16 for a full-frame lens and 11 for an APS-C lens or an APS-C crop mode, on one physical radial
/// grid. Checked on real files of six bodies (eight files) against the cameras' own corrected JPEGs and the raw
/// data (`docs/04` section Lens tags, `docs/05` section R9). The maker note and the SR2 sub-IFD
/// hold other copies that are not read here. The maths lives in `docs/05-raw-processing.md`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LensTable {
    /// The stored values: `raw[0]` is the count, `raw[1..]` the knots.
    pub raw: Vec<i16>,
}

impl LensTable {
    /// The count stored in front of the knots.
    pub fn count(&self) -> Option<i16> {
        self.raw.first().copied()
    }

    /// The knots, without the count.
    pub fn knots(&self) -> &[i16] {
        self.raw.get(1..).unwrap_or(&[])
    }
}

/// The lens-correction tables embedded by the camera. Each part is `None` when the tag is
/// missing (older bodies, adapted lenses).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LensTags {
    /// Vignetting (tag 0x7032).
    pub vignetting: Option<LensTable>,
    /// Chromatic aberration (tag 0x7035).
    pub chromatic_aberration: Option<LensTable>,
    /// Distortion (tag 0x7037).
    pub distortion: Option<LensTable>,
}

impl LensTags {
    /// True when none of the three tables is present.
    pub fn is_empty(&self) -> bool {
        self.vignetting.is_none()
            && self.chromatic_aberration.is_none()
            && self.distortion.is_none()
    }
}

/// Where the levels in [`RawMetadata`] came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelSource {
    /// The decrypted SR2 sub-IFD.
    Sr2,
    /// Plain tags in the raw IFD.
    RawIfd,
    /// No tag was found; a documented default was used.
    Default,
}

impl LevelSource {
    /// `"sr2"`, `"raw_ifd"` or `"default"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sr2 => "sr2",
            Self::RawIfd => "raw_ifd",
            Self::Default => "default",
        }
    }
}

/// Where the raw samples are stored in the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RawStorage {
    /// One or more strips of whole rows.
    Strips {
        /// File offset of each strip.
        offsets: Vec<u64>,
        /// Byte length of each strip.
        byte_counts: Vec<u64>,
        /// Rows in every strip except possibly the last.
        rows_per_strip: u32,
    },
    /// A grid of tiles, left to right then top to bottom.
    Tiles {
        /// Tile width in samples.
        tile_width: u32,
        /// Tile height in rows.
        tile_length: u32,
        /// File offset of each tile.
        offsets: Vec<u64>,
        /// Byte length of each tile.
        byte_counts: Vec<u64>,
    },
}

impl RawStorage {
    /// The offsets and byte counts of the strips or tiles.
    pub fn chunks(&self) -> (&[u64], &[u64]) {
        match self {
            Self::Strips {
                offsets,
                byte_counts,
                ..
            }
            | Self::Tiles {
                offsets,
                byte_counts,
                ..
            } => (offsets, byte_counts),
        }
    }

    /// Total number of stored bytes.
    pub fn total_bytes(&self) -> u64 {
        self.chunks().1.iter().sum()
    }
}

/// Everything the probe learns about a file without decoding pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawMetadata {
    /// `Make` from IFD0 (`SONY`, or `Hasselblad` for the rebadged bodies).
    pub make: String,
    /// `Model` from IFD0 (`ILCE-7RM4`).
    pub model: String,
    /// Sony's model ID from the maker note (tag 0xB001). When it disagrees with `model`, the
    /// model ID wins (`docs/03-camera-profiles.md`).
    pub model_id: Option<u16>,
    /// ARW version bytes from the maker note (tag 0xB000), for example `[3, 2, 0, 0]` for 2.2.
    pub arw_version: Option<[u8; 4]>,
    /// `Software` (firmware string) from IFD0.
    pub software: Option<String>,
    /// ISO, see [`exif::resolve_iso`] for the order of preference.
    pub iso: Option<u32>,
    /// Which tag [`RawMetadata::iso`] came from.
    pub iso_source: Option<IsoSource>,
    /// Exposure time in seconds.
    pub exposure_time_s: Option<f64>,
    /// F-number.
    pub f_number: Option<f64>,
    /// Exposure compensation in EV.
    pub exposure_comp_ev: Option<f64>,
    /// Focal length in millimetres.
    pub focal_length_mm: Option<f64>,
    /// Focal length in 35 mm equivalent, millimetres.
    pub focal_length_35mm: Option<u16>,
    /// Lens model string.
    pub lens_model: Option<String>,
    /// `DateTimeOriginal` with sub-second and offset when the file has them.
    pub capture_time: Option<CaptureTime>,
    /// EXIF `BrightnessValue` (tag 0x9203): the APEX brightness the camera measured, in EV. `None`
    /// when the tag is missing, not a signed rational or out of range.
    pub brightness_value: Option<f64>,
    /// EXIF `MeteringMode` (tag 0x9207): how the camera measured the light.
    pub metering_mode: Option<MeteringMode>,
    /// EXIF `Flash` (tag 0x9209): whether the flash fired (bit 0), and the code as written.
    pub flash: Option<FlashInfo>,
    /// EXIF `ExposureProgram` (tag 0x8822): the shooting mode the camera was set to.
    pub exposure_program: Option<ExposureProgram>,
    /// EXIF `SceneCaptureType` (tag 0xA406): the kind of scene the camera assumed.
    pub scene_capture_type: Option<SceneCaptureType>,
    /// EXIF `ColorSpace` (tag 0xA001) as written: 1 for sRGB, 0xFFFF (uncalibrated) for the Adobe RGB
    /// mode. `None` when the tag is missing.
    pub color_space: Option<u16>,
    /// Sony Dynamic Range Optimizer mode (maker note tag 0xB025). `None` when the tag is missing.
    /// Tag 0xB04F, another DRO tag with the codes Off, Standard and Plus that some compact cameras
    /// write, is not read.
    pub dro_mode: Option<DroMode>,
    /// Sony Creative Style: the name in maker note tag 0xB020, or, when that is absent or empty,
    /// the code in the enciphered 0x9416 table (not on the ILCE-7M5). Creative Look codes read as
    /// `CreativeStyle::Other`.
    pub creative_style: Option<CreativeStyle>,
    /// Sony scene mode (maker note tag 0xB023).
    pub scene_mode: Option<SceneMode>,
    /// EXIF orientation, 1 to 8 (1 = upright).
    pub orientation: u16,
    /// Width of the raw image in samples (the whole readout, including masked margins).
    pub width: u32,
    /// Height of the raw image in rows.
    pub height: u32,
    /// `BitsPerSample` of the raw image as the tag says. cRAW files say 12 although they store
    /// one byte per pixel and decode to a 14-bit scale; use [`RawMetadata::decode`] to know what
    /// the data is.
    pub bits_per_sample: u16,
    /// `SamplesPerPixel` of the raw image (1 for a mosaic, 3 for YCbCr).
    pub samples_per_pixel: u16,
    /// TIFF `Compression` of the raw image.
    pub compression: u16,
    /// TIFF `PhotometricInterpretation` of the raw image.
    pub photometric: u16,
    /// `SonyRawFileType` (tag 0x7000) when present.
    pub raw_file_type: Option<u16>,
    /// Which decoder reads this file.
    pub decode: DecodeKind,
    /// Where the raw data is in the file.
    pub storage: RawStorage,
    /// The colour filter array pattern (RGGB when the file does not say, and for YCbCr files).
    pub cfa: CfaPattern,
    /// The valid image area inside the readout (from `SonyRawImageSize`), when stated.
    pub active_area: Option<Rect>,
    /// The camera's crop inside the readout, when stated.
    pub crop: Option<Rect>,
    /// Black level per CFA position (raster order), on the scale of the decoded samples.
    pub black: [f32; 4],
    /// White level per CFA position, on the scale of the decoded samples.
    pub white: [f32; 4],
    /// Where the black level came from.
    pub black_source: LevelSource,
    /// Where the white level came from.
    pub white_source: LevelSource,
    /// As-shot white balance gains per CFA position, as stored (green is about 1024 on older
    /// bodies), when the file has them.
    pub wb_as_shot: Option<[f32; 4]>,
    /// Where the white balance came from.
    pub wb_source: Option<LevelSource>,
    /// The lens-correction tables, when the file has any.
    pub lens_corrections: Option<LensTags>,
    /// The cRAW tone-curve knots as stored (tag 0x7010). Decoders use `(v >> 2) & 0xFFF`.
    pub tone_curve: Option<[u16; 4]>,
    /// The embedded JPEG to show first (see `find_preview`): the preview of about 1616 x 1080
    /// pixels when the file has one, not the thumbnail and not the full-size JPEG of ARW 4.0
    /// files.
    pub preview: Option<JpegRef>,
    /// GPS directory contents when the file has a GPS directory.
    pub gps: Option<GpsInfo>,
}

/// A readable name for the ARW version bytes of maker note tag 0xB000, for example `ARW 2.3.2`
/// for `[3, 3, 2, 0]`. The first byte is the major version minus one before ARW 4 (`2` is ARW
/// 1.0, `3` is ARW 2.x); the others follow ExifTool's table. `None` for `[0, ...]` and `[1, ...]`
/// (JPEG and SR2, which are not ARW).
pub fn arw_version_name(bytes: [u8; 4]) -> Option<String> {
    let [major, minor, patch, build] = bytes;
    let (major_name, minor_name) = match major {
        0 | 1 => return None,
        2 => (1, 0),
        3 => (2, minor),
        n => (n, minor),
    };
    let mut name = format!("ARW {major_name}.{minor_name}");
    if patch != 0 || build != 0 {
        name.push_str(&format!(".{patch}"));
    }
    if build != 0 {
        name.push_str(&format!(".{build}"));
    }
    Some(name)
}

impl RawMetadata {
    /// The ARW version as text (see [`arw_version_name`]), when the maker note states it.
    pub fn arw_version_name(&self) -> Option<String> {
        self.arw_version.and_then(arw_version_name)
    }

    /// The sample layout implied by the metadata.
    pub fn layout(&self) -> Layout {
        match self.decode {
            DecodeKind::LosslessMS { subsampling } => Layout::Ycbcr { subsampling },
            _ => Layout::Bayer {
                pattern: self.cfa,
                active_area: self.active_area,
                crop: self.crop,
            },
        }
    }
}

/// Reads the metadata of the file at `path` without decoding pixels.
///
/// Fails with [`RawError::NotArw`] for files that are not Sony ARW (including SRF, SR2 and
/// non-TIFF files), [`RawError::Arq`] for pixel-shift ARQ, [`RawError::Corrupt`] for truncated or
/// malformed files and [`RawError::Io`] when the file cannot be read.
pub fn probe(path: impl AsRef<Path>) -> Result<RawMetadata, RawError> {
    let source = FileSource::open(path.as_ref())?;
    probe_source(&source)
}

/// Reads the metadata of an ARW held in memory.
pub fn probe_bytes(bytes: &[u8]) -> Result<RawMetadata, RawError> {
    probe_source(&MemSource(bytes))
}

/// Reads the metadata of an ARW from any [`ByteSource`].
pub fn probe_source(source: &dyn ByteSource) -> Result<RawMetadata, RawError> {
    let (tiff, first_ifd) = Tiff::open(source)?;
    let nodes = tiff.walk(first_ifd)?;
    let exif = exif::read(&tiff, &nodes);
    let make = exif.make.clone().unwrap_or_default();
    let model = exif.model.clone().unwrap_or_default();
    let is_sony_make = make.to_ascii_uppercase().starts_with("SONY");
    let maker = exif
        .maker_note
        .as_ref()
        .and_then(|entry| makernote::parse(&tiff, entry, is_sony_family(&make, &model)));

    // Files that look like TIFF but are not ARW.
    if nodes.iter().any(|n| n.ifd.get(tag::DNG_VERSION).is_some()) {
        return Err(RawError::NotArw);
    }
    if maker.as_ref().and_then(|m| m.file_format) == Some([1, 0, 0, 0]) {
        return Err(RawError::NotArw); // SR2
    }
    if matches!(model.as_str(), "DSC-F828" | "DSC-V3" | "DSC-R1") {
        return Err(RawError::NotArw); // SRF and SR2 cameras
    }

    let candidates: Vec<RawIfd> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| raw_ifd_params(&tiff, node, index))
        .collect();
    let Some(best) = candidates
        .iter()
        .fold(None::<&RawIfd>, |best, c| match best {
            Some(b) if b.pixels() >= c.pixels() => Some(b),
            _ => Some(c),
        })
    else {
        return Err(no_raw_ifd_error(
            &nodes,
            &make,
            &model,
            is_sony_make,
            maker.is_some(),
        ));
    };
    let raw_node = &nodes[best.node];
    let raw_ifd = &raw_node.ifd;

    let sony_tags_in_raw_ifd = raw_ifd
        .entries
        .iter()
        .any(|e| (0x7000..0x7100).contains(&e.tag));
    let looks_like_sony = is_sony_make || maker.is_some() || sony_tags_in_raw_ifd;
    if !looks_like_sony {
        return Err(RawError::NotArw);
    }
    if best.samples == 4 || best.photometric == PHOTOMETRIC_ARQ {
        return Err(RawError::Arq);
    }

    if best.width > MAX_DIMENSION
        || best.height > MAX_DIMENSION
        || u64::from(best.width) * u64::from(best.height) > MAX_PIXELS
    {
        return Err(RawError::corrupt(format!(
            "image size {} x {} is above the 120 megapixel limit",
            best.width, best.height
        )));
    }

    let storage = read_storage(&tiff, raw_ifd, best)?;
    let raw_file_type = raw_ifd
        .get(TAG_SONY_RAW_FILE_TYPE)
        .and_then(|e| e.inline_uint())
        .and_then(|v| u16::try_from(v).ok());
    let ycbcr_subsampling = ycbcr_subsampling(&tiff, raw_ifd, &storage, best);
    let decode = classify(&RawParams {
        compression: best.compression,
        bits: best.bits,
        photometric: best.photometric,
        samples: best.samples,
        width: best.width,
        height: best.height,
        total_bytes: storage.total_bytes(),
        raw_file_type,
        ycbcr_subsampling,
    });

    if let Some(needed) = min_stored_bytes(decode, best.compression, best.pixels())
        && storage.total_bytes() < needed
    {
        return Err(RawError::corrupt(format!(
            "the raw data holds {} bytes but a {} x {} image needs {needed}",
            storage.total_bytes(),
            best.width,
            best.height
        )));
    }

    let cfa = cfa_pattern(&tiff, raw_ifd);
    let (active_area, crop) = areas(&tiff, raw_ifd, best.width, best.height);

    // Levels: SR2 first, then the raw IFD, then defaults.
    let sr2_data = nodes
        .iter()
        .find(|n| matches!(n.kind, IfdKind::Root(0)))
        .and_then(|n| sr2::read(&tiff, &n.ifd));
    let raw_ifd_black = raw_ifd
        .get(TAG_RAW_BLACK_LEVEL)
        .and_then(|e| four_u16(&tiff, e));
    let raw_ifd_white = raw_ifd
        .get(TAG_RAW_WHITE_LEVEL)
        .and_then(|e| sr2::white_levels(&tiff, e));
    let raw_ifd_wb = raw_ifd
        .get(TAG_RAW_WB_LEVELS)
        .and_then(|e| four_u16(&tiff, e));
    let levels = resolve_levels(
        decode,
        best.bits,
        cfa,
        sr2_data.as_ref(),
        RawIfdLevels {
            black: raw_ifd_black,
            white: raw_ifd_white,
            wb: raw_ifd_wb,
        },
    )?;

    let lens_corrections = lens_tags(&tiff, raw_ifd).or_else(|| {
        nodes
            .iter()
            .filter(|n| n.ifd.offset != raw_ifd.offset)
            .find_map(|n| lens_tags(&tiff, &n.ifd))
    });
    let tone_curve = raw_ifd.get(TAG_TONE_CURVE).and_then(|e| four_u16(&tiff, e));

    let sony_iso = maker.as_ref().and_then(|m| m.iso).map(|(iso, _tag)| iso);
    let (iso, iso_source) = exif::resolve_iso(sony_iso, &exif.iso).unzip();

    let mut meta = RawMetadata {
        make,
        model,
        model_id: maker.as_ref().and_then(|m| m.model_id),
        arw_version: maker.as_ref().and_then(|m| m.file_format),
        software: exif.software,
        iso,
        iso_source,
        exposure_time_s: exif.exposure_time_s,
        f_number: exif.f_number,
        exposure_comp_ev: exif.exposure_comp_ev,
        focal_length_mm: exif.focal_length_mm,
        focal_length_35mm: exif.focal_length_35mm,
        lens_model: exif.lens_model,
        capture_time: exif.capture_time,
        brightness_value: exif.brightness_value,
        metering_mode: exif.metering_mode,
        flash: exif.flash,
        exposure_program: exif.exposure_program,
        scene_capture_type: exif.scene_capture_type,
        color_space: exif.color_space,
        dro_mode: maker.as_ref().and_then(|m| m.dro_mode),
        creative_style: maker.as_ref().and_then(|m| m.creative_style.clone()),
        scene_mode: maker.as_ref().and_then(|m| m.scene_mode),
        orientation: exif.orientation.unwrap_or(1),
        width: best.width,
        height: best.height,
        bits_per_sample: best.bits,
        samples_per_pixel: best.samples,
        compression: best.compression,
        photometric: best.photometric,
        raw_file_type,
        decode,
        storage,
        cfa,
        active_area,
        crop,
        black: levels.black,
        white: levels.white,
        black_source: levels.black_source,
        white_source: levels.white_source,
        wb_as_shot: levels.wb,
        wb_source: levels.wb_source,
        lens_corrections,
        tone_curve,
        preview: find_preview(&tiff, &nodes, maker.as_ref()),
        gps: exif.gps,
    };
    quirks::apply(&mut meta);
    Ok(meta)
}

/// True for a `Make`/`Model` pair whose maker note is Sony's even without a header: `SONY`, and
/// the Hasselblad Lunar, Lusso, Stellar and HV models that are rebadged Sony bodies.
fn is_sony_family(make: &str, model: &str) -> bool {
    let make = make.trim().to_ascii_uppercase();
    let model = model.trim().to_ascii_uppercase();
    make.starts_with("SONY")
        || (make.starts_with("HASSELBLAD")
            && ["HV", "STELLAR", "LUSSO", "LUNAR"]
                .iter()
                .any(|prefix| model.starts_with(prefix)))
}

/// Reads the embedded preview JPEG of the file at `path` (see `RawMetadata::preview`).
///
/// Returns `Ok(None)` when the file has no usable preview. The bytes start with `FF D8 FF`.
pub fn read_preview(path: impl AsRef<Path>) -> Result<Option<Vec<u8>>, RawError> {
    let source = FileSource::open(path.as_ref())?;
    let meta = probe_source(&source)?;
    read_preview_from(&source, &meta)
}

/// Reads the preview described by `meta.preview` from `source`.
pub fn read_preview_from(
    source: &dyn ByteSource,
    meta: &RawMetadata,
) -> Result<Option<Vec<u8>>, RawError> {
    let Some(jpeg) = meta.preview else {
        return Ok(None);
    };
    // Previews are far larger than a tag value, so they bypass the walker's value limit; they are
    // bounded by `MAX_PREVIEW_BYTES` here (and by the file size, which `find_preview` checked).
    if jpeg.length > MAX_PREVIEW_BYTES {
        return Err(RawError::corrupt("the preview is too large"));
    }
    let length =
        usize::try_from(jpeg.length).map_err(|_| RawError::corrupt("the preview is too large"))?;
    let mut bytes = vec![0u8; length];
    source.read_at(jpeg.offset, &mut bytes)?;
    Ok(Some(bytes))
}

/// The parameters of one directory that could hold the raw image.
#[derive(Clone, Copy, Debug)]
struct RawIfd {
    node: usize,
    width: u32,
    height: u32,
    bits: u16,
    samples: u16,
    compression: u16,
    photometric: u16,
}

impl RawIfd {
    fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Reads the first value of an unsigned integer tag.
fn first_uint(tiff: &Tiff, ifd: &Ifd, tag: u16) -> Option<u64> {
    tiff.uints(ifd.get(tag)?, 1).ok()?.first().copied()
}

/// Returns the description of a directory when it can hold Sony raw data.
fn raw_ifd_params(tiff: &Tiff, node: &IfdNode, index: usize) -> Option<RawIfd> {
    if !matches!(node.kind, IfdKind::Root(_) | IfdKind::Sub) {
        return None;
    }
    let ifd = &node.ifd;
    // Any compression value qualifies: a Sony raw file with a compression we have never seen is
    // reported as `DecodeKind::Unknown` (and "unsupported"), not as "not an ARW".
    let compression = u16::try_from(first_uint(tiff, ifd, tag::COMPRESSION)?).ok()?;
    let photometric = u16::try_from(first_uint(tiff, ifd, tag::PHOTOMETRIC)?).ok()?;
    let samples = first_uint(tiff, ifd, tag::SAMPLES_PER_PIXEL)
        .and_then(|v| u16::try_from(v).ok())
        .unwrap_or(1);
    let bits = first_uint(tiff, ifd, tag::BITS_PER_SAMPLE)
        .and_then(|v| u16::try_from(v).ok())
        .unwrap_or(0);
    let width = u32::try_from(first_uint(tiff, ifd, tag::IMAGE_WIDTH)?).ok()?;
    let height = u32::try_from(first_uint(tiff, ifd, tag::IMAGE_LENGTH)?).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let new_subfile_type = first_uint(tiff, ifd, tag::NEW_SUBFILE_TYPE).unwrap_or(0);
    let is_reduced_resolution = new_subfile_type & 1 == 1;

    let is_mosaic = matches!(photometric, PHOTOMETRIC_CFA) && samples == 1;
    let is_arq = samples == 4 && compression == COMPRESSION_NONE;
    let is_ycbcr = compression == COMPRESSION_LJPEG
        && photometric == PHOTOMETRIC_YCBCR
        && samples == 3
        && (bits == 0 || bits >= 12);
    if !(is_mosaic || is_ycbcr || is_arq || photometric == PHOTOMETRIC_ARQ) {
        return None;
    }
    // The camera's RGB preview is a reduced-resolution YCbCr directory of 8 bits (LibRaw skips
    // exactly `bps == 8 && NewSubFileType == 1`); without a BitsPerSample tag it cannot be told
    // from data here, so it is skipped too. A reduced-resolution directory that states 12 bits or
    // more is data: nothing says the real Lossless M and S files leave their raw directory
    // unflagged, and rejecting it would make them unreadable. A full-size YCbCr directory
    // without a BitsPerSample tag is accepted (LibRaw reads the precision from the JPEG stream
    // then).
    if is_reduced_resolution && !is_mosaic && bits < 12 {
        return None;
    }
    Some(RawIfd {
        node: index,
        width,
        height,
        bits,
        samples,
        compression,
        photometric,
    })
}

/// Chooses the error for a file in which no raw directory was found.
fn no_raw_ifd_error(
    nodes: &[IfdNode],
    make: &str,
    model: &str,
    is_sony_make: bool,
    has_sony_maker_note: bool,
) -> RawError {
    if model == "DSLR-A100" {
        // The A100's ARW 1 stores the raw data behind a SubIFDs pointer that is not a directory.
        return RawError::Unsupported {
            camera: model.to_owned(),
            variant: DecodeKind::Unknown {
                compression: COMPRESSION_ARW2,
                bits: 12,
            },
        };
    }
    // A Sony file whose SubIFDs pointer leads nowhere has lost its raw directory (a truncated
    // file); anything else without a raw directory is not an ARW.
    let has_sub_ifds_tag = nodes
        .iter()
        .any(|n| n.ifd.get(tag::SUB_IFDS).is_some() && matches!(n.kind, IfdKind::Root(_)));
    let has_readable_sub_ifd = nodes.iter().any(|n| n.kind == IfdKind::Sub);
    if (is_sony_make || has_sony_maker_note) && has_sub_ifds_tag && !has_readable_sub_ifd {
        return RawError::corrupt("the raw image directory is missing or cut off");
    }
    let _ = make;
    RawError::NotArw
}

/// The inputs of [`classify`].
#[derive(Clone, Copy, Debug)]
pub struct RawParams {
    /// TIFF `Compression`.
    pub compression: u16,
    /// TIFF `BitsPerSample` (0 when missing).
    pub bits: u16,
    /// TIFF `PhotometricInterpretation`.
    pub photometric: u16,
    /// TIFF `SamplesPerPixel`.
    pub samples: u16,
    /// Image width in samples.
    pub width: u32,
    /// Image height in rows.
    pub height: u32,
    /// Total bytes of all strips or tiles.
    pub total_bytes: u64,
    /// `SonyRawFileType` when present.
    pub raw_file_type: Option<u16>,
    /// Chroma subsampling of a YCbCr image when known.
    pub ycbcr_subsampling: Option<ChromaSubsampling>,
}

/// Maps the TIFF description of the raw image to a [`DecodeKind`] (`docs/04` section Decoders by
/// TIFF compression).
///
/// | Compression | Bits / layout | Result |
/// |---|---|---|
/// | 32766 | any | `Arw6` |
/// | 7 | YCbCr (6) with 3 samples | `LosslessMS` |
/// | 7 | CFA, 1 sample | `LosslessL` |
/// | 1 | 12 bits | `Uncompressed12` (16-bit words) |
/// | 1 | 14 or 16 bits | `Uncompressed14` |
/// | 1 | other bit depth | `SonyRawFileType` 0 / 1 decides, otherwise `Unknown` |
/// | 32767 | one byte per pixel in the file (or 8 bits) | `Craw` |
/// | 32767 | two bytes per sample in the file | `Uncompressed14` |
/// | 32767 | 12 bits and `bytes * 8 = width * height * 12` | `Uncompressed12` (packed) |
/// | 32767 | 12 bits and a byte count that fits none of the above (A200 to A390 Huffman ARW 1) | `Arw1` |
/// | 32767 | anything else | `Unknown` |
pub fn classify(p: &RawParams) -> DecodeKind {
    let unknown = DecodeKind::Unknown {
        compression: p.compression,
        bits: p.bits,
    };
    let pixels = u64::from(p.width) * u64::from(p.height);
    match p.compression {
        COMPRESSION_ARW6 => DecodeKind::Arw6,
        COMPRESSION_LJPEG => {
            if p.photometric == PHOTOMETRIC_YCBCR && p.samples == 3 {
                DecodeKind::LosslessMS {
                    subsampling: p.ycbcr_subsampling.unwrap_or(ChromaSubsampling::Yuv420),
                }
            } else if p.samples == 1 {
                DecodeKind::LosslessL
            } else {
                unknown
            }
        }
        COMPRESSION_NONE => match p.bits {
            12 => DecodeKind::Uncompressed12,
            14 | 16 => DecodeKind::Uncompressed14,
            _ => match p.raw_file_type {
                Some(0) => DecodeKind::Uncompressed14,
                Some(1) => DecodeKind::Uncompressed12,
                _ => unknown,
            },
        },
        COMPRESSION_ARW2 => {
            // cRAW stores one byte per pixel. The cameras write BitsPerSample 12 for it (every
            // cRAW file from the A500 to the A7 III in metadata-extractor-images does), so the
            // byte count decides, as in LibRaw; 8 bits (rawler's per-camera value) counts too.
            if p.bits == 8 || p.total_bytes == pixels {
                DecodeKind::Craw
            } else if p.total_bytes == pixels * 2 {
                DecodeKind::Uncompressed14
            } else if p.bits == 12 && p.total_bytes * 8 == pixels * 12 {
                DecodeKind::Uncompressed12
            } else if p.bits == 12 {
                // 12 bits and a byte count that fits no other layout: the Huffman-coded ARW 1 of
                // the A200 to A390 (LibRaw's `sony_arw_load_raw`, which it chooses the same way;
                // the real A300 file has 0.98 bytes per pixel).
                DecodeKind::Arw1
            } else {
                unknown
            }
        }
        _ => unknown,
    }
}

/// The fewest bytes the raw data of an uncompressed layout can occupy, or `None` for compressed
/// layouts whose size is not fixed by the image size.
fn min_stored_bytes(decode: DecodeKind, compression: u16, pixels: u64) -> Option<u64> {
    match decode {
        DecodeKind::Uncompressed14 => Some(pixels * 2),
        DecodeKind::Uncompressed12 if compression == COMPRESSION_NONE => Some(pixels * 2),
        DecodeKind::Uncompressed12 => Some(pixels * 12 / 8),
        DecodeKind::Craw => Some(pixels),
        _ => None,
    }
}

/// Reads the strip or tile layout of the raw image and checks it against the file size.
fn read_storage(tiff: &Tiff, ifd: &Ifd, raw: &RawIfd) -> Result<RawStorage, RawError> {
    let file_len = tiff.source_len();
    let read = |tag_ref: u16| -> Result<Vec<u64>, RawError> {
        let entry = ifd
            .get(tag_ref)
            .ok_or_else(|| RawError::corrupt("the raw image has no data location"))?;
        if entry.count as usize > MAX_CHUNKS {
            return Err(RawError::corrupt("the raw image claims too many strips"));
        }
        tiff.uints(entry, MAX_CHUNKS)
    };

    let storage = if ifd.get(tag::TILE_OFFSETS).is_some() {
        let tile_width = first_uint(tiff, ifd, tag::TILE_WIDTH)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| RawError::corrupt("the raw image has no tile width"))?;
        let tile_length = first_uint(tiff, ifd, tag::TILE_LENGTH)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| RawError::corrupt("the raw image has no tile length"))?;
        let offsets = read(tag::TILE_OFFSETS)?;
        let byte_counts = read(tag::TILE_BYTE_COUNTS)?;
        let across = u64::from(raw.width).div_ceil(u64::from(tile_width));
        let down = u64::from(raw.height).div_ceil(u64::from(tile_length));
        if offsets.len() as u64 != across * down {
            return Err(RawError::corrupt(format!(
                "the raw image has {} tiles but its size needs {}",
                offsets.len(),
                across * down
            )));
        }
        RawStorage::Tiles {
            tile_width,
            tile_length,
            offsets,
            byte_counts,
        }
    } else {
        let offsets = read(tag::STRIP_OFFSETS)?;
        let byte_counts = read(tag::STRIP_BYTE_COUNTS)?;
        let rows_per_strip = first_uint(tiff, ifd, tag::ROWS_PER_STRIP)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .map_or(raw.height, |v| v.min(raw.height));
        RawStorage::Strips {
            offsets,
            byte_counts,
            rows_per_strip,
        }
    };

    let (offsets, counts) = storage.chunks();
    if offsets.is_empty() || offsets.len() != counts.len() {
        return Err(RawError::corrupt(
            "the raw image's strip offsets and byte counts do not match",
        ));
    }
    for (offset, count) in offsets.iter().zip(counts) {
        let end = offset
            .checked_add(*count)
            .ok_or_else(|| RawError::corrupt("a raw data offset overflows"))?;
        if end > file_len {
            return Err(RawError::corrupt(
                "the raw data runs past the end of the file (the file is cut short)",
            ));
        }
    }
    Ok(storage)
}

/// Chroma subsampling of a YCbCr raw image: the `YCbCrSubSampling` tag, else the sampling
/// factors in the first tile's JPEG header.
fn ycbcr_subsampling(
    tiff: &Tiff,
    ifd: &Ifd,
    storage: &RawStorage,
    raw: &RawIfd,
) -> Option<ChromaSubsampling> {
    if raw.photometric != PHOTOMETRIC_YCBCR || raw.samples != 3 {
        return None;
    }
    let from_pair = |h: u64, v: u64| match (h, v) {
        (1, 1) => Some(ChromaSubsampling::Yuv444),
        (2, 1) => Some(ChromaSubsampling::Yuv422),
        (2, 2) => Some(ChromaSubsampling::Yuv420),
        _ => None,
    };
    if let Some(entry) = ifd.get(tag::YCBCR_SUBSAMPLING)
        && let Ok(values) = tiff.uints(entry, 2)
        && let [h, v] = values[..]
        && let Some(found) = from_pair(h, v)
    {
        return Some(found);
    }
    let (offsets, counts) = storage.chunks();
    let (offset, count) = (*offsets.first()?, *counts.first()?);
    let head = tiff.read_vec(offset, count.min(2048)).ok()?;
    let (h, v) = jpeg_first_component_sampling(&head)?;
    from_pair(u64::from(h), u64::from(v))
}

/// The first start-of-frame marker of a JPEG header and the position of its length field: the
/// marker byte (`0xC0` baseline, `0xC2` progressive, `0xC3` lossless, ...) tells the coding
/// process. `None` when the bytes are not a JPEG or end before a frame header.
pub fn jpeg_frame_header(bytes: &[u8]) -> Option<(u8, usize)> {
    if bytes.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut pos = 2usize;
    loop {
        // Find the next marker.
        while *bytes.get(pos)? != 0xFF {
            pos += 1;
        }
        while *bytes.get(pos)? == 0xFF {
            pos += 1;
        }
        let marker = *bytes.get(pos)?;
        pos += 1;
        match marker {
            0x00 | 0x01 | 0xD0..=0xD8 => continue,
            0xDA | 0xD9 => return None,
            _ => {}
        }
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            return Some((marker, pos));
        }
        let length = usize::from(u16::from_be_bytes([*bytes.get(pos)?, *bytes.get(pos + 1)?]));
        pos = pos.checked_add(length)?;
    }
}

/// Horizontal and vertical sampling factors of the first component in a JPEG header.
pub fn jpeg_first_component_sampling(bytes: &[u8]) -> Option<(u8, u8)> {
    let (_, pos) = jpeg_frame_header(bytes)?;
    // length, precision, height(2), width(2), count, then id, sampling, table per component.
    let components = *bytes.get(pos + 7)?;
    if components == 0 {
        return None;
    }
    let hv = *bytes.get(pos + 9)?;
    Some((hv >> 4, hv & 0x0F))
}

/// True for the start-of-frame markers of lossless JPEG processes (SOF3, 7, 11 and 15). Sony
/// stores Lossless M and S data, and an extra "RGB preview", this way; no viewer can show it.
fn is_lossless_jpeg_marker(marker: u8) -> bool {
    matches!(marker, 0xC3 | 0xC7 | 0xCB | 0xCF)
}

/// The CFA pattern from `CFARepeatPatternDim` and `CFAPattern`; RGGB when absent or not a 2x2
/// Bayer quad (every Sony body is RGGB).
fn cfa_pattern(tiff: &Tiff, ifd: &Ifd) -> CfaPattern {
    let parsed = (|| {
        let dim = tiff.uints(ifd.get(tag::CFA_REPEAT_PATTERN_DIM)?, 2).ok()?;
        if dim != [2, 2] {
            return None;
        }
        let codes = tiff.uints(ifd.get(tag::CFA_PATTERN)?, 4).ok()?;
        let codes: [u8; 4] = codes
            .iter()
            .map(|c| u8::try_from(*c).ok())
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()?;
        CfaPattern::from_tiff_codes(codes)
    })();
    parsed.unwrap_or(CfaPattern::RGGB)
}

/// Four unsigned 16-bit values from a tag.
fn four_u16(tiff: &Tiff, entry: &Entry) -> Option<[u16; 4]> {
    if entry.count < 4 {
        return None;
    }
    let values = tiff.ints(entry, 4).ok()?;
    let values: Vec<u16> = values
        .into_iter()
        .map(u16::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    values.try_into().ok()
}

/// Two unsigned integers from a tag.
fn two_u32(tiff: &Tiff, ifd: &Ifd, tag_ref: u16) -> Option<[u32; 2]> {
    let entry = ifd.get(tag_ref)?;
    if entry.count < 2 {
        return None;
    }
    let values = tiff.floats(entry, 2).ok()?;
    let to_u32 =
        |v: f64| (v.is_finite() && v >= 0.0 && v <= f64::from(u32::MAX)).then_some(v as u32);
    Some([to_u32(*values.first()?)?, to_u32(*values.get(1)?)?])
}

/// The active area (`SonyRawImageSize`) and the crop (`SonyCropTopLeft` + `SonyCropSize`, else
/// the DNG default crop) inside a `width` x `height` readout. Rectangles that do not fit are
/// ignored.
fn areas(tiff: &Tiff, ifd: &Ifd, width: u32, height: u32) -> (Option<Rect>, Option<Rect>) {
    let active = two_u32(tiff, ifd, TAG_SONY_RAW_IMAGE_SIZE)
        .map(|[w, h]| Rect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        })
        .filter(|r| r.fits_in(width, height));
    let crop = (|| {
        let (origin, size) = (
            two_u32(tiff, ifd, TAG_CROP_TOP_LEFT)
                .or_else(|| two_u32(tiff, ifd, tag::DEFAULT_CROP_ORIGIN))?,
            two_u32(tiff, ifd, TAG_CROP_SIZE)
                .or_else(|| two_u32(tiff, ifd, tag::DEFAULT_CROP_SIZE))?,
        );
        Some(Rect {
            x: origin[0],
            y: origin[1],
            width: size[0],
            height: size[1],
        })
    })()
    .filter(|r| r.fits_in(width, height));
    (active, crop)
}

/// The lens-correction tables of a directory.
fn lens_tags(tiff: &Tiff, ifd: &Ifd) -> Option<LensTags> {
    let table = |tag_ref: u16, max: usize| -> Option<LensTable> {
        let entry = ifd.get(tag_ref)?;
        if !matches!(entry.ty, field_type::SSHORT | field_type::SHORT) || entry.count == 0 {
            return None;
        }
        let values = tiff.ints(entry, max.min(MAX_ARRAY_ELEMENTS)).ok()?;
        Some(LensTable {
            raw: values.into_iter().map(|v| v as i16).collect(),
        })
    };
    let tags = LensTags {
        vignetting: table(TAG_VIGNETTING_PARAMS, 64),
        chromatic_aberration: table(TAG_CA_PARAMS, 128),
        distortion: table(TAG_DISTORTION_PARAMS, 64),
    };
    (!tags.is_empty()).then_some(tags)
}

/// Black level, white level and white balance after source selection and bit-depth fitting.
struct ResolvedLevels {
    black: [f32; 4],
    white: [f32; 4],
    black_source: LevelSource,
    white_source: LevelSource,
    wb: Option<[f32; 4]>,
    wb_source: Option<LevelSource>,
}

/// The levels some bodies repeat as plain tags in the raw IFD, in R, G1, G2, B order.
#[derive(Clone, Copy, Debug, Default)]
struct RawIfdLevels {
    /// Tag 0x7310.
    black: Option<[u16; 4]>,
    /// DNG `WhiteLevel`, tag 0xC61D.
    white: Option<[u16; 4]>,
    /// Tag 0x7313.
    wb: Option<[u16; 4]>,
}

fn resolve_levels(
    decode: DecodeKind,
    bits: u16,
    cfa: CfaPattern,
    sr2: Option<&sr2::Sr2Data>,
    raw_ifd: RawIfdLevels,
) -> Result<ResolvedLevels, RawError> {
    // The scale of the decoded samples: cRAW is expanded to 14 bits by its curve.
    let sample_bits = if decode == DecodeKind::Craw || bits == 0 {
        14
    } else {
        bits
    };
    let (black_rggb, black_source) = match (sr2.and_then(|s| s.black), raw_ifd.black) {
        (Some(b), _) => (Some(b), LevelSource::Sr2),
        (None, Some(b)) => (Some(b), LevelSource::RawIfd),
        (None, None) => (None, LevelSource::Default),
    };
    let (white_rggb, white_source) = match (sr2.and_then(|s| s.white), raw_ifd.white) {
        (Some(w), _) => (Some(w), LevelSource::Sr2),
        (None, Some(w)) => (Some(w), LevelSource::RawIfd),
        (None, None) => (None, LevelSource::Default),
    };

    let default_black = if decode == DecodeKind::Craw {
        512
    } else {
        128u16 << sample_bits.clamp(12, 14).saturating_sub(12)
    };
    let default_white = ((1u32 << sample_bits.clamp(1, 16)) - 1) as u16;
    // Levels read from the file may be on Sony's 14-bit scale while the samples are narrower; the
    // white level tells (`sr2::level_shift`). A default is already on the samples' scale and is
    // never shifted, whatever the level next to it was.
    let shift = white_rggb.map_or(0, |w| sr2::level_shift(w, sample_bits));
    let positions = |levels: [u16; 4]| cfa.rggb_to_positions(levels.map(|v| f32::from(v >> shift)));
    let black = black_rggb.map_or([f32::from(default_black); 4], positions);
    let white = white_rggb.map_or([f32::from(default_white); 4], positions);
    if black.iter().zip(&white).any(|(b, w)| b >= w) {
        return Err(RawError::corrupt(
            "the black level is not below the white level",
        ));
    }

    let (wb_rggb, wb_source) = match (sr2.and_then(|s| s.wb), raw_ifd.wb) {
        (Some(w), _) => (Some(w), Some(LevelSource::Sr2)),
        (None, Some(w)) => (Some(w), Some(LevelSource::RawIfd)),
        (None, None) => (None, None),
    };
    Ok(ResolvedLevels {
        black,
        white,
        black_source,
        white_source,
        wb: wb_rggb.map(|w| cfa.rggb_to_positions(w.map(f32::from))),
        wb_source,
    })
}

/// Finds the embedded JPEG to show first, among the maker note, the `JPEGInterchangeFormat`
/// thumbnails and 8-bit JPEG strips in any directory.
///
/// A real ARW holds several: a 160 x 120 thumbnail, the 1616 x 1080 preview and, from ARW 4.0 on,
/// a full-size JPEG as well (7008 x 4672 and 1.7 MB in the ILCE-7M4 file of
/// metadata-extractor-images). The first thumbnails want the preview, not the full-size JPEG
/// that costs a 33 MP decode, so the choice is the smallest JPEG whose long edge is at least
/// [`PREVIEW_MIN_LONG_EDGE_PX`] pixels, and when none is that big (or no size can be read) the
/// largest JPEG. JPEGs coded with a lossless process are never chosen.
fn find_preview(
    tiff: &Tiff,
    nodes: &[IfdNode],
    maker: Option<&makernote::SonyMakerNote>,
) -> Option<JpegRef> {
    let file_len = tiff.source_len();
    let mut candidates: Vec<JpegRef> = Vec::new();
    if let Some((offset, length)) = maker.and_then(|m| m.preview) {
        candidates.push(JpegRef { offset, length });
    }
    for node in nodes {
        let ifd = &node.ifd;
        if let (Some(offset), Some(length)) = (
            first_uint(tiff, ifd, tag::JPEG_INTERCHANGE_FORMAT),
            first_uint(tiff, ifd, tag::JPEG_INTERCHANGE_FORMAT_LENGTH),
        ) {
            candidates.push(JpegRef { offset, length });
        }
        // An 8-bit JPEG image stored in strips (compression 6 or 7).
        let compression = first_uint(tiff, ifd, tag::COMPRESSION);
        let bits = first_uint(tiff, ifd, tag::BITS_PER_SAMPLE).unwrap_or(8);
        if matches!(compression, Some(6 | 7))
            && bits == 8
            && let (Some(offsets), Some(counts)) = (
                ifd.get(tag::STRIP_OFFSETS)
                    .and_then(|e| tiff.uints(e, 1).ok()),
                ifd.get(tag::STRIP_BYTE_COUNTS)
                    .and_then(|e| tiff.uints(e, 1).ok()),
            )
            && let (Some(offset), Some(length)) = (offsets.first(), counts.first())
            && ifd.get(tag::STRIP_OFFSETS).is_some_and(|e| e.count == 1)
        {
            candidates.push(JpegRef {
                offset: *offset,
                length: *length,
            });
        }
    }
    let usable: Vec<(JpegRef, u32)> = candidates
        .into_iter()
        .filter(|c| {
            (4..=MAX_PREVIEW_BYTES).contains(&c.length)
                && c.offset
                    .checked_add(c.length)
                    .is_some_and(|end| end <= file_len)
        })
        .filter_map(|c| {
            // A JPEG start, not one of the lossless-JPEG streams Sony keeps in some directories
            // (no viewer can decode them), and its size from the frame header (0 when the
            // header is not within the first bytes).
            let head = tiff
                .read_vec(c.offset, c.length.min(JPEG_PROBE_BYTES))
                .ok()?;
            if !head.starts_with(&[0xFF, 0xD8, 0xFF]) {
                return None;
            }
            let frame = jpeg_frame_header(&head);
            if frame.is_some_and(|(marker, _)| is_lossless_jpeg_marker(marker)) {
                return None;
            }
            let long_edge = frame
                .and_then(|(_, at)| {
                    let height = u16::from_be_bytes([*head.get(at + 3)?, *head.get(at + 4)?]);
                    let width = u16::from_be_bytes([*head.get(at + 5)?, *head.get(at + 6)?]);
                    Some(u32::from(height.max(width)))
                })
                .unwrap_or(0);
            Some((c, long_edge))
        })
        .collect();
    usable
        .iter()
        .filter(|(_, edge)| *edge >= PREVIEW_MIN_LONG_EDGE_PX)
        .min_by_key(|(jpeg, edge)| (*edge, jpeg.offset))
        .or_else(|| usable.iter().max_by_key(|(jpeg, _)| jpeg.length))
        .map(|(jpeg, _)| *jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgen::{ArwBuilder, SonyIsoTable};

    fn params(compression: u16, bits: u16, photometric: u16, samples: u16) -> RawParams {
        RawParams {
            compression,
            bits,
            photometric,
            samples,
            width: 100,
            height: 50,
            total_bytes: 0,
            raw_file_type: None,
            ycbcr_subsampling: None,
        }
    }

    #[test]
    fn classify_follows_the_spec_table() {
        use DecodeKind::*;
        // 32767 with 12 bits and exactly width*height*12/8 bytes is packed 12-bit.
        let packed12 = RawParams {
            total_bytes: 100 * 50 * 12 / 8,
            ..params(32767, 12, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&packed12), Uncompressed12);
        // 32767 with 8 bits (rawler's value for cRAW) is cRAW.
        let craw = RawParams {
            total_bytes: 100 * 50,
            ..params(32767, 8, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&craw), Craw);
        // The cameras write BitsPerSample 12 for cRAW too, and one byte per pixel: the ILCE-7M3 file
        // in metadata-extractor-images has 6048 x 4024, 12 bits, SonyRawFileType 2 and a strip of
        // 24 337 152 = 6048 x 4024 bytes. So do the A500, A5000, NEX-5N and NEX-7 files.
        let real_craw = RawParams {
            width: 6048,
            height: 4024,
            total_bytes: 24_337_152,
            raw_file_type: Some(2),
            ..params(32767, 12, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&real_craw), Craw);
        // 32767 with two bytes per pixel in the file is uncompressed 14-bit.
        let wide = RawParams {
            total_bytes: 100 * 50 * 2,
            ..params(32767, 14, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&wide), Uncompressed14);
        // The A200 to A390: 12 bits, Huffman-coded, fewer bytes than pixels (the A300 file in
        // OpenImageIO-images has 9 929 328 bytes for 3880 x 2600 pixels).
        let arw1 = RawParams {
            total_bytes: 3000,
            ..params(32767, 12, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&arw1), Arw1);
        // The same byte count at an unusual bit depth fits no decoder.
        let odd = RawParams {
            total_bytes: 6000,
            ..params(32767, 13, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(
            classify(&odd),
            Unknown {
                compression: 32767,
                bits: 13
            }
        );
        // Compression 1.
        assert_eq!(classify(&params(1, 14, PHOTOMETRIC_CFA, 1)), Uncompressed14);
        assert_eq!(classify(&params(1, 12, PHOTOMETRIC_CFA, 1)), Uncompressed12);
        assert_eq!(classify(&params(1, 16, PHOTOMETRIC_CFA, 1)), Uncompressed14);
        // Lossless.
        assert_eq!(classify(&params(7, 14, PHOTOMETRIC_CFA, 1)), LosslessL);
        assert_eq!(classify(&params(7, 12, PHOTOMETRIC_CFA, 1)), LosslessL);
        assert_eq!(
            classify(&params(7, 14, PHOTOMETRIC_YCBCR, 3)),
            LosslessMS {
                subsampling: ChromaSubsampling::Yuv420
            }
        );
        let s422 = RawParams {
            ycbcr_subsampling: Some(ChromaSubsampling::Yuv422),
            ..params(7, 14, PHOTOMETRIC_YCBCR, 3)
        };
        assert_eq!(
            classify(&s422),
            LosslessMS {
                subsampling: ChromaSubsampling::Yuv422
            }
        );
        // ARW 6.
        assert_eq!(classify(&params(32766, 12, PHOTOMETRIC_CFA, 1)), Arw6);
    }

    #[test]
    fn classify_uses_the_sony_raw_file_type_when_the_bit_depth_is_odd() {
        let odd = |file_type| RawParams {
            raw_file_type: file_type,
            ..params(1, 0, PHOTOMETRIC_CFA, 1)
        };
        assert_eq!(classify(&odd(Some(0))), DecodeKind::Uncompressed14);
        assert_eq!(classify(&odd(Some(1))), DecodeKind::Uncompressed12);
        assert_eq!(
            classify(&odd(None)),
            DecodeKind::Unknown {
                compression: 1,
                bits: 0
            }
        );
        assert_eq!(
            classify(&params(9999, 14, PHOTOMETRIC_CFA, 1)),
            DecodeKind::Unknown {
                compression: 9999,
                bits: 14
            }
        );
        // Lossless with two samples is neither L nor M/S.
        assert!(matches!(
            classify(&params(7, 14, PHOTOMETRIC_CFA, 2)),
            DecodeKind::Unknown { .. }
        ));
    }

    #[test]
    fn lens_table_splits_count_and_knots() {
        let table = LensTable {
            raw: vec![16, -3, 4, 5],
        };
        assert_eq!(table.count(), Some(16));
        assert_eq!(table.knots(), [-3, 4, 5]);
        let empty = LensTable { raw: vec![] };
        assert_eq!(empty.count(), None);
        assert!(empty.knots().is_empty());
    }

    #[test]
    fn jpeg_header_scan_finds_the_first_component_sampling() {
        // SOI, SOF3: length 17, precision 14, height 8, width 8, 3 components (Y 2x1).
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xC3, 0, 17, 14, 0, 8, 0, 8, 3];
        jpeg.extend_from_slice(&[1, 0x21, 0, 2, 0x11, 0, 3, 0x11, 0]);
        assert_eq!(jpeg_first_component_sampling(&jpeg), Some((2, 1)));
        // A DHT segment (0xC4) in front is skipped, not mistaken for a frame header.
        let mut with_dht = vec![0xFF, 0xD8, 0xFF, 0xC4, 0, 4, 0, 0];
        with_dht.extend_from_slice(&jpeg[2..]);
        assert_eq!(jpeg_first_component_sampling(&with_dht), Some((2, 1)));
        // Truncated and non-JPEG input.
        assert_eq!(jpeg_first_component_sampling(&jpeg[..8]), None);
        assert_eq!(jpeg_first_component_sampling(b"not a jpeg"), None);
        assert_eq!(jpeg_first_component_sampling(&[]), None);
        // SOS before any frame header.
        assert_eq!(
            jpeg_first_component_sampling(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]),
            None
        );
    }

    #[test]
    fn arw_version_names_follow_exiftool() {
        for (bytes, name) in [
            ([2, 0, 0, 0], "ARW 1.0"),
            ([3, 0, 0, 0], "ARW 2.0"),
            ([3, 2, 0, 0], "ARW 2.2"),
            ([3, 3, 0, 0], "ARW 2.3"),
            ([3, 3, 1, 0], "ARW 2.3.1"),
            ([3, 3, 2, 0], "ARW 2.3.2"),
            ([3, 3, 5, 0], "ARW 2.3.5"),
            ([4, 0, 0, 0], "ARW 4.0"),
            ([4, 0, 1, 0], "ARW 4.0.1"),
            ([5, 0, 0, 0], "ARW 5.0"),
            ([5, 0, 1, 0], "ARW 5.0.1"),
            ([6, 0, 0, 0], "ARW 6.0"),
        ] {
            assert_eq!(arw_version_name(bytes).as_deref(), Some(name), "{bytes:?}");
        }
        assert_eq!(arw_version_name([1, 0, 0, 0]), None);
        assert_eq!(arw_version_name([0, 0, 0, 2]), None);
    }

    #[test]
    fn read_preview_from_refuses_absurd_lengths_and_reads_what_exists() {
        let file = crate::testgen::ArwBuilder::new("ILCE-7M3", 32, 32).build(&[0u16; 32 * 32]);
        let mut meta = probe_bytes(&file).expect("probe");
        let found = read_preview_from(&MemSource(&file), &meta)
            .expect("read")
            .expect("preview");
        assert_eq!(Some(found.len() as u64), meta.preview.map(|p| p.length));
        meta.preview = Some(JpegRef {
            offset: 0,
            length: MAX_PREVIEW_BYTES + 1,
        });
        assert!(read_preview_from(&MemSource(&file), &meta).is_err());
        meta.preview = Some(JpegRef {
            offset: file.len() as u64 - 10,
            length: 100,
        });
        assert!(matches!(
            read_preview_from(&MemSource(&file), &meta),
            Err(RawError::Corrupt { .. })
        ));
        meta.preview = None;
        assert_eq!(
            read_preview_from(&MemSource(&file), &meta).expect("read"),
            None
        );
    }

    #[test]
    fn level_source_names_are_stable() {
        assert_eq!(LevelSource::Sr2.name(), "sr2");
        assert_eq!(LevelSource::RawIfd.name(), "raw_ifd");
        assert_eq!(LevelSource::Default.name(), "default");
    }

    #[test]
    fn default_levels_follow_the_documented_rules() {
        let cfa = CfaPattern::RGGB;
        // cRAW: black 512, white on the 14-bit scale.
        let craw = resolve_levels(DecodeKind::Craw, 8, cfa, None, RawIfdLevels::default())
            .expect("levels");
        assert_eq!(craw.black, [512.0; 4]);
        assert_eq!(craw.white, [16383.0; 4]);
        assert_eq!(craw.black_source, LevelSource::Default);
        // 14-bit uncompressed: 128 << 2 = 512; 12-bit: 128.
        let u14 = resolve_levels(
            DecodeKind::Uncompressed14,
            14,
            cfa,
            None,
            RawIfdLevels::default(),
        )
        .expect("levels");
        assert_eq!((u14.black[0], u14.white[0]), (512.0, 16383.0));
        let u12 = resolve_levels(
            DecodeKind::Uncompressed12,
            12,
            cfa,
            None,
            RawIfdLevels::default(),
        )
        .expect("levels");
        assert_eq!((u12.black[0], u12.white[0]), (128.0, 4095.0));
        assert!(u12.wb.is_none());
    }

    /// cRAW decodes to a 14-bit scale whatever BitsPerSample says (12 in real files): the SR2
    /// levels of the real ILCE-7M3 file (black 512, white 15360) are kept, as LibRaw reports them.
    #[test]
    fn craw_levels_stay_on_the_14_bit_scale_although_the_tag_says_12_bits() {
        let sr2 = sr2::Sr2Data {
            black: Some([512; 4]),
            white: Some([15_360; 4]),
            ..Default::default()
        };
        let levels = resolve_levels(
            DecodeKind::Craw,
            12,
            CfaPattern::RGGB,
            Some(&sr2),
            RawIfdLevels::default(),
        )
        .expect("levels");
        assert_eq!(levels.black, [512.0; 4]);
        assert_eq!(levels.white, [15_360.0; 4]);
        // The same numbers on packed 12-bit data are on the wrong scale and are shifted.
        let packed = resolve_levels(
            DecodeKind::Uncompressed12,
            12,
            CfaPattern::RGGB,
            Some(&sr2),
            RawIfdLevels::default(),
        )
        .expect("levels");
        assert_eq!(packed.black, [128.0; 4]);
        assert_eq!(packed.white, [3840.0; 4]);
    }

    #[test]
    fn sr2_levels_beat_raw_ifd_levels_which_beat_defaults() {
        let sr2 = sr2::Sr2Data {
            black: Some([600; 4]),
            white: Some([16000; 4]),
            wb: Some([2000, 1024, 1024, 1700]),
            ..Default::default()
        };
        let from_sr2 = resolve_levels(
            DecodeKind::Uncompressed14,
            14,
            CfaPattern::RGGB,
            Some(&sr2),
            RawIfdLevels {
                black: Some([700; 4]),
                white: Some([15000; 4]),
                wb: Some([1, 1, 1, 1]),
            },
        )
        .expect("levels");
        assert_eq!(from_sr2.black, [600.0; 4]);
        assert_eq!(from_sr2.black_source, LevelSource::Sr2);
        assert_eq!(from_sr2.white, [16000.0; 4]);
        assert_eq!(from_sr2.white_source, LevelSource::Sr2);
        assert_eq!(from_sr2.wb, Some([2000.0, 1024.0, 1024.0, 1700.0]));
        assert_eq!(from_sr2.wb_source, Some(LevelSource::Sr2));

        let from_raw = resolve_levels(
            DecodeKind::Uncompressed14,
            14,
            CfaPattern::RGGB,
            None,
            RawIfdLevels {
                black: Some([700; 4]),
                white: None,
                wb: Some([2100, 1024, 1024, 1650]),
            },
        )
        .expect("levels");
        assert_eq!(from_raw.black, [700.0; 4]);
        assert_eq!(from_raw.black_source, LevelSource::RawIfd);
        assert_eq!(from_raw.white_source, LevelSource::Default);
        assert_eq!(from_raw.wb_source, Some(LevelSource::RawIfd));
    }

    /// The ILCE-9 in the lclevy/sony_raw dump has white level 15360 in SR2 and the DNG
    /// `WhiteLevel` 16383 in the raw IFD: SR2 wins, and the raw IFD value stands in for it when
    /// there is no SR2 block (docs/04: SR2, then the raw IFD, then the default).
    #[test]
    fn the_raw_ifd_white_level_comes_after_sr2_and_before_the_default() {
        let sr2 = sr2::Sr2Data {
            white: Some([15360; 4]),
            ..Default::default()
        };
        let raw_ifd = RawIfdLevels {
            white: Some([16000, 16100, 16100, 16200]),
            ..Default::default()
        };
        let cfa = CfaPattern::RGGB;
        let with_sr2 = resolve_levels(DecodeKind::Uncompressed14, 14, cfa, Some(&sr2), raw_ifd)
            .expect("levels");
        assert_eq!(
            (with_sr2.white, with_sr2.white_source),
            ([15360.0; 4], LevelSource::Sr2)
        );
        let without =
            resolve_levels(DecodeKind::Uncompressed14, 14, cfa, None, raw_ifd).expect("levels");
        assert_eq!(
            (without.white, without.white_source),
            ([16000.0, 16100.0, 16100.0, 16200.0], LevelSource::RawIfd)
        );
        // A raw-IFD white level on the 14-bit scale is fitted to 12-bit data like any other.
        let wide = RawIfdLevels {
            white: Some([16383; 4]),
            ..Default::default()
        };
        let fitted =
            resolve_levels(DecodeKind::Uncompressed12, 12, cfa, None, wide).expect("levels");
        assert_eq!(fitted.white, [4095.0; 4]);
        assert_eq!(fitted.black, [128.0; 4]);
    }

    #[test]
    fn inconsistent_levels_are_corrupt() {
        let sr2 = sr2::Sr2Data {
            black: Some([5000; 4]),
            white: Some([4000; 4]),
            ..Default::default()
        };
        let error = resolve_levels(
            DecodeKind::Uncompressed14,
            14,
            CfaPattern::RGGB,
            Some(&sr2),
            RawIfdLevels::default(),
        )
        .err()
        .expect("error");
        assert!(matches!(error, RawError::Corrupt { .. }));
    }

    // --- exposure context and Sony fields through the whole probe ----------------------------

    /// A probe of a synthetic 16 x 16 file written by `builder`.
    fn probe_with(builder: ArwBuilder) -> RawMetadata {
        let file = builder.build(&vec![1000u16; 16 * 16]);
        probe_bytes(&file).expect("the synthetic file probes")
    }

    /// A 16-byte ASCII value holding `name`, NUL padded.
    fn name16(name: &str) -> Vec<u8> {
        let mut value = name.as_bytes().to_vec();
        value.resize(16, 0);
        value
    }

    #[test]
    fn the_exposure_and_sony_fields_come_from_the_file() {
        let meta = probe_with(
            ArwBuilder::new("ILCE-7M3", 16, 16)
                .exif_entry(
                    tag::BRIGHTNESS_VALUE,
                    field_type::SRATIONAL,
                    1,
                    [(-3i32).to_le_bytes(), 2i32.to_le_bytes()].concat(),
                )
                .exif_entry(
                    tag::METERING_MODE,
                    field_type::SHORT,
                    1,
                    2u16.to_le_bytes().to_vec(),
                )
                .exif_entry(
                    tag::FLASH,
                    field_type::SHORT,
                    1,
                    0x19u16.to_le_bytes().to_vec(),
                )
                .exif_entry(
                    tag::EXPOSURE_PROGRAM,
                    field_type::SHORT,
                    1,
                    1u16.to_le_bytes().to_vec(),
                )
                .exif_entry(
                    tag::SCENE_CAPTURE_TYPE,
                    field_type::SHORT,
                    1,
                    2u16.to_le_bytes().to_vec(),
                )
                .maker_entry(
                    makernote::TAG_CREATIVE_STYLE,
                    field_type::ASCII,
                    16,
                    name16("Vivid"),
                )
                .maker_entry(
                    makernote::TAG_DRO_MODE,
                    field_type::LONG,
                    1,
                    3u32.to_le_bytes().to_vec(),
                )
                .maker_entry(
                    makernote::TAG_SCENE_MODE,
                    field_type::LONG,
                    1,
                    6u32.to_le_bytes().to_vec(),
                ),
        );
        assert_eq!(meta.brightness_value, Some(-1.5));
        assert_eq!(meta.metering_mode, Some(MeteringMode::CenterWeighted));
        assert_eq!(
            meta.flash,
            Some(FlashInfo {
                fired: true,
                raw: 0x19
            })
        );
        assert_eq!(meta.exposure_program, Some(ExposureProgram::Manual));
        assert_eq!(meta.scene_capture_type, Some(SceneCaptureType::Portrait));
        assert_eq!(meta.creative_style, Some(CreativeStyle::Vivid));
        assert_eq!(meta.dro_mode, Some(DroMode::Auto));
        assert_eq!(meta.scene_mode, Some(SceneMode::Landscape));
    }

    #[test]
    fn the_colour_space_tag_is_read_as_written() {
        // The builder writes ColorSpace 1 (sRGB) by default.
        let default = probe_with(ArwBuilder::new("ILCE-7M3", 16, 16));
        assert_eq!(default.color_space, Some(1));
        // The Adobe RGB mode writes 0xFFFF (uncalibrated), which replaces the default entry.
        let adobe = probe_with(ArwBuilder::new("ILCE-7M3", 16, 16).exif_entry(
            tag::COLOR_SPACE,
            field_type::SHORT,
            1,
            0xFFFFu16.to_le_bytes().to_vec(),
        ));
        assert_eq!(adobe.color_space, Some(0xFFFF));
    }

    #[test]
    fn a_default_file_reads_its_builder_codes_and_none_for_the_new_sony_tags() {
        let meta = probe_with(ArwBuilder::new("ILCE-7RM4", 16, 16));
        // The builder writes MeteringMode 5, ExposureProgram 2 and Flash 0 by default.
        assert_eq!(meta.metering_mode, Some(MeteringMode::Multi));
        assert_eq!(meta.exposure_program, Some(ExposureProgram::ProgramAe));
        assert_eq!(
            meta.flash,
            Some(FlashInfo {
                fired: false,
                raw: 0
            })
        );
        // Absent from the default file.
        assert_eq!(meta.brightness_value, None);
        assert_eq!(meta.scene_capture_type, None);
        assert_eq!(meta.dro_mode, None);
        assert_eq!(meta.creative_style, None);
        assert_eq!(meta.scene_mode, None);
    }

    #[test]
    fn creative_style_falls_back_to_the_enciphered_table_except_on_the_ilce_7m5() {
        let with_table = |model: &str| {
            probe_with(
                ArwBuilder::new(model, 16, 16)
                    .iso_table(SonyIsoTable::Tag9416)
                    .creative_style_9416(15),
            )
        };
        assert_eq!(
            with_table("ILCE-7M4").creative_style,
            Some(CreativeStyle::Other("FL".to_owned()))
        );
        assert_eq!(with_table("ILCE-7M5").creative_style, None);
        // A name in the plain tag is used whatever the table says.
        let named = probe_with(
            ArwBuilder::new("ILCE-7M4", 16, 16)
                .iso_table(SonyIsoTable::Tag9416)
                .creative_style_9416(15)
                .maker_entry(
                    makernote::TAG_CREATIVE_STYLE,
                    field_type::ASCII,
                    16,
                    name16("Neutral"),
                ),
        );
        assert_eq!(named.creative_style, Some(CreativeStyle::Neutral));
    }

    #[test]
    fn a_broken_new_tag_reads_as_none_and_the_file_still_probes() {
        let meta = probe_with(
            ArwBuilder::new("ILCE-7M3", 16, 16)
                .exif_entry(
                    tag::BRIGHTNESS_VALUE,
                    field_type::ASCII,
                    4,
                    b"abc\0".to_vec(),
                )
                .exif_entry(tag::FLASH, field_type::RATIONAL, 1, vec![0; 8])
                .maker_entry(
                    makernote::TAG_DRO_MODE,
                    field_type::ASCII,
                    4,
                    b"3\0\0\0".to_vec(),
                )
                .maker_entry(
                    makernote::TAG_CREATIVE_STYLE,
                    field_type::LONG,
                    1,
                    1u32.to_le_bytes().to_vec(),
                ),
        );
        assert_eq!(meta.brightness_value, None);
        assert_eq!(meta.flash, None);
        assert_eq!(meta.dro_mode, None);
        assert_eq!(meta.creative_style, None);
        // The rest of the file is still read.
        assert_eq!((meta.width, meta.height), (16, 16));
        assert_eq!(meta.model, "ILCE-7M3");
    }
}
