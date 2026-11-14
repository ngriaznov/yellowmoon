//! DNG input: our own reader of Adobe's Digital Negative (written from the public DNG
//! Specification 1.7 and TIFF 6.0; no code from any other DNG reader).
//!
//! A file is a DNG when IFD0 carries `DNGVersion`. [`probe`] turns it into a [`RawMetadata`] like
//! any ARW (the same fields, so the rest of the program needs no second path) and puts the DNG
//! colour data in [`DngInfo`]; [`decode`] reads the pixels.
//!
//! # What is read
//!
//! The main image is the directory (IFD0 or a SubIFD) with `NewSubFileType` 0 and photometric
//! interpretation 32803 (CFA) or 34892 (LinearRaw); the one with the most pixels wins.
//!
//! * **CFA**: one sample per pixel, 8 to 16 bits, a 2 x 2 Bayer `CFAPattern` in any of its four
//!   orders (through `CFAPlaneColor`), uncompressed or lossless JPEG (compression 7, ITU T.81
//!   process 14, predictors 1 to 7, one to four components), in strips or tiles. A lossless JPEG
//!   with `n` components holds `n` consecutive columns per JPEG sample: the decoded image row is
//!   `plane_0[x], plane_1[x], ..., plane_{n-1}[x], plane_0[x + 1], ...` (the DNG layout).
//! * **LinearRaw**: three interleaved samples per pixel, uncompressed or lossless JPEG. It becomes
//!   [`Layout::Ycbcr`] (already demosaiced camera RGB, no mosaic), the path Lossless M and S take.
//! * **Levels and crop**: `BlackLevel` with `BlackLevelRepeatDim`, `BlackLevelDeltaH` and
//!   `BlackLevelDeltaV`, `WhiteLevel`, `LinearizationTable`, `ActiveArea`, `DefaultCropOrigin` and
//!   `DefaultCropSize`, `Orientation`. The CFA pattern, the black level pattern, the black level
//!   deltas and the default crop are all relative to the active area's top left corner (the
//!   stage 2 image of the specification); [`RawMetadata::crop`] and [`RawMetadata::active_area`]
//!   are in readout coordinates like the ARW's.
//! * **Colour**: `ColorMatrix1/2`, `CalibrationIlluminant1/2`, `CameraCalibration1/2`,
//!   `AnalogBalance`, `AsShotNeutral` or `AsShotWhiteXY`, `ForwardMatrix1/2` and `NoiseProfile`,
//!   kept as written in [`DngInfo`]. `AsShotNeutral` (or the neutral worked out from
//!   `AsShotWhiteXY`) becomes [`RawMetadata::wb_as_shot`].
//! * **`OpcodeList2`**: the gain map (opcode 9), applied to the raw samples at decode.
//!
//! # What is refused
//!
//! Lossy JPEG (34892), JPEG XL (52546), Deflate, floating-point and signed samples, a colour
//! filter that is not 2 x 2 Bayer, other sample layouts, a `DefaultScale` other than 1 : 1, a DNG
//! without raw data, and any opcode of `OpcodeList2` other than the gain map that is not marked
//! optional. Each is a [`RawError::UnsupportedDng`] with one plain sentence; a malformed file is
//! [`RawError::Corrupt`]. Nothing is guessed at and nothing panics.
//!
//! # What is ignored, on purpose
//!
//! * `BaselineExposure`: an ARW has none, and a Sony DNG must render like its ARW. Exposure is
//!   Auto's job here.
//! * `OpcodeList1` (before linearisation: bad pixel fixes, per-row deltas) and `OpcodeList3`
//!   (after demosaicing: warp, vignette, trim). The lens-correction stage reads Sony's tables, and
//!   a DNG's polynomial warp does not map onto them (they would need a polynomial radial model
//!   in the lens stage). The picture is then uncorrected by the DNG's own lens opcodes.
//! * `DefaultUserCrop`, `MaskedAreas`, `ReductionMatrix`, `ProfileHueSatMap` and the other
//!   rendering hints.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::decode::ljpeg;
use crate::error::{DngRefusal, RawError};
use crate::exif;
use crate::layout::{CfaPattern, DecodeKind, MAX_DIMENSION, MIN_DIMENSION, Plane, Rect};
use crate::metadata::{
    LevelSource, PHOTOMETRIC_CFA, RawIfd, RawMetadata, RawStorage, find_preview, read_storage,
};
use crate::tiff::{ByteOrder, ByteSource, Entry, Ifd, IfdKind, IfdNode, Tiff, tag};

/// TIFF photometric interpretation of DNG linear raw data.
pub const PHOTOMETRIC_LINEAR_RAW: u16 = 34892;
/// TIFF compression of lossy JPEG in a DNG.
pub const COMPRESSION_LOSSY_JPEG: u16 = 34892;
/// TIFF compression of JPEG XL in a DNG.
pub const COMPRESSION_JPEG_XL: u16 = 52546;

/// The most samples (all channels) of a DNG image: 50 megapixels of three channels. Declared sizes
/// above this are refused before anything is allocated.
pub const MAX_SAMPLES: u64 = 150_000_000;

/// The largest number of gain maps and other opcodes read from `OpcodeList2`.
const MAX_OPCODES: usize = 256;
/// The most points one gain map may have (all planes).
const MAX_MAP_POINTS: u64 = 1 << 20;

/// DNG tags this reader uses (DNG Specification 1.7, chapter 3).
pub mod tags {
    /// `DNGBackwardVersion`.
    pub const DNG_BACKWARD_VERSION: u16 = 0xC613;
    /// `UniqueCameraModel`.
    pub const UNIQUE_CAMERA_MODEL: u16 = 0xC614;
    /// `CFAPlaneColor`.
    pub const CFA_PLANE_COLOR: u16 = 0xC616;
    /// `CFALayout`.
    pub const CFA_LAYOUT: u16 = 0xC617;
    /// `LinearizationTable`.
    pub const LINEARIZATION_TABLE: u16 = 0xC618;
    /// `BlackLevelRepeatDim`.
    pub const BLACK_LEVEL_REPEAT_DIM: u16 = 0xC619;
    /// `BlackLevel`.
    pub const BLACK_LEVEL: u16 = 0xC61A;
    /// `BlackLevelDeltaH`.
    pub const BLACK_LEVEL_DELTA_H: u16 = 0xC61B;
    /// `BlackLevelDeltaV`.
    pub const BLACK_LEVEL_DELTA_V: u16 = 0xC61C;
    /// `WhiteLevel`.
    pub const WHITE_LEVEL: u16 = 0xC61D;
    /// `DefaultScale`.
    pub const DEFAULT_SCALE: u16 = 0xC61E;
    /// `ColorMatrix1`.
    pub const COLOR_MATRIX_1: u16 = 0xC621;
    /// `ColorMatrix2`.
    pub const COLOR_MATRIX_2: u16 = 0xC622;
    /// `CameraCalibration1`.
    pub const CAMERA_CALIBRATION_1: u16 = 0xC623;
    /// `CameraCalibration2`.
    pub const CAMERA_CALIBRATION_2: u16 = 0xC624;
    /// `AnalogBalance`.
    pub const ANALOG_BALANCE: u16 = 0xC627;
    /// `AsShotNeutral`.
    pub const AS_SHOT_NEUTRAL: u16 = 0xC628;
    /// `AsShotWhiteXY`.
    pub const AS_SHOT_WHITE_XY: u16 = 0xC629;
    /// `CalibrationIlluminant1`.
    pub const CALIBRATION_ILLUMINANT_1: u16 = 0xC65A;
    /// `CalibrationIlluminant2`.
    pub const CALIBRATION_ILLUMINANT_2: u16 = 0xC65B;
    /// `ActiveArea`.
    pub const ACTIVE_AREA: u16 = 0xC68D;
    /// `ForwardMatrix1`.
    pub const FORWARD_MATRIX_1: u16 = 0xC714;
    /// `ForwardMatrix2`.
    pub const FORWARD_MATRIX_2: u16 = 0xC715;
    /// `OpcodeList1`.
    pub const OPCODE_LIST_1: u16 = 0xC740;
    /// `OpcodeList2`.
    pub const OPCODE_LIST_2: u16 = 0xC741;
    /// `OpcodeList3`.
    pub const OPCODE_LIST_3: u16 = 0xC74E;
    /// `NoiseProfile`.
    pub const NOISE_PROFILE: u16 = 0xC761;
    /// TIFF `SampleFormat`.
    pub const SAMPLE_FORMAT: u16 = 0x0153;
    /// TIFF `PlanarConfiguration`.
    pub const PLANAR_CONFIGURATION: u16 = 0x011C;
    /// TIFF/EP `CFARepeatPatternDim`.
    pub const CFA_REPEAT_PATTERN_DIM: u16 = 0x828D;
}

/// The opcode ID of the gain map (DNG Specification 1.7, "Opcodes").
pub const OPCODE_GAIN_MAP: u32 = 9;

/// Whether the DNG holds a mosaic or linear (demosaiced) raw data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DngKind {
    /// `PhotometricInterpretation` 32803: one sample per photosite, 2 x 2 Bayer.
    Cfa,
    /// `PhotometricInterpretation` 34892: three samples per pixel.
    LinearRaw,
}

/// One `GainMap` opcode (opcode 9): a grid of gains laid over a rectangle of the image, applied
/// to the linearised raw samples (lens shading, for example).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainMap {
    /// Top row of the area, relative to the active area.
    pub top: u32,
    /// Left column of the area, relative to the active area.
    pub left: u32,
    /// Bottom edge (exclusive).
    pub bottom: u32,
    /// Right edge (exclusive).
    pub right: u32,
    /// First image plane (sample of each pixel) the map applies to.
    pub plane: u32,
    /// How many planes.
    pub planes: u32,
    /// Rows between the rows the map applies to (2 selects one CFA row of a mosaic).
    pub row_pitch: u32,
    /// Columns between the columns the map applies to.
    pub col_pitch: u32,
    /// Map points down.
    pub points_v: u32,
    /// Map points across.
    pub points_h: u32,
    /// Vertical spacing of the map points, as a fraction of the area's height.
    pub spacing_v: f64,
    /// Horizontal spacing, as a fraction of the area's width.
    pub spacing_h: f64,
    /// Vertical position of the first map point, as a fraction of the area's height.
    pub origin_v: f64,
    /// Horizontal position of the first map point.
    pub origin_h: f64,
    /// Planes in the map (1, or one per plane of the area).
    pub map_planes: u32,
    /// `points_v * points_h * map_planes` gains, row by row, planes interleaved.
    pub gains: Vec<f32>,
}

/// What the DNG tags say that the rest of [`RawMetadata`] has no field for: the colour data and the
/// pieces the decoder needs. Matrices are row by row as written (3 x 3, nine numbers).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DngInfo {
    /// `DNGVersion`.
    pub version: [u8; 4],
    /// Mosaic or linear raw.
    pub kind: DngKind,
    /// True when the file is big endian (16-bit uncompressed samples follow the byte order).
    pub big_endian: bool,
    /// `UniqueCameraModel`, for example `Sony ILCE-7RM4`.
    pub unique_camera_model: Option<String>,
    /// `ColorMatrix1` and `ColorMatrix2`: XYZ (D50 PCS values under the calibration illuminant) to
    /// the reference camera space.
    pub color_matrix: [Option<[f64; 9]>; 2],
    /// `CalibrationIlluminant1` and `2` (EXIF light source codes: 21 is D65, 23 is D50, 17 is A).
    pub calibration_illuminant: [Option<u16>; 2],
    /// `CameraCalibration1` and `2` (identity when absent).
    pub camera_calibration: [Option<[f64; 9]>; 2],
    /// `AnalogBalance` (ones when absent).
    pub analog_balance: Option<[f64; 3]>,
    /// `AsShotNeutral`: the camera's response to the as-shot white, in the camera's own space.
    pub as_shot_neutral: Option<[f64; 3]>,
    /// `AsShotWhiteXY`.
    pub as_shot_white_xy: Option<[f64; 2]>,
    /// `ForwardMatrix1` and `2`.
    pub forward_matrix: [Option<[f64; 9]>; 2],
    /// `NoiseProfile` as written: a `(scale, offset)` pair per plane (one pair for all planes, or
    /// one per plane).
    pub noise_profile: Option<Vec<f64>>,
    /// `LinearizationTable`: applied to every sample before anything else.
    pub linearization: Option<Vec<u16>>,
    /// `BlackLevelDeltaH`, one value per column of the active area.
    pub black_delta_h: Vec<f32>,
    /// `BlackLevelDeltaV`, one value per row of the active area.
    pub black_delta_v: Vec<f32>,
    /// The gain maps of `OpcodeList2`, in file order.
    pub gain_maps: Vec<GainMap>,
}

/// True when IFD0 of the walked file has a `DNGVersion` tag.
pub fn is_dng(nodes: &[IfdNode]) -> bool {
    nodes
        .iter()
        .any(|n| matches!(n.kind, IfdKind::Root(0)) && n.ifd.get(tag::DNG_VERSION).is_some())
}

// ---------------------------------------------------------------------------------------------
// Tag access
// ---------------------------------------------------------------------------------------------

/// The directories to search for a tag: the raw one first, then IFD0.
struct Dirs<'a> {
    raw: &'a Ifd,
    root: &'a Ifd,
}

impl<'a> Dirs<'a> {
    fn entry(&self, tag_ref: u16) -> Option<&'a Entry> {
        self.raw.get(tag_ref).or_else(|| self.root.get(tag_ref))
    }
}

fn floats(tiff: &Tiff, entry: Option<&Entry>, max: usize) -> Option<Vec<f64>> {
    let values = tiff.floats(entry?, max).ok()?;
    values.iter().all(|v| v.is_finite()).then_some(values)
}

fn uints(tiff: &Tiff, entry: Option<&Entry>, max: usize) -> Option<Vec<u64>> {
    tiff.uints(entry?, max).ok()
}

fn first_uint(tiff: &Tiff, entry: Option<&Entry>) -> Option<u64> {
    uints(tiff, entry, 1)?.first().copied()
}

fn matrix9(tiff: &Tiff, entry: Option<&Entry>) -> Option<[f64; 9]> {
    let entry = entry?;
    if entry.count != 9 {
        return None; // a four-colour camera: not used
    }
    floats(tiff, Some(entry), 9)?.try_into().ok()
}

fn triple(tiff: &Tiff, entry: Option<&Entry>) -> Option<[f64; 3]> {
    let entry = entry?;
    if entry.count != 3 {
        return None;
    }
    floats(tiff, Some(entry), 3)?.try_into().ok()
}

fn corrupt<T>(detail: &str) -> Result<T, RawError> {
    Err(RawError::corrupt(detail))
}

fn refuse<T>(why: DngRefusal) -> Result<T, RawError> {
    Err(RawError::UnsupportedDng(why))
}

// ---------------------------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------------------------

/// The geometry and format of the main image directory.
struct MainImage {
    node: usize,
    kind: DngKind,
    width: u32,
    height: u32,
    bits: u16,
    samples: u16,
    compression: u16,
}

/// Picks the main image directory and checks that it is something this reader reads.
fn main_image(tiff: &Tiff, nodes: &[IfdNode]) -> Result<MainImage, RawError> {
    let mut best: Option<(u64, MainImage)> = None;
    let mut saw_picture = false;
    for (index, node) in nodes.iter().enumerate() {
        if !matches!(node.kind, IfdKind::Root(_) | IfdKind::Sub) {
            continue;
        }
        let ifd = &node.ifd;
        let subfile = first_uint(tiff, ifd.get(tag::NEW_SUBFILE_TYPE)).unwrap_or(0);
        let photometric = first_uint(tiff, ifd.get(tag::PHOTOMETRIC)).unwrap_or(0);
        let kind = match photometric {
            p if p == u64::from(PHOTOMETRIC_CFA) => DngKind::Cfa,
            p if p == u64::from(PHOTOMETRIC_LINEAR_RAW) => DngKind::LinearRaw,
            _ => {
                if subfile == 0 && ifd.get(tag::IMAGE_WIDTH).is_some() {
                    saw_picture = true;
                }
                continue;
            }
        };
        if subfile & 5 != 0 {
            continue; // a reduced resolution copy or a transparency mask
        }
        let (Some(width), Some(height)) = (
            first_uint(tiff, ifd.get(tag::IMAGE_WIDTH)).and_then(|v| u32::try_from(v).ok()),
            first_uint(tiff, ifd.get(tag::IMAGE_LENGTH)).and_then(|v| u32::try_from(v).ok()),
        ) else {
            return corrupt("the raw image has no size");
        };
        let compression = first_uint(tiff, ifd.get(tag::COMPRESSION))
            .and_then(|v| u16::try_from(v).ok())
            .unwrap_or(1);
        let samples = first_uint(tiff, ifd.get(tag::SAMPLES_PER_PIXEL))
            .and_then(|v| u16::try_from(v).ok())
            .unwrap_or(1);
        let bits_list = uints(tiff, ifd.get(tag::BITS_PER_SAMPLE), 4).unwrap_or_default();
        let bits = match bits_list.first() {
            Some(first) if bits_list.iter().all(|b| b == first) => {
                u16::try_from(*first).unwrap_or(0)
            }
            _ => 0,
        };
        let pixels = u64::from(width) * u64::from(height);
        let candidate = MainImage {
            node: index,
            kind,
            width,
            height,
            bits,
            samples,
            compression,
        };
        if best.as_ref().is_none_or(|(p, _)| pixels > *p) {
            best = Some((pixels, candidate));
        }
    }
    match best {
        Some((_, image)) => Ok(image),
        None if saw_picture => refuse(DngRefusal::NotRaw),
        None => corrupt("the DNG has no raw image directory"),
    }
}

/// Reads a DNG into [`RawMetadata`] without decoding pixels.
///
/// `nodes` is the result of [`Tiff::walk`]; IFD0 must carry `DNGVersion` ([`is_dng`]).
pub fn probe(tiff: &Tiff, nodes: &[IfdNode]) -> Result<RawMetadata, RawError> {
    let exif = exif::read(tiff, nodes);
    let root = nodes
        .iter()
        .find(|n| matches!(n.kind, IfdKind::Root(0)))
        .map(|n| &n.ifd)
        .ok_or_else(|| RawError::corrupt("the DNG has no first directory"))?;

    let version: [u8; 4] = uints(tiff, root.get(tag::DNG_VERSION), 4)
        .and_then(|v| {
            v.into_iter()
                .map(|b| u8::try_from(b).ok())
                .collect::<Option<Vec<_>>>()
        })
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| RawError::corrupt("the DNG version is damaged"))?;
    if let Some(backward) = uints(tiff, root.get(tags::DNG_BACKWARD_VERSION), 4)
        && let [major, minor, ..] = backward[..]
        && (major > 1 || (major == 1 && minor > 7))
    {
        return refuse(DngRefusal::NewerVersion);
    }

    let main = main_image(tiff, nodes)?;
    let raw_node = &nodes[main.node];
    let dirs = Dirs {
        raw: &raw_node.ifd,
        root,
    };
    let ifd = &raw_node.ifd;

    // Refuse what we cannot read, in the order a person would hit it.
    match main.compression {
        1 | 7 => {}
        COMPRESSION_LOSSY_JPEG => return refuse(DngRefusal::LossyJpeg),
        COMPRESSION_JPEG_XL => return refuse(DngRefusal::JpegXl),
        8 | 32946 => return refuse(DngRefusal::Deflate),
        other => return refuse(DngRefusal::Compression(other)),
    }
    let sample_format = uints(tiff, ifd.get(tags::SAMPLE_FORMAT), 4).unwrap_or_default();
    if sample_format.iter().any(|f| *f != 1) {
        return refuse(DngRefusal::FloatData);
    }
    let planar = first_uint(tiff, ifd.get(tags::PLANAR_CONFIGURATION)).unwrap_or(1);
    let expected_samples = match main.kind {
        DngKind::Cfa => 1,
        DngKind::LinearRaw => 3,
    };
    if main.samples != expected_samples
        || !(8..=16).contains(&main.bits)
        || (main.samples > 1 && planar != 1)
    {
        return refuse(DngRefusal::SampleLayout);
    }
    if let Some(scale) = floats(
        tiff,
        ifd.get(tags::DEFAULT_SCALE)
            .or(root.get(tags::DEFAULT_SCALE)),
        2,
    ) && scale != [1.0, 1.0]
    {
        return refuse(DngRefusal::Scaled);
    }

    if main.width < MIN_DIMENSION
        || main.height < MIN_DIMENSION
        || main.width > MAX_DIMENSION
        || main.height > MAX_DIMENSION
        || u64::from(main.width) * u64::from(main.height) * u64::from(main.samples) > MAX_SAMPLES
    {
        return corrupt(&format!(
            "image size {} x {} is out of range",
            main.width, main.height
        ));
    }

    // The CFA pattern, as written (relative to the active area).
    let declared_cfa = match main.kind {
        DngKind::Cfa => declared_pattern(tiff, ifd)?,
        DngKind::LinearRaw => CfaPattern::RGGB,
    };

    let storage = read_storage(
        tiff,
        ifd,
        &RawIfd::of(
            main.node,
            main.width,
            main.height,
            main.bits,
            main.samples,
            main.compression,
            match main.kind {
                DngKind::Cfa => PHOTOMETRIC_CFA,
                DngKind::LinearRaw => PHOTOMETRIC_LINEAR_RAW,
            },
        ),
    )?;
    check_storage(&storage, &main)?;

    // Active area, then everything relative to it.
    let readout = Rect {
        x: 0,
        y: 0,
        width: main.width,
        height: main.height,
    };
    let active = match uints(tiff, dirs.entry(tags::ACTIVE_AREA), 4).as_deref() {
        Some(&[top, left, bottom, right]) => {
            let to = |v: u64| u32::try_from(v).ok();
            match (to(top), to(left), to(bottom), to(right)) {
                (Some(t), Some(l), Some(b), Some(r)) if b > t && r > l => Some(Rect {
                    x: l,
                    y: t,
                    width: r - l,
                    height: b - t,
                }),
                _ => return corrupt("the active area is damaged"),
            }
        }
        _ => None,
    };
    if let Some(a) = active
        && !a.fits_in(main.width, main.height)
    {
        return corrupt("the active area lies outside the image");
    }
    let area = active.unwrap_or(readout);
    let (ax, ay) = (area.x, area.y);

    let crop = default_crop(tiff, &dirs, area)?;
    // The pattern of the readout: the declared one seen from the readout's own top left corner.
    let cfa = shifted_pattern(declared_cfa, ax & 1, ay & 1);

    let (black, white, black_source, white_source, delta_h, delta_v, table) =
        levels(tiff, &dirs, &main, area, cfa)?;

    // Colour.
    let color = |t: u16| dirs.entry(t);
    let neutral = triple(tiff, color(tags::AS_SHOT_NEUTRAL));
    let white_xy =
        floats(tiff, color(tags::AS_SHOT_WHITE_XY), 2).and_then(|v| <[f64; 2]>::try_from(v).ok());
    let mut info = DngInfo {
        version,
        kind: main.kind,
        big_endian: tiff.order() == ByteOrder::Big,
        unique_camera_model: root
            .get(tags::UNIQUE_CAMERA_MODEL)
            .and_then(|e| tiff.ascii(e).ok())
            .filter(|s| !s.is_empty()),
        color_matrix: [
            matrix9(tiff, color(tags::COLOR_MATRIX_1)),
            matrix9(tiff, color(tags::COLOR_MATRIX_2)),
        ],
        calibration_illuminant: [
            first_uint(tiff, color(tags::CALIBRATION_ILLUMINANT_1))
                .and_then(|v| u16::try_from(v).ok()),
            first_uint(tiff, color(tags::CALIBRATION_ILLUMINANT_2))
                .and_then(|v| u16::try_from(v).ok()),
        ],
        camera_calibration: [
            matrix9(tiff, color(tags::CAMERA_CALIBRATION_1)),
            matrix9(tiff, color(tags::CAMERA_CALIBRATION_2)),
        ],
        analog_balance: triple(tiff, color(tags::ANALOG_BALANCE)),
        as_shot_neutral: neutral.filter(|n| n.iter().all(|v| *v > 0.0)),
        as_shot_white_xy: white_xy.filter(|xy| xy[1] > 0.0),
        forward_matrix: [
            matrix9(tiff, color(tags::FORWARD_MATRIX_1)),
            matrix9(tiff, color(tags::FORWARD_MATRIX_2)),
        ],
        noise_profile: floats(tiff, color(tags::NOISE_PROFILE), 16)
            .filter(|v| !v.is_empty() && v.len() % 2 == 0),
        linearization: table,
        black_delta_h: delta_h,
        black_delta_v: delta_v,
        gain_maps: Vec::new(),
    };
    info.gain_maps = gain_maps(tiff, &dirs, &main, area)?;

    let make = exif.make.clone().unwrap_or_default();
    let model = exif
        .model
        .clone()
        .or_else(|| {
            info.unique_camera_model.as_deref().map(|u| {
                u.strip_prefix(make.as_str())
                    .map_or(u, str::trim)
                    .to_owned()
            })
        })
        .unwrap_or_default();
    let (iso, iso_source) = exif::resolve_iso(None, &exif.iso).unzip();

    let decode = match main.kind {
        DngKind::Cfa => DecodeKind::DngCfa,
        DngKind::LinearRaw => DecodeKind::DngLinear,
    };
    let wb_as_shot = as_shot_gains(&info, main.kind, cfa);

    Ok(RawMetadata {
        make,
        model,
        model_id: None,
        arw_version: None,
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
        orientation: exif.orientation.unwrap_or(1),
        width: main.width,
        height: main.height,
        bits_per_sample: main.bits,
        samples_per_pixel: main.samples,
        compression: main.compression,
        photometric: match main.kind {
            DngKind::Cfa => PHOTOMETRIC_CFA,
            DngKind::LinearRaw => PHOTOMETRIC_LINEAR_RAW,
        },
        raw_file_type: None,
        decode,
        storage,
        cfa,
        active_area: active,
        crop,
        black,
        white,
        black_source,
        white_source,
        wb_as_shot,
        wb_source: wb_as_shot.map(|_| LevelSource::RawIfd),
        lens_corrections: None,
        tone_curve: None,
        preview: find_preview(tiff, nodes, None),
        gps: exif.gps,
        focus: None,
        af_area_setting: None,
        dng: Some(info),
    })
}

/// The `CFAPattern` as the file writes it, through `CFAPlaneColor`; only a 2 x 2 Bayer quad passes.
fn declared_pattern(tiff: &Tiff, ifd: &Ifd) -> Result<CfaPattern, RawError> {
    if let Some(layout) = first_uint(tiff, ifd.get(tags::CFA_LAYOUT))
        && layout != 1
    {
        return refuse(DngRefusal::NotBayer);
    }
    let dim = uints(tiff, ifd.get(tags::CFA_REPEAT_PATTERN_DIM), 2).unwrap_or_default();
    if dim != [2, 2] {
        return refuse(DngRefusal::NotBayer);
    }
    let planes = uints(tiff, ifd.get(tags::CFA_PLANE_COLOR), 8).unwrap_or_else(|| vec![0, 1, 2]);
    let planes = if planes.is_empty() {
        vec![0, 1, 2]
    } else {
        planes
    };
    let codes = uints(tiff, ifd.get(tag::CFA_PATTERN), 4).unwrap_or_default();
    if codes.len() != 4 {
        return refuse(DngRefusal::NotBayer);
    }
    let mut mapped = [0u8; 4];
    for (slot, code) in mapped.iter_mut().zip(&codes) {
        let Some(color) = usize::try_from(*code).ok().and_then(|i| planes.get(i)) else {
            return refuse(DngRefusal::NotBayer);
        };
        let Ok(color) = u8::try_from(*color) else {
            return refuse(DngRefusal::NotBayer);
        };
        *slot = color;
    }
    CfaPattern::from_tiff_codes(mapped).map_or_else(|| refuse(DngRefusal::NotBayer), Ok)
}

/// The pattern of the readout when the active area starts at an odd column (`dx`) or row (`dy`).
fn shifted_pattern(declared: CfaPattern, dx: u32, dy: u32) -> CfaPattern {
    let (dx, dy) = (dx as usize, dy as usize);
    let at = |py: usize, px: usize| declared.0[((py ^ dy) * 2) + (px ^ dx)];
    CfaPattern([at(0, 0), at(0, 1), at(1, 0), at(1, 1)])
}

/// `DefaultCropOrigin` and `DefaultCropSize` (relative to the active area) as a readout rectangle.
fn default_crop(tiff: &Tiff, dirs: &Dirs, area: Rect) -> Result<Option<Rect>, RawError> {
    let (Some(origin), Some(size)) = (
        floats(tiff, dirs.entry(tag::DEFAULT_CROP_ORIGIN), 2),
        floats(tiff, dirs.entry(tag::DEFAULT_CROP_SIZE), 2),
    ) else {
        return Ok(None);
    };
    if origin.len() != 2 || size.len() != 2 {
        return corrupt("the default crop is damaged");
    }
    // The crop must be whole pixels; a fractional one is rounded to the pixel grid.
    let whole = |v: f64| {
        (v.is_finite() && (0.0..=f64::from(u32::MAX)).contains(&v)).then(|| v.round() as u32)
    };
    let (Some(ox), Some(oy), Some(w), Some(h)) = (
        whole(origin[0]),
        whole(origin[1]),
        whole(size[0]),
        whole(size[1]),
    ) else {
        return corrupt("the default crop is damaged");
    };
    let rect = Rect {
        x: area.x.saturating_add(ox),
        y: area.y.saturating_add(oy),
        width: w,
        height: h,
    };
    // A crop that does not lie inside the active area is ignored; the active area is used.
    let inside = rect.width > 0
        && rect.height > 0
        && ox.checked_add(w).is_some_and(|r| r <= area.width)
        && oy.checked_add(h).is_some_and(|b| b <= area.height);
    Ok(inside.then_some(rect))
}

/// Black and white levels, the black deltas and the linearisation table.
#[allow(clippy::type_complexity)]
fn levels(
    tiff: &Tiff,
    dirs: &Dirs,
    main: &MainImage,
    area: Rect,
    cfa: CfaPattern,
) -> Result<
    (
        [f32; 4],
        [f32; 4],
        LevelSource,
        LevelSource,
        Vec<f32>,
        Vec<f32>,
        Option<Vec<u16>>,
    ),
    RawError,
> {
    let spp = usize::from(main.samples);
    let table = match dirs.entry(tags::LINEARIZATION_TABLE) {
        Some(entry) => {
            let values = uints(tiff, Some(entry), 1 << 16)
                .ok_or_else(|| RawError::corrupt("the linearisation table is damaged"))?;
            if values.is_empty() || values.iter().any(|v| *v > u64::from(u16::MAX)) {
                return corrupt("the linearisation table is damaged");
            }
            Some(values.into_iter().map(|v| v as u16).collect::<Vec<u16>>())
        }
        None => None,
    };

    // WhiteLevel: one value per sample; the default is the largest value the table or the bit depth
    // can give.
    let default_white = match &table {
        Some(t) => f64::from(t.iter().copied().max().unwrap_or(u16::MAX)),
        None => f64::from((1u32 << main.bits) - 1),
    };
    let white_entry = dirs.entry(tags::WHITE_LEVEL);
    let white_values = floats(tiff, white_entry, spp)
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![default_white]);
    let white_for = |sample: usize| {
        white_values
            .get(sample)
            .or(white_values.first())
            .copied()
            .unwrap_or(default_white)
    };

    // BlackLevel with its repeat pattern.
    let repeat = uints(tiff, dirs.entry(tags::BLACK_LEVEL_REPEAT_DIM), 2).unwrap_or_default();
    let (rr, rc) = match repeat[..] {
        [r, c] if (1..=16).contains(&r) && (1..=16).contains(&c) => (r as usize, c as usize),
        [] => (1, 1),
        _ => return corrupt("the black level pattern is damaged"),
    };
    let black_values = floats(tiff, dirs.entry(tags::BLACK_LEVEL), rr * rc * spp);
    let black_source = if black_values.is_some() {
        LevelSource::RawIfd
    } else {
        LevelSource::Default
    };
    let black_values = black_values.unwrap_or_else(|| vec![0.0; rr * rc * spp]);
    if black_values.len() < rr * rc * spp {
        // A short list is the first value for every entry (`BlackLevel` of one number).
        if black_values.len() != 1 && black_values.len() != spp {
            return corrupt("the black level list is too short");
        }
    }
    let black_at = |i: usize, j: usize, s: usize| -> f64 {
        let n = black_values.len();
        let index = if n >= rr * rc * spp {
            (i * rc + j) * spp + s
        } else if n == spp {
            s
        } else {
            0
        };
        black_values.get(index).copied().unwrap_or(0.0)
    };

    let mut black = [0f32; 4];
    let mut white = [0f32; 4];
    match main.kind {
        DngKind::Cfa => {
            for py in 0..2usize {
                for px in 0..2usize {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for i in 0..rr {
                        for j in 0..rc {
                            let row_ok = rr % 2 == 1 || ((area.y as usize + i) & 1) == py;
                            let col_ok = rc % 2 == 1 || ((area.x as usize + j) & 1) == px;
                            if row_ok && col_ok {
                                sum += black_at(i, j, 0);
                                count += 1.0;
                            }
                        }
                    }
                    let p = py * 2 + px;
                    black[p] = if count > 0.0 {
                        (sum / count) as f32
                    } else {
                        0.0
                    };
                    white[p] = white_for(0) as f32;
                }
            }
        }
        DngKind::LinearRaw => {
            // Red, green, green, blue.
            let mut per_channel = [0f32; 3];
            for (s, slot) in per_channel.iter_mut().enumerate() {
                let mut sum = 0.0;
                for i in 0..rr {
                    for j in 0..rc {
                        sum += black_at(i, j, s);
                    }
                }
                *slot = (sum / (rr * rc) as f64) as f32;
            }
            black = [
                per_channel[0],
                per_channel[1],
                per_channel[1],
                per_channel[2],
            ];
            let w = [white_for(0), white_for(1), white_for(2)].map(|v| v as f32);
            white = [w[0], w[1], w[1], w[2]];
        }
    }
    let _ = cfa;
    if black
        .iter()
        .zip(&white)
        .any(|(b, w)| !(b.is_finite() && w.is_finite() && b < w && *b >= 0.0))
    {
        return corrupt("the black level is not below the white level");
    }

    // Per-column and per-row deltas.
    let delta = |tag_ref: u16, len: u32| -> Result<Vec<f32>, RawError> {
        match dirs.entry(tag_ref) {
            None => Ok(Vec::new()),
            Some(entry) => {
                if entry.count != len {
                    return corrupt("a black level delta list does not match the active area");
                }
                let values = floats(tiff, Some(entry), len as usize)
                    .ok_or_else(|| RawError::corrupt("a black level delta list is damaged"))?;
                Ok(values.into_iter().map(|v| v as f32).collect())
            }
        }
    };
    let delta_h = delta(tags::BLACK_LEVEL_DELTA_H, area.width)?;
    let delta_v = delta(tags::BLACK_LEVEL_DELTA_V, area.height)?;
    Ok((
        black,
        white,
        black_source,
        if white_entry.is_some() {
            LevelSource::RawIfd
        } else {
            LevelSource::Default
        },
        delta_h,
        delta_v,
        table,
    ))
}

/// Checks the strips or tiles against the image and the format (before any pixel is read).
fn check_storage(storage: &RawStorage, main: &MainImage) -> Result<(), RawError> {
    let spp = u64::from(main.samples);
    let (offsets, counts) = storage.chunks();
    let width = u64::from(main.width);
    let height = u64::from(main.height);
    if main.compression == 7 {
        // A lossless JPEG spends at least one bit on every sample, so a file that is much smaller
        // than that cannot hold the image it claims, and nothing is allocated for it.
        let total: u64 = counts.iter().sum();
        if total.saturating_mul(16) < width * height * spp {
            return corrupt(crate::error::CUT_SHORT);
        }
    }
    match storage {
        RawStorage::Strips { rows_per_strip, .. } => {
            let rps = u64::from(*rows_per_strip).max(1);
            if offsets.len() as u64 != height.div_ceil(rps) {
                return corrupt("the number of strips does not match the image");
            }
            if main.compression == 1 {
                let row_bytes = (width * spp * u64::from(main.bits)).div_ceil(8);
                for (i, count) in counts.iter().enumerate() {
                    let rows = rps.min(height - (i as u64) * rps);
                    if *count < rows * row_bytes {
                        return corrupt(
                            "an uncompressed strip holds fewer bytes than its rows need",
                        );
                    }
                }
            }
        }
        RawStorage::Tiles {
            tile_width,
            tile_length,
            ..
        } => {
            let (tw, tl) = (u64::from(*tile_width), u64::from(*tile_length));
            if tw > 16384 || tl > 16384 || tw * tl * spp > (1 << 26) {
                return corrupt("the tile size is out of range");
            }
            if main.compression == 1 {
                let tile_bytes = tl * (tw * spp * u64::from(main.bits)).div_ceil(8);
                if counts.iter().any(|c| *c < tile_bytes) {
                    return corrupt("an uncompressed tile holds fewer bytes than it needs");
                }
            }
        }
    }
    Ok(())
}

/// The as-shot white balance as gains per CFA position (or red, green, green, blue for linear
/// raw), the form `RawImage::wb_as_shot` has: a white surface records `1 / gain`. Green is
/// about 1024, as the Sony bodies write it. `None` when the file has neither `AsShotNeutral` nor a
/// usable `AsShotWhiteXY`.
fn as_shot_gains(info: &DngInfo, kind: DngKind, cfa: CfaPattern) -> Option<[f32; 4]> {
    let neutral = crate::dng_color::as_shot_neutral(info)?;
    let rggb = [
        1.0 / neutral[0],
        1.0 / neutral[1],
        1.0 / neutral[1],
        1.0 / neutral[2],
    ];
    let scale = 1024.0 / rggb[1];
    let rggb = rggb.map(|g| (g * scale) as f32);
    if !rggb.iter().all(|g| g.is_finite() && *g > 0.0) {
        return None;
    }
    Some(match kind {
        DngKind::Cfa => cfa.rggb_to_positions(rggb),
        DngKind::LinearRaw => rggb,
    })
}

// ---------------------------------------------------------------------------------------------
// OpcodeList2
// ---------------------------------------------------------------------------------------------

fn be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn be_f64(bytes: &[u8], at: usize) -> Option<f64> {
    Some(f64::from_be_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

fn be_f32(bytes: &[u8], at: usize) -> Option<f32> {
    Some(f32::from_be_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Reads `OpcodeList2`: the gain maps. Opcodes are big endian whatever the file's byte order.
fn gain_maps(
    tiff: &Tiff,
    dirs: &Dirs,
    main: &MainImage,
    area: Rect,
) -> Result<Vec<GainMap>, RawError> {
    let Some(entry) = dirs.entry(tags::OPCODE_LIST_2) else {
        return Ok(Vec::new());
    };
    let bytes = tiff
        .entry_bytes(entry)
        .map_err(|_| RawError::corrupt("the opcode list is damaged"))?;
    let bad = || RawError::corrupt("the opcode list is damaged");
    let count = be_u32(&bytes, 0).ok_or_else(bad)? as usize;
    if count > MAX_OPCODES {
        return corrupt("the opcode list is too long");
    }
    let mut at = 4usize;
    let mut maps = Vec::new();
    for _ in 0..count {
        let id = be_u32(&bytes, at).ok_or_else(bad)?;
        let flags = be_u32(&bytes, at + 8).ok_or_else(bad)?;
        let length = be_u32(&bytes, at + 12).ok_or_else(bad)? as usize;
        let start = at.checked_add(16).ok_or_else(bad)?;
        let end = start.checked_add(length).ok_or_else(bad)?;
        let params = bytes.get(start..end).ok_or_else(bad)?;
        at = end;
        if id != OPCODE_GAIN_MAP {
            // Bit 0 marks an opcode a reader may skip. One it may not skip changes the picture.
            if flags & 1 == 0 {
                return refuse(DngRefusal::Opcode(id));
            }
            continue;
        }
        maps.push(parse_gain_map(params, main, area)?);
    }
    Ok(maps)
}

fn parse_gain_map(p: &[u8], main: &MainImage, area: Rect) -> Result<GainMap, RawError> {
    let bad = || RawError::corrupt("a gain map is damaged");
    let u = |i: usize| be_u32(p, i * 4).ok_or_else(bad);
    let (top, left, bottom, right) = (u(0)?, u(1)?, u(2)?, u(3)?);
    let (plane, planes, row_pitch, col_pitch) = (u(4)?, u(5)?, u(6)?, u(7)?);
    let (points_v, points_h) = (u(8)?, u(9)?);
    let spacing_v = be_f64(p, 40).ok_or_else(bad)?;
    let spacing_h = be_f64(p, 48).ok_or_else(bad)?;
    let origin_v = be_f64(p, 56).ok_or_else(bad)?;
    let origin_h = be_f64(p, 64).ok_or_else(bad)?;
    let map_planes = be_u32(p, 72).ok_or_else(bad)?;
    let total = u64::from(points_v) * u64::from(points_h) * u64::from(map_planes);
    if points_v == 0
        || points_h == 0
        || map_planes == 0
        || total > MAX_MAP_POINTS
        || row_pitch == 0
        || col_pitch == 0
        || planes == 0
        || top >= bottom
        || left >= right
        || bottom > area.height
        || right > area.width
        || plane
            .checked_add(planes)
            .is_none_or(|end| end > u32::from(main.samples))
        || !(map_planes == 1 || map_planes >= planes)
        || ![spacing_v, spacing_h, origin_v, origin_h]
            .iter()
            .all(|v| v.is_finite())
    {
        return Err(bad());
    }
    let mut gains = Vec::with_capacity(total as usize);
    for i in 0..total as usize {
        let g = be_f32(p, 76 + i * 4).ok_or_else(bad)?;
        if !g.is_finite() || g < 0.0 {
            return Err(bad());
        }
        gains.push(g);
    }
    Ok(GainMap {
        top,
        left,
        bottom,
        right,
        plane,
        planes,
        row_pitch,
        col_pitch,
        points_v,
        points_h,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        map_planes,
        gains,
    })
}

impl GainMap {
    /// The map's gain for map plane `plane` at position `(v, h)` in the area's own unit square
    /// (0 at the first row or column of the area, 1 at the last): bilinear between the four
    /// nearest map points, the nearest map point outside the map.
    pub fn gain_at(&self, map_plane: usize, v: f64, h: f64) -> f32 {
        let axis = |pos: f64, origin: f64, spacing: f64, points: u32| -> (usize, usize, f32) {
            let last = points as usize - 1;
            if points == 1 || spacing <= 0.0 {
                return (0, 0, 0.0);
            }
            let t = ((pos - origin) / spacing).clamp(0.0, last as f64);
            let i0 = (t.floor() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            (i0, i1, (t - i0 as f64) as f32)
        };
        let (r0, r1, fr) = axis(v, self.origin_v, self.spacing_v, self.points_v);
        let (c0, c1, fc) = axis(h, self.origin_h, self.spacing_h, self.points_h);
        let planes = self.map_planes as usize;
        let at = |r: usize, c: usize| {
            self.gains
                .get((r * self.points_h as usize + c) * planes + map_plane.min(planes - 1))
                .copied()
                .unwrap_or(1.0)
        };
        let top = at(r0, c0) * (1.0 - fc) + at(r0, c1) * fc;
        let bottom = at(r1, c0) * (1.0 - fc) + at(r1, c1) * fc;
        top * (1.0 - fr) + bottom * fr
    }
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

/// How the strips or tiles tile the image: bands of rows, and in each band `across` chunks.
struct Grid {
    /// Rows of one band (rows per strip, or tile length).
    band_rows: usize,
    /// Width of one chunk in pixels (the image width for strips, the tile width for tiles).
    chunk_width: usize,
    /// Chunks across (1 for strips).
    across: usize,
}

fn grid_of(meta: &RawMetadata) -> Result<Grid, RawError> {
    let width = meta.width as usize;
    match &meta.storage {
        RawStorage::Strips { rows_per_strip, .. } => Ok(Grid {
            band_rows: (*rows_per_strip as usize).max(1),
            chunk_width: width,
            across: 1,
        }),
        RawStorage::Tiles {
            tile_width,
            tile_length,
            ..
        } => {
            let (tw, tl) = (*tile_width as usize, *tile_length as usize);
            if tw == 0 || tl == 0 {
                return corrupt("the tile size is out of range");
            }
            Ok(Grid {
                band_rows: tl,
                chunk_width: tw,
                across: width.div_ceil(tw),
            })
        }
    }
}

/// Decodes the raw image of a DNG: the interleaved samples (`3 * width` per row for linear raw),
/// linearised, with the black level deltas removed and the gain maps applied.
pub fn decode(src: &dyn ByteSource, meta: &RawMetadata) -> Result<Plane<u16>, RawError> {
    let info = meta
        .dng
        .as_ref()
        .ok_or_else(|| RawError::corrupt("the file is not a DNG"))?;
    let spp = usize::from(meta.samples_per_pixel);
    let (width, height) = (meta.width as usize, meta.height as usize);
    if (width as u64) * (height as u64) * (spp as u64) > MAX_SAMPLES || width == 0 || height == 0 {
        return corrupt("the image size is out of range");
    }
    let grid = grid_of(meta)?;
    let (offsets, counts) = meta.storage.chunks();
    let row_samples = width * spp;
    let mut data = vec![0u16; row_samples * height];
    let chunk_samples_limit = grid.chunk_width * spp * grid.band_rows;
    let byte_limit = (chunk_samples_limit as u64) * 8 + (1 << 16);
    let params = ChunkFormat {
        bits: meta.bits_per_sample,
        spp,
        compression: meta.compression,
        big_endian: info.big_endian,
    };

    data.par_chunks_mut(row_samples * grid.band_rows)
        .enumerate()
        .try_for_each(|(band, out)| -> Result<(), RawError> {
            let rows = out.len() / row_samples;
            let chunks: Vec<Vec<u16>> = (0..grid.across)
                .into_par_iter()
                .map(|across| {
                    let k = band * grid.across + across;
                    let (Some(&offset), Some(&count)) = (offsets.get(k), counts.get(k)) else {
                        return corrupt("a strip or tile is missing");
                    };
                    let bytes = ljpeg::read_tile(src, offset, count, byte_limit)?;
                    // A strip holds the rows left in the image; a tile always holds a whole tile.
                    let chunk_rows = match meta.storage {
                        RawStorage::Strips { .. } => rows,
                        RawStorage::Tiles { .. } => grid.band_rows,
                    };
                    decode_chunk(&bytes, grid.chunk_width, chunk_rows, &params)
                })
                .collect::<Result<_, _>>()?;
            for (across, chunk) in chunks.iter().enumerate() {
                let x0 = across * grid.chunk_width;
                let copy = grid.chunk_width.min(width - x0) * spp;
                let stride = grid.chunk_width * spp;
                for row in 0..rows {
                    let from = row * stride;
                    let to = row * row_samples + x0 * spp;
                    out[to..to + copy].copy_from_slice(&chunk[from..from + copy]);
                }
            }
            Ok(())
        })?;

    finish_samples(&mut data, meta, info);
    Plane::from_vec(row_samples, height, data)
        .ok_or_else(|| RawError::corrupt("the decoded image has the wrong size"))
}

struct ChunkFormat {
    bits: u16,
    spp: usize,
    compression: u16,
    big_endian: bool,
}

/// Decodes one strip or tile into `rows` rows of `width * spp` samples.
fn decode_chunk(
    bytes: &[u8],
    width: usize,
    rows: usize,
    format: &ChunkFormat,
) -> Result<Vec<u16>, RawError> {
    let row_samples = width * format.spp;
    let mut out = vec![0u16; row_samples * rows];
    if format.compression == 1 {
        let row_bytes = (row_samples * usize::from(format.bits)).div_ceil(8);
        if bytes.len() < row_bytes * rows {
            return corrupt(crate::error::CUT_SHORT);
        }
        for (row, dest) in out.chunks_exact_mut(row_samples).enumerate() {
            unpack_row(
                &bytes[row * row_bytes..(row + 1) * row_bytes],
                format.bits,
                format.big_endian,
                dest,
            );
        }
        return Ok(out);
    }

    // Lossless JPEG. Check the declared geometry before anything is allocated.
    let header = ljpeg::parse_header(bytes)?;
    let components = header.components.len();
    if !(1..=ljpeg::MAX_COMPONENTS).contains(&components)
        || header.components.iter().any(|c| c.h != 1 || c.v != 1)
        || header.width.checked_mul(components) != Some(row_samples)
        || header.height < rows
        || !(2..=16).contains(&header.precision)
    {
        return corrupt("a lossless JPEG does not match its strip or tile");
    }
    let options = ljpeg::Options {
        max_samples: row_samples * header.height + 64,
        sony_420_quirk: false,
    };
    let decoded = ljpeg::decode_scan(&header, bytes, &options)?;
    if decoded.planes.len() != components {
        return corrupt("a lossless JPEG lost a component");
    }
    for (row, dest) in out.chunks_exact_mut(row_samples).enumerate() {
        for (c, plane) in decoded.planes.iter().enumerate() {
            let src = plane
                .data
                .get(row * plane.width..row * plane.width + header.width)
                .ok_or_else(|| RawError::corrupt("a lossless JPEG is shorter than it says"))?;
            for (x, v) in src.iter().enumerate() {
                dest[x * components + c] = *v;
            }
        }
    }
    Ok(out)
}

/// Unpacks one row of uncompressed samples of `bits` bits (8 to 16): whole bytes for 8 and 16 bits
/// (16 bits in the file's byte order), most significant bit first for the others, as TIFF 6.0 packs
/// them. A row starts on a byte boundary.
fn unpack_row(row: &[u8], bits: u16, big_endian: bool, out: &mut [u16]) {
    match bits {
        8 => {
            for (o, b) in out.iter_mut().zip(row) {
                *o = u16::from(*b);
            }
        }
        16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = if big_endian {
                    u16::from_be_bytes([b[0], b[1]])
                } else {
                    u16::from_le_bytes([b[0], b[1]])
                };
            }
        }
        _ => {
            let bits = u32::from(bits);
            let mask = (1u32 << bits) - 1;
            let mut acc = 0u32;
            let mut have = 0u32;
            let mut bytes = row.iter();
            for o in out.iter_mut() {
                while have < bits {
                    // The row is long enough (checked by the caller), so a missing byte reads as 0.
                    acc = (acc << 8) | u32::from(*bytes.next().unwrap_or(&0));
                    have += 8;
                }
                have -= bits;
                *o = ((acc >> have) & mask) as u16;
                acc &= (1 << have) - 1;
            }
        }
    }
}

/// Everything that happens to the samples after they are read: the linearisation table, the black
/// level deltas and the gain maps, in the order the specification gives.
fn finish_samples(data: &mut [u16], meta: &RawMetadata, info: &DngInfo) {
    let spp = usize::from(meta.samples_per_pixel);
    let width = meta.width as usize;
    let row_samples = width * spp;
    let area = meta.active_area.unwrap_or(Rect {
        x: 0,
        y: 0,
        width: meta.width,
        height: meta.height,
    });
    let (ax, ay) = (area.x as usize, area.y as usize);
    let (aw, ah) = (area.width as usize, area.height as usize);
    let has_deltas = !info.black_delta_h.is_empty() || !info.black_delta_v.is_empty();
    if info.linearization.is_none() && !has_deltas && info.gain_maps.is_empty() {
        return;
    }
    // Black level of each sample of a pixel at readout position (x, y), for the gain maps.
    let black_of = |x: usize, y: usize, s: usize| -> f32 {
        match meta.decode {
            DecodeKind::DngLinear => meta.black[[0, 1, 3][s.min(2)]],
            _ => meta.black[(y & 1) * 2 + (x & 1)],
        }
    };
    let white_of = |x: usize, y: usize, s: usize| -> f32 {
        match meta.decode {
            DecodeKind::DngLinear => meta.white[[0, 1, 3][s.min(2)]],
            _ => meta.white[(y & 1) * 2 + (x & 1)],
        }
    };
    data.par_chunks_mut(row_samples)
        .enumerate()
        .for_each(|(y, row)| {
            if let Some(table) = &info.linearization {
                let last = table.len() - 1; // the table is never empty (checked at probe)
                for v in row.iter_mut() {
                    *v = table[usize::from(*v).min(last)];
                }
            }
            let in_area_row = y >= ay && y < ay + ah;
            if has_deltas && in_area_row {
                let dv = info.black_delta_v.get(y - ay).copied().unwrap_or(0.0);
                for x in ax..(ax + aw).min(width) {
                    let dh = info.black_delta_h.get(x - ax).copied().unwrap_or(0.0);
                    let delta = (dv + dh).round();
                    for s in 0..spp {
                        let v = &mut row[x * spp + s];
                        *v = (f32::from(*v) - delta).clamp(0.0, 65535.0) as u16;
                    }
                }
            }
            if in_area_row {
                for map in &info.gain_maps {
                    let ry = y - ay;
                    let ry = ry as u32;
                    if ry < map.top
                        || ry >= map.bottom
                        || !(ry - map.top).is_multiple_of(map.row_pitch)
                    {
                        continue;
                    }
                    let v_unit =
                        f64::from(ry - map.top) / f64::from((map.bottom - map.top - 1).max(1));
                    let mut cx = map.left;
                    while cx < map.right {
                        let h_unit =
                            f64::from(cx - map.left) / f64::from((map.right - map.left - 1).max(1));
                        let x = ax + cx as usize;
                        if x < width {
                            for k in 0..map.planes as usize {
                                let s = map.plane as usize + k;
                                let g = map.gain_at(
                                    if map.map_planes == 1 { 0 } else { k },
                                    v_unit,
                                    h_unit,
                                );
                                let black = black_of(x, y, s);
                                let white = white_of(x, y, s);
                                let sample = &mut row[x * spp + s];
                                let lifted = black + (f32::from(*sample) - black) * g;
                                *sample = lifted.round().clamp(0.0, white.max(black)) as u16;
                            }
                        }
                        cx += map.col_pitch;
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_row_reads_packed_bits_most_significant_first() {
        // Two 12-bit samples 0xABC and 0x123 in three bytes: AB C1 23.
        let mut out = [0u16; 2];
        unpack_row(&[0xAB, 0xC1, 0x23], 12, false, &mut out);
        assert_eq!(out, [0xABC, 0x123]);
        // Three 10-bit samples in four bytes: 0x3FF, 0x001, 0x200.
        let mut out = [0u16; 3];
        unpack_row(&[0xFF, 0xC0, 0x18, 0x00], 10, false, &mut out);
        assert_eq!(out, [0x3FF, 0x001, 0x200]);
        // 16 bits follow the byte order.
        let mut out = [0u16; 1];
        unpack_row(&[0x12, 0x34], 16, true, &mut out);
        assert_eq!(out, [0x1234]);
        unpack_row(&[0x12, 0x34], 16, false, &mut out);
        assert_eq!(out, [0x3412]);
    }

    #[test]
    fn the_declared_pattern_is_seen_from_the_readout() {
        let rggb = CfaPattern::RGGB;
        assert_eq!(shifted_pattern(rggb, 0, 0), CfaPattern::RGGB);
        assert_eq!(shifted_pattern(rggb, 1, 0), CfaPattern::GRBG);
        assert_eq!(shifted_pattern(rggb, 0, 1), CfaPattern::GBRG);
        assert_eq!(shifted_pattern(rggb, 1, 1), CfaPattern::BGGR);
    }

    #[test]
    fn a_gain_map_is_bilinear_and_clamped() {
        let map = GainMap {
            top: 0,
            left: 0,
            bottom: 10,
            right: 10,
            plane: 0,
            planes: 1,
            row_pitch: 1,
            col_pitch: 1,
            points_v: 2,
            points_h: 2,
            spacing_v: 1.0,
            spacing_h: 1.0,
            origin_v: 0.0,
            origin_h: 0.0,
            map_planes: 1,
            gains: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(map.gain_at(0, 0.0, 0.0), 1.0);
        assert_eq!(map.gain_at(0, 0.0, 1.0), 2.0);
        assert_eq!(map.gain_at(0, 1.0, 0.0), 3.0);
        assert_eq!(map.gain_at(0, 1.0, 1.0), 4.0);
        assert_eq!(map.gain_at(0, 0.5, 0.5), 2.5);
        assert_eq!(map.gain_at(0, -1.0, 2.0), 2.0);
    }

    #[test]
    fn dng_messages_are_one_or_two_sentences() {
        for why in [
            DngRefusal::LossyJpeg,
            DngRefusal::JpegXl,
            DngRefusal::Deflate,
            DngRefusal::Compression(9),
            DngRefusal::FloatData,
            DngRefusal::NotBayer,
            DngRefusal::SampleLayout,
            DngRefusal::Scaled,
            DngRefusal::NotRaw,
            DngRefusal::Opcode(3),
            DngRefusal::NewerVersion,
        ] {
            let text = why.message();
            assert!(text.starts_with("This DNG"), "{text}");
            let sentences = text.matches(". ").count() + 1;
            assert!((1..=2).contains(&sentences), "{text}");
            assert!(text.ends_with('.'), "{text}");
        }
        assert_eq!(
            DngRefusal::JpegXl.message(),
            "This DNG uses JPEG XL compression, which Yellowmoon can't read. Export it again from the camera or converter without it."
        );
    }
}
