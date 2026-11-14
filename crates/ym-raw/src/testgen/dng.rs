//! Test-only synthetic DNG writer.
//!
//! [`DngBuilder`] lays a DNG out the way Adobe's converter and the phone cameras do: IFD0 with the
//! colour tags, a SubIFD with the raw image (a CFA mosaic or linear raw, in strips or tiles,
//! uncompressed or as lossless JPEG), an EXIF directory and, optionally, a JPEG preview. It is written
//! independently of the reader in `dng` (the byte order, the packing and the opcode list are laid out
//! here from the DNG Specification 1.7 and TIFF 6.0). Every file is deterministic.
//!
//! ```text
//! header "II*\0" or "MM\0*", IFD0 at the end of the file
//! raw data chunks (strips or tiles)
//! raw IFD   NewSubFileType 0, size, bits, compression, photometric, strips or tiles, CFA tags,
//!           levels, ActiveArea, DefaultCrop, LinearizationTable, OpcodeList2
//! preview   NewSubFileType 1 with a JPEG strip (optional)
//! EXIF IFD  ExposureTime, FNumber, ISO, FocalLength, DateTimeOriginal, LensModel
//! IFD0      Make, Model, Orientation, SubIFDs, ExifIFD, DNGVersion, UniqueCameraModel, colour tags
//! ```

use crate::dng::{DngKind, GainMap};
use crate::layout::CfaPattern;
use crate::testgen::ljpeg_enc::{EncodeParams, HuffmanChoice, encode};

/// How the raw data is compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DngCompression {
    /// TIFF compression 1.
    None,
    /// TIFF compression 7: lossless JPEG with `components` components per stream (1 to 4, each
    /// holding every `components`-th column), the given predictor and optionally a restart
    /// interval in MCU rows.
    LosslessJpeg {
        /// Components per stream.
        components: usize,
        /// Predictor 1 to 7.
        predictor: u8,
    },
}

/// How the raw data is divided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DngStorage {
    /// Strips of this many rows.
    Strips(usize),
    /// Tiles of `(width, length)` samples.
    Tiles(usize, usize),
}

/// One field of a directory under construction.
struct Field {
    tag: u16,
    ty: u16,
    count: u32,
    data: Vec<u8>,
}

/// A synthetic DNG.
#[derive(Clone, Debug)]
pub struct DngBuilder {
    /// `Make`.
    pub make: String,
    /// `Model`.
    pub model: String,
    /// `UniqueCameraModel`.
    pub unique_model: Option<String>,
    /// Image width in pixels (the whole readout).
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Mosaic or linear raw.
    pub kind: DngKind,
    /// The `CFAPattern` as written (relative to the active area).
    pub cfa: CfaPattern,
    /// `BitsPerSample` (8 to 16).
    pub bits: u16,
    /// Compression.
    pub compression: DngCompression,
    /// Strips or tiles.
    pub storage: DngStorage,
    /// Write a big-endian file.
    pub big_endian: bool,
    /// `BlackLevelRepeatDim` (rows, columns); `None` writes no black level.
    pub black_repeat: Option<(u16, u16)>,
    /// `BlackLevel` values (`rows * columns * samples`).
    pub black: Vec<f64>,
    /// `BlackLevelDeltaH`.
    pub black_delta_h: Vec<f64>,
    /// `BlackLevelDeltaV`.
    pub black_delta_v: Vec<f64>,
    /// `WhiteLevel`; `None` writes none.
    pub white: Option<Vec<u32>>,
    /// `LinearizationTable`.
    pub linearization: Option<Vec<u16>>,
    /// `ActiveArea` as top, left, bottom, right.
    pub active_area: Option<[u32; 4]>,
    /// `DefaultCropOrigin` and `DefaultCropSize` (relative to the active area).
    pub crop: Option<([u32; 2], [u32; 2])>,
    /// EXIF orientation.
    pub orientation: u16,
    /// `ColorMatrix1` and `2`.
    pub color_matrix: [Option<[f64; 9]>; 2],
    /// `CalibrationIlluminant1` and `2`.
    pub illuminant: [Option<u16>; 2],
    /// `CameraCalibration1` and `2`.
    pub camera_calibration: [Option<[f64; 9]>; 2],
    /// `AnalogBalance`.
    pub analog_balance: Option<[f64; 3]>,
    /// `AsShotNeutral`.
    pub as_shot_neutral: Option<[f64; 3]>,
    /// `AsShotWhiteXY`.
    pub as_shot_white_xy: Option<[f64; 2]>,
    /// `BaselineExposure` (written to show the reader ignores it).
    pub baseline_exposure: Option<f64>,
    /// `NoiseProfile`.
    pub noise_profile: Option<Vec<f64>>,
    /// The gain maps of `OpcodeList2`.
    pub gain_maps: Vec<GainMap>,
    /// Extra raw opcodes appended to `OpcodeList2`: `(id, flags, parameters)`.
    pub extra_opcodes: Vec<(u32, u32, Vec<u8>)>,
    /// `OpcodeList1` and `OpcodeList3` bytes written as they are (the reader ignores both).
    pub other_opcode_lists: [Option<Vec<u8>>; 2],
    /// A JPEG for the preview directory.
    pub preview: Option<Vec<u8>>,
    /// ISO.
    pub iso: Option<u32>,
    /// Exposure time as a fraction.
    pub exposure_time: Option<(u32, u32)>,
    /// F-number as a fraction.
    pub f_number: Option<(u32, u32)>,
    /// Focal length as a fraction.
    pub focal_length: Option<(u32, u32)>,
    /// Focal length in 35 mm terms.
    pub focal_length_35mm: Option<u16>,
    /// Lens model.
    pub lens_model: Option<String>,
    /// `Compression` written instead of the real one (a DNG that uses something we refuse).
    pub compression_tag: Option<u16>,
    /// `SampleFormat` to write.
    pub sample_format: Option<u16>,
    /// `DefaultScale` to write.
    pub default_scale: Option<[f64; 2]>,
    /// `CFARepeatPatternDim` to write instead of 2 x 2.
    pub cfa_dim: Option<[u16; 2]>,
    /// `PhotometricInterpretation` to write instead of the real one.
    pub photometric_tag: Option<u16>,
    /// `DNGBackwardVersion` to write.
    pub backward_version: Option<[u8; 4]>,
}

impl DngBuilder {
    /// A mosaic DNG with the usual tags: 12-bit samples, uncompressed, one strip, black 64,
    /// white 4095, a D65 colour matrix and a neutral.
    pub fn new(make: &str, model: &str, width: usize, height: usize) -> Self {
        Self {
            make: make.to_owned(),
            model: model.to_owned(),
            unique_model: Some(format!("{make} {model}")),
            width,
            height,
            kind: DngKind::Cfa,
            cfa: CfaPattern::RGGB,
            bits: 12,
            compression: DngCompression::None,
            storage: DngStorage::Strips(height),
            big_endian: false,
            black_repeat: Some((1, 1)),
            black: vec![64.0],
            black_delta_h: Vec::new(),
            black_delta_v: Vec::new(),
            white: Some(vec![4095]),
            linearization: None,
            active_area: None,
            crop: None,
            orientation: 1,
            color_matrix: [
                Some([0.8, -0.2, -0.1, -0.5, 1.4, 0.1, -0.1, 0.2, 0.6]),
                None,
            ],
            illuminant: [Some(21), None],
            camera_calibration: [None, None],
            analog_balance: None,
            as_shot_neutral: Some([0.5, 1.0, 0.7]),
            as_shot_white_xy: None,
            baseline_exposure: None,
            noise_profile: None,
            gain_maps: Vec::new(),
            extra_opcodes: Vec::new(),
            other_opcode_lists: [None, None],
            preview: None,
            iso: Some(100),
            exposure_time: Some((1, 125)),
            f_number: Some((28, 10)),
            focal_length: Some((35, 1)),
            focal_length_35mm: None,
            lens_model: None,
            compression_tag: None,
            sample_format: None,
            default_scale: None,
            cfa_dim: None,
            photometric_tag: None,
            backward_version: None,
        }
    }

    /// The samples per pixel of the image.
    pub fn samples_per_pixel(&self) -> usize {
        match self.kind {
            DngKind::Cfa => 1,
            DngKind::LinearRaw => 3,
        }
    }

    /// Builds the file. `samples` holds `width * height * samples_per_pixel` values below
    /// `2^bits`, row by row, the samples of a pixel together.
    pub fn build(&self, samples: &[u16]) -> Vec<u8> {
        let spp = self.samples_per_pixel();
        assert_eq!(
            samples.len(),
            self.width * self.height * spp,
            "sample count"
        );
        let mut file = Writer::new(self.big_endian);
        file.bytes
            .extend_from_slice(if self.big_endian { b"MM\0*" } else { b"II*\0" });
        file.bytes.extend_from_slice(&[0; 4]);

        // The raw data.
        let (chunks, grid) = self.chunks(samples);
        let mut offsets = Vec::new();
        let mut counts = Vec::new();
        for chunk in &chunks {
            file.align();
            offsets.push(file.bytes.len() as u32);
            counts.push(chunk.len() as u32);
            file.bytes.extend_from_slice(chunk);
        }

        let compression = self.compression_tag.unwrap_or(match self.compression {
            DngCompression::None => 1,
            DngCompression::LosslessJpeg { .. } => 7,
        });
        let photometric = self.photometric_tag.unwrap_or(match self.kind {
            DngKind::Cfa => 32803,
            DngKind::LinearRaw => 34892,
        });
        let mut raw = vec![
            file.long(0x00FE, &[0]),
            file.long(0x0100, &[self.width as u32]),
            file.long(0x0101, &[self.height as u32]),
            file.short(0x0102, &vec![self.bits; spp]),
            file.short(0x0103, &[compression]),
            file.short(0x0106, &[photometric]),
            file.short(0x0115, &[spp as u16]),
        ];
        match grid {
            Grid::Strips { rows } => {
                raw.push(file.long(0x0111, &offsets));
                raw.push(file.long(0x0116, &[rows as u32]));
                raw.push(file.long(0x0117, &counts));
            }
            Grid::Tiles { width, length } => {
                raw.push(file.long(0x0142, &[width as u32]));
                raw.push(file.long(0x0143, &[length as u32]));
                raw.push(file.long(0x0144, &offsets));
                raw.push(file.long(0x0145, &counts));
            }
        }
        if let Some(format) = self.sample_format {
            raw.push(file.short(0x0153, &vec![format; spp]));
        }
        if self.kind == DngKind::Cfa {
            let dim = self.cfa_dim.unwrap_or([2, 2]);
            raw.push(file.short(0x828D, &dim));
            let pattern = self.cfa.tiff_codes();
            raw.push(file.field(0x828E, 1, 4, pattern.to_vec()));
            raw.push(file.field(0xC616, 1, 3, vec![0, 1, 2]));
        }
        if let Some((rows, cols)) = self.black_repeat {
            raw.push(file.short(0xC619, &[rows, cols]));
            raw.push(file.rational(0xC61A, &self.black));
        }
        if !self.black_delta_h.is_empty() {
            raw.push(file.srational(0xC61B, &self.black_delta_h));
        }
        if !self.black_delta_v.is_empty() {
            raw.push(file.srational(0xC61C, &self.black_delta_v));
        }
        if let Some(white) = &self.white {
            raw.push(file.long(0xC61D, white));
        }
        if let Some(table) = &self.linearization {
            raw.push(file.short(0xC618, table));
        }
        if let Some(a) = self.active_area {
            raw.push(file.long(0xC68D, &a));
        }
        if let Some((origin, size)) = self.crop {
            raw.push(file.rational(0xC61F, &origin.map(f64::from)));
            raw.push(file.rational(0xC620, &size.map(f64::from)));
        }
        if let Some(scale) = self.default_scale {
            raw.push(file.rational(0xC61E, &scale));
        }
        if let Some(noise) = &self.noise_profile {
            raw.push(file.double(0xC761, noise));
        }
        let mut list2 = Vec::new();
        for map in &self.gain_maps {
            list2.push((9u32, 1u32, gain_map_bytes(map)));
        }
        list2.extend(self.extra_opcodes.iter().cloned());
        if !list2.is_empty() {
            raw.push(file.field(0xC741, 7, 0, opcode_list(&list2)));
        }
        if let Some(list) = &self.other_opcode_lists[0] {
            raw.push(file.field(0xC740, 7, 0, list.clone()));
        }
        if let Some(list) = &self.other_opcode_lists[1] {
            raw.push(file.field(0xC74E, 7, 0, list.clone()));
        }
        let raw_ifd = file.ifd(raw);

        // The preview directory.
        let preview_ifd = self.preview.as_ref().map(|jpeg| {
            file.align();
            let offset = file.bytes.len() as u32;
            file.bytes.extend_from_slice(jpeg);
            let fields = vec![
                file.long(0x00FE, &[1]),
                file.long(0x0100, &[16]),
                file.long(0x0101, &[16]),
                file.short(0x0102, &[8, 8, 8]),
                file.short(0x0103, &[7]),
                file.short(0x0106, &[6]),
                file.long(0x0111, &[offset]),
                file.short(0x0115, &[3]),
                file.long(0x0117, &[jpeg.len() as u32]),
            ];
            file.ifd(fields)
        });

        // EXIF.
        let mut exif = Vec::new();
        if let Some((n, d)) = self.exposure_time {
            exif.push(file.urational(0x829A, &[(n, d)]));
        }
        if let Some((n, d)) = self.f_number {
            exif.push(file.urational(0x829D, &[(n, d)]));
        }
        if let Some(iso) = self.iso {
            exif.push(file.short(0x8827, &[iso.min(65535) as u16]));
        }
        exif.push(file.ascii(0x9003, "2026:11:13 12:00:00"));
        if let Some((n, d)) = self.focal_length {
            exif.push(file.urational(0x920A, &[(n, d)]));
        }
        if let Some(f) = self.focal_length_35mm {
            exif.push(file.short(0xA405, &[f]));
        }
        if let Some(lens) = &self.lens_model {
            exif.push(file.ascii(0xA434, lens));
        }
        let exif_ifd = file.ifd(exif);

        // IFD0.
        let mut ifd0 = vec![
            file.ascii(0x010F, &self.make),
            file.ascii(0x0110, &self.model),
            file.short(0x0112, &[self.orientation]),
            file.ascii(0x0131, "Yellowmoon test DNG writer"),
        ];
        let mut subs = vec![raw_ifd];
        subs.extend(preview_ifd);
        ifd0.push(file.long(0x014A, &subs));
        ifd0.push(file.long(0x8769, &[exif_ifd]));
        ifd0.push(file.field(0xC612, 1, 4, vec![1, 4, 0, 0]));
        if let Some(b) = self.backward_version {
            ifd0.push(file.field(0xC613, 1, 4, b.to_vec()));
        }
        if let Some(u) = &self.unique_model {
            ifd0.push(file.ascii(0xC614, u));
        }
        for (i, tag_ref) in [(0usize, 0xC621u16), (1, 0xC622)] {
            if let Some(m) = &self.color_matrix[i] {
                ifd0.push(file.srational(tag_ref, m));
            }
        }
        for (i, tag_ref) in [(0usize, 0xC623u16), (1, 0xC624)] {
            if let Some(m) = &self.camera_calibration[i] {
                ifd0.push(file.srational(tag_ref, m));
            }
        }
        for (i, tag_ref) in [(0usize, 0xC65Au16), (1, 0xC65B)] {
            if let Some(v) = self.illuminant[i] {
                ifd0.push(file.short(tag_ref, &[v]));
            }
        }
        if let Some(a) = &self.analog_balance {
            ifd0.push(file.rational(0xC627, a));
        }
        if let Some(n) = &self.as_shot_neutral {
            ifd0.push(file.rational(0xC628, n));
        }
        if let Some(xy) = &self.as_shot_white_xy {
            ifd0.push(file.rational(0xC629, xy));
        }
        if let Some(e) = self.baseline_exposure {
            ifd0.push(file.srational(0xC62A, &[e]));
        }
        let first = file.ifd(ifd0);
        let order = self.big_endian;
        let patch = if order {
            first.to_be_bytes()
        } else {
            first.to_le_bytes()
        };
        file.bytes[4..8].copy_from_slice(&patch);
        file.bytes
    }

    /// Splits the samples into strips or tiles and encodes each.
    fn chunks(&self, samples: &[u16]) -> (Vec<Vec<u8>>, Grid) {
        let spp = self.samples_per_pixel();
        let (w, h) = (self.width, self.height);
        let (grid, across, down, cw, ch) = match self.storage {
            DngStorage::Strips(rows) => (Grid::Strips { rows }, 1, h.div_ceil(rows), w, rows),
            DngStorage::Tiles(tw, tl) => (
                Grid::Tiles {
                    width: tw,
                    length: tl,
                },
                w.div_ceil(tw),
                h.div_ceil(tl),
                tw,
                tl,
            ),
        };
        let mut out = Vec::new();
        for by in 0..down {
            for bx in 0..across {
                let rows = match grid {
                    Grid::Strips { .. } => ch.min(h - by * ch),
                    Grid::Tiles { .. } => ch,
                };
                // The chunk's samples, padded with zeros past the image.
                let mut chunk = vec![0u16; cw * spp * rows];
                for r in 0..rows {
                    let y = by * ch + r;
                    if y >= h {
                        continue;
                    }
                    for x in 0..cw {
                        let ix = bx * cw + x;
                        if ix >= w {
                            continue;
                        }
                        for s in 0..spp {
                            chunk[(r * cw + x) * spp + s] = samples[(y * w + ix) * spp + s];
                        }
                    }
                }
                out.push(self.encode_chunk(&chunk, cw * spp, rows));
            }
        }
        (out, grid)
    }

    fn encode_chunk(&self, chunk: &[u16], row_samples: usize, rows: usize) -> Vec<u8> {
        match self.compression {
            DngCompression::None => {
                let mut bytes = Vec::new();
                for row in chunk.chunks_exact(row_samples) {
                    pack_row(row, self.bits, self.big_endian, &mut bytes);
                }
                bytes
            }
            DngCompression::LosslessJpeg {
                components,
                predictor,
            } => {
                assert_eq!(row_samples % components, 0, "components divide the row");
                let width = row_samples / components;
                let planes: Vec<Vec<u16>> = (0..components)
                    .map(|c| {
                        (0..rows)
                            .flat_map(|r| (0..width).map(move |x| (r, x)))
                            .map(|(r, x)| chunk[r * row_samples + x * components + c])
                            .collect()
                    })
                    .collect();
                encode(
                    &planes,
                    &EncodeParams {
                        precision: self.bits as u8,
                        predictor,
                        point_transform: 0,
                        restart_interval: 0,
                        sony_420_quirk: false,
                        huffman: HuffmanChoice::Optimal,
                        width,
                        height: rows,
                        sampling: vec![(1, 1); components],
                    },
                )
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Grid {
    Strips { rows: usize },
    Tiles { width: usize, length: usize },
}

/// Packs one row of samples: whole bytes for 8 and 16 bits, most significant bit first otherwise,
/// the row padded to a byte.
fn pack_row(row: &[u16], bits: u16, big_endian: bool, out: &mut Vec<u8>) {
    match bits {
        8 => out.extend(row.iter().map(|v| *v as u8)),
        16 => {
            for v in row {
                out.extend_from_slice(&if big_endian {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
        }
        _ => {
            let mut acc = 0u32;
            let mut have = 0u32;
            for v in row {
                acc = (acc << bits) | u32::from(*v);
                have += u32::from(bits);
                while have >= 8 {
                    have -= 8;
                    out.push((acc >> have) as u8);
                    acc &= (1 << have) - 1;
                }
            }
            if have > 0 {
                out.push((acc << (8 - have)) as u8);
            }
        }
    }
}

/// The parameters of a gain map opcode, big endian.
fn gain_map_bytes(m: &GainMap) -> Vec<u8> {
    let mut b = Vec::new();
    for v in [
        m.top,
        m.left,
        m.bottom,
        m.right,
        m.plane,
        m.planes,
        m.row_pitch,
        m.col_pitch,
        m.points_v,
        m.points_h,
    ] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    for v in [m.spacing_v, m.spacing_h, m.origin_v, m.origin_h] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&m.map_planes.to_be_bytes());
    for g in &m.gains {
        b.extend_from_slice(&g.to_be_bytes());
    }
    b
}

/// An opcode list: a count, then for each opcode its ID, the DNG version that introduced it, its
/// flags and the byte count of its parameters; all big endian.
pub fn opcode_list(opcodes: &[(u32, u32, Vec<u8>)]) -> Vec<u8> {
    let mut b = (opcodes.len() as u32).to_be_bytes().to_vec();
    for (id, flags, params) in opcodes {
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0x0101_0000u32.to_be_bytes());
        b.extend_from_slice(&flags.to_be_bytes());
        b.extend_from_slice(&(params.len() as u32).to_be_bytes());
        b.extend_from_slice(params);
    }
    b
}

/// The bytes of the file so far and the helpers that make directory fields.
struct Writer {
    bytes: Vec<u8>,
    big: bool,
}

impl Writer {
    fn new(big: bool) -> Self {
        Self {
            bytes: Vec::new(),
            big,
        }
    }

    fn align(&mut self) {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
    }

    fn u16b(&self, v: u16) -> [u8; 2] {
        if self.big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn u32b(&self, v: u32) -> [u8; 4] {
        if self.big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn field(&self, tag: u16, ty: u16, count: u32, data: Vec<u8>) -> Field {
        let count = if count == 0 {
            // Types of one byte per value: the count is the byte length.
            data.len() as u32
        } else {
            count
        };
        Field {
            tag,
            ty,
            count,
            data,
        }
    }

    fn short(&self, tag: u16, v: &[u16]) -> Field {
        let data = v.iter().flat_map(|x| self.u16b(*x)).collect();
        Field {
            tag,
            ty: 3,
            count: v.len() as u32,
            data,
        }
    }

    fn long(&self, tag: u16, v: &[u32]) -> Field {
        let data = v.iter().flat_map(|x| self.u32b(*x)).collect();
        Field {
            tag,
            ty: 4,
            count: v.len() as u32,
            data,
        }
    }

    fn ascii(&self, tag: u16, s: &str) -> Field {
        let mut data = s.as_bytes().to_vec();
        data.push(0);
        Field {
            tag,
            ty: 2,
            count: data.len() as u32,
            data,
        }
    }

    fn urational(&self, tag: u16, v: &[(u32, u32)]) -> Field {
        let data = v
            .iter()
            .flat_map(|(n, d)| [self.u32b(*n), self.u32b(*d)].concat())
            .collect();
        Field {
            tag,
            ty: 5,
            count: v.len() as u32,
            data,
        }
    }

    /// Unsigned rationals with a denominator of 10000.
    fn rational(&self, tag: u16, v: &[f64]) -> Field {
        let pairs: Vec<(u32, u32)> = v
            .iter()
            .map(|x| ((x * 10000.0).round() as u32, 10000))
            .collect();
        self.urational(tag, &pairs)
    }

    /// Signed rationals with a denominator of 10000.
    fn srational(&self, tag: u16, v: &[f64]) -> Field {
        let data = v
            .iter()
            .flat_map(|x| {
                [
                    self.u32b(((x * 10000.0).round() as i32) as u32),
                    self.u32b(10000),
                ]
                .concat()
            })
            .collect();
        Field {
            tag,
            ty: 10,
            count: v.len() as u32,
            data,
        }
    }

    fn double(&self, tag: u16, v: &[f64]) -> Field {
        let data = v
            .iter()
            .flat_map(|x| {
                if self.big {
                    x.to_be_bytes()
                } else {
                    x.to_le_bytes()
                }
            })
            .collect();
        Field {
            tag,
            ty: 12,
            count: v.len() as u32,
            data,
        }
    }

    /// Writes a directory (and its values) at the end of the file and returns its offset.
    fn ifd(&mut self, mut fields: Vec<Field>) -> u32 {
        fields.sort_by_key(|f| f.tag);
        self.align();
        let at = self.bytes.len() as u32;
        let table = 2 + fields.len() * 12 + 4;
        let mut ext = Vec::new();
        let mut entries = Vec::new();
        for f in &fields {
            entries.extend_from_slice(&self.u16b(f.tag));
            entries.extend_from_slice(&self.u16b(f.ty));
            entries.extend_from_slice(&self.u32b(f.count));
            if f.data.len() <= 4 {
                let mut inline = f.data.clone();
                inline.resize(4, 0);
                entries.extend_from_slice(&inline);
            } else {
                while ext.len() % 2 != 0 {
                    ext.push(0);
                }
                let offset = at + table as u32 + ext.len() as u32;
                entries.extend_from_slice(&self.u32b(offset));
                ext.extend_from_slice(&f.data);
            }
        }
        self.bytes
            .extend_from_slice(&self.u16b(fields.len() as u16));
        self.bytes.extend_from_slice(&entries);
        self.bytes.extend_from_slice(&[0; 4]);
        self.bytes.extend_from_slice(&ext);
        at
    }
}
