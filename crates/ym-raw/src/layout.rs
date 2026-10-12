//! `Layout` (Bayer pattern, active area and crop, or YCbCr subsampling) and the `Plane<T>` pixel buffer (`docs/04-raw-decoding.md` section Output).
//!
//! Also the small geometry and classification types shared by the probe and the decoders: the 2x2
//! colour filter array pattern, rectangles, and [`DecodeKind`], the answer to "which decoder reads
//! this file".

use serde::{Deserialize, Serialize};

use crate::error::RawError;

/// The largest image the container parser accepts, in pixels (120 MP, `docs/04` section Tests).
/// Declared dimensions above this are treated as corrupt before anything is allocated.
pub const MAX_PIXELS: u64 = 120_000_000;

/// The largest width or height the container parser accepts.
pub const MAX_DIMENSION: u32 = 65_535;

/// A row-major buffer of samples with no padding between rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plane<T> {
    /// Samples per row.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// `width * height` samples, row by row.
    pub data: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    /// Allocates a zero-filled plane. Fails with [`RawError::Corrupt`] when the dimensions are
    /// zero, exceed [`MAX_DIMENSION`] or [`MAX_PIXELS`], so a hostile header cannot make the
    /// process allocate gigabytes.
    pub fn try_new(width: usize, height: usize) -> Result<Self, RawError> {
        check_dimensions(width, height)?;
        Ok(Self {
            width,
            height,
            data: vec![T::default(); width * height],
        })
    }
}

impl<T> Plane<T> {
    /// Wraps existing samples. Returns `None` when `data.len() != width * height`.
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Option<Self> {
        (width.checked_mul(height)? == data.len()).then_some(Self {
            width,
            height,
            data,
        })
    }

    /// The sample at column `x`, row `y`, or `None` outside the plane.
    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x < self.width && y < self.height {
            self.data.get(y * self.width + x)
        } else {
            None
        }
    }

    /// One row of samples, or `None` when `y` is outside the plane.
    pub fn row(&self, y: usize) -> Option<&[T]> {
        if y < self.height {
            self.data.get(y * self.width..(y + 1) * self.width)
        } else {
            None
        }
    }

    /// One row of samples for writing, or `None` when `y` is outside the plane.
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [T]> {
        if y < self.height {
            self.data.get_mut(y * self.width..(y + 1) * self.width)
        } else {
            None
        }
    }
}

/// Checks that an image size is non-zero and within the supported limits.
pub fn check_dimensions(width: usize, height: usize) -> Result<(), RawError> {
    if width == 0 || height == 0 {
        return Err(RawError::corrupt("the image has no pixels"));
    }
    if width > MAX_DIMENSION as usize || height > MAX_DIMENSION as usize {
        return Err(RawError::corrupt(format!(
            "image size {width} x {height} is out of range"
        )));
    }
    if (width as u64) * (height as u64) > MAX_PIXELS {
        return Err(RawError::corrupt(format!(
            "image size {width} x {height} is above the 120 megapixel limit"
        )));
    }
    Ok(())
}

/// The colour of one photosite of the colour filter array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CfaColor {
    /// Red filter.
    Red,
    /// Green filter.
    Green,
    /// Blue filter.
    Blue,
}

impl CfaColor {
    /// `'R'`, `'G'` or `'B'`.
    pub fn letter(self) -> char {
        match self {
            Self::Red => 'R',
            Self::Green => 'G',
            Self::Blue => 'B',
        }
    }

    /// The TIFF `CFAPattern` code (0 = red, 1 = green, 2 = blue).
    pub fn tiff_code(self) -> u8 {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
        }
    }

    /// The colour for a TIFF `CFAPattern` code, or `None` for codes this crate does not use
    /// (cyan, magenta, yellow, white).
    pub fn from_tiff_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Red),
            1 => Some(Self::Green),
            2 => Some(Self::Blue),
            _ => None,
        }
    }
}

/// A 2x2 colour filter array pattern in raster order: top-left, top-right, bottom-left,
/// bottom-right. Sony bodies write RGGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CfaPattern(pub [CfaColor; 4]);

impl CfaPattern {
    /// Red, green / green, blue. The pattern of every Sony body.
    pub const RGGB: Self = Self([
        CfaColor::Red,
        CfaColor::Green,
        CfaColor::Green,
        CfaColor::Blue,
    ]);
    /// Blue, green / green, red.
    pub const BGGR: Self = Self([
        CfaColor::Blue,
        CfaColor::Green,
        CfaColor::Green,
        CfaColor::Red,
    ]);
    /// Green, red / blue, green.
    pub const GRBG: Self = Self([
        CfaColor::Green,
        CfaColor::Red,
        CfaColor::Blue,
        CfaColor::Green,
    ]);
    /// Green, blue / red, green.
    pub const GBRG: Self = Self([
        CfaColor::Green,
        CfaColor::Blue,
        CfaColor::Red,
        CfaColor::Green,
    ]);

    /// The filter colour at column `x`, row `y` of the mosaic.
    pub fn color_at(self, x: usize, y: usize) -> CfaColor {
        self.0[(y & 1) * 2 + (x & 1)]
    }

    /// The four TIFF `CFAPattern` codes in raster order.
    pub fn tiff_codes(self) -> [u8; 4] {
        self.0.map(CfaColor::tiff_code)
    }

    /// Builds a pattern from four TIFF `CFAPattern` codes. Returns `None` for unknown codes or a
    /// pattern that is not a valid Bayer quad (two greens, one red, one blue).
    pub fn from_tiff_codes(codes: [u8; 4]) -> Option<Self> {
        let mut colors = [CfaColor::Red; 4];
        for (slot, code) in colors.iter_mut().zip(codes) {
            *slot = CfaColor::from_tiff_code(code)?;
        }
        let count = |wanted: CfaColor| colors.iter().filter(|c| **c == wanted).count();
        (count(CfaColor::Red) == 1 && count(CfaColor::Green) == 2 && count(CfaColor::Blue) == 1)
            .then_some(Self(colors))
    }

    /// The pattern as four letters in raster order, for example `"RGGB"`.
    pub fn name(self) -> String {
        self.0.iter().map(|c| c.letter()).collect()
    }

    /// Maps per-colour values in the order red, green 1, green 2, blue (the order Sony stores
    /// black levels and white balance in) to the four raster positions of this pattern.
    ///
    /// The first green in raster order takes `green1`, the second takes `green2`.
    pub fn rggb_to_positions<T: Copy>(self, rggb: [T; 4]) -> [T; 4] {
        let mut greens_seen = 0;
        self.0.map(|color| match color {
            CfaColor::Red => rggb[0],
            CfaColor::Green => {
                greens_seen += 1;
                rggb[greens_seen]
            }
            CfaColor::Blue => rggb[3],
        })
    }
}

/// An axis-aligned rectangle in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
    /// Left edge (column of the first pixel).
    pub x: u32,
    /// Top edge (row of the first pixel).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Rect {
    /// True when the rectangle has pixels and lies completely inside a `width` x `height` image.
    pub fn fits_in(self, width: u32, height: u32) -> bool {
        self.width > 0
            && self.height > 0
            && self.x.checked_add(self.width).is_some_and(|r| r <= width)
            && self.y.checked_add(self.height).is_some_and(|b| b <= height)
    }
}

/// How the chroma planes of a camera-processed (Lossless M and S) file are subsampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChromaSubsampling {
    /// 4:4:4, no subsampling.
    Yuv444,
    /// 4:2:2, chroma halved horizontally.
    Yuv422,
    /// 4:2:0, chroma halved in both directions.
    Yuv420,
}

impl ChromaSubsampling {
    /// `"4:4:4"`, `"4:2:2"` or `"4:2:0"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Yuv444 => "4:4:4",
            Self::Yuv422 => "4:2:2",
            Self::Yuv420 => "4:2:0",
        }
    }
}

/// How the samples of a decoded image are arranged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    /// A Bayer mosaic: one sample per photosite.
    Bayer {
        /// The 2x2 colour filter pattern.
        pattern: CfaPattern,
        /// The part of the raw readout that holds valid image data. `None` means the whole
        /// readout.
        active_area: Option<Rect>,
        /// The crop the camera asks for (Sony `SonyCropTopLeft` and `SonyCropSize` or the DNG
        /// default crop), inside the readout. `None` means no crop.
        crop: Option<Rect>,
    },
    /// Camera-processed YCbCr (Lossless M and S): white balanced, demosaiced, not a mosaic.
    Ycbcr {
        /// How the chroma planes are subsampled.
        subsampling: ChromaSubsampling,
    },
}

/// Which decoder reads a file's raw data (`docs/04-raw-decoding.md` section Decoders by TIFF
/// compression).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodeKind {
    /// 12-bit uncompressed. TIFF compression 32767 with 12 bits stores packed 12-bit samples
    /// (ARW 2); compression 1 with 12 bits stores 16-bit little-endian words. The metadata's
    /// `compression` field tells the two apart.
    Uncompressed12,
    /// 14-bit uncompressed: 16-bit little-endian words (compression 1, or 32767 with two bytes
    /// per sample).
    Uncompressed14,
    /// cRAW, Sony's lossy "compressed RAW" (compression 32767, one byte per pixel; the cameras
    /// write BitsPerSample 12, the format description says 8).
    Craw,
    /// Lossless L: tiled 4-component lossless JPEG of the Bayer mosaic (compression 7).
    LosslessL,
    /// Lossless M or S: camera-processed 3-component YCbCr lossless JPEG (compression 7,
    /// photometric YCbCr).
    LosslessMS {
        /// The chroma subsampling of the YCbCr planes.
        subsampling: ChromaSubsampling,
    },
    /// ARW 1: 12-bit samples Huffman-coded column by column (compression 32767, 12 bits, a byte
    /// count that fits no uncompressed layout). The A200, A230, A290, A300, A330, A350, A380 and
    /// A390 write it.
    Arw1,
    /// ARW 6.0 "Compressed RAW 2" (compression 32766), the A7 V and A7R VI lossy wavelet codec.
    Arw6,
    /// A raw layout this crate recognises as a Sony raw file but cannot classify (for example the
    /// A100's ARW 1 Huffman coding).
    Unknown {
        /// The TIFF `Compression` value.
        compression: u16,
        /// The TIFF `BitsPerSample` value.
        bits: u16,
    },
}

impl DecodeKind {
    /// A human-readable name for messages, for example `"Lossless M or S"`.
    pub fn describe(&self) -> String {
        match self {
            Self::Uncompressed12 => "12-bit uncompressed".to_owned(),
            Self::Uncompressed14 => "14-bit uncompressed".to_owned(),
            Self::Craw => "compressed (lossy cRAW)".to_owned(),
            Self::Arw1 => "ARW 1 compressed".to_owned(),
            Self::LosslessL => "Lossless L".to_owned(),
            Self::LosslessMS { .. } => "Lossless M or S".to_owned(),
            Self::Arw6 => "ARW 6.0 compressed".to_owned(),
            Self::Unknown { compression, bits } => {
                format!("unrecognised (compression {compression}, {bits} bits)")
            }
        }
    }

    /// A short stable identifier for JSON output: `uncompressed12`, `uncompressed14`, `craw`,
    /// `arw1`, `lossless_l`, `lossless_ms`, `arw6` or `unknown`.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Uncompressed12 => "uncompressed12",
            Self::Uncompressed14 => "uncompressed14",
            Self::Craw => "craw",
            Self::Arw1 => "arw1",
            Self::LosslessL => "lossless_l",
            Self::LosslessMS { .. } => "lossless_ms",
            Self::Arw6 => "arw6",
            Self::Unknown { .. } => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_try_new_rejects_empty_and_oversized_images() {
        assert!(Plane::<u16>::try_new(0, 10).is_err());
        assert!(Plane::<u16>::try_new(10, 0).is_err());
        assert!(Plane::<u16>::try_new(70_000, 10).is_err());
        // 12 000 x 12 000 = 144 MP is above the 120 MP cap.
        assert!(Plane::<u16>::try_new(12_000, 12_000).is_err());
        let plane = Plane::<u16>::try_new(4, 3).expect("small plane");
        assert_eq!(plane.data.len(), 12);
        assert!(plane.data.iter().all(|v| *v == 0));
    }

    #[test]
    fn plane_accessors_stay_in_bounds() {
        let plane = Plane::from_vec(3, 2, vec![1u16, 2, 3, 4, 5, 6]).expect("sized");
        assert_eq!(plane.get(2, 1), Some(&6));
        assert_eq!(plane.get(3, 0), None);
        assert_eq!(plane.row(1), Some(&[4u16, 5, 6][..]));
        assert_eq!(plane.row(2), None);
        assert!(Plane::from_vec(3, 2, vec![0u16; 5]).is_none());
        let mut plane = plane;
        plane.row_mut(0).expect("row 0")[1] = 9;
        assert_eq!(plane.data[1], 9);
        assert!(plane.row_mut(2).is_none());
    }

    #[test]
    fn cfa_pattern_codes_round_trip() {
        for pattern in [
            CfaPattern::RGGB,
            CfaPattern::BGGR,
            CfaPattern::GRBG,
            CfaPattern::GBRG,
        ] {
            assert_eq!(
                CfaPattern::from_tiff_codes(pattern.tiff_codes()),
                Some(pattern)
            );
        }
        assert_eq!(CfaPattern::RGGB.tiff_codes(), [0, 1, 1, 2]);
        assert_eq!(CfaPattern::RGGB.name(), "RGGB");
        assert_eq!(CfaPattern::GBRG.name(), "GBRG");
    }

    #[test]
    fn cfa_pattern_rejects_non_bayer_quads() {
        assert_eq!(CfaPattern::from_tiff_codes([0, 0, 1, 2]), None);
        assert_eq!(CfaPattern::from_tiff_codes([0, 1, 1, 3]), None);
        assert_eq!(CfaPattern::from_tiff_codes([1, 1, 1, 1]), None);
    }

    #[test]
    fn cfa_color_at_wraps_every_two_pixels() {
        let p = CfaPattern::GRBG;
        assert_eq!(p.color_at(0, 0), CfaColor::Green);
        assert_eq!(p.color_at(1, 0), CfaColor::Red);
        assert_eq!(p.color_at(2, 0), CfaColor::Green);
        assert_eq!(p.color_at(0, 1), CfaColor::Blue);
        assert_eq!(p.color_at(3, 3), CfaColor::Green);
    }

    #[test]
    fn rggb_values_map_to_raster_positions_by_colour() {
        let v = [10, 20, 30, 40]; // R, G1, G2, B
        assert_eq!(CfaPattern::RGGB.rggb_to_positions(v), [10, 20, 30, 40]);
        assert_eq!(CfaPattern::BGGR.rggb_to_positions(v), [40, 20, 30, 10]);
        assert_eq!(CfaPattern::GRBG.rggb_to_positions(v), [20, 10, 40, 30]);
        assert_eq!(CfaPattern::GBRG.rggb_to_positions(v), [20, 40, 10, 30]);
    }

    #[test]
    fn rect_fits_checks_all_four_edges_without_overflow() {
        let r = Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 50,
        };
        assert!(r.fits_in(110, 70));
        assert!(!r.fits_in(109, 70));
        assert!(!r.fits_in(110, 69));
        let huge = Rect {
            x: u32::MAX,
            y: 0,
            width: 2,
            height: 1,
        };
        assert!(!huge.fits_in(u32::MAX, 10));
        assert!(!Rect::default().fits_in(10, 10));
    }

    #[test]
    fn decode_kind_ids_and_descriptions_are_distinct() {
        let kinds = [
            DecodeKind::Uncompressed12,
            DecodeKind::Uncompressed14,
            DecodeKind::Craw,
            DecodeKind::Arw1,
            DecodeKind::LosslessL,
            DecodeKind::LosslessMS {
                subsampling: ChromaSubsampling::Yuv422,
            },
            DecodeKind::Arw6,
            DecodeKind::Unknown {
                compression: 9,
                bits: 3,
            },
        ];
        let mut ids: Vec<_> = kinds.iter().map(|k| k.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), kinds.len());
        let mut names: Vec<_> = kinds.iter().map(|k| k.describe()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), kinds.len());
    }
}
