//! Export options and their defaults (`docs/09-export.md` section Options).
//!
//! [`ExportOptions`] is the flat shape the UI and the preferences use (it serialises exactly as
//! `ExportOptions` in `src/api/types.ts`); [`ExportOptions::format_spec`] turns it into the
//! richer [`ExportFormat`] the writers take. Everything is `serde`-serialisable and validated on
//! construction or by [`ExportOptions::validate`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use ym_color::encode::OutputSpace;

use crate::error::ExportError;

/// The default file name template (`docs/09`).
pub const DEFAULT_NAME_TEMPLATE: &str = "{name}_{look}";

/// The default JPEG quality (`docs/09`).
pub const DEFAULT_JPEG_QUALITY: u8 = 92;

/// The smallest long edge the library accepts for [`Size::LongEdge`], in pixels.
pub const MIN_LONG_EDGE_PX: u32 = 16;

/// The largest long edge the library accepts for [`Size::LongEdge`], in pixels (the JPEG limit).
pub const MAX_LONG_EDGE_PX: u32 = 65_535;

/// The step of the UI's long-edge slider, in pixels (`docs/09`: "Long edge 1000–full px in 20 px steps").
pub const LONG_EDGE_STEP_PX: u32 = 20;

/// The smallest long edge the UI offers, in pixels.
pub const UI_MIN_LONG_EDGE_PX: u32 = 1000;

/// JPEG quality, 50 to 100.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct JpegQuality(u8);

impl JpegQuality {
    /// The lowest quality offered.
    pub const MIN: u8 = 50;
    /// The highest quality offered.
    pub const MAX: u8 = 100;

    /// Validates `quality` (50 to 100).
    pub fn new(quality: u8) -> Result<Self, ExportError> {
        Self::try_from(quality).map_err(ExportError::InvalidInput)
    }

    /// The quality as a number.
    #[must_use]
    pub fn get(self) -> u8 {
        self.0
    }

    /// Whether this quality uses 4:4:4 chroma (quality 90 and above); below that the writer uses 4:2:0.
    #[must_use]
    pub fn is_full_chroma(self) -> bool {
        self.0 >= 90
    }
}

impl Default for JpegQuality {
    fn default() -> Self {
        Self(DEFAULT_JPEG_QUALITY)
    }
}

impl TryFrom<u8> for JpegQuality {
    type Error = String;
    fn try_from(q: u8) -> Result<Self, String> {
        if (Self::MIN..=Self::MAX).contains(&q) {
            Ok(Self(q))
        } else {
            Err(format!(
                "JPEG quality must be between {} and {}, not {q}",
                Self::MIN,
                Self::MAX
            ))
        }
    }
}

impl From<JpegQuality> for u8 {
    fn from(q: JpegQuality) -> u8 {
        q.0
    }
}

/// Bits per channel of a TIFF or PNG: 8 or 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum BitDepth {
    /// 8 bits per channel (dithered).
    Eight,
    /// 16 bits per channel.
    Sixteen,
}

impl BitDepth {
    /// The number of bits.
    #[must_use]
    pub fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }
}

impl TryFrom<u8> for BitDepth {
    type Error = String;
    fn try_from(bits: u8) -> Result<Self, String> {
        match bits {
            8 => Ok(Self::Eight),
            16 => Ok(Self::Sixteen),
            other => Err(format!("the bit depth must be 8 or 16, not {other}")),
        }
    }
}

impl From<BitDepth> for u8 {
    fn from(b: BitDepth) -> u8 {
        b.bits()
    }
}

/// TIFF strip compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TiffCompression {
    /// No compression (TIFF compression 1).
    None,
    /// LZW with horizontal predictor (compression 5).
    #[default]
    Lzw,
    /// Deflate (zlib) with horizontal predictor, shown to the user as "ZIP" (compression 8).
    #[serde(alias = "deflate")]
    Zip,
}

/// The file format and its format-specific options, as the writers take them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum ExportFormat {
    /// Baseline JPEG.
    Jpeg {
        /// Quality, 50 to 100.
        quality: JpegQuality,
    },
    /// TIFF, RGB, strips.
    Tiff {
        /// 8 or 16 bits per channel.
        bits: BitDepth,
        /// Strip compression.
        compression: TiffCompression,
    },
    /// PNG, RGB.
    Png {
        /// 8 or 16 bits per channel.
        bits: BitDepth,
    },
    /// DNG (LinearRaw, scene-linear AWG4, look baked in).
    Dng,
}

impl Default for ExportFormat {
    fn default() -> Self {
        Self::Jpeg {
            quality: JpegQuality::default(),
        }
    }
}

impl ExportFormat {
    /// Which format this is, without its options.
    #[must_use]
    pub fn kind(self) -> FormatKind {
        match self {
            Self::Jpeg { .. } => FormatKind::Jpeg,
            Self::Tiff { .. } => FormatKind::Tiff,
            Self::Png { .. } => FormatKind::Png,
            Self::Dng => FormatKind::Dng,
        }
    }

    /// The file extension without the dot (`jpg`, `tif`, `png`, `dng`).
    #[must_use]
    pub fn extension(self) -> &'static str {
        self.kind().extension()
    }

    /// Whether the colour space and size options apply (everything except DNG).
    #[must_use]
    pub fn uses_color_space_and_size(self) -> bool {
        !matches!(self, Self::Dng)
    }
}

/// The four formats, without options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FormatKind {
    /// JPEG.
    #[default]
    Jpeg,
    /// TIFF.
    Tiff,
    /// PNG.
    Png,
    /// DNG.
    Dng,
}

impl FormatKind {
    /// The file extension without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Tiff => "tif",
            Self::Png => "png",
            Self::Dng => "dng",
        }
    }

    /// The format's name as shown to the user.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Dng => "DNG",
        }
    }
}

macro_rules! from_str_words {
    ($ty:ty, $what:literal, { $($word:literal => $value:expr),+ $(,)? }) => {
        impl std::str::FromStr for $ty {
            type Err = String;

            /// Parses the words of the command line (`docs/10`), case-insensitively.
            fn from_str(s: &str) -> Result<Self, String> {
                match s.trim().to_ascii_lowercase().as_str() {
                    $($word => Ok($value),)+
                    other => Err(format!(
                        concat!("unknown ", $what, " \"{}\"; use one of ", $($word, " ",)+),
                        other
                    )),
                }
            }
        }
    };
}

/// The export colour space (JPEG, TIFF and PNG).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    /// sRGB: Rec.709 primaries, sRGB transfer ("Best for sharing and the web.").
    #[default]
    Srgb,
    /// Display P3: P3-D65 primaries, sRGB transfer.
    DisplayP3,
    /// Rec.2020: Rec.2020 primaries, BT.1886 (gamma 2.4) transfer.
    Rec2020,
}

impl ColorSpace {
    /// All three spaces, in menu order.
    pub const ALL: [ColorSpace; 3] = [Self::Srgb, Self::DisplayP3, Self::Rec2020];

    /// The matching display encode of `ym-color` (stage C13).
    #[must_use]
    pub fn output_space(self) -> OutputSpace {
        match self {
            Self::Srgb => OutputSpace::Srgb,
            Self::DisplayP3 => OutputSpace::DisplayP3,
            Self::Rec2020 => OutputSpace::Rec2020,
        }
    }

    /// The space as shown to the user.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB",
            Self::DisplayP3 => "Display P3",
            Self::Rec2020 => "Rec.2020",
        }
    }

    /// The ITU-T H.273 code points `[colour primaries, transfer, matrix, full range]` for the PNG `cICP` chunk.
    ///
    /// Rec.2020 uses transfer 1 (BT.709): H.273 has no separate code for the BT.1886 display
    /// curve, and code 1 is the one it pairs with a BT.1886 display. The embedded ICC profile carries
    /// the exact curve (a pure gamma 2.4).
    #[must_use]
    pub fn cicp(self) -> [u8; 4] {
        match self {
            Self::Srgb => [1, 13, 0, 1],
            Self::DisplayP3 => [12, 13, 0, 1],
            Self::Rec2020 => [9, 1, 0, 1],
        }
    }

    /// The EXIF `ColorSpace` value: 1 for sRGB, 0xFFFF (uncalibrated) for the others.
    #[must_use]
    pub fn exif_color_space(self) -> u16 {
        match self {
            Self::Srgb => 1,
            _ => 0xFFFF,
        }
    }
}

from_str_words!(ColorSpace, "colour space", {
    "srgb" => ColorSpace::Srgb,
    "p3" => ColorSpace::DisplayP3,
    "display-p3" => ColorSpace::DisplayP3,
    "display_p3" => ColorSpace::DisplayP3,
    "rec2020" => ColorSpace::Rec2020,
    "rec.2020" => ColorSpace::Rec2020,
});

from_str_words!(TiffCompression, "TIFF compression", {
    "none" => TiffCompression::None,
    "lzw" => TiffCompression::Lzw,
    "zip" => TiffCompression::Zip,
    "deflate" => TiffCompression::Zip,
});

from_str_words!(FormatKind, "format", {
    "jpeg" => FormatKind::Jpeg,
    "jpg" => FormatKind::Jpeg,
    "tiff" => FormatKind::Tiff,
    "tif" => FormatKind::Tiff,
    "png" => FormatKind::Png,
    "dng" => FormatKind::Dng,
});

/// The output size (JPEG, TIFF and PNG).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Size {
    /// The full frame.
    #[default]
    Full,
    /// Scale so the long edge is `px` pixels (never upscales).
    LongEdge {
        /// The long edge in pixels.
        px: u32,
    },
}

impl Size {
    /// The output dimensions for a `width` x `height` source. A long edge at or above the source's
    /// is the full size (no upscaling); the short edge is rounded half up and at least 1.
    #[must_use]
    pub fn output_dimensions(self, width: u32, height: u32) -> (u32, u32) {
        let Self::LongEdge { px } = self else {
            return (width, height);
        };
        let long = width.max(height);
        if px == 0 || px >= long {
            return (width, height);
        }
        let short = u64::from(width.min(height));
        let scaled = ((short * u64::from(px) * 2 + u64::from(long)) / (2 * u64::from(long))).max(1);
        let scaled = u32::try_from(scaled).unwrap_or(px);
        if width >= height {
            (px, scaled)
        } else {
            (scaled, px)
        }
    }

    /// The long-edge values the UI offers for a frame whose long edge is `full_long_edge_px`:
    /// 1000 up in 20 px steps, then the full size itself. Empty when the frame is shorter than 1000 px.
    #[must_use]
    pub fn ui_long_edge_choices(full_long_edge_px: u32) -> Vec<u32> {
        let mut out: Vec<u32> = (UI_MIN_LONG_EDGE_PX..full_long_edge_px)
            .step_by(LONG_EDGE_STEP_PX as usize)
            .collect();
        if full_long_edge_px >= UI_MIN_LONG_EDGE_PX {
            out.push(full_long_edge_px);
        }
        out
    }

    /// Snaps `px` to the UI's grid: the nearest multiple of 20 from 1000 up, clamped to the frame.
    #[must_use]
    pub fn snap_long_edge(px: u32, full_long_edge_px: u32) -> u32 {
        let snapped = (px.max(UI_MIN_LONG_EDGE_PX) + LONG_EDGE_STEP_PX / 2) / LONG_EDGE_STEP_PX
            * LONG_EDGE_STEP_PX;
        snapped.min(full_long_edge_px)
    }
}

/// What to do when the target file already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IfExists {
    /// Leave the existing file alone and write nothing.
    Skip,
    /// Replace the existing file (atomically).
    Replace,
    /// Keep both: add " 2", " 3" and so on to the new file's name.
    #[default]
    KeepBoth,
}

/// The export options of the panel, preferences and CLI, in the shape the UI uses
/// (`ExportOptions` in `src/api/types.ts`). Defaults are the table in `docs/09`; a field missing from
/// stored preferences takes its default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportOptions {
    /// JPEG, TIFF, PNG or DNG.
    pub format: FormatKind,
    /// JPEG quality, 50 to 100.
    pub jpeg_quality: JpegQuality,
    /// TIFF and PNG share this preference.
    pub bit_depth: BitDepth,
    /// TIFF compression.
    pub tiff_compression: TiffCompression,
    /// The export colour space (not for DNG).
    pub color_space: ColorSpace,
    /// The output size (not for DNG).
    pub size: Size,
    /// The destination folder; `None` means the default next to the source folder.
    pub destination: Option<PathBuf>,
    /// The file name template with `{name}`, `{look}`, `{date}`, `{seq}` and `{camera}`.
    pub name_template: String,
    /// What to do when the file exists.
    pub if_exists: IfExists,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: FormatKind::Jpeg,
            jpeg_quality: JpegQuality::default(),
            bit_depth: BitDepth::Sixteen,
            tiff_compression: TiffCompression::Lzw,
            color_space: ColorSpace::Srgb,
            size: Size::Full,
            destination: None,
            name_template: DEFAULT_NAME_TEMPLATE.to_owned(),
            if_exists: IfExists::KeepBoth,
        }
    }
}

impl ExportOptions {
    /// The format with the options that belong to it.
    #[must_use]
    pub fn format_spec(&self) -> ExportFormat {
        match self.format {
            FormatKind::Jpeg => ExportFormat::Jpeg {
                quality: self.jpeg_quality,
            },
            FormatKind::Tiff => ExportFormat::Tiff {
                bits: self.bit_depth,
                compression: self.tiff_compression,
            },
            FormatKind::Png => ExportFormat::Png {
                bits: self.bit_depth,
            },
            FormatKind::Dng => ExportFormat::Dng,
        }
    }

    /// Checks the options. The type system already covers the quality and bit depth.
    pub fn validate(&self) -> Result<(), ExportError> {
        if let Size::LongEdge { px } = self.size
            && !(MIN_LONG_EDGE_PX..=MAX_LONG_EDGE_PX).contains(&px)
        {
            return Err(ExportError::invalid(format!(
                "the long edge must be between {MIN_LONG_EDGE_PX} and {MAX_LONG_EDGE_PX} px, not {px}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_docs_09_table() {
        let o = ExportOptions::default();
        assert_eq!(o.format, FormatKind::Jpeg);
        assert_eq!(o.jpeg_quality.get(), 92);
        assert_eq!(o.bit_depth, BitDepth::Sixteen);
        assert_eq!(o.tiff_compression, TiffCompression::Lzw);
        assert_eq!(o.color_space, ColorSpace::Srgb);
        assert_eq!(o.size, Size::Full);
        assert_eq!(o.destination, None);
        assert_eq!(o.name_template, "{name}_{look}");
        assert_eq!(o.if_exists, IfExists::KeepBoth);
        assert_eq!(
            o.format_spec(),
            ExportFormat::Jpeg {
                quality: JpegQuality::new(92).unwrap()
            }
        );
    }

    #[test]
    fn serde_matches_the_frontend_shape() {
        let json = serde_json::to_value(ExportOptions::default()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "format": "jpeg",
                "jpeg_quality": 92,
                "bit_depth": 16,
                "tiff_compression": "lzw",
                "color_space": "srgb",
                "size": { "kind": "full" },
                "destination": null,
                "name_template": "{name}_{look}",
                "if_exists": "keep_both"
            })
        );
        let parsed: ExportOptions = serde_json::from_value(serde_json::json!({
            "format": "tiff",
            "jpeg_quality": 80,
            "bit_depth": 8,
            "tiff_compression": "zip",
            "color_space": "display_p3",
            "size": { "kind": "long_edge", "px": 3000 },
            "destination": "/tmp/out",
            "name_template": "{name}",
            "if_exists": "skip"
        }))
        .unwrap();
        assert_eq!(
            parsed.format_spec(),
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Zip
            }
        );
        assert_eq!(parsed.size, Size::LongEdge { px: 3000 });
        assert_eq!(parsed.color_space, ColorSpace::DisplayP3);
        assert_eq!(parsed.destination, Some(PathBuf::from("/tmp/out")));
    }

    #[test]
    fn missing_fields_take_their_defaults() {
        let parsed: ExportOptions =
            serde_json::from_value(serde_json::json!({ "format": "png" })).unwrap();
        assert_eq!(parsed.format, FormatKind::Png);
        assert_eq!(parsed.bit_depth, BitDepth::Sixteen);
        assert_eq!(parsed.name_template, "{name}_{look}");
        let empty: ExportOptions = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, ExportOptions::default());
    }

    #[test]
    fn out_of_range_values_are_rejected_when_parsing() {
        for bad in [
            serde_json::json!({ "jpeg_quality": 49 }),
            serde_json::json!({ "jpeg_quality": 101 }),
            serde_json::json!({ "bit_depth": 12 }),
            serde_json::json!({ "format": "bmp" }),
        ] {
            let mut full = serde_json::to_value(ExportOptions::default()).unwrap();
            full.as_object_mut()
                .unwrap()
                .extend(bad.as_object().unwrap().clone());
            assert!(
                serde_json::from_value::<ExportOptions>(full).is_err(),
                "{bad}"
            );
        }
        assert!(JpegQuality::new(50).is_ok());
        assert!(JpegQuality::new(100).is_ok());
        assert!(JpegQuality::new(49).is_err());
    }

    #[test]
    fn format_enum_round_trips() {
        for f in [
            ExportFormat::Jpeg {
                quality: JpegQuality::new(75).unwrap(),
            },
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            ExportFormat::Dng,
        ] {
            let json = serde_json::to_string(&f).unwrap();
            assert_eq!(serde_json::from_str::<ExportFormat>(&json).unwrap(), f);
        }
    }

    #[test]
    fn long_edge_sizing_rounds_and_never_upscales() {
        assert_eq!(Size::Full.output_dimensions(8640, 5760), (8640, 5760));
        assert_eq!(
            Size::LongEdge { px: 3000 }.output_dimensions(8640, 5760),
            (3000, 2000)
        );
        assert_eq!(
            Size::LongEdge { px: 3000 }.output_dimensions(5760, 8640),
            (2000, 3000)
        );
        assert_eq!(
            Size::LongEdge { px: 1000 }.output_dimensions(6000, 4001),
            (1000, 667)
        );
        assert_eq!(
            Size::LongEdge { px: 9000 }.output_dimensions(8640, 5760),
            (8640, 5760)
        );
        assert_eq!(
            Size::LongEdge { px: 100 }.output_dimensions(10_000, 1),
            (100, 1)
        );
    }

    #[test]
    fn ui_long_edge_grid_is_20_px_steps_ending_at_full() {
        let c = Size::ui_long_edge_choices(1100);
        assert_eq!(c, vec![1000, 1020, 1040, 1060, 1080, 1100]);
        assert!(Size::ui_long_edge_choices(999).is_empty());
        assert_eq!(Size::snap_long_edge(3009, 8640), 3000);
        assert_eq!(Size::snap_long_edge(3011, 8640), 3020);
        assert_eq!(Size::snap_long_edge(5, 8640), 1000);
        assert_eq!(Size::snap_long_edge(9000, 8640), 8640);
    }

    #[test]
    fn validate_rejects_absurd_long_edges() {
        let with = |px| ExportOptions {
            size: Size::LongEdge { px },
            ..ExportOptions::default()
        };
        assert!(with(3).validate().is_err());
        assert!(with(70_000).validate().is_err());
        assert!(with(2000).validate().is_ok());
    }

    #[test]
    fn command_line_words_parse() {
        assert_eq!("srgb".parse::<ColorSpace>().unwrap(), ColorSpace::Srgb);
        assert_eq!("P3".parse::<ColorSpace>().unwrap(), ColorSpace::DisplayP3);
        assert_eq!(
            "rec2020".parse::<ColorSpace>().unwrap(),
            ColorSpace::Rec2020
        );
        assert_eq!(
            "ZIP".parse::<TiffCompression>().unwrap(),
            TiffCompression::Zip
        );
        assert_eq!(
            "lzw".parse::<TiffCompression>().unwrap(),
            TiffCompression::Lzw
        );
        assert_eq!("jpg".parse::<FormatKind>().unwrap(), FormatKind::Jpeg);
        assert_eq!("tif".parse::<FormatKind>().unwrap(), FormatKind::Tiff);
        let err = "bmp".parse::<FormatKind>().unwrap_err();
        assert!(
            err.contains("unknown format") && err.contains("jpeg"),
            "{err}"
        );
    }

    #[test]
    fn cicp_codes_per_color_space() {
        assert_eq!(ColorSpace::Srgb.cicp(), [1, 13, 0, 1]);
        assert_eq!(ColorSpace::DisplayP3.cicp(), [12, 13, 0, 1]);
        assert_eq!(ColorSpace::Rec2020.cicp(), [9, 1, 0, 1]);
    }

    proptest::proptest! {
        /// A long-edge size keeps the aspect ratio to within a pixel, hits the requested long edge and never upscales.
        #[test]
        fn long_edge_sizing_invariants(w in 1_u32..20_000, h in 1_u32..20_000, px in 16_u32..30_000) {
            let (ow, oh) = Size::LongEdge { px }.output_dimensions(w, h);
            proptest::prop_assert!(ow >= 1 && oh >= 1);
            proptest::prop_assert!(ow <= w && oh <= h);
            if px < w.max(h) {
                proptest::prop_assert_eq!(ow.max(oh), px);
                // The short edge follows the aspect ratio to within one pixel.
                let exact = f64::from(w.min(h)) * f64::from(px) / f64::from(w.max(h));
                proptest::prop_assert!((f64::from(ow.min(oh)) - exact).abs() <= 1.0);
                // The long edge of the output is on the same axis as the long edge of the input (or the shape is square).
                proptest::prop_assert!((w >= h) == (ow >= oh) || w == h || ow == oh);
            } else {
                proptest::prop_assert_eq!((ow, oh), (w, h));
            }
        }
    }
}
