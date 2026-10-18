//! Sony maker note (0x927C): model ID 0xB001, ISO and the few enciphered tags we need; failures read as unknown (`docs/04-raw-decoding.md` section Container).
//!
//! The maker note is a normal IFD whose value offsets are absolute positions in the file (they
//! use the same base as the surrounding TIFF). Older bodies put a 12-byte header in front of it:
//! `SONY DSC \0\0\0` and `SONY CAM \0\0\0` on cameras, `SONY MOBILE`, `\0\0SONY PIC\0` on a few
//! compact models and `VHAB     \0` on the Hasselblad rebadges of Sony bodies. **Modern ARW files
//! have no header at all**: the directory starts at the first byte of the value (ExifTool calls
//! this layout `MakerNoteSony5`, "used in SR2 and ARW images"; the ILCE-9 dump in
//! github.com/lclevy/sony_raw shows `63 00` = 99 entries right at the start of the value). The
//! headerless layout is accepted only when the camera's `Make` says Sony (the `sony_family`
//! argument of [`parse`]).
//!
//! What is read, and from where:
//!
//! | Tag | Meaning | Use |
//! |---|---|---|
//! | `0xB001` | `SonyModelID`, 16-bit | [`SonyMakerNote::model_id`] |
//! | `0xB000` | `FileFormat`, four bytes (ARW version, `[3, 2, 0, 0]` is ARW 2.2) | [`SonyMakerNote::file_format`] |
//! | `0x2001` | `PreviewImage`: a 32-byte proprietary header (JPEG length, width, height, ...) then a JPEG | [`SonyMakerNote::preview`] |
//! | `0x9416` | enciphered table of the newest bodies; the 16-bit word at byte 4 is `SonyISO` | [`SonyMakerNote::iso`] |
//! | `0x9405` | enciphered table of older E-mount bodies; same word at byte 4, valid when the first byte is one of eight known values | [`SonyMakerNote::iso`] |
//! | `0xB020` | `CreativeStyle`, a string (`Standard`, `Vivid`, ...; newer bodies write Creative Look codes such as `FL` here too) | [`SonyMakerNote::creative_style`] |
//! | `0x9416` byte 55 | the same `CreativeStyle` as a code, enciphered; read only when `0xB020` is absent or empty, and not on the ILCE-7M5 | [`SonyMakerNote::creative_style`] |
//! | `0xB023` | `SceneMode`, 32 bit | [`SonyMakerNote::scene_mode`] |
//! | `0xB025` | `DynamicRangeOptimizer` (DRO), 32 bit | [`SonyMakerNote::dro_mode`] |
//!
//! `SonyISO` is `100 * 2^(16 - v / 256)` for the stored word `v`. The enciphered tags use a byte
//! substitution cipher ([`decipher_byte`]); only the bytes needed are deciphered, and a tag that
//! is too short, has an unknown first byte or yields an implausible value reads as unknown.
//!
//! The code tables are ExifTool's Sony tag documentation (`https://exiftool.org/TagNames/Sony.html`,
//! tags 0xB020, 0xB023, 0xB025 and the Tag9416 section for offset 55). The plain tags are the
//! ones every ILCE body in the local test set writes (ILCE-5000 to ILCE-7M4 and the A1), and
//! ExifTool's `0x200B` is not DRO (it is `MultiFrameNoiseReduction`).

use serde::{Deserialize, Serialize};

use crate::tiff::{ByteOrder, Entry, Ifd, Tiff, field_type};

/// `SonyModelID`.
pub const TAG_MODEL_ID: u16 = 0xB001;
/// `FileFormat` (ARW version bytes).
pub const TAG_FILE_FORMAT: u16 = 0xB000;
/// `PreviewImage`.
pub const TAG_PREVIEW_IMAGE: u16 = 0x2001;
/// Enciphered table with `SonyISO` for older E-mount bodies.
pub const TAG_9405: u16 = 0x9405;
/// Enciphered table with `SonyISO` for the newest bodies.
pub const TAG_9416: u16 = 0x9416;
/// `CreativeStyle` as a string (plain, not enciphered).
pub const TAG_CREATIVE_STYLE: u16 = 0xB020;
/// `SceneMode` (32 bit).
pub const TAG_SCENE_MODE: u16 = 0xB023;
/// `DynamicRangeOptimizer` (32 bit).
pub const TAG_DRO_MODE: u16 = 0xB025;

/// Length of the header in front of the maker note's IFD.
pub const HEADER_LEN: u64 = 12;
/// Length of the proprietary header in front of the JPEG in tag 0x2001.
pub const PREVIEW_HEADER_LEN: u64 = 32;

/// Byte of the deciphered 0x9416 table that holds the `CreativeStyle` code (ExifTool's Tag9416
/// section, index 55). Read only when tag 0xB020 is absent or empty.
pub const CREATIVE_STYLE_9416_OFFSET: usize = 55;
/// The `SonyModelID` of the ILCE-7M5, whose 0x9416 table has other offsets after its release mode
/// (ExifTool's Tag9416 section). Its `CreativeStyle` is not read from the table.
pub const MODEL_ID_ILCE_7M5: u16 = 407;

/// The fewest bytes a 0x9416 table can have (above its ISO word and the shortest layout
/// LibRaw trusts).
const TAG_9416_MIN_COUNT: u32 = 0x76;
/// The fewest bytes a 0x9405 table can have.
const TAG_9405_MIN_COUNT: u32 = 0x64;

/// First bytes of tag 0x9405 (as stored, enciphered) that mark a layout with `SonyISO` at byte 4.
const TAG_9405_VALID_FIRST_BYTES: [u8; 8] = [0x25, 0x3A, 0x76, 0x7E, 0x8B, 0x9A, 0xB3, 0xE1];

/// The plausible range for `SonyISO`; values outside it are treated as garbage.
const ISO_RANGE: std::ops::RangeInclusive<f64> = 6.0..=crate::exif::MAX_PLAUSIBLE_ISO as f64;

/// The substitution table that undoes Sony's byte cipher for tags 0x2010, 0x9050 and 0x94xx:
/// the enciphered byte `c = b^3 mod 249` for `b < 249`; bytes 249 to 255 are not changed.
const fn build_decipher_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut b = 0usize;
    while b < 256 {
        if b >= 249 {
            table[b] = b as u8;
        } else {
            // 248^3 fits comfortably in usize.
            table[(b * b * b) % 249] = b as u8;
        }
        b += 1;
    }
    table
}

const fn build_encipher_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut b = 0usize;
    while b < 256 {
        table[b] = if b >= 249 {
            b as u8
        } else {
            ((b * b * b) % 249) as u8
        };
        b += 1;
    }
    table
}

static DECIPHER: [u8; 256] = build_decipher_table();
static ENCIPHER: [u8; 256] = build_encipher_table();

/// Undoes Sony's byte cipher for one byte.
pub fn decipher_byte(byte: u8) -> u8 {
    DECIPHER[usize::from(byte)]
}

/// Applies Sony's byte cipher to one byte (used by the test writer).
pub fn encipher_byte(byte: u8) -> u8 {
    ENCIPHER[usize::from(byte)]
}

/// Sony's Dynamic Range Optimizer mode (`DynamicRangeOptimizer`, tag 0xB025). The codes are
/// ExifTool's Sony values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DroMode {
    /// 0: off.
    Off,
    /// 1: standard.
    Standard,
    /// 2: advanced auto.
    AdvancedAuto,
    /// 3: auto.
    Auto,
    /// 8 to 12: advanced level 1 to 5 (`Advanced Lv1` to `Advanced Lv5`).
    AdvancedLevel(u8),
    /// 16 to 23: level 1 to 8 (`Lv1` to `Lv8`).
    Level(u8),
    /// Any other value, kept as written.
    Other(u32),
}

impl DroMode {
    /// The mode for a `DynamicRangeOptimizer` code.
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Off,
            1 => Self::Standard,
            2 => Self::AdvancedAuto,
            3 => Self::Auto,
            8..=12 => Self::AdvancedLevel((code - 7) as u8),
            16..=23 => Self::Level((code - 15) as u8),
            other => Self::Other(other),
        }
    }
}

/// Sony's scene mode (`SceneMode`, tag 0xB023). The codes are ExifTool's Sony values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneMode {
    /// 0: standard.
    Standard,
    /// 1: portrait.
    Portrait,
    /// 2: text.
    Text,
    /// 3: night scene.
    NightScene,
    /// 4: sunset.
    Sunset,
    /// 5: sports.
    Sports,
    /// 6: landscape.
    Landscape,
    /// 7: night portrait.
    NightPortrait,
    /// 8: macro.
    Macro,
    /// 9: super macro.
    SuperMacro,
    /// 16: auto.
    Auto,
    /// 17: night view or portrait.
    NightViewPortrait,
    /// 18: sweep panorama.
    SweepPanorama,
    /// 19: handheld night shot.
    HandheldNightShot,
    /// 20: anti motion blur.
    AntiMotionBlur,
    /// 21: continuous priority AE.
    ContinuousPriorityAe,
    /// 22: auto plus.
    AutoPlus,
    /// 23: 3D sweep panorama.
    SweepPanorama3d,
    /// 24: superior auto.
    SuperiorAuto,
    /// 25: high sensitivity.
    HighSensitivity,
    /// 26: fireworks.
    Fireworks,
    /// 27: food.
    Food,
    /// 28: pet.
    Pet,
    /// 33: HDR.
    Hdr,
    /// 65535: the camera wrote "not applicable".
    NotAvailable,
    /// Any other value, kept as written.
    Other(u32),
}

impl SceneMode {
    /// The mode for a `SceneMode` code.
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Standard,
            1 => Self::Portrait,
            2 => Self::Text,
            3 => Self::NightScene,
            4 => Self::Sunset,
            5 => Self::Sports,
            6 => Self::Landscape,
            7 => Self::NightPortrait,
            8 => Self::Macro,
            9 => Self::SuperMacro,
            16 => Self::Auto,
            17 => Self::NightViewPortrait,
            18 => Self::SweepPanorama,
            19 => Self::HandheldNightShot,
            20 => Self::AntiMotionBlur,
            21 => Self::ContinuousPriorityAe,
            22 => Self::AutoPlus,
            23 => Self::SweepPanorama3d,
            24 => Self::SuperiorAuto,
            25 => Self::HighSensitivity,
            26 => Self::Fireworks,
            27 => Self::Food,
            28 => Self::Pet,
            33 => Self::Hdr,
            65_535 => Self::NotAvailable,
            other => Self::Other(other),
        }
    }
}

/// Sony's Creative Style (`CreativeStyle`, tag 0xB020, or the code in the 0x9416 table). Newer
/// bodies write Creative Look codes in the same setting (`FL`, `VV2`, `IN`, `SH`, `FL2`, `FL3`);
/// those, and any other name the crate does not list, read as [`CreativeStyle::Other`] with the
/// text the camera wrote.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreativeStyle {
    /// `Standard`.
    Standard,
    /// `Vivid`.
    Vivid,
    /// `Neutral`.
    Neutral,
    /// `Portrait`.
    Portrait,
    /// `Landscape`.
    Landscape,
    /// `Clear`.
    Clear,
    /// `Deep`.
    Deep,
    /// `Light`.
    Light,
    /// `Sunset`.
    Sunset,
    /// `Nightview` (night view or portrait).
    NightView,
    /// `Autumnleaves`.
    AutumnLeaves,
    /// `BW` (black and white).
    BlackAndWhite,
    /// `Sepia`.
    Sepia,
    /// Any other style: a Creative Look code (`FL`, `VV2`, ...), `AdobeRGB`, `Real`, `None`, `Off`
    /// (code 255), or a numeric code the crate does not name, written as `code <n>`.
    Other(String),
}

impl CreativeStyle {
    /// The style named by tag 0xB020 (matched without regard to case). An empty name is not a
    /// style; the caller checks that.
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "standard" => Self::Standard,
            "vivid" => Self::Vivid,
            "neutral" => Self::Neutral,
            "portrait" => Self::Portrait,
            "landscape" => Self::Landscape,
            "clear" => Self::Clear,
            "deep" => Self::Deep,
            "light" => Self::Light,
            "sunset" => Self::Sunset,
            "nightview" => Self::NightView,
            "autumnleaves" => Self::AutumnLeaves,
            "bw" => Self::BlackAndWhite,
            "sepia" => Self::Sepia,
            _ => Self::Other(name.to_owned()),
        }
    }

    /// The style for a code of the enciphered 0x9416 table (byte 55 once deciphered).
    pub fn from_table_code(code: u8) -> Self {
        match code {
            0 => Self::Standard,
            1 => Self::Vivid,
            2 => Self::Neutral,
            3 => Self::Portrait,
            4 => Self::Landscape,
            5 => Self::BlackAndWhite,
            6 => Self::Clear,
            7 => Self::Deep,
            8 => Self::Light,
            9 => Self::Sunset,
            10 => Self::NightView,
            11 => Self::AutumnLeaves,
            13 => Self::Sepia,
            15 => Self::Other("FL".to_owned()),
            16 => Self::Other("VV2".to_owned()),
            17 => Self::Other("IN".to_owned()),
            18 => Self::Other("SH".to_owned()),
            19 => Self::Other("FL2".to_owned()),
            20 => Self::Other("FL3".to_owned()),
            255 => Self::Other("Off".to_owned()),
            other => Self::Other(format!("code {other}")),
        }
    }
}

/// What was read from a Sony maker note.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SonyMakerNote {
    /// `SonyModelID` (tag 0xB001).
    pub model_id: Option<u16>,
    /// ARW version bytes (tag 0xB000); `[1, 0, 0, 0]` marks SR2.
    pub file_format: Option<[u8; 4]>,
    /// ISO from an enciphered table, rounded to an integer, with the tag it came from.
    pub iso: Option<(u32, u16)>,
    /// File position and length of the JPEG in `PreviewImage` (tag 0x2001), after its 32-byte
    /// header.
    pub preview: Option<(u64, u64)>,
    /// The Dynamic Range Optimizer mode (tag 0xB025).
    pub dro_mode: Option<DroMode>,
    /// The Creative Style: tag 0xB020 when it has a name, else the 0x9416 table's code.
    pub creative_style: Option<CreativeStyle>,
    /// The scene mode (tag 0xB023).
    pub scene_mode: Option<SceneMode>,
}

/// True when `prefix` (the first bytes of a MakerNote value) starts with one of the Sony headers.
pub fn has_sony_header(prefix: &[u8]) -> bool {
    prefix.starts_with(b"SONY DSC ")
        || prefix.starts_with(b"SONY CAM ")
        || prefix.starts_with(b"SONY MOBILE")
        || prefix.starts_with(b"\0\0SONY PIC\0")
        || prefix.starts_with(b"VHAB     \0")
}

/// The fewest bytes a headerless maker note can have: the entry count and one entry.
const HEADERLESS_MIN_LEN: u64 = 2 + 12;

/// Parses the maker note held by the EXIF `MakerNote` entry.
///
/// A value that starts with one of the Sony headers has its directory 12 bytes in. Any other
/// value is read as a headerless directory when `sony_family` is true (the file's `Make` is Sony,
/// or one of the Hasselblad rebadges of Sony bodies); as in ExifTool, a value that starts `01 00`
/// is not taken for one. Returns `None` when the value is not a Sony maker note or its directory
/// cannot be read.
pub fn parse(tiff: &Tiff, entry: &Entry, sony_family: bool) -> Option<SonyMakerNote> {
    if entry.tag != crate::tiff::tag::MAKER_NOTE {
        return None;
    }
    let (offset, length) = tiff.entry_location(entry).ok().flatten()?;
    let head = tiff.read_vec(offset, length.min(HEADER_LEN)).ok()?;
    let (ifd_at, headerless) = if length >= HEADER_LEN + 2 && has_sony_header(&head) {
        (offset + HEADER_LEN, false)
    } else if sony_family && length >= HEADERLESS_MIN_LEN && !head.starts_with(&[1, 0]) {
        (offset, true)
    } else {
        return None;
    };
    let ifd_len = length - (ifd_at - offset);
    // The maker note's byte order is not stated; ARW files use the TIFF's. Fall back to the
    // other one when the directory does not parse.
    let (reader, ifd) = [tiff.order(), other_order(tiff.order())]
        .into_iter()
        .find_map(|order| {
            let reader = tiff.reordered(order);
            let ifd = reader.read_ifd(ifd_at).ok()?;
            // Without a header there is nothing else to vouch for the bytes, so the entry table
            // must at least fit inside the value.
            if headerless && 2 + ifd.entries.len() as u64 * 12 > ifd_len {
                return None;
            }
            Some((reader, ifd))
        })?;

    let mut note = SonyMakerNote::default();
    if let Some(e) = ifd.get(TAG_MODEL_ID) {
        note.model_id = e.inline_uint().and_then(|v| u16::try_from(v).ok());
    }
    if let Some(e) = ifd.get(TAG_FILE_FORMAT)
        && e.ty == field_type::BYTE
        && e.count == 4
    {
        note.file_format = Some(e.raw_value());
    }
    if let Some(e) = ifd.get(TAG_PREVIEW_IMAGE) {
        note.preview = preview_location(&reader, e);
    }
    for tag in [TAG_9416, TAG_9405] {
        if let Some(e) = ifd.get(tag)
            && let Some(iso) = sony_iso(&reader, e)
        {
            note.iso = Some((iso, tag));
            break;
        }
    }
    note.dro_mode = ifd
        .get(TAG_DRO_MODE)
        .and_then(|e| unsigned_32(&reader, e))
        .map(DroMode::from_code);
    note.scene_mode = ifd
        .get(TAG_SCENE_MODE)
        .and_then(|e| unsigned_32(&reader, e))
        .map(SceneMode::from_code);
    note.creative_style = creative_style(&reader, &ifd, note.model_id);
    Some(note)
}

/// The first value of an unsigned 32-bit maker-note tag (SHORT or LONG). Any other type reads as
/// absent.
fn unsigned_32(reader: &Tiff, entry: &Entry) -> Option<u32> {
    if !matches!(entry.ty, field_type::SHORT | field_type::LONG) {
        return None;
    }
    let value = *reader.uints(entry, 1).ok()?.first()?;
    u32::try_from(value).ok()
}

/// `CreativeStyle`: the name in tag 0xB020 when it has one (an ASCII string). When it does not,
/// the code in the enciphered 0x9416 table at [`CREATIVE_STYLE_9416_OFFSET`], which is not read
/// for the ILCE-7M5 (its offsets differ) and needs the same table size as the ISO word.
fn creative_style(reader: &Tiff, ifd: &Ifd, model_id: Option<u16>) -> Option<CreativeStyle> {
    let named = ifd
        .get(TAG_CREATIVE_STYLE)
        .filter(|e| e.ty == field_type::ASCII)
        .and_then(|e| reader.ascii(e).ok())
        .filter(|name| !name.is_empty());
    if let Some(name) = named {
        return Some(CreativeStyle::from_name(&name));
    }
    let table = ifd.get(TAG_9416)?;
    if model_id == Some(MODEL_ID_ILCE_7M5)
        || !matches!(table.ty, field_type::UNDEFINED | field_type::BYTE)
        || table.count <= TAG_9416_MIN_COUNT
    {
        return None;
    }
    let head = reader
        .entry_prefix(table, CREATIVE_STYLE_9416_OFFSET as u64 + 1)
        .ok()?;
    let code = *head.get(CREATIVE_STYLE_9416_OFFSET)?;
    Some(CreativeStyle::from_table_code(decipher_byte(code)))
}

fn other_order(order: ByteOrder) -> ByteOrder {
    match order {
        ByteOrder::Little => ByteOrder::Big,
        ByteOrder::Big => ByteOrder::Little,
    }
}

/// Locates the JPEG inside tag 0x2001: skip the 32-byte header, require `FF D8 FF`, and use the
/// header's length field when it fits and a JPEG end marker (`FF D9`) sits right where it says.
/// The layout of the header is not documented anywhere (ExifTool skips it and takes everything
/// after it), so a length that does not end on a JPEG end marker is not trusted: it could cut a
/// good preview short. Without a trusted length the whole value after the header is used.
fn preview_location(tiff: &Tiff, entry: &Entry) -> Option<(u64, u64)> {
    let (offset, total) = tiff.entry_location(entry).ok().flatten()?;
    if total <= PREVIEW_HEADER_LEN + 3 {
        return None;
    }
    let head = tiff.read_vec(offset, PREVIEW_HEADER_LEN + 3).ok()?;
    if head.get(PREVIEW_HEADER_LEN as usize..)? != [0xFF, 0xD8, 0xFF] {
        return None;
    }
    let available = total - PREVIEW_HEADER_LEN;
    let declared = u64::from(tiff.order().u32(&head, 0)?);
    let jpeg_at = offset + PREVIEW_HEADER_LEN;
    let declared_is_trusted = declared >= 4
        && declared <= available
        && tiff
            .read_vec(jpeg_at + declared - 2, 2)
            .is_ok_and(|end| end == [0xFF, 0xD9]);
    let length = if declared_is_trusted {
        declared
    } else {
        available
    };
    Some((jpeg_at, length))
}

/// Reads `SonyISO` from the enciphered table 0x9416 or 0x9405.
fn sony_iso(tiff: &Tiff, entry: &Entry) -> Option<u32> {
    if !matches!(entry.ty, field_type::UNDEFINED | field_type::BYTE) {
        return None;
    }
    // The same minimum sizes LibRaw demands before it trusts the layout.
    let min_count = if entry.tag == TAG_9405 {
        TAG_9405_MIN_COUNT
    } else {
        TAG_9416_MIN_COUNT
    };
    if entry.count <= min_count {
        return None;
    }
    let head = tiff.entry_prefix(entry, 8).ok()?;
    if head.len() < 8 {
        return None;
    }
    if entry.tag == TAG_9405 && !TAG_9405_VALID_FIRST_BYTES.contains(&head[0]) {
        return None;
    }
    let word = [decipher_byte(head[4]), decipher_byte(head[5])];
    let raw = match tiff.order() {
        ByteOrder::Little => u16::from_le_bytes(word),
        ByteOrder::Big => u16::from_be_bytes(word),
    };
    iso_from_sony_word(raw)
}

/// `SonyISO = 100 * 2^(16 - v / 256)`, rounded; `None` when implausible.
pub fn iso_from_sony_word(word: u16) -> Option<u32> {
    let iso = 100.0 * (16.0 - f64::from(word) / 256.0).exp2();
    ISO_RANGE.contains(&iso).then(|| iso.round() as u32)
}

/// The stored word for an ISO value (inverse of [`iso_from_sony_word`], rounded to nearest).
pub fn sony_word_for_iso(iso: u32) -> u16 {
    let word = 256.0 * (16.0 - (f64::from(iso.max(1)) / 100.0).log2());
    word.round().clamp(0.0, f64::from(u16::MAX)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cipher_tables_are_inverse_permutations() {
        for b in 0..=255u8 {
            assert_eq!(decipher_byte(encipher_byte(b)), b, "byte {b}");
            assert_eq!(encipher_byte(decipher_byte(b)), b, "byte {b}");
        }
        // Spot checks of the formula c = b^3 mod 249: 2^3 = 8, 10^3 = 1000 mod 249 = 4.
        assert_eq!(encipher_byte(2), 8);
        assert_eq!(encipher_byte(10), 4);
        assert_eq!(decipher_byte(8), 2);
        // Bytes 249..=255 and the fixed points are unchanged.
        for b in [0u8, 1, 82, 83, 84, 165, 166, 167, 248, 249, 255] {
            assert_eq!(encipher_byte(b), b);
        }
    }

    #[test]
    fn iso_word_round_trips_for_the_standard_iso_series() {
        for iso in [
            50u32, 64, 100, 125, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600, 102_400,
        ] {
            let word = sony_word_for_iso(iso);
            let back = iso_from_sony_word(word).expect("plausible");
            let error = (f64::from(back) - f64::from(iso)).abs() / f64::from(iso);
            assert!(error < 0.01, "iso {iso} -> word {word} -> {back}");
        }
        // Whole stops are exact.
        assert_eq!(iso_from_sony_word(sony_word_for_iso(6400)), Some(6400));
        assert_eq!(sony_word_for_iso(100), 4096);
    }

    // --- hand-built maker notes -------------------------------------------------------------

    /// How an entry's value is stored in a test maker note.
    enum Val {
        Inline([u8; 4]),
        Data(Vec<u8>),
    }

    /// Builds a buffer holding an 8-byte filler, then a maker note at offset 8 with the given
    /// header and entries (value data after the table, absolute offsets), and the EXIF entry
    /// that points at it.
    fn note(
        order: ByteOrder,
        header: &[u8],
        entries: Vec<(u16, u16, u32, Val)>,
    ) -> (Vec<u8>, Entry) {
        let put16 = |v: u16| match order {
            ByteOrder::Little => v.to_le_bytes(),
            ByteOrder::Big => v.to_be_bytes(),
        };
        let put32 = |v: u32| match order {
            ByteOrder::Little => v.to_le_bytes(),
            ByteOrder::Big => v.to_be_bytes(),
        };
        let start = 8usize;
        let ifd_at = start + header.len();
        let table_len = 2 + entries.len() * 12 + 4;
        let mut table = put16(entries.len() as u16).to_vec();
        let mut data: Vec<u8> = Vec::new();
        for (tag, ty, count, value) in entries {
            table.extend_from_slice(&put16(tag));
            table.extend_from_slice(&put16(ty));
            table.extend_from_slice(&put32(count));
            match value {
                Val::Inline(v) => table.extend_from_slice(&v),
                Val::Data(d) => {
                    let at = ifd_at + table_len + data.len();
                    table.extend_from_slice(&put32(at as u32));
                    data.extend_from_slice(&d);
                }
            }
        }
        table.extend_from_slice(&put32(0));
        let mut buf = vec![0u8; start];
        buf.extend_from_slice(header);
        buf.extend_from_slice(&table);
        buf.extend_from_slice(&data);
        let value_len = (buf.len() - start) as u32;
        let mut entry = Vec::new();
        entry.extend_from_slice(&put16(crate::tiff::tag::MAKER_NOTE));
        entry.extend_from_slice(&put16(field_type::UNDEFINED));
        entry.extend_from_slice(&put32(value_len));
        entry.extend_from_slice(&put32(start as u32));
        let entry = Entry::parse(&entry, order).expect("entry");
        (buf, entry)
    }

    const HEADER: &[u8] = b"SONY DSC \0\0\0";

    fn short(order: ByteOrder, v: u16) -> [u8; 4] {
        let b = match order {
            ByteOrder::Little => v.to_le_bytes(),
            ByteOrder::Big => v.to_be_bytes(),
        };
        [b[0], b[1], 0, 0]
    }

    fn parse_note(order: ByteOrder, buf: &[u8], entry: &Entry) -> Option<SonyMakerNote> {
        parse_note_as(order, buf, entry, true)
    }

    fn parse_note_as(
        order: ByteOrder,
        buf: &[u8],
        entry: &Entry,
        sony_family: bool,
    ) -> Option<SonyMakerNote> {
        let source = crate::tiff::MemSource(buf);
        let tiff = Tiff::headerless(&source, order, 0);
        parse(&tiff, entry, sony_family)
    }

    #[test]
    fn reads_model_id_and_file_format_in_either_byte_order() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let (buf, entry) = note(
                order,
                HEADER,
                vec![
                    (
                        TAG_FILE_FORMAT,
                        field_type::BYTE,
                        4,
                        Val::Inline([3, 3, 2, 0]),
                    ),
                    (
                        TAG_MODEL_ID,
                        field_type::SHORT,
                        1,
                        Val::Inline(short(order, 375)),
                    ),
                ],
            );
            let parsed = parse_note(order, &buf, &entry).expect("maker note");
            assert_eq!(parsed.model_id, Some(375), "{order:?}");
            assert_eq!(parsed.file_format, Some([3, 3, 2, 0]));
            assert_eq!(parsed.iso, None);
            assert_eq!(parsed.preview, None);
        }
    }

    #[test]
    fn a_maker_note_in_the_other_byte_order_than_the_file_is_still_read() {
        // The reader is told the file is little endian; the maker note is big endian.
        let (buf, entry) = note(
            ByteOrder::Big,
            HEADER,
            vec![(
                TAG_MODEL_ID,
                field_type::SHORT,
                1,
                Val::Inline(short(ByteOrder::Big, 384)),
            )],
        );
        let tiff = Tiff::headerless(&buf, ByteOrder::Little, 0);
        let parsed = parse(&tiff, &entry, true).expect("maker note");
        assert_eq!(parsed.model_id, Some(384));
    }

    /// The layout of every modern ARW (ExifTool's `MakerNoteSony5`): the directory starts at the
    /// first byte of the value. The first bytes of the ILCE-9 note in the lclevy/sony_raw dump are
    /// `63 00 03 10 04 00 10 00 ...`: 99 entries, tag 0x1003, type 4, count 16.
    #[test]
    fn a_maker_note_without_a_header_is_read_when_the_make_is_sony() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let (buf, entry) = note(
                order,
                &[],
                vec![
                    (
                        TAG_FILE_FORMAT,
                        field_type::BYTE,
                        4,
                        Val::Inline([3, 3, 3, 0]),
                    ),
                    (
                        TAG_MODEL_ID,
                        field_type::SHORT,
                        1,
                        Val::Inline(short(order, 358)),
                    ),
                    (
                        TAG_9416,
                        field_type::UNDEFINED,
                        0x200,
                        Val::Data(iso_table(0x200, 0, 1600)),
                    ),
                ],
            );
            let parsed = parse_note_as(order, &buf, &entry, true).expect("headerless note");
            assert_eq!(parsed.model_id, Some(358), "{order:?}");
            assert_eq!(parsed.file_format, Some([3, 3, 3, 0]));
            if order == ByteOrder::Little {
                // (The test table stores its ISO word little endian, as ARW files do.)
                assert_eq!(parsed.iso, Some((1600, TAG_9416)));
            }
            // Another make's headerless value is not ours to read.
            assert_eq!(parse_note_as(order, &buf, &entry, false), None, "{order:?}");
        }
    }

    #[test]
    fn a_headerless_value_must_look_like_a_directory() {
        let model = |order| {
            (
                TAG_MODEL_ID,
                field_type::SHORT,
                1,
                Val::Inline(short(order, 1)),
            )
        };
        // One entry means the value starts `01 00`: ExifTool does not take that for a Sony note.
        let (buf, entry) = note(ByteOrder::Little, &[], vec![model(ByteOrder::Little)]);
        assert_eq!(&buf[8..10], &[1, 0]);
        assert_eq!(parse_note_as(ByteOrder::Little, &buf, &entry, true), None);
        // Two entries are fine.
        let (buf, entry) = note(
            ByteOrder::Little,
            &[],
            vec![
                model(ByteOrder::Little),
                (
                    TAG_FILE_FORMAT,
                    field_type::BYTE,
                    4,
                    Val::Inline([3, 3, 0, 0]),
                ),
            ],
        );
        assert!(parse_note_as(ByteOrder::Little, &buf, &entry, true).is_some());
        // An entry table that does not fit in the value is rejected, however plausible its types.
        let (mut buf, entry) = note(
            ByteOrder::Little,
            &[],
            vec![model(ByteOrder::Little), model(ByteOrder::Little)],
        );
        buf[8..10].copy_from_slice(&30u16.to_le_bytes());
        assert_eq!(parse_note_as(ByteOrder::Little, &buf, &entry, true), None);
        // Arbitrary bytes (a count above the limit) and values too short for one entry.
        let junk = Entry::parse(
            &[0x7C, 0x92, 7, 0, 40, 0, 0, 0, 8, 0, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(
            parse_note_as(ByteOrder::Little, &[0xEE; 64], &junk, true),
            None
        );
        let short_value = Entry::parse(
            &[0x7C, 0x92, 7, 0, 13, 0, 0, 0, 8, 0, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(
            parse_note_as(ByteOrder::Little, &[0x02; 64], &short_value, true),
            None
        );
    }

    #[test]
    fn other_vendors_and_malformed_values_are_not_sony_maker_notes() {
        let model = (
            TAG_MODEL_ID,
            field_type::SHORT,
            1,
            Val::Inline([1, 0, 0, 0]),
        );
        // Wrong header.
        let (buf, entry) = note(ByteOrder::Little, b"Nikon\0\x02\x10\0\0\0\0", vec![model]);
        assert_eq!(parse_note(ByteOrder::Little, &buf, &entry), None);
        // Wrong tag on the pointing entry.
        let (buf, mut entry) = note(
            ByteOrder::Little,
            HEADER,
            vec![(
                TAG_MODEL_ID,
                field_type::SHORT,
                1,
                Val::Inline([1, 0, 0, 0]),
            )],
        );
        entry = Entry::parse(
            &[0x01, 0x01, 7, 0, entry.count as u8, 0, 0, 0, 8, 0, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(parse_note(ByteOrder::Little, &buf, &entry), None);
        // A value that fits in the entry cannot hold a maker note.
        let tiny = Entry::parse(
            &[0x7C, 0x92, 7, 0, 3, 0, 0, 0, b'a', b'b', b'c', 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(parse_note(ByteOrder::Little, &buf, &tiny), None);
        // Header but no room for a directory.
        let short_value = Entry::parse(
            &[0x7C, 0x92, 7, 0, 12, 0, 0, 0, 8, 0, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(parse_note(ByteOrder::Little, &buf, &short_value), None);
        // A directory of garbage after a good header.
        let mut junk = vec![0u8; 8];
        junk.extend_from_slice(HEADER);
        junk.extend_from_slice(&[0xFF; 40]);
        let junk_entry = Entry::parse(
            &[0x7C, 0x92, 7, 0, junk.len() as u8 - 8, 0, 0, 0, 8, 0, 0, 0],
            ByteOrder::Little,
        )
        .expect("entry");
        assert_eq!(parse_note(ByteOrder::Little, &junk, &junk_entry), None);
    }

    /// A deciphered-ISO table: `len` bytes, the ISO word at byte 4 enciphered.
    fn iso_table(len: usize, first: u8, iso: u32) -> Vec<u8> {
        let mut table = vec![0u8; len];
        table[0] = first;
        let word = sony_word_for_iso(iso).to_le_bytes();
        table[4] = encipher_byte(word[0]);
        table[5] = encipher_byte(word[1]);
        table
    }

    fn iso_of(tag: u16, ty: u16, table: Vec<u8>) -> Option<(u32, u16)> {
        let count = table.len() as u32;
        let (buf, entry) = note(
            ByteOrder::Little,
            HEADER,
            vec![(tag, ty, count, Val::Data(table))],
        );
        parse_note(ByteOrder::Little, &buf, &entry).and_then(|n| n.iso)
    }

    #[test]
    fn sony_iso_comes_from_9416_or_a_valid_9405() {
        assert_eq!(
            iso_of(
                TAG_9416,
                field_type::UNDEFINED,
                iso_table(0x200, 0x00, 6400)
            ),
            Some((6400, TAG_9416))
        );
        // 0x9405 needs one of the eight known first bytes.
        for first in TAG_9405_VALID_FIRST_BYTES {
            assert_eq!(
                iso_of(
                    TAG_9405,
                    field_type::UNDEFINED,
                    iso_table(0x100, first, 800)
                ),
                Some((800, TAG_9405)),
                "first byte {first:#x}"
            );
        }
        assert_eq!(
            iso_of(TAG_9405, field_type::UNDEFINED, iso_table(0x100, 0x11, 800)),
            None
        );
        // Tables shorter than LibRaw's minimum are not trusted.
        assert_eq!(
            iso_of(TAG_9416, field_type::UNDEFINED, iso_table(0x76, 0, 800)),
            None
        );
        assert_eq!(
            iso_of(TAG_9405, field_type::UNDEFINED, iso_table(0x64, 0x8B, 800)),
            None
        );
        // The wrong field type is not a table.
        assert_eq!(
            iso_of(TAG_9416, field_type::SHORT, iso_table(0x200, 0, 800)),
            None
        );
        // An all-zero table has the implausible ISO 6.5 million.
        assert_eq!(
            iso_of(TAG_9416, field_type::UNDEFINED, vec![0u8; 0x200]),
            None
        );
    }

    #[test]
    fn tag_9416_wins_over_9405_when_both_exist() {
        let t9416 = iso_table(0x200, 0, 3200);
        let t9405 = iso_table(0x100, 0x8B, 100);
        let (buf, entry) = note(
            ByteOrder::Little,
            HEADER,
            vec![
                (TAG_9405, field_type::UNDEFINED, 0x100, Val::Data(t9405)),
                (TAG_9416, field_type::UNDEFINED, 0x200, Val::Data(t9416)),
            ],
        );
        let parsed = parse_note(ByteOrder::Little, &buf, &entry).expect("maker note");
        assert_eq!(parsed.iso, Some((3200, TAG_9416)));
    }

    fn preview_data(declared: u32, jpeg: &[u8]) -> Vec<u8> {
        let mut data = declared.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 28]);
        data.extend_from_slice(jpeg);
        data
    }

    fn preview_of(data: Vec<u8>) -> Option<(u64, u64)> {
        let count = data.len() as u32;
        let (buf, entry) = note(
            ByteOrder::Little,
            HEADER,
            vec![(
                TAG_PREVIEW_IMAGE,
                field_type::UNDEFINED,
                count,
                Val::Data(data),
            )],
        );
        let offset = 8 + HEADER.len() + 2 + 12 + 4;
        let found = parse_note(ByteOrder::Little, &buf, &entry).and_then(|n| n.preview);
        found.map(|(o, l)| (o - offset as u64, l))
    }

    #[test]
    fn preview_skips_the_32_byte_header_and_honours_the_declared_length() {
        let jpeg = [0xFF, 0xD8, 0xFF, 0xDB, 1, 2, 3, 4, 5, 6, 0xFF, 0xD9];
        // Declared length matches.
        assert_eq!(preview_of(preview_data(12, &jpeg)), Some((32, 12)));
        // Declared shorter than what is stored, with the end marker where it says (the JPEG is
        // followed by padding): trust the declaration.
        let padded = [&jpeg[..], &[0, 0, 0, 0]].concat();
        assert_eq!(preview_of(preview_data(12, &padded)), Some((32, 12)));
        // Declared length that ends in the middle of the JPEG is no length: a preview cut short
        // would not decode, so the whole value after the header is used.
        assert_eq!(preview_of(preview_data(10, &jpeg)), Some((32, 12)));
        assert_eq!(preview_of(preview_data(5, &jpeg)), Some((32, 12)));
        // Declared longer than what is stored, zero or nonsense: use what is there.
        assert_eq!(preview_of(preview_data(500, &jpeg)), Some((32, 12)));
        assert_eq!(preview_of(preview_data(0, &jpeg)), Some((32, 12)));
        // Not a JPEG after the header, or no room for one.
        assert_eq!(
            preview_of(preview_data(12, &[0, 0xD8, 0xFF, 1, 2, 3, 4, 5])),
            None
        );
        assert_eq!(preview_of(vec![0u8; 34]), None);
        assert_eq!(preview_of(vec![0xFF; 20]), None);
    }

    #[test]
    fn implausible_iso_words_are_rejected() {
        assert_eq!(iso_from_sony_word(0), None);
        assert_eq!(iso_from_sony_word(u16::MAX), None);
    }

    #[test]
    fn recognises_the_sony_headers() {
        assert!(has_sony_header(b"SONY DSC \0\0\0\x10\0"));
        assert!(has_sony_header(b"SONY CAM \0\0\0"));
        assert!(has_sony_header(b"SONY MOBILE\0"));
        assert!(has_sony_header(b"\0\0SONY PIC\0xx"));
        assert!(has_sony_header(b"VHAB     \0\0\0"));
        assert!(!has_sony_header(b"Nikon\0\x02\x10"));
        assert!(!has_sony_header(b"SONY"));
        assert!(!has_sony_header(b""));
    }

    // --- DRO, scene mode and creative style -------------------------------------------------

    /// A 32-bit value in the file's byte order (for an inline LONG entry).
    fn long(order: ByteOrder, v: u32) -> [u8; 4] {
        match order {
            ByteOrder::Little => v.to_le_bytes(),
            ByteOrder::Big => v.to_be_bytes(),
        }
    }

    /// Parses a maker note with the older header and the given entries.
    fn note_with(order: ByteOrder, entries: Vec<(u16, u16, u32, Val)>) -> Option<SonyMakerNote> {
        let (buf, entry) = note(order, HEADER, entries);
        parse_note(order, &buf, &entry)
    }

    /// A deciphered 0x9416 table of `len` zero bytes whose `CreativeStyle` byte is `code`.
    fn style_table(len: usize, code: u8) -> Vec<u8> {
        let mut table = vec![0u8; len];
        table[CREATIVE_STYLE_9416_OFFSET] = encipher_byte(code);
        table
    }

    /// A 16-byte ASCII value holding `name`, NUL padded as Sony writes it.
    fn name_bytes(name: &str) -> Vec<u8> {
        let mut value = name.as_bytes().to_vec();
        value.resize(16, 0);
        value
    }

    #[test]
    fn dro_mode_codes_map_to_the_sony_modes() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            for (code, mode) in [
                (0, DroMode::Off),
                (1, DroMode::Standard),
                (2, DroMode::AdvancedAuto),
                (3, DroMode::Auto),
                (8, DroMode::AdvancedLevel(1)),
                (12, DroMode::AdvancedLevel(5)),
                (16, DroMode::Level(1)),
                (23, DroMode::Level(8)),
                (4, DroMode::Other(4)),
                (13, DroMode::Other(13)),
                (24, DroMode::Other(24)),
                (u32::MAX, DroMode::Other(u32::MAX)),
            ] {
                let entries = vec![(
                    TAG_DRO_MODE,
                    field_type::LONG,
                    1,
                    Val::Inline(long(order, code)),
                )];
                let parsed = note_with(order, entries).expect("maker note");
                assert_eq!(parsed.dro_mode, Some(mode), "code {code} in {order:?}");
            }
        }
    }

    #[test]
    fn scene_mode_codes_map_and_not_applicable_is_its_own_mode() {
        let order = ByteOrder::Little;
        for (code, mode) in [
            (0, SceneMode::Standard),
            (7, SceneMode::NightPortrait),
            (16, SceneMode::Auto),
            (23, SceneMode::SweepPanorama3d),
            (33, SceneMode::Hdr),
            (65_535, SceneMode::NotAvailable),
            (99, SceneMode::Other(99)),
        ] {
            let entries = vec![(
                TAG_SCENE_MODE,
                field_type::LONG,
                1,
                Val::Inline(long(order, code)),
            )];
            let parsed = note_with(order, entries).expect("maker note");
            assert_eq!(parsed.scene_mode, Some(mode), "code {code}");
        }
    }

    #[test]
    fn plain_tags_of_the_wrong_type_or_missing_read_as_none() {
        let order = ByteOrder::Little;
        let parsed = note_with(
            order,
            vec![
                (TAG_DRO_MODE, field_type::ASCII, 4, Val::Inline(*b"3\0\0\0")),
                (
                    TAG_SCENE_MODE,
                    field_type::BYTE,
                    4,
                    Val::Inline([16, 0, 0, 0]),
                ),
                // A number where the name belongs.
                (
                    TAG_CREATIVE_STYLE,
                    field_type::LONG,
                    1,
                    Val::Inline(long(order, 1)),
                ),
            ],
        )
        .expect("maker note");
        assert_eq!(parsed.dro_mode, None);
        assert_eq!(parsed.scene_mode, None);
        assert_eq!(parsed.creative_style, None);

        let parsed = note_with(
            order,
            vec![(
                TAG_MODEL_ID,
                field_type::SHORT,
                1,
                Val::Inline(short(order, 375)),
            )],
        )
        .expect("maker note");
        assert_eq!(parsed.dro_mode, None);
        assert_eq!(parsed.scene_mode, None);
        assert_eq!(parsed.creative_style, None);
    }

    #[test]
    fn creative_style_names_map_to_the_styles() {
        let order = ByteOrder::Little;
        let other = |s: &str| CreativeStyle::Other(s.to_owned());
        for (name, style) in [
            ("Standard", CreativeStyle::Standard),
            ("Vivid", CreativeStyle::Vivid),
            ("Neutral", CreativeStyle::Neutral),
            ("Portrait", CreativeStyle::Portrait),
            ("Landscape", CreativeStyle::Landscape),
            ("Clear", CreativeStyle::Clear),
            ("Deep", CreativeStyle::Deep),
            ("Light", CreativeStyle::Light),
            ("Sunset", CreativeStyle::Sunset),
            ("Nightview", CreativeStyle::NightView),
            ("Autumnleaves", CreativeStyle::AutumnLeaves),
            ("BW", CreativeStyle::BlackAndWhite),
            ("Sepia", CreativeStyle::Sepia),
            // Matched without regard to case.
            ("vivid", CreativeStyle::Vivid),
            // Creative Look codes and other names are kept as written.
            ("FL", other("FL")),
            ("VV2", other("VV2")),
            ("AdobeRGB", other("AdobeRGB")),
            ("None", other("None")),
        ] {
            let entries = vec![(
                TAG_CREATIVE_STYLE,
                field_type::ASCII,
                16,
                Val::Data(name_bytes(name)),
            )];
            let parsed = note_with(order, entries).expect("maker note");
            assert_eq!(parsed.creative_style, Some(style), "{name}");
        }
    }

    #[test]
    fn a_name_wins_over_the_table_and_an_empty_name_falls_back_to_it() {
        let order = ByteOrder::Little;
        let table = || {
            (
                TAG_9416,
                field_type::UNDEFINED,
                0x200,
                Val::Data(style_table(0x200, 2)),
            )
        };
        let name = |s: &str| {
            (
                TAG_CREATIVE_STYLE,
                field_type::ASCII,
                16,
                Val::Data(name_bytes(s)),
            )
        };

        let parsed = note_with(order, vec![name("Vivid"), table()]).expect("maker note");
        assert_eq!(parsed.creative_style, Some(CreativeStyle::Vivid));

        let parsed = note_with(order, vec![name(""), table()]).expect("maker note");
        assert_eq!(parsed.creative_style, Some(CreativeStyle::Neutral));

        let parsed = note_with(order, vec![name("")]).expect("maker note");
        assert_eq!(parsed.creative_style, None);
    }

    #[test]
    fn creative_style_table_codes_map_to_the_styles() {
        let order = ByteOrder::Little;
        for (code, style) in [
            (0, CreativeStyle::Standard),
            (1, CreativeStyle::Vivid),
            (2, CreativeStyle::Neutral),
            (3, CreativeStyle::Portrait),
            (4, CreativeStyle::Landscape),
            (5, CreativeStyle::BlackAndWhite),
            (6, CreativeStyle::Clear),
            (7, CreativeStyle::Deep),
            (8, CreativeStyle::Light),
            (9, CreativeStyle::Sunset),
            (10, CreativeStyle::NightView),
            (11, CreativeStyle::AutumnLeaves),
            (13, CreativeStyle::Sepia),
            (15, CreativeStyle::Other("FL".to_owned())),
            (16, CreativeStyle::Other("VV2".to_owned())),
            (17, CreativeStyle::Other("IN".to_owned())),
            (18, CreativeStyle::Other("SH".to_owned())),
            (19, CreativeStyle::Other("FL2".to_owned())),
            (20, CreativeStyle::Other("FL3".to_owned())),
            (255, CreativeStyle::Other("Off".to_owned())),
            (12, CreativeStyle::Other("code 12".to_owned())),
            (21, CreativeStyle::Other("code 21".to_owned())),
        ] {
            let entries = vec![(
                TAG_9416,
                field_type::UNDEFINED,
                0x200,
                Val::Data(style_table(0x200, code)),
            )];
            let parsed = note_with(order, entries).expect("maker note");
            assert_eq!(parsed.creative_style, Some(style), "code {code}");
        }
    }

    #[test]
    fn the_table_is_not_read_for_the_ilce_7m5_or_when_it_is_too_short_or_the_wrong_type() {
        let order = ByteOrder::Little;
        let model = |id: u16| {
            (
                TAG_MODEL_ID,
                field_type::SHORT,
                1,
                Val::Inline(short(order, id)),
            )
        };
        let table =
            |ty: u16, len: usize| (TAG_9416, ty, len as u32, Val::Data(style_table(len, 2)));

        // The ILCE-7M5 has other offsets: no style from the table.
        let parsed = note_with(
            order,
            vec![
                model(MODEL_ID_ILCE_7M5),
                table(field_type::UNDEFINED, 0x200),
            ],
        )
        .expect("maker note");
        assert_eq!(parsed.creative_style, None);
        // Another body with the same table reads it.
        let parsed = note_with(order, vec![model(388), table(field_type::UNDEFINED, 0x200)])
            .expect("maker note");
        assert_eq!(parsed.creative_style, Some(CreativeStyle::Neutral));
        // A table no longer than the ISO word needs is not trusted.
        let parsed =
            note_with(order, vec![table(field_type::UNDEFINED, 0x76)]).expect("maker note");
        assert_eq!(parsed.creative_style, None);
        // A table of another type is not a table.
        let parsed = note_with(order, vec![table(field_type::SHORT, 0x200)]).expect("maker note");
        assert_eq!(parsed.creative_style, None);
    }
}
