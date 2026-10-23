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
//! | `0x2027` | `FocusLocation`, int16u\[4\]: the camera's frame width and height, then the focus point (x, y), in sensor orientation | [`SonyMakerNote::focus`] |
//! | `0x204A` | `FocusLocation2`, the same within one pixel on newer bodies; preferred when present | [`SonyMakerNote::focus`] |
//! | `0x2037` | `FocusFrameSize`, six undefined bytes read as int16u\[3\]: the AF box's width and height, then `257` when valid (`0`: n/a) | [`SonyFocus::box_size`] |
//! | `0x201C` | `AFAreaModeSetting`, int8u (its meaning depends on the body, `docs/20-auto-crop.md` §4) | [`SonyMakerNote::af_area_setting`] |
//!
//! `SonyISO` is `100 * 2^(16 - v / 256)` for the stored word `v`. The enciphered tags use a byte
//! substitution cipher ([`decipher_byte`]); only the bytes needed are deciphered, and a tag that
//! is too short, has an unknown first byte or yields an implausible value reads as unknown.

use crate::tiff::{ByteOrder, Entry, Tiff, field_type};

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
/// `FocusLocation`.
pub const TAG_FOCUS_LOCATION: u16 = 0x2027;
/// `FocusLocation2`.
pub const TAG_FOCUS_LOCATION2: u16 = 0x204A;
/// `FocusFrameSize`.
pub const TAG_FOCUS_FRAME_SIZE: u16 = 0x2037;
/// `AFAreaModeSetting`.
pub const TAG_AF_AREA_MODE_SETTING: u16 = 0x201C;

/// Length of the header in front of the maker note's IFD.
pub const HEADER_LEN: u64 = 12;
/// Length of the proprietary header in front of the JPEG in tag 0x2001.
pub const PREVIEW_HEADER_LEN: u64 = 32;

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
    /// Where the camera focused (tags 0x204A or 0x2027, with 0x2037), as stored.
    pub focus: Option<SonyFocus>,
    /// `AFAreaModeSetting` (tag 0x201C), as stored.
    pub af_area_setting: Option<u8>,
}

/// Where the camera focused, as the maker note stores it: in the units of the camera's own frame
/// (the size of its JPEG, for example 6000 x 4000 on a 6048 x 4024 raw), in sensor orientation,
/// origin at the top left. `docs/20-auto-crop.md` §4 says how Auto crop reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SonyFocus {
    /// The width and height of the camera's frame.
    pub frame: [u16; 2],
    /// The focus point. Sony writes the frame's centre when it does not know the point, and
    /// `0 0 0 0` for panoramas.
    pub point: [u16; 2],
    /// The AF box around the point (tag 0x2037), when the camera marked it valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_size: Option<[u16; 2]>,
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
    note.focus = [TAG_FOCUS_LOCATION2, TAG_FOCUS_LOCATION]
        .into_iter()
        .find_map(|tag| shorts::<4>(&reader, ifd.get(tag)?))
        .map(|[w, h, x, y]| SonyFocus {
            frame: [w, h],
            point: [x, y],
            box_size: ifd
                .get(TAG_FOCUS_FRAME_SIZE)
                .and_then(|e| shorts::<3>(&reader, e))
                .and_then(|[w, h, valid]| (valid != 0 && w > 0 && h > 0).then_some([w, h])),
        });
    if let Some(e) = ifd.get(TAG_AF_AREA_MODE_SETTING)
        && e.ty == field_type::BYTE
        && e.count == 1
    {
        note.af_area_setting = e.inline_uint().and_then(|v| u8::try_from(v).ok());
    }
    Some(note)
}

/// Exactly `N` 16-bit values: a SHORT entry of `N` values, or an UNDEFINED one of `2N` bytes read
/// in the file's byte order (how Sony stores tag 0x2037). `None` for anything else.
fn shorts<const N: usize>(tiff: &Tiff, entry: &Entry) -> Option<[u16; N]> {
    let mut out = [0u16; N];
    match entry.ty {
        field_type::SHORT if entry.count as usize == N => {
            let values = tiff.uints(entry, N).ok()?;
            if values.len() != N {
                return None;
            }
            for (slot, v) in out.iter_mut().zip(&values) {
                *slot = u16::try_from(*v).ok()?;
            }
        }
        field_type::UNDEFINED if entry.count as usize == 2 * N => {
            let bytes = tiff.entry_bytes(entry).ok()?;
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = tiff.order().u16(&bytes, 2 * i)?;
            }
        }
        _ => return None,
    }
    Some(out)
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
    let min_count = if entry.tag == TAG_9405 { 0x64 } else { 0x76 };
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
}
