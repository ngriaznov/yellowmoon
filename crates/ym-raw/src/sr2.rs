//! SR2 private data: pad-cipher decryption and the black level (0x7310, 0x7300), white level (0x787F) and white balance (0x7313, 0x7303) tags (`docs/04-raw-decoding.md` section SR2 private data).
//!
//! An ARW carries its black level, white level and white-balance gains in an encrypted directory,
//! the SR2 sub-IFD. The path to it is:
//!
//! 1. IFD0 tag `DNGPrivateData` (0xC634) holds a 4-byte little-endian offset to the **SR2Private**
//!    directory (a plain IFD).
//! 2. SR2Private lists `SR2SubIFDOffset` (0x7200), `SR2SubIFDLength` (0x7201) and `SR2SubIFDKey`
//!    (0x7221, four bytes, little endian).
//! 3. The block at that offset is encrypted with a pad cipher (see [`apply_pad`]). The decrypted
//!    bytes are a bare IFD that starts at the first byte of the block. **Value offsets inside it are
//!    absolute positions in the file**, so the position of a value inside the decrypted block is
//!    `stored offset - SR2SubIFDOffset` (LibRaw's `parseSonySR2` and rawler's
//!    `IFD::new(.., -sony_offset, ..)` do the same; ExifTool sets the directory's `DataPos` to the
//!    block offset). [`Tiff::headerless`] with a bias of `SR2SubIFDOffset` implements exactly that.
//!
//! Failures (missing tags, bad lengths, garbage after decryption) read as "unknown": every function
//! returns `None` rather than an error, and the caller falls back to other sources.

use crate::layout::CfaPattern;
use crate::tiff::{ByteOrder, Entry, Ifd, MemSource, Tiff, field_type, tag};

/// `SR2SubIFDOffset`: file offset of the encrypted block.
pub const TAG_SUB_IFD_OFFSET: u16 = 0x7200;
/// `SR2SubIFDLength`: length of the encrypted block in bytes.
pub const TAG_SUB_IFD_LENGTH: u16 = 0x7201;
/// `SR2SubIFDKey`: the cipher key.
pub const TAG_SUB_IFD_KEY: u16 = 0x7221;
/// Black level of older bodies: four values.
pub const TAG_BLACK_LEVEL_LEGACY: u16 = 0x7300;
/// White balance of older bodies: four values in G, R, B, G order.
pub const TAG_WB_GRBG: u16 = 0x7303;
/// Black level: four values in R, G, G, B order.
pub const TAG_BLACK_LEVEL: u16 = 0x7310;
/// White balance: four values in R, G, G, B order.
pub const TAG_WB_RGGB: u16 = 0x7313;
/// White level: three values (R, G, B), occasionally one or four.
pub const TAG_WHITE_LEVEL: u16 = 0x787F;

/// The longest SR2 block read (1 MiB). Real blocks are a few kilobytes.
pub const MAX_BLOCK_BYTES: u32 = 1 << 20;

/// Applies Sony's pad cipher to `data` in place. Encryption and decryption are the same
/// operation (an XOR with a pad derived from `key`), so applying it twice restores the input.
///
/// The cipher works on little-endian 32-bit words; `data.len() % 4` trailing bytes are left
/// untouched.
///
/// 1. Four pad words come from `key = key * 48828125 + 1` (wrapping `u32`), word 3 is then
///    replaced by `word3 << 1 | (word0 ^ word2) >> 31`.
/// 2. The pad is extended to 127 words with
///    `pad[p] = (pad[p-4] ^ pad[p-2]) << 1 | (pad[p-3] ^ pad[p-1]) >> 31`.
/// 3. Every pad word is byte-swapped.
/// 4. For each data word, starting with `p = 127`:
///    `pad[p % 128] = pad[(p + 1) % 128] ^ pad[(p + 65) % 128]`, and the data word is XORed with
///    the new pad word.
pub fn apply_pad(data: &mut [u8], mut key: u32) {
    let mut pad = [0u32; 128];
    for word in pad.iter_mut().take(4) {
        key = key.wrapping_mul(48_828_125).wrapping_add(1);
        *word = key;
    }
    pad[3] = (pad[3] << 1) | ((pad[0] ^ pad[2]) >> 31);
    for p in 4..127 {
        pad[p] = ((pad[p - 4] ^ pad[p - 2]) << 1) | ((pad[p - 3] ^ pad[p - 1]) >> 31);
    }
    for word in pad.iter_mut().take(127) {
        *word = word.swap_bytes();
    }
    for (p, chunk) in (127usize..).zip(data.chunks_exact_mut(4)) {
        let next = pad[(p + 1) & 127] ^ pad[(p + 65) & 127];
        pad[p & 127] = next;
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ next;
        chunk.copy_from_slice(&word.to_le_bytes());
    }
}

/// The location and key of the encrypted SR2 block, from the SR2Private directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sr2Block {
    /// File offset of the encrypted block.
    pub offset: u32,
    /// Length of the block in bytes.
    pub length: u32,
    /// Cipher key.
    pub key: u32,
}

/// What the decrypted SR2 sub-IFD tells us. Colour-ordered values are in the order red, green 1,
/// green 2, blue (the order Sony uses for 0x7310 and 0x7313).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sr2Data {
    /// Black level per colour (tag 0x7310, or 0x7300 on older bodies).
    pub black: Option<[u16; 4]>,
    /// The tag the black level came from.
    pub black_tag: Option<u16>,
    /// White level per colour (tag 0x787F). Three stored values (R, G, B) are expanded to four.
    pub white: Option<[u16; 4]>,
    /// As-shot white balance gains (tag 0x7313, or 0x7303 reordered from GRBG to RGGB).
    pub wb: Option<[u16; 4]>,
    /// The tag the white balance came from.
    pub wb_tag: Option<u16>,
}

/// Reads the SR2 data of a file: follows `DNGPrivateData` in `ifd0`, decrypts the block and
/// parses it. Returns `None` when any step is missing or malformed.
pub fn read(tiff: &Tiff, ifd0: &Ifd) -> Option<Sr2Data> {
    let block = read_block_location(tiff, ifd0)?;
    let encrypted = tiff
        .read_vec(u64::from(block.offset), u64::from(block.length))
        .ok()?;
    parse_encrypted_block(encrypted, block)
}

/// Finds the SR2 block offset, length and key through `DNGPrivateData` and SR2Private.
pub fn read_block_location(tiff: &Tiff, ifd0: &Ifd) -> Option<Sr2Block> {
    let private = ifd0.get(tag::DNG_PRIVATE_DATA)?;
    // The value is four bytes, a little-endian offset (even if the entry's declared type or the
    // file's byte order suggest otherwise).
    let bytes = tiff.entry_prefix(private, 4).ok()?;
    let private_offset = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    if private_offset < 8 {
        return None;
    }
    // SR2Private is a plain little-endian IFD with absolute offsets in the same file.
    let reader = tiff.reordered(ByteOrder::Little);
    let sr2_private = reader.read_ifd(u64::from(private_offset)).ok()?;
    let offset = reader
        .uints(sr2_private.get(TAG_SUB_IFD_OFFSET)?, 1)
        .ok()?
        .first()
        .copied()?;
    let length = reader
        .uints(sr2_private.get(TAG_SUB_IFD_LENGTH)?, 1)
        .ok()?
        .first()
        .copied()?;
    let key = key_from_entry(&reader, sr2_private.get(TAG_SUB_IFD_KEY)?)?;
    let offset = u32::try_from(offset).ok()?;
    let length = u32::try_from(length).ok()?;
    if !(14..=MAX_BLOCK_BYTES).contains(&length) || offset < 8 {
        return None;
    }
    Some(Sr2Block {
        offset,
        length,
        key,
    })
}

/// The key is four bytes in file (little-endian) order, whatever the field type says.
fn key_from_entry(tiff: &Tiff, entry: &Entry) -> Option<u32> {
    let bytes = tiff.entry_prefix(entry, 4).ok()?;
    if bytes.len() < 4
        || !matches!(
            entry.ty,
            field_type::UNDEFINED | field_type::BYTE | field_type::LONG
        )
    {
        return None;
    }
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

/// Decrypts `encrypted` (the block described by `block`) and parses the directory inside.
pub fn parse_encrypted_block(mut encrypted: Vec<u8>, block: Sr2Block) -> Option<Sr2Data> {
    apply_pad(&mut encrypted, block.key);
    parse_decrypted(&encrypted, block.offset)
}

/// Parses an already decrypted SR2 sub-IFD. `file_offset` is where the block starts in the file;
/// value offsets in the directory are absolute, so it is the bias for reading them.
pub fn parse_decrypted(decrypted: &[u8], file_offset: u32) -> Option<Sr2Data> {
    let source = MemSource(decrypted);
    let tiff = Tiff::headerless(&source, ByteOrder::Little, u64::from(file_offset));
    let ifd = tiff.read_ifd(0).ok()?;
    let mut data = Sr2Data::default();

    for (tag, is_primary) in [(TAG_BLACK_LEVEL, true), (TAG_BLACK_LEVEL_LEGACY, false)] {
        if data.black.is_some() && !is_primary {
            break;
        }
        if let Some(values) = ifd.get(tag).and_then(|e| four_levels(&tiff, e)) {
            data.black = Some(values);
            data.black_tag = Some(tag);
        }
    }

    if let Some(entry) = ifd.get(TAG_WHITE_LEVEL) {
        data.white = white_levels(&tiff, entry);
    }

    if let Some(values) = ifd.get(TAG_WB_RGGB).and_then(|e| four_levels(&tiff, e)) {
        data.wb = Some(values);
        data.wb_tag = Some(TAG_WB_RGGB);
    } else if let Some(grbg) = ifd.get(TAG_WB_GRBG).and_then(|e| four_levels(&tiff, e)) {
        // Stored G, R, B, G; reorder to R, G, G, B.
        data.wb = Some([grbg[1], grbg[0], grbg[3], grbg[2]]);
        data.wb_tag = Some(TAG_WB_GRBG);
    }

    (data.black.is_some() || data.white.is_some() || data.wb.is_some()).then_some(data)
}

/// Four 16-bit values from a SHORT/SSHORT tag with at least four values.
fn four_levels(tiff: &Tiff, entry: &Entry) -> Option<[u16; 4]> {
    if !matches!(
        entry.ty,
        field_type::SHORT | field_type::SSHORT | field_type::LONG
    ) || entry.count < 4
    {
        return None;
    }
    let values = tiff.ints(entry, 4).ok()?;
    let mut out = [0u16; 4];
    for (slot, value) in out.iter_mut().zip(values.iter()) {
        *slot = u16::try_from(*value).ok()?;
    }
    (values.len() >= 4).then_some(out)
}

/// The white level tag holds one, three (R, G, B) or four values; the result is in the order
/// R, G1, G2, B. Also reads the DNG `WhiteLevel` (0xC61D) of the raw IFD.
pub(crate) fn white_levels(tiff: &Tiff, entry: &Entry) -> Option<[u16; 4]> {
    if !matches!(
        entry.ty,
        field_type::SHORT | field_type::SSHORT | field_type::LONG
    ) {
        return None;
    }
    let values: Vec<u16> = tiff
        .ints(entry, 4)
        .ok()?
        .into_iter()
        .map(u16::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    match values.as_slice() {
        [w] => Some([*w; 4]),
        [r, g, b] => Some([*r, *g, *g, *b]),
        [r, g1, g2, b] => Some([*r, *g1, *g2, *b]),
        _ => None,
    }
}

/// Black and white levels after the bit-depth check, per CFA position of `cfa`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Levels {
    /// Black level per CFA position.
    pub black: [f32; 4],
    /// White level per CFA position.
    pub white: [f32; 4],
    /// How many bits the levels were shifted down to match the data (0 when they were already on
    /// the data's scale).
    pub shift_bits: u8,
}

/// Why SR2 levels were rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelsError {
    /// After scaling, a black level is not below its white level.
    BlackNotBelowWhite,
}

/// The number of bits the levels must be shifted down to match samples of `bits_per_sample`
/// bits: the smallest shift that brings the largest white level to at most `2^bits - 1` (0 when
/// the levels are already on the data's scale, 2 for 14-bit levels on 12-bit data).
pub fn level_shift(white_rggb: [u16; 4], bits_per_sample: u16) -> u8 {
    let max_value = if (1..=16).contains(&bits_per_sample) {
        (1u32 << bits_per_sample) - 1
    } else {
        u32::from(u16::MAX)
    };
    let largest_white = white_rggb.iter().copied().max().map_or(0, u32::from);
    let mut shift = 0u8;
    while shift < 8 && (largest_white >> shift) > max_value {
        shift += 1;
    }
    shift
}

/// Matches SR2 levels to the bit depth of the stored samples and orders them per CFA position.
///
/// Sony writes the levels on a 14-bit scale (black 512, white 16383 and so on). Files whose
/// samples are 12 bits wide (12-bit uncompressed, some cRAW and lossless modes) hold values that
/// are four times smaller. When a white level is larger than the largest value the data can
/// hold (`2^bits - 1`), all levels are shifted right by the smallest number of bits that makes
/// every white level fit, the way the 14-bit scale maps onto a 12-bit one. The result must satisfy
/// `black < white` at every CFA position, otherwise the levels are rejected.
pub fn fit_levels(
    black_rggb: [u16; 4],
    white_rggb: [u16; 4],
    bits_per_sample: u16,
    cfa: CfaPattern,
) -> Result<Levels, LevelsError> {
    let shift = level_shift(white_rggb, bits_per_sample);
    let scale = |v: u16| f32::from(v >> shift);
    let black = cfa.rggb_to_positions(black_rggb.map(scale));
    let white = cfa.rggb_to_positions(white_rggb.map(scale));
    if black.iter().zip(white.iter()).any(|(b, w)| b >= w) {
        return Err(LevelsError::BlackNotBelowWhite);
    }
    Ok(Levels {
        black,
        white,
        shift_bits: shift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First pad words computed with an independent Python transcription of LibRaw's
    /// `sony_decrypt` (the XOR of zero data is the pad itself).
    #[test]
    fn pad_matches_independent_reference_vectors() {
        let cases: [(u32, [u32; 6], u32, u32); 3] = [
            (
                0x0000_0000,
                [
                    0xff18_9372,
                    0xae83_5a5e,
                    0xd0fc_e299,
                    0x1fad_6d8c,
                    0x5fc8_e3d6,
                    0x625d_6ea4,
                ],
                0xaf83_5a5e,
                0x0ef2_0b9b,
            ),
            (
                0x1234_5678,
                [
                    0xbbe9_c514,
                    0x7ddc_32a2,
                    0x4e89_3991,
                    0xafc8_f2a0,
                    0xeac1_f80b,
                    0xa529_8005,
                ],
                0xe4e9_1395,
                0x582a_e3f3,
            ),
            (
                0xDEAD_BEEF,
                [
                    0x2ef1_4285,
                    0x6686_5edd,
                    0xe795_8c33,
                    0x1130_6ea4,
                    0x92c9_9c6d,
                    0xee6c_61f2,
                ],
                0x3260_abbc,
                0x62fb_d87d,
            ),
        ];
        for (key, first, word127, word128) in cases {
            let mut zeros = vec![0u8; 300 * 4];
            apply_pad(&mut zeros, key);
            let word = |i: usize| u32::from_le_bytes(zeros[i * 4..i * 4 + 4].try_into().unwrap());
            for (i, expected) in first.iter().enumerate() {
                assert_eq!(word(i), *expected, "key {key:#x} word {i}");
            }
            assert_eq!(word(127), word127, "key {key:#x} word 127");
            assert_eq!(word(128), word128, "key {key:#x} word 128");
        }
    }

    #[test]
    fn apply_pad_twice_restores_the_input() {
        let original: Vec<u8> = (0..1003u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut data = original.clone();
        apply_pad(&mut data, 0xCAFE_F00D);
        assert_ne!(data[..1000], original[..1000]);
        // The three trailing bytes (1003 % 4) are not touched.
        assert_eq!(data[1000..], original[1000..]);
        apply_pad(&mut data, 0xCAFE_F00D);
        assert_eq!(data, original);
    }

    #[test]
    fn different_keys_give_different_pads() {
        let mut a = vec![0u8; 64];
        let mut b = vec![0u8; 64];
        apply_pad(&mut a, 1);
        apply_pad(&mut b, 2);
        assert_ne!(a, b);
    }

    /// Builds a decrypted SR2 sub-IFD the simple way: entries first, values after, offsets are
    /// absolute (`file_offset` + position).
    fn sr2_block(file_offset: u32, tags: &[(u16, u16, Vec<u16>)]) -> Vec<u8> {
        let n = tags.len();
        let mut values = Vec::new();
        let mut table = Vec::new();
        let values_start = 2 + n * 12 + 4;
        for (tag, ty, vals) in tags {
            table.extend_from_slice(&tag.to_le_bytes());
            table.extend_from_slice(&ty.to_le_bytes());
            table.extend_from_slice(&(vals.len() as u32).to_le_bytes());
            if vals.len() * 2 <= 4 {
                let mut inline = [0u8; 4];
                for (i, v) in vals.iter().enumerate() {
                    inline[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
                table.extend_from_slice(&inline);
            } else {
                let pos = file_offset + (values_start + values.len()) as u32;
                table.extend_from_slice(&pos.to_le_bytes());
                for v in vals {
                    values.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&table);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&values);
        out
    }

    #[test]
    fn parses_modern_black_white_and_wb() {
        let block = sr2_block(
            90_000,
            &[
                (TAG_BLACK_LEVEL, field_type::SHORT, vec![512, 513, 514, 515]),
                (TAG_WB_RGGB, field_type::SHORT, vec![2043, 1024, 1025, 1755]),
                (
                    TAG_WHITE_LEVEL,
                    field_type::SHORT,
                    vec![16300, 16301, 16302],
                ),
            ],
        );
        let data = parse_decrypted(&block, 90_000).expect("sr2");
        assert_eq!(data.black, Some([512, 513, 514, 515]));
        assert_eq!(data.black_tag, Some(TAG_BLACK_LEVEL));
        assert_eq!(data.wb, Some([2043, 1024, 1025, 1755]));
        assert_eq!(data.wb_tag, Some(TAG_WB_RGGB));
        assert_eq!(data.white, Some([16300, 16301, 16301, 16302]));
    }

    #[test]
    fn falls_back_to_the_legacy_tags_and_reorders_grbg() {
        let block = sr2_block(
            4096,
            &[
                (
                    TAG_BLACK_LEVEL_LEGACY,
                    field_type::SHORT,
                    vec![256, 256, 256, 256],
                ),
                // G, R, B, G
                (TAG_WB_GRBG, field_type::SHORT, vec![1024, 2200, 1700, 1030]),
            ],
        );
        let data = parse_decrypted(&block, 4096).expect("sr2");
        assert_eq!(data.black, Some([256; 4]));
        assert_eq!(data.black_tag, Some(TAG_BLACK_LEVEL_LEGACY));
        // R, G1, G2, B
        assert_eq!(data.wb, Some([2200, 1024, 1030, 1700]));
        assert_eq!(data.wb_tag, Some(TAG_WB_GRBG));
        assert_eq!(data.white, None);
    }

    #[test]
    fn primary_tags_win_over_legacy_ones() {
        let block = sr2_block(
            5000,
            &[
                (TAG_BLACK_LEVEL_LEGACY, field_type::SHORT, vec![1, 1, 1, 1]),
                (TAG_WB_GRBG, field_type::SHORT, vec![1, 2, 3, 4]),
                (TAG_BLACK_LEVEL, field_type::SHORT, vec![512, 512, 512, 512]),
                (TAG_WB_RGGB, field_type::SHORT, vec![2000, 1024, 1024, 1800]),
            ],
        );
        let data = parse_decrypted(&block, 5000).expect("sr2");
        assert_eq!(data.black, Some([512; 4]));
        assert_eq!(data.wb, Some([2000, 1024, 1024, 1800]));
    }

    #[test]
    fn white_level_accepts_one_three_or_four_values() {
        for (vals, expected) in [
            (vec![16383u16], [16383; 4]),
            (vec![16000, 16100, 16200], [16000, 16100, 16100, 16200]),
            (vec![1, 2, 3, 4], [1, 2, 3, 4]),
        ] {
            let block = sr2_block(1000, &[(TAG_WHITE_LEVEL, field_type::SHORT, vals)]);
            assert_eq!(
                parse_decrypted(&block, 1000).expect("sr2").white,
                Some(expected)
            );
        }
    }

    #[test]
    fn garbage_and_missing_tags_read_as_unknown() {
        assert_eq!(parse_decrypted(&[], 0), None);
        assert_eq!(parse_decrypted(&[0xFF; 64], 100), None);
        // A directory with only irrelevant tags.
        let block = sr2_block(1000, &[(0x74A0, field_type::SHORT, vec![1, 2])]);
        assert_eq!(parse_decrypted(&block, 1000), None);
        // A short tag (3 values) where four are needed is ignored.
        let block = sr2_block(1000, &[(TAG_WB_RGGB, field_type::SHORT, vec![1, 2, 3])]);
        assert_eq!(parse_decrypted(&block, 1000), None);
        // Offsets pointing before or after the block are ignored.
        let mut block = sr2_block(1000, &[(TAG_WB_RGGB, field_type::SHORT, vec![1, 2, 3, 4])]);
        block[10..14].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(parse_decrypted(&block, 1000), None);
    }

    /// The decrypted SR2 block of the real ILCE-9 file in github.com/lclevy/sony_raw: 209 entries,
    /// SR2SubIFDOffset 56484 and SR2SubIFDLength 56958, the first value (the black level at file
    /// offset 0xE672 = 58994) right behind the entry table with no room for a next-IFD pointer,
    /// 0x7313 stored as signed 16-bit values and the white level as three values (15360). The
    /// first five entries are the bytes the dump prints; the rest are filler of the same kind.
    #[test]
    fn parses_the_layout_of_the_real_ilce_9_block() {
        const OFFSET: u32 = 56_484;
        const LENGTH: usize = 56_958;
        let table_end = 2 + 209 * 12;
        assert_eq!(OFFSET as usize + table_end, 0xE672);
        let mut block = vec![0u8; LENGTH];
        block[..2].copy_from_slice(&209u16.to_le_bytes());
        let mut put = |index: usize, tag: u16, ty: u16, count: u32, value: [u8; 4]| {
            let at = 2 + index * 12;
            block[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            block[at + 2..at + 4].copy_from_slice(&ty.to_le_bytes());
            block[at + 4..at + 8].copy_from_slice(&count.to_le_bytes());
            block[at + 8..at + 12].copy_from_slice(&value);
        };
        let at = |position: usize| (OFFSET + position as u32).to_le_bytes();
        put(0, 0x7310, 3, 4, at(table_end));
        put(1, 0x7311, 3, 1, [0, 1, 0, 0]);
        put(2, 0x7312, 8, 4, at(table_end + 8));
        put(3, 0x7313, 8, 4, at(table_end + 16));
        put(4, 0x7314, 8, 3, at(table_end + 24));
        for index in 5..209 {
            if index == 106 {
                put(index, 0x787F, 3, 3, at(table_end + 36));
            } else {
                // Filler: tags that mean nothing to us, one SHORT stored in the entry.
                put(index, 0x7900 + index as u16, 3, 1, [1, 0, 0, 0]);
            }
        }
        let mut values = Vec::new();
        for v in [512u16, 512, 512, 512] {
            values.extend_from_slice(&v.to_le_bytes()); // black
        }
        for v in [2364i16, 1024, 1024, 1728] {
            values.extend_from_slice(&v.to_le_bytes()); // 0x7312
        }
        for v in [2364i16, 1024, 1024, 1728] {
            values.extend_from_slice(&v.to_le_bytes()); // 0x7313
        }
        for v in [1024i16, 1024, 1024] {
            values.extend_from_slice(&v.to_le_bytes()); // 0x7314
        }
        for v in [1024i16, 1024, 1024] {
            values.extend_from_slice(&v.to_le_bytes()); // 0x7315
        }
        for v in [15_360u16; 3] {
            values.extend_from_slice(&v.to_le_bytes()); // white level
        }
        // 0x7312 is where the dump has it, 8 bytes behind the black level; 0x7313 follows; the
        // white level sits behind 0x7315 (offset 24 + 6 + 6 = 36).
        block[table_end..table_end + values.len()].copy_from_slice(&values);

        let data = parse_decrypted(&block, OFFSET).expect("sr2");
        assert_eq!(data.black, Some([512; 4]));
        assert_eq!(data.wb, Some([2364, 1024, 1024, 1728]));
        assert_eq!(data.wb_tag, Some(TAG_WB_RGGB));
        assert_eq!(data.white, Some([15_360; 4]));

        // Through the cipher with the file's key (bytes 11 22 33 44), length not a multiple of 4.
        let key = 0x4433_2211;
        let mut encrypted = block.clone();
        apply_pad(&mut encrypted, key);
        assert_ne!(encrypted, block);
        let info = Sr2Block {
            offset: OFFSET,
            length: LENGTH as u32,
            key,
        };
        assert_eq!(parse_encrypted_block(encrypted, info), Some(data));
    }

    #[test]
    fn encrypted_round_trip_through_the_cipher() {
        let plain = sr2_block(
            70_000,
            &[
                (TAG_BLACK_LEVEL, field_type::SHORT, vec![512, 512, 512, 512]),
                (TAG_WB_RGGB, field_type::SHORT, vec![2043, 1024, 1024, 1755]),
            ],
        );
        let mut padded = plain.clone();
        while !padded.len().is_multiple_of(4) {
            padded.push(0);
        }
        let key = 0x9E37_79B9;
        let mut encrypted = padded.clone();
        apply_pad(&mut encrypted, key);
        let block = Sr2Block {
            offset: 70_000,
            length: encrypted.len() as u32,
            key,
        };
        let data = parse_encrypted_block(encrypted, block).expect("sr2");
        assert_eq!(data.wb, Some([2043, 1024, 1024, 1755]));
        // The wrong key gives garbage, not data.
        let mut wrong = padded.clone();
        apply_pad(&mut wrong, key);
        let wrong_block = Sr2Block {
            key: key ^ 1,
            ..block
        };
        assert_eq!(parse_encrypted_block(wrong, wrong_block), None);
    }

    #[test]
    fn fit_levels_keeps_levels_that_already_match_the_data() {
        let levels = fit_levels([512; 4], [16300, 16300, 16300, 16300], 14, CfaPattern::RGGB)
            .expect("levels");
        assert_eq!(levels.shift_bits, 0);
        assert_eq!(levels.black, [512.0; 4]);
        assert_eq!(levels.white, [16300.0; 4]);
    }

    #[test]
    fn fit_levels_scales_14_bit_levels_to_12_bit_data() {
        let levels = fit_levels([512; 4], [16383; 4], 12, CfaPattern::RGGB).expect("levels");
        assert_eq!(levels.shift_bits, 2);
        assert_eq!(levels.black, [128.0; 4]);
        assert_eq!(levels.white, [4095.0; 4]);
        // Levels already on a 12-bit scale are left alone.
        let levels = fit_levels([128; 4], [4095; 4], 12, CfaPattern::RGGB).expect("levels");
        assert_eq!(levels.shift_bits, 0);
    }

    #[test]
    fn fit_levels_orders_values_per_cfa_position() {
        let levels = fit_levels(
            [10, 20, 30, 40],
            [1000, 2000, 3000, 4000],
            14,
            CfaPattern::BGGR,
        )
        .expect("levels");
        assert_eq!(levels.black, [40.0, 20.0, 30.0, 10.0]);
        assert_eq!(levels.white, [4000.0, 2000.0, 3000.0, 1000.0]);
    }

    #[test]
    fn fit_levels_rejects_black_at_or_above_white() {
        assert_eq!(
            fit_levels([4000; 4], [3000; 4], 14, CfaPattern::RGGB),
            Err(LevelsError::BlackNotBelowWhite)
        );
        assert_eq!(
            fit_levels([512, 512, 512, 3000], [3000; 4], 14, CfaPattern::RGGB),
            Err(LevelsError::BlackNotBelowWhite)
        );
        // Equal is not below.
        assert_eq!(
            fit_levels([100; 4], [100; 4], 14, CfaPattern::RGGB),
            Err(LevelsError::BlackNotBelowWhite)
        );
    }

    #[test]
    fn fit_levels_handles_odd_bit_depths_without_panicking() {
        for bits in [0u16, 1, 8, 16, 17, 255] {
            let _ = fit_levels([0; 4], [u16::MAX; 4], bits, CfaPattern::RGGB);
        }
    }
}
