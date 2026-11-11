//! Damaged files for the damage matrix (`docs/26` P1-6, task H-03) and the sweep of file messages
//! (`docs/26` P1-2, task H-01): one source of valid synthetic files of every decodable variant,
//! and a closed set of ways to damage their bytes.
//!
//! Everything is deterministic: [`Damage::apply`] depends on the bytes and on the numbers inside
//! the damage only (the "random" bytes come from a [`SplitMix64`] seeded from those numbers).
//!
//! * [`valid_files`] gives one small valid file per variant (uncompressed 14 and 12 bit, packed 12
//!   bit, ARW 1, cRAW, Lossless L with and without restart intervals, Lossless M or S in three
//!   chroma layouts).
//! * [`Damage`] is one kind of damage; [`Damage::apply`] returns the damaged bytes.
//! * [`foreign`] gives files that were never ARW (empty, tiny, text, JPEG, zeros).
//! * [`write_matrix`] writes the damaged-folder matrix to a folder.
//! * [`sweep_damages`] and [`sweep`] are the 1,576-file sweep of the message catalogue test
//!   (`crates/ym-raw/tests/messages.rs`); [`matrix_damages`] is the smaller set the damaged-folder
//!   tests use (`crates/ym-core/tests/damaged_files.rs`, the CLI and the app).

use super::{
    Arw1, ArwBuilder, CrawEncoder, LosslessL, LosslessMs, Packed12, SplitMix64, gradient_mosaic,
    gradient_rgb, tiny_jpeg,
};
use crate::layout::ChromaSubsampling;

/// One way to damage a file. See [`Damage::apply`].
#[derive(Clone, Debug, PartialEq)]
pub enum Damage {
    /// A file of zero bytes (the card reader's "file that never finished copying").
    Empty,
    /// Keeps the first `fraction` of the bytes (0.0 to 1.0).
    Truncate(f64),
    /// Cuts the last `bytes` bytes off.
    Short(usize),
    /// Overwrites the first `bytes` bytes with random bytes.
    Header(usize),
    /// Overwrites the first `bytes` bytes with zeros.
    HeaderZeros(usize),
    /// Overwrites `len` bytes from `offset_frac` of the way into the file with random bytes.
    Garbage(f64, usize),
    /// Overwrites `len` bytes from `offset_frac` of the way into the file with zeros.
    Zeros(f64, usize),
    /// Flips one to four bits. An even seed picks them in the first 2,048 bytes (the headers), an
    /// odd seed anywhere in the file.
    BitFlip(u64),
    /// Writes `value` into the first value slot of every directory entry with `tag` (SHORT or
    /// LONG), the way a damaged size, offset or count would read.
    Tag(u16, u32),
    /// Replaces the file with a text note (what the file is not).
    Text,
    /// Replaces the file with a small valid JPEG (what a card reader's `.ARW` rename of a JPEG is).
    Jpeg,
}

impl Damage {
    /// The damaged copy of `valid` (the input is not changed). [`Damage::Empty`],
    /// [`Damage::Text`] and [`Damage::Jpeg`] ignore the input.
    pub fn apply(&self, valid: &[u8]) -> Vec<u8> {
        let len = valid.len();
        match self {
            Self::Empty => Vec::new(),
            Self::Truncate(fraction) => valid[..at_fraction(len, *fraction)].to_vec(),
            Self::Short(bytes) => valid[..len.saturating_sub(*bytes)].to_vec(),
            Self::Header(bytes) => {
                let mut out = valid.to_vec();
                let mut rng = SplitMix64(0x4845_4144 ^ *bytes as u64);
                for b in &mut out[..(*bytes).min(len)] {
                    *b = rng.next_u64() as u8;
                }
                out
            }
            Self::HeaderZeros(bytes) => {
                let mut out = valid.to_vec();
                out[..(*bytes).min(len)].fill(0);
                out
            }
            Self::Garbage(fraction, n) => {
                let mut out = valid.to_vec();
                let from = at_fraction(len, *fraction);
                let to = from.saturating_add(*n).min(len);
                let mut rng = SplitMix64(fraction.to_bits() ^ (*n as u64).rotate_left(32));
                for b in &mut out[from..to] {
                    *b = rng.next_u64() as u8;
                }
                out
            }
            Self::Zeros(fraction, n) => {
                let mut out = valid.to_vec();
                let from = at_fraction(len, *fraction);
                let to = from.saturating_add(*n).min(len);
                out[from..to].fill(0);
                out
            }
            Self::BitFlip(seed) => {
                let mut out = valid.to_vec();
                if len > 0 {
                    let mut rng = SplitMix64(*seed);
                    let limit = if seed % 2 == 0 { len.min(2048) } else { len };
                    for _ in 0..=rng.below(4) {
                        let at = rng.below(limit as u64) as usize;
                        out[at] ^= 1 << rng.below(8);
                    }
                }
                out
            }
            Self::Tag(tag, value) => {
                let mut out = valid.to_vec();
                patch(&mut out, *tag, *value);
                out
            }
            Self::Text => b"This is not a photo. It is a note about a photo.\n".repeat(40),
            Self::Jpeg => tiny_jpeg(32, 24, 7),
        }
    }

    /// A short name for file names and failure messages: `cut-50`, `bitflip-3`, `tag-0x0100-0`.
    pub fn label(&self) -> String {
        match self {
            Self::Empty => "empty".to_owned(),
            Self::Truncate(f) => format!("cut-{:03}", (f * 1000.0).round() as i64),
            Self::Short(n) => format!("short-{n}"),
            Self::Header(n) => format!("header-{n}"),
            Self::HeaderZeros(n) => format!("header-zeros-{n}"),
            Self::Garbage(f, n) => format!("garbage-{:03}-{n}", (f * 1000.0).round() as i64),
            Self::Zeros(f, n) => format!("zeros-{:03}-{n}", (f * 1000.0).round() as i64),
            Self::BitFlip(seed) => format!("bitflip-{seed}"),
            Self::Tag(tag, value) => format!("tag-{tag:#06x}-{value}"),
            Self::Text => "text".to_owned(),
            Self::Jpeg => "jpeg".to_owned(),
        }
    }

    /// `true` when the damage always changes the container, never only the pixel values: the
    /// decoder has to refuse the file. A file made with one of these is never `Ready`.
    ///
    /// Truncation counts, because the raw data is the last thing in a generated file (as in a
    /// Sony file), so every cut loses pixel data and the file is shorter than its directories say.
    /// [`Damage::Short`] counts from three bytes: a lossless stream ends with the two-byte end
    /// marker, which a decoder may do without, so a file one or two bytes short can still be shown.
    pub fn always_refused(&self) -> bool {
        match self {
            Self::Empty | Self::Text | Self::Jpeg => true,
            Self::Truncate(f) => *f < 1.0,
            Self::Short(n) => *n > 2,
            Self::Header(n) | Self::HeaderZeros(n) => (1..=4).contains(n),
            _ => false,
        }
    }
}

/// `len * fraction` as a byte offset, between 0 and `len`.
fn at_fraction(len: usize, fraction: f64) -> usize {
    let at = (len as f64 * fraction.clamp(0.0, 1.0)) as usize;
    at.min(len)
}

/// The small valid files, one per decodable variant: `(name, bytes)`. The names are file-name
/// safe. Each decodes with `decode_bytes` (the sweep test checks it).
pub fn valid_files() -> Vec<(&'static str, Vec<u8>)> {
    let m = gradient_mosaic(96, 64, 512, 16000, 1);
    let m12 = gradient_mosaic(96, 64, 128, 4000, 2);
    let encoder = CrawEncoder::new();
    let mc = gradient_mosaic(96, 64, 512, encoder.max_value(), 3);
    let ms = |sub, tile: u32| {
        let rgb = gradient_rgb(70, 50, 1100, 9000, 4);
        ArwBuilder::new("ILCE-7RM5", 70, 50)
            .lossless_ms(LosslessMs::new(sub, rgb, 70, 50, [2.0, 1.0, 1.6]).tile(tile, tile))
            .build(&[])
    };
    vec![
        ("unc14", ArwBuilder::new("ILCE-7RM4", 96, 64).build(&m)),
        (
            "unc12",
            ArwBuilder::new("ILCE-7M2", 96, 64)
                .uncompressed(12)
                .build(&m12),
        ),
        (
            "packed12",
            ArwBuilder::new("DSLR-A900", 96, 64)
                .payload(Packed12)
                .build(&m12),
        ),
        (
            "arw1",
            ArwBuilder::new("DSLR-A300", 96, 64)
                .payload(Arw1)
                .black([0; 4])
                .white([3840; 3])
                .build(&m12),
        ),
        (
            "craw",
            ArwBuilder::new("ILCE-7M3", 96, 64).craw(encoder).build(&mc),
        ),
        (
            "lossless_l",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(32, 32))
                .build(&m),
        ),
        (
            "lossless_l_restarts",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(64, 64).restart_rows(2).predictor(4))
                .build(&m),
        ),
        ("ms420", ms(ChromaSubsampling::Yuv420, 32)),
        ("ms422", ms(ChromaSubsampling::Yuv422, 32)),
        ("ms444", ms(ChromaSubsampling::Yuv444, 64)),
    ]
}

/// `(entry offset, field type)` of every directory entry with `tag` in a little-endian TIFF,
/// following next-IFD links, SubIFDs and the EXIF pointer.
fn entries(file: &[u8], tag: u16) -> Vec<(usize, u16)> {
    fn u16at(f: &[u8], at: usize) -> Option<u16> {
        Some(u16::from_le_bytes(f.get(at..at + 2)?.try_into().ok()?))
    }
    fn u32at(f: &[u8], at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(f.get(at..at + 4)?.try_into().ok()?))
    }
    let mut found = Vec::new();
    let mut seen = Vec::new();
    let mut pending = vec![u32at(file, 4).unwrap_or(0) as usize];
    while let Some(at) = pending.pop() {
        if at < 8 || seen.contains(&at) {
            continue;
        }
        seen.push(at);
        let Some(count) = u16at(file, at) else {
            continue;
        };
        for i in 0..usize::from(count) {
            let e = at + 2 + 12 * i;
            let (Some(t), Some(ty), Some(n)) =
                (u16at(file, e), u16at(file, e + 2), u32at(file, e + 4))
            else {
                break;
            };
            if t == tag {
                found.push((e, ty));
            }
            if (t == 0x014A || t == 0x8769) && n == 1 {
                pending.push(u32at(file, e + 8).unwrap_or(0) as usize);
            }
        }
        pending.push(u32at(file, at + 2 + 12 * usize::from(count)).unwrap_or(0) as usize);
    }
    found
}

/// Writes `value` into the first value slot of every entry with `tag` (SHORT or LONG).
fn patch(file: &mut [u8], tag: u16, value: u32) {
    for (e, ty) in entries(file, tag) {
        match ty {
            3 => {
                if let Some(slot) = file.get_mut(e + 8..e + 10) {
                    slot.copy_from_slice(&(value as u16).to_le_bytes());
                }
            }
            4 => {
                if let Some(slot) = file.get_mut(e + 8..e + 12) {
                    slot.copy_from_slice(&value.to_le_bytes());
                }
            }
            _ => {}
        }
    }
}

/// Files that are not ARW at all, and the empty file: `(what, bytes)`.
pub fn foreign() -> Vec<(String, Vec<u8>)> {
    vec![
        ("empty".to_owned(), Damage::Empty.apply(&[])),
        ("one byte".to_owned(), vec![b'I']),
        ("seven bytes".to_owned(), b"II*\0\x08\0\0".to_vec()),
        ("text".to_owned(), Damage::Text.apply(&[])),
        ("jpeg".to_owned(), Damage::Jpeg.apply(&[])),
        ("zeros".to_owned(), vec![0; 4096]),
    ]
}

/// The 157 damages the message sweep applies to one valid file of `len` bytes: truncations at 25
/// fractions and a few bytes short of the end, header overwrites (random and zeros), garbage and
/// zeros in the body, bit flips and extreme values in 14 tags that size buffers.
pub fn sweep_damages(len: usize) -> Vec<Damage> {
    let mut out = Vec::new();
    for k in 0..=24 {
        out.push(Damage::Truncate(f64::from(k) / 24.0));
    }
    for k in [1, 7, 100, 500] {
        out.push(Damage::Short(k));
    }
    for n in [1usize, 2, 4, 8, 16] {
        out.push(Damage::Header(n));
        out.push(Damage::HeaderZeros(n));
    }
    for k in 1..=6u32 {
        let n = (len / 50).max(8);
        out.push(Damage::Garbage(f64::from(k) / 7.0, n));
        out.push(Damage::Zeros(f64::from(k) / 7.0, n));
    }
    for seed in 0..8 {
        out.push(Damage::BitFlip(seed));
    }
    for tag in [
        0x0100u16, 0x0101, 0x0102, 0x0103, 0x0115, 0x0116, 0x0117, 0x0111, 0x0142, 0x0143, 0x0144,
        0x0145, 0x0212, 0x7010,
    ] {
        for value in [0u32, 1, 3, 255, 65535, 0x7FFF_FFFF, u32::MAX] {
            out.push(Damage::Tag(tag, value));
        }
    }
    out
}

/// The 1,576-file sweep: the [`foreign`] files, then [`sweep_damages`] of every [`valid_files`]
/// variant. `(what, bytes)`, `what` is `"{variant} / {damage label}"` or the foreign name.
pub fn sweep() -> Vec<(String, Vec<u8>)> {
    let mut files = foreign();
    for (name, valid) in valid_files() {
        for damage in sweep_damages(valid.len()) {
            files.push((format!("{name} / {}", damage.label()), damage.apply(&valid)));
        }
    }
    files
}

/// The damages of the damaged-folder tests: 19 that between them reach the three sentences of a
/// damaged file, the foreign-file sentence and damage that only touches pixel values.
///
/// Whole-file replacements, truncations from nothing left to one byte short, header damage of
/// three sizes, a block of garbage and a block of zeros in the middle and near the end, bit flips
/// in the header area and in the body, and the two tags that decide how much a decoder reads
/// (image width and strip or tile byte counts) set to an extreme.
pub fn matrix_damages() -> Vec<Damage> {
    vec![
        Damage::Empty,
        Damage::Text,
        Damage::Jpeg,
        Damage::Truncate(0.1),
        Damage::Truncate(0.5),
        Damage::Truncate(0.9),
        Damage::Short(1),
        Damage::Short(100),
        Damage::Header(1),
        Damage::Header(8),
        Damage::HeaderZeros(4),
        Damage::Garbage(0.5, 64),
        Damage::Zeros(0.5, 64),
        Damage::Garbage(0.9, 64),
        Damage::BitFlip(2),
        Damage::BitFlip(3),
        Damage::BitFlip(5),
        Damage::Tag(0x0100, 0x7FFF_FFFF),
        Damage::Tag(0x0117, u32::MAX),
    ]
}

/// One file written by [`write_matrix`].
#[derive(Clone, Debug)]
pub struct MatrixFile {
    /// The file name: `good_{variant}.ARW` or `bad_{variant}_{damage label}.ARW`.
    pub file: String,
    /// The variant of [`valid_files`] it was made from.
    pub variant: &'static str,
    /// `None` for a good (valid) file.
    pub damage: Option<Damage>,
}

/// Writes the damaged-folder matrix into `dir`: for each of the ten [`valid_files`] the valid file
/// itself and one damaged copy per [`matrix_damages`] entry (200 files in all), in the order of the
/// variants. The three damaged-folder tests (core, CLI, app) use this one function.
///
/// # Errors
///
/// The first [`std::io::Error`] of writing a file.
pub fn write_matrix(dir: &std::path::Path) -> std::io::Result<Vec<MatrixFile>> {
    let mut files = Vec::new();
    for (variant, valid) in valid_files() {
        let file = format!("good_{variant}.ARW");
        std::fs::write(dir.join(&file), &valid)?;
        files.push(MatrixFile {
            file,
            variant,
            damage: None,
        });
        for damage in matrix_damages() {
            let file = format!("bad_{variant}_{}.ARW", damage.label());
            std::fs::write(dir.join(&file), damage.apply(&valid))?;
            files.push(MatrixFile {
                file,
                variant,
                damage: Some(damage),
            });
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sweep_has_the_size_the_catalogue_test_counts() {
        assert_eq!(sweep_damages(5000).len(), 157);
        assert_eq!(sweep().len(), 6 + 10 * 157);
    }

    #[test]
    fn damage_is_deterministic_and_leaves_the_input_alone() {
        let (_, valid) = valid_files().swap_remove(0);
        let copy = valid.clone();
        for d in sweep_damages(valid.len())
            .into_iter()
            .chain(matrix_damages())
        {
            assert_eq!(d.apply(&valid), d.apply(&valid), "{}", d.label());
        }
        assert_eq!(valid, copy);
    }

    #[test]
    fn every_damage_changes_the_file_or_says_it_is_a_replacement() {
        let (_, valid) = valid_files().swap_remove(0);
        for d in matrix_damages() {
            assert_ne!(d.apply(&valid), valid, "{}", d.label());
        }
    }

    #[test]
    fn labels_are_unique_inside_the_matrix_and_the_sweep() {
        for set in [matrix_damages(), sweep_damages(5000)] {
            let mut labels: Vec<String> = set.iter().map(Damage::label).collect();
            labels.sort();
            labels.dedup();
            assert_eq!(labels.len(), set.len());
        }
    }
}
