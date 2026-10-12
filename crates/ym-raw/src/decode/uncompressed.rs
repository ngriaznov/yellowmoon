//! Uncompressed decoders: packed 12-bit (ARW 2) and 16-bit-container 14-bit (or 12-bit) samples (`docs/04-raw-decoding.md` section Decoders by TIFF compression).
//!
//! Two layouts exist:
//!
//! * **16-bit words** (TIFF compression 1, or 32767 with two bytes per sample): every sample is a
//!   little-endian `u16` holding a 14-bit (A7R II and later) or 12-bit value. The values are
//!   returned as stored; nothing is shifted or masked, so black and white levels stay on the scale
//!   the file's SR2 data uses.
//! * **Packed 12-bit** (TIFF compression 32767 with 12 bits and exactly 1.5 bytes per pixel, the
//!   A700 / A850 / A900 era "RAW" mode): three bytes hold two samples, the first in the low 12 bits
//!   of the little-endian 24-bit group (`b0 | (b1 & 0x0F) << 8`) and the second in the high 12
//!   bits (`b1 >> 4 | b2 << 4`). Rows are stored back to back without padding. LibRaw's
//!   `packed_load_raw` with `load_flags = 80` does exactly this (its `col ^ 1` swap undoes the
//!   high-first order of its bit reader).
//!
//! Files may store the data in several strips of whole rows; the strips are concatenated in order.

use crate::error::RawError;
use crate::layout::{Plane, check_dimensions};
use crate::metadata::{RawMetadata, RawStorage};
use crate::tiff::ByteSource;

/// How the samples are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// One little-endian `u16` per sample.
    Words16,
    /// Two samples in three bytes.
    Packed12,
}

impl Packing {
    /// Bytes needed for `samples` samples in a row.
    fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::Words16 => width * 2,
            Self::Packed12 => width * 3 / 2,
        }
    }
}

/// Decodes the uncompressed raw image described by `meta`.
pub fn decode(
    src: &dyn ByteSource,
    meta: &RawMetadata,
    packing: Packing,
) -> Result<Plane<u16>, RawError> {
    let (width, height) = (meta.width as usize, meta.height as usize);
    check_dimensions(width, height)?;
    if packing == Packing::Packed12 && width % 2 != 0 {
        return Err(RawError::corrupt(
            "a packed 12-bit image needs an even width",
        ));
    }
    let RawStorage::Strips {
        offsets,
        byte_counts,
        rows_per_strip,
    } = &meta.storage
    else {
        return Err(RawError::corrupt(
            "an uncompressed image must be stored in strips, not tiles",
        ));
    };
    let row_bytes = packing.row_bytes(width);
    let rows_per_strip = (*rows_per_strip as usize).max(1);
    let mut plane = Plane::<u16>::try_new(width, height)?;

    let mut row = 0usize;
    for (&offset, &count) in offsets.iter().zip(byte_counts) {
        if row >= height {
            break;
        }
        let rows = rows_per_strip.min(height - row);
        let need = (rows * row_bytes) as u64;
        if count < need {
            return Err(RawError::corrupt(
                "an uncompressed strip holds fewer bytes than its rows need",
            ));
        }
        let end = offset
            .checked_add(need)
            .ok_or_else(|| RawError::corrupt("a strip offset overflows"))?;
        if end > src.len() {
            return Err(RawError::corrupt(
                "the raw data runs past the end of the file (the file is cut short)",
            ));
        }
        let dest = &mut plane.data[row * width..(row + rows) * width];
        match packing {
            Packing::Words16 => read_words16(src, offset, dest)?,
            Packing::Packed12 => read_packed12(src, offset, dest, width)?,
        }
        row += rows;
    }
    if row < height {
        return Err(RawError::corrupt("the strips do not cover every row"));
    }
    Ok(plane)
}

/// Reads `dest.len()` little-endian words starting at `offset`.
fn read_words16(src: &dyn ByteSource, offset: u64, dest: &mut [u16]) -> Result<(), RawError> {
    src.read_at(offset, bytemuck::cast_slice_mut(dest))?;
    if cfg!(target_endian = "big") {
        for value in dest.iter_mut() {
            *value = u16::from_le(*value);
        }
    }
    Ok(())
}

/// Reads packed 12-bit samples for whole rows of `width` samples into `dest`.
fn read_packed12(
    src: &dyn ByteSource,
    offset: u64,
    dest: &mut [u16],
    width: usize,
) -> Result<(), RawError> {
    // One block of whole rows at a time keeps the temporary buffer small.
    let rows_per_block = (1 << 20) / (width * 3 / 2).max(1);
    let rows_per_block = rows_per_block.max(1);
    let row_bytes = width * 3 / 2;
    let mut buf = vec![0u8; rows_per_block * row_bytes];
    let rows = dest.len() / width;
    let mut row = 0usize;
    while row < rows {
        let n = rows_per_block.min(rows - row);
        src.read_at(offset + (row * row_bytes) as u64, &mut buf[..n * row_bytes])?;
        unpack12(
            &buf[..n * row_bytes],
            &mut dest[row * width..(row + n) * width],
        );
        row += n;
    }
    Ok(())
}

/// Unpacks `packed.len() / 3` groups of three bytes into pairs of 12-bit samples.
pub fn unpack12(packed: &[u8], out: &mut [u16]) {
    for (group, pair) in packed.chunks_exact(3).zip(out.chunks_exact_mut(2)) {
        let (b0, b1, b2) = (
            u16::from(group[0]),
            u16::from(group[1]),
            u16::from(group[2]),
        );
        pair[0] = b0 | ((b1 & 0x0F) << 8);
        pair[1] = (b1 >> 4) | (b2 << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack12_matches_the_documented_bit_order() {
        // Samples 0xABC and 0x123: b0 = 0xBC, b1 = (0x3 << 4) | 0xA, b2 = 0x12.
        let mut out = [0u16; 2];
        unpack12(&[0xBC, 0x3A, 0x12], &mut out);
        assert_eq!(out, [0xABC, 0x123]);
    }
}
