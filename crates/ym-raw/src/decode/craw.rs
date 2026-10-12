//! cRAW (lossy) block decoder, the 0x7010 camera tone curve expansion and the `qstep` quantization map (`docs/04-raw-decoding.md` section cRAW).
//!
//! # Format
//!
//! Every row of the image is stored as `width` bytes. Each 32 pixels of a row are 32 bytes, two
//! interleaved blocks of 16 pixels (block 0 holds the even columns, block 1 the odd columns, so
//! every block is one colour of the Bayer row). A block is 128 bits, read least significant bit
//! first:
//!
//! | Bits | Field |
//! |---|---|
//! | 0..11 | `max`, 11 bits |
//! | 11..22 | `min`, 11 bits |
//! | 22..26 | `imax`, the position of the maximum in the block |
//! | 26..30 | `imin`, the position of the minimum |
//! | 30..128 | 14 deltas of 7 bits, for the other pixels in position order |
//!
//! A delta pixel is `min(2047, (delta << shift) + min)` with `shift = max(0, bitlen(max - min) - 7)`
//! (at most 4). The 11-bit codes then go through the camera curve: the tag 0x7010 holds four knots
//! (each `(v >> 2) & 0xFFF`); with `k0 = 0` and `k5 = 4095` the curve rises by `2^i` per index
//! step between `k_i` and `k_(i+1)`, and a pixel's output is `curve[code << 1]`.
//!
//! # Scale
//!
//! The curve output is not shifted: it is on the 14-bit scale of the camera's uncompressed modes
//! and can exceed 16383 slightly (the real A500 and A5000 files reach 16596). This is what rawler
//! and LibRaw 0.22 (`sony_arw2_load_raw`, black level 512) produce, so black and white levels, the
//! decoded values and `qstep` all share one scale and need no conversion for those oracles. (The
//! 12-bit `>> 2` scale belongs to dcraw and old LibRaw releases.) Decoding is exact and never
//! dithers: rawler adds a pseudo-random offset of up to a quarter of the curve step around each
//! output; we return the curve value itself.
//!
//! # Quantisation map
//!
//! Both quantisation stages are known exactly, so each pixel gets the width of the interval its
//! true value was coded into. In the 11-bit code domain a pixel is stored with a step of
//! `2^shift` codes (1 for the two exact pixels). The map reports that step through the curve:
//! `qstep` is the larger of the two curve distances from the decoded code to the neighbouring
//! codes `2^shift` below and above it (there is no neighbour below the lowest `2^shift` codes;
//! above code 2047 the curve continues with its last step). Inside one curve segment that is
//! `2^shift * 2 * slope` output DN, the "step in the 11-bit domain times the local slope" of the
//! spec. A true value that the encoder rounded to the nearest representable code is within
//! `qstep / 2` of the decoded value; an encoder that truncates puts the true value in
//! `[decoded, decoded + qstep)` instead. The denoise network uses `qstep` as the quantisation
//! noise scale (`docs/06-noise.md`, `q^2 / 12`).
//!
//! One bucket is wider than the rest: a delta of 127 is the largest the format can store, and the
//! block's `max - min` is only known to be below `128 * 2^shift`, so the pixels between
//! `min + 127 * 2^shift` and the block's maximum all decode to the top bucket. For those pixels
//! `qstep` is widened to cover the span up to the block's maximum (the maximum code is known
//! exactly), which keeps both bounds above true for any encoder that rounds to the nearest or
//! truncates.

use half::f16;
use rayon::prelude::*;

use crate::error::RawError;
use crate::layout::{Plane, check_dimensions};
use crate::metadata::{RawMetadata, RawStorage};
use crate::tiff::ByteSource;

/// Entries of the camera curve (12-bit index).
pub const CURVE_LEN: usize = 4096;

/// Pixels per interleaved block pair.
const GROUP: usize = 32;

/// Bytes per block.
const BLOCK_BYTES: usize = 16;

/// Builds the camera curve from the four knots of tag 0x7010 as stored in the file.
///
/// Each knot is `(v >> 2) & 0xFFF`; the knots must not decrease. The result has [`CURVE_LEN`]
/// entries: `curve[0] = 0` and `curve[j] = curve[j - 1] + 2^i` for `k_i < j <= k_(i+1)`.
pub fn build_curve(stored_knots: [u16; 4]) -> Result<Vec<u16>, RawError> {
    let mut knots = [0usize; 6];
    for (slot, stored) in knots[1..5].iter_mut().zip(stored_knots) {
        *slot = usize::from((stored >> 2) & 0xFFF);
    }
    knots[5] = CURVE_LEN - 1;
    if knots.windows(2).any(|w| w[1] < w[0]) {
        return Err(RawError::corrupt("the cRAW tone curve is not increasing"));
    }
    let mut curve = vec![0u16; CURVE_LEN];
    for i in 0..5 {
        for j in knots[i] + 1..=knots[i + 1] {
            curve[j] = curve[j - 1].saturating_add(1 << i);
        }
    }
    Ok(curve)
}

/// The decoded planes of a cRAW image.
#[derive(Clone, Debug)]
pub struct CrawImage {
    /// Sensor values on the 14-bit scale.
    pub data: Plane<u16>,
    /// Quantisation step per pixel in output DN.
    pub qstep: Plane<f16>,
}

/// Output of the curve for an 11-bit code. Codes above 2047 continue the slope of the last step,
/// so the interval above the top codes is never narrower than it really is.
fn curve_at(curve: &[u16], code: usize) -> f32 {
    if code <= 2047 {
        f32::from(curve[code << 1])
    } else {
        let top = f32::from(curve[2047 << 1]);
        top + (top - f32::from(curve[2046 << 1])) * (code - 2047) as f32
    }
}

/// `qstep` for every 11-bit code and every shift 0 to 4 (see the module documentation).
fn qstep_table(curve: &[u16]) -> Vec<[f16; 2048]> {
    let at = |code: usize| curve_at(curve, code);
    (0..5usize)
        .map(|shift| {
            let k = 1usize << shift;
            let mut row = [f16::ZERO; 2048];
            for (code, slot) in row.iter_mut().enumerate() {
                let up = at(code + k) - at(code);
                let step = if code >= k {
                    up.max(at(code) - at(code - k))
                } else {
                    up
                };
                *slot = f16::from_f32(step);
            }
            row
        })
        .collect()
}

/// `qstep` of a delta pixel in the top bucket (delta 127) of a block whose maximum code is
/// `max_code`: twice the distance up to that maximum plus the rounding slack of the maximum
/// itself (see the module documentation).
fn top_bucket_step(curve: &[u16], code: usize, max_code: usize) -> f32 {
    let at = |c: usize| curve_at(curve, c);
    2.0 * (at(max_code) - at(code)).max(0.0) + (at(max_code + 1) - at(max_code))
}

/// The result of decoding one block.
struct Block {
    codes: [u16; 16],
    /// Shift of the delta pixels.
    shift: u32,
    /// Positions that were stored exactly (`max` and `min`).
    exact: [bool; 16],
    /// The stored delta of every delta pixel (0 for the exact ones).
    deltas: [u8; 16],
    /// The block's maximum code.
    max: u16,
}

/// Decodes one 128-bit block. `next` is the byte right after the block (zero past the end of the
/// row): a block with `imax == imin` has fifteen delta pixels but room for fourteen deltas, and
/// LibRaw reads the fifteenth from the following byte, so that is what this does too.
fn decode_block(bytes: &[u8; BLOCK_BYTES], next: u8) -> Block {
    let word = u128::from_le_bytes(*bytes);
    let max = (word & 0x7FF) as i32;
    let min = ((word >> 11) & 0x7FF) as i32;
    let imax = ((word >> 22) & 0xF) as usize;
    let imin = ((word >> 26) & 0xF) as usize;
    let mut shift = 0u32;
    while shift < 4 && (0x80i32 << shift) <= max - min {
        shift += 1;
    }
    let mut codes = [0u16; 16];
    let mut exact = [false; 16];
    let mut deltas = [0u8; 16];
    let mut bit = 30usize;
    for i in 0..16 {
        if i == imax {
            codes[i] = max as u16;
            exact[i] = true;
        } else if i == imin {
            codes[i] = min as u16;
            exact[i] = true;
        } else {
            let delta = if bit < 128 {
                ((word >> bit) & 0x7F) as i32
            } else {
                i32::from(next & 0x7F)
            };
            codes[i] = ((delta << shift) + min).min(0x7FF) as u16;
            deltas[i] = delta as u8;
            bit += 7;
        }
    }
    Block {
        codes,
        shift,
        exact,
        deltas,
        max: max as u16,
    }
}

/// Decodes `width * height` bytes of cRAW data (rows of `width` bytes, `width` a multiple of 32)
/// through `curve`. With `with_qstep` false the returned `qstep` plane is empty.
pub fn decode_rows(
    bytes: &[u8],
    width: usize,
    height: usize,
    curve: &[u16],
    with_qstep: bool,
) -> Result<(Vec<u16>, Vec<f16>), RawError> {
    check_dimensions(width, height)?;
    if !width.is_multiple_of(GROUP) {
        return Err(RawError::corrupt(
            "a cRAW row must be a whole number of 32-pixel groups",
        ));
    }
    if curve.len() != CURVE_LEN {
        return Err(RawError::corrupt("the cRAW curve has the wrong length"));
    }
    if bytes.len() < width * height {
        return Err(RawError::corrupt("the cRAW data is shorter than the image"));
    }
    let table = if with_qstep {
        qstep_table(curve)
    } else {
        Vec::new()
    };
    let mut data = vec![0u16; width * height];
    let mut qstep = if with_qstep {
        vec![f16::ZERO; width * height]
    } else {
        Vec::new()
    };

    let decode_row = |row: usize, out: &mut [u16], mut q: Option<&mut [f16]>| {
        let line = &bytes[row * width..(row + 1) * width];
        for (g, group) in line.chunks_exact(GROUP).enumerate() {
            let base = g * GROUP;
            for half in 0..2usize {
                let start = half * BLOCK_BYTES;
                let mut block_bytes = [0u8; BLOCK_BYTES];
                block_bytes.copy_from_slice(&group[start..start + BLOCK_BYTES]);
                let after = row * width + g * GROUP + start + BLOCK_BYTES;
                let next = if after < (row + 1) * width {
                    bytes[after]
                } else {
                    0
                };
                let block = decode_block(&block_bytes, next);
                for (i, &code) in block.codes.iter().enumerate() {
                    let x = base + half + 2 * i;
                    out[x] = curve[usize::from(code) << 1];
                    if let Some(q) = q.as_deref_mut() {
                        let shift = if block.exact[i] {
                            0
                        } else {
                            block.shift as usize
                        };
                        let mut step = table[shift][usize::from(code)];
                        if !block.exact[i] && block.deltas[i] == 127 {
                            let wide =
                                top_bucket_step(curve, usize::from(code), usize::from(block.max));
                            step = step.max(f16::from_f32(wide));
                        }
                        q[x] = step;
                    }
                }
            }
        }
    };

    if with_qstep {
        data.par_chunks_mut(width)
            .zip(qstep.par_chunks_mut(width))
            .enumerate()
            .for_each(|(row, (out, q))| decode_row(row, out, Some(q)));
    } else {
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out)| decode_row(row, out, None));
    }
    Ok((data, qstep))
}

/// Decodes the cRAW image described by `meta` (one strip of `width * height` bytes, or several
/// strips of whole rows).
pub fn decode(src: &dyn ByteSource, meta: &RawMetadata) -> Result<CrawImage, RawError> {
    let (width, height) = (meta.width as usize, meta.height as usize);
    check_dimensions(width, height)?;
    let knots = meta
        .tone_curve
        .ok_or_else(|| RawError::corrupt("the cRAW file has no tone curve (tag 0x7010)"))?;
    let curve = build_curve(knots)?;
    let RawStorage::Strips {
        offsets,
        byte_counts,
        rows_per_strip,
    } = &meta.storage
    else {
        return Err(RawError::corrupt("a cRAW image must be stored in strips"));
    };
    let rows_per_strip = (*rows_per_strip as usize).max(1);

    let mut bytes = vec![0u8; width * height];
    let mut row = 0usize;
    for (&offset, &count) in offsets.iter().zip(byte_counts) {
        if row >= height {
            break;
        }
        let rows = rows_per_strip.min(height - row);
        let need = rows * width;
        if (count as usize) < need {
            return Err(RawError::corrupt(
                "a cRAW strip holds fewer bytes than its rows need",
            ));
        }
        if offset
            .checked_add(need as u64)
            .is_none_or(|end| end > src.len())
        {
            return Err(RawError::corrupt(
                "the raw data runs past the end of the file (the file is cut short)",
            ));
        }
        src.read_at(offset, &mut bytes[row * width..(row + rows) * width])?;
        row += rows;
    }
    if row < height {
        return Err(RawError::corrupt("the strips do not cover every row"));
    }

    let (data, qstep) = decode_rows(&bytes, width, height, &curve, true)?;
    Ok(CrawImage {
        data: Plane::from_vec(width, height, data)
            .ok_or_else(|| RawError::corrupt("internal size mismatch"))?,
        qstep: Plane::from_vec(width, height, qstep)
            .ok_or_else(|| RawError::corrupt("internal size mismatch"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curve_rises_by_powers_of_two_between_the_knots() {
        // Knots stored as the camera writes them: value << 2.
        let curve = build_curve([100 << 2, 200 << 2, 300 << 2, 400 << 2]).unwrap();
        assert_eq!(curve.len(), CURVE_LEN);
        assert_eq!(curve[0], 0);
        assert_eq!(curve[100], 100); // slope 1 up to k1
        assert_eq!(curve[200], 100 + 2 * 100); // slope 2
        assert_eq!(curve[300], 300 + 4 * 100); // slope 4
        assert_eq!(curve[400], 700 + 8 * 100); // slope 8
        assert_eq!(curve[4095], 1500 + 16 * (4095 - 400)); // slope 16
    }

    #[test]
    fn knots_that_decrease_are_refused() {
        assert!(build_curve([200 << 2, 100 << 2, 300 << 2, 400 << 2]).is_err());
        // The top two bits and the low two bits of a stored knot are not part of the value.
        let a = build_curve([(5 << 2) | 3, 10 << 2, 20 << 2, 30 << 2]).unwrap();
        let b = build_curve([5 << 2, 10 << 2, 20 << 2, 30 << 2]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_flat_block_decodes_to_its_value() {
        // max = min = 1000, imax = 0, imin = 1, all deltas 0.
        let word: u128 = 1000 | (1000 << 11) | (1 << 26); // imax = 0, imin = 1
        let bytes = word.to_le_bytes();
        let block = decode_block(&bytes, 0);
        assert!(block.codes.iter().all(|c| *c == 1000));
        assert_eq!(block.shift, 0);
        assert!(block.exact[0] && block.exact[1] && !block.exact[2]);
    }

    #[test]
    fn deltas_are_scaled_by_the_shift() {
        // max - min = 2000 -> shift 4; delta 5 -> min + 80.
        let mut word: u128 = 2047 | (47 << 11) | (3 << 22) | (4 << 26);
        for k in 0..14 {
            word |= 5u128 << (30 + 7 * k);
        }
        let block = decode_block(&word.to_le_bytes(), 0);
        assert_eq!(block.shift, 4);
        assert_eq!(block.codes[3], 2047);
        assert_eq!(block.codes[4], 47);
        assert_eq!(block.codes[0], 47 + 80);
    }

    #[test]
    fn rows_with_a_partial_group_are_refused() {
        let curve = build_curve([0, 0, 0, 0]).unwrap();
        assert!(decode_rows(&[0u8; 48], 48, 1, &curve, false).is_err());
        assert!(decode_rows(&[0u8; 8], 32, 1, &curve, false).is_err());
    }
}
