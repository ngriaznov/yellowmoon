//! ARW 1 decoder: the Huffman-coded 12-bit format of the A200, A230, A290, A300, A330, A350, A380 and A390 (`docs/04-raw-decoding.md` section Decoders by TIFF compression).
//!
//! TIFF compression 32767 with 12 bits and *fewer* bytes than pixels. The format is the one dcraw
//! calls `sony_arw_load_raw`:
//!
//! * The stream is read most significant bit first, with no byte stuffing, from the start of the
//!   strip.
//! * It codes the image column by column from the **last column to the first**. Within a column
//!   the even rows come first (0, 2, 4, ...), then the odd rows (1, 3, 5, ...). The file stores
//!   8 more rows than `ImageLength` says (`height + 8`; LibRaw and rawler return them, the TIFF
//!   header does not count them); they are decoded, to keep the running sum in step, and dropped.
//! * Every sample is `sum += diff`, a running sum over the whole traversal (it starts at 0 and
//!   never resets), and the sum is the 12-bit sample.
//! * `diff` is a JPEG-style difference: a Huffman-coded bit length `n` (0 to 17), then `n` bits,
//!   with the usual rule that a leading zero bit means negative.
//! * The Huffman code is fixed. Its eighteen codes, longest first (the first code is all zero
//!   bits), are for the bit lengths 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 0, 2, 1
//!   with code lengths 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3, 2, 2, assigned in
//!   that order as increasing binary numbers within each length (dcraw's table).
//!
//! A sum outside 0 to 4095 means the stream is damaged and the file is `Corrupt`.

use crate::error::RawError;
use crate::layout::{Plane, check_dimensions};
use crate::metadata::{RawMetadata, RawStorage};
use crate::tiff::ByteSource;

/// Extra rows the file stores below the image.
pub const EXTRA_ROWS: usize = 8;

/// `(code length, difference bit length)` in the order the codes are assigned.
const CODES: [(u32, u32); 18] = [
    (15, 17),
    (15, 16),
    (14, 15),
    (13, 14),
    (12, 13),
    (11, 12),
    (10, 11),
    (9, 10),
    (8, 9),
    (7, 8),
    (6, 7),
    (5, 6),
    (4, 5),
    (3, 4),
    (3, 3),
    (3, 0),
    (2, 2),
    (2, 1),
];

/// Bits looked at to find a code (the longest code).
const LOOKUP_BITS: u32 = 15;

/// The 15-bit prefix table: entry = `(code length << 8) | difference bit length`.
fn lookup_table() -> Vec<u16> {
    let mut table = vec![0u16; 1 << LOOKUP_BITS];
    let mut next = 0usize;
    for (length, symbol) in CODES {
        let span = 1usize << (LOOKUP_BITS - length);
        for slot in &mut table[next..next + span] {
            *slot = ((length as u16) << 8) | symbol as u16;
        }
        next += span;
    }
    table
}

/// MSB-first bit reader over a byte slice; bits past the end read as zero and are counted.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }

    #[inline(always)]
    fn fill(&mut self) {
        while self.nbits <= 56 {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.acc |= u64::from(byte) << (56 - self.nbits);
            self.pos += 1;
            self.nbits += 8;
        }
    }

    #[inline(always)]
    fn peek(&self, n: u32) -> u32 {
        (self.acc >> (64 - n)) as u32
    }

    #[inline(always)]
    fn skip(&mut self, n: u32) {
        self.acc <<= n;
        self.nbits -= n;
    }

    /// Bits consumed, counting zero bits invented past the end.
    fn consumed(&self) -> usize {
        self.pos * 8 - self.nbits as usize
    }
}

/// Decodes `width x (height + 8)` samples from `data` and keeps the first `height` rows.
pub fn decode_stream(data: &[u8], width: usize, height: usize) -> Result<Vec<u16>, RawError> {
    check_dimensions(width, height)?;
    let stored_rows = height + EXTRA_ROWS;
    let table = lookup_table();
    let mut out = vec![0u16; width * height];
    let mut bits = Bits::new(data);
    let mut sum = 0i32;
    for col in (0..width).rev() {
        // Even rows, then odd rows.
        for start in [0usize, 1] {
            for row in (start..stored_rows).step_by(2) {
                bits.fill();
                let entry = table[bits.peek(LOOKUP_BITS) as usize];
                bits.skip(u32::from(entry >> 8));
                let length = u32::from(entry & 0xFF);
                let diff = match length {
                    0 => 0,
                    // dcraw: a length of 16 is the escape for -32768.
                    16 => -32768,
                    n => {
                        let value = bits.peek(n) as i32;
                        bits.skip(n);
                        if value & (1 << (n - 1)) == 0 {
                            value - ((1 << n) - 1)
                        } else {
                            value
                        }
                    }
                };
                sum += diff;
                if !(0..4096).contains(&sum) {
                    return Err(RawError::corrupt(
                        "the ARW 1 data holds a sample outside the 12-bit range",
                    ));
                }
                if row < height {
                    out[row * width + col] = sum as u16;
                }
            }
        }
        if bits.consumed() > data.len() * 8 {
            return Err(RawError::corrupt(
                "the ARW 1 data ends before the image does",
            ));
        }
    }
    Ok(out)
}

/// Decodes the ARW 1 image described by `meta` (a single strip).
pub fn decode(src: &dyn ByteSource, meta: &RawMetadata) -> Result<Plane<u16>, RawError> {
    let (width, height) = (meta.width as usize, meta.height as usize);
    check_dimensions(width, height)?;
    let RawStorage::Strips {
        offsets,
        byte_counts,
        ..
    } = &meta.storage
    else {
        return Err(RawError::corrupt(
            "an ARW 1 image must be stored in a strip",
        ));
    };
    let (&offset, &count) = offsets
        .first()
        .zip(byte_counts.first())
        .ok_or_else(|| RawError::corrupt("the ARW 1 image has no data"))?;
    if offsets.len() != 1 {
        return Err(RawError::corrupt("an ARW 1 image must be a single strip"));
    }
    // White noise needs about two bytes per sample; four is far beyond any real file and keeps a
    // hostile byte count from allocating more than the file holds.
    if count > (width * height * 4) as u64 || offset.saturating_add(count) > src.len() {
        return Err(RawError::corrupt(
            "the raw data runs past the end of the file (the file is cut short)",
        ));
    }
    let mut data = vec![0u8; count as usize];
    src.read_at(offset, &mut data)?;
    let samples = decode_stream(&data, width, height)?;
    Plane::from_vec(width, height, samples)
        .ok_or_else(|| RawError::corrupt("internal size mismatch"))
}

/// The encoder for the tests: the inverse of [`decode_stream`].
#[cfg(any(test, feature = "testgen"))]
pub fn encode_stream(samples: &[u16], width: usize, height: usize) -> Vec<u8> {
    struct Sink {
        out: Vec<u8>,
        acc: u64,
        nbits: u32,
    }
    impl Sink {
        fn put(&mut self, value: u32, bits: u32) {
            self.acc = (self.acc << bits) | u64::from(value & ((1u32 << bits) - 1));
            self.nbits += bits;
            while self.nbits >= 8 {
                self.out.push((self.acc >> (self.nbits - 8)) as u8);
                self.nbits -= 8;
            }
            self.acc &= (1u64 << self.nbits) - 1;
        }
    }

    let stored_rows = height + EXTRA_ROWS;
    // Codes by difference bit length: (code, code length), assigned the way `lookup_table` does,
    // as running 15-bit prefixes.
    let mut codes = [(0u32, 0u32); 18];
    let mut next = 0u32;
    for (length, symbol) in CODES {
        codes[symbol as usize] = (next >> (LOOKUP_BITS - length), length);
        next += 1 << (LOOKUP_BITS - length);
    }
    let mut sink = Sink {
        out: Vec::new(),
        acc: 0,
        nbits: 0,
    };
    let mut sum = 0i32;
    for col in (0..width).rev() {
        for start in [0usize, 1] {
            for row in (start..stored_rows).step_by(2) {
                // The extra rows repeat the last image row.
                let value = i32::from(samples[row.min(height - 1) * width + col]);
                let diff = value - sum;
                sum = value;
                let length = if diff == 0 {
                    0
                } else {
                    32 - diff.unsigned_abs().leading_zeros()
                };
                let (code, code_length) = codes[length as usize];
                sink.put(code, code_length);
                if length > 0 {
                    let bits = if diff > 0 {
                        diff
                    } else {
                        diff + (1 << length) - 1
                    };
                    sink.put(bits as u32, length);
                }
            }
        }
    }
    if sink.nbits > 0 {
        let pad = 8 - sink.nbits;
        sink.put((1 << pad) - 1, pad);
    }
    sink.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixed_code_is_a_complete_prefix_code() {
        let kraft: f64 = CODES.iter().map(|(l, _)| 1.0 / f64::from(1u32 << l)).sum();
        assert!((kraft - 1.0).abs() < 1e-12, "{kraft}");
        let table = lookup_table();
        assert!(table.iter().all(|e| *e != 0), "every 15-bit prefix decodes");
        assert_eq!(table[0], (15 << 8) | 17); // the all-zero prefix is the first code
    }

    #[test]
    fn round_trips_including_the_extra_rows() {
        let (w, h) = (37, 21);
        let mut v = 100i32;
        let samples: Vec<u16> = (0..w * h)
            .map(|i| {
                v = (v * 97 + 31 * i as i32) % 4096;
                v as u16
            })
            .collect();
        let bytes = encode_stream(&samples, w, h);
        assert_eq!(decode_stream(&bytes, w, h).expect("decodes"), samples);
        assert!(decode_stream(&bytes[..bytes.len() / 2], w, h).is_err());
    }
}
