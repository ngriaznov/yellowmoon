//! Test-only cRAW encoder (M2-03): the inverse of `decode::craw`, with a choice of rounding.
//!
//! A true 14-bit value is first mapped to the nearest (or next lower) 11-bit code of the camera
//! curve, then each block of 16 same-colour pixels is stored as the largest and smallest code
//! exactly plus 7-bit deltas scaled by the block's shift. The rounding of the deltas is chosen
//! against the true value in output DN, so the decoded value is the nearest representable one
//! ([`Rounding::Nearest`]: `|true - decoded| <= qstep / 2`) or the largest one not above the true
//! value ([`Rounding::Floor`]: `0 <= true - decoded < qstep`), which is what the decoder's
//! `qstep` documents.

use super::{PayloadDescription, RawPayload};

/// The tone-curve knots all real cRAW files carry (tag 0x7010, as stored): decoded knots 2000,
/// 2600, 3225 and 3525 on the 12-bit index. The curve then ends at 17204 for the top code.
pub const REAL_KNOTS: [u16; 4] = [8000, 10400, 12900, 14100];

/// How a true value is rounded to the representable ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// To the nearest representable value in output DN.
    Nearest,
    /// To the largest representable value that is not above the true value.
    Floor,
}

/// The cRAW payload.
#[derive(Clone, Debug)]
pub struct CrawEncoder {
    /// The four tone-curve knots as stored in tag 0x7010.
    pub knots: [u16; 4],
    /// Rounding mode.
    pub rounding: Rounding,
}

impl Default for CrawEncoder {
    fn default() -> Self {
        Self {
            knots: REAL_KNOTS,
            rounding: Rounding::Nearest,
        }
    }
}

impl CrawEncoder {
    /// An encoder with the knots of real cameras and nearest rounding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the knots (as stored in the tag).
    pub fn knots(mut self, knots: [u16; 4]) -> Self {
        self.knots = knots;
        self
    }

    /// Replaces the rounding mode.
    pub fn rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    /// The decoded curve (4096 entries).
    pub fn curve(&self) -> Vec<u16> {
        crate::decode::craw::build_curve(self.knots).expect("knots of a test encoder must increase")
    }

    /// The largest value the curve can produce (the top 11-bit code).
    pub fn max_value(&self) -> u16 {
        self.curve()[4094]
    }
}

impl RawPayload for CrawEncoder {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: 32767,
            bits_per_sample: 12,
            photometric: 32803,
            samples_per_pixel: 1,
            tile: None,
            sony_raw_file_type: Some(2),
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>> {
        let (w, h) = (width as usize, height as usize);
        assert_eq!(w % 32, 0, "cRAW rows are whole 32-pixel groups");
        let curve = self.curve();
        let grid: Vec<u16> = (0..2048).map(|code| curve[2 * code]).collect();
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let line = &mosaic[row * w..(row + 1) * w];
            for group in line.chunks_exact(32) {
                for half in 0..2 {
                    let mut block = [0u16; 16];
                    for (i, slot) in block.iter_mut().enumerate() {
                        *slot = group[half + 2 * i];
                    }
                    out.extend_from_slice(&self.encode_block(&block, &grid).to_le_bytes());
                }
            }
        }
        vec![out]
    }
}

impl CrawEncoder {
    fn encode_block(&self, values: &[u16; 16], grid: &[u16]) -> u128 {
        // Step 1: the 11-bit code of every pixel.
        let codes: Vec<usize> = values
            .iter()
            .map(|t| {
                // First code whose grid value is not below t.
                let above = grid.partition_point(|g| g < t).min(2047);
                match self.rounding {
                    Rounding::Floor => {
                        if grid[above] <= *t {
                            above
                        } else {
                            above.saturating_sub(1)
                        }
                    }
                    Rounding::Nearest => {
                        let below = above.saturating_sub(1);
                        if i32::from(*t) - i32::from(grid[below])
                            <= i32::from(grid[above]) - i32::from(*t)
                        {
                            below
                        } else {
                            above
                        }
                    }
                }
            })
            .collect();
        let max_code = *codes.iter().max().unwrap_or(&0);
        let min_code = *codes.iter().min().unwrap_or(&0);
        let imax = codes.iter().position(|c| *c == max_code).unwrap_or(0);
        let imin = if max_code == min_code {
            usize::from(imax == 0)
        } else {
            codes.iter().position(|c| *c == min_code).unwrap_or(0)
        };
        let mut shift = 0u32;
        while shift < 4 && (0x80usize << shift) <= max_code - min_code {
            shift += 1;
        }
        let mut word: u128 = (max_code as u128)
            | ((min_code as u128) << 11)
            | ((imax as u128) << 22)
            | ((imin as u128) << 26);
        let mut bit = 30u32;
        for i in 0..16 {
            if i == imax || i == imin {
                continue;
            }
            let t = i32::from(values[i]);
            let d0 = ((codes[i] - min_code) >> shift).min(127);
            let decoded = |d: usize| usize::min(2047, min_code + (d << shift));
            let delta = match self.rounding {
                Rounding::Floor => d0,
                Rounding::Nearest => {
                    let err = |d: usize| (i32::from(grid[decoded(d)]) - t).abs();
                    if d0 < 127 && err(d0 + 1) < err(d0) {
                        d0 + 1
                    } else {
                        d0
                    }
                }
            };
            word |= (delta as u128) << bit;
            bit += 7;
        }
        word
    }
}
