//! R1 normalize: black subtract, white normalize, clip mask (`docs/05-raw-processing.md` section R1).
//!
//! `x = (raw - black[c]) / (white[c] - black[c])` per CFA position. Negative values stay, because
//! they carry the noise statistics, but are floored at [`NEGATIVE_FLOOR`]. The clip mask flags
//! `raw >= white[c] - 4 * qstep` where the decoder supplies a quantisation step, and
//! `x >= 0.995` where it does not.
//!
//! Pointwise: tiles give bit-identical results to the whole frame, provided the tile's
//! [`Placement`] is right (the per-position levels are indexed by frame parity).

use rayon::prelude::*;

use super::{Cfa, CfaColor, ClipMask, CpuError, Mosaic, Placement};

/// Normalised values are floored here (`docs/05` R1).
pub const NEGATIVE_FLOOR: f32 = -0.05;
/// Without a quantisation step, a photosite is clipped from this normalised level up.
pub const CLIP_FRACTION: f32 = 0.995;
/// With a quantisation step, a photosite is clipped from `white - QSTEP_MARGIN * qstep` up.
pub const QSTEP_MARGIN: f32 = 4.0;

/// A raw sample type: `u16` straight from the decoder or `f32` from a previous step.
pub trait RawSample: Copy + Send + Sync {
    /// The sample as `f32` (exact for `u16`).
    fn to_f32(self) -> f32;
}

impl RawSample for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}

impl RawSample for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Black and white levels per CFA position, in raw units.
///
/// Index `2 * (y & 1) + (x & 1)` of the active area: for RGGB that is R, G1, G2, B. All Sony
/// bodies are RGGB, so this is also the RGGB channel order of SR2 tags 0x7310 and 0x787F.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Levels {
    /// Black level per CFA position.
    pub black: [f32; 4],
    /// White (saturation) level per CFA position.
    pub white: [f32; 4],
}

impl Levels {
    /// The same black and white for all four positions.
    pub const fn uniform(black: f32, white: f32) -> Levels {
        Levels {
            black: [black; 4],
            white: [white; 4],
        }
    }

    /// Levels from four values in the colour order of the SR2 tags (R, G1, G2, B, with G1 the green
    /// in the red rows and G2 the green in the blue rows), for a frame whose first photosite has the
    /// pattern `cfa`: each value lands on the photosite position of its colour, whatever the phase of
    /// the pattern (a frame cropped to an odd origin starts with GRBG, say).
    pub fn from_rggb(black_rggb: [f32; 4], white_rggb: [f32; 4], cfa: Cfa) -> Levels {
        let colour_index = |x: usize, y: usize| match cfa.color_at(x, y) {
            CfaColor::Red => 0,
            CfaColor::Blue => 3,
            CfaColor::Green if cfa.color_at(x ^ 1, y) == CfaColor::Red => 1,
            CfaColor::Green => 2,
        };
        let mut levels = Levels::uniform(0.0, 0.0);
        for p in 0..4 {
            let i = colour_index(p & 1, p >> 1);
            levels.black[p] = black_rggb[i];
            levels.white[p] = white_rggb[i];
        }
        levels
    }

    /// Checks `black < white` everywhere and that all values are finite.
    pub fn validate(&self) -> Result<(), CpuError> {
        for i in 0..4 {
            let (b, w) = (self.black[i], self.white[i]);
            if !(b.is_finite() && w.is_finite() && b < w) {
                return Err(CpuError::Mismatch(format!(
                    "position {i}: black {b} must be below white {w}"
                )));
            }
        }
        Ok(())
    }
}

/// Normalises a whole-frame raw plane. See [`normalize_tile`].
pub fn normalize<T: RawSample>(
    raw: &[T],
    width: usize,
    height: usize,
    cfa: Cfa,
    levels: &Levels,
    qstep: Option<&[f32]>,
) -> Result<(Mosaic, ClipMask), CpuError> {
    normalize_tile(
        raw,
        width,
        height,
        Placement::whole(width, height),
        cfa,
        levels,
        qstep,
    )
}

/// Normalises a raw plane that is the part of the frame described by `place`.
///
/// `cfa` is the pattern at the plane's own first pixel; `levels` are indexed by the *frame*
/// parity of each photosite (`place.x0`, `place.y0` shift the index). `qstep`, when present, has
/// one value per photosite in raw units.
pub fn normalize_tile<T: RawSample>(
    raw: &[T],
    width: usize,
    height: usize,
    place: Placement,
    cfa: Cfa,
    levels: &Levels,
    qstep: Option<&[f32]>,
) -> Result<(Mosaic, ClipMask), CpuError> {
    levels.validate()?;
    if raw.len() != width * height {
        return Err(CpuError::BadLength {
            got: raw.len(),
            expected: width * height,
            width,
            height,
        });
    }
    if let Some(q) = qstep
        && q.len() != raw.len()
    {
        return Err(CpuError::BadLength {
            got: q.len(),
            expected: raw.len(),
            width,
            height,
        });
    }
    let mut out = Mosaic {
        width,
        height,
        cfa,
        place,
        data: vec![0.0; width * height],
    };
    let mut mask = ClipMask::new(width, height);
    let inv: [f32; 4] = std::array::from_fn(|i| 1.0 / (levels.white[i] - levels.black[i]));
    out.data
        .par_chunks_mut(width.max(1))
        .zip(mask.data.par_chunks_mut(width.max(1)))
        .enumerate()
        .for_each(|(y, (orow, mrow))| {
            let rrow = &raw[y * width..(y + 1) * width];
            let qrow = qstep.map(|q| &q[y * width..(y + 1) * width]);
            for x in 0..width {
                let p = Cfa::position(place.x0 + x, place.y0 + y);
                let v = rrow[x].to_f32();
                let n = (v - levels.black[p]) * inv[p];
                orow[x] = n.max(NEGATIVE_FLOOR);
                let clipped = match qrow {
                    Some(q) => v >= levels.white[p] - QSTEP_MARGIN * q[x],
                    None => n >= CLIP_FRACTION,
                };
                mrow[x] = u8::from(clipped);
            }
        });
    Ok((out, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_black_to_zero_and_white_to_one() {
        let lv = Levels {
            black: [512.0, 514.0, 510.0, 512.0],
            white: [16383.0; 4],
        };
        let raw: Vec<u16> = vec![512, 16383, 510, 16383];
        let (m, mask) = normalize(&raw, 2, 2, Cfa::RGGB, &lv, None).unwrap();
        assert_eq!(m.data[0], 0.0);
        assert!((m.data[1] - 1.0).abs() < 1e-6);
        assert_eq!(m.data[2], 0.0);
        assert!((m.data[3] - 1.0).abs() < 1e-6);
        assert_eq!(mask.data, vec![0, 1, 0, 1]);
    }

    #[test]
    fn negative_values_are_kept_down_to_the_floor() {
        let lv = Levels::uniform(512.0, 1000.0);
        let raw: Vec<u16> = vec![500, 400, 512, 600];
        let (m, _) = normalize(&raw, 2, 2, Cfa::RGGB, &lv, None).unwrap();
        // 500 -> -12 / 488 = -0.0246 is kept; 400 -> -0.2295 is floored at -0.05.
        assert!(m.data[0] < 0.0 && m.data[0] > NEGATIVE_FLOOR);
        assert_eq!(m.data[1], NEGATIVE_FLOOR);
    }

    #[test]
    fn qstep_moves_the_clip_threshold() {
        let lv = Levels::uniform(0.0, 1000.0);
        let raw: Vec<u16> = vec![990, 990, 1000, 990];
        let q = vec![0.0, 4.0, 0.0, 2.0];
        let (_, mask) = normalize(&raw, 2, 2, Cfa::RGGB, &lv, Some(&q)).unwrap();
        // 990 >= 1000 - 4 * 4 = 984 -> clipped; 990 >= 1000 - 4 * 2 = 992 -> not; q = 0 needs raw >= 1000.
        assert_eq!(mask.data, vec![0, 1, 1, 0]);
        // Without qstep the 0.995 rule applies: 990 / 1000 = 0.99 is not clipped.
        let (_, mask) = normalize(&raw, 2, 2, Cfa::RGGB, &lv, None).unwrap();
        assert_eq!(mask.data, vec![0, 0, 1, 0]);
    }

    #[test]
    fn levels_follow_frame_parity_in_tiles() {
        let lv = Levels {
            black: [10.0, 20.0, 30.0, 40.0],
            white: [1010.0, 1020.0, 1030.0, 1040.0],
        };
        let (w, h) = (6, 6);
        let raw: Vec<u16> = (0..w * h).map(|i| (i * 7 % 1000) as u16 + 50).collect();
        let (whole, wmask) = normalize(&raw, w, h, Cfa::GRBG, &lv, None).unwrap();
        // Tile at odd origin (1, 3), 4 x 3.
        let tile_raw: Vec<u16> = (3..6)
            .flat_map(|y| (1..5).map(move |x| (y, x)))
            .map(|(y, x)| raw[y * w + x])
            .collect();
        let place = Placement::whole(w, h).offset(1, 3);
        let (tile, tmask) =
            normalize_tile(&tile_raw, 4, 3, place, Cfa::GRBG.shifted(1, 3), &lv, None).unwrap();
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(tile.get(x, y), whole.get(x + 1, y + 3));
                assert_eq!(tmask.data[y * 4 + x], wmask.data[(y + 3) * w + x + 1]);
            }
        }
        assert_eq!(tile.cfa, Cfa::GRBG.shifted(1, 3));
    }

    #[test]
    fn rggb_levels_follow_the_colours_for_every_phase() {
        let black = [10.0, 20.0, 30.0, 40.0]; // R, G1 (red rows), G2 (blue rows), B
        let white = [1010.0, 1020.0, 1030.0, 1040.0];
        for cfa in Cfa::ALL {
            let lv = Levels::from_rggb(black, white, cfa);
            lv.validate().unwrap();
            for y in 0..2 {
                for x in 0..2 {
                    let want = match cfa.color_at(x, y) {
                        CfaColor::Red => 0,
                        CfaColor::Blue => 3,
                        CfaColor::Green if cfa.color_at(x ^ 1, y) == CfaColor::Red => 1,
                        CfaColor::Green => 2,
                    };
                    assert_eq!(
                        lv.black[Cfa::position(x, y)],
                        black[want],
                        "{} ({x},{y})",
                        cfa.name()
                    );
                    assert_eq!(lv.white[Cfa::position(x, y)], white[want]);
                }
            }
        }
        // RGGB is the identity: positions R, G1, G2, B.
        assert_eq!(Levels::from_rggb(black, white, Cfa::RGGB).black, black);
        // GRBG starts with G1 (in a red row), then R; the second row is B, G2.
        assert_eq!(
            Levels::from_rggb(black, white, Cfa::GRBG).black,
            [20.0, 10.0, 40.0, 30.0]
        );
        // BGGR: B, G2 (blue row), G1 (red row), R.
        assert_eq!(
            Levels::from_rggb(black, white, Cfa::BGGR).black,
            [40.0, 30.0, 20.0, 10.0]
        );
    }

    #[test]
    fn rejects_bad_input() {
        let lv = Levels::uniform(100.0, 50.0);
        assert!(normalize(&[0u16; 4], 2, 2, Cfa::RGGB, &lv, None).is_err());
        let lv = Levels::uniform(0.0, 10.0);
        assert!(normalize(&[0u16; 3], 2, 2, Cfa::RGGB, &lv, None).is_err());
        assert!(normalize(&[0u16; 4], 2, 2, Cfa::RGGB, &lv, Some(&[0.0])).is_err());
    }
}
