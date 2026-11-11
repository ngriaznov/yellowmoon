//! The classical blemish proposer (`docs/27` S-07): a gated detect-and-fill.
//!
//! Dark spots are found by a per-channel black top-hat at blemish scale, red spots by a red-excess
//! test (redness above the local skin), both on slightly smoothed log values. A hysteresis
//! threshold and a connected-component test (size, aspect) give the detection mask, grown a little.
//! Inside the mask the low-frequency base of each channel is replaced by a harmonic (membrane)
//! fill from the rim and the fine detail is kept (clamped). **Everywhere outside the mask the
//! output is exactly the input**, so clean skin gets no veil at any Blemish amount.
//!
//! The proposer works in the encoding it receives (`docs/27` 6.1 step 2: square root of the linear
//! value, mean-normalised). All thresholds are in natural-log units of that encoding, which is half
//! the log of the linear ratio. Every constant below is a starting value for S-12 (tuning).

use rayon::prelude::*;

use super::grey::{Grey, components, dilate_disc};
use crate::error::{Result, RetouchError};
use crate::models::BlemishProposer;

/// Smallest value taken to the logarithm (the encoded value of black).
pub const LOG_FLOOR: f64 = 1e-4;
/// Smallest blemish diameter, as a fraction of the face size `D` (`docs/27` S-06: 0.7 %).
pub const SPOT_MIN_FRAC: f64 = 0.007;
/// Largest blemish diameter, as a fraction of `D` (3.5 %). Sets the radius of the top-hat window.
pub const SPOT_MAX_FRAC: f64 = 0.035;
/// Spots whose bounding box is smaller than this many pixels (both sides) are dropped.
pub const MIN_SPOT_SIDE_PX: f64 = 3.0;
/// Components whose longer bounding-box side exceeds this fraction of `D` are not blemishes
/// (lips, brows, creases, shadows).
pub const MAX_SIDE_FRAC: f64 = 0.05;
/// Components more elongated than this (long side over short side) are hairs or lines, not spots.
pub const ASPECT_MAX: f64 = 3.5;
/// Gaussian sigma of the smoothing before the top-hat, as a fraction of `D` (at least
/// [`PRE_SIGMA_MIN_PX`]); it keeps pores and sensor noise out of the detection.
pub const PRE_SIGMA_FRAC: f64 = 0.0035;
/// Lower bound of the pre-smoothing sigma in pixels.
pub const PRE_SIGMA_MIN_PX: f64 = 0.7;
/// Sigma of the base layer of the frequency separation, as a fraction of `D` (at least
/// [`SEP_SIGMA_MIN_PX`]); roughly pore scale, so spots are in the base and texture in the detail.
pub const SEP_SIGMA_FRAC: f64 = 0.003;
/// Lower bound of the separation sigma in pixels.
pub const SEP_SIGMA_MIN_PX: f64 = 1.0;
/// Top-hat depth (natural log of the encoded value, any channel) that seeds a dark spot.
/// 0.05 is about 10 % in linear light.
pub const DARK_HI: f64 = 0.05;
/// Red excess (`ln R - ln G` above the local skin) that seeds a red spot.
pub const RED_HI: f64 = 0.06;
/// A pixel joins a seeded spot when its score is at least this fraction of the seed threshold.
pub const LO_RATIO: f64 = 0.55;
/// Radius by which the detection mask is grown, as a fraction of `D` (at least 1 px).
pub const DILATE_FRAC: f64 = 0.004;
/// The kept detail inside a spot is clamped to this many natural-log units, so the spot's own
/// crisp rim does not come back as a ring.
pub const DETAIL_CLAMP: f64 = 0.03;
/// Over-relaxation factor of the harmonic fill.
pub const SOR_OMEGA: f64 = 1.6;
/// Sweeps of the harmonic fill (fixed, so the result is deterministic).
pub const FILL_SWEEPS: usize = 120;
/// Default side of the output tile in pixels.
pub const TILE_PX: u32 = 320;

/// The score of a pixel from its per-channel black top-hat depths and its red excess: the larger
/// of the deepest channel over [`DARK_HI`] and the excess over [`RED_HI`]. 1 means "a seed".
/// Non-finite or negative inputs count as 0, so the result is finite and non-negative.
pub fn spot_score(tophat: [f64; 3], red_excess: f64) -> f64 {
    let clean = |v: f64| if v.is_finite() { v.max(0.0) } else { 0.0 };
    let dark = tophat.iter().copied().map(clean).fold(0.0, f64::max);
    (dark / DARK_HI).max(clean(red_excess) / RED_HI)
}

/// The classical [`BlemishProposer`]. Built per face with the face size in map pixels.
#[derive(Debug, Clone)]
pub struct ClassicalProposer {
    face_px: f64,
    tile_px: u32,
    closing_r: usize,
    pre_sigma: f64,
    sep_sigma: f64,
    dilate_r: usize,
    min_side: f64,
    max_side: f64,
    border_px: u32,
}

impl ClassicalProposer {
    /// A proposer for a face of `face_px` pixels in the blemish map (`min(face, 450)`; the trait
    /// passes no face size, so it is given here). A non-finite or tiny size counts as 32.
    pub fn new(face_px: f64) -> Self {
        Self::with_tile_px(face_px, TILE_PX)
    }

    /// Like [`ClassicalProposer::new`] with an output tile of `tile_px` (at least 1). The result
    /// does not depend on the tile size except within a few pixels of tile seams, because the
    /// border covers every filter's support and the largest spot.
    pub fn with_tile_px(face_px: f64, tile_px: u32) -> Self {
        let d = if face_px.is_finite() && face_px >= 8.0 {
            face_px.min(8192.0)
        } else {
            32.0
        };
        let closing_r = ((0.5 * SPOT_MAX_FRAC * d).ceil() as usize + 1).max(2);
        let pre_sigma = (PRE_SIGMA_FRAC * d).max(PRE_SIGMA_MIN_PX);
        let sep_sigma = (SEP_SIGMA_FRAC * d).max(SEP_SIGMA_MIN_PX);
        let dilate_r = ((DILATE_FRAC * d).round() as usize).max(1);
        let max_side = (MAX_SIDE_FRAC * d).max(MIN_SPOT_SIDE_PX + 1.0);
        // Support of the detection (blur + opening or closing) + the largest spot (its component
        // is decided as a whole) + the fill's base blur + the growth + a margin.
        let border = 2 * closing_r
            + (3.0 * pre_sigma).ceil() as usize
            + (3.0 * sep_sigma).ceil() as usize
            + max_side.ceil() as usize
            + dilate_r
            + 4;
        Self {
            face_px: d,
            tile_px: tile_px.max(1),
            closing_r,
            pre_sigma,
            sep_sigma,
            dilate_r,
            min_side: MIN_SPOT_SIDE_PX.max(SPOT_MIN_FRAC * d),
            max_side,
            border_px: border as u32,
        }
    }

    /// The face size in map pixels this proposer was built for.
    pub fn face_px(&self) -> f64 {
        self.face_px
    }

    /// The detection mask of a square context (log planes, side `side`), grown by a disc.
    fn detect(&self, logs: &[Grey; 3]) -> Vec<bool> {
        let (w, h) = (logs[0].w, logs[0].h);
        let smooth: Vec<Grey> = logs.iter().map(|l| l.gaussian(self.pre_sigma)).collect();
        let tophat: Vec<Grey> = smooth
            .iter()
            .map(|s| {
                let c = s.closing(self.closing_r);
                Grey::new(
                    w,
                    h,
                    c.data.iter().zip(&s.data).map(|(a, b)| a - b).collect(),
                )
            })
            .collect();
        let rho = Grey::new(
            w,
            h,
            smooth[0]
                .data
                .iter()
                .zip(&smooth[1].data)
                .map(|(r, g)| r - g)
                .collect(),
        );
        let open = rho.opening(self.closing_r);
        let score: Vec<f64> = (0..w * h)
            .map(|i| {
                spot_score(
                    [tophat[0].data[i], tophat[1].data[i], tophat[2].data[i]],
                    rho.data[i] - open.data[i],
                )
            })
            .collect();
        let raw: Vec<bool> = score.iter().map(|&s| s >= LO_RATIO).collect();
        let mut keep = vec![false; w * h];
        for comp in components(&raw, w, h) {
            let (cw, ch) = (comp.width() as f64, comp.height() as f64);
            let (long, short) = (cw.max(ch), cw.min(ch));
            let seeded = comp.pixels.iter().any(|&p| score[p] >= 1.0);
            if !seeded || long < self.min_side || long > self.max_side || long / short > ASPECT_MAX
            {
                continue;
            }
            for &p in &comp.pixels {
                keep[p] = true;
            }
        }
        dilate_disc(&keep, w, h, self.dilate_r)
    }

    /// Fills the masked pixels of one channel: harmonic fill of the base from the rim, plus the
    /// clamped detail. `inside` marks the detection mask. Returns the new log values (only the
    /// masked entries are meaningful).
    fn fill_channel(&self, log: &Grey, inside: &[bool], outside_weight: &Grey) -> Vec<f64> {
        let (w, h) = (log.w, log.h);
        let masked: Vec<f64> = log
            .data
            .iter()
            .zip(inside)
            .map(|(&v, &m)| if m { 0.0 } else { v })
            .collect();
        let num = Grey::new(w, h, masked).gaussian(self.sep_sigma);
        let plain = log.gaussian(self.sep_sigma);
        // Base from the pixels outside the spots only (no spill of the spot into the rim).
        let mut base: Vec<f64> = num
            .data
            .iter()
            .zip(&outside_weight.data)
            .zip(inside)
            .map(|((&n, &d), &m)| if !m && d > 1e-3 { n / d } else { f64::NAN })
            .collect();
        let (mut sum, mut n) = (0.0, 0usize);
        for &b in base.iter().filter(|b| b.is_finite()) {
            sum += b;
            n += 1;
        }
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };
        let list: Vec<usize> = (0..w * h).filter(|&i| inside[i]).collect();
        for &i in &list {
            base[i] = mean;
        }
        for b in base.iter_mut().filter(|b| !b.is_finite()) {
            *b = mean;
        }
        // Successive over-relaxation on the masked pixels, raster order, fixed sweep count.
        for _ in 0..FILL_SWEEPS {
            for &p in &list {
                let (x, y) = (p % w, p / w);
                let (mut acc, mut cnt) = (0.0, 0.0);
                if x > 0 {
                    acc += base[p - 1];
                    cnt += 1.0;
                }
                if x + 1 < w {
                    acc += base[p + 1];
                    cnt += 1.0;
                }
                if y > 0 {
                    acc += base[p - w];
                    cnt += 1.0;
                }
                if y + 1 < h {
                    acc += base[p + w];
                    cnt += 1.0;
                }
                if cnt > 0.0 {
                    base[p] += SOR_OMEGA * (acc / cnt - base[p]);
                }
            }
        }
        list.iter()
            .map(|&p| {
                let detail = (log.data[p] - plain.data[p]).clamp(-DETAIL_CLAMP, DETAIL_CLAMP);
                base[p] + detail
            })
            .collect()
    }
}

impl BlemishProposer for ClassicalProposer {
    fn output_size_px(&self) -> u32 {
        self.tile_px
    }

    fn border_px(&self) -> u32 {
        self.border_px
    }

    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        let (tile, border) = (self.tile_px as usize, self.border_px as usize);
        let side = tile + 2 * border;
        if input.len() != 3 * side * side {
            return Err(RetouchError::Model(format!(
                "the proposer got {} values, expected {}",
                input.len(),
                3 * side * side
            )));
        }
        let plane = side * side;
        let copy_centre = |out: &mut Vec<f32>| {
            for c in 0..3 {
                for y in 0..tile {
                    let start = c * plane + (y + border) * side + border;
                    out.extend_from_slice(&input[start..start + tile]);
                }
            }
        };
        let mut out = Vec::with_capacity(3 * tile * tile);
        let logs: [Grey; 3] = std::array::from_fn(|c| {
            Grey::new(
                side,
                side,
                input[c * plane..(c + 1) * plane]
                    .iter()
                    .map(|&v| f64::from(v).max(LOG_FLOOR).ln())
                    .collect(),
            )
        });
        let inside = self.detect(&logs);
        if !inside.iter().any(|&m| m) {
            copy_centre(&mut out);
            return Ok(out);
        }
        let ones = Grey::new(
            side,
            side,
            inside.iter().map(|&m| if m { 0.0 } else { 1.0 }).collect(),
        )
        .gaussian(self.sep_sigma);
        let filled: Vec<Vec<f64>> = (0..3)
            .into_par_iter()
            .map(|c| self.fill_channel(&logs[c], &inside, &ones))
            .collect();
        for (c, fill) in filled.iter().enumerate() {
            // Position of each masked pixel in `fill`, in raster order.
            let mut k = 0usize;
            let mut chan = vec![0.0f32; plane];
            for p in 0..plane {
                if inside[p] {
                    chan[p] = fill[k].exp() as f32;
                    k += 1;
                } else {
                    chan[p] = input[c * plane + p];
                }
            }
            for y in 0..tile {
                let start = (y + border) * side + border;
                out.extend_from_slice(&chan[start..start + tile]);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod unit {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn score_values() {
        assert_eq!(spot_score([0.0; 3], 0.0), 0.0);
        assert!((spot_score([0.0, DARK_HI, 0.0], 0.0) - 1.0).abs() < 1e-12);
        assert!((spot_score([0.0; 3], RED_HI * 2.0) - 2.0).abs() < 1e-12);
        assert_eq!(spot_score([f64::NAN, -1.0, f64::INFINITY], f64::NAN), 0.0);
    }

    #[test]
    fn geometry_scales_with_the_face() {
        let a = ClassicalProposer::new(450.0);
        let b = ClassicalProposer::new(200.0);
        assert!(a.border_px() > b.border_px());
        assert_eq!(a.output_size_px(), TILE_PX);
        assert_eq!(ClassicalProposer::new(f64::NAN).face_px(), 32.0);
        assert!(a.run(&[0.0; 5]).is_err());
    }

    proptest! {
        #[test]
        fn score_is_finite_and_not_negative(
            t in prop::array::uniform3(-10.0f64..10.0), e in -10.0f64..10.0,
        ) {
            let s = spot_score(t, e);
            prop_assert!(s.is_finite() && s >= 0.0);
        }
    }
}
