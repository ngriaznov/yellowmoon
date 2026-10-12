//! Row noise (`docs/06-noise.md` step 8).
//!
//! Row noise is an offset shared by all photosites of one mosaic row. The diagonal Haar
//! coefficients cancel it, so it is measured separately, inside the smooth windows: for a
//! photosite at `(x, y)` take the second difference down its colour column,
//!
//! ```text
//! d2(x, y) = z(x, y - 2) - 2 z(x, y) + z(x, y + 2),
//! ```
//!
//! which removes any locally linear signal and turns independent row offsets `rho_y` into
//! `rho_(y-2) - 2 rho_y + rho_(y+2)`, of variance `6 sigma_row^2`. The median of `d2` over the smooth
//! photosites of one row segment is the row's residual `D`; it has variance `6 sigma_row^2 + p`, where
//! `p = (pi/2) * 6 * v_pix / N` is the variance of a median of `N` values of variance `6 v_pix`.
//!
//! Subtracting `p` needs the pixel model to be right to within the (small) fraction that row noise
//! makes of the pixel variance, and leftover texture adds to `D` just like row noise does. Both go
//! away with a **cross-product**: the 16 tiles come in rows of tiles that cover the *same* mosaic
//! rows, and a row offset is common to all of them, while pixel noise and texture are independent
//! between tiles. For two segments `i`, `j` of the same row in different tiles,
//! `E[D_i D_j] = 6 sigma_row^2`, with no pixel part to subtract. The estimate is the
//! inverse-variance weighted mean of the cross-products over all such pairs, accepted only when it
//! is at least three standard errors above zero and clamped at zero otherwise.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::EstimatorConfig;
use super::stats::{median, quantile};
use super::tiles::{TileData, TilePlan, WIN, Window};
use crate::model::NoiseModel;

/// Result of the row-noise measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RowNoiseInfo {
    /// False when there were too few smooth rows to measure anything (then `sigma_row` is 0).
    pub measured: bool,
    /// Row segment pairs that contributed.
    pub rows_used: usize,
    /// Estimated row-noise standard deviation, normalised units (0 when not significant).
    pub sigma_row: f64,
}

/// Fewest segment pairs needed for a measurement.
const MIN_PAIRS: usize = 150;

/// Significance (standard errors above zero) a row-noise estimate needs to be reported.
const MIN_SIGNIFICANCE: f64 = 3.0;

/// Pair statistics are clipped at this many standard deviations against leftover structure.
const CLIP_SIGMAS: f64 = 8.0;

/// Measures the row-noise standard deviation.
///
/// `smooth` flags the windows (in the order of `wins`, tile by tile) that belong to surviving,
/// unclipped bins; `model` is the fitted pixel-noise model.
#[allow(clippy::needless_range_loop)]
pub fn measure_row_noise(
    tiles: &[TileData],
    plan: &TilePlan,
    wins: &[Window],
    smooth: &[bool],
    model: &NoiseModel,
    cfg: &EstimatorConfig,
) -> RowNoiseInfo {
    let size = plan.size;
    let n = plan.windows_per_axis();
    let per_tile = plan.windows_per_tile();
    let per_axis = cfg.tiles_per_axis.max(1);
    if per_axis < 2 || tiles.len() != per_axis * per_axis {
        return RowNoiseInfo::default();
    }
    // Noise-level classes: row segments are formed per class so that dark segments, which show row
    // noise best, are not diluted by the shot noise of bright ones. Class limits are terciles of the
    // model variance of the smooth windows.
    let mut levels: Vec<f32> = wins
        .iter()
        .zip(smooth)
        .filter(|&(_, &s)| s)
        .map(|(w, _)| model.pixel_variance(f64::from(w.mean), w.plane as usize) as f32)
        .collect();
    if levels.len() < 3 * cfg.row_min_samples {
        return RowNoiseInfo::default();
    }
    let lim_lo = quantile(&mut levels.clone(), 1.0 / 3.0);
    let lim_hi = quantile(&mut levels, 2.0 / 3.0);
    let class_of = |v: f32| -> usize { usize::from(v > lim_lo) + usize::from(v > lim_hi) };
    // Per tile: for every (row, class) the median second difference and its pixel variance.
    let segments: Vec<Vec<Vec<(f32, f64)>>> = tiles
        .par_iter()
        .enumerate()
        .map(|(t, tile)| {
            // Only every second tile row is used: the statistics have far more pairs than they
            // need, and the cost is proportional to the tiles read.
            if (t / per_axis) % 2 == 1 {
                return Vec::new();
            }
            let mut vals: Vec<Vec<f32>> = (0..size * 3).map(|_| Vec::with_capacity(64)).collect();
            let mut var_sum = vec![0.0f64; size * 3];
            let p = &tile.pixels;
            for plane in 0..4 {
                let (py, px) = (plane >> 1, plane & 1);
                for wi in 0..n {
                    for wj in 0..n {
                        let idx = t * per_tile + plan.window_index(plane, wi, wj);
                        if !smooth[idx] {
                            continue;
                        }
                        let v_pix = model.pixel_variance(f64::from(wins[idx].mean), plane);
                        let class = class_of(v_pix as f32);
                        for r in 0..WIN {
                            // Every second plane row: neighbouring second differences share a row
                            // of pixels, so the other half adds little and costs the same again.
                            if r % 2 == 1 {
                                continue;
                            }
                            let y = 2 * (WIN * wi + r) + py;
                            if y < 2 || y + 2 >= size {
                                continue;
                            }
                            let base = y * size + 2 * WIN * wj + px;
                            let bucket = y * 3 + class;
                            for c in 0..WIN {
                                let x = base + 2 * c;
                                vals[bucket].push(p[x - 2 * size] - 2.0 * p[x] + p[x + 2 * size]);
                            }
                            var_sum[bucket] += v_pix * WIN as f64;
                        }
                    }
                }
            }
            // rows[y] = the segments of that row.
            let mut rows: Vec<Vec<(f32, f64)>> = vec![Vec::new(); size];
            for (b, row) in vals.iter_mut().enumerate() {
                if row.len() >= cfg.row_min_samples {
                    let count = row.len() as f64;
                    let pixel_part =
                        std::f64::consts::FRAC_PI_2 * 6.0 * (var_sum[b] / count) / count;
                    rows[b / 3].push((median(row), pixel_part));
                }
            }
            rows
        })
        .collect();
    // Cross-products between segments of the same row in different tiles of a tile row.
    let mut segs: Vec<(usize, f32, f64)> = Vec::new();
    // (D_i, pixel variance part p_i, D_j, p_j) of every pair.
    let mut pair_data: Vec<(f32, f64, f32, f64)> = Vec::new();
    for tile_row in (0..per_axis).step_by(2) {
        for y in 0..size {
            segs.clear();
            for tile_col in 0..per_axis {
                let t = tile_row * per_axis + tile_col;
                for &(d, p) in &segments[t][y] {
                    segs.push((t, d, p));
                }
            }
            for i in 0..segs.len() {
                for j in i + 1..segs.len() {
                    if segs[i].0 != segs[j].0 {
                        pair_data.push((segs[i].1, segs[i].2, segs[j].1, segs[j].2));
                    }
                }
            }
        }
    }
    if pair_data.len() < MIN_PAIRS {
        return RowNoiseInfo::default();
    }
    // Weighted mean of the cross-products: s = sum w D_i D_j / (6 sum w), w = 1 / (v_i v_j),
    // v = p + 6 s, iterated because the weights depend on s.
    let mut s = 0.0f64;
    let mut sum_w = 0.0f64;
    for _ in 0..6 {
        let (mut num, mut den) = (0.0, 0.0);
        for &(di, pi, dj, pj) in &pair_data {
            let (vi, vj) = (pi + 6.0 * s, pj + 6.0 * s);
            let w = 1.0 / (vi * vj);
            let limit = CLIP_SIGMAS * (vi * vj).sqrt();
            let prod = (f64::from(di) * f64::from(dj)).clamp(-limit, limit);
            num += w * prod;
            den += 6.0 * w;
        }
        sum_w = den / 6.0;
        s = (num / den).max(0.0);
    }
    // Standard error of s with the pair correlation (a segment appears in several pairs) allowed for.
    let se = 3.0f64.sqrt() / (6.0 * sum_w.sqrt());
    RowNoiseInfo {
        measured: true,
        rows_used: pair_data.len(),
        sigma_row: if s > MIN_SIGNIFICANCE * se {
            s.sqrt()
        } else {
            0.0
        },
    }
}
