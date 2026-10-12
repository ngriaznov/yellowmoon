//! Filtered-file check (`docs/06-noise.md` step 9).
//!
//! White noise averaged over 2 x 2 neighbours loses three quarters of its variance, so the
//! variance of 2 x 2-binned residuals over the per-pixel residual variance is 1/4. Spatially
//! correlated noise (a camera that filtered the raw: Sony's "star eater" on long exposures, or other
//! in-camera processing) keeps more of it. Both variances are measured on the smooth windows from
//! plane-fit residuals (a plane fit removes any gradient; the residual sums of squares divided by
//! their degrees of freedom are unbiased, so white noise gives exactly 1/4 in expectation):
//!
//! ```text
//! ratio = sum(ss_bin / 6) / sum(ss_pix / 33)
//! ```
//!
//! If `ratio > 0.4` the file counts as filtered. The Haar coefficients that the main estimate is
//! built on cancel correlated noise (a smoothing filter shrinks `dd` while keeping low frequencies),
//! so the estimate reads low. The pixel-level standard deviation is restored with
//! `factor = sqrt(mean pixel residual variance / mean model pixel variance)`, clamped to at least
//! 1.3 (`docs/06`) and at most 4, and `a`, `b` and `sigma_row` are scaled accordingly. Over-estimating
//! is the safer error for a denoiser.

use serde::{Deserialize, Serialize};

use super::EstimatorConfig;
use super::tiles::{DOF_BIN, DOF_PIX, TileData, TilePlan, Window, residual_sums};
use crate::model::NoiseModel;

/// Result of the filtered-file check.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilteredCheck {
    /// False when there were too few smooth windows to decide.
    pub evaluated: bool,
    /// Windows used.
    pub windows: usize,
    /// Binned-over-pixel residual variance ratio (0.25 for white noise).
    pub ratio: f64,
    /// True when the ratio exceeds the threshold.
    pub flagged: bool,
    /// Factor applied to sigma (1 when not flagged).
    pub factor: f64,
}

impl Default for FilteredCheck {
    fn default() -> Self {
        Self {
            evaluated: false,
            windows: 0,
            ratio: 0.0,
            flagged: false,
            factor: 1.0,
        }
    }
}

/// Fewest smooth windows needed to decide.
const MIN_WINDOWS: usize = 200;

/// Fraction of the (dark half of the) smooth windows that is kept: the ones whose pixel residual is
/// smallest compared with what the fitted model predicts. White noise and uniformly filtered noise
/// look alike in every window, while leftover texture raises the residual of some windows, so the
/// cleanest quarter is the part that texture reaches last. (Ranking by a statistic that excludes
/// the window's own pixels would avoid biasing its residuals, but it also lets windows next to
/// axis-aligned edges through, which the diagonal detail cannot see, and that costs far more.)
const KEEP_FRACTION: f64 = 0.25;

/// Runs the check on the smooth windows. `wins` are the windows of all tiles in tile order, as
/// laid out by `plan`; the residual sums are computed here from `tiles`, for the windows used.
pub fn check_filtered(
    tiles: &[TileData],
    plan: &TilePlan,
    wins: &[Window],
    smooth: &[bool],
    model: &NoiseModel,
    cfg: &EstimatorConfig,
) -> FilteredCheck {
    let per_tile = plan.windows_per_tile();
    let n = plan.windows_per_axis();
    // Only the darker half of the smooth windows: scene texture scales with the signal while noise
    // does not (it has a floor), so shadows are where noise is least outweighed by texture.
    let mut means: Vec<f32> = wins
        .iter()
        .zip(smooth)
        .filter(|&(_, &s)| s)
        .map(|(w, _)| w.mean)
        .collect();
    if means.len() < 2 * MIN_WINDOWS {
        return FilteredCheck::default();
    }
    let dark_limit = super::stats::median(&mut means);
    // (pixel residual variance, binned residual variance, model pixel variance)
    let mut rows: Vec<(f64, f64, f64)> = wins
        .iter()
        .enumerate()
        .zip(smooth)
        .filter(|&((_, w), &s)| s && w.mean <= dark_limit)
        .map(|((idx, w), _)| {
            let (t, local) = (idx / per_tile, idx % per_tile);
            let (plane, wi, wj) = (local / (n * n), (local / n) % n, local % n);
            let (ss_pix, ss_bin) = residual_sums(&tiles[t], plan.size, plane, wi, wj);
            (
                ss_pix / DOF_PIX,
                ss_bin / DOF_BIN,
                model
                    .pixel_variance(f64::from(w.mean), w.plane as usize)
                    .max(1e-14),
            )
        })
        .collect();
    if rows.len() < MIN_WINDOWS {
        return FilteredCheck::default();
    }
    let keep = (((rows.len() as f64) * KEEP_FRACTION).ceil() as usize)
        .max(MIN_WINDOWS / 2)
        .min(rows.len());
    // The `keep` windows with the smallest pixel residual relative to the model (no full sort).
    let key = |r: &(f64, f64, f64)| r.0 / r.2;
    if keep < rows.len() {
        rows.select_nth_unstable_by(keep - 1, |a, b| key(a).total_cmp(&key(b)));
    }
    let (mut s_pix, mut s_bin, mut s_pred) = (0.0, 0.0, 0.0);
    for r in &rows[..keep] {
        s_pix += r.0;
        s_bin += r.1;
        s_pred += r.2;
    }
    let ratio = s_bin / s_pix;
    let flagged = ratio > cfg.filtered_ratio_threshold;
    let factor = if flagged {
        (s_pix / s_pred)
            .sqrt()
            .clamp(cfg.filtered_min_factor, cfg.filtered_max_factor)
    } else {
        1.0
    };
    FilteredCheck {
        evaluated: true,
        windows: keep,
        ratio,
        flagged,
        factor,
    }
}
