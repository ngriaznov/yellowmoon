//! Smooth-region selection and the per-bin variance measurements (`docs/06-noise.md` steps 2 to 4).
//!
//! One *pass* takes all windows, bins them by mean intensity (24 bins), keeps the lowest-energy
//! fraction of each bin (Foi 2008), and turns the diagonal Haar coefficients of the kept windows
//! into one variance point per bin and CFA plane.
//!
//! **Stabilised passes.** In the first pass the energy and the coefficients are used as they are.
//! A bin is wide, though: the variance inside it varies by tens of percent, and selecting the
//! lowest energy then also selects windows whose noise happens to be lower, which biases the
//! measured variance down. Later passes therefore divide each window's energy and coefficients by
//! the noise standard deviation the previous fit predicts at that window; the bin statistic becomes
//! `median(|d / sigma_pred|)` (a dimensionless factor near 1 when the model is right), the point is
//! `factor * var_pred(mean)`, and the fit is repeated. The fixed point of this iteration satisfies
//! the estimating equation "the normalised MAD is 0.6745 in every bin", which does not depend on
//! how wide the bins are.
//!
//! **Texture test.** A bin is kept only if the shape of its coefficient distribution looks
//! Gaussian: for white Gaussian noise `Q75(|d|) / Q50(|d|)` is 1.7055, and texture that survives
//! the energy selection fattens the tails. Bins that fail are dropped, which is what lets the
//! number of surviving bins signal a busy scene (step 6).
//!
//! **Clipped bins.** Bins whose mean is above `clip_bin_threshold` use the sample variance of the
//! coefficients (the MAD is meaningless when many coefficients are exactly zero) and are mapped
//! back to the unclipped pair `(mu, sigma^2)` with the clipped-Gaussian moments of [`super::clip`].

use serde::{Deserialize, Serialize};

use super::bins::{BinLayout, NUM_BINS};
use super::clip;
use super::stats::{mad_variance, median, quantile};
use super::tiles::{COEFFS, Window};
use super::{BinPoint, EstimatorConfig, PointKind};
use crate::model::NoiseModel;

/// `Q75(|z|) / Q50(|z|)` for a standard normal `z`.
pub const GAUSSIAN_TAIL_RATIO: f64 = 1.150_349_4 / 0.674_489_75;

/// Relative variance of a MAD-based variance estimate: `Var(v) / v^2 = MAD_REL_VAR / n`.
pub const MAD_REL_VAR: f64 = 5.5;

/// Relative variance of a sample-variance estimate of clipped data (conservative).
const MOMENT_REL_VAR: f64 = 3.0;

/// A bin counts as clipped when its mean plus this many predicted sigmas reaches the clip level.
const CLIP_REACH_SIGMAS: f64 = 2.5;

/// Windows whose largest coefficient exceeds this many predicted sigmas are left out of the
/// sample variance of a clipped bin.
const OUTLIER_SIGMAS: f64 = 6.0;

/// What a pass knows about the noise already.
pub struct PassParams<'a> {
    /// Intensity bins.
    pub layout: &'a BinLayout,
    /// Model from the previous pass, used to normalise energies and coefficients; `None` in the first pass.
    pub norm_model: Option<&'a NoiseModel>,
    /// Model used to reject outliers in clipped bins (the previous fit, or one built from the prior).
    pub ref_model: Option<&'a NoiseModel>,
    /// Handle bins above the clipping threshold (needs `ref_model` or at least the prior to reject outliers).
    pub include_clipped: bool,
    /// Estimator settings.
    pub cfg: &'a EstimatorConfig,
}

/// What happened to one intensity bin in a pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BinInfo {
    /// Coefficients kept after the energy selection (all planes).
    pub samples: u32,
    /// `Q75(|d|) / Q50(|d|)` of the kept coefficients (0 when not computed).
    pub tail_ratio: f64,
    /// True when the bin failed the texture test.
    pub textured: bool,
    /// True when the bin was treated with the clipped-Gaussian likelihood.
    pub clipped: bool,
    /// True when the bin has enough samples, passed the texture test and produced points.
    pub survived: bool,
}

/// Output of a pass.
pub struct PassResult {
    /// One point per surviving bin and plane.
    pub points: Vec<BinPoint>,
    /// What happened to each bin.
    pub bins: [BinInfo; NUM_BINS],
    /// Per window: kept by the selection in a surviving, unclipped bin (used by the row-noise and
    /// filtered-file steps).
    pub smooth: Vec<bool>,
    /// Number of windows kept (all surviving bins).
    pub windows_kept: usize,
}

fn mean_f32(it: impl Iterator<Item = f32>) -> f64 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for v in it {
        s += f64::from(v);
        n += 1;
    }
    if n == 0 { 0.0 } else { s / n as f64 }
}

/// Runs one measurement pass over the windows.
pub fn measure_pass(wins: &[Window], p: &PassParams<'_>) -> PassResult {
    let cfg = p.cfg;
    let mut scale = vec![1.0f32; wins.len()];
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); NUM_BINS];
    for (i, w) in wins.iter().enumerate() {
        if let Some(b) = p.layout.bin_of(f64::from(w.mean)) {
            members[b].push(i as u32);
            if let Some(m) = p.norm_model {
                scale[i] =
                    (m.pixel_variance(f64::from(w.mean), w.plane as usize) as f32).max(1e-14);
            }
        }
    }
    let mut result = PassResult {
        points: Vec::new(),
        bins: [BinInfo::default(); NUM_BINS],
        smooth: vec![false; wins.len()],
        windows_kept: 0,
    };
    for (bin, idxs) in members.iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let bin_mean = mean_f32(idxs.iter().map(|&i| wins[i as usize].mean));
        // A bin is a clipped bin when it is bright, or when the previous fit says its noise reaches
        // the clip level (a noisy exposure clips well below the nominal threshold).
        let reaches_clip = p.norm_model.is_some_and(|m| {
            bin_mean + CLIP_REACH_SIGMAS * m.pixel_variance(bin_mean, 1).sqrt() > cfg.clip_level
        });
        let clip_bin = cfg.clip_aware && (bin_mean > cfg.clip_bin_threshold || reaches_clip);
        result.bins[bin].clipped = clip_bin;
        if clip_bin && !p.include_clipped {
            continue;
        }
        let keep = if clip_bin {
            cfg.clip_keep_fraction
        } else {
            cfg.keep_fraction
        };
        let mut energies: Vec<f32> = idxs
            .iter()
            .map(|&i| wins[i as usize].energy / scale[i as usize])
            .collect();
        let threshold = quantile(&mut energies, keep);
        let kept: Vec<u32> = idxs
            .iter()
            .copied()
            .filter(|&i| wins[i as usize].energy / scale[i as usize] <= threshold)
            .collect();
        let samples = kept.len() * COEFFS;
        result.bins[bin].samples = samples as u32;
        if samples < cfg.min_samples {
            continue;
        }
        if !clip_bin {
            // Texture test on the pooled normalised coefficients.
            let mut abs: Vec<f32> = kept
                .iter()
                .flat_map(|&i| {
                    let inv = 1.0 / scale[i as usize].sqrt();
                    wins[i as usize].d.iter().map(move |d| (d * inv).abs())
                })
                .collect();
            let q75 = f64::from(quantile(&mut abs.clone(), 0.75));
            let q50 = f64::from(median(&mut abs));
            let ratio = if q50 > 0.0 { q75 / q50 } else { f64::INFINITY };
            result.bins[bin].tail_ratio = ratio;
            let limit =
                GAUSSIAN_TAIL_RATIO * (1.0 + cfg.tail_tolerance + 4.0 / (samples as f64).sqrt());
            if ratio > limit {
                result.bins[bin].textured = true;
                continue;
            }
        }
        let mut produced = false;
        for plane in 0..4u8 {
            let sel: Vec<u32> = kept
                .iter()
                .copied()
                .filter(|&i| wins[i as usize].plane == plane)
                .collect();
            let n_c = sel.len() * COEFFS;
            if n_c < cfg.min_plane_samples {
                continue;
            }
            let mean = mean_f32(sel.iter().map(|&i| wins[i as usize].mean));
            let q2 = mean_f32(sel.iter().map(|&i| wins[i as usize].q2));
            if !clip_bin {
                let mut dn: Vec<f32> = sel
                    .iter()
                    .flat_map(|&i| {
                        let inv = 1.0 / scale[i as usize].sqrt();
                        wins[i as usize].d.iter().map(move |d| d * inv)
                    })
                    .collect();
                let factor = mad_variance(&mut dn);
                let ref_var = mean_f32(sel.iter().map(|&i| scale[i as usize]));
                let var = factor * ref_var - q2 / 12.0;
                if var > 0.0 {
                    result.points.push(BinPoint {
                        plane: Some(plane),
                        bin: bin as u8,
                        kind: PointKind::Mad,
                        mean,
                        var,
                        count: n_c as u32,
                        rel_var: MAD_REL_VAR / n_c as f64,
                    });
                    produced = true;
                }
            } else if let Some(point) = clipped_point(wins, &sel, plane, bin, mean, q2, p) {
                result.points.push(point);
                produced = true;
            }
        }
        result.bins[bin].survived = produced;
        if produced {
            result.windows_kept += kept.len();
            if !clip_bin {
                for &i in &kept {
                    result.smooth[i as usize] = true;
                }
            }
        }
    }
    result
}

/// The point of a clipped bin and plane: sample variance of the coefficients, mapped back to the
/// unclipped `(mu, sigma^2)` pair.
fn clipped_point(
    wins: &[Window],
    sel: &[u32],
    plane: u8,
    bin: usize,
    mean: f64,
    q2: f64,
    p: &PassParams<'_>,
) -> Option<BinPoint> {
    let clip_level = p.cfg.clip_level;
    let (mut ss, mut n) = (0.0f64, 0usize);
    for &i in sel {
        let w = &wins[i as usize];
        if let Some(m) = p.ref_model {
            let limit = OUTLIER_SIGMAS * m.pixel_variance(f64::from(w.mean), plane as usize).sqrt();
            if w.d.iter().any(|d| f64::from(d.abs()) > limit) {
                continue;
            }
        }
        ss += w.d.iter().map(|&d| f64::from(d).powi(2)).sum::<f64>();
        n += COEFFS;
    }
    if n < p.cfg.min_plane_samples {
        return None;
    }
    let observed_var = ss / n as f64;
    let (mu, sigma2) = clip::invert(mean, observed_var, clip_level)?;
    let var = sigma2 - q2 / 12.0;
    if var <= 0.0 {
        return None;
    }
    let u = (clip_level - mu) / sigma2.sqrt();
    let g = clip::g(u).clamp(0.1, 1.0);
    Some(BinPoint {
        plane: Some(plane),
        bin: bin as u8,
        kind: PointKind::ClippedGaussian,
        mean: mu,
        var,
        count: n as u32,
        rel_var: MOMENT_REL_VAR / (n as f64 * g),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_tail_ratio_constant() {
        assert!((GAUSSIAN_TAIL_RATIO - 1.1503494 / 0.67448975).abs() < 1e-5);
        assert!((GAUSSIAN_TAIL_RATIO - 1.70554).abs() < 1e-3);
    }
}
