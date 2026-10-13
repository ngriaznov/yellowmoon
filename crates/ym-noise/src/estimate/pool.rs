//! Batch pooling (`docs/06-noise.md` section Priors, "Batch pooling").
//!
//! Within a session, files from the same model, ISO, bit depth and compression share a pooled
//! estimate: the variance points of every file in the group are merged (weighted by their sample
//! counts) and the model is refitted, so each file benefits from the intensity range covered by its
//! siblings. Nothing is written to disk; the pool lives in memory and is updated as files finish
//! decoding. Row noise is pooled the same way (weighted by the number of row segments); the
//! filtered-file factor is a property of one exposure and is applied per file, never pooled.

use std::collections::HashMap;

use super::{BinPoint, Confidence, NoiseEstimate, PointKind, refit};
use crate::priors::NoisePrior;

/// Files that share these fields share a pooled estimate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolKey {
    /// Camera model string (or any stable id of the body).
    pub model: String,
    /// ISO from the maker note.
    pub iso: u32,
    /// Bit depth of the recorded values.
    pub bit_depth: u8,
    /// Compression mode, as any stable string such as `"lossless"` or `"craw"`.
    pub compression: String,
}

#[derive(Clone, Debug, Default)]
struct Entry {
    points: Vec<BinPoint>,
    files: usize,
    q_sum: f64,
    row_var_sum: f64,
    row_weight: f64,
}

/// In-memory pool of variance points per [`PoolKey`].
#[derive(Clone, Debug, Default)]
pub struct BatchPool {
    entries: HashMap<PoolKey, Entry>,
}

/// Merges two points of the same (plane, bin, kind).
///
/// The mean and the variance are the means weighted by the counts (docs/06 "weighted by bin
/// counts"), and the counts add. **The relative variance is that of the better member, not the
/// `1 / N` of independent samples.** Files of one session do not give independent samples of a
/// bin: the scatter of a bin about the noise line comes largely from what the scene does to it (the
/// position of the bin's mean, the gradients the Haar coefficients still see, the little texture
/// the selection lets through), and a second frame of the same scene repeats that scatter instead
/// of averaging it away. The first version divided the relative variance by the number of files;
/// three frames of one ramp scene then had bins that looked three times as precise as their
/// scatter allowed, every point sat several nominal errors from its line, and the robust refit
/// threw them all away and fell back to the prior (1.77e-4 for a true 5e-4, with the members' own
/// fits at 5.0 to 5.1e-4). Pooling averages the bins and widens the range they cover; it does not
/// tighten them. As a consequence pooling copies of one estimate returns that estimate.
fn merge(a: &BinPoint, b: &BinPoint) -> BinPoint {
    let (na, nb) = (f64::from(a.count), f64::from(b.count));
    let n = na + nb;
    if n <= 0.0 {
        return *a;
    }
    BinPoint {
        plane: a.plane,
        bin: a.bin,
        kind: a.kind,
        // The weighted means in the form `a + (b - a) * w`: two equal points give exactly that
        // point (`(na * x + nb * x) / n` can miss `x` by a rounding step), so pooling copies of one
        // estimate hands the fit exactly the points it had.
        mean: a.mean + (b.mean - a.mean) * (nb / n),
        var: a.var + (b.var - a.var) * (nb / n),
        count: (a.count).saturating_add(b.count),
        rel_var: a.rel_var.min(b.rel_var),
    }
}

impl BatchPool {
    /// An empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct groups.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no file has been added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of files pooled under `key`.
    pub fn files(&self, key: &PoolKey) -> usize {
        self.entries.get(key).map_or(0, |e| e.files)
    }

    /// Adds a finished file to its group and returns the pooled estimate for it.
    ///
    /// The returned model is refitted on all points of the group with `prior`; its row noise is the
    /// pooled row noise; the file's own filtered-file factor is applied on top. Returns `None` when
    /// the file carried no points (a prior-only estimate adds nothing to the pool) and the group
    /// has none either.
    pub fn add(
        &mut self,
        key: PoolKey,
        file: &NoiseEstimate,
        prior: &NoisePrior,
    ) -> Option<NoiseEstimate> {
        let entry = self.entries.entry(key).or_default();
        if file.diagnostics.confidence != Confidence::PriorOnly {
            for p in &file.points {
                match entry
                    .points
                    .iter_mut()
                    .find(|q| q.plane == p.plane && q.bin == p.bin && q.kind == p.kind)
                {
                    Some(q) => *q = merge(q, p),
                    None => entry.points.push(*p),
                }
            }
            entry.files += 1;
            entry.q_sum += file.model.q;
            if file.diagnostics.row_noise.measured {
                let w = file.diagnostics.row_noise.rows_used as f64;
                entry.row_var_sum += w * file.diagnostics.row_noise.sigma_row.powi(2);
                entry.row_weight += w;
            }
        }
        if entry.points.is_empty() {
            return None;
        }
        let q = if entry.files > 0 {
            entry.q_sum / entry.files as f64
        } else {
            file.model.q
        };
        let (mut model, residual_rms) = refit(&entry.points, prior, q)?;
        if entry.row_weight > 0.0 {
            model.sigma_row = (entry.row_var_sum / entry.row_weight).sqrt();
        }
        let fallback = entry.points.iter().any(|p| p.kind == PointKind::Pca);
        let fitted = model;
        if file.diagnostics.filtered.flagged {
            model = model.with_sigma_scaled(file.diagnostics.filtered.factor);
        }
        let mut diagnostics = file.diagnostics.clone();
        diagnostics.confidence = if fallback {
            Confidence::DataWithFallback
        } else {
            Confidence::Data
        };
        diagnostics.residual_rms = residual_rms;
        diagnostics.fitted = fitted;
        diagnostics.texture_fallback = fallback;
        let mut bins: Vec<usize> = entry.points.iter().map(|p| p.bin as usize).collect();
        bins.sort_unstable();
        bins.dedup();
        diagnostics.coverage = super::bins::BinLayout::default().coverage(bins);
        Some(NoiseEstimate {
            model,
            points: entry.points.clone(),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::{Diagnostics, FilteredCheck, RowNoiseInfo};
    use crate::model::NoiseModel;
    use crate::priors::PriorSource;

    fn point(plane: u8, bin: u8, mean: f64, a: f64, b: f64, count: u32) -> BinPoint {
        BinPoint {
            plane: Some(plane),
            bin,
            kind: PointKind::Mad,
            mean,
            var: a * mean + b,
            count,
            rel_var: 5.5 / f64::from(count),
        }
    }

    fn file(points: Vec<BinPoint>, rows: usize, sigma_row: f64) -> NoiseEstimate {
        NoiseEstimate {
            model: NoiseModel::new(1e-3, 1e-6),
            points,
            diagnostics: Diagnostics {
                confidence: Confidence::Data,
                bins: Vec::new(),
                bins_surviving: 0,
                bins_textured: 0,
                clipped_bins_used: 0,
                coverage: 0.0,
                texture_fallback: false,
                probe_ratio: None,
                windows_total: 0,
                windows_kept: 0,
                residual_rms: 0.0,
                high_fraction: 0.0,
                passes: 1,
                filtered: FilteredCheck::default(),
                row_noise: RowNoiseInfo {
                    measured: rows > 0,
                    rows_used: rows,
                    sigma_row,
                },
                prior_source: PriorSource::None,
                fitted: NoiseModel::new(1e-3, 1e-6),
            },
        }
    }

    fn key() -> PoolKey {
        PoolKey {
            model: "ILCE-1".into(),
            iso: 800,
            bit_depth: 14,
            compression: "lossless".into(),
        }
    }

    #[test]
    fn pooling_merges_counts_and_widens_coverage() {
        let (a, b) = (2e-3, 3e-7);
        // File 1 sees only dark bins, file 2 only bright ones; neither alone has a lever arm.
        let dark: Vec<BinPoint> = (0..4)
            .flat_map(|c| (1..4).map(move |k| point(c, k, 0.003 * f64::from(k), a, b, 600)))
            .collect();
        let bright: Vec<BinPoint> = (0..4)
            .flat_map(|c| (10..14).map(move |k| point(c, k, 0.05 * f64::from(k), a, b, 900)))
            .collect();
        let mut pool = BatchPool::new();
        let first = pool
            .add(key(), &file(dark, 0, 0.0), &NoisePrior::none())
            .unwrap();
        assert!(first.diagnostics.coverage < 0.1);
        let second = pool
            .add(key(), &file(bright, 0, 0.0), &NoisePrior::none())
            .unwrap();
        assert_eq!(pool.files(&key()), 2);
        assert!(second.diagnostics.coverage > 3.0 * first.diagnostics.coverage);
        assert!(
            (second.model.a / a - 1.0).abs() < 0.01,
            "a {}",
            second.model.a
        );
        assert!(
            (second.model.b / b - 1.0).abs() < 0.05,
            "b {}",
            second.model.b
        );
    }

    #[test]
    fn pooling_the_same_bin_is_a_count_weighted_mean() {
        let p1 = point(0, 5, 0.10, 1e-3, 1e-6, 400);
        let mut p2 = point(0, 5, 0.12, 1e-3, 1e-6, 1200);
        p2.var *= 1.1;
        let m = merge(&p1, &p2);
        assert_eq!(m.count, 1600);
        assert!((m.mean - (400.0 * 0.10 + 1200.0 * 0.12) / 1600.0).abs() < 1e-12);
        assert!((m.var - (400.0 * p1.var + 1200.0 * p2.var) / 1600.0).abs() < 1e-15);
        // The relative variance is the better member's: no 1/N for samples that share a scene.
        assert!((m.rel_var - 5.5 / 1200.0).abs() < 1e-12);
        // Merging is symmetric and associative where it can be: order does not matter.
        let m2 = merge(&p2, &p1);
        assert!((m2.mean - m.mean).abs() < 1e-15 && (m2.var - m.var).abs() < 1e-18);
        assert_eq!(m2.rel_var, m.rel_var);
    }

    #[test]
    fn pooling_copies_of_a_point_changes_nothing_but_the_count() {
        let p = point(2, 9, 0.17, 1e-3, 1e-6, 700);
        let mut m = p;
        for _ in 0..5 {
            m = merge(&m, &p);
        }
        assert_eq!(m.count, 4200);
        // Exactly the point, not "to a rounding step": `(na * x + nb * x) / n` misses 0.17 by one.
        assert_eq!(m.mean, p.mean);
        assert_eq!(m.var, p.var);
        assert_eq!(m.rel_var, p.rel_var);
    }

    #[test]
    fn groups_are_separate_and_row_noise_is_pooled() {
        let pts = |n| {
            (0..4)
                .flat_map(move |c| {
                    (2..8).map(move |k| point(c, k, 0.02 * f64::from(k), 1e-3, 1e-6, n))
                })
                .collect::<Vec<_>>()
        };
        let mut pool = BatchPool::new();
        let mut other = key();
        other.iso = 1600;
        pool.add(key(), &file(pts(500), 200, 1e-3), &NoisePrior::none())
            .unwrap();
        let pooled = pool
            .add(key(), &file(pts(500), 600, 2e-3), &NoisePrior::none())
            .unwrap();
        let want = ((200.0 * 1e-6 + 600.0 * 4e-6) / 800.0f64).sqrt();
        assert!((pooled.model.sigma_row - want).abs() < 1e-9);
        pool.add(other.clone(), &file(pts(500), 0, 0.0), &NoisePrior::none())
            .unwrap();
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.files(&other), 1);
        assert_eq!(pool.files(&key()), 2);
    }

    #[test]
    fn filtered_factor_stays_with_its_file() {
        let pts: Vec<BinPoint> = (0..4)
            .flat_map(|c| (2..8).map(move |k| point(c, k, 0.02 * f64::from(k), 1e-3, 1e-6, 500)))
            .collect();
        let mut pool = BatchPool::new();
        pool.add(key(), &file(pts.clone(), 0, 0.0), &NoisePrior::none())
            .unwrap();
        let mut filtered = file(pts, 0, 0.0);
        filtered.diagnostics.filtered = FilteredCheck {
            evaluated: true,
            windows: 500,
            ratio: 0.6,
            flagged: true,
            factor: 1.5,
        };
        let out = pool.add(key(), &filtered, &NoisePrior::none()).unwrap();
        assert!((out.model.a / out.diagnostics.fitted.a - 2.25).abs() < 1e-9);
        // The pool itself is unscaled: the next unflagged file gets the plain fit.
        let plain = pool
            .add(key(), &file(vec![], 0, 0.0), &NoisePrior::none())
            .unwrap();
        assert!((plain.model.a / plain.diagnostics.fitted.a - 1.0).abs() < 1e-12);
    }
}
