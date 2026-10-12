//! Texture fallback: a patch-PCA noise estimate for busy scenes (`docs/06-noise.md` step 6,
//! after Chen et al. 2015).
//!
//! For patches that contain only weak texture, the covariance of the patch vectors has a signal
//! part of low rank and a noise part `sigma^2 I`, so the smallest eigenvalue is the noise variance.
//! The method of Chen et al. selects "weakly textured" patches with a confidence test on the patch
//! variance and takes the smallest eigenvalue of their covariance, iterating the threshold with
//! the running estimate. This module implements that idea from the paper's description, with one
//! correction of our own: the smallest eigenvalue of a sample covariance of `N` noise patches in
//! `p` dimensions sits at the lower Marchenko-Pastur edge `sigma^2 (1 - sqrt(p / N))^2`, so the
//! estimate divides by `(1 - sqrt(p' / N))^2`, where `p'` is `p` minus the number of eigenvalues
//! that stick out of the noise bulk.
//!
//! The patches are first mapped through a generalised Anscombe transform with the prior gain, so
//! that the noise is close to homoscedastic inside a patch and across the patches of a bin; the
//! variance found in that domain is then mapped back with the local derivative of the transform.

use rayon::prelude::*;

use super::bins::{BinLayout, NUM_BINS};
use super::tiles::{TileData, TilePlan};

/// Patch side in plane pixels.
pub const PATCH: usize = 7;

/// Patch dimension.
pub const DIM: usize = PATCH * PATCH;

/// Fewest patches a bin needs for an estimate (three per dimension).
pub const MIN_PATCHES: usize = 3 * DIM;

/// Chen et al. confidence level of the patch-variance test.
const CONFIDENCE_Z: f64 = 3.090_232; // standard normal quantile at 0.999

/// Eigenvalues of a symmetric `n x n` matrix, ascending, by Householder tridiagonalisation and the
/// implicit QL algorithm. `a` is overwritten. Returns `None` if QL does not converge.
// The index arithmetic mirrors the textbook algorithm; iterator forms would obscure it.
#[allow(clippy::needless_range_loop)]
pub fn symmetric_eigenvalues(a: &mut [Vec<f64>]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    // Householder reduction to tridiagonal form (eigenvalues only).
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| a[i][k].abs()).sum();
            if scale == 0.0 {
                e[i] = a[i][l];
            } else {
                for k in 0..=l {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                let f = a[i][l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[i][l] = f - g;
                let mut f_acc = 0.0;
                for j in 0..=l {
                    let mut g2 = 0.0;
                    for k in 0..=j {
                        g2 += a[j][k] * a[i][k];
                    }
                    for k in j + 1..=l {
                        g2 += a[k][j] * a[i][k];
                    }
                    e[j] = g2 / h;
                    f_acc += e[j] * a[i][j];
                }
                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f = a[i][j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[j][k] -= f * e[k] + g * a[i][k];
                    }
                }
            }
        } else {
            e[i] = a[i][l];
        }
    }
    e[0] = 0.0;
    for i in 0..n {
        d[i] = a[i][i];
    }
    // Implicit QL.
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 80 {
                return None;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut underflow = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d.sort_by(f64::total_cmp);
    Some(d)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Upper `p`-quantile of the chi-square distribution with `k` degrees of freedom
/// (Wilson-Hilferty approximation, accurate to a fraction of a percent for `k >= 30`).
fn chi2_quantile(k: f64, z: f64) -> f64 {
    let t = 2.0 / (9.0 * k);
    k * (1.0 - t + z * t.sqrt()).powi(3)
}

/// Result of [`chen_variance`].
#[derive(Clone, Copy, Debug)]
pub struct PcaEstimate {
    /// Estimated noise variance of one value of the patches.
    pub variance: f64,
    /// Patches that passed the confidence test in the last iteration.
    pub patches_used: usize,
    /// Number of eigenvalues above the noise bulk in the last iteration (signal rank).
    pub signal_rank: usize,
    /// False when too few patches passed the confidence test and the estimate is taken from the
    /// least textured patches instead; it then over-estimates by an unknown amount.
    pub confident: bool,
}

fn covariance_eigenvalues(patches: &[f32], selected: &[usize]) -> Option<Vec<f64>> {
    let n = selected.len();
    let mut mean = [0.0f64; DIM];
    for &i in selected {
        for (m, &x) in mean.iter_mut().zip(&patches[i * DIM..(i + 1) * DIM]) {
            *m += f64::from(x);
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mean32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
    // Accumulate the upper triangle in f32 with centred data (cheap and accurate enough for a
    // percent-level estimate); the sums are re-expanded to f64 for the eigenvalue solver.
    let mut cov = vec![0.0f32; DIM * DIM];
    let mut x = [0.0f32; DIM];
    for &i in selected {
        for (xv, (&p, &m)) in x
            .iter_mut()
            .zip(patches[i * DIM..(i + 1) * DIM].iter().zip(&mean32))
        {
            *xv = p - m;
        }
        for r in 0..DIM {
            let xr = x[r];
            let row = &mut cov[r * DIM + r..(r + 1) * DIM];
            for (c, &xc) in row.iter_mut().zip(&x[r..]) {
                *c += xr * xc;
            }
        }
    }
    let norm = 1.0 / (n as f64 - 1.0);
    let mut a = vec![vec![0.0f64; DIM]; DIM];
    for r in 0..DIM {
        for c in r..DIM {
            let v = f64::from(cov[r * DIM + c]) * norm;
            a[r][c] = v;
            a[c][r] = v;
        }
    }
    symmetric_eigenvalues(&mut a)
}

/// Chen-style noise variance from patch vectors (`patches.len() / DIM` rows of `DIM` values).
///
/// The noise is assumed white and homoscedastic over the patches (apply the variance-stabilising
/// transform first). Returns `None` with fewer than [`MIN_PATCHES`] usable patches.
pub fn chen_variance(patches: &[f32]) -> Option<PcaEstimate> {
    let total = patches.len() / DIM;
    if total < MIN_PATCHES {
        return None;
    }
    // Patch variances (mean squared deviation about the patch mean).
    let patch_var: Vec<f32> = patches
        .chunks_exact(DIM)
        .map(|p| {
            let m = p.iter().sum::<f32>() / DIM as f32;
            p.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / DIM as f32
        })
        .collect();
    // Confidence threshold for a patch variance of pure noise: s2 ~ sigma^2 chi2_{p-1} / p.
    let tau = chi2_quantile(DIM as f64 - 1.0, CONFIDENCE_Z) / DIM as f64;
    let mut selected: Vec<usize> = (0..total).collect();
    let mut sigma2 = f64::NAN;
    let mut rank = 0usize;
    let mut confident = true;
    let mut estimate = f64::NAN;
    for _ in 0..MAX_ITERATIONS {
        let eig = covariance_eigenvalues(patches, &selected)?;
        let n = selected.len() as f64;
        // Signal rank: eigenvalues above the upper edge of the noise bulk of the previous estimate.
        let ref_var = if sigma2.is_finite() {
            sigma2
        } else {
            eig[0].max(1e-12)
        };
        let mut p_eff = DIM;
        for _ in 0..3 {
            let edge = ref_var * (1.0 + (p_eff as f64 / n).sqrt()).powi(2) * 1.25;
            rank = eig.iter().filter(|&&l| l > edge).count();
            p_eff = (DIM - rank).max(1);
        }
        let mp = (1.0 - (p_eff as f64 / n).sqrt()).powi(2).max(0.05);
        let est = eig[0].max(0.0) / mp;
        let converged = sigma2.is_finite() && (est / sigma2 - 1.0).abs() < 0.01;
        sigma2 = est;
        estimate = est;
        if converged {
            break;
        }
        let thr = (sigma2 * tau) as f32;
        let mut next: Vec<usize> = (0..total).filter(|&i| patch_var[i] <= thr).collect();
        if next.len() < MIN_PATCHES {
            // Texture everywhere: take the least textured patches, and say so.
            confident = false;
            let mut order: Vec<usize> = (0..total).collect();
            order.sort_by(|&a, &b| patch_var[a].total_cmp(&patch_var[b]));
            order.truncate((2 * MIN_PATCHES).min(total));
            order.sort_unstable();
            next = order;
        } else {
            confident = true;
        }
        if next == selected {
            break;
        }
        selected = next;
    }
    Some(PcaEstimate {
        variance: estimate,
        patches_used: selected.len(),
        signal_rank: rank,
        confident,
    })
}

/// Chen iterations at most (the estimate usually settles in three).
const MAX_ITERATIONS: usize = 6;

/// One bin's fallback estimate.
#[derive(Clone, Copy, Debug)]
pub struct PcaPoint {
    /// Intensity bin.
    pub bin: usize,
    /// Mean of the patch means in the bin.
    pub mean: f64,
    /// Noise variance of one photosite at that mean (raw domain, before removing `q^2/12`).
    pub var: f64,
    /// Patches that passed the confidence test.
    pub patches_used: usize,
    /// False when the estimate is an upper bound from the least textured patches.
    pub confident: bool,
}

/// Settings of the fallback.
#[derive(Clone, Copy, Debug)]
pub struct PcaConfig {
    /// Gain of the variance-stabilising transform.
    pub a: f64,
    /// Read variance of the variance-stabilising transform.
    pub b: f64,
    /// Patches kept per bin and tile.
    pub patches_per_bin_tile: usize,
    /// Step between candidate patches, in plane pixels.
    pub stride: usize,
    /// Bit `k` set = collect patches for intensity bin `k` (all bins: `u32::MAX`).
    pub bin_mask: u32,
}

/// Collects patches of the tiles per intensity bin and estimates the noise variance of each bin.
pub fn texture_points(
    tiles: &[TileData],
    plan: &TilePlan,
    layout: &BinLayout,
    cfg: &PcaConfig,
) -> Vec<PcaPoint> {
    let size = plan.size;
    let half = size / 2;
    let c = 0.375 * cfg.a * cfg.a + cfg.b;
    let scale = 2.0 / cfg.a;
    // Gather: per tile, per bin, a flat list of VST patches and their raw means.
    let wanted = cfg.bin_mask & ((1u32 << NUM_BINS) - 1);
    let per_tile: Vec<Vec<(Vec<f32>, Vec<f32>)>> = tiles
        .par_iter()
        .map(|tile| {
            let mut bins: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); NUM_BINS];
            let positions = (half - PATCH) / cfg.stride + 1;
            // Visit (plane, row) pairs in a scrambled but deterministic order, columns in order, so
            // the cap does not favour the top-left of the tile and the reads stay local.
            let lines = 4 * positions;
            let mut step = (2_654_435_761usize % lines.max(1)) | 1;
            while gcd(step, lines.max(1)) != 1 {
                step += 2;
            }
            let mut full = 0u32;
            let mut line = 0usize;
            'scan: for _ in 0..lines {
                line = (line + step) % lines;
                let (plane, i0) = (line / positions, line % positions * cfg.stride);
                let (py, px) = (plane >> 1, plane & 1);
                for jj in 0..positions {
                    let j0 = jj * cfg.stride;
                    let mut sum = 0.0f32;
                    for u in 0..PATCH {
                        let row = (2 * (i0 + u) + py) * size + 2 * j0 + px;
                        for v in 0..PATCH {
                            sum += tile.pixels[row + 2 * v];
                        }
                    }
                    let mean = sum / DIM as f32;
                    let Some(bin) = layout
                        .bin_of(f64::from(mean))
                        .filter(|b| wanted >> b & 1 == 1)
                    else {
                        continue;
                    };
                    let (patches, means) = &mut bins[bin];
                    if means.len() >= cfg.patches_per_bin_tile {
                        continue;
                    }
                    means.push(mean);
                    for u in 0..PATCH {
                        let row = (2 * (i0 + u) + py) * size + 2 * j0 + px;
                        for v in 0..PATCH {
                            let x = f64::from(tile.pixels[row + 2 * v]);
                            patches.push((scale * (cfg.a * x + c).max(0.0).sqrt()) as f32);
                        }
                    }
                    if means.len() >= cfg.patches_per_bin_tile {
                        full |= 1 << bin;
                        if full & wanted == wanted {
                            break 'scan;
                        }
                    }
                }
            }
            bins
        })
        .collect();
    let mut points: Vec<PcaPoint> = (0..NUM_BINS)
        .into_par_iter()
        .filter_map(|bin| {
            let mut patches: Vec<f32> = Vec::new();
            let mut means: Vec<f32> = Vec::new();
            for t in &per_tile {
                patches.extend_from_slice(&t[bin].0);
                means.extend_from_slice(&t[bin].1);
            }
            let est = chen_variance(&patches)?;
            let mean = means.iter().map(|&m| f64::from(m)).sum::<f64>() / means.len() as f64;
            // Back to the raw domain: Var_raw = Var_vst * (a mu + c).
            let var = est.variance * (cfg.a * mean.max(0.0) + c);
            Some(PcaPoint {
                bin,
                mean,
                var,
                patches_used: est.patches_used,
                confident: est.confident,
            })
        })
        .collect();
    points.sort_by_key(|p| p.bin);
    points
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::model::synth::SynthRng;

    /// Reference eigenvalues by cyclic Jacobi rotations.
    fn jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
        let n = a.len();
        for _ in 0..100 {
            let mut off = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off += a[i][j] * a[i][j];
                    }
                }
            }
            if off < 1e-24 {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    if a[p][q].abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let (akp, akq) = (a[k][p], a[k][q]);
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let (apk, aqk) = (a[p][k], a[q][k]);
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                }
            }
        }
        let mut d: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        d.sort_by(f64::total_cmp);
        d
    }

    #[test]
    fn eigenvalues_match_jacobi_and_known_matrices() {
        // Diagonal and a 2 x 2 with known eigenvalues.
        let mut m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let e = symmetric_eigenvalues(&mut m).unwrap();
        assert!((e[0] - 1.0).abs() < 1e-12 && (e[1] - 3.0).abs() < 1e-12);
        // Random symmetric matrices.
        let mut rng = SynthRng::new(5);
        for n in [3usize, 10, 49] {
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in i..n {
                    let v = rng.normal();
                    a[i][j] = v;
                    a[j][i] = v;
                }
            }
            let want = jacobi(a.clone());
            let got = symmetric_eigenvalues(&mut a).unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-9 * (1.0 + w.abs()), "n={n}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn chi2_quantile_is_close_to_tables() {
        // chi2(48) upper 0.001 point is 84.04 (tables).
        let q = chi2_quantile(48.0, CONFIDENCE_Z);
        assert!((q / 84.04 - 1.0).abs() < 0.01, "{q}");
    }

    fn noise_patches(n: usize, sigma: f64, rng: &mut SynthRng) -> Vec<f32> {
        (0..n * DIM)
            .map(|_| (sigma * rng.normal()) as f32)
            .collect()
    }

    #[test]
    fn pure_noise_variance_is_unbiased_after_the_edge_correction() {
        let mut rng = SynthRng::new(8);
        for &n in &[600usize, 2000, 6000] {
            let mut errs = Vec::new();
            for _ in 0..6 {
                let p = noise_patches(n, 1.0, &mut rng);
                let est = chen_variance(&p).unwrap();
                errs.push(est.variance - 1.0);
            }
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            assert!(mean.abs() < 0.06, "n={n}: bias {mean}");
        }
    }

    #[test]
    fn weak_texture_is_separated_from_noise() {
        // Patches = random low-rank texture (rank 5, strong) + unit noise. Without the PCA the
        // patch variance would be far above 1.
        let mut rng = SynthRng::new(9);
        let n = 4000;
        let basis: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..DIM).map(|_| rng.normal() * 0.5).collect())
            .collect();
        let mut p = Vec::with_capacity(n * DIM);
        for _ in 0..n {
            let coeff: Vec<f64> = (0..5).map(|_| 0.5 * rng.normal()).collect();
            for k in 0..DIM {
                let s: f64 = (0..5).map(|r| coeff[r] * basis[r][k]).sum();
                p.push((s + rng.normal()) as f32);
            }
        }
        let est = chen_variance(&p).unwrap();
        assert!(
            (est.variance - 1.0).abs() < 0.12,
            "variance {}",
            est.variance
        );
        assert!(est.signal_rank >= 4, "rank {}", est.signal_rank);
        assert!(est.confident);
    }

    #[test]
    fn strong_texture_everywhere_gives_a_flagged_upper_bound() {
        // Texture of rank 5 that is ten times the noise: no patch passes the confidence test.
        let mut rng = SynthRng::new(10);
        let n = 3000;
        let basis: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..DIM).map(|_| rng.normal() * 0.5).collect())
            .collect();
        let mut p = Vec::with_capacity(n * DIM);
        for _ in 0..n {
            let coeff: Vec<f64> = (0..5).map(|_| 3.0 * rng.normal()).collect();
            for k in 0..DIM {
                let s: f64 = (0..5).map(|r| coeff[r] * basis[r][k]).sum();
                p.push((s + rng.normal()) as f32);
            }
        }
        let est = chen_variance(&p).unwrap();
        assert!(!est.confident);
        // Still close to the truth because the texture is low-rank, never below it by much.
        assert!(
            est.variance > 0.85 && est.variance < 2.5,
            "variance {}",
            est.variance
        );
    }

    #[test]
    fn too_few_patches_give_none() {
        let mut rng = SynthRng::new(1);
        assert!(chen_variance(&noise_patches(MIN_PATCHES - 1, 1.0, &mut rng)).is_none());
    }
}
