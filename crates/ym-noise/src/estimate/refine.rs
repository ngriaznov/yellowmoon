//! YOND-style refinement (`docs/06-noise.md` step 7).
//!
//! After the first denoise of the preview, the noise can be measured against the denoiser's output
//! instead of against the noisy image alone, which is not limited to smooth regions: the denoised
//! image is clean enough to pick flat windows from, and the residual `noisy - denoised` is the noise
//! the denoiser removed. Per intensity bin and CFA plane,
//!
//! ```text
//! var_noise = var(noisy - denoised) - var(denoised - smooth(denoised))
//! ```
//!
//! The second term estimates what the denoiser leaves in (the high-pass content of the denoised
//! image, which in flat windows is retained noise), so the difference is the variance of the full
//! noise. Both variances are robust (`(1.4826 MAD)^2`). `smooth` is a 3 x 3 box mean on the CFA
//! plane; its high-pass has `8/9` of a white residual's variance, so the second term is scaled by
//! `9/8`. The second term also contains any fine detail the denoiser kept, which is not noise, so
//! it is capped at 15% of the first (a denoiser that keeps more noise than that is not one
//! to refine against). The new points replace the old ones and the model is refitted once with the
//! same prior.
//!
//! `noisy` and `denoised` must be mosaics of the same size and normalisation: if the denoiser
//! produces RGB, the caller samples it back at the CFA positions. The method assumes the retained
//! noise is roughly independent of the removed noise, as in the published description; a denoiser
//! that leaves strongly correlated residual noise will still be under-read.

use rayon::prelude::*;

use super::bins::{BinLayout, NUM_BINS};
use super::stats::{mad_variance, quantile};
use super::tiles::{self, COEFFS, WIN, WIN_PX};
use super::{
    BinPoint, Confidence, EstimateError, EstimatorConfig, FilteredCheck, NoiseEstimate,
    PlaneSource, PointKind, refit,
};
use crate::priors::NoisePrior;

/// Largest share of the residual variance that the retained-noise correction may remove.
const MAX_CORRECTION: f64 = 0.15;

/// Residual values per window: all 36 pixels.
const E_PER_WINDOW: usize = WIN * WIN;

/// High-pass values per window: the 4 x 4 pixels with a full 3 x 3 neighbourhood inside the window.
const H_PER_WINDOW: usize = (WIN - 2) * (WIN - 2);

struct RefWindow {
    plane: u8,
    mean: f32,
    energy: f32,
    e: [f32; E_PER_WINDOW],
    h: [f32; H_PER_WINDOW],
}

fn windows_of_tile(noisy: &[f32], denoised: &[f32], size: usize) -> Vec<RefWindow> {
    let n = size / WIN_PX;
    let mut out = Vec::with_capacity(4 * n * n);
    for plane in 0..4 {
        let (py, px) = (plane >> 1, plane & 1);
        for wi in 0..n {
            for wj in 0..n {
                let mut d = [[0.0f32; WIN]; WIN];
                let mut e = [0.0f32; E_PER_WINDOW];
                for r in 0..WIN {
                    let y = 2 * (WIN * wi + r) + py;
                    let base = y * size + 2 * WIN * wj + px;
                    for c in 0..WIN {
                        let (nv, dv) = (noisy[base + 2 * c], denoised[base + 2 * c]);
                        d[r][c] = dv;
                        e[r * WIN + c] = nv - dv;
                    }
                }
                // Texture energy of the denoised window from its Haar blocks, as in `tiles`.
                let (mut dh, mut dv) = ([0.0f64; COEFFS], [0.0f64; COEFFS]);
                let mut mean = 0.0f64;
                for bi in 0..WIN / 2 {
                    for bj in 0..WIN / 2 {
                        let (a, b) = (
                            f64::from(d[2 * bi][2 * bj]),
                            f64::from(d[2 * bi][2 * bj + 1]),
                        );
                        let (c, dd) = (
                            f64::from(d[2 * bi + 1][2 * bj]),
                            f64::from(d[2 * bi + 1][2 * bj + 1]),
                        );
                        let k = bi * (WIN / 2) + bj;
                        dh[k] = 0.5 * (a - b + c - dd);
                        dv[k] = 0.5 * (a + b - c - dd);
                        mean += 0.25 * (a + b + c + dd);
                    }
                }
                mean /= COEFFS as f64;
                let (hb, vb) = (dh.iter().sum::<f64>() / 9.0, dv.iter().sum::<f64>() / 9.0);
                let energy: f64 = dh.iter().map(|v| (v - hb).powi(2)).sum::<f64>()
                    + dv.iter().map(|v| (v - vb).powi(2)).sum::<f64>();
                let mut h = [0.0f32; H_PER_WINDOW];
                for r in 1..WIN - 1 {
                    for c in 1..WIN - 1 {
                        let s: f32 = d[r - 1..=r + 1]
                            .iter()
                            .map(|row| row[c - 1..=c + 1].iter().sum::<f32>())
                            .sum();
                        h[(r - 1) * (WIN - 2) + (c - 1)] = d[r][c] - s / 9.0;
                    }
                }
                out.push(RefWindow {
                    plane: plane as u8,
                    mean: mean as f32,
                    energy: energy as f32,
                    e,
                    h,
                });
            }
        }
    }
    out
}

/// Re-estimates the noise from a noisy and a denoised version of the same mosaic.
///
/// `prev` is the estimate being refined; its row noise and filtered-file result are carried over
/// (the residual already measures the pixel-level variance, so the filtered-file factor is not
/// applied again). The result's points are the refined points.
///
/// # Errors
///
/// Returns [`EstimateError::BadInput`] when the two mosaics differ in size,
/// [`EstimateError::FrameTooSmall`] for tiny frames, and [`EstimateError::NoUsableBins`] when no bin
/// has enough flat windows.
pub fn refine(
    prev: &NoiseEstimate,
    noisy: &dyn PlaneSource,
    denoised: &dyn PlaneSource,
    prior: &NoisePrior,
    cfg: &EstimatorConfig,
) -> Result<NoiseEstimate, EstimateError> {
    let (w, h) = (noisy.width(), noisy.height());
    if denoised.width() != w || denoised.height() != h {
        return Err(EstimateError::BadInput(
            "the noisy and denoised mosaics differ in size".into(),
        ));
    }
    let plan = tiles::plan(w, h, cfg)?;
    let size = plan.size;
    let layout = BinLayout::default();
    let wins: Vec<RefWindow> = plan
        .origins
        .par_iter()
        .map(|&o| {
            let (mut nb, mut db) = (vec![0.0f32; size * size], vec![0.0f32; size * size]);
            noisy.read_rect(o.0, o.1, size, size, &mut nb);
            denoised.read_rect(o.0, o.1, size, size, &mut db);
            windows_of_tile(&nb, &db, size)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); NUM_BINS];
    for (i, win) in wins.iter().enumerate() {
        if let Some(b) = layout
            .bin_of(f64::from(win.mean))
            .filter(|_| f64::from(win.mean) <= cfg.clip_bin_threshold)
        {
            members[b].push(i);
        }
    }
    let q2 = prev.model.q * prev.model.q / 12.0;
    let mut points = Vec::new();
    for (bin, idxs) in members.iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let mut energies: Vec<f32> = idxs.iter().map(|&i| wins[i].energy).collect();
        let threshold = quantile(&mut energies, cfg.keep_fraction);
        let kept: Vec<usize> = idxs
            .iter()
            .copied()
            .filter(|&i| wins[i].energy <= threshold)
            .collect();
        for plane in 0..4u8 {
            let sel: Vec<usize> = kept
                .iter()
                .copied()
                .filter(|&i| wins[i].plane == plane)
                .collect();
            // 16 high-pass values per window set the sample count.
            let n_h = sel.len() * H_PER_WINDOW;
            if n_h < cfg.min_plane_samples {
                continue;
            }
            let mut e: Vec<f32> = sel.iter().flat_map(|&i| wins[i].e).collect();
            let mut hp: Vec<f32> = sel.iter().flat_map(|&i| wins[i].h).collect();
            let var_e = mad_variance(&mut e);
            // The correction is capped: fine detail in the denoised image raises its high-pass
            // energy just like retained noise does, and subtracting it as noise would overshoot.
            let correction = (1.125 * mad_variance(&mut hp)).min(MAX_CORRECTION * var_e);
            let var = var_e - correction - q2;
            if var <= 0.0 {
                continue;
            }
            let mean = sel.iter().map(|&i| f64::from(wins[i].mean)).sum::<f64>() / sel.len() as f64;
            // The residual values of a window are not independent of each other through the
            // denoiser, so count a quarter of them.
            let n_eff = (sel.len() * E_PER_WINDOW) as f64 / 4.0;
            points.push(BinPoint {
                plane: Some(plane),
                bin: bin as u8,
                kind: PointKind::Refined,
                mean,
                var,
                count: (sel.len() * E_PER_WINDOW) as u32,
                rel_var: super::measure::MAD_REL_VAR / n_eff,
            });
        }
    }
    let (mut model, residual_rms) =
        refit(&points, prior, prev.model.q).ok_or(EstimateError::NoUsableBins)?;
    model.sigma_row = prev.diagnostics.row_noise.sigma_row;
    let fitted = model;
    let mut bins: Vec<usize> = points.iter().map(|p| p.bin as usize).collect();
    bins.sort_unstable();
    bins.dedup();
    let mut diagnostics = prev.diagnostics.clone();
    diagnostics.confidence = Confidence::Data;
    diagnostics.coverage = layout.coverage(bins);
    diagnostics.residual_rms = residual_rms;
    diagnostics.passes += 1;
    diagnostics.fitted = fitted;
    diagnostics.windows_total = wins.len();
    diagnostics.windows_kept = members.iter().map(Vec::len).sum();
    // The residual measures the pixel-level variance, so no further correction for filtering.
    diagnostics.filtered = FilteredCheck {
        factor: 1.0,
        ..prev.diagnostics.filtered
    };
    Ok(NoiseEstimate {
        model,
        points,
        diagnostics,
    })
}
