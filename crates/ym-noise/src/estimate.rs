//! The per-image estimator: smooth-region selection, intensity bins, clip-aware fit and the texture
//! fallback (`docs/06-noise.md` section Estimator).
//!
//! The entry point is [`estimate`]. It reads a 2 MP crop grid (16 tiles) from a normalised mosaic,
//! measures the noise variance in smooth regions per intensity bin and CFA plane, fits the model
//! `(a, b, delta_c)` by weighted least squares with the priors of [`crate::priors`], measures row
//! noise, and checks whether the camera filtered the raw. [`refine`](refine::refine) re-estimates
//! from a noisy and a denoised preview, and [`BatchPool`] pools the statistics of files that share
//! model, ISO, bit depth and compression.
//!
//! Everything is written from the papers' descriptions (Foi et al. 2008, Chen et al. 2015, the
//! YOND refinement idea); no reference code was used. See the module docs of the submodules for the
//! details and for the departures from the papers.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::NoiseModel;
use crate::priors::{NoisePrior, PriorSource};

pub mod bins;
pub mod clip;
pub mod filtered;
pub mod fit;
pub mod measure;
pub mod pca;
pub mod pool;
pub mod refine;
pub mod rownoise;
pub mod source;
pub mod stats;
pub mod tiles;

pub use filtered::FilteredCheck;
pub use measure::BinInfo;
pub use pool::{BatchPool, PoolKey};
pub use refine::refine;
pub use rownoise::RowNoiseInfo;
pub use source::{F32Plane, PlaneSource, U16Mosaic};

use bins::{BinLayout, NUM_BINS};
use fit::{FitPoint, fit_map};
use measure::{PassParams, measure_pass};
use tiles::{TileData, TilePlan, Window};

/// Why an estimate could not be made.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EstimateError {
    /// The frame is too small for the tile grid.
    #[error("the frame ({width} x {height}) is too small to estimate noise from")]
    FrameTooSmall {
        /// Frame width.
        width: usize,
        /// Frame height.
        height: usize,
    },
    /// An input buffer or setting is unusable.
    #[error("unusable noise-estimation input: {0}")]
    BadInput(String),
    /// No intensity bin had usable statistics and there is no prior to fall back on.
    #[error("no smooth region with usable noise statistics was found")]
    NoUsableBins,
}

/// Settings of the estimator. The defaults are the values of `docs/06-noise.md`.
#[derive(Clone, Debug)]
pub struct EstimatorConfig {
    /// Total size of the crop grid in photosites (2 MP).
    pub grid_pixels: usize,
    /// Tiles per axis (4, i.e. 16 tiles).
    pub tiles_per_axis: usize,
    /// Fraction of windows kept per intensity bin, lowest energy first (0.40).
    pub keep_fraction: f64,
    /// Fewest coefficients a bin needs to be used (400).
    pub min_samples: usize,
    /// Fewest coefficients a bin and plane need to give a point of their own (100).
    pub min_plane_samples: usize,
    /// Use the clipped-Gaussian treatment for bins near white (false = treat them like any bin;
    /// kept only to measure what clipping does without it).
    pub clip_aware: bool,
    /// Bins with a mean above this are clipped bins (0.85).
    pub clip_bin_threshold: f64,
    /// Fraction of windows kept in clipped bins (0.70; selecting hard on energy would favour the
    /// most clipped windows).
    pub clip_keep_fraction: f64,
    /// Normalised level at which the sensor clips (1.0).
    pub clip_level: f64,
    /// Stabilised passes after the first (1; a second changes the estimates by less than 0.1%).
    pub stabilise_iterations: usize,
    /// Fewer surviving bins than this trigger the texture fallback (6).
    pub min_bins_for_fit: usize,
    /// The texture fallback also runs when this fraction of the points sits well above the fitted
    /// line, which is what texture that the tail test cannot see looks like (0.15).
    pub max_high_fraction: f64,
    /// Run a cheap patch-PCA probe on the best-populated bins of every file; the fallback runs
    /// when the probe finds the noise clearly below the smooth-region fit (texture the other tests
    /// cannot see).
    pub texture_probe: bool,
    /// The probe triggers the fallback when the median ratio of its variance to the fit is below
    /// this (0.95; clean images give 1.05 to 1.1).
    pub probe_ratio_limit: f64,
    /// ... but only down to this ratio (0.5), and fallback points below it times the fit are
    /// dropped. Texture can inflate a smooth-region estimate by a factor of two at most before the
    /// scene is hopeless anyway; a patch-PCA variance far below the fit (0.04 to 0.4 for filtered
    /// noise) is spatially correlated noise, which PCA reads as nearly noise-free and which the
    /// filtered-file check handles instead.
    pub probe_ratio_floor: f64,
    /// Allow the texture fallback.
    pub texture_fallback: bool,
    /// Slack of the Gaussian tail test of a bin (0.25).
    pub tail_tolerance: f64,
    /// Measure row noise.
    pub row_noise: bool,
    /// Fewest smooth photosites in a row segment (24).
    pub row_min_samples: usize,
    /// Run the filtered-file check.
    pub filtered_check: bool,
    /// Binned-over-pixel residual variance ratio above which a file counts as filtered (0.4).
    pub filtered_ratio_threshold: f64,
    /// Smallest sigma factor applied to a filtered file (1.3).
    pub filtered_min_factor: f64,
    /// Largest sigma factor applied to a filtered file (4).
    pub filtered_max_factor: f64,
    /// Patches kept per intensity bin and tile by the texture fallback.
    pub pca_patches_per_bin_tile: usize,
    /// Step between candidate patches of the fallback, in plane pixels.
    pub pca_stride: usize,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            grid_pixels: 2_000_000,
            tiles_per_axis: 4,
            keep_fraction: 0.40,
            min_samples: 400,
            min_plane_samples: 100,
            clip_aware: true,
            clip_bin_threshold: 0.85,
            clip_keep_fraction: 0.70,
            clip_level: 1.0,
            stabilise_iterations: 1,
            min_bins_for_fit: 6,
            max_high_fraction: 0.15,
            texture_probe: true,
            probe_ratio_limit: 0.95,
            probe_ratio_floor: 0.5,
            texture_fallback: true,
            tail_tolerance: 0.25,
            row_noise: true,
            row_min_samples: 24,
            filtered_check: true,
            filtered_ratio_threshold: 0.4,
            filtered_min_factor: 1.3,
            filtered_max_factor: 4.0,
            pca_patches_per_bin_tile: 160,
            pca_stride: 3,
        }
    }
}

/// Input of [`estimate`].
pub struct EstimateInput<'a> {
    /// The normalised mosaic (after R1, before white balance).
    pub mosaic: &'a dyn PlaneSource,
    /// Per-photosite quantisation steps in normalised units (cRAW and ARW 6.0), if the decoder
    /// supplied them; the same size as `mosaic`.
    pub qstep: Option<&'a dyn PlaneSource>,
    /// Quantisation step used when there is no `qstep` map: one DN in normalised units,
    /// `1 / (white - black)`.
    pub q_norm: f64,
}

/// Where a variance point came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointKind {
    /// MAD of the Haar detail coefficients of smooth windows.
    Mad,
    /// Sample variance of a clipped bin mapped back through the clipped-Gaussian moments.
    ClippedGaussian,
    /// Patch-PCA estimate of the texture fallback.
    Pca,
    /// Residual-based estimate of the refinement.
    Refined,
}

/// One variance measurement behind the fit; kept so files can be pooled and refits can be made.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinPoint {
    /// CFA plane, or `None` when pooled over the four planes.
    pub plane: Option<u8>,
    /// Intensity bin.
    pub bin: u8,
    /// How the point was measured.
    pub kind: PointKind,
    /// Mean intensity of the group.
    pub mean: f64,
    /// Photosite variance at that mean with the quantisation term removed.
    pub var: f64,
    /// Number of coefficients (or patches) behind the point.
    pub count: u32,
    /// Relative variance of `var`.
    pub rel_var: f64,
}

impl BinPoint {
    fn fit_point(&self) -> FitPoint {
        FitPoint {
            plane: self.plane,
            mean: self.mean,
            var: self.var,
            rel_var: self.rel_var,
        }
    }
}

/// How much the model rests on the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// Enough smooth bins survived.
    Data,
    /// Too few smooth bins survived; the texture fallback supplied points.
    DataWithFallback,
    /// Nothing usable in the image; the model is the centre of the prior.
    PriorOnly,
}

/// What the estimator saw, for the UI, tests and `public_priors.py`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostics {
    /// How much the model rests on the image.
    pub confidence: Confidence,
    /// What happened to each intensity bin in the last pass.
    pub bins: Vec<BinInfo>,
    /// Bins that survived the sample count and texture test (MAD and clipped points).
    pub bins_surviving: usize,
    /// Bins dropped by the texture test.
    pub bins_textured: usize,
    /// Bins fitted with the clipped-Gaussian treatment.
    pub clipped_bins_used: usize,
    /// Linear span of the bins that produced points, as a fraction of [0, 0.98].
    pub coverage: f64,
    /// True when the texture fallback ran.
    pub texture_fallback: bool,
    /// Median ratio of the patch-PCA probe to the smooth-region fit (`None` if the probe did not run).
    pub probe_ratio: Option<f64>,
    /// Windows read and windows kept by the selection.
    pub windows_total: usize,
    /// Windows kept by the selection.
    pub windows_kept: usize,
    /// Relative RMS residual of the fit.
    pub residual_rms: f64,
    /// Fraction of the points far above the fitted line (texture that survived the selection).
    pub high_fraction: f64,
    /// Passes used.
    pub passes: usize,
    /// Filtered-file check.
    pub filtered: FilteredCheck,
    /// Row-noise measurement.
    pub row_noise: RowNoiseInfo,
    /// Where the prior came from.
    pub prior_source: PriorSource,
    /// The fitted model before row noise and the filtered-file factor were applied.
    pub fitted: NoiseModel,
}

/// Result of [`estimate`] and [`refine()`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseEstimate {
    /// The model to use: fitted, with row noise measured and sigma raised for filtered files.
    pub model: NoiseModel,
    /// The variance points behind the fit (unscaled).
    pub points: Vec<BinPoint>,
    /// What the estimator saw.
    pub diagnostics: Diagnostics,
}

/// Relative variance assigned to a patch-PCA point, divided by the patches used. Larger than the
/// sampling error alone: the bin-to-bin scatter of the PCA estimate on busy synthetic scenes is
/// about 7% for 2000 patches.
const PCA_REL_VAR: f64 = 10.0;

/// A model standing for the prior, used to reject outliers before there is a fit.
fn model_from_prior(prior: &NoisePrior) -> Option<NoiseModel> {
    let a = prior.a_center()?;
    Some(NoiseModel::new(a, prior.b_center().unwrap_or(0.0)))
}

fn fit_to_model(f: &fit::FitOutput, q: f64) -> NoiseModel {
    NoiseModel {
        a: f.a,
        b: f.b,
        delta: f.delta,
        sigma_row: 0.0,
        q,
    }
}

fn fit_points(points: &[BinPoint]) -> Vec<FitPoint> {
    points.iter().map(BinPoint::fit_point).collect()
}

/// Wall-clock time of the stages of one estimate, in milliseconds (for benchmarks; not part of
/// the result, which must stay deterministic).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// Reading and normalising the 16 tiles.
    pub read_ms: f64,
    /// Haar coefficients and window statistics.
    pub windows_ms: f64,
    /// Selection, bin statistics and fits (all passes).
    pub passes_ms: f64,
    /// Texture fallback, if it ran.
    pub fallback_ms: f64,
    /// Row noise and the filtered-file check.
    pub extras_ms: f64,
}

/// Estimates the noise model of one image.
///
/// `prior` comes from [`crate::priors::PriorStore::prior_for`] or is [`NoisePrior::none`] for the
/// prior-free estimate that `training/noise/public_priors.py` collects.
///
/// # Errors
///
/// Returns [`EstimateError::FrameTooSmall`] or [`EstimateError::BadInput`] for unusable input, and
/// [`EstimateError::NoUsableBins`] when no bin has statistics and the prior is empty.
pub fn estimate(
    input: &EstimateInput<'_>,
    prior: &NoisePrior,
    cfg: &EstimatorConfig,
) -> Result<NoiseEstimate, EstimateError> {
    estimate_timed(input, prior, cfg).map(|(e, _)| e)
}

/// Like [`estimate`], also returning the time spent in each stage.
///
/// # Errors
///
/// As for [`estimate`].
pub fn estimate_timed(
    input: &EstimateInput<'_>,
    prior: &NoisePrior,
    cfg: &EstimatorConfig,
) -> Result<(NoiseEstimate, StageTimes), EstimateError> {
    let mut times = StageTimes::default();
    let t = std::time::Instant::now();
    let (w, h) = (input.mosaic.width(), input.mosaic.height());
    if let Some(q) = input.qstep
        && (q.width() != w || q.height() != h)
    {
        return Err(EstimateError::BadInput(
            "the qstep plane and the mosaic differ in size".into(),
        ));
    }
    if input.q_norm.is_nan() || input.q_norm < 0.0 || !input.q_norm.is_finite() {
        return Err(EstimateError::BadInput(
            "q_norm must be a non-negative number".into(),
        ));
    }
    let plan = tiles::plan(w, h, cfg)?;
    let tile_data: Vec<TileData> = plan
        .origins
        .par_iter()
        .map(|&o| tiles::load_tile(input.mosaic, input.qstep, o, plan.size))
        .collect();
    times.read_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let (clip, qn) = (cfg.clip_level as f32, input.q_norm as f32);
    let per_tile: Vec<Vec<Window>> = tile_data
        .par_iter()
        .map(|t| tiles::tile_windows(t, plan.size, clip, qn))
        .collect();
    let wins: Vec<Window> = per_tile.into_iter().flatten().collect();
    times.windows_ms = t.elapsed().as_secs_f64() * 1e3;
    let est = analyse(&tile_data, &plan, &wins, prior, cfg, &mut times)?;
    Ok((est, times))
}

fn analyse(
    tile_data: &[TileData],
    plan: &TilePlan,
    wins: &[Window],
    prior: &NoisePrior,
    cfg: &EstimatorConfig,
    times: &mut StageTimes,
) -> Result<NoiseEstimate, EstimateError> {
    let t_passes = std::time::Instant::now();
    let layout = BinLayout::default();
    let q_rms =
        (wins.iter().map(|w| f64::from(w.q2)).sum::<f64>() / wins.len().max(1) as f64).sqrt();
    let prior_model = model_from_prior(prior);

    // Passes: the first with raw energies and coefficients, the next ones normalised by the
    // previous fit (see `measure`).
    let mut model: Option<NoiseModel> = None;
    let mut last: Option<measure::PassResult> = None;
    let mut last_fit: Option<fit::FitOutput> = None;
    let mut passes = 0;
    for it in 0..=cfg.stabilise_iterations {
        passes += 1;
        let params = PassParams {
            layout: &layout,
            norm_model: model.as_ref(),
            ref_model: model.as_ref().or(prior_model.as_ref()),
            include_clipped: it > 0,
            cfg,
        };
        let pass = measure_pass(wins, &params);
        match fit_map(
            &pass
                .points
                .iter()
                .map(BinPoint::fit_point)
                .collect::<Vec<_>>(),
            prior,
        ) {
            Some(f) => {
                let new_model = fit_to_model(&f, q_rms);
                let converged = model.as_ref().is_some_and(|m| {
                    (new_model.a / m.a - 1.0).abs() < 1e-3
                        && (new_model.variance(0.2, 0) / m.variance(0.2, 0) - 1.0).abs() < 1e-3
                });
                model = Some(new_model);
                last_fit = Some(f);
                last = Some(pass);
                if converged {
                    break;
                }
            }
            None => {
                if last.is_none() {
                    last = Some(pass);
                }
                break;
            }
        }
    }
    // The loop body runs at least once and always stores a pass before it can break.
    let pass = last.ok_or(EstimateError::NoUsableBins)?;
    let surviving = pass.bins.iter().filter(|b| b.survived).count();
    let textured = pass.bins.iter().filter(|b| b.textured).count();
    let clipped_used = pass.bins.iter().filter(|b| b.survived && b.clipped).count();

    times.passes_ms = t_passes.elapsed().as_secs_f64() * 1e3;
    // Texture handling. The fallback runs when too few smooth bins survive (the spec's trigger),
    // when many points sit far above the fit, or when a cheap patch-PCA probe on the best-populated
    // bins finds the noise clearly below the smooth-region fit: texture that survives the energy
    // selection and looks Gaussian cannot be seen from the coefficients alone.
    let mut points = pass.points.clone();
    let mut fallback = false;
    let inconsistent = last_fit
        .as_ref()
        .is_some_and(|f| f.high_fraction > cfg.max_high_fraction);
    let mut run_fallback =
        cfg.texture_fallback && (surviving < cfg.min_bins_for_fit || inconsistent);
    let mut probe_ratio = None;
    if cfg.texture_fallback
        && cfg.texture_probe
        && !run_fallback
        && surviving >= 3
        && let Some(m) = &model
    {
        probe_ratio = probe_texture(tile_data, plan, &layout, &pass, m, cfg);
        run_fallback =
            probe_ratio.is_some_and(|r| r < cfg.probe_ratio_limit && r >= cfg.probe_ratio_floor);
    }
    if run_fallback {
        let (a_p, b_p) = match (&model, surviving) {
            (Some(m), s) if s >= 2 => (m.a, m.b),
            _ => (
                prior.a_center().unwrap_or(5.0e-5),
                prior.b_center().unwrap_or(1.0e-8),
            ),
        };
        let pca_cfg = pca::PcaConfig {
            a: a_p.max(fit::A_MIN),
            b: b_p.max(0.0),
            patches_per_bin_tile: cfg.pca_patches_per_bin_tile,
            stride: cfg.pca_stride.max(1),
            bin_mask: u32::MAX,
        };
        let extra = pca::texture_points(tile_data, plan, &layout, &pca_cfg);
        for p in extra {
            if p.mean > cfg.clip_bin_threshold {
                continue; // clipping changes the patch statistics; the clipped-Gaussian bins cover this range
            }
            let q2 = q_rms * q_rms / 12.0;
            let var = p.var - q2;
            // Far below the smooth-region fit: correlated noise, not texture (see the config).
            if let (Some(m), true) = (&model, surviving >= 3) {
                let fit = (0..4).map(|c| m.pixel_variance(p.mean, c)).sum::<f64>() / 4.0;
                if p.var < cfg.probe_ratio_floor * fit {
                    continue;
                }
            }
            if var > 0.0 {
                // A point from the least textured patches is an upper bound: weigh it down.
                let inflate = if p.confident { 1.0 } else { 4.0 };
                points.push(BinPoint {
                    plane: None,
                    bin: p.bin as u8,
                    kind: PointKind::Pca,
                    mean: p.mean,
                    var,
                    count: p.patches_used as u32,
                    rel_var: inflate * PCA_REL_VAR / p.patches_used as f64,
                });
                fallback = true;
            }
        }
    }

    times.fallback_ms = t_passes.elapsed().as_secs_f64() * 1e3 - times.passes_ms;
    let t_extras = std::time::Instant::now();
    let final_fit = if fallback {
        fit_map(&fit_points(&points), prior)
    } else {
        last_fit
    };

    let final_high = final_fit.as_ref().map_or(0.0, |f| f.high_fraction);
    let (mut model, confidence, residual_rms) = match final_fit {
        Some(f) => (
            fit_to_model(&f, q_rms),
            if fallback {
                Confidence::DataWithFallback
            } else {
                Confidence::Data
            },
            f.residual_rms,
        ),
        None => {
            let m = prior_model
                .map(|m| m.with_quantisation(q_rms))
                .ok_or(EstimateError::NoUsableBins)?;
            (m, Confidence::PriorOnly, 0.0)
        }
    };
    let fitted = model;

    let row_noise = if cfg.row_noise && confidence != Confidence::PriorOnly {
        rownoise::measure_row_noise(tile_data, plan, wins, &pass.smooth, &model, cfg)
    } else {
        RowNoiseInfo::default()
    };
    model.sigma_row = row_noise.sigma_row;
    // Texture raises the binned-over-pixel ratio just as correlated noise does, so the check is
    // not trusted when the fit says texture got through (the fallback ran or many points sit high).
    let texture_suspected = fallback;
    let filtered =
        if cfg.filtered_check && confidence != Confidence::PriorOnly && !texture_suspected {
            filtered::check_filtered(tile_data, plan, wins, &pass.smooth, &model, cfg)
        } else {
            FilteredCheck::default()
        };
    if filtered.flagged {
        model = model.with_sigma_scaled(filtered.factor);
    }

    times.extras_ms = t_extras.elapsed().as_secs_f64() * 1e3;
    let mut covered: Vec<usize> = points.iter().map(|p| p.bin as usize).collect();
    covered.sort_unstable();
    covered.dedup();
    let diagnostics = Diagnostics {
        confidence,
        bins: pass.bins.to_vec(),
        bins_surviving: surviving,
        bins_textured: textured,
        clipped_bins_used: clipped_used,
        coverage: layout.coverage(covered),
        texture_fallback: fallback,
        probe_ratio,
        windows_total: wins.len(),
        windows_kept: pass.windows_kept,
        residual_rms,
        high_fraction: final_high,
        passes,
        filtered,
        row_noise,
        prior_source: prior.source.clone(),
        fitted,
    };
    Ok(NoiseEstimate {
        model,
        points,
        diagnostics,
    })
}

/// Patch-PCA probe: the median ratio of the PCA variance to the fitted pixel variance over the
/// best-populated smooth bins, or `None` when too few bins gave an estimate.
fn probe_texture(
    tile_data: &[TileData],
    plan: &TilePlan,
    layout: &BinLayout,
    pass: &measure::PassResult,
    model: &NoiseModel,
    cfg: &EstimatorConfig,
) -> Option<f64> {
    let mut candidates: Vec<(usize, u32)> = pass
        .bins
        .iter()
        .enumerate()
        .filter(|(_, b)| b.survived && !b.clipped)
        .map(|(k, b)| (k, b.samples))
        .collect();
    candidates.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let mask = candidates
        .iter()
        .take(PROBE_BINS)
        .fold(0u32, |m, &(k, _)| m | 1 << k);
    let pca_cfg = pca::PcaConfig {
        a: model.a.max(fit::A_MIN),
        b: model.b.max(0.0),
        patches_per_bin_tile: cfg.pca_patches_per_bin_tile * 2 / 5,
        stride: PROBE_STRIDE,
        bin_mask: mask,
    };
    let points = pca::texture_points(tile_data, plan, layout, &pca_cfg);
    let mut ratios: Vec<f32> = points
        .iter()
        .filter(|p| p.mean <= cfg.clip_bin_threshold)
        .map(|p| {
            let fit = (0..4).map(|c| model.pixel_variance(p.mean, c)).sum::<f64>() / 4.0;
            (p.var / fit.max(1e-14)) as f32
        })
        .collect();
    (ratios.len() >= PROBE_MIN_BINS).then(|| f64::from(stats::median(&mut ratios)))
}

/// Bins the probe looks at.
const PROBE_BINS: usize = 3;

/// Fewest probe bins that give a verdict.
const PROBE_MIN_BINS: usize = 3;

/// Step between candidate patches of the probe, in plane pixels (the full fallback uses 3).
const PROBE_STRIDE: usize = 6;

/// Refits `points` with `prior` and returns the fit as a model (shared by pooling and refinement).
pub(crate) fn refit(points: &[BinPoint], prior: &NoisePrior, q: f64) -> Option<(NoiseModel, f64)> {
    fit_map(&fit_points(points), prior).map(|f| (fit_to_model(&f, q), f.residual_rms))
}

/// Number of intensity bins (re-exported for callers that size arrays).
pub const BIN_COUNT: usize = NUM_BINS;
