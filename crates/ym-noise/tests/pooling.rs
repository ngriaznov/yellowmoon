//! Batch pooling (`docs/06-noise.md` section Priors, "Batch pooling").
//!
//! Pooling adds the variance points of files that share a model, ISO, bit depth and compression
//! and refits. What it must never do is move the answer for files that carry no new information:
//! copies of one estimate, or further frames of one scene. The first version divided the relative
//! variance of a merged bin by the number of files, and three frames of one ramp scene (each
//! fitted alone at 5.03e-4 to 5.13e-4 for a true 5e-4) pooled to 1.77e-4, the centre of the prior.
//!
//! * `three_frames_of_one_ramp_scene_pool_to_the_single_file_fit` is that reproduction, on the
//!   points the estimator returned for the three frames (`tests/data/pool-ramp-iso800-3-frames.json`).
//! * `copies_of_one_estimate_pool_to_that_estimate` and `frames_with_common_scatter_...` are
//!   property tests of the invariances, with fixed seeds.
//! * `frames_of_one_synthetic_scene_...` runs the whole estimator on procedural scenes.

use proptest::prelude::*;
use ym_noise::estimate::{
    BatchPool, BinPoint, Confidence, Diagnostics, EstimateInput, EstimatorConfig, F32Plane,
    FilteredCheck, NoiseEstimate, PointKind, PoolKey, RowNoiseInfo, estimate,
};
use ym_noise::model::NoiseModel;
use ym_noise::model::synth::{SceneKind, SceneSpec, SynthRng, add_noise, render_scene};
use ym_noise::priors::{BPrior, CameraContext, LogNormal, NoisePrior, PriorSource, PriorStore};

fn key() -> PoolKey {
    PoolKey {
        model: "ILCE-7RM4".into(),
        iso: 800,
        bit_depth: 14,
        compression: "uncompressed14".into(),
    }
}

/// A file's estimate as the pool sees it: its points and a model (the pool refits from the points).
fn estimate_of_points(points: Vec<BinPoint>) -> NoiseEstimate {
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
            row_noise: RowNoiseInfo::default(),
            prior_source: PriorSource::None,
            fitted: NoiseModel::new(1e-3, 1e-6),
        },
    }
}

/// The model a pool gives after it was fed `files` in order.
fn pooled(files: &[&NoiseEstimate], prior: &NoisePrior) -> NoiseEstimate {
    let mut pool = BatchPool::new();
    let mut last = None;
    for f in files {
        last = pool.add(key(), f, prior);
    }
    last.expect("the pool fits")
}

/// The class prior of a full-frame body at ISO 800 (what ym-core hands the pool for an ILCE-7RM4
/// without a public-data prior).
fn class_prior_iso800() -> NoisePrior {
    let prior = PriorStore::empty().prior_for(&CameraContext {
        model_id: None,
        model: Some("ILCE-7RM4".into()),
        iso: Some(800),
        class: None,
    });
    assert!(matches!(prior.source, PriorSource::Class(_)));
    prior
}

fn rel(a: f64, b: f64) -> f64 {
    a / b - 1.0
}

/// Pixel variance of a model at the signal levels the sigma map is read at.
fn probe_variances(m: &NoiseModel) -> [f64; 3] {
    [0.02, 0.18, 0.5].map(|x| (0..4).map(|p| m.pixel_variance(x, p)).sum::<f64>() / 4.0)
}

// ------------------------------------------------------------------------------------------
// The reproduction

#[derive(serde::Deserialize)]
struct Fixture {
    true_a: f64,
    frames: Vec<Vec<BinPoint>>,
}

#[test]
fn three_frames_of_one_ramp_scene_pool_to_the_single_file_fit() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pool-ramp-iso800-3-frames.json");
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(fixture.frames.len(), 3);
    let prior = class_prior_iso800();
    let files: Vec<NoiseEstimate> = fixture
        .frames
        .iter()
        .map(|p| estimate_of_points(p.clone()))
        .collect();
    let own: Vec<f64> = files.iter().map(|f| pooled(&[f], &prior).model.a).collect();
    for a in &own {
        assert!(rel(*a, fixture.true_a).abs() < 0.04, "own fit {a}");
    }
    let own_mean = own.iter().sum::<f64>() / own.len() as f64;
    let refs: Vec<&NoiseEstimate> = files.iter().collect();
    for n in 1..=3 {
        let p = pooled(&refs[..n], &prior);
        eprintln!(
            "{n} files: a {:.4e} (own mean {own_mean:.4e}), b {:.3e}, residual {:.3}",
            p.model.a, p.model.b, p.diagnostics.residual_rms
        );
        assert!(
            rel(p.model.a, own_mean).abs() < 0.03,
            "{n} files pooled: a {:.4e} against the single-file fits' mean {own_mean:.4e}",
            p.model.a
        );
        assert!(
            rel(p.model.a, fixture.true_a).abs() < 0.05,
            "{n} files pooled: a {:.4e} against the truth {:.1e}",
            p.model.a,
            fixture.true_a
        );
        // The pooled fit explains the pooled points about as well as a single fit explains its own.
        assert!(
            p.diagnostics.residual_rms < 0.2,
            "residual {}",
            p.diagnostics.residual_rms
        );
    }
}

// ------------------------------------------------------------------------------------------
// Property: copies of one estimate

/// Variance points of a camera `(a, b)` over `bins` log-spaced intensities and the four planes,
/// each with a scatter of `scatter` (relative, a Gaussian draw per point) about the line and a
/// count that sets its nominal relative variance, `5.5 / count` as the estimator gives it.
fn synthetic_points(a: f64, b: f64, bins: usize, scatter: f64, seed: u64) -> Vec<BinPoint> {
    let mut rng = SynthRng::new(seed);
    let mut out = Vec::new();
    for k in 0..bins {
        let mean = 0.004 * (0.9f64 / 0.004).powf(k as f64 / (bins - 1).max(1) as f64);
        for plane in 0..4u8 {
            if rng.uniform() < 0.15 {
                continue; // not every plane has every bin
            }
            let count = 400 + (rng.uniform() * 3600.0) as u32;
            let var = (a * mean + b) * (1.0 + scatter * rng.normal()).max(0.2);
            out.push(BinPoint {
                plane: Some(plane),
                bin: k as u8,
                kind: PointKind::Mad,
                mean,
                var,
                count,
                rel_var: 5.5 / f64::from(count),
            });
        }
    }
    out
}

fn prior_for(choice: u8, a: f64) -> NoisePrior {
    match choice {
        0 => NoisePrior::none(),
        1 => class_prior_iso800(),
        _ => NoisePrior {
            log_a: Some(LogNormal {
                mean: (a * 1.3).ln(),
                sigma: 0.25,
            }),
            b: BPrior::Gaussian {
                mean: 2e-6,
                sigma: 8e-7,
            },
            delta_sigma: 3e-4,
            source: PriorSource::None,
            widened: false,
        },
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 600,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x504f_4f4c_434f_5059),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// N copies of one file carry no new information: the pooled model is the file's own, a and b
    /// to a fraction of a percent, whatever the scatter of its bins and whatever the prior.
    #[test]
    fn copies_of_one_estimate_pool_to_that_estimate(
        log_a in -9.5f64..-5.5,
        b in 1e-8f64..2e-5,
        bins in 8usize..24,
        scatter in 0.0f64..0.15,
        copies in 2usize..9,
        prior_choice in 0u8..3,
        seed in 0u64..1_000_000,
    ) {
        let a = log_a.exp();
        let prior = prior_for(prior_choice, a);
        let file = estimate_of_points(synthetic_points(a, b, bins, scatter, seed));
        let single = pooled(&[&file], &prior);
        let many: Vec<&NoiseEstimate> = std::iter::repeat_n(&file, copies).collect();
        let p = pooled(&many, &prior);
        prop_assert!(rel(p.model.a, single.model.a).abs() < 0.005,
            "a {:.4e} against {:.4e} with {copies} copies", p.model.a, single.model.a);
        prop_assert!(rel(p.model.b + 1e-12, single.model.b + 1e-12).abs() < 0.02,
            "b {:.4e} against {:.4e}", p.model.b, single.model.b);
        for (x, y) in probe_variances(&p.model).iter().zip(probe_variances(&single.model)) {
            prop_assert!(rel(*x, y).abs() < 0.005, "variance {x:.4e} against {y:.4e}");
        }
    }

    /// Frames of one scene share the scatter of their bins about the noise line (what the scene does
    /// to the bin), and differ only by the independent error of each measurement. Pooling K of them
    /// must stay with the single-file fits however large the common scatter is against the
    /// nominal error (the failing case of the first version: scatter several times the
    /// pooled nominal error).
    #[test]
    fn frames_with_common_scatter_pool_to_the_single_file_fit(
        log_a in -9.0f64..-6.5,
        b in 1e-8f64..5e-6,
        bins in 12usize..22,
        common in 0.0f64..0.09,
        frames in 2usize..9,
        prior_choice in 0u8..3,
        seed in 0u64..1_000_000,
    ) {
        let a = log_a.exp();
        let prior = prior_for(prior_choice, a);
        // The common part: one set of points. Each frame adds its own independent error with the
        // standard deviation its counts promise, as the estimator would measure.
        let base = synthetic_points(a, b, bins, common, seed);
        let mut rng = SynthRng::new(seed ^ 0xF4A3);
        let files: Vec<NoiseEstimate> = (0..frames).map(|_| {
            let pts = base.iter().map(|p| {
                let mut q = *p;
                q.var *= (1.0 + p.rel_var.sqrt() * rng.normal()).max(0.2);
                q
            }).collect();
            estimate_of_points(pts)
        }).collect();
        let singles: Vec<NoiseEstimate> = files.iter().map(|f| pooled(&[f], &prior)).collect();
        let refs: Vec<&NoiseEstimate> = files.iter().collect();
        let p = pooled(&refs, &prior);
        let mean_a = singles.iter().map(|s| s.model.a).sum::<f64>() / frames as f64;
        let (lo, hi) = singles.iter().fold((f64::MAX, f64::MIN), |(lo, hi), s| (lo.min(s.model.a), hi.max(s.model.a)));
        // Within the spread of the single-file fits, widened by a few percent (measured over 600
        // cases: the largest excursion beyond the range is 0.9% up to a common scatter of 0.06 and
        // 3.2% below 0.09; at 0.12, where the scatter is four times the nominal error and the
        // robust iteration no longer settles for single files either, 6.7% and worse).
        let tol = 0.02 + 0.5 * common;
        prop_assert!(p.model.a > lo / (1.0 + tol) && p.model.a < hi * (1.0 + tol),
            "pooled a {:.4e} outside the single fits' range {lo:.4e}..{hi:.4e} (mean {mean_a:.4e}, \
             {frames} frames, common scatter {common:.3})", p.model.a);
        let own_var: Vec<[f64; 3]> = singles.iter().map(|s| probe_variances(&s.model)).collect();
        for (i, v) in probe_variances(&p.model).iter().enumerate() {
            let lo = own_var.iter().map(|o| o[i]).fold(f64::MAX, f64::min);
            let hi = own_var.iter().map(|o| o[i]).fold(f64::MIN, f64::max);
            prop_assert!(*v > lo / (1.0 + tol) && *v < hi * (1.0 + tol),
                "pooled variance {v:.4e} at probe {i} outside {lo:.4e}..{hi:.4e}");
        }
    }
}

/// The case that a run of more than 160 cases found: a file with 11 % scatter pooled with two
/// copies of itself came out 1.6 % off its own fit (9.19e-4 against 9.35e-4). The merged points
/// were the file's own up to the last bit of `(na * x + nb * x) / n`, and the robust iteration is
/// close enough to not settling at that scatter to turn that rounding step into a percent. Merging
/// is now `a + (b - a) * w`, which gives equal points back exactly.
#[test]
fn copies_of_a_file_with_heavy_scatter_pool_to_its_own_fit_exactly() {
    let a = (-6.905001960810012f64).exp();
    let file = estimate_of_points(synthetic_points(
        a,
        1.697964749515054e-5,
        18,
        0.11178390344404855,
        308785,
    ));
    let prior = NoisePrior::none();
    let single = pooled(&[&file], &prior);
    for copies in 2..9 {
        let many: Vec<&NoiseEstimate> = std::iter::repeat_n(&file, copies).collect();
        let p = pooled(&many, &prior);
        assert_eq!(
            p.model.a, single.model.a,
            "{copies} copies: a is the single file's, to the bit"
        );
        assert_eq!(p.model.b, single.model.b, "{copies} copies: b");
    }
}

// ------------------------------------------------------------------------------------------
// Whole estimator on procedural scenes

const W: usize = 768;
const H: usize = 512;
const DN: f64 = 1.0 / 15871.0;

fn estimate_mosaic(m: &[f32], prior: &NoisePrior) -> NoiseEstimate {
    let plane = F32Plane::new(m, W, H).unwrap();
    estimate(
        &EstimateInput {
            mosaic: &plane,
            qstep: None,
            q_norm: DN,
        },
        prior,
        &EstimatorConfig::default(),
    )
    .unwrap()
}

#[test]
fn frames_of_one_synthetic_scene_pool_to_the_single_file_fit() {
    let kinds = [SceneKind::Smooth, SceneKind::Mixed, SceneKind::Busy];
    let mut worst_a: f64 = 0.0;
    let mut worst_var: f64 = 0.0;
    let mut cases = 0;
    for seed in 0..18u64 {
        let kind = kinds[(seed % 3) as usize];
        let frames = 2 + (seed / 3 % 3) as usize; // 2, 3, 4
        let mut rng = SynthRng::new(seed ^ 0xC0DE);
        // ISO 800 on a full-frame body: a around 3e-4 to 8e-4, read noise a few electrons.
        let a = (2.5e-4f64.ln() + (9e-4f64.ln() - 2.5e-4f64.ln()) * rng.uniform()).exp();
        let b =
            (a * (1.5 + 3.0 * rng.uniform())).powi(2) + (DN * (0.6 + 0.8 * rng.uniform())).powi(2);
        let truth = NoiseModel {
            a,
            b,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: DN,
        };
        let scene = render_scene(&SceneSpec {
            width: W,
            height: H,
            seed: 7000 + seed,
            kind,
            exposure: 1.0,
        });
        let prior = class_prior_iso800();
        let files: Vec<NoiseEstimate> = (0..frames as u64)
            .map(|f| {
                let m = add_noise(&scene.signal, W, H, &truth, 0xA5A5 + 31 * seed + f);
                estimate_mosaic(&m, &prior)
            })
            .collect();
        let singles: Vec<NoiseEstimate> = files.iter().map(|f| pooled(&[f], &prior)).collect();
        let refs: Vec<&NoiseEstimate> = files.iter().collect();
        let p = pooled(&refs, &prior);
        let mean_a = singles.iter().map(|s| s.model.a).sum::<f64>() / frames as f64;
        let da = rel(p.model.a, mean_a).abs();
        let own_var: Vec<f64> = singles
            .iter()
            .map(|s| probe_variances(&s.model)[1])
            .collect();
        let mean_var = own_var.iter().sum::<f64>() / frames as f64;
        let dv = rel(probe_variances(&p.model)[1], mean_var).abs();
        eprintln!(
            "seed {seed} {kind:?} {frames} frames: single a {:?} pooled {:.3e} (mean {mean_a:.3e}, truth {a:.3e}), \
             variance at 0.18 off by {:.1}%",
            singles
                .iter()
                .map(|s| format!("{:.2e}", s.model.a))
                .collect::<Vec<_>>(),
            p.model.a,
            dv * 100.0
        );
        worst_a = worst_a.max(da);
        worst_var = worst_var.max(dv);
        cases += 1;
        assert!(
            da < 0.08,
            "seed {seed} ({kind:?}, {frames} frames): pooled a {:.3e} against the single fits' mean {mean_a:.3e}",
            p.model.a
        );
        assert!(
            dv < 0.06,
            "seed {seed} ({kind:?}, {frames} frames): pooled variance at 0.18 is {dv:.3} off the single fits'"
        );
    }
    eprintln!(
        "{cases} scenes: worst a {:.1}%, worst variance {:.1}%",
        worst_a * 100.0,
        worst_var * 100.0
    );
}
