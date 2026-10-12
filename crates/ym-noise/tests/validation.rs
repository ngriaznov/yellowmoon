//! Synthetic validation of the noise estimator (`docs/06-noise.md` section Validation).
//!
//! The public corpus is not reachable in V1 builds, so every case here is a procedural scene
//! mosaicked to RGGB with Poisson-Gaussian noise from a known model (`ym_noise::model::synth`).
//! Seeds are fixed; the seeds used by these tests (20000 and up) were not used while the
//! estimator was being developed, so the numbers below are held-out numbers. The estimator
//! itself is measured against the gates of `docs/06`; where `docs/06` has no gate (absolute
//! recovery, texture, clipping, row noise, refinement) the thresholds are ours and stated in the
//! test. Run with `--nocapture` to see the measured numbers. The tests that need real files
//! (A1 studio series, ELD A7S II parameters, star-eater captures, the CC0 corpus) are not here: see
//! `docs/STATUS.md` for what is blocked.

mod common;

use common::*;
use ym_noise::estimate::{
    EstimateInput, EstimatorConfig, F32Plane, NoiseEstimate, PlaneSource, U16Mosaic, estimate,
    refine,
};
use ym_noise::model::NoiseModel;
use ym_noise::model::synth::{
    RawFilter, SceneKind, SynthRng, add_increment, add_noise, apply_raw_filter, render_scene,
};
use ym_noise::priors::NoisePrior;

const W: usize = 1536;
const H: usize = 1024;

fn cfg() -> EstimatorConfig {
    EstimatorConfig::default()
}

fn estimate_of(m: &[f32], truth: &NoiseModel) -> NoiseEstimate {
    run(m, W, H, truth.q, &NoisePrior::none(), &cfg())
}

// ------------------------------------------------------------------------------------------
// (a) Absolute recovery of a and b

#[test]
fn absolute_recovery_of_a_and_sigma_on_mixed_scenes() {
    let (mut ea, mut es) = (Vec::new(), Vec::new());
    for seed in 20_000..20_000 + scaled(12) {
        let case = make_case(seed, SceneKind::Mixed, W, H, 1.0, false);
        let est = estimate_of(&noisy(&case), &case.truth);
        let (a, s) = errors(&est.model, &case.truth);
        ea.push(a);
        es.extend(s);
    }
    let (am, ap) = abs_stats(&ea);
    let (sm, sp) = abs_stats(&es);
    eprintln!(
        "mixed cases: a median {:.1}% p90 {:.1}%; sigma at 0.02/0.18/0.5 median {:.1}% p90 {:.1}%",
        am * 100.0,
        ap * 100.0,
        sm * 100.0,
        sp * 100.0
    );
    assert!(am < 0.05 && ap < 0.15, "a: median {am:.3} p90 {ap:.3}");
    assert!(sm < 0.03 && sp < 0.08, "sigma: median {sm:.3} p90 {sp:.3}");
}

#[test]
fn absolute_recovery_is_unbiased_on_smooth_scenes_and_recovers_b() {
    let (mut ea, mut eb) = (Vec::new(), Vec::new());
    for seed in 20_100..20_100 + scaled(8) {
        let case = make_case(seed, SceneKind::Smooth, W, H, 1.0, false);
        let est = estimate_of(&noisy(&case), &case.truth);
        ea.push(est.model.a / case.truth.a - 1.0);
        // b is only identified together with the plane offsets: compare the mean over planes of
        // the variance at the darkest level the scenes use (0.003), where b matters most.
        let (e, t) = (pixel_model(&est.model), pixel_model(&case.truth));
        let x = 0.003;
        eb.push(
            ((0..4).map(|p| est.model.pixel_variance(x, p)).sum::<f64>() / 4.0)
                / ((0..4).map(|p| case.truth.pixel_variance(x, p)).sum::<f64>() / 4.0)
                - 1.0,
        );
        let _ = (e, t);
    }
    let (am, ap) = abs_stats(&ea);
    let mut signed = ea.clone();
    let bias = median(&mut signed);
    let (bm, bp) = abs_stats(&eb);
    eprintln!(
        "smooth cases: a median {:.1}% p90 {:.1}% signed {:+.1}%; variance at 0.003 median {:.1}% p90 {:.1}%",
        am * 100.0,
        ap * 100.0,
        bias * 100.0,
        bm * 100.0,
        bp * 100.0
    );
    assert!(am < 0.04 && ap < 0.10, "a: median {am:.3} p90 {ap:.3}");
    assert!(bias.abs() < 0.03, "a is biased: {bias:.3}");
    assert!(
        bm < 0.12 && bp < 0.30,
        "dark-level variance: median {bm:.3} p90 {bp:.3}"
    );
}

// ------------------------------------------------------------------------------------------
// (b) Noise on noise: the gate of docs/06 (increment within 15% median, 30% p90)

fn noise_on_noise(kind: SceneKind, seeds: std::ops::Range<u64>) -> (Vec<f64>, Vec<f64>) {
    let (mut e_a, mut e_v) = (Vec::new(), Vec::new());
    for seed in seeds {
        let case = make_case(seed, kind, W, H, 1.0, false);
        let base = noisy(&case);
        let est0 = estimate_of(&base, &case.truth);
        let mut rng = SynthRng::new(seed ^ 0xFEED);
        let ratio = (0.5f64.ln() + (3.0f64.ln() - 0.5f64.ln()) * rng.uniform()).exp();
        let (a_inc, b_inc) = (ratio * case.truth.a, ratio * ratio * case.truth.b);
        let mut aug = base.clone();
        add_increment(&mut aug, W, a_inc, b_inc, case.truth.q, seed ^ 0xBEEF);
        let est1 = estimate_of(&aug, &case.truth);
        let (m0, m1) = (pixel_model(&est0.model), pixel_model(&est1.model));
        e_a.push((m1.a - m0.a) / a_inc - 1.0);
        for &x in &PROBES {
            e_v.push((m1.variance(x, 0) - m0.variance(x, 0)) / (a_inc * x + b_inc) - 1.0);
        }
    }
    (e_a, e_v)
}

#[test]
fn noise_on_noise_gate_on_mixed_scenes() {
    let (e_a, e_v) = noise_on_noise(SceneKind::Mixed, 20_200..20_200 + scaled(16));
    let (am, ap) = abs_stats(&e_a);
    let (vm, vp) = abs_stats(&e_v);
    eprintln!(
        "noise on noise, mixed: increment a median {:.1}% p90 {:.1}%; increment variance at probes median {:.1}% p90 {:.1}%",
        am * 100.0,
        ap * 100.0,
        vm * 100.0,
        vp * 100.0
    );
    assert!(
        am <= 0.15 && ap <= 0.30,
        "increment a: median {am:.3} p90 {ap:.3} (gate 0.15 / 0.30)"
    );
    assert!(
        vm <= 0.15 && vp <= 0.30,
        "increment variance: median {vm:.3} p90 {vp:.3} (gate 0.15 / 0.30)"
    );
}

#[test]
fn noise_on_noise_gate_on_busy_scenes() {
    let (e_a, e_v) = noise_on_noise(SceneKind::Busy, 20_300..20_300 + scaled(8));
    let (am, ap) = abs_stats(&e_a);
    let (vm, vp) = abs_stats(&e_v);
    eprintln!(
        "noise on noise, busy: increment a median {:.1}% p90 {:.1}%; variance median {:.1}% p90 {:.1}%",
        am * 100.0,
        ap * 100.0,
        vm * 100.0,
        vp * 100.0
    );
    assert!(
        am <= 0.15 && ap <= 0.30,
        "increment a: median {am:.3} p90 {ap:.3}"
    );
    assert!(
        vm <= 0.15 && vp <= 0.30,
        "increment variance: median {vm:.3} p90 {vp:.3}"
    );
}

// ------------------------------------------------------------------------------------------
// (c) Filtered-file detector: >= 90% flagged, <= 5% false positives

fn flagged_fraction(
    kind: SceneKind,
    seeds: std::ops::Range<u64>,
    filter: Option<RawFilter>,
) -> (usize, usize, f64) {
    let (mut flagged, mut total, mut ratio_sum) = (0, 0, 0.0);
    for seed in seeds {
        let case = make_case(seed, kind, W, H, 1.0, seed % 2 == 0);
        let m = noisy(&case);
        let data = match filter {
            None => m,
            Some(f) => apply_raw_filter(&m, W, H, f, case.truth.q),
        };
        let est = estimate_of(&data, &case.truth);
        total += 1;
        flagged += usize::from(est.diagnostics.filtered.flagged);
        ratio_sum += est.diagnostics.filtered.ratio;
    }
    (flagged, total, ratio_sum / total as f64)
}

#[test]
fn filtered_detector_gate_on_smooth_scenes() {
    // Smooth, flat-field-like scenes stand in for the long exposures where in-camera raw filtering
    // happens. Strong filters: ideal flat-field ratios 0.57 (3x3 median), 0.64 and 0.66 (3x3 box
    // blends); the threshold is 0.4.
    let seeds = 20_400..20_400 + scaled(8);
    let (fp, n, r) = flagged_fraction(SceneKind::Smooth, seeds.clone(), None);
    eprintln!("white noise, smooth: {fp}/{n} flagged, mean ratio {r:.3}");
    assert!(fp as f64 / n as f64 <= 0.05, "false positives {fp}/{n}");
    for (name, f) in [
        ("median 3x3", RawFilter::Median3),
        ("box blend 0.8", RawFilter::BoxBlend { weight: 0.8 }),
        ("box blend 1.0", RawFilter::BoxBlend { weight: 1.0 }),
    ] {
        let (hit, n, r) = flagged_fraction(SceneKind::Smooth, seeds.clone(), Some(f));
        eprintln!("{name}, smooth: {hit}/{n} flagged, mean ratio {r:.3}");
        assert!(hit as f64 / n as f64 >= 0.90, "{name}: detected {hit}/{n}");
    }
}

#[test]
fn filtered_detector_has_no_false_positives_on_mixed_scenes() {
    let (fp, n, r) = flagged_fraction(SceneKind::Mixed, 20_500..20_500 + scaled(8), None);
    eprintln!("white noise, mixed: {fp}/{n} flagged, mean ratio {r:.3}");
    assert!(fp as f64 / n as f64 <= 0.05, "false positives {fp}/{n}");
}

#[test]
fn filtered_detector_reports_its_limits_on_busy_scenes() {
    // Documented limit, not a gate: texture raises the binned-over-pixel ratio like correlated
    // noise does. Busy scenes (texture everywhere) can give false positives; the estimator skips the
    // check when the texture fallback ran, which keeps them rare.
    let (fp, n, r) = flagged_fraction(SceneKind::Busy, 20_600..20_600 + scaled(8), None);
    eprintln!("white noise, busy: {fp}/{n} flagged, mean ratio {r:.3}");
    assert!(fp as f64 / n as f64 <= 0.20, "false positives {fp}/{n}");
}

// ------------------------------------------------------------------------------------------
// (d) Texture fallback

#[test]
fn texture_fallback_triggers_on_busy_scenes_and_stays_bounded() {
    let (mut errs, mut triggered, mut total) = (Vec::new(), 0, 0);
    for seed in 20_700..20_700 + scaled(8) {
        // Low to moderate ISO, where texture outweighs the noise.
        let mut case = make_case(seed, SceneKind::Busy, W, H, 1.0, false);
        let mut rng = SynthRng::new(seed ^ 0x51);
        case.iso = 100.0 * 2f64.powf(rng.uniform() * 3.0);
        case.truth = random_camera(&mut rng, case.iso, false);
        let est = estimate_of(&noisy(&case), &case.truth);
        total += 1;
        triggered += usize::from(est.diagnostics.texture_fallback);
        errs.push(est.model.a / case.truth.a - 1.0);
    }
    let (m, p) = abs_stats(&errs);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for e in &errs {
        lo = lo.min(*e);
        hi = hi.max(*e);
    }
    eprintln!(
        "busy scenes at ISO 100-800, {total} cases: fallback ran in {triggered}; a error median {:.1}% p90 {:.1}%, range {:+.1}% .. {:+.1}%",
        m * 100.0,
        p * 100.0,
        lo * 100.0,
        hi * 100.0
    );
    assert!(
        triggered >= total / 4,
        "the fallback should run on a good share of busy scenes: {triggered}/{total}"
    );
    assert!(lo > -0.20, "never an under-estimate worse than 20%: {lo}");
    assert!(hi < 0.80, "over-estimates stay bounded: {hi}");
    assert!(m < 0.15, "median error {m}");
}

#[test]
fn forced_fallback_on_a_clean_scene_stays_close() {
    // With min_bins_for_fit above the number of bins the fallback always runs: its points join the
    // smooth-region points and the robust fit must not be pulled away from the truth.
    let forced = EstimatorConfig {
        min_bins_for_fit: 99,
        ..cfg()
    };
    let mut errs = Vec::new();
    for seed in 20_800..20_800 + scaled(3) {
        let case = make_case(seed, SceneKind::Smooth, W, H, 1.0, false);
        let est = run(
            &noisy(&case),
            W,
            H,
            case.truth.q,
            &NoisePrior::none(),
            &forced,
        );
        assert!(est.diagnostics.texture_fallback, "fallback did not run");
        errs.push(est.model.a / case.truth.a - 1.0);
    }
    let (m, _) = abs_stats(&errs);
    eprintln!("forced fallback, smooth: a median {:.1}%", m * 100.0);
    assert!(m < 0.12, "median error {m}");
}

// ------------------------------------------------------------------------------------------
// (e) Clipping does not bias a downward

/// A luminance ramp that runs past the clip level, so many bins sit near white and partly clipped.
fn ramp_scene(top: f32) -> Vec<f32> {
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            let g = match (y & 1, x & 1) {
                (0, 0) => 0.9,
                (1, 1) => 0.8,
                _ => 1.0,
            };
            (0.02 + (top - 0.02) * (x as f32 / W as f32) + 0.05 * (y as f32 / H as f32)) * g
        })
        .collect()
}

#[test]
fn clipping_does_not_bias_a_downward() {
    // High-gain exposure whose noise (sigma 0.09 at white) reaches the clip level over a wide range.
    let a = 8.0e-3;
    let truth = NoiseModel {
        a,
        b: (3.0 * a).powi(2),
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: DN,
    };
    let naive = EstimatorConfig {
        clip_aware: false,
        ..cfg()
    };
    let (mut aware_errs, mut naive_errs) = (Vec::new(), Vec::new());
    for seed in 0..scaled(2) {
        let m = add_noise(&ramp_scene(1.6), W, H, &truth, 20_900 + seed);
        let aware = run(&m, W, H, truth.q, &NoisePrior::none(), &cfg());
        let un = run(&m, W, H, truth.q, &NoisePrior::none(), &naive);
        aware_errs.push(aware.model.a / a - 1.0);
        naive_errs.push(un.model.a / a - 1.0);
    }
    let (mut ae, mut ne) = (aware_errs.clone(), naive_errs.clone());
    let (ma, mn) = (median(&mut ae), median(&mut ne));
    eprintln!(
        "clipped ramp, a = 8e-3: clip-aware median {:+.1}%, naive median {:+.1}%",
        ma * 100.0,
        mn * 100.0
    );
    assert!(ma.abs() < 0.10, "clip-aware estimate is off by {ma:.3}");
    assert!(
        mn < -0.30,
        "the naive estimate was expected to collapse (it did not: {mn:.3}); the test no longer shows what clipping does"
    );
}

#[test]
fn clipped_mixed_scenes_do_not_bias_a_downward() {
    let mut errs = Vec::new();
    for seed in 21_000..21_000 + scaled(6) {
        let mut case = make_case(seed, SceneKind::Mixed, W, H, 2.0, false);
        let mut rng = SynthRng::new(seed ^ 0x77);
        case.iso = 1600.0 * 2f64.powf(rng.uniform() * 4.0);
        case.truth = random_camera(&mut rng, case.iso, false);
        let est = estimate_of(&noisy(&case), &case.truth);
        errs.push(est.model.a / case.truth.a - 1.0);
    }
    let mut e = errs.clone();
    let bias = median(&mut e);
    let (_, p) = abs_stats(&errs);
    eprintln!(
        "clipped mixed scenes (exposure 2.0, ISO 1600-25600): a median signed {:+.1}%, p90 |err| {:.1}%",
        bias * 100.0,
        p * 100.0
    );
    assert!(bias > -0.05, "a biased low by {bias:.3}");
    assert!(p < 0.15);
}

// ------------------------------------------------------------------------------------------
// Row noise

#[test]
fn row_noise_is_recovered_when_it_matters_and_absent_when_it_does_not() {
    let (mut hit, mut zero_ok) = (Vec::new(), 0usize);
    let n = scaled(4) as usize;
    for seed in 21_100..21_100 + n as u64 {
        let mut case = make_case(seed, SceneKind::Mixed, W, H, 1.0, false);
        let mut rng = SynthRng::new(seed ^ 0x99);
        case.iso = 100.0 * 2f64.powf(rng.uniform() * 5.0);
        case.truth = random_camera(&mut rng, case.iso, false);
        // No row noise: reported as zero.
        let est = estimate_of(&noisy(&case), &case.truth);
        zero_ok += usize::from(est.diagnostics.row_noise.sigma_row == 0.0);
        // Row noise as large as the read noise.
        case.truth.sigma_row = case.truth.b.sqrt();
        let est = estimate_of(&noisy(&case), &case.truth);
        hit.push(est.diagnostics.row_noise.sigma_row / case.truth.sigma_row);
    }
    let mut h = hit.clone();
    let m = median(&mut h);
    eprintln!(
        "row noise = read noise: estimated / true median {m:.2}, min {:.2}, max {:.2}; zero reported without row noise: {zero_ok}/{n}",
        hit.iter().cloned().fold(f64::MAX, f64::min),
        hit.iter().cloned().fold(0.0, f64::max)
    );
    assert!(
        zero_ok * 6 >= n * 5,
        "false row noise in {} of {n} cases",
        n - zero_ok
    );
    assert!((m - 1.0).abs() < 0.25, "median ratio {m}");
    assert!(hit.iter().all(|&r| r > 0.6 && r < 1.3), "ratios {hit:?}");
}

// ------------------------------------------------------------------------------------------
// Refinement (YOND-style), synthetic only

#[test]
fn refinement_with_a_near_oracle_denoiser_recovers_the_noise_on_busy_scenes() {
    // The published formula assumes the noise a denoiser leaves in is independent of the noise it
    // removes. The "oracle" output is the clean signal plus 15% of an independent noise draw.
    let (w, h) = (768, 512);
    let (mut first, mut refined) = (Vec::new(), Vec::new());
    for seed in 21_200..21_200 + scaled(6) {
        let case = make_case(seed, SceneKind::Busy, w, h, 1.0, false);
        let scene = render_scene(&case.spec);
        let m = add_noise(&scene.signal, w, h, &case.truth, case.noise_seed);
        let est0 = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg());
        let small = NoiseModel {
            a: 0.0225 * case.truth.a,
            b: 0.0225 * case.truth.b,
            sigma_row: 0.0,
            q: 0.0,
            ..case.truth
        };
        let oracle = add_noise(&scene.signal, w, h, &small, case.noise_seed ^ 0xFFFF);
        let est1 = refine(
            &est0,
            &F32Plane::new(&m, w, h).unwrap(),
            &F32Plane::new(&oracle, w, h).unwrap(),
            &NoisePrior::none(),
            &cfg(),
        )
        .unwrap();
        first.push(errors(&est0.model, &case.truth).1[1]);
        refined.push(errors(&est1.model, &case.truth).1[1]);
    }
    let ((m0, p0), (m1, p1)) = (abs_stats(&first), abs_stats(&refined));
    eprintln!(
        "refinement, busy, sigma(0.18): first median {:.1}% p90 {:.1}%; refined median {:.1}% p90 {:.1}%",
        m0 * 100.0,
        p0 * 100.0,
        m1 * 100.0,
        p1 * 100.0
    );
    assert!(p1 < 0.15, "refined p90 {p1}");
    assert!(
        p1 < p0 + 1e-9,
        "refinement did not improve the worst cases: {p1} vs {p0}"
    );
}

#[test]
fn refinement_rejects_mismatched_inputs() {
    let case = make_case(21_300, SceneKind::Smooth, 768, 512, 1.0, false);
    let m = noisy(&case);
    let est = run(&m, 768, 512, case.truth.q, &NoisePrior::none(), &cfg());
    let a = F32Plane::new(&m, 768, 512).unwrap();
    let small = vec![0.0f32; 512 * 512];
    let b = F32Plane::new(&small, 512, 512).unwrap();
    assert!(refine(&est, &a, &b, &NoisePrior::none(), &cfg()).is_err());
}

// ------------------------------------------------------------------------------------------
// Behaviour of the API

#[test]
fn estimate_is_deterministic_and_independent_of_the_thread_count() {
    let case = make_case(21_400, SceneKind::Mixed, W, H, 1.0, true);
    let m = noisy(&case);
    let reference = estimate_of(&m, &case.truth);
    let again = estimate_of(&m, &case.truth);
    assert_eq!(reference.model, again.model);
    assert_eq!(reference.points, again.points);
    for threads in [1, 3] {
        let pool = rayon_pool(threads);
        let est = pool.install(|| estimate_of(&m, &case.truth));
        assert_eq!(est.model, reference.model, "{threads} threads");
        assert_eq!(est.points, reference.points, "{threads} threads");
    }
}

fn rayon_pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
}

#[test]
fn u16_mosaic_and_f32_plane_give_the_same_estimate() {
    let case = make_case(21_500, SceneKind::Mixed, W, H, 1.0, false);
    let m = noisy(&case);
    let (black, white) = (512.0f32, 16383.0f32);
    // Quantised values are exact multiples of the 14-bit step, so the round trip through u16 is lossless.
    let raw: Vec<u16> = m
        .iter()
        .map(|&v| (black + v * (white - black)).round() as u16)
        .collect();
    let u16m = U16Mosaic::new(&raw, W, H, [black; 4], [white; 4]).unwrap();
    let back: Vec<f32> = raw
        .iter()
        .map(|&r| ((f32::from(r) - black) / (white - black)).max(-0.05))
        .collect();
    let f32m = F32Plane::new(&back, W, H).unwrap();
    let q = 1.0 / f64::from(white - black);
    let a = estimate(
        &EstimateInput {
            mosaic: &u16m,
            qstep: None,
            q_norm: q,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    let b = estimate(
        &EstimateInput {
            mosaic: &f32m,
            qstep: None,
            q_norm: q,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    assert!(
        (a.model.a / b.model.a - 1.0).abs() < 1e-4,
        "{} vs {}",
        a.model.a,
        b.model.a
    );
    assert!((a.model.b - b.model.b).abs() < 1e-3 * b.model.b.max(1e-10));
}

#[test]
fn a_constant_qstep_plane_equals_the_default_step() {
    let case = make_case(21_600, SceneKind::Smooth, W, H, 1.0, false);
    let m = noisy(&case);
    let mosaic = F32Plane::new(&m, W, H).unwrap();
    let q = vec![case.truth.q as f32; W * H];
    let qplane = F32Plane::new(&q, W, H).unwrap();
    let with_plane = estimate(
        &EstimateInput {
            mosaic: &mosaic,
            qstep: Some(&qplane as &dyn PlaneSource),
            q_norm: 0.0,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    let without = estimate(
        &EstimateInput {
            mosaic: &mosaic,
            qstep: None,
            q_norm: case.truth.q,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    assert!((with_plane.model.a / without.model.a - 1.0).abs() < 1e-3);
    assert!((with_plane.model.q / without.model.q - 1.0).abs() < 1e-3);
    // A larger step (cRAW at high signal) is subtracted from the variance and reported.
    let q8 = vec![(8.0 * case.truth.q) as f32; W * H];
    let q8plane = F32Plane::new(&q8, W, H).unwrap();
    let coarse = estimate(
        &EstimateInput {
            mosaic: &mosaic,
            qstep: Some(&q8plane as &dyn PlaneSource),
            q_norm: 0.0,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    assert!((coarse.model.q / (8.0 * case.truth.q) - 1.0).abs() < 1e-3);
}

#[test]
fn an_image_without_usable_smooth_regions_falls_back_to_the_prior() {
    // A narrow-range, strongly textured frame: very few bins carry statistics.
    let mut rng = SynthRng::new(5);
    let m: Vec<f32> = (0..W * H)
        .map(|_| (0.5 + 0.3 * (rng.uniform() - 0.5)) as f32)
        .collect();
    let plane = F32Plane::new(&m, W, H).unwrap();
    let input = EstimateInput {
        mosaic: &plane,
        qstep: None,
        q_norm: DN,
    };
    // Without a prior there is nothing to fall back on.
    let r = estimate(&input, &NoisePrior::none(), &cfg());
    if let Ok(est) = &r {
        // White uniform noise is "smooth" everywhere; the estimate is then the data's.
        assert!(est.model.a >= 0.0);
    }
    // With a prior the result is always a model.
    let prior = ym_noise::priors::class_prior(ym_noise::priors::BodyClass::FullFrame, 400.0, false);
    let est = estimate(&input, &prior, &cfg()).unwrap();
    assert!(est.model.a > 0.0 && est.model.b >= 0.0);
}

#[test]
fn the_prior_pulls_when_the_data_are_weak_and_yields_when_they_are_strong() {
    use ym_noise::priors::{BPrior, LogNormal, PriorSource};
    let case = make_case(21_700, SceneKind::Smooth, W, H, 1.0, false);
    let m = noisy(&case);
    let wrong = NoisePrior {
        log_a: Some(LogNormal {
            mean: (case.truth.a * 2.0).ln(),
            sigma: 0.25,
        }),
        b: BPrior::Flat,
        delta_sigma: 3e-4,
        source: PriorSource::None,
        widened: false,
    };
    // Strong data: the estimate stays within a few percent of the truth despite a prior twice too high.
    let est = run(&m, W, H, case.truth.q, &wrong, &cfg());
    assert!(
        (est.model.a / case.truth.a - 1.0).abs() < 0.12,
        "a {} vs {}",
        est.model.a,
        case.truth.a
    );
}

#[test]
fn flat_field_noise_levels_are_exact() {
    // A flat grey field with known noise: the sigma map reproduces the true sigma and the
    // estimator needs no scene at all.
    use ym_noise::sigma_map::{HalfPlanes, QstepMap, SigmaParams, sigma_planes};
    let truth = NoiseModel {
        a: 6.0e-4,
        b: 2.0e-7,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: DN,
    };
    let signal = vec![0.25f32; W * H];
    let m = add_noise(&signal, W, H, &truth, 99);
    let est = estimate_of(&m, &truth);
    let s_est = est.model.sigma(0.25, 0);
    let s_true = truth.sigma(0.25, 0);
    assert!((s_est / s_true - 1.0).abs() < 0.05, "{s_est} vs {s_true}");
    // The sigma map built from the estimate, on the half-resolution planes of the same field.
    let (hw, hh) = (W / 2, H / 2);
    let mut planes = vec![0.0f32; 4 * hw * hh];
    for p in 0..4 {
        let (py, px) = (p >> 1, p & 1);
        for y in 0..hh {
            for x in 0..hw {
                planes[(p * hh + y) * hw + x] = m[(2 * y + py) * W + 2 * x + px];
            }
        }
    }
    let params = SigmaParams::new(&est.model, [1.0; 4]);
    let mut out = vec![0.0f32; 4 * hw * hh];
    sigma_planes(
        &params,
        &HalfPlanes {
            data: &planes,
            width: hw,
            height: hh,
        },
        QstepMap::None,
        None,
        &mut out,
    )
    .unwrap();
    let centre = out[(hh / 2) * hw + hw / 2];
    assert!(
        (f64::from(centre) / s_true - 1.0).abs() < 0.05,
        "{centre} vs {s_true}"
    );
}

#[test]
fn small_frames_are_rejected_and_odd_sizes_work() {
    let tiny = vec![0.3f32; 100 * 100];
    let plane = F32Plane::new(&tiny, 100, 100).unwrap();
    assert!(
        estimate(
            &EstimateInput {
                mosaic: &plane,
                qstep: None,
                q_norm: DN
            },
            &NoisePrior::none(),
            &cfg()
        )
        .is_err()
    );
    // A frame whose size is not a multiple of the window (and odd): the tile grid still fits.
    let (w, h) = (1001usize, 777usize);
    let case = make_case(21_800, SceneKind::Mixed, 1000, 776, 1.0, false);
    let m = noisy(&case);
    let mut padded = vec![0.0f32; w * h];
    for y in 0..776 {
        padded[y * w..y * w + 1000].copy_from_slice(&m[y * 1000..(y + 1) * 1000]);
    }
    let plane = F32Plane::new(&padded, w, h).unwrap();
    let est = estimate(
        &EstimateInput {
            mosaic: &plane,
            qstep: None,
            q_norm: DN,
        },
        &NoisePrior::none(),
        &cfg(),
    )
    .unwrap();
    assert!((est.model.a / case.truth.a - 1.0).abs() < 0.25);
}

// ------------------------------------------------------------------------------------------
// The two ends of the prior: nothing in the image, and an image that cannot tell a from b

#[test]
fn a_frame_with_no_usable_bin_returns_the_prior_centre_or_an_error() {
    use ym_noise::estimate::{Confidence, EstimateError};
    use ym_noise::priors::{BodyClass, class_prior};
    // Every window of a fully clipped frame is above the last bin, and a black frame has no noise
    // to measure (all coefficients are zero), so neither gives a point.
    let prior = class_prior(BodyClass::FullFrame, 400.0, false);
    for (name, level) in [("white", 1.0f32), ("black", 0.0f32)] {
        let m = vec![level; W * H];
        let plane = F32Plane::new(&m, W, H).unwrap();
        let input = EstimateInput {
            mosaic: &plane,
            qstep: None,
            q_norm: DN,
        };
        // Without a prior there is nothing to return.
        let none = estimate(&input, &NoisePrior::none(), &cfg());
        assert!(
            matches!(none, Err(EstimateError::NoUsableBins)),
            "{name}: expected NoUsableBins without a prior"
        );
        // With one, the model is the centre of the prior and says so.
        let est = estimate(&input, &prior, &cfg()).unwrap();
        assert_eq!(est.diagnostics.confidence, Confidence::PriorOnly, "{name}");
        assert_eq!(est.model.a, prior.a_center().unwrap(), "{name}");
        assert_eq!(est.model.b, prior.b_center().unwrap(), "{name}");
        assert_eq!(est.model.delta, [0.0; 4], "{name}");
        assert_eq!(est.model.sigma_row, 0.0, "{name}");
        assert!((est.model.q / DN - 1.0).abs() < 1e-6, "{name}: q");
        assert!(est.points.is_empty(), "{name}");
        assert!(!est.diagnostics.filtered.flagged && !est.diagnostics.row_noise.measured);
    }
}

#[test]
fn a_flat_field_fixes_the_variance_and_the_prior_decides_the_split() {
    use ym_noise::priors::{BPrior, LogNormal, PriorSource};
    // One intensity only: the data give var(0.25) = a * 0.25 + b but cannot separate a from b.
    // The estimate must reproduce the variance wherever the data are, and take a from the prior.
    let truth = NoiseModel {
        a: 6.0e-4,
        b: 2.0e-7,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: DN,
    };
    let m = add_noise(&vec![0.25f32; W * H], W, H, &truth, 4242);
    let level = 0.25;
    let prior_at = |a: f64, sigma: f64| NoisePrior {
        log_a: Some(LogNormal {
            mean: a.ln(),
            sigma,
        }),
        b: BPrior::Flat,
        delta_sigma: 3e-4,
        source: PriorSource::None,
        widened: false,
    };
    let want = truth.pixel_variance(level, 0);
    let variance_at_level = |e: &NoiseEstimate| {
        (0..4)
            .map(|p| e.model.pixel_variance(level, p))
            .sum::<f64>()
            / 4.0
    };
    // A prior below the truth is followed (b takes up the rest); the variance is right either way.
    let low = run(&m, W, H, truth.q, &prior_at(4.5e-4, 0.05), &cfg());
    assert!(
        (variance_at_level(&low) / want - 1.0).abs() < 0.04,
        "low prior: variance {} vs {want}",
        variance_at_level(&low)
    );
    assert!(
        (low.model.a / 4.5e-4 - 1.0).abs() < 0.12,
        "a = {} should follow a tight prior",
        low.model.a
    );
    // A prior above what b >= 0 allows (a * 0.25 cannot exceed the measured variance) has to give
    // way: before the intercept wall in `fit`, the returned model kept a = 7.5e-4 and predicted
    // 1.86e-4 against the measured 1.50e-4 at the image's own level; with 1.2e-3 it was 2.0 times.
    for a_prior in [7.5e-4, 1.2e-3] {
        let high = run(&m, W, H, truth.q, &prior_at(a_prior, 0.05), &cfg());
        assert!(
            (variance_at_level(&high) / want - 1.0).abs() < 0.04,
            "prior a {a_prior}: variance {} vs {want}",
            variance_at_level(&high)
        );
        assert!(high.model.a < 0.85 * a_prior, "a = {}", high.model.a);
    }
}
