//! Synthetic validation harness for the noise estimator (`docs/06-noise.md` section Validation).
//!
//! Run in release mode, for example:
//!
//! ```text
//! cargo run --release -p ym-noise --example validate_synthetic -- abs 60
//! cargo run --release -p ym-noise --example validate_synthetic -- all 60
//! ```
//!
//! The corpus is not reachable in V1 builds, so the truth here is a known Poisson-Gaussian model
//! applied to procedural scenes (`ym_noise::model::synth`). Modes: `abs` (absolute recovery of `a`
//! and `sigma`), `nn` (noise on noise), `filt` (filtered-file detector), `busy` (texture fallback),
//! `clip` (clipping bias), `row` (row noise), `refine` (YOND-style refinement), `dbg <seed>`.

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use ym_noise::estimate::EstimatorConfig;
use ym_noise::model::synth::SceneKind;
use ym_noise::priors::NoisePrior;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("abs");
    let n: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let first: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    match mode {
        "abs" => {
            let kind = match args.get(3).map(String::as_str) {
                Some("busy") => SceneKind::Busy,
                Some("smooth") => SceneKind::Smooth,
                _ => SceneKind::Mixed,
            };
            let exposure: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let row = args.get(5).is_some_and(|s| s == "row");
            absolute(n, first, kind, exposure, row)
        }
        "dbg" => debug_case(first),
        "pca" => pca_probe(n, first),
        "clip" => clipping(
            n,
            first,
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0),
        ),
        "ramp" => clip_ramp(n),
        "row" => row_noise(n, first),
        "refine" => refinement(n, first, args.get(3).map(String::as_str).unwrap_or("mixed")),
        "probetime" => probe_time(),
        "delta" => delta_recovery(n, first),
        "nn" => noise_on_noise(n, first, args.get(3).map(String::as_str).unwrap_or("mixed")),
        "filt" => filtered_detector(
            n,
            first,
            args.get(3)
                .map(String::as_str)
                .unwrap_or("smooth,mixed,busy"),
        ),
        other => eprintln!("unknown mode {other}"),
    }
}

fn absolute(n: u64, first: u64, kind: SceneKind, exposure: f32, row: bool) {
    let mut cfg = EstimatorConfig::default();
    if std::env::var_os("YM_NO_FILTER").is_some() {
        cfg.filtered_check = false;
    }
    if let Some(n) = std::env::var("YM_STAB").ok().and_then(|v| v.parse().ok()) {
        cfg.stabilise_iterations = n;
    }
    let (w, h) = (1536, 1024);
    let (mut ea, mut es) = (Vec::new(), Vec::new());
    println!(
        "seed    iso      a_true     a_est   err_a   err_s(0.02) err_s(0.18) err_s(0.5)  bins fb conf"
    );
    for seed in first..first + n {
        let case = make_case(seed, kind, w, h, exposure, row);
        let m = noisy(&case);
        let est = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg);
        let (ea1, es1) = errors(&est.model, &case.truth);
        println!(
            "{seed:5} {:7.0} {:10.3e} {:10.3e} {:+6.1}% {:+8.1}% {:+8.1}% {:+8.1}%   {:2} {} {:?} ratio {:.3}",
            case.iso,
            case.truth.a,
            est.model.a,
            ea1 * 100.0,
            es1[0] * 100.0,
            es1[1] * 100.0,
            es1[2] * 100.0,
            est.diagnostics.bins_surviving,
            est.diagnostics.texture_fallback as u8,
            est.diagnostics.confidence,
            est.diagnostics.filtered.ratio
        );
        println!(
            "        rms {:.3} high {:.3} probe {:?}",
            est.diagnostics.residual_rms,
            est.diagnostics.high_fraction,
            est.diagnostics.probe_ratio
        );
        println!(
            "        sigma_row true {:.2e} est {:.2e} (measured {})",
            case.truth.sigma_row,
            est.model.sigma_row / est.diagnostics.filtered.factor,
            est.diagnostics.row_noise.measured
        );
        ea.push(ea1);
        es.extend(es1);
    }
    let (m, p) = abs_stats(&ea);
    let mut signed = ea.clone();
    println!(
        "a: median |err| {:.1}%  p90 {:.1}%  median signed {:+.1}%",
        m * 100.0,
        p * 100.0,
        median(&mut signed) * 100.0
    );
    let (m, p) = abs_stats(&es);
    println!(
        "sigma: median |err| {:.1}%  p90 {:.1}%",
        m * 100.0,
        p * 100.0
    );
}

fn debug_case(seed: u64) {
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    let kind = match std::env::var("YM_KIND").as_deref() {
        Ok("smooth") => SceneKind::Smooth,
        Ok("busy") => SceneKind::Busy,
        _ => SceneKind::Mixed,
    };
    let case = make_case(seed, kind, w, h, 1.0, false);
    println!("truth {:?} iso {}", case.truth, case.iso);
    let mut m = noisy(&case);
    use ym_noise::model::synth::{RawFilter, apply_raw_filter};
    match std::env::var("YM_FILTER").as_deref() {
        Ok("box1") => {
            m = apply_raw_filter(&m, w, h, RawFilter::BoxBlend { weight: 1.0 }, case.truth.q)
        }
        Ok("median") => m = apply_raw_filter(&m, w, h, RawFilter::Median3, case.truth.q),
        Ok("h121") => m = apply_raw_filter(&m, w, h, RawFilter::Horizontal121, case.truth.q),
        _ => {}
    }
    let est = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg);
    println!("model {:?}", est.model);
    println!(
        "filtered {:?} row {:?}",
        est.diagnostics.filtered, est.diagnostics.row_noise
    );
    println!(
        "diag: surviving {} textured {} clipped {} cov {:.2} fallback {} kept {}/{} resid {:.3} passes {} high {:.3}",
        est.diagnostics.bins_surviving,
        est.diagnostics.bins_textured,
        est.diagnostics.clipped_bins_used,
        est.diagnostics.coverage,
        est.diagnostics.texture_fallback,
        est.diagnostics.windows_kept,
        est.diagnostics.windows_total,
        est.diagnostics.residual_rms,
        est.diagnostics.passes,
        est.diagnostics.high_fraction
    );
    for (k, b) in est.diagnostics.bins.iter().enumerate() {
        println!(
            "bin {k:2}: samples {:7} tail {:5.2} textured {} clipped {} survived {}",
            b.samples, b.tail_ratio, b.textured as u8, b.clipped as u8, b.survived as u8
        );
    }
    for p in &est.points {
        let want = case
            .truth
            .pixel_variance(p.mean, p.plane.unwrap_or(0) as usize);
        println!(
            "pt plane {:?} bin {:2} {:?} mean {:.4} var {:.3e} want {:.3e} ratio {:.3} n {}",
            p.plane,
            p.bin,
            p.kind,
            p.mean,
            p.var,
            want,
            p.var / want,
            p.count
        );
    }
}

/// Experiment: the patch-PCA estimate by itself, on the scene kind in `YM_KIND` (default busy) with
/// the filter in `YM_FILTER` (box1, box08, median, h121) applied to the noisy mosaic.
fn pca_probe(n: u64, first: u64) {
    use ym_noise::estimate::{F32Plane, bins::BinLayout, pca, tiles};
    use ym_noise::model::synth::{RawFilter, apply_raw_filter};
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    let kind = match std::env::var("YM_KIND").as_deref() {
        Ok("smooth") => SceneKind::Smooth,
        Ok("mixed") => SceneKind::Mixed,
        _ => SceneKind::Busy,
    };
    for seed in first..first + n {
        let case = make_case(seed, kind, w, h, 1.0, false);
        let mut m = noisy(&case);
        match std::env::var("YM_FILTER").as_deref() {
            Ok("box1") => {
                m = apply_raw_filter(&m, w, h, RawFilter::BoxBlend { weight: 1.0 }, case.truth.q)
            }
            Ok("box08") => {
                m = apply_raw_filter(&m, w, h, RawFilter::BoxBlend { weight: 0.8 }, case.truth.q)
            }
            Ok("median") => m = apply_raw_filter(&m, w, h, RawFilter::Median3, case.truth.q),
            Ok("h121") => m = apply_raw_filter(&m, w, h, RawFilter::Horizontal121, case.truth.q),
            _ => {}
        }
        let src = F32Plane::new(&m, w, h).unwrap();
        let plan = tiles::plan(w, h, &cfg).unwrap();
        let data: Vec<_> = plan
            .origins
            .iter()
            .map(|&o| tiles::load_tile(&src, None, o, plan.size))
            .collect();
        let est = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg);
        let pc = pca::PcaConfig {
            a: case.truth.a,
            b: case.truth.b,
            patches_per_bin_tile: 160,
            stride: 3,
            bin_mask: u32::MAX,
        };
        let pts = pca::texture_points(&data, &plan, &BinLayout::default(), &pc);
        let mut ratios: Vec<f64> = pts
            .iter()
            .filter(|p| p.mean < 0.85)
            .map(|p| p.var / pixel_model(&est.model).variance(p.mean, 0))
            .collect();
        let conf = pts.iter().filter(|p| p.confident).count();
        println!(
            "seed {seed} iso {:6.0}: {} bins ({} confident), PCA/fit median {:.2}; filtered ratio {:.3}",
            case.iso,
            pts.len(),
            conf,
            median(&mut ratios),
            est.diagnostics.filtered.ratio
        );
    }
}

/// Filtered-file detector: false positives on white noise, detection rate on filtered noise,
/// per scene kind.
fn filtered_detector(n: u64, first: u64, kinds: &str) {
    use ym_noise::model::synth::{RawFilter, apply_raw_filter};
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    let filters: [(&str, Option<RawFilter>); 8] = [
        ("white", None),
        ("median3 (R~0.57)", Some(RawFilter::Median3)),
        (
            "box 3x3 w=1.0 (R~0.66)",
            Some(RawFilter::BoxBlend { weight: 1.0 }),
        ),
        (
            "box 3x3 w=0.6 (R~0.45)",
            Some(RawFilter::BoxBlend { weight: 0.6 }),
        ),
        (
            "box 3x3 w=0.4 (R~0.34)",
            Some(RawFilter::BoxBlend { weight: 0.4 }),
        ),
        (
            "horizontal [1 2 1]/4 (R~0.40)",
            Some(RawFilter::Horizontal121),
        ),
        (
            "box 3x3 w=0.8 (R~0.64)",
            Some(RawFilter::BoxBlend { weight: 0.8 }),
        ),
        ("median3 then box w=0.4", None),
    ];
    for (kind, name) in [
        (SceneKind::Smooth, "smooth"),
        (SceneKind::Mixed, "mixed"),
        (SceneKind::Busy, "busy"),
    ] {
        if !kinds.contains(name) {
            continue;
        }
        println!("== {name} scenes, {n} cases");
        let mut flagged = vec![0usize; filters.len()];
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); filters.len()];
        let mut evaluated = vec![0usize; filters.len()];
        for seed in first..first + n {
            let case = make_case(seed, kind, w, h, 1.0, seed % 2 == 0);
            let m = noisy(&case);
            for (i, (label, f)) in filters.iter().enumerate() {
                let data = match f {
                    None if label.starts_with("median3 then") => {
                        let t = apply_raw_filter(&m, w, h, RawFilter::Median3, case.truth.q);
                        apply_raw_filter(
                            &t,
                            w,
                            h,
                            RawFilter::BoxBlend { weight: 0.4 },
                            case.truth.q,
                        )
                    }
                    None => m.clone(),
                    Some(f) => apply_raw_filter(&m, w, h, *f, case.truth.q),
                };
                let est = run(&data, w, h, case.truth.q, &NoisePrior::none(), &cfg);
                let d = &est.diagnostics.filtered;
                flagged[i] += d.flagged as usize;
                evaluated[i] += d.evaluated as usize;
                if d.evaluated {
                    ratios[i].push(d.ratio);
                }
            }
        }
        for (i, (name, _)) in filters.iter().enumerate() {
            let mut r = ratios[i].clone();
            println!(
                "  {name:30} flagged {:3}/{n} = {:5.1}%   evaluated {:3}   median ratio {:.3}",
                flagged[i],
                100.0 * flagged[i] as f64 / n as f64,
                evaluated[i],
                median(&mut r)
            );
        }
    }
}

/// Noise on noise: add a known Poisson-Gaussian increment to an already noisy mosaic and recover
/// it as the difference of two estimates (`docs/06-noise.md` section Validation).
fn noise_on_noise(n: u64, first: u64, kind: &str) {
    use ym_noise::model::synth::{SynthRng, add_increment};
    let kind = match kind {
        "busy" => SceneKind::Busy,
        "smooth" => SceneKind::Smooth,
        _ => SceneKind::Mixed,
    };
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    let (mut e_a, mut e_v) = (Vec::new(), Vec::new());
    let mut by_ratio: Vec<(f64, f64)> = Vec::new();
    println!("seed   iso   ratio   a_inc      d_a_est    err_a    err_dvar(0.02) (0.18)  (0.5)");
    for seed in first..first + n {
        let case = make_case(seed, kind, w, h, 1.0, false);
        let base = noisy(&case);
        let est0 = run(&base, w, h, case.truth.q, &NoisePrior::none(), &cfg);
        let mut rng = SynthRng::new(seed ^ 0xFEED);
        let ratio = (0.5f64.ln() + (3.0f64.ln() - 0.5f64.ln()) * rng.uniform()).exp();
        let (a_inc, b_inc) = (ratio * case.truth.a, ratio * ratio * case.truth.b);
        let mut aug = base.clone();
        add_increment(&mut aug, w, a_inc, b_inc, case.truth.q, seed ^ 0xBEEF);
        let est1 = run(&aug, w, h, case.truth.q, &NoisePrior::none(), &cfg);
        let (m0, m1) = (pixel_model(&est0.model), pixel_model(&est1.model));
        let da = m1.a - m0.a;
        let err_a = da / a_inc - 1.0;
        let mut ev = [0.0; 3];
        for (k, &x) in PROBES.iter().enumerate() {
            let want = a_inc * x + b_inc;
            ev[k] = (m1.variance(x, 0) - m0.variance(x, 0)) / want - 1.0;
        }
        println!(
            "{seed:5} {:6.0} {ratio:6.2} {a_inc:10.3e} {da:10.3e} {:+7.1}%  {:+8.1}% {:+8.1}% {:+8.1}%",
            case.iso,
            err_a * 100.0,
            ev[0] * 100.0,
            ev[1] * 100.0,
            ev[2] * 100.0
        );
        e_a.push(err_a);
        e_v.extend(ev);
        by_ratio.push((ratio, err_a.abs()));
    }
    let (m, p) = abs_stats(&e_a);
    println!(
        "increment a: median |err| {:.1}%  p90 {:.1}%",
        m * 100.0,
        p * 100.0
    );
    let (m, p) = abs_stats(&e_v);
    println!(
        "increment variance at probes: median |err| {:.1}%  p90 {:.1}%",
        m * 100.0,
        p * 100.0
    );
    for (lo, hi) in [(0.5, 1.0), (1.0, 2.0), (2.0, 3.01)] {
        let sel: Vec<f64> = by_ratio
            .iter()
            .filter(|r| r.0 >= lo && r.0 < hi)
            .map(|r| r.1)
            .collect();
        let mut s = sel.clone();
        println!(
            "  a_inc/a in [{lo}, {hi}): {} cases, median |err_a| {:.1}%  p90 {:.1}%",
            sel.len(),
            median(&mut s) * 100.0,
            percentile(&mut s, 0.9) * 100.0
        );
    }
}

/// Clipping: scenes pushed over the clip level at high gain; the clip-aware estimator against the
/// same estimator with the clipped-Gaussian treatment switched off.
fn clipping(n: u64, first: u64, exposure: f32) {
    let (w, h) = (1536, 1024);
    let aware = EstimatorConfig::default();
    let naive = EstimatorConfig {
        clip_aware: false,
        ..EstimatorConfig::default()
    };
    let (mut ea, mut en) = (Vec::new(), Vec::new());
    println!("seed   iso  clipped%   a_true     err_aware  err_naive   clipped bins used");
    for seed in first..first + n {
        let mut case = make_case(seed, SceneKind::Mixed, w, h, exposure, false);
        // High gain makes the noise wide enough for clipping to reach the mid-bright bins.
        let mut rng = ym_noise::model::synth::SynthRng::new(seed ^ 0x77);
        case.iso = 1600.0 * 2f64.powf(rng.uniform() * 4.0);
        case.truth = random_camera(&mut rng, case.iso, false);
        let m = noisy(&case);
        let clipped = m.iter().filter(|&&v| v >= 0.99999).count() as f64 / m.len() as f64;
        let e1 = run(&m, w, h, case.truth.q, &NoisePrior::none(), &aware);
        let e2 = run(&m, w, h, case.truth.q, &NoisePrior::none(), &naive);
        let (a1, a2) = (
            e1.model.a / case.truth.a - 1.0,
            e2.model.a / case.truth.a - 1.0,
        );
        println!(
            "{seed:5} {:6.0} {:7.1}%  {:10.3e}  {:+7.1}%   {:+7.1}%   {}",
            case.iso,
            clipped * 100.0,
            case.truth.a,
            a1 * 100.0,
            a2 * 100.0,
            e1.diagnostics.clipped_bins_used
        );
        ea.push(a1);
        en.push(a2);
    }
    for (name, v) in [("clip-aware", &ea), ("naive", &en)] {
        let (m, p) = abs_stats(v);
        let mut sv = v.clone();
        println!(
            "{name:11}: median signed {:+.1}%  median |err| {:.1}%  p90 {:.1}%",
            median(&mut sv) * 100.0,
            m * 100.0,
            p * 100.0
        );
    }
}

/// Controlled clipping stress: a luminance ramp that runs well past the clip level, so many bins
/// sit near white and partly clipped.
fn clip_ramp(n: u64) {
    use ym_noise::model::synth::{SynthRng, add_noise};
    let (w, h) = (1536, 1024);
    let aware = EstimatorConfig::default();
    let naive = EstimatorConfig {
        clip_aware: false,
        ..EstimatorConfig::default()
    };
    for &(a, top) in &[
        (1.0e-3, 1.4f32),
        (3.0e-3, 1.5),
        (8.0e-3, 1.6),
        (3.0e-3, 2.5),
    ] {
        let (mut ea, mut en) = (Vec::new(), Vec::new());
        for seed in 0..n {
            let mut rng = SynthRng::new(seed + 555);
            let truth = ym_noise::model::NoiseModel {
                a,
                b: (a * 3.0f64).powi(2),
                delta: [0.0; 4],
                sigma_row: 0.0,
                q: DN,
            };
            let signal: Vec<f32> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let g = match (y & 1, x & 1) {
                        (0, 0) => 0.9,
                        (1, 1) => 0.8,
                        _ => 1.0,
                    };
                    (0.02 + (top - 0.02) * (x as f32 / w as f32) + 0.05 * (y as f32 / h as f32)) * g
                })
                .collect();
            let m = add_noise(
                &signal,
                w,
                h,
                &truth,
                1000 + seed + (rng.uniform() * 1e6) as u64,
            );
            let e1 = run(&m, w, h, truth.q, &NoisePrior::none(), &aware);
            let e2 = run(&m, w, h, truth.q, &NoisePrior::none(), &naive);
            ea.push(e1.model.a / a - 1.0);
            en.push(e2.model.a / a - 1.0);
        }
        let (mut sa, mut sn) = (ea.clone(), en.clone());
        println!(
            "a = {a:.0e}, ramp to {top}: clip-aware median signed {:+.1}%   naive {:+.1}%",
            median(&mut sa) * 100.0,
            median(&mut sn) * 100.0
        );
    }
}

/// Row noise: recovery of sigma_row relative to the read noise, on smooth and mixed scenes.
fn row_noise(n: u64, first: u64) {
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    for kind in [SceneKind::Smooth, SceneKind::Mixed] {
        println!("== {kind:?}");
        for &rel in &[0.0, 0.25, 0.5, 1.0, 2.0] {
            let mut ratios = Vec::new();
            let mut absolute = Vec::new();
            for seed in first..first + n {
                let mut case = make_case(seed, kind, w, h, 1.0, false);
                // Low to moderate ISO, where row noise can show against the shot noise.
                let mut rng = ym_noise::model::synth::SynthRng::new(seed ^ 0x99);
                case.iso = 100.0 * 2f64.powf(rng.uniform() * 5.0);
                case.truth = random_camera(&mut rng, case.iso, false);
                case.truth.sigma_row = rel * case.truth.b.sqrt();
                let m = noisy(&case);
                let est = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg);
                let measured =
                    est.diagnostics.row_noise.sigma_row / est.diagnostics.filtered.factor;
                if rel > 0.0 {
                    ratios.push(measured / case.truth.sigma_row);
                }
                absolute.push(measured / case.truth.b.sqrt());
            }
            let mut a = absolute.clone();
            if rel > 0.0 {
                let mut r = ratios.clone();
                println!(
                    "  sigma_row = {rel:.2} * read noise: est/true median {:.2}  p10 {:.2}  p90 {:.2}",
                    median(&mut r),
                    percentile(&mut r.clone(), 0.1),
                    percentile(&mut r, 0.9)
                );
            } else {
                println!(
                    "  sigma_row = 0: estimated / read noise median {:.3}  p90 {:.3}  max {:.3}",
                    median(&mut a),
                    percentile(&mut a.clone(), 0.9),
                    percentile(&mut a, 1.0)
                );
            }
        }
    }
}

/// YOND-style refinement, synthetic only. Two stand-in denoisers: an "oracle" whose output is the
/// clean signal plus 15% of an independent noise draw (the case the published formula assumes),
/// and a bilateral filter with a 5-sigma range kernel (a real but crude denoiser).
fn refinement(n: u64, first: u64, kind: &str) {
    use ym_noise::estimate::{F32Plane, refine};
    use ym_noise::model::synth::{add_noise, render_scene};
    let kind = match kind {
        "busy" => SceneKind::Busy,
        "smooth" => SceneKind::Smooth,
        _ => SceneKind::Mixed,
    };
    let cfg = EstimatorConfig::default();
    let (w, h) = (768, 512);
    let mut rows: Vec<(f64, f64, f64, f64)> = Vec::new();
    println!("seed    iso   err_s(0.18): first   oracle-refined   bilateral-refined");
    for seed in first..first + n {
        let case = make_case(seed, kind, w, h, 1.0, false);
        let scene = render_scene(&case.spec);
        let m = add_noise(&scene.signal, w, h, &case.truth, case.noise_seed);
        let est0 = run(&m, w, h, case.truth.q, &NoisePrior::none(), &cfg);
        // Oracle: clean signal (plus offsets) and a small independent noise draw.
        let small = ym_noise::model::NoiseModel {
            a: 0.0225 * case.truth.a,
            b: 0.0225 * case.truth.b,
            sigma_row: 0.0,
            q: 0.0,
            ..case.truth
        };
        let oracle = add_noise(&scene.signal, w, h, &small, case.noise_seed ^ 0xFFFF);
        let bil = bilateral_denoise(&m, w, h, &est0.model, 5.0);
        let np = F32Plane::new(&m, w, h).unwrap();
        let r_o = refine(
            &est0,
            &np,
            &F32Plane::new(&oracle, w, h).unwrap(),
            &NoisePrior::none(),
            &cfg,
        );
        let r_b = refine(
            &est0,
            &np,
            &F32Plane::new(&bil, w, h).unwrap(),
            &NoisePrior::none(),
            &cfg,
        );
        let s0 = errors(&est0.model, &case.truth).1[1];
        let so = r_o
            .as_ref()
            .map(|e| errors(&e.model, &case.truth).1[1])
            .unwrap_or(f64::NAN);
        let sb = r_b
            .as_ref()
            .map(|e| errors(&e.model, &case.truth).1[1])
            .unwrap_or(f64::NAN);
        println!(
            "{seed:5} {:6.0}     {:+7.1}%      {:+7.1}%        {:+7.1}%",
            case.iso,
            s0 * 100.0,
            so * 100.0,
            sb * 100.0
        );
        rows.push((s0, so, sb, 0.0));
    }
    let col = |i: usize| -> Vec<f64> { rows.iter().map(|r| [r.0, r.1, r.2][i]).collect() };
    for (name, i) in [
        ("first estimate", 0),
        ("oracle-refined", 1),
        ("bilateral-refined", 2),
    ] {
        let (m, p) = abs_stats(&col(i));
        let mut c = col(i);
        println!(
            "{name:18} sigma(0.18): median |err| {:.1}%  p90 {:.1}%  median signed {:+.1}%",
            m * 100.0,
            p * 100.0,
            median(&mut c) * 100.0
        );
    }
}

/// Timing of the probe pieces on one scene (single thread when RAYON_NUM_THREADS=1).
fn probe_time() {
    use ym_noise::estimate::{F32Plane, bins::BinLayout, pca, tiles};
    let cfg = EstimatorConfig::default();
    let (w, h) = (2048, 1536);
    let case = make_case(31, SceneKind::Mixed, w, h, 1.0, false);
    let m = noisy(&case);
    let src = F32Plane::new(&m, w, h).unwrap();
    let plan = tiles::plan(w, h, &cfg).unwrap();
    let data: Vec<_> = plan
        .origins
        .iter()
        .map(|&o| tiles::load_tile(&src, None, o, plan.size))
        .collect();
    let layout = BinLayout::default();
    for (name, mask, per_tile, stride) in [
        (
            "3 bins, 64/tile, stride 6",
            0b0000_0000_0000_1110_0000_0000u32,
            64usize,
            6usize,
        ),
        (
            "3 bins, 32/tile, stride 6",
            0b0000_0000_0000_1110_0000_0000u32,
            32,
            6,
        ),
        ("all bins, 160/tile, stride 3", u32::MAX, 160, 3),
    ] {
        let pc = pca::PcaConfig {
            a: case.truth.a,
            b: case.truth.b,
            patches_per_bin_tile: per_tile,
            stride,
            bin_mask: mask,
        };
        let t0 = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..10 {
            n = pca::texture_points(&data, &plan, &layout, &pc).len();
        }
        println!(
            "{name:32}: {:.1} ms per call ({n} bins)",
            t0.elapsed().as_secs_f64() * 100.0
        );
    }
}

/// Recovery of the plane offsets delta_c (compared after removing their mean, which the data
/// cannot see), on smooth scenes at low and moderate ISO.
fn delta_recovery(n: u64, first: u64) {
    let cfg = EstimatorConfig::default();
    let (w, h) = (1536, 1024);
    let (mut err, mut spread, mut b_err) = (Vec::new(), Vec::new(), Vec::new());
    for seed in first..first + n {
        let mut case = make_case(seed, SceneKind::Smooth, w, h, 1.0, false);
        let mut rng = ym_noise::model::synth::SynthRng::new(seed ^ 0x1234);
        case.iso = 100.0 * 2f64.powf(rng.uniform() * 4.0);
        case.truth = random_camera(&mut rng, case.iso, false);
        // Offsets of a few DN, as the ELD calibration of the A7S II shows.
        case.truth.delta = std::array::from_fn(|_| rng.normal() * 3.0 * DN);
        let est = run(&noisy(&case), w, h, case.truth.q, &NoisePrior::none(), &cfg);
        let mean_t = case.truth.delta.iter().sum::<f64>() / 4.0;
        let mean_e = est.model.delta.iter().sum::<f64>() / 4.0;
        for c in 0..4 {
            let (t, e) = (case.truth.delta[c] - mean_t, est.model.delta[c] - mean_e);
            err.push((e - t) / DN);
            spread.push(t / DN);
        }
        // Variance at the darkest level, where the offsets act through b - a delta.
        let x = 0.003;
        for c in 0..4 {
            b_err.push(est.model.pixel_variance(x, c) / case.truth.pixel_variance(x, c) - 1.0);
        }
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    println!(
        "delta_c (DN, mean removed): true spread rms {:.2}, estimation error rms {:.2}; per-plane variance at 0.003: median |err| {:.1}%",
        rms(&spread),
        rms(&err),
        median(&mut b_err.iter().map(|x| x.abs()).collect::<Vec<_>>()) * 100.0
    );
}
