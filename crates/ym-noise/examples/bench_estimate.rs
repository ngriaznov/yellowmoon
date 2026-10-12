//! Timing of the noise estimator on a 50 MP synthetic mosaic (2 MP crop grid).
//!
//! `cargo run --release -p ym-noise --example bench_estimate -- [reps] [threads]`
//!
//! The mosaic is a `u16` RGGB frame (like the decoder's output) built from a procedural scene with
//! known Poisson-Gaussian noise; the estimator reads it through `U16Mosaic`, so the timing includes
//! R1 normalisation of the sampled tiles. The target in `docs/06-noise.md` is below 80 ms per file
//! on an Apple M1; this machine is not one, so the numbers are indicative only.

use std::time::Instant;

use rayon::prelude::*;
use ym_noise::estimate::{EstimateInput, EstimatorConfig, U16Mosaic, estimate_timed};
use ym_noise::model::NoiseModel;
use ym_noise::model::synth::{SceneKind, SceneSpec, SynthRng, add_noise, render_scene};
use ym_noise::priors::NoisePrior;

const WIDTH: usize = 8192;
const HEIGHT: usize = 6144;
const BLACK: f32 = 512.0;
const WHITE: f32 = 16383.0;

/// Builds the 50 MP frame band by band (the scene generator is a pure function of its seed and the
/// band's rows, so bands are generated independently and in parallel).
fn make_frame(truth: &NoiseModel) -> Vec<u16> {
    let band = 512;
    let mut raw = vec![0u16; WIDTH * HEIGHT];
    raw.par_chunks_mut(WIDTH * band)
        .enumerate()
        .for_each(|(i, chunk)| {
            let spec = SceneSpec {
                width: WIDTH,
                height: band,
                seed: 7000 + i as u64,
                kind: SceneKind::Mixed,
                exposure: 1.0,
            };
            let scene = render_scene(&spec);
            let noisy = add_noise(&scene.signal, WIDTH, band, truth, 99 + i as u64);
            for (o, &v) in chunk.iter_mut().zip(&noisy) {
                *o = (BLACK + v * (WHITE - BLACK)).round().clamp(0.0, 65535.0) as u16;
            }
        });
    raw
}

/// CPU time (user + system) of this process in milliseconds, from `/proc/self/stat`
/// (10 ms ticks); `None` where `/proc` is not available. Unlike wall-clock time it does not grow
/// when other processes compete for the cores.
fn cpu_ms() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesised command name; utime and stime are fields 14 and 15.
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let (u, s): (f64, f64) = (f.get(11)?.parse().ok()?, f.get(12)?.parse().ok()?);
    Some((u + s) * 10.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    if let Some(t) = args.get(1).and_then(|s| s.parse::<usize>().ok()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("thread pool");
    }
    let mut rng = SynthRng::new(3);
    let truth = NoiseModel {
        a: 1.2e-3,
        b: 4.0e-7,
        delta: std::array::from_fn(|_| rng.normal() * 1.0e-4),
        sigma_row: 0.0,
        q: 1.0 / f64::from(WHITE - BLACK),
    };
    let t0 = Instant::now();
    let raw = make_frame(&truth);
    println!(
        "built a {WIDTH} x {HEIGHT} frame ({:.1} MP) in {:.1} s",
        (WIDTH * HEIGHT) as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let mosaic = U16Mosaic::new(&raw, WIDTH, HEIGHT, [BLACK; 4], [WHITE; 4]).expect("mosaic");
    let cfg = EstimatorConfig::default();
    let input = EstimateInput {
        mosaic: &mosaic,
        qstep: None,
        q_norm: truth.q,
    };
    let mut times = Vec::new();
    let mut last = None;
    let cpu0 = cpu_ms();
    let mut stage_min = [f64::MAX; 5];
    for _ in 0..reps {
        let t = Instant::now();
        let (est, st) = estimate_timed(&input, &NoisePrior::none(), &cfg).expect("estimate");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        stage_min = [
            stage_min[0].min(st.read_ms),
            stage_min[1].min(st.windows_ms),
            stage_min[2].min(st.passes_ms),
            stage_min[3].min(st.fallback_ms),
            stage_min[4].min(st.extras_ms),
        ];
        last = Some(est);
    }
    if let (Some(c0), Some(c1)) = (cpu0, cpu_ms()) {
        println!(
            "CPU time per estimate (all threads): {:.1} ms",
            (c1 - c0) / reps as f64
        );
    }
    times.sort_by(f64::total_cmp);
    let est = last.expect("at least one run");
    println!(
        "threads {}: estimate over {} runs: min {:.1} ms, median {:.1} ms, max {:.1} ms",
        rayon::current_num_threads(),
        reps,
        times[0],
        times[times.len() / 2],
        times[times.len() - 1]
    );
    println!(
        "best per stage: read {:.1} windows {:.1} passes {:.1} fallback {:.1} extras {:.1} ms",
        stage_min[0], stage_min[1], stage_min[2], stage_min[3], stage_min[4]
    );
    println!(
        "a {:.4e} (truth {:.4e}), b {:.3e} (truth {:.3e}), bins {} fallback {} filtered {}",
        est.model.a,
        truth.a,
        est.model.b,
        truth.b,
        est.diagnostics.bins_surviving,
        est.diagnostics.texture_fallback,
        est.diagnostics.filtered.flagged
    );
}
