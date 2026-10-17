//! Timing of the adaptive ratio demosaic against Menon and LMMSE.
//!
//! Run: `cargo run -p ym-gpu --example adaptive_ratio_timing [-- <megapixels>]` (default 8 MP).
//! Add `--release` for numbers that mean something for shipping code; the dev profile of this
//! workspace is `opt-level = 1` for the workspace crates.
//!
//! Each row is the best of three runs, in wall seconds per megapixel and, on Linux, CPU seconds per megapixel (user + system
//! time of the whole process from `/proc/self/stat`, 10 ms resolution): on a busy machine only the
//! CPU column is dependable.
//!
//! * `whole` is [`adaptive_ratio::demosaic_whole`]: one piece, one thread.
//! * `banded` is [`adaptive_ratio::demosaic`] / [`demosaic::demosaic_with`]: bands of
//!   [`ym_gpu::cpu::filters::BAND_ROWS`] rows in parallel, on a pool of 1 thread and on the default
//!   pool.

use std::time::Instant;

use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn};
use ym_noise::model::NoiseModel;

/// How many times each measurement is repeated (the best run is reported).
const REPEATS: usize = 3;

/// User + system CPU seconds of this process so far (Linux), 10 ms resolution.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may contain spaces; the fields after it start at ") ".
    let rest = &stat[stat.rfind(") ")? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // utime and stime are fields 14 and 15 of the whole line: indices 11 and 12 after the name.
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

/// Runs `f` [`REPEATS`] times and prints the best wall and CPU seconds per megapixel.
fn time<T>(label: &str, mp: f64, mut f: impl FnMut() -> T) -> T {
    let (mut best_wall, mut best_cpu) = (f64::INFINITY, f64::INFINITY);
    let mut last = None;
    for _ in 0..REPEATS {
        let (cpu0, t0) = (cpu_seconds(), Instant::now());
        last = Some(f());
        best_wall = best_wall.min(t0.elapsed().as_secs_f64());
        if let Some(c) = cpu0.zip(cpu_seconds()).map(|(a, b)| b - a) {
            best_cpu = best_cpu.min(c);
        }
    }
    if best_cpu.is_finite() {
        println!(
            "  {label:<34} {:>7.3} s/MP wall {:>7.3} s/MP cpu",
            best_wall / mp,
            best_cpu / mp
        );
    } else {
        println!("  {label:<34} {:>7.3} s/MP wall", best_wall / mp);
    }
    last.expect("at least one run")
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
}

fn main() {
    let mp_arg: f64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(8.0);
    let h = ((mp_arg * 1e6 / 1.5).sqrt() as usize) & !1;
    let w = (h * 3 / 2) & !1;
    let mp = (w * h) as f64 / 1e6;
    // A tile of a textured scene repeated over the frame (generating a natural scene of this
    // size would dominate the run), with a little noise on top.
    let (tw, th) = (512usize, 512usize);
    let tile = synth::mosaic_from_rgb(&synth::natural(tw, th, 3), Cfa::RGGB);
    let mut m = Mosaic::new(w, h, Cfa::RGGB);
    for y in 0..h {
        for x in 0..w {
            m.data[y * w + x] = 0.4 * tile.data[(y % th) * tw + x % tw];
        }
    }
    synth::add_noise(&mut m.data, 0.002, 9);
    let low = NoiseDn::from_model(&NoiseModel {
        a: 2e-5,
        b: 1e-8,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    });
    let inputs = DemosaicInputs {
        gains: [2.0, 1.0, 1.6],
        noise: low,
        ca: None,
    };
    println!(
        "== {w} x {h} ({mp:.1} MP), {} hardware threads",
        rayon::current_num_threads()
    );
    let one = pool(1);
    let plain = Params::default();
    let matching = Params {
        shadow_match: ShadowMatch::On,
        ..Params::default()
    };
    for (name, params) in [("plain", plain), ("shadow matching", matching)] {
        println!("adaptive ratio, {name}:");
        let a = time("whole (1 thread)", mp, || {
            adaptive_ratio::demosaic_whole(&m, &inputs, &params).expect("run")
        });
        let b = time("banded, 1-thread pool", mp, || {
            one.install(|| adaptive_ratio::demosaic(&m, &inputs, &params).expect("run"))
        });
        let c = time("banded, default pool", mp, || {
            adaptive_ratio::demosaic(&m, &inputs, &params).expect("run")
        });
        assert!(a == b && b == c, "the three runs must be bit-identical");
    }
    println!("existing methods (banded):");
    for method in [Method::Bilinear, Method::Menon, Method::Lmmse] {
        time(&format!("{}, 1-thread pool", method.name()), mp, || {
            one.install(|| demosaic::demosaic(&m, method))
        });
        time(&format!("{}, default pool", method.name()), mp, || {
            demosaic::demosaic(&m, method)
        });
    }
}
