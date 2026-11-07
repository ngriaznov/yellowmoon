//! Times the Depth Anything V2-Small runtime on seeded random weights (the cost does not depend on the values):
//! `cargo run --release -p ym-ml --example depth_bench [-- WIDTH HEIGHT]` (default 784 x 518), on all threads and on one.

use std::time::Instant;

use ym_ml::depth::{DepthAnything, multiply_adds, synthetic_weights};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (
        args.first().copied().unwrap_or(784),
        args.get(1).copied().unwrap_or(518),
    );
    let t = Instant::now();
    let model = DepthAnything::from_bytes(&synthetic_weights(1), None).expect("loads");
    println!(
        "synthetic weights written and loaded in {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let rgb: Vec<f32> = (0..w * h * 3)
        .map(|i| ((i * 37 % 1001) as f32 / 500.0) - 1.0)
        .collect();
    let gmac = multiply_adds(w, h) as f64 / 1e9;
    println!("{w} x {h}: {gmac:.1} GMAC");
    for threads in [rayon::current_num_threads(), 1] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let _ = model.infer(&rgb, w, h).expect("warm-up");
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let _ = model.infer(&rgb, w, h).expect("runs");
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{threads} threads: best of 3 {best:.2} s ({:.1} GMAC/s)",
                gmac / best
            );
        });
    }
}
