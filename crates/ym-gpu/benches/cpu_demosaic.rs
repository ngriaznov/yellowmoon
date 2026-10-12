//! Demosaic timings: the three classical methods on 24 MP and 50 MP mosaics.
//!
//! Run: `cargo bench -p ym-gpu --bench cpu_demosaic` (optionally with `24` or `50`).

use std::time::Instant;

use ym_gpu::cpu::Cfa;
use ym_gpu::cpu::demosaic::{self, Method};
use ym_gpu::cpu::synth;

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let sizes: Vec<f64> = if args.is_empty() {
        vec![24.0, 50.0]
    } else {
        args.iter().filter_map(|a| a.parse().ok()).collect()
    };
    for mp in sizes {
        let h = ((mp * 1e6 / 1.5).sqrt() as usize) & !1;
        let w = (h * 3 / 2) & !1;
        let m = synth::noise_mosaic(w, h, Cfa::RGGB, 3);
        let real = (w * h) as f64 / 1e6;
        println!(
            "== {w} x {h} ({real:.1} MP), {} threads",
            rayon::current_num_threads()
        );
        for method in Method::ALL {
            let t = Instant::now();
            let out = demosaic::demosaic(&m, method);
            let s = t.elapsed().as_secs_f64();
            println!(
                "  {:<9} {s:>8.3} s  {:>8.1} MP/s  (checksum {:.3})",
                method.name(),
                real / s,
                out.data[out.data.len() / 2]
            );
        }
    }
}
