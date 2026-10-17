//! Speed of output sharpening (`docs/algorithms/04-output-sharpening.md`): milliseconds per
//! megapixel for the screen and print presets, one thread and all threads.
//!
//! Run: `cargo run -p ym-gpu --example output_sharpen_bench -- 6 24` (frame sizes in megapixels,
//! default `6`). The development profile optimises this crate lightly (`opt-level = 1`); set
//! `CARGO_PROFILE_DEV_OPT_LEVEL=3` for release-like numbers without building the release profile.
//! The frame is synthetic (soft gradient, bars, a diagonal line and noise), 3:2, interleaved
//! display-linear RGB. The stage is content independent apart from branch prediction in the
//! halo limiter, so the numbers say how fast the code runs, not how it scales with content.

use std::time::Instant;

use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};

/// CPU seconds (user + system, all threads) used by this process so far; `None` off Linux.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let (u, s): (f64, f64) = (f.get(11)?.parse().ok()?, f.get(12)?.parse().ok()?);
    Some((u + s) / 100.0)
}

/// A deterministic 3:2 frame of about `mp` megapixels.
fn frame(mp: f64) -> (usize, usize, Vec<f32>) {
    let h = ((mp * 1e6 / 1.5).sqrt() as usize) & !1;
    let w = (h * 3 / 2) & !1;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.008
    };
    let mut px = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let mut v = 0.1 + 0.3 * (x as f32 / w as f32) + 0.1 * (y as f32 / h as f32);
            if (x / 64 + y / 64) % 2 == 0 {
                v += 0.15;
            }
            if x + y == w / 2 || x + y == w / 2 + 1 {
                v = 0.9;
            }
            px.push(v * 1.1 + noise());
            px.push(v + noise());
            px.push(v * 0.7 + noise());
        }
    }
    (w, h, px)
}

fn main() {
    let sizes: Vec<f64> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() { vec![6.0] } else { sizes };
    let mut no_halo = OutputSharpen::screen();
    no_halo.halo_pct = 0.0;
    let presets: [(&str, OutputSharpen, f32); 4] = [
        ("screen, full size", OutputSharpen::screen(), 1.0),
        (
            "screen, 2048 px from 6000",
            OutputSharpen::screen(),
            scale_for_edges(2048, 6000),
        ),
        ("screen, halo 0, full size", no_halo, 1.0),
        ("print, full size", OutputSharpen::print(), 1.0),
    ];
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    for mp in sizes {
        let (w, h, px) = frame(mp);
        let actual_mp = (w * h) as f64 / 1e6;
        println!("{w} x {h} ({actual_mp:.1} MP), {cores} cores");
        for (label, p, s) in &presets {
            for threads in [1, cores] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool");
                let mut work = px.clone();
                let (c0, t0) = (cpu_seconds(), Instant::now());
                pool.install(|| {
                    p.apply_interleaved_in_place(w, h, &mut work, *s)
                        .expect("sharpen")
                });
                let wall = t0.elapsed().as_secs_f64();
                let cpu = match (c0, cpu_seconds()) {
                    (Some(a), Some(b)) => format!(", cpu {:.2} s", b - a),
                    _ => String::new(),
                };
                println!(
                    "  {label:<28} {threads:>2} thread(s): {:>8.0} ms = {:>6.1} ms/MP{cpu}",
                    wall * 1e3,
                    wall * 1e3 / actual_mp
                );
            }
        }
    }
}
