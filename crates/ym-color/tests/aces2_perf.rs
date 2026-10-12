//! Cost of the ACES 2 output transform on the CPU (development measurement, not a gate).
//!
//! ```text
//! cargo test --release -p ym-color --test aces2_perf -- --ignored --nocapture
//! ```
//!
//! Measured on the development machine (4 vCPU Xeon @ 2.8 GHz container shared with other
//! builds, rustc 1.97, release profile with thin LTO, one thread, photographic AP0 input):
//!
//! | what | cost |
//! |---|---|
//! | table construction (one limiting gamut, 100 nit) | 10 to 35 ms |
//! | `transform_ap0`, Viewport / sRGB / Display P3 | 0.9 to 1.1 microseconds per pixel |
//! | `transform_ap0`, Rec.2020 | 0.9 to 1.4 microseconds per pixel |
//!
//! That is 21 to 34 s for a 24 MP frame on one thread, 6 to 9 s on four. It is a reference for
//! tests and for machines without a usable GPU, not a rendering path. The numbers are printed
//! rather than asserted because timing depends on the machine and on what else is running.

use std::time::Instant;

use ym_color::aces2::{Aces2Output, Aces2Params, Aces2Transform, Primaries};

/// SplitMix64, a tiny deterministic generator (no external dependency).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Photographic-looking AP0 pixels: exposure -6..+5 stops around grey, moderate saturation.
fn pixels(n: usize) -> Vec<[f64; 3]> {
    let mut rng = Rng(0xAC35_2024);
    (0..n)
        .map(|_| {
            let scale = 0.18 * 2.0_f64.powf(-6.0 + 11.0 * rng.unit());
            [
                scale * (0.2 + 1.6 * rng.unit()),
                scale * (0.2 + 1.6 * rng.unit()),
                scale * (0.2 + 1.6 * rng.unit()),
            ]
        })
        .collect()
}

#[test]
#[ignore = "timing measurement, run with --release --ignored --nocapture"]
fn cost_per_pixel() {
    let start = Instant::now();
    let params = Aces2Params::new(100.0, Primaries::P3_D65);
    let construction = start.elapsed();
    println!(
        "construction (P3-D65 limited, 100 nit): {:.1} ms",
        construction.as_secs_f64() * 1e3
    );
    drop(params);

    let data = pixels(2_000_000);
    for output in Aces2Output::ALL {
        let t = Aces2Transform::new(output);
        // Warm up.
        let mut sink = 0.0;
        for p in data.iter().take(10_000) {
            sink += t.transform_ap0(*p)[0];
        }
        let start = Instant::now();
        for p in &data {
            sink += t.transform_ap0(*p)[0];
        }
        let dt = start.elapsed();
        println!(
            "per pixel {:<11} {:.3} us/pixel  ({:.2} s for 24 MP on one thread)  [{sink:.1}]",
            output.name(),
            dt.as_secs_f64() * 1e6 / data.len() as f64,
            dt.as_secs_f64() / data.len() as f64 * 24e6
        );
    }
}
