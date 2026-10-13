//! Timing of the CA path of the adaptive ratio demosaic against the plain path (RA-03b).
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example ra03b_timing \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' [-- <megapixels>]
//! ```
//!
//! The default is 4 MP. Each row is the best of three runs in wall seconds and, on Linux, CPU
//! seconds per megapixel (user + system time of the whole process). `whole` is
//! `adaptive_ratio::demosaic_whole`: one piece, one thread. `banded` is `adaptive_ratio::demosaic`
//! on the default pool. The tile of the frame sits in the middle of a 6000 x 4000 frame so that
//! the misregistration is a few tenths of a pixel to more than a pixel, as on a real lens, and
//! `curves` is the cost of the misregistration plane alone (`LensGeometry::ca_curves` is built
//! once per frame and costs microseconds).

use std::time::Instant;

use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params};
use ym_gpu::cpu::demosaic::{CaCurves, DemosaicInputs};
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, NoiseDn, Placement};
use ym_noise::model::NoiseModel;

const REPEATS: usize = 3;

/// User + system CPU seconds of this process so far (Linux), 10 ms resolution.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &stat[stat.rfind(") ")? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

fn time<T>(label: &str, mp: f64, mut f: impl FnMut() -> T) {
    let (mut best_wall, mut best_cpu) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..REPEATS {
        let (cpu0, t0) = (cpu_seconds(), Instant::now());
        std::hint::black_box(f());
        best_wall = best_wall.min(t0.elapsed().as_secs_f64());
        if let Some(c) = cpu0.zip(cpu_seconds()).map(|(a, b)| b - a) {
            best_cpu = best_cpu.min(c);
        }
    }
    println!(
        "  {label:<34} {:>7.3} s/MP wall {:>7.3} s/MP cpu",
        best_wall / mp,
        best_cpu / mp
    );
}

fn main() {
    let mp: f64 = std::env::args()
        .nth(1)
        .map_or(4.0, |a| a.parse().expect("megapixels"));
    let w = ((mp * 1e6 * 1.5).sqrt() as usize) & !1;
    let h = ((mp * 1e6 / 1.5).sqrt() as usize) & !1;
    let mp = (w * h) as f64 / 1e6;
    let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, 7), Cfa::RGGB);
    let (fw, fh) = (6000usize, 4000usize);
    m.place = Placement {
        x0: fw / 2 + 400,
        y0: fh / 2 + 200,
        frame_width: fw,
        frame_height: fh,
    };
    // A real-looking table: 16 knots per colour, a few thousand units at the corner.
    let tags = LensTags {
        vignetting: vec![],
        ca: (0..16)
            .map(|j| (1.5 * f64::from(j * j)) as i16)
            .chain((0..16).map(|j| (-1.2 * f64::from(j * j)) as i16))
            .collect(),
        distortion: (0..16).map(|j| (2 * j * j) as i16).collect(),
    };
    let geo = LensGeometry::new(fw, fh, &tags, LensSwitches::default()).expect("lens");
    let curves: CaCurves = geo.ca_curves().expect("curves").expect("CA");
    let noise = NoiseDn::from_model(&NoiseModel {
        a: 2e-5,
        b: 1e-8,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    });
    let params = Params::default();
    let plain = DemosaicInputs {
        gains: [2.0, 1.0, 1.6],
        noise,
        ca: None,
    };
    let with = DemosaicInputs {
        ca: Some(&curves),
        ..plain
    };
    let (cx, cy) = (curves.centre_x, curves.centre_y);
    let r = ((m.place.x0 as f64 + w as f64 / 2.0 - cx)
        .hypot(m.place.y0 as f64 + h as f64 / 2.0 - cy))
    .round();
    println!(
        "{w} x {h} ({mp:.2} MP) at {r} px from the centre: |R - B| there {:.2} px",
        curves.misregistration_px(r)
    );
    time("whole, plain", mp, || {
        adaptive_ratio::demosaic_whole(&m, &plain, &params).expect("plain")
    });
    time("whole, CA path", mp, || {
        adaptive_ratio::demosaic_whole(&m, &with, &params).expect("CA path")
    });
    time("banded, plain (default pool)", mp, || {
        adaptive_ratio::demosaic(&m, &plain, &params).expect("plain")
    });
    time("banded, CA path (default pool)", mp, || {
        adaptive_ratio::demosaic(&m, &with, &params).expect("CA path")
    });
    // The misregistration alone: the curve evaluation of every R and B site of the frame.
    time("curves only (R and B sites)", mp, || {
        let mut s = 0.0f64;
        for y in 0..h {
            let dy = (m.place.y0 + y) as f64 + 0.5 - cy;
            for x in (y & 1..w).step_by(2) {
                let dx = (m.place.x0 + x) as f64 + 0.5 - cx;
                s += curves.misregistration_px((dx * dx + dy * dy).sqrt());
            }
        }
        s
    });
}
