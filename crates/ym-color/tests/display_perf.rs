//! Cost of the CPU reference chain (development measurement, not a gate).
//!
//! ```text
//! cargo test --release -p ym-color --test display_perf -- --ignored --nocapture
//! ```
//!
//! Prints microseconds per pixel on one thread and megapixels per second on all cores for the full
//! chain with the cine rendering (camera RGB in, dithered 8-bit sRGB-encoded P3 out), in two settings: Auto
//! (zero grade, Neutral look, identity curves) and everything on (grade, a LogC4 look with a 33-point
//! cube, curves). The numbers are printed rather than asserted because timing depends on the machine
//! and on what else is running; the results of the last run are recorded in `docs/02`.

use std::sync::Arc;
use std::time::Instant;

use ym_color::cdl::Cdl;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{ChainParams, ColorChain, Frame, Layout, LookSetting, View};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

const W: usize = 1024;
const H: usize = 512;

fn matrix() -> [[f64; 3]; 3] {
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
}

fn image() -> Vec<f32> {
    // Photographic-looking planar camera RGB: a smooth exposure gradient with a hash texture.
    let mut v = vec![0.0_f32; 3 * W * H];
    let mut state = 0x1234_5678_9ABC_DEF0_u64;
    for y in 0..H {
        for x in 0..W {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let jitter = ((state >> 40) as f64 / (1u64 << 24) as f64) * 0.6 + 0.7;
            let scale = 0.18 * (-6.0 + 11.0 * (x as f64 / W as f64)).exp2() * jitter;
            let i = y * W + x;
            v[i] = (scale * 1.1) as f32;
            v[W * H + i] = scale as f32;
            v[2 * W * H + i] = (scale * 0.8) as f32;
        }
    }
    v
}

fn auto_params() -> ChainParams {
    ChainParams::new(matrix(), ToneParams::new(0.5, 0.5, 16.0, 0.002).unwrap())
}

fn everything_params() -> ChainParams {
    let mut p = auto_params();
    p.grade = GradeParams {
        contrast: 20.0,
        brightness: 10.0,
        shadows: 15.0,
        highlights: -20.0,
        saturation: 12.0,
        balance_midtones: ColorBalance { a: 0.2, b: -0.1 },
        ..GradeParams::default()
    };
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).unwrap();
    let cube = Lut3d::bake(33, [0.0; 3], [1.0; 3], |c| {
        [
            c[0] * 0.98 + 0.01 * c[1],
            c[1] * 0.97,
            (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
        ]
    })
    .unwrap();
    let look = LookTransform::new(LookInput::LogC4, Some(cdl), Some(Lut::from_cube(cube))).unwrap();
    p.look = Some(LookSetting::new(Arc::new(look), 80.0));
    p.curves = CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).unwrap(),
        ..CurveSet::default()
    };
    p
}

/// CPU seconds used by the calling thread (Linux `schedstat`), 0 when unavailable.
fn thread_cpu_seconds() -> f64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .next()
                .and_then(|n| n.parse::<f64>().ok())
        })
        .map_or(0.0, |ns| ns / 1e9)
}

/// CPU seconds used by the whole process (Linux `stat`, user + system, 100 Hz ticks), 0 when unavailable.
fn process_cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // Fields after the parenthesised command name; utime and stime are the 12th and 13th of them.
    let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else {
        return 0.0;
    };
    let f: Vec<f64> = rest
        .split_whitespace()
        .skip(11)
        .take(2)
        .filter_map(|v| v.parse().ok())
        .collect();
    if f.len() == 2 {
        (f[0] + f[1]) / 100.0
    } else {
        0.0
    }
}

/// Best wall time over `runs`, with the matching CPU seconds of the calling thread and of the process.
fn best_of(runs: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for _ in 0..runs {
        let (t0, c0, p0) = (Instant::now(), thread_cpu_seconds(), process_cpu_seconds());
        f();
        let wall = t0.elapsed().as_secs_f64();
        if wall < best.0 {
            best = (wall, thread_cpu_seconds() - c0, process_cpu_seconds() - p0);
        }
    }
    best
}

#[test]
#[ignore = "timing measurement, run with --release --ignored --nocapture"]
fn cost_per_pixel() {
    let data = image();
    let src = Frame::new(W, H, Layout::Planar, &data).unwrap();
    let pixels = (W * H) as f64;
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    println!("image {W} x {H} = {:.2} MP, {cores} cores", pixels / 1e6);
    for (name, params) in [
        ("Auto (zero grade, Neutral, identity curves)", auto_params()),
        (
            "everything on (grade, look 33^3, curves)",
            everything_params(),
        ),
    ] {
        let t = Instant::now();
        let chain = ColorChain::new(&params).unwrap();
        println!("{name}: prepare {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let mut out = vec![0_u8; 3 * W * H];
        let (single, single_cpu, _) = best_of(5, || {
            chain
                .render_u8(View::Final, &src, (0, 0), &mut out, 1)
                .unwrap();
        });
        let (multi, _, multi_cpu) = best_of(5, || {
            chain
                .render_u8(View::Final, &src, (0, 0), &mut out, 0)
                .unwrap();
        });
        println!(
            "  one thread: wall {:.3} us/pixel, CPU {:.3} us/pixel ({:.2} Mpx/s of CPU time)",
            single / pixels * 1e6,
            single_cpu / pixels * 1e6,
            pixels / single_cpu / 1e6,
        );
        println!(
            "  {cores} threads: wall {:.2} Mpx/s; process CPU {:.3} us/pixel; projected on {cores} free cores {:.2} Mpx/s \
             (24 MP frame: {:.1} s)",
            pixels / multi / 1e6,
            multi_cpu / pixels * 1e6,
            cores as f64 * pixels / multi_cpu / 1e6,
            24.0e6 / (cores as f64 * pixels / multi_cpu),
        );
        assert!(out.iter().any(|b| *b != 0));
    }
    // The DNG stop point skips C12 and C13.
    let chain = ColorChain::new(&everything_params()).unwrap();
    let mut out = vec![0.0_f32; 3 * W * H];
    let (dng, _, dng_cpu) = best_of(5, || {
        chain
            .render_dng_linear_awg4_f32(&src, Layout::Planar, &mut out, 0)
            .unwrap();
    });
    println!(
        "DNG stop (C1-C11, {cores} threads): wall {:.2} Mpx/s, process CPU {:.3} us/pixel",
        pixels / dng / 1e6,
        dng_cpu / pixels * 1e6
    );
}
