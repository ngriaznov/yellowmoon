//! CPU reference throughput for the per-pixel stages and the `.cube` parser.
//!
//! Run with `cargo bench -p ym-color --bench color_cpu`. This is a plain `Instant` harness (no
//! statistics): it prints million pixels per second so regressions of 2x or more are obvious.

use std::hint::black_box;
use std::time::Instant;

use ym_color::cdl::Cdl;
use ym_color::cube;
use ym_color::curve::{Curve, CurveSet};
use ym_color::gamut::LinearAwg4;
use ym_color::grade::{ColorBalance, Grade, GradeParams};
use ym_color::logc::{LogC4, logc4_decode, logc4_encode};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

const N: usize = 1_000_000;

fn bench<T>(name: &str, mut f: impl FnMut(usize) -> T) {
    // Warm up, then time N calls.
    for i in 0..10_000 {
        black_box(f(i));
    }
    let t = Instant::now();
    for i in 0..N {
        black_box(f(i));
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{name:34} {:8.2} Mpix/s  ({:6.1} ns/pixel)",
        N as f64 / secs / 1e6,
        secs / N as f64 * 1e9
    );
}

fn pixel(i: usize) -> [f64; 3] {
    let t = (i % 1021) as f64 / 1021.0;
    [t, (t * 7.0).fract(), 1.0 - t]
}

fn main() {
    bench("logc4 encode (3 channels)", |i| {
        let p = pixel(i);
        [logc4_encode(p[0]), logc4_encode(p[1]), logc4_encode(p[2])]
    });
    bench("logc4 decode (3 channels)", |i| {
        let p = pixel(i);
        [logc4_decode(p[0]), logc4_decode(p[1]), logc4_decode(p[2])]
    });
    let tone = ToneParams::new(0.5, 0.5, 16.0, 0.003).unwrap();
    bench("C7 roll-off and toe", |i| {
        let p = pixel(i);
        tone.apply(LinearAwg4([p[0] * 20.0, p[1] * 2.0, p[2]]))
    });
    let grade = Grade::new(&GradeParams {
        contrast: 20.0,
        brightness: 10.0,
        shadows: 15.0,
        highlights: -20.0,
        black: 5.0,
        white: -5.0,
        balance_shadows: ColorBalance { a: 0.2, b: -0.1 },
        balance_midtones: ColorBalance { a: 0.0, b: 0.1 },
        balance_highlights: ColorBalance { a: -0.1, b: 0.0 },
        saturation: 12.0,
    });
    bench("C8 + C9 (all controls)", |i| {
        let p = pixel(i);
        grade.apply_linear(LinearAwg4([p[0] * 4.0, p[1], p[2]]))
    });
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.01, 0.0, -0.01], [1.0, 1.1, 1.0], 1.1).unwrap();
    bench("CDL", |i| cdl.apply(pixel(i)));
    let lut33 = Lut3d::bake(33, [0.0; 3], [1.0; 3], |c| {
        [c[0] * c[1], c[2].sqrt(), 1.0 - c[0]]
    })
    .unwrap();
    bench("tetrahedral 33^3", |i| lut33.sample(pixel(i)));
    let look = LookTransform::new(
        LookInput::LogC3Ei800,
        Some(cdl),
        Some(Lut::from_cube(lut33.clone())),
    )
    .unwrap();
    bench("LogC3 look (convert, CDL, LUT, back)", |i| {
        look.apply_logc4(LogC4(pixel(i)), 80.0).unwrap()
    });
    let curves = CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.25), (0.7, 0.8), (1.0, 1.0)]).unwrap(),
        ..Default::default()
    }
    .bake();
    bench("C11 luma curve", |i| curves.apply(LogC4(pixel(i))));

    // Parser throughput on a 65-point cube (274,625 rows, like ARRI's LogC4 looks).
    let text = Lut3d::bake(65, [0.0; 3], [1.0; 3], |c| {
        [c[0] * 0.9 + 0.05, c[1], c[2] * c[2]]
    })
    .unwrap()
    .to_cube_string("bench");
    let t = Instant::now();
    let runs = 5;
    for _ in 0..runs {
        black_box(cube::parse(black_box(&text)).unwrap());
    }
    let secs = t.elapsed().as_secs_f64() / f64::from(runs);
    println!(
        "{:34} {:8.1} ms per 65^3 cube ({:.1} MB, {:.0} MB/s)",
        "parse .cube",
        secs * 1e3,
        text.len() as f64 / 1e6,
        text.len() as f64 / 1e6 / secs
    );
}
