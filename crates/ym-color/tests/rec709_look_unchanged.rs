//! LogC3-to-Rec.709 looks keep working when the cine rendering replaces ACES 2 as the display transform
//! (owner, 2026-10-29 and 2026-10-30, `docs/02` C12: "Keep them working").
//!
//! The 100 % display-linear picture is the look's own LUT output and does not involve the rendering at all:
//! its numbers are the ones taken from the chain before ARRI's display LUT was removed (commit `b2c352d`),
//! with the same synthetic look, grade and curves as below, and they are still pinned **bit for bit** (1e-9
//! leaves room for another maths library). Three columns legitimately moved with the cine rendering and were
//! re-baselined (2026-10-30), each for a stated reason:
//!
//! * the picture of a look at 60 %: the intensity blend runs in display-linear between the rendering and the
//!   look's picture, and the rendering is about three quarters of a stop darker than the placed ACES 2 it replaces;
//! * the DNG of a look at any intensity above 0: it holds the scene values the rendering turns into the
//!   picture (the inverse of the cine rendering, without the +0.85 EV placement the ACES 2 inverse divided out),
//!   so the numbers are those of the new inverse.
//!
//! Every number is a weighted sum over the same ten camera pixels, so one number stands for ten pixels of three
//! channels each; a look at 100 % takes the place of C12, one at 60 % blends with it. The LUT is our own
//! synthetic cube; nothing here comes from an ARRI file.

use std::sync::Arc;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{ChainParams, ColorChain, LookSetting, OutputTarget};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

fn fused_matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * gain))
}

fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap()
}

fn pixels() -> [Vec3; 10] {
    [
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.001, 0.001, 0.001],
        [1.0, 1.0, 1.0],
        [2.0, 1.5, 1.0],
        [12.0, 9.0, 6.0],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
    ]
}

fn tweak(c: Vec3) -> Vec3 {
    [
        (c[0] * 1.05 + 0.02).min(1.0) * 0.98 + 0.01 * c[1],
        c[1] * 0.97 + 0.03 * c[2] * c[2],
        (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
    ]
}

/// A synthetic "LogC3 to Rec.709" LUT of our own: a smooth tone map with a small colour twist.
fn rec709_baked_look() -> Arc<LookTransform> {
    let lut = Lut::from_cube(
        Lut3d::bake(33, [0.0; 3], [1.0; 3], |c| {
            let t = tweak(c);
            [t[0].powf(1.1), t[1] * 0.95 + 0.02, t[2].powf(0.9)]
        })
        .unwrap(),
    );
    Arc::new(LookTransform::new(LookInput::LogC3ToRec709, None, Some(lut)).unwrap())
}

fn grade_params() -> GradeParams {
    GradeParams {
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
    }
}

fn curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).unwrap(),
        r: Curve::new(&[(0.0, 0.02), (0.5, 0.52), (1.0, 1.0)]).unwrap(),
        g: Curve::identity(),
        b: Curve::new(&[(0.0, 0.0), (0.4, 0.36), (1.0, 0.97)]).unwrap(),
    }
}

/// A weighted sum of every channel of every picture, so that one number stands for all of them.
fn fold(values: impl Iterator<Item = Vec3>) -> f64 {
    values
        .enumerate()
        .map(|(i, v)| {
            v.iter()
                .enumerate()
                .map(|(c, x)| x * (1.0 + 0.37 * (i * 3 + c) as f64))
                .sum::<f64>()
        })
        .sum()
}

/// What the chain gives for one look intensity and one output: the display-linear picture and the DNG's.
fn sums(intensity_pct: f64, output: OutputTarget) -> (f64, f64) {
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = grade_params();
    p.curves = curves();
    p.output = output;
    p.look = Some(LookSetting::new(rec709_baked_look(), intensity_pct));
    let chain = ColorChain::new(&p).unwrap();
    assert!(chain.look_includes_display_rendering());
    let display = fold(
        pixels()
            .into_iter()
            .map(|px| chain.apply_to_display_linear(px)),
    );
    let dng = fold(pixels().into_iter().map(|px| chain.apply_dng_pixel(px)));
    (display, dng)
}

/// `(look intensity %, output, display-linear sum, DNG sum)`. The display sums at 100 % are those of the chain
/// before the removal of ARRI's display LUT (they do not involve the rendering); the display sums at 60 % and
/// every DNG sum are those of the cine rendering (2026-10-30), see the module notes.
const EXPECTED: [(f64, OutputTarget, f64, f64); 8] = [
    (
        100.0,
        OutputTarget::Viewport,
        55.87828240964409,
        366.08224011437704,
    ),
    (
        100.0,
        OutputTarget::Srgb,
        56.26786257306631,
        366.0822401143773,
    ),
    (
        100.0,
        OutputTarget::DisplayP3,
        55.87828240964409,
        366.08224011437704,
    ),
    (
        100.0,
        OutputTarget::Rec2020,
        55.3218921985454,
        366.0822401143773,
    ),
    (
        60.0,
        OutputTarget::Viewport,
        62.71630125400795,
        471.8487904348414,
    ),
    (
        60.0,
        OutputTarget::Srgb,
        63.212960533931934,
        472.843462050988,
    ),
    (
        60.0,
        OutputTarget::DisplayP3,
        62.71630125400795,
        471.8487904348414,
    ),
    (
        60.0,
        OutputTarget::Rec2020,
        61.91336828967983,
        472.2146648970814,
    ),
];

#[test]
fn a_logc3_to_rec709_look_renders_as_it_did_before_the_cine_rendering() {
    let mut rows = Vec::new();
    for intensity in [100.0, 60.0] {
        for output in OutputTarget::ALL {
            rows.push((intensity, output, sums(intensity, output)));
        }
    }
    assert_eq!(rows.len(), EXPECTED.len());
    for ((intensity, output, display, dng), (i, o, (d, g))) in EXPECTED.iter().zip(&rows) {
        assert_eq!((intensity, output), (i, o));
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);
        assert!(
            close(*display, *d),
            "{intensity} % {output:?}: display {d:e}, expected {display:e}"
        );
        assert!(
            close(*dng, *g),
            "{intensity} % {output:?}: DNG {g:e}, expected {dng:e}"
        );
    }
}
