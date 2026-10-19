//! Colour chains and images shared by the colour-kernel tests: our own synthetic LUTs, curves and
//! grades, never anything from ARRI.

#![allow(dead_code)]

use std::sync::Arc;

use ym_color::camera_default::CameraDefault;
use ym_color::cdl::Cdl;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ArriDisplayLut, ChainParams, DisplayLutInput, DisplayLutOutput, DisplayLutPrimaries,
    DisplayLutTransfer, DisplayTransform, LookSetting, OutputTarget,
};
use ym_color::exposure::white::WhiteLift;
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut1d, Lut3d};
use ym_color::tone::ToneParams;
use ym_color::warp::{WARP_MAX_PINS, WarpParams, WarpPin};
use ym_gpu::cpu::RgbImage;

use super::{Rng, f16_round};

/// A white-preserving matrix standing in for the fused C1 to C6 (rows sum to 1), with +0.25 EV.
pub fn fused_matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    let m = [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ];
    m.map(|row| row.map(|v| v * gain))
}

/// C7 with the Sony clip 6.5 stops over grey (luma 16): the highlight range acts on the brightest few
/// percent of [`camera_image`] (from 5.5 stops), the toe near black.
pub fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.5, 16.0, 0.002).expect("tone")
}

/// The clip luma of a Sony body at the camera's exposure: 3.58 stops over middle grey (`docs/02` C7).
pub const AUTO_CLIP_STOPS: f64 = 3.58;

/// The clip luma `stops` over middle grey.
pub fn clip_luma(stops: f64) -> f64 {
    0.18 * stops.exp2()
}

/// C7 with the highlight range at `highlights` and the clip `clip_stops` over grey, the toe as [`tone`].
pub fn range_tone(highlights: f64, clip_stops: f64) -> ToneParams {
    ToneParams::new(highlights, 0.5, clip_luma(clip_stops), 0.002).expect("tone")
}

/// C7 off: no highlight range, no cap, no toe.
pub fn identity_tone() -> ToneParams {
    ToneParams::IDENTITY
}

pub fn grade_params() -> GradeParams {
    GradeParams {
        contrast: 20.0,
        brightness: -15.0,
        shadows: 30.0,
        highlights: -25.0,
        black: 20.0,
        white: -10.0,
        balance_shadows: ColorBalance { a: 0.4, b: -0.2 },
        balance_midtones: ColorBalance { a: -0.3, b: 0.5 },
        balance_highlights: ColorBalance { a: 0.2, b: 0.2 },
        saturation: 15.0,
    }
}

/// A smooth, non-trivial per-colour function standing in for a creative LUT.
pub fn tweak(c: Vec3) -> Vec3 {
    [
        (c[0] * 1.05 + 0.02).min(1.0) * 0.98 + 0.01 * c[1],
        c[1] * 0.97 + 0.03 * c[2] * c[2],
        (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
    ]
}

pub fn cube_lut(size: usize, f: impl FnMut(Vec3) -> Vec3) -> Lut {
    Lut::from_cube(Lut3d::bake(size, [0.0; 3], [1.0; 3], f).expect("cube"))
}

pub fn logc4_look() -> Arc<LookTransform> {
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).expect("cdl");
    Arc::new(
        LookTransform::new(LookInput::LogC4, Some(cdl), Some(cube_lut(17, tweak))).expect("look"),
    )
}

/// A LogC3 look with a 1D shaper in front of the cube and a CDL.
pub fn logc3_look() -> Arc<LookTransform> {
    let cdl = Cdl::new([0.95, 1.0, 1.05], [0.01, 0.0, -0.01], [1.05, 1.0, 1.0], 0.9).expect("cdl");
    let shaper = Lut1d::new(
        33,
        [0.0; 3],
        [1.0; 3],
        (0..33)
            .map(|i| {
                let x = f64::from(i) / 32.0;
                [x.powf(0.9), x.powf(1.0), (x * 1.02).min(1.0)]
            })
            .collect(),
    )
    .expect("shaper");
    let mut lut = cube_lut(9, tweak);
    lut.shaper = Some(shaper);
    Arc::new(LookTransform::new(LookInput::LogC3Ei800, Some(cdl), Some(lut)).expect("look"))
}

/// A LogC3-to-Rec.709 look (look and display rendering baked together).
pub fn baked_look() -> Arc<LookTransform> {
    let lut = cube_lut(17, |c| {
        let t = tweak(c);
        [t[0].powf(1.1), t[1].powf(1.05), t[2].powf(1.1)]
    });
    Arc::new(LookTransform::new(LookInput::LogC3ToRec709, None, Some(lut)).expect("look"))
}

pub fn curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.02), (0.3, 0.26), (0.6, 0.66), (1.0, 0.98)]).expect("curve"),
        r: Curve::new(&[(0.0, 0.0), (0.5, 0.54), (1.0, 1.0)]).expect("curve"),
        g: Curve::identity(),
        b: Curve::new(&[(0.0, 0.01), (0.4, 0.37), (1.0, 1.0)]).expect("curve"),
    }
}

pub fn display_lut(input: DisplayLutInput, output: DisplayLutOutput) -> DisplayTransform {
    let lut = cube_lut(17, |c| c.map(|x| (x * 1.02).clamp(0.0, 1.0).powf(1.3)));
    DisplayTransform::arri(ArriDisplayLut::new(Arc::new(lut), input, output).expect("display lut"))
}

/// A body-shaped camera default for the Before view: a published-looking D65 matrix (strong red-green
/// overlap, negative off-diagonals), an as-shot white that differs from the initial gains.
pub fn camera_default() -> CameraDefault {
    let cm = [
        [0.7657, -0.2847, -0.0607],
        [-0.4083, 1.1966, 0.2389],
        [-0.0684, 0.1418, 0.5844],
    ];
    CameraDefault::new(&cm, [2.05, 1.0, 1.48], [1.9, 1.0, 1.6]).expect("camera default")
}

pub fn base(target: OutputTarget) -> ChainParams {
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.output = target;
    p.camera_default = camera_default();
    p
}

/// A chain with everything on: tone, grade, a LogC4 look at 70 %, curves, ACES 2.
pub fn full(target: OutputTarget) -> ChainParams {
    let mut p = base(target);
    p.grade = grade_params();
    p.look = Some(LookSetting::new(logc4_look(), 70.0));
    p.curves = curves();
    p
}

/// Colour warp pins (C9b) placed where the colours of [`camera_image`] are after C9 (half its pixels
/// have a chroma `|c|` under 0.02, a tenth over 0.06; tones from 3 stops under grey to 4 over): moves
/// with the default and with split tonal weights, pivots under and over grey, exposure up and down, a
/// long move with the smallest range (the flow takes 12 steps for the set), an exposure-only pin, a
/// lock between them and a still pin (left out by the preparation).
pub fn warp_pins() -> WarpParams {
    WarpParams {
        pins: vec![
            WarpPin {
                range: 0.05,
                ..WarpPin::moving([0.045, 0.015], [0.015, 0.06])
            },
            WarpPin {
                range: 0.06,
                shadows: 1.0,
                highlights: 0.2,
                pivot_ev: -2.0,
                exposure_ev: -0.6,
                ..WarpPin::moving([-0.03, 0.04], [-0.07, 0.0])
            },
            WarpPin {
                range: 0.04,
                shadows: 0.0,
                highlights: 1.0,
                pivot_ev: 1.0,
                exposure_ev: 0.8,
                ..WarpPin::moving([-0.02, -0.05], [0.01, -0.08])
            },
            WarpPin {
                range: 0.01,
                ..WarpPin::moving([0.06, -0.03], [-0.04, -0.09])
            },
            WarpPin {
                range: 0.03,
                exposure_ev: 1.2,
                ..WarpPin::moving([0.0, -0.03], [0.0, -0.03])
            },
            WarpPin {
                range: 0.025,
                ..WarpPin::locked_at([0.03, 0.03])
            },
            WarpPin::moving([0.1, 0.1], [0.1, 0.1]),
        ],
    }
}

/// A random set of 0 to [`WARP_MAX_PINS`] pins (and sometimes a few more, which are ignored): about a
/// third locked, some still, sources mostly where the colours of [`camera_image`] are and sometimes
/// beyond the limit of 0.5, and every control drawn a little beyond its range now and then, so the
/// preparation's clamping is exercised too.
pub fn random_warp(rng: &mut Rng) -> WarpParams {
    let count = (rng.next_u64() % (WARP_MAX_PINS as u64 + 3)) as usize;
    let point = |rng: &mut Rng, reach: f64| {
        let (r, a) = (
            reach * rng.unit().sqrt(),
            rng.range(0.0, std::f64::consts::TAU),
        );
        [r * a.cos(), r * a.sin()]
    };
    let pins = (0..count)
        .map(|_| {
            let source = if rng.unit() < 0.05 {
                point(rng, 0.7)
            } else {
                point(rng, 0.12)
            };
            let range = rng.range(0.005, 0.3);
            if rng.unit() < 0.3 {
                return WarpPin {
                    range,
                    ..WarpPin::locked_at(source)
                };
            }
            let offset = point(rng, 0.15);
            let target = if rng.unit() < 0.15 {
                source
            } else {
                [source[0] + offset[0], source[1] + offset[1]]
            };
            WarpPin {
                source,
                target,
                locked: false,
                range,
                shadows: rng.range(-0.1, 1.1),
                highlights: rng.range(-0.1, 1.1),
                pivot_ev: rng.range(-7.0, 7.0),
                exposure_ev: if rng.unit() < 0.5 {
                    rng.range(-2.5, 2.5)
                } else {
                    0.0
                },
            }
        })
        .collect();
    WarpParams { pins }
}

/// All named configurations the parity tests cover.
pub fn cases() -> Vec<(String, ChainParams)> {
    let mut v: Vec<(String, ChainParams)> = Vec::new();
    for t in OutputTarget::ALL {
        v.push((format!("auto/{}", t.name()), base(t)));
        v.push((format!("full/{}", t.name()), full(t)));
    }
    // Saturation (C9, the chroma scaled about the luminance in linear AWG4) and Contrast (C9, the tone
    // curve on the luma): alone, and on top of the full grade, look and curves. -100 and 100 are the
    // ends of the control, 60 and -60 are in the middle.
    for sat in [-100.0, -60.0, 60.0, 100.0] {
        let mut p = base(OutputTarget::Viewport);
        p.grade.saturation = sat;
        v.push((format!("saturation/{sat:+}"), p));
        let mut p = full(OutputTarget::Viewport);
        p.grade.saturation = sat;
        v.push((format!("saturation+full/{sat:+}"), p));
    }
    for contrast in [-100.0, -60.0, 60.0, 100.0] {
        let mut p = base(OutputTarget::Viewport);
        p.grade.contrast = contrast;
        v.push((format!("contrast/{contrast:+}"), p));
        let mut p = full(OutputTarget::Viewport);
        p.grade.contrast = contrast;
        v.push((format!("contrast+full/{contrast:+}"), p));
    }
    // A CDL without a LUT, and a LUT without a CDL (the kernel's two optional parts of C10).
    let mut p = full(OutputTarget::Viewport);
    let cdl_only =
        Cdl::new([1.2, 0.95, 0.8], [0.03, -0.01, 0.0], [0.9, 1.1, 1.3], 0.7).expect("cdl");
    p.look = Some(LookSetting::new(
        Arc::new(LookTransform::new(LookInput::LogC4, Some(cdl_only), None).expect("look")),
        80.0,
    ));
    v.push(("cdl-only-look".into(), p.clone()));
    p.look = Some(LookSetting::new(
        Arc::new(
            LookTransform::new(LookInput::LogC4, None, Some(cube_lut(33, tweak))).expect("look"),
        ),
        100.0,
    ));
    v.push(("lut-only-look".into(), p.clone()));
    p.look = Some(LookSetting::new(logc3_look(), 100.0));
    v.push(("logc3-look/100".into(), p.clone()));
    p.look = Some(LookSetting::new(logc3_look(), 35.0));
    v.push(("logc3-look/35".into(), p));
    for t in [
        OutputTarget::Srgb,
        OutputTarget::Viewport,
        OutputTarget::Rec2020,
    ] {
        for pct in [0.0, 25.0, 100.0] {
            let mut p = full(t);
            p.look = Some(LookSetting::new(baked_look(), pct));
            v.push((format!("baked-look/{}/{pct}", t.name()), p));
        }
    }
    for (name, input, output) in [
        (
            "709-g24",
            DisplayLutInput::LogC4,
            DisplayLutOutput::REC709_G24,
        ),
        (
            "p3-g26-logc3",
            DisplayLutInput::LogC3Ei800,
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::P3D65,
                transfer: DisplayLutTransfer::Gamma(2.6),
            },
        ),
        (
            "2020-srgb",
            DisplayLutInput::LogC4,
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::Rec2020,
                transfer: DisplayLutTransfer::Srgb,
            },
        ),
    ] {
        for t in [OutputTarget::Srgb, OutputTarget::Rec2020] {
            let mut p = full(t);
            p.display = display_lut(input, output);
            v.push((format!("display-lut/{name}/{}", t.name()), p));
        }
    }
    // Display LUT together with a display-baked look (both feed C12).
    let mut p = full(OutputTarget::DisplayP3);
    p.display = display_lut(DisplayLutInput::LogC4, DisplayLutOutput::REC709_G24);
    p.look = Some(LookSetting::new(baked_look(), 60.0));
    v.push(("display-lut+baked-look".into(), p));
    v.extend(range_cases());
    // The white lift (C12 step 1b): alone, with everything on, a narrow strong ramp, behind the highlight
    // range at its default (both stages run: the range on the scene side, the lift after C11), on the own
    // branch of a display-baked look, and (ignored) with a display LUT.
    let lift = WhiteLift::new(1.1, 2.8).expect("lift");
    for t in [OutputTarget::Viewport, OutputTarget::Srgb] {
        let mut p = base(t);
        p.white_lift = lift;
        v.push((format!("lift/auto/{}", t.name()), p));
        let mut p = full(t);
        p.white_lift = lift;
        v.push((format!("lift/full/{}", t.name()), p));
    }
    let mut p = base(OutputTarget::Viewport);
    p.white_lift = WhiteLift::new(2.6, 1.3).expect("lift");
    v.push(("lift/strong-narrow".into(), p));
    let mut p = base(OutputTarget::Srgb);
    p.tone = range_tone(1.0, AUTO_CLIP_STOPS);
    p.white_lift = lift;
    v.push(("lift/behind-the-range".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.look = Some(LookSetting::new(baked_look(), 25.0));
    p.white_lift = lift;
    v.push(("lift/baked-look/25".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.display = display_lut(DisplayLutInput::LogC4, DisplayLutOutput::REC709_G24);
    p.white_lift = lift;
    v.push(("lift/display-lut-ignores-it".into(), p));
    // The colour warp (C9b): alone, after the full grade and before the look and the curves, and
    // feeding a display LUT and a display-baked look.
    let mut p = base(OutputTarget::Viewport);
    p.warp = warp_pins();
    v.push(("warp/auto/viewport".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.warp = warp_pins();
    v.push(("warp/full/srgb".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.display = display_lut(DisplayLutInput::LogC3Ei800, DisplayLutOutput::REC709_G24);
    p.warp = warp_pins();
    v.push(("warp/display-lut".into(), p));
    let mut p = full(OutputTarget::Viewport);
    p.look = Some(LookSetting::new(baked_look(), 60.0));
    p.warp = warp_pins();
    v.push(("warp/baked-look/60".into(), p));
    v
}

/// The highlight range of C7 (`docs/02`): 0, 0.5 and 1 at the clip of a Sony body at Auto (3.58 stops over
/// grey, so a fifth of [`camera_image`] lies above the knee), with everything on, a clip inside the fade
/// (1.6 stops: the knee stays at +1 stop and the expansion is partly faded), a clip over the ALEXA's (the
/// Exposure 5 stops up: no expansion, only the cap, which a fifth of the pixels reach), and the range in
/// front of a display-baked look and of a display LUT (both see it, as everything after C7 does).
pub fn range_cases() -> Vec<(String, ChainParams)> {
    let mut v: Vec<(String, ChainParams)> = Vec::new();
    for r in [0.0, 0.5, 1.0] {
        let mut p = base(OutputTarget::Viewport);
        p.tone = range_tone(r, AUTO_CLIP_STOPS);
        v.push((format!("range/auto/{r}/viewport"), p));
    }
    let mut p = base(OutputTarget::Srgb);
    p.tone = range_tone(0.5, AUTO_CLIP_STOPS);
    v.push(("range/auto/0.5/srgb".into(), p));
    for t in [OutputTarget::Viewport, OutputTarget::Rec2020] {
        let mut p = full(t);
        p.tone = range_tone(1.0, AUTO_CLIP_STOPS);
        v.push((format!("range/full/1/{}", t.name()), p));
    }
    let mut p = base(OutputTarget::Srgb);
    p.tone = range_tone(1.0, 1.6);
    v.push(("range/fade/1.6".into(), p));
    let mut p = base(OutputTarget::Srgb);
    p.matrix = fused_matrix().map(|row| row.map(|x| x * 32.0));
    p.tone = range_tone(0.5, AUTO_CLIP_STOPS + 5.0);
    v.push(("range/over-the-cap".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.tone = range_tone(0.5, AUTO_CLIP_STOPS);
    p.look = Some(LookSetting::new(baked_look(), 25.0));
    v.push(("range/baked-look/25".into(), p));
    let mut p = full(OutputTarget::Srgb);
    p.tone = range_tone(1.0, AUTO_CLIP_STOPS);
    p.display = display_lut(DisplayLutInput::LogC3Ei800, DisplayLutOutput::REC709_G24);
    v.push(("range/display-lut".into(), p));
    v
}

/// A random camera image: scene-linear values over 17 stops, a few slightly negative samples, a
/// few very bright ones, one exact zero. Every value is representable in f16 so that the CPU
/// reference sees exactly what the GPU reads in every storage mode.
pub fn camera_image(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let lum = 0.18 * rng.range(-12.0, 5.5).exp2();
        let mut px = [
            lum * rng.range(0.35, 1.65),
            lum * rng.range(0.35, 1.65),
            lum * rng.range(0.35, 1.65),
        ];
        if i % 23 == 5 {
            px[1] = -0.004 * rng.unit();
        }
        if i % 101 == 7 {
            px = [60.0 * rng.unit(), 40.0 * rng.unit(), 80.0 * rng.unit()];
        }
        if i == 0 {
            px = [0.0; 3];
        }
        if i == 1 {
            px = [0.18; 3];
        }
        for c in px {
            rgb.push(f16_round(c as f32));
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

/// The camera triple of pixel `i` as f64.
pub fn pixel(img: &RgbImage, i: usize) -> Vec3 {
    let n = img.plane_len();
    [
        f64::from(img.data[i]),
        f64::from(img.data[n + i]),
        f64::from(img.data[2 * n + i]),
    ]
}
