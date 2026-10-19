//! M1-09 monotonicity on the GPU: C7 (the highlight range and the toe) and every C9 control behave as
//! the CPU reference promises (`docs/02` sections C7 and C9, `ym-color` tests `tone.rs` and `grade.rs`).
//!
//! Runs on the f32-storage device (strict ramps need unrounded inputs) and, for the C9 control
//! sweeps, on every device.

mod common;

use common::chains::identity_tone;
use common::{Rng, TestGpu, f16_round, gpus};
use ym_color::consts::LOGC4_GRAY;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain};
use ym_color::gamut::awg4_luminance;
use ym_color::grade::{
    CONTRAST_BLACK, CONTRAST_COLOR, CONTRAST_TOP, ColorBalance, E1, E2, GradeParams,
    SATURATION_BOOST, contrast_curve, contrast_slope,
};
use ym_color::logc::{logc4_decode, logc4_encode};
use ym_color::tone::{HIGHLIGHT_CAP_STOPS, ToneParams, luma};
use ym_gpu::cpu::RgbImage;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

fn image_of(px: &[[f32; 3]]) -> RgbImage {
    let inter: Vec<f32> = px.iter().flatten().copied().collect();
    RgbImage::from_interleaved(px.len(), 1, &inter).expect("image")
}

fn run(dev: &TestGpu, params: &ChainParams, px: &[[f32; 3]], stop: Stop) -> Vec<[f32; 3]> {
    let chain = ColorChain::new(params).expect("chain");
    let img = image_of(px);
    let n = px.len();
    let out = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            PreparedChain::new(ctx, &chain)?.render(ctx, &img, (0, 0), ColorOutput::Stage(stop))
        })
        .expect("render");
    let PixelData::F32Planar(v) = out.data else {
        panic!()
    };
    (0..n).map(|i| [v[i], v[n + i], v[2 * n + i]]).collect()
}

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

#[test]
fn the_highlight_range_and_toe_never_decrease_with_the_input() {
    let Some(dev) = f32_device() else { return };
    // A grey ramp from slightly negative to 60 (log spaced above 0), and one channel ramped while the
    // others stay fixed.
    let mut ramp: Vec<f32> = vec![-0.01, -0.001, 0.0];
    // 4000 log-spaced points from 1e-5 to 60 (2^22.5 apart): the ramp runs through the knee (at least
    // +1 stop, 0.36), the Sony clip (1 to 6 stops over grey below) and the ALEXA cap (7.64 stops, 35.9)
    // of every parameter set, and through the toe's noise floors.
    ramp.extend((0..4000).map(|i| 1e-5 * (i as f32 * 0.005_630_54).exp2()));
    ramp.retain(|v| *v <= 60.0);
    ramp.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert!(
        ramp.last()
            .is_some_and(|&top| f64::from(top) > 0.18 * HIGHLIGHT_CAP_STOPS.exp2() * 1.5),
        "the ramp must pass the cap: top {:?}",
        ramp.last()
    );
    let grey: Vec<[f32; 3]> = ramp.iter().map(|&v| [v, v, v]).collect();
    let chroma: Vec<[f32; 3]> = ramp.iter().map(|&v| [v, 0.4, 0.05]).collect();
    let mut checked = 0;
    let (mut ranged, mut capped) = (0usize, 0usize);
    for highlights in [0.0, 0.3, 0.5, 1.0] {
        for low_light in [0.0, 0.25, 0.5, 1.0] {
            for clip_stops in [1.0, 1.6, 2.0, 3.58, 6.0] {
                for floor in [0.0, 0.002, 0.02] {
                    let clip = 0.18 * f64::exp2(clip_stops);
                    let tone = ToneParams::new(highlights, low_light, clip, floor).expect("tone");
                    let range = tone.range();
                    let params = ChainParams::new(IDENTITY, tone);
                    for (label, px) in [("grey", &grey), ("one channel", &chroma)] {
                        let out = run(dev, &params, px, Stop::C7);
                        for i in 1..out.len() {
                            // Where the CPU holds the luma at the cap a grey is flat, and the f32 log2 and
                            // exp2 of the gain wiggle by a few 1e-7 (allowed: 1e-5, for a log2 a few ulp
                            // off as Metal's fast-math one is); everywhere else the output rises by at
                            // least the input's step (0.4 %), so only the last bits may wiggle.
                            let y = luma(px[i].map(f64::from));
                            let s = (y / 0.18).log2();
                            let at_cap = y > 0.0 && range.stops(s) >= HIGHLIGHT_CAP_STOPS - 1e-9;
                            ranged += usize::from(y > 0.0 && range.gain(y) > 1.0);
                            capped += usize::from(at_cap);
                            let rel = if at_cap { 1e-5 } else { 6e-7 };
                            for (ch, (&a, &b)) in out[i - 1].iter().zip(&out[i]).enumerate() {
                                if label == "one channel" && ch != 0 {
                                    continue;
                                }
                                assert!(
                                    b >= a - rel * a.abs().max(1e-3),
                                    "{label} h={highlights} l={low_light} clip={clip_stops} floor={floor}: output fell from {a} to {b} at input {:?}",
                                    px[i]
                                );
                            }
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(
        ranged > 10_000 && capped > 10_000,
        "{ranged} ranged, {capped} capped samples"
    );
    eprintln!(
        "C7 monotone over {checked} parameter sets x {} ramp points ({ranged} samples in the range, {capped} at the cap)",
        grey.len()
    );
}

/// The ALEXA cap on the device: a coloured ramp from half a stop to eight stops past it comes out with
/// its luma at `0.18 * 2^7.64` (35.9) to f32 precision and its channel ratios kept, whatever the amount
/// and the clip (one inside the fade, the Auto clip, one over the cap).
#[test]
fn nothing_leaves_the_device_above_the_alexa_clip() {
    let Some(dev) = f32_device() else { return };
    let cap = (0.18 * HIGHLIGHT_CAP_STOPS.exp2()) as f32;
    let px: Vec<[f32; 3]> = (0..200)
        .map(|i| {
            // The luma is 0.66 v: 8.5 to 16.5 stops for v is 7.9 to 15.9 for the luma, over the cap from
            // the first pixel.
            let v = 0.18 * (8.5 + 0.04 * f64::from(i)).exp2();
            [v as f32, (0.6 * v) as f32, (0.2 * v) as f32]
        })
        .collect();
    for r in [0.0, 0.5, 1.0] {
        for clip_stops in [1.0, 3.58, 9.0] {
            let tone = ToneParams::new(r, 0.0, 0.18 * f64::exp2(clip_stops), 0.0).expect("tone");
            let out = run(dev, &ChainParams::new(IDENTITY, tone), &px, Stop::C7);
            for (o, p) in out.iter().zip(&px) {
                let y = luma(o.map(f64::from));
                // The f32 log2 of up to 16 stops, a few ulp off (Metal's fast-math), is a relative 1e-5.
                assert!(
                    (y / f64::from(cap) - 1.0).abs() < 1e-5,
                    "r {r} clip {clip_stops}: {p:?} -> {o:?} (luma {y}, cap {cap})"
                );
                // The ratios are the input's.
                assert!((f64::from(o[1] / o[0]) - 0.6).abs() < 1e-6);
            }
        }
    }
}

/// Which way Contrast moves every channel of a pixel whose luma is `l` as the control rises: away
/// from grey between black and the top code value (+1 above grey, -1 below), toward the end beyond
/// them (the line of slope `1 / k`); at grey, black and the top code value nothing moves.
fn contrast_direction(l: f64) -> f64 {
    let side = if l > LOGC4_GRAY {
        1.0
    } else if l < LOGC4_GRAY {
        -1.0
    } else {
        0.0
    };
    if (CONTRAST_BLACK..=CONTRAST_TOP).contains(&l) {
        side
    } else {
        -side
    }
}

fn grade_chain(g: GradeParams) -> ChainParams {
    let mut p = ChainParams::new(IDENTITY, identity_tone());
    p.grade = g;
    p
}

/// Camera values: a grey ramp over 13 stops and a set of colours.
fn probe_pixels() -> Vec<[f32; 3]> {
    let mut v: Vec<[f32; 3]> = (-9..=4)
        .map(|s| {
            let x = 0.18 * (s as f32).exp2();
            [x, x, x]
        })
        .collect();
    v.extend([
        [0.5, 0.2, 0.1],
        [0.1, 0.4, 0.2],
        [0.05, 0.1, 0.6],
        [0.9, 0.7, 0.05],
        [0.02, 0.01, 0.015],
        [0.3, 0.3, 0.05],
    ]);
    v
}

#[test]
fn every_tonal_control_raises_or_keeps_every_channel() {
    type Setter = fn(&mut GradeParams, f64);
    let controls: [(&str, Setter); 5] = [
        ("brightness", |g, x| g.brightness = x),
        ("shadows", |g, x| g.shadows = x),
        ("highlights", |g, x| g.highlights = x),
        ("black", |g, x| g.black = x),
        ("white", |g, x| g.white = x),
    ];
    let px = probe_pixels();
    for dev in gpus() {
        for (name, set) in controls {
            let mut prev: Option<Vec<[f32; 3]>> = None;
            for step in 0..=20 {
                let mut g = GradeParams::default();
                set(&mut g, -100.0 + 10.0 * f64::from(step));
                let out = run(dev, &grade_chain(g), &px, Stop::C9);
                if let Some(p) = &prev {
                    for (i, (a, b)) in p.iter().zip(&out).enumerate() {
                        for ch in 0..3 {
                            assert!(
                                b[ch] >= a[ch] - 2e-6,
                                "{:?} {name}: channel {ch} of pixel {i} fell from {} to {} at step {step}",
                                dev.mode,
                                a[ch],
                                b[ch]
                            );
                        }
                    }
                }
                prev = Some(out);
            }
        }
    }
}

#[test]
fn contrast_saturation_and_color_balance_are_monotone_in_their_control() {
    let px = probe_pixels();
    for dev in gpus() {
        // Contrast: for a luma between black and the top code value the mean of the three channels
        // (and every channel of a neutral) moves away from grey as the value rises (up above grey, down
        // below it); beyond the ends toward the end. The colour step keeps the mean, so a channel of a
        // coloured pixel need not follow it. The chain's matrix and C7 are the identity, so the luma is
        // the camera value's.
        let lumas: Vec<f64> = px
            .iter()
            .map(|p| logc4_encode(awg4_luminance(p.map(f64::from))))
            .collect();
        let mut prev = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
        for step in 1..=10 {
            let g = GradeParams {
                contrast: 10.0 * f64::from(step),
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for (i, (a, b)) in prev.iter().zip(&out).enumerate() {
                let l = lumas[i];
                let dir = contrast_direction(l);
                let neutral = px[i][0] == px[i][1] && px[i][1] == px[i][2];
                let mean =
                    |x: &[f32; 3]| (f64::from(x[0]) + f64::from(x[1]) + f64::from(x[2])) / 3.0;
                assert!(
                    (mean(b) - mean(a)) * dir >= -2e-6,
                    "{:?} contrast: pixel {i}: mean {} -> {} (luma {l})",
                    dev.mode,
                    mean(a),
                    mean(b)
                );
                for ch in 0..3 {
                    assert!(
                        !neutral || f64::from(b[ch] - a[ch]) * dir >= -2e-6,
                        "{:?} contrast: pixel {i} ch {ch}: {} -> {} (luma {l})",
                        dev.mode,
                        a[ch],
                        b[ch]
                    );
                }
            }
            prev = out;
        }
        // Saturation scales the chroma about the luminance by a factor that rises with the control:
        // the distance of every channel from Y (in linear light) never shrinks as the control rises.
        let mut prev_dist: Vec<[f64; 3]> = vec![[0.0; 3]; px.len()];
        for step in 0..=20 {
            let g = GradeParams {
                saturation: -100.0 + 10.0 * f64::from(step),
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for i in 0..px.len() {
                let lin = px[i].map(f64::from);
                let y = awg4_luminance(lin);
                let o = out[i].map(|c| logc4_decode(f64::from(c)));
                // f32 codes: a relative 5e-6 of the largest channel, and no less than 2e-8 (near black an
                // ulp of a code is about 2e-9 in linear light, and the decode's exp2 near 2^6 about the same).
                let tol = 5e-6 * lin.iter().fold(4e-3f64, |m, c| m.max(c.abs()));
                for ch in 0..3 {
                    let d = (o[ch] - y).abs();
                    assert!(
                        d >= prev_dist[i][ch] - tol,
                        "{:?} saturation: pixel {i} ch {ch} lost chroma at step {step}",
                        dev.mode
                    );
                    prev_dist[i][ch] = d;
                }
            }
        }
        // Colour balance displaces along a fixed direction; the displacement grows with the radius.
        for angle in [0.3f64, 1.9, 4.0] {
            let mut prev_len = 0.0f32;
            let zero = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
            for step in 1..=5 {
                let r = f64::from(step) / 5.0;
                let b = ColorBalance {
                    a: r * angle.cos(),
                    b: r * angle.sin(),
                };
                let g = GradeParams {
                    balance_midtones: b,
                    ..GradeParams::default()
                };
                let out = run(dev, &grade_chain(g), &px, Stop::C9);
                let probe = 3; // a mid-grey sample, where the midtone zone is strongest
                let len: f32 = (0..3)
                    .map(|ch| (out[probe][ch] - zero[probe][ch]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    len >= prev_len - 1e-6,
                    "{:?} balance: angle {angle} radius {r}",
                    dev.mode
                );
                prev_len = len;
            }
            assert!(
                prev_len > 1e-3,
                "colour balance did nothing at the mid-grey probe"
            );
        }
    }
}

#[test]
fn a_single_control_at_its_extremes_keeps_the_neutral_ramp_increasing() {
    let Some(dev) = f32_device() else { return };
    let ramp: Vec<[f32; 3]> = (0..1024)
        .map(|i| {
            let x = 0.18 * (-8.0 + 14.0 * i as f32 / 1023.0).exp2();
            [x, x, x]
        })
        .collect();
    let mut gs: Vec<(String, GradeParams)> = Vec::new();
    for v in [-100.0, 100.0] {
        gs.push((
            format!("contrast {v}"),
            GradeParams {
                contrast: v,
                ..Default::default()
            },
        ));
        gs.push((
            format!("brightness {v}"),
            GradeParams {
                brightness: v,
                ..Default::default()
            },
        ));
        gs.push((
            format!("shadows {v}"),
            GradeParams {
                shadows: v,
                ..Default::default()
            },
        ));
        gs.push((
            format!("highlights {v}"),
            GradeParams {
                highlights: v,
                ..Default::default()
            },
        ));
        gs.push((
            format!("black {v}"),
            GradeParams {
                black: v,
                ..Default::default()
            },
        ));
        gs.push((
            format!("white {v}"),
            GradeParams {
                white: v,
                ..Default::default()
            },
        ));
    }
    for (name, g) in gs {
        let out = run(dev, &grade_chain(g), &ramp, Stop::C9);
        let mut min_slope = f32::MAX;
        for i in 1..out.len() {
            let d = out[i][1] - out[i - 1][1];
            assert!(d > 0.0, "{name}: the neutral ramp folds at point {i}");
            min_slope = min_slope.min(d);
        }
        eprintln!("{name}: smallest step {min_slope:.2e}");
    }
}

/// Saturation keeps the luminance (`docs/02` C9, design of 2026-10-19), checked on the kernel's own
/// C9 output, not only against the CPU: `Y` stays, neutrals stay, the chroma keeps its direction
/// (the hue), `s > 0` never takes a channel below 0, `s = -100` is the grey of the same `Y`, and a
/// pixel `s > 0` leaves alone (`Y <= 0` or a channel at or below 0) keeps its codes bit for bit.
#[test]
fn saturation_keeps_the_luminance_on_the_gpu() {
    let mut px = probe_pixels();
    px.extend([
        [0.8, 0.05, 0.03],
        [0.04, 0.6, 0.06],
        [0.03, 0.08, 0.7],
        [4.0, 1.0, 0.2],
        [0.2, 0.2, 0.05],
        [0.02, 0.02, 0.57], // every channel above Y (r < 0): the plain factor F
        [0.1, 0.1184, 0.5], // r near 0 with a large chroma
        [0.4, -0.01, 0.1],  // a channel below 0: left alone for s > 0
        [0.01, 0.003, 0.2], // Y < 0: left alone for s > 0
    ]);
    for dev in gpus() {
        let base = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
        let mut coloured = 0;
        for s in [-100.0f64, -60.0, -25.0, 25.0, 60.0, 100.0] {
            let g = GradeParams {
                saturation: s,
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for (i, (a, b)) in base.iter().zip(&out).enumerate() {
                let lin = a.map(|c| logc4_decode(f64::from(c)));
                let o = b.map(|c| logc4_decode(f64::from(c)));
                let y = awg4_luminance(lin);
                let min = lin[0].min(lin[1]).min(lin[2]);
                // The scale of the f32 errors: the largest channel, and no less than 4e-3 (near black an
                // ulp of a code is about 2e-9 in linear light, and the decode's exp2 near 2^6 about the same).
                let big = lin.iter().fold(4e-3f64, |m, c| m.max(c.abs()));
                let ctx = format!(
                    "{:?} s={s} pixel {i} {:?}: {lin:?} -> {o:?}",
                    dev.mode, px[i]
                );
                if s > 0.0 && (y <= 0.0 || min <= 0.0) {
                    assert_eq!(
                        a.map(f32::to_bits),
                        b.map(f32::to_bits),
                        "left alone, {ctx}"
                    );
                    continue;
                }
                // The luminance, to the f32 precision of the codes (an ulp of a code is a relative 6e-7 of
                // its linear value; 5e-6 of the largest channel leaves room for a log2 or exp2 a few ulp
                // off, as Metal's fast-math ones are).
                assert!(
                    (awg4_luminance(o) - y).abs() <= 5e-6 * big,
                    "Y moved, {ctx}"
                );
                let neutral = (a[0] - a[1]).abs() <= 1e-7 && (a[1] - a[2]).abs() <= 1e-7;
                if neutral {
                    for ch in 0..3 {
                        assert!((b[ch] - a[ch]).abs() <= 1e-6, "neutral moved, {ctx}");
                    }
                    continue;
                }
                coloured += 1;
                // The hue: rgb' - Y is rgb - Y times one factor, between 0 and F.
                let far = (0..3)
                    .max_by(|&p, &q| (lin[p] - y).abs().total_cmp(&(lin[q] - y).abs()))
                    .unwrap_or(0);
                let lambda = (o[far] - y) / (lin[far] - y);
                assert!(
                    (-1e-5..=1.0 + SATURATION_BOOST * s.max(0.0) / 100.0 + 1e-4).contains(&lambda),
                    "{lambda}, {ctx}"
                );
                for ch in 0..3 {
                    assert!(
                        (o[ch] - y - lambda * (lin[ch] - y)).abs() <= 1e-5 * big,
                        "hue, {ctx}"
                    );
                    if s == -100.0 {
                        assert!((o[ch] - y).abs() <= 5e-6 * big, "s = -100 gives Y, {ctx}");
                    }
                }
                if s > 0.0 {
                    assert!(
                        o.iter().all(|&c| c >= -1e-6 * big),
                        "a channel went below 0, {ctx}"
                    );
                }
            }
        }
        assert!(
            coloured >= 6 * 12,
            "the probe must hold coloured pixels: {coloured}"
        );
    }
}

fn chroma(x: [f64; 3]) -> [f64; 2] {
    [
        E1[0] * x[0] + E1[1] * x[1] + E1[2] * x[2],
        E2[0] * x[0] + E2[1] * x[1] + E2[2] * x[2],
    ]
}

/// Contrast's colour step (`docs/02` C9, the owner's answer of 2026-10-19) on the kernel's own C9
/// output, not only against the CPU: the mean of the three channels moves by `T(L) - L` (the colour
/// step keeps it), the chroma keeps its direction (the hue) and is scaled by `f` between
/// `1 - beta (1 - min(k, 1/k))` and `1 + beta (max(k, 1/k) - 1)`, the `f` of the formula, neutrals stay
/// exactly neutral, and midtones gain colour for c > 0 and lose it for c < 0. Inputs are
/// f16-representable, so every storage mode sees the same pixels.
#[test]
fn contrast_keeps_the_hue_and_adds_colour_on_the_gpu() {
    let mut px = probe_pixels();
    px.extend([
        [0.8, 0.05, 0.03],
        [0.04, 0.6, 0.06],
        [0.03, 0.08, 0.7],
        [4.0, 1.0, 0.2],
        [60.0, 30.0, 5.0],
        [-0.004, 0.01, 0.02],
        [0.0002, 0.0001, 0.0003],
        // Midtones: every LogC4 channel between 0.21 and 0.55.
        [0.25, 0.18, 0.12],
        [0.10, 0.16, 0.30],
        [0.15, 0.28, 0.14],
        [0.9, 0.5, 0.25],
    ]);
    let px: Vec<[f32; 3]> = px.into_iter().map(|p| p.map(f16_round)).collect();
    let lumas: Vec<f64> = px
        .iter()
        .map(|p| logc4_encode(awg4_luminance(p.map(f64::from))))
        .collect();
    for dev in gpus() {
        // With every control at 0 C9 is C8: the kernel's own v.
        let v_all = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
        let (mut checked, mut midtones) = (0usize, 0usize);
        let mut worst_f = 0.0f64;
        for c in [-100.0, -60.0, -30.0, 30.0, 60.0, 100.0] {
            let k = contrast_slope(c);
            let (lo, hi) = (k.min(1.0 / k), k.max(1.0 / k));
            let g = GradeParams {
                contrast: c,
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for i in 0..px.len() {
                let (v, o) = (v_all[i].map(f64::from), out[i].map(f64::from));
                let l = lumas[i];
                let ctx = format!("{:?} c={c} pixel {i} {:?}: {v:?} -> {o:?}", dev.mode, px[i]);
                let big = v.iter().fold(1.0f64, |m, x| m.max(x.abs()));
                let mean = |x: [f64; 3]| (x[0] + x[1] + x[2]) / 3.0;
                let shift = contrast_curve(l, k) - l;
                assert!(
                    (mean(o) - mean(v) - shift).abs() <= 2e-6 * big,
                    "mean, {ctx}"
                );
                if v_all[i][0] == v_all[i][1] && v_all[i][1] == v_all[i][2] {
                    assert!(
                        out[i][0] == out[i][1] && out[i][1] == out[i][2],
                        "neutral, {ctx}"
                    );
                    continue;
                }
                let (cv, co) = (chroma(v), chroma(o));
                let (lv, lo_) = (cv[0].hypot(cv[1]), co[0].hypot(co[1]));
                if lv < 1e-2 {
                    continue;
                }
                checked += 1;
                // The same direction: parallel, and the same way round.
                assert!(
                    (cv[0] * co[1] - cv[1] * co[0]).abs() <= 1e-6 * big * (lv + lo_),
                    "hue, {ctx}"
                );
                assert!(cv[0] * co[0] + cv[1] * co[1] > 0.0, "hue, {ctx}");
                let f = lo_ / lv;
                let p = v.map(|x| contrast_curve(x, k));
                let cp = chroma(p);
                let want = 1.0 + CONTRAST_COLOR * (cp[0].hypot(cp[1]) / lv - 1.0);
                worst_f = worst_f.max((f - want).abs());
                assert!((f - want).abs() <= 1e-4, "f {f} vs {want}, {ctx}");
                assert!(
                    f >= 1.0 - CONTRAST_COLOR * (1.0 - lo) - 1e-4
                        && f <= 1.0 + CONTRAST_COLOR * (hi - 1.0) + 1e-4,
                    "f {f}, {ctx}"
                );
                if v.iter().all(|x| (0.21..=0.55).contains(x)) {
                    midtones += 1;
                    assert_eq!(f > 1.0, c > 0.0, "midtone colour, f {f}, {ctx}");
                }
            }
        }
        assert!(
            checked >= 6 * 15 && midtones >= 6 * 3,
            "{checked} coloured, {midtones} midtone checks"
        );
        eprintln!(
            "{:?}: contrast colour step on {checked} coloured pixel settings, worst |f - f_formula| {worst_f:.1e}",
            dev.mode
        );
    }
}

/// Contrast on chromas of a few f32 ulps (the colour step's neutral test and its guarded division and
/// square root, `docs/02` C9) and on exact neutrals, on the f32-storage device (f16 cannot hold them):
/// every output finite and within the gate of the CPU, exact neutrals exactly neutral.
#[test]
fn contrast_on_near_neutrals_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut px: Vec<[f32; 3]> = Vec::new();
    for s in -10..=8 {
        let x = 0.18 * (s as f32).exp2();
        let up = f32::from_bits(x.to_bits() + 1);
        let up3 = f32::from_bits(x.to_bits() + 3);
        px.extend([
            [x, x, x],
            [x, up, x],
            [up, x, x],
            [x, x, up3],
            [x, x * (1.0 + 1e-6), x],
            [x * (1.0 - 1e-5), x, x * (1.0 + 1e-5)],
        ]);
    }
    for c in [-100.0, -60.0, -30.0, 30.0, 60.0, 100.0] {
        let g = GradeParams {
            contrast: c,
            ..GradeParams::default()
        };
        let params = grade_chain(g);
        let chain = ColorChain::new(&params).expect("chain");
        let out = run(dev, &params, &px, Stop::C9);
        let mut worst = 0.0f64;
        for (i, (o, p)) in out.iter().zip(&px).enumerate() {
            let cpu = chain.trace(p.map(f64::from)).c9_logc4.0;
            for ch in 0..3 {
                assert!(o[ch].is_finite(), "c={c} pixel {i} {p:?}: {o:?}");
                worst = worst.max((f64::from(o[ch]) - cpu[ch]).abs());
            }
            if p[0] == p[1] && p[1] == p[2] {
                assert!(o[0] == o[1] && o[1] == o[2], "c={c} neutral {p:?}: {o:?}");
            }
        }
        eprintln!("contrast {c} on near neutrals: worst |GPU - CPU| at C9 {worst:.1e}");
        assert!(worst <= 2e-4, "c={c}: {worst}");
    }
}

/// Parity of the new Contrast and Saturation (`docs/02` C9) with the CPU reference on pixels made for
/// their branches: a grey ramp from 2^-12 to 2^12 of grey (below black and above the top code value
/// after the curve), deep blues with every channel above `Y` (r < 0), r near 0 with a large chroma,
/// r on both sides of the kernel's series limit (0.05), r near 1, channels at and below 0, `Y <= 0`,
/// negative lumas and 400 seeded random colours, for every pair of Contrast and Saturation in
/// {-100, -60, -30, 0, 30, 60, 100}, alone and with the other controls set, at C9, in every storage
/// mode (inputs are f16-representable).
///
/// Measured on Mesa lavapipe, the same in all three storage modes (with the contrast's colour step and
/// the saturation boost 1.5): worst |GPU - CPU| at C9 4.5e-5 in LogC4 code values against the 2e-4
/// gate, at a code of -72 (a far out-of-gamut blue whose luminance is negative, at Contrast 100 and
/// Saturation -100: an f32 ulp there is 8e-6); 9.4e-6 among the codes of photos (-0.2 to 1.5; 1.2e-5
/// before the boost, at the darkest channel of a strong orange that a boost takes near 0 as
/// `Y - lambda (Y - min)`, a difference of two larger numbers in f32). The CPU reference is f64.
#[test]
fn contrast_and_saturation_match_the_cpu_on_every_branch() {
    let mut px: Vec<[f32; 3]> = (-12..=12)
        .map(|s| {
            let x = 0.18 * (s as f32).exp2();
            [x, x, x]
        })
        .collect();
    px.extend([
        [0.02, 0.02, 0.57],
        [0.02, 0.02, 0.3],
        [0.05, 0.05, 0.3],
        [0.1, 0.1184, 0.5],
        [0.1, 0.1185, 0.5],
        [0.4, 0.38, 0.36],
        [0.5, 0.2, 0.1],
        [0.9, 0.7, 0.05],
        [0.9, 0.7, 0.001],
        [0.3, 0.0, 0.2],
        [0.4, -0.01, 0.1],
        [0.01, 0.003, 0.2],
        [-0.01, -0.02, -0.005],
        [-0.004, 0.01, 0.02],
        [60.0, 30.0, 5.0],
        [400.0, 600.0, 300.0],
        [3000.0, 1000.0, 200.0],
    ]);
    // r = 1 - min / Y on both sides of the series limit, and near 1: red and green set, blue solved.
    for r in [0.001, 0.01, 0.049, 0.0501, 0.051, 0.2, 0.9, 0.999] {
        let (red, green) = (0.3f64, 0.25f64);
        // min = blue = (1 - r) Y with Y = wr red + wg green + wb blue.
        let w = ym_color::gamut::awg4_luminance_row();
        let blue = (1.0 - r) * (w[0] * red + w[1] * green) / (1.0 - (1.0 - r) * w[2]);
        px.push([red as f32, green as f32, blue as f32]);
    }
    let mut rng = Rng(0xC0_5A7);
    for _ in 0..400 {
        let lum = 0.18 * rng.range(-10.0, 7.0).exp2();
        px.push([0, 1, 2].map(|_| (lum * rng.range(-0.05, 2.0)) as f32));
    }
    let px: Vec<[f32; 3]> = px.into_iter().map(|p| p.map(f16_round)).collect();
    let values = [-100.0, -60.0, -30.0, 0.0, 30.0, 60.0, 100.0];
    let others = GradeParams {
        brightness: -20.0,
        shadows: 30.0,
        highlights: -25.0,
        white: 15.0,
        balance_midtones: ColorBalance { a: 0.3, b: -0.4 },
        ..GradeParams::default()
    };
    for dev in gpus() {
        let mut worst = 0.0f64;
        let mut at = String::new();
        // The same among the codes a photo has after C9 (-0.2 to 1.5): the largest differences are
        // at the codes of far out-of-gamut colours (a negative luminance graded to -46), an f32 ulp
        // of which is 4e-6.
        let mut worst_photo = 0.0f64;
        for c in values {
            for s in values {
                for with_others in [false, true] {
                    let base = if with_others {
                        others
                    } else {
                        GradeParams::default()
                    };
                    let g = GradeParams {
                        contrast: c,
                        saturation: s,
                        ..base
                    };
                    let params = grade_chain(g);
                    let chain = ColorChain::new(&params).expect("chain");
                    let out = run(dev, &params, &px, Stop::C9);
                    for (i, (o, p)) in out.iter().zip(&px).enumerate() {
                        let cpu = chain.trace(p.map(f64::from)).c9_logc4.0;
                        for ch in 0..3 {
                            let d = (f64::from(o[ch]) - cpu[ch]).abs();
                            if (-0.2..=1.5).contains(&cpu[ch]) {
                                worst_photo = worst_photo.max(d);
                            }
                            if d > worst {
                                worst = d;
                                at = format!(
                                    "c={c} s={s} others={with_others} pixel {i} {p:?} ch {ch}: GPU {} CPU {}",
                                    o[ch], cpu[ch]
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "{:?}: contrast x saturation on {} pixels, worst |GPU - CPU| at C9 {worst:.1e} (gate 2e-4) at {at}; {worst_photo:.1e} among codes from -0.2 to 1.5",
            dev.mode,
            px.len()
        );
        assert!(worst <= 2e-4, "{:?}: {worst} at {at}", dev.mode);
    }
}
