//! M1-09 monotonicity on the GPU: C7 (the highlight range and the toe) and every C9 control behave as
//! the CPU reference promises (`docs/02` sections C7 and C9, `ym-color` tests `tone.rs` and `grade.rs`).
//!
//! Runs on the f32-storage device (strict ramps need unrounded inputs) and, for the C9 control
//! sweeps, on every device.

mod common;

use common::chains::identity_tone;
use common::{TestGpu, gpus};
use ym_color::consts::LOGC4_GRAY;
use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain};
use ym_color::grade::{ColorBalance, GradeParams};
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
        ramp.last().is_some_and(|&top| f64::from(top) > 0.18 * HIGHLIGHT_CAP_STOPS.exp2() * 1.5),
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
                            // exp2 of the gain wiggle by a few 1e-7; everywhere else the output rises by
                            // at least the input's step (0.4 %), so only the last bits may wiggle.
                            let y = luma(px[i].map(f64::from));
                            let s = (y / 0.18).log2();
                            let at_cap = y > 0.0 && range.stops(s) >= HIGHLIGHT_CAP_STOPS - 1e-9;
                            ranged += usize::from(y > 0.0 && range.gain(y) > 1.0);
                            capped += usize::from(at_cap);
                            let rel = if at_cap { 4e-6 } else { 6e-7 };
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
    assert!(ranged > 10_000 && capped > 10_000, "{ranged} ranged, {capped} capped samples");
    eprintln!(
        "C7 monotone over {checked} parameter sets x {} ramp points ({ranged} samples in the range, {capped} at the cap)",
        grey.len()
    );
}

/// The ALEXA cap on the device: a grey ramp far past it comes out at `0.18 * 2^7.64` (35.9) to f32
/// precision, and nothing comes out above it.
#[test]
fn nothing_leaves_the_device_above_the_alexa_clip() {
    let Some(dev) = f32_device() else { return };
    let cap = (0.18 * HIGHLIGHT_CAP_STOPS.exp2()) as f32;
    let px: Vec<[f32; 3]> = (0..200)
        .map(|i| {
            let v = 0.18 * (7.7 + 0.25 * f64::from(i)).exp2();
            [v as f32, (0.6 * v) as f32, (0.2 * v) as f32]
        })
        .collect();
    for r in [0.0, 0.5, 1.0] {
        for clip_stops in [1.0, 3.58, 9.0] {
            let tone = ToneParams::new(r, 0.0, 0.18 * f64::exp2(clip_stops), 0.0).expect("tone");
            let out = run(dev, &ChainParams::new(IDENTITY, tone), &px, Stop::C7);
            for (o, p) in out.iter().zip(&px) {
                let y = luma(o.map(f64::from));
                assert!(
                    (y / f64::from(cap) - 1.0).abs() < 4e-6,
                    "r {r} clip {clip_stops}: {p:?} -> {o:?} (luma {y}, cap {cap})"
                );
                // The ratios are the input's.
                assert!((f64::from(o[1] / o[0]) - 0.6).abs() < 1e-6);
            }
        }
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
    let gray = LOGC4_GRAY as f32;
    for dev in gpus() {
        // Contrast moves a channel away from grey: up above it, down below it.
        let base = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
        let mut prev = base.clone();
        for step in 1..=10 {
            let g = GradeParams {
                contrast: 10.0 * f64::from(step),
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for (i, (a, b)) in prev.iter().zip(&out).enumerate() {
                for ch in 0..3 {
                    if base[i][ch] > gray {
                        assert!(b[ch] >= a[ch] - 2e-6, "contrast up: pixel {i} ch {ch}");
                    } else {
                        assert!(b[ch] <= a[ch] + 2e-6, "contrast down: pixel {i} ch {ch}");
                    }
                }
            }
            prev = out;
        }
        // Saturation scales every difference between channels (v - M) by 1 + s/100, so the spread
        // between the channels never shrinks as the control rises.
        let neutral = run(dev, &grade_chain(GradeParams::default()), &px, Stop::C9);
        let mut prev_dist: Vec<f32> = vec![0.0; px.len()];
        for step in 0..=20 {
            let g = GradeParams {
                saturation: -100.0 + 10.0 * f64::from(step),
                ..GradeParams::default()
            };
            let out = run(dev, &grade_chain(g), &px, Stop::C9);
            for i in 0..px.len() {
                // The pivot M is not known here without a second render, so measure the spread: the
                // sum of the distances between channels.
                let spread = (out[i][0] - out[i][1]).abs()
                    + (out[i][1] - out[i][2]).abs()
                    + (out[i][0] - out[i][2]).abs();
                assert!(
                    spread >= prev_dist[i] - 3e-6,
                    "{:?} saturation: pixel {i} lost spread",
                    dev.mode
                );
                prev_dist[i] = spread;
            }
            let _ = &neutral;
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

/// Saturation is subtractive (`docs/02` C9, owner's decision of 2026-10-13): the pivot is the
/// pixel's brightest channel M. Checked on the kernel's own C9 output, not only against the CPU:
/// M and neutral pixels stay, `s > 0` pushes the other channels down, `s < 0` pulls them up to M,
/// and `s = -100` makes every channel equal to M.
#[test]
fn saturation_is_subtractive_on_the_gpu() {
    let mut px = probe_pixels();
    px.extend([
        [0.8, 0.05, 0.03],
        [0.04, 0.6, 0.06],
        [0.03, 0.08, 0.7],
        [4.0, 1.0, 0.2],
        [0.2, 0.2, 0.05],
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
            let gain = (1.0 + s / 100.0) as f32;
            for (i, (a, b)) in base.iter().zip(&out).enumerate() {
                let m = a[0].max(a[1]).max(a[2]);
                let ctx = format!("{:?} s={s} pixel {i} {:?}: {a:?} -> {b:?}", dev.mode, px[i]);
                assert!(
                    (b[0].max(b[1]).max(b[2]) - m).abs() <= 1e-7,
                    "M moved, {ctx}"
                );
                let neutral = (a[0] - a[1]).abs() <= 1e-7 && (a[1] - a[2]).abs() <= 1e-7;
                if !neutral {
                    coloured += 1;
                }
                for ch in 0..3 {
                    let want = m + (a[ch] - m) * gain;
                    assert!((b[ch] - want).abs() <= 3e-6, "formula, {ctx}");
                    if neutral {
                        assert!((b[ch] - a[ch]).abs() <= 1e-7, "neutral moved, {ctx}");
                    } else if s > 0.0 {
                        assert!(
                            b[ch] <= a[ch] + 2e-6,
                            "s > 0 must not raise a channel, {ctx}"
                        );
                    } else {
                        assert!(
                            b[ch] >= a[ch] - 2e-6 && b[ch] <= m + 2e-6,
                            "s < 0 must move a channel up toward M, {ctx}"
                        );
                    }
                    if s == -100.0 {
                        assert!((b[ch] - m).abs() <= 1e-6, "s = -100 gives M, {ctx}");
                    }
                }
            }
        }
        assert!(coloured >= 6 * 8, "the probe must hold coloured pixels");
    }
}
