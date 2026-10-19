//! M1-09, M1-10, M1-11 (GPU halves), C11 and C13: the per-pixel colour chain kernel against
//! `ym_color::display::ColorChain`, stage by stage, on random inputs, in every storage mode the
//! device supports.
//!
//! Gates (`docs/02` section Tests, `docs/15`): C7 to C11 (C9b, the colour warp, included) within 2e-4
//! in LogC4 units; the displayed
//! result (C12 and C13) within 1 code value at 10 bits; tiles equal the whole image; the 8-bit
//! dither is the CPU's. Inputs are f16-representable, so the CPU reference sees exactly what the
//! GPU reads; the remaining difference is f32 against f64 arithmetic.

mod common;

use common::chains::{self, camera_image, pixel};
use common::{TestGpu, gpus};
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, OutputTarget, StageTrace, View};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// C7 to C11 gate, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

fn render(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    outputs: &[ColorOutput],
) -> Vec<ColorImage> {
    let chain = chain.clone();
    let img = img.clone();
    let outputs = outputs.to_vec();
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let prepared = PreparedChain::new(ctx, &chain)?;
            outputs
                .iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o))
                .collect()
        })
        .expect("GPU render")
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

/// Largest `|gpu - cpu|` over all pixels and channels, with the pixel where it happens.
fn worst_abs(gpu: &[f32], n: usize, cpu: &[Vec3], tol: impl Fn(usize) -> f64) -> (f64, f64, usize) {
    let mut worst = 0.0f64;
    let mut worst_ratio = 0.0f64;
    let mut at = 0;
    for (i, c) in cpu.iter().enumerate() {
        for ch in 0..3 {
            let d = (f64::from(gpu[ch * n + i]) - c[ch]).abs();
            let ratio = d / tol(i);
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = d;
                at = i;
            }
        }
    }
    (worst, worst_ratio, at)
}

#[test]
fn every_stage_matches_the_cpu_trace() {
    let (w, h) = (67usize, 41usize);
    let img = camera_image(w, h, 0xBEEF);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    let stops = [
        Stop::C6,
        Stop::C7,
        Stop::C8,
        Stop::C9,
        Stop::C9b,
        Stop::C10,
        Stop::C11,
        Stop::C12,
        Stop::C13,
    ];
    for dev in gpus() {
        // The full matrix of cases on the automatic device, a representative subset elsewhere.
        let all = chains::cases();
        let selected: Vec<_> = if dev.mode == gpus()[0].mode || dev.mode == StorageMode::F32 {
            all
        } else {
            all.into_iter()
                .filter(|(name, _)| {
                    name.starts_with("full/viewport")
                        || name.starts_with("saturation")
                        || name.starts_with("contrast")
                        || name.starts_with("baked-look/srgb/25")
                        || name.starts_with("display-lut/p3")
                        || name.starts_with("logc3-look/35")
                        || name.starts_with("warp/full")
                })
                .collect()
        };
        let mut summary: Vec<(String, [f64; 9])> = Vec::new();
        for (name, params) in selected {
            let chain = ColorChain::new(&params).expect("chain");
            let traces: Vec<StageTrace> = cams.iter().map(|&c| chain.trace(c)).collect();
            // Guard against a vacuous comparison: the scene must use a good part of the display range.
            let mid = traces
                .iter()
                .filter(|t| t.c13_encoded.iter().all(|&v| v > 0.02 && v < 0.98))
                .count();
            assert!(
                mid * 10 > n,
                "{name}: only {mid} of {n} pixels are mid-range"
            );
            // And a warp must move a good part of them (by more than five times the gate), or C9b
            // would be compared where it is the identity. The four `warp/` cases (the flow's 12
            // steps), measured on Mesa lavapipe in all three storage modes (with the contrast and
            // saturation of 2026-10-19, the contrast's colour step and the saturation boost 1.5
            // included): C9b 4.2e-7, C10 4.5e-7, C11 4.9e-7 (gate 2e-4), C13 at most 4.2e-5 (gate one
            // 10-bit code). The `contrast/` and `saturation/` cases (alone and on the full chain, -100
            // to 100) are in the per-stage line printed below: C9 8.4e-7 on every chain.
            let warped = traces
                .iter()
                .filter(|t| {
                    (0..3).any(|k| (t.c9b_logc4.0[k] - t.c9_logc4.0[k]).abs() > 5.0 * LOGC4_GATE)
                })
                .count();
            assert_eq!(
                params.warp.is_identity(),
                warped == 0,
                "{name}: {warped} pixels warped"
            );
            if name.starts_with("warp/") {
                assert!(
                    warped * 5 > n,
                    "{name}: only {warped} of {n} pixels are warped"
                );
            }
            let outs: Vec<ColorOutput> = stops.iter().map(|&s| ColorOutput::Stage(s)).collect();
            let gpu = render(dev, &chain, &img, &outs);
            let mut maxes = [0.0f64; 9];
            for (k, stop) in stops.iter().enumerate() {
                let g = planar(&gpu[k]);
                let cpu: Vec<Vec3> = traces
                    .iter()
                    .map(|t| match stop {
                        Stop::C6 => t.c6_linear_awg4,
                        Stop::C7 => t.c7_linear_awg4,
                        Stop::C8 => t.c8_logc4.0,
                        Stop::C9 => t.c9_logc4.0,
                        Stop::C9b => t.c9b_logc4.0,
                        Stop::C10 => t.c10_logc4.0,
                        Stop::C11 => t.c11_logc4.0,
                        Stop::C12 => t.c12_display_linear,
                        _ => t.c13_encoded,
                    })
                    .collect();
                // Linear stages: relative to the size of the input terms (C7 also to its output: the
                // highlight range multiplies a pixel by up to 2^6, and the f32 log2 and exp2 of its gain
                // are good to a relative 1e-5 where the range is steepest, 5e-5 leaving room for a
                // log2 a few ulp off; 5e-5 is 5e-6 in LogC4); LogC4: the 2e-4 gate; display-linear:
                // 2e-4 relative plus a small floor; code values: one 10-bit code.
                let cams_ref = &cams;
                let tol = |i: usize| -> f64 {
                    let mag: f64 = cams_ref[i].iter().map(|v| v.abs()).sum();
                    match stop {
                        Stop::C6 => 6e-6 * mag + 1e-9,
                        Stop::C7 => {
                            6e-6 * mag + 5e-5 * cpu[i].iter().map(|v| v.abs()).sum::<f64>() + 1e-9
                        }
                        Stop::C12 => 5e-4 * cpu[i].iter().cloned().fold(0.0f64, f64::max).max(0.02),
                        Stop::C13 => CODE10,
                        _ => LOGC4_GATE,
                    }
                };
                let (abs, ratio, at) = worst_abs(g, n, &cpu, tol);
                maxes[k] = abs;
                assert!(
                    ratio <= 1.0,
                    "{:?} {name} {stop:?}: |GPU - CPU| = {abs:.3e} (ratio to the gate {ratio:.2}) at pixel {at}: camera {:?}",
                    dev.mode,
                    cams[at]
                );
            }
            summary.push((name, maxes));
        }
        let mut worst = [0.0f64; 9];
        let mut warp_worst = [0.0f64; 9];
        for (name, m) in &summary {
            for k in 0..9 {
                worst[k] = worst[k].max(m[k]);
                if name.starts_with("warp/") {
                    warp_worst[k] = warp_worst[k].max(m[k]);
                }
            }
        }
        eprintln!(
            "{:?}: {} chains, worst |GPU - CPU| per stage: C6 {:.1e}, C7 {:.1e}, C8 {:.1e}, C9 {:.1e}, C9b {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C12 {:.1e} (display-linear), C13 {:.1e} (code values; gate {CODE10:.1e})",
            dev.mode,
            summary.len(),
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            worst[4],
            worst[5],
            worst[6],
            worst[7],
            worst[8]
        );
        eprintln!(
            "{:?}: the warp chains alone: C9b {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4), C13 {:.1e} (code values)",
            dev.mode, warp_worst[4], warp_worst[5], warp_worst[6], warp_worst[8]
        );
    }
}

/// Controls set one at a time, in combinations and to random values: the fixed chains above set
/// every grade control at once, so a mistake in how one control (or one sign, or one zone) is
/// handled when the others are zero would hide in them. 60 seeded parameter sets: a random subset
/// of the C9 controls, random monotone curves, a random CDL look at a random intensity and a random
/// colour warp (0 to 18 pins, about a third locked, some still, controls drawn beyond their ranges:
/// `chains::random_warp`, from a stream of its own so the other draws are those of the sets before
/// the warp existed), for all four outputs, checked at C9, C9b, C10, C11 and C13.
///
/// Measured on Mesa lavapipe, the same in all three storage modes (with the contrast and saturation of
/// 2026-10-19, the contrast's colour step and the saturation boost 1.5 included): C9 4.3e-7, C9b
/// 6.6e-7, C10 6.4e-7, C11 6.2e-7 against the 2e-4 gate, C13 2.0e-5 against one 10-bit code
/// (9.8e-4); 57 of the 60 sets have a warp that acts, and 29 % of all pixels move by more than 1e-3.
#[test]
fn random_parameter_sets_match_the_cpu_trace() {
    use std::sync::Arc;
    use ym_color::cdl::Cdl;
    use ym_color::curve::{Curve, CurveSet};
    use ym_color::display::LookSetting;
    use ym_color::grade::{ColorBalance, GradeParams};
    use ym_color::look::{LookInput, LookTransform};

    let (w, h) = (40usize, 25usize);
    let img = camera_image(w, h, 77);
    let n = w * h;
    for dev in gpus() {
        let mut rng = common::Rng(0x9A7A);
        let mut worst = [0.0f64; 5];
        let (mut warped_sets, mut warped_pixels) = (0usize, 0usize);
        for round in 0..60 {
            let control = |rng: &mut common::Rng, p: f64| {
                if rng.unit() < p {
                    rng.range(-100.0, 100.0)
                } else {
                    0.0
                }
            };
            let balance = |rng: &mut common::Rng| {
                if rng.unit() < 0.4 {
                    let (r, a) = (rng.range(0.0, 1.2), rng.range(0.0, std::f64::consts::TAU));
                    ColorBalance {
                        a: r * a.cos(),
                        b: r * a.sin(),
                    }
                } else {
                    ColorBalance { a: 0.0, b: 0.0 }
                }
            };
            let grade = GradeParams {
                contrast: control(&mut rng, 0.5),
                brightness: control(&mut rng, 0.5),
                shadows: control(&mut rng, 0.4),
                highlights: control(&mut rng, 0.4),
                black: control(&mut rng, 0.4),
                white: control(&mut rng, 0.4),
                balance_shadows: balance(&mut rng),
                balance_midtones: balance(&mut rng),
                balance_highlights: balance(&mut rng),
                saturation: control(&mut rng, 0.5),
            };
            let mut params = chains::base(OutputTarget::ALL[round % 4]);
            params.grade = grade;
            if rng.unit() < 0.5 {
                // A monotone curve on the red channel through 2 to 6 random points.
                let k = 2 + (rng.next_u64() % 5) as usize;
                let mut xs: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
                xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                xs.dedup_by(|a, b| (*a - *b).abs() < 0.02);
                let mut y = 0.0;
                let pts: Vec<(f64, f64)> = xs
                    .iter()
                    .map(|&x| {
                        y = (y + rng.range(0.0, 0.3)).min(1.0);
                        (x, (x * 0.7 + y * 0.3).clamp(0.0, 1.0))
                    })
                    .collect();
                if let Ok(c) = Curve::new(&pts) {
                    params.curves = CurveSet {
                        luma: Curve::identity(),
                        r: c,
                        g: Curve::identity(),
                        b: Curve::identity(),
                    };
                }
            }
            if rng.unit() < 0.5 {
                let mut v = |lo: f64, hi: f64| rng.range(lo, hi);
                let cdl = Cdl::new(
                    [v(0.7, 1.4), v(0.7, 1.4), v(0.7, 1.4)],
                    [v(-0.05, 0.05), v(-0.05, 0.05), v(-0.05, 0.05)],
                    [v(0.7, 1.4), v(0.7, 1.4), v(0.7, 1.4)],
                    v(0.0, 1.6),
                )
                .expect("cdl");
                let intensity = v(0.0, 100.0);
                params.look = Some(LookSetting::new(
                    Arc::new(LookTransform::new(LookInput::LogC4, Some(cdl), None).expect("look")),
                    intensity,
                ));
            }
            let mut warp_rng = common::Rng(0x3A2B_0000 + round as u64);
            params.warp = chains::random_warp(&mut warp_rng);
            warped_sets += usize::from(!params.warp.is_identity());
            let chain = ColorChain::new(&params).expect("chain");
            let gpu = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Stage(Stop::C9),
                    ColorOutput::Stage(Stop::C9b),
                    ColorOutput::Stage(Stop::C10),
                    ColorOutput::Stage(Stop::C11),
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            for i in 0..n {
                let t = chain.trace(pixel(&img, i));
                let cpu = [
                    t.c9_logc4.0,
                    t.c9b_logc4.0,
                    t.c10_logc4.0,
                    t.c11_logc4.0,
                    t.c13_encoded,
                ];
                warped_pixels +=
                    usize::from((0..3).any(|k| (t.c9b_logc4.0[k] - t.c9_logc4.0[k]).abs() > 1e-3));
                for (k, out) in gpu.iter().enumerate() {
                    let v = planar(out);
                    for ch in 0..3 {
                        worst[k] = worst[k].max((f64::from(v[ch * n + i]) - cpu[k][ch]).abs());
                    }
                }
            }
        }
        eprintln!(
            "{:?}: 60 random parameter sets ({warped_sets} with a warp that acts, {warped_pixels} of {} pixels moved by more than 1e-3), worst |GPU - CPU|: C9 {:.1e}, C9b {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C13 {:.1e} (gate {CODE10:.1e})",
            dev.mode,
            60 * n,
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            worst[4]
        );
        // Not vacuous: most sets warp, and a good part of their pixels moves.
        assert!(warped_sets >= 40, "{warped_sets} of 60 sets warp");
        assert!(warped_pixels * 10 > 60 * n, "{warped_pixels} pixels warped");
        for (k, gate) in [LOGC4_GATE, LOGC4_GATE, LOGC4_GATE, LOGC4_GATE, CODE10]
            .into_iter()
            .enumerate()
        {
            assert!(worst[k] <= gate, "{:?} stage {k}: {}", dev.mode, worst[k]);
        }
    }
}

/// The renders of `chain` at every stop that could show a stray warp, as raw bits: C9, C9b, C10, C11,
/// C13 and the DNG stop as f32 planes, and the 8-bit (dithered) and 16-bit pictures.
fn warp_witness(dev: &TestGpu, chain: &ColorChain, img: &RgbImage) -> Vec<ColorImage> {
    render(
        dev,
        chain,
        img,
        &[
            ColorOutput::Stage(Stop::C9),
            ColorOutput::Stage(Stop::C9b),
            ColorOutput::Stage(Stop::C10),
            ColorOutput::Stage(Stop::C11),
            ColorOutput::Stage(Stop::DngLinearAwg4),
            ColorOutput::Stage(Stop::C13),
            ColorOutput::Rgba8 { dither: true },
            ColorOutput::U16 { bits: 16 },
        ],
    )
}

fn same_bits(a: &ColorImage, b: &ColorImage) -> bool {
    match (&a.data, &b.data) {
        (PixelData::F32Planar(x), PixelData::F32Planar(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        _ => a == b,
    }
}

/// C9b, exactness (`docs/02`): a warp that moves nothing (no pins, locked pins only, still pins, pins
/// past the sixteenth) is skipped, so every output is bit for bit the output of the same chain
/// without a warp, and C9b is C9 bit for bit.
#[test]
fn a_warp_that_moves_nothing_leaves_every_output_bit_identical() {
    use ym_color::warp::{WarpParams, WarpPin};
    let img = camera_image(45, 31, 0x1DE7);
    let still = |s: [f64; 2]| WarpPin {
        range: 0.2,
        shadows: 0.0,
        pivot_ev: 3.0,
        ..WarpPin::moving(s, s)
    };
    let identities: Vec<(&str, Vec<WarpPin>)> = vec![
        ("no pins", vec![]),
        (
            "locked pins only",
            vec![
                WarpPin::locked_at([0.03, 0.02]),
                WarpPin {
                    range: 0.25,
                    ..WarpPin::locked_at([-0.02, 0.0])
                },
            ],
        ),
        ("still pins", vec![still([0.04, 0.0]), still([0.0, -0.03])]),
        (
            "still and locked pins",
            vec![still([0.04, 0.0]), WarpPin::locked_at([0.0, 0.02])],
        ),
        ("a moving pin after sixteen locks", {
            let mut pins = vec![WarpPin::locked_at([0.02, 0.02]); 16];
            pins.push(WarpPin::moving([0.03, 0.0], [0.0, 0.05]));
            pins
        }),
    ];
    for dev in gpus() {
        for target in [OutputTarget::Viewport, OutputTarget::Srgb] {
            let params = chains::full(target);
            let plain = warp_witness(dev, &ColorChain::new(&params).expect("chain"), &img);
            assert!(same_bits(&plain[0], &plain[1]), "no warp: C9b is C9");
            for (name, pins) in &identities {
                let mut p = params.clone();
                p.warp = WarpParams { pins: pins.clone() };
                assert!(p.warp.is_identity(), "{name}");
                let got = warp_witness(dev, &ColorChain::new(&p).expect("chain"), &img);
                for (k, (a, b)) in got.iter().zip(&plain).enumerate() {
                    assert!(
                        same_bits(a, b),
                        "{:?} {target:?} {name}: output {k} changed",
                        dev.mode
                    );
                }
            }
            // The control: one moving pin changes C9b and what follows, not C9.
            let mut p = params.clone();
            p.warp = WarpParams {
                pins: vec![WarpPin::moving([0.03, 0.0], [0.0, 0.05])],
            };
            let got = warp_witness(dev, &ColorChain::new(&p).expect("chain"), &img);
            assert!(same_bits(&got[0], &plain[0]), "C9 is before the warp");
            for k in 1..got.len() {
                assert!(!same_bits(&got[k], &plain[k]), "{target:?}: output {k}");
            }
        }
    }
}

/// The uniform packing at its limits: 16 moving pins (every row of `warp_mv`), 1 moving pin and 15
/// locks (every row of `warp_lk` a lock can use), 8 and 8, and 16 moving pins followed by locks (the
/// preparation ignores the pins past the sixteenth, so must the kernel). The sources are the C9
/// chroma of pixels of the image itself, far apart, each with a range that reaches some pixels, so a
/// pin packed into the wrong row, or read from the wrong one, moves pixels by up to its move (0.02 or
/// more, 100 times the gate). Each moving pin moves its own pixel on the device, and each lock holds
/// its own.
///
/// Measured on Mesa lavapipe, the same in all three storage modes (with the contrast and saturation of
/// 2026-10-19, the contrast's colour step and the saturation boost 1.5 included): C9b 9.1e-7, C10
/// 8.7e-7, C11 8.2e-7 (gate 2e-4), C13 4.4e-6 (gate one 10-bit code, 9.8e-4).
#[test]
fn the_most_pins_the_uniform_holds_match_the_cpu() {
    use ym_color::warp::{WARP_MAX_PINS, WarpParams, WarpPin, chroma_of};
    let (w, h) = (96usize, 64usize);
    let img = camera_image(w, h, 0x16F1);
    let n = w * h;
    let base = chains::full(OutputTarget::Viewport);
    let chain0 = ColorChain::new(&base).expect("chain");
    // Candidate pixels: chroma between 0.03 and 0.15 (where a pin acts and the guard is 1), taken
    // greedily so that any two are at least 0.03 apart. (0.12 until the saturation of 2026-10-19,
    // which spreads the image's chroma less: 29 spots fitted below 0.12.)
    let traces: Vec<_> = (0..n).map(|i| chain0.trace(pixel(&img, i))).collect();
    let mut spots: Vec<(usize, [f64; 2])> = Vec::new();
    for (i, t) in traces.iter().enumerate() {
        let c = chroma_of(t.c9_logc4);
        let r = c[0].hypot(c[1]);
        if (0.03..0.15).contains(&r)
            && spots
                .iter()
                .all(|(_, s)| (s[0] - c[0]).hypot(s[1] - c[1]) > 0.03)
        {
            spots.push((i, c));
        }
    }
    assert!(spots.len() >= 2 * WARP_MAX_PINS, "{} spots", spots.len());
    // Moves of 0.02 to 0.032 and ranges of 0.012 to 0.024, from under one range to 2.7 ranges: the
    // flow takes 23 steps for the 16 moving pins together.
    let moving = |k: usize, c: [f64; 2]| {
        let (a, len) = (k as f64 * 0.7, 0.02 + 0.003 * (k % 5) as f64);
        WarpPin {
            range: 0.012 + 0.004 * (k % 4) as f64,
            shadows: 0.6 + 0.1 * (k % 3) as f64,
            highlights: 1.0 - 0.1 * (k % 5) as f64,
            pivot_ev: -3.0 + 0.4 * k as f64,
            exposure_ev: 0.1 * (k as f64 - 8.0),
            ..WarpPin::moving(c, [c[0] + len * a.cos(), c[1] + len * a.sin()])
        }
    };
    let locked = |c: [f64; 2]| WarpPin {
        range: 0.012,
        ..WarpPin::locked_at(c)
    };
    let sets: Vec<(&str, Vec<WarpPin>, usize)> = vec![
        (
            "16 moving",
            (0..16).map(|k| moving(k, spots[k].1)).collect(),
            0,
        ),
        (
            "1 moving, 15 locks",
            std::iter::once(moving(0, spots[0].1))
                .chain((1..16).map(|k| locked(spots[k].1)))
                .collect(),
            15,
        ),
        (
            "8 moving, 8 locks",
            (0..16)
                .map(|k| {
                    if k % 2 == 0 {
                        moving(k, spots[k].1)
                    } else {
                        locked(spots[k].1)
                    }
                })
                .collect(),
            8,
        ),
        (
            "16 moving, then locks (ignored)",
            (0..16)
                .map(|k| moving(k, spots[k].1))
                .chain((16..20).map(|k| locked(spots[k].1)))
                .collect(),
            0,
        ),
    ];
    for dev in gpus() {
        let mut worst = [0.0f64; 4];
        let mut least = f32::INFINITY;
        for (name, pins, lock_count) in &sets {
            let mut params = base.clone();
            params.warp = WarpParams { pins: pins.clone() };
            let warp = ym_color::warp::Warp::new(&params.warp);
            assert_eq!(warp.locks().len(), *lock_count, "{name}");
            let chain = ColorChain::new(&params).expect("chain");
            let outs = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Stage(Stop::C9),
                    ColorOutput::Stage(Stop::C9b),
                    ColorOutput::Stage(Stop::C10),
                    ColorOutput::Stage(Stop::C11),
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            let g: Vec<&[f32]> = outs.iter().map(planar).collect();
            // Every moving pin moves its own pixel, and every lock holds its own (to an f32 rounding
            // of the lock's factor near 0).
            for (k, pin) in pins.iter().take(WARP_MAX_PINS).enumerate() {
                let i = spots[k].0;
                let shift = (0..3)
                    .map(|ch| (g[1][ch * n + i] - g[0][ch * n + i]).abs())
                    .fold(0.0f32, f32::max);
                if pin.locked {
                    assert!(
                        shift <= 1e-6,
                        "{:?} {name}: lock {k} moved by {shift}",
                        dev.mode
                    );
                } else {
                    // Overlapping pins share: the least moved is a short move whose neighbours
                    // dilute it (6.1e-3 on Mesa lavapipe); a pin read from the wrong row would
                    // leave its pixel where it was.
                    assert!(
                        shift > 1e-3,
                        "{:?} {name}: pin {k} moved by {shift}",
                        dev.mode
                    );
                    least = least.min(shift);
                }
            }
            for i in 0..n {
                let t = chain.trace(pixel(&img, i));
                let cpu = [t.c9b_logc4.0, t.c10_logc4.0, t.c11_logc4.0, t.c13_encoded];
                for (k, want) in cpu.iter().enumerate() {
                    for ch in 0..3 {
                        let d = (f64::from(g[k + 1][ch * n + i]) - want[ch]).abs();
                        worst[k] = worst[k].max(d);
                        let gate = if k == 3 { CODE10 } else { LOGC4_GATE };
                        assert!(
                            d <= gate,
                            "{:?} {name}: stage {k} at pixel {i}/{ch}: {d:e}",
                            dev.mode
                        );
                    }
                }
            }
        }
        eprintln!(
            "{:?}: the most pins, worst |GPU - CPU|: C9b {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C13 {:.1e} (gate {CODE10:.1e}); the least a pin moved its own pixel: {least:.1e}",
            dev.mode, worst[0], worst[1], worst[2], worst[3]
        );
    }
}

/// C9b where the flow stretches most (`docs/02` C9b): a long move at the smallest range. Colours just
/// behind the source, on the line of the move, in a sliver about `2 rho^2 / |delta|` long, are
/// stretched along the path, so the flow there magnifies the rounding of f32 (up to about
/// `|delta|^2 / (2 rho^2)` times). The pixels are the camera triple of one colour of middling chroma,
/// each channel scaled by up to 1 % either way (f16-representable), so their C9 chroma is a cloud
/// about 1e-3 across around the pin's source, sliver included; the pin moves that colour by 0.48
/// across neutral at range 0.01 (the flow's 37 steps).
///
/// Measured on Mesa lavapipe, the same in all three storage modes: C9b 7.2e-5 against the 2e-4 gate,
/// with 34 % of the pixels stretched (moved by a tenth to nine tenths of the move). The magnification
/// grows with the move: the same cloud moved by 0.68 gave 9.3e-5, by 0.87 (the source at 0.37 sent
/// to 0.5 on the far side) 2.0e-4, so the longest moves at range 0.01 reach the gate (docs/02 C9b).
#[test]
fn a_long_narrow_move_matches_the_cpu_where_the_flow_stretches() {
    use ym_color::warp::{WARP_RANGE_MIN, WarpParams, WarpPin, chroma_of};
    let params = chains::base(OutputTarget::Viewport);
    let chain0 = ColorChain::new(&params).expect("chain");
    let seed = camera_image(64, 64, 0x10_4E);
    let (i0, s) = (0..64 * 64)
        .map(|i| (i, chroma_of(chain0.trace(pixel(&seed, i)).c9_logc4)))
        .find(|(_, c)| (0.08..0.12).contains(&c[0].hypot(c[1])))
        .expect("a colour of middling chroma");
    let cam = pixel(&seed, i0);
    let (w, h) = (128usize, 96usize);
    let n = w * h;
    let mut rng = common::Rng(0x5117_C9B0);
    let rgb: Vec<f32> = (0..n)
        .flat_map(|_| {
            let k = [
                rng.range(0.99, 1.01),
                rng.range(0.99, 1.01),
                rng.range(0.99, 1.01),
            ];
            (0..3).map(move |ch| common::f16_round((cam[ch] * k[ch]) as f32))
        })
        .collect();
    let img = RgbImage::from_interleaved(w, h, &rgb).expect("image");
    // The target: the source turned by 150 degrees, at 0.4 from neutral.
    let (sin, cos) = 150f64.to_radians().sin_cos();
    let r = 0.4 / s[0].hypot(s[1]);
    let t = [r * (s[0] * cos - s[1] * sin), r * (s[0] * sin + s[1] * cos)];
    let delta = (t[0] - s[0]).hypot(t[1] - s[1]);
    let mut params = params;
    params.warp = WarpParams {
        pins: vec![WarpPin {
            range: WARP_RANGE_MIN,
            ..WarpPin::moving(s, t)
        }],
    };
    let warp = ym_color::warp::Warp::new(&params.warp);
    assert_eq!(warp.steps(), 37, "|delta| = {delta}");
    let chain = ColorChain::new(&params).expect("chain");
    let traces: Vec<StageTrace> = (0..n).map(|i| chain.trace(pixel(&img, i))).collect();
    // Not vacuous: a good part of the cloud is stretched along the path.
    let stretched = traces
        .iter()
        .filter(|t| {
            let (a, b) = (chroma_of(t.c9_logc4), chroma_of(t.c9b_logc4));
            let moved = (b[0] - a[0]).hypot(b[1] - a[1]) / delta;
            (0.1..0.9).contains(&moved)
        })
        .count();
    assert!(stretched * 10 > n, "{stretched} of {n} pixels stretched");
    for dev in gpus() {
        let outs = render(dev, &chain, &img, &[ColorOutput::Stage(Stop::C9b)]);
        let g = planar(&outs[0]);
        let mut worst = 0.0f64;
        for (i, t) in traces.iter().enumerate() {
            for ch in 0..3 {
                let d = (f64::from(g[ch * n + i]) - t.c9b_logc4.0[ch]).abs();
                worst = worst.max(d);
            }
        }
        eprintln!(
            "{:?}: a long narrow move (|delta| {delta:.2}, range 0.01, {} steps), {} of {n} pixels stretched, worst |GPU - CPU| at C9b {worst:.1e} (gate {LOGC4_GATE:.0e})",
            dev.mode,
            warp.steps(),
            stretched
        );
        assert!(worst <= LOGC4_GATE, "{:?}: {worst:e}", dev.mode);
    }
}

#[test]
fn quantised_outputs_match_the_cpu_encoding() {
    let (w, h) = (53usize, 37usize);
    let img = camera_image(w, h, 0x51DE);
    let n = w * h;
    for dev in gpus() {
        for target in [
            OutputTarget::Viewport,
            OutputTarget::Srgb,
            OutputTarget::Rec2020,
        ] {
            let params = chains::full(target);
            let chain = ColorChain::new(&params).expect("chain");
            let out = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Rgba8 { dither: true },
                    ColorOutput::Rgba8 { dither: false },
                    ColorOutput::U16 { bits: 10 },
                    ColorOutput::U16 { bits: 16 },
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            let (
                PixelData::Rgba8(dith),
                PixelData::Rgba8(plain),
                PixelData::U16Planar(q10),
                PixelData::U16Planar(q16),
            ) = (&out[0].data, &out[1].data, &out[2].data, &out[3].data)
            else {
                panic!("formats")
            };
            let mut exact = 0usize;
            let mut total = 0usize;
            for i in 0..n {
                let (x, y) = (i % w, i / w);
                let cam = pixel(&img, i);
                let enc = chain.apply_pixel(cam);
                let cpu8 = chain.quantize_u8(enc, x, y);
                let cpu_plain = chain.quantize_u16(enc, 8);
                let cpu10 = chain.quantize_u16(enc, 10);
                let cpu16 = chain.quantize_u16(enc, 16);
                for ch in 0..3 {
                    total += 1;
                    let g8 = i32::from(dith[4 * i + ch]);
                    assert!(
                        (g8 - i32::from(cpu8[ch])).abs() <= 1,
                        "{target:?} dithered u8 at {i}/{ch}"
                    );
                    exact += usize::from(g8 == i32::from(cpu8[ch]));
                    assert!(
                        (i32::from(plain[4 * i + ch]) - i32::from(cpu_plain[ch])).abs() <= 1,
                        "{target:?} plain u8 at {i}/{ch}"
                    );
                    assert!(
                        (i32::from(q10[ch * n + i]) - i32::from(cpu10[ch])).abs() <= 1,
                        "{target:?} 10-bit at {i}/{ch}"
                    );
                    assert!(
                        (i32::from(q16[ch * n + i]) - i32::from(cpu16[ch])).abs() <= 64,
                        "{target:?} 16-bit at {i}/{ch}: {} vs {}",
                        q16[ch * n + i],
                        cpu16[ch]
                    );
                }
                assert_eq!(dith[4 * i + 3], 255);
            }
            eprintln!(
                "{:?} {target:?}: {exact} of {total} dithered 8-bit samples equal the CPU's ({:.4} %), the rest differ by one code",
                dev.mode,
                100.0 * exact as f64 / total as f64
            );
            // The two differ only where the f32 and f64 values straddle a rounding boundary.
            assert!(
                exact as f64 >= 0.995 * total as f64,
                "{:?} {target:?}: only {exact} of {total} dithered samples equal the CPU's",
                dev.mode
            );
        }
    }
}

#[test]
fn tiles_equal_the_whole_image_including_the_dither() {
    let (w, h) = (131usize, 77usize);
    let img = camera_image(w, h, 0x7115);
    let params = chains::full(OutputTarget::DisplayP3);
    let chain = ColorChain::new(&params).expect("chain");
    for dev in gpus() {
        let whole = render(
            dev,
            &chain,
            &img,
            &[
                ColorOutput::Rgba8 { dither: true },
                ColorOutput::Stage(Stop::C13),
            ],
        );
        for tile in [32usize, 50, 64, 130] {
            let plan = TilePlan::with_halo(w, h, tile, 0);
            let chain2 = chain.clone();
            let img2 = img.clone();
            let results = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let prepared = PreparedChain::new(ctx, &chain2)?;
                    let mut collected = Vec::new();
                    for out in [
                        ColorOutput::Rgba8 { dither: true },
                        ColorOutput::Stage(Stop::C13),
                    ] {
                        let mut tiles = Vec::new();
                        prepared.render_tiled(ctx, &img2, &plan, out, 2, token, &mut |t, im| {
                            tiles.push((t, im));
                            Ok(())
                        })?;
                        collected.push(tiles);
                    }
                    Ok(collected)
                })
                .expect("tiled render");
            // Reassemble and compare bit for bit.
            let PixelData::Rgba8(whole8) = &whole[0].data else {
                panic!()
            };
            let mut tiled8 = vec![0u8; whole8.len()];
            for (t, im) in &results[0] {
                let PixelData::Rgba8(d) = &im.data else {
                    panic!()
                };
                for y in 0..t.core.h {
                    let dst = ((t.core.y + y) * w + t.core.x) * 4;
                    tiled8[dst..dst + t.core.w * 4]
                        .copy_from_slice(&d[y * t.core.w * 4..][..t.core.w * 4]);
                }
            }
            assert_eq!(
                &tiled8, whole8,
                "{:?}: 8-bit dithered tiles of {tile} differ from the whole image",
                dev.mode
            );
            let wholef = planar(&whole[1]);
            let mut worst = 0.0f32;
            for (t, im) in &results[1] {
                let d = planar(im);
                let c: Rect = t.core;
                for ch in 0..3 {
                    for y in 0..c.h {
                        for x in 0..c.w {
                            let a = d[ch * c.w * c.h + y * c.w + x];
                            let b = wholef[ch * w * h + (c.y + y) * w + c.x + x];
                            worst = worst.max((a - b).abs());
                        }
                    }
                }
            }
            assert!(
                worst <= 1e-4,
                "{:?} tile {tile}: tiled vs whole {worst}",
                dev.mode
            );
        }
    }
}

#[test]
fn the_dng_stop_and_the_before_view_match_the_cpu() {
    let (w, h) = (41usize, 29usize);
    let img = camera_image(w, h, 0xD176);
    let n = w * h;
    for dev in gpus() {
        for before in [
            ym_color::display::BeforePoint::CameraLog,
            ym_color::display::BeforePoint::AfterGrade,
        ] {
            let mut params = chains::full(OutputTarget::Viewport);
            params.before = before;
            let chain = ColorChain::new(&params).expect("chain");
            let out = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Stage(Stop::DngLinearAwg4),
                    ColorOutput::Stage(Stop::Before),
                ],
            );
            let dng = planar(&out[0]);
            let bef = planar(&out[1]);
            for i in 0..n {
                let cam = pixel(&img, i);
                let want = chain.apply_dng_pixel(cam);
                let view = chain.apply_view_pixel(View::Before, cam);
                for ch in 0..3 {
                    let tol = 2e-4 * want[ch].abs().max(0.01);
                    assert!(
                        (f64::from(dng[ch * n + i]) - want[ch]).abs() <= tol,
                        "{:?} DNG at {i}/{ch}: {} vs {}",
                        dev.mode,
                        dng[ch * n + i],
                        want[ch]
                    );
                    assert!(
                        (f64::from(bef[ch * n + i]) - view[ch]).abs() <= LOGC4_GATE,
                        "{:?} Before ({before:?}) at {i}/{ch}",
                        dev.mode
                    );
                }
            }
        }
    }
}

/// Largest `|gpu - cpu|` of the Before view's camera default over every pixel and channel, in encoded
/// code values and in display-linear light, for one chain on one device.
fn camera_default_errors(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    n: usize,
) -> (f64, f64, usize) {
    let out = render(dev, chain, img, &[ColorOutput::Stage(Stop::Before)]);
    let gpu = planar(&out[0]);
    let space = chain.space();
    let (mut worst_code, mut worst_linear, mut mid) = (0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        let cam = pixel(img, i);
        let want = chain.apply_view_pixel(View::Before, cam);
        let mut in_range = true;
        for ch in 0..3 {
            let got = f64::from(gpu[ch * n + i]);
            worst_code = worst_code.max((got - want[ch]).abs());
            // Display-linear light: decode both encoded values (the CPU one is clamped like the GPU's).
            let lin_gpu = space.decode_channel(got);
            let lin_cpu = space.decode_channel(want[ch]);
            worst_linear = worst_linear.max((lin_gpu - lin_cpu).abs());
            in_range &= want[ch] > 0.02 && want[ch] < 0.98;
        }
        mid += usize::from(in_range);
    }
    (worst_code, worst_linear, mid)
}

#[test]
fn the_camera_default_before_view_matches_the_cpu_in_every_output_and_storage_mode() {
    let (w, h) = (67usize, 41usize);
    let img = camera_image(w, h, 0xCA3E);
    let n = w * h;
    for dev in gpus() {
        let mut worst = (0.0f64, 0.0f64);
        for target in OutputTarget::ALL {
            for (name, params) in [
                ("base", chains::base(target)),
                ("full", chains::full(target)),
            ] {
                assert_eq!(params.before, ym_color::display::BeforePoint::CameraDefault);
                let chain = ColorChain::new(&params).expect("chain");
                let (code, linear, mid) = camera_default_errors(dev, &chain, &img, n);
                assert!(
                    mid * 10 > n,
                    "{target:?}/{name}: only {mid} of {n} pixels are mid-range"
                );
                // One 10-bit code in the encoded picture, 2e-4 in display-linear light.
                assert!(
                    code <= CODE10,
                    "{:?} {target:?}/{name}: Before differs by {code} (one code is {CODE10})",
                    dev.mode
                );
                assert!(
                    linear <= 2e-4,
                    "{:?} {target:?}/{name}: Before differs by {linear} in linear light",
                    dev.mode
                );
                worst = (worst.0.max(code), worst.1.max(linear));
            }
        }
        eprintln!(
            "camera default Before, {:?}: max |GPU - CPU| {:.2e} in encoded values, {:.2e} in display-linear light",
            dev.mode, worst.0, worst.1
        );
    }
}

#[test]
fn the_camera_default_before_view_ignores_every_setting_on_the_device_too() {
    let (w, h) = (53usize, 37usize);
    let img = camera_image(w, h, 0xB3F0);
    for dev in gpus() {
        // Everything on against nothing on: another fused matrix, tone (the highlight range at full
        // strength), grade, look, curves, white lift.
        let rich = chains::full(OutputTarget::Viewport);
        let mut bare = chains::base(OutputTarget::Viewport);
        bare.matrix = chains::fused_matrix().map(|r| r.map(|v| v * 3.0));
        bare.tone = chains::range_tone(1.0, chains::AUTO_CLIP_STOPS);
        bare.white_lift = ym_color::exposure::white::WhiteLift::new(0.8, 0.5).expect("lift");
        let outs = [
            ColorOutput::Stage(Stop::Before),
            ColorOutput::BeforeRgba8 { dither: true },
            ColorOutput::BeforeRgba8 { dither: false },
        ];
        let a = render(dev, &ColorChain::new(&rich).expect("chain"), &img, &outs);
        let b = render(dev, &ColorChain::new(&bare).expect("chain"), &img, &outs);
        let (fa, fb) = (planar(&a[0]), planar(&b[0]));
        assert!(
            fa.iter().zip(fb).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{:?}: the Before view changed with the settings",
            dev.mode
        );
        for k in 1..3 {
            let (PixelData::Rgba8(x), PixelData::Rgba8(y)) = (&a[k].data, &b[k].data) else {
                panic!("not RGBA8");
            };
            assert_eq!(
                x, y,
                "{:?}: RGBA8 Before changed with the settings",
                dev.mode
            );
        }
    }
}

#[test]
fn non_finite_input_is_sanitised_like_the_cpu() {
    // f16 storage cannot hold NaN or 1e30 faithfully, so this runs on f32 storage.
    let Some(dev) = gpus().iter().find(|d| d.mode == StorageMode::F32) else {
        return;
    };
    let mut px = vec![
        [f32::NAN, 0.2, 0.3],
        [f32::INFINITY, 0.1, 0.1],
        [0.2, f32::NEG_INFINITY, 0.1],
        [1.0e30, -1.0e30, 5.0],
        [1.0e12, 0.5, 0.5],
        [0.18, 0.18, 0.18],
    ];
    px.push([f32::NAN; 3]);
    let w = px.len();
    let inter: Vec<f32> = px.iter().flatten().copied().collect();
    let img = RgbImage::from_interleaved(w, 1, &inter).expect("image");
    for target in OutputTarget::ALL {
        let chain = ColorChain::new(&chains::full(target)).expect("chain");
        let out = render(dev, &chain, &img, &[ColorOutput::Stage(Stop::C13)]);
        let g = planar(&out[0]);
        for (i, p) in px.iter().enumerate() {
            let want = chain.apply_pixel([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            for ch in 0..3 {
                let got = f64::from(g[ch * w + i]);
                assert!(
                    got.is_finite() && (0.0..=1.0).contains(&got),
                    "{target:?} px {i}: {got}"
                );
                assert!(
                    (got - want[ch]).abs() <= 2.0 * CODE10,
                    "{target:?} px {i} ch {ch}: GPU {got} vs CPU {}",
                    want[ch]
                );
            }
        }
    }
}

#[test]
fn rendering_twice_gives_identical_bytes() {
    let img = camera_image(64, 33, 3);
    let chain = ColorChain::new(&chains::full(OutputTarget::Viewport)).expect("chain");
    for dev in gpus() {
        let a = render(dev, &chain, &img, &[ColorOutput::Rgba8 { dither: true }]);
        let b = render(dev, &chain, &img, &[ColorOutput::Rgba8 { dither: true }]);
        assert_eq!(a, b);
    }
}

#[test]
fn bad_bit_depths_and_mismatched_plans_are_refused() {
    let Some(dev) = gpus().first() else { return };
    let img = camera_image(16, 16, 1);
    let chain = ColorChain::new(&chains::base(OutputTarget::Srgb)).expect("chain");
    let c2 = chain.clone();
    let i2 = img.clone();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedChain::new(ctx, &c2)?;
            p.render(ctx, &i2, (0, 0), ColorOutput::U16 { bits: 17 })
        });
    assert!(
        matches!(r, Err(ym_gpu::device::GpuError::InvalidInput(_))),
        "{r:?}"
    );
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedChain::new(ctx, &chain)?;
            let plan = TilePlan::with_halo(32, 32, 16, 0);
            p.render_tiled(
                ctx,
                &img,
                &plan,
                ColorOutput::Stage(Stop::C13),
                1,
                &CancelToken::new(),
                &mut |_, _| Ok(()),
            )
        });
    assert!(
        matches!(r, Err(ym_gpu::device::GpuError::InvalidInput(_))),
        "{r:?}"
    );
}

// ------------------------------------------------------------------------------------------------
// Persistent tables, device in / device out, the mask plane, the Before view as RGBA8
// ------------------------------------------------------------------------------------------------

use std::sync::Arc;
use std::time::Instant;

use common::chains::{cube_lut, logc4_look, tweak};
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::LookSetting;
use ym_color::look::LookTransform;
use ym_color::lut::Lut;
use ym_gpu::device::GpuContext;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::color::{ChainTables, release_tables};
use ym_gpu::stream::PassList;

/// Chains of different shape for the device-API tests: ACES 2, a log-to-log look, a baked look, a
/// display LUT with a shaper-less cube, curves.
fn view_chains() -> Vec<(String, ym_color::display::ChainParams)> {
    chains::cases()
        .into_iter()
        .filter(|(name, _)| {
            name.starts_with("full/viewport")
                || name.starts_with("auto/srgb")
                || name.starts_with("baked-look/srgb/25")
                || name.starts_with("display-lut/p3")
                || name.starts_with("logc3-look/35")
                || name.starts_with("saturation+full/+60")
        })
        .collect()
}

/// Runs `body` on the automatic (f32) test device.
fn on_f32<R: Send + 'static>(
    body: impl FnOnce(&GpuContext) -> Result<R, ym_gpu::device::GpuError> + Send + 'static,
) -> Option<R> {
    let dev = gpus().first()?;
    assert_eq!(
        dev.mode,
        StorageMode::F32,
        "the automatic device stores f32"
    );
    Some(
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| body(ctx))
            .expect("GPU job"),
    )
}

#[test]
fn device_to_device_equals_the_host_wrapper_bit_for_bit() {
    let (w, h) = (67usize, 41usize);
    let img = camera_image(w, h, 0xD0D0);
    for (name, params) in view_chains() {
        let chain = ColorChain::new(&params).expect("chain");
        let img = img.clone();
        let out = on_f32(move |ctx| {
            let prepared = PreparedChain::new(ctx, &chain)?;
            let src = GpuImage::upload_rgb(ctx, "device test input", &img)?;
            let mut all = Vec::new();
            for stop in [Stop::C6, Stop::C11, Stop::C12, Stop::C13, Stop::Before] {
                let host = prepared.render(ctx, &img, (0, 0), ColorOutput::Stage(stop))?;
                let dst = GpuImage::alloc(ctx, "device test out", w, h, 3, img.place)?;
                let mut passes = PassList::new(ctx, "device colour");
                prepared.encode_stage(&mut passes, &src, Rect::whole(w, h), stop, &dst, (0, 0))?;
                let id = dst.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                let device = dst.take(&done, id)?;
                all.push((stop, planar_owned(&host), device));
            }
            Ok(all)
        })
        .expect("device");
        for (stop, host, device) in out {
            assert_eq!(
                host.len(),
                device.len(),
                "{name} {stop:?}: the lengths differ"
            );
            for (i, (a, b)) in host.iter().zip(&device).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} {stop:?}: sample {i} differs between the host wrapper and the device entry point"
                );
            }
        }
    }
}

fn planar_owned(img: &ColorImage) -> Vec<f32> {
    planar(img).to_vec()
}

#[test]
fn a_window_lands_at_its_offset_in_a_wider_destination_and_touches_nothing_else() {
    // The export writes tiles into the slab of their row: the colour pass takes a window of the
    // source and writes it at an x offset of a wider image, an odd width included (the pair of
    // pixels a thread handles must not spill into the neighbour).
    let (w, h) = (53usize, 29usize);
    let img = camera_image(w, h, 0xA11);
    let chain = ColorChain::new(&chains::full(OutputTarget::Srgb)).expect("chain");
    let (img2, chain2) = (img.clone(), chain.clone());
    let (whole, slab) = on_f32(move |ctx| {
        let prepared = PreparedChain::new(ctx, &chain2)?;
        let whole = prepared.render(ctx, &img2, (0, 0), ColorOutput::Stage(Stop::C12))?;
        let src = GpuImage::upload_rgb(ctx, "slab test input", &img2)?;
        // A destination of 80 x 40 holding the picture at (10, 5), written in two windows of odd
        // widths (21 and 32 columns), the rest left alone.
        let dst = GpuImage::alloc(ctx, "slab", 80, 40, 3, img2.place)?;
        let mut passes = PassList::new(ctx, "slab");
        for (x0, wid) in [(0usize, 21usize), (21, 32)] {
            prepared.encode_stage(
                &mut passes,
                &src,
                Rect::new(x0, 0, wid, h),
                Stop::C12,
                &dst,
                (10 + x0, 5),
            )?;
        }
        let id = dst.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        Ok((planar_owned(&whole), dst.take(&done, id)?))
    })
    .expect("device");
    for c in 0..3 {
        for y in 0..40usize {
            for x in 0..80usize {
                let got = slab[c * 80 * 40 + y * 80 + x];
                let inside = (10..10 + w).contains(&x) && (5..5 + h).contains(&y);
                if inside {
                    let want = whole[c * w * h + (y - 5) * w + (x - 10)];
                    assert_eq!(got.to_bits(), want.to_bits(), "plane {c} at ({x}, {y})");
                } else {
                    assert!(got.is_nan(), "plane {c} at ({x}, {y}) was written: {got}");
                }
            }
        }
    }
}

#[test]
fn the_mask_plane_keeps_the_letterbox_out_of_the_colour_pass() {
    // A source with a fourth plane (the sampler's output): masked pixels are 0 in a stage output and
    // carry the mask on; the others are the plain render. In RGBA8 they are the background colour.
    let (w, h) = (45usize, 22usize);
    let img = camera_image(w, h, 0xFACE);
    let mask: Vec<f32> = (0..w * h)
        .map(|i| f32::from(u8::from((i % w) < 7 || (i / w) > 17 || i % 11 == 3)))
        .collect();
    let chain = ColorChain::new(&chains::full(OutputTarget::Viewport)).expect("chain");
    let (img2, mask2) = (img.clone(), mask.clone());
    let (plain, stage, rgba, plain_rgba) = on_f32(move |ctx| {
        let prepared = PreparedChain::new(ctx, &chain)?;
        let plain = prepared.render(ctx, &img2, (0, 0), ColorOutput::Stage(Stop::C12))?;
        let plain_rgba =
            prepared.render(ctx, &img2, (0, 0), ColorOutput::Rgba8 { dither: true })?;
        let mut data = img2.data.clone();
        data.extend_from_slice(&mask2);
        let src = GpuImage::upload(ctx, "masked input", w, h, 4, &data, img2.place)?;
        let dst = GpuImage::alloc(ctx, "masked out", w, h, 4, img2.place)?;
        let words = GpuImage::alloc(ctx, "unused", 2, 2, 3, img2.place)?; // keeps the pool busy
        drop(words);
        let rgba_buf = ctx.pool().acquire(
            "rgba",
            (w * h * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let mut passes = PassList::new(ctx, "masked");
        prepared.encode_stage(
            &mut passes,
            &src,
            Rect::whole(w, h),
            Stop::C12,
            &dst,
            (0, 0),
        )?;
        prepared.encode_rgba8(
            &mut passes,
            &src,
            Rect::whole(w, h),
            false,
            true,
            (0, 0),
            [200, 100, 50],
            &rgba_buf,
            w,
            (0, 0),
        )?;
        let a = dst.encode_read(&mut passes)?;
        let b = passes.read(&rgba_buf, 0, (w * h * 4) as u64)?;
        let done = passes.submit()?.finish()?;
        Ok((
            planar_owned(&plain),
            dst.take(&done, a)?,
            done.read(b).to_vec(),
            match plain_rgba.data {
                PixelData::Rgba8(v) => v,
                other => panic!("{other:?}"),
            },
        ))
    })
    .expect("device");
    let n = w * h;
    for i in 0..n {
        let gone = mask[i] != 0.0;
        for c in 0..3 {
            let got = stage[c * n + i];
            if gone {
                assert_eq!(got.to_bits(), 0, "pixel {i} plane {c}: outside is zero");
            } else {
                assert_eq!(
                    got.to_bits(),
                    plain[c * n + i].to_bits(),
                    "pixel {i} plane {c}"
                );
            }
        }
        assert_eq!(stage[3 * n + i], mask[i], "pixel {i}: the mask follows");
        if gone {
            assert_eq!(&rgba[4 * i..4 * i + 4], &[200, 100, 50, 255], "pixel {i}");
        } else {
            assert_eq!(
                &rgba[4 * i..4 * i + 4],
                &plain_rgba[4 * i..4 * i + 4],
                "pixel {i}"
            );
        }
    }
}

#[test]
fn the_before_view_is_an_rgba8_stop() {
    let (w, h) = (58usize, 31usize);
    let img = camera_image(w, h, 0xBEF0);
    for (name, params) in view_chains().into_iter().take(3) {
        let chain = ColorChain::new(&params).expect("chain");
        for dev in gpus() {
            let gpu = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::BeforeRgba8 { dither: true },
                    ColorOutput::BeforeRgba8 { dither: false },
                    ColorOutput::Stage(Stop::Before),
                ],
            );
            let PixelData::Rgba8(dithered) = &gpu[0].data else {
                panic!("not RGBA8");
            };
            let PixelData::Rgba8(rounded) = &gpu[1].data else {
                panic!("not RGBA8");
            };
            // The CPU's Before view, 8-bit with the chain's dither (on) and rounded.
            let frame = ym_color::display::Frame::planar(w, h, &img.data).expect("frame");
            let mut cpu = vec![0u8; 3 * w * h];
            chain
                .render_u8(View::Before, &frame, (0, 0), &mut cpu, 1)
                .expect("cpu");
            let f32s = planar(&gpu[2]);
            let n = w * h;
            let (mut off_by_more, mut total) = (0usize, 0usize);
            for i in 0..n {
                assert_eq!(dithered[4 * i + 3], 255);
                for c in 0..3 {
                    // Rounded codes are the f32 stage, rounded.
                    let want = (f32s[c * n + i].clamp(0.0, 1.0) * 255.0 + 0.5).floor() as i32;
                    assert_eq!(i32::from(rounded[4 * i + c]), want, "{name} pixel {i}");
                    // Dithered codes agree with the CPU's within a code (f32 against f64 moves a
                    // value across a dither threshold now and then).
                    let d = (i32::from(dithered[4 * i + c]) - i32::from(cpu[3 * i + c])).abs();
                    assert!(d <= 1, "{name} pixel {i} channel {c}: {d} codes");
                    off_by_more += usize::from(d != 0);
                    total += 1;
                }
            }
            assert!(
                off_by_more * 100 <= total,
                "{name}: {off_by_more} of {total} codes differ from the CPU's"
            );
        }
    }
}

/// A fresh set of tables against a freshly opened counter reading: the uploads a prepare costs.
fn prepare_cost(
    ctx: &GpuContext,
    tables: &mut ChainTables,
    chain: &ColorChain,
) -> ym_gpu::stream::CounterSnapshot {
    let before = ctx.counters();
    let _ = PreparedChain::with_tables(ctx, tables, chain).expect("prepare");
    ctx.counters().since(before)
}

fn chain_with(look: Option<Arc<LookTransform>>, curves: CurveSet) -> ColorChain {
    let mut p = chains::base(OutputTarget::Viewport);
    p.grade = chains::grade_params();
    p.look = look.map(|l| LookSetting::new(l, 80.0));
    p.curves = curves;
    ColorChain::new(&p).expect("chain")
}

fn other_look(bump: f64) -> Arc<LookTransform> {
    let lut: Lut = cube_lut(17, move |c| {
        let t = tweak(c);
        [t[0] + bump, t[1], t[2]]
    });
    Arc::new(LookTransform::new(ym_color::look::LookInput::LogC4, None, Some(lut)).expect("look"))
}

#[test]
fn an_unchanged_chain_uploads_nothing_a_curve_edit_16_kb_and_a_look_one_cube_once() {
    let look_a = other_look(0.0);
    let look_b = other_look(0.03);
    let edited = CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.5, 0.45), (1.0, 1.0)]).expect("curve"),
        ..CurveSet::default()
    };
    let cube_bytes =
        |l: &LookTransform| (l.lut().expect("lut").cube.to_rgba_f32().len() * 4) as u64;
    let (ba, bb) = (cube_bytes(&look_a), cube_bytes(&look_b));
    let measured = on_f32(move |ctx| {
        let mut t = ChainTables::new();
        let a = chain_with(Some(Arc::clone(&look_a)), CurveSet::default());
        let first = prepare_cost(ctx, &mut t, &a);
        // 1. The same chain again, and a clone of it: nothing is uploaded.
        let again = prepare_cost(ctx, &mut t, &a);
        let cloned = prepare_cost(ctx, &mut t, &a.clone());
        // 2. A grade edit changes the per-frame block only.
        let mut p = a.params().clone();
        p.grade.contrast = 33.0;
        let graded = prepare_cost(ctx, &mut t, &ColorChain::new(&p).expect("chain"));
        // 3. A curve edit: the 16 KB of the four tables.
        let c = chain_with(Some(Arc::clone(&look_a)), edited.clone());
        let curve = prepare_cost(ctx, &mut t, &c);
        let curve_again = prepare_cost(ctx, &mut t, &c);
        // 4. A look change: the new cube once; going back finds the old one; the same content under
        //    another Arc (a reloaded look) is found by its hash.
        let b = chain_with(Some(Arc::clone(&look_b)), CurveSet::default());
        let look_new = prepare_cost(ctx, &mut t, &b);
        let look_back = prepare_cost(ctx, &mut t, &a);
        let reloaded = chain_with(Some(other_look(0.03)), CurveSet::default());
        let look_reloaded = prepare_cost(ctx, &mut t, &reloaded);
        // 5. No look at all: nothing.
        let none = chain_with(None, CurveSet::default());
        let look_none = prepare_cost(ctx, &mut t, &none);
        Ok((
            first,
            again,
            cloned,
            graded,
            curve,
            curve_again,
            look_new,
            look_back,
            look_reloaded,
            look_none,
            t.lut_count(),
            t.curve_set_count(),
        ))
    })
    .expect("device");
    let (
        first,
        again,
        cloned,
        graded,
        curve,
        curve_again,
        look_new,
        look_back,
        reloaded,
        none,
        luts,
        sets,
    ) = measured;
    eprintln!(
        "first prepare: {} uploads, {} bytes",
        first.uploads, first.uploaded_bytes
    );
    assert!(
        first.uploaded_bytes > ba,
        "the first chain makes the tables"
    );
    for (what, c) in [
        ("the same chain", again),
        ("a clone", cloned),
        ("a grade edit", graded),
    ] {
        assert_eq!(c.uploads, 0, "{what}: uploads");
        assert_eq!(c.uploaded_bytes, 0, "{what}: bytes");
        assert_eq!(c.submissions, 0, "{what}: submissions");
    }
    assert_eq!(
        curve.uploaded_bytes,
        4 * 1024 * 4,
        "a curve edit uploads the 16 KB of the curves"
    );
    assert_eq!(curve.uploads, 1);
    assert_eq!(curve_again.uploaded_bytes, 0);
    assert_eq!(
        look_new.uploaded_bytes, bb,
        "a new look uploads its cube, once"
    );
    assert_eq!(look_new.uploads, 1);
    assert_eq!(look_back.uploaded_bytes, 0, "the first look is still there");
    assert_eq!(
        reloaded.uploaded_bytes, 0,
        "the same cube under another Arc is found by content"
    );
    assert_eq!(none.uploaded_bytes, 0);
    assert_eq!(luts, 2, "two distinct cubes are held");
    assert_eq!(sets, 2, "and two distinct curve sets");
}

#[test]
fn a_prepared_chain_from_the_cache_renders_the_bits_of_a_fresh_one() {
    // No stale table after an edit: render a chain, edit a curve and the look, go back, and compare
    // every render with the one a private, empty set of tables makes.
    let img = camera_image(41, 27, 0x5EED);
    let look_a = other_look(0.0);
    let look_b = other_look(0.05);
    let curves_b = CurveSet {
        r: Curve::new(&[(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]).expect("curve"),
        ..chains::curves()
    };
    let sequence = vec![
        chain_with(Some(Arc::clone(&look_a)), chains::curves()),
        chain_with(Some(Arc::clone(&look_b)), chains::curves()),
        chain_with(Some(Arc::clone(&look_b)), curves_b.clone()),
        chain_with(Some(Arc::clone(&look_a)), chains::curves()),
        chain_with(None, curves_b),
        chain_with(Some(look_b), chains::curves()),
    ];
    let result = on_f32(move |ctx| {
        release_tables(ctx);
        let mut out = Vec::new();
        for chain in &sequence {
            let cached = PreparedChain::new(ctx, chain)?.render(
                ctx,
                &img,
                (0, 0),
                ColorOutput::Stage(Stop::C13),
            )?;
            let mut fresh_tables = ChainTables::new();
            let fresh = PreparedChain::with_tables(ctx, &mut fresh_tables, chain)?.render(
                ctx,
                &img,
                (0, 0),
                ColorOutput::Stage(Stop::C13),
            )?;
            out.push((cached, fresh));
        }
        release_tables(ctx);
        Ok(out)
    })
    .expect("device");
    let mut distinct = std::collections::BTreeSet::new();
    for (i, (cached, fresh)) in result.iter().enumerate() {
        assert_eq!(
            cached, fresh,
            "step {i}: the cached tables gave other pixels"
        );
        distinct.insert(
            planar(cached)
                .iter()
                .fold(0u32, |a, v| a.wrapping_add(v.to_bits())),
        );
    }
    assert!(distinct.len() >= 4, "the edits must change the picture");
}

#[test]
fn preparing_an_unchanged_chain_costs_microseconds() {
    // "An unchanged chain costs under 0.2 ms of CPU" (docs/05, view kernels): the average over many
    // prepares of the same chain through the device's own tables.
    let mut p = chains::full(OutputTarget::Viewport);
    p.look = Some(LookSetting::new(logc4_look(), 70.0));
    let chain = ColorChain::new(&p).expect("chain");
    let ms = on_f32(move |ctx| {
        let _warm = PreparedChain::new(ctx, &chain)?;
        let before = ctx.counters();
        let t = Instant::now();
        let n = 200;
        for _ in 0..n {
            let _ = PreparedChain::new(ctx, &chain)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
        assert_eq!(ctx.counters().since(before).uploads, 0);
        Ok(ms)
    })
    .expect("device");
    eprintln!("prepare of an unchanged chain: {ms:.4} ms");
    assert!(ms < 0.2, "{ms} ms per prepare");
}

#[test]
fn device_images_need_an_f32_device() {
    // The f16 modes are the legacy kernels' (`docs/17-gpu.md` section 2): a device image cannot be
    // bound to a kernel that reads f16, and says so instead of reading garbage.
    for dev in gpus().iter().skip(1) {
        let img = camera_image(8, 8, 1);
        let chain = ColorChain::new(&chains::base(OutputTarget::Srgb)).expect("chain");
        let r = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let prepared = PreparedChain::new(ctx, &chain)?;
                let src = GpuImage::upload_rgb(ctx, "f16 device", &img)?;
                let dst = GpuImage::alloc(ctx, "f16 device out", 8, 8, 3, img.place)?;
                let mut passes = PassList::new(ctx, "f16");
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    Rect::whole(8, 8),
                    Stop::C12,
                    &dst,
                    (0, 0),
                )
            });
        assert!(
            matches!(r, Err(ym_gpu::device::GpuError::InvalidInput(_))),
            "{:?}: {r:?}",
            dev.mode
        );
    }
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    common::limits::assert_max_binding_bytes(
        "colour",
        Rect::new(0, 0, 2000, 1250),
        ym_gpu::kernels::color::max_binding_bytes,
    );
}

#[test]
fn device_entry_points_refuse_bad_windows_and_planes() {
    let img = camera_image(16, 16, 2);
    let chain = ColorChain::new(&chains::base(OutputTarget::Srgb)).expect("chain");
    let r = on_f32(move |ctx| {
        let prepared = PreparedChain::new(ctx, &chain)?;
        let src = GpuImage::upload_rgb(ctx, "bad", &img)?;
        let dst = GpuImage::alloc(ctx, "bad out", 16, 16, 3, img.place)?;
        let four = GpuImage::alloc(ctx, "bad four", 16, 16, 4, img.place)?;
        let mut results = Vec::new();
        let mut passes = PassList::new(ctx, "bad");
        // A window outside the source, an empty one, one that does not fit the destination, a
        // destination with another number of planes, a one-plane source.
        results.push(prepared.encode_stage(
            &mut passes,
            &src,
            Rect::new(8, 8, 16, 16),
            Stop::C12,
            &dst,
            (0, 0),
        ));
        results.push(prepared.encode_stage(
            &mut passes,
            &src,
            Rect::new(0, 0, 0, 4),
            Stop::C12,
            &dst,
            (0, 0),
        ));
        results.push(prepared.encode_stage(
            &mut passes,
            &src,
            Rect::whole(16, 16),
            Stop::C12,
            &dst,
            (1, 0),
        ));
        results.push(prepared.encode_stage(
            &mut passes,
            &src,
            Rect::whole(16, 16),
            Stop::C12,
            &four,
            (0, 0),
        ));
        let one = GpuImage::alloc(ctx, "bad one", 16, 16, 1, img.place)?;
        results.push(prepared.encode_stage(
            &mut passes,
            &one,
            Rect::whole(16, 16),
            Stop::C12,
            &one,
            (0, 0),
        ));
        Ok(results.into_iter().map(|r| r.is_err()).collect::<Vec<_>>())
    })
    .expect("device");
    assert_eq!(r, vec![true; 5]);
}

#[test]
fn tiny_masked_windows_land_exactly_at_odd_offsets_in_both_outputs() {
    // The colour pass handles two pixels a thread: windows of 1 x 1, 1 x N, N x 1 and odd widths, cut
    // from the middle of a masked source and written at odd offsets of a wider destination, as stage
    // planes (mask plane included) and as RGBA8 words with the dither at the window's absolute
    // position. Everything outside the window stays untouched.
    let (w, h) = (9usize, 6usize);
    let img = camera_image(w, h, 0x7171);
    let mask: Vec<f32> = (0..w * h)
        .map(|i| f32::from(u8::from(i % 4 == 1 || i / w == 3)))
        .collect();
    let chain = ColorChain::new(&chains::full(OutputTarget::Viewport)).expect("chain");
    let windows = [
        Rect::new(0, 0, 1, 1),
        Rect::new(4, 2, 1, 3),
        Rect::new(2, 1, 3, 1),
        Rect::new(1, 0, 7, 5),
        Rect::new(0, 0, 9, 6),
    ];
    let bg = [7u8, 8, 9];
    let (img2, mask2) = (img.clone(), mask.clone());
    let (plain_stage, plain_rgba, stages, words) = on_f32(move |ctx| {
        let prepared = PreparedChain::new(ctx, &chain)?;
        let plain_stage = prepared.render(ctx, &img2, (0, 0), ColorOutput::Stage(Stop::C12))?;
        let mut data = img2.data.clone();
        data.extend_from_slice(&mask2);
        let src = GpuImage::upload(ctx, "tiny masked", w, h, 4, &data, img2.place)?;
        let (dw, dh) = (20usize, 14usize);
        let mut stages = Vec::new();
        let mut words = Vec::new();
        let mut plain_rgba = Vec::new();
        for (k, win) in windows.into_iter().enumerate() {
            let at = (3 + 2 * (k % 2), 1 + k);
            // Stage planes.
            let dst = GpuImage::alloc(ctx, "tiny dst", dw, dh, 4, img2.place)?;
            let mut passes = PassList::new(ctx, "tiny stage");
            prepared.encode_stage(&mut passes, &src, win, Stop::C12, &dst, at)?;
            let id = dst.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            stages.push((win, at, dst.take(&done, id)?));
            // RGBA8 words in a buffer of 25 words a row.
            let stride = 25usize;
            let bytes = (stride * dh * 4) as u64;
            let buf = ctx.pool().acquire(
                "tiny rgba",
                bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )?;
            let mut passes = PassList::new(ctx, "tiny rgba");
            prepared.encode_rgba8(
                &mut passes,
                &src,
                win,
                false,
                true,
                (win.x, win.y),
                bg,
                &buf,
                stride,
                at,
            )?;
            let id = passes.read(&buf, 0, bytes)?;
            let done = passes.submit()?.finish()?;
            words.push((win, at, stride, done.read(id).to_vec()));
            // The plain render of the same window, for the dither at the same absolute position.
            let part = img2.extract(win);
            let r = prepared.render(
                ctx,
                &part,
                (win.x, win.y),
                ColorOutput::Rgba8 { dither: true },
            )?;
            let PixelData::Rgba8(v) = r.data else {
                panic!("not RGBA8")
            };
            plain_rgba.push(v);
        }
        Ok((planar_owned(&plain_stage), plain_rgba, stages, words))
    })
    .expect("device");
    let (dw, dh) = (20usize, 14usize);
    for (win, at, got) in &stages {
        for c in 0..4 {
            for y in 0..dh {
                for x in 0..dw {
                    let g = got[c * dw * dh + y * dw + x];
                    let inside =
                        (at.0..at.0 + win.w).contains(&x) && (at.1..at.1 + win.h).contains(&y);
                    if !inside {
                        assert!(g.is_nan(), "{win:?}: plane {c} at ({x}, {y}) was written");
                        continue;
                    }
                    let (sx, sy) = (win.x + x - at.0, win.y + y - at.1);
                    let gone = mask[sy * w + sx] != 0.0;
                    let want = match (c, gone) {
                        (3, _) => mask[sy * w + sx],
                        (_, true) => 0.0,
                        (c, false) => plain_stage[c * w * h + sy * w + sx],
                    };
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{win:?}: plane {c} at ({x}, {y})"
                    );
                }
            }
        }
    }
    for ((win, at, stride, got), plain) in words.iter().zip(&plain_rgba) {
        for y in 0..dh {
            for x in 0..*stride {
                let g = &got[4 * (y * stride + x)..4 * (y * stride + x) + 4];
                let inside = (at.0..at.0 + win.w).contains(&x) && (at.1..at.1 + win.h).contains(&y);
                if !inside {
                    // Outside the window the pooled buffer still holds the poison fill.
                    assert_eq!(
                        u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
                        ym_gpu::pool::POISON_WORD,
                        "{win:?}: the word at ({x}, {y}) was written"
                    );
                    continue;
                }
                let (lx, ly) = (x - at.0, y - at.1);
                let (sx, sy) = (win.x + lx, win.y + ly);
                if mask[sy * w + sx] != 0.0 {
                    assert_eq!(
                        g,
                        &[bg[0], bg[1], bg[2], 255],
                        "{win:?}: letterbox at ({x}, {y})"
                    );
                } else {
                    let p = &plain[4 * (ly * win.w + lx)..4 * (ly * win.w + lx) + 4];
                    assert_eq!(g, p, "{win:?}: pixel ({x}, {y})");
                }
            }
        }
    }
}
