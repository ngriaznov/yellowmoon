//! M1-09, M1-10, M1-11 (GPU halves), C11 and C13: the per-pixel colour chain kernel against
//! `ym_color::display::ColorChain`, stage by stage, on random inputs, in every storage mode the
//! device supports.
//!
//! Gates (`docs/02` section Tests, `docs/15`): C7 to C11 within 2e-4 in LogC4 units; the displayed
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
                        || name.starts_with("baked-look/srgb/25")
                        || name.starts_with("display-lut/p3")
                        || name.starts_with("logc3-look/35")
                })
                .collect()
        };
        let mut summary: Vec<(String, [f64; 8])> = Vec::new();
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
            let outs: Vec<ColorOutput> = stops.iter().map(|&s| ColorOutput::Stage(s)).collect();
            let gpu = render(dev, &chain, &img, &outs);
            let mut maxes = [0.0f64; 8];
            for (k, stop) in stops.iter().enumerate() {
                let g = planar(&gpu[k]);
                let cpu: Vec<Vec3> = traces
                    .iter()
                    .map(|t| match stop {
                        Stop::C6 => t.c6_linear_awg4,
                        Stop::C7 => t.c7_linear_awg4,
                        Stop::C8 => t.c8_logc4.0,
                        Stop::C9 => t.c9_logc4.0,
                        Stop::C10 => t.c10_logc4.0,
                        Stop::C11 => t.c11_logc4.0,
                        Stop::C12 => t.c12_display_linear,
                        _ => t.c13_encoded,
                    })
                    .collect();
                // Linear stages: relative to the size of the input terms; LogC4: the 2e-4 gate;
                // display-linear: 2e-4 relative plus a small floor; code values: one 10-bit code.
                let cams_ref = &cams;
                let tol = |i: usize| -> f64 {
                    let mag: f64 = cams_ref[i].iter().map(|v| v.abs()).sum();
                    match stop {
                        Stop::C6 | Stop::C7 => 6e-6 * mag + 1e-9,
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
        let mut worst = [0.0f64; 8];
        for (_, m) in &summary {
            for k in 0..8 {
                worst[k] = worst[k].max(m[k]);
            }
        }
        eprintln!(
            "{:?}: {} chains, worst |GPU - CPU| per stage: C6 {:.1e}, C7 {:.1e}, C8 {:.1e}, C9 {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C12 {:.1e} (display-linear), C13 {:.1e} (code values; gate {CODE10:.1e})",
            dev.mode,
            summary.len(),
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            worst[4],
            worst[5],
            worst[6],
            worst[7]
        );
    }
}

/// Controls set one at a time, in combinations and to random values: the fixed chains above set
/// every grade control at once, so a mistake in how one control (or one sign, or one zone) is
/// handled when the others are zero would hide in them. 60 seeded parameter sets: a random subset
/// of the C9 controls, random monotone curves and a random CDL look at a random intensity, for all
/// four outputs.
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
        let mut worst = [0.0f64; 4];
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
            let chain = ColorChain::new(&params).expect("chain");
            let gpu = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Stage(Stop::C9),
                    ColorOutput::Stage(Stop::C10),
                    ColorOutput::Stage(Stop::C11),
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            for i in 0..n {
                let t = chain.trace(pixel(&img, i));
                let cpu = [t.c9_logc4.0, t.c10_logc4.0, t.c11_logc4.0, t.c13_encoded];
                for (k, out) in gpu.iter().enumerate() {
                    let v = planar(out);
                    for ch in 0..3 {
                        worst[k] = worst[k].max((f64::from(v[ch * n + i]) - cpu[k][ch]).abs());
                    }
                }
            }
        }
        eprintln!(
            "{:?}: 60 random parameter sets, worst |GPU - CPU|: C9 {:.1e}, C10 {:.1e}, C11 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C13 {:.1e} (gate {CODE10:.1e})",
            dev.mode, worst[0], worst[1], worst[2], worst[3]
        );
        for (k, gate) in [LOGC4_GATE, LOGC4_GATE, LOGC4_GATE, CODE10]
            .into_iter()
            .enumerate()
        {
            assert!(worst[k] <= gate, "{:?} stage {k}: {}", dev.mode, worst[k]);
        }
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
                let want = chain.apply_dng_pixel(cam).expect("dng");
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
