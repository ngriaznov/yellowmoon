//! M0-07: the spike chain (R1 normalise + R4 gains, bilinear demosaic, fixed matrix, sRGB) on the
//! GPU against the CPU reference, whole frame and tiled.
//!
//! Gates: GPU equals the CPU reference within 1e-3 (the mosaic and the demosaiced planes are held as
//! f16, which accounts for nearly all of it; with f32 storage the difference is ~1e-6); tiled
//! equals whole-frame within 1e-4 across tile seams; peak GPU memory for 50 MP stays under
//! 1.5 GiB (`gpu_spike_50mp`, `#[ignore]`: run with `-- --ignored`).
//!
//! The 1e-3 gate against the plain f32 reference is wide enough to hide a real error of a few
//! 1e-4 on the f16 devices, so `spike_matches_an_f16_aware_reference_to_rounding_noise` also checks
//! every storage mode against a reference that rounds to f16 at the same two places the GPU does:
//! there the two agree to about 1e-6.

mod common;

use common::{Rng, gpus};
use ym_gpu::cpu::chain::RawFrame;
use ym_gpu::cpu::demosaic::bilinear;
use ym_gpu::cpu::gains::apply_gains;
use ym_gpu::cpu::normalize::{Levels, normalize};
use ym_gpu::cpu::tiling::{Rect, Stage};
use ym_gpu::cpu::{Cfa, demosaic::Method};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::spike::{OutputFormat, SpikeParams, TileData, cpu_reference, run_spike};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// Spike gate: GPU against CPU.
const GATE: f32 = 1e-3;
/// Tiled against whole-frame on the same device.
const SEAM_GATE: f32 = 1e-4;

fn params(cfa: Cfa) -> SpikeParams {
    SpikeParams {
        cfa,
        levels: Levels {
            black: [512.0, 514.0, 513.0, 511.0],
            white: [16000.0, 16010.0, 15990.0, 16005.0],
        },
        gains: [2.0, 1.0, 1.6],
        // Balanced camera RGB to linear sRGB-ish; rows sum to 1.
        matrix: [[1.6, -0.4, -0.2], [-0.2, 1.4, -0.2], [0.05, -0.35, 1.3]],
    }
}

/// A smooth scene with texture, noise, a clipped highlight and a below-black corner.
fn raw_scene(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f64 / w as f64;
            let fy = y as f64 / h as f64;
            let base =
                0.45 + 0.35 * (fx * 7.0).sin() * (fy * 5.0).cos() + 0.3 * (fx - 0.5) * (fy - 0.4);
            let clip = if fx > 0.8 && fy < 0.2 { 1.6 } else { 1.0 };
            let v = (base * clip + 0.02 * (rng.unit() - 0.5)).clamp(-0.02, 1.4);
            out.push((512.0 + v * 15488.0).max(0.0) as u16);
        }
    }
    out
}

/// Full-range uncorrelated noise on every photosite: the worst case for f16 storage (every
/// neighbour differs, values reach about 2 after the gains).
fn white_noise_scene(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    (0..w * h)
        .map(|_| (512.0 + rng.range(-0.02, 1.4) * 15488.0).max(0.0) as u16)
        .collect()
}

/// The spike chain on the CPU with the f16 rounding the GPU applies to the balanced mosaic and to
/// the demosaiced planes (`f16 = false`: plain f32, as the F32 storage mode), as planar [0, 1].
fn f16_aware_reference(data: &[u16], w: usize, h: usize, p: &SpikeParams, f16: bool) -> Vec<f32> {
    let round = |v: &mut f32| {
        if f16 {
            *v = common::f16_round(*v);
        }
    };
    let (mut m, _) = normalize(data, w, h, p.cfa, &p.levels, None).expect("normalize");
    apply_gains(&mut m, p.gains);
    m.data.iter_mut().for_each(round);
    let mut rgb = bilinear(&m);
    rgb.data.iter_mut().for_each(round);
    let n = w * h;
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        let c = [
            f64::from(rgb.data[i]),
            f64::from(rgb.data[n + i]),
            f64::from(rgb.data[2 * n + i]),
        ];
        let lin = ym_color::consts::linalg::mul_vec(&p.matrix, c);
        for (ch, v) in lin.iter().enumerate() {
            out[ch * n + i] = ym_color::encode::srgb_encode(*v).clamp(0.0, 1.0) as f32;
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn spike_matches_the_cpu_reference_tiled_in_every_storage_mode_and_phase() {
    for dev in gpus() {
        for (cfa, w, h, tile) in [
            (Cfa::RGGB, 203usize, 157usize, 64usize),
            (Cfa::GRBG, 128, 96, 64),
            (Cfa::GBRG, 99, 70, 32),
            (Cfa::BGGR, 130, 66, 128),
        ] {
            let data = raw_scene(w, h, 7 + w as u64);
            let raw = RawFrame {
                width: w,
                height: h,
                cfa,
                data: &data,
                qstep: None,
            };
            let p = params(cfa);
            let stages = [
                Stage::Normalize,
                Stage::Gains,
                Stage::Demosaic(Method::Bilinear),
            ];
            let plan = TilePlan::new(w, h, tile, &stages);
            let mut worst = 0.0f32;
            let mut tiles = 0;
            let p2 = p.clone();
            let data2 = data.clone();
            let stats = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa,
                        data: &data2,
                        qstep: None,
                    };
                    let mut outs = Vec::new();
                    let stats = run_spike(
                        ctx,
                        &raw,
                        &p2,
                        &plan,
                        OutputFormat::F32Planar,
                        2,
                        token,
                        &mut |t| {
                            outs.push(t);
                            Ok(())
                        },
                    )?;
                    Ok((stats, outs))
                })
                .expect("spike run");
            let (_stats, outs) = stats;
            for out in &outs {
                let core = out.tile.core;
                let cpu = cpu_reference(&raw, &p, core).expect("cpu reference");
                let TileData::F32Planar(gpu) = &out.data else {
                    panic!("wrong format")
                };
                let d = max_abs(gpu, &cpu.data);
                worst = worst.max(d);
                tiles += 1;
            }
            let gate = if dev.mode == StorageMode::F32 {
                2e-5
            } else {
                GATE
            };
            eprintln!(
                "{:?} {:?} {w}x{h} tile {tile}: {tiles} tiles, worst |GPU - CPU| = {worst:.3e} (gate {gate:.0e})",
                dev.mode,
                cfa.name()
            );
            assert!(
                worst <= gate,
                "{:?} {} {w}x{h}: {worst}",
                dev.mode,
                cfa.name()
            );
        }
    }
}

#[test]
fn spike_matches_an_f16_aware_reference_to_rounding_noise() {
    // Every sample within 2e-5 of the reference (measured: 1.4e-6 on lavapipe in every mode). A GPU
    // may round an f32 value that sits on an f16 rounding boundary the other way, which moves one
    // sample by an f16 step; so up to 0.1 % of the samples may exceed the tight bound, none the
    // 1e-3 gate.
    const TIGHT: f32 = 2e-5;
    for dev in gpus() {
        let f16 = dev.mode.is_f16();
        for (name, smooth) in [("scene", true), ("white noise", false)] {
            for (cfa, w, h, tile) in [
                (Cfa::RGGB, 203usize, 157usize, 64usize),
                (Cfa::GBRG, 99, 70, 32),
                (Cfa::BGGR, 130, 66, 50),
            ] {
                let data = if smooth {
                    raw_scene(w, h, 7 + w as u64)
                } else {
                    white_noise_scene(w, h, 3 + h as u64)
                };
                let p = params(cfa);
                let want = f16_aware_reference(&data, w, h, &p, f16);
                let stages = [
                    Stage::Normalize,
                    Stage::Gains,
                    Stage::Demosaic(Method::Bilinear),
                ];
                let plan = TilePlan::new(w, h, tile, &stages);
                let (data2, p2) = (data.clone(), p.clone());
                let outs = dev
                    .gpu
                    .run(JobSpec::new(Priority::Export), move |ctx, token| {
                        let raw = RawFrame {
                            width: w,
                            height: h,
                            cfa,
                            data: &data2,
                            qstep: None,
                        };
                        let mut v = Vec::new();
                        run_spike(
                            ctx,
                            &raw,
                            &p2,
                            &plan,
                            OutputFormat::F32Planar,
                            2,
                            token,
                            &mut |t| {
                                v.push(t);
                                Ok(())
                            },
                        )?;
                        Ok(v)
                    })
                    .expect("spike run");
                let (mut worst, mut over, mut total) = (0.0f32, 0usize, 0usize);
                let mut worst_plain = 0.0f32;
                for o in &outs {
                    let TileData::F32Planar(g) = &o.data else {
                        panic!("wrong format")
                    };
                    let c = o.tile.core;
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa,
                        data: &data,
                        qstep: None,
                    };
                    let plain = cpu_reference(&raw, &p, c).expect("cpu reference");
                    worst_plain = worst_plain.max(max_abs(g, &plain.data));
                    for ch in 0..3 {
                        for y in 0..c.h {
                            for x in 0..c.w {
                                let a = g[ch * c.w * c.h + y * c.w + x];
                                let b = want[ch * w * h + (c.y + y) * w + c.x + x];
                                let d = (a - b).abs();
                                worst = worst.max(d);
                                over += usize::from(d > TIGHT);
                                total += 1;
                            }
                        }
                    }
                }
                eprintln!(
                    "{:?} {name} {} {w}x{h}: worst {worst:.2e} against the f16-aware reference, {over}/{total} above {TIGHT:.0e}; against the plain f32 reference {worst_plain:.2e}",
                    dev.mode,
                    cfa.name()
                );
                assert!(worst <= GATE, "{:?} {name}: {worst}", dev.mode);
                // The storage limit itself: the 1e-3 gate holds on the scene, and full-range noise
                // (values up to 1.4 x white, then the gains and a matrix with negative terms, so
                // a dark output is a difference of large f16-rounded terms) stays within 1e-2.
                let plain_bound = if smooth { GATE } else { 1e-2 };
                assert!(
                    worst_plain <= plain_bound,
                    "{:?} {name} {}: {worst_plain} against the plain f32 reference",
                    dev.mode,
                    cfa.name()
                );
                assert!(
                    over * 1000 <= total,
                    "{:?} {name} {}: {over} of {total} samples differ from the f16-aware reference by more than {TIGHT}",
                    dev.mode,
                    cfa.name()
                );
            }
        }
    }
}

#[test]
fn tiled_equals_whole_frame_across_seams() {
    for dev in gpus() {
        let (w, h) = (151usize, 113usize);
        let data = raw_scene(w, h, 99);
        let p = params(Cfa::RGGB);
        let stages = [
            Stage::Normalize,
            Stage::Gains,
            Stage::Demosaic(Method::Bilinear),
        ];
        let run = |tile: usize| {
            let plan = TilePlan::new(w, h, tile, &stages);
            let data = data.clone();
            let p = p.clone();
            dev.gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa: Cfa::RGGB,
                        data: &data,
                        qstep: None,
                    };
                    let mut full = vec![0.0f32; 3 * w * h];
                    run_spike(
                        ctx,
                        &raw,
                        &p,
                        &plan,
                        OutputFormat::F32Planar,
                        2,
                        token,
                        &mut |t| {
                            let TileData::F32Planar(d) = &t.data else {
                                unreachable!()
                            };
                            let c = t.tile.core;
                            for ch in 0..3 {
                                for y in 0..c.h {
                                    let dst = ch * w * h + (c.y + y) * w + c.x;
                                    full[dst..dst + c.w]
                                        .copy_from_slice(&d[ch * c.w * c.h + y * c.w..][..c.w]);
                                }
                            }
                            Ok(())
                        },
                    )?;
                    Ok(full)
                })
                .expect("run")
        };
        let whole = run(256);
        for tile in [32, 50, 64] {
            let tiled = run(tile);
            let d = max_abs(&whole, &tiled);
            eprintln!("{:?}: tiles of {tile} vs whole frame: {d:.3e}", dev.mode);
            assert!(d <= SEAM_GATE, "{:?} tile {tile}: {d}", dev.mode);
        }
    }
}

#[test]
fn output_formats_agree_with_the_float_output() {
    for dev in gpus() {
        let (w, h) = (97usize, 61usize);
        let data = raw_scene(w, h, 5);
        let p = params(Cfa::RGGB);
        let plan = TilePlan::with_halo(w, h, 64, 1);
        let data2 = data.clone();
        let p2 = p.clone();
        let (f32s, u8s, u16s) = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, token| {
                let raw = RawFrame {
                    width: w,
                    height: h,
                    cfa: Cfa::RGGB,
                    data: &data2,
                    qstep: None,
                };
                let collect = |fmt| -> Result<Vec<_>, ym_gpu::device::GpuError> {
                    let mut v = Vec::new();
                    run_spike(ctx, &raw, &p2, &plan, fmt, 1, token, &mut |t| {
                        v.push(t);
                        Ok(())
                    })?;
                    Ok(v)
                };
                Ok((
                    collect(OutputFormat::F32Planar)?,
                    collect(OutputFormat::Rgba8)?,
                    collect(OutputFormat::U16Planar)?,
                ))
            })
            .expect("run");
        for ((a, b), c) in f32s.iter().zip(&u8s).zip(&u16s) {
            let (TileData::F32Planar(f), TileData::Rgba8(q8), TileData::U16Planar(q16)) =
                (&a.data, &b.data, &c.data)
            else {
                panic!("formats")
            };
            let n = a.width * a.height;
            for i in 0..n {
                for ch in 0..3 {
                    let v = f[ch * n + i];
                    let e8 = (v * 255.0).round() as i32;
                    assert!(
                        (i32::from(q8[4 * i + ch]) - e8).abs() <= 1,
                        "u8 at {i} ch {ch}"
                    );
                    let e16 = (v * 65535.0).round() as i32;
                    assert!(
                        (i32::from(q16[ch * n + i]) - e16).abs() <= 1,
                        "u16 at {i} ch {ch}"
                    );
                }
                assert_eq!(q8[4 * i + 3], 255);
            }
        }
    }
}

#[test]
fn a_cancelled_run_stops_between_tiles() {
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (128usize, 128usize);
    let data = raw_scene(w, h, 1);
    let p = params(Cfa::RGGB);
    let plan = TilePlan::with_halo(w, h, 32, 1);
    let r = dev.gpu.run(JobSpec::new(Priority::Export), move |ctx, _| {
        let raw = RawFrame {
            width: w,
            height: h,
            cfa: Cfa::RGGB,
            data: &data,
            qstep: None,
        };
        let token = CancelToken::new();
        let mut seen = 0;
        let r = run_spike(
            ctx,
            &raw,
            &p,
            &plan,
            OutputFormat::Rgba8,
            1,
            &token,
            &mut |_| {
                seen += 1;
                if seen == 3 {
                    token.cancel();
                }
                Ok(())
            },
        );
        Ok((r.is_err(), seen))
    });
    let (was_err, seen) = r.expect("job");
    assert!(was_err);
    assert!((3..=4).contains(&seen), "{seen} tiles before the stop");
}

#[test]
fn a_mismatched_plan_is_refused() {
    let Some(dev) = gpus().first() else { return };
    let data = raw_scene(32, 32, 1);
    let p = params(Cfa::RGGB);
    let plan = TilePlan::with_halo(64, 64, 32, 1);
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let raw = RawFrame {
                width: 32,
                height: 32,
                cfa: Cfa::RGGB,
                data: &data,
                qstep: None,
            };
            run_spike(
                ctx,
                &raw,
                &p,
                &plan,
                OutputFormat::Rgba8,
                1,
                token,
                &mut |_| Ok(()),
            )
        });
    assert!(
        matches!(r, Err(ym_gpu::device::GpuError::InvalidInput(_))),
        "{r:?}"
    );
}

/// Runs the spike chain with a plan and a pattern chosen by the caller and returns the error.
fn spike_error(
    plan: TilePlan,
    frame_cfa: Cfa,
    params_cfa: Cfa,
) -> Option<ym_gpu::device::GpuError> {
    let dev = gpus().first()?;
    let (w, h) = (64usize, 64usize);
    let data = raw_scene(w, h, 1);
    let p = params(params_cfa);
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let raw = RawFrame {
                width: w,
                height: h,
                cfa: frame_cfa,
                data: &data,
                qstep: None,
            };
            run_spike(
                ctx,
                &raw,
                &p,
                &plan,
                OutputFormat::Rgba8,
                1,
                token,
                &mut |_| Ok(()),
            )
        })
        .err()
}

#[test]
fn a_plan_without_the_demosaic_halo_or_a_mismatched_pattern_is_refused() {
    if gpus().is_empty() {
        return;
    }
    // Tiles without any halo would show seams at every tile border instead of failing.
    let err = spike_error(TilePlan::with_halo(64, 64, 32, 0), Cfa::RGGB, Cfa::RGGB)
        .expect("a plan with no halo is refused");
    assert!(
        matches!(&err, ym_gpu::device::GpuError::InvalidInput(m) if m.contains("halo")),
        "{err:?}"
    );
    // The GPU reads the pattern of the parameters, the CPU reference that of the frame: they must agree.
    let err = spike_error(TilePlan::with_halo(64, 64, 32, 1), Cfa::RGGB, Cfa::GRBG)
        .expect("a frame and parameters with different patterns are refused");
    assert!(
        matches!(&err, ym_gpu::device::GpuError::InvalidInput(m) if m.contains("pattern")),
        "{err:?}"
    );
    // The right plan and pattern run.
    assert!(spike_error(TilePlan::with_halo(64, 64, 32, 1), Cfa::RGGB, Cfa::RGGB).is_none());
}

#[test]
fn raw_scene_is_deterministic() {
    assert_eq!(raw_scene(16, 16, 3), raw_scene(16, 16, 3));
    let _ = Rect::whole(1, 1);
}

/// Cheap deterministic 50 MP scene: smooth gradients, texture from a hash, a clipped corner.
fn big_raw_scene(w: usize, h: usize) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    for (y, row) in out.chunks_mut(w).enumerate() {
        let fy = y as f32 / h as f32;
        for (x, v) in row.iter_mut().enumerate() {
            let fx = x as f32 / w as f32;
            let hash = ((x as u32).wrapping_mul(0x9E37_79B1)
                ^ (y as u32).wrapping_mul(0x85EB_CA77))
            .wrapping_mul(0xC2B2_AE3D);
            let noise = (hash >> 20) as f32 / 4096.0 - 0.5;
            let base = 0.5 + 0.4 * (fx * 9.0).sin() * (fy * 6.0).cos() + 0.25 * (fx - 0.5);
            let clip = if fx > 0.85 && fy < 0.15 { 1.7 } else { 1.0 };
            let s = (base * clip + 0.03 * noise).clamp(-0.02, 1.5);
            *v = (512.0 + s * 15488.0).max(0.0) as u16;
        }
    }
    out
}

fn resident_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// The M0-07 gate: a 50 MP frame through the spike chain in 2048 px tiles, GPU equal to the CPU
/// reference within 1e-3, peak GPU memory under 1.5 GiB. Slow on lavapipe (software Vulkan): speeds
/// printed here mean nothing for Metal. Run with
/// `WGPU_BACKEND=vulkan YM_REQUIRE_GPU=1 cargo test -p ym-gpu --test gpu_spike --release -- --ignored --nocapture gpu_spike_50mp`.
#[test]
#[ignore = "50 MP frame on a software rasteriser: takes minutes; run with --ignored"]
fn gpu_spike_50mp() {
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (8192usize, 6144usize);
    let t0 = std::time::Instant::now();
    let data = big_raw_scene(w, h);
    eprintln!(
        "scene {w}x{h} = {:.1} MP generated in {:.1?}",
        (w * h) as f64 / 1e6,
        t0.elapsed()
    );
    let p = params(Cfa::RGGB);
    let stages = [
        Stage::Normalize,
        Stage::Gains,
        Stage::Demosaic(Method::Bilinear),
    ];
    let plan = TilePlan::new(w, h, 2048, &stages);
    let tiles = plan.tiles().len();
    let plan_rows = plan.rows();
    let result = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let raw = RawFrame {
                width: w,
                height: h,
                cfa: Cfa::RGGB,
                data: &data,
                qstep: None,
            };
            let t = std::time::Instant::now();
            let mut worst = 0.0f32;
            let mut rows_done = 0;
            let stats = run_spike(
                ctx,
                &raw,
                &p,
                &plan,
                OutputFormat::F32Planar,
                2,
                token,
                &mut |out| {
                    let cpu = cpu_reference(&raw, &p, out.tile.core)
                        .map_err(|e| ym_gpu::device::GpuError::Cpu(e.to_string()))?;
                    let TileData::F32Planar(g) = &out.data else {
                        unreachable!()
                    };
                    let d = max_abs(g, &cpu.data);
                    worst = worst.max(d);
                    if out.tile.col + 1 == plan.cols() {
                        rows_done += 1;
                    }
                    Ok(())
                },
            )?;
            Ok((
                stats,
                worst,
                rows_done,
                t.elapsed(),
                ctx.pool().stats(),
                ctx.info().clone(),
            ))
        })
        .expect("50 MP run");
    let (stats, worst, rows_done, elapsed, pool, info) = result;
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    eprintln!(
        "adapter: {} ({}), storage {}",
        info.name,
        info.backend,
        info.storage_mode.name()
    );
    eprintln!(
        "{tiles} tiles of 2048 px, {rows_done} tile rows streamed, {elapsed:.1?} (GPU + CPU reference; lavapipe speed is meaningless)"
    );
    eprintln!("worst |GPU - CPU| over the frame: {worst:.3e} (gate 1e-3)");
    eprintln!(
        "peak GPU memory (pool ledger): allocated {:.3} GiB, in use {:.3} GiB; buffers created {}, reused {}",
        gib(stats.peak_allocated_bytes),
        gib(stats.peak_in_use_bytes),
        stats.buffers_created,
        stats.reuses
    );
    if let Some(a) = stats.allocator_total_bytes {
        eprintln!("driver allocator report at the end: {:.3} GiB", gib(a));
    }
    eprintln!("pool after the run: {pool:?}");
    if let Some(kib) = resident_kib() {
        eprintln!(
            "process peak resident set: {:.2} GiB (includes the 100 MB raw frame and the CPU reference)",
            kib as f64 / 1048576.0
        );
    }
    assert!(
        worst <= GATE,
        "GPU differs from the CPU reference by {worst}"
    );
    assert!(
        stats.peak_allocated_bytes < (3 << 29),
        "peak GPU memory {:.3} GiB",
        gib(stats.peak_allocated_bytes)
    );
    assert_eq!(rows_done, plan_rows);
}
