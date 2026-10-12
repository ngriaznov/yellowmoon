//! The raw front end and the colour chain together on the GPU (raw `u16` in, display pixels out, the
//! demosaiced planes never leaving the device), the row streaming for export, and the tile size
//! recommendation.

mod common;

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::chains::{self, fused_matrix};
use common::{Rng, f16_round, gpus};
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_gpu::cpu::Cfa;
use ym_gpu::cpu::chain::RawFrame;
use ym_gpu::cpu::demosaic::{Method, bilinear};
use ym_gpu::cpu::gains::apply_gains;
use ym_gpu::cpu::normalize::{Levels, normalize};
use ym_gpu::cpu::tiling::Stage;
use ym_gpu::device::{Gpu, GpuConfig, StorageMode};
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::kernels::raw::{
    RawParams, RowBand, RowStreamer, recommended_tile_px, run_raw_to_display,
};
use ym_gpu::kernels::spike;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

const CODE10: f64 = 1.0 / 1023.0;
/// A twentieth of a 10-bit code: how closely the f16-aware reference must be matched.
const TIGHT: f64 = CODE10 / 20.0;

fn raw_params(cfa: Cfa) -> RawParams {
    RawParams {
        cfa,
        levels: Levels {
            black: [512.0, 514.0, 513.0, 511.0],
            white: [16000.0, 16010.0, 15990.0, 16005.0],
        },
        gains: [1.9, 1.0, 1.5],
    }
}

fn scene(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f64 / w as f64;
            let fy = y as f64 / h as f64;
            let base = 0.4 + 0.3 * (fx * 6.0).sin() * (fy * 4.0).cos() + 0.35 * fx * fy;
            let clip = if fx > 0.8 && fy < 0.25 { 1.5 } else { 1.0 };
            let v = (base * clip + 0.03 * (rng.unit() - 0.5)).clamp(-0.01, 1.3);
            out.push((512.0 + v * 15488.0).max(0.0) as u16);
        }
    }
    out
}

fn stages() -> [Stage<'static>; 3] {
    [
        Stage::Normalize,
        Stage::Gains,
        Stage::Demosaic(Method::Bilinear),
    ]
}

/// CPU reference: normalise, gains, bilinear demosaic, colour chain; with the f16 rounding the GPU
/// applies to the mosaic and to the demosaiced planes when `f16` is set.
fn cpu_display(
    data: &[u16],
    w: usize,
    h: usize,
    p: &RawParams,
    chain: &ColorChain,
    f16: bool,
) -> Vec<[f64; 3]> {
    let (mut m, _) = normalize(data, w, h, p.cfa, &p.levels, None).expect("normalize");
    apply_gains(&mut m, p.gains);
    if f16 {
        m.data.iter_mut().for_each(|v| *v = f16_round(*v));
    }
    let mut rgb = bilinear(&m);
    if f16 {
        rgb.data.iter_mut().for_each(|v| *v = f16_round(*v));
    }
    let n = w * h;
    (0..n)
        .map(|i| {
            chain.apply_pixel([
                f64::from(rgb.data[i]),
                f64::from(rgb.data[n + i]),
                f64::from(rgb.data[2 * n + i]),
            ])
        })
        .collect()
}

#[test]
fn raw_to_display_matches_the_cpu_chain_tiled_in_every_storage_mode() {
    for dev in gpus() {
        for (cfa, w, h, tile) in [
            (Cfa::RGGB, 203usize, 157usize, 64usize),
            (Cfa::GBRG, 120, 77, 50),
        ] {
            let data = scene(w, h, 11);
            let p = raw_params(cfa);
            let mut cp = ChainParams::new(fused_matrix(), chains::tone());
            cp.grade = chains::grade_params();
            cp.curves = chains::curves();
            cp.output = OutputTarget::Viewport;
            let chain = ColorChain::new(&cp).expect("chain");
            let want = cpu_display(&data, w, h, &p, &chain, dev.mode != StorageMode::F32);
            let plan = TilePlan::new(w, h, tile, &stages());
            let (chain2, data2, p2) = (chain.clone(), data.clone(), p.clone());
            let (got, stats) = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa,
                        data: &data2,
                        qstep: None,
                    };
                    let prepared = PreparedChain::new(ctx, &chain2)?;
                    let mut full = vec![0.0f32; 3 * w * h];
                    let stats = run_raw_to_display(
                        ctx,
                        &raw,
                        &p2,
                        &prepared,
                        &plan,
                        ColorOutput::Stage(Stop::C13),
                        2,
                        token,
                        &mut |t, img| {
                            let PixelData::F32Planar(d) = &img.data else {
                                unreachable!()
                            };
                            let c = t.core;
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
                    Ok((full, stats))
                })
                .expect("pipeline");
            let mut worst = 0.0f64;
            let mut over_tight = 0usize;
            for (i, px) in want.iter().enumerate() {
                for ch in 0..3 {
                    let d = (f64::from(got[ch * w * h + i]) - px[ch]).abs();
                    worst = worst.max(d);
                    over_tight += usize::from(d > TIGHT);
                }
            }
            eprintln!(
                "{:?} {} {w}x{h} tile {tile}: raw to display, worst {:.4} codes at 10 bits; {} tiles, peak {:.1} MiB",
                dev.mode,
                cfa.name(),
                worst * 1023.0,
                stats.tiles,
                stats.peak_allocated_bytes as f64 / 1048576.0
            );
            assert!(
                worst <= CODE10,
                "{:?} {}: {:.3} codes",
                dev.mode,
                cfa.name(),
                worst * 1023.0
            );
            // The reference rounds to f16 where the GPU does, so the two agree to f32 noise (1.8e-6
            // measured, a twentieth of a 10-bit code is the bound). A sample on an f16 rounding
            // boundary may land one step away on another GPU; a handful is allowed, not a pattern.
            assert!(
                over_tight * 1000 <= 3 * w * h,
                "{:?} {}: {over_tight} of {} samples differ from the f16-aware reference by more than {TIGHT}",
                dev.mode,
                cfa.name(),
                3 * w * h
            );
        }
    }
}

#[test]
fn tile_rows_stream_top_to_bottom_and_equal_the_whole_frame() {
    for dev in gpus() {
        let (w, h) = (150usize, 130usize);
        let data = scene(w, h, 5);
        let p = raw_params(Cfa::RGGB);
        let chain = ColorChain::new(&chains::full(OutputTarget::Srgb)).expect("chain");
        let run = |tile: usize| -> Vec<RowBand> {
            let plan = TilePlan::new(w, h, tile, &stages());
            let (chain, data, p) = (chain.clone(), data.clone(), p.clone());
            dev.gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa: Cfa::RGGB,
                        data: &data,
                        qstep: None,
                    };
                    let prepared = PreparedChain::new(ctx, &chain)?;
                    let mut bands = Vec::new();
                    {
                        let mut sink = |b: RowBand| {
                            bands.push(b);
                            Ok(())
                        };
                        let mut streamer = RowStreamer::new(&plan, &mut sink);
                        run_raw_to_display(
                            ctx,
                            &raw,
                            &p,
                            &prepared,
                            &plan,
                            ColorOutput::Rgba8 { dither: true },
                            2,
                            token,
                            &mut |t, img| streamer.push(t, img.data),
                        )?;
                        assert_eq!(streamer.rows_pending(), 0);
                    }
                    Ok(bands)
                })
                .expect("stream")
        };
        let whole = run(256);
        assert_eq!(whole.len(), 1);
        let PixelData::Rgba8(whole_px) = &whole[0].data else {
            panic!()
        };
        for tile in [32usize, 64] {
            let bands = run(tile);
            let mut y = 0;
            let mut joined: Vec<u8> = Vec::new();
            for b in &bands {
                assert_eq!(b.y0, y, "rows arrive in order without gaps");
                assert_eq!(b.width, w);
                let PixelData::Rgba8(px) = &b.data else {
                    panic!()
                };
                assert_eq!(px.len(), w * b.height * 4);
                joined.extend_from_slice(px);
                y += b.height;
            }
            assert_eq!(y, h);
            assert_eq!(bands.len(), h.div_ceil(tile));
            assert_eq!(
                &joined, whole_px,
                "{:?}: streamed rows with tiles of {tile} differ from the whole frame",
                dev.mode
            );
        }
    }
}

/// The device thread never pre-empts a running job, so a long export has to be cut into jobs for
/// the priority queue to mean anything: here one job per tile row, at Export priority, with a
/// Viewport job arriving while the second row runs. It must run before the third row, and the
/// pixels must equal those of the same export run as one job.
#[test]
fn an_export_cut_into_row_jobs_lets_a_viewport_job_in_between_and_gives_the_same_pixels() {
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (150usize, 130usize);
    let data = Arc::new(scene(w, h, 5));
    let p = raw_params(Cfa::RGGB);
    let chain = ColorChain::new(&chains::full(OutputTarget::Srgb)).expect("chain");
    let plan = TilePlan::new(w, h, 32, &stages());
    assert_eq!(plan.rows(), 5);
    let output = ColorOutput::Rgba8 { dither: true };

    // The export as one job.
    let one_job = {
        let (data, p, chain, plan) = (Arc::clone(&data), p.clone(), chain.clone(), plan.clone());
        dev.gpu
            .run(JobSpec::new(Priority::Export), move |ctx, token| {
                let raw = RawFrame {
                    width: w,
                    height: h,
                    cfa: Cfa::RGGB,
                    data: &data,
                    qstep: None,
                };
                let prepared = PreparedChain::new(ctx, &chain)?;
                let mut tiles = Vec::new();
                run_raw_to_display(
                    ctx,
                    &raw,
                    &p,
                    &prepared,
                    &plan,
                    output,
                    2,
                    token,
                    &mut |t, img| {
                        tiles.push((t, img.data));
                        Ok(())
                    },
                )?;
                Ok(tiles)
            })
            .expect("one job")
    };

    // The same export as one job per tile row, with a hook in the second row's job.
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let mut hook = Some((started_tx, release_rx));
    let handles: Vec<_> = (0..plan.rows())
        .map(|r| {
            let (data, p, chain, rows_plan) = (
                Arc::clone(&data),
                p.clone(),
                chain.clone(),
                plan.tile_rows(r..r + 1),
            );
            let log = Arc::clone(&log);
            let hook = if r == 1 { hook.take() } else { None };
            dev.gpu.submit(
                JobSpec::new(Priority::Export).owned_by(1),
                move |ctx, token| {
                    log.lock().expect("log").push(format!("row {r}"));
                    if let Some((started, release)) = hook {
                        let _ = started.send(());
                        let _ = release.recv_timeout(Duration::from_secs(30));
                    }
                    let raw = RawFrame {
                        width: w,
                        height: h,
                        cfa: Cfa::RGGB,
                        data: &data,
                        qstep: None,
                    };
                    let prepared = PreparedChain::new(ctx, &chain)?;
                    let mut tiles = Vec::new();
                    run_raw_to_display(
                        ctx,
                        &raw,
                        &p,
                        &prepared,
                        &rows_plan,
                        output,
                        2,
                        token,
                        &mut |t, img| {
                            tiles.push((t, img.data));
                            Ok(())
                        },
                    )?;
                    Ok::<_, ym_gpu::device::GpuError>(tiles)
                },
            )
        })
        .collect();
    started_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the second row started");
    let viewport = {
        let log = Arc::clone(&log);
        dev.gpu
            .submit(JobSpec::new(Priority::Viewport).owned_by(1), move |_, _| {
                log.lock().expect("log").push("viewport".to_string());
            })
    };
    release_tx.send(()).expect("release the second row");
    viewport
        .wait_timeout(Duration::from_secs(60))
        .expect("the viewport job ran")
        .expect("viewport job");
    let mut chunked = Vec::new();
    for h in handles {
        chunked.extend(h.wait().expect("row job").expect("row job result"));
    }
    assert_eq!(
        *log.lock().expect("log"),
        ["row 0", "row 1", "viewport", "row 2", "row 3", "row 4"],
        "the viewport job runs between the rows of the export"
    );

    // Same tiles, same bytes, and the rows stream out in order.
    let join = |tiles: Vec<(ym_gpu::tiles::Tile, PixelData)>| -> Vec<u8> {
        let mut bands = Vec::new();
        {
            let mut sink = |b: RowBand| {
                bands.push(b);
                Ok(())
            };
            let mut streamer = RowStreamer::new(&plan, &mut sink);
            for (t, d) in tiles {
                streamer.push(t, d).expect("push");
            }
            assert_eq!(streamer.rows_pending(), 0);
        }
        let mut y = 0;
        let mut bytes = Vec::new();
        for b in &bands {
            assert_eq!(b.y0, y);
            y += b.height;
            let PixelData::Rgba8(px) = &b.data else {
                panic!("format")
            };
            bytes.extend_from_slice(px);
        }
        assert_eq!(y, h);
        bytes
    };
    assert_eq!(
        join(chunked),
        join(one_job),
        "an export cut into row jobs differs from the same export run as one job"
    );
}

#[test]
fn the_spike_rows_stream_the_same_way() {
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (96usize, 70usize);
    let data = scene(w, h, 2);
    let p = raw_params(Cfa::RGGB);
    let sp = spike::SpikeParams {
        cfa: p.cfa,
        levels: p.levels,
        gains: p.gains,
        matrix: fused_matrix(),
    };
    let plan = TilePlan::new(w, h, 32, &stages());
    let bands = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let raw = RawFrame {
                width: w,
                height: h,
                cfa: Cfa::RGGB,
                data: &data,
                qstep: None,
            };
            let mut bands = Vec::new();
            {
                let mut sink = |b: RowBand| {
                    bands.push(b);
                    Ok(())
                };
                let mut streamer = RowStreamer::new(&plan, &mut sink);
                spike::run_spike(
                    ctx,
                    &raw,
                    &sp,
                    &plan,
                    spike::OutputFormat::U16Planar,
                    2,
                    token,
                    &mut |t| streamer.push(t.tile, t.data),
                )?;
            }
            Ok(bands)
        })
        .expect("spike rows");
    assert_eq!(bands.iter().map(|b| b.height).sum::<usize>(), h);
    assert!(bands.windows(2).all(|w| w[1].y0 == w[0].y0 + w[0].height));
}

#[test]
fn the_tile_size_follows_the_device_budget() {
    let Some(dev) = gpus().first() else { return };
    let per_px = spike::bytes_per_pixel(dev.mode, spike::OutputFormat::F32Planar);
    let big = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            Ok(recommended_tile_px(ctx, per_px, 1))
        })
        .expect("tile size");
    assert_eq!(big, 2048, "a 4 GiB budget takes 2048 px tiles");
    let small = Gpu::start(GpuConfig {
        memory_budget_bytes: Some(160 << 20),
        ..GpuConfig::default()
    })
    .expect("device");
    let tile = small
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            Ok(recommended_tile_px(ctx, per_px, 1))
        })
        .expect("tile size");
    assert_eq!(tile, 1024, "a 160 MiB budget falls back to 1024 px tiles");
}
