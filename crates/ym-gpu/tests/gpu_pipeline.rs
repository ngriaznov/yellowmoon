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

// ------------------------------------------------------------------------------------------------
// G-04: the view tail on the device: sampler, colour, output sharpening and C13 as device images
// ------------------------------------------------------------------------------------------------

use common::chains::camera_image;
use ym_color::consts::Vec3;
use ym_color::encode::{self as c13, OutputSpace};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::OutputSharpen;
use ym_gpu::cpu::sampler::SamplerGrid;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::encode::{self, C13, Encoding, band_bytes, encode_host, row_words};
use ym_gpu::kernels::output_sharpen::PreparedSharpen;
use ym_gpu::kernels::sampler::{encode_sample, sample};
use ym_gpu::stream::PassList;

/// A display-linear image: values from a little below 0 to above 1 (the encode clamps), a few exact
/// zeros and ones.
fn linear_image(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for v in &mut img.data {
        *v = rng.range(-0.1, 1.4) as f32;
    }
    img.set_pixel(0, 0, [0.0, 0.0, 0.0]);
    img.set_pixel(1, 0, [1.0, 1.0, 1.0]);
    img.set_pixel(2, 0, [0.0031308, 0.04, 0.18]);
    img
}

fn on_gpu<R: Send + 'static>(
    body: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, ym_gpu::device::GpuError>
    + Send
    + 'static,
) -> Option<R> {
    let dev = gpus().first()?;
    Some(
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| body(ctx))
            .expect("GPU job"),
    )
}

#[test]
fn the_encode_kernel_matches_the_cpu_c13() {
    // `ym_color::encode::encode_8bit_dithered` and `quantize` are the CPU's C13: the 8-bit codes must
    // be within one code and 99 % exact (f32 against f64 moves a value across a dither threshold now
    // and then), the 16-bit codes within 8 (docs/gpu-plan.md G-08 states the same).
    let (w, h) = (131usize, 47usize);
    let img = linear_image(w, h, 0xC13);
    for (space, origin) in [
        (OutputSpace::Srgb, (0usize, 0usize)),
        (OutputSpace::DisplayP3, (37, 1010)),
        (OutputSpace::Rec2020, (4093, 8191)),
    ] {
        let i2 = img.clone();
        let (rgb8, rgba8, rgb16, rgb10) = on_gpu(move |ctx| {
            let whole = Rect::whole(w, h);
            let c = C13 {
                space,
                dither: true,
                origin,
            };
            let bg = [0, 0, 0];
            Ok((
                encode_host(ctx, &i2, None, whole, Encoding::Rgb8 { background: bg }, c)?,
                encode_host(ctx, &i2, None, whole, Encoding::Rgba8 { background: bg }, c)?,
                encode_host(ctx, &i2, None, whole, Encoding::Rgb16 { bits: 16 }, c)?,
                encode_host(ctx, &i2, None, whole, Encoding::Rgb16 { bits: 10 }, c)?,
            ))
        })
        .expect("device");
        let n = w * h;
        let (mut off8, mut worst8, mut worst16, mut worst10) = (0usize, 0i32, 0i32, 0i32);
        for i in 0..n {
            let (x, y) = (i % w, i / w);
            let lin: Vec3 = [
                f64::from(img.data[i]),
                f64::from(img.data[n + i]),
                f64::from(img.data[2 * n + i]),
            ];
            let cpu8 = c13::encode_8bit_dithered(lin, space, origin.0 + x, origin.1 + y);
            let enc = space.encode_clamped(lin);
            for c in 0..3 {
                let g = i32::from(rgb8.bytes[3 * i + c]);
                let d = (g - i32::from(cpu8[c])).abs();
                worst8 = worst8.max(d);
                off8 += usize::from(d != 0);
                assert_eq!(
                    rgba8.bytes[4 * i + c],
                    rgb8.bytes[3 * i + c],
                    "RGBA8 and RGB8 differ at pixel {i}"
                );
                let g16 = i32::from(u16::from_le_bytes([
                    rgb16.bytes[6 * i + 2 * c],
                    rgb16.bytes[6 * i + 2 * c + 1],
                ]));
                worst16 = worst16.max((g16 - i32::from(c13::quantize(enc[c], 16))).abs());
                let g10 = i32::from(u16::from_le_bytes([
                    rgb10.bytes[6 * i + 2 * c],
                    rgb10.bytes[6 * i + 2 * c + 1],
                ]));
                worst10 = worst10.max((g10 - i32::from(c13::quantize(enc[c], 10))).abs());
            }
            assert_eq!(rgba8.bytes[4 * i + 3], 255);
        }
        eprintln!(
            "{space:?}: 8-bit worst {worst8} code, {off8} of {} differ; 16-bit worst {worst16}; 10-bit worst {worst10}",
            3 * n
        );
        assert!(worst8 <= 1, "{space:?}: 8-bit codes off by {worst8}");
        assert!(
            off8 * 100 <= 3 * n,
            "{space:?}: {off8} of {} codes differ",
            3 * n
        );
        assert!(worst16 <= 8, "{space:?}: 16-bit codes off by {worst16}");
        assert!(worst10 <= 1, "{space:?}: 10-bit codes off by {worst10}");
    }
}

#[test]
fn bands_of_rows_give_the_bytes_of_the_whole_picture() {
    // The dither is at absolute positions: a picture encoded in bands (each band's origin is its first
    // row) equals the picture encoded at once, for every layout.
    let (w, h) = (53usize, 61usize);
    let img = linear_image(w, h, 0xBA1D);
    for encoding in [
        Encoding::Rgba8 {
            background: [1, 2, 3],
        },
        Encoding::Rgb8 {
            background: [1, 2, 3],
        },
        Encoding::Rgb16 { bits: 16 },
    ] {
        let i2 = img.clone();
        let (whole, bands) = on_gpu(move |ctx| {
            let c = |y: usize| C13 {
                space: OutputSpace::DisplayP3,
                dither: true,
                origin: (100, 7 + y),
            };
            let whole = encode_host(ctx, &i2, None, Rect::whole(w, h), encoding, c(0))?;
            let mut bytes = Vec::new();
            for (y0, rows) in [(0usize, 17usize), (17, 20), (37, 24)] {
                let band = encode_host(ctx, &i2, None, Rect::new(0, y0, w, rows), encoding, c(y0))?;
                bytes.extend_from_slice(&band.bytes);
            }
            Ok((whole.bytes, bytes))
        })
        .expect("device");
        assert_eq!(
            whole, bands,
            "{encoding:?}: bands differ from the whole picture"
        );
    }
}

#[test]
fn the_encode_kernel_agrees_with_the_colour_kernels_own_c13() {
    // Colour to C12 on the device, then the encode kernel, against the colour kernel's own RGBA8.
    let (w, h) = (73usize, 39usize);
    let img = camera_image(w, h, 0xE0C);
    for target in [
        OutputTarget::Viewport,
        OutputTarget::Srgb,
        OutputTarget::Rec2020,
    ] {
        for dither in [true, false] {
            let chain = ColorChain::new(&chains::full(target)).expect("chain");
            let space = chain.space();
            let i2 = img.clone();
            let (own, encoded) = on_gpu(move |ctx| {
                let prepared = PreparedChain::new(ctx, &chain)?;
                let own = prepared.render(ctx, &i2, (5, 9), ColorOutput::Rgba8 { dither })?;
                let src = GpuImage::upload_rgb(ctx, "c12", &i2)?;
                let c12 = GpuImage::alloc(ctx, "c12 out", w, h, 3, i2.place)?;
                let bytes = band_bytes(Encoding::Rgba8 { background: [0; 3] }, w, h);
                let band = ctx.pool().acquire(
                    "band",
                    bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )?;
                let mut passes = PassList::new(ctx, "c12 then c13");
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    Rect::whole(w, h),
                    Stop::C12,
                    &c12,
                    (0, 0),
                )?;
                encode::encode_c13(
                    &mut passes,
                    &c12,
                    Rect::whole(w, h),
                    Encoding::Rgba8 { background: [0; 3] },
                    C13 {
                        space,
                        dither,
                        origin: (5, 9),
                    },
                    &band,
                    0,
                    0,
                )?;
                let id = passes.read(&band, 0, bytes)?;
                let done = passes.submit()?.finish()?;
                Ok((own, done.read(id).to_vec()))
            })
            .expect("device");
            let PixelData::Rgba8(own) = own.data else {
                panic!("not RGBA8")
            };
            assert_eq!(own, encoded, "{target:?} dither {dither}");
        }
    }
}

#[test]
fn the_letterbox_is_painted_by_the_encode_kernel() {
    let (w, h) = (30usize, 20usize);
    let img = linear_image(w, h, 0x1E7);
    let mask: Vec<u8> = (0..w * h).map(|i| u8::from(i % 7 == 0 || i < 40)).collect();
    let (m2, i2) = (mask.clone(), img.clone());
    let (rgba, rgb8, rgb16) = on_gpu(move |ctx| {
        let whole = Rect::whole(w, h);
        let c = C13 {
            space: OutputSpace::Srgb,
            dither: true,
            origin: (0, 0),
        };
        let bg = [11, 22, 33];
        Ok((
            encode_host(
                ctx,
                &i2,
                Some(&m2),
                whole,
                Encoding::Rgba8 { background: bg },
                c,
            )?,
            encode_host(
                ctx,
                &i2,
                Some(&m2),
                whole,
                Encoding::Rgb8 { background: bg },
                c,
            )?,
            encode_host(ctx, &i2, Some(&m2), whole, Encoding::Rgb16 { bits: 16 }, c)?,
        ))
    })
    .expect("device");
    let mut painted = 0;
    for i in 0..w * h {
        if mask[i] != 0 {
            painted += 1;
            assert_eq!(
                &rgba.bytes[4 * i..4 * i + 4],
                &[11, 22, 33, 255],
                "RGBA8 pixel {i}"
            );
            assert_eq!(
                &rgb8.bytes[3 * i..3 * i + 3],
                &[11, 22, 33],
                "RGB8 pixel {i}"
            );
            assert_eq!(&rgb16.bytes[6 * i..6 * i + 6], &[0; 6], "RGB16 pixel {i}");
        } else {
            assert_ne!(
                &rgba.bytes[4 * i..4 * i + 3],
                &[11, 22, 33],
                "pixel {i} is inside"
            );
        }
    }
    assert!(painted > 80);
}

#[test]
fn the_encode_binding_and_row_sizes_are_what_the_writers_expect() {
    common::limits::assert_max_binding_bytes(
        "encode",
        Rect::new(0, 0, 2000, 1250),
        encode::max_binding_bytes,
    );
    // 8658 pixels of RGB8 are 2165 groups of four pixels, three words each.
    assert_eq!(
        row_words(Encoding::Rgb8 { background: [0; 3] }, 8658),
        2165 * 3
    );
    assert_eq!(row_words(Encoding::Rgb16 { bits: 16 }, 8658), 4329 * 3);
}

/// A grid for an output pixel `q` -> canvas `R(-angle) (scale q + t - c) + c`.
fn rotated_grid(bw: usize, bh: usize, angle_deg: f64, scale: f64, t: (f64, f64)) -> SamplerGrid {
    let (s, c) = (-angle_deg).to_radians().sin_cos();
    let (cx, cy) = (bw as f64 * 0.5, bh as f64 * 0.5);
    SamplerGrid::new(
        [
            [c * scale, -s * scale, c * (t.0 - cx) - s * (t.1 - cy) + cx],
            [s * scale, c * scale, s * (t.0 - cx) + c * (t.1 - cy) + cy],
        ],
        (bw as f64, bh as f64),
        (bw as f64, bh as f64),
        1,
        0,
        SamplerGrid::is_unrotated_angle(angle_deg),
    )
}

#[test]
fn the_device_view_road_equals_the_host_wrapper_road_bit_for_bit() {
    // Sampler, colour pass to C12, output sharpening on the photo's bounding box (the stage mirrors at
    // the photo's edge, the letterbox is zero), C13 with the letterbox painted: all on the device as
    // device images, against the same four steps through the host wrappers (upload and readback
    // between each). The two roads run the same kernels with the same inputs, so the bytes must be
    // equal; what this catches is a stride, an offset, a mask or a window that the entry points mix up.
    let (bw, bh) = (240usize, 180usize);
    let camera = camera_image(bw, bh, 0x71E);
    let grid = rotated_grid(bw, bh, 7.0, 0.8, (-30.0, -20.0));
    let (w, h) = (260usize, 200usize);
    let bounds = grid.inside_bounds((0, 0), w, h).expect("a photo in view");
    let window = Rect::new(bounds.0, bounds.1, bounds.2 - bounds.0, bounds.3 - bounds.1);
    assert!(
        window.w < w || window.h < h,
        "the scene must have a letterbox ({window:?} in {w}x{h})"
    );
    let chain = ColorChain::new(&chains::full(OutputTarget::Viewport)).expect("chain");
    let sharpen = OutputSharpen::screen();
    let bg = [231u8, 231, 231];
    let (c2, cam2, sh2) = (chain.clone(), camera.clone(), sharpen.clone());
    let (device, host) = on_gpu(move |ctx| {
        let space = c2.space();
        let tail = C13 {
            space,
            dither: true,
            origin: (0, 0),
        };
        let prepared = PreparedChain::new(ctx, &c2)?;
        let sharp = PreparedSharpen::new(ctx, &sh2, 1.0)?;

        // The device road: no readback until the end.
        let src = GpuImage::upload_rgb(ctx, "camera", &cam2)?;
        let bytes = band_bytes(Encoding::Rgba8 { background: bg }, w, h);
        let band = ctx.pool().acquire(
            "view",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let mut passes = PassList::new(ctx, "view");
        let sampled = encode_sample(&mut passes, &src, (0, 0), &grid, (0, 0), w, h)?;
        let c12 = GpuImage::alloc(ctx, "c12", w, h, 4, sampled.place)?;
        prepared.encode_stage(
            &mut passes,
            &sampled,
            Rect::whole(w, h),
            Stop::C12,
            &c12,
            (0, 0),
        )?;
        let sharpened = GpuImage::alloc(ctx, "sharpened", w, h, 4, sampled.place)?;
        passes
            .encoder()
            .copy_buffer_to_buffer(&c12.buf, 0, &sharpened.buf, 0, c12.byte_len());
        sharp.encode(
            &mut passes,
            &c12,
            window,
            window,
            &sharpened,
            (window.x, window.y),
        )?;
        encode::encode_c13(
            &mut passes,
            &sharpened,
            Rect::whole(w, h),
            Encoding::Rgba8 { background: bg },
            tail,
            &band,
            0,
            0,
        )?;
        let id = passes.read(&band, 0, bytes)?;
        let _keep = passes.keep(src.buf);
        let done = passes.submit()?.finish()?;
        let device = done.read(id).to_vec();

        // The host road: every step reads its input from the host and returns to it.
        let sampled = sample(ctx, &cam2, (0, 0), &grid, (0, 0), w, h)?;
        let mut display = prepared
            .render(ctx, &sampled.img, (0, 0), ColorOutput::Stage(Stop::C12))?
            .data;
        let PixelData::F32Planar(planes) = &mut display else {
            panic!("not planes");
        };
        let mask = sampled.outside.clone().expect("a letterbox");
        for c in 0..3 {
            for i in 0..w * h {
                if mask[i] != 0 {
                    planes[c * w * h + i] = 0.0;
                }
            }
        }
        let c12 = RgbImage::from_planar(w, h, planes.clone()).expect("image");
        let inner = sharp.run(ctx, &c12.extract(window))?;
        let mut merged = c12.clone();
        merged.paste(&inner, Rect::whole(window.w, window.h), window.x, window.y);
        let host = encode_host(
            ctx,
            &merged,
            Some(&mask),
            Rect::whole(w, h),
            Encoding::Rgba8 { background: bg },
            tail,
        )?
        .bytes;
        Ok((device, host))
    })
    .expect("device");
    assert_eq!(device.len(), host.len());
    let diffs: Vec<usize> = (0..device.len() / 4)
        .filter(|&i| device[4 * i..4 * i + 4] != host[4 * i..4 * i + 4])
        .collect();
    assert!(
        diffs.is_empty(),
        "{} pixels differ between the device road and the host road, first at ({}, {})",
        diffs.len(),
        diffs[0] % w,
        diffs[0] / w
    );
    // The scene is not trivial: some pixels are the letterbox, some are the picture.
    let paper = device.chunks_exact(4).filter(|p| p[..3] == bg).count();
    assert!(
        paper > 100 && paper < w * h - 1000,
        "{paper} paper pixels of {}",
        w * h
    );
}
