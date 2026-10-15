//! G-10: device residency (`docs/gpu-plan.md` section 11, task G-10). What stays on the device
//! between two jobs, and the proof that keeping it there changes no bit.
//!
//! * **A resident tile is the host tile.** `GpuChain::run_tile_device` leaves the tile of the raw
//!   chain on the device; its readback equals `run_tile_front` bit for bit (cropped on the device
//!   when the last stages were identities, assembled on the device when the demosaic core was cut),
//!   the job reads nothing back and is one submission.
//! * **The handoff to the export tail.** `ExportTail::add_tile_device` makes the very bands
//!   `add_tile` makes from the host copy; a tile of the chain costs at most three submissions
//!   (the chain's job, the tail's tile job and the slab it opens) and no readback, and the readbacks
//!   of the whole run are the bands.
//! * **The pieces.** The copy kernel copies rectangles exactly; a picture in bands (`RowBands`)
//!   gives a window equal to the rows of the host picture whether it lies in one band or crosses
//!   several; the stepwise reduced path (`ReducedRun`) is the one-piece run.
//! * **Nothing leaks, nothing reads what it did not write.** Poison, a token set while a tile runs,
//!   an injected fault: the pool returns to where it was.
//! * **The memory gate** (`#[ignore]`d, 50 MP): the raw chain's tiles and the export tail together,
//!   by the pool ledger, never come near 2 GiB, and give everything back.

mod common;
#[path = "g08/mod.rs"]
mod g08;

use std::sync::Arc;

use common::exact::{assert_bit_equal, assert_poison_neutral};
use common::{TestGpu, gpus};
use ym_color::display::{ColorChain, OutputTarget};
use ym_color::encode::OutputSpace;
use ym_gpu::chain::{DeviceRunner, DeviceWork, GpuChain, StageRange};
use ym_gpu::cpu::chain::{Chain, Prepared, RawFrame};
use ym_gpu::cpu::demosaic::Method;
use ym_gpu::cpu::denoise::BilateralParams;
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::sampler::SamplerGrid;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, NoiseDn, Placement, RgbImage};
use ym_gpu::device::{Gpu, GpuContext, GpuError, StorageMode};
use ym_gpu::image::{GpuImage, RowBands};
use ym_gpu::kernels::copy::{Piece, encode_copy_rect, encode_crop, encode_gather};
use ym_gpu::kernels::prep::DeviceRaw;
use ym_gpu::kernels::{KernelId, KernelSet};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::stream::{CounterSnapshot, PassList};
use ym_gpu::tail::{Band, BandFormat, ExportTail, TailRoute, TailSharpen, TailSpec};

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

fn job() -> JobSpec {
    JobSpec::new(Priority::Export)
}

// ---------------------------------------------------------------------------------------------
// A photo and its chain (the scaffolding of `gpu_chain.rs` in short)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lens {
    /// No lens, no vignetting: the stages that do nothing leave a larger image than the tile needs
    /// (the device crops it).
    Bare,
    /// Vignetting, distortion and a CA table (corrected on the mosaic).
    Full,
}

fn mild_tags() -> LensTags {
    LensTags {
        vignetting: vec![0, -400, -1500, -3500, -6000],
        distortion: vec![0, 30, 100, 220, 400, 650],
        ca: vec![0, 100, 400, 900, 0, -100, -400, -900],
    }
}

struct Photo {
    width: usize,
    height: usize,
    cfa: Cfa,
    data: Vec<u16>,
    chain: Chain,
    prepared: Prepared,
}

impl Photo {
    fn new(width: usize, height: usize, cfa: Cfa, seed: u64, lens: Lens, light: bool) -> Photo {
        let img = synth::natural(width, height, seed);
        let m = synth::mosaic_from_rgb(&img, cfa);
        let mut rng = synth::Rng::new(seed + 7);
        let data: Vec<u16> = m
            .data
            .iter()
            .map(|&v| {
                let x = 100.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 65435.0;
                x.clamp(0.0, 65535.0) as u16
            })
            .collect();
        Photo::with_data(width, height, cfa, data, lens, light)
    }

    fn with_data(
        width: usize,
        height: usize,
        cfa: Cfa,
        data: Vec<u16>,
        lens: Lens,
        light: bool,
    ) -> Photo {
        let tags = mild_tags();
        let geometry = (lens == Lens::Full).then(|| {
            LensGeometry::new(width, height, &tags, LensSwitches::default())
                .expect("lens")
                .with_ca_on_mosaic()
                .with_auto_crop()
        });
        let chain = Chain {
            levels: Levels {
                black: [100.0, 102.0, 101.0, 99.0],
                white: [65535.0; 4],
            },
            vignette: (lens != Lens::Bare)
                .then(|| VignetteProfile::from_tag(&tags.vignetting).expect("vignette")),
            gains: [2.0, 1.0, 1.6],
            highlight_recovery: !light,
            method: Method::AdaptiveRatio,
            denoise: Some(BilateralParams::default()),
            noise_reduction: NoiseReduction {
                luminance: 40.0,
                color: 60.0,
                detail: 70.0,
            },
            iso: 800.0,
            sigma: 0.01,
            sigma_map: None,
            noise: Some(NoiseDn { g: 40.0, r: 90.0 }),
            sharpen: Some(
                SharpenParams {
                    amount: 90.0,
                    radius_px: 0.8,
                }
                .into(),
            ),
            lens: geometry,
        };
        let raw = RawFrame {
            width,
            height,
            cfa,
            data: &data,
            qstep: None,
        };
        let prepared = chain.prepare(&raw).expect("prepare");
        Photo {
            width,
            height,
            cfa,
            data,
            chain,
            prepared,
        }
    }

    fn output_size(&self) -> (usize, usize) {
        self.chain.output_size(self.width, self.height)
    }

    fn upload(&self, ctx: &GpuContext) -> Result<Arc<DeviceRaw>, GpuError> {
        Ok(Arc::new(DeviceRaw::upload(
            ctx,
            "residency photo",
            self.width,
            self.height,
            self.cfa,
            &self.data,
            None,
        )?))
    }

    fn device_chain(&self) -> Result<GpuChain<'_>, GpuError> {
        GpuChain::new(&self.chain, &self.prepared, StageRange::RAW_TO_LENS)
    }
}

/// Runs the device work of a tile as one job of a device from outside its thread, as `ym-core`'s
/// route does (the CPU stages of a tile stay on the calling thread).
struct Submitting<'a> {
    gpu: &'a Gpu,
    spec: JobSpec,
    token: CancelToken,
}

impl DeviceRunner for Submitting<'_> {
    fn run_device(&self, work: DeviceWork, _cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        self.gpu
            .run_with_token(self.spec, self.token.clone(), move |ctx, token| {
                work.run(ctx, token)
            })
    }

    fn run_device_resident(
        &self,
        work: DeviceWork,
        _cancel: &CancelToken,
    ) -> Result<GpuImage, GpuError> {
        self.gpu
            .run_with_token(self.spec, self.token.clone(), move |ctx, token| {
                work.run_resident(ctx, token)
            })
    }
}

/// The same, on the device thread itself, with the demosaic core cut at `limit` padded pixels.
struct Cutting<'a> {
    ctx: &'a GpuContext,
    limit: u64,
}

impl DeviceRunner for Cutting<'_> {
    fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
        let DeviceWork::Raw(w) = work else {
            panic!("the front is a raw group");
        };
        w.run_with_limit(self.ctx, cancel, self.limit)
    }

    fn run_device_resident(
        &self,
        work: DeviceWork,
        cancel: &CancelToken,
    ) -> Result<GpuImage, GpuError> {
        let DeviceWork::Raw(w) = work else {
            panic!("the front is a raw group");
        };
        w.run_resident_with_limit(self.ctx, cancel, self.limit)
    }
}

fn same_bits(what: &str, a: &RgbImage, b: &RgbImage) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
    assert_eq!(a.place, b.place, "{what}: placement");
    assert_bit_equal(what, &a.data, &b.data);
}

// ---------------------------------------------------------------------------------------------
// A resident tile is the host tile
// ---------------------------------------------------------------------------------------------

#[test]
fn a_tile_left_on_the_device_is_the_tile_read_back_bit_for_bit_and_costs_one_submission() {
    let Some(dev) = f32_device() else { return };
    let mut seed = 4200;
    for (cfa, lens) in [
        (Cfa::RGGB, Lens::Full),
        (Cfa::GRBG, Lens::Bare),
        (Cfa::BGGR, Lens::Full),
        (Cfa::GBRG, Lens::Bare),
    ] {
        seed += 1;
        let photo = Arc::new(Photo::new(171, 133, cfa, seed, lens, false));
        let (ow, oh) = photo.output_size();
        for out in [
            Rect::whole(ow, oh),
            Rect::new(0, 0, 40.min(ow), 31.min(oh)),
            Rect::new(
                ow.saturating_sub(37),
                oh.saturating_sub(29),
                37.min(ow),
                29.min(oh),
            ),
            Rect::new(ow / 3, oh / 3, 33.min(ow - ow / 3), 21.min(oh - oh / 3)),
        ] {
            let p = Arc::clone(&photo);
            let (host, resident, ran) = dev
                .gpu
                .run(job(), move |ctx, cancel| {
                    let raw = p.upload(ctx)?;
                    let g = p.device_chain()?;
                    let host = g.run_tile_front(ctx, &raw, out, cancel)?;
                    let before = ctx.counters();
                    let image = g.run_tile_device(ctx, &raw, out, cancel)?;
                    let ran = ctx.counters().since(before);
                    assert_eq!((image.width, image.height), (out.w, out.h));
                    Ok((host, image.read_rgb(ctx)?, ran))
                })
                .expect("tiles");
            let label = format!("{} lens {lens:?} {out:?}", cfa.name());
            same_bits(&label, &host, &resident);
            assert_eq!(ran.readbacks, 0, "{label}: the tile is not read back");
            assert_eq!(ran.submissions, 1, "{label}: one job, one submission");
        }
    }
}

#[test]
fn a_cut_demosaic_core_is_assembled_on_the_device_with_the_host_assemblys_bits() {
    let Some(dev) = f32_device() else { return };
    let mut seed = 4300;
    for cfa in [Cfa::RGGB, Cfa::BGGR] {
        for lens in [Lens::Full, Lens::Bare] {
            seed += 1;
            let photo = Arc::new(Photo::new(173, 131, cfa, seed, lens, false));
            let (ow, oh) = photo.output_size();
            let out = Rect::new(3, 2, ow - 5, oh - 4);
            let p = Arc::clone(&photo);
            let (whole, host_cut, device_cut) = dev
                .gpu
                .run(job(), move |ctx, cancel| {
                    let raw = p.upload(ctx)?;
                    let g = p.device_chain()?;
                    let plain = Cutting {
                        ctx,
                        limit: ym_gpu::chain::MAX_PHYS_AREA,
                    };
                    let cutter = Cutting { ctx, limit: 4_500 };
                    let whole = g.run_tile_front(&plain, &raw, out, cancel)?;
                    let host_cut = g.run_tile_front(&cutter, &raw, out, cancel)?;
                    let device_cut = g
                        .run_tile_device(&cutter, &raw, out, cancel)?
                        .read_rgb(ctx)?;
                    Ok((whole, host_cut, device_cut))
                })
                .expect("tiles");
            let label = format!("{} lens {lens:?}", cfa.name());
            same_bits(
                &format!("{label}: host cut against whole"),
                &host_cut,
                &whole,
            );
            same_bits(
                &format!("{label}: device cut against host cut"),
                &device_cut,
                &host_cut,
            );
        }
    }
}

#[test]
fn a_resident_tile_is_poison_clean_and_a_cancelled_one_hands_everything_back() {
    let Some(dev) = f32_device() else { return };
    let photo = Arc::new(Photo::new(150, 110, Cfa::RGGB, 4400, Lens::Full, false));
    let (ow, oh) = photo.output_size();
    let out = Rect::new(7, 5, ow - 9, oh - 8);
    let p = Arc::clone(&photo);
    let got = assert_poison_neutral("the resident tile", move |gpu| {
        let p = Arc::clone(&p);
        gpu.run(job(), move |ctx, cancel| {
            let raw = p.upload(ctx)?;
            p.device_chain()?
                .run_tile_device(ctx, &raw, out, cancel)?
                .read(ctx)
        })
        .expect("tile")
    });
    assert!(got.is_some(), "a plain device is available");

    // A token that is set before the job: nothing runs and nothing stays allocated. One that is set
    // while the tile is being made: the tile is still handed back (a submitted list cannot be
    // stopped), and dropping it returns its buffers.
    let p = Arc::clone(&photo);
    let (baseline, after_cancel, after_tile, after_drop) = dev
        .gpu
        .run(job(), move |ctx, cancel| {
            let raw = p.upload(ctx)?;
            let g = p.device_chain()?;
            let baseline = ctx.pool().stats().in_use_bytes;
            let token = CancelToken::new();
            token.cancel();
            let cancelled = g.run_tile_device(ctx, &raw, out, &token);
            assert!(
                matches!(cancelled, Err(GpuError::Cancelled)),
                "{cancelled:?}"
            );
            let after_cancel = ctx.pool().stats().in_use_bytes;
            let tile = g.run_tile_device(ctx, &raw, out, cancel)?;
            let after_tile = ctx.pool().stats().in_use_bytes;
            drop(tile);
            let after_drop = ctx.pool().stats().in_use_bytes;
            Ok((baseline, after_cancel, after_tile, after_drop))
        })
        .expect("jobs");
    assert_eq!(after_cancel, baseline, "a cancelled tile allocates nothing");
    assert!(after_tile > baseline, "the tile is on the device");
    assert!(
        after_tile - baseline <= 4 * 3 * 4 * (out.w + 2) as u64 * (out.h + 2) as u64,
        "only the tile (and its padding) stays: {} bytes for {} x {} pixels",
        after_tile - baseline,
        out.w,
        out.h
    );
    assert_eq!(
        after_drop, baseline,
        "dropping the tile hands everything back"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_failed_resident_tile_hands_everything_back_and_leaves_a_usable_device() {
    use ym_gpu::device::GpuFault;
    if f32_device().is_none() {
        return;
    }
    let gpu = Gpu::start(ym_gpu::device::GpuConfig {
        poison: true,
        ..Default::default()
    })
    .expect("device");
    let photo = Arc::new(Photo::new(150, 110, Cfa::GBRG, 4410, Lens::Full, false));
    let (ow, oh) = photo.output_size();
    let out = Rect::new(2, 3, ow - 5, oh - 6);
    let p = Arc::clone(&photo);
    let baseline = gpu
        .run(job(), move |ctx, _| {
            let raw = p.upload(ctx)?;
            drop(raw);
            Ok(ctx.pool().stats().in_use_bytes)
        })
        .expect("baseline");
    for fault in [GpuFault::FailTile(1), GpuFault::OutOfBudget] {
        let p = Arc::clone(&photo);
        // The upload happens before the fault is armed: only the tile fails.
        let (raw, in_use) = gpu
            .run(job(), move |ctx, _| {
                let raw = p.upload(ctx)?;
                Ok((raw, ctx.pool().stats().in_use_bytes))
            })
            .expect("upload");
        gpu.set_fault(Some(fault));
        let p = Arc::clone(&photo);
        let r = Arc::clone(&raw);
        let failed = gpu.run(job(), move |ctx, cancel| {
            p.device_chain()?
                .run_tile_device(ctx, &r, out, cancel)
                .map(|_| ())
        });
        gpu.set_fault(None);
        assert!(failed.is_err(), "{fault:?}: the tile fails");
        let after = gpu
            .run(job(), |ctx, _| Ok(ctx.pool().stats().in_use_bytes))
            .expect("pool");
        assert_eq!(
            after, in_use,
            "{fault:?}: the failed tile left nothing behind"
        );
        drop(raw);
        // And the device works: the same tile runs.
        let p = Arc::clone(&photo);
        gpu.run(job(), move |ctx, cancel| {
            let raw = p.upload(ctx)?;
            p.device_chain()?
                .run_tile_device(ctx, &raw, out, cancel)
                .map(|_| ())
        })
        .expect("the device still works");
        let again = gpu
            .run(job(), |ctx, _| Ok(ctx.pool().stats().in_use_bytes))
            .expect("pool");
        assert_eq!(again, baseline, "{fault:?}: everything is back");
    }
}

#[test]
fn a_runner_that_reads_every_tile_back_refuses_a_resident_one() {
    let Some(dev) = f32_device() else { return };
    struct ReadsBack<'a>(&'a GpuContext);
    impl DeviceRunner for ReadsBack<'_> {
        fn run_device(&self, work: DeviceWork, cancel: &CancelToken) -> Result<RgbImage, GpuError> {
            work.run(self.0, cancel)
        }
    }
    let photo = Arc::new(Photo::new(64, 48, Cfa::RGGB, 4420, Lens::Bare, true));
    let (ow, oh) = photo.output_size();
    let p = Arc::clone(&photo);
    let r = dev
        .gpu
        .run(job(), move |ctx, cancel| {
            let raw = p.upload(ctx)?;
            Ok(p.device_chain()?
                .run_tile_device(&ReadsBack(ctx), &raw, Rect::whole(ow, oh), cancel)
                .map(|_| ()))
        })
        .expect("job");
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
}

// ---------------------------------------------------------------------------------------------
// The copy kernel
// ---------------------------------------------------------------------------------------------

fn pattern(w: usize, h: usize, planes: usize, seed: u32) -> Vec<f32> {
    (0..planes * w * h)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) as f32 * 1e-6 - 1000.0)
        .collect()
}

#[test]
fn the_copy_kernel_moves_rectangles_exactly_for_every_plane_count() {
    let Some(dev) = f32_device() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            for planes in [1usize, 3, 4] {
                for (sw, sh, rect, at, dw, dh) in [
                    (
                        37usize,
                        29usize,
                        Rect::new(0, 0, 37, 29),
                        (0usize, 0usize),
                        37usize,
                        29usize,
                    ),
                    (37, 29, Rect::new(5, 3, 20, 17), (7, 9), 41, 33),
                    (37, 29, Rect::new(36, 28, 1, 1), (40, 32), 41, 33),
                    (16, 16, Rect::new(1, 1, 14, 14), (0, 0), 14, 14),
                    (1, 1, Rect::new(0, 0, 1, 1), (2, 1), 3, 2),
                ] {
                    let host = pattern(sw, sh, planes, 7);
                    let src = GpuImage::upload(
                        ctx,
                        "src",
                        sw,
                        sh,
                        planes,
                        &host,
                        Placement::whole(sw, sh),
                    )?;
                    // The destination starts as a known pattern, so what the kernel leaves alone is
                    // checked too.
                    let base = pattern(dw, dh, planes, 99);
                    let dst = GpuImage::upload(
                        ctx,
                        "dst",
                        dw,
                        dh,
                        planes,
                        &base,
                        Placement::whole(dw, dh),
                    )?;
                    let mut passes = PassList::new(ctx, "copy test");
                    encode_copy_rect(&mut passes, &src, rect, &dst, at)?;
                    passes.submit()?.finish()?;
                    let got = dst.read(ctx)?;
                    let mut want = base.clone();
                    for c in 0..planes {
                        for y in 0..rect.h {
                            for x in 0..rect.w {
                                want[(c * dh + at.1 + y) * dw + at.0 + x] =
                                    host[(c * sh + rect.y + y) * sw + rect.x + x];
                            }
                        }
                    }
                    assert_bit_equal(
                        &format!("{planes} planes, {sw}x{sh} {rect:?} at {at:?}"),
                        &got,
                        &want,
                    );

                    // The crop keeps the rectangle's place in the source's frame.
                    let mut passes = PassList::new(ctx, "crop test");
                    let cropped = encode_crop(&mut passes, &src, rect)?;
                    passes.submit()?.finish()?;
                    assert_eq!(cropped.place, src.place.offset(rect.x, rect.y));
                    let got = cropped.read(ctx)?;
                    let mut want = Vec::new();
                    for c in 0..planes {
                        for y in 0..rect.h {
                            let from = (c * sh + rect.y + y) * sw + rect.x;
                            want.extend_from_slice(&host[from..from + rect.w]);
                        }
                    }
                    assert_bit_equal("crop", &got, &want);
                }
            }
            // What the kernel refuses.
            let a = GpuImage::upload(
                ctx,
                "a",
                8,
                8,
                3,
                &pattern(8, 8, 3, 1),
                Placement::whole(8, 8),
            )?;
            let b = GpuImage::upload(
                ctx,
                "b",
                8,
                8,
                1,
                &pattern(8, 8, 1, 2),
                Placement::whole(8, 8),
            )?;
            let mut passes = PassList::new(ctx, "refusals");
            for r in [
                encode_copy_rect(&mut passes, &a, Rect::whole(8, 8), &b, (0, 0)),
                encode_copy_rect(&mut passes, &a, Rect::new(0, 0, 9, 8), &a, (0, 0)),
                encode_copy_rect(&mut passes, &a, Rect::new(4, 4, 0, 4), &a, (0, 0)),
                encode_copy_rect(&mut passes, &a, Rect::whole(8, 8), &a, (1, 0)),
            ] {
                assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
            }
            Ok(())
        })
        .expect("job");
}

// ---------------------------------------------------------------------------------------------
// A picture in bands
// ---------------------------------------------------------------------------------------------

/// A planar picture of `planes` planes cut into bands of the given heights.
fn bands_of(
    ctx: &GpuContext,
    host: &[f32],
    w: usize,
    heights: &[usize],
    planes: usize,
) -> Result<RowBands, GpuError> {
    let total: usize = heights.iter().sum();
    let mut bands = Vec::new();
    let mut y0 = 0;
    for &h in heights {
        let mut part = Vec::with_capacity(planes * w * h);
        for c in 0..planes {
            let from = (c * total + y0) * w;
            part.extend_from_slice(&host[from..from + w * h]);
        }
        bands.push(GpuImage::upload(
            ctx,
            "band",
            w,
            h,
            planes,
            &part,
            Placement {
                x0: 0,
                y0,
                frame_width: w,
                frame_height: total,
            },
        )?);
        y0 += h;
    }
    RowBands::new(bands)
}

#[test]
fn a_picture_in_bands_reads_back_whole_and_gives_windows_equal_to_its_rows() {
    let Some(dev) = f32_device() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            for (w, heights, planes) in [
                (37usize, vec![10usize, 7, 12, 1, 9], 3usize),
                (16, vec![5], 3),
                (21, vec![4, 4, 4], 1),
                (9, vec![3, 8], 4),
            ] {
                let total: usize = heights.iter().sum();
                let host = pattern(w, total, planes, 3);
                let bands = bands_of(ctx, &host, w, &heights, planes)?;
                assert_eq!(
                    (bands.width, bands.height, bands.planes),
                    (w, total, planes)
                );
                assert_eq!(bands.bands().len(), heights.len());
                assert_bit_equal("the whole picture", &bands.read(ctx)?, &host);
                assert_eq!(
                    bands.max_band_bytes(),
                    heights
                        .iter()
                        .map(|&h| GpuImage::buffer_bytes(w, h, planes))
                        .max()
                        .unwrap_or(0)
                );
                for rows in [
                    0..total,
                    0..1,
                    total - 1..total,
                    3..total.min(15),
                    2..total.min(3),
                ] {
                    if rows.is_empty() {
                        continue;
                    }
                    let mut passes = PassList::new(ctx, "window");
                    let (win, off) = bands.window(&mut passes, rows.clone())?;
                    passes.submit()?.finish()?;
                    // The rows sit `off` rows down the window, and the window says where it is.
                    let got = win.read(ctx)?;
                    let n = rows.len();
                    let mut want = Vec::new();
                    for c in 0..planes {
                        for y in 0..n {
                            let from = (c * total + rows.start + y) * w;
                            want.extend_from_slice(&host[from..from + w]);
                        }
                    }
                    let mut have = Vec::new();
                    for c in 0..planes {
                        for y in 0..n {
                            let from = (c * win.height + off + y) * w;
                            have.extend_from_slice(&got[from..from + w]);
                        }
                    }
                    assert_bit_equal(&format!("window {rows:?}"), &have, &want);
                    assert_eq!(win.place.y0 + off, rows.start, "{rows:?}: where it sits");
                }
                for rows in [0..0, 0..total + 1, total..total] {
                    let mut passes = PassList::new(ctx, "refusals");
                    assert!(
                        matches!(
                            bands.window(&mut passes, rows.clone()),
                            Err(GpuError::InvalidInput(_))
                        ),
                        "{rows:?}"
                    );
                }
            }
            // Bands that do not make a picture are refused.
            let a = GpuImage::alloc(
                ctx,
                "a",
                8,
                4,
                3,
                Placement {
                    x0: 0,
                    y0: 0,
                    frame_width: 8,
                    frame_height: 9,
                },
            )?;
            let gap = GpuImage::alloc(
                ctx,
                "gap",
                8,
                4,
                3,
                Placement {
                    x0: 0,
                    y0: 5,
                    frame_width: 8,
                    frame_height: 9,
                },
            )?;
            assert!(matches!(
                RowBands::new(vec![a, gap]),
                Err(GpuError::InvalidInput(_))
            ));
            assert!(matches!(
                RowBands::new(vec![]),
                Err(GpuError::InvalidInput(_))
            ));
            Ok(())
        })
        .expect("job");
}

/// A planar picture of `planes` planes cut into a grid of tiles of `tw x th` (the last ones smaller),
/// as the pieces the cache of the 100 % view holds.
fn tiles_of(
    ctx: &GpuContext,
    host: &[f32],
    (w, h): (usize, usize),
    (tw, th): (usize, usize),
    planes: usize,
) -> Result<Vec<Piece>, GpuError> {
    let mut pieces = Vec::new();
    for y0 in (0..h).step_by(th) {
        for x0 in (0..w).step_by(tw) {
            let (pw, ph) = (tw.min(w - x0), th.min(h - y0));
            let mut part = Vec::with_capacity(planes * pw * ph);
            for c in 0..planes {
                for y in 0..ph {
                    let from = (c * h + y0 + y) * w + x0;
                    part.extend_from_slice(&host[from..from + pw]);
                }
            }
            let image =
                GpuImage::upload(ctx, "tile", pw, ph, planes, &part, Placement::whole(pw, ph))?;
            pieces.push(Piece {
                at: (x0, y0),
                image: Arc::new(image),
            });
        }
    }
    Ok(pieces)
}

#[test]
fn a_window_gathered_from_pieces_is_the_host_crop_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let (w, h, planes) = (50usize, 30usize, 3usize);
            let host = pattern(w, h, planes, 11);
            let crop = |r: Rect| -> Vec<f32> {
                let mut want = Vec::new();
                for c in 0..planes {
                    for y in 0..r.h {
                        let from = (c * h + r.y + y) * w + r.x;
                        want.extend_from_slice(&host[from..from + r.w]);
                    }
                }
                want
            };
            let tiles = tiles_of(ctx, &host, (w, h), (16, 11), planes)?;
            let bands = bands_of(ctx, &host, w, &[10, 7, 12, 1], planes)?;
            let windows = [
                Rect::new(0, 0, w, h),
                Rect::new(3, 2, 9, 7),
                Rect::new(10, 5, 30, 20),
                Rect::new(15, 10, 2, 2),
                Rect::new(49, 29, 1, 1),
                Rect::new(32, 22, 18, 8),
                Rect::new(0, 11, 16, 11),
            ];
            for (what, pieces) in [("tiles", tiles.clone()), ("bands", bands.pieces())] {
                for win in windows {
                    let mut passes = PassList::new(ctx, "gather");
                    let got = encode_gather(&mut passes, &pieces, win)?;
                    passes.submit()?.finish()?;
                    assert_eq!((got.width, got.height, got.planes), (win.w, win.h, planes));
                    assert_bit_equal(&format!("{what} {win:?}"), &got.read(ctx)?, &crop(win));
                }
            }
            // A window that is one piece is that piece, with no copy.
            let exact = Rect::new(16, 11, 16, 11);
            let one = tiles.iter().find(|p| p.rect() == exact).expect("a tile");
            let mut passes = PassList::new(ctx, "exact");
            let got = encode_gather(&mut passes, std::slice::from_ref(one), exact)?;
            assert!(
                Arc::ptr_eq(&got, &one.image),
                "no copy for a window that is a piece"
            );
            // What it refuses: no piece, an empty window, a window the pieces do not cover.
            let mut passes = PassList::new(ctx, "refusals");
            let hole: Vec<Piece> = tiles.iter().filter(|p| p.at != (16, 11)).cloned().collect();
            for r in [
                encode_gather(&mut passes, &[], exact),
                encode_gather(&mut passes, &tiles, Rect::new(4, 4, 0, 3)),
                encode_gather(&mut passes, &hole, Rect::new(10, 5, 30, 20)),
                encode_gather(&mut passes, &tiles, Rect::new(40, 20, 20, 20)),
            ] {
                assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
            }
            Ok(())
        })
        .expect("job");
}

/// A cold start compiles what the first frame of the first photo waits for before what only a zoomed
/// view or an export needs (`KernelId::ALL` is the warm-up order).
#[test]
fn a_cold_start_compiles_the_fit_view_then_the_view_stages_then_the_full_resolution_chain() {
    use KernelId::*;
    let at = |id: KernelId| KernelId::ALL.iter().position(|k| *k == id).expect("listed");
    let fit = [
        Prep,
        PrepMask,
        HighlightsSuper,
        Highlights,
        MosaicCa,
        Sigma,
        Bin,
        BinSigma,
        Reduce,
        ReduceFold,
        SigmaExpand,
        Bilateral,
        Lens,
        Copy,
    ];
    let view = [
        Sampler,
        Color,
        OutputSharpenLuma,
        OutputSharpenHighpassRows,
        OutputSharpenHighpassCols,
        OutputSharpenGainTerms,
        OutputSharpenBlur,
        OutputSharpenAccumulate,
        OutputSharpenWindow,
        OutputSharpenTransfer,
        Encode,
    ];
    let unused = [SpikeNormalize, SpikeDemosaic, SpikeMatrix, Probe];
    let last_of = |ids: &[KernelId]| ids.iter().map(|&k| at(k)).max().expect("some");
    let first_of = |ids: &[KernelId]| ids.iter().map(|&k| at(k)).min().expect("some");
    assert_eq!(first_of(&fit), 0);
    assert!(
        last_of(&fit) < first_of(&view),
        "the fit view before the view stages"
    );
    let chain: Vec<KernelId> = KernelId::ALL
        .into_iter()
        .filter(|k| !fit.contains(k) && !view.contains(k) && !unused.contains(k) && *k != Resize)
        .collect();
    assert!(!chain.is_empty());
    assert!(
        last_of(&view) < first_of(&chain),
        "the view stages before the full-resolution chain"
    );
    assert!(
        last_of(&chain) < at(Resize),
        "the export tail after the chain"
    );
    assert!(
        at(Resize) < first_of(&unused),
        "what no render uses is last"
    );
    // Every kernel is listed once, and the sets are views of the one order.
    let mut seen = KernelId::ALL.to_vec();
    seen.sort_by_key(|k| *k as usize);
    seen.dedup();
    assert_eq!(seen.len(), KernelId::ALL.len());
    assert_eq!(
        KernelSet::All.kernels().collect::<Vec<_>>(),
        KernelId::ALL.to_vec()
    );
    let view_set: Vec<KernelId> = KernelSet::View.kernels().collect();
    assert!(view_set.windows(2).all(|p| at(p[0]) < at(p[1])));
}

// ---------------------------------------------------------------------------------------------
// What recording costs (the UniformArena / BindGroupCache / ReadbackRing decision, X-17)
// ---------------------------------------------------------------------------------------------

/// The measurement the plan asks for before an arena of uniforms, a cache of bind groups or a ring of
/// readback buffers is built (`docs/gpu-plan.md` V-2, X-17): the CPU time of recording one dispatch
/// (a uniform buffer from the pool, a queue write, a bind group, a compute pass), the number of
/// dispatches, uploads and submissions a tile costs, and the share of the tile's wall time that
/// recording is. The numbers go to the log (`[G-10] ...`, and a GitHub annotation); the bounds are
/// sanity bounds. The Mac session (M0-H4) reads the same lines on Metal, where a tile takes
/// milliseconds and the share may be larger than on lavapipe.
#[test]
fn what_recording_costs_per_dispatch_and_per_tile_is_measured_for_the_arena_decision() {
    use std::time::{Duration, Instant};
    let Some(dev) = f32_device() else { return };
    // The cost of one recorded dispatch: copies of 16 x 16 pixels, so that the device's own time is
    // the least it can be, in lists of 250.
    let (per_dispatch, per_run) = dev
        .gpu
        .run(job(), |ctx, _| {
            let a = GpuImage::upload(
                ctx,
                "a",
                64,
                64,
                3,
                &pattern(64, 64, 3, 5),
                Placement::whole(64, 64),
            )?;
            let b = GpuImage::alloc(ctx, "b", 64, 64, 3, Placement::whole(64, 64))?;
            let record = |passes: &mut PassList<'_>| -> Result<(), GpuError> {
                for _ in 0..250 {
                    encode_copy_rect(passes, &a, Rect::new(0, 0, 16, 16), &b, (0, 0))?;
                }
                Ok(())
            };
            // One list first, so that the pipeline is compiled and the pool is warm.
            let mut passes = PassList::new(ctx, "warm");
            record(&mut passes)?;
            passes.submit()?.finish()?;
            let (mut rec, mut run) = (Duration::ZERO, Duration::ZERO);
            for _ in 0..8 {
                let t = Instant::now();
                let mut passes = PassList::new(ctx, "micro");
                record(&mut passes)?;
                rec += t.elapsed();
                let t = Instant::now();
                passes.submit()?.finish()?;
                run += t.elapsed();
            }
            Ok((rec.as_secs_f64() / 2000.0, run.as_secs_f64() / 2000.0))
        })
        .expect("micro");
    let mut lines = vec![format!(
        "recording one dispatch (uniform from the pool, queue write, bind group, pass): {:.1} us; submit and finish: {:.1} us per dispatch for trivial passes",
        per_dispatch * 1e6,
        per_run * 1e6
    )];
    // A tile of the raw chain, from the resident words, as the view and the export make them.
    let photo = Arc::new(Photo::new(512, 384, Cfa::RGGB, 91, Lens::Full, false));
    let (ow, oh) = photo.output_size();
    let out = Rect::new(ow / 4, oh / 4, 256.min(ow / 2), 192.min(oh / 2));
    let p = Arc::clone(&photo);
    let (wall, ran) = dev
        .gpu
        .run(job(), move |ctx, cancel| {
            let raw = p.upload(ctx)?;
            let g = p.device_chain()?;
            // A first tile compiles the pipelines of the chain.
            drop(g.run_tile_device(ctx, &raw, out, cancel)?);
            let before = ctx.counters();
            let t = Instant::now();
            drop(g.run_tile_device(ctx, &raw, out, cancel)?);
            Ok((t.elapsed().as_secs_f64(), ctx.counters().since(before)))
        })
        .expect("tile");
    let share = ran.dispatches as f64 * per_dispatch / wall;
    lines.push(format!(
        "a {}x{} tile of R1..R9 from the resident words: {} submissions, {} dispatches, {} uploads ({} B), {} readbacks, {:.0} ms; recording is about {:.2} % of it",
        out.w,
        out.h,
        ran.submissions,
        ran.dispatches,
        ran.uploads,
        ran.uploaded_bytes,
        ran.readbacks,
        wall * 1e3,
        share * 1e2
    ));
    for line in &lines {
        eprintln!("[G-10] {line}");
        if std::env::var_os("GITHUB_ACTIONS").is_some() {
            eprintln!("::notice title=G-10 recording cost::{line}");
        }
    }
    assert!(
        per_dispatch < 5e-3,
        "recording a dispatch took {per_dispatch} s"
    );
    assert!(
        ran.submissions <= 3,
        "{} submissions for one tile",
        ran.submissions
    );
    assert!(ran.dispatches > 0 && ran.readbacks == 0);
}

// ---------------------------------------------------------------------------------------------
// The handoff to the export tail
// ---------------------------------------------------------------------------------------------

fn tail_spec(
    ow: usize,
    oh: usize,
    out: (usize, usize),
    sharpen_pct: f64,
    format: BandFormat,
) -> TailSpec {
    let chain =
        Arc::new(ColorChain::new(&common::chains::full(OutputTarget::Srgb)).expect("colour chain"));
    TailSpec {
        rendered: (ow, oh),
        out,
        chain,
        // One picture pixel per buffer pixel, no turn: the chain's output is the camera image.
        grid: SamplerGrid::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            (ow as f64, oh as f64),
            (ow as f64, oh as f64),
            1,
            0,
            true,
        ),
        sharpen: (sharpen_pct > 0.0).then(|| TailSharpen {
            params: OutputSharpen::screen().with_amount_pct(sharpen_pct as f32),
            scale: scale_for_edges(out.0.max(out.1), ow.max(oh)),
        }),
        space: OutputSpace::Srgb,
        format,
    }
}

/// What a run through the tail did.
struct Run {
    bands: Vec<Band>,
    /// The largest number of submissions the tile phase of one tile cost (the chain's job and the
    /// tail's tile job, the slab it opens), and the readbacks of the whole run.
    worst_tile_submissions: u64,
    tile_readbacks: u64,
    total_readbacks: u64,
    tiles: usize,
    /// Seconds the chain's job took for each tile (the call of `run_tile_device` or
    /// `run_tile_front`, the device's work plus the host stages around it).
    chain_seconds: Vec<f64>,
}

/// The raw chain's tiles of `photo` through the tail, in tile rows, with the tiles handed over as host
/// images (`resident` false) or left on the device (`resident` true).
fn through_the_tail(
    gpu: &Gpu,
    photo: &Arc<Photo>,
    tile: usize,
    out: (usize, usize),
    sharpen_pct: f64,
    format: BandFormat,
    resident: bool,
) -> Run {
    let (ow, oh) = photo.output_size();
    let p = Arc::clone(photo);
    let raw = gpu.run(job(), move |ctx, _| p.upload(ctx)).expect("upload");
    let runner = Submitting {
        gpu,
        spec: job(),
        token: CancelToken::new(),
    };
    let chain = photo.device_chain().expect("chain");
    let route = TailRoute::new(gpu, job(), CancelToken::new());
    let mut tail =
        ExportTail::new(route, tail_spec(ow, oh, out, sharpen_pct, format)).expect("tail");
    let mut bands = Vec::new();
    let mut worst = 0u64;
    let mut tile_readbacks = 0u64;
    let mut tiles = 0usize;
    let mut chain_seconds = Vec::new();
    let start: CounterSnapshot = gpu.counters();
    let mut oy = 0;
    while oy < oh {
        let th = tile.min(oh - oy);
        let mut ox = 0;
        while ox < ow {
            let tw = tile.min(ow - ox);
            // The chain's tile is the rectangle the sampler reads, with a margin for its filters.
            let (x0, y0) = (ox.saturating_sub(4), oy.saturating_sub(4));
            let (x1, y1) = ((ox + tw + 4).min(ow), (oy + th + 4).min(oh));
            let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
            let before = gpu.counters();
            if resident {
                let started = std::time::Instant::now();
                let image = chain
                    .run_tile_device(&runner, &raw, rect, &CancelToken::new())
                    .expect("a resident tile");
                chain_seconds.push(started.elapsed().as_secs_f64());
                tail.add_tile_device(Arc::new(image), (x0, y0), (ox, oy), tw, th)
                    .expect("a device tile");
            } else {
                let started = std::time::Instant::now();
                let image = chain
                    .run_tile_front(&runner, &raw, rect, &CancelToken::new())
                    .expect("a host tile");
                chain_seconds.push(started.elapsed().as_secs_f64());
                tail.add_tile(Arc::new(image), (x0, y0), (ox, oy), tw, th)
                    .expect("a host tile");
            }
            let ran = gpu.counters().since(before);
            worst = worst.max(ran.submissions);
            tile_readbacks += ran.readbacks;
            tiles += 1;
            ox += tw;
        }
        bands.extend(tail.advance(oy + th).expect("advance"));
        oy += th;
    }
    bands.extend(tail.finish().expect("finish"));
    let total = gpu.counters().since(start);
    Run {
        bands,
        worst_tile_submissions: worst,
        tile_readbacks,
        total_readbacks: total.readbacks,
        tiles,
        chain_seconds,
    }
}

/// The plan's bound of three submissions per tile (the chain's job, the tail's tile job, the slabs the
/// tile opens) holds when a tile is taller than a slab: a tile of 1024 rows over slabs of 512 opens two
/// of them, and they are made by one job (the 50 MP run of `fifty_megapixels_...` found four
/// submissions when each slab was a job of its own). The camera RGB here is a pattern, the chain is
/// not involved: the tail's share is one job for the slabs a tile opens and one for the tile.
#[test]
fn a_tile_taller_than_a_slab_opens_its_slabs_in_one_job() {
    let Some(gpu) = private_device() else { return };
    let (ow, oh, tile) = (300usize, 2300usize, 1024usize);
    let route = TailRoute::new(&gpu, job(), CancelToken::new());
    let mut tail = ExportTail::new(route, tail_spec(ow, oh, (ow, oh), 0.0, BandFormat::Sixteen))
        .expect("tail");
    let mut per_tile = Vec::new();
    let mut oy = 0;
    while oy < oh {
        let th = tile.min(oh - oy);
        let mut ox = 0;
        while ox < ow {
            let tw = tile.min(ow - ox);
            let (x0, y0) = (ox.saturating_sub(4), oy.saturating_sub(4));
            let (x1, y1) = ((ox + tw + 4).min(ow), (oy + th + 4).min(oh));
            let (rw, rh) = (x1 - x0, y1 - y0);
            let region = gpu
                .run(job(), move |ctx, _| {
                    GpuImage::upload(
                        ctx,
                        "camera",
                        rw,
                        rh,
                        3,
                        &pattern(rw, rh, 3, (oy + ox) as u32),
                        Placement::whole(rw, rh),
                    )
                })
                .expect("camera rgb");
            let before = gpu.counters();
            tail.add_tile_device(Arc::new(region), (x0, y0), (ox, oy), tw, th)
                .expect("a device tile");
            per_tile.push(gpu.counters().since(before).submissions);
            ox += tw;
        }
        tail.advance(oy + th).expect("advance");
        oy += th;
    }
    tail.finish().expect("finish");
    eprintln!("[G-10] tail submissions of tiles of 1024 rows over slabs of 512: {per_tile:?}");
    // The first tile of a row opens two slabs and is the tile's own job: two submissions. (A tile row
    // of 2300 rows: 1024, 1024 and 252; the last tile row opens one slab at most.)
    assert_eq!(per_tile[0], 2, "two slabs in one job, and the tile");
    assert!(per_tile.iter().all(|&n| n <= 2), "{per_tile:?}");
}

/// A device of this test's own, in poison mode: the tests of a binary run in parallel on the shared
/// device, and the counters of a device count everything that happens on it.
fn private_device() -> Option<Gpu> {
    f32_device()?;
    Some(
        Gpu::start(ym_gpu::device::GpuConfig {
            poison: true,
            ..Default::default()
        })
        .expect("a device of its own"),
    )
}

#[test]
fn tiles_handed_over_on_the_device_make_the_bands_the_host_copies_make() {
    let Some(gpu) = private_device() else { return };
    let mut seed = 4500;
    for (name, lens, size, tile, out, pct, format) in [
        (
            "plain, 8 bit",
            Lens::Full,
            (190usize, 150usize),
            80usize,
            None,
            0.0,
            BandFormat::Eight { dither: true },
        ),
        (
            "sharpened, 16 bit",
            Lens::Bare,
            (176, 140),
            64,
            None,
            100.0,
            BandFormat::Sixteen,
        ),
        (
            "shrunk and sharpened, 8 bit",
            Lens::Full,
            (200, 160),
            90,
            Some((101usize, 80usize)),
            100.0,
            BandFormat::Eight { dither: true },
        ),
    ] {
        seed += 1;
        let photo = Arc::new(Photo::new(size.0, size.1, Cfa::RGGB, seed, lens, true));
        let (ow, oh) = photo.output_size();
        let out = out.unwrap_or((ow, oh));
        let host = through_the_tail(&gpu, &photo, tile, out, pct, format, false);
        let resident = through_the_tail(&gpu, &photo, tile, out, pct, format, true);
        assert_eq!(
            host.bands, resident.bands,
            "{name}: the bands of a resident run are the host run's"
        );
        assert!(!resident.bands.is_empty());
        // The host run reads every tile back; the resident one reads nothing but the bands.
        assert!(host.tile_readbacks as usize >= host.tiles, "{name}");
        assert_eq!(resident.tile_readbacks, 0, "{name}: no tile is read back");
        assert_eq!(
            resident.total_readbacks as usize,
            resident.bands.len(),
            "{name}: the readbacks are the bands"
        );
        // The plan: at most three submissions per tile (the chain's job, the tail's tile job, the slab
        // that tile opened); the host path has the same count, with a readback on top.
        assert!(
            resident.worst_tile_submissions <= 3,
            "{name}: {} submissions for one tile",
            resident.worst_tile_submissions
        );
        eprintln!(
            "[G-10] {name}: {} tiles, at most {} submissions per tile (host copies: {}), readbacks {} (host copies: {})",
            resident.tiles,
            resident.worst_tile_submissions,
            host.worst_tile_submissions,
            resident.total_readbacks,
            host.total_readbacks
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The memory gate
// ---------------------------------------------------------------------------------------------

/// 50 MP (8688 x 5792): the raw chain's tiles of 1024 px, from the photo's resident words, through
/// the export tail with the tiles left on the device. By the pool ledger the whole run (the words,
/// the tiles in flight, the slabs, the sharpening window, the bands) peaks under 2 GiB (the plan's
/// gate; the host path has the same working set plus a tile's readback and upload), and the device
/// holds only the words once the run is over. Ignored: it makes 100 MB of words and runs the chain
/// over 50 million pixels (minutes on lavapipe in a test build); wave 5 runs it on Metal.
#[test]
#[ignore = "50 MP: run with --ignored (wave 5 runs it on Metal)"]
fn fifty_megapixels_through_the_chain_and_the_tail_stay_under_two_gib() {
    let Some(gpu) = common::start_unpoisoned(ym_gpu::device::StoragePreference::Auto) else {
        return;
    };
    let (w, h) = (8688usize, 5792usize);
    let mut rng = synth::Rng::new(77);
    let data: Vec<u16> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let base = 400.0 + 9000.0 * (x as f32 / w as f32) * (0.4 + 0.6 * (y as f32 / h as f32));
            (base + 120.0 * rng.next_f32()).min(65535.0) as u16
        })
        .collect();
    // R5's offsets are the CPU's whole-frame pass; the gate is about the device.
    let photo = Arc::new(Photo::with_data(w, h, Cfa::RGGB, data, Lens::Full, true));
    let (ow, oh) = photo.output_size();
    gpu.run(job(), |ctx, _| {
        ctx.pool().reset_peaks();
        Ok(())
    })
    .expect("reset");
    let baseline = gpu
        .run(job(), |ctx, _| Ok(ctx.pool().stats().in_use_bytes))
        .expect("pool");
    let started = std::time::Instant::now();
    let run = through_the_tail(
        &gpu,
        &photo,
        1024,
        (ow, oh),
        100.0,
        BandFormat::Sixteen,
        true,
    );
    let seconds = started.elapsed().as_secs_f64();
    let (peak, in_use) = gpu
        .run(job(), |ctx, _| {
            let s = ctx.pool().stats();
            Ok((s.peak_in_use_bytes, s.in_use_bytes))
        })
        .expect("pool");
    let rows: usize = run.bands.iter().map(|b| b.rows).sum();
    assert_eq!(rows, oh, "the bands cover the picture");
    eprintln!(
        "[G-10] 50 MP chain + tail ({ow} x {oh}, {} tiles): {seconds:.0} s, at most {} submissions per tile, peak in use {} MiB (the gate: 2048 MiB), in use after {} MiB (before {} MiB)",
        run.tiles,
        run.worst_tile_submissions,
        peak >> 20,
        in_use >> 20,
        baseline >> 20
    );
    let mean = run.chain_seconds.iter().sum::<f64>() / run.chain_seconds.len().max(1) as f64;
    let longest = run.chain_seconds.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "[G-10] 50 MP chain job per 1024 px tile (about 1.1 MP of mosaic each): mean {mean:.2} s, longest {longest:.2} s over {} tiles",
        run.chain_seconds.len()
    );
    assert!(peak < 2 << 30, "peak {peak} bytes");
    assert!(run.worst_tile_submissions <= 3);
    assert_eq!(run.tile_readbacks, 0);
    // The run is over: what is left is what the colour chain keeps for the next frame (its tables)
    // and the pool's free lists, not the words (they went with `through_the_tail`) or any tile.
    let after = gpu
        .run(job(), |ctx, _| {
            ym_gpu::kernels::color::release_tables(ctx);
            ctx.pool().trim();
            Ok(ctx.pool().stats().in_use_bytes)
        })
        .expect("pool");
    assert_eq!(after, baseline, "everything is handed back");
}
