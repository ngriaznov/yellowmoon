//! R6's coarse colour pass on the device (`shaders/coarse_chroma.wgsl`, `kernels::coarse_chroma`) against its
//! CPU reference `cpu::coarse_chroma::CoarseChroma::apply`.
//!
//! What is asserted, and what was measured on lavapipe (first measured value; the bound is twice that,
//! `docs/17-gpu.md` section 12):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | natural frames with colour blotches of 10 to 40 px and noise, from 9 x 7 to 211 x 157, two cut out of a larger frame at odd origins, four noise levels, Colour 25 to 100 | max `abs(gpu - cpu)` at most [`BOUND`] | 4.77e-7 over 631 494 samples, mean 2.4e-8 |
//! | a flat colour | the input bit for bit, as on the CPU | |
//! | tiles (two sizes, and windows at odd origins with the stage's halo) | bit-identical to the whole frame | 0 differing |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | `LP_NUM_THREADS=1` against the default | bit-identical | |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`); the
//! Metal runners run the same file (`mac-gpu.yml`).

mod common;
#[path = "denoise_common/mod.rs"]
mod denoise_common;

use common::exact::{
    assert_all_finite, assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole,
};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use denoise_common::{FNV_START, compare, fnv, placed};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::bm_dct::LevelNoise;
use ym_gpu::cpu::coarse_chroma::{CoarseChroma, HALO_PX};
use ym_gpu::cpu::denoise_space::WORKING_GAIN_EV;
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{RgbImage, synth};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::coarse_chroma::{self as gpu_cc, coarse_chroma, encode_coarse_chroma};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// Largest `|gpu - cpu|` of the stage (linear, white 1.0): twice the first measured value on lavapipe (4.77e-7
/// over 631 494 samples, about 4 ulp of samples near 1: the round trip through the working encoding in `f32`,
/// as method A's; the cells, weights and interpolation add less).
const BOUND: f32 = 9.6e-7;

/// The noise models `(a, b)` of the measurement, ISO 100 to 12800 (`tests/gpu_bm_dct.rs`).
const LEVELS: [(f64, f64); 4] = [(2.5e-5, 4e-8), (4e-4, 6e-7), (2e-3, 2e-6), (4e-3, 4e-6)];

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// One run on the device thread.
fn on_device<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("the device job")
}

fn noise(a: f64, b: f64) -> LevelNoise {
    LevelNoise {
        a,
        b,
        gains: [1.9, 1.0, 1.5],
        encoding_gain: WORKING_GAIN_EV.exp2(),
        chroma_scale: 0.0,
    }
}

fn stage(level: usize, color: f32) -> CoarseChroma {
    let (a, b) = LEVELS[level];
    CoarseChroma::resolve(noise(a, b), color).expect("a stage")
}

/// Balanced linear camera RGB as method A leaves it: a natural scene, smooth colour blotches of about 10 to
/// 40 px (red against blue at the same luma, `amp` of the level), and fine noise of `sigma`.
fn blotchy(w: usize, h: usize, seed: u64, amp: f32, sigma: f32) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    let n = img.plane_len();
    let mut rng = synth::Rng::new(seed + 3);
    let phase = rng.next_f32() * 6.0;
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / 9.0, y as f32 / 9.0);
            let t = (fx * 1.3 + fy * 0.6 + phase).sin() * (fy * 1.1 - fx * 0.5).cos()
                + 0.5 * (fx * 0.31 - fy * 0.27).sin();
            let i = y * w + x;
            let base = 0.5 * img.data[n + i] * img.data[n + i] + 0.01;
            img.data[i] = base * (1.9 + amp * t) + sigma * rng.gaussian();
            img.data[n + i] = base + 0.3 * sigma * rng.gaussian();
            img.data[2 * n + i] = base * (1.5 - amp * t) + sigma * rng.gaussian();
        }
    }
    img
}

/// The frames of the parity tests.
fn frames() -> Vec<(&'static str, RgbImage)> {
    vec![
        ("blotchy 9x7", blotchy(9, 7, 1, 0.05, 0.002)),
        ("blotchy 64x48", blotchy(64, 48, 2, 0.05, 0.002)),
        ("blotchy strong 131x97", blotchy(131, 97, 3, 0.3, 0.004)),
        ("blotchy 211x157", blotchy(211, 157, 4, 0.08, 0.003)),
        (
            "blotchy cut at (37, 29) of 400x300",
            placed(blotchy(150, 110, 5, 0.06, 0.002), 37, 29, 400, 300),
        ),
        (
            "blotchy cut at (2, 3) of 120x90",
            placed(blotchy(77, 61, 6, 0.04, 0.001), 2, 3, 120, 90),
        ),
    ]
}

fn run_gpu(dev: &TestGpu, img: &RgbImage, core: Rect, stage: CoarseChroma) -> RgbImage {
    let img = img.clone();
    on_device(dev, move |ctx| coarse_chroma(ctx, &img, core, &stage))
}

#[test]
fn the_stage_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut changed = 0usize;
    let mut total = 0usize;
    for (k, (name, img)) in frames().into_iter().enumerate() {
        let (w, h) = (img.width, img.height);
        for (j, color) in [25.0f32, 50.0, 100.0].into_iter().enumerate() {
            let s = stage((k + j) % LEVELS.len(), color);
            let want = s.apply(&img);
            let got = run_gpu(dev, &img, Rect::whole(w, h), s);
            assert_eq!((got.width, got.height, got.place), (w, h, img.place));
            assert_all_finite(name, &got.data);
            let e = compare(&got.data, &want.data);
            changed += want
                .data
                .iter()
                .zip(&img.data)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            total += got.data.len();
            eprintln!(
                "[R6] coarse colour {name} C {color}: max {:.3e} at {} mean {:.2e}",
                e.max, e.at, e.mean
            );
            worst = worst.max(e.max);
        }
    }
    assert!(
        changed > total / 2,
        "the stage changed {changed} of {total} samples"
    );
    denoise_common::ci_note(
        "coarse colour",
        &format!(
            "coarse colour pass on {}: worst {worst:.3e} over {total} samples",
            dev.gpu.info().name
        ),
    );
    assert!(worst <= BOUND, "{worst:e}");
}

#[test]
fn a_flat_colour_is_left_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (40usize, 32usize);
    let mut data = Vec::with_capacity(3 * w * h);
    for v in [0.1f32, 0.12, 0.08] {
        data.extend(std::iter::repeat_n(v, w * h));
    }
    let img = RgbImage::from_planar(w, h, data).expect("planes");
    for level in 0..LEVELS.len() {
        let s = stage(level, 100.0);
        assert_eq!(s.apply(&img), img);
        let got = run_gpu(dev, &img, Rect::whole(w, h), s);
        assert_bit_equal("a flat colour", &got.data, &img.data);
    }
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (211usize, 157usize);
    let img = blotchy(w, h, 21, 0.08, 0.003);
    let frame = Rect::whole(w, h);
    for s in [stage(1, 60.0), stage(3, 30.0)] {
        assert_eq!(s.halo_px(), HALO_PX);
        let render = |core: Rect| {
            let rect = core.grow(Halo::uniform(HALO_PX), frame);
            run_gpu(dev, &img.extract(rect), core.relative_to(&rect), s)
        };
        let what = format!("coarse colour at strength {}", s.strength);
        let whole = assert_tiles_equal_whole(&what, w, h, 3, &[40, 96], |core| render(core).data);
        let whole_img = RgbImage::from_planar(w, h, whole.clone()).expect("planes");
        for core in [
            Rect::new(23, 31, 57, 49),
            Rect::new(97, 17, 61, 63),
            Rect::new(131, 97, w - 131, h - 97),
            Rect::new(3, 101, 101, h - 101),
        ] {
            let got = render(core);
            assert_eq!(got.place, img.place.offset(core.x, core.y));
            assert_bit_equal(
                &format!("{what}: window {core:?}"),
                &got.data,
                &whole_img.extract(core).data,
            );
        }
        let want = s.apply(&img);
        let e = compare(&whole, &want.data);
        eprintln!(
            "[R6] {what} tiles: whole frame against the CPU {:.3e}",
            e.max
        );
        assert!(e.max <= BOUND, "{:e}", e.max);
    }
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = blotchy(83, 67, 5, 0.08, 0.003);
    assert_poison_neutral("coarse colour", |gpu| {
        let img = img.clone();
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut out = Vec::new();
            for (level, color) in [(1usize, 60.0f32), (3, 25.0)] {
                let s = stage(level, color);
                out.extend(coarse_chroma(ctx, &img, Rect::whole(83, 67), &s)?.data);
                out.extend(coarse_chroma(ctx, &img, Rect::new(5, 7, 61, 43), &s)?.data);
            }
            Ok(out)
        })
        .expect("run")
    });
}

#[test]
fn the_stage_runs_in_one_pass_list_with_its_three_dispatches() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (90usize, 60usize);
    let img = blotchy(w, h, 31, 0.08, 0.003);
    let core = Rect::new(11, 9, 70, 44);
    let s = stage(2, 60.0);
    let (a, subs, recorded) = {
        let img = img.clone();
        on_device(dev, move |ctx| {
            let before = ctx.counters();
            let input = GpuImage::upload_rgb(ctx, "in", &img)?;
            let mut passes = PassList::new(ctx, "coarse colour");
            let start = passes.dispatch_count();
            let out = encode_coarse_chroma(&mut passes, &input, core, &s)?;
            let recorded = passes.dispatch_count() - start;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = out.take(&done, read)?;
            Ok((data, ctx.counters().since(before).submissions, recorded))
        })
    };
    let b = run_gpu(dev, &img, core, s);
    assert_bit_equal("device entry point against the host wrapper", &a, &b.data);
    assert_eq!(subs, 1, "one submission");
    assert_eq!(recorded as usize, gpu_cc::DISPATCHES);
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut hash = FNV_START;
        for (img, s) in [
            (blotchy(97, 61, 1, 0.08, 0.003), stage(1, 60.0)),
            (blotchy(61, 97, 2, 0.3, 0.004), stage(3, 35.0)),
        ] {
            let whole = Rect::whole(img.width, img.height);
            hash = fnv(hash, &run_gpu(dev, &img, whole, s).data);
        }
        hash
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    assert_max_binding_bytes(
        "coarse colour",
        Rect::new(0, 0, side, side),
        gpu_cc::max_binding_bytes,
    );
}

#[test]
fn bad_inputs_are_refused_with_the_reason() {
    let Some(dev) = f32_device() else { return };
    let img = blotchy(30, 20, 1, 0.05, 0.002);
    let whole = Rect::whole(30, 20);
    let run = |img: RgbImage, core: Rect, s: CoarseChroma| {
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                coarse_chroma(ctx, &img, core, &s)
            })
    };
    assert!(run(img.clone(), whole, stage(1, 50.0)).is_ok());
    let nan = CoarseChroma {
        strength: f64::NAN,
        ..stage(1, 50.0)
    };
    let huge = CoarseChroma {
        noise: LevelNoise {
            a: 1e300,
            ..noise(4e-4, 6e-7)
        },
        ..stage(1, 50.0)
    };
    let mut outside = img.clone();
    outside.place.x0 = 5;
    for (what, result) in [
        ("a NaN strength", run(img.clone(), whole, nan)),
        ("a noise beyond f32", run(img.clone(), whole, huge)),
        (
            "core outside",
            run(img.clone(), Rect::new(10, 10, 30, 20), stage(1, 50.0)),
        ),
        (
            "empty core",
            run(img.clone(), Rect::new(0, 0, 0, 5), stage(1, 50.0)),
        ),
        ("outside its frame", run(outside, whole, stage(1, 50.0))),
    ] {
        let err = result.expect_err(what);
        assert!(matches!(err, GpuError::InvalidInput(_)), "{what}: {err}");
    }
}

/// Timing on a 2048 x 1365 frame on this adapter (meaningless for Metal; the number goes into the docs):
/// `cargo test -p ym-gpu --test gpu_coarse_chroma timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn timing_on_a_2048_by_1365_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let img = blotchy(w, h, 9, 0.08, 0.003);
    let s = stage(2, 50.0);
    let ms = on_device(dev, move |ctx| {
        let small = GpuImage::upload_rgb(ctx, "warm", &img.extract(Rect::new(0, 0, 64, 64)))?;
        let mut passes = PassList::new(ctx, "warm");
        let _ = encode_coarse_chroma(&mut passes, &small, Rect::whole(64, 64), &s)?;
        passes.submit()?.finish()?;
        let input = GpuImage::upload_rgb(ctx, "in", &img)?;
        let t = std::time::Instant::now();
        let mut passes = PassList::new(ctx, "whole");
        let _ = encode_coarse_chroma(&mut passes, &input, Rect::whole(w, h), &s)?;
        passes.submit()?.finish()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    });
    let t = std::time::Instant::now();
    let _ = s.apply(&blotchy(w, h, 9, 0.08, 0.003));
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "coarse colour, {w} x {h} on {}: {ms:.0} ms ({:.0} ms/MP), {} dispatches; the CPU reference {cpu_ms:.0} ms on one thread",
        dev.gpu.info().name,
        ms / (w * h) as f64 * 1e6,
        gpu_cc::DISPATCHES
    );
}
