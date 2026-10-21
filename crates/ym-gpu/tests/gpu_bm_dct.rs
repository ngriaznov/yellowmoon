//! R6's single-frame denoise by method A on the device (`shaders/bm_dct*.wgsl`, `kernels::bm_dct`) against
//! its CPU reference `cpu::frame_denoise::FrameDenoise::apply` (the working encoding, `cpu::bm_dct::denoise`,
//! the decoding).
//!
//! What is asserted, and what was measured on lavapipe (first measured value; the bound is twice that,
//! `docs/17-gpu.md` section 12):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | 30 frames from 8 x 8 to 301 x 203, R 0 to 3, luma only, chroma only and both, blend 0 and 15, linear values from below 0 to above 1 | max `abs(gpu - cpu)` at most [`BOUND`]; at most [`OVER_1E4`] of the samples over 1e-4 | 2.09e-6, 0 over 1e-4 of 1.9 million (no decision flipped) |
//! | the whole frames of the tile test (four settings) | at most [`FLIP_LARGEST`], at most [`FLIP_OVER_1E4`] of a frame over 1e-4 | 1.55e-6 in three; one decision of the luma channel went the other way in the fourth: 1.68e-3, 204 of 99 381 over 1e-4 |
//! | noise-free synthetic content (a ramp, flat patches, clipped plateaus), where exact ties of mirror-image candidates are broken by each side's rounding | at most [`TIE_PRONE_LARGEST`], [`TIE_PRONE_OVER_1E4`] over 1e-4 | 7.3e-3, 5.7e-2 |
//! | strengths that filter nothing, frames under 8 px | the CPU's encode and decode within [`ROUND_TRIP_BOUND`] | 2.4e-7 |
//! | tiles (two sizes on the even grid, and windows at odd origins with the 23 px halo, one at the far edges) | bit-identical to the whole frame | 0 differing |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | windows of an 8658 x 5772 frame at its origin, its middle and its far corner | within [`FLIP_LARGEST`] of the CPU at the same place | 6.6e-7 |
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
use denoise_common::{FNV_START, compare, fnv};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::frame_denoise::{DenoiseMethod, FrameDenoise};
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{Placement, RgbImage, synth};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::bm_dct::{self, HALO_PX, frame_denoise, frame_denoise_core};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// Largest `|gpu - cpu|` (linear, white 1.0) of the frames of `matches_the_cpu_on_every_frame`: twice the first
/// measured value (2.09e-6 on lavapipe: no decision went the other way in 1.9 million samples). Metal gave the
/// same numbers to the last digit (macOS 14 and 15, Mac GPU run 39516017050, 2026-10-21: 2.086e-6, 0 over
/// 1e-4), as the kernel pins its arithmetic, so one set of bounds holds for every adapter.
const BOUND: f32 = 4.2e-6;
/// The share of those samples over 1e-4: twice the first measured value (0, on lavapipe and on Metal).
const OVER_1E4: f64 = 0.0;
/// A frame on which a decision went the other way (one of the frames of
/// `tiles_equal_the_whole_frame_bit_for_bit`, luma only, R 0): twice the first measured values (largest
/// 1.68e-3, 204 of 99 381 samples over 1e-4, on lavapipe and on Metal alike). A flip moves the reference's 8
/// blocks by up to about `tau / 512` in the encoding.
const FLIP_LARGEST: f32 = 3.4e-3;
/// See [`FLIP_LARGEST`].
const FLIP_OVER_1E4: f64 = 4.1e-3;
/// Largest `|gpu - cpu|` when the method changes nothing (the input encoded and decoded): twice the first
/// measured value (2.38e-7, two ulp of the values near 1; Metal the same).
const ROUND_TRIP_BOUND: f32 = 4.8e-7;

/// The bounds of a frame on which a decision may go the other way: `(largest, share over 1e-4)`.
fn flip_bounds(_dev: &TestGpu) -> (f32, f64) {
    (FLIP_LARGEST, FLIP_OVER_1E4)
}

/// The bounds of the adapter: `(largest, share over 1e-4)`.
fn bounds(_dev: &TestGpu) -> (f32, f64) {
    (BOUND, OVER_1E4)
}

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

fn stage(luma: f64, chroma: f64, radius: usize, blend: f64) -> FrameDenoise {
    FrameDenoise {
        method: DenoiseMethod::BmDct,
        luma,
        chroma,
        radius,
        blend,
    }
}

/// Balanced linear camera RGB as R6 has it: a natural scene placed around middle grey of the raw scale (0.083)
/// with gains, so that the highlights go above 1 and the noise takes the shadows below 0; Gaussian noise of
/// `sigma` (linear) on every channel and three times that on red and blue.
fn linear_scene(w: usize, h: usize, seed: u64, sigma: f32) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    let n = img.plane_len();
    let gains = [1.9f32, 1.0, 1.5];
    let mut rng = synth::Rng::new(seed.wrapping_add(77));
    for (c, g) in gains.iter().enumerate() {
        for v in &mut img.data[c * n..(c + 1) * n] {
            let base = 0.5 * *v * *v * g;
            let k = if c == 1 { 1.0 } else { 3.0 };
            *v = base + k * sigma * rng.gaussian();
        }
    }
    img
}

fn run_gpu(dev: &TestGpu, img: &RgbImage, fd: FrameDenoise) -> RgbImage {
    let img = img.clone();
    on_device(dev, move |ctx| frame_denoise(ctx, &img, &fd))
}

fn run_gpu_core(dev: &TestGpu, img: &RgbImage, core: Rect, fd: FrameDenoise) -> RgbImage {
    let img = img.clone();
    on_device(dev, move |ctx| frame_denoise_core(ctx, &img, core, &fd))
}

#[test]
fn matches_the_cpu_on_every_frame() {
    let Some(dev) = f32_device() else { return };
    // (width, height, sigma, L, C, R, blend)
    let mut cases: Vec<(usize, usize, f32, f64, f64, usize, f64)> = vec![
        (8, 8, 0.004, 10.0, 40.0, 1, 0.0),
        (9, 13, 0.004, 10.0, 40.0, 1, 15.0),
        (13, 9, 0.006, 15.0, 60.0, 0, 15.0),
        (17, 31, 0.004, 8.0, 30.0, 2, 0.0),
        (31, 17, 0.004, 8.0, 30.0, 3, 15.0),
        (37, 29, 0.003, 8.0, 0.0, 1, 0.0),
        (29, 37, 0.003, 0.0, 30.0, 1, 0.0),
        (53, 61, 0.005, 12.0, 45.0, 1, 15.0),
        (64, 48, 0.004, 9.0, 100.0, 1, 15.0),
        (101, 77, 0.004, 9.0, 35.0, 0, 15.0),
        (101, 77, 0.004, 9.0, 35.0, 1, 15.0),
        (101, 77, 0.004, 9.0, 35.0, 2, 15.0),
        (101, 77, 0.004, 9.0, 35.0, 3, 15.0),
        (131, 97, 0.002, 4.0, 15.0, 1, 0.0),
        (131, 97, 0.008, 20.0, 80.0, 1, 15.0),
        (97, 131, 0.004, 9.0, 0.0, 2, 15.0),
        (97, 131, 0.004, 0.0, 35.0, 2, 15.0),
        (160, 121, 0.004, 50.0, 60.0, 1, 0.0),
        (199, 151, 0.004, 9.0, 35.0, 1, 15.0),
        (301, 203, 0.004, 9.0, 35.0, 1, 15.0),
        (301, 203, 0.006, 14.0, 100.0, 0, 30.0),
        (203, 301, 0.004, 9.0, 35.0, 3, 15.0),
    ];
    // Luma only and chroma only on the larger sizes, both radii ends.
    for (w, h, r) in [(211, 157, 0), (157, 211, 3), (250, 180, 1), (180, 250, 2)] {
        cases.push((w, h, 0.004, 9.0, 0.0, r, 15.0));
        cases.push((w, h, 0.004, 0.0, 35.0, r, 0.0));
    }
    let (mut worst, mut over4, mut over3, mut total) = (0.0f32, 0usize, 0usize, 0usize);
    for (k, (w, h, sigma, luma, chroma, radius, blend)) in cases.into_iter().enumerate() {
        let img = linear_scene(w, h, 100 + k as u64, sigma);
        let fd = stage(luma, chroma, radius, blend);
        let want = fd.apply(&img);
        let got = run_gpu(dev, &img, fd);
        assert_eq!((got.width, got.height, got.place), (w, h, img.place));
        assert_all_finite(&format!("case {k} {w}x{h}"), &got.data);
        let e = compare(&got.data, &want.data);
        let o4 = denoise_common::Err::over(&got.data, &want.data, 1e-4);
        let o3 = denoise_common::Err::over(&got.data, &want.data, 1e-3);
        eprintln!(
            "[R6] method A case {k} {w}x{h} L {luma} C {chroma} R {radius} B {blend}: max {:.3e} at {} mean {:.2e}, over 1e-4: {o4}, over 1e-3: {o3}",
            e.max, e.at, e.mean
        );
        worst = worst.max(e.max);
        over4 += o4;
        over3 += o3;
        total += got.data.len();
        assert!(e.max <= bounds(dev).0, "case {k} {w}x{h}: {:e}", e.max);
    }
    denoise_common::ci_note(
        "method A",
        &format!(
            "method A on {}: worst {worst:.3e} over {total} samples, {over4} over 1e-4 ({:.2e}), {over3} over 1e-3",
            dev.gpu.info().name,
            over4 as f64 / total as f64
        ),
    );
    assert!(
        (over4 as f64) <= bounds(dev).1 * total as f64,
        "{over4} of {total} samples over 1e-4"
    );
}

#[test]
fn strengths_that_filter_nothing_and_frames_under_8_px_give_the_cpus_round_trip() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    for (k, (w, h, fd)) in [
        (40, 30, stage(0.0, 0.0, 1, 15.0)),
        (40, 30, stage(5e-6, 5e-6, 1, 0.0)),
        (7, 20, stage(10.0, 40.0, 1, 15.0)),
        (20, 7, stage(10.0, 40.0, 1, 0.0)),
        (3, 3, stage(10.0, 40.0, 1, 0.0)),
        (1, 1, stage(10.0, 40.0, 1, 0.0)),
    ]
    .into_iter()
    .enumerate()
    {
        let img = linear_scene(w, h, 40 + k as u64, 0.01);
        let want = fd.apply(&img);
        let got = run_gpu(dev, &img, fd);
        let e = compare(&got.data, &want.data);
        // The CPU's stage is its input encoded and decoded: within an ulp or two of the input itself.
        let to_input = compare(&want.data, &img.data);
        eprintln!(
            "[R6] method A identity case {k} {w}x{h}: against the CPU {:.3e}, the CPU against its input {:.3e}",
            e.max, to_input.max
        );
        worst = worst.max(e.max);
    }
    denoise_common::ci_note(
        "method A identity",
        &format!(
            "method A, nothing to filter, on {}: worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
    assert!(worst <= ROUND_TRIP_BOUND, "{worst:e}");
}

/// The window of the frame that carries `core` and the halo of the method, and the core inside it.
fn window_of(img: &RgbImage, core: Rect) -> (RgbImage, Rect) {
    let frame = Rect::whole(img.width, img.height);
    let rect = core.grow(Halo::uniform(HALO_PX), frame);
    (img.extract(rect), core.relative_to(&rect))
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (211usize, 157usize);
    let img = linear_scene(w, h, 21, 0.004);
    for fd in [
        stage(9.0, 35.0, 1, 15.0),
        stage(9.0, 0.0, 0, 0.0),
        stage(0.0, 35.0, 3, 15.0),
        stage(14.0, 60.0, 2, 0.0),
    ] {
        let whole =
            assert_tiles_equal_whole(&format!("method A {fd:?}"), w, h, 3, &[48, 100], |core| {
                let (sub, rel) = window_of(&img, core);
                run_gpu_core(dev, &sub, rel, fd).data
            });
        // Windows at odd origins, one of them reaching the right and the bottom edge, and the whole.
        let whole_img = RgbImage::from_planar(w, h, whole.clone()).expect("planes");
        for core in [
            Rect::new(23, 31, 57, 49),
            Rect::new(61, 17, 77, 63),
            Rect::new(131, 97, w - 131, h - 97),
            Rect::new(3, 101, 101, h - 101),
            Rect::new(150, 0, w - 150, 41),
        ] {
            let (sub, rel) = window_of(&img, core);
            let got = run_gpu_core(dev, &sub, rel, fd);
            assert_eq!(got.place, img.place.offset(core.x, core.y));
            assert_bit_equal(
                &format!("method A {fd:?}: window {core:?}"),
                &got.data,
                &whole_img.extract(core).data,
            );
        }
        // And the whole frame is the CPU's within the bound of a decision that went the other way.
        let want = fd.apply(&img);
        let e = compare(&whole, &want.data);
        let o4 = denoise_common::Err::over(&whole, &want.data, 1e-4);
        let o5 = denoise_common::Err::over(&whole, &want.data, 1e-5);
        eprintln!(
            "[R6] method A tiles {fd:?}: whole frame against the CPU {:.3e}, {o5} over 1e-5, {o4} over 1e-4 of {}",
            e.max,
            whole.len()
        );
        assert!(e.max <= flip_bounds(dev).0, "{:e}", e.max);
        assert!(
            (o4 as f64) <= flip_bounds(dev).1 * whole.len() as f64,
            "{o4} over 1e-4"
        );
    }
}

#[test]
fn a_halo_one_pixel_short_changes_a_pixel() {
    // The 23 px of reading A-d are needed: with 22 the window computes a reference the whole frame computes
    // differently near the edge of the core, or misses one (the CPU's own test of the halo, on the device).
    let Some(dev) = f32_device() else { return };
    let (w, h) = (160usize, 120usize);
    let img = linear_scene(w, h, 5, 0.006);
    let fd = stage(12.0, 50.0, 1, 0.0);
    let whole = run_gpu(dev, &img, fd);
    let mut differs = false;
    for core in [
        Rect::new(40, 30, 60, 50),
        Rect::new(47, 35, 61, 51),
        Rect::new(51, 41, 50, 40),
    ] {
        let rect = core.grow(Halo::uniform(HALO_PX - 1), Rect::whole(w, h));
        let got = run_gpu_core(dev, &img.extract(rect), core.relative_to(&rect), fd);
        differs |= got.data != whole.extract(core).data;
    }
    assert!(
        differs,
        "a halo of 22 px gave the whole frame's pixels in every window"
    );
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = linear_scene(83, 67, 5, 0.005);
    let small = linear_scene(6, 9, 6, 0.005);
    assert_poison_neutral("method A", |gpu| {
        let (img, small) = (img.clone(), small.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut out = Vec::new();
            for fd in [
                stage(10.0, 40.0, 1, 15.0),
                stage(10.0, 0.0, 3, 0.0),
                stage(0.0, 40.0, 0, 15.0),
                stage(0.0, 0.0, 1, 15.0),
            ] {
                out.extend(frame_denoise(ctx, &img, &fd)?.data);
                // A window's core, so the recycled buffers are read at an offset.
                out.extend(frame_denoise_core(ctx, &img, Rect::new(5, 7, 61, 43), &fd)?.data);
            }
            out.extend(frame_denoise(ctx, &small, &stage(10.0, 40.0, 1, 15.0))?.data);
            Ok(out)
        })
        .expect("run")
    });
}

#[test]
fn positions_are_the_frames_up_to_the_far_corner_of_a_50_mp_frame() {
    // The grid, the candidates and the extra column and row are positions of the frame (readings G4, A-a): a
    // window placed at the origin, in the middle and at the far corner of an 8658 x 5772 frame gets the CPU's
    // pixels at the same place.
    let Some(dev) = f32_device() else { return };
    let (fw, fh) = (8658usize, 5772usize);
    let (w, h) = (121usize, 93usize);
    let fd = stage(9.0, 35.0, 1, 15.0);
    let mut worst = 0.0f32;
    for (k, (x0, y0)) in [
        (0, 0),
        (4001, 2999),
        (fw - w, fh - h),
        (fw - w - 3, fh - h - 5),
    ]
    .into_iter()
    .enumerate()
    {
        let mut img = linear_scene(w, h, 60 + k as u64, 0.004);
        img.place = Placement {
            x0,
            y0,
            frame_width: fw,
            frame_height: fh,
        };
        let want = fd.apply(&img);
        let got = run_gpu(dev, &img, fd);
        assert_eq!(got.place, img.place);
        let e = compare(&got.data, &want.data);
        eprintln!(
            "[R6] method A at ({x0}, {y0}) of {fw} x {fh}: {:.3e}",
            e.max
        );
        worst = worst.max(e.max);
        // The CPU's own run at another place is another picture (the grid moved), so the comparison sees
        // the place.
        let mut moved = img.clone();
        moved.place = Placement::whole(w, h);
        if (x0, y0) != (0, 0) {
            assert!(fd.apply(&moved).max_abs_diff(&want) > 1e-4, "({x0}, {y0})");
        }
    }
    denoise_common::ci_note(
        "method A far corner",
        &format!(
            "method A at the far corner of 8658 x 5772 on {}: worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
    assert!(worst <= flip_bounds(dev).0, "{worst:e}");
}

#[test]
fn the_stage_runs_in_one_pass_list_with_the_dispatches_it_announces() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (130usize, 90usize);
    let img = linear_scene(w, h, 31, 0.004);
    let core = Rect::new(23, 23, 84, 44);
    let fd = stage(9.0, 35.0, 1, 15.0);
    let (a, subs, recorded) = {
        let img = img.clone();
        on_device(dev, move |ctx| {
            let before = ctx.counters();
            let input = GpuImage::upload_rgb(ctx, "in", &img)?;
            let mut passes = PassList::new(ctx, "method A");
            let out = bm_dct::encode_frame_denoise(&mut passes, &input, core, &fd)?;
            let recorded = passes.dispatch_count();
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = out.take(&done, read)?;
            Ok((data, ctx.counters().since(before).submissions, recorded))
        })
    };
    let b = run_gpu_core(dev, &img, core, fd);
    assert_bit_equal("device entry point against the host wrapper", &a, &b.data);
    assert_eq!(subs, 1, "one submission");
    assert_eq!(recorded as usize, bm_dct::dispatches(w, h, img.place, &fd));
    // R = 1 on a 130 x 90 frame: 16 regular passes; 122 and 82 are not multiples of 6, so 4 + 4 + 1 extra.
    assert_eq!(recorded, 2 + 16 + 9);
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut hash = FNV_START;
        for (w, h, seed, fd) in [
            (97usize, 61usize, 1u64, stage(9.0, 35.0, 1, 15.0)),
            (61, 97, 2, stage(14.0, 0.0, 3, 0.0)),
        ] {
            let img = linear_scene(w, h, seed, 0.005);
            hash = fnv(hash, &run_gpu(dev, &img, fd).data);
        }
        hash
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    assert_max_binding_bytes(
        "method A",
        Rect::new(0, 0, side, side),
        bm_dct::max_binding_bytes,
    );
}

#[test]
fn bad_inputs_are_refused_with_the_reason() {
    let Some(dev) = f32_device() else { return };
    let img = linear_scene(30, 20, 1, 0.004);
    let ok = stage(9.0, 35.0, 1, 15.0);
    let run = |core: Rect, fd: FrameDenoise, place: Option<Placement>| {
        let mut img = img.clone();
        if let Some(p) = place {
            img.place = p;
        }
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                frame_denoise_core(ctx, &img, core, &fd)
            })
    };
    let whole = Rect::whole(30, 20);
    assert!(run(whole, ok, None).is_ok());
    for (what, result) in [
        (
            "method B",
            run(
                whole,
                FrameDenoise {
                    method: DenoiseMethod::NlBayesRgb,
                    ..ok
                },
                None,
            ),
        ),
        (
            "the bilateral",
            run(
                whole,
                FrameDenoise {
                    method: DenoiseMethod::Bilateral,
                    ..ok
                },
                None,
            ),
        ),
        (
            "a NaN strength",
            run(whole, stage(f64::NAN, 35.0, 1, 0.0), None),
        ),
        ("core outside", run(Rect::new(10, 10, 30, 20), ok, None)),
        ("empty core", run(Rect::new(0, 0, 0, 5), ok, None)),
        (
            "a window outside its frame",
            run(
                whole,
                ok,
                Some(Placement {
                    x0: 10,
                    y0: 0,
                    frame_width: 35,
                    frame_height: 20,
                }),
            ),
        ),
    ] {
        let err = result.expect_err(what);
        assert!(matches!(err, GpuError::InvalidInput(_)), "{what}: {err}");
    }
}

/// Timing on a 2048 x 1365 frame on this adapter (meaningless for Metal; the number goes into the docs):
/// `cargo test -p ym-gpu --release --test gpu_bm_dct -- --ignored --nocapture`. Prints the device time of
/// the stage on the whole frame and in 1024 px tiles with the 23 px halo, and the dispatches per tile.
#[test]
#[ignore = "timing; run in release with --ignored --nocapture"]
fn timing_on_a_2048_by_1365_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let img = linear_scene(w, h, 9, 0.004);
    for (luma, chroma) in [(9.0, 35.0), (9.0, 0.0)] {
        let fd = stage(luma, chroma, 1, 15.0);
        let input = img.clone();
        let (whole_ms, tiles_ms, per_tile) = on_device(dev, move |ctx| {
            // Warm-up: the pipelines compiled.
            let small = GpuImage::upload_rgb(ctx, "warm", &input.extract(Rect::new(0, 0, 64, 64)))?;
            let mut passes = PassList::new(ctx, "warm");
            let _ = bm_dct::encode_frame_denoise(&mut passes, &small, Rect::whole(64, 64), &fd)?;
            passes.submit()?.finish()?;
            let gpu_in = GpuImage::upload_rgb(ctx, "in", &input)?;
            let t = std::time::Instant::now();
            let mut passes = PassList::new(ctx, "whole");
            let out = bm_dct::encode_frame_denoise(&mut passes, &gpu_in, Rect::whole(w, h), &fd)?;
            passes.submit()?.finish()?;
            let whole_ms = t.elapsed().as_secs_f64() * 1e3;
            drop(out);
            let mut per_tile = Vec::new();
            let t = std::time::Instant::now();
            for core in ym_gpu::cpu::tiling::tile_grid(w, h, 1024, 1024, 2) {
                let rect = core.grow(Halo::uniform(HALO_PX), Rect::whole(w, h));
                let sub = GpuImage::upload_rgb(ctx, "tile", &input.extract(rect))?;
                let mut passes = PassList::new(ctx, "tile");
                let _ =
                    bm_dct::encode_frame_denoise(&mut passes, &sub, core.relative_to(&rect), &fd)?;
                per_tile.push((core, passes.dispatch_count()));
                passes.submit()?.finish()?;
            }
            Ok((whole_ms, t.elapsed().as_secs_f64() * 1e3, per_tile))
        });
        println!(
            "method A, L {luma} C {chroma} R 1, {w} x {h} on {}: whole frame {whole_ms:.0} ms ({:.0} ms/MP), 1024 px tiles {tiles_ms:.0} ms",
            dev.gpu.info().name,
            whole_ms / (w * h) as f64 * 1e6
        );
        for (core, n) in per_tile {
            println!("  tile {core:?}: {n} dispatches");
        }
    }
}

/// Noise-free synthetic content on which distances and coefficients tie in exact arithmetic: a ramp, flat
/// patches of 16 px (a block that straddles an edge has mirror-image candidates whose distances are the same
/// terms summed in the opposite order), and a scene pushed into the clip with noise only where it is not
/// clipped.
fn tie_prone_scene(w: usize, h: usize, kind: usize, seed: u64) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = synth::Rng::new(seed);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let rgb = match kind {
                0 => [0.02 + 0.5 * fx, 0.03 + 0.3 * fy, 0.05 + 0.2 * (fx + fy)],
                1 => {
                    let v = ((x / 16 + 3 * (y / 16)) % 7) as f32 * 0.05 + 0.02;
                    [v * 1.2, v, v * 0.8]
                }
                _ => {
                    let n = 0.004 * rng.gaussian();
                    let v = (fx * 3.0 + (fy * 7.0).sin()).min(1.0);
                    let c = |k: f32| if v * k >= 1.0 { 1.0 } else { v * k + n };
                    [c(1.4), c(1.0), c(0.7)]
                }
            };
            img.set_pixel(x, y, rgb);
        }
    }
    img
}

/// Tie-prone content, all cases together: the share of the samples over 1e-4 and the largest difference,
/// twice the first measured values. The CPU breaks a tie of two candidates whose distances are the same terms
/// in another order by the rounding of its `f64` sums, the device by that of its `f32` sums, so the two take
/// different (equally valid) groups there; a raw photo always has noise, which leaves no such ties
/// (`matches_the_cpu_on_every_frame`).
/// First measured on lavapipe: 35 044 of 611 901 samples over 1e-4 (5.73e-2, nearly all on the flat patches
/// at R 1, whose blocks straddle the edges), largest 7.3e-3.
const TIE_PRONE_OVER_1E4: f64 = 0.115;
/// See [`TIE_PRONE_OVER_1E4`].
const TIE_PRONE_LARGEST: f32 = 1.46e-2;

#[test]
fn tie_prone_content_differs_only_by_tie_flips() {
    let Some(dev) = f32_device() else { return };
    let (mut worst, mut over4, mut total) = (0.0f32, 0usize, 0usize);
    for kind in 0..3 {
        for (k, fd) in [
            stage(9.0, 35.0, 1, 15.0),
            stage(9.0, 0.0, 3, 0.0),
            stage(20.0, 100.0, 0, 0.0),
        ]
        .into_iter()
        .enumerate()
        {
            let img = tie_prone_scene(173, 131, kind, 3 + k as u64);
            let want = fd.apply(&img);
            let got = run_gpu(dev, &img, fd);
            let e = compare(&got.data, &want.data);
            let o4 = denoise_common::Err::over(&got.data, &want.data, 1e-4);
            let o3 = denoise_common::Err::over(&got.data, &want.data, 1e-3);
            eprintln!(
                "[R6] method A tie-prone kind {kind} L {} C {} R {}: max {:.3e} at {}, over 1e-4 {o4}, over 1e-3 {o3}",
                fd.luma, fd.chroma, fd.radius, e.max, e.at
            );
            worst = worst.max(e.max);
            over4 += o4;
            total += got.data.len();
        }
    }
    denoise_common::ci_note(
        "method A tie-prone",
        &format!(
            "method A tie-prone content on {}: worst {worst:.3e}, {over4} of {total} over 1e-4 ({:.2e})",
            dev.gpu.info().name,
            over4 as f64 / total as f64
        ),
    );
    assert!(worst <= TIE_PRONE_LARGEST, "{worst:e}");
    assert!(
        (over4 as f64) <= TIE_PRONE_OVER_1E4 * total as f64,
        "{over4} of {total} over 1e-4"
    );
}
