//! R6's false-colour suppression on the device (`shaders/false_color.wgsl` and R7's `guided.wgsl`,
//! `kernels::false_color`) against its CPU reference `cpu::false_color::{weights, suppress}`.
//!
//! What is asserted, and what was measured on lavapipe (first measured value; the bound is twice that,
//! `docs/17-gpu.md` section 12):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | the gate's weights on noisy natural frames, 2 and 3 px gratings with colour bands, stripes on half a frame, from 9 x 7 to 211 x 157 | max `abs(w_gpu - w_cpu)` at most [`WEIGHT_BOUND`] | 5.07e-7 (0 in the gratings) |
//! | the stage on the same frames, Colour 30 to 100, scalar and plane sigma | max `abs(gpu - cpu)` at most [`BOUND`] | 1.19e-7 over 957 372 samples |
//! | ramps, steps and colour edges | weights 0 and the input bit for bit, as on the CPU | |
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
use denoise_common::{FNV_START, compare, fnv, sigma_plane};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::false_color::{self as cpu_fc, suppress, weights};
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{NoiseSigma, RgbImage, synth};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::bilateral::{SigmaInput, encode_sigma};
use ym_gpu::kernels::false_color::{
    self as gpu_fc, encode_false_color, false_color, false_color_weights, halo_px,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// Largest `|w_gpu - w_cpu|` of the gate on a software adapter: twice the first measured value (5.07e-7 on
/// lavapipe; 0 inside the gratings, where the weight is 1).
const WEIGHT_BOUND: f32 = 1.02e-6;
/// Largest `|gpu - cpu|` of the stage (linear, white 1.0) on a software adapter: twice the first measured value
/// (1.19e-7 on lavapipe over 957 372 samples, one ulp; the guided filter is the CPU's bits).
const BOUND: f32 = 2.4e-7;
/// The same two on a hardware adapter: starting values until the first Metal run prints its numbers
/// (`ci_note`; the gate pins its arithmetic and the guided filter gives the CPU's bits, so Metal should give
/// lavapipe's bits).
const HW_WEIGHT_BOUND: f32 = 1e-4;
/// See [`HW_WEIGHT_BOUND`].
const HW_BOUND: f32 = 1e-5;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

fn software(dev: &TestGpu) -> bool {
    dev.gpu.info().is_software
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

/// Balanced linear camera RGB around middle grey of the raw scale, with noise of `sigma` (three times that on
/// red and blue), as the stage gets it after method A.
fn natural(w: usize, h: usize, seed: u64, sigma: f32) -> RgbImage {
    let mut img = synth::natural(w, h, seed);
    let n = img.plane_len();
    let mut rng = synth::Rng::new(seed + 5);
    for (c, g) in [1.9f32, 1.0, 1.5].iter().enumerate() {
        for v in &mut img.data[c * n..(c + 1) * n] {
            let k = if c == 1 { 1.0 } else { 3.0 };
            *v = 0.5 * *v * *v * g + k * sigma * rng.gaussian();
        }
    }
    img
}

/// A grating of `period` px (neutral stripes of two levels) with a blue and orange false colour in bands of 3
/// rows, as the demosaic leaves it (the CPU unit test's scene), plus a trace of noise.
fn grating(w: usize, h: usize, period: usize, seed: u64) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = synth::Rng::new(seed);
    for y in 0..h {
        for x in 0..w {
            let luma: f32 = if x % period < period / 2 { 0.03 } else { 0.25 };
            let tint = if (y / 3) % 2 == 0 { 1.0 } else { -1.0 };
            let rgb = [luma + 0.04 * tint, luma, luma - 0.04 * tint]
                .map(|v| (v + 0.001 * rng.gaussian()).max(0.0));
            img.set_pixel(x, y, rgb);
        }
    }
    img
}

/// A natural frame with 2 px stripes on its right half (the CPU chain test's scene, before the mosaic).
fn half_striped(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = natural(w, h, seed, 0.002);
    let n = w * h;
    for y in 0..h {
        for x in w / 2..w {
            let v = if x % 2 == 0 { 0.05 } else { 0.6 };
            for c in 0..3 {
                img.data[c * n + y * w + x] = v + if c == 0 { 0.02 } else { 0.0 };
            }
        }
    }
    img
}

/// The frames of the parity tests.
fn frames() -> Vec<(&'static str, RgbImage)> {
    vec![
        ("natural 9x7", natural(9, 7, 1, 0.004)),
        ("natural 64x48", natural(64, 48, 2, 0.004)),
        ("natural noisy 131x97", natural(131, 97, 3, 0.012)),
        ("grating 2 px 48x40", grating(48, 40, 2, 4)),
        ("grating 3 px 61x45", grating(61, 45, 3, 5)),
        ("grating 2 px 211x157", grating(211, 157, 2, 6)),
        ("half striped 140x96", half_striped(140, 96, 7)),
        ("half striped 97x131", half_striped(97, 131, 8)),
    ]
}

#[test]
fn the_weights_match_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut opened = 0usize;
    for (name, img) in frames() {
        let (w, h) = (img.width, img.height);
        let want: Vec<f32> = weights(&img).iter().map(|&v| v as f32).collect();
        let got = {
            let img = img.clone();
            on_device(dev, move |ctx| {
                false_color_weights(ctx, &img, Rect::whole(w, h))
            })
        };
        assert_all_finite(name, &got);
        assert!(got.iter().all(|v| (0.0..=1.0).contains(v)), "{name}");
        let e = compare(&got, &want);
        opened += want.iter().filter(|&&v| v > 0.0).count();
        eprintln!(
            "[R6] false-colour weights {name}: max {:.3e} at {}",
            e.max, e.at
        );
        worst = worst.max(e.max);
    }
    assert!(opened > 10_000, "the gate opened at {opened} pixels only");
    denoise_common::ci_note(
        "false-colour weights",
        &format!(
            "false-colour weights on {}: worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
    let bound = if software(dev) {
        WEIGHT_BOUND
    } else {
        HW_WEIGHT_BOUND
    };
    assert!(worst <= bound, "{worst:e}");
}

/// An owned sigma source for the device closures.
#[derive(Clone)]
enum SigmaOwned {
    Scalar(f32),
    Plane(Vec<f32>),
}

impl SigmaOwned {
    fn as_input(&self) -> SigmaInput<'_> {
        match self {
            SigmaOwned::Scalar(v) => SigmaInput::Scalar(*v),
            SigmaOwned::Plane(p) => SigmaInput::Plane(p),
        }
    }

    fn as_cpu(&self) -> NoiseSigma<'_> {
        match self {
            SigmaOwned::Scalar(v) => NoiseSigma::Scalar(*v),
            SigmaOwned::Plane(p) => NoiseSigma::Plane(p),
        }
    }

    /// The part of the source a window of the frame reads.
    fn cut(&self, frame_w: usize, rect: Rect) -> SigmaOwned {
        match self {
            SigmaOwned::Scalar(v) => SigmaOwned::Scalar(*v),
            SigmaOwned::Plane(p) => {
                let mut out = Vec::with_capacity(rect.area());
                for y in rect.y..rect.y1() {
                    out.extend_from_slice(&p[y * frame_w + rect.x..y * frame_w + rect.x1()]);
                }
                SigmaOwned::Plane(out)
            }
        }
    }
}

fn run_gpu(dev: &TestGpu, img: &RgbImage, core: Rect, sigma: &SigmaOwned, color: f32) -> RgbImage {
    let (img, sigma) = (img.clone(), sigma.clone());
    on_device(dev, move |ctx| {
        false_color(ctx, &img, core, &sigma.as_input(), color)
    })
}

#[test]
fn the_stage_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut changed = 0usize;
    let mut total = 0usize;
    for (k, (name, img)) in frames().into_iter().enumerate() {
        let (w, h) = (img.width, img.height);
        for (j, color) in [30.0f32, 50.0, 60.0, 100.0].into_iter().enumerate() {
            let sigma = if (k + j) % 2 == 0 {
                SigmaOwned::Scalar(0.004)
            } else {
                SigmaOwned::Plane(sigma_plane(w, h, k as u64 + 10))
            };
            let want = suppress(&img, &sigma.as_cpu(), color);
            let got = run_gpu(dev, &img, Rect::whole(w, h), &sigma, color);
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
                "[R6] false colour {name} C {color}: max {:.3e} at {} mean {:.2e}",
                e.max, e.at, e.mean
            );
            worst = worst.max(e.max);
        }
    }
    assert!(
        changed > total / 20,
        "the stage changed {changed} of {total} samples"
    );
    denoise_common::ci_note(
        "false colour",
        &format!(
            "false-colour suppression on {}: worst {worst:.3e} over {total} samples",
            dev.gpu.info().name
        ),
    );
    let bound = if software(dev) { BOUND } else { HW_BOUND };
    assert!(worst <= bound, "{worst:e}");
}

#[test]
fn ramps_steps_and_colour_edges_are_left_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let image = |f: &dyn Fn(usize, usize, usize) -> f32| {
        let (w, h) = (32usize, 24usize);
        let mut data = Vec::with_capacity(3 * w * h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data.push(f(c, x, y));
                }
            }
        }
        RgbImage::from_planar(w, h, data).expect("planes")
    };
    for img in [
        image(&|_, x, _| 0.01 + 0.004 * x as f32),
        image(&|_, x, _| if x < 16 { 0.02 } else { 0.3 }),
        image(&|c, x, _| match (c, x < 16) {
            (0, true) | (2, false) => 0.2,
            _ => 0.05,
        }),
    ] {
        let whole = Rect::whole(img.width, img.height);
        let w = {
            let img = img.clone();
            on_device(dev, move |ctx| false_color_weights(ctx, &img, whole))
        };
        assert!(w.iter().all(|&v| v == 0.0), "{w:?}");
        let got = run_gpu(dev, &img, whole, &SigmaOwned::Scalar(0.002), 50.0);
        assert_bit_equal("the identity", &got.data, &img.data);
        assert_eq!(suppress(&img, &NoiseSigma::Scalar(0.002), 50.0), img);
    }
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (211usize, 157usize);
    let mut img = grating(w, h, 2, 21);
    // Natural content on the left half, so that the gate opens and closes inside the tiles.
    let nat = natural(w, h, 22, 0.004);
    let n = w * h;
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w / 2 {
                img.data[c * n + y * w + x] = nat.data[c * n + y * w + x];
            }
        }
    }
    let frame = Rect::whole(w, h);
    for (color, sigma) in [
        (60.0f32, SigmaOwned::Plane(sigma_plane(w, h, 3))),
        (30.0, SigmaOwned::Scalar(0.003)),
    ] {
        let halo = halo_px(color);
        let render = |core: Rect| {
            let rect = core.grow(Halo::uniform(halo), frame);
            run_gpu(
                dev,
                &img.extract(rect),
                core.relative_to(&rect),
                &sigma.cut(w, rect),
                color,
            )
        };
        let whole = assert_tiles_equal_whole(
            &format!("false colour C {color}"),
            w,
            h,
            3,
            &[40, 96],
            |core| render(core).data,
        );
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
                &format!("false colour C {color}: window {core:?}"),
                &got.data,
                &whole_img.extract(core).data,
            );
        }
        let want = suppress(&img, &sigma.as_cpu(), color);
        let e = compare(&whole, &want.data);
        eprintln!(
            "[R6] false colour tiles C {color}: whole frame against the CPU {:.3e}",
            e.max
        );
        let bound = if software(dev) { BOUND } else { HW_BOUND };
        assert!(e.max <= bound, "{:e}", e.max);
    }
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = half_striped(83, 67, 5);
    let plane = sigma_plane(83, 67, 9);
    assert_poison_neutral("false colour", |gpu| {
        let (img, plane) = (img.clone(), plane.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut out = Vec::new();
            for color in [60.0f32, 25.0] {
                let whole = Rect::whole(83, 67);
                out.extend(false_color(ctx, &img, whole, &SigmaInput::Plane(&plane), color)?.data);
                out.extend(
                    false_color(
                        ctx,
                        &img,
                        Rect::new(5, 7, 61, 43),
                        &SigmaInput::Scalar(0.003),
                        color,
                    )?
                    .data,
                );
                out.extend(false_color_weights(ctx, &img, whole)?);
            }
            Ok(out)
        })
        .expect("run")
    });
}

#[test]
fn the_stage_runs_in_one_pass_list_with_its_eight_dispatches() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (90usize, 60usize);
    let img = half_striped(w, h, 31);
    let core = Rect::new(11, 9, 70, 44);
    let (a, subs, recorded) = {
        let img = img.clone();
        on_device(dev, move |ctx| {
            let before = ctx.counters();
            let input = GpuImage::upload_rgb(ctx, "in", &img)?;
            let mut passes = PassList::new(ctx, "false colour");
            let sigma = encode_sigma(&mut passes, &SigmaInput::Scalar(0.003), Rect::whole(w, h))?;
            let start = passes.dispatch_count();
            let out = encode_false_color(&mut passes, &input, core, &sigma, (0, 0), 60.0)?;
            let recorded = passes.dispatch_count() - start;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = out.take(&done, read)?;
            Ok((data, ctx.counters().since(before).submissions, recorded))
        })
    };
    let b = run_gpu(dev, &img, core, &SigmaOwned::Scalar(0.003), 60.0);
    assert_bit_equal("device entry point against the host wrapper", &a, &b.data);
    assert_eq!(subs, 1, "one submission");
    assert_eq!(recorded as usize, gpu_fc::DISPATCHES);
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut hash = FNV_START;
        for (img, color) in [
            (half_striped(97, 61, 1), 60.0f32),
            (grating(61, 97, 3, 2), 35.0),
        ] {
            let whole = Rect::whole(img.width, img.height);
            hash = fnv(
                hash,
                &run_gpu(dev, &img, whole, &SigmaOwned::Scalar(0.004), color).data,
            );
        }
        hash
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    assert_max_binding_bytes(
        "false colour",
        Rect::new(0, 0, side, side),
        gpu_fc::max_binding_bytes,
    );
}

#[test]
fn bad_inputs_are_refused_with_the_reason_and_colour_zero_is_the_identity() {
    let Some(dev) = f32_device() else { return };
    let img = natural(30, 20, 1, 0.004);
    let whole = Rect::whole(30, 20);
    assert_eq!(
        run_gpu(dev, &img, whole, &SigmaOwned::Scalar(0.004), 0.0).data,
        img.data
    );
    let run = |core: Rect, color: f32, sigma: SigmaOwned| {
        let img = img.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                false_color(ctx, &img, core, &sigma.as_input(), color)
            })
    };
    assert!(run(whole, 50.0, SigmaOwned::Scalar(0.004)).is_ok());
    for (what, result) in [
        (
            "a NaN colour",
            run(whole, f32::NAN, SigmaOwned::Scalar(0.004)),
        ),
        (
            "core outside",
            run(Rect::new(10, 10, 30, 20), 50.0, SigmaOwned::Scalar(0.004)),
        ),
        (
            "empty core",
            run(Rect::new(0, 0, 0, 5), 50.0, SigmaOwned::Scalar(0.004)),
        ),
        (
            "short sigma plane",
            run(whole, 50.0, SigmaOwned::Plane(vec![0.01; 19])),
        ),
    ] {
        let err = result.expect_err(what);
        assert!(matches!(err, GpuError::InvalidInput(_)), "{what}: {err}");
    }
    assert_eq!(cpu_fc::GATE_HALO_PX, 5);
}

/// Timing on a 2048 x 1365 frame on this adapter (meaningless for Metal; the number goes into the docs):
/// `cargo test -p ym-gpu --test gpu_false_color timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn timing_on_a_2048_by_1365_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let img = half_striped(w, h, 9);
    let ms = on_device(dev, move |ctx| {
        let small = GpuImage::upload_rgb(ctx, "warm", &img.extract(Rect::new(0, 0, 64, 64)))?;
        let mut passes = PassList::new(ctx, "warm");
        let sigma = encode_sigma(&mut passes, &SigmaInput::Scalar(0.003), Rect::whole(64, 64))?;
        let _ = encode_false_color(
            &mut passes,
            &small,
            Rect::whole(64, 64),
            &sigma,
            (0, 0),
            60.0,
        )?;
        passes.submit()?.finish()?;
        let input = GpuImage::upload_rgb(ctx, "in", &img)?;
        let mut passes = PassList::new(ctx, "sigma");
        let sigma = encode_sigma(&mut passes, &SigmaInput::Scalar(0.003), Rect::whole(w, h))?;
        passes.submit()?.finish()?;
        let t = std::time::Instant::now();
        let mut passes = PassList::new(ctx, "whole");
        let _ = encode_false_color(&mut passes, &input, Rect::whole(w, h), &sigma, (0, 0), 60.0)?;
        passes.submit()?.finish()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    });
    println!(
        "false colour, C 60, {w} x {h} on {}: {ms:.0} ms ({:.0} ms/MP), {} dispatches",
        dev.gpu.info().name,
        ms / (w * h) as f64 * 1e6,
        gpu_fc::DISPATCHES
    );
}
