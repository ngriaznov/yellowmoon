//! G-06: the R7 kernels, the colour guided filter (`shaders/guided.wgsl`, `kernels::guided`) and the
//! detail re-injection (`shaders/detail.wgsl`, `kernels::detail`), against their CPU references
//! `cpu::detail::{chroma_guided_whole, inject_detail}`, and the R6 denoise to R7 sequence of one tile
//! with its pre-denoise copy.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/gpu-plan.md` X-16):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | guided filter, 16 frames from 1 x 1 to 200 x 150, C = 10 to 100 (radius 2 to 8), scalar and plane sigma (zeros included) | max `abs(gpu - cpu)` at most [`GUIDED_BOUND`] (software; hardware [`HW_BOUND`]) | 0: the CPU's bits |
//! | bright flat noisy scenes (level 0.2 to 0.9, noise 0.002 to 0.01, sigma 0 to 0.01): the variance cancels most there | the same bound | 0 (plain f32 sums gave 6.3e-6, and 2.3e-5 without regularisation) |
//! | detail injection, beta 0.0375 to 0.6 | max at most [`INJECT_BOUND`] | 0 |
//! | tiles of the guided filter (two sizes, the stage's 2 r of context) | bit-identical to the whole frame | 0 differing |
//! | denoise, guided filter, injection chained on the device (a tile with the halos of the stages) | at most [`SEQUENCE_BOUND`] against the CPU stage sequence | 6.6e-7 |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | `LP_NUM_THREADS=1` against the default | bit-identical | |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`);
//! the Metal runners run the same file (`mac-gpu.yml`).

mod common;
#[path = "g06/mod.rs"]
mod g06;

use common::exact::{
    assert_all_finite, assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole,
};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use g06::{
    FNV_START, bright_flat_scene, chroma_noisy_scene, compare, fnv, noisy_scene, sigma_plane,
};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::denoise::{BilateralParams, bilateral_whole};
use ym_gpu::cpu::detail::{chroma_guided_whole, inject_detail as cpu_inject};
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{LUMA, NoiseSigma, RgbImage};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::bilateral::{SigmaInput, encode_bilateral, encode_sigma};
use ym_gpu::kernels::detail::{
    self as gpu_detail, encode_classical_luma, encode_inject_detail, inject_detail as gpu_inject,
};
use ym_gpu::kernels::guided::{
    self as gpu_guided, MAX_RADIUS_PX, chroma_guided as gpu_chroma_guided, encode_guided, radius_px,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The guided filter, software adapter. The first measured value is 0 (the sums are exact and the
/// CPU's stored f32 values are reproduced, so all 344,166 samples of the cases below and the 960,000
/// of the bright flat scenes are the CPU's bits): the bound is two ulp at white, the rounding of one
/// operation more than the model has.
const GUIDED_BOUND: f32 = 2.4e-7;
/// The detail injection, software adapter. The first measured value is 0: one ulp at white.
const INJECT_BOUND: f32 = 1.2e-7;
/// The denoise to injection sequence of a tile, software adapter: twice the first measured value
/// (6.6e-7, the `exp` of the bilateral filter).
const SEQUENCE_BOUND: f32 = 1.4e-6;
/// Hardware adapters, for the tests that print what they measured on Metal (macos-14 and macos-15, run 38694506557,
/// G-09): the guided filter and the detail injection gave the CPU's bits there (worst 0), so the hardware bound is the
/// software one, two ulp and one ulp at white. Metal contracts and re-associates sums and has its own division, which
/// the kernels hide behind exact sums.
///
/// For the tests whose Metal value was not printed yet (the bright flat scenes, the tiles and the denoise-to-injection
/// sequence now print it): the plan's starting value for neighbourhood stages (`docs/gpu-plan.md` section 7), to be
/// lowered to twice what the next Metal run prints.
const HW_BOUND: f32 = 1e-5;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// The bound of a test whose Metal value was printed: the software bound on both kinds of adapter.
fn bound(dev: &TestGpu, software: f32) -> f32 {
    let _ = dev;
    software
}

/// The bound of a test that has not printed a Metal value yet.
fn bound_unmeasured(dev: &TestGpu, software: f32) -> f32 {
    if dev.gpu.info().is_software {
        software
    } else {
        HW_BOUND
    }
}

fn on_device<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("the device job")
}

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

    fn plane(&self, len: usize) -> Vec<f32> {
        match self {
            SigmaOwned::Scalar(v) => vec![*v; len],
            SigmaOwned::Plane(p) => p.clone(),
        }
    }
}

fn run_guided(
    dev: &TestGpu,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaOwned,
    color: f32,
) -> RgbImage {
    let (img, sigma) = (img.clone(), sigma.clone());
    on_device(dev, move |ctx| {
        gpu_chroma_guided(ctx, &img, core, &sigma.as_input(), color)
    })
}

#[test]
fn guided_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut total = 0usize;
    // (width, height, chroma noise, sigma, colour strength)
    let cases: Vec<(usize, usize, f32, SigmaOwned, f32)> = vec![
        (1, 1, 0.03, SigmaOwned::Scalar(0.03), 50.0),
        (2, 3, 0.03, SigmaOwned::Scalar(0.03), 50.0),
        (5, 7, 0.03, SigmaOwned::Scalar(0.02), 100.0),
        (9, 17, 0.03, SigmaOwned::Scalar(0.03), 10.0),
        (33, 20, 0.03, SigmaOwned::Scalar(0.03), 25.0),
        (
            48,
            40,
            0.03,
            SigmaOwned::Plane(sigma_plane(48, 40, 1)),
            50.0,
        ),
        (64, 48, 0.01, SigmaOwned::Scalar(0.01), 75.0),
        (64, 48, 0.003, SigmaOwned::Scalar(0.003), 100.0),
        (
            80,
            60,
            0.03,
            SigmaOwned::Plane(sigma_plane(80, 60, 2)),
            100.0,
        ),
        (131, 97, 0.03, SigmaOwned::Scalar(0.03), 50.0),
        (
            131,
            97,
            0.01,
            SigmaOwned::Plane(sigma_plane(131, 97, 3)),
            25.0,
        ),
        (97, 131, 0.03, SigmaOwned::Scalar(0.02), 60.0),
        (11, 131, 0.03, SigmaOwned::Scalar(0.03), 50.0),
        (131, 11, 0.03, SigmaOwned::Scalar(0.03), 100.0),
        (
            200,
            150,
            0.03,
            SigmaOwned::Plane(sigma_plane(200, 150, 4)),
            50.0,
        ),
        (200, 150, 0.03, SigmaOwned::Scalar(0.0), 50.0),
    ];
    for (i, (w, h, noise, sigma, color)) in cases.into_iter().enumerate() {
        let img = chroma_noisy_scene(w, h, 40 + i as u64, noise);
        let plane = sigma.plane(w * h);
        let want = chroma_guided_whole(&img, &NoiseSigma::Plane(&plane), 0, color);
        let got = run_guided(dev, &img, Rect::whole(w, h), &sigma, color);
        assert_all_finite(&format!("case {i} {w}x{h}"), &got.data);
        let e = compare(&got.data, &want.data);
        eprintln!(
            "[G-06] guided case {i} {w}x{h} C {color} (radius {}): max {:.3e} at {} mean {:.2e}",
            radius_px(color),
            e.max,
            e.at,
            e.mean
        );
        worst = worst.max(e.max);
        total += got.data.len();
        let limit = bound(dev, GUIDED_BOUND);
        assert!(e.max <= limit, "case {i} {w}x{h}: {:e} > {limit:e}", e.max);
    }
    g06::ci_note(
        "guided",
        &format!(
            "guided filter on {}: worst {worst:.3e} over {total} samples",
            dev.gpu.info().name
        ),
    );
}

#[test]
fn guided_on_bright_flat_scenes_and_without_regularisation() {
    // The variance of a bright noisy flat area is a difference of two numbers four orders larger than
    // it, where f32 sums and f64 sums rounded per plane part ways most (plain f32 sums gave 6.3e-6 and,
    // with sigma 0, 2.3e-5); the exact window sums of the kernel give the CPU's bits.
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    for (i, (level, sigma, color, sig_in)) in [
        (0.9f32, 0.003f32, 50.0f32, 0.003f32),
        (0.9, 0.003, 100.0, 0.003),
        (0.5, 0.01, 25.0, 0.01),
        (0.9, 0.01, 50.0, 0.0),
        (0.2, 0.002, 50.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let (w, h) = (160usize, 120usize);
        let img = bright_flat_scene(w, h, 90 + i as u64, level, sigma);
        let plane = vec![sig_in; w * h];
        let want = chroma_guided_whole(&img, &NoiseSigma::Plane(&plane), 0, color);
        let got = run_guided(
            dev,
            &img,
            Rect::whole(w, h),
            &SigmaOwned::Scalar(sig_in),
            color,
        );
        let e = compare(&got.data, &want.data);
        eprintln!(
            "[G-06] guided bright flat {i}: level {level} noise {sigma} C {color} sigma {sig_in}: max {:.3e} mean {:.2e}",
            e.max, e.mean
        );
        worst = worst.max(e.max);
        assert!(e.max <= bound_unmeasured(dev, GUIDED_BOUND), "{:e}", e.max);
    }
    g06::ci_note(
        "guided bright flat",
        &format!(
            "guided filter, bright flat scenes on {}: worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
}

#[test]
fn guided_leaves_the_luma_alone_and_a_zero_strength_is_a_copy() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (70usize, 50usize);
    let img = chroma_noisy_scene(w, h, 7, 0.03);
    let got = run_guided(
        dev,
        &img,
        Rect::whole(w, h),
        &SigmaOwned::Scalar(0.03),
        50.0,
    );
    let (a, b) = (img.luma_plane(), got.luma_plane());
    let e = compare(&b, &a);
    eprintln!("[G-06] guided luma change: max {:.3e}", e.max);
    assert!(e.max < 2e-6, "the luma changed by {:e}", e.max);
    // The chroma noise went down.
    let n = img.plane_len();
    let chroma_sd = |im: &RgbImage| {
        let y = im.luma_plane();
        let s: f64 = im.data[..n]
            .iter()
            .zip(&y)
            .map(|(r, y)| f64::from(r - y).powi(2))
            .sum();
        (s / n as f64).sqrt()
    };
    assert!(chroma_sd(&got) < 0.8 * chroma_sd(&img));
    // Zero strength: the host wrapper returns the core, and the entry point refuses to run.
    let core = Rect::new(3, 4, 40, 30);
    let copy = run_guided(dev, &img, core, &SigmaOwned::Scalar(0.03), 0.0);
    assert_bit_equal(
        "a zero colour strength",
        &copy.data,
        &img.extract(core).data,
    );
    let refused = dev.gpu.run(JobSpec::new(Priority::Viewport), {
        let img = img.clone();
        move |ctx, _| {
            let input = GpuImage::upload_rgb(ctx, "in", &img)?;
            let mut passes = PassList::new(ctx, "guided");
            let sg = encode_sigma(&mut passes, &SigmaInput::Scalar(0.01), Rect::whole(w, h))?;
            encode_guided(&mut passes, &input, Rect::whole(w, h), &sg, (0, 0), 0.0).map(|_| ())
        }
    });
    assert!(
        matches!(refused, Err(GpuError::InvalidInput(_))),
        "{refused:?}"
    );
}

/// The cut of the whole frame that carries a tile's core and `halo` pixels of context.
fn tile_of(img: &RgbImage, sigma: &[f32], core: Rect, halo: usize) -> (RgbImage, Rect, Vec<f32>) {
    let rect = core.grow(Halo::uniform(halo), Rect::whole(img.width, img.height));
    let sub = img.extract(rect);
    let mut sig = Vec::with_capacity(rect.area());
    for y in rect.y..rect.y1() {
        sig.extend_from_slice(&sigma[y * img.width + rect.x..y * img.width + rect.x1()]);
    }
    (sub, core.relative_to(&rect), sig)
}

#[test]
fn guided_tiles_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    let img = chroma_noisy_scene(w, h, 61, 0.03);
    let plane = sigma_plane(w, h, 11);
    let mut worst = 0.0f32;
    for color in [25.0f32, 100.0] {
        let halo = 2 * radius_px(color);
        let whole = assert_tiles_equal_whole("guided", w, h, 3, &[48, 100], |core| {
            let (sub, rel, sig) = tile_of(&img, &plane, core, halo);
            run_guided(dev, &sub, rel, &SigmaOwned::Plane(sig), color).data
        });
        let want = chroma_guided_whole(&img, &NoiseSigma::Plane(&plane), 0, color);
        let e = compare(&whole, &want.data);
        eprintln!(
            "[G-06] guided C {color} 203x157 tiles: against the CPU max {:.3e}",
            e.max
        );
        worst = worst.max(e.max);
        assert!(e.max <= bound_unmeasured(dev, GUIDED_BOUND), "{:e}", e.max);
    }
    g06::ci_note(
        "guided tiles",
        &format!(
            "guided filter, 203x157 tiles on {}: against the CPU worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
}

#[test]
fn a_tile_with_less_context_than_the_receptive_field_differs() {
    // The halo is exactly 2 r: one pixel less gives other pixels at the core's edge. The test would
    // pass trivially if the filter ignored its neighbourhood.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (120usize, 90usize);
    let img = chroma_noisy_scene(w, h, 62, 0.03);
    let plane = sigma_plane(w, h, 12);
    let color = 50.0;
    let r = radius_px(color);
    let core = Rect::new(40, 30, 30, 20);
    let whole = run_guided(
        dev,
        &img,
        Rect::whole(w, h),
        &SigmaOwned::Plane(plane.clone()),
        color,
    );
    let want = whole.extract(core);
    let tile = |halo: usize| {
        let (sub, rel, sig) = tile_of(&img, &plane, core, halo);
        run_guided(dev, &sub, rel, &SigmaOwned::Plane(sig), color)
    };
    assert_bit_equal("halo 2 r", &tile(2 * r).data, &want.data);
    assert!(
        compare(&tile(2 * r - 1).data, &want.data).max > 1e-6,
        "one pixel less context must change the core"
    );
}

#[test]
fn detail_injection_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    for (i, (w, h, beta)) in [
        (1usize, 1usize, 0.15f32),
        (7, 5, 0.0375),
        (64, 48, 0.15),
        (131, 97, 0.6),
        (97, 131, 0.3),
    ]
    .into_iter()
    .enumerate()
    {
        let net = noisy_scene(w, h, 70 + i as u64, 0.01);
        let classical = noisy_scene(w, h, 170 + i as u64, 0.02);
        let want = cpu_inject(&net, &classical, beta).expect("cpu");
        let got = on_device(dev, {
            let (net, classical) = (net.clone(), classical.clone());
            move |ctx| gpu_inject(ctx, &net, &classical, Rect::whole(w, h), beta)
        });
        let e = compare(&got.data, &want.data);
        eprintln!(
            "[G-06] inject case {i} {w}x{h} beta {beta}: max {:.3e}",
            e.max
        );
        worst = worst.max(e.max);
        assert!(e.max <= bound(dev, INJECT_BOUND), "{:e}", e.max);
    }
    g06::ci_note(
        "detail",
        &format!(
            "detail injection on {}: worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
    // A zero beta is the net image, and the sizes must agree.
    let net = noisy_scene(10, 10, 1, 0.01);
    let other = noisy_scene(9, 10, 2, 0.01);
    let r = on_device(dev, {
        let (net, other) = (net.clone(), other.clone());
        move |ctx| Ok(gpu_inject(ctx, &net, &other, Rect::whole(10, 10), 0.3).is_err())
    });
    assert!(r, "different sizes must be refused");
    let same = on_device(dev, {
        let net = net.clone();
        move |ctx| gpu_inject(ctx, &net, &net, Rect::whole(10, 10), 0.0)
    });
    assert_bit_equal("beta 0", &same.data, &net.data);
}

/// The crop of an image to `rect` of the frame (the CPU chain's `crop_to` between two stages).
fn crop(img: &RgbImage, outer: Rect, rect: Rect) -> RgbImage {
    img.extract(rect.relative_to(&outer))
}

/// The R6 denoise to R7 sequence of one tile: the rectangles of the stages, as the tile stepper cuts
/// them (each stage's input is the crop the stage before it leaves), the CPU functions on the CPU,
/// and the three kernels chained on the device with the sigma plane made once for the tile, the
/// pre-denoise luma kept as one plane and every origin of the chain given as the position of one
/// rectangle in another.
#[test]
fn the_denoise_to_detail_sequence_of_a_tile_matches_the_cpu_stage_sequence() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (180usize, 140usize);
    let frame = chroma_noisy_scene(w, h, 80, 0.02);
    let sigma = sigma_plane(w, h, 13);
    let (scale, color, beta) = (1.5f32, 60.0f32, 0.3f32);
    let r = radius_px(color);
    // The output rectangle of R7 and the rectangles that lead to it.
    let r2 = Rect::new(50, 40, 60, 50);
    let bounds = Rect::whole(w, h);
    let r1 = r2.grow(Halo::uniform(2 * r), bounds);
    let r0 = r1.grow(Halo::uniform(2), bounds);
    let tile = |rect: Rect| frame.extract(rect);
    let sigma_of = |rect: Rect| -> Vec<f32> {
        let mut v = Vec::with_capacity(rect.area());
        for y in rect.y..rect.y1() {
            v.extend_from_slice(&sigma[y * w + rect.x..y * w + rect.x1()]);
        }
        v
    };

    // The CPU: denoise on R0 (sigma scaled), crop to R1, guided on R1, inject on R1, crop to R2.
    let input = tile(r0);
    let scaled: Vec<f32> = sigma_of(r0).iter().map(|s| s * scale).collect();
    let denoised = bilateral_whole(
        &input,
        &NoiseSigma::Plane(&scaled),
        0,
        &BilateralParams::default(),
    );
    let classical = crop(&input, r0, r1);
    let denoised = crop(&denoised, r0, r1);
    let s1 = sigma_of(r1);
    let colored = chroma_guided_whole(&denoised, &NoiseSigma::Plane(&s1), 0, color);
    let injected = cpu_inject(&colored, &classical, beta).expect("inject");
    let want = crop(&injected, r1, r2);

    // The device, in one pass list.
    let (got, submissions) = on_device(dev, {
        let (input, plane) = (input.clone(), sigma_of(r0));
        move |ctx| {
            let before = ctx.counters();
            let gin = GpuImage::upload_rgb(ctx, "tile", &input)?;
            let mut passes = PassList::new(ctx, "R6 to R7");
            let sg = encode_sigma(&mut passes, &SigmaInput::Plane(&plane), r0)?;
            let luma = encode_classical_luma(&mut passes, &gin, r1.relative_to(&r0))?;
            let den = encode_bilateral(
                &mut passes,
                &gin,
                r1.relative_to(&r0),
                &sg,
                (0, 0),
                scale,
                &BilateralParams::default(),
            )?;
            let origin1 = (r1.x - r0.x, r1.y - r0.y);
            let col = encode_guided(&mut passes, &den, r2.relative_to(&r1), &sg, origin1, color)?;
            let origin2 = (r2.x - r1.x, r2.y - r1.y);
            let out = encode_inject_detail(
                &mut passes,
                &col,
                Rect::whole(col.width, col.height),
                &luma,
                origin2,
                beta,
            )?;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = out.take(&done, read)?;
            let subs = ctx.counters().since(before).submissions;
            // Keep the intermediates alive until the list has finished.
            drop((gin, sg, luma, den, col));
            Ok((data, subs))
        }
    });
    let e = compare(&got, &want.data);
    eprintln!(
        "[G-06] R6 denoise -> R7 sequence on {r2:?}: max {:.3e} mean {:.2e}",
        e.max, e.mean
    );
    g06::ci_note(
        "denoise to injection",
        &format!(
            "R6 denoise to R7 sequence on {}: max {:.3e}",
            dev.gpu.info().name,
            e.max
        ),
    );
    assert!(e.max <= bound_unmeasured(dev, SEQUENCE_BOUND), "{:e}", e.max);
    assert_eq!(submissions, 1);
    assert_eq!(got.len(), 3 * r2.area());
    // The luma is the CPU's too: the stage changed the picture (a pass-through would be far off).
    let untouched = crop(&input, r0, r2);
    assert!(
        compare(&untouched.data, &want.data).max > 1e-3,
        "the stages did nothing"
    );
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = chroma_noisy_scene(70, 50, 5, 0.03);
    let plane = sigma_plane(70, 50, 9);
    let ran = assert_poison_neutral("guided", |gpu| {
        let (img, plane) = (img.clone(), plane.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            gpu_chroma_guided(
                ctx,
                &img,
                Rect::whole(70, 50),
                &SigmaInput::Plane(&plane),
                80.0,
            )
        })
        .expect("run")
        .data
    });
    let _ = ran;
    let net = noisy_scene(40, 30, 6, 0.01);
    let classical = noisy_scene(40, 30, 7, 0.02);
    assert_poison_neutral("detail", |gpu| {
        let (net, classical) = (net.clone(), classical.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            gpu_inject(ctx, &net, &classical, Rect::new(2, 2, 30, 20), 0.2)
        })
        .expect("run")
        .data
    });
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    g06::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut h = FNV_START;
        for (w, hh, seed, color) in [(97usize, 61usize, 1u64, 50.0f32), (203, 157, 2, 100.0)] {
            let img = chroma_noisy_scene(w, hh, seed, 0.02);
            let plane = sigma_plane(w, hh, seed);
            let out = run_guided(
                dev,
                &img,
                Rect::whole(w, hh),
                &SigmaOwned::Plane(plane),
                color,
            );
            h = fnv(h, &out.data);
        }
        h
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    let rect = Rect::new(0, 0, side, side);
    assert_max_binding_bytes("guided", rect, gpu_guided::max_binding_bytes);
    assert_max_binding_bytes("detail", rect, gpu_detail::max_binding_bytes);
    // Ten scratch planes in two bindings of 6 and 4 planes, not one of 10.
    assert!(gpu_guided::max_binding_bytes(rect) < 10 * 4 * rect.area() as u64);
    let _ = LUMA;
}

#[test]
fn bad_inputs_are_refused_with_the_reason() {
    let Some(dev) = f32_device() else { return };
    let img = chroma_noisy_scene(30, 30, 1, 0.02);
    let run = |core: Rect, sigma: SigmaOwned, color: f32| {
        let (img, sigma) = (img.clone(), sigma.clone());
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                gpu_chroma_guided(ctx, &img, core, &sigma.as_input(), color)
            })
    };
    let whole = Rect::whole(30, 30);
    assert!(run(whole, SigmaOwned::Scalar(0.02), 50.0).is_ok());
    for (what, result) in [
        (
            "NaN strength",
            run(whole, SigmaOwned::Scalar(0.02), f32::NAN),
        ),
        (
            "core outside",
            run(Rect::new(20, 20, 20, 20), SigmaOwned::Scalar(0.02), 50.0),
        ),
        (
            "empty core",
            run(Rect::new(0, 0, 0, 3), SigmaOwned::Scalar(0.02), 50.0),
        ),
        (
            "short sigma plane",
            run(whole, SigmaOwned::Plane(vec![0.01; 5]), 50.0),
        ),
        (
            "a radius above the maximum",
            run(whole, SigmaOwned::Scalar(0.02), 1000.0),
        ),
    ] {
        let err = result.expect_err(what);
        assert!(matches!(err, GpuError::InvalidInput(_)), "{what}: {err}");
    }
    assert_eq!(
        radius_px(1000.0),
        62,
        "the radius rule is the CPU's, unclamped"
    );
    assert!(radius_px(1000.0) > MAX_RADIUS_PX);
}
