//! The R8 Richardson-Lucy kernel (`shaders/richardson_lucy.wgsl`, `kernels::richardson_lucy`)
//! against its CPU reference `cpu::sharpen::richardson_lucy_whole`.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/17-gpu.md` section 12):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | iterations 1 to 10 (amount 30 to 300), radius 0.3 to 2.5, scalar and plane sigma, frames from 1 x 1 to 200 x 150, in the frame's corner, cut from the middle of an 8658 x 5772 frame and from its far corner | max `abs(gpu - cpu)` at most [`BOUND_N4`] for n up to 4 and [`BOUND`] up to 10 (software; hardware [`HW_BOUND`]) | by iterations: n1 2.4e-7, n2 2.4e-7, n3 3.6e-7, n4 7.2e-7, n5 7.2e-7, n6 4.2e-7, n7 3.6e-7, n8 3.6e-7, n9 4.2e-7, n10 5.4e-7 |
//! | tiles with the plan's halo, Auto values (n = 4, halo 24) | bit-identical to the whole frame | 0 differing |
//! | tiles beyond the halo cap (n = 10, radius 1.5: exact radius 120 against the cap of 24) | at most [`BEYOND_CAP_BOUND`] from the whole frame | 2.1e-7 |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | `LP_NUM_THREADS=1` against the default | bit-identical | |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`);
//! the Metal runners run the same file (`mac-gpu.yml`).

mod common;
#[path = "denoise_common/mod.rs"]
mod denoise_common;

use common::exact::{
    assert_all_finite, assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole,
};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use denoise_common::{FNV_START, compare, fnv, placed, sigma_plane};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::filters::gaussian_kernel;
use ym_gpu::cpu::sharpen::{SharpenParams, richardson_lucy_whole};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{NoiseSigma, RgbImage, mirror};
use ym_gpu::device::{GpuContext, GpuError, StorageMode};
use ym_gpu::kernels::bilateral::SigmaInput;
use ym_gpu::kernels::richardson_lucy::{
    self, RlPlan, max_binding_bytes, richardson_lucy as gpu_rl,
};
use ym_gpu::queue::{JobSpec, Priority};

/// n up to 4 (the Auto values), software adapter: twice the first measured value (7.15e-7).
const BOUND_N4: f32 = 1.5e-6;
/// n up to 10, software adapter: twice the worst first measured value (7.15e-7, at n = 4 and 5; the
/// others are at most 5.4e-7).
const BOUND: f32 = 1.5e-6;
/// Hardware adapters, `matches_the_cpu_for_every_iteration_count`, which prints its worst value by iteration count: Metal
/// (macos-14 and macos-15, run 38694506557) gave 2.38e-7 (n = 1) rising to 1.19e-6 (n = 4) and 2.26e-6 (n = 10), so the
/// bounds are twice the worst of n up to 4 and of n up to 10.
const HW_BOUND_N4: f32 = 2.38e-6;
/// See [`HW_BOUND_N4`]: n from 5 to 10.
const HW_BOUND_N10: f32 = 4.52e-6;
/// Hardware adapters, the tests whose Metal value was not printed yet (the tile and PSF tests now print it): the
/// starting value for Richardson-Lucy (`docs/17-gpu.md` section 12), to be lowered to twice what the next Metal run prints.
const HW_BOUND: f32 = 5e-5;
/// The distance of a tile beyond the halo cap from the whole frame: twice the first measured value
/// (2.1e-7; the CPU's own bands differ from its whole-frame run by 9.5e-7 there).
const BEYOND_CAP_BOUND: f32 = 5e-7;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// The bound of a test that has not printed a Metal value yet.
fn bound(dev: &TestGpu, software: f32) -> f32 {
    if dev.gpu.info().is_software {
        software
    } else {
        HW_BOUND.max(software)
    }
}

/// The bound of the iteration-count test, whose Metal values were printed.
fn bound_by_iterations(dev: &TestGpu, n: usize) -> f32 {
    match (dev.gpu.info().is_software, n <= 4) {
        (true, true) => BOUND_N4,
        (true, false) => BOUND,
        (false, true) => HW_BOUND_N4,
        (false, false) => HW_BOUND_N10,
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

/// A separable Gaussian blur of every channel (mirror edge), to make a soft image to sharpen.
fn blur(img: &RgbImage, sigma: f32) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut out = img.clone();
    let k = gaussian_kernel(sigma, (3.0 * sigma).ceil() as usize);
    let r = (k.len() / 2) as isize;
    for c in 0..3 {
        let src = img.plane(c).to_vec();
        let mut tmp = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                tmp[y * w + x] = (-r..=r)
                    .map(|d| k[(d + r) as usize] * src[y * w + mirror(x as isize + d, w)])
                    .sum();
            }
        }
        let dst = out.plane_mut(c);
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = (-r..=r)
                    .map(|d| k[(d + r) as usize] * tmp[mirror(y as isize + d, h) * w + x])
                    .sum();
            }
        }
    }
    out
}

/// A soft, noisy scene of text and natural texture: what R8 is made for.
fn soft_scene(w: usize, h: usize, seed: u64, noise: f32) -> RgbImage {
    let sharp = if seed.is_multiple_of(2) {
        synth::fine_text(w, h, 2)
    } else {
        synth::natural(w, h, seed)
    };
    let mut img = blur(&sharp, 0.9);
    synth::add_noise(&mut img.data, noise, seed + 99);
    img
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

fn run_gpu(
    dev: &TestGpu,
    img: &RgbImage,
    core: Rect,
    sigma: &SigmaOwned,
    params: SharpenParams,
) -> RgbImage {
    let (img, sigma) = (img.clone(), sigma.clone());
    on_device(dev, move |ctx| {
        gpu_rl(ctx, &img, core, &sigma.as_input(), &params)
    })
}

fn cpu_reference(img: &RgbImage, sigma: &SigmaOwned, params: &SharpenParams) -> RgbImage {
    let plane = sigma.plane(img.width * img.height);
    richardson_lucy_whole(img, params, &NoiseSigma::Plane(&plane), 0)
}

/// One case: width, height, the frame it is cut from (`x0, y0, frame width, frame height`), the
/// sigma, the amount and the radius.
type RlCase = (
    usize,
    usize,
    Option<(usize, usize, usize, usize)>,
    SigmaOwned,
    f32,
    f32,
);

#[test]
fn matches_the_cpu_for_every_iteration_count() {
    let Some(dev) = f32_device() else { return };
    let mut worst_by_n = [0.0f32; 11];
    // (width, height, frame, sigma, amount, radius): `frame` places the image in a bigger frame.
    let mut cases: Vec<RlCase> = Vec::new();
    for n in 1..=10usize {
        let amount = 30.0 * n as f32;
        cases.push((96, 72, None, SigmaOwned::Scalar(0.004), amount, 0.8));
        cases.push((
            120,
            90,
            Some((4000, 2800, 8658, 5772)),
            SigmaOwned::Plane(sigma_plane(120, 90, n as u64)),
            amount,
            1.2,
        ));
    }
    for (w, h, radius, amount) in [
        (1usize, 1usize, 0.8f32, 120.0f32),
        (2, 2, 0.8, 120.0),
        (3, 5, 0.3, 60.0),
        (7, 9, 2.5, 90.0),
        (33, 17, 0.3, 300.0),
        (17, 33, 2.5, 300.0),
        (131, 97, 0.8, 120.0),
        (200, 150, 1.0, 120.0),
        (200, 150, 0.7, 150.0),
    ] {
        cases.push((w, h, None, SigmaOwned::Scalar(0.003), amount, radius));
    }
    // The far corner of an 8658 x 5772 frame: where the PSF is widest and a coordinate of the order
    // of 1e4 held in f32 would round.
    cases.push((
        90,
        70,
        Some((8658 - 90, 5772 - 70, 8658, 5772)),
        SigmaOwned::Scalar(0.004),
        120.0,
        0.8,
    ));
    cases.push((
        90,
        70,
        Some((0, 0, 8658, 5772)),
        SigmaOwned::Scalar(0.004),
        120.0,
        0.8,
    ));
    for (i, (w, h, frame, sigma, amount, radius)) in cases.into_iter().enumerate() {
        let params = SharpenParams {
            amount,
            radius_px: radius,
        };
        let mut img = soft_scene(w, h, 5 + i as u64, 0.004);
        if let Some((x0, y0, fw, fh)) = frame {
            img = placed(img, x0, y0, fw, fh);
        }
        let want = cpu_reference(&img, &sigma, &params);
        let got = run_gpu(dev, &img, Rect::whole(w, h), &sigma, params);
        assert_all_finite(&format!("case {i} {w}x{h}"), &got.data);
        let e = compare(&got.data, &want.data);
        let n = params.iterations();
        eprintln!(
            "[R6..R8] rl case {i} {w}x{h} n {n} radius {radius} frame {frame:?}: max {:.3e} at {} mean {:.2e}",
            e.max, e.at, e.mean
        );
        worst_by_n[n] = worst_by_n[n].max(e.max);
        let limit = bound_by_iterations(dev, n);
        assert!(
            e.max <= limit,
            "case {i} {w}x{h} n {n}: {:e} > {limit:e}",
            e.max
        );
        // It did something: R8 changes the image.
        if w * h > 100 {
            assert!(
                compare(&got.data, &img.data).max > 1e-3,
                "case {i}: nothing sharpened"
            );
        }
    }
    let line: Vec<String> = (1..=10)
        .map(|n| format!("n{n} {:.2e}", worst_by_n[n]))
        .collect();
    denoise_common::ci_note(
        "richardson-lucy",
        &format!(
            "Richardson-Lucy on {}: worst by iterations: {}",
            dev.gpu.info().name,
            line.join(", ")
        ),
    );
}

#[test]
fn zero_iterations_return_the_input_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let img = soft_scene(40, 30, 3, 0.01);
    for amount in [0.0f32, 14.0, -5.0] {
        let params = SharpenParams {
            amount,
            radius_px: 0.8,
        };
        assert!(RlPlan::new(&params).is_none());
        let core = Rect::new(2, 3, 30, 20);
        let got = run_gpu(dev, &img, core, &SigmaOwned::Scalar(0.01), params);
        assert_bit_equal("zero iterations", &got.data, &img.extract(core).data);
    }
}

/// The cut of the whole frame that carries a tile's core and `halo` pixels of context, keeping the
/// placement.
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
fn tiles_with_the_plans_halo_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    // The Auto values: n = 4, rho 0.8, exact radius 24 = the cap.
    let params = SharpenParams {
        amount: 120.0,
        radius_px: 0.8,
    };
    let halo = RlPlan::new(&params).expect("plan").halo_px();
    assert_eq!(halo, 24);
    let img = placed(soft_scene(w, h, 31, 0.004), 1000, 700, 8658, 5772);
    let plane = sigma_plane(w, h, 21);
    let whole = assert_tiles_equal_whole("richardson-lucy", w, h, 3, &[48, 100], |core| {
        let (sub, rel, sig) = tile_of(&img, &plane, core, halo);
        run_gpu(dev, &sub, rel, &SigmaOwned::Plane(sig), params).data
    });
    let want = cpu_reference(&img, &SigmaOwned::Plane(plane), &params);
    let e = compare(&whole, &want.data);
    denoise_common::ci_note(
        "rl tiles",
        &format!(
            "Richardson-Lucy 203x157 tiles (n 4) on {}: against the CPU max {:.3e}",
            dev.gpu.info().name,
            e.max
        ),
    );
    assert!(e.max <= bound(dev, BOUND_N4), "{:e}", e.max);
}

#[test]
fn beyond_the_halo_cap_a_tile_differs_from_the_whole_frame_by_a_little() {
    // n = 10 at radius 1.5: sigma_max 1.875, kernel radius 6, exact receptive field 2 * 10 * 6 = 120 px
    // against the scheduler's cap of 24. The CPU's bands differ from its whole-frame run the same way
    // (about 2e-7 on photographic content, `docs/05`).
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    let params = SharpenParams {
        amount: 300.0,
        radius_px: 1.5,
    };
    let plan = RlPlan::new(&params).expect("plan");
    assert_eq!(
        (plan.iterations(), plan.radius_px(), plan.halo_px()),
        (10, 6, 24)
    );
    let img = soft_scene(w, h, 33, 0.004);
    let plane = sigma_plane(w, h, 22);
    let sigma = SigmaOwned::Plane(plane.clone());
    let whole = run_gpu(dev, &img, Rect::whole(w, h), &sigma, params);
    let mut worst = 0.0f32;
    for core in [
        Rect::new(50, 40, 60, 50),
        Rect::new(0, 0, 70, 60),
        Rect::new(133, 97, 70, 60),
    ] {
        let (sub, rel, sig) = tile_of(&img, &plane, core, plan.halo_px());
        let tile = run_gpu(dev, &sub, rel, &SigmaOwned::Plane(sig), params);
        let want = whole.extract(core);
        let e = compare(&tile.data, &want.data);
        worst = worst.max(e.max);
    }
    denoise_common::ci_note(
        "rl beyond the cap",
        &format!(
            "Richardson-Lucy beyond the cap (n 10, exact radius 120, halo 24) on {}: tile against whole {worst:.3e}",
            dev.gpu.info().name
        ),
    );
    assert!(worst <= BEYOND_CAP_BOUND, "{worst:e}");
    // The CPU's own band decomposition has the same property: report it for comparison.
    let cpu_whole = cpu_reference(&img, &sigma, &params);
    let plane2 = plane.clone();
    let banded = ym_gpu::cpu::sharpen::richardson_lucy(&img, &params, &NoiseSigma::Plane(&plane2));
    eprintln!(
        "[R6..R8] rl beyond the cap on the CPU: banded against whole {:.3e}",
        compare(&banded.data, &cpu_whole.data).max
    );
}

#[test]
fn the_psf_follows_the_position_in_the_frame() {
    // The same pixels at the centre and in the far corner of a 8658 x 5772 frame get different PSFs
    // (sigma 0.8 against 1.0): the results differ, and each matches the CPU's at its own place.
    let Some(dev) = f32_device() else { return };
    let base = soft_scene(80, 60, 41, 0.004);
    let params = SharpenParams {
        amount: 120.0,
        radius_px: 0.8,
    };
    let sigma = SigmaOwned::Scalar(0.004);
    let centre = placed(base.clone(), 8658 / 2 - 40, 5772 / 2 - 30, 8658, 5772);
    let corner = placed(base, 8658 - 80, 5772 - 60, 8658, 5772);
    let a = run_gpu(dev, &centre, Rect::whole(80, 60), &sigma, params);
    let b = run_gpu(dev, &corner, Rect::whole(80, 60), &sigma, params);
    assert!(
        compare(&a.data, &b.data).max > 1e-4,
        "the PSF did not depend on the position"
    );
    let mut worst = 0.0f32;
    for (img, got) in [(&centre, &a), (&corner, &b)] {
        let want = cpu_reference(img, &sigma, &params);
        let e = compare(&got.data, &want.data).max;
        worst = worst.max(e);
        assert!(e <= bound(dev, BOUND_N4), "{e:e}");
    }
    denoise_common::ci_note(
        "rl psf",
        &format!(
            "Richardson-Lucy, PSF at the centre and the far corner on {}: against the CPU worst {worst:.3e}",
            dev.gpu.info().name
        ),
    );
}

#[test]
fn damping_holds_flat_noise_down_on_the_device_as_on_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut img = RgbImage::new(64, 64);
    img.data.fill(0.3);
    synth::add_noise(&mut img.data, 0.01, 9);
    let params = SharpenParams {
        amount: 120.0,
        radius_px: 0.8,
    };
    let sd = |im: &RgbImage| {
        let y = im.luma_plane();
        let m = y.iter().sum::<f32>() / y.len() as f32;
        (y.iter().map(|v| (v - m).powi(2)).sum::<f32>() / y.len() as f32).sqrt()
    };
    let plain = run_gpu(
        dev,
        &img,
        Rect::whole(64, 64),
        &SigmaOwned::Scalar(0.0),
        params,
    );
    let damped = run_gpu(
        dev,
        &img,
        Rect::whole(64, 64),
        &SigmaOwned::Scalar(0.01),
        params,
    );
    assert!(
        sd(&damped) < sd(&plain),
        "{} !< {}",
        sd(&damped),
        sd(&plain)
    );
    assert!(sd(&damped) < 1.5 * sd(&img));
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = soft_scene(70, 50, 5, 0.004);
    let plane = sigma_plane(70, 50, 9);
    let params = SharpenParams {
        amount: 150.0,
        radius_px: 1.1,
    };
    assert_poison_neutral("richardson-lucy", |gpu| {
        let (img, plane) = (img.clone(), plane.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            gpu_rl(
                ctx,
                &img,
                Rect::whole(70, 50),
                &SigmaInput::Plane(&plane),
                &params,
            )
        })
        .expect("run")
        .data
    });
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut h = FNV_START;
        for (w, hh, seed, amount) in [(97usize, 61usize, 1u64, 120.0f32), (203, 157, 2, 300.0)] {
            let img = soft_scene(w, hh, seed, 0.004);
            let plane = sigma_plane(w, hh, seed);
            let params = SharpenParams {
                amount,
                radius_px: 0.9,
            };
            let out = run_gpu(
                dev,
                &img,
                Rect::whole(w, hh),
                &SigmaOwned::Plane(plane),
                params,
            );
            h = fnv(h, &out.data);
        }
        h
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    assert_max_binding_bytes(
        "richardson-lucy",
        Rect::new(0, 0, side, side),
        max_binding_bytes,
    );
    let _ = richardson_lucy::RlPlan::new;
}

#[test]
fn bad_inputs_are_refused_with_the_reason() {
    let Some(dev) = f32_device() else { return };
    let img = soft_scene(30, 30, 1, 0.01);
    let params = SharpenParams {
        amount: 120.0,
        radius_px: 0.8,
    };
    let run = |core: Rect, sigma: SigmaOwned| {
        let img = img.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                gpu_rl(ctx, &img, core, &sigma.as_input(), &params)
            })
    };
    assert!(run(Rect::whole(30, 30), SigmaOwned::Scalar(0.01)).is_ok());
    for (what, result) in [
        (
            "core outside",
            run(Rect::new(20, 20, 20, 20), SigmaOwned::Scalar(0.01)),
        ),
        (
            "empty core",
            run(Rect::new(0, 0, 0, 3), SigmaOwned::Scalar(0.01)),
        ),
        (
            "short sigma plane",
            run(Rect::whole(30, 30), SigmaOwned::Plane(vec![0.01; 5])),
        ),
    ] {
        let err = result.expect_err(what);
        assert!(matches!(err, GpuError::InvalidInput(_)), "{what}: {err}");
    }
}
