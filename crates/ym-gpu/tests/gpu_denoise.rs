//! The R6 bilateral denoise kernel (`shaders/bilateral.wgsl`, `kernels::bilateral`) against its
//! CPU reference `cpu::denoise::bilateral_whole`, and the noise-sigma plane of a tile
//! (`shaders/sigma_expand.wgsl`) against the rule of `ym-core`'s `full_sigma_map`.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/17-gpu.md` section 12):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | 14 frames from 1 x 1 to 200 x 150, noise 0.002 to 0.03, scalar and plane sigma, scale 0.6 to 2.2, strength 0.5 to 2.5, radius 1 to 4 | max `abs(gpu - cpu)` at most [`BOUND`] (software adapters; hardware: [`HW_BOUND`], the starting threshold, tightened from the Metal numbers) | 4.17e-7 (131 x 97, radius 4); at most 3.6e-7 elsewhere; no sample over 2e-6 |
//! | zero sigma, zero scale or zero strength | the input, bit for bit | 0 differing |
//! | tiles (two sizes, the stage's own 2 px of context) | bit-identical to the whole frame | 0 differing |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | the sigma plane, every cell: rectangles at odd origins, odd frames, with and without gains | bit-identical to the CPU rule | 0 differing |
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
use denoise_common::{FNV_START, compare, fnv, noisy_scene, sigma_plane};
use ym_gpu::chain::MAX_PHYS_AREA;
use ym_gpu::cpu::denoise::{BilateralParams, bilateral_whole};
use ym_gpu::cpu::tiling::{Halo, Rect};
use ym_gpu::cpu::{NoiseSigma, RgbImage};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::bilateral::{
    self, MAX_RADIUS_PX, SigmaHalf, SigmaInput, bilateral as gpu_bilateral, expand_sigma,
};
use ym_gpu::queue::{JobSpec, Priority};

/// Largest `|gpu - cpu|` of a frame (white 1.0) on a software adapter: twice the first measured value
/// (4.17e-7 on lavapipe).
const BOUND: f32 = 8.4e-7;
/// The same on a hardware adapter for the 14 frames of `matches_the_cpu_on_every_frame...`, which print their worst
/// value (`ci_note`). Metal (macos-14 and macos-15, run 38694506557, the line `bilateral on Apple Paravirtual device`):
/// 6.557e-7, so twice that (fast-math contracts and re-associates the sums, `exp` and the divisions are the
/// driver's).
const HW_BOUND: f32 = 1.32e-6;
/// The same for the tests whose Metal value was not printed yet (the tile test now prints it): the starting
/// value for neighbourhood stages (`docs/17-gpu.md` section 12), to be lowered to twice what the next Metal run prints.
const HW_BOUND_UNMEASURED: f32 = 1e-5;

fn bound(dev: &TestGpu) -> f32 {
    if dev.gpu.info().is_software {
        BOUND
    } else {
        HW_BOUND
    }
}

/// The bound of a test that has not printed a Metal value yet.
fn bound_unmeasured(dev: &TestGpu) -> f32 {
    if dev.gpu.info().is_software {
        BOUND
    } else {
        HW_BOUND_UNMEASURED
    }
}

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// One run on the device thread.
fn on_device<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, ym_gpu::device::GpuError>
    + Send
    + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("the device job")
}

fn run_gpu(
    dev: &TestGpu,
    img: &RgbImage,
    core: Rect,
    sigma: SigmaOwned,
    scale: f32,
    p: BilateralParams,
) -> RgbImage {
    let img = img.clone();
    on_device(dev, move |ctx| {
        gpu_bilateral(ctx, &img, core, &sigma.as_input(), scale, &p)
    })
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

    /// The CPU's sigma with the scale applied (`sigma_tile`: the value times the scale, in `f32`).
    fn scaled(&self, len: usize, scale: f32) -> Vec<f32> {
        match self {
            SigmaOwned::Scalar(v) => vec![*v * scale; len],
            SigmaOwned::Plane(p) => p.iter().map(|s| *s * scale).collect(),
        }
    }
}

fn cpu_reference(img: &RgbImage, sigma: &SigmaOwned, scale: f32, p: &BilateralParams) -> RgbImage {
    let plane = sigma.scaled(img.width * img.height, scale);
    bilateral_whole(img, &NoiseSigma::Plane(&plane), 0, p)
}

#[test]
fn matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut worst_over = 0.0f64;
    let mut total = 0usize;
    // (size, noise, sigma, scale, params)
    let cases: Vec<(usize, usize, f32, SigmaOwned, f32, BilateralParams)> = {
        let default = BilateralParams::default();
        let r = |radius_px, spatial_sigma_px, strength| BilateralParams {
            radius_px,
            spatial_sigma_px,
            strength,
        };
        vec![
            (1, 1, 0.01, SigmaOwned::Scalar(0.01), 1.0, default),
            (2, 2, 0.01, SigmaOwned::Scalar(0.01), 1.0, default),
            (3, 5, 0.02, SigmaOwned::Scalar(0.02), 1.0, default),
            (
                7,
                9,
                0.01,
                SigmaOwned::Plane(sigma_plane(7, 9, 1)),
                1.7,
                default,
            ),
            (33, 17, 0.01, SigmaOwned::Scalar(0.01), 1.7, r(2, 1.5, 0.5)),
            (
                64,
                48,
                0.02,
                SigmaOwned::Plane(sigma_plane(64, 48, 2)),
                1.0,
                r(1, 0.8, 1.0),
            ),
            (
                64,
                48,
                0.002,
                SigmaOwned::Scalar(0.002),
                1.0,
                r(3, 2.0, 2.5),
            ),
            (
                131,
                97,
                0.01,
                SigmaOwned::Plane(sigma_plane(131, 97, 3)),
                1.7,
                default,
            ),
            (131, 97, 0.02, SigmaOwned::Scalar(0.02), 0.6, r(4, 2.5, 1.0)),
            (
                97,
                131,
                0.005,
                SigmaOwned::Plane(sigma_plane(97, 131, 4)),
                2.2,
                r(2, 1.5, 1.8),
            ),
            (5, 131, 0.01, SigmaOwned::Scalar(0.01), 1.0, default),
            (131, 5, 0.01, SigmaOwned::Scalar(0.01), 1.0, default),
            (
                200,
                150,
                0.01,
                SigmaOwned::Plane(sigma_plane(200, 150, 5)),
                1.0,
                default,
            ),
            (200, 150, 0.03, SigmaOwned::Scalar(0.03), 1.0, default),
        ]
    };
    for (i, (w, h, noise, sigma, scale, p)) in cases.into_iter().enumerate() {
        let img = noisy_scene(w, h, 10 + i as u64, noise);
        let want = cpu_reference(&img, &sigma, scale, &p);
        let got = run_gpu(dev, &img, Rect::whole(w, h), sigma.clone(), scale, p);
        assert_all_finite(&format!("case {i} {w}x{h}"), &got.data);
        let e = compare(&got.data, &want.data);
        let over =
            denoise_common::Err::over(&got.data, &want.data, 2e-6) as f64 / got.data.len() as f64;
        eprintln!(
            "[R6..R8] bilateral case {i} {w}x{h} {p:?} scale {scale}: max {:.3e} at {} mean {:.2e}, over 2e-6: {over:.2e}",
            e.max, e.at, e.mean
        );
        worst = worst.max(e.max);
        worst_over = worst_over.max(over);
        total += got.data.len();
        let limit = bound(dev);
        assert!(e.max <= limit, "case {i} {w}x{h}: {:e} > {limit:e}", e.max);
    }
    denoise_common::ci_note(
        "bilateral",
        &format!(
            "bilateral on {}: worst {worst:.3e} over {total} samples, worst share over 2e-6 {worst_over:.2e}",
            dev.gpu.info().name
        ),
    );
}

#[test]
fn a_zero_sigma_scale_or_strength_returns_the_input_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let img = noisy_scene(40, 30, 3, 0.02);
    let whole = Rect::whole(40, 30);
    let cases = [
        (SigmaOwned::Scalar(0.0), 1.0, BilateralParams::default()),
        (SigmaOwned::Scalar(0.01), 0.0, BilateralParams::default()),
        (
            SigmaOwned::Scalar(0.01),
            1.0,
            BilateralParams {
                strength: 0.0,
                ..BilateralParams::default()
            },
        ),
    ];
    for (i, (sigma, scale, p)) in cases.into_iter().enumerate() {
        let got = run_gpu(dev, &img, whole, sigma, scale, p);
        assert_bit_equal(&format!("identity case {i}"), &got.data, &img.data);
    }
    // A plane that is zero in places leaves exactly those pixels alone and filters the others.
    let (w, h) = (48usize, 40usize);
    let img = noisy_scene(w, h, 4, 0.02);
    let plane = sigma_plane(w, h, 7);
    let got = run_gpu(
        dev,
        &img,
        Rect::whole(w, h),
        SigmaOwned::Plane(plane.clone()),
        1.0,
        BilateralParams::default(),
    );
    let n = w * h;
    let (mut kept, mut changed) = (0, 0);
    for (i, &sigma) in plane.iter().enumerate() {
        let same = (0..3).all(|c| got.data[c * n + i].to_bits() == img.data[c * n + i].to_bits());
        if sigma == 0.0 {
            assert!(same, "a pixel with sigma 0 changed at {i}");
            kept += 1;
        } else if !same {
            changed += 1;
        }
    }
    assert!(
        kept > 20 && changed > n / 2,
        "kept {kept}, changed {changed}"
    );
}

/// The cut of the whole frame that carries a tile's core and the 2 px of context the stage needs.
fn tile_of(img: &RgbImage, plane: &[f32], core: Rect) -> (RgbImage, Rect, Vec<f32>) {
    let bounds = Rect::whole(img.width, img.height);
    let rect = core.grow(Halo::uniform(2), bounds);
    let sub = img.extract(rect);
    let mut sig = Vec::with_capacity(rect.area());
    for y in rect.y..rect.y1() {
        sig.extend_from_slice(&plane[y * img.width + rect.x..y * img.width + rect.x1()]);
    }
    (sub, core.relative_to(&rect), sig)
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    let img = noisy_scene(w, h, 21, 0.01);
    let plane = sigma_plane(w, h, 8);
    let p = BilateralParams::default();
    let whole = assert_tiles_equal_whole("bilateral", w, h, 3, &[32, 100], |core| {
        let (sub, rel, sig) = tile_of(&img, &plane, core);
        run_gpu(dev, &sub, rel, SigmaOwned::Plane(sig), 1.3, p).data
    });
    // And the whole frame is the CPU's within the bound.
    let want = cpu_reference(&img, &SigmaOwned::Plane(plane), 1.3, &p);
    let e = compare(&whole, &want.data);
    denoise_common::ci_note(
        "bilateral tiles",
        &format!(
            "bilateral 203x157 tiles on {}: against the CPU max {:.3e}",
            dev.gpu.info().name,
            e.max
        ),
    );
    assert!(e.max <= bound_unmeasured(dev), "{:e}", e.max);
}

#[test]
fn a_poisoned_device_and_a_plain_one_give_the_same_bits() {
    let img = noisy_scene(70, 50, 5, 0.015);
    let plane = sigma_plane(70, 50, 9);
    let ran = assert_poison_neutral("bilateral", |gpu| {
        let img = img.clone();
        let plane = plane.clone();
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            gpu_bilateral(
                ctx,
                &img,
                Rect::whole(70, 50),
                &SigmaInput::Plane(&plane),
                1.0,
                &BilateralParams::default(),
            )
        })
        .expect("run")
        .data
    });
    let _ = ran;
}

#[test]
fn the_sigma_plane_has_the_cpus_bits_for_every_cell() {
    let Some(dev) = f32_device() else { return };
    // The CPU rule, written out as `full_sigma_map` has it.
    let reference = |half: &SigmaHalf<'_>, rect: Rect| -> Vec<f32> {
        let mut out = Vec::with_capacity(rect.area());
        for y in rect.y..rect.y1() {
            let cy = (y / 2).min(half.height - 1);
            for x in rect.x..rect.x1() {
                let cx = (x / 2).min(half.width - 1);
                let g = half.gains.map_or(1.0, |g| g[cy * half.width + cx]);
                out.push(half.data[cy * half.width + cx] * g);
            }
        }
        out
    };
    let mut cells_checked = 0usize;
    // 41 x 29 cells cover 82 x 58 photosites; frames with an odd side have one more photosite than
    // the cells cover, which reads the last cell.
    let (cw, ch) = (41usize, 29usize);
    let mut rng = ym_gpu::cpu::synth::Rng::new(77);
    let data: Vec<f32> = (0..cw * ch)
        .map(|_| 0.001 + 0.02 * rng.next_f32())
        .collect();
    let gains: Vec<f32> = (0..cw * ch).map(|_| 1.0 + 3.0 * rng.next_f32()).collect();
    for with_gains in [false, true] {
        let half = SigmaHalf {
            width: cw,
            height: ch,
            data: &data,
            gains: with_gains.then_some(gains.as_slice()),
        };
        for rect in [
            Rect::new(0, 0, 82, 58),
            Rect::new(0, 0, 83, 59),
            Rect::new(1, 1, 80, 56),
            Rect::new(37, 21, 45, 38),
            Rect::new(81, 57, 2, 2),
            Rect::new(82, 58, 1, 1),
            Rect::new(3, 0, 1, 58),
            Rect::new(0, 5, 83, 1),
        ] {
            let got = on_device(dev, {
                let (data, gains) = (data.clone(), gains.clone());
                move |ctx| {
                    let half = SigmaHalf {
                        width: cw,
                        height: ch,
                        data: &data,
                        gains: with_gains.then_some(gains.as_slice()),
                    };
                    expand_sigma(ctx, &SigmaInput::Half(&half), rect)
                }
            });
            let want = reference(&half, rect);
            assert_bit_equal(
                &format!("sigma plane {rect:?} gains {with_gains}"),
                &got,
                &want,
            );
            // The Rust form of the rule used by callers agrees with it too.
            for (i, v) in want.iter().enumerate() {
                let (x, y) = (rect.x + i % rect.w, rect.y + i / rect.w);
                assert_eq!(half.at(x, y).to_bits(), v.to_bits());
            }
            cells_checked += want.len();
        }
    }
    // The scalar and the plane sources.
    let got = on_device(dev, |ctx| {
        expand_sigma(ctx, &SigmaInput::Scalar(0.0123), Rect::new(5, 6, 33, 21))
    });
    assert!(got.len() == 33 * 21 && got.iter().all(|v| v.to_bits() == 0.0123f32.to_bits()));
    let plane: Vec<f32> = (0..33 * 21).map(|i| i as f32 * 1e-4).collect();
    let got = on_device(dev, {
        let plane = plane.clone();
        move |ctx| expand_sigma(ctx, &SigmaInput::Plane(&plane), Rect::new(5, 6, 33, 21))
    });
    assert_bit_equal("plane source", &got, &plane);
    eprintln!("[R6..R8] sigma plane: {cells_checked} cells bit-identical to the CPU rule");
}

#[test]
fn the_bilateral_stage_and_the_sigma_plane_run_in_one_pass_list_without_a_readback_between() {
    // Device in, device out: the sigma plane and the filter chained inside one list equal the host
    // wrapper (which is a layer on the same two functions) bit for bit, and cost one submission.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (90usize, 60usize);
    let img = noisy_scene(w, h, 31, 0.01);
    let cells: Vec<f32> = (0..(w / 2) * (h / 2))
        .map(|i| 0.004 + 1e-5 * (i % 97) as f32)
        .collect();
    let gains: Vec<f32> = (0..(w / 2) * (h / 2))
        .map(|i| 1.0 + 1e-3 * (i % 13) as f32)
        .collect();
    let core = Rect::new(2, 3, 80, 50);
    let (a, subs) = {
        let (img, cells, gains) = (img.clone(), cells.clone(), gains.clone());
        on_device(dev, move |ctx| {
            let half = SigmaHalf {
                width: w / 2,
                height: h / 2,
                data: &cells,
                gains: Some(&gains),
            };
            let before = ctx.counters();
            let input = ym_gpu::image::GpuImage::upload_rgb(ctx, "in", &img)?;
            let mut passes = ym_gpu::stream::PassList::new(ctx, "chain");
            let sg =
                bilateral::encode_sigma(&mut passes, &SigmaInput::Half(&half), Rect::whole(w, h))?;
            let out = bilateral::encode_bilateral(
                &mut passes,
                &input,
                core,
                &sg,
                (0, 0),
                1.0,
                &BilateralParams::default(),
            )?;
            let read = out.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            let data = out.take(&done, read)?;
            let subs = ctx.counters().since(before).submissions;
            Ok((data, subs))
        })
    };
    let b = on_device(dev, {
        let img = img.clone();
        move |ctx| {
            let half = SigmaHalf {
                width: w / 2,
                height: h / 2,
                data: &cells,
                gains: Some(&gains),
            };
            gpu_bilateral(
                ctx,
                &img,
                core,
                &SigmaInput::Half(&half),
                1.0,
                &BilateralParams::default(),
            )
        }
    });
    assert_bit_equal("device chain against the host wrapper", &a, &b.data);
    assert_eq!(
        subs, 1,
        "the sigma plane and the filter share one submission"
    );
}

#[test]
fn thread_count_does_not_change_a_bit() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent("thread_count_does_not_change_a_bit", || {
        let mut h = FNV_START;
        for (w, hh, seed) in [(97usize, 61usize, 1u64), (203, 157, 2)] {
            let img = noisy_scene(w, hh, seed, 0.012);
            let plane = sigma_plane(w, hh, seed);
            let out = run_gpu(
                dev,
                &img,
                Rect::whole(w, hh),
                SigmaOwned::Plane(plane),
                1.4,
                BilateralParams::default(),
            );
            h = fnv(h, &out.data);
        }
        h
    });
}

#[test]
fn bindings_fit_the_design_binding_at_the_largest_rectangle() {
    // The largest compute rectangle (`docs/17-gpu.md` section 4): MAX_PHYS_AREA pixels, here
    // 1600 x 1562.
    let side = (MAX_PHYS_AREA as f64).sqrt() as usize;
    assert_max_binding_bytes(
        "bilateral",
        Rect::new(0, 0, side, side),
        bilateral::max_binding_bytes,
    );
}

#[test]
fn bad_inputs_are_refused_with_the_reason() {
    let Some(dev) = f32_device() else { return };
    let img = noisy_scene(20, 20, 1, 0.01);
    let ok = BilateralParams::default();
    let run = |core: Rect, sigma: SigmaOwned, scale: f32, p: BilateralParams| {
        let img = img.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                gpu_bilateral(ctx, &img, core, &sigma.as_input(), scale, &p)
            })
    };
    let whole = Rect::whole(20, 20);
    assert!(run(whole, SigmaOwned::Scalar(0.01), 1.0, ok).is_ok());
    for (what, result) in [
        (
            "radius above the maximum",
            run(
                whole,
                SigmaOwned::Scalar(0.01),
                1.0,
                BilateralParams {
                    radius_px: MAX_RADIUS_PX + 1,
                    ..ok
                },
            ),
        ),
        (
            "spatial sigma zero",
            run(
                whole,
                SigmaOwned::Scalar(0.01),
                1.0,
                BilateralParams {
                    spatial_sigma_px: 0.0,
                    ..ok
                },
            ),
        ),
        (
            "negative strength",
            run(
                whole,
                SigmaOwned::Scalar(0.01),
                1.0,
                BilateralParams {
                    strength: -1.0,
                    ..ok
                },
            ),
        ),
        (
            "NaN scale",
            run(whole, SigmaOwned::Scalar(0.01), f32::NAN, ok),
        ),
        (
            "core outside",
            run(Rect::new(10, 10, 20, 20), SigmaOwned::Scalar(0.01), 1.0, ok),
        ),
        (
            "empty core",
            run(Rect::new(0, 0, 0, 5), SigmaOwned::Scalar(0.01), 1.0, ok),
        ),
        (
            "short sigma plane",
            run(whole, SigmaOwned::Plane(vec![0.01; 19]), 1.0, ok),
        ),
    ] {
        let err = result.expect_err(what);
        assert!(
            matches!(err, ym_gpu::device::GpuError::InvalidInput(_)),
            "{what}: {err}"
        );
    }
}
