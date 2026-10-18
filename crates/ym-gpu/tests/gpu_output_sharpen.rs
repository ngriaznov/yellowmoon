//! The output sharpening kernel (`kernels::output_sharpen`, `shaders/output_sharpen.wgsl`)
//! against its CPU reference (`cpu::output_sharpen`, spec `docs/algorithms/04-output-sharpening.md`).
//!
//! # What is compared, and with which bound
//!
//! * Every vector of `tests/cpu_data/output_sharpening/` (the C++ oracle's output): the flat-field and
//!   zero-amount vectors must come back **bit-exact**; the others within the vector's own tolerance
//!   (1e-5), and the GPU is also compared with the CPU port (same bound).
//! * Seeded random frames (white noise in [-0.05, 3], a noisy busy scene, a photographic-like scene)
//!   with the seven parameter sets of the reference's self-test, sizes that are not multiples of the
//!   workgroup, tile size or halo, whole-frame and tiled. On noisy or photographic content the bound
//!   is [`NOISY_BOUND`] (1e-5, spec 5.4) on the **maximum** absolute error over all samples.
//! * Clean synthetic content with the halo limiter on: the reference counts window samples by exact
//!   comparison (spec 5.4), so values that are equal in exact arithmetic but one ulp apart after `f32`
//!   accumulation could flip a count. Since the tie tolerance (spec 3.5, default `2^-12`) the limiter
//!   is continuous and such values count alike on both paths, so these frames come within 1.5e-5; the
//!   bounds stay as they were. Isolated noiseless edges: the bound is on the 99.9th percentile
//!   ([`CLEAN_P999_BOUND`]). Periodic content (bars, checkerboards: thousands of reflection ties): the
//!   maximum is capped by [`CLEAN_MAX_BOUND`] and the number of samples over [`NOISY_BOUND`] is
//!   printed; the same frames without the limiter meet the tight bound.
//! * A large frame (2048 x 1365) through the tiling path with the screen preset at full size and at
//!   the scale of a 2048 px export from 6000 px.
//! * Tiles equal the whole frame **bit for bit** on the GPU (the tiling contract of spec 5.2); two runs
//!   are bit-identical; the result does not depend on the device's storage mode (no f16 here).
//! * Properties of spec 6.2 on the GPU (added in review): a collapsed kernel is a bit-exact no-op,
//!   `f(2x) = 2 f(x)` and `f(x/4) = f(x)/4` bit for bit without a threshold, channel ratios survive,
//!   NaN and Inf samples do not panic and are classified as on the CPU (software adapters), and a frame
//!   side beyond the dispatch limit is refused with `InvalidInput` instead of a wgpu panic.
//!
//! # Measured (Mesa lavapipe, llvmpipe on LLVM 20.1.2)
//!
//! Maximum absolute error against the CPU port: the 16 vectors 7.15e-7 (tie tolerance 0; asserted
//! within 2e-6, and within each vector's own tolerance, 0 for the bit-exact ones); noisy and
//! photographic frames 9.3e-6 with the print preset's long kernels, the screen preset under 4e-6
//! (asserted 1e-5; 1.43e-6 with exact comparisons, the soft counts amplify the `f32` sums); white
//! noise 4.9e-6 relative (asserted 1e-5); tiny frames 1.13e-6; flip and transpose 6.8e-6; isolated
//! edges, the diagonal too, 2.4e-7; the 2048 x 1365 frame without the limiter 9.5e-7, with it 2.3e-6,
//! no sample over 1e-5 (4.3e-5 on 26 of 8.4 million samples with exact comparisons; asserted 99.9th
//! percentile 1e-5 and maximum 2e-4); bars and checkerboards with the limiter 1.5e-5 (1.5e-4 to
//! 8.3e-3 with exact comparisons). With the tie tolerance at 0 the kernel is bit-identical to the
//! C++ reference's
//! `Accumulation::F32` mode on all 37 cases checked with a scratch driver fed by the ignored test
//! `dump_cases_for_the_cpp_oracle` (see
//! `docs/algorithms/04-output-sharpening.md` section 10), so every difference from the CPU port is the
//! `f32` against `f64` accumulation.
//!
//! # CFA phases
//!
//! Not applicable: this is an RGB stage after the demosaic (spec 4.8). The geometric invariants are
//! checked instead: flip and transpose equivariance, and frame sizes from 1 x 1 to well past the
//! kernel length (the mirror semantics of spec 4.1).
//!
//! # Non-bit-exact parts
//!
//! The CPU accumulates convolutions, window sums and the envelope formula in `f64`; WGSL has no `f64`,
//! so the GPU accumulates in `f32` with the taps rounded to `f32` (the model of the C++ reference's
//! `Accumulation::F32`). Square roots and divisions are the shader's own integer routines, correctly
//! rounded on every device ([`the_shader_square_root_and_division_are_correctly_rounded_on_the_device`]).
//! The luma is the one quantity that must be bit-identical, because the limiter compares it exactly:
//! [`luma_has_the_cpu_bits`] checks that on every adapter.

mod common;

use std::path::{Path, PathBuf};

use common::{Rng, TestGpu, gpus};
use serde_json::Value;
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::output_sharpen::{ExtraStage, Method, Mode, OutputSharpen, scale_for_edges};
use ym_gpu::device::{GpuContext, GpuError};
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::release_tables;
use ym_gpu::kernels::output_sharpen::{PreparedSharpen, SharpenTrace};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

/// Spec 5.4: noisy or photographic content, maximum absolute error.
const NOISY_BOUND: f64 = 1e-5;
/// Spec 5.4: clean synthetic content with the limiter on, 99.9th percentile of the absolute error.
const CLEAN_P999_BOUND: f64 = 1e-5;
/// Clean synthetic content with the limiter on: the maximum error of a flipped rank count (spec 5.4
/// says "up to about 1e-2").
const CLEAN_MAX_BOUND: f64 = 2e-2;
/// A noiseless diagonal edge with the limiter on, 99.9th percentile (measured up to 2.7e-4, with the
/// long kernels of the print distance 30000 and method 2).
const DIAGONAL_P999_BOUND: f64 = 1e-3;
/// A 2.8 MP photographic-like frame with the limiter on: the maximum error. Among 70 million window
/// pairs a few thousand differ by less than the `f32` rounding of the accumulations, so a handful of
/// rank counts flipped even with noise: 4.3e-5 on 26 of 8.4 million samples with exact comparisons,
/// 2.3e-6 with the tie tolerance (the limiter is continuous).
const LARGE_FRAME_MAX_BOUND: f64 = 2e-4;

// ---- Metal (hardware adapters) ---------------------------------------------------------------------
//
// Metal compiles with fast-math and wgpu has no switch for it. Hardware adapters first had looser
// bounds of their own (twice the Metal baseline at d629dbe); the Metal run of b6f2e4b then failed four
// of these tests on later cases (vectors 1.36e-5, noisy p99.9 1.5e-4, the flipped image 8.6e-4, the
// 2048 px export 1.96e-3). None of that was inherent to `f32`: Metal's `sqrt` left a third of the
// lumas 1 or 2 ulp off, its division is the dividend times an approximate reciprocal, and it fused the
// blurs' products into multiply-adds. Each moved a value the limiter compares by an ulp, and at a
// near-tie a rank count flipped. The kernel now pins every such rounding (`opaque`) and computes its
// square roots and quotients in integer arithmetic, correctly rounded (`output_sharpen.wgsl`), so a
// hardware adapter computes lavapipe's bits: a C port of the shader compiled with clang `-O2`/`-O3
// -ffast-math -ffp-contract=fast -mfma`, with a square root and a reciprocal 1 to 2 ulp off, gives the
// bits of the strict build on every case of these tests (`docs/17-gpu.md` section 13). So there is
// no hardware column any more: every adapter meets the bounds above. On hardware the tests still
// write their numbers to the CI log (`ci_note`).

fn hardware(dev: &TestGpu) -> bool {
    !dev.gpu.info().is_software
}

/// A note that reaches the CI log even when the test passes: written to the process's own stderr
/// (the harness captures `eprintln!`, not this) and, on GitHub Actions, as an annotation.
fn ci_note(text: &str) {
    use std::io::Write as _;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "[output sharpening] {text}");
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::notice title=Output sharpening::{text}");
    }
}

// ------------------------------------------------------------------------------------------------
// Running the kernel
// ------------------------------------------------------------------------------------------------

fn run_gpu<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("GPU job")
}

fn gpu_sharpen(dev: &TestGpu, p: &OutputSharpen, scale: f32, img: &RgbImage) -> RgbImage {
    let (p, img) = (p.clone(), img.clone());
    run_gpu(dev, move |ctx| {
        PreparedSharpen::new(ctx, &p, scale)?.run(ctx, &img)
    })
}

fn gpu_traced(
    dev: &TestGpu,
    p: &OutputSharpen,
    scale: f32,
    img: &RgbImage,
) -> (RgbImage, SharpenTrace) {
    let (p, img) = (p.clone(), img.clone());
    run_gpu(dev, move |ctx| {
        PreparedSharpen::new(ctx, &p, scale)?.run_traced(ctx, &img)
    })
}

/// Sharpens tile by tile (cores of `tile_px`, the exact halo) and pastes the cores.
fn gpu_sharpen_tiled(
    dev: &TestGpu,
    p: &OutputSharpen,
    scale: f32,
    img: &RgbImage,
    tile_px: usize,
    depth: usize,
) -> RgbImage {
    let (p, img) = (p.clone(), img.clone());
    run_gpu(dev, move |ctx| {
        let prepared = PreparedSharpen::new(ctx, &p, scale)?;
        let plan = prepared.tile_plan(img.width, img.height, tile_px);
        let mut out = RgbImage::new(img.width, img.height);
        let n = out.plane_len();
        let width = img.width;
        let mut tiles = 0usize;
        prepared.run_tiled(
            ctx,
            &img,
            &plan,
            depth,
            &CancelToken::new(),
            &mut |tile, done| {
                let PixelData::F32Planar(v) = done.data else {
                    panic!("expected f32 planes");
                };
                let c = tile.core;
                assert_eq!((done.width, done.height), (c.w, c.h));
                for ch in 0..3 {
                    for y in 0..c.h {
                        let dst = ch * n + (c.y + y) * width + c.x;
                        out.data[dst..dst + c.w]
                            .copy_from_slice(&v[ch * c.w * c.h + y * c.w..][..c.w]);
                    }
                }
                tiles += 1;
                Ok(())
            },
        )?;
        assert_eq!(tiles, plan.tiles().len());
        Ok(out)
    })
}

// ------------------------------------------------------------------------------------------------
// Metrics
// ------------------------------------------------------------------------------------------------

/// Error statistics of `a` against the reference `b`.
#[derive(Clone, Copy, Debug)]
struct ErrStats {
    max: f64,
    p999: f64,
    /// Samples over [`NOISY_BOUND`].
    over: usize,
    total: usize,
}

impl ErrStats {
    fn between(a: &[f32], b: &[f32]) -> ErrStats {
        assert_eq!(a.len(), b.len());
        // A sample that is NaN on either side counts as an infinite error: `f64::max` and the
        // comparisons below would otherwise skip it, and a stale (poisoned) read would pass.
        let mut d: Vec<f64> = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let e = (f64::from(x) - f64::from(y)).abs();
                if e.is_nan() { f64::INFINITY } else { e }
            })
            .collect();
        let over = d.iter().filter(|&&v| v > NOISY_BOUND).count();
        let max = d.iter().copied().fold(0.0, f64::max);
        d.sort_by(f64::total_cmp);
        let rank = ((0.999 * d.len() as f64).ceil() as usize).clamp(1, d.len());
        ErrStats {
            max,
            p999: d[rank - 1],
            over,
            total: d.len(),
        }
    }

    fn of_images(a: &RgbImage, b: &RgbImage) -> ErrStats {
        assert_eq!((a.width, a.height), (b.width, b.height));
        ErrStats::between(&a.data, &b.data)
    }

    fn line(&self) -> String {
        format!(
            "max {:.3e}, p99.9 {:.3e}, {} of {} samples over {:.0e}",
            self.max, self.p999, self.over, self.total, NOISY_BOUND
        )
    }
}

fn bit_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn all_finite(img: &RgbImage) -> bool {
    img.data.iter().all(|v| v.is_finite())
}

// ------------------------------------------------------------------------------------------------
// Test images
// ------------------------------------------------------------------------------------------------

/// Gaussian noise from additions only (twelve uniforms, mean 0, variance 1).
fn gauss(rng: &mut Rng) -> f64 {
    (0..12).map(|_| rng.unit()).sum::<f64>() - 6.0
}

fn flat(w: usize, h: usize, rgb: [f32; 3]) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.set_pixel(x, y, rgb);
        }
    }
    img
}

fn add_noise(img: &mut RgbImage, sigma: f64, seed: u64) {
    let mut rng = Rng(seed);
    for v in &mut img.data {
        *v = (f64::from(*v) + sigma * gauss(&mut rng)) as f32;
    }
}

fn random_image(w: usize, h: usize, lo: f32, hi: f32, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut img = RgbImage::new(w, h);
    for v in &mut img.data {
        *v = rng.range(f64::from(lo), f64::from(hi)) as f32;
    }
    img
}

/// Edges, a bar pattern and a diagonal line, without noise (the reference's busy scene).
fn busy_scene(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut v = if x * 3 < w {
                0.08f32
            } else if x * 3 < 2 * w {
                0.42
            } else {
                0.2
            };
            if y > h / 2 && (x / 2) % 2 == 0 {
                v += 0.25;
            }
            if x + y == w / 2 || x + y == w / 2 + 1 {
                v = 0.9;
            }
            img.set_pixel(x, y, [v * 1.1, v, v * 0.7]);
        }
    }
    img
}

/// The busy scene with noise of sigma 0.004.
fn busy(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = busy_scene(w, h);
    add_noise(&mut img, 0.004, seed);
    img
}

/// A photographic-like frame: smooth gradients, soft and hard edges, fine texture, a bright disk with
/// values above 1, dark corners, coloured regions and sensor-like noise (sigma 0.004 plus shot noise).
fn photo_like(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Rng(seed);
    let (cx, cy) = (w as f64 * 0.62, h as f64 * 0.38);
    let radius = (w.min(h) as f64) * 0.12;
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            // Low-frequency light with a hard horizon and a soft shadow.
            let mut v = 0.04 + 0.5 * fy * fy + 0.1 * (6.0 * fx).sin().abs();
            if fy > 0.55 {
                v += 0.06;
            }
            let soft = 1.0 / (1.0 + (-(fx - 0.3) * 40.0).exp());
            v += 0.12 * soft;
            // Fine texture (foliage-like) in the lower left.
            if fx < 0.4 && fy > 0.6 {
                v += 0.05 * ((x * 7 + y * 3) as f64 * 0.9).sin() * ((x * 5) as f64 * 0.37).cos();
            }
            // A bright disk (highlights above white) with a ring.
            let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            if d < radius {
                v += 2.2;
            } else if d < radius * 1.15 {
                v += 0.3;
            }
            let noise = (0.004 + 0.02 * v.max(0.0).sqrt() * 0.2) * gauss(&mut rng);
            let tint = [1.1, 1.0, 0.75 + 0.2 * fx];
            let rgb = [
                ((v + noise) * tint[0]) as f32,
                ((v + noise) * tint[1]) as f32,
                ((v + noise) * tint[2]) as f32,
            ];
            img.set_pixel(x, y, rgb);
        }
    }
    img
}

/// A checkerboard of 1 px cells with a diagonal edge and a 2 px stripe field, without noise.
fn checker_and_diagonals(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 3 {
                if (x + y) % 2 == 0 { 0.7f32 } else { 0.1 }
            } else if x < 2 * w / 3 {
                if (x / 2) % 2 == 0 { 0.6 } else { 0.2 }
            } else if x + y > w + h / 3 {
                0.5
            } else {
                0.15
            };
            img.set_pixel(x, y, [v, v * 0.9, v * 0.8]);
        }
    }
    img
}

fn transpose(img: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(img.height, img.width);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set_pixel(y, x, img.pixel(x, y));
        }
    }
    out
}

fn flip_x(img: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set_pixel(img.width - 1 - x, y, img.pixel(x, y));
        }
    }
    out
}

/// The reference's seven parameter sets of its self-test, with the output scale of each.
fn param_sets() -> Vec<(&'static str, OutputSharpen, f32)> {
    let mut v = Vec::new();
    let mut p = OutputSharpen::screen();
    v.push(("screen preset", p.clone(), 1.0)); // method 1, halo 60
    p.method = Method::HardThreshold;
    p.radius_px = 2.5;
    p.halo_pct = 0.0;
    p.threshold_levels = 2.0;
    v.push(("method 0, R 2.5, no halo", p.clone(), 1.0));
    p.method = Method::GradientRamp;
    p.halo_pct = 60.0;
    v.push(("method 2, R 2.5, halo", p.clone(), 1.0));
    p.method = Method::SmoothGain;
    v.push(("downscaled s 0.37", p.clone(), 0.37));
    v.push(("print preset", OutputSharpen::print(), 1.0));
    let mut q = OutputSharpen::print();
    q.print_distance = 30000.0;
    q.method = Method::GradientRamp;
    q.halo_pct = 40.0;
    v.push(("print, long kernels, method 2, halo 40", q, 1.0));
    let mut e = OutputSharpen::screen();
    let mut c = ExtraStage {
        method: Method::HardThreshold,
        auto_method: true,
        amount_pct: 150.0,
        radius_px: 0.6,
        threshold_levels: 1.0,
        halo_pct: 20.0,
    };
    e.extra_stages.push(c);
    c.method = Method::GradientRamp;
    c.auto_method = false;
    c.radius_px = 2.0;
    c.halo_pct = 0.0;
    e.extra_stages.push(c);
    v.push(("three stages", e, 1.0));
    v
}

/// The same parameters with the halo limiter off in every stage.
fn without_halo(p: &OutputSharpen) -> OutputSharpen {
    let mut q = p.clone();
    q.halo_pct = 0.0;
    for e in &mut q.extra_stages {
        e.halo_pct = 0.0;
    }
    q
}

// ------------------------------------------------------------------------------------------------
// Vectors
// ------------------------------------------------------------------------------------------------

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/output_sharpening")
}

fn vectors() -> Vec<(String, Value)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(vector_dir())
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("vector file");
            let v: Value = serde_json::from_str(&text).expect("vector json");
            (
                p.file_stem().expect("stem").to_string_lossy().into_owned(),
                v,
            )
        })
        .collect()
}

fn f32_of(v: &Value, key: &str) -> f32 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}")) as f32
}

fn method_of(v: &Value) -> Method {
    Method::from_id(v["method_id"].as_u64().expect("method_id") as u32).expect("method")
}

fn params_of(v: &Value) -> (OutputSharpen, f32) {
    let p = &v["params"];
    let mode = match p["mode"].as_str().expect("mode") {
        "screen" => Mode::Screen,
        "print" => Mode::Print,
        other => panic!("mode {other}"),
    };
    let lw = p["luma_weights"].as_array().expect("luma weights");
    let extra_stages = p["extra_stages"]
        .as_array()
        .expect("extra stages")
        .iter()
        .map(|e| ExtraStage {
            method: method_of(e),
            auto_method: e["auto_method"].as_bool().expect("auto_method"),
            amount_pct: f32_of(e, "amount_pct"),
            radius_px: f32_of(e, "radius_px"),
            threshold_levels: f32_of(e, "threshold_levels"),
            halo_pct: f32_of(e, "halo_pct"),
        })
        .collect();
    (
        OutputSharpen {
            mode,
            method: method_of(p),
            amount_pct: f32_of(p, "amount_pct"),
            radius_px: f32_of(p, "radius_px"),
            threshold_levels: f32_of(p, "threshold_levels"),
            halo_pct: f32_of(p, "halo_pct"),
            print_distance: f32_of(p, "print_distance"),
            luma_weights: [
                lw[0].as_f64().expect("w") as f32,
                lw[1].as_f64().expect("w") as f32,
                lw[2].as_f64().expect("w") as f32,
            ],
            extra_stages,
            // The reference compares exactly; the tie tolerance is ours (spec 7.1 item 5).
            halo_tie_tolerance: 0.0,
        },
        f32_of(p, "scale"),
    )
}

fn image_of(v: &Value) -> RgbImage {
    assert_eq!(v["layout"].as_str(), Some("planar_rgb"));
    let w = v["width"].as_u64().expect("width") as usize;
    let h = v["height"].as_u64().expect("height") as usize;
    let data: Vec<f32> = v["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|x| x.as_f64().expect("sample") as f32)
        .collect();
    RgbImage::from_planar(w, h, data).expect("planar image")
}

#[test]
fn vectors_match_the_oracle_and_the_cpu_port() {
    let all = vectors();
    assert_eq!(all.len(), 16, "the package has 16 vectors");
    for (di, dev) in gpus().iter().enumerate() {
        let mut worst_vs_expected = 0.0f64;
        let mut worst_vs_cpu = 0.0f64;
        for (name, v) in &all {
            let (os, scale) = params_of(v);
            let input = image_of(&v["input"]);
            let expected = image_of(&v["expected"]);
            let tol = v["tolerance"].as_f64().expect("tolerance");
            let cpu = os.apply(&input, scale).expect("cpu apply");
            let out = gpu_sharpen(dev, &os, scale, &input);
            assert!(all_finite(&out), "{name}: non-finite output");
            let vs_expected = ErrStats::of_images(&out, &expected);
            let vs_cpu = ErrStats::of_images(&out, &cpu);
            if di == 0 {
                eprintln!(
                    "{name:<34} vs oracle: {} | vs cpu port: max {:.3e}",
                    vs_expected.line(),
                    vs_cpu.max
                );
            }
            if tol == 0.0 {
                // Flat fields and amount 0: the difference form gives exact zeros on the GPU too.
                assert!(bit_equal(&out, &expected), "{name}: must be bit-exact");
                assert!(
                    bit_equal(&out, &cpu),
                    "{name}: must equal the CPU port exactly"
                );
            } else {
                assert!(
                    vs_expected.max <= tol,
                    "{name}: error {:e} against the oracle is above the tolerance {tol:e}",
                    vs_expected.max
                );
                assert!(
                    vs_cpu.max <= tol,
                    "{name}: error {:e} against the CPU port is above {tol:e}",
                    vs_cpu.max
                );
            }
            worst_vs_expected = worst_vs_expected.max(vs_expected.max);
            worst_vs_cpu = worst_vs_cpu.max(vs_cpu.max);
        }
        eprintln!(
            "[{}] vectors: max |gpu - oracle| {worst_vs_expected:.3e}, max |gpu - cpu port| {worst_vs_cpu:.3e}",
            dev.mode.name()
        );
        if hardware(dev) {
            ci_note(&format!(
                "hardware vectors: max |gpu - oracle| {worst_vs_expected:.3e}, max |gpu - cpu port| {worst_vs_cpu:.3e} (lavapipe 7.15e-7; Metal at b6f2e4b 1.36e-5)"
            ));
        }
        assert!(
            worst_vs_expected <= 2e-6,
            "the 16 vectors must stay within 2e-6 on the GPU, got {worst_vs_expected:e}"
        );
    }
}

// ------------------------------------------------------------------------------------------------
// The luma, bit for bit
// ------------------------------------------------------------------------------------------------

/// The CPU's luma, in the operation order of spec 3.1.
fn luma_of(w: &[f32; 3], r: f32, g: f32, b: f32) -> f32 {
    (((w[0] * r) * r + (w[1] * g) * g) + (w[2] * b) * b).sqrt()
}

#[test]
fn luma_has_the_cpu_bits() {
    // The limiter compares luma values for equality and order, so the GPU's luma must be the CPU's
    // `f32` luma, every bit, on every adapter: the products and the partial sum are hidden from the
    // optimiser and the square root is the shader's integer one (Metal's own `sqrt` left 27 to 37 % of
    // the lumas 1 or 2 ulp off at b6f2e4b).
    let images = [
        random_image(37, 29, -0.05, 3.0, 11),
        random_image(64, 16, 0.0, 1.0, 12),
        busy(40, 33, 13),
        photo_like(61, 47, 14),
    ];
    let weights = [
        [0.299f32, 0.587, 0.114],
        [0.212_639, 0.715_168_7, 0.072_192_3],
        [0.262_700_2, 0.677_998_1, 0.059_301_7],
    ];
    for dev in gpus().iter().take(1) {
        let software = dev.gpu.info().is_software;
        for img in &images {
            for w in weights {
                let mut p = OutputSharpen::screen();
                p.luma_weights = w;
                let (_out, trace) = gpu_traced(dev, &p, 1.0, img);
                let n = img.plane_len();
                assert_eq!(trace.luma.len(), n);
                let mut differ = 0usize;
                let mut worst_ulps = 0u32;
                for (i, &gpu) in trace.luma.iter().enumerate() {
                    let cpu = luma_of(&w, img.data[i], img.data[n + i], img.data[2 * n + i]);
                    if gpu.to_bits() != cpu.to_bits() {
                        differ += 1;
                        let ulps = (i64::from(gpu.to_bits()) - i64::from(cpu.to_bits()))
                            .unsigned_abs() as u32;
                        worst_ulps = worst_ulps.max(ulps);
                    }
                }
                eprintln!(
                    "luma {}x{} weights {w:?}: {differ} of {n} pixels differ (worst {worst_ulps} ulp)",
                    img.width, img.height
                );
                if !software {
                    ci_note(&format!(
                        "hardware luma {}x{} weights {w:?}: {differ} of {n} pixels differ from the CPU's bits (worst {worst_ulps} ulp)",
                        img.width, img.height
                    ));
                }
                assert_eq!(
                    differ, 0,
                    "the luma must have the CPU's bits (worst {worst_ulps} ulp)"
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Random frames against the CPU port
// ------------------------------------------------------------------------------------------------

#[test]
fn noisy_and_photographic_frames_match_the_cpu_port_within_1e_5() {
    let Some(dev) = gpus().first() else { return };
    let frames: Vec<(&str, RgbImage)> = vec![
        ("busy+noise 71x66", busy(71, 66, 21)),
        ("busy+noise 130x97", busy(130, 97, 22)),
        ("photo-like 150x113", photo_like(150, 113, 23)),
        ("photo-like 33x20", photo_like(33, 20, 24)),
    ];
    let mut worst = 0.0f64;
    let mut worst_p999 = 0.0f64;
    for (name, img) in &frames {
        for (set, p, scale) in param_sets() {
            let cpu = p.apply(img, scale).expect("cpu");
            let gpu = gpu_sharpen(dev, &p, scale, img);
            assert!(all_finite(&gpu));
            let e = ErrStats::of_images(&gpu, &cpu);
            eprintln!("{name:<20} {set:<40} {}", e.line());
            if hardware(dev) && e.max > NOISY_BOUND {
                ci_note(&format!("hardware {name} / {set}: {}", e.line()));
            }
            assert!(
                e.max <= NOISY_BOUND,
                "{name} / {set}: max error {:e} above {NOISY_BOUND:e} ({})",
                e.max,
                e.line()
            );
            worst = worst.max(e.max);
            worst_p999 = worst_p999.max(e.p999);
        }
    }
    eprintln!(
        "noisy and photographic frames: worst max error {worst:.3e}, worst p99.9 {worst_p999:.3e}"
    );
    if hardware(dev) {
        ci_note(&format!(
            "hardware noisy and photographic frames: worst max error {worst:.3e}, worst p99.9 {worst_p999:.3e} (lavapipe 1.43e-6)"
        ));
    }
}

#[test]
fn white_noise_frames_match_within_a_relative_1e_5() {
    // Uniform white noise in [-0.05, 3] is far harder than a photograph: the luma jumps by a factor
    // of 100 between neighbours, the ratio hits its cap of 16 and outputs reach tens. The bound is
    // relative to max(1, |value|).
    let Some(dev) = gpus().first() else { return };
    let img = random_image(73, 59, -0.05, 3.0, 31);
    let mut worst = 0.0f64;
    for (set, p, scale) in param_sets() {
        let cpu = p.apply(&img, scale).expect("cpu");
        let gpu = gpu_sharpen(dev, &p, scale, &img);
        assert!(all_finite(&gpu));
        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        for (&g, &c) in gpu.data.iter().zip(&cpu.data) {
            let d = (f64::from(g) - f64::from(c)).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / f64::from(c).abs().max(1.0));
        }
        eprintln!("white noise {set:<40} max abs {max_abs:.3e}, max rel {max_rel:.3e}");
        assert!(max_rel <= NOISY_BOUND, "{set}: relative error {max_rel:e}");
        worst = worst.max(max_rel);
    }
    eprintln!("white noise: worst relative error {worst:.3e}");
}

/// A hard step edge at `x0` (vertical), `y0` (horizontal) or along `x + y = c` (diagonal), noiseless.
fn single_edge(w: usize, h: usize, kind: &str) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let high = match kind {
                "vertical" => x >= w / 3,
                "horizontal" => y >= h * 2 / 5,
                _ => x + y >= (w + h) * 2 / 5,
            };
            let v = if high { 0.5f32 } else { 0.1 };
            img.set_pixel(x, y, [v * 1.05, v, v * 0.8]);
        }
    }
    img
}

#[test]
fn isolated_clean_edges_with_the_limiter_on_stay_within_the_percentile_bound() {
    // Spec 5.4: on clean synthetic content with the limiter on, compare the 99.9th percentile and
    // report the maximum. A single noiseless vertical or horizontal edge has exact ties on its plateaus
    // (the difference form keeps them exact on the GPU) and no reflection ties across the edge
    // direction, so no rank count moves and the tight bound holds. A diagonal edge is one of the
    // examples the spec names: transposed neighbourhoods tie in exact arithmetic, and the f32 passes
    // (rows, then columns) round them differently. Measured: 0.5 % of the samples of the screen
    // preset move, by at most 3.7e-5; with the long kernels of print distance 30000 and method 2,
    // 9 % move, by at most 3.7e-4.
    let Some(dev) = gpus().first() else { return };
    for kind in ["vertical", "horizontal", "diagonal"] {
        let img = single_edge(64, 56, kind);
        for (set, p, scale) in param_sets() {
            let cpu = p.apply(&img, scale).expect("cpu");
            let gpu = gpu_sharpen(dev, &p, scale, &img);
            let e = ErrStats::of_images(&gpu, &cpu);
            eprintln!("{kind:<10} edge {set:<40} {}", e.line());
            if kind == "diagonal" {
                assert!(
                    e.p999 <= DIAGONAL_P999_BOUND,
                    "{kind} / {set}: p99.9 {:e}",
                    e.p999
                );
                assert!(e.max <= CLEAN_MAX_BOUND, "{kind} / {set}: max {:e}", e.max);
            } else {
                assert!(e.max <= CLEAN_P999_BOUND, "{kind} / {set}: max {:e}", e.max);
            }
        }
    }
}

#[test]
fn periodic_clean_content_moves_rank_counts_only_inside_the_documented_bound() {
    // Noiseless bars and checkerboards are the worst case of spec 5.4: thousands of window samples are
    // equal in exact arithmetic (reflections of each other), the f64 path rounds them to equal f32
    // values and the f32 path does not. With exact comparisons (the reference) the limiter's counts
    // then differ: 0.3 % to 19 % of the pixels have a different rank count, every sample over 1e-5
    // lies within 8 px of such a pixel, and the largest error is 8.3e-3 (lavapipe against the C++
    // oracle, to whose f32-accumulation mode the GPU is bit-identical). With the tie tolerance the
    // limiter is continuous and the ties hold on both paths: 1.5e-5 at most (the bars with the screen
    // preset). Without the limiter the same content matches within 1e-5.
    let Some(dev) = gpus().first() else { return };
    let frames: Vec<(&str, RgbImage)> = vec![
        ("bars+diagonal 71x66", busy_scene(71, 66)),
        (
            "checker+stripes+diagonal 90x64",
            checker_and_diagonals(90, 64),
        ),
    ];
    for (name, img) in &frames {
        for (set, p, scale) in param_sets() {
            let limiter_on = p.halo_pct > 0.0 || p.extra_stages.iter().any(|e| e.halo_pct > 0.0);
            let cpu = p.apply(img, scale).expect("cpu");
            let gpu = gpu_sharpen(dev, &p, scale, img);
            assert!(all_finite(&gpu));
            let e = ErrStats::of_images(&gpu, &cpu);
            eprintln!(
                "{name:<30} {set:<40} limiter {:<3} {}",
                if limiter_on { "on" } else { "off" },
                e.line()
            );
            if limiter_on {
                assert!(e.max <= CLEAN_MAX_BOUND, "{name} / {set}: max {:e}", e.max);
            } else {
                // No limiter, no discontinuity: the tight bound holds even on ties.
                assert!(
                    e.max <= NOISY_BOUND,
                    "{name} / {set}: {:e} without a limiter",
                    e.max
                );
            }
            // The same frame with the limiter off in every stage is within the tight bound.
            let q = without_halo(&p);
            let cpu_off = q.apply(img, scale).expect("cpu");
            let gpu_off = gpu_sharpen(dev, &q, scale, img);
            let off = ErrStats::of_images(&gpu_off, &cpu_off);
            assert!(
                off.max <= NOISY_BOUND,
                "{name} / {set}: limiter off, max error {:e}",
                off.max
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Development aid: cases for the C++ oracle
// ------------------------------------------------------------------------------------------------

/// The parameters in the `key value` form the scratch driver of the C++ reference reads.
fn params_txt(p: &OutputSharpen, scale: f32) -> String {
    let mut s = format!(
        "mode {} method {} amount {:?} radius {:?} threshold {:?} halo {:?} print_distance {:?} scale {:?} lw {:?} {:?} {:?}\n",
        p.mode as u32,
        p.method.id(),
        p.amount_pct,
        p.radius_px,
        p.threshold_levels,
        p.halo_pct,
        p.print_distance,
        scale,
        p.luma_weights[0],
        p.luma_weights[1],
        p.luma_weights[2]
    );
    for e in &p.extra_stages {
        s.push_str(&format!(
            "extra {} {} {:?} {:?} {:?} {:?}\n",
            e.method.id(),
            u32::from(e.auto_method),
            e.amount_pct,
            e.radius_px,
            e.threshold_levels,
            e.halo_pct
        ));
    }
    s
}

fn write_f32s(path: &Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write");
}

#[test]
#[ignore = "development aid: with YM_SHARPEN_DUMP=<dir> writes cases (input, parameters, GPU and CPU outputs, GPU trace) for a C++ oracle driver"]
fn dump_cases_for_the_cpp_oracle() {
    let Ok(root) = std::env::var("YM_SHARPEN_DUMP") else {
        eprintln!("YM_SHARPEN_DUMP is not set; nothing to do");
        return;
    };
    let Some(dev) = gpus().first() else { return };
    let frames: Vec<(&str, RgbImage)> = vec![
        ("busy_scene", busy_scene(71, 66)),
        ("busy_noisy", busy(71, 66, 21)),
        ("checker_diag", checker_and_diagonals(90, 64)),
        ("photo", photo_like(150, 113, 23)),
        ("white_noise", random_image(73, 59, -0.05, 3.0, 31)),
    ];
    let mut sets = param_sets();
    // With YM_SHARPEN_DUMP_LARGE set, also the 2048 x 1365 photographic frame (screen preset at full
    // size and at the scale of a 2048 px export from 6000 px), 33 MB of input per case.
    let mut frames = frames;
    if std::env::var_os("YM_SHARPEN_DUMP_LARGE").is_some() {
        frames = vec![("photo_large", photo_like(2048, 1365, 71))];
        let screen = OutputSharpen::screen();
        sets = vec![
            ("screen preset", screen.clone(), 1.0),
            (
                "screen, 2048 from 6000",
                screen,
                scale_for_edges(2048, 6000),
            ),
        ];
    }
    for (fname, img) in &frames {
        for (si, (set, p, scale)) in sets.iter().enumerate() {
            let dir = Path::new(&root).join(format!("{fname}__{si}"));
            std::fs::create_dir_all(&dir).expect("dir");
            std::fs::write(dir.join("name.txt"), format!("{fname} / {set}\n")).expect("name");
            std::fs::write(dir.join("params.txt"), params_txt(p, *scale)).expect("params");
            let mut input = Vec::new();
            input.extend_from_slice(&(img.width as u32).to_le_bytes());
            input.extend_from_slice(&(img.height as u32).to_le_bytes());
            input.extend(img.data.iter().flat_map(|x| x.to_le_bytes()));
            std::fs::write(dir.join("input.bin"), input).expect("input");
            let cpu = p.apply(img, *scale).expect("cpu");
            write_f32s(&dir.join("cpu_out.bin"), &cpu.data);
            let (gpu, trace) = gpu_traced(dev, p, *scale, img);
            write_f32s(&dir.join("gpu_out.bin"), &gpu.data);
            let mut t = trace.luma.clone();
            t.extend_from_slice(&trace.free_delta);
            t.extend_from_slice(&trace.halo_delta);
            write_f32s(&dir.join("gpu_trace.bin"), &t);
        }
    }
    eprintln!("wrote {} cases to {root}", frames.len() * sets.len());
}

// ------------------------------------------------------------------------------------------------
// Tiling, determinism, storage modes
// ------------------------------------------------------------------------------------------------

#[test]
fn tiles_of_core_plus_halo_match_the_whole_frame_bit_for_bit() {
    // The tiling contract of spec 5.2 on the GPU: the arithmetic of a pixel depends only on pixels
    // within the receptive field, so any tile size, with any number of tiles in flight, gives the
    // whole-frame bits. Tile sizes 16 and 20 are smaller than the largest halos (21 and 41 px), so
    // their compute rectangles are mostly halo; 34 and 50 do not divide the frame.
    let Some(dev) = gpus().first() else { return };
    let img = busy(71, 66, 41);
    for (set, p, scale) in param_sets() {
        let whole = gpu_sharpen(dev, &p, scale, &img);
        for (tile_px, depth) in [(16, 1), (20, 3), (34, 2), (50, 1), (200, 4)] {
            let tiled = gpu_sharpen_tiled(dev, &p, scale, &img, tile_px, depth);
            assert!(
                bit_equal(&whole, &tiled),
                "{set}: {tile_px} px tiles (depth {depth}) differ from the whole frame by {:e}",
                ErrStats::of_images(&whole, &tiled).max
            );
        }
    }
}

#[test]
fn tiles_match_the_whole_frame_on_clean_periodic_content_too() {
    // The limiter's rank counts are the most fragile part; tiles still agree bit for bit.
    let Some(dev) = gpus().first() else { return };
    let img = checker_and_diagonals(90, 64);
    for (set, p, scale) in param_sets() {
        let whole = gpu_sharpen(dev, &p, scale, &img);
        let tiled = gpu_sharpen_tiled(dev, &p, scale, &img, 24, 2);
        assert!(bit_equal(&whole, &tiled), "{set}: tiled differs from whole");
    }
}

#[test]
fn two_runs_are_bit_identical_and_the_storage_mode_does_not_matter() {
    // No atomics, no reductions, fixed sum orders: a rerun gives the same bits. The kernel keeps
    // every plane in f32 whatever the device's storage mode, so devices of the three modes agree too.
    let img = photo_like(97, 61, 51);
    let sets = param_sets();
    for (set, p, scale) in &sets {
        let mut first: Option<RgbImage> = None;
        for dev in gpus() {
            let a = gpu_sharpen(dev, p, *scale, &img);
            let b = gpu_sharpen(dev, p, *scale, &img);
            assert!(
                bit_equal(&a, &b),
                "{set}: two runs differ on {}",
                dev.mode.name()
            );
            match &first {
                None => first = Some(a),
                Some(f) => assert!(
                    bit_equal(f, &a),
                    "{set}: {} differs from the first device",
                    dev.mode.name()
                ),
            }
        }
    }
}

#[test]
fn flip_and_transpose_equivariance_within_1e_5() {
    // The CPU port gives 2.4e-7 and 1.2e-7 (spec 6.2); the passes run rows first, so a transposed
    // image sums in a different order and the f32 results differ in the last bits only.
    let Some(dev) = gpus().first() else { return };
    let img = busy(53, 41, 61);
    for (set, p, scale) in param_sets() {
        let out = gpu_sharpen(dev, &p, scale, &img);
        let flipped = gpu_sharpen(dev, &p, scale, &flip_x(&img));
        let transposed = gpu_sharpen(dev, &p, scale, &transpose(&img));
        let ef = ErrStats::of_images(&flipped, &flip_x(&out));
        let et = ErrStats::of_images(&transposed, &transpose(&out));
        eprintln!(
            "{set:<40} flip max {:.3e}, transpose max {:.3e}",
            ef.max, et.max
        );
        if hardware(dev) && ef.max.max(et.max) > NOISY_BOUND {
            ci_note(&format!(
                "hardware flip / transpose, {set}: max {:.3e} / {:.3e}",
                ef.max, et.max
            ));
        }
        assert!(ef.max <= NOISY_BOUND, "{set}: flip {:e}", ef.max);
        assert!(et.max <= NOISY_BOUND, "{set}: transpose {:e}", et.max);
    }
}

// ------------------------------------------------------------------------------------------------
// The large frame through the tiling path
// ------------------------------------------------------------------------------------------------

#[test]
fn a_2048_by_1365_frame_through_the_tiling_path() {
    // The screen preset at full size and at the scale of a 2048 px export from a 6000 px render, and
    // the print preset, on a photographic-like frame, in 512 px tiles with two in flight, against the
    // CPU port. Tiles against the whole frame on the GPU, bit for bit. With the limiter on, a few
    // rank counts flip even on noisy content at this size (see LARGE_FRAME_MAX_BOUND), so the bound
    // is the 99.9th percentile at 1e-5 and a looser maximum; without the limiter the maximum meets
    // 1e-5.
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (2048usize, 1365usize);
    let img = photo_like(w, h, 71);
    let screen_no_halo = without_halo(&OutputSharpen::screen());
    let cases = [
        ("screen, full size", OutputSharpen::screen(), 1.0f32),
        ("screen, full size, no limiter", screen_no_halo, 1.0),
        (
            "screen, 2048 from 6000",
            OutputSharpen::screen(),
            scale_for_edges(2048, 6000),
        ),
        ("print, full size", OutputSharpen::print(), 1.0),
    ];
    for (name, p, scale) in cases {
        let limiter_on = p.halo_pct > 0.0;
        let started = std::time::Instant::now();
        let tiled = gpu_sharpen_tiled(dev, &p, scale, &img, 512, 2);
        let gpu_time = started.elapsed();
        let started = std::time::Instant::now();
        let cpu = p.apply(&img, scale).expect("cpu");
        let cpu_time = started.elapsed();
        let e = ErrStats::of_images(&tiled, &cpu);
        eprintln!(
            "{name:<30} halo {:>2} px: {} (gpu tiled {:.1?}, cpu {:.1?}; {})",
            p.halo_px(scale).expect("halo"),
            e.line(),
            gpu_time,
            cpu_time,
            dev.gpu.info().name
        );
        assert!(all_finite(&tiled));
        if hardware(dev) {
            ci_note(&format!("hardware {name}: {}", e.line()));
        }
        if limiter_on {
            assert!(e.p999 <= CLEAN_P999_BOUND, "{name}: p99.9 {:e}", e.p999);
            assert!(
                e.max <= LARGE_FRAME_MAX_BOUND,
                "{name}: max error {:e}",
                e.max
            );
        } else {
            assert!(e.max <= NOISY_BOUND, "{name}: max error {:e}", e.max);
        }
        let whole = gpu_sharpen(dev, &p, scale, &img);
        assert!(
            bit_equal(&whole, &tiled),
            "{name}: tiles differ from the whole frame"
        );
        if name == "screen, full size" {
            // A second run with another number of tiles in flight: same bits.
            let again = gpu_sharpen_tiled(dev, &p, scale, &img, 512, 3);
            assert!(
                bit_equal(&again, &tiled),
                "{name}: a second tiled run differs"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Edge cases
// ------------------------------------------------------------------------------------------------

#[test]
fn tiny_and_odd_sizes_follow_the_mirror_semantics() {
    // 1 x 1 up to sizes just past the kernel lengths, in every mode and method: kernels longer than
    // the frame wrap around through the mirror period (spec 4.1), on the GPU as on the CPU.
    let Some(dev) = gpus().first() else { return };
    let mut worst = 0.0f64;
    for (w, h) in [
        (1, 1),
        (2, 1),
        (1, 5),
        (3, 3),
        (5, 3),
        (7, 5),
        (13, 2),
        (14, 16),
        (17, 31),
    ] {
        let img = random_image(w, h, 0.0, 1.0, (w * 100 + h) as u64);
        for (set, p, scale) in param_sets() {
            let cpu = p.apply(&img, scale).expect("cpu");
            let gpu = gpu_sharpen(dev, &p, scale, &img);
            assert!(all_finite(&gpu), "{w}x{h} / {set}: non-finite");
            let e = ErrStats::of_images(&gpu, &cpu);
            assert!(e.max <= NOISY_BOUND, "{w}x{h} / {set}: {}", e.line());
            worst = worst.max(e.max);
        }
        // A 1 x 1 image and a flat image come back unchanged.
        let flat_img = flat(w, h, [0.3, 0.2, 0.1]);
        let out = gpu_sharpen(dev, &OutputSharpen::screen(), 1.0, &flat_img);
        assert!(bit_equal(&out, &flat_img), "{w}x{h}: a flat image changed");
    }
    eprintln!("tiny sizes: worst max error {worst:.3e}");
}

#[test]
fn flat_fields_come_back_bit_exact_for_every_parameter_set() {
    let Some(dev) = gpus().first() else { return };
    let colours = [
        [0.2f32, 0.18, 0.12],
        [1.7, 1.25, -0.02],
        [0.0, 0.0, 0.0],
        [3e-5, 2e-5, 1e-5],
    ];
    for (set, p, scale) in param_sets() {
        for c in colours {
            let img = flat(23, 17, c);
            let out = gpu_sharpen(dev, &p, scale, &img);
            assert!(bit_equal(&out, &img), "{set}: flat {c:?} changed");
        }
    }
}

#[test]
fn no_stage_is_a_copy_and_the_sink_gets_the_input_cores() {
    let Some(dev) = gpus().first() else { return };
    let img = random_image(40, 30, -0.05, 2.0, 81);
    let zero = OutputSharpen::screen().with_amount_pct(0.0);
    assert!(bit_equal(&gpu_sharpen(dev, &zero, 1.0, &img), &img));
    assert!(bit_equal(
        &gpu_sharpen_tiled(dev, &zero, 1.0, &img, 16, 2),
        &img
    ));
    let mut small = OutputSharpen::screen();
    small.radius_px = 0.09;
    assert!(bit_equal(&gpu_sharpen(dev, &small, 1.0, &img), &img));
    // The traced run of a no-op setup returns zero deltas and the luma.
    let (out, trace) = gpu_traced(dev, &zero, 1.0, &img);
    assert!(bit_equal(&out, &img));
    assert!(trace.free_delta.iter().all(|&v| v == 0.0));
    assert!(trace.halo_delta.iter().all(|&v| v == 0.0));
}

#[test]
fn halo_suppression_cuts_the_overshoot_of_a_hard_step() {
    // Spec 6.2 on the GPU: a hard 0.1 to 0.5 step with amount 300 and R 1.5. Halo 100 cuts the
    // overshoot from about 0.43 to about 0.007 and the undershoot from 0.10 (clipped at black) to
    // about 0.007.
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (48usize, 16usize);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 0.1f32 } else { 0.5 };
            img.set_pixel(x, y, [v, v, v]);
        }
    }
    let mut p = OutputSharpen::screen();
    p.amount_pct = 300.0;
    p.radius_px = 1.5;
    p.threshold_levels = 0.0;
    let measure = |halo: f32| {
        let mut q = p.clone();
        q.halo_pct = halo;
        let out = gpu_sharpen(dev, &q, 1.0, &img);
        let row: Vec<f32> = (0..w).map(|x| out.pixel(x, h / 2)[1]).collect();
        let over = row.iter().copied().fold(f32::MIN, f32::max) - 0.5;
        let under = 0.1 - row.iter().copied().fold(f32::MAX, f32::min);
        (over, under)
    };
    let (over0, under0) = measure(0.0);
    let (over1, under1) = measure(100.0);
    eprintln!(
        "halo 0: overshoot {over0:.4}, undershoot {under0:.4}; halo 100: {over1:.4}, {under1:.4}"
    );
    assert!(over0 > 0.3, "overshoot without the limiter: {over0}");
    assert!(over1 < 0.02 && under1 < 0.02, "limited: {over1}, {under1}");
}

#[test]
fn bad_inputs_are_refused_and_buffers_return_to_the_pool() {
    let Some(dev) = gpus().first() else { return };
    let img = busy(40, 30, 91);
    // Invalid parameters and scales.
    let nan = OutputSharpen::screen().with_amount_pct(f32::NAN);
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            PreparedSharpen::new(ctx, &nan, 1.0).map(|_| ())
        });
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    for scale in [0.0f32, 1.5, f32::NAN] {
        let r = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                PreparedSharpen::new(ctx, &OutputSharpen::screen(), scale).map(|_| ())
            });
        assert!(
            matches!(r, Err(GpuError::InvalidInput(_))),
            "scale {scale}: {r:?}"
        );
    }
    // A plan for another frame size, and a plan with too little halo.
    let wrong = img.clone();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedSharpen::new(ctx, &OutputSharpen::screen(), 1.0)?;
            let plan = p.tile_plan(wrong.width + 1, wrong.height, 16);
            p.run_tiled(ctx, &wrong, &plan, 1, &CancelToken::new(), &mut |_, _| {
                Ok(())
            })
        });
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    let short = img.clone();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedSharpen::new(ctx, &OutputSharpen::screen(), 1.0)?;
            let plan =
                ym_gpu::tiles::TilePlan::with_halo(short.width, short.height, 16, p.halo_px() - 1);
            p.run_tiled(ctx, &short, &plan, 1, &CancelToken::new(), &mut |_, _| {
                Ok(())
            })
        });
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    // An image whose buffer does not match its size.
    let mut broken = img.clone();
    broken.data.pop();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            PreparedSharpen::new(ctx, &OutputSharpen::screen(), 1.0)?
                .run(ctx, &broken)
                .map(|_| ())
        });
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    // Cancelling stops the tile loop.
    let cancelled = img.clone();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedSharpen::new(ctx, &OutputSharpen::screen(), 1.0)?;
            let plan = p.tile_plan(cancelled.width, cancelled.height, 16);
            let token = CancelToken::new();
            let mut seen = 0usize;
            let r = p.run_tiled(ctx, &cancelled, &plan, 1, &token, &mut |_, _| {
                seen += 1;
                token.cancel();
                Ok(())
            });
            assert_eq!(seen, 1, "the loop must stop after the cancelled tile");
            r
        });
    assert!(matches!(r, Err(GpuError::Cancelled)), "{r:?}");
    // After all that, every buffer a run took is back in the pool. (The device keeps its persistent
    // tables, here the dither thresholds, for as long as it lives: they are released before the
    // baseline is taken and again before the comparison, so the ledger shows the runs alone.)
    let (base, in_use, live) = run_gpu(dev, {
        let img = img.clone();
        move |ctx| {
            release_tables(ctx);
            let base = ctx.pool().stats().in_use_bytes;
            {
                let p = PreparedSharpen::new(ctx, &OutputSharpen::print(), 1.0)?;
                p.run(ctx, &img)?;
                let plan = p.tile_plan(img.width, img.height, 16);
                p.run_tiled(ctx, &img, &plan, 3, &CancelToken::new(), &mut |_, _| Ok(()))?;
            }
            release_tables(ctx);
            let s = ctx.pool().stats();
            Ok((base, s.in_use_bytes, s.live_buffers))
        }
    });
    assert_eq!(
        in_use,
        base,
        "{live} live buffers, {} bytes still in use after the runs",
        in_use - base
    );
}

#[test]
fn the_traced_run_reports_the_stage_planes() {
    // `run_traced` returns the same pixels as `run`, the luma of the input and the accumulators
    // before the limiter; a setup without a limited share has zeros in Hacc, one with it does not.
    let Some(dev) = gpus().first() else { return };
    let img = busy(40, 30, 101);
    let screen = OutputSharpen::screen();
    let plain = gpu_sharpen(dev, &screen, 1.0, &img);
    let (traced, trace) = gpu_traced(dev, &screen, 1.0, &img);
    assert!(bit_equal(&plain, &traced));
    assert_eq!((trace.width, trace.height), (40, 30));
    assert!(
        trace.halo_delta.iter().any(|&v| v != 0.0),
        "halo 60 must fill Hacc"
    );
    assert!(trace.free_delta.iter().any(|&v| v != 0.0));
    let (_o, none) = gpu_traced(dev, &without_halo(&screen), 1.0, &img);
    assert!(
        none.halo_delta.iter().all(|&v| v == 0.0),
        "no limited share, no Hacc"
    );
    assert_eq!(
        trace.luma, none.luma,
        "the luma does not depend on the stages"
    );
}

// ------------------------------------------------------------------------------------------------
// Properties of spec 6.2 on the GPU, and the limits of the host
// ------------------------------------------------------------------------------------------------

#[test]
fn a_collapsed_kernel_is_a_finite_no_op_for_every_method() {
    // Spec 4.6 and 6.2: for a radius just above 0.09 the outer taps are 0 in f64, the high-pass is the
    // identity, `d = 0` exactly, and method 2's `left` is 0 (the ramp is defined as 0). The stage
    // exists but changes nothing, bit for bit, and nothing is NaN.
    let Some(dev) = gpus().first() else { return };
    let img = random_image(40, 30, 0.01, 1.5, 5);
    for method in [
        Method::HardThreshold,
        Method::SmoothGain,
        Method::GradientRamp,
    ] {
        for halo_pct in [0.0f32, 60.0] {
            let mut p = OutputSharpen::screen();
            p.radius_px = 0.0911;
            p.method = method;
            p.halo_pct = halo_pct;
            let plan = p.plan(1.0).expect("plan");
            assert_eq!(plan.stages().len(), 1, "the stage must exist");
            let k = plan.kernel(0);
            assert_eq!(k.len(), 3);
            assert_eq!(k[0], 0.0, "the kernel must have collapsed to one tap");
            let cpu = p.apply(&img, 1.0).expect("cpu");
            let gpu = gpu_sharpen(dev, &p, 1.0, &img);
            assert!(all_finite(&gpu), "{method:?} / halo {halo_pct}: non-finite");
            assert!(bit_equal(&cpu, &img), "{method:?}: the CPU port changed");
            assert!(
                bit_equal(&gpu, &img),
                "{method:?} / halo {halo_pct}: the GPU changed a collapsed-kernel image by {:e}",
                ErrStats::of_images(&gpu, &img).max
            );
        }
    }
}

#[test]
fn power_of_two_scaling_is_exact_without_a_threshold() {
    // Spec 1.3 and 6.2: with `T = 0` the stage is homogeneous of degree 1, so `f(2x) = 2 f(x)` and
    // `f(x / 4) = f(x) / 4` hold bit for bit when no absolute constant takes part (the gain epsilon
    // and the luma floor do not, at these levels). It catches a constant that was typed in the wrong
    // unit, on the GPU as on the CPU.
    let Some(dev) = gpus().first() else { return };
    let base = photo_like(53, 41, 111);
    let scaled = |k: f32| {
        RgbImage::from_planar(
            base.width,
            base.height,
            base.data.iter().map(|v| v * k).collect(),
        )
        .expect("scaled image")
    };
    let (x2, x4) = (scaled(2.0), scaled(0.25));
    for method in [
        Method::HardThreshold,
        Method::SmoothGain,
        Method::GradientRamp,
    ] {
        for (halo_pct, radius_px, scale) in [(0.0f32, 1.0f32, 1.0f32), (60.0, 2.5, 0.37)] {
            let mut p = OutputSharpen::screen();
            p.threshold_levels = 0.0;
            p.method = method;
            p.halo_pct = halo_pct;
            p.radius_px = radius_px;
            let f1 = gpu_sharpen(dev, &p, scale, &base);
            let f2 = gpu_sharpen(dev, &p, scale, &x2);
            let f4 = gpu_sharpen(dev, &p, scale, &x4);
            let mut changed = 0usize;
            for i in 0..f1.data.len() {
                assert_eq!(
                    (f1.data[i] * 2.0).to_bits(),
                    f2.data[i].to_bits(),
                    "{method:?} / halo {halo_pct}: f(2x) differs from 2 f(x) at sample {i}"
                );
                assert_eq!(
                    (f1.data[i] * 0.25).to_bits(),
                    f4.data[i].to_bits(),
                    "{method:?} / halo {halo_pct}: f(x/4) differs from f(x)/4 at sample {i}"
                );
                changed += usize::from(f1.data[i].to_bits() != base.data[i].to_bits());
            }
            assert!(changed > base.data.len() / 4, "the stage must do something");
        }
    }
}

#[test]
fn channel_ratios_are_preserved_per_pixel() {
    // Spec 6.2 "Hue": the transfer multiplies the three channels by one ratio, so the channel ratios
    // of a pixel survive up to the rounding of the product (relative error 1e-6).
    let Some(dev) = gpus().first() else { return };
    let img = photo_like(61, 47, 121);
    let out = gpu_sharpen(dev, &OutputSharpen::screen(), 1.0, &img);
    let mut checked = 0usize;
    let mut changed = 0usize;
    let mut worst = 0.0f64;
    for y in 0..img.height {
        for x in 0..img.width {
            let (a, b) = (img.pixel(x, y), out.pixel(x, y));
            if a.iter().any(|v| v.abs() < 1e-3) {
                continue;
            }
            let ratio = |c: usize| f64::from(b[c]) / f64::from(a[c]);
            let r0 = ratio(0);
            changed += usize::from((r0 - 1.0).abs() > 1e-4);
            if r0 == 0.0 {
                continue;
            }
            for c in 1..3 {
                worst = worst.max((ratio(c) / r0 - 1.0).abs());
            }
            checked += 1;
        }
    }
    eprintln!(
        "hue: {checked} pixels checked, {changed} sharpened, worst relative error {worst:.2e}"
    );
    assert!(
        checked > img.width * img.height / 2,
        "too few pixels checked"
    );
    assert!(changed > 100, "the stage must change some pixels");
    assert!(worst <= 1e-6, "channel ratios moved by {worst:e}");
}

#[test]
fn non_finite_samples_do_not_panic_and_behave_like_the_cpu_port() {
    // Spec 4.7: inputs must be finite and the stage does not sanitise; a NaN spreads through the blurs
    // and only its own pixel's ratio is forced to 0. The GPU must not hang or panic on such input, and
    // on a software adapter (IEEE comparisons, no fast-math) it classifies every output sample as the
    // CPU port does. Hardware drivers may treat NaN comparisons differently, so there it only runs.
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (40usize, 30usize);
    let mut img = random_image(w, h, 0.05, 0.85, 7);
    let n = img.plane_len();
    img.data[12 * w + 20] = f32::NAN;
    img.data[n + 5 * w + 7] = f32::INFINITY;
    img.data[2 * n + 25 * w + 33] = f32::NEG_INFINITY;
    for (name, p) in [
        ("screen", OutputSharpen::screen()),
        ("print", OutputSharpen::print()),
    ] {
        let cpu = p.apply(&img, 1.0).expect("cpu");
        let gpu = gpu_sharpen(dev, &p, 1.0, &img);
        assert_eq!((gpu.width, gpu.height), (w, h));
        let class = |v: f32| (v.is_nan(), v.is_infinite());
        let mismatches = gpu
            .data
            .iter()
            .zip(&cpu.data)
            .filter(|&(&a, &b)| class(a) != class(b))
            .count();
        let non_finite = gpu.data.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "{name}: {non_finite} non-finite samples, {mismatches} classified unlike the CPU"
        );
        if dev.gpu.info().is_software {
            assert_eq!(mismatches, 0, "{name}: NaN and Inf classes differ from CPU");
        }
    }
}

#[test]
fn a_side_beyond_the_dispatch_limit_is_refused_not_panicked() {
    // A 1 x N column with N > 16 x max_compute_workgroups_per_dimension would need more workgroups on
    // y than the device allows, which wgpu reports as a validation panic. The host refuses it with
    // `InvalidInput` before touching the device.
    let Some(dev) = gpus().first() else { return };
    let limit = dev.gpu.info().limits.max_compute_workgroups_per_dimension as usize;
    if limit > 100_000 {
        eprintln!("skipped: the device allows {limit} workgroups per dimension");
        return;
    }
    let h = limit * 16 + 1;
    let img = flat(1, h, [0.2, 0.2, 0.2]);
    let p = OutputSharpen::screen();
    let r = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            PreparedSharpen::new(ctx, &p, 1.0)?
                .run(ctx, &img)
                .map(|_| ())
        });
    assert!(matches!(r, Err(GpuError::InvalidInput(_))), "{r:?}");
    // The device still works afterwards.
    let ok = flat(5, 5, [0.2, 0.2, 0.2]);
    let out = gpu_sharpen(dev, &OutputSharpen::screen(), 1.0, &ok);
    assert!(bit_equal(&out, &ok));
}

// ------------------------------------------------------------------------------------------------
// Device in, device out; the fused C13 tail; identities that are exact by construction
// ------------------------------------------------------------------------------------------------

use ym_color::encode::OutputSpace;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::encode::{C13, Encoding, encode_c13, encode_host};
use ym_gpu::kernels::output_sharpen::FusedC13;
use ym_gpu::stream::PassList;

/// The planar samples of the rectangle `r` of `img`, dense.
fn cut(img: &RgbImage, r: Rect) -> Vec<f32> {
    let n = img.plane_len();
    let mut v = Vec::with_capacity(3 * r.w * r.h);
    for c in 0..3 {
        for y in r.y..r.y1() {
            v.extend_from_slice(&img.data[c * n + y * img.width + r.x..][..r.w]);
        }
    }
    v
}

#[test]
fn the_device_entry_point_sharpens_a_window_and_writes_its_core_at_an_offset() {
    let Some(dev) = gpus().first() else { return };
    let img = busy(90, 70, 201);
    let window = Rect::new(10, 8, 60, 50);
    let core = Rect::new(20, 15, 30, 25);
    for (set, p, scale) in param_sets() {
        // The stage reads the window and mirrors at its edge: the reference is the host wrapper on
        // the window cut out of the image, and its core.
        let part = img.extract(window);
        let host = gpu_sharpen(dev, &p, scale, &part);
        let want = cut(&host, core.relative_to(&window));
        let (p2, img2) = (p.clone(), img.clone());
        let got = run_gpu(dev, move |ctx| {
            let prepared = PreparedSharpen::new(ctx, &p2, scale)?;
            let src = GpuImage::upload_rgb(ctx, "device sharpen input", &img2)?;
            // A destination wider than the core, the core placed at (5, 3); the rest stays poison.
            let dst = GpuImage::alloc(ctx, "device sharpen out", 41, 33, 3, img2.place)?;
            let mut passes = PassList::new(ctx, "device sharpen");
            prepared.encode(&mut passes, &src, window, core, &dst, (5, 3))?;
            let id = dst.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            dst.take(&done, id)
        });
        for c in 0..3 {
            for y in 0..33usize {
                for x in 0..41usize {
                    let g = got[c * 41 * 33 + y * 41 + x];
                    let inside = (5..5 + core.w).contains(&x) && (3..3 + core.h).contains(&y);
                    if inside {
                        let w = want[c * core.w * core.h + (y - 3) * core.w + (x - 5)];
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "{set}: plane {c} at ({x}, {y}) differs from the host wrapper"
                        );
                    } else {
                        assert!(g.is_nan(), "{set}: plane {c} at ({x}, {y}) was written");
                    }
                }
            }
        }
    }
}

#[test]
fn the_fourth_plane_of_the_source_is_not_read_by_the_f32_road() {
    let Some(dev) = gpus().first() else { return };
    let img = busy(48, 36, 202);
    let p = OutputSharpen::screen();
    let plain = gpu_sharpen(dev, &p, 1.0, &img);
    let mut data = img.data.clone();
    data.extend((0..48 * 36).map(|i| if i % 5 == 0 { 1.0f32 } else { 0.0 }));
    let (p2, place) = (p, img.place);
    let got = run_gpu(dev, move |ctx| {
        let prepared = PreparedSharpen::new(ctx, &p2, 1.0)?;
        let src = GpuImage::upload(ctx, "masked", 48, 36, 4, &data, place)?;
        let dst = GpuImage::alloc(ctx, "out", 48, 36, 3, place)?;
        let mut passes = PassList::new(ctx, "masked sharpen");
        let whole = Rect::whole(48, 36);
        prepared.encode(&mut passes, &src, whole, whole, &dst, (0, 0))?;
        let id = dst.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        dst.take(&done, id)
    });
    assert!(
        plain
            .data
            .iter()
            .zip(&got)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the mask plane changed the sharpened pixels"
    );
}

/// Encodes `img` (display-linear, three planes) to RGBA8 with the encode kernel.
fn encode_rgba(
    dev: &TestGpu,
    img: &RgbImage,
    mask: Option<Vec<u8>>,
    c13: C13,
    bg: [u8; 3],
) -> Vec<u8> {
    let img = img.clone();
    run_gpu(dev, move |ctx| {
        let window = Rect::whole(img.width, img.height);
        encode_host(
            ctx,
            &img,
            mask.as_deref(),
            window,
            Encoding::Rgba8 { background: bg },
            c13,
        )
    })
    .bytes
}

/// Panics naming the first RGBA8 pixel (and how many) that differs.
fn assert_bytes_equal(what: &str, a: &[u8], b: &[u8], width: usize) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let diffs: Vec<usize> = (0..a.len() / 4)
        .filter(|&i| a[4 * i..4 * i + 4] != b[4 * i..4 * i + 4])
        .collect();
    if let Some(&i) = diffs.first() {
        panic!(
            "{what}: {} pixels differ; first at ({}, {}): {:?} against {:?}",
            diffs.len(),
            i % width,
            i / width,
            &a[4 * i..4 * i + 4],
            &b[4 * i..4 * i + 4]
        );
    }
}

#[test]
fn the_fused_c13_tail_equals_the_chained_pair_bit_for_bit() {
    let Some(dev) = gpus().first() else { return };
    let img = photo_like(97, 61, 203);
    let p = OutputSharpen::screen();
    for (space, dither, origin) in [
        (OutputSpace::Srgb, true, (37usize, 101usize)),
        (OutputSpace::Rec2020, true, (3, 5)),
        (OutputSpace::DisplayP3, false, (0, 0)),
    ] {
        let sharpened = gpu_sharpen(dev, &p, 1.0, &img);
        let chained = encode_rgba(
            dev,
            &sharpened,
            None,
            C13 {
                space,
                dither,
                origin,
            },
            [9, 9, 9],
        );
        let (p2, img2) = (p.clone(), img.clone());
        let fused = run_gpu(dev, move |ctx| {
            let prepared = PreparedSharpen::new(ctx, &p2, 1.0)?;
            prepared.run_rgba8(
                ctx,
                &img2,
                FusedC13 {
                    space,
                    dither,
                    origin,
                    background: [9, 9, 9],
                },
            )
        });
        let ym_gpu::kernels::PixelData::Rgba8(fused) = fused.data else {
            panic!("not RGBA8");
        };
        assert_bytes_equal(
            &format!("{space:?} dither {dither}: fused against chained"),
            &fused,
            &chained,
            97,
        );
    }
}

#[test]
fn the_fused_tail_paints_the_letterbox_and_leaves_the_rest_of_the_view_alone() {
    // A four-plane source (colour + mask), the stage on its bounding box, the tail fused: against the
    // chained road (copy the frame, sharpen the core over it, encode with the mask). The fused pass
    // writes the core only, so the view is filled with the background first.
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (90usize, 70usize);
    let img = photo_like(w, h, 204);
    let mask: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            f32::from(u8::from(
                !(12..80).contains(&x) || !(9..62).contains(&y) || x + y < 40,
            ))
        })
        .collect();
    // The bounding box of the unmasked pixels.
    let bbox = {
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for (i, &m) in mask.iter().enumerate() {
            if m == 0.0 {
                let (x, y) = (i % w, i / w);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
        Rect::new(x0, y0, x1 - x0, y1 - y0)
    };
    let p = OutputSharpen::screen();
    let bg = [180u8, 40, 220];
    let tail = FusedC13 {
        space: OutputSpace::DisplayP3,
        dither: true,
        origin: (11, 7),
        background: bg,
    };
    let mut data = img.data.clone();
    data.extend_from_slice(&mask);
    let place = img.place;
    let (chained, fused) = run_gpu(dev, move |ctx| {
        let prepared = PreparedSharpen::new(ctx, &p, 1.0)?;
        let src = GpuImage::upload(ctx, "view", w, h, 4, &data, place)?;
        // Chained: a copy of the frame, the core sharpened over it, then C13 with the mask.
        let sharp = GpuImage::alloc(ctx, "sharp", w, h, 4, place)?;
        let bytes = (w * h * 4) as u64;
        let band = ctx.pool().acquire(
            "chained rgba",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let mut passes = PassList::new(ctx, "chained");
        passes
            .encoder()
            .copy_buffer_to_buffer(&src.buf, 0, &sharp.buf, 0, src.byte_len());
        prepared.encode(&mut passes, &src, bbox, bbox, &sharp, (bbox.x, bbox.y))?;
        encode_c13(
            &mut passes,
            &sharp,
            Rect::whole(w, h),
            Encoding::Rgba8 { background: bg },
            C13 {
                space: tail.space,
                dither: tail.dither,
                origin: tail.origin,
            },
            &band,
            0,
            0,
        )?;
        let a = passes.read(&band, 0, bytes)?;
        let done = passes.submit()?.finish()?;
        let chained = done.read(a).to_vec();
        // Fused: the view filled with the background, the core written by the sharpening's last pass.
        let view = ctx.pool().acquire(
            "fused rgba",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let fill: Vec<u8> = (0..w * h)
            .flat_map(|_| [bg[0], bg[1], bg[2], 255])
            .collect();
        ym_gpu::stream::upload_bytes(ctx, &view, 0, &fill)?;
        let mut passes = PassList::new(ctx, "fused");
        // The tail's origin is the picture position of the *core's* (0, 0): the frame's, moved by the
        // core's offset in it, so that every pixel keeps the dither texel the chained road gives it.
        let fused_tail = FusedC13 {
            origin: (tail.origin.0 + bbox.x, tail.origin.1 + bbox.y),
            ..tail
        };
        prepared.encode_rgba8(
            &mut passes,
            &src,
            bbox,
            bbox,
            &view,
            w,
            (bbox.x, bbox.y),
            fused_tail,
        )?;
        let b = passes.read(&view, 0, bytes)?;
        let done = passes.submit()?.finish()?;
        Ok((chained, done.read(b).to_vec()))
    });
    assert_bytes_equal(
        "the fused tail against the chained pair",
        &fused,
        &chained,
        w,
    );
    // Spot checks: the masked corner is the background, an inside pixel is not.
    assert_eq!(&fused[..4], &[bg[0], bg[1], bg[2], 255]);
    let inside = (35 * w + 45) * 4;
    assert_ne!(&fused[inside..inside + 3], &bg);
}

#[test]
fn a_setup_without_a_stage_copies_its_core_on_the_device() {
    let Some(dev) = gpus().first() else { return };
    let img = random_image(40, 30, -0.05, 2.0, 205);
    let zero = OutputSharpen::screen().with_amount_pct(0.0);
    let core = Rect::new(7, 4, 20, 17);
    let img2 = img.clone();
    let got = run_gpu(dev, move |ctx| {
        let prepared = PreparedSharpen::new(ctx, &zero, 1.0)?;
        assert!(prepared.is_noop());
        let src = GpuImage::upload_rgb(ctx, "noop", &img2)?;
        let dst = GpuImage::alloc(ctx, "noop out", 20, 17, 3, img2.place)?;
        let mut passes = PassList::new(ctx, "noop");
        prepared.encode(&mut passes, &src, Rect::whole(40, 30), core, &dst, (0, 0))?;
        let id = dst.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        dst.take(&done, id)
    });
    let want = cut(&img, core);
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a setup without a stage must copy bit for bit"
    );
}

#[test]
fn collapsed_stages_are_dropped_and_the_rest_runs() {
    // A stage whose kernel has collapsed to one tap is the identity (spec 4.6); the setup drops it
    // and a setup that has nothing else is a no-op. An extra stage that does something survives.
    let Some(dev) = gpus().first() else { return };
    let img = busy(60, 44, 206);
    let mut collapsed = OutputSharpen::screen();
    collapsed.radius_px = 0.0911;
    let c2 = collapsed.clone();
    let noop = run_gpu(dev, move |ctx| {
        let p = PreparedSharpen::new(ctx, &c2, 1.0)?;
        assert_eq!(p.plan().stages().len(), 1, "the plan has the stage");
        Ok((p.is_noop(), p.halo_px()))
    });
    assert!(noop.0, "the collapsed stage must be dropped");
    assert!(
        noop.1 > 0,
        "the receptive field is the plan's, so tiles keep their halo"
    );
    assert!(bit_equal(&gpu_sharpen(dev, &collapsed, 1.0, &img), &img));
    // Together with a stage that works, the collapsed one changes nothing.
    let mut both = OutputSharpen::screen();
    both.extra_stages.push(ExtraStage {
        method: Method::HardThreshold,
        auto_method: false,
        amount_pct: 80.0,
        radius_px: 0.0911,
        threshold_levels: 0.0,
        halo_pct: 40.0,
    });
    let cpu = both.apply(&img, 1.0).expect("cpu");
    let gpu = gpu_sharpen(dev, &both, 1.0, &img);
    let alone = gpu_sharpen(dev, &OutputSharpen::screen(), 1.0, &img);
    assert!(
        bit_equal(&gpu, &alone),
        "the collapsed extra stage changed the result"
    );
    assert!(ErrStats::of_images(&gpu, &cpu).max <= NOISY_BOUND);
}

#[test]
fn a_flat_patch_inside_a_busy_frame_comes_back_bit_for_bit() {
    // Where the delta is exactly 0 the ratio is exactly 1 (module note: Metal divides by multiplying
    // with a reciprocal, which is not 1 for many a / a), so a flat patch further from any edge than
    // the receptive field is untouched, on every device.
    let Some(dev) = gpus().first() else { return };
    let (w, h) = (140usize, 120usize);
    let mut img = busy(w, h, 207);
    let patch = Rect::new(40, 30, 60, 60);
    for y in patch.y..patch.y1() {
        for x in patch.x..patch.x1() {
            img.set_pixel(x, y, [0.31, 0.22, 0.17]);
        }
    }
    for (set, p, scale) in param_sets() {
        let halo = p.halo_px(scale).expect("halo");
        let out = gpu_sharpen(dev, &p, scale, &img);
        if patch.w <= 2 * halo + 4 || patch.h <= 2 * halo + 4 {
            continue; // the receptive field leaves no interior
        }
        let inner = Rect::new(
            patch.x + halo,
            patch.y + halo,
            patch.w - 2 * halo,
            patch.h - 2 * halo,
        );
        let (a, b) = (cut(&out, inner), cut(&img, inner));
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{set}: the flat patch changed"
        );
    }
}

#[test]
fn the_binding_stays_inside_the_design_size() {
    common::limits::assert_max_binding_bytes(
        "output sharpening",
        Rect::new(0, 0, 2000, 1250),
        ym_gpu::kernels::output_sharpen::max_binding_bytes,
    );
}

#[test]
fn the_fused_tail_equals_the_chained_pair_on_tiny_frames_and_every_parameter_set() {
    // 1 x 1, 1 x N, N x 1, 2 x 2 and odd frames (the mirror at the edge, the tail on a core of a
    // single pixel), for every parameter set, with and without the dither, at an absolute origin that
    // is not a multiple of the dither period.
    let Some(dev) = gpus().first() else { return };
    let mut cases = 0usize;
    for (w, h) in [(1usize, 1usize), (1, 7), (7, 1), (2, 2), (3, 5), (9, 4)] {
        let img = random_image(w, h, 0.0, 1.3, 300 + (w * 31 + h) as u64);
        for (set, p, scale) in param_sets() {
            for dither in [true, false] {
                let origin = (64 * 5 + 3, 61usize);
                let sharpened = gpu_sharpen(dev, &p, scale, &img);
                let chained = encode_rgba(
                    dev,
                    &sharpened,
                    None,
                    C13 {
                        space: OutputSpace::Srgb,
                        dither,
                        origin,
                    },
                    [1, 2, 3],
                );
                let (p2, img2) = (p.clone(), img.clone());
                let fused = run_gpu(dev, move |ctx| {
                    PreparedSharpen::new(ctx, &p2, scale)?.run_rgba8(
                        ctx,
                        &img2,
                        FusedC13 {
                            space: OutputSpace::Srgb,
                            dither,
                            origin,
                            background: [1, 2, 3],
                        },
                    )
                });
                let ym_gpu::kernels::PixelData::Rgba8(fused) = fused.data else {
                    panic!("not RGBA8");
                };
                assert_bytes_equal(
                    &format!("{w}x{h} {set} dither {dither}: fused against chained"),
                    &fused,
                    &chained,
                    w,
                );
                cases += 1;
            }
        }
    }
    assert!(cases >= 6 * 2, "{cases} cases");
}

#[test]
fn the_luma_pass_hides_every_factor_and_partial_sum_from_the_optimiser() {
    // K8: Metal's fast-math re-associates `(w c) c` and fuses it with the next addition; an `abs` leaves
    // the first (measured with clang -O2 -ffast-math: 16 % of lumas one ulp off). Each factor, each
    // product and the partial sum goes through `opaque`, an OR with a uniform that is 0 at run time.
    // The luma bits are checked on the device by `luma_has_the_cpu_bits`; this pins the shader text,
    // because lavapipe cannot show a regression (it does not re-associate).
    let src = include_str!("../src/shaders/output_sharpen.wgsl");
    let start = src.find("fn sharpen_luma(").expect("the luma pass");
    let body = &src[start
        ..src[start..]
            .find("\n}\n")
            .map(|e| start + e)
            .expect("its end")];
    for needle in [
        "opaque(opaque(sp.luma_r * r) * r)",
        "opaque(opaque(sp.luma_g * g) * g)",
        "opaque(opaque(sp.luma_b * b) * b)",
        "opaque(a0 + a1)",
        "sharpen_sqrt(s01 + a2)",
    ] {
        assert!(
            body.contains(needle),
            "the luma pass lost `{needle}`:\n{body}"
        );
    }
    assert!(
        !body.contains("abs("),
        "`abs` is not a barrier against re-association"
    );
    assert!(
        src.contains("bitcast<f32>(bitcast<u32>(x) | sp.zero)"),
        "opaque is the OR with the run-time zero (an XOR applied twice cancels)"
    );
}

/// The text of the WGSL function `name` of `output_sharpen.wgsl`, up to its closing brace.
fn wgsl_function(src: &str, name: &str) -> String {
    let start = src
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("`fn {name}` in output_sharpen.wgsl"));
    let end = src[start..]
        .find("\n}\n")
        .map(|e| start + e + 3)
        .expect("its end");
    src[start..end].to_string()
}

#[test]
fn the_shader_square_root_and_division_are_correctly_rounded_on_the_device() {
    // K8: `sharpen_sqrt` and `sharpen_div` compute in integers, so they give the IEEE result whatever
    // the device's `sqrt` and `/` do (Metal's fast-math ones are 1 to 2 ulp off). Run on the device on
    // 2^21 square roots (every 16th mantissa over the whole range of two exponents, one odd and one
    // even, which are both branches of the algorithm, and random others) and 2^21 random quotients with
    // normal operands and results,
    // against Rust's `f32::sqrt` and `/`. The C model of the same routines matched IEEE on all 16.8
    // million mantissas of 11 exponents and 259 million quotients.
    let Some(dev) = gpus().first() else { return };
    let src = include_str!("../src/shaders/output_sharpen.wgsl");
    let shader = format!(
        "@group(0) @binding(0) var<storage, read> a_in: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> b_in: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> out_v: array<f32>;\n\
         {}\n{}\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let i = gid.x;\n\
             if (i >= arrayLength(&a_in)) {{ return; }}\n\
             out_v[2u * i] = sharpen_sqrt(a_in[i]);\n\
             out_v[2u * i + 1u] = sharpen_div(a_in[i], b_in[i]);\n\
         }}\n",
        wgsl_function(src, "sharpen_sqrt"),
        wgsl_function(src, "sharpen_div")
    );
    let n = 1usize << 21;
    let mut rng = Rng(0x5157_4454);
    let mut bits = |lo: u32, hi: u32| -> u32 {
        let e = lo + (rng.next_u64() % u64::from(hi - lo + 1)) as u32;
        let sign = (rng.next_u64() & 1) as u32;
        (sign << 31) | (e << 23) | (rng.next_u64() as u32 & 0x7f_ffff)
    };
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let qa = bits(64, 190);
        let qb = bits(64, 190);
        // The square roots: every 16th mantissa (the whole range, 2^19 of them) at each of the
        // exponents 126 and 127 for the first half, random positive normals for the second; the
        // divisions use the same `a`.
        let sa = if i < n / 2 {
            let e = 126 + (i & 1) as u32;
            (e << 23) | (((i >> 1) as u32 * 16) & 0x7f_ffff)
        } else {
            qa & 0x7fff_ffff
        };
        a.push(f32::from_bits(if i % 3 == 0 {
            sa
        } else {
            sa | (qa & 0x8000_0000)
        }));
        b.push(f32::from_bits(qb));
    }
    let (a2, b2) = (a.clone(), b.clone());
    let got: Vec<f32> = run_gpu(dev, move |ctx| {
        use wgpu::util::DeviceExt as _;
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sharpen sqrt and div"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sharpen sqrt and div"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let input = |label: &str, v: &[f32]| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(v),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let a_buf = input("a", &a2);
        let b_buf = input("b", &b2);
        let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sqrt and div"),
            size: (a2.len() * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharpen sqrt and div"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut passes = PassList::new(ctx, "sharpen sqrt and div");
        {
            let mut pass = passes
                .encoder()
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((a2.len() / 256) as u32, 1, 1);
        }
        let id = passes.read(&out, 0, out.size())?;
        let mut done = passes.submit()?.finish()?;
        Ok(bytemuck::pod_collect_to_vec(&done.take(id)))
    });
    let (mut sqrt_bad, mut div_bad, mut div_checked) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let want_sqrt = a[i].sqrt();
        if got[2 * i].to_bits() != want_sqrt.to_bits() && !(a[i] < 0.0 && got[2 * i].is_nan()) {
            sqrt_bad += 1;
        }
        let want_div = a[i] / b[i];
        if want_div.is_normal() {
            div_checked += 1;
            if got[2 * i + 1].to_bits() != want_div.to_bits() {
                div_bad += 1;
            }
        }
    }
    eprintln!(
        "{}: {sqrt_bad} of {n} square roots and {div_bad} of {div_checked} quotients differ from IEEE",
        dev.gpu.info().name
    );
    if hardware(dev) {
        ci_note(&format!(
            "hardware integer sqrt and division: {sqrt_bad} of {n} square roots and {div_bad} of {div_checked} quotients differ from IEEE"
        ));
    }
    assert!(div_checked > n * 9 / 10, "{div_checked} quotients checked");
    assert_eq!(
        sqrt_bad, 0,
        "the shader's square root must be correctly rounded"
    );
    assert_eq!(
        div_bad, 0,
        "the shader's division must be correctly rounded"
    );
}

#[test]
fn the_passes_pin_every_rounding_the_limiter_sees() {
    // K8, the source side of the test above and of `luma_has_the_cpu_bits`: lavapipe neither fuses nor
    // re-associates, so it cannot show that a barrier went missing. Every square root and division of
    // the passes is the shader's integer one, and the sums and products named here go through `opaque`
    // (the clang fast-math model of `docs/17-gpu.md` section 13 needs each of these groups: without
    // the blur's, the 2048 px export from 6000 px moves by 1.96e-3, as it did on Metal).
    let src = include_str!("../src/shaders/output_sharpen.wgsl");
    let passes_start = src.find("fn sharpen_luma(").expect("the passes");
    let passes = &src[passes_start..];
    for needle in [
        "lum[i] = sharpen_sqrt(s01 + a2);",
        "acc = opaque(acc + opaque(taps[sp.tap_off + j] * opaque(lum[row + xx] - c)));",
        "acc = opaque(acc + opaque(taps[sp.tap_off + j] * opaque(opaque(lum[o] - c) + tmp[o])));",
        "let mag = sharpen_sqrt(opaque(gx * gx) + opaque(gy * gy));",
        "work_a[sp.plane + i] = opaque(v * v) + GAIN_EPS;",
        "acc = opaque(acc + opaque(taps[sp.tap_off + j] * work_a[row + u32(mir(x + i32(j) - r, w))]));",
        "acc = opaque(acc + opaque(taps[sp.tap_off + j] * work_b[o]));",
        "let wg = opaque(sharpen_div(work_a[i], work_a[sp.plane + i]));",
        "dacc[i] = dacc[i] + opaque(sp.a_free * s);",
        "hacc[i] = hacc[i] + opaque(sp.a_halo * s);",
        "let s = opaque(c + hacc[i]);",
        "let sl = opaque(l + hacc[o]);",
        "let ds = opaque(sl - s);",
        "e_shi = opaque(e_shi + select(0.0, es, es > 0.0));",
        "n_shi = opaque(n_shi + side_weight(-ds, ts));",
        "let w = sharpen_div(opaque(opaque(2.0 * t) - d), t);",
        "return opaque(d - inside);",
        "let den = opaque(opaque(near + opaque(far * HALO_INV_N)) + eps);",
        "let d_hi = opaque(sharpen_div(m_hi, n_hi));",
        "v_hi = opaque(v_hi) + opaque(max_l * t);",
        "ratio = opaque(sharpen_div(lp, l));",
    ] {
        assert!(passes.contains(needle), "the passes lost `{needle}`");
    }
    // No division or square root of the device outside the two integer routines' fallbacks.
    for (line_no, line) in passes.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        assert!(
            !code.contains(" / ") || code.contains("sp.tap_n / 2u"),
            "line {line_no} of the passes divides with the device's `/`: {line}"
        );
        assert!(
            !code.contains("sqrt(") || code.contains("sharpen_sqrt("),
            "line {line_no} of the passes uses the device's `sqrt`: {line}"
        );
    }
}
