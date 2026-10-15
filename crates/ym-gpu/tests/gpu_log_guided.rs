//! RA-09: the log-guided fast demosaic kernel (`shaders/log_guided.wgsl`,
//! `kernels::log_guided`) against its CPU reference
//! `cpu::demosaic::log_guided::demosaic_float` (spec `docs/algorithms/02-log-guided-demosaic.md`
//! section 3.9, the all-float form).
//!
//! What is compared, with which bound (each asserted, each printed with `--nocapture`) and what
//! was measured on lavapipe (Mesa 25.2, LLVM 20.1) with the shader's own `log2` and `exp2` (the
//! values before them, with the device's: in brackets):
//!
//! | Case | Asserted bound on `max |gpu - cpu|` | Measured |
//! |---|---|---|
//! | the 16 vectors, `f32` storage, against `expected_float` | [`VECTOR_BOUND`] absolute, and the vectors' own `tolerance_float` of 1e-4 | 4.8e-7 (`highlights_*`, values up to 3.4); 2.7e-7 on `zone_plate`; at most 4.2e-7 elsewhere (3.1e-6) |
//! | random frames in `0..=1.3`, all four CFA phases, odd and even sizes, `guide_detail` -50/0/100 | [`FRAME_BOUND`] absolute | 3.6e-6 (4.8e-6) |
//! | frames from -0.05 to 20 with NaN and infinity samples | [`WIDE_BOUND`] relative to the frame's peak | 4.2e-7 (1.3e-6) |
//! | stripes of one direction at scales 1e3 to 1e30 (one structure measure at `p`, the other as large as the signal) | [`WIDE_BOUND`] relative to the peak, every sample finite | 1.25e-7 at every scale (1.1e-6; Metal at b6f2e4b: 1.06e-5 at 1e30) |
//! | sizes 1 x 1 to 7 x 7 (the mirror wraps inside the halo) | [`FRAME_BOUND`] absolute | 2.1e-6 (1.5e-5, 1 x 3, a strip whose horizontal neighbours are the pixel itself) |
//! | tiny and odd shapes (1 x 1, 1 x 9, 3 x 33, ...) with tile sizes 0 to 8 | bit-identical to the whole frame | 0 |
//! | tiles of 32 to 100 px on 203 x 157, tiled against the whole GPU frame | bit-identical | 0 |
//! | the same, against the CPU | [`FRAME_BOUND`] absolute | 9.5e-7 (2.4e-6) |
//! | a large frame, 2048 x 1365 through the tiling path (4 tiles of 1024) | [`FRAME_BOUND`] absolute | 1.6e-6 (3.2e-6) |
//! | crops by one row or column (the Bayer phase flips), interior | [`FRAME_BOUND`] | 0 |
//! | flips of the mosaic | [`FRAME_BOUND`] | 1.8e-7 (1.2e-7) |
//! | `f16` storage (packed and native), against the CPU on the `f16`-rounded input | [`FRAME_BOUND`] plus half an `f16` ulp of the value | 4.9e-4 (the half ulp at 1.0: output rounding) |
//!
//! The receptive field found by perturbing single photosites is exactly 5 px, as the spec says;
//! two runs, and two tiled runs, are bit-identical in every storage mode.
//!
//! Where the kernel is not bit-exact: `log2` and `exp2` are the shader's own `f32` routines (the
//! CPU rounds the `f64` ones; the power takes the guide filter's sum as two parts, see
//! `kernels::log_guided`), the guide filters, the box sums and the reconstruction accumulate in
//! `f32` (the CPU: `f64`), and the green blend is evaluated in the spec's `f32` form
//! `t = (D / E)^3`, `C (HA + t VA) / (HC + t VC)`. The box mean is the exact quotient `sum / 25`
//! in both (the float form has no floor, so the `floor(sum / 25)` question of spec section 8 does
//! not arise). The filters about the centre value and the second differences as differences of
//! differences are exact rewrites that keep the `f32` error relative to the differences (see the
//! module documentation of `kernels::log_guided`).
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); without an adapter these tests skip with a notice, as the other GPU tests do.

mod common;

/// Prints like `println!` and writes the same line to the process's own stderr as well (the harness captures `println!` and
/// `eprintln!`, not this), so that the numbers of a passing test reach the CI log of the Apple runners, from which a
/// hardware column is made (G-09). No annotation: the budget of those is spent by the older notes.
macro_rules! note {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr(),
            format_args!("[G-09 {}] {line}\n", file!().rsplit('/').next().unwrap_or("")),
        );
    }};
}

use std::path::Path;

use common::{Rng, TestGpu, gpus};
use ym_gpu::cpu::demosaic::log_guided::{self, HALO_PX};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Stage;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_gpu::device::{GpuError, StorageMode};
use ym_gpu::kernels::log_guided as gpu_lg;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tiles::TilePlan;
use ym_noise::model::NoiseModel;

/// Vectors, `f32` storage: the GPU against `expected_float`. A fifth of the vectors' own tolerance
/// (1e-4); measured on lavapipe: 4.8e-7 (`highlights_*`, values up to 3.4; 3.1e-6 with the device's
/// `log2` and `exp2`).
const VECTOR_BOUND: f64 = 2e-5;
/// Random frames in `0..=1.3` (and the 1-pixel-wide strips of the tiny-size sweep): absolute.
/// Measured on lavapipe: 3.6e-6 (random frames), 1.6e-6 (2048 x 1365), 2.1e-6 (sizes 1 x 1 to 7 x 7;
/// with the device's `log2` and `exp2`: 4.8e-6, 3.2e-6 and 1.5e-5).
const FRAME_BOUND: f64 = 5e-5;
/// Frames up to 20 with hostile samples, and stripes up to 1e30: relative to the frame's largest
/// finite sample. Measured on lavapipe: 4.2e-7 and 1.25e-7 (1.3e-6 and 1.1e-6 with the device's
/// `log2` and `exp2`; Metal at b6f2e4b, its fast-math `log2`: 1.06e-5 on the stripes at 1e30).
const WIDE_BOUND: f64 = 1e-5;

// ---------------------------------------------------------------------------------------------
// Devices and runners
// ---------------------------------------------------------------------------------------------

/// The device with `f32` storage (always available when there is an adapter at all).
fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

fn run_gpu(dev: &TestGpu, m: &Mosaic, noise: NoiseDn, detail: i32) -> RgbImage {
    let m = m.clone();
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            gpu_lg::demosaic(ctx, &m, noise, detail)
        })
        .expect("GPU log-guided demosaic")
}

fn run_gpu_tiled(
    dev: &TestGpu,
    m: &Mosaic,
    noise: NoiseDn,
    detail: i32,
    tile_px: usize,
) -> RgbImage {
    let m = m.clone();
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            gpu_lg::demosaic_tiled(ctx, &m, noise, detail, tile_px, token)
        })
        .expect("GPU tiled log-guided demosaic")
}

fn cpu(m: &Mosaic, noise: NoiseDn, detail: i32) -> RgbImage {
    log_guided::demosaic_float(m, noise, detail)
}

/// Largest `|a - b|` and where it is (`plane`, `x`, `y`), over all samples.
struct Diff {
    max_abs: f64,
    at: (usize, usize, usize),
    /// Largest finite value of the reference, for the relative bounds.
    peak: f64,
}

fn diff(a: &RgbImage, b: &RgbImage) -> Diff {
    assert_eq!((a.width, a.height), (b.width, b.height), "image sizes");
    let mut d = Diff {
        max_abs: 0.0,
        at: (0, 0, 0),
        peak: 0.0,
    };
    for c in 0..3 {
        for (i, (&x, &y)) in a.plane(c).iter().zip(b.plane(c)).enumerate() {
            assert!(x.is_finite(), "GPU sample ({c}, {i}) is {x}");
            let e = (f64::from(x) - f64::from(y)).abs();
            if e > d.max_abs {
                d.max_abs = e;
                d.at = (c, i % a.width, i / a.width);
            }
            d.peak = d.peak.max(f64::from(y.abs()));
        }
    }
    d
}

fn bits_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.data.len() == b.data.len()
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------------------------

/// A seeded natural-looking mosaic with a trace of noise, values in about `0..=peak`.
fn scene(w: usize, h: usize, cfa: Cfa, seed: u64, peak: f32) -> Mosaic {
    let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
    for v in &mut m.data {
        *v *= peak;
    }
    synth::add_noise(&mut m.data, 0.004 * peak, seed ^ 0x5eed);
    for v in &mut m.data {
        // R1 clamps at -0.05 before the gains.
        *v = v.max(-0.05);
    }
    m
}

/// Frames up to 20 with deep negatives, a few NaN and infinite samples and some values at the
/// ends of the range: the property-test input of the spec (6.2, item 7).
fn hostile(w: usize, h: usize, cfa: Cfa, seed: u64) -> Mosaic {
    let mut rng = Rng(seed);
    let mut m = scene(w, h, cfa, seed, 6.0);
    for v in &mut m.data {
        let r = rng.unit();
        if r < 0.004 {
            *v = f32::NAN;
        } else if r < 0.006 {
            *v = f32::INFINITY;
        } else if r < 0.008 {
            *v = f32::NEG_INFINITY;
        } else if r < 0.03 {
            *v = rng.range(10.0, 20.0) as f32;
        } else if r < 0.06 {
            *v = rng.range(-0.05, 0.0) as f32;
        }
    }
    m
}

fn noise_model(a: f64, b: f64) -> NoiseDn {
    NoiseDn::from_model(&NoiseModel::new(a, b))
}

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

struct Vector {
    name: String,
    guide_detail: i32,
    noise: NoiseDn,
    mosaic: Mosaic,
    expected_float: Vec<f32>,
    tolerance_float: f64,
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn load_vectors() -> Vec<Vector> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/log_guided_demosaic");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).expect("vector file"))
                    .expect("vector json");
            let n = &v["params"]["noise"];
            let mut model = NoiseModel::new(n["a"].as_f64().unwrap(), n["b"].as_f64().unwrap());
            model.sigma_row = n["sigma_row"].as_f64().unwrap();
            model.q = n["q"].as_f64().unwrap();
            let input = &v["input"];
            let (w, h) = (
                input["width"].as_u64().unwrap() as usize,
                input["height"].as_u64().unwrap() as usize,
            );
            let cfa = Cfa::from_name(input["cfa"].as_str().unwrap()).expect("cfa name");
            Vector {
                name: v["name"].as_str().unwrap().to_owned(),
                guide_detail: v["params"]["guide_detail"].as_i64().unwrap() as i32,
                noise: NoiseDn::from_model(&model),
                mosaic: Mosaic::from_data(w, h, cfa, floats(&input["data"])).expect("mosaic"),
                expected_float: floats(&v["expected_float"]["data"]),
                tolerance_float: v["tolerance_float"].as_f64().unwrap(),
            }
        })
        .collect()
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "plane lengths");
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn every_vector_matches_expected_float_on_the_gpu() {
    let Some(dev) = f32_device() else { return };
    let vectors = load_vectors();
    assert_eq!(vectors.len(), 16, "the vector set");
    let mut worst = 0.0f64;
    for v in &vectors {
        let out = run_gpu(dev, &v.mosaic, v.noise, v.guide_detail);
        // Planar output of the right shape, in the layout the vectors use.
        assert_eq!((out.width, out.height), (v.mosaic.width, v.mosaic.height));
        let d = max_abs_diff(&out.data, &v.expected_float);
        // The CPU reference reproduces the vectors to 0.0; the GPU is compared with both.
        let cpu_d = diff(&out, &cpu(&v.mosaic, v.noise, v.guide_detail));
        println!(
            "{:<22} {}: max |gpu - expected_float| {d:.3e} (vector tolerance {:.0e}, asserted {VECTOR_BOUND:.0e}); against the CPU port {:.3e}",
            v.name,
            v.mosaic.cfa.name(),
            v.tolerance_float,
            cpu_d.max_abs
        );
        assert!(
            d <= v.tolerance_float,
            "{}: the GPU differs from expected_float by {d}, the vector's tolerance is {}",
            v.name,
            v.tolerance_float
        );
        assert!(
            d <= VECTOR_BOUND,
            "{}: the GPU differs from expected_float by {d}, asserted bound {VECTOR_BOUND}",
            v.name
        );
        worst = worst.max(d);
    }
    note!("vectors, worst |gpu - expected_float|: {worst:.3e}");
}

#[test]
fn vectors_in_every_storage_mode_stay_within_half_an_f16_ulp() {
    // The input is rounded to f16 on upload, so the reference runs on the rounded input; the output
    // planes are f16 too, which adds at most half an f16 ulp (2^-11 relative) to the f32 bound.
    for dev in gpus() {
        let mut worst = 0.0f64;
        for v in &load_vectors() {
            let mut m = v.mosaic.clone();
            if dev.mode.is_f16() {
                m.data.iter_mut().for_each(|x| *x = common::f16_round(*x));
            }
            let got = run_gpu(dev, &m, v.noise, v.guide_detail);
            let want = cpu(&m, v.noise, v.guide_detail);
            let e = bounded_error(&got, &want, dev.mode, FRAME_BOUND);
            worst = worst.max(e);
        }
        println!(
            "vectors, storage {}: worst |gpu - cpu| {worst:.3e}",
            dev.mode.name()
        );
    }
}

/// Asserts `|got - want| <= abs_bound + (f16 half ulp of want, when the storage is f16)` and returns
/// the largest absolute error.
fn bounded_error(got: &RgbImage, want: &RgbImage, mode: StorageMode, abs_bound: f64) -> f64 {
    let half_ulp = if mode.is_f16() { 2f64.powi(-11) } else { 0.0 };
    let mut worst = 0.0f64;
    for c in 0..3 {
        for (i, (&g, &w)) in got.plane(c).iter().zip(want.plane(c)).enumerate() {
            let e = (f64::from(g) - f64::from(w)).abs();
            let allowed = abs_bound + 1.001 * half_ulp * f64::from(w.abs());
            assert!(
                e <= allowed,
                "storage {}: plane {c} sample ({}, {}): gpu {g} against cpu {w}, error {e} > {allowed}",
                mode.name(),
                i % got.width,
                i / got.width
            );
            worst = worst.max(e);
        }
    }
    worst
}

// ---------------------------------------------------------------------------------------------
// Random frames, all four phases
// ---------------------------------------------------------------------------------------------

#[test]
fn random_frames_match_in_all_four_phases_and_guide_details() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 100;
    for cfa in Cfa::ALL {
        // Odd and even sizes, narrower and wider than a workgroup (16) and than the halo (5).
        for (w, h) in [(67usize, 41usize), (40, 40), (31, 18), (9, 7)] {
            for (detail, noise) in [
                (0, NoiseDn::ZERO),
                (100, noise_model(2e-4, 4e-8)),
                (-50, noise_model(1e-3, 1e-6)),
            ] {
                seed += 1;
                let m = scene(w, h, cfa, seed, 1.3);
                let got = run_gpu(dev, &m, noise, detail);
                let d = diff(&got, &cpu(&m, noise, detail));
                println!(
                    "{} {w}x{h} detail {detail:>4} p-noise r={:<8.2}: max |gpu - cpu| {:.3e} at plane {} ({}, {})",
                    cfa.name(),
                    noise.r,
                    d.max_abs,
                    d.at.0,
                    d.at.1,
                    d.at.2
                );
                assert!(
                    d.max_abs <= FRAME_BOUND,
                    "{} {w}x{h} detail {detail}: {} > {FRAME_BOUND}",
                    cfa.name(),
                    d.max_abs
                );
                worst = worst.max(d.max_abs);
            }
        }
    }
    note!("random frames, worst |gpu - cpu|: {worst:.3e} (bound {FRAME_BOUND:.0e})");
}

#[test]
fn hostile_frames_with_negatives_nan_infinity_and_large_values_match() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f64;
    for (i, cfa) in Cfa::ALL.into_iter().enumerate() {
        let m = hostile(83, 59, cfa, 900 + i as u64);
        for noise in [NoiseDn::ZERO, noise_model(5e-4, 1e-7)] {
            let got = run_gpu(dev, &m, noise, 0);
            let want = cpu(&m, noise, 0);
            let d = diff(&got, &want);
            let rel = d.max_abs / d.peak;
            println!(
                "{} hostile 83x59 r={:<8.2}: max |gpu - cpu| {:.3e} = {rel:.2e} of the peak {:.2}",
                cfa.name(),
                noise.r,
                d.max_abs,
                d.peak
            );
            assert!(rel <= WIDE_BOUND, "{}: {rel} > {WIDE_BOUND}", cfa.name());
            // Native samples pass through unchanged (a non-finite one is written as 0), the
            // interpolated ones are never negative.
            for y in 0..m.height {
                for x in 0..m.width {
                    let c = m.cfa.color_at(x, y).plane();
                    let v = m.data[y * m.width + x];
                    let native = if v.is_finite() { v } else { 0.0 };
                    for plane in 0..3 {
                        let g = got.pixel(x, y)[plane];
                        if plane == c {
                            assert_eq!(g.to_bits(), native.to_bits(), "native ({x}, {y})");
                        } else {
                            assert!(g >= 0.0, "interpolated ({x}, {y}) plane {plane} is {g}");
                        }
                    }
                }
            }
            worst = worst.max(rel);
        }
    }
    note!("hostile frames, worst relative error: {worst:.3e} (bound {WIDE_BOUND:.0e})");
}

#[test]
fn huge_values_in_strongly_directional_frames_stay_finite_and_match() {
    // Stripes of one direction make one structure measure close to `p` and the other as large as
    // the signal, so the blend weights differ by (signal / p)^3. Divided through by the weight of
    // the horizontal estimate (the spec's `t` form), `t * vc` overflowed `f32` for values above
    // about 1e5 and gave NaN on the GPU where the CPU's `f64` is finite. The kernel divides through
    // by the larger weight instead; it stays finite until the 25-sample box sum of the structure
    // measures overflows, near 1e33 (the CPU's `f32` planes: near 3e33). The accuracy at large
    // scales is that of L = log2(p + A / 2) near 116, where an ulp is a relative 5e-6 of the guide:
    // Metal's fast-math `log2` gave 1.06e-5 of the peak at 1e30 (b6f2e4b); the shader's own `log2`
    // and two-part `exp2` give 1.25e-7 on lavapipe and, in the clang fast-math model, 1.8e-7.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (40usize, 30usize);
    let mut worst = 0.0f64;
    for scale in [1e3f32, 1e6, 1e10, 1e20, 1e30] {
        for transpose in [false, true] {
            let mut m = Mosaic::new(w, h, Cfa::RGGB);
            for y in 0..h {
                for x in 0..w {
                    let stripe = if transpose { y } else { x };
                    let level = [0.1f32, 3.0, 0.5, 2.0, 0.05][stripe % 5];
                    m.data[y * w + x] = level * scale * (1.0 + 0.1 * ((x + y) % 2) as f32);
                }
            }
            let got = run_gpu(dev, &m, NoiseDn::ZERO, 0);
            let want = cpu(&m, NoiseDn::ZERO, 0);
            let d = diff(&got, &want); // also asserts every GPU sample is finite
            let rel = d.max_abs / d.peak;
            println!(
                "stripes (transposed: {transpose}) at scale {scale:e}: {rel:.2e} of the peak {:.2e}",
                d.peak
            );
            assert!(rel <= WIDE_BOUND, "scale {scale:e} {transpose}: {rel}");
            worst = worst.max(rel);
        }
    }
    note!("huge values, worst relative error: {worst:.3e} (bound {WIDE_BOUND:.0e})");
}

#[test]
fn tiny_buffers_wrap_the_mirror_like_the_cpu() {
    // Narrower than the halo: the mirror wraps several times (vector `tiny_6x5`). Sizes down to 1.
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 5000;
    for cfa in [Cfa::RGGB, Cfa::BGGR] {
        for h in 1..=7usize {
            for w in 1..=7usize {
                seed += 1;
                let m = scene(w, h, cfa, seed, 1.0);
                let got = run_gpu(dev, &m, NoiseDn::ZERO, 40);
                let d = diff(&got, &cpu(&m, NoiseDn::ZERO, 40));
                assert!(d.max_abs <= FRAME_BOUND, "{w}x{h}: {}", d.max_abs);
                worst = worst.max(d.max_abs);
            }
        }
    }
    note!("sizes 1x1 to 7x7, worst |gpu - cpu|: {worst:.3e}");
}

#[test]
fn cropping_by_one_row_or_column_changes_the_phase_and_keeps_the_interior() {
    // Spec 6.2 item 3, on the GPU: the interior (5 px from the cut) is unchanged by a one-pixel
    // crop, which flips the Bayer phase of the buffer. The comparison is to the uncropped GPU run,
    // so f32 rounding order is the same on both sides: allowed up to the f32 rounding of the planes.
    let Some(dev) = f32_device() else { return };
    let m = scene(70, 54, Cfa::RGGB, 77, 1.3);
    let noise = noise_model(2e-4, 4e-8);
    let full = run_gpu(dev, &m, noise, 0);
    let mut worst = 0.0f64;
    for (dx, dy) in [(1usize, 0usize), (0, 1), (1, 1)] {
        let rect = ym_gpu::cpu::Rect::new(dx, dy, m.width - dx, m.height - dy);
        let cropped = m.extract(rect);
        assert_ne!(cropped.cfa, m.cfa);
        let got = run_gpu(dev, &cropped, noise, 0);
        for c in 0..3 {
            for y in HALO_PX..cropped.height - HALO_PX {
                for x in HALO_PX..cropped.width - HALO_PX {
                    let a = got.plane(c)[y * got.width + x];
                    let b = full.plane(c)[(y + dy) * full.width + x + dx];
                    worst = worst.max((f64::from(a) - f64::from(b)).abs());
                }
            }
        }
    }
    note!("crops by (1,0), (0,1), (1,1): interior differs by {worst:.3e}");
    assert!(worst <= FRAME_BOUND, "{worst}");
}

// ---------------------------------------------------------------------------------------------
// Tiling and the large frame
// ---------------------------------------------------------------------------------------------

#[test]
fn tiles_with_halo_equal_the_whole_frame_at_tile_and_halo_boundaries() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    for (cfa, seed) in [(Cfa::RGGB, 31u64), (Cfa::GBRG, 32)] {
        let m = scene(w, h, cfa, seed, 1.3);
        let noise = noise_model(2e-4, 4e-8);
        let whole = run_gpu(dev, &m, noise, 0);
        let reference = cpu(&m, noise, 0);
        let d_whole = diff(&whole, &reference);
        assert!(d_whole.max_abs <= FRAME_BOUND);
        for tile_px in [32usize, 50, 64, 100] {
            let tiled = run_gpu_tiled(dev, &m, noise, 0, tile_px);
            let to_whole = diff(&tiled, &whole);
            let to_cpu = diff(&tiled, &reference);
            println!(
                "{} {w}x{h} tiles of {tile_px}: tiled against whole GPU {:.3e}, against CPU {:.3e}",
                cfa.name(),
                to_whole.max_abs,
                to_cpu.max_abs
            );
            // Same operations in the same order on the same inputs: expect identical bits.
            assert!(
                bits_equal(&tiled, &whole),
                "tile {tile_px}: not bit-identical to the whole frame"
            );
            assert!(
                to_cpu.max_abs <= FRAME_BOUND,
                "tile {tile_px}: {}",
                to_cpu.max_abs
            );
        }
    }
}

#[test]
fn tiling_is_bit_identical_on_tiny_and_odd_shapes_and_degenerate_tile_sizes() {
    // Strips one pixel wide or high, frames smaller than the halo and tile sizes of 0 and 1 (which
    // the plan rounds up to 2): no panic, no seam, the same bits as the whole frame.
    let Some(dev) = f32_device() else { return };
    let mut seed = 8000;
    for (w, h) in [(1usize, 1usize), (1, 9), (2, 2), (7, 3), (3, 33), (17, 29)] {
        for cfa in [Cfa::RGGB, Cfa::GBRG] {
            seed += 1;
            let m = scene(w, h, cfa, seed, 1.3);
            let whole = run_gpu(dev, &m, NoiseDn::ZERO, 0);
            for tile_px in [0usize, 1, 3, 8] {
                let tiled = run_gpu_tiled(dev, &m, NoiseDn::ZERO, 0, tile_px);
                assert!(
                    bits_equal(&tiled, &whole),
                    "{} {w}x{h}, tile_px {tile_px}: not bit-identical to the whole frame",
                    cfa.name()
                );
            }
        }
    }
}

#[test]
fn a_plan_without_context_or_for_another_size_is_refused() {
    let Some(dev) = f32_device() else { return };
    let m = scene(64, 48, Cfa::RGGB, 3, 1.0);
    // A plan without any halo: tile rectangles equal their cores.
    let no_context = TilePlan::new(64, 48, 32, &[]);
    let wrong_size = TilePlan::new(
        65,
        48,
        32,
        &[Stage::Demosaic(ym_gpu::cpu::demosaic::Method::LogGuided)],
    );
    let empty = Mosaic {
        width: 0,
        height: 0,
        cfa: Cfa::RGGB,
        place: m.place,
        data: Vec::new(),
    };
    let err = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let a = gpu_lg::demosaic_planned(ctx, &m, NoiseDn::ZERO, 0, &no_context, token);
            let b = gpu_lg::demosaic_planned(ctx, &m, NoiseDn::ZERO, 0, &wrong_size, token);
            let c = gpu_lg::demosaic(ctx, &empty, NoiseDn::ZERO, 0);
            Ok((a.err(), b.err(), c.err()))
        })
        .expect("job");
    for (what, e) in [
        ("no context", err.0),
        ("wrong size", err.1),
        ("empty mosaic", err.2),
    ] {
        assert!(
            matches!(e, Some(GpuError::InvalidInput(_))),
            "{what}: {e:?}"
        );
    }
}

#[test]
fn a_cancelled_token_stops_the_tiled_run() {
    let Some(dev) = f32_device() else { return };
    let m = scene(96, 64, Cfa::RGGB, 4, 1.0);
    let e = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| {
            let token = CancelToken::new();
            token.cancel();
            Ok(gpu_lg::demosaic_tiled(ctx, &m, NoiseDn::ZERO, 0, 32, &token).err())
        })
        .expect("job");
    assert!(matches!(e, Some(GpuError::Cancelled)), "{e:?}");
}

#[test]
fn a_large_frame_through_the_tiling_path_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let m = scene(w, h, Cfa::RGGB, 2048, 1.3);
    let noise = noise_model(2e-4, 4e-8);
    let started = std::time::Instant::now();
    let tiled = run_gpu_tiled(dev, &m, noise, 0, 1024);
    let gpu_time = started.elapsed();
    let reference = cpu(&m, noise, 0);
    let d = diff(&tiled, &reference);
    println!(
        "{w}x{h} in tiles of 1024 ({} tiles): max |gpu - cpu| {:.3e} at plane {} ({}, {}); GPU {gpu_time:.2?} on {}",
        TilePlan::new(
            w,
            h,
            1024,
            &[Stage::Demosaic(ym_gpu::cpu::demosaic::Method::LogGuided)]
        )
        .tiles()
        .len(),
        d.max_abs,
        d.at.0,
        d.at.1,
        d.at.2,
        dev.gpu.info().name
    );
    assert!(d.max_abs <= FRAME_BOUND, "{}", d.max_abs);
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn two_gpu_runs_are_bit_identical_in_every_storage_mode() {
    for dev in gpus() {
        let m = hostile(97, 61, Cfa::GRBG, 1234);
        let noise = noise_model(3e-4, 1e-7);
        let a = run_gpu(dev, &m, noise, 25);
        let b = run_gpu(dev, &m, noise, 25);
        assert!(bits_equal(&a, &b), "storage {}", dev.mode.name());
        let ta = run_gpu_tiled(dev, &m, noise, 25, 40);
        let tb = run_gpu_tiled(dev, &m, noise, 25, 40);
        assert!(bits_equal(&ta, &tb), "tiled, storage {}", dev.mode.name());
    }
}

#[test]
fn f16_storage_frames_stay_within_half_an_f16_ulp_of_the_f32_bound() {
    for dev in gpus() {
        if !dev.mode.is_f16() {
            continue;
        }
        let mut worst = 0.0f64;
        for (i, cfa) in Cfa::ALL.into_iter().enumerate() {
            let mut m = scene(75, 50, cfa, 700 + i as u64, 1.3);
            m.data.iter_mut().for_each(|x| *x = common::f16_round(*x));
            let noise = noise_model(2e-4, 4e-8);
            let got = run_gpu(dev, &m, noise, 0);
            let want = cpu(&m, noise, 0);
            worst = worst.max(bounded_error(&got, &want, dev.mode, FRAME_BOUND));
        }
        note!("storage {}: worst |gpu - cpu| {worst:.3e}", dev.mode.name());
    }
}

// ---------------------------------------------------------------------------------------------
// Properties of spec section 6.2, on the GPU
// ---------------------------------------------------------------------------------------------

#[test]
fn flat_fields_come_back_in_all_four_phases() {
    // Spec 6.2 item 1: the float form returns a flat field within 1e-6 relative.
    let Some(dev) = f32_device() else { return };
    for cfa in Cfa::ALL {
        for (rgb, detail, noise) in [
            ([0.25f32; 3], 0, NoiseDn::ZERO),
            ([0.42, 0.30, 0.18], -50, noise_model(2e-4, 4e-8)),
            ([0.42, 0.30, 0.18], 100, NoiseDn::ZERO),
            ([2.5, 1.5, 3.0], 0, noise_model(2e-4, 4e-8)),
        ] {
            let (w, h) = (23usize, 18usize);
            let mut m = Mosaic::new(w, h, cfa);
            for y in 0..h {
                for x in 0..w {
                    m.data[y * w + x] = rgb[cfa.color_at(x, y).plane()];
                }
            }
            let out = run_gpu(dev, &m, noise, detail);
            for c in 0..3 {
                for &v in out.plane(c) {
                    assert!(
                        (f64::from(v) - f64::from(rgb[c])).abs() <= 1e-6 * f64::from(rgb[c]),
                        "{} flat {rgb:?} detail {detail}: plane {c} has {v}",
                        cfa.name()
                    );
                }
            }
        }
    }
}

#[test]
fn zero_and_all_negative_fields_give_zero_interpolated_samples() {
    // Spec section 4: a field of zeros has a constant guide and output 0; negatives are seen as 0
    // by the interpolation, while native samples keep their value.
    let Some(dev) = f32_device() else { return };
    for (value, noise) in [(0.0f32, NoiseDn::ZERO), (-0.03, noise_model(1e-3, 1e-6))] {
        let mut m = Mosaic::new(19, 14, Cfa::GBRG);
        m.data.fill(value);
        let out = run_gpu(dev, &m, noise, 0);
        for y in 0..m.height {
            for x in 0..m.width {
                let native = m.cfa.color_at(x, y).plane();
                for c in 0..3 {
                    let v = out.pixel(x, y)[c];
                    let want = if c == native { value } else { 0.0 };
                    assert_eq!(
                        v.to_bits(),
                        want.to_bits(),
                        "value {value} ({x}, {y}) plane {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn flipping_the_mosaic_flips_the_output() {
    // Spec 6.2 item 4. The GPU sums in f32, so the flipped run differs from the flipped output by
    // rounding, not by the bit.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (51usize, 38usize);
    let m = scene(w, h, Cfa::RGGB, 41, 1.3);
    let noise = noise_model(2e-4, 4e-8);
    let base = run_gpu(dev, &m, noise, 0);
    let mut worst = 0.0f64;
    for (flip_x, flip_y) in [(true, false), (false, true), (true, true)] {
        let mut f = m.clone();
        f.cfa = m.cfa.shifted(
            if flip_x { w - 1 } else { 0 },
            if flip_y { h - 1 } else { 0 },
        );
        for y in 0..h {
            for x in 0..w {
                let sx = if flip_x { w - 1 - x } else { x };
                let sy = if flip_y { h - 1 - y } else { y };
                f.data[y * w + x] = m.data[sy * w + sx];
            }
        }
        let got = run_gpu(dev, &f, noise, 0);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let sx = if flip_x { w - 1 - x } else { x };
                    let sy = if flip_y { h - 1 - y } else { y };
                    let a = got.plane(c)[y * w + x];
                    let b = base.plane(c)[sy * w + sx];
                    worst = worst.max((f64::from(a) - f64::from(b)).abs());
                }
            }
        }
    }
    note!("flips: worst |flipped run - flipped output| {worst:.3e}");
    assert!(worst <= FRAME_BOUND, "{worst}");
}

#[test]
fn one_photosite_reaches_exactly_the_halo_and_no_further() {
    // Spec 1.4 and 6.2 item 5: the receptive field is 5 px (Chebyshev), reached with the sharpest
    // guide (`guide_detail` 100: at 0 the outer taps are zero and the reach is shorter). The four
    // passes read the same inputs as the CPU reference, so a perturbation must leave every pixel
    // farther than 5 px bit-identical, and some pixel at exactly 5 px must change.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (41usize, 37usize);
    let noise = NoiseDn::ZERO;
    let mut reach = 0usize;
    for cfa in [Cfa::RGGB, Cfa::GBRG] {
        let m = scene(w, h, cfa, 55, 1.0);
        let base = run_gpu(dev, &m, noise, 100);
        for (px, py) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let (cx, cy) = (20 + px, 18 + py);
            let mut bumped = m.clone();
            let i = cy * w + cx;
            bumped.data[i] = bumped.data[i] * 4.0 + 0.3;
            let out = run_gpu(dev, &bumped, noise, 100);
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        let j = y * w + x;
                        if out.plane(c)[j].to_bits() != base.plane(c)[j].to_bits() {
                            let r = x.abs_diff(cx).max(y.abs_diff(cy));
                            assert!(
                                r <= HALO_PX,
                                "{} ({cx}, {cy}) changed plane {c} at ({x}, {y}), {r} px away",
                                cfa.name()
                            );
                            reach = reach.max(r);
                        }
                    }
                }
            }
        }
    }
    println!("receptive-field radius on the GPU: {reach} px");
    assert_eq!(
        reach, HALO_PX,
        "the reach found by perturbing single photosites"
    );
}

#[test]
fn out_of_range_guide_detail_is_clamped_like_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let m = scene(45, 33, Cfa::BGGR, 66, 1.3);
    let noise = noise_model(2e-4, 4e-8);
    for detail in [1000, -1000, 101, -51] {
        let got = run_gpu(dev, &m, noise, detail);
        let d = diff(&got, &cpu(&m, noise, detail));
        assert!(d.max_abs <= FRAME_BOUND, "detail {detail}: {}", d.max_abs);
    }
}
