//! RA-09: the R8 diffraction deconvolution kernel (`shaders/diffraction.wgsl`,
//! `kernels::diffraction`) against its CPU reference `cpu::sharpen::diffraction` (spec
//! `docs/algorithms/03-diffraction-deconvolution.md`, sections 3.4 to 3.12 and the GPU notes of
//! section 5).
//!
//! What is compared, with which bound (each asserted, each printed with `--nocapture`):
//!
//! | Case | Asserted bound on `max |gpu - cpu|` (normalised units) |
//! |---|---|
//! | the 13 image vectors against their `expected` data (vector tolerance 2e-5, exact vectors bit-equal) | [`VECTOR_BOUND`] |
//! | random frames: natural, edge, text, highlight, dark (negative samples) and clipped (plateaus at white) scenes blurred by the plan's own PSF with Poisson-Gaussian noise, R 2 to 6, r_G 1 to 6, pitches 2.4 to 8.4 um, f/1 to f/50, scales 1, 0.6 and 1/4, strengths 0.5 and 3, both gates, other luma weights and gains | [`FRAME_BOUND`] |
//! | sizes 1 x 1 to 7 x 7 and frames narrower than the halo (the mirror wraps several times) | [`FRAME_BOUND`] |
//! | values from -0.05 to 4 (spec 6.2 item 7), zero noise model, strength 3 | [`FRAME_BOUND`] |
//! | the same hostile idea on frames the stage acts on (values -0.05 to 0.7, with and without 1 % spikes up to 4; the CPU port must change them) | [`FRAME_BOUND`] |
//! | a NaN, an infinity or 1e30 in one sample (outside the contract): no panic, nothing changes beyond the halo | bit-identical beyond the halo |
//! | tiles of 32 to 100 px over 203 x 157 with the plan's exact halo, against the whole GPU frame | bit-identical |
//! | the same tiles against the CPU | [`FRAME_BOUND`] |
//! | a large frame, 2048 x 1365 through the tiling path | [`FRAME_BOUND`] |
//!
//! The measured values are in the doc comments of the bounds. The CPU port reproduces the vectors
//! bit for bit (`tests/cpu_diffraction.rs`), so a GPU error against the CPU port and against the
//! vector are the same number up to the vectors' own `f32` storage.
//!
//! The stage works on demosaiced RGB, so there is no CFA phase to vary (spec 4, "CFA phases: not
//! applicable"): what stands in for the four phases is the frame position, covered by the
//! tile-boundary tests (a tile at the frame edge and one in the interior), the sizes below the halo
//! and the flips.
//!
//! Where the kernel is not bit-exact, and why:
//!
//! * every window sum, convolution, product and the 3 x 3 solve accumulate in `f32` (the CPU port:
//!   `f64`, rounded once per stored plane). A replica of the CPU port with the 9 x 9 weighted moments
//!   summed in `f32` and the per-pixel Cramer solve done in `f32` moved the corrected luma by at most
//!   4e-4 DN on 100 x 100 crops around the three worst pixels of a 2048 x 1365 frame and by 0.011 DN
//!   (1.6e-7 normalised) on a hard noise-free edge scene, so these are not what limits the parity: the
//!   ridge (0.0005 of the trace) bounds the condition number of the 3 x 3 system at about 2000;
//! * `L_s` is formed as `Y + strength * delta` instead of `Y + strength * ((Y + delta) - Y)`;
//! * the luma is the one place where `f32` is not good enough. The envelope counts of spec 3.5
//!   compare neighbouring lumas for order and equality, and the plain `f32` sum
//!   `a_R (S R) + a_G (S G) + a_B (S B)` is 1 to 2 ulp off the CPU's `f64` luma in a third of the
//!   pixels. A count that flips moves an envelope by up to 8 %, and on a 2048 x 1365 natural frame
//!   that alone gave an error of 9.1e-5 at a handful of pixels (feeding the CPU port the `f32` luma
//!   and nothing else reproduced the GPU's worst pixels to 0.5 %). The shader therefore forms the
//!   luma as an exact expansion (`exact_luma`: Dekker products and Knuth sums, every rounded result
//!   hidden from the optimiser by an xor with a runtime zero) and gets the CPU's bits:
//!   `the_luma_plane_has_the_cpu_bits` counts 0 differences in 240000 pixels, and the large frame
//!   drops to 4.8e-7. `fma` is not used for the products: lavapipe lowers it to a multiply and an
//!   add, which lost every error term;
//! * a noise-free smooth ramp is outside what can be compared at all. Every window is evenly spread,
//!   the envelopes collapse to `lower = upper = mid`, `Y` sits within a rounding of both layer
//!   thresholds, and the dark/bright split is decided by rounding: the CPU port itself moves by up
//!   to 3.5e-3 when such an input changes by 0.6 ulp. Real frames and every scene here carry sensor
//!   noise (sigma0 is at least 1 DN).
//!
//! There are no transcendental functions on the GPU (the Gaussian taps are computed on the host).
//! See the module documentation of `kernels::diffraction` for the rest.
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); without an adapter these tests skip with a notice, as the other GPU tests do.

mod common;

use std::path::{Path, PathBuf};

use common::{TestGpu, gpus};
use serde_json::Value;
use ym_gpu::cpu::sharpen::CaptureSharpen;
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};
use ym_gpu::cpu::tiling::Stage;
use ym_gpu::cpu::{NoiseDn, RgbImage, mirror, synth};
use ym_gpu::device::{GpuContext, GpuError};
use ym_gpu::kernels::diffraction::PreparedDiffraction;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tiles::TilePlan;
use ym_noise::model::NoiseModel;

/// The image vectors, `f32` GPU against `expected` (the vectors' own tolerance is 2e-5). Measured on
/// lavapipe: 8.9e-7 (`edge_diagonal`), 4.5e-7 (`zone_plate`), 2e-7 and below elsewhere.
const VECTOR_BOUND: f64 = 5e-6;
/// Random frames, flips, tiles, sizes, hostile values and the large frame (the spec's suggested
/// WGSL parity gate is 5e-5). Measured on lavapipe: 5.1e-6 on the random frames (R 2 to 6, f/1 to
/// f/50, five scene kinds), 4.3e-6 for strength 3 and for values from -0.05 to 4, 3.9e-6 over the tiled
/// 203 x 157 frames, 3.8e-6 for the flips, 9.2e-7 for sizes 1 x 1 to 7 x 7, 4.8e-7 on 2048 x 1365.
const FRAME_BOUND: f64 = 2e-5;
/// The stage must change a plain scene by at least this much (normalised units) for the parity
/// comparison to mean something. Measured minimum: 5.2e-2.
const MIN_EFFECT: f64 = 1e-2;

// ---------------------------------------------------------------------------------------------
// Devices, plans and runners
// ---------------------------------------------------------------------------------------------

/// The automatic device. The kernel keeps `f32` planes in every storage mode, so one device is
/// enough for parity; the determinism test runs on all of them.
fn device() -> Option<&'static TestGpu> {
    gpus().first()
}

/// The controls of a scene (the reference's `Params`): square pixels, gains (2, 1, 1.5), about ISO 100.
///
/// `a`, `b`, `sigma_row` and `q` are the noise model the plan is made with; the noise added to the
/// scene is `scene_noise` (the same model by default). They differ in the "zero noise model" case:
/// a stage told there is no noise still gets a noisy photograph. A scene with no noise at all is
/// avoided on purpose: on a smooth noise-free ramp every window is evenly spread, the envelopes
/// collapse to `lower = upper = mid` and the dark/bright split is decided by rounding, on the CPU
/// port too (a 0.6 ulp change of such an input moves its output by up to 3.5e-3).
#[derive(Clone, Copy, Debug)]
struct Cfg {
    pitch_um: f64,
    f_number: f64,
    scale: f64,
    strength: f32,
    g0: [f32; 3],
    a: f64,
    b: f64,
    sigma_row: f64,
    q: f64,
    /// `(a, b, sigma_row)` of the noise added to the scene.
    scene_noise: (f64, f64, f64),
    luma: [f32; 3],
    noise_gate: bool,
}

fn cfg(pitch_um: f64, f_number: f64) -> Cfg {
    Cfg {
        pitch_um,
        f_number,
        scale: 1.0,
        strength: 1.0,
        g0: [2.0, 1.0, 1.5],
        a: 2.5e-5,
        b: 4e-8,
        sigma_row: 5e-5,
        q: 0.0,
        scene_noise: (2.5e-5, 4e-8, 5e-5),
        luma: SPEC_LUMA,
        noise_gate: false,
    }
}

/// `c` with the noise model of the plan and of the scene both set to `(a, b, sigma_row)`.
fn with_noise(c: Cfg, a: f64, b: f64, sigma_row: f64) -> Cfg {
    Cfg {
        a,
        b,
        sigma_row,
        scene_noise: (a, b, sigma_row),
        ..c
    }
}

impl Cfg {
    fn model(&self) -> NoiseModel {
        NoiseModel {
            a: self.a,
            b: self.b,
            delta: [0.0; 4],
            sigma_row: self.sigma_row,
            q: self.q,
        }
    }

    /// Signal-independent variance of the noise added to the scene.
    fn floor_var(&self) -> f64 {
        self.scene_noise.1 + self.scene_noise.2 * self.scene_noise.2 + self.q * self.q / 12.0
    }

    fn setup(&self) -> Setup {
        let params = DiffractionParams::new(self.pitch_um, self.f_number)
            .with_scale(self.scale)
            .with_strength(self.strength);
        let plan = DiffractionPlan::new(
            &params,
            self.g0,
            &NoiseDn::from_model(&self.model()),
            self.luma,
        )
        .expect("an active plan");
        Setup {
            plan: plan.with_noise_gate(self.noise_gate),
            g0: self.g0,
            luma: self.luma,
        }
    }
}

/// A plan with the gains and luma weights it was made with (the kernel needs them again).
#[derive(Clone, Debug)]
struct Setup {
    plan: DiffractionPlan,
    g0: [f32; 3],
    luma: [f32; 3],
}

/// Runs `job` on the device thread with this setup uploaded.
fn with_prepared<R, F>(dev: &TestGpu, s: &Setup, job: F) -> Result<R, GpuError>
where
    R: Send + 'static,
    F: FnOnce(&GpuContext, &CancelToken, &PreparedDiffraction) -> Result<R, GpuError>
        + Send
        + 'static,
{
    let s = s.clone();
    dev.gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let prepared = PreparedDiffraction::new(ctx, &s.plan, s.g0, s.luma)?;
            job(ctx, token, &prepared)
        })
}

fn run_gpu(dev: &TestGpu, s: &Setup, img: &RgbImage) -> RgbImage {
    let img = img.clone();
    with_prepared(dev, s, move |ctx, _, p| p.run(ctx, &img)).expect("GPU diffraction run")
}

fn run_gpu_tiled(dev: &TestGpu, s: &Setup, img: &RgbImage, tile_px: usize) -> RgbImage {
    let img = img.clone();
    with_prepared(dev, s, move |ctx, token, p| {
        p.run_tiled(ctx, &img, tile_px, token)
    })
    .expect("GPU tiled diffraction run")
}

/// The CPU port's whole-buffer run, the reference of the parity tests.
fn cpu(s: &Setup, img: &RgbImage) -> RgbImage {
    s.plan.apply_whole(img)
}

/// Largest `|a - b|` and where it is (`plane`, `x`, `y`).
struct Diff {
    max_abs: f64,
    at: (usize, usize, usize),
}

fn diff(a: &RgbImage, b: &RgbImage) -> Diff {
    assert_eq!((a.width, a.height), (b.width, b.height), "image sizes");
    let mut d = Diff {
        max_abs: 0.0,
        at: (0, 0, 0),
    };
    for c in 0..3 {
        for (i, (&x, &y)) in a.plane(c).iter().zip(b.plane(c)).enumerate() {
            assert!(x.is_finite(), "GPU sample ({c}, {i}) is {x}");
            let e = (f64::from(x) - f64::from(y)).abs();
            if e > d.max_abs {
                d.max_abs = e;
                d.at = (c, i % a.width, i / a.width);
            }
        }
    }
    d
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------------------------------
// Scenes: the reference's recipe (a sharp scene, blurred by the plan's own PSF, with noise from
// the model, independent per channel in unbalanced units)
// ---------------------------------------------------------------------------------------------

/// Blurs every channel with the plan's kernel (mirror borders), simulated diffraction: an
/// independent, direct implementation of the convolution.
fn psf_blur(img: &RgbImage, plan: &DiffractionPlan) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let r = plan.radius() as isize;
    let kw = 2 * plan.radius() + 1;
    let mut out = img.clone();
    for c in 0..3 {
        let src = img.plane(c);
        let dst = out.plane_mut(c);
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f64;
                for dy in -r..=r {
                    let sy = mirror(y as isize + dy, h);
                    for dx in -r..=r {
                        let k =
                            f64::from(plan.kernel()[(dy + r) as usize * kw + (dx + r) as usize]);
                        s += k * f64::from(src[sy * w + mirror(x as isize + dx, w)]);
                    }
                }
                dst[y * w + x] = s as f32;
            }
        }
    }
    out
}

/// Poisson-Gaussian noise from the model, in unbalanced units, independent per channel.
fn add_noise(img: &mut RgbImage, c: &Cfg, seed: u64) {
    let mut rng = synth::Rng::new(seed);
    let n = img.plane_len();
    let floor_var = c.floor_var();
    for ch in 0..3 {
        let g = f64::from(c.g0[ch]);
        for ix in 0..n {
            let u = f64::from(img.data[ch * n + ix]) / g;
            let sd = (c.scene_noise.0 * u.max(0.0) + floor_var).sqrt();
            img.data[ch * n + ix] = (g * (u + sd * f64::from(rng.gaussian()))) as f32;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Smooth fields and soft-edged shapes.
    Natural,
    /// Saturated colour regions with straight and slanted edges.
    Edges,
    /// Fine text in several colours.
    Text,
    /// Natural, with values up to 4 (R5 reconstructions and highlights above white).
    Highlights,
    /// Dark scene whose noise crosses zero (negative samples).
    Dark,
    /// Natural, overexposed and clipped at white after the noise: plateaus of exactly equal
    /// samples (ties in the envelope counts) next to the clip penalty.
    Clipped,
}

/// A balanced, blurred and noisy frame of `kind`.
fn scene(kind: Kind, w: usize, h: usize, seed: u64, c: &Cfg, plan: &DiffractionPlan) -> RgbImage {
    // The clip penalty starts at 0.8 of white (spec 2.4): the plain scenes stay below it so the
    // stage works (a window with a clipped sample is gated), the highlight scenes go far above.
    let (mut sharp, peak) = match kind {
        Kind::Natural => (synth::natural(w, h, seed), 0.75f32),
        Kind::Edges => (synth::edges(w, h), 0.75),
        Kind::Text => (synth::fine_text(w, h, 3), 0.75),
        Kind::Highlights => (synth::natural(w, h, seed), 4.0),
        Kind::Dark => (synth::natural(w, h, seed), 0.012),
        Kind::Clipped => (synth::natural(w, h, seed), 1.5),
    };
    // Unbalanced values, then the gains (the stage sees balanced data).
    for v in &mut sharp.data {
        *v *= peak;
    }
    let mut img = psf_blur(&sharp, plan);
    let n = img.plane_len();
    for ch in 0..3 {
        for v in &mut img.data[ch * n..(ch + 1) * n] {
            *v *= c.g0[ch];
        }
    }
    add_noise(&mut img, c, seed ^ 0x6e6f_6973);
    if kind == Kind::Clipped {
        for ch in 0..3 {
            for v in &mut img.data[ch * n..(ch + 1) * n] {
                *v = v.min(c.g0[ch]);
            }
        }
    }
    img
}

fn flat(w: usize, h: usize, rgb: [f32; 3]) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for (ch, &v) in rgb.iter().enumerate() {
        img.plane_mut(ch).fill(v);
    }
    img
}

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/diffraction_deconvolution")
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}"))
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// The setup a vector's `params` describe, or `None` when the stage is inactive.
fn setup_of(params: &Value) -> Option<Setup> {
    let g0 = floats(&params["g0"]);
    let luma = floats(&params["luma"]);
    let n = &params["noise"];
    let model = NoiseModel {
        a: num(n, "a"),
        b: num(n, "b"),
        delta: [0.0; 4],
        sigma_row: num(n, "sigma_row"),
        q: num(n, "q"),
    };
    let p = DiffractionParams {
        pixel_pitch_um: [
            num(params, "pixel_pitch_x_um"),
            num(params, "pixel_pitch_y_um"),
        ],
        f_number: num(params, "f_number"),
        scale: num(params, "scale"),
        strength: num(params, "strength") as f32,
    };
    let g0 = [g0[0], g0[1], g0[2]];
    let luma = [luma[0], luma[1], luma[2]];
    DiffractionPlan::new(&p, g0, &NoiseDn::from_model(&model), luma).map(|plan| Setup {
        plan,
        g0,
        luma,
    })
}

/// The image vectors: name, setup (`None` for an inactive stage), input, expected data, tolerance.
struct Vector {
    name: String,
    setup: Option<Setup>,
    input: RgbImage,
    expected: Vec<f32>,
    tolerance: f64,
}

fn image_vectors() -> Vec<Vector> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(vector_dir())
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).expect("vector file");
        let v: Value = serde_json::from_str(&text).expect("vector json");
        if v["input"].is_null() {
            continue; // planner vectors: nothing per pixel to run
        }
        let input = &v["input"];
        let (w, h) = (num(input, "width") as usize, num(input, "height") as usize);
        assert_eq!(input["layout"], "planar_rgb");
        out.push(Vector {
            name: path
                .file_stem()
                .expect("stem")
                .to_string_lossy()
                .into_owned(),
            setup: setup_of(&v["params"]),
            input: RgbImage::from_planar(w, h, floats(&input["data"])).expect("input image"),
            expected: floats(&v["expected"]["data"]),
            tolerance: num(&v, "tolerance"),
        });
    }
    out
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "plane lengths");
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn every_image_vector_matches_on_the_gpu() {
    let Some(dev) = device() else { return };
    let vectors = image_vectors();
    assert_eq!(vectors.len(), 13, "the image vector set");
    let mut worst = 0.0f64;
    for v in &vectors {
        let Some(setup) = &v.setup else {
            // An inactive stage returns its input; there is no plan to upload.
            assert_eq!(
                v.tolerance, 0.0,
                "{}: only exact vectors are inactive",
                v.name
            );
            assert!(bits_equal(&v.expected, &v.input.data), "{}", v.name);
            continue;
        };
        let got = run_gpu(dev, setup, &v.input);
        assert_eq!((got.width, got.height), (v.input.width, v.input.height));
        let d = max_abs_diff(&got.data, &v.expected);
        let against_cpu = diff(&got, &cpu(setup, &v.input));
        println!(
            "{:<28} {:>2}x{:<2}: max |gpu - expected| {d:.3e} (vector tolerance {:.0e}, asserted {VECTOR_BOUND:.0e}); against the CPU port {:.3e}",
            v.name, v.input.width, v.input.height, v.tolerance, against_cpu.max_abs
        );
        if v.tolerance == 0.0 {
            // Flat fields: an exact identity, on the GPU as on the CPU.
            assert!(
                bits_equal(&got.data, &v.input.data),
                "{}: the exact vector is not bit-equal to its input",
                v.name
            );
            assert!(bits_equal(&got.data, &v.expected), "{}", v.name);
        } else {
            assert!(
                d <= v.tolerance,
                "{}: the GPU differs from expected by {d}, the vector's tolerance is {}",
                v.name,
                v.tolerance
            );
            assert!(
                d <= VECTOR_BOUND,
                "{}: the GPU differs from expected by {d}, asserted bound {VECTOR_BOUND}",
                v.name
            );
            worst = worst.max(d);
        }
    }
    println!("vectors, worst |gpu - expected|: {worst:.3e}");
}

// ---------------------------------------------------------------------------------------------
// Random frames
// ---------------------------------------------------------------------------------------------

#[test]
fn random_frames_match_the_cpu_across_plans_scenes_and_sizes() {
    let Some(dev) = device() else { return };
    // (pitch um, f-number): kernel radius 2 to 7 and Gaussian radius 1 to 8, halos 12 to 25.
    let optics = [
        (3.76, 2.8),
        (3.76, 5.6),
        (3.76, 8.0),
        (3.76, 11.0),
        (3.76, 16.0),
        (3.76, 22.0),
        (3.76, 50.0),
        (2.4, 4.0),
        (2.4, 22.0),
        (5.93, 11.0),
        (8.4, 1.0),
    ];
    let mut worst = 0.0f64;
    let mut worst_effect_min = f64::MAX;
    let mut seed = 100u64;
    for &(pitch, f) in &optics {
        let c = cfg(pitch, f);
        let setup = c.setup();
        for (kind, w, h) in [
            (Kind::Natural, 67usize, 41usize),
            (Kind::Edges, 64, 48),
            (Kind::Text, 80, 56),
            (Kind::Highlights, 53, 61),
            (Kind::Dark, 45, 38),
            (Kind::Clipped, 58, 47),
        ] {
            seed += 1;
            let img = scene(kind, w, h, seed, &c, &setup.plan);
            let got = run_gpu(dev, &setup, &img);
            let want = cpu(&setup, &img);
            let d = diff(&got, &want);
            // What the stage did, so that a pass is not a pass on a gated frame.
            let effect = diff(&want, &img).max_abs;
            println!(
                "{pitch:>4} um f/{f:<4} R {} r_G {} halo {:>2} {kind:?} {w}x{h}: max |gpu - cpu| {:.3e} at plane {} ({}, {}); the stage changed the frame by up to {effect:.3e}",
                setup.plan.radius(),
                setup.plan.gauss_radius(),
                setup.plan.halo_px(),
                d.max_abs,
                d.at.0,
                d.at.1,
                d.at.2
            );
            if matches!(kind, Kind::Natural | Kind::Text | Kind::Edges) && f >= 5.6 {
                assert!(
                    effect >= MIN_EFFECT,
                    "{pitch} um f/{f} {kind:?}: the stage did nothing ({effect:e})"
                );
                worst_effect_min = worst_effect_min.min(effect);
            }
            assert!(
                d.max_abs <= FRAME_BOUND,
                "{pitch} um f/{f} {kind:?}: {} > {FRAME_BOUND}",
                d.max_abs
            );
            worst = worst.max(d.max_abs);
        }
    }
    println!(
        "random frames, worst |gpu - cpu|: {worst:.3e} (bound {FRAME_BOUND:.0e}); the smallest effect of the stage on a plain scene at f/5.6 or more: {worst_effect_min:.3e}"
    );
}

#[test]
fn strengths_scales_noise_levels_gates_and_luma_weights_match_the_cpu() {
    let Some(dev) = device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 300u64;
    let base = cfg(3.76, 11.0);
    let variants: Vec<(&str, Cfg)> = vec![
        (
            "strength 0.5",
            Cfg {
                strength: 0.5,
                ..base
            },
        ),
        (
            "strength 3",
            Cfg {
                strength: 3.0,
                ..base
            },
        ),
        (
            "scale 1/4",
            Cfg {
                scale: 0.25,
                ..base
            },
        ),
        ("scale 0.6", Cfg { scale: 0.6, ..base }),
        ("ISO 6400", with_noise(base, 1.6e-3, 2.5e-6, 3e-4)),
        (
            // The plan is told there is no noise (sigma0 at its 1 DN floor, no shot noise); the
            // scene still carries the ISO 100 noise.
            "zero noise model",
            Cfg {
                a: 0.0,
                b: 0.0,
                sigma_row: 0.0,
                ..base
            },
        ),
        (
            "noise-gate variant, f/11",
            Cfg {
                noise_gate: true,
                ..base
            },
        ),
        (
            "noise-gate variant, f/22",
            Cfg {
                noise_gate: true,
                ..cfg(3.76, 22.0)
            },
        ),
        (
            "Rec. 709 luma weights",
            Cfg {
                luma: [0.2126, 0.7152, 0.0722],
                ..base
            },
        ),
        (
            "unit gains",
            Cfg {
                g0: [1.0, 1.0, 1.0],
                ..base
            },
        ),
        (
            "large gains",
            Cfg {
                g0: [2.6, 1.0, 1.9],
                ..base
            },
        ),
        (
            "unequal pitches",
            // The larger pitch is used; the plan is the same as a square 4.0 um.
            Cfg {
                pitch_um: 4.0,
                ..base
            },
        ),
    ];
    for (name, c) in variants {
        let setup = c.setup();
        for kind in [Kind::Natural, Kind::Edges, Kind::Highlights] {
            seed += 1;
            let img = scene(kind, 71, 49, seed, &c, &setup.plan);
            let got = run_gpu(dev, &setup, &img);
            let d = diff(&got, &cpu(&setup, &img));
            println!(
                "{name:<26} {kind:?}: max |gpu - cpu| {:.3e} at plane {} ({}, {})",
                d.max_abs, d.at.0, d.at.1, d.at.2
            );
            assert!(
                d.max_abs <= FRAME_BOUND,
                "{name} {kind:?}: {} > {FRAME_BOUND}",
                d.max_abs
            );
            worst = worst.max(d.max_abs);
        }
    }
    println!("variants, worst |gpu - cpu|: {worst:.3e} (bound {FRAME_BOUND:.0e})");
}

#[test]
fn tiny_buffers_wrap_the_mirror_like_the_cpu() {
    // Narrower than the halo (12 to 25 px): the mirror wraps several times (vector `tiny_7x5`).
    // Sizes down to 1 x 1.
    let Some(dev) = device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 5000u64;
    for (pitch, f) in [(3.76, 8.0), (2.4, 22.0)] {
        let c = cfg(pitch, f);
        let setup = c.setup();
        for h in 1..=7usize {
            for w in 1..=7usize {
                seed += 1;
                let img = scene(Kind::Natural, w, h, seed, &c, &setup.plan);
                let got = run_gpu(dev, &setup, &img);
                let d = diff(&got, &cpu(&setup, &img));
                assert!(d.max_abs <= FRAME_BOUND, "{w}x{h}: {}", d.max_abs);
                worst = worst.max(d.max_abs);
            }
        }
    }
    // A 1-pixel-wide strip and a long thin frame.
    let c = cfg(3.76, 11.0);
    let setup = c.setup();
    for (w, h) in [(1usize, 40usize), (40, 1), (2, 63), (97, 3)] {
        seed += 1;
        let img = scene(Kind::Edges, w, h, seed, &c, &setup.plan);
        let got = run_gpu(dev, &setup, &img);
        let d = diff(&got, &cpu(&setup, &img));
        assert!(d.max_abs <= FRAME_BOUND, "{w}x{h}: {}", d.max_abs);
        worst = worst.max(d.max_abs);
    }
    println!("sizes 1x1 to 7x7 and thin strips, worst |gpu - cpu|: {worst:.3e}");
}

#[test]
fn hostile_values_do_not_produce_nan_or_inf_and_match_the_cpu() {
    // Spec 6.2 item 7: sizes up to 48 x 40, values from -0.05 to 4, a zero noise model, strength 3,
    // f/1 to f/50.
    let Some(dev) = device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 9000u64;
    for (pitch, f) in [(3.76, 1.0), (3.76, 8.0), (2.4, 50.0)] {
        // The hostile frames carry their own heavy noise; the plan is told there is none.
        let c = Cfg {
            a: 0.0,
            b: 0.0,
            sigma_row: 0.0,
            strength: 3.0,
            ..cfg(pitch, f)
        };
        let setup = c.setup();
        for (w, h) in [(48usize, 40usize), (31, 17), (1, 1), (16, 16)] {
            seed += 1;
            let mut rng = synth::Rng::new(seed);
            let mut img = RgbImage::new(w, h);
            for v in &mut img.data {
                let u = rng.next_f32();
                *v = if u < 0.1 {
                    -0.05 * rng.next_f32()
                } else if u > 0.9 {
                    4.0 * rng.next_f32()
                } else {
                    u
                };
            }
            let got = run_gpu(dev, &setup, &img);
            assert!(
                got.data.iter().all(|v| v.is_finite()),
                "NaN or Inf at {w}x{h}"
            );
            let d = diff(&got, &cpu(&setup, &img));
            println!(
                "f/{f} {w}x{h}: max |gpu - cpu| {:.3e} (peak input 4)",
                d.max_abs
            );
            assert!(d.max_abs <= FRAME_BOUND, "{w}x{h} f/{f}: {}", d.max_abs);
            worst = worst.max(d.max_abs);
        }
    }
    println!("values -0.05..4, strength 3, worst |gpu - cpu|: {worst:.3e}");
}

#[test]
fn hostile_values_that_the_stage_acts_on_match_the_cpu() {
    // Review of RA-09: in the test above, every 9 x 9 window of the 0.1 fraction of samples up to 4
    // holds a sample far above white, and at f/1 and f/50 the CPU port returns its input unchanged
    // (the clip penalty gates the frame, or the fit explains too little and fades out), so that
    // comparison is an identity there and proves little about the solve. Here the values are
    // uniform in [-0.05, 0.7] (below the clip threshold, a tenth of them negative), without and with
    // 1 % of the pixels at up to 4 (windows near a spike are gated, the rest is not), strength 3,
    // zero noise model. From f/1 to f/8 the stage must change the frame, or the check means nothing
    // (from f/16 on, pure noise is faded out by the fit and the CPU returns its input, so those
    // optics are left to the other tests). Measured on lavapipe: changes up to 2.0 (strength 3
    // extrapolates), |gpu - cpu| at most 1.3e-6.
    let Some(dev) = device() else { return };
    let mut worst = 0.0f64;
    let mut seed = 9500u64;
    for (pitch, f) in [(3.76, 1.0), (3.76, 2.8), (3.76, 5.6), (3.76, 8.0)] {
        let c = Cfg {
            a: 0.0,
            b: 0.0,
            sigma_row: 0.0,
            strength: 3.0,
            ..cfg(pitch, f)
        };
        let setup = c.setup();
        for spikes in [0.0f32, 0.01] {
            for (w, h) in [(48usize, 40usize), (31, 17)] {
                seed += 1;
                let mut rng = synth::Rng::new(seed);
                let mut img = RgbImage::new(w, h);
                let n = w * h;
                for i in 0..n {
                    let spike = rng.next_f32() < spikes;
                    for ch in 0..3 {
                        let u = rng.next_f32();
                        img.data[ch * n + i] = if spike {
                            4.0 * rng.next_f32()
                        } else if u < 0.1 {
                            -0.05 * rng.next_f32()
                        } else {
                            0.7 * u
                        };
                    }
                }
                let got = run_gpu(dev, &setup, &img);
                assert!(
                    got.data.iter().all(|v| v.is_finite()),
                    "NaN or Inf at {w}x{h} f/{f}"
                );
                let want = cpu(&setup, &img);
                let d = diff(&got, &want);
                let effect = diff(&want, &img).max_abs;
                println!(
                    "f/{f} {w}x{h} spikes {spikes}: max |gpu - cpu| {:.3e}; the stage changed the frame by up to {effect:.3e}",
                    d.max_abs
                );
                if w == 48 {
                    assert!(
                        effect > 1e-3,
                        "f/{f} {w}x{h} spikes {spikes}: the stage did nothing ({effect:e})"
                    );
                }
                assert!(
                    d.max_abs <= FRAME_BOUND,
                    "f/{f} {w}x{h} spikes {spikes}: {} > {FRAME_BOUND}",
                    d.max_abs
                );
                worst = worst.max(d.max_abs);
            }
        }
    }
    println!("hostile frames the stage acts on, worst |gpu - cpu|: {worst:.3e}");
}

#[test]
fn a_non_finite_sample_stays_local_and_does_not_panic() {
    // Non-finite input is outside the contract (spec 4: not guarded, it propagates), but a bad
    // sample must neither crash the kernel nor reach farther than the receptive field: every output
    // sample beyond the halo is bit-identical to the run without it. A huge finite value stays
    // finite (the clip penalty gates its neighbourhood).
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 11.0);
    let setup = c.setup();
    let halo = setup.plan.halo_px();
    let (w, h) = (90usize, 70usize);
    let clean = scene(Kind::Natural, w, h, 21, &c, &setup.plan);
    let base = run_gpu(dev, &setup, &clean);
    let (bx, by) = (40usize, 35usize);
    for (name, bad) in [
        ("NaN", f32::NAN),
        ("+Inf", f32::INFINITY),
        ("-Inf", f32::NEG_INFINITY),
        ("1e30", 1e30),
    ] {
        let mut dirty = clean.clone();
        dirty.data[w * h + by * w + bx] = bad;
        let out = run_gpu(dev, &setup, &dirty);
        let mut reach = 0usize;
        for ch in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let j = ch * w * h + y * w + x;
                    if out.data[j].to_bits() != base.data[j].to_bits() {
                        reach = reach.max(x.abs_diff(bx).max(y.abs_diff(by)));
                    }
                }
            }
        }
        println!("{name}: the bad sample changes pixels up to {reach} px away (halo {halo})");
        assert!(reach <= halo, "{name}: reach {reach} > halo {halo}");
        if bad == 1e30 {
            assert!(
                out.data.iter().all(|v| v.is_finite()),
                "1e30 left a non-finite sample"
            );
        }
    }
}

#[test]
fn flipping_and_transposing_the_frame_flips_the_output() {
    // Spec 6.2 item 4: the stage commutes with the symmetries of the square. The GPU sums in f32 in a
    // fixed order, so the flipped run differs from the flipped output by rounding, not by the bit.
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 11.0);
    let setup = c.setup();
    let (w, h) = (51usize, 38usize);
    let img = scene(Kind::Edges, w, h, 41, &c, &setup.plan);
    let base = run_gpu(dev, &setup, &img);
    let mut worst = 0.0f64;
    for t in 0..3 {
        let mut moved = if t == 0 {
            RgbImage::new(h, w)
        } else {
            RgbImage::new(w, h)
        };
        for y in 0..h {
            for x in 0..w {
                let px = img.pixel(x, y);
                match t {
                    0 => moved.set_pixel(y, x, px),
                    1 => moved.set_pixel(w - 1 - x, y, px),
                    _ => moved.set_pixel(x, h - 1 - y, px),
                }
            }
        }
        let got = run_gpu(dev, &setup, &moved);
        for y in 0..h {
            for x in 0..w {
                let want = base.pixel(x, y);
                let have = match t {
                    0 => got.pixel(y, x),
                    1 => got.pixel(w - 1 - x, y),
                    _ => got.pixel(x, h - 1 - y),
                };
                for ch in 0..3 {
                    worst = worst.max((f64::from(want[ch]) - f64::from(have[ch])).abs());
                }
            }
        }
    }
    println!("transpose, flip x, flip y: worst |moved run - moved output| {worst:.3e}");
    assert!(worst <= FRAME_BOUND, "{worst}");
}

// ---------------------------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------------------------

#[test]
fn flat_fields_and_strength_zero_come_back_bit_exact() {
    // Spec 6.2 items 1 and 2: flat fields (including above white and slightly negative) and
    // strength 0 are identities.
    let Some(dev) = device() else { return };
    let levels: [[f32; 3]; 6] = [
        [0.0; 3],
        [0.18; 3],
        [0.42, 0.30, 0.18],
        [1.9, 1.3, 1.6],
        [3.5, 2.0, 2.9],
        [-0.02, -0.01, -0.03],
    ];
    for (pitch, f) in [(3.76, 2.8), (3.76, 16.0)] {
        let setup = cfg(pitch, f).setup();
        for lv in levels {
            let img = flat(37, 29, lv);
            let got = run_gpu(dev, &setup, &img);
            assert!(
                bits_equal(&got.data, &img.data),
                "f/{f} flat {lv:?}: not bit-exact"
            );
        }
    }
    // Strength 0 on a busy frame.
    let c = Cfg {
        strength: 0.0,
        ..cfg(3.76, 16.0)
    };
    let setup = c.setup();
    for kind in [Kind::Edges, Kind::Highlights, Kind::Dark] {
        let img = scene(kind, 60, 44, 7, &c, &setup.plan);
        let got = run_gpu(dev, &setup, &img);
        assert!(
            bits_equal(&got.data, &img.data),
            "strength 0 {kind:?}: not bit-exact"
        );
    }
}

#[test]
fn the_output_keeps_the_placement_of_the_input() {
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 8.0);
    let setup = c.setup();
    let mut img = scene(Kind::Natural, 40, 30, 11, &c, &setup.plan);
    img.place = ym_gpu::cpu::Placement {
        x0: 12,
        y0: 8,
        frame_width: 400,
        frame_height: 300,
    };
    let got = run_gpu(dev, &setup, &img);
    assert_eq!(got.place, img.place);
    let tiled = run_gpu_tiled(dev, &setup, &img, 16);
    assert_eq!(tiled.place, img.place);
}

// ---------------------------------------------------------------------------------------------
// Tiling and the large frame
// ---------------------------------------------------------------------------------------------

#[test]
fn tiles_with_the_exact_halo_equal_the_whole_frame_at_tile_and_halo_boundaries() {
    let Some(dev) = device() else { return };
    let (w, h) = (203usize, 157usize);
    // The smallest, a middle and the largest halo.
    for (pitch, f, kind, seed) in [
        (3.76, 2.8, Kind::Natural, 31u64),
        (3.76, 16.0, Kind::Edges, 32),
        (2.4, 50.0, Kind::Text, 33),
    ] {
        let c = cfg(pitch, f);
        let setup = c.setup();
        let img = scene(kind, w, h, seed, &c, &setup.plan);
        let whole = run_gpu(dev, &setup, &img);
        let reference = cpu(&setup, &img);
        let d_whole = diff(&whole, &reference);
        assert!(d_whole.max_abs <= FRAME_BOUND, "{}", d_whole.max_abs);
        for tile_px in [32usize, 50, 64, 100] {
            let tiled = run_gpu_tiled(dev, &setup, &img, tile_px);
            let to_cpu = diff(&tiled, &reference);
            println!(
                "{pitch} um f/{f} halo {} {w}x{h} tiles of {tile_px}: tiled against the CPU {:.3e} (whole frame {:.3e})",
                setup.plan.halo_px(),
                to_cpu.max_abs,
                d_whole.max_abs
            );
            // Same operations in the same order on the same inputs: identical bits.
            assert!(
                bits_equal(&tiled.data, &whole.data),
                "f/{f} tile {tile_px}: not bit-identical to the whole frame"
            );
            assert!(to_cpu.max_abs <= FRAME_BOUND, "{}", to_cpu.max_abs);
        }
    }
}

#[test]
fn a_tile_plan_built_from_the_stage_gives_the_same_pixels() {
    // `TilePlan::new` over `Stage::Sharpen(Diffraction)` is what the scheduler would build.
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 22.0);
    let setup = c.setup();
    let img = scene(Kind::Natural, 150, 120, 3, &c, &setup.plan);
    let whole = run_gpu(dev, &setup, &img);
    let params = DiffractionParams::new(3.76, 22.0);
    let stage = Stage::Sharpen(CaptureSharpen::Diffraction(params));
    let plan = TilePlan::new(150, 120, 48, &[stage]);
    let img2 = img.clone();
    let tiled = with_prepared(dev, &setup, move |ctx, token, p| {
        assert_eq!(p.halo_px(), params.halo_px());
        p.run_planned(ctx, &img2, &plan, token)
    })
    .expect("planned run");
    assert!(bits_equal(&tiled.data, &whole.data));
}

#[test]
fn a_large_frame_through_the_tiling_path_matches_the_cpu() {
    let Some(dev) = device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let c = cfg(3.76, 11.0);
    let setup = c.setup();
    let img = scene(Kind::Natural, w, h, 2048, &c, &setup.plan);
    let started = std::time::Instant::now();
    let tiled = run_gpu_tiled(dev, &setup, &img, 512);
    let gpu_time = started.elapsed();
    let started = std::time::Instant::now();
    let reference = setup.plan.apply(&img);
    let cpu_time = started.elapsed();
    let d = diff(&tiled, &reference);
    let samples = 3 * w * h;
    let over = |t: f64| {
        tiled
            .data
            .iter()
            .zip(&reference.data)
            .filter(|(a, b)| (f64::from(**a) - f64::from(**b)).abs() > t)
            .count()
    };
    println!(
        "{w}x{h} in tiles of 512 ({} tiles, halo {}): max |gpu - cpu| {:.3e} at plane {} ({}, {}); samples above 1e-6: {}, 5e-6: {}, 1e-5: {} of {samples}; the stage changed the frame by up to {:.3e}; GPU {gpu_time:.2?} on {}, CPU {cpu_time:.2?}",
        TilePlan::new(w, h, 512, &[]).tiles().len(),
        setup.plan.halo_px(),
        d.max_abs,
        d.at.0,
        d.at.1,
        d.at.2,
        over(1e-6),
        over(5e-6),
        over(1e-5),
        diff(&reference, &img).max_abs,
        dev.gpu.info().name
    );
    assert!(d.max_abs <= FRAME_BOUND, "{}", d.max_abs);
    // At most one sample in 100000 above 1e-5 (measured: none above 1e-6).
    assert!(
        over(1e-5) * 100_000 <= samples,
        "{} samples above 1e-5",
        over(1e-5)
    );
}

#[test]
fn the_luma_plane_has_the_cpu_bits() {
    // The envelope counts of spec 3.5 compare neighbouring lumas for order and equality, so a luma
    // that is 1 to 2 ulp off in a third of the pixels (the plain f32 formula) moves counts and, through
    // them, the layers: on a 2048 x 1365 frame that alone gave an error of 9.1e-5. The shader forms the
    // luma with a double-float expansion (`exact_luma`) and gets the correctly rounded f32 of the exact
    // value, which is what the CPU port's f64 sum rounds to except in about one pixel in 10^7 or fewer.
    let Some(dev) = device() else { return };
    let mut checked = 0usize;
    for (kind, seed, gains) in [
        (Kind::Natural, 21u64, [2.0f32, 1.0, 1.5]),
        (Kind::Highlights, 22, [2.6, 1.0, 1.9]),
        (Kind::Dark, 23, [1.0, 1.0, 1.0]),
        (Kind::Clipped, 24, [2.0, 1.0, 1.5]),
    ] {
        let c = Cfg {
            g0: gains,
            ..cfg(3.76, 11.0)
        };
        let setup = c.setup();
        let img = scene(kind, 300, 200, seed, &c, &setup.plan);
        let (_, cpu_luma) = setup.plan.corrected_luma(&img);
        let (_, trace) = with_prepared(dev, &setup, {
            let img = img.clone();
            move |ctx, _, p| p.run_traced(ctx, &img)
        })
        .expect("traced run");
        assert_eq!(trace.luma.len(), cpu_luma.len());
        let mismatches = trace
            .luma
            .iter()
            .zip(&cpu_luma)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{kind:?}: {mismatches} of {} luma values differ from the CPU's",
            cpu_luma.len()
        );
        assert_eq!(mismatches, 0, "{kind:?}");
        checked += cpu_luma.len();
        // D + B = Y up to the rounding of the middle case (spec 3.6).
        for i in 0..cpu_luma.len() {
            let y = f64::from(trace.luma[i]);
            let sum = f64::from(trace.dark[i]) + f64::from(trace.bright[i]);
            assert!(
                (sum - y).abs() <= 1.5 * f64::from(f32::EPSILON) * y.abs().max(1e-30) + 1e-30,
                "{kind:?} pixel {i}: D + B = {sum}, Y = {y}"
            );
        }
    }
    println!("{checked} luma values checked");
}

#[test]
fn one_pixel_reaches_exactly_the_halo_and_no_further() {
    // Spec 1.5: every output pixel depends on the input within 10 + R + r_G pixels. A perturbation
    // must leave every pixel farther than that bit-identical.
    let Some(dev) = device() else { return };
    let (w, h) = (101usize, 91usize);
    for (pitch, f) in [(3.76, 8.0), (3.76, 22.0)] {
        let c = cfg(pitch, f);
        let setup = c.setup();
        let halo = setup.plan.halo_px();
        let img = scene(Kind::Natural, w, h, 55, &c, &setup.plan);
        let base = run_gpu(dev, &setup, &img);
        let (cx, cy) = (50usize, 45usize);
        // A spike that stays below the clip penalty (0.8 of white, spec 2.4), which would gate the
        // window and hide the reach.
        let mut bumped = img.clone();
        for ch in 0..3 {
            let i = ch * w * h + cy * w + cx;
            let g = c.g0[ch];
            bumped.data[i] = if bumped.data[i] > 0.3 * g {
                0.05 * g
            } else {
                0.7 * g
            };
        }
        let out = run_gpu(dev, &setup, &bumped);
        let mut reach = 0usize;
        for ch in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let j = ch * w * h + y * w + x;
                    if out.data[j].to_bits() != base.data[j].to_bits() {
                        reach = reach.max(x.abs_diff(cx).max(y.abs_diff(cy)));
                    }
                }
            }
        }
        println!("f/{f}: receptive-field radius on the GPU {reach} px (halo {halo})");
        assert!(reach <= halo, "f/{f}: reach {reach} > halo {halo}");
        assert!(reach > 0, "the perturbation must change something");
    }
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn two_gpu_runs_are_bit_identical_in_every_storage_mode() {
    let c = cfg(3.76, 16.0);
    let setup = c.setup();
    let img = scene(Kind::Highlights, 97, 61, 1234, &c, &setup.plan);
    let mut first: Option<RgbImage> = None;
    for dev in gpus() {
        let a = run_gpu(dev, &setup, &img);
        let b = run_gpu(dev, &setup, &img);
        assert!(bits_equal(&a.data, &b.data), "storage {}", dev.mode.name());
        let ta = run_gpu_tiled(dev, &setup, &img, 40);
        let tb = run_gpu_tiled(dev, &setup, &img, 40);
        assert!(
            bits_equal(&ta.data, &tb.data),
            "tiled, storage {}",
            dev.mode.name()
        );
        // The planes are f32 in every storage mode: the same bits on every device of this adapter.
        match &first {
            None => first = Some(a),
            Some(f) => assert!(
                bits_equal(&f.data, &a.data),
                "storage {} differs from the first device",
                dev.mode.name()
            ),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Errors and sizes
// ---------------------------------------------------------------------------------------------

#[test]
fn bad_inputs_are_refused() {
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 11.0);
    let setup = c.setup();
    let img = scene(Kind::Natural, 64, 48, 3, &c, &setup.plan);
    let empty = RgbImage {
        width: 0,
        height: 0,
        place: img.place,
        data: Vec::new(),
    };
    let mut short = img.clone();
    short.data.pop();
    // A plan without any halo, and one for another size.
    let no_context = TilePlan::new(64, 48, 32, &[]);
    let wrong_size = TilePlan::with_halo(65, 48, 32, setup.plan.halo_px());
    let (img2, empty2, short2) = (img.clone(), empty.clone(), short.clone());
    let errors = with_prepared(dev, &setup, move |ctx, token, p| {
        let a = p.run(ctx, &empty2).err();
        let b = p.run(ctx, &short2).err();
        let c = p.run_planned(ctx, &img2, &no_context, token).err();
        let d = p.run_planned(ctx, &img2, &wrong_size, token).err();
        let e = p.run_tiled(ctx, &empty2, 32, token).err();
        Ok((a, b, c, d, e))
    })
    .expect("job");
    for (what, e) in [
        ("empty image", errors.0),
        ("short buffer", errors.1),
        ("no context", errors.2),
        ("wrong size", errors.3),
        ("empty image, tiled", errors.4),
    ] {
        assert!(
            matches!(e, Some(GpuError::InvalidInput(_))),
            "{what}: {e:?}"
        );
    }
    // Gains the planner would refuse.
    let s = setup.clone();
    let bad_gain = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, _| {
            Ok(PreparedDiffraction::new(ctx, &s.plan, [2.0, 0.0, 1.5], s.luma).err())
        })
        .expect("job");
    assert!(
        matches!(bad_gain, Some(GpuError::InvalidInput(_))),
        "{bad_gain:?}"
    );
}

#[test]
fn a_cancelled_token_stops_the_tiled_run() {
    let Some(dev) = device() else { return };
    let c = cfg(3.76, 8.0);
    let setup = c.setup();
    let img = scene(Kind::Natural, 96, 64, 4, &c, &setup.plan);
    let e = with_prepared(dev, &setup, move |ctx, _, p| {
        let token = CancelToken::new();
        token.cancel();
        Ok(p.run_tiled(ctx, &img, 32, &token).err())
    })
    .expect("job");
    assert!(matches!(e, Some(GpuError::Cancelled)), "{e:?}");
}

#[test]
fn the_recommended_tile_size_fits_the_device() {
    let Some(dev) = device() else { return };
    let setup = cfg(2.4, 50.0).setup();
    let (tile, halo, max_binding) = with_prepared(dev, &setup, |ctx, _, p| {
        Ok((
            p.recommended_tile_px(ctx),
            p.halo_px(),
            ctx.info().limits.max_storage_buffer_binding_size,
        ))
    })
    .expect("job");
    let side = (tile + 2 * halo) as u64;
    assert!(tile >= 256, "{tile}");
    assert!(
        side * side * 44 <= max_binding || tile == 256,
        "the work buffer of a {tile} px tile does not fit one binding"
    );
}

#[test]
fn a_side_beyond_the_dispatch_limit_is_refused_not_panicked() {
    // A side of more than 16 x max_compute_workgroups_per_dimension pixels (1,048,560 on the default
    // limits) would be a validation error that wgpu raises as a panic.
    let Some(dev) = device() else { return };
    let setup = cfg(3.76, 8.0).setup();
    let (limit, long) = dev
        .gpu
        .run(JobSpec::new(Priority::Export), |ctx, _| {
            let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
            Ok((limit, 16 * limit + 1))
        })
        .expect("job");
    // Only the default limit is cheap to exceed: 12 bytes per pixel of input.
    if long > 4_000_000 {
        eprintln!("skipped: the device allows {limit} workgroups per dimension");
        return;
    }
    let img = RgbImage::new(1, long);
    let e = with_prepared(dev, &setup, move |ctx, _, p| Ok(p.run(ctx, &img).err())).expect("job");
    assert!(matches!(e, Some(GpuError::InvalidInput(_))), "{e:?}");
}
