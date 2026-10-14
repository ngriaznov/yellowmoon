//! RA-09: the adaptive ratio demosaic kernel (`shaders/adaptive_ratio.wgsl`,
//! `kernels::adaptive_ratio`) against its CPU reference `cpu::demosaic::adaptive_ratio` (spec
//! `docs/algorithms/01-adaptive-ratio-demosaic.md`; RA-03 plain path, RA-03b CA path).
//!
//! What is compared, with which bound (each asserted, each printed with `--nocapture`) and what
//! was measured on lavapipe (Mesa 25.2, LLVM 20.1). "Over" counts samples that differ from the CPU
//! by more than 1e-4 (the spec's "fraction of samples above 1e-4"), "fine" those over 2e-5.
//!
//! | Case | Asserted bound | Measured |
//! |---|---|---|
//! | the 16 stable vectors against the oracle's `expected` | the vectors' own tolerance, 1e-5 | 1.4e-6 (`highlights_and_negatives`, values to 3), 3e-7 on the others |
//! | the 2 tie-sensitive vectors (`edge_knight`, `ca_weighting`) | [`TIE_VECTORS`]: at most 20 / 100 samples over, 2e-3 (Metal: [`TIE_VECTORS_METAL`], 80 / 100) | 10 / 58 samples over, 4.2e-4 / 6.1e-4 (the CPU port moves 8 / 26 samples by up to 6.6e-4 / 1.1e-3 under 1 ulp of input noise); Metal 40 / 56, 6.6e-4 / 1.08e-3 |
//! | random frames: 4 phases x 3 noise models x 5 parameter sets x 4 sizes (240 runs, 894 240 samples) | [`FRAME_FRACTION`] over, [`FRAME_FINE_FRACTION`] fine, [`FRAME_BOUND`] | 0 over, 0 fine, worst 7.4e-6 |
//! | the same on the balanced entry (R6's), and against the unbalanced entry | as above | 0 over, worst 1.7e-6 (CPU), 1.4e-6 (unbalanced GPU) |
//! | CA path: random curves, off-centre optical centres, placements anywhere in a 6000 x 4000 frame (32 runs) | as above | 0 over, 0 fine, worst 2.4e-6 |
//! | 2 x 2 to 15 x 15, strips of two, all phases (44 runs) | 5 x [`FRAME_FRACTION`] | 0 over, worst 2.3e-6 |
//! | values from -0.05 to 3e11 (up to the kernel's limit), 4 phases | 5e-2 of the peak | 2.7e-6 of the peak |
//! | flat fields, 4 phases x 3 gain sets x 4 levels | 4e-7 relative plus a floor | 6.0e-7 relative / 4.2e-7 absolute (low noise), 4.8e-7 absolute (high noise, matching) |
//! | 2048 x 1365 through tiles of 1024, balanced | [`LARGE_FRACTION`], [`FRAME_BOUND`], at most 8 x the CPU's own response to 1 ulp + 64 | 70 / 18 of 8 386 560 over (8.3e-6 / 2.1e-6), worst 1.2e-2 / 4.0e-3; the CPU under 1-ulp jitter: 344 / 118 over, worst 1.8e-2 / 1.3e-2 |
//! | tiles of 32 to 100 px on 203 x 157 (all three kinds of frame), plans with exactly 11 / 13 px of context | bit-identical to the whole GPU frame | 0 differing samples |
//! | crops by one photosite (the Bayer phase flips), interior | bit-identical to the uncropped GPU frame | 0 differing samples |
//! | two runs, runs on recycled (poisoned) buffers for every parameter set, every storage mode | bit-identical | 0 differing samples |
//! | one perturbed photosite | the same reach as the CPU port and no further than the halo | 8 px plain, 10 px with matching, equal to the CPU's |
//! | gains or a noise model whose constants leave the f32 range (shot-noise gain 1e50, read noise 1e40, gain 1e30 or 3e38) | `InvalidInput` for the GPU where the CPU is finite, never an `Ok` image of NaN | refused in all eight cases |
//!
//! # Why the GPU is not bit-exact, and what the differences are
//!
//! The CPU runs every per-pixel formula in `f64` and rounds once per stored plane; WGSL has no `f64`,
//! so the kernel runs the same formulas in `f32`, in the same order (module documentation of
//! `kernels::adaptive_ratio`). Where the algorithm only computes a value, that costs a few `1e-7`
//! relative (the vectors: 2.4e-7 absolute at worst, 6e-9 for the dark ones). Where it **decides**
//! (pass-1 direction type, pass-2 argmin over eight directions, step-1 diagonal, step-3 diagonal),
//! two sums that are equal or nearly equal in exact arithmetic can compare the other way round after
//! `f32` rounding, and the pixel takes the other of two equally plausible estimates: a difference of
//! `1e-4` to `1e-3` of the signal at that pixel and its neighbours. Noise-free synthetic content (the
//! edge vectors: exact ties along a 2:1 or 45 degree edge) is full of such ties, and so is real raw
//! data, whose quantised ADC values tie exactly (on crops of real mosaics 4e-6 to 4.5e-4 of the
//! samples flip, up to 1.5 % of white; spec section 12), while the noisy synthetic scenes below have
//! few: the fractions asserted here are bounds for synthetic content. The CPU port is itself that
//! sensitive: `cpu_is_as_sensitive_as_the_gpu_is_different` runs the
//! CPU reference on inputs perturbed by 1 ulp and counts the samples it moves by more than `1e-4`;
//! the GPU/CPU difference counts below are of the same size, which is the evidence that they are
//! decision flips and not a defect of the port. (`over` is the number of samples differing by more
//! than 1e-4: the spec asks for "the fraction of samples above 1e-4" instead of bit equality.)
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); without an adapter these tests skip with a notice, as the other GPU tests do.

mod common;

use std::path::{Path, PathBuf};

use common::{Rng, TestGpu, gpus};
use serde_json::Value;
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::{CaCurves, DemosaicInputs, Method, RadialCurve};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::{Rect, Stage};
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, Placement, RgbImage};
use ym_gpu::device::{GpuError, StorageMode};
use ym_gpu::kernels::adaptive_ratio::{self as gpu_ar, Balance, MAX_BALANCED_DN};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_gpu::tiles::TilePlan;
use ym_noise::model::NoiseModel;

const VECTOR_DIR: &str = "tests/cpu_data/adaptive_ratio_demosaic";

/// The 16 vectors without ties: the GPU against the oracle within the vectors' own tolerance.
const VECTOR_TOLERANCE: f64 = 1e-5;
/// The two vectors that sit on exact ties (`edge_knight`: a 2:1 edge; `ca_weighting`: a 45 degree
/// edge with a position-dependent CA weight): the most samples over 1e-4 and the largest difference
/// allowed. Measured: 10 and 58 samples, 4.2e-4 and 6.1e-4.
const TIE_VECTORS: [(&str, usize, f64); 2] =
    [("edge_knight", 20, 2e-3), ("ca_weighting", 100, 2e-3)];
/// The same two vectors on Metal, which has its own column (`docs/gpu-plan.md` section 12, the run on
/// the Apple-silicon runners): Metal compiles with fast-math, which re-associates and contracts the
/// sums the decisions compare, so a tie in exact arithmetic is decided by different rounding noise
/// than on lavapipe's separately rounded operations (and than on the CPU's `f64`). Measured there:
/// `edge_knight` 40 samples over 1e-4 (lavapipe 10), worst 6.6e-4; `ca_weighting` 56 (lavapipe 58),
/// worst 1.08e-3. `edge_knight` is the vector that failed there (40 against a limit of 20): its Metal
/// limit is twice the first measurement. `ca_weighting` passed within the lavapipe limit (56 against
/// 100), so a column that was looser than lavapipe's would have no evidence behind it: it keeps 100.
/// The lavapipe limits above are untouched and the worst-difference limit is the same for both.
const TIE_VECTORS_METAL: [(&str, usize, f64); 2] =
    [("edge_knight", 80, 2e-3), ("ca_weighting", 100, 2e-3)];

/// The tie-vector limits for `dev`: Metal has its own column, every other backend the lavapipe one.
fn tie_vectors(dev: &TestGpu) -> &'static [(&'static str, usize, f64); 2] {
    if dev.gpu.info().backend == "Metal" {
        &TIE_VECTORS_METAL
    } else {
        &TIE_VECTORS
    }
}
/// Random frames: the fraction of samples that may differ from the CPU by more than 1e-4 (measured:
/// 0 of 894 240 on the sweep, 0 of 178 224 with CA).
const FRAME_FRACTION: f64 = 1e-4;
/// Random frames: the fraction over 2e-5, where a systematic error would show first (measured 0).
const FRAME_FINE_FRACTION: f64 = 1e-4;
/// Random frames: the largest difference allowed anywhere. A flipped decision moves one pixel by the
/// difference of two plausible estimates, here at most a few percent of the signal.
const FRAME_BOUND: f64 = 5e-2;
/// The large frame: the fraction over 1e-4 (measured 8.3e-6 and 2.1e-6).
const LARGE_FRACTION: f64 = 5e-5;

// ---------------------------------------------------------------------------------------------
// Devices and runners
// ---------------------------------------------------------------------------------------------

/// The device with `f32` storage (always available when there is an adapter at all).
fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// One run's inputs, owned so that they can move into the device thread.
#[derive(Clone)]
struct Case {
    mosaic: Mosaic,
    balance: Balance,
    gains: [f32; 3],
    noise: NoiseDn,
    ca: Option<CaCurves>,
    params: Params,
}

impl Case {
    fn new(mosaic: Mosaic, noise: NoiseDn) -> Case {
        Case {
            mosaic,
            balance: Balance::Unbalanced,
            gains: [1.9, 1.0, 1.4],
            noise,
            ca: None,
            params: Params::default(),
        }
    }

    fn inputs(&self) -> DemosaicInputs<'_> {
        DemosaicInputs {
            gains: self.gains,
            noise: self.noise,
            ca: self.ca.as_ref(),
        }
    }

    fn cpu(&self) -> RgbImage {
        self.try_cpu().expect("the CPU reference accepts this case")
    }

    fn try_cpu(&self) -> Result<RgbImage, adaptive_ratio::Error> {
        match self.balance {
            Balance::Unbalanced => {
                adaptive_ratio::demosaic_whole(&self.mosaic, &self.inputs(), &self.params)
            }
            Balance::Balanced => {
                adaptive_ratio::demosaic_balanced_whole(&self.mosaic, &self.inputs(), &self.params)
            }
        }
    }

    fn try_gpu(&self, dev: &TestGpu) -> Result<RgbImage, GpuError> {
        let c = self.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                gpu_ar::demosaic(ctx, &c.mosaic, c.balance, &c.inputs(), &c.params)
            })
    }

    fn gpu(&self, dev: &TestGpu) -> RgbImage {
        self.try_gpu(dev).expect("GPU adaptive ratio demosaic")
    }

    fn gpu_tiled(&self, dev: &TestGpu, tile_px: usize) -> RgbImage {
        let c = self.clone();
        dev.gpu
            .run(JobSpec::new(Priority::Export), move |ctx, token| {
                gpu_ar::demosaic_tiled(
                    ctx,
                    &c.mosaic,
                    c.balance,
                    &c.inputs(),
                    &c.params,
                    tile_px,
                    token,
                )
            })
            .expect("GPU tiled adaptive ratio demosaic")
    }
}

// ---------------------------------------------------------------------------------------------
// Comparing images
// ---------------------------------------------------------------------------------------------

/// How a GPU image differs from a CPU one.
struct Stats {
    max_abs: f64,
    /// (plane, x, y) of the largest difference.
    at: (usize, usize, usize),
    /// Samples that differ by more than `1e-4` (spec 5: the fraction to report).
    over: usize,
    /// Samples that differ by more than `2e-5`: a systematic error shows here before it shows in
    /// `over`.
    fine: usize,
    total: usize,
    /// Largest finite value of the reference.
    peak: f64,
}

impl Stats {
    fn fraction(&self) -> f64 {
        self.over as f64 / self.total as f64
    }
}

fn compare(gpu: &RgbImage, cpu: &RgbImage) -> Stats {
    assert_eq!(
        (gpu.width, gpu.height),
        (cpu.width, cpu.height),
        "image sizes"
    );
    let mut s = Stats {
        max_abs: 0.0,
        at: (0, 0, 0),
        over: 0,
        fine: 0,
        total: gpu.data.len(),
        peak: 0.0,
    };
    for c in 0..3 {
        for (i, (&g, &r)) in gpu.plane(c).iter().zip(cpu.plane(c)).enumerate() {
            assert!(g.is_finite(), "GPU sample (plane {c}, {i}) is {g}");
            let e = (f64::from(g) - f64::from(r)).abs();
            if e > s.max_abs {
                s.max_abs = e;
                s.at = (c, i % gpu.width, i / gpu.width);
            }
            s.over += usize::from(e > 1e-4);
            s.fine += usize::from(e > 2e-5);
            s.peak = s.peak.max(f64::from(r.abs()));
        }
    }
    s
}

fn bits_equal(a: &RgbImage, b: &RgbImage) -> bool {
    a.data.len() == b.data.len()
        && a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------------------------------
// Scenes and models
// ---------------------------------------------------------------------------------------------

/// A seeded natural-looking unbalanced mosaic with a trace of noise, values in about `-0.05..=peak`.
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

/// A dark, noisy scene: the case shadow matching is for (SNR 23.9 dB at 18 % grey).
fn dark_scene(w: usize, h: usize, cfa: Cfa, seed: u64) -> Mosaic {
    let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
    for v in &mut m.data {
        *v *= 0.12;
    }
    synth::add_noise(&mut m.data, 0.012, seed ^ 0xda4c);
    for v in &mut m.data {
        *v = v.max(-0.05);
    }
    m
}

/// The noise model of the spec's table 2.2: `a`, `b`, `sigma_row`, `q` normalised.
fn nz(a: f64, b: f64, sigma_row: f64, q: f64) -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a,
        b,
        delta: [0.0; 4],
        sigma_row,
        q,
    })
}

/// SNR 39.5 dB at 18 % grey: shadow matching off.
fn low_noise() -> NoiseDn {
    nz(2e-5, 1e-8, 0.0, 1.0 / 65535.0)
}

/// SNR 23.9 dB at 18 % grey: shadow matching on.
fn high_noise() -> NoiseDn {
    nz(5e-4, 4e-5, 1e-3, 0.0)
}

/// The scene that goes with a noise model: dark and noisy for the high one.
fn scene_for(noise: NoiseDn, w: usize, h: usize, cfa: Cfa, seed: u64) -> Mosaic {
    if noise == high_noise() {
        dark_scene(w, h, cfa, seed)
    } else {
        scene(w, h, cfa, seed, 1.3)
    }
}

/// The parameter sets of the sweep: the defaults, shadow matching forced on and off, the sharpening
/// off, and a lower SNR target (`detail + strength` = 60 gives s = 8).
fn param_sets() -> [Params; 5] {
    let d = Params::default();
    [
        d,
        Params {
            shadow_match: ShadowMatch::On,
            ..d
        },
        Params {
            shadow_match: ShadowMatch::Off,
            ..d
        },
        Params {
            shadow_sharpening: false,
            ..d
        },
        Params {
            detail: 40.0,
            strength: 20.0,
            ..d
        },
    ]
}

/// Random lateral CA curves with a few knots (negative and wide radii included), shifts up to
/// `max_shift` px, over a `fw x fh` frame whose optical centre is moved off the middle by up to
/// `off` px.
fn random_curves(rng: &mut Rng, fw: usize, fh: usize, max_shift: f64, off: f64) -> CaCurves {
    let hd = ((fw * fw + fh * fh) as f64).sqrt() / 2.0;
    let curve = |rng: &mut Rng| {
        let n = 2 + (rng.next_u64() % 10) as usize;
        let mut r = -rng.range(0.0, 50.0);
        let mut knots = Vec::new();
        for _ in 0..n {
            knots.push((r, rng.range(-max_shift, max_shift)));
            r += rng.range(1.0, 2.0 * hd / n as f64);
        }
        RadialCurve::from_knots(&knots).expect("curve")
    };
    let red = curve(rng);
    let blue = curve(rng);
    CaCurves::new(
        fw as f64 / 2.0 + rng.range(-off, off),
        fh as f64 / 2.0 + rng.range(-off, off),
        red,
        blue,
    )
    .expect("curves")
}

/// Puts the mosaic at `(x0, y0)` of a `fw x fh` frame.
fn place(m: &mut Mosaic, x0: usize, y0: usize, fw: usize, fh: usize) {
    m.place = Placement {
        x0,
        y0,
        frame_width: fw,
        frame_height: fh,
    };
}

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

fn doubles(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}"))
}

fn vector_curve(v: &Value) -> RadialCurve {
    RadialCurve::new(doubles(&v["radius_px"]), doubles(&v["shift_px"])).expect("curve")
}

/// One vector of the package: the case, the oracle's output and its tolerance.
struct Vector {
    name: String,
    case: Case,
    expected: Vec<f32>,
    tolerance: f64,
}

fn load_vectors() -> Vec<Vector> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(VECTOR_DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let json: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("vector file"))
                    .expect("vector json");
            let params = &json["params"];
            let input = &json["input"];
            let (w, h) = (
                input["width"].as_u64().expect("width") as usize,
                input["height"].as_u64().expect("height") as usize,
            );
            let cfa = Cfa::from_name(input["cfa"].as_str().expect("cfa")).expect("known pattern");
            let data: Vec<f32> = doubles(&input["data"]).iter().map(|v| *v as f32).collect();
            let mut mosaic = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
            let ca = if params["ca"].is_null() {
                None
            } else {
                let ca = &params["ca"];
                let knots = &ca["lens_knots"];
                mosaic.place = Placement {
                    x0: ca["origin_x"].as_u64().expect("origin_x") as usize,
                    y0: ca["origin_y"].as_u64().expect("origin_y") as usize,
                    frame_width: knots["frame_width"].as_u64().expect("frame width") as usize,
                    frame_height: knots["frame_height"].as_u64().expect("frame height") as usize,
                };
                Some(
                    CaCurves::new(
                        num(ca, "centre_x"),
                        num(ca, "centre_y"),
                        vector_curve(&ca["red"]),
                        vector_curve(&ca["blue"]),
                    )
                    .expect("curves"),
                )
            };
            let g = doubles(&params["gains"]);
            let n = &params["noise"];
            let shadow_match = match params["shadow_match"].as_str().expect("shadow_match") {
                "auto" => ShadowMatch::Auto,
                "on" => ShadowMatch::On,
                "off" => ShadowMatch::Off,
                other => panic!("unknown shadow_match {other}"),
            };
            Vector {
                name: path
                    .file_stem()
                    .expect("stem")
                    .to_string_lossy()
                    .into_owned(),
                case: Case {
                    mosaic,
                    balance: Balance::Unbalanced,
                    gains: [g[0] as f32, g[1] as f32, g[2] as f32],
                    noise: nz(num(n, "a"), num(n, "b"), num(n, "sigma_row"), num(n, "q")),
                    ca,
                    params: Params {
                        detail: num(params, "detail") as f32,
                        strength: num(params, "strength") as f32,
                        shadow_match,
                        shadow_sharpening: params["shadow_sharpening"].as_bool().expect("bool"),
                    },
                },
                expected: doubles(&json["expected"]["data"])
                    .iter()
                    .map(|v| *v as f32)
                    .collect(),
                tolerance: num(&json, "tolerance"),
            }
        })
        .collect()
}

#[test]
fn every_vector_matches_the_oracle_on_the_gpu() {
    let Some(dev) = f32_device() else { return };
    let vectors = load_vectors();
    assert_eq!(
        vectors.len(),
        18,
        "the vector set (17 plain and the CA vector)"
    );
    let mut worst_stable = 0.0f64;
    for v in &vectors {
        let got = v.case.gpu(dev);
        let cpu = v.case.cpu();
        let to_cpu = compare(&got, &cpu);
        let mut to_oracle = 0.0f64;
        let mut over = 0usize;
        for (a, b) in got.data.iter().zip(&v.expected) {
            let e = (f64::from(*a) - f64::from(*b)).abs();
            to_oracle = to_oracle.max(e);
            over += usize::from(e > 1e-4);
        }
        println!(
            "{:<26} {} {:>2}x{:<2}: max |gpu - oracle| {to_oracle:.3e} ({over} of {} over 1e-4; vector tolerance {:.0e}), max |gpu - cpu| {:.3e}",
            v.name,
            v.case.mosaic.cfa.name(),
            got.width,
            got.height,
            got.data.len(),
            v.tolerance,
            to_cpu.max_abs,
        );
        if let Some(&(_, max_over, max_abs)) = tie_vectors(dev).iter().find(|t| t.0 == v.name) {
            // Exact ties in exact arithmetic: the GPU takes the other of two equal estimates at some
            // pixels along the edge. Bounded, and of the size of the CPU's own response to 1 ulp
            // (`cpu_is_as_sensitive_as_the_gpu_is_different`).
            assert!(
                over <= max_over && to_oracle <= max_abs,
                "{}: {over} samples over 1e-4 (allowed {max_over}), worst {to_oracle:.3e} (allowed {max_abs})",
                v.name
            );
        } else {
            assert!(
                to_oracle <= VECTOR_TOLERANCE.min(v.tolerance),
                "{}: the GPU differs from the oracle by {to_oracle}, tolerance {}",
                v.name,
                v.tolerance
            );
            worst_stable = worst_stable.max(to_oracle);
        }
    }
    println!("the 16 stable vectors: worst |gpu - oracle| {worst_stable:.3e}");
}

// ---------------------------------------------------------------------------------------------
// Random frames: all phases, noise models, parameters, CA
// ---------------------------------------------------------------------------------------------

/// Totals of a sweep.
#[derive(Default)]
struct Totals {
    runs: usize,
    samples: usize,
    over: usize,
    fine: usize,
    worst: f64,
    worst_label: String,
}

impl Totals {
    fn add(&mut self, label: String, s: &Stats) {
        self.runs += 1;
        self.samples += s.total;
        self.over += s.over;
        self.fine += s.fine;
        if s.max_abs > self.worst {
            self.worst = s.max_abs;
            self.worst_label = label;
        }
    }

    fn fraction(&self) -> f64 {
        self.over as f64 / self.samples as f64
    }

    fn fine_fraction(&self) -> f64 {
        self.fine as f64 / self.samples as f64
    }

    fn report(&self, what: &str) {
        println!(
            "{what}: {} runs, {} of {} samples over 1e-4 ({:.2e}), {} over 2e-5, worst |gpu - cpu| {:.3e} ({})",
            self.runs,
            self.over,
            self.samples,
            self.fraction(),
            self.fine,
            self.worst,
            self.worst_label
        );
    }
}

#[test]
fn random_frames_match_in_all_phases_noise_models_and_parameters() {
    let Some(dev) = f32_device() else { return };
    let mut totals = Totals::default();
    let mut seed = 100;
    for cfa in Cfa::ALL {
        // Odd and even sizes, narrower and wider than a workgroup (16) and than the halo (13).
        for (w, h) in [(67usize, 41usize), (40, 40), (31, 18), (9, 7)] {
            for (n, noise) in [
                ("zero", NoiseDn::ZERO),
                ("low", low_noise()),
                ("high", high_noise()),
            ] {
                for (k, params) in param_sets().into_iter().enumerate() {
                    seed += 1;
                    let mut c = Case::new(scene_for(noise, w, h, cfa, seed), noise);
                    c.params = params;
                    let s = compare(&c.gpu(dev), &c.cpu());
                    let label = format!("{} {w}x{h} noise {n} params {k}", cfa.name());
                    assert!(
                        s.max_abs <= FRAME_BOUND,
                        "{label}: {} at {:?}",
                        s.max_abs,
                        s.at
                    );
                    totals.add(label, &s);
                }
            }
        }
    }
    totals.report("random frames");
    assert!(totals.fraction() <= FRAME_FRACTION, "{}", totals.fraction());
    assert!(
        totals.fine_fraction() <= FRAME_FINE_FRACTION,
        "{}",
        totals.fine_fraction()
    );
}

#[test]
fn the_balanced_entry_matches_the_cpu_and_the_unbalanced_entry() {
    // R6's entry (spec 1.3): the balanced mosaic is divided by the gains of its colour first. The GPU
    // result for the balanced mosaic m * g0 agrees with the GPU result for m (measured 3e-8 on the
    // CPU) and with the CPU's `demosaic_balanced`.
    let Some(dev) = f32_device() else { return };
    let mut totals = Totals::default();
    let mut against_unbalanced = Totals::default();
    for (i, cfa) in Cfa::ALL.into_iter().enumerate() {
        for (n, noise) in [("low", low_noise()), ("high", high_noise())] {
            let m = scene_for(noise, 53, 38, cfa, 400 + i as u64);
            let mut plain = Case::new(m.clone(), noise);
            plain.gains = [2.1, 1.0, 1.6];
            let mut balanced = plain.clone();
            balanced.balance = Balance::Balanced;
            for y in 0..m.height {
                for x in 0..m.width {
                    let g = plain.gains[m.cfa.color_at(x, y).plane()];
                    balanced.mosaic.data[y * m.width + x] = m.data[y * m.width + x] * g;
                }
            }
            let from_balanced = balanced.gpu(dev);
            let s = compare(&from_balanced, &balanced.cpu());
            let label = format!("balanced {} noise {n}", cfa.name());
            assert!(s.max_abs <= FRAME_BOUND, "{label}: {}", s.max_abs);
            totals.add(label.clone(), &s);
            let e = compare(&from_balanced, &plain.gpu(dev));
            assert!(
                e.max_abs <= FRAME_BOUND,
                "{label} against unbalanced {}",
                e.max_abs
            );
            against_unbalanced.add(label, &e);
        }
    }
    totals.report("balanced entry against the CPU");
    against_unbalanced.report("balanced entry against the unbalanced entry (GPU)");
    assert!(totals.fraction() <= FRAME_FRACTION, "{}", totals.fraction());
    assert!(
        against_unbalanced.fraction() <= FRAME_FRACTION,
        "{}",
        against_unbalanced.fraction()
    );
}

#[test]
fn the_ca_path_matches_the_cpu_with_random_curves_and_placements() {
    let Some(dev) = f32_device() else { return };
    let mut totals = Totals::default();
    let mut rng = Rng(77);
    let mut seed = 2000;
    for cfa in Cfa::ALL {
        for (w, h) in [(60usize, 44usize), (37, 29)] {
            for (n, noise) in [("low", low_noise()), ("high", high_noise())] {
                for max_shift in [0.6, 3.0] {
                    seed += 1;
                    // A tile somewhere in a 6000 x 4000 frame, mostly away from the centre.
                    let (fw, fh) = (6000usize, 4000usize);
                    let x0 = (rng.next_u64() % (fw - w) as u64) as usize;
                    let y0 = (rng.next_u64() % (fh - h) as u64) as usize;
                    let mut m = scene_for(noise, w, h, cfa, seed);
                    place(&mut m, x0, y0, fw, fh);
                    let mut c = Case::new(m, noise);
                    c.ca = Some(random_curves(&mut rng, fw, fh, max_shift, 300.0));
                    let s = compare(&c.gpu(dev), &c.cpu());
                    let label = format!(
                        "CA {} {w}x{h} at ({x0},{y0}) noise {n} shift {max_shift}",
                        cfa.name()
                    );
                    assert!(s.max_abs <= FRAME_BOUND, "{label}: {}", s.max_abs);
                    totals.add(label, &s);
                }
            }
        }
    }
    totals.report("CA path");
    assert!(totals.fraction() <= FRAME_FRACTION, "{}", totals.fraction());
    assert!(
        totals.fine_fraction() <= FRAME_FINE_FRACTION,
        "{}",
        totals.fine_fraction()
    );
}

#[test]
fn a_ca_curve_with_a_constant_misregistration_does_not_depend_on_the_placement() {
    // Spec 3.6: a misregistration that is the same everywhere gives the same pixels wherever the tile
    // sits in the frame, and a growing one does not.
    let Some(dev) = f32_device() else { return };
    let curve = |a: f64, b: f64| RadialCurve::from_knots(&[(0.0, a), (4000.0, b)]).expect("curve");
    let m = scene(48, 36, Cfa::RGGB, 9, 1.3);
    let run = |curves: CaCurves, x0: usize, y0: usize| {
        let mut mm = m.clone();
        place(&mut mm, x0, y0, 6000, 4000);
        let mut c = Case::new(mm, low_noise());
        c.ca = Some(curves);
        c.gpu(dev)
    };
    let flat = |x| CaCurves::new(3000.0, 2000.0, curve(0.0, 0.0), curve(x, x)).expect("flat");
    let a = run(flat(0.8), 100, 100);
    let b = run(flat(0.8), 5000, 3000);
    assert!(
        bits_equal(&a, &b),
        "a constant misregistration depends on the placement"
    );
    // |R - B| grows from 0 at the centre to 0.8 px at r = 4000: 0.22 px at (1100, 600) from the
    // centre and 0.03 px at (100, 100) from it, so the weights differ between the two tiles.
    let ramp = || CaCurves::new(3000.0, 2000.0, curve(0.0, 0.0), curve(0.0, 0.8)).expect("ramp");
    let c = run(ramp(), 3900, 2500);
    let d = run(ramp(), 2900, 1900);
    assert!(
        !bits_equal(&c, &d),
        "a growing misregistration ignores the placement"
    );
}

// ---------------------------------------------------------------------------------------------
// Tiling, the large frame
// ---------------------------------------------------------------------------------------------

#[test]
fn tiles_with_halo_equal_the_whole_frame_at_tile_and_halo_boundaries() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203usize, 157usize);
    let mut rng = Rng(5);
    for (cfa, seed, noise, label) in [
        (Cfa::RGGB, 31u64, low_noise(), "RGGB low noise (halo 11)"),
        (
            Cfa::GBRG,
            32,
            high_noise(),
            "GBRG high noise, balanced (shadow matching, halo 13)",
        ),
        (Cfa::BGGR, 33, low_noise(), "BGGR CA path"),
    ] {
        let mut m = scene_for(noise, w, h, cfa, seed);
        let mut c = if label.contains("CA") {
            place(&mut m, 2301, 1777, 6000, 4000);
            let mut c = Case::new(m, noise);
            c.ca = Some(random_curves(&mut rng, 6000, 4000, 2.0, 200.0));
            c
        } else {
            Case::new(m, noise)
        };
        if label.contains("balanced") {
            c.balance = Balance::Balanced;
        }
        let whole = c.gpu(dev);
        let reference = c.cpu();
        let d_whole = compare(&whole, &reference);
        assert!(d_whole.max_abs <= FRAME_BOUND);
        for tile_px in [32usize, 50, 64, 100] {
            let tiled = c.gpu_tiled(dev, tile_px);
            let to_cpu = compare(&tiled, &reference);
            let identical = bits_equal(&tiled, &whole);
            println!(
                "{label} {w}x{h}, tiles of {tile_px}: tiled against whole GPU {}, against CPU {:.3e}",
                if identical {
                    "bit-identical".to_owned()
                } else {
                    format!("{:.3e}", compare(&tiled, &whole).max_abs)
                },
                to_cpu.max_abs
            );
            // The same operations in the same order on the same inputs: identical bits.
            assert!(
                identical,
                "{label}: tile {tile_px} is not bit-identical to the whole frame"
            );
        }
    }
}

#[test]
fn tiling_is_bit_identical_on_tiny_and_odd_shapes_and_degenerate_tile_sizes() {
    // Strips two pixels wide or high, frames smaller than the halo and tile sizes of 0 and 1 (which
    // the plan rounds up to 2): no panic, no seam, the same bits as the whole frame.
    let Some(dev) = f32_device() else { return };
    let mut seed = 8000;
    for (w, h) in [(2usize, 2usize), (2, 9), (7, 3), (3, 33), (17, 29)] {
        for cfa in [Cfa::RGGB, Cfa::GBRG] {
            seed += 1;
            let c = Case::new(scene(w, h, cfa, seed, 1.3), low_noise());
            let whole = c.gpu(dev);
            for tile_px in [0usize, 1, 3, 8] {
                let tiled = c.gpu_tiled(dev, tile_px);
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
fn a_plan_without_context_or_for_another_size_is_refused_and_a_cancel_stops_the_run() {
    let Some(dev) = f32_device() else { return };
    let c = Case::new(scene(64, 48, Cfa::RGGB, 3, 1.0), low_noise());
    let no_context = TilePlan::new(64, 48, 32, &[]);
    let wrong_size = TilePlan::new(65, 48, 32, &[Stage::Demosaic(Method::AdaptiveRatio)]);
    let cc = c.clone();
    let errs = dev
        .gpu
        .run(JobSpec::new(Priority::Export), move |ctx, token| {
            let a = gpu_ar::demosaic_planned(
                ctx,
                &cc.mosaic,
                cc.balance,
                &cc.inputs(),
                &cc.params,
                &no_context,
                token,
            );
            let b = gpu_ar::demosaic_planned(
                ctx,
                &cc.mosaic,
                cc.balance,
                &cc.inputs(),
                &cc.params,
                &wrong_size,
                token,
            );
            let cancelled = CancelToken::new();
            cancelled.cancel();
            let d = gpu_ar::demosaic_tiled(
                ctx,
                &cc.mosaic,
                cc.balance,
                &cc.inputs(),
                &cc.params,
                32,
                &cancelled,
            );
            Ok((a.err(), b.err(), d.err()))
        })
        .expect("job");
    for (what, e) in [("no context", errs.0), ("wrong size", errs.1)] {
        assert!(
            matches!(e, Some(GpuError::InvalidInput(_))),
            "{what}: {e:?}"
        );
    }
    assert!(matches!(errs.2, Some(GpuError::Cancelled)), "{:?}", errs.2);
}

#[test]
fn plans_with_exactly_the_declared_context_work_and_less_is_refused() {
    // The plain configuration needs 11 px of context (the spec's bound; the exact reach is 10), the
    // one with shadow matching 13 (exact 12): a plan with that much gives the whole-frame bits, a
    // plan with one pixel less is refused.
    let Some(dev) = f32_device() else { return };
    for (noise, halo) in [(low_noise(), 11usize), (high_noise(), 13)] {
        let c = Case::new(scene_for(noise, 120, 97, Cfa::GRBG, 14), noise);
        let whole = c.gpu(dev);
        let run = |plan_halo: usize| {
            let cc = c.clone();
            let plan = TilePlan::with_halo(120, 97, 40, plan_halo);
            dev.gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    gpu_ar::demosaic_planned(
                        ctx,
                        &cc.mosaic,
                        cc.balance,
                        &cc.inputs(),
                        &cc.params,
                        &plan,
                        token,
                    )
                })
        };
        let tiled = run(halo).expect("a plan with the declared context");
        assert!(bits_equal(&tiled, &whole), "halo {halo}");
        let e = run(halo - 1);
        assert!(
            matches!(e, Err(GpuError::InvalidInput(_))),
            "halo {}: {e:?}",
            halo - 1
        );
    }
}

#[test]
fn a_large_frame_through_the_tiling_path_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (2048usize, 1365usize);
    let mut totals = Totals::default();
    for (label, noise) in [
        ("low noise", low_noise()),
        ("high noise (matching)", high_noise()),
    ] {
        let mut c = Case::new(scene_for(noise, w, h, Cfa::RGGB, 2048), noise);
        // The balanced mosaic of the chain: R4 multiplied every photosite by its gain.
        c.balance = Balance::Balanced;
        for y in 0..h {
            for x in 0..w {
                c.mosaic.data[y * w + x] *= c.gains[c.mosaic.cfa.color_at(x, y).plane()];
            }
        }
        let started = std::time::Instant::now();
        let tiled = c.gpu_tiled(dev, 1024);
        let gpu_time = started.elapsed();
        let started = std::time::Instant::now();
        let reference = c.cpu();
        let cpu_time = started.elapsed();
        let s = compare(&tiled, &reference);
        println!(
            "{w}x{h} {label} in tiles of 1024 ({} tiles): max |gpu - cpu| {:.3e} at plane {} ({}, {}), {} of {} samples over 1e-4 ({:.2e}); GPU {gpu_time:.2?} on {}, CPU {cpu_time:.2?}",
            TilePlan::new(w, h, 1024, &[Stage::Demosaic(Method::AdaptiveRatio)])
                .tiles()
                .len(),
            s.max_abs,
            s.at.0,
            s.at.1,
            s.at.2,
            s.over,
            s.total,
            s.fraction(),
            dev.gpu.info().name
        );
        assert!(s.max_abs <= FRAME_BOUND, "{}", s.max_abs);
        assert!(s.fraction() <= LARGE_FRACTION, "{}", s.fraction());
        // The CPU's own response to 1 ulp of input noise, for scale (see the module documentation).
        let cpu_jitter = compare(&jitter(&c, 7, 1.2e-7).cpu(), &reference);
        println!(
            "  the CPU reference under 1-ulp jitter: {} of {} samples over 1e-4, max {:.3e}",
            cpu_jitter.over, cpu_jitter.total, cpu_jitter.max_abs
        );
        assert!(
            s.over <= 8 * cpu_jitter.over + 64,
            "{label}: the GPU differs at {} samples, the CPU moves at {} under 1 ulp",
            s.over,
            cpu_jitter.over
        );
        totals.add(label.to_owned(), &s);
    }
    totals.report("large frames");
}

// ---------------------------------------------------------------------------------------------
// Edge sizes and hostile values
// ---------------------------------------------------------------------------------------------

#[test]
fn tiny_odd_and_strip_sizes_match_the_cpu_and_sizes_below_two_are_refused() {
    let Some(dev) = f32_device() else { return };
    let mut totals = Totals::default();
    let mut seed = 5000;
    for cfa in Cfa::ALL {
        for (w, h) in [
            (2usize, 2usize),
            (2, 3),
            (3, 2),
            (2, 17),
            (17, 2),
            (3, 3),
            (5, 3),
            (4, 5),
            (7, 13),
            (13, 7),
            (15, 15),
            (1, 1),
            (1, 2),
            (2, 1),
            (1, 40),
            (40, 1),
        ] {
            seed += 1;
            let noise = if seed % 2 == 0 {
                high_noise()
            } else {
                low_noise()
            };
            let c = Case::new(scene(w, h, cfa, seed, 1.0), noise);
            if w < 2 || h < 2 {
                // The CPU reference says `BadSize`; the GPU refuses the same sizes.
                assert!(c.try_cpu().is_err(), "{w}x{h}");
                let e = c.try_gpu(dev);
                assert!(
                    matches!(e, Err(GpuError::InvalidInput(_))),
                    "{w}x{h}: {e:?}"
                );
                continue;
            }
            let s = compare(&c.gpu(dev), &c.cpu());
            let label = format!("{} {w}x{h}", cfa.name());
            assert!(s.max_abs <= FRAME_BOUND, "{label}: {}", s.max_abs);
            totals.add(label, &s);
        }
    }
    totals.report("tiny sizes (2 x 2 to 15 x 15, strips)");
    assert!(
        totals.fraction() <= 5.0 * FRAME_FRACTION,
        "{}",
        totals.fraction()
    );
}

/// White noise: every sample uniform in [0, 1], no structure at all, so no region is flat and no
/// comparison of the kernel is decided by anything but the data. From 3 x 3 up (and in strips of two)
/// the device agrees with the CPU to rounding; a 2 x 2 mosaic is the one degenerate size, where the
/// mirror makes the left and the right neighbour of every site the same sample, and so the top and the
/// bottom one: every direction test is an exact tie, which `f32` and `f64` break differently (in a
/// separate run, 20 of 60 random 2 x 2 mosaics differed by more than 1e-4, up to 1.7 of white; the
/// result stays finite).
/// The chain never hands the kernel a mosaic that small except as a whole tiny photograph.
/// Measured: 480 mosaics from 3 x 3 up, none over 1e-4, worst 5.7e-6; the bound is twice that.
#[test]
fn white_noise_mosaics_match_the_cpu_from_three_by_three_up_and_two_by_two_stays_finite() {
    let Some(dev) = f32_device() else { return };
    let mut rng = Rng(7701);
    let mut worst = 0.0f64;
    let mut runs = 0usize;
    for cfa in Cfa::ALL {
        for (w, h) in [
            (3usize, 3usize),
            (4, 4),
            (5, 5),
            (8, 8),
            (12, 12),
            (16, 16),
            (24, 24),
            (2, 8),
            (3, 9),
            (8, 2),
        ] {
            for k in 0..12 {
                let data: Vec<f32> = (0..w * h).map(|_| rng.range(0.0, 1.0) as f32).collect();
                let mosaic = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
                let noise = if k % 2 == 0 {
                    high_noise()
                } else {
                    low_noise()
                };
                let c = Case::new(mosaic, noise);
                let s = compare(&c.gpu(dev), &c.cpu());
                worst = worst.max(s.max_abs);
                runs += 1;
                assert!(
                    s.max_abs <= 1.2e-5,
                    "{} {w}x{h} #{k}: {:e} at {:?}",
                    cfa.name(),
                    s.max_abs,
                    s.at
                );
            }
        }
    }
    println!("white noise from 3 x 3 up: {runs} mosaics, worst |gpu - cpu| {worst:.2e}");
    // 2 x 2: no panic, a finite picture of the size of the mosaic.
    for cfa in Cfa::ALL {
        for k in 0..8 {
            let data: Vec<f32> = (0..4).map(|_| rng.range(0.0, 1.0) as f32).collect();
            let mosaic = Mosaic::from_data(2, 2, cfa, data).expect("mosaic");
            let noise = if k % 2 == 0 {
                high_noise()
            } else {
                low_noise()
            };
            let out = Case::new(mosaic, noise).gpu(dev);
            assert_eq!((out.width, out.height), (2, 2));
            assert!(
                out.data.iter().all(|v| v.is_finite()),
                "{} 2x2 #{k} is not finite",
                cfa.name()
            );
        }
    }
}

#[test]
fn nan_infinity_and_overflowing_values_are_refused_without_a_panic() {
    let Some(dev) = f32_device() else { return };
    let base = Case::new(scene(30, 24, Cfa::RGGB, 12, 1.0), low_noise());
    let cpu_ok = base.cpu();
    for (what, value) in [
        ("NaN", f32::NAN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("1e38", 1.0e38),
        ("-1e38", -1.0e38),
        ("beyond the kernel's limit", 2.0e12),
    ] {
        let mut c = base.clone();
        c.mosaic.data[7 * 30 + 11] = value;
        // The CPU refuses NaN, infinity and 1e38; it accepts 2e12 and -1e38 (the negative part is
        // clamped and restored at the exit), which the GPU refuses (the limit holds for both signs).
        if !what.starts_with("beyond") && what != "-1e38" {
            assert!(c.try_cpu().is_err(), "{what}: the CPU accepts it");
        }
        let e = c.try_gpu(dev);
        assert!(matches!(e, Err(GpuError::InvalidInput(_))), "{what}: {e:?}");
    }
    // Bad gains, noise and parameters are refused as on the CPU.
    for gains in [
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, f32::NAN],
        [f32::INFINITY, 1.0, 1.0],
    ] {
        let mut c = base.clone();
        c.gains = gains;
        assert!(
            matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))),
            "{gains:?}"
        );
        c.balance = Balance::Balanced;
        assert!(
            matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))),
            "{gains:?}"
        );
    }
    for noise in [
        NoiseDn::new(-1.0, 0.0),
        NoiseDn::new(0.0, f64::NAN),
        NoiseDn::new(f64::INFINITY, 0.0),
    ] {
        let mut c = base.clone();
        c.noise = noise;
        assert!(
            matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))),
            "{noise:?}"
        );
    }
    let mut c = base.clone();
    c.params.detail = f32::NAN;
    assert!(matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))));
    // A CA centre that is not finite, and knots that do not fit f32.
    let curve = RadialCurve::from_knots(&[(0.0, 0.0), (100.0, 1.0)]).expect("curve");
    let mut ca = CaCurves::new(10.0, 10.0, curve.clone(), curve).expect("curves");
    ca.centre_x = f64::NAN;
    let mut c = base.clone();
    c.ca = Some(ca);
    assert!(c.try_cpu().is_err());
    assert!(matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))));
    let huge = RadialCurve::from_knots(&[(0.0, 0.0), (1.0e300, 1.0e300)]).expect("curve");
    let mut c = base.clone();
    c.ca = Some(CaCurves::new(10.0, 10.0, huge.clone(), huge).expect("curves"));
    assert!(matches!(c.try_gpu(dev), Err(GpuError::InvalidInput(_))));
    // The device still works afterwards, with the same result as before.
    let again = base.gpu(dev);
    assert!(compare(&again, &cpu_ok).max_abs <= FRAME_BOUND);
}

#[test]
fn constants_beyond_the_f32_range_are_refused_not_turned_into_a_nan_image() {
    // The CPU reference runs in f64 and accepts these: a shot-noise gain of 1e50 DN per DN, a read
    // noise of 1e40 DN, a gain of 1e30 against a read noise of a few DN (all valid by
    // `NoiseDn::is_valid` and the gain check). Rounded to f32 the per-frame constants (g g_c,
    // (r g_c)^2, rho, 4 g_c, r) are infinities, and the kernel used to return `Ok` with every sample
    // NaN. It refuses them once per frame instead, like a sample beyond the kernel's limit.
    let Some(dev) = f32_device() else { return };
    let zero = Mosaic::new(30, 24, Cfa::RGGB);
    let dark = scene(30, 24, Cfa::RGGB, 12, 0.5);
    for (what, mosaic, gains, noise) in [
        (
            "shot-noise gain 1e50",
            &dark,
            [1.9, 1.0, 1.4],
            NoiseDn::new(1.0e50, 1.0),
        ),
        (
            "read noise 1e40",
            &dark,
            [1.9, 1.0, 1.4],
            NoiseDn::new(1.0, 1.0e40),
        ),
        (
            "gain 1e30 against read noise 6",
            &zero,
            [1.0e30, 1.0, 1.0],
            NoiseDn::new(1.0, 6.0),
        ),
        (
            "gain 3e38",
            &zero,
            [1.0, 1.0, 3.0e38],
            NoiseDn::new(0.0, 0.0),
        ),
    ] {
        for params in [
            Params::default(),
            Params {
                shadow_match: ShadowMatch::On,
                ..Params::default()
            },
        ] {
            let mut c = Case::new(mosaic.clone(), noise);
            c.gains = gains;
            c.params = params;
            let cpu = c
                .try_cpu()
                .unwrap_or_else(|e| panic!("{what}: the CPU refuses: {e}"));
            assert!(
                cpu.data.iter().all(|v| v.is_finite()),
                "{what}: the CPU is not finite"
            );
            let e = c.try_gpu(dev);
            assert!(
                matches!(e, Err(GpuError::InvalidInput(_))),
                "{what}, matching {:?}: {e:?}",
                params.shadow_match
            );
        }
    }
    // Large constants that do fit f32 are not refused and still match the CPU: gains of 1e4 and 1e-4
    // around green and a read noise ten times the high model's, on a scene of a few thousand DN (a
    // signal below one DN is below the resolution of the offset-ratio forms at `k` = 65535 DN in
    // f32, which is not what this checks).
    let mut c = Case::new(
        scene(30, 24, Cfa::RGGB, 12, 0.05),
        NoiseDn::new(30.0, 4.0e3),
    );
    c.gains = [1.0e4, 1.0, 1.0e-4];
    c.params.shadow_match = ShadowMatch::On;
    let s = compare(&c.gpu(dev), &c.cpu());
    assert!(
        s.peak > 100.0 && s.max_abs <= FRAME_BOUND * s.peak,
        "{} of the peak {}",
        s.max_abs,
        s.peak
    );
}

#[test]
fn negatives_and_very_large_values_match_the_cpu() {
    let Some(dev) = f32_device() else { return };
    // Values from -0.05 (R1's clamp) up to far above white: R5's output, and the hostile scales of
    // the spec's robustness test, up to the kernel's limit. The error is relative to the peak.
    let mut worst = 0.0f64;
    let mut totals = Totals::default();
    for (i, cfa) in Cfa::ALL.into_iter().enumerate() {
        for scale in [1.0f32, 3.0, 1.0e3, 1.0e6, 1.0e9, 1.0e11] {
            let mut rng = Rng(900 + i as u64);
            let mut m = scene(61, 43, cfa, 900 + i as u64, 1.0);
            for v in &mut m.data {
                let r = rng.unit();
                if r < 0.03 {
                    *v = rng.range(-0.05, 0.0) as f32; // negatives down to the clamp
                } else if r < 0.06 {
                    *v *= 3.0; // highlights
                }
                if *v > 0.0 {
                    *v *= scale;
                }
            }
            assert!(
                f64::from(scale) * 65535.0 * 3.0 * 1.9 < MAX_BALANCED_DN,
                "the sweep stays inside the kernel's limit"
            );
            for noise in [NoiseDn::ZERO, low_noise()] {
                let c = Case::new(m.clone(), noise);
                let s = compare(&c.gpu(dev), &c.cpu());
                let rel = s.max_abs / s.peak;
                worst = worst.max(rel);
                assert!(
                    rel <= 5e-2,
                    "{} scale {scale:e} noise r={}: {rel:.3e} of the peak {:.3e}",
                    cfa.name(),
                    noise.r,
                    s.peak
                );
                totals.add(format!("{} scale {scale:e}", cfa.name()), &s);
            }
        }
    }
    println!("hostile scales, worst error relative to the peak: {worst:.3e}");
}

#[test]
fn a_negative_sample_keeps_its_gain_times_value_in_its_own_channel() {
    // Spec 6.2: against the same sample at 0, the own channel differs by exactly gain * value
    // (to f32), and nothing else in the image changes (the residual is never interpolated).
    let Some(dev) = f32_device() else { return };
    for cfa in Cfa::ALL {
        let base = scene(33, 27, cfa, 21, 1.0);
        let (x, y) = (14usize, 11usize);
        let mut zeroed = base.clone();
        zeroed.data[y * 33 + x] = 0.0;
        let mut negative = base.clone();
        negative.data[y * 33 + x] = -0.037;
        let mut a = Case::new(zeroed, low_noise());
        a.gains = [2.0, 1.0, 1.5];
        let mut b = a.clone();
        b.mosaic = negative;
        let (ga, gb) = (a.gpu(dev), b.gpu(dev));
        let own = base.cfa.color_at(x, y).plane();
        for c in 0..3 {
            let d = gb.pixel(x, y)[c] - ga.pixel(x, y)[c];
            let want = if c == own { a.gains[own] * -0.037 } else { 0.0 };
            assert!(
                (d - want).abs() <= 2e-7,
                "{} plane {c}: differs by {d}, want {want}",
                cfa.name()
            );
        }
        for c in 0..3 {
            for j in 0..33 * 27 {
                if (j % 33, j / 33) != (x, y) {
                    assert_eq!(
                        ga.plane(c)[j].to_bits(),
                        gb.plane(c)[j].to_bits(),
                        "{} plane {c} sample {j}",
                        cfa.name()
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------------------------

#[test]
fn a_side_beyond_the_dispatch_limit_is_refused_not_panicked() {
    // A 2 x N column with N beyond 16 x max_compute_workgroups_per_dimension would need more
    // workgroups on y than the device allows, which wgpu reports as a validation panic. The host
    // refuses it with `InvalidInput` before touching the device.
    let Some(dev) = gpus().first() else { return };
    let limit = dev.gpu.info().limits.max_compute_workgroups_per_dimension as usize;
    if limit > 100_000 {
        eprintln!("skipped: the device allows {limit} workgroups per dimension");
        return;
    }
    let h = limit * 16 + 1;
    let mut c = Case::new(Mosaic::new(2, h, Cfa::RGGB), low_noise());
    c.mosaic.data.fill(0.2);
    let e = c.try_gpu(dev);
    assert!(matches!(e, Err(GpuError::InvalidInput(_))), "{e:?}");
    // The device still works afterwards.
    let ok = Case::new(scene(11, 9, Cfa::RGGB, 1, 1.0), low_noise());
    assert!(compare(&ok.gpu(dev), &ok.cpu()).max_abs <= FRAME_BOUND);
}

#[test]
fn a_frame_too_big_for_one_binding_is_refused_and_a_tile_of_it_is_not() {
    // The planes of one run live in one storage binding (128 MiB by default). A frame whose padded
    // planes do not fit is refused with `BufferTooLarge` instead of failing in the driver; the tiled
    // entry point is for such frames.
    let Some(dev) = f32_device() else { return };
    let limit = dev.gpu.info().limits.max_storage_buffer_binding_size as usize;
    if limit > (1 << 30) {
        eprintln!("skipped: the device allows a binding of {limit} bytes");
        return;
    }
    let w = 1000usize;
    let h = limit / (20 * w) + 200; // 5 planes of 4 bytes: more than the binding allows
    let mut c = Case::new(Mosaic::new(w, h, Cfa::RGGB), low_noise());
    c.mosaic.data.fill(0.1);
    let e = c.try_gpu(dev);
    assert!(matches!(e, Err(GpuError::BufferTooLarge { .. })), "{e:?}");
    // A tile of the same frame is fine and gives what the CPU gives.
    let sub = c.mosaic.extract(Rect::new(0, 0, 200, 150));
    let ok = Case::new(sub, low_noise());
    assert!(compare(&ok.gpu(dev), &ok.cpu()).max_abs <= 1e-5);
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn two_runs_and_runs_on_recycled_buffers_are_bit_identical_in_every_storage_mode() {
    let mut reference: Option<RgbImage> = None;
    for dev in gpus() {
        let mut m = dark_scene(97, 61, Cfa::GRBG, 1234);
        place(&mut m, 1200, 800, 6000, 4000);
        let mut c = Case::new(m, high_noise());
        c.ca = Some(random_curves(&mut Rng(3), 6000, 4000, 2.0, 100.0));
        let a = c.gpu(dev);
        let b = c.gpu(dev);
        assert!(bits_equal(&a, &b), "storage {}", dev.mode.name());
        // Poison the pool: run other frames of the same and larger size classes so that the next run
        // gets recycled buffers full of someone else's data. A pass that read a value no earlier
        // pass wrote would show here, for every combination of the optional passes (the
        // sharpening, the matching, the CA path).
        let poison = |seed: u64| {
            let mut other = Case::new(scene(97, 61, Cfa::RGGB, 50 + seed, 3.0), low_noise());
            other.params = param_sets()[(1 + seed as usize) % 5];
            let _ = other.gpu(dev);
            let big = Case::new(dark_scene(130, 90, Cfa::BGGR, 60 + seed), high_noise());
            let _ = big.gpu(dev);
        };
        for seed in 0..3u64 {
            poison(seed);
        }
        let after = c.gpu(dev);
        assert!(
            bits_equal(&a, &after),
            "recycled buffers, storage {}",
            dev.mode.name()
        );
        for (k, params) in param_sets().into_iter().enumerate() {
            for (noise, ca) in [(low_noise(), false), (high_noise(), true)] {
                let mut m = scene_for(noise, 70, 52, Cfa::GBRG, 700 + k as u64);
                place(&mut m, 3000, 1500, 6000, 4000);
                let mut c = Case::new(m, noise);
                c.params = params;
                if ca {
                    c.ca = Some(random_curves(&mut Rng(9), 6000, 4000, 1.5, 100.0));
                }
                let first = c.gpu(dev);
                poison(10 + k as u64);
                assert!(
                    bits_equal(&first, &c.gpu(dev)),
                    "params {k}, CA {ca}, storage {}: a pass read a value that no pass wrote",
                    dev.mode.name()
                );
            }
        }
        let ta = c.gpu_tiled(dev, 40);
        let tb = c.gpu_tiled(dev, 40);
        assert!(bits_equal(&ta, &tb), "tiled, storage {}", dev.mode.name());
        assert!(
            bits_equal(&ta, &a),
            "tiled against whole, storage {}",
            dev.mode.name()
        );
        // Every storage mode runs the same f32 shader: the same bits across devices.
        match &reference {
            None => reference = Some(a),
            Some(r) => assert!(
                bits_equal(r, &a),
                "storage {} differs from the first device",
                dev.mode.name()
            ),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Properties of spec section 6.2, on the GPU
// ---------------------------------------------------------------------------------------------

#[test]
fn flat_fields_come_back_as_gain_times_value_in_all_four_phases() {
    // Spec 6.2 item 1: constant colour in, gain * value out. The reference measures 6.4e-8 relative
    // in f64. In f32 the offset-ratio forms `(x + k) r - k` of passes 1 and 3 lose the low bits of
    // `x + k`, where `k` is 555 DN for the low noise model and 17202 DN for the high one: an absolute
    // error of about `k * 6e-8`, whatever the level. Measured: 6e-7 relative (4e-7 absolute, at white
    // times 6.5) for the low model, and for the high one 4.8e-7 absolute (1.4e-5 relative, at level
    // 0.003), which is 0.03 DN of a 16-bit scale.
    let Some(dev) = f32_device() else { return };
    // (noise, parameters, allowed absolute error in units of white, label)
    let configs = [
        (low_noise(), Params::default(), 1e-8, "low noise"),
        (
            high_noise(),
            Params {
                shadow_match: ShadowMatch::On,
                ..Params::default()
            },
            2e-7,
            "high noise, matching",
        ),
    ];
    for (noise, params, floor, label) in configs {
        let (mut worst_rel, mut worst_abs) = (0.0f64, 0.0f64);
        for cfa in Cfa::ALL {
            for gains in [[1.0f32; 3], [1.9, 1.0, 1.4], [2.7, 1.0, 1.1]] {
                for level in [0.003f32, 0.18, 1.0, 2.4] {
                    let (w, h) = (23usize, 18usize);
                    let mut m = Mosaic::new(w, h, cfa);
                    m.data.fill(level);
                    let mut c = Case::new(m, noise);
                    c.gains = gains;
                    c.params = params;
                    let out = c.gpu(dev);
                    for ch in 0..3 {
                        let want = f64::from(gains[ch]) * f64::from(level);
                        for &v in out.plane(ch) {
                            let abs = (f64::from(v) - want).abs();
                            worst_abs = worst_abs.max(abs);
                            worst_rel = worst_rel.max(abs / want);
                            assert!(
                                abs <= 4e-7 * want + floor,
                                "{label} {} gains {gains:?} level {level}: plane {ch} has {v}, want {want}",
                                cfa.name()
                            );
                        }
                    }
                }
            }
        }
        println!(
            "flat fields, {label}: worst absolute error {worst_abs:.3e}, worst relative {worst_rel:.3e}"
        );
    }
}

#[test]
fn cropping_by_one_row_or_column_changes_the_phase_and_keeps_the_interior() {
    // Spec 6.2 item 3, on the GPU: a crop by one photosite flips the Bayer phase of the buffer; the
    // interior (the halo away from the cut) is the same as in the uncropped GPU run.
    let Some(dev) = f32_device() else { return };
    for (noise, halo) in [(low_noise(), 11usize), (high_noise(), 13)] {
        let m = scene_for(noise, 80, 64, Cfa::RGGB, 77);
        let full = Case::new(m.clone(), noise);
        let whole = full.gpu(dev);
        for (dx, dy) in [(1usize, 0usize), (0, 1), (1, 1)] {
            let cropped = m.extract(Rect::new(dx, dy, m.width - dx, m.height - dy));
            assert_ne!(cropped.cfa, m.cfa);
            let mut c = full.clone();
            c.mosaic = cropped;
            let got = c.gpu(dev);
            let mut differ = 0usize;
            let mut worst = 0.0f64;
            for ch in 0..3 {
                for y in halo..got.height - halo {
                    for x in halo..got.width - halo {
                        let a = got.plane(ch)[y * got.width + x];
                        let b = whole.plane(ch)[(y + dy) * whole.width + x + dx];
                        worst = worst.max((f64::from(a) - f64::from(b)).abs());
                        differ += usize::from(a.to_bits() != b.to_bits());
                    }
                }
            }
            println!(
                "crop ({dx},{dy}) halo {halo}: {differ} interior samples differ, worst {worst:.3e}"
            );
            assert_eq!(differ, 0, "crop ({dx},{dy}), halo {halo}");
        }
    }
}

#[test]
fn one_photosite_reaches_the_exact_receptive_field_and_no_further() {
    // Spec 1.5 and 6.2 item 5: perturbing one photosite changes the output within the declared halo
    // (11 px, 13 with shadow matching); the CPU port measures a reach of 9 px plain with the
    // sharpening active and 10 px with matching. Every sample farther than the halo is bit-identical,
    // and the GPU reach equals the CPU's for the same perturbations.
    let Some(dev) = f32_device() else { return };
    let (w, h) = (61usize, 55usize);
    // A dark scene against a noise model whose shadow gate is open (T = 24 DN here), matching off;
    // and the noisy dark scene with matching on.
    let plain_noise = NoiseDn::new(0.05, 0.5);
    for (noise, halo, label) in [
        (plain_noise, 11usize, "plain, sharpening on"),
        (high_noise(), 13, "shadow matching on"),
    ] {
        let (mut reach, mut cpu_reach) = (0usize, 0usize);
        for cfa in [Cfa::RGGB, Cfa::GBRG] {
            let mut m = scene_for(noise, w, h, cfa, 55);
            if halo == 11 {
                for v in &mut m.data {
                    *v *= 1.0e-4;
                }
            }
            let peak = m.data.iter().copied().fold(0.0, f32::max);
            let c = Case::new(m, noise);
            let (base, cpu_base) = (c.gpu(dev), c.cpu());
            for (px, py) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                let (cx, cy) = (30 + px, 26 + py);
                let mut bumped = c.clone();
                let i = cy * w + cx;
                bumped.mosaic.data[i] = bumped.mosaic.data[i] * 2.0 + 0.5 * peak;
                let (out, cpu_out) = (bumped.gpu(dev), bumped.cpu());
                for ch in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            let j = y * w + x;
                            let r = x.abs_diff(cx).max(y.abs_diff(cy));
                            if out.plane(ch)[j].to_bits() != base.plane(ch)[j].to_bits() {
                                assert!(
                                    r <= halo,
                                    "{label} {} ({cx}, {cy}) changed plane {ch} at ({x}, {y}), {r} px away",
                                    cfa.name()
                                );
                                reach = reach.max(r);
                            }
                            if cpu_out.plane(ch)[j].to_bits() != cpu_base.plane(ch)[j].to_bits() {
                                cpu_reach = cpu_reach.max(r);
                            }
                        }
                    }
                }
            }
        }
        println!(
            "{label}: receptive-field radius {reach} px on the GPU, {cpu_reach} px on the CPU (halo {halo})"
        );
        assert!(reach <= halo, "{label}: reach {reach}");
        assert_eq!(
            reach, cpu_reach,
            "{label}: the GPU and the CPU port reach differently"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The evidence for the explanation of the differences
// ---------------------------------------------------------------------------------------------

/// `case` with every photosite multiplied by `1 + r * rel`, `r` uniform in `[-1, 1)`.
fn jitter(case: &Case, seed: u64, rel: f32) -> Case {
    let mut rng = Rng(seed);
    let mut c = case.clone();
    for v in &mut c.mosaic.data {
        let r = (rng.unit() * 2.0 - 1.0) as f32;
        *v *= 1.0 + r * rel;
    }
    c
}

#[test]
fn cpu_is_as_sensitive_as_the_gpu_is_different() {
    // The CPU reference moved by 1 ulp of input noise (relative 1.2e-7) and the GPU moved by f32
    // arithmetic: where the GPU differs from the CPU, the CPU differs from itself by as much under a
    // rounding-sized perturbation. Counted on the tie vectors, the zone plate and noisy frames.
    let Some(dev) = f32_device() else { return };
    let mut cases: Vec<(String, Case)> = load_vectors()
        .into_iter()
        .filter(|v| TIE_VECTORS.iter().any(|t| t.0 == v.name) || v.name == "zone_plate")
        .map(|v| (v.name, v.case))
        .collect();
    cases.push((
        "natural 120x90 low noise".into(),
        Case::new(scene(120, 90, Cfa::GBRG, 31, 1.3), low_noise()),
    ));
    cases.push((
        "dark 120x90 high noise".into(),
        Case::new(dark_scene(120, 90, Cfa::RGGB, 32), high_noise()),
    ));
    for (name, case) in cases {
        let base = case.cpu();
        let gpu = compare(&case.gpu(dev), &base);
        let mut cpu_over = Vec::new();
        for seed in 1..=4u64 {
            let s = compare(&jitter(&case, seed, 1.2e-7).cpu(), &base);
            cpu_over.push((s.over, s.max_abs));
        }
        let most = cpu_over.iter().map(|c| c.0).max().unwrap_or(0);
        let max_abs = cpu_over.iter().map(|c| c.1).fold(0.0, f64::max);
        println!(
            "{name:<26} GPU vs CPU: {} over 1e-4, max {:.2e} | CPU under 1-ulp jitter (4 seeds): {:?} over 1e-4, max {max_abs:.2e}",
            gpu.over,
            gpu.max_abs,
            cpu_over.iter().map(|c| c.0).collect::<Vec<_>>(),
        );
        // The GPU's f32 sums are noisier than 1 ulp of input, so allow a factor of eight and a floor.
        assert!(
            gpu.over <= 8 * most + 16,
            "{name}: the GPU differs at {} samples, the CPU moves at most {most} under 1 ulp",
            gpu.over
        );
    }
}

// MODEL-DUMP-BEGIN (temporary, removed before the end of the task)
#[test]
#[ignore = "development aid: YM_MODEL_DUMP=<dir> writes the MSL of every kernel for the clang model"]
fn zz_msl_dump() {
    let Ok(root) = std::env::var("YM_MODEL_DUMP") else {
        return;
    };
    let root = std::path::Path::new(&root);
    std::fs::create_dir_all(root).expect("dir");
    for id in ym_gpu::kernels::KernelId::ALL {
        let src = ym_gpu::kernels::shader_source(ym_gpu::device::StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&src).expect("parse");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("valid");
        let mut resources = naga::back::msl::BindingMap::default();
        for (_, var) in module.global_variables.iter() {
            if let Some(rb) = var.binding {
                let mutable = matches!(var.space, naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE));
                resources.insert(
                    rb,
                    naga::back::msl::BindTarget {
                        buffer: Some(rb.binding as u8),
                        texture: None,
                        sampler: None,
                        external_texture: None,
                        mutable,
                    },
                );
            }
        }
        let mut map = naga::back::msl::EntryPointResourceMap::default();
        map.insert(
            "main".to_string(),
            naga::back::msl::EntryPointResources {
                resources,
                immediates_buffer: None,
                sizes_buffer: Some(30),
            },
        );
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            fake_missing_bindings: false,
            per_entry_point_map: map,
            ..naga::back::msl::Options::default()
        };
        let (msl, _) = naga::back::msl::write_string(
            &module,
            &info,
            &options,
            &naga::back::msl::PipelineOptions::default(),
        )
        .expect("msl");
        std::fs::write(root.join(format!("{id:?}.metal")), msl).expect("write");
    }
}
#[test]
#[ignore = "development aid: YM_AR_DISPATCH_LOG=<file> records the dispatches of the vectors and frames"]
fn zz_ar_dump() {
    use ym_gpu::kernels::adaptive_ratio::model_log;
    if std::env::var_os("YM_AR_DISPATCH_LOG").is_none() {
        return;
    }
    let Some(dev) = f32_device() else { return };
    let save = |name: &str, c: &Case| {
        model_log(7, name.as_bytes());
        let got = c.gpu(dev);
        model_log(5, bytemuck::cast_slice(&got.data));
        model_log(6, bytemuck::cast_slice(&c.cpu().data));
    };
    for v in load_vectors() {
        save(&format!("vec {}", v.name), &v.case);
    }
    if std::env::var_os("YM_AR_DUMP_FRAMES").is_some() {
        let mut seed = 100;
        for cfa in Cfa::ALL {
            for (w, h) in [(67usize, 41usize), (40, 40), (31, 18), (9, 7)] {
                for (n, noise) in [
                    ("zero", NoiseDn::ZERO),
                    ("low", low_noise()),
                    ("high", high_noise()),
                ] {
                    for (k, params) in param_sets().into_iter().enumerate() {
                        seed += 1;
                        let mut c = Case::new(scene_for(noise, w, h, cfa, seed), noise);
                        c.params = params;
                        save(&format!("frame {} {w}x{h} {n} {k}", cfa.name()), &c);
                    }
                }
            }
        }
    }
}
// MODEL-DUMP-END
