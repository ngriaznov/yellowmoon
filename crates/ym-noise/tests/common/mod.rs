//! Shared helpers of the synthetic validation: random cameras, scenes, and error statistics.
//! Used by `tests/validation.rs` and by the `validate_synthetic` example.
#![allow(dead_code)]

use ym_noise::estimate::{EstimateInput, EstimatorConfig, F32Plane, NoiseEstimate, estimate};
use ym_noise::model::NoiseModel;
use ym_noise::model::synth::{SceneKind, SceneSpec, SynthRng, add_noise, render_scene};
use ym_noise::priors::NoisePrior;

/// Number of cases to run: `n` times the integer in the environment variable `YM_NOISE_CASES`
/// (default 1). The default counts keep the suite short in debug builds; set it to 3 or more for the
/// numbers quoted in `docs/STATUS.md` (or use the release-mode `validate_synthetic` example).
pub fn scaled(n: u64) -> u64 {
    n * std::env::var("YM_NOISE_CASES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&k| k > 0)
        .unwrap_or(1)
}

/// 14-bit normalisation: one DN.
pub const DN: f64 = 1.0 / 15871.0;

/// A random camera at an ISO-like setting: gain proportional to ISO, read noise from an
/// input-referred electron count plus a post-gain DN floor, plane offsets of a few DN.
pub fn random_camera(rng: &mut SynthRng, iso: f64, with_row_noise: bool) -> NoiseModel {
    let a100 = (1.5e-5f64.ln() + (4.0e-5f64.ln() - 1.5e-5f64.ln()) * rng.uniform()).exp();
    let a = a100 * iso / 100.0;
    let sigma_e = (1.2f64.ln() + (6.0f64.ln() - 1.2f64.ln()) * rng.uniform()).exp();
    let post_dn = 0.6 + 0.8 * rng.uniform();
    let b = (a * sigma_e).powi(2) + (post_dn * DN).powi(2);
    let delta = std::array::from_fn(|_| (rng.normal() * 1.5 * DN).clamp(-4.0 * DN, 4.0 * DN));
    let sigma_row = if with_row_noise {
        0.15 * b.sqrt() * (0.5 + rng.uniform())
    } else {
        0.0
    };
    NoiseModel {
        a,
        b,
        delta,
        sigma_row,
        q: DN,
    }
}

/// The noise model as the estimator sees it inside rows (without the row term), no offsets.
pub fn pixel_model(m: &NoiseModel) -> NoiseModel {
    NoiseModel {
        delta: [0.0; 4],
        sigma_row: 0.0,
        ..*m
    }
}

/// ISO ladder: thirds of a stop from 100 to 25600.
pub fn iso_ladder() -> Vec<f64> {
    (0..=24)
        .map(|k| 100.0 * 2f64.powf(k as f64 / 3.0))
        .collect()
}

pub struct Case {
    pub spec: SceneSpec,
    pub truth: NoiseModel,
    pub noise_seed: u64,
    pub iso: f64,
}

pub fn make_case(
    seed: u64,
    kind: SceneKind,
    width: usize,
    height: usize,
    exposure: f32,
    with_row_noise: bool,
) -> Case {
    let mut rng = SynthRng::new(seed.wrapping_mul(0x9E37_79B9) ^ 0x1234_5678);
    let isos = iso_ladder();
    let iso = isos[(rng.uniform() * isos.len() as f64) as usize % isos.len()];
    Case {
        spec: SceneSpec {
            width,
            height,
            seed,
            kind,
            exposure,
        },
        truth: random_camera(&mut rng, iso, with_row_noise),
        noise_seed: seed ^ 0xABCD_EF01,
        iso,
    }
}

pub fn noisy(case: &Case) -> Vec<f32> {
    let scene = render_scene(&case.spec);
    add_noise(
        &scene.signal,
        case.spec.width,
        case.spec.height,
        &case.truth,
        case.noise_seed,
    )
}

pub fn run(
    mosaic: &[f32],
    w: usize,
    h: usize,
    q: f64,
    prior: &NoisePrior,
    cfg: &EstimatorConfig,
) -> NoiseEstimate {
    let plane = F32Plane::new(mosaic, w, h).unwrap();
    estimate(
        &EstimateInput {
            mosaic: &plane,
            qstep: None,
            q_norm: q,
        },
        prior,
        cfg,
    )
    .unwrap()
}

/// Intensities at which sigma errors are reported.
pub const PROBES: [f64; 3] = [0.02, 0.18, 0.5];

/// Relative error of sigma at the probe intensities (estimate over truth, minus one), computed
/// with the pixel model (no row term) on both sides, and the signed relative error of `a`.
pub fn errors(est: &NoiseModel, truth: &NoiseModel) -> (f64, [f64; 3]) {
    let (e, t) = (pixel_model(est), pixel_model(truth));
    let s = PROBES.map(|x| e.sigma(x, 0) / t.sigma(x, 0) - 1.0);
    (est.a / truth.a - 1.0, s)
}

pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1;
    v[k]
}

/// Median of |x| and its 90th percentile.
pub fn abs_stats(v: &[f64]) -> (f64, f64) {
    let mut a: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    (median(&mut a.clone()), percentile(&mut a, 0.9))
}

/// A small edge-preserving denoiser for the refinement test: a bilateral filter on each CFA
/// plane (5 x 5 same-colour neighbours) whose range kernel is `range_sigmas` times the noise
/// sigma the model predicts at the centre value.
pub fn bilateral_denoise(
    mosaic: &[f32],
    w: usize,
    h: usize,
    model: &NoiseModel,
    range_sigmas: f64,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let plane = ((y & 1) << 1) | (x & 1);
            let c = f64::from(mosaic[y * w + x]);
            let sigma = model.sigma(c.max(0.0), plane).max(1e-6) * range_sigmas;
            let inv = 1.0 / (2.0 * sigma * sigma);
            let (mut acc, mut wsum) = (0.0f64, 0.0f64);
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (yy, xx) = (y as i32 + 2 * dy, x as i32 + 2 * dx);
                    if yy < 0 || xx < 0 || yy >= h as i32 || xx >= w as i32 {
                        continue;
                    }
                    let v = f64::from(mosaic[yy as usize * w + xx as usize]);
                    let spatial = (-(f64::from(dx * dx + dy * dy)) / 4.0).exp();
                    let weight = spatial * (-(v - c).powi(2) * inv).exp();
                    acc += weight * v;
                    wsum += weight;
                }
            }
            out[y * w + x] = (acc / wsum) as f32;
        }
    }
    out
}
