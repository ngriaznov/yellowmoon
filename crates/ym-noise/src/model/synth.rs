//! Synthetic mosaics with known Poisson-Gaussian noise, for validation, benchmarks and examples.
//!
//! V1 has no calibration frames and the corpus is not always reachable, so the estimator is
//! validated on procedural scenes (gradients, edges, fine texture, flat patches, clipped skies)
//! mosaicked to RGGB and corrupted with noise from a known [`NoiseModel`]. Everything is a pure
//! function of its seed: scene pixels are hashed from their coordinates, and noise comes from one
//! ChaCha8 stream per mosaic row, so results do not depend on the number of threads.
//!
//! The noise generator follows the model exactly. For a photosite in plane `c` with
//! electron-proportional signal `s >= 0` (normalised units):
//!
//! ```text
//! x = a * Poisson(s / a) + delta_c + N(0, b) + rho_row,   rho_row ~ N(0, sigma_row^2)
//! x = round(clip(x, -0.05, 1) / q) * q
//! ```
//!
//! so the recorded mean is `s + delta_c` and the variance is `a s + b + sigma_row^2 + q^2/12`.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use rayon::prelude::*;

use super::NoiseModel;

/// The lowest value R1 keeps (`docs/05-raw-processing.md` section R1).
pub const NEGATIVE_CLAMP: f32 = -0.05;

/// Small deterministic random source: uniform, normal and Poisson draws on a ChaCha8 stream.
pub struct SynthRng {
    rng: ChaCha8Rng,
    spare: Option<f64>,
}

impl SynthRng {
    /// A stream seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            spare: None,
        }
    }

    /// The stream for row `row` of a field generated from `seed`; independent of other rows.
    pub fn for_row(seed: u64, row: usize) -> Self {
        Self::new(mix64(
            seed ^ mix64(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(row as u64 + 1)),
        ))
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = 1.0 - self.uniform(); // (0, 1]
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }

    /// Poisson with mean `lambda`: exact (Knuth) below 30, a rounded-free normal approximation
    /// above, which has the exact mean and variance (what the validation needs).
    pub fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        if lambda < 30.0 {
            let limit = (-lambda).exp();
            let mut k = 0.0;
            let mut p = self.uniform();
            while p > limit {
                k += 1.0;
                p *= self.uniform();
            }
            k
        } else {
            (lambda + lambda.sqrt() * self.normal()).max(0.0)
        }
    }
}

/// SplitMix64 finaliser: a cheap, well-mixed 64-bit hash.
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hash01(seed: u64, a: i64, b: i64) -> f32 {
    let h = mix64(seed ^ mix64(a as u64) ^ mix64((b as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)));
    (h >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

/// Smooth value noise in [-1, 1] with feature size `wavelength_px`.
fn value_noise(seed: u64, x: f32, y: f32, wavelength_px: f32) -> f32 {
    let fx = x / wavelength_px;
    let fy = y / wavelength_px;
    let (ix, iy) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - ix, fy - iy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let (ix, iy) = (ix as i64, iy as i64);
    let v = |dx: i64, dy: i64| hash01(seed, ix + dx, iy + dy) * 2.0 - 1.0;
    let top = v(0, 0) + (v(1, 0) - v(0, 0)) * sx;
    let bot = v(0, 1) + (v(1, 1) - v(0, 1)) * sx;
    top + (bot - top) * sy
}

/// What mix of materials a procedural scene contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// A typical photograph: flat walls and sky, gradients, texture, edges, some dark and bright areas.
    Mixed,
    /// Texture and stripes almost everywhere, no flat patches larger than a few pixels.
    Busy,
    /// Only flat patches and gentle gradients.
    Smooth,
}

/// Parameters of a procedural scene.
#[derive(Clone, Copy, Debug)]
pub struct SceneSpec {
    /// Mosaic width in photosites (even).
    pub width: usize,
    /// Mosaic height in photosites (even).
    pub height: usize,
    /// Seed of the scene.
    pub seed: u64,
    /// Material mix.
    pub kind: SceneKind,
    /// Global brightness scale; values above about 1.4 push parts of the scene over the clip level.
    pub exposure: f32,
}

/// A noise-free RGGB mosaic of electron-proportional signal in normalised units. Values can
/// exceed 1.0: the noise generator clips them.
pub struct Scene {
    /// Mosaic width.
    pub width: usize,
    /// Mosaic height.
    pub height: usize,
    /// Row-major signal, `width * height` values.
    pub signal: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Material {
    Flat,
    Ramp,
    Texture,
    Stripes,
    Bright,
    Dark,
}

fn pick_material(kind: SceneKind, u: f32) -> Material {
    let table: &[(Material, f32)] = match kind {
        SceneKind::Mixed => &[
            (Material::Flat, 0.24),
            (Material::Ramp, 0.20),
            (Material::Texture, 0.22),
            (Material::Stripes, 0.08),
            (Material::Bright, 0.12),
            (Material::Dark, 0.14),
        ],
        SceneKind::Busy => &[(Material::Texture, 0.80), (Material::Stripes, 0.20)],
        SceneKind::Smooth => &[
            (Material::Flat, 0.55),
            (Material::Ramp, 0.35),
            (Material::Dark, 0.10),
        ],
    };
    let mut acc = 0.0;
    for (m, w) in table {
        acc += w;
        if u < acc {
            return *m;
        }
    }
    table[table.len() - 1].0
}

struct Shape {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    disc: bool,
    gain: f32,
}

fn make_shapes(spec: &SceneSpec, count: usize) -> Vec<Shape> {
    let mut rng = SynthRng::new(mix64(spec.seed ^ 0x5348_4150_4553));
    let (w, h) = (spec.width as f32, spec.height as f32);
    (0..count)
        .map(|_| {
            let size = 12.0 + (rng.uniform() as f32).powi(2) * 0.18 * w.min(h);
            Shape {
                cx: rng.uniform() as f32 * w,
                cy: rng.uniform() as f32 * h,
                rx: size * (0.4 + rng.uniform() as f32),
                ry: size * (0.4 + rng.uniform() as f32),
                disc: rng.uniform() < 0.5,
                gain: 0.25 * 16.0f32.powf(rng.uniform() as f32),
            }
        })
        .collect()
}

/// Renders the scene described by `spec` as an RGGB mosaic of signal.
///
/// Cells of a jittered grid get a material (flat, ramp, texture, stripes, bright, dark) and a
/// level; shapes with hard edges are laid over them; the three colour channels differ by slowly
/// varying gains, so the planes sit at different levels as in an un-balanced raw file.
pub fn render_scene(spec: &SceneSpec) -> Scene {
    let (w, h) = (spec.width, spec.height);
    assert!(w % 2 == 0 && h % 2 == 0, "scene size must be even");
    let cell = ((w.min(h) as f32) / 5.0).clamp(64.0, 220.0);
    let shapes = make_shapes(spec, 6 + (w * h) / 150_000);
    let seed = spec.seed;
    let mut rng = SynthRng::new(mix64(seed ^ 0xC010_5EED));
    let r_gain = 0.42 + 0.45 * rng.uniform() as f32;
    let b_gain = 0.38 + 0.42 * rng.uniform() as f32;
    let exposure = spec.exposure;
    let kind = spec.kind;
    let mut signal = vec![0.0f32; w * h];
    signal.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, out) in row.iter_mut().enumerate() {
            let (fx, fy) = (x as f32, y as f32);
            // Nearest jittered-grid seed point.
            let (gx, gy) = ((fx / cell).floor() as i64, (fy / cell).floor() as i64);
            let mut best = (f32::MAX, 0i64, 0i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cx, cy) = (gx + dx, gy + dy);
                    let px = (cx as f32 + 0.15 + 0.7 * hash01(seed ^ 1, cx, cy)) * cell;
                    let py = (cy as f32 + 0.15 + 0.7 * hash01(seed ^ 2, cx, cy)) * cell;
                    let d = (px - fx).powi(2) + (py - fy).powi(2);
                    if d < best.0 {
                        best = (d, cx, cy);
                    }
                }
            }
            let (_, cx, cy) = best;
            let u = hash01(seed ^ 3, cx, cy);
            let level_u = hash01(seed ^ 4, cx, cy);
            let amp_u = hash01(seed ^ 5, cx, cy);
            let wave_u = hash01(seed ^ 6, cx, cy);
            let ang_u = hash01(seed ^ 7, cx, cy);
            let base = (0.02f32.ln() + (0.9f32.ln() - 0.02f32.ln()) * level_u).exp();
            let mut level = match pick_material(kind, u) {
                Material::Flat => base,
                Material::Ramp => {
                    let t = ((fx - cx as f32 * cell) * ang_u.cos()
                        + (fy - cy as f32 * cell) * ang_u.sin())
                        / cell;
                    base * (1.0 + 0.45 * (amp_u - 0.5) * t.clamp(-1.0, 2.0))
                }
                Material::Texture => {
                    let wl = 2.4 + 7.0 * wave_u;
                    let n = 0.65 * value_noise(seed ^ 11, fx, fy, wl)
                        + 0.35 * value_noise(seed ^ 12, fx, fy, wl * 2.3);
                    base * (1.0 + (0.12 + 0.38 * amp_u) * n)
                }
                Material::Stripes => {
                    let wl = 3.0 + 11.0 * wave_u;
                    let phase = (fx * ang_u.cos() + fy * ang_u.sin()) / wl;
                    base * (1.0 + (0.10 + 0.30 * amp_u) * (std::f32::consts::TAU * phase).sin())
                }
                Material::Bright => {
                    1.05 + 0.75 * level_u + 0.05 * value_noise(seed ^ 13, fx, fy, 40.0)
                }
                Material::Dark => 0.003 + 0.03 * level_u,
            };
            for s in &shapes {
                let inside = if s.disc {
                    ((fx - s.cx) / s.rx).powi(2) + ((fy - s.cy) / s.ry).powi(2) < 1.0
                } else {
                    (fx - s.cx).abs() < s.rx && (fy - s.cy).abs() < s.ry * 0.6
                };
                if inside {
                    level *= s.gain;
                }
            }
            let lum = (level * exposure).max(0.0);
            let gain = match (y & 1, x & 1) {
                (0, 0) => r_gain * (1.0 + 0.12 * value_noise(seed ^ 21, fx, fy, 400.0)),
                (1, 1) => b_gain * (1.0 + 0.12 * value_noise(seed ^ 22, fx, fy, 400.0)),
                _ => 1.0,
            };
            *out = lum * gain;
        }
    });
    Scene {
        width: w,
        height: h,
        signal,
    }
}

/// Adds Poisson-Gaussian noise from `truth` to the electron-proportional `signal` of a
/// `width x height` RGGB mosaic and returns the recorded mosaic (clipped at 1.0, floored at
/// [`NEGATIVE_CLAMP`], quantised to `truth.q` when it is positive).
pub fn add_noise(
    signal: &[f32],
    width: usize,
    height: usize,
    truth: &NoiseModel,
    seed: u64,
) -> Vec<f32> {
    assert_eq!(signal.len(), width * height);
    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let mut rng = SynthRng::for_row(seed, y);
        let row_offset = truth.sigma_row * rng.normal();
        let read_sigma = truth.b.sqrt();
        let src = &signal[y * width..(y + 1) * width];
        for (x, (o, &s)) in row.iter_mut().zip(src).enumerate() {
            let plane = ((y & 1) << 1) | (x & 1);
            let lambda = f64::from(s.max(0.0)) / truth.a;
            let shot = truth.a * rng.poisson(lambda);
            let v = shot + truth.delta[plane] + read_sigma * rng.normal() + row_offset;
            *o = finish(v, truth.q);
        }
    });
    out
}

fn finish(v: f64, q: f64) -> f32 {
    let mut v = v.clamp(f64::from(NEGATIVE_CLAMP), 1.0);
    if q > 0.0 {
        v = ((v / q).round() * q).clamp(f64::from(NEGATIVE_CLAMP), 1.0);
    }
    v as f32
}

/// Adds a heteroscedastic Gaussian increment `N(0, a_inc y + b_inc)` to an already noisy mosaic
/// `y` in place (the noise-on-noise test of `docs/06-noise.md` section Validation), then clips at
/// 1.0 and quantises to `q` again when it is positive.
pub fn add_increment(mosaic: &mut [f32], width: usize, a_inc: f64, b_inc: f64, q: f64, seed: u64) {
    mosaic
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let mut rng = SynthRng::for_row(seed ^ 0x1AC4_E3E7, y);
            for v in row.iter_mut() {
                let var = a_inc * f64::from(v.max(0.0)) + b_inc;
                let n = var.sqrt() * rng.normal();
                *v = finish(f64::from(*v) + n, q);
            }
        });
}

/// Spatial filters that imitate in-camera raw filtering (the "filtered file" case of `docs/06`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawFilter {
    /// `(1 - weight) * x + weight * mean3x3(x)` on each CFA plane.
    BoxBlend {
        /// Weight of the 3x3 same-colour mean, in [0, 1].
        weight: f32,
    },
    /// 3x3 median of same-colour neighbours.
    Median3,
    /// Horizontal `[1 2 1] / 4` smoothing of same-colour neighbours.
    Horizontal121,
}

/// Applies `filter` to each CFA plane of the mosaic and returns the result, requantised to `q`.
pub fn apply_raw_filter(
    mosaic: &[f32],
    width: usize,
    height: usize,
    filter: RawFilter,
    q: f64,
) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            // Same-colour neighbours are two photosites apart; at the border mirror inwards so
            // the plane (parity) is kept.
            let at = |dx: isize, dy: isize| -> f32 {
                let mirror = |p: usize, d: isize, n: usize| -> usize {
                    let q = p as isize + 2 * d;
                    if q < 0 || q >= n as isize {
                        (p as isize - 2 * d) as usize
                    } else {
                        q as usize
                    }
                };
                mosaic[mirror(y, dy, height) * width + mirror(x, dx, width)]
            };
            let v = match filter {
                RawFilter::BoxBlend { weight } => {
                    let mut s = 0.0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            s += at(dx, dy);
                        }
                    }
                    (1.0 - weight) * at(0, 0) + weight * s / 9.0
                }
                RawFilter::Median3 => {
                    let mut v = [0.0f32; 9];
                    let mut i = 0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            v[i] = at(dx, dy);
                            i += 1;
                        }
                    }
                    v.sort_by(f32::total_cmp);
                    v[4]
                }
                RawFilter::Horizontal121 => 0.25 * at(-1, 0) + 0.5 * at(0, 0) + 0.25 * at(1, 0),
            };
            *o = finish(f64::from(v), q);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_rows_are_independent_of_order() {
        let mut a = SynthRng::for_row(7, 3);
        let mut b = SynthRng::for_row(7, 3);
        assert_eq!(a.normal().to_bits(), b.normal().to_bits());
        let mut c = SynthRng::for_row(7, 4);
        assert_ne!(
            SynthRng::for_row(7, 3).uniform().to_bits(),
            c.uniform().to_bits()
        );
    }

    #[test]
    fn poisson_and_normal_have_the_right_moments() {
        let mut rng = SynthRng::new(1);
        for &lambda in &[3.0, 12.0, 25.0, 400.0, 20_000.0] {
            let n = 200_000;
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let k = rng.poisson(lambda);
                s += k;
                s2 += k * k;
            }
            let mean = s / n as f64;
            let var = s2 / n as f64 - mean * mean;
            assert!(
                (mean / lambda - 1.0).abs() < 0.01,
                "mean {mean} vs {lambda}"
            );
            assert!((var / lambda - 1.0).abs() < 0.03, "var {var} vs {lambda}");
        }
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..200_000 {
            let z = rng.normal();
            s += z;
            s2 += z * z;
        }
        assert!((s / 200_000.0).abs() < 0.01);
        assert!((s2 / 200_000.0 - 1.0).abs() < 0.02);
    }

    #[test]
    fn scene_is_deterministic_and_has_a_wide_range() {
        let spec = SceneSpec {
            width: 512,
            height: 384,
            seed: 11,
            kind: SceneKind::Mixed,
            exposure: 1.0,
        };
        let a = render_scene(&spec);
        let b = render_scene(&spec);
        assert_eq!(a.signal, b.signal);
        let max = a.signal.iter().cloned().fold(0.0f32, f32::max);
        let min = a.signal.iter().cloned().fold(f32::MAX, f32::min);
        assert!(min >= 0.0 && max > 0.5, "range {min}..{max}");
    }

    #[test]
    fn noise_matches_the_model_on_a_flat_field() {
        let (w, h) = (512, 512);
        let truth = NoiseModel {
            a: 1e-3,
            b: 4e-7,
            delta: [2e-4, 0.0, 0.0, -2e-4],
            sigma_row: 0.0,
            q: 1.0 / 16383.0,
        };
        let signal = vec![0.2f32; w * h];
        let noisy = add_noise(&signal, w, h, &truth, 5);
        for plane in 0..4 {
            let (mut s, mut s2, mut n) = (0.0f64, 0.0f64, 0.0f64);
            for y in (plane >> 1..h).step_by(2) {
                for x in (plane & 1..w).step_by(2) {
                    let v = f64::from(noisy[y * w + x]);
                    s += v;
                    s2 += v * v;
                    n += 1.0;
                }
            }
            let mean = s / n;
            let var = s2 / n - mean * mean;
            let want_mean = 0.2 + truth.delta[plane];
            let want_var = truth.variance(0.2 + truth.delta[plane], plane);
            assert!((mean - want_mean).abs() < 3e-4, "plane {plane} mean {mean}");
            assert!(
                (var / want_var - 1.0).abs() < 0.03,
                "plane {plane} var {var} vs {want_var}"
            );
        }
    }

    #[test]
    fn increment_adds_the_requested_variance() {
        let (w, h) = (512, 512);
        let mut m = vec![0.3f32; w * h];
        add_increment(&mut m, w, 2e-3, 1e-6, 0.0, 9);
        let mean: f64 = m.iter().map(|&v| f64::from(v)).sum::<f64>() / (w * h) as f64;
        let var: f64 = m
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / (w * h) as f64;
        let want = 2e-3 * 0.3 + 1e-6;
        assert!((var / want - 1.0).abs() < 0.03, "var {var} vs {want}");
    }

    #[test]
    fn filters_keep_the_plane_parity_and_correlate_the_noise() {
        let (w, h) = (256, 256);
        let truth = NoiseModel::new(1e-3, 1e-6);
        let noisy = add_noise(&vec![0.3f32; w * h], w, h, &truth, 3);
        let filtered = apply_raw_filter(&noisy, w, h, RawFilter::BoxBlend { weight: 1.0 }, 0.0);
        let var = |m: &[f32]| {
            let mean: f64 = m.iter().map(|&v| f64::from(v)).sum::<f64>() / m.len() as f64;
            m.iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / m.len() as f64
        };
        assert!(var(&filtered) < 0.3 * var(&noisy));
        let med = apply_raw_filter(&noisy, w, h, RawFilter::Median3, 0.0);
        assert!(var(&med) < 0.7 * var(&noisy));
    }
}
