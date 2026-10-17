//! Helpers shared by the raw front tests (`gpu_front.rs`, `gpu_lens.rs`, `gpu_reduced.rs`,
//! `gpu_chain.rs`): raw scenes with clipping, the CPU front of the chain (R1, R3, R4, R5, R5b) on a
//! whole frame, and a device job runner.
//!
//! A directory module, so cargo does not build it as a test binary of its own.

#![allow(dead_code)]

use ym_gpu::cpu::gains;
use ym_gpu::cpu::highlights;
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::vignette::{self, VignetteProfile};
use ym_gpu::cpu::{Cfa, ClipMask, Mosaic};
use ym_gpu::device::{GpuContext, GpuError};
use ym_gpu::queue::{JobSpec, Priority};

use super::common::TestGpu;

/// A raw frame as the decoder delivers it.
#[derive(Clone)]
pub struct RawScene {
    pub width: usize,
    pub height: usize,
    pub cfa: Cfa,
    pub data: Vec<u16>,
    pub qstep: Option<Vec<f32>>,
}

/// The tags of a mild vignetting profile (positive knots: the corners are darker, the gain above 1).
pub const VIGNETTE_KNOTS: [i16; 6] = [0, 400, 1500, 3500, 6000, 9000];

pub fn vignette_profile() -> VignetteProfile {
    VignetteProfile::from_tag(&VIGNETTE_KNOTS).expect("vignette")
}

/// Levels with a different black and white at each of the four positions.
pub fn uneven_levels() -> Levels {
    Levels {
        black: [100.0, 102.0, 101.0, 99.0],
        white: [16383.0, 16000.0, 16100.0, 15900.0],
    }
}

/// Levels with the full 16-bit range (words 0 and 65535 are in range).
pub fn full_levels() -> Levels {
    Levels {
        black: [0.0, 4.0, 2.0, 1.0],
        white: [65535.0; 4],
    }
}

impl RawScene {
    /// Noisy natural content exposed so that highlights clip, with the words 0 and the white level
    /// of each position in it, for `levels`.
    pub fn new(width: usize, height: usize, cfa: Cfa, seed: u64, levels: &Levels) -> RawScene {
        let img = synth::natural(width, height, seed);
        let m = synth::mosaic_from_rgb(&img, cfa);
        let mut rng = synth::Rng::new(seed + 7);
        let mut data: Vec<u16> = m
            .data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let p = ((i / width) & 1) * 2 + ((i % width) & 1);
                let range = levels.white[p] - levels.black[p];
                let x = levels.black[p] + (v * 1.35 + 0.004 * rng.gaussian()) * range;
                x.clamp(0.0, 65535.0) as u16
            })
            .collect();
        // The extremes: a word of 0, a word at the white level of its position, one above it.
        if data.len() > 8 {
            let last = data.len() - 1;
            data[0] = 0;
            data[1] = levels.white[1].min(65535.0) as u16;
            data[2] = 65535;
            data[width.min(last)] = levels.black[2] as u16;
        }
        RawScene {
            width,
            height,
            cfa,
            data,
            qstep: None,
        }
    }

    /// The same scene with a step plane of the given kind.
    pub fn with_steps(mut self, many: bool, seed: u64) -> RawScene {
        let mut rng = synth::Rng::new(seed);
        let table = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        self.qstep = Some(
            (0..self.data.len())
                .map(|_| {
                    if many {
                        // Many distinct f16 values (more than 256).
                        half::f16::from_f32(rng.range(0.0, 60.0)).to_f32()
                    } else {
                        table[(rng.next_u64() % table.len() as u64) as usize]
                    }
                })
                .collect(),
        );
        self
    }
}

/// R1, then R3 when `vignette` is set, then R4 gains, on the CPU over the whole frame, with R1's mask.
pub fn cpu_front(
    s: &RawScene,
    levels: &Levels,
    gains_rgb: [f32; 3],
    vig: Option<&VignetteProfile>,
) -> (Mosaic, ClipMask) {
    let (mut m, mask) = normalize::normalize(
        &s.data,
        s.width,
        s.height,
        s.cfa,
        levels,
        s.qstep.as_deref(),
    )
    .expect("normalize");
    if let Some(v) = vig {
        vignette::apply_vignette(&mut m, v);
    }
    gains::apply_gains(&mut m, gains_rgb);
    (m, mask)
}

/// R5 on the CPU with the whole-frame offsets: what `Chain::prepare` and the stepper make.
pub fn cpu_highlights(m: &mut Mosaic, mask: &ClipMask) -> highlights::ChromaOffsets {
    highlights::reconstruct(m, mask).expect("reconstruct")
}

/// One run on the device thread.
pub fn on_device<R: Send + 'static>(
    dev: &TestGpu,
    job: impl FnOnce(&GpuContext) -> Result<R, GpuError> + Send + 'static,
) -> R {
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| job(ctx))
        .expect("the device job")
}

/// The part `rect` of a whole-frame plane of `width` samples a row.
pub fn crop(plane: &[f32], width: usize, rect: ym_gpu::cpu::Rect) -> Vec<f32> {
    let mut out = Vec::with_capacity(rect.w * rect.h);
    for y in rect.y..rect.y1() {
        out.extend_from_slice(&plane[y * width + rect.x..y * width + rect.x1()]);
    }
    out
}

/// The largest `|a - b| / max(1, |b|)` and where: an absolute error for values up to white and a
/// relative one above it (R1 words beyond the white level reach values of 10 and more after the gains,
/// where one ulp is 1e-6).
pub fn compare_scaled(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = if x == y || (x.is_nan() && y.is_nan()) {
            0.0
        } else if x.is_nan() || y.is_nan() {
            f32::INFINITY
        } else {
            (x - y).abs() / y.abs().max(1.0)
        };
        if d.is_nan() || d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// One of the ten synthetic highlight scenes of `cpu_highlights.rs`: a light source of constant
/// chromaticity whose intensity falls off from a peak above the clip level, as a raw frame.
pub struct HighlightScene {
    pub name: &'static str,
    pub color: [f32; 3],
    pub peak: f32,
    pub disc: bool,
}

/// The ten scenes: five colours, a Gaussian glow and a hard-edged disc each.
pub fn highlight_scenes() -> Vec<HighlightScene> {
    let base = [
        ("warm lamp", [1.0, 0.62, 0.35], 2.6),
        ("blue sky", [0.45, 0.65, 1.0], 2.4),
        ("green LED", [0.35, 1.0, 0.45], 2.8),
        ("magenta sign", [1.0, 0.3, 0.9], 2.4),
        ("neutral lamp", [1.0, 1.0, 1.0], 3.0),
    ];
    let mut out = Vec::new();
    for disc in [false, true] {
        for (name, color, peak) in base {
            out.push(HighlightScene {
                name,
                color,
                peak,
                disc,
            });
        }
    }
    out
}

impl HighlightScene {
    /// The scene as a raw frame of `size x size` photosites: the balanced colour divided by the
    /// gains is what the sensor saw, clipped at white, quantised to 16-bit words around `levels`.
    pub fn raw(&self, size: usize, cfa: Cfa, levels: &Levels, gains: [f32; 3]) -> RawScene {
        let mut data = vec![0u16; size * size];
        for y in 0..size {
            for x in 0..size {
                let (dx, dy) = (
                    x as f32 + 0.5 - size as f32 / 2.0,
                    y as f32 + 0.5 - size as f32 / 2.0,
                );
                let falloff = if self.disc {
                    1.0 / (1.0 + (((dx * dx + dy * dy).sqrt() - 0.19 * size as f32) / 2.5).exp())
                } else {
                    (-(dx * dx + dy * dy) / (2.0 * (0.14 * size as f32).powi(2))).exp()
                };
                let i = self.peak * falloff + 0.03;
                let c = cfa.color_at(x, y).plane();
                let sensor = (self.color[c] * i / gains[c]).min(1.0);
                let p = (y & 1) * 2 + (x & 1);
                let v = levels.black[p] + sensor * (levels.white[p] - levels.black[p]);
                data[y * size + x] = v.round().clamp(0.0, 65535.0) as u16;
            }
        }
        RawScene {
            width: size,
            height: size,
            cfa,
            data,
            qstep: None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The reduced path's CPU reference (copies of ym-core's `sigma_plane`, `bin_mosaic`, `gain_plane`
// and `binned_sigma`; `crates/ym-core/tests/gpu_reduced.rs` pins the device against the originals)
// ---------------------------------------------------------------------------------------------

use ym_gpu::cpu::{CfaColor, RgbImage};
use ym_noise::model::NoiseModel;
use ym_noise::sigma_map::{HalfPlanes, QstepMap, SigmaParams, sigma_quad};

/// A noise model with visible shot noise, a read floor and an offset per position.
pub fn noise_model() -> NoiseModel {
    NoiseModel {
        a: 4e-4,
        b: 6e-8,
        delta: [2e-4, 0.0, 0.0, -1e-4],
        sigma_row: 2e-4,
        q: 1.0 / 16383.0,
    }
}

/// `ym-core`'s `sigma_plane` over the whole frame in one band: R1 and R4 (no vignetting), the four
/// positions of each cell, `sigma_quad` without a step plane.
pub fn ref_sigma_plane(
    s: &RawScene,
    levels: &Levels,
    g0: [f32; 3],
    model: &NoiseModel,
) -> Vec<f32> {
    let (hw, hh) = (s.width / 2, s.height / 2);
    let (mut m, _mask) =
        normalize::normalize(&s.data, s.width, s.height, s.cfa, levels, None).expect("normalize");
    gains::apply_gains(&mut m, g0);
    let g0_pos: [f32; 4] = std::array::from_fn(|p| g0[s.cfa.color_at(p & 1, p >> 1).plane()]);
    let sp = SigmaParams::new(model, g0_pos);
    let mut planes = vec![0.0f32; 4 * hw * hh];
    for p in 0..4usize {
        let (px, py) = (p & 1, p >> 1);
        for j in 0..hh {
            for i in 0..hw {
                planes[(p * hh + j) * hw + i] = m.data[(2 * j + py) * s.width + 2 * i + px];
            }
        }
    }
    let mut out = vec![0.0f32; hw * hh];
    sigma_quad(
        &sp,
        &HalfPlanes {
            data: &planes,
            width: hw,
            height: hh,
        },
        QstepMap::None,
        None,
        &mut out,
    )
    .expect("sigma");
    out
}

pub fn ref_sigma_params(s: &RawScene, g0: [f32; 3], model: &NoiseModel) -> SigmaParams {
    let g0_pos: [f32; 4] = std::array::from_fn(|p| g0[s.cfa.color_at(p & 1, p >> 1).plane()]);
    SigmaParams::new(model, g0_pos)
}

fn hat_integral(u: f64) -> f64 {
    if u <= -2.0 {
        0.0
    } else if u <= 0.0 {
        (u + 2.0) * (u + 2.0) / 4.0
    } else if u <= 2.0 {
        2.0 - (2.0 - u) * (2.0 - u) / 4.0
    } else {
        2.0
    }
}

pub fn ref_lattice_weights(k: usize, q: usize) -> Vec<f32> {
    (-1..=(k / 2) as isize)
        .map(|j| {
            let t = 2.0 * j as f64 + q as f64 + 0.5;
            ((hat_integral(k as f64 - t) - hat_integral(-t)) / k as f64) as f32
        })
        .collect()
}

fn same_parity(i: isize, n: usize) -> usize {
    let n = n as isize;
    if i < 0 {
        (i + 2).min(n - 1) as usize
    } else if i >= n {
        (i - 2).max(0) as usize
    } else {
        i as usize
    }
}

/// `ym-core`'s `bin_mosaic`, single-threaded.
pub fn ref_bin_mosaic(m: &Mosaic, k: usize) -> RgbImage {
    let (w, h) = (m.width, m.height);
    let (ow, oh) = (w / k, h / k);
    let mut img = RgbImage::new(ow, oh);
    let n = ow * oh;
    let parity = |color: CfaColor| {
        (0..2usize)
            .flat_map(|y| (0..2usize).map(move |x| (x, y)))
            .find(|&(x, y)| m.cfa.color_at(x, y) == color)
            .unwrap_or((0, 0))
    };
    let lattices = [CfaColor::Red, CfaColor::Blue].map(|c| {
        let (qx, qy) = parity(c);
        (
            qx,
            qy,
            ref_lattice_weights(k, qx),
            ref_lattice_weights(k, qy),
        )
    });
    let green_count = (k * k / 2) as f32;
    for oy in 0..oh {
        let mut sums = vec![0.0f32; ow];
        for y in oy * k..(oy + 1) * k {
            let row = &m.data[y * w..(y + 1) * w];
            let first = usize::from(m.cfa.color_at(0, y) != CfaColor::Green);
            for (ox, s) in sums.iter_mut().enumerate() {
                let block = &row[ox * k..(ox + 1) * k];
                *s += block[first..].iter().step_by(2).sum::<f32>();
            }
        }
        for (d, s) in img.data[n + oy * ow..n + (oy + 1) * ow]
            .iter_mut()
            .zip(&sums)
        {
            *d = s / green_count;
        }
        for (plane, (qx, qy, wx, wy)) in [0usize, 2].into_iter().zip(&lattices) {
            let mut acc = vec![0.0f32; ow];
            for (ii, wyi) in wy.iter().enumerate() {
                let y = (oy * k) as isize + 2 * (ii as isize - 1) + *qy as isize;
                let row = &m.data[same_parity(y, h) * w..(same_parity(y, h) + 1) * w];
                for (ox, a) in acc.iter_mut().enumerate() {
                    let x0 = (ox * k) as isize + *qx as isize;
                    let line: f32 = wx
                        .iter()
                        .enumerate()
                        .map(|(jj, wxj)| wxj * row[same_parity(x0 + 2 * (jj as isize - 1), w)])
                        .sum();
                    *a += wyi * line;
                }
            }
            img.data[plane * n + oy * ow..plane * n + (oy + 1) * ow].copy_from_slice(&acc);
        }
    }
    img
}

/// `ym-core`'s `gain_plane` at the pixels of a frame binned by `k`.
pub fn ref_gain_plane(
    v: &VignetteProfile,
    w: usize,
    h: usize,
    k: usize,
    ow: usize,
    oh: usize,
) -> Vec<f32> {
    let (wf, hf) = (w as f64, h as f64);
    let (cx, cy) = (wf * 0.5, hf * 0.5);
    let inv_hd = 1.0 / (0.5 * wf.hypot(hf));
    let kk = k as f64;
    let mut g = vec![1.0f32; ow * oh];
    for oy in 0..oh {
        let dy = (oy as f64 + 0.5) * kk - cy;
        for ox in 0..ow {
            let dx = (ox as f64 + 0.5) * kk - cx;
            g[oy * ow + ox] = v.gain_at(dx.hypot(dy) * inv_hd) as f32;
        }
    }
    g
}

/// The binning's noise factor as `ym-core` forms it.
pub fn ref_noise_factor(k: usize) -> f32 {
    let k = k.max(2);
    let w = [
        f64::from(ym_gpu::cpu::LUMA[0]),
        f64::from(ym_gpu::cpu::LUMA[1]),
        f64::from(ym_gpu::cpu::LUMA[2]),
    ];
    let per_axis: f64 = ref_lattice_weights(k, 0)
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum();
    let red_blue = per_axis * per_axis;
    let green = 2.0 / (k * k) as f64;
    (w[0] * w[0] * red_blue + w[1] * w[1] * green + w[2] * w[2] * red_blue).sqrt() as f32
}

/// `ym-core`'s `binned_sigma`.
pub fn ref_binned_sigma(
    sigma: &[f32],
    sw: usize,
    sh: usize,
    gains: Option<&[f32]>,
    k: usize,
    ow: usize,
    oh: usize,
) -> Vec<f32> {
    let step = (k / 2).max(1);
    let f = ref_noise_factor(k);
    let mut out = vec![0.0f32; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut s2 = 0.0f32;
            let mut n = 0.0f32;
            for cy in oy * step..(oy + 1) * step {
                let cy = cy.min(sh - 1);
                for cx in ox * step..(ox + 1) * step {
                    let s = sigma[cy * sw + cx.min(sw - 1)];
                    s2 += s * s;
                    n += 1.0;
                }
            }
            let g = gains.map_or(1.0, |g| g[oy * ow + ox]);
            out[oy * ow + ox] = (s2 / n).sqrt() * f * g;
        }
    }
    out
}
