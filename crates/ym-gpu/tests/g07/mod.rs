//! Helpers shared by the G-07 test files (`gpu_front.rs`, `gpu_lens.rs`, `gpu_reduced.rs`,
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
