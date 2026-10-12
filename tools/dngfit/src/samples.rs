//! Test colours: scene-linear AWG4, the pixels a DNG holds (`docs/09`, "render a set of synthetic test images (neutral
//! ramp, hue/saturation sweeps, our golden photos)").
//!
//! All sets avoid negative AWG4 components (the DNG writer clips them to 0, so the file would not hold the colour that
//! was asked for) and are made from a seed, so a run is repeatable.

use std::path::Path;

use anyhow::{Context, Result};
use golden::scene::{Sidecar, list_sidecars};
use ym_color::consts::{Vec3, linalg};
use ym_color::deltae::{d65_white_xyz, lab_to_xyz};
use ym_color::gamut;

/// A small deterministic generator (xorshift64*), so the fit needs no random-number crate.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator from a seed (any value; zero is replaced).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// The next number in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// A standard normal number (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// A neutral ramp: `n` greys from `x_lo` to `x_hi`, geometrically spaced.
#[must_use]
pub fn neutral_ramp(n: usize, x_lo: f64, x_hi: f64) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1).max(1) as f64;
            let x = (x_lo * (x_hi / x_lo).powf(t)) as f32;
            [x; 3]
        })
        .collect()
}

/// The scene-linear AWG4 colour of CIE L*a*b* (D65, white at scene value `white`), or `None` when a component of
/// AWG4 would be negative.
#[must_use]
pub fn awg4_of_lab(lab: Vec3, white: f64) -> Option<Vec3> {
    let xyz = lab_to_xyz(lab, d65_white_xyz()).map(|c| c * white);
    let rgb = linalg::mul_vec(&gamut::matrices().xyz_to_awg4, xyz);
    rgb.iter().all(|&c| c >= 0.0).then_some(rgb)
}

/// Hue and saturation sweeps: `hues` hues, `chromas` chroma steps (CIELAB C* from 0 to `max_chroma`) at each lightness
/// of `lightness` (L*), scene white at 1.0. Colours that would need a negative AWG4 component are left out.
#[must_use]
pub fn hue_chroma_sweep(
    hues: usize,
    chromas: usize,
    lightness: &[f64],
    max_chroma: f64,
) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for &l in lightness {
        for hi in 0..hues {
            let h = (hi as f64 + 0.5) / hues as f64 * std::f64::consts::TAU;
            for ci in 0..chromas {
                let c = max_chroma * (ci as f64 + 0.5) / chromas as f64;
                if let Some(rgb) = awg4_of_lab([l, c * h.cos(), c * h.sin()], 1.0) {
                    out.push(rgb.map(|v| v as f32));
                }
            }
        }
    }
    out
}

/// Random colours of the kind photographs have: lightness spread over the scene's range (L* 3 to 100, and a tail of
/// highlights up to 8 times the white), chroma from a half-normal with a scale of 30 and a tail, any hue.
#[must_use]
pub fn random_colors(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // log-uniform luminance factor: -7 to +3 stops around the white
        let stops = -7.0 + 10.0 * rng.uniform();
        let white = stops.exp2();
        let l = 20.0 + 80.0 * rng.uniform();
        let chroma = if rng.uniform() < 0.15 {
            90.0 * rng.uniform()
        } else {
            (30.0 * rng.normal()).abs()
        };
        let h = std::f64::consts::TAU * rng.uniform();
        if let Some(rgb) = awg4_of_lab([l, chroma * h.cos(), chroma * h.sin()], white) {
            out.push(rgb.map(|v| v as f32));
        }
    }
    out
}

/// The colours of the synthetic golden scenes' ground truth: for every scene sidecar the ALEXA-estimate AWG4 value of
/// each chart patch, at exposures of `ev_steps` stops around the reference (reflectance-referred: a perfect white is 1).
pub fn golden_patch_colors(scenes_dir: &Path, ev_steps: &[f64]) -> Result<Vec<[f32; 3]>> {
    let mut out = Vec::new();
    for path in list_sidecars(scenes_dir)? {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let sidecar: Sidecar =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let truth = sidecar.truth()?;
        for p in truth
            .probes
            .iter()
            .filter(|p| p.group == "chart" && !p.clipped_channels.iter().any(|&c| c))
        {
            for &ev in ev_steps {
                let g = ev.exp2();
                let c = p.alexa.awg4;
                if c.iter().all(|&v| v >= 0.0) {
                    out.push(c.map(|v| (v * g) as f32));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_repeatable_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            let (x, y) = (a.uniform(), b.uniform());
            assert_eq!(x, y);
            assert!((0.0..1.0).contains(&x));
        }
        let mut r = Rng::new(1);
        let mean = (0..4000).map(|_| r.normal()).sum::<f64>() / 4000.0;
        assert!(mean.abs() < 0.1, "{mean}");
    }

    #[test]
    fn sets_have_the_requested_shape_and_no_negative_components() {
        let ramp = neutral_ramp(5, 0.001, 16.0);
        assert_eq!(ramp.len(), 5);
        assert!((ramp[0][0] - 0.001).abs() < 1e-6 && (ramp[4][2] - 16.0).abs() < 1e-3);
        let sweep = hue_chroma_sweep(24, 6, &[30.0, 60.0, 85.0], 80.0);
        assert!(sweep.len() > 200 && sweep.len() <= 24 * 6 * 3);
        assert!(sweep.iter().flatten().all(|&c| c >= 0.0));
        let r = random_colors(500, 3);
        assert_eq!(r.len(), 500);
        assert_eq!(r, random_colors(500, 3));
        assert_ne!(r, random_colors(500, 4));
        assert!(r.iter().flatten().all(|&c| c >= 0.0 && c.is_finite()));
    }

    #[test]
    fn lab_neutrals_are_neutral_in_awg4() {
        let c = awg4_of_lab([50.0, 0.0, 0.0], 1.0).unwrap();
        assert!((c[0] - c[1]).abs() < 1e-9 && (c[1] - c[2]).abs() < 1e-9);
        assert!(c[0] > 0.17 && c[0] < 0.19, "L* 50 is about 18 %: {}", c[0]);
        // a very saturated blue is outside AWG4's positive octant
        assert!(awg4_of_lab([30.0, 40.0, -120.0], 1.0).is_none());
    }
}
