//! The glow kernel `K_g` and the veiling kernel `K_v` (`docs/21` section 4, L3).
//!
//! # The glow kernel
//!
//! Per PSF wavelength `j` the lens gives a tail `(1 + r / (r0 chroma_j))^-power`, cut smoothly over the last
//! 15 % of its reach and normalised to unit energy; the camera's channel weights `w_c(l_j)` mix the eight
//! kernels into one kernel per channel, `K_c`, again of unit energy. The kernels are tabulated on a fine radial
//! grid ([`GLOW_TABLE_NODES`] nodes out to the reach).
//!
//! # The split
//!
//! The kernel is split by a window `s(r)` that is 1 inside the **core** (up to 70 % of the core radius), falls
//! to 0 at the core radius in a cosine, and is 0 beyond: the core part is `K_c s`, the tail part `K_c (1 - s)`.
//! The core is at most 0.0015 half-diagonals and at most 8 px of the full-resolution frame, and it is gathered
//! per tile at the tile's scale ([`CoreTaps`]); the tail part is smooth and goes to the tail plane.
//!
//! Both parts are normalised on the pixel grid they are used on: the core's taps sum to `a_g m_c` with the core's
//! mass `m_c = integral of K_c s`, and the tail kernel on the tail plane sums to `a_g (1 - m_c)`, so the two add
//! up to exactly the glow's fraction `a_g` of light at every scale.

use crate::consts::{
    GLOW_CORE_FLAT, GLOW_CORE_MAX_HD, GLOW_CORE_MAX_PX, GLOW_TABLE_NODES, GLOW_TAPER_START,
    PSF_WAVELENGTHS, VEIL_FLAT,
};
use crate::l3::profile::cosine_window;
use crate::plan::LensPlan;

/// The glow kernels of one lens on one camera.
#[derive(Clone, Debug)]
pub(crate) struct GlowProfiles {
    /// The core's radius in half-diagonals (0: no core).
    core_radius: f64,
    /// The tail's reach in half-diagonals.
    reach: f64,
    /// Spacing of the radial nodes.
    dr: f64,
    /// `K_c` at the nodes `i dr`, per channel: unit energy per half-diagonal squared.
    k: [Vec<f64>; 3],
    /// The mass of `K_c s` per channel.
    m_core: [f64; 3],
    /// The veiling kernel's radius.
    veil_radius: f64,
    /// The glow's fraction of the light at Flares 100, `a_g`.
    fraction: f64,
}

impl GlowProfiles {
    /// The kernels of `plan`'s lens on its camera.
    pub(crate) fn new(plan: &LensPlan) -> Self {
        let g = &plan.lens().flare.glow;
        let reach = g.reach;
        let f = plan.frame();
        let full_half_diag = 0.5 * ((f.width * f.width + f.height * f.height) as f64).sqrt();
        let core_radius = g
            .core_radius
            .min(GLOW_CORE_MAX_HD)
            .min(GLOW_CORE_MAX_PX / full_half_diag)
            .max(0.0);
        let n = GLOW_TABLE_NODES;
        let dr = reach / n as f64;
        let two_pi = 2.0 * std::f64::consts::PI;

        // One unit-energy kernel per PSF wavelength.
        let mut per_wavelength: Vec<Vec<f64>> = Vec::with_capacity(PSF_WAVELENGTHS);
        for j in 0..PSF_WAVELENGTHS {
            let r0 = g.r0 * g.chroma[j];
            let mut table: Vec<f64> = (0..=n)
                .map(|i| {
                    let r = i as f64 * dr;
                    libm::pow(1.0 + r / r0, -g.power) * cosine_window(r / reach, GLOW_TAPER_START)
                })
                .collect();
            let mut mass = 0.0;
            for (i, v) in table.iter().enumerate() {
                let w = if i == n { 0.5 } else { 1.0 };
                mass += w * two_pi * (i as f64 * dr) * v * dr;
            }
            for v in &mut table {
                *v /= mass;
            }
            per_wavelength.push(table);
        }
        // Mixed into the camera's channels.
        let weights = plan.camera().weights();
        let k: [Vec<f64>; 3] = std::array::from_fn(|c| {
            (0..=n)
                .map(|i| {
                    let mut s = 0.0;
                    for j in 0..PSF_WAVELENGTHS {
                        s += weights[c][j] * per_wavelength[j][i];
                    }
                    s
                })
                .collect()
        });
        let mut profiles = GlowProfiles {
            core_radius,
            reach,
            dr,
            k,
            m_core: [0.0; 3],
            veil_radius: plan.lens().flare.veil.radius,
            fraction: g.fraction,
        };
        for c in 0..3 {
            let mut mass = 0.0;
            for i in 0..=n {
                let r = i as f64 * dr;
                if r >= core_radius {
                    break;
                }
                mass += two_pi * r * profiles.k[c][i] * profiles.window(r) * dr;
            }
            profiles.m_core[c] = mass.clamp(0.0, 1.0);
        }
        profiles
    }

    /// The tail's reach in half-diagonals.
    pub(crate) fn reach(&self) -> f64 {
        self.reach
    }

    /// The core's mass per channel: the share of the glow gathered per tile.
    pub(crate) fn core_mass(&self) -> [f64; 3] {
        self.m_core
    }

    /// The core window `s(r)`.
    #[inline]
    pub(crate) fn window(&self, r: f64) -> f64 {
        if self.core_radius <= 0.0 {
            return 0.0;
        }
        cosine_window(r / self.core_radius, GLOW_CORE_FLAT)
    }

    /// `K_c(r)`: the channel's kernel (0 beyond the reach).
    #[inline]
    pub(crate) fn kernel(&self, c: usize, r: f64) -> f64 {
        let x = r / self.dr;
        let n = GLOW_TABLE_NODES;
        if x >= n as f64 {
            return 0.0;
        }
        let i = x as usize;
        let t = x - i as f64;
        let k = &self.k[c];
        k[i] + t * (k[i + 1] - k[i])
    }

    /// The tail part `K_c (1 - s)` at radius `r`.
    #[inline]
    pub(crate) fn tail(&self, c: usize, r: f64) -> f64 {
        self.kernel(c, r) * (1.0 - self.window(r))
    }

    /// The veiling kernel at radius `r` (before normalisation): flat, with a cosine edge.
    #[inline]
    pub(crate) fn veil(&self, r: f64) -> f64 {
        cosine_window(r / self.veil_radius, VEIL_FLAT)
    }

    /// The veiling kernel's radius in half-diagonals.
    pub(crate) fn veil_radius(&self) -> f64 {
        self.veil_radius
    }

    /// The core's taps for a frame whose half-diagonal is `half_diag` pixels.
    pub(crate) fn core_taps(&self, half_diag: f64) -> CoreTaps {
        let r_px = self.core_radius * half_diag;
        let empty = self.core_radius <= 0.0
            || self.fraction <= 0.0
            || self.m_core.iter().all(|m| *m <= 0.0);
        if empty {
            return CoreTaps {
                radius: 0,
                weights: [Vec::new(), Vec::new(), Vec::new()],
                lanes: Vec::new(),
            };
        }
        let radius = (libm::floor(r_px) as usize).min(GLOW_CORE_MAX_PX as usize);
        let side = 2 * radius + 1;
        let weights: [Vec<f64>; 3] = std::array::from_fn(|c| {
            let mut w = vec![0.0; side * side];
            let mut sum = 0.0;
            for (q, v) in w.iter_mut().enumerate() {
                let dx = (q % side) as f64 - radius as f64;
                let dy = (q / side) as f64 - radius as f64;
                let r = (dx * dx + dy * dy).sqrt() / half_diag;
                *v = self.kernel(c, r) * self.window(r);
                sum += *v;
            }
            if sum > 0.0 {
                let scale = self.fraction * self.m_core[c] / sum;
                for v in &mut w {
                    *v *= scale;
                }
            }
            w
        });
        let lanes = (0..weights[0].len())
            .map(|q| {
                [
                    weights[0][q] as f32,
                    weights[1][q] as f32,
                    weights[2][q] as f32,
                    0.0,
                ]
            })
            .collect();
        CoreTaps {
            radius,
            weights,
            lanes,
        }
    }
}

/// The glow core of one frame size: `(2 radius + 1)^2` weights per channel, row-major, summing to the core's
/// mass times the glow's fraction of light, `a_g m_c`.
#[derive(Clone, Debug)]
pub(crate) struct CoreTaps {
    /// The taps reach this many pixels from the centre.
    pub(crate) radius: usize,
    /// The weights per channel (empty when the lens has no core).
    pub(crate) weights: [Vec<f64>; 3],
    /// The same weights as one four-lane vector per tap (red, green, blue and a zero), for the gather.
    pub(crate) lanes: Vec<[f32; 4]>,
}

impl CoreTaps {
    /// True when there is nothing to gather.
    pub(crate) fn is_empty(&self) -> bool {
        self.weights[0].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amounts::{LensAmounts, LensKind};
    use crate::plan::{Axis, FrameInfo};
    use crate::spectral::CameraResponse;
    use crate::test_support::write_synthetic_lenses;

    fn plan(w: usize, h: usize) -> LensPlan {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_lenses(dir.path()).unwrap();
        let lib = crate::data::LensLibrary::from_dir(dir.path()).unwrap();
        let amounts = LensAmounts {
            kind: LensKind::Classic,
            flares: 100.0,
            edges: 0.0,
            anamorphic: 0.0,
        };
        let frame = FrameInfo {
            width: w,
            height: h,
            axis: Axis::X,
            f_number: None,
            clip: [1.0; 3],
        };
        LensPlan::new(&lib, &amounts, frame, &CameraResponse::neutral()).unwrap()
    }

    #[test]
    fn the_kernels_have_unit_energy_and_the_core_is_a_small_part() {
        let p = plan(6000, 4000);
        let g = GlowProfiles::new(&p);
        let two_pi = 2.0 * std::f64::consts::PI;
        for c in 0..3 {
            let mut mass = 0.0;
            for i in 0..=GLOW_TABLE_NODES {
                let r = i as f64 * g.dr;
                let w = if i == GLOW_TABLE_NODES { 0.5 } else { 1.0 };
                mass += w * two_pi * r * g.k[c][i] * g.dr;
            }
            assert!((mass - 1.0).abs() < 1e-9, "channel {c}: {mass}");
            assert!(g.m_core[c] > 0.0 && g.m_core[c] < 0.1, "{:?}", g.m_core);
        }
        // 0.0015 half-diagonals is 5.4 px of a 3605 px half-diagonal, below the 8 px limit.
        assert!((g.core_radius - 0.0015).abs() < 1e-15);
    }

    #[test]
    fn core_taps_sum_to_the_core_mass_at_every_scale() {
        let p = plan(6000, 4000);
        let g = GlowProfiles::new(&p);
        let full = 0.5 * (6000.0f64 * 6000.0 + 4000.0 * 4000.0).sqrt();
        for k in [1.0, 2.0, 4.0, 16.0, 64.0] {
            let taps = g.core_taps(full / k);
            for c in 0..3 {
                let sum: f64 = taps.weights[c].iter().sum();
                let want = g.fraction * g.m_core[c];
                assert!((sum - want).abs() < 1e-15, "k={k} c={c}");
            }
            assert!(taps.radius <= 8);
        }
        // A tiny frame has a one-tap core.
        let t = g.core_taps(100.0);
        assert_eq!(t.radius, 0);
        assert_eq!(t.weights[0].len(), 1);
    }

    #[test]
    fn the_window_is_one_inside_and_zero_outside() {
        let p = plan(600, 400);
        let g = GlowProfiles::new(&p);
        assert_eq!(g.window(0.0), 1.0);
        assert_eq!(g.window(g.core_radius), 0.0);
        assert_eq!(g.window(2.0 * g.core_radius), 0.0);
        let mid = g.window(0.85 * g.core_radius);
        assert!(mid > 0.0 && mid < 1.0);
    }
}
