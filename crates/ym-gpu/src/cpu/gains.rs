//! R4 initial white-balance gains `g0` (`docs/05-raw-processing.md` section R4).
//!
//! `g0 = 1 / w_scene_initial` with green = 1, from the illuminant estimate at open; if the
//! estimate fails the integrator falls back to the as-shot gains from SR2 0x7313. Choosing
//! between them is the integrator's job. This module turns either source into `g0` and applies it
//! to the mosaic before R5 and R6.

use rayon::prelude::*;

use super::{CfaColor, CpuError, Mosaic};

/// Gains `[red, green, blue]` from as-shot white-balance levels in RGGB order (SR2 0x7313, or the
/// regrouped 0x7303): `[R / G, 1, B / G]` with `G` the mean of the two green levels.
///
/// Sony stores these as multipliers (green is typically 1024), so no inversion is applied.
pub fn g0_from_wb_levels(levels_rggb: [f32; 4]) -> Result<[f32; 3], CpuError> {
    let [r, g1, g2, b] = levels_rggb;
    if !levels_rggb.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(CpuError::BadWhiteBalance(levels_rggb));
    }
    let g = 0.5 * (g1 + g2);
    Ok([r / g, 1.0, b / g])
}

/// Gains `1 / w` normalised to green = 1, from the scene white `w` (the camera RGB of a neutral
/// object under the estimated light, any scale).
pub fn g0_from_scene_white(w: [f64; 3]) -> Result<[f32; 3], CpuError> {
    if !w.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(CpuError::BadWhiteBalance([
            w[0] as f32,
            w[1] as f32,
            w[1] as f32,
            w[2] as f32,
        ]));
    }
    Ok([(w[1] / w[0]) as f32, 1.0, (w[1] / w[2]) as f32])
}

/// Multiplies every photosite by its channel's gain, in place. Pointwise (exact under tiling).
///
/// The clip mask is unchanged: a clipped photosite now sits at `gain` instead of 1.0, and R5 uses
/// the mask, not a threshold, to find it.
pub fn apply_gains(m: &mut Mosaic, gains: [f32; 3]) {
    let w = m.width.max(1);
    let cfa = m.cfa;
    m.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let g = [
            gains[cfa.color_at(0, y).plane()],
            gains[cfa.color_at(1, y).plane()],
        ];
        for (x, v) in row.iter_mut().enumerate() {
            *v *= g[x & 1];
        }
    });
}

/// The clip level of channel `c` after [`apply_gains`] (before vignetting): the gain itself.
pub fn clip_level(gains: [f32; 3], c: CfaColor) -> f32 {
    gains[c.plane()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cfa;

    #[test]
    fn wb_levels_become_green_normalised_gains() {
        let g = g0_from_wb_levels([2048.0, 1024.0, 1024.0, 1536.0]).unwrap();
        assert_eq!(g, [2.0, 1.0, 1.5]);
        let g = g0_from_wb_levels([2000.0, 1000.0, 1100.0, 1575.0]).unwrap();
        assert!((g[0] - 2000.0 / 1050.0).abs() < 1e-6 && g[1] == 1.0 && (g[2] - 1.5).abs() < 1e-6);
        assert!(g0_from_wb_levels([0.0, 1.0, 1.0, 1.0]).is_err());
        assert!(g0_from_wb_levels([f32::NAN, 1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn scene_white_inverts() {
        // A neutral under tungsten reads R:G:B = 1.0 : 0.5 : 0.25 -> gains 0.5, 1, 2.
        let g = g0_from_scene_white([1.0, 0.5, 0.25]).unwrap();
        assert_eq!(g, [0.5, 1.0, 2.0]);
    }

    #[test]
    fn applies_per_cfa_channel_for_every_phase() {
        for cfa in Cfa::ALL {
            let mut m = Mosaic::new(4, 4, cfa);
            m.data.fill(1.0);
            apply_gains(&mut m, [2.0, 1.0, 3.0]);
            for y in 0..4 {
                for x in 0..4 {
                    let want = [2.0, 1.0, 3.0][cfa.color_at(x, y).plane()];
                    assert_eq!(m.get(x, y), want);
                }
            }
        }
    }
}
