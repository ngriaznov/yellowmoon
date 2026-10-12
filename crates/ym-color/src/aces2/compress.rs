// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Chroma compression and gamut compression of the ACES 2 output transform.
//!
//! Port of the `chroma_compress_*`, `compress_gamut`, `gamut_compress_fwd` and helper functions
//! in `Lib.Academy.OutputTransform.ctl`, forward direction only.

use super::cam::HUE_LIMIT;
use super::matrix::V3;
use super::params::{Aces2Params, BASE_INDEX, TABLE_SIZE, UPPER_WRAP_INDEX};

// Chroma compression constants.
pub(super) const CHROMA_COMPRESS: f64 = 2.4;
pub(super) const CHROMA_COMPRESS_FACT: f64 = 3.3;
pub(super) const CHROMA_EXPAND: f64 = 1.3;
pub(super) const CHROMA_EXPAND_FACT: f64 = 0.69;
pub(super) const CHROMA_EXPAND_THR: f64 = 0.5;

// Gamut compression constants.
pub(super) const SMOOTH_CUSPS: f64 = 0.12;
pub(super) const SMOOTH_M: f64 = 0.27;
pub(super) const CUSP_MID_BLEND: f64 = 1.3;
pub(super) const FOCUS_GAIN_BLEND: f64 = 0.3;
pub(super) const FOCUS_DISTANCE: f64 = 1.35;
pub(super) const FOCUS_DISTANCE_SCALING: f64 = 1.75;
pub(super) const COMPRESSION_THRESHOLD: f64 = 0.75;

#[inline]
pub(super) fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// `reach_M_from_table`: linear interpolation of the reach table at hue `h` (degrees, 0..=360).
#[inline]
pub(super) fn reach_m_from_table(h: f64, table: &[f64]) -> f64 {
    // The table is uniform: one entry per degree, entry `BASE_INDEX` is hue 0.
    let base = (h / HUE_LIMIT * TABLE_SIZE as f64) as usize;
    let t = h - base as f64;
    let i_lo = base + BASE_INDEX;
    lerp(table[i_lo], table[i_lo + 1], t)
}

/// `toe`: remaps `x` between 0 and `limit`; `k1` and `k2` change the size and shape of the toe.
#[inline]
pub(super) fn toe(x: f64, limit: f64, k1_in: f64, k2_in: f64) -> f64 {
    if x > limit {
        return x;
    }
    let k2 = k2_in.max(0.001);
    let k1 = (k1_in * k1_in + k2 * k2).sqrt();
    let k3 = (limit + k1) / (limit + k2);

    let minus_b = k3 * x - k1;
    let minus_c = k2 * k3 * x;
    0.5 * (minus_b + (minus_b * minus_b + 4.0 * minus_c).sqrt())
}

/// `chroma_compress_norm`: hue dependent normalisation of M (a fit of the display cusp M).
#[inline]
pub(super) fn chroma_compress_norm(h_deg: f64, chroma_compress_scale: f64) -> f64 {
    let hr = h_deg.to_radians();
    let a = hr.cos();
    let b = hr.sin();
    let cos_hr2 = a * a - b * b;
    let sin_hr2 = 2.0 * a * b;
    let cos_hr3 = 4.0 * a * a * a - 3.0 * a;
    let sin_hr3 = 3.0 * b - 4.0 * b * b * b;

    let m = 11.34072 * a
        + 16.46899 * cos_hr2
        + 7.88380 * cos_hr3
        + 14.66441 * b
        + -6.37224 * sin_hr2
        + 9.19364 * sin_hr3
        + 77.12896;

    m * chroma_compress_scale
}

/// `compute_compression_vector_slope`.
#[inline]
pub(super) fn compute_compression_vector_slope(
    intersect_j: f64,
    focus_j: f64,
    limit_j_max: f64,
    slope_gain: f64,
) -> f64 {
    let direction_scalar = if intersect_j < focus_j {
        intersect_j
    } else {
        limit_j_max - intersect_j
    };
    direction_scalar * (intersect_j - focus_j) / (focus_j * slope_gain)
}

/// `solve_J_intersect`: J where the compression line through (J, M) meets the achromatic axis.
#[inline]
pub(super) fn solve_j_intersect(j: f64, m: f64, focus_j: f64, max_j: f64, slope_gain: f64) -> f64 {
    let m_scaled = m / slope_gain;
    let a = m_scaled / focus_j;

    // The discriminants are non-negative for J <= max_j; `max(0)` only absorbs rounding.
    if j < focus_j {
        let b = 1.0 - m_scaled;
        let c = -j;
        let det = b * b - 4.0 * a * c;
        let root = det.max(0.0).sqrt();
        -2.0 * c / (b + root)
    } else {
        let b = -(1.0 + m_scaled + max_j * a);
        let c = max_j * m_scaled + j;
        let det = b * b - 4.0 * a * c;
        let root = det.max(0.0).sqrt();
        -2.0 * c / (b - root)
    }
}

/// `smin_scaled`: smooth minimum about the scaled reference, based upon a cubic polynomial.
#[inline]
pub(super) fn smin_scaled(a: f64, b: f64, scale_reference: f64) -> f64 {
    let s_scaled = SMOOTH_CUSPS * scale_reference;
    let h = (s_scaled - (a - b).abs()).max(0.0) / s_scaled;
    a.min(b) - h * h * h * s_scaled * (1.0 / 6.0)
}

/// `estimate_line_and_boundary_intersection_M`.
#[inline]
pub(super) fn estimate_line_and_boundary_intersection_m(
    j_axis_intersect: f64,
    slope: f64,
    inv_gamma: f64,
    j_max: f64,
    m_max: f64,
    j_intersection_reference: f64,
) -> f64 {
    // Line defined by     J = slope * x + J_axis_intersect
    // Boundary defined by J = J_max * (x / M_max) ^ (1/inv_gamma)
    // Approximate as we do not want to iteratively solve intersection of a straight line and an
    // exponential. Calculate a shifted intersection from the original intersection using the
    // inverse of the exponential and the provided reference.
    //
    // The ratio is clamped at 0: where the compression line meets the axis at (or, by rounding, a
    // hair beyond) the top of the gamut the reference takes `pow` of a tiny negative number and
    // gets NaN. The limit is a boundary M of 0, which this gives.
    let normalised_j = (j_axis_intersect / j_intersection_reference).max(0.0);
    let shifted_intersection = j_intersection_reference * normalised_j.powf(inv_gamma);

    // Now find the M intersection of two lines: the line from the origin to (J_max, M_max) and the
    // line from the shifted intersection with the given slope.
    shifted_intersection * m_max / (j_max - slope * m_max)
}

/// `find_gamut_boundary_intersection`.
#[inline]
pub(super) fn find_gamut_boundary_intersection(
    jm_cusp: [f64; 2],
    j_max: f64,
    gamma_top_inv: f64,
    gamma_bottom_inv: f64,
    j_intersect_source: f64,
    slope: f64,
    j_intersect_cusp: f64,
) -> f64 {
    let m_boundary_lower = estimate_line_and_boundary_intersection_m(
        j_intersect_source,
        slope,
        gamma_bottom_inv,
        jm_cusp[0],
        jm_cusp[1],
        j_intersect_cusp,
    );

    // The upper hull is flipped and thus 'zeroed' at J_max. Also note we negate the slope.
    let f_j_intersect_cusp = j_max - j_intersect_cusp;
    let f_j_intersect_source = j_max - j_intersect_source;
    let f_jm_cusp_j = j_max - jm_cusp[0];
    let m_boundary_upper = estimate_line_and_boundary_intersection_m(
        f_j_intersect_source,
        -slope,
        gamma_top_inv,
        f_jm_cusp_j,
        jm_cusp[1],
        f_j_intersect_cusp,
    );

    // Smooth minimum between the two calculated values for the M component.
    smin_scaled(m_boundary_lower, m_boundary_upper, jm_cusp[1])
}

/// `get_focus_gain`.
#[inline]
pub(super) fn get_focus_gain(
    j: f64,
    analytical_threshold: f64,
    limit_j_max: f64,
    focus_dist: f64,
) -> f64 {
    let mut gain = limit_j_max * focus_dist;

    if j > analytical_threshold {
        // Approximate inverse required above threshold due to the introduction of J in the calculation.
        let mut gain_adjustment =
            ((limit_j_max - analytical_threshold) / (limit_j_max - j).max(0.0001)).log10();
        gain_adjustment = gain_adjustment * gain_adjustment + 1.0;
        gain *= gain_adjustment;
    }

    gain
}

/// `compute_focus_J`.
#[inline]
pub(super) fn compute_focus_j(cusp_j: f64, mid_j: f64, limit_j_max: f64) -> f64 {
    lerp(
        cusp_j,
        mid_j,
        (CUSP_MID_BLEND - (cusp_j / limit_j_max)).min(1.0),
    )
}

/// `remap_M` (forward): Reinhard-style compression of M above a threshold below the gamut boundary.
#[inline]
fn remap_m(m: f64, gamut_boundary_m: f64, reach_boundary_m: f64) -> f64 {
    let boundary_ratio = gamut_boundary_m / reach_boundary_m;
    let proportion = boundary_ratio.max(COMPRESSION_THRESHOLD);
    let threshold = proportion * gamut_boundary_m;

    if m <= threshold || proportion >= 1.0 {
        return m;
    }

    // Translate to place threshold at zero.
    let m_offset = m - threshold;
    let gamut_offset = gamut_boundary_m - threshold;
    let reach_offset = reach_boundary_m - threshold;

    let scale = reach_offset / ((reach_offset / gamut_offset) - 1.0);
    let nd = m_offset / scale;

    // Shift result back to absolute (forward Reinhard remap).
    threshold + scale * nd / (1.0 + nd)
}

/// Hue dependent gamut parameters (`HueDependentGamutParams`).
#[derive(Clone, Copy, Debug)]
pub(super) struct HueDependentGamutParams {
    pub jm_cusp: [f64; 2],
    pub gamma_bottom_inv: f64,
    pub gamma_top_inv: f64,
    pub focus_j: f64,
    pub analytical_threshold: f64,
}

impl Aces2Params {
    /// `lookup_hue_interval`: upper index of the hue table interval containing `h`.
    ///
    /// The hues are almost uniform, so the binary search starts in a window derived from
    /// `hue_linearity_search_range`.
    pub(super) fn lookup_hue_interval(&self, h: f64) -> usize {
        let i0 = BASE_INDEX + (h / HUE_LIMIT * TABLE_SIZE as f64) as usize;
        let mut i = i0;
        let mut i_lo = (i0 as i64 + self.hue_linearity_search_range[0] as i64).max(0) as usize;
        let mut i_hi = (i0 as i64 + self.hue_linearity_search_range[1] as i64)
            .min(UPPER_WRAP_INDEX as i64) as usize;

        while i_lo + 1 < i_hi {
            if h > self.hue_table[i] {
                i_lo = i;
            } else {
                i_hi = i;
            }
            i = (i_lo + i_hi) / 2;
        }

        i_hi.max(1)
    }

    /// `init_HueDependentGamutParams`.
    pub(super) fn hue_dependent_gamut_params(&self, hue: f64) -> HueDependentGamutParams {
        let i_hi = self.lookup_hue_interval(hue);
        let i_lo = i_hi - 1;
        let t = (hue - self.hue_table[i_lo]) / (self.hue_table[i_hi] - self.hue_table[i_lo]);

        let cusp_j = lerp(self.cusp_j_table[i_lo], self.cusp_j_table[i_hi], t);
        let cusp_m = lerp(self.cusp_m_table[i_lo], self.cusp_m_table[i_hi], t);

        HueDependentGamutParams {
            jm_cusp: [cusp_j, cusp_m],
            gamma_bottom_inv: self.lower_hull_gamma_inv,
            gamma_top_inv: lerp(
                self.upper_hull_gamma_inv_table[i_lo],
                self.upper_hull_gamma_inv_table[i_hi],
                t,
            ),
            focus_j: compute_focus_j(cusp_j, self.mid_j, self.limit_j_max),
            analytical_threshold: lerp(cusp_j, self.limit_j_max, FOCUS_GAIN_BLEND),
        }
    }

    /// `chroma_compress_fwd`: compress M given the tone-mapped lightness `tonemapped_j`.
    ///
    /// Returns (tone-mapped J, compressed M, h).
    pub(super) fn chroma_compress_fwd(&self, jmh: V3, tonemapped_j: f64) -> V3 {
        let (j, m, h) = (jmh[0], jmh[1], jmh[2]);

        let mut m_compr = m;

        if m != 0.0 {
            let n_j = tonemapped_j / self.limit_j_max;
            let sn_j = (1.0 - n_j).max(0.0);
            let m_norm = chroma_compress_norm(h, self.chroma_compress_scale);
            let limit = n_j.powf(self.model_gamma_inv) * reach_m_from_table(h, &self.reach_m_table)
                / m_norm;

            let toe_limit = limit - 0.001;
            let toe_sn_j_sat = sn_j * self.sat;
            let toe_sqrt_n_j_sat_thr = (n_j * n_j + self.sat_thr).sqrt();
            let toe_n_j_compr = n_j * self.compr;

            // Rescale M with the tone-scaled J to get M into the same range as J after the tone
            // scale. The rescaling uses the Hellwig 2022 model gamma to keep the M/J ratio correct
            // (keeping the chromaticities constant).
            m_compr = m * (tonemapped_j / j).powf(self.model_gamma_inv);

            // Normalise M with the rendering space cusp M.
            m_compr /= m_norm;

            // Expand the colourfulness by running the toe function in reverse. The goal is to
            // expand less saturated colours less and more saturated colours more. The expansion
            // increases saturation in the shadows and mid-tones but not in the highlights.
            m_compr = limit
                - toe(
                    limit - m_compr,
                    toe_limit,
                    toe_sn_j_sat,
                    toe_sqrt_n_j_sat_thr,
                );

            // Compress the colourfulness. The goal is to compress less saturated colours more and
            // more saturated colours less, especially in the highlights.
            m_compr = toe(m_compr, limit, toe_n_j_compr, sn_j);

            // Denormalise.
            m_compr *= m_norm;
        }

        [tonemapped_j, m_compr, h]
    }

    /// `compress_gamut` (forward): moves (J, M) along the compression vector into the gamut.
    fn compress_gamut(&self, jmh: V3, jx: f64, hdp: &HueDependentGamutParams) -> V3 {
        let (j, m, h) = (jmh[0], jmh[1], jmh[2]);

        let slope_gain = get_focus_gain(
            jx,
            hdp.analytical_threshold,
            self.limit_j_max,
            self.focus_dist,
        );
        let j_intersect_source = solve_j_intersect(j, m, hdp.focus_j, self.limit_j_max, slope_gain);
        let gamut_slope = compute_compression_vector_slope(
            j_intersect_source,
            hdp.focus_j,
            self.limit_j_max,
            slope_gain,
        );

        let j_intersect_cusp = solve_j_intersect(
            hdp.jm_cusp[0],
            hdp.jm_cusp[1],
            hdp.focus_j,
            self.limit_j_max,
            slope_gain,
        );

        let gamut_boundary_m = find_gamut_boundary_intersection(
            hdp.jm_cusp,
            self.limit_j_max,
            hdp.gamma_top_inv,
            hdp.gamma_bottom_inv,
            j_intersect_source,
            gamut_slope,
            j_intersect_cusp,
        );

        // Also catches NaN, which the reference lets through (see the note in
        // `estimate_line_and_boundary_intersection_m`).
        if gamut_boundary_m.is_nan() || gamut_boundary_m <= 0.0 {
            return [j, 0.0, h];
        }

        let reach_max_m = reach_m_from_table(h, &self.reach_m_table);

        let reach_boundary_m = estimate_line_and_boundary_intersection_m(
            j_intersect_source,
            gamut_slope,
            self.model_gamma_inv,
            self.limit_j_max,
            reach_max_m,
            self.limit_j_max,
        );

        let remapped_m = remap_m(m, gamut_boundary_m, reach_boundary_m);

        [j_intersect_source + remapped_m * gamut_slope, remapped_m, h]
    }

    /// `gamut_compress_fwd`.
    pub(super) fn gamut_compress_fwd(&self, jmh: V3) -> V3 {
        let (j, m, h) = (jmh[0], jmh[1], jmh[2]);

        if j <= 0.0 {
            return [0.0, 0.0, h];
        }

        if m < 0.0 || j > self.limit_j_max {
            return [j, 0.0, h];
        }

        let hdp = self.hue_dependent_gamut_params(h);
        self.compress_gamut(jmh, j, &hdp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aces2::{Aces2Params, Primaries};

    /// Reference interval search over the whole table: smallest index with `h <= hue[i]`.
    fn brute_force_interval(p: &Aces2Params, h: f64) -> usize {
        (1..=UPPER_WRAP_INDEX)
            .find(|&i| h <= p.hue_table[i])
            .unwrap_or(UPPER_WRAP_INDEX)
    }

    #[test]
    fn narrowed_hue_search_matches_a_full_search() {
        // The two SDR tables the outputs use: the narrowed search is exact for every hue.
        for limiting in [Primaries::P3_D65, Primaries::REC709] {
            let p = Aces2Params::new(100.0, limiting);
            // Includes both ends, the table hues themselves and values a hair either side.
            let mut hues: Vec<f64> = (0..=36_000).map(|i| i as f64 * 0.01).collect();
            for &h in &p.hue_table[BASE_INDEX..UPPER_WRAP_INDEX] {
                hues.extend([h, h + 1e-9, (h - 1e-9).max(0.0)]);
            }
            for h in hues {
                let got = p.lookup_hue_interval(h);
                let want = brute_force_interval(&p, h).max(1);
                assert_eq!(got, want, "hue {h} ({limiting:?})");
                assert!(p.hue_table[got - 1] < h || got == 1 || h == p.hue_table[got - 1]);
                assert!(h <= p.hue_table[got]);
            }
        }
    }

    #[test]
    fn narrowed_hue_search_is_at_most_one_interval_off_for_other_peaks_and_gamuts() {
        // The reference's search window (kept as is, see `determine_hue_linearity_search_range`)
        // can return the interval next to the right one for tables that run more than a degree
        // ahead of the uniform grid or end ahead of it. It must never be further off than that,
        // and the extrapolation it causes has to stay inside the neighbouring entries.
        for (peak, limiting) in [
            (48.0, Primaries::P3_D65),
            (203.0, Primaries::P3_D65),
            (1000.0, Primaries::P3_D65),
            (10_000.0, Primaries::REC709),
            (100.0, Primaries::REC2020),
            (4000.0, Primaries::REC2020),
        ] {
            let p = Aces2Params::new(peak, limiting);
            let mut off = 0;
            for k in 0..=36_000 {
                let h = k as f64 * 0.01;
                let got = p.lookup_hue_interval(h);
                let want = brute_force_interval(&p, h).max(1);
                assert!(
                    got.abs_diff(want) <= 1,
                    "hue {h} ({peak} nit, {limiting:?}): {got} vs {want}"
                );
                off += usize::from(got != want);
            }
            // Measured: 0 to 30 of the 36 001 hues (a few hundredths of a degree around a handful
            // of table entries); never a systematic error.
            assert!(
                off <= 60,
                "{off} of 36001 hues off ({peak} nit, {limiting:?})"
            );
        }
    }

    #[test]
    fn reach_lookup_handles_the_seam() {
        let p = Aces2Params::new(100.0, Primaries::P3_D65);
        let at0 = reach_m_from_table(0.0, &p.reach_m_table);
        let at360 = reach_m_from_table(360.0, &p.reach_m_table);
        let almost = reach_m_from_table(359.999_999, &p.reach_m_table);
        assert_eq!(at0, at360, "hue 360 is hue 0");
        assert!((almost - at0).abs() < 1e-3 * at0);
        assert!(at0 > 0.0);
    }

    #[test]
    fn toe_is_monotonic_continuous_and_bounded() {
        let (limit, k1, k2) = (1.5, 0.4, 0.9);
        let mut prev = toe(0.0, limit, k1, k2);
        assert!(prev.abs() < 1e-12, "toe(0) = {prev}");
        for i in 1..=300 {
            let x = i as f64 * 0.01;
            let y = toe(x, limit, k1, k2);
            assert!(y >= prev, "toe not monotonic at {x}");
            assert!(y <= x + 1e-12, "toe never lifts a value");
            prev = y;
        }
        // Continuous at the limit.
        assert!((toe(limit - 1e-9, limit, k1, k2) - toe(limit + 1e-9, limit, k1, k2)).abs() < 1e-7);
    }

    #[test]
    fn boundary_estimate_survives_an_intersection_just_above_the_top() {
        // Rounding can put the compression line's axis intersection a hair above the maximum J
        // (in float32 this is 100.000008 against 100). The reference takes pow() of the negative
        // distance and gets NaN, which turns the pixel black; the limit is a boundary M of 0.
        let m = estimate_line_and_boundary_intersection_m(-7.6e-6, -2e-9, 0.9, 20.0, 80.0, 34.0);
        assert_eq!(m, 0.0);
        let boundary = find_gamut_boundary_intersection(
            [50.0, 60.0],
            100.0,
            0.8,
            0.88,
            100.000_008,
            -2e-9,
            65.78,
        );
        assert!(
            boundary.is_finite() && boundary.abs() < 1e-3,
            "boundary {boundary}"
        );
        // And the solver's discriminants may not go negative.
        assert!(solve_j_intersect(100.0, 0.07, 45.5, 100.0, 4038.9).is_finite());
    }

    #[test]
    fn remap_m_is_identity_below_the_threshold_and_bounded_by_the_boundary() {
        let (gamut, reach) = (40.0, 100.0);
        assert_eq!(remap_m(10.0, gamut, reach), 10.0);
        let mut prev = 0.0;
        // Up to the reach boundary (nothing is more colourful than the reach gamut).
        for i in 0..=100 {
            let m = i as f64;
            let r = remap_m(m, gamut, reach);
            assert!(r >= prev);
            assert!(
                r <= gamut + 1e-9,
                "compressed M {r} exceeds the gamut boundary at {m}"
            );
            prev = r;
        }
        // The reach boundary lands on the gamut boundary.
        assert!((remap_m(reach, gamut, reach) - gamut).abs() < 1e-9);
        // Beyond the reach boundary the curve keeps rising towards its asymptote.
        assert!(remap_m(2.0 * reach, gamut, reach) > gamut);
    }
}
