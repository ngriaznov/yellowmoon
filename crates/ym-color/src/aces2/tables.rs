// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Pre-computed tables of the ACES 2 output transform.
//!
//! Port of the table generation in `Lib.Academy.OutputTransform.ctl` (`make_reach_m_table`,
//! `make_uniform_hue_gamut_table`, `make_upper_hull_gamma_table`,
//! `determine_hue_linearity_search_range`). Every table has `TOTAL_TABLE_SIZE` entries: one lower
//! wrap entry, 360 nominal entries and two upper wrap entries, so a hue lookup never needs a
//! modulo. All search tolerances are those of the reference: they are part of the algorithm, the
//! tables differ from the reference by what those tolerances leave open and nothing more.

use super::cam::{HUE_LIMIT, JMhParams, REF_LUMINANCE};
use super::compress::{
    FOCUS_GAIN_BLEND, SMOOTH_CUSPS, SMOOTH_M, compute_compression_vector_slope, compute_focus_j,
    find_gamut_boundary_intersection, get_focus_gain, lerp, solve_j_intersect,
};
use super::matrix::V3;
use super::params::{BASE_INDEX, TABLE_SIZE, TOTAL_TABLE_SIZE, UPPER_WRAP_INDEX};

const CUSP_CORNER_COUNT: usize = 6;
const TOTAL_CORNER_COUNT: usize = CUSP_CORNER_COUNT + 2;
const MAX_SORTED_CORNERS: usize = 2 * CUSP_CORNER_COUNT;
const REACH_CUSP_TOLERANCE: f64 = 1e-3;
const DISPLAY_CUSP_TOLERANCE: f64 = 1e-7;

const GAMMA_MINIMUM: f64 = 0.0;
const GAMMA_MAXIMUM: f64 = 5.0;
const GAMMA_SEARCH_STEP: f64 = 0.4;
const GAMMA_ACCURACY: f64 = 1e-5;

const TEST_COUNT: usize = 5;
const TEST_POSITIONS: [f64; TEST_COUNT] = [0.01, 0.1, 0.5, 0.8, 0.99];

#[inline]
fn midpoint(low: f64, high: f64) -> f64 {
    (low + high) / 2.0
}

#[inline]
fn any_below_zero(rgb: V3) -> bool {
    rgb[0] < 0.0 || rgb[1] < 0.0 || rgb[2] < 0.0
}

/// `make_reach_m_table`: M of the reach gamut boundary at `limit_j_max`, per degree of hue.
pub(super) fn make_reach_m_table(params: &JMhParams, limit_j_max: f64) -> Vec<f64> {
    let mut table = vec![0.0; TOTAL_TABLE_SIZE];

    for i in 0..TABLE_SIZE {
        let hue = i as f64 * HUE_LIMIT / TABLE_SIZE as f64;

        const SEARCH_RANGE: f64 = 50.0;
        const SEARCH_MAXIMUM: f64 = 1300.0;
        let mut low = 0.0;
        let mut high = low + SEARCH_RANGE;
        let mut outside = false;

        while !outside && high < SEARCH_MAXIMUM {
            let new_limit_rgb = params.jmh_to_rgb([limit_j_max, high, hue]);
            outside = any_below_zero(new_limit_rgb);
            if !outside {
                low = high;
                high += SEARCH_RANGE;
            }
        }

        while high - low > 1e-2 {
            let sample_m = (high + low) / 2.0;
            let new_limit_rgb = params.jmh_to_rgb([limit_j_max, sample_m, hue]);
            if any_below_zero(new_limit_rgb) {
                high = sample_m;
            } else {
                low = sample_m;
            }
        }

        table[i + BASE_INDEX] = high;
    }

    // Copy last populated entry to the first empty spot and the first two to the last empty spots.
    table[0] = table[TABLE_SIZE];
    table[UPPER_WRAP_INDEX] = table[BASE_INDEX];
    table[UPPER_WRAP_INDEX + 1] = table[BASE_INDEX + 1];
    table
}

/// `generate_unit_cube_cusp_corners`, order R, Y, G, C, B, M so hues rotate in order.
fn unit_cube_cusp_corner(corner: usize) -> V3 {
    let bit = |offset: usize| {
        if (corner + offset) % CUSP_CORNER_COUNT < 3 {
            1.0
        } else {
            0.0
        }
    };
    [bit(1), bit(5), bit(3)]
}

fn rotate_to_lowest_hue(
    temp: &[V3; CUSP_CORNER_COUNT],
    min_index: usize,
) -> [V3; TOTAL_CORNER_COUNT] {
    let mut out = [[0.0; 3]; TOTAL_CORNER_COUNT];
    // Rotate entries placing lowest at [1] (not [0]).
    for i in 0..CUSP_CORNER_COUNT {
        out[i + 1] = temp[(i + min_index) % CUSP_CORNER_COUNT];
    }
    // Copy end elements to create a cycle.
    out[0] = out[CUSP_CORNER_COUNT];
    out[CUSP_CORNER_COUNT + 1] = out[1];
    out
}

fn lowest_hue_index(jmh: &[V3; CUSP_CORNER_COUNT]) -> usize {
    let mut min_index = 0;
    for (i, c) in jmh.iter().enumerate() {
        if c[2] < jmh[min_index][2] {
            min_index = i;
        }
    }
    min_index
}

/// `build_limiting_cusp_corners_tables`: RGB and JMh of the limiting gamut's cusp corners.
fn build_limiting_cusp_corners(
    params: &JMhParams,
    peak_luminance: f64,
) -> ([V3; TOTAL_CORNER_COUNT], [V3; TOTAL_CORNER_COUNT]) {
    let mut temp_rgb = [[0.0; 3]; CUSP_CORNER_COUNT];
    let mut temp_jmh = [[0.0; 3]; CUSP_CORNER_COUNT];

    for i in 0..CUSP_CORNER_COUNT {
        let c = unit_cube_cusp_corner(i);
        let k = peak_luminance / REF_LUMINANCE;
        temp_rgb[i] = [k * c[0], k * c[1], k * c[2]];
        temp_jmh[i] = params.rgb_to_jmh(temp_rgb[i]);
    }
    let min_index = lowest_hue_index(&temp_jmh);

    let rgb_corners = rotate_to_lowest_hue(&temp_rgb, min_index);
    let mut jmh_corners = rotate_to_lowest_hue(&temp_jmh, min_index);

    // Wrap the hues; to maintain monotonicity these entries fall outside [0, 360).
    jmh_corners[0][2] -= HUE_LIMIT;
    jmh_corners[CUSP_CORNER_COUNT + 1][2] += HUE_LIMIT;

    (rgb_corners, jmh_corners)
}

/// `find_reach_corners_table`: JMh of the reach primaries scaled until their J equals `limit_j_max`.
fn find_reach_corners(
    params_reach: &JMhParams,
    limit_j_max: f64,
    forward_limit: f64,
) -> [V3; TOTAL_CORNER_COUNT] {
    // Use the equivalent achromatic value to search for the J value and avoid the non-linear
    // transform during the search.
    let limit_a = params_reach.j_to_a(limit_j_max);

    let mut temp_jmh = [[0.0; 3]; CUSP_CORNER_COUNT];
    for (i, jmh) in temp_jmh.iter_mut().enumerate() {
        let rgb_vector = unit_cube_cusp_corner(i);

        let mut lower = 0.0;
        let mut upper = forward_limit;

        while (upper - lower) > REACH_CUSP_TOLERANCE {
            let test = (lower + upper) / 2.0;
            let test_corner = [
                test * rgb_vector[0],
                test * rgb_vector[1],
                test * rgb_vector[2],
            ];
            let a = params_reach.rgb_to_aab(test_corner)[0];
            if a < limit_a {
                lower = test;
            } else {
                upper = test;
            }
        }

        *jmh = params_reach.rgb_to_jmh([
            upper * rgb_vector[0],
            upper * rgb_vector[1],
            upper * rgb_vector[2],
        ]);
    }
    let min_index = lowest_hue_index(&temp_jmh);

    let mut corners = rotate_to_lowest_hue(&temp_jmh, min_index);
    corners[0][2] -= HUE_LIMIT;
    corners[CUSP_CORNER_COUNT + 1][2] += HUE_LIMIT;
    corners
}

/// `extract_sorted_cube_hues`: merge of the two sorted corner hue lists; returns the hues and how
/// many are unique.
fn extract_sorted_cube_hues(
    reach_jmh: &[V3; TOTAL_CORNER_COUNT],
    limit_jmh: &[V3; TOTAL_CORNER_COUNT],
) -> ([f64; MAX_SORTED_CORNERS], usize) {
    let mut sorted = [0.0; MAX_SORTED_CORNERS];
    let mut idx = 0;
    let mut reach_idx = 1;
    let mut limit_idx = 1;
    while reach_idx < CUSP_CORNER_COUNT + 1 || limit_idx < CUSP_CORNER_COUNT + 1 {
        // Once one list is exhausted its sentinel (the wrapped first entry) is larger than any
        // remaining hue of the other list, so the merge needs no special case.
        let reach_hue = reach_jmh[reach_idx][2];
        let limit_hue = limit_jmh[limit_idx][2];
        if reach_hue == limit_hue {
            sorted[idx] = reach_hue;
            reach_idx += 1;
            limit_idx += 1; // when equal consume both
        } else if reach_hue < limit_hue {
            sorted[idx] = reach_hue;
            reach_idx += 1;
        } else {
            sorted[idx] = limit_hue;
            limit_idx += 1;
        }
        idx += 1;
    }
    (sorted, idx)
}

/// `build_hue_sample_interval`: fill `samples` uniformly spaced hues in [lower, upper).
fn build_hue_sample_interval(
    samples: usize,
    lower: f64,
    upper: f64,
    table: &mut [f64],
    base: usize,
) {
    if samples == 0 {
        return;
    }
    let delta = (upper - lower) / samples as f64;
    for i in 0..samples {
        table[base + i] = lower + i as f64 * delta;
    }
}

/// `build_hue_table`: a nearly uniform 1 degree hue table that still contains every corner hue.
fn build_hue_table(sorted_hues: &[f64; MAX_SORTED_CORNERS], unique_hues: usize) -> Vec<f64> {
    let mut hue_table = vec![0.0; TOTAL_TABLE_SIZE];

    let ideal_spacing = TABLE_SIZE as f64 / HUE_LIMIT;
    let mut samples_count = [0_i64; MAX_SORTED_CORNERS + 2];
    let mut last_idx: i64 = -1;
    // Ensure we can always sample at hue 0.
    let mut min_index: i64 = if sorted_hues[0] == 0.0 { 0 } else { 1 };
    let max_index = TABLE_SIZE as i64 - 1;

    for hue_idx in 0..unique_hues {
        let mut nominal_idx = ((sorted_hues[hue_idx] * ideal_spacing).round() as i64)
            .max(min_index)
            .min(max_index);

        if last_idx == nominal_idx {
            // The last two hues should sample at the same index, adjust them: move the previous
            // sample down if we can.
            if hue_idx > 1 && samples_count[hue_idx - 2] != (samples_count[hue_idx - 1] - 1) {
                samples_count[hue_idx - 1] -= 1;
            } else {
                nominal_idx += 1;
            }
        }
        samples_count[hue_idx] = nominal_idx.min(max_index);
        min_index = nominal_idx;
        last_idx = nominal_idx;
    }

    let mut total_samples: usize = 0;
    // Special case for the ends.
    build_hue_sample_interval(
        samples_count[0] as usize,
        0.0,
        sorted_hues[0],
        &mut hue_table,
        total_samples + 1,
    );
    total_samples += samples_count[0] as usize;

    for i in 1..unique_hues {
        let samples = (samples_count[i] - samples_count[i - 1]) as usize;
        build_hue_sample_interval(
            samples,
            sorted_hues[i - 1],
            sorted_hues[i],
            &mut hue_table,
            total_samples + 1,
        );
        total_samples += samples;
    }
    build_hue_sample_interval(
        TABLE_SIZE - total_samples,
        sorted_hues[unique_hues - 1],
        HUE_LIMIT,
        &mut hue_table,
        total_samples + 1,
    );

    hue_table[0] = hue_table[TABLE_SIZE] - HUE_LIMIT;
    hue_table[UPPER_WRAP_INDEX] = hue_table[BASE_INDEX] + HUE_LIMIT;
    hue_table[UPPER_WRAP_INDEX + 1] = hue_table[BASE_INDEX + 1] + HUE_LIMIT;
    hue_table
}

/// `find_display_cusp_for_hue`: (J, M) of the limiting gamut cusp at `hue`.
///
/// Finds the line segment between two cusp corners that contains the hue, then bisects along the
/// segment (all points on it are valid cusp locations). The reference starts every search at t = 0;
/// so does this.
fn find_display_cusp_for_hue(
    hue: f64,
    rgb_corners: &[V3; TOTAL_CORNER_COUNT],
    jmh_corners: &[V3; TOTAL_CORNER_COUNT],
    params: &JMhParams,
) -> [f64; 2] {
    let mut upper_corner = 1;
    for (i, c) in jmh_corners.iter().enumerate().skip(1) {
        if c[2] > hue {
            upper_corner = i;
            break;
        }
    }
    let lower_corner = upper_corner - 1;

    // hue should now be within [lower_corner, upper_corner); handle an exact match.
    if jmh_corners[lower_corner][2] == hue {
        return [jmh_corners[lower_corner][0], jmh_corners[lower_corner][1]];
    }

    let cusp_lower = rgb_corners[lower_corner];
    let cusp_upper = rgb_corners[upper_corner];
    let sample_at = |t: f64| {
        params.rgb_to_jmh([
            lerp(cusp_lower[0], cusp_upper[0], t),
            lerp(cusp_lower[1], cusp_upper[1], t),
            lerp(cusp_lower[2], cusp_upper[2], t),
        ])
    };

    let mut lower_t = 0.0;
    let mut upper_t = 1.0;

    // There is an edge case where we need to search towards the range when across the [0, 360)
    // boundary; each edge needs the directions swapped. This is handled by comparing against the
    // appropriate corner to make sure we are still in the expected range between the lower and
    // upper corner hue limits.
    while (upper_t - lower_t) > DISPLAY_CUSP_TOLERANCE {
        let sample_t = midpoint(lower_t, upper_t);
        let jmh = sample_at(sample_t);
        if jmh[2] < jmh_corners[lower_corner][2] {
            upper_t = sample_t;
        } else if jmh[2] >= jmh_corners[upper_corner][2] {
            lower_t = sample_t;
        } else if jmh[2] > hue {
            upper_t = sample_t;
        } else {
            lower_t = sample_t;
        }
    }

    // Use the midpoint of the final interval for the actual sample.
    let jmh = sample_at(midpoint(lower_t, upper_t));
    [jmh[0], jmh[1]]
}

/// Hue, cusp J and cusp M tables of the limiting gamut.
pub(super) struct CuspTables {
    pub hue: Vec<f64>,
    pub cusp_j: Vec<f64>,
    pub cusp_m: Vec<f64>,
}

/// `make_uniform_hue_gamut_table`: sample the hues as uniformly as possible while still sampling
/// the corners of the limiting gamut and of the reach primaries at `limit_j_max`.
pub(super) fn make_uniform_hue_gamut_table(
    reach_params: &JMhParams,
    limit_params: &JMhParams,
    peak_luminance: f64,
    forward_limit: f64,
    limit_j_max: f64,
) -> CuspTables {
    let reach_jmh_corners = find_reach_corners(reach_params, limit_j_max, forward_limit);
    let (limiting_rgb_corners, limiting_jmh_corners) =
        build_limiting_cusp_corners(limit_params, peak_luminance);
    let (sorted_hues, unique_hues) =
        extract_sorted_cube_hues(&reach_jmh_corners, &limiting_jmh_corners);
    let hue = build_hue_table(&sorted_hues, unique_hues);

    let mut cusp_j = vec![0.0; TOTAL_TABLE_SIZE];
    let mut cusp_m = vec![0.0; TOTAL_TABLE_SIZE];
    for i in BASE_INDEX..UPPER_WRAP_INDEX {
        let jm = find_display_cusp_for_hue(
            hue[i],
            &limiting_rgb_corners,
            &limiting_jmh_corners,
            limit_params,
        );
        cusp_j[i] = jm[0];
        // The cusp M is inflated slightly so the smooth minimum of the two hulls touches the cusp.
        cusp_m[i] = jm[1] * (1.0 + SMOOTH_M * SMOOTH_CUSPS);
    }

    // Copy extra entries to ease the code that handles hues wrapping around.
    for table in [&mut cusp_j, &mut cusp_m] {
        table[0] = table[TABLE_SIZE];
        table[UPPER_WRAP_INDEX] = table[BASE_INDEX];
        table[UPPER_WRAP_INDEX + 1] = table[BASE_INDEX + 1];
    }

    CuspTables {
        hue,
        cusp_j,
        cusp_m,
    }
}

/// One sample position of the upper hull gamma search (`TestData`).
#[derive(Clone, Copy)]
struct GammaTest {
    test_h: f64,
    j_intersect_source: f64,
    slope: f64,
    j_intersect_cusp: f64,
}

/// `generate_gamma_test_data`.
fn generate_gamma_test_data(
    jm_cusp: [f64; 2],
    hue: f64,
    limit_j_max: f64,
    mid_j: f64,
    focus_dist: f64,
) -> [GammaTest; TEST_COUNT] {
    let analytical_threshold = lerp(jm_cusp[0], limit_j_max, FOCUS_GAIN_BLEND);
    let focus_j = compute_focus_j(jm_cusp[0], mid_j, limit_j_max);

    let mut data = [GammaTest {
        test_h: hue,
        j_intersect_source: 0.0,
        slope: 0.0,
        j_intersect_cusp: 0.0,
    }; TEST_COUNT];
    for (test, pos) in data.iter_mut().zip(TEST_POSITIONS) {
        let test_j = lerp(jm_cusp[0], limit_j_max, pos);
        let slope_gain = get_focus_gain(test_j, analytical_threshold, limit_j_max, focus_dist);
        let j_intersect = solve_j_intersect(test_j, jm_cusp[1], focus_j, limit_j_max, slope_gain);
        test.j_intersect_source = j_intersect;
        test.slope =
            compute_compression_vector_slope(j_intersect, focus_j, limit_j_max, slope_gain);
        test.j_intersect_cusp =
            solve_j_intersect(jm_cusp[0], jm_cusp[1], focus_j, limit_j_max, slope_gain);
    }
    data
}

/// Inputs shared by every `evaluate_gamma_fit` call.
struct GammaFit<'a> {
    peak_luminance: f64,
    limit_j_max: f64,
    lower_hull_gamma_inv: f64,
    limit_params: &'a JMhParams,
}

impl GammaFit<'_> {
    /// `evaluate_gamma_fit`: true if every test point lands outside the top of the gamut shell.
    fn evaluate(
        &self,
        jm_cusp: [f64; 2],
        tests: &[GammaTest; TEST_COUNT],
        top_gamma_inv: f64,
    ) -> bool {
        let luminance_limit = self.peak_luminance / REF_LUMINANCE;

        for t in tests {
            // Compute the gamut boundary intersection.
            let approx_limit_m = find_gamut_boundary_intersection(
                jm_cusp,
                self.limit_j_max,
                top_gamma_inv,
                self.lower_hull_gamma_inv,
                t.j_intersect_source,
                t.slope,
                t.j_intersect_cusp,
            );
            let approx_limit_j = t.j_intersect_source + t.slope * approx_limit_m;

            let new_limit_rgb =
                self.limit_params
                    .jmh_to_rgb([approx_limit_j, approx_limit_m, t.test_h]);

            // If no channel exceeds the luminance limit we are not outside the top gamut shell.
            let outside_hull = new_limit_rgb.iter().any(|&c| c > luminance_limit);
            if !outside_hull {
                return false;
            }
        }
        true
    }
}

/// `make_upper_hull_gamma_table`: the smallest gamma (stored inverted) per hue for which the
/// approximated upper hull lies outside the real gamut shell at five test lightness values.
pub(super) fn make_upper_hull_gamma_table(
    cusps: &CuspTables,
    peak_luminance: f64,
    limit_j_max: f64,
    mid_j: f64,
    focus_dist: f64,
    lower_hull_gamma_inv: f64,
    limit_params: &JMhParams,
) -> Vec<f64> {
    let mut table = vec![0.0; TOTAL_TABLE_SIZE];
    let fit = GammaFit {
        peak_luminance,
        limit_j_max,
        lower_hull_gamma_inv,
        limit_params,
    };

    for (i, entry) in table
        .iter_mut()
        .enumerate()
        .take(UPPER_WRAP_INDEX)
        .skip(BASE_INDEX)
    {
        // Get the cusp from the cusp table at this hue position.
        let hue = cusps.hue[i];
        let jm_cusp = [cusps.cusp_j[i], cusps.cusp_m[i]];

        let tests = generate_gamma_test_data(jm_cusp, hue, limit_j_max, mid_j, focus_dist);

        let search_range = GAMMA_SEARCH_STEP;
        let mut low = GAMMA_MINIMUM;
        let mut high = low + search_range;
        let mut outside = false;
        while !outside && high < GAMMA_MAXIMUM {
            if !fit.evaluate(jm_cusp, &tests, 1.0 / high) {
                low = high;
                high += search_range;
            } else {
                outside = true;
            }
        }

        while (high - low) > GAMMA_ACCURACY {
            let test_gamma = midpoint(high, low);
            if fit.evaluate(jm_cusp, &tests, 1.0 / test_gamma) {
                high = test_gamma;
            } else {
                low = test_gamma;
            }
        }

        *entry = 1.0 / high;
    }

    // Copy last populated entry to the first empty spot and the first two to the last empty spots.
    table[0] = table[TABLE_SIZE];
    table[UPPER_WRAP_INDEX] = table[BASE_INDEX];
    table[UPPER_WRAP_INDEX + 1] = table[BASE_INDEX + 1];
    table
}

/// `determine_hue_linearity_search_range`: how far the real hue table strays from a uniform one,
/// used to narrow the binary search in `lookup_hue_interval`.
///
/// Kept exactly as the reference has it (lower padding 0, upper padding 1), including its one
/// weakness: the window starts at an entry that is assumed to lie below the searched hue, which
/// only holds when no entry runs a whole degree ahead of the uniform grid, and the window is cut at
/// the last nominal entry. For the 100 nit SDR tables of Rec.709 and P3-D65 the largest lead is
/// 0.3 degrees and the search is exact for every hue (`narrowed_hue_search_matches_a_full_search`).
/// Other peaks and gamuts (for example 10 000 nit Rec.709, hues in [359, 359.03)) can return the
/// neighbouring interval, so the lookup extrapolates slightly; OpenColorIO does the same, and the
/// outputs are compared to it.
pub(super) fn determine_hue_linearity_search_range(hue_table: &[f64]) -> [i32; 2] {
    const LOWER_PADDING: i32 = 0;
    const UPPER_PADDING: i32 = 1;

    let mut range = [LOWER_PADDING, UPPER_PADDING];

    for (i, &hue) in hue_table
        .iter()
        .enumerate()
        .take(UPPER_WRAP_INDEX)
        .skip(BASE_INDEX)
    {
        let pos = BASE_INDEX as i32 + (hue / HUE_LIMIT * TABLE_SIZE as f64) as i32;
        let delta = i as i32 - pos;
        range[0] = range[0].min(delta + LOWER_PADDING);
        range[1] = range[1].max(delta + UPPER_PADDING);
    }
    range
}
