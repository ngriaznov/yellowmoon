// ACES 2.0 output transform (forward, SDR), stage C12 of docs/02. A port of the CPU reference in
// ym_color::aces2, which is itself a port of the ACES project's CTL (aces-core, Apache-2.0;
// Copyright Contributors to the ACES Project). The pre-computed tables (reach M, hue, cusp J and M,
// upper hull gamma) and all matrices are built on the CPU by ym_color and uploaded: `CU.aces` (a member
// of the colour chain's one uniform block, K3) holds the scalars and matrices, `tables` the five
// 363-entry hue tables at T_ACES.
//
// Row-vector matrices of the reference (`v * M`) are uploaded transposed, so every product here is
// `mv3(row0, row1, row2, v)` with the rows being the columns of the reference matrix.
//
// The reference runs in f64; this is f32, like the CTL and OpenColorIO. The NaN guards of the CPU
// port are kept: where the tone-scaled J lands on the display maximum the compression line's axis
// intersection can round a hair above it; the ratio is clamped at 0 and a non-positive boundary M
// means "no colour left".

struct AcesU {
    in_cam0: vec4<f32>,   // input (AP0) JMh: RGB -> adapted cone responses, rows = columns of the matrix
    in_cam1: vec4<f32>,
    in_cam2: vec4<f32>,
    in_aab0: vec4<f32>,   // compressed cone responses -> (A, a, b)
    in_aab1: vec4<f32>,
    in_aab2: vec4<f32>,
    lim_aab0: vec4<f32>,  // limiting gamut: (A, a, b) -> cone responses
    lim_aab1: vec4<f32>,
    lim_aab2: vec4<f32>,
    lim_cam0: vec4<f32>,  // limiting gamut: adapted cone responses -> RGB
    lim_cam1: vec4<f32>,
    lim_cam2: vec4<f32>,
    l2d0: vec4<f32>,      // limiting RGB -> encoding RGB (column-vector form)
    l2d1: vec4<f32>,
    l2d2: vec4<f32>,
    in_p: vec4<f32>,      // f_l_n, cz, inv_cz, a_w_j
    in_q: vec4<f32>,      // inv_a_w_j, limit_j_max, model_gamma_inv, forward_limit
    lim_p: vec4<f32>,     // inv_cz of the limiting gamut, mid_j, focus_dist, lower_hull_gamma_inv
    ts0: vec4<f32>,       // tone scale: n_r, g, t_1, s_2
    ts1: vec4<f32>,       // m_2, peak_linear, 0, 0
    cc: vec4<f32>,        // sat, sat_thr, compr, chroma_compress_scale
    hue: vec4<i32>,       // hue_linearity_search_range lo, hi; x != 0 when l2d is not the identity
};


const CAM_NL_OFFSET: f32 = 27.13;
const SMOOTH_CUSPS: f32 = 0.12;
const CUSP_MID_BLEND: f32 = 1.3;
const FOCUS_GAIN_BLEND: f32 = 0.3;
const COMPRESSION_THRESHOLD: f32 = 0.75;
// BASE_INDEX and UPPER_WRAP_INDEX (the hue-table geometry) are generated from ym_color.

fn tab(k: u32, i: u32) -> f32 {
    return tables[T_ACES + k * T_ACES_N + i];
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn pa_fwd_u(rc: f32) -> f32 {
    let f = pow_pos(rc, 0.42);
    return f / (CAM_NL_OFFSET + f);
}

fn pa_inv_u(ra: f32) -> f32 {
    let r = min(ra, 0.99);
    let f = (CAM_NL_OFFSET * r) / (1.0 - r);
    return pow_pos(f, 1.0 / 0.42);
}

// CTL `sign(y) * fabs(x)` with sign(0) = 0.
fn sign_abs(x: f32, y: f32) -> f32 {
    if (y < 0.0) {
        return -abs(x);
    }
    if (y > 0.0) {
        return abs(x);
    }
    return 0.0;
}

fn pa_fwd(v: f32) -> f32 {
    return sign_abs(pa_fwd_u(abs(v)), v);
}

fn pa_inv(v: f32) -> f32 {
    return sign_abs(pa_inv_u(abs(v)), v);
}

fn wrap360(h: f32) -> f32 {
    let y = h % 360.0;
    if (y < 0.0) {
        return y + 360.0;
    }
    return y;
}

// RGB (AP0, 1.0 = 100 nits) -> (J, M, h).
fn rgb_to_jmh(rgb: vec3<f32>) -> vec3<f32> {
    let m = mv3(CU.aces.in_cam0.xyz, CU.aces.in_cam1.xyz, CU.aces.in_cam2.xyz, rgb);
    let a = vec3<f32>(pa_fwd(m.x), pa_fwd(m.y), pa_fwd(m.z));
    let aab = mv3(CU.aces.in_aab0.xyz, CU.aces.in_aab1.xyz, CU.aces.in_aab2.xyz, a);
    if (aab.x <= 0.0) {
        return vec3<f32>(0.0);
    }
    let j = 100.0 * pow_pos(aab.x, CU.aces.in_p.y);
    let mm = sqrt(aab.y * aab.y + aab.z * aab.z);
    var h = 0.0;
    if (mm > 0.0) {
        h = wrap360(degrees(atan2(aab.z, aab.y)));
    }
    return vec3<f32>(j, mm, h);
}

// (J, M, h) -> RGB of the limiting primaries.
fn jmh_to_rgb_limit(jmh: vec3<f32>) -> vec3<f32> {
    let hr = radians(jmh.z);
    let a = pow_pos(jmh.x * (1.0 / 100.0), CU.aces.lim_p.x);
    let aab = vec3<f32>(a, jmh.y * cos(hr), jmh.y * sin(hr));
    let cone = mv3(CU.aces.lim_aab0.xyz, CU.aces.lim_aab1.xyz, CU.aces.lim_aab2.xyz, aab);
    let m = vec3<f32>(pa_inv(cone.x), pa_inv(cone.y), pa_inv(cone.z));
    return mv3(CU.aces.lim_cam0.xyz, CU.aces.lim_cam1.xyz, CU.aces.lim_cam2.xyz, m);
}

// Luminance (nits) to lightness J of the input space.
fn y_to_j(y: f32) -> f32 {
    let ra = pa_fwd_u(abs(y) * CU.aces.in_p.x);
    let j = 100.0 * pow_pos(ra * CU.aces.in_q.x, CU.aces.in_p.y);
    return sign_abs(j, y);
}

// Lightness J of the input space to luminance (nits).
fn j_to_y(j: f32) -> f32 {
    let a = pow_pos(abs(j) * (1.0 / 100.0), CU.aces.in_p.z);
    return pa_inv_u(CU.aces.in_p.w * a) / CU.aces.in_p.x;
}

fn tonescale_fwd(x: f32) -> f32 {
    let f = CU.aces.ts1.x * pow_pos(max(x, 0.0) / (x + CU.aces.ts0.w), CU.aces.ts0.y);
    let h = max(f * f / (f + CU.aces.ts0.z), 0.0);
    return h * CU.aces.ts0.x;
}

fn reach_m_from_table(h: f32) -> f32 {
    let base = u32(h / 360.0 * 360.0);
    let t = h - f32(base);
    let i = base + BASE_INDEX;
    return lerpf(tab(0u, i), tab(0u, i + 1u), t);
}

fn toe_fn(x: f32, limit: f32, k1_in: f32, k2_in: f32) -> f32 {
    if (x > limit) {
        return x;
    }
    let k2 = max(k2_in, 0.001);
    let k1 = sqrt(k1_in * k1_in + k2 * k2);
    let k3 = (limit + k1) / (limit + k2);
    let minus_b = k3 * x - k1;
    let minus_c = k2 * k3 * x;
    return 0.5 * (minus_b + sqrt(max(minus_b * minus_b + 4.0 * minus_c, 0.0)));
}

fn chroma_compress_norm(h_deg: f32) -> f32 {
    let hr = radians(h_deg);
    let a = cos(hr);
    let b = sin(hr);
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
    return m * CU.aces.cc.w;
}

// Tone-scaled lightness and chroma compression: (J, M, h) -> (J_ts, M', h).
fn chroma_compress_fwd(jmh: vec3<f32>, tonemapped_j: f32) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;
    var m_compr = m;
    if (m != 0.0) {
        let limit_j_max = CU.aces.in_q.y;
        let model_gamma_inv = CU.aces.in_q.z;
        let n_j = tonemapped_j / limit_j_max;
        let sn_j = max(1.0 - n_j, 0.0);
        let m_norm = chroma_compress_norm(h);
        let limit = pow_pos(n_j, model_gamma_inv) * reach_m_from_table(h) / m_norm;

        let toe_limit = limit - 0.001;
        let toe_sn_j_sat = sn_j * CU.aces.cc.x;
        let toe_sqrt_n_j_sat_thr = sqrt(n_j * n_j + CU.aces.cc.y);
        let toe_n_j_compr = n_j * CU.aces.cc.z;

        // Rescale M with the tone-scaled J (Hellwig 2022 model gamma), normalise by the cusp M.
        m_compr = m * pow_pos(tonemapped_j / j, model_gamma_inv);
        m_compr = m_compr / m_norm;
        // Expand the colourfulness of less saturated colours less, by running the toe in reverse.
        m_compr = limit - toe_fn(limit - m_compr, toe_limit, toe_sn_j_sat, toe_sqrt_n_j_sat_thr);
        // Compress the colourfulness.
        m_compr = toe_fn(m_compr, limit, toe_n_j_compr, sn_j);
        m_compr = m_compr * m_norm;
    }
    return vec3<f32>(tonemapped_j, m_compr, h);
}

fn solve_j_intersect(j: f32, m: f32, focus_j: f32, max_j: f32, slope_gain: f32) -> f32 {
    let m_scaled = m / slope_gain;
    let a = m_scaled / focus_j;
    if (j < focus_j) {
        let b = 1.0 - m_scaled;
        let c = -j;
        let det = b * b - 4.0 * a * c;
        let root = sqrt(max(det, 0.0));
        return -2.0 * c / (b + root);
    }
    let b = -(1.0 + m_scaled + max_j * a);
    let c = max_j * m_scaled + j;
    let det = b * b - 4.0 * a * c;
    let root = sqrt(max(det, 0.0));
    return -2.0 * c / (b - root);
}

fn compression_vector_slope(intersect_j: f32, focus_j: f32, limit_j_max: f32, slope_gain: f32) -> f32 {
    var direction_scalar = limit_j_max - intersect_j;
    if (intersect_j < focus_j) {
        direction_scalar = intersect_j;
    }
    return direction_scalar * (intersect_j - focus_j) / (focus_j * slope_gain);
}

fn smin_scaled(a: f32, b: f32, scale_reference: f32) -> f32 {
    let s_scaled = SMOOTH_CUSPS * scale_reference;
    let h = max(s_scaled - abs(a - b), 0.0) / s_scaled;
    return min(a, b) - h * h * h * s_scaled * (1.0 / 6.0);
}

fn line_boundary_intersection_m(
    j_axis_intersect: f32,
    slope: f32,
    inv_gamma: f32,
    j_max: f32,
    m_max: f32,
    j_intersection_reference: f32,
) -> f32 {
    let normalised_j = max(j_axis_intersect / j_intersection_reference, 0.0);
    let shifted_intersection = j_intersection_reference * pow_pos(normalised_j, inv_gamma);
    return shifted_intersection * m_max / (j_max - slope * m_max);
}

fn find_gamut_boundary_intersection(
    cusp_j: f32,
    cusp_m: f32,
    j_max: f32,
    gamma_top_inv: f32,
    gamma_bottom_inv: f32,
    j_intersect_source: f32,
    slope: f32,
    j_intersect_cusp: f32,
) -> f32 {
    let m_lower = line_boundary_intersection_m(
        j_intersect_source, slope, gamma_bottom_inv, cusp_j, cusp_m, j_intersect_cusp);
    let f_j_intersect_cusp = j_max - j_intersect_cusp;
    let f_j_intersect_source = j_max - j_intersect_source;
    let f_jm_cusp_j = j_max - cusp_j;
    let m_upper = line_boundary_intersection_m(
        f_j_intersect_source, -slope, gamma_top_inv, f_jm_cusp_j, cusp_m, f_j_intersect_cusp);
    return smin_scaled(m_lower, m_upper, cusp_m);
}

fn get_focus_gain(j: f32, analytical_threshold: f32, limit_j_max: f32, focus_dist: f32) -> f32 {
    var gain = limit_j_max * focus_dist;
    if (j > analytical_threshold) {
        let r = (limit_j_max - analytical_threshold) / max(limit_j_max - j, 0.0001);
        var adj = log2(r) * LOG10_2;
        adj = adj * adj + 1.0;
        gain = gain * adj;
    }
    return gain;
}

fn compute_focus_j(cusp_j: f32, mid_j: f32, limit_j_max: f32) -> f32 {
    return lerpf(cusp_j, mid_j, min(CUSP_MID_BLEND - (cusp_j / limit_j_max), 1.0));
}

fn remap_m(m: f32, gamut_boundary_m: f32, reach_boundary_m: f32) -> f32 {
    let boundary_ratio = gamut_boundary_m / reach_boundary_m;
    let proportion = max(boundary_ratio, COMPRESSION_THRESHOLD);
    let threshold = proportion * gamut_boundary_m;
    if (m <= threshold || proportion >= 1.0) {
        return m;
    }
    let m_offset = m - threshold;
    let gamut_offset = gamut_boundary_m - threshold;
    let reach_offset = reach_boundary_m - threshold;
    let scale = reach_offset / ((reach_offset / gamut_offset) - 1.0);
    let nd = m_offset / scale;
    return threshold + scale * nd / (1.0 + nd);
}

// Upper index of the hue-table interval containing h (binary search from a window around the
// uniform position, as the reference).
fn lookup_hue_interval(h: f32) -> u32 {
    let i0 = i32(BASE_INDEX) + i32(h / 360.0 * 360.0);
    var i = i0;
    var i_lo = max(i0 + CU.aces.hue.x, 0);
    var i_hi = min(i0 + CU.aces.hue.y, i32(UPPER_WRAP_INDEX));
    for (var iter = 0; iter < 32; iter++) {
        if (i_lo + 1 >= i_hi) {
            break;
        }
        if (h > tab(1u, u32(i))) {
            i_lo = i;
        } else {
            i_hi = i;
        }
        i = (i_lo + i_hi) / 2;
    }
    return u32(max(i_hi, 1));
}

fn gamut_compress_fwd(jmh: vec3<f32>) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;
    if (j <= 0.0) {
        return vec3<f32>(0.0, 0.0, h);
    }
    let limit_j_max = CU.aces.in_q.y;
    if (m < 0.0 || j > limit_j_max) {
        return vec3<f32>(j, 0.0, h);
    }
    // Hue dependent gamut parameters.
    let i_hi = lookup_hue_interval(h);
    let i_lo = i_hi - 1u;
    let t = (h - tab(1u, i_lo)) / (tab(1u, i_hi) - tab(1u, i_lo));
    let cusp_j = lerpf(tab(2u, i_lo), tab(2u, i_hi), t);
    let cusp_m = lerpf(tab(3u, i_lo), tab(3u, i_hi), t);
    let gamma_top_inv = lerpf(tab(4u, i_lo), tab(4u, i_hi), t);
    let gamma_bottom_inv = CU.aces.lim_p.w;
    let mid_j = CU.aces.lim_p.y;
    let focus_dist = CU.aces.lim_p.z;
    let focus_j = compute_focus_j(cusp_j, mid_j, limit_j_max);
    let analytical_threshold = lerpf(cusp_j, limit_j_max, FOCUS_GAIN_BLEND);

    let slope_gain = get_focus_gain(j, analytical_threshold, limit_j_max, focus_dist);
    let j_intersect_source = solve_j_intersect(j, m, focus_j, limit_j_max, slope_gain);
    let gamut_slope = compression_vector_slope(j_intersect_source, focus_j, limit_j_max, slope_gain);
    let j_intersect_cusp = solve_j_intersect(cusp_j, cusp_m, focus_j, limit_j_max, slope_gain);
    let gamut_boundary_m = find_gamut_boundary_intersection(
        cusp_j, cusp_m, limit_j_max, gamma_top_inv, gamma_bottom_inv,
        j_intersect_source, gamut_slope, j_intersect_cusp);
    // Also catches NaN (a comparison with NaN is false, so the negation takes this branch).
    if (!(gamut_boundary_m > 0.0)) {
        return vec3<f32>(j, 0.0, h);
    }
    let reach_max_m = reach_m_from_table(h);
    let reach_boundary_m = line_boundary_intersection_m(
        j_intersect_source, gamut_slope, CU.aces.in_q.z, limit_j_max, reach_max_m, limit_j_max);
    let remapped_m = remap_m(m, gamut_boundary_m, reach_boundary_m);
    return vec3<f32>(j_intersect_source + remapped_m * gamut_slope, remapped_m, h);
}

// Linear ACES2065-1 (AP0) -> RGB of the limiting primaries, units of 100 nits, before any clamp.
fn aces2_limit_rgb(ap0_in: vec3<f32>) -> vec3<f32> {
    let ap0 = vec3<f32>(sanitize(ap0_in.x), sanitize(ap0_in.y), sanitize(ap0_in.z));
    let ap1 = mv3(AP0_TO_AP1_0, AP0_TO_AP1_1, AP0_TO_AP1_2, ap0);
    let clamped = clamp(ap1, vec3<f32>(0.0), vec3<f32>(CU.aces.in_q.w));
    let ap0c = mv3(AP1_TO_AP0_0, AP1_TO_AP0_1, AP1_TO_AP0_2, clamped);
    let jmh = rgb_to_jmh(ap0c);
    // Tone scale on J, then chroma compression.
    let linear = j_to_y(jmh.x) / 100.0;
    let j_ts = y_to_j(tonescale_fwd(linear));
    let tm = chroma_compress_fwd(jmh, j_ts);
    let gc = gamut_compress_fwd(tm);
    return jmh_to_rgb_limit(gc);
}

// Linear AP0 -> display-linear RGB in the output's encoding primaries, clamped to [0, peak].
fn aces2_display_linear(ap0: vec3<f32>) -> vec3<f32> {
    let peak = CU.aces.ts1.y;
    let limited = clamp(aces2_limit_rgb(ap0), vec3<f32>(0.0), vec3<f32>(peak));
    var rgb = limited;
    if (CU.aces.hue.z != 0) {
        rgb = mv3(CU.aces.l2d0.xyz, CU.aces.l2d1.xyz, CU.aces.l2d2.xyz, limited);
    }
    return min(max(rgb, vec3<f32>(0.0)), vec3<f32>(peak));
}
