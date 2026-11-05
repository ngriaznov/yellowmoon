// The cine rendering, stage C12 of docs/02 (*Cine rendering (C12, default)*): scene-linear AWG4 in,
// display-linear RGB in the primaries of the output out (1.0 is the 100 nit SDR white). A port of the CPU
// reference `ym_color::display::cine` (`CineRendering::forward`) in f32. Every constant is written in front of
// the module from the Rust table (`CINE_*`, `kernels/shader.rs`), and the gamut stage's rows for the output's own
// primaries (the matrix from XYZ as `K0 = M_i0 XN`, `K2 = M_i2 ZN`, step 7) are in `CU.cine_lo*`.
//
// Accurate to about 2e-7 of the display colour (F-08d, docs/24 section 3.9c): the logs, exps, cube roots and the gamut stage are
// the pair arithmetic of precise.wgsl (docs/17 section 13 has the argument that fast-math cannot break it); the rest of the
// rendering is plain f32.
//
// Safe under Metal's fast-math (docs/17 K8), by the notes of docs/02 C12 (GPU notes):
//   * no atan2, sign or comparison of two nearly equal values: the hue enters as (cs, sn) = (ca, cb) / chi and cos 2t,
//     sin 2t, cos 3t, sin 3t are polynomials of it; every conditional is a min, max or clamp at a C1 or continuous
//     junction (the soft clips, the chroma limit, the floor);
//   * every log and pow has a positive argument (Ys >= Y_MIN, chi >= 1e-7, (1 + d^2) / (r^2 + d^2) > 0), pow is
//     written exp(x * log(y)), every denominator is bounded away from 0;
//   * nothing relies on NaN or infinity: the input is clamped to +-SCENE_LIMIT (1e6, finite whatever the camera
//     sample), so X / (XN Ys) is at most about 1e15 and its square 1e30 is inside f32; the chroma limit brings chi to
//     at most chi_m before anything else uses it (the tanh argument is held to 20); after that the largest
//     intermediate is (1 + ca2)^3, at most 2.3e3 in magnitude;
//   * the smooth maximum of the luminance floor is written in the cancellation-free form for Y < Yf;
//   * the chromatic distance of the luminance floor is formed from the channel differences, never from X, Y, Z of a
//     large triple: dX and dZ of a neutral colour are exactly 0 (see step 2), whatever the sign and size of the
//     neutral and whether the compiler fuses or re-associates, instead of rounding noise of about 1e-7 |x| that
//     decides the luminance (docs/02 C12, GPU notes).

fn cine_soft_clip(z: f32, w: f32) -> f32 {
    let lo = 1.0 - w;
    let t = clamp(z, lo, 1.0 + w) - lo;
    return min(z - t * t / (4.0 * w), 1.0);
}

fn cine_sigmoid(x: f32) -> f32 {
    return px_div(1.0, px_op(1.0 + px_exp(clamp(-x, -80.0, 80.0))));
}

// r ((1 + d^2) / (r^2 + d^2))^(1/3): odd, smooth at 0, 1 at 1.
fn cine_cbrt_d(r: f32) -> f32 {
    let q = px_div(CINE_ONE_PLUS_DELTA2, px_op(px_op(r * r) + CINE_DELTA2));
    return px_op(r * px_pow(q, 0.3333333432674407, -9.934107e-9));
}

// k1 cs + l1 sn + k2 cos 2t + l2 sin 2t + k3 cos 3t + l3 sin 3t of the unit direction (cs, sn); a = (k1, l1, k2),
// b = (l2, k3, l3).
fn cine_harm(cs: f32, sn: f32, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let c2 = cs * cs - sn * sn;
    let s2 = 2.0 * cs * sn;
    let c3 = cs * c2 - sn * s2;
    let s3 = sn * c2 + cs * s2;
    return dot(vec3<f32>(cs, sn, c2), a) + dot(vec3<f32>(s2, c3, s3), b);
}

// cine_cbrt_d(r) - 1 without the rounding of a number near 1: r e^g - 1 = (r - 1) + r (e^g - 1), g = ln(q) / 3. Away from
// r = 1 (a colour far from the neutral axis, or a negative triple) the two terms are large and cancel, and the plain
// form is the better one: the branch is on r alone and both forms are the same function.
fn cine_cbrt_d_m1(r: f32) -> f32 {
    if (r < 0.5 || r > 2.0) {
        return px_op(cine_cbrt_d(r) - 1.0);
    }
    let q = px_div(CINE_ONE_PLUS_DELTA2, px_op(px_op(r * r) + CINE_DELTA2));
    let g = px_op(px_log(q) * 0.3333333432674407);
    return px_op(px_op(r - 1.0) + px_op(r * px_expm1(g)));
}

// ---- the plain f32 rendering of a view (the arithmetic before F-08d, `px_fast()`) ---------------------------------
//
// A view runs the rendering with the compiler's own f32 operations: the compensated arithmetic below is for the
// export, which the owner accepted slower (docs/24 section 3.9c, "Speed"). These are the former functions, unchanged.

fn cine_sigmoid_plain(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// r ((1 + d^2) / (r^2 + d^2))^(1/3): odd, smooth at 0, 1 at 1.
fn cine_cbrt_d_plain(r: f32) -> f32 {
    let q = CINE_ONE_PLUS_DELTA2 / (r * r + CINE_DELTA2);
    return r * exp(log(q) * (1.0 / 3.0));
}

fn cine_display_linear_plain(awg4: vec3<f32>) -> vec3<f32> {
    // 0, 1. The limit and the matrix to XYZ.
    let x = clamp(awg4, vec3<f32>(-CINE_SCENE_LIMIT), vec3<f32>(CINE_SCENE_LIMIT));
    let xyz = mv3(AWG4_TO_XYZ_0, AWG4_TO_XYZ_1, AWG4_TO_XYZ_2, x);
    // 2. The luminance floor: a smooth maximum of Y and rho times the distance of the chromaticity from the
    // neutral axis. dX = X - XN Y and dZ = Z - ZN Y are linear in x, with the rows DX = row_X - XN row_Y and
    // DZ = row_Z - ZN row_Y (CINE_DX, CINE_DZ, formed in f64 on the host). AWG4's white maps to (XN, 1, ZN), so
    // each row sums to 0 (to 2e-16 in f64) and DX . x = DX . (x - x_g 1): only the differences to the green
    // channel enter, and the green entry is not needed. Taken from X, Y and Z, the two terms of dX are numbers of
    // size |x| whose difference is rounding noise of about 1e-7 |x| for a neutral (1e-2 at -1e5), different
    // for every order of the sums and for every fusion into a multiply-add; yf and then Ys would be that noise.
    // From the differences a neutral gives exactly 0 (x - x is 0, a product or sum of zeros is 0, whatever the
    // compiler does with the products and sums), and any other colour gives dX with an error relative to its
    // own differences.
    let d = vec2<f32>(x.x - x.y, x.z - x.y);
    let dx = dot(CINE_DX, d);
    let dz = dot(CINE_DZ, d);
    let yf = CINE_RHO * sqrt(dx * dx + dz * dz + CINE_EPS_FLOOR);
    let w = CINE_WFRAC * yf;
    let dy = xyz.y - yf;
    let root = sqrt(dy * dy + w * w);
    var smooth_max = 0.5 * (xyz.y + yf + root);
    if (dy < 0.0) {
        smooth_max = yf + 0.5 * w * w / (root - dy);
    }
    let ys = max(smooth_max, CINE_Y_MIN);
    let ly = log(ys);
    // 3. Opponent coordinates relative to the colour's own luminance, and the soft limit of the chroma.
    let ca0 = cine_cbrt_d_plain(xyz.x / (CINE_XN * ys)) - 1.0;
    let cb0 = 1.0 - cine_cbrt_d_plain(xyz.z / (CINE_ZN * ys));
    let chi0 = sqrt(ca0 * ca0 + cb0 * cb0 + CINE_EPS_CHROMA);
    let excess = max(chi0 - CINE_CHI_K, 0.0);
    let limited = min(chi0, CINE_CHI_K)
        + (CINE_CHI_M - CINE_CHI_K) * tanh(min(excess / (CINE_CHI_M - CINE_CHI_K), 20.0));
    let ca = ca0 * (limited / chi0);
    let cb = cb0 * (limited / chi0);
    let chi = sqrt(ca * ca + cb * cb + CINE_EPS_CHROMA);
    let cs = ca / chi;
    let sn = cb / chi;
    // 4. The tone curve of the neutral axis: H = (1 + c Ys^-a)^-b, then the soft clip at the top.
    let u = (ly - CINE_LN_GREY) * CINE_INV_LN2;
    let h = exp(-CINE_B * log(1.0 + CINE_C * exp(-CINE_A * ly)));
    let ln_yt = log(cine_soft_clip(CINE_KAP * h, CINE_WC));
    // 5. Chroma, hue and lightness terms.
    let g0 = CINE_G_PEAK / (1.0 + exp(CINE_GB * ly - CINE_GB_LYC));
    let bh = cine_sigmoid_plain((u - 1.5) * (1.0 / 1.5));
    let env = chi * chi / (chi * chi + CINE_ENV2);
    let gh = env * (cine_harm(cs, sn, CINE_GH_B0_A, CINE_GH_B0_B) + bh * cine_harm(cs, sn, CINE_GH_BH_A, CINE_GH_BH_B));
    let dt = env * (cine_harm(cs, sn, CINE_DT_B0_A, CINE_DT_B0_B) + bh * cine_harm(cs, sn, CINE_DT_BH_A, CINE_DT_BH_B));
    let rh = env * (cine_harm(cs, sn, CINE_RH_B0_A, CINE_RH_B0_B) + bh * cine_harm(cs, sn, CINE_RH_BH_A, CINE_RH_BH_B));
    // chi^rp / (chi^rp + e^(rp r_chi)) = 1 / (1 + e^(rp (r_chi - ln chi))).
    let chenv = 1.0 / (1.0 + exp(CINE_RP * (CINE_R_CHI - log(chi))));
    let lr = chenv * (CINE_R_DARK * cine_sigmoid_plain(-(u - CINE_R_YD)) + CINE_R_HI) + rh;
    // 6. The display colour before the gamut stage.
    let g = g0 * exp(gh);
    let cd = cos(dt);
    let sd = sin(dt);
    let ca2 = g * (ca * cd - cb * sd);
    let cb2 = g * (ca * sd + cb * cd);
    let y2 = exp(ln_yt + lr);
    let t1 = 1.0 + ca2;
    let t2 = 1.0 - cb2;
    // 7. The gamut stage on the output's own primaries, on channels relative to the luminance.
    let yd = max(y2, CINE_Y2_MIN);
    let rel = mv3(CU.cine_m0.xyz, CU.cine_m1.xyz, CU.cine_m2.xyz, vec3<f32>(CINE_XN * t1 * t1 * t1, 1.0, CINE_ZN * t2 * t2 * t2))
        * (y2 / yd);
    let r = max(rel, vec3<f32>(CINE_KL)) + CINE_KL * (exp(min(rel - vec3<f32>(CINE_KL), vec3<f32>(0.0)) / CINE_KL) - vec3<f32>(1.0));
    let v = (1.0 + CINE_KH) * r * yd;
    return vec3<f32>(cine_soft_clip(v.x, CINE_KH), cine_soft_clip(v.y, CINE_KH), cine_soft_clip(v.z, CINE_KH));
}

fn cine_display_linear(awg4: vec3<f32>) -> vec3<f32> {
    if (px_fast()) {
        return cine_display_linear_plain(awg4);
    }
    // 0, 1. The limit and the matrix to XYZ.
    let x = clamp(awg4, vec3<f32>(-CINE_SCENE_LIMIT), vec3<f32>(CINE_SCENE_LIMIT));
    let xyz = px_mv3(AWG4_TO_XYZ_0, AWG4_TO_XYZ_1, AWG4_TO_XYZ_2, x)
        + vec3<f32>(dot(AWG4_TO_XYZ_0_LO, x), dot(AWG4_TO_XYZ_1_LO, x), dot(AWG4_TO_XYZ_2_LO, x));
    // 2. The luminance floor: a smooth maximum of Y and rho times the distance of the chromaticity from the
    // neutral axis. dX = X - XN Y and dZ = Z - ZN Y are linear in x, with the rows DX = row_X - XN row_Y and
    // DZ = row_Z - ZN row_Y (CINE_DX, CINE_DZ, formed in f64 on the host). AWG4's white maps to (XN, 1, ZN), so
    // each row sums to 0 (to 2e-16 in f64) and DX . x = DX . (x - x_g 1): only the differences to the green
    // channel enter, and the green entry is not needed. Taken from X, Y and Z, the two terms of dX are numbers of
    // size |x| whose difference is rounding noise of about 1e-7 |x| for a neutral (1e-2 at -1e5), different
    // for every order of the sums and for every fusion into a multiply-add; yf and then Ys would be that noise.
    // From the differences a neutral gives exactly 0 (x - x is 0, a product or sum of zeros is 0, whatever the
    // compiler does with the products and sums), and any other colour gives dX with an error relative to its
    // own differences.
    let d = vec2<f32>(x.x - x.y, x.z - x.y);
    let dx = dot(CINE_DX, d);
    let dz = dot(CINE_DZ, d);
    let yf = CINE_RHO * sqrt(dx * dx + dz * dz + CINE_EPS_FLOOR);
    let w = CINE_WFRAC * yf;
    let dy = xyz.y - yf;
    let root = sqrt(dy * dy + w * w);
    var smooth_max = 0.5 * (xyz.y + yf + root);
    if (dy < 0.0) {
        smooth_max = yf + 0.5 * w * w / (root - dy);
    }
    let ys = max(smooth_max, CINE_Y_MIN);
    let ly = px_log(ys);
    // 3. Opponent coordinates relative to the colour's own luminance, and the soft limit of the chroma.
    let ca0 = cine_cbrt_d_m1(px_div(xyz.x, px_op(CINE_XN * ys)));
    let cb0 = -cine_cbrt_d_m1(px_div(xyz.z, px_op(CINE_ZN * ys)));
    let chi0 = sqrt(ca0 * ca0 + cb0 * cb0 + CINE_EPS_CHROMA);
    let excess = max(chi0 - CINE_CHI_K, 0.0);
    let limited = min(chi0, CINE_CHI_K)
        + (CINE_CHI_M - CINE_CHI_K) * tanh(min(excess / (CINE_CHI_M - CINE_CHI_K), 20.0));
    let ca = ca0 * (limited / chi0);
    let cb = cb0 * (limited / chi0);
    let chi = sqrt(ca * ca + cb * cb + CINE_EPS_CHROMA);
    let cs = ca / chi;
    let sn = cb / chi;
    // 4. The tone curve of the neutral axis: H = (1 + c Ys^-a)^-b, then the soft clip at the top.
    let u = (ly - CINE_LN_GREY) * CINE_INV_LN2;
    let h = px_exp(-CINE_B * px_log1p(px_op(CINE_C * px_exp(-CINE_A * ly))));
    let ln_yt = px_log(cine_soft_clip(CINE_KAP * h, CINE_WC));
    // 5. Chroma, hue and lightness terms.
    let g0 = px_div(CINE_G_PEAK, px_op(1.0 + px_exp(CINE_GB * ly - CINE_GB_LYC)));
    let bh = cine_sigmoid((u - 1.5) * (1.0 / 1.5));
    let env = chi * chi / (chi * chi + CINE_ENV2);
    let gh = env * (cine_harm(cs, sn, CINE_GH_B0_A, CINE_GH_B0_B) + bh * cine_harm(cs, sn, CINE_GH_BH_A, CINE_GH_BH_B));
    let dt = env * (cine_harm(cs, sn, CINE_DT_B0_A, CINE_DT_B0_B) + bh * cine_harm(cs, sn, CINE_DT_BH_A, CINE_DT_BH_B));
    let rh = env * (cine_harm(cs, sn, CINE_RH_B0_A, CINE_RH_B0_B) + bh * cine_harm(cs, sn, CINE_RH_BH_A, CINE_RH_BH_B));
    // chi^rp / (chi^rp + e^(rp r_chi)) = 1 / (1 + e^(rp (r_chi - ln chi))).
    let chenv = px_div(1.0, px_op(1.0 + px_exp(CINE_RP * (CINE_R_CHI - px_log(chi)))));
    let lr = chenv * (CINE_R_DARK * cine_sigmoid(-(u - CINE_R_YD)) + CINE_R_HI) + rh;
    // 6. The display colour before the gamut stage.
    let g = g0 * px_exp(gh);
    let cd = cos(dt);
    let sd = sin(dt);
    let ca2 = g * (ca * cd - cb * sd);
    let cb2 = g * (ca * sd + cb * cd);
    let y2 = px_exp(ln_yt + lr);
    // 7. The gamut stage on the output's own primaries, on channels relative to the luminance.
    let yd = max(y2, CINE_Y2_MIN);
    // M (XN t1^3, 1, ZN t2^3) with M (XN, 1, ZN) = (1, 1, 1): a channel is 1 + K0 s1 + K2 s2, where s1 = t1^3 - 1 and
    // s2 = t2^3 - 1 come from the small chroma terms ca2 and cb2 (t1 = 1 + ca2 rounded to f32 and cubed is 2e-7 off).
    let s1 = px_op(ca2 * px_op(3.0 + px_op(ca2 * px_op(3.0 + ca2))));
    let s2 = -px_op(cb2 * px_op(3.0 - px_op(cb2 * px_op(3.0 - cb2))));
    var rel = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i++) {
        var row = CU.cine_lo0;
        if (i == 1u) {
            row = CU.cine_lo1;
        } else if (i == 2u) {
            row = CU.cine_lo2;
        }
        let a = px_two_prod(row.x, s1);
        let b = px_two_prod(row.y, s2);
        let sum = px_two_sum(a.x, b.x);
        let w = px_two_sum(1.0, sum.x);
        let lo = px_op(px_op(px_op(a.y + b.y) + sum.y) + px_op(w.y + px_op(px_op(row.z * s1) + px_op(row.w * s2))));
        rel[i] = px_op(w.x + lo) * (y2 / yd);
    }
    let r = max(rel, vec3<f32>(CINE_KL)) + CINE_KL * (vec3<f32>(px_exp(min(rel.x - CINE_KL, 0.0) / CINE_KL), px_exp(min(rel.y - CINE_KL, 0.0) / CINE_KL), px_exp(min(rel.z - CINE_KL, 0.0) / CINE_KL)) - vec3<f32>(1.0));
    let v = (1.0 + CINE_KH) * r * yd;
    return vec3<f32>(cine_soft_clip(v.x, CINE_KH), cine_soft_clip(v.y, CINE_KH), cine_soft_clip(v.z, CINE_KH));
}
