// P5, colour out (docs/24-film.md 3.8c): the film density to LogC4 / AWG4. T3a (tetrahedral) and the paper's curves
// for a negative (a slide skips them), the print exposure shift, T3b' (the scan, tetrahedral), then the per-pixel
// RETURN of the cine rendering's inverse: the display step in closed form, the start from the core table and a FIXED
// number of damped Newton steps with the exact Jacobian (dual numbers through steps 4 to 6), the nearest-reachable clamp,
// the scene side in closed form, and LogC4. A port of `ym_color::display::cine` (`core_table.rs`, `inverse.rs`) in f32.
// The CPU's data-dependent loops (the Newton iteration for the shared luminance, the cube-root inverse, the luminance
// floor) run a fixed number of iterations here (FX_*_ITERATIONS), and the candidates of a Newton step are tried in the
// CPU's order and kept by the first that lowers the residual (`found`); no loop count depends on the data.
// Every constant is a `CINE_*` of the generated block or an `FX_*` of kernels/film.rs, written from the CPU side.

fn fr_exp(x: f32) -> f32 {
    return exp(clamp(x, -80.0, 80.0));
}

// expm1 without the cancellation of exp(x) - 1 for small |x|.
fn fr_expm1(x: f32) -> f32 {
    if (abs(x) < 0.1) {
        return x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x * (1.0 / 720.0))))));
    }
    return fr_exp(x) - 1.0;
}

// The cube root of a value of either sign.
fn fr_cbrt(x: f32) -> f32 {
    let a = abs(x);
    if (a <= 1.0e-37) {
        return 0.0;
    }
    let r = exp(log(a) * (1.0 / 3.0));
    return select(r, -r, x < 0.0);
}

fn fr_big(x: f32) -> f32 {
    return select(1.0e30, x, !is_nan_f(x) && x < 1.0e30);
}

fn fr_norm_inf(r: vec3<f32>) -> f32 {
    return fr_big(max(abs(r.x), max(abs(r.y), abs(r.z))));
}

fn fr_atanh(t: f32) -> f32 {
    return 0.5 * log((1.0 + t) / (1.0 - t));
}


// ---- the display step (inverse.rs `gamut_inverse`, `display_side`) -----------------------------------------------

fn rt_hold(c: f32) -> f32 {
    if (is_nan_f(c)) {
        return FX_D_MIN;
    }
    return clamp(c, FX_D_MIN, 1.0);
}

// Undoes the soft ceiling of step 7 on one channel.
fn rt_ceiling_inverse(o: f32) -> f32 {
    let kh = CINE_KH;
    let z0 = 1.0 - kh;
    var z = o;
    if (o > z0) {
        // 1 - (o - z0) / kh = (1 - o) / kh with z0 = 1 - kh: written so that the fold at o = 1 (the display white, held
        // there) has no cancellation; the square root of a rounding error of 1e-7 would be 3e-4.
        z = z0 + 2.0 * kh * (1.0 - sqrt(max((1.0 - o) / kh, 0.0)));
    }
    return z / (1.0 + kh);
}

// Display RGB in the output's primaries to (Y2, ca2, cb2). `to_xyz` rows are fu.m[0..3] (the output's RGB to XYZ).
fn rt_display_side(display: vec3<f32>) -> vec3<f32> {
    let kl = CINE_KL;
    let yw = fu.m[1].xyz;
    let v = vec3<f32>(
        rt_ceiling_inverse(rt_hold(display.x)),
        rt_ceiling_inverse(rt_hold(display.y)),
        rt_ceiling_inverse(rt_hold(display.z)),
    );
    let y0 = dot(yw, v);
    var y2 = y0;
    if (v.x < kl * y0 || v.y < kl * y0 || v.z < kl * y0) {
        // Newton in t = ln Y2 on the convex, decreasing g(t) = sum y_i Finv(v_i e^-t) - 1, started to the right of the
        // root, a fixed number of iterations.
        var t = log(y0);
        for (var it = 0u; it < FX_Y2_ITERATIONS; it++) {
            let e = fr_exp(-t);
            var g = -1.0;
            var dg = 0.0;
            for (var c = 0u; c < 3u; c++) {
                let s = v[c] * e;
                if (s >= kl) {
                    g += yw[c] * s;
                    dg -= yw[c] * s;
                } else {
                    g += yw[c] * kl * (1.0 + log(s / kl));
                    dg -= yw[c] * kl;
                }
            }
            t -= g / dg;
        }
        y2 = fr_exp(t);
    }
    var rgb = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let s = v[c] / y2;
        rgb[c] = y2 * select(kl * (1.0 + log(s / kl)), s, s >= kl);
    }
    let xyz = mv3(fu.m[0].xyz, fu.m[1].xyz, fu.m[2].xyz, rgb);
    let y2c = max(xyz.y, CINE_Y2_MIN);
    let ca2 = fr_cbrt(xyz.x / (CINE_XN * y2c)) - 1.0;
    let cb2 = 1.0 - fr_cbrt(xyz.z / (CINE_ZN * y2c));
    return vec3<f32>(y2c, ca2, cb2);
}

// ---- the core, plain and with derivatives -----------------------------------------------------------------------

fn rt_soft_clip(z: f32, w: f32) -> f32 {
    let lo = 1.0 - w;
    let t = clamp(z, lo, 1.0 + w) - lo;
    return min(z - t * t / (4.0 * w), 1.0);
}

fn rt_sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + fr_exp(-x));
}

fn rt_harm(cs: f32, sn: f32, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let c2 = cs * cs - sn * sn;
    let s2 = 2.0 * cs * sn;
    let c3 = cs * c2 - sn * s2;
    let s3 = sn * c2 + cs * s2;
    return dot(vec3<f32>(cs, sn, c2), a) + dot(vec3<f32>(s2, c3, s3), b);
}

// Steps 4 to 6: (ln Ys, ca, cb) to (ln Y2, ca2, cb2).
fn rt_core(q: vec3<f32>) -> vec3<f32> {
    let ly = q.x;
    let ca = q.y;
    let cb = q.z;
    let chi = sqrt(ca * ca + cb * cb + CINE_EPS_CHROMA);
    let cs = ca / chi;
    let sn = cb / chi;
    let u = (ly - CINE_LN_GREY) * CINE_INV_LN2;
    let h = fr_exp(-CINE_B * log(1.0 + CINE_C * fr_exp(-CINE_A * ly)));
    let yt = rt_soft_clip(CINE_KAP * h, CINE_WC);
    let g0 = CINE_G_PEAK / (1.0 + fr_exp(CINE_GB * ly - CINE_GB_LYC));
    let bh = rt_sigmoid((u - 1.5) * (1.0 / 1.5));
    let env = chi * chi / (chi * chi + CINE_ENV2);
    let gh = env * (rt_harm(cs, sn, CINE_GH_B0_A, CINE_GH_B0_B) + bh * rt_harm(cs, sn, CINE_GH_BH_A, CINE_GH_BH_B));
    let dt = env * (rt_harm(cs, sn, CINE_DT_B0_A, CINE_DT_B0_B) + bh * rt_harm(cs, sn, CINE_DT_BH_A, CINE_DT_BH_B));
    let rh = env * (rt_harm(cs, sn, CINE_RH_B0_A, CINE_RH_B0_B) + bh * rt_harm(cs, sn, CINE_RH_BH_A, CINE_RH_BH_B));
    let chenv = 1.0 / (1.0 + fr_exp(CINE_RP * (CINE_R_CHI - log(chi))));
    let lr = chenv * (CINE_R_DARK * rt_sigmoid(-(u - CINE_R_YD)) + CINE_R_HI) + rh;
    let g = g0 * fr_exp(gh);
    let cd = cos(dt);
    let sd = sin(dt);
    return vec3<f32>(log(max(yt, 1.0e-37)) + lr, g * (ca * cd - cb * sd), g * (ca * sd + cb * cd));
}

// A value and its derivatives with respect to the three inputs (real.rs `Dual3`).
struct D3 {
    v: f32,
    d: vec3<f32>,
}

fn d3c(v: f32) -> D3 {
    return D3(v, vec3<f32>(0.0));
}

fn d3in(v: f32, i: u32) -> D3 {
    var d = vec3<f32>(0.0);
    d[i] = 1.0;
    return D3(v, d);
}

fn d3add(a: D3, b: D3) -> D3 {
    return D3(a.v + b.v, a.d + b.d);
}

fn d3sub(a: D3, b: D3) -> D3 {
    return D3(a.v - b.v, a.d - b.d);
}

fn d3mul(a: D3, b: D3) -> D3 {
    return D3(a.v * b.v, a.d * b.v + a.v * b.d);
}

fn d3div(a: D3, b: D3) -> D3 {
    let inv = 1.0 / b.v;
    let q = a.v * inv;
    return D3(q, (a.d - q * b.d) * inv);
}

fn d3neg(a: D3) -> D3 {
    return D3(-a.v, -a.d);
}

fn d3sc(a: D3, k: f32) -> D3 {
    return D3(a.v * k, a.d * k);
}

fn d3exp(a: D3) -> D3 {
    let e = fr_exp(a.v);
    return D3(e, a.d * e);
}

fn d3ln(a: D3) -> D3 {
    let v = max(a.v, 1.0e-37);
    return D3(log(v), a.d / v);
}

fn d3sqrt(a: D3) -> D3 {
    let s = sqrt(max(a.v, 0.0));
    return D3(s, a.d * (0.5 / max(s, 1.0e-30)));
}

fn d3sin(a: D3) -> D3 {
    return D3(sin(a.v), a.d * cos(a.v));
}

fn d3cos(a: D3) -> D3 {
    return D3(cos(a.v), a.d * (-sin(a.v)));
}

fn d3clamp(a: D3, lo: f32, hi: f32) -> D3 {
    if (a.v < lo) {
        return d3c(lo);
    }
    if (a.v > hi) {
        return d3c(hi);
    }
    return a;
}

fn d3min(a: D3, hi: f32) -> D3 {
    if (a.v > hi) {
        return d3c(hi);
    }
    return a;
}

fn d3sigmoid(x: D3) -> D3 {
    return d3div(d3c(1.0), d3add(d3c(1.0), d3exp(d3neg(x))));
}

fn d3soft_clip(z: D3, w: f32) -> D3 {
    let z0 = 1.0 - w;
    let t = d3sub(d3clamp(z, z0, 1.0 + w), d3c(z0));
    return d3min(d3sub(z, d3sc(d3mul(t, t), 1.0 / (4.0 * w))), 1.0);
}

fn d3harm(c: D3, s: D3, a: vec3<f32>, b: vec3<f32>) -> D3 {
    let c2 = d3sub(d3mul(c, c), d3mul(s, s));
    let s2 = d3sc(d3mul(c, s), 2.0);
    let c3 = d3sub(d3mul(c, c2), d3mul(s, s2));
    let s3 = d3add(d3mul(s, c2), d3mul(c, s2));
    var acc = d3sc(c, a.x);
    acc = d3add(acc, d3sc(s, a.y));
    acc = d3add(acc, d3sc(c2, a.z));
    acc = d3add(acc, d3sc(s2, b.x));
    acc = d3add(acc, d3sc(c3, b.y));
    acc = d3add(acc, d3sc(s3, b.z));
    return acc;
}

// The Jacobian of the core at q, one row for each output.
struct Jac {
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
}

fn rt_jac(q: vec3<f32>) -> Jac {
    let ly = d3in(q.x, 0u);
    let ca = d3in(q.y, 1u);
    let cb = d3in(q.z, 2u);
    let chi = d3sqrt(d3add(d3add(d3mul(ca, ca), d3mul(cb, cb)), d3c(CINE_EPS_CHROMA)));
    let cs = d3div(ca, chi);
    let sn = d3div(cb, chi);
    let u = d3sc(d3sub(ly, d3c(CINE_LN_GREY)), CINE_INV_LN2);
    let h = d3exp(d3neg(d3sc(d3ln(d3add(d3c(1.0), d3sc(d3exp(d3neg(d3sc(ly, CINE_A))), CINE_C))), CINE_B)));
    let yt = d3soft_clip(d3sc(h, CINE_KAP), CINE_WC);
    let g0 = d3div(d3c(CINE_G_PEAK), d3add(d3c(1.0), d3exp(d3sub(d3sc(ly, CINE_GB), d3c(CINE_GB_LYC)))));
    let bh = d3sigmoid(d3sc(d3sub(u, d3c(1.5)), 1.0 / 1.5));
    let chi2 = d3mul(chi, chi);
    let env = d3div(chi2, d3add(chi2, d3c(CINE_ENV2)));
    let gh = d3mul(env, d3add(d3harm(cs, sn, CINE_GH_B0_A, CINE_GH_B0_B), d3mul(bh, d3harm(cs, sn, CINE_GH_BH_A, CINE_GH_BH_B))));
    let dt = d3mul(env, d3add(d3harm(cs, sn, CINE_DT_B0_A, CINE_DT_B0_B), d3mul(bh, d3harm(cs, sn, CINE_DT_BH_A, CINE_DT_BH_B))));
    let rh = d3mul(env, d3add(d3harm(cs, sn, CINE_RH_B0_A, CINE_RH_B0_B), d3mul(bh, d3harm(cs, sn, CINE_RH_BH_A, CINE_RH_BH_B))));
    let chenv = d3div(d3c(1.0), d3add(d3c(1.0), d3exp(d3sc(d3sub(d3c(CINE_R_CHI), d3ln(chi)), CINE_RP))));
    let lr = d3add(d3mul(chenv, d3add(d3sc(d3sigmoid(d3neg(d3sub(u, d3c(CINE_R_YD)))), CINE_R_DARK), d3c(CINE_R_HI))), rh);
    let g = d3mul(g0, d3exp(gh));
    let cd = d3cos(dt);
    let sd = d3sin(dt);
    let o0 = d3add(d3ln(yt), lr);
    let o1 = d3mul(g, d3sub(d3mul(ca, cd), d3mul(cb, sd)));
    let o2 = d3mul(g, d3add(d3mul(ca, sd), d3mul(cb, cd)));
    return Jac(o0.d, o1.d, o2.d);
}

// ---- the inverse of a 3 x 3 matrix and the damped Newton steps (core_table.rs `refine`) --------------------------

struct Inv3 {
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
    ok: bool,
}

fn rt_inverse3(a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>) -> Inv3 {
    let c00 = a1.y * a2.z - a1.z * a2.y;
    let c01 = a1.z * a2.x - a1.x * a2.z;
    let c02 = a1.x * a2.y - a1.y * a2.x;
    let det = a0.x * c00 + a0.y * c01 + a0.z * c02;
    if (is_nan_f(det) || abs(det) <= 1.0e-30) {
        return Inv3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), false);
    }
    let id = 1.0 / det;
    return Inv3(
        vec3<f32>(c00, a0.z * a2.y - a0.y * a2.z, a0.y * a1.z - a0.z * a1.y) * id,
        vec3<f32>(c01, a0.x * a2.z - a0.z * a2.x, a0.z * a1.x - a0.x * a1.z) * id,
        vec3<f32>(c02, a0.y * a2.x - a0.x * a2.y, a0.x * a1.y - a0.y * a1.x) * id,
        true,
    );
}

fn rt_refine(q0: vec3<f32>, goal: vec3<f32>, steps: u32) -> vec3<f32> {
    var q = q0;
    var r = rt_core(q) - goal;
    var cost = fr_norm_inf(r);
    for (var s = 0u; s < steps; s++) {
        if (cost < FX_CONVERGED) {
            continue;
        }
        let j = rt_jac(q);
        var found = false;
        var qn = q;
        var rn = r;
        var cn = cost;
        let ji = rt_inverse3(j.r0, j.r1, j.r2);
        if (ji.ok) {
            let step = -vec3<f32>(dot(ji.r0, r), dot(ji.r1, r), dot(ji.r2, r));
            let size = max(abs(step.x), max(abs(step.y), abs(step.z)));
            let scale = select(1.0, FX_STEP_LIMIT / size, size > FX_STEP_LIMIT);
            for (var k = 0u; k < 3u; k++) {
                if (!found) {
                    let sc = scale * select(select(FX_STEP_SCALE_2, FX_STEP_SCALE_1, k == 1u), FX_STEP_SCALE_0, k == 0u);
                    let cand = q + sc * step;
                    let rr = rt_core(cand) - goal;
                    let cc = fr_norm_inf(rr);
                    if (cc < cost) {
                        found = true;
                        qn = cand;
                        rn = rr;
                        cn = cc;
                    }
                }
            }
        }
        if (!found) {
            // Damped least squares: -(J^T J + mu I)^-1 J^T r, a short step along the descent direction.
            let col0 = vec3<f32>(j.r0.x, j.r1.x, j.r2.x);
            let col1 = vec3<f32>(j.r0.y, j.r1.y, j.r2.y);
            let col2 = vec3<f32>(j.r0.z, j.r1.z, j.r2.z);
            let a0 = vec3<f32>(dot(col0, col0), dot(col0, col1), dot(col0, col2));
            let a1 = vec3<f32>(dot(col1, col0), dot(col1, col1), dot(col1, col2));
            let a2 = vec3<f32>(dot(col2, col0), dot(col2, col1), dot(col2, col2));
            let g = vec3<f32>(dot(col0, r), dot(col1, r), dot(col2, r));
            let sc = max(a0.x + a1.y + a2.z, FX_LM_TRACE_FLOOR) / 3.0;
            for (var k = 0u; k < 3u; k++) {
                if (!found) {
                    let mu = select(select(FX_LM_MU_2, FX_LM_MU_1, k == 1u), FX_LM_MU_0, k == 0u) * sc;
                    let inv = rt_inverse3(a0 + vec3<f32>(mu, 0.0, 0.0), a1 + vec3<f32>(0.0, mu, 0.0), a2 + vec3<f32>(0.0, 0.0, mu));
                    if (inv.ok) {
                        let step = clamp(-vec3<f32>(dot(inv.r0, g), dot(inv.r1, g), dot(inv.r2, g)), vec3<f32>(-FX_STEP_LIMIT), vec3<f32>(FX_STEP_LIMIT));
                        let cand = q + step;
                        let rr = rt_core(cand) - goal;
                        let cc = fr_norm_inf(rr);
                        if (cc < cost) {
                            found = true;
                            qn = cand;
                            rn = rr;
                            cn = cc;
                        }
                    }
                }
            }
        }
        if (found) {
            q = qn;
            r = rn;
            cost = cn;
        }
    }
    return q;
}

// ---- the start from the core table (core_table.rs `start`) ------------------------------------------------------

// ln of the scene luminance whose neutral renders to display luminance t (inverse.rs `tone_inverse`, in the log domain:
// the closed form q = h^(1/b), s = (c q / (1 - q))^(1/a) with 1 - q = -expm1(ln q)).
fn rt_ln_tone_inverse(t_in: f32) -> f32 {
    let t = min(t_in, 1.0);
    let z0 = 1.0 - CINE_WC;
    var z = t;
    if (t > z0) {
        z = z0 + 2.0 * CINE_WC * (1.0 - sqrt(max((1.0 - t) / CINE_WC, 0.0)));
    }
    let lq = min(log(max(z / CINE_KAP, 1.0e-37)) / CINE_B, -1.0e-6);
    return (log(CINE_C) + lq - log(-fr_expm1(lq))) / CINE_A;
}

struct Start {
    q: vec3<f32>,
    rho_max: f32,
    rho: f32,
}

fn rt_start(c: vec3<f32>) -> Start {
    let ns = fu.h[0].x;
    let nt = fu.h[0].y;
    let nk = fu.h[0].z;
    let s_lo = fu.p[0].x;
    let s_span = fu.p[0].y;
    let s_full = rt_ln_tone_inverse(c.x);
    let t = clamp((s_full - s_lo) / s_span, 0.0, 1.0);
    let fs = clamp((1.0 - sqrt(1.0 - t)) * f32(ns - 1u), 0.0, f32(ns - 1u));
    let rho = sqrt(c.y * c.y + c.z * c.z);
    var theta = 0.0;
    if (rho > 0.0) {
        theta = atan2(c.z, c.y);
    }
    let ft = (theta + FX_PI) / (2.0 * FX_PI) * f32(nt);
    let tf = floor(ft);
    let j0 = u32(max(tf, 0.0)) % nt;
    let j1 = (j0 + 1u) % nt;
    let i0 = min(u32(fs), ns - 2u);
    let wt = ft - tf;
    let ws = fs - f32(i0);
    var acc = vec3<f32>(0.0);
    var ln_reach = 0.0;
    for (var di = 0u; di < 2u; di++) {
        for (var dj = 0u; dj < 2u; dj++) {
            let a = select(1.0 - ws, ws, di == 1u);
            let b = select(1.0 - wt, wt, dj == 1u);
            let j = select(j0, j1, dj == 1u);
            let ray = (i0 + di) * nt + j;
            let w = a * b;
            let ln_r = reach[ray];
            ln_reach += w * ln_r;
            let u = min(rho / fr_exp(ln_r), 1.0);
            let fk = clamp(f32(nk - 1u) * (1.0 - sqrt(1.0 - u)), 0.0, f32(nk - 1u));
            let k0 = min(u32(fk), nk - 2u);
            let wk = fk - f32(k0);
            let o0 = (ray * nk + k0) * 3u;
            let n0 = vec3<f32>(nodes[o0], nodes[o0 + 1u], nodes[o0 + 2u]);
            let n1 = vec3<f32>(nodes[o0 + 3u], nodes[o0 + 4u], nodes[o0 + 5u]);
            acc += w * (n0 + wk * (n1 - n0));
        }
    }
    return Start(vec3<f32>(acc.x + s_full, acc.y, acc.z), fr_exp(ln_reach), rho);
}

// ---- the scene side, closed form (inverse.rs `scene_side` with strict = false) ----------------------------------

// The inverse of the smooth cube root by Newton on r^3 - P r^2 - P d^2 = 0 from the upper bound.
fn rt_smooth_cbrt_inverse(q: f32) -> f32 {
    let d2 = CINE_DELTA2;
    let aq = abs(q);
    let p = aq * aq * aq / (1.0 + d2);
    var r = p + fr_cbrt(p * d2);
    for (var it = 0u; it < FX_CBRT_ITERATIONS; it++) {
        let h = r * r * (r - p) - p * d2;
        let dh = r * (3.0 * r - 2.0 * p);
        if (dh > 0.0) {
            r -= h / dh;
        }
    }
    return select(r, -r, q < 0.0);
}

fn rt_scene_side(q: vec3<f32>) -> vec3<f32> {
    var ca = q.y;
    var cb = q.z;
    var ln_ys = q.x;
    if (is_nan_f(ca) || is_nan_f(cb) || is_nan_f(ln_ys)) {
        ca = 0.0;
        cb = 0.0;
        ln_ys = 0.0;
    }
    let l = sqrt(ca * ca + cb * cb);
    if (l > FX_CHROMA_CAP) {
        ca *= FX_CHROMA_CAP / l;
        cb *= FX_CHROMA_CAP / l;
    }
    ln_ys = clamp(ln_ys, log(CINE_Y_MIN) + 1.0e-6, FX_LN_YS_MAX);
    let limited = sqrt(ca * ca + cb * cb);
    if (limited > CINE_CHI_K) {
        let t = min((limited - CINE_CHI_K) / (CINE_CHI_M - CINE_CHI_K), 0.99999);
        let chi = CINE_CHI_K + (CINE_CHI_M - CINE_CHI_K) * fr_atanh(t);
        ca *= chi / limited;
        cb *= chi / limited;
    }
    let ym = fr_exp(ln_ys);
    let x = CINE_XN * ym * rt_smooth_cbrt_inverse(1.0 + ca);
    let z = CINE_ZN * ym * rt_smooth_cbrt_inverse(1.0 - cb);
    // The luminance floor, Ys = smoothmax(Y, Yf(Y)), by Newton from Y = Ys (an upper bound of the root).
    var y = ym;
    for (var it = 0u; it < FX_FLOOR_ITERATIONS; it++) {
        let dx = x - CINE_XN * y;
        let dz = z - CINE_ZN * y;
        let sq = sqrt(dx * dx + dz * dz + CINE_EPS_FLOOR);
        let yf = CINE_RHO * sq;
        let dyf = -CINE_RHO * (CINE_XN * dx + CINE_ZN * dz) / sq;
        let w = CINE_WFRAC * yf;
        let dw = CINE_WFRAC * dyf;
        let rt = sqrt((y - yf) * (y - yf) + w * w);
        let g = 0.5 * (y + yf + rt) - ym;
        var dg = 0.5 * (1.0 + dyf);
        if (rt > 0.0) {
            dg = 0.5 * (1.0 + dyf + ((y - yf) * (1.0 - dyf) + w * dw) / rt);
        }
        if (abs(dg) < 1.0e-6) {
            dg = select(1.0e-6, -1.0e-6, dg < 0.0);
        }
        let lim = 0.5 * abs(y) + 1.0e-12;
        y -= clamp(g / dg, -lim, lim);
    }
    return mv3(XYZ_TO_AWG4_0, XYZ_TO_AWG4_1, XYZ_TO_AWG4_2, vec3<f32>(x, y, z));
}

// The whole return: a display colour to LogC4 / AWG4 (bake.rs `return_to_logc4`).
fn rt_return(display: vec3<f32>) -> vec3<f32> {
    let c = rt_display_side(display);
    let st = rt_start(c);
    var q = st.q;
    if (st.rho <= st.rho_max) {
        // Inside the reachable set of the ray: the Newton steps from the table's start.
        q = rt_refine(st.q, vec3<f32>(log(max(c.x, 1.0e-37)), c.y, c.z), fu.op.w);
    }
    // Beyond the boundary of the ray the pre-image of its end is the start itself (the nearest reachable colour).
    return logc4_encode3(rt_scene_side(q));
}

// ---- the pass ---------------------------------------------------------------------------------------------------

fn pc_curve(ch: u32, x: f32) -> f32 {
    let n = FX_CURVE_ENTRIES;
    let lo = fu.p[5][ch];
    let k = fu.p[6][ch];
    let pos = fx_grid((x - lo) * k, n);
    let i = min(u32(floor(pos)), n - 2u);
    let t = pos - f32(i);
    let a = curves[ch * n + i];
    let b = curves[ch * n + i + 1u];
    return a + (b - a) * t;
}

// The paper curve of channel `ch` at x_hi + x_lo, as a pair (the position and the interpolation carry their rounding).
fn pc_curve_pair(ch: u32, xh: f32, xl: f32) -> vec2<f32> {
    let n = FX_CURVE_ENTRIES;
    let d = px_two_sum(xh, -fu.p[5][ch]);
    let dl = px_op(px_op(d.y + xl) - fu.p[13][ch]);
    let m = px_two_prod(d.x, fu.p[6][ch]);
    var ph = m.x;
    var pl = px_op(px_op(m.y + px_op(d.x * fu.p[14][ch])) + px_op(dl * fu.p[6][ch]));
    if (is_nan_f(ph) || ph <= 0.0) {
        ph = 0.0;
        pl = 0.0;
    } else if (ph >= f32(n - 1u)) {
        ph = f32(n - 1u);
        pl = 0.0;
    }
    var i = min(u32(floor(ph)), n - 2u);
    var t = px_op(px_op(ph - f32(i)) + pl);
    if (t < 0.0 && i > 0u) {
        i = i - 1u;
        t = px_op(t + 1.0);
    }
    let a = curves[ch * n + i];
    let b = curves[ch * n + i + 1u];
    let df = px_two_sum(b, -a);
    let pr = px_two_prod(df.x, t);
    let s = px_two_sum(a, pr.x);
    return px_two_sum(s.x, px_op(s.y + px_op(pr.y + px_op(df.y * t))));
}

fn rt_main(gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if (ix >= fu.win.z || iy >= fu.win.w) {
        return;
    }
    let x = fu.win.x + ix;
    let y = fu.win.y + iy;
    let d = vec3<f32>(src[fx_a(x, y, 0u)], src[fx_a(x, y, 1u)], src[fx_a(x, y, 2u)]);
    let t3b_lo = fu.p[3].xyz;
    let t3b_hi = fu.p[4].xyz;
    var display = vec3<f32>(0.0);
    if (px_fast()) {
        // A view: the plain f32 tables, the paper's curves and the return (the arithmetic before F-08d), and no shadow (the
        // colour pass reads none).
        var shown = vec3<f32>(0.0);
        if ((fu.op.z & 1u) != 0u) {
            shown = rt_t3b(fu.h[1].y, t3b_lo, t3b_hi, fu.p[3].w, d);
        } else {
            let lp = rt_t3a(fu.h[1].x, fu.p[1].xyz, fu.p[2].xyz, fu.p[1].w, d) + vec3<f32>(fu.p[0].z);
            let dp = vec3<f32>(pc_curve(0u, lp.x), pc_curve(1u, lp.y), pc_curve(2u, lp.z));
            shown = rt_t3b(fu.h[1].y, t3b_lo, t3b_hi, fu.p[3].w, dp);
        }
        let plain = rt_return(shown);
        for (var c = 0u; c < 3u; c++) {
            dst[fx_b(x, y, c)] = plain[c];
        }
        return;
    }
    var dh = d;
    var dl = vec3<f32>(0.0);
    if ((fu.op.z & 1u) == 0u) {
        // A negative: T3a, the print exposure shift and the paper's curves, each carried as a pair.
        let ta = rt_t3a_pair(fu.h[1].x, fu.p[1].xyz, fu.p[7].xyz, fu.p[8].xyz, fu.p[9].xyz, fu.p[1].xyz, fu.p[2].xyz, fu.p[1].w, d, vec3<f32>(0.0));
        var lh = vec3<f32>(0.0);
        var ll = vec3<f32>(0.0);
        for (var c = 0u; c < 3u; c++) {
            let s = px_two_sum(ta.hi[c], fu.p[0].z);
            lh[c] = s.x;
            ll[c] = px_op(px_op(s.y + ta.lo[c]) + fu.p[0].w);
        }
        for (var c = 0u; c < 3u; c++) {
            let v = pc_curve_pair(c, lh[c], ll[c]);
            dh[c] = v.x;
            dl[c] = v.y;
        }
    }
    let tb = rt_t3b_pair(fu.h[1].y, t3b_lo, fu.p[10].xyz, fu.p[11].xyz, fu.p[12].xyz, t3b_lo, t3b_hi, fu.p[3].w, dh, dl);
    display = tb.hi + tb.lo;
    let out = rt_return(display);
    for (var c = 0u; c < 3u; c++) {
        dst[fx_b(x, y, c)] = out[c];
        // The display colour the return was asked for, as the display step holds it: what the cine rendering gives back for
        // a reachable colour (the colour pass uses it where this pixel's LogC4 reaches C12 unchanged).
        shadow[fx_c(x, y, c)] = rt_hold(display[c]);
    }
}
