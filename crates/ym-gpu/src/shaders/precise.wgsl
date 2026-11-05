// Accurate elementary functions for the colour pass (F-08d, docs/24 section 3.9c, docs/17-gpu.md K8): log, exp and pow
// from basic operations, each within about one unit in the last place of f32 on every backend. The device's own `log`,
// `exp`, `pow` and `sin` are allowed 3 + 2|x| ulp by Vulkan (up to 45 ulp measured for `exp2`) and are approximate under
// Metal's fast-math, and the output sharpening of an export turns a relative error of 1e-6 in a near-black sample into
// hundreds of codes. The including module defines `px_zero() -> u32`: a uniform word that is 0 at run time, which the
// compiler cannot know.
//
// Fast-math (K8): every product, sum and difference that matters goes through `px_op`, an OR with that word (never an
// XOR: two barriers in a row must stay a barrier). The barrier is the identity on the bits and hides the operation from
// contraction into a multiply-add, reassociation and algebraic rewrites; the error-free transformations (`px_two_sum`,
// `px_two_prod`) are therefore evaluated as written. Divisions are `px_div`: a quotient and its exact remainder, so
// that a reciprocal-based divide is corrected to the correctly rounded value. Nothing here decides a branch by a
// rounding except the choice of the series' interval (the mantissa's bits) and the half-integer split of the exponent
// (a floor of a barrier'd value, whose either choice is the same value within the polynomial's accuracy).

const PX_TWO_OVER_LN2: f32 = 2.8853900817779268;
const PX_ATANH_3: f32 = 0.3333333333333333;
const PX_ATANH_5: f32 = 0.2;
const PX_ATANH_7: f32 = 0.14285714285714285;
const PX_ATANH_9: f32 = 0.1111111111111111;
const PX_EXP2_1: f32 = 0.6931471805599453;
const PX_EXP2_2: f32 = 0.2402265069591007;
const PX_EXP2_3: f32 = 0.05550410866482158;
const PX_EXP2_4: f32 = 0.009618129107628477;
const PX_EXP2_5: f32 = 0.0013333558146428443;
const PX_EXP2_6: f32 = 0.00015403530393381606;
const PX_EXP2_7: f32 = 0.00001525273380405984;
// ln 2 and log2 e as hi + lo (the lo part is the f32 rounding error of the hi part).
const PX_LN2_HI: f32 = 0.693147182464599609375;
const PX_LN2_LO: f32 = -1.904654323148236e-9;
const PX_LOG2E_HI: f32 = 1.44269502162933349609375;
const PX_LOG2E_LO: f32 = 1.92596298909109e-8;

fn px_op(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | px_zero());
}

fn px_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// a + b = s + e exactly (Knuth).
fn px_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = px_op(a + b);
    let bb = px_op(s - a);
    let ad = px_op(s - bb);
    let ea = px_op(a - ad);
    let eb = px_op(b - bb);
    return vec2<f32>(s, px_op(ea + eb));
}

// a = hi + lo with at most 12 significant bits in each (Veltkamp).
fn px_split(a: f32) -> vec2<f32> {
    let c = px_op(4097.0 * a);
    let t = px_op(c - a);
    let hi = px_op(c - t);
    return vec2<f32>(hi, px_op(a - hi));
}

// a b = p + e exactly (Dekker).
fn px_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = px_op(a * b);
    let x = px_split(a);
    let y = px_split(b);
    let e1 = px_op(px_op(x.x * y.x) - p);
    let e2 = px_op(e1 + px_op(x.x * y.y));
    let e3 = px_op(e2 + px_op(x.y * y.x));
    return vec2<f32>(p, px_op(e3 + px_op(x.y * y.y)));
}

// The correctly rounded x / d (the quotient, its exact remainder, the remainder's quotient).
fn px_div(x: f32, d: f32) -> f32 {
    let q = px_op(x / d);
    let p = px_two_prod(q, d);
    let rem = px_op(px_op(x - p.x) - p.y);
    let c1 = px_op(rem / d);
    let p2 = px_two_prod(c1, d);
    let rem2 = px_op(px_op(rem - p2.x) - p2.y);
    return px_op(q + px_op(c1 + px_op(rem2 / d)));
}

// log2(x) = k + small as the pair (k, small) for a positive normal x: the exponent and the series of 2 atanh(s) / ln 2
// with s = (m - 1) / (m + 1) for the mantissa m in [0.707, 1.414]. Anything else takes the device's log2.
fn px_log2_parts(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u || e == 0xffu || (bits >> 31u) != 0u) {
        return vec2<f32>(log2(x), 0.0);
    }
    var k = i32(e) - 127;
    var mbits = (bits & 0x7fffffu) | 0x3f800000u;
    if ((bits & 0x7fffffu) > 0x3504f3u) {
        mbits = (bits & 0x7fffffu) | 0x3f000000u;
        k = k + 1;
    }
    let m = bitcast<f32>(mbits);
    let s = px_div(px_op(m - 1.0), px_op(m + 1.0));
    let s2 = px_op(s * s);
    var q = PX_ATANH_9;
    q = px_op(px_op(s2 * q) + PX_ATANH_7);
    q = px_op(px_op(s2 * q) + PX_ATANH_5);
    q = px_op(px_op(s2 * q) + PX_ATANH_3);
    q = px_op(px_op(s2 * q) + 1.0);
    return vec2<f32>(f32(k), px_op(px_op(PX_TWO_OVER_LN2 * s) * q));
}

// 2^(hi + lo), the sum never rounded.
fn px_exp2(hi: f32, lo: f32) -> f32 {
    if (!px_finite(hi) || !px_finite(lo)) {
        return exp2(hi + lo);
    }
    let nh = floor(hi);
    let t = px_op(px_op(hi - nh) + lo);
    let nt = floor(px_op(t + 0.5));
    let f = px_op(t - nt);
    let n = clamp(i32(nh) + i32(nt), -120, 254);
    let h1 = n / 2;
    let h2 = n - h1;
    let s1 = bitcast<f32>(u32(h1 + 127) << 23u);
    let s2 = bitcast<f32>(u32(h2 + 127) << 23u);
    var p = PX_EXP2_7;
    p = px_op(px_op(f * p) + PX_EXP2_6);
    p = px_op(px_op(f * p) + PX_EXP2_5);
    p = px_op(px_op(f * p) + PX_EXP2_4);
    p = px_op(px_op(f * p) + PX_EXP2_3);
    p = px_op(px_op(f * p) + PX_EXP2_2);
    p = px_op(px_op(f * p) + PX_EXP2_1);
    p = px_op(px_op(f * p) + 1.0);
    return px_op(px_op(p * s1) * s2);
}

// The natural logarithm of a positive normal x.
fn px_log(x: f32) -> f32 {
    let l = px_log2_parts(x);
    let p = px_two_prod(l.x, PX_LN2_HI);
    let tail = px_op(px_op(l.y * PX_LN2_HI) + px_op(l.x * PX_LN2_LO));
    return px_op(p.x + px_op(p.y + tail));
}

// e^x: x log2 e as a pair, so that the exponent carries no rounding of its own.
fn px_exp(x: f32) -> f32 {
    let p = px_two_prod(x, PX_LOG2E_HI);
    let lo = px_op(p.y + px_op(x * PX_LOG2E_LO));
    return px_exp2(p.x, lo);
}

// x^c for a positive x and c = c_hi + c_lo.
fn px_pow(x: f32, c_hi: f32, c_lo: f32) -> f32 {
    let l = px_log2_parts(x);
    let p1 = px_two_prod(l.x, c_hi);
    let p2 = px_two_prod(l.y, c_hi);
    let s = px_two_sum(p1.x, p2.x);
    let tail = px_op(px_op(l.x * c_lo) + px_op(l.y * c_lo));
    let lo = px_op(px_op(px_op(s.y + p1.y) + p2.y) + tail);
    return px_exp2(s.x, lo);
}

// LogC4 of a scene-linear value (ym_color::logc::logc4_encode) with no rounding of the constants and the log carried
// as a pair: A x + 64 as a pair, its log2 as the integer k, the series term and the pair's low part, then
// C + (k - 6) B / 14 + small B / 14 summed from the largest term down. Within an ulp of the f32 result whatever the
// size of x; the toe (x < T) is a single product.
fn px_logc4_encode(x: f32) -> f32 {
    if (x < LC4_T) {
        return px_div(px_op(x - LC4_T), LC4_S);
    }
    if (x > 1.0e30) {
        // Beyond what a camera sample can be (sanitised to 1e30): the products of the pair would leave f32.
        return logc4_encode(x);
    }
    let pa = px_two_prod(LC4P_A, x);
    let v = px_two_sum(pa.x, 64.0);
    let lo = px_op(px_op(v.y + pa.y) + px_op(LC4P_A_LO * x));
    let vv = px_two_sum(v.x, lo);
    let l = px_log2_parts(vv.x);
    let corr = px_op(px_div(vv.y, vv.x) * 1.4426950408889634);
    let kk = l.x - 6.0;
    let small = px_op(l.y + corr);
    let p = px_two_prod(kk, LC4P_BB);
    let r = px_two_sum(LC4P_C, p.x);
    let tail = px_op(px_op(px_op(p.y + px_op(kk * LC4P_BB_LO)) + px_op(small * LC4P_BB)) + px_op(LC4P_C_LO + r.y));
    return px_op(r.x + tail);
}

// 2^(hi + lo) - 1 with the relative accuracy of the result, also where it is small (no cancellation of 2^x - 1): the
// exponent is split as in `px_exp2` into an integer n and f in [-0.5, 0.5], 2^f - 1 = q is the series' own sum
// (Horner without the leading 1), and the result is (2^n - 1) + 2^n q, an exact integer plus one product.
fn px_exp2m1(hi: f32, lo: f32) -> f32 {
    if (!px_finite(hi) || !px_finite(lo) || hi > 120.0 || hi < -120.0) {
        return px_op(px_exp2(hi, lo) - 1.0);
    }
    let nh = floor(hi);
    let t = px_op(px_op(hi - nh) + lo);
    let nt = floor(px_op(t + 0.5));
    let f = px_op(t - nt);
    let n = i32(nh) + i32(nt);
    var p = PX_EXP2_7;
    p = px_op(px_op(f * p) + PX_EXP2_6);
    p = px_op(px_op(f * p) + PX_EXP2_5);
    p = px_op(px_op(f * p) + PX_EXP2_4);
    p = px_op(px_op(f * p) + PX_EXP2_3);
    p = px_op(px_op(f * p) + PX_EXP2_2);
    p = px_op(px_op(f * p) + PX_EXP2_1);
    let q = px_op(f * p);
    let sc = bitcast<f32>(u32(n + 127) << 23u);
    return px_op(px_op(sc - 1.0) + px_op(sc * q));
}

// ym_color::logc::logc4_decode: the exponent w = 14 (e - C) / B as a pair, (2^w - 1) * 64 / A with no cancellation.
fn px_logc4_decode(e: f32) -> f32 {
    if (e < 0.0) {
        return px_op(px_op(e * LC4_S) + LC4_T);
    }
    let d = px_two_sum(e, -LC4P_C);
    let dl = px_op(d.y - LC4P_C_LO);
    let q = px_two_prod(d.x, LC4P_IBB);
    let lo = px_op(px_op(q.y + px_op(d.x * LC4P_IBB_LO)) + px_op(dl * LC4P_IBB));
    let m = px_exp2m1(q.x, lo);
    return px_op(px_op(m * LC4P_64_OVER_A) + px_op(m * LC4P_64_OVER_A_LO));
}

fn px_logc4_encode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(px_logc4_encode(v.x), px_logc4_encode(v.y), px_logc4_encode(v.z));
}

fn px_logc4_decode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(px_logc4_decode(v.x), px_logc4_decode(v.y), px_logc4_decode(v.z));
}

// r . v with every product exact (`px_two_prod`) and the sum carried as an unevaluated pair: one rounding, at the end.
fn px_dot3(r: vec3<f32>, v: vec3<f32>) -> f32 {
    let a = px_two_prod(r.x, v.x);
    let b = px_two_prod(r.y, v.y);
    let c = px_two_prod(r.z, v.z);
    let s1 = px_two_sum(a.x, b.x);
    let s2 = px_two_sum(s1.x, c.x);
    let lo = px_op(px_op(px_op(a.y + b.y) + c.y) + s1.y);
    return px_op(s2.x + px_op(s2.y + lo));
}

fn px_mv3(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(px_dot3(r0, v), px_dot3(r1, v), px_dot3(r2, v));
}

// e^x - 1 with the relative accuracy of the result also for small |x| (no cancellation).
fn px_expm1(x: f32) -> f32 {
    let p = px_two_prod(x, PX_LOG2E_HI);
    let lo = px_op(p.y + px_op(x * PX_LOG2E_LO));
    return px_exp2m1(p.x, lo);
}

// ln(1 + x) for x >= 0 with the relative accuracy of the result also for small x: 2 atanh(x / (2 + x)) as a series below
// 0.5 (s <= 0.2, short by 1e-10), and above that the log of the pair 1 + x.
fn px_log1p(x: f32) -> f32 {
    if (x < 0.5) {
        let s = px_div(x, px_op(2.0 + x));
        let s2 = px_op(s * s);
        var q = 0.09090909090909091;
        q = px_op(px_op(s2 * q) + PX_ATANH_9);
        q = px_op(px_op(s2 * q) + PX_ATANH_7);
        q = px_op(px_op(s2 * q) + PX_ATANH_5);
        q = px_op(px_op(s2 * q) + PX_ATANH_3);
        q = px_op(px_op(s2 * q) + 1.0);
        return px_op(px_op(2.0 * s) * q);
    }
    let z = px_two_sum(1.0, x);
    return px_op(px_log(z.x) + px_div(z.y, z.x));
}
