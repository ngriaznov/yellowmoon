// The working encoding of the single-frame denoise (`cpu::denoise_space`) and the exact arithmetic it rests on,
// shared by the kernels that read the encoded image: method A (`bm_dct*.wgsl`) and the false-colour gate
// (`false_color.wgsl`). A module that includes this file defines `bm_op(x: f32) -> f32`, the OR with its own
// uniform's zero word (docs/17-gpu.md K8), and the constants `kernels::bm_dct` writes from the CPU reference (the
// knee, slope and gain of the encoding, the sRGB power segment, the series coefficients).
//
// Fast-math (Metal): every product, sum and difference is behind `bm_op`, every division is correctly rounded
// (`bm_div`), and the power is our own (`bm_pow`: log2 and exp2 from basic operations behind the barrier) rather
// than the device's `pow`, so the arithmetic is one IEEE f32 sequence on every backend.

// Finite, tested on the bits so that fast-math cannot remove the test.
fn bm_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// a + b = s + e exactly (Knuth), every rounded result and error term hidden.
fn bm_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = bm_op(a + b);
    let bb = bm_op(s - a);
    let ad = bm_op(s - bb);
    let ea = bm_op(a - ad);
    let eb = bm_op(b - bb);
    return vec2<f32>(s, bm_op(ea + eb));
}

// a = hi + lo with at most 12 significant bits in each (Veltkamp).
fn bm_split(a: f32) -> vec2<f32> {
    let c = bm_op(4097.0 * a);
    let t = bm_op(c - a);
    let hi = bm_op(c - t);
    return vec2<f32>(hi, bm_op(a - hi));
}

// a b = p + e exactly (Dekker); the partial products of the 12-bit halves are exact in f32.
fn bm_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = bm_op(a * b);
    let x = bm_split(a);
    let y = bm_split(b);
    let e1 = bm_op(bm_op(x.x * y.x) - p);
    let e2 = bm_op(e1 + bm_op(x.x * y.y));
    let e3 = bm_op(e2 + bm_op(x.y * y.x));
    return vec2<f32>(p, bm_op(e3 + bm_op(x.y * y.y)));
}

// The correctly rounded f32 of x / d whatever the accuracy of the driver's division (Metal's fast-math one is
// a reciprocal product): the quotient, its exact remainder and the remainder's quotient refined once
// (`div_cr` of guided.wgsl).
fn bm_div(x: f32, d: f32) -> f32 {
    let q = bm_op(x / d);
    let p = bm_two_prod(q, d);
    let rem = bm_op(bm_op(x - p.x) - p.y);
    let c1 = bm_op(rem / d);
    let p2 = bm_two_prod(c1, d);
    let rem2 = bm_op(bm_op(rem - p2.x) - p2.y);
    return bm_op(q + bm_op(c1 + bm_op(rem2 / d)));
}

// log2(x) = k + small for a positive normal x, as the pair (k, small): the exponent, and the series of
// 2 atanh(s) / ln 2 with s = (m - 1) / (m + 1) for the mantissa m in [0.707, 1.414] (`lg_log2` of
// log_guided.wgsl, with the division correctly rounded). Kept apart so the caller multiplies them without
// rounding their sum. Anything else takes the device's log2.
fn bm_log2_parts(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u || e == 0xffu || (bits >> 31u) != 0u) {
        return vec2<f32>(log2(x), 0.0);
    }
    var k = i32(e) - 127;
    var mbits = (bits & 0x7fffffu) | 0x3f800000u;
    if ((bits & 0x7fffffu) > 0x3504f3u) {
        // m above sqrt(2): halve it.
        mbits = (bits & 0x7fffffu) | 0x3f000000u;
        k = k + 1;
    }
    let m = bitcast<f32>(mbits);
    let s = bm_div(bm_op(m - 1.0), bm_op(m + 1.0));
    let s2 = bm_op(s * s);
    var q = BM_ATANH_9;
    q = bm_op(bm_op(s2 * q) + BM_ATANH_7);
    q = bm_op(bm_op(s2 * q) + BM_ATANH_5);
    q = bm_op(bm_op(s2 * q) + BM_ATANH_3);
    q = bm_op(bm_op(s2 * q) + 1.0);
    return vec2<f32>(f32(k), bm_op(bm_op(BM_TWO_OVER_LN2 * s) * q));
}

// 2^(hi + lo) from basic operations, the sum never rounded (`lg_exp2` of log_guided.wgsl): hi - floor(hi) is
// exact, lo is added to that fraction, the result is split into an integer n and f in [-0.5, 0.5], 2^f is the
// Taylor polynomial of degree 7 and 2^n is built from its exponent bits in two halves (exact). A non-finite
// part takes the device's exp2.
fn bm_exp2(hi: f32, lo: f32) -> f32 {
    if (!bm_finite(hi) || !bm_finite(lo)) {
        return exp2(hi + lo);
    }
    let nh = floor(hi);
    let t = bm_op(bm_op(hi - nh) + lo);
    let nt = floor(bm_op(t + 0.5));
    let f = bm_op(t - nt);
    let n = clamp(i32(nh) + i32(nt), -120, 254);
    let h1 = n / 2;
    let h2 = n - h1;
    let s1 = bitcast<f32>(u32(h1 + 127) << 23u);
    let s2 = bitcast<f32>(u32(h2 + 127) << 23u);
    var p = BM_EXP2_7;
    p = bm_op(bm_op(f * p) + BM_EXP2_6);
    p = bm_op(bm_op(f * p) + BM_EXP2_5);
    p = bm_op(bm_op(f * p) + BM_EXP2_4);
    p = bm_op(bm_op(f * p) + BM_EXP2_3);
    p = bm_op(bm_op(f * p) + BM_EXP2_2);
    p = bm_op(bm_op(f * p) + BM_EXP2_1);
    p = bm_op(bm_op(f * p) + 1.0);
    return bm_op(bm_op(p * s1) * s2);
}

// x^(c_hi + c_lo) for a positive x: exp2((k + small) (c_hi + c_lo)) with the product formed as a pair
// (`bm_two_prod`, `bm_two_sum`), so the exponent carries no rounding of its own; within about one ulp.
fn bm_pow(x: f32, c_hi: f32, c_lo: f32) -> f32 {
    let l = bm_log2_parts(x);
    let p1 = bm_two_prod(l.x, c_hi);
    let p2 = bm_two_prod(l.y, c_hi);
    let s = bm_two_sum(p1.x, p2.x);
    let tail = bm_op(bm_op(l.x * c_lo) + bm_op(l.y * c_lo));
    let lo = bm_op(bm_op(bm_op(s.y + p1.y) + p2.y) + tail);
    return bm_exp2(s.x, lo);
}

// The working encoding of a balanced linear sample, srgb(2^gain x) with the continuous knee
// (`cpu::denoise_space::encode`).
fn bm_encode(x: f32) -> f32 {
    let v = bm_op(x * BM_GAIN);
    if (v < BM_KNEE_X) {
        return bm_op(BM_SLOPE * v);
    }
    return bm_op(bm_op(BM_SRGB_A * bm_pow(v, BM_INV_GAMMA_HI, BM_INV_GAMMA_LO)) - BM_SRGB_B);
}

// The inverse of the sRGB curve on the whole line (`cpu::denoise_space::srgb_decode`).
fn bm_srgb_decode(y: f32) -> f32 {
    if (y < BM_KNEE_Y) {
        return bm_div(y, BM_SLOPE);
    }
    return bm_pow(bm_div(bm_op(y + BM_SRGB_B), BM_SRGB_A), BM_GAMMA_HI, BM_GAMMA_LO);
}

// The inverse of `bm_encode` (`cpu::denoise_space::decode`).
fn bm_decode(y: f32) -> f32 {
    return bm_div(bm_srgb_decode(y), BM_GAIN);
}

// The slope of the sRGB curve at v (`cpu::denoise_space::srgb_slope`): the linear segment's below the knee,
// 1.055 / 2.4 v^(1 / 2.4 - 1) above.
fn bm_srgb_slope(v: f32) -> f32 {
    if (v < BM_KNEE_X) {
        return BM_SLOPE;
    }
    return bm_op(BM_SRGB_DA * bm_pow(v, BM_DEXP_HI, BM_DEXP_LO));
}
