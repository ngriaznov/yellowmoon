// The working encoding of the single-frame denoise (`cpu::denoise_space`) and the exact arithmetic it rests on,
// its BT.709 Y/Cb/Cr and the chroma noise of the photo in it (`cpu::bm_dct::LevelNoise::chroma_sigma`), shared
// by the kernels that read the encoded image: method A (`bm_dct*.wgsl`), the false-colour gate
// (`false_color.wgsl`) and the coarse colour pass (`coarse_chroma.wgsl`). A module that includes this file
// defines `bm_op(x: f32) -> f32`, the OR with its own uniform's zero word (docs/17-gpu.md K8), and the constants
// `kernels::bm_dct` writes from the CPU reference (the knee, slope and gain of the encoding, the sRGB power
// segment, the series coefficients, the BT.709 weights and the squared weights of the chroma noise).
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

// BT.709 Y/Cb/Cr of encoded R, G, B (`cpu::ycbcr::to_ycbcr`): Y in the CPU's order of terms,
// Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748.
fn bm_to_ycc(r: f32, g: f32, b: f32) -> vec3<f32> {
    let y = bm_op(bm_op(bm_op(BM_KR * r) + bm_op(BM_KG * g)) + bm_op(BM_KB * b));
    return vec3<f32>(y, bm_div(bm_op(b - y), BM_CB_DIV), bm_div(bm_op(r - y), BM_CR_DIV));
}

// R, G, B of Y/Cb/Cr (`cpu::ycbcr::from_ycbcr`): R = Y + 1.5748 Cr, B = Y + 1.8556 Cb,
// G = (Y - KR R - KB B) / KG.
fn bm_from_ycc(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    let r = bm_op(y + bm_op(BM_CR_DIV * cr));
    let b = bm_op(y + bm_op(BM_CB_DIV * cb));
    let g = bm_div(bm_op(bm_op(y - bm_op(BM_KR * r)) - bm_op(BM_KB * b)), BM_KG);
    return vec3<f32>(r, g, b);
}

// The correctly rounded square root of `x`, in integer arithmetic only (Metal's fast-math `sqrt` is 1 to 2 ulp
// off): `fc_sqrt` of false_color.wgsl, the method of `sharpen_sqrt` in output_sharpen.wgsl. For a positive
// normal x the mantissa is doubled when the exponent is odd, the integer root of N = mp 2^26 is taken digit by
// digit, and its next bit and remainder decide the rounding to nearest, ties to even. Zero returns itself;
// subnormals, infinities, NaN and negatives take the device's `sqrt`.
fn bm_sqrt(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    if ((bits & 0x7fffffffu) == 0u) {
        return x;
    }
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u || e == 0xffu || (bits >> 31u) != 0u) {
        return sqrt(x);
    }
    let odd = e & 1u;
    let mp = ((bits & 0x7fffffu) | 0x800000u) << odd;
    var rem = 0u;
    var root = 0u;
    for (var k = 0u; k < 26u; k = k + 1u) {
        var two = 0u;
        if (k < 13u) {
            two = (mp >> (24u - 2u * k)) & 3u;
        }
        rem = (rem << 2u) | two;
        let trial = (root << 2u) | 1u;
        if (rem >= trial) {
            rem = rem - trial;
            root = (root << 1u) | 1u;
        } else {
            root = root << 1u;
        }
    }
    var q = root >> 1u;
    var round_bit = root & 1u;
    var sticky = u32(rem != 0u);
    if (odd == 1u) {
        q = root >> 2u;
        round_bit = (root >> 1u) & 1u;
        sticky = sticky | (root & 1u);
    }
    q = q + (round_bit & (sticky | (q & 1u)));
    return bitcast<f32>(((((e + 127u) >> 1u) - 1u) << 23u) + q);
}

// The noise variance of one channel in the working encoding at its encoded value e (`cpu::bm_dct`
// `LevelNoise::chroma_sigma`): the level v = max(srgb_decode(e) / G, 0) (NaN, tested on the bits, read as 0, as
// `f64::max` does), the encoding's slope s = G srgb'(G v), and (g a v + g^2 b) s^2 in the CPU's order of terms.
fn bm_channel_var(e: f32, g: f32, lvl: vec4<f32>) -> f32 {
    let lin = bm_div(bm_srgb_decode(e), lvl.z);
    var v = 0.0;
    if ((bitcast<u32>(lin) & 0x7fffffffu) <= 0x7f800000u && lin > 0.0) {
        v = lin;
    }
    let s = bm_op(lvl.z * bm_srgb_slope(bm_op(lvl.z * v)));
    let n = bm_op(bm_op(bm_op(g * lvl.x) * v) + bm_op(bm_op(g * g) * lvl.y));
    return bm_op(bm_op(n * s) * s);
}

// The variance of channel c (0 R, 1 G, 2 B) at the colour whose mean Y, Cb and Cr (working encoding) is m:
// R, G and B of m (`bm_from_ycc`, as `LevelNoise::chroma_sigma`) and `bm_channel_var`. `lvl` is (a, b, G,
// chroma_scale), `gains` the gains of R, G and B.
fn bm_level_var(m: vec3<f32>, c: u32, lvl: vec4<f32>, gains: vec4<f32>) -> f32 {
    let rgb = bm_from_ycc(m.x, m.y, m.z);
    return bm_channel_var(rgb[c], gains[c], lvl);
}

// tau_C = chroma_scale sigma_C from the variances v of R, G and B: those of Cb and Cr with the squared weights
// (BM_WCB2, BM_WCR2, each square formed in f64 on the host) and sigma_C = sqrt((vb + vr) / 2), the halving
// exact and the root correctly rounded (`LevelNoise::chroma_sigma`).
fn bm_tau_of(v: vec3<f32>, chroma_scale: f32) -> f32 {
    let vb = bm_op(bm_op(bm_op(BM_WCB2.x * v.x) + bm_op(BM_WCB2.y * v.y)) + bm_op(BM_WCB2.z * v.z));
    let vr = bm_op(bm_op(bm_op(BM_WCR2.x * v.x) + bm_op(BM_WCR2.y * v.y)) + bm_op(BM_WCR2.z * v.z));
    return bm_op(chroma_scale * bm_sqrt(bm_op(bm_op(vb + vr) * 0.5)));
}

// The chroma threshold of a reference block whose mean Y, Cb and Cr is m: tau_C = chroma_scale sigma_C(m)
// (`cpu::bm_dct` `Ctx::filter` and `LevelNoise::chroma_sigma`). The pass kernel runs it in two steps, the three
// variances on three threads (`bm_level_var`) and their sum and root on one (`bm_tau_of`); this is the same
// arithmetic in one call, for the tests.
fn bm_chroma_tau(m: vec3<f32>, lvl: vec4<f32>, gains: vec4<f32>) -> f32 {
    let v = vec3<f32>(
        bm_level_var(m, 0u, lvl, gains),
        bm_level_var(m, 1u, lvl, gains),
        bm_level_var(m, 2u, lvl, gains),
    );
    return bm_tau_of(v, lvl.w);
}
