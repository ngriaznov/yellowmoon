// RA-09: log-guided fast demosaic, the all-float form of docs/algorithms/02-log-guided-demosaic.md
// section 3.9 (CPU reference: ym_gpu::cpu::demosaic::log_guided::demosaic_float).
//
// One entry point, four dispatches (the spec's section 5 split), selected by `u.ctl.x`:
//   0  Lh  = horizontal 5-tap guide filter of L = log2(p + A / 2), the log taken on the fly at +-2
//   1  C   = 2^max(vertical 5-tap filter of Lh, 0), the guide
//   2  SH, SV = horizontal 5-sums of the structure measures Dh, Dv (taken from C on the fly)
//   3  vertical 5-sums, box means, structure weights and the guide-ratio reconstruction
// Every dispatch is per pixel over a small neighbourhood; there are no reductions and no barriers,
// so the result does not depend on the order in which threads run.
//
// Borders: instead of padding the mosaic once by 5 px (spec 3.0) every read of an input or an
// intermediate plane is mirrored into the buffer (`lg_mir`, which is `mirror_idx`: period 2 (n - 1),
// edge sample not repeated, Bayer parity kept). Every stage is symmetric under reflection, so this
// is the same function (spec 3.0: "mirroring each intermediate plane at the buffer edge instead
// gives identical results"). The receptive field is 5 px, as the spec says (tested).
//
// Precision: f32 everywhere. The intermediate planes (Lh, C, SH, SV) stay f32 whatever the storage
// mode of the device is; only the mosaic (input) and the three output planes use ST_ELEM, the
// storage type of the other raw kernels. The green blend is evaluated in the spec's f32-friendly
// form (3.9: the cube-weighted blend divided through by a weight), divided through by the LARGER of
// the two weights so that the remaining weight is (smaller / larger)^3 <= 1 and nothing overflows;
// when D <= E this is the spec's t = (D / E)^3 form exactly. The guide filters and the second
// differences are written about the centre value (exact rewrites, see the passes), so that f32
// rounding is relative to the differences of neighbouring values, not to the values themselves.
//
// The numbers that are not typed here: the offset p, the kernel taps K / 1024 and the working
// scale S = 65535 arrive in the uniform, computed by the Rust side from the same code as the CPU
// reference. The coefficients of `lg_log2` and `lg_exp2` are mathematical (1 / (2j + 1), 2 / ln 2,
// ln(2)^k / k!), checked against their definitions by a unit test of `kernels/log_guided.rs`.
//
// Fast-math (K8). Metal compiles with fast-math: `log2` is up to 3 ulp off (MSL), and one ulp of
// L = log2(p + A / 2) near 116 (a sample of 1e30 on the working scale) is a relative 5e-6 of the
// guide C = 2^L, which the reconstruction's ratios pass on (Metal at b6f2e4b: 1.06e-5 of the peak at
// a scale of 1e30, growing with the scale; the bound is 1e-5). So the logarithm and the power are this
// file's own, from basic operations (`lg_log2`, `lg_exp2`), and the power takes the guide filter's
// sum as two parts, the centre value and the small filter term, so that the sum is never rounded to
// an ulp of 116 first (the CPU sums in f64). Every partial sum, difference and product of the filters,
// the structure measures and the reconstruction passes through `lg_opaque`, so the compiler can
// neither fuse nor re-associate them (the second differences, written as differences of differences
// to stay exact, would otherwise be folded back into `(a + b) - 2 c`). A C port of this file compiled
// with clang -O2 and -O3 -ffast-math -ffp-contract=fast -mfma, with `log2` 2 ulp and a reciprocal 1
// ulp off, gives at most 1.8e-7 of the peak on the stripes at every scale (docs/gpu-plan.md
// section 12).

struct U {
    dims: vec4<u32>,   // x: width, y: height, z: stride (even), w: plane length (stride * height)
    cfa: vec4<u32>,    // colour (0 R, 1 G, 2 B) at local position 2 * (y & 1) + (x & 1)
    taps: vec4<f32>,   // guide kernel K / 1024: x: outer tap k, y: inner tap, z: centre tap
    misc: vec4<f32>,   // x: p (log offset and structure floor), y: S (working scale)
    ctl: vec4<u32>,    // x: pass, y: always 0 (the optimiser cannot know it, see `lg_opaque`)
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> mosaic: array<ST_ELEM>;
// Four f32 planes of `dims.w` samples: Lh, C, SH, SV.
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> rgb: array<ST_ELEM>;

//@ACCESSORS mosaic ro
//@ACCESSORS rgb

// The series of lg_log2: 2 / ln 2 and 1 / (2j + 1); the polynomial of lg_exp2: ln(2)^k / k!.
const LG_TWO_OVER_LN2: f32 = 2.8853900817779268;
const LG_ATANH_3: f32 = 0.3333333333333333;
const LG_ATANH_5: f32 = 0.2;
const LG_ATANH_7: f32 = 0.14285714285714285;
const LG_ATANH_9: f32 = 0.1111111111111111;
const LG_EXP2_1: f32 = 0.6931471805599453;
const LG_EXP2_2: f32 = 0.2402265069591007;
const LG_EXP2_3: f32 = 0.05550410866482158;
const LG_EXP2_4: f32 = 0.009618129107628477;
const LG_EXP2_5: f32 = 0.0013333558146428443;
const LG_EXP2_6: f32 = 0.00015403530393381606;
const LG_EXP2_7: f32 = 0.00001525273380405984;

// Hides a value from algebraic rewriting: an or with `u.ctl.y`, which is always 0 and which the
// compiler cannot know. A shader compiler that runs with fast-math (Metal's does by default:
// reassociation, reciprocal rewrites) turns `x * (a / b)` into `(x * a) / b`, and with `x` and `a` both
// on the 16-bit working scale of a very bright frame (A up to 2e35) the product overflows even though
// every factor and the result are finite. The quotients below are ratios of two values of one scale, close
// to 1, and a barrier after each keeps them one rounded value that no rewrite can fold into a product
// of the large factors. Nothing else changes: the barrier is the identity on the bits.
//
// An OR, not an XOR: two barriers in a row must stay a barrier, and an XOR applied twice cancels
// (x ^ z ^ z == x, which compilers fold; fast-math folds `0 + x` to x, so a running sum that starts at
// 0 gives two barriers in a row). An OR is idempotent.
fn lg_opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | u.ctl.y);
}

// log2(x) of a positive normal x from basic operations: the exponent plus (2 / ln 2) atanh(s) with
// s = (m - 1) / (m + 1) and the mantissa m in [sqrt(1/2), sqrt(2)], so |s| <= 0.1716 and the odd
// series to s^9 is short by under 1e-9. Within an ulp of the result where it is large, within 4e-9
// where it is small (measured against f64 on 10^8 arguments). Zero, subnormals, infinities and NaN
// take the device's `log2` (the argument p + A / 2 is at least 1 here).
fn lg_log2(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u || e == 0xffu || (bits >> 31u) != 0u) {
        return log2(x);
    }
    var k = i32(e) - 127;
    var mbits = (bits & 0x7fffffu) | 0x3f800000u;
    if ((bits & 0x7fffffu) > 0x3504f3u) {
        // m above sqrt(2): halve it.
        mbits = (bits & 0x7fffffu) | 0x3f000000u;
        k = k + 1;
    }
    let m = bitcast<f32>(mbits);
    let s = lg_opaque(lg_opaque(m - 1.0) / lg_opaque(m + 1.0));
    let s2 = lg_opaque(s * s);
    var q = LG_ATANH_9;
    q = lg_opaque(lg_opaque(s2 * q) + LG_ATANH_7);
    q = lg_opaque(lg_opaque(s2 * q) + LG_ATANH_5);
    q = lg_opaque(lg_opaque(s2 * q) + LG_ATANH_3);
    q = lg_opaque(lg_opaque(s2 * q) + 1.0);
    let small = lg_opaque(lg_opaque(LG_TWO_OVER_LN2 * s) * q);
    return lg_opaque(f32(k) + small);
}

// 2^(hi + lo) from basic operations, with the sum never rounded: hi - floor(hi) is exact, the small
// part is added to that fraction, and the result is split into an integer n and f in [-0.5, 0.5]
// (both steps exact). 2^f is the Taylor polynomial of degree 7 (short by under 6e-9 relative), 2^n is
// built from its exponent bits in two halves (exact; a result beyond the f32 range overflows to
// infinity in the last product, as the CPU's cast does). Within 1.7 ulp of 2^(hi + lo) in f64
// (measured on 10^8 arguments; the device's exp2 of the rounded sum: up to 45 ulp in IEEE). A
// non-finite part takes the device's `exp2`.
fn lg_exp2(hi: f32, lo: f32) -> f32 {
    if (!lg_is_finite(hi) || !lg_is_finite(lo)) {
        return exp2(hi + lo);
    }
    let nh = floor(hi);
    let t = lg_opaque(lg_opaque(hi - nh) + lo);
    let nt = floor(lg_opaque(t + 0.5));
    let f = lg_opaque(t - nt);
    let n = clamp(i32(nh) + i32(nt), -120, 254);
    let h1 = n / 2;
    let h2 = n - h1;
    let s1 = bitcast<f32>(u32(h1 + 127) << 23u);
    let s2 = bitcast<f32>(u32(h2 + 127) << 23u);
    var p = LG_EXP2_7;
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_6);
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_5);
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_4);
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_3);
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_2);
    p = lg_opaque(lg_opaque(f * p) + LG_EXP2_1);
    p = lg_opaque(lg_opaque(f * p) + 1.0);
    return lg_opaque(lg_opaque(p * s1) * s2);
}

// Infinity and NaN, tested on the bits so that fast-math cannot remove the test.
fn lg_is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// A non-finite sample counts as 0 (spec section 4).
fn lg_finite_or_zero(x: f32) -> f32 {
    return select(0.0, x, lg_is_finite(x));
}

// `mirror_idx` with a fast path: only the few reads past an edge need the modulo.
fn lg_mir(i: i32, n: i32) -> i32 {
    if (i >= 0 && i < n) {
        return i;
    }
    return mirror_idx(i, n);
}

// The mosaic sample at (x, y) of the mirrored buffer.
fn lg_mosaic(x: i32, y: i32) -> f32 {
    let w = i32(u.dims.x);
    let h = i32(u.dims.y);
    return lg_finite_or_zero(mosaic_ld(u32(lg_mir(y, h)) * u.dims.z + u32(lg_mir(x, w))));
}

// A = S max(v, 0): the sample on the 16-bit working scale (spec 3.9, step 3.1). No upper clamp.
fn lg_a(x: i32, y: i32) -> f32 {
    return u.misc.y * max(lg_mosaic(x, y), 0.0);
}

// Sample (x, y) of scratch plane `k` (0 Lh, 1 C, 2 SH, 3 SV); x and y are mirrored into the buffer.
fn lg_plane(k: u32, x: i32, y: i32) -> f32 {
    let w = i32(u.dims.x);
    let h = i32(u.dims.y);
    return scratch[k * u.dims.w + u32(lg_mir(y, h)) * u.dims.z + u32(lg_mir(x, w))];
}

fn lg_c(x: i32, y: i32) -> f32 {
    return lg_plane(1u, x, y);
}

// Pass 0: Lh(x, y) = sum_i K_i / 1024 * L(x - 2 + i, y) with L = log2(p + A / 2).
//
// The taps sum to exactly 1, so the filter is evaluated about the centre value,
// Lh = Lc + sum_{i != c} t_i (L_i - Lc). It is the same function in real arithmetic; in f32 it keeps
// the rounding error relative to the differences (small where the guide is smooth, which is where
// the structure measures below come out close to p) instead of relative to L itself (about 14),
// and a guide that is flat or symmetric about the centre comes out exactly flat or symmetric.
// L at (x, y) of the mirrored buffer: log2(p + A / 2), the argument pinned.
fn lg_l(x: i32, y: i32) -> f32 {
    return lg_log2(lg_opaque(u.misc.x + lg_opaque(0.5 * lg_a(x, y))));
}

fn lg_pass_lh(x: i32, y: i32) {
    let lc = lg_l(x, y);
    var acc = lg_opaque(u.taps.x * lg_opaque(lg_l(x - 2, y) - lc));
    acc = lg_opaque(acc + lg_opaque(u.taps.y * lg_opaque(lg_l(x - 1, y) - lc)));
    acc = lg_opaque(acc + lg_opaque(u.taps.y * lg_opaque(lg_l(x + 1, y) - lc)));
    acc = lg_opaque(acc + lg_opaque(u.taps.x * lg_opaque(lg_l(x + 2, y) - lc)));
    scratch[u32(y) * u.dims.z + u32(x)] = lc + acc;
}

// Pass 1: C(x, y) = 2^max(sum_j K_j / 1024 * Lh(x, y - 2 + j), 0), evaluated about the centre value
// as in pass 0, the power taken of the two parts (`lg_exp2`). The lower clamp mirrors the fixed form
// (C >= 1, which also keeps every later denominator at least 1).
fn lg_pass_c(x: i32, y: i32) {
    let lc = lg_plane(0u, x, y);
    var acc = lg_opaque(u.taps.x * lg_opaque(lg_plane(0u, x, y - 2) - lc));
    acc = lg_opaque(acc + lg_opaque(u.taps.y * lg_opaque(lg_plane(0u, x, y - 1) - lc)));
    acc = lg_opaque(acc + lg_opaque(u.taps.y * lg_opaque(lg_plane(0u, x, y + 1) - lc)));
    acc = lg_opaque(acc + lg_opaque(u.taps.x * lg_opaque(lg_plane(0u, x, y + 2) - lc)));
    var c = 1.0;
    if (lg_opaque(lc + acc) > 0.0) {
        c = lg_exp2(lc, acc);
    }
    scratch[u.dims.w + u32(y) * u.dims.z + u32(x)] = c;
}

// Pass 2: the horizontal 5-sums, over columns x - 2 .. x + 2, of the structure measures
//   Dh = |(C(x - 1) + C(x + 1)) / 2 - C(x)| + p   and   Dv = |(C(y - 1) + C(y + 1)) / 2 - C(y)| + p.
// Each second difference is written as |((C_a - C) + (C_b - C)) / 2|: a difference of two f32
// values within a factor of two of each other is exact, so where the guide is smooth (the measure
// close to p) the result carries no rounding error of the size of C (up to 1e5 here, an ulp of
// 0.008). The measures beyond the buffer edge are those of the mirrored column, computed from the
// mirrored guide. The guide values are read once into locals: 7 of the row and 5 above and below.
fn lg_pass_structure(x: i32, y: i32) {
    let p = u.misc.x;
    var mid: array<f32, 7>;
    var up: array<f32, 5>;
    var down: array<f32, 5>;
    for (var i = 0; i < 7; i = i + 1) {
        mid[i] = lg_c(x - 3 + i, y);
    }
    for (var i = 0; i < 5; i = i + 1) {
        up[i] = lg_c(x - 2 + i, y - 1);
        down[i] = lg_c(x - 2 + i, y + 1);
    }
    var sh = 0.0;
    var sv = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let c = mid[i + 1];
        sh = lg_opaque(sh + lg_opaque(abs(0.5 * lg_opaque(lg_opaque(mid[i] - c) + lg_opaque(mid[i + 2] - c))) + p));
        sv = lg_opaque(sv + lg_opaque(abs(0.5 * lg_opaque(lg_opaque(up[i] - c) + lg_opaque(down[i] - c))) + p));
    }
    let i = u32(y) * u.dims.z + u32(x);
    scratch[2u * u.dims.w + i] = sh;
    scratch[3u * u.dims.w + i] = sv;
}

// Pass 3, one pixel: the vertical 5-sums, the box means and the reconstruction of spec 3.7.
// Returns (R, G, B) on the normalised scale. The pixel may lie in the row padding (x >= width);
// its value is then meaningless and is never read back.
fn lg_pixel(x: i32, y: i32) -> vec3<f32> {
    let w = i32(u.dims.x);
    let s = u.misc.y;

    // Where the chroma photosites of this row sit and whether they are red.
    let row = 2u * u32(y & 1);
    let c0 = u.cfa[row];
    let c1 = u.cfa[row + 1u];
    let green_first = c0 == 1u;
    let chroma_x = select(0, 1, green_first);
    let chroma_red = select(c0 == 0u, c1 == 0u, green_first);

    let cc = lg_c(x, y);
    let nat = lg_mosaic(x, y);
    let ha = 0.5 * (lg_a(x - 1, y) + lg_a(x + 1, y));
    let va = 0.5 * (lg_a(x, y - 1) + lg_a(x, y + 1));
    let hc = 0.5 * (lg_c(x - 1, y) + lg_c(x + 1, y));
    let vc = 0.5 * (lg_c(x, y - 1) + lg_c(x, y + 1));

    var chroma: f32;
    var green: f32;
    var other: f32;
    if ((x & 1) == chroma_x) {
        // A red or blue site: native chroma, green from the structure weights, the other chroma
        // from the four diagonal neighbours.
        let xs = min(x, w - 1);
        var sum_h = lg_plane(2u, xs, y - 2);
        sum_h = lg_opaque(sum_h + lg_plane(2u, xs, y - 1));
        sum_h = lg_opaque(sum_h + lg_plane(2u, xs, y));
        sum_h = lg_opaque(sum_h + lg_plane(2u, xs, y + 1));
        sum_h = lg_opaque(sum_h + lg_plane(2u, xs, y + 2));
        var sum_v = lg_plane(3u, xs, y - 2);
        sum_v = lg_opaque(sum_v + lg_plane(3u, xs, y - 1));
        sum_v = lg_opaque(sum_v + lg_plane(3u, xs, y));
        sum_v = lg_opaque(sum_v + lg_plane(3u, xs, y + 1));
        sum_v = lg_opaque(sum_v + lg_plane(3u, xs, y + 2));
        // The 5 x 5 box means (a division, not a reciprocal: the float form has no floor).
        let box_h = sum_h / 25.0;
        let box_v = sum_v / 25.0;
        // w_h = max(box_v, 1)^3 and w_v = max(box_h, 1)^3 (spec 3.7), divided through by the larger
        // of the two. The weights are then 1 and k = (smaller / larger)^3 <= 1: the spec's
        // t = (D / E)^3 form (divided by w_h only) is the same blend, but t reaches 1e15 on a strong
        // horizontal edge and t * vc overflows f32 for mosaic values above about 1e5 (the CPU's f64
        // does not). When box_h <= box_v this is exactly the spec's form (w_h = 1, w_v = t).
        let dh = max(box_h, 1.0);
        let dv = max(box_v, 1.0);
        let horizontal_is_larger = dh > dv;
        let r = min(dh, dv) / max(dh, dv);
        let k = lg_opaque(lg_opaque(r * r) * r);
        let wh = select(1.0, k, horizontal_is_larger);
        let wv = select(k, 1.0, horizontal_is_larger);
        let q = lg_opaque(cc / lg_opaque(lg_opaque(wh * hc) + lg_opaque(wv * vc)));
        green = lg_opaque(lg_opaque(wh * ha) + lg_opaque(wv * va)) * q;
        let sa = lg_opaque(lg_opaque(0.5 * lg_opaque(lg_a(x - 1, y - 1) + lg_a(x - 1, y + 1)))
            + lg_opaque(0.5 * lg_opaque(lg_a(x + 1, y - 1) + lg_a(x + 1, y + 1))));
        let sc = lg_opaque(lg_opaque(0.5 * lg_opaque(lg_c(x - 1, y - 1) + lg_c(x - 1, y + 1)))
            + lg_opaque(0.5 * lg_opaque(lg_c(x + 1, y - 1) + lg_c(x + 1, y + 1))));
        chroma = nat;
        other = lg_opaque(sa * lg_opaque(cc / sc)) / s;
        green = lg_opaque(green) / s;
    } else {
        // A green site: the colour that owns the row from the horizontal neighbours, the other
        // colour from the vertical ones.
        green = nat;
        chroma = lg_opaque(lg_opaque(cc / hc) * ha) / s;
        other = lg_opaque(lg_opaque(cc / vc) * va) / s;
    }
    if (chroma_red) {
        return vec3<f32>(chroma, green, other);
    }
    return vec3<f32>(other, green, chroma);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pass_id = u.ctl.x;
    if (pass_id == 3u) {
        // Two horizontally adjacent pixels per thread (a packed f16 word is written by one thread).
        let x = gid.x * 2u;
        let y = gid.y;
        if (x >= u.dims.z || y >= u.dims.y) {
            return;
        }
        let a = lg_pixel(i32(x), i32(y));
        let b = lg_pixel(i32(x) + 1, i32(y));
        let i = y * u.dims.z + x;
        let pl = u.dims.w;
        rgb_st2(i, a.x, b.x);
        rgb_st2(pl + i, a.y, b.y);
        rgb_st2(2u * pl + i, a.z, b.z);
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= u.dims.x || gid.y >= u.dims.y) {
        return;
    }
    switch pass_id {
        case 0u: {
            lg_pass_lh(x, y);
        }
        case 1u: {
            lg_pass_c(x, y);
        }
        default: {
            lg_pass_structure(x, y);
        }
    }
}
