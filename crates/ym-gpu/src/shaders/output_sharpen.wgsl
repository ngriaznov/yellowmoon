// Output sharpening: the GPU port of `ym_gpu::cpu::output_sharpen`
// (docs/algorithms/04-output-sharpening.md). A luma unsharp mask in linear light with three
// threshold methods, an optional halo limiter and a ratio transfer back to RGB.
//
// One pipeline per pass (see `kernels/output_sharpen.rs`, which appends the `main` of each pass to
// this text). Every pass is a plain function here so that naga validates all of them in every
// generated module. All planes are f32 in storage buffers whatever the device's storage mode
// (spec 3.9: planes are f32, and the halo limiter compares luma values for equality and order, so
// f16 storage would change its rank counts); there is no f16 and no accessor expansion in this file.
//
// Numerics. WGSL has no f64, so convolutions and window sums accumulate in f32: the model of the
// C++ reference's `Accumulation::F32` mode (spec 5.4). The kernel taps are the CPU's f64 taps rounded
// to f32 on upload. The high-pass and the window sums are taken as differences to the centre sample
// (spec 3.2, 3.5), so a flat region gives exact zeros here as on the CPU. Per-pixel formulas keep the
// operation order of the spec; min and max follow the C++ semantics (`std::min`, `std::max`: the first
// argument wins when the comparison fails), written with `select` so a driver cannot reorder them.
// No pass uses a fused multiply-add on purpose.
//
// Fast-math (K8). Metal compiles with fast-math and wgpu cannot switch it off: the compiler may fuse
// `a * b + c` into one multiply-add, re-associate sums and products, and its `sqrt` and `a / b` are
// not correctly rounded. The limiter compares luma values and S = L + Hacc for order, so with exact
// comparisons (tie tolerance 0) the last bit of any of them flips a rank count where two values tie
// within an ulp, and a flipped count moves a pixel by up to 2e-3 (the Metal runs at b6f2e4b); with
// the tolerance the limiter is continuous, but its soft counts still amplify a last-bit change. So
// every operation whose rounding matters is pinned: each product that meets an addition, each partial
// sum and each difference passes through `opaque` (one operation per barrier, nothing left to fuse or
// re-associate), and the square roots and quotients are made correctly rounded with integer
// arithmetic (`sharpen_sqrt`, `sharpen_div`).
// IEEE arithmetic is unchanged by both (lavapipe gives the same bits as before), and naga's MSL of
// these passes compiled with clang -O2/-O3 -ffast-math -ffp-contract=fast -mfma -mrecip=all gives
// lavapipe's bits on every case of the tests (docs/17-gpu.md section 13).

struct SharpenParams {
    width: u32,      // compute rectangle, pixels
    height: u32,
    plane: u32,      // samples per plane (width * height)
    direction: u32,  // blur: 0 rows (work_a to work_b), 1 columns (work_b to work_a)
    core_x: u32,     // transfer: the core inside the compute rectangle
    core_y: u32,
    core_w: u32,
    core_h: u32,
    tap_off: u32,    // first tap of the pass's kernel in `taps`
    tap_n: u32,      // number of taps (odd)
    planes: u32,     // blur: planes to blur (z of the dispatch)
    method: u32,     // threshold method of the stage: 0 hard, 1 smooth gain, 2 gradient ramp
    luma_r: f32,     // luma weights
    luma_g: f32,
    luma_b: f32,
    threshold: f32,  // T of the stage, normalised linear units
    a_free: f32,     // share of the stage's delta added straight to D
    a_halo: f32,     // share added to Hacc (the limited part)
    g0: f32,         // method 2: gradient of an ideal step whose high-pass response is T
    inv: f32,        // method 2: 0.25 / g0
    use_gain: u32,   // methods 1 and 2 with T > 0: the gain map is in work_a
    use_ramp: u32,   // method 2: the kernel has not collapsed to one tap
    halo_acc: u32,   // the stage adds to Hacc
    limiter: u32,    // the transfer applies the halo limiter
    in_stride: u32,  // the source image: samples per row, per plane, and the window's origin in it
    in_plane: u32,
    in_x: u32,
    in_y: u32,
    out_stride: u32, // the destination: samples per row, per plane, and the pixel of the core's (0, 0)
    out_plane: u32,
    out_x: u32,
    out_y: u32,
    out_kind: u32,   // transfer: 0 f32 planes, 1 RGBA8 with C13 fused
    space: u32,      // fused C13: output transfer, 0 sRGB, 1 BT.1886
    dither: u32,     // fused C13: blue-noise dither of the 8-bit codes
    has_mask: u32,   // the source has a fourth plane, 1 outside the photo (the letterbox)
    org_x: u32,      // fused C13: absolute position of the core's (0, 0), for the dither
    org_y: u32,
    bg: u32,         // fused C13: the letterbox colour, RGBA8 packed
    copy_only: u32,  // transfer: ratio 1, no luma, no stages (a setup without a stage)
    zero: u32,       // always 0 at run time and unknown to the compiler (see `opaque`)
    tie_rel: f32,    // the halo limiter's tie tolerance, relative to the window's centre (spec 3.5)
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> sp: SharpenParams;
@group(0) @binding(1) var<storage, read> taps: array<f32>;           // every kernel, f32
@group(0) @binding(2) var<storage, read> img: array<f32>;            // the source image (RGB planes, optionally a mask plane)
@group(0) @binding(3) var<storage, read_write> lum: array<f32>;      // L
@group(0) @binding(4) var<storage, read_write> tmp: array<f32>;      // t, the horizontal part of the high-pass
@group(0) @binding(5) var<storage, read_write> hp: array<f32>;       // h = G * L - L (d = -h)
@group(0) @binding(6) var<storage, read_write> dacc: array<f32>;     // D, the free delta
@group(0) @binding(7) var<storage, read_write> hacc: array<f32>;     // Hacc, the halo-limited delta
@group(0) @binding(8) var<storage, read_write> work_a: array<f32>;   // gain terms / envelope maps, blurred
@group(0) @binding(9) var<storage, read_write> work_b: array<f32>;   // the horizontal half of a blur
@group(0) @binding(10) var<storage, read_write> out_px: array<u32>;  // the core: f32 bits per plane, or RGBA8 words
@group(0) @binding(11) var<storage, read> dither_tab: array<f32>;    // the 64 x 64 blue-noise thresholds (fused C13)

// Constants of the spec (3.3, 3.5, 3.7).
const GAIN_EPS: f32 = 1e-10;
const LUMA_FLOOR: f32 = 1e-20;
const RATIO_MAX: f32 = 16.0;
const HALO_INV_N: f32 = 0.04;   // 1 / 25 samples of the 5 x 5 window
const HALO_INV_R: f32 = 0.5;    // 1 / window radius 2
const HALO_K_MIN_F: f32 = 6.0;  // (25 + 1) >> 2
const HALO_PULL_FADE: f32 = 0.25; // with a tie tolerance the pull fades over the last quarter (spec 3.5)

// std::max(a, b) and std::min(a, b) of the reference.
fn fmax_(a: f32, b: f32) -> f32 {
    return select(a, b, a < b);
}

fn fmin_(a: f32, b: f32) -> f32 {
    return select(a, b, b < a);
}

// Mirrored index without repeating the edge sample, with the in-range case first.
fn mir(i: i32, n: i32) -> i32 {
    if (i >= 0 && i < n) {
        return i;
    }
    return mirror_idx(i, n);
}

// Sample `c` (0 R, 1 G, 2 B, 3 the outside mask) of pixel (x, y) of the window, from the source image.
fn src_at(c: u32, x: u32, y: u32) -> f32 {
    return img[c * sp.in_plane + (sp.in_y + y) * sp.in_stride + sp.in_x + x];
}

// Hides a value from algebraic rewriting: an OR with `sp.zero`, which is 0 at run time and unknown to
// the compiler, puts an integer operation between the producer of a value and its consumers that no
// floating-point rule can see through. OR, not XOR: two barriers in a row must stay a barrier. An
// XOR applied twice cancels (x ^ z ^ z == x for every z, and compilers fold it), and fast-math
// folds a sum that starts at 0 (`0 + x` is x), so the first `opaque(acc + opaque(t * d))` of a running
// sum became the bare product, which the next sum then fused into a multiply-add (seen in clang's
// code for the adaptive ratio kernel). An OR is idempotent: (x | z) | z == x | z, still one barrier.
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | sp.zero);
}

// The correctly rounded square root of `x`, in integer arithmetic only (Metal's fast-math `sqrt` is
// 1 to 2 ulp off for about a third of the lumas: the hardware notes of `luma_has_the_cpu_bits`).
// For a positive normal x = m 2^(e - 150), the mantissa is doubled when e is odd so that the exponent
// is even, and the integer root of N = mp 2^26 (50 or 51 bits, 26 pairs of bits) is taken digit by
// digit: `root` = floor(sqrt(N)), `rem` = N - root^2 (below 2^28, so u32 holds every step). The top 24
// bits of `root` are the mantissa; the next bit and whether anything is left decide the rounding to
// nearest, ties to even. Zero returns itself; subnormals, infinities, NaN and negatives (none of which
// a luma or a gradient magnitude of finite samples is) take the device's `sqrt`.
fn sharpen_sqrt(x: f32) -> f32 {
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
    // q holds the leading bit, so adding it to the exponent field minus one carries a rounding
    // overflow into the exponent.
    return bitcast<f32>(((((e + 127u) >> 1u) - 1u) << 23u) + q);
}

// The correctly rounded quotient a / b, whatever the device's division does (a fast-math division is
// the dividend times an approximate reciprocal). The device's quotient of the two 24-bit mantissas
// gives an estimate q of the quotient's mantissa, a few units off at most; the remainder
// ma 2^k - q mb is then small (under 2^28), so the low 32 bits of the two products give it exactly in
// wrapping u32 arithmetic, and up to four steps move q to the nearest value (|2 r| < mb; a quotient of
// two 24-bit mantissas never lies exactly halfway, so there is no tie). A zero dividend gives the
// signed zero; subnormal, infinite or NaN operands and a quotient outside the normal range take the
// device's division. Measured against IEEE on 3.2 10^8 quotients, also compiled with clang
// -ffast-math -mrecip=all (an approximate reciprocal for the estimate): 0 differ.
fn sharpen_div(a: f32, b: f32) -> f32 {
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let ea = (ab >> 23u) & 0xffu;
    let eb = (bb >> 23u) & 0xffu;
    let sign = (ab ^ bb) & 0x80000000u;
    if (eb == 0u || eb == 0xffu || ea == 0xffu) {
        return a / b;
    }
    if (ea == 0u) {
        if ((ab & 0x7fffffffu) == 0u) {
            return bitcast<f32>(sign);
        }
        return a / b;
    }
    let ma = (ab & 0x7fffffu) | 0x800000u;
    let mb = (bb & 0x7fffffu) | 0x800000u;
    let big = ma >= mb;
    let k = select(24u, 23u, big);
    let ex = i32(ea) - i32(eb) + select(126, 127, big);
    if (ex < 1 || ex > 254) {
        return a / b;
    }
    let qf = f32(ma) / f32(mb);
    var q = u32(qf * bitcast<f32>((127u + k) << 23u) + 0.5);
    var r = bitcast<i32>((ma << k) - q * mb);
    let m = i32(mb);
    for (var i = 0; i < 4; i = i + 1) {
        if (2 * r > m) {
            q = q + 1u;
            r = r - m;
        } else if (2 * r < -m) {
            q = q - 1u;
            r = r + m;
        }
    }
    // q holds the leading bit, so a quotient that rounds up to the next power of two carries into the
    // exponent.
    return bitcast<f32>(sign | ((u32(ex - 1) << 23u) + q));
}

// L = sqrt(((wR R) R + (wG G) G) + (wB B) B), f32, in exactly this order (spec 3.1).
//
// The limiter compares luma values for equality and order, so a compiler that changes the last bit of
// L flips rank counts. Metal compiles with fast-math, which does two things to this expression. It
// fuses `(w c) c + a` into one multiply-add, and it re-associates the product `(w c) c` into
// `w (c c)`, which rounds differently. Passing each product through `abs` stops the first but not the
// second: read off the optimised code of clang `-O2 -ffast-math -ffp-contract=fast` (LLVM, as Metal's
// compiler), `abs` left 16 % of 2 million random lumas one ulp off the strict f32 value (19 % without
// it). So every factor and every partial sum is hidden behind `opaque`: each is an operand of exactly
// one operation, and there is nothing left to fuse or re-associate (the same model: 0 of 2 million
// lumas differ, also with `-O3`). The square root is `sharpen_sqrt`, correctly rounded whatever the
// driver's `sqrt` does: Metal's left 27 to 37 % of the lumas 1 or 2 ulp off the CPU's (b6f2e4b), and
// a luma moved by one ulp at a near-tie flips a rank count (in the busy 130 x 97 frame, one ulp at any
// of five pixels moves the output by 4.7e-4 to 2.1e-3).
fn sharpen_luma(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let i = gid.y * sp.width + gid.x;
    let r = src_at(0u, gid.x, gid.y);
    let g = src_at(1u, gid.x, gid.y);
    let b = src_at(2u, gid.x, gid.y);
    let a0 = opaque(opaque(sp.luma_r * r) * r);
    let a1 = opaque(opaque(sp.luma_g * g) * g);
    let a2 = opaque(opaque(sp.luma_b * b) * b);
    let s01 = opaque(a0 + a1);
    lum[i] = sharpen_sqrt(s01 + a2);
}

// t(x, y) = sum_j k_j (L(x + j - r, y) - L(x, y))  (spec 3.2). The difference, the product and the
// running sum each pass through `opaque`: no multiply-add, no re-association of the sum.
fn sharpen_highpass_rows(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let w = i32(sp.width);
    let x = i32(gid.x);
    let row = gid.y * sp.width;
    let c = lum[row + gid.x];
    let r = i32(sp.tap_n / 2u);
    var acc = 0.0;
    for (var j = 0u; j < sp.tap_n; j = j + 1u) {
        let xx = u32(mir(x + i32(j) - r, w));
        acc = opaque(acc + opaque(taps[sp.tap_off + j] * opaque(lum[row + xx] - c)));
    }
    tmp[row + gid.x] = acc;
}

// h(x, y) = sum_j k_j ((L(x, y + j - r) - L(x, y)) + t(x, y + j - r))  (spec 3.2), pinned as the rows.
fn sharpen_highpass_cols(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let h = i32(sp.height);
    let y = i32(gid.y);
    let i = gid.y * sp.width + gid.x;
    let c = lum[i];
    let r = i32(sp.tap_n / 2u);
    var acc = 0.0;
    for (var j = 0u; j < sp.tap_n; j = j + 1u) {
        let o = u32(mir(y + i32(j) - r, h)) * sp.width + gid.x;
        acc = opaque(acc + opaque(taps[sp.tap_off + j] * opaque(opaque(lum[o] - c) + tmp[o])));
    }
    hp[i] = acc;
}

// B = L + h, the blurred luma, at (x, y) of the compute rectangle.
fn blurred_luma(x: u32, y: u32) -> f32 {
    let o = y * sp.width + x;
    return lum[o] + hp[o];
}

// The numerator and denominator planes of the gain maps (spec 3.3), written to work_a planes 0 and 1:
// method 1: P = |d| max(|d| - T, 0);  method 2: P = d^2 w with w the ramp on the Scharr gradient of B.
// Both: Q = d^2 + eps. The gradient's terms, its squares and the magnitude are pinned (`opaque`,
// `sharpen_sqrt`), and so is d^2 before it meets the epsilon.
fn sharpen_gain_terms(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let i = gid.y * sp.width + gid.x;
    let v = -hp[i];
    var pv = 0.0;
    if (sp.method == 1u) {
        let a = abs(v);
        pv = a * fmax_(a - sp.threshold, 0.0);
    } else {
        var wgt = 0.0;
        if (sp.use_ramp == 1u) {
            let w = i32(sp.width);
            let h = i32(sp.height);
            let xm = u32(mir(i32(gid.x) - 1, w));
            let xp = u32(mir(i32(gid.x) + 1, w));
            let ym = u32(mir(i32(gid.y) - 1, h));
            let yp = u32(mir(i32(gid.y) + 1, h));
            let um = opaque(blurred_luma(xm, ym));
            let u0 = opaque(blurred_luma(gid.x, ym));
            let up = opaque(blurred_luma(xp, ym));
            let cm = opaque(blurred_luma(xm, gid.y));
            let cp = opaque(blurred_luma(xp, gid.y));
            let dm = opaque(blurred_luma(xm, yp));
            let d0 = opaque(blurred_luma(gid.x, yp));
            let dp = opaque(blurred_luma(xp, yp));
            let gx = opaque(opaque((cp - cm) * 0.3125)
                + opaque(opaque(opaque(opaque(up + dp) - um) - dm) * 0.09375));
            let gy = opaque(opaque((d0 - u0) * 0.3125)
                + opaque(opaque(opaque(opaque(dm + dp) - um) - up) * 0.09375));
            let mag = sharpen_sqrt(opaque(gx * gx) + opaque(gy * gy));
            let t = sp.inv * opaque(mag - sp.g0);
            wgt = fmin_(fmax_(t, 0.0), 1.0);
        }
        pv = opaque(v * v) * wgt;
    }
    work_a[i] = pv;
    work_a[sp.plane + i] = opaque(v * v) + GAIN_EPS;
}

// One half of the separable blur of `planes` planes (spec 3.6): rows read work_a and write work_b,
// columns read work_b and write work_a. Taps left to right from 0, mirrored reads, f32 accumulator.
// The dispatch has one z layer per plane. Products and running sums are pinned: fused into
// multiply-adds, the gain-map blur moved the gain and so S by an ulp, and at a tie within an ulp a rank
// count flipped (the 1.96e-3 of the 2048 px export from 6000 px on Metal, reproduced exactly by the
// clang fast-math model and gone with the barriers).
fn sharpen_blur(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height || gid.z >= sp.planes) {
        return;
    }
    let base = gid.z * sp.plane;
    let r = i32(sp.tap_n / 2u);
    var acc = 0.0;
    if (sp.direction == 0u) {
        let w = i32(sp.width);
        let x = i32(gid.x);
        let row = base + gid.y * sp.width;
        for (var j = 0u; j < sp.tap_n; j = j + 1u) {
            acc = opaque(acc + opaque(taps[sp.tap_off + j] * work_a[row + u32(mir(x + i32(j) - r, w))]));
        }
        work_b[row + gid.x] = acc;
    } else {
        let h = i32(sp.height);
        let y = i32(gid.y);
        for (var j = 0u; j < sp.tap_n; j = j + 1u) {
            let o = base + u32(mir(y + i32(j) - r, h)) * sp.width + gid.x;
            acc = opaque(acc + opaque(taps[sp.tap_off + j] * work_b[o]));
        }
        work_a[base + gid.y * sp.width + gid.x] = acc;
    }
}

// The stage's delta s and its accumulation (spec 3.3 and 3.4):
// D += a_free s, Hacc += a_halo s (only when the stage has a limited share). The gain is the exact
// quotient, s is pinned before the products and each product before its sum.
fn sharpen_accumulate(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let i = gid.y * sp.width + gid.x;
    let v = -hp[i];
    var s = v;
    if (sp.method == 0u) {
        let t = sp.threshold;
        s = v - fmin_(fmax_(v, -t), t);
    } else if (sp.use_gain == 1u) {
        let wg = opaque(sharpen_div(work_a[i], work_a[sp.plane + i]));
        s = v * fmin_(fmax_(wg, 0.0), 1.0);
    }
    s = opaque(s);
    dacc[i] = dacc[i] + opaque(sp.a_free * s);
    if (sp.halo_acc == 1u) {
        hacc[i] = hacc[i] + opaque(sp.a_halo * s);
    }
}

// The tie tolerance of a window centre `c` (spec 3.5): |c| times the relative tolerance, one f32
// product as on the CPU (pinned, so nothing folds it into a later expression).
fn tie_tolerance(c: f32) -> f32 {
    return opaque(abs(c) * sp.tie_rel);
}

// How much a difference `d` from the window centre counts at or below it (spec 3.5): fully within the
// tolerance `t` and below it, falling linearly to 0 at 2t above it, (2t - d) / t clamped to [0, 1].
// Where that is between 0 and 1, 2t - d is exact in f32 (d is within a factor of two of 2t), so the
// weight is the exact value correctly rounded, as close to the CPU's f64 weight as f32 allows. With
// t = 0 the exact comparison d <= 0. A NaN counts nowhere (the CPU's `max` and `min` drop it too).
// The count at or above the centre is side_weight(-d, t).
fn side_weight(d: f32, t: f32) -> f32 {
    if (t > 0.0) {
        let w = sharpen_div(opaque(opaque(2.0 * t) - d), t);
        return select(fmin_(fmax_(w, 0.0), 1.0), 0.0, is_nan_f(w));
    }
    return select(0.0, 1.0, d <= 0.0);
}

// The part of a difference `d` beyond the tolerance `t` (spec 3.5): d shrunk towards 0 by t.
fn beyond(d: f32, t: f32) -> f32 {
    let inside = select(select(d, t, d > t), -t, d < -t);
    return opaque(d - inside);
}

// The weight of the plateau pull on the side whose spread per sample is `near`, when the other side's
// is `far` (spec 3.5, with a tie tolerance): far / (near + far / N + eps) while near <= 0.75 far,
// fading linearly to 0 at near = far. 0 when far <= near or either is NaN.
fn pull_weight(near: f32, far: f32, eps: f32) -> f32 {
    if (!(far > near)) {
        return 0.0;
    }
    let fade = fmin_(sharpen_div(opaque(far - near), opaque(HALO_PULL_FADE * far)), 1.0);
    let den = opaque(opaque(near + opaque(far * HALO_INV_N)) + eps);
    return sharpen_div(opaque(fade * far), den);
}

// The four envelope maps of the halo limiter (spec 3.5) on the 5 x 5 window, written to work_a planes
// 0 to 3 as v_hi, w_hi, v_lo, w_lo. With a tie tolerance (sp.tie_rel > 0) the limiter is continuous:
// a sample counts on each side by `side_weight` and adds its difference `beyond` the tolerance, and
// the plateau pull fades (`pull_weight`) and ramps in over one count above k_min; values that tie in
// exact arithmetic then tie on every path, although the colour chain in f32 moves them by a few ulps.
// With the tolerance at 0 this is the reference's exact comparison and switch. The sums are f32 sums,
// window row by row, left to right. S, every difference and every running sum are pinned, the
// quotients are exact (`sharpen_div`) and each product that meets a sum is pinned, so a device
// computes the bits lavapipe computes.
fn sharpen_window(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let w = i32(sp.width);
    let h = i32(sp.height);
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * sp.width + gid.x;
    let c = lum[i];
    let s = opaque(c + hacc[i]);
    let tl = tie_tolerance(c);
    let ts = tie_tolerance(s);
    var n_lo = 0.0;
    var n_hi = 0.0;
    var n_slo = 0.0;
    var n_shi = 0.0;
    var e_lo = 0.0;
    var e_hi = 0.0;
    var e_slo = 0.0;
    var e_shi = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let row = u32(mir(y + dy, h)) * sp.width;
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let o = row + u32(mir(x + dx, w));
            let l = lum[o];
            let sl = opaque(l + hacc[o]);
            let dl = opaque(l - c);
            let ds = opaque(sl - s);
            n_lo = opaque(n_lo + side_weight(dl, tl));
            n_hi = opaque(n_hi + side_weight(-dl, tl));
            n_slo = opaque(n_slo + side_weight(ds, ts));
            n_shi = opaque(n_shi + side_weight(-ds, ts));
            // A NaN difference fails both comparisons and adds nothing.
            let el = beyond(dl, tl);
            let es = beyond(ds, ts);
            e_lo = opaque(e_lo + select(0.0, el, el < 0.0));
            e_hi = opaque(e_hi + select(0.0, el, el > 0.0));
            e_slo = opaque(e_slo + select(0.0, es, es < 0.0));
            e_shi = opaque(e_shi + select(0.0, es, es > 0.0));
        }
    }
    // Robust local maximum / minimum of L: c + 2 mean(l - c) over the samples on one side.
    let m_hi = opaque(sharpen_div(2.0 * e_hi, n_hi));
    let m_lo = opaque(sharpen_div(2.0 * e_lo, n_lo));
    let max_l = c + m_hi;
    let min_l = c + m_lo;
    var w_hi = fmax_(1.0 - opaque(abs(opaque(n_hi - n_shi)) * HALO_INV_R), HALO_INV_N);
    var v_hi = opaque(s + opaque(sharpen_div(2.0 * e_shi, n_shi))) * w_hi;
    var w_lo = fmax_(1.0 - opaque(abs(opaque(n_lo - n_slo)) * HALO_INV_R), HALO_INV_N);
    var v_lo = opaque(s + opaque(sharpen_div(2.0 * e_slo, n_slo))) * w_lo;
    let d_hi = opaque(sharpen_div(m_hi, n_hi));
    let d_lo = opaque(sharpen_div(-m_lo, n_lo));
    // On the flat side of an edge, pull the envelope to the plateau of L.
    if (sp.tie_rel > 0.0) {
        let eps = opaque(opaque(opaque(2.0 * tl) * HALO_INV_N) * HALO_INV_N);
        let g_hi = fmin_(fmax_(opaque(n_hi - HALO_K_MIN_F), 0.0), 1.0);
        let g_lo = fmin_(fmax_(opaque(n_lo - HALO_K_MIN_F), 0.0), 1.0);
        let t_hi = opaque(g_hi * pull_weight(d_hi, d_lo, eps));
        let t_lo = opaque(g_lo * pull_weight(d_lo, d_hi, eps));
        if (t_hi > 0.0) {
            v_hi = opaque(v_hi) + opaque(max_l * t_hi);
            w_hi = w_hi + t_hi;
        }
        if (t_lo > 0.0) {
            v_lo = opaque(v_lo) + opaque(min_l * t_lo);
            w_lo = w_lo + t_lo;
        }
    } else {
        if (n_hi > HALO_K_MIN_F && d_hi < d_lo) {
            let t = opaque(sharpen_div(d_lo, opaque(d_hi + opaque(d_lo * HALO_INV_N))));
            v_hi = opaque(v_hi) + opaque(max_l * t);
            w_hi = w_hi + t;
        }
        if (n_lo > HALO_K_MIN_F && d_lo < d_hi) {
            let t = opaque(sharpen_div(d_hi, opaque(d_lo + opaque(d_hi * HALO_INV_N))));
            v_lo = opaque(v_lo) + opaque(min_l * t);
            w_lo = w_lo + t;
        }
    }
    work_a[i] = v_hi;
    work_a[sp.plane + i] = w_hi;
    work_a[2u * sp.plane + i] = v_lo;
    work_a[3u * sp.plane + i] = w_lo;
}

// The halo limiter's clamp (spec 3.5, with the four maps blurred in work_a) and the ratio transfer
// (spec 3.7), over the core of the compute rectangle. Writes the core as three f32 planes of the
// destination image (`out_kind` 0) or as RGBA8 words with C13, the dither and the letterbox fused
// (`out_kind` 1).
//
// The ratio is 1 exactly when the pixel has no delta (spec 3.7: `(L + 0) / max(L, floor)` is 1 in
// IEEE arithmetic). Metal's fast-math default turns `a / b` into `a * (1 / b)`, which gives
// 0.99999994 for many `a == b`, so a flat field or a collapsed kernel would change by one ulp; the
// branch makes the identity exact whatever the division does. A NaN delta or luma takes the general
// road (bit test, because fast-math may assume no NaN and fold a comparison). The envelopes and the
// ratio are exact quotients (`sharpen_div`), pinned before their sums.
fn sharpen_transfer(gid: vec3<u32>) {
    if (gid.x >= sp.core_w || gid.y >= sp.core_h) {
        return;
    }
    let px = sp.core_x + gid.x;
    let py = sp.core_y + gid.y;
    var ratio = 1.0;
    if (sp.copy_only == 0u) {
        let i = py * sp.width + px;
        let l0 = lum[i];
        var d = dacc[i];
        if (sp.limiter == 1u) {
            let up = fmax_(opaque(sharpen_div(work_a[i], work_a[sp.plane + i])) - l0, 0.0);
            let lo = fmin_(opaque(sharpen_div(work_a[2u * sp.plane + i], work_a[3u * sp.plane + i])) - l0, 0.0);
            d = opaque(d + fmax_(lo, fmin_(hacc[i], up)));
        }
        if (d != 0.0 || is_nan_f(d) || is_nan_f(l0) || l0 < LUMA_FLOOR) {
            let l = fmax_(l0, LUMA_FLOOR);
            let lp = opaque(l0 + d);
            ratio = opaque(sharpen_div(lp, l));
            // A NaN fails the comparison and gives 0.
            ratio = select(0.0, fmin_(ratio, RATIO_MAX), ratio >= 0.0);
        }
    }
    let r = src_at(0u, px, py) * ratio;
    let g = src_at(1u, px, py) * ratio;
    let b = src_at(2u, px, py) * ratio;
    let o = (sp.out_y + gid.y) * sp.out_stride + sp.out_x + gid.x;
    if (sp.out_kind == 0u) {
        out_px[o] = bitcast<u32>(r);
        out_px[sp.out_plane + o] = bitcast<u32>(g);
        out_px[2u * sp.out_plane + o] = bitcast<u32>(b);
        return;
    }
    if (sp.has_mask != 0u && src_at(3u, px, py) != 0.0) {
        out_px[o] = sp.bg;
        return;
    }
    let e = c13_encode(vec3<f32>(r, g, b), sp.space);
    var thr = 0.0;
    if (sp.dither != 0u) {
        let ax = sp.org_x + gid.x;
        let ay = sp.org_y + gid.y;
        thr = dither_tab[(ay % 64u) * 64u + (ax % 64u)];
    }
    let dithered = sp.dither != 0u;
    out_px[o] = c13_code8(e.x, thr, dithered) | (c13_code8(e.y, thr, dithered) << 8u)
        | (c13_code8(e.z, thr, dithered) << 16u) | 0xff000000u;
}
