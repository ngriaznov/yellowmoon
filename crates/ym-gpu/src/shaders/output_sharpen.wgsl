// Output sharpening (RA-09): the GPU port of `ym_gpu::cpu::output_sharpen`
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
// No pass uses a fused multiply-add on purpose; WGSL cannot forbid one, so a driver that contracts
// `a * b + c` can move the last bit (see the test notes in tests/gpu_output_sharpen.rs).

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
};

@group(0) @binding(0) var<uniform> sp: SharpenParams;
@group(0) @binding(1) var<storage, read> taps: array<f32>;           // every kernel, f32
@group(0) @binding(2) var<storage, read> img: array<f32>;            // RGB planes of the compute rectangle
@group(0) @binding(3) var<storage, read_write> lum: array<f32>;      // L
@group(0) @binding(4) var<storage, read_write> tmp: array<f32>;      // t, the horizontal part of the high-pass
@group(0) @binding(5) var<storage, read_write> hp: array<f32>;       // h = G * L - L (d = -h)
@group(0) @binding(6) var<storage, read_write> dacc: array<f32>;     // D, the free delta
@group(0) @binding(7) var<storage, read_write> hacc: array<f32>;     // Hacc, the halo-limited delta
@group(0) @binding(8) var<storage, read_write> work_a: array<f32>;   // gain terms / envelope maps, blurred
@group(0) @binding(9) var<storage, read_write> work_b: array<f32>;   // the horizontal half of a blur
@group(0) @binding(10) var<storage, read_write> out_px: array<f32>;  // sharpened RGB of the core

// Constants of the spec (3.3, 3.5, 3.7).
const GAIN_EPS: f32 = 1e-10;
const LUMA_FLOOR: f32 = 1e-20;
const RATIO_MAX: f32 = 16.0;
const HALO_INV_N: f32 = 0.04;   // 1 / 25 samples of the 5 x 5 window
const HALO_INV_R: f32 = 0.5;    // 1 / window radius 2
const HALO_K_MIN: i32 = 6;      // (25 + 1) >> 2

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

// L = sqrt(((wR R) R + (wG G) G) + (wB B) B), f32, in exactly this order (spec 3.1).
fn sharpen_luma(gid: vec3<u32>) {
    if (gid.x >= sp.width || gid.y >= sp.height) {
        return;
    }
    let i = gid.y * sp.width + gid.x;
    let r = img[i];
    let g = img[sp.plane + i];
    let b = img[2u * sp.plane + i];
    let a0 = (sp.luma_r * r) * r;
    let a1 = (sp.luma_g * g) * g;
    let a2 = (sp.luma_b * b) * b;
    lum[i] = sqrt((a0 + a1) + a2);
}

// t(x, y) = sum_j k_j (L(x + j - r, y) - L(x, y))  (spec 3.2)
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
        acc = acc + taps[sp.tap_off + j] * (lum[row + xx] - c);
    }
    tmp[row + gid.x] = acc;
}

// h(x, y) = sum_j k_j ((L(x, y + j - r) - L(x, y)) + t(x, y + j - r))  (spec 3.2)
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
        acc = acc + taps[sp.tap_off + j] * ((lum[o] - c) + tmp[o]);
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
// Both: Q = d^2 + eps.
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
            let um = blurred_luma(xm, ym);
            let u0 = blurred_luma(gid.x, ym);
            let up = blurred_luma(xp, ym);
            let cm = blurred_luma(xm, gid.y);
            let cp = blurred_luma(xp, gid.y);
            let dm = blurred_luma(xm, yp);
            let d0 = blurred_luma(gid.x, yp);
            let dp = blurred_luma(xp, yp);
            let gx = (cp - cm) * 0.3125 + (((up + dp) - um) - dm) * 0.09375;
            let gy = (d0 - u0) * 0.3125 + (((dm + dp) - um) - up) * 0.09375;
            let t = sp.inv * (sqrt(gx * gx + gy * gy) - sp.g0);
            wgt = fmin_(fmax_(t, 0.0), 1.0);
        }
        pv = v * v * wgt;
    }
    work_a[i] = pv;
    work_a[sp.plane + i] = v * v + GAIN_EPS;
}

// One half of the separable blur of `planes` planes (spec 3.6): rows read work_a and write work_b,
// columns read work_b and write work_a. Taps left to right from 0, mirrored reads, f32 accumulator.
// The dispatch has one z layer per plane.
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
            acc = acc + taps[sp.tap_off + j] * work_a[row + u32(mir(x + i32(j) - r, w))];
        }
        work_b[row + gid.x] = acc;
    } else {
        let h = i32(sp.height);
        let y = i32(gid.y);
        for (var j = 0u; j < sp.tap_n; j = j + 1u) {
            let o = base + u32(mir(y + i32(j) - r, h)) * sp.width + gid.x;
            acc = acc + taps[sp.tap_off + j] * work_b[o];
        }
        work_a[base + gid.y * sp.width + gid.x] = acc;
    }
}

// The stage's delta s and its accumulation (spec 3.3 and 3.4):
// D += a_free s, Hacc += a_halo s (only when the stage has a limited share).
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
        let wg = work_a[i] / work_a[sp.plane + i];
        s = v * fmin_(fmax_(wg, 0.0), 1.0);
    }
    dacc[i] = dacc[i] + sp.a_free * s;
    if (sp.halo_acc == 1u) {
        hacc[i] = hacc[i] + sp.a_halo * s;
    }
}

// The four envelope maps of the halo limiter (spec 3.5) on the 5 x 5 window, written to work_a planes
// 0 to 3 as v_hi, w_hi, v_lo, w_lo. Counts are exact comparisons of f32 values; the sums are f32 sums
// of differences to the centre, window row by row, left to right.
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
    let s = c + hacc[i];
    var n_lo = 0;
    var n_hi = 0;
    var n_slo = 0;
    var n_shi = 0;
    var e_lo = 0.0;
    var e_hi = 0.0;
    var e_slo = 0.0;
    var e_shi = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let row = u32(mir(y + dy, h)) * sp.width;
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let o = row + u32(mir(x + dx, w));
            let l = lum[o];
            let sl = l + hacc[o];
            if (l <= c) {
                n_lo = n_lo + 1;
                e_lo = e_lo + (l - c);
            }
            if (l >= c) {
                n_hi = n_hi + 1;
                e_hi = e_hi + (l - c);
            }
            if (sl <= s) {
                n_slo = n_slo + 1;
                e_slo = e_slo + (sl - s);
            }
            if (sl >= s) {
                n_shi = n_shi + 1;
                e_shi = e_shi + (sl - s);
            }
        }
    }
    let nlo = f32(n_lo);
    let nhi = f32(n_hi);
    let nslo = f32(n_slo);
    let nshi = f32(n_shi);
    // Robust local maximum / minimum of L: c + 2 mean(l - c) over the samples on one side.
    let m_hi = (2.0 * e_hi) / nhi;
    let m_lo = (2.0 * e_lo) / nlo;
    let max_l = c + m_hi;
    let min_l = c + m_lo;
    var w_hi = fmax_(1.0 - f32(abs(n_hi - n_shi)) * HALO_INV_R, HALO_INV_N);
    var v_hi = (s + (2.0 * e_shi) / nshi) * w_hi;
    var w_lo = fmax_(1.0 - f32(abs(n_lo - n_slo)) * HALO_INV_R, HALO_INV_N);
    var v_lo = (s + (2.0 * e_slo) / nslo) * w_lo;
    let d_hi = m_hi / nhi;
    let d_lo = -m_lo / nlo;
    // On the flat side of an edge, pull the envelope to the plateau of L.
    if (n_hi > HALO_K_MIN && d_hi < d_lo) {
        let t = d_lo / (d_hi + d_lo * HALO_INV_N);
        v_hi = v_hi + max_l * t;
        w_hi = w_hi + t;
    }
    if (n_lo > HALO_K_MIN && d_lo < d_hi) {
        let t = d_hi / (d_lo + d_hi * HALO_INV_N);
        v_lo = v_lo + min_l * t;
        w_lo = w_lo + t;
    }
    work_a[i] = v_hi;
    work_a[sp.plane + i] = w_hi;
    work_a[2u * sp.plane + i] = v_lo;
    work_a[3u * sp.plane + i] = w_lo;
}

// The halo limiter's clamp (spec 3.5, with the four maps blurred in work_a) and the ratio transfer
// (spec 3.7), over the core of the compute rectangle. Writes the core's RGB planes densely.
fn sharpen_transfer(gid: vec3<u32>) {
    if (gid.x >= sp.core_w || gid.y >= sp.core_h) {
        return;
    }
    let i = (sp.core_y + gid.y) * sp.width + sp.core_x + gid.x;
    let o = gid.y * sp.core_w + gid.x;
    let l0 = lum[i];
    var d = dacc[i];
    if (sp.limiter == 1u) {
        let up = fmax_(work_a[i] / work_a[sp.plane + i] - l0, 0.0);
        let lo = fmin_(work_a[2u * sp.plane + i] / work_a[3u * sp.plane + i] - l0, 0.0);
        d = d + fmax_(lo, fmin_(hacc[i], up));
    }
    let l = fmax_(l0, LUMA_FLOOR);
    let lp = l0 + d;
    var ratio = lp / l;
    // A NaN fails the comparison and gives 0.
    ratio = select(0.0, fmin_(ratio, RATIO_MAX), ratio >= 0.0);
    let cp = sp.core_w * sp.core_h;
    out_px[o] = img[i] * ratio;
    out_px[cp + o] = img[sp.plane + i] * ratio;
    out_px[2u * cp + o] = img[2u * sp.plane + i] * ratio;
}
