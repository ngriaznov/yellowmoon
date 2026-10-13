// R8 diffraction deconvolution (RA-09): the GPU port of `ym_gpu::cpu::sharpen::diffraction`
// (docs/algorithms/03-diffraction-deconvolution.md, sections 3.4 to 3.12). The planner (PSF kernel,
// sigma_G, the noise constants) stays on the CPU and arrives as taps and uniforms; every per-pixel
// stage runs here.
//
// One pipeline per pass (see `kernels/diffraction.rs`, which appends the `main` of each pass to this
// text). Every pass is a plain function here so that naga validates all of them in every generated
// module. All planes are f32 in storage buffers whatever the device's storage mode: the envelope
// estimate compares neighbouring luma values for equality and order (spec 3.5), and the moments
// subtract nearly equal sums, so an f16 plane would change counts and cancel the signal; there is no
// f16 and no accessor expansion in this file.
//
// Numerics. WGSL has no f64, so every window sum, convolution and product accumulates in f32 where the
// CPU reference accumulates in f64 and rounds once when it stores a plane (spec 3.13 says an all-f32
// implementation stays within 1e-5 of the reference). The exception is the luma: the envelope counts
// compare neighbouring lumas for order and equality, so `exact_luma` forms it as an exact expansion and
// gets the CPU's bits (see the notes at `opaque`). The order of the sums is the CPU's: windows are
// summed along x first and the row sums along y second, the PSF taps row by row, the Gaussian taps from
// the left. The per-pixel solve is the CPU's Cramer solve on the cofactors, in f32. min and max follow
// the C++ semantics (`std::min`, `std::max`: the first argument wins when the comparison fails) and are
// written with `select` so a driver cannot reorder them. No pass uses a fused multiply-add on purpose;
// WGSL cannot forbid one, so a driver that contracts `a * b + c` can move the last bit.
//
// Planes (each `plane` = width * height samples of the compute rectangle):
//   img   (read)       the three input planes R, G, B, balanced camera RGB, normalised
//   base  (read/write) Y, W, Pc, D, B, Hp, F1, F2, F3, EPS      (P_Y ... P_EPS)
//   work  (read/write) lower, upper | tmp | the 11 row sums of the moments
//   coef  (read/write) the four per-pixel outputs of the solve: uD, uB, uH and the weight
//   out   (write)      the corrected RGB of the core, dense planar

struct DiffParams {
    width: u32,         // compute rectangle, pixels
    height: u32,
    plane: u32,         // samples per plane (width * height)
    src: u32,           // Gaussian passes: 0 blurs Y (writes Hp), 1 blurs Z = F1 + F2 (writes F3)
    core_x: u32,        // output pass: the core inside the compute rectangle
    core_y: u32,
    core_w: u32,
    core_h: u32,
    kernel_radius: u32, // R
    gauss_radius: u32,  // r_G
    kernel_off: u32,    // first tap of the (2R + 1)^2 PSF kernel in `taps`
    gauss_off: u32,     // first tap of the 2 r_G + 1 Gaussian taps in `taps`
    zero: u32,          // always 0; the optimiser cannot know it (see `opaque`)
    luma_r: f32,        // luma weights a_R, a_G, a_B
    luma_g: f32,
    luma_b: f32,
    scale: f32,         // working scale S = 65535
    g0_r: f32,          // R4 gains
    g0_g: f32,
    g0_b: f32,
    clip_t: f32,        // clip threshold T = 0.8 S, DN
    sigma0_sq: f32,     // sigma0^2 (DN^2)
    sigma1: f32,        // sigma1 (DN)
    gate_factor: f32,   // factor on the 9 x 9 mean of the clip penalty in the gate (1 = the spec)
    strength: f32,      // blend between input and corrected luma, 0..3
};

@group(0) @binding(0) var<uniform> dp: DiffParams;
@group(0) @binding(1) var<storage, read> taps: array<f32>;
@group(0) @binding(2) var<storage, read> img: array<f32>;
@group(0) @binding(3) var<storage, read_write> base: array<f32>;
@group(0) @binding(4) var<storage, read_write> work: array<f32>;
@group(0) @binding(5) var<storage, read_write> coef: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_px: array<f32>;

// Plane indices in `base`.
const P_Y: u32 = 0u;     // luma, DN
const P_W: u32 = 1u;     // inverse noise variance
const P_PEN: u32 = 2u;   // clip penalty
const P_D: u32 = 3u;     // dark layer
const P_B: u32 = 4u;     // bright layer
const P_HP: u32 = 5u;    // Y - G(Y)
const P_F1: u32 = 6u;    // K * D
const P_F2: u32 = 7u;    // K * B
const P_F3: u32 = 8u;    // Z - G(Z), Z = F1 + F2
const P_EPS: u32 = 9u;   // Y - Z

// Plane indices in `work` for the moments (the order of the rows of spec 3.8).
const M_S11: u32 = 0u;
const M_S22: u32 = 1u;
const M_S33: u32 = 2u;
const M_S12: u32 = 3u;
const M_S13: u32 = 4u;
const M_S23: u32 = 5u;
const M_T1: u32 = 6u;
const M_T2: u32 = 7u;
const M_T3: u32 = 8u;
const M_Q: u32 = 9u;
const M_P: u32 = 10u;

// Constants of the spec (3.4, 3.5, 3.9, 3.12).
const GATE: f32 = 0.5;           // residual gate, fraction of the mean clip penalty
const RIDGE: f32 = 0.0005;       // ridge, fraction of S11 + S22 + S33
const MOMENT_FLOOR: f32 = 1e-15; // floor of S11, S22, S33
const CD_MIN: f32 = -0.5;        // lower bound of cD
const WEIGHT_MIN: f32 = 0.01;    // minimum confidence weight
const DIVIDE_GUARD: f32 = 1e-20; // envelope divide guard
const PULL_GAIN: f32 = 12.5;     // envelope pull-in
const PULL_BAND: f32 = 0.04;
const RATIO_CAP: f32 = 1.5;      // cap of the luma-to-RGB gain
const FLOOR_DN: f32 = 1.0;       // transfer floors, DN

// std::max(a, b) and std::min(a, b) of the reference.
fn fmax_(a: f32, b: f32) -> f32 {
    return select(a, b, a < b);
}

fn fmin_(a: f32, b: f32) -> f32 {
    return select(a, b, b < a);
}

// A mean over a window is clamped at 0 and reads NaN as 0 (the `box+` of the spec).
fn nonneg(m: f32) -> f32 {
    return select(0.0, m, m >= 0.0);
}

// Finite test that survives fast-math: looks at the exponent bits, not at the value.
fn finite_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// Mirrored index without repeating the edge sample, with the in-range case first.
fn mir(i: i32, n: i32) -> i32 {
    if (i >= 0 && i < n) {
        return i;
    }
    return mirror_idx(i, n);
}

// The value the Gaussian passes blur: Y, or Z = F1 + F2 (the f32 sum the CPU stores).
fn blur_source(i: u32) -> f32 {
    if (dp.src == 0u) {
        return base[P_Y * dp.plane + i];
    }
    return base[P_F1 * dp.plane + i] + base[P_F2 * dp.plane + i];
}

// Hides a value from algebraic simplification. The error-free transformations below need every rounded
// result to stay what it is, but a shader compiler may fuse a product into the sum that consumes it,
// re-associate, or cancel `a - (s - bb)` as real-number algebra: measured on Mesa's lavapipe, the plain
// `two_sum` and `two_prod` returned error terms of exactly 0 (and so did one built on `fma`, which
// lavapipe lowers to a multiply and an add). XOR with dp.zero (always 0, and unknown to the compiler)
// puts an integer operation between the producer and its consumers that the compiler cannot see through.
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ dp.zero);
}

// a + b = s + e exactly (Knuth), with every rounded result hidden from the optimiser.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    let bb = opaque(s - a);
    let ad = opaque(s - bb);
    return vec2<f32>(s, (a - ad) + (b - bb));
}

// a = hi + lo with at most 12 significant bits in each (Veltkamp).
fn split12(a: f32) -> vec2<f32> {
    let c = opaque(4097.0 * a);
    let t = opaque(c - a);
    let hi = opaque(c - t);
    return vec2<f32>(hi, a - hi);
}

// a b = p + e exactly (Dekker). `fma` is not used: it is not guaranteed to be fused on every driver
// (Mesa's llvmpipe lowers it to a multiply and an add), but the products of 12-bit halves are exact.
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = opaque(a * b);
    let x = split12(a);
    let y = split12(b);
    let e1 = opaque(x.x * y.x - p);
    let e2 = opaque(e1 + x.x * y.y);
    let e3 = opaque(e2 + x.y * y.x);
    return vec2<f32>(p, e3 + x.y * y.y);
}

// a (S v) as a pair (hi, lo) whose sum is the exact product to about 2^-48 relative.
fn scaled_term(a: f32, v: f32) -> vec2<f32> {
    let x = two_prod(dp.scale, v);   // S v = x.x + x.y exactly
    let p = two_prod(a, x.x);        // a x.x = p.x + p.y exactly
    return vec2<f32>(p.x, p.y + a * x.y);
}

// Y = a_R S R + a_G S G + a_B S B, DN. The CPU port forms it in f64 and rounds once to f32; the
// expansion above gives the correctly rounded f32 of the exact sum in all but about one pixel in two
// million, so the luma has the CPU's bits (an f32 sum is 1 to 2 ulp off in a third of the pixels, and
// the envelope counts of 3.5 compare neighbouring lumas for order and equality).
fn exact_luma(r: f32, g: f32, b: f32) -> f32 {
    let tr = scaled_term(dp.luma_r, r);
    let tg = scaled_term(dp.luma_g, g);
    let tb = scaled_term(dp.luma_b, b);
    let s1 = two_sum(tr.x, tg.x);
    let s2 = two_sum(s1.x, tb.x);
    // The correction is hidden too, so that the final sum stays one rounding of s2.x and the whole of it.
    let small = opaque((s1.y + s2.y) + ((tr.y + tg.y) + tb.y));
    return s2.x + small;
}

// 3.4: luma Y (DN), inverse noise variance w and clip penalty Pc, per pixel.
fn diffraction_luma(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let i = gid.y * dp.width + gid.x;
    let xr = dp.scale * img[i];
    let xg = dp.scale * img[dp.plane + i];
    let xb = dp.scale * img[2u * dp.plane + i];
    let y = exact_luma(img[i], img[dp.plane + i], img[2u * dp.plane + i]);
    let wi = 1.0 / (dp.sigma0_sq + dp.sigma1 * fmax_(y, 0.0));
    // e_c = a_c max(x_c / g0_c - T, 0), sorted ascending.
    var e0 = dp.luma_r * fmax_(xr / dp.g0_r - dp.clip_t, 0.0);
    var e1 = dp.luma_g * fmax_(xg / dp.g0_g - dp.clip_t, 0.0);
    var e2 = dp.luma_b * fmax_(xb / dp.g0_b - dp.clip_t, 0.0);
    if (e1 < e0) {
        let t = e0;
        e0 = e1;
        e1 = t;
    }
    if (e2 < e1) {
        let t = e1;
        e1 = e2;
        e2 = t;
    }
    if (e1 < e0) {
        let t = e0;
        e0 = e1;
        e1 = t;
    }
    let excess = (25.0 * e0 + 5.0 * e1) + e2;
    base[P_Y * dp.plane + i] = y;
    base[P_W * dp.plane + i] = wi;
    base[P_PEN * dp.plane + i] = 1.0 + (wi * excess) * excess;
}

// 3.5: the lower and upper envelope estimates over the 5 x 5 window of Y, into work planes 0 and 1.
fn diffraction_envelope(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let w = i32(dp.width);
    let h = i32(dp.height);
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * dp.width + gid.x;
    let c = base[P_Y * dp.plane + i];
    var sp = 0.0;
    var sn = 0.0;
    var np = 0;
    var nn = 0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let row = P_Y * dp.plane + u32(mir(y + dy, h)) * dp.width;
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let d = base[row + u32(mir(x + dx, w))] - c;
            if (d >= 0.0) {
                sp = sp + d;
                np = np + 1;
            }
            if (d <= 0.0) {
                sn = sn + d;
                nn = nn + 1;
            }
        }
    }
    let gp = (2.0 * sp) / (f32(np * (np - 1)) + DIVIDE_GUARD);
    let gn = (-2.0 * sn) / (f32(nn * (nn - 1)) + DIVIDE_GUARD);
    let top = c + gp * f32(np - 1);
    let bottom = c - gn * f32(nn - 1);
    let mid = 0.5 * (top + bottom);
    let pull = PULL_GAIN * fmin_(fmin_(gp, gn), PULL_BAND * (top - bottom));
    work[i] = fmin_(bottom + pull, mid);
    work[dp.plane + i] = fmax_(top - pull, mid);
}

// The 5 x 5 mean of plane `plane` of `work` around (x, y): x first, then y, over 25.
fn mean5_work(plane: u32, x: i32, y: i32) -> f32 {
    let w = i32(dp.width);
    let h = i32(dp.height);
    var acc = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let row = plane * dp.plane + u32(mir(y + dy, h)) * dp.width;
        var rs = 0.0;
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            rs = rs + work[row + u32(mir(x + dx, w))];
        }
        acc = acc + rs;
    }
    return acc / 25.0;
}

// 3.6: the dark and bright layers, D + B = Y.
fn diffraction_layers(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * dp.width + gid.x;
    let lm = nonneg(mean5_work(0u, x, y));
    let hm = nonneg(mean5_work(1u, x, y));
    let yv = base[P_Y * dp.plane + i];
    var d = 0.0;
    var b = 0.0;
    if (yv <= lm) {
        d = yv;
    } else if (yv >= hm) {
        b = yv;
    } else {
        let t = (yv - lm) / (hm - lm);
        b = hm * t;
        d = yv - b;
    }
    base[P_D * dp.plane + i] = d;
    base[P_B * dp.plane + i] = b;
}

// 3.7: the horizontal half of the Gaussian, blur_source -> work plane 0.
fn diffraction_gauss_rows(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let w = i32(dp.width);
    let x = i32(gid.x);
    let r = i32(dp.gauss_radius);
    let row = gid.y * dp.width;
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k = k + 1) {
        acc = acc + taps[dp.gauss_off + u32(k)] * blur_source(row + u32(mir(x + k - r, w)));
    }
    work[row + gid.x] = acc;
}

// 3.7: the vertical half, work plane 0 -> Hp = Y - G(Y) (src 0) or F3 = Z - G(Z) (src 1).
fn diffraction_gauss_cols(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let h = i32(dp.height);
    let y = i32(gid.y);
    let r = i32(dp.gauss_radius);
    let i = gid.y * dp.width + gid.x;
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k = k + 1) {
        acc = acc + taps[dp.gauss_off + u32(k)] * work[u32(mir(y + k - r, h)) * dp.width + gid.x];
    }
    let v = blur_source(i) - acc;
    if (dp.src == 0u) {
        base[P_HP * dp.plane + i] = v;
    } else {
        base[P_F3 * dp.plane + i] = v;
    }
}

// 3.7: F1 = K * D, F2 = K * B (both on the PSF kernel, row by row, f32 accumulators) and the
// residual eps = Y - (F1 + F2).
fn diffraction_psf(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let w = i32(dp.width);
    let h = i32(dp.height);
    let x = i32(gid.x);
    let y = i32(gid.y);
    let r = i32(dp.kernel_radius);
    let kw = 2 * r + 1;
    let i = gid.y * dp.width + gid.x;
    var a1 = 0.0;
    var a2 = 0.0;
    for (var dy = 0; dy < kw; dy = dy + 1) {
        let row = u32(mir(y + dy - r, h)) * dp.width;
        for (var dx = 0; dx < kw; dx = dx + 1) {
            let k = taps[dp.kernel_off + u32(dy * kw + dx)];
            let o = row + u32(mir(x + dx - r, w));
            a1 = a1 + k * base[P_D * dp.plane + o];
            a2 = a2 + k * base[P_B * dp.plane + o];
        }
    }
    base[P_F1 * dp.plane + i] = a1;
    base[P_F2 * dp.plane + i] = a2;
    base[P_EPS * dp.plane + i] = base[P_Y * dp.plane + i] - (a1 + a2);
}

// 3.8, the horizontal half: for each of the 11 moments the sum over the 9 samples of the row of the
// products w u v (evaluated (w u) v) of the planes, into work planes M_*.
fn diffraction_moment_rows(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let w = i32(dp.width);
    let x = i32(gid.x);
    let i = gid.y * dp.width + gid.x;
    let row = gid.y * dp.width;
    var s11 = 0.0;
    var s22 = 0.0;
    var s33 = 0.0;
    var s12 = 0.0;
    var s13 = 0.0;
    var s23 = 0.0;
    var t1 = 0.0;
    var t2 = 0.0;
    var t3 = 0.0;
    var q = 0.0;
    var p = 0.0;
    for (var dx = -4; dx <= 4; dx = dx + 1) {
        let o = row + u32(mir(x + dx, w));
        let wi = base[P_W * dp.plane + o];
        let f1 = base[P_F1 * dp.plane + o];
        let f2 = base[P_F2 * dp.plane + o];
        let f3 = base[P_F3 * dp.plane + o];
        let e = base[P_EPS * dp.plane + o];
        s11 = s11 + (wi * f1) * f1;
        s22 = s22 + (wi * f2) * f2;
        s33 = s33 + (wi * f3) * f3;
        s12 = s12 + (wi * f1) * f2;
        s13 = s13 + (wi * f1) * f3;
        s23 = s23 + (wi * f2) * f3;
        t1 = t1 + (wi * f1) * e;
        t2 = t2 + (wi * f2) * e;
        t3 = t3 + (wi * f3) * e;
        q = q + (wi * e) * e;
        p = p + base[P_PEN * dp.plane + o];
    }
    work[M_S11 * dp.plane + i] = s11;
    work[M_S22 * dp.plane + i] = s22;
    work[M_S33 * dp.plane + i] = s33;
    work[M_S12 * dp.plane + i] = s12;
    work[M_S13 * dp.plane + i] = s13;
    work[M_S23 * dp.plane + i] = s23;
    work[M_T1 * dp.plane + i] = t1;
    work[M_T2 * dp.plane + i] = t2;
    work[M_T3 * dp.plane + i] = t3;
    work[M_Q * dp.plane + i] = q;
    work[M_P * dp.plane + i] = p;
}

// The sum over the 9 row sums of one moment around row y, in column x.
fn moment_cols(plane: u32, x: u32, y: i32, h: i32) -> f32 {
    var acc = 0.0;
    for (var dy = -4; dy <= 4; dy = dy + 1) {
        acc = acc + work[plane * dp.plane + u32(mir(y + dy, h)) * dp.width + x];
    }
    return acc;
}

struct Solution {
    cd: f32,
    cb: f32,
    ch: f32,
    wgt: f32,
};

// 3.9: the constrained, ridge-regularised 3-parameter least squares with the noise gate. The outputs
// are the coefficients already multiplied by gain * omega, and omega.
fn solve_pixel(
    s11_in: f32, s12: f32, s13: f32, s22_in: f32, s23: f32, s33_in: f32,
    t1: f32, t2: f32, t3: f32, q: f32, p: f32,
) -> Solution {
    let t = GATE * p;
    if (q <= t) {
        // The residual is at the noise level: leave the pixel alone.
        return Solution(0.0, 0.0, 0.0, 1.0 + (t - q) / t);
    }
    let s11 = fmax_(s11_in, MOMENT_FLOOR);
    let s22 = fmax_(s22_in, MOMENT_FLOOR);
    let s33 = fmax_(s33_in, MOMENT_FLOOR);

    let lambda = RIDGE * ((s11 + s22) + s33);
    let m11 = s11 + lambda;
    let m22 = s22 + lambda;
    let m33 = s33 + lambda;
    let cof11 = m22 * m33 - s23 * s23;
    let cof12 = s13 * s23 - s12 * m33;
    let cof13 = s12 * s23 - s13 * m22;
    let cof22 = m11 * m33 - s13 * s13;
    let cof23 = s12 * s13 - s23 * m11;
    let cof33 = m11 * m22 - s12 * s12;
    let det = (m11 * cof11 + s12 * cof12) + s13 * cof13;

    var cd = 0.0;
    var cb = 0.0;
    var ch = ((t1 * cof13 + t2 * cof23) + t3 * cof33) / det;
    var tie = false;
    if (ch >= 0.0 && ch < 1.0) {
        cd = ((t1 * cof11 + t2 * cof12) + t3 * cof13) / det;
        cb = ((t1 * cof12 + t2 * cof22) + t3 * cof23) / det;
        tie = cd > cb;
    } else {
        ch = fmin_(fmax_(ch, 0.0), 1.0);
        let t1p = t1 - s13 * ch;
        let t2p = t2 - s23 * ch;
        let d2 = s11 * s22 - s12 * s12;
        cd = (s22 * t1p - s12 * t2p) / d2;
        cb = (s11 * t2p - s12 * t1p) / d2;
        if (cd > cb) {
            let c = (t1p + t2p) / ((s11 + 2.0 * s12) + s22);
            cd = c;
            cb = c;
        }
    }
    if (tie) {
        // Re-solve with cD = cB = c; unknowns (c, cH), no ridge.
        let a = (s11 + 2.0 * s12) + s22;
        let b = s13 + s23;
        let ts = t1 + t2;
        let d = a * s33 - b * b;
        ch = (a * t3 - b * ts) / d;
        var c = 0.0;
        if (ch >= 0.0 && ch < 1.0) {
            c = (s33 * ts - b * t3) / d;
        } else {
            ch = fmin_(fmax_(ch, 0.0), 1.0);
            c = (ts - b * ch) / a;
        }
        cd = c;
        cb = c;
    }
    if (cd < CD_MIN) {
        cd = CD_MIN;
        cb = fmax_(cb, CD_MIN);
    }
    if (!finite_f(cd) || !finite_f(cb) || !finite_f(ch)) {
        // A degenerate system: leave the pixel alone.
        return Solution(0.0, 0.0, 0.0, 1.0);
    }

    // The weighted residual energy after the fit (clamped diagonal, no ridge).
    let e = ((((q - 2.0 * ((cd * t1 + cb * t2) + ch * t3)) + s11 * cd * cd) + s22 * cb * cb)
        + s33 * ch * ch)
        + 2.0 * ((s12 * cd * cb + s13 * cd * ch) + s23 * cb * ch);

    var gain = 1.0;
    if (q < p) {
        gain = fmax_((q - t) / (p - t), 0.0);
    }
    let k = 1.0 / GATE;
    let lo = 0.25 * q + k;
    if (e > lo) {
        let hi = 0.5 * q + k;
        gain = gain * fmax_((hi - e) / (hi - lo), 0.0);
    }
    var wgt = 1.0;
    if (e > k) {
        wgt = fmax_((k / e) * (k / e), WEIGHT_MIN);
    }
    let f = gain * wgt;
    return Solution(cd * f, cb * f, ch * f, wgt);
}

// 3.8 (the vertical half) and 3.9: the moments of the 9 x 9 window, then the solve. Writes the four
// coefficient planes uD, uB, uH and omega.
fn diffraction_solve(gid: vec3<u32>) {
    if (gid.x >= dp.width || gid.y >= dp.height) {
        return;
    }
    let h = i32(dp.height);
    let y = i32(gid.y);
    let i = gid.y * dp.width + gid.x;
    let s11 = nonneg(moment_cols(M_S11, gid.x, y, h) / 81.0);
    let s22 = nonneg(moment_cols(M_S22, gid.x, y, h) / 81.0);
    let s33 = nonneg(moment_cols(M_S33, gid.x, y, h) / 81.0);
    let s12 = moment_cols(M_S12, gid.x, y, h) / 81.0;
    let s13 = moment_cols(M_S13, gid.x, y, h) / 81.0;
    let s23 = moment_cols(M_S23, gid.x, y, h) / 81.0;
    let t1 = moment_cols(M_T1, gid.x, y, h) / 81.0;
    let t2 = moment_cols(M_T2, gid.x, y, h) / 81.0;
    let t3 = moment_cols(M_T3, gid.x, y, h) / 81.0;
    let q = nonneg(moment_cols(M_Q, gid.x, y, h) / 81.0);
    let p = nonneg(moment_cols(M_P, gid.x, y, h) / 81.0) * dp.gate_factor;
    let s = solve_pixel(s11, s12, s13, s22, s23, s33, t1, t2, t3, q, p);
    coef[i] = s.cd;
    coef[dp.plane + i] = s.cb;
    coef[2u * dp.plane + i] = s.ch;
    coef[3u * dp.plane + i] = s.wgt;
}

// The 5 x 5 mean of plane `plane` of `coef` around (x, y): x first, then y, over 25.
fn mean5_coef(plane: u32, x: i32, y: i32) -> f32 {
    let w = i32(dp.width);
    let h = i32(dp.height);
    var acc = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let row = plane * dp.plane + u32(mir(y + dy, h)) * dp.width;
        var rs = 0.0;
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            rs = rs + coef[row + u32(mir(x + dx, w))];
        }
        acc = acc + rs;
    }
    return acc / 25.0;
}

// 3.10 to 3.12: the confidence-weighted 5 x 5 average of the coefficients, the corrected luma, the
// strength blend and the capped ratio plus additive remainder, over the core of the compute rectangle.
// Writes the core's RGB planes densely.
fn diffraction_output(gid: vec3<u32>) {
    if (gid.x >= dp.core_w || gid.y >= dp.core_h) {
        return;
    }
    let cx = dp.core_x + gid.x;
    let cy = dp.core_y + gid.y;
    let i = cy * dp.width + cx;
    let m1 = mean5_coef(0u, i32(cx), i32(cy));
    let m2 = mean5_coef(1u, i32(cx), i32(cy));
    let m3 = mean5_coef(2u, i32(cx), i32(cy));
    let mw = mean5_coef(3u, i32(cx), i32(cy));
    let y = base[P_Y * dp.plane + i];
    let d = base[P_D * dp.plane + i];
    let b = base[P_B * dp.plane + i];
    let hp = base[P_HP * dp.plane + i];
    // L = Y + (D uD + B uB + Hp uH) / omega, L_s = Y + strength (L - Y).
    let num = (d * m1 + b * m2) + hp * m3;
    let ls = y + dp.strength * (num / mw);
    // k = min(max(L_s, 1) / max(Y, 1), 1.5); off = (L_s - Y k) / S; out_c = in_c k + off.
    var k = fmin_(fmax_(ls, FLOOR_DN) / fmax_(y, FLOOR_DN), RATIO_CAP);
    var off = (ls - y * k) / dp.scale;
    if (ls == y) {
        // Nothing was changed: an exact identity, whatever the accuracy of the divide.
        k = 1.0;
        off = 0.0;
    }
    let o = gid.y * dp.core_w + gid.x;
    let cp = dp.core_w * dp.core_h;
    out_px[o] = img[i] * k + off;
    out_px[cp + o] = img[dp.plane + i] * k + off;
    out_px[2u * cp + o] = img[2u * dp.plane + i] * k + off;
}
