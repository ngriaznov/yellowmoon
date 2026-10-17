// R7 colour noise reduction: the GPU port of `ym_gpu::cpu::detail::chroma_guided_whole`
// (`docs/05-raw-processing.md` section R7, "Colour"). A guided filter (He et al.) of the two chroma
// planes R - Y and B - Y, guided by the luma Y, with a box radius r = round(2 + 6 C / 100) (2 to 8) and
// a per-pixel eps = max(sigma^2 (C / 50)^2, 1e-10), then the image is rebuilt from the filtered
// chroma and the unchanged luma:
//
//   mean_x   = box(x)                         x in Y, Y*Y, Cr, Y*Cr, Cb, Y*Cb      (rows, then columns)
//   a = (box(Y*P) - mean_Y mean_P) / (max(box(Y*Y) - mean_Y^2, 0) + eps),  b = mean_P - a mean_Y
//   q = box(a) Y + box(b)                     for P in Cr, Cb                      (rows, then columns)
//
// Four passes, one pipeline each (the order of the CPU's `guided`): horizontal means of the six
// quantities (`gd_rows`), vertical means and the coefficients a and b of both chroma planes
// (`gd_cols`), horizontal means of the four coefficient planes (`gd_smooth_rows`), vertical means and
// the output (`gd_out`).
//
// A box mean is the mean of the samples that exist (the window is clipped at the edge of the input
// rectangle, it is not mirrored: `filters::box_mean`), and it is formed separably with the CPU's
// stored intermediate: the horizontal means are rounded to f32 before the vertical ones. The CPU adds
// in f64 and rounds once per stored plane; here every window is a fixed-length direct sum of at most
// 17 samples (D4: nothing runs along a whole row, a pixel depends on its window and not on where the
// rectangle starts), added **exactly** (Knuth's two-sum into a pair, `hi + lo`) and divided by the
// count with a corrected quotient, so the f32 result is the correctly rounded mean, which is what
// the CPU's f64 sum rounds to (a rate of order one value in 10^7 differs by an ulp). The tiles equal
// the whole frame (the receptive field is 2 r: two means in series, D5).
//
// Why the exactness: the variance max(box(Y*Y) - mean_Y^2, 0) is a difference of two numbers near
// mean_Y^2, so a mean that is off by one ulp moves it by 6e-8 mean_Y^2; in a bright flat noisy area
// (level 0.9, sigma 0.003) that is 5e-3 of var + eps, the coefficient a moves by as much, and the
// output by 6e-6 (measured: plain f32 sums against the CPU, 6.3e-6; 2.3e-5 without regularisation).
// With the CPU's own stored values reproduced, the remaining f32 operations are the CPU's, in its
// order, and the two paths part by rounding of the last bit.
//
// Every product that is subtracted from a number of its own size, and the luma and the squares that
// feed the means, are formed behind `opaque` (an OR with a uniform that is zero at run time, the
// technique of `diffraction.wgsl`): a compiler that contracts `m1 - m0 * m0` into a fused multiply-add
// (Metal, fast-math) would otherwise change exactly the digits this filter is sensitive to.
//
// Fast-math (Metal, K8): the exactness above rests on `opaque` hiding every rounded value and every
// error term of the two-sum and the quotients from the optimiser, as in `diffraction.wgsl`, and on
// every division being refined from its exact remainder (`run_mean`, `div_cr`), so a divide that is
// off by a few ulp (Metal's fast-math one may be) still gives the correctly rounded quotient; nothing
// rests on infinities. The shader assumes round-to-nearest f32 add and multiply. A C model of the
// passes compiled with clang -O3 -ffast-math -ffp-contract=fast -mfma (and with -mrecip, and with a
// division made 3 ulp wrong on purpose) gives the strict bits in every sample.
//
// Bindings: 0 uniform, 1 the input planes, 2 and 3 two scratch buffers (6 and 4 planes of the input's
// size, `w * h` per plane, no padding), 4 the sigma plane, 5 the output (the core).

struct GdParams {
    dims: vec4<u32>,   // input width, height, row stride, plane length
    core: vec4<u32>,   // core x, y, width, height in the input
    dst: vec4<u32>,    // output row stride, output plane length, sigma row stride, radius
    sig: vec4<u32>,    // the input's origin in the sigma plane (x, y), scratch plane length, zero
    k: vec4<f32>,      // (C / 50)^2
    luma: vec4<f32>,   // luma weights (R, G, B)
};

@group(0) @binding(0) var<uniform> gp: GdParams;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> s1: array<f32>;
@group(0) @binding(3) var<storage, read_write> s2: array<f32>;
@group(0) @binding(4) var<storage, read> sgm: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;


// Always 0 at run time, which the optimiser cannot know: OR-ing a value with it hides the value from
// reassociation, contraction and algebraic rewrites (see the notes at the top).
// An OR, not an XOR: two barriers in a row must stay a barrier. An XOR applied twice cancels
// (x ^ z ^ z == x for every z, and compilers fold it), and fast-math folds a sum that starts at 0
// (`0 + x` is x), so the first `opaque(acc + opaque(t))` of a running sum becomes the bare `t`, which
// the next sum then fuses into a multiply-add (seen in clang's code for the adaptive ratio kernel;
// docs/17-gpu.md K8). An OR is idempotent: (x | z) | z == x | z, still one barrier.
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | gp.sig.w);
}

// a + b = s + e exactly (Knuth), every rounded result and error term hidden.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    let bb = opaque(s - a);
    let ad = opaque(s - bb);
    let ea = opaque(a - ad);
    let eb = opaque(b - bb);
    return vec2<f32>(s, ea + eb);
}

// a = hi + lo with at most 12 significant bits in each (Veltkamp).
fn split12(a: f32) -> vec2<f32> {
    let c = opaque(4097.0 * a);
    let t = opaque(c - a);
    let hi = opaque(c - t);
    return vec2<f32>(hi, opaque(a - hi));
}

// a b = p + e exactly (Dekker); the partial products of 12-bit halves are exact in f32, so a
// contraction of the sums below cannot change a bit either.
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = opaque(a * b);
    let x = split12(a);
    let y = split12(b);
    let e1 = opaque(x.x * y.x - p);
    let e2 = opaque(e1 + x.x * y.y);
    let e3 = opaque(e2 + x.y * y.x);
    return vec2<f32>(p, e3 + x.y * y.y);
}

// The running sum of a window: `hi + lo` is the exact sum of the terms added so far, to about 2^-47.
struct Run {
    hi: f32,
    lo: f32,
};

fn run_add(r: Run, x: f32) -> Run {
    let t = two_sum(r.hi, x);
    return Run(t.x, r.lo + t.y);
}

// The correctly rounded f32 of (hi + lo) / count, also where the exact mean is a tie between two f32
// values and whatever the accuracy of the driver's division (Metal's fast-math divide is allowed
// 2.5 ulp): the quotient of the leading part, its exact remainder (q n is formed exactly by
// `two_prod`), the remainder's own quotient refined once the same way, and one rounding at the end.
// A C model of this sequence under clang -O3 -ffast-math -ffp-contract=fast, with a division that
// is off by up to 3 ulp, gives the f64 mean rounded to f32 in 2 million of 2 million windows (without
// the refinement, 28,916 differ: exact ties broken the wrong way).
fn run_mean(r: Run, count: u32) -> f32 {
    let n = f32(count);
    let q = opaque(r.hi / n);
    let p = two_prod(q, n);
    let rem = opaque(opaque(opaque(r.hi - p.x) - p.y) + r.lo);
    let c1 = opaque(rem / n);
    let p2 = two_prod(c1, n);
    let rem2 = opaque(opaque(rem - p2.x) - p2.y);
    let c = opaque(c1 + opaque(rem2 / n));
    return q + c;
}

// The correctly rounded f32 of x / d whatever the accuracy of the driver's division: the quotient,
// its exact remainder and the remainder's quotient refined once, as in `run_mean`. The CPU divides
// in f32 (or rounds an f64 quotient), so this gives its bits where a fast-math divide would be off
// by an ulp.
fn div_cr(x: f32, d: f32) -> f32 {
    let q = opaque(x / d);
    let p = two_prod(q, d);
    let rem = opaque(opaque(x - p.x) - p.y);
    let c1 = opaque(rem / d);
    let p2 = two_prod(c1, d);
    let rem2 = opaque(opaque(rem - p2.x) - p2.y);
    return q + opaque(c1 + opaque(rem2 / d));
}

// Y = a_R R + a_G G + a_B B as the CPU forms it: three rounded products added left to right. Every
// intermediate is hidden, so a fused multiply-add cannot change the rounding the squares and means
// below are sensitive to.
fn luma_of(r: f32, g: f32, b: f32) -> f32 {
    return opaque(opaque(opaque(gp.luma.x * r) + opaque(gp.luma.y * g)) + opaque(gp.luma.z * b));
}

// The 6 quantities of the first pass in the order of `s1`: mean Y, mean Y*Y, mean Cr, mean Y*Cr,
// mean Cb, mean Y*Cb.
fn quantities_at(x: u32, y: u32) -> array<f32, 6> {
    let i = y * gp.dims.z + x;
    let r = img[i];
    let g = img[gp.dims.w + i];
    let b = img[2u * gp.dims.w + i];
    let yl = luma_of(r, g, b);
    let cr = r - yl;
    let cb = b - yl;
    return array<f32, 6>(
        yl,
        opaque(yl * yl),
        cr,
        opaque(yl * cr),
        cb,
        opaque(yl * cb),
    );
}

// Horizontal window means of the six quantities (pass 1).
fn gd_rows(gid: vec3<u32>) {
    let w = gp.dims.x;
    if (gid.x >= w || gid.y >= gp.dims.y) {
        return;
    }
    let r = gp.dst.w;
    let lo = max(gid.x, r) - r;
    let hi = min(gid.x + r, w - 1u);
    var acc = array<Run, 6>(
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
    );
    for (var xs = lo; xs <= hi; xs++) {
        var q = quantities_at(xs, gid.y);
        for (var k = 0u; k < 6u; k++) {
            acc[k] = run_add(acc[k], q[k]);
        }
    }
    let n = hi - lo + 1u;
    let o = gid.y * w + gid.x;
    for (var k = 0u; k < 6u; k++) {
        s1[k * gp.sig.z + o] = run_mean(acc[k], n);
    }
}

// Vertical window means of the six planes of `s1` at one pixel.
fn column_means(x: u32, y: u32, planes: u32) -> array<f32, 6> {
    let w = gp.dims.x;
    let r = gp.dst.w;
    let lo = max(y, r) - r;
    let hi = min(y + r, gp.dims.y - 1u);
    var acc = array<Run, 6>(
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
        Run(0.0, 0.0),
    );
    for (var ys = lo; ys <= hi; ys++) {
        for (var k = 0u; k < planes; k++) {
            acc[k] = run_add(acc[k], s1[k * gp.sig.z + ys * w + x]);
        }
    }
    let n = hi - lo + 1u;
    var mean = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var k = 0u; k < planes; k++) {
        mean[k] = run_mean(acc[k], n);
    }
    return mean;
}

// Vertical means, then the coefficients a and b of both chroma planes (pass 2).
fn gd_cols(gid: vec3<u32>) {
    let w = gp.dims.x;
    if (gid.x >= w || gid.y >= gp.dims.y) {
        return;
    }
    let m = column_means(gid.x, gid.y, 6u);
    let s = sgm[(gp.sig.y + gid.y) * gp.dst.z + gp.sig.x + gid.x];
    let eps = max(opaque(s * s) * gp.k.x, 1e-10);
    // The products that are subtracted from a number of their own size are rounded on their own, as
    // the CPU's stored f32 values are.
    let inv = opaque(div_cr(1.0, max(m[1] - opaque(m[0] * m[0]), 0.0) + eps));
    let o = gid.y * w + gid.x;
    // Cr: mean m[2], correlation with Y m[3]; Cb: m[4] and m[5].
    let a_cr = opaque(opaque(m[3] - opaque(m[0] * m[2])) * inv);
    let a_cb = opaque(opaque(m[5] - opaque(m[0] * m[4])) * inv);
    s2[o] = a_cr;
    s2[gp.sig.z + o] = m[2] - opaque(a_cr * m[0]);
    s2[2u * gp.sig.z + o] = a_cb;
    s2[3u * gp.sig.z + o] = m[4] - opaque(a_cb * m[0]);
}

// Horizontal window means of the four coefficient planes (pass 3), into the first four planes of `s1`.
fn gd_smooth_rows(gid: vec3<u32>) {
    let w = gp.dims.x;
    if (gid.x >= w || gid.y >= gp.dims.y) {
        return;
    }
    let r = gp.dst.w;
    let lo = max(gid.x, r) - r;
    let hi = min(gid.x + r, w - 1u);
    var acc = array<Run, 4>(Run(0.0, 0.0), Run(0.0, 0.0), Run(0.0, 0.0), Run(0.0, 0.0));
    for (var xs = lo; xs <= hi; xs++) {
        for (var k = 0u; k < 4u; k++) {
            acc[k] = run_add(acc[k], s2[k * gp.sig.z + gid.y * w + xs]);
        }
    }
    let n = hi - lo + 1u;
    let o = gid.y * w + gid.x;
    for (var k = 0u; k < 4u; k++) {
        s1[k * gp.sig.z + o] = run_mean(acc[k], n);
    }
}

// Vertical means of the smoothed coefficients and the output of the core (pass 4).
fn gd_out(gid: vec3<u32>) {
    if (gid.x >= gp.core.z || gid.y >= gp.core.w) {
        return;
    }
    let px = gp.core.x + gid.x;
    let py = gp.core.y + gid.y;
    let m = column_means(px, py, 4u);
    let i = py * gp.dims.z + px;
    let r = img[i];
    let g = img[gp.dims.w + i];
    let b = img[2u * gp.dims.w + i];
    let yl = luma_of(r, g, b);
    // The products are rounded on their own as the CPU's are (no fused multiply-add), and the
    // division by the green weight is correctly rounded (`div_cr`), as the CPU's.
    let fcr = opaque(opaque(m[0] * yl) + m[1]);
    let fcb = opaque(opaque(m[2] * yl) + m[3]);
    let red = fcr + yl;
    let blue = fcb + yl;
    let o = gid.y * gp.dst.x + gid.x;
    dst[o] = red;
    dst[gp.dst.y + o] =
        div_cr(opaque(opaque(yl - opaque(gp.luma.x * red)) - opaque(gp.luma.z * blue)), gp.luma.y);
    dst[2u * gp.dst.y + o] = blue;
}
