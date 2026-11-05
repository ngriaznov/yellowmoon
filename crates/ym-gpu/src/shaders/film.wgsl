//@@ common
// The Film node's kernels (docs/24-film.md sections 3.8c, 3.9 and 3.9a; docs/17-gpu.md). Plain functions and the one
// uniform block `FU`; `kernels/film.rs` prepends the generated constants (FILM_*) and appends the declarations and the
// `main` of each pipeline. Every image is a planar f32 image of three planes (`GpuImage`); the pixel (x, y) of the
// photo frame is at `(y - y0) * stride + (x - x0)` of the image whose frame origin is (x0, y0) (the block's a, b and
// c images). A pass computes the window `win` of the frame, so a pixel depends on its absolute coordinates only
// (docs/17 D4, D5): no position is a running sum, and the grain's hash takes the absolute integer coordinates.
//
// Safe under Metal's fast-math (docs/17 K8). The notes are in kernels/film.rs, "Fast-math"; in short: NaN is tested on
// the bits (`is_nan_f`), every log, sqrt and pow has a positive argument by a clamp or a floor in front of it, every
// division is by a clamped or floored value, no `fma`, no loop count depends on the data, and the decisions that
// compare computed values (the tetrahedron of the 3D tables, the Newton candidate that lowers the residual, the
// locus segment hit) are continuous in the result where they tie.

struct FU {
    // The photo frame (or the cell grid of a cell blur): width, height, then the cell grid of an upsample (w, h).
    frame: vec4<i32>,
    // The first input image: frame x and y of its pixel (0, 0), row stride, plane length.
    a: vec4<i32>,
    // The output image: the same.
    b: vec4<i32>,
    // The second input image: the same.
    c: vec4<i32>,
    // The window the pass computes: first frame x and y, width, height.
    win: vec4<i32>,
    // x: the operation, y: a count or radius, z: flags, w: a factor or the seed.
    op: vec4<u32>,
    // Per kernel parameters.
    p: array<vec4<f32>, 16>,
    // Per kernel matrices and ranges.
    m: array<vec4<f32>, 6>,
    // Per kernel integer words (the grain hash constants; the core table's sizes).
    h: array<vec4<u32>, 3>,
}

fn fx_a(x: i32, y: i32, c: u32) -> u32 {
    return c * u32(fu.a.w) + u32(y - fu.a.y) * u32(fu.a.z) + u32(x - fu.a.x);
}

fn fx_b(x: i32, y: i32, c: u32) -> u32 {
    return c * u32(fu.b.w) + u32(y - fu.b.y) * u32(fu.b.z) + u32(x - fu.b.x);
}

fn fx_c(x: i32, y: i32, c: u32) -> u32 {
    return c * u32(fu.c.w) + u32(y - fu.c.y) * u32(fu.c.z) + u32(x - fu.c.x);
}

// A value held to 0 when NaN (by the bits), else clamped to [lo, hi].
fn fx_clamp_nan0(x: f32, lo: f32, hi: f32) -> f32 {
    if (is_nan_f(x)) {
        return 0.0;
    }
    return clamp(x, lo, hi);
}

// The position along a table axis of n nodes: clamped into 0..n-1, NaN is 0 (bake.rs `grid_pos`).
fn fx_grid(p: f32, n: u32) -> f32 {
    return fx_clamp_nan0(p, 0.0, f32(n - 1u));
}

//@@ colour_in
// ---- P1: the scene stage (stage 1) and T1 ------------------------------------------------------------------------

// The tolerance on the segment parameter of the locus hit, in f32 (the CPU uses 1e-12 in f64): a ray through a vertex
// of the polygon must hit one of the two segments that meet there.
const FX_S_TOL: f32 = 1.0e-5;

fn fx_tanh_pos(x: f32) -> f32 {
    // tanh for x >= 0 without a large intermediate: 1 - 2 / (exp(2x) + 1), the argument held to 20.
    let e = exp(2.0 * min(x, 20.0));
    return 1.0 - 2.0 / (e + 1.0);
}

fn ci_locus_distance(u: vec2<f32>, w: vec2<f32>, n: u32) -> f32 {
    var best = 1.0e30;
    for (var k = 0u; k < n; k++) {
        let k1 = select(k + 1u, 0u, k + 1u == n);
        let a = vec2<f32>(locus[2u * k], locus[2u * k + 1u]);
        let b = vec2<f32>(locus[2u * k1], locus[2u * k1 + 1u]);
        let e = b - a;
        let den = u.x * e.y - u.y * e.x;
        if (abs(den) < 1.0e-18) {
            continue;
        }
        let aw = a - w;
        let t = (aw.x * e.y - aw.y * e.x) / den;
        let s = (aw.x * u.y - aw.y * u.x) / den;
        if (t > 0.0 && s >= -FX_S_TOL && s <= 1.0 + FX_S_TOL && t < best) {
            best = t;
        }
    }
    return select(1.0, best, best < 1.0e29);
}

// scene.rs `compress`: the soft compression of a chromaticity toward the reference white.
fn ci_compress(p: vec2<f32>, w: vec2<f32>, n: u32) -> vec2<f32> {
    let d = p - w;
    let r = sqrt(d.x * d.x + d.y * d.y);
    if (r < 1.0e-12) {
        return p;
    }
    let u = d / r;
    let big_r = ci_locus_distance(u, w, n);
    let k = fu.p[1].y * big_r;
    if (r <= k) {
        return p;
    }
    let l = fu.p[1].z * big_r;
    let r2 = k + (l - k) * fx_tanh_pos((r - k) / (l - k));
    return w + u * r2;
}

// Bilinear T1 at the table position (px, py) given as pairs hi + lo (cells), held at the edge.
fn ci_t1_pos(pxh: f32, pxl: f32, pyh: f32, pyl: f32, nx: u32, ny: u32) -> vec3<f32> {
    // A position held at the edge has no low part; NaN is 0.
    let gx = fx_grid(pxh, nx);
    let gy = fx_grid(pyh, ny);
    let lx = select(0.0, pxl, gx == pxh);
    let ly = select(0.0, pyl, gy == pyh);
    let ix = min(u32(floor(gx)), nx - 2u);
    let iy = min(u32(floor(gy)), ny - 2u);
    let tx = px_op(px_op(gx - f32(ix)) + lx);
    let ty = px_op(px_op(gy - f32(iy)) + ly);
    let o00 = 4u * (ix + nx * iy);
    let o10 = o00 + 4u;
    let o01 = o00 + 4u * nx;
    let o11 = o01 + 4u;
    let a = vec3<f32>(t1[o00], t1[o00 + 1u], t1[o00 + 2u]);
    let b = vec3<f32>(t1[o10], t1[o10 + 1u], t1[o10 + 2u]);
    let c = vec3<f32>(t1[o01], t1[o01 + 1u], t1[o01 + 2u]);
    let d = vec3<f32>(t1[o11], t1[o11 + 1u], t1[o11 + 2u]);
    let lo = vec3<f32>(px_op(a.x + px_op((b.x - a.x) * tx)), px_op(a.y + px_op((b.y - a.y) * tx)), px_op(a.z + px_op((b.z - a.z) * tx)));
    let hi = vec3<f32>(px_op(c.x + px_op((d.x - c.x) * tx)), px_op(c.y + px_op((d.y - c.y) * tx)), px_op(c.z + px_op((d.z - c.z) * tx)));
    return vec3<f32>(px_op(lo.x + px_op((hi.x - lo.x) * ty)), px_op(lo.y + px_op((hi.y - lo.y) * ty)), px_op(lo.z + px_op((hi.z - lo.z) * ty)));
}

// The table position of the coordinate x_hi + x_lo for 1 / step = k_hi + k_lo, as a pair.
fn ci_pos(xh: f32, xl: f32, kh: f32, kl: f32) -> vec2<f32> {
    let p = px_two_prod(xh, kh);
    return vec2<f32>(p.x, px_op(px_op(p.y + px_op(xh * kl)) + px_op(xl * kh)));
}

// The quotient a / b as a pair (hi, lo): the correctly rounded quotient and the quotient of its exact remainder.
fn ci_div_pair(a: f32, b: f32) -> vec2<f32> {
    let q = px_div(a, b);
    let p = px_two_prod(q, b);
    let r = px_op(px_op(a - p.x) - p.y);
    return vec2<f32>(q, px_op(r / b));
}

// ---- the plain f32 colour-in of a view (the arithmetic before F-08d, `px_fast()`) --------------------------------------
//
// A view runs the Film passes with the compiler's own f32 operations; the pair arithmetic is for the export, which the owner
// accepted slower (docs/24 section 3.9c, "Speed"). These are the former functions, unchanged.

// Bilinear T1 at the chromaticity (x, y), held at the edge.
fn ci_t1_plain(x: f32, y: f32, nx: u32, ny: u32, step: f32) -> vec3<f32> {
    let fxp = fx_grid(x / step, nx);
    let fyp = fx_grid(y / step, ny);
    let ix = min(u32(floor(fxp)), nx - 2u);
    let iy = min(u32(floor(fyp)), ny - 2u);
    let tx = fxp - f32(ix);
    let ty = fyp - f32(iy);
    let o00 = 4u * (ix + nx * iy);
    let o10 = o00 + 4u;
    let o01 = o00 + 4u * nx;
    let o11 = o01 + 4u;
    let a = vec3<f32>(t1[o00], t1[o00 + 1u], t1[o00 + 2u]);
    let b = vec3<f32>(t1[o10], t1[o10 + 1u], t1[o10 + 2u]);
    let c = vec3<f32>(t1[o01], t1[o01 + 1u], t1[o01 + 2u]);
    let d = vec3<f32>(t1[o11], t1[o11 + 1u], t1[o11 + 2u]);
    let lo = a + (b - a) * tx;
    let hi = c + (d - c) * tx;
    return lo + (hi - lo) * ty;
}

fn ci_plain(x: i32, y: i32, lc: vec3<f32>) {
    let rgb = logc4_decode3(lc);
    let xyz = mv3(fu.m[0].xyz, fu.m[1].xyz, fu.m[2].xyz, rgb);
    let sum = xyz.x + xyz.y + xyz.z;
    let white = fu.p[0].xy;
    let n = fu.op.y;
    var lum = 0.0;
    if (!is_nan_f(xyz.y) && abs(xyz.y) < 1.0e30) {
        lum = max(xyz.y, 0.0);
    }
    var q = white;
    if (!is_nan_f(sum) && sum > 0.0 && sum < 1.0e30) {
        q = ci_compress(vec2<f32>(xyz.x / sum, xyz.y / sum), white, n);
    }
    let v = ci_t1_plain(q.x, q.y, fu.op.z, fu.op.w, fu.p[1].x);
    let scale = lum * fu.p[0].z;
    for (var c = 0u; c < 3u; c++) {
        dst[fx_b(x, y, c)] = max(scale * v[c], 1.0e-30);
    }
}

// LogC4 / AWG4 in, the raw exposure E (linear) out: stages 1 and 2, then the power of ten (the CPU's
// `pow10(exposure_of(..))` is `max(scale * T1, 1e-30)` up to the rounding of a log10 and a pow10). The decode, the
// luminance, the chromaticity and the position in T1 are carried with their rounding errors (`precise.wgsl`), so that
// the exposure is within a few 1e-8 of the CPU's before the f32 store.
fn ci_main(gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if (ix >= fu.win.z || iy >= fu.win.w) {
        return;
    }
    let x = fu.win.x + ix;
    let y = fu.win.y + iy;
    let lc = vec3<f32>(src[fx_a(x, y, 0u)], src[fx_a(x, y, 1u)], src[fx_a(x, y, 2u)]);
    if (px_fast()) {
        ci_plain(x, y, lc);
        return;
    }
    let rgb = px_logc4_decode3(lc);
    let xyz = px_mv3(fu.m[0].xyz, fu.m[1].xyz, fu.m[2].xyz, rgb)
        + vec3<f32>(dot(fu.m[3].xyz, rgb), dot(fu.m[4].xyz, rgb), dot(fu.m[5].xyz, rgb));
    let sum = px_op(px_op(xyz.x + xyz.y) + xyz.z);
    let white = fu.p[0].xy;
    let n = fu.op.y;
    var lum = 0.0;
    if (!is_nan_f(xyz.y) && abs(xyz.y) < 1.0e30) {
        lum = max(xyz.y, 0.0);
    }
    var pos = vec4<f32>(0.0);
    var q = white;
    if (!is_nan_f(sum) && sum > 0.0 && sum < 1.0e30) {
        let cx = ci_div_pair(xyz.x, sum);
        let cy = ci_div_pair(xyz.y, sum);
        q = ci_compress(vec2<f32>(cx.x, cy.x), white, n);
        if (q.x == cx.x && q.y == cy.x) {
            // Not compressed: the chromaticity keeps its low parts into the table position.
            let px = ci_pos(cx.x, cx.y, fu.p[2].x, fu.p[2].y);
            let py = ci_pos(cy.x, cy.y, fu.p[2].x, fu.p[2].y);
            pos = vec4<f32>(px.x, px.y, py.x, py.y);
        } else {
            let px = ci_pos(q.x, 0.0, fu.p[2].x, fu.p[2].y);
            let py = ci_pos(q.y, 0.0, fu.p[2].x, fu.p[2].y);
            pos = vec4<f32>(px.x, px.y, py.x, py.y);
        }
    } else {
        let px = ci_pos(q.x, 0.0, fu.p[2].x, fu.p[2].y);
        let py = ci_pos(q.y, 0.0, fu.p[2].x, fu.p[2].y);
        pos = vec4<f32>(px.x, px.y, py.x, py.y);
    }
    let v = ci_t1_pos(pos.x, pos.y, pos.z, pos.w, fu.op.z, fu.op.w);
    let scale = px_two_prod(lum, fu.p[0].z);
    let scale_lo = px_op(scale.y + px_op(lum * fu.p[0].w));
    for (var c = 0u; c < 3u; c++) {
        let e = px_op(px_op(scale.x * v[c]) + px_op(scale_lo * v[c]));
        dst[fx_b(x, y, c)] = max(e, 1.0e-30);
    }
}

//@@ spatial
// ---- P2, P3's diffusion, P4's cloud: the spatial operations ------------------------------------------------------

const SP_SCALE: u32 = 0u;
const SP_ROWS: u32 = 1u;
const SP_COLS: u32 = 2u;
const SP_CELLS: u32 = 3u;
const SP_UPSAMPLE: u32 = 4u;
const SP_ADD: u32 = 5u;

// The centre weight of the sampled Gaussian `taps` and the horizontal sum: w0 s(x) + sum_i w_i (s(x - i) + s(x + i)),
// the reads clamped into [0, bw), in the CPU's order.
fn sp_rows(x: i32, y: i32, c: u32, r: i32) -> f32 {
    let bw = fu.frame.x;
    var acc = taps[0] * src[fx_a(x, y, c)];
    for (var i = 1; i <= r; i++) {
        let xl = max(x - i, 0);
        let xr = min(x + i, bw - 1);
        acc += taps[u32(i)] * (src[fx_a(xl, y, c)] + src[fx_a(xr, y, c)]);
    }
    return acc;
}

fn sp_cols(x: i32, y: i32, c: u32, r: i32) -> f32 {
    let bh = fu.frame.y;
    var acc = taps[0] * src[fx_a(x, y, c)];
    for (var i = 1; i <= r; i++) {
        let yu = max(y - i, 0);
        let yd = min(y + i, bh - 1);
        acc += taps[u32(i)] * (src[fx_a(x, yu, c)] + src[fx_a(x, yd, c)]);
    }
    return acc;
}

// The mean of the f x f block of the cell (i, j), the block cut to the frame, row-major.
fn sp_cell(i: i32, j: i32, c: u32, f: i32) -> f32 {
    let fw = fu.frame.x;
    let fh = fu.frame.y;
    let px0 = i * f;
    let px1 = min((i + 1) * f, fw);
    let py0 = j * f;
    let py1 = min((j + 1) * f, fh);
    var acc = 0.0;
    for (var yy = py0; yy < py1; yy++) {
        for (var xx = px0; xx < px1; xx++) {
            acc += src[fx_a(xx, yy, c)];
        }
    }
    return acc / f32((py1 - py0) * (px1 - px0));
}

// The cell-grid coordinate of pixel x for cells of f pixels: (x - (f - 1) / 2) / f. f is a power of two, so the
// subtraction and the division are exact in f32 for any frame the design allows (no position of order 1e4 is rounded).
fn sp_centre(x: i32, f: i32) -> f32 {
    return (f32(x) - 0.5 * f32(f - 1)) / f32(f);
}

fn sp_upsample(x: i32, y: i32, c: u32, f: i32) -> f32 {
    let ncx = fu.frame.z;
    let ncy = fu.frame.w;
    let ux = sp_centre(x, f);
    let uy = sp_centre(y, f);
    let ixf = floor(ux);
    let jyf = floor(uy);
    let tx = ux - ixf;
    let ty = uy - jyf;
    let i0 = min(max(i32(ixf), 0), ncx - 1);
    let i1 = min(max(i32(ixf) + 1, 0), ncx - 1);
    let j0 = min(max(i32(jyf), 0), ncy - 1);
    let j1 = min(max(i32(jyf) + 1, 0), ncy - 1);
    let a = src[fx_a(i0, j0, c)];
    let b = src[fx_a(i1, j0, c)];
    let cc = src[fx_a(i0, j1, c)];
    let d = src[fx_a(i1, j1, c)];
    let top = a * (1.0 - tx) + b * tx;
    let bot = cc * (1.0 - tx) + d * tx;
    return top * (1.0 - ty) + bot * ty;
}

fn sp_main(gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if (ix >= fu.win.z || iy >= fu.win.w) {
        return;
    }
    let x = fu.win.x + ix;
    let y = fu.win.y + iy;
    let op = fu.op.x;
    let r = i32(fu.op.y);
    let accumulate = (fu.op.z & 1u) != 0u;
    let wgt = fu.p[0].xyz;
    for (var c = 0u; c < 3u; c++) {
        var v = 0.0;
        if (op == SP_SCALE) {
            dst[fx_b(x, y, c)] = wgt[c] * src[fx_a(x, y, c)];
            continue;
        } else if (op == SP_ROWS) {
            v = sp_rows(x, y, c, r);
        } else if (op == SP_COLS) {
            v = sp_cols(x, y, c, r);
        } else if (op == SP_CELLS) {
            v = sp_cell(x, y, c, i32(fu.op.w));
        } else if (op == SP_UPSAMPLE) {
            v = sp_upsample(x, y, c, i32(fu.op.w));
        } else {
            // SP_ADD: a + norm * c.
            dst[fx_b(x, y, c)] = src[fx_a(x, y, c)] + fu.p[0].x * src2[fx_c(x, y, c)];
            continue;
        }
        let o = fx_b(x, y, c);
        if (accumulate) {
            dst[o] = dst[o] + wgt[c] * v;
        } else {
            dst[o] = v;
        }
    }
}

//@@ develop
// ---- P3: the film curves, the coupler step and the develop ------------------------------------------------------

const DV_DENSITY: u32 = 0u;
const DV_INHIBITOR: u32 = 1u;
const DV_WITH_Q: u32 = 2u;

// The 1D tables are FX_CURVE_ENTRIES (4096) f32 each, three channels in a row. p[1].xyz is the lower end of the axes
// and p[2].xyz the factor (entries - 1) / (hi - lo).
fn dv_curve(ch: u32, x: f32) -> f32 {
    let n = FX_CURVE_ENTRIES;
    let lo = fu.p[1][ch];
    let k = fu.p[2][ch];
    let pos = fx_grid((x - lo) * k, n);
    let i = min(u32(floor(pos)), n - 2u);
    let t = pos - f32(i);
    let a = curves[ch * n + i];
    let b = curves[ch * n + i + 1u];
    return a + (b - a) * t;
}

// The slope of the piece x lies in; 0 outside the axis.
fn dv_slope(ch: u32, x: f32) -> f32 {
    let n = FX_CURVE_ENTRIES;
    let lo = fu.p[1][ch];
    let k = fu.p[2][ch];
    let hi = lo + f32(n - 1u) / k;
    if (is_nan_f(x) || x < lo || x > hi) {
        return 0.0;
    }
    let pos = (x - lo) * k;
    let i = min(u32(floor(pos)), n - 2u);
    return (curves[ch * n + i + 1u] - curves[ch * n + i]) * k;
}

fn dv_c0(le: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dv_curve(0u, le.x), dv_curve(1u, le.y), dv_curve(2u, le.z));
}

fn dv_mrow(i: u32) -> vec3<f32> {
    return fu.m[i].xyz;
}

fn dv_inh(d: vec3<f32>) -> vec3<f32> {
    let a = fu.p[0].x;
    return a * vec3<f32>(dot(dv_mrow(0u), d), dot(dv_mrow(1u), d), dot(dv_mrow(2u), d));
}

// g_i = C0_i(log E_i - amount (M d)_i) - d_i.
fn dv_resid(le: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let q = dv_inh(d);
    return vec3<f32>(dv_curve(0u, le.x - q.x), dv_curve(1u, le.y - q.y), dv_curve(2u, le.z - q.z)) - d;
}

fn dv_norm(g: vec3<f32>) -> f32 {
    let n = abs(g.x) + abs(g.y) + abs(g.z);
    return select(1.0e30, n, !is_nan_f(n) && n < 1.0e30);
}

// Cramer's rule, the CPU's `solve3`; the third return is 0 when singular.
fn dv_solve3(j0: vec3<f32>, j1: vec3<f32>, j2: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    let det = j0.x * (j1.y * j2.z - j1.z * j2.y) - j0.y * (j1.x * j2.z - j1.z * j2.x) + j0.z * (j1.x * j2.y - j1.y * j2.x);
    if (is_nan_f(det) || abs(det) < 1.0e-30) {
        return vec4<f32>(0.0);
    }
    let dx = b.x * (j1.y * j2.z - j1.z * j2.y) - j0.y * (b.y * j2.z - j1.z * b.z) + j0.z * (b.y * j2.y - j1.y * b.z);
    let dy = j0.x * (b.y * j2.z - j1.z * b.z) - b.x * (j1.x * j2.z - j1.z * j2.x) + j0.z * (j1.x * b.z - b.y * j2.x);
    let dz = j0.x * (j1.y * b.z - b.y * j2.y) - j0.y * (j1.x * b.z - b.y * j2.x) + b.x * (j1.x * j2.y - j1.y * j2.x);
    return vec4<f32>(dx / det, dy / det, dz / det, 1.0);
}

// Stage 5, the density with the couplers acting: a damped Newton iteration on the three layers, a FIXED number of
// iterations (FX_COUPLER_ITERATIONS) and of step halvings (FX_COUPLER_TRIALS); an iteration that finds no step that
// lowers the residual keeps d (the CPU stops there).
fn dv_density(le: vec3<f32>) -> vec3<f32> {
    var d = dv_c0(le);
    let a = fu.p[0].x;
    if (a == 0.0) {
        return d;
    }
    var g = dv_resid(le, d);
    for (var it = 0u; it < FX_COUPLER_ITERATIONS; it++) {
        // The Jacobian: -s_i a M_ij - delta_ij, s_i the slope at the effective exposure.
        let q = dv_inh(d);
        let s = vec3<f32>(dv_slope(0u, le.x - q.x), dv_slope(1u, le.y - q.y), dv_slope(2u, le.z - q.z));
        let j0 = -s.x * a * dv_mrow(0u) - vec3<f32>(1.0, 0.0, 0.0);
        let j1 = -s.y * a * dv_mrow(1u) - vec3<f32>(0.0, 1.0, 0.0);
        let j2 = -s.z * a * dv_mrow(2u) - vec3<f32>(0.0, 0.0, 1.0);
        let st = dv_solve3(j0, j1, j2, -g);
        let ng = dv_norm(g);
        var found = st.w == 0.0 || ng < FX_COUPLER_TOL;
        var t = 1.0;
        for (var k = 0u; k < FX_COUPLER_TRIALS; k++) {
            if (!found) {
                let cand = d + t * st.xyz;
                let gc = dv_resid(le, cand);
                if (dv_norm(gc) < ng) {
                    d = cand;
                    g = gc;
                    found = true;
                }
            }
            t *= 0.5;
        }
    }
    return d;
}

fn dv_log10_plain(e: vec3<f32>) -> vec3<f32> {
    return log(max(e, vec3<f32>(1.0e-30))) * FX_LOG10_E;
}

// ---- the same develop, carried as pairs (F-08d, docs/24 section 3.9c) ----------------------------------------------------
//
// The CPU's pass is f64 inside and stores an f32 density. The log of the exposure, the position in the 4096-entry curve, the
// interpolation, the inhibitor and the coupler iteration are pairs here (hi + lo, `precise.wgsl`), so the f32 that is stored
// is the rounding of a value within a few 1e-9 of the CPU's, not a value an ulp or two away: the density is the exponent of
// the print's light (the display is 10^-D), and 6e-8 of a density of 2 is 3e-7 of the light.

struct DvPair {
    hi: vec3<f32>,
    lo: vec3<f32>,
}

const DV_LOG10_2_HI: f32 = 0.3010300099849701;
const DV_LOG10_2_LO: f32 = -1.432098889475597e-8;

// log10 of a positive f32 as a pair.
fn dv_log10_one(x: f32) -> vec2<f32> {
    let l = px_log2_parts(max(x, 1.0e-30));
    let a = px_two_prod(l.x, DV_LOG10_2_HI);
    let b = px_two_prod(l.y, DV_LOG10_2_HI);
    let s = px_two_sum(a.x, b.x);
    let lo = px_op(px_op(px_op(s.y + a.y) + b.y) + px_op(px_op(l.x * DV_LOG10_2_LO) + px_op(l.y * DV_LOG10_2_LO)));
    return px_two_sum(s.x, lo);
}

fn dv_log10_pair(e: vec3<f32>) -> DvPair {
    let x = dv_log10_one(e.x);
    let y = dv_log10_one(e.y);
    let z = dv_log10_one(e.z);
    return DvPair(vec3<f32>(x.x, y.x, z.x), vec3<f32>(x.y, y.y, z.y));
}

// The film curve of channel `ch` at xh + xl as a pair; the axis ends are pairs (p[1] + p[3], the factor p[2] + p[4]).
fn dv_curve_pair(ch: u32, xh: f32, xl: f32) -> vec2<f32> {
    let n = FX_CURVE_ENTRIES;
    let d = px_two_sum(xh, -fu.p[1][ch]);
    let dl = px_op(px_op(d.y + xl) - fu.p[3][ch]);
    let m = px_two_prod(d.x, fu.p[2][ch]);
    var ph = m.x;
    var pl = px_op(px_op(m.y + px_op(d.x * fu.p[4][ch])) + px_op(dl * fu.p[2][ch]));
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

fn dv_c0_pair(x: DvPair) -> DvPair {
    let r = dv_curve_pair(0u, x.hi.x, x.lo.x);
    let g = dv_curve_pair(1u, x.hi.y, x.lo.y);
    let b = dv_curve_pair(2u, x.hi.z, x.lo.z);
    return DvPair(vec3<f32>(r.x, g.x, b.x), vec3<f32>(r.y, g.y, b.y));
}

// a * M d for a density pair, one rounding (the matrix rows are f32 rounded from f64; their error is 3e-8 of a number of
// the size of the inhibitor, a few 1e-9 of a density).
fn dv_inh_pair(d: DvPair) -> vec3<f32> {
    let a = fu.p[0].x;
    var q = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i++) {
        let r = dv_mrow(i);
        q[i] = a * px_op(px_dot3(r, d.hi) + px_dot3(r, d.lo));
    }
    return q;
}

// x = le - q as a pair.
fn dv_sub(le: DvPair, q: vec3<f32>) -> DvPair {
    var hi = vec3<f32>(0.0);
    var lo = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let s = px_two_sum(le.hi[c], -q[c]);
        hi[c] = s.x;
        lo[c] = px_op(s.y + le.lo[c]);
    }
    return DvPair(hi, lo);
}

// g_i = C0_i(log E_i - amount (M d)_i) - d_i, as f32 (it is the small difference of two pairs).
fn dv_resid_pair(le: DvPair, d: DvPair) -> vec3<f32> {
    let c = dv_c0_pair(dv_sub(le, dv_inh_pair(d)));
    return vec3<f32>(
        px_op(px_op(c.hi.x - d.hi.x) + px_op(c.lo.x - d.lo.x)),
        px_op(px_op(c.hi.y - d.hi.y) + px_op(c.lo.y - d.lo.y)),
        px_op(px_op(c.hi.z - d.hi.z) + px_op(c.lo.z - d.lo.z)),
    );
}

fn dv_add(d: DvPair, step: vec3<f32>) -> DvPair {
    var hi = vec3<f32>(0.0);
    var lo = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let s = px_two_sum(d.hi[c], step[c]);
        hi[c] = s.x;
        lo[c] = px_op(s.y + d.lo[c]);
    }
    return DvPair(hi, lo);
}

// Stage 5 as `dv_density`, with the density and the residual carried as pairs; the Jacobian and the step are f32.
fn dv_density_pair(le: DvPair) -> DvPair {
    var d = dv_c0_pair(le);
    let a = fu.p[0].x;
    if (a == 0.0) {
        return d;
    }
    var g = dv_resid_pair(le, d);
    for (var it = 0u; it < FX_COUPLER_ITERATIONS; it++) {
        let dq = dv_inh_pair(d);
        let x = dv_sub(le, dq);
        let s = vec3<f32>(dv_slope(0u, x.hi.x), dv_slope(1u, x.hi.y), dv_slope(2u, x.hi.z));
        let j0 = -s.x * a * dv_mrow(0u) - vec3<f32>(1.0, 0.0, 0.0);
        let j1 = -s.y * a * dv_mrow(1u) - vec3<f32>(0.0, 1.0, 0.0);
        let j2 = -s.z * a * dv_mrow(2u) - vec3<f32>(0.0, 0.0, 1.0);
        let st = dv_solve3(j0, j1, j2, -g);
        let ng = dv_norm(g);
        var found = st.w == 0.0 || ng < FX_COUPLER_TOL;
        var t = 1.0;
        for (var k = 0u; k < FX_COUPLER_TRIALS; k++) {
            if (!found) {
                let cand = dv_add(d, t * st.xyz);
                let gc = dv_resid_pair(le, cand);
                if (dv_norm(gc) < ng) {
                    d = cand;
                    g = gc;
                    found = true;
                }
            }
            t *= 0.5;
        }
    }
    return d;
}

fn dv_main(gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if (ix >= fu.win.z || iy >= fu.win.w) {
        return;
    }
    let x = fu.win.x + ix;
    let y = fu.win.y + iy;
    let e = vec3<f32>(src[fx_a(x, y, 0u)], src[fx_a(x, y, 1u)], src[fx_a(x, y, 2u)]);
    var out = vec3<f32>(0.0);
    if (px_fast()) {
        // A view: the plain f32 develop (the arithmetic before F-08d).
        let le = dv_log10_plain(e);
        if (fu.op.x == DV_DENSITY) {
            out = dv_density(le);
        } else if (fu.op.x == DV_INHIBITOR) {
            out = dv_inh(dv_density(le));
        } else {
            let q = vec3<f32>(src2[fx_c(x, y, 0u)], src2[fx_c(x, y, 1u)], src2[fx_c(x, y, 2u)]);
            out = vec3<f32>(dv_curve(0u, le.x - q.x), dv_curve(1u, le.y - q.y), dv_curve(2u, le.z - q.z));
        }
        for (var c = 0u; c < 3u; c++) {
            dst[fx_b(x, y, c)] = out[c];
        }
        return;
    }
    let le = dv_log10_pair(e);
    if (fu.op.x == DV_DENSITY) {
        let d = dv_density_pair(le);
        out = d.hi + d.lo;
    } else if (fu.op.x == DV_INHIBITOR) {
        out = dv_inh_pair(dv_density_pair(le));
    } else {
        let q = vec3<f32>(src2[fx_c(x, y, 0u)], src2[fx_c(x, y, 1u)], src2[fx_c(x, y, 2u)]);
        let c = dv_c0_pair(dv_sub(le, q));
        out = c.hi + c.lo;
    }
    for (var c = 0u; c < 3u; c++) {
        dst[fx_b(x, y, c)] = out[c];
    }
}

//@@ grain
// ---- P4: the grain ----------------------------------------------------------------------------------------------

fn gr_fmix(h_in: u32) -> u32 {
    var h = h_in;
    h ^= h >> 16u;
    h = h * fu.h[1].z;
    h ^= h >> 13u;
    h = h * fu.h[1].w;
    h ^= h >> 16u;
    return h;
}

// grain.rs `hash32`: exact integer maths. The constants are fu.h[0] (seed xor, x, y, channel * 8 + sublayer) and
// fu.h[1].xy (the word number's multiplier and addend), fu.h[1].zw the finalizer's.
fn gr_hash(seed: u32, x: u32, y: u32, channel: u32, sublayer: u32, k: u32) -> u32 {
    var h = gr_fmix(seed ^ fu.h[0].x);
    h = gr_fmix(h ^ (x * fu.h[0].y));
    h = gr_fmix(h ^ (y * fu.h[0].z));
    h = gr_fmix(h ^ ((channel * 8u + sublayer) * fu.h[0].w));
    return gr_fmix(h ^ (k * fu.h[1].x + fu.h[1].y));
}

// The integer sum of the four centred 16-bit uniforms of a draw, exactly (|sum| <= 262140: exact in f32 as well).
fn gr_sum(seed: u32, x: u32, y: u32, channel: u32, sublayer: u32) -> i32 {
    let a = gr_hash(seed, x, y, channel, sublayer, 0u);
    let b = gr_hash(seed, x, y, channel, sublayer, 1u);
    return 2 * i32(a & 0xffffu) - 65535 + 2 * i32(a >> 16u) - 65535 + 2 * i32(b & 0xffffu) - 65535 + 2 * i32(b >> 16u) - 65535;
}

// Table of one channel and sublayer: 1024 entries over 0 to d_top, linear between.
fn gr_amp(ch: u32, sub: u32, dens: f32) -> f32 {
    let n = FX_GRAIN_ENTRIES;
    let x = fx_clamp_nan0(dens / fu.p[1][ch], 0.0, 1.0) * f32(n - 1u);
    let i = min(u32(x), n - 2u);
    let f = x - f32(i);
    let o = (ch * 3u + sub) * n + i;
    return lut[o] + f * (lut[o + 1u] - lut[o]);
}

fn gr_main(gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    if (ix >= fu.win.z || iy >= fu.win.w) {
        return;
    }
    let x = fu.win.x + ix;
    let y = fu.win.y + iy;
    let seed = fu.op.w;
    if (fu.op.x == 1u) {
        // The probe of the tests: the two words of a draw and its integer sum for the channel op.y, sublayer op.z.
        dst[fx_b(x, y, 0u)] = bitcast<f32>(gr_hash(seed, u32(x), u32(y), fu.op.y, fu.op.z, 0u));
        dst[fx_b(x, y, 1u)] = bitcast<f32>(gr_hash(seed, u32(x), u32(y), fu.op.y, fu.op.z, 1u));
        dst[fx_b(x, y, 2u)] = f32(gr_sum(seed, u32(x), u32(y), fu.op.y, fu.op.z));
        return;
    }
    let scale = fu.p[0].x;
    for (var c = 0u; c < 3u; c++) {
        let dens = src[fx_a(x, y, c)];
        var g = 0.0;
        for (var s = 0u; s < 3u; s++) {
            let n = f32(gr_sum(seed, u32(x), u32(y), c, s)) * FX_NORMAL_SCALE;
            g += gr_amp(c, s, dens) * n;
        }
        dst[fx_b(x, y, c)] = g * scale;
    }
}
