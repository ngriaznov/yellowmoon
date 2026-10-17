// R8 Richardson-Lucy capture sharpening: the GPU port of
// `ym_gpu::cpu::sharpen::richardson_lucy_whole` (`docs/05-raw-processing.md` section R8). Deconvolution
// of the linear luma with a Gaussian PSF whose sigma grows with the squared radius from the frame
// centre, `sigma(r) = rho (1 + r^2 / 4)`, damped against noise, `n = round(A / 30)` iterations (0 to 10):
//
//   d = max(Y, 0), u0 = d
//   per iteration:  e = B u,  q = 1 + w (d - e) / max(e, 1e-4),  w = r^2 / (r^2 + (lambda sigma_n)^2),
//                   r = d - e,   u = max(u B^T q, 0)
//   result: every channel + (u - d)
//
// B and its transpose are separable gathers (rows, then columns) with the kernel of the pixel being
// written, interpolated between the 33 tabulated Gaussians of the plan's bank (`lv` = the pixel's
// fractional level, made once from the squared radius). The kernels, the bank and the rest of the
// plan come from the CPU as data; the squared radius of a pixel is the sum of a column term and a row
// term made on the host in f64 from the frame position (`axes`), so a pixel's PSF is a function of
// its position in the frame and not of the rectangle it is computed in (D4).
//
// Planes of the working buffer `wk` (each w * h of the compute rectangle, no padding):
//   0 d, 1 u, 2 lv (the level), 3 q, 4 tmp. Passes: `rl_prepare` (d, u and lv for the whole
// rectangle), then per iteration `rl_blur_rows` (u to tmp), `rl_blur_cols_ratio` (tmp to q: the
// second half of B u and the damped ratio), `rl_blur_rows` (q to tmp), `rl_blur_cols_update` (tmp to
// u: the second half of B^T q and the update), and at the end `rl_output` for the core.
//
// The sums run in the CPU's order (taps from the left, rows then columns) in f32, where the CPU's
// also are f32. Borders mirror at the edge of the input rectangle (`cpu::mirror`); a tile whose
// rectangle carries the plan's halo (2 n ceil(3 sigma_max), capped at 24 px) gives the whole-frame
// bits (D5), and beyond the cap by about 2e-7, as the CPU's bands do.
//
// Fast-math (Metal, K8): a positive-weight blur, a ratio and a product; nothing here rests on exact
// rounding or on infinities. A contracted multiply-add moves a tap sum by an ulp.
//
// Bindings: 0 uniform, 1 the kernel bank (33 rows of `taps`), 2 the axis terms (columns, then rows),
// 3 the input planes, 4 the sigma plane, 5 the working planes, 6 the output planes (the core).

struct RlParams {
    dims: vec4<u32>,   // input width, height, row stride, plane length
    core: vec4<u32>,   // core x, y, width, height in the input
    dst: vec4<u32>,    // output row stride, output plane length, sigma row stride, working plane length
    cfg: vec4<u32>,    // radius, taps (2 radius + 1), iterations, source plane of `rl_blur_rows`
    sig: vec4<u32>,    // the input's origin in the sigma plane (x, y), 0, 0
    k: vec4<f32>,      // rho, lo (the smallest level's sigma), 1 / step, lambda (the damping sigmas)
    luma: vec4<f32>,   // luma weights (R, G, B), the floor of the ratio
};

@group(0) @binding(0) var<uniform> rp: RlParams;
@group(0) @binding(1) var<storage, read> bank: array<f32>;
@group(0) @binding(2) var<storage, read> axes: array<f32>;
@group(0) @binding(3) var<storage, read> img: array<f32>;
@group(0) @binding(4) var<storage, read> sgm: array<f32>;
@group(0) @binding(5) var<storage, read_write> wk: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

const P_D: u32 = 0u;
const P_U: u32 = 1u;
const P_LV: u32 = 2u;
const P_Q: u32 = 3u;
const P_TMP: u32 = 4u;

fn luma_in(i: u32) -> f32 {
    return rp.luma.x * img[i] + rp.luma.y * img[rp.dims.w + i] + rp.luma.z * img[2u * rp.dims.w + i];
}

// d, u and the PSF level of every pixel of the rectangle.
fn rl_prepare(gid: vec3<u32>) {
    let w = rp.dims.x;
    if (gid.x >= w || gid.y >= rp.dims.y) {
        return;
    }
    let pp = rp.dst.w;
    let o = gid.y * w + gid.x;
    let d = max(luma_in(gid.y * rp.dims.z + gid.x), 0.0);
    wk[P_D * pp + o] = d;
    wk[P_U * pp + o] = d;
    let r2 = axes[gid.x] + axes[w + gid.y];
    let psf = rp.k.x * (1.0 + 0.25 * r2);
    wk[P_LV * pp + o] = clamp((psf - rp.k.y) * rp.k.z, 0.0, 32.0);
}

// The tap `k` of the kernel of a pixel with level `lv`: the CPU's `(1 - f) a[k] + f b[k]` between the
// two tabulated rows around the level.
fn tap_of(l: u32, f: f32, k: u32) -> f32 {
    let t = rp.cfg.y;
    return (1.0 - f) * bank[l * t + k] + f * bank[(l + 1u) * t + k];
}

// tmp = H(plane `rp.cfg.w`): the horizontal gather with the pixel's kernel.
fn rl_blur_rows(gid: vec3<u32>) {
    let w = rp.dims.x;
    if (gid.x >= w || gid.y >= rp.dims.y) {
        return;
    }
    let pp = rp.dst.w;
    let o = gid.y * w + gid.x;
    let lv = wk[P_LV * pp + o];
    let l = min(u32(lv), 31u);
    let f = lv - f32(l);
    let r = i32(rp.cfg.x);
    let src = rp.cfg.w * pp + gid.y * w;
    var acc = 0.0;
    for (var k = 0u; k < rp.cfg.y; k++) {
        let sx = u32(mirror_idx(i32(gid.x) + i32(k) - r, i32(w)));
        acc += tap_of(l, f, k) * wk[src + sx];
    }
    wk[P_TMP * pp + o] = acc;
}

// V(tmp) at one pixel, with the pixel's kernel.
fn column_blur(x: u32, y: u32, l: u32, f: f32) -> f32 {
    let w = rp.dims.x;
    let pp = rp.dst.w;
    let r = i32(rp.cfg.x);
    var acc = 0.0;
    for (var k = 0u; k < rp.cfg.y; k++) {
        let sy = u32(mirror_idx(i32(y) + i32(k) - r, i32(rp.dims.y)));
        acc += tap_of(l, f, k) * wk[P_TMP * pp + sy * w + x];
    }
    return acc;
}

// e = V(tmp), then q = 1 + w (d - e) / max(e, floor) with the damping weight w.
fn rl_blur_cols_ratio(gid: vec3<u32>) {
    let w = rp.dims.x;
    if (gid.x >= w || gid.y >= rp.dims.y) {
        return;
    }
    let pp = rp.dst.w;
    let o = gid.y * w + gid.x;
    let lv = wk[P_LV * pp + o];
    let l = min(u32(lv), 31u);
    let f = lv - f32(l);
    let e = column_blur(gid.x, gid.y, l, f);
    let ee = max(e, rp.luma.w);
    let resid = wk[P_D * pp + o] - e;
    let sn = sgm[(rp.sig.y + gid.y) * rp.dst.z + rp.sig.x + gid.x];
    let lam = rp.k.w * sn;
    var wgt = 1.0;
    if (lam > 0.0) {
        wgt = resid * resid / (resid * resid + lam * lam);
    }
    wk[P_Q * pp + o] = 1.0 + wgt * resid / ee;
}

// c = V(tmp), then u = max(u c, 0).
fn rl_blur_cols_update(gid: vec3<u32>) {
    let w = rp.dims.x;
    if (gid.x >= w || gid.y >= rp.dims.y) {
        return;
    }
    let pp = rp.dst.w;
    let o = gid.y * w + gid.x;
    let lv = wk[P_LV * pp + o];
    let l = min(u32(lv), 31u);
    let f = lv - f32(l);
    let c = column_blur(gid.x, gid.y, l, f);
    wk[P_U * pp + o] = max(wk[P_U * pp + o] * c, 0.0);
}

// The core of the result: every channel plus (u - d).
fn rl_output(gid: vec3<u32>) {
    if (gid.x >= rp.core.z || gid.y >= rp.core.w) {
        return;
    }
    let px = rp.core.x + gid.x;
    let py = rp.core.y + gid.y;
    let i = py * rp.dims.z + px;
    let o = gid.y * rp.dst.x + gid.x;
    if (rp.cfg.z == 0u) {
        // No iteration: the stage is the identity, bit for bit (adding 0.0 would turn -0.0 into +0.0).
        dst[o] = img[i];
        dst[rp.dst.y + o] = img[rp.dims.w + i];
        dst[2u * rp.dst.y + o] = img[2u * rp.dims.w + i];
        return;
    }
    let pp = rp.dst.w;
    let k = py * rp.dims.x + px;
    let delta = wk[P_U * pp + k] - wk[P_D * pp + k];
    dst[o] = img[i] + delta;
    dst[rp.dst.y + o] = img[rp.dims.w + i] + delta;
    dst[2u * rp.dst.y + o] = img[2u * rp.dims.w + i] + delta;
}
