// R9 lens resample (G-07): the GPU port of `cpu::resample::resample_lens` (`docs/05-raw-processing.md`
// section R9). One Catmull-Rom resample per colour plane from the output pixel to the source position
// the lens tables give: distortion `F(r)` for all three planes, the lateral CA factors `K_red(r)` and
// `K_blue(r)` for red and blue (only when the geometry keeps its CA in R9; with the CA on the mosaic
// the factors are 1), edge pixels replicated at the edge of the source image.
//
//   source = centre + d F K,   d = pixel centre - centre,   r = |d| / half diagonal of the source frame
//
// Positions are kept small (D4): the source position is the output position plus the displacement
// `d (F K - 1)`, so the integer pixel index is added to the fraction of the displacement and no float
// ever holds a position of the order of the frame size. The factors are the tables minus one, which
// the host forms exactly (`d_i 2^-14`, `ca_i 2^-21`), interpolated linearly in `u = r (n - 1) - 0.5`
// as `LinearKnots::eval` does; the offset from the centre is the exact integer `X = 2 x + 1 - width`
// (twice the offset), the radius `sqrt(X^2 + Y^2) / hypot(width, height)`.
//
// Bindings: 0 uniform, 1 the source planes, 2 the output planes.

struct LensU {
    zero: vec4<u32>,   // always +0.0 bits
    src: vec4<u32>,    // source width, height, row stride, plane length
    out: vec4<i32>,    // source column and row of the first output pixel's centre (output - source origin), output width, height
    dst: vec4<u32>,    // output row stride, plane length, 0, 0
    fr: vec4<i32>,     // doubled offset of the first pixel's centre from the frame centre (x, y), knots of the distortion, of one CA curve
    flags: vec4<u32>,  // the CA is applied here (1) or on the mosaic (0)
    k: vec4<f32>,      // (n_distortion - 1) / hypot, (n_ca - 1) / hypot
    dist: array<vec4<f32>, 8>,   // F_i - 1
    car: array<vec4<f32>, 8>,    // K_red,i - 1
    cab: array<vec4<f32>, 8>,    // K_blue,i - 1
};

@group(0) @binding(0) var<uniform> pu: LensU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn knot(tab: u32, i: u32) -> f32 {
    if (tab == 0u) {
        return pu.dist[i >> 2u][i & 3u];
    }
    if (tab == 1u) {
        return pu.car[i >> 2u][i & 3u];
    }
    return pu.cab[i >> 2u][i & 3u];
}

// The table `tab` (n knots) minus one, linear between knots and held beyond them, at the root of the
// doubled squared radius (`root = sqrt(X^2 + Y^2)`, `scale = (n - 1) / hypot`).
fn factor_minus_one(tab: u32, n: u32, scale: f32, root: f32) -> f32 {
    if (n == 0u) {
        return 0.0;
    }
    let u = root * scale - 0.5;
    if (!(u > 0.0)) {
        return knot(tab, 0u);
    }
    if (u >= f32(n - 1u)) {
        return knot(tab, n - 1u);
    }
    let i = u32(u);
    let t = u - f32(i);
    let a = knot(tab, i);
    return a + opaque(opaque(knot(tab, i + 1u) - a) * t);
}

fn cr_weights(t: f32) -> vec4<f32> {
    let t2 = opaque(t * t);
    let t3 = opaque(t2 * t);
    let w0 = opaque(opaque(opaque(-0.5 * t3) + t2) - opaque(0.5 * t));
    let w1 = opaque(opaque(opaque(1.5 * t3) - opaque(2.5 * t2)) + 1.0);
    let w2 = opaque(opaque(opaque(-1.5 * t3) + opaque(2.0 * t2)) + opaque(0.5 * t));
    let w3 = opaque(opaque(0.5 * t3) - opaque(0.5 * t2));
    return vec4<f32>(w0, w1, w2, w3);
}

fn tap(plane: u32, x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(pu.src.x) - 1));
    let cy = u32(clamp(y, 0, i32(pu.src.y) - 1));
    return src[plane * pu.src.w + cy * pu.src.z + cx];
}

// The Catmull-Rom sample of plane `plane` at the pixel (ix, iy) of the source plus the displacement
// (ex, ey) (the CPU's `sample_catmull_rom` at position `ix + 0.5 + e`).
fn sample(plane: u32, ix: i32, iy: i32, ex: f32, ey: f32) -> f32 {
    let fx = floor(ex);
    let fy = floor(ey);
    let wx = cr_weights(ex - fx);
    let wy = cr_weights(ey - fy);
    let i0 = ix + i32(fx);
    let j0 = iy + i32(fy);
    var acc = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let b = j0 - 1 + k;
        let p0 = opaque(wx.x * tap(plane, i0 - 1, b));
        let p1 = opaque(wx.y * tap(plane, i0, b));
        let p2 = opaque(wx.z * tap(plane, i0 + 1, b));
        let p3 = opaque(wx.w * tap(plane, i0 + 2, b));
        let row = opaque(opaque(opaque(p0 + p1) + p2) + p3);
        acc = opaque(acc + opaque(wy[k] * row));
    }
    return acc;
}

fn lens_main(gid: vec3<u32>) {
    if (i32(gid.x) >= pu.out.z || i32(gid.y) >= pu.out.w) {
        return;
    }
    let x2 = pu.fr.x + 2 * i32(gid.x);
    let y2 = pu.fr.y + 2 * i32(gid.y);
    let s2 = u32(abs(x2)) * u32(abs(x2)) + u32(abs(y2)) * u32(abs(y2));
    let root = sqrt(f32(s2));
    let nd = u32(pu.fr.z);
    let nc = u32(pu.fr.w);
    let f1 = factor_minus_one(0u, nd, pu.k.x, root);
    var kr = 0.0;
    var kb = 0.0;
    if (pu.flags.x != 0u) {
        kr = factor_minus_one(1u, nc, pu.k.y, root);
        kb = factor_minus_one(2u, nc, pu.k.y, root);
    }
    // F K - 1 = (F - 1) + (K - 1) + (F - 1)(K - 1).
    let fr1 = opaque(opaque(f1 + kr) + opaque(f1 * kr));
    let fb1 = opaque(opaque(f1 + kb) + opaque(f1 * kb));
    let dx = 0.5 * f32(x2);
    let dy = 0.5 * f32(y2);
    let ix = pu.out.x + i32(gid.x);
    let iy = pu.out.y + i32(gid.y);
    let o = gid.y * pu.dst.x + gid.x;
    dst[o] = sample(0u, ix, iy, opaque(dx * fr1), opaque(dy * fr1));
    dst[pu.dst.y + o] = sample(1u, ix, iy, opaque(dx * f1), opaque(dy * f1));
    dst[2u * pu.dst.y + o] = sample(2u, ix, iy, opaque(dx * fb1), opaque(dy * fb1));
}
