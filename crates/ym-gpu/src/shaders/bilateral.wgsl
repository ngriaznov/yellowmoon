// R6 bilateral denoise (G-06): the GPU port of `ym_gpu::cpu::denoise::bilateral_whole`
// (`docs/05-raw-processing.md` section R6, "Denoise"). A 5 x 5 (radius `rad`, up to 4) bilateral filter
// in the opponent space (Y, R - Y, B - Y) with the luma weights of `cpu::LUMA`:
//
//   w(q) = spatial(q) * exp(-(dY^2 + dCr^2 + dCb^2) / (2 h^2)),  h = strength * sqrt(6) * sigma(centre)
//
// The window is read with the mirror of the input rectangle's edge (without repeating the edge sample,
// `cpu::mirror`, the same as the CPU's padded planes), a tap whose exponent is above 9 is skipped, a
// pixel whose `h` is not above 1e-12 keeps its exact input values, and the output is rebuilt from the
// filtered opponent values (red and blue from the chroma, green from the luma equation). The order of
// the sums is the CPU's: rows of the window top to bottom, each row left to right.
//
// One thread per pixel of the core. The input is the compute rectangle with its context; the output is
// the core only. The opponent values are formed per tap from the stored planes (the same expressions as
// the CPU's `to_opponent`), so a pixel depends on its window and on nothing else: a tile gives the bits
// of the same pixels in a whole-frame run (D5). The sigma plane is the tile's plane
// (`sigma_expand.wgsl`); this kernel multiplies it by the network scale `(L / L_auto)^1.5`, the one
// multiplication the CPU's `sigma_tile` makes.
//
// Fast-math (Metal, K8): nothing here depends on exact rounding. The one decision, `u > 9`, drops a
// tap of weight 1.2e-4 at most; a flip at the boundary moves the output by less than that times the
// local noise, and is counted by the tests as a tie flip, not an error.
//
// Bindings: 0 uniform, 1 spatial weights (side * side), 2 the input planes, 3 the sigma plane,
// 4 the output planes.

struct BilParams {
    dims: vec4<u32>,   // input width, height, row stride, plane length
    core: vec4<u32>,   // core x, y, width, height in the input
    dst: vec4<u32>,    // output row stride, output plane length, sigma row stride, radius
    sig: vec4<u32>,    // the input's origin in the sigma plane (x, y)
    k: vec4<f32>,      // strength * sqrt(6), the sigma scale
    luma: vec4<f32>,   // luma weights (R, G, B)
};

@group(0) @binding(0) var<uniform> bp: BilParams;
@group(0) @binding(1) var<storage, read> spatial: array<f32>;
@group(0) @binding(2) var<storage, read> img: array<f32>;
@group(0) @binding(3) var<storage, read> sgm: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

// (Y, R - Y, B - Y) of the input pixel at (x, y).
fn opponent_at(x: u32, y: u32) -> vec3<f32> {
    let i = y * bp.dims.z + x;
    let r = img[i];
    let g = img[bp.dims.w + i];
    let b = img[2u * bp.dims.w + i];
    let yl = bp.luma.x * r + bp.luma.y * g + bp.luma.z * b;
    return vec3<f32>(yl, r - yl, b - yl);
}

fn bilateral_filter(gid: vec3<u32>) {
    if (gid.x >= bp.core.z || gid.y >= bp.core.w) {
        return;
    }
    let px = bp.core.x + gid.x;
    let py = bp.core.y + gid.y;
    let o = gid.y * bp.dst.x + gid.x;
    let ii = py * bp.dims.z + px;
    let s = sgm[(bp.sig.y + py) * bp.dst.z + bp.sig.x + px] * bp.k.y;
    let hh = bp.k.x * s;
    if (hh <= 1e-12) {
        // Untouched pixels keep their exact input values.
        dst[o] = img[ii];
        dst[bp.dst.y + o] = img[bp.dims.w + ii];
        dst[2u * bp.dst.y + o] = img[2u * bp.dims.w + ii];
        return;
    }
    let inv2h2 = 0.5 / (hh * hh);
    let c = opponent_at(px, py);
    let w = i32(bp.dims.x);
    let h = i32(bp.dims.y);
    let rad = i32(bp.dst.w);
    let side = 2 * rad + 1;
    var sy = 0.0;
    var scr = 0.0;
    var scb = 0.0;
    var sw = 0.0;
    for (var dy = 0; dy < side; dy++) {
        let yy = u32(mirror_idx(i32(py) + dy - rad, h));
        for (var dx = 0; dx < side; dx++) {
            let xx = u32(mirror_idx(i32(px) + dx - rad, w));
            let q = opponent_at(xx, yy);
            let d = q - c;
            let u = (d.x * d.x + d.y * d.y + d.z * d.z) * inv2h2;
            if (u > 9.0) {
                continue;
            }
            let wt = spatial[u32(dy * side + dx)] * exp(-u);
            sy += wt * q.x;
            scr += wt * q.y;
            scb += wt * q.z;
            sw += wt;
        }
    }
    // The centre pixel has weight 1, so sw >= 1.
    let inv = 1.0 / sw;
    let fy = sy * inv;
    let red = scr * inv + fy;
    let blue = scb * inv + fy;
    dst[o] = red;
    dst[bp.dst.y + o] = (fy - bp.luma.x * red - bp.luma.z * blue) / bp.luma.y;
    dst[2u * bp.dst.y + o] = blue;
}
