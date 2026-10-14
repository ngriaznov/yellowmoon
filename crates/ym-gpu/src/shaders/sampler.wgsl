// The geometry sampler (docs/gpu-plan.md G-04): camera RGB through a grid that maps output pixels to
// buffer positions (orientation, quarter turns, straighten, crop), the GPU twin of
// `ym_gpu::cpu::sampler::sample_to_camera`. Three kernels: a pixel copy, Catmull-Rom, and a box of
// n x n bilinear sub-samples. The output is a four-plane image: R, G, B and a plane that is 1.0 where
// the pixel is outside the photo (the letterbox) and 0.0 inside; outside pixels are 0 in R, G, B.
//
// Positions (D4 and D5 of the plan). The host collapses the position chain into one affine map in f64
// and splits what each column and each row of the *picture* contributes into an integer part and a
// fraction in [0, 1): `cols` for the x part (J * (X, 0)), `rows` for the y part (J * (0, Y) + the
// offset, less the integer origin of the source image). A source position is the sum of one entry of
// each: the integer parts are added as integers and the two fractions as f32, so there is no f32
// value of order 1e4 anywhere and the precision is that of a fraction (1e-7 px). The entries depend
// only on the absolute column and row, never on the window being rendered, so a tile and the whole
// frame sample every pixel with the same bits.
//
// The inside test is not evaluated here either. The host gives, for every output row and sub-sample,
// the exact interval of columns the CPU's f64 predicate accepts (`inside`), so the mask equals the
// CPU's bit for bit and no rounding can move a pixel across the edge of the photo.
//
// The arithmetic of the samples follows the CPU port operation by operation: the Catmull-Rom
// weights, `acc += wy_k (wx_0 r_0 + wx_1 r_1 + wx_2 r_2 + wx_3 r_3)` row by row, the bilinear blend
// and the division of the box by the number of sub-samples that fell inside.

struct SamplerU {
    dims: vec4<u32>,    // x: window width, y: window height, z: output stride (even), w: output plane length
    src: vec4<u32>,     // x: source width, y: source height, z: source stride, w: source plane length
    mode: vec4<u32>,    // x: kernel (0 copy, 1 Catmull-Rom, 2 box), y: sub-samples per axis (1 unless box)
};

struct PosEntry {
    i: vec2<i32>,       // integer parts of the contribution to the source x and y
    f: vec2<f32>,       // fractions, in [0, 1)
};

@group(0) @binding(0) var<uniform> su: SamplerU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<PosEntry>;
@group(0) @binding(3) var<storage, read> rows: array<PosEntry>;
@group(0) @binding(4) var<storage, read> inside: array<vec2<i32>>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

// Sample c at pixel (x, y) of the source, the edge pixel replicated.
fn px(c: u32, x: i32, y: i32) -> f32 {
    let xx = u32(clamp(x, 0, i32(su.src.x) - 1));
    let yy = u32(clamp(y, 0, i32(su.src.y) - 1));
    return src[c * su.src.w + yy * su.src.z + xx];
}

fn cr_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

// Catmull-Rom at the position (pi + pf): u = p - 0.5, i0 = floor(u), t = u - i0 (the CPU's
// `sample_catmull_rom`), written with the integer and the fraction apart.
fn sample_cr(c: u32, pi: vec2<i32>, pf: vec2<f32>) -> f32 {
    let ux = pf.x - 0.5;
    let uy = pf.y - 0.5;
    let fx = floor(ux);
    let fy = floor(uy);
    let i0 = pi.x + i32(fx);
    let j0 = pi.y + i32(fy);
    let wx = cr_weights(ux - fx);
    let wy = cr_weights(uy - fy);
    var acc = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let yy = j0 - 1 + k;
        let r = wx.x * px(c, i0 - 1, yy) + wx.y * px(c, i0, yy)
            + wx.z * px(c, i0 + 1, yy) + wx.w * px(c, i0 + 2, yy);
        acc = acc + wy[k] * r;
    }
    return acc;
}

// The bilinear blend of the CPU's `bilinear`: u = clamp(p - 0.5, 0, size - 1), then the four
// neighbours. A u below 0 or at / above size - 1 clamps to the edge pixel with a zero fraction.
fn sample_bil(c: u32, pi: vec2<i32>, pf: vec2<f32>) -> f32 {
    let ux = pf.x - 0.5;
    let uy = pf.y - 0.5;
    let flx = floor(ux);
    let fly = floor(uy);
    var x0 = pi.x + i32(flx);
    var y0 = pi.y + i32(fly);
    var fx = ux - flx;
    var fy = uy - fly;
    let last_x = i32(su.src.x) - 1;
    let last_y = i32(su.src.y) - 1;
    if (x0 < 0) {
        x0 = 0;
        fx = 0.0;
    } else if (x0 >= last_x) {
        x0 = last_x;
        fx = 0.0;
    }
    if (y0 < 0) {
        y0 = 0;
        fy = 0.0;
    } else if (y0 >= last_y) {
        y0 = last_y;
        fy = 0.0;
    }
    let a = px(c, x0, y0) * (1.0 - fx) + px(c, x0 + 1, y0) * fx;
    let b = px(c, x0, y0 + 1) * (1.0 - fx) + px(c, x0 + 1, y0 + 1) * fx;
    return a * (1.0 - fy) + b * fy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= su.dims.x || y >= su.dims.y) {
        return;
    }
    let kernel = su.mode.x;
    let taps = su.mode.y;
    var rgb = vec3<f32>(0.0);
    var gone = true;
    if (kernel < 2u) {
        let iv = inside[y];
        if (i32(x) >= iv.x && i32(x) <= iv.y) {
            gone = false;
            let ce = cols[x];
            let re = rows[y];
            let pi = ce.i + re.i;
            let pf = ce.f + re.f;
            if (kernel == 0u) {
                let ix = clamp(pi.x + i32(floor(pf.x)), 0, i32(su.src.x) - 1);
                let iy = clamp(pi.y + i32(floor(pf.y)), 0, i32(su.src.y) - 1);
                rgb = vec3<f32>(px(0u, ix, iy), px(1u, ix, iy), px(2u, ix, iy));
            } else {
                rgb = vec3<f32>(sample_cr(0u, pi, pf), sample_cr(1u, pi, pf), sample_cr(2u, pi, pf));
            }
        }
    } else {
        var acc = vec3<f32>(0.0);
        var count = 0u;
        for (var j = 0u; j < taps; j = j + 1u) {
            for (var i = 0u; i < taps; i = i + 1u) {
                let iv = inside[(y * taps + j) * taps + i];
                if (i32(x) >= iv.x && i32(x) <= iv.y) {
                    let ce = cols[x * taps + i];
                    let re = rows[y * taps + j];
                    let pi = ce.i + re.i;
                    let pf = ce.f + re.f;
                    acc = acc + vec3<f32>(sample_bil(0u, pi, pf), sample_bil(1u, pi, pf), sample_bil(2u, pi, pf));
                    count = count + 1u;
                }
            }
        }
        if (count > 0u) {
            rgb = acc / f32(count);
            gone = false;
        }
    }
    let o = y * su.dims.z + x;
    let pl = su.dims.w;
    out[o] = rgb.x;
    out[pl + o] = rgb.y;
    out[2u * pl + o] = rgb.z;
    out[3u * pl + o] = select(0.0, 1.0, gone);
}
