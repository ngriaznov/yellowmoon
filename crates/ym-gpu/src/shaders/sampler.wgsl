// The geometry sampler (docs/17-gpu.md section 11): camera RGB through a grid that maps output
// pixels to buffer positions (orientation, quarter turns, straighten, crop), the GPU twin of
// `ym_gpu::cpu::sampler::sample_to_camera`. Three kernels: a pixel copy, Catmull-Rom, and a box of
// n x n bilinear sub-samples. The output is a four-plane image: R, G, B and a plane that is 1.0
// where the pixel is outside the photo (the letterbox) and 0.0 inside; outside pixels are 0 in
// R, G, B.
//
// Positions (D4 and D5 of docs/17-gpu.md). The host collapses the position chain into one affine
// map in f64 and splits what each column and each row of the *picture* contributes into an integer
// part and a fraction in [0, 1): `cols` for the x part (J * (X, 0)), `rows` for the y part
// (J * (0, Y) + the offset, less the integer origin of the source image). A source position is the
// sum of one entry of each: the integer parts are added as integers and the two fractions as f32, so
// there is no f32 value of order 1e4 anywhere and the precision is that of a fraction (1e-7 px). The
// entries depend only on the absolute column and row, never on the window being rendered, so a tile
// and the whole frame sample every pixel with the same bits.
//
// The inside test is not evaluated here either. The host gives, for every output row and sub-sample,
// the exact interval of columns the CPU's f64 predicate accepts (`inside`), so the mask equals the
// CPU's bit for bit and no rounding can move a pixel across the edge of the photo.
//
// The lens's bend (L4, docs/21 section 4). A grid may carry the anamorphic bend, the lens's own
// distortion and the zoom, which break the separability above: the photo position a canvas position
// reads is F^-1(c / z), a non-affine map. The tables still give the position the affine chain gives
// without the bend, exactly, as an integer and a fraction. `warp_pos` turns that position into the
// canvas's offset from its centre in half-diagonal units (the integer part exact, so there is no
// f32 of order 1e4: D4), finds the photo's offset with the CPU's Newton steps, and adds only the
// *displacement* the bend makes (a few percent of the half-diagonal at most) to the exact position.
// The Newton iteration is written for e = x - t, the offset from the start, so that F(x) - t is e
// plus the small terms F(x) - x, and nothing of order 1 is ever subtracted from something of order 1.
// An error in the input offset (about 6e-8 of the half-diagonal) therefore reaches the position
// scaled by the displacement's slope (0.1 to 0.2), not whole. The result is clamped to the buffer, as
// the CPU clamps the sensor fraction. Everything is a function of the absolute column and row
// through the tables, so a tile and the whole frame agree bit for bit (D5).
//
// The map is the CPU's (`SamplerWarp`, `ym_lens::LensWarp`): the lens's radial distortion, then the
// bend x' = x1 (1 + Gx), y' = y1 (1 + Gy) with Gx = a1 x^2 + a2 y^2 + a3 x^4 + a4 x^2 y^2 + a5 y^4 (b
// for Gy), written with the same grouping and in the same order, and its Jacobian as the product of
// the two stages'. No decision depends on a rounding (K8 of docs/17-gpu.md): the one branch is the
// guard on a determinant of exactly 0, the CPU's, which the data checks keep far from the map's range
// (the determinant is at least 0.5 over the photo and a margin).
//
// The arithmetic of the samples follows the CPU port operation by operation: the Catmull-Rom
// weights, `acc += wy_k (wx_0 r_0 + wx_1 r_1 + wx_2 r_2 + wx_3 r_3)` row by row, the bilinear blend
// and the division of the box by the number of sub-samples that fell inside.

struct SamplerU {
    dims: vec4<u32>,    // x: window width, y: window height, z: output stride (even), w: output plane length
    src: vec4<u32>,     // x: source width, y: source height, z: source stride, w: source plane length
    mode: vec4<u32>,    // x: kernel (0 copy, 1 Catmull-Rom, 2 box), y: sub-samples per axis (1 unless box), z: 1 with a warp
    wk: vec4<u32>,      // x: Newton steps, y: 1 when the anamorphic axis is the canvas's y
    wx: vec4<f32>,      // a1 .. a4: the terms of x' for x^2, y^2, x^4, x^2 y^2 (a5, for y^4, is wc.x)
    wy: vec4<f32>,      // b1 .. b4: the same for y' (b5 is wc.y)
    wc: vec4<f32>,      // a5, b5, d1, d2
    wb: vec4<f32>,      // x: zoom, y: zoom - 1, z: half-diagonal (canvas px), w: 1 / half-diagonal
    sl: vec4<f32>,      // canvas displacement -> buffer displacement, row-major (s00, s01, s10, s11)
    si: vec4<f32>,      // its inverse, row-major
    ci: vec4<i32>,      // x, y: integer part of the canvas centre's buffer position, relative to the source window
    cf: vec4<f32>,      // x, y: its fraction
    lim: vec4<i32>,     // x0, y0, x1, y1: the buffer, relative to the source window (the clamp)
};

struct PosEntry {
    i: vec2<i32>,       // integer parts of the contribution to the source x and y
    f: vec2<f32>,       // fractions, in [0, 1)
};

struct Pos {
    i: vec2<i32>,
    f: vec2<f32>,
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

// The position (pi + pf) the chain gives without the bend, moved by the bend; see the notes above.
fn warp_pos(pi: vec2<i32>, pf: vec2<f32>) -> Pos {
    // The canvas centre to this position, in buffer pixels: the integers apart from the fractions.
    let di = vec2<f32>(pi - su.ci.xy);
    let df = pf - su.cf.xy;
    // In canvas pixels (the inverse of the dihedral map), then in half-diagonal units.
    let ni = vec2<f32>(su.si.x * di.x + su.si.y * di.y, su.si.z * di.x + su.si.w * di.y);
    let nf = vec2<f32>(su.si.x * df.x + su.si.y * df.y, su.si.z * df.x + su.si.w * df.y);
    var n = ni * su.wb.w + nf * su.wb.w;
    if (su.wk.y != 0u) {
        n = n.yx;
    }
    let t = n / su.wb.x;
    let ax = su.wx;
    let ay = su.wy;
    let ax5 = su.wc.x;
    let ay5 = su.wc.y;
    let d1 = su.wc.z;
    let d2 = su.wc.w;
    var e = vec2<f32>(0.0);
    for (var it = 0u; it < su.wk.x; it = it + 1u) {
        let x = t.x + e.x;
        let y = t.y + e.y;
        let r2 = x * x + y * y;
        let gm = d1 * r2 + d2 * r2 * r2;
        let g = 1.0 + gm;
        let gp = d1 + 2.0 * d2 * r2;
        let x1 = x * g;
        let y1 = y * g;
        let xx = x1 * x1;
        let yy = y1 * y1;
        // The bend's G of each axis, as the CPU writes it: xx (c0 + c2 xx + c3 yy) + yy (c1 + c4 yy).
        let gx = xx * (ax.x + ax.z * xx + ax.w * yy) + yy * (ax.y + ax5 * yy);
        let gy = xx * (ay.x + ay.z * xx + ay.w * yy) + yy * (ay.y + ay5 * yy);
        // F(x) - x, and the residual F(x) - t = e + (F(x) - x).
        let rx = e.x + (x * gm + x1 * gx);
        let ry = e.y + (y * gm + y1 * gy);
        let a = g + 2.0 * x * x * gp;
        let b = 2.0 * x * y * gp;
        let d = g + 2.0 * y * y * gp;
        // The second stage's Jacobian at (x1, y1).
        let k00 = 1.0 + gx + xx * (2.0 * (ax.x + 2.0 * ax.z * xx + ax.w * yy));
        let k01 = x1 * y1 * (2.0 * (ax.y + ax.w * xx + 2.0 * ax5 * yy));
        let k10 = y1 * x1 * (2.0 * (ay.x + 2.0 * ay.z * xx + ay.w * yy));
        let k11 = 1.0 + gy + yy * (2.0 * (ay.y + ay.w * xx + 2.0 * ay5 * yy));
        let j00 = k00 * a + k01 * b;
        let j01 = k00 * b + k01 * d;
        let j10 = k10 * a + k11 * b;
        let j11 = k10 * b + k11 * d;
        let det = j00 * j11 - j01 * j10;
        if (det != 0.0) {
            e = e - vec2<f32>((j11 * rx - j01 * ry) / det, (-j10 * rx + j00 * ry) / det);
        }
    }
    // The photo's offset is t + e, the canvas's n = z t: the displacement is e - (z - 1) t.
    var disp = e - su.wb.y * t;
    if (su.wk.y != 0u) {
        disp = disp.yx;
    }
    let dc = disp * su.wb.z;
    let db = vec2<f32>(su.sl.x * dc.x + su.sl.y * dc.y, su.sl.z * dc.x + su.sl.w * dc.y);
    let moved = pf + db;
    let fl = floor(moved);
    var qi = pi + vec2<i32>(fl);
    var qf = moved - fl;
    // The buffer's edge: the CPU clamps the position to it.
    if (qi.x < su.lim.x) {
        qi.x = su.lim.x;
        qf.x = 0.0;
    } else if (qi.x >= su.lim.z) {
        qi.x = su.lim.z;
        qf.x = 0.0;
    }
    if (qi.y < su.lim.y) {
        qi.y = su.lim.y;
        qf.y = 0.0;
    } else if (qi.y >= su.lim.w) {
        qi.y = su.lim.w;
        qf.y = 0.0;
    }
    return Pos(qi, qf);
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
            var pi = ce.i + re.i;
            var pf = ce.f + re.f;
            if (su.mode.z != 0u) {
                let moved = warp_pos(pi, pf);
                pi = moved.i;
                pf = moved.f;
            }
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
                    var pi = ce.i + re.i;
                    var pf = ce.f + re.f;
                    if (su.mode.z != 0u) {
                        let moved = warp_pos(pi, pf);
                        pi = moved.i;
                        pf = moved.f;
                    }
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
