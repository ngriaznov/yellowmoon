// R6's false-colour suppression after the single-frame denoise: the GPU port of
// `ym_gpu::cpu::false_color::suppress` (`docs/algorithms/05-single-frame-denoise.md`, *Integration*). R7's colour
// guided filter (`guided.wgsl`, which `kernels::false_color` runs first, the CPU's bits) is used where the luma
// alternates, and the rest of the image is left as the method made it. Four pipelines of this file
// (`kernels/false_color.rs` appends one `main` to each):
//
//   false_color_luma    one thread per buffer pixel: Y of the working encoding, KR W(r) + KG W(g) + KB W(b)
//                       (`bm_encode` of working_encoding.wgsl), into scratch plane 0;
//   false_color_rows    one thread per buffer pixel: the alternation s = max(0, -(Y - Y_left)(Y_right - Y)) plus
//                       the same along y (neighbours clamped to the buffer), and its 5-sample row sum, left to
//                       right from 0, into scratch plane 1;
//   false_color_energy  one thread per buffer pixel: the 5-sample column sum of the row sums, top to bottom from
//                       0, and E = sqrt(sum / 25), into scratch plane 0 (Y is no longer read);
//   false_color_blend   one thread per core pixel: M = the maximum of E over 5 x 5 (clamped; a maximum is exact,
//                       so it is the CPU's rows-then-columns value), w = smoothstep((M - T0) / (T1 - T0)), and
//                       out = x + w (G - x) per channel, x itself where w = 0; or w alone (mode 1, for the tests).
//
// Every index is relative to the buffer and clamped at its edge, as on the CPU, so a tile with the stage's halo
// gives the whole frame's bits (the clamp then touches only pixels outside its core, or the frame's own edge).
//
// Fast-math (Metal, docs/17-gpu.md K8): every product, sum and difference is behind `bm_op` (an OR with the
// uniform's zero word), the quotients are correctly rounded (`bm_div`), the root is correctly rounded in integer
// arithmetic (`fc_sqrt`, the method of `sharpen_sqrt` in output_sharpen.wgsl), and the encoding's power is our own
// (`bm_pow`). Maxima and clamps are exact. Nothing here is a decision with a jump: w is continuous in M and the
// blend is continuous in w, so f32 against the CPU's f64 differs by rounding alone.
//
// Bindings: 0 uniform, 1 the input planes, 2 the scratch (2 dense planes of the buffer), 3 the guided filter's
// output (the core), 4 the output (the core).

struct FcParams {
    dims: vec4<u32>,   // buffer width, height, scratch plane length (width * height), the zero word
    io: vec4<u32>,     // input row stride, input plane length, output row stride, output plane length
    core: vec4<u32>,   // core x, y, width, height in the buffer
    gd: vec4<u32>,     // guided output row stride, its plane length, mode (0 blend, 1 weights), 0
};

@group(0) @binding(0) var<uniform> fp: FcParams;
@group(0) @binding(1) var<storage, read> fc_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> fc_scr: array<f32>;
@group(0) @binding(3) var<storage, read> fc_gd: array<f32>;
@group(0) @binding(4) var<storage, read_write> fc_out: array<f32>;

// Always 0 at run time, which the optimiser cannot know: OR-ing a value with it hides the value from
// reassociation, contraction and algebraic rewrites. An OR, never an XOR (docs/17-gpu.md K8). The helpers of
// working_encoding.wgsl call it too.
fn bm_op(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | fp.dims.w);
}

// A column and a row clamped to the buffer.
fn fc_cx(x: i32) -> u32 {
    return u32(clamp(x, 0, i32(fp.dims.x) - 1));
}

fn fc_cy(y: i32) -> u32 {
    return u32(clamp(y, 0, i32(fp.dims.y) - 1));
}

// The correctly rounded square root of `x`, in integer arithmetic only (Metal's fast-math `sqrt` is 1 to 2 ulp
// off): `sharpen_sqrt` of output_sharpen.wgsl. For a positive normal x the mantissa is doubled when the
// exponent is odd, the integer root of N = mp 2^26 is taken digit by digit, and its next bit and remainder
// decide the rounding to nearest, ties to even. Zero returns itself; subnormals, infinities, NaN and negatives
// take the device's `sqrt`.
fn fc_sqrt(x: f32) -> f32 {
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
    return bitcast<f32>(((((e + 127u) >> 1u) - 1u) << 23u) + q);
}

// Y at buffer pixel (x, y) (scratch plane 0, written by `fc_luma`).
fn fc_y(x: u32, y: u32) -> f32 {
    return fc_scr[y * fp.dims.x + x];
}

// The alternation at buffer pixel (x, y): max(0, -(c - l)(r - c)) + max(0, -(c - u)(d - c)).
fn fc_alternation(x: u32, y: u32) -> f32 {
    let c = fc_y(x, y);
    let l = fc_y(fc_cx(i32(x) - 1), y);
    let r = fc_y(fc_cx(i32(x) + 1), y);
    let u = fc_y(x, fc_cy(i32(y) - 1));
    let d = fc_y(x, fc_cy(i32(y) + 1));
    let sx = max(-bm_op(bm_op(c - l) * bm_op(r - c)), 0.0);
    let sy = max(-bm_op(bm_op(c - u) * bm_op(d - c)), 0.0);
    return bm_op(sx + sy);
}

fn fc_luma(gid: vec3<u32>) {
    if (gid.x >= fp.dims.x || gid.y >= fp.dims.y) {
        return;
    }
    let s = gid.y * fp.io.x + gid.x;
    let er = bm_encode(fc_src[s]);
    let eg = bm_encode(fc_src[fp.io.y + s]);
    let eb = bm_encode(fc_src[2u * fp.io.y + s]);
    fc_scr[gid.y * fp.dims.x + gid.x] =
        bm_op(bm_op(bm_op(BM_KR * er) + bm_op(BM_KG * eg)) + bm_op(BM_KB * eb));
}

fn fc_rows(gid: vec3<u32>) {
    if (gid.x >= fp.dims.x || gid.y >= fp.dims.y) {
        return;
    }
    var acc = 0.0;
    for (var d = -FC_BOX_R; d <= FC_BOX_R; d++) {
        acc = bm_op(acc + fc_alternation(fc_cx(i32(gid.x) + d), gid.y));
    }
    fc_scr[fp.dims.z + gid.y * fp.dims.x + gid.x] = acc;
}

fn fc_energy(gid: vec3<u32>) {
    if (gid.x >= fp.dims.x || gid.y >= fp.dims.y) {
        return;
    }
    var acc = 0.0;
    for (var d = -FC_BOX_R; d <= FC_BOX_R; d++) {
        acc = bm_op(acc + fc_scr[fp.dims.z + fc_cy(i32(gid.y) + d) * fp.dims.x + gid.x]);
    }
    fc_scr[gid.y * fp.dims.x + gid.x] = fc_sqrt(bm_div(acc, FC_BOX_AREA));
}

// The weight of the guided filter at buffer pixel (x, y), from the energy plane.
fn fc_weight(x: i32, y: i32) -> f32 {
    var m = 0.0;
    for (var dy = -FC_DILATE_R; dy <= FC_DILATE_R; dy++) {
        let row = fc_cy(y + dy) * fp.dims.x;
        for (var dx = -FC_DILATE_R; dx <= FC_DILATE_R; dx++) {
            m = max(m, fc_scr[row + fc_cx(x + dx)]);
        }
    }
    let t = clamp(bm_div(bm_op(m - FC_T0), FC_SPAN), 0.0, 1.0);
    return bm_op(bm_op(t * t) * bm_op(3.0 - bm_op(2.0 * t)));
}

fn fc_blend(gid: vec3<u32>) {
    if (gid.x >= fp.core.z || gid.y >= fp.core.w) {
        return;
    }
    let x = fp.core.x + gid.x;
    let y = fp.core.y + gid.y;
    let w = fc_weight(i32(x), i32(y));
    let o = gid.y * fp.io.z + gid.x;
    if (fp.gd.z == 1u) {
        fc_out[o] = w;
        return;
    }
    let s = y * fp.io.x + x;
    let g = gid.y * fp.gd.x + gid.x;
    for (var c = 0u; c < 3u; c++) {
        let xv = fc_src[c * fp.io.y + s];
        var v = xv;
        if (w != 0.0) {
            v = bm_op(xv + bm_op(w * bm_op(fc_gd[c * fp.gd.y + g] - xv)));
        }
        fc_out[c * fp.io.w + o] = v;
    }
}
