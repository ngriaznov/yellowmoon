// The magnification of a view above 100 % (docs/17-gpu.md section 16), the GPU twin of
// `ym_gpu::cpu::magnify::magnify`: each device pixel of the view is a Catmull-Rom sample of the
// sharpened 100 % picture under it (`src`, three planes of display-linear RGB and the letterbox mask),
// or the paper where its centre is outside the final image or on a pixel outside the photo; then, for
// the viewport, C13 (the output transfer, the blue-noise dither at the pixel's place in the view, the
// quantiser) and the paper colour, as one RGBA8 word per pixel. The other output writes the
// magnified display-linear picture as four f32 planes (R, G, B and 1.0 on the paper), so that this
// pass followed by the encode kernel is the fused pass bit for bit (tests/gpu_magnify.rs).
//
// No position is computed here. The host gives one tap per column and one per row (`cols`, `rows`,
// `cpu::magnify::AxisTap`): the four source columns (rows) the filter reads, the edge replicated, their
// four f32 weights, the column (row) of the mask sample and whether the centre is inside the final
// image along that axis, all made in f64 by the CPU reference's own code. So no f32 position of order
// 1e4 exists (D4) and the only arithmetic is the sum, which is the CPU's:
//
//   acc = 0; acc += wy_k * (wx_0 r_0 + wx_1 r_1 + wx_2 r_2 + wx_3 r_3), k = 0..3, left to right, f32.
//
// Fast-math (K8, docs/17-gpu.md section 13). Metal may fuse a product into the sum that takes it, or
// re-associate the sum; either changes the rounding. Every product and every partial sum is therefore
// hidden behind the OR barrier (`opaque`, with `mu.src.w`, a uniform word that is 0 at run time and
// unknown to the compiler): one operation per barrier, nothing left to fuse or reorder, so the sum has
// the CPU's bits under fast-math and under IEEE alike. There is no division, no square root and no
// comparison of a computed value; the paper test reads the host's flags and the mask's stored zeros.

struct MagnifyU {
    view: vec4<u32>,   // x: view width, y: view height, z: output stride (words or samples a row), w: output kind (0 planes, 1 RGBA8)
    src: vec4<u32>,    // x: source stride, y: source plane length, z: 1 when the source has a mask plane, w: zero (the barrier)
    c13: vec4<u32>,    // x: transfer (0 sRGB, 1 BT.1886), y: dither, z: paper colour (RGBA8 word), w: output plane length (kind 0)
    org: vec4<u32>,    // x, y: the place of the view's pixel (0, 0) in the picture the dither texels are counted in
};

struct Tap {
    idx: vec4<u32>,    // the four source columns (rows), the edge replicated
    w: vec4<f32>,      // their Catmull-Rom weights
    at: u32,           // the column (row) of the mask sample
    inside: u32,       // 1 when the centre is inside the final image along this axis
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> mu: MagnifyU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<Tap>;
@group(0) @binding(3) var<storage, read> rows: array<Tap>;
@group(0) @binding(4) var<storage, read> dither_tab: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<u32>;

// The fast-math barrier: an OR with a uniform zero (never an XOR, docs/17-gpu.md K8).
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | mu.src.w);
}

fn src_at(c: u32, x: u32, y: u32) -> f32 {
    return src[c * mu.src.y + y * mu.src.x + x];
}

// One channel at the taps (cx, ry), every product and partial sum behind a barrier.
fn magnified(c: u32, cx: Tap, ry: Tap) -> f32 {
    var acc = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let y = ry.idx[k];
        let p0 = opaque(cx.w.x * src_at(c, cx.idx.x, y));
        let p1 = opaque(cx.w.y * src_at(c, cx.idx.y, y));
        let p2 = opaque(cx.w.z * src_at(c, cx.idx.z, y));
        let p3 = opaque(cx.w.w * src_at(c, cx.idx.w, y));
        let r = opaque(opaque(opaque(p0 + p1) + p2) + p3);
        acc = opaque(acc + opaque(ry.w[k] * r));
    }
    return acc;
}

fn is_paper(cx: Tap, ry: Tap) -> bool {
    if (cx.inside == 0u || ry.inside == 0u) {
        return true;
    }
    return mu.src.z != 0u && src_at(3u, cx.at, ry.at) != 0.0;
}

fn threshold_at(x: u32, y: u32) -> f32 {
    if (mu.c13.y == 0u) {
        return 0.0;
    }
    return dither_tab[((mu.org.y + y) % 64u) * 64u + ((mu.org.x + x) % 64u)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= mu.view.x || y >= mu.view.y) {
        return;
    }
    let cx = cols[x];
    let ry = rows[y];
    let paper = is_paper(cx, ry);
    let o = y * mu.view.z + x;
    var lin = vec3<f32>(0.0);
    if (!paper) {
        lin = vec3<f32>(magnified(0u, cx, ry), magnified(1u, cx, ry), magnified(2u, cx, ry));
    }
    if (mu.view.w == 1u) {
        if (paper) {
            dst[o] = mu.c13.z;
            return;
        }
        let e = c13_encode(lin, mu.c13.x);
        let thr = threshold_at(x, y);
        let dithered = mu.c13.y != 0u;
        dst[o] = c13_code8(e.x, thr, dithered) | (c13_code8(e.y, thr, dithered) << 8u)
            | (c13_code8(e.z, thr, dithered) << 16u) | 0xff000000u;
    } else {
        let plane = mu.c13.w;
        dst[o] = bitcast<u32>(lin.x);
        dst[plane + o] = bitcast<u32>(lin.y);
        dst[2u * plane + o] = bitcast<u32>(lin.z);
        dst[3u * plane + o] = bitcast<u32>(select(0.0, 1.0, paper));
    }
}
