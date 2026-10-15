// The photo's raw words on the device, read by the raw-front kernels (G-07). `DeviceRaw` keeps the
// words of a readout in one or two row slabs (a slab is at most one 128 MiB binding: a 61 MP plane of
// u16 fits one, a 70 MP plane needs two), two samples per u32, the first in the low half, linear in
// the slab (`(y - first_row) * width + x`). The cRAW quantisation step is a second pair of slabs: mode
// 1 holds f16 bits, two per word, mode 2 one byte per sample that indexes the table `qtab` (the step
// set of a file has at most 256 values). The slab bindings are always bound: a photo with one slab
// binds it twice, a photo without a step plane binds the words as the step buffers, and `dims.w` says
// which are read.
//
// A kernel that includes this file declares the uniform `pu` with a member `fr: RawFrame` (and the
// zero word of `front_opaque.wgsl`) and uses bindings 1 to 4 for these buffers.

struct RawFrame {
    dims: vec4<u32>,   // width, height, first row of the second slab (== height: one slab), quantisation mode
    black: vec4<f32>,  // black level per CFA position of the frame, 2 * (y & 1) + (x & 1)
    inv: vec4<f32>,    // 1 / (white - black), as the CPU forms it in f32
    white: vec4<f32>,
    clip: vec4<f32>,   // QSTEP_MARGIN, CLIP_FRACTION, 0, 0
    qtab: array<vec4<f32>, 64>,
};

@group(0) @binding(1) var<storage, read> raw0: array<u32>;
@group(0) @binding(2) var<storage, read> raw1: array<u32>;
@group(0) @binding(3) var<storage, read> qs0: array<u32>;
@group(0) @binding(4) var<storage, read> qs1: array<u32>;

// The raw word of photosite (x, y) of the frame.
fn raw_u16(x: u32, y: u32) -> u32 {
    let w = pu.fr.dims.x;
    if (y >= pu.fr.dims.z) {
        let i = (y - pu.fr.dims.z) * w + x;
        return (raw1[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
    }
    let i = y * w + x;
    return (raw0[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
}

// An f16 given by its bits, as an f32, exactly (no f16 capability is needed; steps are finite).
fn f16_to_f32(h: u32) -> f32 {
    let neg = (h & 0x8000u) != 0u;
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x3ffu;
    if (e == 0u) {
        // Zero or subnormal: m * 2^-24, exact.
        let v = f32(m) * 5.9604645e-8;
        return select(v, -v, neg);
    }
    let v = bitcast<f32>(((e + 112u) << 23u) | (m << 13u));
    return select(v, -v, neg);
}

// The quantisation step of photosite (x, y) in raw units (mode 1 or 2 only).
fn q_at(x: u32, y: u32) -> f32 {
    let w = pu.fr.dims.x;
    var i: u32;
    var word: u32;
    if (y >= pu.fr.dims.z) {
        i = (y - pu.fr.dims.z) * w + x;
        if (pu.fr.dims.w == 1u) {
            word = qs1[i >> 1u];
        } else {
            word = qs1[i >> 2u];
        }
    } else {
        i = y * w + x;
        if (pu.fr.dims.w == 1u) {
            word = qs0[i >> 1u];
        } else {
            word = qs0[i >> 2u];
        }
    }
    if (pu.fr.dims.w == 1u) {
        return f16_to_f32((word >> ((i & 1u) * 16u)) & 0xffffu);
    }
    let b = (word >> ((i & 3u) * 8u)) & 0xffu;
    return pu.fr.qtab[b >> 2u][b & 3u];
}

// R1's clip rule, as `cpu::normalize` writes it: with a quantisation step plane `raw >= white - 4 q`,
// without one `(raw - black) * inv >= 0.995`.
fn is_clipped(x: u32, y: u32) -> bool {
    let p = 2u * (y & 1u) + (x & 1u);
    let v = f32(raw_u16(x, y));
    if (pu.fr.dims.w == 0u) {
        let n = opaque((v - pu.fr.black[p]) * pu.fr.inv[p]);
        return n >= pu.fr.clip.y;
    }
    return v >= pu.fr.white[p] - pu.fr.clip.x * q_at(x, y);
}

// R1 for one photosite: `max((raw - black) * inv, floor)`.
fn normalised(x: u32, y: u32, floor_v: f32) -> f32 {
    let p = 2u * (y & 1u) + (x & 1u);
    let v = f32(raw_u16(x, y));
    return max(opaque((v - pu.fr.black[p]) * pu.fr.inv[p]), floor_v);
}
