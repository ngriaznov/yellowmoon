// The input of L1 and L2 with the colour of the light it holds, and L2's input with the headroom of the small
// clipped lights (docs/21 section 4, *As built (the light's colour)* and *As built (the headroom of a clipped
// light)*). Every pixel goes through ll_light (lens_light_colour.wgsl, included before this text); the CPU
// references are `ym_lens::lit_input` and `ym_lens::light::l2_input`. A function of one pixel and its place, so
// a tile and the whole frame give the same bits.
//
// Modes (mode.x):
//   0  the light's colour alone, over the input (L1's input);
//   1  L2's picture and extra light over the input raw window (L1 leaves the picture alone): the picture is the
//      raw one, the extra light the light in its own colour less it, with the headroom of a clipped small light;
//      the extra to the output planes, the picture to the own planes;
//   2  over L1's output: the same blended by the share L1 left the pixel as it is (its output within LL_L1_FROM to
//      LL_L1_TO clip levels of the picture with the light's colour, a smoothstep) with L1's output, which the
//      pixels it changed gather and spread none of.
//
// The headroom of a pixel: a g nu in every channel. a: how much of a clipped light the pixel is, the larger of
// the clip test and the white light weight; g: scale (2^(LL_SHARE ev) - 1) with the EV of the lens-plane pixel
// the pixel lies in (the map) and the cap's scale; nu: the smallest channel of the light's colour.
//
// Bindings: 0 uniform, 1 the raw planes, 2 the planes written, 3 L1's output (mode 2), 4 the picture written
// (modes 1 and 2), 5 the map (the EV of the lens plane, row-major).
//
// Generated before this text (kernels/lens_light.rs): LL_SHARE, LL_MAX, LL_L1_FROM, LL_L1_TO.

struct LightU {
    clip: vec4<f32>,   // g0 of each channel, 0
    ramp: vec4<f32>,   // the clip test's ramp: from, to, 0, 0
    src: vec4<u32>,    // input row stride, plane length, 0, 0
    dst: vec4<u32>,    // output row stride, plane length, width, height
    mode: vec4<u32>,   // the mode, the picture's scale k, the lens plane's scale k_L, 0
    org: vec4<i32>,    // the output's first frame column and row, the input's first frame column and row
    yin: vec4<i32>,    // L1's output: its first frame column and row, 0, 0
    ys: vec4<u32>,     // L1's output: row stride, plane length, 0, 0
    map: vec4<u32>,    // the map's width and height (0: no light), 0, 0
    mapf: vec4<f32>,   // the cap's scale, 0, 0, 0
};

@group(0) @binding(0) var<uniform> pu: LightU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> yin: array<f32>;
@group(0) @binding(4) var<storage, read_write> kept: array<f32>;
@group(0) @binding(5) var<storage, read> evmap: array<f32>;

// The EV of the lens-plane pixel the frame pixel (i, j) of the picture at scale k lies in: pixel i covers
// [i k, (i + 1) k) of the full frame, its centre falls in the plane pixel floor((2 i + 1) k / (2 k_L)).
fn ll_ev(i: u32, j: u32) -> f32 {
    if (pu.map.x == 0u) {
        return 0.0;
    }
    let kl = max(pu.mode.z, 1u);
    let px = min(((2u * i + 1u) * pu.mode.y) / (2u * kl), pu.map.x - 1u);
    let py = min(((2u * j + 1u) * pu.mode.y) / (2u * kl), pu.map.y - 1u);
    return evmap[py * pu.map.x + px];
}

// The light's colour of the pixel in xyz and the headroom it holds above what it records in w
// (`light::pixel_light`).
fn ll_headroom(v: vec3<f32>, ev: f32) -> vec4<f32> {
    let lw = ll_light(v, pu.clip.xyz, pu.ramp.xy);
    var m = 1.0e30;
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (pu.clip[c] > 0.0) {
            m = min(m, v[c] / pu.clip[c]);
        }
    }
    var f = 0.0;
    if (m < 1.0e29) {
        f = clamp((m - pu.ramp.x) / (pu.ramp.y - pu.ramp.x), 0.0, 1.0);
    }
    let a = max(f, lw.w);
    if (a <= 0.0 || ev <= 0.0) {
        return vec4<f32>(lw.xyz, 0.0);
    }
    let nu = max(min(min(lw.x, lw.y), lw.z), 0.0);
    let gain = pu.mapf.x * (exp2(LL_SHARE * ev) - 1.0);
    return vec4<f32>(lw.xyz, min(a * gain * nu, LL_MAX));
}

fn ll_main(gid: vec3<u32>) {
    if (gid.x >= pu.dst.z || gid.y >= pu.dst.w) {
        return;
    }
    let b = gid.y * pu.dst.x + gid.x;
    if (pu.mode.x == 0u) {
        let a = gid.y * pu.src.x + gid.x;
        let v = vec3<f32>(src[a], src[pu.src.y + a], src[2u * pu.src.y + a]);
        let y = ll_light(v, pu.clip.xyz, pu.ramp.xy).xyz;
        dst[b] = y.x;
        dst[pu.dst.y + b] = y.y;
        dst[2u * pu.dst.y + b] = y.z;
        return;
    }
    // The frame pixel, and the raw picture's pixel under it.
    let i = u32(pu.org.x) + gid.x;
    let j = u32(pu.org.y) + gid.y;
    let a = u32(i32(j) - pu.org.w) * pu.src.x + u32(i32(i) - pu.org.z);
    let x = vec3<f32>(src[a], src[pu.src.y + a], src[2u * pu.src.y + a]);
    let hr = ll_headroom(x, ll_ev(i, j));
    let lit = hr.xyz;
    // Without L1: the picture gathers as it is, and the light in its own colour less it, with the headroom, is
    // spread beside it.
    let picture = x;
    let more = lit - x + vec3<f32>(hr.w);
    if (pu.mode.x == 1u) {
        dst[b] = more.x;
        dst[pu.dst.y + b] = more.y;
        dst[2u * pu.dst.y + b] = more.z;
        kept[b] = picture.x;
        kept[pu.dst.y + b] = picture.y;
        kept[2u * pu.dst.y + b] = picture.z;
        return;
    }
    // Mode 2: L1's output at the same pixel.
    let l = u32(i32(j) - pu.yin.y) * pu.ys.x + u32(i32(i) - pu.yin.x);
    let v = vec3<f32>(yin[l], yin[pu.ys.y + l], yin[2u * pu.ys.y + l]);
    var d = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        var g0 = pu.clip[c];
        if (!(g0 > 0.0)) {
            g0 = 1.0;
        }
        d = max(d, abs(v[c] - lit[c]) / g0);
    }
    let t = clamp((d - LL_L1_FROM) / (LL_L1_TO - LL_L1_FROM), 0.0, 1.0);
    let m = 1.0 - t * t * (3.0 - 2.0 * t);
    let e2 = m * more;
    let k = m * picture + (1.0 - m) * v;
    dst[b] = e2.x;
    dst[pu.dst.y + b] = e2.y;
    dst[2u * pu.dst.y + b] = e2.z;
    kept[b] = k.x;
    kept[pu.dst.y + b] = k.y;
    kept[2u * pu.dst.y + b] = k.z;
}
