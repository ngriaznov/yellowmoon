// The input of L1 and L2 with the colour of the light it holds (docs/21 section 4, *As built (the light's
// colour)*): every pixel through ll_light_colour (lens_light_colour.wgsl, included before this text). The CPU
// reference is `ym_lens::lit_input`. A function of one pixel, so a tile and the whole frame give the same bits.
//
// Bindings: 0 uniform, 1 the input's planes, 2 the planes written (same size, own stride).

struct LightU {
    clip: vec4<f32>,   // g0 of each channel, 0
    ramp: vec4<f32>,   // the clip test's ramp: from, to, 0, 0
    src: vec4<u32>,    // input row stride, plane length, 0, 0
    dst: vec4<u32>,    // output row stride, plane length, width, height
};

@group(0) @binding(0) var<uniform> pu: LightU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn ll_main(gid: vec3<u32>) {
    if (gid.x >= pu.dst.z || gid.y >= pu.dst.w) {
        return;
    }
    let a = gid.y * pu.src.x + gid.x;
    let b = gid.y * pu.dst.x + gid.x;
    let v = vec3<f32>(src[a], src[pu.src.y + a], src[2u * pu.src.y + a]);
    let y = ll_light_colour(v, pu.clip.xyz, pu.ramp.xy);
    dst[b] = y.x;
    dst[pu.dst.y + b] = y.y;
    dst[2u * pu.dst.y + b] = y.z;
}
