// M0-07 spike, kernel 3: fixed 3x3 matrix, then the sRGB transfer (IEC 61966-2-1), clamped to
// [0, 1]. Runs over the tile *core* only: `src` says where the core starts inside the compute
// rectangle held in the planes. Output kinds: 0 = f32 planar, 1 = RGBA8, 2 = u16 planar (pairs).

struct U {
    dims: vec4<u32>,     // x: out width, y: out height, z: out stride (even), w: out plane length
    src: vec4<u32>,      // x: src stride, y: src plane length, z: core dx, w: core dy
    parity: vec4<u32>,   // z: output kind
    black: vec4<f32>,
    inv_range: vec4<f32>,
    gain: vec4<f32>,
    cfa: vec4<u32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rgb: array<ST_ELEM>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

//@ACCESSORS rgb ro

fn encode_px(i: u32) -> vec3<f32> {
    let pl = u.src.y;
    let c = vec3<f32>(rgb_ld(i), rgb_ld(pl + i), rgb_ld(2u * pl + i));
    let lin = vec3<f32>(dot(u.m0.xyz, c), dot(u.m1.xyz, c), dot(u.m2.xyz, c));
    return vec3<f32>(unit(srgb_encode(lin.x)), unit(srgb_encode(lin.y)), unit(srgb_encode(lin.z)));
}

fn q8(v: f32) -> u32 {
    return u32(floor(v * 255.0 + 0.5));
}

fn q16(v: f32) -> u32 {
    return u32(floor(v * 65535.0 + 0.5));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * 2u;
    let y = gid.y;
    if (x >= u.dims.z || y >= u.dims.y) {
        return;
    }
    let s = (y + u.src.w) * u.src.x + x + u.src.z;
    let a = encode_px(s);
    let b = encode_px(s + 1u);
    let o = y * u.dims.z + x;
    let pl = u.dims.w;
    let kind = u.parity.z;
    if (kind == 0u) {
        out[o] = bitcast<u32>(a.x);
        out[o + 1u] = bitcast<u32>(b.x);
        out[pl + o] = bitcast<u32>(a.y);
        out[pl + o + 1u] = bitcast<u32>(b.y);
        out[2u * pl + o] = bitcast<u32>(a.z);
        out[2u * pl + o + 1u] = bitcast<u32>(b.z);
    } else if (kind == 1u) {
        out[o] = q8(a.x) | (q8(a.y) << 8u) | (q8(a.z) << 16u) | 0xff000000u;
        out[o + 1u] = q8(b.x) | (q8(b.y) << 8u) | (q8(b.z) << 16u) | 0xff000000u;
    } else {
        out[o >> 1u] = q16(a.x) | (q16(b.x) << 16u);
        out[(pl + o) >> 1u] = q16(a.y) | (q16(b.y) << 16u);
        out[(2u * pl + o) >> 1u] = q16(a.z) | (q16(b.z) << 16u);
    }
}
