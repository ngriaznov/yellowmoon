// M0-07 spike, kernel 2: bilinear demosaic (ym_gpu::cpu::demosaic::bilinear), mirrored edges.
// Writes three planes (R, G, B) of `plane length` samples each.

struct U {
    dims: vec4<u32>,     // x: width, y: height, z: stride (even), w: plane length
    src: vec4<u32>,
    parity: vec4<u32>,
    black: vec4<f32>,
    inv_range: vec4<f32>,
    gain: vec4<f32>,
    cfa: vec4<u32>,      // colour (0 R, 1 G, 2 B) at local position 2 * (y & 1) + (x & 1)
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> mosaic: array<ST_ELEM>;
@group(0) @binding(2) var<storage, read_write> rgb: array<ST_ELEM>;

//@ACCESSORS mosaic ro
//@ACCESSORS rgb

fn color_at(x: i32, y: i32) -> u32 {
    return u.cfa[2u * u32(y & 1) + u32(x & 1)];
}

fn at(x: i32, y: i32) -> f32 {
    let w = i32(u.dims.x);
    let h = i32(u.dims.y);
    return mosaic_ld(u32(mirror_idx(y, h)) * u.dims.z + u32(mirror_idx(x, w)));
}

fn demosaic_px(x: i32, y: i32) -> vec3<f32> {
    let w = i32(u.dims.x);
    let v = at(x, y);
    let cross = 0.25 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1));
    let diag = 0.25 * (at(x - 1, y - 1) + at(x + 1, y - 1) + at(x - 1, y + 1) + at(x + 1, y + 1));
    let horiz = 0.5 * (at(x - 1, y) + at(x + 1, y));
    let vert = 0.5 * (at(x, y - 1) + at(x, y + 1));
    let c = color_at(x, y);
    if (c == 0u) {
        return vec3<f32>(v, cross, diag);
    }
    if (c == 2u) {
        return vec3<f32>(diag, cross, v);
    }
    // Green: the colour found left/right of it is the one that owns its row.
    if (color_at(mirror_idx(x + 1, w), y) == 0u) {
        return vec3<f32>(horiz, v, vert);
    }
    return vec3<f32>(vert, v, horiz);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * 2u;
    let y = gid.y;
    if (x >= u.dims.z || y >= u.dims.y) {
        return;
    }
    let a = demosaic_px(i32(x), i32(y));
    let b = demosaic_px(i32(x) + 1, i32(y));
    let i = y * u.dims.z + x;
    let pl = u.dims.w;
    rgb_st2(i, a.x, b.x);
    rgb_st2(pl + i, a.y, b.y);
    rgb_st2(2u * pl + i, a.z, b.z);
}
