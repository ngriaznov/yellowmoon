// M0-07 spike, kernel 1: R1 normalise + R4 gains (docs/05). One thread handles two horizontally
// adjacent photosites, so packed f16 words are written by exactly one thread.
//
//   x = max((raw - black[p]) * inv_range[p], -0.05) * gain[p]
//
// p is the CFA position 2 * (frame_y & 1) + (frame_x & 1); `gain[p]` is the gain of the colour at
// that position (ym_gpu::cpu::normalize::normalize_tile followed by gains::apply_gains).

struct U {
    dims: vec4<u32>,     // x: width, y: height, z: stride (even), w: plane length
    src: vec4<u32>,
    parity: vec4<u32>,   // x: frame x0 & 1, y: frame y0 & 1
    black: vec4<f32>,
    inv_range: vec4<f32>,
    gain: vec4<f32>,
    cfa: vec4<u32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> raw: array<u32>;      // u16 samples, two per word
@group(0) @binding(2) var<storage, read_write> mosaic: array<ST_ELEM>;

//@ACCESSORS mosaic

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * 2u;
    let y = gid.y;
    if (x >= u.dims.z || y >= u.dims.y) {
        return;
    }
    let i = y * u.dims.z + x;           // even: the stride is even
    let word = raw[i >> 1u];
    let v0 = f32(word & 0xffffu);
    let v1 = f32(word >> 16u);
    let row = 2u * ((u.parity.y + y) & 1u);
    let p0 = row + ((u.parity.x + x) & 1u);
    let p1 = row + ((u.parity.x + x + 1u) & 1u);
    let n0 = max((v0 - u.black[p0]) * u.inv_range[p0], -0.05) * u.gain[p0];
    let n1 = max((v1 - u.black[p1]) * u.inv_range[p1], -0.05) * u.gain[p1];
    mosaic_st2(i, n0, n1);
}
