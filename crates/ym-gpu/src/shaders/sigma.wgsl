// The noise sigma plane at half resolution (G-07): the GPU port of `ym-core`'s `sigma_plane`, which is
// R1 and R4 on the mosaic (no vignetting) followed by `ym_noise::sigma_map::sigma_quad`
// (`docs/06-noise.md`). One thread per 2 x 2 cell:
//
//   plane_p(a, b) = max((raw(2a + px, 2b + py) - black) * inv, -0.05) * g0_p      p = 2 py + px
//   x_p           = box5(plane_p) / g0_p                the 5 x 5 mean, edges replicated
//   sigma_p       = g0_p * sqrt(a max(x_p - delta_p, 0) + k0 + q^2 / 12)
//   sigma         = sqrt((sigma_0^2 + sigma_1^2 + sigma_2^2 + sigma_3^2) / 4)
//
// The box mean is the CPU's two passes: five values added left to right and scaled by 0.2, then the
// five row results added top to bottom and scaled by 0.2; each of the 25 values is normalised where it
// is read (a cell reads 100 raw words), so the pass needs no scratch plane and no tile edge: a cell
// depends only on its 5 x 5 neighbourhood of cells (D5). Every product and partial sum is hidden from
// fast-math (`opaque`), so the sums keep the CPU's order.
//
// Bindings: 0 uniform, 1 to 4 the raw words and steps (`front_raw.wgsl`), 5 the sigma plane.

struct SigmaU {
    zero: vec4<u32>,   // always +0.0 bits
    fr: RawFrame,
    cells: vec4<u32>,  // cell columns, cell rows, output row stride, 0
    g0: vec4<f32>,     // R4 gain per CFA position of the frame
    inv_g: vec4<f32>,  // 1 / g_total per position (1 where g_total is not positive)
    g_tot: vec4<f32>,  // max(g_total, 0) per position
    delta: vec4<f32>,  // the model's delta per position
    k: vec4<f32>,      // a, k0 (b + sigma_row^2), q^2 / 12, the floor of R1
    norm: vec4<f32>,   // the box filter's 1 / 5, 0.25, 0, 0
};

@group(0) @binding(0) var<uniform> pu: SigmaU;
@group(0) @binding(5) var<storage, read_write> sigma_out: array<f32>;

// Plane p's balanced value at cell (a, b), the cell index clamped to the plane.
fn plane_value(p: u32, a: i32, b: i32) -> f32 {
    let ca = u32(clamp(a, 0, i32(pu.cells.x) - 1));
    let cb = u32(clamp(b, 0, i32(pu.cells.y) - 1));
    let x = 2u * ca + (p & 1u);
    let y = 2u * cb + (p >> 1u);
    return opaque(normalised(x, y, pu.k.w) * pu.g0[p]);
}

fn sigma_main(gid: vec3<u32>) {
    if (gid.x >= pu.cells.x || gid.y >= pu.cells.y) {
        return;
    }
    let a = i32(gid.x);
    let b = i32(gid.y);
    var s2 = 0.0;
    for (var p = 0u; p < 4u; p = p + 1u) {
        // The vertical pass over the five horizontal results.
        var col = 0.0;
        for (var dy = -2; dy <= 2; dy = dy + 1) {
            var row = 0.0;
            for (var dx = -2; dx <= 2; dx = dx + 1) {
                row = opaque(row + plane_value(p, a + dx, b + dy));
            }
            col = opaque(col + opaque(row * pu.norm.x));
        }
        let mean5 = opaque(col * pu.norm.x);
        let x = opaque(mean5 * pu.inv_g[p]);
        let shot = opaque(pu.k.x * max(x - pu.delta[p], 0.0));
        let v = opaque(opaque(shot + pu.k.y) + pu.k.z);
        let sg = opaque(pu.g_tot[p] * sqrt(v));
        s2 = opaque(s2 + opaque(sg * sg));
    }
    sigma_out[gid.y * pu.cells.z + gid.x] = sqrt(opaque(pu.norm.y * s2));
}
