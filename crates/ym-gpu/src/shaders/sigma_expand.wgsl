// The noise sigma of a tile (G-06): one f32 plane for a rectangle of the frame, made from one of two
// sources. Mode 0 fills the rectangle with one value (the chain's scalar sigma). Mode 1 expands the
// half-resolution plane of the photo with the rule of `ym_core::pipeline::raw::full_sigma_map`: the
// sigma of the 2 x 2 cell the pixel sits in (the cell index clamped to the plane, `(x / 2).min(w - 1)`
// in integers, exactly the CPU's), times the vignetting gain at the cell's centre when the photo has
// one. The product is the one multiplication of the CPU's map, so the plane holds the CPU's bits for
// every cell; it is stored before any stage uses it, which keeps a compiler from re-associating it
// with the stage's own factors. The cells arrive as a rectangle of the half plane (`cells`) cut by the
// host, so a tile uploads a quarter of its pixels, not the photo's plane.
//
// Bindings: 0 uniform, 1 cell sigmas, 2 cell gains (a dummy word when `has_gains` is 0), 3 the plane.

struct SgParams {
    rect: vec4<u32>,    // x, y of the rectangle in the frame, width, height
    cells: vec4<u32>,   // first cell column and row in the buffer, buffer width, buffer height
    limit: vec4<u32>,   // last cell column and row of the whole plane, has_gains, mode
    dst: vec4<u32>,     // row stride of the output plane
    scalar: vec4<f32>,  // mode 0: the value
};

@group(0) @binding(0) var<uniform> sp: SgParams;
@group(0) @binding(1) var<storage, read> cell_data: array<f32>;
@group(0) @binding(2) var<storage, read> cell_gain: array<f32>;
@group(0) @binding(3) var<storage, read_write> sigma_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.rect.z || gid.y >= sp.rect.w) {
        return;
    }
    let o = gid.y * sp.dst.x + gid.x;
    if (sp.limit.w == 0u) {
        sigma_out[o] = sp.scalar.x;
        return;
    }
    let cx = min((sp.rect.x + gid.x) >> 1u, sp.limit.x) - sp.cells.x;
    let cy = min((sp.rect.y + gid.y) >> 1u, sp.limit.y) - sp.cells.y;
    let k = cy * sp.cells.z + cx;
    var v = cell_data[k];
    if (sp.limit.z != 0u) {
        v = v * cell_gain[k];
    }
    sigma_out[o] = v;
}
