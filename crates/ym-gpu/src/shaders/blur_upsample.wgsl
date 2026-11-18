// The read-back of a level of cells to pixels (docs/33 section 3.5: "the result is read back bilinearly"): for each pixel of
// the window the cell that holds it (nearest) or the four cells around its centre (bilinear, a cell's centre at
// (I + 1/2) f). A cell outside the grid reads 0, so a pixel near the frame's edge keeps its weight (`D`) with its sum (`N`).
//
// Bindings: 0 uniform, 1 the cells (four planes, image A, its origin the cell index of its first cell, `op.zw` its size),
// 2 the pixels (four planes, image B, its origin the place of its pixel (0, 0)). `op.x` is the level, `op.y` the mode.

@group(0) @binding(1) var<storage, read> cells: array<f32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;

fn blur_cell(i: i32, j: i32, plane: u32) -> f32 {
    if (i < bu.a.x || j < bu.a.y || i >= bu.a.x + i32(bu.op.z) || j >= bu.a.y + i32(bu.op.w)) {
        return 0.0;
    }
    return cells[bu_at(bu.a, i, j, plane)];
}

fn blur_upsample(gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let level = bu.op.x;
    let f = f32(1u << level);
    for (var c = 0u; c < 4u; c = c + 1u) {
        var v = 0.0;
        if (bu.op.y == 0u) {
            v = blur_cell(x >> level, y >> level, c);
        } else {
            let u = (f32(x) + 0.5) / f - 0.5;
            let w = (f32(y) + 0.5) / f - 0.5;
            let i0 = i32(floor(u));
            let j0 = i32(floor(w));
            let tx = u - f32(i0);
            let ty = w - f32(j0);
            let top = (1.0 - tx) * blur_cell(i0, j0, c) + tx * blur_cell(i0 + 1, j0, c);
            let bottom = (1.0 - tx) * blur_cell(i0, j0 + 1, c) + tx * blur_cell(i0 + 1, j0 + 1, c);
            v = (1.0 - ty) * top + ty * bottom;
        }
        pixels[bu_at(bu.b, x, y, c)] = v;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur_upsample(gid);
}
