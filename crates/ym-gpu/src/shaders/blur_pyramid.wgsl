// One level of the pyramid of the weighted planes (docs/33 section 3.5, the coarse grid and Zoom's pyramid): each cell is the
// mean of its 2 x 2 children at absolute alignment, so a cell never depends on the tile. A child outside the lower level's
// grid reads 0 (it lies outside the frame: the grid holds every cell the frame reaches that the read area covers).
//
// Bindings: 0 uniform, 1 the lower level (four planes, image A, its origin the cell index of its first cell, `op.zw` its
// size in cells), 2 this level (four planes, image B).

@group(0) @binding(1) var<storage, read> lower: array<f32>;
@group(0) @binding(2) var<storage, read_write> upper: array<f32>;

fn blur_child(ci: i32, cj: i32, plane: u32) -> f32 {
    if (ci < bu.a.x || cj < bu.a.y || ci >= bu.a.x + i32(bu.op.z) || cj >= bu.a.y + i32(bu.op.w)) {
        return 0.0;
    }
    return lower[bu_at(bu.a, ci, cj, plane)];
}

fn blur_pyramid(gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let i = bu.win.x + i32(gid.x);
    let j = bu.win.y + i32(gid.y);
    for (var c = 0u; c < 4u; c = c + 1u) {
        let s00 = blur_child(2 * i, 2 * j, c);
        let s10 = blur_child(2 * i + 1, 2 * j, c);
        let s01 = blur_child(2 * i, 2 * j + 1, c);
        let s11 = blur_child(2 * i + 1, 2 * j + 1, c);
        upper[bu_at(bu.b, i, j, c)] = ((s00 + s10) + (s01 + s11)) * 0.25;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur_pyramid(gid);
}
