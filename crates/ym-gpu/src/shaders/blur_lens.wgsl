// The Lens gather of a Blur node (docs/33-blur.md section 3.5): every cell of the output window becomes the table's weighted sum
// of the cells around it, the table being the coverage of a cell by the aperture (a disc or a regular polygon), normalised and
// trimmed to at most 17 x 17 cells (8 x 8 supersampled in f64, a function of R / f, the blades and the rotation alone, made on
// the CPU and sent as an f32 copy). A cell outside the source grid has no weight; the taps run row by row, dx fastest, a weight
// of exactly 0 skipped, which is the order of the CPU twin (`ym_color::blur::lens`).
//
// At level 0 a cell is a pixel and this is the exact kernel's gather. At a higher level the cells are blurred and the caller reads
// them back to pixels bilinearly (`blur_upsample.wgsl`).
//
// Bindings: 0 uniform, 1 the source cells (four planes: `omega X` and `omega` of the cell), 2 the output cells (four planes).

struct LU {
    // The source image's place of its cell (0, 0) (cell indices), its row stride and its plane length.
    src: vec4<i32>,
    // The same for the output image.
    dst: vec4<i32>,
    // The window computed: first column, first row, width, height (cell indices).
    win: vec4<i32>,
    // x, y: the source grid's width and height in cells; z: the table's half width; w: 0.
    dims: vec4<i32>,
    // The table, row-major over (2 half + 1)^2 entries, four to a vector.
    table: array<vec4<f32>, 73>,
}

@group(0) @binding(0) var<uniform> lu: LU;
@group(0) @binding(1) var<storage, read> cells: array<f32>;
@group(0) @binding(2) var<storage, read_write> outc: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= lu.win.z || i32(gid.y) >= lu.win.w) {
        return;
    }
    let i = lu.win.x + i32(gid.x);
    let j = lu.win.y + i32(gid.y);
    let half = lu.dims.z;
    let side = 2 * half + 1;
    let sp = u32(lu.src.w);
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var dy = -half; dy <= half; dy = dy + 1) {
        let jj = j + dy;
        if (jj < lu.src.y || jj >= lu.src.y + lu.dims.y) {
            continue;
        }
        let row = (dy + half) * side;
        for (var dx = -half; dx <= half; dx = dx + 1) {
            let ii = i + dx;
            if (ii < lu.src.x || ii >= lu.src.x + lu.dims.x) {
                continue;
            }
            let idx = u32(row + dx + half);
            let k = lu.table[idx >> 2u][idx & 3u];
            if (k == 0.0) {
                continue;
            }
            let o = u32(jj - lu.src.y) * u32(lu.src.z) + u32(ii - lu.src.x);
            a0 = a0 + k * cells[o];
            a1 = a1 + k * cells[o + sp];
            a2 = a2 + k * cells[o + 2u * sp];
            a3 = a3 + k * cells[o + 3u * sp];
        }
    }
    let d = u32(j - lu.dst.y) * u32(lu.dst.z) + u32(i - lu.dst.x);
    let dp = u32(lu.dst.w);
    outc[d] = a0;
    outc[d + dp] = a1;
    outc[d + 2u * dp] = a2;
    outc[d + 3u * dp] = a3;
}
