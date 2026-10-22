// R6's coarse colour pass after method A and its false-colour suppression: the GPU port of
// `ym_gpu::cpu::coarse_chroma::CoarseChroma::apply` (`docs/algorithms/05-single-frame-denoise.md`, *Coarse
// colour*). Cells of CC_CELL x CC_CELL pixels anchored to the frame's origin, a bilateral filter of their colour
// within the photo's chroma noise, and the corrections interpolated back to the pixels as a change of Cb and Cr.
// Three pipelines of this file (`kernels/coarse_chroma.rs` appends one `main` to each):
//
//   coarse_chroma_cells    one thread per cell of the buffer: the mean Y, Cb and Cr of the working encoding
//                          (`bm_encode`, `bm_to_ycc`) over the cell's pixels in the buffer, summed in raster order
//                          from 0 and divided by their count, into scratch planes 0 to 2, and the cell's
//                          1 / (2 h^2) with h = strength sigma_C(cell) (`bm_level_var`, `bm_tau_of`) into plane 3
//                          (-1 where h is not above 0 or the quotient is not finite);
//   coarse_chroma_correct  one thread per cell: the weights exp(-d^2 max(own, theirs)) of the cells of the window
//                          (the narrower of the two widths; cut at the buffer's cells, a cell without a width left
//                          out), rows then columns from the window's corner, and the correction, the weighted mean of
//                          the differences of Cb and Cr from the cell's own, into scratch planes 4 and 5 (0 where the
//                          cell has no width);
//   coarse_chroma_apply    one thread per core pixel: the bilinear interpolation of the corrections of the four
//                          cells around it (centres at CC_CELL k + (CC_CELL - 1) / 2 in frame coordinates, cut at
//                          the buffer's cells), dR = CR_DIV dCr, dB = CB_DIV dCb, dG = -(KR dR + KB dB) / KG in
//                          the encoding, and the decoding; a pixel whose correction is 0 keeps its input bits.
//
// The cells and the interpolation are positions of the frame and every read is cut at the buffer's cells as on
// the CPU, so a tile with the stage's halo gives the whole frame's bits.
//
// Fast-math (Metal, docs/17-gpu.md K8): every product, sum and difference is behind `bm_op` (an OR with the
// uniform's zero word), the quotients are correctly rounded (`bm_div`), the encoding's power and the weight's
// exponential are our own (`bm_pow`, `bm_exp2` of working_encoding.wgsl, the exponent formed as an exact pair) and
// sigma_C's root is correctly rounded (`bm_sqrt`). The positions are exact in f32 (integers and quarters). Nothing
// is a decision with a jump: the weights, the interpolation and the change are continuous in the input, so f32
// against the CPU's f64 differs by rounding alone.
//
// Bindings: 0 uniform, 1 the input planes, 2 the scratch (6 dense planes of the buffer's cells), 3 the output
// (the core).

struct CcParams {
    dims: vec4<u32>,   // buffer width, height, cell plane length (cols * rows), the zero word
    io: vec4<u32>,     // input row stride, input plane length, output row stride, output plane length
    core: vec4<u32>,   // core x, y, width, height in the buffer
    grid: vec4<u32>,   // frame x and y of the buffer's pixel (0, 0), the buffer's cell columns and rows
    cell: vec4<u32>,   // frame column and row of the buffer's first cell, the window radius in cells, 0
    lvl: vec4<f32>,    // the photo's noise (`LevelNoise`): a, b, the encoding's gain G, and the strength
    gains: vec4<f32>,  // the gains g_c of R, G and B, 0
};

@group(0) @binding(0) var<uniform> cp: CcParams;
@group(0) @binding(1) var<storage, read> cc_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> cc_scr: array<f32>;
@group(0) @binding(3) var<storage, read_write> cc_out: array<f32>;

// Always 0 at run time, which the optimiser cannot know: OR-ing a value with it hides the value from
// reassociation, contraction and algebraic rewrites. An OR, never an XOR (docs/17-gpu.md K8). The helpers of
// working_encoding.wgsl call it too.
fn bm_op(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | cp.dims.w);
}

// exp(x) = 2^(x log2(e)), the exponent formed as the exact pair x LOG2E_HI (`bm_two_prod`) plus x LOG2E_LO, so
// the weight carries the rounding of `bm_exp2` alone.
fn cc_exp(x: f32) -> f32 {
    let p = bm_two_prod(x, CC_LOG2E_HI);
    return bm_exp2(p.x, bm_op(p.y + bm_op(x * CC_LOG2E_LO)));
}

fn cc_cells(gid: vec3<u32>) {
    let cols = cp.grid.z;
    if (gid.x >= cols || gid.y >= cp.grid.w) {
        return;
    }
    // The cell's pixels in the buffer: its frame span cut at the buffer's edges.
    let fx = (cp.cell.x + gid.x) * CC_CELL;
    let fy = (cp.cell.y + gid.y) * CC_CELL;
    let x0 = max(fx, cp.grid.x) - cp.grid.x;
    let y0 = max(fy, cp.grid.y) - cp.grid.y;
    let x1 = min(fx + CC_CELL, cp.grid.x + cp.dims.x) - cp.grid.x;
    let y1 = min(fy + CC_CELL, cp.grid.y + cp.dims.y) - cp.grid.y;
    var sy = 0.0;
    var sb = 0.0;
    var sr = 0.0;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            let s = y * cp.io.x + x;
            let ycc = bm_to_ycc(
                bm_encode(cc_src[s]),
                bm_encode(cc_src[cp.io.y + s]),
                bm_encode(cc_src[2u * cp.io.y + s]),
            );
            sy = bm_op(sy + ycc.x);
            sb = bm_op(sb + ycc.y);
            sr = bm_op(sr + ycc.z);
        }
    }
    let count = f32((x1 - x0) * (y1 - y0));
    let k = gid.y * cols + gid.x;
    let n = cp.dims.z;
    let m = vec3<f32>(bm_div(sy, count), bm_div(sb, count), bm_div(sr, count));
    cc_scr[k] = m.x;
    cc_scr[n + k] = m.y;
    cc_scr[2u * n + k] = m.z;
    // h = strength sigma_C at the cell's colour, `LevelNoise::chroma_sigma` scaled as method A's tau_C is, and
    // 1 / (2 h^2) as the CPU's `1.0 / (2.0 * h * h)`; -1 for a cell without a width.
    let v = vec3<f32>(
        bm_level_var(m, 0u, cp.lvl, cp.gains),
        bm_level_var(m, 1u, cp.lvl, cp.gains),
        bm_level_var(m, 2u, cp.lvl, cp.gains),
    );
    let h = bm_tau_of(v, cp.lvl.w);
    let inv = bm_div(1.0, bm_op(bm_op(2.0 * h) * h));
    var w = -1.0;
    if (h > 0.0 && bm_finite(h) && bm_finite(inv)) {
        w = inv;
    }
    cc_scr[3u * n + k] = w;
}

fn cc_correct(gid: vec3<u32>) {
    let cols = cp.grid.z;
    let rows = cp.grid.w;
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    let n = cp.dims.z;
    let k = gid.y * cols + gid.x;
    let own = vec2<f32>(cc_scr[n + k], cc_scr[2u * n + k]);
    let own_inv = cc_scr[3u * n + k];
    var db = 0.0;
    var dr = 0.0;
    if (own_inv >= 0.0) {
        let r = i32(cp.cell.z);
        let wy0 = u32(max(i32(gid.y) - r, 0));
        let wy1 = u32(min(i32(gid.y) + r, i32(rows) - 1));
        let wx0 = u32(max(i32(gid.x) - r, 0));
        let wx1 = u32(min(i32(gid.x) + r, i32(cols) - 1));
        var sw = 0.0;
        var sb = 0.0;
        var sr = 0.0;
        for (var wy = wy0; wy <= wy1; wy++) {
            for (var wx = wx0; wx <= wx1; wx++) {
                let j = wy * cols + wx;
                // Two cells are averaged only within the noise of both: the narrower weight.
                let j_inv = cc_scr[3u * n + j];
                if (j_inv < 0.0) {
                    continue;
                }
                let eb = bm_op(cc_scr[n + j] - own.x);
                let er = bm_op(cc_scr[2u * n + j] - own.y);
                let d2 = bm_op(bm_op(eb * eb) + bm_op(er * er));
                let wt = cc_exp(bm_op(-d2 * max(own_inv, j_inv)));
                sw = bm_op(sw + wt);
                sb = bm_op(sb + bm_op(wt * eb));
                sr = bm_op(sr + bm_op(wt * er));
            }
        }
        // The weighted mean of the differences, exactly 0 where the window's colour is the cell's. The cell
        // itself weighs 1, so sw >= 1.
        db = bm_div(sb, sw);
        dr = bm_div(sr, sw);
    }
    cc_scr[4u * n + k] = db;
    cc_scr[5u * n + k] = dr;
}

// The two cells along one axis around frame position p (their indices in the buffer's cells, cut at them) and
// the weight of the second: t = (p - (CC_CELL - 1) / 2) / CC_CELL, exact in f32 (`Grid::interpolate`).
fn cc_axis(p: u32, first: u32, len: u32) -> vec3<f32> {
    let t = bm_op(bm_op(f32(p) - CC_CENTRE) * CC_INV_CELL);
    let k0 = floor(t);
    let frac = bm_op(t - k0);
    let last = f32(len - 1u);
    let a = clamp(bm_op(k0 - f32(first)), 0.0, last);
    let b = clamp(bm_op(bm_op(k0 + 1.0) - f32(first)), 0.0, last);
    return vec3<f32>(a, b, frac);
}

fn cc_lerp(a: f32, b: f32, t: f32) -> f32 {
    return bm_op(a + bm_op(bm_op(b - a) * t));
}

fn cc_apply(gid: vec3<u32>) {
    if (gid.x >= cp.core.z || gid.y >= cp.core.w) {
        return;
    }
    let x = cp.core.x + gid.x;
    let y = cp.core.y + gid.y;
    let cols = cp.grid.z;
    let n = cp.dims.z;
    let ax = cc_axis(cp.grid.x + x, cp.cell.x, cols);
    let ay = cc_axis(cp.grid.y + y, cp.cell.y, cp.grid.w);
    let xa = u32(ax.x);
    let xb = u32(ax.y);
    let ya = u32(ay.x) * cols;
    let yb = u32(ay.y) * cols;
    var d = vec2<f32>(0.0, 0.0);
    for (var c = 0u; c < 2u; c++) {
        let base = (4u + c) * n;
        let top = cc_lerp(cc_scr[base + ya + xa], cc_scr[base + ya + xb], ax.z);
        let bottom = cc_lerp(cc_scr[base + yb + xa], cc_scr[base + yb + xb], ax.z);
        d[c] = cc_lerp(top, bottom, ay.z);
    }
    let s = y * cp.io.x + x;
    let o = gid.y * cp.io.z + gid.x;
    if (d.x == 0.0 && d.y == 0.0) {
        for (var c = 0u; c < 3u; c++) {
            cc_out[c * cp.io.w + o] = cc_src[c * cp.io.y + s];
        }
        return;
    }
    let de_r = bm_op(BM_CR_DIV * d.y);
    let de_b = bm_op(BM_CB_DIV * d.x);
    let de_g = bm_div(-bm_op(bm_op(BM_KR * de_r) + bm_op(BM_KB * de_b)), BM_KG);
    let de = vec3<f32>(de_r, de_g, de_b);
    for (var c = 0u; c < 3u; c++) {
        let e = bm_encode(cc_src[c * cp.io.y + s]);
        cc_out[c * cp.io.w + o] = bm_decode(bm_op(e + de[c]));
    }
}
