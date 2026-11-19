// Soft's kernel of the Blur node (docs/33-blur.md section 3.5, docs/17-gpu.md section 11b; `kernels/blur_soft.rs`): the
// pixel-integrated Gaussian cut at 4 sigma, as a direct separable sum on the pixels (sigma up to 4 px) or on a coarse grid of
// cells of f x f pixels at absolute multiples of f (beyond): a quadratic B-spline reduction of the weighted planes to the cells,
// the Gaussian of the cells, and the B-spline read-back. One entry, five modes (`bu.op.x`):
//
//   0  reduce along x: cell column I of pixel row y = (1/f) sum_x spline(x, I) src(x, y), x in [(I - 1) f, (I + 2) f)
//   1  reduce along y: cell (I, J) = (1/f) sum_y spline(y, J) src(I, y), y in [(J - 1) f, (J + 2) f)
//   2  the Gaussian along x: dst(i, j) = sum_k w[|k|] src(i + k, j)
//   3  the Gaussian along y
//   4  the read-back: pixel (x, y) = sum_J sum_I spline(y, J) spline(x, I) src(I, J), cells (I, J) in the grid
//
// Bindings: 0 uniform BU, 1 the source (four planes, image A), 2 the result (four planes, image B), 3 the one-sided weights
// of the Gaussian (a storage buffer of 36 floats: at most 16 taps each side; the contract allows one uniform, K3). `bu.c` is the range the source holds, in its own
// coordinates: first column, first row, width, height (taps outside it are skipped, as the CPU's); for the read-back it is the
// grid of cells. `bu.op.y` is f, `bu.op.z` the taps each side. Every sum runs in ascending order; the spline's weights are
// integers over a power of two, exact in f32 (docs/33 section 3.7).

@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> wts: array<f32>;

fn soft_w(k: u32) -> f32 {
    return wts[k];
}

// The quadratic B-spline weight of pixel x on cell i of cells of f pixels (ym_color::blur::soft::spline).
fn soft_spline(x: i32, i: i32, f: i32) -> f32 {
    let d = abs(2 * x + 1 - (2 * i + 1) * f);
    if (d <= f) {
        return f32(3 * f * f - d * d) / f32(4 * f * f);
    }
    if (d < 3 * f) {
        let a = f32(3 * f - d);
        return a * a / f32(8 * f * f);
    }
    return 0.0;
}

fn soft_src_in(i: i32, j: i32) -> bool {
    return i >= bu.c.x && j >= bu.c.y && i < bu.c.x + bu.c.z && j < bu.c.y + bu.c.w;
}

fn soft_src(i: i32, j: i32, plane: u32) -> f32 {
    return src[bu_at(bu.a, i, j, plane)];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let mode = bu.op.x;
    let f = i32(bu.op.y);
    let half = i32(bu.op.z);
    for (var c = 0u; c < 4u; c = c + 1u) {
        var acc = 0.0;
        if (mode == 0u) {
            // x is the cell column, y the pixel row.
            let lo = max((x - 1) * f, bu.c.x);
            let hi = min((x + 2) * f, bu.c.x + bu.c.z);
            for (var px = lo; px < hi; px = px + 1) {
                acc = acc + soft_spline(px, x, f) * soft_src(px, y, c);
            }
            acc = acc / f32(f);
        } else if (mode == 1u) {
            // x is the cell column (a column of the source), y the cell row.
            let lo = max((y - 1) * f, bu.c.y);
            let hi = min((y + 2) * f, bu.c.y + bu.c.w);
            for (var py = lo; py < hi; py = py + 1) {
                acc = acc + soft_spline(py, y, f) * soft_src(x, py, c);
            }
            acc = acc / f32(f);
        } else if (mode == 2u) {
            for (var k = -half; k <= half; k = k + 1) {
                let i = x + k;
                if (i >= bu.c.x && i < bu.c.x + bu.c.z) {
                    acc = acc + soft_w(u32(abs(k))) * soft_src(i, y, c);
                }
            }
        } else if (mode == 3u) {
            for (var k = -half; k <= half; k = k + 1) {
                let j = y + k;
                if (j >= bu.c.y && j < bu.c.y + bu.c.w) {
                    acc = acc + soft_w(u32(abs(k))) * soft_src(x, j, c);
                }
            }
        } else {
            // The read-back at pixel (x, y): the nearest cell is x div f (floor).
            let i0 = i32(floor(f32(x) / f32(f)));
            let j0 = i32(floor(f32(y) / f32(f)));
            for (var j = j0 - 1; j <= j0 + 1; j = j + 1) {
                if (j < bu.c.y || j >= bu.c.y + bu.c.w) {
                    continue;
                }
                let ky = soft_spline(y, j, f);
                var line = 0.0;
                for (var i = i0 - 1; i <= i0 + 1; i = i + 1) {
                    if (i < bu.c.x || i >= bu.c.x + bu.c.z) {
                        continue;
                    }
                    line = line + soft_spline(x, i, f) * soft_src(i, j, c);
                }
                acc = acc + ky * line;
            }
        }
        dst[bu_at(bu.b, x, y, c)] = acc;
    }
}
