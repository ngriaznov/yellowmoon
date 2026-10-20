// Method A, one pass of A.5 (`bm_dct.wgsl` has the notes): one workgroup of 256 threads per reference block of
// the pass, the reference at frame (refs.x + refs.z gx, refs.y + refs.w gy) for workgroup (gx, gy). The host
// dispatches the passes in the CPU's order (`cpu::bm_dct` `Ctx::passes`: the regular passes (i, j), j outer,
// then the extra column, row and corner) and only references whose read span lies inside the buffer
// (reading A-d), so the reference, its candidates and its group are all buffer pixels.
//
// Per reference (`cpu::bm_dct` `Ctx::filter` and `Ctx::scatter`):
//   A  the 24 x 24 luma window of the search (the candidates' corners lie within 8 px of the reference's) into
//      workgroup memory;
//   B  per candidate (25) and row (8): the row's share of D = sum K(i, j) |Y_ref - Y_cand|, left to right;
//   C  per candidate: D, the rows in order; a block not wholly inside the frame, or a D that is not finite, is
//      no candidate (reading A-a, A-e);
//   D  thread 0: the group, as `cpu::bm_dct::select` (the 7 nearest by D, a tie to the smaller index, the
//      reference for a slot over theta or without a candidate);
//   E  the group's 8 blocks of every filtered channel into workgroup memory;
//   F  the 8-point transform along the rows (one line per thread), G along the columns, H across the blocks
//      with the hard threshold |X| < tau -> 0 (the DC term included) and the inverse across in the same thread,
//      I the inverse along the columns, J along the rows (`Dct8::forward_3d`, `inverse_3d`: the even and odd
//      halves, the dot products left to right);
//   K  the aggregation, acc_ch += K(i, j) v / 512, w += K(i, j), slot by slot with a storage barrier
//      between the slots: inside a slot the 64 pixels are distinct, so the four planes of a pixel are added by
//      four threads without a race, and slot k + 1 sees slot k. Two references of a pass are 24 px apart and
//      write only inside their 24 x 24 footprints, so a pixel gets at most one reference per pass, and the
//      passes are dispatches in order: every pixel's sums are added in the CPU's order, which is what makes a
//      tile equal the whole frame bit for bit.
//
// Workgroup memory: 576 + 200 + 25 + 25 + 16 + 1536 words, 9.5 KiB.
// Bindings: 0 uniform, 2 the Y/Cb/Cr scratch (3 planes), 3 the accumulators (4 planes).

@group(0) @binding(2) var<storage, read> bm_ycc: array<f32>;
@group(0) @binding(3) var<storage, read_write> bm_acc: array<f32>;

var<workgroup> bm_win: array<f32, 576>;
var<workgroup> bm_part: array<f32, 200>;
var<workgroup> bm_dist: array<f32, 25>;
var<workgroup> bm_ok: array<u32, 25>;
var<workgroup> bm_sx: array<u32, 8>;
var<workgroup> bm_sy: array<u32, 8>;
var<workgroup> bm_grp: array<f32, 1536>;

// K(i, j) at q = 8 i + j.
fn bm_kwin(q: u32) -> f32 {
    return bp.kwin[q >> 2u][q & 3u];
}

// Whether the block at frame (rx + dx, ry + dy) lies wholly inside the frame.
fn bm_inside(rx: u32, ry: u32, dx: i32, dy: i32) -> bool {
    let cx = i32(rx) + dx;
    let cy = i32(ry) + dy;
    return cx >= 0 && cy >= 0 && u32(cx) + 8u <= bp.frame.x && u32(cy) + 8u <= bp.frame.y;
}

// r.x a + r.y b + r.z c + r.w d, left to right, every product and partial sum pinned.
fn bm_dot4(r: vec4<f32>, a: f32, b: f32, c: f32, d: f32) -> f32 {
    var acc = bm_op(r.x * a);
    acc = bm_op(acc + bm_op(r.y * b));
    acc = bm_op(acc + bm_op(r.z * c));
    return bm_op(acc + bm_op(r.w * d));
}

// The forward 8-point transform (`Dct8::forward`): the orthonormal DCT-II times sqrt(8), from the even and odd
// halves x[n] +- x[7 - n].
fn bm_fwd8(x: array<f32, 8>) -> array<f32, 8> {
    let s0 = bm_op(x[0] + x[7]);
    let s1 = bm_op(x[1] + x[6]);
    let s2 = bm_op(x[2] + x[5]);
    let s3 = bm_op(x[3] + x[4]);
    let d0 = bm_op(x[0] - x[7]);
    let d1 = bm_op(x[1] - x[6]);
    let d2 = bm_op(x[2] - x[5]);
    let d3 = bm_op(x[3] - x[4]);
    return array<f32, 8>(
        bm_dot4(BM_E0, s0, s1, s2, s3),
        bm_dot4(BM_O0, d0, d1, d2, d3),
        bm_dot4(BM_E1, s0, s1, s2, s3),
        bm_dot4(BM_O1, d0, d1, d2, d3),
        bm_dot4(BM_E2, s0, s1, s2, s3),
        bm_dot4(BM_O2, d0, d1, d2, d3),
        bm_dot4(BM_E3, s0, s1, s2, s3),
        bm_dot4(BM_O3, d0, d1, d2, d3),
    );
}

// The inverse 8-point transform (`Dct8::inverse`, the transpose): out[n] = even + odd, out[7 - n] = even - odd.
fn bm_inv8(c: array<f32, 8>) -> array<f32, 8> {
    let e0 = bm_dot4(BM_EC0, c[0], c[2], c[4], c[6]);
    let e1 = bm_dot4(BM_EC1, c[0], c[2], c[4], c[6]);
    let e2 = bm_dot4(BM_EC2, c[0], c[2], c[4], c[6]);
    let e3 = bm_dot4(BM_EC3, c[0], c[2], c[4], c[6]);
    let o0 = bm_dot4(BM_OC0, c[1], c[3], c[5], c[7]);
    let o1 = bm_dot4(BM_OC1, c[1], c[3], c[5], c[7]);
    let o2 = bm_dot4(BM_OC2, c[1], c[3], c[5], c[7]);
    let o3 = bm_dot4(BM_OC3, c[1], c[3], c[5], c[7]);
    return array<f32, 8>(
        bm_op(e0 + o0),
        bm_op(e1 + o1),
        bm_op(e2 + o2),
        bm_op(e3 + o3),
        bm_op(e3 - o3),
        bm_op(e2 - o2),
        bm_op(e1 - o1),
        bm_op(e0 - o0),
    );
}

// The share of row i of the window in the distance of the candidate whose block starts at window index `co`
// (the reference's at `ro`): sum over j of K(i, j) |Y_ref - Y_cand|, left to right.
fn bm_row_distance(ro: u32, co: u32, i: u32) -> f32 {
    var p = bm_op(bm_kwin(i * 8u) * abs(bm_op(bm_win[ro] - bm_win[co])));
    for (var j = 1u; j < 8u; j++) {
        p = bm_op(p + bm_op(bm_kwin(i * 8u + j) * abs(bm_op(bm_win[ro + j] - bm_win[co + j]))));
    }
    return p;
}

// A distance from its 8 row shares at `at`, in row order.
fn bm_distance(at: u32) -> f32 {
    var d = bm_part[at];
    for (var i = 1u; i < 8u; i++) {
        d = bm_op(d + bm_part[at + i]);
    }
    return d;
}

// What a filtered sample adds to its channel's accumulator: K(i, j) v / 512 (`cpu::bm_dct` `scatter`), v the
// inverse transform's output (512 times the filtered sample; the division is exact).
fn bm_term(kk: f32, v: f32) -> f32 {
    return bm_op(bm_op(kk * v) * BM_INV_RT_GAIN);
}

fn bm_load8(base: u32, st: u32) -> array<f32, 8> {
    var x: array<f32, 8>;
    for (var n = 0u; n < 8u; n++) {
        x[n] = bm_grp[base + n * st];
    }
    return x;
}

fn bm_store8(base: u32, st: u32, x: array<f32, 8>) {
    var v = x;
    for (var n = 0u; n < 8u; n++) {
        bm_grp[base + n * st] = v[n];
    }
}

// Across the 8 blocks of the line at `base` (stride 64): the forward transform, the hard threshold
// |X| < tau -> 0 and the inverse, in place.
fn bm_across(base: u32, tau: f32) {
    var y = bm_fwd8(bm_load8(base, 64u));
    for (var q = 0u; q < 8u; q++) {
        if (abs(y[q]) < tau) {
            y[q] = 0.0;
        }
    }
    bm_store8(base, 64u, bm_inv8(y));
}

fn bm_pass(wid: vec3<u32>, lid: u32) {
    let rx = bp.refs.x + bp.refs.z * wid.x;
    let ry = bp.refs.y + bp.refs.w * wid.y;
    let bw = bp.buf.x;
    let n = bp.frame.z;
    // The reference reads only buffer pixels (the host's plan), so it is at or after the buffer's corner.
    let bx = rx - bp.buf.z;
    let by = ry - bp.buf.w;
    let luma = bm_on(1u);
    let chroma = bm_on(2u);
    let first = select(1u, 0u, luma);
    // The lines of one direction of the filtered channels: 64 per channel.
    let lines = (select(0u, 1u, luma) + select(0u, 2u, chroma)) * 64u;

    // A: the luma window, frame pixels only (the others are never read).
    for (var t = lid; t < 576u; t += 256u) {
        let fx = i32(rx) + i32(t % 24u) - 8;
        let fy = i32(ry) + i32(t / 24u) - 8;
        var v = 0.0;
        if (fx >= 0 && fy >= 0 && u32(fx) < bp.frame.x && u32(fy) < bp.frame.y) {
            v = bm_ycc[(u32(fy) - bp.buf.w) * bw + u32(fx) - bp.buf.z];
        }
        bm_win[t] = v;
    }
    workgroupBarrier();

    // B: the row shares of the distances.
    if (lid < 200u) {
        let c = lid / 8u;
        let i = lid % 8u;
        let dx = i32(c % 5u) * 4 - 8;
        let dy = i32(c / 5u) * 4 - 8;
        var p = 0.0;
        if (c != 12u && bm_inside(rx, ry, dx, dy)) {
            let ro = (8u + i) * 24u + 8u;
            let co = u32(8 + dy + i32(i)) * 24u + u32(8 + dx);
            p = bm_row_distance(ro, co, i);
        }
        bm_part[lid] = p;
    }
    workgroupBarrier();

    // C: the distances and which blocks are candidates.
    if (lid < 25u) {
        let d = bm_distance(lid * 8u);
        let dx = i32(lid % 5u) * 4 - 8;
        let dy = i32(lid / 5u) * 4 - 8;
        bm_dist[lid] = d;
        bm_ok[lid] = select(0u, 1u, lid != 12u && bm_inside(rx, ry, dx, dy) && bm_finite(d));
    }
    workgroupBarrier();

    // D: the group (`cpu::bm_dct::select`): an insertion into the 7 best, after every kept entry of a distance
    // at or below the new one, so a tie keeps the smaller index first.
    if (lid == 0u) {
        var bd: array<f32, 7>;
        var bi: array<u32, 7>;
        var m = 0u;
        for (var idx = 0u; idx < 25u; idx++) {
            if (bm_ok[idx] == 0u) {
                continue;
            }
            let di = bm_dist[idx];
            var at = m;
            for (var q = 0u; q < m; q++) {
                if (bd[q] > di) {
                    at = q;
                    break;
                }
            }
            if (at >= 7u) {
                continue;
            }
            var q = min(m, 6u);
            loop {
                if (q <= at) {
                    break;
                }
                bd[q] = bd[q - 1u];
                bi[q] = bi[q - 1u];
                q = q - 1u;
            }
            bd[at] = di;
            bi[at] = idx;
            m = min(m + 1u, 7u);
        }
        bm_sx[0] = bx;
        bm_sy[0] = by;
        for (var q = 0u; q < 7u; q++) {
            var idx = 12u;
            if (q < m && bd[q] <= bp.tau.z) {
                idx = bi[q];
            }
            bm_sx[q + 1u] = u32(i32(bx) + i32(idx % 5u) * 4 - 8);
            bm_sy[q + 1u] = u32(i32(by) + i32(idx / 5u) * 4 - 8);
        }
    }
    workgroupBarrier();

    // E: the group of every filtered channel.
    for (var t = lid; t < 8u * lines; t += 256u) {
        let ch = first + t / 512u;
        let q = t % 512u;
        let k = q / 64u;
        bm_grp[ch * 512u + q] = bm_ycc[ch * n + (bm_sy[k] + (q / 8u) % 8u) * bw + bm_sx[k] + q % 8u];
    }
    workgroupBarrier();

    // F: rows (block a, row b), G: columns (block a, column b).
    if (lid < lines) {
        let ch = first + lid / 64u;
        let base = ch * 512u + ((lid / 8u) % 8u) * 64u + (lid % 8u) * 8u;
        bm_store8(base, 1u, bm_fwd8(bm_load8(base, 1u)));
    }
    workgroupBarrier();
    if (lid < lines) {
        let ch = first + lid / 64u;
        let base = ch * 512u + ((lid / 8u) % 8u) * 64u + lid % 8u;
        bm_store8(base, 8u, bm_fwd8(bm_load8(base, 8u)));
    }
    workgroupBarrier();

    // H: across the blocks at (i, j), the hard threshold and the inverse across.
    if (lid < lines) {
        let ch = first + lid / 64u;
        bm_across(ch * 512u + lid % 64u, select(bp.tau.y, bp.tau.x, ch == 0u));
    }
    workgroupBarrier();

    // I: the inverse along the columns, J: along the rows.
    if (lid < lines) {
        let ch = first + lid / 64u;
        let base = ch * 512u + ((lid / 8u) % 8u) * 64u + lid % 8u;
        bm_store8(base, 8u, bm_inv8(bm_load8(base, 8u)));
    }
    workgroupBarrier();
    if (lid < lines) {
        let ch = first + lid / 64u;
        let base = ch * 512u + ((lid / 8u) % 8u) * 64u + (lid % 8u) * 8u;
        bm_store8(base, 1u, bm_inv8(bm_load8(base, 1u)));
    }
    workgroupBarrier();

    // K: the aggregation, plane `plane` (0 to 2 a channel, 3 the weight) of pixel q of every slot in turn.
    let q = lid % 64u;
    let plane = lid / 64u;
    let on = plane == 3u || (plane == 0u && luma) || (plane != 0u && plane < 3u && chroma);
    let kk = bm_kwin(q);
    let i = q / 8u;
    let j = q % 8u;
    for (var k = 0u; k < 8u; k++) {
        if (on) {
            let at = plane * n + (bm_sy[k] + i) * bw + bm_sx[k] + j;
            var term = kk;
            if (plane < 3u) {
                term = bm_term(kk, bm_grp[plane * 512u + k * 64u + q]);
            }
            bm_acc[at] = bm_op(bm_acc[at] + term);
        }
        storageBarrier();
    }
}
