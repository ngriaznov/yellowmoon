// The tetrahedral sampler of a 3D table of f32 vec4 nodes (bake.rs `Table3::sample`, the formula of the look LUTs). @NAME@ and
// @TAB@ are replaced for T3a and T3b'. Needs fx_clamp_nan0 (film.wgsl).


fn rt_@NAME@(size: u32, lo: vec3<f32>, hi: vec3<f32>, power: f32, p: vec3<f32>) -> vec3<f32> {
    var idx = vec3<u32>(0u);
    var fr = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let v = fx_clamp_nan0(p[c], lo[c], hi[c]);
        var u = (v - lo[c]) / (hi[c] - lo[c]);
        if (power == 2.0) {
            u = sqrt(max(u, 0.0));
        } else if (power != 1.0) {
            u = exp(log(max(u, 1.0e-30)) / power);
        }
        let pos = u * f32(size - 1u);
        let i = min(u32(max(pos, 0.0)), size - 2u);
        idx[c] = i;
        fr[c] = pos - f32(i);
    }
    let n = size;
    let i = idx.x;
    let j = idx.y;
    let k = idx.z;
    let fr_r = fr.x;
    let fr_g = fr.y;
    let fr_b = fr.z;
    let c000 = 4u * (i + n * (j + n * k));
    let c111 = 4u * ((i + 1u) + n * ((j + 1u) + n * (k + 1u)));
    var p1 = 0u;
    var p2 = 0u;
    var w = vec3<f32>(0.0);
    if (fr_r > fr_g) {
        if (fr_g > fr_b) {
            p1 = 4u * ((i + 1u) + n * (j + n * k));
            p2 = 4u * ((i + 1u) + n * ((j + 1u) + n * k));
            w = vec3<f32>(fr_r, fr_g, fr_b);
        } else if (fr_r > fr_b) {
            p1 = 4u * ((i + 1u) + n * (j + n * k));
            p2 = 4u * ((i + 1u) + n * (j + n * (k + 1u)));
            w = vec3<f32>(fr_r, fr_b, fr_g);
        } else {
            p1 = 4u * (i + n * (j + n * (k + 1u)));
            p2 = 4u * ((i + 1u) + n * (j + n * (k + 1u)));
            w = vec3<f32>(fr_b, fr_r, fr_g);
        }
    } else if (fr_b > fr_g) {
        p1 = 4u * (i + n * (j + n * (k + 1u)));
        p2 = 4u * (i + n * ((j + 1u) + n * (k + 1u)));
        w = vec3<f32>(fr_b, fr_g, fr_r);
    } else if (fr_b > fr_r) {
        p1 = 4u * (i + n * ((j + 1u) + n * k));
        p2 = 4u * (i + n * ((j + 1u) + n * (k + 1u)));
        w = vec3<f32>(fr_g, fr_b, fr_r);
    } else {
        p1 = 4u * (i + n * ((j + 1u) + n * k));
        p2 = 4u * ((i + 1u) + n * ((j + 1u) + n * k));
        w = vec3<f32>(fr_g, fr_r, fr_b);
    }
    let a = vec3<f32>(@TAB@[c000], @TAB@[c000 + 1u], @TAB@[c000 + 2u]);
    let b = vec3<f32>(@TAB@[p1], @TAB@[p1 + 1u], @TAB@[p1 + 2u]);
    let d = vec3<f32>(@TAB@[p2], @TAB@[p2 + 1u], @TAB@[p2 + 2u]);
    let e = vec3<f32>(@TAB@[c111], @TAB@[c111 + 1u], @TAB@[c111 + 2u]);
    return a + w.x * (b - a) + w.y * (d - b) + w.z * (e - d);
}

// ---- the same lookup, carried as pairs (F-08d, docs/24 section 3.9c) ---------------------------------------------------
//
// The position of a density in the table is `u (n - 1)` with `u` the warped, scaled density: in f32 it has an error of an
// ulp of a number of up to n - 1, a few 1e-6 of a cell, and a cell of T3b' changes the display colour by several percent.
// The input is a pair (hi + lo), the axis ends and 1 / (hi - lo) are pairs, the square root of the warp is corrected by
// its exact remainder, and the interpolation sums the (exact) differences of the nodes with exact products, so the result is
// a pair too. The decision of the tetrahedron compares the fractions (continuous across the ties) and the cell of the
// position is the integer part of its hi part with the lo part moved into the fraction.

fn rt_@NAME@_pos(v_hi: f32, v_lo: f32, size: u32, lo_h: f32, lo_l: f32, r_h: f32, r_l: f32, lo_edge: f32, hi_edge: f32, power: f32) -> vec2<f32> {
    let fs = f32(size - 1u);
    if (is_nan_f(v_hi) || v_hi <= lo_edge) {
        return vec2<f32>(0.0, 0.0);
    }
    if (v_hi >= hi_edge) {
        return vec2<f32>(fs, 0.0);
    }
    let d = px_two_sum(v_hi, -lo_h);
    let dl = px_op(px_op(d.y + v_lo) - lo_l);
    let m = px_two_prod(d.x, r_h);
    let u0h = m.x;
    let u0l = px_op(px_op(m.y + px_op(d.x * r_l)) + px_op(dl * r_h));
    var uh = u0h;
    var ul = u0l;
    if (power == 2.0) {
        let s = sqrt(max(u0h, 0.0));
        let sq = px_two_prod(s, s);
        let resid = px_op(px_op(px_op(u0h - sq.x) - sq.y) + u0l);
        uh = s;
        ul = select(0.0, px_op(resid / px_op(2.0 * s)), s > 0.0);
    } else if (power != 1.0) {
        uh = exp(log(max(u0h, 1.0e-30)) / power);
        ul = 0.0;
    }
    let pp = px_two_prod(uh, fs);
    return vec2<f32>(pp.x, px_op(pp.y + px_op(ul * fs)));
}

// a + w0 (b - a) + w1 (d - b) + w2 (e - d) of one channel as a pair.
fn rt_@NAME@_tet(a: f32, b: f32, d: f32, e: f32, w0: f32, w1: f32, w2: f32) -> vec2<f32> {
    var hi = a;
    var lo = 0.0;
    for (var k = 0u; k < 3u; k++) {
        var w = w0;
        var x = b;
        var y = a;
        if (k == 1u) {
            w = w1;
            x = d;
            y = b;
        } else if (k == 2u) {
            w = w2;
            x = e;
            y = d;
        }
        let df = px_two_sum(x, -y);
        let pr = px_two_prod(w, df.x);
        let s = px_two_sum(hi, pr.x);
        hi = s.x;
        lo = px_op(px_op(lo + s.y) + px_op(pr.y + px_op(w * df.y)));
    }
    let r = px_two_sum(hi, lo);
    return r;
}

struct @NAME@Pair {
    hi: vec3<f32>,
    lo: vec3<f32>,
}

fn rt_@NAME@_pair(size: u32, lo_h: vec3<f32>, lo_l: vec3<f32>, r_h: vec3<f32>, r_l: vec3<f32>, lo_edge: vec3<f32>, hi_edge: vec3<f32>, power: f32, ph: vec3<f32>, pl: vec3<f32>) -> @NAME@Pair {
    var idx = vec3<u32>(0u);
    var fr = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        let pos = rt_@NAME@_pos(ph[c], pl[c], size, lo_h[c], lo_l[c], r_h[c], r_l[c], lo_edge[c], hi_edge[c], power);
        var i = min(u32(max(floor(pos.x), 0.0)), size - 2u);
        var t = px_op(px_op(pos.x - f32(i)) + pos.y);
        if (t < 0.0 && i > 0u) {
            i = i - 1u;
            t = px_op(t + 1.0);
        }
        idx[c] = i;
        fr[c] = t;
    }
    let n = size;
    let i = idx.x;
    let j = idx.y;
    let k = idx.z;
    let fr_r = fr.x;
    let fr_g = fr.y;
    let fr_b = fr.z;
    let c000 = 4u * (i + n * (j + n * k));
    let c111 = 4u * ((i + 1u) + n * ((j + 1u) + n * (k + 1u)));
    var p1 = 0u;
    var p2 = 0u;
    var w = vec3<f32>(0.0);
    if (fr_r > fr_g) {
        if (fr_g > fr_b) {
            p1 = 4u * ((i + 1u) + n * (j + n * k));
            p2 = 4u * ((i + 1u) + n * ((j + 1u) + n * k));
            w = vec3<f32>(fr_r, fr_g, fr_b);
        } else if (fr_r > fr_b) {
            p1 = 4u * ((i + 1u) + n * (j + n * k));
            p2 = 4u * ((i + 1u) + n * (j + n * (k + 1u)));
            w = vec3<f32>(fr_r, fr_b, fr_g);
        } else {
            p1 = 4u * (i + n * (j + n * (k + 1u)));
            p2 = 4u * ((i + 1u) + n * (j + n * (k + 1u)));
            w = vec3<f32>(fr_b, fr_r, fr_g);
        }
    } else if (fr_b > fr_g) {
        p1 = 4u * (i + n * (j + n * (k + 1u)));
        p2 = 4u * (i + n * ((j + 1u) + n * (k + 1u)));
        w = vec3<f32>(fr_b, fr_g, fr_r);
    } else if (fr_b > fr_r) {
        p1 = 4u * (i + n * ((j + 1u) + n * k));
        p2 = 4u * (i + n * ((j + 1u) + n * (k + 1u)));
        w = vec3<f32>(fr_g, fr_b, fr_r);
    } else {
        p1 = 4u * (i + n * ((j + 1u) + n * k));
        p2 = 4u * ((i + 1u) + n * ((j + 1u) + n * k));
        w = vec3<f32>(fr_g, fr_r, fr_b);
    }
    var out: @NAME@Pair;
    for (var c = 0u; c < 3u; c++) {
        let r = rt_@NAME@_tet(@TAB@[c000 + c], @TAB@[p1 + c], @TAB@[p2 + c], @TAB@[c111 + c], w.x, w.y, w.z);
        out.hi[c] = r.x;
        out.lo[c] = r.y;
    }
    return out;
}
