// R5 highlight reconstruction on the mosaic, opposed-channel method: the GPU port of
// `cpu::highlights` (`docs/05-raw-processing.md` section R5, clean-room from the five-step
// description). Four entry points share this file:
//
//   hl_super          one thread per 2 x 2 superpixel of the input rectangle: the channel values
//                     (red, the mean of the two greens, blue), their cube roots, and the clipped flag
//                     of each channel (3 bits, set when any photosite of the channel in the quad is a
//                     clip-mask bit).
//   hl_apply          one thread per photosite of the core: a clipped photosite takes
//                     max(value, estimate) with the estimate the cube-root mean of the unclipped other
//                     channels over the 3 x 3 superpixels around it, cubed, plus the channel's global
//                     offset; a quad of four clipped photosites is raised to its brightest (step 5).
//   hl_reduce_partial one thread per superpixel of the core rows: the reference terms of the global
//                     chroma offsets (value - estimate, an exact difference as two floats), summed in a
//                     fixed binary tree in thread-index order, one partial per workgroup (D2).
//   hl_fold           one workgroup: the partials folded in index order with compensated summation,
//                     the offsets (and the counts) written.
//
// The CPU's `Superpixels::build` keeps the cube root with `f32::cbrt`; the device forms it correctly
// rounded (`cbrt_f`: a start from `exp2(log2 / 3)` and one Newton correction with the residual formed
// exactly), one ulp from the CPU's libm where that is not correctly rounded. The 3 x 3 sums run in the
// CPU's order (rows, columns, channels), each add and product behind `opaque`, and the mean is a
// correctly rounded quotient (`div_exact`), so a fast-math compiler can neither fuse the cube with the
// offset nor re-associate the sums nor approximate the division.
//
// Bindings: 0 uniform, 1 the mosaic of the input rectangle, 2 the clip bits (`prep.wgsl`), 3 the
// superpixel planes (6 planes of sw * sh: cube roots, then values), 4 the superpixel flags, 5 the
// offsets (3 floats, then 1 pad, then 3 counts as bits), 6 the output core, 7 the partials.

struct HlU {
    zero: vec4<u32>,   // always +0.0 bits
    dims: vec4<u32>,   // input width, height, row stride, 0
    core: vec4<u32>,   // x, y, width, height of the output core in the input
    dst: vec4<u32>,    // output row stride, sw, sh, mask words per row
    cfa: vec4<u32>,    // plane index (0 red, 1 green, 2 blue) at (0,0), (1,0), (0,1), (1,1) of the input
    rows: vec4<u32>,   // reduction: first and one-past-last superpixel row of the core, partial count, first partial of this dispatch
};

@group(0) @binding(0) var<uniform> pu: HlU;
@group(0) @binding(1) var<storage, read> mosaic: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> spc: array<f32>;
@group(0) @binding(4) var<storage, read_write> spf: array<u32>;
@group(0) @binding(5) var<storage, read_write> offs: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<u32>;

fn clipped_bit(x: u32, y: u32) -> u32 {
    return (mask[y * pu.dst.w + (x >> 5u)] >> (x & 31u)) & 1u;
}

fn plane_of(x: u32, y: u32) -> u32 {
    return pu.cfa[(y & 1u) * 2u + (x & 1u)];
}

// The cube root of x, real for negative x, correctly rounded but for a tie within 1e-14: an
// `exp2(log2 / 3)` start (a part in a million), the residual `y^3 - x` formed exactly with two Dekker
// products, and one Newton correction. The CPU's `f32::cbrt` is the platform libm's (glibc's is one
// ulp from the correctly rounded value in one case of ten), so the stage cannot be bit-equal to it;
// being correctly rounded keeps the difference to that single ulp, with no bias.
fn cbrt_f(x: f32) -> f32 {
    let a = min(abs(x), 1.0e12);
    if (!(a > 1.0e-30)) {
        return 0.0;
    }
    let y0 = exp2(log2(a) * 0.33333334);
    let sq = two_prod(y0, y0);
    let cu = two_prod(sq.x, y0);
    let r = opaque(opaque(opaque(cu.x - a) + cu.y) + opaque(sq.y * y0));
    let d = r / opaque(3.0 * sq.x);
    let y = opaque(y0 - d);
    return select(y, -y, x < 0.0);
}

fn sp_count() -> u32 {
    return pu.dst.y * pu.dst.z;
}

fn hl_super(gid: vec3<u32>) {
    if (gid.x >= pu.dst.y || gid.y >= pu.dst.z) {
        return;
    }
    let n = sp_count();
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var cnt = vec3<f32>(0.0, 0.0, 0.0);
    var clip = 0u;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let x = 2u * gid.x + (q & 1u);
        let y = 2u * gid.y + (q >> 1u);
        let c = plane_of(x, y);
        sum[c] = sum[c] + mosaic[y * pu.dims.z + x];
        cnt[c] = cnt[c] + 1.0;
        clip = clip | (clipped_bit(x, y) << c);
    }
    let idx = gid.y * pu.dst.y + gid.x;
    for (var c = 0u; c < 3u; c = c + 1u) {
        // A channel has one or two photosites in a quad: the reciprocal is exact.
        let v = opaque(sum[c] * select(1.0, 0.5, cnt[c] > 1.5));
        spc[c * n + idx] = cbrt_f(v);
        spc[(3u + c) * n + idx] = v;
    }
    spf[idx] = clip;
}

// The opposed estimate for channel c at superpixel (i, j): the cube of the mean of the cube roots of
// the unclipped other channels over the clamped 3 x 3 neighbourhood. Returns (estimate, has).
fn opposed(i: u32, j: u32, c: u32) -> vec2<f32> {
    let n = sp_count();
    let sw = pu.dst.y;
    let sh = pu.dst.z;
    let i0 = max(i, 1u) - 1u;
    let i1 = min(i + 1u, sw - 1u);
    let j0 = max(j, 1u) - 1u;
    let j1 = min(j + 1u, sh - 1u);
    var sum = 0.0;
    var cnt = 0u;
    for (var jj = j0; jj <= j1; jj = jj + 1u) {
        for (var ii = i0; ii <= i1; ii = ii + 1u) {
            let idx = jj * sw + ii;
            let fl = spf[idx];
            for (var k = 0u; k < 3u; k = k + 1u) {
                if (k != c && ((fl >> k) & 1u) == 0u) {
                    sum = opaque(sum + spc[k * n + idx]);
                    cnt = cnt + 1u;
                }
            }
        }
    }
    if (cnt == 0u) {
        return vec2<f32>(0.0, 0.0);
    }
    let mean = div_exact(sum, f32(cnt));
    return vec2<f32>(opaque(opaque(mean * mean) * mean), 1.0);
}

fn hl_apply(gid: vec3<u32>) {
    if (gid.x >= pu.core.z || gid.y >= pu.core.w) {
        return;
    }
    let x = pu.core.x + gid.x;
    let y = pu.core.y + gid.y;
    let v0 = mosaic[y * pu.dims.z + x];
    let o = gid.y * pu.dst.x + gid.x;
    let i = x >> 1u;
    let j = y >> 1u;
    if (i >= pu.dst.y || j >= pu.dst.z || clipped_bit(x, y) == 0u) {
        dst[o] = v0;
        return;
    }
    let c = plane_of(x, y);
    var v = v0;
    let e = opposed(i, j, c);
    if (e.y > 0.0) {
        v = max(v, opaque(e.x + bitcast<f32>(offs[c])));
    }
    // Step 5: a quad whose four photosites are all clipped is levelled to its brightest.
    let x0 = 2u * i;
    let y0 = 2u * j;
    if ((clipped_bit(x0, y0) & clipped_bit(x0 + 1u, y0) & clipped_bit(x0, y0 + 1u)
        & clipped_bit(x0 + 1u, y0 + 1u)) == 1u) {
        let a = mosaic[y0 * pu.dims.z + x0];
        let b = mosaic[y0 * pu.dims.z + x0 + 1u];
        let d = mosaic[(y0 + 1u) * pu.dims.z + x0];
        let f = mosaic[(y0 + 1u) * pu.dims.z + x0 + 1u];
        v = max(v, max(max(a, b), max(d, f)));
    }
    dst[o] = v;
}

// ---- the reduction ------------------------------------------------------------------------------

struct Acc {
    s: vec3<f32>,    // the running sums
    c: vec3<f32>,    // their compensation
    n: vec3<u32>,    // the counts
};

var<workgroup> w_s: array<vec3<f32>, 256>;
var<workgroup> w_c: array<vec3<f32>, 256>;
var<workgroup> w_n: array<vec3<u32>, 256>;

// (a + b) as a rounded sum and its exact error.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    let bb = opaque(s - a);
    let e = opaque(opaque(a - opaque(s - bb)) + opaque(b - bb));
    return vec2<f32>(s, e);
}

// Adds the pair (s2, c2) to (s1, c1) keeping the error.
fn add_pair(s1: f32, c1: f32, s2: f32, c2: f32) -> vec2<f32> {
    let t = two_sum(s1, s2);
    let c = opaque(opaque(c1 + c2) + t.y);
    let h = two_sum(t.x, c);
    return vec2<f32>(h.x, h.y);
}

fn near_flags(i: u32, j: u32) -> u32 {
    let sw = pu.dst.y;
    let sh = pu.dst.z;
    let i0 = max(i, 1u) - 1u;
    let i1 = min(i + 1u, sw - 1u);
    let j0 = max(j, 1u) - 1u;
    let j1 = min(j + 1u, sh - 1u);
    var f = 0u;
    for (var jj = j0; jj <= j1; jj = jj + 1u) {
        for (var ii = i0; ii <= i1; ii = ii + 1u) {
            f = f | spf[jj * sw + ii];
        }
    }
    return f;
}

fn hl_reduce_partial(
    gid: vec3<u32>,
    lid: vec3<u32>,
    wid: vec3<u32>,
    groups: vec3<u32>,
) {
    let t = lid.y * 16u + lid.x;
    var s = vec3<f32>(0.0, 0.0, 0.0);
    var c = vec3<f32>(0.0, 0.0, 0.0);
    var cnt = vec3<u32>(0u, 0u, 0u);
    let i = gid.x;
    let j = pu.rows.x + gid.y;
    if (i < pu.dst.y && j < pu.rows.y && spf[j * pu.dst.y + i] == 0u) {
        let near = near_flags(i, j);
        let n = sp_count();
        let idx = j * pu.dst.y + i;
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            if (((near >> ch) & 1u) == 0u) {
                continue;
            }
            let e = opposed(i, j, ch);
            if (e.y > 0.0) {
                // value - estimate as two floats, exactly.
                let d = two_sum(spc[(3u + ch) * n + idx], -e.x);
                s[ch] = d.x;
                c[ch] = d.y;
                cnt[ch] = 1u;
            }
        }
    }
    w_s[t] = s;
    w_c[t] = c;
    w_n[t] = cnt;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
            var a_s = w_s[t];
            var a_c = w_c[t];
            let b_s = w_s[t + stride];
            let b_c = w_c[t + stride];
            for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                let r = add_pair(a_s[ch], a_c[ch], b_s[ch], b_c[ch]);
                a_s[ch] = r.x;
                a_c[ch] = r.y;
            }
            w_s[t] = a_s;
            w_c[t] = a_c;
            w_n[t] = w_n[t] + w_n[t + stride];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let wg = pu.rows.w + wid.y * groups.x + wid.x;
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            partials[wg * 9u + ch] = bitcast<u32>(w_s[0][ch]);
            partials[wg * 9u + 3u + ch] = bitcast<u32>(w_c[0][ch]);
            partials[wg * 9u + 6u + ch] = w_n[0][ch];
        }
    }
}

// One workgroup of 256 threads folds `pu.rows.z` partials: thread t takes a contiguous chunk, adds it
// in index order, then the 256 results are combined in the same fixed tree.
fn hl_fold(t: u32) {
    let total = pu.rows.z;
    let chunk = (total + 255u) / 256u;
    let lo = min(t * chunk, total);
    let hi = min(lo + chunk, total);
    var s = vec3<f32>(0.0, 0.0, 0.0);
    var c = vec3<f32>(0.0, 0.0, 0.0);
    var cnt = vec3<u32>(0u, 0u, 0u);
    for (var p = lo; p < hi; p = p + 1u) {
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            let r = add_pair(
                s[ch], c[ch],
                bitcast<f32>(partials[p * 9u + ch]), bitcast<f32>(partials[p * 9u + 3u + ch]),
            );
            s[ch] = r.x;
            c[ch] = r.y;
            cnt[ch] = cnt[ch] + partials[p * 9u + 6u + ch];
        }
    }
    w_s[t] = s;
    w_c[t] = c;
    w_n[t] = cnt;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
            var a_s = w_s[t];
            var a_c = w_c[t];
            let b_s = w_s[t + stride];
            let b_c = w_c[t + stride];
            for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                let r = add_pair(a_s[ch], a_c[ch], b_s[ch], b_c[ch]);
                a_s[ch] = r.x;
                a_c[ch] = r.y;
            }
            w_s[t] = a_s;
            w_c[t] = a_c;
            w_n[t] = w_n[t] + w_n[t + stride];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            var mean = 0.0;
            if (w_n[0][ch] > 0u) {
                mean = div_exact(opaque(w_s[0][ch] + w_c[0][ch]), f32(w_n[0][ch]));
            }
            offs[ch] = bitcast<u32>(mean);
            offs[4u + ch] = w_n[0][ch];
        }
        offs[3u] = 0u;
    }
}
