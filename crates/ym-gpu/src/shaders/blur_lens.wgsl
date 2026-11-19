// The Lens kernel of a Blur node (docs/33-blur.md section 3.5, BL-07b; `kernels/blur_lens.rs`): the coverage-weighted aperture over
// the pixels, three modes of one entry point (`bu.op.x`), all over the weighted planes of the pyramid's level 0 (four planes:
// `omega X` and `omega`, image A, the range `bu.c` of the read area).
//
// Mode 0, the exact table (R <= 8 px): every pixel of the made window adds the table's nonzero taps (dx, dy, k, 0), storage
// `rows`, in the CPU's order (rows from the top, columns from the left), a tap outside the read area skipped.
//
// Mode 1, the scan: one thread per row of the read area (lane 0 of each group row) makes the row's prefix sums in integer
// fixed point, `2^-32`, 64 bits in two words (`phi`, `plo`, a vec4 of the four planes per column, W + 2 columns a row, the
// exclusive prefix at column i, the last two the row's total) and the raw values `pv` (zeros past the row). Integer sums are
// exact in any order, so the prefix of a window of any origin differs from the frame's by a constant and the differences the
// kernel reads are the same bits (a tile equals the whole frame).
//
// Mode 2, the rows: every pixel of the made window adds the sub-rows of the row table (storage `rows`, two vec4 each:
// (ka, kb, fa, fb) and (dy, w, 0, 0); `bu.op.y` of them), in table order: `w * ((whole + vb ub) - va ua)`, `whole` the exact
// integer difference of the prefix at the end pixels as a float, `va` and `vb` those pixels' own values, `ua` and `ub` the
// fractions of them the chord covers. A pixel outside the read area has no weight.
//
// Bindings: 0 uniform BU, 1 the source planes, 2 the result (four planes, image B over the made window), 3 `rows`, 4 and 5 the
// prefix words, 6 the raw values.
//
// Safe under Metal's fast-math (docs/17 K8): the integer arithmetic is exact; the float tests are the NaN bits and clamps; the
// scalings are by powers of two.

@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> phi: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> plo: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> pv: array<vec4<f32>>;

const TWO32: f32 = 4294967296.0;
const UNIT: f32 = 2.3283064365386963e-10;
const FIXED_MAX: f32 = 16384.0;

// A value as the prefix's integer, round(v * 2^32) clamped to +-16384 first (NaN is 0): (lo, hi) of the two's complement.
fn quant(v: f32) -> vec2<u32> {
    var a = v;
    if (is_nan_f(a)) {
        a = 0.0;
    }
    a = clamp(a, -FIXED_MAX, FIXED_MAX);
    let m = round(abs(a) * TWO32);
    let hi = floor(m / TWO32);
    let lo = m - hi * TWO32;
    var r = vec2<u32>(u32(lo), u32(hi));
    if (a < 0.0) {
        let nlo = ~r.x + 1u;
        let carry = select(0u, 1u, nlo == 0u);
        r = vec2<u32>(nlo, ~r.y + carry);
    }
    return r;
}

// A prefix difference (hi, lo of a minus b, four planes) as a float times 2^-32, the magnitude first so the conversion has no
// cancellation.
fn diff_to_f32(ahi: vec4<u32>, alo: vec4<u32>, bhi: vec4<u32>, blo: vec4<u32>) -> vec4<f32> {
    var lo = alo - blo;
    var hi = ahi - bhi - select(vec4<u32>(0u), vec4<u32>(1u), alo < blo);
    let neg = (hi >> vec4<u32>(31u)) != vec4<u32>(0u);
    let nlo = ~lo + vec4<u32>(1u);
    let nhi = ~hi + select(vec4<u32>(0u), vec4<u32>(1u), nlo == vec4<u32>(0u));
    lo = select(lo, nlo, neg);
    hi = select(hi, nhi, neg);
    let f = (vec4<f32>(hi) * TWO32 + vec4<f32>(lo)) * UNIT;
    return select(f, -f, neg);
}

fn load4(i: i32, j: i32) -> vec4<f32> {
    return vec4<f32>(
        src[bu_at(bu.a, i, j, 0u)],
        src[bu_at(bu.a, i, j, 1u)],
        src[bu_at(bu.a, i, j, 2u)],
        src[bu_at(bu.a, i, j, 3u)],
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mode = bu.op.x;
    if (mode == 1u) {
        // The scan: lane 0 of each row of lanes.
        let r = i32(gid.y);
        if (gid.x != 0u || r >= bu.c.w) {
            return;
        }
        let w = bu.c.z;
        let base = u32(r) * u32(w + 2);
        var ahi = vec4<u32>(0u);
        var alo = vec4<u32>(0u);
        for (var i = 0; i < w; i = i + 1) {
            let v = load4(bu.c.x + i, bu.c.y + r);
            phi[base + u32(i)] = ahi;
            plo[base + u32(i)] = alo;
            pv[base + u32(i)] = v;
            let q0 = quant(v.x);
            let q1 = quant(v.y);
            let q2 = quant(v.z);
            let q3 = quant(v.w);
            let qlo = vec4<u32>(q0.x, q1.x, q2.x, q3.x);
            let qhi = vec4<u32>(q0.y, q1.y, q2.y, q3.y);
            let nlo = alo + qlo;
            let carry = select(vec4<u32>(0u), vec4<u32>(1u), nlo < alo);
            ahi = ahi + qhi + carry;
            alo = nlo;
        }
        for (var t = 0; t < 2; t = t + 1) {
            phi[base + u32(w + t)] = ahi;
            plo[base + u32(w + t)] = alo;
            pv[base + u32(w + t)] = vec4<f32>(0.0);
        }
        return;
    }
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let n = bu.op.y;
    var acc = vec4<f32>(0.0);
    if (mode == 0u) {
        for (var t = 0u; t < n; t = t + 1u) {
            let tap = rows[t];
            let i = x + i32(tap.x);
            let j = y + i32(tap.y);
            if (i < bu.c.x || j < bu.c.y || i >= bu.c.x + bu.c.z || j >= bu.c.y + bu.c.w) {
                continue;
            }
            acc = acc + tap.z * load4(i, j);
        }
    } else {
        let w = bu.c.z;
        let h = bu.c.w;
        let xr = x - bu.c.x;
        let yr = y - bu.c.y;
        for (var t = 0u; t < n; t = t + 1u) {
            let e0 = rows[2u * t];
            let e1 = rows[2u * t + 1u];
            let yy = yr + i32(e1.x);
            if (yy < 0 || yy >= h) {
                continue;
            }
            var ia = xr + i32(e0.x);
            var ua = e0.z;
            if (ia < 0) {
                ia = 0;
                ua = 0.0;
            } else if (ia >= w) {
                ia = w;
                ua = 0.0;
            }
            var ib = xr + i32(e0.y);
            var ub = e0.w;
            if (ib < 0) {
                ib = 0;
                ub = 0.0;
            } else if (ib >= w) {
                ib = w;
                ub = 0.0;
            }
            let base = u32(yy) * u32(w + 2);
            let ka = base + u32(ia);
            let kb = base + u32(ib);
            let whole = diff_to_f32(phi[kb], plo[kb], phi[ka], plo[ka]);
            acc = acc + e1.y * ((whole + pv[kb] * ub) - pv[ka] * ua);
        }
    }
    dst[bu_at(bu.b, x, y, 0u)] = acc.x;
    dst[bu_at(bu.b, x, y, 1u)] = acc.y;
    dst[bu_at(bu.b, x, y, 2u)] = acc.z;
    dst[bu_at(bu.b, x, y, 3u)] = acc.w;
}
