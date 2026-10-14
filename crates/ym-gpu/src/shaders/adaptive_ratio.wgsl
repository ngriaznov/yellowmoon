// RA-09: the adaptive ratio demosaic on the GPU (docs/algorithms/01-adaptive-ratio-demosaic.md; CPU
// reference: ym_gpu::cpu::demosaic::adaptive_ratio). The three passes of the spec (green, green
// refinement with the optional CA weighting, chroma with the shadow sharpening and the shadow colour
// matching) are cut into 11 dispatches, one pipeline each (see `kernels/adaptive_ratio.rs`, which
// appends the `main` of each pass to this text). Every pass is a plain function here so that naga
// validates all of them in every generated module.
//
// Layout. All planes are f32 storage in every storage mode of the device (the decisions below compare
// sums of absolute differences for order, which f16 planes would destroy). The planes live in one
// scratch buffer of `dims.z`-sample planes (`PL_*`), all of the PADDED size `dims.x * dims.y`: the core
// rectangle plus the margin of the receptive field on every side (spec 3.1: the padding is made once
// at the entry, mirrored at the buffer edge without repeating the edge sample, with the left and top
// pad chosen so that red sits at (even, even) of the plane: every pass below assumes RGGB). A pass
// evaluates only the rectangle `region` of the plane that the core depends on (the CPU port's
// `Margins`), so every read is of a value an earlier pass wrote.
//
// Precision. WGSL has no f64, so every formula that the CPU runs in f64 runs here in f32, in the
// operation order of the spec; the planes are stored as f32 exactly where the CPU stores them. The
// decisions (pass-1 direction, pass-2 argmin, step-1 diagonal, step-3 diagonal) are therefore exact
// only up to ties of relative size 1e-7; the tests count how many flip. Two forms differ from the
// CPU's text, both exact in real arithmetic and chosen so that nothing overflows or loses the small
// terms:
//   * pass 1 and step 1 weigh the samples by the INVERSE costs `1 / w_j` (the spec's section 5
//     recommendation) instead of the products of the other costs (a product of three costs reaches
//     1e18 for 1e6 DN data; the inverse form stays below the largest sample);
//   * the CA field is evaluated per site in f32 from the f32 knots (spec 5), not in f64.
// No fused multiply-add is used on purpose.
//
// Fast-math (K8). Metal compiles with fast-math: it fuses `a * b + c` into multiply-adds, re-associates
// sums and products, and its division and square root are not correctly rounded. The decisions above
// compare sums of absolute differences for order, and a tie in exact arithmetic (an edge at 2:1, a
// 45 degree edge, a symmetric neighbourhood) is then decided by which rounding each side got; every
// later value is computed from values a decision chose. So every product that meets a sum, every
// partial sum and every difference passes through `ar_op` (one operation per barrier: nothing to
// fuse or re-associate), and every division and the one square root are computed in integer
// arithmetic, correctly rounded (`ar_div`, `ar_sqrt`). IEEE arithmetic is unchanged by all of that;
// it also stops the re-association of constants that lavapipe's compiler (NIR) did in a few chains
// (a few samples moved by 1 to 3 ulp). A model of Apple's compiler (naga's MSL of these passes
// compiled with clang -O2 -ffast-math -ffp-contract=fast -mfma -mrecip=all) gives the bits of the
// strict compile on every case it was run on (docs/gpu-plan.md section 12).
//
// Inputs. The host refuses a mosaic with NaN, infinity or a value whose scaled magnitude would
// overflow the squares of the matching step (see `kernels/adaptive_ratio.rs`), so the shader has no
// guard for them beyond the NaN test of the CA field.

struct ArParams {
    dims: vec4<u32>,    // x: padded width = row stride of every plane, y: padded height, z: plane length (x * y), w: width of the input rectangle (the mirror period of a row)
    geom: vec4<u32>,    // x: plane column of core column 0, y: plane row of core row 0, z: core width, w: core height
    region: vec4<u32>,  // x0, y0, x1, y1 of this dispatch in plane coordinates (half open)
    ctl: vec4<u32>,     // x: 1 when the input is balanced (divide by the gains), y: channel (matching), z: 1 with the CA path, w: row stride of the output planes
    org: vec4<i32>,     // x, y: (input column, row) of plane column 0, row 0 (the core corner minus the pad); z: height of the input rectangle; w: row stride of the input
    gains: vec4<f32>,   // g_R, g_G, g_B and the working scale S
    cst: vec4<f32>,     // rho_R, rho_B, K (pass-1 offset), k (pass-3 offset)
    cst2: vec4<f32>,    // T (shadow gate), r (noise sigma in DN)
    mult: vec4<f32>,    // matching: (4 g_R, 2 g_G, 4 g_B)
    vg: vec4<f32>,      // matching: g * g_c, the shot-noise variance per DN of the balanced channel c
    vr2: vec4<f32>,     // matching: (r g_c)^2
    cai: vec4<i32>,     // CA: frame column of plane column 0 minus floor(centre_x), the same for rows, number of red knots, of blue knots
    caf: vec4<f32>,     // CA: 0.5 - fract(centre_x), 0.5 - fract(centre_y)
};

@group(0) @binding(0) var<uniform> u: ArParams;
@group(0) @binding(1) var<storage, read> inp: array<f32>;            // the input mosaic of the compute rectangle, row stride `org.w` (>= its width: a device image has an even stride)
@group(0) @binding(2) var<storage, read_write> sc: array<f32>;       // the planes below
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;     // 3 planes of the core, row stride `ctl.w`
@group(0) @binding(4) var<storage, read> curves: array<vec2<f32>>;   // CA knots (radius, shift): red, then blue

// Scratch planes.
const PL_M: u32 = 0u;       // M = S max(m_u, 0), padded (spec 3.1)
const PL_A: u32 = 1u;       // pass-1 green G1, later the luma proxy of the shadow sharpening
const PL_R: u32 = 2u;       // the white-balanced planes of pass 3
const PL_G: u32 = 3u;
const PL_B: u32 = 4u;
const PL_SIMPLE: u32 = 5u;  // matching: Gauss3(sparse_c) m_c of the channel in hand
const PL_LOW: u32 = 6u;     // matching: L_c, 3 planes
const PL_ERR: u32 = 9u;     // matching: (S3_c - L_c)^2, 3 planes

// ---------------------------------------------------------------------------------------------
// Plane access
// ---------------------------------------------------------------------------------------------

// The colour of plane position (x, y): 0 red at (even, even), 2 blue at (odd, odd), 1 green.
fn ar_site(x: u32, y: u32) -> u32 {
    let a = x & 1u;
    let b = y & 1u;
    if (a == 0u && b == 0u) {
        return 0u;
    }
    if (a == 1u && b == 1u) {
        return 2u;
    }
    return 1u;
}

// Sample `i` of plane `pl`.
fn ar_ld(pl: u32, i: u32) -> f32 {
    return sc[pl * u.dims.z + i];
}

fn ar_st(pl: u32, i: u32, v: f32) {
    sc[pl * u.dims.z + i] = v;
}

// Hides a value from algebraic rewriting: an OR with the bits of `u.cst2.z`, which is always +0.0 and
// which the compiler cannot know (module notes, K8). The identity on the bits. OR, not XOR: two
// barriers in a row must stay a barrier. An XOR applied twice cancels (x ^ z ^ z == x for every z,
// and compilers fold it), and fast-math folds `0 + x` to x, so in the running sums of step 2 below
// (which start at 0) the first product was left bare and fused into the next sum (clang's code for
// this file, -ffast-math -mfma: one ulp in 6 % of the step-2 values). An OR is idempotent.
fn ar_op(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | bitcast<u32>(u.cst2.z));
}

// The correctly rounded quotient a / b, whatever the device's division does (a fast-math division is
// the dividend times an approximate reciprocal). The device's quotient of the two 24-bit mantissas
// gives an estimate q of the quotient's mantissa, a few units off at most; the remainder
// ma 2^k - q mb is then small (under 2^28), so the low 32 bits of the two products give it exactly in
// wrapping u32 arithmetic, and up to four steps move q to the nearest value (|2 r| < mb; a quotient of
// two 24-bit mantissas never lies exactly halfway, so there is no tie). A zero dividend gives the
// signed zero; subnormal, infinite or NaN operands and a quotient outside the normal range take the
// device's division. Measured against IEEE on 3.2 10^8 quotients, also compiled with clang
// -ffast-math -mrecip=all (an approximate reciprocal for the estimate): 0 differ.
fn ar_div(a: f32, b: f32) -> f32 {
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let ea = (ab >> 23u) & 0xffu;
    let eb = (bb >> 23u) & 0xffu;
    let sign = (ab ^ bb) & 0x80000000u;
    if (eb == 0u || eb == 0xffu || ea == 0xffu) {
        return a / b;
    }
    if (ea == 0u) {
        if ((ab & 0x7fffffffu) == 0u) {
            return bitcast<f32>(sign);
        }
        return a / b;
    }
    let ma = (ab & 0x7fffffu) | 0x800000u;
    let mb = (bb & 0x7fffffu) | 0x800000u;
    let big = ma >= mb;
    let k = select(24u, 23u, big);
    let ex = i32(ea) - i32(eb) + select(126, 127, big);
    if (ex < 1 || ex > 254) {
        return a / b;
    }
    let qf = f32(ma) / f32(mb);
    var q = u32(qf * bitcast<f32>((127u + k) << 23u) + 0.5);
    var r = bitcast<i32>((ma << k) - q * mb);
    let m = i32(mb);
    for (var i = 0; i < 4; i = i + 1) {
        if (2 * r > m) {
            q = q + 1u;
            r = r - m;
        } else if (2 * r < -m) {
            q = q - 1u;
            r = r + m;
        }
    }
    // q holds the leading bit, so a quotient that rounds up to the next power of two carries into the
    // exponent.
    return bitcast<f32>(sign | ((u32(ex - 1) << 23u) + q));
}

// The correctly rounded square root in integer arithmetic only: the integer root of N = mp 2^26 digit
// by digit (mp the mantissa, doubled for an odd exponent), rounding to nearest, ties to even. Zero
// returns itself; subnormals, infinities, NaN and negatives take the device's `sqrt`.
fn ar_sqrt(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    if ((bits & 0x7fffffffu) == 0u) {
        return x;
    }
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u || e == 0xffu || (bits >> 31u) != 0u) {
        return sqrt(x);
    }
    let odd = e & 1u;
    let mp = ((bits & 0x7fffffu) | 0x800000u) << odd;
    var rem = 0u;
    var root = 0u;
    for (var k = 0u; k < 26u; k = k + 1u) {
        var two = 0u;
        if (k < 13u) {
            two = (mp >> (24u - 2u * k)) & 3u;
        }
        rem = (rem << 2u) | two;
        let trial = (root << 2u) | 1u;
        if (rem >= trial) {
            rem = rem - trial;
            root = (root << 1u) | 1u;
        } else {
            root = root << 1u;
        }
    }
    var q = root >> 1u;
    var round_bit = root & 1u;
    var sticky = u32(rem != 0u);
    if (odd == 1u) {
        q = root >> 2u;
        round_bit = (root >> 1u) & 1u;
        sticky = sticky | (root & 1u);
    }
    q = q + (round_bit & (sticky | (q & 1u)));
    return bitcast<f32>(((((e + 127u) >> 1u) - 1u) << 23u) + q);
}

// The sample (dx, dy) away from flat index `b` of plane `pl`.
fn ar_at(pl: u32, b: i32, dx: i32, dy: i32) -> f32 {
    return sc[pl * u.dims.z + u32(b + dy * i32(u.dims.x) + dx)];
}

// The same for the mosaic plane M.
fn ar_m(b: i32, dx: i32, dy: i32) -> f32 {
    return sc[u32(b + dy * i32(u.dims.x) + dx)];
}

// Whether this thread lies inside the dispatch region.
fn ar_in_region(gid: vec3<u32>) -> bool {
    return u.region.x + gid.x < u.region.z && u.region.y + gid.y < u.region.w;
}

// ---------------------------------------------------------------------------------------------
// Pass 0: the padded, scaled and clamped mosaic (spec 3.1), and the native samples with the gains
// ---------------------------------------------------------------------------------------------

// M(px, py) = S max(m_u(mirror), 0) with m_u = m / g of the site's colour when the input is balanced
// (R6's entry, the exact inverse of R4, spec 1.3). The native red and blue planes of pass 3 (spec
// 3.4.1: R = g_R M at R sites, B = g_B M at B sites) are written here too, over the whole plane.
fn ar_prepare(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let sx = mirror_idx(i32(x) + u.org.x, i32(u.dims.w));
    let sy = mirror_idx(i32(y) + u.org.y, u.org.z);
    let c = ar_site(x, y);
    var mu = inp[u32(sy) * u32(u.org.w) + u32(sx)];
    if (u.ctl.x == 1u) {
        mu = ar_div(mu, u.gains[c]);
    }
    let m = ar_op(u.gains.w * max(mu, 0.0));
    let i = y * u.dims.x + x;
    ar_st(PL_M, i, m);
    if (c == 0u) {
        ar_st(PL_R, i, m * u.gains.x);
    } else if (c == 2u) {
        ar_st(PL_B, i, m * u.gains.z);
    }
}

// ---------------------------------------------------------------------------------------------
// The CA field (spec 3.6)
// ---------------------------------------------------------------------------------------------

// shift(r) of the curve whose n knots start at `off`: linear between knots, the end segments extended
// linearly (RadialCurve::shift_at). The knots are f32: two radii that round to the same value give a
// zero-length segment, whose t is taken as 0.
fn ar_curve(off: u32, n: u32, r: f32) -> f32 {
    var j = 0u;
    if (r >= curves[off + n - 1u].x) {
        j = n - 2u;
    } else if (r > curves[off].x) {
        var lo = 0u;
        var hi = n - 1u;
        while (hi - lo > 1u) {
            let mid = (lo + hi) / 2u;
            if (curves[off + mid].x <= r) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        j = lo;
    }
    let a = curves[off + j];
    let b = curves[off + j + 1u];
    let span = b.x - a.x;
    var t = 0.0;
    if (span > 0.0) {
        t = ar_div(ar_op(r - a.x), span);
    }
    return a.y + ar_op(ar_op(b.y - a.y) * t);
}

// ca = |shift_R(r) - shift_B(r)| in pixels at the site (x, y) of the plane (0 on the plain path): r is
// the distance of the site's pixel centre from the optical centre, from the site's own frame position
// (a padded position outside the compute rectangle uses its geometric position, as the CPU does).
fn ar_misreg(x: u32, y: u32) -> f32 {
    if (u.ctl.z == 0u) {
        return 0.0;
    }
    let dx = f32(u.cai.x + i32(x)) + u.caf.x;
    let dy = f32(u.cai.y + i32(y)) + u.caf.y;
    let r = ar_sqrt(ar_op(ar_op(dx * dx) + ar_op(dy * dy)));
    let nr = u32(u.cai.z);
    let nb = u32(u.cai.w);
    return abs(ar_curve(0u, nr, r) - ar_curve(nr, nb, r));
}

// ---------------------------------------------------------------------------------------------
// Pass 1: green (spec 3.2)
// ---------------------------------------------------------------------------------------------

// The 5 x 5 fallback of the direction decision: 0 horizontal (type 1), 1 vertical (type 2), 2 all four.
fn ar_wide(b: i32, rho: f32, dl: f32, dr: f32, du: f32, dd: f32) -> u32 {
    var l5 = 0.0;
    var r5 = 0.0;
    var u5 = 0.0;
    var d5 = 0.0;
    for (var j = -4; j <= 4; j = j + 2) {
        for (var i = -4; i <= 4; i = i + 2) {
            let q = ar_op(rho * ar_m(b, i, j));
            l5 = ar_op(l5 + abs(ar_op(ar_op(dl + ar_m(b, i - 1, j)) - q)));
            r5 = ar_op(r5 + abs(ar_op(ar_op(dr + ar_m(b, i + 1, j)) - q)));
            u5 = ar_op(u5 + abs(ar_op(ar_op(du + ar_m(b, i, j - 1)) - q)));
            d5 = ar_op(d5 + abs(ar_op(ar_op(dd + ar_m(b, i, j + 1)) - q)));
        }
    }
    if (max(l5, r5) < min(u5, d5)) {
        return 0u;
    }
    if (min(l5, r5) > max(u5, d5)) {
        return 1u;
    }
    return 2u;
}

// G1 at the R or B site with flat index `b`: `rho` = g_c / g_G of the site's colour, `k` = K, `ca` the
// R/B misregistration at the site.
fn ar_green_estimate(b: i32, rho: f32, k: f32, ca: f32) -> f32 {
    let c = ar_m(b, 0, 0);
    let cs = ar_op(rho * c);
    let gu = ar_m(b, 0, -1);
    let gd = ar_m(b, 0, 1);
    let gl = ar_m(b, -1, 0);
    let gr = ar_m(b, 1, 0);
    // Centre colour differences.
    let du = ar_op(cs - gu);
    let dd = ar_op(cs - gd);
    let dl = ar_op(cs - gl);
    let dr = ar_op(cs - gr);
    let w2 = ar_op(1.6 * rho);
    let c2 = ar_op(2.0 * c);
    let vbase = ar_op(ar_op(ar_op(ar_op(abs(gl - ar_m(b, -1, -2)) + abs(gl - ar_m(b, -1, 2)))
        + abs(gr - ar_m(b, 1, -2))) + abs(gr - ar_m(b, 1, 2)))
        + ar_op(w2 * abs(ar_op(ar_op(ar_m(b, 0, -2) - c2) + ar_m(b, 0, 2)))));
    let hbase = ar_op(ar_op(ar_op(ar_op(abs(gu - ar_m(b, -2, -1)) + abs(gu - ar_m(b, 2, -1)))
        + abs(gd - ar_m(b, -2, 1))) + abs(gd - ar_m(b, 2, 1)))
        + ar_op(w2 * abs(ar_op(ar_op(ar_m(b, -2, 0) - c2) + ar_m(b, 2, 0)))));
    let q_n = ar_op(rho * ar_m(b, 0, -4));
    let q_s = ar_op(rho * ar_m(b, 0, 4));
    let q_w = ar_op(rho * ar_m(b, -4, 0));
    let q_e = ar_op(rho * ar_m(b, 4, 0));
    var su = ar_op(ar_op(vbase + abs(ar_op(ar_op(du + ar_m(b, 0, -5)) - q_n)))
        + abs(ar_op(ar_op(du + ar_m(b, 0, 3)) - q_s)));
    var sd = ar_op(ar_op(vbase + abs(ar_op(ar_op(dd + ar_m(b, 0, -3)) - q_n)))
        + abs(ar_op(ar_op(dd + ar_m(b, 0, 5)) - q_s)));
    var sv = ar_op(ar_op(vbase + abs(ar_m(b, 0, -5) - ar_m(b, 0, -3)))
        + abs(ar_m(b, 0, 3) - ar_m(b, 0, 5)));
    var sl = ar_op(ar_op(hbase + abs(ar_op(ar_op(dl + ar_m(b, -5, 0)) - q_w)))
        + abs(ar_op(ar_op(dl + ar_m(b, 3, 0)) - q_e)));
    var sr = ar_op(ar_op(hbase + abs(ar_op(ar_op(dr + ar_m(b, -3, 0)) - q_w)))
        + abs(ar_op(ar_op(dr + ar_m(b, 5, 0)) - q_e)));
    var sh = ar_op(ar_op(hbase + abs(ar_m(b, -5, 0) - ar_m(b, -3, 0)))
        + abs(ar_m(b, 3, 0) - ar_m(b, 5, 0)));
    // The 3 x 3 same-colour grid, centre included.
    for (var j = -2; j <= 2; j = j + 2) {
        for (var i = -2; i <= 2; i = i + 2) {
            let q = ar_op(rho * ar_m(b, i, j));
            let ql = ar_m(b, i - 1, j);
            let qr = ar_m(b, i + 1, j);
            let qu = ar_m(b, i, j - 1);
            let qd = ar_m(b, i, j + 1);
            sl = ar_op(sl + abs(ar_op(ar_op(dl + ql) - q)));
            sr = ar_op(sr + abs(ar_op(ar_op(dr + qr) - q)));
            su = ar_op(su + abs(ar_op(ar_op(du + qu) - q)));
            sd = ar_op(sd + abs(ar_op(ar_op(dd + qd) - q)));
            sh = ar_op(sh + abs(ql - qr));
            sv = ar_op(sv + abs(qu - qd));
        }
    }
    // The CA relaxation factor: exactly 1 on the plain path and under 0.1 px, unbounded above. A NaN
    // misregistration (curves that overflow) relaxes nothing, as on the CPU.
    var f = 1.0;
    if (!is_nan_f(ca) && ca >= 0.1) {
        f = ar_op(1.0 + ar_op(2.5 * ar_op(ar_op(ca * ca) * ca)));
    }
    var dir = 2u;
    if (max(sl, sr) < ar_op(f * min(su, sd)) && sh < sv) {
        dir = 0u;
    } else if (ar_op(f * min(sl, sr)) > max(su, sd) && sv < sh) {
        dir = 1u;
    } else {
        // Flat test: the two green-pair means agree within about two sigma of unit-gain Poisson noise.
        let s_h = ar_op(gl + gr);
        let s_v = ar_op(gu + gd);
        let half_diff = ar_op(0.5 * abs(s_h - s_v));
        if (ar_op(half_diff * half_diff) < ar_op(s_h + s_v)) {
            dir = 2u;
        } else {
            dir = ar_wide(b, rho, dl, dr, du, dd);
        }
    }
    // Costs of the four directions (always >= 1).
    let w_u = ar_op(ar_op(ar_op(ar_op(ar_op(ar_op(rho * abs(ar_m(b, 0, -2) - c)) + abs(ar_m(b, 0, -3) - gu))
        + abs(ar_m(b, -1, -2) - gl)) + abs(ar_m(b, 1, -2) - gr)) + abs(gu - gd)) + 1.0);
    let w_d = ar_op(ar_op(ar_op(ar_op(ar_op(ar_op(rho * abs(ar_m(b, 0, 2) - c)) + abs(ar_m(b, 0, 3) - gd))
        + abs(ar_m(b, -1, 2) - gl)) + abs(ar_m(b, 1, 2) - gr)) + abs(gu - gd)) + 1.0);
    let w_l = ar_op(ar_op(ar_op(ar_op(ar_op(ar_op(rho * abs(ar_m(b, -2, 0) - c)) + abs(ar_m(b, -3, 0) - gl))
        + abs(ar_m(b, -2, -1) - gu)) + abs(ar_m(b, -2, 1) - gd)) + abs(gl - gr)) + 1.0);
    let w_r = ar_op(ar_op(ar_op(ar_op(ar_op(ar_op(rho * abs(ar_m(b, 2, 0) - c)) + abs(ar_m(b, 3, 0) - gr))
        + abs(ar_m(b, 2, -1) - gu)) + abs(ar_m(b, 2, 1) - gd)) + abs(gl - gr)) + 1.0);
    let c_u = ar_op(0.5 * ar_op(ar_m(b, 0, -2) + c));
    let c_d = ar_op(0.5 * ar_op(ar_m(b, 0, 2) + c));
    let c_l = ar_op(0.5 * ar_op(ar_m(b, -2, 0) + c));
    let c_r = ar_op(0.5 * ar_op(ar_m(b, 2, 0) + c));
    // The weight of each green is proportional to the inverse of its own cost (the spec's product of
    // the other costs, divided through by the product of all of them); unused directions get 0.
    var a_u = 0.0;
    var a_d = 0.0;
    var a_l = 0.0;
    var a_r = 0.0;
    if (dir == 0u) {
        a_l = ar_div(1.0, w_l);
        a_r = ar_div(1.0, w_r);
    } else if (dir == 1u) {
        a_u = ar_div(1.0, w_u);
        a_d = ar_div(1.0, w_d);
    } else {
        a_u = ar_div(1.0, w_u);
        a_d = ar_div(1.0, w_d);
        a_l = ar_div(1.0, w_l);
        a_r = ar_div(1.0, w_r);
    }
    let sum = ar_op(ar_op(ar_op(a_u + a_d) + a_l) + a_r);
    let ng = ar_op(ar_op(ar_op(ar_op(a_u * gu) + ar_op(a_d * gd)) + ar_op(a_l * gl)) + ar_op(a_r * gr));
    let nc = ar_op(ar_op(ar_op(ar_op(a_u * c_u) + ar_op(a_d * c_d)) + ar_op(a_l * c_l)) + ar_op(a_r * c_r));
    let ks = ar_op(k * sum);
    let num = max(ar_op(ks + ng), ks);
    let den = max(ar_op(ks + nc), ks);
    return ar_op(ar_op(c + k) * ar_div(num, den)) - k;
}

fn ar_green(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let i = y * u.dims.x + x;
    let c = ar_site(x, y);
    if (c == 1u) {
        ar_st(PL_A, i, ar_ld(PL_M, i));
        return;
    }
    let rho = select(u.cst.x, u.cst.y, c == 2u);
    ar_st(PL_A, i, ar_green_estimate(i32(i), rho, u.cst.z, ar_misreg(x, y)));
}

// ---------------------------------------------------------------------------------------------
// Pass 2: green refinement (spec 3.3, and 3.6 for the CA weighting)
// ---------------------------------------------------------------------------------------------

// G2 at the R or B site with flat index `b`. `use_ca` selects the CA variant with the weight `kappa`.
fn ar_refine_site(b: i32, use_ca: bool, kappa: f32) -> f32 {
    let g0 = ar_at(PL_A, b, 0, 0);
    let n = ar_at(PL_A, b, 0, -1);
    let s = ar_at(PL_A, b, 0, 1);
    let w = ar_at(PL_A, b, -1, 0);
    let e = ar_at(PL_A, b, 1, 0);
    let nw = ar_at(PL_A, b, -1, -1);
    let ne = ar_at(PL_A, b, 1, -1);
    let sw = ar_at(PL_A, b, -1, 1);
    let se = ar_at(PL_A, b, 1, 1);
    let c0 = ar_m(b, 0, 0);
    var v = ar_op(abs(n - g0) + abs(s - g0));
    var h = ar_op(abs(w - g0) + abs(e - g0));
    if (use_ca) {
        // The axis scores also see the same-colour gradients two photosites out, weighted by kappa.
        v = ar_op(v + ar_op(kappa * ar_op(abs(ar_m(b, 0, -2) - c0) + abs(ar_m(b, 0, 2) - c0))));
        h = ar_op(h + ar_op(kappa * ar_op(abs(ar_m(b, -2, 0) - c0) + abs(ar_m(b, 2, 0) - c0))));
    }
    let g_4a = ar_at(PL_A, b, -1, 2);
    let g_4b = ar_at(PL_A, b, 1, -2);
    let g_5a = ar_at(PL_A, b, -2, 1);
    let g_5b = ar_at(PL_A, b, 2, -1);
    let g_6a = ar_at(PL_A, b, -1, -2);
    let g_6b = ar_at(PL_A, b, 1, 2);
    let g_7a = ar_at(PL_A, b, -2, -1);
    let g_7b = ar_at(PL_A, b, 2, 1);
    // Scores of the six oblique directions (indices 2 to 7).
    let s2 = ar_op(ar_op(ar_op(ar_op(abs(s - e) + abs(n - w)) + abs(ar_m(b, 2, -2) - c0))
        + abs(ar_m(b, -2, 2) - c0)) + abs(ar_m(b, -1, 1) - ar_m(b, 1, -1)));
    let s3 = ar_op(ar_op(ar_op(ar_op(abs(s - w) + abs(n - e)) + abs(ar_m(b, -2, -2) - c0))
        + abs(ar_m(b, 2, 2) - c0)) + abs(ar_m(b, -1, -1) - ar_m(b, 1, 1)));
    let s4 = ar_op(ar_op(ar_op(abs(s - ne) + abs(n - sw)) + abs(g_4a - g0)) + abs(g_4b - g0));
    let s5 = ar_op(ar_op(ar_op(abs(e - sw) + abs(w - ne)) + abs(g_5a - g0)) + abs(g_5b - g0));
    let s6 = ar_op(ar_op(ar_op(abs(n - se) + abs(s - nw)) + abs(g_6a - g0)) + abs(g_6b - g0));
    let s7 = ar_op(ar_op(ar_op(abs(w - se) + abs(e - nw)) + abs(g_7a - g0)) + abs(g_7b - g0));
    // Strict comparisons: ties keep the lower index. A tie of V and H selects the horizontal axis.
    var best = 0u;
    var low = v;
    if (h < v) {
        best = 1u;
        low = h;
    }
    if (s2 < low) {
        best = 2u;
        low = s2;
    }
    if (s3 < low) {
        best = 3u;
        low = s3;
    }
    if (s4 < low) {
        best = 4u;
        low = s4;
    }
    if (s5 < low) {
        best = 5u;
        low = s5;
    }
    if (s6 < low) {
        best = 6u;
        low = s6;
    }
    if (s7 < low) {
        best = 7u;
        low = s7;
    }
    if (best < 2u && !use_ca) {
        return g0;
    }
    var means: array<f32, 8>;
    let g2 = ar_op(2.0 * g0);
    means[0] = ar_div(ar_op(ar_op(g0 + n) + s), 3.0);
    means[1] = ar_div(ar_op(ar_op(g0 + w) + e), 3.0);
    means[2] = ar_div(ar_op(ar_op(g2 + ne) + sw), 4.0);
    means[3] = ar_div(ar_op(ar_op(g2 + nw) + se), 4.0);
    means[4] = ar_div(ar_op(ar_op(g0 + g_4a) + g_4b), 3.0);
    means[5] = ar_div(ar_op(ar_op(g0 + g_5a) + g_5b), 3.0);
    means[6] = ar_div(ar_op(ar_op(g0 + g_6a) + g_6b), 3.0);
    means[7] = ar_div(ar_op(ar_op(g0 + g_7a) + g_7b), 3.0);
    let axis = select(means[1], means[0], v < h);
    if (!use_ca) {
        return 0.5 * ar_op(axis + means[best]);
    }
    if (best < 2u) {
        // A 3-tap mean along the winning axis, which takes over from G0 as kappa grows; the weight of
        // G0 is 1 + 18 t + 102 t^3 with t = (1 - min(1, 2 kappa))^2.
        let t0 = 1.0 - min(ar_op(2.0 * kappa), 1.0);
        let t = ar_op(t0 * t0);
        let wt = ar_op(ar_op(1.0 + ar_op(18.0 * t)) + ar_op(102.0 * ar_op(ar_op(t * t) * t)));
        let wg0 = ar_op(wt * g0);
        var num = ar_op(ar_op(wg0 + w) + e);
        if (best == 0u) {
            num = ar_op(ar_op(wg0 + n) + s);
        }
        return ar_div(num, ar_op(wt + 2.0));
    }
    let k2 = ar_op(2.0 * kappa);
    return ar_div(ar_op(axis + ar_op(ar_op(k2 + 1.0) * means[best])), ar_op(k2 + 2.0));
}

// Writes the white-balanced green plane: PG = g_G G2 (spec 3.4.1; G2 = G1 at G sites).
fn ar_refine(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let i = y * u.dims.x + x;
    var g2 = ar_ld(PL_A, i);
    if (ar_site(x, y) != 1u) {
        var kappa = 0.0;
        if (u.ctl.z == 1u) {
            // The pass-2 weight is min(ca, 1); a NaN misregistration counts as 1, as on the CPU.
            let ca = ar_misreg(x, y);
            kappa = select(min(ca, 1.0), 1.0, is_nan_f(ca));
        }
        g2 = ar_refine_site(i32(i), u.ctl.z == 1u, kappa);
    }
    ar_st(PL_G, i, g2 * u.gains.y);
}

// ---------------------------------------------------------------------------------------------
// Pass 3: chroma (spec 3.4)
// ---------------------------------------------------------------------------------------------

// Step 1 (3.4.2): the missing colour at an R or B site from the two diagonal samples of that colour
// along the smoother diagonal. At an R site X is the B plane and Y the R plane, at a B site the
// reverse. It reads only native samples, so the threads of the dispatch do not depend on each other.
// The cost-weighted ratio is evaluated with the inverse costs (the spec's weights divided through by
// the product of both costs): the same value, with no product of two costs.
fn ar_step1(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let c = ar_site(x, y);
    if (c == 1u) {
        return;
    }
    let i = y * u.dims.x + x;
    let b = i32(i);
    let k = u.cst.w;
    let px = select(PL_B, PL_R, c == 2u);
    let py = select(PL_R, PL_B, c == 2u);
    let x_ul = ar_at(px, b, -1, -1);
    let x_ur = ar_at(px, b, 1, -1);
    let x_dl = ar_at(px, b, -1, 1);
    let x_dr = ar_at(px, b, 1, 1);
    let g_ul = ar_at(PL_G, b, -1, -1);
    let g_ur = ar_at(PL_G, b, 1, -1);
    let g_dl = ar_at(PL_G, b, -1, 1);
    let g_dr = ar_at(PL_G, b, 1, 1);
    let g0 = ar_at(PL_G, b, 0, 0);
    let y0 = ar_at(py, b, 0, 0);
    // "\" is smoother: A = up-left, B = down-right; else A = down-left, B = up-right.
    let main_diag = abs(x_ul - x_dr) < abs(x_dl - x_ur);
    let x_a = select(x_dl, x_ul, main_diag);
    let g_a = select(g_dl, g_ul, main_diag);
    let x_b = select(x_ur, x_dr, main_diag);
    let g_b = select(g_ur, g_dr, main_diag);
    let c_a = select(ar_at(py, b, -2, 2), ar_at(py, b, -2, -2), main_diag);
    let c_b = select(ar_at(py, b, 2, -2), ar_at(py, b, 2, 2), main_diag);
    var w_a = ar_op(ar_op(1.0 + abs(g_a - g0)) + abs(c_a - y0));
    var w_b = ar_op(ar_op(1.0 + abs(g_b - g0)) + abs(c_b - y0));
    // X not dominant over G: penalise high-chroma samples.
    let x_sum = ar_op(ar_op(ar_op(x_ul + x_ur) + x_dl) + x_dr);
    let g_sum = ar_op(ar_op(ar_op(g_ul + g_ur) + g_dl) + g_dr);
    if (ar_op(0.75 * x_sum) < g_sum) {
        w_a = ar_op(w_a * ar_op(ar_div(abs(x_a - g_a), max(ar_op(ar_op(x_a + g_a) + 1.0), 1.0)) + 0.03));
        w_b = ar_op(w_b * ar_op(ar_div(abs(x_b - g_b), max(ar_op(ar_op(x_b + g_b) + 1.0), 1.0)) + 0.03));
    }
    let i_a = ar_div(1.0, w_a);
    let i_b = ar_div(1.0, w_b);
    let t = ar_op(ar_op(i_a + i_b) * k);
    let den = max(ar_op(ar_op(ar_op(g_b * i_b) + ar_op(g_a * i_a)) + t), t);
    let num = ar_op(ar_op(ar_op(x_b * i_b) + ar_op(x_a * i_a)) + t);
    ar_st(px, i, ar_op(ar_op(g0 + k) * ar_div(num, den)) - k);
}

// R or B at a G site (3.4.3) from the four axis neighbours ordered up, down, left, right: `gn` their
// greens, `xn` their values of the colour, `cost0` the costs before the saturation penalty, `sum_g`
// the sum of the greens.
fn ar_step2_channel(g0: f32, k: f32, gn: vec4<f32>, xn: vec4<f32>, cost0: vec4<f32>, sum_g: f32) -> f32 {
    var cost = cost0;
    if (ar_op(0.75 * ar_op(ar_op(ar_op(xn.x + xn.y) + xn.z) + xn.w)) < sum_g) {
        for (var j = 0; j < 4; j = j + 1) {
            let chroma = ar_div(abs(gn[j] - xn[j]), max(ar_op(ar_op(gn[j] + xn[j]) + 1.0), 1.0));
            cost[j] = ar_op(cost[j] * ar_op(chroma + 0.03));
        }
    }
    var wsum = 0.0;
    var gsum = 0.0;
    var xsum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        let wj = ar_div(1.0, cost[j]);
        wsum = ar_op(wsum + wj);
        gsum = ar_op(gsum + ar_op(wj * gn[j]));
        xsum = ar_op(xsum + ar_op(wj * xn[j]));
    }
    let t = ar_op(k * wsum);
    let den = max(ar_op(gsum + t), t);
    return ar_op(ar_op(g0 + k) * ar_div(ar_op(xsum + t), den)) - k;
}

// Step 2 (3.4.3): R and B at a G site from the four axis neighbours (R/B sites that carry both colours
// after step 1). Both colours are computed before either is written; the thread reads R and B only at
// R/B sites and writes them only at its own G site.
fn ar_step2(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    if (ar_site(x, y) != 1u) {
        return;
    }
    let i = y * u.dims.x + x;
    let b = i32(i);
    let k = u.cst.w;
    let g0 = ar_at(PL_G, b, 0, 0);
    let gn = vec4<f32>(ar_at(PL_G, b, 0, -1), ar_at(PL_G, b, 0, 1), ar_at(PL_G, b, -1, 0), ar_at(PL_G, b, 1, 0));
    let d_v = abs(gn.x - gn.y);
    let d_h = abs(gn.z - gn.w);
    let cost0 = vec4<f32>(
        ar_op(ar_op(1.0 + d_v) + abs(ar_at(PL_G, b, 0, -2) - g0)),
        ar_op(ar_op(1.0 + d_v) + abs(ar_at(PL_G, b, 0, 2) - g0)),
        ar_op(ar_op(1.0 + d_h) + abs(ar_at(PL_G, b, -2, 0) - g0)),
        ar_op(ar_op(1.0 + d_h) + abs(ar_at(PL_G, b, 2, 0) - g0)),
    );
    let sum_g = ar_op(ar_op(ar_op(gn.x + gn.y) + gn.z) + gn.w);
    let rn = vec4<f32>(ar_at(PL_R, b, 0, -1), ar_at(PL_R, b, 0, 1), ar_at(PL_R, b, -1, 0), ar_at(PL_R, b, 1, 0));
    let bn = vec4<f32>(ar_at(PL_B, b, 0, -1), ar_at(PL_B, b, 0, 1), ar_at(PL_B, b, -1, 0), ar_at(PL_B, b, 1, 0));
    let r = ar_step2_channel(g0, k, gn, rn, cost0, sum_g);
    let bl = ar_step2_channel(g0, k, gn, bn, cost0, sum_g);
    ar_st(PL_R, i, r);
    ar_st(PL_B, i, bl);
}

// The unbalanced luma proxy of the shadow sharpening (x4 scale), floored at 1, stored as f32 (3.4.4).
fn ar_luma(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let i = (u.region.y + gid.y) * u.dims.x + u.region.x + gid.x;
    let yl = ar_op(ar_op(ar_op(ar_div(ar_ld(PL_R, i), u.gains.x) + ar_div(ar_op(2.0 * ar_ld(PL_G, i)), u.gains.y))
        + ar_div(ar_ld(PL_B, i), u.gains.z)) + 1.0);
    ar_st(PL_A, i, max(yl, 1.0));
}

// Step 3 (3.4.4): a soft-limited sharpening of the luma proxy along the smoother diagonal, where every
// native green of the 3 x 3 neighbourhood is below T, fading out at 4T. All three channels get the
// same factor. The thread reads and writes only its own R, G and B.
fn ar_sharpen(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let i = y * u.dims.x + x;
    let b = i32(i);
    // Largest native green of the 3 x 3 window (>= 0: M is clamped).
    var gmax = 0.0;
    if (ar_site(x, y) == 1u) {
        gmax = max(gmax, ar_m(b, 0, 0));
        gmax = max(gmax, ar_m(b, -1, -1));
        gmax = max(gmax, ar_m(b, 1, -1));
        gmax = max(gmax, ar_m(b, -1, 1));
        gmax = max(gmax, ar_m(b, 1, 1));
    } else {
        gmax = max(gmax, ar_m(b, 0, -1));
        gmax = max(gmax, ar_m(b, 0, 1));
        gmax = max(gmax, ar_m(b, -1, 0));
        gmax = max(gmax, ar_m(b, 1, 0));
    }
    let t = u.cst2.x;
    let f = clamp(ar_div(ar_op(ar_op(4.0 * t) - gmax), ar_op(3.0 * t)), 0.0, 1.0);
    if (f <= 0.0) {
        return;
    }
    let y0 = ar_at(PL_A, b, 0, 0);
    let pm = ar_op(ar_at(PL_A, b, -1, -1) + ar_at(PL_A, b, 1, 1));
    let pa = ar_op(ar_at(PL_A, b, 1, -1) + ar_at(PL_A, b, -1, 1));
    // The diagonal with the smaller second difference.
    let y2 = ar_op(2.0 * y0);
    var pair = pa;
    if (abs(y2 - pm) < abs(y2 - pa)) {
        pair = pm;
    }
    let lap = ar_op(5.0 * ar_op(y2 - pair));
    let lim = min(y0, u.cst2.y);
    let den = ar_op(lim + abs(lap));
    var add = 0.0;
    if (den > 0.0) {
        add = ar_div(ar_op(lim * lap), den);
    }
    let gain = ar_div(ar_op(y0 + ar_op(ar_op(0.75 * f) * add)), y0);
    ar_st(PL_R, i, ar_ld(PL_R, i) * gain);
    ar_st(PL_G, i, ar_ld(PL_G, i) * gain);
    ar_st(PL_B, i, ar_ld(PL_B, i) * gain);
}

// ---------------------------------------------------------------------------------------------
// Shadow colour matching (3.4.5)
// ---------------------------------------------------------------------------------------------

// sparse_c: the native sample of colour `c` at plane position (x, y), 0 at the other sites.
fn ar_sparse(c: u32, x: i32, y: i32) -> f32 {
    if (ar_site(u32(x), u32(y)) != c) {
        return 0.0;
    }
    return sc[u32(y) * u.dims.x + u32(x)];
}

// The horizontal row sum a + 2 b + c of three sparse samples centred on (x, y).
fn ar_sparse_row(c: u32, x: i32, y: i32) -> f32 {
    return ar_op(ar_op(ar_sparse(c, x - 1, y) + ar_op(2.0 * ar_sparse(c, x, y))) + ar_sparse(c, x + 1, y));
}

// The same for the stored plane `pl`.
fn ar_plane_row(pl: u32, x: i32, y: i32) -> f32 {
    let r = pl * u.dims.z + u32(y) * u.dims.x + u32(x);
    return ar_op(ar_op(sc[r - 1u] + ar_op(2.0 * sc[r])) + sc[r + 1u]);
}

// simple_c = Gauss3(sparse_c) m_c of the channel `u.ctl.y`: the bilinear demosaic of the native samples.
fn ar_simple(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = i32(u.region.x + gid.x);
    let y = i32(u.region.y + gid.y);
    let c = u.ctl.y;
    // acc = 1 row(-1) + 2 row(0) + 1 row(1), top to bottom.
    var acc = 0.0;
    acc = ar_op(acc + ar_sparse_row(c, x, y - 1));
    acc = ar_op(acc + ar_op(2.0 * ar_sparse_row(c, x, y)));
    acc = ar_op(acc + ar_sparse_row(c, x, y + 1));
    ar_st(PL_SIMPLE, u32(y) * u.dims.x + u32(x), ar_op(acc / 16.0) * u.mult[c]);
}

// L_c = Gauss3(simple_c) and the squared error (S3_c - L_c)^2 of the channel `u.ctl.y`.
fn ar_low_err(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = i32(u.region.x + gid.x);
    let y = i32(u.region.y + gid.y);
    let c = u.ctl.y;
    var acc = 0.0;
    acc = ar_op(acc + ar_plane_row(PL_SIMPLE, x, y - 1));
    acc = ar_op(acc + ar_op(2.0 * ar_plane_row(PL_SIMPLE, x, y)));
    acc = ar_op(acc + ar_plane_row(PL_SIMPLE, x, y + 1));
    let low = ar_op(acc / 16.0);
    let i = u32(y) * u.dims.x + u32(x);
    ar_st(PL_LOW + c, i, low);
    let e = ar_op(ar_ld(PL_R + c, i) - low);
    ar_st(PL_ERR + c, i, e * e);
}

// The weight of the 5 x 5 Gaussian at distance classes (ay, ax) = (2 - |dy|, 2 - |dx|); the matrix
// [[2 7 12 7 2] [7 31 52 31 7] [12 52 127 52 12] [7 31 52 31 7] [2 7 12 7 2]] is symmetric in them.
fn ar_w5(ay: i32, ax: i32) -> f32 {
    let lo = min(ay, ax);
    let hi = max(ay, ax);
    if (lo == 0) {
        return select(select(2.0, 7.0, hi == 1), 12.0, hi == 2);
    }
    if (lo == 1) {
        return select(31.0, 52.0, hi == 2);
    }
    return 127.0;
}

// Gauss5 of plane `pl` around flat index `b`: weights over 571, summed row by row from the top left.
fn ar_gauss5(pl: u32, b: i32) -> f32 {
    var acc = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            acc = ar_op(acc + ar_op(ar_w5(2 - abs(dy), 2 - abs(dx)) * ar_at(pl, b, dx, dy)));
        }
    }
    return ar_div(acc, 571.0);
}

// Box3 of plane `pl` around flat index `b`, row by row from the top left, over 9.
fn ar_box3(pl: u32, b: i32) -> f32 {
    var acc = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            acc = ar_op(acc + ar_at(pl, b, dx, dy));
        }
    }
    return ar_div(acc, 9.0);
}

// The matched output of the core pixel (x, y): S3_c + w (L_c - Gauss5(S3_c)) with the weight w that
// says how much of the difference between the result and the bilinear demosaic noise explains.
fn ar_match(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let i = y * u.dims.x + x;
    let b = i32(i);
    var e = 0.0;
    e = ar_op(e + ar_box3(PL_ERR, b));
    e = ar_op(e + ar_box3(PL_ERR + 1u, b));
    e = ar_op(e + ar_box3(PL_ERR + 2u, b));
    // The expected variance of one balanced sample, summed over the channels.
    var vr = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        vr = ar_op(vr + ar_op(ar_op(u.vg[c] * max(ar_ld(PL_LOW + c, i), 0.0)) + u.vr2[c]));
    }
    let excess = max(ar_op(e - ar_op(0.5 * vr)), 0.0);
    let a = ar_op(ar_op(3.0 * excess) + 1.0);
    let bb = ar_op(vr + 1.0);
    let top = max(a, bb);
    let a2 = max(ar_op(a - ar_op(0.5 * top)), 0.0);
    let b2 = max(ar_op(bb - ar_op(0.5 * top)), 0.0);
    let wt = ar_div(b2, ar_op(a2 + b2));
    var p3 = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let s3 = ar_ld(PL_R + c, i);
        p3[c] = s3 + ar_op(wt * ar_op(ar_ld(PL_LOW + c, i) - ar_gauss5(PL_R + c, b)));
    }
    ar_write_out(x, y, p3);
}

// ---------------------------------------------------------------------------------------------
// Exit (spec 3.5)
// ---------------------------------------------------------------------------------------------

// Writes the output pixel of plane position (x, y): out_c = (P3_c + [c = colour] g_c S min(m_u, 0)) / S.
// The part of a negative measurement that the entry clamp removed goes back into the photosite's own
// channel, white-balanced. The input pixel is the core pixel's, so no mirroring is needed.
fn ar_write_out(x: u32, y: u32, p3: vec3<f32>) {
    let c = ar_site(x, y);
    var mu = inp[u32(i32(y) + u.org.y) * u32(u.org.w) + u32(i32(x) + u.org.x)];
    if (u.ctl.x == 1u) {
        mu = ar_div(mu, u.gains[c]);
    }
    let removed = ar_op(min(mu, 0.0) * u.gains.w);
    var v = p3;
    v[c] = v[c] + ar_op(u.gains[c] * removed);
    let o = (y - u.geom.y) * u.ctl.w + (x - u.geom.x);
    let n = u.ctl.w * u.geom.w;
    outp[o] = ar_div(v.x, u.gains.w);
    outp[n + o] = ar_div(v.y, u.gains.w);
    outp[2u * n + o] = ar_div(v.z, u.gains.w);
}

// The output without shadow colour matching: the pass-3 result over the core.
fn ar_exit(gid: vec3<u32>) {
    if (!ar_in_region(gid)) {
        return;
    }
    let x = u.region.x + gid.x;
    let y = u.region.y + gid.y;
    let i = y * u.dims.x + x;
    ar_write_out(x, y, vec3<f32>(ar_ld(PL_R, i), ar_ld(PL_G, i), ar_ld(PL_B, i)));
}
