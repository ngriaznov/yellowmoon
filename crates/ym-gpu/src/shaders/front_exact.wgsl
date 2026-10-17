// Exactly rounded arithmetic for the raw-front kernels (K8 of docs/17-gpu.md): products with
// their error terms and a correctly rounded quotient, none of it fused or re-associated by a fast-math
// compiler (every operation passes through `opaque`, `front_opaque.wgsl`, included before this file).

// Dekker's split of an f32 into two halves of 12 bits (exact).
fn dekker_split(a: f32) -> vec2<f32> {
    let c = opaque(4097.0 * a);
    let hi = opaque(c - opaque(c - a));
    return vec2<f32>(hi, opaque(a - hi));
}

// a * b as a rounded product and its exact error (Dekker, no fused multiply-add).
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = opaque(a * b);
    let sa = dekker_split(a);
    let sb = dekker_split(b);
    let t1 = opaque(opaque(sa.x * sb.x) - p);
    let t2 = opaque(t1 + opaque(sa.x * sb.y));
    let t3 = opaque(t2 + opaque(sa.y * sb.x));
    return vec2<f32>(p, opaque(t3 + opaque(sa.y * sb.y)));
}

// a / b for a positive b below 2^24, correctly rounded whatever the division is: the quotient may be a
// few ulp off under fast-math (reciprocal multiplication), so its exact remainder `a - q b` (the
// product by Dekker, the differences exact) gives the correction, which is far below the quotient's
// last place and so cannot be perturbed by the division that forms it. A quotient that is already
// correctly rounded is returned unchanged.
fn div_exact(a: f32, b: f32) -> f32 {
    let q0 = a / b;
    let p = two_prod(q0, b);
    let r = opaque(opaque(a - p.x) - p.y);
    return opaque(q0 + opaque(r / b));
}

// ---- double-floats ----------------------------------------------------------------------------
// A double-float is a pair (hi, lo) of f32 with |lo| at most half a unit in the last place of hi; the
// pair holds 48 bits. The raw front uses them where the CPU computes in f64 and the result is rounded
// to f32 at the end (the vignetting gain), so that the f32 the device makes is the one the CPU makes
// except where the f64 value lies within 1e-14 of a rounding boundary. Every operation is hidden from
// fast-math (`opaque`), so the error terms survive.

// a + b with |a| >= |b| (or a == 0), as a pair.
fn df_quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    return vec2<f32>(s, opaque(b - opaque(s - a)));
}

// a + b as a pair, exactly.
fn df_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = opaque(a + b);
    let bb = opaque(s - a);
    return vec2<f32>(s, opaque(opaque(a - opaque(s - bb)) + opaque(b - bb)));
}

fn df_neg(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.x, -a.y);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x);
    let t = df_two_sum(a.y, b.y);
    let c = opaque(s.y + t.x);
    let v = df_quick_two_sum(s.x, c);
    let d = opaque(t.y + v.y);
    return df_quick_two_sum(v.x, d);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = two_prod(a.x, b.x);
    let cross = opaque(opaque(a.x * b.y) + opaque(a.y * b.x));
    return df_quick_two_sum(p.x, opaque(p.y + cross));
}

// The square root of a u32 as a pair: the integer is split exactly, the f32 root is corrected once
// with the exact residual (a relative error of about 1e-13 whatever the square root instruction does).
fn df_sqrt_u32(n: u32) -> vec2<f32> {
    if (n == 0u) {
        return vec2<f32>(0.0, 0.0);
    }
    let hi = f32(n);
    // The exact remainder of the split, whichever way the conversion rounded (|lo| is at most 256).
    let lo = f32(bitcast<i32>(n - u32(hi)));
    let y0 = sqrt(hi);
    let p = two_prod(y0, y0);
    let r = opaque(opaque(opaque(hi - p.x) - p.y) + lo);
    return df_quick_two_sum(y0, opaque(r / opaque(2.0 * y0)));
}

// 1 / a as a pair: an f32 quotient and two Newton steps with the residual formed in double-floats.
fn df_recip(a: vec2<f32>) -> vec2<f32> {
    var q = vec2<f32>(1.0 / a.x, 0.0);
    for (var it = 0; it < 2; it = it + 1) {
        let e = df_add(vec2<f32>(1.0, 0.0), df_neg(df_mul(a, q)));
        q = df_add(q, df_mul(q, e));
    }
    return q;
}
