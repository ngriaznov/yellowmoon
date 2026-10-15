// Exactly rounded arithmetic for the raw-front kernels (G-07, K8 of docs/gpu-plan.md): products with
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
