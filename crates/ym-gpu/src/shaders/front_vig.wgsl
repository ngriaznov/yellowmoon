// R3's vignetting gain on the device: `1 / max(G(r), 1e-4)` with `G` the piecewise-linear
// interpolation of the lens knots (`cpu::vignette::VignetteProfile`, `cpu::spline::LinearKnots`), the
// knot `i` of `n` at the normalised radius `(i + 0.5) / (n - 1)`. The CPU forms the gain in f64 and
// multiplies the photosite by `gain as f32`; so that the device multiplies by the same f32, the gain is
// formed in double-floats (`front_exact.wgsl`): the knots are (hi, lo) pairs, the radius is the exact
// integer `X^2 + Y^2` of the doubled offset of the pixel centre from the frame centre under a
// double-float square root, `u = sqrt(X^2 + Y^2) (n - 1) / sqrt(w^2 + h^2) - 0.5` (the constant is a
// pair too), the knot index and the fraction come from the pair, and the quotient is a double-float
// reciprocal. The radius never forms a large float position (D4 of docs/17-gpu.md): the same value
// in a tile as in the whole frame.
//
// A kernel that includes this file has the member `vig: VigU` in its uniform `pu` and includes
// `front_opaque.wgsl` and `front_exact.wgsl` before it.

struct VigU {
    info: vec4<u32>,   // n (0: no vignetting), 0, 0, 0
    k: vec4<f32>,      // (n - 1) / sqrt(width^2 + height^2) as a pair (hi, lo), the smallest attenuation, 0
    att: array<vec4<f32>, 16>,   // the knot attenuations as pairs: knot i is (att[i / 2][2 (i % 2)], att[i / 2][2 (i % 2) + 1])
}

fn vig_att(i: u32) -> vec2<f32> {
    let v = pu.vig.att[i >> 1u];
    if ((i & 1u) == 0u) {
        return vec2<f32>(v.x, v.y);
    }
    return vec2<f32>(v.z, v.w);
}

// The gain at the pixel centre whose doubled offset from the frame centre is (dx2, dy2).
fn vig_gain(dx2: i32, dy2: i32) -> f32 {
    let ax = u32(abs(dx2));
    let ay = u32(abs(dy2));
    let s = df_sqrt_u32(ax * ax + ay * ay);
    let scale = vec2<f32>(pu.vig.k.x, pu.vig.k.y);
    let u = df_add(df_mul(s, scale), vec2<f32>(-0.5, 0.0));
    let last = pu.vig.info.x - 1u;
    var att: vec2<f32>;
    if (u.x < 0.0 || (u.x == 0.0 && u.y <= 0.0) || u.x != u.x) {
        att = vig_att(0u);
    } else if (u.x > f32(last) || (u.x == f32(last) && u.y >= 0.0)) {
        att = vig_att(last);
    } else {
        var i = u32(floor(u.x));
        // The fraction from the pair; the integer part of the pair may be one off in the last place.
        var t = df_two_sum(opaque(u.x - f32(i)), u.y);
        if (t.x < 0.0 && i > 0u) {
            i = i - 1u;
            t = df_add(t, vec2<f32>(1.0, 0.0));
        } else if (t.x >= 1.0 && i + 1u < last) {
            i = i + 1u;
            t = df_add(t, vec2<f32>(-1.0, 0.0));
        }
        i = min(i, last - 1u);
        let a = vig_att(i);
        let b = vig_att(i + 1u);
        att = df_add(a, df_mul(df_add(b, df_neg(a)), t));
    }
    if (att.x < pu.vig.k.z) {
        att = vec2<f32>(pu.vig.k.z, 0.0);
    }
    return df_recip(att).x;
}
