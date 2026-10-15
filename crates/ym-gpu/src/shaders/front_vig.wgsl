// R3's vignetting gain on the device (G-07): `1 / max(G(r), 1e-4)` with `G` the piecewise-linear
// interpolation of the lens knots (`cpu::vignette::VignetteProfile`, `cpu::spline::LinearKnots`), the
// knot `i` of `n` at the normalised radius `(i + 0.5) / (n - 1)`. The radius never forms a large float
// position: the pixel centre's offset from the frame centre is the exact integer `X = 2 x + 1 - width`
// (twice the offset), so `X^2 + Y^2` is an exact integer and `r = sqrt(X^2 + Y^2) / sqrt(width^2 +
// height^2)`; the knot index and the fraction are then taken from one f32 `u = r (n - 1) - 0.5`
// (D4 of docs/gpu-plan.md: the same value in a tile as in the whole frame).
//
// A kernel that includes this file has the member `vig: VigU` in its uniform `pu`.

struct VigU {
    info: vec4<u32>,   // n (0: no vignetting), 0, 0, 0
    k: vec4<f32>,      // (n - 1) / sqrt(width^2 + height^2), the smallest attenuation, 0, 0
    att: array<vec4<f32>, 8>,   // the knot attenuations, up to 32
};

fn vig_att(i: u32) -> f32 {
    return pu.vig.att[i >> 2u][i & 3u];
}

// The gain at the pixel centre whose doubled offset from the frame centre is (dx2, dy2).
fn vig_gain(dx2: i32, dy2: i32) -> f32 {
    let ax = u32(abs(dx2));
    let ay = u32(abs(dy2));
    let s2 = ax * ax + ay * ay;
    let u = sqrt(f32(s2)) * pu.vig.k.x - 0.5;
    let last = pu.vig.info.x - 1u;
    var att: f32;
    if (!(u > 0.0)) {
        att = vig_att(0u);
    } else if (u >= f32(last)) {
        att = vig_att(last);
    } else {
        let i = u32(u);
        let t = u - f32(i);
        let a = vig_att(i);
        att = a + (vig_att(i + 1u) - a) * t;
    }
    return 1.0 / max(att, pu.vig.k.y);
}
