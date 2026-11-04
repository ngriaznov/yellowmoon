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
