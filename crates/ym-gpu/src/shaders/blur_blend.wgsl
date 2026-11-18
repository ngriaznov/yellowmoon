// The last pass of a Blur node (docs/33 section 3.3): the blend in linear light and the encode, on the difference form
// `out = X + m (N - X D) / (D + EPS)`, with the exact branches: `m = 0` gives the input's bits (no encode of a decode) and
// `m = 1` gives `encode(N / (D + EPS))`. A mask that is NaN is 0, and one outside [0, 1] is held to it.
//
// Bindings: 0 uniform, 1 the picture (LogC4 / AWG4, image A), 2 the node's mask (plane 0, image B), 3 the sums (N in planes
// 0 to 2, D in plane 3, image C), 4 the picture the node makes (three planes, image D). The window is the made area.

@group(0) @binding(1) var<storage, read> in_v: array<f32>;
@group(0) @binding(2) var<storage, read> in_m: array<f32>;
@group(0) @binding(3) var<storage, read> in_s: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_v: array<f32>;

fn blur_blend(gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let ia = bu_at(bu.a, x, y, 0u);
    let v = vec3<f32>(in_v[ia], in_v[ia + u32(bu.a.w)], in_v[ia + 2u * u32(bu.a.w)]);
    let m = blur_clean_mask(in_m[bu_at(bu.b, x, y, 0u)]);
    var o = v;
    if (m != 0.0) {
        let is = bu_at(bu.c, x, y, 0u);
        let sp = u32(bu.c.w);
        let n = vec3<f32>(in_s[is], in_s[is + sp], in_s[is + 2u * sp]);
        let d = in_s[is + 3u * sp];
        let q = d + BLUR_EPS;
        if (m == 1.0) {
            o = logc4_encode3(n / q);
        } else {
            let lin = logc4_decode3(v);
            o = logc4_encode3(lin + m * ((n - lin * d) / q));
        }
    }
    let io = bu_at(bu.d, x, y, 0u);
    let op = u32(bu.d.w);
    out_v[io] = o.x;
    out_v[io + op] = o.y;
    out_v[io + 2u * op] = o.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur_blend(gid);
}
