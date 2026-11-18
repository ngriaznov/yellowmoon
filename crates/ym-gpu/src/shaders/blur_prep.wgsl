// Pass 1 of a Blur node: decode the picture to linear AWG4, make the weight and the weighted planes
// (docs/33 section 3.2): X = decode(v), h = the highlight weight, omega = h m (Respect mask edges on) or h, and the planes
// omega X (three) and omega, over the window. Highlights is 0 for every type but Lens, which gives h = 1.
//
// Bindings: 0 uniform, 1 the picture (LogC4 / AWG4, three planes, image A), 2 the node's mask (plane 0, image B),
// 3 the weighted planes (four planes, image C).

@group(0) @binding(1) var<storage, read> in_v: array<f32>;
@group(0) @binding(2) var<storage, read> in_m: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_s: array<f32>;

fn blur_prep(gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let ia = bu_at(bu.a, x, y, 0u);
    let v = vec3<f32>(in_v[ia], in_v[ia + u32(bu.a.w)], in_v[ia + 2u * u32(bu.a.w)]);
    let m = blur_clean_mask(in_m[bu_at(bu.b, x, y, 0u)]);
    let lin = logc4_decode3(v);
    var h = 1.0;
    let hl = bu.p.x;
    if (hl > 0.0) {
        let luma = lin.x * BLUR_LUMA.x + lin.y * BLUR_LUMA.y + lin.z * BLUR_LUMA.z;
        let ev = log2(max(luma, 1e-6) / 0.18);
        h = 1.0 + BLUR_HIGHLIGHT_GAIN * hl * smoothstep(BLUR_HIGHLIGHT_EV.x, BLUR_HIGHLIGHT_EV.y, ev);
    }
    var w = h;
    if (bu.p.y > 0.5) {
        w = h * m;
    }
    let o = bu_at(bu.c, x, y, 0u);
    let pl = u32(bu.c.w);
    out_s[o] = lin.x * w;
    out_s[o + pl] = lin.y * w;
    out_s[o + 2u * pl] = lin.z * w;
    out_s[o + 3u * pl] = w;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur_prep(gid);
}
