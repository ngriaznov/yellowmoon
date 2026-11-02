// The mask view's overlay (docs/23 section 5.8): `out = display + w m (accent - display)` on a window of a
// display-linear picture, in place, where `m` is a node's mask held in plane 0 of another image and `w`
// the overlay's weight (0.55). It runs between output sharpening and C13 (docs/algorithms/04 and docs/02).
// The picture keeps its other planes (the letterbox mask in plane 3 is not touched, and C13 paints the paper
// there whatever is under it).
//
// Bindings: 0 uniform, 1 the picture (read_write, planes 0 to 2), 2 the mask image (plane 0).

struct TintU {
    img: vec4<u32>,     // picture: row stride, plane length, column and row of the window's first pixel
    mask: vec4<u32>,    // mask image: row stride, plane length, column and row of the window's first pixel
    size: vec4<u32>,    // window width, height, 0, 0
    accent: vec4<f32>,  // the accent in the picture's linear primaries, and the weight
};

@group(0) @binding(0) var<uniform> tu: TintU;
@group(0) @binding(1) var<storage, read_write> pic: array<f32>;
@group(0) @binding(2) var<storage, read> msk: array<f32>;

fn tint_main(gid: vec3<u32>) {
    if (gid.x >= tu.size.x || gid.y >= tu.size.y) {
        return;
    }
    let at = (tu.img.w + gid.y) * tu.img.x + tu.img.z + gid.x;
    let at_m = (tu.mask.w + gid.y) * tu.mask.x + tu.mask.z + gid.x;
    let m = clamp(msk[at_m], 0.0, 1.0);
    let k = tu.accent.w * m;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let d = pic[c * tu.img.y + at];
        pic[c * tu.img.y + at] = d + k * (tu.accent[c] - d);
    }
}
