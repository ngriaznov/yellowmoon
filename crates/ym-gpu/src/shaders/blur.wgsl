// The Blur node's shared passes (docs/33-blur.md sections 3.2, 3.3 and 6; docs/17-gpu.md section 11b). The scaffold every type
// plugs its kernel into: decode to linear and the weights (`blur_prep.wgsl`), the coarse grid and pyramid at absolute
// alignment (`blur_pyramid.wgsl`), the bilinear or nearest read-back of a level (`blur_upsample.wgsl`) and the blend with the
// encode (`blur_blend.wgsl`). `kernels/blur_stack.rs` prepends the generated constants and appends nothing: each pass has its
// own file with its own bindings and its own `main`; this file holds what they share.
//
// Every image is a planar f32 image (`GpuImage`): the sample of plane c at column i, row j of an image whose origin is
// (ox, oy) is `c * plane + (j - oy) * stride + (i - ox)`. The origin is the place of the image's pixel (0, 0): absolute pixels
// of the photo's extent for pictures, cell indices for the cells of a level. A pass computes the window `win` and every
// pixel depends on its absolute place only (docs/17 D4, D5): no position is a running sum, so a tile equals the whole frame
// bit for bit.
//
// Safe under Metal's fast-math (docs/17 K8): the float tests are the exact `m == 0` and `m == 1` branches and `D + EPS`,
// which choose between results that agree within rounding; NaN is tested on the bits (`is_nan_f`); no `fma`; the sums of
// the pyramid are of four terms in a fixed order and their scale is a power of two.

struct BU {
    // Image A, B, C, D: the place of the image's pixel (0, 0), its row stride and its plane length.
    a: vec4<i32>,
    b: vec4<i32>,
    c: vec4<i32>,
    d: vec4<i32>,
    // The window the pass computes: first column, first row, width, height (absolute pixels, or cell indices).
    win: vec4<i32>,
    // x: Highlights as a share of 1 (prep), y: Respect mask edges as 0 or 1 (prep), z, w: 0.
    p: vec4<f32>,
    // x: the level (upsample), y: bilinear (1) or nearest (0) (upsample), z, w: the width and height of a cell grid read.
    op: vec4<u32>,
}

@group(0) @binding(0) var<uniform> bu: BU;

fn bu_at(im: vec4<i32>, i: i32, j: i32, plane: u32) -> u32 {
    return plane * u32(im.w) + u32(j - im.y) * u32(im.z) + u32(i - im.x);
}

fn blur_clean_mask(m: f32) -> f32 {
    if (is_nan_f(m)) {
        return 0.0;
    }
    return clamp(m, 0.0, 1.0);
}
