// R7 detail re-injection (G-06): the GPU port of `ym_gpu::cpu::detail::inject_detail`
// (`docs/05-raw-processing.md` section R7, "Detail"): `Y += beta * (Y_classical - Y_net)`, the same
// delta added to the three channels, so `R - Y` and `B - Y` are unchanged. In the classical path the
// "network output" is the denoised image and `Y_classical` the demosaic before the denoise, so the
// stage puts back a fraction of what the denoise removed.
//
// Fast-math (Metal, K8): every product and partial sum is hidden behind `opaque`, so the stage keeps
// the CPU's bits under a compiler that contracts and re-associates (a C model of the sequence under
// clang -O3 -ffast-math -ffp-contract=fast -mfma gives the strict bits; without the barriers 17 % of
// the samples are an ulp off).
//
// Two small passes. `detail_luma` forms Y_classical, the luma plane of the image that goes into the
// denoise, for the pixels the denoise outputs (the classical image itself is never kept: the stage
// reads nothing but its luma); `detail_inject` adds the delta. The luma is formed with the CPU's
// expression, left to right, from the stored planes, so the value the CPU makes from its copy of the
// image is the value stored here.
//
// Bindings: 0 uniform, 1 the input planes, 2 the classical luma plane, 3 the output planes.

struct DtParams {
    dims: vec4<u32>,   // input width, height, row stride, plane length
    core: vec4<u32>,   // core x, y, width, height in the input
    dst: vec4<u32>,    // output row stride, output plane length, luma plane row stride, 0
    org: vec4<u32>,    // the input's origin in the luma plane (x, y), 0, zero (always 0: see `opaque`)
    k: vec4<f32>,      // beta
    luma: vec4<f32>,   // luma weights (R, G, B)
};

@group(0) @binding(0) var<uniform> dt: DtParams;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> yc: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

// Always 0 at run time, which the optimiser cannot know: a value OR-ed with it is hidden from
// reassociation and from contraction into a fused multiply-add (the technique of `diffraction.wgsl`).
// An OR, not an XOR: two barriers in a row must stay a barrier. An XOR applied twice cancels
// (x ^ z ^ z == x for every z, and compilers fold it), and fast-math folds a sum that starts at 0
// (`0 + x` is x), so the first `opaque(acc + opaque(t))` of a running sum becomes the bare `t`, which
// the next sum then fuses into a multiply-add (seen in clang's code for the adaptive ratio kernel;
// docs/gpu-plan.md K8). An OR is idempotent: (x | z) | z == x | z, still one barrier.
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | dt.org.w);
}

// The luma as the CPU forms it: three rounded products added left to right, every intermediate hidden
// so that Metal's fast-math compiler cannot fuse or reorder them.
fn luma_at(x: u32, y: u32) -> f32 {
    let i = y * dt.dims.z + x;
    let r = opaque(dt.luma.x * img[i]);
    let g = opaque(dt.luma.y * img[dt.dims.w + i]);
    let b = opaque(dt.luma.z * img[2u * dt.dims.w + i]);
    return opaque(opaque(r + g) + b);
}

// Y_classical of the core pixels of the input, into a plane of the core's size (row stride `dst.z`).
fn detail_luma(gid: vec3<u32>) {
    if (gid.x >= dt.core.z || gid.y >= dt.core.w) {
        return;
    }
    yc[gid.y * dt.dst.z + gid.x] = luma_at(dt.core.x + gid.x, dt.core.y + gid.y);
}

// Y += beta (Y_classical - Y) for the core pixels of the input.
fn detail_inject(gid: vec3<u32>) {
    if (gid.x >= dt.core.z || gid.y >= dt.core.w) {
        return;
    }
    let px = dt.core.x + gid.x;
    let py = dt.core.y + gid.y;
    let i = py * dt.dims.z + px;
    let yn = luma_at(px, py);
    let d = opaque(dt.k.x * opaque(yc[(dt.org.y + py) * dt.dst.z + dt.org.x + px] - yn));
    let o = gid.y * dt.dst.x + gid.x;
    dst[o] = img[i] + d;
    dst[dt.dst.y + o] = img[dt.dims.w + i] + d;
    dst[2u * dt.dst.y + o] = img[2u * dt.dims.w + i] + d;
}
