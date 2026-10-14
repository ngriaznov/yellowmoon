// R7 detail re-injection (G-06): the GPU port of `ym_gpu::cpu::detail::inject_detail`
// (`docs/05-raw-processing.md` section R7, "Detail"): `Y += beta * (Y_classical - Y_net)`, the same
// delta added to the three channels, so `R - Y` and `B - Y` are unchanged. In the classical path the
// "network output" is the denoised image and `Y_classical` the demosaic before the denoise, so the
// stage puts back a fraction of what the denoise removed.
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
    org: vec4<u32>,    // the input's origin in the luma plane (x, y)
    k: vec4<f32>,      // beta
    luma: vec4<f32>,   // luma weights (R, G, B)
};

@group(0) @binding(0) var<uniform> dt: DtParams;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> yc: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn luma_at(x: u32, y: u32) -> f32 {
    let i = y * dt.dims.z + x;
    return dt.luma.x * img[i] + dt.luma.y * img[dt.dims.w + i] + dt.luma.z * img[2u * dt.dims.w + i];
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
    let d = dt.k.x * (yc[(dt.org.y + py) * dt.dst.z + dt.org.x + px] - yn);
    let o = gid.y * dt.dst.x + gid.x;
    dst[o] = img[i] + d;
    dst[dt.dst.y + o] = img[dt.dims.w + i] + d;
    dst[2u * dt.dst.y + o] = img[2u * dt.dims.w + i] + d;
}
