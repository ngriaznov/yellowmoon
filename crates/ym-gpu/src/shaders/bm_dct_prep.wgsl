// Method A, pass 1 (`bm_dct.wgsl` has the notes): one thread per pixel of the buffer. The working encoding
// of the three linear planes, BT.709 Y/Cb/Cr of the encoded values (`cpu::ycbcr::to_ycbcr`: Y in the CPU's
// order of terms, Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748, every product and quotient pinned) into the
// scratch planes, and the four accumulator planes set to 0. The scratch planes are dense (`width * height`,
// no padding), the input has its own row stride.
//
// Bindings: 0 uniform, 1 the input planes, 2 the Y/Cb/Cr scratch (3 planes), 3 the accumulators (4 planes).

@group(0) @binding(1) var<storage, read> bm_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> bm_ycc: array<f32>;
@group(0) @binding(3) var<storage, read_write> bm_acc: array<f32>;

fn bm_prep(gid: vec3<u32>) {
    if (gid.x >= bp.buf.x || gid.y >= bp.buf.y) {
        return;
    }
    let s = gid.y * bp.io.x + gid.x;
    let er = bm_encode(bm_src[s]);
    let eg = bm_encode(bm_src[bp.io.y + s]);
    let eb = bm_encode(bm_src[2u * bp.io.y + s]);
    let ycc = bm_to_ycc(er, eg, eb);
    let n = bp.frame.z;
    let i = gid.y * bp.buf.x + gid.x;
    bm_ycc[i] = ycc.x;
    bm_ycc[n + i] = ycc.y;
    bm_ycc[2u * n + i] = ycc.z;
    bm_acc[i] = 0.0;
    bm_acc[n + i] = 0.0;
    bm_acc[2u * n + i] = 0.0;
    bm_acc[3u * n + i] = 0.0;
}
