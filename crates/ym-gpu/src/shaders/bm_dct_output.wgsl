// Method A, the last pass (`bm_dct.wgsl` has the notes): one thread per pixel of the core. Where a reference
// reached the pixel (w > 0) a filtered channel is acc / w (correctly rounded), elsewhere it keeps its input
// value, and with the blend Y is then `d + (x - d) B / 100` with the input's Y x, Cb and Cr keeping their
// filtered values (`cpu::bm_dct` `finish`); back to RGB (`cpu::ycbcr::from_ycbcr`: R = Y + 1.5748 Cr,
// B = Y + 1.8556 Cb, G = (Y - KR R - KB B) / KG); the decoding. In mode 0 (a frame under 8 px or no channel to
// filter, where `cpu::bm_dct::denoise` returns its input) the pixel is the input encoded and decoded, as
// `FrameDenoise::apply` makes it, and bindings 2 and 3 are not read; in mode 1 binding 1 is not read.
//
// Bindings: 0 uniform, 1 the input planes, 2 the Y/Cb/Cr scratch, 3 the accumulators, 4 the output (the core).

@group(0) @binding(1) var<storage, read> bm_src: array<f32>;
@group(0) @binding(2) var<storage, read> bm_ycc: array<f32>;
@group(0) @binding(3) var<storage, read> bm_acc: array<f32>;
@group(0) @binding(4) var<storage, read_write> bm_out: array<f32>;

fn bm_output(gid: vec3<u32>) {
    if (gid.x >= bp.core.z || gid.y >= bp.core.w) {
        return;
    }
    let x = bp.core.x + gid.x;
    let y = bp.core.y + gid.y;
    let o = gid.y * bp.io.z + gid.x;
    if (bp.ctl.y == 0u) {
        let s = y * bp.io.x + x;
        bm_out[o] = bm_decode(bm_encode(bm_src[s]));
        bm_out[bp.io.w + o] = bm_decode(bm_encode(bm_src[bp.io.y + s]));
        bm_out[2u * bp.io.w + o] = bm_decode(bm_encode(bm_src[2u * bp.io.y + s]));
        return;
    }
    let n = bp.frame.z;
    let i = y * bp.buf.x + x;
    var yv = bm_ycc[i];
    var cb = bm_ycc[n + i];
    var cr = bm_ycc[2u * n + i];
    let w = bm_acc[3u * n + i];
    if (w > 0.0) {
        if (bm_on(1u)) {
            // yv is the input's Y until here.
            let d = bm_div(bm_acc[i], w);
            if (bm_on(4u)) {
                yv = bm_blend(d, yv, bp.tau.w);
            } else {
                yv = d;
            }
        }
        if (bm_on(2u)) {
            cb = bm_div(bm_acc[n + i], w);
            cr = bm_div(bm_acc[2u * n + i], w);
        }
    }
    let rgb = bm_from_ycc(yv, cb, cr);
    bm_out[o] = bm_decode(rgb.x);
    bm_out[bp.io.w + o] = bm_decode(rgb.y);
    bm_out[2u * bp.io.w + o] = bm_decode(rgb.z);
}
