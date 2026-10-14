// C13 for the picture (docs/02-colour-pipeline.md, stage C13): the output transfer function of a
// display-linear image, the quantiser, the blue-noise dither at the pixel's absolute position and the
// letterbox, written in the layout the consumer reads:
//
//   kind 1  RGBA8, one word per pixel (the viewport, the thumbnails);
//   kind 3  RGB8 interleaved, rows padded to four pixels (three words), for the JPEG / PNG / TIFF bands;
//   kind 4  RGB16 interleaved, rows padded to two pixels (three words), little-endian samples.
//
// The source is a `GpuImage`: three f32 planes, optionally a fourth that is 1 outside the photo. A
// thread writes one pixel (kind 1), four pixels (kind 3) or two (kind 4), so that every word has a
// single writer and no atomics are needed. The maths are the helpers of c13.wgsl, the same functions
// the colour chain uses for its own C13, so the two cannot differ.

struct EncodeU {
    width: u32,       // the window, pixels
    height: u32,
    src_stride: u32,  // the source image: samples per row and per plane, and the window's origin
    src_plane: u32,
    src_x: u32,
    src_y: u32,
    dst_stride: u32,  // words per destination row
    dst_x: u32,       // the window's place in the destination: pixels (kind 1 only) and rows
    dst_y: u32,
    kind: u32,
    space: u32,       // output transfer: 0 sRGB, 1 BT.1886
    dither: u32,      // 8-bit kinds: blue-noise dither, else rounding
    has_mask: u32,    // the source has a fourth plane
    bg: u32,          // letterbox colour, RGBA8 packed
    org_x: u32,       // absolute position of the window's (0, 0), for the dither
    org_y: u32,
    bits: u32,        // kind 4: significant bits, 1 to 16
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> eu: EncodeU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> dither_tab: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

fn src_at(c: u32, x: u32, y: u32) -> f32 {
    return src[c * eu.src_plane + (eu.src_y + y) * eu.src_stride + eu.src_x + x];
}

fn is_gone(x: u32, y: u32) -> bool {
    return eu.has_mask != 0u && src_at(3u, x, y) != 0.0;
}

// The encoded, unclamped-to-code values of one pixel: C13 of the three display-linear channels.
fn encoded_at(x: u32, y: u32) -> vec3<f32> {
    let lin = vec3<f32>(src_at(0u, x, y), src_at(1u, x, y), src_at(2u, x, y));
    return c13_encode(lin, eu.space);
}

fn threshold_at(x: u32, y: u32) -> f32 {
    if (eu.dither == 0u) {
        return 0.0;
    }
    return dither_tab[((eu.org_y + y) % 64u) * 64u + ((eu.org_x + x) % 64u)];
}

// The three 8-bit codes of pixel (x, y), or the letterbox colour; packed 0xRRGGBB (r in the low byte).
fn code8_at(x: u32, y: u32) -> u32 {
    if (is_gone(x, y)) {
        return eu.bg & 0x00ffffffu;
    }
    let e = encoded_at(x, y);
    let thr = threshold_at(x, y);
    let dithered = eu.dither != 0u;
    return c13_code8(e.x, thr, dithered) | (c13_code8(e.y, thr, dithered) << 8u)
        | (c13_code8(e.z, thr, dithered) << 16u);
}

fn code16_at(x: u32, y: u32) -> vec3<u32> {
    if (is_gone(x, y)) {
        return vec3<u32>(0u);
    }
    let e = encoded_at(x, y);
    return vec3<u32>(c13_qbits(e.x, eu.bits), c13_qbits(e.y, eu.bits), c13_qbits(e.z, eu.bits));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    if (y >= eu.height) {
        return;
    }
    let row = (eu.dst_y + y) * eu.dst_stride;
    if (eu.kind == 1u) {
        let x = gid.x;
        if (x >= eu.width) {
            return;
        }
        dst[row + eu.dst_x + x] = code8_at(x, y) | 0xff000000u;
    } else if (eu.kind == 3u) {
        let x = gid.x * 4u;
        if (x >= eu.width) {
            return;
        }
        var px: array<u32, 4>;
        for (var i = 0u; i < 4u; i = i + 1u) {
            px[i] = 0u;
            if (x + i < eu.width) {
                px[i] = code8_at(x + i, y);
            }
        }
        // Bytes r0 g0 b0 r1 | g1 b1 r2 g2 | b2 r3 g3 b3 of the four pixels.
        let o = row + gid.x * 3u;
        dst[o] = (px[0] & 0xffffffu) | (px[1] << 24u);
        dst[o + 1u] = ((px[1] >> 8u) & 0xffffu) | ((px[2] & 0xffffu) << 16u);
        dst[o + 2u] = ((px[2] >> 16u) & 0xffu) | (px[3] << 8u);
    } else {
        let x = gid.x * 2u;
        if (x >= eu.width) {
            return;
        }
        let a = code16_at(x, y);
        var b = vec3<u32>(0u);
        if (x + 1u < eu.width) {
            b = code16_at(x + 1u, y);
        }
        // Samples r0 g0 | b0 r1 | g1 b1, two 16-bit samples a word.
        let o = row + gid.x * 3u;
        dst[o] = a.x | (a.y << 16u);
        dst[o + 1u] = a.z | (b.x << 16u);
        dst[o + 2u] = b.y | (b.z << 16u);
    }
}
