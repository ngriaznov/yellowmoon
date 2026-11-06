// Resamples a rectangle of a finished RGBA8 picture into a cell-sized RGBA8 frame, bilinear: the draft the viewer shows
// for a zoom whose own picture is not rendered yet, made from the whole-image level of a nearby zoom (`ym-core`
// `pipeline/frame/level.rs`). A display-referred picture is resampled as it is; the exact frame replaces it.
//
// Output pixel (i, j) reads the source at x = sx (i + 0.5) + tx, y = sy (j + 0.5) + ty, in source pixel edges (the source
// pixel k spans [k, k + 1), its centre is k + 0.5); the four nearest source pixels are mixed, indices clamp at the edge.
//
// Fast-math (Metal, docs/17-gpu.md K8): there is no float in this kernel. The CPU converts the map once (`DraftFixed`) to
// i32 positions in 1/16384 of a source pixel: u = step * i + origin, the left tap is u >> 14 (arithmetic, so negatives
// floor), the weight of the right tap is (u & 16383) >> 6 (8 bits, of 256). The mix is
// (top * (256 - wb) + bottom * wb + 32768) >> 16 with top = p00 * (256 - wa) + p10 * wa, all in u32 (the largest value is
// 255 * 65536 + 32768). Integer operations are exact and no compiler may re-associate or fuse them, so every device gives
// the bytes of the CPU reference (`resample_rgba8`). The positions fit i32 by the check in `DraftFixed::new`.
//
// Bindings: 0 uniform, 1 the source pixels (one u32 each, red in the low byte), 2 the destination pixels.

struct DraftU {
    size: vec4<u32>,  // source width, source height, destination width, destination height
    map: vec4<i32>,   // sx, ox, sy, oy (1/16384 source pixel)
};

@group(0) @binding(0) var<uniform> pu: DraftU;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn unpack_rgba(p: u32) -> vec4<u32> {
    return vec4<u32>(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u, p >> 24u);
}

fn draft_main(gid: vec3<u32>) {
    if (gid.x >= pu.size.z || gid.y >= pu.size.w) {
        return;
    }
    let u = pu.map.x * i32(gid.x) + pu.map.y;
    let v = pu.map.z * i32(gid.y) + pu.map.w;
    let fu = u >> 14u;
    let fv = v >> 14u;
    let wa = u32((u & 16383) >> 6u);
    let wb = u32((v & 16383) >> 6u);
    let last_x = i32(pu.size.x) - 1;
    let last_y = i32(pu.size.y) - 1;
    let x0 = u32(clamp(fu, 0, last_x));
    let x1 = u32(clamp(fu + 1, 0, last_x));
    let y0 = u32(clamp(fv, 0, last_y));
    let y1 = u32(clamp(fv + 1, 0, last_y));
    let w = pu.size.x;
    let p00 = unpack_rgba(src[y0 * w + x0]);
    let p10 = unpack_rgba(src[y0 * w + x1]);
    let p01 = unpack_rgba(src[y1 * w + x0]);
    let p11 = unpack_rgba(src[y1 * w + x1]);
    let top = p00 * (256u - wa) + p10 * wa;
    let bottom = p01 * (256u - wa) + p11 * wa;
    let q = (top * (256u - wb) + bottom * wb + vec4<u32>(32768u)) >> vec4<u32>(16u);
    dst[gid.y * pu.size.z + gid.x] = q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}
