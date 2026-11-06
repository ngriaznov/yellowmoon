// Resamples a rectangle of a finished RGBA8 picture into a cell-sized RGBA8 frame, bilinear: the draft the viewer shows
// for a zoom whose own picture is not rendered yet, made from the whole-image level of a nearby zoom (`ym-core`
// `pipeline/frame/level.rs`). A display-referred picture is resampled as it is; the exact frame replaces it.
//
// Output pixel (i, j) reads the source at x = sx (i + 0.5) + tx, y = sy (j + 0.5) + ty, in source pixel edges (the source
// pixel k spans [k, k + 1), its centre is k + 0.5); the four nearest source pixels are mixed, indices clamp at the edge.
//
// Bindings: 0 uniform, 1 the source pixels (one u32 each, red in the low byte), 2 the destination pixels.

struct DraftU {
    size: vec4<u32>,  // source width, source height, destination width, destination height
    map: vec4<f32>,   // sx, tx, sy, ty
};

@group(0) @binding(0) var<uniform> pu: DraftU;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn unpack_rgba(p: u32) -> vec4<f32> {
    return vec4<f32>(f32(p & 255u), f32((p >> 8u) & 255u), f32((p >> 16u) & 255u), f32(p >> 24u));
}

fn draft_main(gid: vec3<u32>) {
    if (gid.x >= pu.size.z || gid.y >= pu.size.w) {
        return;
    }
    let u = pu.map.x * (f32(gid.x) + 0.5) + pu.map.y - 0.5;
    let v = pu.map.z * (f32(gid.y) + 0.5) + pu.map.w - 0.5;
    let fu = floor(u);
    let fv = floor(v);
    let a = u - fu;
    let b = v - fv;
    let last_x = i32(pu.size.x) - 1;
    let last_y = i32(pu.size.y) - 1;
    let x0 = u32(clamp(i32(fu), 0, last_x));
    let x1 = u32(clamp(i32(fu) + 1, 0, last_x));
    let y0 = u32(clamp(i32(fv), 0, last_y));
    let y1 = u32(clamp(i32(fv) + 1, 0, last_y));
    let w = pu.size.x;
    let p00 = unpack_rgba(src[y0 * w + x0]);
    let p10 = unpack_rgba(src[y0 * w + x1]);
    let p01 = unpack_rgba(src[y1 * w + x0]);
    let p11 = unpack_rgba(src[y1 * w + x1]);
    let top = p00 * (1.0 - a) + p10 * a;
    let bottom = p01 * (1.0 - a) + p11 * a;
    let c = clamp(floor((top * (1.0 - b) + bottom * b) + vec4<f32>(0.5)), vec4<f32>(0.0), vec4<f32>(255.0));
    let q = vec4<u32>(c);
    dst[gid.y * pu.size.z + gid.x] = q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}
