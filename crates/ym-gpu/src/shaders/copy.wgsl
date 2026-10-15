// Copies a rectangle of a planar image into another planar image (G-10): the crop of a tile on the
// device (the stages that were identities leave a larger image than the tile needs), the assembly of
// the parts of a demosaic core that was cut, and the window a stage reads out of a picture held in
// bands. Data movement only: no sample is computed on, so the bits are the source's under any
// compiler, fast-math included.
//
// Bindings: 0 uniform, 1 the source planes, 2 the destination planes.

struct CopyU {
    src: vec4<u32>,   // row stride, plane length, column and row of the rectangle's first pixel
    dst: vec4<u32>,   // row stride, plane length, column and row of its first pixel in the destination
    size: vec4<u32>,  // rectangle width, height, planes, 0
};

@group(0) @binding(0) var<uniform> pu: CopyU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn copy_main(gid: vec3<u32>) {
    if (gid.x >= pu.size.x || gid.y >= pu.size.y) {
        return;
    }
    for (var c = 0u; c < pu.size.z; c = c + 1u) {
        let at_src = c * pu.src.y + (pu.src.w + gid.y) * pu.src.x + pu.src.z + gid.x;
        let at_dst = c * pu.dst.y + (pu.dst.w + gid.y) * pu.dst.x + pu.dst.z + gid.x;
        dst[at_dst] = src[at_src];
    }
}
