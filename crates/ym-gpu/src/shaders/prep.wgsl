// R1 normalise, R3 vignetting and R4 gains in one pass, and the clip mask: the GPU port of
// `cpu::normalize::normalize_tile`, `cpu::vignette::apply_vignette` and `cpu::gains::apply_gains`
// (`docs/05-raw-processing.md` sections R1, R3 and R4) for a rectangle of the frame, read from the
// photo's resident raw words.
//
//   x = max((raw - black[p]) * inv[p], -0.05)     per CFA position p of the frame
//   x = x * (1 / G(r))                            when the photo has a vignetting profile
//   x = x * gain[p]                               R4
//
// Each product is a pass through `opaque`, so a compiler that contracts and re-associates cannot
// fuse or reorder the three multiplications (the CPU rounds each). Without the vignetting profile the
// output has the CPU's bits; with it the gain is the f32 form of the profile (`front_vig.wgsl`).
//
// The mask is one bit per photosite of the rectangle, row after row, 32 per word (`prep_mask`: one
// thread per word, no atomics), so a tile of 1100 x 1100 photosites needs 150 KB of it. R5 reads it.
//
// Bindings: 0 uniform, 1 to 4 the raw words and steps (`front_raw.wgsl`), 5 the mosaic of the
// rectangle, 6 the mask bits.

struct PrepU {
    zero: vec4<u32>,   // always +0.0 bits (`front_opaque.wgsl`)
    fr: RawFrame,
    vig: VigU,
    rect: vec4<u32>,   // x, y, width, height of the rectangle in the frame
    dst: vec4<u32>,    // row stride of the mosaic, words of the mask per row, 0, 0
    flags: vec4<u32>,  // vignetting on, gains on, 0, 0
    gain: vec4<f32>,   // R4 gain per CFA position of the frame
    k: vec4<f32>,      // the floor of R1 (-0.05), 0, 0, 0
};

@group(0) @binding(0) var<uniform> pu: PrepU;
@group(0) @binding(5) var<storage, read_write> mosaic: array<f32>;
@group(0) @binding(6) var<storage, read_write> mask: array<u32>;

fn prep_main(gid: vec3<u32>) {
    if (gid.x >= pu.rect.z || gid.y >= pu.rect.w) {
        return;
    }
    let fx = pu.rect.x + gid.x;
    let fy = pu.rect.y + gid.y;
    let p = 2u * (fy & 1u) + (fx & 1u);
    var n = normalised(fx, fy, pu.k.x);
    if (pu.flags.x != 0u) {
        let g = vig_gain(2 * i32(fx) + 1 - i32(pu.fr.dims.x), 2 * i32(fy) + 1 - i32(pu.fr.dims.y));
        n = opaque(n * g);
    }
    if (pu.flags.y != 0u) {
        n = opaque(n * pu.gain[p]);
    }
    mosaic[gid.y * pu.dst.x + gid.x] = n;
}

fn prep_mask(gid: vec3<u32>) {
    if (gid.x >= pu.dst.y || gid.y >= pu.rect.w) {
        return;
    }
    var word = 0u;
    let x0 = gid.x * 32u;
    for (var b = 0u; b < 32u; b = b + 1u) {
        let x = x0 + b;
        if (x >= pu.rect.z) {
            break;
        }
        if (is_clipped(pu.rect.x + x, pu.rect.y + gid.y)) {
            word = word | (1u << b);
        }
    }
    mask[gid.y * pu.dst.y + gid.x] = word;
}
