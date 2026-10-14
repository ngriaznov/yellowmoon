// C13 helpers shared by the kernels that quantise a picture: the colour chain, the encode kernel and
// the fused tail of output sharpening (docs/02-colour-pipeline.md, stage C13). Pure functions of
// their arguments; each kernel brings its own dither table and its own output layout.

// The output transfer of one display-linear channel, clamped to [0, 1]: 0 sRGB (also Display P3),
// 1 BT.1886 (Rec.2020), odd-symmetric, NaN to 0 (ym_color::encode::OutputSpace::encode_clamped).
fn c13_one(x: f32, space: u32) -> f32 {
    if (space == 1u) {
        return unit(bt1886_inverse(x));
    }
    return unit(srgb_encode(x));
}

fn c13_encode(lin: vec3<f32>, space: u32) -> vec3<f32> {
    return vec3<f32>(c13_one(lin.x, space), c13_one(lin.y, space), c13_one(lin.z, space));
}

// round(unit(v) * 255)
fn c13_q8(v: f32) -> u32 {
    return u32(floor(unit(v) * 255.0 + 0.5));
}

// floor(unit(v) * 255 + threshold), at most 255 (ym_color::encode::dither_8bit)
fn c13_dither8(v: f32, thr: f32) -> u32 {
    return u32(min(floor(unit(v) * 255.0 + thr), 255.0));
}

// round(unit(v) * (2^bits - 1)), bits 1 to 16
fn c13_qbits(v: f32, bits: u32) -> u32 {
    let maxv = f32((1u << bits) - 1u);
    return u32(floor(unit(v) * maxv + 0.5));
}

// The 8-bit code of one channel: dithered with `thr` or rounded.
fn c13_code8(v: f32, thr: f32, dithered: bool) -> u32 {
    if (dithered) {
        return c13_dither8(v, thr);
    }
    return c13_q8(v);
}
