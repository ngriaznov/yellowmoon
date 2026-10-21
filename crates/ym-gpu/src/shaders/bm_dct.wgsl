// R6's single-frame denoise by method A of `docs/algorithms/05-single-frame-denoise.md` (block matching and
// 3D DCT hard thresholding): the GPU port of `ym_gpu::cpu::frame_denoise::FrameDenoise::apply` for method A,
// that is the working encoding (`cpu::denoise_space`), `cpu::bm_dct::denoise` (BT.709 Y/Cb/Cr, the passes of
// A.5, the blend of Y) and the decoding. Three pipelines (`kernels/bm_dct.rs` appends one `main` to each):
//
//   bm_dct_prep    one thread per buffer pixel: the encoding, Y/Cb/Cr into three scratch planes, the four
//                  accumulator planes (Y, Cb, Cr, w) set to 0;
//   bm_dct         one workgroup of 256 threads per reference block of one pass (one dispatch per pass, in
//                  the CPU's pass order): the 25 candidate distances, the group, the chroma threshold of the
//                  reference block (switch 8), the 3D transform, the hard threshold, the inverse and the
//                  aggregation, slot by slot;
//   bm_dct_output  one thread per core pixel: acc / w, the blend of Y with the input's Y, back to RGB, the
//                  decoding (or, when the method changes nothing, the encoding and decoding alone, as the CPU).
//
// This file holds what the three share: the uniform block, the barrier and the Y/Cb/Cr conversions; the
// encoding and its exact arithmetic are `working_encoding.wgsl`. The constants (the encoding's knee, slope and
// gain, the BT.709 weights, the series coefficients, the DCT matrix) are written in front of them from the CPU
// reference by `kernels/bm_dct.rs`; the Kaiser window K(i, j) is in the uniform block.
//
// Fast-math (Metal, docs/17-gpu.md K8 and section 13). The result depends on decisions that compare sums (the
// hard threshold |c| < tau, the order of the candidate distances, D > theta), so every product, sum and
// difference that feeds one is formed behind `bm_op` (an OR with the uniform's zero word, which the compiler
// cannot know is zero), every division is the correctly rounded quotient (`bm_div`), and the power of the
// encoding is our own (`bm_pow`: log2 and exp2 from basic operations behind the barrier) rather than the
// device's `pow`. With that the arithmetic is the same IEEE f32 sequence on every backend, so Metal makes the
// decisions lavapipe makes; what stays a tolerance is f32 against the CPU's f64 (module notes of
// `kernels/bm_dct.rs`).

struct BmParams {
    buf: vec4<u32>,          // buffer width, height, frame x0, y0
    frame: vec4<u32>,        // frame width, height, scratch plane length (width * height), the zero word
    refs: vec4<u32>,         // first reference column and row of the pass (frame), their spacings
    ctl: vec4<u32>,          // switches (1 luma, 2 chroma, 4 blend, 8 the chroma threshold per reference block),
                             // mode (output: 0 encode and decode, 1 filtered)
    core: vec4<u32>,         // core x, y, width, height in the buffer
    io: vec4<u32>,           // input row stride, input plane length, output row stride, output plane length
    tau: vec4<f32>,          // tau Y, tau C (the one chroma threshold, without switch 8), theta, blend weight B / 100
    lvl: vec4<f32>,          // the noise of switch 8 (`cpu::bm_dct::LevelNoise`): a, b, the encoding's gain G,
                             // chroma_scale
    gains: vec4<f32>,        // the gains g_c of R, G and B, 0
    kwin: array<vec4<f32>, 16>,  // K(i, j), row-major
};

@group(0) @binding(0) var<uniform> bp: BmParams;

// Always 0 at run time, which the optimiser cannot know: OR-ing a value with it hides the value from
// reassociation, contraction and algebraic rewrites. An OR, never an XOR (docs/17-gpu.md K8): an XOR applied
// twice cancels, an OR is idempotent.
fn bm_op(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | bp.frame.w);
}

// BT.709 Y/Cb/Cr of encoded R, G, B (`cpu::ycbcr::to_ycbcr`): Y in the CPU's order of terms,
// Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748.
fn bm_to_ycc(r: f32, g: f32, b: f32) -> vec3<f32> {
    let y = bm_op(bm_op(bm_op(BM_KR * r) + bm_op(BM_KG * g)) + bm_op(BM_KB * b));
    return vec3<f32>(y, bm_div(bm_op(b - y), BM_CB_DIV), bm_div(bm_op(r - y), BM_CR_DIV));
}

// R, G, B of Y/Cb/Cr (`cpu::ycbcr::from_ycbcr`): R = Y + 1.5748 Cr, B = Y + 1.8556 Cb,
// G = (Y - KR R - KB B) / KG.
fn bm_from_ycc(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    let r = bm_op(y + bm_op(BM_CR_DIV * cr));
    let b = bm_op(y + bm_op(BM_CB_DIV * cb));
    let g = bm_div(bm_op(bm_op(y - bm_op(BM_KR * r)) - bm_op(BM_KB * b)), BM_KG);
    return vec3<f32>(r, g, b);
}

// The blend `d + (e - d) t` (`cpu::bm_dct` `finish`, on Y).
fn bm_blend(d: f32, e: f32, t: f32) -> f32 {
    return bm_op(d + bm_op(bm_op(e - d) * t));
}

// Whether the switch `bit` of `ctl.x` is on (1 luma, 2 chroma, 4 blend, 8 the chroma threshold per reference
// block).
fn bm_on(bit: u32) -> bool {
    return (bp.ctl.x & bit) != 0u;
}
