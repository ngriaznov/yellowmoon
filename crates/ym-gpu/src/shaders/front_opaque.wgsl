// The runtime zero of the raw-front kernels (G-07, K8 of docs/gpu-plan.md). Every kernel of the family
// (prep, highlights, mosaic CA, lens, sigma, bin) keeps `zero` as the first member of its uniform block
// `pu`, and the host writes +0.0 bits into `zero.x`. A value OR-ed with it is hidden from the compiler:
// Metal compiles with fast-math, which re-associates sums and contracts a product that meets a sum
// into a fused multiply-add, and the stages that must keep the CPU's rounding (a product followed by
// a product, a weighted sum of taps) pass each operation through here. An OR and not an XOR: an
// XOR applied twice cancels, an OR is idempotent (`diffraction.wgsl`, `richardson_lucy.wgsl`).
fn opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu.zero.x);
}
