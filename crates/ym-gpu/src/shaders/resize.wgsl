// The Lanczos-3 resize of the export tail (docs/gpu-plan.md G-08), the GPU twin of
// `ym_export::resize::resize_lanczos3`: one separable pass per dispatch, horizontal (every output
// column from a run of columns of the same row) or vertical (every output row from a run of rows of
// the same column), three planes at a time.
//
// The taps are the CPU's (`kernels::resize::Taps`, planned in f64: the support scaled by the shrink
// factor, taps outside the image dropped, the rest normalised to sum to 1) rounded to f32. The sum
// is in difference form: out = r + sum_k w_k (v_k - r), with r the sample at the middle tap. A flat
// field (every v_k equal to r) is then exactly r in any arithmetic, fused or re-associated, which is
// what the CPU's f64 accumulation gives for free and a plain sum of f32 products does not. On other
// content the f32 sum of at most ~60 taps is within 1e-6 of the CPU's f64 result (relative to the
// local range), so nothing here depends on a rounding decision (no K8 barriers are needed).
//
// Layout. Images are planar f32 with an even row stride (`GpuImage`). A dispatch covers `dims.x`
// destination columns and `dims.y` destination rows starting at destination row `dst.z`:
//   horizontal  tap entry = x;                  source row = src.z + y;     source column = start + k
//   vertical    tap entry = dst.w + y;          source row = start - src.z + k;  source column = x
// where `src.z` is, for the vertical pass, the absolute input row held in row 0 of the source image
// (the host guarantees every tap lies inside the source).

struct ResizeU {
    dims: vec4<u32>,    // x: columns, y: rows, z: mode (0 horizontal, 1 vertical), w: planes
    src: vec4<u32>,     // x: stride, y: plane length, z: first row read (H) / input row of row 0 (V)
    dst: vec4<u32>,     // x: stride, y: plane length, z: first row written, w: first tap entry (V)
};

@group(0) @binding(0) var<uniform> ru: ResizeU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> taps: array<vec4<u32>>;   // (first input index, count, weight offset, 0)
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= ru.dims.x || y >= ru.dims.y) {
        return;
    }
    var e: vec4<u32>;
    var first: u32;     // index of tap 0 in plane 0
    var step: u32;      // distance between two taps
    if (ru.dims.z == 0u) {
        e = taps[x];
        first = (ru.src.z + y) * ru.src.x + e.x;
        step = 1u;
    } else {
        e = taps[ru.dst.w + y];
        first = (e.x - ru.src.z) * ru.src.x + x;
        step = ru.src.x;
    }
    let count = e.y;
    let mid = count / 2u;
    let out_index = (ru.dst.z + y) * ru.dst.x + x;
    for (var c = 0u; c < ru.dims.w; c = c + 1u) {
        let p = c * ru.src.y + first;
        let r = src[p + mid * step];
        var acc = 0.0;
        for (var k = 0u; k < count; k = k + 1u) {
            acc = acc + weights[e.z + k] * (src[p + k * step] - r);
        }
        dst[c * ru.dst.y + out_index] = r + acc;
    }
}
