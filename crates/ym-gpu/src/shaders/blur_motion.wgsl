// Motion's kernel of the Blur node (docs/33-blur.md section 3.5, docs/17-gpu.md section 11b; `kernels/blur_motion.rs`): the exact
// line, gathered over the pixels it touches. For every pixel (x, y) of the window the sums of the four weighted planes
//
//   dst(x, y) = sum over taps t of w_t * src(x + dx_t, y + dy_t)
//
// in the order of the taps (rows from the top, columns from the left: the CPU's `ym_color::blur::motion::taps`); a tap whose pixel
// lies outside the range the source holds is skipped, as the CPU's.
//
// Bindings: 0 uniform BU, 1 the source (four planes, image A), 2 the result (four planes, image B), 3 the taps, a storage array
// of vec4<f32> (dx, dy, w, 0), `bu.op.y` of them. `bu.c` is the range the source holds, in its own coordinates: first column, first
// row, width, height.

@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> taps: array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= bu.win.z || i32(gid.y) >= bu.win.w) {
        return;
    }
    let x = bu.win.x + i32(gid.x);
    let y = bu.win.y + i32(gid.y);
    let n = bu.op.y;
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var t = 0u; t < n; t = t + 1u) {
        let tap = taps[t];
        let i = x + i32(tap.x);
        let j = y + i32(tap.y);
        if (i < bu.c.x || j < bu.c.y || i >= bu.c.x + bu.c.z || j >= bu.c.y + bu.c.w) {
            continue;
        }
        let k = tap.z;
        acc = acc + k * vec4<f32>(
            src[bu_at(bu.a, i, j, 0u)],
            src[bu_at(bu.a, i, j, 1u)],
            src[bu_at(bu.a, i, j, 2u)],
            src[bu_at(bu.a, i, j, 3u)],
        );
    }
    dst[bu_at(bu.b, x, y, 0u)] = acc.x;
    dst[bu_at(bu.b, x, y, 1u)] = acc.y;
    dst[bu_at(bu.b, x, y, 2u)] = acc.z;
    dst[bu_at(bu.b, x, y, 3u)] = acc.w;
}
