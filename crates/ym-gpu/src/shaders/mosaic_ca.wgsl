// R5b lateral chromatic aberration corrected on the mosaic: the GPU port of
// `cpu::lens::correct_mosaic_ca` (`docs/05-raw-processing.md` section R9, "Real files"). Every red and
// every blue photosite takes the value its colour has at the position where the lens put the feature
// that green shows at the photosite, interpolated with the Catmull-Rom weights of R9 from the
// photosites of its own colour (the colour's lattice, a site every two photosites), edge sites
// replicated at the edge of the input rectangle. Green is never touched.
//
//   d = (x + 0.5 - cx, y + 0.5 - cy),  r = |d|,  s = shift(r)        per colour curve, linear in r
//   source = (x, y) + s d / r                                        in photosites
//   lattice position = site index + (s d / r) / 2
//
// The position never forms a large float: the offset from the centre is an integer plus a constant
// fraction (`org`, `frac`), and the displacement is split into its integer part and its fraction
// before it is added to the integer lattice index (D4). One thread per photosite of the core.
//
// Bindings: 0 uniform, 1 the mosaic of the input rectangle, 2 the curves (red knots, then blue:
// radius and shift in pixels), 3 the output core.

struct CaU {
    zero: vec4<u32>,   // always +0.0 bits
    dims: vec4<u32>,   // input width, height, row stride, 0
    core: vec4<u32>,   // core x, y, width, height in the input
    dst: vec4<u32>,    // output row stride, red knots, blue knots, 0
    cfa: vec4<u32>,    // plane index at (0,0), (1,0), (0,1), (1,1) of the input
    lat: vec4<u32>,    // red lattice phase (ox, oy), blue lattice phase (ox, oy)
    org: vec4<i32>,    // frame column of the input's column 0 and frame row of its row 0, each minus floor(centre)
    frac: vec4<f32>,   // 0.5 - fract(cx), 0.5 - fract(cy)
};

@group(0) @binding(0) var<uniform> pu: CaU;
@group(0) @binding(1) var<storage, read> mosaic: array<f32>;
@group(0) @binding(2) var<storage, read> curves: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn plane_of(x: u32, y: u32) -> u32 {
    return pu.cfa[(y & 1u) * 2u + (x & 1u)];
}

// shift(r) of the curve whose n knots start at `off`: linear between knots, the end segments extended
// linearly (`RadialCurve::shift_at`). A zero-length segment (two radii that round to one f32) has t = 0.
fn curve_shift(off: u32, n: u32, r: f32) -> f32 {
    var j = 0u;
    if (r >= curves[off + n - 1u].x) {
        j = n - 2u;
    } else if (r > curves[off].x) {
        var lo = 0u;
        var hi = n - 1u;
        while (hi - lo > 1u) {
            let mid = (lo + hi) / 2u;
            if (curves[off + mid].x <= r) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        j = lo;
    }
    let a = curves[off + j];
    let b = curves[off + j + 1u];
    let span = b.x - a.x;
    var t = 0.0;
    if (span > 0.0) {
        t = (r - a.x) / span;
    }
    return a.y + opaque(opaque(b.y - a.y) * t);
}

// The four Catmull-Rom weights, with the CPU's expressions (`resample::catmull_rom_weights`), each
// operation rounded once.
fn cr_weights(t: f32) -> vec4<f32> {
    let t2 = opaque(t * t);
    let t3 = opaque(t2 * t);
    let w0 = opaque(opaque(opaque(-0.5 * t3) + t2) - opaque(0.5 * t));
    let w1 = opaque(opaque(opaque(1.5 * t3) - opaque(2.5 * t2)) + 1.0);
    let w2 = opaque(opaque(opaque(-1.5 * t3) + opaque(2.0 * t2)) + opaque(0.5 * t));
    let w3 = opaque(opaque(0.5 * t3) - opaque(0.5 * t2));
    return vec4<f32>(w0, w1, w2, w3);
}

fn lattice_value(ox: u32, oy: u32, a: i32, b: i32, nx: i32, ny: i32) -> f32 {
    let ca = u32(clamp(a, 0, nx - 1));
    let cb = u32(clamp(b, 0, ny - 1));
    return mosaic[(oy + 2u * cb) * pu.dims.z + ox + 2u * ca];
}

fn ca_main(gid: vec3<u32>) {
    if (gid.x >= pu.core.z || gid.y >= pu.core.w) {
        return;
    }
    let x = pu.core.x + gid.x;
    let y = pu.core.y + gid.y;
    let v0 = mosaic[y * pu.dims.z + x];
    let o = gid.y * pu.dst.x + gid.x;
    let c = plane_of(x, y);
    if (c == 1u) {
        dst[o] = v0;
        return;
    }
    let blue = c == 2u;
    let ox = select(pu.lat.x, pu.lat.z, blue);
    let oy = select(pu.lat.y, pu.lat.w, blue);
    let dx = f32(pu.org.x + i32(x)) + pu.frac.x;
    let dy = f32(pu.org.y + i32(y)) + pu.frac.y;
    let r = sqrt(opaque(dx * dx) + opaque(dy * dy));
    if (r < 1.0e-9) {
        dst[o] = v0;
        return;
    }
    let s = select(curve_shift(0u, pu.dst.y, r), curve_shift(pu.dst.y, pu.dst.z, r), blue);
    // The displacement in photosites, then in lattice steps (two photosites).
    let hx = 0.5 * opaque(opaque(s * dx) / r);
    let hy = 0.5 * opaque(opaque(s * dy) / r);
    let fx = floor(hx);
    let fy = floor(hy);
    let wx = cr_weights(hx - fx);
    let wy = cr_weights(hy - fy);
    let nx = i32((pu.dims.x - ox + 1u) / 2u);
    let ny = i32((pu.dims.y - oy + 1u) / 2u);
    let i0 = i32((x - ox) / 2u) + i32(fx);
    let j0 = i32((y - oy) / 2u) + i32(fy);
    var acc = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let b = j0 - 1 + k;
        let p0 = opaque(wx.x * lattice_value(ox, oy, i0 - 1, b, nx, ny));
        let p1 = opaque(wx.y * lattice_value(ox, oy, i0, b, nx, ny));
        let p2 = opaque(wx.z * lattice_value(ox, oy, i0 + 1, b, nx, ny));
        let p3 = opaque(wx.w * lattice_value(ox, oy, i0 + 2, b, nx, ny));
        let row = opaque(opaque(opaque(p0 + p1) + p2) + p3);
        acc = opaque(acc + opaque(wy[k] * row));
    }
    dst[o] = acc;
}
