// The Zoom gather of a Blur node (docs/33-blur.md section 3.5): a radial line of up to 32 taps read trilinearly from the pyramid
// of the weighted planes (`omega X` and `omega`, 2 x 2 box means at absolute alignment). For a pixel at v = p - c from the centre
// the line is a |v| pixels, n = clamp(ceil(a |v|) + 1, 2, 32) taps at p + v a t_i, t_i = (i + 1/2) / n - 1/2, equal weights; the
// level they read is clamp(log2 s, 0, L) with the spacing s = a |v| / (n - 1), the fraction blending the two levels around it.
// A cell outside the source grid has no weight. The CPU twin is `ym_color::blur::zoom`, in f64, in the same order.
//
// One dispatch per level k: a pixel is made by the dispatch of the lower level l0 it reads (the others return at once), so a
// dispatch binds at most two levels (K3: one uniform, a few storage buffers) however many levels the frame has. The positions are
// kept exact: the pixel's own integer coordinate is added to the tap's offset as an integer, so a 10 000 px frame loses nothing
// to f32 (a tap's place is an integer and a fraction).
//
// Bindings: 0 uniform, 1 the level k (four planes), 2 the level k + 1 (the same buffer as 1 at the top level), 3 the output
// (four planes over the made area).

struct ZU {
    // The place of the first cell of level k's image (cell indices), its row stride and its plane length.
    lo: vec4<i32>,
    // The same for level k + 1.
    hi: vec4<i32>,
    // The place of pixel (0, 0) of the output image, its row stride and its plane length.
    dst: vec4<i32>,
    // The pixels made: first column, first row, width, height.
    win: vec4<i32>,
    // The grids' sizes in cells: level k's width and height, level k + 1's width and height.
    dims: vec4<i32>,
    // The centre's integer parts (x, y), the level k of this dispatch and the top level L.
    cen: vec4<i32>,
    // The span a, the centre's fractions (x, y), 0.
    p: vec4<f32>,
}

@group(0) @binding(0) var<uniform> zu: ZU;
@group(0) @binding(1) var<storage, read> lo_cells: array<f32>;
@group(0) @binding(2) var<storage, read> hi_cells: array<f32>;
@group(0) @binding(3) var<storage, read_write> outc: array<f32>;

struct Axis {
    i: i32,
    t: f32,
}

// The cell and the weight of a tap at the pixel `x` plus the offset `d`, at a level of cells of 2^level pixels: with the tap at
// q = x + d, the cell u = (q + 1/2) / f - 1/2, floor and fraction, worked out on the integers (floor(d) is added to x exactly).
fn axis_pos(x: i32, d: f32, level: u32) -> Axis {
    let fl = floor(d);
    let frac = d - fl;
    let j = x + i32(fl);
    let f = i32(1u << level);
    let a2 = 2 * j + 1 - f;
    let i = a2 >> (level + 1u);
    let rem = a2 - (i << (level + 1u));
    let r = f32(rem) * 0.5 + frac;
    var t = r / f32(f);
    var ii = i;
    if (t >= 1.0) {
        ii = ii + 1;
        t = t - 1.0;
    }
    return Axis(ii, t);
}

fn cell_lo(i: i32, j: i32) -> vec4<f32> {
    if (i < zu.lo.x || j < zu.lo.y || i >= zu.lo.x + zu.dims.x || j >= zu.lo.y + zu.dims.y) {
        return vec4<f32>(0.0);
    }
    let o = u32(j - zu.lo.y) * u32(zu.lo.z) + u32(i - zu.lo.x);
    let pl = u32(zu.lo.w);
    return vec4<f32>(lo_cells[o], lo_cells[o + pl], lo_cells[o + 2u * pl], lo_cells[o + 3u * pl]);
}

fn cell_hi(i: i32, j: i32) -> vec4<f32> {
    if (i < zu.hi.x || j < zu.hi.y || i >= zu.hi.x + zu.dims.z || j >= zu.hi.y + zu.dims.w) {
        return vec4<f32>(0.0);
    }
    let o = u32(j - zu.hi.y) * u32(zu.hi.z) + u32(i - zu.hi.x);
    let pl = u32(zu.hi.w);
    return vec4<f32>(hi_cells[o], hi_cells[o + pl], hi_cells[o + 2u * pl], hi_cells[o + 3u * pl]);
}

fn bil_lo(px: Axis, py: Axis) -> vec4<f32> {
    let top = (1.0 - px.t) * cell_lo(px.i, py.i) + px.t * cell_lo(px.i + 1, py.i);
    let bottom = (1.0 - px.t) * cell_lo(px.i, py.i + 1) + px.t * cell_lo(px.i + 1, py.i + 1);
    return (1.0 - py.t) * top + py.t * bottom;
}

fn bil_hi(px: Axis, py: Axis) -> vec4<f32> {
    let top = (1.0 - px.t) * cell_hi(px.i, py.i) + px.t * cell_hi(px.i + 1, py.i);
    let bottom = (1.0 - px.t) * cell_hi(px.i, py.i + 1) + px.t * cell_hi(px.i + 1, py.i + 1);
    return (1.0 - py.t) * top + py.t * bottom;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= zu.win.z || i32(gid.y) >= zu.win.w) {
        return;
    }
    let x = zu.win.x + i32(gid.x);
    let y = zu.win.y + i32(gid.y);
    let a = zu.p.x;
    let vx = f32(x - zu.cen.x) - zu.p.y;
    let vy = f32(y - zu.cen.y) - zu.p.z;
    let line = a * sqrt(vx * vx + vy * vy);
    var n = 2;
    if (line > 0.0) {
        n = clamp(i32(ceil(min(line, 40.0))) + 1, 2, 32);
    }
    let sp = line / f32(n - 1);
    let top = u32(zu.cen.w);
    var ell = 0.0;
    if (sp > 1.0) {
        ell = min(log2(sp), f32(top));
    }
    var l0 = u32(floor(ell));
    var fr = ell - floor(ell);
    if (l0 >= top) {
        l0 = top;
        fr = 0.0;
    }
    if (l0 != u32(zu.cen.z)) {
        return;
    }
    let two = fr > 0.0;
    var acc = vec4<f32>(0.0);
    for (var i = 0; i < n; i = i + 1) {
        let t = (f32(i) + 0.5) / f32(n) - 0.5;
        let dx = vx * a * t;
        let dy = vy * a * t;
        var r = bil_lo(axis_pos(x, dx, l0), axis_pos(y, dy, l0));
        if (two) {
            let r1 = bil_hi(axis_pos(x, dx, l0 + 1u), axis_pos(y, dy, l0 + 1u));
            r = (1.0 - fr) * r + fr * r1;
        }
        acc = acc + r;
    }
    acc = acc / f32(n);
    let o = u32(y - zu.dst.y) * u32(zu.dst.z) + u32(x - zu.dst.x);
    let dp = u32(zu.dst.w);
    outc[o] = acc.x;
    outc[o + dp] = acc.y;
    outc[o + 2u * dp] = acc.z;
    outc[o + 3u * dp] = acc.w;
}
