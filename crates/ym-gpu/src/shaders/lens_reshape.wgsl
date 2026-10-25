// L1, the lens's bokeh reshaping, on the device (docs/21 section 4, L1; the CPU reference is
// `ym_lens::l1::tile`). One module, twelve kernels, run in this order over the regions of one tile
// (kernels/lens_reshape.rs):
//
//   lr_guide     the footprint means of the square-root colours of the blur map's cells (the colour guide of
//                the upsampling), over the cells `rbs` reads.
//   lr_blur      the blur map at the pixels of `rb`: the joint bilateral upsampling guided by the pixel's own
//                colours, with the acceptance tests; 0 where there is no blur.
//   lr_lumroot   the square root of the luminance over `rbs` grown by 6.
//   lr_grad      the gradient energy of that field and of its 3 x 3 mean, over `rbs` grown by 4.
//   lr_flags     the places whose 5 x 5 energy exceeds the mean's by SHARP_RATIO (an edge sharper than its blur
//                says), over `rbs` grown by 2.
//   lr_dilate    over `rbs`: the flags dilated by 2, and lifted around a ball's rim and light (a strand and a
//                subject stay): the sharp places, 0 or 1 per pixel.
//   lr_protect   per pixel of `rb`: the density of the sharp places in the window around it holds the blur back
//                (smoothly), the ball's own blur inside its take-out disc, and the pixels a drawn oval may not
//                cover (`info`: bit 1 in a ball, bit 2 occluded).
//   lr_cover     the horizontal counts of blurred pixels in the feather's window, over the columns of `r0`.
//   lr_ramp      the stretch's ramp (0 to 1) towards the border of the blur map, over `r0` (inside a ball's disc at
//                least the ball's weight); what it stores is the gathers' blur: the map's, held at the cap, times the
//                ramp.
//   lr_takeout   the picture over `r0` with each ball's disc replaced by its ring's plane fit (by the ball's
//                smooth weight).
//   lr_vgather   the gather across the anamorphic axis over `r1`.
//   lr_final     the gather along the tangent over the output, and the balls' own light drawn back stretched (each
//                pixel of the patch reads the picture where it was before the stretch).
//
// Every pixel is a function of its absolute column and row, of the input within the halo and of the lists, in
// the CPU reference's order; reads of the intermediate fields clamp to the frame (the reference's clamp to a
// region that touches the frame's edge, since a region never ends inside the frame within reach of a pixel
// it serves). A position is an exact integer plus a small offset (D4 of docs/17-gpu.md): a ball's centre is its
// integer part and a fraction, a tap is the pixel plus an offset of at most the line's length. Decisions that
// compare sums sit behind the OR barrier (K8).
//
// Bindings: 0 uniform, 1 the input planes (the final pass reads the balls' light from them), 2 the blur (over rb),
// 3 info bits (over rb), 4 scratch (the flags' fields and the sharp places, then the taken-out picture), 5 the vertical gather's output (over r1), 6 the output
// planes (over want), 7 the gathers' blur (over r0), 8 the scale's table (profiles, the cells' geometry, the
// balls), 9 the bins, 10 the map's levels, 11 the summed-area table, 12 the cells' colour guide, 13 the
// horizontal counts.

struct ReshapeU {
    zero: vec4<u32>,        // x: always 0 (the opaque word of the barrier)
    frame: vec4<f32>,       // frame width, height, half-diagonal in pixels, noise of the square-root luminance
    par: vec4<f32>,         // stretch per unit of blur, Edges, the guide's width squared, the upsampling's support
    par2: vec4<f32>,        // the map level's cell size in pixels along x and y, the sharp test's gate, 0
    ints: vec4<i32>,        // frame width, height, 1 when the gather lines run along y, feather
    mode: vec4<i32>,        // 1 with the vertical gather, 1 with the tangential gather, balls, 0
    lvl: vec4<i32>,         // level's first cell, its columns, its rows, level 0's columns
    map: vec4<i32>,         // level 0's rows, first cell column read, first cell row read, cell columns read
    cellr: vec4<i32>,       // cell rows read, one past the last cell column read, ... row, 0
    offs: vec4<u32>,        // table: the columns' cells, the rows' cells, the balls; the bins' lists
    bins: vec4<u32>,        // bins across, bins down, bin side in pixels, 0
    src: vec4<i32>,         // the input: frame column and row of its first pixel, width, height
    srcs: vec4<u32>,        // the input: row stride, plane length
    wnt: vec4<i32>,         // the output rectangle: column, row, width, height
    r1: vec4<i32>,          // the tangential gather's input
    r0: vec4<i32>,          // the vertical gather's input, the picture taken out
    rb: vec4<i32>,          // the blur map's region
    e6: vec4<i32>,          // rb grown by 6
    e4: vec4<i32>,          // rb grown by 4
    e2: vec4<i32>,          // rb grown by 2
    cnt: vec4<i32>,         // the counts' first row and rows
    scr: vec4<u32>,         // scratch offsets (f32 elements): the square roots, the energies, their mean's, the flags
    dst: vec4<u32>,         // output row stride, plane length
    rbs: vec4<i32>,         // rb grown by the sharp places' window: the region the sharp places are known over
    scr2: vec4<u32>,        // scratch offsets (f32 elements): the sharp places (0 or 1), 0, 0, 0
};

@group(0) @binding(0) var<uniform> pu: ReshapeU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> bt: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;
@group(0) @binding(4) var<storage, read_write> scr: array<f32>;
@group(0) @binding(5) var<storage, read_write> o1: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(7) var<storage, read_write> rmp: array<f32>;
@group(0) @binding(8) var<storage, read> tab: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> bins: array<u32>;
@group(0) @binding(10) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> sat: array<u32>;
@group(0) @binding(12) var<storage, read_write> cg: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> cnt: array<u32>;

// Generated before this text (kernels/lens_reshape.rs, from ym_lens::consts): LR_B_MIN, LR_B_CAP, LR_MIN_LINE,
// LR_ACCEPT2, LR_ACCEPT_CONF, LR_SUBJECT_SHARE, LR_SHARP_FROM, LR_SHARP_RATIO, LR_STRAND_VALLEY,
// LR_STRAND_MARGIN, LR_RIM_REACH, LR_DILATE, LR_HALF_TAPS, LR_SHAPE_VECTORS, LR_PROFILE_SCALE, LR_RATIO_FROM,
// LR_RATIO_TO, LR_SOFT_HALF, LR_SOFT_FROM, LR_SOFT_TO, LR_GLOW_FADE, LR_RIM_FADE.

// A smoothstep between two bounds (the reference's `smoothstep`).
fn lr_ss(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The weight of a tap at `u` (its offset over the line's half-length): the biweight (1 - u^2)^2.
fn lr_profile(u: f32) -> f32 {
    let a = 1.0 - u * u;
    if (a <= 0.0) {
        return 0.0;
    }
    return a * a;
}

// The weight of a sample whose blur is `ratio` times the pixel's own.
fn lr_sample_w(ratio: f32) -> f32 {
    return lr_ss(LR_RATIO_FROM, LR_RATIO_TO, ratio);
}

// The share of a line a pixel keeps, by its own blur.
fn lr_knee(b: f32) -> f32 {
    return lr_ss(0.5 * LR_B_MIN, LR_B_MIN, b);
}

// Hides a value from the optimiser (docs/17-gpu.md K8): an OR with a word that is 0 at run time.
fn lr_opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu.zero.x);
}

fn lr_at(r: vec4<i32>, x: i32, y: i32) -> u32 {
    return u32((y - r.y) * r.z + (x - r.x));
}

// The input's value of channel c at the frame pixel (x, y), held inside the frame and the input.
fn lr_src(c: u32, x: i32, y: i32) -> f32 {
    let cx = clamp(clamp(x, 0, pu.ints.x - 1), pu.src.x, pu.src.x + pu.src.z - 1);
    let cy = clamp(clamp(y, 0, pu.ints.y - 1), pu.src.y, pu.src.y + pu.src.w - 1);
    return src[c * pu.srcs.y + u32(cy - pu.src.y) * pu.srcs.x + u32(cx - pu.src.x)];
}

fn lr_root(x: i32, y: i32) -> vec3<f32> {
    return vec3<f32>(
        sqrt(max(lr_src(0u, x, y), 0.0)),
        sqrt(max(lr_src(1u, x, y), 0.0)),
        sqrt(max(lr_src(2u, x, y), 0.0)),
    );
}

// The detector's luminance, 0.25 r + 0.5 g + 0.25 b.
fn lr_lum(x: i32, y: i32) -> f32 {
    let a = lr_opaque(0.25 * lr_src(0u, x, y));
    let b = lr_opaque(0.5 * lr_src(1u, x, y));
    let c = lr_opaque(0.25 * lr_src(2u, x, y));
    return lr_opaque(lr_opaque(a + b) + c);
}

// The 3 x 3 (1, 2, 1) smoothed root colours at a pixel.
fn lr_smooth(x: i32, y: i32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var j = 0; j < 3; j = j + 1) {
        let wy = select(0.25, 0.5, j == 1);
        for (var i = 0; i < 3; i = i + 1) {
            let wx = select(0.25, 0.5, i == 1);
            acc = acc + (wx * wy) * lr_root(x + i - 1, y + j - 1);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------------------------------------
// The cells of the blur map

struct LrSums {
    num: f32,
    den: f32,
    nrm: f32,
    sub: f32,
    nearest2: f32,
};

// What the cells of the map's level around the pixel (x, y) add up to, by the colour-guided weights of the
// upsampling. The cells in range are those whose centre is within the support (a cell or two more are looked at
// and refused by the distance test, so a rounding of the range cannot leave one out).
fn lr_sums(x: i32, y: i32) -> LrSums {
    let px = f32(x) + 0.5;
    let py = f32(y) + 0.5;
    let sup = pu.par.w;
    let sup2 = sup * sup;
    let sx = pu.par2.x;
    let sy = pu.par2.y;
    let g = lr_smooth(x, y);
    let lw = pu.lvl.y;
    let ncx = pu.map.w;
    let cx_lo = pu.map.y;
    let cy_lo = pu.map.z;
    let j_lo = max(i32(floor((py - sup) / sy)) - 1, cy_lo);
    let j_hi = min(i32(floor((py + sup) / sy)) + 2, pu.cellr.z);
    let i_lo = max(i32(floor((px - sup) / sx)) - 1, cx_lo);
    let i_hi = min(i32(floor((px + sup) / sx)) + 2, pu.cellr.y);
    var c: LrSums;
    c.num = 0.0;
    c.den = 0.0;
    c.nrm = 0.0;
    c.sub = 0.0;
    c.nearest2 = 1.0e30;
    for (var j = j_lo; j < j_hi; j = j + 1) {
        let yr = tab[pu.offs.y + u32(j)];
        let dy = f32(i32(yr.x) - y) + (yr.y - 0.5);
        for (var i = i_lo; i < i_hi; i = i + 1) {
            let xr = tab[pu.offs.x + u32(i)];
            let dx = f32(i32(xr.x) - x) + (xr.y - 0.5);
            let d2 = dx * dx + dy * dy;
            if (d2 >= sup2) {
                continue;
            }
            let t = 1.0 - d2 / sup2;
            let ws = t * t;
            let gc = cg[u32((j - cy_lo) * ncx + (i - cx_lo))].xyz;
            let dd = g - gc;
            let dist2 = lr_opaque(lr_opaque(lr_opaque(dd.x * dd.x) + lr_opaque(dd.y * dd.y)) + lr_opaque(dd.z * dd.z));
            let tt = 1.0 + dist2 / pu.par.z;
            let k = ws / (tt * tt);
            let cell = cells[u32(pu.lvl.x + j * lw + i)];
            if (cell.x > 0.0 && cell.y >= LR_ACCEPT_CONF) {
                c.nearest2 = min(c.nearest2, dist2);
            }
            c.num = lr_opaque(c.num + lr_opaque(k * cell.x));
            c.den = lr_opaque(c.den + lr_opaque(k * cell.y));
            c.sub = lr_opaque(c.sub + lr_opaque(k * cell.z));
            c.nrm = lr_opaque(c.nrm + k);
        }
    }
    return c;
}

// True when the cells around a pixel known to be in focus carry LR_SUBJECT_SHARE of their weight.
fn lr_subject(x: i32, y: i32) -> bool {
    let c = lr_sums(x, y);
    return c.nrm > 0.0 && c.sub >= lr_opaque(LR_SUBJECT_SHARE * c.nrm);
}

// The footprint means of the root colours of the cells (columns x_lo.., rows y_lo..).
fn lr_guide_main(gid: vec3<u32>) {
    if (gid.x >= u32(pu.map.w) || gid.y >= u32(pu.cellr.x)) {
        return;
    }
    let xr = tab[pu.offs.x + u32(pu.map.y) + gid.x];
    let yr = tab[pu.offs.y + u32(pu.map.z) + gid.y];
    var acc = vec3<f32>(0.0);
    var n = 0.0;
    for (var y = i32(yr.z); y < i32(yr.w); y = y + 1) {
        for (var x = i32(xr.z); x < i32(xr.w); x = x + 1) {
            acc = acc + lr_root(x, y);
            n = n + 1.0;
        }
    }
    var g = vec3<f32>(0.0);
    if (n > 0.0) {
        g = acc / n;
    }
    cg[gid.y * u32(pu.map.w) + gid.x] = vec4<f32>(g, 0.0);
}

// The blur map at a pixel of rb.
fn lr_blur_main(gid: vec3<u32>) {
    let r = pu.rb;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let idx = gid.y * u32(r.z) + gid.x;
    let px = f32(x) + 0.5;
    let py = f32(y) + 0.5;
    let sup = pu.par.w;
    let mw = pu.lvl.w;
    let mh = pu.map.x;
    // Places with no blur anywhere near (level 0's cells) keep none.
    let s0x = pu.frame.x / f32(mw);
    let s0y = pu.frame.y / f32(mh);
    let a0 = max(i32(floor((px - sup) / s0x)), 0);
    let a1 = min(i32(floor((px + sup) / s0x)), mw - 1);
    let b0 = max(i32(floor((py - sup) / s0y)), 0);
    let b1 = min(i32(floor((py + sup) / s0y)), mh - 1);
    var blurred = 0u;
    if (a0 <= a1 && b0 <= b1 && a0 < mw && b0 < mh) {
        let s = u32(mw + 1);
        blurred = sat[u32(b1 + 1) * s + u32(a1 + 1)] + sat[u32(b0) * s + u32(a0)]
            - sat[u32(b0) * s + u32(a1 + 1)] - sat[u32(b1 + 1) * s + u32(a0)];
    }
    var out = 0.0;
    if (blurred != 0u) {
        let c = lr_sums(x, y);
        if (c.nrm > 0.0 && c.den >= lr_opaque(0.25 * c.nrm) && c.den > 0.0 && c.nearest2 <= LR_ACCEPT2) {
            let v = c.num / c.den;
            if (v >= LR_B_MIN) {
                out = v;
            }
        }
    }
    bt[idx] = out;
}

// ---------------------------------------------------------------------------------------------------------
// The sharp places

fn lr_lumroot_main(gid: vec3<u32>) {
    let r = pu.e6;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    scr[pu.scr.x + gid.y * u32(r.z) + gid.x] = sqrt(max(lr_lum(x, y), 0.0));
}

// The field of square roots at a pixel, the coordinates held inside the frame.
fn lr_v(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, pu.ints.x - 1);
    let cy = clamp(y, 0, pu.ints.y - 1);
    return scr[pu.scr.x + lr_at(pu.e6, cx, cy)];
}

// The 3 x 3 mean of the field at a pixel (rows of three, then the three rows, left to right and top to bottom).
fn lr_s3(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, pu.ints.x - 1);
    let cy = clamp(y, 0, pu.ints.y - 1);
    var tot = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let h = lr_opaque(lr_opaque(lr_v(cx - 1, cy + dy) + lr_v(cx, cy + dy)) + lr_v(cx + 1, cy + dy));
        if (dy == -1) {
            tot = h;
        } else {
            tot = lr_opaque(tot + h);
        }
    }
    return lr_opaque(tot) / 9.0;
}

fn lr_grad_main(gid: vec3<u32>) {
    let r = pu.e4;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let gxa = lr_opaque(0.5 * (lr_v(x + 1, y) - lr_v(x - 1, y)));
    let gya = lr_opaque(0.5 * (lr_v(x, y + 1) - lr_v(x, y - 1)));
    let gxb = lr_opaque(0.5 * (lr_s3(x + 1, y) - lr_s3(x - 1, y)));
    let gyb = lr_opaque(0.5 * (lr_s3(x, y + 1) - lr_s3(x, y - 1)));
    let at = gid.y * u32(r.z) + gid.x;
    scr[pu.scr.y + at] = lr_opaque(lr_opaque(gxa * gxa) + lr_opaque(gya * gya));
    scr[pu.scr.z + at] = lr_opaque(lr_opaque(gxb * gxb) + lr_opaque(gyb * gyb));
}

// The 5 x 5 sum of an energy plane (offset `plane`) around a pixel: rows of five, then the five rows.
fn lr_window(plane: u32, x: i32, y: i32) -> f32 {
    var tot = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        let yy = clamp(y + dy, 0, pu.ints.y - 1);
        var row = scr[plane + lr_at(pu.e4, clamp(x - 2, 0, pu.ints.x - 1), yy)];
        for (var dx = -1; dx <= 2; dx = dx + 1) {
            row = lr_opaque(row + scr[plane + lr_at(pu.e4, clamp(x + dx, 0, pu.ints.x - 1), yy)]);
        }
        if (dy == -2) {
            tot = row;
        } else {
            tot = lr_opaque(tot + row);
        }
    }
    return tot;
}

fn lr_flags_main(gid: vec3<u32>) {
    let r = pu.e2;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let sa = lr_window(pu.scr.y, x, y);
    let sb = lr_window(pu.scr.z, x, y);
    var flag = 0.0;
    if (sa > pu.par2.z && sa > lr_opaque(LR_SHARP_RATIO * sb)) {
        flag = 1.0;
    }
    scr[pu.scr.w + gid.y * u32(r.z) + gid.x] = flag;
}

// The sharp places at a pixel (0 or 1), the coordinates held inside the frame (the region they are known over is
// `rbs`, which touches the frame's edge wherever the frame ends).
fn lr_lf(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, pu.ints.x - 1);
    let cy = clamp(y, 0, pu.ints.y - 1);
    return scr[pu.scr2.x + lr_at(pu.rbs, cx, cy)];
}

// ---------------------------------------------------------------------------------------------------------
// The balls

fn lr_sh(s: u32, v: u32) -> vec4<f32> {
    return tab[pu.offs.z + s * LR_SHAPE_VECTORS + v];
}

// The pixel centre's offset from the ball's centre, an exact integer difference plus the fractions.
fn lr_off(s0: vec4<f32>, x: i32, y: i32) -> vec2<f32> {
    return vec2<f32>(f32(x - i32(s0.x)) + (0.5 - s0.y), f32(y - i32(s0.z)) + (0.5 - s0.w));
}

// The first and one past the last entry of the bin of pixel (x, y) in the bins' lists.
fn lr_bin(x: i32, y: i32) -> vec2<u32> {
    let b = (u32(y) / pu.bins.z) * pu.bins.x + u32(x) / pu.bins.z;
    return vec2<u32>(pu.offs.w + bins[b], pu.offs.w + bins[b + 1u]);
}

fn lr_rim(i: u32) -> f32 {
    return tab[6u + i / 4u][i % 4u];
}

// The lens's rim profile at the normalised radius s, linear between its 16 samples.
fn lr_rim_at(s: f32) -> f32 {
    let x = clamp(s, 0.0, 1.0) * 15.0;
    let i = min(u32(x), 14u);
    let t = x - f32(i);
    return lr_rim(i) + t * (lr_rim(i + 1u) - lr_rim(i));
}

// The signed distance (approximately) to the ellipse's edge and the normalised radius at an offset d.
fn lr_ellipse(d: vec2<f32>, a: f32, b: f32, m: vec2<f32>, edge: f32) -> vec2<f32> {
    let u = d.x * m.x + d.y * m.y;
    let v = -d.x * m.y + d.y * m.x;
    let g = (u / a) * (u / a) + (v / b) * (v / b);
    if (g < 0.0625 && edge <= 1.0) {
        return vec2<f32>(-b * 0.5, sqrt(g));
    }
    let gx = 2.0 * u / (a * a);
    let gy = 2.0 * v / (b * b);
    let grad = max(sqrt(gx * gx + gy * gy), 1.0e-12);
    return vec2<f32>((g - 1.0) / grad, sqrt(g));
}

fn lr_cover(sd: f32, edge: f32) -> f32 {
    let t = clamp(0.5 - sd / edge, 0.0, 1.0);
    if (edge > 1.0) {
        return t * t * (3.0 - 2.0 * t);
    }
    return t;
}

// Is the pixel (x, y) in the box of the ball's reach, [floor(cx - reach), ceil(cx + reach)] by the same for y?
fn lr_in_reach(s0: vec4<f32>, reach: f32, x: i32, y: i32) -> bool {
    return !(x - i32(s0.x) < i32(floor(s0.y - reach)) || x - i32(s0.x) > i32(ceil(s0.y + reach))
        || y - i32(s0.z) < i32(floor(s0.w - reach)) || y - i32(s0.z) > i32(ceil(s0.w + reach)));
}

// Coverage (0 to 1; a rim `edge` pixels wide) and normalised radius of ball s at the pixel (x, y); 0 outside the
// box of the ball's reach, [floor(cx - reach), ceil(cx + reach)] by the same for y, which is where the reference
// draws (the first-order distance of the coverage leaves a faint tail beyond it).
fn lr_coverage(s: u32, x: i32, y: i32) -> vec2<f32> {
    let s0 = lr_sh(s, 0u);
    let s1 = lr_sh(s, 1u);
    let s2 = lr_sh(s, 2u);
    if (!lr_in_reach(s0, lr_sh(s, 4u).z, x, y)) {
        return vec2<f32>(0.0, 0.0);
    }
    let d = lr_off(s0, x, y);
    if (s2.z == 0.0) {
        let e = lr_ellipse(d, s1.x, s1.y, s1.zw, s2.w);
        return vec2<f32>(lr_cover(e.x, s2.w), e.y);
    }
    let e1 = lr_ellipse(d + s2.xy, s1.x, s1.y, s1.zw, s2.w);
    let e2 = lr_ellipse(d - s2.xy, s1.x, s1.y, s1.zw, s2.w);
    return vec2<f32>(lr_cover(max(e1.x, e2.x), s2.w), max(e1.y, e2.y));
}

// A thin dark line: much darker than the pixels three either side of it along a row or a column.
fn lr_valley(lum: f32, a: f32, b: f32) -> bool {
    let m = min(a, b);
    let margin = lr_opaque(lr_opaque(LR_STRAND_MARGIN * sqrt(max(m, 0.0))) * pu.frame.w);
    return lum < lr_opaque(LR_STRAND_VALLEY * m) - margin;
}

fn lr_thin_dark(x: i32, y: i32) -> bool {
    let lum = lr_lum(x, y);
    return lr_valley(lum, lr_lum(x - 3, y), lr_lum(x + 3, y))
        || lr_valley(lum, lr_lum(x, y - 3), lr_lum(x, y + 3));
}

// Over rbs: the flags dilated by LR_DILATE pixels in both directions, and lifted around the balls: a flagged pixel
// on a ball's rim or in its light is the ball's own edge, not detail, unless it is a thin dark line (a strand in
// front of the light) or the edge of a subject beside it. 0 or 1.
fn lr_dilate_main(gid: vec3<u32>) {
    let r = pu.rbs;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    var m = 0.0;
    for (var dy = -LR_DILATE; dy <= LR_DILATE; dy = dy + 1) {
        let yy = clamp(y + dy, 0, pu.ints.y - 1);
        for (var dx = -LR_DILATE; dx <= LR_DILATE; dx = dx + 1) {
            let xx = clamp(x + dx, 0, pu.ints.x - 1);
            m = max(m, scr[pu.scr.w + lr_at(pu.e2, xx, yy)]);
        }
    }
    var zero = m > 0.0;
    if (zero) {
        let range = lr_bin(x, y);
        // Is the pixel on a ball's rim, or in its light? (Any ball: the order does not matter.)
        var zone = false;
        for (var k = range.x; k < range.y; k = k + 1u) {
            let s = bins[k];
            let s0 = lr_sh(s, 0u);
            let s4 = lr_sh(s, 4u);
            let d = length(lr_off(s0, x, y));
            var core = false;
            if (d < s4.x) {
                core = lr_lum(x, y) >= s4.y;
            }
            if (abs(d - s4.x) <= LR_RIM_REACH || core) {
                zone = true;
                break;
            }
        }
        if (zone && !lr_thin_dark(x, y) && !lr_subject(x, y)) {
            zero = false;
        }
    }
    scr[pu.scr2.x + gid.y * u32(r.z) + gid.x] = select(0.0, 1.0, zero);
}

// Per pixel of rb: the density of the sharp places in the window around it holds the blur back, smoothly; the
// balls' own blur inside their take-out discs, and the pixels a drawn oval may not cover.
fn lr_protect_main(gid: vec3<u32>) {
    let r = pu.rb;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let idx = gid.y * u32(r.z) + gid.x;
    let hk = pu.frame.z;
    let range = lr_bin(x, y);
    // The share of the window that is sharp (the counts are whole numbers, so any order sums them alike).
    var tot = 0.0;
    for (var dy = -LR_SOFT_HALF; dy <= LR_SOFT_HALF; dy = dy + 1) {
        var row = 0.0;
        for (var dx = -LR_SOFT_HALF; dx <= LR_SOFT_HALF; dx = dx + 1) {
            row = row + lr_lf(x + dx, y + dy);
        }
        tot = tot + row;
    }
    let side = f32(2 * LR_SOFT_HALF + 1);
    let density = lr_opaque(tot) / (side * side);
    var b = bt[idx];
    let hold = lr_ss(LR_SOFT_FROM, LR_SOFT_TO, density);
    // It acts from a blur of about LR_SHARP_FROM pixels (a smoothstep over 3 to 5 at 4).
    let px = lr_ss(LR_SHARP_FROM - 1.0, LR_SHARP_FROM + 1.0, b * hk);
    b = b * (1.0 - hold * px);
    let zero = lr_lf(x, y) > 0.0;
    // Inside a ball's take-out disc the map is the ball's radius (the first ball in the list that holds the pixel).
    var in_ball = false;
    if (!zero) {
        for (var k = range.x; k < range.y; k = k + 1u) {
            let s = bins[k];
            let d = lr_off(lr_sh(s, 0u), x, y);
            let r_in = lr_sh(s, 3u).w + 1.0;
            if (d.x * d.x + d.y * d.y <= r_in * r_in) {
                in_ball = true;
                if (b < LR_B_MIN) {
                    b = lr_sh(s, 4u).w;
                }
            }
        }
    }
    // Where a drawn oval may not go: pixels of a sharp subject, or a strand, that have no blur of their own.
    var occ = false;
    if (b < LR_B_MIN) {
        var covered = false;
        for (var k = range.x; k < range.y; k = k + 1u) {
            if (lr_coverage(bins[k], x, y).x > 0.0) {
                covered = true;
                break;
            }
        }
        if (covered) {
            occ = (zero && lr_thin_dark(x, y)) || lr_subject(x, y);
        }
    }
    bt[idx] = b;
    info[idx] = select(0u, 1u, in_ball) | select(0u, 2u, occ);
}

// ---------------------------------------------------------------------------------------------------------
// The ramp

// Horizontal counts of the pixels with a blur in the window of the feather, over the columns of r0 and the rows
// the vertical window reaches.
fn lr_cover_main(gid: vec3<u32>) {
    let r = pu.r0;
    if (gid.x >= u32(r.z) || gid.y >= u32(pu.cnt.y)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = pu.cnt.x + i32(gid.y);
    let f = pu.ints.w;
    var n = 0u;
    for (var d = -f; d <= f; d = d + 1) {
        let xx = clamp(x + d, 0, pu.ints.x - 1);
        n = n + select(0u, 1u, bt[lr_at(pu.rb, xx, y)] >= LR_B_MIN);
    }
    cnt[gid.y * u32(r.z) + gid.x] = n;
}

// The ramp: the share of the window that has a blur, from 0 at one half to 1 at nine tenths, smoothed; in a ball's
// disc at least the ball's weight. What is stored is the gathers' blur: the map's, held at the cap, times the ramp.
fn lr_ramp_main(gid: vec3<u32>) {
    let r = pu.r0;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let f = pu.ints.w;
    let last = pu.cnt.x + pu.cnt.y - 1;
    var total = 0u;
    if (f > 0) {
        for (var d = -f; d <= f; d = d + 1) {
            let yy = clamp(y + d, pu.cnt.x, last);
            total = total + cnt[u32(yy - pu.cnt.x) * u32(r.z) + gid.x];
        }
    }
    var v = 1.0;
    if (f > 0) {
        let n = f32((2 * f + 1) * (2 * f + 1));
        let t = clamp((f32(total) / n - 0.5) / 0.4, 0.0, 1.0);
        v = t * t * (3.0 - 2.0 * t);
    }
    // Inside a ball's disc the stretch is at least the ball's weight: the full stretch over the ball, fading out over
    // the glow's width at the disc's edge.
    if ((info[lr_at(pu.rb, x, y)] & 1u) != 0u) {
        let range = lr_bin(x, y);
        var w = 0.0;
        for (var k = range.x; k < range.y; k = k + 1u) {
            let s = bins[k];
            let d = length(lr_off(lr_sh(s, 0u), x, y));
            let r_in = lr_sh(s, 3u).w + 1.0;
            if (d * d <= r_in * r_in) {
                w = max(w, lr_weight(s, d));
            }
        }
        v = max(v, w);
    }
    rmp[gid.y * u32(r.z) + gid.x] = lr_opaque(min(bt[lr_at(pu.rb, x, y)], LR_B_CAP) * v);
}

// ---------------------------------------------------------------------------------------------------------
// Taking the balls out

// The weight of ball s's own light and of its take-out at the distance d from its centre (the reference's
// `Shape::weight`): 1 over the ball, falling to 0 half a pixel beyond the take-out radius over LR_GLOW_FADE of the
// glow's width, at least a pixel.
fn lr_weight(s: u32, d: f32) -> f32 {
    let r_out = lr_sh(s, 3u).w;
    let r_ball = lr_sh(s, 4u).x;
    let fade = max(LR_GLOW_FADE * (r_out - r_ball), 1.0) + 0.5;
    let t = clamp((r_out + 0.5 - d) / fade, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn lr_takeout_main(gid: vec3<u32>) {
    let r = pu.r0;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    var v = vec3<f32>(lr_src(0u, x, y), lr_src(1u, x, y), lr_src(2u, x, y));
    if (bt[lr_at(pu.rb, x, y)] >= LR_B_MIN) {
        let range = lr_bin(x, y);
        for (var k = range.x; k < range.y; k = k + 1u) {
            let s = bins[k];
            let d = lr_off(lr_sh(s, 0u), x, y);
            let m = lr_weight(s, length(d));
            if (m <= 0.0) {
                continue;
            }
            for (var c = 0u; c < 3u; c = c + 1u) {
                let f = lr_sh(s, 5u + c);
                let fit = max(f.x + f.y * d.x + f.z * d.y, 0.0);
                v[c] = v[c] + m * (fit - v[c]);
            }
        }
    }
    let plane = u32(r.z * r.w);
    let at = gid.y * u32(r.z) + gid.x;
    scr[at] = v.x;
    scr[plane + at] = v.y;
    scr[2u * plane + at] = v.z;
}

// ---------------------------------------------------------------------------------------------------------
// The gather across the anamorphic axis

// The pixel at position j along the line through (x, y), held inside r0: its x and y.
fn lr_line(j: i32, x: i32, y: i32) -> vec2<i32> {
    if (pu.ints.z == 1) {
        return vec2<i32>(x, clamp(j, pu.r0.y, pu.r0.y + pu.r0.w - 1));
    }
    return vec2<i32>(clamp(j, pu.r0.x, pu.r0.x + pu.r0.z - 1), y);
}

fn lr_wk(p: vec2<i32>) -> vec3<f32> {
    let plane = u32(pu.r0.z * pu.r0.w);
    let i = lr_at(pu.r0, p.x, p.y);
    return vec3<f32>(scr[i], scr[plane + i], scr[2u * plane + i]);
}

fn lr_vgather_main(gid: vec3<u32>) {
    let r = pu.r1;
    if (gid.x >= u32(r.z) || gid.y >= u32(r.w)) {
        return;
    }
    let x = r.x + i32(gid.x);
    let y = r.y + i32(gid.y);
    let plane1 = u32(r.z * r.w);
    let at = gid.y * u32(r.z) + gid.x;
    let own = lr_wk(vec2<i32>(x, y));
    let bp = rmp[lr_at(pu.r0, x, y)];
    let len = lr_opaque(lr_opaque(lr_opaque(pu.par.x * bp) * pu.frame.z) * lr_knee(bp));
    var out = own;
    if (len >= LR_MIN_LINE) {
        let pos = select(x, y, pu.ints.z == 1);
        let half = clamp(i32(ceil(len)), 1, LR_HALF_TAPS);
        let step = len / f32(half);
        // The pixel's own value (weight 1), then the taps outwards on each side, the weight of a tap the profile
        // times the sample rule's smoothstep of its blur over the pixel's own.
        var acc = own;
        var wsum = 1.0;
        for (var side = 0; side < 2; side = side + 1) {
            let sign = select(-1.0, 1.0, side == 1);
            for (var k = 1; k <= half; k = k + 1) {
                let off = sign * f32(k) * step;
                let wp = lr_profile(off / len);
                if (wp <= 0.0) {
                    break;
                }
                let f0 = floor(off);
                let fr = off - f0;
                let pa = lr_line(pos + i32(f0), x, y);
                let pb = lr_line(pos + i32(f0) + 1, x, y);
                let ba = rmp[lr_at(pu.r0, pa.x, pa.y)];
                let bb = rmp[lr_at(pu.r0, pb.x, pb.y)];
                let bs = ba + fr * (bb - ba);
                let wt = wp * lr_sample_w(bs / bp);
                if (wt <= 0.0) {
                    continue;
                }
                let va = lr_wk(pa);
                let vb = lr_wk(pb);
                acc = acc + wt * (va + fr * (vb - va));
                wsum = wsum + wt;
            }
        }
        out = acc / wsum;
    }
    o1[at] = out.x;
    o1[plane1 + at] = out.y;
    o1[2u * plane1 + at] = out.z;
}

// ---------------------------------------------------------------------------------------------------------
// The gather along the tangent, and the balls drawn back

fn lr_radius(i: u32) -> f32 {
    return tab[i / 4u][i % 4u];
}

fn lr_swirl(i: u32) -> f32 {
    return tab[3u + i / 4u][i % 4u];
}

// The lens's swirl profile at the field radius rho, linear between its nine radii, held beyond the last.
fn lr_swirl_at(rho: f32) -> f32 {
    if (rho <= lr_radius(0u)) {
        return lr_swirl(0u);
    }
    for (var i = 1u; i < 9u; i = i + 1u) {
        if (rho <= lr_radius(i)) {
            let t = (rho - lr_radius(i - 1u)) / (lr_radius(i) - lr_radius(i - 1u));
            return lr_swirl(i - 1u) + t * (lr_swirl(i) - lr_swirl(i - 1u));
        }
    }
    return lr_swirl(8u);
}

fn lr_o1(x: i32, y: i32) -> vec3<f32> {
    let r = pu.r1;
    let plane = u32(r.z * r.w);
    let i = lr_at(r, clamp(x, r.x, r.x + r.z - 1), clamp(y, r.y, r.y + r.w - 1));
    return vec3<f32>(o1[i], o1[plane + i], o1[2u * plane + i]);
}

// The gathers' blur at a frame pixel, the coordinates held inside r0.
fn lr_be(x: i32, y: i32) -> f32 {
    let r = pu.r0;
    return rmp[lr_at(r, clamp(x, r.x, r.x + r.z - 1), clamp(y, r.y, r.y + r.w - 1))];
}

// The blur map at a frame pixel, the coordinates held inside r0 (the reference's `sample_blur` reads r0).
fn lr_bt_at(x: i32, y: i32) -> f32 {
    let r = pu.r0;
    return bt[lr_at(pu.rb, clamp(x, r.x, r.x + r.z - 1), clamp(y, r.y, r.y + r.w - 1))];
}

// The factor the lens's rim profile puts on a ball's light at s radii from its centre: the profile over its mean
// across the ball, fading to 1 over LR_RIM_FADE radii beyond its edge (the reference's `rim_gain`).
fn lr_rim_gain(s: f32, strength: f32) -> f32 {
    let g = 1.0 + strength * (lr_rim_at(s) / tab[10u].x - 1.0);
    let t = clamp((s - 1.0) / LR_RIM_FADE, 0.0, 1.0);
    let fade = t * t * (3.0 - 2.0 * t);
    return g + fade * (1.0 - g);
}

// The stretched light of ball s at the pixel (x, y) (the reference's `redraw`): the picture above the ring's plane
// fit, read where the pixel was before the stretch, as the rim profile modulates it, where the ball was taken out,
// cut by the cat's-eye mask, in the part of a shared disc this ball owns.
fn lr_ball_light(s: u32, x: i32, y: i32) -> vec3<f32> {
    let s0 = lr_sh(s, 0u);
    let s4 = lr_sh(s, 4u);
    if (!lr_in_reach(s0, s4.z, x, y)) {
        return vec3<f32>(0.0);
    }
    let s1 = lr_sh(s, 1u);
    let m = s1.zw;
    let d = lr_off(s0, x, y);
    // Where the pixel was before the stretch: along the long axis 1 / f of the distance, across it the same.
    let u = (d.x * m.x + d.y * m.y) / lr_sh(s, 3u).x;
    let v = -d.x * m.y + d.y * m.x;
    var alpha = lr_weight(s, length(vec2<f32>(u, v)));
    if (alpha <= 0.0) {
        return vec3<f32>(0.0);
    }
    let sx = u * m.x - v * m.y;
    let sy = u * m.y + v * m.x;
    // The light of a place an earlier ball's disc covers is shared: that ball has its weight there, this one the rest.
    for (var o = 0u; o < 3u; o = o + 1u) {
        let e = lr_sh(s, 8u + o);
        if (e.z > 0.0) {
            let t = clamp((e.z + 0.5 - length(vec2<f32>(sx - e.x, sy - e.y))) / e.w, 0.0, 1.0);
            alpha = alpha * (1.0 - t * t * (3.0 - 2.0 * t));
        }
    }
    if (alpha <= 0.0) {
        return vec3<f32>(0.0);
    }
    var mask = 1.0;
    if (lr_sh(s, 2u).z > 0.0) {
        mask = lr_coverage(s, x, y).x;
        if (mask <= 0.0) {
            return vec3<f32>(0.0);
        }
    }
    // The source position (pixel centres at i + 0.5): the ball's integer part plus a small offset.
    let tx = s0.y - 0.5 + sx;
    let ty = s0.w - 0.5 + sy;
    let fx = floor(tx);
    let fy = floor(ty);
    let wx = tx - fx;
    let wy = ty - fy;
    let ix = i32(s0.x) + i32(fx);
    let iy = i32(s0.z) + i32(fy);
    let b00 = lr_bt_at(ix, iy);
    let b10 = lr_bt_at(ix + 1, iy);
    let b01 = lr_bt_at(ix, iy + 1);
    let b11 = lr_bt_at(ix + 1, iy + 1);
    let btop = b00 + wx * (b10 - b00);
    let bbot = b01 + wx * (b11 - b01);
    let taken = lr_ss(0.5 * LR_B_MIN, LR_B_MIN, btop + wy * (bbot - btop));
    if (taken <= 0.0) {
        return vec3<f32>(0.0);
    }
    let gain = lr_rim_gain(length(vec2<f32>(sx, sy)) / s4.x, lr_sh(s, 3u).y);
    let k = alpha * mask * taken * gain;
    var light = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let p00 = lr_src(c, ix, iy);
        let p10 = lr_src(c, ix + 1, iy);
        let p01 = lr_src(c, ix, iy + 1);
        let p11 = lr_src(c, ix + 1, iy + 1);
        let top = p00 + wx * (p10 - p00);
        let bot = p01 + wx * (p11 - p01);
        let orig = top + wy * (bot - top);
        let f = lr_sh(s, 5u + c);
        let fit = max(f.x + f.y * sx + f.z * sy, 0.0);
        light[c] = k * max(orig - fit, 0.0);
    }
    return light;
}

fn lr_final_main(gid: vec3<u32>) {
    let w = pu.wnt;
    if (gid.x >= u32(w.z) || gid.y >= u32(w.w)) {
        return;
    }
    let x = w.x + i32(gid.x);
    let y = w.y + i32(gid.y);
    var col = lr_o1(x, y);
    let bi = lr_at(pu.rb, x, y);
    let braw = bt[bi];
    let bp = rmp[lr_at(pu.r0, x, y)];
    if (pu.mode.y == 1 && bp > 0.5 * LR_B_MIN) {
        let dx = f32(2 * x + 1 - pu.ints.x) * 0.5;
        let dy = f32(2 * y + 1 - pu.ints.y) * 0.5;
        let rr = sqrt(dx * dx + dy * dy);
        let sigma = lr_swirl_at(rr / pu.frame.z);
        let len = lr_opaque(lr_opaque(lr_opaque(lr_opaque(lr_opaque(sigma * pu.par.y) * LR_PROFILE_SCALE) * bp) * pu.frame.z) * lr_knee(bp));
        if (rr >= 1.0e-9 && len >= LR_MIN_LINE) {
            let ux = -dy / rr;
            let uy = dx / rr;
            let half = clamp(i32(ceil(len)), 1, LR_HALF_TAPS);
            let step = len / f32(half);
            var acc = col;
            var wsum = 1.0;
            for (var side = 0; side < 2; side = side + 1) {
                let sign = select(-1.0, 1.0, side == 1);
                for (var k = 1; k <= half; k = k + 1) {
                    let off = sign * f32(k) * step;
                    let wp = lr_profile(off / len);
                    if (wp <= 0.0) {
                        break;
                    }
                    let ox = off * ux;
                    let oy = off * uy;
                    let fx = floor(ox);
                    let fy = floor(oy);
                    let tx = ox - fx;
                    let ty = oy - fy;
                    let x0 = x + i32(fx);
                    let y0 = y + i32(fy);
                    // The blur at the sample, bilinear between the four pixels around it.
                    let b00 = lr_be(x0, y0);
                    let b10 = lr_be(x0 + 1, y0);
                    let b01 = lr_be(x0, y0 + 1);
                    let b11 = lr_be(x0 + 1, y0 + 1);
                    let btop = b00 + tx * (b10 - b00);
                    let bbot = b01 + tx * (b11 - b01);
                    let bs = btop + ty * (bbot - btop);
                    let wt = wp * lr_sample_w(bs / bp);
                    if (wt <= 0.0) {
                        continue;
                    }
                    let p00 = lr_o1(x0, y0);
                    let p10 = lr_o1(x0 + 1, y0);
                    let p01 = lr_o1(x0, y0 + 1);
                    let p11 = lr_o1(x0 + 1, y0 + 1);
                    let top = p00 + tx * (p10 - p00);
                    let bot = p01 + tx * (p11 - p01);
                    acc = acc + wt * (top + ty * (bot - top));
                    wsum = wsum + wt;
                }
            }
            col = acc / wsum;
        }
    }
    // The balls' light drawn back, in list order, except where a sharp subject (or a strand) has no blur of its own.
    if (!(braw < LR_B_MIN && (info[bi] & 2u) != 0u)) {
        let range = lr_bin(x, y);
        for (var k = range.x; k < range.y; k = k + 1u) {
            col = col + lr_ball_light(bins[k], x, y);
        }
    }
    let at = gid.y * pu.dst.x + gid.x;
    dst[at] = col.x;
    dst[pu.dst.y + at] = col.y;
    dst[2u * pu.dst.y + at] = col.z;
}
