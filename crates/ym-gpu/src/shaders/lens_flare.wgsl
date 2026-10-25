// L3, the lens's highlight optics, on the device (docs/21 section 4, L3; the CPU reference is
// `ym_lens::l3`). Three kernels share this module:
//
//   lfl_restore  the restored light of the input: the light's colour (a pixel left at the clip pattern made
//                neutral, a saturated channel raised to the mean of the others; lens_light_colour.wgsl) times
//                1 + f (2^h - 1), f the clip test, as (r, g, b, 0) over the rectangle the glow core reads.
//   lfl_nodes    the nodes of one grid of the ghosts that vary slowly: the sum, in list order, of the
//                densities of the grid's ghosts at the node's place (a pixel corner, 2, 4 or 8 px apart on
//                the absolute pixel grid), as (r, g, b, 0).
//   lfl_main     one output pixel: out = keep Z + F (core + tail + items + grids), the sum in the CPU's
//                order: the glow core gathered from the restored light (the taps of the scale), the tail
//                plane read with a Catmull-Rom curve and held at 0 or above, then the drawn items (ghosts
//                that are drawn at every pixel, starbursts and streaks) of the pixel's bin in list order,
//                then the grids read by bilinear interpolation.
//
// Positions (D4 and D5 of docs/17-gpu.md). A pixel is its absolute column and row (exact integers); its
// field coordinates are (i + 0.5 - w / 2) / H in f32, the same function of the column and row whatever
// the window being rendered. The items come from the CPU reference's own item list, culled to the pixel
// by bins of the absolute pixel grid, so the sum of a pixel has the same terms in the same order in a tile
// and in the whole frame.
//
// Bindings of lfl_main: 0 uniform, 1 the restored light, 2 Z (planar), 3 the output (planar), 4 the photo's
// planes (tail plane, then the spike tables), 5 the scale's data (header, core taps, items), 6 the bins,
// 7 the lens's tables (rim, aperture, normal tail), 8 the grids' nodes. Bindings of lfl_restore: 0 uniform,
// 2 the input's planes (the same slot as Z: the uniform's `zr` and `zs` describe whichever image it is),
// 8 the restored light it writes. Bindings of lfl_nodes: 0 uniform, 5, 6, 7 as above, 8 the nodes it writes.

struct FlareU {
    zero: vec4<u32>,        // x: always 0 (the opaque word of the barrier)
    frame: vec4<f32>,       // frame width, height, half-diagonal in pixels, 0
    mix: vec4<f32>,         // F, keep, the spike table's radius, the tail plane's half-diagonal
    ints: vec4<i32>,        // frame width, height, 1 when the axis is the buffer's y, core radius (-1: no core)
    outr: vec4<i32>,        // the rectangle written: frame column and row, width, height
    resr: vec4<i32>,        // the restored light: frame column and row of its first pixel, width, height
    zr: vec4<i32>,          // Z: frame column and row of its first pixel, width, height
    zs: vec4<u32>,          // Z row stride, plane length, output row stride, plane length
    offs: vec4<u32>,        // core taps offset, items offset, item count, 1 with a streak profile
    bins: vec4<u32>,        // bins across, bins down, bin side in pixels, offset of the bin item lists
    tail: vec4<i32>,        // tail plane width, height, offset of the spike tables in the planes buffer, 0
    clip: vec4<f32>,        // g0 of each channel, 2^h - 1
    ramp: vec4<f32>,        // the clip test's ramp: from, to
    lg: array<vec4<i32>, 5>,  // the grids of spacing 2, 4, 8, 16, 32: first node column and row, nodes across, down
    lgo: array<vec4<u32>, 2>, // where each grid starts in the node buffer (vectors): grids 0 to 3, then grid 4,
                              // the bit set of the grids in use
    lvl: vec4<u32>,         // the grid lfl_nodes fills (0 to 4)
};

@group(0) @binding(0) var<uniform> pu: FlareU;
@group(0) @binding(1) var<storage, read> res: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> zin: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read> planes: array<f32>;
@group(0) @binding(5) var<storage, read> scale: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bins: array<u32>;
@group(0) @binding(7) var<storage, read> tab: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> aux: array<vec4<f32>>;

// Generated before this text (kernels/lens_flare.rs): FL_SPIKE, FL_TAB_*, FL_NODES, FL_MAX_NORMALS and the
// constants of the lens plan.

// ---------------------------------------------------------------------------------------------------------
// The restored light

fn lfl_restore_main(gid: vec3<u32>) {
    if (gid.x >= u32(pu.resr.z) || gid.y >= u32(pu.resr.w)) {
        return;
    }
    let x = pu.resr.x + i32(gid.x);
    let y = pu.resr.y + i32(gid.y);
    let p = u32(y - pu.zr.y) * pu.zs.x + u32(x - pu.zr.x);
    let v = vec3<f32>(zin[p], zin[pu.zs.y + p], zin[2u * pu.zs.y + p]);
    // The factor reads the pixel as it is; the colour is the light's (ym_lens Restore::light_colour: the clip
    // pattern's cast taken off, a saturated channel raised to the mean of the others).
    var m = 1.0e30;
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (pu.clip[c] > 0.0) {
            m = min(m, v[c] / pu.clip[c]);
        }
    }
    var factor = 1.0;
    if (m < 1.0e29) {
        let f = clamp((m - pu.ramp.x) / (pu.ramp.y - pu.ramp.x), 0.0, 1.0);
        factor = 1.0 + f * pu.clip.w;
    }
    let base = ll_light_colour(v, pu.clip.xyz, pu.ramp.xy);
    aux[gid.y * u32(pu.resr.z) + gid.x] = vec4<f32>(base * factor, 0.0);
}

// ---------------------------------------------------------------------------------------------------------
// The tail plane

fn lfl_catmull(t: f32, n: i32) -> array<f32, 4> {
    let fl = floor(t);
    let f = t - fl;
    let f2 = f * f;
    let f3 = f * f * f;
    return array<f32, 4>(
        0.5 * (-f3 + 2.0 * f2 - f),
        0.5 * (3.0 * f3 - 5.0 * f2 + 2.0),
        0.5 * (-3.0 * f3 + 4.0 * f2 + f),
        0.5 * (f3 - f2),
    );
}

fn lfl_catmull_index(t: f32, n: i32, k: i32) -> u32 {
    let i0 = i32(floor(t));
    return u32(clamp(i0 + k - 1, 0, n - 1));
}

fn lfl_tail(i: i32, j: i32) -> vec3<f32> {
    let tw = pu.tail.x;
    let th = pu.tail.y;
    let half_w = 0.5 * pu.frame.x;
    let half_h = 0.5 * pu.frame.y;
    let u = (f32(i) + 0.5 - half_w) / pu.frame.z;
    let v = (f32(j) + 0.5 - half_h) / pu.frame.z;
    let tx = u * pu.mix.w + 0.5 * f32(tw) - 0.5;
    let ty = v * pu.mix.w + 0.5 * f32(th) - 0.5;
    var cw = lfl_catmull(tx, tw);
    var rw = lfl_catmull(ty, th);
    var out = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let plane = c * u32(tw * th);
        var acc = 0.0;
        for (var ry = 0; ry < 4; ry = ry + 1) {
            let base = plane + lfl_catmull_index(ty, th, ry) * u32(tw);
            var s = 0.0;
            for (var rx = 0; rx < 4; rx = rx + 1) {
                s = s + cw[rx] * planes[base + lfl_catmull_index(tx, tw, rx)];
            }
            acc = acc + rw[ry] * s;
        }
        out[c] = max(acc, 0.0);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------
// The glow core

fn lfl_core(i: i32, j: i32) -> vec3<f32> {
    let r = pu.ints.w;
    let side = u32(2 * r + 1);
    let dx_lo = max(-r, -i);
    let dx_hi = min(r, pu.ints.x - 1 - i);
    let dy_lo = max(-r, -j);
    let dy_hi = min(r, pu.ints.y - 1 - j);
    var sum = vec4<f32>(0.0);
    for (var dy = dy_lo; dy <= dy_hi; dy = dy + 1) {
        let row = u32(j + dy - pu.resr.y) * u32(pu.resr.z);
        let wrow = u32(dy + r) * side;
        for (var dx = dx_lo; dx <= dx_hi; dx = dx + 1) {
            let w = scale[pu.offs.x + wrow + u32(dx + r)];
            let p = res[row + u32(i + dx - pu.resr.x)];
            sum = sum + w * p;
        }
    }
    return sum.xyz;
}

// ---------------------------------------------------------------------------------------------------------
// Tables: the rim curve, the aperture, the normal tail

fn lfl_rim(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0) * 15.0;
    let i = min(u32(x), 14u);
    let u = x - f32(i);
    let a = 1.0 - u;
    let b = u;
    let y0 = tab[i / 4u][i % 4u];
    let y1 = tab[(i + 1u) / 4u][(i + 1u) % 4u];
    let m0 = tab[FL_TAB_RIM_M + i / 4u][i % 4u];
    let m1 = tab[FL_TAB_RIM_M + (i + 1u) / 4u][(i + 1u) % 4u];
    let h = 1.0 / 15.0;
    let v = a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * (h * h / 6.0);
    return max(v, 0.0);
}

// A node of the normal tail's table: (Q, dQ/dz, R, dR/dz).
fn lfl_node(i: u32) -> vec4<f32> {
    return tab[FL_TAB_NODES + i];
}

// Cubic Hermite interpolation of two of a node's pairs (offset 0: Q, 2: the ramp).
fn lfl_hermite(z: f32, which: u32) -> f32 {
    let p = (z + 8.0) * 256.0;
    let i = min(u32(p), FL_NODES - 2u);
    let t = p - f32(i);
    let a = lfl_node(i);
    let b = lfl_node(i + 1u);
    var a0 = a.x;
    var a1 = a.y;
    var b0 = b.x;
    var b1 = b.y;
    if (which == 1u) {
        a0 = a.z;
        a1 = a.w;
        b0 = b.z;
        b1 = b.w;
    }
    let h = 1.0 / 256.0;
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * a0
        + (t3 - 2.0 * t2 + t) * h * a1
        + (-2.0 * t3 + 3.0 * t2) * b0
        + (t3 - t2) * h * b1;
}

// Q(z): 1 far below 0, 0 far above.
fn lfl_q(z: f32) -> f32 {
    if (z >= 8.0) {
        return 0.0;
    }
    if (z <= -8.0) {
        return 1.0;
    }
    return lfl_hermite(z, 0u);
}

// The ramp max(z, 0) smoothed by the standard normal.
fn lfl_ramp(z: f32) -> f32 {
    if (z >= 8.0) {
        return z;
    }
    if (z <= -8.0) {
        return 0.0;
    }
    return lfl_hermite(z, 1u);
}

// A shape as a gauge: the aperture polygon blended with its inscribed circle, or a circle.
fn lfl_gauge(shape: i32, x: f32, y: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    let head = tab[FL_TAB_APERTURE];
    let n = u32(head.x);
    if (shape == 1 || n == 0u) {
        return r;
    }
    var m = -1.0e30;
    for (var k = 0u; k < n; k = k + 1u) {
        let v = tab[FL_TAB_APERTURE + 1u + k / 2u];
        var nx = v.x;
        var ny = v.y;
        if ((k & 1u) == 1u) {
            nx = v.z;
            ny = v.w;
        }
        m = max(m, nx * x + ny * y);
    }
    return (1.0 - head.y) * m + head.y * r;
}

// A ghost's radial profile at the normalised radius t (1 on the edge).
fn lfl_profile(t: f32, sigma_g: f32) -> f32 {
    let z = (1.0 - t) / sigma_g;
    var arg = t;
    if (z <= -8.0) {
        arg = 1.0;
    } else if (z < 8.0) {
        arg = 1.0 - sigma_g * lfl_ramp(z);
    }
    return lfl_rim(arg) * lfl_q(-z);
}

// ---------------------------------------------------------------------------------------------------------
// The items

fn lfl_ghost(rec: u32, shape: i32, a: f32, c: f32) -> vec3<f32> {
    let v0 = scale[rec];
    let v1 = scale[rec + 1u];
    let v2 = scale[rec + 2u];
    let v3 = scale[rec + 3u];
    let x = (a - v0.x) / v1.z;
    let y = c - v0.y;
    let gauge = lfl_gauge(shape, x, y) / v1.y;
    if (gauge > v2.y) {
        return vec3<f32>(0.0);
    }
    let prof = lfl_profile(gauge, v1.w);
    let dx = a - v0.z;
    let dy = c - v0.w;
    let d = sqrt(dx * dx + dy * dy);
    // Brighter on the side that faces the optical axis: 1 + beta (x ux + y uy) / s, held at 0 or above.
    let shade = max(1.0 + v3.w * (x * v3.y + y * v3.z) / v1.y, 0.0);
    let dens = prof * shade * lfl_q((d - v1.x) / v2.x);
    if (dens > 0.0) {
        return vec3<f32>(v2.z, v2.w, v3.x) * dens;
    }
    return vec3<f32>(0.0);
}

// The integral of the tent max(0, 1 - |t|) from minus infinity to t.
fn lfl_tent(t: f32) -> f32 {
    if (t <= -1.0) {
        return 0.0;
    }
    if (t <= 0.0) {
        return 0.5 * (t + 1.0) * (t + 1.0);
    }
    if (t <= 1.0) {
        return 1.0 - 0.5 * (1.0 - t) * (1.0 - t);
    }
    return 1.0;
}

// The tent weights of a pixel's interval [e0, e1] over the cells of a spike table along one direction, for
// the cell m: the index of the first cell, and the weight of cell m.
fn lfl_tent_first(e0: f32, centre: f32, delta: f32, r: f32) -> i32 {
    let x0 = (e0 - centre + r) / delta;
    return clamp(i32(floor(x0 - 1.5)) + 1, 0, i32(FL_SPIKE));
}

fn lfl_tent_last(e1: f32, centre: f32, delta: f32, r: f32) -> i32 {
    let x1 = (e1 - centre + r) / delta;
    return clamp(i32(ceil(x1 + 0.5)), 0, i32(FL_SPIKE));
}

fn lfl_tent_weight(m: i32, e0: f32, e1: f32, centre: f32, delta: f32, r: f32) -> f32 {
    let x0 = (e0 - centre + r) / delta;
    let x1 = (e1 - centre + r) / delta;
    let c = f32(m) + 0.5;
    return lfl_tent(x1 - c) - lfl_tent(x0 - c);
}

// The edges of the pixel (p) along an axis in field coordinates.
fn lfl_edge(p: i32, size: f32) -> f32 {
    return (f32(p) - 0.5 * size) / pu.frame.z;
}

fn lfl_burst(rec: u32, i: i32, j: i32) -> vec3<f32> {
    let v0 = scale[rec];
    let v1 = scale[rec + 1u];
    let r = pu.mix.z;
    let delta = 2.0 * r / f32(FL_SPIKE);
    // Along the axis and across it.
    var pa = i;
    var pc = j;
    var sa = pu.frame.x;
    var sc = pu.frame.y;
    if (pu.ints.z == 1) {
        pa = j;
        pc = i;
        sa = pu.frame.y;
        sc = pu.frame.x;
    }
    let a0 = lfl_edge(pa, sa);
    let a1 = lfl_edge(pa + 1, sa);
    let c0 = lfl_edge(pc, sc);
    let c1 = lfl_edge(pc + 1, sc);
    let m_lo = lfl_tent_first(a0, v0.x, delta, r);
    let m_hi = lfl_tent_last(a1, v0.x, delta, r);
    let n_lo = lfl_tent_first(c0, v0.y, delta, r);
    let n_hi = lfl_tent_last(c1, v0.y, delta, r);
    var mass = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let base = u32(pu.tail.z) + ch * FL_SPIKE * FL_SPIKE;
        var total = 0.0;
        for (var m = m_lo; m < m_hi; m = m + 1) {
            // The table collapsed over the pixel's rows (across the axis), at column m.
            var vm = 0.0;
            for (var n = n_lo; n < n_hi; n = n + 1) {
                let wn = lfl_tent_weight(n, c0, c1, v0.y, delta, r);
                vm = vm + wn * planes[base + u32(n) * FL_SPIKE + u32(m)];
            }
            total = total + lfl_tent_weight(m, a0, a1, v0.x, delta, r) * vm;
        }
        mass[ch] = total;
    }
    let per_area = pu.frame.z * pu.frame.z;
    return vec3<f32>(v0.z, v0.w, v1.x) * mass * per_area;
}

// The Gauss-Legendre rule of 8 points on [a, b] for the streak's height.
fn lfl_height(y: f32, core: f32, power: f32, length: f32) -> f32 {
    let x = y / length;
    var win = 1.0;
    if (x >= 1.0) {
        win = 0.0;
    } else if (x > FL_STREAK_FLAT) {
        win = 0.5 * (1.0 + cos(3.14159265358979 * (x - FL_STREAK_FLAT) / (1.0 - FL_STREAK_FLAT)));
    }
    return pow(1.0 + y / core, -power) * win;
}

fn lfl_gauss8(a: f32, b: f32, core: f32, power: f32, length: f32) -> f32 {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    var s = 0.0;
    s = s + 0.362683783378362 * (lfl_height(mid - half * 0.1834346424956498, core, power, length) + lfl_height(mid + half * 0.1834346424956498, core, power, length));
    s = s + 0.3137066458778873 * (lfl_height(mid - half * 0.525532409916329, core, power, length) + lfl_height(mid + half * 0.525532409916329, core, power, length));
    s = s + 0.2223810344533745 * (lfl_height(mid - half * 0.7966664774136267, core, power, length) + lfl_height(mid + half * 0.7966664774136267, core, power, length));
    s = s + 0.1012285362903763 * (lfl_height(mid - half * 0.9602898564975363, core, power, length) + lfl_height(mid + half * 0.9602898564975363, core, power, length));
    return s * half;
}

// The mass of the streak's half profile from 0 to x >= 0 (ym_lens AlongProfile::half_cdf).
fn lfl_half_cdf(x: f32) -> f32 {
    let p0 = scale[0];
    let p1 = scale[1];
    let core = p0.x;
    let power = p0.y;
    let length = p0.z;
    let flat = p0.w;
    if (x <= flat) {
        let u = 1.0 + x / core;
        if (abs(power - 1.0) < 1.0e-6) {
            return core * log(u);
        }
        return core * (1.0 - pow(u, 1.0 - power)) / (power - 1.0);
    }
    if (x >= length) {
        return p1.y;
    }
    let width = length - flat;
    let t = (x - flat) / width * 64.0;
    let i = min(u32(t), 63u);
    let a = flat + width * f32(i) / 64.0;
    let part = lfl_gauss8(a, x, core, power, length);
    let cut = scale[3u + i / 4u][i % 4u];
    return p1.x + cut + part;
}

fn lfl_cdf(a: f32) -> f32 {
    let share = lfl_half_cdf(abs(a)) / scale[1].y;
    if (a >= 0.0) {
        return 0.5 + 0.5 * share;
    }
    return 0.5 - 0.5 * share;
}

// The mass of the streak's profile between a0 < a1 (in units of the light's reach): cdf(a1) - cdf(a0). Where the
// interval is small and on one side of the light, the difference of two cumulative values in f32 loses digits
// (each is good to 6e-8 of itself, the pixel's mass can be 1e-4), so the mass is integrated from the height
// instead (8 Gauss-Legendre points, ym_lens AlongProfile::cdf is the same number in f64).
fn lfl_mass(a0: f32, a1: f32) -> f32 {
    let p0 = scale[0];
    let width = a1 - a0;
    if (width < 0.1 * p0.x && (a0 >= 0.0 || a1 <= 0.0)) {
        let x0 = min(abs(a0), abs(a1));
        let x1 = max(abs(a0), abs(a1));
        return 0.5 * lfl_gauss8(x0, x1, p0.x, p0.y, p0.z) / scale[1].y;
    }
    return lfl_cdf(a1) - lfl_cdf(a0);
}

fn lfl_streak(rec: u32, i: i32, j: i32) -> vec3<f32> {
    let v0 = scale[rec];
    let v1 = scale[rec + 1u];
    let v2 = scale[rec + 2u];
    var pa = i;
    var pc = j;
    if (pu.ints.z == 1) {
        pa = j;
        pc = i;
    }
    // The pixel's edges relative to the light, in half-diagonals. The light's place (v0.x along the axis, v0.y
    // across it) is in pixels of this frame, so the two edges of a pixel are off by the same amount (which moves
    // the whole streak by 1e-4 px at 50 MP) and the pixel's width stays 1 / half-diagonal to 1e-6; edges taken
    // from the frame's centre in f32 would each be 3e-4 px off, 5e-4 of a pixel's width.
    let a0 = (f32(pa) - v0.x) / pu.frame.z;
    let a1 = (f32(pa + 1) - v0.x) / pu.frame.z;
    let c0 = (f32(pc) - v0.y) / pu.frame.z;
    let c1 = (f32(pc + 1) - v0.y) / pu.frame.z;
    // Across the axis: the core (Gaussian of v0.z) and the wider halo that holds FL_STREAK_HALO_SHARE of the light.
    let wide = FL_STREAK_HALO_WIDTH * v0.z;
    let mc = (1.0 - FL_STREAK_HALO_SHARE) * (lfl_q(c0 / v0.z) - lfl_q(c1 / v0.z))
        + FL_STREAK_HALO_SHARE * (lfl_q(c0 / wide) - lfl_q(c1 / wide));
    if (mc <= 0.0) {
        return vec3<f32>(0.0);
    }
    // Along it: the attachment's profile stretched to the light's reach (v1.z); the colour turns from the light's
    // (v0.w, v1.x, v1.y) to the tail's (v1.w, v2.x, v2.y) over FL_STREAK_TURN of the reach, a smooth step.
    let ma = lfl_mass(a0 / v1.z, a1 / v1.z);
    let turn = clamp(abs(0.5 * (a0 + a1)) / (FL_STREAK_TURN * scale[0].z * v1.z), 0.0, 1.0);
    let w = turn * turn * (3.0 - 2.0 * turn);
    let per_area = pu.frame.z * pu.frame.z;
    return mix(vec3<f32>(v0.w, v1.x, v1.y), vec3<f32>(v1.w, v2.x, v2.y), w) * (ma * mc * per_area);
}

fn lfl_items(i: i32, j: i32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    if (pu.offs.z == 0u) {
        return acc;
    }
    let bin = u32(j) / pu.bins.z * pu.bins.x + u32(i) / pu.bins.z;
    let first = bins[bin];
    let last = bins[bin + 1u];
    let u = (f32(i) + 0.5 - 0.5 * pu.frame.x) / pu.frame.z;
    let v = (f32(j) + 0.5 - 0.5 * pu.frame.y) / pu.frame.z;
    var a = u;
    var c = v;
    if (pu.ints.z == 1) {
        a = v;
        c = u;
    }
    for (var t = first; t < last; t = t + 1u) {
        let k = bins[pu.bins.w + t];
        let rec = pu.offs.y + 6u * k;
        let span = bitcast<vec4<i32>>(scale[rec + 5u]);
        if (i < span.x || i >= span.y || j < span.z || j >= span.w) {
            continue;
        }
        let kind = bitcast<vec4<i32>>(scale[rec + 4u]);
        if (kind.x == 0) {
            // A ghost on a grid (kind.z above 1) is in the nodes.
            if (kind.z <= 1) {
                acc = acc + lfl_ghost(rec, kind.y, a, c);
            }
        } else if (kind.x == 1) {
            acc = acc + lfl_burst(rec, i, j);
        } else {
            acc = acc + lfl_streak(rec, i, j);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------------------------------------
// The grids of the ghosts that vary slowly

fn lfl_floor_div(a: i32, b: i32) -> i32 {
    var q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0))) {
        q = q - 1;
    }
    return q;
}

fn lfl_grid(li: u32) -> vec4<i32> {
    return pu.lg[li];
}

fn lfl_grid_offset(li: u32) -> u32 {
    return pu.lgo[li / 4u][li % 4u];
}

// One node of the grid pu.lvl.x: the sum, in list order, of the densities of the grid's ghosts at the node's
// place. A node (j, k) of the grid of spacing l is the pixel corner (l j + l / 2, l k + l / 2); a ghost
// counts when its pixel rectangle grown by l holds the node. The node's bin is the bin of the nearest
// pixel of the frame (the bins hold the rectangles grown by the largest spacing).
fn lfl_nodes_main(gid: vec3<u32>) {
    let li = pu.lvl.x;
    let g = lfl_grid(li);
    if (gid.x >= u32(g.z) || gid.y >= u32(g.w)) {
        return;
    }
    let l = i32(2u << li);
    let px = l * (g.x + i32(gid.x)) + l / 2;
    let py = l * (g.y + i32(gid.y)) + l / 2;
    var acc = vec3<f32>(0.0);
    if (pu.offs.z != 0u) {
        let bx = u32(clamp(px, 0, pu.ints.x - 1)) / pu.bins.z;
        let by = u32(clamp(py, 0, pu.ints.y - 1)) / pu.bins.z;
        let bin = by * pu.bins.x + bx;
        let first = bins[bin];
        let last = bins[bin + 1u];
        let u = (f32(px) - 0.5 * pu.frame.x) / pu.frame.z;
        let v = (f32(py) - 0.5 * pu.frame.y) / pu.frame.z;
        var a = u;
        var c = v;
        if (pu.ints.z == 1) {
            a = v;
            c = u;
        }
        for (var t = first; t < last; t = t + 1u) {
            let k = bins[pu.bins.w + t];
            let rec = pu.offs.y + 6u * k;
            let kind = bitcast<vec4<i32>>(scale[rec + 4u]);
            if (kind.x != 0 || kind.z != l) {
                continue;
            }
            let span = bitcast<vec4<i32>>(scale[rec + 5u]);
            if (px < span.x - l || px >= span.y + l || py < span.z - l || py >= span.w + l) {
                continue;
            }
            acc = acc + lfl_ghost(rec, kind.y, a, c);
        }
    }
    aux[lfl_grid_offset(li) + gid.y * u32(g.z) + gid.x] = vec4<f32>(acc, 0.0);
}

// The grids' light at the pixel (i, j): each grid in use read by bilinear interpolation between the four
// nodes around the pixel's centre.
fn lfl_layer(i: i32, j: i32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var li = 0u; li < 5u; li = li + 1u) {
        if ((pu.lgo[1].y & (1u << li)) == 0u) {
            continue;
        }
        let g = lfl_grid(li);
        let l = i32(2u << li);
        let nx = 2 * i + 1 - l;
        let ny = 2 * j + 1 - l;
        let jn = lfl_floor_div(nx, 2 * l);
        let kn = lfl_floor_div(ny, 2 * l);
        let tx = f32(nx - 2 * l * jn) / f32(2 * l);
        let ty = f32(ny - 2 * l * kn) / f32(2 * l);
        let w = u32(g.z);
        let base = lfl_grid_offset(li) + u32(kn - g.y) * w + u32(jn - g.x);
        let n00 = aux[base].xyz;
        let n10 = aux[base + 1u].xyz;
        let n01 = aux[base + w].xyz;
        let n11 = aux[base + w + 1u].xyz;
        let top = n00 + tx * (n10 - n00);
        let bot = n01 + tx * (n11 - n01);
        acc = acc + (top + ty * (bot - top));
    }
    return acc;
}

fn lfl_main_main(gid: vec3<u32>) {
    if (gid.x >= u32(pu.outr.z) || gid.y >= u32(pu.outr.w)) {
        return;
    }
    let i = pu.outr.x + i32(gid.x);
    let j = pu.outr.y + i32(gid.y);
    var acc = vec3<f32>(0.0);
    if (pu.ints.w >= 0) {
        acc = lfl_core(i, j);
    }
    acc = acc + lfl_tail(i, j);
    acc = acc + lfl_items(i, j);
    if (pu.lgo[1].y != 0u) {
        acc = acc + lfl_layer(i, j);
    }
    let zp = u32(j - pu.zr.y) * pu.zs.x + u32(i - pu.zr.x);
    let at = gid.y * pu.zs.z + gid.x;
    let keep = pu.mix.y;
    let f = pu.mix.x;
    dst[at] = keep * zin[zp] + f * acc.x;
    dst[pu.zs.w + at] = keep * zin[pu.zs.y + zp] + f * acc.y;
    dst[2u * pu.zs.w + at] = keep * zin[2u * pu.zs.y + zp] + f * acc.z;
}
