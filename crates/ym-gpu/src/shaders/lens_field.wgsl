// L2, the lens's field optics, on the device (docs/21 section 4, L2; the CPU reference is
// `ym_lens::field`). Three kernels share this module:
//
//   lf_pyramid  builds the box-reduced levels of the input: level 0 is the input over the region the
//               gathers read (rounded outwards to blocks of 2^lmax frame pixels, the frame's edge
//               replicated), level l the mean of the aligned 2 x 2 blocks of level l - 1. A texel is
//               (r, g, b, 0).
//   lf_gather   the per-pixel kernel: the spread function of the pixel's field radius, resampled to
//               n x n cells, turned to the radial direction and scaled to half-width omega, read from
//               the pyramid by bilinear taps. With `dst.z == 1` it also applies the gain and writes the
//               output (no fringe shift); otherwise it writes the gather without the gain.
//   lf_finish   reads each channel of the gather at its fringe shift along the axis (linear
//               interpolation) and applies the gain.
//
// Positions (D4 and D5 of docs/17-gpu.md). A pixel is its absolute column and row (exact integers).
// A tap position is the pixel centre plus a small offset (at most about 90 px), formed in f32 and
// split, for the level of the pyramid, into an exact integer base and a small fraction: no f32 value of
// order 1e4 exists, and every pixel is the same function of its column and row whatever the window
// being rendered. The pyramid's texels are means of aligned blocks of absolute frame coordinates, so
// two windows build equal texels where they overlap.
//
// Bindings: 0 uniform, 1 tables (header and cell tables, `ym_lens::device::FieldTablesData`), 2 pyramid,
// 3 the input planes, 4 the output planes, 5 the gather (read by lf_finish), 6 the picture's pyramid (read by
// the extra light's gather only). With `ex.x` 2 the gather kernel reads the pyramid of the extra light (*As
// built (the headroom of a clipped light)*) at binding 2 and adds it to what the output planes hold, by the
// share of how much the lens changes the light of the pixel with the extra in it.

struct FieldU {
    zero: vec4<u32>,            // x: always 0 (the opaque word of the barrier)
    frame: vec4<f32>,           // frame width, height, half-diagonal in pixels, the table radius scale
    plan: vec4<f32>,            // e', alpha, side softness, fringe boost
    ints: vec4<i32>,            // frame width, height, lmax, 1 when the axis is the buffer's y
    src: vec4<i32>,             // the input: frame column and row of its first pixel, width, height
    srcs: vec4<u32>,            // the input: row stride, plane length
    win: vec4<i32>,             // the rectangle written: frame column and row, width, height
    dst: vec4<u32>,             // output row stride, plane length, 1 to apply the gain here, level to build
    gth: vec4<i32>,             // the gather: frame column and row of its first pixel, width, height
    gs: vec4<u32>,              // the gather: row stride, plane length
    lvl: array<vec4<i32>, 13>,  // per level: offset in the pyramid (vec4 units), width, height
    org: array<vec4<i32>, 13>,  // per level: column and row of its first texel (blocks of that level)
    ex: vec4<u32>,              // x: 2 when the gather is the extra light, added to the output planes; 0 otherwise
};

@group(0) @binding(0) var<uniform> pu: FieldU;
@group(0) @binding(1) var<storage, read> tab: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pyr: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> src: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<storage, read> gat: array<f32>;
@group(0) @binding(6) var<storage, read> pyr_o: array<vec4<f32>>;

// Generated before this text (kernels/lens_field.rs, from ym_lens::consts and ym_lens::device): LF_RADII,
// LF_W_MAX, LF_W_KNEE, LF_COPY_BELOW_PX, LF_MAX_TAPS, LF_SPACING_MAX, LF_HEADER and LF_OFF, the offsets of
// the cell tables of side 3, 5, ..., 17 after the header.

// Hides a value from the optimiser (docs/17-gpu.md K8): an OR with a word that is 0 at run time, which
// is idempotent where an XOR is not.
fn lf_opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu.zero.x);
}

fn lf_radius(s: u32) -> f32 {
    return tab[s / 4u][s % 4u];
}

fn lf_half_width(s: u32) -> f32 {
    return tab[3u + s / 4u][s % 4u];
}

fn lf_illumination(s: u32) -> f32 {
    return tab[6u + s / 4u][s % 4u];
}

// The (x, y) centroid of channel c at radius index s, in table units.
fn lf_centroid(s: u32, c: u32) -> vec2<f32> {
    return tab[9u + s * 3u + c].xy;
}

// The slopes of the monotone cubics through w(rho) and through T(rho) at radius index s.
fn lf_half_width_slope(s: u32) -> f32 {
    return tab[36u + s / 4u][s % 4u];
}

fn lf_illumination_slope(s: u32) -> f32 {
    return tab[39u + s / 4u][s % 4u];
}

// The cubic Hermite curve through (0, y0) and (1, y1) with slopes d0 h and d1 h, at t
// (ym_lens::field::hermite).
fn lf_hermite(y0: f32, y1: f32, d0: f32, d1: f32, h: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * d0
        + (3.0 * t2 - 2.0 * t3) * y1 + (t3 - t2) * h * d1;
}

// The concave C1 curve through (0, y0) and (h, y1) whose slope falls from d0 to d1 (d0 >= (y1 - y0) / h >=
// d1), at x = t h (ym_lens::field::concave_piece): a quadratic when the secant is the mean of the two
// slopes, else a straight run at the slope of one end and a quadratic at the other.
fn lf_concave_piece(y0: f32, y1: f32, d0: f32, d1: f32, h: f32, t: f32) -> f32 {
    let x = t * h;
    let dd = d0 - d1;
    if (dd <= 1.0e-9) {
        return y0 + d0 * x;
    }
    let m = (y1 - y0) / h;
    if (m <= 0.5 * (d0 + d1)) {
        let h1 = clamp(2.0 * h * (m - d1) / dd, 0.0, h);
        if (x <= h1) {
            if (h1 <= 0.0) {
                return y0;
            }
            return y0 + d0 * x - dd * x * x / (2.0 * h1);
        }
        return y0 + 0.5 * h1 * (d0 + d1) + d1 * (x - h1);
    }
    let h2 = clamp(2.0 * h * (d0 - m) / dd, 0.0, h);
    let h1 = h - h2;
    if (x <= h1) {
        return y0 + d0 * x;
    }
    let u = x - h1;
    return y0 + d0 * x - dd * u * u / (2.0 * h2);
}

// The relative illumination T at the table radius rho_t (ym_lens::field::FieldTables::illumination_at):
// the concave curve through the nine values, and beyond 1 the end slope carried straight on.
fn lf_illumination_at(rho_t: f32) -> f32 {
    let last = LF_RADII - 1u;
    if (rho_t > 1.0) {
        return lf_illumination(last) + lf_illumination_slope(last) * (rho_t - 1.0);
    }
    let rho = clamp(rho_t, 0.0, 1.0);
    var s = 0u;
    loop {
        if (s + 2u < LF_RADII && rho > lf_radius(s + 1u)) {
            s = s + 1u;
        } else {
            break;
        }
    }
    let ra = lf_radius(s);
    let rb = lf_radius(s + 1u);
    let t = clamp((rho - ra) / (rb - ra), 0.0, 1.0);
    return lf_concave_piece(lf_illumination(s), lf_illumination(s + 1u), lf_illumination_slope(s),
        lf_illumination_slope(s + 1u), rb - ra, t);
}

// Bends a half-width towards LF_W_MAX without a corner (ym_lens::field::soft_cap): the width up to
// LF_W_KNEE, then K + (W - K) u / (1 + u) for u = (x - K) / (W - K).
fn lf_soft_cap(x: f32) -> f32 {
    if (x <= LF_W_KNEE) {
        return x;
    }
    let r = LF_W_MAX - LF_W_KNEE;
    let u = (x - LF_W_KNEE) / r;
    return LF_W_KNEE + r * u / (1.0 + u);
}

// One pixel's kernel parameters (ym_lens::field::PixelKernel).
struct LfPix {
    gain: f32,
    omega: f32,
    seg: u32,
    t: f32,
    cs: f32,
    sn: f32,
};

fn lf_pixel(i: i32, j: i32) -> LfPix {
    let fw = pu.frame.x;
    let fh = pu.frame.y;
    let h = pu.frame.z;
    let u = (f32(i) + 0.5 - 0.5 * fw) / h;
    let v = (f32(j) + 0.5 - 0.5 * fh) / h;
    var xa = u;
    var ya = v;
    if (pu.ints.w == 1) {
        xa = v;
        ya = u;
    }
    let ax = pu.plan.y * xa;
    // The oval radius as it is: beyond 1 the tables' shapes are held at the last radius, while w carries on
    // by its own law (and T by the end slope, at the radius over the scale).
    let rho_raw = sqrt(ax * ax + ya * ya);
    let rho = clamp(rho_raw, 0.0, 1.0);
    // The segment: the table radius below rho and the blend towards the next one.
    var s = 0u;
    loop {
        if (s + 2u < LF_RADII && rho > lf_radius(s + 1u)) {
            s = s + 1u;
        } else {
            break;
        }
    }
    let ra = lf_radius(s);
    let rb = lf_radius(s + 1u);
    let t = clamp((rho - ra) / (rb - ra), 0.0, 1.0);
    // The gain reads T at the oval radius over the scale; the width reads the oval radius itself.
    let illum = lf_illumination_at(rho_raw / pu.frame.w);
    var w = lf_hermite(lf_half_width(s), lf_half_width(s + 1u), lf_half_width_slope(s),
        lf_half_width_slope(s + 1u), rb - ra, t);
    if (rho_raw > 1.0) {
        let at_edge = LF_RADII - 1u;
        w = w + lf_half_width_slope(at_edge) * (rho_raw - 1.0);
    }
    let gain = 1.0 - pu.plan.x * (1.0 - illum);
    let softness = 1.0 + pu.plan.z * xa * xa;
    let ew = lf_opaque(pu.plan.x * softness);
    let omega = lf_soft_cap(lf_opaque(ew * w)) * h;
    let rr = sqrt(u * u + v * v);
    var cs = 1.0;
    var sn = 0.0;
    if (rr > 1.0e-12) {
        cs = u / rr;
        sn = v / rr;
    }
    var p: LfPix;
    p.gain = gain;
    p.omega = select(omega, 0.0, omega < LF_COPY_BELOW_PX);
    p.seg = s;
    p.t = t;
    p.cs = cs;
    p.sn = sn;
    return p;
}

// Taps per side for a half-width of omega texels.
fn lf_taps(omega: f32) -> u32 {
    return min(u32(2.0 * ceil(omega) + 1.0), LF_MAX_TAPS);
}

// The mip level that brings the cells of an omega-pixel kernel within the spacing of each other.
fn lf_level(omega: f32, lmax: u32) -> u32 {
    var level = 0u;
    var w = omega;
    loop {
        if (level >= 12u) {
            break;
        }
        if (!(2.0 * w / f32(lf_taps(w)) > LF_SPACING_MAX)) {
            break;
        }
        w = w * 0.5;
        level = level + 1u;
    }
    return min(level, lmax);
}

// Bilinear sample of level `lev` at the texel (ix, iy) (absolute, in that level's blocks) and the
// fractions (fx, fy), the texel indices held inside the level.
fn lf_sample(lev: u32, ix: i32, iy: i32, fx: f32, fy: f32) -> vec4<f32> {
    let l = pu.lvl[lev];
    let o = pu.org[lev];
    let cx = clamp(ix - o.x, 0, l.y - 2);
    let cy = clamp(iy - o.y, 0, l.z - 2);
    let at = u32(l.x) + u32(cy) * u32(l.y) + u32(cx);
    let row = u32(l.y);
    let a = pyr[at];
    let b = pyr[at + 1u];
    let c = pyr[at + row];
    let d = pyr[at + row + 1u];
    let top = a + fx * (b - a);
    let bot = c + fx * (d - c);
    return top + fy * (bot - top);
}

// The gather of the pixel (i, j) without the gain.
fn lf_gather_pixel(i: i32, j: i32, k: LfPix) -> vec3<f32> {
    if (k.omega == 0.0) {
        // A copy: the input pixel itself (level 0).
        let l = pu.lvl[0];
        let o = pu.org[0];
        let p = pyr[u32(l.x) + u32(j - o.y) * u32(l.y) + u32(i - o.x)];
        return p.xyz;
    }
    let level = lf_level(k.omega, u32(pu.ints.z));
    let lv = i32(level);
    let inv = 1.0 / f32(1u << level);
    let n = lf_taps(k.omega * inv);
    let nf = f32(n);
    let step = 2.0 * k.omega / nf;
    let c0 = 0.5 * f32(n - 1u);
    // The pixel centre in texels of the level, as an exact integer base and a remainder: texel =
    // (a + 2 d) / 2^(lv + 1) for the offset d in pixels, a = 2 i + 1 - 2^lv.
    let shift = u32(lv + 1);
    let ax = 2 * i + 1 - (1 << level);
    let ay = 2 * j + 1 - (1 << level);
    let qx = ax >> shift;
    let qy = ay >> shift;
    let rx = f32(ax - (qx << shift));
    let ry = f32(ay - (qy << shift));
    let inv2 = 1.0 / f32(1u << shift);
    // The two tables the blend reads, and their weights.
    let base = LF_HEADER + LF_OFF[(n - 3u) / 2u] + k.seg * n * n;
    let t0 = 1.0 - k.t;
    let t1 = k.t;
    var acc = vec3<f32>(0.0);
    for (var row = 0u; row < n; row = row + 1u) {
        let b = (f32(row) - c0) * step;
        var sum = vec4<f32>(0.0);
        for (var col = 0u; col < n; col = col + 1u) {
            let a = (f32(col) - c0) * step;
            // Table x is radial (cs, sn), table y tangential (sn, -cs).
            let dx = a * k.cs + b * k.sn;
            let dy = a * k.sn - b * k.cs;
            let fxt = (rx + 2.0 * dx) * inv2;
            let fyt = (ry + 2.0 * dy) * inv2;
            let flx = floor(fxt);
            let fly = floor(fyt);
            let s = lf_sample(level, qx + i32(flx), qy + i32(fly), fxt - flx, fyt - fly);
            let q = row * n + col;
            let w = t0 * tab[base + q] + t1 * tab[base + n * n + q];
            sum = sum + w * s;
        }
        acc = acc + sum.xyz;
    }
    return acc;
}

// True when the reach of the pixel's gather holds any of the extra light: the coarsest level of its pyramid has
// the means of blocks of 2^lmax pixels of a picture that is never negative, which are 0 over a block with no
// light (the CPU reference tests the pixels themselves, `ym_lens::field::Occupancy`; a pixel skipped here is 0
// there too).
fn lf_has_light(i: i32, j: i32, k: LfPix) -> bool {
    let lmax = u32(pu.ints.z);
    let level = lf_level(k.omega, lmax);
    let reach = i32(ceil(k.omega * 1.4142135)) + i32(ceil(1.5 * f32(1u << level))) + 3;
    let l = pu.lvl[lmax];
    let o = pu.org[lmax];
    let tx0 = max((i - reach) >> lmax, o.x);
    let tx1 = min((i + reach) >> lmax, o.x + l.y - 1);
    let ty0 = max((j - reach) >> lmax, o.y);
    let ty1 = min((j + reach) >> lmax, o.y + l.z - 1);
    for (var ty = ty0; ty <= ty1; ty = ty + 1) {
        for (var tx = tx0; tx <= tx1; tx = tx + 1) {
            let t = pyr[u32(l.x) + u32(ty - o.y) * u32(l.y) + u32(tx - o.x)];
            if (t.x != 0.0 || t.y != 0.0 || t.z != 0.0) {
                return true;
            }
        }
    }
    return false;
}

fn lf_pyramid_main(gid: vec3<u32>) {
    let lev = dst_level();
    let l = pu.lvl[lev];
    if (gid.x >= u32(l.y) || gid.y >= u32(l.z)) {
        return;
    }
    let at = u32(l.x) + gid.y * u32(l.y) + gid.x;
    if (lev == 0u) {
        let o = pu.org[0];
        let fw = pu.ints.x;
        let fh = pu.ints.y;
        // The frame's edge replicated, then the input's (a covering input makes the second a no-op).
        let x = clamp(clamp(o.x + i32(gid.x), 0, fw - 1), pu.src.x, pu.src.x + pu.src.z - 1);
        let y = clamp(clamp(o.y + i32(gid.y), 0, fh - 1), pu.src.y, pu.src.y + pu.src.w - 1);
        let p = u32(y - pu.src.y) * pu.srcs.x + u32(x - pu.src.x);
        pyr[at] = vec4<f32>(src[p], src[pu.srcs.y + p], src[2u * pu.srcs.y + p], 0.0);
        return;
    }
    let pl = pu.lvl[lev - 1u];
    let pw = u32(pl.y);
    let p0 = u32(pl.x) + (2u * gid.y) * pw + 2u * gid.x;
    let a = pyr[p0];
    let b = pyr[p0 + 1u];
    let c = pyr[p0 + pw];
    let d = pyr[p0 + pw + 1u];
    pyr[at] = ((a + b) + (c + d)) * 0.25;
}

fn dst_level() -> u32 {
    return pu.dst.w;
}

fn lf_gather_main(gid: vec3<u32>) {
    if (gid.x >= u32(pu.win.z) || gid.y >= u32(pu.win.w)) {
        return;
    }
    let i = pu.win.x + i32(gid.x);
    let j = pu.win.y + i32(gid.y);
    let k = lf_pixel(i, j);
    let at = gid.y * pu.dst.x + gid.x;
    if (pu.ex.x == 2u) {
        // The extra light: its gather, added to the picture's gather the output planes hold by the share mu of how
        // much the lens changes the light of the pixel with the extra in it (ym_lens::field::add_extra). A pixel
        // with no kernel gets none.
        if (k.omega == 0.0 || !lf_has_light(i, j, k)) {
            return;
        }
        let more = lf_gather_pixel(i, j, k);
        if (all(more == vec3<f32>(0.0))) {
            return;
        }
        let base = vec3<f32>(dst[at], dst[pu.dst.y + at], dst[2u * pu.dst.y + at]);
        // The pixel's own light: the picture's and the extra light's level 0 texels at the pixel.
        let l0 = pu.lvl[0];
        let o0 = pu.org[0];
        let p0 = u32(l0.x) + u32(j - o0.y) * u32(l0.y) + u32(i - o0.x);
        let lit = pyr_o[p0].xyz + pyr[p0].xyz;
        let moved = base + more;
        var change = 0.0;
        for (var c = 0u; c < 3u; c = c + 1u) {
            change = max(change, abs(moved[c] - lit[c]) / max(lit[c], LF_CHANGE_FLOOR));
        }
        let t = clamp((change - LF_CHANGE_FROM) / (LF_CHANGE_TO - LF_CHANGE_FROM), 0.0, 1.0);
        let mu = t * t * (3.0 - 2.0 * t);
        if (mu > 0.0) {
            let sum = base + mu * more;
            dst[at] = sum.x;
            dst[pu.dst.y + at] = sum.y;
            dst[2u * pu.dst.y + at] = sum.z;
        }
        return;
    }
    let px = lf_gather_pixel(i, j, k);
    if (pu.dst.z == 1u) {
        dst[at] = px.x * k.gain;
        dst[pu.dst.y + at] = px.y * k.gain;
        dst[2u * pu.dst.y + at] = px.z * k.gain;
    } else {
        dst[at] = px.x;
        dst[pu.dst.y + at] = px.y;
        dst[2u * pu.dst.y + at] = px.z;
    }
}

// The shift in pixels along the axis at which channel c is read after the gather: the boost times the
// axis part of the channel's centroid offset (0 for a copy).
fn lf_fringe_shift(k: LfPix, c: u32) -> f32 {
    let boost = pu.plan.w;
    if (boost == 0.0 || k.omega == 0.0) {
        return 0.0;
    }
    let a = lf_centroid(k.seg, c);
    let b = lf_centroid(k.seg + 1u, c);
    let cx = (1.0 - k.t) * a.x + k.t * b.x;
    let cy = (1.0 - k.t) * a.y + k.t * b.y;
    var along = cx * k.cs + cy * k.sn;
    if (pu.ints.w == 1) {
        along = cx * k.sn - cy * k.cs;
    }
    return boost * k.omega * along;
}

// The gather's value of channel c at the frame pixel (x, y), held inside the frame and the gathered
// rectangle.
fn lf_gathered(c: u32, x: i32, y: i32) -> f32 {
    let cx = clamp(clamp(x, 0, pu.ints.x - 1), pu.gth.x, pu.gth.x + pu.gth.z - 1);
    let cy = clamp(clamp(y, 0, pu.ints.y - 1), pu.gth.y, pu.gth.y + pu.gth.w - 1);
    return gat[c * pu.gs.y + u32(cy - pu.gth.y) * pu.gs.x + u32(cx - pu.gth.x)];
}

fn lf_finish_main(gid: vec3<u32>) {
    if (gid.x >= u32(pu.win.z) || gid.y >= u32(pu.win.w)) {
        return;
    }
    let i = pu.win.x + i32(gid.x);
    let j = pu.win.y + i32(gid.y);
    let k = lf_pixel(i, j);
    let at = gid.y * pu.dst.x + gid.x;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let shift = lf_fringe_shift(k, c);
        var v: f32;
        if (shift == 0.0) {
            v = lf_gathered(c, i, j);
        } else {
            // Linear interpolation between the two pixels around the shifted position; the integer
            // part is added as an integer so no f32 holds a position of order 1e4.
            let fl = floor(shift);
            let f = shift - fl;
            let n = i32(fl);
            var a: f32;
            var b: f32;
            if (pu.ints.w == 1) {
                a = lf_gathered(c, i, j + n);
                b = lf_gathered(c, i, j + n + 1);
            } else {
                a = lf_gathered(c, i + n, j);
                b = lf_gathered(c, i + n + 1, j);
            }
            v = a + f * (b - a);
        }
        dst[c * pu.dst.y + at] = v * k.gain;
    }
}
