// The Face Refinement node on the device (docs/27-skin-retouch.md sections 6.2 and 7, S-13): the per-pixel apply of the cached
// blemish map and the low-frequency operator of dark circles, even skin, skin grading and teeth whitening, over one region
// of the working image. The CPU twin is `ym-retouch` (`apply_blemish_toned`, `retouch_tones_at`: the per-pixel maths in
// `lowfreq.rs` and `blemish.rs`, the tone relation in `tone.rs`); this is the same arithmetic in f32 where the CPU uses f64.
//
// One kernel, one entry point, an `op` in the uniform (the pattern of `mask_raster.wgsl`). Every op is a short, separate
// dispatch over a region, recorded by `kernels::face` in the CPU's order, so a pixel's value is the same expression
// whichever tile the region is met in (the region is always made whole). The box filter is a sliding sum that restarts at
// every 64th output (the direct sum of the window), so the f32 drift of a long row is bounded; nothing relies on NaN or
// infinity: a value is tested before it is used.
//
// Bindings: 0 the uniform; 1 `fc_pic`, the working image of the region (linear AWG4: three planes and a fourth that says
// whether a tool changed the pixel), read_write; 2 `fc_lay`, the layer scratch (dense planes), read_write; 3 `fc_src`, the
// node's input, LogC4 / AWG4, planar (only DECODE reads it); 4 `fc_out`, the node's picture over the area the render made
// (only ENCODE writes it), read_write; 5 `fc_bytes`, the masks and the protection plane as bytes, four to a word; 6 `fc_fd`,
// the blemish maps and the tone curves, read_write; 7 `fc_words`, the raster planes of the protection (u16 codes, two to a
// word).
//
// Ops:
//    0 DECODE     fc_pic = logc4_decode(fc_src) over the region, the changed flag cleared.
//    1 BLEMISH    section 6.2 for one face: the map bilinear at the pixel, the strength, the gain (with the tone relation).
//    2 DOWN       a layer plane: the mean of the ds x ds block of a plane of fc_pic or of a mask (an odd edge averages what exists).
//    3 BOX        one box filter of radius q along rows or columns, edges replicated, over some planes of the layer.
//    4 MUL        dst = x * m (the planes of x by one plane m).
//    5 MCOMB      the mask-normalised blur's last step: m num / max(den, 1/255) + (1 - m) x.
//    6 YPLANE     the mean of three planes.
//    7 LOWFINAL   the per-pixel part of section 7.2 to 7.7 over the padded region: dark circles, even skin, grading.
//    8 TSELECT    teeth whitening's L_max: the nearest-rank 99th percentile of the mean over the mask's pixels (one workgroup).
//    9 TAPPLY     teeth whitening (7.8) per pixel.
//   10 ENCODE     fc_out = logc4_encode(fc_pic) where the flag says a tool changed the pixel; a picture of zeros elsewhere.
//   11 TONE       the tone relation (tone.rs) of the region's pixels against the analysis picture's quantiles (a workgroup a channel).
//   12 COPY       a region's picture (fc_src, three planes of the region's size) into the node's picture (fc_out) at the region's place; a
//                 pixel of zeros is not written (the node's picture is zeros there already).

struct FU {
    op: vec4<u32>,    // x: the op; y: (LOWFINAL) the length of a layer plane
    pic: vec4<u32>,   // the working image: row stride, plane length, width, height
    src: vec4<u32>,   // the input image: row stride, plane length, column and row of the region's first pixel in it
    outi: vec4<i32>,  // the output image: row stride, plane length, column and row of the region's first pixel in it
    outs: vec4<i32>,  // the output image: width, height
    prot: vec4<u32>,  // the protection: x: 0 none, 1 raster codes, 2 bytes; y: row stride; z, w: column and row of the region's first pixel in the plane
    pro2: vec4<u32>,  // x: the first word of the raster plane, or the first byte of the byte plane
    a: vec4<i32>,
    b: vec4<i32>,
    c: vec4<i32>,
    d: vec4<i32>,
    f: vec4<f32>,
    g: vec4<f32>,
    h: vec4<f32>,
};

@group(0) @binding(0) var<uniform> FU_: FU;
@group(0) @binding(1) var<storage, read_write> fc_pic: array<f32>;
@group(0) @binding(2) var<storage, read_write> fc_lay: array<f32>;
@group(0) @binding(3) var<storage, read> fc_src: array<f32>;
@group(0) @binding(4) var<storage, read_write> fc_out: array<f32>;
@group(0) @binding(5) var<storage, read> fc_bytes: array<u32>;
@group(0) @binding(6) var<storage, read_write> fc_fd: array<f32>;
@group(0) @binding(7) var<storage, read> fc_words: array<u32>;

const FC_SEG: u32 = 64u;
const FC_TONE_FLOOR: f32 = 1.0e-4;
const FC_BINS: u32 = 32u;
const FC_BLOCK: u32 = 100u;       // a tone channel's block in fc_fd: status, knots, slopes, ln A, ln X, the analysis' quantiles
const FC_MIN_PAIRS: u32 = 256u;
const FC_SLOPE_TOL: f32 = 0.04;

fn fc_finite(x: f32) -> bool {
    return abs(x) < 3.0e38;
}

// ---- addressing -----------------------------------------------------------------------------------------

fn fc_at(c: u32, x: u32, y: u32) -> u32 {
    return c * FU_.pic.y + y * FU_.pic.x + x;
}

fn fc_byte(base: u32, i: u32) -> u32 {
    let at = base + i;
    return (fc_bytes[at >> 2u] >> (8u * (at & 3u))) & 255u;
}

// The protection plane's byte (0 to 255; 255 leaves the pixel alone) at a pixel of the region.
fn fc_protect(x: u32, y: u32) -> u32 {
    if (FU_.prot.x == 0u) {
        return 0u;
    }
    if (FU_.prot.x == 1u) {
        let e = (FU_.prot.w + y) * FU_.prot.y + FU_.prot.z + x;
        let code = (fc_words[FU_.pro2.x + (e >> 1u)] >> (16u * (e & 1u))) & 65535u;
        return (code * 255u + 32767u) / 65535u;
    }
    return fc_byte(FU_.pro2.x, y * FU_.prot.y + x);
}

fn fc_keep(x: u32, y: u32) -> f32 {
    return 1.0 - f32(fc_protect(x, y)) / 255.0;
}

// ---- 0 DECODE -------------------------------------------------------------------------------------------

fn fc_decode(gid: vec3<u32>) {
    if (gid.x >= FU_.pic.z || gid.y >= FU_.pic.w) {
        return;
    }
    let s = (FU_.src.w + gid.y) * FU_.src.x + FU_.src.z + gid.x;
    let v = vec3<f32>(fc_src[s], fc_src[FU_.src.y + s], fc_src[2u * FU_.src.y + s]);
    let l = logc4_decode3(v);
    let i = fc_at(0u, gid.x, gid.y);
    fc_pic[i] = l.x;
    fc_pic[FU_.pic.y + i] = l.y;
    fc_pic[2u * FU_.pic.y + i] = l.z;
    fc_pic[3u * FU_.pic.y + i] = 0.0;
}

// ---- 10 ENCODE ------------------------------------------------------------------------------------------

fn fc_encode(gid: vec3<u32>) {
    if (gid.x >= FU_.pic.z || gid.y >= FU_.pic.w) {
        return;
    }
    let i = fc_at(0u, gid.x, gid.y);
    if (fc_pic[3u * FU_.pic.y + i] == 0.0) {
        return;
    }
    let ox = FU_.outi.z + i32(gid.x);
    let oy = FU_.outi.w + i32(gid.y);
    if (ox < 0 || oy < 0 || ox >= FU_.outs.x || oy >= FU_.outs.y) {
        return;
    }
    var e = logc4_encode3(vec3<f32>(fc_pic[i], fc_pic[FU_.pic.y + i], fc_pic[2u * FU_.pic.y + i]));
    // A picture of three zeros reads as "no picture here": a pixel that really encodes to zeros is nudged.
    if (e.x == 0.0 && e.y == 0.0 && e.z == 0.0) {
        e.x = 1.0e-30;
    }
    let o = u32(oy) * u32(FU_.outi.x) + u32(ox);
    fc_out[o] = e.x;
    fc_out[u32(FU_.outi.y) + o] = e.y;
    fc_out[2u * u32(FU_.outi.y) + o] = e.z;
}

// ---- the tone relation (tone.rs) ------------------------------------------------------------------------

// The block of channel `ch`: status (0 none, 1 plain, 2 a curve), the number of knots, the slopes below and above, the knots'
// ln A (32), their ln X (32) and the analysis picture's quantiles (32).
fn fc_blk(ch: u32) -> u32 {
    return u32(FU_.b.x) + ch * FC_BLOCK;
}

// Whether the relation is a curve: no channel lacks values, and some channel is not an exposure change.
fn fc_tone_active() -> bool {
    var any_curve = false;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let s = fc_fd[fc_blk(ch)];
        if (s == 0.0) {
            return false;
        }
        if (s == 2.0) {
            any_curve = true;
        }
    }
    return any_curve;
}

fn fc_forward(ch: u32, la: f32) -> f32 {
    let b = fc_blk(ch);
    let n = u32(fc_fd[b + 1u]);
    let a0 = fc_fd[b + 4u];
    let x0 = fc_fd[b + 4u + FC_BINS];
    if (la <= a0) {
        return x0 + fc_fd[b + 2u] * (la - a0);
    }
    let an = fc_fd[b + 4u + n - 1u];
    if (la >= an) {
        return fc_fd[b + 4u + FC_BINS + n - 1u] + fc_fd[b + 3u] * (la - an);
    }
    // The last knot not above `la`.
    var k = 0u;
    for (var i = 1u; i < n; i = i + 1u) {
        if (fc_fd[b + 4u + i] <= la) {
            k = i;
        }
    }
    let ak = fc_fd[b + 4u + k];
    let t = (la - ak) / (fc_fd[b + 5u + k] - ak);
    let xk = fc_fd[b + 4u + FC_BINS + k];
    return xk + t * (fc_fd[b + 5u + FC_BINS + k] - xk);
}

fn fc_inverse(ch: u32, lx: f32) -> f32 {
    let b = fc_blk(ch);
    let n = u32(fc_fd[b + 1u]);
    let x0 = fc_fd[b + 4u + FC_BINS];
    if (lx <= x0) {
        return fc_fd[b + 4u] + (lx - x0) / fc_fd[b + 2u];
    }
    let xn = fc_fd[b + 4u + FC_BINS + n - 1u];
    if (lx >= xn) {
        return fc_fd[b + 4u + n - 1u] + (lx - xn) / fc_fd[b + 3u];
    }
    // The first knot with ln X not below `lx`.
    var k = n - 1u;
    for (var i = n - 1u; i > 0u; i = i - 1u) {
        if (fc_fd[b + 4u + FC_BINS + i] >= lx) {
            k = i;
        }
    }
    let x_lo = fc_fd[b + 3u + FC_BINS + k];
    let x_hi = fc_fd[b + 4u + FC_BINS + k];
    var t = 1.0;
    if (x_hi > x_lo) {
        t = (lx - x_lo) / (x_hi - x_lo);
    }
    let a_lo = fc_fd[b + 3u + k];
    return a_lo + t * (fc_fd[b + 4u + k] - a_lo);
}

// The input value `x` of channel `ch` with the gain `g` applied in the domain where the curve cancels (ToneRelation::gain).
fn fc_tone_gain(ch: u32, x: f32, g: f32) -> f32 {
    if (x > FC_TONE_FLOOR && g > 0.0 && fc_finite(g)) {
        let la = fc_inverse(ch, log(x)) + log(g);
        return exp(fc_forward(ch, la));
    }
    return x * g;
}

// ---- 1 BLEMISH ------------------------------------------------------------------------------------------

fn fc_blemish(gid: vec3<u32>) {
    if (gid.x >= FU_.pic.z || gid.y >= FU_.pic.w) {
        return;
    }
    let mw = u32(FU_.a.y);
    let mh = u32(FU_.a.z);
    let v = FU_.f.z + f32(gid.y) * FU_.f.w;
    if (!(v >= 0.0 && v < f32(mh))) {
        return;
    }
    let u = FU_.f.x + f32(gid.x) * FU_.f.y;
    if (!(u >= 0.0 && u < f32(mw))) {
        return;
    }
    let x0 = u32(floor(u));
    let x1 = min(x0 + 1u, mw - 1u);
    let fx = u - f32(x0);
    let y0 = u32(floor(v));
    let y1 = min(y0 + 1u, mh - 1u);
    let fy = v - f32(y0);
    let base = u32(FU_.a.x);
    var s4 = vec4<f32>(0.0);
    for (var c = 0u; c < 4u; c = c + 1u) {
        let t00 = fc_fd[base + 4u * (y0 * mw + x0) + c];
        let t01 = fc_fd[base + 4u * (y0 * mw + x1) + c];
        let t10 = fc_fd[base + 4u * (y1 * mw + x0) + c];
        let t11 = fc_fd[base + 4u * (y1 * mw + x1) + c];
        let top = t00 + (t01 - t00) * fx;
        let bottom = t10 + (t11 - t10) * fx;
        s4[c] = top + (bottom - top) * fy;
    }
    // threshold_strength
    let amount = FU_.g.x;
    var s = clamp(s4.w / max(1.0 - amount, 0.01), 0.0, 1.0);
    if (amount < 0.25) {
        s = s * 4.0 * amount;
    }
    if (amount > 0.75) {
        s = s + (1.0 - s) * (4.0 * (amount - 0.75));
    }
    s = s * fc_keep(gid.x, gid.y);
    let i = fc_at(0u, gid.x, gid.y);
    let old = vec3<f32>(fc_pic[i], fc_pic[FU_.pic.y + i], fc_pic[2u * FU_.pic.y + i]);
    var new_v = old;
    var curve = false;
    if (FU_.a.w != 0) {
        curve = fc_tone_active();
    }
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (curve) {
            let g = 1.0 + (s4[c] - 1.0) * s;
            new_v[c] = fc_tone_gain(c, old[c], g);
        } else {
            new_v[c] = old[c] * (1.0 + (s4[c] - 1.0) * s);
        }
    }
    if (any(new_v != old)) {
        fc_pic[i] = new_v.x;
        fc_pic[FU_.pic.y + i] = new_v.y;
        fc_pic[2u * FU_.pic.y + i] = new_v.z;
        fc_pic[3u * FU_.pic.y + i] = 1.0;
    }
}

// ---- 2 DOWN ---------------------------------------------------------------------------------------------

// a: the rectangle of the source (x, y, w, h) in the region; b: the destination (offset, layer width, layer height, ds);
// c: x: the plane of fc_pic, or the first byte of a mask the size of the rectangle; y: 0 a plane of the picture, 1 a mask.
fn fc_down(gid: vec3<u32>) {
    let lw = u32(FU_.b.y);
    let lh = u32(FU_.b.z);
    if (gid.x >= lw || gid.y >= lh) {
        return;
    }
    let ds = u32(FU_.b.w);
    let rw = u32(FU_.a.z);
    let rh = u32(FU_.a.w);
    var acc = 0.0;
    var n = 0.0;
    for (var y = ds * gid.y; y < min(ds * gid.y + ds, rh); y = y + 1u) {
        for (var x = ds * gid.x; x < min(ds * gid.x + ds, rw); x = x + 1u) {
            if (FU_.c.y == 0) {
                acc = acc + fc_pic[u32(FU_.c.x) * FU_.pic.y + (u32(FU_.a.y) + y) * FU_.pic.x + u32(FU_.a.x) + x];
            } else {
                acc = acc + f32(fc_byte(u32(FU_.c.x), y * rw + x)) / 255.0;
            }
            n = n + 1.0;
        }
    }
    fc_lay[u32(FU_.b.x) + gid.y * lw + gid.x] = acc / n;
}

// ---- 3 BOX ----------------------------------------------------------------------------------------------

// a: the source offset, the destination offset, the layer width and height; b: the radius, the axis (0 along rows, 1 along
// columns), the number of planes, the length of a plane. gid.x is the line (a row, or a column), gid.y the 64-output segment,
// gid.z the plane.
fn fc_box(gid: vec3<u32>) {
    let lw = u32(FU_.a.z);
    let lh = u32(FU_.a.w);
    let q = FU_.b.x;
    let vertical = FU_.b.y != 0;
    var lines = lh;
    var len = lw;
    if (vertical) {
        lines = lw;
        len = lh;
    }
    if (gid.x >= lines || gid.y * FC_SEG >= len || gid.z >= u32(FU_.b.z)) {
        return;
    }
    let plane = gid.z * u32(FU_.b.w);
    let sbase = u32(FU_.a.x) + plane;
    let dbase = u32(FU_.a.y) + plane;
    var stride_e = 1u;
    var start = gid.x * lw;
    if (vertical) {
        stride_e = lw;
        start = gid.x;
    }
    let last = i32(len) - 1;
    let inv = 1.0 / f32(2 * q + 1);
    let first = gid.y * FC_SEG;
    var acc = 0.0;
    for (var i = -q; i <= q; i = i + 1) {
        acc = acc + fc_lay[sbase + start + u32(clamp(i32(first) + i, 0, last)) * stride_e];
    }
    let end = min(first + FC_SEG, len);
    for (var x = first; x < end; x = x + 1u) {
        fc_lay[dbase + start + x * stride_e] = acc * inv;
        acc = acc + fc_lay[sbase + start + u32(clamp(i32(x) + q + 1, 0, last)) * stride_e]
            - fc_lay[sbase + start + u32(clamp(i32(x) - q, 0, last)) * stride_e];
    }
}

// ---- 4 MUL, 5 MCOMB, 6 YPLANE ---------------------------------------------------------------------------

// The element of a one-dimensional op: rows of 4096 threads (the dispatch is `FLAT_X` groups wide).
fn fc_lin(gid: vec3<u32>) -> u32 {
    return gid.y * 4096u + gid.x;
}

// a: x's offset, m's offset, the destination offset, the length of a plane; b.x: the number of planes.
fn fc_mul(gid: vec3<u32>) {
    let n = u32(FU_.a.w);
    let id = fc_lin(gid);
    if (id >= n) {
        return;
    }
    let m = fc_lay[u32(FU_.a.y) + id];
    for (var p = 0; p < FU_.b.x; p = p + 1) {
        let k = u32(p) * n + id;
        fc_lay[u32(FU_.a.z) + k] = fc_lay[u32(FU_.a.x) + k] * m;
    }
}

// a: m's offset, num's, den's, x's; b: the destination offset, the length of a plane, the number of planes.
fn fc_mcomb(gid: vec3<u32>) {
    let n = u32(FU_.b.y);
    let id = fc_lin(gid);
    if (id >= n) {
        return;
    }
    let m = fc_lay[u32(FU_.a.x) + id];
    let den = max(fc_lay[u32(FU_.a.z) + id], 1.0 / 255.0);
    for (var p = 0; p < FU_.b.z; p = p + 1) {
        let k = u32(p) * n + id;
        fc_lay[u32(FU_.b.x) + k] = m * fc_lay[u32(FU_.a.y) + k] / den + (1.0 - m) * fc_lay[u32(FU_.a.w) + k];
    }
}

// a: the three planes' offset, the destination offset, the length of a plane.
fn fc_yplane(gid: vec3<u32>) {
    let n = u32(FU_.a.z);
    let id = fc_lin(gid);
    if (id >= n) {
        return;
    }
    let t = u32(FU_.a.x);
    fc_lay[u32(FU_.a.y) + id] = (fc_lay[t + id] + fc_lay[t + n + id] + fc_lay[t + 2u * n + id]) / 3.0;
}

// ---- 7 LOWFINAL -----------------------------------------------------------------------------------------

fn fc_sum3(p: vec3<f32>) -> f32 {
    return max(p.x + p.y + p.z, 0.001);
}

// 7.5: the weight of a blend, and the blend.
fn fc_blend(src: vec3<f32>, nw: vec3<f32>, k: f32, sp: f32) -> vec3<f32> {
    let s = fc_sum3(src);
    let n = fc_sum3(nw);
    var w = 0.0;
    if (n < s) {
        let q = n / s;
        let q2 = q * q;
        let q4 = q2 * q2;
        let q8 = q4 * q4;
        w = q8 * q8;
    } else {
        let q = s / n;
        w = (1.0 - sp) + sp * q * q * q;
    }
    let f = k * w;
    return src + f * (nw - src);
}

fn fc_smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn fc_layer3(off: i32, n: u32, li: u32) -> vec3<f32> {
    let o = u32(off);
    return vec3<f32>(fc_lay[o + li], fc_lay[o + n + li], fc_lay[o + 2u * n + li]);
}

// a: the padded region (x, y, w, h) in the region of the working image; b: the layer width, ds, the offset of T (three
// planes), of Y; c: the offsets of C_dc (three planes), C_se, C_g and the shine colour (three planes), -1 for none; d: the
// first byte of the dark-circle, even-skin and grading masks over the rectangle, -1 for none, and flags (1 contrast, 2 shadow
// desaturation, 4 shine removal); f: k_dc, sp_dc, k_se, sp_se; g: gc, gd, gs.
fn fc_lowfinal(gid: vec3<u32>) {
    let rw = u32(FU_.a.z);
    let rh = u32(FU_.a.w);
    if (gid.x >= rw || gid.y >= rh) {
        return;
    }
    let x = u32(FU_.a.x) + gid.x;
    let y = u32(FU_.a.y) + gid.y;
    let keep = fc_keep(x, y);
    if (keep == 0.0) {
        return;
    }
    let ln = FU_.op.y;
    let li = (gid.y / u32(FU_.b.y)) * u32(FU_.b.x) + gid.x / u32(FU_.b.y);
    let ri = gid.y * rw + gid.x;
    let i = fc_at(0u, x, y);
    let old = vec3<f32>(fc_pic[i], fc_pic[FU_.pic.y + i], fc_pic[2u * FU_.pic.y + i]);
    var cur = old;
    var touched = false;
    if (FU_.d.x >= 0 && fc_byte(u32(FU_.d.x), ri) != 0u) {
        let t = fc_layer3(FU_.b.z, ln, li);
        let cc = fc_layer3(FU_.c.x, ln, li);
        let nw = cur - t + cc;
        cur = fc_blend(cur, nw, keep * FU_.f.x, FU_.f.y);
        touched = true;
    }
    if (FU_.d.y >= 0 && fc_byte(u32(FU_.d.y), ri) != 0u) {
        let yv = fc_lay[u32(FU_.b.w) + li];
        let cse = fc_lay[u32(FU_.c.y) + li];
        let gain = cse / max(yv, 0.001);
        cur = fc_blend(cur, cur * gain, keep * FU_.f.z, FU_.f.w);
        touched = true;
    }
    if (FU_.d.z >= 0) {
        let byte = fc_byte(u32(FU_.d.z), ri);
        if (byte != 0u) {
            let m = keep * f32(byte) / 255.0;
            let yv = fc_lay[u32(FU_.b.w) + li];
            let cg = fc_lay[u32(FU_.c.z) + li];
            let flags = u32(FU_.d.w);
            if ((flags & 1u) != 0u && yv > 0.0 && cg > 0.0) {
                let y_new = cg * pow(yv / cg, 1.0 + 0.5 * FU_.g.x);
                let gain = y_new / max(yv, 0.001);
                cur = cur + m * (cur * gain - cur);
            }
            if ((flags & 2u) != 0u) {
                var w = 0.0;
                if (cg > 0.0) {
                    w = clamp((cg - yv) / (0.5 * cg), 0.0, 1.0);
                }
                let l = (cur.x + cur.y + cur.z) / 3.0;
                let f = 1.0 - 0.6 * FU_.g.y * w * m;
                cur = vec3<f32>(l) + f * (cur - vec3<f32>(l));
            }
            if ((flags & 4u) != 0u && yv > 0.0 && cg > 0.0 && FU_.g.z > 0.0) {
                let t = fc_layer3(FU_.b.z, ln, li);
                let cc = fc_layer3(FU_.c.w, ln, li);
                let e = log(yv / cg) - 0.1;
                let tmax = max(t.x, max(t.y, t.z));
                var sat = 1.0;
                if (tmax > 0.0) {
                    sat = 1.0 - min(t.x, min(t.y, t.z)) / tmax;
                }
                let w = fc_smoothstep(0.0, 0.4, e) * (1.0 - fc_smoothstep(0.15, 0.35, sat)) * m;
                if (w > 0.0) {
                    let level = cg * pow(e, 0.1) / max((cc.x + cc.y + cc.z) / 3.0, 0.001);
                    let f = min(FU_.g.z * w, 1.0);
                    let t_new = t + f * (cc * level - t);
                    cur = cur + (t_new - t);
                }
            }
            touched = true;
        }
    }
    if (touched && any(cur != old)) {
        fc_pic[i] = cur.x;
        fc_pic[FU_.pic.y + i] = cur.y;
        fc_pic[2u * FU_.pic.y + i] = cur.z;
        fc_pic[3u * FU_.pic.y + i] = 1.0;
    }
}

// ---- 8 TSELECT, 9 TAPPLY --------------------------------------------------------------------------------

// The total order of f32 as an unsigned integer.
fn fc_key(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b & 0x80000000u) == 0u) {
        return b | 0x80000000u;
    }
    return ~b;
}

fn fc_unkey(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) {
        return bitcast<f32>(k & 0x7fffffffu);
    }
    return bitcast<f32>(~k);
}

var<workgroup> fc_hist: array<atomic<u32>, 256>;

// The k-th smallest (k from 1) of the mean of the pixels where the teeth mask is not 0, by an 8-bit radix select: four rounds,
// each a histogram of the next byte of the keys that match the bytes chosen so far. a: the rectangle; b: the first byte of the
// mask, k, the layer offset the answer goes to.
fn fc_tselect(local: u32) {
    let rw = u32(FU_.a.z);
    let rh = u32(FU_.a.w);
    let total = rw * rh;
    var prefix = 0u;
    var k = u32(FU_.b.y);
    for (var rnd = 0u; rnd < 4u; rnd = rnd + 1u) {
        let shift = 24u - 8u * rnd;
        // Cleared by all, then counted.
        atomicStore(&fc_hist[local], 0u);
        workgroupBarrier();
        for (var idx = local; idx < total; idx = idx + 256u) {
            if (fc_byte(u32(FU_.b.x), idx) != 0u) {
                let x = u32(FU_.a.x) + idx % rw;
                let y = u32(FU_.a.y) + idx / rw;
                let i = fc_at(0u, x, y);
                let l = (fc_pic[i] + fc_pic[FU_.pic.y + i] + fc_pic[2u * FU_.pic.y + i]) / 3.0;
                let key = fc_key(l);
                // The bytes above this round's must be the ones chosen.
                var hi = 0u;
                if (rnd > 0u) {
                    hi = key >> (shift + 8u);
                }
                if (hi == prefix) {
                    atomicAdd(&fc_hist[(key >> shift) & 255u], 1u);
                }
            }
        }
        workgroupBarrier();
        // Every thread walks the same 256 counts and finds the same byte.
        var digit = 255u;
        var below = 0u;
        for (var d = 0u; d < 256u; d = d + 1u) {
            let c = atomicLoad(&fc_hist[d]);
            if (k <= below + c) {
                digit = d;
                break;
            }
            below = below + c;
        }
        k = k - below;
        prefix = (prefix << 8u) | digit;
        workgroupBarrier();
    }
    if (local == 0u) {
        fc_lay[u32(FU_.b.z)] = fc_unkey(prefix);
    }
}

// a: the rectangle; b: the first byte of the mask, the layer offset of L_max; f.x: tw.
fn fc_tapply(gid: vec3<u32>) {
    let rw = u32(FU_.a.z);
    let rh = u32(FU_.a.w);
    if (gid.x >= rw || gid.y >= rh) {
        return;
    }
    let byte = fc_byte(u32(FU_.b.x), gid.y * rw + gid.x);
    let x = u32(FU_.a.x) + gid.x;
    let y = u32(FU_.a.y) + gid.y;
    let keep = fc_keep(x, y);
    if (byte == 0u || keep == 0.0) {
        return;
    }
    let tw = FU_.f.x;
    let m = keep * f32(byte) / 255.0;
    let i = fc_at(0u, x, y);
    let old = vec3<f32>(fc_pic[i], fc_pic[FU_.pic.y + i], fc_pic[2u * FU_.pic.y + i]);
    let l = (old.x + old.y + old.z) / 3.0;
    let f = 1.0 - 0.7 * tw * m;
    let cap = fc_lay[u32(FU_.b.y)] / max(l, 1.0e-4);
    let gain = min(1.0 + 0.15 * tw * m, max(cap, 1.0));
    let nw = (vec3<f32>(l) + f * (old - vec3<f32>(l))) * gain;
    if (any(nw != old)) {
        fc_pic[i] = nw.x;
        fc_pic[FU_.pic.y + i] = nw.y;
        fc_pic[2u * FU_.pic.y + i] = nw.z;
        fc_pic[3u * FU_.pic.y + i] = 1.0;
    }
}

// ---- 12 COPY --------------------------------------------------------------------------------------------

fn fc_copy(gid: vec3<u32>) {
    if (gid.x >= FU_.pic.z || gid.y >= FU_.pic.w) {
        return;
    }
    let ox = FU_.outi.z + i32(gid.x);
    let oy = FU_.outi.w + i32(gid.y);
    if (ox < 0 || oy < 0 || ox >= FU_.outs.x || oy >= FU_.outs.y) {
        return;
    }
    let s = gid.y * FU_.src.x + gid.x;
    let v = vec3<f32>(fc_src[s], fc_src[FU_.src.y + s], fc_src[2u * FU_.src.y + s]);
    if (v.x == 0.0 && v.y == 0.0 && v.z == 0.0) {
        return;
    }
    let o = u32(oy) * u32(FU_.outi.x) + u32(ox);
    fc_out[o] = v.x;
    fc_out[u32(FU_.outi.y) + o] = v.y;
    fc_out[2u * u32(FU_.outi.y) + o] = v.z;
}

// ---- 11 TONE --------------------------------------------------------------------------------------------

// One workgroup for each channel (workgroup_id.x). a: the sample grid's columns, rows and step; b.x: the first block of fc_fd.
// The samples are the CPU's: every `step`-th pixel of every `step`-th row of the region, the values above the floor
// (finite) kept. The 32 quantiles of their logarithms are found by the 8-bit radix select (four rounds each), then the curve is
// made as `ToneRelation::estimate` makes it from the analysis picture's quantiles (already in the block).
var<workgroup> fc_tn: atomic<u32>;
var<workgroup> fc_nx: u32;
var<workgroup> fc_q: array<f32, 32>;

fn fc_sample(ch: u32, idx: u32) -> f32 {
    let cols = u32(FU_.a.x);
    let step = u32(FU_.a.z);
    let x = (idx % cols) * step;
    let y = (idx / cols) * step;
    return fc_pic[fc_at(ch, x, y)];
}

fn fc_usable(v: f32) -> bool {
    return fc_finite(v) && v > FC_TONE_FLOOR;
}

fn fc_tone(ch: u32, local: u32) {
    let total = u32(FU_.a.x) * u32(FU_.a.y);
    let blk = fc_blk(ch);
    if (local == 0u) {
        atomicStore(&fc_tn, 0u);
    }
    workgroupBarrier();
    for (var idx = local; idx < total; idx = idx + 256u) {
        if (fc_usable(fc_sample(ch, idx))) {
            atomicAdd(&fc_tn, 1u);
        }
    }
    workgroupBarrier();
    if (local == 0u) {
        fc_nx = atomicLoad(&fc_tn);
    }
    let nx = workgroupUniformLoad(&fc_nx);
    // A channel with too few values: no relation at all.
    let enough = nx >= FC_MIN_PAIRS;
    // The quantile `k` of the sorted logarithms is the element at ((2k + 1) nx) / 64.
    for (var kq = 0u; kq < FC_BINS; kq = kq + 1u) {
        if (!enough) {
            break;
        }
        var rank = min(((2u * kq + 1u) * nx) / (2u * FC_BINS), nx - 1u) + 1u;
        var prefix = 0u;
        for (var rnd = 0u; rnd < 4u; rnd = rnd + 1u) {
            let shift = 24u - 8u * rnd;
            atomicStore(&fc_hist[local], 0u);
            workgroupBarrier();
            for (var idx = local; idx < total; idx = idx + 256u) {
                let v = fc_sample(ch, idx);
                if (fc_usable(v)) {
                    let key = fc_key(v);
                    var hi = 0u;
                    if (rnd > 0u) {
                        hi = key >> (shift + 8u);
                    }
                    if (hi == prefix) {
                        atomicAdd(&fc_hist[(key >> shift) & 255u], 1u);
                    }
                }
            }
            workgroupBarrier();
            var digit = 255u;
            var below = 0u;
            for (var d = 0u; d < 256u; d = d + 1u) {
                let c = atomicLoad(&fc_hist[d]);
                if (rank <= below + c) {
                    digit = d;
                    break;
                }
                below = below + c;
            }
            rank = rank - below;
            prefix = (prefix << 8u) | digit;
            workgroupBarrier();
        }
        if (local == 0u) {
            fc_q[kq] = log(fc_unkey(prefix));
        }
        workgroupBarrier();
    }
    if (local != 0u) {
        return;
    }
    // The curve (ToneRelation::estimate, one channel), by one thread: the knots where the analysis quantile rises by more than
    // 1e-9, the input's made non-decreasing.
    fc_fd[blk] = 0.0;
    if (!enough) {
        return;
    }
    var n = 0u;
    var last_la = 0.0;
    var last_lx = 0.0;
    for (var kq = 0u; kq < FC_BINS; kq = kq + 1u) {
        let a = fc_fd[blk + 4u + 2u * FC_BINS + kq];
        if (n > 0u && a <= last_la + 1.0e-9) {
            continue;
        }
        var x = fc_q[kq];
        if (n > 0u) {
            x = max(last_lx, x);
        }
        fc_fd[blk + 4u + n] = a;
        fc_fd[blk + 4u + FC_BINS + n] = x;
        last_la = a;
        last_lx = x;
        n = n + 1u;
    }
    if (n < 3u) {
        return;
    }
    fc_fd[blk + 1u] = f32(n);
    // The slopes beyond the ends: the first three knots and the last three, between 0.05 and 4.
    let lo_b = min(2u, n - 1u);
    var hi_a = 0u;
    if (n >= 3u) {
        hi_a = n - 3u;
    }
    let s_lo = clamp(
        (fc_fd[blk + 4u + FC_BINS + lo_b] - fc_fd[blk + 4u + FC_BINS]) / (fc_fd[blk + 4u + lo_b] - fc_fd[blk + 4u]),
        0.05, 4.0);
    let s_hi = clamp(
        (fc_fd[blk + 4u + FC_BINS + n - 1u] - fc_fd[blk + 4u + FC_BINS + hi_a]) / (fc_fd[blk + 4u + n - 1u] - fc_fd[blk + 4u + hi_a]),
        0.05, 4.0);
    fc_fd[blk + 2u] = s_lo;
    fc_fd[blk + 3u] = s_hi;
    // The regression slope of ln X on ln A over the knots: an exposure change has slope 1.
    var ma = 0.0;
    var mx = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        ma = ma + fc_fd[blk + 4u + i];
        mx = mx + fc_fd[blk + 4u + FC_BINS + i];
    }
    ma = ma / f32(n);
    mx = mx / f32(n);
    var cov = 0.0;
    var vr = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let da = fc_fd[blk + 4u + i] - ma;
        cov = cov + da * (fc_fd[blk + 4u + FC_BINS + i] - mx);
        vr = vr + da * da;
    }
    let slope = cov / vr;
    var status = 1.0;
    if (abs(slope - 1.0) > FC_SLOPE_TOL || abs(s_lo - 1.0) > 2.0 * FC_SLOPE_TOL || abs(s_hi - 1.0) > 2.0 * FC_SLOPE_TOL) {
        status = 2.0;
    }
    fc_fd[blk] = status;
}

// ---------------------------------------------------------------------------------------------------------

fn face_main(gid: vec3<u32>, local: u32, group: vec3<u32>) {
    switch (FU_.op.x) {
        case 0u: { fc_decode(gid); }
        case 1u: { fc_blemish(gid); }
        case 2u: { fc_down(gid); }
        case 3u: { fc_box(gid); }
        case 4u: { fc_mul(gid); }
        case 5u: { fc_mcomb(gid); }
        case 6u: { fc_yplane(gid); }
        case 7u: { fc_lowfinal(gid); }
        case 8u: { fc_tselect(local); }
        case 9u: { fc_tapply(gid); }
        case 10u: { fc_encode(gid); }
        case 11u: { fc_tone(group.x, local); }
        case 12u: { fc_copy(gid); }
        default: {}
    }
}
