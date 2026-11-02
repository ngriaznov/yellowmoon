// The raster masks' pass (docs/23-nodes.md sections 4.3 and 5.5, docs/algorithms/06-masks.md section 5), the device twin
// of ym_color::mask_raster: the resample of a row's photo-frame plane S0 to the output grid, Grow (a separable maximum or
// minimum), Feather (cascaded box means), the guided filter of Refine edges (the Fast Guided Filter above 16 samples) and
// the quantising of the result to the u16 planes that the colour pass reads (binding 7 of color.wgsl).
//
// One kernel, one entry point, an `op` in the uniform. Every op is a short, separate dispatch over a window of the output
// grid, so a plane is computed in the order the host records them and every pixel of every op is the same expression at every
// tile (docs/17 D4 and D5): every window fold is a blocked fold with absolute block origins (van Herk and Gil-Werman: blocks
// of the window's width, a suffix of one block joined to a prefix of the next, the scans restarting at each block boundary
// in a fixed order), cut to the frame (the whole output image) and to the plane. It costs a few operations a pixel at any
// radius, and a window's value is made from the pixels of the window only, so a tile is the whole frame bit for bit. The
// mean is divided by the number of samples the window kept.
//
// Bindings: 0 the uniform; 1 `work`, the f32 planes of the pass (read_write); 2 `s0`, a source plane as u16 codes two to a word;
// 3 `samp`, the sampler's image (planes: colour, the letterbox mask at 3, the photo-frame position at 4 and 5); 4 `guide`,
// the Input's luma as plane 0 of a colour-pass stage image; 5 `outp`, the u16 planes the colour pass reads, two to a word
// (read_write).
//
// Ops (the fields of RU a op reads are named in each):
//   0 RESAMPLE  work[d] = S0 at the place of each pixel of the window (the pixel outside the photo is 0).
//   1 SCAN      the first half of a window fold of plane A along one axis (mean, max or min over [x - r, x + r]): the
//               prefix and the suffix scans of every block of 2 r + 1 samples, one thread for a (line, block), into
//               the planes B (prefixes) and C (suffixes).
//   8 COMB      the second half: window = suffix[x - r] joined to prefix[x + r], into the destination plane.
//   2 MUL       work[d] = A * B (the guide squared, the guide times the mask).
//   3 COEF      a and b of the guided filter, in place over the planes of the guided products.
//   4 BLOCK     a coarse plane: the mean of step x step blocks of plane A.
//   5 UP        q = a G + b with a and b bilinearly upsampled from the coarse grid, quantised into outp.
//   6 FINAL     q = a G + b at full resolution, quantised into outp.
//   7 QUANT     work[a] quantised into outp.

struct RU {
    op: vec4<u32>,    // x: the op; y: fold mode (0 mean, 1 max, 2 min); z: fold axis (0 x, 1 y); w: fold radius
    pa: vec4<i32>,    // plane A: x: offset in `work` (elements), y: width, z: height, w: stride
    pb: vec4<i32>,    // plane B: x: offset in `work`, y: width, z: height, w: stride (SCAN, COMB: the prefixes)
    pc: vec4<i32>,    // plane C: x: offset in `work` (COEF: the products' planes), y: offset of plane D, z: offset of plane E, w: stride (SCAN, COMB: x the suffixes)
    pd: vec4<i32>,    // the destination plane: x: offset in `work`, y: width, z: height, w: stride
    org_a: vec4<i32>, // x, y: the absolute origin of plane A (and of B, C, D, E: planes of one op share it); z: a is the guide (1), b is the guide (2), both (3)
    org_d: vec4<i32>, // x, y: the absolute origin of the destination plane (BLOCK: in cells; UP: the coarse plane's); z: step (BLOCK, UP); w: -
    frame: vec4<i32>, // x0, y0, x1, y1: the frame in the units of plane A (UP: the coarse frame in the units of the coarse plane)
    win: vec4<i32>,   // x, y: the output extent's first pixel in plane A's pixels; z, w: the extent's size (FINAL, UP, QUANT)
    ins: vec4<i32>,   // x0, y0, x1, y1: the part of the extent that has planes, in plane A's pixels (the rest is 0)
    outp: vec4<u32>,  // x: the stride of an output plane (elements), y: its length in words, z: the output plane index, w: -
    samp: vec4<u32>,  // x: the sampler image's stride, y: its plane length, z, w: the window's first pixel in it (RESAMPLE)
    s0: vec4<u32>,    // x: the source plane's width, y: its height, z: the taps of the box average, w: -
    f: vec4<f32>,     // x: eps (COEF); y: the shrink in source pixels per output pixel (RESAMPLE)
    cg: vec4<i32>,    // UP: the coarse plane that the window's pixels interpolate in: x, y: its origin, z, w: its size
};

@group(0) @binding(0) var<uniform> RU_: RU;
@group(0) @binding(1) var<storage, read_write> work: array<f32>;
@group(0) @binding(2) var<storage, read> s0: array<u32>;
@group(0) @binding(3) var<storage, read> samp: array<f32>;
@group(0) @binding(4) var<storage, read> guide: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<u32>;

const OP_RESAMPLE: u32 = 0u;
const OP_SCAN: u32 = 1u;
const OP_MUL: u32 = 2u;
const OP_COEF: u32 = 3u;
const OP_BLOCK: u32 = 4u;
const OP_UP: u32 = 5u;
const OP_FINAL: u32 = 6u;
const OP_QUANT: u32 = 7u;
const OP_COMB: u32 = 8u;

fn unit01(x: f32) -> f32 {
    if (is_nan_f(x)) {
        return 0.0;
    }
    return clamp(x, 0.0, 1.0);
}

// ---- S0 ---------------------------------------------------------------------------------------

fn s0_code(i: i32, j: i32) -> f32 {
    let sw = i32(RU_.s0.x);
    let sh = i32(RU_.s0.y);
    let ci = clamp(i, 0, sw - 1);
    let cj = clamp(j, 0, sh - 1);
    let idx = u32(cj * sw + ci);
    let word = s0[idx >> 1u];
    return f32((word >> ((idx & 1u) * 16u)) & 0xffffu) / 65535.0;
}

fn s0_bilinear(u: f32, v: f32) -> f32 {
    let fu = floor(u);
    let fv = floor(v);
    let tu = u - fu;
    let tv = v - fv;
    let i = i32(fu);
    let j = i32(fv);
    let top = s0_code(i, j) * (1.0 - tu) + s0_code(i + 1, j) * tu;
    let bottom = s0_code(i, j + 1) * (1.0 - tu) + s0_code(i + 1, j + 1) * tu;
    return top * (1.0 - tv) + bottom * tv;
}

fn resample(pos: vec2<f32>) -> f32 {
    let u = pos.x * f32(RU_.s0.x) - 0.5;
    let v = pos.y * f32(RU_.s0.y) - 0.5;
    let taps = RU_.s0.z;
    if (taps <= 1u) {
        return unit01(s0_bilinear(u, v));
    }
    let n = f32(taps);
    var sum = 0.0;
    for (var b = 0u; b < taps; b++) {
        let dv = ((f32(b) + 0.5) / n - 0.5) * RU_.f.y;
        for (var a = 0u; a < taps; a++) {
            let du = ((f32(a) + 0.5) / n - 0.5) * RU_.f.y;
            sum += s0_bilinear(u + du, v + dv);
        }
    }
    return unit01(sum / (n * n));
}

// ---- Planes ------------------------------------------------------------------------------------

// A sample of plane A (`which` 1: the guide, else `work`) at (x, y) of the plane.
fn load_a(x: i32, y: i32) -> f32 {
    let i = u32(y * RU_.pa.w + x);
    if ((RU_.org_a.z & 1) != 0) {
        return guide[i];
    }
    return work[u32(RU_.pa.x) + i];
}

fn load_b(x: i32, y: i32) -> f32 {
    let i = u32(y * RU_.pb.w + x);
    if ((RU_.org_a.z & 2) != 0) {
        return guide[i];
    }
    return work[u32(RU_.pb.x) + i];
}

fn store_d(x: i32, y: i32, v: f32) {
    work[u32(RU_.pd.x + y * RU_.pd.w + x)] = v;
}

// floor(a / b) for a positive b.
fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    if (a % b < 0) {
        return q - 1;
    }
    return q;
}

// What a missing half of a window contributes to a fold, and two partial results joined.
fn fold_identity(mode: u32) -> f32 {
    if (mode == 0u) {
        return 0.0;
    }
    if (mode == 1u) {
        return -1.0e30;
    }
    return 1.0e30;
}

fn fold_join(mode: u32, a: f32, b: f32) -> f32 {
    if (mode == 0u) {
        return a + b;
    }
    if (mode == 1u) {
        return max(a, b);
    }
    return min(a, b);
}

// The element `k` of line `line` of a plane along the fold's axis (axis 0: the line is a row, 1: a column).
fn line_index(line: i32, k: i32, stride: i32) -> u32 {
    if (RU_.op.z == 0u) {
        return u32(line * stride + k);
    }
    return u32(k * stride + line);
}

fn quantise(q: f32) -> u32 {
    if (is_nan_f(q)) {
        return 0u;
    }
    return u32(clamp(q, 0.0, 1.0) * 65535.0 + 0.5);
}

// Writes the codes of the pixel pair (x, x + 1) of the output window (row y) into the output plane.
fn put_pair(x: i32, y: i32, a: u32, b: u32) {
    let idx = u32(y) * RU_.outp.x + u32(x);
    outp[RU_.outp.z * RU_.outp.y + (idx >> 1u)] = a | (b << 16u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let op = RU_.op.x;
    let x = i32(gid.x);
    let y = i32(gid.y);

    if (op == OP_RESAMPLE) {
        if (x >= RU_.pd.y || y >= RU_.pd.z) {
            return;
        }
        let si = (RU_.samp.w + u32(y)) * RU_.samp.x + RU_.samp.z + u32(x);
        var v = 0.0;
        if (samp[3u * RU_.samp.y + si] == 0.0) {
            v = resample(vec2<f32>(samp[4u * RU_.samp.y + si], samp[5u * RU_.samp.y + si]));
        }
        store_d(x, y, v);
        return;
    }

    if (op == OP_SCAN) {
        // A thread is a (line, block): x the line, y the block (counted from the block of the first sample).
        var lines = RU_.pa.y;
        var len = RU_.pa.z;
        var org = RU_.org_a.y;
        var f0 = RU_.frame.y;
        var f1 = RU_.frame.w;
        if (RU_.op.z == 0u) {
            lines = RU_.pa.z;
            len = RU_.pa.y;
            org = RU_.org_a.x;
            f0 = RU_.frame.x;
            f1 = RU_.frame.z;
        }
        let lo_av = max(org, f0);
        let hi_av = min(org + len, f1);
        let w = 2 * i32(RU_.op.w) + 1;
        if (x >= lines || lo_av >= hi_av) {
            return;
        }
        let start = (floor_div(lo_av, w) + y) * w;
        let first = max(start, lo_av);
        let last = min(start + w, hi_av) - 1;
        if (first > last) {
            return;
        }
        let mode = RU_.op.y;
        // The prefixes, upwards from the start of the block.
        var acc = 0.0;
        for (var k = first; k <= last; k++) {
            var v = 0.0;
            if (RU_.op.z == 0u) {
                v = load_a(k - org, x);
            } else {
                v = load_a(x, k - org);
            }
            if (k == first) {
                acc = v;
            } else {
                acc = fold_join(mode, acc, v);
            }
            work[u32(RU_.pb.x) + line_index(x, k - org, RU_.pb.w)] = acc;
        }
        // The suffixes, downwards from the end of the block.
        for (var k = last; k >= first; k--) {
            var v = 0.0;
            if (RU_.op.z == 0u) {
                v = load_a(k - org, x);
            } else {
                v = load_a(x, k - org);
            }
            if (k == last) {
                acc = v;
            } else {
                acc = fold_join(mode, acc, v);
            }
            work[u32(RU_.pc.x) + line_index(x, k - org, RU_.pc.w)] = acc;
        }
        return;
    }

    if (op == OP_COMB) {
        if (x >= RU_.pa.y || y >= RU_.pa.z) {
            return;
        }
        var at = y;
        var line = x;
        var len = RU_.pa.z;
        var org = RU_.org_a.y;
        var f0 = RU_.frame.y;
        var f1 = RU_.frame.w;
        if (RU_.op.z == 0u) {
            at = x;
            line = y;
            len = RU_.pa.y;
            org = RU_.org_a.x;
            f0 = RU_.frame.x;
            f1 = RU_.frame.z;
        }
        let lo_av = max(org, f0);
        let hi_av = min(org + len, f1);
        let abs_at = org + at;
        if (abs_at < lo_av || abs_at >= hi_av) {
            store_d(x, y, 0.0);
            return;
        }
        let mode = RU_.op.y;
        let r = i32(RU_.op.w);
        let w = 2 * r + 1;
        let lo_w = abs_at - r;
        let hi_w = abs_at + r;
        let id = fold_identity(mode);
        // The window's end in its second block: the prefix there (past the plane, the plane's last prefix when it is the
        // same block, else nothing).
        var upper = id;
        if (hi_w < hi_av) {
            upper = work[u32(RU_.pb.x) + line_index(line, hi_w - org, RU_.pb.w)];
        } else if (floor_div(hi_w, w) == floor_div(hi_av - 1, w)) {
            upper = work[u32(RU_.pb.x) + line_index(line, hi_av - 1 - org, RU_.pb.w)];
        }
        var total = upper;
        if (lo_w % w != 0) {
            var lower = id;
            if (lo_w >= lo_av) {
                lower = work[u32(RU_.pc.x) + line_index(line, lo_w - org, RU_.pc.w)];
            } else if (floor_div(lo_w, w) == floor_div(lo_av, w)) {
                lower = work[u32(RU_.pc.x) + line_index(line, lo_av - org, RU_.pc.w)];
            }
            total = fold_join(mode, lower, upper);
        }
        if (mode == 0u) {
            total = total / f32(min(hi_w, hi_av - 1) - max(lo_w, lo_av) + 1);
        }
        store_d(x, y, total);
        return;
    }

    if (op == OP_MUL) {
        if (x >= RU_.pa.y || y >= RU_.pa.z) {
            return;
        }
        store_d(x, y, load_a(x, y) * load_b(x, y));
        return;
    }

    if (op == OP_COEF) {
        if (x >= RU_.pa.y || y >= RU_.pa.z) {
            return;
        }
        // pa: mean(guide), pb: mean(p), pc.x: mean(guide^2), pc.y: mean(guide p); a replaces the third, b the fourth.
        let i = y * RU_.pa.w + x;
        let mg = work[u32(RU_.pa.x + i)];
        let mp = work[u32(RU_.pb.x + i)];
        let mgg = work[u32(RU_.pc.x + i)];
        let mgp = work[u32(RU_.pc.y + i)];
        let v = mgg - mg * mg;
        let cov = mgp - mg * mp;
        let a = cov / (v + RU_.f.x);
        let b = mp - a * mg;
        work[u32(RU_.pc.x + i)] = a;
        work[u32(RU_.pc.y + i)] = b;
        return;
    }

    if (op == OP_BLOCK) {
        // The destination is the coarse plane (cells), the source plane A the fine one.
        if (x >= RU_.pd.y || y >= RU_.pd.z) {
            return;
        }
        let step = RU_.org_d.z;
        let cx = RU_.org_d.x + x;
        let cy = RU_.org_d.y + y;
        let x0 = max(max(cx * step, RU_.frame.x), RU_.org_a.x);
        let x1 = min(min((cx + 1) * step, RU_.frame.z), RU_.org_a.x + RU_.pa.y);
        let y0 = max(max(cy * step, RU_.frame.y), RU_.org_a.y);
        let y1 = min(min((cy + 1) * step, RU_.frame.w), RU_.org_a.y + RU_.pa.z);
        var sum = 0.0;
        var n = 0.0;
        for (var yy = y0; yy < y1; yy++) {
            for (var xx = x0; xx < x1; xx++) {
                sum += load_a(xx - RU_.org_a.x, yy - RU_.org_a.y);
                n += 1.0;
            }
        }
        var v = 0.0;
        if (n > 0.0) {
            v = sum / n;
        }
        store_d(x, y, v);
        return;
    }

    // The ops below write the output window: a thread is a pair of adjacent pixels of a row.
    let px = 2 * x;
    if (px >= RU_.win.z || y >= RU_.win.w) {
        return;
    }
    let two = px + 1 < RU_.win.z;
    var qa = 0u;
    var qb = 0u;
    for (var k = 0; k < 2; k++) {
        if (k == 1 && !two) {
            break;
        }
        let wx = RU_.win.x + px + k;
        let wy = RU_.win.y + y;
        var q = 0.0;
        if (wx < RU_.ins.x || wx >= RU_.ins.z || wy < RU_.ins.y || wy >= RU_.ins.w) {
            // Outside the window that has planes (the paper of a view): no plane, no code.
            q = 0.0;
        } else if (op == OP_QUANT) {
            q = load_a(wx, wy);
        } else if (op == OP_FINAL) {
            // pa: a-bar, pb: b-bar, the guide at (wx, wy).
            let g = guide[u32(wy * RU_.pc.w + wx)];
            q = work[u32(RU_.pa.x + wy * RU_.pa.w + wx)] * g + work[u32(RU_.pb.x + wy * RU_.pb.w + wx)];
        } else {
            // UP: a-bar and b-bar on the coarse plane (pa, pb: offsets and size; org_d: its origin; cg: the clamp box).
            let step = f32(RU_.org_d.z);
            let ax = RU_.org_a.x + wx;
            let ay = RU_.org_a.y + wy;
            let ux = (f32(ax) + 0.5) / step - 0.5;
            let vy = (f32(ay) + 0.5) / step - 0.5;
            let fi = floor(ux);
            let fj = floor(vy);
            let tx = ux - fi;
            let ty = vy - fj;
            let c0 = clamp(clamp(i32(fi), RU_.frame.x, RU_.frame.z - 1), RU_.cg.x, RU_.cg.x + RU_.cg.z - 1) - RU_.cg.x;
            let c1 = clamp(clamp(i32(fi) + 1, RU_.frame.x, RU_.frame.z - 1), RU_.cg.x, RU_.cg.x + RU_.cg.z - 1) - RU_.cg.x;
            let r0 = clamp(clamp(i32(fj), RU_.frame.y, RU_.frame.w - 1), RU_.cg.y, RU_.cg.y + RU_.cg.w - 1) - RU_.cg.y;
            let r1 = clamp(clamp(i32(fj) + 1, RU_.frame.y, RU_.frame.w - 1), RU_.cg.y, RU_.cg.y + RU_.cg.w - 1) - RU_.cg.y;
            let ao = RU_.pa.x;
            let bo = RU_.pb.x;
            let cw = RU_.pa.y;
            let a_top = work[u32(ao + r0 * cw + c0)] * (1.0 - tx) + work[u32(ao + r0 * cw + c1)] * tx;
            let a_bot = work[u32(ao + r1 * cw + c0)] * (1.0 - tx) + work[u32(ao + r1 * cw + c1)] * tx;
            let b_top = work[u32(bo + r0 * cw + c0)] * (1.0 - tx) + work[u32(bo + r0 * cw + c1)] * tx;
            let b_bot = work[u32(bo + r1 * cw + c0)] * (1.0 - tx) + work[u32(bo + r1 * cw + c1)] * tx;
            let abar = a_top * (1.0 - ty) + a_bot * ty;
            let bbar = b_top * (1.0 - ty) + b_bot * ty;
            let g = guide[u32(wy * RU_.pc.w + wx)];
            q = abar * g + bbar;
        }
        if (k == 0) {
            qa = quantise(q);
        } else {
            qb = quantise(q);
        }
    }
    put_pair(px, y, qa, qb);
}
