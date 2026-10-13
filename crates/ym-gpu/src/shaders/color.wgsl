// The per-pixel colour chain C1 to C13 (docs/02-colour-pipeline.md) in one compute pass, mirroring
// ym_color::display::ColorChain stage by stage:
//
//   camera RGB --M--> C6 linear AWG4 --C7--> --C8--> LogC4 --C9--> --C10 look--> --C11 curves-->
//   --C12 ACES 2 or ARRI display LUT--> display linear --C13--> code values
//
// `mode.x` selects the stop (so each stage can be checked against `ColorChain::trace`), `mode.y` the
// output format. One thread handles two horizontally adjacent pixels. All maths is f32; image
// samples are read through the storage accessors (f16 or f32), look cubes and curves are f32
// storage (never hardware-filtered, never rounded to f16).

struct LutDesc {
    cube: vec4<u32>,     // x: offset of the cube in `luts` (vec4 units), y: size (0: no cube), z: has shaper, w: shaper offset
    shaper: vec4<u32>,   // x: shaper size
    dmin: vec4<f32>,     // cube domain
    dmax: vec4<f32>,
    smin: vec4<f32>,     // shaper domain
    smax: vec4<f32>,
};

struct ChainU {
    dims: vec4<u32>,     // x: out width, y: out height, z: out stride (even), w: out plane length
    src: vec4<u32>,      // x: src stride, y: src plane length, z: dx, w: dy
    origin: vec4<u32>,   // x, y: absolute pixel position of output (0, 0), for the dither
    mode: vec4<u32>,     // x: stop, y: out kind (0 f32 planar, 1 u8 rgba, 2 u16 planar), z: bits, w: dither
    flags: vec4<u32>,    // x: grade bits (1 contrast, 2 saturation), y: curve bits (1 luma, 2 r, 4 g, 8 b), z: look kind (0 none, 1 log-to-log, 2 display-baked), w: display kind (0 ACES 2, 1 display LUT)
    lk: vec4<u32>,       // x: look input (0 LogC4, 1 LogC3), y: look has CDL, w: before point (0 camera log, 1 after grade)
    dp: vec4<u32>,       // x: display LUT input (0 LogC4, 1 LogC3), y: display LUT transfer (0 sRGB, 1 gamma), z: apply display-LUT primaries matrix, w: apply look-to-output matrix
    sp: vec4<u32>,       // x: output transfer (0 sRGB, 1 BT.1886)
    m0: vec4<f32>,       // fused C1-C6 matrix rows
    m1: vec4<f32>,
    m2: vec4<f32>,
    tone: vec4<f32>,     // knee, span, tau
    grade0: vec4<f32>,   // contrast gain, brightness, saturation gain
    zone_amt: vec4<f32>, // shadows, highlights, black, white (each / 100)
    zone_ctr: vec4<f32>, // their centres (LogC4 codes)
    bal0: vec4<f32>,     // (a, b) shadows, (a, b) midtones
    bal1: vec4<f32>,     // (a, b) highlights
    bal_ctr: vec4<f32>,  // zone centres
    fl: vec4<f32>,       // x: look intensity fraction, y: display LUT gamma
    cdl_slope: vec4<f32>,  // w: saturation
    cdl_offset: vec4<f32>,
    cdl_power: vec4<f32>,
    look_lut: LutDesc,
    disp_lut: LutDesc,
    disp_m0: vec4<f32>,  // display LUT primaries -> output primaries
    disp_m1: vec4<f32>,
    disp_m2: vec4<f32>,
    look_m0: vec4<f32>,  // Rec.709 -> output primaries (display-baked look)
    look_m1: vec4<f32>,
    look_m2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> CU: ChainU;
// binding 1: ACES 2 uniform (aces2.wgsl)
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read> luts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> img: array<ST_ELEM>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;

//@ACCESSORS img ro

// ---- C7: highlight roll-off and low-light toe ----------------------------------------------------

fn c7_tone(rgb: vec3<f32>) -> vec3<f32> {
    let n = max(rgb.x, max(rgb.y, rgb.z));
    let k = CU.tone.x;
    let span = CU.tone.y;
    let tau = CU.tone.z;
    var m = n;
    if (n > k) {
        m = k + span * tanh(min((n - k) / span, 20.0));
    }
    var toe = 1.0;
    if (tau != 0.0) {
        if (m == 0.0) {
            toe = 0.0;
        } else {
            let r = tau / m;
            toe = 1.0 / (1.0 + r * r);
        }
    }
    var s = toe;
    if (n != 0.0) {
        s = (m / n) * toe;
    }
    return rgb * s;
}

// ---- C9: primary grade ---------------------------------------------------------------------------

fn bump(l: f32, mu: f32, sigma: f32) -> f32 {
    let z = (l - mu) / sigma;
    return exp(-0.5 * z * z);
}

fn c9_grade(v: vec3<f32>, l: f32) -> vec3<f32> {
    var o = v;
    if ((CU.flags.x & 1u) != 0u) {
        o = vec3<f32>(GRAY) + (o - vec3<f32>(GRAY)) * CU.grade0.x;
    }
    var shift = CU.grade0.y * STOP * bump(l, GRAY, BRIGHTNESS_SIGMA);
    if (CU.zone_amt.x != 0.0) {
        shift += CU.zone_amt.x * ZONE_CONTROL_SIGMA * bump(l, CU.zone_ctr.x, ZONE_CONTROL_SIGMA);
    }
    if (CU.zone_amt.y != 0.0) {
        shift += CU.zone_amt.y * ZONE_CONTROL_SIGMA * bump(l, CU.zone_ctr.y, ZONE_CONTROL_SIGMA);
    }
    if (CU.zone_amt.z != 0.0) {
        shift += CU.zone_amt.z * ZONE_CONTROL_SIGMA * bump(l, CU.zone_ctr.z, ZONE_CONTROL_SIGMA);
    }
    if (CU.zone_amt.w != 0.0) {
        shift += CU.zone_amt.w * ZONE_CONTROL_SIGMA * bump(l, CU.zone_ctr.w, ZONE_CONTROL_SIGMA);
    }
    o = o + vec3<f32>(shift);
    if (CU.bal0.x != 0.0 || CU.bal0.y != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, CU.bal_ctr.x, ZONE_SIGMA);
        o = o + w * (CU.bal0.x * E1 + CU.bal0.y * E2);
    }
    if (CU.bal0.z != 0.0 || CU.bal0.w != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, CU.bal_ctr.y, ZONE_SIGMA);
        o = o + w * (CU.bal0.z * E1 + CU.bal0.w * E2);
    }
    if (CU.bal1.x != 0.0 || CU.bal1.y != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, CU.bal_ctr.z, ZONE_SIGMA);
        o = o + w * (CU.bal1.x * E1 + CU.bal1.y * E2);
    }
    if ((CU.flags.x & 2u) != 0u) {
        // Saturation is subtractive: about the brightest channel M, so M and neutrals stay put.
        let m = max(o.x, max(o.y, o.z));
        o = vec3<f32>(m) + (o - vec3<f32>(m)) * CU.grade0.z;
    }
    return o;
}

// ---- C10: CDL and tetrahedral 3D LUT ---------------------------------------------------------------

fn cdl_apply(v: vec3<f32>) -> vec3<f32> {
    let sop = max(v * CU.cdl_slope.xyz + CU.cdl_offset.xyz, vec3<f32>(0.0));
    var x = sop;
    if (CU.cdl_power.x != 1.0) {
        x.x = pow_pos(sop.x, CU.cdl_power.x);
    }
    if (CU.cdl_power.y != 1.0) {
        x.y = pow_pos(sop.y, CU.cdl_power.y);
    }
    if (CU.cdl_power.z != 1.0) {
        x.z = pow_pos(sop.z, CU.cdl_power.z);
    }
    let sat = CU.cdl_slope.w;
    if (sat == 1.0) {
        return x;
    }
    let l = dot(vec3<f32>(0.2126, 0.7152, 0.0722), x);
    return vec3<f32>(l) + sat * (x - vec3<f32>(l));
}

// Position of x in a table of `size` entries over [mn, mx]: clamped, scaled to 0 ..= size - 1.
fn grid_pos(x: f32, mn: f32, mx: f32, size: u32) -> f32 {
    var c = mn;
    if (!is_nan_f(x)) {
        c = clamp(x, mn, mx);
    }
    return (c - mn) / (mx - mn) * f32(size - 1u);
}

fn lut_at(d: LutDesc, r: u32, g: u32, b: u32) -> vec3<f32> {
    return luts[d.cube.x + r + d.cube.y * (g + d.cube.y * b)].xyz;
}

// Tetrahedral interpolation (ym_color::lut::Lut3d::sample), manual: never hardware trilinear.
fn cube_sample(d: LutDesc, rgb: vec3<f32>) -> vec3<f32> {
    let size = d.cube.y;
    let pr = grid_pos(rgb.x, d.dmin.x, d.dmax.x, size);
    let pg = grid_pos(rgb.y, d.dmin.y, d.dmax.y, size);
    let pb = grid_pos(rgb.z, d.dmin.z, d.dmax.z, size);
    let i = min(u32(floor(pr)), size - 2u);
    let j = min(u32(floor(pg)), size - 2u);
    let k = min(u32(floor(pb)), size - 2u);
    let fr = pr - f32(i);
    let fg = pg - f32(j);
    let fb = pb - f32(k);
    let c000 = lut_at(d, i, j, k);
    let c111 = lut_at(d, i + 1u, j + 1u, k + 1u);
    var p1: vec3<f32>;
    var p2: vec3<f32>;
    var w0: f32;
    var w1: f32;
    var w2: f32;
    if (fr > fg) {
        if (fg > fb) {
            // R > G > B: c000 -> c100 -> c110 -> c111
            p1 = lut_at(d, i + 1u, j, k);
            p2 = lut_at(d, i + 1u, j + 1u, k);
            w0 = fr; w1 = fg; w2 = fb;
        } else if (fr > fb) {
            // R > B >= G: c000 -> c100 -> c101 -> c111
            p1 = lut_at(d, i + 1u, j, k);
            p2 = lut_at(d, i + 1u, j, k + 1u);
            w0 = fr; w1 = fb; w2 = fg;
        } else {
            // B >= R > G: c000 -> c001 -> c101 -> c111
            p1 = lut_at(d, i, j, k + 1u);
            p2 = lut_at(d, i + 1u, j, k + 1u);
            w0 = fb; w1 = fr; w2 = fg;
        }
    } else if (fb > fg) {
        // B > G >= R: c000 -> c001 -> c011 -> c111
        p1 = lut_at(d, i, j, k + 1u);
        p2 = lut_at(d, i, j + 1u, k + 1u);
        w0 = fb; w1 = fg; w2 = fr;
    } else if (fb > fr) {
        // G >= B > R: c000 -> c010 -> c011 -> c111
        p1 = lut_at(d, i, j + 1u, k);
        p2 = lut_at(d, i, j + 1u, k + 1u);
        w0 = fg; w1 = fb; w2 = fr;
    } else {
        // G >= R >= B: c000 -> c010 -> c110 -> c111
        p1 = lut_at(d, i, j + 1u, k);
        p2 = lut_at(d, i + 1u, j + 1u, k);
        w0 = fg; w1 = fr; w2 = fb;
    }
    return c000 + w0 * (p1 - c000) + w1 * (p2 - p1) + w2 * (c111 - p2);
}

fn shaper_channel(d: LutDesc, x: f32, mn: f32, mx: f32, c: u32) -> f32 {
    let size = d.shaper.x;
    let pos = grid_pos(x, mn, mx, size);
    let i = min(u32(floor(pos)), size - 2u);
    let f = pos - f32(i);
    let a = luts[d.cube.w + i][c];
    let b = luts[d.cube.w + i + 1u][c];
    return a + (b - a) * f;
}

// Shaper (if any), then the cube (if any): ym_color::lut::Lut::sample.
fn lut_sample(d: LutDesc, rgb: vec3<f32>) -> vec3<f32> {
    if (d.cube.y == 0u) {
        return rgb;
    }
    var x = rgb;
    if (d.cube.z != 0u) {
        x = vec3<f32>(
            shaper_channel(d, rgb.x, d.smin.x, d.smax.x, 0u),
            shaper_channel(d, rgb.y, d.smin.y, d.smax.y, 1u),
            shaper_channel(d, rgb.z, d.smin.z, d.smax.z, 2u),
        );
    }
    return cube_sample(d, x);
}

// CDL then LUT in the look's own encoding (LookTransform::apply_own_encoding).
fn look_own(v: vec3<f32>) -> vec3<f32> {
    var x = v;
    if (CU.lk.y != 0u) {
        x = cdl_apply(x);
    }
    return lut_sample(CU.look_lut, x);
}

// C10 for log-to-log looks, blended in LogC4.
fn c10_look(v: vec3<f32>) -> vec3<f32> {
    if (CU.flags.z != 1u) {
        return v;
    }
    var o: vec3<f32>;
    if (CU.lk.x == 0u) {
        o = look_own(v);
    } else {
        o = logc3_to_logc4(look_own(logc4_to_logc3(v)));
    }
    let k = CU.fl.x;
    if (k >= 1.0) {
        return o;
    }
    return v + k * (o - v);
}

// ---- C11: curves -----------------------------------------------------------------------------------

// Linear interpolation in a 1024-entry table, slope-1 extension outside [0, 1].
fn curve_eval(c: u32, x: f32) -> f32 {
    let base = T_CURVES + c * 1024u;
    if (x < 0.0) {
        return x + tables[base];
    }
    if (x > 1.0) {
        return x - 1.0 + tables[base + 1023u];
    }
    let pos = x * 1023.0;
    let i = min(u32(floor(pos)), 1022u);
    let f = pos - f32(i);
    let a = tables[base + i];
    let b = tables[base + i + 1u];
    return a + (b - a) * f;
}

fn c11_curves(v: vec3<f32>) -> vec3<f32> {
    var o = v;
    let bits = CU.flags.y;
    if ((bits & 1u) != 0u) {
        let l = logc4_encode(dot(Y_AWG4, logc4_decode3(v)));
        let shift = curve_eval(0u, l) - l;
        o = o + vec3<f32>(shift);
    }
    if ((bits & 2u) != 0u) {
        o.x = curve_eval(1u, o.x);
    }
    if ((bits & 4u) != 0u) {
        o.y = curve_eval(2u, o.y);
    }
    if ((bits & 8u) != 0u) {
        o.z = curve_eval(3u, o.z);
    }
    return o;
}

// ---- C12: display transform --------------------------------------------------------------------------

fn srgb_decode(e: f32) -> f32 {
    let a = abs(e);
    var x: f32;
    if (a <= 0.04045) {
        x = a / 12.92;
    } else {
        x = pow((a + 0.055) / 1.055, 2.4);
    }
    return copysign_f(x, e);
}

// A display-LUT output code value to display-linear in the LUT's primaries (negative gives 0).
fn disp_decode(code: f32) -> f32 {
    if (is_nan_f(code) || code <= 0.0) {
        return 0.0;
    }
    if (CU.dp.y == 1u) {
        return pow_pos(code, CU.fl.y);
    }
    return srgb_decode(code);
}

fn display_ours(v: vec3<f32>) -> vec3<f32> {
    if (CU.flags.w == 0u) {
        let ap0 = mv3(AWG4_TO_AP0_0, AWG4_TO_AP0_1, AWG4_TO_AP0_2, logc4_decode3(v));
        return aces2_display_linear(ap0);
    }
    var input = v;
    if (CU.dp.x == 1u) {
        input = logc4_to_logc3(v);
    }
    let codes = lut_sample(CU.disp_lut, input);
    let lin = vec3<f32>(disp_decode(codes.x), disp_decode(codes.y), disp_decode(codes.z));
    if (CU.dp.z != 0u) {
        return mv3(CU.disp_m0.xyz, CU.disp_m1.xyz, CU.disp_m2.xyz, lin);
    }
    return lin;
}

fn c12_display(v: vec3<f32>) -> vec3<f32> {
    let ours = display_ours(v);
    if (CU.flags.z != 2u) {
        return ours;
    }
    // A LogC3-to-Rec.709 look replaces C12; the blend is in display-linear.
    let k = CU.fl.x;
    if (k <= 0.0) {
        return ours;
    }
    let enc = look_own(logc4_to_logc3(v));
    var theirs = vec3<f32>(pow_pos(enc.x, 2.4), pow_pos(enc.y, 2.4), pow_pos(enc.z, 2.4));
    if (CU.dp.w != 0u) {
        theirs = mv3(CU.look_m0.xyz, CU.look_m1.xyz, CU.look_m2.xyz, theirs);
    }
    if (k >= 1.0) {
        return theirs;
    }
    return ours + k * (theirs - ours);
}

// ---- C13 ----------------------------------------------------------------------------------------------

fn encode_one(x: f32) -> f32 {
    if (CU.sp.x == 1u) {
        return unit(bt1886_inverse(x));
    }
    return unit(srgb_encode(x));
}

fn c13_encode(lin: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(encode_one(lin.x), encode_one(lin.y), encode_one(lin.z));
}

// ---- the chain -------------------------------------------------------------------------------------------

// The value at the requested stop (mode.x): 0 C6, 1 C7, 2 C8, 3 C9, 4 C10, 5 C11, 6 DNG (C11
// decoded to linear AWG4), 7 C12, 8 C13 (unquantised), 9 the Before view.
fn run_pixel(cam_in: vec3<f32>) -> vec3<f32> {
    let stop = CU.mode.x;
    let cam = vec3<f32>(sanitize(cam_in.x), sanitize(cam_in.y), sanitize(cam_in.z));
    let c6 = mv3(CU.m0.xyz, CU.m1.xyz, CU.m2.xyz, cam);
    if (stop == 0u) {
        return c6;
    }
    let c7 = c7_tone(c6);
    if (stop == 1u) {
        return c7;
    }
    let c8 = logc4_encode3(c7);
    if (stop == 2u) {
        return c8;
    }
    if (stop == 9u && CU.lk.w == 0u) {
        return vec3<f32>(unit(c8.x), unit(c8.y), unit(c8.z));
    }
    // L of stage C9: the LogC4 code of the luminance of the linear values entering C8.
    let l = logc4_encode(dot(Y_AWG4, c7));
    let c9 = c9_grade(c8, l);
    if (stop == 3u) {
        return c9;
    }
    if (stop == 9u) {
        return vec3<f32>(unit(c9.x), unit(c9.y), unit(c9.z));
    }
    let c10 = c10_look(c9);
    if (stop == 4u) {
        return c10;
    }
    let c11 = c11_curves(c10);
    if (stop == 5u) {
        return c11;
    }
    if (stop == 6u) {
        return logc4_decode3(c11);
    }
    let c12 = c12_display(c11);
    if (stop == 7u) {
        return c12;
    }
    return c13_encode(c12);
}

fn q8(v: f32) -> u32 {
    return u32(floor(unit(v) * 255.0 + 0.5));
}

fn dither8(v: f32, thr: f32) -> u32 {
    return u32(min(floor(unit(v) * 255.0 + thr), 255.0));
}

fn qbits(v: f32) -> u32 {
    let maxv = f32((1u << CU.mode.z) - 1u);
    return u32(floor(unit(v) * maxv + 0.5));
}

fn threshold(ax: u32, ay: u32) -> f32 {
    return tables[T_DITHER + (ay % 64u) * 64u + (ax % 64u)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * 2u;
    let y = gid.y;
    if (x >= CU.dims.z || y >= CU.dims.y) {
        return;
    }
    let s = (y + CU.src.w) * CU.src.x + x + CU.src.z;
    let spl = CU.src.y;
    let a = run_pixel(vec3<f32>(img_ld(s), img_ld(spl + s), img_ld(2u * spl + s)));
    let b = run_pixel(vec3<f32>(img_ld(s + 1u), img_ld(spl + s + 1u), img_ld(2u * spl + s + 1u)));
    let o = y * CU.dims.z + x;
    let pl = CU.dims.w;
    let kind = CU.mode.y;
    if (kind == 0u) {
        out[o] = bitcast<u32>(a.x);
        out[o + 1u] = bitcast<u32>(b.x);
        out[pl + o] = bitcast<u32>(a.y);
        out[pl + o + 1u] = bitcast<u32>(b.y);
        out[2u * pl + o] = bitcast<u32>(a.z);
        out[2u * pl + o + 1u] = bitcast<u32>(b.z);
    } else if (kind == 1u) {
        let ax = CU.origin.x + x;
        let ay = CU.origin.y + y;
        if (CU.mode.w != 0u) {
            let ta = threshold(ax, ay);
            let tb = threshold(ax + 1u, ay);
            out[o] = dither8(a.x, ta) | (dither8(a.y, ta) << 8u) | (dither8(a.z, ta) << 16u) | 0xff000000u;
            out[o + 1u] = dither8(b.x, tb) | (dither8(b.y, tb) << 8u) | (dither8(b.z, tb) << 16u) | 0xff000000u;
        } else {
            out[o] = q8(a.x) | (q8(a.y) << 8u) | (q8(a.z) << 16u) | 0xff000000u;
            out[o + 1u] = q8(b.x) | (q8(b.y) << 8u) | (q8(b.z) << 16u) | 0xff000000u;
        }
    } else {
        out[o >> 1u] = qbits(a.x) | (qbits(b.x) << 16u);
        out[(pl + o) >> 1u] = qbits(a.y) | (qbits(b.y) << 16u);
        out[(2u * pl + o) >> 1u] = qbits(a.z) | (qbits(b.z) << 16u);
    }
}
