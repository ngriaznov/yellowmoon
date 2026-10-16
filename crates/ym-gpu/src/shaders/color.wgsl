// The per-pixel colour chain C1 to C13 (docs/02-colour-pipeline.md) in one compute pass, mirroring
// ym_color::display::ColorChain stage by stage:
//
//   camera RGB --M--> C6 linear AWG4 --C7--> --C8--> LogC4 --C9--> --C10 look--> --C11 curves-->
//   --C12 ACES 2 or ARRI display LUT--> display linear --C13--> code values
//
// C12 on the ACES 2 path starts with the white lift (docs/02, step 1b): a scene-linear gain on the tones
// above +1 stop (`white_lift_gain`, amount and 1 / span in `CU.fl.z` and `CU.fl.w`, 0 for none and for a
// display LUT), then the placement and ACES 2. The DNG stop (6) carries it too.
//
// `mode.x` selects the stop (so each stage can be checked against `ColorChain::trace`), `mode.y` the
// output format. One thread handles two horizontally adjacent pixels. All maths is f32; image
// samples are read through the storage accessors (f16 or f32), look cubes and curves are f32
// storage (never hardware-filtered, never rounded to f16).
//
// Persistent tables (G-04): the dither thresholds and the ACES 2 tables (`tables`), the four curve
// tables (`curves`) and the look and display cubes (`look_luts`, `disp_luts`) are separate buffers the
// host keeps across frames and refills only when their content changes. What changes every frame is
// this one uniform block (about 1 KB, the ACES 2 scalars included: K3 allows one uniform buffer).
//
// A source with a fourth plane (`sp.y` = 1) carries the letterbox mask of the sampler (1 outside the
// photo): masked pixels are not computed, they are 0 in a stage output (whose own fourth plane
// receives the mask) and the background colour in RGBA8.

struct LutDesc {
    cube: vec4<u32>,     // x: offset of the cube in `luts` (vec4 units), y: size (0: no cube), z: has shaper, w: shaper offset
    shaper: vec4<u32>,   // x: shaper size
    dmin: vec4<f32>,     // cube domain
    dmax: vec4<f32>,
    smin: vec4<f32>,     // shaper domain
    smax: vec4<f32>,
};

struct ChainU {
    dims: vec4<u32>,     // x: window width, y: window height, z: destination stride, w: destination plane length
    src: vec4<u32>,      // x: src stride, y: src plane length, z: dx, w: dy (window origin in the source)
    origin: vec4<u32>,   // x, y: absolute pixel position of the window's (0, 0), for the dither; z, w: its place in the destination
    mode: vec4<u32>,     // x: stop, y: out kind (0 f32 planar, 1 u8 rgba, 2 u16 planar), z: bits, w: dither
    flags: vec4<u32>,    // x: grade bits (1 contrast, 2 saturation), y: curve bits (1 luma, 2 r, 4 g, 8 b), z: look kind (0 none, 1 log-to-log, 2 display-baked), w: display kind (0 ACES 2, 1 display LUT)
    lk: vec4<u32>,       // x: look input (0 LogC4, 1 LogC3), y: look has CDL, w: before point (0 camera log, 1 after grade, 2 camera default)
    dp: vec4<u32>,       // x: display LUT input (0 LogC4, 1 LogC3), y: display LUT transfer (0 sRGB, 1 gamma), z: apply display-LUT primaries matrix, w: apply the Rec.709-to-output matrix (look_m*: display-baked look and the Before view)
    sp: vec4<u32>,       // x: output transfer (0 sRGB, 1 BT.1886), y: the source has a mask plane, z: letterbox colour (RGBA8)
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
    fl: vec4<f32>,       // x: look intensity fraction, y: display LUT gamma, z: white lift amount (stops, 0: none), w: 1 / the lift's ramp span (1 / stops)
    cdl_slope: vec4<f32>,  // w: saturation
    cdl_offset: vec4<f32>,
    cdl_power: vec4<f32>,
    look_lut: LutDesc,
    disp_lut: LutDesc,
    disp_m0: vec4<f32>,  // display LUT primaries -> output primaries
    disp_m1: vec4<f32>,
    disp_m2: vec4<f32>,
    look_m0: vec4<f32>,  // Rec.709 -> output primaries (display-baked look, Before view)
    look_m1: vec4<f32>,
    look_m2: vec4<f32>,
    cam0: vec4<f32>,     // Before view, camera default: camera RGB entering the chain -> scene-linear Rec.709, rows
    cam1: vec4<f32>,
    cam2: vec4<f32>,
    camt: vec4<f32>,     // tone curve n, s, k and the white level at which it reaches 1
    aces: AcesU,         // the ACES 2 scalars and matrices (aces2.wgsl)
};

@group(0) @binding(0) var<uniform> CU: ChainU;
@group(0) @binding(1) var<storage, read> tables: array<f32>;       // dither thresholds, then the ACES 2 tables
@group(0) @binding(2) var<storage, read> curves: array<f32>;       // the four 1024-entry curve tables
@group(0) @binding(3) var<storage, read> look_luts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> disp_luts: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> img: array<ST_ELEM>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

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

//@BEGIN_LUT_FUNCS
fn lut_at@P@(d: LutDesc, r: u32, g: u32, b: u32) -> vec3<f32> {
    return @L@[d.cube.x + r + d.cube.y * (g + d.cube.y * b)].xyz;
}

// Tetrahedral interpolation (ym_color::lut::Lut3d::sample), manual: never hardware trilinear.
fn cube_sample@P@(d: LutDesc, rgb: vec3<f32>) -> vec3<f32> {
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
    let c000 = lut_at@P@(d, i, j, k);
    let c111 = lut_at@P@(d, i + 1u, j + 1u, k + 1u);
    var p1: vec3<f32>;
    var p2: vec3<f32>;
    var w0: f32;
    var w1: f32;
    var w2: f32;
    if (fr > fg) {
        if (fg > fb) {
            // R > G > B: c000 -> c100 -> c110 -> c111
            p1 = lut_at@P@(d, i + 1u, j, k);
            p2 = lut_at@P@(d, i + 1u, j + 1u, k);
            w0 = fr; w1 = fg; w2 = fb;
        } else if (fr > fb) {
            // R > B >= G: c000 -> c100 -> c101 -> c111
            p1 = lut_at@P@(d, i + 1u, j, k);
            p2 = lut_at@P@(d, i + 1u, j, k + 1u);
            w0 = fr; w1 = fb; w2 = fg;
        } else {
            // B >= R > G: c000 -> c001 -> c101 -> c111
            p1 = lut_at@P@(d, i, j, k + 1u);
            p2 = lut_at@P@(d, i + 1u, j, k + 1u);
            w0 = fb; w1 = fr; w2 = fg;
        }
    } else if (fb > fg) {
        // B > G >= R: c000 -> c001 -> c011 -> c111
        p1 = lut_at@P@(d, i, j, k + 1u);
        p2 = lut_at@P@(d, i, j + 1u, k + 1u);
        w0 = fb; w1 = fg; w2 = fr;
    } else if (fb > fr) {
        // G >= B > R: c000 -> c010 -> c011 -> c111
        p1 = lut_at@P@(d, i, j + 1u, k);
        p2 = lut_at@P@(d, i, j + 1u, k + 1u);
        w0 = fg; w1 = fb; w2 = fr;
    } else {
        // G >= R >= B: c000 -> c010 -> c110 -> c111
        p1 = lut_at@P@(d, i, j + 1u, k);
        p2 = lut_at@P@(d, i + 1u, j + 1u, k);
        w0 = fg; w1 = fr; w2 = fb;
    }
    return c000 + w0 * (p1 - c000) + w1 * (p2 - p1) + w2 * (c111 - p2);
}

fn shaper_channel@P@(d: LutDesc, x: f32, mn: f32, mx: f32, c: u32) -> f32 {
    let size = d.shaper.x;
    let pos = grid_pos(x, mn, mx, size);
    let i = min(u32(floor(pos)), size - 2u);
    let f = pos - f32(i);
    let a = @L@[d.cube.w + i][c];
    let b = @L@[d.cube.w + i + 1u][c];
    return a + (b - a) * f;
}

// Shaper (if any), then the cube (if any): ym_color::lut::Lut::sample.
fn lut_sample@P@(d: LutDesc, rgb: vec3<f32>) -> vec3<f32> {
    if (d.cube.y == 0u) {
        return rgb;
    }
    var x = rgb;
    if (d.cube.z != 0u) {
        x = vec3<f32>(
            shaper_channel@P@(d, rgb.x, d.smin.x, d.smax.x, 0u),
            shaper_channel@P@(d, rgb.y, d.smin.y, d.smax.y, 1u),
            shaper_channel@P@(d, rgb.z, d.smin.z, d.smax.z, 2u),
        );
    }
    return cube_sample@P@(d, x);
}

//@END_LUT_FUNCS

// CDL then LUT in the look's own encoding (LookTransform::apply_own_encoding).
fn look_own(v: vec3<f32>) -> vec3<f32> {
    var x = v;
    if (CU.lk.y != 0u) {
        x = cdl_apply(x);
    }
    return lut_sample_look(CU.look_lut, x);
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
    let base = c * 1024u;
    if (x < 0.0) {
        return x + curves[base];
    }
    if (x > 1.0) {
        return x - 1.0 + curves[base + 1023u];
    }
    let pos = x * 1023.0;
    let i = min(u32(floor(pos)), 1022u);
    let f = pos - f32(i);
    let a = curves[base + i];
    let b = curves[base + i + 1u];
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

// ---- C12 step 1b: the white lift ------------------------------------------------------------------------

// The scene-linear gain of ym_color::exposure::white::WhiteLift: 2^(A * smoothstep((s - knee) / span)) with s the
// stops over middle grey of the luma (Rec.709 weights on the AWG4 triple). Exactly 1 up to the knee. Only
// called when the amount is not 0 (`CU.fl.z`), which the host sets for the ACES 2 path alone.
fn white_lift_gain(lin: vec3<f32>) -> f32 {
    let y = dot(LUMA_709, lin);
    if (is_nan_f(y) || y <= 0.0) {
        return 1.0;
    }
    let stops = log2(y * (1.0 / WHITE_LIFT_GREY));
    let t = clamp((stops - WHITE_LIFT_KNEE) * CU.fl.w, 0.0, 1.0);
    return exp2(CU.fl.z * (t * t * (3.0 - 2.0 * t)));
}

// C11 decoded to linear AWG4 and lifted: the scene values ACES 2 takes in (before the placement).
fn lifted_scene(v: vec3<f32>) -> vec3<f32> {
    let lin = logc4_decode3(v);
    if (CU.fl.z == 0.0) {
        return lin;
    }
    return lin * white_lift_gain(lin);
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
        let ap0 = mv3(AWG4_TO_AP0_0, AWG4_TO_AP0_1, AWG4_TO_AP0_2, lifted_scene(v));
        return aces2_display_linear(ap0);
    }
    var input = v;
    if (CU.dp.x == 1u) {
        input = logc4_to_logc3(v);
    }
    let codes = lut_sample_disp(CU.disp_lut, input);
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

fn c13_encode_chain(lin: vec3<f32>) -> vec3<f32> {
    return c13_encode(lin, CU.sp.x);
}

// ---- the Before view: the camera default ---------------------------------------------------------------

// The camera tone curve of ym_color::camera_default::CameraTone: y = min(1, k * t / (t + s)), t = x^n, one
// channel; 0 for NaN and x <= 0, exactly 1 from the white level up (so t never overflows).
fn cam_tone(x: f32) -> f32 {
    if (is_nan_f(x) || x <= 0.0) {
        return 0.0;
    }
    if (x >= CU.camt.w) {
        return 1.0;
    }
    let t = pow(x, CU.camt.x);
    return min(CU.camt.z * t / (t + CU.camt.y), 1.0);
}

// ym_color::display::ColorChain::apply_before_pixel at the camera default: the folded 3x3 (as-shot white
// balance, body matrix, camera gain) to scene-linear Rec.709, the curve per channel, the output primaries
// and C13. `cam` is the sanitised camera RGB the chain receives.
fn before_camera_default(cam: vec3<f32>) -> vec3<f32> {
    let lin = mv3(CU.cam0.xyz, CU.cam1.xyz, CU.cam2.xyz, cam);
    var shown = vec3<f32>(cam_tone(lin.x), cam_tone(lin.y), cam_tone(lin.z));
    if (CU.dp.w != 0u) {
        shown = mv3(CU.look_m0.xyz, CU.look_m1.xyz, CU.look_m2.xyz, shown);
    }
    return c13_encode_chain(shown);
}

// ---- the chain -------------------------------------------------------------------------------------------

// The value at the requested stop (mode.x): 0 C6, 1 C7, 2 C8, 3 C9, 4 C10, 5 C11, 6 DNG (C11
// decoded to linear AWG4), 7 C12, 8 C13 (unquantised), 9 the Before view (the camera default encoded for
// the output, or the LogC4 signal at C8 or C9 as code values).
fn run_pixel(cam_in: vec3<f32>) -> vec3<f32> {
    let stop = CU.mode.x;
    let cam = vec3<f32>(sanitize(cam_in.x), sanitize(cam_in.y), sanitize(cam_in.z));
    if (stop == 9u && CU.lk.w == 2u) {
        return before_camera_default(cam);
    }
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
        // The DNG's pixels: C11 decoded, with the white lift when C12 is our ACES 2 (the host sets
        // the amount only then); a look with its own display rendering has its own highlights.
        if (CU.flags.z == 2u && CU.fl.x > 0.0) {
            return logc4_decode3(c11);
        }
        return lifted_scene(c11);
    }
    let c12 = c12_display(c11);
    if (stop == 7u) {
        return c12;
    }
    return c13_encode_chain(c12);
}

fn threshold(ax: u32, ay: u32) -> f32 {
    return tables[T_DITHER + (ay % 64u) * 64u + (ax % 64u)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x * 2u;
    let y = gid.y;
    if (x >= CU.dims.x || y >= CU.dims.y) {
        return;
    }
    // The second pixel of the pair exists unless the window has an odd width.
    let two = x + 1u < CU.dims.x;
    let s = (y + CU.src.w) * CU.src.x + x + CU.src.z;
    let spl = CU.src.y;
    let masked = CU.sp.y != 0u;
    var gone_a = false;
    var gone_b = false;
    if (masked) {
        gone_a = img_ld(3u * spl + s) != 0.0;
        gone_b = two && img_ld(3u * spl + s + 1u) != 0.0;
    }
    var a = vec3<f32>(0.0);
    var b = vec3<f32>(0.0);
    if (!gone_a) {
        a = run_pixel(vec3<f32>(img_ld(s), img_ld(spl + s), img_ld(2u * spl + s)));
    }
    if (two && !gone_b) {
        b = run_pixel(vec3<f32>(img_ld(s + 1u), img_ld(spl + s + 1u), img_ld(2u * spl + s + 1u)));
    }
    let o = (y + CU.origin.w) * CU.dims.z + x + CU.origin.z;
    let pl = CU.dims.w;
    let kind = CU.mode.y;
    if (kind == 0u) {
        out[o] = bitcast<u32>(a.x);
        out[pl + o] = bitcast<u32>(a.y);
        out[2u * pl + o] = bitcast<u32>(a.z);
        if (masked) {
            out[3u * pl + o] = bitcast<u32>(select(0.0, 1.0, gone_a));
        }
        if (two) {
            out[o + 1u] = bitcast<u32>(b.x);
            out[pl + o + 1u] = bitcast<u32>(b.y);
            out[2u * pl + o + 1u] = bitcast<u32>(b.z);
            if (masked) {
                out[3u * pl + o + 1u] = bitcast<u32>(select(0.0, 1.0, gone_b));
            }
        }
    } else if (kind == 1u) {
        let ax = CU.origin.x + x;
        let ay = CU.origin.y + y;
        let dithered = CU.mode.w != 0u;
        let ta = threshold(ax, ay);
        let tb = threshold(ax + 1u, ay);
        var wa = c13_code8(a.x, ta, dithered) | (c13_code8(a.y, ta, dithered) << 8u)
            | (c13_code8(a.z, ta, dithered) << 16u) | 0xff000000u;
        if (gone_a) {
            wa = CU.sp.z;
        }
        out[o] = wa;
        if (two) {
            var wb = c13_code8(b.x, tb, dithered) | (c13_code8(b.y, tb, dithered) << 8u)
                | (c13_code8(b.z, tb, dithered) << 16u) | 0xff000000u;
            if (gone_b) {
                wb = CU.sp.z;
            }
            out[o + 1u] = wb;
        }
    } else {
        let bits = CU.mode.z;
        out[o >> 1u] = c13_qbits(a.x, bits) | (c13_qbits(b.x, bits) << 16u);
        out[(pl + o) >> 1u] = c13_qbits(a.y, bits) | (c13_qbits(b.y, bits) << 16u);
        out[(2u * pl + o) >> 1u] = c13_qbits(a.z, bits) | (c13_qbits(b.z, bits) << 16u);
    }
}
