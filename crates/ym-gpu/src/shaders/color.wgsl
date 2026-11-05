// The per-pixel colour chain C1 to C13 (docs/02-colour-pipeline.md) in one compute pass, mirroring
// ym_color::display::ColorChain stage by stage:
//
//   camera RGB --M--> C6 linear AWG4 --C7--> --C8--> LogC4 --nodes--> --C10 look--> --C12 cine rendering--> display linear --C13--> code values
//
// The nodes are the steps of the plan (ym_color::plan, docs/23 section 5.4): each corrector runs its trims, C9 (the
// primary grade), C9b (the colour warp) and C11 (the curves), a step that is at its identity or bypassed running no code,
// then the mask blend out = in + m (node(in) - in) (N-10: `node_mask`; a node with no mask rows, no Invert and no key
// keeps its own bits). The pictures live in eight registers and the keys in sixteen slots (private arrays,
// `run_nodes`); a plan that needs more is several passes of this kernel with f32 buffers between them (`pbuf`).
// Every node's parameters, Main's too, are a record of the storage buffer `nodes` (the layout of `NodeRecU` in
// kernels/color.rs, read a vec4 at a time by `nu` and `nf` at the rows R_*), followed by the mask rows (`MaskOpU`,
// MASK_ROW_VEC4 vec4 each, from `CU.mrow.x`); the uniform block holds the frame (window, mode, matrix, C7, look, display),
// the photo frame (`CU.photo`) and the pass. The curve is in front of the look since pipeline 0.9.0 (docs/02 C10, C11).
//
// The mask shapes are placed in the photo frame (docs/23 section 3.6): a source with six planes (`CU.sp.w`) carries each
// pixel's position in the photo, written by the sampler (planes 4 and 5, oriented fractions in [0, 1]); `CU.photo` is the
// photo's aspect and the narrowest edge ramp (MIN_RAMP_PIXELS output pixels) in units of its long edge. Without positions the shapes select nothing.
//
// C7 is the highlight range (docs/02 C7: a gain on the luma's stops over grey above a knee, up to the
// ALEXA clip; `c7_range`, its numbers in `CU.tone` and `CU.fl.z`, its expansion weighted by the achromatic gate of the
// pixel's purity, `achromatic_gate`, edges in `CU.gate.xy`) and then the low-light toe. C12 is the cine
// rendering (docs/02, `cine_display_linear` in cine.wgsl): the last LogC4 value (the nodes, then the look) decoded
// to linear AWG4 goes straight into it, with nothing in front, and the matrix from XYZ to the output's own primaries
// is in `CU.cine_m*`. The DNG stop (6) is that value decoded.
//
// `mode.x` selects the stop (so each stage can be checked against `ColorChain::trace`), `mode.y` the
// output format. Stops 3, 10 and 5 (and the Before view's "after grade" point) are Main's C9, C9b and C11 values;
// stop 11 ends the nodes after the step `CU.stopn.x` and runs the Output (a node's thumbnail); stop 12 ends at that
// step's mask m and writes it as grey code values (the mask view). One thread handles two horizontally adjacent
// pixels. All maths is f32; image samples are read through the storage accessors (f16 or f32), look cubes and curves are
// f32 storage (never hardware-filtered, never rounded to f16).
//
// Persistent tables: the dither thresholds (`tables`), the curve tables (`curves`, four 1024-entry tables for every node
// that has a curve) and the look cube (`look_luts`) are separate buffers the host keeps across frames and refills only
// when their content changes. What changes every frame is the uniform block (about 0.6 KB) and the node records (1200
// bytes a node, the colour warp's pins included, written beside the uniform for each frame).
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
    flags: vec4<u32>,    // z: look kind (0 none, 1 log-to-log, 2 display-baked); x, y, w: 0
    lk: vec4<u32>,       // x: look input (0 LogC4, 1 LogC3), y: look has CDL, w: before point (0 camera log, 1 after grade, 2 camera default)
    dp: vec4<u32>,       // x, y, z: 0; w: apply the Rec.709-to-output matrix (look_m*: display-baked look and the Before view)
    sp: vec4<u32>,       // x: output transfer (0 sRGB, 1 BT.1886), y: the source has a mask plane, z: letterbox colour (RGBA8), w: the source has the photo-frame position planes (4 and 5)
    m0: vec4<f32>,       // fused C1-C6 matrix rows
    m1: vec4<f32>,
    m2: vec4<f32>,
    tone: vec4<f32>,     // C7: x the range's knee, y the Sony clip (stops over grey; 1e30: none), z the toe's tau, w the range's expansion (stops; 0: none)
    fl: vec4<f32>,       // x: look intensity fraction, y: 0, z: the highlight range's cap (stops over grey; 1e30: none), w: 0
    cdl_slope: vec4<f32>,  // w: saturation
    cdl_offset: vec4<f32>,
    cdl_power: vec4<f32>,
    look_lut: LutDesc,
    cine_m0: vec4<f32>,  // C12, the cine rendering: XYZ -> the output's own primaries (rows of the matrix)
    cine_m1: vec4<f32>,
    cine_m2: vec4<f32>,
    look_m0: vec4<f32>,  // Rec.709 -> output primaries (display-baked look, Before view)
    look_m1: vec4<f32>,
    look_m2: vec4<f32>,
    cam0: vec4<f32>,     // Before view, camera default: camera RGB entering the chain -> scene-linear Rec.709, rows
    cam1: vec4<f32>,
    cam2: vec4<f32>,
    camt: vec4<f32>,     // tone curve n, s, k and the white level at which it reaches 1
    gate: vec4<f32>,     // the achromatic gate (docs/02 C7): x, y the highlight range's purity edge `from` and 1 / (to - from); z, w: 0
    plan: vec4<u32>,     // x: this pass's first record in `nodes`, y: its steps, z: bit 0: the plan's last pass (it writes `out`), w: the plan's result (a reference, see REF_*)
    stopn: vec4<u32>,    // x: the record at which stops 11 and 12 end (NODE_NONE: none), y, z, w: 0
    mrow: vec4<u32>,     // x: the first mask row in `nodes`, in vec4 rows (after the records), y, z, w: 0
    photo: vec4<f32>,    // the photo frame: x, y: the photo's width and height over its long edge, z: the narrowest edge ramp (MIN_RAMP_PIXELS output pixels) in units of the long edge, w: 0
    rst: vec4<u32>,      // the raster planes (binding 7): x: words in a plane, y: the first row of this dispatch in the window, z: a plane's row stride (elements), w: the planes (0: none)
};

// One step of the plan with the node it runs (docs/23 section 5.4): the flags and the wiring of the step, then the node's
// parameters. Main's are a record like every other node's. A control at its identity has its bit clear in `ctl`, so the
// kernel runs no code for it (the integers are decided on the host, in f64: nothing here rests on a rounding, K8).
// A record is REC_ROWS vec4 rows, the layout of `NodeRecU` (kernels/color.rs; the offsets R_* are written from the Rust
// struct):
//   R_CTL   (u32) x: N_* bits, y: grade bits (1 contrast, 2 saturation), z: curve bits (1 luma, 2 r, 4 g, 8 b), w: the first of the node's four curve tables (in `curves`, 1024 entries each)
//   R_WIRE  (u32) x: the picture it reads (REF_*), y: the key it reads (NODE_NONE: none), z: its mask rows, w: the first of them
//   R_OUTS  (u32) x: the register its picture goes to, y: the plane of `pbuf` it goes to, z: the key slot its mask goes to, w: the plane of `pbuf` (each NODE_NONE: not written)
//   R_GAIN  (f32) the trims' gain in linear AWG4 (xyz)
//   R_GRADE0 x: contrast slope k at grey, y: brightness, z: saturation factor (< 1 blend, > 1 boost F); w: 0
//   R_ZONE_AMT shadows, highlights, black, white (each / 100);  R_ZONE_CTR their centres (LogC4 codes)
//   R_BAL0 (a, b) shadows, (a, b) midtones;  R_BAL1 (a, b) highlights;  R_BAL_CTR zone centres
//   R_WARP_N (u32) C9b: x moving pins (0: the stage is skipped), y locked pins, z flow steps
//   R_WARP_MV WARP_MOVE_ROWS rows, three a moving pin (c9b_warp);  R_WARP_LK WARP_LOCK_ROWS rows, one a locked pin (c9b_warp)
// A mask row is MASK_ROW_VEC4 rows (the layout of `MaskOpU`): (u32) x: MK_* kind, y: MF_* flags, z: the raster plane
// (NODE_NONE: none), w: the opacity as f32 bits; then the kind's parameters as f32 (`ym_color::mask::MaskOp::params`):
// Linear cx, cy, sin, cos | feather; Radial cx, cy, sin, cos | rx, ry, feather; Colour range a0, b0, width, q;
// Brightness range low - q, high, q.

@group(0) @binding(0) var<uniform> CU: ChainU;
@group(0) @binding(1) var<storage, read> tables: array<f32>;       // dither thresholds
@group(0) @binding(2) var<storage, read> curves: array<f32>;       // four 1024-entry curve tables for each node that has a curve
@group(0) @binding(3) var<storage, read> look_luts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> img: array<ST_ELEM>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read> nodes: array<vec4<u32>>;   // the plan's steps with their nodes, in plan order, then the mask rows
@group(0) @binding(7) var<storage, read> rasters: array<u32>;      // the raster masks' u16 planes of the window, two pixels to a word (docs/23 section 5.5)
@group(0) @binding(8) var<storage, read_write> pbuf: array<f32>;   // the f32 planes that cross from one pass to the next (a picture is three planes, a key one)

//@ACCESSORS img ro

// One row of the node records, as u32 or as f32 (the buffer is raw so that the mask rows can follow the records).
fn nu(ni: u32, row: u32) -> vec4<u32> {
    return nodes[ni * REC_ROWS + row];
}

fn nf(ni: u32, row: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(nodes[ni * REC_ROWS + row]);
}

// ---- The achromatic gate (docs/02 C7) ----------------------------------------------------------------

// ym_color::tone::AchromaticGate: g = 1 - smoothstep(from, to, p) of the purity p = (max - min) / max of a linear AWG4
// pixel, held to 2 (the CPU's PURITY_MAX) so that nothing is large (K8). `lo` (the CPU's `from`) and 1 / (to - from) are the lanes of
// `CU.gate`. Exactly 1.0 for a grey pixel (max == min gives p = 0 and t = 0) and for every p up to `from`. Called only
// where the stage expands (above the C7 knee), so the largest channel is at least the luma,
// over 0.36; a largest channel at or under 0 (never reached) gives 1. The gate is continuous in the pixel.
fn achromatic_gate(rgb: vec3<f32>, lo: f32, inv_span: f32) -> f32 {
    let mx = max(rgb.x, max(rgb.y, rgb.z));
    let mn = min(rgb.x, min(rgb.y, rgb.z));
    if (mx <= 0.0) {
        return 1.0;
    }
    let p = min((mx - mn) / mx, 2.0);
    let t = clamp((p - lo) * inv_span, 0.0, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

// ---- C7: highlight range and low-light toe ---------------------------------------------------------

// The highlight range of ym_color::tone::HighlightRange (docs/02 C7). With s the stops over grey of the
// luma (Rec.709 weights on the AWG4 triple), s' = s + E * phi(u) above the knee, u = (s - knee) / (clip -
// knee), phi = u^2 up to the clip and 2u - 1 above it, then s' = min(s', cap); the pixel is multiplied by
// 2^(s' - s). Where s' is s (below the knee and the cap, or no range at all) the pixel is returned
// untouched, bit for bit. The host writes the knee and the clip into `CU.tone.xy`, the expansion E into
// `CU.tone.w` (0: none) and the cap into `CU.fl.z`; no range is a knee and a cap of 1e30 (HIGHLIGHT_OFF_STOPS,
// finite for Metal's fast-math, K8). Every value here is finite: the camera samples are sanitised to at
// most 1e30, so with any exposure the controls allow the luma stays under about 1e32, its stops under
// about 110 and the gain at the cap over about 2^-102, a normal f32. The two branch points (the knee and
// u = 1) are where the map is C1, so a rounding that moves a pixel across one moves its value by nothing
// measurable. The expansion E is weighted by the achromatic gate of the pixel's purity (`achromatic_gate`, the
// edges in `CU.gate.xy`): s' = s + g * E * phi, then the cap, which the gate does not touch. A grey pixel has
// g = 1.0 exactly and so the old result bit for bit.
fn c7_range(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(LUMA_709, rgb);
    if (is_nan_f(y) || y <= 0.0) {
        return rgb;
    }
    let s = log2(y / HIGHLIGHT_GREY);
    let knee = CU.tone.x;
    let e = CU.tone.w;
    var sp = s;
    if (e > 0.0 && s > knee) {
        // E > 0 only when the span is at least HIGHLIGHT_MIN_SPAN_STOPS (0.25), so the division is safe.
        let u = (s - knee) / (CU.tone.y - knee);
        sp = s + (achromatic_gate(rgb, CU.gate.x, CU.gate.y) * e) * select(2.0 * u - 1.0, u * u, u <= 1.0);
    }
    sp = min(sp, CU.fl.z);
    if (sp == s) {
        return rgb;
    }
    return rgb * exp2(sp - s);
}

// C7: the highlight range, then the toe on the gained pixel's norm n = max(R, G, B):
// n^2 / (n^2 + tau^2) as 1 / (1 + (tau / n)^2), 0 for n = 0; tau = 0 returns the range's output untouched.
// tau / n is limited to 1e18 in size so that its square stays finite (K8): the scale is then at most
// 1e-36, which is the CPU's 0 to every digit that reaches a picture.
fn c7_tone(rgb: vec3<f32>) -> vec3<f32> {
    let g = c7_range(rgb);
    let tau = CU.tone.z;
    if (tau == 0.0) {
        return g;
    }
    let n = max(g.x, max(g.y, g.z));
    var toe = 0.0;
    if (n != 0.0) {
        let r = clamp(tau / n, -1.0e18, 1.0e18);
        toe = 1.0 / (1.0 + r * r);
    }
    return g * toe;
}

// ---- C9: primary grade ---------------------------------------------------------------------------

fn bump(l: f32, mu: f32, sigma: f32) -> f32 {
    let z = (l - mu) / sigma;
    return exp(-0.5 * z * z);
}

// Contrast (ym_color::grade::contrast_curve, docs/02 C9): the tone curve T_k on the luma, fixed at black
// (CONTRAST_BLACK, the code of linear 0), grey and the top code value (CONTRAST_TOP). Between them
// h_k(t) = k t / (1 + (k - 1) t) scaled into each half, beyond them the line of slope 1 / k through the end.
// k (the record's grade0.x) is 2^(0.6 c / 100), from 0.66 to 1.52: t stays in [0, 1] on the curved pieces, so every
// divisor is at least 0.66 (K8: nothing rests on an infinity, and nothing on a rounding at the 2e-4 gate).
fn contrast_h(t: f32, k: f32) -> f32 {
    return k * t / (1.0 + (k - 1.0) * t);
}

fn contrast_curve(x: f32, k: f32) -> f32 {
    if (x >= CONTRAST_TOP) {
        return CONTRAST_TOP + (x - CONTRAST_TOP) / k;
    }
    if (x >= GRAY) {
        return GRAY + (CONTRAST_TOP - GRAY) * contrast_h((x - GRAY) / (CONTRAST_TOP - GRAY), k);
    }
    if (x > CONTRAST_BLACK) {
        return GRAY - (GRAY - CONTRAST_BLACK) * contrast_h((GRAY - x) / (GRAY - CONTRAST_BLACK), k);
    }
    return CONTRAST_BLACK + (x - CONTRAST_BLACK) / k;
}

// Contrast with its colour step (ym_color::grade::contrast_pixel, docs/02 C9): u = v + T(l) - l on all
// three channels, then u's chroma c_u = (e1.u, e2.u) scaled by f = 1 + beta (|c_p| / |c_u| - 1), with
// c_p the chroma of T on each channel of v and beta = CONTRAST_COLOR, keeping its direction:
// u + (f - 1) (c_u.x e1 + c_u.y e2). |c_u| below CONTRAST_NEUTRAL leaves the pixel at u. As e1 and e2
// sum to 0, c_u is the chroma of v, and it is taken from v (as on the CPU): in f32 the rounding of the
// offset would otherwise add to it.
//
// Fast-math (K8). Nothing rests on a rounding at the 2e-4 gate, nor on an infinity or a NaN:
// * an exact neutral (equal channels) returns u first, whatever a fused dot product makes of its chroma
//   (with FMA, e1.v of three equal codes can round to a few 1e-9 instead of 0);
// * the neutral test squares c_u only where both components are below CONTRAST_NEUTRAL (1e-9), so the
//   square cannot overflow, and one that underflows to 0 is a neutral anyway;
// * both chromas are divided by m = max(|c_u.x|, |c_u.y|), at least 7e-10 past the test, so their
//   squares stay near 1 for any code (a sanitised 1e30 makes codes near 1e32, whose squares would
//   overflow f32);
// * the squared ratio is limited to [min(k, 1/k)^2, max(k, 1/k)^2] (k from 0.66 to 1.52), the range of
//   T's slope, which the exact ratio never leaves: the square root never sees 0, even where f32 rounds
//   the per-channel curve of a chroma of a few ulp to a neutral.
// A C port of this function compiled with clang -O3 -ffast-math -ffp-contract=fast -mfma -mrecip=all
// against -O2 -ffp-contract=off (200,000 pixels: exact neutrals, chromas of one ulp and of 1e-13, codes
// to +-20 and garbage codes to +-1.4e32, k over its range): every output finite, exact neutrals neutral
// in both, the two at most 3.1e-7 apart among photo codes.
fn c9_contrast(v: vec3<f32>, l: f32, k: f32) -> vec3<f32> {
    let u = v + vec3<f32>(contrast_curve(l, k) - l);
    if (v.x == v.y && v.y == v.z) {
        return u;
    }
    let cu = vec2<f32>(dot(E1, v), dot(E2, v));
    let m = max(abs(cu.x), abs(cu.y));
    if (m < CONTRAST_NEUTRAL && dot(cu, cu) < CONTRAST_NEUTRAL * CONTRAST_NEUTRAL) {
        return u;
    }
    let p = vec3<f32>(contrast_curve(v.x, k), contrast_curve(v.y, k), contrast_curve(v.z, k));
    let a = cu / m;
    let b = vec2<f32>(dot(E1, p), dot(E2, p)) / m;
    let lo = min(k, 1.0 / k);
    let hi = max(k, 1.0 / k);
    let ratio = sqrt(clamp(dot(b, b) / dot(a, a), lo * lo, hi * hi));
    let g = CONTRAST_COLOR * (ratio - 1.0);
    return u + g * (cu.x * E1 + cu.y * E2);
}

// Below this distance r of the darkest channel under the luminance, the boost's chroma scale is taken
// from its series (f32 would lose the digits of 1 - q^F to cancellation; see c9_saturation).
const SAT_SERIES_LIMIT: f32 = 0.05;

// Saturation (ym_color::grade::saturate, docs/02 C9): in linear AWG4, the chroma rgb - Y scaled about the
// luminance Y by lambda, rgb' = Y + lambda (rgb - Y), then encoded again. `f` (the record's grade0.z) below 1 is the
// blend 1 + s / 100; above 1 it is F = 1 + SATURATION_BOOST s / 100 and lambda = (1 - q^F) / r with
// q = min / Y and r = 1 - q (the CPU's -expm1(F ln(1 - r)) / r): F for r <= 0 (every channel at or above Y,
// the CPU's r < 1e-9), falling to 1 as the darkest channel nears 0. A pixel with Y <= 0 or a channel at or
// below 0 is returned as it came for F > 1, as is a pixel with a code above SAT_CODE_LIMIT (7.5) for any f.
//
// Fast-math (K8) and f32: the decode of a code up to 7.5 is at most 2^120 and the encode of the result at
// most about 1e37, both finite (a code of 10, which Contrast -100 can make of a sanitised 1e30, would
// overflow the decode). The division by Y happens only for 0 < min < Y, and q is kept at least 1e-30, so
// pow's log2 sees a normal number. For r < SAT_SERIES_LIMIT, 1 - q^F in f32 would cancel (an ulp of q^F
// over r is 1e-5 of lambda at r = 0.005, and lambda times a chroma that can be many times Y where the blue
// weight of Y is negative); there lambda is the series F (1 - (F-1)/2 r (1 - (F-2)/3 r (1 - (F-3)/4 r))),
// whose first neglected term is below 1e-7 of lambda. Above it a fast pow off by 1e-6 moves lambda by at
// most 2e-5. The CPU reference is f64 throughout; nothing here rests on a rounding at the 2e-4 gate.
fn c9_saturation(v: vec3<f32>, f: f32) -> vec3<f32> {
    if (max(v.x, max(v.y, v.z)) > SAT_CODE_LIMIT) {
        return v;
    }
    let rgb = logc4_decode3(v);
    let y = dot(Y_AWG4, rgb);
    var lambda = f;
    if (f > 1.0) {
        let m = min(rgb.x, min(rgb.y, rgb.z));
        if (y <= 0.0 || m <= 0.0) {
            return v;
        }
        if (m < y) {
            let q = max(m / y, 1.0e-30);
            let r = 1.0 - q;
            if (r < SAT_SERIES_LIMIT) {
                let t3 = 1.0 - 0.25 * (f - 3.0) * r;
                let t2 = 1.0 - (f - 2.0) / 3.0 * r * t3;
                lambda = f * (1.0 - 0.5 * (f - 1.0) * r * t2);
            } else {
                lambda = (1.0 - pow(q, f)) / r;
            }
        }
    }
    return logc4_encode3(vec3<f32>(y) + lambda * (rgb - vec3<f32>(y)));
}

fn c9_grade(v: vec3<f32>, l: f32, ni: u32) -> vec3<f32> {
    var o = v;
    if ((nu(ni, R_CTL).y & 1u) != 0u) {
        // Contrast: one offset for all three channels, the luma's change on the curve, then part of the
        // colour the curve on each channel would add.
        o = c9_contrast(o, l, nf(ni, R_GRADE0).x);
    }
    var shift = nf(ni, R_GRADE0).y * STOP * bump(l, GRAY, BRIGHTNESS_SIGMA);
    if (nf(ni, R_ZONE_AMT).x != 0.0) {
        shift += nf(ni, R_ZONE_AMT).x * ZONE_CONTROL_SIGMA * bump(l, nf(ni, R_ZONE_CTR).x, ZONE_CONTROL_SIGMA);
    }
    if (nf(ni, R_ZONE_AMT).y != 0.0) {
        shift += nf(ni, R_ZONE_AMT).y * ZONE_CONTROL_SIGMA * bump(l, nf(ni, R_ZONE_CTR).y, ZONE_CONTROL_SIGMA);
    }
    if (nf(ni, R_ZONE_AMT).z != 0.0) {
        shift += nf(ni, R_ZONE_AMT).z * ZONE_CONTROL_SIGMA * bump(l, nf(ni, R_ZONE_CTR).z, ZONE_CONTROL_SIGMA);
    }
    if (nf(ni, R_ZONE_AMT).w != 0.0) {
        shift += nf(ni, R_ZONE_AMT).w * ZONE_CONTROL_SIGMA * bump(l, nf(ni, R_ZONE_CTR).w, ZONE_CONTROL_SIGMA);
    }
    o = o + vec3<f32>(shift);
    if (nf(ni, R_BAL0).x != 0.0 || nf(ni, R_BAL0).y != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, nf(ni, R_BAL_CTR).x, ZONE_SIGMA);
        o = o + w * (nf(ni, R_BAL0).x * E1 + nf(ni, R_BAL0).y * E2);
    }
    if (nf(ni, R_BAL0).z != 0.0 || nf(ni, R_BAL0).w != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, nf(ni, R_BAL_CTR).y, ZONE_SIGMA);
        o = o + w * (nf(ni, R_BAL0).z * E1 + nf(ni, R_BAL0).w * E2);
    }
    if (nf(ni, R_BAL1).x != 0.0 || nf(ni, R_BAL1).y != 0.0) {
        let w = BALANCE_STRENGTH * bump(l, nf(ni, R_BAL_CTR).z, ZONE_SIGMA);
        o = o + w * (nf(ni, R_BAL1).x * E1 + nf(ni, R_BAL1).y * E2);
    }
    if ((nu(ni, R_CTL).y & 2u) != 0u) {
        o = c9_saturation(o, nf(ni, R_GRADE0).z);
    }
    return o;
}

// ---- C9b: colour warp ----------------------------------------------------------------------------

// The pins of ym_color::warp::Warp (docs/02 C9b), prepared on the host by `Warp::new` (clamped, still
// pins left out, the pivot as a LogC4 code, the number of flow steps): `warp_n.x` moving pins of
// three rows each in `warp_mv`, [s.x, s.y, delta.x, delta.y], [1 / (2 rho^2), shadows, highlights,
// pivot code], [exposure in stops, saturation, 0, 0], `warp_n.y` locked pins in `warp_lk`, [l.x, l.y,
// 1 / (2 rho^2), 0], and the flow's `warp_n.z` steps, all of the record `nodes[ni]`. No moving pin skips the stage: the output is
// then bit for bit what it was before the warp existed.
//
// The flow: in N steps every pin's Gaussian slides from its source by the share of its move this
// pixel takes, to `s + gamma w delta`, and the pixel follows,
// `c += (gamma / N) sum(w phi delta) / max(1, sum phi)` with the centres at `s + (k / N) gamma w delta`;
// gamma (the neutral guard times the locks' hold) and the tonal weights w are the pixel's own, taken
// once. The exposure and the saturation are one-step formulas at the pixel's own chroma; the
// saturation K then scales the chroma the pixel ends on, (1 + K) (c + dd).
//
// Fast-math (K8). Nothing here rests on a rounding (the gate is 2e-4 in LogC4) except in the thin
// sliver just behind the source of a long, narrow move, which the flow stretches along the path and
// where it magnifies whatever rounding there is (docs/02 C9b); and nothing may rest on an infinity or
// a NaN (Metal's fast-math may assume there are none):
// * every Gaussian goes through `warp_gauss`, whose exponent is limited to WARP_EXP_LIMIT: an
//   absurd chroma (a garbage pixel of 1e30) gives exp(-80), a normal number, never exp(-inf);
// * a neutral pixel returns before any square root is taken (fast-math may compute sqrt(x) as
//   x * rsqrt(x), a NaN at 0); the threshold, |c| < 1e-12, is far above the denormals a fast-math
//   device flushes, and the CPU moves such a pixel by less than 1e-20;
// * the lock product is of factors in [0, 1], every division is by max(1, sum of reaches) >= 1, and
//   the step count is at least 1 and at most WARP_MAX_STEPS whatever the uniform holds;
// * the tone L decodes v limited to WARP_LUMA_CEIL, so the decode's exp2 stays finite in f32 (it
//   overflows from a code of about 8; real pixels after C9 stay below 3, and from 0.8 up the tonal
//   weight is the highlights' already).
// docs/02 C9b gives the check of a C port of this function under clang -ffast-math.
const WARP_NEUTRAL_R2: f32 = 1.0e-24;
const WARP_LUMA_CEIL: f32 = 7.5;

fn warp_gauss(d2: f32, inv_two_sigma_sq: f32) -> f32 {
    return exp(-min(d2 * inv_two_sigma_sq, WARP_EXP_LIMIT));
}

// A moving pin's tonal weight at the tone `l`: Shadows below the pivot, Highlights above it.
fn warp_tonal_weight(b: vec4<f32>, l: f32) -> f32 {
    let t = smoothstep(b.w - WARP_TONE_HALF_WIDTH, b.w + WARP_TONE_HALF_WIDTH, l);
    return b.y + (b.z - b.y) * t;
}

fn c9b_warp(v: vec3<f32>, ni: u32) -> vec3<f32> {
    let moves = min(nu(ni, R_WARP_N).x, WARP_MAX_PINS);
    if (moves == 0u) {
        return v;
    }
    let c = vec2<f32>(dot(E1, v), dot(E2, v));
    let r2 = dot(c, c);
    if (r2 < WARP_NEUTRAL_R2) {
        return v;
    }
    let guard = smoothstep(0.0, WARP_NEUTRAL_GUARD, sqrt(r2));
    var hold = 1.0;
    let locks = min(nu(ni, R_WARP_N).y, WARP_LOCK_ROWS);
    for (var j = 0u; j < locks; j++) {
        let lk = nf(ni, R_WARP_LK + j);
        let d = c - lk.xy;
        hold = hold * (1.0 - warp_gauss(dot(d, d), lk.z));
    }
    if (hold <= 0.0) {
        return v;
    }
    let gamma = guard * hold;
    let l = logc4_encode(dot(Y_AWG4, logc4_decode3(min(v, vec3<f32>(WARP_LUMA_CEIL)))));
    // The exposure and the saturation, at the pixel's own chroma with the pins at their sources.
    var reach = 0.0;
    var exposure = 0.0;
    var saturation = 0.0;
    for (var i = 0u; i < moves; i++) {
        let a = nf(ni, R_WARP_MV + 3u * i);
        let b = nf(ni, R_WARP_MV + 3u * i + 1u);
        let x = nf(ni, R_WARP_MV + 3u * i + 2u);
        let d = c - a.xy;
        let phi = warp_gauss(dot(d, d), b.x);
        reach = reach + phi;
        let wphi = warp_tonal_weight(b, l) * phi;
        exposure = exposure + wphi * x.x;
        saturation = saturation + wphi * x.y;
    }
    let e = gamma / max(reach, 1.0) * exposure;
    // K, kept at -1 or more so that 1 + K >= 0 (|K| <= 1 already: the weights and gamma are in [0, 1],
    // every pin's saturation in [-1, 1]).
    let k = max(gamma / max(reach, 1.0) * saturation, -1.0);
    // The flow, the move gathered in `dd`.
    let steps = clamp(nu(ni, R_WARP_N).z, 1u, WARP_MAX_STEPS);
    let n = f32(steps);
    let h = gamma / n;
    var dd = vec2<f32>(0.0);
    for (var k = 0u; k < steps; k++) {
        let tau = f32(k) / n;
        let p = c + dd;
        var sum = 0.0;
        var shift = vec2<f32>(0.0);
        for (var i = 0u; i < moves; i++) {
            let a = nf(ni, R_WARP_MV + 3u * i);
            let b = nf(ni, R_WARP_MV + 3u * i + 1u);
            let w = warp_tonal_weight(b, l);
            let d = p - (a.xy + (tau * (gamma * w)) * a.zw);
            let phi = warp_gauss(dot(d, d), b.x);
            sum = sum + phi;
            shift = shift + (w * phi) * a.zw;
        }
        dd = dd + (h / max(sum, 1.0)) * shift;
    }
    // Saturation scales the chroma the pixel ends on, (1 + K) (c + dd); K = 0 leaves dd as it is.
    if (k != 0.0) {
        dd = dd + k * (c + dd);
    }
    return v + dd.x * E1 + dd.y * E2 + vec3<f32>(e * STOP);
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

// Linear interpolation in a 1024-entry table (the table number `t` of `curves`), slope-1 extension outside [0, 1].
fn curve_eval(t: u32, x: f32) -> f32 {
    let base = t * 1024u;
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

// The four curves of the node record `ni`: its tables are `ctl.w` to `ctl.w + 3` (luma, R, G, B).
fn c11_curves(v: vec3<f32>, ni: u32) -> vec3<f32> {
    var o = v;
    let bits = nu(ni, R_CTL).z;
    let t0 = nu(ni, R_CTL).w;
    if ((bits & 1u) != 0u) {
        let l = logc4_encode(dot(Y_AWG4, logc4_decode3(v)));
        let shift = curve_eval(t0, l) - l;
        o = o + vec3<f32>(shift);
    }
    if ((bits & 2u) != 0u) {
        o.x = curve_eval(t0 + 1u, o.x);
    }
    if ((bits & 4u) != 0u) {
        o.y = curve_eval(t0 + 2u, o.y);
    }
    if ((bits & 8u) != 0u) {
        o.z = curve_eval(t0 + 3u, o.z);
    }
    return o;
}

// ---- C12: the cine rendering -----------------------------------------------------------------------------

// The cine rendering (cine.wgsl) for the output's primaries (`CU.cine_m*`) of C11's value `v` (`lin`: the same colour as linear AWG4). `exact` is the display colour the Film
// pass asked its return for (0 where there is none): the return is the rendering's inverse, so for a colour it could reach
// the rendering of its answer *is* that display colour, and the f32 round trip through LogC4, the inverse and the
// rendering is only noise of about 1e-6 (docs/24 section 3.9c). Where the rendering agrees with it within `SNAP_TOLERANCE`
// the exact colour is the answer; where it does not (a colour the rendering cannot make: the return gave the nearest one)
// the rendering is.
const SNAP_TOLERANCE: f32 = 2.0e-5;

fn display_ours(lin: vec3<f32>, exact: vec3<f32>) -> vec3<f32> {
    let made = cine_display_linear(lin);
    if (exact.x > 0.0 && exact.y > 0.0 && exact.z > 0.0) {
        let tol = SNAP_TOLERANCE * exact + vec3<f32>(1.0e-9);
        let d = abs(made - exact);
        if (d.x <= tol.x && d.y <= tol.y && d.z <= tol.z) {
            return exact;
        }
    }
    return made;
}

fn c12_display(v: vec3<f32>, lin: vec3<f32>, exact: vec3<f32>) -> vec3<f32> {
    let ours = display_ours(lin, exact);
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

// ---- the nodes (docs/23 section 5.4) -------------------------------------------------------------------------

// One node on one pixel: its trims, C9, C9b and C11 (docs/23 section 1.3), each left out when its bit in `ctl.x` is clear,
// the whole node when `N_SKIP` is set (bypassed, or every step at its identity). `v_in` is the picture the node reads;
// `c7` is the linear AWG4 triple C8 was encoded from, which is the node's exact linear input when it reads the Input
// (`from_input`: the luma of C9 is taken from it, with no decode of an encode); a node that reads another node's picture
// decodes it. `stage` is 1, 2 or 3 to return Main's value after C9, C9b or C11 (the trace's stops), 0 otherwise.
// The bits are decided on the host in f64, so no rounding here decides whether a step runs.
fn node_apply(ni: u32, v_in: vec3<f32>, c7: vec3<f32>, from_input: bool, stage: u32) -> vec3<f32> {
    let bits = nu(ni, R_CTL).x;
    if ((bits & N_SKIP) != 0u) {
        return v_in;
    }
    var v = v_in;
    var lin = c7;
    // 1. The trims: a gain in linear AWG4.
    if ((bits & N_TRIM) != 0u) {
        if (!from_input) {
            lin = logc4_decode3(v_in);
        }
        lin = lin * nf(ni, R_GAIN).xyz;
        v = logc4_encode3(lin);
    }
    // 2. C9, with L the LogC4 code of the luminance of the linear values entering it.
    if ((bits & N_GRADE) != 0u) {
        if (!from_input && (bits & N_TRIM) == 0u) {
            lin = logc4_decode3(v);
        }
        let l = logc4_encode(dot(Y_AWG4, lin));
        v = c9_grade(v, l, ni);
    }
    if (stage == 1u) {
        return v;
    }
    // 3. C9b.
    if ((bits & N_WARP) != 0u) {
        v = c9b_warp(v, ni);
    }
    if (stage == 2u) {
        return v;
    }
    // 4. C11.
    if ((bits & N_CURVES) != 0u) {
        v = c11_curves(v, ni);
    }
    return v;
}

// ---- masks (docs/23 section 4, docs/algorithms/06 sections 3 and 4) ----------------------------------------------------

// S(t) = t^2 (3 - 2t) for t held to [0, 1].
fn smooth01(t: f32) -> f32 {
    let c = clamp(t, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
}

// One row's own s in [0, 1] at a pixel (ym_color::mask::MaskOp::select). `p0` and `p1` are the row's parameters, `v_in` the
// node's input, `l_in` its luma code, `pos` the pixel's place in the photo frame (the shapes select nothing without one).
fn mask_select(kind: u32, plane: u32, rpix: u32, p0: vec4<f32>, p1: vec4<f32>, v_in: vec3<f32>, l_in: f32, pos: vec2<f32>) -> f32 {
    if (kind == MK_LINEAR) {
        if (CU.sp.w == 0u) {
            return 0.0;
        }
        // The signed distance from the centre line along n = (-sin, cos) in units of the long edge, and the width of the
        // transition, centred on the line: the feather, and at least one output pixel (CU.photo.z).
        let x = (pos.x - p0.x) * CU.photo.x;
        let y = (pos.y - p0.y) * CU.photo.y;
        let s = -p0.z * x + p0.w * y;
        let w = max(p1.x, CU.photo.z);
        return 1.0 - smooth01((s + 0.5 * w) / w);
    }
    if (kind == MK_RADIAL) {
        if (CU.sp.w == 0u) {
            return 0.0;
        }
        // The pixel rotated by -angle in units of the radii, and the ramp q in units of d: the feather, and at least the
        // narrowest ramp (CU.photo.z) measured across the outline, which is |grad d| d-units per unit of L (1/r for a circle,
        // 1/rx to 1/ry round an ellipse). It is centred on the contour the feather alone has (d = 1 - f/2), so that a feather
        // of a pixel or more runs from 1 - f to the outline and Feather 0 is centred on the outline.
        let x = (pos.x - p0.x) * CU.photo.x;
        let y = (pos.y - p0.y) * CU.photo.y;
        let u = (p0.w * x + p0.z * y) / p1.x;
        let w = (-p0.z * x + p0.w * y) / p1.y;
        let d = sqrt(u * u + w * w);
        let grad = clamp(sqrt((u / p1.x) * (u / p1.x) + (w / p1.y) * (w / p1.y)) / max(d, 1.0e-9), 1.0 / max(p1.x, p1.y), 1.0 / min(p1.x, p1.y));
        let q = max(p1.z, CU.photo.z * grad);
        return 1.0 - smooth01((d - (1.0 - 0.5 * p1.z - 0.5 * q)) / q);
    }
    if (kind == MK_COLOR) {
        let a = dot(E1, v_in) - p0.x;
        let b = dot(E2, v_in) - p0.y;
        let d = sqrt(a * a + b * b);
        return 1.0 - smooth01((d - p0.z) / p0.w);
    }
    if (kind == MK_BRIGHTNESS) {
        return smooth01((l_in - p0.x) / p0.z) * (1.0 - smooth01((l_in - p0.y) / p0.z));
    }
    if (kind == MK_RASTER) {
        // The plane the render made for this row (a u16 code over 65535 at the pixel); a row with no plane, and a chain
        // rendered with no planes, select nothing.
        if (plane >= CU.rst.w) {
            return 0.0;
        }
        let word = rasters[plane * CU.rst.x + (rpix >> 1u)];
        return f32((word >> ((rpix & 1u) * 16u)) & 0xffffu) / 65535.0;
    }
    return 0.0;
}

// The node's mask m (ym_color::mask::PreparedMasks::mask). `key_in` is the linked key's value when `has_key`: an implicit first
// region row. Each row's a = opacity (invert ? 1 - s : s); the regions and the ranges fold apart, in list order, an add as
// 1 - (1 - acc)(1 - a) and a subtract as acc (1 - a), the first row of a group taken exactly (a added, 1 - a subtracted);
// m = R K with a group that has no row 1, and the node's Invert gives 1 - m.
fn node_mask(ni: u32, v_in: vec3<f32>, c7: vec3<f32>, from_input: bool, pos: vec2<f32>, rpix: u32, key_in: f32, has_key: bool) -> f32 {
    let bits = nu(ni, R_CTL).x;
    let rows = nu(ni, R_WIRE);
    var l_in = 0.0;
    if ((bits & N_LUMA) != 0u) {
        var lin = c7;
        if (!from_input) {
            lin = logc4_decode3(v_in);
        }
        l_in = logc4_encode(dot(Y_AWG4, lin));
    }
    var regions = key_in;
    var has_regions = has_key;
    var ranges = 0.0;
    var has_ranges = false;
    for (var i = 0u; i < rows.z; i++) {
        let b = CU.mrow.x + (rows.w + i) * MASK_ROW_VEC4;
        let head = nodes[b];
        let s = mask_select(head.x, head.z, rpix, bitcast<vec4<f32>>(nodes[b + 1u]), bitcast<vec4<f32>>(nodes[b + 2u]), v_in, l_in, pos);
        var a = s;
        if ((head.y & MF_INVERT) != 0u) {
            a = 1.0 - s;
        }
        a = bitcast<f32>(head.w) * a;
        let subtract = (head.y & MF_SUBTRACT) != 0u;
        if (head.x == MK_COLOR || head.x == MK_BRIGHTNESS) {
            if (!has_ranges) {
                ranges = select(a, 1.0 - a, subtract);
                has_ranges = true;
            } else if (subtract) {
                ranges = ranges * (1.0 - a);
            } else {
                ranges = 1.0 - (1.0 - ranges) * (1.0 - a);
            }
        } else {
            if (!has_regions) {
                regions = select(a, 1.0 - a, subtract);
                has_regions = true;
            } else if (subtract) {
                regions = regions * (1.0 - a);
            } else {
                regions = 1.0 - (1.0 - regions) * (1.0 - a);
            }
        }
    }
    var m = 1.0;
    if (has_regions) {
        m = regions;
    }
    if (has_ranges) {
        m = m * ranges;
    }
    if ((bits & N_INVERT) != 0u) {
        m = 1.0 - m;
    }
    return m;
}

struct NodesOut {
    v: vec3<f32>,
    done: bool,    // the value is the pixel's final value (a stop inside the nodes, or a pass that is not the last)
};

fn plane_pic(plane: u32, n: u32, pix: u32) -> vec3<f32> {
    return vec3<f32>(pbuf[plane * n + pix], pbuf[(plane + 1u) * n + pix], pbuf[(plane + 2u) * n + pix]);
}

// The steps of this pass in registers and key slots, then the plan's result (the value the Output reads). `c7` and `c8` are
// the Input's picture, `pix` the pixel's place in the window (the planes of `pbuf` are window-sized), `pos` its place in the
// photo frame (for the mask shapes).
fn run_nodes(c7: vec3<f32>, c8: vec3<f32>, pix: u32, pos: vec2<f32>, rpix: u32) -> NodesOut {
    let stop = CU.mode.x;
    // Main's own stages for the trace's stops: C9 (3, and the Before view's "after grade" point), C9b (10), C11 (5).
    var main_stage = 0u;
    if (stop == 3u || (stop == 9u && CU.lk.w == 1u)) {
        main_stage = 1u;
    } else if (stop == 10u) {
        main_stage = 2u;
    } else if (stop == 5u) {
        main_stage = 3u;
    }
    let n = CU.dims.x * CU.dims.y;
    var regs: array<vec3<f32>, PLAN_REGISTERS>;
    var slots: array<f32, PLAN_KEYS>;
    let first = CU.plan.x;
    let count = CU.plan.y;
    for (var k = 0u; k < count; k++) {
        let ni = first + k;
        let wire = nu(ni, R_WIRE);
        let outs = nu(ni, R_OUTS);
        var v_in = c8;
        var from_input = true;
        let kind = wire.x >> REF_SHIFT;
        let at = wire.x & REF_INDEX;
        if (kind == REF_REGISTER) {
            v_in = regs[at];
            from_input = false;
        } else if (kind == REF_PICTURE) {
            v_in = plane_pic(at, n, pix);
            from_input = false;
        }
        var stage = 0u;
        if ((nu(ni, R_CTL).x & N_MAIN) != 0u) {
            stage = main_stage;
        }
        let film = (nu(ni, R_CTL).x & N_FILM) != 0u;
        var nv = v_in;
        if (!film) {
            nv = node_apply(ni, v_in, c7, from_input, stage);
        }
        if (stage != 0u) {
            return NodesOut(nv, true);
        }
        // The mask blend (docs/23 section 1.3): out = v_in + m (nv - v_in). A node with no mask rows, no Invert and no key
        // (`N_BLEND` clear, an integer decided on the host) has m = 1 and takes no blend, so its output is the node's own
        // bits; one that has takes m exactly 1 as nv and exactly 0 as v_in, and the blend between. m is also the node's key
        // output, which a bypassed or skipped node still has (its picture is its input), and the mask view's grey.
        let ctl = nu(ni, R_CTL).x;
        let blends = (ctl & N_BLEND) != 0u;
        let skipped = (ctl & N_SKIP) != 0u;
        var key_in = 0.0;
        var has_key = false;
        if (wire.y != NODE_NONE) {
            has_key = true;
            let ki = wire.y & REF_INDEX;
            if ((wire.y >> REF_SHIFT) == REF_SLOT) {
                key_in = slots[ki];
            } else {
                key_in = pbuf[ki * n + pix];
            }
        }
        let key_written = outs.z != NODE_NONE || outs.w != NODE_NONE || (CU.mode.x == 12u && ni == CU.stopn.x);
        var m = 1.0;
        if (blends && (!skipped || key_written)) {
            m = node_mask(ni, v_in, c7, from_input, pos, rpix, key_in, has_key);
        }
        var out_v = nv;
        if (film) {
            // A Film node (docs/24 section 3.9c): the picture the Film passes made, in the first planes of `pbuf`, blended by the
            // node's mask times its Strength (`R_GAIN.x`), as the CPU's `run_film`: m is also the key output, bypass or not.
            m = 1.0;
            if (blends || key_written) {
                m = node_mask(ni, v_in, c7, from_input, pos, rpix, key_in, has_key);
            }
            m = m * nf(ni, R_GAIN).x;
            if ((ctl & N_FILM_LIVE) != 0u && m != 0.0) {
                let made = plane_pic(nu(ni, R_CTL).w, n, pix);
                if (m == 1.0) {
                    out_v = made;
                } else {
                    out_v = v_in + m * (made - v_in);
                }
            }
        } else if (blends && !skipped) {
            if (m == 0.0) {
                out_v = v_in;
            } else if (m != 1.0) {
                out_v = v_in + m * (nv - v_in);
            }
        }
        if (outs.x != NODE_NONE) {
            regs[outs.x] = out_v;
        }
        if (outs.y != NODE_NONE) {
            pbuf[outs.y * n + pix] = out_v.x;
            pbuf[(outs.y + 1u) * n + pix] = out_v.y;
            pbuf[(outs.y + 2u) * n + pix] = out_v.z;
        }
        if (outs.z != NODE_NONE) {
            slots[outs.z] = m;
        }
        if (outs.w != NODE_NONE) {
            pbuf[outs.w * n + pix] = m;
        }
        if (ni == CU.stopn.x) {
            // Stop 11: the picture after this node goes through the Output; stop 12: its mask m, as grey code values.
            if (CU.mode.x == 12u) {
                return NodesOut(vec3<f32>(m), true);
            }
            return NodesOut(out_v, false);
        }
    }
    if ((CU.plan.z & 1u) == 0u) {
        return NodesOut(vec3<f32>(0.0), true);
    }
    if (main_stage != 0u) {
        // Main is not in the plan: the trace's C9, C9b and C11 are C8.
        return NodesOut(c8, true);
    }
    let res = CU.plan.w;
    let rkind = res >> REF_SHIFT;
    let ri = res & REF_INDEX;
    var v = c8;
    if (rkind == REF_REGISTER) {
        v = regs[ri];
    } else if (rkind == REF_PICTURE) {
        v = plane_pic(ri, n, pix);
    } else if (rkind == REF_SLOT) {
        v = vec3<f32>(slots[ri]);
    } else if (rkind == REF_KEY) {
        v = vec3<f32>(pbuf[ri * n + pix]);
    }
    return NodesOut(v, false);
}

// ---- the chain -------------------------------------------------------------------------------------------

// The value at the requested stop (mode.x): 0 C6, 1 C7, 2 C8, 13 the guide of Refine edges (the Input's luma code, three times), 3 C9, 10 C9b, 5 C11 (the curve), 4 C10 (the look,
// after the curve), 6 DNG (C10 decoded to linear AWG4), 7 C12, 8 C13 (unquantised), 9 the Before view (the
// camera default encoded for the output, or the LogC4 signal at C8 or C9 as code values: the reading point
// "after the grade" is C9, before the warp, as ColorChain::apply_before_pixel), 11 the Output after the node
// `CU.stopn.x`, 12 that node's mask as grey code values. Stops 3, 10 and 5 are Main's stages (StageTrace's), whatever
// else the plan runs. The stage numbers are names and did not change when the curve moved in front of the look; they
// are the trace's fields (`StageTrace::c10_logc4`, `c11_logc4`). `pix` is the pixel's place in the window, `pos` its place in
// the photo frame.
fn run_pixel(cam_in: vec3<f32>, pix: u32, pos: vec2<f32>, rpix: u32) -> vec3<f32> {
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
    if (stop == 13u) {
        // The guide of Refine edges: the Input's luma code (what a Brightness range of a node that reads the Input sees,
        // no node's grade in it), held to [0, 1], in all three planes (ym_color::ColorChain::guide_pixel).
        let g = unit(logc4_encode(dot(Y_AWG4, c7)));
        return vec3<f32>(g, g, g);
    }
    let c8 = px_logc4_encode3(c7);
    if (stop == 2u) {
        return c8;
    }
    if (stop == 9u && CU.lk.w == 0u) {
        return vec3<f32>(unit(c8.x), unit(c8.y), unit(c8.z));
    }
    // The nodes: the plan's steps of this pass.
    let nodes_out = run_nodes(c7, c8, pix, pos, rpix);
    if (stop == 9u) {
        return vec3<f32>(unit(nodes_out.v.x), unit(nodes_out.v.y), unit(nodes_out.v.z));
    }
    if (nodes_out.done) {
        return nodes_out.v;
    }
    let c11 = nodes_out.v;
    let c10 = c10_look(c11);
    if (stop == 4u) {
        return c10;
    }
    if (stop == 6u) {
        // The DNG's pixels: the last LogC4 value (the nodes, then the look) decoded to linear AWG4 (the inverse of
        // a display rendering that is not ours is the CPU's alone, ColorChain::apply_dng_pixel).
        return logc4_decode3(c10);
    }
    // C11's value is exactly a value we know more about: the Input's own C8 (its linear AWG4 is C7, with no decode of an
    // encode), or a Film node's picture (the exact display colour its return was asked for). A bit-for-bit match, an
    // integer test; a value any node, look or curve has touched matches neither.
    var lin10 = vec3<f32>(0.0);
    var exact = vec3<f32>(0.0);
    let c10_bits = bitcast<vec3<u32>>(c10);
    if (all(c10_bits == bitcast<vec3<u32>>(c8))) {
        lin10 = c7;
    } else {
        lin10 = px_logc4_decode3(c10);
        let n = CU.dims.x * CU.dims.y;
        for (var k = 0u; k < CU.stopn.y; k++) {
            if (all(c10_bits == bitcast<vec3<u32>>(plane_pic(6u * k, n, pix)))) {
                exact = plane_pic(6u * k + 3u, n, pix);
            }
        }
    }
    let c12 = c12_display(c10, lin10, exact);
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
    let pix = y * CU.dims.x + x;
    // The pixel's place in the raster planes (the window's, two to a word): the dispatch may be a band of rows of it.
    let rpix = (CU.rst.y + y) * CU.rst.z + x;
    // The pixels' places in the photo frame, from the sampler's fifth and sixth planes (the shapes of the nodes read them).
    var pos_a = vec2<f32>(0.0);
    var pos_b = vec2<f32>(0.0);
    if (CU.sp.w != 0u) {
        pos_a = vec2<f32>(img_ld(4u * spl + s), img_ld(5u * spl + s));
        if (two) {
            pos_b = vec2<f32>(img_ld(4u * spl + s + 1u), img_ld(5u * spl + s + 1u));
        }
    }
    if (!gone_a) {
        a = run_pixel(vec3<f32>(img_ld(s), img_ld(spl + s), img_ld(2u * spl + s)), pix, pos_a, rpix);
    }
    if (two && !gone_b) {
        b = run_pixel(vec3<f32>(img_ld(s + 1u), img_ld(spl + s + 1u), img_ld(2u * spl + s + 1u)), pix + 1u, pos_b, rpix + 1u);
    }
    if ((CU.plan.z & 1u) == 0u) {
        // A pass that is not the last one has written its buffers; the picture is the last pass's.
        return;
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
