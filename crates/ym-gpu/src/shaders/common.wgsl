// Shared helpers for every kernel. Per-pixel maths is f32 in registers; the formulas are those of
// docs/02-colour-pipeline.md and the CPU reference in ym-color. Constants (LC4_*, LC3_*, matrices)
// are generated from ym_color::consts at shader build time, so nothing is typed twice.

// NaN test that survives fast-math: looks at the bits, not at the value.
fn is_nan_f(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// A NaN camera sample counts as 0 and infinities are limited to +-1e30 (as ym_color::display).
fn sanitize(x: f32) -> f32 {
    if (is_nan_f(x)) {
        return 0.0;
    }
    return clamp(x, -1.0e30, 1.0e30);
}

// Clamp to [0, 1], NaN to 0 (ym_color::display `unit`).
fn unit(x: f32) -> f32 {
    if (is_nan_f(x)) {
        return 0.0;
    }
    return clamp(x, 0.0, 1.0);
}

// Index mirrored into 0..n without repeating the edge sample (ym_gpu::cpu::mirror).
fn mirror_idx(i: i32, n: i32) -> i32 {
    if (n <= 1) {
        return 0;
    }
    let period = 2 * (n - 1);
    var j = ((i % period) + period) % period;
    if (j >= n) {
        j = period - j;
    }
    return j;
}

// Rows of a 3x3 matrix times a column vector.
fn mv3(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
}

// |a| with the sign of s.
fn copysign_f(a: f32, s: f32) -> f32 {
    let m = abs(a);
    return select(m, -m, s < 0.0);
}

// sRGB encode (IEC 61966-2-1), odd-symmetric (ym_color::encode::srgb_encode).
fn srgb_encode(x: f32) -> f32 {
    let a = abs(x);
    var e: f32;
    if (a <= 0.0031308) {
        e = 12.92 * a;
    } else {
        e = 1.055 * pow(a, 1.0 / 2.4) - 0.055;
    }
    return copysign_f(e, x);
}

// BT.1886 inverse EOTF, zero black level: x^(1/2.4), odd-symmetric.
fn bt1886_inverse(x: f32) -> f32 {
    let a = abs(x);
    var e = 0.0;
    if (a > 0.0) {
        e = pow(a, 1.0 / 2.4);
    }
    return copysign_f(e, x);
}

fn pow_pos(x: f32, p: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    return pow(x, p);
}

fn logc4_encode(x: f32) -> f32 {
    if (x >= LC4_T) {
        return (log2(LC4_A * x + 64.0) - 6.0) / 14.0 * LC4_B + LC4_C;
    }
    return (x - LC4_T) / LC4_S;
}

fn logc4_decode(e: f32) -> f32 {
    if (e >= 0.0) {
        return (exp2(14.0 * (e - LC4_C) / LC4_B + 6.0) - 64.0) / LC4_A;
    }
    return e * LC4_S + LC4_T;
}

fn logc4_encode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(logc4_encode(v.x), logc4_encode(v.y), logc4_encode(v.z));
}

fn logc4_decode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(logc4_decode(v.x), logc4_decode(v.y), logc4_decode(v.z));
}

fn logc3_encode(x: f32) -> f32 {
    if (x > LC3_CUT) {
        return LC3_C * log2(LC3_A * x + LC3_B) * LOG10_2 + LC3_D;
    }
    return LC3_E * x + LC3_F;
}

fn logc3_decode(t: f32) -> f32 {
    if (t > LC3_E * LC3_CUT + LC3_F) {
        return (exp2((t - LC3_D) / LC3_C * LOG2_10) - LC3_B) / LC3_A;
    }
    return (t - LC3_F) / LC3_E;
}

fn logc3_encode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(logc3_encode(v.x), logc3_encode(v.y), logc3_encode(v.z));
}

fn logc3_decode3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(logc3_decode(v.x), logc3_decode(v.y), logc3_decode(v.z));
}

// LogC4 -> linear AWG4 -> AWG3 -> LogC3 EI 800 (ym_color::logc::LogC4::to_logc3_ei800).
fn logc4_to_logc3(v: vec3<f32>) -> vec3<f32> {
    let lin = mv3(AWG4_TO_AWG3_0, AWG4_TO_AWG3_1, AWG4_TO_AWG3_2, logc4_decode3(v));
    return logc3_encode3(lin);
}

// LogC3 EI 800 -> linear AWG3 -> AWG4 -> LogC4 (ym_color::logc::LogC3Ei800::to_logc4).
fn logc3_to_logc4(v: vec3<f32>) -> vec3<f32> {
    let lin = mv3(AWG3_TO_AWG4_0, AWG3_TO_AWG4_1, AWG3_TO_AWG4_2, logc3_decode3(v));
    return logc4_encode3(lin);
}
