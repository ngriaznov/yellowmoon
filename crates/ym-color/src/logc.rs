//! LogC3 (EI 800) and LogC4 encode and decode, including the exact LogC3 to LogC4 conversion (`docs/02-colour-pipeline.md` sections LogC4 and LogC3).
//!
//! The scalar functions are the exact published formulas in f64:
//!
//! ```text
//! LogC4  encode(x) = x >= t ? (log2(a x + 64) - 6) / 14 * b + c : (x - t) / s
//!        decode(E) = E >= 0 ? (2^(14 (E - c) / b + 6) - 64) / a : E s + t
//! LogC3  encode(x) = x > cut ? c log10(a x + b) + d : e x + f
//!        decode(t) = t > e cut + f ? (10^((t - d) / c) - b) / a : (t - f) / e
//! ```
//!
//! Negative linear values pass through the linear segments unclipped (the LogC4 spec requires it
//! for out-of-gamut colours). [`LogC4`] and [`LogC3Ei800`] are per-pixel newtypes so encoded and
//! linear triples cannot be mixed.

use crate::consts::{LOGC3_EI800 as C3, LOGC4 as C4, Vec3};
use crate::gamut::{LinearAwg3, LinearAwg4};

/// LogC4 encode of a scene-linear value (relative to 0.18 middle grey).
#[must_use]
pub fn logc4_encode(x: f64) -> f64 {
    if x >= C4.t {
        ((C4.a * x + 64.0).log2() - 6.0) / 14.0 * C4.b + C4.c
    } else {
        (x - C4.t) / C4.s
    }
}

/// LogC4 decode of a code value to scene-linear.
#[must_use]
pub fn logc4_decode(e: f64) -> f64 {
    if e >= 0.0 {
        ((14.0 * (e - C4.c) / C4.b + 6.0).exp2() - 64.0) / C4.a
    } else {
        e * C4.s + C4.t
    }
}

/// LogC3 (EI 800) encode of a scene-linear value.
#[must_use]
pub fn logc3_ei800_encode(x: f64) -> f64 {
    if x > C3.cut {
        C3.c * (C3.a * x + C3.b).log10() + C3.d
    } else {
        C3.e * x + C3.f
    }
}

/// LogC3 (EI 800) decode of a code value to scene-linear.
#[must_use]
pub fn logc3_ei800_decode(t: f64) -> f64 {
    if t > C3.e * C3.cut + C3.f {
        (10.0_f64.powf((t - C3.d) / C3.c) - C3.b) / C3.a
    } else {
        (t - C3.f) / C3.e
    }
}

/// Encodes each channel of a linear triple to LogC4.
#[must_use]
pub fn logc4_encode_rgb(rgb: Vec3) -> Vec3 {
    [
        logc4_encode(rgb[0]),
        logc4_encode(rgb[1]),
        logc4_encode(rgb[2]),
    ]
}

/// Decodes each channel of a LogC4 triple to scene-linear.
#[must_use]
pub fn logc4_decode_rgb(rgb: Vec3) -> Vec3 {
    [
        logc4_decode(rgb[0]),
        logc4_decode(rgb[1]),
        logc4_decode(rgb[2]),
    ]
}

/// Encodes each channel of a linear triple to LogC3 (EI 800).
#[must_use]
pub fn logc3_ei800_encode_rgb(rgb: Vec3) -> Vec3 {
    [
        logc3_ei800_encode(rgb[0]),
        logc3_ei800_encode(rgb[1]),
        logc3_ei800_encode(rgb[2]),
    ]
}

/// Decodes each channel of a LogC3 (EI 800) triple to scene-linear.
#[must_use]
pub fn logc3_ei800_decode_rgb(rgb: Vec3) -> Vec3 {
    [
        logc3_ei800_decode(rgb[0]),
        logc3_ei800_decode(rgb[1]),
        logc3_ei800_decode(rgb[2]),
    ]
}

/// A LogC4-encoded AWG4 RGB triple (the pipeline's working encoding from C8 to C12).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LogC4(pub Vec3);

/// A LogC3 (EI 800) encoded AWG3 RGB triple (the input encoding of LogC3 looks).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LogC3Ei800(pub Vec3);

impl From<Vec3> for LogC4 {
    fn from(v: Vec3) -> Self {
        Self(v)
    }
}

impl From<Vec3> for LogC3Ei800 {
    fn from(v: Vec3) -> Self {
        Self(v)
    }
}

impl LinearAwg4 {
    /// LogC4 encode of each channel (stage C8).
    #[must_use]
    pub fn to_logc4(self) -> LogC4 {
        LogC4(logc4_encode_rgb(self.0))
    }
}

impl LogC4 {
    /// LogC4 decode of each channel to linear AWG4.
    #[must_use]
    pub fn to_linear(self) -> LinearAwg4 {
        LinearAwg4(logc4_decode_rgb(self.0))
    }

    /// Exact conversion to LogC3 EI 800: LogC4 to linear AWG4, to AWG3, to LogC3 EI 800.
    #[must_use]
    pub fn to_logc3_ei800(self) -> LogC3Ei800 {
        self.to_linear().to_awg3().to_logc3_ei800()
    }

    /// LogC4 code value of the luminance of this triple: `encode(Y(decode(rgb)))` with the AWG4 `Y` row.
    #[must_use]
    pub fn luma(self) -> f64 {
        logc4_encode(self.to_linear().luminance())
    }
}

impl LinearAwg3 {
    /// LogC3 EI 800 encode of each channel.
    #[must_use]
    pub fn to_logc3_ei800(self) -> LogC3Ei800 {
        LogC3Ei800(logc3_ei800_encode_rgb(self.0))
    }
}

impl LogC3Ei800 {
    /// LogC3 EI 800 decode of each channel to linear AWG3.
    #[must_use]
    pub fn to_linear(self) -> LinearAwg3 {
        LinearAwg3(logc3_ei800_decode_rgb(self.0))
    }

    /// Exact conversion to LogC4: LogC3 to linear AWG3, to AWG4, to LogC4.
    #[must_use]
    pub fn to_logc4(self) -> LogC4 {
        self.to_linear().to_awg4().to_logc4()
    }
}

/// Luminance in LogC4 of a linear AWG4 triple, `encode(Y)`. This is the `L` of stage C9, taken
/// from the linear values before C8.
#[must_use]
pub fn logc4_luma_of_linear(rgb: LinearAwg4) -> f64 {
    logc4_encode(rgb.luminance())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_values() {
        assert!((logc4_encode(0.18) - 0.278_395_836_5).abs() < 1e-10);
        assert!((logc4_encode(0.0) - 0.092_864_125_1).abs() < 1e-10);
        assert!((logc4_decode(1.0) - 469.8).abs() < 0.05);
        assert!((logc3_ei800_encode(0.18) - 400.0 / 1023.0).abs() < 2e-8);
    }

    #[test]
    fn gray_constant_matches_encode() {
        assert!((logc4_encode(0.18) - crate::consts::LOGC4_GRAY).abs() < 1e-15);
    }

    #[test]
    fn logc4_is_continuous_at_the_toe() {
        let below = logc4_encode(C4.t - 1e-12);
        let above = logc4_encode(C4.t + 1e-12);
        assert!((below - above).abs() < 1e-9);
        assert!(logc4_encode(C4.t).abs() < 1e-15);
    }
}
