//! Display encode to Display P3, sRGB and Rec.2020, and 8-bit dithering (stage C13) (`docs/02-colour-pipeline.md` section C13).
//!
//! * **Transfer functions.** sRGB (IEC 61966-2-1) for the sRGB and Display P3 outputs; the BT.1886
//!   inverse (a pure `1/2.4` power) for Rec.2020. Both are extended to negative inputs by odd
//!   symmetry, `f(-x) = -f(x)`; display encoding clamps to `[0, 1]` before quantising, so
//!   the extension only matters to callers that look at unclamped values.
//! * **Quantisation** to 8, 10 or 16 bits: `round(clamp(v, 0, 1) * (2^bits - 1))`. A NaN is 0.
//! * **Dither** (8-bit outputs only): a fixed 64 x 64 blue-noise texture of ranks, built once with
//!   void-and-cluster ([`blue_noise::generate`], seed [`blue_noise::SEED`]) and embedded as a table
//!   so the output is identical on every platform. The threshold at pixel `(x, y)` is
//!   `(rank[y mod 64][x mod 64] + 0.5) / 4096`, the code value is `floor(v * 255 + threshold)`
//!   (clamped to 255). Over the 4096 texels the thresholds are exactly the 4096 evenly spaced values
//!   in `(0, 1)`, so a constant input averages to its exact unquantised value. All three channels
//!   of a pixel use the same threshold (achromatic noise). `x` and `y` are absolute pixel
//!   coordinates in the output image, so tiled and whole-image rendering agree.

pub mod blue_noise;
mod blue_noise_data;

use crate::consts::Vec3;
use crate::gamut::LinearRec709;

/// The output colour spaces of stage C13.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputSpace {
    /// sRGB: Rec.709 primaries, sRGB transfer.
    Srgb,
    /// Display P3: P3-D65 primaries, sRGB transfer (the viewport and the P3 export).
    DisplayP3,
    /// Rec.2020: Rec.2020 primaries, BT.1886 (gamma 2.4) transfer.
    Rec2020,
}

impl OutputSpace {
    /// Encodes one linear channel with this space's transfer function (odd-symmetric, unclamped).
    #[must_use]
    pub fn encode_channel(self, linear: f64) -> f64 {
        match self {
            Self::Srgb | Self::DisplayP3 => srgb_encode(linear),
            Self::Rec2020 => bt1886_inverse(linear),
        }
    }

    /// Decodes one channel with this space's transfer function (odd-symmetric, unclamped).
    #[must_use]
    pub fn decode_channel(self, encoded: f64) -> f64 {
        match self {
            Self::Srgb | Self::DisplayP3 => srgb_decode(encoded),
            Self::Rec2020 => bt1886_eotf(encoded),
        }
    }

    /// Converts display-linear Rec.709 (the output of a LogC3-to-Rec.709 look, see
    /// [`crate::look::LookTransform::eval_display_rec709`]) to this space's linear primaries, so it can
    /// be blended with the display transform's output.
    #[must_use]
    pub fn from_linear_rec709(self, rgb: LinearRec709) -> Vec3 {
        match self {
            Self::Srgb => rgb.0,
            Self::DisplayP3 => rgb.to_display_p3().0,
            Self::Rec2020 => rgb.to_rec2020().0,
        }
    }

    /// Encodes a linear triple (already in this space's primaries) and clamps to `[0, 1]`.
    #[must_use]
    pub fn encode_clamped(self, linear: Vec3) -> Vec3 {
        linear.map(|c| unit(self.encode_channel(c)))
    }
}

fn unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

/// sRGB encode (linear to code value), odd-symmetric.
#[must_use]
pub fn srgb_encode(x: f64) -> f64 {
    let a = x.abs();
    let e = if a <= 0.003_130_8 {
        12.92 * a
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    e.copysign(x)
}

/// sRGB decode (code value to linear), odd-symmetric.
#[must_use]
pub fn srgb_decode(e: f64) -> f64 {
    let a = e.abs();
    let x = if a <= 0.040_45 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    x.copysign(e)
}

/// BT.1886 inverse EOTF with a black level of zero: `x^(1/2.4)`, odd-symmetric.
#[must_use]
pub fn bt1886_inverse(x: f64) -> f64 {
    x.abs().powf(1.0 / 2.4).copysign(x)
}

/// BT.1886 EOTF with a black level of zero: `e^2.4`, odd-symmetric.
#[must_use]
pub fn bt1886_eotf(e: f64) -> f64 {
    e.abs().powf(2.4).copysign(e)
}

/// Largest code value of a `bits`-bit output (`2^bits - 1`); `bits` is clamped to 1 to 16.
#[must_use]
pub fn max_code(bits: u32) -> u32 {
    (1_u32 << bits.clamp(1, 16)) - 1
}

/// Quantises `v` (nominally `[0, 1]`) to a `bits`-bit code value (1 to 16 bits), rounding to nearest.
#[must_use]
pub fn quantize(v: f64, bits: u32) -> u16 {
    let max = f64::from(max_code(bits));
    (unit(v) * max).round() as u16
}

/// The dither threshold in `(0, 1)` at absolute pixel `(x, y)`.
#[must_use]
pub fn dither_threshold(x: usize, y: usize) -> f64 {
    let rank = blue_noise_data::BLUE_NOISE_64
        [(y % blue_noise::SIDE) * blue_noise::SIDE + (x % blue_noise::SIDE)];
    (f64::from(rank) + 0.5) / blue_noise::TEXELS as f64
}

/// Quantises `v` to 8 bits with blue-noise dither at absolute pixel `(x, y)`.
#[must_use]
pub fn dither_8bit(v: f64, x: usize, y: usize) -> u8 {
    let scaled = unit(v) * 255.0;
    (scaled + dither_threshold(x, y)).floor().min(255.0) as u8
}

/// C13 for an 8-bit output: encode with the space's transfer, then dither each channel.
#[must_use]
pub fn encode_8bit_dithered(linear: Vec3, space: OutputSpace, x: usize, y: usize) -> [u8; 3] {
    let e = space.encode_clamped(linear);
    [
        dither_8bit(e[0], x, y),
        dither_8bit(e[1], x, y),
        dither_8bit(e[2], x, y),
    ]
}

/// C13 for a 10- or 16-bit output (no dither): encode, then quantise to `bits` bits.
#[must_use]
pub fn encode_quantized(linear: Vec3, space: OutputSpace, bits: u32) -> [u16; 3] {
    let e = space.encode_clamped(linear);
    [
        quantize(e[0], bits),
        quantize(e[1], bits),
        quantize(e[2], bits),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_round_trips() {
        for space in [
            OutputSpace::Srgb,
            OutputSpace::DisplayP3,
            OutputSpace::Rec2020,
        ] {
            for k in -50..=1100 {
                let x = f64::from(k) / 1000.0;
                let y = space.decode_channel(space.encode_channel(x));
                assert!((y - x).abs() < 1e-12, "{space:?} {x}: {y}");
            }
        }
    }

    #[test]
    fn quantize_basics() {
        assert_eq!(quantize(0.0, 8), 0);
        assert_eq!(quantize(1.0, 8), 255);
        assert_eq!(quantize(1.0, 10), 1023);
        assert_eq!(quantize(1.0, 16), 65535);
        assert_eq!(quantize(f64::NAN, 8), 0);
        assert_eq!(quantize(-3.0, 8), 0);
        assert_eq!(quantize(7.0, 8), 255);
        assert_eq!(quantize(0.5, 8), 128);
    }

    #[test]
    fn dither_is_in_range_and_exact_at_integers() {
        for y in 0..64 {
            for x in 0..64 {
                let t = dither_threshold(x, y);
                assert!(t > 0.0 && t < 1.0);
                for code in [0_u8, 1, 127, 254, 255] {
                    assert_eq!(dither_8bit(f64::from(code) / 255.0, x, y), code);
                }
            }
        }
    }
}
