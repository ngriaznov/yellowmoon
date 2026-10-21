//! The working encoding of the single-frame denoisers (`docs/algorithms/05-single-frame-denoise.md`,
//! *Integration*).
//!
//! The methods of that specification work on an image "in the non-linear (display- or log-encoded) state the
//! user sees", black 0 and white 1. The raw chain holds linear camera RGB balanced with the initial gains, white
//! 1.0 at the sensor's clip before the gains. The working encoding puts it in that state without depending on
//! anything the user changes after R6 (exposure, look, curves are colour stages, so a slider there must not
//! re-run the raw chain):
//!
//! `W(x) = srgb(g x)`, `g = 2^CAMERA_GAIN_EV` (the camera's own exposure, which is Auto exposure), and `srgb` the
//! sRGB transfer function extended to the whole line: `12.92 v` below `0.0031308` (the slope made exact at the knee) (negative values included, so
//! noise below black keeps its sign) and `1.055 v^(1/2.4) − 0.055` above (also beyond 1, where highlights and
//! the white balance gains take a sample). Middle grey of a photo at the camera's exposure lands near 0.46, as in
//! an sRGB picture; the power 1/2.4 is close to the square root that evens out shot noise, so one threshold suits
//! the shadows and the lights alike. `decode` is the exact inverse.

use super::RgbImage;
use ym_color::exposure::CAMERA_GAIN_EV;

/// The gain of the encoding in stops: the camera's own exposure.
pub const WORKING_GAIN_EV: f64 = CAMERA_GAIN_EV;

/// The knee of the sRGB curve on the linear side.
const KNEE_X: f64 = 0.003_130_8;
/// The slope of the linear segment: sRGB's 12.92, made to meet the power segment at the knee exactly
/// (`(1.055 KNEE_X^(1/2.4) − 0.055) / KNEE_X`), so that the curve and its inverse are continuous and strictly
/// increasing (the rounded 12.92 leaves a step of 3e-8 there).
const SLOPE: f64 = 12.919_990_891_366_469;
/// The knee on the encoded side.
const KNEE_Y: f64 = SLOPE * KNEE_X;

/// The knee on the linear side, the slope of the linear segment and the knee on the encoded side, for the
/// device kernel (`kernels::bm_dct`), which writes them into its shader.
pub(crate) const fn knee_and_slope() -> (f64, f64, f64) {
    (KNEE_X, SLOPE, KNEE_Y)
}

/// The sRGB transfer function on the whole line (see the module documentation).
#[must_use]
pub fn srgb_encode(v: f64) -> f64 {
    if v < KNEE_X {
        SLOPE * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// The inverse of [`srgb_encode`].
#[must_use]
pub fn srgb_decode(y: f64) -> f64 {
    if y < KNEE_Y {
        y / SLOPE
    } else {
        ((y + 0.055) / 1.055).powf(2.4)
    }
}

/// The slope of [`srgb_encode`] at `v` (`v` at or below 0 takes the linear segment's).
#[must_use]
pub fn srgb_slope(v: f64) -> f64 {
    if v < KNEE_X {
        SLOPE
    } else {
        1.055 / 2.4 * v.powf(1.0 / 2.4 - 1.0)
    }
}

/// The working encoding of a balanced linear sample: `srgb(2^gain_ev x)`.
#[must_use]
pub fn encode(x: f64, gain_ev: f64) -> f64 {
    srgb_encode(x * gain_ev.exp2())
}

/// The inverse of [`encode`].
#[must_use]
pub fn decode(y: f64, gain_ev: f64) -> f64 {
    srgb_decode(y) / gain_ev.exp2()
}

/// Every sample of `img` through [`encode`].
#[must_use]
pub fn encode_image(img: &RgbImage, gain_ev: f64) -> RgbImage {
    map(img, |x| encode(x, gain_ev))
}

/// Every sample of `img` through [`decode`].
#[must_use]
pub fn decode_image(img: &RgbImage, gain_ev: f64) -> RgbImage {
    map(img, |y| decode(y, gain_ev))
}

fn map(img: &RgbImage, f: impl Fn(f64) -> f64) -> RgbImage {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = f(f64::from(*v)) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encoding_is_srgb_of_the_camera_exposure_and_round_trips() {
        // Grey at the camera's exposure (0.18 / 2^1.11 on the raw scale) lands at sRGB's 18 % grey.
        let grey_raw = 0.18 / WORKING_GAIN_EV.exp2();
        assert!((encode(grey_raw, WORKING_GAIN_EV) - srgb_encode(0.18)).abs() < 1e-12);
        assert!((srgb_encode(0.18) - 0.461_356).abs() < 1e-5);
        for x in [
            -0.05, -1e-4, 0.0, 1e-4, 0.002, 0.01, 0.1, 0.5, 1.0, 2.5, 8.0,
        ] {
            let y = encode(x, WORKING_GAIN_EV);
            assert!(
                (decode(y, WORKING_GAIN_EV) - x).abs() <= 1e-12 * (1.0 + x.abs()),
                "{x}"
            );
        }
        // Continuous at the knee, and monotone through it and beyond white.
        let below = SLOPE * KNEE_X;
        let above = 1.055 * KNEE_X.powf(1.0 / 2.4) - 0.055;
        assert!((below - above).abs() < 1e-15);
        assert!(srgb_decode(KNEE_Y - 1e-12) < srgb_decode(KNEE_Y));
        let mut last = f64::NEG_INFINITY;
        for i in -100..2000 {
            let y = srgb_encode(f64::from(i) * 0.001);
            assert!(y > last);
            last = y;
        }
    }

    #[test]
    fn negative_samples_keep_their_sign() {
        assert!(encode(-0.01, WORKING_GAIN_EV) < 0.0);
        assert_eq!(encode(0.0, WORKING_GAIN_EV), 0.0);
    }
}
