//! False-colour suppression after R6's single-frame denoise (`docs/algorithms/05-single-frame-denoise.md`,
//! *Integration*).
//!
//! The demosaic leaves false colour where the luma has fine stripes near the sensor's limit (blinds,
//! shutters, fabric, the golden chart's 2 px grating). The R7 colour guided filter that ran after the
//! bilateral denoise removed it everywhere, at the cost of smearing colour across every edge the luma does
//! not share. Method A keeps it: block matching groups the repeating pattern and the hard threshold keeps
//! what the group agrees on. This stage runs R7's colour guided filter ([`detail::chroma_guided_filter`], at
//! the photo's Colour and noise) only where the luma alternates, and leaves the rest as the method made it:
//!
//! 1. `Y`: the luma of the working encoding (`denoise_space`, BT.709 weights of `ycbcr`), so one threshold
//!    holds in the shadows and the lights.
//! 2. Alternation: `s = max(0, −(Y(x) − Y(x − 1)) (Y(x + 1) − Y(x)))`, plus the same along `y`. A ramp, a
//!    step and a blurred edge give 0; stripes of amplitude `A` and a period of 2 px give `(2A)²` per
//!    direction.
//! 3. `E = sqrt(box(s))` over 5 × 5, then `M = max(E)` over 5 × 5, so the ends of a striped area are
//!    covered too.
//! 4. `w = smoothstep((M − T0) / (T1 − T0))` and `out = x + w (G − x)` per channel, with `G` the guided
//!    filter's output (which changes only the chroma).
//!
//! Buffers are clamped at their edges; a tile asks for [`halo_px`] of context, so the clamp touches only
//! pixels outside its core or at the frame's own edge, and a tile equals the whole frame bit for bit.
//!
//! [`T0`] and [`T1`] were measured with the harness of `examples/denoise_gate.rs` (`docs/algorithms/05`,
//! *Measurement*): wider gates remove more of the false colour and cost more of the PSNR.

use super::denoise_space::{WORKING_GAIN_EV, encode};
use super::detail::{self, NoiseReduction, guided_halo_px};
use super::ycbcr::{KB, KG, KR};
use super::{NoiseSigma, RgbImage};

/// The stripe energy at which the guided filter starts to take over (working-encoding units).
pub const T0: f64 = 0.05;
/// The stripe energy at which the guided filter's output is used in full.
pub const T1: f64 = 0.15;
/// The radius of the box mean of the alternation.
pub const BOX_RADIUS_PX: usize = 2;
/// The radius of the dilation (maximum) of the stripe energy.
pub const DILATE_RADIUS_PX: usize = 2;
/// The context the gate reads on each side: the alternation's 1 px, the box and the dilation.
pub const GATE_HALO_PX: usize = 1 + BOX_RADIUS_PX + DILATE_RADIUS_PX;

/// The context the stage needs on each side at Colour `color`: the guided filter's and the gate's.
#[must_use]
pub fn halo_px(color: f32) -> usize {
    let r = NoiseReduction {
        luminance: 0.0,
        color,
        detail: 0.0,
    }
    .chroma_radius_px();
    guided_halo_px(r).max(GATE_HALO_PX)
}

/// The weight of the guided filter at every pixel of `img` (balanced linear RGB), in `[0, 1]`.
#[must_use]
pub fn weights(img: &RgbImage) -> Vec<f64> {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    if n == 0 {
        return Vec::new();
    }
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let e = |c: usize| encode(f64::from(img.data[c * n + i]), WORKING_GAIN_EV);
            KR * e(0) + KG * e(1) + KB * e(2)
        })
        .collect();
    let at = |x: isize, row: isize| -> f64 {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = row.clamp(0, h as isize - 1) as usize;
        y[yi * w + xi]
    };
    let mut s = vec![0.0; n];
    for row in 0..h as isize {
        for x in 0..w as isize {
            let c = at(x, row);
            let sx = (-(c - at(x - 1, row)) * (at(x + 1, row) - c)).max(0.0);
            let sy = (-(c - at(x, row - 1)) * (at(x, row + 1) - c)).max(0.0);
            s[row as usize * w + x as usize] = sx + sy;
        }
    }
    let area = ((2 * BOX_RADIUS_PX + 1) * (2 * BOX_RADIUS_PX + 1)) as f64;
    let boxed = separable(&s, w, h, BOX_RADIUS_PX, 0.0, |acc, v| acc + v);
    let energy: Vec<f64> = boxed.iter().map(|&v| (v / area).sqrt()).collect();
    let dilated = separable(&energy, w, h, DILATE_RADIUS_PX, 0.0, f64::max);
    dilated
        .iter()
        .map(|&m| {
            let t = ((m - T0) / (T1 - T0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

/// A separable window reduction of radius `r` with clamped edges: rows, then columns, each fold over the
/// window in order from its first sample.
fn separable(
    src: &[f64],
    w: usize,
    h: usize,
    r: usize,
    init: f64,
    fold: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let r = r as isize;
    let mut rows = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w as isize {
            let mut acc = init;
            for d in -r..=r {
                acc = fold(acc, src[y * w + (x + d).clamp(0, w as isize - 1) as usize]);
            }
            rows[y * w + x as usize] = acc;
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h as isize {
        for x in 0..w {
            let mut acc = init;
            for d in -r..=r {
                acc = fold(acc, rows[(y + d).clamp(0, h as isize - 1) as usize * w + x]);
            }
            out[y as usize * w + x] = acc;
        }
    }
    out
}

/// The stage: R7's colour guided filter at `color`, used where [`weights`] says the luma alternates.
/// `color <= 0` (no colour noise reduction) returns `img` as it is.
#[must_use]
pub fn suppress(img: &RgbImage, sigma_est: &NoiseSigma, color: f32) -> RgbImage {
    if color <= 0.0 {
        return img.clone();
    }
    let guided = detail::chroma_guided_filter(img, sigma_est, color);
    let wts = weights(img);
    let n = img.plane_len();
    let mut out = img.clone();
    for (i, &wt) in wts.iter().enumerate() {
        if wt == 0.0 {
            continue;
        }
        for c in 0..3 {
            let k = c * n + i;
            let x = f64::from(img.data[k]);
            out.data[k] = (x + wt * (f64::from(guided.data[k]) - x)) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> RgbImage {
        let mut data = Vec::with_capacity(3 * w * h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data.push(f(c, x, y));
                }
            }
        }
        RgbImage::from_planar(w, h, data).unwrap()
    }

    #[test]
    fn ramps_steps_and_colour_edges_get_no_weight() {
        // A ramp, a sharp step and an edge between two colours: no alternation anywhere.
        let ramp = image(32, 24, |_, x, _| 0.01 + 0.004 * x as f32);
        let step = image(32, 24, |_, x, _| if x < 16 { 0.02 } else { 0.3 });
        let colour = image(32, 24, |c, x, _| match (c, x < 16) {
            (0, true) => 0.2,
            (2, false) => 0.2,
            _ => 0.05,
        });
        for img in [&ramp, &step, &colour] {
            assert!(weights(img).iter().all(|&w| w == 0.0));
            assert_eq!(&suppress(img, &NoiseSigma::Scalar(0.002), 50.0), img);
        }
    }

    #[test]
    fn a_2px_grating_with_false_colour_loses_most_of_it() {
        // Neutral stripes of a 2 px period with a blue and orange false colour that alternates in bands of
        // 3 px, as the demosaic leaves it.
        let img = image(48, 40, |c, x, y| {
            let luma: f32 = if x % 2 == 0 { 0.03 } else { 0.25 };
            let tint = if (y / 3) % 2 == 0 { 1.0 } else { -1.0 };
            let k = match c {
                0 => 0.04 * tint,
                2 => -0.04 * tint,
                _ => 0.0,
            };
            (luma + k).max(0.0)
        });
        let wts = weights(&img);
        let inner = |v: &[f64]| -> Vec<f64> {
            (8..32)
                .flat_map(|y| (8..40).map(move |x| y * 48 + x))
                .map(|i| v[i])
                .collect()
        };
        assert!(
            inner(&wts).iter().all(|&w| w == 1.0),
            "full weight inside the grating"
        );
        let out = suppress(&img, &NoiseSigma::Scalar(0.002), 50.0);
        let spread = |im: &RgbImage| {
            let n = im.plane_len();
            let rb: Vec<f64> = (0..n)
                .map(|i| f64::from(im.data[i] - im.data[2 * n + i]))
                .collect();
            let v = inner(&rb);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        assert!(
            spread(&out) < 0.5 * spread(&img),
            "{} {}",
            spread(&out),
            spread(&img)
        );
    }

    #[test]
    fn colour_zero_is_the_identity_and_the_halo_covers_both_parts() {
        let img = image(20, 20, |c, x, y| ((x * 7 + y * 3 + c) % 5) as f32 * 0.05);
        assert_eq!(suppress(&img, &NoiseSigma::Scalar(0.01), 0.0), img);
        assert_eq!(halo_px(50.0), 10);
        assert_eq!(halo_px(1.0), GATE_HALO_PX);
        assert_eq!(GATE_HALO_PX, 5);
    }

    #[test]
    fn weights_are_bounded_and_finite() {
        let img = image(17, 13, |c, x, y| {
            ((x * 31 + y * 17 + c * 7) % 11) as f32 * 0.1 - 0.05
        });
        let wts = weights(&img);
        assert_eq!(wts.len(), 17 * 13);
        assert!(wts.iter().all(|w| (0.0..=1.0).contains(w)));
        assert!(weights(&image(0, 0, |_, _, _| 0.0)).is_empty());
    }
}
