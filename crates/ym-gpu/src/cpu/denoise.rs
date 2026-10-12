//! R6 classical denoise used when the neural path is off or unavailable (`docs/05-raw-processing.md` section R6).
//!
//! A light bilateral filter in an opponent space `(Y, R - Y, B - Y)` (luma weights [`LUMA`]),
//! applied after the classical demosaic so previews are not noisier than the export. The range
//! kernel uses the Euclidean distance in the opponent space, scaled by the noise σ of the centre
//! pixel (scalar or σ map), so flat noisy areas are averaged and edges, whose differences exceed
//! the noise, are kept:
//!
//! `w(q) = exp(-|dx|^2 / 2 s^2) * exp(-(dY^2 + dCr^2 + dCb^2) / 2 h^2)`, `h = strength * sqrt(6) * sigma`
//!
//! `sqrt(6) sigma` is the RMS opponent-space distance between two pixels of pure noise (three
//! channels, each difference with variance `2 sigma^2`), so `strength = 1` weights a typical noise
//! pair at about 0.6 and larger values smooth more. Defaults: window radius 2 (5x5), spatial
//! `s = 1.5` px, `strength = 1`.
//!
//! Receptive field: `radius_px` (2). Borders mirror. The filter is a gather, so tiles with that
//! halo are exact.

use super::filters::{self, BAND_ROWS};
use super::{LUMA, NoiseSigma, RgbImage};

/// Parameters of the bilateral filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilateralParams {
    /// Half-width of the window in pixels (the halo).
    pub radius_px: usize,
    /// Standard deviation of the spatial Gaussian in pixels.
    pub spatial_sigma_px: f32,
    /// Multiplier of the range bandwidth (see the module documentation).
    pub strength: f32,
}

impl Default for BilateralParams {
    fn default() -> Self {
        BilateralParams {
            radius_px: 2,
            spatial_sigma_px: 1.5,
            strength: 1.0,
        }
    }
}

/// The opponent-space planes `(Y, R - Y, B - Y)` of an image.
fn to_opponent(img: &RgbImage) -> [Vec<f32>; 3] {
    let y = img.luma_plane();
    let [r, _, b] = img.planes();
    let cr: Vec<f32> = r.iter().zip(&y).map(|(r, y)| r - y).collect();
    let cb: Vec<f32> = b.iter().zip(&y).map(|(b, y)| b - y).collect();
    [y, cr, cb]
}

/// Bilateral denoise of a whole buffer (no banding). `sigma_row0` is the row of the σ map that
/// the buffer's first row corresponds to (`0` for a whole-image σ map or a scalar).
pub fn bilateral_whole(
    img: &RgbImage,
    sigma: &NoiseSigma,
    sigma_row0: usize,
    params: &BilateralParams,
) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut out = img.clone();
    if w == 0 || h == 0 {
        return out;
    }
    let [y, cr, cb] = to_opponent(img);
    let r = params.radius_px;
    let side = 2 * r + 1;
    let mut spatial = vec![0.0f32; side * side];
    for dy in 0..side {
        for dx in 0..side {
            let d2 = ((dx as isize - r as isize).pow(2) + (dy as isize - r as isize).pow(2)) as f32;
            spatial[dy * side + dx] =
                (-0.5 * d2 / (params.spatial_sigma_px * params.spatial_sigma_px)).exp();
        }
    }
    // Mirror-padded planes so the window needs no bounds checks.
    let (yp, crp, cbp) = (
        filters::pad_mirror(&y, w, h, r),
        filters::pad_mirror(&cr, w, h, r),
        filters::pad_mirror(&cb, w, h, r),
    );
    let pw = w + 2 * r;
    let n = w * h;
    let root6 = 6f32.sqrt();
    for py in 0..h {
        for px in 0..w {
            let i = py * w + px;
            let s = sigma.at((sigma_row0 + py) * w + px);
            let hh = params.strength * root6 * s;
            if hh <= 1e-12 {
                continue; // untouched pixels keep their exact input values
            }
            let inv2h2 = 0.5 / (hh * hh);
            let (cy, ccr, ccb) = (y[i], cr[i], cb[i]);
            let (mut sy, mut scr, mut scb, mut sw) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for dy in 0..side {
                let base = (py + dy) * pw + px;
                let (ry, rcr, rcb) = (
                    &yp[base..base + side],
                    &crp[base..base + side],
                    &cbp[base..base + side],
                );
                let srow = &spatial[dy * side..(dy + 1) * side];
                for k in 0..side {
                    let (dyv, dcr, dcb) = (ry[k] - cy, rcr[k] - ccr, rcb[k] - ccb);
                    let u = (dyv * dyv + dcr * dcr + dcb * dcb) * inv2h2;
                    if u > 9.0 {
                        continue; // weight below 1.3e-4
                    }
                    let wt = srow[k] * (-u).exp();
                    sy += wt * ry[k];
                    scr += wt * rcr[k];
                    scb += wt * rcb[k];
                    sw += wt;
                }
            }
            // The centre pixel has weight exp(0) * 1 > 0, so sw > 0.
            let inv = 1.0 / sw;
            let (fy, fcr, fcb) = (sy * inv, scr * inv, scb * inv);
            let red = fcr + fy;
            let blue = fcb + fy;
            out.data[i] = red;
            out.data[n + i] = (fy - LUMA[0] * red - LUMA[2] * blue) / LUMA[1];
            out.data[2 * n + i] = blue;
        }
    }
    out
}

/// Bilateral denoise in parallel bands with a halo of `params.radius_px`. `sigma` is indexed by
/// the position in `img` (a whole-image map when `img` is a whole frame).
pub fn bilateral_opponent(
    img: &RgbImage,
    sigma: &NoiseSigma,
    params: &BilateralParams,
) -> RgbImage {
    let base_y0 = img.place.y0;
    filters::run_rgb_bands(img, params.radius_px, BAND_ROWS, |band| {
        bilateral_whole(band, sigma, band.place.y0 - base_y0, params)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth;

    fn noisy_flat(w: usize, h: usize, level: [f32; 3], sigma: f32, seed: u64) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for (c, v) in level.iter().enumerate() {
            img.plane_mut(c).fill(*v);
        }
        synth::add_noise(&mut img.data, sigma, seed);
        img
    }

    fn std_dev(p: &[f32]) -> f64 {
        let m = p.iter().map(|&v| f64::from(v)).sum::<f64>() / p.len() as f64;
        (p.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>() / p.len() as f64).sqrt()
    }

    #[test]
    fn reduces_noise_in_flat_areas() {
        let img = noisy_flat(64, 64, [0.4, 0.5, 0.3], 0.02, 3);
        let out = bilateral_opponent(&img, &NoiseSigma::Scalar(0.02), &BilateralParams::default());
        for c in 0..3 {
            let before = std_dev(img.plane(c));
            let after = std_dev(out.plane(c));
            assert!(after < 0.75 * before, "channel {c}: {before} -> {after}");
        }
    }

    #[test]
    fn keeps_a_strong_edge() {
        let mut img = RgbImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                img.set_pixel(
                    x,
                    y,
                    if x < 32 {
                        [0.2, 0.2, 0.2]
                    } else {
                        [0.8, 0.7, 0.6]
                    },
                );
            }
        }
        synth::add_noise(&mut img.data, 0.01, 5);
        let out = bilateral_opponent(&img, &NoiseSigma::Scalar(0.01), &BilateralParams::default());
        // The step survives: pixels right next to the edge stay on their side.
        assert!(out.pixel(31, 10)[0] < 0.3 && out.pixel(32, 10)[0] > 0.7);
    }

    #[test]
    fn zero_sigma_is_identity() {
        let img = synth::natural(30, 20, 2);
        assert_eq!(
            bilateral_opponent(&img, &NoiseSigma::Scalar(0.0), &BilateralParams::default()),
            img
        );
    }

    #[test]
    fn banded_equals_whole_with_a_sigma_plane() {
        let img = synth::natural(50, 300, 7);
        let sig: Vec<f32> = (0..50 * 300)
            .map(|i| 0.005 + 0.00002 * (i % 97) as f32)
            .collect();
        let p = BilateralParams::default();
        let whole = bilateral_whole(&img, &NoiseSigma::Plane(&sig), 0, &p);
        let banded = bilateral_opponent(&img, &NoiseSigma::Plane(&sig), &p);
        assert!(whole.max_abs_diff(&banded) < 1e-6);
    }
}
