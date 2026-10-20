//! Appendix D of `docs/algorithms/05-single-frame-denoise.md`: the global noise level of an encoded image, the two
//! numbers that would condition a learned denoiser. No trained network exists (reading D-b), so this is the
//! estimate alone.
//!
//! 1. `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = −0.117188 R − 0.394228 G + 0.511415 B`,
//!    `Cr = 0.511416 R − 0.464522 G − 0.046894 B` (appendix D's own coefficients, not [`super::ycbcr`]'s).
//! 2. The rectangle: the best (lowest-score) region of the detector run as for method C, on the coarsest level of
//!    the RGB pyramid of base 256 ([`flat_regions::detect_frame`]), mapped to full resolution by rounding
//!    ([`flat_regions::Region::to_pixels`]) (reading D-a); or a rectangle given by the caller, clipped to the image.
//! 3. `hp = x − G ∗ x`, `G` the 21 × 21 Gaussian of σ 18.9, separable, its taps normalised to sum 1, mirrored at
//!    the image edges without repeating the edge sample ([`super::mirror`]) (reading D-b). Only the rectangle's
//!    pixels are computed; the blur reads 10 px around it.
//! 4. Inside the rectangle, `σY` is the population standard deviation (dividing by `N`) of `hp_Y`, and `σC`
//!    that of the `2N` samples of `hp_Cb` and `hp_Cr` taken together (one mean).
//! 5. `level = min(5 σ, 1)^0.7` for each.
//!
//! Arithmetic is `f64` and sequential; `exp` and `pow` come from `libm`, so the bits do not depend on the
//! platform's maths library.
//!
//! The detector picks the flattest cell, so on pure noise its rectangle reads low by selection when the detector
//! runs at full resolution (a frame under 512 px on the short side: σY 8 % low on 512 × 384 of pure noise); with
//! one pyramid step or more the selection happens at the coarse level and the estimate is within 3 % (five seeds
//! on 1024 × 768).

use super::flat_regions;
use super::{Rect, RgbImage, mirror};

/// Appendix D's luma weights.
const Y_W: [f64; 3] = [0.2126, 0.7152, 0.0722];
/// Appendix D's Cb weights.
const CB_W: [f64; 3] = [-0.117188, -0.394228, 0.511415];
/// Appendix D's Cr weights.
const CR_W: [f64; 3] = [0.511416, -0.464522, -0.046894];
/// Half-width of the 21 × 21 blur.
const BLUR_RADIUS: usize = 10;
/// σ of the blur in pixels.
const BLUR_SIGMA: f64 = 18.9;
/// The detector runs as for method C.
const DETECTOR_BASE: usize = 256;

/// The noise level of an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseLevel {
    /// Standard deviation of the high-passed luma inside the rectangle.
    pub sigma_y: f64,
    /// Standard deviation of the high-passed Cb and Cr together inside the rectangle.
    pub sigma_c: f64,
    /// `min(5 σY, 1)^0.7`.
    pub level_y: f64,
    /// `min(5 σC, 1)^0.7`.
    pub level_c: f64,
    /// The rectangle measured, in pixels of the image.
    pub rect: Rect,
}

/// The conditioning level of a standard deviation: `min(5 σ, 1)^0.7`.
pub fn level(sigma: f64) -> f64 {
    libm::pow((5.0 * sigma).min(1.0), 0.7)
}

/// The noise level inside the detector's best flat region; `None` when the detector finds no region (an image
/// smaller than one detector cell at its coarsest level).
pub fn estimate(img: &RgbImage) -> Option<NoiseLevel> {
    let (_, regions) = flat_regions::detect_frame(img, DETECTOR_BASE);
    let rect = regions.first()?.to_pixels(img.width, img.height);
    (rect.w > 0 && rect.h > 0).then(|| estimate_in(img, rect))
}

/// The normalised 1D taps of the blur.
fn blur_taps() -> [f64; 2 * BLUR_RADIUS + 1] {
    let mut k: [f64; 2 * BLUR_RADIUS + 1] = std::array::from_fn(|i| {
        let d = i as f64 - BLUR_RADIUS as f64;
        libm::exp(-0.5 * d * d / (BLUR_SIGMA * BLUR_SIGMA))
    });
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Population standard deviation of the samples.
fn std_dev<'a>(samples: impl Iterator<Item = &'a f64> + Clone) -> f64 {
    let (mut n, mut s) = (0usize, 0.0f64);
    for v in samples.clone() {
        s += v;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let mean = s / n as f64;
    let var = samples.map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// The noise level inside `rect` (clipped to the image; an empty rectangle gives zeros).
pub fn estimate_in(img: &RgbImage, rect: Rect) -> NoiseLevel {
    let (w, h) = (img.width, img.height);
    let (x0, x1) = (rect.x.min(w), rect.x1().min(w));
    let (y0, y1) = (rect.y.min(h), rect.y1().min(h));
    let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
    if rect.w == 0 || rect.h == 0 {
        return NoiseLevel {
            sigma_y: 0.0,
            sigma_c: 0.0,
            level_y: level(0.0),
            level_c: level(0.0),
            rect,
        };
    }
    let r = BLUR_RADIUS;
    let taps = blur_taps();
    let (pw, ph) = (rect.w + 2 * r, rect.h + 2 * r);
    // The rectangle with r px of mirrored context, as Y, Cb, Cr.
    let mut win: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(pw * ph));
    let [pr, pg, pb] = img.planes();
    for py in 0..ph {
        let sy = mirror(y0 as isize + py as isize - r as isize, h);
        for px in 0..pw {
            let sx = mirror(x0 as isize + px as isize - r as isize, w);
            let i = sy * w + sx;
            let rgb = [f64::from(pr[i]), f64::from(pg[i]), f64::from(pb[i])];
            for (plane, wts) in win.iter_mut().zip([Y_W, CB_W, CR_W]) {
                plane.push(wts[0] * rgb[0] + wts[1] * rgb[1] + wts[2] * rgb[2]);
            }
        }
    }
    // High-pass inside the rectangle: horizontal blur on every padded row, then vertical.
    let hp: [Vec<f64>; 3] = std::array::from_fn(|c| {
        let src = &win[c];
        let mut tmp = Vec::with_capacity(ph * rect.w);
        for row in src.chunks_exact(pw) {
            tmp.extend((0..rect.w).map(|x| {
                taps.iter()
                    .zip(&row[x..x + 2 * r + 1])
                    .map(|(k, v)| k * v)
                    .sum::<f64>()
            }));
        }
        let mut out = Vec::with_capacity(rect.w * rect.h);
        for y in 0..rect.h {
            for x in 0..rect.w {
                let blur: f64 = taps
                    .iter()
                    .enumerate()
                    .map(|(t, k)| k * tmp[(y + t) * rect.w + x])
                    .sum();
                out.push(src[(y + r) * pw + x + r] - blur);
            }
        }
        out
    });
    let sigma_y = std_dev(hp[0].iter());
    let sigma_c = std_dev(hp[1].iter().chain(&hp[2]));
    NoiseLevel {
        sigma_y,
        sigma_c,
        level_y: level(sigma_y),
        level_c: level(sigma_c),
        rect,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth::Rng;

    /// Flat grey 0.5 with independent Gaussian noise of `sigmas` per channel.
    fn noisy_grey(w: usize, h: usize, sigmas: [f64; 3], seed: u64) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        let mut rng = Rng::new(seed);
        for (c, s) in sigmas.iter().enumerate() {
            for v in img.plane_mut(c) {
                *v = (0.5 + s * f64::from(rng.gaussian())) as f32;
            }
        }
        img
    }

    /// The analytic σ of the high-passed plane with weights `wts` of independent channel noise `sigmas`:
    /// `var(x − G ∗ x) = var(x) · (1 − 2 G₀ + Σ G²)` for white noise.
    fn analytic(wts: [f64; 3], sigmas: [f64; 3]) -> f64 {
        let taps = blur_taps();
        let g0 = taps[BLUR_RADIUS] * taps[BLUR_RADIUS];
        let sum_sq = taps.iter().map(|v| v * v).sum::<f64>().powi(2);
        let var: f64 = (0..3).map(|c| (wts[c] * sigmas[c]).powi(2)).sum();
        (var * (1.0 - 2.0 * g0 + sum_sq)).sqrt()
    }

    #[test]
    fn sigma_y_and_sigma_c_match_their_analytic_values_on_flat_grey() {
        let sigmas = [0.01, 0.02, 0.03];
        let img = noisy_grey(1024, 768, sigmas, 7);
        let want_y = analytic(Y_W, sigmas);
        let want_c =
            (0.5 * (analytic(CB_W, sigmas).powi(2) + analytic(CR_W, sigmas).powi(2))).sqrt();
        let given = estimate_in(&img, Rect::new(100, 80, 300, 220));
        assert_eq!(given.rect, Rect::new(100, 80, 300, 220));
        let found = estimate(&img).expect("a flat region");
        for (name, est) in [("given", given), ("detector", found)] {
            println!(
                "{name} rectangle {:?}: σY {:.5} (analytic {want_y:.5}), σC {:.5} (analytic {want_c:.5})",
                est.rect, est.sigma_y, est.sigma_c
            );
            assert!(
                (est.sigma_y / want_y - 1.0).abs() < 0.10,
                "{name}: σY {}",
                est.sigma_y
            );
            assert!(
                (est.sigma_c / want_c - 1.0).abs() < 0.10,
                "{name}: σC {}",
                est.sigma_c
            );
            assert_eq!(est.level_y, libm::pow((5.0 * est.sigma_y).min(1.0), 0.7));
            assert_eq!(est.level_c, libm::pow((5.0 * est.sigma_c).min(1.0), 0.7));
        }
        // The detector's rectangle: a 22 px square at the coarsest level (768 / 256: one step), so 44 px at full
        // resolution, less where it is clipped at the frame's edge.
        assert!((23..=44).contains(&found.rect.w) && (23..=44).contains(&found.rect.h));
        assert!(found.rect.x1() <= 1024 && found.rect.y1() <= 768);
    }

    #[test]
    fn levels_saturate_at_one_and_vanish_without_noise() {
        assert_eq!(level(0.0), 0.0);
        assert_eq!(level(0.2), 1.0);
        assert_eq!(level(0.5), 1.0);
        assert!((level(0.1) - 0.5f64.powf(0.7)).abs() < 1e-15);
        let flat = noisy_grey(64, 64, [0.0; 3], 1);
        let est = estimate(&flat).expect("a flat region");
        // Rounding of the normalised blur leaves at most a few ulp.
        assert!(est.sigma_y < 1e-12 && est.sigma_c < 1e-12, "{est:?}");
        assert!(est.level_y < 1e-6 && est.level_c < 1e-6, "{est:?}");
    }

    #[test]
    fn rectangle_is_clipped_and_a_tiny_image_has_no_estimate() {
        let img = noisy_grey(40, 30, [0.02; 3], 2);
        let est = estimate_in(&img, Rect::new(30, 20, 50, 50));
        assert_eq!(est.rect, Rect::new(30, 20, 10, 10));
        assert!(est.sigma_y > 0.0);
        assert_eq!(estimate_in(&img, Rect::new(45, 0, 5, 5)).sigma_y, 0.0);
        assert!(estimate(&noisy_grey(12, 12, [0.02; 3], 3)).is_none());
    }
}
