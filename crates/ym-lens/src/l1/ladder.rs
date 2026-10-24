//! Separable Gaussian blurs and gradients on single-channel planes: the scale ladder of the bokeh detector.
//!
//! Every output sample is computed on its own from the input, in a fixed order, so rows can run in parallel and
//! the result is the same bits for any number of threads.

use rayon::prelude::*;

/// A normalised, symmetric Gaussian kernel `k[0..=r]` of standard deviation `sigma` (`k[0]` the centre).
pub(super) fn kernel(sigma: f64) -> Vec<f64> {
    let r = (3.5 * sigma).ceil().max(1.0) as usize;
    let mut k: Vec<f64> = (0..=r)
        .map(|i| {
            let x = i as f64 / sigma;
            libm::exp(-0.5 * x * x)
        })
        .collect();
    let sum = k[0] + 2.0 * k[1..].iter().sum::<f64>();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// `src` (`w x h`, row-major) blurred with a Gaussian of `sigma` pixels; the border is replicated.
///
/// Each output sample is a sum over the taps in a fixed order (the centre, then each tap's pair outwards), so
/// rows can run in parallel with the same bits for any number of threads.
pub(super) fn blur(src: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    debug_assert_eq!(src.len(), w * h);
    let k = kernel(sigma);
    let r = k.len() - 1;
    // Horizontal pass: the interior without clamps, the borders with them.
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            let mut acc = k[0] * f64::from(s[x]);
            if x >= r && x + r < w {
                for (i, ki) in k.iter().enumerate().skip(1) {
                    acc += ki * (f64::from(s[x - i]) + f64::from(s[x + i]));
                }
            } else {
                for (i, ki) in k.iter().enumerate().skip(1) {
                    let lo = x.saturating_sub(i);
                    let hi = (x + i).min(w - 1);
                    acc += ki * (f64::from(s[lo]) + f64::from(s[hi]));
                }
            }
            *out = acc as f32;
        }
    });
    // Vertical pass: one output row at a time, adding whole rows (sequential memory).
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut acc: Vec<f64> = tmp[y * w..(y + 1) * w].iter().map(|v| k[0] * f64::from(*v)).collect();
        for (i, ki) in k.iter().enumerate().skip(1) {
            let lo = &tmp[y.saturating_sub(i) * w..][..w];
            let hi = &tmp[(y + i).min(h - 1) * w..][..w];
            for x in 0..w {
                acc[x] += ki * (f64::from(lo[x]) + f64::from(hi[x]));
            }
        }
        for (o, a) in row.iter_mut().zip(&acc) {
            *o = *a as f32;
        }
    });
    out
}

/// Central-difference gradient `(gx, gy)` at `(x, y)` of a `w x h` plane (border replicated).
#[inline]
pub(super) fn gradient_at(s: &[f32], w: usize, h: usize, x: usize, y: usize) -> (f64, f64) {
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(w - 1);
    let yu = y.saturating_sub(1);
    let yd = (y + 1).min(h - 1);
    let gx = 0.5 * (f64::from(s[y * w + xr]) - f64::from(s[y * w + xl]));
    let gy = 0.5 * (f64::from(s[yd * w + x]) - f64::from(s[yu * w + x]));
    (gx, gy)
}

/// The gradient magnitude of every sample of a `w x h` plane.
pub(super) fn gradient_magnitude(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (gx, gy) = gradient_at(src, w, h, x, y);
            *o = (gx * gx + gy * gy).sqrt() as f32;
        }
    });
    out
}

/// The `p`-quantile (0 to 1) of the values, by selection; `values` is reordered.
pub(super) fn quantile(values: &mut [f32], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    let (_, v, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    f64::from(*v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sums_to_one() {
        for s in [0.5, 1.0, 1.4142, 4.0, 11.3] {
            let k = kernel(s);
            let sum = k[0] + 2.0 * k[1..].iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn blur_keeps_a_constant_and_the_mean() {
        let (w, h) = (40, 30);
        let flat = vec![0.7f32; w * h];
        let b = blur(&flat, w, h, 3.0);
        assert!(b.iter().all(|v| (v - 0.7).abs() < 1e-6));
        let mut img = vec![0.0f32; w * h];
        img[15 * w + 20] = 1.0;
        let b = blur(&img, w, h, 2.0);
        let sum: f64 = b.iter().map(|v| f64::from(*v)).sum();
        assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    }

    #[test]
    fn blur_of_a_step_has_the_gaussian_gradient_peak() {
        let (w, h) = (128, 8);
        let mut img = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 64..w {
                img[y * w + x] = 1.0;
            }
        }
        let sigma = 4.0;
        let b = blur(&img, w, h, sigma);
        let g = gradient_magnitude(&b, w, h);
        let peak = g[4 * w + 63].max(g[4 * w + 64]);
        let expect = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        assert!((f64::from(peak) - expect).abs() < 0.02 * expect, "{peak} vs {expect}");
    }
}
