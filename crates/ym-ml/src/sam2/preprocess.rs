//! The picture to the encoder's input (`docs/algorithms/06-masks.md` section 6.2): an antialiased bilinear
//! resize to 1024 x 1024 that does not keep the aspect ratio, then `(x - mean) / std` per channel.
//!
//! The resize is our own: a separable triangle filter whose support is scaled by the shrink factor,
//! written from the description of what PyTorch's `interpolate(..., mode="bilinear", align_corners=False,
//! antialias=True)` computes (the filter and the support of Pillow's resampling, which torchvision's
//! `Resize` on a tensor selects). The width goes first and the height second, the weights are computed
//! in `f64` and rounded to `f32`, and each output value is the weights times the inputs summed in index order in
//! `f32`.

use rayon::prelude::*;

use super::{IMAGE_PX, PIXEL_MEAN, PIXEL_STD};

/// The taps of one output index along one axis: the first input index and the weights, which sum to 1.
#[derive(Debug, Clone)]
pub(crate) struct Taps {
    /// First input index.
    pub start: usize,
    /// Weights of the inputs `start..`.
    pub weights: Vec<f32>,
}

/// The triangle filter taps for resizing `n_in` samples to `n_out` (PyTorch's antialiased bilinear with
/// `align_corners = false`): scale `n_in / n_out`; the support is the scale when shrinking and 1 when
/// enlarging; output `i` is centred on `scale * (i + 0.5)` and takes the inputs whose centres fall within
/// the support, each weighted by `max(0, 1 - |distance / max(scale, 1)|)`; the weights are normalised to
/// sum to 1.
pub(crate) fn taps(n_in: usize, n_out: usize) -> Vec<Taps> {
    let scale = n_in as f64 / n_out as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..n_out)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = (((center + support + 0.5) as i64).max(0) as usize).min(n_in);
            let hi = hi.max(lo + 1).min(n_in);
            let lo = lo.min(hi - 1);
            let mut w: Vec<f64> = (lo..hi)
                .map(|j| {
                    let x = ((j as f64) - center + 0.5) * inv;
                    (1.0 - x.abs()).max(0.0)
                })
                .collect();
            let total: f64 = w.iter().sum();
            if total > 0.0 {
                for v in &mut w {
                    *v /= total;
                }
            } else {
                // No input centre inside the support (cannot happen for sizes of one or more): the nearest input.
                let near = ((center - 0.5).round().max(lo as f64) as usize).min(hi - 1) - lo;
                w.iter_mut().for_each(|v| *v = 0.0);
                w[near] = 1.0;
            }
            Taps {
                start: lo,
                weights: w.into_iter().map(|v| v as f32).collect(),
            }
        })
        .collect()
}

/// The model's input for an 8-bit RGB picture of `width` x `height` pixels (rows top to bottom, `R G B` per pixel):
/// the picture as `[1024][1024][3]` `f32` (rows, columns, channels), resized and normalised.
///
/// Callers have checked that the picture is not empty and that `rgb` holds `3 * width * height` bytes.
pub(crate) fn input_tensor(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    debug_assert_eq!(rgb.len(), 3 * width * height);
    let (tx, ty) = (taps(width, IMAGE_PX), taps(height, IMAGE_PX));
    // Width first: `[height][1024][3]`, from the picture as `u8 / 255` in `f32` (torchvision's `ToTensor`).
    let mut wide = vec![0.0f32; height * IMAGE_PX * 3];
    wide.par_chunks_mut(IMAGE_PX * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let src = &rgb[y * width * 3..(y + 1) * width * 3];
            for (x, t) in tx.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &w) in t.weights.iter().enumerate() {
                        acc += w * (f32::from(src[(t.start + j) * 3 + c]) / 255.0);
                    }
                    row[x * 3 + c] = acc;
                }
            }
        });
    // Then the height, and the normalisation.
    let mut out = vec![0.0f32; IMAGE_PX * IMAGE_PX * 3];
    out.par_chunks_mut(IMAGE_PX * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let t = &ty[y];
            for x in 0..IMAGE_PX {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &w) in t.weights.iter().enumerate() {
                        acc += w * wide[((t.start + j) * IMAGE_PX + x) * 3 + c];
                    }
                    row[x * 3 + c] = (acc - PIXEL_MEAN[c]) / PIXEL_STD[c];
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_sum_to_one_and_a_same_size_resize_is_the_identity() {
        for (n_in, n_out) in [
            (1024, 1024),
            (736, 1024),
            (1024, 1024),
            (2000, 1024),
            (7, 1024),
            (1, 1024),
        ] {
            for t in taps(n_in, n_out) {
                let sum: f64 = t.weights.iter().map(|&w| f64::from(w)).sum();
                assert!((sum - 1.0).abs() < 1e-6, "{n_in} to {n_out}: {sum}");
                assert!(t.start + t.weights.len() <= n_in);
            }
        }
        let identity = taps(1024, 1024);
        for (i, t) in identity.iter().enumerate() {
            // PyTorch reads one input more than the filter needs (its weight is exactly 0).
            assert_eq!(t.start, i);
            assert_eq!(t.weights[0], 1.0);
            assert!(t.weights[1..].iter().all(|&w| w == 0.0), "{t:?}");
        }
    }

    #[test]
    fn enlarging_is_plain_bilinear_with_half_pixel_centres() {
        // 2 to 4: output 1 is centred on 0.5 * (1 + 0.5) = 0.75 in input pixel units, between input centres 0.5 and 1.5.
        let t = taps(2, 4);
        assert_eq!(t[1].start, 0);
        assert!(
            (t[1].weights[0] - 0.75).abs() < 1e-6 && (t[1].weights[1] - 0.25).abs() < 1e-6,
            "{:?}",
            t[1]
        );
        // The ends repeat the edge pixel.
        assert_eq!(t[0].start, 0);
        assert!((t[0].weights[0] - 1.0).abs() < 1e-6, "{:?}", t[0]);
    }

    #[test]
    fn shrinking_widens_the_support() {
        // 4 to 2: scale 2, support 2: output 0 is centred on 1.0 and reads inputs 0..3 with the triangle of half-width 2.
        let t = taps(4, 2);
        assert_eq!(t[0].start, 0);
        assert_eq!(t[0].weights.len(), 3);
        let w: Vec<f64> = t[0].weights.iter().map(|&v| f64::from(v)).collect();
        // distances (j + 0.5 - 1) / 2 = -0.25, 0.25, 0.75 -> 0.75, 0.75, 0.25, normalised by 1.75.
        assert!(
            (w[0] - 0.75 / 1.75).abs() < 1e-6
                && (w[1] - 0.75 / 1.75).abs() < 1e-6
                && (w[2] - 0.25 / 1.75).abs() < 1e-6,
            "{w:?}"
        );
    }

    #[test]
    fn a_picture_of_the_model_s_size_is_only_normalised() {
        let rgb: Vec<u8> = (0..IMAGE_PX * IMAGE_PX * 3)
            .map(|i| (i * 7 % 256) as u8)
            .collect();
        let t = input_tensor(&rgb, IMAGE_PX, IMAGE_PX);
        for i in (0..t.len()).step_by(997) {
            let want = (f32::from(rgb[i]) / 255.0 - PIXEL_MEAN[i % 3]) / PIXEL_STD[i % 3];
            assert_eq!(t[i].to_bits(), want.to_bits(), "value {i}");
        }
    }

    #[test]
    fn the_smallest_pictures_and_extreme_shapes_resize_without_trouble() {
        for (w, h) in [(1usize, 1usize), (1, 3), (3, 1), (2, 5000), (5000, 2)] {
            let rgb = vec![90u8; w * h * 3];
            let t = input_tensor(&rgb, w, h);
            let want = (90.0 / 255.0 - PIXEL_MEAN[1]) / PIXEL_STD[1];
            assert!(
                (t[1] - want).abs() < 1e-5 && (t[t.len() - 2] - want).abs() < 1e-5,
                "{w} x {h}"
            );
        }
    }

    #[test]
    fn a_flat_picture_stays_flat_and_is_normalised() {
        let (w, h) = (37, 23);
        let rgb = vec![200u8; w * h * 3];
        let t = input_tensor(&rgb, w, h);
        assert_eq!(t.len(), IMAGE_PX * IMAGE_PX * 3);
        for c in 0..3 {
            let want = (200.0 / 255.0 - PIXEL_MEAN[c]) / PIXEL_STD[c];
            for px in [0usize, 1000 * IMAGE_PX + 3, 1023 * IMAGE_PX + 1023] {
                assert!(
                    (t[px * 3 + c] - want).abs() < 1e-5,
                    "channel {c}: {} against {want}",
                    t[px * 3 + c]
                );
            }
        }
    }
}
