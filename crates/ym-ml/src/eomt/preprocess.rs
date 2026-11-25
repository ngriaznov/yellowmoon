//! The picture to the network's input (`docs/34` section 4b.1): the 8-bit picture is resized to fit 640 x 640 keeping its aspect
//! ratio, padded at the bottom and the right with 0 (in 8-bit values), rescaled by 1/255 and normalised with the ImageNet mean and
//! standard deviation, as Hugging Face's `EomtImageProcessor` does for this checkpoint (`preprocessor_config.json`:
//! `size {shortest_edge 640, longest_edge 640}`, `resample` bilinear, `do_pad`, `do_normalize`).
//!
//! The size rule is `get_size_with_aspect_ratio(size, 640, 640)` of `transformers.image_transforms`, with the same `f64`
//! operations in the same order, so that the size comes out the same. The resize is `torchvision.transforms.v2.functional.resize`
//! on an 8-bit tensor with `antialias=True`, which PyTorch runs as Pillow does: a separable triangle filter whose support grows with
//! the shrink factor (the taps of `sam2::preprocess`), the width first, the weights as integers of 22 fractional bits, every pass
//! `(sum of weight * value + 2^21) >> 22` clamped to 0..=255 and stored as an 8-bit value. It is integer arithmetic, so it gives the
//! same bits everywhere. The same values as the reference on all but 0 to 0.2 % of the pixels, which differ by one level
//! (measured on the fixture's pictures and on three photographs).

use rayon::prelude::*;

use crate::sam2::preprocess::taps_f64;

/// The side of the (square, padded) network input, in pixels.
pub const IMAGE_PX: usize = 640;
/// The ImageNet mean.
pub(crate) const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// The ImageNet standard deviation.
pub(crate) const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// `rescale_factor` of the processor (1/255 as Python prints it), as `f32`.
const RESCALE: f32 = 0.003_921_568_6;

/// The size (height, width) a `height` x `width` picture is resized to before it is padded: `size = shortest_edge = longest_edge
/// = 640` of `get_size_with_aspect_ratio`. Both are at least 1 and at most 640.
pub fn fit_size(height: usize, width: usize) -> (usize, usize) {
    let (h, w) = (height as f64, width as f64);
    let size = IMAGE_PX as f64;
    let min = h.min(w);
    let max = h.max(w);
    let mut raw: Option<f64> = None;
    let mut short = IMAGE_PX;
    if max / min * size > size {
        let r = size * min / max;
        raw = Some(r);
        // Python's `int(round(raw))`: round half to even.
        short = (r.round_ties_even() as usize).max(1);
    }
    let (oh, ow) = if (height <= width && height == short) || (width <= height && width == short) {
        (height, width)
    } else if width < height {
        let oh = match raw {
            Some(r) => (r * h / w) as usize,
            None => (short as f64 * h / w) as usize,
        };
        (oh, short)
    } else {
        let ow = match raw {
            Some(r) => (r * w / h) as usize,
            None => (short as f64 * w / h) as usize,
        };
        (short, ow)
    };
    (oh.clamp(1, IMAGE_PX), ow.clamp(1, IMAGE_PX))
}

/// Fractional bits of the integer weights (Pillow's and PyTorch's `PRECISION_BITS = 32 - 8 - 2`).
const PRECISION: u32 = 22;

/// The taps of one axis with integer weights: `(first input index, weights)`; each weight is `floor(w * 2^22 + 0.5)`.
fn fixed_taps(n_in: usize, n_out: usize) -> Vec<(usize, Vec<i32>)> {
    taps_f64(n_in, n_out)
        .into_iter()
        .map(|(start, w)| {
            (
                start,
                w.into_iter()
                    .map(|v| (v * f64::from(1u32 << PRECISION) + 0.5).floor() as i32)
                    .collect(),
            )
        })
        .collect()
}

/// One output value of the 8-bit kernel: the weights (which sum to about 2^22) times the 8-bit values, rounded and shifted.
fn level(sum: i32) -> u8 {
    ((sum + (1 << (PRECISION - 1))) >> PRECISION).clamp(0, 255) as u8
}

/// The resized picture as 8-bit values, `[oh][ow][3]`.
fn resize_levels(rgb: &[u8], width: usize, height: usize, ow: usize, oh: usize) -> Vec<u8> {
    let (tx, ty) = (fixed_taps(width, ow), fixed_taps(height, oh));
    let mut wide = vec![0u8; height * ow * 3];
    wide.par_chunks_mut(ow * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let src = &rgb[y * width * 3..(y + 1) * width * 3];
            for (x, (start, weights)) in tx.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0i32;
                    for (j, &w) in weights.iter().enumerate() {
                        acc += w * i32::from(src[(start + j) * 3 + c]);
                    }
                    row[x * 3 + c] = level(acc);
                }
            }
        });
    let mut out = vec![0u8; oh * ow * 3];
    out.par_chunks_mut(ow * 3).enumerate().for_each(|(y, row)| {
        let (start, weights) = &ty[y];
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0i32;
                for (j, &w) in weights.iter().enumerate() {
                    acc += w * i32::from(wide[((start + j) * ow + x) * 3 + c]);
                }
                row[x * 3 + c] = level(acc);
            }
        }
    });
    out
}

/// The network's input for an 8-bit RGB picture of `width` x `height` pixels (rows top to bottom, `R G B` per pixel): the picture
/// as `[640][640][3]` `f32`, resized to fit, padded and normalised, and the size it was resized to (height, width).
///
/// Callers have checked that the picture is not empty and that `rgb` holds `3 * width * height` bytes.
pub fn input_tensor(rgb: &[u8], width: usize, height: usize) -> (Vec<f32>, (usize, usize)) {
    debug_assert_eq!(rgb.len(), 3 * width * height);
    let (oh, ow) = fit_size(height, width);
    let levels = resize_levels(rgb, width, height, ow, oh);
    let normal = |level: f32, c: usize| (level * RESCALE - MEAN[c]) / STD[c];
    let pad = [normal(0.0, 0), normal(0.0, 1), normal(0.0, 2)];
    let mut out = vec![0.0f32; IMAGE_PX * IMAGE_PX * 3];
    out.par_chunks_mut(IMAGE_PX * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..IMAGE_PX {
                for c in 0..3 {
                    row[x * 3 + c] = if y < oh && x < ow {
                        normal(f32::from(levels[(y * ow + x) * 3 + c]), c)
                    } else {
                        pad[c]
                    };
                }
            }
        });
    (out, (oh, ow))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_size_rule_is_the_one_of_get_size_with_aspect_ratio() {
        // Values computed with `transformers.image_transforms.get_size_with_aspect_ratio((h, w), 640, 640)`.
        assert_eq!(fit_size(1024, 820), (640, 512));
        assert_eq!(fit_size(426, 640), (426, 640));
        assert_eq!(fit_size(960, 640), (640, 427));
        assert_eq!(fit_size(640, 640), (640, 640));
        assert_eq!(fit_size(100, 100), (640, 640));
        assert_eq!(fit_size(683, 1024), (427, 640));
        assert_eq!(fit_size(1024, 683), (640, 427));
        // A square picture of any size is resized to 640 x 640.
        assert_eq!(fit_size(1, 1), (640, 640));
        assert_eq!(fit_size(1080, 1920), (360, 640));
        assert_eq!(fit_size(3, 7), (274, 640));
    }

    #[test]
    fn the_side_is_the_networks() {
        assert_eq!(IMAGE_PX, crate::eomt::network::IMAGE_PX);
    }

    #[test]
    fn every_size_fits_in_the_input() {
        for (h, w) in [
            (1, 5000),
            (5000, 1),
            (3, 7),
            (2000, 3000),
            (1080, 1920),
            (7, 1),
        ] {
            let (oh, ow) = fit_size(h, w);
            assert!(
                (1..=IMAGE_PX).contains(&oh) && (1..=IMAGE_PX).contains(&ow),
                "{h}x{w} -> {oh}x{ow}"
            );
        }
    }

    #[test]
    fn a_picture_of_the_fit_size_is_only_normalised_and_padded() {
        let (w, h) = (640usize, 426usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        let (t, (oh, ow)) = input_tensor(&rgb, w, h);
        assert_eq!((oh, ow), (426, 640));
        assert_eq!(t.len(), IMAGE_PX * IMAGE_PX * 3);
        for i in (0..w * h * 3).step_by(997) {
            let want = (f32::from(rgb[i]) * RESCALE - MEAN[i % 3]) / STD[i % 3];
            assert_eq!(t[i].to_bits(), want.to_bits(), "value {i}");
        }
        // Row 426 and below are the padding: the normalised 0.
        let pad = (0.0 * RESCALE - MEAN[1]) / STD[1];
        assert_eq!(t[(430 * IMAGE_PX + 10) * 3 + 1].to_bits(), pad.to_bits());
    }

    #[test]
    fn a_flat_picture_stays_flat() {
        let rgb = vec![200u8; 37 * 23 * 3];
        let (t, (oh, ow)) = input_tensor(&rgb, 37, 23);
        assert_eq!((oh, ow), fit_size(23, 37));
        let want = (200.0 * RESCALE - MEAN[0]) / STD[0];
        assert_eq!(t[0].to_bits(), want.to_bits());
        assert_eq!(
            t[((oh - 1) * IMAGE_PX + ow - 1) * 3].to_bits(),
            want.to_bits()
        );
    }
}
