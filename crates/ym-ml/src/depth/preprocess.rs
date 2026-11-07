//! The pre-processing of Depth Anything V2's `image2tensor` (`depth_anything_v2/dpt.py` lines 196 to 221 and
//! `util/transform.py`, github.com/DepthAnything/Depth-Anything-V2, Apache-2.0): the size the picture is resized to, the cubic
//! resize and the ImageNet normalisation.
//!
//! **Parity to be confirmed by the oracle.** The size rule is a direct port of upstream's. The resize is written from
//! the documented behaviour of OpenCV's `cv2.resize(..., interpolation=cv2.INTER_CUBIC)` (half-pixel centres, a 4 x 4 window
//! with Keys' kernel at `A = -0.75`, border replicate, no antialiasing when shrinking, `f32` weights, horizontal pass and then
//! vertical pass in `f64`); it has not been compared with OpenCV yet. `tests/depth.rs` measures it against the oracle's
//! tensors when the fixtures exist.

use rayon::prelude::*;

use super::{MAX_SIDE_PX, PATCH};
use crate::model::{MlError, Result};

/// `input_size=518` of `infer_image` (`dpt.py` line 187): the short side of the network input.
pub const NETWORK_SHORT_SIDE_PX: usize = 518;
/// `NormalizeImage(mean=[0.485, 0.456, 0.406], ...)` (`dpt.py` line 207).
pub const IMAGE_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
/// `NormalizeImage(..., std=[0.229, 0.224, 0.225])` (`dpt.py` line 207).
pub const IMAGE_STD: [f64; 3] = [0.229, 0.224, 0.225];

/// A normalised picture for the network: `[height][width][3]`, `(x - mean) / std` per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    /// Width in pixels (a multiple of 14).
    pub width: usize,
    /// Height in pixels (a multiple of 14).
    pub height: usize,
    /// `width * height * 3` values.
    pub data: Vec<f32>,
}

/// `Resize.constrain_to_multiple_of(x, min_val=518)` (`transform.py` lines 51 to 60) for `resize_method='lower_bound'`:
/// `np.round(x / 14) * 14` (numpy rounds halves to even), and the ceiling instead when that is under `min_val`.
fn constrain_to_multiple_of(x: f64, min_val: f64) -> usize {
    let m = PATCH as f64;
    let mut y = (x / m).round_ties_even() * m;
    if y < min_val {
        y = (x / m).ceil() * m;
    }
    y as usize
}

/// The size `Resize.get_size` (`transform.py` lines 62 to 107) gives with `width=height=518`, `keep_aspect_ratio=True`,
/// `ensure_multiple_of=14` and `resize_method='lower_bound'`: the larger of the two scale factors is used for both sides
/// (the short side becomes 518), each side is rounded to a multiple of 14 and is at least 518. Returns `(width, height)`.
pub fn network_size(width: usize, height: usize) -> (usize, usize) {
    let target = NETWORK_SHORT_SIDE_PX as f64;
    let mut scale_height = target / height as f64;
    let mut scale_width = target / width as f64;
    if scale_width > scale_height {
        scale_height = scale_width;
    } else {
        scale_width = scale_height;
    }
    (
        constrain_to_multiple_of(scale_width * width as f64, target),
        constrain_to_multiple_of(scale_height * height as f64, target),
    )
}

/// OpenCV's cubic weights for the fraction `x` (`interpolateCubic`, `A = -0.75`), `f32`.
fn opencv_cubic(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// One axis of the resize: for each destination index the four clamped source indices and weights. The source coordinate is
/// `(dst + 0.5) * (src / dst) - 0.5` in `f32` (`resize` of `imgwarp.cpp`: `fx = (float)((dx + 0.5) * scale_x - 0.5)` with a
/// `double` scale).
fn axis(n_src: usize, n_dst: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = n_src as f64 / n_dst as f64;
    (0..n_dst)
        .map(|d| {
            let fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let sx = fx.floor();
            let idx = std::array::from_fn(|k| {
                (sx as isize - 1 + k as isize).clamp(0, n_src as isize - 1) as usize
            });
            (idx, opencv_cubic(fx - sx))
        })
        .collect()
}

/// Resizes `rgb` (`[sh][sw][3]`, values in 0..1) to `dw` x `dh` and normalises it with the ImageNet mean and standard deviation
/// in `f64`, as `image2tensor` does for `(image / 255)`: resize first, normalise after. The resize is a separable 4-tap cubic;
/// there is no low-pass before shrinking, as upstream.
fn resize_normalise(rgb: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let (ax, ay) = (axis(sw, dw), axis(sh, dh));
    let mut out = vec![0.0f32; dw * dh * 3];
    out.par_chunks_mut(dw * 3).enumerate().for_each(|(y, row)| {
        let (iy, wy) = &ay[y];
        // The four source rows, resized horizontally.
        let mut lines = [
            vec![0.0f64; dw * 3],
            vec![0.0f64; dw * 3],
            vec![0.0f64; dw * 3],
            vec![0.0f64; dw * 3],
        ];
        for (line, &sy) in lines.iter_mut().zip(iy) {
            let src = &rgb[sy * sw * 3..][..sw * 3];
            for (x, (ix, wx)) in ax.iter().enumerate() {
                for ch in 0..3 {
                    line[x * 3 + ch] = f64::from(src[ix[0] * 3 + ch]) * f64::from(wx[0])
                        + f64::from(src[ix[1] * 3 + ch]) * f64::from(wx[1])
                        + f64::from(src[ix[2] * 3 + ch]) * f64::from(wx[2])
                        + f64::from(src[ix[3] * 3 + ch]) * f64::from(wx[3]);
                }
            }
        }
        for (i, d) in row.iter_mut().enumerate() {
            let v = lines[0][i] * f64::from(wy[0])
                + lines[1][i] * f64::from(wy[1])
                + lines[2][i] * f64::from(wy[2])
                + lines[3][i] * f64::from(wy[3]);
            let ch = i % 3;
            *d = ((v - IMAGE_MEAN[ch]) / IMAGE_STD[ch]) as f32;
        }
    });
    out
}

/// Prepares a picture for the network: [`network_size`], the cubic resize and the normalisation. `rgb` holds `[y][x][rgb]` sRGB
/// values in 0..1 (the 8-bit values divided by 255).
///
/// # Errors
///
/// [`MlError::Shape`] when `rgb` is not `3 * width * height` values, the picture is empty, or the network size would be
/// over [`MAX_SIDE_PX`] on a side (a panorama of more than about 4:1); [`MlError::Input`] for a value that is not finite.
pub fn prepare_input(rgb: &[f32], width: usize, height: usize) -> Result<Prepared> {
    if width == 0 || height == 0 || rgb.len() != width * height * 3 {
        return Err(MlError::Shape(format!(
            "a {width} x {height} picture needs {} values, found {}",
            width * height * 3,
            rgb.len()
        )));
    }
    if rgb.iter().any(|v| !v.is_finite()) {
        return Err(MlError::Input(
            "the picture holds a value that is not finite".into(),
        ));
    }
    let (w, h) = network_size(width, height);
    if w > MAX_SIDE_PX || h > MAX_SIDE_PX {
        return Err(MlError::Shape(format!(
            "a {width} x {height} picture would need a {w} x {h} network input (over {MAX_SIDE_PX} px on a side)"
        )));
    }
    Ok(Prepared {
        width: w,
        height: h,
        data: resize_normalise(rgb, width, height, w, h),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_network_size_follows_the_upstream_rule() {
        // 3:2 landscape and portrait: short side 518, long side 777 rounded to 784 (56 * 14).
        assert_eq!(network_size(6000, 4000), (784, 518));
        assert_eq!(network_size(4000, 6000), (518, 784));
        assert_eq!(network_size(1000, 1000), (518, 518));
        // 16:9: 518 * 16 / 9 = 920.9 -> 921 / 14 = 65.8 -> 66 * 14 = 924.
        assert_eq!(network_size(1920, 1080), (924, 518));
        // A photo smaller than the network size is scaled up to it.
        assert_eq!(network_size(300, 200), (784, 518));
        // 1:1.5 and 4:3 (518 * 4 / 3 = 690.7 -> 49.3 -> 49 * 14 = 686).
        assert_eq!(network_size(4032, 3024), (686, 518));
        // Halves go to the even multiple (numpy's rounding): x / 14 = 40.5 -> 40.
        assert_eq!(constrain_to_multiple_of(567.0, 518.0), 560);
        assert_eq!(constrain_to_multiple_of(581.0, 518.0), 588);
        // The short side is 518 up to rounding error and rounds to 518; under the minimum the ceiling is taken (upstream's rule;
        // it cannot happen for a side that is at least the short side).
        assert_eq!(constrain_to_multiple_of(517.9999, 518.0), 518);
        assert_eq!(constrain_to_multiple_of(500.0, 518.0), 504);
        // Both sides are always multiples of 14 and at least 518.
        for (w, h) in [
            (5, 7),
            (1, 3000),
            (3000, 1),
            (777, 778),
            (6240, 4160),
            (4160, 6240),
        ] {
            let (nw, nh) = network_size(w, h);
            assert!(
                nw % 14 == 0 && nh % 14 == 0 && nw >= 518 && nh >= 518,
                "{w}x{h} -> {nw}x{nh}"
            );
        }
    }

    #[test]
    fn the_cubic_weights_are_opencvs() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let w = opencv_cubic(x);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            // They are the Keys kernel with A = -0.75 at the four distances (a cross-check in f64).
            let k = |d: f64| {
                const A: f64 = -0.75;
                let d = d.abs();
                if d <= 1.0 {
                    ((A + 2.0) * d - (A + 3.0)) * d * d + 1.0
                } else if d < 2.0 {
                    ((A * d - 5.0 * A) * d + 8.0 * A) * d - 4.0 * A
                } else {
                    0.0
                }
            };
            let x = f64::from(x);
            for (j, d) in [1.0 + x, x, 1.0 - x, 2.0 - x].into_iter().enumerate() {
                assert!((f64::from(w[j]) - k(d)).abs() < 2e-6, "x {x} tap {j}");
            }
        }
        assert_eq!(opencv_cubic(0.0), [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_constant_picture_stays_constant_and_the_same_size_is_the_identity() {
        let (w, h) = (37, 29);
        let rgb: Vec<f32> = (0..w * h).flat_map(|_| [0.2f32, 0.5, 0.8]).collect();
        let out = resize_normalise(&rgb, w, h, 70, 45);
        for px in out.chunks_exact(3) {
            for ch in 0..3 {
                let want =
                    ((f64::from([0.2f32, 0.5, 0.8][ch]) - IMAGE_MEAN[ch]) / IMAGE_STD[ch]) as f32;
                assert!((px[ch] - want).abs() < 1e-5);
            }
        }
        let ramp: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32 / 251.0).collect();
        let same = resize_normalise(&ramp, w, h, w, h);
        for (i, (&o, &r)) in same.iter().zip(&ramp).enumerate() {
            let ch = i % 3;
            let want = ((f64::from(r) - IMAGE_MEAN[ch]) / IMAGE_STD[ch]) as f32;
            assert!((o - want).abs() < 1e-5, "{i}");
        }
    }

    #[test]
    fn a_linear_ramp_stays_linear_when_resized() {
        // Cubic convolution reproduces a linear function, away from the replicated border.
        let (sw, sh, dw, dh) = (60usize, 40usize, 90usize, 70usize);
        let rgb: Vec<f32> = (0..sw * sh)
            .flat_map(|i| {
                let v = (i % sw) as f32 / sw as f32 * 0.5 + 0.25;
                [v, v, v]
            })
            .collect();
        let out = resize_normalise(&rgb, sw, sh, dw, dh);
        for x in 6..dw - 6 {
            let src_x = (x as f64 + 0.5) * sw as f64 / dw as f64 - 0.5;
            let v = src_x / sw as f64 * 0.5 + 0.25;
            let want = ((v - IMAGE_MEAN[1]) / IMAGE_STD[1]) as f32;
            let got = out[(20 * dw + x) * 3 + 1];
            assert!((got - want).abs() < 1e-4, "x {x}: {got} vs {want}");
        }
    }

    #[test]
    fn prepare_input_checks_its_input() {
        let rgb = vec![0.5f32; 30 * 20 * 3];
        let p = prepare_input(&rgb, 30, 20).expect("a small picture is scaled up");
        assert_eq!((p.width, p.height), (784, 518));
        assert_eq!(p.data.len(), 784 * 518 * 3);
        assert!(prepare_input(&rgb, 30, 21).is_err());
        assert!(prepare_input(&[], 0, 0).is_err());
        let mut bad = rgb.clone();
        bad[5] = f32::NAN;
        assert!(matches!(
            prepare_input(&bad, 30, 20),
            Err(MlError::Input(_))
        ));
        // A 10:1 strip would need a network input far over the limit.
        let strip = vec![0.0f32; 1000 * 100 * 3];
        assert!(matches!(
            prepare_input(&strip, 1000, 100),
            Err(MlError::Shape(_))
        ));
    }
}
