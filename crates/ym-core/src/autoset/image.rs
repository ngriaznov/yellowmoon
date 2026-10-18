//! The image operations of the `autoset` dump (`docs/18-auto-settings.md` section 6): the area resize that brings two
//! images to one size, the central crop and the quarter turn of the embedded JPEG, and the luminance of its pixels.
//!
//! Everything here is plain arithmetic in f64 on row-major buffers: no decoder and no file access (the dump decodes
//! the JPEG with `zune-jpeg`).

use ym_color::consts::Vec3;
use ym_color::exposure::srgb_to_linear;

/// Rec.709 luminance weights of linear sRGB (`tools/eval/metrics.py`, `SRGB_Y_ROW`; `ym_color`'s gate uses the same).
pub const SRGB_Y_ROW: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Adobe RGB (1998) luminance weights (`tools/eval/metrics.py`, `ADOBE_RGB_Y_ROW`).
pub const ADOBE_RGB_Y_ROW: [f64; 3] = [0.29734, 0.62736, 0.07529];

/// Adobe RGB (1998) decoding: a pure power of 563/256 (`ADOBE_RGB_GAMMA`).
const ADOBE_RGB_GAMMA: f64 = 563.0 / 256.0;

/// Area-average resample of a row-major `sw` x `sh` image to `dw` x `dh` (PIL's `BOX` filter on floats).
///
/// Output pixel `(x, y)` covers the source rectangle `[x sw/dw, (x+1) sw/dw) x [y sh/dh, (y+1) sh/dh)`; its value is
/// the mean of the source pixels under it, each weighted by the exact fraction of the pixel inside the rectangle.
/// Downsampling is the use; a larger output reads the source pixel under each output pixel, as the box filter does.
///
/// Non-finite source values take no part: an output pixel is the weighted mean of its finite source values, and NaN
/// when it has none (the caller drops NaN before a quantile). Each output pixel covers the same source area, so the
/// mean of the output equals the mean of the source when every source value is finite.
///
/// # Panics
///
/// If `src.len() != sw * sh` or a dimension is zero. Callers pass images they have just built.
#[must_use]
pub fn area_resize(src: &[f64], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f64> {
    assert_eq!(src.len(), sw * sh, "the source is not sw x sh");
    assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0, "a dimension is zero");
    let columns = footprints(sw, dw);
    let rows = footprints(sh, dh);
    let mut out = Vec::with_capacity(dw * dh);
    for row in &rows {
        for column in &columns {
            let (mut sum, mut weight) = (0.0, 0.0);
            for &(sy, wy) in row {
                for &(sx, wx) in column {
                    let v = src[sy * sw + sx];
                    if v.is_finite() {
                        sum += wx * wy * v;
                        weight += wx * wy;
                    }
                }
            }
            out.push(if weight > 0.0 { sum / weight } else { f64::NAN });
        }
    }
    out
}

/// For each of `dst` output cells along an axis of `src` source cells: the source cells it covers and the fraction of
/// each cell inside it (the fractions of one output cell sum to the scale `src / dst`).
fn footprints(src: usize, dst: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let start = d as f64 * scale;
            let end = (d + 1) as f64 * scale;
            let first = start.floor() as usize;
            let last = (end.ceil() as usize).min(src);
            (first..last)
                .filter_map(|s| {
                    let overlap = end.min((s + 1) as f64) - start.max(s as f64);
                    (overlap > 0.0).then_some((s, overlap))
                })
                .collect()
        })
        .collect()
}

/// The centred `new_w` x `new_h` window of an interleaved 8-bit RGB image of `w` x `h` (integer offsets, rounded
/// down). Asks for a window no larger than the image.
///
/// # Panics
///
/// If the window is larger than the image or the buffer is not `w * h * 3` bytes.
#[must_use]
pub fn central_crop(
    rgb: &[u8],
    w: usize,
    h: usize,
    new_w: usize,
    new_h: usize,
) -> (Vec<u8>, usize, usize) {
    assert_eq!(rgb.len(), w * h * 3, "the image is not w x h RGB");
    assert!(
        new_w >= 1 && new_h >= 1 && new_w <= w && new_h <= h,
        "the window is outside the image"
    );
    let (x0, y0) = ((w - new_w) / 2, (h - new_h) / 2);
    let mut out = Vec::with_capacity(new_w * new_h * 3);
    for y in y0..y0 + new_h {
        let start = (y * w + x0) * 3;
        out.extend_from_slice(&rgb[start..start + new_w * 3]);
    }
    (out, new_w, new_h)
}

/// The image turned a quarter turn clockwise: `w` x `h` becomes `h` x `w`.
///
/// # Panics
///
/// If the buffer is not `w * h * 3` bytes.
#[must_use]
pub fn quarter_turn(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    assert_eq!(rgb.len(), w * h * 3, "the image is not w x h RGB");
    let mut out = vec![0_u8; rgb.len()];
    for y in 0..h {
        for x in 0..w {
            // Source (x, y) lands at column h - 1 - y, row x of the turned image, which is h wide.
            let to = (x * h + (h - 1 - y)) * 3;
            let from = (y * w + x) * 3;
            out[to..to + 3].copy_from_slice(&rgb[from..from + 3]);
        }
    }
    (out, h, w)
}

/// The embedded JPEG in the orientation and the aspect ratio of the preview it is compared with.
///
/// A quarter turn when one of the two is portrait and the other landscape; then, when the aspect ratios differ by more
/// than `tolerance` (relative), a central crop to the preview's aspect ratio. Returns the pixels and their size.
///
/// # Panics
///
/// If the buffer is not `w * h * 3` bytes.
#[must_use]
pub fn fit_to_preview(
    rgb: &[u8],
    w: usize,
    h: usize,
    preview: (usize, usize),
    tolerance: f64,
) -> (Vec<u8>, usize, usize) {
    let (pw, ph) = preview;
    let (mut buffer, mut w, mut h) = if (w >= h) == (pw >= ph) {
        (rgb.to_vec(), w, h)
    } else {
        quarter_turn(rgb, w, h)
    };
    let (preview_aspect, image_aspect) = (pw as f64 / ph as f64, w as f64 / h as f64);
    if (image_aspect / preview_aspect - 1.0).abs() > tolerance {
        let (nw, nh) = if image_aspect > preview_aspect {
            (((h as f64) * preview_aspect).round() as usize, h)
        } else {
            (w, ((w as f64) / preview_aspect).round() as usize)
        };
        let (cropped, cw, ch) = central_crop(&buffer, w, h, nw.clamp(1, w), nh.clamp(1, h));
        buffer = cropped;
        (w, h) = (cw, ch);
    }
    (buffer, w, h)
}

/// The image at the common size: unchanged when it is already `cw` x `ch`, else area-averaged down to it.
#[must_use]
pub fn to_common(values: Vec<f64>, sw: usize, sh: usize, cw: usize, ch: usize) -> Vec<f64> {
    if (sw, sh) == (cw, ch) {
        values
    } else {
        area_resize(&values, sw, sh, cw, ch)
    }
}

/// The camera RGB of pixel `i` of an interleaved `f32` buffer, widened to `f64`; `None` when a channel is not finite.
///
/// # Panics
///
/// If the buffer holds fewer than `i + 1` pixels.
#[must_use]
pub fn rgb_of(buffer: &[f32], i: usize) -> Option<Vec3> {
    let p = &buffer[3 * i..3 * i + 3];
    let rgb = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
    rgb.iter().all(|v| v.is_finite()).then_some(rgb)
}

/// Linear luminance of an sRGB-encoded 8-bit RGB image (IEC 61966-2-1 decoding, then the Rec.709 weights), one value
/// per pixel.
#[must_use]
pub fn srgb_luminance(rgb: &[u8]) -> Vec<f64> {
    luminance_with(rgb, srgb_to_linear, SRGB_Y_ROW)
}

/// Linear luminance of an Adobe RGB-encoded 8-bit RGB image (the pure power 563/256, then the Adobe RGB weights).
#[must_use]
pub fn adobe_rgb_luminance(rgb: &[u8]) -> Vec<f64> {
    luminance_with(rgb, |code| code.powf(ADOBE_RGB_GAMMA), ADOBE_RGB_Y_ROW)
}

/// The luminance of the encoded values, as `tools/eval` takes it for the display contrast fit: the Rec.709 weights on
/// the code values `/ 255` (no decoding), one value per pixel.
#[must_use]
pub fn encoded_luma(rgb: &[u8]) -> Vec<f64> {
    rgb.chunks_exact(3)
        .map(|p| {
            SRGB_Y_ROW[0] * f64::from(p[0]) / 255.0
                + SRGB_Y_ROW[1] * f64::from(p[1]) / 255.0
                + SRGB_Y_ROW[2] * f64::from(p[2]) / 255.0
        })
        .collect()
}

/// Decodes each 8-bit channel with `decode` (through a table of the 256 codes, as the study does) and weights the
/// three linear channels.
fn luminance_with(rgb: &[u8], decode: impl Fn(f64) -> f64, row: [f64; 3]) -> Vec<f64> {
    let table: Vec<f64> = (0..=255_u16)
        .map(|v| decode(f64::from(v) / 255.0))
        .collect();
    rgb.chunks_exact(3)
        .map(|p| {
            row[0] * table[usize::from(p[0])]
                + row[1] * table[usize::from(p[1])]
                + row[2] * table[usize::from(p[2])]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_constant_image_stays_constant_at_any_size() {
        let src = vec![0.3; 7 * 5];
        for (dw, dh) in [(3, 2), (7, 5), (1, 1), (6, 4)] {
            let out = area_resize(&src, 7, 5, dw, dh);
            assert_eq!(out.len(), dw * dh);
            assert!(
                out.iter().all(|v| (v - 0.3).abs() < 1e-15),
                "{dw}x{dh}: {out:?}"
            );
        }
    }

    #[test]
    fn a_four_by_four_image_halves_to_the_means_of_its_blocks() {
        let src: Vec<f64> = (0..16).map(f64::from).collect();
        let out = area_resize(&src, 4, 4, 2, 2);
        // Block (0, 0) is 0, 1, 4, 5; block (1, 0) is 2, 3, 6, 7; and so on.
        assert_eq!(out, vec![2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn a_non_integer_ratio_conserves_the_mean() {
        let src: Vec<f64> = (0..35).map(|i| (f64::from(i) * 0.37).sin() + 2.0).collect();
        let mean_in = src.iter().sum::<f64>() / src.len() as f64;
        for (dw, dh) in [(2, 2), (3, 2), (4, 3)] {
            let out = area_resize(&src, 7, 5, dw, dh);
            let mean_out = out.iter().sum::<f64>() / out.len() as f64;
            assert!(
                (mean_out - mean_in).abs() < 1e-12,
                "{dw}x{dh}: {mean_out} vs {mean_in}"
            );
        }
    }

    #[test]
    fn a_fractional_pixel_is_weighted_by_its_overlap() {
        // Three source pixels 0, 3, 6 to one output pixel: the output covers all three equally, so the mean is 3.
        let out = area_resize(&[0.0, 3.0, 6.0], 3, 1, 1, 1);
        assert!((out[0] - 3.0).abs() < 1e-15);
        // Two source pixels (1, 4) to three output pixels, each covering 2/3 of a source pixel. The middle output
        // pixel takes 1/3 of each source pixel, so its value is (1 + 4) / 2.
        let out = area_resize(&[1.0, 4.0], 2, 1, 3, 1);
        assert!((out[0] - 1.0).abs() < 1e-15, "{out:?}");
        assert!((out[1] - 2.5).abs() < 1e-12, "{out:?}");
        assert!((out[2] - 4.0).abs() < 1e-15, "{out:?}");
    }

    #[test]
    fn non_finite_values_are_left_out_and_an_all_non_finite_block_is_nan() {
        let src = [f64::NAN, 2.0, 4.0, f64::NAN];
        let out = area_resize(&src, 2, 2, 1, 1);
        assert!((out[0] - 3.0).abs() < 1e-15);
        let out = area_resize(&[f64::NAN, f64::NAN], 2, 1, 1, 1);
        assert!(out[0].is_nan());
    }

    #[test]
    fn the_central_crop_takes_the_middle() {
        // A 4 x 2 image whose pixel (x, y) has red = x and green = y.
        let mut rgb = Vec::new();
        for y in 0..2u8 {
            for x in 0..4u8 {
                rgb.extend_from_slice(&[x, y, 0]);
            }
        }
        let (out, w, h) = central_crop(&rgb, 4, 2, 2, 2);
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, vec![1, 0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn a_landscape_jpeg_is_turned_for_a_portrait_preview() {
        let rgb: Vec<u8> = (0..4 * 2 * 3).map(|i| i as u8).collect();
        let (out, w, h) = fit_to_preview(&rgb, 4, 2, (1, 2), 0.02);
        assert_eq!((w, h), (2, 4));
        assert_eq!(out, quarter_turn(&rgb, 4, 2).0);
    }

    #[test]
    fn a_wider_jpeg_is_cropped_to_the_preview_aspect_and_a_close_one_is_kept() {
        let wide: Vec<u8> = (0..8 * 4 * 3).map(|i| i as u8).collect();
        let (out, w, h) = fit_to_preview(&wide, 8, 4, (5, 5), 0.02);
        assert_eq!((w, h), (4, 4));
        assert_eq!(
            out,
            central_crop(&wide, 8, 4, 4, 4).0,
            "the centre of the image"
        );
        // 160 x 108 (1.48) against a 150 x 100 preview (1.50): within 2 %, so nothing changes.
        let close = vec![7_u8; 160 * 108 * 3];
        let (_, cw, ch) = fit_to_preview(&close, 160, 108, (150, 100), 0.02);
        assert_eq!((cw, ch), (160, 108));
        // 160 x 100 (1.60): 7 % off, so the width is cropped to 150.
        let far = vec![7_u8; 160 * 100 * 3];
        let (_, fw, fh) = fit_to_preview(&far, 160, 100, (150, 100), 0.02);
        assert_eq!((fw, fh), (150, 100));
    }

    #[test]
    fn a_quarter_turn_moves_the_top_left_corner_to_the_top_right() {
        // 3 wide, 2 high: pixel (x, y) has red = x, green = y.
        let mut rgb = Vec::new();
        for y in 0..2u8 {
            for x in 0..3u8 {
                rgb.extend_from_slice(&[x, y, 0]);
            }
        }
        let (out, w, h) = quarter_turn(&rgb, 3, 2);
        assert_eq!((w, h), (2, 3));
        // The turned image is 2 wide: its top row is the original left column read bottom to top.
        assert_eq!(
            &out[0..3],
            &[0, 1, 0],
            "top-left of the turned image is the bottom-left source pixel"
        );
        assert_eq!(
            &out[3..6],
            &[0, 0, 0],
            "top-right is the top-left source pixel"
        );
        // Four quarter turns are the identity.
        let mut again = rgb.clone();
        let (mut dims_w, mut dims_h) = (3, 2);
        for _ in 0..4 {
            let (next, nw, nh) = quarter_turn(&again, dims_w, dims_h);
            again = next;
            dims_w = nw;
            dims_h = nh;
        }
        assert_eq!(again, rgb);
    }

    #[test]
    fn the_srgb_luminance_of_white_and_black_and_the_weights() {
        let y = srgb_luminance(&[255, 255, 255, 0, 0, 0]);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert_eq!(y[1], 0.0);
        // Half-grey code 128 decodes to about 0.2158 linear; the Rec.709 weights sum to 1.
        let grey = srgb_luminance(&[128, 128, 128])[0];
        assert!((grey - srgb_to_linear(128.0 / 255.0)).abs() < 1e-12);
        assert!((SRGB_Y_ROW.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn the_encoded_luma_is_the_weighted_code_values() {
        let y = encoded_luma(&[255, 0, 0, 0, 255, 0]);
        assert!((y[0] - 0.2126).abs() < 1e-15);
        assert!((y[1] - 0.7152).abs() < 1e-15);
    }
}
