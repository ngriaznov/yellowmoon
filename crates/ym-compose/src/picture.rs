//! The planes the analysis reads from a picture (`docs/20-auto-crop.md` §5 and §6): luma for the
//! straightener and the focus measure, CIELAB cells for the saliency, and the percentile rule they
//! share.

use crate::Picture;

/// Side of a map cell in picture pixels (`docs/20` §6: 4 × 4 boxes).
pub(crate) const CELL: usize = 4;

/// A plane of `f64` values, rows top to bottom.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plane {
    /// Width in samples.
    pub width: usize,
    /// Height in samples.
    pub height: usize,
    /// `width × height` values, row-major.
    pub data: Vec<f64>,
}

impl Plane {
    /// A plane of zeros.
    pub(crate) fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// The value at column `x`, row `y`.
    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }
}

/// Luma `Y = 0.2126 R' + 0.7152 G' + 0.0722 B'` of the 8-bit values over 255, gamma-encoded
/// (`docs/20` §5).
pub(crate) fn luma(picture: &Picture<'_>) -> Plane {
    let (w, h) = (picture.width(), picture.height());
    let data = picture
        .rgb()
        .chunks_exact(3)
        .map(|p| {
            (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0
        })
        .collect();
    Plane {
        width: w,
        height: h,
        data,
    }
}

/// The sRGB transfer function decoded for the 256 code values.
fn decode_table() -> [f64; 256] {
    let mut table = [0.0; 256];
    for (code, out) in table.iter_mut().enumerate() {
        let v = code as f64 / 255.0;
        *out = if v <= 0.040_45 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
    }
    table
}

/// CIELAB's companding function.
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// The picture on the map of `docs/20` §6 in CIELAB (D65): `[L, a, b]`, each
/// `⌊W / 4⌋ × ⌊H / 4⌋` cells. Each cell averages its 4 × 4 pixels in linear light (sRGB decoded)
/// before the conversion; the remainder at the right and bottom is dropped.
pub(crate) fn lab_cells(picture: &Picture<'_>) -> [Plane; 3] {
    let (w, h) = (picture.width(), picture.height());
    let (mw, mh) = (w / CELL, h / CELL);
    let decode = decode_table();
    let rgb = picture.rgb();
    let mut planes = [
        Plane::zeros(mw, mh),
        Plane::zeros(mw, mh),
        Plane::zeros(mw, mh),
    ];
    let n = (CELL * CELL) as f64;
    for cy in 0..mh {
        for cx in 0..mw {
            let mut sum = [0.0; 3];
            for y in cy * CELL..(cy + 1) * CELL {
                for x in cx * CELL..(cx + 1) * CELL {
                    let i = (y * w + x) * 3;
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += decode[usize::from(rgb[i + c])];
                    }
                }
            }
            let [r, g, b] = sum.map(|s| s / n);
            // Linear sRGB to XYZ (D65), over the white of the same matrix.
            let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
            let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
            let z = (0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b) / 1.088_83;
            let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
            let i = cy * mw + cx;
            planes[0].data[i] = 116.0 * fy - 16.0;
            planes[1].data[i] = 500.0 * (fx - fy);
            planes[2].data[i] = 200.0 * (fy - fz);
        }
    }
    planes
}

/// The `p` quantile of `values` (`0 ≤ p ≤ 1`): the value at index `⌊p (n − 1)⌋` of the sorted
/// values, 0 for no values. Values must not be NaN.
pub(crate) fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    let k = ((v.len() - 1) as f64 * p.clamp(0.0, 1.0)).floor() as usize;
    let (_, nth, _) = v.select_nth_unstable_by(k, f64::total_cmp);
    *nth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_weights_the_channels_by_rec_709() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let p = Picture::new(4, 1, &rgb).unwrap();
        let y = luma(&p);
        assert!((y.data[0] - 0.2126).abs() < 1e-12);
        assert!((y.data[1] - 0.7152).abs() < 1e-12);
        assert!((y.data[2] - 0.0722).abs() < 1e-12);
        assert!((y.data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lab_of_white_grey_and_red() {
        // 8 x 4 pixels: two cells, white on the left, sRGB red on the right; the remainder of a
        // 9th column would be dropped.
        let mut rgb = Vec::new();
        for _ in 0..4 {
            for x in 0..9 {
                rgb.extend_from_slice(if x < 4 {
                    &[255, 255, 255]
                } else {
                    &[255, 0, 0]
                });
            }
        }
        let p = Picture::new(9, 4, &rgb).unwrap();
        let [l, a, b] = lab_cells(&p);
        assert_eq!((l.width, l.height), (2, 1));
        assert!(
            (l.data[0] - 100.0).abs() < 1e-3 && a.data[0].abs() < 1e-3 && b.data[0].abs() < 1e-3
        );
        // sRGB red is about L 53.24, a 80.09, b 67.20.
        assert!((l.data[1] - 53.24).abs() < 0.05, "{}", l.data[1]);
        assert!((a.data[1] - 80.09).abs() < 0.1, "{}", a.data[1]);
        assert!((b.data[1] - 67.20).abs() < 0.1, "{}", b.data[1]);
    }

    #[test]
    fn percentile_takes_the_lower_rank() {
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.99), 4.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
