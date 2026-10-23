//! The plane the straightener reads from a picture (`docs/20-auto-crop.md` §5), luma, and the
//! percentile rule it uses.

use crate::Picture;

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
    fn percentile_takes_the_lower_rank() {
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.99), 4.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
