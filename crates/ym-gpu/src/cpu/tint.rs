//! The mask view's overlay on the CPU (`docs/23-nodes.md` section 5.8): the reference of the device's tint
//! kernel (`crate::kernels::tint`).
//!
//! `out = display + 0.55 m (accent - display)` on display-linear pixels, after output sharpening and before
//! C13. The arithmetic of every pixel is `ym_color::mask::overlay_pixel`, in `f64`; the pixels are `f32`.

use ym_color::consts::Vec3;
use ym_color::mask::overlay_pixel;

use super::RgbImage;

/// Tints interleaved display-linear pixels (`r, g, b, r, g, b, ...`) in place with the mask `m` (one value for
/// each pixel) and `accent` in the same primaries.
///
/// # Panics
///
/// Never; a mask shorter than the pixels tints the pixels it has a value for.
pub fn overlay_interleaved(rgb: &mut [f32], m: &[f32], accent: Vec3) {
    for (px, &m) in rgb.chunks_exact_mut(3).zip(m) {
        let out = overlay_pixel(
            [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])],
            f64::from(m),
            accent,
        );
        px[0] = out[0] as f32;
        px[1] = out[1] as f32;
        px[2] = out[2] as f32;
    }
}

/// Tints the pixels of a planar image in place (the first three planes of `image.data` are the colour); `m` has
/// one value for each pixel.
pub fn overlay_planar(image: &mut RgbImage, m: &[f32], accent: Vec3) {
    let n = image.width * image.height;
    for (i, &m) in m.iter().enumerate().take(n) {
        let out = overlay_pixel(
            [
                f64::from(image.data[i]),
                f64::from(image.data[n + i]),
                f64::from(image.data[2 * n + i]),
            ],
            f64::from(m),
            accent,
        );
        image.data[i] = out[0] as f32;
        image.data[n + i] = out[1] as f32;
        image.data[2 * n + i] = out[2] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's weight (`docs/23` section 5.8), written out again here.
    const OVERLAY_WEIGHT: f64 = 0.55;

    #[test]
    fn the_overlay_is_the_formula() {
        let accent = [1.0, 0.159, 0.6445];
        let mut px = [0.2f32, 0.5, 0.8, 0.0, 0.0, 0.0, 2.0, 1.0, 0.1];
        let m = [1.0f32, 0.5, 0.0];
        let before = px;
        overlay_interleaved(&mut px, &m, accent);
        for (i, &m) in m.iter().enumerate() {
            for c in 0..3 {
                let d = f64::from(before[3 * i + c]);
                let want = d + OVERLAY_WEIGHT * f64::from(m) * (accent[c] - d);
                assert!((f64::from(px[3 * i + c]) - want).abs() < 1e-6, "{i} {c}");
            }
        }
        // m = 0 leaves the pixel as it was, to the bit.
        assert_eq!(&px[6..9], &before[6..9]);
    }

    #[test]
    fn planar_and_interleaved_agree() {
        let accent = [0.9, 0.2, 0.6];
        let (w, h) = (3, 2);
        let n = w * h;
        let mut planar = RgbImage::new(w, h);
        let mut inter = vec![0.0f32; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                let v = 0.1 * (i as f32 + 1.0) * (c as f32 + 1.0);
                planar.data[c * n + i] = v;
                inter[3 * i + c] = v;
            }
        }
        let m: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        overlay_planar(&mut planar, &m, accent);
        overlay_interleaved(&mut inter, &m, accent);
        for i in 0..n {
            for c in 0..3 {
                assert_eq!(planar.data[c * n + i], inter[3 * i + c]);
            }
        }
    }
}
