//! A plain bilinear demosaic, used only to make the camera RGB that Lossless M and S files store.
//!
//! The M and S formats hold camera-processed RGB, not a mosaic; the synthetic set has only mosaics, so the writer
//! interpolates one. Every channel of a pixel is the mean of the pixels of that colour in its 3 x 3 neighbourhood
//! (the pixel itself when it has that colour), with coordinates clamped at the border.

use ym_raw::layout::{CfaColor, CfaPattern};

/// Interleaved RGB (`3 * width * height` values) from a mosaic. `pedestal_shift` is added to every value (the
/// camera RGB of M and S files sits on twice the sensor black level, so a mosaic with black `b` needs `+ b`).
pub fn bilinear_rgb(
    mosaic: &[u16],
    width: usize,
    height: usize,
    cfa: CfaPattern,
    pedestal_shift: i32,
) -> Vec<u16> {
    assert_eq!(mosaic.len(), width * height);
    let channel = |c: CfaColor| match c {
        CfaColor::Red => 0usize,
        CfaColor::Green => 1,
        CfaColor::Blue => 2,
    };
    let mut out = vec![0u16; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0u32; 3];
            let mut count = [0u32; 3];
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let yy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let xx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    let c = channel(cfa.color_at(xx, yy));
                    sum[c] += u32::from(mosaic[yy * width + xx]);
                    count[c] += 1;
                }
            }
            for c in 0..3 {
                // Every 3 x 3 window holds at least one pixel of each colour in a Bayer pattern.
                let mean = (sum[c] + count[c] / 2) / count[c].max(1);
                out[(y * width + x) * 3 + c] =
                    (mean as i64 + i64::from(pedestal_shift)).clamp(0, 65_535) as u16;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_color_stays_flat_for_every_pattern() {
        for cfa in [
            CfaPattern::RGGB,
            CfaPattern::GRBG,
            CfaPattern::GBRG,
            CfaPattern::BGGR,
        ] {
            let (w, h) = (8, 6);
            let mosaic: Vec<u16> = (0..w * h)
                .map(|i| match cfa.color_at(i % w, i / w) {
                    CfaColor::Red => 1000,
                    CfaColor::Green => 2000,
                    CfaColor::Blue => 3000,
                })
                .collect();
            let rgb = bilinear_rgb(&mosaic, w, h, cfa, 0);
            for px in rgb.chunks_exact(3) {
                assert_eq!(px, [1000, 2000, 3000], "{cfa:?}");
            }
            let shifted = bilinear_rgb(&mosaic, w, h, cfa, 512);
            assert_eq!(&shifted[..3], [1512, 2512, 3512]);
        }
    }

    #[test]
    fn a_horizontal_ramp_is_interpolated_linearly() {
        // Every channel of the mosaic is the same ramp 0, 100, 200, ...: bilinear interpolation of a linear ramp is exact
        // away from the border for the channels sampled at every other column.
        let (w, h) = (12, 4);
        let mosaic: Vec<u16> = (0..w * h).map(|i| ((i % w) * 100) as u16).collect();
        let rgb = bilinear_rgb(&mosaic, w, h, CfaPattern::RGGB, 0);
        for x in 2..w - 2 {
            let at = (h / 2 * w + x) * 3;
            for c in 0..3 {
                let v = rgb[at + c] as i64;
                assert!((v - (x * 100) as i64).abs() <= 50, "x {x} channel {c}: {v}");
            }
        }
    }

    #[test]
    fn values_are_clamped_to_sixteen_bits() {
        let mosaic = vec![65_535u16; 16];
        assert!(
            bilinear_rgb(&mosaic, 4, 4, CfaPattern::RGGB, 1000)
                .iter()
                .all(|&v| v == 65_535)
        );
        assert!(
            bilinear_rgb(
                &mosaic[..16].iter().map(|_| 5u16).collect::<Vec<_>>(),
                4,
                4,
                CfaPattern::RGGB,
                -100
            )
            .iter()
            .all(|&v| v == 0)
        );
    }
}
