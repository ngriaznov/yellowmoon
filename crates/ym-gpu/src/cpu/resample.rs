//! R9 single-pass resample that applies distortion and chromatic aberration together (`docs/05-raw-processing.md` section R9).
//!
//! One Catmull-Rom (bicubic, a = -0.5) resample from output pixel to source position, per colour
//! plane, using the positions of [`LensGeometry`]. The shared sampling helpers are public so other
//! stages (and the WGSL parity tests) use the same weights.
//!
//! Sampling convention: a plane's pixel `i` is centred at `i + 0.5`. For a position `s`,
//! `u = s - 0.5`, `i0 = floor(u)`, `t = u - i0`, and the taps are `i0 - 1 ..= i0 + 2` with weights
//! [`catmull_rom_weights`]`(t)`; indices beyond the plane replicate the edge pixel. Catmull-Rom has
//! negative lobes, so results can over- and undershoot; nothing is clamped.

use rayon::prelude::*;

use super::lens::LensGeometry;
use super::{CpuError, Placement, Rect, RgbImage, clamp_index};

/// The four Catmull-Rom weights for fractional position `t` in `[0, 1)`; they sum to 1.
#[inline(always)]
pub fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Bicubic (Catmull-Rom) sample of a `w x h` plane at continuous position `(sx, sy)` in the
/// plane's own coordinates. Edge pixels are replicated.
#[inline]
pub fn sample_catmull_rom(plane: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> f32 {
    let (ux, uy) = (sx - 0.5, sy - 0.5);
    let (fx, fy) = (ux.floor(), uy.floor());
    let (i0, j0) = (fx as isize, fy as isize);
    let wx = catmull_rom_weights((ux - fx) as f32);
    let wy = catmull_rom_weights((uy - fy) as f32);
    if i0 >= 1 && j0 >= 1 && (i0 as usize) + 2 < w && (j0 as usize) + 2 < h {
        // Interior: all sixteen taps exist, no clamping.
        let base = (j0 as usize - 1) * w + (i0 as usize - 1);
        let mut acc = 0.0f32;
        for (k, &wyk) in wy.iter().enumerate() {
            let row = &plane[base + k * w..base + k * w + 4];
            acc += wyk * (wx[0] * row[0] + wx[1] * row[1] + wx[2] * row[2] + wx[3] * row[3]);
        }
        return acc;
    }
    let xs: [usize; 4] = std::array::from_fn(|k| clamp_index(i0 - 1 + k as isize, w));
    let mut acc = 0.0f32;
    for (k, &wyk) in wy.iter().enumerate() {
        let row = &plane[clamp_index(j0 - 1 + k as isize, h) * w..][..w];
        let r = wx[0] * row[xs[0]] + wx[1] * row[xs[1]] + wx[2] * row[xs[2]] + wx[3] * row[xs[3]];
        acc += wyk * r;
    }
    acc
}

/// Resamples `src` (a whole frame or a tile of it, see its placement) into the rectangle `out` of
/// the cropped output of `geo`. The result has the size of `out` and the placement
/// `(out.x, out.y)` in a frame of the cropped size.
///
/// `src` must contain [`LensGeometry::source_rect`]`(out)`; samples beyond it replicate the edge
/// of the buffer (which equals the frame edge only if the buffer reaches it).
pub fn resample_lens(src: &RgbImage, geo: &LensGeometry, out: Rect) -> Result<RgbImage, CpuError> {
    let (cw, ch) = geo.output_size();
    if out.x1() > cw || out.y1() > ch {
        return Err(CpuError::RectOutside {
            rect: out,
            width: cw,
            height: ch,
        });
    }
    if (src.place.frame_width, src.place.frame_height) != (geo.frame_width(), geo.frame_height()) {
        return Err(CpuError::Mismatch(format!(
            "source frame {}x{} differs from the lens geometry's {}x{}",
            src.place.frame_width,
            src.place.frame_height,
            geo.frame_width(),
            geo.frame_height()
        )));
    }
    let mut dst = RgbImage::new(out.w, out.h);
    dst.place = Placement {
        x0: out.x,
        y0: out.y,
        frame_width: cw,
        frame_height: ch,
    };
    if out.is_empty() {
        return Ok(dst);
    }
    let n = dst.plane_len();
    let (ox, oy) = (src.place.x0 as f64, src.place.y0 as f64);
    let (sw, sh) = (src.width, src.height);
    let (r_plane, rest) = dst.data.split_at_mut(n);
    let (g_plane, b_plane) = rest.split_at_mut(n);
    let planes = [src.plane(0), src.plane(1), src.plane(2)];
    r_plane
        .par_chunks_mut(out.w)
        .zip(g_plane.par_chunks_mut(out.w))
        .zip(b_plane.par_chunks_mut(out.w))
        .enumerate()
        .for_each(|(j, ((rr, gg), bb))| {
            let py = (geo.crop().y + out.y + j) as f64 + 0.5;
            for i in 0..out.w {
                let px = (geo.crop().x + out.x + i) as f64 + 0.5;
                let pos = geo.source_positions(px, py);
                rr[i] = sample_catmull_rom(planes[0], sw, sh, pos[0].0 - ox, pos[0].1 - oy);
                gg[i] = sample_catmull_rom(planes[1], sw, sh, pos[1].0 - ox, pos[1].1 - oy);
                bb[i] = sample_catmull_rom(planes[2], sw, sh, pos[2].0 - ox, pos[2].1 - oy);
            }
        });
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_and_interpolate() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let w = catmull_rom_weights(t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert_eq!(catmull_rom_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
        let w1 = catmull_rom_weights(1.0 - 1e-7);
        assert!((w1[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn exact_on_linear_ramps_and_at_pixel_centres() {
        let (w, h) = (16, 12);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| 0.5 * (i % w) as f32 + 0.25 * (i / w) as f32)
            .collect();
        // Pixel centres return the pixel.
        assert!((sample_catmull_rom(&plane, w, h, 5.5, 4.5) - plane[4 * w + 5]).abs() < 1e-6);
        // A ramp is reproduced between pixels (away from the clamped border).
        let v = sample_catmull_rom(&plane, w, h, 7.3, 6.8);
        let want = 0.5 * (7.3 - 0.5) + 0.25 * (6.8 - 0.5);
        assert!((v - want as f32).abs() < 1e-4, "{v} vs {want}");
    }

    #[test]
    fn identity_geometry_returns_the_input() {
        let mut img = RgbImage::new(20, 14);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = ((i * 31) % 17) as f32 / 17.0;
        }
        let geo = LensGeometry::identity(20, 14);
        let out = resample_lens(&img, &geo, Rect::whole(20, 14)).unwrap();
        assert!(out.max_abs_diff(&img) < 1e-6);
    }

    #[test]
    fn rejects_out_of_range_and_mismatched_frames() {
        let img = RgbImage::new(20, 14);
        let geo = LensGeometry::identity(20, 14);
        assert!(resample_lens(&img, &geo, Rect::new(10, 0, 20, 14)).is_err());
        let geo2 = LensGeometry::identity(30, 14);
        assert!(resample_lens(&img, &geo2, Rect::whole(20, 14)).is_err());
    }
}
