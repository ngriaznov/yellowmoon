//! Rotated square regions of a picture and their resampling into a network input (`docs/27` S-09, S-10).
//!
//! MediaPipe feeds Face Mesh a square crop of the face turned so that the line between the eyes is level, and this module
//! makes the same crop: a [`Roi`] (centre, side and rotation in picture pixels), [`crop_tensor`] (the `N x N x 3` input) and
//! the map from crop pixels back to the picture ([`Roi::to_picture`]). The rotation follows MediaPipe's
//! `DetectionsToRectsCalculator`: `rotation = atan2(dy, dx)` of the vector between the two eye points (picture pixels, y
//! down), so that crop x runs along the eye line; a crop pixel `(u, v)` in `0..1` is the picture point
//! `centre + R(rotation) * ((u - 0.5) * side, (v - 0.5) * side)`.
//!
//! The sampling is bilinear at `i / N` for crop pixel `i` (OpenCV's `warpPerspective` of the four corners, no half-pixel
//! shift, which is what MediaPipe's `ImageToTensor` does), black outside the picture. [`Resample::Box`] first reduces the
//! part of the picture the crop covers by whole blocks of `k = floor(0.5 / scale)` pixels (`scale` is crop pixels per
//! picture pixel), so a crop of a 24 MP photo is not aliased; with `k = 1` it is exactly [`Resample::MediaPipe`].

use rayon::prelude::*;

use crate::mediapipe::blazeface::{PixelSource, Resample};
use crate::model::{MlError, Result};

/// A square region of the picture, rotated about its centre. All values are in picture pixels (y down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roi {
    /// The centre.
    pub center_px: [f64; 2],
    /// The side of the square.
    pub side_px: f64,
    /// The rotation in radians: the crop's x axis points along `(cos, sin)` in the picture.
    pub rotation_rad: f64,
}

impl Roi {
    /// The picture point of the crop position `(u, v)`, each `0..1` over the square (`0.5, 0.5` is the centre).
    pub fn to_picture(&self, u: f64, v: f64) -> [f64; 2] {
        let (s, c) = self.rotation_rad.sin_cos();
        let (dx, dy) = ((u - 0.5) * self.side_px, (v - 0.5) * self.side_px);
        [
            self.center_px[0] + dx * c - dy * s,
            self.center_px[1] + dx * s + dy * c,
        ]
    }

    /// The crop position `(u, v)` of a picture point (the inverse of [`Roi::to_picture`]).
    pub fn from_picture(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.rotation_rad.sin_cos();
        let (dx, dy) = (p[0] - self.center_px[0], p[1] - self.center_px[1]);
        [
            (dx * c + dy * s) / self.side_px + 0.5,
            (-dx * s + dy * c) / self.side_px + 0.5,
        ]
    }

    /// The same region with the side multiplied by `factor`.
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            side_px: self.side_px * factor,
            ..*self
        }
    }

    /// The rectangle `(x0, y0, x1, y1)` that holds the rotated square.
    pub fn bounds(&self) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            let p = self.to_picture(u, v);
            b = [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ];
        }
        b
    }
}

/// The rotation that makes the line from `start` to `end` level in the crop: `atan2(dy, dx)` (picture pixels, y down),
/// MediaPipe's `DetectionsToRects` with a target angle of 0, normalised to `-pi..pi`.
pub fn level_rotation(start: [f64; 2], end: [f64; 2]) -> f64 {
    (end[1] - start[1]).atan2(end[0] - start[0])
}

/// A picture reduced by whole blocks, with the offset to read it back (see the module documentation).
struct Reduced {
    x0: i64,
    y0: i64,
    k: usize,
    w: usize,
    h: usize,
    data: Vec<[f32; 3]>,
}

impl Reduced {
    fn new(src: &dyn PixelSource, roi: &Roi, k: usize) -> Self {
        let b = roi.bounds();
        let (pw, ph) = (src.width() as i64, src.height() as i64);
        let margin = (k + 1) as i64;
        let x0 = ((b[0].floor() as i64) - margin).clamp(0, pw);
        let y0 = ((b[1].floor() as i64) - margin).clamp(0, ph);
        let x1 = ((b[2].ceil() as i64) + margin).clamp(0, pw);
        let y1 = ((b[3].ceil() as i64) + margin).clamp(0, ph);
        let (rw, rh) = (
            ((x1 - x0).max(1) as usize).div_ceil(k),
            ((y1 - y0).max(1) as usize).div_ceil(k),
        );
        let mut data = vec![[0.0f32; 3]; rw * rh];
        if x1 > x0 && y1 > y0 {
            data.par_chunks_mut(rw).enumerate().for_each(|(j, row)| {
                let ya = y0 + (j * k) as i64;
                let yb = (ya + k as i64).min(y1);
                for (i, out) in row.iter_mut().enumerate() {
                    let xa = x0 + (i * k) as i64;
                    let xb = (xa + k as i64).min(x1);
                    let mut acc = [0.0f32; 3];
                    for y in ya..yb {
                        let mut line = [0.0f32; 3];
                        for x in xa..xb {
                            let p = src.rgb(x as usize, y as usize);
                            line[0] += p[0];
                            line[1] += p[1];
                            line[2] += p[2];
                        }
                        for c in 0..3 {
                            acc[c] += line[c];
                        }
                    }
                    let n = ((xb - xa) * (yb - ya)) as f32;
                    *out = [acc[0] / n, acc[1] / n, acc[2] / n];
                }
            });
        }
        Self {
            x0,
            y0,
            k,
            w: rw,
            h: rh,
            data,
        }
    }

    /// Bilinear sample at the picture coordinate `(sx, sy)` (pixel centres at integers), black beyond one pixel outside.
    fn sample(&self, sx: f64, sy: f64, pw: usize, ph: usize) -> [f32; 3] {
        let (fw, fh) = (pw as f64, ph as f64);
        if sx <= -1.0 || sy <= -1.0 || sx >= fw || sy >= fh {
            return [0.0; 3];
        }
        // The centre of a block of k pixels starting at x0 + i * k is at x0 + i * k + (k - 1) / 2.
        let half = (self.k as f64 - 1.0) / 2.0;
        let rx = (sx - self.x0 as f64 - half) / self.k as f64;
        let ry = (sy - self.y0 as f64 - half) / self.k as f64;
        let (fx, fy) = (rx.floor(), ry.floor());
        let (tx, ty) = ((rx - fx) as f32, (ry - fy) as f32);
        let at = |x: i64, y: i64| -> [f32; 3] {
            let x = x.clamp(0, self.w as i64 - 1) as usize;
            let y = y.clamp(0, self.h as i64 - 1) as usize;
            self.data[y * self.w + x]
        };
        let (x0, y0) = (fx as i64, fy as i64);
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1, y0),
            at(x0, y0 + 1),
            at(x0 + 1, y0 + 1),
        );
        // The picture's own edge blends to black over one pixel, as OpenCV's constant border does.
        let edge = |s: f64, len: f64| -> f32 {
            if s < 0.0 {
                (s + 1.0) as f32
            } else if s > len - 1.0 {
                (len - s) as f32
            } else {
                1.0
            }
        };
        let cover = edge(sx, fw) * edge(sy, fh);
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let top = a[i] * (1.0 - tx) + b[i] * tx;
            let bottom = c[i] * (1.0 - tx) + d[i] * tx;
            out[i] = (top * (1.0 - ty) + bottom * ty) * cover;
        }
        out
    }
}

/// The `out_px x out_px x 3` input of a network for `roi`: row-major, channels last, each value `lo + v * (hi - lo)` for the
/// picture value `v` in `0..1`.
///
/// # Errors
///
/// [`MlError::Input`] when the region is empty or not finite, or `out_px` is 0.
pub fn crop_tensor(
    src: &dyn PixelSource,
    roi: &Roi,
    out_px: usize,
    resample: Resample,
    range: (f32, f32),
) -> Result<Vec<f32>> {
    let ok = roi.side_px.is_finite()
        && roi.side_px > 0.0
        && roi.center_px.iter().all(|v| v.is_finite())
        && roi.rotation_rad.is_finite()
        && out_px > 0;
    if !ok {
        return Err(MlError::Input(format!("region {roi:?} into {out_px} px")));
    }
    let scale = out_px as f64 / roi.side_px;
    let k = match resample {
        Resample::MediaPipe => 1,
        Resample::Box => ((0.5 / scale).floor() as usize).max(1),
    };
    let reduced = Reduced::new(src, roi, k);
    let (pw, ph) = (src.width(), src.height());
    let (lo, hi) = range;
    let mut out = vec![0.0f32; out_px * out_px * 3];
    out.par_chunks_mut(out_px * 3)
        .enumerate()
        .for_each(|(j, row)| {
            for (i, px) in row.chunks_exact_mut(3).enumerate() {
                let p = roi.to_picture(i as f64 / out_px as f64, j as f64 / out_px as f64);
                for (o, v) in px.iter_mut().zip(reduced.sample(p[0], p[1], pw, ph)) {
                    *o = lo + v * (hi - lo);
                }
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mediapipe::blazeface::RgbF32;

    fn picture(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        v
    }

    #[test]
    fn the_map_to_the_picture_and_back_agree_and_level_the_eyes() {
        let roi = Roi {
            center_px: [100.0, 80.0],
            side_px: 60.0,
            rotation_rad: 0.4,
        };
        let p = roi.to_picture(0.25, 0.75);
        let q = roi.from_picture(p);
        assert!((q[0] - 0.25).abs() < 1e-12 && (q[1] - 0.75).abs() < 1e-12);
        assert_eq!(roi.to_picture(0.5, 0.5), [100.0, 80.0]);
        // Two eyes on a line tilted by 20 degrees land on a level line in the crop.
        let a = 20.0f64.to_radians();
        let (e0, e1) = ([90.0, 70.0], [90.0 + 30.0 * a.cos(), 70.0 + 30.0 * a.sin()]);
        let roi = Roi {
            center_px: [100.0, 80.0],
            side_px: 100.0,
            rotation_rad: level_rotation(e0, e1),
        };
        let (c0, c1) = (roi.from_picture(e0), roi.from_picture(e1));
        assert!((c0[1] - c1[1]).abs() < 1e-12, "{c0:?} {c1:?}");
        assert!(c1[0] > c0[0]);
        assert!((roi.scaled(2.0).side_px - 200.0).abs() < 1e-12);
        let b = roi.bounds();
        assert!(b[0] < 100.0 && b[2] > 100.0 && b[1] < 80.0 && b[3] > 80.0);
    }

    #[test]
    fn an_unrotated_crop_reads_the_picture_at_i_over_n() {
        // A ramp in x: value = x / 63. Crop the whole 64 x 64 picture into 32 x 32: crop pixel i reads x = 2 i.
        let data = picture(64, 64, |x, _| [x as f32 / 63.0; 3]);
        let src = RgbF32::new(64, 64, &data).expect("picture");
        let roi = Roi {
            center_px: [32.0, 32.0],
            side_px: 64.0,
            rotation_rad: 0.0,
        };
        let t = crop_tensor(&src, &roi, 32, Resample::MediaPipe, (0.0, 1.0)).expect("crop");
        for i in 0..32 {
            let want = (2 * i) as f32 / 63.0;
            assert!(
                (t[i * 3] - want).abs() < 1e-6,
                "i {i}: {} against {want}",
                t[i * 3]
            );
        }
        // The range maps 0..1 to -1..1.
        let t2 = crop_tensor(&src, &roi, 32, Resample::MediaPipe, (-1.0, 1.0)).expect("crop");
        assert!((t2[3] - (2.0 * t[3] - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn outside_the_picture_is_black_and_a_half_turn_flips_the_crop() {
        let data = picture(32, 32, |_, _| [0.8; 3]);
        let src = RgbF32::new(32, 32, &data).expect("picture");
        // A crop centred on the corner: three quarters of it is outside.
        let roi = Roi {
            center_px: [0.0, 0.0],
            side_px: 32.0,
            rotation_rad: 0.0,
        };
        let t = crop_tensor(&src, &roi, 16, Resample::MediaPipe, (0.0, 1.0)).expect("crop");
        assert_eq!(t[0], 0.0);
        let at = |x: usize, y: usize| t[(y * 16 + x) * 3];
        assert!((at(12, 12) - 0.8).abs() < 1e-6 && at(2, 12) == 0.0 && at(12, 2) == 0.0);
        // A half turn about the centre of an asymmetric picture reverses it.
        let ramp = picture(32, 32, |x, y| [x as f32 / 31.0, y as f32 / 31.0, 0.0]);
        let src = RgbF32::new(32, 32, &ramp).expect("picture");
        let mut roi = Roi {
            center_px: [16.0, 16.0],
            side_px: 32.0,
            rotation_rad: 0.0,
        };
        let a = crop_tensor(&src, &roi, 8, Resample::MediaPipe, (0.0, 1.0)).expect("crop");
        roi.rotation_rad = std::f64::consts::PI;
        let b = crop_tensor(&src, &roi, 8, Resample::MediaPipe, (0.0, 1.0)).expect("crop");
        // The middle row (j = 4 reads y = 16 either way): crop pixel i reads x = 4 i (unrotated) or 32 - 4 i (turned; pixel 0
        // is at the picture's edge, black). Red is x / 31.
        let red = |t: &[f32], i: usize| t[(4 * 8 + i) * 3];
        assert!(red(&a, 0) < 0.01 && red(&a, 1) > red(&a, 0) && red(&a, 2) > red(&a, 1));
        assert!(
            red(&b, 1) > 0.85 && red(&b, 2) < red(&b, 1) && red(&b, 3) < red(&b, 2),
            "{:?}",
            &b[96..120]
        );
    }

    #[test]
    fn box_filtering_removes_aliasing_and_a_large_picture_keeps_its_mean() {
        // A checkerboard of single pixels averages to grey under the box filter; plain sampling would read black or white.
        let data = picture(256, 256, |x, y| [((x + y) % 2) as f32; 3]);
        let src = RgbF32::new(256, 256, &data).expect("picture");
        let roi = Roi {
            center_px: [128.0, 128.0],
            side_px: 256.0,
            rotation_rad: 0.3,
        };
        let t = crop_tensor(&src, &roi, 16, Resample::Box, (0.0, 1.0)).expect("crop");
        // Away from the corners (outside the picture once turned) every value is close to 0.5.
        let centre = &t[(8 * 16 + 8) * 3..][..3];
        assert!((centre[0] - 0.5).abs() < 0.15, "{centre:?}");
        assert!(
            crop_tensor(
                &src,
                &Roi {
                    side_px: 0.0,
                    ..roi
                },
                16,
                Resample::Box,
                (0.0, 1.0)
            )
            .is_err()
        );
        assert!(crop_tensor(&src, &roi, 0, Resample::Box, (0.0, 1.0)).is_err());
    }
}
