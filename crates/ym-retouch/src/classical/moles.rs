//! The classical beauty-mark detector (`docs/27` S-07): dark, compact, brown, isolated blobs.
//!
//! The face crop is reduced to at most [`DETECT_MAX_SIDE_PX`] pixels and taken to log light.
//! Candidates are the local maxima in space and scale of a difference of Gaussians on log
//! luminance (dark blobs give a positive response) over scales covering blobs of 1.2 % to 4.5 % of
//! the face. Each candidate is measured against the median of its surroundings: its extent is the
//! connected region darker than half its depth; size, aspect, fill, contrast and colour are tested
//! (brown darkens red as well as green, a papule darkens green and leaves red); strongest first,
//! overlapping candidates are dropped and at most [`MAX_MARKS`] boxes are returned, padded by
//! [`BOX_PAD`]. Every constant is a starting value for S-12 (tuning).
//!
//! Known limit: the detector sees no skin mask, so compact dark non-skin features inside the face
//! rectangle (nostrils, pupils) can be boxed; that only punches harmless holes into a mask that
//! excludes them anyway, but they use up places of the cap, so very dark things are refused
//! ([`MAX_CONTRAST`]).

use rayon::prelude::*;

use super::grey::Grey;
use crate::blemish::area_downscale;
use crate::error::Result;
use crate::models::BeautyMarkDetector;
use crate::types::{Face, LinearAwg4Image, Rect};

/// The crop is reduced to this many pixels on its longer side (never enlarged).
pub const DETECT_MAX_SIDE_PX: f64 = 320.0;
/// Smallest mole diameter as a fraction of the face size (1.2 %).
pub const MOLE_MIN_FRAC: f64 = 0.012;
/// Largest mole diameter as a fraction of the face size (4.5 %).
pub const MOLE_MAX_FRAC: f64 = 0.045;
/// Relative tolerance on the size limits (the measured extent is half-depth, not exact).
pub const SIZE_TOLERANCE: f64 = 0.3;
/// Sigmas of the difference-of-Gaussians ladder as fractions of the face size.
pub const DOG_SIGMA_FRACS: [f64; 6] = [0.004, 0.0058, 0.0084, 0.0122, 0.0177, 0.0256];
/// Ratio of the outer to the inner sigma of each difference of Gaussians.
pub const DOG_RATIO: f64 = 1.6;
/// Smallest DoG response (natural log of linear luminance) of a candidate.
pub const DOG_MIN: f64 = 0.04;
/// Most candidates examined per face (strongest response first).
pub const MAX_CANDIDATES: usize = 64;
/// Smallest depth of a mole: luminance contrast with the local skin, in natural log of linear
/// light (0.4 is about 33 % darker; lighter brown marks are left to blemish removal).
pub const MIN_CONTRAST: f64 = 0.4;
/// Darkest accepted (0.9 is about 60 % darker); darker compact things are pupils or nostrils.
pub const MAX_CONTRAST: f64 = 1.4;
/// Smallest `ln R` drop (red must darken a little for brown).
pub const MIN_RED_DROP: f64 = 0.04;
/// Smallest ratio of the red drop to the green drop for a brown blob; acne is near 0.
pub const MIN_BROWN_RATIO: f64 = 0.35;
/// Largest aspect ratio (long side over short side of the extent).
pub const MAX_ASPECT: f64 = 1.6;
/// Smallest fraction of the extent's bounding box the region fills (an ellipse fills 0.785).
pub const MIN_FILL: f64 = 0.55;
/// Most boxes returned per face.
pub const MAX_MARKS: usize = 10;
/// Padding of each box on each side as a fraction of its size (30 % in total).
pub const BOX_PAD: f64 = 0.30;
/// Smoothing sigma in pixels applied to the channels before measuring a candidate.
pub const MEASURE_SIGMA_PX: f64 = 0.7;

/// The ratio of the red drop to the green drop of a blob (`ln R` and `ln G` against the local
/// skin; both negative for a dark blob): near 0 for a red papule, 0.5 and more for melanin. Zero
/// when green does not darken. Always finite.
pub fn brown_ratio(delta_ln_r: f64, delta_ln_g: f64) -> f64 {
    if !delta_ln_r.is_finite() || !delta_ln_g.is_finite() || delta_ln_g >= -1e-9 {
        return 0.0;
    }
    (delta_ln_r / delta_ln_g).clamp(-10.0, 10.0)
}

/// The classical [`BeautyMarkDetector`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassicalBeautyMarks;

struct Candidate {
    response: f64,
    x: usize,
    y: usize,
    level: usize,
}

struct Mark {
    contrast: f64,
    /// Padded box in map pixels: `(x0, y0, x1, y1)`, `x1` and `y1` exclusive.
    bbox: (f64, f64, f64, f64),
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

impl BeautyMarkDetector for ClassicalBeautyMarks {
    fn detect(&self, image: &LinearAwg4Image, face: &Face) -> Result<Vec<Rect>> {
        image.validate()?;
        let r = face.rect;
        let (iw, ih) = (i64::from(image.width_px), i64::from(image.height_px));
        let (x0, y0) = (i64::from(r.x_px).max(0), i64::from(r.y_px).max(0));
        let (x1, y1) = (r.right_px().min(iw), r.bottom_px().min(ih));
        if x1 - x0 < 16 || y1 - y0 < 16 {
            return Ok(Vec::new());
        }
        let (cw, ch) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut crop = Vec::with_capacity(cw * ch);
        for y in y0..y1 {
            let s = (y * iw + x0) as usize;
            crop.extend_from_slice(&image.data[s..s + cw]);
        }
        let face_px = f64::from(r.width_px.max(r.height_px)).max(1.0);
        let k = (DETECT_MAX_SIDE_PX / face_px.max(cw.max(ch) as f64)).min(1.0);
        let (w, h) = if k < 1.0 {
            (
                ((cw as f64 * k).round() as usize).max(1),
                ((ch as f64 * k).round() as usize).max(1),
            )
        } else {
            (cw, ch)
        };
        let small = if (w, h) == (cw, ch) {
            crop
        } else {
            area_downscale(&crop, (cw, ch), (w, h))
        };
        // Scale factors from map pixels back to the crop (and so to the image).
        let (sx, sy) = (cw as f64 / w as f64, ch as f64 / h as f64);
        let d = face_px * (w as f64 / cw as f64);
        let ln = |v: f32| f64::from(v).max(1e-4).ln();
        let plane = |f: &dyn Fn(&[f32; 3]) -> f64| Grey::new(w, h, small.iter().map(f).collect());
        let ly = plane(&|p| {
            ((f64::from(p[0]) + 2.0 * f64::from(p[1]) + f64::from(p[2])) / 4.0)
                .max(1e-4)
                .ln()
        });
        let lr = plane(&|p| ln(p[0]));
        let lg = plane(&|p| ln(p[1]));
        let (ly, lr, lg) = (
            ly.gaussian(MEASURE_SIGMA_PX),
            lr.gaussian(MEASURE_SIGMA_PX),
            lg.gaussian(MEASURE_SIGMA_PX),
        );

        // Scale space.
        let sigmas: Vec<f64> = DOG_SIGMA_FRACS.iter().map(|f| f * d).collect();
        let responses: Vec<Grey> = sigmas
            .par_iter()
            .map(|&s| {
                let inner = ly.gaussian(s);
                let outer = ly.gaussian(s * DOG_RATIO);
                Grey::new(
                    w,
                    h,
                    outer
                        .data
                        .iter()
                        .zip(&inner.data)
                        .map(|(o, i)| o - i)
                        .collect(),
                )
            })
            .collect();
        let mut cands = Vec::new();
        let n = responses.len();
        for (level, resp) in responses.iter().enumerate() {
            for y in 1..h.saturating_sub(1) {
                for x in 1..w.saturating_sub(1) {
                    let v = resp.data[y * w + x];
                    if v < DOG_MIN {
                        continue;
                    }
                    let mut is_max = true;
                    'nb: for dl in -1i32..=1 {
                        let l = level as i32 + dl;
                        if l < 0 || l >= n as i32 {
                            continue;
                        }
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                if (dl, dy, dx) == (0, 0, 0) {
                                    continue;
                                }
                                let o = responses[l as usize].data
                                    [(y as i32 + dy) as usize * w + (x as i32 + dx) as usize];
                                let earlier = (dl, dy, dx) < (0, 0, 0);
                                if o > v || (o == v && earlier) {
                                    is_max = false;
                                    break 'nb;
                                }
                            }
                        }
                    }
                    if is_max {
                        cands.push(Candidate {
                            response: v,
                            x,
                            y,
                            level,
                        });
                    }
                }
            }
        }
        cands.sort_by(|a, b| {
            b.response
                .total_cmp(&a.response)
                .then(a.y.cmp(&b.y))
                .then(a.x.cmp(&b.x))
                .then(a.level.cmp(&b.level))
        });
        cands.truncate(MAX_CANDIDATES);

        let marks: Vec<Mark> = cands
            .par_iter()
            .filter_map(|c| self.measure(c, &sigmas, d, &ly, &lr, &lg))
            .collect();
        let mut marks = marks;
        marks.sort_by(|a, b| {
            b.contrast
                .total_cmp(&a.contrast)
                .then(a.bbox.1.total_cmp(&b.bbox.1))
                .then(a.bbox.0.total_cmp(&b.bbox.0))
        });
        let mut kept: Vec<Mark> = Vec::new();
        for m in marks {
            let overlaps = kept.iter().any(|k| {
                m.bbox.0 < k.bbox.2
                    && k.bbox.0 < m.bbox.2
                    && m.bbox.1 < k.bbox.3
                    && k.bbox.1 < m.bbox.3
            });
            if !overlaps {
                kept.push(m);
            }
            if kept.len() == MAX_MARKS {
                break;
            }
        }
        Ok(kept
            .iter()
            .map(|m| {
                let fx0 = (m.bbox.0 * sx).floor() as i64 + x0;
                let fy0 = (m.bbox.1 * sy).floor() as i64 + y0;
                let fx1 = (m.bbox.2 * sx).ceil() as i64 + x0;
                let fy1 = (m.bbox.3 * sy).ceil() as i64 + y0;
                Rect::new(
                    fx0 as i32,
                    fy0 as i32,
                    (fx1 - fx0).max(1) as u32,
                    (fy1 - fy0).max(1) as u32,
                )
            })
            .collect())
    }
}

impl ClassicalBeautyMarks {
    /// Measures one candidate against the median of its surroundings and applies the tests.
    fn measure(
        &self,
        c: &Candidate,
        sigmas: &[f64],
        d: f64,
        ly: &Grey,
        lr: &Grey,
        lg: &Grey,
    ) -> Option<Mark> {
        let (w, h) = (ly.w as i64, ly.h as i64);
        let half = (6.4 * sigmas[c.level]).ceil() as i64;
        let (wx0, wy0) = ((c.x as i64 - half).max(0), (c.y as i64 - half).max(0));
        let (wx1, wy1) = (
            (c.x as i64 + half + 1).min(w),
            (c.y as i64 + half + 1).min(h),
        );
        let (ww, wh) = ((wx1 - wx0) as usize, (wy1 - wy0) as usize);
        let at =
            |g: &Grey, x: usize, y: usize| g.data[(wy0 as usize + y) * ly.w + wx0 as usize + x];
        let window = |g: &Grey| -> Vec<f64> {
            let mut v = Vec::with_capacity(ww * wh);
            for y in 0..wh {
                for x in 0..ww {
                    v.push(at(g, x, y));
                }
            }
            v
        };
        let (wy, wr, wg) = (window(ly), window(lr), window(lg));
        let bg_y = median(&mut wy.clone());
        let bg_r = median(&mut wr.clone());
        let bg_g = median(&mut wg.clone());
        let (cx, cy) = (c.x - wx0 as usize, c.y - wy0 as usize);
        let depth = bg_y - wy[cy * ww + cx];
        if depth < MIN_CONTRAST * 0.5 {
            return None;
        }
        // The extent: the region around the centre darker than half the depth.
        let dark: Vec<bool> = wy.iter().map(|&v| bg_y - v >= 0.5 * depth).collect();
        let mut region = vec![false; ww * wh];
        let mut stack = vec![cy * ww + cx];
        region[cy * ww + cx] = true;
        let (mut bx0, mut by0, mut bx1, mut by1) = (cx, cy, cx, cy);
        let mut area = 0usize;
        let mut touches_edge = false;
        while let Some(p) = stack.pop() {
            let (x, y) = (p % ww, p / ww);
            area += 1;
            bx0 = bx0.min(x);
            bx1 = bx1.max(x);
            by0 = by0.min(y);
            by1 = by1.max(y);
            if x == 0 || y == 0 || x + 1 == ww || y + 1 == wh {
                touches_edge = true;
            }
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < ww && ny < wh {
                    let q = ny * ww + nx;
                    if dark[q] && !region[q] {
                        region[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if touches_edge {
            return None;
        }
        let (bw, bh) = ((bx1 - bx0 + 1) as f64, (by1 - by0 + 1) as f64);
        let (long, short) = (bw.max(bh), bw.min(bh));
        if long > MOLE_MAX_FRAC * d * (1.0 + SIZE_TOLERANCE)
            || long < MOLE_MIN_FRAC * d * (1.0 - SIZE_TOLERANCE)
            || long / short > MAX_ASPECT
            || (area as f64) < MIN_FILL * bw * bh
        {
            return None;
        }
        let mean = |v: &[f64], bg: f64| {
            let s: f64 = region
                .iter()
                .zip(v)
                .filter(|(m, _)| **m)
                .map(|(_, x)| x - bg)
                .sum();
            s / area as f64
        };
        let (dy, dr, dg) = (mean(&wy, bg_y), mean(&wr, bg_r), mean(&wg, bg_g));
        let contrast = -dy;
        if !(MIN_CONTRAST..=MAX_CONTRAST).contains(&contrast)
            || dr > -MIN_RED_DROP
            || brown_ratio(dr, dg) < MIN_BROWN_RATIO
        {
            return None;
        }
        let (gx, gy) = (wx0 as f64 + bx0 as f64, wy0 as f64 + by0 as f64);
        let (px, py) = (bw * BOX_PAD * 0.5, bh * BOX_PAD * 0.5);
        Some(Mark {
            contrast,
            bbox: (gx - px, gy - py, gx + bw + px, gy + bh + py),
        })
    }
}

#[cfg(test)]
mod unit {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn brown_ratio_separates_pigments() {
        assert!(brown_ratio(-0.6, -0.9) > MIN_BROWN_RATIO);
        assert!(brown_ratio(-0.02, -0.3) < MIN_BROWN_RATIO);
        assert_eq!(brown_ratio(-0.1, 0.1), 0.0);
    }

    #[test]
    fn tiny_faces_give_nothing() {
        let img = LinearAwg4Image::filled(10, 10, [0.4, 0.3, 0.2]).unwrap();
        let face = Face {
            rect: Rect::new(0, 0, 10, 10),
            scale: 1.0,
        };
        assert!(ClassicalBeautyMarks.detect(&img, &face).unwrap().is_empty());
    }

    proptest! {
        #[test]
        fn ratio_is_finite(r in -5.0f64..5.0, g in -5.0f64..5.0) {
            let q = brown_ratio(r, g);
            prop_assert!(q.is_finite() && (-10.0..=10.0).contains(&q));
        }
    }
}
