//! The feather of `docs/27` section 5.1.
//!
//! Every pass writes disjoint rows from read-only inputs, so the result does not depend on the
//! number of rayon threads.

use rayon::prelude::*;

use crate::error::{Result, RetouchError};
use crate::types::Mask;

/// Largest working mask (pixels) the feather allocates.
const MAX_WORK_PIXELS: u64 = 1 << 27;

/// Feathers a mask: a positive `r_px` grows it and softens it outward, a negative one shrinks it
/// and softens it inward (`docs/27` section 5.1).
///
/// The mask is resized by `f = max(512 / width, 512 / height)`, dilated by a disc of radius
/// `trunc(r_px * f)` (after inverting for a negative radius), blurred with a Gaussian of side
/// `6r + 1` that reflects at the borders (OpenCV's reflect-101), and resized back. A radius that
/// truncates to 0 returns the mask unchanged. The result never goes below the input for a
/// positive radius, nor above it for a negative one (a resize or the blur of a thin mask could
/// otherwise break that).
pub fn feather(mask: &Mask, r_px: f64) -> Result<Mask> {
    mask.validate()?;
    let (w, h) = (mask.width_px as usize, mask.height_px as usize);
    let f = (512.0 / w as f64).max(512.0 / h as f64);
    let scaled = if r_px.is_finite() {
        (r_px * f).trunc()
    } else {
        0.0
    };
    if scaled == 0.0 {
        return Ok(mask.clone());
    }
    let (sw, sh) = (
        ((w as f64 * f).round() as usize).max(1),
        ((h as f64 * f).round() as usize).max(1),
    );
    if (sw as u64) * (sh as u64) > MAX_WORK_PIXELS {
        return Err(RetouchError::TooLarge {
            width: sw as u64,
            height: sh as u64,
        });
    }
    let inverted = scaled < 0.0;
    // A disc larger than the working mask changes nothing more; capping bounds the work.
    let r = (scaled.abs().min(sw.max(sh) as f64)) as usize;

    let mut work = resize_bilinear(&mask.data, w, h, sw, sh);
    if inverted {
        invert(&mut work);
    }
    let work = dilate_disc(&work, sw, sh, r);
    let mut work = gaussian(&work, sw, sh, r);
    if inverted {
        invert(&mut work);
    }
    let mut out = resize_bilinear(&work, sw, sh, w, h);
    out.par_iter_mut()
        .zip(mask.data.par_iter())
        .for_each(|(o, &m)| {
            *o = if inverted { (*o).min(m) } else { (*o).max(m) };
        });
    Ok(Mask {
        width_px: mask.width_px,
        height_px: mask.height_px,
        data: out,
    })
}

fn invert(data: &mut [u8]) {
    data.par_iter_mut().for_each(|v| *v = 255 - *v);
}

/// Bilinear resize with pixel-centre alignment, rounded to the nearest byte.
fn resize_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    let axis = |dn: usize, sn: usize| -> Vec<(usize, usize, f64)> {
        (0..dn)
            .map(|d| {
                let s =
                    ((d as f64 + 0.5) * sn as f64 / dn as f64 - 0.5).clamp(0.0, (sn - 1) as f64);
                let i0 = s.floor() as usize;
                (i0, (i0 + 1).min(sn - 1), s - i0 as f64)
            })
            .collect()
    };
    let xs = axis(dw, sw);
    let ys = axis(dh, sh);
    let mut out = vec![0u8; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = ys[y];
        let (r0, r1) = (&src[y0 * sw..(y0 + 1) * sw], &src[y1 * sw..(y1 + 1) * sw]);
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1, fx) = xs[x];
            let top = f64::from(r0[x0]) * (1.0 - fx) + f64::from(r0[x1]) * fx;
            let bottom = f64::from(r1[x0]) * (1.0 - fx) + f64::from(r1[x1]) * fx;
            *o = (top * (1.0 - fy) + bottom * fy + 0.5)
                .floor()
                .clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// Grayscale dilation with a disc: `out(p) = max of src over pixels within r of p` (pixels outside
/// the image do not count).
fn dilate_disc(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return src.to_vec();
    }
    // Half-width of the disc on each row offset dy.
    let half_width = |dy: usize| -> usize {
        let target = r * r - dy * dy;
        let mut k = (target as f64).sqrt() as usize;
        while k * k > target {
            k -= 1;
        }
        while (k + 1) * (k + 1) <= target {
            k += 1;
        }
        k
    };
    let mut out = vec![0u8; w * h];
    // `horizontal` holds, for the current k, the maximum over x-k..=x+k of each row.
    let mut horizontal = src.to_vec();
    for k in 0..=r {
        if k > 0 {
            horizontal
                .par_chunks_mut(w)
                .zip(src.par_chunks(w))
                .for_each(|(hrow, srow)| {
                    for x in 0..w {
                        let mut v = hrow[x];
                        if x >= k {
                            v = v.max(srow[x - k]);
                        }
                        if x + k < w {
                            v = v.max(srow[x + k]);
                        }
                        hrow[x] = v;
                    }
                });
        }
        let offsets: Vec<isize> = (0..=r)
            .filter(|&dy| half_width(dy) == k)
            .flat_map(|dy| {
                if dy == 0 {
                    vec![0]
                } else {
                    vec![dy as isize, -(dy as isize)]
                }
            })
            .collect();
        if offsets.is_empty() {
            continue;
        }
        let horizontal = &horizontal;
        out.par_chunks_mut(w).enumerate().for_each(|(y, orow)| {
            for &dy in &offsets {
                let yy = y as isize + dy;
                if yy < 0 || yy >= h as isize {
                    continue;
                }
                let hrow = &horizontal[yy as usize * w..(yy as usize + 1) * w];
                for (o, &v) in orow.iter_mut().zip(hrow) {
                    *o = (*o).max(v);
                }
            }
        });
    }
    out
}

/// Index reflected at the borders without repeating the edge sample (`gfedcb|abcdefgh|gfedcba`).
fn reflect101(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let m = i.rem_euclid(period);
    (if m >= n as isize { period - m } else { m }) as usize
}

/// The Gaussian of section 5.1 step 4: a square kernel of side `k = 6r + ((6r) & 1) + 1`,
/// `sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8`, as two separable passes.
fn gaussian(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let k = 6 * r + ((6 * r) & 1) + 1;
    let sigma = 0.3 * ((k as f64 - 1.0) * 0.5 - 1.0) + 0.8;
    let c = (k - 1) / 2;
    let mut weights: Vec<f64> = (0..k)
        .map(|i| {
            let d = i as f64 - c as f64;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|v| *v /= total);

    let xmap: Vec<usize> = (0..w + 2 * c)
        .map(|i| reflect101(i as isize - c as isize, w))
        .collect();
    let mut horizontal = vec![0f64; w * h];
    horizontal
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let row = &src[y * w..(y + 1) * w];
            let padded: Vec<f64> = xmap.iter().map(|&x| f64::from(row[x])).collect();
            for (x, o) in out_row.iter_mut().enumerate() {
                *o = weights
                    .iter()
                    .zip(&padded[x..x + k])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        });
    let mut out = vec![0u8; w * h];
    let horizontal = &horizontal;
    out.par_chunks_mut(w).enumerate().for_each(|(y, orow)| {
        let mut acc = vec![0f64; w];
        for (i, &wt) in weights.iter().enumerate() {
            let yy = reflect101(y as isize + i as isize - c as isize, h);
            for (a, &v) in acc.iter_mut().zip(&horizontal[yy * w..(yy + 1) * w]) {
                *a += wt * v;
            }
        }
        for (o, a) in orow.iter_mut().zip(acc) {
            *o = (a + 0.5).floor().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(w: u32, h: u32, cx: f64, cy: f64, r: f64, v: u8) -> Mask {
        let mut m = Mask::zeros(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (f64::from(x) + 0.5 - cx, f64::from(y) + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    m.data[(y * w + x) as usize] = v;
                }
            }
        }
        m
    }

    #[test]
    fn radius_zero_is_identity_and_size_is_kept() {
        let m = disc(97, 61, 48.0, 30.0, 15.0, 200);
        assert_eq!(feather(&m, 0.0).unwrap(), m);
        assert_eq!(
            feather(&m, 0.01).unwrap(),
            m,
            "truncates to 0 after scaling"
        );
        assert_eq!(feather(&m, f64::NAN).unwrap(), m);
        for r in [-6.0, 3.0, 5.0] {
            let out = feather(&m, r).unwrap();
            assert_eq!(out.size_px(), m.size_px());
            assert_eq!(out.data.len(), m.data.len());
        }
    }

    #[test]
    fn positive_never_lowers_negative_never_raises() {
        let mut shapes = vec![
            disc(97, 61, 48.0, 30.0, 15.0, 255),
            disc(64, 64, 20.0, 40.0, 9.0, 130),
        ];
        // A thin, isolated mask: the Gaussian alone would lower it.
        let mut dot = Mask::zeros(64, 64).unwrap();
        dot.data[32 * 64 + 32] = 255;
        shapes.push(dot);
        for m in &shapes {
            for r in [1.0, 2.5, 6.0] {
                let out = feather(m, r).unwrap();
                assert!(out.data.iter().zip(&m.data).all(|(o, i)| o >= i), "r={r}");
                assert!(
                    out.data.iter().zip(&m.data).any(|(o, i)| o > i),
                    "r={r} should grow"
                );
            }
            for r in [-1.0, -2.5, -6.0] {
                let out = feather(m, r).unwrap();
                assert!(out.data.iter().zip(&m.data).all(|(o, i)| o <= i), "r={r}");
            }
        }
        let m = &shapes[0];
        assert!(
            feather(m, -6.0)
                .unwrap()
                .data
                .iter()
                .zip(&m.data)
                .any(|(o, i)| o < i)
        );
    }

    #[test]
    fn dilation_is_a_disc() {
        let mut src = vec![0u8; 21 * 21];
        src[10 * 21 + 10] = 255;
        let out = dilate_disc(&src, 21, 21, 5);
        for y in 0..21i32 {
            for x in 0..21i32 {
                let inside = (x - 10) * (x - 10) + (y - 10) * (y - 10) <= 25;
                assert_eq!(out[(y * 21 + x) as usize] == 255, inside, "({x},{y})");
            }
        }
    }

    #[test]
    fn gaussian_keeps_constants_and_reflects() {
        let src = vec![200u8; 9 * 7];
        assert_eq!(gaussian(&src, 9, 7, 3), src);
        assert_eq!(reflect101(-1, 5), 1);
        assert_eq!(reflect101(5, 5), 3);
        assert_eq!(reflect101(-9, 5), 1);
        assert_eq!(reflect101(3, 1), 0);
    }

    #[test]
    fn independent_of_thread_count() {
        let m = disc(120, 90, 60.0, 45.0, 25.0, 255);
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| (feather(&m, 4.0).unwrap(), feather(&m, -4.0).unwrap()))
        };
        assert_eq!(run(1), run(4));
    }
}
