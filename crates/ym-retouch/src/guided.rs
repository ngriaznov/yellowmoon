//! The guided filter (He, Sun and Tang, 2010) for masks: pulls the edge of a soft mask to the edges of a picture.
//!
//! This is the filter of `docs/algorithms/06-masks.md` section 5.5 ("Refine edges"), with the same maths (`a = cov / (var +
//! eps)`, `b = mean_p - a * mean_g`, `q = mean_a * g + mean_b`, `q` held to `0..1`, windows clipped at the plane's edge and
//! divided by the number of samples) and the same Fast Guided Filter above 16 pixels of radius (`s = ceil(r / 16)`, the guide
//! and the mask averaged over blocks of `s x s` pixels aligned at multiples of `s` from the plane's corner, `a` and `b` found
//! on that grid with a window of `round(r / s)` cells and upsampled bilinearly). It is written again here on a plane of
//! `f32` instead of the colour crate's `f64` planes because a face mask of a 24 MP photo is millions of pixels and
//! `ym-retouch` does not depend on `ym-color`.
//!
//! Determinism: the window sums are summed-area tables in `f64` built in a fixed order; the passes over pixels write
//! disjoint rows, so the result does not depend on the thread count.

use rayon::prelude::*;

/// The radius above which the filter works on a grid of blocks (`docs/algorithms/06` section 5.5).
pub const SUBSAMPLE_ABOVE_PX: usize = 16;

/// The block size and the window radius in blocks for a radius of `radius_px` pixels.
pub fn grid_of(radius_px: usize) -> (usize, usize) {
    if radius_px > SUBSAMPLE_ABOVE_PX {
        let s = radius_px.div_ceil(SUBSAMPLE_ABOVE_PX);
        (s, ((radius_px as f64 / s as f64).round() as usize).max(1))
    } else {
        (1, radius_px.max(1))
    }
}

/// The mean over the window of half-side `r` around every cell of a `w x h` plane, clipped at the plane's edge.
fn box_mean(plane: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    // Summed-area table with a zero row and column in front.
    let sw = w + 1;
    let mut sat = vec![0.0f64; sw * (h + 1)];
    for y in 0..h {
        let mut row = 0.0f64;
        for x in 0..w {
            row += plane[y * w + x];
            sat[(y + 1) * sw + x + 1] = sat[y * sw + x + 1] + row;
        }
    }
    let mut out = vec![0.0f64; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for (x, v) in line.iter_mut().enumerate() {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let sum = sat[y1 * sw + x1] - sat[y0 * sw + x1] - sat[y1 * sw + x0] + sat[y0 * sw + x0];
            *v = sum / ((x1 - x0) * (y1 - y0)) as f64;
        }
    });
    out
}

/// Averages `plane` (`w x h`) over blocks of `s x s` pixels aligned at the corner; a block at the edge is the mean of the
/// pixels it has.
fn pool(plane: &[f32], w: usize, h: usize, s: usize) -> (Vec<f64>, usize, usize) {
    let (gw, gh) = (w.div_ceil(s), h.div_ceil(s));
    let mut out = vec![0.0f64; gw * gh];
    out.par_chunks_mut(gw).enumerate().for_each(|(by, line)| {
        let (y0, y1) = (by * s, ((by + 1) * s).min(h));
        for (bx, v) in line.iter_mut().enumerate() {
            let (x0, x1) = (bx * s, ((bx + 1) * s).min(w));
            let mut sum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += f64::from(plane[y * w + x]);
                }
            }
            *v = sum / ((x1 - x0) * (y1 - y0)) as f64;
        }
    });
    (out, gw, gh)
}

/// The guided filter of the mask plane `p` with the guide `guide`, both `w x h`, values `0..1`: radius `radius_px` pixels,
/// regularisation `eps` (on the guide in `0..1`; 1e-2 is smooth, 1e-5 follows even faint edges). A radius of 0 returns `p`.
///
/// # Panics
///
/// Never; planes of the wrong length return `p` unchanged.
pub fn guided_filter(
    p: &[f32],
    guide: &[f32],
    w: usize,
    h: usize,
    radius_px: usize,
    eps: f32,
) -> Vec<f32> {
    if radius_px == 0 || w == 0 || h == 0 || p.len() != w * h || guide.len() != w * h {
        return p.to_vec();
    }
    let (s, rs) = grid_of(radius_px);
    let (mg, gw, gh) = pool(guide, w, h, s);
    let (mp, _, _) = pool(p, w, h, s);
    let mean = |plane: &[f64]| box_mean(plane, gw, gh, rs);
    let (mean_g, mean_p) = (mean(&mg), mean(&mp));
    let gg: Vec<f64> = mg.iter().map(|v| v * v).collect();
    let gp: Vec<f64> = mg.iter().zip(&mp).map(|(a, b)| a * b).collect();
    // The products of the block means, not the means of the products: the Fast Guided Filter's choice.
    let (mean_gg, mean_gp) = (mean(&gg), mean(&gp));
    let eps = f64::from(eps);
    let mut a = vec![0.0f64; gw * gh];
    let mut b = vec![0.0f64; gw * gh];
    for i in 0..gw * gh {
        let var = mean_gg[i] - mean_g[i] * mean_g[i];
        let cov = mean_gp[i] - mean_g[i] * mean_p[i];
        a[i] = cov / (var.max(0.0) + eps);
        b[i] = mean_p[i] - a[i] * mean_g[i];
    }
    let (ma, mb) = (mean(&a), mean(&b));
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        // The grid coordinate of the pixel centre, and its two neighbours and weight, held to the grid.
        let gy = (y as f64 + 0.5) / s as f64 - 0.5;
        let (fy, ty) = (gy.floor(), gy - gy.floor());
        let (y0, y1) = (
            (fy.max(0.0) as usize).min(gh - 1),
            ((fy + 1.0).max(0.0) as usize).min(gh - 1),
        );
        for (x, v) in line.iter_mut().enumerate() {
            let gx = (x as f64 + 0.5) / s as f64 - 0.5;
            let (fx, tx) = (gx.floor(), gx - gx.floor());
            let (x0, x1) = (
                (fx.max(0.0) as usize).min(gw - 1),
                ((fx + 1.0).max(0.0) as usize).min(gw - 1),
            );
            let lerp = |plane: &[f64]| {
                let top = plane[y0 * gw + x0] * (1.0 - tx) + plane[y0 * gw + x1] * tx;
                let bottom = plane[y1 * gw + x0] * (1.0 - tx) + plane[y1 * gw + x1] * tx;
                top * (1.0 - ty) + bottom * ty
            };
            let q = lerp(&ma) * f64::from(guide[y * w + x]) + lerp(&mb);
            *v = q.clamp(0.0, 1.0) as f32;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An explicit double loop of the filter at `s = 1`, for comparison.
    fn reference(p: &[f32], g: &[f32], w: usize, h: usize, r: usize, eps: f64) -> Vec<f32> {
        let mean = |plane: &dyn Fn(usize, usize) -> f64, x: usize, y: usize| {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
            let mut s = 0.0;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    s += plane(xx, yy);
                }
            }
            s / ((x1 - x0) * (y1 - y0)) as f64
        };
        let gf = |x: usize, y: usize| f64::from(g[y * w + x]);
        let pf = |x: usize, y: usize| f64::from(p[y * w + x]);
        let mut a = vec![0.0; w * h];
        let mut b = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mg, mp) = (mean(&gf, x, y), mean(&pf, x, y));
                let var = mean(&|xx, yy| gf(xx, yy) * gf(xx, yy), x, y) - mg * mg;
                let cov = mean(&|xx, yy| gf(xx, yy) * pf(xx, yy), x, y) - mg * mp;
                a[y * w + x] = cov / (var + eps);
                b[y * w + x] = mp - a[y * w + x] * mg;
            }
        }
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let ma = mean(&|xx, yy| a[yy * w + xx], x, y);
                let mb = mean(&|xx, yy| b[yy * w + xx], x, y);
                out[y * w + x] = (ma * gf(x, y) + mb).clamp(0.0, 1.0) as f32;
            }
        }
        out
    }

    fn pseudo(i: usize) -> f32 {
        ((i.wrapping_mul(2_654_435_761) >> 7) % 1000) as f32 / 1000.0
    }

    #[test]
    fn matches_an_explicit_double_loop_at_small_radii() {
        let (w, h) = (23, 17);
        let g: Vec<f32> = (0..w * h)
            .map(|i| 0.5 * pseudo(i) + 0.2 * ((i % w) as f32 / w as f32))
            .collect();
        let p: Vec<f32> = (0..w * h)
            .map(|i| if (i % w) > 10 { 1.0 } else { 0.0 })
            .collect();
        for (r, eps) in [(1usize, 1e-2f32), (3, 3.2e-4), (7, 1e-5)] {
            let got = guided_filter(&p, &g, w, h, r, eps);
            let want = reference(&p, &g, w, h, r, f64::from(eps));
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 2e-5, "radius {r}: {worst}");
        }
    }

    #[test]
    fn pulls_a_soft_edge_to_the_edge_of_the_guide() {
        // The guide has a step at x = 40; the mask is a soft ramp centred at x = 40. The filter sharpens it onto the step.
        let (w, h) = (80, 20);
        let g: Vec<f32> = (0..w * h)
            .map(|i| if (i % w) < 40 { 0.2 } else { 0.8 })
            .collect();
        let p: Vec<f32> = (0..w * h)
            .map(|i| (((i % w) as f32 - 34.0) / 12.0).clamp(0.0, 1.0))
            .collect();
        let q = guided_filter(&p, &g, w, h, 8, 1e-4);
        let row = &q[10 * w..][..w];
        assert!(row[30] < 0.1 && row[50] > 0.9, "{} {}", row[30], row[50]);
        // The ramp over 12 pixels became a jump at the guide's step (between pixels 39 and 40).
        let jump = row[40] - row[39];
        assert!(jump > 0.3, "the edge is a step now: {jump}");
        let ramp = p[10 * w + 40] - p[10 * w + 39];
        assert!(jump > 5.0 * ramp, "{jump} against the ramp's {ramp}");
    }

    #[test]
    fn a_flat_guide_gives_the_mask_back_smoothed_and_a_zero_radius_is_the_identity() {
        let (w, h) = (30, 30);
        let g = vec![0.5f32; w * h];
        let p: Vec<f32> = (0..w * h).map(pseudo).collect();
        assert_eq!(guided_filter(&p, &g, w, h, 0, 1e-3), p);
        let q = guided_filter(&p, &g, w, h, 4, 1e-3);
        let spread = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
        };
        assert!(spread(&q) < spread(&p) * 0.1);
        // A wrong length is returned unchanged.
        assert_eq!(guided_filter(&p[..10], &g, w, h, 4, 1e-3), p[..10].to_vec());
    }

    #[test]
    fn the_subsampled_form_follows_the_full_one_and_is_independent_of_threads() {
        let (w, h) = (96, 64);
        let g: Vec<f32> = (0..w * h)
            .map(|i| if (i % w) < 48 { 0.25 } else { 0.75 } + 0.02 * pseudo(i))
            .collect();
        let p: Vec<f32> = (0..w * h)
            .map(|i| (((i % w) as f32 - 40.0) / 14.0).clamp(0.0, 1.0))
            .collect();
        assert_eq!(grid_of(16), (1, 16));
        assert_eq!(grid_of(17), (2, 9));
        assert_eq!(grid_of(48), (3, 16));
        let fast = guided_filter(&p, &g, w, h, 20, 1e-3);
        let (s, _) = grid_of(20);
        assert_eq!(s, 2);
        // The step of the guide is found: the mask is near 0 left of it and near 1 right of it.
        let row = &fast[32 * w..][..w];
        assert!(row[30] < 0.1 && row[66] > 0.9, "{} {}", row[30], row[66]);
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let pool4 = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let a = pool1.install(|| guided_filter(&p, &g, w, h, 20, 1e-3));
        let b = pool4.install(|| guided_filter(&p, &g, w, h, 20, 1e-3));
        assert_eq!(a, b);
        assert_eq!(a, fast);
    }
}
