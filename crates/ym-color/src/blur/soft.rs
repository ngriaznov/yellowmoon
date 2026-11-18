//! Soft's fast method (`docs/33-blur.md` 3.5): the pixel-integrated Gaussian cut at 4 sigma, **a direct separable sum up to
//! `sigma = 4` px and on a coarse grid beyond** (Film's rule, `docs/24` 3.9a, with two changes, below).
//!
//! The coarse grid: the weighted planes `omega X` and `omega` are reduced to cells of `f x f` pixels at **absolute multiples of
//! `f`** (so a cell never depends on the window), the cells are blurred by a direct separable sum, and the result is read back
//! at the pixels (a cell's centre at `(I + 1/2) f`). `f = 2^j` is the smallest power of two (at least 2) that leaves the cell
//! sigma at most [`CELL_SIGMA_MAX`] cells, so a pass has at most `2 ceil(4 * 4) + 1 = 33` taps over `1 / f^2` of the pixels.
//!
//! **The reduction and the read-back are quadratic B-splines, not a box and a bilinear** (`docs/33` 11, item 40). Film's box
//! mean puts the mass of a block at the block's centre, so a light of a few pixels moves by up to `(f - 1) / 2` pixels and
//! its blurred peak is off by 1 to 7 % at the widths Soft reaches (measured, `tests/blur_soft.rs`). The B-spline weights are
//! positive (so `D >= 0` and the blend stays a weighted mean), sum to 1, keep the first moment exact, and **their second
//! moment about the pixel is `f^2 / 4` whatever the pixel's phase in the cell** (a tent's is not), so a point's blur has
//! the same width wherever it sits. The reduction and the read-back add `f^2 / 2` to the variance and the cell sigma makes
//! it up ([`cell_sigma`]). Each of the two is `3 x 3` cells a pixel.
//!
//! Only whole cells are used: a cell is valid when every pixel of its support (`3 f` pixels) is in the window or outside the
//! frame (the sums renormalise at the frame's edge, `docs/33` 3.4); a cell the window cuts inside the frame is left out. The
//! reach of the scheduler holds every cell a pixel of the window's own rows reads ([`super::reach_px`], [`reads_px`]), so a
//! window with its halo equals the frame bit for bit.

use super::{Frame, Plane4, gather_separable, gauss_weights, gauss_weights_cut};

/// A Gaussian up to this wide (pixels) is a direct sum; a wider one is done on the coarse grid.
pub const DIRECT_MAX_SIGMA_PX: f64 = 4.0;

/// The widest Gaussian of cells: the coarse grid's `f` is the smallest that brings the cell sigma to this many cells.
pub const CELL_SIGMA_MAX: f64 = 4.0;

/// The variance of the reduction and the read-back together, in `f^2`: a quadratic B-spline's second moment is `1/4` each.
const SPLAT_VARIANCE: f64 = 0.5;

/// The sigma of the cells' Gaussian, in cells, for a Gaussian of `sigma_px` pixels on cells of `f` pixels: the variance of the
/// pixel-integrated Gaussian is `sigma^2 + 1/12`, the reduction and the read-back add `f^2 / 2`, and the cells' own integrated
/// Gaussian adds `1/12` cell squared.
#[must_use]
pub fn cell_sigma(sigma_px: f64, f: usize) -> f64 {
    let f2 = (f * f) as f64;
    (((sigma_px * sigma_px + 1.0 / 12.0 - f2 * SPLAT_VARIANCE) / f2) - 1.0 / 12.0)
        .max(0.0)
        .sqrt()
}

/// The level `j` of the coarse grid (`f = 2^j`) for a Gaussian of `sigma_px` pixels: 0 for a direct sum, else the smallest
/// `j >= 1` with [`cell_sigma`] at most [`CELL_SIGMA_MAX`].
#[must_use]
pub fn cell_level(sigma_px: f64) -> u32 {
    if sigma_px.is_nan() || sigma_px <= DIRECT_MAX_SIGMA_PX {
        return 0;
    }
    (1..30)
        .find(|&j| cell_sigma(sigma_px, 1usize << j) <= CELL_SIGMA_MAX)
        .unwrap_or(30)
}

/// The weight of pixel `x` on cell `i` of cells of `f` pixels: the quadratic B-spline of `s = |u - i|`,
/// `u = (x + 1/2) / f - 1/2`: `3/4 - s^2` up to `1/2`, `(3/2 - s)^2 / 2` up to `3/2`, 0 beyond. With `d = 2 x + 1 - (2 i + 1) f`
/// it is `(3 f^2 - d^2) / (4 f^2)` and `(3 f - |d|)^2 / (8 f^2)`: integers over a power of two, exact in `f64` and `f32`.
#[must_use]
pub fn spline(x: i64, i: i64, f: i64) -> f64 {
    let d = (2 * x + 1 - (2 * i + 1) * f).abs();
    if 2 * d <= 2 * f {
        (3 * f * f - d * d) as f64 / (4 * f * f) as f64
    } else if d < 3 * f {
        let a = (3 * f - d) as f64;
        a * a / (8 * f * f) as f64
    } else {
        0.0
    }
}

/// The cells of one axis that are valid for the window `[w0, w1)` of the frame `[0, full)`: the first index and the count. A
/// cell `i` has the support `[(i - 1) f, (i + 2) f)`, and is valid when the part of it inside the frame is inside the window
/// (a cell cut by the frame's edge counts its missing pixels as 0) and it has a pixel in the frame.
#[must_use]
pub fn cell_range(w0: i64, w1: i64, full: i64, f: i64) -> (i64, i64) {
    let ceil = |a: i64| -((-a).div_euclid(f));
    let lo = if w0 <= 0 { -1 } else { ceil(w0) + 1 };
    let hi = if w1 >= full {
        ceil(full) + 1
    } else {
        w1.div_euclid(f) - 1
    };
    (lo, (hi - lo).max(0))
}

/// The taps each side of the cells' Gaussian: `ceil(4 sigma_c)`, and at most `4 sigma / f` so that the cells a pixel reads lie
/// within `4 sigma + 3 f` of it (what the scheduler's reach holds, `docs/33` 3.8). The cut is the same 4 sigma but for a
/// fraction of a cell; the renormalised weights differ by under 2e-4.
#[must_use]
pub fn cell_half(sigma_px: f64, f: usize) -> i64 {
    let by_cell = (super::SIGMA_CUT * cell_sigma(sigma_px, f)).ceil() as i64;
    let by_reach = (super::SIGMA_CUT * sigma_px / f as f64).floor() as i64;
    by_cell.min(by_reach)
}

/// The weights of the cells' Gaussian for a Gaussian of `sigma_px` pixels on cells of `f` pixels, `2 half + 1` of them.
#[must_use]
pub fn cell_weights(sigma_px: f64, f: usize) -> Vec<f64> {
    gauss_weights_cut(cell_sigma(sigma_px, f), cell_half(sigma_px, f))
}

/// How far from a pixel the fast method reads, in pixels, for a Gaussian of `sigma_px`: `ceil(4 sigma)` on a direct sum,
/// `(3 + half) f - 1` on the coarse grid (the support of the cells the read-back reaches). The scheduler's reach
/// ([`super::reach_px`]) is never less.
#[must_use]
pub fn reads_px(sigma_px: f64) -> usize {
    let level = cell_level(sigma_px);
    if level == 0 {
        return (super::SIGMA_CUT * sigma_px).ceil().max(0.0) as usize;
    }
    let f = 1usize << level;
    (3 + cell_half(sigma_px, f).max(0) as usize) * f - 1
}

/// The sums of the weighted planes `p` (a window of `w x h` pixels at `frame.x0, frame.y0`) by the Soft Gaussian of `sigma`
/// pixels: the direct separable sum (the oracle's own) up to [`DIRECT_MAX_SIGMA_PX`], else the coarse grid.
pub fn gather(p: &[Plane4], w: usize, h: usize, frame: &Frame, sigma: f64) -> Vec<Plane4> {
    let level = cell_level(sigma);
    if level == 0 {
        return gather_separable(p, w, h, &gauss_weights(sigma));
    }
    let f = 1i64 << level;
    let (x0, y0) = (frame.x0 as i64, frame.y0 as i64);
    let (ci0, cw) = cell_range(x0, x0 + w as i64, frame.full_w as i64, f);
    let (cj0, ch) = cell_range(y0, y0 + h as i64, frame.full_h as i64, f);
    if cw == 0 || ch == 0 {
        // No whole cell (a window narrower than a cell): nothing to read, the sums are 0 and the blend keeps the input.
        return vec![[0.0; 4]; w * h];
    }
    let (cw, ch) = (cw as usize, ch as usize);
    let inv = 1.0 / f as f64;
    // The reduction along x: for each row of the window, the cells' columns (a pixel outside the frame counts 0).
    let mut rows = vec![[0.0; 4]; cw * h];
    for y in 0..h {
        for ci in 0..cw {
            let i = ci0 + ci as i64;
            let mut acc = [0.0; 4];
            for x in ((i - 1) * f).max(0)..((i + 2) * f).min(frame.full_w as i64) {
                let k = spline(x, i, f);
                let q = &p[y * w + (x - x0) as usize];
                for c in 0..4 {
                    acc[c] += k * q[c];
                }
            }
            rows[y * cw + ci] = acc.map(|a| a * inv);
        }
    }
    // The reduction along y.
    let mut cells = vec![[0.0; 4]; cw * ch];
    for cj in 0..ch {
        let j = cj0 + cj as i64;
        for ci in 0..cw {
            let mut acc = [0.0; 4];
            for y in ((j - 1) * f).max(0)..((j + 2) * f).min(frame.full_h as i64) {
                let k = spline(y, j, f);
                let q = &rows[(y - y0) as usize * cw + ci];
                for c in 0..4 {
                    acc[c] += k * q[c];
                }
            }
            cells[cj * cw + ci] = acc.map(|a| a * inv);
        }
    }
    let blurred = gather_separable(&cells, cw, ch, &cell_weights(sigma, f as usize));
    // The read-back; a cell outside the grid reads 0. The nearest cell of a pixel is `x div f`.
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        let ay = y0 + y as i64;
        let j0 = ay.div_euclid(f);
        for x in 0..w {
            let ax = x0 + x as i64;
            let i0 = ax.div_euclid(f);
            let mut acc = [0.0; 4];
            for j in j0 - 1..=j0 + 1 {
                let ky = spline(ay, j, f);
                let b = j - cj0;
                if b < 0 || b >= ch as i64 {
                    continue;
                }
                let mut line = [0.0; 4];
                for i in i0 - 1..=i0 + 1 {
                    let a = i - ci0;
                    if a < 0 || a >= cw as i64 {
                        continue;
                    }
                    let k = spline(ax, i, f);
                    let q = &blurred[b as usize * cw + a as usize];
                    for c in 0..4 {
                        line[c] += k * q[c];
                    }
                }
                for c in 0..4 {
                    acc[c] += ky * line[c];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}
