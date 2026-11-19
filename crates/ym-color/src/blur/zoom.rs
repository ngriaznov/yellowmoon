//! Zoom, the fast method (`docs/33-blur.md` section 3.5, BL-09): a radial line read from a **pyramid**, in f64.
//!
//! The exact kernel ([`super::gather_zoom`]) takes at least 64 bilinear taps along the line of every pixel. The fast method
//! takes at most [`MAX_TAPS`] (32) and reads them **trilinearly** from the pyramid of the weighted planes `omega X` and
//! `omega` (2 x 2 box means at absolute alignment, [`super::lens::box_level`]), so a long line is read from coarse cells:
//!
//! * a pixel at `v = p - c` from the centre has the line `a |v|` (a length in pixels) and `n = clamp(ceil(a |v|) + 1, 2, 32)`
//!   taps at `q_i = p + v a t_i`, `t_i = (i + 1/2) / n - 1/2`, equal weights ([`tap_count`]);
//! * the taps' spacing is `s = a |v| / (n - 1)` ([`spacing_px`]) and the level they read is `clamp(log2 s, 0, L)`, the
//!   fraction blending the two levels around it ([`level_of`]): near the centre the sharp pixels, far from it coarse cells;
//! * `L` ([`top_level`]) is the smallest level whose cell is at least the largest spacing the frame has (`a d_max / 31`);
//! * a cell outside the grid has no weight, one the frame's edge cuts counts its missing pixels as 0 (the Lens' rule), so the
//!   picture's edge renormalises as the exact kernel's does and a flat field stays flat.
//!
//! It is **not a convolution** (the line's length depends on the pixel), so energy is conserved to a few per mille, a flat
//! field exactly. A window of the [`reach`] equals the whole frame bit for bit: every number is a function of the pixel, the
//! settings and the cells at absolute positions. The device does the same in f32 (`ym_gpu::kernels::blur_zoom`).

use super::lens::{Grid, box_level};
use super::Plane4;

/// The most taps along a Zoom line.
pub const MAX_TAPS: usize = 32;

/// The number of taps along a line of `line_px` pixels: `clamp(ceil(line) + 1, 2, 32)`.
#[must_use]
pub fn tap_count(line_px: f64) -> usize {
    if !(line_px > 0.0) {
        return 2;
    }
    if line_px >= (MAX_TAPS - 1) as f64 {
        return MAX_TAPS;
    }
    (line_px.ceil() as usize + 1).clamp(2, MAX_TAPS)
}

/// The spacing of the taps of a line of `line_px` pixels, `line / (n - 1)`.
#[must_use]
pub fn spacing_px(line_px: f64) -> f64 {
    let n = tap_count(line_px);
    line_px / (n - 1) as f64
}

/// The level the taps read for a spacing of `spacing` pixels, with the fraction between two levels: `clamp(log2 s, 0, top)`
/// (0 for a spacing of one pixel or less: the pixels themselves).
#[must_use]
pub fn level_of(spacing: f64, top: u32) -> f64 {
    if spacing <= 1.0 {
        0.0
    } else {
        spacing.log2().min(f64::from(top))
    }
}

/// The distance from the centre (a fraction of the frame, clamped to the frame) to the farthest corner, pixels.
#[must_use]
pub fn d_max_px(center: [f64; 2], full_w: usize, full_h: usize) -> f64 {
    let c = [
        center[0].clamp(0.0, 1.0) * full_w as f64,
        center[1].clamp(0.0, 1.0) * full_h as f64,
    ];
    let far = |p: f64, size: usize| p.max(size as f64 - p);
    far(c[0], full_w).hypot(far(c[1], full_h))
}

/// The highest level of the pyramid a Zoom of span `a` (`ym_color::blur::zoom_span`) reads: the smallest `L` with
/// `2^L >= a d_max / 31` (the largest spacing of the frame), 0 when that is one pixel or less.
#[must_use]
pub fn top_level(a: f64, center: [f64; 2], full_w: usize, full_h: usize) -> u32 {
    let smax = a * d_max_px(center, full_w, full_h) / (MAX_TAPS - 1) as f64;
    if smax.is_nan() || smax <= 1.0 {
        return 0;
    }
    let mut level = 0u32;
    while f64::from(1u32 << level) < smax && level < 30 {
        level += 1;
    }
    level
}

/// The reach of a Zoom in pixels (`docs/33` section 3.8): `ceil((a / 2) d_max)` for the line, plus what the trilinear read
/// takes beyond a tap: the 2 x 2 cells around it at the top level, `1.5 * 2^L`, and two pixels for the rounding of positions. A
/// scalar for the whole tile (the farthest corner), 0 for `a <= 0`. The spec's table has `2^L` where this has `1.5 * 2^L + 2`
/// (`docs/33` section 11, item 55).
#[must_use]
pub fn reach(a: f64, center: [f64; 2], full_w: usize, full_h: usize) -> usize {
    if !(a > 0.0) {
        return 0;
    }
    let half_line = (a / 2.0 * d_max_px(center, full_w, full_h)).max(0.0).ceil() as usize;
    let top = top_level(a, center, full_w, full_h);
    half_line + (3usize << top).div_ceil(2) + 2
}

/// The levels of the pyramid of a window (`x0, y0, w, h` of a frame of `frame` pixels) up to `top`: the grid and the planes of
/// each, level 0 first. Stops early (a shorter list) when a level has no whole cell in the window.
fn pyramid(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    frame: (usize, usize),
    top: u32,
) -> Vec<(Grid, Vec<Plane4>)> {
    let (x0, y0, w, h) = window;
    let mut levels = vec![(
        Grid {
            level: 0,
            i0: x0 as i64,
            j0: y0 as i64,
            w,
            h,
        },
        planes.to_vec(),
    )];
    for l in 1..=top {
        let Some(up) = Grid::of(window, frame, l) else {
            break;
        };
        let cells = {
            let (g, c) = &levels[l as usize - 1];
            box_level(c, g, &up)
        };
        levels.push((up, cells));
    }
    levels
}

/// The bilinear read of the cells of one level at the frame position `(qx, qy)` (pixel `i` at `i`), a cell's centre at
/// `(I + 1/2) f`; a cell outside the grid reads 0. At level 0 it is the pixels' bilinear read.
fn read(level: &(Grid, Vec<Plane4>), qx: f64, qy: f64) -> Plane4 {
    let (grid, cells) = level;
    let f = (1u64 << grid.level) as f64;
    let u = (qx + 0.5) / f - 0.5;
    let v = (qy + 0.5) / f - 0.5;
    let (i0, j0) = (u.floor() as i64, v.floor() as i64);
    let (tx, ty) = (u - i0 as f64, v - j0 as f64);
    let cell = |i: i64, j: i64| -> Plane4 {
        if grid.has(i, j) {
            cells[grid.index(i, j)]
        } else {
            [0.0; 4]
        }
    };
    let (c00, c10, c01, c11) = (
        cell(i0, j0),
        cell(i0 + 1, j0),
        cell(i0, j0 + 1),
        cell(i0 + 1, j0 + 1),
    );
    let mut s = [0.0; 4];
    for k in 0..4 {
        let top = (1.0 - tx) * c00[k] + tx * c10[k];
        let bottom = (1.0 - tx) * c01[k] + tx * c11[k];
        s[k] = (1.0 - ty) * top + ty * bottom;
    }
    s
}

/// The sums `[N (3), D]` of a Zoom over a window of weighted planes `[omega X (3), omega]` (`w x h` pixels at `(x0, y0)` of a
/// frame of `frame` pixels), by the pyramid. `a` is the span ([`super::zoom_span`] of the Amount) and `center` the centre as a
/// fraction of the frame. A pixel with no whole cell of its level in the window reads 0 there (the window is smaller than a cell).
#[must_use]
pub fn gather(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    frame: (usize, usize),
    params: (f64, [f64; 2]),
) -> Vec<Plane4> {
    gather_region(planes, window, frame, params, window)
}

/// [`gather`] over `region` only (`x, y, w, h` in the frame, inside the window): the pixels of the region, row by row. The cells are
/// those of the whole window, so a pixel is the same bits as in [`gather`].
#[must_use]
pub fn gather_region(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    frame: (usize, usize),
    (a, center): (f64, [f64; 2]),
    region: (usize, usize, usize, usize),
) -> Vec<Plane4> {
    let top = top_level(a, center, frame.0, frame.1);
    let levels = pyramid(planes, window, frame, top);
    let (rx, ry, rw, rh) = region;
    let cx = center[0] * frame.0 as f64 - 0.5;
    let cy = center[1] * frame.1 as f64 - 0.5;
    let mut out = vec![[0.0; 4]; rw * rh];
    for y in 0..rh {
        for x in 0..rw {
            let (px, py) = ((rx + x) as f64, (ry + y) as f64);
            let (vx, vy) = (px - cx, py - cy);
            let line = a * vx.hypot(vy);
            let n = tap_count(line);
            let ell = level_of(line / (n - 1) as f64, top);
            let mut l0 = ell.floor() as usize;
            let mut frac = ell - l0 as f64;
            if l0 >= top as usize {
                l0 = top as usize;
                frac = 0.0;
            }
            let Some(lo) = levels.get(l0) else {
                continue;
            };
            let hi = if frac > 0.0 { levels.get(l0 + 1) } else { None };
            if frac > 0.0 && hi.is_none() {
                continue;
            }
            let mut acc = [0.0; 4];
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64 - 0.5;
                let (qx, qy) = (px + vx * a * t, py + vy * a * t);
                let mut r = read(lo, qx, qy);
                if let Some(hi) = hi {
                    let r1 = read(hi, qx, qy);
                    for k in 0..4 {
                        r[k] = (1.0 - frac) * r[k] + frac * r1[k];
                    }
                }
                for k in 0..4 {
                    acc[k] += r[k];
                }
            }
            for v in &mut acc {
                *v /= n as f64;
            }
            out[y * rw + x] = acc;
        }
    }
    out
}
