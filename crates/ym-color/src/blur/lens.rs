//! Lens, the fast method: **row segments** (`docs/33-blur.md` section 3.5, BL-07b), in f64.
//!
//! The exact kernel ([`super::aperture_table`]) is the coverage of each pixel by the aperture, a direct sum over up to
//! `(2 R + 3)^2` pixels: 235 000 taps at `R = 240` px. A disc and a convex regular `n`-gon meet each pixel row of the kernel's
//! support in **one interval** `[x0(y), x1(y)]`, so with a prefix sum of each row of the weighted planes every kernel row is
//! two lookups, and a pixel costs about `2 R + 1` row lookups (times the sub-rows below) instead of `(2 R + 1)^2` taps:
//!
//! 1. **The row table** ([`row_table`], a function of `R`, the blades and the rotation alone). Each pixel row `dy` of the
//!    kernel is cut at the aperture's vertices (and at its widest point, for a disc) into pieces over which each end of the
//!    chord moves monotonically (a polygon's linearly), and a piece of height `h` whose ends move `d` pixels gets
//!    `S = clamp(ceil(3 d), 1, 32)` **sub-rows** at its midpoints, each of weight `h / S` and covering the chord's exact `x`
//!    overlap with every pixel (a fraction of the end pixels). That is the pixel's coverage by the aperture, integrated exactly
//!    in `x` and by the midpoint rule with `S` samples in `y`: the rim is anti-aliased and continuous in `R` and in the
//!    rotation, a flat edge is exact (the cuts), and a pole of a disc (where the chord moves fast) gets up to 32 sub-rows.
//!    About 3 to 5 sub-rows a row on average. The weights are normalised to sum 1 over the table's own area, so a flat field
//!    stays flat.
//! 2. **The prefix** of each row of the window, in **integer fixed point** (`2^-32`, 64 bits): sums of integers are exact in any
//!    order, so a window of any origin gives the same bits as the whole frame (hard rule 8; a floating prefix would differ by
//!    its rounding with where the window starts). A value is clamped to `+-` [`FIXED_MAX`] first.
//! 3. A pixel adds its sub-rows in table order: `w * (F(b) - F(a))`, `F` the row's running integral, which is the prefix at the
//!    whole pixels (an exact integer difference) plus the two end pixels' own values times the fraction of them the chord
//!    covers (`a = x + x0`: whole part `x + ka`, fraction `fa`, both exact: no float depends on the window's origin).
//!
//! `R <= 8` px ([`EXACT_MAX_RADIUS`]) is **the exact kernel bit for bit**: the table's nonzero taps, summed in its order. A
//! pixel outside the window has no weight (the window is the made area grown by the reach and cut to the frame), so the
//! picture's edge renormalises as the exact kernel's does.
//!
//! The device does the same in f32 (`ym_gpu::kernels::blur_lens`), the prefix in two `u32` words per plane; the arithmetic here
//! is written in the device's order so that the two agree to rounding.

use super::{Plane4, Table, aperture_table};

/// The aperture's radius in pixels up to which the fast method is the exact table (`docs/33` section 3.5).
pub const EXACT_MAX_RADIUS: f64 = 8.0;

/// The prefix's fixed point: a value is the integer `v * 2^FIXED_SHIFT`.
pub const FIXED_SHIFT: u32 = 32;

/// The magnitude a weighted value is clamped to before the prefix (131072 times the largest LogC4 linear value is far above any
/// picture): a row of 16384 columns of it is at most `2^14 * 2^14 * 2^32 = 2^60`, inside 64 bits.
pub const FIXED_MAX: f64 = 16384.0;

/// Sub-rows a row gets per pixel its chord's ends move across it ([`row_table`]).
pub const SUBROWS_PER_PX: f64 = 3.0;

/// The most sub-rows of one piece of a row.
pub const SUBROWS_MAX: usize = 32;

/// Whether the aperture of radius `r_px` is made by row segments (`r_px > 8`); else it is the exact table.
#[must_use]
pub fn uses_rows(r_px: f64) -> bool {
    r_px > EXACT_MAX_RADIUS
}

/// The pixels a pixel reads on each side: `ceil(R) + 1` (the aperture's pixels, and the end pixel a chord reaches into).
#[must_use]
pub fn reach(r_px: f64) -> usize {
    if r_px.is_nan() {
        return 1;
    }
    r_px.max(0.0).ceil() as usize + 1
}

/// One sub-row of the row table: a horizontal chord of the aperture at row `dy`, weight `w`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubRow {
    /// The kernel row (pixels below the centre).
    pub dy: i32,
    /// The chord's left end `x0 + 1/2` is `ka + fa`, `ka` an integer and `fa` in `[0, 1)`.
    pub ka: i32,
    /// The chord's right end `x1 + 1/2` is `kb + fb`.
    pub kb: i32,
    /// The fraction part of the left end.
    pub fa: f64,
    /// The fraction part of the right end.
    pub fb: f64,
    /// The sub-row's weight: the share of the row it carries over the table's area, so that the whole table sums to 1.
    pub w: f64,
}

/// The aperture: a disc (`n = 0`) or a regular `n`-gon of circumradius `r`, rotated by `rot` radians, centred on the origin.
struct Shape {
    r: f64,
    n: u32,
    apothem: f64,
    normals: Vec<(f64, f64)>,
    vertices_y: Vec<f64>,
}

impl Shape {
    fn new(r: f64, blades: u32, rotation_deg: f64) -> Self {
        let n = blades;
        let rot = rotation_deg.to_radians();
        let tau = std::f64::consts::TAU;
        let (apothem, normals, vertices_y) = if n == 0 {
            (r, vec![], vec![])
        } else {
            (
                r * (std::f64::consts::PI / f64::from(n)).cos(),
                (0..n)
                    .map(|i| {
                        let a = rot + (f64::from(i) + 0.5) * tau / f64::from(n);
                        (a.cos(), a.sin())
                    })
                    .collect(),
                (0..n)
                    .map(|i| r * (rot + f64::from(i) * tau / f64::from(n)).sin())
                    .collect(),
            )
        };
        Self {
            r,
            n,
            apothem,
            normals,
            vertices_y,
        }
    }

    fn y_range(&self) -> (f64, f64) {
        if self.n == 0 {
            (-self.r, self.r)
        } else {
            let lo = self
                .vertices_y
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let hi = self
                .vertices_y
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        }
    }

    /// The chord's ends at height `y`, or `None` when it is empty.
    fn chord(&self, y: f64) -> Option<(f64, f64)> {
        let (lo, hi) = self.chord_loose(y)?;
        (lo <= hi).then_some((lo, hi))
    }

    /// The chord's ends at height `y` where `y` is within the shape's rows up to rounding: at a vertex the two ends may cross
    /// by rounding, and are then both their mean (the vertex). `None` for a parallel side the point is outside of.
    fn chord_loose(&self, y: f64) -> Option<(f64, f64)> {
        if self.n == 0 {
            let c = (self.r * self.r - y * y).max(0.0).sqrt();
            return Some((-c, c));
        }
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for &(c, s) in &self.normals {
            let rhs = self.apothem - y * s;
            if c.abs() < 1e-12 {
                if rhs < 0.0 {
                    return None;
                }
            } else if c > 0.0 {
                hi = hi.min(rhs / c);
            } else {
                lo = lo.max(rhs / c);
            }
        }
        if lo > hi {
            Some((0.5 * (lo + hi), 0.5 * (lo + hi)))
        } else {
            Some((lo, hi))
        }
    }
}

/// The row table of the aperture of radius `r_px` (`blades` 0 for a disc, else the polygon's blades; `rotation_deg`), in the
/// order the sums run: `dy` ascending, a row's pieces from the top, a piece's sub-rows from the top. Empty for a radius that is
/// not positive and finite. The weights sum to 1 over the table's own area (`sum w (x1 - x0) = 1`).
#[must_use]
pub fn row_table(r_px: f64, blades: u32, rotation_deg: f64) -> Vec<SubRow> {
    if !(r_px > 0.0 && r_px.is_finite()) {
        return Vec::new();
    }
    let shape = Shape::new(r_px, blades, rotation_deg);
    let (y_lo, y_hi) = shape.y_range();
    let half = r_px.ceil() as i32 + 1;
    let mut out: Vec<SubRow> = Vec::new();
    let mut area = 0.0;
    for dy in -half..=half {
        let a0 = (f64::from(dy) - 0.5).max(y_lo);
        let b0 = (f64::from(dy) + 0.5).min(y_hi);
        if b0 - a0 <= 1e-12 {
            continue;
        }
        let mut cuts = vec![a0];
        let mut inner: Vec<f64> = if shape.n == 0 {
            vec![0.0]
        } else {
            shape.vertices_y.clone()
        };
        inner.retain(|&v| v > a0 && v < b0);
        inner.sort_by(f64::total_cmp);
        cuts.extend(inner);
        cuts.push(b0);
        for pair in cuts.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let h = b - a;
            if h <= 1e-12 {
                continue;
            }
            // How far each end of the chord moves over the piece: its ends are monotone (linear for a polygon).
            let near = |y: f64| shape.chord_loose(y).unwrap_or((0.0, 0.0));
            let (la, ra) = near(a.clamp(y_lo, y_hi));
            let (lb, rb) = near(b.clamp(y_lo, y_hi));
            let d = (lb - la).abs().max((rb - ra).abs());
            let s = ((SUBROWS_PER_PX * d).ceil().max(1.0) as usize).min(SUBROWS_MAX);
            for i in 0..s {
                let y = a + (i as f64 + 0.5) / s as f64 * h;
                let Some((x0, x1)) = shape.chord(y) else {
                    continue;
                };
                if x1 <= x0 {
                    continue;
                }
                let (u0, u1) = (x0 + 0.5, x1 + 0.5);
                let (ka, kb) = (u0.floor(), u1.floor());
                let w = h / s as f64;
                area += w * (x1 - x0);
                out.push(SubRow {
                    dy,
                    ka: ka as i32,
                    kb: kb as i32,
                    fa: u0 - ka,
                    fb: u1 - kb,
                    w,
                });
            }
        }
    }
    if area > 0.0 {
        for e in &mut out {
            e.w /= area;
        }
    }
    out
}

/// `v` as the prefix's integer: `round(v * 2^32)` with the value clamped to `+-` [`FIXED_MAX`] (NaN is 0).
#[must_use]
pub fn quantize(v: f64) -> i64 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(-FIXED_MAX, FIXED_MAX) * 4_294_967_296.0).round_ties_even() as i64
}

/// The exact table's nonzero taps `(dx, dy, k)` of the aperture of radius `r_px`, in the order the exact gather sums.
#[must_use]
pub fn exact_taps(r_px: f64, blades: u32, rotation_deg: f64) -> Vec<(i64, i64, f64)> {
    taps(&aperture_table(r_px, blades, rotation_deg))
}

/// The sums of a window `w x h` of weighted planes over the rows `rows` by the row table: the prefix of every row in integers,
/// then for each pixel of `rows` the sub-rows in table order.
fn gather_row_segments(
    p: &[Plane4],
    w: usize,
    h: usize,
    table: &[SubRow],
    rows: std::ops::Range<usize>,
) -> Vec<Plane4> {
    let stride = w + 2;
    let mut prefix = vec![[0i64; 4]; stride * h];
    for y in 0..h {
        let mut acc = [0i64; 4];
        for x in 0..w {
            prefix[y * stride + x] = acc;
            let q = &p[y * w + x];
            for k in 0..4 {
                acc[k] += quantize(q[k]);
            }
        }
        prefix[y * stride + w] = acc;
        prefix[y * stride + w + 1] = acc;
    }
    let unit = 1.0 / 4_294_967_296.0;
    let mut out = Vec::with_capacity(w * rows.len());
    for y in rows {
        for x in 0..w {
            let mut acc = [0.0f64; 4];
            for e in table {
                let yy = y as i64 + i64::from(e.dy);
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                let yy = yy as usize;
                // The end pixel (clamped to the window: left of it nothing, right of it all) and the fraction of it.
                let end = |k: i32, f: f64| -> (usize, f64) {
                    let i = x as i64 + i64::from(k);
                    if i < 0 {
                        (0, 0.0)
                    } else if i >= w as i64 {
                        (w, 0.0)
                    } else {
                        (i as usize, f)
                    }
                };
                let (ia, ua) = end(e.ka, e.fa);
                let (ib, ub) = end(e.kb, e.fb);
                let (pa, pb) = (&prefix[yy * stride + ia], &prefix[yy * stride + ib]);
                let zero = [0.0; 4];
                let va = if ia < w { &p[yy * w + ia] } else { &zero };
                let vb = if ib < w { &p[yy * w + ib] } else { &zero };
                for k in 0..4 {
                    let whole = (pb[k] - pa[k]) as f64 * unit;
                    acc[k] += e.w * ((whole + vb[k] * ub) - va[k] * ua);
                }
            }
            out.push(acc);
        }
    }
    out
}

/// A grid of cells at absolute multiples of `f = 2^level` (the device's `CellGrid`, `ym_gpu::cpu::blur`): cell `(i, j)` holds
/// the pixels `[i f, (i + 1) f) x [j f, (j + 1) f)`, and the grid holds the cells whose pixels the window covers wherever the
/// frame has any (a cell cut by the frame's edge counts its missing pixels as 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// The level: a cell is `2^level` pixels on a side.
    pub level: u32,
    /// The index of the first column of cells.
    pub i0: i64,
    /// The index of the first row of cells.
    pub j0: i64,
    /// The columns of cells.
    pub w: usize,
    /// The rows of cells.
    pub h: usize,
}

/// The cells of one axis: the first index and the count, for a read range `[r0, r1)` of a frame range `[0, f1)`.
fn axis(r0: i64, r1: i64, f1: i64, f: i64) -> (i64, i64) {
    let ceil = |a: i64| -((-a).div_euclid(f));
    let lo = if r0 <= 0 { 0 } else { ceil(r0) };
    let hi = if r1 >= f1 { ceil(f1) } else { r1.div_euclid(f) };
    (lo, hi - lo)
}

impl Grid {
    /// The grid of `level` for a window of `w x h` pixels at `(x0, y0)` of a frame of `frame_w x frame_h`, or `None` when no
    /// cell of the level is whole in it.
    #[must_use]
    pub fn of(
        window: (usize, usize, usize, usize),
        frame: (usize, usize),
        level: u32,
    ) -> Option<Self> {
        let (x0, y0, w, h) = window;
        if w == 0 || h == 0 || level > 30 {
            return None;
        }
        let f = 1i64 << level;
        let (i0, nw) = axis(x0 as i64, (x0 + w) as i64, frame.0 as i64, f);
        let (j0, nh) = axis(y0 as i64, (y0 + h) as i64, frame.1 as i64, f);
        (nw > 0 && nh > 0).then_some(Self {
            level,
            i0,
            j0,
            w: nw as usize,
            h: nh as usize,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.w * self.h
    }

    pub(super) fn has(&self, i: i64, j: i64) -> bool {
        i >= self.i0 && j >= self.j0 && i < self.i0 + self.w as i64 && j < self.j0 + self.h as i64
    }

    pub(super) fn index(&self, i: i64, j: i64) -> usize {
        (j - self.j0) as usize * self.w + (i - self.i0) as usize
    }
}

/// The 2 x 2 box means of `child` over `grid`, a child outside its grid being 0, summed `((c00 + c10) + (c01 + c11)) / 4`.
pub(super) fn box_level(child: &[Plane4], child_grid: &Grid, grid: &Grid) -> Vec<Plane4> {
    let at = |i: i64, j: i64| -> Plane4 {
        if child_grid.has(i, j) {
            child[child_grid.index(i, j)]
        } else {
            [0.0; 4]
        }
    };
    let mut out = Vec::with_capacity(grid.len());
    for y in 0..grid.h {
        for x in 0..grid.w {
            let (i, j) = (grid.i0 + x as i64, grid.j0 + y as i64);
            let (a, b, c, d) = (
                at(2 * i, 2 * j),
                at(2 * i + 1, 2 * j),
                at(2 * i, 2 * j + 1),
                at(2 * i + 1, 2 * j + 1),
            );
            let mut s = [0.0; 4];
            for k in 0..4 {
                s[k] = ((a[k] + b[k]) + (c[k] + d[k])) * 0.25;
            }
            out.push(s);
        }
    }
    out
}

/// The table's nonzero taps `(dx, dy, k)` in row-major order (`dy` then `dx` ascending): the order the exact gather and the device
/// sum in, a weight of 0 left out.
fn taps(table: &Table) -> Vec<(i64, i64, f64)> {
    let side = (2 * table.half + 1) as usize;
    let mut out = Vec::new();
    for dy in -table.half..=table.half {
        for dx in -table.half..=table.half {
            let k = table.weights[(dy + table.half) as usize * side + (dx + table.half) as usize];
            if k != 0.0 {
                out.push((dx, dy, k));
            }
        }
    }
    out
}

/// The `w x h` planes `p` replaced by the table's weighted sums of the pixels around them, a pixel outside the planes being 0:
/// the exact gather (`gather_table`) with the zero weights listed out and the bounds tested only near the edge. The sums are the
/// same bits (the same taps added in the same order).
fn gather_taps(p: &[Plane4], w: usize, h: usize, table: &Table) -> Vec<Plane4> {
    let list = taps(table);
    let half = table.half;
    let offsets: Vec<(isize, f64)> = list
        .iter()
        .map(|&(dx, dy, k)| (dy as isize * w as isize + dx as isize, k))
        .collect();
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 4];
            let inside = x as i64 >= half
                && y as i64 >= half
                && x as i64 + half < w as i64
                && y as i64 + half < h as i64;
            if inside {
                let at = (y * w + x) as isize;
                for &(off, k) in &offsets {
                    let q = &p[(at + off) as usize];
                    for c in 0..4 {
                        acc[c] += k * q[c];
                    }
                }
            } else {
                for &(dx, dy, k) in &list {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let q = &p[yy as usize * w + xx as usize];
                    for c in 0..4 {
                        acc[c] += k * q[c];
                    }
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// The sums `[N (3), D]` of a Lens over a window of weighted planes `[omega X (3), omega]` (`w x h` pixels at `(x0, y0)` of a
/// frame of `frame` pixels), by row segments. `r_px` is the aperture's radius in pixels. At `r_px <= 8` it is the exact
/// kernel's gather, bit for bit.
#[must_use]
pub fn gather(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    frame: (usize, usize),
    params: (f64, u32, f64),
) -> Vec<Plane4> {
    gather_rows(planes, window, frame, params, 0..window.3)
}

/// [`gather`] over the window's rows `rows` only: those rows' pixels, row by row, the same bits as in [`gather`]. The prefix is
/// the whole window's (a pixel reads rows above and below it).
#[must_use]
pub fn gather_rows(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    _frame: (usize, usize),
    (r_px, blades, rotation_deg): (f64, u32, f64),
    rows: std::ops::Range<usize>,
) -> Vec<Plane4> {
    let (_, _, w, h) = window;
    if !uses_rows(r_px) {
        let all = gather_taps(planes, w, h, &aperture_table(r_px, blades, rotation_deg));
        return all[rows.start * w..rows.end * w].to_vec();
    }
    gather_row_segments(planes, w, h, &row_table(r_px, blades, rotation_deg), rows)
}
