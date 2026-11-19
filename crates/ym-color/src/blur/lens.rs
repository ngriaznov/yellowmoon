//! Lens, the fast method (`docs/33-blur.md` section 3.5, BL-07): the coverage of an aperture over **cells**, in f64.
//!
//! The exact kernel ([`super::aperture_table`]) is a direct sum over up to `(2 R + 3)^2` pixels: 235 000 taps at `R = 240`
//! px. The fast method is the Film node's coarse grid (`docs/24` section 3.9a) with a table in place of a Gaussian:
//!
//! 1. the weighted planes `omega X` and `omega` are reduced to the mean of each `f x f` block, **at absolute multiples of
//!    `f`** (so a cell is the same number in every tile), by `level` rounds of 2 x 2 box means; `f = 2^level` is the smallest
//!    cell with `R / f <= 8` ([`level_for`], [`MAX_CELL_RADIUS`]);
//! 2. every cell is replaced by the table's weighted sum of the cells around it: the **coverage of a cell by the aperture**
//!    (8 x 8 supersampled, f64, a function of `R / f`, the blades and the rotation alone: [`cell_table`]), at most 17 x 17
//!    cells and about 200 nonzero;
//! 3. a pixel reads its four nearest blurred cells bilinearly (a cell's centre at `(I + 1/2) f`).
//!
//! The reads soften the aperture's edge by about `f / 2` (at most `R / 16`), which is the method's error ([`super`] holds it
//! to `docs/33` section 3.7's bound against the exact kernel). With `R <= 8` the cell is the pixel (`level` 0) and the method
//! **is** the exact kernel, bit for bit. A cell outside the frame has no weight (it is not in the grid) and a cell the frame's
//! edge cuts counts its missing pixels as 0, so the picture's edge renormalises as the exact kernel's does.
//!
//! The device does the same in f32 (`ym_gpu::kernels::blur_lens`); the arithmetic here is written in the device's order so
//! that the two agree to rounding.

use super::{Plane4, Table, aperture_table};

/// The aperture's radius in cells at most: the table is at most 17 x 17 cells (`docs/33` section 3.5, `R / f <= 8`).
pub const MAX_CELL_RADIUS: f64 = 8.0;

/// The level `L` of the cell size `f = 2^L`: the smallest with `r_px / f <= 8` (0 when `r_px <= 8`).
#[must_use]
pub fn level_for(r_px: f64) -> u32 {
    // A radius that is NaN or at most 8 px is level 0.
    if r_px.is_nan() || r_px <= MAX_CELL_RADIUS {
        return 0;
    }
    let mut level = 0;
    while r_px / f64::from(1u32 << level) > MAX_CELL_RADIUS && level < 30 {
        level += 1;
    }
    level
}

/// The cell size `f = 2^level` in pixels of the aperture of radius `r_px`.
#[must_use]
pub fn cell_size(r_px: f64) -> usize {
    1usize << level_for(r_px)
}

/// The weights of the aperture over cells at `level`: the coverage of each `f x f` cell by the aperture of radius `r_px / f`
/// centred on a cell's centre, normalised to sum 1 and **trimmed** to the square that holds every nonzero weight (so at most
/// 17 x 17). At level 0 it is the exact table's weights, trimmed.
#[must_use]
pub fn cell_table(r_px: f64, blades: u32, rotation_deg: f64, level: u32) -> Table {
    let t = aperture_table(r_px / f64::from(1u32 << level), blades, rotation_deg);
    let side = (2 * t.half + 1) as usize;
    let mut half = 0i64;
    for dy in -t.half..=t.half {
        for dx in -t.half..=t.half {
            if t.weights[(dy + t.half) as usize * side + (dx + t.half) as usize] != 0.0 {
                half = half.max(dx.abs()).max(dy.abs());
            }
        }
    }
    if half == t.half {
        return t;
    }
    let ns = (2 * half + 1) as usize;
    let mut weights = Vec::with_capacity(ns * ns);
    for dy in -half..=half {
        for dx in -half..=half {
            weights.push(t.weights[(dy + t.half) as usize * side + (dx + t.half) as usize]);
        }
    }
    Table { half, weights }
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

    fn len(&self) -> usize {
        self.w * self.h
    }

    fn has(&self, i: i64, j: i64) -> bool {
        i >= self.i0 && j >= self.j0 && i < self.i0 + self.w as i64 && j < self.j0 + self.h as i64
    }

    fn index(&self, i: i64, j: i64) -> usize {
        (j - self.j0) as usize * self.w + (i - self.i0) as usize
    }
}

/// The 2 x 2 box means of `child` over `grid`, a child outside its grid being 0, summed `((c00 + c10) + (c01 + c11)) / 4`.
fn box_level(child: &[Plane4], child_grid: &Grid, grid: &Grid) -> Vec<Plane4> {
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

/// The pixels of a window read back from `cells` bilinearly (a cell's centre at `(I + 1/2) f`), a cell outside the grid being 0.
fn read_back(cells: &[Plane4], grid: &Grid, window: (usize, usize, usize, usize)) -> Vec<Plane4> {
    let (x0, y0, w, h) = window;
    let f = (1u64 << grid.level) as f64;
    let cell = |i: i64, j: i64| -> Plane4 {
        if grid.has(i, j) {
            cells[grid.index(i, j)]
        } else {
            [0.0; 4]
        }
    };
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = ((x0 + x) as f64, (y0 + y) as f64);
            let u = (px + 0.5) / f - 0.5;
            let v = (py + 0.5) / f - 0.5;
            let (i0, j0) = (u.floor() as i64, v.floor() as i64);
            let (tx, ty) = (u - i0 as f64, v - j0 as f64);
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
            out.push(s);
        }
    }
    out
}

/// The sums `[N (3), D]` of a Lens over a window of weighted planes `[omega X (3), omega]` (`w x h` pixels at `(x0, y0)` of a
/// frame of `frame` pixels), by the cell method. `r_px` is the aperture's radius in pixels. At `r_px <= 8` it is the exact
/// kernel's gather, bit for bit.
#[must_use]
pub fn gather(
    planes: &[Plane4],
    window: (usize, usize, usize, usize),
    frame: (usize, usize),
    (r_px, blades, rotation_deg): (f64, u32, f64),
) -> Vec<Plane4> {
    let (_, _, w, h) = window;
    let level = level_for(r_px);
    if level == 0 {
        return gather_taps(planes, w, h, &aperture_table(r_px, blades, rotation_deg));
    }
    let mut cells = planes.to_vec();
    let mut grid = Grid {
        level: 0,
        i0: window.0 as i64,
        j0: window.1 as i64,
        w,
        h,
    };
    for l in 1..=level {
        let Some(up) = Grid::of(window, frame, l) else {
            // No whole cell at this size in the window: nothing is read (the window is smaller than a cell).
            return vec![[0.0; 4]; w * h];
        };
        cells = box_level(&cells, &grid, &up);
        grid = up;
    }
    let table = cell_table(r_px, blades, rotation_deg, level);
    let blurred = gather_taps(&cells, grid.w, grid.h, &table);
    read_back(&blurred, &grid, window)
}
