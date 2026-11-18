//! The Blur node's shared passes on the CPU (`docs/33-blur.md` sections 3.2, 3.3 and 6): the twins of the scaffold's device
//! passes (`crate::kernels::blur_stack`), in `f64`, on dense planar planes. Each is the arithmetic of `ym_color::blur` (the
//! weight, the blend) and of `ym_color::logc` (the decode and the encode); the device does the same in `f32`.
//!
//! * [`prep`]: decode to linear AWG4, the highlight weight, the weight `omega` and the planes `omega X` and `omega`;
//! * [`pyramid_level`]: the 2 x 2 box means of a level at absolute alignment (the coarse grid is one level of it);
//! * [`upsample`]: the nearest or bilinear read-back of a level to pixels;
//! * [`blend`]: the blend in linear light and the encode, with the exact `m = 0` and `m = 1` branches.
//!
//! The cell grid ([`CellGrid`]) is shared with the device: which cells of a level exist is a function of the read area and
//! the frame alone.

use ym_color::blur::{EPS, blend_linear, highlight_weight, omega};
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_color::mask_raster::Area;

/// A grid of cells of `f x f` pixels, `f = 2^level`, at absolute multiples of `f` (`docs/33` section 3.5): cell `(i, j)` holds
/// the pixels `[i f, (i + 1) f) x [j f, (j + 1) f)`. The grid holds the cells whose pixels the read area covers wherever the frame
/// has any (a cell cut by the frame's edge counts its missing pixels as 0), so every cell of it is the same number in every
/// tile that holds it. Level 0 is the pixels themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellGrid {
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

/// The cells of one axis: the first index and the count, for a read range `[r0, r1)` of a frame range `[f0, f1)`.
fn axis(r0: i64, r1: i64, f0: i64, f1: i64, f: i64) -> (i64, i64) {
    let ceil = |a: i64| -((-a).div_euclid(f));
    // A cell is whole when the part of its pixels inside the frame lies inside the read range: it may start before the
    // frame's first pixel when the read range starts there, and end after the frame's last one when it ends there.
    let lo = if r0 <= f0 { f0.div_euclid(f) } else { ceil(r0) };
    let hi = if r1 >= f1 { ceil(f1) } else { r1.div_euclid(f) };
    (lo, hi - lo)
}

impl CellGrid {
    /// The grid of `level` for the read area `read` of the frame `frame` (absolute pixels, `read` inside `frame`), or `None`
    /// when no cell of the level is whole in it.
    #[must_use]
    pub fn of(read: Area, frame: Area, level: u32) -> Option<Self> {
        if read.is_empty() || level > 30 {
            return None;
        }
        let f = 1i64 << level;
        let (i0, w) = axis(read.x, read.x1(), frame.x, frame.x1(), f);
        let (j0, h) = axis(read.y, read.y1(), frame.y, frame.y1(), f);
        (w > 0 && h > 0).then_some(Self {
            level,
            i0,
            j0,
            w: w as usize,
            h: h as usize,
        })
    }

    /// The side of a cell in pixels.
    #[must_use]
    pub fn side(&self) -> i64 {
        1i64 << self.level
    }

    /// The pixels the cells cover (a cell cut by the frame's edge covers its missing pixels too).
    #[must_use]
    pub fn pixels(&self) -> Area {
        let f = self.side();
        Area {
            x: self.i0 * f,
            y: self.j0 * f,
            w: self.w * f as usize,
            h: self.h * f as usize,
        }
    }

    /// Whether the cell `(i, j)` is in the grid.
    #[must_use]
    pub fn has(&self, i: i64, j: i64) -> bool {
        i >= self.i0 && j >= self.j0 && i < self.i0 + self.w as i64 && j < self.j0 + self.h as i64
    }

    /// The number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.w * self.h
    }

    /// Whether the grid has no cell (a grid is never empty: [`CellGrid::of`] gives `None` then).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// The weighted planes of the pixels (`docs/33` section 3.2): `logc4` (three planes of `w x h`, LogC4 / AWG4) and the node's
/// mask `m` (`w x h`) become four planes, `omega X` and `omega`. `highlights` is Lens' control, 0 to 100 (0 for the other types);
/// `respect` is Respect mask edges.
#[must_use]
pub fn prep(
    logc4: &[f64],
    m: &[f64],
    w: usize,
    h: usize,
    highlights: f64,
    respect: bool,
) -> Vec<f64> {
    let n = w * h;
    let mut out = vec![0.0; 4 * n];
    for i in 0..n {
        let x = logc4_decode_rgb([logc4[i], logc4[n + i], logc4[2 * n + i]]);
        let mv = if m[i].is_nan() {
            0.0
        } else {
            m[i].clamp(0.0, 1.0)
        };
        let wt = omega(mv, highlight_weight(x, highlights), respect);
        out[i] = x[0] * wt;
        out[n + i] = x[1] * wt;
        out[2 * n + i] = x[2] * wt;
        out[3 * n + i] = wt;
    }
    out
}

/// One level of the pyramid: the cells of `grid` as the mean of their 2 x 2 children in `child` (four planes over `child_grid`,
/// the level below), a child outside its grid being 0. The sum is `((c00 + c10) + (c01 + c11)) / 4`, the device's order.
#[must_use]
pub fn pyramid_level(child: &[f64], child_grid: &CellGrid, grid: &CellGrid) -> Vec<f64> {
    let (cn, n) = (child_grid.len(), grid.len());
    let at = |i: i64, j: i64, c: usize| -> f64 {
        if child_grid.has(i, j) {
            child[c * cn
                + (j - child_grid.j0) as usize * child_grid.w
                + (i - child_grid.i0) as usize]
        } else {
            0.0
        }
    };
    let mut out = vec![0.0; 4 * n];
    for c in 0..4 {
        for y in 0..grid.h {
            for x in 0..grid.w {
                let (i, j) = (grid.i0 + x as i64, grid.j0 + y as i64);
                let s = ((at(2 * i, 2 * j, c) + at(2 * i + 1, 2 * j, c))
                    + (at(2 * i, 2 * j + 1, c) + at(2 * i + 1, 2 * j + 1, c)))
                    * 0.25;
                out[c * n + y * grid.w + x] = s;
            }
        }
    }
    out
}

/// The whole pyramid from level 0 (`base`, four planes over `base_grid`) up to `levels`: the grid and the planes of each level,
/// level 0 first. Stops early (a shorter list) when a level has no whole cell.
#[must_use]
pub fn pyramid(
    base: &[f64],
    base_grid: CellGrid,
    frame: Area,
    levels: u32,
) -> Vec<(CellGrid, Vec<f64>)> {
    let mut out = vec![(base_grid, base.to_vec())];
    for level in 1..=levels {
        let Some(grid) = CellGrid::of(base_grid.pixels(), frame, level) else {
            break;
        };
        let (child_grid, child) = &out[level as usize - 1];
        let planes = pyramid_level(child, child_grid, &grid);
        out.push((grid, planes));
    }
    out
}

/// The read-back of the cells of `grid` (four planes) to the pixels of `out`: the cell that holds the pixel (`bilinear` false) or
/// the four cells around its centre, a cell's centre at `(I + 1/2) f`. A cell outside the grid is 0. Four planes over `out`.
#[must_use]
pub fn upsample(cells: &[f64], grid: &CellGrid, out: Area, bilinear: bool) -> Vec<f64> {
    let (cn, n) = (grid.len(), out.w * out.h);
    let cell = |i: i64, j: i64, c: usize| -> f64 {
        if grid.has(i, j) {
            cells[c * cn + (j - grid.j0) as usize * grid.w + (i - grid.i0) as usize]
        } else {
            0.0
        }
    };
    let f = grid.side() as f64;
    let mut res = vec![0.0; 4 * n];
    for c in 0..4 {
        for y in 0..out.h {
            for x in 0..out.w {
                let (px, py) = (out.x + x as i64, out.y + y as i64);
                let v = if bilinear {
                    let u = (px as f64 + 0.5) / f - 0.5;
                    let w = (py as f64 + 0.5) / f - 0.5;
                    let (i0, j0) = (u.floor() as i64, w.floor() as i64);
                    let (tx, ty) = (u - i0 as f64, w - j0 as f64);
                    let top = (1.0 - tx) * cell(i0, j0, c) + tx * cell(i0 + 1, j0, c);
                    let bottom = (1.0 - tx) * cell(i0, j0 + 1, c) + tx * cell(i0 + 1, j0 + 1, c);
                    (1.0 - ty) * top + ty * bottom
                } else {
                    cell(px >> grid.level, py >> grid.level, c)
                };
                res[c * n + y * out.w + x] = v;
            }
        }
    }
    res
}

/// The Lens gather over cells (`docs/33` section 3.5): each cell of `out` (a sub-grid of `grid`, the same level) becomes the
/// table's weighted sum of the cells around it, a cell outside `grid` being 0. `cells` is four planes over `grid`; `table` is
/// `(2 half + 1)^2` weights, row-major, a weight of 0 skipped, `dy` then `dx` ascending: the device's order. Four planes over
/// `out`.
#[must_use]
pub fn lens_cells(
    cells: &[f64],
    grid: &CellGrid,
    (half, table): (i64, &[f64]),
    out: &CellGrid,
) -> Vec<f64> {
    let (gn, on) = (grid.len(), out.len());
    let side = (2 * half + 1) as usize;
    let mut res = vec![0.0; 4 * on];
    for y in 0..out.h {
        for x in 0..out.w {
            let (i, j) = (out.i0 + x as i64, out.j0 + y as i64);
            let mut acc = [0.0; 4];
            for dy in -half..=half {
                for dx in -half..=half {
                    let (ii, jj) = (i + dx, j + dy);
                    if !grid.has(ii, jj) {
                        continue;
                    }
                    let k = table[(dy + half) as usize * side + (dx + half) as usize];
                    if k == 0.0 {
                        continue;
                    }
                    let at = (jj - grid.j0) as usize * grid.w + (ii - grid.i0) as usize;
                    for c in 0..4 {
                        acc[c] += k * cells[c * gn + at];
                    }
                }
            }
            for c in 0..4 {
                res[c * on + y * out.w + x] = acc[c];
            }
        }
    }
    res
}

/// The picture a node makes over `made` (`docs/33` section 3.3): `logc4` and `m` are over `read` (three planes and one), `sums`
/// is `N` and `D` over `made` (four planes). The pixel's own bits where `m = 0`; else the blend in linear light and the
/// encode. Three planes over `made`.
#[must_use]
pub fn blend(logc4: &[f64], m: &[f64], read: Area, sums: &[f64], made: Area) -> Vec<f64> {
    let (rn, n) = (read.w * read.h, made.w * made.h);
    let mut out = vec![0.0; 3 * n];
    for y in 0..made.h {
        for x in 0..made.w {
            let (ax, ay) = (made.x + x as i64, made.y + y as i64);
            let r = (ay - read.y) as usize * read.w + (ax - read.x) as usize;
            let v = [logc4[r], logc4[rn + r], logc4[2 * rn + r]];
            let mv = if m[r].is_nan() {
                0.0
            } else {
                m[r].clamp(0.0, 1.0)
            };
            let i = y * made.w + x;
            let o = if mv == 0.0 {
                v
            } else {
                let sum = [sums[i], sums[n + i], sums[2 * n + i]];
                let lin = blend_linear(logc4_decode_rgb(v), mv, sum, sums[3 * n + i]);
                logc4_encode_rgb(lin)
            };
            out[i] = o[0];
            out[n + i] = o[1];
            out[2 * n + i] = o[2];
        }
    }
    out
}

/// The `D + EPS` the blend divides by (`ym_color::blur::EPS`), for the tests.
pub const BLEND_EPS: f64 = EPS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_zero_is_the_read_area_and_higher_levels_hold_whole_cells() {
        let frame = Area::whole(100, 80);
        let read = Area {
            x: 10,
            y: 20,
            w: 50,
            h: 30,
        };
        let g0 = CellGrid::of(read, frame, 0).expect("level 0");
        assert_eq!((g0.i0, g0.j0, g0.w, g0.h), (10, 20, 50, 30));
        // Level 3 (8 px): the cells inside [10, 60) x [20, 50): columns 2..7 (16..56), rows 3..6 (24..48).
        let g3 = CellGrid::of(read, frame, 3).expect("level 3");
        assert_eq!((g3.i0, g3.j0, g3.w, g3.h), (2, 3, 5, 3));
        assert!(read.contains(&g3.pixels()));
    }

    #[test]
    fn a_read_area_that_touches_the_frame_keeps_the_cut_cells() {
        // The read area is the whole 100 x 80 frame: a cell of 32 that the frame's edge cuts is in the grid.
        let frame = Area::whole(100, 80);
        let g = CellGrid::of(frame, frame, 5).expect("level 5");
        assert_eq!((g.i0, g.j0, g.w, g.h), (0, 0, 4, 3));
        assert_eq!(g.pixels().x1(), 128);
    }

    #[test]
    fn a_level_with_no_whole_cell_has_no_grid() {
        let frame = Area::whole(100, 80);
        let read = Area {
            x: 3,
            y: 3,
            w: 10,
            h: 10,
        };
        assert!(CellGrid::of(read, frame, 4).is_none());
    }

    #[test]
    fn the_pyramid_is_the_box_mean_of_the_block() {
        let frame = Area::whole(16, 16);
        let g0 = CellGrid::of(frame, frame, 0).expect("grid");
        let base: Vec<f64> = (0..4 * 256).map(|i| f64::from(i % 97) * 0.5).collect();
        let levels = pyramid(&base, g0, frame, 2);
        assert_eq!(levels.len(), 3);
        let (g2, l2) = &levels[2];
        for j in 0..g2.h {
            for i in 0..g2.w {
                for c in 0..4 {
                    let mut s = 0.0;
                    for y in 0..4 {
                        for x in 0..4 {
                            s += base[c * 256 + (4 * j + y) * 16 + 4 * i + x];
                        }
                    }
                    assert!((l2[c * g2.len() + j * g2.w + i] - s / 16.0).abs() < 1e-12);
                }
            }
        }
    }
}
