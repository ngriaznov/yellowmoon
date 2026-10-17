//! The geometry sampler (`docs/17-gpu.md` section 11): camera RGB sampled through a grid that maps
//! the pixels of a view or an export to positions in a pixel buffer, through the camera's EXIF
//! orientation, the user's quarter turns, the straighten angle and the crop. This is the CPU
//! reference of `shaders/sampler.wgsl` and the implementation `ym-core` calls (a port of its
//! `pipeline::frame::sample_to_camera`, with its `Grid` reduced to plain data, [`SamplerGrid`]).
//!
//! # The model
//!
//! Positions are continuous: pixel `i` covers `[i, i + 1)` and has its centre at `i + 0.5`. An
//! output pixel `(x, y)` samples the buffer at [`SamplerGrid::source`]: the output position goes
//! through the affine `to_canvas` (crop and straighten), is divided by the canvas size, turned by the
//! user's quarter turns and by the EXIF orientation (a signed permutation of the fractions) and
//! scaled to the buffer. A position outside the canvas (beyond a tolerance of 1e-6 px) is *outside
//! the photo*: the pixel is zero and marked in the mask. Three kernels:
//!
//! * [`Kernel::Copy`]: the pixel the grid points at (an exact grid: no straighten angle, one buffer
//!   pixel per output pixel, centres on centres);
//! * [`Kernel::Bicubic`]: Catmull-Rom, for rotation and magnification and shrinking by up to 1.25;
//! * [`Kernel::Box`]`(n)`: bilinear averaged over `n x n` sub-samples per output pixel, for shrinking
//!   by more than 1.25 (`n` from 2 to 4). A sub-sample outside the photo does not count; a pixel with
//!   none is outside.
//!
//! The arithmetic is `f64` for positions and `f32` for samples, as in `ym-core`.
//!
//! # For the device kernel
//!
//! [`SamplerGrid::affine_to_buffer`] collapses the whole position chain into one `f64` affine map from
//! the output position to the buffer position (the dihedral part is a signed permutation, so the
//! composition is affine), and [`SamplerGrid::inside_rows`] gives, per output row and sub-sample, the
//! exact interval of pixels the CPU predicate accepts. The kernel evaluates positions in `f32`
//! relative to the window it reads (the positions are small displacements, never absolute
//! coordinates of order 1e4: D4 of `docs/17-gpu.md`) and takes the inside test from those
//! intervals, so its mask equals the CPU's bit for bit.

use rayon::prelude::*;

use super::RgbImage;
use super::resample::sample_catmull_rom;

/// Largest straighten angle that counts as "none" (degrees).
const ANGLE_EPS_DEG: f64 = 1e-9;

/// The tolerance of the inside test, in canvas pixels: it keeps the edge pixels of an exact grid inside.
const INSIDE_TOL: f64 = 1e-6;

/// A user's clockwise quarter turn count and the camera's EXIF orientation, on fractions of the
/// photo: a point of the canvas to the same spot of the oriented photo.
#[must_use]
pub fn canvas_to_oriented(quarter_turns: u8, [u, v]: [f64; 2]) -> [f64; 2] {
    match quarter_turns % 4 {
        1 => [v, 1.0 - u],
        2 => [1.0 - u, 1.0 - v],
        3 => [1.0 - v, u],
        _ => [u, v],
    }
}

/// A point of the photo as the camera oriented it to the same spot on the sensor (both as fractions).
/// Orientations: 1 upright, 2 mirrored, 3 turned 180 degrees, 4 flipped, 5 transposed, 6 rotated 90
/// degrees clockwise to display, 7 transversed, 8 rotated 90 degrees counter-clockwise. A value
/// outside 1 to 8 is treated as 1.
#[must_use]
pub fn oriented_to_sensor(exif_orientation: u16, [u, v]: [f64; 2]) -> [f64; 2] {
    match exif_orientation {
        2 => [1.0 - u, v],
        3 => [1.0 - u, 1.0 - v],
        4 => [u, 1.0 - v],
        5 => [v, u],
        6 => [v, 1.0 - u],
        7 => [1.0 - v, 1.0 - u],
        8 => [1.0 - v, u],
        _ => [u, v],
    }
}

/// Rounds a coordinate that is within 1e-9 of an integer or half integer, so exact grids stay exact
/// through the floating point maps.
#[inline]
fn snap(v: f64) -> f64 {
    let r = (v * 2.0).round() / 2.0;
    if (v - r).abs() < 1e-9 { r } else { v }
}

/// The sampling kernel a grid asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Read the pixel the grid points at.
    Copy,
    /// Catmull-Rom bicubic.
    Bicubic,
    /// Bilinear averaged over `n x n` sub-samples per output pixel (shrinking).
    Box(usize),
}

impl Kernel {
    /// Sub-samples per axis: 1 for the point kernels, `n` for the box.
    #[must_use]
    pub fn taps(self) -> usize {
        match self {
            Kernel::Box(n) => n,
            _ => 1,
        }
    }
}

/// An affine map `(x, y) -> (a x + b y + c, d x + e y + f)`, row-major `[[a, b, c], [d, e, f]]`.
pub type Affine2 = [[f64; 3]; 2];

fn apply(m: &Affine2, x: f64, y: f64) -> (f64, f64) {
    (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )
}

/// Maps output pixels to positions in a pixel buffer: what `ym-core`'s `Grid` holds, as plain data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerGrid {
    /// Output position to canvas pixels (crop and straighten applied).
    pub to_canvas: Affine2,
    /// The canvas size in pixels (the buffer's, sides swapped by a quarter turn).
    pub canvas: (f64, f64),
    /// The buffer size in pixels.
    pub buffer: (f64, f64),
    /// The camera's EXIF orientation, 1 to 8.
    pub exif_orientation: u16,
    /// The user's clockwise quarter turns.
    pub quarter_turns: u8,
    /// Whether the straighten angle is zero.
    pub unrotated: bool,
}

impl SamplerGrid {
    /// A grid from its parts (the fields of `ym-core`'s `Grid`).
    #[must_use]
    pub fn new(
        to_canvas: Affine2,
        canvas: (f64, f64),
        buffer: (f64, f64),
        exif_orientation: u16,
        quarter_turns: u8,
        unrotated: bool,
    ) -> Self {
        Self {
            to_canvas,
            canvas,
            buffer,
            exif_orientation,
            quarter_turns,
            unrotated,
        }
    }

    /// Whether the straighten angle counts as zero (`angle_deg` in degrees).
    #[must_use]
    pub fn is_unrotated_angle(angle_deg: f64) -> bool {
        angle_deg.abs() < ANGLE_EPS_DEG
    }

    /// The buffer position (continuous, pixel `i` centred at `i + 0.5`) that output pixel `(x, y)`
    /// samples, or `None` when it falls outside the photo.
    #[inline]
    #[must_use]
    pub fn source(&self, x: usize, y: usize) -> Option<(f64, f64)> {
        self.source_at(x as f64 + 0.5, y as f64 + 0.5)
    }

    /// As [`SamplerGrid::source`] for a continuous output position.
    #[inline]
    #[must_use]
    pub fn source_at(&self, ox: f64, oy: f64) -> Option<(f64, f64)> {
        let (px, py) = apply(&self.to_canvas, ox, oy);
        if px < -INSIDE_TOL
            || py < -INSIDE_TOL
            || px > self.canvas.0 + INSIDE_TOL
            || py > self.canvas.1 + INSIDE_TOL
        {
            return None;
        }
        let (u, v) = (px / self.canvas.0, py / self.canvas.1);
        let o = canvas_to_oriented(self.quarter_turns, [u, v]);
        let s = oriented_to_sensor(self.exif_orientation, o);
        Some((
            snap(s[0].clamp(0.0, 1.0) * self.buffer.0),
            snap(s[1].clamp(0.0, 1.0) * self.buffer.1),
        ))
    }

    /// How far one step along the output's x axis and one along its y axis move in the buffer, in
    /// buffer pixels. The canvas has the buffer's pixel size (it only swaps the sides), so these are
    /// the lengths of the columns of the output-to-canvas map, the same everywhere.
    #[must_use]
    pub fn step_lengths(&self) -> (f64, f64) {
        let m = &self.to_canvas;
        (m[0][0].hypot(m[1][0]), m[0][1].hypot(m[1][1]))
    }

    /// Whether the grid is an exact pixel copy: no rotation, one buffer pixel per output pixel and
    /// the output's pixel centres on the buffer's (it may still turn or mirror the image).
    #[must_use]
    pub fn is_pixel_copy(&self) -> bool {
        if !self.unrotated {
            return false;
        }
        let (sx, sy) = self.step_lengths();
        let one = |v: f64| (v - 1.0).abs() < 1e-9;
        let (px, py) = apply(&self.to_canvas, 0.5, 0.5);
        let aligned = |v: f64| (v - 0.5 - (v - 0.5).round()).abs() < 1e-9;
        one(sx) && one(sy) && aligned(px) && aligned(py)
    }

    /// The kernel the grid asks for: a copy for an exact grid, Catmull-Rom up to a shrink of 1.25,
    /// the box filter beyond.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        if self.is_pixel_copy() {
            return Kernel::Copy;
        }
        let (sx, sy) = self.step_lengths();
        let ratio = sx.max(sy);
        if ratio <= 1.25 {
            Kernel::Bicubic
        } else {
            Kernel::Box((ratio - 0.25).ceil().clamp(2.0, 4.0) as usize)
        }
    }

    /// The position chain without the inside test, the clamp or the snap: output position to buffer
    /// position, exactly the arithmetic of [`SamplerGrid::source_at`] otherwise.
    fn raw_at(&self, ox: f64, oy: f64) -> (f64, f64) {
        let (px, py) = apply(&self.to_canvas, ox, oy);
        let (u, v) = (px / self.canvas.0, py / self.canvas.1);
        let o = canvas_to_oriented(self.quarter_turns, [u, v]);
        let s = oriented_to_sensor(self.exif_orientation, o);
        (s[0] * self.buffer.0, s[1] * self.buffer.1)
    }

    /// The whole chain as one affine map from the output position to the buffer position (before the
    /// clamp to the buffer): `buffer = a x + b y + c, d x + e y + f`. The dihedral part is a signed
    /// permutation, so the composition is affine; the coefficients come from three evaluations in
    /// `f64`, and the ones that must be exact on an exact grid are snapped (entries within 1e-9 of an
    /// integer, offsets within 1e-9 of a half integer).
    #[must_use]
    pub fn affine_to_buffer(&self) -> Affine2 {
        let (x0, y0) = self.raw_at(0.0, 0.0);
        let (x1, y1) = self.raw_at(1.0, 0.0);
        let (x2, y2) = self.raw_at(0.0, 1.0);
        let near_int = |v: f64| {
            let r = v.round();
            if (v - r).abs() < 1e-9 { r } else { v }
        };
        [
            [near_int(x1 - x0), near_int(x2 - x0), snap(x0)],
            [near_int(y1 - y0), near_int(y2 - y0), snap(y0)],
        ]
    }

    /// For every output row `y` of the window `[off.0, off.0 + w) x [off.1, off.1 + h)`, and for every
    /// sub-sample `(j, i)` of an `n x n` box (`n` = 1 for the point kernels, whose single sub-sample
    /// is the pixel centre), the inclusive interval of window columns `x` for which
    /// [`SamplerGrid::source_at`] accepts the sub-sample position
    /// `(off.0 + x + (i + 0.5) / n, off.1 + y + (j + 0.5) / n)`. An empty interval is `(1, 0)`.
    /// Row-major: entry `(y * n + j) * n + i`. The intervals are the CPU predicate's own, found
    /// algebraically and then corrected against [`SamplerGrid::source_at`], so a kernel that tests
    /// `lo <= x <= hi` classifies exactly as the CPU does.
    #[must_use]
    pub fn inside_rows(&self, off: (usize, usize), w: usize, h: usize, n: usize) -> Vec<[i32; 2]> {
        let n = n.max(1);
        let mut out = vec![[1, 0]; h * n * n];
        // Serial on purpose: a row is a handful of divisions (microseconds for a whole tile), and the
        // device thread that calls this must not wait for the rayon pool, whose workers may be
        // waiting for the device.
        for (y, cells) in out.chunks_mut(n * n).enumerate() {
            for j in 0..n {
                let oy = (off.1 + y) as f64 + (j as f64 + 0.5) / n as f64;
                for i in 0..n {
                    let fx = (i as f64 + 0.5) / n as f64;
                    cells[j * n + i] = self.row_interval(off.0, w, oy, fx);
                }
            }
        }
        out
    }

    /// The columns of a window starting at output column `off_x` whose sub-sample at offset `fx` is
    /// inside, on the row at output position `oy`.
    fn row_interval(&self, off_x: usize, w: usize, oy: f64, fx: f64) -> [i32; 2] {
        let ox = |x: i64| off_x as f64 + x as f64 + fx;
        let inside = |x: i64| self.source_at(ox(x), oy).is_some();
        let last = w as i64 - 1;
        // The real interval of x from the two linear constraints on the canvas position.
        let m = &self.to_canvas;
        let mut lo = f64::NEG_INFINITY;
        let mut hi = f64::INFINITY;
        let mut constant_ok = true;
        for (a, b, c, size) in [
            (m[0][0], m[0][1], m[0][2], self.canvas.0),
            (m[1][0], m[1][1], m[1][2], self.canvas.1),
        ] {
            // a (off_x + x + fx) + b oy + c in [-tol, size + tol]
            let base = a * (off_x as f64 + fx) + b * oy + c;
            if a.abs() < 1e-12 {
                if base < -INSIDE_TOL || base > size + INSIDE_TOL {
                    constant_ok = false;
                }
                continue;
            }
            let (t0, t1) = ((-INSIDE_TOL - base) / a, (size + INSIDE_TOL - base) / a);
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
        }
        if !constant_ok || lo > hi {
            return [1, 0];
        }
        let clampi = |v: f64| v.clamp(-1.0e9, 1.0e9);
        let mut xl = clampi(lo.ceil()) as i64;
        let mut xh = clampi(hi.floor()) as i64;
        xl = xl.clamp(-1, last + 1);
        xh = xh.clamp(-1, last + 1);
        // Correct against the predicate itself: it is the truth.
        while xl > 0 && inside(xl - 1) {
            xl -= 1;
        }
        while xl <= last && !inside(xl) && xl <= xh + 1 {
            xl += 1;
        }
        while xh < last && inside(xh + 1) {
            xh += 1;
        }
        while xh >= 0 && !inside(xh) && xh >= xl - 1 {
            xh -= 1;
        }
        let (xl, xh) = (xl.max(0), xh.min(last));
        if xl > xh || !inside(xl) {
            [1, 0]
        } else {
            [xl as i32, xh as i32]
        }
    }

    /// The bounding box `(x0, y0, x1, y1)` (exclusive ends) of the pixels of the window
    /// `[off.0, off.0 + w) x [off.1, off.1 + h)` that have a position inside the photo
    /// ([`Sampled::outside`] is 0 there), or `None` when there is none. Found row by row from the
    /// exact intervals, without touching a pixel: the rectangle a device frame hands to the output
    /// sharpening ("the stage works on the bounding box of the pixels inside the photo").
    #[must_use]
    pub fn inside_bounds(
        &self,
        off: (usize, usize),
        w: usize,
        h: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let n = self.kernel().taps();
        let rows = self.inside_rows(off, w, h, n);
        let mut b: Option<(usize, usize, usize, usize)> = None;
        for y in 0..h {
            let cells = &rows[y * n * n..(y + 1) * n * n];
            // A pixel is inside when any of its sub-samples is: the union of the intervals (their
            // hull, since the sub-samples are within one pixel of each other).
            let mut lo = i32::MAX;
            let mut hi = i32::MIN;
            for c in cells.iter().filter(|c| c[0] <= c[1]) {
                lo = lo.min(c[0]);
                hi = hi.max(c[1]);
            }
            if lo > hi {
                continue;
            }
            let (lo, hi) = (lo as usize, hi as usize);
            b = Some(match b {
                None => (lo, y, hi + 1, y + 1),
                Some((x0, y0, x1, _)) => (x0.min(lo), y0, x1.max(hi + 1), y + 1),
            });
        }
        b
    }
}

/// Camera RGB sampled through a grid, with a mask of the pixels that fell outside the photo.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    /// The sampled pixels (zero outside the photo).
    pub img: RgbImage,
    /// 1 where the pixel is outside the photo; `None` when every pixel is inside.
    pub outside: Option<Vec<u8>>,
}

#[inline]
fn bilinear(plane: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> f32 {
    let (ux, uy) = (
        (sx - 0.5).clamp(0.0, w as f64 - 1.0),
        (sy - 0.5).clamp(0.0, h as f64 - 1.0),
    );
    let (x0, y0) = (ux.floor() as usize, uy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((ux - x0 as f64) as f32, (uy - y0 as f64) as f32);
    let a = plane[y0 * w + x0] * (1.0 - fx) + plane[y0 * w + x1] * fx;
    let b = plane[y1 * w + x0] * (1.0 - fx) + plane[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

/// Samples the `w x h` output pixels at offset `off` of `grid` from `src`, an image whose pixel
/// `(0, 0)` is buffer pixel `origin`.
#[must_use]
pub fn sample_to_camera(
    src: &RgbImage,
    origin: (usize, usize),
    grid: &SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Sampled {
    let kernel = grid.kernel();
    let mut img = RgbImage::new(w, h);
    let mut outside = vec![0u8; w * h];
    let n = w * h;
    let (rp, rest) = img.data.split_at_mut(n);
    let (gp, bp) = rest.split_at_mut(n);
    let (sw, sh) = (src.width, src.height);
    let planes = [src.plane(0), src.plane(1), src.plane(2)];
    let (ox, oy) = (origin.0 as f64, origin.1 as f64);
    (
        rp.par_chunks_mut(w.max(1)),
        gp.par_chunks_mut(w.max(1)),
        bp.par_chunks_mut(w.max(1)),
        outside.par_chunks_mut(w.max(1)),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(y, (r, g, b, out))| {
            let rows = [r, g, b];
            for x in 0..w {
                let (gx, gy) = (off.0 + x, off.1 + y);
                match kernel {
                    Kernel::Copy | Kernel::Bicubic => {
                        let Some((sx, sy)) = grid.source(gx, gy) else {
                            out[x] = 1;
                            continue;
                        };
                        let (px, py) = (sx - ox, sy - oy);
                        for c in 0..3 {
                            rows[c][x] = if kernel == Kernel::Copy {
                                let ix = (px.floor() as isize).clamp(0, sw as isize - 1) as usize;
                                let iy = (py.floor() as isize).clamp(0, sh as isize - 1) as usize;
                                planes[c][iy * sw + ix]
                            } else {
                                sample_catmull_rom(planes[c], sw, sh, px, py)
                            };
                        }
                    }
                    Kernel::Box(taps) => {
                        let mut acc = [0.0f32; 3];
                        let mut count = 0u32;
                        for j in 0..taps {
                            for i in 0..taps {
                                let (fx, fy) = (
                                    gx as f64 + (i as f64 + 0.5) / taps as f64,
                                    gy as f64 + (j as f64 + 0.5) / taps as f64,
                                );
                                if let Some((sx, sy)) = grid.source_at(fx, fy) {
                                    for c in 0..3 {
                                        acc[c] += bilinear(planes[c], sw, sh, sx - ox, sy - oy);
                                    }
                                    count += 1;
                                }
                            }
                        }
                        if count == 0 {
                            out[x] = 1;
                        } else {
                            for c in 0..3 {
                                rows[c][x] = acc[c] / count as f32;
                            }
                        }
                    }
                }
            }
        });
    let any_outside = outside.iter().any(|&v| v != 0);
    Sampled {
        img,
        outside: any_outside.then_some(outside),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid for an `ow x oh` output of a `bw x bh` buffer with the given orientation and turns, no
    /// straightening, one buffer pixel per output pixel (the canvas is the buffer, sides swapped by
    /// the turns).
    fn exact_grid(bw: usize, bh: usize, exif: u16, turns: u8) -> (SamplerGrid, usize, usize) {
        let swaps = ((5..=8).contains(&exif)) != (turns % 2 == 1);
        let (cw, ch) = if swaps { (bh, bw) } else { (bw, bh) };
        let grid = SamplerGrid::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            (cw as f64, ch as f64),
            (bw as f64, bh as f64),
            exif,
            turns,
            true,
        );
        (grid, cw, ch)
    }

    /// Independent forward maps of the 8 EXIF orientations on a `w x h` buffer: the pixel of the
    /// oriented image that buffer pixel `(x, y)` lands on (the same table `ym-core`'s geometry tests
    /// use).
    fn forward(exif: u16, x: usize, y: usize, w: usize, h: usize) -> (usize, usize) {
        match exif {
            2 => (w - 1 - x, y),
            3 => (w - 1 - x, h - 1 - y),
            4 => (x, h - 1 - y),
            5 => (y, x),
            6 => (h - 1 - y, x),
            7 => (h - 1 - y, w - 1 - x),
            8 => (y, w - 1 - x),
            _ => (x, y),
        }
    }

    #[test]
    fn an_exact_grid_is_a_copy_for_all_orientations_and_turns() {
        let (bw, bh) = (7usize, 4usize);
        let mut img = RgbImage::new(bw, bh);
        for y in 0..bh {
            for x in 0..bw {
                img.set_pixel(x, y, [x as f32, y as f32, (x * 10 + y) as f32]);
            }
        }
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                let (grid, cw, ch) = exact_grid(bw, bh, exif, turns);
                assert_eq!(grid.kernel(), Kernel::Copy, "exif {exif} turns {turns}");
                let s = sample_to_camera(&img, (0, 0), &grid, (0, 0), cw, ch);
                assert!(s.outside.is_none());
                // Every buffer pixel lands where the independent forward maps say.
                for y in 0..bh {
                    for x in 0..bw {
                        let (mut ox, mut oy) = forward(exif, x, y, bw, bh);
                        let (mut w, mut h) = if (5..=8).contains(&exif) {
                            (bh, bw)
                        } else {
                            (bw, bh)
                        };
                        for _ in 0..turns {
                            // One more clockwise quarter turn: (x, y) -> (h - 1 - y, x).
                            (ox, oy) = (h - 1 - oy, ox);
                            (w, h) = (h, w);
                        }
                        assert_eq!((w, h), (cw, ch));
                        assert_eq!(
                            s.img.pixel(ox, oy),
                            img.pixel(x, y),
                            "exif {exif} turns {turns} pixel ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_collapsed_affine_reproduces_the_position_chain() {
        // Rotated and scaled grids on every orientation: the affine equals `source_at` up to the
        // clamp and the snap.
        let (bw, bh) = (640.0, 480.0);
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                let swaps = ((5..=8).contains(&exif)) != (turns % 2 == 1);
                let (cw, ch) = if swaps { (bh, bw) } else { (bw, bh) };
                let (s, c) = 0.2f64.sin_cos();
                let grid = SamplerGrid::new(
                    [
                        [0.7 * c, -0.7 * s, 33.3 + cw * 0.1],
                        [0.7 * s, 0.7 * c, 21.7 + ch * 0.1],
                    ],
                    (cw, ch),
                    (bw, bh),
                    exif,
                    turns,
                    false,
                );
                let m = grid.affine_to_buffer();
                for (ox, oy) in [(0.5, 0.5), (100.25, 40.75), (399.5, 299.5), (17.0, 311.0)] {
                    let Some((bx, by)) = grid.source_at(ox, oy) else {
                        continue;
                    };
                    let (ax, ay) = apply(&m, ox, oy);
                    assert!((ax.clamp(0.0, bw) - bx).abs() < 5e-9, "x {ax} vs {bx}");
                    assert!((ay.clamp(0.0, bh) - by).abs() < 5e-9, "y {ay} vs {by}");
                }
            }
        }
    }

    #[test]
    fn the_inside_rows_are_the_predicates_own() {
        // Compare the intervals with a brute-force scan of the predicate, on rotated and cropped
        // grids, for the point kernels and for box sub-samples.
        let (bw, bh) = (200.0, 150.0);
        for (angle, scale, n) in [(0.0, 1.0, 1), (0.3, 0.8, 1), (-0.7, 1.9, 2), (1.2, 2.6, 3)] {
            let (s, c) = f64::sin_cos(angle);
            let grid = SamplerGrid::new(
                [
                    [scale * c, -scale * s, -40.0],
                    [scale * s, scale * c, -25.5],
                ],
                (bw, bh),
                (bw, bh),
                1,
                0,
                SamplerGrid::is_unrotated_angle(angle),
            );
            let (off, w, h) = ((3usize, 5usize), 160usize, 120usize);
            let rows = grid.inside_rows(off, w, h, n);
            for y in 0..h {
                for j in 0..n {
                    for i in 0..n {
                        let iv = rows[(y * n + j) * n + i];
                        for x in 0..w {
                            let inside = grid
                                .source_at(
                                    (off.0 + x) as f64 + (i as f64 + 0.5) / n as f64,
                                    (off.1 + y) as f64 + (j as f64 + 0.5) / n as f64,
                                )
                                .is_some();
                            let claimed = iv[0] <= x as i32 && x as i32 <= iv[1];
                            assert_eq!(
                                inside, claimed,
                                "angle {angle} scale {scale} row {y} sub ({j}, {i}) x {x}: {iv:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_inside_bounds_are_the_masks() {
        let (bw, bh) = (120.0, 90.0);
        let (s, c) = f64::sin_cos(0.35);
        let grid = SamplerGrid::new(
            [[c, -s, -30.0], [s, c, -10.0]],
            (bw, bh),
            (bw, bh),
            1,
            0,
            false,
        );
        let src = RgbImage::new(120, 90);
        let (w, h) = (150usize, 130usize);
        let sampled = sample_to_camera(&src, (0, 0), &grid, (0, 0), w, h);
        let mask = sampled.outside.expect("a rotated grid has empty corners");
        let mut want: Option<(usize, usize, usize, usize)> = None;
        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] == 0 {
                    want = Some(match want {
                        None => (x, y, x + 1, y + 1),
                        Some((x0, y0, x1, y1)) => {
                            (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1))
                        }
                    });
                }
            }
        }
        assert_eq!(grid.inside_bounds((0, 0), w, h), want);
    }
}
