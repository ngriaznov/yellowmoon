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
//! the photo*: the pixel is zero and marked in the mask.
//!
//! # The lens's bend (L4, `docs/21` section 4)
//!
//! A grid may carry a [`SamplerWarp`]: the anamorphic bend, the lens's distortion and the zoom. It
//! sits between the canvas position and the division by the canvas size: the canvas position is
//! replaced by the photo position it reads, `F^-1(c / z)` in half-diagonal units about the canvas
//! centre ([`SamplerWarp::canvas_to_photo`], three Newton steps in `f64`). The inside test is on the
//! canvas position, before the warp; the position it gives is clamped to the buffer as before. A grid
//! with a warp is never a pixel copy, and its kernel is chosen by the largest shrink over the canvas
//! ([`SamplerGrid::kernel`]). Three kernels:
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
//! [`SamplerGrid::affine_to_buffer`] collapses the position chain, without the warp, into one `f64`
//! affine map from the output position to the buffer position (the dihedral part is a signed
//! permutation, so the composition is affine), and [`SamplerGrid::inside_rows`] gives, per output row
//! and sub-sample, the exact interval of pixels the CPU predicate accepts. The kernel evaluates
//! positions in `f32` relative to the window it reads (the positions are small displacements, never
//! absolute coordinates of order 1e4: D4 of `docs/17-gpu.md`) and takes the inside test from those
//! intervals, so its mask equals the CPU's bit for bit. With a warp the kernel adds the warp's
//! displacement to that exact position ([`SamplerGrid::warp_frame`] gives it the linear part of the
//! dihedral map and the canvas centre).

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

/// The inverse of an affine map, or `None` for a singular one.
fn invert(m: &Affine2) -> Option<Affine2> {
    let [[a, b, c], [d, e, f]] = *m;
    let det = a * e - b * d;
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
    Some([[ia, ib, -(ia * c + ib * f)], [id, ie, -(id * c + ie * f)]])
}

/// The part `[t0, t1]` of the segment `a + t (b - a)`, `t` in `[0, 1]`, that lies in the box
/// `lo..=hi`, or `None` when none does (Liang-Barsky).
fn clip_segment(
    a: (f64, f64),
    b: (f64, f64),
    lo: (f64, f64),
    hi: (f64, f64),
) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (start, step, lo, hi) in [(a.0, b.0 - a.0, lo.0, hi.0), (a.1, b.1 - a.1, lo.1, hi.1)] {
        if step.abs() < 1e-12 {
            if start < lo || start > hi {
                return None;
            }
        } else {
            let (u, v) = ((lo - start) / step, (hi - start) / step);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// The lens's bend as the sampler reads it (L4, `docs/21` section 4): the anamorphic attachment's
/// bend (ten constants), the lens's own radial distortion and the zoom that keeps the corners filled.
/// A few numbers of plain data, `Copy`, so it can sit in a grid and in a cache key.
///
/// The map `F` takes a photo position to a canvas position, in half-diagonal units about the canvas
/// centre with `x` along the anamorphic axis (the canvas's horizontal, or its vertical when
/// [`SamplerWarp::axis_y`] is set):
///
/// ```text
/// r' = r (1 + d1 r^2 + d2 r^4)        (x1, y1) = (x, y) r'/r
/// x2 = x1 (1 + a1 x1^2 + a2 y1^2 + a3 x1^4 + a4 x1^2 y1^2 + a5 y1^4)
/// y2 = y1 (1 + b1 x1^2 + b2 y1^2 + b3 x1^4 + b4 x1^2 y1^2 + b5 y1^4)
/// ```
///
/// and the canvas is `z F(p)`. Sampling needs the inverse: a canvas position `c` reads the photo at
/// `F^-1(c / z)`, found by [`SamplerWarp::steps`] Newton steps from `c / z` with the 2 x 2 Jacobian, in
/// `f64`. The operations are those of `ym_lens::LensWarp` (`+ - * /` and one `sqrt`, in the same
/// order), so the two give the same bits; `ym-core` builds this from a `LensWarp` and a test holds
/// them equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerWarp {
    /// `a1 .. a5` (already times `q - 1`): the terms of `x2` for `x^2, y^2, x^4, x^2 y^2, y^4`.
    pub bend_x: [f64; 5],
    /// `b1 .. b5` (already times `q - 1`): the terms of `y2` for `x^2, y^2, x^4, x^2 y^2, y^4`.
    pub bend_y: [f64; 5],
    /// `e k1`: the distortion's `r^2` coefficient, already scaled by Edges.
    pub d1: f64,
    /// `e k2`: the distortion's `r^4` coefficient, already scaled by Edges.
    pub d2: f64,
    /// The zoom `z`, 1 or more.
    pub zoom: f64,
    /// Newton steps of the inverse (`ym_lens::consts::NEWTON_STEPS`).
    pub steps: u8,
    /// The anamorphic axis is the canvas's `y` axis, not its `x`: the geometry of a DNG, whose canvas
    /// is the sensor buffer, has it when the picture's horizontal is the buffer's vertical.
    pub axis_y: bool,
}

/// `G` of one axis for the ten constants' half: `c[0] x^2 + c[1] y^2 + c[2] x^4 + c[3] x^2 y^2 + c[4] y^4`,
/// from `xx = x^2` and `yy = y^2` (the same expression as `ym_lens`'s, operation by operation).
fn bend_poly(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    xx * (c[0] + c[2] * xx + c[3] * yy) + yy * (c[1] + c[4] * yy)
}

/// `(dG/dx) / x`: `2 (c0 + 2 c2 x^2 + c3 y^2)`.
fn bend_poly_dx_over_x(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    2.0 * (c[0] + 2.0 * c[2] * xx + c[3] * yy)
}

/// `(dG/dy) / y`: `2 (c1 + c3 x^2 + 2 c4 y^2)`.
fn bend_poly_dy_over_y(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    2.0 * (c[1] + c[3] * xx + 2.0 * c[4] * yy)
}

impl SamplerWarp {
    /// A warp along the canvas's horizontal.
    #[must_use]
    pub fn new(bend_x: [f64; 5], bend_y: [f64; 5], d1: f64, d2: f64, zoom: f64, steps: u8) -> Self {
        Self {
            bend_x,
            bend_y,
            d1,
            d2,
            zoom,
            steps,
            axis_y: false,
        }
    }

    /// The same map for a canvas turned a quarter turn or transposed: the roles of `x` and `y`
    /// swap. (The map is odd in `x` and in `y`, so mirrors and half turns need no change.)
    #[must_use]
    pub fn transposed(mut self) -> Self {
        self.axis_y = !self.axis_y;
        self
    }

    /// True when the map is the identity: the ten bend constants and the distortion are all 0.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.bend_x.iter().all(|&c| c == 0.0)
            && self.bend_y.iter().all(|&c| c == 0.0)
            && self.d1 == 0.0
            && self.d2 == 0.0
    }

    /// The half-diagonal of a canvas: the unit of the map.
    #[must_use]
    pub fn half_diagonal(canvas: (f64, f64)) -> f64 {
        0.5 * (canvas.0 * canvas.0 + canvas.1 * canvas.1).sqrt()
    }

    /// `F(x, y)` in the axis frame (`x` along the axis).
    fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let g = 1.0 + self.d1 * r2 + self.d2 * r2 * r2;
        let (x1, y1) = (x * g, y * g);
        let (xx, yy) = (x1 * x1, y1 * y1);
        let gx = bend_poly(&self.bend_x, xx, yy);
        let gy = bend_poly(&self.bend_y, xx, yy);
        (x1 * (1.0 + gx), y1 * (1.0 + gy))
    }

    /// `F` and its Jacobian `[[dX/dx, dX/dy], [dY/dx, dY/dy]]`.
    fn forward_jacobian(&self, x: f64, y: f64) -> ((f64, f64), [[f64; 2]; 2]) {
        let r2 = x * x + y * y;
        let g = 1.0 + self.d1 * r2 + self.d2 * r2 * r2;
        // dg/d(r^2).
        let gp = self.d1 + 2.0 * self.d2 * r2;
        let (x1, y1) = (x * g, y * g);
        // Jacobian of (x1, y1) with respect to (x, y).
        let a = g + 2.0 * x * x * gp;
        let b = 2.0 * x * y * gp;
        let d = g + 2.0 * y * y * gp;
        // Second stage: X = x1 (1 + Gx), Y = y1 (1 + Gy), at (x1, y1).
        let (xx, yy) = (x1 * x1, y1 * y1);
        let gx = bend_poly(&self.bend_x, xx, yy);
        let gy = bend_poly(&self.bend_y, xx, yy);
        let fx = x1 * (1.0 + gx);
        let fy = y1 * (1.0 + gy);
        let j00 = 1.0 + gx + xx * bend_poly_dx_over_x(&self.bend_x, xx, yy);
        let j01 = x1 * y1 * bend_poly_dy_over_y(&self.bend_x, xx, yy);
        let j10 = y1 * x1 * bend_poly_dx_over_x(&self.bend_y, xx, yy);
        let j11 = 1.0 + gy + yy * bend_poly_dy_over_y(&self.bend_y, xx, yy);
        let j = [
            [j00 * a + j01 * b, j00 * b + j01 * d],
            [j10 * a + j11 * b, j10 * b + j11 * d],
        ];
        ((fx, fy), j)
    }

    /// `F^-1(tx, ty)` by Newton steps from `(tx, ty)`.
    fn inverse(&self, tx: f64, ty: f64) -> (f64, f64) {
        let (mut x, mut y) = (tx, ty);
        for _ in 0..self.steps {
            let ((fx, fy), j) = self.forward_jacobian(x, y);
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let (rx, ry) = (fx - tx, fy - ty);
            x -= (j[1][1] * rx - j[0][1] * ry) / det;
            y -= (-j[1][0] * rx + j[0][0] * ry) / det;
        }
        (x, y)
    }

    /// The photo position a canvas position reads: `F^-1(c / z)`, in the pixels of a canvas of size
    /// `canvas` (continuous positions, pixel `i` covering `[i, i + 1)`). The photo has the canvas's
    /// size (the same frame, before straightening).
    #[must_use]
    pub fn canvas_to_photo(&self, canvas: (f64, f64), x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * canvas.0, 0.5 * canvas.1, Self::half_diagonal(canvas));
        let (nx, ny) = ((x - cx) / h / self.zoom, (y - cy) / h / self.zoom);
        let (px, py) = if self.axis_y {
            let (py, px) = self.inverse(ny, nx);
            (px, py)
        } else {
            self.inverse(nx, ny)
        };
        (cx + px * h, cy + py * h)
    }

    /// The canvas position of a photo position: the forward map `z F(p)`.
    #[must_use]
    pub fn photo_to_canvas(&self, canvas: (f64, f64), x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * canvas.0, 0.5 * canvas.1, Self::half_diagonal(canvas));
        let (nx, ny) = ((x - cx) / h, (y - cy) / h);
        let (fx, fy) = if self.axis_y {
            let (fy, fx) = self.forward(ny, nx);
            (fx, fy)
        } else {
            self.forward(nx, ny)
        };
        (cx + fx * self.zoom * h, cy + fy * self.zoom * h)
    }

    /// The largest local shrink of the map over a canvas: how many photo pixels one pixel of a
    /// step `v` on the canvas spans, at the worst point of the canvas, for each step in `steps`
    /// (canvas pixels, e.g. the columns of an output-to-canvas map), the largest of them.
    ///
    /// The local scale at the canvas position `c` is `|J^-1 v| / z`, `J` the Jacobian of `F` at the
    /// photo position `c` reads. The canvas is sampled on a lattice of `(SHRINK_LATTICE + 1)^2`
    /// points that holds its corners and edges, where the bend stretches most; the rule is a pure
    /// function of the warp, the canvas and the steps, so the CPU and the device (which ask the same
    /// grid) choose the same kernel. The identity has a shrink of `|v|`.
    #[must_use]
    pub fn max_shrink(&self, canvas: (f64, f64), steps: &[(f64, f64)]) -> f64 {
        if self.is_identity() {
            return steps.iter().map(|&(x, y)| x.hypot(y)).fold(0.0, f64::max);
        }
        let mut worst: f64 = 0.0;
        let (cx, cy, h) = (0.5 * canvas.0, 0.5 * canvas.1, Self::half_diagonal(canvas));
        for j in 0..=SHRINK_LATTICE {
            for i in 0..=SHRINK_LATTICE {
                let x = canvas.0 * i as f64 / SHRINK_LATTICE as f64;
                let y = canvas.1 * j as f64 / SHRINK_LATTICE as f64;
                let (nx, ny) = ((x - cx) / h / self.zoom, (y - cy) / h / self.zoom);
                // The axis frame: swap the canvas's axes in, work, swap out (lengths do not change).
                let (ax, ay) = if self.axis_y { (ny, nx) } else { (nx, ny) };
                let (px, py) = self.inverse(ax, ay);
                let (_, jac) = self.forward_jacobian(px, py);
                let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                if det == 0.0 || !det.is_finite() {
                    continue;
                }
                for &(vx, vy) in steps {
                    let (ux, uy) = if self.axis_y { (vy, vx) } else { (vx, vy) };
                    // J^-1 u = adj(J) u / det.
                    let sx = (jac[1][1] * ux - jac[0][1] * uy) / det;
                    let sy = (-jac[1][0] * ux + jac[0][0] * uy) / det;
                    worst = worst.max(sx.hypot(sy) / self.zoom);
                }
            }
        }
        worst
    }
}

/// Intervals of the lattice [`SamplerWarp::max_shrink`] looks at the canvas on, per axis.
const SHRINK_LATTICE: usize = 8;

/// Samples along each side of a region when [`SamplerGrid::source_bounds`] follows its edge through the
/// warp (`docs/21` section 4, L4).
const BOUNDS_EDGE_SAMPLES: usize = 64;

/// What the bounds of a region add to the caller's margin when a warp bends its edge: the curve between
/// two samples can leave their box by a fraction of a pixel (`docs/21` section 4, L4: "plus the
/// filter's support and 2 px").
const BOUNDS_WARP_EXTRA_PX: f64 = 2.0;

/// The dihedral part of the position chain in the form the device kernel adds a warp's displacement
/// with ([`SamplerGrid::warp_frame`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpFrame {
    /// The linear part of the map from canvas pixels to buffer pixels: a buffer displacement is
    /// `linear * canvas displacement`. For a canvas the size of the buffer (sides swapped by the turns) it is a
    /// signed permutation.
    pub linear: [[f64; 2]; 2],
    /// Its inverse.
    pub inverse: [[f64; 2]; 2],
    /// The buffer position of the canvas centre.
    pub centre: (f64, f64),
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
    /// The lens's bend between the canvas position and the buffer (L4); `None` for the identity.
    pub warp: Option<SamplerWarp>,
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
            warp: None,
        }
    }

    /// The same grid through the lens's bend. The identity warp is no warp (`None`), so a grid
    /// without a bend keeps its bytes and its kernels.
    #[must_use]
    pub fn with_warp(mut self, warp: Option<SamplerWarp>) -> Self {
        self.warp = warp.filter(|w| !w.is_identity());
        self
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

    /// The canvas position of an output position (crop and straighten applied), or `None` when it
    /// falls outside the canvas by more than a tolerance of 1e-6 px: the inside test. It is made
    /// before the warp, on the position the output has on the canvas, so the lens's bend never
    /// changes which pixels are inside.
    #[inline]
    #[must_use]
    pub fn canvas_position(&self, ox: f64, oy: f64) -> Option<(f64, f64)> {
        let (px, py) = apply(&self.to_canvas, ox, oy);
        if px < -INSIDE_TOL
            || py < -INSIDE_TOL
            || px > self.canvas.0 + INSIDE_TOL
            || py > self.canvas.1 + INSIDE_TOL
        {
            return None;
        }
        Some((px, py))
    }

    /// As [`SamplerGrid::source`] for a continuous output position.
    #[inline]
    #[must_use]
    pub fn source_at(&self, ox: f64, oy: f64) -> Option<(f64, f64)> {
        let (px, py) = self.canvas_position(ox, oy)?;
        // The lens's bend: the photo position this canvas position reads.
        let (px, py) = match &self.warp {
            Some(w) => w.canvas_to_photo(self.canvas, px, py),
            None => (px, py),
        };
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
        if !self.unrotated || self.warp.is_some() {
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
    ///
    /// A grid with a warp is never a copy. Its shrink is the largest over the canvas of the local
    /// scale of the output's two steps through the warp ([`SamplerWarp::max_shrink`]): one number for
    /// the grid, the same for a tile and the whole frame, for the CPU and the device. At full
    /// resolution the attachment's bend is far below 1.25, so exports and 100 % views stay on
    /// Catmull-Rom.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        if self.is_pixel_copy() {
            return Kernel::Copy;
        }
        let ratio = match &self.warp {
            Some(w) => {
                let m = &self.to_canvas;
                w.max_shrink(self.canvas, &[(m[0][0], m[1][0]), (m[0][1], m[1][1])])
            }
            None => {
                let (sx, sy) = self.step_lengths();
                sx.max(sy)
            }
        };
        if ratio <= 1.25 {
            Kernel::Bicubic
        } else {
            Kernel::Box((ratio - 0.25).ceil().clamp(2.0, 4.0) as usize)
        }
    }

    /// The position chain without the inside test, the warp, the clamp or the snap: output position
    /// to buffer position, exactly the arithmetic of [`SamplerGrid::source_at`] otherwise.
    fn raw_at(&self, ox: f64, oy: f64) -> (f64, f64) {
        let (px, py) = apply(&self.to_canvas, ox, oy);
        self.canvas_to_buffer(px, py)
    }

    /// The dihedral part of the chain: a canvas position (pixels) to the buffer position, with no
    /// warp, clamp or snap.
    fn canvas_to_buffer(&self, px: f64, py: f64) -> (f64, f64) {
        let (u, v) = (px / self.canvas.0, py / self.canvas.1);
        let o = canvas_to_oriented(self.quarter_turns, [u, v]);
        let s = oriented_to_sensor(self.exif_orientation, o);
        (s[0] * self.buffer.0, s[1] * self.buffer.1)
    }

    /// The dihedral part of the chain as the device kernel adds a warp's displacement with: a
    /// displacement on the canvas, in canvas pixels, moves the buffer position by `linear` times it.
    /// The entries come from evaluations in `f64` and are snapped to integers within 1e-9 (a canvas
    /// the buffer's size is a signed permutation, so they are 0 and +-1 exactly).
    #[must_use]
    pub fn warp_frame(&self) -> WarpFrame {
        let (cx, cy) = (0.5 * self.canvas.0, 0.5 * self.canvas.1);
        let c = self.canvas_to_buffer(cx, cy);
        let ex = self.canvas_to_buffer(cx + 1.0, cy);
        let ey = self.canvas_to_buffer(cx, cy + 1.0);
        let near_int = |v: f64| {
            let r = v.round();
            if (v - r).abs() < 1e-9 { r } else { v }
        };
        let linear = [
            [near_int(ex.0 - c.0), near_int(ey.0 - c.0)],
            [near_int(ex.1 - c.1), near_int(ey.1 - c.1)],
        ];
        let det = linear[0][0] * linear[1][1] - linear[0][1] * linear[1][0];
        let inv = |v: f64| near_int(v / det);
        let inverse = [
            [inv(linear[1][1]), inv(-linear[0][1])],
            [inv(-linear[1][0]), inv(linear[0][0])],
        ];
        WarpFrame {
            linear,
            inverse,
            centre: (snap(c.0), snap(c.1)),
        }
    }

    /// The smallest rectangle of the buffer (integer pixels, `x0 <= x < x1`) that holds every
    /// position the output rectangle `[ox, ox + ow) x [oy, oy + oh)` samples, widened by `margin`
    /// pixels and clipped to the buffer. `None` when the whole rectangle lies outside the photo.
    ///
    /// Without a warp the map is affine and the four corners say it all. With one the rectangle's
    /// image is bent, so the part of its edge that lies on the canvas and the part of the canvas's
    /// edge that lies in the rectangle (the boundary of what the rectangle samples, each clipped
    /// exactly) are followed through the warp at 65 points a side, the result clamped to the buffer
    /// as the sampler does, and the box is widened by 2 px more (the curve between two samples).
    /// The map is a diffeomorphism of the canvas, so the box of the boundary is the box of the
    /// region.
    #[must_use]
    pub fn source_bounds(
        &self,
        ox: usize,
        oy: usize,
        ow: usize,
        oh: usize,
        margin: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let (rx0, ry0) = (ox as f64, oy as f64);
        let (rx1, ry1) = ((ox + ow) as f64, (oy + oh) as f64);
        let mut lo = (f64::INFINITY, f64::INFINITY);
        let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        let mut extra = 0.0;
        match &self.warp {
            None => {
                for (cx, cy) in [(rx0, ry0), (rx1, ry0), (rx0, ry1), (rx1, ry1)] {
                    // The unclamped canvas position: the corners of a rotated rectangle may lie
                    // outside the canvas while the rectangle still overlaps it.
                    let (bx, by) = self.raw_at(cx, cy);
                    lo = (lo.0.min(bx), lo.1.min(by));
                    hi = (hi.0.max(bx), hi.1.max(by));
                    any = true;
                }
            }
            Some(w) => {
                extra = BOUNDS_WARP_EXTRA_PX;
                let (cw, ch) = self.canvas;
                let mut take = |px: f64, py: f64| {
                    let (px, py) = (px.clamp(0.0, cw), py.clamp(0.0, ch));
                    let (qx, qy) = w.canvas_to_photo(self.canvas, px, py);
                    let (bx, by) = self.canvas_to_buffer(qx, qy);
                    let (bx, by) = (bx.clamp(0.0, self.buffer.0), by.clamp(0.0, self.buffer.1));
                    lo = (lo.0.min(bx), lo.1.min(by));
                    hi = (hi.0.max(bx), hi.1.max(by));
                    any = true;
                };
                let n = BOUNDS_EDGE_SAMPLES;
                // (a) The rectangle's edges, where they are on the canvas.
                let corners = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)];
                for k in 0..4 {
                    let (a, b) = (
                        apply(&self.to_canvas, corners[k].0, corners[k].1),
                        apply(
                            &self.to_canvas,
                            corners[(k + 1) % 4].0,
                            corners[(k + 1) % 4].1,
                        ),
                    );
                    let tol = INSIDE_TOL;
                    if let Some((t0, t1)) = clip_segment(a, b, (-tol, -tol), (cw + tol, ch + tol)) {
                        for i in 0..=n {
                            let t = t0 + (t1 - t0) * i as f64 / n as f64;
                            take(a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1));
                        }
                    }
                }
                // (b) The canvas's edges, where they are in the rectangle.
                if let Some(inv) = invert(&self.to_canvas) {
                    let cc = [(0.0, 0.0), (cw, 0.0), (cw, ch), (0.0, ch)];
                    for k in 0..4 {
                        let (a, b) = (cc[k], cc[(k + 1) % 4]);
                        let (pa, pb) = (apply(&inv, a.0, a.1), apply(&inv, b.0, b.1));
                        if let Some((t0, t1)) = clip_segment(pa, pb, (rx0, ry0), (rx1, ry1)) {
                            for i in 0..=n {
                                let t = t0 + (t1 - t0) * i as f64 / n as f64;
                                take(a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1));
                            }
                        }
                    }
                }
            }
        }
        if !any {
            return None;
        }
        let m = margin as f64 + extra;
        let x0 = (lo.0 - m).floor().max(0.0);
        let y0 = (lo.1 - m).floor().max(0.0);
        let x1 = (hi.0 + m).ceil().min(self.buffer.0);
        let y1 = (hi.1 + m).ceil().min(self.buffer.1);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
    }

    /// The whole chain, without the warp, as one affine map from the output position to the buffer
    /// position (before the clamp to the buffer): `buffer = a x + b y + c, d x + e y + f`. A warp
    /// adds a displacement to this position ([`SamplerGrid::warp_frame`]). The dihedral part is a signed
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
        let inside = |x: i64| self.canvas_position(ox(x), oy).is_some();
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

    // -----------------------------------------------------------------------------------------
    // The lens's bend (L4)
    // -----------------------------------------------------------------------------------------

    /// The attachment's shipped constants (data version 5, per unit of `q - 1`).
    const SHIPPED_X: [f64; 5] = [-0.0402, -0.0494, 0.0069, 0.0128, 0.0329];
    const SHIPPED_Y: [f64; 5] = [-0.1157, -0.0196, 0.0294, 0.0297, -0.0529];

    /// The attachment's shipped constants at `q - 1 = extra`, with the zoom fitted on `canvas` as
    /// `ym_lens` does it (200 samples a side of the edge, 1 px margin).
    fn fitted(extra: f64, d1: f64, d2: f64, canvas: (f64, f64)) -> SamplerWarp {
        let mut w = SamplerWarp::new(
            SHIPPED_X.map(|c| c * extra),
            SHIPPED_Y.map(|c| c * extra),
            d1,
            d2,
            1.0,
            3,
        );
        let inside = |w: &SamplerWarp| {
            (0..=200).all(|i| {
                let t = f64::from(i) / 200.0;
                [
                    (t * canvas.0, 0.0),
                    (t * canvas.0, canvas.1),
                    (0.0, t * canvas.1),
                    (canvas.0, t * canvas.1),
                ]
                .into_iter()
                .all(|(x, y)| {
                    let (px, py) = w.canvas_to_photo(canvas, x, y);
                    px >= 1.0 && px <= canvas.0 - 1.0 && py >= 1.0 && py <= canvas.1 - 1.0
                })
            })
        };
        if inside(&w) {
            return w;
        }
        let (mut lo, mut hi) = (1.0f64, 1.0f64);
        while hi < 4.0 {
            hi += 0.0625;
            w.zoom = hi;
            if inside(&w) {
                break;
            }
            lo = hi;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            w.zoom = mid;
            if inside(&w) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        w.zoom = hi;
        w
    }

    #[test]
    fn the_warp_inverts_its_forward_map_on_both_axes() {
        for canvas in [(6000.0, 4000.0), (4000.0, 6000.0), (211.0, 157.0)] {
            for warp in [
                fitted(1.0, 0.0, 0.0, canvas),
                fitted(1.0, -0.04, 0.01, canvas),
                fitted(0.0, 0.05, -0.02, canvas).transposed(),
                fitted(0.5, 0.0, 0.0, canvas).transposed(),
            ] {
                for fx in [0.0, 0.07, 0.5, 0.93, 1.0] {
                    for fy in [0.0, 0.31, 0.5, 1.0] {
                        let (x, y) = (fx * canvas.0, fy * canvas.1);
                        // A photo position that maps inside the canvas: the forward map and back.
                        let (cx, cy) = warp.photo_to_canvas(canvas, x, y);
                        let (px, py) = warp.canvas_to_photo(canvas, cx, cy);
                        assert!(
                            (px - x).abs() < 1e-9 * canvas.0 && (py - y).abs() < 1e-9 * canvas.0,
                            "{canvas:?} ({x}, {y}) -> ({px}, {py})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_map_is_odd_in_each_axis_bit_for_bit() {
        // No decentre, shear or twist term: a mirrored position reads the mirrored photo position, forward
        // and inverse, in the normalised frame (the sampler needs nothing for the mirrors and the half turns
        // of the orientations).
        let canvas = (600.0, 400.0);
        for warp in [
            fitted(1.0, 0.0, 0.0, canvas),
            fitted(0.4, -0.04, 0.01, canvas),
        ] {
            for (tx, ty) in [(0.31, 0.12), (0.8, 0.55), (0.02, 0.4), (0.7, 0.01)] {
                let (x, y) = warp.inverse(tx, ty);
                let (fx, fy) = warp.forward(tx, ty);
                for (sx, sy) in [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    let (mx, my) = warp.inverse(sx * tx, sy * ty);
                    assert_eq!(
                        (mx.to_bits(), my.to_bits()),
                        ((sx * x).to_bits(), (sy * y).to_bits())
                    );
                    let (mx, my) = warp.forward(sx * tx, sy * ty);
                    assert_eq!(
                        (mx.to_bits(), my.to_bits()),
                        ((sx * fx).to_bits(), (sy * fy).to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn a_transposed_warp_is_the_warp_of_the_transposed_canvas() {
        // The DNG's geometry: the canvas is the buffer, the picture's horizontal is the buffer's
        // vertical. Swapping x and y of the canvas and of the point swaps the answer.
        let (w, h) = (300.0, 200.0);
        let warp = fitted(1.0, -0.03, 0.01, (w, h));
        let turned = warp.transposed();
        for (x, y) in [(10.0, 20.0), (150.0, 100.0), (290.0, 199.0), (1.0, 150.0)] {
            let (px, py) = warp.canvas_to_photo((w, h), x, y);
            let (qy, qx) = turned.canvas_to_photo((h, w), y, x);
            assert_eq!((px, py), (qx, qy));
        }
    }

    #[test]
    fn the_identity_warp_is_no_warp_and_changes_no_bits() {
        let (bw, bh) = (211.0, 157.0);
        let (s, c) = 0.2f64.sin_cos();
        let plain = SamplerGrid::new(
            [[0.9 * c, -0.9 * s, 3.5], [0.9 * s, 0.9 * c, 4.25]],
            (bw, bh),
            (bw, bh),
            6,
            1,
            false,
        );
        let same = plain.with_warp(Some(SamplerWarp::new([0.0; 5], [0.0; 5], 0.0, 0.0, 1.0, 3)));
        assert_eq!(same, plain);
        assert_eq!(same.warp, None);
    }

    #[test]
    fn a_warp_leaves_the_inside_test_alone() {
        // The inside test is on the canvas position, before the warp: the intervals and the bounds of
        // the mask are those of the grid without it.
        let (bw, bh) = (200.0, 150.0);
        for (angle, scale, n) in [(0.0, 1.0, 1), (0.3, 0.8, 1), (-0.7, 1.9, 2)] {
            let (s, c) = f64::sin_cos(angle);
            let plain = SamplerGrid::new(
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
            let bent = plain.with_warp(Some(fitted(1.0, -0.03, 0.0, (bw, bh))));
            assert!(bent.warp.is_some());
            assert_eq!(
                bent.inside_rows((3, 5), 160, 120, n),
                plain.inside_rows((3, 5), 160, 120, n)
            );
            assert_eq!(
                bent.inside_bounds((0, 0), 170, 130),
                plain.inside_bounds((0, 0), 170, 130)
            );
            // And the predicate itself.
            for (ox, oy) in [(0.5, 0.5), (60.5, 40.5), (150.5, 120.5), (199.5, 10.5)] {
                assert_eq!(
                    bent.source_at(ox, oy).is_some(),
                    plain.source_at(ox, oy).is_some()
                );
            }
        }
    }

    #[test]
    fn a_warped_position_is_the_inverse_map_of_the_canvas_position_clamped_to_the_buffer() {
        let (bw, bh) = (300.0, 200.0);
        let warp = fitted(1.0, -0.03, 0.0, (bw, bh));
        let grid = SamplerGrid::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            (bw, bh),
            (bw, bh),
            1,
            0,
            true,
        )
        .with_warp(Some(warp));
        for (ox, oy) in [(0.5, 0.5), (100.5, 50.5), (299.5, 199.5), (150.0, 100.0)] {
            let (px, py) = warp.canvas_to_photo((bw, bh), ox, oy);
            let (sx, sy) = grid.source_at(ox, oy).expect("inside");
            assert!((sx - px.clamp(0.0, bw)).abs() < 1e-9 && (sy - py.clamp(0.0, bh)).abs() < 1e-9);
        }
        // The canvas centre is a fixed point; the corners read inside the photo.
        let (cx, cy) = grid.source_at(150.0, 100.0).expect("inside");
        assert!((cx - 150.0).abs() < 1e-9 && (cy - 100.0).abs() < 1e-9);
        for (ox, oy) in [(0.0, 0.0), (bw, 0.0), (0.0, bh), (bw, bh)] {
            let (sx, sy) = grid.source_at(ox, oy).expect("inside");
            assert!(
                sx > 0.5 && sx < bw - 0.5 && sy > 0.5 && sy < bh - 0.5,
                "{sx}, {sy}"
            );
        }
        // A zoom that is too small reads beyond the photo, and the position is clamped to it.
        let loose = SamplerWarp { zoom: 1.0, ..warp };
        let grid = grid.with_warp(Some(loose));
        let (sx, sy) = grid.source_at(0.0, 0.0).expect("inside");
        assert!((0.0..=bw).contains(&sx) && (0.0..=bh).contains(&sy));
        let (px, _) = loose.canvas_to_photo((bw, bh), 0.0, 0.0);
        assert!(
            px < 0.0 && sx == 0.0,
            "the corner reads beyond the photo: {px}"
        );
    }

    #[test]
    fn a_warped_grid_is_never_a_copy_and_its_kernel_follows_the_worst_shrink() {
        let (bw, bh) = (600.0, 400.0);
        let canvas = (bw, bh);
        let plain = SamplerGrid::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            canvas,
            canvas,
            1,
            0,
            true,
        );
        assert_eq!(plain.kernel(), Kernel::Copy);
        // The shipped attachment at full strength: the worst shrink is the sides' and corners' (about
        // 1.02: the sides are drawn in by 2.5 % and the zoom takes most of it back), so a 1:1 grid is
        // Catmull-Rom, not a copy and not a box.
        let bent = plain.with_warp(Some(fitted(1.0, 0.0, 0.0, canvas)));
        assert!(!bent.is_pixel_copy());
        assert_eq!(bent.kernel(), Kernel::Bicubic);
        let worst = bent
            .warp
            .expect("a warp")
            .max_shrink(canvas, &[(1.0, 0.0), (0.0, 1.0)]);
        assert!(worst > 1.01 && worst < 1.05, "worst shrink {worst}");
        // The same grid shrunk by 1.24: alone that is Catmull-Rom (1.24 <= 1.25), but the worst places
        // shrink 1.24 x 1.018 through the bend, so the whole grid is a box.
        let shrunk = SamplerGrid::new(
            [[1.24, 0.0, 0.0], [0.0, 1.24, 0.0]],
            canvas,
            canvas,
            1,
            0,
            true,
        );
        assert_eq!(
            shrunk.kernel(),
            Kernel::Bicubic,
            "1.24 alone is Catmull-Rom"
        );
        let both = shrunk.with_warp(Some(fitted(1.0, 0.0, 0.0, canvas)));
        assert!(
            matches!(both.kernel(), Kernel::Box(2)),
            "{:?}",
            both.kernel()
        );
        // Distortion alone: a pincushion's edge shrinks the other way round, and the worst shrink is
        // still the largest local scale.
        let pin = plain.with_warp(Some(fitted(0.0, 0.05, -0.02, canvas)));
        assert!(matches!(pin.kernel(), Kernel::Bicubic | Kernel::Box(_)));
    }

    #[test]
    fn the_lattice_finds_the_worst_shrink_of_a_fine_scan() {
        // The rule takes the maximum over a 9 x 9 lattice that holds the corners and edges. A scan of
        // 61 x 41 points of the canvas must not find more than 1e-3 above it.
        let canvas = (600.0, 400.0);
        for warp in [
            fitted(1.0, 0.0, 0.0, canvas),
            fitted(1.0, -0.05, 0.02, canvas),
            fitted(0.0, 0.05, -0.02, canvas),
            fitted(1.0, 0.04, 0.0, canvas).transposed(),
        ] {
            let steps = [(1.0, 0.0), (0.0, 1.0)];
            let lattice = warp.max_shrink(canvas, &steps);
            let (cx, cy, h) = (300.0, 200.0, SamplerWarp::half_diagonal(canvas));
            let mut fine = 0.0f64;
            for j in 0..=40 {
                for i in 0..=60 {
                    let (x, y) = (f64::from(i) * 10.0, f64::from(j) * 10.0);
                    let (nx, ny) = ((x - cx) / h / warp.zoom, (y - cy) / h / warp.zoom);
                    let (ax, ay) = if warp.axis_y { (ny, nx) } else { (nx, ny) };
                    let (px, py) = warp.inverse(ax, ay);
                    let (_, jac) = warp.forward_jacobian(px, py);
                    let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                    for (vx, vy) in steps {
                        let (ux, uy) = if warp.axis_y { (vy, vx) } else { (vx, vy) };
                        let sx = (jac[1][1] * ux - jac[0][1] * uy) / det;
                        let sy = (-jac[1][0] * ux + jac[0][0] * uy) / det;
                        fine = fine.max(sx.hypot(sy) / warp.zoom);
                    }
                }
            }
            assert!(
                fine <= lattice + 1e-3,
                "lattice {lattice} below a fine scan {fine}"
            );
        }
    }

    #[test]
    fn the_dihedral_part_is_a_signed_permutation_for_every_orientation_and_turn() {
        let (bw, bh) = (640.0, 480.0);
        for exif in 1..=8u16 {
            for turns in 0..4u8 {
                let swaps = ((5..=8).contains(&exif)) != (turns % 2 == 1);
                let canvas = if swaps { (bh, bw) } else { (bw, bh) };
                let grid = SamplerGrid::new(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    canvas,
                    (bw, bh),
                    exif,
                    turns,
                    true,
                );
                let f = grid.warp_frame();
                for row in f.linear.iter().chain(&f.inverse) {
                    for &v in row {
                        assert!(
                            v == 0.0 || v == 1.0 || v == -1.0,
                            "exif {exif} turns {turns}: {f:?}"
                        );
                    }
                }
                assert_eq!(f.centre, (bw / 2.0, bh / 2.0));
                // linear * inverse = identity.
                let l = f.linear;
                let i = f.inverse;
                assert_eq!(l[0][0] * i[0][0] + l[0][1] * i[1][0], 1.0);
                assert_eq!(l[1][0] * i[0][1] + l[1][1] * i[1][1], 1.0);
                assert_eq!(l[0][0] * i[0][1] + l[0][1] * i[1][1], 0.0);
            }
        }
    }

    #[test]
    fn the_source_bounds_without_a_warp_are_the_four_corners() {
        // The old code of `ym-core`'s `Grid::source_bounds`, as the reference.
        let (bw, bh) = (400.0, 300.0);
        let (s, c) = 0.3f64.sin_cos();
        for (exif, turns) in [(1u16, 0u8), (6, 1), (3, 2), (8, 3)] {
            let swaps = ((5..=8).contains(&exif)) != (turns % 2 == 1);
            let canvas = if swaps { (bh, bw) } else { (bw, bh) };
            let grid = SamplerGrid::new(
                [[1.3 * c, -1.3 * s, 20.0], [1.3 * s, 1.3 * c, -15.0]],
                canvas,
                (bw, bh),
                exif,
                turns,
                false,
            );
            let (ox, oy, ow, oh, margin) = (10usize, 20usize, 90usize, 70usize, 3usize);
            let corners = [
                (ox as f64, oy as f64),
                ((ox + ow) as f64, oy as f64),
                (ox as f64, (oy + oh) as f64),
                ((ox + ow) as f64, (oy + oh) as f64),
            ];
            let mut lo = (f64::INFINITY, f64::INFINITY);
            let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (cx, cy) in corners {
                let (px, py) = apply(&grid.to_canvas, cx, cy);
                let (u, v) = (px / canvas.0, py / canvas.1);
                let o = canvas_to_oriented(turns, [u, v]);
                let sn = oriented_to_sensor(exif, o);
                let (bx, by) = (sn[0] * bw, sn[1] * bh);
                lo = (lo.0.min(bx), lo.1.min(by));
                hi = (hi.0.max(bx), hi.1.max(by));
            }
            let m = margin as f64;
            let want = (
                (lo.0 - m).floor().max(0.0) as usize,
                (lo.1 - m).floor().max(0.0) as usize,
                (hi.0 + m).ceil().min(bw) as usize,
                (hi.1 + m).ceil().min(bh) as usize,
            );
            assert_eq!(grid.source_bounds(ox, oy, ow, oh, margin), Some(want));
        }
    }

    #[test]
    fn the_source_bounds_through_a_warp_hold_every_position_the_rectangle_samples() {
        // Brute force: every output pixel of the rectangle that is inside the photo reads a position;
        // with Catmull-Rom's 2 px (a margin of 3 includes them) the bounds hold all of them, for
        // rotated, cropped and magnified grids and a rectangle that pokes out of the canvas.
        let (bw, bh) = (360.0, 240.0);
        let mut worst_slack = 0.0f64;
        for (exif, turns) in [(1u16, 0u8), (6, 1), (3, 2), (5, 3)] {
            let swaps = ((5..=8).contains(&exif)) != (turns % 2 == 1);
            let canvas = if swaps { (bh, bw) } else { (bw, bh) };
            for (angle, scale, t) in [
                (0.0, 1.0, (0.0, 0.0)),
                (4.0, 1.0, (10.0, -6.0)),
                (-17.0, 1.4, (-30.0, 20.0)),
                (0.0, 0.5, (40.0, 30.0)),
            ] {
                let (s, c) = (-f64::to_radians(angle)).sin_cos();
                let (cx, cy) = (canvas.0 * 0.5, canvas.1 * 0.5);
                let to_canvas = [
                    [c * scale, -s * scale, c * (t.0 - cx) - s * (t.1 - cy) + cx],
                    [s * scale, c * scale, s * (t.0 - cx) + c * (t.1 - cy) + cy],
                ];
                for extra in [1.0, 0.4] {
                    let grid = SamplerGrid::new(to_canvas, canvas, (bw, bh), exif, turns, false)
                        .with_warp(Some(fitted(extra, -0.03, 0.01, canvas)));
                    for (ox, oy, ow, oh) in [
                        (0usize, 0usize, 400usize, 300usize),
                        (30, 20, 100, 80),
                        (250, 150, 120, 90),
                        (0, 100, 64, 64),
                    ] {
                        let mut lo = (f64::INFINITY, f64::INFINITY);
                        let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                        let mut any = false;
                        for y in oy..oy + oh {
                            for x in ox..ox + ow {
                                if let Some((sx, sy)) = grid.source(x, y) {
                                    lo = (lo.0.min(sx), lo.1.min(sy));
                                    hi = (hi.0.max(sx), hi.1.max(sy));
                                    any = true;
                                }
                            }
                        }
                        let got = grid.source_bounds(ox, oy, ow, oh, 3);
                        if !any {
                            continue;
                        }
                        let (x0, y0, x1, y1) = got.expect("the rectangle samples the photo");
                        // The positions of pixel centres plus 2 px of filter, inside the bounds.
                        assert!(
                            (x0 as f64) <= (lo.0 - 2.0).max(0.0)
                                && (y0 as f64) <= (lo.1 - 2.0).max(0.0)
                                && (x1 as f64) >= (hi.0 + 2.0).min(bw)
                                && (y1 as f64) >= (hi.1 + 2.0).min(bh),
                            "exif {exif} turns {turns} angle {angle} scale {scale} rect {:?}: bounds {:?} vs positions {lo:?} {hi:?}",
                            (ox, oy, ow, oh),
                            (x0, y0, x1, y1)
                        );
                        // And they are tight: within 6 px of the positions' box (3 margin + 2 extra
                        // + the pixel rounding).
                        let slack = [
                            lo.0 - x0 as f64,
                            lo.1 - y0 as f64,
                            x1 as f64 - hi.0,
                            y1 as f64 - hi.1,
                        ]
                        .into_iter()
                        .fold(0.0f64, f64::max);
                        worst_slack = worst_slack.max(slack);
                    }
                }
            }
        }
        assert!(
            worst_slack < 8.0,
            "the bounds are loose by {worst_slack} px"
        );
    }
}
