//! L2: field optics, the CPU reference (`docs/21` section 4, L2).
//!
//! # What a pixel does
//!
//! An output pixel at field position `(u, v)` (the pixel centre minus the frame centre, over the frame's
//! half-diagonal `H`) gathers its input with a kernel:
//!
//! * the **oval field radius** `rho' = sqrt((alpha x_a)^2 + y_a^2)` with `x_a`, `y_a` the field coordinates
//!   along and across the anamorphic axis and `alpha = 1 + kappa (q - 1)`; it selects the spread function
//!   tables (held at the last table beyond 1) and the relative illumination `T`;
//! * the **edge amount** `e' = e + (1 - e) kappa (q - 1)`: Edges, plus the attachment's own share, so that at
//!   Edges 0 the anamorphic terms act alone (`docs/21` section 4, L2);
//! * the **half-width** `omega = min(e' (1 + S) w(rho'), w_max) H` pixels, with the side softness
//!   `S = a_s (q - 1) x_a^2`;
//! * the **gain** `1 - e' (1 - T(rho'))`.
//!
//! The kernel is the projected channel table (`sum_j w_c(l_j) P_j`), blended linearly between the two
//! neighbouring radii, resampled area-weighted to `n x n` cells (`n = min(17, 2 ceil(omega) + 1)`), turned so
//! that table x points radially outwards and table y anticlockwise, and scaled to `+-omega`. Each cell is a
//! point mass at its centre, read from the input by bilinear interpolation. Because the channel tables have
//! their own centroids, the channels are shifted against each other: that is the lateral colour. With the
//! attachment the axis part of each channel's centroid offset is multiplied by `1 + h (q - 1)`.
//!
//! A kernel with `omega` below half a pixel is a copy: the pixel is the input pixel times the gain.
//!
//! # Wide kernels
//!
//! A kernel wider than the tap budget at one tap a pixel reads a 2x, 4x, ... box-reduced copy of the input (a
//! mip level whose blocks are aligned to absolute frame coordinates, so that any tiling reads the same texels)
//! until its cells are at most one texel apart. The input is replicated beyond the frame edge.
//!
//! # Determinism
//!
//! A pixel depends only on its absolute position, the plan and the input in its receptive field
//! ([`halo`]); sums run in a fixed order, and only `+ - * /`, `sqrt` and truncation are used.

use rayon::prelude::*;

use crate::consts::{COPY_BELOW_PX, MAX_TAPS_SIDE, PSF_SIZE, PSF_WAVELENGTHS, RADII, TAP_SPACING_MAX_PX, W_MAX};
use crate::data::LensData;
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::{Axis, LensPlan};
use crate::simd::F4;
use crate::spectral::CameraResponse;

/// Number of table sizes `n = 3, 5, ..., MAX_TAPS_SIDE`.
const SIZES: usize = (MAX_TAPS_SIDE - 3) / 2 + 1;

/// The most mip levels a kernel can ask for (a safety bound; real kernels need 0 to 4).
const MAX_LEVEL: u32 = 12;

/// Knobs of the CPU reference that tests turn; production code uses [`Options::default`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    /// Largest spacing of the taps in texels (default [`TAP_SPACING_MAX_PX`]).
    pub spacing_max: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            spacing_max: TAP_SPACING_MAX_PX,
        }
    }
}

/// The lens's spread functions projected on a camera's three channels, at every table size.
#[derive(Clone, Debug)]
pub(crate) struct FieldTables {
    radii: [f64; RADII],
    half_width: [f64; RADII],
    illumination: [f64; RADII],
    /// Largest `w(rho)`.
    pub(crate) half_width_max: f64,
    /// Centroid of each channel table in table units (x radial, y tangential), indexed `[radius][channel]`.
    centroid: [[[f64; 2]; 3]; RADII],
    /// Largest centroid distance from the table centre, in table units.
    pub(crate) centroid_max: f64,
    /// `tables[s]` holds, for `n = 3 + 2 s`, the `n x n` tables indexed `[radius][cell]`, each cell the weights
    /// of the three channels and a zero (so a cell is one 4-lane vector).
    tables: Vec<Vec<V4>>,
}

/// Four `f32` lanes: the three channels of a texel or of a cell's weights, and a padding lane.
type V4 = [f32; 4];

/// Centre of cell `k` of `n` cells over `[-1, 1]`.
fn cell_centre(k: usize, n: usize) -> f64 {
    -1.0 + (2 * k + 1) as f64 / n as f64
}

/// Area-weighted resampling of a 33 x 33 mass table to `n x n` cells over the same span. Mass is kept.
fn resample(src: &[f64], n: usize) -> Vec<f64> {
    let m = PSF_SIZE;
    if n == m {
        return src.to_vec();
    }
    // Fraction of source cell k that target cell a covers, with both cut on the common grid of 33 n steps.
    let frac = |a: usize, k: usize| -> f64 {
        let (a0, a1) = (a * m, (a + 1) * m);
        let (k0, k1) = (k * n, (k + 1) * n);
        let overlap = a1.min(k1).saturating_sub(a0.max(k0));
        overlap as f64 / n as f64
    };
    // Along x: tmp[j][a] = sum_k src[j][k] F(a, k).
    let mut tmp = vec![0.0; m * n];
    for j in 0..m {
        for a in 0..n {
            let mut s = 0.0;
            for k in 0..m {
                s += src[j * m + k] * frac(a, k);
            }
            tmp[j * n + a] = s;
        }
    }
    // Along y: out[b][a] = sum_j F(b, j) tmp[j][a].
    let mut out = vec![0.0; n * n];
    for b in 0..n {
        for a in 0..n {
            let mut s = 0.0;
            for j in 0..m {
                s += frac(b, j) * tmp[j * n + a];
            }
            out[b * n + a] = s;
        }
    }
    out
}

impl FieldTables {
    /// Projects a lens's tables on the camera's channels and resamples them to every size.
    pub(crate) fn new(lens: &LensData, camera: &CameraResponse) -> Self {
        let cells = PSF_SIZE * PSF_SIZE;
        let mut proj = vec![0.0f64; RADII * 3 * cells];
        let mut centroid = [[[0.0; 2]; 3]; RADII];
        for r in 0..RADII {
            for c in 0..3 {
                let dst = &mut proj[(r * 3 + c) * cells..][..cells];
                for j in 0..PSF_WAVELENGTHS {
                    let w = camera.weights()[c][j];
                    for (d, s) in dst.iter_mut().zip(lens.psf(r, j)) {
                        *d += w * f64::from(*s);
                    }
                }
                let sum: f64 = dst.iter().sum();
                for d in dst.iter_mut() {
                    *d /= sum;
                }
                let (mut cx, mut cy) = (0.0, 0.0);
                for (i, m) in dst.iter().enumerate() {
                    cx += m * cell_centre(i % PSF_SIZE, PSF_SIZE);
                    cy += m * cell_centre(i / PSF_SIZE, PSF_SIZE);
                }
                centroid[r][c] = [cx, cy];
            }
        }
        let mut tables = Vec::with_capacity(SIZES);
        for s in 0..SIZES {
            let n = 3 + 2 * s;
            let nn = n * n;
            let mut t = vec![[0.0f32; 4]; RADII * nn];
            for r in 0..RADII {
                for c in 0..3 {
                    let mut cell = resample(&proj[(r * 3 + c) * cells..][..cells], n);
                    // Each kernel sums to 1 (to the rounding of the sum).
                    let sum: f64 = cell.iter().sum();
                    for v in &mut cell {
                        *v /= sum;
                    }
                    for (q, v) in cell.iter().enumerate() {
                        t[r * nn + q][c] = *v as f32;
                    }
                }
            }
            tables.push(t);
        }
        let mut radii = [0.0; RADII];
        let mut half_width = [0.0; RADII];
        let mut illumination = [0.0; RADII];
        radii.copy_from_slice(&lens.field.radii);
        half_width.copy_from_slice(&lens.field.psf_half_width);
        illumination.copy_from_slice(&lens.field.relative_illumination);
        let centroid_max = centroid
            .iter()
            .flatten()
            .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
            .fold(0.0, f64::max);
        FieldTables {
            radii,
            half_width,
            illumination,
            half_width_max: half_width.iter().copied().fold(0.0, f64::max),
            centroid,
            centroid_max,
            tables,
        }
    }

    /// The segment `s` (0 to 7) and blend `t` (0 to 1) of field radius `rho` (held at 1 beyond it).
    fn segment(&self, rho: f64) -> (usize, f64) {
        let rho = rho.clamp(0.0, 1.0);
        let mut s = 0;
        while s + 2 < RADII && rho > self.radii[s + 1] {
            s += 1;
        }
        let (a, b) = (self.radii[s], self.radii[s + 1]);
        (s, ((rho - a) / (b - a)).clamp(0.0, 1.0))
    }

    /// The `n x n` table of `radius`: `n^2` cells of channel weights, row-major (x radial, y tangential).
    fn table(&self, n: usize, radius: usize) -> &[V4] {
        let len = n * n;
        &self.tables[(n - 3) / 2][radius * len..][..len]
    }
}

/// Everything the pixels of one call share.
#[derive(Clone, Copy)]
pub(crate) struct Geometry<'a> {
    tables: &'a FieldTables,
    /// Frame size at this scale.
    fw: f64,
    fh: f64,
    /// Half-diagonal in pixels.
    h: f64,
    axis: Axis,
    alpha: f64,
    side: f64,
    e: f64,
    fringe: f64,
    opts: Options,
    /// Most mip levels any pixel of the frame can ask for.
    lmax: u32,
    /// Largest context any pixel reads, in pixels.
    halo: usize,
}

/// One pixel's kernel parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelKernel {
    /// The gain `1 - e' (1 - T)`.
    pub gain: f64,
    /// The kernel's half-width in pixels (0 when the kernel is a copy).
    pub omega: f64,
    /// Lower table of the blend (the radius index).
    pub seg: usize,
    /// Weight of the upper table.
    pub t: f64,
    /// Radial unit vector in the frame (x right, y down): x component.
    pub cs: f64,
    /// Radial unit vector in the frame: y component.
    pub sn: f64,
}

/// Taps per side for a half-width of `omega` texels.
fn taps_for(omega: f64) -> usize {
    ((2.0 * omega.ceil() + 1.0) as usize).min(MAX_TAPS_SIDE)
}

/// The mip level that brings the cells of an `omega`-pixel kernel within `spacing_max` texels of each other.
fn level_for(omega: f64, spacing_max: f64) -> u32 {
    let mut level = 0;
    let mut w = omega;
    while level < MAX_LEVEL && 2.0 * w / taps_for(w) as f64 > spacing_max {
        w *= 0.5;
        level += 1;
    }
    level
}

/// `floor(x)` as an integer, without the library call of `f64::floor` on baseline x86-64.
#[inline(always)]
fn floor_i64(x: f64) -> i64 {
    let t = x as i64;
    t - i64::from((t as f64) > x)
}

impl<'a> Geometry<'a> {
    /// The geometry of `plan` for a frame of `fw x fh` pixels at the scale of the caller.
    ///
    /// `None` when the plan has no field optics (nothing to do and nothing to read).
    pub(crate) fn new(plan: &'a LensPlan, fw: usize, fh: usize, opts: Options) -> Option<Self> {
        let tables = plan.field_tables()?;
        let e = plan.field_amount();
        let (fwf, fhf) = (fw as f64, fh as f64);
        let h = 0.5 * (fwf * fwf + fhf * fhf).sqrt();
        let axis = plan.frame().axis;
        let (alpha, side, fringe) = (plan.alpha(), plan.side_softness(), plan.fringe_factor());
        // Upper bounds on the half-width, over the whole frame.
        let along = match axis {
            Axis::X => fwf,
            Axis::Y => fhf,
        };
        let xa_max = 0.5 * along / h;
        let omega_max = (e * (1.0 + side * xa_max * xa_max) * tables.half_width_max).min(W_MAX) * h;
        let mut lmax = 0;
        let mut o = COPY_BELOW_PX;
        while o < omega_max + 0.25 {
            lmax = lmax.max(level_for(o.min(omega_max), opts.spacing_max));
            o += 0.25;
        }
        let halo = if omega_max < COPY_BELOW_PX {
            0
        } else {
            // A tap lies within sqrt(2) omega of the pixel centre, plus the fringe shift, which is at most
            // (boost) times the channel's centroid offset; it reads texels covering P +- 1.5 texels at the
            // coarsest level.
            let reach =
                omega_max * (std::f64::consts::SQRT_2 + (fringe - 1.0).max(0.0) * tables.centroid_max);
            let margin = 1.5 * f64::from(1u32 << lmax);
            reach.ceil() as usize + margin.ceil() as usize + 3
        };
        Some(Geometry {
            tables,
            fw: fwf,
            fh: fhf,
            h,
            axis,
            alpha,
            side,
            e,
            fringe,
            opts,
            lmax,
            halo,
        })
    }

    /// Pixels of context an output rectangle needs around it.
    pub(crate) fn halo(&self) -> usize {
        self.halo
    }

    /// The kernel of the pixel at integer frame position `(i, j)`.
    pub(crate) fn pixel(&self, i: usize, j: usize) -> PixelKernel {
        let u = (i as f64 + 0.5 - 0.5 * self.fw) / self.h;
        let v = (j as f64 + 0.5 - 0.5 * self.fh) / self.h;
        let (xa, ya) = match self.axis {
            Axis::X => (u, v),
            Axis::Y => (v, u),
        };
        let ax = self.alpha * xa;
        let rho = (ax * ax + ya * ya).sqrt();
        let tb = self.tables;
        let (seg, t) = tb.segment(rho);
        let illum = (1.0 - t) * tb.illumination[seg] + t * tb.illumination[seg + 1];
        let gain = 1.0 - self.e * (1.0 - illum);
        let w = (1.0 - t) * tb.half_width[seg] + t * tb.half_width[seg + 1];
        let softness = 1.0 + self.side * xa * xa;
        let omega = (self.e * softness * w).min(W_MAX) * self.h;
        let rr = (u * u + v * v).sqrt();
        let (cs, sn) = if rr > 1e-12 { (u / rr, v / rr) } else { (1.0, 0.0) };
        PixelKernel {
            gain,
            omega: if omega < COPY_BELOW_PX { 0.0 } else { omega },
            seg,
            t,
            cs,
            sn,
        }
    }

    /// The blended centroid of channel `c` in table units.
    #[cfg(test)]
    fn centroid(&self, k: &PixelKernel, c: usize) -> [f64; 2] {
        let a = self.tables.centroid[k.seg][c];
        let b = self.tables.centroid[k.seg + 1][c];
        [(1.0 - k.t) * a[0] + k.t * b[0], (1.0 - k.t) * a[1] + k.t * b[1]]
    }

    /// The centroid offset of channel `c` in pixels (frame x, y) including the attachment's fringe boost.
    #[cfg(test)]
    pub(crate) fn centroid_px(&self, k: &PixelKernel, c: usize) -> [f64; 2] {
        let [cx, cy] = self.centroid(k, c);
        // Table x is radial (cs, sn), table y is tangential (sn, -cs).
        let dx = k.omega * (cx * k.cs + cy * k.sn);
        let dy = k.omega * (cx * k.sn - cy * k.cs);
        let s = self.fringe_shifts(k)[c];
        [dx + s[0], dy + s[1]]
    }

    /// The taps of channel `c` of a pixel, as `(dx, dy, weight)` with offsets from the pixel centre in
    /// pixels, for tests and reference checks. Positions and weights are those the gather uses at mip
    /// level 0 (`n` is the level-0 tap count).
    #[cfg(test)]
    pub(crate) fn taps(&self, k: &PixelKernel, c: usize) -> Vec<(f64, f64, f64)> {
        if k.omega == 0.0 {
            return vec![(0.0, 0.0, 1.0)];
        }
        let level = level_for(k.omega, self.opts.spacing_max).min(self.lmax);
        let n = taps_for(k.omega / f64::from(1u32 << level));
        let shift = self.fringe_shifts(k)[c];
        let (lo, hi) = (self.tables.table(n, k.seg), self.tables.table(n, k.seg + 1));
        let mut out = Vec::with_capacity(n * n);
        for b in 0..n {
            for a in 0..n {
                let (tx, ty) = (cell_centre(a, n), cell_centre(b, n));
                let ox = k.omega * (tx * k.cs + ty * k.sn);
                let oy = k.omega * (tx * k.sn - ty * k.cs);
                let q = (b * n + a) * 3 + c;
                out.push((ox + shift[0], oy + shift[1], (1.0 - k.t) * lo[q] + k.t * hi[q]));
            }
        }
        out
    }

    /// Per-channel extra shift in pixels (frame x, y) from the fringe boost: the boost times the axis part of
    /// the channel's centroid offset.
    fn fringe_shifts(&self, k: &PixelKernel) -> [[f64; 2]; 3] {
        let boost = self.fringe - 1.0;
        let mut s = [[0.0; 2]; 3];
        if boost == 0.0 || k.omega == 0.0 {
            return s;
        }
        for (c, sc) in s.iter_mut().enumerate() {
            let a = self.tables.centroid[k.seg][c];
            let b = self.tables.centroid[k.seg + 1][c];
            let cx = (1.0 - k.t) * a[0] + k.t * b[0];
            let cy = (1.0 - k.t) * a[1] + k.t * b[1];
            match self.axis {
                Axis::X => sc[0] = boost * k.omega * (cx * k.cs + cy * k.sn),
                Axis::Y => sc[1] = boost * k.omega * (cx * k.sn - cy * k.cs),
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------------------
// The input pyramid

/// One level of the input pyramid: texel `(bx, by)` is the mean of the input over the aligned block of
/// `2^L x 2^L` frame pixels starting at `(bx 2^L, by 2^L)`. A texel is its three channels and a zero.
struct Level {
    bx0: i64,
    by0: i64,
    w: usize,
    h: usize,
    data: Vec<V4>,
}

struct Pyramid {
    levels: Vec<Level>,
}

fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

impl Pyramid {
    /// Builds the levels `0..=lmax` over `need` (frame coordinates, may lie outside the frame), rounded
    /// outwards to blocks of `2^lmax`. Pixels outside the frame replicate the edge; pixels outside the input
    /// (which a covering input makes unread) take the nearest input pixel.
    fn build(input: &ImageRef, need: [i64; 4], lmax: u32) -> Pyramid {
        let b = 1i64 << lmax;
        let x0 = floor_div(need[0], b) * b;
        let y0 = floor_div(need[1], b) * b;
        let x1 = ceil_div(need[2], b) * b;
        let y1 = ceil_div(need[3], b) * b;
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);

        let (fw, fh) = (input.frame_width as i64, input.frame_height as i64);
        let (ix0, iy0) = (input.x0 as i64, input.y0 as i64);
        let (iw, ih) = (input.width, input.height);
        let planes = [input.plane(0), input.plane(1), input.plane(2)];
        let mut base = vec![[0.0f32; 4]; w * h];
        base.par_chunks_mut(w).enumerate().for_each(|(row, d)| {
            let y = (y0 + row as i64).clamp(0, fh - 1).clamp(iy0, iy0 + ih as i64 - 1);
            let at = (y - iy0) as usize * iw;
            for (col, px) in d.iter_mut().enumerate() {
                let x = (x0 + col as i64).clamp(0, fw - 1).clamp(ix0, ix0 + iw as i64 - 1);
                let i = at + (x - ix0) as usize;
                *px = [planes[0][i], planes[1][i], planes[2][i], 0.0];
            }
        });
        let mut levels = vec![Level {
            bx0: x0,
            by0: y0,
            w,
            h,
            data: base,
        }];
        for l in 1..=lmax {
            let prev = &levels[(l - 1) as usize];
            let (pw, w2, h2) = (prev.w, prev.w / 2, prev.h / 2);
            let mut data = vec![[0.0f32; 4]; w2 * h2];
            data.par_chunks_mut(w2).enumerate().for_each(|(row, d)| {
                let r0 = &prev.data[2 * row * pw..][..pw];
                let r1 = &prev.data[(2 * row + 1) * pw..][..pw];
                for (col, px) in d.iter_mut().enumerate() {
                    let (a, b) = (r0[2 * col], r0[2 * col + 1]);
                    let (c, dd) = (r1[2 * col], r1[2 * col + 1]);
                    for k in 0..3 {
                        px[k] = ((a[k] + b[k]) + (c[k] + dd[k])) * 0.25;
                    }
                }
            });
            let level = Level {
                bx0: prev.bx0 / 2,
                by0: prev.by0 / 2,
                w: w2,
                h: h2,
                data,
            };
            levels.push(level);
        }
        Pyramid { levels }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Sampling

/// Fractional bits of the fixed-point tap positions.
const FRAC: u32 = 20;
const FRAC_MASK: i64 = (1 << FRAC) - 1;
const FRAC_SCALE: f32 = 1.0 / (1u32 << FRAC) as f32;

/// A fixed-point value from a real one.
#[inline(always)]
fn fix(x: f64) -> i64 {
    floor_i64(x * f64::from(1u32 << FRAC) + 0.5)
}

#[inline(always)]
fn lerp(a: F4, b: F4, t: F4) -> F4 {
    a.add(t.mul(b.sub(a)))
}

/// Bilinear sample of `lv` at the texel coordinate `(ux, uy)`: fixed point, relative to the level's first
/// texel centre, so texel `(i, j)` is at `(i, j) << FRAC`.
///
/// With `CLAMP` the texel indices are held inside the level (for a pixel whose taps might leave it, which a
/// covering input rules out); without it they are used as they are.
#[inline(always)]
fn sample<const CLAMP: bool>(lv: &Level, ux: i64, uy: i64) -> F4 {
    let (mut ix, mut iy) = (ux >> FRAC, uy >> FRAC);
    let fx = F4::splat((ux & FRAC_MASK) as i32 as f32 * FRAC_SCALE);
    let fy = F4::splat((uy & FRAC_MASK) as i32 as f32 * FRAC_SCALE);
    if CLAMP {
        ix = ix.max(0).min(lv.w as i64 - 2);
        iy = iy.max(0).min(lv.h as i64 - 2);
    }
    let at = iy as usize * lv.w + ix as usize;
    // One slice for both rows: a single bounds check.
    let rows = &lv.data[at..at + lv.w + 2];
    let top = lerp(F4::load(&rows[0]), F4::load(&rows[1]), fx);
    let bot = lerp(F4::load(&rows[lv.w]), F4::load(&rows[lv.w + 1]), fx);
    lerp(top, bot, fy)
}

// ---------------------------------------------------------------------------------------------------------
// The pass

/// L2 on `out` (frame coordinates at the caller's scale): the planar output, `3 * out.w * out.h` samples.
///
/// `input` must cover `out` grown by [`Geometry::halo`] and clamped to the frame. With no field optics
/// (`plan.field_tables()` is `None`) the result is the input's `out` rectangle, bit for bit.
pub(crate) fn apply(
    plan: &LensPlan,
    input: &ImageRef,
    out: PixelRect,
    opts: Options,
) -> Result<Vec<f32>, LensError> {
    input.validate()?;
    let (fw, fh) = (input.frame_width, input.frame_height);
    if out.x1() > fw || out.y1() > fh {
        return Err(LensError::Frame(format!(
            "the output {out} does not fit the {fw}x{fh} frame"
        )));
    }
    let Some(geo) = Geometry::new(plan, fw, fh, opts) else {
        return copy_rect(input, out);
    };
    let need = out.grown(geo.halo, fw, fh);
    if !input.rect().contains(&need) {
        return Err(LensError::InputTooSmall {
            needed: need.to_string(),
            have: input.rect().to_string(),
        });
    }
    if out.w == 0 || out.h == 0 {
        return Ok(Vec::new());
    }
    // The pyramid covers the output grown by the halo, even beyond the frame (replicated).
    let halo = geo.halo as i64;
    let region = [
        out.x as i64 - halo,
        out.y as i64 - halo,
        out.x1() as i64 + halo,
        out.y1() as i64 + halo,
    ];
    let pyramid = Pyramid::build(input, region, geo.lmax);

    let plane = out.w * out.h;
    let mut result = vec![0.0f32; 3 * plane];
    let (r, rest) = result.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    r.par_chunks_mut(out.w)
        .zip(g.par_chunks_mut(out.w))
        .zip(b.par_chunks_mut(out.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = out.y + row;
            for col in 0..out.w {
                let px = pixel(&geo, &pyramid, out.x + col, j);
                rr[col] = px[0] as f32;
                gr[col] = px[1] as f32;
                br[col] = px[2] as f32;
            }
        });
    Ok(result)
}

/// The `out` rectangle of `input`, copied.
pub(crate) fn copy_rect(input: &ImageRef, out: PixelRect) -> Result<Vec<f32>, LensError> {
    if !input.rect().contains(&out) {
        return Err(LensError::InputTooSmall {
            needed: out.to_string(),
            have: input.rect().to_string(),
        });
    }
    let mut result = Vec::with_capacity(3 * out.w * out.h);
    for c in 0..3 {
        let plane = input.plane(c);
        for y in out.y..out.y1() {
            let row = (y - input.y0) * input.width + (out.x - input.x0);
            result.extend_from_slice(&plane[row..row + out.w]);
        }
    }
    Ok(result)
}

/// One output pixel (all three channels), already multiplied by the gain.
fn pixel(geo: &Geometry, pyr: &Pyramid, i: usize, j: usize) -> [f64; 3] {
    let k = geo.pixel(i, j);
    let base_lv = &pyr.levels[0];
    if k.omega == 0.0 {
        // A copy: read the input pixel itself.
        let at = (j as i64 - base_lv.by0) as usize * base_lv.w + (i as i64 - base_lv.bx0) as usize;
        let px = base_lv.data[at];
        return [
            f64::from(px[0]) * k.gain,
            f64::from(px[1]) * k.gain,
            f64::from(px[2]) * k.gain,
        ];
    }
    let (cx, cy) = (i as f64 + 0.5, j as f64 + 0.5);
    let level = level_for(k.omega, geo.opts.spacing_max).min(geo.lmax);
    let scale = f64::from(1u32 << level);
    let inv = 1.0 / scale;
    let n = taps_for(k.omega * inv);
    let lv = &pyr.levels[level as usize];
    let nf = n as f64;
    // Table (tx, ty) -> frame offset omega (tx r + ty t) with r = (cs, sn), t = (sn, -cs). Positions below
    // are texel coordinates of the level, fixed point, relative to its first texel centre.
    let step = 2.0 * k.omega / nf;
    let first = k.omega * (-1.0 + 1.0 / nf);
    let to_texel = |pos: f64, origin: i64| fix(pos * inv - 0.5) - (origin << FRAC);
    let bx = to_texel(cx + first * (k.cs + k.sn), lv.bx0);
    let by = to_texel(cy + first * (k.sn - k.cs), lv.by0);
    let (ea_x, ea_y) = (fix(step * k.cs * inv), fix(step * k.sn * inv));
    let (eb_x, eb_y) = (fix(step * k.sn * inv), fix(-step * k.cs * inv));
    let shifts = geo.fringe_shifts(&k);
    let shifted = shifts.iter().any(|s| s[0] != 0.0 || s[1] != 0.0);
    let shift_fix: [[i64; 2]; 3] =
        std::array::from_fn(|c| [fix(shifts[c][0] * inv), fix(shifts[c][1] * inv)]);

    // The positions the taps take: base + col ea + row eb, col and row in 0..n. Do they stay inside the
    // level? (They do whenever the input covers the halo; the clamped path keeps a short input safe.)
    let last = (n - 1) as i64;
    let reach = shift_fix.iter().flatten().map(|v| v.abs()).max().unwrap_or(0);
    let span = |b: i64, ea: i64, eb: i64| {
        (
            b + (last * ea).min(0) + (last * eb).min(0) - reach,
            b + (last * ea).max(0) + (last * eb).max(0) + reach,
        )
    };
    let (x_lo, x_hi) = span(bx, ea_x, eb_x);
    let (y_lo, y_hi) = span(by, ea_y, eb_y);
    let inside = (x_lo >> FRAC) >= 0
        && (x_hi >> FRAC) <= lv.w as i64 - 2
        && (y_lo >> FRAC) >= 0
        && (y_hi >> FRAC) <= lv.h as i64 - 2;

    let (t0, t1) = (F4::splat((1.0 - k.t) as f32), F4::splat(k.t as f32));
    let lo = geo.tables.table(n, k.seg);
    let hi = geo.tables.table(n, k.seg + 1);
    let mut acc = [0.0f64; 3];
    let mut run = |clamp: bool| {
        if !shifted {
            for row in 0..n {
                let (mut px, mut py) = (bx + row as i64 * eb_x, by + row as i64 * eb_y);
                let wl = &lo[row * n..(row + 1) * n];
                let wh = &hi[row * n..(row + 1) * n];
                let mut sum = F4::splat(0.0);
                for (a, b) in wl.iter().zip(wh) {
                    let s = if clamp {
                        sample::<true>(lv, px, py)
                    } else {
                        sample::<false>(lv, px, py)
                    };
                    let w = t0.mul(F4::load(a)).add(t1.mul(F4::load(b)));
                    sum = sum.add(w.mul(s));
                    px += ea_x;
                    py += ea_y;
                }
                let sum = sum.to_array();
                for c in 0..3 {
                    acc[c] += f64::from(sum[c]);
                }
            }
        } else {
            for c in 0..3 {
                let mut total = 0.0f64;
                for row in 0..n {
                    let mut px = bx + shift_fix[c][0] + row as i64 * eb_x;
                    let mut py = by + shift_fix[c][1] + row as i64 * eb_y;
                    let wl = &lo[row * n..(row + 1) * n];
                    let wh = &hi[row * n..(row + 1) * n];
                    let mut sum = F4::splat(0.0);
                    for (a, b) in wl.iter().zip(wh) {
                        let s = if clamp {
                            sample::<true>(lv, px, py)
                        } else {
                            sample::<false>(lv, px, py)
                        };
                        let w = t0.mul(F4::load(a)).add(t1.mul(F4::load(b)));
                        sum = sum.add(w.mul(s));
                        px += ea_x;
                        py += ea_y;
                    }
                    total += f64::from(sum.to_array()[c]);
                }
                acc[c] = total;
            }
        }
    };
    run(!inside);
    [acc[0] * k.gain, acc[1] * k.gain, acc[2] * k.gain]
}

/// The context `plan` needs around an output rectangle of a `fw x fh` frame (0 without field optics).
pub(crate) fn halo(plan: &LensPlan, fw: usize, fh: usize) -> usize {
    Geometry::new(plan, fw, fh, Options::default()).map_or(0, |g| g.halo)
}
