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
//! attachment the axis part of each channel's centroid offset is multiplied by `1 + h (q - 1)`: after the
//! gather, channel `c` is read `h (q - 1)` times its centroid offset's axis part further along the axis, by
//! linear interpolation between neighbouring pixels (to first order the same as translating the kernel, and a
//! single gather instead of three). The gain is applied last.
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

use crate::consts::{
    COPY_BELOW_PX, MAX_TAPS_SIDE, PSF_SIZE, PSF_WAVELENGTHS, RADII, TAP_SPACING_MAX_PX, W_MAX,
};
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
    /// Context the gather of a pixel reads, in pixels.
    gather_halo: usize,
    /// Pixels the fringe shift reads beyond the output (0 without the attachment's fringe).
    pad: usize,
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
        let (gather_halo, pad) = if omega_max < COPY_BELOW_PX {
            (0, 0)
        } else {
            // A tap lies within sqrt(2) omega of the pixel centre; it reads texels covering P +- 1.5 texels at
            // the coarsest level.
            let reach = omega_max * std::f64::consts::SQRT_2;
            let margin = 1.5 * f64::from(1u32 << lmax);
            let gather_halo = reach.ceil() as usize + margin.ceil() as usize + 3;
            // The fringe shift is the boost times the centroid offset, at most `centroid_max` omega.
            let shift = (fringe - 1.0).max(0.0) * tables.centroid_max * omega_max;
            let pad = if shift > 0.0 {
                shift.ceil() as usize + 2
            } else {
                0
            };
            (gather_halo, pad)
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
            gather_halo,
            pad,
        })
    }

    /// Pixels of context an output rectangle needs around it.
    pub(crate) fn halo(&self) -> usize {
        self.gather_halo + self.pad
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
        let (cs, sn) = if rr > 1e-12 {
            (u / rr, v / rr)
        } else {
            (1.0, 0.0)
        };
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
        [
            (1.0 - k.t) * a[0] + k.t * b[0],
            (1.0 - k.t) * a[1] + k.t * b[1],
        ]
    }

    /// The centroid offset of channel `c` in pixels (frame x, y), without the attachment's fringe boost.
    #[cfg(test)]
    pub(crate) fn centroid_px(&self, k: &PixelKernel, c: usize) -> [f64; 2] {
        let [cx, cy] = self.centroid(k, c);
        // Table x is radial (cs, sn), table y is tangential (sn, -cs).
        [
            k.omega * (cx * k.cs + cy * k.sn),
            k.omega * (cx * k.sn - cy * k.cs),
        ]
    }

    /// The taps of channel `c` of a pixel, as `(dx, dy, weight)` with offsets from the pixel centre in
    /// pixels, for tests and reference checks. Positions and weights are those the gather uses at its mip
    /// level (the tap count `n` is that level's).
    #[cfg(test)]
    pub(crate) fn taps(&self, k: &PixelKernel, c: usize) -> Vec<(f64, f64, f64)> {
        if k.omega == 0.0 {
            return vec![(0.0, 0.0, 1.0)];
        }
        let level = level_for(k.omega, self.opts.spacing_max).min(self.lmax);
        let n = taps_for(k.omega / f64::from(1u32 << level));
        let (lo, hi) = (self.tables.table(n, k.seg), self.tables.table(n, k.seg + 1));
        let mut out = Vec::with_capacity(n * n);
        for b in 0..n {
            for a in 0..n {
                let (tx, ty) = (cell_centre(a, n), cell_centre(b, n));
                let ox = k.omega * (tx * k.cs + ty * k.sn);
                let oy = k.omega * (tx * k.sn - ty * k.cs);
                let q = b * n + a;
                out.push((
                    ox,
                    oy,
                    (1.0 - k.t) * f64::from(lo[q][c]) + k.t * f64::from(hi[q][c]),
                ));
            }
        }
        out
    }

    /// The shift, in pixels along the axis, at which channel `c` is read after the gather: the boost
    /// `h (q - 1)` times the axis part of the channel's centroid offset (0 for a copy).
    pub(crate) fn fringe_shift(&self, k: &PixelKernel, c: usize) -> f64 {
        let boost = self.fringe - 1.0;
        if boost == 0.0 || k.omega == 0.0 {
            return 0.0;
        }
        let a = self.tables.centroid[k.seg][c];
        let b = self.tables.centroid[k.seg + 1][c];
        let cx = (1.0 - k.t) * a[0] + k.t * b[0];
        let cy = (1.0 - k.t) * a[1] + k.t * b[1];
        boost
            * k.omega
            * match self.axis {
                Axis::X => cx * k.cs + cy * k.sn,
                Axis::Y => cx * k.sn - cy * k.cs,
            }
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
            let y = (y0 + row as i64)
                .clamp(0, fh - 1)
                .clamp(iy0, iy0 + ih as i64 - 1);
            let at = (y - iy0) as usize * iw;
            for (col, px) in d.iter_mut().enumerate() {
                let x = (x0 + col as i64)
                    .clamp(0, fw - 1)
                    .clamp(ix0, ix0 + iw as i64 - 1);
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
    let need = out.grown(geo.halo(), fw, fh);
    if !input.rect().contains(&need) {
        return Err(LensError::InputTooSmall {
            needed: need.to_string(),
            have: input.rect().to_string(),
        });
    }
    if out.w == 0 || out.h == 0 {
        return Ok(Vec::new());
    }
    // Pass 1: the gather over the output grown by the fringe's reach. The pyramid covers that grown by the
    // gather's halo, even beyond the frame (replicated).
    let gathered_rect = out.grown(geo.pad, fw, fh);
    let halo = geo.gather_halo as i64;
    let region = [
        gathered_rect.x as i64 - halo,
        gathered_rect.y as i64 - halo,
        gathered_rect.x1() as i64 + halo,
        gathered_rect.y1() as i64 + halo,
    ];
    let pyramid = Pyramid::build(input, region, geo.lmax);
    let gathered = gather_region(&geo, &pyramid, gathered_rect);

    // Pass 2: the fringe shift of each channel and the gain.
    Ok(finish(&geo, &gathered, gathered_rect, out))
}

/// The gather (without the gain) of every pixel of `rect`: planar, `3 * rect.w * rect.h` samples.
fn gather_region(geo: &Geometry, pyr: &Pyramid, rect: PixelRect) -> Vec<f32> {
    let plane = rect.w * rect.h;
    let mut result = vec![0.0f32; 3 * plane];
    let (r, rest) = result.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    r.par_chunks_mut(rect.w)
        .zip(g.par_chunks_mut(rect.w))
        .zip(b.par_chunks_mut(rect.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = rect.y + row;
            for col in 0..rect.w {
                let px = gather(geo, pyr, rect.x + col, j);
                rr[col] = px[0] as f32;
                gr[col] = px[1] as f32;
                br[col] = px[2] as f32;
            }
        });
    result
}

/// The output: every channel of `out` read from `gathered` (the gather over `rect`) at its fringe shift along
/// the axis, times the gain.
fn finish(geo: &Geometry, gathered: &[f32], rect: PixelRect, out: PixelRect) -> Vec<f32> {
    let (gp, op) = (rect.w * rect.h, out.w * out.h);
    let (fw, fh) = (geo.fw as i64, geo.fh as i64);
    let mut result = vec![0.0f32; 3 * op];
    let (r, rest) = result.split_at_mut(op);
    let (g, b) = rest.split_at_mut(op);
    r.par_chunks_mut(out.w)
        .zip(g.par_chunks_mut(out.w))
        .zip(b.par_chunks_mut(out.w))
        .enumerate()
        .for_each(|(row, ((rr, gr), br))| {
            let j = out.y + row;
            for col in 0..out.w {
                let i = out.x + col;
                let k = geo.pixel(i, j);
                let mut px = [0.0f64; 3];
                for (c, v) in px.iter_mut().enumerate() {
                    let plane = &gathered[c * gp..(c + 1) * gp];
                    let shift = geo.fringe_shift(&k, c);
                    *v = if shift == 0.0 {
                        f64::from(plane[(j - rect.y) * rect.w + (i - rect.x)])
                    } else {
                        // Linear interpolation between the two pixels around i + shift (or j + shift).
                        let (along, lo, hi, first, last) = match geo.axis {
                            Axis::X => (i, rect.x, rect.x1(), 0, fw),
                            Axis::Y => (j, rect.y, rect.y1(), 0, fh),
                        };
                        let pos = along as f64 + shift;
                        let i0 = floor_i64(pos);
                        let f = pos - i0 as f64;
                        let at = |q: i64| -> f64 {
                            let q =
                                q.clamp(first, last - 1).clamp(lo as i64, hi as i64 - 1) as usize;
                            f64::from(match geo.axis {
                                Axis::X => plane[(j - rect.y) * rect.w + (q - rect.x)],
                                Axis::Y => plane[(q - rect.y) * rect.w + (i - rect.x)],
                            })
                        };
                        let (a, bb) = (at(i0), at(i0 + 1));
                        a + f * (bb - a)
                    };
                }
                rr[col] = (px[0] * k.gain) as f32;
                gr[col] = (px[1] * k.gain) as f32;
                br[col] = (px[2] * k.gain) as f32;
            }
        });
    result
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

/// One pixel's gather (all three channels), without the gain.
fn gather(geo: &Geometry, pyr: &Pyramid, i: usize, j: usize) -> [f64; 3] {
    let k = geo.pixel(i, j);
    let base_lv = &pyr.levels[0];
    if k.omega == 0.0 {
        // A copy: read the input pixel itself.
        let at = (j as i64 - base_lv.by0) as usize * base_lv.w + (i as i64 - base_lv.bx0) as usize;
        let px = base_lv.data[at];
        return [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
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

    // The positions the taps take: base + col ea + row eb, col and row in 0..n. Do they stay inside the
    // level? (They do whenever the input covers the halo; the clamped path keeps a short input safe.)
    let last = (n - 1) as i64;
    let span = |b: i64, ea: i64, eb: i64| {
        (
            b + (last * ea).min(0) + (last * eb).min(0),
            b + (last * ea).max(0) + (last * eb).max(0),
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
    };
    run(!inside);
    acc
}

/// The context `plan` needs around an output rectangle of a `fw x fh` frame (0 without field optics).
pub(crate) fn halo(plan: &LensPlan, fw: usize, fh: usize) -> usize {
    Geometry::new(plan, fw, fh, Options::default()).map_or(0, |g| g.halo())
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::amounts::{LensAmounts, LensKind};
    use crate::data::LensLibrary;
    use crate::plan::FrameInfo;
    use crate::test_support::write_synthetic_lenses;

    fn library() -> &'static LensLibrary {
        static LIB: OnceLock<LensLibrary> = OnceLock::new();
        LIB.get_or_init(|| {
            let dir = tempfile::tempdir().expect("temp dir");
            write_synthetic_lenses(dir.path()).expect("write lenses");
            LensLibrary::from_dir(dir.path()).expect("load lenses")
        })
    }

    /// The A7R IV's projection: its channels differ, so the lateral colour shows.
    fn camera() -> &'static CameraResponse {
        static CAM: OnceLock<CameraResponse> = OnceLock::new();
        CAM.get_or_init(|| {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
            );
            let curves = ym_color::spectral::SpectralCurves::from_path(path).expect("curves");
            CameraResponse::from_curves(&curves).expect("projection")
        })
    }

    fn plan(
        kind: LensKind,
        edges: f64,
        anamorphic: f64,
        w: usize,
        h: usize,
        axis: Axis,
    ) -> LensPlan {
        let amounts = LensAmounts {
            kind,
            flares: 0.0,
            edges,
            anamorphic,
        };
        let frame = FrameInfo {
            width: w,
            height: h,
            axis,
            f_number: None,
            clip: [1.0; 3],
        };
        LensPlan::new(library(), &amounts, frame, camera()).expect("plan")
    }

    /// A deterministic image with texture at every scale: planar RGB values in 0.05 to 1.
    fn noise(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..3 * w * h)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                0.05 + 0.95 * ((state >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    fn whole(w: usize, h: usize, data: &[f32]) -> ImageRef<'_> {
        ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data,
        }
    }

    #[test]
    fn resample_keeps_mass_and_is_exact_at_33() {
        let src: Vec<f64> = (0..33 * 33).map(|i| 1.0 + f64::from(i % 7)).collect();
        let total: f64 = src.iter().sum();
        assert_eq!(resample(&src, 33), src);
        for n in [3, 5, 9, 17] {
            let r = resample(&src, n);
            assert_eq!(r.len(), n * n);
            assert!(
                (r.iter().sum::<f64>() - total).abs() < 1e-9 * total,
                "n = {n}"
            );
        }
        // A uniform table stays uniform.
        let flat = vec![1.0 / 1089.0; 1089];
        for v in resample(&flat, 5) {
            assert!((v - 1.0 / 25.0).abs() < 1e-12);
        }
    }

    #[test]
    fn level_choice_keeps_the_tap_budget() {
        assert_eq!(level_for(0.6, 1.0), 0);
        assert_eq!(level_for(8.0, 1.0), 0);
        assert_eq!(level_for(8.5, 1.0), 0);
        assert_eq!(level_for(8.6, 1.0), 1);
        assert_eq!(level_for(17.0, 1.0), 1);
        assert_eq!(level_for(17.5, 1.0), 2);
        assert_eq!(level_for(62.0, 1.0), 3);
        for omega in [0.7, 3.2, 9.0, 21.0, 43.0, 62.0] {
            let l = level_for(omega, 1.0);
            let n = taps_for(omega / f64::from(1u32 << l));
            assert!(n <= MAX_TAPS_SIDE);
            assert!(2.0 * omega / f64::from(1u32 << l) / n as f64 <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn kernels_sum_to_one_and_follow_the_projected_centroids() {
        let p = plan(LensKind::Vintage, 100.0, 0.0, 1500, 1000, Axis::X);
        let geo = Geometry::new(&p, 1500, 1000, Options::default()).unwrap();
        let mut checked = 0;
        for (i, j) in [
            (750, 500),
            (1200, 500),
            (300, 120),
            (1400, 950),
            (750, 40),
            (60, 500),
        ] {
            let k = geo.pixel(i, j);
            if k.omega == 0.0 {
                continue;
            }
            checked += 1;
            for c in 0..3 {
                let taps = geo.taps(&k, c);
                let sum: f64 = taps.iter().map(|t| t.2).sum();
                assert!(
                    (sum - 1.0).abs() < 2e-7,
                    "sum {sum} at ({i},{j}) channel {c}"
                );
                let (mx, my) = (
                    taps.iter().map(|t| t.0 * t.2).sum::<f64>(),
                    taps.iter().map(|t| t.1 * t.2).sum::<f64>(),
                );
                let want = geo.centroid_px(&k, c);
                // The area-weighted resampling moves the centroid by a small part of a cell.
                let cell = 2.0 * k.omega
                    / taps_for(k.omega / f64::from(1u32 << level_for(k.omega, 1.0))) as f64;
                assert!(
                    (mx - want[0]).abs() < 0.2 * cell + 1e-6,
                    "x {mx} vs {} at ({i},{j}) c{c}",
                    want[0]
                );
                assert!(
                    (my - want[1]).abs() < 0.2 * cell + 1e-6,
                    "y {my} vs {} at ({i},{j}) c{c}",
                    want[1]
                );
            }
        }
        assert!(checked >= 4);
    }

    #[test]
    fn red_moves_outwards_and_blue_inwards() {
        // Lateral colour of the synthetic lenses: long wavelengths shift radially outwards.
        let p = plan(LensKind::Vintage, 100.0, 0.0, 1500, 1000, Axis::X);
        let geo = Geometry::new(&p, 1500, 1000, Options::default()).unwrap();
        // Right of the centre: radial is +x. Below it: radial is +y. Up-left: radial points up-left.
        for (i, j, sx, sy) in [
            (1300, 500, 1.0, 0.0),
            (750, 900, 0.0, 1.0),
            (200, 100, -1.0, -1.0),
        ] {
            let k = geo.pixel(i, j);
            let r = geo.centroid_px(&k, 0);
            let b = geo.centroid_px(&k, 2);
            let dot = |v: [f64; 2]| v[0] * sx + v[1] * sy;
            assert!(dot(r) > 0.01, "red at ({i},{j}) is {r:?}");
            assert!(dot(b) < -0.01, "blue at ({i},{j}) is {b:?}");
        }
    }

    #[test]
    fn table_y_is_anticlockwise() {
        // A table whose mass sits at +y (tangential) must land anticlockwise of the radial direction on screen
        // (y down): to the right of the centre, anticlockwise is up; below the centre it is to the right.
        let p = plan(LensKind::Default, 100.0, 0.0, 1500, 1000, Axis::X);
        let mut tables = p.field_tables().unwrap().clone();
        for s in 0..SIZES {
            let n = 3 + 2 * s;
            let len = n * n;
            let mut one = vec![[0.0f32; 4]; len];
            // Cell (x = middle, y = last): table position (0, +1 - 1/n).
            one[(n - 1) * n + n / 2] = [1.0, 1.0, 1.0, 0.0];
            for r in 0..RADII {
                tables.tables[s][r * len..(r + 1) * len].copy_from_slice(&one);
            }
        }
        let geo = Geometry {
            tables: &tables,
            ..Geometry::new(&p, 1500, 1000, Options::default()).unwrap()
        };
        let right = geo.pixel(1300, 500);
        let t = geo.taps(&right, 0);
        let heavy = t.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        assert!(
            heavy.1 < -0.5 * right.omega && heavy.0.abs() < 0.2 * right.omega,
            "right: {heavy:?}"
        );
        let below = geo.pixel(750, 900);
        let t = geo.taps(&below, 0);
        let heavy = t.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        assert!(
            heavy.0 > 0.5 * below.omega && heavy.1.abs() < 0.2 * below.omega,
            "below: {heavy:?}"
        );
    }

    /// The box-reduced image the reference reads: texel `(bx, by)` of level `l` is the mean of the aligned block.
    fn ref_texel(img: &[f32], w: usize, h: usize, l: u32, bx: i64, by: i64, c: usize) -> f64 {
        let b = 1i64 << l;
        let mut sum = 0.0;
        for y in by * b..(by + 1) * b {
            for x in bx * b..(bx + 1) * b {
                let (xc, yc) = (
                    x.clamp(0, w as i64 - 1) as usize,
                    y.clamp(0, h as i64 - 1) as usize,
                );
                sum += f64::from(img[c * w * h + yc * w + xc]);
            }
        }
        sum / (b * b) as f64
    }

    fn ref_bilinear(img: &[f32], w: usize, h: usize, l: u32, px: f64, py: f64, c: usize) -> f64 {
        let s = f64::from(1u32 << l);
        let (u, v) = (px / s - 0.5, py / s - 0.5);
        let (i0, j0) = (u.floor(), v.floor());
        let (fx, fy) = (u - i0, v - j0);
        let (i0, j0) = (i0 as i64, j0 as i64);
        let t = |dx, dy| ref_texel(img, w, h, l, i0 + dx, j0 + dy, c);
        (1.0 - fy) * ((1.0 - fx) * t(0, 0) + fx * t(1, 0))
            + fy * ((1.0 - fx) * t(0, 1) + fx * t(1, 1))
    }

    /// The reference gather of one pixel, without the gain: the taps of `Geometry::taps` read from
    /// independently built texels.
    fn ref_gather(img: &[f32], w: usize, h: usize, geo: &Geometry, i: usize, j: usize) -> [f64; 3] {
        let k = geo.pixel(i, j);
        if k.omega == 0.0 {
            return std::array::from_fn(|c| f64::from(img[c * w * h + j * w + i]));
        }
        let level = level_for(k.omega, geo.opts.spacing_max).min(geo.lmax);
        std::array::from_fn(|c| {
            geo.taps(&k, c)
                .iter()
                .map(|(dx, dy, wt)| {
                    wt * ref_bilinear(
                        img,
                        w,
                        h,
                        level,
                        i as f64 + 0.5 + dx,
                        j as f64 + 0.5 + dy,
                        c,
                    )
                })
                .sum()
        })
    }

    /// The reference result of one pixel: each channel's gather read at its fringe shift along the axis (linear
    /// interpolation between the pixels around it, edge pixels replicated), times the gain.
    fn ref_pixel(img: &[f32], w: usize, h: usize, geo: &Geometry, i: usize, j: usize) -> [f64; 3] {
        let k = geo.pixel(i, j);
        std::array::from_fn(|c| {
            let shift = geo.fringe_shift(&k, c);
            let g = |x: i64, y: i64| {
                let (x, y) = (
                    x.clamp(0, w as i64 - 1) as usize,
                    y.clamp(0, h as i64 - 1) as usize,
                );
                ref_gather(img, w, h, geo, x, y)[c]
            };
            let v = if shift == 0.0 {
                g(i as i64, j as i64)
            } else {
                let (along, other) = match geo.axis {
                    Axis::X => (i, j),
                    Axis::Y => (j, i),
                };
                let pos = along as f64 + shift;
                let i0 = pos.floor();
                let f = pos - i0;
                let at = |q: i64| match geo.axis {
                    Axis::X => g(q, other as i64),
                    Axis::Y => g(other as i64, q),
                };
                let (a, b) = (at(i0 as i64), at(i0 as i64 + 1));
                a + f * (b - a)
            };
            v * k.gain
        })
    }

    fn check_against_reference(p: &LensPlan, w: usize, h: usize, opts: Options, tol: f64) {
        let data = noise(w, h, 7);
        let out = apply(p, &whole(w, h, &data), PixelRect::new(0, 0, w, h), opts).unwrap();
        let geo = Geometry::new(p, w, h, opts).unwrap();
        let mut compared = 0;
        let mut wide = 0;
        let mut shifted = 0;
        for j in (0..h).step_by(37).chain([0, h - 1]) {
            for i in (0..w).step_by(41).chain([0, w - 1]) {
                let want = ref_pixel(&data, w, h, &geo, i, j);
                let k = geo.pixel(i, j);
                if k.omega > 0.0 {
                    wide += 1;
                    if (0..3).any(|c| geo.fringe_shift(&k, c).abs() > 0.05) {
                        shifted += 1;
                    }
                }
                for c in 0..3 {
                    let got = f64::from(out[c * w * h + j * w + i]);
                    assert!(
                        (got - want[c]).abs() < tol,
                        "pixel ({i},{j}) channel {c}: {got} vs {}",
                        want[c]
                    );
                }
                compared += 1;
            }
        }
        assert!(
            compared > 100 && wide > 50,
            "{compared} compared, {wide} with a kernel"
        );
        let fringe = p.fringe_factor() > 1.0;
        assert_eq!(
            shifted > 20,
            fringe,
            "{shifted} pixels read at a fringe shift"
        );
    }

    #[test]
    fn gather_matches_an_independent_reference() {
        // Kernels of a few pixels, read at level 0.
        check_against_reference(
            &plan(LensKind::Petzval, 100.0, 0.0, 600, 400, Axis::X),
            600,
            400,
            Options::default(),
            3e-5,
        );
    }

    #[test]
    fn gather_matches_the_reference_with_the_attachment() {
        // Anamorphic 100: oval radius, side softness and the fringe's per-channel shifts.
        check_against_reference(
            &plan(LensKind::Vintage, 60.0, 100.0, 600, 400, Axis::X),
            600,
            400,
            Options::default(),
            3e-5,
        );
        check_against_reference(
            &plan(LensKind::Triplet, 0.0, 100.0, 1500, 1000, Axis::Y),
            1500,
            1000,
            Options::default(),
            3e-5,
        );
    }

    #[test]
    fn gather_matches_the_reference_on_reduced_levels() {
        // A tight spacing makes even small kernels read the 2x, 4x and 8x copies.
        let opts = Options { spacing_max: 0.45 };
        let p = plan(LensKind::Petzval, 100.0, 40.0, 600, 400, Axis::X);
        let geo = Geometry::new(&p, 600, 400, opts).unwrap();
        assert!(geo.lmax >= 2, "lmax {}", geo.lmax);
        check_against_reference(&p, 600, 400, opts, 3e-5);
    }

    #[test]
    fn wide_kernels_use_reduced_levels_at_full_budget() {
        let p = plan(LensKind::Petzval, 100.0, 0.0, 6000, 4000, Axis::X);
        let geo = Geometry::new(&p, 6000, 4000, Options::default()).unwrap();
        // w_max = 0.012 half-diagonals of 3606 px is 43 px; level 3 brings its cells within a texel.
        assert_eq!(geo.lmax, 3);
        assert!(geo.halo() > 50 && geo.halo() < 80, "halo {}", geo.halo());
    }

    #[test]
    fn small_kernels_are_a_copy_times_the_gain() {
        let p = plan(LensKind::Default, 50.0, 0.0, 600, 400, Axis::X);
        let data = noise(600, 400, 3);
        let out = apply(
            &p,
            &whole(600, 400, &data),
            PixelRect::new(290, 190, 20, 20),
            Options::default(),
        )
        .unwrap();
        let geo = Geometry::new(&p, 600, 400, Options::default()).unwrap();
        for j in 190..210 {
            for i in 290..310 {
                let k = geo.pixel(i, j);
                assert_eq!(
                    k.omega, 0.0,
                    "centre kernels of Edges 50 are below half a pixel"
                );
                for c in 0..3 {
                    let want = (f64::from(data[c * 600 * 400 + j * 600 + i]) * k.gain) as f32;
                    assert_eq!(out[c * 400 + (j - 190) * 20 + (i - 290)], want);
                }
            }
        }
    }

    #[test]
    fn mean_tap_count_is_reported() {
        // Not an assertion about speed: a guard that the tap budget holds (at most 17 x 17 a pixel).
        let p = plan(LensKind::Default, 100.0, 0.0, 3000, 2000, Axis::X);
        let geo = Geometry::new(&p, 3000, 2000, Options::default()).unwrap();
        let (mut total, mut count) = (0usize, 0usize);
        for j in (0..2000).step_by(10) {
            for i in (0..3000).step_by(10) {
                let k = geo.pixel(i, j);
                if k.omega > 0.0 {
                    let n = taps_for(k.omega / f64::from(1u32 << level_for(k.omega, 1.0)));
                    assert!(n <= MAX_TAPS_SIDE);
                    total += n * n;
                }
                count += 1;
            }
        }
        println!("mean taps per pixel: {:.1}", total as f64 / count as f64);
        assert!(total as f64 / (count as f64) <= (MAX_TAPS_SIDE * MAX_TAPS_SIDE) as f64);
    }
}
