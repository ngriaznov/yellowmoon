//! The per-tile pass of L1 (`docs/21` section 4, L1, steps 1, 3, 4 and 5).
//!
//! For a rectangle of the frame at scale `k`:
//!
//! 1. the photo's blur map is brought to the tile's pixels by a joint bilateral upsampling guided by the tile's
//!    own pixels (radius 8), and set to 0 where the tile's local gradient energy shows a sharp edge (hair,
//!    fine detail: sharper than the blur says);
//! 2. the balls are taken out of the picture (their disc replaced by the plane fit of their ring);
//! 3. two gathers, across the anamorphic axis and then along the tangent, with the per-pixel length set by the
//!    blur and the rule that a sample counts only when its blur is at least half the pixel's own;
//! 4. the balls are drawn back as ellipses carrying the energy they had.
//!
//! Pixels whose blur is below `b_min` are never touched. Every pixel is a function of the planes, of input
//! pixels within the halo and of absolute coordinates, in a fixed order, so any tiling gives the same bits as
//! the whole frame.

use rayon::prelude::*;

use crate::consts::{B_CAP, B_MIN, MAX_STRETCH_TAPS, UPSAMPLE_RADIUS_PX};
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::{Axis, LensPlan};

use super::BokehPlanes;
use super::draw::{Shape, ShapeParams, profile_at, rim_at, rim_disc_mean, shape_of};
use super::line::{Lines, floor_log2};

/// Pixels farther than this from the edge of a gather's reach are not read: a safety margin for the bilinear
/// neighbour of the last sample.
const GATHER_PAD: usize = 1;
/// The sharp-edge test runs only where the blur is at least this many pixels at the tile's scale.
const SHARP_TEST_FROM_PX: f64 = 4.0;
/// A pixel is sharp when the gradient energy of the picture exceeds that of its 3 x 3 mean by this factor.
const SHARP_RATIO: f64 = 1.4;
/// ... and the mean gradient energy of its 5 x 5 window exceeds this many noise variances.
const SHARP_GATE: f64 = 16.0;
/// Dilation of the sharp places in pixels.
const SHARP_DILATE: i64 = 2;
/// The sharp-edge test sees a ball's rim this far beyond it, in pixels (gradient, box, window and dilation).
const BALL_RIM_REACH_PX: f64 = 7.0;
/// A line shorter than this, in pixels, is not gathered.
const MIN_LINE_PX: f64 = 0.05;

/// A rectangle of values with clamped reads: reads outside the rectangle take the nearest value inside, which
/// is the frame's edge replication wherever the rectangle touches the frame's edge.
struct Field {
    rect: PixelRect,
    data: Vec<f32>,
}

impl Field {
    #[inline]
    fn get(&self, x: i64, y: i64) -> f32 {
        let xx = x.clamp(self.rect.x as i64, self.rect.x1() as i64 - 1) as usize - self.rect.x;
        let yy = y.clamp(self.rect.y as i64, self.rect.y1() as i64 - 1) as usize - self.rect.y;
        self.data[yy * self.rect.w + xx]
    }
}

/// What one pass over a tile shares.
pub(super) struct Pass<'a> {
    bp: &'a BokehPlanes,
    plan: &'a LensPlan,
    fw: usize,
    fh: usize,
    hk: f64,
    k: usize,
    q: f64,
    e: f64,
    axis: Axis,
    v_on: bool,
    t_on: bool,
    /// Longest vertical and tangential line in pixels.
    v_reach: f64,
    t_reach: f64,
    /// Map level, and the tile pixels per cell of it.
    level: usize,
    sx: f64,
    sy: f64,
    /// Spatial support of the upsampling, in tile pixels.
    support: f64,
}

impl<'a> Pass<'a> {
    /// The pass for a frame of `fw x fh` pixels at scale `k`, or `None` when nothing is to be reshaped.
    pub(super) fn new(
        plan: &'a LensPlan,
        bp: &'a BokehPlanes,
        k: usize,
        fw: usize,
        fh: usize,
    ) -> Option<Self> {
        if !bp.active() {
            return None;
        }
        let q = plan.q();
        let e = plan.edges();
        let lens = plan.lens();
        let v_on = q > 1.0 + 1e-12;
        let t_on = e > 0.0 && lens.swirl_max() > 0.0;
        let cat_on = e > 0.0 && lens.cats_eye_max() > 0.0;
        if !(v_on || t_on || cat_on) || bp.balls().is_empty() && !(v_on || t_on) {
            return None;
        }
        let hk = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
        let b = bp.max_blur().min(B_CAP);
        let v_reach = if v_on { (q - 1.0) * b * hk } else { 0.0 };
        let t_reach = if t_on {
            lens.swirl_max() * e * b * hk
        } else {
            0.0
        };
        // The map's cells in tile pixels, at the coarsest level that keeps them at least a pixel apart.
        let (mw, mh) = bp.size();
        let (mut sx, mut sy) = (fw as f64 / mw as f64, fh as f64 / mh as f64);
        let mut level = 0;
        while sx.min(sy) < 1.0 && level + 1 < bp.levels() {
            level += 1;
            sx *= 2.0;
            sy *= 2.0;
        }
        let support = (UPSAMPLE_RADIUS_PX as f64).min(1.6 * sx.max(sy)).max(1.2);
        Some(Pass {
            bp,
            plan,
            fw,
            fh,
            hk,
            k,
            q,
            e,
            axis: plan.frame().axis,
            v_on,
            t_on,
            v_reach,
            t_reach,
            level,
            sx,
            sy,
            support,
        })
    }

    /// Rows of the vertical gather's reach and of the tangential one, with the bilinear neighbour.
    fn v_halo(&self) -> usize {
        if self.v_on {
            self.v_reach.ceil() as usize + GATHER_PAD
        } else {
            0
        }
    }

    fn t_halo(&self) -> usize {
        if self.t_on {
            self.t_reach.ceil() as usize + GATHER_PAD
        } else {
            0
        }
    }

    /// Pixels of the tile's own picture the blur map's upsampling and the sharp-edge test read around a pixel.
    fn guide_reach(&self) -> usize {
        let foot = (self.sx.max(self.sy) / 2.0).ceil() as usize;
        // The upsampling: its support, the footprint of the farthest cell and the guide's 3 x 3 smoothing.
        // The sharp-edge test reaches 6 (gradient, box, 5 x 5 window, dilation).
        (self.support.ceil() as usize + foot + 3).max(7)
    }

    /// The context in pixels the pass needs around the rectangle it produces.
    pub(super) fn halo(&self) -> usize {
        self.v_halo() + self.t_halo() + self.guide_reach()
    }

    /// The pass over `want`. `input` covers `want` grown by [`Pass::halo`], clamped to the frame.
    pub(super) fn run(&self, input: &ImageRef, want: PixelRect) -> Result<Vec<f32>, LensError> {
        let (fw, fh) = (self.fw, self.fh);
        let r1 = want.grown(self.t_halo(), fw, fh);
        let r0 = r1.grown(self.v_halo(), fw, fh);
        let need = r0.grown(self.guide_reach(), fw, fh);
        if !input.rect().contains(&need) {
            return Err(LensError::InputTooSmall {
                needed: need.to_string(),
                have: input.rect().to_string(),
            });
        }
        let lens = self.plan.lens();
        let rim_mean = rim_disc_mean(&lens.rim);
        let sp = ShapeParams {
            fw: fw as f64,
            fh: fh as f64,
            hk: self.hk,
            q: self.q,
            e: self.e,
            axis_x: self.axis == Axis::X,
            radii: &lens.field.radii,
            swirl: &lens.field.swirl,
            cats_eye: &lens.field.cats_eye,
            rim: &lens.rim,
            rim_mean,
        };
        // The balls that change and touch this tile.
        let shapes: Vec<Shape> = self
            .bp
            .balls()
            .iter()
            .filter_map(|b| shape_of(b, &sp))
            .filter(|s| {
                let reach = s.reach.max(s.r_out + 1.0);
                touches(&r0, s.cx, s.cy, reach)
            })
            .collect();
        // The blur map at the tile's pixels.
        let bt = self.tile_blur(input, r0, &shapes);
        if !self.v_on && !self.t_on && shapes.is_empty() {
            return crop_rgb(input, want);
        }
        if bt.iter().all(|b| *b == 0.0) {
            return crop_rgb(input, want);
        }
        // The working picture over r0, with the balls taken out.
        let mut wk = crop_rgb(input, r0)?;
        take_out(&mut wk, r0, &bt, &shapes);
        // The two gathers.
        let o1 = if self.v_on {
            self.gather_vertical(&wk, r0, &bt, r1)
        } else {
            crop_planes(&wk, r0, r1)
        };
        let mut o2 = if self.t_on {
            self.gather_tangential(&o1, r1, &bt, r0, want)
        } else {
            crop_planes(&o1, r1, want)
        };
        // The balls drawn back.
        redraw(&mut o2, want, &bt, r0, &shapes, &sp);
        Ok(o2)
    }

    /// Noise of the square-root luminance per pixel at this scale.
    fn noise_t(&self) -> f64 {
        let ratio = (self.bp.reduce() * self.plan.lens_plane_scale()) as f64 / self.k as f64;
        (self.bp.noise_v() * ratio).max(0.0015)
    }

    /// The blur map over `r0`: upsampled with a joint bilateral filter, then zeroed at sharp edges and below
    /// `b_min`.
    fn tile_blur(&self, input: &ImageRef, r0: PixelRect, shapes: &[Shape]) -> Vec<f32> {
        let bp = self.bp;
        let (fw, fh) = (self.fw, self.fh);
        let (mw, mh) = bp.size();
        let level = bp.level(self.level);
        let (lw, lh) = (level.w, level.h);
        let sigma_g = (3.0 * self.noise_t()).max(0.06);
        let sg2 = sigma_g * sigma_g;
        let sup = self.support;
        // The tile's square-root colours over everything the upsampling reads.
        let reach = self.support.ceil() as usize + (self.sx.max(self.sy) / 2.0).ceil() as usize + 3;
        let er = r0.grown(reach, fw, fh);
        let root = root_field(input, er);
        // Cell geometry: footprints and centres along each axis.
        let axis_cells = |s: f64, n: usize, full: usize| -> Vec<(f64, usize, usize)> {
            (0..n)
                .map(|c| {
                    let lo = c as f64 * s;
                    let hi = ((c + 1) as f64 * s).min(full as f64);
                    let p_lo = ((lo - 0.5).ceil().max(0.0)) as usize;
                    let p_hi =
                        (((hi - 0.5).ceil().max(0.0)) as usize).clamp(p_lo + 1, full.max(p_lo + 1));
                    (0.5 * (lo + hi), p_lo.min(full - 1), p_hi.min(full))
                })
                .collect()
        };
        let cols = axis_cells(self.sx, lw, fw);
        let rows = axis_cells(self.sy, lh, fh);
        // Footprint means of the root colours for the cells within reach of r0.
        let cx_lo = ((r0.x as f64 - sup) / self.sx).floor().max(0.0) as usize;
        let cx_hi = (((r0.x1() as f64 + sup) / self.sx).ceil() as usize + 1).min(lw);
        let cy_lo = ((r0.y as f64 - sup) / self.sy).floor().max(0.0) as usize;
        let cy_hi = (((r0.y1() as f64 + sup) / self.sy).ceil() as usize + 1).min(lh);
        let ncx = cx_hi.saturating_sub(cx_lo);
        let ncy = cy_hi.saturating_sub(cy_lo);
        let mut cell_guide = vec![[0.0f32; 3]; ncx * ncy];
        cell_guide
            .par_chunks_mut(ncx.max(1))
            .enumerate()
            .for_each(|(j, row)| {
                let (_, y0, y1) = rows[cy_lo + j];
                for (i, g) in row.iter_mut().enumerate() {
                    let (_, x0, x1) = cols[cx_lo + i];
                    let mut acc = [0.0f64; 3];
                    let mut n = 0.0f64;
                    for yy in y0..y1 {
                        for xx in x0..x1 {
                            let v = root.get(xx as i64, yy as i64);
                            acc[0] += f64::from(v[0]);
                            acc[1] += f64::from(v[1]);
                            acc[2] += f64::from(v[2]);
                            n += 1.0;
                        }
                    }
                    if n > 0.0 {
                        *g = [
                            (acc[0] / n) as f32,
                            (acc[1] / n) as f32,
                            (acc[2] / n) as f32,
                        ];
                    }
                }
            });
        let (s0x, s0y) = (fw as f64 / mw as f64, fh as f64 / mh as f64);
        let mut bt = vec![0.0f32; r0.w * r0.h];
        bt.par_chunks_mut(r0.w).enumerate().for_each(|(row, out)| {
            let y = r0.y + row;
            let py = y as f64 + 0.5;
            // Rows of cells within support.
            let j_lo = (((py - sup) / self.sy).floor().max(0.0) as usize).max(cy_lo);
            let j_hi = ((((py + sup) / self.sy).floor() as usize) + 1).min(cy_hi);
            for (col, o) in out.iter_mut().enumerate() {
                let x = r0.x + col;
                let px = x as f64 + 0.5;
                // Skip places with no blur anywhere near (level 0 cells).
                let a0 = ((px - sup) / s0x).floor().max(0.0) as usize;
                let a1 = (((px + sup) / s0x).floor() as usize).min(mw - 1);
                let b0 = ((py - sup) / s0y).floor().max(0.0) as usize;
                let b1 = (((py + sup) / s0y).floor() as usize).min(mh - 1);
                if bp.count_blurred(a0, a1, b0, b1) == 0 {
                    continue;
                }
                let g = smoothed_root(&root, x as i64, y as i64);
                let i_lo = (((px - sup) / self.sx).floor().max(0.0) as usize).max(cx_lo);
                let i_hi = ((((px + sup) / self.sx).floor() as usize) + 1).min(cx_hi);
                let (mut num, mut den, mut nrm) = (0.0f64, 0.0f64, 0.0f64);
                for j in j_lo..j_hi {
                    let dy = rows[j].0 - py;
                    for i in i_lo..i_hi {
                        let dx = cols[i].0 - px;
                        let d2 = dx * dx + dy * dy;
                        if d2 >= sup * sup {
                            continue;
                        }
                        let t = 1.0 - d2 / (sup * sup);
                        let ws = t * t;
                        let cg = &cell_guide[(j - cy_lo) * ncx + (i - cx_lo)];
                        let (d0, d1, d2c) = (
                            f64::from(g[0] - cg[0]),
                            f64::from(g[1] - cg[1]),
                            f64::from(g[2] - cg[2]),
                        );
                        let tt = 1.0 + (d0 * d0 + d1 * d1 + d2c * d2c) / sg2;
                        let k = ws / (tt * tt);
                        let idx = j * lw + i;
                        num += k * f64::from(level.a[idx]);
                        den += k * f64::from(level.c[idx]);
                        nrm += k;
                    }
                }
                if nrm > 0.0 && den >= 0.25 * nrm && den > 0.0 {
                    let v = num / den;
                    if v >= B_MIN {
                        *o = v as f32;
                    }
                }
            }
        });
        // Zero the map where the picture shows an edge sharper than the blur says, except around a ball: its
        // own rim is a sharp edge.
        let mut zero = self.sharp_places(input, r0, &bt);
        for s in shapes {
            let reach = s.r_out.max(s.r_ball + BALL_RIM_REACH_PX) + 1.0;
            let x0 = ((s.cx - reach).floor().max(r0.x as f64)) as usize;
            let x1 = (((s.cx + reach).ceil() as usize) + 1).min(r0.x1());
            let y0 = ((s.cy - reach).floor().max(r0.y as f64)) as usize;
            let y1 = (((s.cy + reach).ceil() as usize) + 1).min(r0.y1());
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f64 + 0.5 - s.cx, y as f64 + 0.5 - s.cy);
                    if dx * dx + dy * dy <= (reach - 1.0) * (reach - 1.0) {
                        zero[(y - r0.y) * r0.w + (x - r0.x)] = false;
                    }
                }
            }
        }
        // The test needs a blur wide enough at this scale to tell from a sharp edge.
        let hk = self.hk;
        for (b, z) in bt.iter_mut().zip(&zero) {
            if *z && f64::from(*b) * hk >= SHARP_TEST_FROM_PX {
                *b = 0.0;
            }
        }
        bt
    }

    /// Pixels of `r0` next to an edge sharper than their blur.
    ///
    /// The picture's gradient energy is compared with that of its 3 x 3 mean, summed over 5 x 5 windows: a
    /// blurred edge loses almost nothing to a 3 x 3 mean, a sharp one (hair, fine detail) loses a lot. Only
    /// windows with gradient well above the noise count. The result is dilated by 2 pixels.
    fn sharp_places(&self, input: &ImageRef, r0: PixelRect, bt: &[f32]) -> Vec<bool> {
        let (fw, fh) = (self.fw, self.fh);
        let hk = self.hk;
        // Only where some pixel is blurred enough to tell.
        if !bt.iter().any(|b| f64::from(*b) * hk >= SHARP_TEST_FROM_PX) {
            return vec![false; r0.w * r0.h];
        }
        let v = lum_root_field(input, r0.grown(6, fw, fh));
        let s3 = box_mean3(&v, r0.grown(5, fw, fh));
        let p3 = r0.grown(4, fw, fh);
        let ea = grad_energy(&v, p3);
        let eb = grad_energy(&s3, p3);
        let p4 = r0.grown(2, fw, fh);
        let sa = window_sum(&ea, p4, 2);
        let sb = window_sum(&eb, p4, 2);
        let sigma = self.noise_t();
        let gate = (SHARP_GATE * 25.0 * sigma * sigma) as f32;
        let flags = Field {
            rect: p4,
            data: sa
                .data
                .iter()
                .zip(&sb.data)
                .map(|(a, b)| f32::from(*a > gate && f64::from(*a) > SHARP_RATIO * f64::from(*b)))
                .collect(),
        };
        // Dilate by 2 pixels (5 x 5): rows, then columns.
        let rows = window_max(&flags, r0, SHARP_DILATE);
        rows.data.iter().map(|f| *f > 0.0).collect()
    }

    /// The gather across the anamorphic axis over `r1`; `wk` covers `r0`.
    ///
    /// A pixel whose whole line passes the sample rule (every pixel on it has at least half the pixel's own
    /// blur) gets the exact mean of the picture along the line from running integrals ([`Lines`]); any other
    /// pixel averages explicit taps with the rule applied to each.
    fn gather_vertical(&self, wk: &[f32], r0: PixelRect, bt: &[f32], r1: PixelRect) -> Vec<f32> {
        let plane0 = r0.w * r0.h;
        let plane1 = r1.w * r1.h;
        let mut out = vec![0.0f32; 3 * plane1];
        let (a, rest) = out.split_at_mut(plane1);
        let (b, c) = rest.split_at_mut(plane1);
        let along_y = self.axis == Axis::X;
        let n_line = if along_y { r0.h } else { r0.w };
        let (lo, hi) = if along_y {
            (r0.y as i64, r0.y1() as i64 - 1)
        } else {
            (r0.x as i64, r0.x1() as i64 - 1)
        };
        let q1 = self.q - 1.0;
        // Windows of the sample rule are up to 2 reach + 2 pixels long.
        let max_k = floor_log2(2 * self.v_reach.ceil() as usize + 3);
        let origin = if along_y { r0.y } else { r0.x };
        let lines = Lines::new(wk, bt, (r0.w, r0.h), along_y, origin, max_k);
        a.par_chunks_mut(r1.w)
            .zip(b.par_chunks_mut(r1.w))
            .zip(c.par_chunks_mut(r1.w))
            .enumerate()
            .for_each(|(row, ((ra, rb), rc))| {
                let y = r1.y + row;
                for col in 0..r1.w {
                    let x = r1.x + col;
                    let i0 = (y - r0.y) * r0.w + (x - r0.x);
                    let bp = f64::from(bt[i0]);
                    let len = q1 * bp.min(B_CAP) * self.hk;
                    if bp < B_MIN || len < MIN_LINE_PX {
                        ra[col] = wk[i0];
                        rb[col] = wk[plane0 + i0];
                        rc[col] = wk[2 * plane0 + i0];
                        continue;
                    }
                    let (lx, ly) = (x - r0.x, y - r0.y);
                    let pos = if along_y { y } else { x } as f64;
                    // The pixels the line touches (nearest to its ends), inside the region.
                    let first = (((pos - len + 0.5).floor().max(origin as f64) as usize) - origin)
                        .min(n_line - 1);
                    let last = (((pos + len + 0.5).floor().max(origin as f64) as usize) - origin)
                        .min(n_line - 1);
                    if lines
                        .min_blur(lx, ly, first, last)
                        .is_some_and(|m| f64::from(m) >= 0.5 * bp)
                    {
                        ra[col] = lines.mean(0, lx, ly, pos, len) as f32;
                        rb[col] = lines.mean(1, lx, ly, pos, len) as f32;
                        rc[col] = lines.mean(2, lx, ly, pos, len) as f32;
                        continue;
                    }
                    let (taps, half) = taps_for(len);
                    let step = len / half as f64;
                    let pos = if along_y { y } else { x } as f64;
                    let mut acc = [0.0f64; 3];
                    let mut wsum = 0.0f64;
                    for t in 0..taps {
                        let s = pos + (t as f64 - half as f64) * step;
                        let ni = ((s + 0.5).floor() as i64).clamp(lo, hi);
                        let (nx, ny) = if along_y {
                            (x as i64, ni)
                        } else {
                            (ni, y as i64)
                        };
                        let bs = f64::from(bt[(ny as usize - r0.y) * r0.w + (nx as usize - r0.x)]);
                        if bs < 0.5 * bp {
                            continue;
                        }
                        let wt = if t == 0 || t == taps - 1 { 0.5 } else { 1.0 };
                        let f0 = s.floor();
                        let fr = s - f0;
                        let p0 = (f0 as i64).clamp(lo, hi);
                        let p1 = (f0 as i64 + 1).clamp(lo, hi);
                        let (x0, y0, x1, y1) = if along_y {
                            (x as i64, p0, x as i64, p1)
                        } else {
                            (p0, y as i64, p1, y as i64)
                        };
                        let j0 = (y0 as usize - r0.y) * r0.w + (x0 as usize - r0.x);
                        let j1 = (y1 as usize - r0.y) * r0.w + (x1 as usize - r0.x);
                        for ch in 0..3 {
                            let v0 = f64::from(wk[ch * plane0 + j0]);
                            let v1 = f64::from(wk[ch * plane0 + j1]);
                            acc[ch] += wt * (v0 + fr * (v1 - v0));
                        }
                        wsum += wt;
                    }
                    if wsum > 0.0 {
                        ra[col] = (acc[0] / wsum) as f32;
                        rb[col] = (acc[1] / wsum) as f32;
                        rc[col] = (acc[2] / wsum) as f32;
                    } else {
                        ra[col] = wk[i0];
                        rb[col] = wk[plane0 + i0];
                        rc[col] = wk[2 * plane0 + i0];
                    }
                }
            });
        out
    }

    /// The gather along the tangent over `want`; `o1` covers `r1`, the blur map `r0`.
    fn gather_tangential(
        &self,
        o1: &[f32],
        r1: PixelRect,
        bt: &[f32],
        r0: PixelRect,
        want: PixelRect,
    ) -> Vec<f32> {
        let plane1 = r1.w * r1.h;
        let plane_w = want.w * want.h;
        let mut out = vec![0.0f32; 3 * plane_w];
        let lens = self.plan.lens();
        let (radii, swirl) = (&lens.field.radii, &lens.field.swirl);
        let (cxm, cym) = (0.5 * self.fw as f64, 0.5 * self.fh as f64);
        let (a, rest) = out.split_at_mut(plane_w);
        let (b, c) = rest.split_at_mut(plane_w);
        a.par_chunks_mut(want.w)
            .zip(b.par_chunks_mut(want.w))
            .zip(c.par_chunks_mut(want.w))
            .enumerate()
            .for_each(|(row, ((ra, rb), rc))| {
                let y = want.y + row;
                for col in 0..want.w {
                    let x = want.x + col;
                    let i1 = (y - r1.y) * r1.w + (x - r1.x);
                    let copy = |ra: &mut [f32], rb: &mut [f32], rc: &mut [f32]| {
                        ra[col] = o1[i1];
                        rb[col] = o1[plane1 + i1];
                        rc[col] = o1[2 * plane1 + i1];
                    };
                    let bp = f64::from(bt[(y - r0.y) * r0.w + (x - r0.x)]);
                    if bp < B_MIN {
                        copy(ra, rb, rc);
                        continue;
                    }
                    let (dx, dy) = (x as f64 + 0.5 - cxm, y as f64 + 0.5 - cym);
                    let rr = (dx * dx + dy * dy).sqrt();
                    let rho = rr / self.hk;
                    let sigma = profile_at(radii, swirl, rho);
                    let len = sigma * self.e * bp.min(B_CAP) * self.hk;
                    if rr < 1e-9 || len < MIN_LINE_PX {
                        copy(ra, rb, rc);
                        continue;
                    }
                    let (ux, uy) = (-dy / rr, dx / rr);
                    let (taps, half) = taps_for(len);
                    let step = len / half as f64;
                    let mut acc = [0.0f64; 3];
                    let mut wsum = 0.0f64;
                    for t in 0..taps {
                        let off = (t as f64 - half as f64) * step;
                        let (sx, sy) = (x as f64 + off * ux, y as f64 + off * uy);
                        let nx = ((sx + 0.5).floor() as i64).clamp(r0.x as i64, r0.x1() as i64 - 1)
                            as usize;
                        let ny = ((sy + 0.5).floor() as i64).clamp(r0.y as i64, r0.y1() as i64 - 1)
                            as usize;
                        let bs = f64::from(bt[(ny - r0.y) * r0.w + (nx - r0.x)]);
                        if bs < 0.5 * bp {
                            continue;
                        }
                        let wt = if t == 0 || t == taps - 1 { 0.5 } else { 1.0 };
                        let (fx, fy) = (sx.floor(), sy.floor());
                        let (tx, ty) = (sx - fx, sy - fy);
                        let cl = |v: i64, lo: usize, hi: usize| {
                            v.clamp(lo as i64, hi as i64 - 1) as usize
                        };
                        let (x0, x1) = (
                            cl(fx as i64, r1.x, r1.x1()),
                            cl(fx as i64 + 1, r1.x, r1.x1()),
                        );
                        let (y0, y1) = (
                            cl(fy as i64, r1.y, r1.y1()),
                            cl(fy as i64 + 1, r1.y, r1.y1()),
                        );
                        let idx = |xx: usize, yy: usize| (yy - r1.y) * r1.w + (xx - r1.x);
                        let (i00, i10, i01, i11) =
                            (idx(x0, y0), idx(x1, y0), idx(x0, y1), idx(x1, y1));
                        for ch in 0..3 {
                            let p = &o1[ch * plane1..(ch + 1) * plane1];
                            let top =
                                f64::from(p[i00]) + tx * (f64::from(p[i10]) - f64::from(p[i00]));
                            let bot =
                                f64::from(p[i01]) + tx * (f64::from(p[i11]) - f64::from(p[i01]));
                            acc[ch] += wt * (top + ty * (bot - top));
                        }
                        wsum += wt;
                    }
                    if wsum > 0.0 {
                        ra[col] = (acc[0] / wsum) as f32;
                        rb[col] = (acc[1] / wsum) as f32;
                        rc[col] = (acc[2] / wsum) as f32;
                    } else {
                        copy(ra, rb, rc);
                    }
                }
            });
        out
    }
}

/// Samples on a line of half-length `len` pixels: an odd count with the centre included, at most the cap.
/// Returns the count and the number of steps either side of the centre.
fn taps_for(len: f64) -> (usize, usize) {
    let half = (len.ceil() as usize).clamp(1, (MAX_STRETCH_TAPS - 1) / 2);
    (2 * half + 1, half)
}

/// True when the disc of radius `reach` around `(cx, cy)` touches the rectangle.
fn touches(r: &PixelRect, cx: f64, cy: f64, reach: f64) -> bool {
    cx + reach >= r.x as f64
        && cx - reach <= r.x1() as f64
        && cy + reach >= r.y as f64
        && cy - reach <= r.y1() as f64
}

/// The `rect` of `input`, copied (planar).
fn crop_rgb(input: &ImageRef, rect: PixelRect) -> Result<Vec<f32>, LensError> {
    if !input.rect().contains(&rect) {
        return Err(LensError::InputTooSmall {
            needed: rect.to_string(),
            have: input.rect().to_string(),
        });
    }
    let mut out = Vec::with_capacity(3 * rect.w * rect.h);
    for c in 0..3 {
        let plane = input.plane(c);
        for y in rect.y..rect.y1() {
            let row = (y - input.y0) * input.width + (rect.x - input.x0);
            out.extend_from_slice(&plane[row..row + rect.w]);
        }
    }
    Ok(out)
}

/// The `inner` rectangle of a planar buffer covering `outer`.
fn crop_planes(data: &[f32], outer: PixelRect, inner: PixelRect) -> Vec<f32> {
    let plane = outer.w * outer.h;
    let mut out = Vec::with_capacity(3 * inner.w * inner.h);
    for c in 0..3 {
        for y in inner.y..inner.y1() {
            let row = c * plane + (y - outer.y) * outer.w + (inner.x - outer.x);
            out.extend_from_slice(&data[row..row + inner.w]);
        }
    }
    out
}

/// Values `f(x, y)` over a rectangle, row-parallel.
fn map_rect<T: Send>(rect: PixelRect, f: impl Fn(i64, i64) -> T + Sync) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(rect.w * rect.h);
    let rows: Vec<Vec<T>> = (0..rect.h)
        .into_par_iter()
        .map(|row| {
            let y = (rect.y + row) as i64;
            (0..rect.w).map(|col| f((rect.x + col) as i64, y)).collect()
        })
        .collect();
    for r in rows {
        out.extend(r);
    }
    out
}

/// The 3 x 3 mean of `f` over `dst` (reads clamp to `f`'s rectangle): rows, then columns.
fn box_mean3(f: &Field, dst: PixelRect) -> Field {
    let hs = horizontal(f, dst, 1, |acc, v| acc + v);
    let data = vertical(&hs, dst, 1, |acc, v| acc + v, |sum| sum / 9.0);
    Field { rect: dst, data }
}

/// The sum of `f` over the `(2 half + 1)^2` window of every pixel of `dst`.
fn window_sum(f: &Field, dst: PixelRect, half: i64) -> Field {
    let hs = horizontal(f, dst, half, |acc, v| acc + v);
    let data = vertical(&hs, dst, half, |acc, v| acc + v, |sum| sum);
    Field { rect: dst, data }
}

/// The maximum of `f` over the `(2 half + 1)^2` window of every pixel of `dst`.
fn window_max(f: &Field, dst: PixelRect, half: i64) -> Field {
    let hs = horizontal(f, dst, half, f32::max);
    let data = vertical(&hs, dst, half, f32::max, |m| m);
    Field { rect: dst, data }
}

/// A fold of `f` along `x` over `2 half + 1` pixels around each pixel of `dst`, for every row `f` has that `dst`
/// (grown by `half` rows) needs; the result holds those rows, `dst.w` wide. Reads clamp to `f`'s rectangle.
fn horizontal(
    f: &Field,
    dst: PixelRect,
    half: i64,
    fold: impl Fn(f32, f32) -> f32 + Sync,
) -> (PixelRect, Vec<f32>) {
    let y0 = (dst.y as i64 - half).max(f.rect.y as i64) as usize;
    let y1 = ((dst.y1() as i64 + half).min(f.rect.y1() as i64)) as usize;
    let rows = y1 - y0;
    let rect = PixelRect::new(dst.x, y0, dst.w, rows);
    let mut out = vec![0.0f32; dst.w * rows];
    out.par_chunks_mut(dst.w).enumerate().for_each(|(r, row)| {
        let y = (y0 + r) as i64;
        let (lo, hi) = (f.rect.x as i64, f.rect.x1() as i64 - 1);
        for (c, o) in row.iter_mut().enumerate() {
            let x = (dst.x + c) as i64;
            let mut acc = f.get(x - half, y);
            for d in -half + 1..=half {
                acc = fold(acc, f.get((x + d).clamp(lo, hi), y));
            }
            *o = acc;
        }
    });
    (rect, out)
}

/// The same fold along `y` of the rows `horizontal` made, giving `dst` (reads clamp to those rows).
fn vertical(
    hs: &(PixelRect, Vec<f32>),
    dst: PixelRect,
    half: i64,
    fold: impl Fn(f32, f32) -> f32 + Sync,
    finish: impl Fn(f32) -> f32 + Sync,
) -> Vec<f32> {
    let (rect, rows) = hs;
    let (lo, hi) = (rect.y as i64, rect.y1() as i64 - 1);
    let mut out = vec![0.0f32; dst.w * dst.h];
    out.par_chunks_mut(dst.w).enumerate().for_each(|(r, row)| {
        let y = (dst.y + r) as i64;
        let at = |yy: i64| -> &[f32] {
            let i = (yy.clamp(lo, hi) - rect.y as i64) as usize;
            &rows[i * rect.w..(i + 1) * rect.w]
        };
        for (c, o) in row.iter_mut().enumerate() {
            let mut acc = at(y - half)[c];
            for d in -half + 1..=half {
                acc = fold(acc, at(y + d)[c]);
            }
            *o = finish(acc);
        }
    });
    out
}

/// The gradient energy (central differences, clamped reads) of `f` over `dst`.
fn grad_energy(f: &Field, dst: PixelRect) -> Field {
    let mut data = vec![0.0f32; dst.w * dst.h];
    data.par_chunks_mut(dst.w).enumerate().for_each(|(r, row)| {
        let y = (dst.y + r) as i64;
        for (c, o) in row.iter_mut().enumerate() {
            let x = (dst.x + c) as i64;
            let gx = 0.5 * (f64::from(f.get(x + 1, y)) - f64::from(f.get(x - 1, y)));
            let gy = 0.5 * (f64::from(f.get(x, y + 1)) - f64::from(f.get(x, y - 1)));
            *o = (gx * gx + gy * gy) as f32;
        }
    });
    Field { rect: dst, data }
}

/// Square roots of the three channels of `input` over `rect`, as `[f32; 3]` per pixel with clamped reads.
struct RootField {
    rect: PixelRect,
    data: Vec<[f32; 3]>,
}

impl RootField {
    #[inline]
    fn get(&self, x: i64, y: i64) -> [f32; 3] {
        let xx = x.clamp(self.rect.x as i64, self.rect.x1() as i64 - 1) as usize - self.rect.x;
        let yy = y.clamp(self.rect.y as i64, self.rect.y1() as i64 - 1) as usize - self.rect.y;
        self.data[yy * self.rect.w + xx]
    }
}

fn root_field(input: &ImageRef, rect: PixelRect) -> RootField {
    let planes = [input.plane(0), input.plane(1), input.plane(2)];
    let data = map_rect(rect, |x, y| {
        let i = (y as usize - input.y0) * input.width + (x as usize - input.x0);
        [
            planes[0][i].max(0.0).sqrt(),
            planes[1][i].max(0.0).sqrt(),
            planes[2][i].max(0.0).sqrt(),
        ]
    });
    RootField { rect, data }
}

/// The 3 x 3 (1, 2, 1) smoothed root colours at a pixel.
#[inline]
fn smoothed_root(f: &RootField, x: i64, y: i64) -> [f32; 3] {
    const W: [f32; 3] = [0.25, 0.5, 0.25];
    let mut acc = [0.0f32; 3];
    for (j, wy) in W.iter().enumerate() {
        for (i, wx) in W.iter().enumerate() {
            let v = f.get(x + i as i64 - 1, y + j as i64 - 1);
            let k = wx * wy;
            acc[0] += k * v[0];
            acc[1] += k * v[1];
            acc[2] += k * v[2];
        }
    }
    acc
}

/// The square root of the luminance of `input` over `rect`.
fn lum_root_field(input: &ImageRef, rect: PixelRect) -> Field {
    let planes = [input.plane(0), input.plane(1), input.plane(2)];
    let data = map_rect(rect, |x, y| {
        let i = (y as usize - input.y0) * input.width + (x as usize - input.x0);
        super::detector::luminance(planes[0][i], planes[1][i], planes[2][i])
            .max(0.0)
            .sqrt()
    });
    Field { rect, data }
}

/// Replaces each changed ball's disc by the plane fit of its ring, where the blur map says the pixel is out of
/// focus.
fn take_out(wk: &mut [f32], r0: PixelRect, bt: &[f32], shapes: &[Shape]) {
    let plane = r0.w * r0.h;
    for s in shapes {
        let reach = s.r_out + 1.0;
        let x0 = ((s.cx - reach).floor().max(r0.x as f64)) as usize;
        let x1 = (((s.cx + reach).ceil() as usize) + 1).min(r0.x1());
        let y0 = ((s.cy - reach).floor().max(r0.y as f64)) as usize;
        let y1 = (((s.cy + reach).ceil() as usize) + 1).min(r0.y1());
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = (x as f64 + 0.5 - s.cx, y as f64 + 0.5 - s.cy);
                let m = (s.r_out + 0.5 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
                if m <= 0.0 {
                    continue;
                }
                let i = (y - r0.y) * r0.w + (x - r0.x);
                if f64::from(bt[i]) < B_MIN {
                    continue;
                }
                for c in 0..3 {
                    let f = &s.fit[c];
                    let fit = (f[0] + f[1] * dx + f[2] * dy).max(0.0);
                    let v = f64::from(wk[c * plane + i]);
                    wk[c * plane + i] = (v + m * (fit - v)) as f32;
                }
            }
        }
    }
}

/// Adds every ball's ellipse to `out` (over `want`), where the blur map says the pixel is out of focus.
fn redraw(
    out: &mut [f32],
    want: PixelRect,
    bt: &[f32],
    r0: PixelRect,
    shapes: &[Shape],
    sp: &ShapeParams,
) {
    let plane = want.w * want.h;
    for s in shapes {
        let reach = s.reach;
        let x0 = ((s.cx - reach).floor().max(want.x as f64)) as usize;
        let x1 = (((s.cx + reach).ceil() as usize) + 1).min(want.x1());
        let y0 = ((s.cy - reach).floor().max(want.y as f64)) as usize;
        let y1 = (((s.cy + reach).ceil() as usize) + 1).min(want.y1());
        for y in y0..y1 {
            for x in x0..x1 {
                let bi = (y - r0.y) * r0.w + (x - r0.x);
                if f64::from(bt[bi]) < B_MIN {
                    continue;
                }
                let (cov, rn) = s.coverage(x as f64 + 0.5, y as f64 + 0.5);
                if cov <= 0.0 {
                    continue;
                }
                let k = cov * rim_at(sp.rim, rn);
                let i = (y - want.y) * want.w + (x - want.x);
                for c in 0..3 {
                    out[c * plane + i] += (s.density[c] * k) as f32;
                }
            }
        }
    }
}
