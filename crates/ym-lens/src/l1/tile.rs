//! The per-tile pass of L1 (`docs/21` section 4, L1, steps 1, 3, 4 and 5).
//!
//! For a rectangle of the frame at scale `k`:
//!
//! 1. the photo's blur map is brought to the tile's pixels by a joint bilateral upsampling guided by the tile's
//!    own pixels (radius 8), and held back where the tile's picture shows detail sharper than the blur says (hair,
//!    fine detail), by the density of such places in a window, so that no single pixel switches;
//! 2. the balls are taken out of the picture (their disc replaced by the plane fit of their ring);
//! 3. two gathers, across the anamorphic axis and then along the tangent, with the per-pixel length set by the
//!    blur, a biweight profile along the line and, for each sample, the weight of a smoothstep of its blur over
//!    the pixel's own;
//! 4. the balls are drawn back as ellipses carrying the energy they had.
//!
//! Pixels whose blur is below `b_min` are never touched. Every pixel is a function of the planes, of input
//! pixels within the halo and of absolute coordinates, in a fixed order, so any tiling gives the same bits as
//! the whole frame.

use rayon::prelude::*;

use crate::consts::{
    ACCEPT_COLOUR, ACCEPT_CONFIDENCE, B_CAP, B_MIN, BALL_RIM_REACH_PX, FEATHER_FRACTION,
    FEATHER_MAX_PX, FEATHER_MIN_PX, GATHER_PAD, GUIDE_SIGMA_MIN, GUIDE_SIGMA_NOISE,
    MAX_STRETCH_TAPS, MIN_LINE_PX, NOISE_FLOOR, PROFILE_SCALE, SAMPLE_RATIO_FROM, SAMPLE_RATIO_TO,
    SHARP_DILATE, SHARP_GATE, SHARP_RATIO, SHARP_SOFT_FROM, SHARP_SOFT_HALF, SHARP_SOFT_TO,
    SHARP_TEST_FROM_PX, STRAND_MARGIN_NOISE, STRAND_VALLEY, SUBJECT_SHARE, UPSAMPLE_RADIUS_PX,
};
use crate::device::{BokehCells, BokehPass, BokehTile};
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::{Axis, LensPlan};

use super::BokehPlanes;
use super::detector::luminance;
use super::draw::{Shape, ShapeParams, profile_at, rim_at, rim_disc_mean, shape_of};

/// What the cells of the blur map around a pixel add up to, by the colour-guided weights of the upsampling.
struct CellSums {
    /// Sum of weight times blur, of weight, and of the weights of the guide (the weights `k` themselves).
    num: f64,
    den: f64,
    nrm: f64,
    /// Sum of weight times the share of the cell known to be in focus.
    sub: f64,
    /// The squared colour distance to the nearest blurred cell of confidence [`ACCEPT_CONFIDENCE`] or more.
    nearest2: f64,
}

/// The cells of one level of the blur map around the pixels of a tile, with the tile's colours.
struct CellView<'a> {
    level: &'a super::MapLevel,
    cols: &'a [(f64, usize, usize)],
    rows: &'a [(f64, usize, usize)],
    cell_guide: &'a [[f32; 3]],
    cx_lo: usize,
    cx_hi: usize,
    cy_lo: usize,
    cy_hi: usize,
    ncx: usize,
    root: &'a RootField,
    sx: f64,
    sy: f64,
    sup: f64,
    sg2: f64,
}

impl CellView<'_> {
    /// The sums over the cells within the support of pixel `(x, y)`.
    fn sums(&self, x: usize, y: usize) -> CellSums {
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        let (sup, level) = (self.sup, self.level);
        let lw = level.w;
        let g = smoothed_root(self.root, x as i64, y as i64);
        let j_lo = (((py - sup) / self.sy).floor().max(0.0) as usize).max(self.cy_lo);
        let j_hi = ((((py + sup) / self.sy).floor() as usize) + 1).min(self.cy_hi);
        let i_lo = (((px - sup) / self.sx).floor().max(0.0) as usize).max(self.cx_lo);
        let i_hi = ((((px + sup) / self.sx).floor() as usize) + 1).min(self.cx_hi);
        let mut c = CellSums {
            num: 0.0,
            den: 0.0,
            nrm: 0.0,
            sub: 0.0,
            nearest2: f64::INFINITY,
        };
        // How far the pixel's colour is from the nearest confident cell's: a pixel of a different colour than every
        // cell around it (the rim of a sharp subject) is not part of their blur.
        for j in j_lo..j_hi {
            let dy = self.rows[j].0 - py;
            for i in i_lo..i_hi {
                let dx = self.cols[i].0 - px;
                let d2 = dx * dx + dy * dy;
                if d2 >= sup * sup {
                    continue;
                }
                let t = 1.0 - d2 / (sup * sup);
                let ws = t * t;
                let cg = &self.cell_guide[(j - self.cy_lo) * self.ncx + (i - self.cx_lo)];
                let (d0, d1, d2c) = (
                    f64::from(g[0] - cg[0]),
                    f64::from(g[1] - cg[1]),
                    f64::from(g[2] - cg[2]),
                );
                let dist2 = d0 * d0 + d1 * d1 + d2c * d2c;
                let tt = 1.0 + dist2 / self.sg2;
                let k = ws / (tt * tt);
                let idx = j * lw + i;
                if level.a[idx] > 0.0 && level.c[idx] >= ACCEPT_CONFIDENCE as f32 {
                    c.nearest2 = c.nearest2.min(dist2);
                }
                c.num += k * f64::from(level.a[idx]);
                c.den += k * f64::from(level.c[idx]);
                c.sub += k * f64::from(level.s[idx]);
                c.nrm += k;
            }
        }
        c
    }
}

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
    /// Half-length of the vertical stretch per unit of blur ([`LensPlan::stretch_per_blur`]).
    stretch: f64,
    /// Width in pixels of the soft edge the stretch has at the border of the blur map (0 without stretch).
    feather: usize,
    /// Map level, and the tile pixels per cell of it.
    level: usize,
    sx: f64,
    sy: f64,
    /// Spatial support of the upsampling, in tile pixels.
    support: f64,
    /// The same numbers in the form the device takes them, which is also where the geometry of a pass over a
    /// rectangle comes from ([`BokehPass::tile`]).
    dev: BokehPass,
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
        let stretch = plan.stretch_per_blur();
        let v_reach = if v_on { stretch * b * hk } else { 0.0 };
        let t_reach = if t_on {
            lens.swirl_max() * e * b * hk * PROFILE_SCALE
        } else {
            0.0
        };
        // The soft edge of the stretch at the map's border: a fraction of the longest line, within bounds.
        let feather = if v_on || t_on {
            ((FEATHER_FRACTION * v_reach.max(t_reach)).round() as usize)
                .clamp(FEATHER_MIN_PX, FEATHER_MAX_PX)
        } else {
            0
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
        let mut pass = Pass {
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
            stretch,
            feather,
            level,
            sx,
            sy,
            support,
            dev: BokehPass {
                width: fw,
                height: fh,
                half_diag: hk,
                noise: 0.0,
                stretch,
                edges: e,
                lines_along_y: plan.frame().axis == Axis::X,
                vertical: v_on,
                tangential: t_on,
                v_halo: 0,
                t_halo: 0,
                feather,
                guide_reach: 0,
                map: (mw, mh),
                level,
                level_size: (bp.level(level).w, bp.level(level).h),
                level_offset: (0..level).map(|l| bp.level(l).w * bp.level(l).h).sum(),
                cell: (sx, sy),
                support,
                sigma_guide: 0.0,
            },
        };
        let noise = pass.noise_t();
        pass.dev.noise = noise;
        pass.dev.sigma_guide = (GUIDE_SIGMA_NOISE * noise).max(GUIDE_SIGMA_MIN);
        pass.dev.v_halo = pass.v_halo();
        pass.dev.t_halo = pass.t_halo();
        pass.dev.guide_reach = pass.guide_reach();
        Some(pass)
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
        self.v_halo() + self.t_halo() + self.guide_reach() + self.feather + SHARP_SOFT_HALF
    }

    /// The rectangles a pass over `want` works on and the cells of the map they read ([`BokehPass::tile`]).
    pub(super) fn rects(&self, want: PixelRect) -> BokehTile {
        self.dev.tile(want)
    }

    /// The numbers of this pass in the form the device takes them.
    pub(super) fn device(&self) -> BokehPass {
        self.dev
    }

    /// What the shapes of the balls are made from, for this scale and plan.
    pub(super) fn shape_params(&self) -> ShapeParams<'a> {
        let lens = self.plan.lens();
        ShapeParams {
            fw: self.fw as f64,
            fh: self.fh as f64,
            hk: self.hk,
            q: self.q,
            e: self.e,
            axis_x: self.axis == Axis::X,
            radii: &lens.field.radii,
            swirl: &lens.field.swirl,
            cats_eye: &lens.field.cats_eye,
            rim: &lens.rim,
            rim_mean: rim_disc_mean(&lens.rim),
        }
    }

    /// The shapes of every ball the pass changes, in list order.
    pub(super) fn shapes(&self) -> Vec<Shape> {
        let sp = self.shape_params();
        self.bp
            .balls()
            .iter()
            .filter_map(|b| shape_of(b, &sp))
            .collect()
    }

    /// The pass over `want`. `input` covers `want` grown by [`Pass::halo`], clamped to the frame.
    pub(super) fn run(&self, input: &ImageRef, want: PixelRect) -> Result<Vec<f32>, LensError> {
        // The blur map is made over `rb`, a feather wider than `r0`, so that the soft edge of the stretch
        // at the border of the map is the same in every tile.
        let BokehTile {
            r1,
            r0,
            rb,
            rbs,
            need,
            ..
        } = self.rects(want);
        if !input.rect().contains(&need) {
            return Err(LensError::InputTooSmall {
                needed: need.to_string(),
                have: input.rect().to_string(),
            });
        }
        let sp = self.shape_params();
        // The balls that change and touch this tile.
        let shapes: Vec<Shape> = self
            .shapes()
            .into_iter()
            .filter(|s| {
                // Everything of the ball that reads or writes pixels of the tile: its oval, its take-out disc and
                // the zone around its rim in which the sharp-edge test is lifted.
                let reach = s
                    .reach
                    .max(s.r_out + 1.0)
                    .max(s.r_ball + BALL_RIM_REACH_PX + 1.0);
                touches(&rbs, s.cx, s.cy, reach)
            })
            .collect();
        // The blur map at the tile's pixels.
        let (bt, be, occ) = self.tile_blur(input, rbs, rb, r0, &shapes);
        if !self.v_on && !self.t_on && shapes.is_empty() {
            return crop_rgb(input, want);
        }
        // Nothing blurred and no ball to draw: the picture as it is. (A ball whose oval reaches into this tile is
        // drawn here whatever the map says, so its disc, which may lie in another tile, decides nothing.)
        if shapes.is_empty() && bt.iter().all(|b| *b == 0.0) {
            return crop_rgb(input, want);
        }
        // The working picture over r0, with the balls taken out.
        let mut wk = crop_rgb(input, r0)?;
        take_out(&mut wk, r0, &bt, &shapes);
        // The two gathers.
        let o1 = if self.v_on {
            self.gather_vertical(&wk, r0, &be, r1)
        } else {
            crop_planes(&wk, r0, r1)
        };
        let mut o2 = if self.t_on {
            self.gather_tangential(&o1, r1, &be, r0, want)
        } else {
            crop_planes(&o1, r1, want)
        };
        // The balls drawn back.
        redraw(&mut o2, want, &bt, &occ, r0, &shapes, &sp);
        Ok(o2)
    }

    /// Noise of the square-root luminance per pixel at this scale.
    fn noise_t(&self) -> f64 {
        let ratio = (self.bp.reduce() * self.plan.lens_plane_scale()) as f64 / self.k as f64;
        (self.bp.noise_v() * ratio).max(NOISE_FLOOR)
    }

    /// The blur map over `out` (made over `rb`, which holds `out` and the feather): upsampled with a joint
    /// bilateral filter, then held back where the picture shows sharp detail (by the density of the sharp places
    /// in a window around the pixel, not pixel by pixel) and below `b_min`; and the stretch's ramp (0 to 1), which
    /// fades the stretch out towards the border of the map. The sharp places are known over `rbs`, `rb` and the
    /// window's reach.
    fn tile_blur(
        &self,
        input: &ImageRef,
        rbs: PixelRect,
        rb: PixelRect,
        out: PixelRect,
        shapes: &[Shape],
    ) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
        let r0 = rb;
        let bp = self.bp;
        let (fw, fh) = (self.fw, self.fh);
        let (mw, mh) = bp.size();
        let level = bp.level(self.level);
        let (lw, lh) = (level.w, level.h);
        let sigma_g = self.dev.sigma_guide;
        let sg2 = sigma_g * sigma_g;
        let sup = self.support;
        // The tile's square-root colours over everything the upsampling reads.
        let reach = self.support.ceil() as usize + (self.sx.max(self.sy) / 2.0).ceil() as usize + 3;
        let er = rbs.grown(reach, fw, fh);
        let root = root_field(input, er);
        // Cell geometry: footprints and centres along each axis.
        let axis_cells = |s: f64, n: usize, full: usize| -> Vec<(f64, usize, usize)> {
            (0..n).map(|c| axis_cell(s, c, full)).collect()
        };
        let cols = axis_cells(self.sx, lw, fw);
        let rows = axis_cells(self.sy, lh, fh);
        // Footprint means of the root colours for the cells within reach of rbs.
        let BokehCells {
            x_lo: cx_lo,
            x_hi: cx_hi,
            y_lo: cy_lo,
            y_hi: cy_hi,
        } = self.dev.cells_of(rbs);
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
        let view = CellView {
            level,
            cols: &cols,
            rows: &rows,
            cell_guide: &cell_guide,
            cx_lo,
            cx_hi,
            cy_lo,
            cy_hi,
            ncx,
            root: &root,
            sx: self.sx,
            sy: self.sy,
            sup,
            sg2,
        };
        let (s0x, s0y) = (fw as f64 / mw as f64, fh as f64 / mh as f64);
        let mut bt = vec![0.0f32; r0.w * r0.h];
        bt.par_chunks_mut(r0.w).enumerate().for_each(|(row, out)| {
            let y = r0.y + row;
            let py = y as f64 + 0.5;
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
                let c = view.sums(x, y);
                if c.nrm > 0.0
                    && c.den >= 0.25 * c.nrm
                    && c.den > 0.0
                    && c.nearest2 <= ACCEPT_COLOUR * ACCEPT_COLOUR
                {
                    let v = c.num / c.den;
                    if v >= B_MIN {
                        *o = v as f32;
                    }
                }
            }
        });
        // The sharp places over rbs, except around a ball: its own rim is a sharp edge, and so is a highlight in
        // it (a glass ornament's, at the centre).
        let hk = self.hk;
        let tested = !shapes.is_empty()
            || bt
                .iter()
                .any(|b| f64::from(*b) * hk >= SHARP_TEST_FROM_PX - 1.0);
        let mut zero = self.sharp_places(input, rbs, tested);
        let zi = |x: usize, y: usize| (y - rbs.y) * rbs.w + (x - rbs.x);
        let planes = [input.plane(0), input.plane(1), input.plane(2)];
        let lum_at = |x: i64, y: i64| -> f64 {
            let xc = x.clamp(input.x0 as i64, (input.x0 + input.width) as i64 - 1) as usize;
            let yc = y.clamp(input.y0 as i64, (input.y0 + input.height) as i64 - 1) as usize;
            let i = (yc - input.y0) * input.width + (xc - input.x0);
            f64::from(luminance(planes[0][i], planes[1][i], planes[2][i]))
        };
        let noise_t = self.noise_t();
        // A thin dark line (a valley: much darker than the pixels three either side of it along a row or a column).
        let thin_dark = |x: i64, y: i64| -> bool {
            let lum = lum_at(x, y);
            let margin = |v: f64| STRAND_MARGIN_NOISE * v.max(0.0).sqrt() * noise_t;
            let valley = |a: f64, b: f64| lum < STRAND_VALLEY * a.min(b) - margin(a.min(b));
            valley(lum_at(x - 3, y), lum_at(x + 3, y)) || valley(lum_at(x, y - 3), lum_at(x, y + 3))
        };
        for s in shapes {
            let reach = s.r_out.max(s.r_ball + BALL_RIM_REACH_PX) + 1.0;
            let x0 = ((s.cx - reach).floor().max(rbs.x as f64)) as usize;
            let x1 = (((s.cx + reach).ceil() as usize) + 1).min(rbs.x1());
            let y0 = ((s.cy - reach).floor().max(rbs.y as f64)) as usize;
            let y1 = (((s.cy + reach).ceil() as usize) + 1).min(rbs.y1());
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f64 + 0.5 - s.cx, y as f64 + 0.5 - s.cy);
                    let d = (dx * dx + dy * dy).sqrt();
                    let lum = lum_at(x as i64, y as i64);
                    // A thin dark line is a strand in front of the ball, not its rim: it stays protected.
                    if thin_dark(x as i64, y as i64) {
                        continue;
                    }
                    // Nor is the edge of a subject beside the light (a hat) the ball's rim.
                    if zero[zi(x, y)] {
                        let c = view.sums(x, y);
                        if c.nrm > 0.0 && c.sub >= SUBJECT_SHARE * c.nrm {
                            continue;
                        }
                    }
                    // Along the rim: a thin strand across the ball or beside it stays protected (see above).
                    let on_rim = (d - s.r_ball).abs() <= BALL_RIM_REACH_PX;
                    // Inside the ball a sharp pixel as bright as the ball is the ball's own light.
                    let core = d < s.r_ball && lum >= s.core;
                    if on_rim || core {
                        zero[zi(x, y)] = false;
                    }
                }
            }
        }
        // The density of the sharp places in the window around a pixel holds the stretch back, smoothly: nothing
        // for a speck, everything for a region of detail or a strand along its length. It acts from a blur of
        // about 4 px at the tile's scale, which the test needs to tell sharp from blurred (a smoothstep over 3 to
        // 5 px, so that no pixel changes abruptly).
        let density = box_mean(
            &Field {
                rect: rbs,
                data: zero.iter().map(|z| f32::from(u8::from(*z))).collect(),
            },
            rb,
            SHARP_SOFT_HALF as i64,
        );
        for (b, d) in bt.iter_mut().zip(&density) {
            let hold = smoothstep(SHARP_SOFT_FROM, SHARP_SOFT_TO, f64::from(*d));
            let px = smoothstep(
                SHARP_TEST_FROM_PX - 1.0,
                SHARP_TEST_FROM_PX + 1.0,
                f64::from(*b) * hk,
            );
            *b = (f64::from(*b) * (1.0 - hold * px)) as f32;
        }
        // Inside a ball's take-out disc the map is the ball's radius: the rim of a light is a colour edge the
        // guided upsampling stops at, which would cut notches into the ball. (Beyond the disc the drawn oval is
        // cut by what is known to be sharp, `occ` below, so a subject in front of the light hides part of it.)
        let mut in_ball = vec![false; r0.w * r0.h];
        for s in shapes {
            let reach = s.r_out;
            let floor_b = ((s.r_ball / hk) as f32).max(B_MIN as f32);
            let x0 = ((s.cx - reach).floor().max(r0.x as f64)) as usize;
            let x1 = (((s.cx + reach).ceil() as usize) + 1).min(r0.x1());
            let y0 = ((s.cy - reach).floor().max(r0.y as f64)) as usize;
            let y1 = (((s.cy + reach).ceil() as usize) + 1).min(r0.y1());
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y - r0.y) * r0.w + (x - r0.x);
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let (dx, dy) = (px - s.cx, py - s.cy);
                    if dx * dx + dy * dy <= s.r_out * s.r_out && !zero[zi(x, y)] {
                        in_ball[i] = true;
                        if bt[i] < B_MIN as f32 {
                            bt[i] = floor_b;
                        }
                    }
                }
            }
        }
        // The ramp: the share of the (2 F + 1)^2 window around a pixel that has a blur, from 0 at one half (the
        // map's border) to 1 at nine tenths, smoothed. A pixel inside a ball has the full stretch.
        let mut ramp = vec![1.0f32; r0.w * r0.h];
        if self.feather > 0 {
            let cover = Field {
                rect: r0,
                data: bt.iter().map(|b| f32::from(*b >= B_MIN as f32)).collect(),
            };
            let share = box_mean(&cover, r0, self.feather as i64);
            for ((r, c), ball) in ramp.iter_mut().zip(&share).zip(&in_ball) {
                if !*ball {
                    let t = ((*c - 0.5) / 0.4).clamp(0.0, 1.0);
                    *r = t * t * (3.0 - 2.0 * t);
                }
            }
        }
        // Where a drawn ball may not go: pixels of a sharp subject, found by the guided share of the cells known to
        // be in focus, or strands (thin dark lines the picture's sharp-edge test flags). Only where an oval would
        // be drawn on a pixel with no blur of its own; elsewhere the map has already decided.
        let mut occ = vec![false; r0.w * r0.h];
        for s in shapes {
            let reach = s.reach;
            let x0 = ((s.cx - reach).floor().max(r0.x as f64)) as usize;
            let x1 = (((s.cx + reach).ceil() as usize) + 1).min(r0.x1());
            let y0 = ((s.cy - reach).floor().max(r0.y as f64)) as usize;
            let y1 = (((s.cy + reach).ceil() as usize) + 1).min(r0.y1());
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y - r0.y) * r0.w + (x - r0.x);
                    if bt[i] >= B_MIN as f32 || occ[i] {
                        continue;
                    }
                    if s.coverage(x as f64 + 0.5, y as f64 + 0.5).0 <= 0.0 {
                        continue;
                    }
                    // (A flag of the sharp-edge test protects a strand, not a speck: only a thin dark line counts.)
                    let c = view.sums(x, y);
                    occ[i] = (zero[zi(x, y)] && thin_dark(x as i64, y as i64))
                        || (c.nrm > 0.0 && c.sub >= SUBJECT_SHARE * c.nrm);
                }
            }
        }
        if let Ok(dir) = std::env::var("YM_L1_DUMP") {
            let wr = |name: &str, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/{name}.f32"), bytes);
            };
            wr("bt", &bt);
            wr("ramp", &ramp);
            wr("density", &density);
            wr("occ", &occ.iter().map(|z| f32::from(*z)).collect::<Vec<_>>());
            let _ = std::fs::write(
                format!("{dir}/rect.txt"),
                format!("{} {} {} {}", r0.x, r0.y, r0.w, r0.h),
            );
        }
        // What the gathers use as every pixel's blur: the map's, held at the cap and faded out towards its border,
        // so that the stretch of a pixel and what it takes from the others both die away at the border.
        let be: Vec<f32> = bt
            .iter()
            .zip(&ramp)
            .map(|(b, r)| (f64::from(*b).min(B_CAP) * f64::from(*r)) as f32)
            .collect();
        (
            crop_scalar(&bt, r0, out),
            crop_scalar(&be, r0, out),
            crop_bool(&occ, r0, out),
        )
    }

    /// Pixels of `rs` next to an edge sharper than their blur.
    ///
    /// The picture's gradient energy is compared with that of its 3 x 3 mean, summed over 5 x 5 windows: a
    /// blurred edge loses almost nothing to a 3 x 3 mean, a sharp one (hair, fine detail) loses a lot. Only
    /// windows with gradient well above the noise count. The result is dilated by 2 pixels.
    ///
    /// The flags are wanted at the pixels blurred by about [`SHARP_TEST_FROM_PX`] or more and around the balls
    /// (`tested`), nowhere else; a region with neither is not tested. (A region's flags never depend on what the
    /// region holds: the same pixel gets the same flag in every tile.)
    fn sharp_places(&self, input: &ImageRef, rs: PixelRect, tested: bool) -> Vec<bool> {
        let (fw, fh) = (self.fw, self.fh);
        if !tested {
            return vec![false; rs.w * rs.h];
        }
        let v = lum_root_field(input, rs.grown(6, fw, fh));
        let s3 = box_mean3(&v, rs.grown(5, fw, fh));
        let p3 = rs.grown(4, fw, fh);
        let ea = grad_energy(&v, p3);
        let eb = grad_energy(&s3, p3);
        let p4 = rs.grown(2, fw, fh);
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
        let rows = window_max(&flags, rs, SHARP_DILATE as i64);
        rows.data.iter().map(|f| *f > 0.0).collect()
    }

    /// The gather across the anamorphic axis over `r1`; `wk` covers `r0`.
    ///
    /// A pixel averages the picture over a line of half-length `l` through it, `2 ceil(l) + 1` evenly spaced taps (at
    /// most 65) read with linear interpolation: its own value with the weight 1, every other tap with the weight of
    /// the profile at its place (a biweight) times the sample rule's smoothstep of its blur over the pixel's own.
    /// The kernel (`lens_reshape.wgsl`) runs the same loops in `f32`.
    fn gather_vertical(&self, wk: &[f32], r0: PixelRect, bt: &[f32], r1: PixelRect) -> Vec<f32> {
        let plane0 = r0.w * r0.h;
        let plane1 = r1.w * r1.h;
        let mut out = vec![0.0f32; 3 * plane1];
        let (a, rest) = out.split_at_mut(plane1);
        let (b, c) = rest.split_at_mut(plane1);
        let along_y = self.axis == Axis::X;
        let (lo, hi) = if along_y {
            (r0.y as i64, r0.y1() as i64 - 1)
        } else {
            (r0.x as i64, r0.x1() as i64 - 1)
        };
        let q1 = self.stretch;
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
                    let len = q1 * bp * self.hk * knee(bp);
                    if len < MIN_LINE_PX {
                        ra[col] = wk[i0];
                        rb[col] = wk[plane0 + i0];
                        rc[col] = wk[2 * plane0 + i0];
                        continue;
                    }
                    let pos = if along_y { y } else { x } as i64;
                    // The index of the pixel at position `j` along the line through `(x, y)`, held inside the
                    // region.
                    let (step, base) = if along_y {
                        (r0.w, x - r0.x)
                    } else {
                        (1, (y - r0.y) * r0.w)
                    };
                    let at = |j: i64| -> usize { base + (j.clamp(lo, hi) - lo) as usize * step };
                    let half = taps_for(len);
                    let tap_step = len / half as f64;
                    let pos = pos as f64;
                    let mut acc = [0.0f64; 3];
                    let mut wsum = 0.0f64;
                    // The pixel's own value (weight 1), then the taps outwards on each side.
                    let own = [wk[i0], wk[plane0 + i0], wk[2 * plane0 + i0]];
                    for ch in 0..3 {
                        acc[ch] = f64::from(own[ch]);
                    }
                    wsum += 1.0;
                    for sign in [-1.0f64, 1.0] {
                        for k in 1..=half {
                            let off = sign * k as f64 * tap_step;
                            let wp = profile(off / len);
                            if wp <= 0.0 {
                                break;
                            }
                            let s = pos + off;
                            let f0 = s.floor();
                            let fr = s - f0;
                            let (j0, j1) = (at(f0 as i64), at(f0 as i64 + 1));
                            let (b0, b1) = (f64::from(bt[j0]), f64::from(bt[j1]));
                            let bs = b0 + fr * (b1 - b0);
                            let wt = wp * sample_weight(bs / bp);
                            if wt <= 0.0 {
                                continue;
                            }
                            for ch in 0..3 {
                                let v0 = f64::from(wk[ch * plane0 + j0]);
                                let v1 = f64::from(wk[ch * plane0 + j1]);
                                acc[ch] += wt * (v0 + fr * (v1 - v0));
                            }
                            wsum += wt;
                        }
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
                    if bp <= 0.5 * B_MIN {
                        copy(ra, rb, rc);
                        continue;
                    }
                    let (dx, dy) = (x as f64 + 0.5 - cxm, y as f64 + 0.5 - cym);
                    let rr = (dx * dx + dy * dy).sqrt();
                    let rho = rr / self.hk;
                    let sigma = profile_at(radii, swirl, rho);
                    let len = sigma * self.e * PROFILE_SCALE * bp * self.hk * knee(bp);
                    if rr < 1e-9 || len < MIN_LINE_PX {
                        copy(ra, rb, rc);
                        continue;
                    }
                    let (ux, uy) = (-dy / rr, dx / rr);
                    let half = taps_for(len);
                    let step = len / half as f64;
                    let mut acc = [0.0f64; 3];
                    let mut wsum = 0.0f64;
                    for ch in 0..3 {
                        acc[ch] = f64::from(o1[ch * plane1 + i1]);
                    }
                    wsum += 1.0;
                    let cl = |v: i64, lo: usize, hi: usize| v.clamp(lo as i64, hi as i64 - 1) as usize;
                    for sign in [-1.0f64, 1.0] {
                        for k in 1..=half {
                            let off = sign * k as f64 * step;
                            let wp = profile(off / len);
                            if wp <= 0.0 {
                                break;
                            }
                            let (sx, sy) = (x as f64 + off * ux, y as f64 + off * uy);
                            let (fx, fy) = (sx.floor(), sy.floor());
                            let (tx, ty) = (sx - fx, sy - fy);
                            // The blur at the sample, bilinear between the four pixels around it.
                            let (bx0, bx1) = (
                                cl(fx as i64, r0.x, r0.x1()),
                                cl(fx as i64 + 1, r0.x, r0.x1()),
                            );
                            let (by0, by1) = (
                                cl(fy as i64, r0.y, r0.y1()),
                                cl(fy as i64 + 1, r0.y, r0.y1()),
                            );
                            let bat = |xx: usize, yy: usize| {
                                f64::from(bt[(yy - r0.y) * r0.w + (xx - r0.x)])
                            };
                            let btop = bat(bx0, by0) + tx * (bat(bx1, by0) - bat(bx0, by0));
                            let bbot = bat(bx0, by1) + tx * (bat(bx1, by1) - bat(bx0, by1));
                            let bs = btop + ty * (bbot - btop);
                            let wt = wp * sample_weight(bs / bp);
                            if wt <= 0.0 {
                                continue;
                            }
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
                                let top = f64::from(p[i00])
                                    + tx * (f64::from(p[i10]) - f64::from(p[i00]));
                                let bot = f64::from(p[i01])
                                    + tx * (f64::from(p[i11]) - f64::from(p[i01]));
                                acc[ch] += wt * (top + ty * (bot - top));
                            }
                            wsum += wt;
                        }
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

/// The weight of a tap at `u` (its offset over the line's half-length, -1 to 1): a biweight, `(1 - u^2)^2`.
#[inline]
fn profile(u: f64) -> f64 {
    let a = 1.0 - u * u;
    if a <= 0.0 { 0.0 } else { a * a }
}

/// `0` below `e0`, `1` above `e1`, a smoothstep between.
#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The weight of a sample whose blur is `ratio` times the pixel's own: nothing from [`SAMPLE_RATIO_FROM`], all of
/// it from [`SAMPLE_RATIO_TO`].
#[inline]
fn sample_weight(ratio: f64) -> f64 {
    smoothstep(SAMPLE_RATIO_FROM, SAMPLE_RATIO_TO, ratio)
}

/// The share of a line a pixel keeps, by its own blur `bp`: nothing at half of `b_min`, all of it from `b_min`.
#[inline]
fn knee(bp: f64) -> f64 {
    smoothstep(0.5 * B_MIN, B_MIN, bp)
}

/// Samples on a line of half-length `len` pixels: the centre and this many either side of it, at most the cap.
fn taps_for(len: f64) -> usize {
    (len.ceil() as usize).clamp(1, (MAX_STRETCH_TAPS - 1) / 2)
}

/// True when the disc of radius `reach` around `(cx, cy)` touches the rectangle.
fn touches(r: &PixelRect, cx: f64, cy: f64, reach: f64) -> bool {
    cx + reach >= r.x as f64
        && cx - reach <= r.x1() as f64
        && cy + reach >= r.y as f64
        && cy - reach <= r.y1() as f64
}

/// Cell `c` of the map's level along an axis of `full` pixels with cells `s` pixels wide: its centre in pixels, and
/// the first pixel and one past the last pixel of its footprint (the pixels whose centres it covers).
pub(super) fn axis_cell(s: f64, c: usize, full: usize) -> (f64, usize, usize) {
    let lo = c as f64 * s;
    let hi = ((c + 1) as f64 * s).min(full as f64);
    let p_lo = ((lo - 0.5).ceil().max(0.0)) as usize;
    let p_hi = (((hi - 0.5).ceil().max(0.0)) as usize).clamp(p_lo + 1, full.max(p_lo + 1));
    (0.5 * (lo + hi), p_lo.min(full - 1), p_hi.min(full))
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

/// The `inner` rectangle of a boolean mask covering `outer`.
fn crop_bool(data: &[bool], outer: PixelRect, inner: PixelRect) -> Vec<bool> {
    let mut out = Vec::with_capacity(inner.w * inner.h);
    for y in inner.y..inner.y1() {
        let row = (y - outer.y) * outer.w + (inner.x - outer.x);
        out.extend_from_slice(&data[row..row + inner.w]);
    }
    out
}

/// The `inner` rectangle of a single-channel buffer covering `outer`.
fn crop_scalar(data: &[f32], outer: PixelRect, inner: PixelRect) -> Vec<f32> {
    let mut out = Vec::with_capacity(inner.w * inner.h);
    for y in inner.y..inner.y1() {
        let row = (y - outer.y) * outer.w + (inner.x - outer.x);
        out.extend_from_slice(&data[row..row + inner.w]);
    }
    out
}

/// The mean of `f` over the `(2 half + 1)^2` window of every pixel of `dst` (reads clamp to `f`'s rectangle).
///
/// `f` holds counts (0 or 1: the pixels with a blur, the sharp places): their sums are whole numbers, which `f64`
/// holds exactly in any order, so running sums cost a few operations a pixel whatever the window, and any tiling
/// of the frame gives the same bits.
fn box_mean(f: &Field, dst: PixelRect, half: i64) -> Vec<f32> {
    let side = (2 * half + 1) as usize;
    let n = (side * side) as f32;
    let width = dst.w + 2 * half as usize;
    let rows = dst.h + 2 * half as usize;
    // The sums along x of the rows the vertical window reaches.
    let mut hs = vec![0.0f64; rows * dst.w];
    hs.par_chunks_mut(dst.w).enumerate().for_each(|(r, row)| {
        let y = dst.y as i64 - half + r as i64;
        let mut pre = vec![0.0f64; width + 1];
        for i in 0..width {
            pre[i + 1] = pre[i] + f64::from(f.get(dst.x as i64 - half + i as i64, y));
        }
        for (c, o) in row.iter_mut().enumerate() {
            *o = pre[c + side] - pre[c];
        }
    });
    // Then down each column.
    let mut out = vec![0.0f32; dst.w * dst.h];
    let columns: Vec<Vec<f32>> = (0..dst.w)
        .into_par_iter()
        .map(|c| {
            let mut acc: f64 = (0..side).map(|r| hs[r * dst.w + c]).sum();
            let mut col = Vec::with_capacity(dst.h);
            col.push(acc as f32 / n);
            for r in 1..dst.h {
                acc += hs[(r + side - 1) * dst.w + c] - hs[(r - 1) * dst.w + c];
                col.push(acc as f32 / n);
            }
            col
        })
        .collect();
    for (c, col) in columns.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            out[r * dst.w + c] = *v;
        }
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

/// Adds every ball's ellipse to `out` (over `want`), except on the pixels of a sharp subject (`occ`) that have no
/// blur of their own: the oval is the ball's light, and a ball is not clipped where the map is merely unsure.
fn redraw(
    out: &mut [f32],
    want: PixelRect,
    bt: &[f32],
    occ: &[bool],
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
                if f64::from(bt[bi]) < B_MIN && occ[bi] {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The variance of the profile over `[-1, 1]` against its mass, by a fine sum.
    fn moments() -> (f64, f64) {
        let n = 200_000;
        let (mut mass, mut var) = (0.0, 0.0);
        for i in 0..n {
            let u = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
            let w = profile(u);
            mass += w;
            var += w * u * u;
        }
        (mass * 2.0 / n as f64, var / mass)
    }

    #[test]
    fn the_profile_has_the_variance_the_scale_assumes() {
        // A biweight has a variance of 1/7 of its squared half-length, a box of 1/3: the gathers' lines are
        // `PROFILE_SCALE` = sqrt(7/3) times the half-length of the box with the same spread.
        let (mass, var) = moments();
        assert!((mass - 16.0 / 15.0).abs() < 1e-6, "mass {mass}");
        assert!((var - 1.0 / 7.0).abs() < 1e-6, "variance {var}");
        let scale = (var * 3.0).recip().sqrt();
        assert!((scale - PROFILE_SCALE).abs() < 1e-6, "{scale}");
    }

    #[test]
    fn a_line_of_the_planned_length_turns_a_disc_into_the_spread_of_the_ellipse() {
        // A disc of radius `b` has a one-dimensional variance of `b^2 / 4` along a line through it; the ellipse
        // `q` times taller has `q^2 b^2 / 4`. The biweight line of the plan's half-length (per unit of `b`) adds
        // its own `l^2 / 7`, so the two are the same, whatever the amount.
        let (_, var) = moments();
        for q in [1.1f64, 1.25, 1.5, 1.75, 2.0] {
            let l = (3.0 * (q * q - 1.0)).sqrt() / 2.0 * PROFILE_SCALE;
            let spread = 0.25 + l * l * var;
            assert!(
                (spread - q * q / 4.0).abs() < 1e-6,
                "q {q}: {spread} against {}",
                q * q / 4.0
            );
        }
    }

    #[test]
    fn the_profile_and_the_sample_weight_are_continuous_and_end_at_zero() {
        assert_eq!(profile(1.0), 0.0);
        assert_eq!(profile(-1.0), 0.0);
        assert_eq!(profile(1.5), 0.0);
        assert_eq!(profile(0.0), 1.0);
        // Smooth at the end: the value and the slope vanish there.
        assert!(profile(0.999) < 1e-5);
        assert_eq!(sample_weight(0.0), 0.0);
        assert_eq!(sample_weight(SAMPLE_RATIO_FROM), 0.0);
        assert_eq!(sample_weight(SAMPLE_RATIO_TO), 1.0);
        assert_eq!(sample_weight(3.0), 1.0);
        let mut prev = 0.0;
        for i in 0..=200 {
            let w = sample_weight(i as f64 / 100.0);
            assert!(w >= prev - 1e-15 && (w - prev).abs() < 0.03, "{i}: {prev} -> {w}");
            prev = w;
        }
        assert_eq!(knee(0.0), 0.0);
        assert_eq!(knee(B_MIN), 1.0);
        assert!(knee(0.75 * B_MIN) > 0.0 && knee(0.75 * B_MIN) < 1.0);
    }
}
