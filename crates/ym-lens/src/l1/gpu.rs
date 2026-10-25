//! L1's data in the form the device kernels take (`crate::device`, `docs/21` section 4, L1, *As built (L1 on the
//! device)*): the blur map's levels and the shapes of the balls.
//!
//! Everything here is made from the CPU reference's own [`tile::Pass`](super::tile::Pass), so the kernel and the
//! reference read one source: the same plan, the same scale, the same shapes (computed in `f64`), the same cell
//! geometry. Only the per-pixel work is written twice.

use crate::consts::{B_MIN, BALL_RIM_REACH_PX};
use crate::device::{
    BOKEH_BIN_PX, BOKEH_HEADER, BOKEH_SHAPE_VECTORS, BokehPass, BokehPlaneData, BokehScaleData,
};
use crate::plan::LensPlan;
use crate::planes::LensPlanes;

use super::draw::{Shape, rim_disc_mean};
use super::tile::{Pass, axis_cell};

/// The blur map of the planes: every level's cells and the table of blurred cells.
pub(crate) fn plane_data(planes: &LensPlanes) -> Option<BokehPlaneData> {
    let bp = planes.bokeh()?;
    if !bp.active() || bp.levels() == 0 {
        return None;
    }
    let mut levels = Vec::with_capacity(bp.levels());
    let mut cells: Vec<[f32; 4]> = Vec::new();
    for i in 0..bp.levels() {
        let l = bp.level(i);
        levels.push((l.w, l.h, cells.len()));
        cells.extend((0..l.w * l.h).map(|j| [l.a[j], l.c[j], l.s[j], 0.0]));
    }
    Some(BokehPlaneData {
        size: bp.size(),
        levels,
        cells,
        nonzero: bp.nonzero.clone(),
    })
}

/// The pass of the plan at scale `k` over a frame of `width x height`, or `None` when L1 leaves the picture alone.
pub(crate) fn device_pass(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    width: usize,
    height: usize,
) -> Option<BokehPass> {
    let bp = planes.bokeh()?;
    Pass::new(plan, bp, k, width, height).map(|p| p.device())
}

/// `values` padded with zeros to `vectors` four-lane vectors.
fn pad4(values: &[f64], vectors: usize) -> Vec<[f32; 4]> {
    (0..vectors)
        .map(|v| {
            let mut out = [0.0f32; 4];
            for (lane, o) in out.iter_mut().enumerate() {
                *o = values.get(4 * v + lane).map_or(0.0, |x| *x as f32);
            }
            out
        })
        .collect()
}

/// A whole number as an `f32` value (exact below 2^24: positions and counts of pixels are far below it), so that
/// no denormal bit pattern is ever read as a float.
fn whole(i: i32) -> f32 {
    i as f32
}

/// A position as its integer part (floor) and the fraction, which the kernel adds to the small difference of
/// the pixel's column with the integer part, so that no `f32` of the order of the frame's size enters a distance.
fn split(x: f64) -> (i32, f32) {
    let f = x.floor();
    (f as i32, (x - f) as f32)
}

/// One ball's record ([`BOKEH_SHAPE_VECTORS`] vectors, `crate::device` documents them).
fn shape_record(s: &Shape, half_diag: f64) -> [[f32; 4]; BOKEH_SHAPE_VECTORS] {
    let (cxi, cxf) = split(s.cx);
    let (cyi, cyf) = split(s.cy);
    // The blur a pixel inside the ball gets where the map has none: the ball's radius, at least `b_min`.
    let floor_b = ((s.r_ball / half_diag) as f32).max(B_MIN as f32);
    let fit = |c: usize| {
        [
            s.fit[c][0] as f32,
            s.fit[c][1] as f32,
            s.fit[c][2] as f32,
            0.0,
        ]
    };
    let owner = |i: usize| {
        let (dx, dy, r) = s.earlier[i];
        [dx as f32, dy as f32, r as f32, 0.0]
    };
    [
        [whole(cxi), cxf, whole(cyi), cyf],
        [s.a as f32, s.b as f32, s.m.0 as f32, s.m.1 as f32],
        [
            s.half_off.0 as f32,
            s.half_off.1 as f32,
            s.delta as f32,
            s.edge as f32,
        ],
        [s.f as f32, 0.0, 0.0, s.r_out as f32],
        [s.r_ball as f32, s.core as f32, s.reach as f32, floor_b],
        fit(0),
        fit(1),
        fit(2),
        owner(0),
        owner(1),
        owner(2),
    ]
}

/// How far from its centre a ball can read or change a pixel: what it draws (`reach`), the disc it takes out of
/// the picture, and the zone around its rim in which the sharp-edge test is lifted; with a margin.
fn influence(s: &Shape) -> f64 {
    s.reach
        .max(s.r_out + 1.0)
        .max(s.r_ball + BALL_RIM_REACH_PX + 1.0)
        + 2.0
}

/// The shapes and profiles of the plan at scale `k` over a frame of `width x height`.
pub(crate) fn scale_data(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    width: usize,
    height: usize,
) -> Option<BokehScaleData> {
    let bp = planes.bokeh()?;
    let pass = Pass::new(plan, bp, k, width, height)?;
    let dev = pass.device();
    let shapes = pass.shapes();
    let lens = plan.lens();
    let mut vectors: Vec<[f32; 4]> = Vec::new();
    vectors.extend(pad4(&lens.field.radii, 3));
    vectors.extend(pad4(&lens.field.swirl, 3));
    vectors.extend(pad4(&lens.rim, 4));
    vectors.push([rim_disc_mean(&lens.rim) as f32, 0.0, 0.0, 0.0]);
    debug_assert_eq!(vectors.len(), BOKEH_HEADER);
    let cell = |s: f64, c: usize, full: usize| {
        let (centre, p_lo, p_hi) = axis_cell(s, c, full);
        let (ci, cf) = split(centre);
        [whole(ci), cf, whole(p_lo as i32), whole(p_hi as i32)]
    };
    let x_offset = vectors.len();
    vectors.extend((0..dev.level_size.0).map(|c| cell(dev.cell.0, c, width)));
    let y_offset = vectors.len();
    vectors.extend((0..dev.level_size.1).map(|c| cell(dev.cell.1, c, height)));
    let shape_offset = vectors.len();
    for s in &shapes {
        vectors.extend(shape_record(s, dev.half_diag));
    }
    // The bins: every ball in the bins its influence's box touches, in list order.
    let (bins_x, bins_y) = (width.div_ceil(BOKEH_BIN_PX), height.div_ceil(BOKEH_BIN_PX));
    let mut per_bin: Vec<Vec<u32>> = vec![Vec::new(); bins_x * bins_y];
    for (i, s) in shapes.iter().enumerate() {
        let r = influence(s);
        let span = |c: f64, n: usize| -> (usize, usize) {
            let lo = ((c - r) / BOKEH_BIN_PX as f64).floor().max(0.0) as usize;
            let hi = (((c + r) / BOKEH_BIN_PX as f64).floor().max(0.0) as usize).min(n - 1);
            (lo.min(n - 1), hi)
        };
        let ((x0, x1), (y0, y1)) = (span(s.cx, bins_x), span(s.cy, bins_y));
        for by in y0..=y1 {
            for bx in x0..=x1 {
                per_bin[by * bins_x + bx].push(i as u32);
            }
        }
    }
    let mut bins =
        Vec::with_capacity(per_bin.len() + 1 + per_bin.iter().map(Vec::len).sum::<usize>());
    let mut at = 0u32;
    bins.push(0);
    for b in &per_bin {
        at += b.len() as u32;
        bins.push(at);
    }
    for b in &per_bin {
        bins.extend_from_slice(b);
    }
    Some(BokehScaleData {
        pass: dev,
        vectors,
        x_offset,
        y_offset,
        shape_offset,
        shapes: shapes.len(),
        bins,
        bins_x,
        bins_y,
    })
}
