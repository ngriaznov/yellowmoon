//! The per-tile pass of L3 (`docs/21` section 4, L3): the CPU reference.
//!
//! For each output pixel, in `f64`, in a fixed order:
//!
//! 1. the glow's **core**, gathered from the restored picture (the input times the clip test's boost) within
//!    8 px, in four `f32` lanes (red, green, blue);
//! 2. the **tail plane** (glow tail and veil, made once per photo), read at the pixel's position with a
//!    Catmull-Rom curve and held at 0 or above;
//! 3. the **drawn items** (ghosts, starbursts, streaks) whose bounds hold the pixel, in list order;
//!
//! and then `out = (1 - a) Z + F (1 + 2 + 3)`. Everything is a function of the planes, the input within the
//! core's reach and the pixel's absolute position, so any tiling gives the same bits as the whole frame.

use rayon::prelude::*;

use crate::consts::CLIP_RAMP;
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::l3::geometry::{Frame, Span};
use crate::l3::glow::CoreTaps;
use crate::l3::items::Items;
use crate::l3::restore::Restore;
use crate::l3::terms::Terms;
use crate::plan::LensPlan;
use crate::planes::LensPlanes;
use crate::simd::F4;

/// Rows of output a task handles at once (the unit items are culled for).
const BAND_ROWS: usize = 16;

/// Context L3 needs around a rectangle of a `fw x fh` frame at scale `k`: the core's reach in pixels.
pub(crate) fn halo(plan: &LensPlan, planes: &LensPlanes, _k: usize, fw: usize, fh: usize) -> usize {
    let terms = Terms::new(plan);
    if terms.a <= 0.0 {
        return 0;
    }
    let Some(glow) = planes.flare().glow() else {
        return 0;
    };
    let frame = Frame::new(fw, fh, plan.frame().axis);
    let taps = glow.core_taps(frame.half_diag);
    if taps.is_empty() { 0 } else { taps.radius }
}

/// The weights and indices of a Catmull-Rom curve at `t` over `n` samples (held at the ends).
#[inline]
fn catmull_rom(t: f64, n: usize) -> ([usize; 4], [f64; 4]) {
    let i0 = libm::floor(t);
    let f = t - i0;
    let i0 = i0 as isize;
    let last = n as isize - 1;
    let idx = [
        (i0 - 1).clamp(0, last) as usize,
        i0.clamp(0, last) as usize,
        (i0 + 1).clamp(0, last) as usize,
        (i0 + 2).clamp(0, last) as usize,
    ];
    let (f2, f3) = (f * f, f * f * f);
    let w = [
        0.5 * (-f3 + 2.0 * f2 - f),
        0.5 * (3.0 * f3 - 5.0 * f2 + 2.0),
        0.5 * (-3.0 * f3 + 4.0 * f2 + f),
        0.5 * (f3 - f2),
    ];
    (idx, w)
}

/// The restored light of `rect` of the input: one `(red, green, blue, 0)` vector per pixel, row-major.
fn restored_pixels(input: &ImageRef, rs: &Restore, rect: PixelRect) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; rect.w * rect.h];
    let plane_n = input.width * input.height;
    out.par_chunks_mut(rect.w)
        .enumerate()
        .for_each(|(row, px)| {
            let y = rect.y + row - input.y0;
            for (col, p) in px.iter_mut().enumerate() {
                let at = y * input.width + (rect.x + col - input.x0);
                let x = [
                    f64::from(input.data[at]),
                    f64::from(input.data[plane_n + at]),
                    f64::from(input.data[2 * plane_n + at]),
                ];
                let v = rs.restored(x);
                *p = [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
            }
        });
    out
}

/// The core gather of the pixel `(x, y)`: the restored light within the taps' reach, in four `f32` lanes (the
/// sum of at most 289 products, one rounded operation per lane, the same bits on every platform).
#[inline]
fn core_at(
    taps: &CoreTaps,
    restored: &[[f32; 4]],
    region: PixelRect,
    frame: &Frame,
    x: usize,
    y: usize,
) -> [f64; 3] {
    let rr = taps.radius as isize;
    let side = 2 * taps.radius + 1;
    let dx_lo = (-rr).max(-(x as isize));
    let dx_hi = rr.min(frame.w as isize - 1 - x as isize);
    let dy_lo = (-rr).max(-(y as isize));
    let dy_hi = rr.min(frame.h as isize - 1 - y as isize);
    let mut sum = F4::splat(0.0);
    for dy in dy_lo..=dy_hi {
        let row = ((y as isize + dy) as usize - region.y) * region.w;
        let wrow = ((dy + rr) as usize) * side;
        for dx in dx_lo..=dx_hi {
            let col = (x as isize + dx) as usize - region.x;
            let w = F4::load(&taps.lanes[wrow + (dx + rr) as usize]);
            let p = F4::load(&restored[row + col]);
            sum = sum.add(w.mul(p));
        }
    }
    let s = sum.to_array();
    [f64::from(s[0]), f64::from(s[1]), f64::from(s[2])]
}

/// L3 on `out`: `z` is the output of L2 over `out`; `input` is the input of L1 (the core reads it).
pub(crate) fn apply(
    plan: &LensPlan,
    planes: &LensPlanes,
    _k: usize,
    input: &ImageRef,
    z: Vec<f32>,
    out: PixelRect,
) -> Result<Vec<f32>, LensError> {
    let terms = Terms::new(plan);
    if terms.a <= 0.0 || out.w == 0 || out.h == 0 {
        return Ok(z);
    }
    let flare = planes.flare();
    let Some(glow) = flare.glow() else {
        return Err(LensError::Frame(
            "the planes were built without highlight data".into(),
        ));
    };
    if z.len() != 3 * out.w * out.h {
        return Err(LensError::Frame(format!(
            "L3 needs {} samples for {out}, found {}",
            3 * out.w * out.h,
            z.len()
        )));
    }
    let (fw, fh) = (input.frame_width, input.frame_height);
    let frame = Frame::new(fw, fh, plan.frame().axis);

    // The core: the restored picture around the tile.
    let taps = glow.core_taps(frame.half_diag);
    let reach = if taps.is_empty() { 0 } else { taps.radius };
    let region = out.grown(reach, fw, fh);
    if !input.rect().contains(&region) {
        return Err(LensError::InputTooSmall {
            needed: region.to_string(),
            have: input.rect().to_string(),
        });
    }
    let rs = Restore::new(plan, CLIP_RAMP);
    let restored = if taps.is_empty() {
        Vec::new()
    } else {
        restored_pixels(input, &rs, region)
    };

    // The tail plane's sample positions: one set per column and per row of the tile.
    let tail_w = flare.tail(0).1;
    let tail_h = flare.tail(0).2;
    let tail_half = 0.5 * ((tail_w * tail_w + tail_h * tail_h) as f64).sqrt();
    let cols: Vec<([usize; 4], [f64; 4])> = (out.x..out.x1())
        .map(|i| {
            let t = frame.u(i as f64 + 0.5) * tail_half + 0.5 * tail_w as f64 - 0.5;
            catmull_rom(t, tail_w)
        })
        .collect();
    let rows: Vec<([usize; 4], [f64; 4])> = (out.y..out.y1())
        .map(|j| {
            let t = frame.v(j as f64 + 0.5) * tail_half + 0.5 * tail_h as f64 - 0.5;
            catmull_rom(t, tail_h)
        })
        .collect();
    let tails: [&[f64]; 3] = std::array::from_fn(|c| flare.tail(c).0);

    // The drawn items that touch the tile.
    let cull = Span {
        x0: out.x,
        x1: out.x1(),
        y0: out.y,
        y1: out.y1(),
    };
    let items = Items::build(
        plan.lens(),
        plan.attachment(),
        plan.camera(),
        &terms,
        frame,
        flare.sources(),
        flare.spikes(),
        &cull,
    );

    let plane = out.w * out.h;
    let mut result = vec![0.0f32; 3 * plane];
    let (r, rest) = result.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    let (zr, zg, zb) = (&z[..plane], &z[plane..2 * plane], &z[2 * plane..]);
    let keep = 1.0 - terms.a;
    r.par_chunks_mut(BAND_ROWS * out.w)
        .zip(g.par_chunks_mut(BAND_ROWS * out.w))
        .zip(b.par_chunks_mut(BAND_ROWS * out.w))
        .enumerate()
        .for_each(|(band_i, ((rr, gr), br))| {
            let row0 = band_i * BAND_ROWS;
            let rows_here = rr.len() / out.w;
            let band = Span {
                x0: out.x,
                x1: out.x1(),
                y0: out.y + row0,
                y1: out.y + row0 + rows_here,
            };
            let mut acc = vec![[0.0f64; 3]; out.w * rows_here];
            for row in 0..rows_here {
                let j = out.y + row0 + row;
                let (ridx, rw) = &rows[row0 + row];
                for col in 0..out.w {
                    let i = out.x + col;
                    let mut a = if taps.is_empty() {
                        [0.0; 3]
                    } else {
                        core_at(&taps, &restored, region, &frame, i, j)
                    };
                    let (cidx, cw) = &cols[col];
                    for c in 0..3 {
                        let mut v = 0.0;
                        for ry in 0..4 {
                            let base = ridx[ry] * tail_w;
                            let mut s = 0.0;
                            for rx in 0..4 {
                                s += cw[rx] * tails[c][base + cidx[rx]];
                            }
                            v += rw[ry] * s;
                        }
                        a[c] += v.max(0.0);
                    }
                    acc[row * out.w + col] = a;
                }
            }
            for item in &items.list {
                items.add(item, &band, &mut acc);
            }
            for row in 0..rows_here {
                let zrow = (row0 + row) * out.w;
                for col in 0..out.w {
                    let a = acc[row * out.w + col];
                    let at = row * out.w + col;
                    rr[at] = (keep * f64::from(zr[zrow + col]) + terms.f * a[0]) as f32;
                    gr[at] = (keep * f64::from(zg[zrow + col]) + terms.f * a[1]) as f32;
                    br[at] = (keep * f64::from(zb[zrow + col]) + terms.f * a[2]) as f32;
                }
            }
        });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catmull_rom_passes_through_samples_and_sums_to_one() {
        let (idx, w) = catmull_rom(5.0, 10);
        assert_eq!(idx, [4, 5, 6, 7]);
        assert_eq!(w, [0.0, 1.0, 0.0, 0.0]);
        for t in [0.0, 0.3, 4.7, 8.999, -2.5, 12.0] {
            let (_, w) = catmull_rom(t, 10);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14, "t = {t}");
        }
        // Held at the ends.
        let (idx, _) = catmull_rom(-3.0, 10);
        assert_eq!(idx, [0, 0, 0, 0]);
        let (idx, _) = catmull_rom(11.5, 10);
        assert_eq!(idx, [9, 9, 9, 9]);
    }
}
