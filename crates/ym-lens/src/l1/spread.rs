//! Edge-aware normalised convolution over a fixed pyramid (`docs/21` section 7, step 3).
//!
//! The detector measures the blur at sparse places (edges, bright balls). This module spreads those values to
//! every pixel: a joint bilateral filter guided by the plane's own colours, run on a pyramid of 2x reductions
//! with a small window at every level, whose results are pushed from the coarsest level to the finest with a
//! guided upsampling, finer levels counting more. A value therefore travels along its own surface, over any
//! distance, and stops at colour edges.
//!
//! The input is three sums per pixel (weight `W`, weight times value `S`, weight times value squared `Q`; the
//! value is a log radius); the output is, per pixel, a mean, a variance and how much evidence backs them.
//!
//! Every output sample is a function of fixed windows read in a fixed order, so rows run in parallel with the
//! same bits for any number of threads.

use rayon::prelude::*;

/// Window radius, in cells of a level, of the bilateral smoothing.
const RADIUS: i64 = 1;
/// Spatial standard deviation of the smoothing, in cells.
const SPATIAL_SIGMA: f64 = 0.9;
/// The coarsest level keeps at least this many cells on its shorter side.
const MIN_SIDE: usize = 6;

/// A colour triple per cell.
type Rgb = [f32; 3];

/// One level of the guide pyramid.
struct Level {
    w: usize,
    h: usize,
    guide: Vec<Rgb>,
}

/// The guide image (the plane's colours) and its reductions.
pub(super) struct GuidePyramid {
    levels: Vec<Level>,
    /// Squared range tolerance: colours further apart than this (in the guide's units) count little.
    sigma_g2: f64,
}

/// `1 / (1 + d^2 / sigma^2)^2`: the range weight of two colours `d` apart.
#[inline(always)]
fn range_weight(a: &Rgb, b: &Rgb, sigma_g2: f64) -> f64 {
    let (dr, dg, db) = (
        f64::from(a[0] - b[0]),
        f64::from(a[1] - b[1]),
        f64::from(a[2] - b[2]),
    );
    let t = 1.0 + (dr * dr + dg * dg + db * db) / sigma_g2;
    1.0 / (t * t)
}

/// Mean-pools `src` (`w x h`) to `(w + 1) / 2 x (h + 1) / 2`; a cell is the mean of the children it has.
fn pool<T, F>(
    src: &[T],
    w: usize,
    h: usize,
    zero: T,
    add: F,
    scale: impl Fn(T, f32) -> T + Sync,
) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Sync,
{
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![zero; w2 * h2];
    out.par_chunks_mut(w2).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = zero;
            let mut n = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < w && sy < h {
                        acc = add(acc, src[sy * w + sx]);
                        n += 1.0;
                    }
                }
            }
            *o = scale(acc, 1.0 / n);
        }
    });
    out
}

impl GuidePyramid {
    /// The pyramid of `base` (`w x h`); `sigma_g` is the range tolerance in the guide's units.
    pub(super) fn new(base: Vec<Rgb>, w: usize, h: usize, sigma_g: f64) -> Self {
        let mut levels = vec![Level { w, h, guide: base }];
        loop {
            let last = levels.last().expect("at least one level");
            if last.w.min(last.h) < 2 * MIN_SIDE {
                break;
            }
            let guide = pool(
                &last.guide,
                last.w,
                last.h,
                [0.0f32; 3],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                |a, s| [a[0] * s, a[1] * s, a[2] * s],
            );
            let (w2, h2) = (last.w.div_ceil(2), last.h.div_ceil(2));
            levels.push(Level {
                w: w2,
                h: h2,
                guide,
            });
        }
        GuidePyramid {
            levels,
            sigma_g2: sigma_g * sigma_g,
        }
    }

    /// Number of levels.
    #[cfg(test)]
    pub(super) fn depth(&self) -> usize {
        self.levels.len()
    }
}

/// What the spread gives at every pixel of the finest level.
pub(super) struct Spread {
    /// How much of the pixel's value is backed by evidence (0 to 1): 1 where readings are dense enough nearby at
    /// some level, less where only a coarser level or nothing reaches.
    pub cover: Vec<f32>,
    /// Mean of the readings that reach the pixel.
    pub mean: Vec<f32>,
    /// Variance of those readings.
    pub var: Vec<f32>,
}

/// The sums of one level: weight, weight times value, weight times value squared, per cell.
type Sums = [f32; 3];

/// The bilateral smoothing of one level's sums with a `(2 RADIUS + 1)^2` window.
fn smooth(level: &Level, sums: &[Sums], sigma_g2: f64) -> Vec<Sums> {
    let (w, h) = (level.w, level.h);
    let side = (2 * RADIUS + 1) as usize;
    let spatial: Vec<f64> = (0..side * side)
        .map(|i| {
            let (dx, dy) = (
                (i % side) as f64 - RADIUS as f64,
                (i / side) as f64 - RADIUS as f64,
            );
            libm::exp(-(dx * dx + dy * dy) / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA))
        })
        .collect();
    let mut out = vec![[0.0f32; 3]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let centre = &level.guide[y * w + x];
            let mut acc = [0.0f64; 3];
            for dy in -RADIUS..=RADIUS {
                let ny = y as i64 + dy;
                if ny < 0 || ny >= h as i64 {
                    continue;
                }
                for dx in -RADIUS..=RADIUS {
                    let nx = x as i64 + dx;
                    if nx < 0 || nx >= w as i64 {
                        continue;
                    }
                    let i = ny as usize * w + nx as usize;
                    let s = &sums[i];
                    if s[0] == 0.0 {
                        continue;
                    }
                    let k = spatial[((dy + RADIUS) as usize) * side + (dx + RADIUS) as usize]
                        * range_weight(centre, &level.guide[i], sigma_g2);
                    acc[0] += k * f64::from(s[0]);
                    acc[1] += k * f64::from(s[1]);
                    acc[2] += k * f64::from(s[2]);
                }
            }
            *o = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
        }
    });
    out
}

/// Guided upsampling of the parent level's result `(cover, cover * mean, cover * second moment)` to the child
/// level.
fn upsample(child: &Level, parent: &Level, parent_sums: &[Sums], sigma_g2: f64) -> Vec<Sums> {
    let (w, h) = (child.w, child.h);
    let mut out = vec![[0.0f32; 3]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            // The child cell centre in parent cell coordinates (parent cell i has its centre at i).
            let px = x as f64 / 2.0 - 0.25;
            let py = y as f64 / 2.0 - 0.25;
            let (ix, iy) = (px.floor() as i64, py.floor() as i64);
            let (fx, fy) = (px - ix as f64, py - iy as f64);
            let centre = &child.guide[y * w + x];
            let mut acc = [0.0f64; 3];
            let mut wsum = 0.0f64;
            for (oy, wy) in [(0i64, 1.0 - fy), (1, fy)] {
                for (ox, wx) in [(0i64, 1.0 - fx), (1, fx)] {
                    let (nx, ny) = (ix + ox, iy + oy);
                    if nx < 0 || ny < 0 || nx >= parent.w as i64 || ny >= parent.h as i64 {
                        continue;
                    }
                    let i = ny as usize * parent.w + nx as usize;
                    let k = wx * wy * (range_weight(centre, &parent.guide[i], sigma_g2) + 1e-4);
                    let s = &parent_sums[i];
                    acc[0] += k * f64::from(s[0]);
                    acc[1] += k * f64::from(s[1]);
                    acc[2] += k * f64::from(s[2]);
                    wsum += k;
                }
            }
            if wsum > 0.0 {
                *o = [
                    (acc[0] / wsum) as f32,
                    (acc[1] / wsum) as f32,
                    (acc[2] / wsum) as f32,
                ];
            }
        }
    });
    out
}

/// Spreads the sums `(w0, s0, q0)` (one entry per pixel of the finest level) over the guide's pyramid.
///
/// Every level smooths its own cells' sums; a level is trusted as far as its window holds `need` of weight
/// (readings dense enough nearby); what a level does not cover is filled from the next coarser one, by a
/// guided upsampling: pull-push interpolation of sparse readings.
pub(super) fn spread(
    guide: &GuidePyramid,
    w0: &[f32],
    s0: &[f32],
    q0: &[f32],
    need: f64,
) -> Spread {
    let base = &guide.levels[0];
    debug_assert_eq!(w0.len(), base.w * base.h);
    let n = base.w * base.h;
    let mut sums: Vec<Vec<Sums>> = Vec::with_capacity(guide.levels.len());
    let mut first = Vec::with_capacity(n);
    for i in 0..n {
        first.push([w0[i], s0[i], q0[i]]);
    }
    sums.push(first);
    for l in 1..guide.levels.len() {
        let prev = &guide.levels[l - 1];
        let pooled = pool(
            &sums[l - 1],
            prev.w,
            prev.h,
            [0.0f32; 3],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            |a, s| [a[0] * s, a[1] * s, a[2] * s],
        );
        sums.push(pooled);
    }
    // Smooth every level.
    let smoothed: Vec<Vec<Sums>> = guide
        .levels
        .iter()
        .zip(&sums)
        .map(|(lv, s)| smooth(lv, s, guide.sigma_g2))
        .collect();
    drop(sums);
    // Push from the coarsest level to the finest: `acc` holds (cover, cover * mean, cover * second moment).
    let depth = guide.levels.len();
    let combine = |own: &Sums, up: Option<&Sums>| -> Sums {
        let alpha = (f64::from(own[0]) / need).min(1.0);
        let (mut cov, mut m1, mut m2) = (0.0f64, 0.0f64, 0.0f64);
        if own[0] > 0.0 {
            let w = f64::from(own[0]);
            cov = alpha;
            m1 = alpha * f64::from(own[1]) / w;
            m2 = alpha * f64::from(own[2]) / w;
        }
        if let Some(u) = up {
            let rest = 1.0 - alpha;
            cov += rest * f64::from(u[0]);
            m1 += rest * f64::from(u[1]);
            m2 += rest * f64::from(u[2]);
        }
        [cov as f32, m1 as f32, m2 as f32]
    };
    let mut acc: Vec<Sums> = smoothed[depth - 1]
        .iter()
        .map(|s| combine(s, None))
        .collect();
    for l in (0..depth - 1).rev() {
        let up = upsample(&guide.levels[l], &guide.levels[l + 1], &acc, guide.sigma_g2);
        acc = smoothed[l]
            .par_iter()
            .zip(up.par_iter())
            .map(|(s, u)| combine(s, Some(u)))
            .collect();
    }
    let mut out = Spread {
        cover: Vec::with_capacity(n),
        mean: Vec::with_capacity(n),
        var: Vec::with_capacity(n),
    };
    for a in &acc {
        let cov = f64::from(a[0]);
        if cov > 1e-9 {
            let mean = f64::from(a[1]) / cov;
            let var = (f64::from(a[2]) / cov - mean * mean).max(0.0);
            out.cover.push(cov.min(1.0) as f32);
            out.mean.push(mean as f32);
            out.var.push(var as f32);
        } else {
            out.cover.push(0.0);
            out.mean.push(0.0);
            out.var.push(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane with two flat colour regions, left and right, and one value on each side.
    fn two_regions(
        w: usize,
        h: usize,
        edge: usize,
    ) -> (GuidePyramid, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut guide = vec![[0.0f32; 3]; w * h];
        let (mut ws, mut ss, mut qs) = (
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
        );
        for y in 0..h {
            for x in 0..w {
                guide[y * w + x] = if x < edge { [0.2; 3] } else { [0.8; 3] };
            }
        }
        // Samples on a grid, ln r = 1 on the left and 3 on the right.
        for y in (8..h).step_by(24) {
            for x in (8..w).step_by(24) {
                let v = if x < edge { 1.0f32 } else { 3.0 };
                let i = y * w + x;
                ws[i] = 1.0;
                ss[i] = v;
                qs[i] = v * v;
            }
        }
        (GuidePyramid::new(guide, w, h, 0.1), ws, ss, qs)
    }

    #[test]
    fn a_value_spreads_over_its_surface_and_stops_at_the_colour_edge() {
        let (w, h, edge) = (96, 64, 52);
        let (g, ws, ss, qs) = two_regions(w, h, edge);
        assert!(g.depth() >= 4);
        let sp = spread(&g, &ws, &ss, &qs, 0.05);
        for (x, y, expect) in [(5, 5, 1.0f32), (30, 60, 1.0), (62, 3, 3.0), (90, 50, 3.0)] {
            let i = y * w + x;
            assert!(
                sp.cover[i] > 0.3,
                "no evidence at ({x}, {y}): {}",
                sp.cover[i]
            );
            let mean = sp.mean[i];
            assert!(
                (mean - expect).abs() < 0.05,
                "({x}, {y}): {mean} vs {expect}"
            );
            assert!(sp.var[i] < 0.05, "variance {} at ({x}, {y})", sp.var[i]);
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let (w, h, edge) = (96, 64, 40);
        let (g, ws, ss, qs) = two_regions(w, h, edge);
        let a = spread(&g, &ws, &ss, &qs, 0.2);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| spread(&g, &ws, &ss, &qs, 0.2));
        assert_eq!(a.cover, b.cover);
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.var, b.var);
    }
}
