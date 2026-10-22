//! The evidence for the subject map (`docs/20-auto-crop.md` §6 steps 1 and 2): the minimum barrier
//! saliency of Zhang, Sclaroff, Lin, Shen, Price and Mech, *Minimum Barrier Salient Object
//! Detection at 80 FPS* (ICCV 2015), written from the paper, and a focus measure.

use crate::picture::{CELL, Plane, percentile};

/// Pairs of raster passes of the barrier distance.
const MBD_PASS_PAIRS: usize = 3;
/// The saliency is divided by this percentile of itself.
const SALIENCY_PERCENTILE: f64 = 0.99;
/// The focus measure is divided by this percentile of itself.
const FOCUS_PERCENTILE: f64 = 0.95;

/// The saliency `M` of a map of CIELAB cells: the minimum barrier distance to the border on L, a
/// and b, summed, divided by its 99th percentile and clamped to 1.
pub(crate) fn saliency(lab: &[Plane; 3]) -> Plane {
    let (w, h) = (lab[0].width, lab[0].height);
    let mut sum = Plane::zeros(w, h);
    for channel in lab {
        let d = barrier_distance(channel);
        for (s, v) in sum.data.iter_mut().zip(&d.data) {
            *s += v;
        }
    }
    let p = percentile(&sum.data, SALIENCY_PERCENTILE);
    for v in &mut sum.data {
        *v = if p > 0.0 { (*v / p).min(1.0) } else { 0.0 };
    }
    sum
}

/// The minimum barrier distance of every cell to the border of the plane by the raster-scan
/// approximation: border cells 0, the others infinite, then [`MBD_PASS_PAIRS`] pairs of passes
/// (forward with the upper and left neighbours, backward with the lower and right ones), each cell
/// keeping the highest and lowest value along its best path so far.
fn barrier_distance(img: &Plane) -> Plane {
    let (w, h) = (img.width, img.height);
    let mut dist = Plane::zeros(w, h);
    let mut high = img.data.clone();
    let mut low = img.data.clone();
    for y in 0..h {
        for x in 0..w {
            if !(x == 0 || y == 0 || x + 1 == w || y + 1 == h) {
                dist.data[y * w + x] = f64::INFINITY;
            }
        }
    }
    let mut relax = |i: usize, n: usize, dist: &mut Plane| {
        if dist.data[n].is_infinite() {
            return;
        }
        let v = img.data[i];
        let hi = high[n].max(v);
        let lo = low[n].min(v);
        if hi - lo < dist.data[i] {
            dist.data[i] = hi - lo;
            high[i] = hi;
            low[i] = lo;
        }
    };
    for _ in 0..MBD_PASS_PAIRS {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if y > 0 {
                    relax(i, i - w, &mut dist);
                }
                if x > 0 {
                    relax(i, i - 1, &mut dist);
                }
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if y + 1 < h {
                    relax(i, i + w, &mut dist);
                }
                if x + 1 < w {
                    relax(i, i + 1, &mut dist);
                }
            }
        }
    }
    // A cell no path reached (none in a connected grid) has no barrier.
    for v in &mut dist.data {
        if v.is_infinite() {
            *v = 0.0;
        }
    }
    dist
}

/// The focus measure `F` on a `map_w × map_h` map of a luma plane and its weight `f`
/// (`docs/20` §6 step 2): the absolute 4-neighbour Laplacian (edges clamped) averaged over each
/// cell, divided by its 95th percentile and clamped to 1, then a 3 × 3 box mean (over the cells
/// inside the map); `f = clamp((p95 − p50) / p95, 0, 1)` of the cell means.
pub(crate) fn focus(luma: &Plane, map_w: usize, map_h: usize) -> (Plane, f64) {
    let (w, h) = (luma.width, luma.height);
    let mut cells = Plane::zeros(map_w, map_h);
    if map_w == 0 || map_h == 0 {
        return (cells, 0.0);
    }
    let at = |x: isize, y: isize| {
        luma.at(
            x.clamp(0, w as isize - 1) as usize,
            y.clamp(0, h as isize - 1) as usize,
        )
    };
    let n = (CELL * CELL) as f64;
    for cy in 0..map_h {
        for cx in 0..map_w {
            let mut s = 0.0;
            for y in cy * CELL..(cy + 1) * CELL {
                for x in cx * CELL..(cx + 1) * CELL {
                    let (xi, yi) = (x as isize, y as isize);
                    s += (at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi - 1) + at(xi, yi + 1)
                        - 4.0 * at(xi, yi))
                    .abs();
                }
            }
            cells.data[cy * map_w + cx] = s / n;
        }
    }
    let p95 = percentile(&cells.data, FOCUS_PERCENTILE);
    if p95 <= 0.0 {
        return (Plane::zeros(map_w, map_h), 0.0);
    }
    let p50 = percentile(&cells.data, 0.5);
    let weight = ((p95 - p50) / p95).clamp(0.0, 1.0);
    let scaled: Vec<f64> = cells.data.iter().map(|v| (v / p95).min(1.0)).collect();
    let mut out = Plane::zeros(map_w, map_h);
    for cy in 0..map_h {
        for cx in 0..map_w {
            let (mut s, mut k) = (0.0, 0.0);
            for y in cy.saturating_sub(1)..(cy + 2).min(map_h) {
                for x in cx.saturating_sub(1)..(cx + 2).min(map_w) {
                    s += scaled[y * map_w + x];
                    k += 1.0;
                }
            }
            out.data[cy * map_w + cx] = s / k;
        }
    }
    (out, weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> Plane {
        let mut p = Plane::zeros(w, h);
        for y in 0..h {
            for x in 0..w {
                p.data[y * w + x] = f(x, y);
            }
        }
        p
    }

    #[test]
    fn the_barrier_of_a_blob_is_its_contrast_and_the_border_is_zero() {
        // A bright square on a flat field: its barrier is its contrast, the field's is 0.
        let img = plane(20, 16, |x, y| {
            if (6..12).contains(&x) && (5..10).contains(&y) {
                10.0
            } else {
                1.0
            }
        });
        let d = barrier_distance(&img);
        assert_eq!(d.at(0, 0), 0.0);
        assert_eq!(d.at(19, 7), 0.0);
        assert_eq!(d.at(8, 7), 9.0);
        assert_eq!(d.at(3, 3), 0.0);
    }

    #[test]
    fn the_barrier_goes_round_a_wall_through_its_gap() {
        // A high ring around a low interior: closed, the interior's barrier is the wall's height;
        // with a gap at the lower right (reached only by the backward passes), it is 0.
        let ring = |gap: bool| {
            plane(11, 11, move |x, y| {
                let on_ring = (2..=8).contains(&x)
                    && (2..=8).contains(&y)
                    && (x == 2 || x == 8 || y == 2 || y == 8);
                if on_ring && !(gap && x == 8 && y == 7) {
                    9.0
                } else {
                    1.0
                }
            })
        };
        assert_eq!(barrier_distance(&ring(false)).at(4, 4), 8.0);
        assert_eq!(barrier_distance(&ring(true)).at(4, 4), 0.0);
    }

    #[test]
    fn saliency_is_high_on_a_coloured_blob() {
        let l = plane(32, 24, |_, _| 50.0);
        let a = plane(32, 24, |x, y| {
            if (10..20).contains(&x) && (8..16).contains(&y) {
                60.0
            } else {
                0.0
            }
        });
        let b = plane(32, 24, |_, _| 0.0);
        let m = saliency(&[l, a, b]);
        assert_eq!(m.at(15, 12), 1.0);
        assert_eq!(m.at(2, 2), 0.0);
    }

    #[test]
    fn focus_finds_the_sharp_part() {
        // Fine checks in the left quarter, flat elsewhere: the median cell is flat, so the
        // measure has its full weight.
        let luma = plane(
            64,
            32,
            |x, y| if x < 16 && (x + y) % 2 == 0 { 0.8 } else { 0.2 },
        );
        let (f, weight) = focus(&luma, 16, 8);
        // The cells next to the flat part have a smaller Laplacian, so the box mean is under 1.
        assert!(f.at(2, 4) > 0.8, "{}", f.at(2, 4));
        assert!(f.at(13, 4) < 0.01);
        assert!(weight > 0.99);
    }
}
