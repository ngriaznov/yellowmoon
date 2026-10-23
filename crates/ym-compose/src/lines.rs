//! Straight segments of a picture (`docs/20-auto-crop.md` §5 steps 1 to 3): gradients, line
//! support regions grown after von Gioi, Jakubowicz, Morel and Randall, *LSD: a Fast Line Segment
//! Detector with a False Detection Control* (TPAMI 2010), written from the paper, and the segment
//! each region's moments describe.

use crate::picture::Plane;

/// Standard deviation of the Gaussian blur in pixels.
const BLUR_SIGMA: f64 = 1.0;
/// Taps on each side of the blur's centre (a 7-tap kernel).
const BLUR_RADIUS: usize = 3;
/// Pixels this close to the border get no gradient.
const BORDER: usize = 2;
/// The smallest magnitude threshold.
const MIN_THRESHOLD: f64 = 0.03;
/// The threshold is at least this many times the median magnitude.
const MEDIAN_FACTOR: f64 = 2.0;
/// Magnitude bins for the order of the seeds.
const SEED_BINS: usize = 256;
/// A pixel joins a region when its gradient direction is within this angle of the region's.
const TOLERANCE_DEG: f64 = 22.5;
/// Regions with fewer pixels are dropped.
const MIN_REGION_PX: usize = 12;
/// The shortest segment in pixels.
const MIN_LENGTH_PX: f64 = 20.0;
/// The shortest segment as a fraction of the long side.
const MIN_LENGTH_FRACTION: f64 = 0.04;
/// A segment is at least this many times as long as it is wide.
const MIN_ELONGATION: f64 = 5.0;
/// A segment's region fills at least this share of its length times its width.
const MIN_FILL: f64 = 0.5;
/// The mean magnitude at which a segment's weight is its full length.
const FULL_WEIGHT_MAGNITUDE: f64 = 0.15;

/// One straight segment of the picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Segment {
    /// Midpoint across, in pixels (pixel `i` has its centre at `i + 0.5`).
    pub x: f64,
    /// Midpoint down, in pixels.
    pub y: f64,
    /// Direction in degrees in (−90, 90], y down: a positive angle is turned clockwise from the
    /// x axis.
    pub theta_deg: f64,
    /// Length in pixels: the span along the axis plus 1.
    pub length: f64,
    /// `length × min(1, mean magnitude / 0.15)`.
    pub weight: f64,
}

/// The gradient field: magnitude and unit direction per pixel (0 where there is none).
struct Gradients {
    width: usize,
    height: usize,
    magnitude: Vec<f64>,
    ux: Vec<f64>,
    uy: Vec<f64>,
}

/// The straight segments of a luma plane, in the order their regions were grown.
pub(crate) fn segments(luma: &Plane) -> Vec<Segment> {
    let g = gradients(&blur(luma));
    let long_side = luma.width.max(luma.height) as f64;
    let min_length = MIN_LENGTH_PX.max(MIN_LENGTH_FRACTION * long_side);
    regions(&g)
        .iter()
        .filter_map(|r| segment_of(r, &g, min_length))
        .collect()
}

/// The normalised 7-tap Gaussian kernel.
fn kernel() -> [f64; 2 * BLUR_RADIUS + 1] {
    let mut k = [0.0; 2 * BLUR_RADIUS + 1];
    for (i, v) in k.iter_mut().enumerate() {
        let d = i as f64 - BLUR_RADIUS as f64;
        *v = (-d * d / (2.0 * BLUR_SIGMA * BLUR_SIGMA)).exp();
    }
    let sum: f64 = k.iter().sum();
    k.map(|v| v / sum)
}

/// The Gaussian blur, separable, edges clamped (`docs/20` §5 step 1).
fn blur(src: &Plane) -> Plane {
    let (w, h) = (src.width, src.height);
    let k = kernel();
    let r = BLUR_RADIUS as isize;
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let mut across = Plane::zeros(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (t, kv) in k.iter().enumerate() {
                s += kv * src.at(clamp(x as isize + t as isize - r, w), y);
            }
            across.data[y * w + x] = s;
        }
    }
    let mut out = Plane::zeros(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (t, kv) in k.iter().enumerate() {
                s += kv * across.at(x, clamp(y as isize + t as isize - r, h));
            }
            out.data[y * w + x] = s;
        }
    }
    out
}

/// Sobel 3 × 3 divided by 8, so the magnitude is the luma change per pixel; pixels within
/// [`BORDER`] of the border get none.
fn gradients(b: &Plane) -> Gradients {
    let (w, h) = (b.width, b.height);
    let mut g = Gradients {
        width: w,
        height: h,
        magnitude: vec![0.0; w * h],
        ux: vec![0.0; w * h],
        uy: vec![0.0; w * h],
    };
    if w < 2 * BORDER + 1 || h < 2 * BORDER + 1 {
        return g;
    }
    for y in BORDER..h - BORDER {
        for x in BORDER..w - BORDER {
            let gx = (b.at(x + 1, y - 1) + 2.0 * b.at(x + 1, y) + b.at(x + 1, y + 1)
                - b.at(x - 1, y - 1)
                - 2.0 * b.at(x - 1, y)
                - b.at(x - 1, y + 1))
                / 8.0;
            let gy = (b.at(x - 1, y + 1) + 2.0 * b.at(x, y + 1) + b.at(x + 1, y + 1)
                - b.at(x - 1, y - 1)
                - 2.0 * b.at(x, y - 1)
                - b.at(x + 1, y - 1))
                / 8.0;
            let m = gx.hypot(gy);
            let i = y * w + x;
            g.magnitude[i] = m;
            if m > 0.0 {
                g.ux[i] = gx / m;
                g.uy[i] = gy / m;
            }
        }
    }
    g
}

/// The line support regions (`docs/20` §5 step 2) as lists of pixel indices, in the order they
/// were grown; regions under [`MIN_REGION_PX`] are left out.
fn regions(g: &Gradients) -> Vec<Vec<usize>> {
    let (w, h) = (g.width, g.height);
    if w < 2 * BORDER + 1 || h < 2 * BORDER + 1 {
        return Vec::new();
    }
    // The median over the pixels that have a gradient.
    let mut interior = Vec::with_capacity((w - 2 * BORDER) * (h - 2 * BORDER));
    for y in BORDER..h - BORDER {
        interior.extend_from_slice(&g.magnitude[y * w + BORDER..y * w + w - BORDER]);
    }
    let threshold = MIN_THRESHOLD.max(MEDIAN_FACTOR * crate::picture::percentile(&interior, 0.5));
    let max = g.magnitude.iter().copied().fold(0.0, f64::max);
    if max < threshold {
        return Vec::new();
    }
    // Seeds by falling magnitude: 256 bins from the largest down, raster order within a bin
    // (a counting sort, stable in the raster order).
    let bin_of = |m: f64| ((m / max * SEED_BINS as f64) as usize).min(SEED_BINS - 1);
    let mut counts = [0usize; SEED_BINS];
    for &m in &g.magnitude {
        if m >= threshold {
            counts[bin_of(m)] += 1;
        }
    }
    let mut start = [0usize; SEED_BINS];
    let mut acc = 0;
    for b in (0..SEED_BINS).rev() {
        start[b] = acc;
        acc += counts[b];
    }
    let mut seeds = vec![0usize; acc];
    for (i, &m) in g.magnitude.iter().enumerate() {
        if m >= threshold {
            let b = bin_of(m);
            seeds[start[b]] = i;
            start[b] += 1;
        }
    }

    let cos_tol = TOLERANCE_DEG.to_radians().cos();
    let mut used = vec![false; w * h];
    let mut out = Vec::new();
    for &seed in &seeds {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let mut region = vec![seed];
        let (mut sx, mut sy) = (g.ux[seed], g.uy[seed]);
        let mut next = 0;
        while next < region.len() {
            let p = region[next];
            next += 1;
            let (px, py) = ((p % w) as isize, (p / w) as isize);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (px + dx, py + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let q = ny as usize * w + nx as usize;
                    if used[q] || g.magnitude[q] < threshold {
                        continue;
                    }
                    // Within the tolerance of the region's mean direction (the sum of its unit
                    // vectors, compared without normalising it).
                    let norm = sx.hypot(sy);
                    if g.ux[q] * sx + g.uy[q] * sy >= cos_tol * norm {
                        used[q] = true;
                        region.push(q);
                        sx += g.ux[q];
                        sy += g.uy[q];
                    }
                }
            }
        }
        if region.len() >= MIN_REGION_PX {
            out.push(region);
        }
    }
    out
}

/// The segment a region describes, if it is one (`docs/20` §5 step 3).
fn segment_of(region: &[usize], g: &Gradients, min_length: f64) -> Option<Segment> {
    let w = g.width;
    let pos = |i: usize| ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    let (mut m0, mut mx, mut my) = (0.0, 0.0, 0.0);
    for &i in region {
        let (x, y) = pos(i);
        let m = g.magnitude[i];
        m0 += m;
        mx += m * x;
        my += m * y;
    }
    if m0 <= 0.0 {
        return None;
    }
    let (cx, cy) = (mx / m0, my / m0);
    let (mut ixx, mut iyy, mut ixy) = (0.0, 0.0, 0.0);
    for &i in region {
        let (x, y) = pos(i);
        let m = g.magnitude[i];
        let (dx, dy) = (x - cx, y - cy);
        ixx += m * dx * dx;
        iyy += m * dy * dy;
        ixy += m * dx * dy;
    }
    // The major axis of the second moments, folded into (−90°, 90°].
    let mut theta = 0.5 * (2.0 * ixy).atan2(ixx - iyy);
    if theta <= -std::f64::consts::FRAC_PI_2 {
        theta += std::f64::consts::PI;
    }
    let (s, c) = theta.sin_cos();
    let (mut t_min, mut t_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut n_min, mut n_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &i in region {
        let (x, y) = pos(i);
        let (dx, dy) = (x - cx, y - cy);
        let t = dx * c + dy * s;
        let n = -dx * s + dy * c;
        t_min = t_min.min(t);
        t_max = t_max.max(t);
        n_min = n_min.min(n);
        n_max = n_max.max(n);
    }
    let length = t_max - t_min + 1.0;
    let width = n_max - n_min + 1.0;
    if length < min_length
        || length < MIN_ELONGATION * width
        || (region.len() as f64) < MIN_FILL * length * width
    {
        return None;
    }
    let mean_magnitude = m0 / region.len() as f64;
    let mid = 0.5 * (t_min + t_max);
    Some(Segment {
        x: cx + mid * c,
        y: cy + mid * s,
        theta_deg: theta.to_degrees(),
        length,
        weight: length * (mean_magnitude / FULL_WEIGHT_MAGNITUDE).min(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A luma plane with a straight dark-to-light edge through the centre at `theta_deg`
    /// (y down), anti-aliased by 8 × 8 samples per pixel.
    fn edge(w: usize, h: usize, theta_deg: f64, contrast: f64) -> Plane {
        let (s, c) = theta_deg.to_radians().sin_cos();
        let mut p = Plane::zeros(w, h);
        let n = 8;
        for y in 0..h {
            for x in 0..w {
                let mut v = 0.0;
                for sy in 0..n {
                    for sx in 0..n {
                        let px = x as f64 + (sx as f64 + 0.5) / n as f64 - w as f64 / 2.0;
                        let py = y as f64 + (sy as f64 + 0.5) / n as f64 - h as f64 / 2.0;
                        // Signed distance across the line.
                        let d = -px * s + py * c;
                        v += if d > 0.0 {
                            0.5 + contrast / 2.0
                        } else {
                            0.5 - contrast / 2.0
                        };
                    }
                }
                p.data[y * w + x] = v / (n * n) as f64;
            }
        }
        p
    }

    #[test]
    fn one_edge_gives_one_segment_at_its_angle() {
        for theta in [0.0, 3.0, -7.5, 88.0, -89.0, 90.0, 45.0] {
            let segs = segments(&edge(300, 200, theta, 0.6));
            assert_eq!(segs.len(), 1, "theta {theta}: {segs:?}");
            let s = segs[0];
            let mut d = s.theta_deg - theta;
            if d > 90.0 {
                d -= 180.0;
            }
            if d < -90.0 {
                d += 180.0;
            }
            assert!(d.abs() < 0.05, "theta {theta}: found {}", s.theta_deg);
            assert!(s.theta_deg > -90.0 && s.theta_deg <= 90.0);
            // The midpoint lies on the line through the centre.
            let (sn, cs) = theta.to_radians().sin_cos();
            let dist = -(s.x - 150.0) * sn + (s.y - 100.0) * cs;
            assert!(dist.abs() < 0.6, "theta {theta}: midpoint off by {dist}");
            assert!(s.weight > 0.0 && s.weight <= s.length);
        }
    }

    #[test]
    fn a_flat_or_faint_picture_has_no_segments() {
        assert!(segments(&Plane::zeros(100, 80)).is_empty());
        assert!(segments(&edge(200, 120, 5.0, 0.02)).is_empty());
        assert!(segments(&Plane::zeros(3, 3)).is_empty());
    }
}
