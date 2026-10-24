//! Bright balls: out-of-focus lights, found and measured by their size (`docs/21` section 4, L1 step 2, and
//! section 7, step 4).
//!
//! A defocused light is a disc with a sharp rim, so the edge measure of the blur map reads it as sharp. The
//! balls are therefore found directly: a scale-space blob search (differences of the ladder's Gaussians, in the
//! square-root domain of the ladder's levels) proposes centres and sizes; each candidate is measured on the linear plane by the area of
//! its half-level contour; and it must be a flat-topped, round, compact, bright disc, at least 1 EV above the
//! median of the ring around it, at least twice the smallest reshaped radius, and consistent with the blur the
//! edges around it say (when they say anything). Balls are kept brightest first, at most 256; of two closer than
//! 1.2 radii the dimmer is dropped. A scene with fewer than three balls and none confirmed by the edges around it
//! is treated as having none: a lone round light (the moon, a lamp in focus) is more likely sharp than blurred.

use crate::consts::MAX_BALLS;
use crate::error::LensError;

use super::disc::{LEVELS, level_sigma};

/// A ball on the detector's working grid, in pixels of that grid.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BallPx {
    /// Centre, continuous pixel coordinates (pixel `i` covers `[i, i + 1)`).
    pub cx: f64,
    pub cy: f64,
    /// Radius of the half-level contour.
    pub r: f64,
    /// Radius of the disc that is taken out (1.15 `r` plus a margin).
    pub r_out: f64,
    /// Per channel, the light above the ring's plane fit, summed over the disc.
    pub energy: [f64; 3],
    /// Per channel, the ring's plane fit `a + b dx + c dy` (`dx`, `dy` in pixels from the centre).
    pub fit: [[f64; 3]; 3],
    /// The 90th percentile of the disc's luminance.
    pub peak: f64,
    /// True when the edges around the ball agree with its size.
    pub supported: bool,
}

/// Everything the search reads.
pub(super) struct BallInput<'a> {
    pub w: usize,
    pub h: usize,
    /// The working grid's red, green and blue planes (linear).
    pub rgb: [&'a [f32]; 3],
    /// Its luminance (linear).
    pub y: &'a [f32],
    /// The scale ladder of the linear luminance.
    pub ladder: &'a [Vec<f32>],
    /// Noise of the square-root luminance, per pixel.
    pub noise_v: f64,
    /// Half-diagonal of the working grid in pixels.
    pub hd: f64,
    /// Smallest disc radius the map reshapes, in pixels (`b_min`, or the floor of the native blur).
    pub r_min_px: f64,
    /// Largest ball radius, in pixels.
    pub r_max_px: f64,
    /// The edge samples' weight and weight times log radius at every pixel (for the consistency test).
    pub ev_w: &'a [f32],
    pub ev_s: &'a [f32],
}

/// Margin of the take-out disc beyond 1.15 `r`, in working-grid pixels.
pub(super) const TAKE_OUT_MARGIN_PX: f64 = 1.5;
/// A ball is at least this much brighter than the median of its ring (1 EV).
const MIN_RATIO: f64 = 2.0;
/// A flat top: this fraction of the disc is within 0.5 EV of the peak.
const FLAT_FRACTION: f64 = 0.6;
/// The most elongated a ball may be (ratio of the contour's principal axes; cat's-eye balls are lemons).
const MAX_ELONGATION: f64 = 1.4;
/// Candidate cap.
const MAX_CANDIDATES: usize = 1200;
/// Largest ratio between a ball's radius and the radius the edges around it give.
const CONSISTENT: f64 = 1.6;
/// Edge evidence (sum of weights in the ring around a ball) needed to hold a ball to the edges' radius.
const EVIDENCE_MIN: f64 = 1.5;
/// A scene needs this many balls, or one confirmed by the edges.
const MIN_CLUSTER: usize = 3;

#[derive(Clone, Copy, Debug)]
struct Candidate {
    score: f32,
    x: usize,
    y: usize,
    r0: f64,
}

#[derive(Clone, Debug)]
struct Refined {
    cx: f64,
    cy: f64,
    r: f64,
    peak: f64,
}

fn median(values: &mut [f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, v, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    f64::from(*v)
}

/// Visits every pixel whose centre lies within `[r_lo, r_hi]` of `(cx, cy)`, as `(x, y, dx, dy, d)`.
fn for_each_in_annulus(
    w: usize,
    h: usize,
    cx: f64,
    cy: f64,
    r_lo: f64,
    r_hi: f64,
    mut f: impl FnMut(usize, usize, f64, f64, f64),
) {
    let x0 = (cx - r_hi).floor().max(0.0) as usize;
    let y0 = (cy - r_hi).floor().max(0.0) as usize;
    let x1 = ((cx + r_hi).ceil() as usize + 1).min(w);
    let y1 = ((cy + r_hi).ceil() as usize + 1).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d >= r_lo && d <= r_hi {
                f(x, y, dx, dy, d);
            }
        }
    }
}

/// The scale-space maxima of the differences of the ladder's Gaussians, brightest response first.
fn candidates(inp: &BallInput) -> Vec<Candidate> {
    let (w, h) = (inp.w, inp.h);
    let first = 2usize;
    let last = LEVELS - 2; // DoG of levels m and m + 1 for m = 2 ..= 9
    let r_lo = 0.7 * 2.0 * inp.r_min_px;
    let r_hi = 1.5 * inp.r_max_px;
    let mut dogs: Vec<(usize, f64, Vec<f32>)> = Vec::new();
    for m in first..=last {
        let (s0, s1) = (level_sigma(m), level_sigma(m + 1));
        let r0 = std::f64::consts::SQRT_2 * (s0 * s1).sqrt();
        let d: Vec<f32> = inp.ladder[m + 1]
            .iter()
            .zip(&inp.ladder[m])
            .map(|(a, b)| a.max(0.0).sqrt() - b.max(0.0).sqrt())
            .collect();
        dogs.push((m, r0, d));
    }
    let noise = 3.0 * inp.noise_v.max(0.002);
    let mut out = Vec::new();
    for (mi, (m, r0, d)) in dogs.iter().enumerate() {
        if *r0 < r_lo || *r0 > r_hi {
            continue;
        }
        // The response of a bright disc is positive: the second Gaussian is darker than the first.
        // `ladder[m + 1] - ladder[m]` is negative at a bright centre, so the response is its negation.
        let theta = (0.008f64).max(5.0 * 0.115 * noise / level_sigma(*m)) as f32;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let v = -d[y * w + x];
                if v <= theta {
                    continue;
                }
                let mut is_max = true;
                'nb: for (nj, (_, _, other)) in dogs.iter().enumerate() {
                    if nj + 1 < mi || nj > mi + 1 {
                        continue;
                    }
                    for dy in 0..3usize {
                        for dx in 0..3usize {
                            let (nx, ny) = (x + dx - 1, y + dy - 1);
                            if nj == mi && dx == 1 && dy == 1 {
                                continue;
                            }
                            let u = -other[ny * w + nx];
                            // Ties go to the neighbour that comes first in (level, row, column) order.
                            let before = (nj, ny, nx) < (mi, y, x);
                            if u > v || (before && u == v) {
                                is_max = false;
                                break 'nb;
                            }
                        }
                    }
                }
                if is_max {
                    out.push(Candidate {
                        score: v,
                        x,
                        y,
                        r0: *r0,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });
    out.truncate(MAX_CANDIDATES);
    out
}

/// Measures one candidate on the linear luminance, or rejects it.
fn refine(inp: &BallInput, c: &Candidate) -> Option<Refined> {
    let (w, h) = (inp.w, inp.h);
    let y = inp.y;
    let sigma_y = |level: f64| 2.0 * level.max(1e-6).sqrt() * inp.noise_v.max(0.002) * 3.0;
    let mut cx = c.x as f64 + 0.5;
    let mut cy = c.y as f64 + 0.5;
    let mut r = c.r0;
    let mut result: Option<Refined> = None;
    for _ in 0..2 {
        // The ring's median and the plateau's median.
        let mut ring = Vec::new();
        for_each_in_annulus(w, h, cx, cy, 1.5 * r, 2.3 * r, |x, yy, _, _, _| {
            ring.push(y[yy * w + x]);
        });
        let mut top = Vec::new();
        for_each_in_annulus(w, h, cx, cy, 0.0, 0.6 * r, |x, yy, _, _, _| {
            top.push(y[yy * w + x]);
        });
        if ring.len() < 8 || top.len() < 4 {
            return None;
        }
        let y_ring = median(&mut ring);
        let y_top = median(&mut top);
        // At least 1 EV above the ring, and by more than the noise.
        if y_top < MIN_RATIO * y_ring || y_top - y_ring < sigma_y(y_top) {
            return None;
        }
        let span = y_top - y_ring;
        let alpha = |x: usize, yy: usize| -> f64 {
            ((f64::from(y[yy * w + x]) - y_ring) / span).clamp(0.0, 1.0)
        };
        // Flood fill (4-connected) of the pixels with alpha above 0.08 from the centre, inside a window.
        let win = (2.0 * r).ceil() as i64 + 1;
        let (ix, iy) = (cx.floor() as i64, cy.floor() as i64);
        if ix < 0 || iy < 0 || ix >= w as i64 || iy >= h as i64 {
            return None;
        }
        if alpha(ix as usize, iy as usize) < 0.5 {
            return None;
        }
        let side = (2 * win + 1) as usize;
        let mut seen = vec![false; side * side];
        let mut stack = vec![(ix, iy)];
        seen[(win * side as i64 + win) as usize] = true;
        let (mut area, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
        let mut points: Vec<(f64, f64, f64)> = Vec::new();
        while let Some((px, py)) = stack.pop() {
            let a = alpha(px as usize, py as usize);
            points.push((px as f64 + 0.5, py as f64 + 0.5, a));
            area += a;
            mx += a * (px as f64 + 0.5);
            my += a * (py as f64 + 0.5);
            for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (px + dx, py + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let (lx, ly) = (nx - ix + win, ny - iy + win);
                if lx < 0 || ly < 0 || lx >= side as i64 || ly >= side as i64 {
                    // The blob leaves the window: it is not a compact disc of this size.
                    if alpha(nx as usize, ny as usize) > 0.08 {
                        return None;
                    }
                    continue;
                }
                let li = ly as usize * side + lx as usize;
                if seen[li] {
                    continue;
                }
                seen[li] = true;
                if alpha(nx as usize, ny as usize) > 0.08 {
                    stack.push((nx, ny));
                }
            }
        }
        if area < 4.0 {
            return None;
        }
        cx = mx / area;
        cy = my / area;
        for &(px, py, a) in &points {
            let (dx, dy) = (px - cx, py - cy);
            sxx += a * dx * dx;
            sxy += a * dx * dy;
            syy += a * dy * dy;
        }
        let (sxx, sxy, syy) = (sxx / area, sxy / area, syy / area);
        let tr = sxx + syy;
        let det = sxx * syy - sxy * sxy;
        let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
        let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(1e-12));
        let elong = (l1 / l2).sqrt();
        r = (area / std::f64::consts::PI).sqrt();
        if elong > MAX_ELONGATION {
            return None;
        }
        // Flat-topped: most of the disc is within 0.5 EV of its 90th percentile.
        let mut inside = Vec::new();
        for_each_in_annulus(w, h, cx, cy, 0.0, 0.9 * r, |x, yy, _, _, _| {
            inside.push(y[yy * w + x]);
        });
        if inside.len() < 4 {
            return None;
        }
        let mut sorted = inside.clone();
        let p90 = {
            let idx = ((sorted.len() - 1) as f64 * 0.9).round() as usize;
            let (_, v, _) = sorted.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
            f64::from(*v)
        };
        let floor = p90 * std::f64::consts::FRAC_1_SQRT_2;
        let flat = inside.iter().filter(|v| f64::from(**v) >= floor).count() as f64
            / inside.len() as f64;
        if flat < FLAT_FRACTION {
            return None;
        }
        result = Some(Refined {
            cx,
            cy,
            r,
            peak: p90,
        });
    }
    let rf = result?;
    // The size the scale-space gave and the size measured must agree.
    let ratio = rf.r / c.r0;
    if !(1.0 / CONSISTENT..=CONSISTENT).contains(&ratio) {
        return None;
    }
    if rf.r < 2.0 * inp.r_min_px || rf.r > inp.r_max_px {
        return None;
    }
    Some(rf)
}

/// Solves the 3 x 3 system `m x = v` by Gaussian elimination with partial pivoting; `None` when singular.
fn solve3(mut m: [[f64; 3]; 3], mut v: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        v.swap(col, piv);
        for row in col + 1..3 {
            let f = m[row][col] / m[col][col];
            for k in col..3 {
                m[row][k] -= f * m[col][k];
            }
            v[row] -= f * v[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut s = v[row];
        for k in row + 1..3 {
            s -= m[row][k] * x[k];
        }
        x[row] = s / m[row][row];
    }
    Some(x)
}

/// Least-squares plane `a + b dx + c dy` through `samples = (dx, dy, value)`.
fn fit_plane(samples: &[(f64, f64, f64)]) -> Option<[f64; 3]> {
    let mut m = [[0.0f64; 3]; 3];
    let mut v = [0.0f64; 3];
    for &(dx, dy, z) in samples {
        let row = [1.0, dx, dy];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += row[i] * row[j];
            }
            v[i] += row[i] * z;
        }
    }
    solve3(m, v)
}

/// The ring's plane fit per channel (robust: samples far from the first luminance fit are dropped).
fn ring_fit(inp: &BallInput, ball: &Refined, r_out: f64, owner: &[i32]) -> [[f64; 3]; 3] {
    let (w, h) = (inp.w, inp.h);
    let mut samples: Vec<(usize, f64, f64)> = Vec::new(); // (pixel, dx, dy)
    for_each_in_annulus(
        w,
        h,
        ball.cx,
        ball.cy,
        r_out + 1.0,
        r_out + 0.6 * ball.r + 2.0,
        |x, y, dx, dy, _| {
            if owner[y * w + x] < 0 {
                samples.push((y * w + x, dx, dy));
            }
        },
    );
    let channel = |c: usize, keep: &[bool]| -> Option<[f64; 3]> {
        let pts: Vec<(f64, f64, f64)> = samples
            .iter()
            .zip(keep)
            .filter(|(_, k)| **k)
            .map(|(s, _)| (s.1, s.2, f64::from(inp.rgb[c][s.0])))
            .collect();
        if pts.len() < 12 {
            return None;
        }
        fit_plane(&pts)
    };
    let mut keep = vec![true; samples.len()];
    for _ in 0..2 {
        let lum: Vec<(f64, f64, f64)> = samples
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(s, _)| (s.1, s.2, f64::from(inp.y[s.0])))
            .collect();
        if lum.len() < 12 {
            break;
        }
        let Some(p) = fit_plane(&lum) else { break };
        let mut res: Vec<f32> = samples
            .iter()
            .map(|s| (f64::from(inp.y[s.0]) - (p[0] + p[1] * s.1 + p[2] * s.2)).abs() as f32)
            .collect();
        let mad = median(&mut res);
        let limit = 3.0 * 1.4826 * mad + 1e-6;
        for (i, s) in samples.iter().enumerate() {
            let r = (f64::from(inp.y[s.0]) - (p[0] + p[1] * s.1 + p[2] * s.2)).abs();
            keep[i] = r <= limit;
        }
    }
    let mut out = [[0.0f64; 3]; 3];
    for (c, o) in out.iter_mut().enumerate() {
        *o = channel(c, &keep).unwrap_or_else(|| {
            // Too few ring samples: a flat level, the median of what there is.
            let mut vals: Vec<f32> = samples.iter().map(|s| inp.rgb[c][s.0]).collect();
            [median(&mut vals), 0.0, 0.0]
        });
    }
    out
}

/// Finds the balls of a plane.
pub(super) fn find_balls(
    inp: &BallInput,
    cancel: &dyn Fn() -> bool,
) -> Result<Vec<BallPx>, LensError> {
    let (w, h) = (inp.w, inp.h);
    let cands = candidates(inp);
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // Phase A: the shape tests.
    let mut refined: Vec<Refined> = cands.iter().filter_map(|c| refine(inp, c)).collect();
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // Brightest first, ties by position.
    refined.sort_by(|a, b| {
        b.peak
            .total_cmp(&a.peak)
            .then(a.cy.total_cmp(&b.cy))
            .then(a.cx.total_cmp(&b.cx))
    });
    // The footprints of every candidate that passed the shape tests (the edge evidence skips them: their
    // rims are sharp edges that say nothing about the blur around).
    let mut claim = vec![false; w * h];
    for b in &refined {
        let r_out = 1.15 * b.r + TAKE_OUT_MARGIN_PX;
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, 1.4 * r_out, |x, y, _, _, _| {
            claim[y * w + x] = true;
        });
    }
    // Phase B: overlap and consistency, brightest first.
    let mut accepted: Vec<(Refined, bool)> = Vec::new();
    'cand: for b in &refined {
        for (a, _) in &accepted {
            let d = ((a.cx - b.cx).powi(2) + (a.cy - b.cy).powi(2)).sqrt();
            if d < 1.2 * a.r.max(b.r) {
                continue 'cand;
            }
        }
        let (mut ew, mut es) = (0.0f64, 0.0f64);
        for_each_in_annulus(w, h, b.cx, b.cy, 1.5 * b.r, 3.0 * b.r, |x, y, _, _, _| {
            let i = y * w + x;
            if !claim[i] {
                ew += f64::from(inp.ev_w[i]);
                es += f64::from(inp.ev_s[i]);
            }
        });
        let supported = ew >= EVIDENCE_MIN;
        if supported {
            let mu = es / ew;
            if (libm::log(b.r) - mu).abs() > libm::log(CONSISTENT) {
                continue;
            }
        }
        accepted.push((b.clone(), supported));
        if accepted.len() >= MAX_BALLS {
            break;
        }
    }
    // A scene needs a cluster of balls, or balls the edges confirm.
    if accepted.len() < MIN_CLUSTER {
        accepted.retain(|(_, s)| *s);
    }
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // Ownership of the discs (the brightest ball owns a pixel two discs share).
    let mut owner = vec![-1i32; w * h];
    for (i, (b, _)) in accepted.iter().enumerate() {
        let r_out = 1.15 * b.r + TAKE_OUT_MARGIN_PX;
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, r_out, |x, y, _, _, _| {
            let o = &mut owner[y * w + x];
            if *o < 0 {
                *o = i as i32;
            }
        });
    }
    let mut out = Vec::with_capacity(accepted.len());
    for (i, (b, supported)) in accepted.iter().enumerate() {
        let r_out = 1.15 * b.r + TAKE_OUT_MARGIN_PX;
        let fit = ring_fit(inp, b, r_out, &owner);
        let mut energy = [0.0f64; 3];
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, r_out, |x, y, dx, dy, _| {
            if owner[y * w + x] == i as i32 {
                for c in 0..3 {
                    let base = fit[c][0] + fit[c][1] * dx + fit[c][2] * dy;
                    energy[c] += f64::from(inp.rgb[c][y * w + x]) - base;
                }
            }
        });
        for e in &mut energy {
            *e = e.max(0.0);
        }
        out.push(BallPx {
            cx: b.cx,
            cy: b.cy,
            r: b.r,
            r_out,
            energy,
            fit,
            peak: b.peak,
            supported: *supported,
        });
    }
    Ok(out)
}
