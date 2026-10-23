//! The straightener (`docs/20-auto-crop.md` §5): the camera's roll from the verticals and the
//! horizon among the picture's straight segments.

use serde::Serialize;

use crate::Picture;
use crate::lines::{self, Segment};
use crate::picture;

/// Segments within this angle of vertical or horizontal take part.
const NEAR_DEG: f64 = 12.0;
/// Tukey's biweight constant in degrees.
const TUKEY_C_DEG: f64 = 1.5;
/// Iterations of the biweight.
const BIWEIGHT_ITERATIONS: usize = 8;
/// The lean `k` is held at 0 when the weighted standard deviation of `x / W` is below this.
const MIN_SPREAD: f64 = 0.1;
/// The verticals count with at least this many segments ...
const MIN_VERTICALS: usize = 3;
/// ... of at least this total length over the picture's height ...
const MIN_VERTICAL_LENGTH: f64 = 0.6;
/// ... when the inliers carry at least this share of the weight ...
const MIN_INLIER_SHARE: f64 = 0.75;
/// ... and their weighted RMS residual is at most this, in degrees.
const MAX_RMS_DEG: f64 = 1.0;
/// Bin width of the horizontals' histogram in degrees.
const BIN_DEG: f64 = 0.2;
/// Horizontals within this angle of the histogram's peak are grouped.
const PEAK_WINDOW_DEG: f64 = 0.6;
/// Height of a cluster of collinear horizontals over the picture's height.
const CLUSTER_HEIGHT: f64 = 0.02;
/// The horizon counts when its cluster's total length is at least this over the picture's width.
const MIN_HORIZON_LENGTH: f64 = 0.35;
/// The verticals and the horizon agree within this, in degrees.
const AGREEMENT_DEG: f64 = 1.0;
/// A roll at most this large is level, in degrees.
const LEVEL_DEG: f64 = 0.2;
/// A roll larger than this is taken as meant, in degrees.
const MAX_TILT_DEG: f64 = 10.0;

/// The straightener's answer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Straighten {
    /// Within 0.2° of level.
    Level,
    /// Turn the photo by `angle_deg` (clockwise positive) to level it.
    Rotate {
        /// The straighten angle in degrees.
        angle_deg: f64,
    },
    /// No answer: no horizon or verticals, they disagree, or the tilt is beyond 10°.
    Unsure,
}

/// One estimate of the roll: `tau_deg` clockwise positive, and the length of the segments that
/// carry it (the weight of the estimate in the decision).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Estimate {
    tau_deg: f64,
    length: f64,
}

/// The straightener's answer for a picture (`docs/20` §5), and where the horizon it found crosses
/// the picture's centre column, as a fraction of the picture's height from the top.
pub(crate) fn straighten(picture: &Picture<'_>) -> (Straighten, Option<f64>) {
    let segments = lines::segments(&picture::luma(picture));
    let (w, h) = (picture.width() as f64, picture.height() as f64);
    let horizon = horizon(&segments, w, h);
    (
        decide(verticals(&segments, w, h), horizon.map(|(e, _)| e)),
        horizon.map(|(_, at)| at / h),
    )
}

/// The weighted median of `(value, weight)` pairs: the first value, in ascending order (ties in
/// the given order), at which the cumulative weight reaches half the total.
fn weighted_median(pairs: &[(f64, f64)]) -> f64 {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = sorted.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for &(v, w) in &sorted {
        acc += w;
        if acc >= 0.5 * total {
            return v;
        }
    }
    sorted.last().map_or(0.0, |p| p.0)
}

/// `τ_v` from the near-vertical segments (`docs/20` §5 step 4).
fn verticals(segments: &[Segment], w: f64, h: f64) -> Option<Estimate> {
    // (deviation δ, lateral position u = (x − W/2) / W, weight, length).
    let near: Vec<(f64, f64, f64, f64)> = segments
        .iter()
        .filter(|s| s.theta_deg.abs() >= 90.0 - NEAR_DEG)
        .map(|s| {
            let d = if s.theta_deg > 0.0 {
                s.theta_deg - 90.0
            } else {
                s.theta_deg + 90.0
            };
            (d, (s.x - 0.5 * w) / w, s.weight, s.length)
        })
        .collect();
    let total_length: f64 = near.iter().map(|s| s.3).sum();
    let total_weight: f64 = near.iter().map(|s| s.2).sum();
    if near.len() < MIN_VERTICALS || total_length < MIN_VERTICAL_LENGTH * h || total_weight <= 0.0 {
        return None;
    }
    let mean_u = near.iter().map(|s| s.2 * s.1).sum::<f64>() / total_weight;
    let var_u = near
        .iter()
        .map(|s| s.2 * (s.1 - mean_u) * (s.1 - mean_u))
        .sum::<f64>()
        / total_weight;
    let hold_k = var_u.sqrt() < MIN_SPREAD;

    let pairs: Vec<(f64, f64)> = near.iter().map(|s| (s.0, s.2)).collect();
    let mut tau = weighted_median(&pairs);
    let mut k = 0.0;
    for _ in 0..BIWEIGHT_ITERATIONS {
        let (mut s0, mut s1, mut s2, mut t0, mut t1) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(d, u, wt, _) in &near {
            let r = d - tau - k * u;
            if r.abs() >= TUKEY_C_DEG {
                continue;
            }
            let b = 1.0 - (r / TUKEY_C_DEG) * (r / TUKEY_C_DEG);
            let wb = wt * b * b;
            s0 += wb;
            s1 += wb * u;
            s2 += wb * u * u;
            t0 += wb * d;
            t1 += wb * u * d;
        }
        if s0 <= 0.0 {
            return None;
        }
        let det = s0 * s2 - s1 * s1;
        if hold_k {
            tau = t0 / s0;
        } else if det > 1e-12 * s0 * s0 {
            tau = (s2 * t0 - s1 * t1) / det;
            k = (s0 * t1 - s1 * t0) / det;
        } else {
            tau = (t0 - k * s1) / s0;
        }
    }

    let (mut in_weight, mut in_sq, mut in_length) = (0.0, 0.0, 0.0);
    for &(d, u, wt, len) in &near {
        let r = d - tau - k * u;
        if r.abs() <= TUKEY_C_DEG {
            in_weight += wt;
            in_sq += wt * r * r;
            in_length += len;
        }
    }
    if in_weight < MIN_INLIER_SHARE * total_weight || (in_sq / in_weight).sqrt() > MAX_RMS_DEG {
        return None;
    }
    Some(Estimate {
        tau_deg: tau,
        length: in_length,
    })
}

/// `τ_h` from the collinear near-horizontal segments (`docs/20` §5 step 5), and the height in pixels
/// at which the cluster crosses the centre column (its crossings' mean, weighted as the segments).
fn horizon(segments: &[Segment], w: f64, h: f64) -> Option<(Estimate, f64)> {
    let near: Vec<&Segment> = segments
        .iter()
        .filter(|s| s.theta_deg.abs() <= NEAR_DEG)
        .collect();
    if near.is_empty() {
        return None;
    }
    // The weight histogram over ±12° in bins of 0.2°, smoothed by [1, 2, 3, 2, 1] / 9.
    let bins = (2.0 * NEAR_DEG / BIN_DEG).round() as usize;
    let mut hist = vec![0.0; bins];
    for s in &near {
        let b = ((s.theta_deg + NEAR_DEG) / BIN_DEG).floor();
        hist[(b.max(0.0) as usize).min(bins - 1)] += s.weight;
    }
    const SMOOTH: [f64; 5] = [1.0, 2.0, 3.0, 2.0, 1.0];
    let mut peak = 0;
    let mut peak_value = f64::NEG_INFINITY;
    for b in 0..bins {
        let mut v = 0.0;
        for (t, kv) in SMOOTH.iter().enumerate() {
            let j = b as isize + t as isize - 2;
            if j >= 0 && (j as usize) < bins {
                v += kv * hist[j as usize];
            }
        }
        v /= 9.0;
        // The first bin wins a tie.
        if v > peak_value {
            peak_value = v;
            peak = b;
        }
    }
    let peak_deg = -NEAR_DEG + BIN_DEG * (peak as f64 + 0.5);

    // Where each chosen segment's line crosses the centre column, sorted (ties in segment order).
    let mut crossings: Vec<(f64, &Segment)> = near
        .iter()
        .filter(|s| (s.theta_deg - peak_deg).abs() <= PEAK_WINDOW_DEG)
        .map(|s| (s.y + (0.5 * w - s.x) * s.theta_deg.to_radians().tan(), *s))
        .collect();
    crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Greedy clusters 2 % of H tall; the heaviest wins (the first on a tie).
    let mut best: Option<(f64, f64, f64, f64)> = None; // (weight, Σ weight δ, Σ length, Σ weight y)
    let mut i = 0;
    while i < crossings.len() {
        let top = crossings[i].0;
        let (mut wt, mut wd, mut len, mut wy) = (0.0, 0.0, 0.0, 0.0);
        while i < crossings.len() && crossings[i].0 - top <= CLUSTER_HEIGHT * h {
            let (y, s) = crossings[i];
            wt += s.weight;
            wd += s.weight * s.theta_deg;
            len += s.length;
            wy += s.weight * y;
            i += 1;
        }
        if best.is_none_or(|b| wt > b.0) {
            best = Some((wt, wd, len, wy));
        }
    }
    let (wt, wd, len, wy) = best?;
    if wt <= 0.0 || len < MIN_HORIZON_LENGTH * w {
        return None;
    }
    Some((
        Estimate {
            tau_deg: wd / wt,
            length: len,
        },
        wy / wt,
    ))
}

/// The answer from the two estimates (`docs/20` §5 step 6).
fn decide(verticals: Option<Estimate>, horizon: Option<Estimate>) -> Straighten {
    let tau = match (verticals, horizon) {
        (Some(v), Some(h)) => {
            if (v.tau_deg - h.tau_deg).abs() > AGREEMENT_DEG {
                return Straighten::Unsure;
            }
            (v.tau_deg * v.length + h.tau_deg * h.length) / (v.length + h.length)
        }
        (Some(e), None) | (None, Some(e)) => e.tau_deg,
        (None, None) => return Straighten::Unsure,
    };
    if !tau.is_finite() || tau.abs() > MAX_TILT_DEG {
        Straighten::Unsure
    } else if tau.abs() <= LEVEL_DEG {
        Straighten::Level
    } else {
        Straighten::Rotate { angle_deg: -tau }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(x: f64, y: f64, theta_deg: f64, length: f64) -> Segment {
        Segment {
            x,
            y,
            theta_deg,
            length,
            weight: length,
        }
    }

    /// A vertical at lateral position `u` leaning by `tau + k u` degrees.
    fn vertical(u: f64, y: f64, tau: f64, k: f64, length: f64) -> Segment {
        let d = tau + k * u;
        let theta = if d <= 0.0 { 90.0 + d } else { d - 90.0 };
        seg(384.0 + u * 768.0, y, theta, length)
    }

    #[test]
    fn verticals_fit_the_roll_and_the_lean_of_a_pitched_camera() {
        let mut segs = Vec::new();
        for i in 0..12 {
            let u = -0.45 + 0.08 * i as f64;
            segs.push(vertical(u, 200.0 + 13.0 * i as f64, 2.3, 9.0, 60.0));
        }
        // Two outliers do not move it.
        segs.push(vertical(0.1, 100.0, 9.0, 0.0, 60.0));
        segs.push(vertical(-0.2, 300.0, -6.0, 0.0, 60.0));
        let v = verticals(&segs, 768.0, 512.0).unwrap();
        assert!((v.tau_deg - 2.3).abs() < 1e-6, "{v:?}");
        assert!((v.length - 720.0).abs() < 1e-9);
    }

    #[test]
    fn verticals_need_three_segments_and_enough_length() {
        let segs: Vec<Segment> = (0..2)
            .map(|i| vertical(0.1 * i as f64, 100.0, 1.0, 0.0, 400.0))
            .collect();
        assert_eq!(verticals(&segs, 768.0, 512.0), None);
        let segs: Vec<Segment> = (0..5)
            .map(|i| vertical(0.1 * i as f64, 100.0, 1.0, 0.0, 50.0))
            .collect();
        assert_eq!(verticals(&segs, 768.0, 512.0), None);
    }

    #[test]
    fn scattered_verticals_do_not_count() {
        let segs: Vec<Segment> = (0..20)
            .map(|i| {
                vertical(
                    -0.4 + 0.04 * i as f64,
                    250.0,
                    (i as f64 * 7.3) % 11.0 - 5.5,
                    0.0,
                    60.0,
                )
            })
            .collect();
        assert_eq!(verticals(&segs, 768.0, 512.0), None);
    }

    #[test]
    fn the_horizon_is_the_heaviest_collinear_cluster() {
        let t = 3.0f64;
        let tan = t.to_radians().tan();
        // Three pieces of one line through (384, 300), and a heavier-looking pair of short
        // parallels at other heights that do not reach 0.35 W on their own.
        let on_line = |x: f64| seg(x, 300.0 + (x - 384.0) * tan, t, 120.0);
        let segs = vec![
            on_line(100.0),
            on_line(400.0),
            on_line(650.0),
            seg(300.0, 100.0, t, 90.0),
            seg(500.0, 400.0, t + 0.1, 90.0),
        ];
        let (h, at) = horizon(&segs, 768.0, 512.0).unwrap();
        assert!((h.tau_deg - t).abs() < 1e-12);
        assert!((h.length - 360.0).abs() < 1e-9);
        // It crosses the centre column where the line does.
        assert!((at - 300.0).abs() < 1e-9, "{at}");
        // Too short a cluster does not count.
        assert_eq!(horizon(&segs[..2], 768.0, 512.0), None);
    }

    #[test]
    fn the_decision_bands() {
        let e = |tau_deg| {
            Some(Estimate {
                tau_deg,
                length: 100.0,
            })
        };
        assert_eq!(decide(None, None), Straighten::Unsure);
        assert_eq!(decide(e(0.15), None), Straighten::Level);
        assert_eq!(decide(None, e(-0.2)), Straighten::Level);
        assert_eq!(decide(e(2.0), None), Straighten::Rotate { angle_deg: -2.0 });
        assert_eq!(decide(e(10.5), None), Straighten::Unsure);
        assert_eq!(decide(e(2.0), e(3.5)), Straighten::Unsure);
        assert_eq!(
            decide(
                Some(Estimate {
                    tau_deg: 2.0,
                    length: 300.0
                }),
                Some(Estimate {
                    tau_deg: 2.4,
                    length: 100.0
                })
            ),
            Straighten::Rotate { angle_deg: -2.1 }
        );
    }

    #[test]
    fn the_answer_serialises_with_its_kind() {
        let j = serde_json::to_string(&Straighten::Rotate { angle_deg: -1.5 }).unwrap();
        assert_eq!(j, r#"{"kind":"rotate","angle_deg":-1.5}"#);
        assert_eq!(
            serde_json::to_string(&Straighten::Level).unwrap(),
            r#"{"kind":"level"}"#
        );
        assert_eq!(
            serde_json::to_string(&Straighten::Unsure).unwrap(),
            r#"{"kind":"unsure"}"#
        );
    }
}
