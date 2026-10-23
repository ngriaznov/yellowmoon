//! The frame proposer (`docs/20-auto-crop.md` §7): for each aspect, the frame closest to the crop
//! model's crop.

use serde::Serialize;

/// Candidate sizes as fractions of the kept frame's sides, largest first.
const SIZES: [f64; 6] = [1.0, 0.94, 0.88, 0.82, 0.76, 0.70];
/// Positions along each axis where a frame has room to move.
const GRID: usize = 9;
/// A frame overlapping a chosen one by more than this intersection over union is dropped.
const MAX_OVERLAP: f64 = 0.6;
/// Frames in a proposal, at most.
const MAX_FRAMES: usize = 3;
/// Halvings of the search that shrinks the Free frame into the photo.
const SHRINK_STEPS: usize = 48;

/// A rectangle of the canvas in fractions of its width and height (`Crop.rect`'s space).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rect {
    /// Left edge.
    pub x: f64,
    /// Top edge.
    pub y: f64,
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

/// The ratio a frame keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ratio {
    /// The canvas's own (Original).
    Canvas,
    /// Width over height in pixels of the canvas.
    Fixed(f64),
    /// The crop model's own shape (Free).
    Free,
}

/// One request: the user's quarter turns (the canvas), the ratio and the straighten angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameRequest {
    /// Clockwise quarter turns of the user's (0 to 3).
    pub quarter_turns: u8,
    /// The ratio of the frames.
    pub ratio: Ratio,
    /// The straighten angle in degrees, clockwise positive.
    pub angle_deg: f64,
}

/// One proposed frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Frame {
    /// The frame on the canvas.
    pub rect: Rect,
    /// Its intersection over union with the crop model's crop, rounded to 4 decimals.
    pub score: f64,
}

/// The frames for one request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Proposal {
    /// Best first, at most 3, never empty.
    pub frames: Vec<Frame>,
    /// The first frame is the kept frame (the largest of the ratio at the angle).
    pub keep: bool,
}

/// The largest frame of `ratio` (width over height) centred on a `canvas_w × canvas_h` canvas whose
/// photo is turned by `angle_deg`, with no empty corner: `(width, height)` in the canvas's units
/// (`docs/20` §7 step 3). The frame also stays inside the canvas, which binds only when
/// `tan(|θ| / 2)` exceeds `min(ratio, 1 / ratio)`, far beyond the straightener's 10°.
#[must_use]
pub fn largest_frame(canvas_w: f64, canvas_h: f64, ratio: f64, angle_deg: f64) -> (f64, f64) {
    let t = angle_deg.to_radians();
    let (c, s) = (t.cos().abs(), t.sin().abs());
    let h = (canvas_w / (ratio * c + s))
        .min(canvas_h / (ratio * s + c))
        .min(canvas_w / ratio)
        .min(canvas_h);
    (ratio * h, h)
}

/// The user's turn: a point of the canvas to the same spot of the picture (fractions), as
/// `ym_core::pipeline::geometry::canvas_to_oriented` defines it.
#[cfg(test)]
fn canvas_to_picture(quarter_turns: u8, [u, v]: [f64; 2]) -> [f64; 2] {
    match quarter_turns % 4 {
        1 => [v, 1.0 - u],
        2 => [1.0 - u, 1.0 - v],
        3 => [1.0 - v, u],
        _ => [u, v],
    }
}

/// A point of the picture to the same spot of the canvas after the user's quarter turns
/// (fractions), as `ym_core::pipeline::geometry::oriented_to_canvas` defines it.
pub(crate) fn picture_to_canvas(quarter_turns: u8, [a, b]: [f64; 2]) -> [f64; 2] {
    match quarter_turns % 4 {
        1 => [1.0 - b, a],
        2 => [1.0 - a, 1.0 - b],
        3 => [b, 1.0 - a],
        _ => [a, b],
    }
}

/// A rectangle in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Area {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl Area {
    fn width(&self) -> f64 {
        self.x1 - self.x0
    }

    fn height(&self) -> f64 {
        self.y1 - self.y0
    }

    fn area(&self) -> f64 {
        self.width() * self.height()
    }

    fn intersection_over_union(&self, o: &Area) -> f64 {
        let w = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let h = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        let inter = w * h;
        let union = self.area() + o.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }

    fn corners(&self) -> [(f64, f64); 4] {
        [
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x0, self.y1),
            (self.x1, self.y1),
        ]
    }

    /// `self` scaled by `s` about its centre.
    fn scaled(&self, s: f64) -> Area {
        let (cx, cy) = (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1));
        let (hw, hh) = (0.5 * s * self.width(), 0.5 * s * self.height());
        Area {
            x0: cx - hw,
            y0: cy - hh,
            x1: cx + hw,
            y1: cy + hh,
        }
    }
}

/// The canvas of one request: the photo turned by the user's quarter turns, then by the
/// straighten angle about the centre (`docs/20` §7 step 1).
struct Canvas {
    /// Canvas size in pixels of the picture.
    width: f64,
    height: f64,
    /// Cosine and sine of the straighten angle.
    cos: f64,
    sin: f64,
}

impl Canvas {
    fn new(width: f64, height: f64, angle_deg: f64) -> Self {
        let (sin, cos) = angle_deg.to_radians().sin_cos();
        Self {
            width,
            height,
            cos,
            sin,
        }
    }

    /// Where a canvas point samples the photo (turned, not yet rotated): `c + R(−angle)(q − c)`.
    fn unrotate(&self, x: f64, y: f64) -> (f64, f64) {
        let (cx, cy) = (0.5 * self.width, 0.5 * self.height);
        let (dx, dy) = (x - cx, y - cy);
        (
            cx + self.cos * dx + self.sin * dy,
            cy - self.sin * dx + self.cos * dy,
        )
    }

    /// Where a point of the photo (turned, not yet rotated) lands on the canvas:
    /// `c + R(angle)(p − c)`.
    fn rotate(&self, x: f64, y: f64) -> (f64, f64) {
        let (cx, cy) = (0.5 * self.width, 0.5 * self.height);
        let (dx, dy) = (x - cx, y - cy);
        (
            cx + self.cos * dx - self.sin * dy,
            cy + self.sin * dx + self.cos * dy,
        )
    }

    /// Whether every corner of `a` is inside the canvas and the turned photo.
    fn holds(&self, a: &Area) -> bool {
        let eps = 1e-9 * (self.width + self.height);
        a.corners().iter().all(|&(x, y)| {
            let (px, py) = self.unrotate(x, y);
            (-eps..=self.width + eps).contains(&x)
                && (-eps..=self.height + eps).contains(&y)
                && px >= -eps
                && px <= self.width + eps
                && py >= -eps
                && py <= self.height + eps
        })
    }
}

/// Positions of a frame's near edge along an axis with `slack` pixels of room: 9 evenly spaced,
/// or one, centred, where the frame fits exactly.
fn positions(slack: f64, side: f64) -> Vec<f64> {
    if slack <= 1e-9 * side {
        vec![slack / 2.0]
    } else {
        (0..GRID)
            .map(|i| slack * i as f64 / (GRID - 1) as f64)
            .collect()
    }
}

/// The candidates of `docs/20` §7 step 4, the kept frame first.
fn candidates(canvas: &Canvas, kept: Area) -> Vec<Area> {
    let mut out = vec![kept];
    for (si, &s) in SIZES.iter().enumerate() {
        let (w, h) = (s * kept.width(), s * kept.height());
        let xs = positions(canvas.width - w, canvas.width);
        let ys = positions(canvas.height - h, canvas.height);
        for (iy, &y) in ys.iter().enumerate() {
            for (ix, &x) in xs.iter().enumerate() {
                // The centred frame of the largest size is the kept frame, already first.
                if si == 0 && ix == xs.len() / 2 && iy == ys.len() / 2 {
                    continue;
                }
                let a = Area {
                    x0: x,
                    y0: y,
                    x1: x + w,
                    y1: y + h,
                };
                if canvas.holds(&a) {
                    out.push(a);
                }
            }
        }
    }
    out
}

/// The crop model's crop (`[x0, y0, x1, y1]` fractions of the picture) on the canvas
/// (`docs/20` §7 step 2): its corners turned by the quarter turns, its centre moved where the
/// straighten angle takes it, its size kept.
fn target(crop: [f64; 4], quarter_turns: u8, canvas: &Canvas) -> Area {
    let [a, b] = picture_to_canvas(quarter_turns, [crop[0], crop[1]]);
    let [c, d] = picture_to_canvas(quarter_turns, [crop[2], crop[3]]);
    let (x0, x1) = (a.min(c) * canvas.width, a.max(c) * canvas.width);
    let (y0, y1) = (b.min(d) * canvas.height, b.max(d) * canvas.height);
    let (cx, cy) = canvas.rotate(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    let (hw, hh) = (0.5 * (x1 - x0), 0.5 * (y1 - y0));
    Area {
        x0: cx - hw,
        y0: cy - hh,
        x1: cx + hw,
        y1: cy + hh,
    }
}

/// Rounds to multiples of `1 / scale`, writing 0 rather than −0.
fn round_to(v: f64, scale: f64) -> f64 {
    let r = (v * scale).round() / scale;
    if r == 0.0 { 0.0 } else { r }
}

/// A frame in canvas pixels as a [`Rect`] of canvas fractions rounded to 6 decimals, inside
/// `[0, 1]`.
fn to_rect(a: &Area, width: f64, height: f64) -> Rect {
    let f = |v: f64, side: f64| round_to((v / side).clamp(0.0, 1.0), 1e6);
    let (x0, x1) = (f(a.x0, width), f(a.x1, width));
    let (y0, y1) = (f(a.y0, height), f(a.y1, height));
    Rect {
        x: x0,
        y: y0,
        width: round_to(x1 - x0, 1e6),
        height: round_to(y1 - y0, 1e6),
    }
}

/// The frames for one request (`docs/20` §7) on a `width × height` picture whose crop model
/// answered `crop`.
pub(crate) fn propose(
    width: usize,
    height: usize,
    crop: [f64; 4],
    request: &FrameRequest,
) -> Proposal {
    let turns = request.quarter_turns % 4;
    let (cw, ch) = if turns % 2 == 1 {
        (height as f64, width as f64)
    } else {
        (width as f64, height as f64)
    };
    let angle = if request.angle_deg.is_finite() {
        request.angle_deg
    } else {
        0.0
    };
    let crop = if crop.iter().all(|v| v.is_finite()) {
        crop.map(|v| v.clamp(0.0, 1.0))
    } else {
        [0.0, 0.0, 1.0, 1.0]
    };
    let canvas = Canvas::new(cw, ch, angle);
    let target = target(crop, turns, &canvas);
    let centred = |r: f64| {
        let (w, h) = largest_frame(cw, ch, r, angle);
        Area {
            x0: (cw - w) / 2.0,
            y0: (ch - h) / 2.0,
            x1: (cw - w) / 2.0 + w,
            y1: (ch - h) / 2.0 + h,
        }
    };
    let ratio = match request.ratio {
        Ratio::Fixed(r) if r.is_finite() && r > 0.0 => r,
        Ratio::Canvas | Ratio::Fixed(_) => cw / ch,
        Ratio::Free => {
            // The crop itself, shrunk about its centre (keeping its shape) by as little as puts
            // it inside the canvas and the turned photo.
            let kept = centred(cw / ch);
            let frame = if canvas.holds(&target) {
                target
            } else {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..SHRINK_STEPS {
                    let mid = 0.5 * (lo + hi);
                    if canvas.holds(&target.scaled(mid)) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                target.scaled(lo)
            };
            let rect = to_rect(&frame, cw, ch);
            return Proposal {
                keep: rect == to_rect(&kept, cw, ch),
                frames: vec![Frame {
                    rect,
                    score: round_to(frame.intersection_over_union(&target), 1e4),
                }],
            };
        }
    };
    let kept = centred(ratio);
    let cands = candidates(&canvas, kept);
    // Closest first; a stable sort keeps the lower index (the larger frame) first on a tie.
    let mut order: Vec<(usize, f64)> = cands
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.intersection_over_union(&target)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut chosen: Vec<(usize, f64)> = Vec::with_capacity(MAX_FRAMES);
    for &(i, s) in &order {
        if chosen.len() == MAX_FRAMES {
            break;
        }
        if chosen
            .iter()
            .any(|&(c, _)| cands[c].intersection_over_union(&cands[i]) > MAX_OVERLAP)
        {
            continue;
        }
        chosen.push((i, s));
    }
    Proposal {
        keep: chosen[0].0 == 0,
        frames: chosen
            .iter()
            .map(|&(i, s)| Frame {
                rect: to_rect(&cands[i], cw, ch),
                score: round_to(s, 1e4),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The picture of the tests: 768 × 512.
    const PW: usize = 768;
    const PH: usize = 512;

    fn request(quarter_turns: u8, ratio: Ratio, angle_deg: f64) -> FrameRequest {
        FrameRequest {
            quarter_turns,
            ratio,
            angle_deg,
        }
    }

    fn area(r: &Rect) -> Area {
        Area {
            x0: r.x,
            y0: r.y,
            x1: r.x + r.width,
            y1: r.y + r.height,
        }
    }

    #[test]
    fn the_turns_map_points_as_geometry_does_and_back() {
        for turns in 0..4u8 {
            for &(u, v) in &[(0.1, 0.2), (0.5, 0.5), (0.9, 0.35)] {
                let [a, b] = canvas_to_picture(turns, [u, v]);
                let back = picture_to_canvas(turns, [a, b]);
                assert!((back[0] - u).abs() < 1e-12 && (back[1] - v).abs() < 1e-12);
            }
        }
        // One clockwise quarter turn takes the picture's left edge to the canvas's top.
        assert_eq!(picture_to_canvas(1, [0.0, 0.5]), [0.5, 0.0]);
    }

    #[test]
    fn largest_frame_matches_a_brute_force_search() {
        // The tallest centred frame of the ratio whose corners are inside both the canvas and the
        // turned photo, found by scanning heights.
        let inside = |cw: f64, ch: f64, r: f64, deg: f64, h: f64| {
            let w = r * h;
            if w > cw * (1.0 + 1e-12) || h > ch * (1.0 + 1e-12) {
                return false;
            }
            let (s, c) = deg.to_radians().sin_cos();
            [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                .iter()
                .all(|&(sx, sy)| {
                    let (x, y) = (sx * w / 2.0, sy * h / 2.0);
                    // Back into the photo: R(−θ).
                    let (px, py) = (c * x + s * y, -s * x + c * y);
                    px.abs() <= cw / 2.0 * (1.0 + 1e-12) && py.abs() <= ch / 2.0 * (1.0 + 1e-12)
                })
        };
        for &(cw, ch) in &[(768.0, 512.0), (512.0, 768.0), (600.0, 600.0)] {
            for &r in &[cw / ch, 1.0, 0.8, 1.25, 16.0 / 9.0, 9.0 / 16.0, 3.0] {
                for &deg in &[0.0, 0.3, -1.4, 4.0, -7.5, 10.0, 25.0, -45.0] {
                    let (w, h) = largest_frame(cw, ch, r, deg);
                    assert!((w / h - r).abs() < 1e-9);
                    let steps = 200_000;
                    let step = ch.max(cw) / steps as f64;
                    let mut best = 0.0;
                    for k in 1..=steps {
                        let hk = k as f64 * step;
                        if inside(cw, ch, r, deg, hk) {
                            best = hk;
                        } else {
                            break;
                        }
                    }
                    assert!(
                        (h - best).abs() <= step * 1.0001,
                        "{cw} x {ch}, ratio {r}, {deg}°: formula {h}, search {best}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_whole_crop_keeps_the_photo() {
        for ratio in [Ratio::Canvas, Ratio::Free] {
            let p = propose(PW, PH, [0.0, 0.0, 1.0, 1.0], &request(0, ratio, 0.0));
            assert!(p.keep, "{ratio:?}: {p:?}");
            let r = p.frames[0].rect;
            assert_eq!((r.x, r.y, r.width, r.height), (0.0, 0.0, 1.0, 1.0));
            assert_eq!(p.frames[0].score, 1.0);
        }
    }

    #[test]
    fn the_closest_candidate_comes_first() {
        let crops = [
            [0.05, 0.1, 0.7, 0.95],
            [0.3, 0.0, 1.0, 0.8],
            [0.2, 0.25, 0.6, 0.75],
            [0.0, 0.3, 0.55, 1.0],
        ];
        for crop in crops {
            for turns in 0..4u8 {
                for ratio in [Ratio::Canvas, Ratio::Fixed(1.0), Ratio::Fixed(16.0 / 9.0)] {
                    for angle in [0.0, 2.5, -6.0] {
                        let p = propose(PW, PH, crop, &request(turns, ratio, angle));
                        let (cw, ch) = if turns % 2 == 1 {
                            (512.0, 768.0)
                        } else {
                            (768.0, 512.0)
                        };
                        let canvas = Canvas::new(cw, ch, angle);
                        let t = target(crop, turns, &canvas);
                        let r = match ratio {
                            Ratio::Fixed(r) => r,
                            _ => cw / ch,
                        };
                        let (kw, kh) = largest_frame(cw, ch, r, angle);
                        let kept = Area {
                            x0: (cw - kw) / 2.0,
                            y0: (ch - kh) / 2.0,
                            x1: (cw + kw) / 2.0,
                            y1: (ch + kh) / 2.0,
                        };
                        let best = candidates(&canvas, kept)
                            .iter()
                            .map(|c| c.intersection_over_union(&t))
                            .fold(0.0, f64::max);
                        assert!(
                            (p.frames[0].score - best).abs() < 1e-4,
                            "{crop:?} turns {turns} {ratio:?} {angle}°: {} vs {best}",
                            p.frames[0].score
                        );
                        assert!(p.frames.windows(2).all(|w| w[0].score >= w[1].score));
                    }
                }
            }
        }
    }

    #[test]
    fn free_gives_the_crop_itself_turned_with_the_photo() {
        let crop = [0.1, 0.2, 0.4, 0.9];
        let f = propose(PW, PH, crop, &request(0, Ratio::Free, 0.0)).frames[0].rect;
        assert_eq!((f.x, f.y, f.width, f.height), (0.1, 0.2, 0.3, 0.7));
        // After one clockwise quarter turn the left part of the picture is the top of the canvas.
        let f = propose(PW, PH, crop, &request(1, Ratio::Free, 0.0)).frames[0].rect;
        assert!(
            (f.x - 0.1).abs() < 1e-6 && (f.y - 0.1).abs() < 1e-6,
            "{f:?}"
        );
        assert!(
            (f.width - 0.7).abs() < 1e-6 && (f.height - 0.3).abs() < 1e-6,
            "{f:?}"
        );
        // At an angle a crop at the photo's edge is shrunk into the turned photo.
        let p = propose(PW, PH, [0.0, 0.0, 0.5, 1.0], &request(0, Ratio::Free, 4.0));
        assert!(p.frames[0].score < 1.0 && p.frames[0].score > 0.8, "{p:?}");
    }

    #[test]
    fn a_crop_on_one_side_moves_the_square_there() {
        let left = [0.0, 0.0, 0.55, 1.0];
        let first =
            |turns| propose(PW, PH, left, &request(turns, Ratio::Fixed(1.0), 0.0)).frames[0].rect;
        assert!(first(0).x < 0.02, "{:?}", first(0));
        assert!(first(1).y < 0.02, "{:?}", first(1));
        assert!(first(2).x + first(2).width > 0.98, "{:?}", first(2));
        assert!(first(3).y + first(3).height > 0.98, "{:?}", first(3));
    }

    #[test]
    fn frames_are_inside_the_canvas_and_the_turned_photo_with_the_ratio() {
        let crops = [
            [0.1, 0.05, 0.8, 0.9],
            [0.0, 0.0, 1.0, 1.0],
            [0.45, 0.3, 0.98, 0.99],
        ];
        for crop in crops {
            for turns in 0..4u8 {
                let (cw, ch) = if turns % 2 == 1 {
                    (512.0, 768.0)
                } else {
                    (768.0, 512.0)
                };
                for ratio in [
                    Ratio::Canvas,
                    Ratio::Free,
                    Ratio::Fixed(1.0),
                    Ratio::Fixed(0.8),
                    Ratio::Fixed(1.25),
                    Ratio::Fixed(16.0 / 9.0),
                    Ratio::Fixed(9.0 / 16.0),
                ] {
                    for angle in [0.0, 0.7, -3.3, 8.0, -10.0] {
                        let p = propose(PW, PH, crop, &request(turns, ratio, angle));
                        assert!(!p.frames.is_empty() && p.frames.len() <= MAX_FRAMES);
                        let (s, c) = angle.to_radians().sin_cos();
                        for f in &p.frames {
                            let q = f.rect;
                            for v in [q.x, q.y, q.width, q.height, q.x + q.width, q.y + q.height] {
                                assert!((-1e-12..=1.0 + 1e-9).contains(&v), "{q:?}");
                            }
                            let r = match ratio {
                                Ratio::Canvas => Some(cw / ch),
                                Ratio::Fixed(r) => Some(r),
                                Ratio::Free => None,
                            };
                            if let Some(r) = r {
                                let got = q.width * cw / (q.height * ch);
                                assert!(
                                    (got / r - 1.0).abs() < 1e-4,
                                    "{ratio:?} {angle}°: {got} for {r}"
                                );
                            }
                            for (x, y) in [
                                (q.x, q.y),
                                (q.x + q.width, q.y),
                                (q.x, q.y + q.height),
                                (q.x + q.width, q.y + q.height),
                            ] {
                                let (dx, dy) = (x * cw - cw / 2.0, y * ch - ch / 2.0);
                                let (px, py) = (c * dx + s * dy, -s * dx + c * dy);
                                assert!(
                                    px.abs() <= cw / 2.0 + 1e-3 && py.abs() <= ch / 2.0 + 1e-3,
                                    "turns {turns} {ratio:?} {angle}°: {q:?}"
                                );
                            }
                            for v in [q.x, q.y, q.width, q.height] {
                                assert!(((v * 1e6).round() - v * 1e6).abs() < 1e-6);
                            }
                            assert!(((f.score * 1e4).round() - f.score * 1e4).abs() < 1e-6);
                        }
                        for (i, a) in p.frames.iter().enumerate() {
                            for b in &p.frames[i + 1..] {
                                assert!(
                                    area(&a.rect).intersection_over_union(&area(&b.rect))
                                        <= MAX_OVERLAP + 1e-4
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
