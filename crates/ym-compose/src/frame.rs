//! The frame proposer (`docs/20-auto-crop.md` §7).

use serde::Serialize;

use crate::CameraHint;
use crate::subject::SubjectMaps;

/// Candidate sizes as fractions of the kept frame's sides, largest first.
const SIZES: [f64; 6] = [1.0, 0.94, 0.88, 0.82, 0.76, 0.70];
/// Positions along each axis where a frame has room to move.
const GRID: usize = 9;
/// The band inside a frame's edges where the subject counts as cut, over the canvas's short side.
const CUT_BAND: f64 = 0.03;
/// Weight of the subject mass a frame loses against the kept frame.
const LOST_WEIGHT: f64 = 4.0;
/// Weight of the subject mass a frame gains over the kept frame: less than a loss, so the
/// photographer's framing wins unless the gain is large.
const GAIN_WEIGHT: f64 = 1.0;
/// Weight of the subject mass in the band inside a frame's edges.
const CUT_WEIGHT: f64 = 2.0;
/// Weight of a frame's shift from the kept frame's centre, in the kept frame's sides.
const SHIFT_WEIGHT: f64 = 0.5;
/// Weight of the area given up.
const AREA_WEIGHT: f64 = 1.5;
/// Weight of how much further a frame moves the horizon from where the photographer put it than
/// the kept frame does, in the frame's height (its width when a quarter turn stands it up).
const HORIZON_WEIGHT: f64 = 2.0;
/// The camera's AF box grows by this fraction of its size on each side before frames must hold it.
const BOX_GROWTH: f64 = 0.1;
/// A frame overlapping a chosen one by more than this intersection over union is dropped.
const MAX_OVERLAP: f64 = 0.6;
/// Frames in a proposal, at most.
const MAX_FRAMES: usize = 3;
/// The best score must exceed this for a frame to come before the kept frame (`docs/20` §7 step 6:
/// on real photos, frames that scored less lost to the centred frame more often than they won) ...
const MIN_SCORE: f64 = 0.15;
/// ... or this when the photographer placed the AF point on purpose.
const MIN_SCORE_PLACED: f64 = 0.3;

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
    /// The canvas's own (Original and Free).
    Canvas,
    /// Width over height in pixels of the canvas.
    Fixed(f64),
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
    /// Its score (0 for the kept frame), rounded to 4 decimals.
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
/// `ym_core::pipeline::geometry::canvas_to_oriented` defines it. [`turned_cell`] is its exact
/// permutation of cells.
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

/// The map cell (column, row) of a `map_w × map_h` map that canvas cell `(i, j)` shows after the
/// user's quarter turns: the exact permutation `geometry::canvas_to_oriented` makes of the cell
/// centres.
fn turned_cell(
    quarter_turns: u8,
    map_w: usize,
    map_h: usize,
    i: usize,
    j: usize,
) -> (usize, usize) {
    match quarter_turns % 4 {
        1 => (j, map_h - 1 - i),
        2 => (map_w - 1 - i, map_h - 1 - j),
        3 => (map_w - 1 - j, i),
        _ => (i, j),
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

    fn contains(&self, (x, y): (f64, f64), eps: f64) -> bool {
        x >= self.x0 - eps && x <= self.x1 + eps && y >= self.y0 - eps && y <= self.y1 + eps
    }
}

/// A summed-area table of a `width × height` grid of cells, read at fractional cell positions as
/// if each cell's value were spread evenly over it.
struct SummedArea {
    width: usize,
    height: usize,
    /// `(width + 1) × (height + 1)`: entry `(i, j)` sums the cells left of column `i` and above
    /// row `j`.
    table: Vec<f64>,
}

impl SummedArea {
    fn new(width: usize, height: usize, values: &[f64]) -> Self {
        let stride = width + 1;
        let mut table = vec![0.0; stride * (height + 1)];
        for j in 0..height {
            let mut row = 0.0;
            for i in 0..width {
                row += values[j * width + i];
                table[(j + 1) * stride + i + 1] = table[j * stride + i + 1] + row;
            }
        }
        Self {
            width,
            height,
            table,
        }
    }

    /// The sum over `[0, x] × [0, y]`, `x` and `y` in cells (bilinear between table entries).
    fn at(&self, x: f64, y: f64) -> f64 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let x = x.clamp(0.0, self.width as f64);
        let y = y.clamp(0.0, self.height as f64);
        let i = (x.floor() as usize).min(self.width - 1);
        let j = (y.floor() as usize).min(self.height - 1);
        let (tx, ty) = (x - i as f64, y - j as f64);
        let stride = self.width + 1;
        let t = |i: usize, j: usize| self.table[j * stride + i];
        let top = t(i, j) + tx * (t(i + 1, j) - t(i, j));
        let bottom = t(i, j + 1) + tx * (t(i + 1, j + 1) - t(i, j + 1));
        top + ty * (bottom - top)
    }

    /// The sum over the rectangle `[x0, x1] × [y0, y1]` in cells.
    fn sum(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        self.at(x1, y1) - self.at(x0, y1) - self.at(x1, y0) + self.at(x0, y0)
    }
}

/// The maps on the canvas of one request (`docs/20` §7 step 1).
struct Canvas {
    /// Canvas size in pixels of the picture.
    width: f64,
    height: f64,
    /// Cosine and sine of the straighten angle.
    cos: f64,
    sin: f64,
    /// Cells across and down.
    cells_w: usize,
    cells_h: usize,
    subject: SummedArea,
    distraction: SummedArea,
    rho: f64,
}

impl Canvas {
    fn new(width: f64, height: f64, maps: &SubjectMaps, quarter_turns: u8, angle_deg: f64) -> Self {
        let (sin, cos) = angle_deg.to_radians().sin_cos();
        let turned = quarter_turns % 2 == 1;
        let (cells_w, cells_h) = if turned {
            (maps.height, maps.width)
        } else {
            (maps.width, maps.height)
        };
        let n = cells_w * cells_h;
        let have_maps = n > 0 && maps.subject.len() == n && maps.distraction.len() == n;
        let mut canvas = Self {
            width,
            height,
            cos,
            sin,
            cells_w,
            cells_h,
            subject: SummedArea::new(0, 0, &[]),
            distraction: SummedArea::new(0, 0, &[]),
            rho: maps.rho,
        };
        if !have_maps {
            return canvas;
        }
        // The maps turned by the quarter turns: an exact permutation.
        let mut ts = vec![0.0; n];
        let mut td = vec![0.0; n];
        for j in 0..cells_h {
            for i in 0..cells_w {
                let (x, y) = turned_cell(quarter_turns, maps.width, maps.height, i, j);
                ts[j * cells_w + i] = maps.subject[y * maps.width + x];
                td[j * cells_w + i] = maps.distraction[y * maps.width + x];
            }
        }
        // Sampled at the straighten angle: each cell centre rotated back about the canvas centre,
        // bilinear in the turned maps, 0 outside the photo.
        let (sx, sy) = (width / cells_w as f64, height / cells_h as f64);
        let (mut s, mut d) = (ts.clone(), td.clone());
        if angle_deg != 0.0 {
            for j in 0..cells_h {
                for i in 0..cells_w {
                    let (px, py) = canvas.unrotate((i as f64 + 0.5) * sx, (j as f64 + 0.5) * sy);
                    let k = j * cells_w + i;
                    if !(0.0..=width).contains(&px) || !(0.0..=height).contains(&py) {
                        s[k] = 0.0;
                        d[k] = 0.0;
                        continue;
                    }
                    let (fx, fy) = (px / sx - 0.5, py / sy - 0.5);
                    s[k] = bilinear(&ts, cells_w, cells_h, fx, fy);
                    d[k] = bilinear(&td, cells_w, cells_h, fx, fy);
                }
            }
        }
        canvas.subject = SummedArea::new(cells_w, cells_h, &s);
        canvas.distraction = SummedArea::new(cells_w, cells_h, &d);
        canvas
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

    /// Whether every corner of `a` is inside the turned photo.
    fn holds(&self, a: &Area) -> bool {
        let eps = 1e-9 * (self.width + self.height);
        a.corners().iter().all(|&(x, y)| {
            let (px, py) = self.unrotate(x, y);
            px >= -eps && px <= self.width + eps && py >= -eps && py <= self.height + eps
        })
    }

    /// The sum of a table over a rectangle in canvas pixels.
    fn sum(&self, table: &SummedArea, a: &Area) -> f64 {
        let (kx, ky) = (
            self.cells_w as f64 / self.width,
            self.cells_h as f64 / self.height,
        );
        table.sum(a.x0 * kx, a.y0 * ky, a.x1 * kx, a.y1 * ky)
    }

    /// Cell size in canvas pixels, across and down.
    fn cell_size(&self) -> (f64, f64) {
        if self.cells_w == 0 || self.cells_h == 0 {
            (crate::picture::CELL as f64, crate::picture::CELL as f64)
        } else {
            (
                self.width / self.cells_w as f64,
                self.height / self.cells_h as f64,
            )
        }
    }
}

/// The bilinear value of a `w × h` grid at `(x, y)` in cell-centre coordinates (cell `i` has its
/// centre at `i`), clamped at the edges.
fn bilinear(values: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (i0, j0) = (x.floor() as usize, y.floor() as usize);
    let (i1, j1) = ((i0 + 1).min(w - 1), (j0 + 1).min(h - 1));
    let (tx, ty) = (x - i0 as f64, y - j0 as f64);
    let v = |i: usize, j: usize| values[j * w + i];
    let top = v(i0, j0) + tx * (v(i1, j0) - v(i0, j0));
    let bottom = v(i0, j1) + tx * (v(i1, j1) - v(i0, j1));
    top + ty * (bottom - top)
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

/// The camera's AF box grown by [`BOX_GROWTH`] on each side and clamped to the picture, as its
/// four corners on the canvas.
fn camera_box(hint: &CameraHint, quarter_turns: u8, canvas: &Canvas) -> Option<[(f64, f64); 4]> {
    let [px, py] = hint.point?;
    let [bw, bh] = hint.box_size?;
    if !(px.is_finite() && py.is_finite() && bw.is_finite() && bh.is_finite()) {
        return None;
    }
    let (hw, hh) = (bw.abs() * (0.5 + BOX_GROWTH), bh.abs() * (0.5 + BOX_GROWTH));
    let (x0, x1) = ((px - hw).clamp(0.0, 1.0), (px + hw).clamp(0.0, 1.0));
    let (y0, y1) = ((py - hh).clamp(0.0, 1.0), (py + hh).clamp(0.0, 1.0));
    let corner = |a: f64, b: f64| {
        let [u, v] = picture_to_canvas(quarter_turns, [a, b]);
        canvas.rotate(u * canvas.width, v * canvas.height)
    };
    Some([
        corner(x0, y0),
        corner(x1, y0),
        corner(x0, y1),
        corner(x1, y1),
    ])
}

/// `r` less the bands `band` px wide inside its edges that lie more than one cell inside `outer`'s:
/// the subject between the two is what `r` cuts (`docs/20` §7 step 5).
fn without_bands(r: &Area, outer: &Area, band: f64, (cell_x, cell_y): (f64, f64)) -> Area {
    let shrink = |active: bool| if active { band } else { 0.0 };
    Area {
        x0: r.x0 + shrink(r.x0 > outer.x0 + cell_x),
        x1: r.x1 - shrink(r.x1 < outer.x1 - cell_x),
        y0: r.y0 + shrink(r.y0 > outer.y0 + cell_y),
        y1: r.y1 - shrink(r.y1 < outer.y1 - cell_y),
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

/// The frames for one request (`docs/20` §7) on a `width × height` picture.
pub(crate) fn propose(
    width: usize,
    height: usize,
    maps: &SubjectMaps,
    hint: &CameraHint,
    horizon: Option<f64>,
    request: &FrameRequest,
) -> Proposal {
    let turns = request.quarter_turns % 4;
    let (cw, ch) = if turns % 2 == 1 {
        (height as f64, width as f64)
    } else {
        (width as f64, height as f64)
    };
    let ratio = match request.ratio {
        Ratio::Fixed(r) if r.is_finite() && r > 0.0 => r,
        Ratio::Canvas | Ratio::Fixed(_) => cw / ch,
    };
    let angle = if request.angle_deg.is_finite() {
        request.angle_deg
    } else {
        0.0
    };
    let (kw, kh) = largest_frame(cw, ch, ratio, angle);
    let kept = Area {
        x0: (cw - kw) / 2.0,
        y0: (ch - kh) / 2.0,
        x1: (cw - kw) / 2.0 + kw,
        y1: (ch - kh) / 2.0 + kh,
    };
    let canvas = Canvas::new(cw, ch, maps, turns, angle);
    let cands = candidates(&canvas, kept);
    let held = camera_box(hint, turns, &canvas);

    // What every score is measured against: the kept frame's.
    let kept_subject = canvas.sum(&canvas.subject, &kept);
    let kept_distraction = canvas.sum(&canvas.distraction, &kept);
    let kept_area = kept.area();
    let band = CUT_BAND * cw.min(ch);
    let cell = canvas.cell_size();
    // The subject mass a frame's edges slice through: in the bands inside its edges that lie
    // inside the canvas (an edge on the canvas's border cuts nothing). The kept frame of an aspect
    // other than the canvas's can slice through the subject too, and a frame that slices less
    // gains.
    let whole = Area {
        x0: 0.0,
        y0: 0.0,
        x1: cw,
        y1: ch,
    };
    let cut_of = |r: &Area| {
        canvas.sum(&canvas.subject, r)
            - canvas.sum(&canvas.subject, &without_bands(r, &whole, band, cell))
    };
    let kept_cut = cut_of(&kept);
    let (kcx, kcy) = (0.5 * (kept.x0 + kept.x1), 0.5 * (kept.y0 + kept.y1));
    // The horizon on the canvas, along the axis across it (x when a quarter turn stood it up), in
    // canvas pixels, and where the photographer put it: its fraction of the canvas. The straighten
    // angle moves its crossing of the centre line by under 2 % of its distance from the centre at
    // 10°, so it is left out. A frame's horizon offset is how far from that fraction the horizon
    // sits in the frame.
    let horizon = horizon.map(|h| {
        let [u, v] = picture_to_canvas(turns, [0.5, h]);
        if turns % 2 == 1 {
            (true, u * cw, u)
        } else {
            (false, v * ch, v)
        }
    });
    let horizon_offset = |r: &Area| {
        horizon.map_or(0.0, |(across_x, at, placed)| {
            let (lo, hi) = if across_x { (r.x0, r.x1) } else { (r.y0, r.y1) };
            ((at - lo) / (hi - lo) - placed).abs()
        })
    };
    let kept_horizon_offset = horizon_offset(&kept);

    // With a camera box, frames that cut it are dropped, the kept frame too.
    let holds_box = |r: &Area| {
        held.as_ref()
            .is_none_or(|corners| corners.iter().all(|&p| r.contains(p, 1e-9 * (cw + ch))))
    };
    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(cands.len());
    if holds_box(&kept) {
        scored.push((0, 0.0));
    }
    for (i, r) in cands.iter().enumerate().skip(1) {
        if !holds_box(r) {
            continue;
        }
        let inside = canvas.sum(&canvas.subject, r);
        let lost = (kept_subject - inside).max(0.0);
        let gained = (inside - kept_subject).max(0.0);
        let cut = cut_of(r) - kept_cut;
        let removed = canvas.rho * (kept_distraction - canvas.sum(&canvas.distraction, r));
        // How far the frame moves the photo's centre: the photographer framed it, so a frame
        // moves only for what it holds or leaves out.
        let shift = (0.5 * (r.x0 + r.x1) - kcx).hypot(0.5 * (r.y0 + r.y1) - kcy)
            / (kept.x1 - kept.x0).max(kept.y1 - kept.y0);
        let a = r.area() / kept_area;
        // Only a frame that moves the horizon further than the kept frame pays: a sky or a field
        // the kept frame's edge crosses would otherwise pull the frame to the photo's top or
        // bottom, where an edge cuts nothing.
        let horizon_moved = (horizon_offset(r) - kept_horizon_offset).max(0.0);
        let score = -LOST_WEIGHT * lost + GAIN_WEIGHT * gained - CUT_WEIGHT * cut + removed
            - SHIFT_WEIGHT * shift
            - AREA_WEIGHT * (1.0 - a)
            - HORIZON_WEIGHT * horizon_moved;
        scored.push((i, score));
    }

    // When every frame cuts the box, the kept frame alone is the answer: it is never empty.
    if scored.is_empty() {
        scored.push((0, 0.0));
    }
    // Best first; a stable sort keeps the lower index first on a tie.
    let mut order = scored;
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let threshold = if hint.placed && hint.point.is_some() {
        MIN_SCORE_PLACED
    } else {
        MIN_SCORE
    };
    // `order` is never empty (above), so `chosen` gets at least its first entry.
    let mut chosen: Vec<(usize, f64)> = Vec::with_capacity(MAX_FRAMES);
    if order[0].1 <= threshold && order.iter().any(|&(i, _)| i == 0) {
        chosen.push((0, 0.0));
    }
    for &(i, s) in &order {
        if chosen.len() == MAX_FRAMES {
            break;
        }
        if chosen
            .iter()
            .any(|&(c, _)| c == i || cands[c].intersection_over_union(&cands[i]) > MAX_OVERLAP)
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
                score: if i == 0 { 0.0 } else { round_to(s, 1e4) },
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 192 × 128 cells: the map of a 768 × 512 picture.
    const MW: usize = 192;
    const MH: usize = 128;

    /// Maps with uniform blocks `(x0, y0, x1, y1)` in cells for the subject and the distractions.
    fn maps(
        subject: &[(usize, usize, usize, usize)],
        distraction: &[(usize, usize, usize, usize)],
        rho: f64,
    ) -> SubjectMaps {
        let fill = |blocks: &[(usize, usize, usize, usize)]| {
            let mut v = vec![0.0; MW * MH];
            for &(x0, y0, x1, y1) in blocks {
                for y in y0..y1 {
                    for x in x0..x1 {
                        v[y * MW + x] = 1.0;
                    }
                }
            }
            let s: f64 = v.iter().sum();
            if s > 0.0 {
                v.iter_mut().for_each(|x| *x /= s);
            }
            v
        };
        SubjectMaps {
            width: MW,
            height: MH,
            subject: fill(subject),
            distraction: fill(distraction),
            rho,
        }
    }

    fn request(quarter_turns: u8, ratio: Ratio, angle_deg: f64) -> FrameRequest {
        FrameRequest {
            quarter_turns,
            ratio,
            angle_deg,
        }
    }

    fn run(m: &SubjectMaps, hint: &CameraHint, r: FrameRequest) -> Proposal {
        propose(MW * 4, MH * 4, m, hint, None, &r)
    }

    fn holds_rect(outer: &Rect, inner: (f64, f64, f64, f64)) -> bool {
        let e = 2e-6;
        inner.0 >= outer.x - e
            && inner.1 >= outer.y - e
            && inner.2 <= outer.x + outer.width + e
            && inner.3 <= outer.y + outer.height + e
    }

    #[test]
    fn the_turned_maps_are_the_permutation_geometry_defines() {
        let (w, h) = (7, 5);
        for turns in 0..4u8 {
            let (cw, ch) = if turns % 2 == 1 { (h, w) } else { (w, h) };
            for j in 0..ch {
                for i in 0..cw {
                    let u = (i as f64 + 0.5) / cw as f64;
                    let v = (j as f64 + 0.5) / ch as f64;
                    let [a, b] = canvas_to_picture(turns, [u, v]);
                    let expect = (
                        (a * w as f64).floor() as usize,
                        (b * h as f64).floor() as usize,
                    );
                    assert_eq!(
                        turned_cell(turns, w, h, i, j),
                        expect,
                        "turns {turns} cell {i} {j}"
                    );
                    let back = picture_to_canvas(turns, [a, b]);
                    assert!((back[0] - u).abs() < 1e-12 && (back[1] - v).abs() < 1e-12);
                }
            }
        }
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
    fn without_a_subject_the_kept_frame_is_first() {
        let m = maps(&[], &[], 0.0);
        let p = run(&m, &CameraHint::default(), request(0, Ratio::Canvas, 0.0));
        assert!(p.keep);
        assert_eq!(
            p.frames[0],
            Frame {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0
                },
                score: 0.0
            }
        );
        let p = run(
            &m,
            &CameraHint::default(),
            request(0, Ratio::Fixed(1.0), 0.0),
        );
        assert!(p.keep);
        assert_eq!(
            p.frames[0].rect,
            Rect {
                x: 0.166667,
                y: 0.0,
                width: 0.666666,
                height: 1.0
            }
        );
        // At an angle the kept frame is the largest of the ratio with no empty corner.
        let p = run(&m, &CameraHint::default(), request(0, Ratio::Canvas, 4.0));
        let (w, h) = largest_frame(768.0, 512.0, 1.5, 4.0);
        assert!(p.keep);
        assert!((p.frames[0].rect.width - w / 768.0).abs() < 2e-6);
        assert!((p.frames[0].rect.height - h / 512.0).abs() < 2e-6);
    }

    #[test]
    fn a_centred_subject_is_kept() {
        let m = maps(&[(80, 48, 112, 80)], &[], 0.0);
        for ratio in [
            Ratio::Canvas,
            Ratio::Fixed(1.0),
            Ratio::Fixed(0.8),
            Ratio::Fixed(16.0 / 9.0),
        ] {
            for angle in [0.0, 2.5] {
                let p = run(&m, &CameraHint::default(), request(0, ratio, angle));
                assert!(p.keep, "{ratio:?} at {angle}°: {p:?}");
                assert_eq!(p.frames[0].score, 0.0);
            }
        }
    }

    #[test]
    fn a_distraction_at_an_edge_is_cropped_away_without_losing_the_subject() {
        // The subject left of centre, a strong distraction (40 % of the thresholded mass) 2 cells
        // from the right edge.
        let subject = (60, 44, 96, 84);
        let m = maps(&[subject], &[(182, 40, 190, 88)], 0.4);
        let p = run(&m, &CameraHint::default(), request(0, Ratio::Canvas, 0.0));
        assert!(!p.keep, "{p:?}");
        let r = p.frames[0].rect;
        assert!(
            r.x + r.width <= 182.0 / MW as f64 + 1e-9,
            "the distraction is cut away: {r:?}"
        );
        let s = (
            60.0 / MW as f64,
            44.0 / MH as f64,
            96.0 / MW as f64,
            84.0 / MH as f64,
        );
        assert!(holds_rect(&r, s), "the subject stays: {r:?}");
        assert!(p.frames[0].score > MIN_SCORE);
        // The kept frame is among the others, or overlaps the first too much to be listed.
        assert!(p.frames.len() <= MAX_FRAMES);

        // The same proposal with the AF point placed on purpose needs a score over 0.3.
        let placed = CameraHint {
            point: Some([0.4, 0.5]),
            box_size: None,
            placed: true,
        };
        let q = run(&m, &placed, request(0, Ratio::Canvas, 0.0));
        assert_eq!(q.keep, p.frames[0].score <= MIN_SCORE_PLACED, "{q:?}");
    }

    #[test]
    fn a_subject_the_centred_frame_slices_is_brought_inside_and_one_inside_stays() {
        // The kept square of a landscape canvas spans cells 32 to 160 across. A subject across its
        // left edge: the square slides left to hold it whole.
        let m = maps(&[(24, 36, 40, 50)], &[], 0.0);
        let p = run(
            &m,
            &CameraHint::default(),
            request(0, Ratio::Fixed(1.0), 0.0),
        );
        assert!(!p.keep, "{p:?}");
        let s = (
            24.0 / MW as f64,
            36.0 / MH as f64,
            40.0 / MW as f64,
            50.0 / MH as f64,
        );
        assert!(holds_rect(&p.frames[0].rect, s), "{p:?}");
        // A subject well inside the centred square, if off its centre, leaves it where the
        // photographer put it.
        let m = maps(&[(42, 36, 54, 50)], &[], 0.0);
        let p = run(
            &m,
            &CameraHint::default(),
            request(0, Ratio::Fixed(1.0), 0.0),
        );
        assert!(p.keep, "{p:?}");
        // With Original at 0° the whole photo is kept: nothing is sliced or left out.
        let p = run(&m, &CameraHint::default(), request(0, Ratio::Canvas, 0.0));
        assert!(p.keep);
    }

    #[test]
    fn a_gain_moves_the_frame_only_when_it_is_large() {
        // A subject in the centred square, and a part of it outside the square's left edge (cells
        // 20 to 28, the square spans 32 to 160). A small part (6 % of the mass) is not worth moving
        // the photographer's framing for, though holding it would gain mass.
        let square = request(0, Ratio::Fixed(1.0), 0.0);
        let m = maps(&[(80, 48, 112, 80), (20, 60, 28, 68)], &[], 0.0);
        let p = run(&m, &CameraHint::default(), square);
        assert!(p.keep, "{p:?}");
        // A large one (27 %) is: the square slides left to hold it.
        let m = maps(&[(80, 48, 112, 80), (20, 40, 28, 88)], &[], 0.0);
        let p = run(&m, &CameraHint::default(), square);
        assert!(!p.keep, "{p:?}");
        let part = (
            20.0 / MW as f64,
            40.0 / MH as f64,
            28.0 / MW as f64,
            88.0 / MH as f64,
        );
        assert!(holds_rect(&p.frames[0].rect, part), "{p:?}");
    }

    #[test]
    fn a_frame_keeps_the_horizon_where_the_photographer_put_it() {
        let none = CameraHint::default();
        let at = |m: &SubjectMaps, horizon, r: &FrameRequest| {
            propose(MW * 4, MH * 4, m, &none, horizon, r)
        };
        // A subject in the middle and a strip of it (20 % of the mass) across the kept 16:9
        // frame's bottom edge (the frame spans cells 10 to 118 down): without a horizon the frame
        // slides down to hold the strip.
        let m = maps(&[(66, 44, 126, 84), (60, 112, 110, 124)], &[], 0.0);
        let wide = request(0, Ratio::Fixed(16.0 / 9.0), 0.0);
        let free = at(&m, None, &wide);
        assert!(!free.keep && free.frames[0].rect.y > 0.1, "{free:?}");
        // With the horizon across the middle of the photo, sliding down would lift it in the frame
        // from its middle to 40 % of its height: the frame stays.
        assert!(at(&m, Some(0.5), &wide).keep);
        // A cost, not a lock: a strip of 35 % still moves it.
        let heavy = maps(&[(66, 44, 126, 84), (60, 112, 170, 124)], &[], 0.0);
        assert!(!at(&heavy, Some(0.5), &wide).keep);
        // A quarter turn stands the horizon up, and it holds a frame that would slide across: the
        // strip at the picture's top is at the canvas's right after one clockwise turn.
        let m = maps(&[(66, 44, 126, 84), (60, 4, 110, 16)], &[], 0.0);
        let tall = request(1, Ratio::Fixed(9.0 / 16.0), 0.0);
        let free = at(&m, None, &tall);
        assert!(!free.keep && free.frames[0].rect.x > 0.1, "{free:?}");
        assert!(at(&m, Some(0.5), &tall).keep);
        // A frame that leaves the horizon where it is pays nothing: Square slides sideways the
        // same with it as without.
        let m = maps(&[(24, 36, 40, 50)], &[], 0.0);
        let square = request(0, Ratio::Fixed(1.0), 0.0);
        let (with, without) = (at(&m, Some(0.5), &square), at(&m, None, &square));
        assert!(!with.keep);
        assert_eq!(with.frames[0], without.frames[0]);
    }

    #[test]
    fn the_camera_box_is_never_cut() {
        // A subject across the kept square's left edge pulls the square left ...
        let m = maps(&[(24, 36, 40, 50)], &[], 0.0);
        // The AF box in the empty field to the right, inside the kept square.
        let hint = CameraHint {
            point: Some([0.79, 0.5]),
            box_size: Some([0.04, 0.06]),
            placed: false,
        };
        // ... which would leave out the camera's box in the empty field to the right.
        let grown = (0.79 - 0.024, 0.5 - 0.036, 0.79 + 0.024, 0.5 + 0.036);
        let free = run(
            &m,
            &CameraHint::default(),
            request(0, Ratio::Fixed(1.0), 0.0),
        );
        assert!(
            !holds_rect(&free.frames[0].rect, grown),
            "without the box the frame leaves it out"
        );
        let p = run(&m, &hint, request(0, Ratio::Fixed(1.0), 0.0));
        for f in &p.frames {
            assert!(holds_rect(&f.rect, grown), "{f:?} cuts the box");
        }
        for ratio in [Ratio::Canvas, Ratio::Fixed(0.8), Ratio::Fixed(16.0 / 9.0)] {
            for angle in [0.0, -3.0] {
                let p = run(&m, &hint, request(0, ratio, angle));
                for f in &p.frames {
                    assert!(holds_rect(&f.rect, grown), "{ratio:?} {angle}°: {f:?}");
                }
            }
        }
    }

    #[test]
    fn quarter_turns_follow_geometry() {
        // A subject near the picture's left edge, vertically centred.
        let m = maps(&[(4, 56, 20, 72)], &[], 0.0);
        let square = Ratio::Fixed(1.0);
        // One clockwise turn: the picture's left edge is the canvas's top edge.
        let p = run(&m, &CameraHint::default(), request(1, square, 0.0));
        assert!(!p.keep && p.frames[0].rect.y < 0.05, "{p:?}");
        // Three turns: the bottom edge.
        let p = run(&m, &CameraHint::default(), request(3, square, 0.0));
        assert!(
            !p.keep && p.frames[0].rect.y + p.frames[0].rect.height > 0.95,
            "{p:?}"
        );
        // Two turns: the right edge.
        let p = run(&m, &CameraHint::default(), request(2, square, 0.0));
        assert!(
            !p.keep && p.frames[0].rect.x + p.frames[0].rect.width > 0.95,
            "{p:?}"
        );
        // None: the left edge.
        let p = run(&m, &CameraHint::default(), request(0, square, 0.0));
        assert!(!p.keep && p.frames[0].rect.x < 0.05, "{p:?}");
    }

    #[test]
    fn frames_are_inside_the_canvas_and_the_turned_photo_with_the_ratio() {
        let m = maps(
            &[(30, 20, 60, 60), (120, 70, 150, 110)],
            &[(170, 10, 185, 30)],
            0.2,
        );
        let hint = CameraHint::default();
        for turns in 0..4u8 {
            let (cw, ch) = if turns % 2 == 1 {
                (512.0, 768.0)
            } else {
                (768.0, 512.0)
            };
            for ratio in [
                Ratio::Canvas,
                Ratio::Fixed(1.0),
                Ratio::Fixed(0.8),
                Ratio::Fixed(1.25),
                Ratio::Fixed(16.0 / 9.0),
                Ratio::Fixed(9.0 / 16.0),
            ] {
                let r = match ratio {
                    Ratio::Canvas => cw / ch,
                    Ratio::Fixed(r) => r,
                };
                for angle in [0.0, 0.7, -3.3, 8.0, -10.0] {
                    let p = run(&m, &hint, request(turns, ratio, angle));
                    assert!(!p.frames.is_empty() && p.frames.len() <= MAX_FRAMES);
                    let (s, c) = angle.to_radians().sin_cos();
                    for f in &p.frames {
                        let q = f.rect;
                        for v in [q.x, q.y, q.width, q.height, q.x + q.width, q.y + q.height] {
                            assert!((-1e-12..=1.0 + 1e-9).contains(&v), "{q:?}");
                        }
                        // The ratio in canvas pixels, within the rounding.
                        let got = q.width * cw / (q.height * ch);
                        assert!(
                            (got / r - 1.0).abs() < 1e-4,
                            "turns {turns} {ratio:?} {angle}°: {got} for {r}"
                        );
                        // Every corner samples the photo.
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
                        // Rounded to 6 and 4 decimals.
                        for v in [q.x, q.y, q.width, q.height] {
                            assert!(((v * 1e6).round() - v * 1e6).abs() < 1e-6);
                        }
                        assert!(((f.score * 1e4).round() - f.score * 1e4).abs() < 1e-6);
                    }
                    // No two frames overlap by more than 0.6.
                    for (i, a) in p.frames.iter().enumerate() {
                        for b in &p.frames[i + 1..] {
                            let ar = |r: &Rect| Area {
                                x0: r.x,
                                y0: r.y,
                                x1: r.x + r.width,
                                y1: r.y + r.height,
                            };
                            assert!(
                                ar(&a.rect).intersection_over_union(&ar(&b.rect))
                                    <= MAX_OVERLAP + 1e-4
                            );
                        }
                    }
                }
            }
        }
    }
}
