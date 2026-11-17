//! Teeth by SAM 2.1 prompted from the face mesh (`docs/27` 5.6, S-17): everything about the teeth that needs no network.
//!
//! The segmenter ([`TeethSegmenter`](crate::TeethSegmenter)) is run by `ym-core`: it encodes a **crop around the mouth**
//! ([`MouthCrop`]) with SAM 2.1 small, prompts it with what the mesh knows ([`teeth_prompt`]: a box around the opening of
//! the mouth, positive points inside it, negative points on the lips), and chooses one of the four masks SAM answers with
//! ([`select_candidate`]). Nothing here looks at a colour: where the teeth are is the network's answer, and the mesh only
//! says where to look. The pieces are plain functions of the 478 landmarks and of the masks' numbers, so that the rules are
//! tested on synthetic answers, with no weights.
//!
//! Coordinates: the crop is a square of the picture; a point of the prompt is a fraction of its side, `0..1` from the
//! top-left corner (what `ym_ml::sam2::Prompt` takes). The masks SAM answers are `256 x 256` logits over the same square
//! ([`GRID_PX`]).

use crate::error::Result;
use crate::face_parts::FaceGeometry;
use crate::face_topology::{INNER_LIPS, OUTER_LIPS};
use crate::types::{BoundedMask, Mask, Rect};

/// The side of the grid the decoder's logits are on.
pub const GRID_PX: usize = 256;

/// The crop's side as a multiple of the mouth's width (corner to corner of the outer lips). Measured on public portraits
/// (`docs/27` S-17 as built): wide enough for SAM to see the lips and the cheeks around the teeth, narrow enough that the
/// teeth are large in SAM's 1024 px.
pub const CROP_FACTOR: f64 = 3.5;
/// The box of the prompt is the opening's bounding box grown by this share of the opening's width on every side.
pub const BOX_PAD: f64 = 0.04;
/// An opening whose height at a column is under this share of its width gets one positive point in the middle of the
/// column in place of two rows.
pub const THIN_OPENING: f64 = 0.06;
/// An opening taller than this share of its width gets a negative point in the cavity ([`PromptShape::cavity`]).
pub const CAVITY_OPENING: f64 = 0.25;

/// Where the points of a prompt sit; [`PromptShape::DEFAULT`] is the one `docs/27` 5.6 fixes, the others were compared by S-17.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromptShape {
    /// A positive point of the upper row sits this share of the column's height below the upper lip's inner edge (where the
    /// upper teeth show in a smile).
    pub upper_row: f64,
    /// A positive point of the lower row sits this share of the column's height above the lower lip (the lower teeth); `None`
    /// for no lower row.
    pub lower_row: Option<f64>,
    /// The columns of positive points, as shares of the width between the corners of the opening.
    pub columns: &'static [f64],
    /// No point of a row is farther from its lip than this share of the opening's width (a tooth is as tall as a share of the
    /// mouth, whatever the jaw does: in a wide-open mouth a share of the column's height would land on the tongue).
    pub row_cap: f64,
    /// A negative point in the middle of the opening, this share of its height down, when it is taller than
    /// [`CAVITY_OPENING`] of its width (the tongue and the dark behind the teeth); `None` for none.
    pub cavity: Option<f64>,
    /// Negative points on the band just under the upper lip's inner contour, this share of the opening's width below it, in
    /// the middle columns (S-18: the gum shows there in a smile); `None` for none.
    pub band_negatives: Option<f64>,
}

impl PromptShape {
    /// The shape of the spec.
    pub const DEFAULT: Self = Self {
        upper_row: 0.22,
        lower_row: None,
        columns: &[0.2, 0.35, 0.5, 0.65, 0.8],
        row_cap: 0.05,
        cavity: None,
        band_negatives: None,
    };
}

/// An opening lower than this many pixels of the picture shows no teeth to SAM: the face gets none (the mesh puts the lips
/// together to within a pixel or two on a closed mouth, and a mouth that small is not worth 30 s of the encoder).
pub const MIN_OPENING_PX: f64 = 3.0;
/// An opening whose height is under this share of its width shows no teeth either.
pub const MIN_OPENING_ASPECT: f64 = 0.03;

/// A candidate is refused when more than this share of it lies on the lips or outside the mouth's outer contour.
pub const MAX_SPILL: f64 = 0.30;
/// A candidate is refused when it covers less than this share of the opening.
pub const MIN_COVER: f64 = 0.04;
/// The weight of a candidate's cover of the opening in its score.
pub const COVER_WEIGHT: f64 = 0.3;
/// The weight of a candidate's spill in its score.
pub const SPILL_WEIGHT: f64 = 1.0;
/// A candidate that covers more than this share of an opening taller than [`CAVITY_ASPECT`] of its width is the whole
/// cavity (teeth, tongue and the dark behind them), not the teeth.
pub const CAVITY_COVER: f64 = 0.97;
/// The height over the width of the opening above which [`CAVITY_COVER`] applies.
pub const CAVITY_ASPECT: f64 = 0.30;

/// A square of the picture around the mouth that SAM looks at, in picture pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouthCrop {
    /// Left edge, continuous picture coordinates (may be negative).
    pub x_px: f64,
    /// Top edge.
    pub y_px: f64,
    /// Side.
    pub side_px: f64,
}

impl MouthCrop {
    /// The square of `factor` times the mouth's width centred on the outer lips; `None` for a mouth narrower than 4 px.
    pub fn from_geometry(geometry: &FaceGeometry, factor: f64) -> Option<Self> {
        let p = geometry.points();
        let (a, b) = (
            p[usize::from(OUTER_LIPS[0])],
            p[usize::from(OUTER_LIPS[10])],
        );
        let width = (a[0] - b[0]).hypot(a[1] - b[1]);
        if width.is_nan() || width < 4.0 {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &i in &OUTER_LIPS {
            let q = p[usize::from(i)];
            x0 = x0.min(q[0]);
            x1 = x1.max(q[0]);
            y0 = y0.min(q[1]);
            y1 = y1.max(q[1]);
        }
        let side = (factor * width).max(8.0);
        Some(Self {
            x_px: (x0 + x1) / 2.0 - side / 2.0,
            y_px: (y0 + y1) / 2.0 - side / 2.0,
            side_px: side,
        })
    }

    /// A picture point as a fraction of the crop.
    pub fn to_unit(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.x_px) / self.side_px,
            (p[1] - self.y_px) / self.side_px,
        ]
    }
}

/// How the prompt is made (S-17 compared the three).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptStyle {
    /// The opening's box alone.
    Box,
    /// Positive points in the opening and negative points on the lips.
    Points,
    /// Both.
    BoxAndPoints,
}

/// A point of the prompt, as fractions of the crop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromptPoint {
    /// Fraction of the crop's width.
    pub x: f32,
    /// Fraction of the crop's height.
    pub y: f32,
    /// `true` inside the teeth's place, `false` on the lips.
    pub positive: bool,
}

/// The prompt SAM is given: a box and points, as fractions of the crop (`ym_ml::sam2::Prompt` takes the same numbers).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TeethPrompt {
    /// `[x0, y0, x1, y1]`.
    pub bbox: Option<[f32; 4]>,
    /// The points, in order.
    pub points: Vec<PromptPoint>,
}

/// The mouth in its own frame: `ex` runs from the left corner of the opening to the right one, `ey` is down the mouth.
struct Frame {
    origin: [f64; 2],
    ex: [f64; 2],
    ey: [f64; 2],
}

impl Frame {
    fn of(geometry: &FaceGeometry) -> Option<Self> {
        let p = geometry.points();
        let (a, b) = (
            p[usize::from(INNER_LIPS[0])],
            p[usize::from(INNER_LIPS[10])],
        );
        let w = (b[0] - a[0]).hypot(b[1] - a[1]);
        if w.is_nan() || w < 1.0 {
            return None;
        }
        let ex = [(b[0] - a[0]) / w, (b[1] - a[1]) / w];
        Some(Self {
            origin: a,
            ex,
            ey: [-ex[1], ex[0]],
        })
    }

    fn to_mouth(&self, p: [f64; 2]) -> [f64; 2] {
        let d = [p[0] - self.origin[0], p[1] - self.origin[1]];
        [
            d[0] * self.ex[0] + d[1] * self.ex[1],
            d[0] * self.ey[0] + d[1] * self.ey[1],
        ]
    }

    fn to_picture(&self, s: f64, t: f64) -> [f64; 2] {
        [
            self.origin[0] + s * self.ex[0] + t * self.ey[0],
            self.origin[1] + s * self.ex[1] + t * self.ey[1],
        ]
    }
}

/// The span `(top, bottom)` of a polygon (in the mouth's frame) along the vertical line at `s`; `None` when the line misses it.
fn column_span(poly: &[[f64; 2]], s: f64) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a[0] <= s) == (b[0] <= s) {
            continue;
        }
        let t = (s - a[0]) / (b[0] - a[0]);
        let y = a[1] + t * (b[1] - a[1]);
        lo = lo.min(y);
        hi = hi.max(y);
    }
    (lo <= hi).then_some((lo, hi))
}

/// The prompt for a mouth: `None` when the mesh gives no usable opening.
///
/// * **Box**: the opening's (inner lip contour) bounding box in the crop, grown by [`BOX_PAD`] of its width.
/// * **Positive points**: on the columns of the [`PromptShape`] between the corners of the opening, one row `upper_row` of the
///   column's height below the upper lip (where the upper teeth show in a smile), at most `row_cap` of the opening's width
///   from the lip; a column thinner than [`THIN_OPENING`] of the width gets one point in its middle. A lower row and a
///   negative point in the cavity are options that S-17 compared and dropped.
/// * **Negative points**: on the lips, halfway between the inner and the outer contour at the middle of the upper and of the
///   lower lip, and at each corner of the mouth.
pub fn teeth_prompt(
    geometry: &FaceGeometry,
    crop: &MouthCrop,
    style: PromptStyle,
) -> Option<TeethPrompt> {
    teeth_prompt_with(geometry, crop, style, &PromptShape::DEFAULT)
}

/// [`teeth_prompt`] with another [`PromptShape`].
pub fn teeth_prompt_with(
    geometry: &FaceGeometry,
    crop: &MouthCrop,
    style: PromptStyle,
    shape: &PromptShape,
) -> Option<TeethPrompt> {
    let frame = Frame::of(geometry)?;
    let p = geometry.points();
    let at = |i: u16| p[usize::from(i)];
    let unit = |q: [f64; 2]| {
        let u = crop.to_unit(q);
        [u[0] as f32, u[1] as f32]
    };
    let inner: Vec<[f64; 2]> = INNER_LIPS.iter().map(|&i| at(i)).collect();
    let mut prompt = TeethPrompt::default();
    if style != PromptStyle::Points {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for q in &inner {
            x0 = x0.min(q[0]);
            x1 = x1.max(q[0]);
            y0 = y0.min(q[1]);
            y1 = y1.max(q[1]);
        }
        let pad = BOX_PAD * (x1 - x0);
        let a = unit([x0 - pad, y0 - pad]);
        let b = unit([x1 + pad, y1 + pad]);
        prompt.bbox = Some([a[0], a[1], b[0], b[1]]);
    }
    if style != PromptStyle::Box {
        let local: Vec<[f64; 2]> = inner.iter().map(|&q| frame.to_mouth(q)).collect();
        let width = local[10][0] - local[0][0];
        for &f in shape.columns {
            let s = local[0][0] + f * width;
            let Some((top, bottom)) = column_span(&local, s) else {
                continue;
            };
            let gap = bottom - top;
            let cap = shape.row_cap * width;
            let mut ts: Vec<f64> = Vec::new();
            if gap < THIN_OPENING * width {
                ts.push(top + 0.5 * gap);
            } else {
                ts.push(top + (shape.upper_row * gap).min(cap));
                if let Some(lower) = shape.lower_row {
                    ts.push(bottom - (lower * gap).min(cap));
                }
            }
            for t in ts {
                let q = unit(frame.to_picture(s, t));
                prompt.points.push(PromptPoint {
                    x: q[0],
                    y: q[1],
                    positive: true,
                });
            }
            if let (Some(depth), true) = (
                shape.cavity,
                (f - 0.5).abs() < 1e-9 && gap >= CAVITY_OPENING * width,
            ) {
                let q = unit(frame.to_picture(s, top + depth * gap));
                prompt.points.push(PromptPoint {
                    x: q[0],
                    y: q[1],
                    positive: false,
                });
            }
        }
        if let Some(depth) = shape.band_negatives {
            for &f in &[0.3, 0.5, 0.7] {
                let s = local[0][0] + f * width;
                let Some((top, _)) = column_span(&local, s) else {
                    continue;
                };
                let q = unit(frame.to_picture(s, top + depth * width));
                prompt.points.push(PromptPoint {
                    x: q[0],
                    y: q[1],
                    positive: false,
                });
            }
        }
        // Negative: the middle of the upper and of the lower lip, the two corners.
        let mid = |a: u16, b: u16| {
            let (a, b) = (at(a), at(b));
            [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
        };
        for (o, i) in [(5usize, 5usize), (15, 15), (0, 0), (10, 10)] {
            let q = unit(mid(OUTER_LIPS[o], INNER_LIPS[i]));
            prompt.points.push(PromptPoint {
                x: q[0],
                y: q[1],
                positive: false,
            });
        }
    }
    let finite = prompt
        .points
        .iter()
        .all(|q| q.x.is_finite() && q.y.is_finite())
        && prompt.bbox.is_none_or(|b| b.iter().all(|v| v.is_finite()));
    (finite && (prompt.bbox.is_some() || prompt.points.iter().any(|q| q.positive)))
        .then_some(prompt)
}

/// Where the points of the gum query sit (`docs/27` 5.6, S-18).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumShape {
    /// The columns, as shares of the width between the corners of the opening.
    pub columns: &'static [f64],
    /// A positive point sits this share of the opening's width below the upper lip's inner contour (the band the gum shows in).
    pub depth: f64,
    /// A negative point (the teeth) sits this share of the way down the first answer's column, from its top to its bottom.
    pub teeth_at: f64,
}

impl GumShape {
    /// The shape the spec fixes.
    pub const DEFAULT: Self = Self {
        columns: &[0.3, 0.5, 0.7],
        depth: 0.015,
        teeth_at: 0.6,
    };
}

/// The prompt of the second query, the one that asks SAM for the gum: a positive point on the band just under the upper lip's
/// inner contour, negative points on the teeth the first answer (`teeth_logits`, `GRID_PX` squared) found, and the lips as
/// negatives as before. `None` when no column of the first answer has a positive point above its teeth point.
pub fn gum_prompt(
    geometry: &FaceGeometry,
    crop: &MouthCrop,
    teeth_logits: &[f32],
    shape: &GumShape,
) -> Option<TeethPrompt> {
    if teeth_logits.len() != GRID_PX * GRID_PX {
        return None;
    }
    let frame = Frame::of(geometry)?;
    let p = geometry.points();
    let at = |i: u16| p[usize::from(i)];
    let unit = |q: [f64; 2]| {
        let u = crop.to_unit(q);
        [u[0] as f32, u[1] as f32]
    };
    let local: Vec<[f64; 2]> = INNER_LIPS.iter().map(|&i| frame.to_mouth(at(i))).collect();
    let width = local[10][0] - local[0][0];
    let cell = crop.side_px / GRID_PX as f64;
    let inside = |s: f64, t: f64| {
        let u = crop.to_unit(frame.to_picture(s, t));
        let (x, y) = (u[0] * GRID_PX as f64, u[1] * GRID_PX as f64);
        x >= 0.0
            && y >= 0.0
            && x < GRID_PX as f64
            && y < GRID_PX as f64
            && teeth_logits[y as usize * GRID_PX + x as usize] > 0.0
    };
    let mut prompt = TeethPrompt::default();
    for &f in shape.columns {
        let s = local[0][0] + f * width;
        let Some((top, bottom)) = column_span(&local, s) else {
            continue;
        };
        let (mut first, mut last) = (f64::MAX, f64::MIN);
        let mut t = top;
        while t <= bottom {
            if inside(s, t) {
                first = first.min(t);
                last = last.max(t);
            }
            t += cell;
        }
        if first > last {
            continue;
        }
        let gum = top + shape.depth * width;
        let teeth = first + shape.teeth_at * (last - first);
        if gum >= teeth {
            continue;
        }
        for (t, positive) in [(gum, true), (teeth, false)] {
            let q = unit(frame.to_picture(s, t));
            prompt.points.push(PromptPoint {
                x: q[0],
                y: q[1],
                positive,
            });
        }
    }
    if prompt.points.is_empty() {
        return None;
    }
    for (o, i) in [(5usize, 5usize), (15, 15), (0, 0), (10, 10)] {
        let (a, b) = (at(OUTER_LIPS[o]), at(INNER_LIPS[i]));
        let q = unit([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
        prompt.points.push(PromptPoint {
            x: q[0],
            y: q[1],
            positive: false,
        });
    }
    prompt
        .points
        .iter()
        .all(|q| q.x.is_finite() && q.y.is_finite())
        .then_some(prompt)
}

/// The mouth's regions on SAM's grid: where the opening is, and where the lips are.
#[derive(Debug, Clone, PartialEq)]
pub struct Regions {
    opening: Vec<bool>,
    lips: Vec<bool>,
    inside_outer: Vec<bool>,
    opening_area: usize,
    /// The opening's height over its width.
    pub aspect: f64,
}

/// Fills a polygon (fractions of the crop) on the grid, even-odd at the cells' centres.
fn fill(poly: &[[f64; 2]]) -> Vec<bool> {
    let mut out = vec![false; GRID_PX * GRID_PX];
    for (y, row) in out.chunks_mut(GRID_PX).enumerate() {
        let v = (y as f64 + 0.5) / GRID_PX as f64;
        // Crossings of the row's line, sorted.
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a[1] <= v) != (b[1] <= v) {
                xs.push(a[0] + (v - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
            }
        }
        xs.sort_unstable_by(f64::total_cmp);
        for pair in xs.chunks_exact(2) {
            let from = (pair[0] * GRID_PX as f64 - 0.5).ceil().max(0.0) as usize;
            let to = ((pair[1] * GRID_PX as f64 - 0.5).floor() + 1.0).clamp(0.0, GRID_PX as f64)
                as usize;
            for cell in row.iter_mut().take(to).skip(from) {
                *cell = true;
            }
        }
    }
    out
}

impl Regions {
    /// Whether grid cell `i` (row-major) is in the opening.
    pub fn opening_at(&self, i: usize) -> bool {
        self.opening.get(i).copied().unwrap_or(false)
    }

    /// The regions of a mouth on the crop's grid: the opening (inner lip contour), the lips (between the contours) and the
    /// mouth (inside the outer contour). `None` when the opening covers no cell or is lower than [`MIN_OPENING_PX`] or
    /// [`MIN_OPENING_ASPECT`] of its width.
    pub fn from_geometry(geometry: &FaceGeometry, crop: &MouthCrop) -> Option<Self> {
        let p = geometry.points();
        let ring = |ids: &[u16]| -> Vec<[f64; 2]> {
            ids.iter()
                .map(|&i| crop.to_unit(p[usize::from(i)]))
                .collect()
        };
        let opening = fill(&ring(&INNER_LIPS));
        let inside_outer = fill(&ring(&OUTER_LIPS));
        let lips: Vec<bool> = inside_outer
            .iter()
            .zip(&opening)
            .map(|(&o, &i)| o && !i)
            .collect();
        let opening_area = opening.iter().filter(|&&v| v).count();
        if opening_area == 0 {
            return None;
        }
        let frame = Frame::of(geometry)?;
        let local: Vec<[f64; 2]> = INNER_LIPS
            .iter()
            .map(|&i| frame.to_mouth(p[usize::from(i)]))
            .collect();
        let width = (local[10][0] - local[0][0]).max(1e-9);
        let (lo, hi) = local.iter().fold((f64::MAX, f64::MIN), |(lo, hi), q| {
            (lo.min(q[1]), hi.max(q[1]))
        });
        if hi - lo < MIN_OPENING_PX || (hi - lo) / width < MIN_OPENING_ASPECT {
            return None;
        }
        Some(Self {
            opening,
            lips,
            inside_outer,
            opening_area,
            aspect: (hi - lo) / width,
        })
    }
}

/// What one candidate mask (logit above 0) does on the mouth's regions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateStats {
    /// SAM's predicted IoU.
    pub iou: f32,
    /// Cells of the mask.
    pub area: usize,
    /// The share of the opening it covers.
    pub cover: f64,
    /// The share of the mask on the lips or outside the mouth's outer contour.
    pub spill: f64,
    /// `iou + COVER_WEIGHT * cover - SPILL_WEIGHT * spill`.
    pub score: f64,
}

/// Measures a candidate's `logits` (`GRID_PX` squared) against the regions.
pub fn candidate_stats(logits: &[f32], iou: f32, regions: &Regions) -> CandidateStats {
    let (mut area, mut in_opening, mut off) = (0usize, 0usize, 0usize);
    for (i, &v) in logits.iter().enumerate().take(GRID_PX * GRID_PX) {
        if v > 0.0 {
            area += 1;
            if regions.opening[i] {
                in_opening += 1;
            } else if regions.lips[i] || !regions.inside_outer[i] {
                off += 1;
            }
        }
    }
    let cover = in_opening as f64 / regions.opening_area as f64;
    let spill = if area == 0 {
        0.0
    } else {
        off as f64 / area as f64
    };
    CandidateStats {
        iou,
        area,
        cover,
        spill,
        score: f64::from(iou) + COVER_WEIGHT * cover - SPILL_WEIGHT * spill,
    }
}

/// How far a candidate's top edge stays below the opening's top edge (the upper lip's inner contour), as a share of the
/// opening's width, averaged over the columns where both exist and capped at [`LIP_GAP_CAP`] per column. A mask that runs up to
/// the lip across the whole opening (teeth and gum together, or teeth that reach the lip) has 0; a mask that stops at the teeth's
/// own edge under a visible band of gum has the band's height (S-18).
pub fn lip_gap(logits: &[f32], regions: &Regions) -> f64 {
    let width = {
        let (mut x0, mut x1) = (GRID_PX, 0usize);
        for (i, &o) in regions.opening.iter().enumerate() {
            if o {
                x0 = x0.min(i % GRID_PX);
                x1 = x1.max(i % GRID_PX);
            }
        }
        (x1.saturating_sub(x0) + 1) as f64
    };
    let (mut sum, mut cols) = (0.0, 0usize);
    for x in 0..GRID_PX {
        let mut open_top = None;
        let mut mask_top = None;
        for y in 0..GRID_PX {
            let i = y * GRID_PX + x;
            if regions.opening[i] {
                open_top.get_or_insert(y);
                if logits[i] > 0.0 {
                    mask_top = Some(y);
                    break;
                }
            }
        }
        if let (Some(o), Some(m)) = (open_top, mask_top) {
            sum += ((m - o) as f64 / width).min(LIP_GAP_CAP);
            cols += 1;
        }
    }
    if cols == 0 { 0.0 } else { sum / cols as f64 }
}

/// The most of the opening's width one column of [`lip_gap`] counts.
pub const LIP_GAP_CAP: f64 = 0.10;

/// Why no candidate was used. The `Display` is the sentence of the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// SAM answered no mask at all, or only masks that cover under [`MIN_COVER`] of the opening.
    Nothing,
    /// Every candidate with a mask spilled over more than [`MAX_SPILL`] onto the lips and the face.
    SpillsOnTheLips,
    /// Every candidate with a mask is the whole cavity of an open mouth.
    WholeCavity,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nothing => "SAM found no teeth in the opening of the mouth",
            Self::SpillsOnTheLips => "every mask SAM made spills onto the lips",
            Self::WholeCavity => "every mask SAM made is the whole opening of the mouth",
        })
    }
}

/// Chooses among the masks SAM answered: `candidates` are `(logits, predicted IoU)`. A candidate is **refused** when it
/// covers under [`MIN_COVER`] of the opening, when more than [`MAX_SPILL`] of it lies off the opening (on the lips or
/// outside the mouth), or when it covers more than [`CAVITY_COVER`] of an opening taller than [`CAVITY_ASPECT`] of its
/// width. Of the rest the one of the highest `score` (predicted IoU, plus [`COVER_WEIGHT`] of the cover, minus
/// [`SPILL_WEIGHT`] of the spill) wins; a tie goes to the lower index. The stats of every candidate come with the answer.
pub fn select_candidate(
    candidates: &[(&[f32], f32)],
    regions: &Regions,
) -> (std::result::Result<usize, Refusal>, Vec<CandidateStats>) {
    let stats: Vec<CandidateStats> = candidates
        .iter()
        .map(|(logits, iou)| candidate_stats(logits, *iou, regions))
        .collect();
    let mut best: Option<usize> = None;
    let (mut small, mut spills, mut cavity) = (0, 0, 0);
    for (i, s) in stats.iter().enumerate() {
        if s.area == 0 || s.cover < MIN_COVER || !s.score.is_finite() {
            small += 1;
        } else if s.spill > MAX_SPILL {
            spills += 1;
        } else if s.cover > CAVITY_COVER && regions.aspect > CAVITY_ASPECT {
            cavity += 1;
        } else if best.is_none_or(|b| s.score > stats[b].score) {
            best = Some(i);
        }
    }
    let verdict = best.ok_or(if spills > 0 && spills >= cavity {
        Refusal::SpillsOnTheLips
    } else if cavity > 0 {
        Refusal::WholeCavity
    } else {
        let _ = small;
        Refusal::Nothing
    });
    (verdict, stats)
}

/// The mask of a candidate on the picture: `sigmoid(logit)` of the `GRID_PX` squared logits sampled bilinearly (the logit
/// before the sigmoid, so the edge stays sharp) at every pixel of the crop that lies in the picture, as 0 to 255. `None`
/// when the crop is outside the picture or the mask is empty.
///
/// # Errors
///
/// [`crate::RetouchError`] from the mask constructors (never for a valid size).
pub fn crop_mask(
    logits: &[f32],
    crop: &MouthCrop,
    size_px: (u32, u32),
) -> Result<Option<BoundedMask>> {
    if logits.len() != GRID_PX * GRID_PX {
        return Ok(None);
    }
    let x0 = crop.x_px.floor().max(0.0) as i64;
    let y0 = crop.y_px.floor().max(0.0) as i64;
    let x1 = ((crop.x_px + crop.side_px).ceil() as i64).min(i64::from(size_px.0));
    let y1 = ((crop.y_px + crop.side_px).ceil() as i64).min(i64::from(size_px.1));
    if x1 <= x0 || y1 <= y0 {
        return Ok(None);
    }
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let n = GRID_PX as f64;
    let mut data = vec![0u8; w * h];
    let mut any = false;
    for (j, row) in data.chunks_mut(w).enumerate() {
        let gy =
            ((y0 as f64 + j as f64 + 0.5 - crop.y_px) / crop.side_px * n - 0.5).clamp(0.0, n - 1.0);
        let (iy, fy) = (gy.floor() as usize, gy.fract());
        let iy1 = (iy + 1).min(GRID_PX - 1);
        for (i, out) in row.iter_mut().enumerate() {
            let gx = ((x0 as f64 + i as f64 + 0.5 - crop.x_px) / crop.side_px * n - 0.5)
                .clamp(0.0, n - 1.0);
            let (ix, fx) = (gx.floor() as usize, gx.fract());
            let ix1 = (ix + 1).min(GRID_PX - 1);
            let l = |yy: usize, xx: usize| f64::from(logits[yy * GRID_PX + xx]);
            let top = l(iy, ix) * (1.0 - fx) + l(iy, ix1) * fx;
            let bottom = l(iy1, ix) * (1.0 - fx) + l(iy1, ix1) * fx;
            let v = top * (1.0 - fy) + bottom * fy;
            let p = 1.0 / (1.0 + (-v).exp());
            let b = (255.0 * p).round() as u8;
            any |= b > 0;
            *out = b;
        }
    }
    if !any {
        return Ok(None);
    }
    let region = Rect::new(x0 as i32, y0 as i32, w as u32, h as u32);
    Ok(Some(BoundedMask::new(
        region,
        Mask::new(w as u32, h as u32, data)?,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mouth of 100 px across in a 400 px picture: the opening is an ellipse of 80 x 24 px, the outer lips one of 100 x 44.
    pub(crate) fn mouth(open_h: f64) -> FaceGeometry {
        let mut pts = vec![[0.0, 0.0]; crate::face_parts::LANDMARK_COUNT];
        let (cx, cy) = (200.0, 250.0);
        // Chin and forehead so that `up` and `height_px` exist.
        pts[10] = [200.0, 50.0];
        pts[152] = [200.0, 380.0];
        let ellipse = |ids: &[u16], rx: f64, ry: f64, pts: &mut Vec<[f64; 2]>| {
            // The 20 landmarks go clockwise from the left corner: 10 along the bottom, 10 along the top.
            for (k, &id) in ids.iter().enumerate() {
                let a = std::f64::consts::PI - std::f64::consts::TAU * k as f64 / 20.0;
                // k = 0 is the left corner (angle pi), k = 10 the right one (angle 0).
                let x = cx + rx * a.cos();
                let y = cy + ry * a.sin();
                pts[usize::from(id)] = [x, y];
            }
        };
        ellipse(&OUTER_LIPS, 50.0, 22.0, &mut pts);
        ellipse(&INNER_LIPS, 40.0, open_h / 2.0, &mut pts);
        FaceGeometry::new(pts).expect("geometry")
    }

    #[test]
    fn the_crop_is_centred_on_the_mouth_and_is_a_multiple_of_its_width() {
        let g = mouth(24.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        // The corners of the outer lips are 100 px apart.
        assert!(
            (crop.side_px - 100.0 * CROP_FACTOR).abs() < 1e-9,
            "{crop:?}"
        );
        assert!((crop.x_px - (200.0 - crop.side_px / 2.0)).abs() < 1e-9);
        assert!((crop.y_px - (250.0 - crop.side_px / 2.0)).abs() < 3.0);
        let u = crop.to_unit([200.0, 250.0]);
        assert!((u[0] - 0.5).abs() < 1e-9 && (u[1] - 0.5).abs() < 0.03);
    }

    #[test]
    fn a_box_prompt_is_the_padded_opening() {
        let g = mouth(24.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let p = teeth_prompt(&g, &crop, PromptStyle::Box).expect("prompt");
        assert!(p.points.is_empty());
        let [x0, y0, x1, y1] = p.bbox.expect("box");
        // The opening is 80 px across, 24 px tall; padded by 3.2 px on each side.
        assert!(
            (f64::from(x1 - x0) * crop.side_px - 86.4).abs() < 0.5,
            "{x0} {x1}"
        );
        assert!(
            (f64::from(y1 - y0) * crop.side_px - 30.4).abs() < 0.5,
            "{y0} {y1}"
        );
    }

    fn cell(q: &PromptPoint) -> usize {
        let (x, y) = (
            (f64::from(q.x) * GRID_PX as f64) as usize,
            (f64::from(q.y) * GRID_PX as f64) as usize,
        );
        y * GRID_PX + x
    }

    #[test]
    fn point_prompts_put_positives_in_the_opening_and_negatives_on_the_lips() {
        let g = mouth(24.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let regions = Regions::from_geometry(&g, &crop).expect("regions");
        let p = teeth_prompt(&g, &crop, PromptStyle::Points).expect("prompt");
        assert!(p.bbox.is_none());
        let pos: Vec<_> = p.points.iter().filter(|q| q.positive).collect();
        let neg: Vec<_> = p.points.iter().filter(|q| !q.positive).collect();
        assert_eq!(
            (pos.len(), neg.len()),
            (5, 4),
            "one upper row of five columns, and four lip points"
        );
        for q in &pos {
            assert!(regions.opening[cell(q)], "{q:?} is not in the opening");
        }
        for q in &neg {
            assert!(regions.lips[cell(q)], "{q:?} is not on the lips");
        }
        // The upper row hangs from the upper lip: under it by `upper_row` of the column's height, but by at most `row_cap` of the
        // opening's width (80 px across, so 4 px of the 24 px of height here).
        let side = crop.side_px as f32;
        let tops: Vec<f32> = pos.iter().map(|q| q.y * side).collect();
        let mid = tops[2];
        assert!(
            (mid - (250.0 - crop.y_px as f32 - 12.0 + 4.0)).abs() < 0.6,
            "{mid}"
        );
        // Both together.
        let both = teeth_prompt(&g, &crop, PromptStyle::BoxAndPoints).expect("prompt");
        assert!(both.bbox.is_some() && both.points.len() == 9);
    }

    #[test]
    fn a_lower_row_and_a_cavity_point_are_options() {
        let g = mouth(40.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let regions = Regions::from_geometry(&g, &crop).expect("regions");
        let shape = PromptShape {
            lower_row: Some(0.18),
            cavity: Some(0.6),
            ..PromptShape::DEFAULT
        };
        let p = teeth_prompt_with(&g, &crop, PromptStyle::Points, &shape).expect("prompt");
        let pos: Vec<_> = p.points.iter().filter(|q| q.positive).collect();
        assert_eq!(pos.len(), 10);
        // The upper point and the lower one of a column: both in the opening, the upper one above the lower one.
        assert!(pos.chunks(2).all(|c| c[0].y < c[1].y));
        assert!(pos.iter().all(|q| regions.opening[cell(q)]));
        // The cavity point is a negative point inside the opening (4 lip points and 1 more).
        let neg: Vec<_> = p.points.iter().filter(|q| !q.positive).collect();
        assert_eq!(neg.len(), 5);
        assert_eq!(neg.iter().filter(|q| regions.opening[cell(q)]).count(), 1);
        // A mouth that is not tall has none.
        let g = mouth(10.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let p = teeth_prompt_with(&g, &crop, PromptStyle::Points, &shape).expect("prompt");
        assert_eq!(p.points.iter().filter(|q| !q.positive).count(), 4);
    }

    #[test]
    fn a_mouth_that_is_barely_open_has_nothing_to_look_at() {
        let crop_of = |g: &FaceGeometry| MouthCrop::from_geometry(g, CROP_FACTOR).expect("crop");
        for h in [0.5, 2.0, 2.9] {
            let g = mouth(h);
            assert!(
                Regions::from_geometry(&g, &crop_of(&g)).is_none(),
                "an opening of {h} px"
            );
        }
        let g = mouth(3.5);
        assert!(Regions::from_geometry(&g, &crop_of(&g)).is_some());
        // Tall and narrow is as good as short and wide only when it is as tall as the floor says.
        let mut pts = mouth(24.0).points().to_vec();
        for &i in &INNER_LIPS {
            pts[usize::from(i)][0] = 200.0 + (pts[usize::from(i)][0] - 200.0) * 20.0;
        }
        let wide = FaceGeometry::new(pts).expect("geometry");
        assert!(
            Regions::from_geometry(&wide, &crop_of(&wide)).is_none(),
            "24 px tall and 1600 px wide: under 3 % of the width"
        );
    }

    #[test]
    fn a_thin_opening_gets_one_point_per_column() {
        let g = mouth(4.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let shape = PromptShape {
            lower_row: Some(0.18),
            ..PromptShape::DEFAULT
        };
        let p = teeth_prompt_with(&g, &crop, PromptStyle::Points, &shape).expect("prompt");
        assert_eq!(p.points.iter().filter(|q| q.positive).count(), 5);
    }

    #[test]
    fn a_tilted_mouth_is_prompted_along_its_own_axis() {
        // The same mouth turned by 20 degrees about its centre.
        let g = mouth(24.0);
        let (s, c) = 20f64.to_radians().sin_cos();
        let pts: Vec<[f64; 2]> = g
            .points()
            .iter()
            .map(|p| {
                let (dx, dy) = (p[0] - 200.0, p[1] - 250.0);
                [200.0 + c * dx - s * dy, 250.0 + s * dx + c * dy]
            })
            .collect();
        let g = FaceGeometry::new(pts).expect("geometry");
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let regions = Regions::from_geometry(&g, &crop).expect("regions");
        let p = teeth_prompt(&g, &crop, PromptStyle::Points).expect("prompt");
        for q in p.points.iter().filter(|q| q.positive) {
            let (x, y) = (
                (f64::from(q.x) * GRID_PX as f64) as usize,
                (f64::from(q.y) * GRID_PX as f64) as usize,
            );
            assert!(regions.opening[y * GRID_PX + x], "{q:?}");
        }
    }

    /// Logits that are 8 inside the cells of `inside` and -8 elsewhere.
    fn logits(inside: impl Fn(usize, usize) -> bool) -> Vec<f32> {
        (0..GRID_PX * GRID_PX)
            .map(|i| {
                if inside(i % GRID_PX, i / GRID_PX) {
                    8.0
                } else {
                    -8.0
                }
            })
            .collect()
    }

    fn regions(open_h: f64) -> Regions {
        let g = mouth(open_h);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        Regions::from_geometry(&g, &crop).expect("regions")
    }

    #[test]
    fn the_candidate_that_fills_the_opening_beats_a_sliver_and_a_spill() {
        let r = regions(24.0);
        let opening = logits(|x, y| r.opening[y * GRID_PX + x]);
        let sliver = logits(|x, y| r.opening[y * GRID_PX + x] && x < 128 && y % 7 == 0);
        let spill = logits(|x, y| r.inside_outer[y * GRID_PX + x]);
        let cands = [
            (&sliver[..], 0.95f32),
            (&spill[..], 0.99),
            (&opening[..], 0.90),
        ];
        let (verdict, stats) = select_candidate(&cands, &r);
        assert_eq!(verdict, Ok(2), "{stats:?}");
        assert!(
            stats[1].spill > MAX_SPILL,
            "the spill is on the lips: {:?}",
            stats[1]
        );
        assert!(stats[2].cover > 0.99 && stats[2].spill == 0.0);
    }

    #[test]
    fn among_good_candidates_the_better_score_wins_and_a_tie_the_lower_index() {
        let r = regions(24.0);
        let full = logits(|x, y| r.opening[y * GRID_PX + x]);
        let half = logits(|x, y| r.opening[y * GRID_PX + x] && x < 128);
        let (v, _) = select_candidate(&[(&half[..], 0.9), (&full[..], 0.9)], &r);
        assert_eq!(v, Ok(1), "the larger cover scores higher at the same IoU");
        let (v, _) = select_candidate(&[(&full[..], 0.9), (&full[..], 0.9)], &r);
        assert_eq!(v, Ok(0));
        let (v, _) = select_candidate(&[(&full[..], 0.5), (&half[..], 0.95)], &r);
        assert_eq!(v, Ok(1), "a clearly better IoU outweighs the cover");
    }

    #[test]
    fn nothing_usable_says_why() {
        let r = regions(24.0);
        let empty = logits(|_, _| false);
        let (v, _) = select_candidate(&[(&empty[..], 0.9)], &r);
        assert_eq!(v, Err(Refusal::Nothing));
        let (v, _) = select_candidate(&[], &r);
        assert_eq!(v, Err(Refusal::Nothing));
        let everywhere = logits(|_, _| true);
        let (v, _) = select_candidate(&[(&everywhere[..], 0.9)], &r);
        assert_eq!(v, Err(Refusal::SpillsOnTheLips));
        // The whole cavity of a wide-open mouth.
        let tall = regions(40.0);
        assert!(tall.aspect > CAVITY_ASPECT);
        let cavity = logits(|x, y| tall.opening[y * GRID_PX + x]);
        let (v, _) = select_candidate(&[(&cavity[..], 0.9)], &tall);
        assert_eq!(v, Err(Refusal::WholeCavity));
        // The same mask on a low opening is fine: a smile is all teeth.
        let low = logits(|x, y| r.opening[y * GRID_PX + x]);
        assert!(select_candidate(&[(&low[..], 0.9)], &r).0.is_ok());
        // A NaN score is refused, not chosen.
        assert_eq!(
            select_candidate(&[(&low[..], f32::NAN)], &r).0,
            Err(Refusal::Nothing)
        );
    }

    #[test]
    fn band_negatives_sit_in_the_opening_just_under_the_upper_lip() {
        let g = mouth(24.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let regions = Regions::from_geometry(&g, &crop).expect("regions");
        let shape = PromptShape {
            band_negatives: Some(0.03),
            ..PromptShape::DEFAULT
        };
        let p = teeth_prompt_with(&g, &crop, PromptStyle::Points, &shape).expect("prompt");
        let neg: Vec<_> = p.points.iter().filter(|q| !q.positive).collect();
        assert_eq!(neg.len(), 7, "three band points and the four lip points");
        // The three band points come first among the negatives: in the opening, above the positive of their column.
        let pos: Vec<_> = p.points.iter().filter(|q| q.positive).collect();
        for (k, q) in neg.iter().take(3).enumerate() {
            assert!(regions.opening[cell(q)], "{q:?}");
            let above = pos
                .iter()
                .find(|r| (r.x - q.x).abs() < 0.02)
                .expect("column");
            assert!(q.y < above.y + 1e-6 || k == 1, "{q:?} {above:?}");
        }
        // The default shape has none.
        let d = teeth_prompt(&g, &crop, PromptStyle::Points).expect("prompt");
        assert_eq!(d.points.iter().filter(|q| !q.positive).count(), 4);
    }

    #[test]
    fn the_gum_query_puts_a_positive_above_a_negative_on_the_teeth() {
        let g = mouth(30.0);
        let crop = MouthCrop::from_geometry(&g, CROP_FACTOR).expect("crop");
        let r = Regions::from_geometry(&g, &crop).expect("regions");
        // The first answer: the lower 60 % of the opening.
        let top = (0..GRID_PX * GRID_PX)
            .filter(|&i| r.opening[i])
            .map(|i| i / GRID_PX)
            .min()
            .expect("opening");
        let bottom = (0..GRID_PX * GRID_PX)
            .filter(|&i| r.opening[i])
            .map(|i| i / GRID_PX)
            .max()
            .expect("opening");
        let cut = top + (bottom - top) * 2 / 5;
        let teeth = logits(|x, y| r.opening[y * GRID_PX + x] && y >= cut);
        let p = gum_prompt(&g, &crop, &teeth, &GumShape::DEFAULT).expect("prompt");
        assert_eq!(p.points.len(), 3 * 2 + 4);
        for pair in p.points[..6].chunks(2) {
            assert!(pair[0].positive && !pair[1].positive);
            assert!(
                pair[0].y < pair[1].y,
                "the gum point is above the teeth point"
            );
            assert!(r.opening[cell(&pair[0])]);
            assert!(
                teeth[cell(&pair[1])] > 0.0,
                "the negative is on the first answer"
            );
        }
        // Nothing found, or an answer that runs up to the lip so that the point of the gum would sit on the teeth: no query.
        assert!(gum_prompt(&g, &crop, &logits(|_, _| false), &GumShape::DEFAULT).is_none());
        let all = logits(|x, y| r.opening[y * GRID_PX + x]);
        let tight = GumShape {
            depth: 0.5,
            ..GumShape::DEFAULT
        };
        assert!(gum_prompt(&g, &crop, &all, &tight).is_none());
        assert!(gum_prompt(&g, &crop, &all[..10], &GumShape::DEFAULT).is_none());
    }

    #[test]
    fn the_lip_gap_is_zero_for_a_mask_at_the_lip_and_grows_with_the_band() {
        let r = regions(30.0);
        let first = (0..GRID_PX * GRID_PX)
            .find(|&i| r.opening[i])
            .map(|i| i / GRID_PX)
            .expect("opening");
        let at_lip = logits(|x, y| r.opening[y * GRID_PX + x]);
        assert!(lip_gap(&at_lip, &r) < 1e-9, "{}", lip_gap(&at_lip, &r));
        let below = logits(|x, y| r.opening[y * GRID_PX + x] && y >= first + 4);
        let deeper = logits(|x, y| r.opening[y * GRID_PX + x] && y >= first + 8);
        let (a, b) = (lip_gap(&below, &r), lip_gap(&deeper, &r));
        assert!(a > 0.0 && b > a, "{a} {b}");
        assert!(b <= LIP_GAP_CAP);
        assert_eq!(lip_gap(&logits(|_, _| false), &r), 0.0);
    }

    #[test]
    fn the_crop_mask_places_the_logits_on_the_picture() {
        let crop = MouthCrop {
            x_px: -10.0,
            y_px: 20.0,
            side_px: 128.0,
        };
        // The left half of the crop is on, the right half off.
        let l = logits(|x, _| x < GRID_PX / 2);
        let m = crop_mask(&l, &crop, (200, 100))
            .expect("mask")
            .expect("not empty");
        assert_eq!(m.region.x_px, 0, "clipped at the picture's left edge");
        assert_eq!(m.region.y_px, 20);
        assert_eq!(m.region.height_px, 80, "clipped at the bottom");
        assert!(m.value_at(10, 40) > 250);
        assert!(m.value_at(100, 40) < 5);
        assert_eq!(m.value_at(150, 40), 0, "outside the crop");
        // Outside the picture: nothing.
        let far = MouthCrop {
            x_px: 500.0,
            ..crop
        };
        assert!(crop_mask(&l, &far, (200, 100)).expect("ok").is_none());
        // An empty mask: nothing.
        assert!(
            crop_mask(&logits(|_, _| false), &crop, (200, 100))
                .expect("ok")
                .is_none()
        );
    }
}
