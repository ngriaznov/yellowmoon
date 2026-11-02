//! Masks on the CPU reference: the shapes (Linear, Radial), the ranges (Colour, Brightness), how the
//! rows of a node combine with each other and with a linked key, and the node's mask `m`
//! (`docs/23-nodes.md` section 4, `docs/algorithms/06-masks.md` sections 3 and 4).
//!
//! A node's **mask rows** ([`NodeMasks`]) are the enabled rows of its list in order, each with a kind, a
//! mode (add or subtract), an Invert and an Opacity. [`NodeMasks::prepare`] turns them into
//! [`PreparedMasks`]: one [`MaskOp`] for every row (the kind, the flags, the opacity and eight numbers
//! derived once in `f64`: sines and cosines, the ranges' ramps) and the node's Invert. **The device reads
//! the same [`MaskOp`]s** (`ym-gpu`'s `MaskOpU` is these numbers narrowed to `f32`), so the two evaluations
//! cannot disagree about a constant, and [`PreparedMasks::mask`] is the one statement of the combination
//! rule that the kernel's loop mirrors line by line.
//!
//! # What a mask reads
//!
//! `m` is a function of the node's **input** `v_in` (never of what the node made of it) and of the pixel's
//! place in the **photo frame**, for the shapes:
//!
//! * the **shapes** read the pixel's position, `(x, y)` in [0, 1] over the photo as the camera oriented it
//!   after the lens corrections and before the lens bend, the crop, the straightening and the turn
//!   (`docs/23` section 3.6), as a [`PhotoSpot`]: the position and the [`PhotoFrame`] of the render (the
//!   photo's aspect, and the size of one output pixel in units of the photo's long edge). A call that has
//!   no position (`None`) gives every shape `s = 0`;
//! * the **Colour range** reads the chroma `(e1 . v, e2 . v)` of `v_in` (C9's axes), the **Brightness
//!   range** the luma code `L_in = encode(Y(decode(v_in)))`;
//! * a **raster** kind (Object, Subject, People, Background, Brush) selects nothing yet: `s = 0` until
//!   their tasks (N-13 to N-19) give it a plane.
//!
//! # Combination
//!
//! `a = opacity * (invert ? 1 - s : s)`. A linked key is an implicit first region row (add, opacity 1, no
//! Invert) whose `s` is the source node's `m`. The region rows (the key, Linear, Radial and the raster
//! kinds) fold into `R` and the range rows into `K`: an add is `acc = 1 - (1 - acc)(1 - a)` and a subtract
//! `acc = acc (1 - a)`, starting from 0, or from 1 when the group's first row is a subtract; a group with
//! no row is 1. `m = R K`, and the node's Invert gives `m = 1 - m`. The first row of a group is taken
//! exactly (`a` for an add, `1 - a` for a subtract), so a node with only a key input has `m` equal to its
//! source's, bit for bit.

use crate::consts::Vec3;
use crate::gamut::LinearAwg4;
use crate::grade::{E1, E2};
use crate::logc::{LogC4, logc4_luma_of_linear};

/// The narrowest ramp of a range's edge, in LogC4 code values (`docs/algorithms/06` section 4: about a third
/// of a stop; an edge is then at most `1.5 / 0.02 = 75` steep, so an input that differs by `1e-5` between the
/// device and the CPU moves `m` by at most `7.5e-4`).
pub const RANGE_RAMP_MIN: f64 = 0.02;
/// The Brightness range's softness 1 is this wide, in code values (`q = max(0.25 sigma, RANGE_RAMP_MIN)`).
pub const BRIGHTNESS_SOFTNESS_SPAN: f64 = 0.25;
/// The most mask rows a node has (`docs/23` section 3.5). The kernel does not depend on it.
pub const MAX_MASK_ROWS: usize = 64;

// ---------------------------------------------------------------------------------------------
// The photo frame
// ---------------------------------------------------------------------------------------------

/// The narrowest a shape's edge is made, in output pixels, **in total and centred on the edge**: the coverage
/// of a hard edge by one pixel (`docs/algorithms/06` section 4, the lead's decision of 2026-11-02). A shape's
/// transition is `max(its feather, MIN_RAMP_PIXELS ε_px)` wide, centred on the contour the feather alone would
/// have, so a hard edge previews as it exports.
pub const MIN_RAMP_PIXELS: f64 = 1.0;

/// What a render knows about the photo frame: its aspect and the size of one output pixel in it.
///
/// Lengths in the settings are fractions of the photo's **long edge** `L`. The frame is `W x H` photo
/// pixels (the photo as the camera oriented it, after the lens corrections), so a pixel at `(x, y)` is
/// at `((x - cx) W, (y - cy) H) / L` from a centre `(cx, cy)` in those units, and a circle is a circle
/// on any photo. `pixel` is `1 / u_px`: one output pixel in the same units, and [`PhotoFrame::ramp`] the
/// narrowest a shape's edge is made (`docs/17` K8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotoFrame {
    /// `(W, H) / L`: one of them is 1.
    pub aspect: [f64; 2],
    /// One output pixel in units of `L` (`1 / u_px`).
    pub pixel: f64,
}

impl PhotoFrame {
    /// The narrowest transition of a shape's edge in units of `L`: [`MIN_RAMP_PIXELS`] output pixels.
    #[must_use]
    pub fn ramp(&self) -> f64 {
        MIN_RAMP_PIXELS * self.pixel
    }

    /// The frame of a `width x height` photo (any unit) whose output pixel is `pixel` of that unit.
    /// A zero or non-finite size reads as a square photo of one unit.
    #[must_use]
    pub fn new(width: f64, height: f64, pixel: f64) -> Self {
        let long = width.max(height);
        if !(long.is_finite() && long > 0.0 && width > 0.0 && height > 0.0) {
            return Self {
                aspect: [1.0, 1.0],
                pixel: 1.0,
            };
        }
        let pixel = if pixel.is_finite() && pixel > 0.0 {
            pixel / long
        } else {
            1.0 / long
        };
        Self {
            aspect: [width / long, height / long],
            pixel,
        }
    }
}

/// Where a pixel is in the photo frame, and the frame it is in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotoSpot {
    /// `(x, y)` in [0, 1] over the oriented photo.
    pub pos: [f64; 2],
    /// The render's frame.
    pub frame: PhotoFrame,
}

/// The place of a pixel in the photo frame, when the caller knows it. `None`: the shapes select nothing.
pub type PixelAt = Option<PhotoSpot>;

/// The photo-frame position of the pixels of a frame of pixels (`Frame` of `display`): the geometry that
/// maps output pixels to the photo implements it (`ym-gpu`'s sampler grid).
pub trait PhotoMap: Sync {
    /// The position `(x, y)` in [0, 1] over the oriented photo of the centre of pixel `(x, y)` of the frame
    /// (its own coordinates).
    fn position(&self, x: usize, y: usize) -> [f64; 2];
}

// ---------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------

/// What a mask row selects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskKind {
    /// A band across the photo (`docs/algorithms/06` section 4.1).
    Linear {
        /// A point of the centre line in the photo frame.
        center: [f64; 2],
        /// The line's angle in degrees, clockwise from the horizontal.
        angle_deg: f64,
        /// Feather, 0 to 1 (half the transition's width is `0.5 f` of the long edge).
        feather: f64,
    },
    /// An ellipse (section 4.2).
    Radial {
        /// The centre in the photo frame.
        center: [f64; 2],
        /// The two radii, fractions of the long edge.
        radii: [f64; 2],
        /// The rotation of the first axis in degrees, clockwise.
        angle_deg: f64,
        /// Feather, 0 to 1 (the ramp, inwards from the outline, in units of the radius).
        feather: f64,
    },
    /// A range on the chroma of the node's input (section 4.3).
    ColorRange {
        /// The centre in the plane of `e1`, `e2`.
        center: [f64; 2],
        /// The width, 0.005 to 0.3.
        width: f64,
        /// Softness, 0 to 1.
        softness: f64,
    },
    /// A range on the luma code of the node's input (section 4.4).
    BrightnessRange {
        /// The low end, a LogC4 code value.
        low: f64,
        /// The high end. A `low` above it reads as both at their mean.
        high: f64,
        /// Softness, 0 to 1.
        softness: f64,
    },
    /// Object, Subject, People, Background and Brush: made from a raster plane, which is not built yet.
    /// Selects nothing (`s = 0`).
    Raster,
}

/// A mask row as the chain runs it (the rows with the eye off are not here: they are absent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskRow {
    /// What the row selects.
    pub kind: MaskKind,
    /// The row takes away from the mask instead of adding to it.
    pub subtract: bool,
    /// This row alone is inverted.
    pub invert: bool,
    /// 0 to 1.
    pub opacity: f64,
}

impl MaskRow {
    /// An added, not inverted row at full opacity.
    #[must_use]
    pub fn new(kind: MaskKind) -> Self {
        Self {
            kind,
            subtract: false,
            invert: false,
            opacity: 1.0,
        }
    }
}

/// A node's mask rows (the enabled ones, in order) and the node's own Invert.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodeMasks {
    /// The rows, in list order.
    pub rows: Vec<MaskRow>,
    /// The node's Invert: `m = 1 - m` after the rows combine.
    pub invert: bool,
}

impl NodeMasks {
    /// No rows and no Invert: `m = 1` and the node blends nothing (until a key is linked).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && !self.invert
    }

    /// The prepared form: every row's constants derived once.
    #[must_use]
    pub fn prepare(&self) -> PreparedMasks {
        let ops: Vec<MaskOp> = self.rows.iter().map(MaskOp::of).collect();
        PreparedMasks {
            needs_luma: ops.iter().any(|o| o.kind == OpKind::Brightness),
            needs_position: ops
                .iter()
                .any(|o| matches!(o.kind, OpKind::Linear | OpKind::Radial)),
            ops,
            invert: self.invert,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Prepared rows
// ---------------------------------------------------------------------------------------------

/// The kind of a prepared row. The numbers are the kernel's (`MK_*` in `color.wgsl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Linear.
    Linear,
    /// Radial.
    Radial,
    /// Colour range.
    Color,
    /// Brightness range.
    Brightness,
    /// A raster kind: selects nothing yet.
    Raster,
}

impl OpKind {
    /// The kernel's code for the kind.
    #[must_use]
    pub fn code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Radial => 1,
            Self::Color => 2,
            Self::Brightness => 3,
            Self::Raster => 4,
        }
    }
}

/// One mask row ready to evaluate: the kind, the flags, the opacity and up to eight numbers derived
/// from the row's fields once, in `f64` (the kernel's record is these, narrowed). Per kind:
///
/// | Kind | `params` |
/// |---|---|
/// | Linear | `cx, cy, sin(angle), cos(angle), feather` |
/// | Radial | `cx, cy, sin(angle), cos(angle), rx, ry, feather` |
/// | Colour range | `a0, b0, width, q` with `q = max(softness width, RANGE_RAMP_MIN)` |
/// | Brightness range | `low - q, high, q` with `q = max(0.25 softness, RANGE_RAMP_MIN)` |
/// | Raster | none |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskOp {
    /// The kind.
    pub kind: OpKind,
    /// Subtract instead of add.
    pub subtract: bool,
    /// This row alone is inverted.
    pub invert: bool,
    /// 0 to 1.
    pub opacity: f64,
    /// The derived numbers (the table above); the rest are 0.
    pub params: [f64; 8],
}

/// `x` in [0, 1] if it is a number, `default` if it is not.
fn unit_or(x: f64, default: f64) -> f64 {
    if x.is_nan() {
        default
    } else {
        x.clamp(0.0, 1.0)
    }
}

fn finite_or(x: f64, default: f64) -> f64 {
    if x.is_finite() { x } else { default }
}

impl MaskOp {
    /// The row's constants. Numbers outside their ranges are held to them (the settings have already
    /// clamped what a person can enter); a number that is not a number reads as the field's default.
    #[must_use]
    pub fn of(row: &MaskRow) -> Self {
        let mut params = [0.0; 8];
        let kind = match row.kind {
            MaskKind::Linear {
                center,
                angle_deg,
                feather,
            } => {
                let (s, c) = finite_or(angle_deg, 0.0).to_radians().sin_cos();
                params[..5].copy_from_slice(&[
                    unit_or(center[0], 0.5),
                    unit_or(center[1], 0.5),
                    s,
                    c,
                    unit_or(feather, 0.2),
                ]);
                OpKind::Linear
            }
            MaskKind::Radial {
                center,
                radii,
                angle_deg,
                feather,
            } => {
                let (s, c) = finite_or(angle_deg, 0.0).to_radians().sin_cos();
                let radius = |r: f64, d: f64| {
                    if r.is_nan() { d } else { r.clamp(0.002, 1.0) }
                };
                params[..7].copy_from_slice(&[
                    unit_or(center[0], 0.5),
                    unit_or(center[1], 0.5),
                    s,
                    c,
                    radius(radii[0], 0.25),
                    radius(radii[1], 0.18),
                    unit_or(feather, 0.3),
                ]);
                OpKind::Radial
            }
            MaskKind::ColorRange {
                center,
                width,
                softness,
            } => {
                let width = if width.is_nan() {
                    0.05
                } else {
                    width.clamp(0.005, 0.3)
                };
                let q = (unit_or(softness, 0.5) * width).max(RANGE_RAMP_MIN);
                params[..4].copy_from_slice(&[
                    finite_or(center[0], 0.0),
                    finite_or(center[1], 0.0),
                    width,
                    q,
                ]);
                OpKind::Color
            }
            MaskKind::BrightnessRange {
                low,
                high,
                softness,
            } => {
                let (mut low, mut high) = (unit_or(low, 0.35), unit_or(high, 1.0));
                if low > high {
                    let mean = 0.5 * (low + high);
                    (low, high) = (mean, mean);
                }
                let q = (BRIGHTNESS_SOFTNESS_SPAN * unit_or(softness, 0.5)).max(RANGE_RAMP_MIN);
                params[..3].copy_from_slice(&[low - q, high, q]);
                OpKind::Brightness
            }
            MaskKind::Raster => OpKind::Raster,
        };
        Self {
            kind,
            subtract: row.subtract,
            invert: row.invert,
            opacity: unit_or(row.opacity, 1.0),
            params,
        }
    }

    /// Whether the row belongs to the ranges (`K`) rather than to the regions (`R`).
    #[must_use]
    pub fn is_range(&self) -> bool {
        matches!(self.kind, OpKind::Color | OpKind::Brightness)
    }

    /// The row's own `s` in [0, 1] at a pixel: `v_in` is the node's input, `l_in` its luma code, `spot`
    /// the pixel's place in the photo frame.
    #[must_use]
    pub fn select(&self, v_in: LogC4, l_in: f64, spot: PixelAt) -> f64 {
        let p = &self.params;
        match self.kind {
            OpKind::Linear => {
                let Some(spot) = spot else { return 0.0 };
                let (x, y) = frame_xy(p[0], p[1], &spot);
                // The signed distance from the centre line along n = (-sin, cos), and the width of the
                // transition, centred on the line: the feather, and at least one output pixel.
                let s = -p[2] * x + p[3] * y;
                let w = p[4].max(spot.frame.ramp());
                1.0 - smooth_step((s + 0.5 * w) / w)
            }
            OpKind::Radial => {
                let Some(spot) = spot else { return 0.0 };
                let (x, y) = frame_xy(p[0], p[1], &spot);
                // The pixel rotated by -angle, in units of the radii.
                let u = (p[3] * x + p[2] * y) / p[4];
                let w = (-p[2] * x + p[3] * y) / p[5];
                let d = (u * u + w * w).sqrt();
                // The ramp, in units of d: the feather, and at least one pixel on the narrower side. It is
                // centred on the contour the feather alone has, d = 1 - f/2 (the feather runs inwards from the
                // outline), so with a feather of a pixel or more it is [1 - f, 1] and at Feather 0 it is one
                // pixel centred on the outline.
                let q = p[6].max(spot.frame.ramp() / p[4].min(p[5]));
                1.0 - smooth_step((d - (1.0 - 0.5 * p[6] - 0.5 * q)) / q)
            }
            OpKind::Color => {
                let a = dot(E1, v_in.0) - p[0];
                let b = dot(E2, v_in.0) - p[1];
                let d = (a * a + b * b).sqrt();
                1.0 - smooth_step((d - p[2]) / p[3])
            }
            OpKind::Brightness => {
                // S((L - (low - q)) / q) * (1 - S((L - high) / q)), params = (low - q, high, q).
                smooth_step((l_in - p[0]) / p[2]) * (1.0 - smooth_step((l_in - p[1]) / p[2]))
            }
            OpKind::Raster => 0.0,
        }
    }

    /// The row's `a`: its `s`, inverted if the row is, times its opacity.
    #[must_use]
    pub fn amount(&self, v_in: LogC4, l_in: f64, spot: PixelAt) -> f64 {
        let s = self.select(v_in, l_in, spot);
        self.opacity * if self.invert { 1.0 - s } else { s }
    }
}

/// `((px - cx) W, (py - cy) H) / L`: the pixel's offset from a centre, in units of the long edge.
fn frame_xy(cx: f64, cy: f64, spot: &PhotoSpot) -> (f64, f64) {
    (
        (spot.pos[0] - cx) * spot.frame.aspect[0],
        (spot.pos[1] - cy) * spot.frame.aspect[1],
    )
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `S(t) = t^2 (3 - 2t)` for `t` held to [0, 1] (the smoothstep of the maths file; a NaN is 0).
#[must_use]
pub fn smooth_step(t: f64) -> f64 {
    if t.is_nan() {
        return 0.0;
    }
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One step of a group's fold: `acc` is `None` while the group is empty. The first row is taken exactly
/// (`a` added, `1 - a` subtracted); after it an add is the screen `1 - (1 - acc)(1 - a)` and a subtract
/// `acc (1 - a)`.
#[must_use]
pub fn fold(acc: Option<f64>, subtract: bool, a: f64) -> f64 {
    match (acc, subtract) {
        (None, false) => a,
        (None, true) => 1.0 - a,
        (Some(acc), false) => 1.0 - (1.0 - acc) * (1.0 - a),
        (Some(acc), true) => acc * (1.0 - a),
    }
}

/// A node's mask rows ready to evaluate.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PreparedMasks {
    ops: Vec<MaskOp>,
    invert: bool,
    needs_luma: bool,
    needs_position: bool,
}

impl PreparedMasks {
    /// The prepared rows, in list order (the kernel's records).
    #[must_use]
    pub fn ops(&self) -> &[MaskOp] {
        &self.ops
    }

    /// The node's Invert.
    #[must_use]
    pub fn invert(&self) -> bool {
        self.invert
    }

    /// Whether a row reads the luma of the node's input (a Brightness range).
    #[must_use]
    pub fn needs_luma(&self) -> bool {
        self.needs_luma
    }

    /// Whether a row reads the pixel's place in the photo (a Linear or a Radial).
    #[must_use]
    pub fn needs_position(&self) -> bool {
        self.needs_position
    }

    /// Whether the node blends by a mask: it has rows, or its Invert is on, or a key is linked to it
    /// (`has_key`). A node that does not keeps the bits of its own output (`docs/23` section 15).
    #[must_use]
    pub fn blends(&self, has_key: bool) -> bool {
        has_key || self.invert || !self.ops.is_empty()
    }

    /// The node's mask `m` at a pixel: `key` is the linked key's value (the source node's `m`), `v_in` the
    /// node's input, `l_in` its luma code (read only by a Brightness range: pass any number when
    /// [`PreparedMasks::needs_luma`] is false) and `spot` the pixel's place in the photo frame.
    #[must_use]
    pub fn mask(&self, key: Option<f64>, v_in: LogC4, l_in: f64, spot: PixelAt) -> f64 {
        let mut regions = key;
        let mut ranges: Option<f64> = None;
        for op in &self.ops {
            let a = op.amount(v_in, l_in, spot);
            let acc = if op.is_range() {
                &mut ranges
            } else {
                &mut regions
            };
            *acc = Some(fold(*acc, op.subtract, a));
        }
        let m = match (regions, ranges) {
            (None, None) => 1.0,
            (Some(r), None) => r,
            (None, Some(k)) => k,
            (Some(r), Some(k)) => r * k,
        };
        if self.invert { 1.0 - m } else { m }
    }

    /// The luma code `L_in` of a node's input: `encode(Y(lin))` from the exact linear values when the node
    /// reads the Input (`linear`), else from the decoded `v_in`.
    #[must_use]
    pub fn luma_of(v_in: LogC4, linear: Option<Vec3>) -> f64 {
        logc4_luma_of_linear(LinearAwg4(linear.unwrap_or_else(|| v_in.to_linear().0)))
    }
}
