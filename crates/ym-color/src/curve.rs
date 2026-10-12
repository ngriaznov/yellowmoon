//! The luma and per-channel tone curve (up to 16 control points) and its 1D table (stage C11) (`docs/02-colour-pipeline.md` section C11).
//!
//! Four monotone cubic splines on `[0, 1]` (Fritsch and Carlson, 1980): luma and R, G, B. Each has
//! 2 to [`MAX_CONTROL_POINTS`] control points in `[0, 1]^2` with strictly increasing `x`, is
//! evaluated exactly by [`Curve::eval`] and baked into a [`TABLE_SIZE`]-entry table
//! ([`Curve::bake`]) that the GPU samples as a 1D texture with linear filtering.
//!
//! # Conventions
//!
//! * **Spline.** Initial tangents are the mean of the two neighbouring secant slopes (zero at a
//!   local extremum or flat segment, the one-sided slope at the two ends), then limited so that
//!   `alpha^2 + beta^2 <= 9` on every segment (Fritsch-Carlson). The spline is C1, passes through
//!   every control point, is monotone on every segment where the data are, and never overshoots
//!   the data of a segment.
//! * **Outside the control points.** Between 0 and the first point and between the last point and 1
//!   the curve holds the end value (a flat extension).
//! * **Outside `[0, 1]`.** LogC4 values can leave `[0, 1]` (negative linear light, very bright
//!   highlights). [`CurveTable::eval_extended`] continues with slope 1 from the end of the table, so
//!   the curve's offset at 0 (or at 1) carries on as a constant shift and the map stays monotone.
//! * **Texture sampling.** A 1D texture of [`TABLE_SIZE`] texels with linear filtering and
//!   clamp-to-edge, sampled at `u = (x * 1023 + 0.5) / 1024` for `x` in `[0, 1]`, gives exactly
//!   [`CurveTable::lookup`]; the slope-1 extension for `x` outside `[0, 1]` is done in the shader.
//! * **Luma.** The luma curve is applied as `v += curve(L) - L` on all three channels with `L` the
//!   LogC4 luma of the pixel before the curve ([`crate::logc::LogC4::luma`]); the R, G and B curves
//!   then act on the result per channel ([`BakedCurves::apply`]).

use crate::consts::Vec3;
use crate::logc::LogC4;

/// Most control points one curve may have.
pub const MAX_CONTROL_POINTS: usize = 16;

/// Entries in a baked table (the 1D texture width).
pub const TABLE_SIZE: usize = 1024;

/// Why a curve was rejected.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum CurveError {
    /// A curve needs at least its two end points.
    #[error("a curve needs at least 2 control points, got {0}")]
    TooFewPoints(usize),
    /// More than [`MAX_CONTROL_POINTS`] points.
    #[error("a curve has at most {MAX_CONTROL_POINTS} control points, got {0}")]
    TooManyPoints(usize),
    /// A coordinate is not finite or lies outside `[0, 1]`.
    #[error("control point {index} ({x}, {y}) is outside the unit square")]
    OutOfRange {
        /// Index of the point.
        index: usize,
        /// Its `x`.
        x: f64,
        /// Its `y`.
        y: f64,
    },
    /// `x` must strictly increase from one point to the next.
    #[error("control point {index} does not have a larger x than the point before it")]
    NotIncreasing {
        /// Index of the offending point.
        index: usize,
    },
}

/// A control point in the unit square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Input (LogC4 code value, `0..=1`).
    pub x: f64,
    /// Output.
    pub y: f64,
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Self { x, y }
    }
}

/// A monotone cubic (Fritsch-Carlson) curve through up to 16 control points.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    points: Vec<Point>,
    tangents: Vec<f64>,
    identity: bool,
}

impl Default for Curve {
    fn default() -> Self {
        Self::identity()
    }
}

impl Curve {
    /// The identity curve through `(0, 0)` and `(1, 1)`.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            points: vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 1.0 }],
            tangents: vec![1.0, 1.0],
            identity: true,
        }
    }

    /// Builds the spline through `points`.
    ///
    /// # Errors
    ///
    /// [`CurveError`] when there are fewer than 2 or more than 16 points, a coordinate is not finite
    /// or outside `[0, 1]`, or `x` does not strictly increase.
    pub fn new<P: Into<Point> + Copy>(points: &[P]) -> Result<Self, CurveError> {
        let pts: Vec<Point> = points.iter().map(|&p| p.into()).collect();
        if pts.len() < 2 {
            return Err(CurveError::TooFewPoints(pts.len()));
        }
        if pts.len() > MAX_CONTROL_POINTS {
            return Err(CurveError::TooManyPoints(pts.len()));
        }
        for (index, p) in pts.iter().enumerate() {
            let ok = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
            if !ok(p.x) || !ok(p.y) {
                return Err(CurveError::OutOfRange {
                    index,
                    x: p.x,
                    y: p.y,
                });
            }
            if index > 0 && p.x <= pts[index - 1].x {
                return Err(CurveError::NotIncreasing { index });
            }
        }
        let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.y).collect();
        let tangents = fritsch_carlson_tangents(&xs, &ys);
        let identity = pts.iter().all(|p| p.x == p.y);
        Ok(Self {
            points: pts,
            tangents,
            identity,
        })
    }

    /// The control points.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Whether every control point lies on `y = x` (the curve is then exactly the identity on its
    /// range).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.identity && self.points[0].x == 0.0 && self.points[self.points.len() - 1].x == 1.0
    }

    /// Evaluates the curve at `x`; flat outside the first and last control point.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.points.len();
        let first = self.points[0];
        let last = self.points[n - 1];
        if x.is_nan() {
            return first.y;
        }
        if x <= first.x {
            return first.y;
        }
        if x >= last.x {
            return last.y;
        }
        if self.identity {
            return x;
        }
        // Segment k with points[k].x <= x < points[k + 1].x.
        let k = self
            .points
            .partition_point(|p| p.x <= x)
            .saturating_sub(1)
            .min(n - 2);
        let (p0, p1) = (self.points[k], self.points[k + 1]);
        hermite(
            p0.x,
            p0.y,
            self.tangents[k],
            p1.x,
            p1.y,
            self.tangents[k + 1],
            x,
        )
    }

    /// Bakes the curve into a [`TABLE_SIZE`]-entry table sampled at `i / (TABLE_SIZE - 1)`.
    #[must_use]
    pub fn bake(&self) -> CurveTable {
        let last = (TABLE_SIZE - 1) as f64;
        let values = (0..TABLE_SIZE)
            .map(|i| self.eval(i as f64 / last))
            .collect();
        CurveTable {
            values,
            identity: self.is_identity(),
        }
    }
}

/// Initial tangents by Fritsch and Carlson (1980), with the monotonicity limiter, for strictly
/// increasing `xs` (at least two) and matching `ys`.
pub(crate) fn fritsch_carlson_tangents(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let delta: Vec<f64> = (0..n - 1)
        .map(|k| (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]))
        .collect();
    let mut m = vec![0.0; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for k in 1..n - 1 {
        m[k] = if delta[k - 1] * delta[k] > 0.0 {
            0.5 * (delta[k - 1] + delta[k])
        } else {
            0.0
        };
    }
    for k in 0..n - 1 {
        if delta[k] == 0.0 {
            m[k] = 0.0;
            m[k + 1] = 0.0;
            continue;
        }
        let alpha = m[k] / delta[k];
        let beta = m[k + 1] / delta[k];
        let norm2 = alpha * alpha + beta * beta;
        if norm2 > 9.0 {
            let tau = 3.0 / norm2.sqrt();
            m[k] = tau * alpha * delta[k];
            m[k + 1] = tau * beta * delta[k];
        }
    }
    m
}

/// Fritsch and Carlson tangents for a **periodic** curve through `xs.len() >= 2` points with
/// strictly increasing `xs` inside one `period` (the segment after the last point runs to
/// `xs[0] + period`). Every point has two neighbours, so the end points get the same
/// mean-of-secants tangent and the same limiter as the inner ones: the closed curve is C1 at every
/// point, including the seam, and never overshoots the data of a segment.
///
/// The limiter runs once round the cycle from segment 0. Scaling a tangent down can only lower the
/// `alpha^2 + beta^2` of the neighbouring segment, so one pass leaves every segment inside the
/// monotonicity region, including the closing segment.
pub(crate) fn fritsch_carlson_tangents_periodic(xs: &[f64], ys: &[f64], period: f64) -> Vec<f64> {
    let n = xs.len();
    let width = |k: usize| {
        if k + 1 < n {
            xs[k + 1] - xs[k]
        } else {
            xs[0] + period - xs[n - 1]
        }
    };
    let delta: Vec<f64> = (0..n)
        .map(|k| (ys[(k + 1) % n] - ys[k]) / width(k))
        .collect();
    let mut m: Vec<f64> = (0..n)
        .map(|k| {
            let (before, after) = (delta[(k + n - 1) % n], delta[k]);
            if before * after > 0.0 {
                0.5 * (before + after)
            } else {
                0.0
            }
        })
        .collect();
    for k in 0..n {
        let next = (k + 1) % n;
        if delta[k] == 0.0 {
            m[k] = 0.0;
            m[next] = 0.0;
            continue;
        }
        let alpha = m[k] / delta[k];
        let beta = m[next] / delta[k];
        let norm2 = alpha * alpha + beta * beta;
        if norm2 > 9.0 {
            let tau = 3.0 / norm2.sqrt();
            m[k] = tau * alpha * delta[k];
            m[next] = tau * beta * delta[k];
        }
    }
    m
}

/// Cubic Hermite value at `x` on the segment `(x0, y0, m0)` to `(x1, y1, m1)`.
pub(crate) fn hermite(x0: f64, y0: f64, m0: f64, x1: f64, y1: f64, m1: f64, x: f64) -> f64 {
    let h = x1 - x0;
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
}

/// A baked curve: [`TABLE_SIZE`] samples of the curve at `i / (TABLE_SIZE - 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveTable {
    values: Vec<f64>,
    identity: bool,
}

impl CurveTable {
    /// The table entries.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The table entries as f32, the form uploaded as a 1D texture.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        self.values.iter().map(|&v| v as f32).collect()
    }

    /// Whether the source curve is the exact identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Linear interpolation in the table with `x` clamped to `[0, 1]` (clamp-to-edge sampling).
    #[must_use]
    pub fn lookup(&self, x: f64) -> f64 {
        let last = (TABLE_SIZE - 1) as f64;
        let pos = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) } * last;
        let i = (pos.floor() as usize).min(TABLE_SIZE - 2);
        let f = pos - i as f64;
        self.values[i] + (self.values[i + 1] - self.values[i]) * f
    }

    /// [`CurveTable::lookup`] inside `[0, 1]`; outside it continues with slope 1 from the end value
    /// (`x + table[0]` below 0, `x - 1 + table[last]` above 1).
    #[must_use]
    pub fn eval_extended(&self, x: f64) -> f64 {
        if x < 0.0 {
            x + self.values[0]
        } else if x > 1.0 {
            x - 1.0 + self.values[TABLE_SIZE - 1]
        } else {
            self.lookup(x)
        }
    }
}

/// The four curves of the Curve tab: luma and R, G, B.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CurveSet {
    /// Luma curve.
    pub luma: Curve,
    /// Red curve.
    pub r: Curve,
    /// Green curve.
    pub g: Curve,
    /// Blue curve.
    pub b: Curve,
}

impl CurveSet {
    /// Whether all four curves are the identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.luma.is_identity()
            && self.r.is_identity()
            && self.g.is_identity()
            && self.b.is_identity()
    }

    /// Bakes all four curves.
    #[must_use]
    pub fn bake(&self) -> BakedCurves {
        BakedCurves {
            luma: self.luma.bake(),
            r: self.r.bake(),
            g: self.g.bake(),
            b: self.b.bake(),
        }
    }
}

/// The four baked tables of stage C11.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedCurves {
    /// Luma table.
    pub luma: CurveTable,
    /// Red table.
    pub r: CurveTable,
    /// Green table.
    pub g: CurveTable,
    /// Blue table.
    pub b: CurveTable,
}

impl BakedCurves {
    /// Applies stage C11 to a LogC4 pixel: luma as `v += curve(L) - L`, then R, G, B per channel.
    #[must_use]
    pub fn apply(&self, v: LogC4) -> LogC4 {
        let mut out: Vec3 = v.0;
        if !self.luma.is_identity() {
            let l = v.luma();
            let shift = self.luma.eval_extended(l) - l;
            for c in &mut out {
                *c += shift;
            }
        }
        if !self.r.is_identity() {
            out[0] = self.r.eval_extended(out[0]);
        }
        if !self.g.is_identity() {
            out[1] = self.g.eval_extended(out[1]);
        }
        if !self.b.is_identity() {
            out[2] = self.b.eval_extended(out[2]);
        }
        LogC4(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_exact() {
        let c = Curve::identity();
        for x in [0.0, 0.1, 0.5, 0.999, 1.0] {
            assert_eq!(c.eval(x), x);
        }
        assert!(c.is_identity());
        assert!(c.bake().is_identity());
    }

    #[test]
    fn limits() {
        let pts: Vec<(f64, f64)> = (0..17)
            .map(|i| (f64::from(i) / 16.0, f64::from(i) / 16.0))
            .collect();
        assert_eq!(Curve::new(&pts), Err(CurveError::TooManyPoints(17)));
        assert!(Curve::new(&pts[..16]).is_ok());
        assert_eq!(Curve::new(&pts[..1]), Err(CurveError::TooFewPoints(1)));
    }
}
