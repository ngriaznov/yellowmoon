//! Look application: CDL then LUT in LogC4, LogC3 looks via exact conversion, intensity blend (stage C10) (`docs/02-colour-pipeline.md` section C10, `docs/08-looks.md`).
//!
//! A look is `CDL -> LUT`, applied **in its own encoding**, then blended with the unmodified pixel
//! by the look intensity `k = intensity_pct / 100` in LogC4 (`v' = in + k (out - in)`):
//!
//! | [`LookInput`] | Chain |
//! |---|---|
//! | [`LookInput::LogC4`] | `v -> CDL -> LUT`, applied directly |
//! | [`LookInput::LogC3Ei800`] | LogC4 to linear AWG4 to AWG3 to LogC3 EI 800, `CDL -> LUT`, back to linear AWG3 to AWG4 to LogC4 |
//! | [`LookInput::LogC3ToRec709`] | LogC4 to LogC3 EI 800, `CDL -> LUT`, then the gamma 2.4 decode of the LUT's Rec.709 output gives display-linear Rec.709 |
//!
//! The first two return LogC4 ([`LookTransform::apply_logc4`]). The third replaces the display
//! transform (C12): it returns display-linear Rec.709 ([`LookTransform::eval_display_rec709`]), and
//! the pipeline blends it with its own C12 output with [`blend_display_linear`] (both converted to
//! the same linear primaries first, for example with [`crate::gamut::LinearRec709::to_display_p3`]).
//!
//! The built-in **Neutral** look is [`LookTransform::neutral`]: the identity.
//!
//! # Parametric looks (`data/looks/builtin/*.json`)
//!
//! A [`ParametricLook`] is a luma tone curve, a CDL and three colour curves, all evaluated in LogC4,
//! and is **baked** into a 65-point cube ([`ParametricLook::bake`], [`ParametricLook::to_transform`])
//! when loaded, like every parametric look, so the CPU and the GPU read it as an ordinary cube and
//! need no kernel of their own for it.
//!
//! ```json
//! {
//!   "schema": "yellowmoon.parametric-look/1",
//!   "name": "Example",
//!   "input": "logc4",
//!   "luma_curve":  [ { "x": 0.0, "y": 0.0 }, { "x": 0.2784, "y": 0.2784 }, { "x": 0.6, "y": 0.63 }, { "x": 1.0, "y": 1.0 } ],
//!   "cdl": { "slope": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0], "power": [1.0, 1.0, 1.0], "saturation": 1.0 },
//!   "hue_vs_sat":  [ { "hue_deg":  30.0, "value": 1.15 }, { "hue_deg": 210.0, "value": 0.9 } ],
//!   "hue_vs_hue":  [ { "hue_deg":  60.0, "value": -6.0 } ],
//!   "luma_vs_sat": [ { "luma": 0.15, "value": 0.8 }, { "luma": 0.6, "value": 1.1 } ]
//! }
//! ```
//!
//! * Every field except `schema` and `name` is optional; a missing curve is neutral, so
//!   `{"schema": "...", "name": "Neutral"}` is the identity look. Unknown fields are rejected, except the
//!   two blocks of older files, `red_chroma` and `red_density`, which are read and ignored (the steps they
//!   configured went with the cine rendering, `docs/02`).
//! * `luma_curve`: a tone curve on the pixel's luminance, applied **first**. 2 to
//!   [`crate::curve::MAX_CONTROL_POINTS`] (16) points `{"x", "y"}` in `[0, 1]^2` with strictly
//!   increasing `x`; `x` is the LogC4 code value of the luminance of the pixel
//!   ([`LogC4::luma`]: `encode(Y(decode(rgb)))` with the AWG4 `Y` row), `y` what that luminance
//!   becomes. It is applied as `v += T(L) - L` on all three channels, the way the Curve tab's
//!   luma curve is (`docs/02` C11): one shift of the three codes, so it moves tone and leaves the
//!   differences between the channels in LogC4 (hue and the log ratios of the colour) alone. `T` is
//!   the monotone cubic of [`crate::curve::Curve`] (Fritsch-Carlson, up to 16 points), flat between
//!   0 and the first point and between the last point and 1, and continued with slope 1 below 0
//!   and above 1 (as [`crate::curve::CurveTable::eval_extended`]).
//!   Points on `y = x` make the identity. The `x` of `luma_vs_sat` below is a different luma (Rec.709
//!   weights on the LogC4 codes); the two are never mixed.
//! * `cdl`: the ASC CDL of [`crate::cdl`], applied after the luma curve (all four fields required
//!   inside it).
//! * `hue_vs_sat`: multiplies the chroma by `value` (1 is unchanged, 0 is grey, at most 8).
//!   `hue_deg` in `[0, 360)`, strictly increasing, at most 64 points; the curve wraps around.
//! * `hue_vs_hue`: rotates the hue by `value` degrees (within +-180; positive turns red towards
//!   yellow and green). Same hue axis and limits.
//! * `luma_vs_sat`: multiplies the chroma by `value` as a function of the pixel's luma
//!   (`luma` is the Rec.709-weighted LogC4 luma of the CDL output, in `[0, 1]`, strictly
//!   increasing); flat outside the first and last point.
//! * Curves are Fritsch-Carlson monotone cubics through their points, so they never overshoot the
//!   values you give; one point makes a constant. Hue curves are periodic: the segment after the
//!   last point runs on to the first point plus 360 degrees, and every point (the first and last
//!   included) has two neighbours, so the curve is C1 across 0 / 360.
//! * **Hue and chroma** are measured in LogC4 RGB with the axes `E1`, `E2` of [`crate::grade`]:
//!   `p = v . E1`, `q = v . E2`, chroma `hypot(p, q)`, hue `atan2(q, p)` in degrees (0 red, 60
//!   yellow, 120 green, 180 cyan, 240 blue, 300 magenta). Scaling or rotating `(p, q)` keeps the
//!   channel mean; the Rec.709 luma is then restored, so these curves never change luma.
//! * **Order:** `luma_curve`, then `cdl`, then the three colour curves (which see the CDL output).

use serde::{Deserialize, Serialize};

use crate::cdl::{Cdl, CdlError};
use crate::consts::{REC709_LUMA_WEIGHTS, Vec3};
use crate::curve::{
    Curve, CurveError, fritsch_carlson_tangents, fritsch_carlson_tangents_periodic, hermite,
};
use crate::gamut::LinearRec709;
use crate::grade::{E1, E2};
use crate::logc::{LogC3Ei800, LogC4};
use crate::lut::{BAKE_SIZE, Lut, Lut3d, LutError};

/// Gamma of the Rec.709 display encoding that LogC3-to-Rec.709 look LUTs output (BT.1886 style, 2.4).
pub const DISPLAY_GAMMA: f64 = 2.4;

/// The schema identifier of parametric look files.
pub const SCHEMA_ID: &str = "yellowmoon.parametric-look/1";

/// Most points in one parametric curve.
pub const MAX_PROFILE_POINTS: usize = 64;

/// Largest chroma multiplier a curve may ask for.
pub const MAX_SATURATION_FACTOR: f64 = 8.0;

/// The largest value a parametric look's CDL may reach on the unit cube of LogC4 inputs, before its
/// saturation. A real look stays within about -0.5 to 2; a slope and power that send white to 1000 or
/// beyond (or to infinity: slope 5.9 with power 1405 does) leave nothing usable, and the hue curves
/// after the CDL would turn such values into NaN.
pub const CDL_OUTPUT_LIMIT: f64 = 1000.0;

/// What a look expects as input (`docs/08-looks.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LookInput {
    /// LogC4 / AWG4, log to log.
    #[serde(rename = "logc4")]
    LogC4,
    /// LogC3 EI 800 / AWG3, log to log.
    #[serde(rename = "logc3_ei800")]
    LogC3Ei800,
    /// LogC3 EI 800 / AWG3 in, Rec.709 (gamma 2.4) out: look and display rendering baked together.
    #[serde(rename = "logc3_to_rec709")]
    LogC3ToRec709,
}

/// Why a look could not be built or applied.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LookError {
    /// The look includes its own display rendering; use [`LookTransform::eval_display_rec709`].
    #[error("this look includes display rendering; it returns display-linear Rec.709, not LogC4")]
    IncludesDisplayRendering,
    /// This look returns LogC4; use [`LookTransform::apply_logc4`].
    #[error("this look returns LogC4, not display-linear Rec.709")]
    NoDisplayRendering,
    /// A LogC3-to-Rec.709 look needs a LUT.
    #[error("a look that includes display rendering needs a LUT")]
    MissingLut,
    /// The JSON could not be parsed.
    #[error("look file is not valid: {0}")]
    Json(String),
    /// The file names another schema.
    #[error("unsupported look schema {0:?}, expected {SCHEMA_ID:?}")]
    UnsupportedSchema(String),
    /// A parametric look must be evaluated in LogC4.
    #[error("parametric looks are evaluated in LogC4; input {0:?} is not supported")]
    UnsupportedInput(LookInput),
    /// The CDL block is out of range.
    #[error("CDL: {0}")]
    Cdl(#[from] CdlError),
    /// A curve is not valid.
    #[error("{section}: {reason}")]
    BadCurve {
        /// Which curve (`luma_curve`, `hue_vs_sat`, `hue_vs_hue` or `luma_vs_sat`).
        section: &'static str,
        /// What is wrong.
        reason: String,
    },
    /// The cube could not be built.
    #[error("LUT: {0}")]
    Lut(#[from] LutError),
    /// The CDL sends a LogC4 input inside the unit cube beyond [`CDL_OUTPUT_LIMIT`].
    #[error(
        "CDL: the {channel} channel reaches {value:e} at white; a look's CDL must stay below {CDL_OUTPUT_LIMIT}"
    )]
    CdlOutOfRange {
        /// `red`, `green` or `blue`.
        channel: &'static str,
        /// What the slope, offset and power make of an input of 1.0 (may be infinite).
        value: f64,
    },
}

/// The intensity as the fraction `k` in `[0, 1]` (a NaN counts as 0).
#[must_use]
pub fn intensity_fraction(intensity_pct: f64) -> f64 {
    if intensity_pct.is_nan() {
        0.0
    } else {
        intensity_pct.clamp(0.0, 100.0) / 100.0
    }
}

/// Blends in LogC4: `in + k (out - in)`. Exactly `input` at 0 % and exactly `output` at 100 %.
#[must_use]
pub fn blend_logc4(input: LogC4, output: LogC4, intensity_pct: f64) -> LogC4 {
    LogC4(blend(input.0, output.0, intensity_fraction(intensity_pct)))
}

/// Blends two display-linear triples in the **same** linear primaries: `own + k (look - own)`, where
/// `own` is the pipeline's C12 output and `look` the look LUT's decoded output. Exactly `own` at
/// 0 % and exactly `look` at 100 %.
#[must_use]
pub fn blend_display_linear(own: Vec3, look: Vec3, intensity_pct: f64) -> Vec3 {
    blend(own, look, intensity_fraction(intensity_pct))
}

fn blend(a: Vec3, b: Vec3, k: f64) -> Vec3 {
    if k <= 0.0 {
        a
    } else if k >= 1.0 {
        b
    } else {
        [
            a[0] + k * (b[0] - a[0]),
            a[1] + k * (b[1] - a[1]),
            a[2] + k * (b[2] - a[2]),
        ]
    }
}

/// Decodes a Rec.709 gamma-2.4 encoded triple to display-linear (negative values clamp to 0).
#[must_use]
pub fn rec709_gamma24_decode(encoded: Vec3) -> LinearRec709 {
    LinearRec709(encoded.map(|c| if c > 0.0 { c.powf(DISPLAY_GAMMA) } else { 0.0 }))
}

/// A look ready to apply: an input encoding, an optional CDL and an optional LUT.
#[derive(Debug, Clone, PartialEq)]
pub struct LookTransform {
    input: LookInput,
    cdl: Option<Cdl>,
    lut: Option<Lut>,
}

impl LookTransform {
    /// The built-in Neutral look: the identity in LogC4.
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            input: LookInput::LogC4,
            cdl: None,
            lut: None,
        }
    }

    /// Builds a look.
    ///
    /// # Errors
    ///
    /// [`LookError::MissingLut`] for a [`LookInput::LogC3ToRec709`] look without a LUT.
    pub fn new(input: LookInput, cdl: Option<Cdl>, lut: Option<Lut>) -> Result<Self, LookError> {
        if input == LookInput::LogC3ToRec709 && lut.is_none() {
            return Err(LookError::MissingLut);
        }
        Ok(Self { input, cdl, lut })
    }

    /// The input encoding the look expects.
    #[must_use]
    pub fn input(&self) -> LookInput {
        self.input
    }

    /// The CDL, if any.
    #[must_use]
    pub fn cdl(&self) -> Option<&Cdl> {
        self.cdl.as_ref()
    }

    /// The LUT, if any.
    #[must_use]
    pub fn lut(&self) -> Option<&Lut> {
        self.lut.as_ref()
    }

    /// Whether the look replaces the display transform ("includes display rendering").
    #[must_use]
    pub fn includes_display_rendering(&self) -> bool {
        self.input == LookInput::LogC3ToRec709
    }

    /// Whether the look leaves every LogC4 pixel unchanged (no LUT, and no or an identity CDL).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.input == LookInput::LogC4
            && self.lut.is_none()
            && self.cdl.as_ref().is_none_or(Cdl::is_identity)
    }

    /// `CDL -> LUT` on a triple already in the look's own encoding.
    #[must_use]
    pub fn apply_own_encoding(&self, rgb: Vec3) -> Vec3 {
        let after_cdl = match &self.cdl {
            Some(cdl) => cdl.apply(rgb),
            None => rgb,
        };
        match &self.lut {
            Some(lut) => lut.sample(after_cdl),
            None => after_cdl,
        }
    }

    /// The look's unblended output for a LogC4 pixel, as LogC4 (log-to-log looks only).
    ///
    /// # Errors
    ///
    /// [`LookError::IncludesDisplayRendering`] for a [`LookInput::LogC3ToRec709`] look.
    pub fn eval_logc4(&self, v: LogC4) -> Result<LogC4, LookError> {
        match self.input {
            LookInput::LogC4 => Ok(LogC4(self.apply_own_encoding(v.0))),
            LookInput::LogC3Ei800 => {
                let encoded = v.to_logc3_ei800();
                Ok(LogC3Ei800(self.apply_own_encoding(encoded.0)).to_logc4())
            }
            LookInput::LogC3ToRec709 => Err(LookError::IncludesDisplayRendering),
        }
    }

    /// C10 for log-to-log looks: the look's output blended with the input in LogC4 by `intensity_pct` (0 to 100).
    ///
    /// # Errors
    ///
    /// [`LookError::IncludesDisplayRendering`] for a [`LookInput::LogC3ToRec709`] look.
    pub fn apply_logc4(&self, v: LogC4, intensity_pct: f64) -> Result<LogC4, LookError> {
        if intensity_fraction(intensity_pct) <= 0.0 {
            // Still reject display-referred looks, so the error does not depend on the slider.
            if self.includes_display_rendering() {
                return Err(LookError::IncludesDisplayRendering);
            }
            return Ok(v);
        }
        Ok(blend_logc4(v, self.eval_logc4(v)?, intensity_pct))
    }

    /// The unblended output of a LogC3-to-Rec.709 look for a LogC4 pixel: display-linear Rec.709.
    ///
    /// # Errors
    ///
    /// [`LookError::NoDisplayRendering`] for a log-to-log look.
    pub fn eval_display_rec709(&self, v: LogC4) -> Result<LinearRec709, LookError> {
        if self.input != LookInput::LogC3ToRec709 {
            return Err(LookError::NoDisplayRendering);
        }
        let encoded = self.apply_own_encoding(v.to_logc3_ei800().0);
        Ok(rec709_gamma24_decode(encoded))
    }
}

// ---- Parametric looks ------------------------------------------------------------------------

/// The `cdl` block of a parametric look file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CdlSpec {
    /// Slope per channel.
    pub slope: [f64; 3],
    /// Offset per channel.
    pub offset: [f64; 3],
    /// Power per channel.
    pub power: [f64; 3],
    /// Saturation.
    pub saturation: f64,
}

/// A point of a hue curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuePoint {
    /// Hue in degrees, `[0, 360)`.
    pub hue_deg: f64,
    /// The curve value (a chroma multiplier for `hue_vs_sat`, degrees for `hue_vs_hue`).
    pub value: f64,
}

/// A point of the luma curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LumaPoint {
    /// LogC4 luma code value, `[0, 1]`.
    pub luma: f64,
    /// Chroma multiplier.
    pub value: f64,
}

/// A control point of the `luma_curve` tone curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvePoint {
    /// Input: the LogC4 code value of the pixel's luminance ([`LogC4::luma`]), `[0, 1]`.
    pub x: f64,
    /// Output: the LogC4 code value that luminance becomes, `[0, 1]`.
    pub y: f64,
}

/// The JSON form of a parametric look (see the module documentation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParametricLookSpec {
    /// Must be [`SCHEMA_ID`].
    pub schema: String,
    /// Display name.
    pub name: String,
    /// Input encoding; only `logc4` is accepted (and it is the default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<LookInput>,
    /// The luma tone curve, applied first: 2 to 16 points, `v += T(L) - L`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub luma_curve: Vec<CurvePoint>,
    /// The CDL, applied after the luma curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cdl: Option<CdlSpec>,
    /// Chroma multiplier by hue.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hue_vs_sat: Vec<HuePoint>,
    /// Hue rotation in degrees by hue.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hue_vs_hue: Vec<HuePoint>,
    /// Chroma multiplier by luma.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub luma_vs_sat: Vec<LumaPoint>,
    /// Read and ignored: the block of older files that set the strengths of the red chroma step, which
    /// went with the cine rendering. Never written.
    #[serde(default, skip_serializing)]
    pub red_chroma: Option<serde_json::Value>,
    /// Read and ignored: the block of older files that set the strengths of the red density step, which
    /// went with the cine rendering. Never written.
    #[serde(default, skip_serializing)]
    pub red_density: Option<serde_json::Value>,
}

/// A Fritsch-Carlson spline on arbitrary increasing abscissae.
#[derive(Debug, Clone, PartialEq)]
struct Spline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    tangents: Vec<f64>,
}

impl Spline {
    fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let tangents = if xs.len() >= 2 {
            fritsch_carlson_tangents(&xs, &ys)
        } else {
            vec![0.0; xs.len()]
        };
        Self { xs, ys, tangents }
    }

    /// A spline through at least two points that repeats every `period`: the closing segment runs
    /// from the last point to a copy of the first one a period later, and that copy carries the
    /// first point's own tangent, so the curve is C1 at the seam.
    fn new_periodic(mut xs: Vec<f64>, mut ys: Vec<f64>, period: f64) -> Self {
        let mut tangents = fritsch_carlson_tangents_periodic(&xs, &ys, period);
        if let (Some(&x0), Some(&y0), Some(&m0)) = (xs.first(), ys.first(), tangents.first()) {
            xs.push(x0 + period);
            ys.push(y0);
            tangents.push(m0);
        }
        Self { xs, ys, tangents }
    }

    /// Value at `x`, flat outside the first and last abscissa.
    fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 1 || x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let k = self
            .xs
            .partition_point(|&v| v <= x)
            .saturating_sub(1)
            .min(n - 2);
        hermite(
            self.xs[k],
            self.ys[k],
            self.tangents[k],
            self.xs[k + 1],
            self.ys[k + 1],
            self.tangents[k + 1],
            x,
        )
    }
}

/// A curve over the hue circle (period 360 degrees) or the luma axis, with a neutral default.
#[derive(Debug, Clone, PartialEq)]
struct Profile {
    spline: Option<Spline>,
    periodic: bool,
    neutral: f64,
}

impl Profile {
    fn constant(neutral: f64) -> Self {
        Self {
            spline: None,
            periodic: false,
            neutral,
        }
    }

    fn from_points(points: &[(f64, f64)], periodic: bool, neutral: f64) -> Self {
        if points.is_empty() {
            return Self::constant(neutral);
        }
        let periodic = periodic && points.len() >= 2;
        let (xs, ys): (Vec<f64>, Vec<f64>) = points.iter().copied().unzip();
        let spline = if periodic {
            Spline::new_periodic(xs, ys, 360.0)
        } else {
            Spline::new(xs, ys)
        };
        Self {
            spline: Some(spline),
            periodic,
            neutral,
        }
    }

    fn is_neutral(&self) -> bool {
        self.spline.is_none()
    }

    fn eval(&self, x: f64) -> f64 {
        match &self.spline {
            None => self.neutral,
            Some(s) if self.periodic => {
                // Fold x into one period starting at the first control point.
                let start = s.xs.first().copied().unwrap_or(0.0);
                let folded = start + (x - start).rem_euclid(360.0);
                s.eval(folded)
            }
            Some(s) => s.eval(x),
        }
    }
}

fn check_points<T: Copy>(
    section: &'static str,
    points: &[T],
    key: impl Fn(T) -> f64,
    value: impl Fn(T) -> f64,
    key_range: (f64, f64, bool),
    value_range: (f64, f64),
    key_name: &str,
) -> Result<(), LookError> {
    let bad = |reason: String| LookError::BadCurve { section, reason };
    if points.len() > MAX_PROFILE_POINTS {
        return Err(bad(format!(
            "{} points, at most {MAX_PROFILE_POINTS} are allowed",
            points.len()
        )));
    }
    let (lo, hi, hi_inclusive) = key_range;
    let mut prev = f64::NEG_INFINITY;
    for (i, &p) in points.iter().enumerate() {
        let (k, v) = (key(p), value(p));
        if !k.is_finite() || !v.is_finite() {
            return Err(bad(format!("point {i} is not finite")));
        }
        let in_range = k >= lo && (k < hi || (hi_inclusive && k <= hi));
        if !in_range {
            return Err(bad(format!(
                "point {i}: {key_name} {k} is outside {lo} to {hi}"
            )));
        }
        if k <= prev {
            return Err(bad(format!("point {i}: {key_name} {k} does not increase")));
        }
        if v < value_range.0 || v > value_range.1 {
            return Err(bad(format!(
                "point {i}: value {v} is outside {} to {}",
                value_range.0, value_range.1
            )));
        }
        prev = k;
    }
    Ok(())
}

/// A luma tone curve, a CDL and hue-vs-sat, hue-vs-hue and luma-vs-sat curves, evaluated in LogC4.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricLook {
    name: String,
    /// The tone curve on the AWG4 luminance; `None` is neutral (also for points on `y = x`).
    luma_curve: Option<Curve>,
    cdl: Cdl,
    hue_vs_sat: Profile,
    hue_vs_hue: Profile,
    luma_vs_sat: Profile,
}

impl ParametricLook {
    /// The identity look with a name.
    #[must_use]
    pub fn neutral(name: &str) -> Self {
        Self {
            name: name.to_string(),
            luma_curve: None,
            cdl: Cdl::identity(),
            hue_vs_sat: Profile::constant(1.0),
            hue_vs_hue: Profile::constant(0.0),
            luma_vs_sat: Profile::constant(1.0),
        }
    }

    /// Parses and validates a parametric look file.
    ///
    /// # Errors
    ///
    /// [`LookError`]: bad JSON or unknown fields, a different schema, an input other than LogC4,
    /// an out-of-range CDL, or an invalid curve (`luma_curve` included: fewer than 2 or more than 16
    /// points, a point outside the unit square, an `x` that does not increase).
    pub fn from_json(text: &str) -> Result<Self, LookError> {
        let spec: ParametricLookSpec =
            serde_json::from_str(text).map_err(|e| LookError::Json(e.to_string()))?;
        Self::from_spec(&spec)
    }

    /// Validates a spec.
    ///
    /// # Errors
    ///
    /// As [`ParametricLook::from_json`], except the JSON-level errors.
    pub fn from_spec(spec: &ParametricLookSpec) -> Result<Self, LookError> {
        if spec.schema != SCHEMA_ID {
            return Err(LookError::UnsupportedSchema(spec.schema.clone()));
        }
        if let Some(input) = spec.input
            && input != LookInput::LogC4
        {
            return Err(LookError::UnsupportedInput(input));
        }
        let luma_curve = if spec.luma_curve.is_empty() {
            None
        } else {
            let points: Vec<(f64, f64)> = spec.luma_curve.iter().map(|p| (p.x, p.y)).collect();
            let curve = Curve::new(&points).map_err(|e: CurveError| LookError::BadCurve {
                section: "luma_curve",
                reason: e.to_string(),
            })?;
            // Points on y = x are the identity; keep the look exactly neutral then.
            (!curve.is_identity()).then_some(curve)
        };
        let cdl = match &spec.cdl {
            Some(c) => Cdl::new(c.slope, c.offset, c.power, c.saturation)?,
            None => Cdl::identity(),
        };
        // Slope >= 0 and power > 0 make each channel's slope-offset-power rise with its input, so an input
        // of 1.0 gives its largest value on the unit cube.
        for (k, channel) in ["red", "green", "blue"].into_iter().enumerate() {
            let value = (cdl.slope[k] + cdl.offset[k]).max(0.0).powf(cdl.power[k]);
            if value.is_nan() || value >= CDL_OUTPUT_LIMIT {
                return Err(LookError::CdlOutOfRange { channel, value });
            }
        }
        let hue_key = (0.0, 360.0, false);
        check_points(
            "hue_vs_sat",
            &spec.hue_vs_sat,
            |p| p.hue_deg,
            |p| p.value,
            hue_key,
            (0.0, MAX_SATURATION_FACTOR),
            "hue_deg",
        )?;
        check_points(
            "hue_vs_hue",
            &spec.hue_vs_hue,
            |p| p.hue_deg,
            |p| p.value,
            hue_key,
            (-180.0, 180.0),
            "hue_deg",
        )?;
        check_points(
            "luma_vs_sat",
            &spec.luma_vs_sat,
            |p| p.luma,
            |p| p.value,
            (0.0, 1.0, true),
            (0.0, MAX_SATURATION_FACTOR),
            "luma",
        )?;
        let hue_points =
            |pts: &[HuePoint]| pts.iter().map(|p| (p.hue_deg, p.value)).collect::<Vec<_>>();
        let luma_points: Vec<(f64, f64)> =
            spec.luma_vs_sat.iter().map(|p| (p.luma, p.value)).collect();
        Ok(Self {
            name: spec.name.clone(),
            luma_curve,
            cdl,
            hue_vs_sat: Profile::from_points(&hue_points(&spec.hue_vs_sat), true, 1.0),
            hue_vs_hue: Profile::from_points(&hue_points(&spec.hue_vs_hue), true, 0.0),
            luma_vs_sat: Profile::from_points(&luma_points, false, 1.0),
        })
    }

    /// The look's display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The CDL.
    #[must_use]
    pub fn cdl(&self) -> &Cdl {
        &self.cdl
    }

    /// The luma tone curve, if the look has one that is not the identity.
    #[must_use]
    pub fn luma_curve(&self) -> Option<&Curve> {
        self.luma_curve.as_ref()
    }

    /// Whether the look is exactly the identity in LogC4 (no luma curve, identity CDL and no curves).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.luma_curve.is_none()
            && self.cdl.is_identity()
            && self.hue_vs_sat.is_neutral()
            && self.hue_vs_hue.is_neutral()
            && self.luma_vs_sat.is_neutral()
    }

    /// The luma tone curve of the look applied to a LogC4 triple: `v += T(L) - L` on every
    /// channel, `L` the LogC4 code of the AWG4 luminance. A look without the curve returns `v`.
    ///
    /// `T` is exact here (the spline, not a table); outside `[0, 1]` it continues with slope 1.
    /// A shift that is not a finite number (an input far outside any LogC4 signal) counts as 0.
    #[must_use]
    pub fn apply_luma_curve(&self, v: Vec3) -> Vec3 {
        let Some(curve) = &self.luma_curve else {
            return v;
        };
        let l = LogC4(v).luma();
        let t = if l < 0.0 {
            l + curve.eval(0.0)
        } else if l > 1.0 {
            l - 1.0 + curve.eval(1.0)
        } else {
            curve.eval(l)
        };
        let shift = t - l;
        if !shift.is_finite() {
            return v;
        }
        [v[0] + shift, v[1] + shift, v[2] + shift]
    }

    /// Evaluates the look exactly (no LUT) on a LogC4 triple: the luma curve, the CDL, then the hue
    /// and luma-vs-saturation curves.
    #[must_use]
    pub fn evaluate(&self, v: Vec3) -> Vec3 {
        let v1 = self.cdl.apply(self.apply_luma_curve(v));
        if self.hue_vs_sat.is_neutral()
            && self.hue_vs_hue.is_neutral()
            && self.luma_vs_sat.is_neutral()
        {
            return v1;
        }
        let w = REC709_LUMA_WEIGHTS;
        let luma = w[0] * v1[0] + w[1] * v1[1] + w[2] * v1[2];
        let mean = (v1[0] + v1[1] + v1[2]) / 3.0;
        let p = v1[0] * E1[0] + v1[1] * E1[1] + v1[2] * E1[2];
        let q = v1[0] * E2[0] + v1[1] * E2[1] + v1[2] * E2[2];
        let chroma = p.hypot(q);
        if chroma < 1e-12 {
            return v1;
        }
        let angle = q.atan2(p);
        let hue_deg = angle.to_degrees().rem_euclid(360.0);
        let factor = self.hue_vs_sat.eval(hue_deg) * self.luma_vs_sat.eval(luma);
        let rotated = angle + self.hue_vs_hue.eval(hue_deg).to_radians();
        let (p2, q2) = (
            chroma * factor * rotated.cos(),
            chroma * factor * rotated.sin(),
        );
        let mut out = [
            mean + p2 * E1[0] + q2 * E2[0],
            mean + p2 * E1[1] + q2 * E2[1],
            mean + p2 * E1[2] + q2 * E2[2],
        ];
        // Restore the luma of the CDL output so the curves only move colour.
        let luma_out = w[0] * out[0] + w[1] * out[1] + w[2] * out[2];
        let fix = luma - luma_out;
        for c in &mut out {
            *c += fix;
        }
        out
    }

    /// Bakes the look into a `size`-point cube over `[0, 1]^3` (LogC4 code values).
    ///
    /// # Errors
    ///
    /// [`LookError::Lut`] for a size below 2 or a non-finite result.
    pub fn bake(&self, size: usize) -> Result<Lut3d, LookError> {
        Ok(Lut3d::bake(size, [0.0; 3], [1.0; 3], |rgb| {
            self.evaluate(rgb)
        })?)
    }

    /// Bakes the look into the 65-point cube used at load time and wraps it as a LogC4 look.
    ///
    /// # Errors
    ///
    /// As [`ParametricLook::bake`].
    pub fn to_transform(&self) -> Result<LookTransform, LookError> {
        if self.is_identity() {
            return Ok(LookTransform::neutral());
        }
        let cube = self.bake(BAKE_SIZE)?;
        let lut = Lut {
            title: Some(self.name.clone()),
            shaper: None,
            cube,
        };
        LookTransform::new(LookInput::LogC4, None, Some(lut))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The input the nightly fuzzer found on 2026-10-17 (`fuzz/corpus/look_json/cdl-overflow.json`): the
    /// blue channel's slope 5.9 and power 1405 send white to infinity, and the CDL's saturation then made
    /// `inf - inf`, a NaN, out of white.
    const CDL_OVERFLOW: &str = include_str!("../fuzz/corpus/look_json/cdl-overflow.json");

    #[test]
    fn a_cdl_that_overflows_at_white_is_refused_with_its_channel() {
        match ParametricLook::from_json(CDL_OVERFLOW) {
            Err(LookError::CdlOutOfRange { channel, value }) => {
                assert_eq!(channel, "blue");
                assert!(value.is_infinite());
            }
            other => panic!("expected CdlOutOfRange, got {other:?}"),
        }
        // Just under the limit is accepted and evaluates without a NaN anywhere on the cube's corners.
        let near = CDL_OVERFLOW.replace("5.90 ", "1.0").replace("1405", "2.9");
        let look = ParametricLook::from_json(&near).expect("a strong but finite CDL is a look");
        for corner in 0..8 {
            let rgb = std::array::from_fn(|k| f64::from((corner >> k) & 1));
            assert!(look.evaluate(rgb).iter().all(|v| v.is_finite()), "{rgb:?}");
        }
        // Blue offset -0.01: 9.99^3.5 is about 3150, over the limit (9.99^3 = 997 would pass).
        let over = CDL_OVERFLOW.replace("5.90 ", "10.0").replace("1405", "3.5");
        assert!(matches!(
            ParametricLook::from_json(&over),
            Err(LookError::CdlOutOfRange {
                channel: "blue",
                ..
            })
        ));
    }

    #[test]
    fn neutral_is_identity() {
        let n = LookTransform::neutral();
        assert!(n.is_identity());
        let v = LogC4([0.3, 0.5, 0.1]);
        assert_eq!(n.apply_logc4(v, 100.0).unwrap(), v);
        assert_eq!(n.eval_logc4(v).unwrap(), v);
    }

    #[test]
    fn blend_endpoints_are_exact() {
        let a = [0.2, 0.4, 0.6];
        let b = [0.9, 0.1, 0.3];
        assert_eq!(blend_display_linear(a, b, 0.0), a);
        assert_eq!(blend_display_linear(a, b, 100.0), b);
        assert_eq!(blend_display_linear(a, b, -5.0), a);
        assert_eq!(blend_display_linear(a, b, 500.0), b);
        assert_eq!(blend_display_linear(a, b, f64::NAN), a);
        let m = blend_display_linear(a, b, 50.0);
        assert!((m[0] - 0.55).abs() < 1e-15);
    }

    #[test]
    fn schema_is_enforced() {
        let json = r#"{"schema":"other/1","name":"x"}"#;
        assert!(matches!(
            ParametricLook::from_json(json),
            Err(LookError::UnsupportedSchema(_))
        ));
        let neutral = format!(r#"{{"schema":"{SCHEMA_ID}","name":"Neutral"}}"#);
        let look = ParametricLook::from_json(&neutral).unwrap();
        assert!(look.is_identity());
        assert!(look.to_transform().unwrap().is_identity());
    }
}
