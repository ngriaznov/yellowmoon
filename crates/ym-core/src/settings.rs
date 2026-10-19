//! `DevelopSettings`: every control in the UI and every CLI flag as one field, the patch that
//! `update_settings` applies, and `ResolvedSettings`, the settings with every `Auto` resolved
//! against the photo's automatic estimates (`docs/01-architecture.md` section DevelopSettings).
//!
//! # One JSON shape
//!
//! The serde shape of every type here is the shape of `src/api/types.ts` (snake_case fields,
//! tagged unions with a `kind` tag), so the same JSON is
//!
//! * what crosses the bridge to the webview,
//! * the `ym:Settings` property of an export's XMP packet,
//! * what `ym render --settings <file>` reads (`crates/ym-core/schema/settings.schema.json` is its
//!   JSON Schema), and
//! * what the golden-image manifest stores.
//!
//! Reading is lenient about *missing* fields (they take their defaults, so a settings file may
//! hold only the fields it changes) and strict about *unknown* ones (a typo such as `exposur` is an
//! error, not a silent no-op). Writing always writes every field.
//!
//! # Auto and Manual
//!
//! `exposure`, `contrast`, `highlights`, `shadows`, `black`, `white_balance`, `saturation`, the
//! shadows and highlights colour-balance wheels, `sharpen` and `noise_reduction` are [`AutoOr`]:
//! `Auto` resolves against [`AutoEstimates`] when the photo is rendered, `Manual` is the user's
//! value. Auto contrast, highlights, shadows, black and saturation are 0, and the wheels `[0, 0]`,
//! unless Apple Auto answered for the photo (`docs/19` section 5); a plain number (or point) in an
//! older document reads as Manual, and 0 (or `[0, 0]`) as Auto. A patch that carries a `Manual`
//! value is what "moving a slider" sends; a patch that carries `Auto` is Reset.
//! [`ResolvedSettings`] answers what the panel shows: every `Auto` replaced by the number it
//! resolves to, and which fields are still automatic (the AUTO and EDITED tags).
//!
//! # Ranges
//!
//! Values are clamped, never rejected, when they are numbers outside their range
//! ([`PatchReport::clamped`] lists them); a value that is not a number, a curve that is not a curve
//! and a rotation that is not a quarter turn are errors ([`SettingsError`]). A step (0.01 EV, 50 K)
//! is the UI's granularity: the core keeps the value it is given, so `--wb 5025` means 5025 K.
//! The colour warp keeps at most [`WARP_MAX_PINS`] pins (later ones are cut) and clamps each pin the
//! way `ym_color::warp::WarpPin::clamped` does ([`ColorWarp`]).

use serde::{Deserialize, Deserializer, Serialize};
use ym_color::chain::ChainError;
use ym_color::exposure::ExposureSetting as ColorExposure;
use ym_color::exposure::white::OFFSET_RANGE_EV;
use ym_color::warp::{WARP_COORD_LIMIT, WARP_MAX_PINS, WARP_RANGE_DEFAULT, WarpParams, WarpPin};
use ym_color::wb::{LightSource, WbReadout, WhiteBalance as ColorWhiteBalance};
use ym_looks::library::LookId;
use ym_profiles::profile::Profile;

use crate::auto::{AutoEstimates, AutoSummary};
use crate::session::PhotoId;

// ---------------------------------------------------------------------------------------------
// Ranges (docs/01 table)
// ---------------------------------------------------------------------------------------------

/// Range of `look_intensity` (percent).
pub const LOOK_INTENSITY_RANGE: (f64, f64) = (0.0, 100.0);
/// Range of a manual exposure (stops, absolute). The UI steps by 0.01.
pub const EXPOSURE_RANGE_EV: (f64, f64) = (-4.0, 4.0);
/// Range of contrast, brightness, highlights, shadows, white, black and saturation.
pub const PERCENT_RANGE: (f64, f64) = (-100.0, 100.0);
/// Range of a manual Kelvin. The UI steps by 50 K.
pub const KELVIN_RANGE_K: (f64, f64) = (2000.0, 11000.0);
/// Range of a manual tint (positive is magenta).
pub const TINT_RANGE: (f64, f64) = (-50.0, 50.0);
/// Range of the manual sharpening amount.
pub const SHARPEN_AMOUNT_RANGE: (f64, f64) = (0.0, 300.0);
/// Range of the manual sharpening radius (pixels).
pub const SHARPEN_RADIUS_RANGE_PX: (f64, f64) = (0.3, 2.5);
/// Range of the output sharpening amount (percent): Detail › Sharpening › Output. The UI steps by 5.
pub const OUTPUT_SHARPENING_RANGE: (f64, f64) = (0.0, 200.0);
/// Default output sharpening amount (percent). 0 creates no stage (`docs/algorithms/04`).
pub const OUTPUT_SHARPENING_DEFAULT: f64 = 100.0;
/// Range of the white lift offset (stops, added to the automatic amount): Light › White lift.
/// The UI steps by 0.05. A negative offset of the automatic amount (or beyond) gives no lift.
pub const WHITE_OFFSET_RANGE_EV: (f64, f64) = (-OFFSET_RANGE_EV, OFFSET_RANGE_EV);
/// Range of each manual noise-reduction strength.
pub const NOISE_REDUCTION_RANGE: (f64, f64) = (0.0, 100.0);
/// Range of both amounts of RAW › Range (`rolloff`): the highlight range and the low-light toe.
pub const ROLLOFF_RANGE: (f64, f64) = (0.0, 1.0);
/// Range of the crop's straighten angle (degrees).
pub const CROP_ANGLE_RANGE_DEG: (f64, f64) = (-45.0, 45.0);
/// The most control points one curve may have.
pub const MAX_CURVE_POINTS: usize = 16;
/// The fewest control points one curve needs (its two end points).
pub const MIN_CURVE_POINTS: usize = 2;
/// The smallest side of a crop rectangle, as a fraction of the photo.
pub const MIN_CROP_SIDE: f64 = 1e-3;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a settings document or a patch was refused. Nothing is applied when this is returned.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SettingsError {
    /// A number is not finite.
    #[error("The value for {0} is not a number.")]
    NotFinite(&'static str),
    /// A curve cannot be used.
    #[error("The {channel} curve {problem}.")]
    Curve {
        /// `luma`, `red`, `green` or `blue`.
        channel: &'static str,
        /// What is wrong, as a verb phrase ("has more than 16 points").
        problem: String,
    },
    /// A crop rectangle has no area.
    #[error("The crop has no area.")]
    EmptyCrop,
    /// A rotation that is not a multiple of 90 degrees.
    #[error("Rotation must be 0, 90, 180 or 270 degrees, not {0}.")]
    Orientation(i64),
    /// A partial manual value was sent for an automatic field and there are no automatic values to
    /// fill the missing part from.
    #[error(
        "The {0} patch leaves out part of a manual value, and this photo has no automatic estimates yet to fill it from."
    )]
    NeedsEstimates(&'static str),
    /// The automatic values could not be turned into a colour chain.
    #[error("The colour settings can't be used: {0}")]
    Chain(String),
    /// Pick white refused the spot (too bright, too dark, or outside the photo); the text is the
    /// message to show.
    #[error("{0}")]
    WhitePick(String),
    /// A settings document is not valid JSON of the right shape (a misspelt field included).
    #[error("The settings can't be read: {0}")]
    Parse(String),
}

// ---------------------------------------------------------------------------------------------
// Fields and stage groups
// ---------------------------------------------------------------------------------------------

/// The render-graph group of stages a setting first affects (`docs/01` section Render graph).
///
/// A change invalidates its group and every later one: `Raw` before `Colour` before `Grade`.
/// `Geometry` is applied when sampling for display and export, so no pixel stage re-runs, and
/// `Output` (sharpening) is the last: it re-runs only its own pass over the rendered frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageGroup {
    /// Vignetting, highlights, demosaic and denoise, detail controls, sharpening, lens resample
    /// (R3 to R9).
    Raw,
    /// White balance, sensor map, ARRI matrix, AWG4, exposure, highlight range and toe, LogC4 (C1
    /// to C8).
    Color,
    /// Primary grade, look, curve, display transform and encode (C9 to C13).
    Grade,
    /// Crop, straighten and orientation.
    Geometry,
    /// Output sharpening of the rendered display-linear frame (`docs/01` section Render graph).
    /// It follows geometry because it works on the pixels at the size they are shown or written,
    /// and nothing before it re-runs when it changes.
    Output,
}

/// One field of [`DevelopSettings`], for change reports and Reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingsField {
    /// `look`.
    Look,
    /// `look_intensity`.
    LookIntensity,
    /// `exposure`.
    Exposure,
    /// `contrast`.
    Contrast,
    /// `brightness`.
    Brightness,
    /// `highlights`.
    Highlights,
    /// `shadows`.
    Shadows,
    /// `white`.
    White,
    /// `white_offset_ev`.
    #[serde(rename = "white_offset_ev")]
    WhiteOffset,
    /// `black`.
    Black,
    /// `light_source`.
    LightSource,
    /// `white_balance`.
    WhiteBalance,
    /// `color_balance`.
    ColorBalance,
    /// `saturation`.
    Saturation,
    /// `color_warp`.
    ColorWarp,
    /// `sharpen`.
    Sharpen,
    /// `output_sharpening`.
    OutputSharpening,
    /// `noise_reduction`.
    NoiseReduction,
    /// `curve`.
    Curve,
    /// `decode`.
    Decode,
    /// `rolloff`.
    Rolloff,
    /// `crop`.
    Crop,
    /// `orientation`.
    Orientation,
    /// `white_pick`.
    WhitePick,
}

impl SettingsField {
    /// Every field, in the order of the settings table.
    pub const ALL: [SettingsField; 24] = [
        SettingsField::Look,
        SettingsField::LookIntensity,
        SettingsField::Exposure,
        SettingsField::Contrast,
        SettingsField::Brightness,
        SettingsField::Highlights,
        SettingsField::Shadows,
        SettingsField::White,
        SettingsField::WhiteOffset,
        SettingsField::Black,
        SettingsField::LightSource,
        SettingsField::WhiteBalance,
        SettingsField::ColorBalance,
        SettingsField::Saturation,
        SettingsField::ColorWarp,
        SettingsField::Sharpen,
        SettingsField::OutputSharpening,
        SettingsField::NoiseReduction,
        SettingsField::Curve,
        SettingsField::Decode,
        SettingsField::Rolloff,
        SettingsField::Crop,
        SettingsField::Orientation,
        SettingsField::WhitePick,
    ];

    /// The JSON name of the field.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Look => "look",
            Self::LookIntensity => "look_intensity",
            Self::Exposure => "exposure",
            Self::Contrast => "contrast",
            Self::Brightness => "brightness",
            Self::Highlights => "highlights",
            Self::Shadows => "shadows",
            Self::White => "white",
            Self::WhiteOffset => "white_offset_ev",
            Self::Black => "black",
            Self::LightSource => "light_source",
            Self::WhiteBalance => "white_balance",
            Self::ColorBalance => "color_balance",
            Self::Saturation => "saturation",
            Self::ColorWarp => "color_warp",
            Self::Sharpen => "sharpen",
            Self::OutputSharpening => "output_sharpening",
            Self::NoiseReduction => "noise_reduction",
            Self::Curve => "curve",
            Self::Decode => "decode",
            Self::Rolloff => "rolloff",
            Self::Crop => "crop",
            Self::Orientation => "orientation",
            Self::WhitePick => "white_pick",
        }
    }

    /// The first group of render stages this field affects (`docs/01` section Render graph).
    #[must_use]
    pub fn group(self) -> StageGroup {
        match self {
            Self::Decode | Self::NoiseReduction | Self::Sharpen => StageGroup::Raw,
            Self::LightSource
            | Self::WhiteBalance
            | Self::WhitePick
            | Self::Exposure
            | Self::Rolloff => StageGroup::Color,
            Self::Crop | Self::Orientation => StageGroup::Geometry,
            Self::OutputSharpening => StageGroup::Output,
            Self::Look
            | Self::LookIntensity
            | Self::Contrast
            | Self::Brightness
            | Self::Highlights
            | Self::Shadows
            | Self::White
            | Self::WhiteOffset
            | Self::Black
            | Self::ColorBalance
            | Self::Saturation
            | Self::ColorWarp
            | Self::Curve => StageGroup::Grade,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------------------------

/// A value that is either resolved from the automatic estimates or set by hand.
///
/// JSON: `{ "kind": "auto" }` or `{ "kind": "manual", ...fields }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AutoOr<M> {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual(M),
}

impl<M> AutoOr<M> {
    /// True for `Auto`.
    #[must_use]
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The manual value, if there is one.
    #[must_use]
    pub fn manual(&self) -> Option<&M> {
        match self {
            Self::Auto => None,
            Self::Manual(m) => Some(m),
        }
    }
}

/// A manual exposure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureValue {
    /// Absolute exposure in stops, -4 to +4.
    pub ev: f64,
}

/// A manual contrast, highlights, shadows, black or saturation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PercentValue {
    /// -100 to 100.
    pub value: f64,
}

/// `contrast`, `highlights`, `shadows`, `black` or `saturation`: Auto (0, or Apple Auto's value for
/// the photo, `docs/19` section 5) or a manual value from -100 to 100. JSON: `{ "kind": "auto" }` or
/// `{ "kind": "manual", "value": 12 }`; an older document may give a plain number, which reads as
/// Manual with that value, except 0, which reads as Auto (0 was the default).
pub type PercentSetting = AutoOr<PercentValue>;
/// `contrast`: Auto or a manual value ([`PercentSetting`]).
pub type ContrastSetting = PercentSetting;
/// `saturation`: Auto or a manual value ([`PercentSetting`]).
pub type SaturationSetting = PercentSetting;

impl PercentSetting {
    /// A manual value.
    #[must_use]
    pub fn manual_value(value: f64) -> Self {
        Self::Manual(PercentValue { value })
    }

    /// The value in effect: the manual value, or `auto` (the photo's Auto value) while the setting is
    /// Auto.
    #[must_use]
    pub fn resolve(&self, auto: f64) -> f64 {
        match self {
            Self::Auto => auto,
            Self::Manual(m) => m.value,
        }
    }
}

/// Reads `contrast`, `highlights`, `shadows`, `black` or `saturation` from a settings document, the
/// `ym:Settings` of a file or a golden manifest: the `AutoOr` form, or a plain number from a document
/// written before Apple Auto (or, for highlights, shadows and black, before its prompt version 2),
/// which reads as Manual with that value, except 0, which reads as Auto ([`percent_from_value`]).
fn percent_document<'de, D>(d: D) -> Result<PercentSetting, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(d)?;
    percent_from_value(value).map_err(serde::de::Error::custom)
}

/// As [`percent_document`] for a patch: a field that is absent or `null` is no change (`None`).
fn percent_patch<'de, D>(d: D) -> Result<Option<PercentSetting>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<serde_json::Value>::deserialize(d)?
        .map(|value| percent_from_value(value).map_err(serde::de::Error::custom))
        .transpose()
}

/// The setting of one JSON value: a number is the old form (0 is Auto, any other number Manual; one
/// out of range is clamped later like any manual value), an object the `AutoOr` form (`{ "kind": ...
/// }`, unknown fields refused).
fn percent_from_value(value: serde_json::Value) -> Result<PercentSetting, String> {
    match value.as_f64() {
        Some(0.0) => Ok(AutoOr::Auto),
        Some(v) => Ok(PercentSetting::manual_value(v)),
        None => serde_json::from_value(value).map_err(|e| e.to_string()),
    }
}

/// A manual white balance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteBalanceValue {
    /// Colour temperature, 2000 to 11000 K.
    pub kelvin: f64,
    /// Tint, -50 to 50, positive towards magenta.
    pub tint: f64,
}

/// Manual capture sharpening.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharpenValue {
    /// Amount, 0 to 300.
    pub amount: f64,
    /// PSF radius in pixels, 0.3 to 2.5.
    pub radius_px: f64,
}

/// Manual noise reduction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseReductionValue {
    /// Luminance strength, 0 to 100.
    pub luminance: f64,
    /// Colour strength, 0 to 100.
    pub color: f64,
    /// Detail strength, 0 to 100.
    pub detail: f64,
}

/// `exposure`: Auto or an absolute value in stops.
pub type ExposureSetting = AutoOr<ExposureValue>;
/// `white_balance`: Auto or Kelvin and tint.
pub type WhiteBalanceSetting = AutoOr<WhiteBalanceValue>;
/// `sharpen`: Auto or amount and radius.
pub type SharpenSetting = AutoOr<SharpenValue>;
/// `noise_reduction`: Auto or the three strengths.
pub type NoiseReductionSetting = AutoOr<NoiseReductionValue>;

impl ExposureSetting {
    /// A manual exposure in stops.
    #[must_use]
    pub fn manual_ev(ev: f64) -> Self {
        Self::Manual(ExposureValue { ev })
    }

    /// The same setting as `ym-color`'s type, for the colour chain.
    #[must_use]
    pub fn to_color(&self) -> ColorExposure {
        match self {
            Self::Auto => ColorExposure::Auto,
            Self::Manual(m) => ColorExposure::Manual { ev: m.ev },
        }
    }
}

impl WhiteBalanceSetting {
    /// A manual white balance.
    #[must_use]
    pub fn manual_kelvin(kelvin: f64, tint: f64) -> Self {
        Self::Manual(WhiteBalanceValue { kelvin, tint })
    }

    /// The same setting as `ym-color`'s type, for the colour chain.
    #[must_use]
    pub fn to_color(&self) -> ColorWhiteBalance {
        match self {
            Self::Auto => ColorWhiteBalance::Auto,
            Self::Manual(m) => ColorWhiteBalance::Manual {
                kelvin: m.kelvin,
                tint: m.tint,
            },
        }
    }
}

impl SharpenSetting {
    /// A manual sharpening.
    #[must_use]
    pub fn manual_values(amount: f64, radius_px: f64) -> Self {
        Self::Manual(SharpenValue { amount, radius_px })
    }
}

impl NoiseReductionSetting {
    /// A manual noise reduction.
    #[must_use]
    pub fn manual_values(luminance: f64, color: f64, detail: f64) -> Self {
        Self::Manual(NoiseReductionValue {
            luminance,
            color,
            detail,
        })
    }
}

/// A point in the colour-balance disc: `[a, b]`, each within the unit disc.
pub type ColorBalancePoint = [f64; 2];

/// A manual shadows or highlights wheel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BalanceValue {
    /// The point `[a, b]` in the unit disc.
    pub point: ColorBalancePoint,
}

/// The shadows or highlights colour-balance wheel: Auto (`[0, 0]`, or the colour Apple Auto gave
/// that zone, `docs/19` section 5) or a manual point. JSON: `{ "kind": "auto" }` or
/// `{ "kind": "manual", "point": [0.1, -0.2] }`; an older document may give a plain point, which
/// reads as Manual at that point, except `[0, 0]`, which reads as Auto (the centre was the default).
pub type BalanceSetting = AutoOr<BalanceValue>;

impl BalanceSetting {
    /// A manual point.
    #[must_use]
    pub fn manual_point(point: ColorBalancePoint) -> Self {
        Self::Manual(BalanceValue { point })
    }

    /// The point in effect: the manual point, or `auto` (the photo's Auto point) while Auto.
    #[must_use]
    pub fn resolve(&self, auto: ColorBalancePoint) -> ColorBalancePoint {
        match self {
            Self::Auto => auto,
            Self::Manual(m) => m.point,
        }
    }
}

/// Reads a shadows or highlights wheel from a settings document: the `AutoOr` form, or a plain
/// point from a document written before Apple Auto's prompt version 2 ([`balance_from_value`]).
fn balance_document<'de, D>(d: D) -> Result<BalanceSetting, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(d)?;
    balance_from_value(value).map_err(serde::de::Error::custom)
}

/// As [`balance_document`] for a patch: a field that is absent or `null` is no change (`None`).
fn balance_patch<'de, D>(d: D) -> Result<Option<BalanceSetting>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<serde_json::Value>::deserialize(d)?
        .map(|value| balance_from_value(value).map_err(serde::de::Error::custom))
        .transpose()
}

/// The wheel of one JSON value: an array is the old form (`[0, 0]` is Auto, any other point Manual;
/// one outside the disc is scaled back later like any manual point), an object the `AutoOr` form.
fn balance_from_value(value: serde_json::Value) -> Result<BalanceSetting, String> {
    if value.is_array() {
        let point: ColorBalancePoint = serde_json::from_value(value).map_err(|e| e.to_string())?;
        return Ok(if point == [0.0, 0.0] {
            AutoOr::Auto
        } else {
            BalanceSetting::manual_point(point)
        });
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Colour balance wheels (shadows, midtones, highlights). The shadows and highlights wheels are
/// Auto or a manual point ([`BalanceSetting`], Auto by default); the midtones wheel is a plain point,
/// `[0, 0]` by default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ColorBalance {
    /// The shadows wheel: Auto or a point.
    #[serde(deserialize_with = "balance_document")]
    pub shadows: BalanceSetting,
    /// The midtones wheel.
    pub midtones: ColorBalancePoint,
    /// The highlights wheel: Auto or a point.
    #[serde(deserialize_with = "balance_document")]
    pub highlights: BalanceSetting,
}

impl Default for ColorBalance {
    fn default() -> Self {
        Self {
            shadows: AutoOr::Auto,
            midtones: [0.0, 0.0],
            highlights: AutoOr::Auto,
        }
    }
}

/// The three wheels' points in effect ([`ResolvedValues::color_balance`]).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ColorBalancePoints {
    /// The shadows wheel.
    pub shadows: ColorBalancePoint,
    /// The midtones wheel.
    pub midtones: ColorBalancePoint,
    /// The highlights wheel.
    pub highlights: ColorBalancePoint,
}

/// A point of the colour warp's chroma plane, `[a, b]`: the chroma of a LogC4 pixel along the
/// colour-balance axes of C9 (`a` red against cyan, `b` green against blue; `docs/02` C9b).
pub type ChromaPoint = [f64; 2];

/// One pin of Colour › Colour warp (`docs/02` C9b): it moves the colours near `source` to `target`.
///
/// JSON: every field but `source` may be left out when reading (`target` is then `source`, `locked`
/// false, `range` 0.04, `shadows` and `highlights` 1, `pivot_ev` and `exposure_ev` 0); unknown fields
/// are refused, and writing always writes every field. The values are clamped when the settings are
/// checked ([`ColorWarp`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "ColorWarpPinDocument")]
pub struct ColorWarpPin {
    /// The colour the pin takes hold of.
    pub source: ChromaPoint,
    /// Where that colour moves to; a locked pin's target is its source.
    pub target: ChromaPoint,
    /// A locked pin holds the colours within its range in place.
    pub locked: bool,
    /// How far the pin reaches across the plane, 0.01 to 0.25 (chroma units).
    pub range: f64,
    /// How much of the move applies to the shadows, 0 to 1.
    pub shadows: f64,
    /// How much of the move applies to the highlights, 0 to 1.
    pub highlights: f64,
    /// Where the shadows end and the highlights begin, in stops from middle grey, -6 to +6.
    pub pivot_ev: f64,
    /// A change of brightness of the pin's colours, in stops, -2 to +2.
    pub exposure_ev: f64,
}

/// A pin as a document may write it: everything but `source` has a default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorWarpPinDocument {
    source: ChromaPoint,
    #[serde(default, deserialize_with = "present")]
    target: Option<ChromaPoint>,
    #[serde(default)]
    locked: bool,
    #[serde(default = "default_warp_range")]
    range: f64,
    #[serde(default = "one")]
    shadows: f64,
    #[serde(default = "one")]
    highlights: f64,
    #[serde(default)]
    pivot_ev: f64,
    #[serde(default)]
    exposure_ev: f64,
}

/// A field that may be left out but not `null`: present means `Some`.
fn present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(d).map(Some)
}

fn default_warp_range() -> f64 {
    WARP_RANGE_DEFAULT
}

fn one() -> f64 {
    1.0
}

impl From<ColorWarpPinDocument> for ColorWarpPin {
    fn from(d: ColorWarpPinDocument) -> Self {
        Self {
            source: d.source,
            target: d.target.unwrap_or(d.source),
            locked: d.locked,
            range: d.range,
            shadows: d.shadows,
            highlights: d.highlights,
            pivot_ev: d.pivot_ev,
            exposure_ev: d.exposure_ev,
        }
    }
}

impl From<ColorWarpPin> for WarpPin {
    fn from(p: ColorWarpPin) -> Self {
        Self {
            source: p.source,
            target: p.target,
            locked: p.locked,
            range: p.range,
            shadows: p.shadows,
            highlights: p.highlights,
            pivot_ev: p.pivot_ev,
            exposure_ev: p.exposure_ev,
        }
    }
}

impl From<WarpPin> for ColorWarpPin {
    fn from(p: WarpPin) -> Self {
        Self {
            source: p.source,
            target: p.target,
            locked: p.locked,
            range: p.range,
            shadows: p.shadows,
            highlights: p.highlights,
            pivot_ev: p.pivot_ev,
            exposure_ev: p.exposure_ev,
        }
    }
}

/// Colour › Colour warp: its pins, in order; no pins by default (the stage is then skipped).
///
/// At most [`WARP_MAX_PINS`]: checking the settings cuts the list there and clamps each pin as
/// `WarpPin::clamped` does (points scaled back to `|c| <= 0.5`, the range to 0.01…0.25, the tonal
/// weights to 0…1, the pivot to ±6 and the exposure to ±2 stops; a locked pin's target is its source,
/// and its weights, pivot and exposure are the defaults). A patch replaces the whole list.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ColorWarp {
    /// The pins.
    pub pins: Vec<ColorWarpPin>,
}

impl ColorWarp {
    /// The same pins as `ym-color`'s parameters, for stage C9b.
    #[must_use]
    pub fn to_color(&self) -> WarpParams {
        WarpParams {
            pins: self.pins.iter().copied().map(WarpPin::from).collect(),
        }
    }
}

/// A curve control point, `[x, y]` within [0, 1]².
pub type CurvePoint = [f64; 2];

/// The identity curve: `(0, 0)` and `(1, 1)`.
#[must_use]
pub fn identity_curve() -> Vec<CurvePoint> {
    vec![[0.0, 0.0], [1.0, 1.0]]
}

/// Luma plus R, G and B curves, 2 to 16 control points each, `x` strictly increasing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Curves {
    /// The luma curve.
    pub luma: Vec<CurvePoint>,
    /// The red curve.
    pub red: Vec<CurvePoint>,
    /// The green curve.
    pub green: Vec<CurvePoint>,
    /// The blue curve.
    pub blue: Vec<CurvePoint>,
}

impl Default for Curves {
    fn default() -> Self {
        Self {
            luma: identity_curve(),
            red: identity_curve(),
            green: identity_curve(),
            blue: identity_curve(),
        }
    }
}

/// RAW › Decode switches; all true by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DecodeOptions {
    /// Neural demosaic and denoise (off: the classical path, also for export).
    pub neural_demosaic: bool,
    /// Highlight reconstruction (R5).
    pub highlight_recovery: bool,
    /// Vignetting and distortion corrections from the camera's tags.
    pub lens_corrections: bool,
    /// Lateral chromatic-aberration correction.
    pub chromatic_aberration: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            neural_demosaic: true,
            highlight_recovery: true,
            lens_corrections: true,
            chromatic_aberration: true,
        }
    }
}

/// RAW › Range (`docs/02` C7), each 0 to 1: Highlights 1 and Low light 0.5 by default. The JSON name
/// `rolloff` is kept from the earlier roll-off card.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Rolloff {
    /// The highlight range of C7 (`docs/02`): where the camera's clipping point lands, from where
    /// it was recorded (0) to white, the clip of an ALEXA at EI 800 (1); default 1. Mid grey and
    /// everything under the knee, one stop below the camera's clip, never move.
    pub highlights: f64,
    /// Low-light toe strength (C7).
    pub low_light: f64,
}

impl Default for Rolloff {
    fn default() -> Self {
        Self {
            highlights: 1.0,
            low_light: 0.5,
        }
    }
}

/// A rectangle in image coordinates normalised to [0, 1]².
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormRect {
    /// Left edge.
    pub x: f64,
    /// Top edge.
    pub y: f64,
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl NormRect {
    /// The whole photo.
    pub const FULL: NormRect = NormRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };
}

/// Crop aspect presets (Free, Original, 1:1, 4:5, 16:9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropAspect {
    /// No constraint.
    Free,
    /// The photo's own shape.
    Original,
    /// 1:1.
    Square,
    /// 4:5 (5:4 on a landscape frame).
    FourByFive,
    /// 16:9.
    SixteenByNine,
}

/// The crop (per photo, in memory only): a rectangle, a straighten angle and an aspect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crop {
    /// The frame, in the coordinates of the straightened photo, normalised.
    pub rect: NormRect,
    /// Straighten angle, -45 to 45 degrees.
    pub angle_deg: f64,
    /// The aspect preset the frame follows.
    pub aspect: CropAspect,
}

/// Orientation applied after the camera's EXIF orientation: a number of quarter turns, clockwise.
///
/// JSON: the number of degrees (0, 90, 180 or 270). Any multiple of 90 is accepted when reading
/// and normalised (-90 is 270, 360 is 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "i64", into = "u16")]
pub enum Orientation {
    /// No extra rotation.
    #[default]
    Deg0,
    /// 90 degrees clockwise.
    Deg90,
    /// 180 degrees.
    Deg180,
    /// 270 degrees clockwise (90 counter-clockwise).
    Deg270,
}

impl Orientation {
    /// The rotation in degrees: 0, 90, 180 or 270.
    #[must_use]
    pub fn degrees(self) -> u16 {
        match self {
            Self::Deg0 => 0,
            Self::Deg90 => 90,
            Self::Deg180 => 180,
            Self::Deg270 => 270,
        }
    }

    /// One more quarter turn clockwise (the Rotate 90 button).
    #[must_use]
    pub fn rotated_cw(self) -> Self {
        match self {
            Self::Deg0 => Self::Deg90,
            Self::Deg90 => Self::Deg180,
            Self::Deg180 => Self::Deg270,
            Self::Deg270 => Self::Deg0,
        }
    }
}

impl TryFrom<i64> for Orientation {
    type Error = String;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        if value % 90 != 0 {
            return Err(SettingsError::Orientation(value).to_string());
        }
        Ok(match value.rem_euclid(360) {
            0 => Self::Deg0,
            90 => Self::Deg90,
            180 => Self::Deg180,
            _ => Self::Deg270,
        })
    }
}

impl From<Orientation> for u16 {
    fn from(o: Orientation) -> Self {
        o.degrees()
    }
}

// ---------------------------------------------------------------------------------------------
// DevelopSettings
// ---------------------------------------------------------------------------------------------

/// Every control in the UI and every CLI flag maps to one field. Defaults reproduce the automatic
/// development (`docs/01-architecture.md` section DevelopSettings).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DevelopSettings {
    /// Look tab. The user's default look (Signature out of the box; a photo a session opens gets it
    /// through [`DevelopSettings::with_look`]).
    pub look: LookId,
    /// 0 to 100 percent.
    pub look_intensity: f64,
    /// Light › Exposure: Auto or an absolute value, -4 to +4 EV.
    pub exposure: ExposureSetting,
    /// Light › Exposure and Look › Tone: Auto (0, or Apple Auto's contrast, `docs/19` section 5) or a
    /// manual value, -100 to 100: a tone curve on the luma about mid grey that keeps black and the top
    /// code value (docs/02 C9). Auto is the default; a plain number in an older document reads as
    /// Manual with that value, and 0 as Auto.
    #[serde(deserialize_with = "percent_document")]
    pub contrast: ContrastSetting,
    /// -100 to 100.
    pub brightness: f64,
    /// Light › High dynamic range: Auto (0, or Apple Auto's highlights, `docs/19` section 5) or a
    /// manual value, -100 to 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub highlights: PercentSetting,
    /// Auto (0, or Apple Auto's shadows) or a manual value, -100 to 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub shadows: PercentSetting,
    /// -100 to 100.
    pub white: f64,
    /// Light › White lift: stops added to the automatic white lift (`docs/02` C12 step 1b), -3 to +3,
    /// default 0 (the automatic amount). A negative offset of the automatic amount or more gives no
    /// lift. A plain number, not Auto; a settings document without the field reads as 0. The lift
    /// works alongside RAW › Range ([`DevelopSettings::rolloff`], C7), after it in the chain.
    pub white_offset_ev: f64,
    /// Auto (0, or Apple Auto's black level) or a manual value, -100 to 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub black: PercentSetting,
    /// Colour › White balance, the light presets.
    pub light_source: LightSource,
    /// Colour › White balance: Auto or Kelvin and tint (the Kelvin also as Look › Tone › Temperature).
    pub white_balance: WhiteBalanceSetting,
    /// The three colour-balance wheels; the shadows and highlights wheels are Auto (`[0, 0]`, or the
    /// colour Apple Auto gave that zone) or a point.
    pub color_balance: ColorBalance,
    /// Colour › Colour balance and Look › Tone: Auto (0, or Apple Auto's saturation, `docs/19` section
    /// 5) or a manual value, -100 to 100; the chroma scaled about the luminance, which stays (docs/02
    /// C9). Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub saturation: SaturationSetting,
    /// Colour › Colour warp: the pins of stage C9b (docs/02); none by default.
    pub color_warp: ColorWarp,
    /// Detail › Sharpening: Auto or amount and radius.
    pub sharpen: SharpenSetting,
    /// Detail › Sharpening › Output: the strength of the output sharpening in percent, 0 to 200,
    /// default 100 (a plain number, not Auto; 0 is off). A document without the field reads as 100.
    pub output_sharpening: f64,
    /// Detail › Noise reduction: Auto or luminance, colour and detail.
    pub noise_reduction: NoiseReductionSetting,
    /// Luma, red, green and blue curves.
    pub curve: Curves,
    /// RAW › Decode switches.
    pub decode: DecodeOptions,
    /// RAW › Range: the highlight range and the low-light toe (C7).
    pub rolloff: Rolloff,
    /// Crop; `null` when the photo is not cropped.
    pub crop: Option<Crop>,
    /// Orientation after the camera's EXIF orientation.
    pub orientation: Orientation,
    /// Pick white: the sampled point `[x, y]` in [0, 1]², measured on the photo as the camera
    /// oriented it (EXIF orientation applied, before the user's crop, straighten and
    /// `orientation`), so it keeps pointing at the same spot when those change. See
    /// [`crate::auto::display_to_sensor`].
    pub white_pick: Option<[f64; 2]>,
}

impl Default for DevelopSettings {
    /// Auto everywhere and the identity look, Neutral: the blank development, which needs no look
    /// library to render (tests, a core without one) and is what the frontend's `defaultSettings()`
    /// and `settings.schema.json` call the default. It is not what a user sees first: a photo a
    /// session opens starts from [`DevelopSettings::with_look`] with the user's default look
    /// (Signature out of the box, `ym_core::preferences::DEFAULT_LOOK`), and a settings document
    /// that names no look reads as that look too ([`DevelopSettings::from_json`]).
    fn default() -> Self {
        Self::with_look(LookId::neutral())
    }
}

impl DevelopSettings {
    /// The settings of a photo nobody has touched: Auto everywhere and `look` at 100 percent.
    #[must_use]
    pub fn with_look(look: LookId) -> Self {
        Self {
            look,
            look_intensity: 100.0,
            exposure: AutoOr::Auto,
            contrast: AutoOr::Auto,
            brightness: 0.0,
            highlights: AutoOr::Auto,
            shadows: AutoOr::Auto,
            white: 0.0,
            white_offset_ev: 0.0,
            black: AutoOr::Auto,
            light_source: LightSource::Auto,
            white_balance: AutoOr::Auto,
            color_balance: ColorBalance::default(),
            saturation: AutoOr::Auto,
            color_warp: ColorWarp::default(),
            sharpen: AutoOr::Auto,
            output_sharpening: OUTPUT_SHARPENING_DEFAULT,
            noise_reduction: AutoOr::Auto,
            curve: Curves::default(),
            decode: DecodeOptions::default(),
            rolloff: Rolloff::default(),
            crop: None,
            orientation: Orientation::Deg0,
            white_pick: None,
        }
    }

    /// The fields that differ from `other`.
    #[must_use]
    pub fn diff(&self, other: &DevelopSettings) -> Vec<SettingsField> {
        SettingsField::ALL
            .into_iter()
            .filter(|f| !self.field_equals(other, *f))
            .collect()
    }

    fn field_equals(&self, o: &DevelopSettings, field: SettingsField) -> bool {
        match field {
            SettingsField::Look => self.look == o.look,
            SettingsField::LookIntensity => self.look_intensity == o.look_intensity,
            SettingsField::Exposure => self.exposure == o.exposure,
            SettingsField::Contrast => self.contrast == o.contrast,
            SettingsField::Brightness => self.brightness == o.brightness,
            SettingsField::Highlights => self.highlights == o.highlights,
            SettingsField::Shadows => self.shadows == o.shadows,
            SettingsField::White => self.white == o.white,
            SettingsField::WhiteOffset => self.white_offset_ev == o.white_offset_ev,
            SettingsField::Black => self.black == o.black,
            SettingsField::LightSource => self.light_source == o.light_source,
            SettingsField::WhiteBalance => self.white_balance == o.white_balance,
            SettingsField::ColorBalance => self.color_balance == o.color_balance,
            SettingsField::Saturation => self.saturation == o.saturation,
            SettingsField::ColorWarp => self.color_warp == o.color_warp,
            SettingsField::Sharpen => self.sharpen == o.sharpen,
            SettingsField::OutputSharpening => self.output_sharpening == o.output_sharpening,
            SettingsField::NoiseReduction => self.noise_reduction == o.noise_reduction,
            SettingsField::Curve => self.curve == o.curve,
            SettingsField::Decode => self.decode == o.decode,
            SettingsField::Rolloff => self.rolloff == o.rolloff,
            SettingsField::Crop => self.crop == o.crop,
            SettingsField::Orientation => self.orientation == o.orientation,
            SettingsField::WhitePick => self.white_pick == o.white_pick,
        }
    }

    /// Restores `field` to its default (Auto for the automatic ones). `default_look` is the user's
    /// default look, which Reset on the Look card restores.
    pub fn reset_field(&mut self, field: SettingsField, default_look: &LookId) {
        let d = DevelopSettings::with_look(default_look.clone());
        match field {
            SettingsField::Look => self.look = d.look,
            SettingsField::LookIntensity => self.look_intensity = d.look_intensity,
            SettingsField::Exposure => self.exposure = d.exposure,
            SettingsField::Contrast => self.contrast = d.contrast,
            SettingsField::Brightness => self.brightness = d.brightness,
            SettingsField::Highlights => self.highlights = d.highlights,
            SettingsField::Shadows => self.shadows = d.shadows,
            SettingsField::White => self.white = d.white,
            SettingsField::WhiteOffset => self.white_offset_ev = d.white_offset_ev,
            SettingsField::Black => self.black = d.black,
            SettingsField::LightSource => self.light_source = d.light_source,
            SettingsField::WhiteBalance => self.white_balance = d.white_balance,
            SettingsField::ColorBalance => self.color_balance = d.color_balance,
            SettingsField::Saturation => self.saturation = d.saturation,
            SettingsField::ColorWarp => self.color_warp = d.color_warp,
            SettingsField::Sharpen => self.sharpen = d.sharpen,
            SettingsField::OutputSharpening => self.output_sharpening = d.output_sharpening,
            SettingsField::NoiseReduction => self.noise_reduction = d.noise_reduction,
            SettingsField::Curve => self.curve = d.curve,
            SettingsField::Decode => self.decode = d.decode,
            SettingsField::Rolloff => self.rolloff = d.rolloff,
            SettingsField::Crop => self.crop = d.crop,
            SettingsField::Orientation => self.orientation = d.orientation,
            SettingsField::WhitePick => self.white_pick = d.white_pick,
        }
    }

    /// What "Re-run" on the Automatic card resets: exposure, white balance (the light preset, the
    /// Kelvin and tint and a picked white) and noise reduction go back to Auto, and the white lift's
    /// offset (the Automatic card's White row) back to 0, the automatic amount. The values Apple
    /// Auto sets besides those (highlights, shadows, black, contrast, saturation and the shadows and
    /// highlights wheels) go back to Auto only when `apple_auto` says Apple Auto acts on the photo
    /// (its state is pending, answered or failed: the card then shows their rows,
    /// [`AppleAuto::acts`]); otherwise they stay as they are, as before Apple Auto (`docs/19`
    /// section 5).
    ///
    /// [`AppleAuto::acts`]: crate::apple_auto::AppleAuto::acts
    pub fn reset_automatic(&mut self, apple_auto: bool) {
        self.exposure = AutoOr::Auto;
        if apple_auto {
            self.contrast = AutoOr::Auto;
            self.saturation = AutoOr::Auto;
            self.highlights = AutoOr::Auto;
            self.shadows = AutoOr::Auto;
            self.black = AutoOr::Auto;
            self.color_balance.shadows = AutoOr::Auto;
            self.color_balance.highlights = AutoOr::Auto;
        }
        self.light_source = LightSource::Auto;
        self.white_balance = AutoOr::Auto;
        self.white_pick = None;
        self.noise_reduction = AutoOr::Auto;
        self.white_offset_ev = 0.0;
    }

    /// Reads a settings document (the `ym:Settings` JSON, `ym render --settings`, a golden
    /// manifest) the way a photo that was just opened would take it: fields the document leaves
    /// out have their defaults, and a document without `look` gets `default_look`, the user's
    /// default look (`Default` alone would say Neutral). Values are then clamped like
    /// [`DevelopSettings::validated`] does; the fields that were changed are returned.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Parse`] for text that is not JSON, a field that does not exist or a value
    /// of the wrong type, and the errors of [`DevelopSettings::validated`].
    pub fn from_json(
        text: &str,
        default_look: &LookId,
    ) -> Result<(Self, Vec<SettingsField>), SettingsError> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| SettingsError::Parse(e.to_string()))?;
        Self::from_value(value, default_look)
    }

    /// [`DevelopSettings::from_json`] for a document that is already parsed (the value for one
    /// file out of a settings file keyed by file name).
    ///
    /// # Errors
    ///
    /// As [`DevelopSettings::from_json`].
    pub fn from_value(
        value: serde_json::Value,
        default_look: &LookId,
    ) -> Result<(Self, Vec<SettingsField>), SettingsError> {
        let names_a_look = value.get("look").is_some();
        let mut settings: DevelopSettings =
            serde_json::from_value(value).map_err(|e| SettingsError::Parse(e.to_string()))?;
        if !names_a_look {
            settings.look = default_look.clone();
        }
        settings.validated()
    }

    /// The settings after clamping every value to its range and normalising curves and the crop.
    ///
    /// Settings read from a file (`--settings`, a golden manifest) go through this before use;
    /// patches are checked by [`DevelopSettings::apply_patch`].
    ///
    /// # Errors
    ///
    /// [`SettingsError`] for a value that is not a number, a curve that cannot be used or a crop
    /// without area.
    pub fn validated(mut self) -> Result<(Self, Vec<SettingsField>), SettingsError> {
        let mut clamped = Vec::new();
        self.sanitise(&mut clamped)?;
        Ok((self, clamped))
    }

    /// Clamps and normalises in place; `clamped` collects the fields that were changed.
    fn sanitise(&mut self, clamped: &mut Vec<SettingsField>) -> Result<(), SettingsError> {
        let mut note = |field: SettingsField, changed: bool| {
            if changed && !clamped.contains(&field) {
                clamped.push(field);
            }
        };
        let (v, c) = clamp_to("look_intensity", self.look_intensity, LOOK_INTENSITY_RANGE)?;
        self.look_intensity = v;
        note(SettingsField::LookIntensity, c);
        for (name, field, setting) in [
            ("contrast", SettingsField::Contrast, &mut self.contrast),
            (
                "saturation",
                SettingsField::Saturation,
                &mut self.saturation,
            ),
            (
                "highlights",
                SettingsField::Highlights,
                &mut self.highlights,
            ),
            ("shadows", SettingsField::Shadows, &mut self.shadows),
            ("black", SettingsField::Black, &mut self.black),
        ] {
            if let AutoOr::Manual(m) = setting {
                let (v, c) = clamp_to(name, m.value, PERCENT_RANGE)?;
                m.value = v;
                note(field, c);
            }
        }
        for (name, field, value) in [
            (
                "brightness",
                SettingsField::Brightness,
                &mut self.brightness,
            ),
            ("white", SettingsField::White, &mut self.white),
        ] {
            let (v, c) = clamp_to(name, *value, PERCENT_RANGE)?;
            *value = v;
            note(field, c);
        }
        if let AutoOr::Manual(m) = &mut self.exposure {
            let (v, c) = clamp_to("exposure", m.ev, EXPOSURE_RANGE_EV)?;
            m.ev = v;
            note(SettingsField::Exposure, c);
        }
        if let AutoOr::Manual(m) = &mut self.white_balance {
            let (k, ck) = clamp_to("white_balance.kelvin", m.kelvin, KELVIN_RANGE_K)?;
            let (t, ct) = clamp_to("white_balance.tint", m.tint, TINT_RANGE)?;
            (m.kelvin, m.tint) = (k, t);
            note(SettingsField::WhiteBalance, ck || ct);
        }
        if let AutoOr::Manual(m) = &mut self.sharpen {
            let (a, ca) = clamp_to("sharpen.amount", m.amount, SHARPEN_AMOUNT_RANGE)?;
            let (r, cr) = clamp_to("sharpen.radius_px", m.radius_px, SHARPEN_RADIUS_RANGE_PX)?;
            (m.amount, m.radius_px) = (a, r);
            note(SettingsField::Sharpen, ca || cr);
        }
        let (v, c) = clamp_to(
            "output_sharpening",
            self.output_sharpening,
            OUTPUT_SHARPENING_RANGE,
        )?;
        self.output_sharpening = v;
        note(SettingsField::OutputSharpening, c);
        let (v, c) = clamp_to(
            "white_offset_ev",
            self.white_offset_ev,
            WHITE_OFFSET_RANGE_EV,
        )?;
        self.white_offset_ev = v;
        note(SettingsField::WhiteOffset, c);
        if let AutoOr::Manual(m) = &mut self.noise_reduction {
            let mut changed = false;
            for (name, v) in [
                ("noise_reduction.luminance", &mut m.luminance),
                ("noise_reduction.color", &mut m.color),
                ("noise_reduction.detail", &mut m.detail),
            ] {
                let (x, c) = clamp_to(name, *v, NOISE_REDUCTION_RANGE)?;
                *v = x;
                changed |= c;
            }
            note(SettingsField::NoiseReduction, changed);
        }
        let mut changed = false;
        let cb = &mut self.color_balance;
        let mut points: Vec<(&'static str, &mut ColorBalancePoint)> =
            vec![("color_balance.midtones", &mut cb.midtones)];
        if let AutoOr::Manual(m) = &mut cb.shadows {
            points.push(("color_balance.shadows", &mut m.point));
        }
        if let AutoOr::Manual(m) = &mut cb.highlights {
            points.push(("color_balance.highlights", &mut m.point));
        }
        for (name, point) in points {
            let (p, c) = clamp_to_disc(name, *point)?;
            *point = p;
            changed |= c;
        }
        note(SettingsField::ColorBalance, changed);
        note(
            SettingsField::ColorWarp,
            normalise_warp(&mut self.color_warp)?,
        );
        let mut changed = false;
        for (name, v) in [
            ("rolloff.highlights", &mut self.rolloff.highlights),
            ("rolloff.low_light", &mut self.rolloff.low_light),
        ] {
            let (x, c) = clamp_to(name, *v, ROLLOFF_RANGE)?;
            *v = x;
            changed |= c;
        }
        note(SettingsField::Rolloff, changed);
        let mut changed = false;
        for (channel, points) in [
            ("luma", &mut self.curve.luma),
            ("red", &mut self.curve.red),
            ("green", &mut self.curve.green),
            ("blue", &mut self.curve.blue),
        ] {
            changed |= normalise_curve(channel, points)?;
        }
        note(SettingsField::Curve, changed);
        if let Some(crop) = &mut self.crop {
            note(SettingsField::Crop, normalise_crop(crop)?);
        }
        if let Some(p) = &mut self.white_pick {
            let mut changed = false;
            for v in p.iter_mut() {
                let (x, c) = clamp_to("white_pick", *v, (0.0, 1.0))?;
                *v = x;
                changed |= c;
            }
            note(SettingsField::WhitePick, changed);
        }
        Ok(())
    }
}

/// Clamps a number to a range; a value that is not finite is an error. The flag says whether the
/// value was changed.
fn clamp_to(
    name: &'static str,
    v: f64,
    (lo, hi): (f64, f64),
) -> Result<(f64, bool), SettingsError> {
    if !v.is_finite() {
        return Err(SettingsError::NotFinite(name));
    }
    let c = v.clamp(lo, hi);
    Ok((c + 0.0, c != v))
}

/// Scales a point back into the unit disc.
fn clamp_to_disc(name: &'static str, [a, b]: [f64; 2]) -> Result<([f64; 2], bool), SettingsError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(SettingsError::NotFinite(name));
    }
    let r = a.hypot(b);
    if r > 1.0 {
        Ok(([a / r, b / r], true))
    } else {
        Ok(([a + 0.0, b + 0.0], false))
    }
}

/// Cuts the colour warp to [`WARP_MAX_PINS`] pins and clamps each one as `WarpPin::clamped` does.
/// A number that is not finite, in any pin, is an error. Returns whether anything changed.
fn normalise_warp(warp: &mut ColorWarp) -> Result<bool, SettingsError> {
    for pin in &warp.pins {
        for (name, values) in [
            ("color_warp.source", &pin.source[..]),
            ("color_warp.target", &pin.target[..]),
            ("color_warp.range", &[pin.range][..]),
            ("color_warp.shadows", &[pin.shadows][..]),
            ("color_warp.highlights", &[pin.highlights][..]),
            ("color_warp.pivot_ev", &[pin.pivot_ev][..]),
            ("color_warp.exposure_ev", &[pin.exposure_ev][..]),
        ] {
            if !values.iter().all(|v| v.is_finite()) {
                return Err(SettingsError::NotFinite(name));
            }
        }
    }
    let before = warp.pins.clone();
    warp.pins.truncate(WARP_MAX_PINS);
    for pin in &mut warp.pins {
        let c = WarpPin::from(*pin).clamped();
        *pin = ColorWarpPin {
            source: within_warp_disc(c.source),
            target: within_warp_disc(c.target),
            locked: c.locked,
            range: c.range + 0.0,
            shadows: c.shadows + 0.0,
            highlights: c.highlights + 0.0,
            pivot_ev: c.pivot_ev + 0.0,
            exposure_ev: c.exposure_ev + 0.0,
        };
    }
    Ok(warp.pins != before)
}

/// A point that `WarpPin::clamped` scaled back onto the disc of radius [`WARP_COORD_LIMIT`], moved
/// inside it by the last ulp when rounding left it just outside (`p * limit / |p|` lands outside in
/// about one case in fifty): a second check must find nothing to change, or every later patch would
/// report the warp as changed.
fn within_warp_disc(p: ChromaPoint) -> ChromaPoint {
    let mut p = p;
    for _ in 0..8 {
        if p[0].hypot(p[1]) <= WARP_COORD_LIMIT {
            break;
        }
        p = [p[0] * (1.0 - f64::EPSILON), p[1] * (1.0 - f64::EPSILON)];
    }
    [p[0] + 0.0, p[1] + 0.0]
}

/// Clamps a curve's points into the unit square, sorts them by `x` and checks the rules of
/// `ym_color::curve` (2 to 16 points, `x` strictly increasing). Returns whether anything changed.
fn normalise_curve(
    channel: &'static str,
    points: &mut Vec<CurvePoint>,
) -> Result<bool, SettingsError> {
    let bad = |problem: String| SettingsError::Curve { channel, problem };
    if points.len() < MIN_CURVE_POINTS {
        return Err(bad(format!(
            "needs at least {MIN_CURVE_POINTS} points, it has {}",
            points.len()
        )));
    }
    if points.len() > MAX_CURVE_POINTS {
        return Err(bad(format!(
            "has {} points, at most {MAX_CURVE_POINTS} are allowed",
            points.len()
        )));
    }
    let before = points.clone();
    for p in points.iter_mut() {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(SettingsError::NotFinite("curve"));
        }
        *p = [p[0].clamp(0.0, 1.0) + 0.0, p[1].clamp(0.0, 1.0) + 0.0];
    }
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if points.windows(2).any(|w| w[0][0] >= w[1][0]) {
        return Err(bad("has two points at the same position".to_owned()));
    }
    Ok(*points != before)
}

/// Brings a crop into the unit square and its angle into range. Returns whether anything changed.
fn normalise_crop(crop: &mut Crop) -> Result<bool, SettingsError> {
    let before = *crop;
    let r = &mut crop.rect;
    if ![r.x, r.y, r.width, r.height, crop.angle_deg]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(SettingsError::NotFinite("crop"));
    }
    if r.width < MIN_CROP_SIDE || r.height < MIN_CROP_SIDE {
        return Err(SettingsError::EmptyCrop);
    }
    r.width = r.width.min(1.0);
    r.height = r.height.min(1.0);
    r.x = r.x.clamp(0.0, 1.0 - r.width);
    r.y = r.y.clamp(0.0, 1.0 - r.height);
    crop.angle_deg = crop
        .angle_deg
        .clamp(CROP_ANGLE_RANGE_DEG.0, CROP_ANGLE_RANGE_DEG.1)
        + 0.0;
    Ok(*crop != before)
}

// ---------------------------------------------------------------------------------------------
// Patches
// ---------------------------------------------------------------------------------------------

/// Deserialises a field that distinguishes "absent" from "null": absent stays `None`, `null` is
/// `Some(None)`, a value is `Some(Some(v))`. Use with `#[serde(default)]`.
fn double_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// A manual white balance with parts left out: the missing part keeps the current value (the
/// stored one, or the automatic one while the setting is Auto).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteBalanceManualPatch {
    /// New Kelvin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kelvin: Option<f64>,
    /// New tint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
}

/// A manual sharpening with parts left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharpenManualPatch {
    /// New amount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<f64>,
    /// New radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius_px: Option<f64>,
}

/// A manual noise reduction with parts left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseReductionManualPatch {
    /// New luminance strength.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub luminance: Option<f64>,
    /// New colour strength.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<f64>,
    /// New detail strength.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<f64>,
}

/// Decode switches to change; the others stay as they are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodePatch {
    /// New value of `neural_demosaic`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neural_demosaic: Option<bool>,
    /// New value of `highlight_recovery`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight_recovery: Option<bool>,
    /// New value of `lens_corrections`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens_corrections: Option<bool>,
    /// New value of `chromatic_aberration`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chromatic_aberration: Option<bool>,
}

/// RAW › Range amounts to change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloffPatch {
    /// New highlight range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f64>,
    /// New low-light toe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_light: Option<f64>,
}

/// Colour-balance wheels to change. The shadows and highlights wheels take Auto (Reset) or Manual
/// with a point; a plain point reads as Manual at that point, and `[0, 0]` as Auto.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorBalancePatch {
    /// New shadows wheel.
    #[serde(
        default,
        deserialize_with = "balance_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub shadows: Option<BalanceSetting>,
    /// New midtones wheel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub midtones: Option<ColorBalancePoint>,
    /// New highlights wheel.
    #[serde(
        default,
        deserialize_with = "balance_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub highlights: Option<BalanceSetting>,
}

/// Curves to replace; the others stay as they are.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvesPatch {
    /// New luma curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub luma: Option<Vec<CurvePoint>>,
    /// New red curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub red: Option<Vec<CurvePoint>>,
    /// New green curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub green: Option<Vec<CurvePoint>>,
    /// New blue curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blue: Option<Vec<CurvePoint>>,
}

/// A partial update of [`DevelopSettings`] (`update_settings(photo_id, patch)`): only the fields
/// present change.
///
/// The JSON is `Partial<DevelopSettings>` of `src/api/types.ts`, with two generosities: the nested
/// objects (`decode`, `rolloff`, `color_balance`, `curve`) may carry only the parts to change, and
/// a manual `white_balance`, `sharpen` or `noise_reduction` may leave out a part, which then keeps
/// its current (resolved) value. `crop: null` and `white_pick: null` clear the field. `color_warp`
/// replaces the whole list of pins (`{ "pins": [] }` removes them all).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettingsPatch {
    /// New look.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<LookId>,
    /// New look intensity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look_intensity: Option<f64>,
    /// New exposure (Manual to set it, Auto to reset it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureSetting>,
    /// New contrast: Auto (Reset), or Manual with its value. A plain number reads as Manual with that
    /// value, and 0 as Auto.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub contrast: Option<ContrastSetting>,
    /// New brightness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brightness: Option<f64>,
    /// New highlights: Auto (Reset), or Manual with its value. A plain number reads as Manual with
    /// that value, and 0 as Auto.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub highlights: Option<PercentSetting>,
    /// New shadows, read like `highlights`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub shadows: Option<PercentSetting>,
    /// New white.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white: Option<f64>,
    /// New white lift offset, stops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_offset_ev: Option<f64>,
    /// New black, read like `highlights`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub black: Option<PercentSetting>,
    /// New light preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_source: Option<LightSource>,
    /// New white balance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_balance: Option<AutoOr<WhiteBalanceManualPatch>>,
    /// Colour-balance wheels to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_balance: Option<ColorBalancePatch>,
    /// New saturation: Auto (Reset), or Manual with its value. A plain number reads as Manual with
    /// that value, and 0 as Auto.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub saturation: Option<SaturationSetting>,
    /// New colour warp: the whole list of pins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_warp: Option<ColorWarp>,
    /// New sharpening.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpen: Option<AutoOr<SharpenManualPatch>>,
    /// New output sharpening amount, percent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_sharpening: Option<f64>,
    /// New noise reduction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<AutoOr<NoiseReductionManualPatch>>,
    /// Curves to replace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<CurvesPatch>,
    /// Decode switches to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodePatch>,
    /// RAW › Range amounts to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolloff: Option<RolloffPatch>,
    /// New crop; `Some(None)` (JSON `null`) removes it.
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub crop: Option<Option<Crop>>,
    /// New orientation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<Orientation>,
    /// New white pick; `Some(None)` (JSON `null`) clears it. A pick also sets `light_source` and
    /// `white_balance` to Auto unless the same patch says otherwise: the picked spot then renders
    /// neutral, and the Kelvin and tint rows show the picked light (`docs/11` section Pick white).
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub white_pick: Option<Option<[f64; 2]>>,
}

/// The automatic values that fill the missing part of a partial manual patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoSeeds {
    /// `exposure` Auto resolves to this (stops).
    pub exposure_ev: f64,
    /// `white_balance` Auto resolves to this Kelvin.
    pub kelvin: f64,
    /// `white_balance` Auto resolves to this tint.
    pub tint: f64,
    /// `sharpen` Auto resolves to this.
    pub sharpen: SharpenValue,
    /// `noise_reduction` Auto resolves to this.
    pub noise_reduction: NoiseReductionValue,
}

/// What [`DevelopSettings::apply_patch`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatchReport {
    /// The fields whose value changed (including `light_source` and `white_balance` when a white
    /// pick reset them).
    pub changed: Vec<SettingsField>,
    /// The fields in which a number was out of range and was clamped.
    pub clamped: Vec<SettingsField>,
}

impl PatchReport {
    /// The first render stage group any changed field affects, if anything changed.
    #[must_use]
    pub fn first_group(&self) -> Option<StageGroup> {
        self.changed.iter().map(|f| f.group()).min()
    }
}

impl SettingsPatch {
    /// True when the patch carries no change.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == SettingsPatch::default()
    }

    /// A patch that moves the exposure slider: Manual at `ev`.
    #[must_use]
    pub fn exposure_ev(ev: f64) -> Self {
        Self {
            exposure: Some(ExposureSetting::manual_ev(ev)),
            ..Self::default()
        }
    }

    /// A patch that sets the Kelvin and tint by hand.
    #[must_use]
    pub fn white_balance(kelvin: f64, tint: f64) -> Self {
        Self {
            white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(kelvin),
                tint: Some(tint),
            })),
            ..Self::default()
        }
    }

    /// A patch that moves only the Kelvin slider (the tint keeps its current value).
    #[must_use]
    pub fn kelvin(kelvin: f64) -> Self {
        Self {
            white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(kelvin),
                tint: None,
            })),
            ..Self::default()
        }
    }

    /// A patch that sets the sharpening by hand.
    #[must_use]
    pub fn sharpen(amount: f64, radius_px: f64) -> Self {
        Self {
            sharpen: Some(AutoOr::Manual(SharpenManualPatch {
                amount: Some(amount),
                radius_px: Some(radius_px),
            })),
            ..Self::default()
        }
    }

    /// A patch that sets the noise reduction by hand.
    #[must_use]
    pub fn noise_reduction(luminance: f64, color: f64, detail: f64) -> Self {
        Self {
            noise_reduction: Some(AutoOr::Manual(NoiseReductionManualPatch {
                luminance: Some(luminance),
                color: Some(color),
                detail: Some(detail),
            })),
            ..Self::default()
        }
    }

    /// The patch that takes `field` of `from` (every part of it).
    #[must_use]
    pub fn of_field(from: &DevelopSettings, field: SettingsField) -> Self {
        let mut p = Self::default();
        match field {
            SettingsField::Look => p.look = Some(from.look.clone()),
            SettingsField::LookIntensity => p.look_intensity = Some(from.look_intensity),
            SettingsField::Exposure => p.exposure = Some(from.exposure),
            SettingsField::Contrast => p.contrast = Some(from.contrast),
            SettingsField::Brightness => p.brightness = Some(from.brightness),
            SettingsField::Highlights => p.highlights = Some(from.highlights),
            SettingsField::Shadows => p.shadows = Some(from.shadows),
            SettingsField::White => p.white = Some(from.white),
            SettingsField::WhiteOffset => p.white_offset_ev = Some(from.white_offset_ev),
            SettingsField::Black => p.black = Some(from.black),
            SettingsField::LightSource => p.light_source = Some(from.light_source),
            SettingsField::WhiteBalance => {
                p.white_balance = Some(match from.white_balance {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(WhiteBalanceManualPatch {
                        kelvin: Some(m.kelvin),
                        tint: Some(m.tint),
                    }),
                });
            }
            SettingsField::ColorBalance => {
                p.color_balance = Some(ColorBalancePatch {
                    shadows: Some(from.color_balance.shadows),
                    midtones: Some(from.color_balance.midtones),
                    highlights: Some(from.color_balance.highlights),
                });
            }
            SettingsField::Saturation => p.saturation = Some(from.saturation),
            SettingsField::ColorWarp => p.color_warp = Some(from.color_warp.clone()),
            SettingsField::Sharpen => {
                p.sharpen = Some(match from.sharpen {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(SharpenManualPatch {
                        amount: Some(m.amount),
                        radius_px: Some(m.radius_px),
                    }),
                });
            }
            SettingsField::OutputSharpening => p.output_sharpening = Some(from.output_sharpening),
            SettingsField::NoiseReduction => {
                p.noise_reduction = Some(match from.noise_reduction {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(NoiseReductionManualPatch {
                        luminance: Some(m.luminance),
                        color: Some(m.color),
                        detail: Some(m.detail),
                    }),
                });
            }
            SettingsField::Curve => {
                p.curve = Some(CurvesPatch {
                    luma: Some(from.curve.luma.clone()),
                    red: Some(from.curve.red.clone()),
                    green: Some(from.curve.green.clone()),
                    blue: Some(from.curve.blue.clone()),
                });
            }
            SettingsField::Decode => {
                p.decode = Some(DecodePatch {
                    neural_demosaic: Some(from.decode.neural_demosaic),
                    highlight_recovery: Some(from.decode.highlight_recovery),
                    lens_corrections: Some(from.decode.lens_corrections),
                    chromatic_aberration: Some(from.decode.chromatic_aberration),
                });
            }
            SettingsField::Rolloff => {
                p.rolloff = Some(RolloffPatch {
                    highlights: Some(from.rolloff.highlights),
                    low_light: Some(from.rolloff.low_light),
                });
            }
            SettingsField::Crop => p.crop = Some(from.crop),
            SettingsField::Orientation => p.orientation = Some(from.orientation),
            SettingsField::WhitePick => p.white_pick = Some(from.white_pick),
        }
        p
    }

    /// The Reset patch for `field`: restores its default (Auto for the automatic ones).
    #[must_use]
    pub fn reset(field: SettingsField, default_look: &LookId) -> Self {
        Self::of_field(&DevelopSettings::with_look(default_look.clone()), field)
    }

    /// The patch that takes every field of `from` that is in `fields` (every part of each).
    #[must_use]
    pub fn of_fields(from: &DevelopSettings, fields: &[SettingsField]) -> Self {
        let mut patch = Self::default();
        for field in fields {
            patch.absorb(Self::of_field(from, *field));
        }
        patch
    }

    /// Takes over the fields `other` sets. Each field of a patch built by [`SettingsPatch::of_field`]
    /// belongs to one field only, so nothing is overwritten.
    fn absorb(&mut self, other: Self) {
        macro_rules! take {
            ($($field:ident),* $(,)?) => {
                $(if other.$field.is_some() { self.$field = other.$field; })*
            };
        }
        take!(
            look,
            look_intensity,
            exposure,
            contrast,
            brightness,
            highlights,
            shadows,
            white,
            white_offset_ev,
            black,
            light_source,
            white_balance,
            color_balance,
            saturation,
            color_warp,
            sharpen,
            output_sharpening,
            noise_reduction,
            curve,
            decode,
            rolloff,
            crop,
            orientation,
            white_pick,
        );
    }

    /// What Copy settings takes from a photo, to paste onto others (`docs/11` §Copy and paste
    /// settings): every field of the photo's settings but [`COPY_LEAVES_OUT`], which belong to
    /// the photo's own geometry and pixels (the crop and the turn, and the point of Pick white).
    ///
    /// * Auto stays Auto (exposure, contrast, highlights, shadows, black, light preset, white balance,
    ///   the shadows and highlights wheels, saturation, sharpening, noise reduction): each photo gets
    ///   its own automatic values (with Apple Auto, its own answer). Manual values are taken as they are, and so is the
    ///   white lift's offset (a number of stops added to each photo's own automatic amount).
    /// * A white balance that came from a picked white (a white picked, the light preset and the
    ///   white balance both Auto) becomes a manual white balance at the Kelvin and tint the pick
    ///   resolved to: the point is not meaningful on another photo, the light it found is.
    ///
    /// Every part of every manual value is in the patch, so it applies to a photo that has no
    /// automatic estimates yet.
    #[must_use]
    pub fn copied_from(source: &ResolvedSettings) -> Self {
        let settings = &source.settings;
        let fields: Vec<SettingsField> = SettingsField::ALL
            .into_iter()
            .filter(|f| !COPY_LEAVES_OUT.contains(f))
            .collect();
        let mut patch = Self::of_fields(settings, &fields);
        let white_came_from_pick = settings.white_pick.is_some()
            && settings.light_source == LightSource::Auto
            && settings.white_balance.is_auto();
        if white_came_from_pick {
            patch.white_balance = Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(source.values.kelvin),
                tint: Some(source.values.tint),
            }));
        }
        patch
    }
}

/// The fields Copy settings leaves out: the crop and the orientation (the photo's own geometry)
/// and the Pick white point (a spot of that photo's pixels). See [`SettingsPatch::copied_from`].
pub const COPY_LEAVES_OUT: [SettingsField; 3] = [
    SettingsField::Crop,
    SettingsField::Orientation,
    SettingsField::WhitePick,
];

impl DevelopSettings {
    /// Applies a patch with the rules of the module documentation.
    ///
    /// The patch is applied atomically: on `Err` the settings are untouched. Numbers out of range
    /// are clamped and reported in [`PatchReport::clamped`]. `seeds` are the photo's resolved
    /// automatic values; they fill the missing part of a partial manual `white_balance`,
    /// `sharpen` or `noise_reduction` while the setting is `Auto` (a patch with only `kelvin` on
    /// an Auto white balance is Manual at that Kelvin and the automatic tint).
    ///
    /// # Errors
    ///
    /// [`SettingsError`] for a value that is not a number, an unusable curve or crop, and a
    /// partial manual patch that needs `seeds` when there are none.
    pub fn apply_patch(
        &mut self,
        patch: &SettingsPatch,
        seeds: Option<&AutoSeeds>,
    ) -> Result<PatchReport, SettingsError> {
        let mut next = self.clone();
        if let Some(v) = &patch.look {
            next.look = v.clone();
        }
        if let Some(v) = patch.look_intensity {
            next.look_intensity = v;
        }
        if let Some(v) = patch.exposure {
            next.exposure = v;
        }
        if let Some(v) = patch.contrast {
            next.contrast = v;
        }
        if let Some(v) = patch.brightness {
            next.brightness = v;
        }
        if let Some(v) = patch.highlights {
            next.highlights = v;
        }
        if let Some(v) = patch.shadows {
            next.shadows = v;
        }
        if let Some(v) = patch.white {
            next.white = v;
        }
        if let Some(v) = patch.black {
            next.black = v;
        }
        if let Some(v) = patch.light_source {
            next.light_source = v;
        }
        if let Some(wb) = &patch.white_balance {
            next.white_balance = match wb {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let base = match self.white_balance {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.kelvin, p.tint) {
                            (_, Some(k), Some(t)) => WhiteBalanceValue { kelvin: k, tint: t },
                            (Some(s), _, _) => WhiteBalanceValue {
                                kelvin: s.kelvin,
                                tint: s.tint,
                            },
                            (None, _, _) => {
                                return Err(SettingsError::NeedsEstimates("white_balance"));
                            }
                        },
                    };
                    AutoOr::Manual(WhiteBalanceValue {
                        kelvin: p.kelvin.unwrap_or(base.kelvin),
                        tint: p.tint.unwrap_or(base.tint),
                    })
                }
            };
        }
        if let Some(cb) = &patch.color_balance {
            if let Some(v) = cb.shadows {
                next.color_balance.shadows = v;
            }
            if let Some(v) = cb.midtones {
                next.color_balance.midtones = v;
            }
            if let Some(v) = cb.highlights {
                next.color_balance.highlights = v;
            }
        }
        if let Some(v) = patch.saturation {
            next.saturation = v;
        }
        if let Some(v) = &patch.color_warp {
            next.color_warp = v.clone();
        }
        if let Some(s) = &patch.sharpen {
            next.sharpen = match s {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let base = match self.sharpen {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.amount, p.radius_px) {
                            (_, Some(a), Some(r)) => SharpenValue {
                                amount: a,
                                radius_px: r,
                            },
                            (Some(s), _, _) => s.sharpen,
                            (None, _, _) => return Err(SettingsError::NeedsEstimates("sharpen")),
                        },
                    };
                    AutoOr::Manual(SharpenValue {
                        amount: p.amount.unwrap_or(base.amount),
                        radius_px: p.radius_px.unwrap_or(base.radius_px),
                    })
                }
            };
        }
        if let Some(v) = patch.output_sharpening {
            next.output_sharpening = v;
        }
        if let Some(v) = patch.white_offset_ev {
            next.white_offset_ev = v;
        }
        if let Some(n) = &patch.noise_reduction {
            next.noise_reduction = match n {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let base = match self.noise_reduction {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.luminance, p.color, p.detail) {
                            (_, Some(l), Some(c), Some(d)) => NoiseReductionValue {
                                luminance: l,
                                color: c,
                                detail: d,
                            },
                            (Some(s), _, _, _) => s.noise_reduction,
                            (None, ..) => {
                                return Err(SettingsError::NeedsEstimates("noise_reduction"));
                            }
                        },
                    };
                    AutoOr::Manual(NoiseReductionValue {
                        luminance: p.luminance.unwrap_or(base.luminance),
                        color: p.color.unwrap_or(base.color),
                        detail: p.detail.unwrap_or(base.detail),
                    })
                }
            };
        }
        if let Some(c) = &patch.curve {
            if let Some(v) = &c.luma {
                next.curve.luma = v.clone();
            }
            if let Some(v) = &c.red {
                next.curve.red = v.clone();
            }
            if let Some(v) = &c.green {
                next.curve.green = v.clone();
            }
            if let Some(v) = &c.blue {
                next.curve.blue = v.clone();
            }
        }
        if let Some(d) = &patch.decode {
            if let Some(v) = d.neural_demosaic {
                next.decode.neural_demosaic = v;
            }
            if let Some(v) = d.highlight_recovery {
                next.decode.highlight_recovery = v;
            }
            if let Some(v) = d.lens_corrections {
                next.decode.lens_corrections = v;
            }
            if let Some(v) = d.chromatic_aberration {
                next.decode.chromatic_aberration = v;
            }
        }
        if let Some(r) = &patch.rolloff {
            if let Some(v) = r.highlights {
                next.rolloff.highlights = v;
            }
            if let Some(v) = r.low_light {
                next.rolloff.low_light = v;
            }
        }
        if let Some(v) = patch.crop {
            next.crop = v;
        }
        if let Some(v) = patch.orientation {
            next.orientation = v;
        }
        if let Some(v) = patch.white_pick {
            next.white_pick = v;
            if v.is_some() {
                // A picked white sets the white balance: the spot renders neutral and the Kelvin
                // and tint rows show the picked light.
                if patch.light_source.is_none() {
                    next.light_source = LightSource::Auto;
                }
                if patch.white_balance.is_none() {
                    next.white_balance = AutoOr::Auto;
                }
            }
        }
        let mut clamped = Vec::new();
        next.sanitise(&mut clamped)?;
        let changed = self.diff(&next);
        // Only report clamping in fields the patch touched: a stored value is always in range.
        clamped.retain(|f| changed.contains(f) || patch_touches(patch, *f));
        *self = next;
        Ok(PatchReport { changed, clamped })
    }
}

/// Whether the patch names `field`.
fn patch_touches(p: &SettingsPatch, field: SettingsField) -> bool {
    match field {
        SettingsField::Look => p.look.is_some(),
        SettingsField::LookIntensity => p.look_intensity.is_some(),
        SettingsField::Exposure => p.exposure.is_some(),
        SettingsField::Contrast => p.contrast.is_some(),
        SettingsField::Brightness => p.brightness.is_some(),
        SettingsField::Highlights => p.highlights.is_some(),
        SettingsField::Shadows => p.shadows.is_some(),
        SettingsField::White => p.white.is_some(),
        SettingsField::WhiteOffset => p.white_offset_ev.is_some(),
        SettingsField::Black => p.black.is_some(),
        SettingsField::LightSource => p.light_source.is_some(),
        SettingsField::WhiteBalance => p.white_balance.is_some(),
        SettingsField::ColorBalance => p.color_balance.is_some(),
        SettingsField::Saturation => p.saturation.is_some(),
        SettingsField::ColorWarp => p.color_warp.is_some(),
        SettingsField::Sharpen => p.sharpen.is_some(),
        SettingsField::OutputSharpening => p.output_sharpening.is_some(),
        SettingsField::NoiseReduction => p.noise_reduction.is_some(),
        SettingsField::Curve => p.curve.is_some(),
        SettingsField::Decode => p.decode.is_some(),
        SettingsField::Rolloff => p.rolloff.is_some(),
        SettingsField::Crop => p.crop.is_some(),
        SettingsField::Orientation => p.orientation.is_some(),
        SettingsField::WhitePick => p.white_pick.is_some(),
    }
}

// ---------------------------------------------------------------------------------------------
// Resolved settings
// ---------------------------------------------------------------------------------------------

/// Which `Auto` fields are still automatic; the UI shows the AUTO tag for these and EDITED for the
/// others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoFlags {
    /// `exposure` is Auto.
    pub exposure: bool,
    /// `contrast` is Auto.
    pub contrast: bool,
    /// `highlights` is Auto.
    pub highlights: bool,
    /// `shadows` is Auto.
    pub shadows: bool,
    /// `black` is Auto.
    pub black: bool,
    /// The shadows colour-balance wheel is Auto.
    pub balance_shadows: bool,
    /// The highlights colour-balance wheel is Auto.
    pub balance_highlights: bool,
    /// `saturation` is Auto.
    pub saturation: bool,
    /// `light_source` is Auto.
    pub light_source: bool,
    /// `white_balance` is Auto.
    pub white_balance: bool,
    /// `sharpen` is Auto.
    pub sharpen: bool,
    /// `noise_reduction` is Auto.
    pub noise_reduction: bool,
}

/// [`DevelopSettings`] with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedValues {
    /// `look`.
    pub look: LookId,
    /// `look_intensity`.
    pub look_intensity: f64,
    /// The exposure in stops (`EV_auto` while `exposure` is Auto).
    pub exposure_ev: f64,
    /// The contrast in effect: the manual value, or the photo's Auto contrast while `contrast` is Auto
    /// (0 without Apple Auto, `docs/19` section 5).
    pub contrast: f64,
    /// `brightness`.
    pub brightness: f64,
    /// The highlights in effect: the manual value, or the photo's Auto value (0 without Apple Auto).
    pub highlights: f64,
    /// The shadows in effect, as `highlights`.
    pub shadows: f64,
    /// `white`.
    pub white: f64,
    /// `white_offset_ev`.
    pub white_offset_ev: f64,
    /// The white lift in effect in stops: the automatic amount plus `white_offset_ev`, never below 0
    /// (`docs/02` C12 step 1b).
    pub white_ev: f64,
    /// The black in effect, as `highlights`.
    pub black: f64,
    /// `light_source`.
    pub light_source: LightSource,
    /// The Kelvin the White balance row shows: the user's value, or the scene light's readout
    /// (the preset's temperature while a preset is chosen).
    pub kelvin: f64,
    /// The tint the White balance row shows.
    pub tint: f64,
    /// The wheels' points in effect: the manual points, or the photo's Auto points (`[0, 0]` without
    /// Apple Auto) for the shadows and highlights wheels while they are Auto.
    pub color_balance: ColorBalancePoints,
    /// The saturation in effect: the manual value, or the photo's Auto saturation while `saturation` is
    /// Auto (0 without Apple Auto).
    pub saturation: f64,
    /// `color_warp`.
    pub color_warp: ColorWarp,
    /// The sharpening in effect.
    pub sharpen: SharpenValue,
    /// `output_sharpening`.
    pub output_sharpening: f64,
    /// The noise reduction in effect.
    pub noise_reduction: NoiseReductionValue,
    /// `curve`.
    pub curve: Curves,
    /// `decode`.
    pub decode: DecodeOptions,
    /// `rolloff`.
    pub rolloff: Rolloff,
    /// `crop`.
    pub crop: Option<Crop>,
    /// `orientation`.
    pub orientation: Orientation,
    /// `white_pick`.
    pub white_pick: Option<[f64; 2]>,
}

/// What `get_settings` and `update_settings` return: the panel's whole state for one photo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedSettings {
    /// The photo.
    pub photo_id: PhotoId,
    /// The stored settings, with `Auto` kept as `Auto`.
    pub settings: DevelopSettings,
    /// The same settings with every `Auto` resolved.
    pub values: ResolvedValues,
    /// Which fields are still automatic.
    pub auto: AutoFlags,
    /// The automatic estimates the `Auto`s resolved against.
    pub estimates: AutoSummary,
}

impl ResolvedSettings {
    /// Resolves `settings` against the photo's estimates.
    ///
    /// The Kelvin and tint come from the colour chain, so they follow a light preset or a picked
    /// white while `white_balance` is Auto (`docs/02` C1 and C3-C4). `picked_white` is the sample
    /// that belongs to `settings.white_pick` ([`crate::auto::pick_white_sample`]); without it a
    /// white pick has no effect on the readout.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Chain`] when the colour chain cannot be built from the values.
    pub fn resolve(
        photo_id: PhotoId,
        settings: &DevelopSettings,
        estimates: &AutoEstimates,
        profile: &Profile,
        picked_white: Option<[f64; 3]>,
    ) -> Result<Self, SettingsError> {
        let chain = estimates
            .fused_chain(profile, settings, picked_white)
            .map_err(|e| match e {
                ChainError::Pick(p) => SettingsError::WhitePick(p.to_string()),
                other => SettingsError::Chain(other.to_string()),
            })?;
        let WbReadout { kelvin_k, tint } = chain.readout();
        let auto_sharpen = estimates.sharpen;
        let auto_nr = estimates.noise_reduction;
        let values = ResolvedValues {
            look: settings.look.clone(),
            look_intensity: settings.look_intensity,
            exposure_ev: chain.exposure_ev(),
            contrast: settings.contrast.resolve(estimates.contrast),
            brightness: settings.brightness,
            highlights: settings.highlights.resolve(estimates.highlights),
            shadows: settings.shadows.resolve(estimates.shadows),
            white: settings.white,
            white_offset_ev: settings.white_offset_ev,
            white_ev: estimates.white.lift(settings.white_offset_ev).amount_ev(),
            black: settings.black.resolve(estimates.black),
            light_source: settings.light_source,
            kelvin: kelvin_k,
            tint,
            color_balance: ColorBalancePoints {
                shadows: settings
                    .color_balance
                    .shadows
                    .resolve(estimates.balance_shadows),
                midtones: settings.color_balance.midtones,
                highlights: settings
                    .color_balance
                    .highlights
                    .resolve(estimates.balance_highlights),
            },
            saturation: settings.saturation.resolve(estimates.saturation),
            color_warp: settings.color_warp.clone(),
            sharpen: settings.sharpen.manual().copied().unwrap_or(auto_sharpen),
            output_sharpening: settings.output_sharpening,
            noise_reduction: settings
                .noise_reduction
                .manual()
                .copied()
                .unwrap_or(auto_nr),
            curve: settings.curve.clone(),
            decode: settings.decode,
            rolloff: settings.rolloff,
            crop: settings.crop,
            orientation: settings.orientation,
            white_pick: settings.white_pick,
        };
        Ok(Self {
            photo_id,
            settings: settings.clone(),
            values,
            auto: AutoFlags {
                exposure: settings.exposure.is_auto(),
                contrast: settings.contrast.is_auto(),
                highlights: settings.highlights.is_auto(),
                shadows: settings.shadows.is_auto(),
                black: settings.black.is_auto(),
                balance_shadows: settings.color_balance.shadows.is_auto(),
                balance_highlights: settings.color_balance.highlights.is_auto(),
                saturation: settings.saturation.is_auto(),
                light_source: settings.light_source == LightSource::Auto,
                white_balance: settings.white_balance.is_auto(),
                sharpen: settings.sharpen.is_auto(),
                noise_reduction: settings.noise_reduction.is_auto(),
            },
            estimates: estimates.summary(),
        })
    }

    /// The numbers that fill the missing part of a partial manual patch: what the panel shows.
    #[must_use]
    pub fn seeds(&self) -> AutoSeeds {
        AutoSeeds {
            exposure_ev: self.values.exposure_ev,
            kelvin: self.values.kelvin,
            tint: self.values.tint,
            sharpen: self.values.sharpen,
            noise_reduction: self.values.noise_reduction,
        }
    }

    /// The settings as an export records them (`ym:Settings`, the CLI's `settings` object).
    ///
    /// Exposure, contrast, highlights, shadows, black, saturation, the shadows and highlights wheels,
    /// sharpening and noise reduction that are Auto are written as the manual values they resolved to: a Manual value renders exactly like the Auto it came from, and
    /// the file then says which numbers were used (which Auto chose them is in `ym:Auto` and
    /// `ym:AppleAuto`, `docs/19` section 9). White balance is left as it is: a manual white balance
    /// at the Kelvin and tint shown does *not* render like Auto (the tint scales the green gain,
    /// and Kelvin is rounded to 50 K), so an Auto white balance stays Auto here and the numbers it
    /// resolved to are in the `auto` object of the CLI output.
    #[must_use]
    pub fn pinned_settings(&self) -> DevelopSettings {
        let mut s = self.settings.clone();
        if s.exposure.is_auto() {
            s.exposure = ExposureSetting::manual_ev(self.values.exposure_ev);
        }
        if s.contrast.is_auto() {
            s.contrast = PercentSetting::manual_value(self.values.contrast);
        }
        if s.saturation.is_auto() {
            s.saturation = PercentSetting::manual_value(self.values.saturation);
        }
        for (setting, value) in [
            (&mut s.highlights, self.values.highlights),
            (&mut s.shadows, self.values.shadows),
            (&mut s.black, self.values.black),
        ] {
            if setting.is_auto() {
                *setting = PercentSetting::manual_value(value);
            }
        }
        for (setting, point) in [
            (
                &mut s.color_balance.shadows,
                self.values.color_balance.shadows,
            ),
            (
                &mut s.color_balance.highlights,
                self.values.color_balance.highlights,
            ),
        ] {
            if setting.is_auto() {
                *setting = BalanceSetting::manual_point(point);
            }
        }
        if s.sharpen.is_auto() {
            s.sharpen = AutoOr::Manual(self.values.sharpen);
        }
        if s.noise_reduction.is_auto() {
            s.noise_reduction = AutoOr::Manual(self.values.noise_reduction);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_the_automatic_development() {
        let s = DevelopSettings::default();
        assert_eq!(s.look.as_str(), "builtin:neutral");
        assert_eq!(s.look_intensity, 100.0);
        assert!(s.exposure.is_auto() && s.white_balance.is_auto());
        assert!(s.sharpen.is_auto() && s.noise_reduction.is_auto());
        assert_eq!(s.light_source, LightSource::Auto);
        assert_eq!(s.curve.luma, identity_curve());
        assert_eq!(
            s.rolloff,
            Rolloff {
                highlights: 1.0,
                low_light: 0.5
            }
        );
        assert!(s.decode.neural_demosaic && s.decode.chromatic_aberration);
        assert!(s.crop.is_none() && s.white_pick.is_none());
        assert_eq!(s.orientation, Orientation::Deg0);
        assert_eq!(s.output_sharpening, 100.0);
    }

    #[test]
    fn output_sharpening_is_a_plain_number_that_clamps_and_defaults_to_100() {
        // A document without the field reads as 100 (older settings, older XMP).
        let (s, clamped) =
            DevelopSettings::from_json("{}", &LookId::builtin("neutral")).expect("reads");
        assert_eq!(s.output_sharpening, 100.0);
        assert!(clamped.is_empty());
        let (s, clamped) = DevelopSettings::from_json(
            r#"{ "output_sharpening": 250 }"#,
            &LookId::builtin("neutral"),
        )
        .expect("reads");
        assert_eq!(s.output_sharpening, 200.0);
        assert_eq!(clamped, vec![SettingsField::OutputSharpening]);
        let (s, _) = DevelopSettings::from_json(
            r#"{ "output_sharpening": -5 }"#,
            &LookId::builtin("neutral"),
        )
        .expect("reads");
        assert_eq!(s.output_sharpening, 0.0);
        // A value that is not a number is refused, like every other field.
        assert!(
            DevelopSettings::from_json(
                r#"{ "output_sharpening": "a lot" }"#,
                &LookId::builtin("neutral")
            )
            .is_err()
        );
    }

    #[test]
    fn the_white_offset_is_a_plain_number_that_clamps_and_defaults_to_zero() {
        let neutral = LookId::builtin("neutral");
        // A document without the field reads as 0, the automatic amount (older settings, older XMP).
        let (s, clamped) = DevelopSettings::from_json("{}", &neutral).expect("reads");
        assert_eq!(s.white_offset_ev, 0.0);
        assert!(clamped.is_empty());
        let (s, clamped) =
            DevelopSettings::from_json(r#"{ "white_offset_ev": 5 }"#, &neutral).expect("reads");
        assert_eq!(s.white_offset_ev, 3.0);
        assert_eq!(clamped, vec![SettingsField::WhiteOffset]);
        let (s, _) =
            DevelopSettings::from_json(r#"{ "white_offset_ev": -9.5 }"#, &neutral).expect("reads");
        assert_eq!(s.white_offset_ev, -3.0);
        let (s, clamped) =
            DevelopSettings::from_json(r#"{ "white_offset_ev": -0.35 }"#, &neutral).expect("reads");
        assert_eq!(s.white_offset_ev, -0.35);
        assert!(clamped.is_empty());
        // Not a number: refused, like every other field.
        assert!(DevelopSettings::from_json(r#"{ "white_offset_ev": "a lot" }"#, &neutral).is_err());
        // The patch sets it, Reset puts it back, and it belongs to the grade group (C12 step 1b).
        let mut s = DevelopSettings::default();
        let report = s
            .apply_patch(
                &SettingsPatch {
                    white_offset_ev: Some(0.5),
                    ..SettingsPatch::default()
                },
                None,
            )
            .expect("patch");
        assert_eq!(s.white_offset_ev, 0.5);
        assert_eq!(report.changed, vec![SettingsField::WhiteOffset]);
        s.reset_field(SettingsField::WhiteOffset, &neutral);
        assert_eq!(s.white_offset_ev, 0.0);
        assert_eq!(SettingsField::WhiteOffset.name(), "white_offset_ev");
        assert_eq!(SettingsField::WhiteOffset.group(), StageGroup::Grade);
    }

    #[test]
    fn orientation_serialises_as_degrees_and_normalises() {
        assert_eq!(serde_json::to_string(&Orientation::Deg270).unwrap(), "270");
        for (text, want) in [
            ("0", Orientation::Deg0),
            ("90", Orientation::Deg90),
            ("-90", Orientation::Deg270),
            ("360", Orientation::Deg0),
            ("450", Orientation::Deg90),
        ] {
            assert_eq!(
                serde_json::from_str::<Orientation>(text).unwrap(),
                want,
                "{text}"
            );
        }
        assert!(serde_json::from_str::<Orientation>("45").is_err());
        assert!(serde_json::from_str::<Orientation>("\"90\"").is_err());
        assert_eq!(Orientation::Deg270.rotated_cw(), Orientation::Deg0);
    }

    #[test]
    fn stage_groups_follow_the_render_graph_table() {
        assert_eq!(SettingsField::Decode.group(), StageGroup::Raw);
        assert_eq!(SettingsField::NoiseReduction.group(), StageGroup::Raw);
        assert_eq!(SettingsField::Sharpen.group(), StageGroup::Raw);
        for f in [
            SettingsField::LightSource,
            SettingsField::WhiteBalance,
            SettingsField::WhitePick,
            SettingsField::Exposure,
            SettingsField::Rolloff,
        ] {
            assert_eq!(f.group(), StageGroup::Color, "{f:?}");
        }
        for f in [SettingsField::Crop, SettingsField::Orientation] {
            assert_eq!(f.group(), StageGroup::Geometry);
        }
        // Output sharpening is the last stage: it follows geometry and re-runs nothing before it.
        assert_eq!(SettingsField::OutputSharpening.group(), StageGroup::Output);
        assert!(StageGroup::Geometry < StageGroup::Output);
        assert_eq!(SettingsField::Contrast.group(), StageGroup::Grade);
        assert_eq!(SettingsField::Look.group(), StageGroup::Grade);
        assert!(StageGroup::Raw < StageGroup::Color && StageGroup::Color < StageGroup::Grade);
        // Every field has a name that is the serde name.
        for f in SettingsField::ALL {
            assert_eq!(
                serde_json::to_string(&f).unwrap(),
                format!("\"{}\"", f.name())
            );
        }
    }
}
