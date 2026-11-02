//! `DevelopSettings`: the settings of a photo, which are **a graph of nodes** (`docs/23-nodes.md`
//! section 3, the owner's decision of 2026-11-01), the node patch that `update_settings` applies,
//! and `ResolvedSettings`, the settings with every `Auto` resolved against the photo's automatic
//! estimates (`docs/01-architecture.md` section DevelopSettings).
//!
//! # The graph
//!
//! `DevelopSettings` has one field, `nodes`, a [`NodeGraph`]: the Input (`"input"`), Main (the
//! corrector with id 1), any other correctors and the Output (`"output"`), with typed links between
//! them. Every field that used to be a top-level setting is a field of one node under the same name
//! and with the same type: the Input holds the raw, light and framing settings, the Output the look
//! and the output sharpening, a corrector (Main included) its grade, trims and masks. The rest of
//! the core reads them through [`DevelopSettings::input`], [`DevelopSettings::main_grade`],
//! [`DevelopSettings::output`] and their `_mut` twins. Making the settings node-first moved fields
//! and changed no pixel: the render still uses the Input, Main and the Output as it used the flat
//! fields (the other nodes render from N-03 on).
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
//! Reading is lenient about *missing* fields (a document is read over the default graph: the three
//! fixed nodes need not be listed, a listed node's missing fields take their defaults, `links` left
//! out are the default links) and strict about *unknown* ones (a typo such as `exposur` is an error,
//! not a silent no-op). A document in the flat form of Yellowmoon 0.1 is refused
//! ([`SettingsError::OldForm`]). Writing always writes every field.
//!
//! # Auto and Manual
//!
//! The Input's `exposure`, `white_balance`, `sharpen` and `noise_reduction`, and Main's `contrast`,
//! `highlights`, `shadows`, `black`, `saturation` and shadows and highlights colour-balance wheels
//! are [`AutoOr`]: `Auto` resolves against [`AutoEstimates`] when the photo is rendered, `Manual` is
//! the user's value. **Auto belongs to Main**: in any other corrector Auto reads as a manual 0 and is
//! written as one. Auto contrast, highlights, shadows, black and saturation are 0, and the wheels
//! `[0, 0]`, unless Apple Auto answered for the photo (`docs/19` section 5); a plain number (or
//! point) in a document reads as Manual, and 0 (or `[0, 0]`) as Auto. A patch that carries a
//! `Manual` value is what "moving a slider" sends; a patch that carries `Auto` is Reset.
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
//! way `ym_color::warp::WarpPin::clamped` does ([`ColorWarp`]). The lens (`docs/21` section 2,
//! [`LensSettings`]) is a kind and three amounts, each clamped to [`LENS_AMOUNT_RANGE`]; a patch may
//! carry only the parts it changes ([`LensPatch`]). The rules of the graph itself (`docs/23`
//! section 3.5) are [`GraphError`].

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use ym_color::chain::ChainError;
use ym_color::exposure::ExposureSetting as ColorExposure;
use ym_color::warp::{WARP_COORD_LIMIT, WARP_MAX_PINS, WARP_RANGE_DEFAULT, WarpParams, WarpPin};
use ym_color::wb::{LightSource, WbReadout, WhiteBalance as ColorWhiteBalance};
use ym_lens::LensAmounts;
/// Which lens (`docs/21` section 2): the ids of `data/lenses/` and of `--lens`, `default`,
/// `classic`, `vintage`, `petzval`, `triplet`, `sonnar` and `detuned`, as JSON strings of the same
/// spelling.
pub use ym_lens::LensKind;
use ym_looks::library::LookId;
use ym_profiles::profile::Profile;

use crate::auto::{AutoEstimates, AutoSummary, MainAuto};
use crate::session::PhotoId;

mod graph;
mod masks;
mod patch;

pub use graph::{
    CorrectorNode, DEFAULT_NAME, GraphError, INPUT_POS, InputNode, MAIN_ID, MAIN_NAME,
    MAX_CORRECTORS, MAX_NAME_CHARS, NodeGrade, NodeGraph, NodeId, NodeKind, NodeLink, NodePos,
    NodeRef, NodeTrim, OUTPUT_POS, OutputNode, POS_RANGE, Port, TRIM_EXPOSURE_RANGE_EV, TRIM_RANGE,
};
pub use masks::{
    BackgroundMask, BrightnessRangeMask, BrushMask, BrushStroke, ColorRangeMask, FindMask,
    LinearMask, MASK_KINDS, MAX_BRUSH_POINTS, MAX_MASKS_PER_NODE, MAX_OBJECT_POINTS, MaskItem,
    MaskKind, MaskMode, MaskRowPatch, MaskSettings, MaskShape, MaskUnavailable, MasksPatch,
    NO_OBJECT_MODEL_SENTENCE, ObjectMask, RadialMask, mask_kinds, mask_kinds_with,
    mask_unavailable, mask_unavailable_with, unmakeable_mask_sentence,
    unmakeable_mask_sentence_with,
};
pub use patch::{COPY_LEAVES_OUT, GradePatch, NodePatch, PatchReport, SettingsPatch, TrimPatch};

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
/// Range of each manual noise-reduction strength.
pub const NOISE_REDUCTION_RANGE: (f64, f64) = (0.0, 100.0);
/// Range of both amounts of RAW › Range (`rolloff`): the highlight range and the low-light toe.
pub const ROLLOFF_RANGE: (f64, f64) = (0.0, 1.0);
/// Default highlight range (`rolloff.highlights`): 0.6. It was 1 until pipeline 0.8.0; the owner chose 0.6 on
/// 2026-10-30 ("gentler range", `docs/02` C7) together with a knee two stops under the camera's clip. A
/// document without the field reads as this; a document with an explicit value keeps it.
pub const ROLLOFF_HIGHLIGHTS_DEFAULT: f64 = 0.6;
/// Default low-light toe (`rolloff.low_light`): 0.5.
pub const ROLLOFF_LOW_LIGHT_DEFAULT: f64 = 0.5;
/// Range of the three lens amounts, Flares, Edges and Anamorphic (`docs/21` section 2).
pub const LENS_AMOUNT_RANGE: (f64, f64) = (0.0, 100.0);
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
    /// A number is not finite. The text names the field as a path from the document
    /// (`nodes.items[2].trim.tint`).
    #[error("The value for {0} is not a number.")]
    NotFinite(String),
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
    /// A rule of the graph is broken (`docs/23` section 3.5).
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// A flat settings document of Yellowmoon 0.1 (`docs/23` section 3.12). Nothing was saved in 0.1,
    /// so there is nothing to carry over.
    #[error(
        "This settings file is in the old form of Yellowmoon 0.1. Settings are now a graph of nodes: `ym render --help` shows the new form, and `ym render --json` prints a photo's settings in it."
    )]
    OldForm,
    /// A mask row patch names another kind than the row's.
    #[error("A mask can't change its kind. Add a new mask instead.")]
    MaskKind,
    /// A mask row has no such field.
    #[error("{mask} mask has no field called “{field}”.")]
    UnknownMaskField {
        /// "A Radial", "An Object".
        mask: String,
        /// The field.
        field: String,
    },
    /// A row patch names a row the node does not have.
    #[error("Node {node} has no mask number {index}.")]
    NoMaskRow {
        /// The node.
        node: String,
        /// The index as the patch gave it.
        index: String,
    },
}

/// The sentence a flat document of Yellowmoon 0.1 is refused with.
pub const OLD_FORM_SENTENCE: &str = "This settings file is in the old form of Yellowmoon 0.1. Settings are now a graph of nodes: `ym render --help` shows the new form, and `ym render --json` prints a photo's settings in it.";

/// The top-level names of a settings document of Yellowmoon 0.1 (`docs/23` section 3.12). A
/// document that has no `nodes` and only these keys is refused with [`SettingsError::OldForm`].
pub const OLD_FORM_KEYS: [&str; 25] = [
    "look",
    "look_intensity",
    "exposure",
    "contrast",
    "brightness",
    "highlights",
    "shadows",
    "white",
    "black",
    "light_source",
    "white_balance",
    "color_balance",
    "saturation",
    "color_warp",
    "lens",
    "sharpen",
    "output_sharpening",
    "noise_reduction",
    "curve",
    "decode",
    "rolloff",
    "crop",
    "orientation",
    "white_pick",
    "white_offset_ev",
];

// ---------------------------------------------------------------------------------------------
// Fields and stage groups
// ---------------------------------------------------------------------------------------------

/// The render-graph group of stages a setting first affects (`docs/01` section Render graph).
///
/// A change invalidates its group and every later one: `Raw` before `Lens` before `Colour` before
/// `Grade`.
/// `Geometry` is applied when sampling for display and export, so no pixel stage re-runs, and
/// `Output` (sharpening) is the last: it re-runs only its own pass over the rendered frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageGroup {
    /// Vignetting, highlights, demosaic and denoise, detail controls, sharpening, lens resample
    /// (R3 to R9).
    Raw,
    /// The emulated lens: bokeh reshaping, field optics and highlight optics (L1 to L3, `docs/21`
    /// section 3), on the light of the raw group. The anamorphic bend (L4) is geometry, applied
    /// while sampling.
    Lens,
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

/// One field of a node, for change reports and Reset. The same name can belong to the nodes of
/// several kinds (`pos` belongs to every node); [`SettingsField::belongs_to`] says which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingsField {
    /// The Output's `look`.
    Look,
    /// The Output's `look_intensity`.
    LookIntensity,
    /// The Input's `exposure`.
    Exposure,
    /// The grade's `contrast`.
    Contrast,
    /// The grade's `brightness`.
    Brightness,
    /// The grade's `highlights`.
    Highlights,
    /// The grade's `shadows`.
    Shadows,
    /// The grade's `white`.
    White,
    /// The grade's `black`.
    Black,
    /// The Input's `light_source`.
    LightSource,
    /// The Input's `white_balance`.
    WhiteBalance,
    /// The grade's `color_balance`.
    ColorBalance,
    /// The grade's `saturation`.
    Saturation,
    /// The grade's `color_warp`.
    ColorWarp,
    /// The Input's `lens`.
    Lens,
    /// The Input's `sharpen`.
    Sharpen,
    /// The Output's `output_sharpening`.
    OutputSharpening,
    /// The Input's `noise_reduction`.
    NoiseReduction,
    /// The grade's `curve`.
    Curve,
    /// The Input's `decode`.
    Decode,
    /// The Input's `rolloff`.
    Rolloff,
    /// The Input's `crop`.
    Crop,
    /// The Input's `orientation`.
    Orientation,
    /// The Input's `white_pick`.
    WhitePick,
    /// A corrector's `name`.
    Name,
    /// A corrector's `bypass`.
    Bypass,
    /// A node's `pos`.
    Pos,
    /// A corrector's `trim`.
    Trim,
    /// A corrector's `masks`.
    Masks,
}

impl SettingsField {
    /// Every field, in the order of the settings table (the Output's and the Input's, then the grade's,
    /// then the corrector's own).
    pub const ALL: [SettingsField; 29] = [
        SettingsField::Look,
        SettingsField::LookIntensity,
        SettingsField::Exposure,
        SettingsField::Contrast,
        SettingsField::Brightness,
        SettingsField::Highlights,
        SettingsField::Shadows,
        SettingsField::White,
        SettingsField::Black,
        SettingsField::LightSource,
        SettingsField::WhiteBalance,
        SettingsField::ColorBalance,
        SettingsField::Saturation,
        SettingsField::ColorWarp,
        SettingsField::Lens,
        SettingsField::Sharpen,
        SettingsField::OutputSharpening,
        SettingsField::NoiseReduction,
        SettingsField::Curve,
        SettingsField::Decode,
        SettingsField::Rolloff,
        SettingsField::Crop,
        SettingsField::Orientation,
        SettingsField::WhitePick,
        SettingsField::Name,
        SettingsField::Bypass,
        SettingsField::Pos,
        SettingsField::Trim,
        SettingsField::Masks,
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
            Self::Black => "black",
            Self::LightSource => "light_source",
            Self::WhiteBalance => "white_balance",
            Self::ColorBalance => "color_balance",
            Self::Saturation => "saturation",
            Self::ColorWarp => "color_warp",
            Self::Lens => "lens",
            Self::Sharpen => "sharpen",
            Self::OutputSharpening => "output_sharpening",
            Self::NoiseReduction => "noise_reduction",
            Self::Curve => "curve",
            Self::Decode => "decode",
            Self::Rolloff => "rolloff",
            Self::Crop => "crop",
            Self::Orientation => "orientation",
            Self::WhitePick => "white_pick",
            Self::Name => "name",
            Self::Bypass => "bypass",
            Self::Pos => "pos",
            Self::Trim => "trim",
            Self::Masks => "masks",
        }
    }

    /// True for the fields of a corrector's grade (`contrast`, `brightness`, `highlights`,
    /// `shadows`, `white`, `black`, `color_balance`, `saturation`, `color_warp`, `curve`).
    #[must_use]
    pub fn is_grade(self) -> bool {
        matches!(
            self,
            Self::Contrast
                | Self::Brightness
                | Self::Highlights
                | Self::Shadows
                | Self::White
                | Self::Black
                | Self::ColorBalance
                | Self::Saturation
                | Self::ColorWarp
                | Self::Curve
        )
    }

    /// The field's path in its node's JSON: `grade.contrast` for a grade field, else its name.
    #[must_use]
    pub fn path(self) -> String {
        if self.is_grade() {
            format!("grade.{}", self.name())
        } else {
            self.name().to_owned()
        }
    }

    /// Whether a node of this kind has the field.
    #[must_use]
    pub fn belongs_to(self, kind: NodeKind) -> bool {
        match kind {
            NodeKind::Input => matches!(
                self,
                Self::Exposure
                    | Self::LightSource
                    | Self::WhiteBalance
                    | Self::WhitePick
                    | Self::Rolloff
                    | Self::Decode
                    | Self::NoiseReduction
                    | Self::Sharpen
                    | Self::Lens
                    | Self::Crop
                    | Self::Orientation
                    | Self::Pos
            ),
            NodeKind::Output => matches!(
                self,
                Self::Look | Self::LookIntensity | Self::OutputSharpening | Self::Pos
            ),
            NodeKind::Corrector => {
                self.is_grade()
                    || matches!(
                        self,
                        Self::Name | Self::Bypass | Self::Pos | Self::Trim | Self::Masks
                    )
            }
        }
    }

    /// The first group of render stages this field affects (`docs/01` section Render graph);
    /// `None` for the fields that re-run nothing (`name` and `pos`).
    #[must_use]
    pub fn group(self) -> Option<StageGroup> {
        Some(match self {
            Self::Decode | Self::NoiseReduction | Self::Sharpen => StageGroup::Raw,
            Self::Lens => StageGroup::Lens,
            Self::LightSource
            | Self::WhiteBalance
            | Self::WhitePick
            | Self::Exposure
            | Self::Rolloff => StageGroup::Color,
            Self::Crop | Self::Orientation => StageGroup::Geometry,
            Self::OutputSharpening => StageGroup::Output,
            Self::Name | Self::Pos => return None,
            Self::Look
            | Self::LookIntensity
            | Self::Bypass
            | Self::Trim
            | Self::Masks
            | Self::Contrast
            | Self::Brightness
            | Self::Highlights
            | Self::Shadows
            | Self::White
            | Self::Black
            | Self::ColorBalance
            | Self::Saturation
            | Self::ColorWarp
            | Self::Curve => StageGroup::Grade,
        })
    }
}

/// A field of a node: what [`PatchReport`] names. Written `<node>.<field path>`, for example
/// `1.grade.contrast` or `input.exposure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldRef {
    /// The node.
    pub node: NodeId,
    /// The field.
    pub field: SettingsField,
}

impl FieldRef {
    /// A field of a node.
    #[must_use]
    pub const fn new(node: NodeId, field: SettingsField) -> Self {
        Self { node, field }
    }
}

impl std::fmt::Display for FieldRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.node, self.field.path())
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

/// The three wheels' points in effect ([`GradeValues::color_balance`]).
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
/// false, `range` 0.02, `shadows` and `highlights` 1, `pivot_ev`, `exposure_ev` and `saturation` 0); unknown fields
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
    /// A change of saturation of the pin's colours, -1 to +1 (shown as -100 to +100 %): -1 turns
    /// them grey, +1 doubles their distance from neutral.
    pub saturation: f64,
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
    #[serde(default)]
    saturation: f64,
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
            saturation: d.saturation,
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
            saturation: p.saturation,
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
            saturation: p.saturation,
        }
    }
}

/// Colour › Colour warp: its pins, in order; no pins by default (the stage is then skipped).
///
/// At most [`WARP_MAX_PINS`]: checking the settings cuts the list there and clamps each pin as
/// `WarpPin::clamped` does (points scaled back to `|c| <= 0.5`, the range to 0.01…0.25, the tonal
/// weights to 0…1, the pivot to ±6, the exposure to ±2 stops and the saturation to ±1; a locked
/// pin's target is its source, and its weights, pivot, exposure and saturation are the defaults).
/// A patch replaces the whole list.
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

/// RAW › Range (`docs/02` C7), each 0 to 1: Highlights 0.6 ([`ROLLOFF_HIGHLIGHTS_DEFAULT`]) and Low light 0.5
/// by default. The JSON name `rolloff` is kept from the earlier roll-off card. A document that leaves
/// `highlights` out reads as 0.6 (`default` on the struct); an explicit value, 1 included, is kept.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Rolloff {
    /// The highlight range of C7 (`docs/02`): where the camera's clipping point lands, from where
    /// it was recorded (0) to white, the clip of an ALEXA at EI 800 (1); default 0.6. Mid grey and
    /// everything under the knee, two stops below the camera's clip, never move.
    pub highlights: f64,
    /// Low-light toe strength (C7).
    pub low_light: f64,
}

impl Default for Rolloff {
    fn default() -> Self {
        Self {
            highlights: ROLLOFF_HIGHLIGHTS_DEFAULT,
            low_light: ROLLOFF_LOW_LIGHT_DEFAULT,
        }
    }
}

/// Look › Lens: a lens and three amounts, each 0 to 100 ([`LENS_AMOUNT_RANGE`]).
///
/// All three amounts at 0 is no lens, whatever `kind` says ([`LensSettings::is_identity`]): no lens
/// stage runs and the render has the bytes it has without the field. Choosing a lens does not move
/// the amounts by itself: the card and `--lens` send the lens's starting amounts (`LensInfo`) in
/// the same patch (`docs/21` section 12, decision 6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LensSettings {
    /// The lens.
    pub kind: LensKind,
    /// Glow, veiling glare, ghosts and starbursts (L3), 0 to 100.
    pub flares: f64,
    /// Edge softness, darkening, colour fringes and the lens's distortion (L2 and L4), 0 to 100.
    pub edges: f64,
    /// The anamorphic attachment in front of the lens (L1 to L4), 0 to 100: taller bokeh that takes the lens's
    /// own swirl and cat's-eye shapes, streaks and the bend.
    pub anamorphic: f64,
}

impl Default for LensSettings {
    /// The Default lens with all three amounts at 0: no lens.
    fn default() -> Self {
        Self {
            kind: LensKind::Default,
            flares: 0.0,
            edges: 0.0,
            anamorphic: 0.0,
        }
    }
}

impl LensSettings {
    /// True when all three amounts are 0, whatever the lens: no lens stage runs.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.flares == 0.0 && self.edges == 0.0 && self.anamorphic == 0.0
    }
}

impl From<LensSettings> for LensAmounts {
    fn from(l: LensSettings) -> Self {
        LensAmounts {
            kind: l.kind,
            flares: l.flares,
            edges: l.edges,
            anamorphic: l.anamorphic,
        }
    }
}

impl From<LensAmounts> for LensSettings {
    fn from(a: LensAmounts) -> Self {
        LensSettings {
            kind: a.kind,
            flares: a.flares,
            edges: a.edges,
            anamorphic: a.anamorphic,
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
    /// The rect and angle came from Auto crop and follow it while the Crop panel is open
    /// (`docs/20-auto-crop.md` §2). Rendering never reads it. JSON: absent is `false`, and `false`
    /// is not written.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub auto: bool,
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

/// The settings of one photo: the graph of its nodes (`docs/23-nodes.md` section 3). Every control
/// in the UI and every CLI flag is a field of one node (`docs/01-architecture.md` section
/// DevelopSettings). Defaults reproduce the automatic development.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DevelopSettings {
    /// The graph: the Input, Main, any other correctors, the Output and the links.
    pub nodes: NodeGraph,
}

impl Default for DevelopSettings {
    /// Auto everywhere and the identity look, Neutral: the blank development, which needs no look
    /// library to render (tests, a core without one) and is what `settings.schema.json` calls the
    /// default. It is not what a user sees first: a photo a session opens starts from
    /// [`DevelopSettings::with_look`] with the user's default look (Signature out of the box,
    /// `ym_core::preferences::DEFAULT_LOOK`), and a settings document that names no look reads as
    /// that look too ([`DevelopSettings::from_json`]).
    fn default() -> Self {
        Self::with_look(LookId::neutral())
    }
}

impl DevelopSettings {
    /// The settings of a photo nobody has touched: the default graph, Auto everywhere, `look` at 100
    /// percent on the Output.
    #[must_use]
    pub fn with_look(look: LookId) -> Self {
        Self {
            nodes: NodeGraph::with_look(look),
        }
    }

    /// The Input: the raw, light and framing settings.
    #[must_use]
    pub fn input(&self) -> &InputNode {
        &self.nodes.input
    }

    /// The Input, to change.
    pub fn input_mut(&mut self) -> &mut InputNode {
        &mut self.nodes.input
    }

    /// Main, node 1: the corrector the simple view's grade controls and Auto belong to.
    #[must_use]
    pub fn main(&self) -> &CorrectorNode {
        &self.nodes.main
    }

    /// Main, to change.
    pub fn main_mut(&mut self) -> &mut CorrectorNode {
        &mut self.nodes.main
    }

    /// Main's grade.
    #[must_use]
    pub fn main_grade(&self) -> &NodeGrade {
        &self.nodes.main.grade
    }

    /// Main's grade, to change.
    pub fn main_grade_mut(&mut self) -> &mut NodeGrade {
        &mut self.nodes.main.grade
    }

    /// The Output: the look and the output sharpening.
    #[must_use]
    pub fn output(&self) -> &OutputNode {
        &self.nodes.output
    }

    /// The Output, to change.
    pub fn output_mut(&mut self) -> &mut OutputNode {
        &mut self.nodes.output
    }

    /// What "Re-run" on the Automatic card resets: the Input's exposure, white balance (the light
    /// preset, the Kelvin and tint and a picked white) and noise reduction go back to Auto. The
    /// values Apple Auto sets besides those (Main's highlights, shadows, black, contrast, saturation
    /// and the shadows and highlights wheels) go back to Auto only when `apple_auto` says Apple Auto
    /// acts on the photo (its state is pending, answered or failed: the card then shows their rows,
    /// [`AppleAuto::acts`]); otherwise they stay as they are, as before Apple Auto (`docs/19`
    /// section 5). The rest of the graph is left alone.
    ///
    /// [`AppleAuto::acts`]: crate::apple_auto::AppleAuto::acts
    pub fn reset_automatic(&mut self, apple_auto: bool) {
        let input = &mut self.nodes.input;
        input.exposure = AutoOr::Auto;
        input.light_source = LightSource::Auto;
        input.white_balance = AutoOr::Auto;
        input.white_pick = None;
        input.noise_reduction = AutoOr::Auto;
        if apple_auto {
            let g = &mut self.nodes.main.grade;
            g.contrast = AutoOr::Auto;
            g.saturation = AutoOr::Auto;
            g.highlights = AutoOr::Auto;
            g.shadows = AutoOr::Auto;
            g.black = AutoOr::Auto;
            g.color_balance.shadows = AutoOr::Auto;
            g.color_balance.highlights = AutoOr::Auto;
        }
    }

    /// Reads a settings document (the `ym:Settings` JSON, `ym render --settings`, a golden
    /// manifest) over the default graph the way a photo that was just opened would take it: fields
    /// the document leaves out have their defaults, and a document without an Output `look` gets
    /// `default_look`, the user's default look (`Default` alone would say Neutral). Values are then
    /// clamped like [`DevelopSettings::validated`] does; the fields that were changed are
    /// returned.
    ///
    /// # Errors
    ///
    /// [`SettingsError::OldForm`] for a flat document of Yellowmoon 0.1, [`SettingsError::Parse`]
    /// for text that is not JSON, a field that does not exist or a value of the wrong type, and the
    /// errors of [`DevelopSettings::validated`] and of the rules of the graph.
    pub fn from_json(
        text: &str,
        default_look: &LookId,
    ) -> Result<(Self, Vec<FieldRef>), SettingsError> {
        let value: Value =
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
        value: Value,
        default_look: &LookId,
    ) -> Result<(Self, Vec<FieldRef>), SettingsError> {
        let Value::Object(mut map) = value else {
            return Err(SettingsError::Parse(
                "the settings must be a JSON object".to_owned(),
            ));
        };
        let nodes = map.remove("nodes");
        if nodes.is_none()
            && !map.is_empty()
            && map.keys().all(|k| OLD_FORM_KEYS.contains(&k.as_str()))
        {
            return Err(SettingsError::OldForm);
        }
        if let Some(key) = map.keys().next() {
            return Err(SettingsError::Parse(format!(
                "unknown field `{key}`, expected `nodes`"
            )));
        }
        let graph = match nodes {
            Some(v) => NodeGraph::read(v, default_look)?,
            None => NodeGraph::with_look(default_look.clone()),
        };
        Self { nodes: graph }.validated()
    }

    /// The settings after clamping every value to its range and normalising curves and the crop,
    /// and after checking the rules of the graph.
    ///
    /// Settings read from a file (`--settings`, a golden manifest) go through this before use;
    /// patches are checked by [`DevelopSettings::apply_patch`].
    ///
    /// # Errors
    ///
    /// [`SettingsError`] for a value that is not a number, a curve that cannot be used, a crop
    /// without area or a rule of the graph that is broken.
    pub fn validated(mut self) -> Result<(Self, Vec<FieldRef>), SettingsError> {
        let mut s = graph::Sanitiser::default();
        self.nodes.sanitise(&mut s)?;
        Ok((self, s.clamped))
    }
}

/// Clamps a number to a range; a value that is not finite is an error. The flag says whether the
/// value was changed.
fn clamp_to(name: &str, v: f64, (lo, hi): (f64, f64)) -> Result<(f64, bool), SettingsError> {
    if !v.is_finite() {
        return Err(SettingsError::NotFinite(name.to_owned()));
    }
    let c = v.clamp(lo, hi);
    Ok((c + 0.0, c != v))
}

/// Scales a point back into the unit disc.
fn clamp_to_disc(name: &str, [a, b]: [f64; 2]) -> Result<([f64; 2], bool), SettingsError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(SettingsError::NotFinite(name.to_owned()));
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
fn normalise_warp(warp: &mut ColorWarp, prefix: &str) -> Result<bool, SettingsError> {
    for pin in &warp.pins {
        for (name, values) in [
            ("source", &pin.source[..]),
            ("target", &pin.target[..]),
            ("range", &[pin.range][..]),
            ("shadows", &[pin.shadows][..]),
            ("highlights", &[pin.highlights][..]),
            ("pivot_ev", &[pin.pivot_ev][..]),
            ("exposure_ev", &[pin.exposure_ev][..]),
            ("saturation", &[pin.saturation][..]),
        ] {
            if !values.iter().all(|v| v.is_finite()) {
                return Err(SettingsError::NotFinite(format!(
                    "{prefix}.color_warp.{name}"
                )));
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
            saturation: c.saturation + 0.0,
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
    prefix: &str,
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
            return Err(SettingsError::NotFinite(format!(
                "{prefix}.curve.{channel}"
            )));
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
fn normalise_crop(crop: &mut Crop, prefix: &str) -> Result<bool, SettingsError> {
    let before = *crop;
    let r = &mut crop.rect;
    if ![r.x, r.y, r.width, r.height, crop.angle_deg]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(SettingsError::NotFinite(format!("{prefix}.crop")));
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

/// Look › Lens parts to change; the others stay as they are (`{ "lens": { "flares": 40 } }`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LensPatch {
    /// New lens. The amounts do not follow it: send the lens's starting amounts in the same patch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<LensKind>,
    /// New Flares.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flares: Option<f64>,
    /// New Edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<f64>,
    /// New Anamorphic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anamorphic: Option<f64>,
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

// ---------------------------------------------------------------------------------------------
// Resolved settings
// ---------------------------------------------------------------------------------------------

/// Which of the Input's `Auto` fields are still automatic; the UI shows the AUTO tag for these and
/// EDITED for the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAutoFlags {
    /// `exposure` is Auto.
    pub exposure: bool,
    /// `light_source` is Auto.
    pub light_source: bool,
    /// `white_balance` is Auto.
    pub white_balance: bool,
    /// `sharpen` is Auto.
    pub sharpen: bool,
    /// `noise_reduction` is Auto.
    pub noise_reduction: bool,
}

/// Which of Main's `Auto` fields are still automatic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MainAutoFlags {
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
}

/// Which `Auto` fields are still automatic, grouped by node: only the Input and Main can be Auto
/// (`docs/23` section 3.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoFlags {
    /// The Input's.
    pub input: InputAutoFlags,
    /// Main's.
    pub main: MainAutoFlags,
}

/// The Input with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedInput {
    /// The card's place.
    pub pos: NodePos,
    /// The exposure in stops (`EV_auto` while `exposure` is Auto).
    pub exposure_ev: f64,
    /// `light_source`.
    pub light_source: LightSource,
    /// The Kelvin the White balance row shows: the user's value, or the scene light's readout
    /// (the preset's temperature while a preset is chosen).
    pub kelvin: f64,
    /// The tint the White balance row shows.
    pub tint: f64,
    /// The sharpening in effect.
    pub sharpen: SharpenValue,
    /// The noise reduction in effect.
    pub noise_reduction: NoiseReductionValue,
    /// `lens`.
    pub lens: LensSettings,
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

/// The Output (it has no Auto, so its values are its fields).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedOutput {
    /// The card's place.
    pub pos: NodePos,
    /// `look`.
    pub look: LookId,
    /// `look_intensity`.
    pub look_intensity: f64,
    /// `output_sharpening`.
    pub output_sharpening: f64,
}

/// A corrector's grade with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradeValues {
    /// The contrast in effect: the manual value, or the photo's Auto contrast while `contrast` is
    /// Auto (0 without Apple Auto, `docs/19` section 5).
    pub contrast: f64,
    /// `brightness`.
    pub brightness: f64,
    /// The highlights in effect: the manual value, or the photo's Auto value (0 without Apple Auto).
    pub highlights: f64,
    /// The shadows in effect, as `highlights`.
    pub shadows: f64,
    /// `white`.
    pub white: f64,
    /// The black in effect, as `highlights`.
    pub black: f64,
    /// The wheels' points in effect: the manual points, or the photo's Auto points (`[0, 0]` without
    /// Apple Auto) for the shadows and highlights wheels while they are Auto.
    pub color_balance: ColorBalancePoints,
    /// The saturation in effect.
    pub saturation: f64,
    /// `color_warp`.
    pub color_warp: ColorWarp,
    /// `curve`.
    pub curve: Curves,
}

impl GradeValues {
    /// The grade with its Auto resolved against `auto`.
    #[must_use]
    pub fn of(g: &NodeGrade, auto: &MainAuto) -> Self {
        Self {
            contrast: g.contrast.resolve(auto.contrast),
            brightness: g.brightness,
            highlights: g.highlights.resolve(auto.highlights),
            shadows: g.shadows.resolve(auto.shadows),
            white: g.white,
            black: g.black.resolve(auto.black),
            color_balance: ColorBalancePoints {
                shadows: g.color_balance.shadows.resolve(auto.balance_shadows),
                midtones: g.color_balance.midtones,
                highlights: g.color_balance.highlights.resolve(auto.balance_highlights),
            },
            saturation: g.saturation.resolve(auto.saturation),
            color_warp: g.color_warp.clone(),
            curve: g.curve.clone(),
        }
    }
}

/// A corrector with its grade resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCorrector {
    /// The node's id (Main is 1).
    pub id: u32,
    /// `name`.
    pub name: String,
    /// `bypass`.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// `trim`.
    pub trim: NodeTrim,
    /// The grade with its Auto resolved (only Main has Auto).
    pub grade: GradeValues,
    /// `masks`.
    pub masks: MaskSettings,
}

/// The settings with every `Auto` replaced by the number it resolves to, node by node
/// (`docs/23` section 3.9): the Input, Main, the Output and the other correctors as they are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedValues {
    /// The Input.
    pub input: ResolvedInput,
    /// Main, node 1.
    pub main: ResolvedCorrector,
    /// The other correctors, by id (their values are their fields: they have no Auto).
    pub nodes: Vec<ResolvedCorrector>,
    /// The Output.
    pub output: ResolvedOutput,
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
    /// The automatic estimates the `Auto`s resolved against, grouped by node.
    pub estimates: AutoSummary,
}

impl ResolvedSettings {
    /// Resolves `settings` against the photo's estimates.
    ///
    /// The Kelvin and tint come from the colour chain, so they follow a light preset or a picked
    /// white while `white_balance` is Auto (`docs/02` C1 and C3-C4). `picked_white` is the sample
    /// that belongs to the Input's `white_pick` ([`crate::auto::pick_white_sample`]); without it a
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
        let input = settings.input();
        let auto_input = estimates.input_auto();
        let main_auto = estimates.main_auto();
        let none = MainAuto {
            contrast: 0.0,
            saturation: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            black: 0.0,
            balance_shadows: [0.0, 0.0],
            balance_highlights: [0.0, 0.0],
        };
        let corrector = |id: u32, c: &CorrectorNode, auto: &MainAuto| ResolvedCorrector {
            id,
            name: c.name.clone(),
            bypass: c.bypass,
            pos: c.pos,
            trim: c.trim,
            grade: GradeValues::of(&c.grade, auto),
            masks: c.masks.clone(),
        };
        let values = ResolvedValues {
            input: ResolvedInput {
                pos: input.pos,
                exposure_ev: chain.exposure_ev(),
                light_source: input.light_source,
                kelvin: kelvin_k,
                tint,
                sharpen: input
                    .sharpen
                    .manual()
                    .copied()
                    .unwrap_or(auto_input.sharpen),
                noise_reduction: input
                    .noise_reduction
                    .manual()
                    .copied()
                    .unwrap_or(auto_input.noise_reduction),
                lens: input.lens,
                decode: input.decode,
                rolloff: input.rolloff,
                crop: input.crop,
                orientation: input.orientation,
                white_pick: input.white_pick,
            },
            main: corrector(MAIN_ID, settings.main(), &main_auto),
            nodes: settings
                .nodes
                .extra
                .iter()
                .map(|(id, c)| corrector(*id, c, &none))
                .collect(),
            output: ResolvedOutput {
                pos: settings.output().pos,
                look: settings.output().look.clone(),
                look_intensity: settings.output().look_intensity,
                output_sharpening: settings.output().output_sharpening,
            },
        };
        let g = settings.main_grade();
        Ok(Self {
            photo_id,
            settings: settings.clone(),
            values,
            auto: AutoFlags {
                input: InputAutoFlags {
                    exposure: input.exposure.is_auto(),
                    light_source: input.light_source == LightSource::Auto,
                    white_balance: input.white_balance.is_auto(),
                    sharpen: input.sharpen.is_auto(),
                    noise_reduction: input.noise_reduction.is_auto(),
                },
                main: MainAutoFlags {
                    contrast: g.contrast.is_auto(),
                    highlights: g.highlights.is_auto(),
                    shadows: g.shadows.is_auto(),
                    black: g.black.is_auto(),
                    balance_shadows: g.color_balance.shadows.is_auto(),
                    balance_highlights: g.color_balance.highlights.is_auto(),
                    saturation: g.saturation.is_auto(),
                },
            },
            estimates: estimates.summary(),
        })
    }

    /// The numbers that fill the missing part of a partial manual patch: what the panel shows.
    #[must_use]
    pub fn seeds(&self) -> AutoSeeds {
        let v = &self.values.input;
        AutoSeeds {
            exposure_ev: v.exposure_ev,
            kelvin: v.kelvin,
            tint: v.tint,
            sharpen: v.sharpen,
            noise_reduction: v.noise_reduction,
        }
    }

    /// The settings as an export records them (`ym:Settings`, the CLI's `settings` object).
    ///
    /// The Input's exposure, sharpening and noise reduction, and Main's contrast, highlights,
    /// shadows, black, saturation and shadows and highlights wheels that are Auto are written as
    /// the manual values they resolved to: a Manual value renders exactly like the Auto it came
    /// from, and the file then says which numbers were used (which Auto chose them is in `ym:Auto`
    /// and `ym:AppleAuto`, `docs/19` section 9). White balance is left as it is: a manual white
    /// balance at the Kelvin and tint shown does *not* render like Auto (the tint scales the green
    /// gain, and Kelvin is rounded to 50 K), so an Auto white balance stays Auto here and the
    /// numbers it resolved to are in the `auto` object of the CLI output. Nothing else is pinned:
    /// only the Input and Main can be Auto.
    #[must_use]
    pub fn pinned_settings(&self) -> DevelopSettings {
        let mut s = self.settings.clone();
        let input = &self.values.input;
        if s.input().exposure.is_auto() {
            s.input_mut().exposure = ExposureSetting::manual_ev(input.exposure_ev);
        }
        if s.input().sharpen.is_auto() {
            s.input_mut().sharpen = AutoOr::Manual(input.sharpen);
        }
        if s.input().noise_reduction.is_auto() {
            s.input_mut().noise_reduction = AutoOr::Manual(input.noise_reduction);
        }
        let v = &self.values.main.grade;
        let g = s.main_grade_mut();
        for (setting, value) in [
            (&mut g.contrast, v.contrast),
            (&mut g.saturation, v.saturation),
            (&mut g.highlights, v.highlights),
            (&mut g.shadows, v.shadows),
            (&mut g.black, v.black),
        ] {
            if setting.is_auto() {
                *setting = PercentSetting::manual_value(value);
            }
        }
        for (setting, point) in [
            (&mut g.color_balance.shadows, v.color_balance.shadows),
            (&mut g.color_balance.highlights, v.color_balance.highlights),
        ] {
            if setting.is_auto() {
                *setting = BalanceSetting::manual_point(point);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_the_default_graph() {
        let s = DevelopSettings::default();
        assert_eq!(s.output().look.as_str(), "builtin:neutral");
        assert_eq!(s.output().look_intensity, 100.0);
        assert!(s.input().exposure.is_auto() && s.input().white_balance.is_auto());
        assert!(s.input().sharpen.is_auto() && s.input().noise_reduction.is_auto());
        assert_eq!(s.input().light_source, LightSource::Auto);
        assert_eq!(s.main_grade().curve.luma, identity_curve());
        assert_eq!(
            s.input().rolloff,
            Rolloff {
                highlights: 0.6,
                low_light: 0.5
            }
        );
        assert!(s.input().decode.neural_demosaic && s.input().decode.chromatic_aberration);
        assert!(s.input().crop.is_none() && s.input().white_pick.is_none());
        assert_eq!(s.input().orientation, Orientation::Deg0);
        assert_eq!(s.output().output_sharpening, 100.0);
        assert_eq!(s.nodes.links, NodeGraph::default_links());
        assert_eq!(s.main().name, "Main");
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
        use SettingsField as F;
        assert_eq!(F::Decode.group(), Some(StageGroup::Raw));
        assert_eq!(F::NoiseReduction.group(), Some(StageGroup::Raw));
        assert_eq!(F::Sharpen.group(), Some(StageGroup::Raw));
        for f in [
            F::LightSource,
            F::WhiteBalance,
            F::WhitePick,
            F::Exposure,
            F::Rolloff,
        ] {
            assert_eq!(f.group(), Some(StageGroup::Color), "{f:?}");
        }
        for f in [F::Crop, F::Orientation] {
            assert_eq!(f.group(), Some(StageGroup::Geometry));
        }
        // Output sharpening is the last stage: it follows geometry and re-runs nothing before it.
        assert_eq!(F::OutputSharpening.group(), Some(StageGroup::Output));
        assert!(StageGroup::Geometry < StageGroup::Output);
        assert_eq!(F::Contrast.group(), Some(StageGroup::Grade));
        assert_eq!(F::Look.group(), Some(StageGroup::Grade));
        // A corrector's trim, bypass and masks are the nodes' group; its name and place re-run nothing.
        for f in [F::Trim, F::Bypass, F::Masks] {
            assert_eq!(f.group(), Some(StageGroup::Grade), "{f:?}");
        }
        assert_eq!(F::Name.group(), None);
        assert_eq!(F::Pos.group(), None);
        // The lens sits between the raw group and the colour group (`docs/21` section 3).
        assert_eq!(F::Lens.group(), Some(StageGroup::Lens));
        assert!(StageGroup::Raw < StageGroup::Lens && StageGroup::Lens < StageGroup::Color);
        assert!(StageGroup::Raw < StageGroup::Color && StageGroup::Color < StageGroup::Grade);
        // Every field has a name that is the serde name.
        for f in SettingsField::ALL {
            assert_eq!(
                serde_json::to_string(&f).unwrap(),
                format!("\"{}\"", f.name())
            );
        }
        assert_eq!(SettingsField::ALL.len(), 29);
    }
}
