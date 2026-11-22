//! The Colour Uniform node's settings (`docs/32-color-uniform.md` section 5): a target colour and three amounts that
//! pull the colours the node's masks let through toward it, as plain numbers in the units the panel shows.
//!
//! A Colour Uniform node is a numbered node of the graph, like a corrector: `picture` and `key` ports, masks, Bypass.
//! Its ids share the number space of the correctors. It has no range of its own: its masks are its whole selection
//! (section 2.3). Until CU-04 the render treats it as the identity; the core converts the units once, in
//! `ym_color::uniform` (section 3.2).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::NodePos;
use super::graph::{DEFAULT_UNIFORM_NAME, NodeGrade, NodeTrim};
use super::masks::MaskSettings;

/// Range of the target's Saturation: the chroma of the colour times 100 (a skin tone is about 5 to 12). A value
/// below the range's start is held at 2, so a target is never grey (`docs/32` section 3.7).
pub const TARGET_SATURATION_RANGE: (f64, f64) = (2.0, 50.0);
/// Range of the target's Lightness, in stops from middle grey.
pub const TARGET_LIGHTNESS_RANGE_EV: (f64, f64) = (-4.0, 8.0);
/// Range of each Compress amount, in percent.
pub const UNIFORM_AMOUNT_RANGE: (f64, f64) = (0.0, 100.0);

/// A target colour: where the node pulls the colours it acts on (`docs/32` section 5.1). The three numbers are in the
/// units the panel shows, and all three are required. It is read from a JSON object only (a list of three numbers is
/// refused, as serde would otherwise take it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct TargetValue {
    /// The hue in degrees. Any finite number; it is read modulo 360 into [0, 360) and never clamped or reported.
    pub hue_deg: f64,
    /// The chroma of the colour times 100, 2 to 50.
    pub saturation: f64,
    /// Stops from middle grey (code 0.2784), -4 to +8.
    pub lightness_ev: f64,
}

/// What [`TargetValue`] is read from: the object form, with every field required.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetRaw {
    hue_deg: f64,
    saturation: f64,
    lightness_ev: f64,
}

impl TryFrom<Value> for TargetValue {
    type Error = String;

    fn try_from(v: Value) -> Result<Self, String> {
        if !v.is_object() {
            return Err(
                "a target must be an object with hue_deg, saturation and lightness_ev".to_owned(),
            );
        }
        let r: TargetRaw = serde_json::from_value(v).map_err(|e| e.to_string())?;
        Ok(Self {
            hue_deg: r.hue_deg,
            saturation: r.saturation,
            lightness_ev: r.lightness_ev,
        })
    }
}

/// What [`UniformAmounts`] is read from: the object form, a part left out reading as 0.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
struct AmountsRaw {
    hue: f64,
    saturation: f64,
    lightness: f64,
}

impl TryFrom<Value> for UniformAmounts {
    type Error = String;

    fn try_from(v: Value) -> Result<Self, String> {
        if !v.is_object() {
            return Err("compress must be an object with hue, saturation and lightness".to_owned());
        }
        let r: AmountsRaw = serde_json::from_value(v).map_err(|e| e.to_string())?;
        Ok(Self {
            hue: r.hue,
            saturation: r.saturation,
            lightness: r.lightness,
        })
    }
}

/// `compress`: how far the colours the masks let through are pulled to the target on each axis, in percent.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct UniformAmounts {
    /// Hue, 0 to 100. 0 moves nothing.
    pub hue: f64,
    /// Saturation, 0 to 100.
    pub saturation: f64,
    /// Lightness, 0 to 100.
    pub lightness: f64,
}

impl UniformAmounts {
    /// Whether every amount is 0.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.hue <= 0.0 && self.saturation <= 0.0 && self.lightness <= 0.0
    }
}

/// A Colour Uniform node (`docs/32` section 5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ColorUniformNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// The colour the node pulls toward; `null` is none, and then the node does nothing.
    pub target: Option<TargetValue>,
    /// How far each axis is pulled.
    pub compress: UniformAmounts,
    /// The corrector's trims, run after the pull (`docs/32` section 12a). Written only when not all 0, so a node that does not
    /// use them is the one older versions wrote and read.
    #[serde(skip_serializing_if = "trim_is_default")]
    pub trim: NodeTrim,
    /// The corrector's grade, colour warp and curves, run after the trims. Written only when it differs from the neutral
    /// grade (Auto reads as a manual 0: this node is not Main).
    #[serde(skip_serializing_if = "grade_is_neutral")]
    pub grade: NodeGrade,
    /// The masks, as a corrector's: the node's whole selection.
    pub masks: MaskSettings,
}

fn trim_is_default(t: &NodeTrim) -> bool {
    *t == NodeTrim::default()
}

fn grade_is_neutral(g: &NodeGrade) -> bool {
    *g == NodeGrade::neutral()
}

impl Default for ColorUniformNode {
    /// No target, every amount 0, no masks: the identity.
    fn default() -> Self {
        Self {
            name: DEFAULT_UNIFORM_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            target: None,
            compress: UniformAmounts::default(),
            trim: NodeTrim::default(),
            grade: NodeGrade::neutral(),
            masks: MaskSettings::default(),
        }
    }
}

impl ColorUniformNode {
    /// Whether the pull does nothing: no target, or every amount 0 (`docs/32` section 2.5).
    #[must_use]
    pub fn pull_is_inert(&self) -> bool {
        self.target.is_none() || self.compress.is_zero()
    }

    /// Whether the node does nothing to the picture: bypassed, or no pull and a trim and grade at their defaults
    /// (`docs/32` sections 2.5 and 12a.1).
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.bypass
            || (self.pull_is_inert()
                && trim_is_default(&self.trim)
                && grade_is_neutral(&self.grade))
    }
}

/// Reads a hue in degrees into [0, 360): the Euclidean remainder, so 360 is 0 and -1 is 359. The caller has checked
/// that the number is finite.
#[must_use]
pub fn wrap_hue_deg(hue: f64) -> f64 {
    let h = hue.rem_euclid(360.0);
    // A tiny negative hue rounds up to 360 in the remainder.
    if h >= 360.0 { 0.0 } else { h + 0.0 }
}
