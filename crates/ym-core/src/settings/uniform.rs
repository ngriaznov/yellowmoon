//! The Colour Uniform node's settings (`docs/32-color-uniform.md` section 5): a target colour and three amounts that
//! pull the colours the node's masks let through toward it, as plain numbers in the units the panel shows.
//!
//! A Colour Uniform node is a numbered node of the graph, like a corrector: `picture` and `key` ports, masks, Bypass.
//! Its ids share the number space of the correctors. It has no range of its own: its masks are its whole selection
//! (section 2.3). Until CU-04 the render treats it as the identity; the core converts the units once, in
//! `ym_color::uniform` (section 3.2).

use serde::{Deserialize, Serialize};

use super::NodePos;
use super::graph::DEFAULT_UNIFORM_NAME;
use super::masks::MaskSettings;

/// Range of the target's Saturation: the chroma of the colour times 100 (a skin tone is about 5 to 12). A value
/// below the range's start is held at 2, so a target is never grey (`docs/32` section 3.7).
pub const TARGET_SATURATION_RANGE: (f64, f64) = (2.0, 50.0);
/// Range of the target's Lightness, in stops from middle grey.
pub const TARGET_LIGHTNESS_RANGE_EV: (f64, f64) = (-4.0, 8.0);
/// Range of each Compress amount, in percent.
pub const UNIFORM_AMOUNT_RANGE: (f64, f64) = (0.0, 100.0);

/// A target colour: where the node pulls the colours it acts on (`docs/32` section 5.1). The three numbers are in the
/// units the panel shows.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetValue {
    /// The hue in degrees. Any finite number; it is read modulo 360 into [0, 360) and never clamped or reported.
    pub hue_deg: f64,
    /// The chroma of the colour times 100, 2 to 50.
    pub saturation: f64,
    /// Stops from middle grey (code 0.2784), -4 to +8.
    pub lightness_ev: f64,
}

/// `compress`: how far the colours the masks let through are pulled to the target on each axis, in percent.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
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
    /// The masks, as a corrector's: the node's whole selection.
    pub masks: MaskSettings,
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
            masks: MaskSettings::default(),
        }
    }
}

impl ColorUniformNode {
    /// Whether the node does nothing to the picture: bypassed, no target, or every amount 0 (`docs/32` section 2.5).
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.bypass || self.target.is_none() || self.compress.is_zero()
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
