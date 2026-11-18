//! The Blur node's settings (`docs/33-blur.md` section 5): one node with a Type (Soft, Lens, Motion or Zoom), a group of
//! values for each type, and the switch "Respect mask edges", as plain numbers in the units the panel shows.
//!
//! A Blur node is a numbered node of the graph, like a corrector: `picture` and `key` ports, masks, Bypass. Its ids share the
//! number space of the correctors. Every type's group is always present in a stored node, so switching the type keeps the
//! values (`docs/33` section 2.1). Until BL-04 the render treats the node as the identity; the core converts the units to pixels
//! once, in `ym_color::blur` (`docs/33` section 2.2).

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::NodePos;
use super::graph::DEFAULT_BLUR_NAME;
use super::masks::MaskSettings;

/// Range of every amount (Radius, Length, Amount, Highlights): 0 to 100. The meaning of 100 is in `ym_color::blur`
/// (4 % of the long edge for Radius, 8 % for Length, a tenth of the distance for Amount; `docs/33` section 2.2).
// TODO(BL-03): the meaning of 100 (`RADIUS_MAX`, `MOTION_MAX`, `ZOOM_MAX`) is `ym_color::blur`'s; BL-04 wires it up.
pub const BLUR_AMOUNT_RANGE: (f64, f64) = (0.0, 100.0);
/// The fewest blades a polygon has: 3. A `blades` of 1 or 2 is held at this (and reported); 0 is a round aperture.
pub const BLUR_BLADES_MIN: u32 = 3;
/// The most blades a polygon has: 12. The panel offers 5 to 9; a file may hold what a panel does not offer.
pub const BLUR_BLADES_MAX: u32 = 12;
/// Range of each coordinate of the Zoom centre, a point of the photo frame.
pub const BLUR_CENTER_RANGE: (f64, f64) = (0.0, 1.0);
/// The default of every amount: a new node blurs a little at once (`docs/33` section 2.5).
pub const BLUR_DEFAULT_AMOUNT: f64 = 20.0;

/// The kind of blur (`docs/33` section 2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlurType {
    /// A Gaussian.
    #[default]
    Soft,
    /// A disc or a regular polygon, the shape of a lens's aperture.
    Lens,
    /// A line through each pixel.
    Motion,
    /// A radial line about a centre.
    Zoom,
}

impl BlurType {
    /// Every type, in the order of the Type control.
    pub const ALL: [BlurType; 4] = [Self::Soft, Self::Lens, Self::Motion, Self::Zoom];

    /// The JSON word of the type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Soft => "soft",
            Self::Lens => "lens",
            Self::Motion => "motion",
            Self::Zoom => "zoom",
        }
    }

    /// Reads the JSON word of a type.
    ///
    /// # Errors
    ///
    /// The sentence of an unknown type ("A Blur node has no type called “x”.").
    pub fn parse(word: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|t| t.name() == word)
            .ok_or_else(|| format!("A Blur node has no type called “{word}”."))
    }
}

impl<'de> Deserialize<'de> for BlurType {
    /// A word; any other string is refused with the sentence of [`BlurType::parse`], and anything but a string with a
    /// plain one (`"spin"` is not accepted until BL-17).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(s) => Self::parse(&s).map_err(de::Error::custom),
            other => Err(de::Error::custom(format!(
                "a Blur node's type must be a word, not {other}"
            ))),
        }
    }
}

/// Soft's values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlurSoft {
    /// Radius, 0 to 100: the blur radius as a share of the long edge, 100 being 4 %.
    pub radius: f64,
}

impl Default for BlurSoft {
    /// Radius 20.
    fn default() -> Self {
        Self {
            radius: BLUR_DEFAULT_AMOUNT,
        }
    }
}

/// Lens' values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlurLens {
    /// Radius, 0 to 100: the aperture's radius as a share of the long edge, 100 being 4 %.
    pub radius: f64,
    /// The blades of the aperture: 0 is a disc, 3 to 12 a regular polygon (the panel offers 5 to 9). 1 and 2 are held at 3.
    pub blades: u32,
    /// Turns the polygon, in degrees. Any finite number, read modulo 360 and never reported.
    pub rotation_deg: f64,
    /// Highlights, 0 to 100: how much bright lights count for in the blur. 0 is the plain aperture.
    pub highlights: f64,
}

impl Default for BlurLens {
    /// Radius 20, a round aperture, no rotation, Highlights 0.
    fn default() -> Self {
        Self {
            radius: BLUR_DEFAULT_AMOUNT,
            blades: 0,
            rotation_deg: 0.0,
            highlights: 0.0,
        }
    }
}

/// Motion's values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlurMotion {
    /// Length, 0 to 100: the length of the line as a share of the long edge, 100 being 8 %.
    pub length: f64,
    /// The direction of the line in degrees, 0 along the picture's width. Any finite number, read modulo 180 and never
    /// reported.
    pub angle_deg: f64,
}

impl Default for BlurMotion {
    /// Length 20, angle 0.
    fn default() -> Self {
        Self {
            length: BLUR_DEFAULT_AMOUNT,
            angle_deg: 0.0,
        }
    }
}

/// Zoom's values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlurZoom {
    /// Amount, 0 to 100: a pixel at distance `d` from the centre is spread over a line of `Amount / 100 * 0.10 * d`.
    pub amount: f64,
    /// The centre, `[x, y]` in the photo frame (0 to 1 each way), so it follows crop and rotation.
    pub center: [f64; 2],
}

impl Default for BlurZoom {
    /// Amount 20, the centre of the picture.
    fn default() -> Self {
        Self {
            amount: BLUR_DEFAULT_AMOUNT,
            center: [0.5, 0.5],
        }
    }
}

/// A Blur node (`docs/33` section 5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlurNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// The chosen type.
    #[serde(rename = "type")]
    pub blur_type: BlurType,
    /// Soft's values.
    pub soft: BlurSoft,
    /// Lens' values.
    pub lens: BlurLens,
    /// Motion's values.
    pub motion: BlurMotion,
    /// Zoom's values.
    pub zoom: BlurZoom,
    /// Whether colours from outside the mask are kept out of the blur (`docs/33` section 2.3).
    pub respect_mask_edges: bool,
    /// The masks, as a corrector's: the node's whole selection and, with Respect mask edges, its weights.
    pub masks: MaskSettings,
}

impl Default for BlurNode {
    /// Soft at Radius 20, no masks, Respect mask edges on, not bypassed.
    fn default() -> Self {
        Self {
            name: DEFAULT_BLUR_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            blur_type: BlurType::Soft,
            soft: BlurSoft::default(),
            lens: BlurLens::default(),
            motion: BlurMotion::default(),
            zoom: BlurZoom::default(),
            respect_mask_edges: true,
            masks: MaskSettings::default(),
        }
    }
}

impl BlurNode {
    /// The amount of the chosen type (Radius, Radius, Length or Amount): a node is judged by its own type's amount only.
    #[must_use]
    pub fn amount(&self) -> f64 {
        match self.blur_type {
            BlurType::Soft => self.soft.radius,
            BlurType::Lens => self.lens.radius,
            BlurType::Motion => self.motion.length,
            BlurType::Zoom => self.zoom.amount,
        }
    }

    /// Whether the node does nothing to the picture: bypassed, or the chosen type's amount is 0 (`docs/33` section 2.5).
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.bypass || self.amount() <= 0.0
    }
}

/// Reads an angle in degrees into [0, `period`): the Euclidean remainder. The caller has checked that it is finite.
#[must_use]
pub fn wrap_deg(angle: f64, period: f64) -> f64 {
    let a = angle.rem_euclid(period);
    // A tiny negative angle rounds up to the period in the remainder.
    if a >= period { 0.0 } else { a + 0.0 }
}
