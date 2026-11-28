//! The Reference Grade node's settings (`docs/34-reference-grade.md` section 3): a corrector's trims and grade, two hue
//! curves, seven element grades, a Strength and the Skin protection switch, as plain numbers in the units the panel shows.
//!
//! A Reference Grade node is a numbered node of the graph, like a corrector: `picture` and `key` ports, masks, Bypass. Its ids
//! share the number space of the correctors. A fit writes `grade.curve`, `grade.saturation`, both hue curves and every element
//! (RG-06); until then the person edits them. The node never holds the reference picture or anything computed from it
//! (`docs/34` section 3.7). Strength scales what the fit writes at render time, in one function, [`ReferenceGradeNode::effective`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::graph::{GraphError, NodeGrade, NodeTrim};
use super::masks::MaskSettings;
use super::{AutoOr, Curves, NodePos, PercentValue, SettingsError};

/// The name a new Reference Grade node has (five letters, so the card does not cut it).
pub const DEFAULT_REFERENCE_NAME: &str = "Match";
/// The most points a hue curve has.
pub const MAX_HUE_POINTS: usize = ym_color::hue::MAX_HUE_POINTS;
/// Range of a `hue_vs_hue` point's value, in degrees (the parametric look's limits).
pub const HUE_VS_HUE_RANGE_DEG: (f64, f64) = (-180.0, 180.0);
/// Range of a `hue_vs_sat` point's value, a chroma multiplier (the parametric look's limits).
pub const HUE_VS_SAT_RANGE: (f64, f64) = (0.0, 8.0);
/// Range of an element's exposure offset, in stops.
pub const ELEMENT_EXPOSURE_RANGE_EV: (f64, f64) = (-1.0, 1.0);
/// Range of Match brightness, in percent (`docs/34` section 5.8).
pub const MATCH_BRIGHTNESS_RANGE: (f64, f64) = (0.0, 100.0);
/// Range of an element's chroma ratio.
pub const ELEMENT_CHROMA_RANGE: (f64, f64) = (0.5, 1.6);
/// The members of `elements`, in the order of `docs/34` section 2.3: the six elements, then the subject.
pub const ELEMENT_NAMES: [&str; 7] = [
    "sky",
    "vegetation",
    "water",
    "ground",
    "built",
    "skin",
    "subject",
];
/// The largest hue turn of each of the six elements (the fit's own caps, `docs/34` section 4a.1), in degrees.
pub const ELEMENT_HUE_CAPS_DEG: [f64; 6] = [40.0, 35.0, 35.0, 30.0, 25.0, 15.0];

/// A point of a hue curve (`docs/34` section 4.1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuePoint {
    /// The hue in degrees on the LogC4 chroma-plane axis (0 red, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta).
    /// Any finite number; it is read modulo 360 into [0, 360) and never reported.
    pub hue_deg: f64,
    /// `hue_vs_hue`: the degrees the hue turns, -180 to 180. `hue_vs_sat`: the chroma multiplier, 0 to 8.
    pub value: f64,
}

/// `hue_curves`: two periodic curves over the hue circle. An empty list is the identity; one point is a constant.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HueCurves {
    /// How far the hue turns at each hue, in degrees.
    pub hue_vs_hue: Vec<HuePoint>,
    /// What the chroma is multiplied by at each hue.
    pub hue_vs_sat: Vec<HuePoint>,
}

impl HueCurves {
    /// Whether both curves are empty.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.hue_vs_hue.is_empty() && self.hue_vs_sat.is_empty()
    }
}

/// The points of a curve as `(hue_deg, value)` pairs, in the order they are stored (sorted by hue once sanitised).
#[must_use]
pub fn hue_pairs(points: &[HuePoint]) -> Vec<(f64, f64)> {
    points.iter().map(|p| (p.hue_deg, p.value)).collect()
}

/// One element's grade (`docs/34` section 4a.1): an exposure offset, a hue turn and a chroma ratio.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ElementValues {
    /// The exposure offset in stops, -1 to 1.
    pub exposure_ev: f64,
    /// The hue turn in degrees, within the element's cap.
    pub hue_deg: f64,
    /// The chroma ratio, 0.5 to 1.6.
    pub chroma: f64,
}

impl Default for ElementValues {
    /// The identity: 0, 0, 1.
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            hue_deg: 0.0,
            chroma: 1.0,
        }
    }
}

/// The subject's grade: an exposure offset alone.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SubjectValues {
    /// The exposure offset in stops, -1 to 1.
    pub exposure_ev: f64,
}

/// `elements`: the six element grades and the subject's (`docs/34` section 4a.1).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Elements {
    /// The sky.
    pub sky: ElementValues,
    /// Trees, grass, flowers, plants.
    pub vegetation: ElementValues,
    /// Sea, river, other water.
    pub water: ElementValues,
    /// Sand, earth, road, rock, mountain, snow.
    pub ground: ElementValues,
    /// Buildings, walls, roofs and the surfaces of interiors.
    pub built: ElementValues,
    /// The skin of people.
    pub skin: ElementValues,
    /// The person, or the main object.
    pub subject: SubjectValues,
}

impl Elements {
    /// The six elements in order, then the subject as an element with only its exposure (hue 0, chroma 1).
    #[must_use]
    pub fn all(&self) -> [ElementValues; 7] {
        [
            self.sky,
            self.vegetation,
            self.water,
            self.ground,
            self.built,
            self.skin,
            ElementValues {
                exposure_ev: self.subject.exposure_ev,
                ..ElementValues::default()
            },
        ]
    }

    /// The six elements, to change, in order.
    pub fn six_mut(&mut self) -> [&mut ElementValues; 6] {
        [
            &mut self.sky,
            &mut self.vegetation,
            &mut self.water,
            &mut self.ground,
            &mut self.built,
            &mut self.skin,
        ]
    }

    /// Whether every element is at the identity.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// A Reference Grade node (`docs/34` section 3.1). It writes every field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReferenceGradeNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// The person's trims (the fit does not write them).
    pub trim: NodeTrim,
    /// The corrector's grade, colour warp and curves. The fit writes `curve` and `saturation`; the rest is the person's.
    /// Auto reads as a manual 0, silently.
    pub grade: NodeGrade,
    /// The two hue curves.
    pub hue_curves: HueCurves,
    /// The element grades.
    pub elements: Elements,
    /// 0 to 100, in percent: how much of the fit is shown (`docs/34` section 3.2).
    pub strength: f64,
    /// Whether the next fit uses the walls that keep skin from drifting; it changes no pixel by itself.
    pub skin_protection: bool,
    /// 0 to 100, in percent: how far the next fit moves the photo's brightness toward the reference's (`docs/34` section 5.8);
    /// it changes no pixel by itself. A document without it reads 0.
    pub match_brightness: f64,
    /// The masks, as a corrector's: they limit where the whole node acts.
    pub masks: MaskSettings,
}

impl Default for ReferenceGradeNode {
    /// Every field at the identity, Strength 100, Skin protection on, Match brightness 0.
    fn default() -> Self {
        Self {
            name: DEFAULT_REFERENCE_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            trim: NodeTrim::default(),
            grade: NodeGrade::neutral(),
            hue_curves: HueCurves::default(),
            elements: Elements::default(),
            strength: 100.0,
            skin_protection: true,
            match_brightness: 0.0,
            masks: MaskSettings::default(),
        }
    }
}

/// `x + k (y - x)`, held to the unit square, with the two ends exact: Strength 100 is the stored value and 0 the identity.
fn mix_knot(x: f64, y: f64, k: f64) -> f64 {
    if k >= 1.0 {
        y
    } else if k <= 0.0 {
        x
    } else {
        (x + k * (y - x)).clamp(0.0, 1.0)
    }
}

/// A Reference Grade node's fields as the render runs them: Strength applied, Auto read as a manual 0 and nothing else (no
/// masks, no name). Made by [`ReferenceGradeNode::effective`].
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveReference {
    /// The person's trims, as stored (Strength does not scale them).
    pub trim: NodeTrim,
    /// The grade with the fitted curves and the saturation scaled by Strength.
    pub grade: NodeGrade,
    /// The hue curves with their values scaled by Strength.
    pub hue_curves: HueCurves,
    /// The element grades scaled by Strength.
    pub elements: Elements,
}

impl EffectiveReference {
    /// Whether the hue curves and the element grades do nothing (every value neutral): the device needs neither.
    #[must_use]
    pub fn hue_and_elements_are_identity(&self) -> bool {
        self.hue_curves.hue_vs_hue.iter().all(|p| p.value == 0.0)
            && self.hue_curves.hue_vs_sat.iter().all(|p| p.value == 1.0)
            && self.elements == Elements::default()
    }
}

impl ReferenceGradeNode {
    /// The node's fields as the render runs them, with Strength applied (`docs/34` section 3.2): for each fitted field and
    /// `k = strength / 100`, a curve's knot `(x, y)` becomes `(x, x + k (y - x))`, the manual saturation `s` becomes `k s`, a
    /// `hue_vs_hue` value `k d`, a `hue_vs_sat` value `1 + k (s - 1)`, an element's `exposure_ev` and `hue_deg` `k e` and
    /// `k h` and its chroma `c^k`. Auto in the grade reads as a manual 0. It is what the CPU reference, the cache key and the
    /// device's host packing all read, so the kernel never sees Strength. Strength 100 gives the stored values exactly and
    /// Strength 0 the identity of every fitted field, bit for bit.
    #[must_use]
    pub fn effective(&self) -> EffectiveReference {
        let k = (self.strength / 100.0).clamp(0.0, 1.0);
        let mut n = EffectiveReference {
            trim: self.trim,
            grade: self.grade.clone(),
            hue_curves: self.hue_curves.clone(),
            elements: self.elements,
        };
        n.grade.make_manual();
        if k >= 1.0 {
            return n;
        }
        for curve in [
            &mut n.grade.curve.luma,
            &mut n.grade.curve.red,
            &mut n.grade.curve.green,
            &mut n.grade.curve.blue,
        ] {
            for p in curve.iter_mut() {
                p[1] = mix_knot(p[0], p[1], k);
            }
        }
        if let AutoOr::Manual(PercentValue { value }) = &mut n.grade.saturation {
            *value = if k <= 0.0 { 0.0 } else { k * *value };
        }
        for p in &mut n.hue_curves.hue_vs_hue {
            p.value = if k <= 0.0 { 0.0 } else { k * p.value };
        }
        for p in &mut n.hue_curves.hue_vs_sat {
            p.value = if k <= 0.0 {
                1.0
            } else {
                1.0 + k * (p.value - 1.0)
            };
        }
        for e in n.elements.six_mut() {
            if k <= 0.0 {
                *e = ElementValues::default();
            } else {
                e.exposure_ev *= k;
                e.hue_deg *= k;
                e.chroma = e.chroma.powf(k);
            }
        }
        n.elements.subject.exposure_ev = if k <= 0.0 {
            0.0
        } else {
            k * n.elements.subject.exposure_ev
        };
        n
    }

    /// Whether the node does nothing to the picture: bypassed, or its effective trims, grade, hue curves and elements are at
    /// their defaults (`docs/34` section 7.1).
    #[must_use]
    pub fn is_inert(&self) -> bool {
        if self.bypass {
            return true;
        }
        let mut e = self.effective();
        // A curve whose knots all sit on `y = x` is the identity whatever its number of knots (a fit at Strength 0 leaves them so).
        for (curve, neutral) in [
            (&mut e.grade.curve.luma, Curves::default().luma),
            (&mut e.grade.curve.red, Curves::default().red),
            (&mut e.grade.curve.green, Curves::default().green),
            (&mut e.grade.curve.blue, Curves::default().blue),
        ] {
            if curve.iter().all(|p| p[0] == p[1]) {
                *curve = neutral;
            }
        }
        e.trim == NodeTrim::default()
            && e.grade == NodeGrade::neutral()
            && e.hue_and_elements_are_identity()
    }
}

// ---------------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------------

/// Checks the shape of a Reference Grade node's `hue_curves` and `elements` in a document before serde reads them, so that a
/// list is not taken for a group's fields and a member the node does not have gets its own sentence (`docs/34` section 3.4).
///
/// # Errors
///
/// [`SettingsError::Parse`] for a group, a member or a point that is not an object, and [`GraphError::UnknownElement`] for a
/// member of `elements` that is not one of [`ELEMENT_NAMES`].
pub fn check_shape(
    hue_curves: Option<&Value>,
    elements: Option<&Value>,
) -> Result<(), SettingsError> {
    if let Some(v) = hue_curves {
        let Some(map) = v.as_object() else {
            return Err(SettingsError::Parse(
                "a Reference Grade node's `hue_curves` must be an object".to_owned(),
            ));
        };
        for (name, list) in map {
            if let Some(items) = list.as_array() {
                if items.iter().any(|p| !p.is_object()) {
                    return Err(SettingsError::Parse(format!(
                        "every point of `hue_curves.{name}` must be an object with hue_deg and value"
                    )));
                }
            } else if matches!(name.as_str(), "hue_vs_hue" | "hue_vs_sat") {
                return Err(SettingsError::Parse(format!(
                    "`hue_curves.{name}` must be a list of points"
                )));
            }
        }
    }
    if let Some(v) = elements {
        let Some(map) = v.as_object() else {
            return Err(SettingsError::Parse(
                "a Reference Grade node's `elements` must be an object".to_owned(),
            ));
        };
        check_members(map.iter())?;
    }
    Ok(())
}

/// The shape rule of a patch's `elements` (the member names), as [`check_shape`] has it.
///
/// # Errors
///
/// As [`check_shape`].
pub fn check_patch_shape(elements: &Value) -> Result<(), SettingsError> {
    match elements.as_object() {
        Some(map) => check_members(map.iter()),
        None => Ok(()),
    }
}

fn check_members<'a>(
    members: impl Iterator<Item = (&'a String, &'a Value)>,
) -> Result<(), SettingsError> {
    for (name, value) in members {
        if !ELEMENT_NAMES.contains(&name.as_str()) {
            return Err(GraphError::UnknownElement(name.clone()).into());
        }
        if !value.is_object() {
            return Err(SettingsError::Parse(format!(
                "`elements.{name}` must be an object"
            )));
        }
    }
    Ok(())
}
