//! The masks of a corrector node (`docs/23-nodes.md` section 3.3): rows of ten kinds that add up
//! into the node's mask, their ranges, their checking and the row patches (`masks.edit`,
//! `strokes_append`, `strokes_keep`).
//!
//! N-02 reads, checks, patches, copies and writes masks; it renders none of them (N-03 on).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::{SettingsError, clamp_to};
use crate::vision::VisionSupport;

/// The most mask rows one node may hold.
pub const MAX_MASKS_PER_NODE: usize = 64;
/// The most points a Brush mask may hold in all its strokes.
pub const MAX_BRUSH_POINTS: usize = 100_000;
/// The most points an Object mask may hold.
pub const MAX_OBJECT_POINTS: usize = 256;

/// Range of every opacity, feather, refine, level, softness, hardness and low/high value.
pub const UNIT_RANGE: (f64, f64) = (0.0, 1.0);
/// Range of Grow.
pub const GROW_RANGE: (f64, f64) = (-1.0, 1.0);
/// Range of a Radial mask's radii (fractions of the photo's long edge).
pub const RADII_RANGE: (f64, f64) = (0.002, 1.0);
/// Range of a Linear or Radial mask's angle (degrees).
pub const MASK_ANGLE_RANGE_DEG: (f64, f64) = (-180.0, 180.0);
/// Range of a brush stroke's size (fraction of the long edge).
pub const STROKE_SIZE_RANGE: (f64, f64) = (0.001, 0.25);
/// Range of a Colour range mask's width.
pub const COLOR_RANGE_WIDTH_RANGE: (f64, f64) = (0.005, 0.3);
/// The longest centre of a Colour range mask, in the chroma plane of C9.
pub const COLOR_RANGE_CENTER_LIMIT: f64 = 0.5;

/// How a mask row combines with the rows before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskMode {
    /// The row adds to the mask.
    #[default]
    Add,
    /// The row takes away from the mask.
    Subtract,
}

/// The kinds of mask row, as their JSON names.
pub const MASK_KINDS: [&str; 10] = [
    "object",
    "subject",
    "people",
    "background",
    "depth",
    "linear",
    "radial",
    "brush",
    "color_range",
    "brightness_range",
];

/// A kind of mask row (`docs/23` section 3.3), the typed twin of [`MASK_KINDS`]: what the Add mask
/// menu offers and what [`mask_kinds`] lists. Written as the row's `kind` is (`color_range`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    /// SAM 2.1 from clicks or a box.
    Object,
    /// Apple Vision's foreground.
    Subject,
    /// Apple Vision's people.
    People,
    /// The blur map's out-of-focus area.
    Background,
    /// A band of near or far from the depth map.
    Depth,
    /// A gradient across the photo.
    Linear,
    /// An oval with a soft edge.
    Radial,
    /// Strokes painted on the photo.
    Brush,
    /// A chroma centre, width and softness.
    ColorRange,
    /// A low, a high and a softness on the tones.
    BrightnessRange,
}

impl MaskKind {
    /// Every kind in the order of the Add mask menu (Find, Shapes, Keys).
    pub const ALL: [MaskKind; 10] = [
        MaskKind::Object,
        MaskKind::Subject,
        MaskKind::People,
        MaskKind::Background,
        MaskKind::Depth,
        MaskKind::Linear,
        MaskKind::Radial,
        MaskKind::Brush,
        MaskKind::ColorRange,
        MaskKind::BrightnessRange,
    ];

    /// The kind's JSON name: the row's `kind`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            MaskKind::Object => "object",
            MaskKind::Subject => "subject",
            MaskKind::People => "people",
            MaskKind::Background => "background",
            MaskKind::Depth => "depth",
            MaskKind::Linear => "linear",
            MaskKind::Radial => "radial",
            MaskKind::Brush => "brush",
            MaskKind::ColorRange => "color_range",
            MaskKind::BrightnessRange => "brightness_range",
        }
    }

    /// The kind of a row's shape.
    #[must_use]
    pub fn of(shape: &MaskShape) -> MaskKind {
        match shape {
            MaskShape::Object(_) => MaskKind::Object,
            MaskShape::Subject(_) => MaskKind::Subject,
            MaskShape::People(_) => MaskKind::People,
            MaskShape::Background(_) => MaskKind::Background,
            MaskShape::Depth(_) => MaskKind::Depth,
            MaskShape::Linear(_) => MaskKind::Linear,
            MaskShape::Radial(_) => MaskKind::Radial,
            MaskShape::Brush(_) => MaskKind::Brush,
            MaskShape::ColorRange(_) => MaskKind::ColorRange,
            MaskShape::BrightnessRange(_) => MaskKind::BrightnessRange,
        }
    }
}

/// What a machine can make now (`docs/23` section 9, `mask_kinds`): whether the SAM weights are there and load
/// (Object), what the Vision probe said (Subject and People, `docs/23` section 3.10) and whether the depth weights are
/// there and load (Depth, section 3.10a). The lists, the reasons and the sentences below are all functions of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSupport {
    /// The SAM weights are there and have not been found damaged (`ObjectModel::available`).
    pub object: bool,
    /// What each Vision request said when it was tried (`VisionMasks::support`).
    pub vision: VisionSupport,
    /// The depth weights are there and have not been found damaged (`DepthModel::available`).
    pub depth: bool,
}

impl MaskSupport {
    /// A machine with these parts, and no depth model ([`MaskSupport::with_depth`] adds it).
    #[must_use]
    pub const fn new(object: bool, vision: VisionSupport) -> Self {
        Self {
            object,
            vision,
            depth: false,
        }
    }

    /// A machine that has nothing beyond Yellowmoon's own code: no Object model, and no Vision code (`not_a_mac` off
    /// macOS, `not_supported` on a Mac). What [`mask_kinds`] and [`mask_unavailable`] describe.
    #[must_use]
    pub const fn bare() -> Self {
        Self::new(false, VisionSupport::none_here())
    }

    /// This machine with the SAM weights said to be there or not.
    #[must_use]
    pub const fn with_object(mut self, object: bool) -> Self {
        self.object = object;
        self
    }

    /// This machine with the depth weights said to be there or not.
    #[must_use]
    pub const fn with_depth(mut self, depth: bool) -> Self {
        self.depth = depth;
        self
    }

    /// This machine with this Vision answer.
    #[must_use]
    pub const fn with_vision(mut self, vision: VisionSupport) -> Self {
        self.vision = vision;
        self
    }

    /// Whether it can make a kind of mask.
    #[must_use]
    pub fn can_make(self, kind: MaskKind) -> bool {
        match kind {
            MaskKind::Object => self.object,
            MaskKind::Subject => self.vision.subject.is_supported(),
            MaskKind::People => self.vision.people.is_supported(),
            MaskKind::Depth => self.depth,
            _ => true,
        }
    }
}

/// The kinds of mask a machine can make with nothing but Yellowmoon's code, in the order of the Add mask menu: what
/// [`mask_kinds_with`] gives for [`MaskSupport::bare`]. The shapes, Brush and the ranges that are built always are
/// here, and so is Background (N-15: the blur map is made on the CPU from the photo itself, with the learned model or
/// without it). **Depth is not**: it needs the depth weights (D-04). **Object is not**: it needs the SAM weights (N-18). **Subject and People are not**: they need the
/// Vision probe (N-19). What this machine can make is [`mask_kinds_with`] of its [`MaskSupport`]
/// (`Core::mask_kinds`, `Renderer::mask_kinds`).
#[must_use]
pub fn mask_kinds() -> Vec<MaskKind> {
    mask_kinds_with(MaskSupport::bare())
}

/// The kinds of mask this machine can make now (`docs/23` section 9, `mask_kinds`), in the order of the Add mask
/// menu: what the menu lists, and what a photo may use without being refused. Object is the first where the SAM
/// weights are there, then Subject and People where the Vision probe said yes, then Background, Depth where the
/// depth weights are there, and the shapes and ranges.
#[must_use]
pub fn mask_kinds_with(support: MaskSupport) -> Vec<MaskKind> {
    MaskKind::ALL
        .into_iter()
        .filter(|&k| support.can_make(k))
        .collect()
}

/// Why this machine cannot make a kind of mask, by its wire name in `ym doctor --json`
/// (`masks.unavailable`, `docs/10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskUnavailable {
    /// The SAM weights are missing or damaged (Object).
    WeightsMissing,
    /// The depth weights are missing or damaged (Depth). Its wire name is `weights_missing`, like the SAM weights'; it is a
    /// reason of its own so that `ym doctor` names the two files apart ("Object (the SAM weights are missing), Depth (the
    /// depth weights are missing)").
    DepthWeightsMissing,
    /// This is not a Mac (Subject and People).
    NotAMac,
    /// This Mac's macOS is older than the Vision requests need (Subject and People).
    MacosTooOld,
    /// This computer cannot make the kind: a Mac whose Vision request failed its probe (a virtual Mac), or a build of
    /// Yellowmoon that has no Vision code (an older SDK, or the code switched off).
    NotSupported,
}

impl MaskUnavailable {
    /// The reason's wire name: `weights_missing`, `not_a_mac`, `macos_too_old` or `not_supported`.
    #[must_use]
    pub fn wire(self) -> &'static str {
        match self {
            Self::WeightsMissing | Self::DepthWeightsMissing => "weights_missing",
            Self::NotAMac => "not_a_mac",
            Self::MacosTooOld => "macos_too_old",
            Self::NotSupported => "not_supported",
        }
    }

    /// The reason in the parentheses of `ym doctor`'s Masks line, for a group of kinds (`plural`) or for one kind
    /// that has the reason to itself.
    #[must_use]
    pub fn doctor_text(self, plural: bool) -> &'static str {
        match (self, plural) {
            (Self::WeightsMissing, _) => "the SAM weights are missing",
            (Self::DepthWeightsMissing, _) => "the depth weights are missing",
            (Self::NotAMac, true) => "this is not a Mac that supports them",
            (Self::NotAMac, false) => "this is not a Mac that supports it",
            (Self::MacosTooOld, true) => "this Mac's macOS is too old for them",
            (Self::MacosTooOld, false) => "this Mac's macOS is too old for it",
            (Self::NotSupported, true) => "this computer can't make them",
            (Self::NotSupported, false) => "this computer can't make it",
        }
    }
}

impl MaskKind {
    /// The kind's name in the interface and in `ym doctor` ("Colour range").
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            MaskKind::Object => "Object",
            MaskKind::Subject => "Subject",
            MaskKind::People => "People",
            MaskKind::Background => "Background",
            MaskKind::Depth => "Depth",
            MaskKind::Linear => "Linear",
            MaskKind::Radial => "Radial",
            MaskKind::Brush => "Brush",
            MaskKind::ColorRange => "Colour range",
            MaskKind::BrightnessRange => "Brightness range",
        }
    }
}

/// The kinds of mask a machine with [`MaskSupport::bare`] cannot make, each with its reason:
/// [`mask_unavailable_with`] of it.
#[must_use]
pub fn mask_unavailable() -> Vec<(MaskKind, MaskUnavailable)> {
    mask_unavailable_with(MaskSupport::bare())
}

/// The kinds of mask this machine cannot make, each with its reason, in the order of the Add mask menu: the
/// complement of [`mask_kinds_with`] in [`MaskKind::ALL`]. `ym doctor` prints it and a photo that needs one of them
/// fails with `error.kind` `mask` ([`unmakeable_mask_sentence_with`]). Object without its model is `weights_missing`;
/// Subject and People are what the Vision probe said (`not_a_mac`, `macos_too_old` or `not_supported`).
#[must_use]
pub fn mask_unavailable_with(support: MaskSupport) -> Vec<(MaskKind, MaskUnavailable)> {
    MaskKind::ALL
        .into_iter()
        .filter(|&k| !support.can_make(k))
        .map(|k| {
            let why = match k {
                MaskKind::Object => MaskUnavailable::WeightsMissing,
                MaskKind::Subject => support
                    .vision
                    .subject
                    .reason()
                    .unwrap_or(MaskUnavailable::NotSupported),
                MaskKind::People => support
                    .vision
                    .people
                    .reason()
                    .unwrap_or(MaskUnavailable::NotSupported),
                MaskKind::Depth => MaskUnavailable::DepthWeightsMissing,
                _ => MaskUnavailable::NotSupported,
            };
            (k, why)
        })
        .collect()
}

/// The sentence for an Object row on a copy of Yellowmoon that has no Object model (`docs/23` section 9, the `mask`
/// error): what went wrong and what to do.
pub const NO_OBJECT_MODEL_SENTENCE: &str =
    "This copy of Yellowmoon has no Object model. Reinstall the app.";

/// The sentence for a Depth row on a copy of Yellowmoon that has no depth model (`docs/23` section 3.10a, the `mask`
/// error): what went wrong and what to do.
pub const NO_DEPTH_MODEL_SENTENCE: &str =
    "This copy of Yellowmoon has no Depth model. Reinstall the app.";

/// The sentence a Depth row shows in the Mask tab on a copy of Yellowmoon that has no depth model (`docs/23` section
/// 3.10a and 8): "Depth masks need the depth model. Reinstall the app."
pub const DEPTH_ROW_NO_MODEL_SENTENCE: &str =
    "Depth masks need the depth model. Reinstall the app.";

/// The sentence a Subject or People row shows in the Mask tab on a computer that cannot make it (`docs/23` section
/// 3.10 and 7.6): "Subject masks need a Mac that supports them". `None` for the other kinds.
#[must_use]
pub fn unsupported_row_sentence(kind: MaskKind) -> Option<String> {
    match kind {
        MaskKind::Subject | MaskKind::People => Some(format!(
            "{} masks need a Mac that supports them",
            kind.label()
        )),
        _ => None,
    }
}

/// [`unmakeable_mask_sentence_with`] for [`MaskSupport::bare`].
#[must_use]
pub fn unmakeable_mask_sentence(node: u32, kind: MaskKind) -> Option<String> {
    unmakeable_mask_sentence_with(MaskSupport::bare(), node, kind)
}

/// The plain sentence for a node whose mask of this kind this computer cannot make (`error.kind` `mask`,
/// `docs/23` section 3.10 and `docs/10`): what went wrong and what to do next. `None` for a kind it can make.
/// An Object row without the model is [`NO_OBJECT_MODEL_SENTENCE`], whatever the node. A Subject or People row on a
/// computer that cannot make it is the sentence of section 3.10 whatever the reason (an older macOS, a Mac that cannot
/// run the requests, a computer that is not a Mac): "Node 2 has a Subject mask, and this computer cannot make one. Use
/// a Mac that supports Subject masks, or remove the mask."
#[must_use]
pub fn unmakeable_mask_sentence_with(
    support: MaskSupport,
    node: u32,
    kind: MaskKind,
) -> Option<String> {
    let (_, why) = mask_unavailable_with(support)
        .into_iter()
        .find(|(k, _)| *k == kind)?;
    let (article, name) = match kind {
        MaskKind::Object => return Some(NO_OBJECT_MODEL_SENTENCE.to_owned()),
        MaskKind::Depth => return Some(NO_DEPTH_MODEL_SENTENCE.to_owned()),
        MaskKind::Subject => ("a", "Subject"),
        MaskKind::People => ("a", "People"),
        MaskKind::Background => ("a", "Background"),
        // The other kinds can always be made.
        _ => return None,
    };
    // Subject and People: the sentence of section 3.10, whatever the reason.
    if matches!(kind, MaskKind::Subject | MaskKind::People) {
        return Some(format!(
            "Node {node} has {article} {name} mask, and this computer cannot make one. Use a Mac that supports {name} masks, or remove the mask."
        ));
    }
    Some(match why {
        MaskUnavailable::NotAMac | MaskUnavailable::MacosTooOld => format!(
            "Node {node} has {article} {name} mask, and this computer cannot make one. Use a Mac that supports {name} masks, or remove the mask."
        ),
        MaskUnavailable::WeightsMissing | MaskUnavailable::DepthWeightsMissing => format!(
            "Node {node} has {article} {name} mask, and the model weights are missing. Reinstall Yellowmoon, or remove the mask."
        ),
        MaskUnavailable::NotSupported => format!(
            "Node {node} has {article} {name} mask, and this version of Yellowmoon cannot make one yet. Remove the mask, or update Yellowmoon."
        ),
    })
}

/// An Object mask: SAM 2.1 from clicks or a box (`docs/23` section 4.4). With no point and no box it
/// selects nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ObjectMask {
    /// Clicks `[x, y, positive]` in the photo frame, at most [`MAX_OBJECT_POINTS`].
    pub points: Vec<(f64, f64, bool)>,
    /// A box `[x0, y0, x1, y1]` in the photo frame, or `None`.
    #[serde(rename = "box")]
    pub box_: Option<[f64; 4]>,
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

impl Default for ObjectMask {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            box_: None,
            feather: 0.0,
            grow: 0.0,
            refine: 0.5,
        }
    }
}

/// A Subject or People mask: Apple Vision (`docs/23` section 4.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FindMask {
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

impl Default for FindMask {
    fn default() -> Self {
        Self {
            feather: 0.0,
            grow: 0.0,
            refine: 0.5,
        }
    }
}

/// A Background mask: the blur map with a level and a softness (`docs/23` section 4.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BackgroundMask {
    /// Where the blur map's own decision falls, 0 to 1.
    pub level: f64,
    /// Softness of the threshold, 0 to 1.
    pub softness: f64,
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

impl Default for BackgroundMask {
    fn default() -> Self {
        Self {
            level: 0.5,
            softness: 0.5,
            feather: 0.0,
            grow: 0.0,
            refine: 0.5,
        }
    }
}

/// A Depth mask: a band of the photo's depth order (`docs/23` section 4.8, `docs/algorithms/06` section 5.8).
///
/// `near` and `far` are places in the photo's depth order, not distances: 0 is the nearest part of the photo and 1 the
/// farthest, so `near` 0 and `far` 0.35 is the nearest 35 % of the photo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DepthMask {
    /// Where the band starts, 0 (the nearest part of the photo) to 1.
    pub near: f64,
    /// Where the band ends, 0 to 1. A `near` above `far` is read as both at their mean.
    pub far: f64,
    /// The width of the ramp at both ends of the band, 0 to 1.
    pub softness: f64,
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

impl Default for DepthMask {
    fn default() -> Self {
        Self {
            near: 0.0,
            far: 0.5,
            softness: 0.5,
            feather: 0.0,
            grow: 0.0,
            refine: 0.5,
        }
    }
}

/// A Linear mask: a line, an angle and a feather.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LinearMask {
    /// A point of the line in the photo frame.
    pub center: [f64; 2],
    /// The line's angle in degrees, -180 to 180.
    pub angle_deg: f64,
    /// Feather, 0 to 1.
    pub feather: f64,
}

impl Default for LinearMask {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            angle_deg: 0.0,
            feather: 0.2,
        }
    }
}

/// A Radial mask: an ellipse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RadialMask {
    /// The centre in the photo frame.
    pub center: [f64; 2],
    /// The two radii, 0.002 to 1 of the long edge.
    pub radii: [f64; 2],
    /// The angle in degrees, -180 to 180.
    pub angle_deg: f64,
    /// Feather, 0 to 1.
    pub feather: f64,
}

impl Default for RadialMask {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            radii: [0.25, 0.18],
            angle_deg: 0.0,
            feather: 0.3,
        }
    }
}

/// One stroke of a Brush mask.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BrushStroke {
    /// The brush's size, 0.001 to 0.25 of the long edge.
    pub size: f64,
    /// The brush's hardness, 0 to 1.
    pub hardness: f64,
    /// An eraser stroke.
    pub erase: bool,
    /// The stroke's points in the photo frame.
    pub points: Vec<[f64; 2]>,
}

impl Default for BrushStroke {
    fn default() -> Self {
        Self {
            size: 0.03,
            hardness: 0.5,
            erase: false,
            points: Vec::new(),
        }
    }
}

/// A Brush mask: strokes drawn into a raster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BrushMask {
    /// The strokes in the order they were drawn, at most [`MAX_BRUSH_POINTS`] points in all.
    pub strokes: Vec<BrushStroke>,
    /// Feather, 0 to 1.
    pub feather: f64,
    /// Grow, -1 to 1.
    pub grow: f64,
    /// Refine edges, 0 to 1.
    pub refine: f64,
}

impl Default for BrushMask {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            feather: 0.0,
            grow: 0.0,
            refine: 0.0,
        }
    }
}

impl BrushMask {
    /// The number of points in all the strokes.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.strokes.iter().map(|s| s.points.len()).sum()
    }
}

/// A Colour range mask: a chroma centre, a width and a softness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ColorRangeMask {
    /// The centre in the chroma plane of C9, at most 0.5 long.
    pub center: [f64; 2],
    /// The width, 0.005 to 0.3.
    pub width: f64,
    /// The softness, 0 to 1.
    pub softness: f64,
}

impl Default for ColorRangeMask {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            width: 0.05,
            softness: 0.5,
        }
    }
}

/// A Brightness range mask: a low, a high and a softness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BrightnessRangeMask {
    /// The low end, 0 to 1.
    pub low: f64,
    /// The high end, 0 to 1. A `low` above `high` is read as both at their mean.
    pub high: f64,
    /// The softness, 0 to 1.
    pub softness: f64,
}

impl Default for BrightnessRangeMask {
    fn default() -> Self {
        Self {
            low: 0.35,
            high: 1.0,
            softness: 0.5,
        }
    }
}

/// What a mask row finds or draws: the fields that depend on its kind.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskShape {
    /// SAM 2.1 from clicks or a box.
    Object(ObjectMask),
    /// Apple Vision's foreground.
    Subject(FindMask),
    /// Apple Vision's people.
    People(FindMask),
    /// The blur map.
    Background(BackgroundMask),
    /// A band of the depth map.
    Depth(DepthMask),
    /// A line and a feather.
    Linear(LinearMask),
    /// An ellipse.
    Radial(RadialMask),
    /// Strokes.
    Brush(BrushMask),
    /// A chroma range.
    ColorRange(ColorRangeMask),
    /// A brightness range.
    BrightnessRange(BrightnessRangeMask),
}

impl MaskShape {
    /// The kind's JSON name (`"color_range"`).
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Object(_) => "object",
            Self::Subject(_) => "subject",
            Self::People(_) => "people",
            Self::Background(_) => "background",
            Self::Depth(_) => "depth",
            Self::Linear(_) => "linear",
            Self::Radial(_) => "radial",
            Self::Brush(_) => "brush",
            Self::ColorRange(_) => "color_range",
            Self::BrightnessRange(_) => "brightness_range",
        }
    }

    /// The kind as the person reads it in a sentence ("an Object mask").
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Object(_) => "an Object",
            Self::Subject(_) => "a Subject",
            Self::People(_) => "a People",
            Self::Background(_) => "a Background",
            Self::Depth(_) => "a Depth",
            Self::Linear(_) => "a Linear",
            Self::Radial(_) => "a Radial",
            Self::Brush(_) => "a Brush",
            Self::ColorRange(_) => "a Colour range",
            Self::BrightnessRange(_) => "a Brightness range",
        }
    }

    /// The default shape of a kind named by its JSON name, if it is one.
    #[must_use]
    pub fn default_of(kind: &str) -> Option<Self> {
        Some(match kind {
            "object" => Self::Object(ObjectMask::default()),
            "subject" => Self::Subject(FindMask::default()),
            "people" => Self::People(FindMask::default()),
            "background" => Self::Background(BackgroundMask::default()),
            "depth" => Self::Depth(DepthMask::default()),
            "linear" => Self::Linear(LinearMask::default()),
            "radial" => Self::Radial(RadialMask::default()),
            "brush" => Self::Brush(BrushMask::default()),
            "color_range" => Self::ColorRange(ColorRangeMask::default()),
            "brightness_range" => Self::BrightnessRange(BrightnessRangeMask::default()),
            _ => return None,
        })
    }

    /// The names of the fields a row of this kind has besides `kind`, `mode`, `invert`, `enabled`
    /// and `opacity`.
    #[must_use]
    pub fn own_fields(&self) -> &'static [&'static str] {
        match self {
            Self::Object(_) => &["points", "box", "feather", "grow", "refine"],
            Self::Subject(_) | Self::People(_) => &["feather", "grow", "refine"],
            Self::Background(_) => &["level", "softness", "feather", "grow", "refine"],
            Self::Depth(_) => &["near", "far", "softness", "feather", "grow", "refine"],
            Self::Linear(_) => &["center", "angle_deg", "feather"],
            Self::Radial(_) => &["center", "radii", "angle_deg", "feather"],
            Self::Brush(_) => &["strokes", "feather", "grow", "refine"],
            Self::ColorRange(_) => &["center", "width", "softness"],
            Self::BrightnessRange(_) => &["low", "high", "softness"],
        }
    }

    /// The same shape from a kind's JSON name and the row's other fields (the common ones already
    /// taken out). Unknown fields are refused.
    fn read(kind: &str, fields: Map<String, Value>) -> Result<Self, SettingsError> {
        let value = Value::Object(fields);
        let parse = |e: serde_json::Error| SettingsError::Parse(e.to_string());
        Ok(match kind {
            "object" => Self::Object(serde_json::from_value(value).map_err(parse)?),
            "subject" => Self::Subject(serde_json::from_value(value).map_err(parse)?),
            "people" => Self::People(serde_json::from_value(value).map_err(parse)?),
            "background" => Self::Background(serde_json::from_value(value).map_err(parse)?),
            "depth" => Self::Depth(serde_json::from_value(value).map_err(parse)?),
            "linear" => Self::Linear(serde_json::from_value(value).map_err(parse)?),
            "radial" => Self::Radial(serde_json::from_value(value).map_err(parse)?),
            "brush" => Self::Brush(serde_json::from_value(value).map_err(parse)?),
            "color_range" => Self::ColorRange(serde_json::from_value(value).map_err(parse)?),
            "brightness_range" => {
                Self::BrightnessRange(serde_json::from_value(value).map_err(parse)?)
            }
            other => {
                return Err(SettingsError::Parse(format!(
                    "a mask can't be of the kind “{other}”; the kinds are {}",
                    MASK_KINDS.join(", ")
                )));
            }
        })
    }
}

/// One row of a node's masks (`docs/23` section 3.3). JSON: one flat object, `kind` and the
/// common fields beside the kind's own.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskItem {
    /// Add or subtract.
    pub mode: MaskMode,
    /// This mask alone is inverted.
    pub invert: bool,
    /// Off (the eye) counts as absent.
    pub enabled: bool,
    /// 0 to 1.
    pub opacity: f64,
    /// The kind and its fields.
    pub shape: MaskShape,
}

impl MaskItem {
    /// A row of this shape with the other fields at their defaults.
    #[must_use]
    pub fn new(shape: MaskShape) -> Self {
        Self {
            mode: MaskMode::Add,
            invert: false,
            enabled: true,
            opacity: 1.0,
            shape,
        }
    }

    /// An Object row with no point and no box: it selects nothing.
    #[must_use]
    pub fn emptied(&self) -> Self {
        let mut item = self.clone();
        if let MaskShape::Object(o) = &mut item.shape {
            o.points.clear();
            o.box_ = None;
        }
        item
    }

    /// Reads a row from its JSON object.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Parse`] for a kind that is not one of the ten, a missing `kind`, a field
    /// the kind does not have, or a value of the wrong type.
    pub fn from_map(mut map: Map<String, Value>) -> Result<Self, SettingsError> {
        let parse = |e: serde_json::Error| SettingsError::Parse(e.to_string());
        let kind = match map.remove("kind") {
            Some(Value::String(k)) => k,
            _ => {
                return Err(SettingsError::Parse(
                    "a mask needs a `kind` (object, subject, people, background, depth, linear, radial, brush, color_range or brightness_range)"
                        .to_owned(),
                ));
            }
        };
        let mode = match map.remove("mode") {
            Some(v) => serde_json::from_value(v).map_err(parse)?,
            None => MaskMode::Add,
        };
        let invert = match map.remove("invert") {
            Some(v) => serde_json::from_value(v).map_err(parse)?,
            None => false,
        };
        let enabled = match map.remove("enabled") {
            Some(v) => serde_json::from_value(v).map_err(parse)?,
            None => true,
        };
        let opacity = match map.remove("opacity") {
            Some(v) => serde_json::from_value(v).map_err(parse)?,
            None => 1.0,
        };
        let shape = MaskShape::read(&kind, map)?;
        Ok(Self {
            mode,
            invert,
            enabled,
            opacity,
            shape,
        })
    }
}

impl Serialize for MaskItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.shape).map_err(serde::ser::Error::custom)?;
        if let Value::Object(map) = &mut value {
            map.insert(
                "mode".to_owned(),
                serde_json::to_value(self.mode).map_err(serde::ser::Error::custom)?,
            );
            map.insert("invert".to_owned(), Value::Bool(self.invert));
            map.insert("enabled".to_owned(), Value::Bool(self.enabled));
            map.insert(
                "opacity".to_owned(),
                serde_json::to_value(self.opacity).map_err(serde::ser::Error::custom)?,
            );
        }
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MaskItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            // A `Parse` error already says "The settings can't be read:" where the document is read; the inner
            // text alone goes up, so the sentence is not said twice.
            Value::Object(map) => Self::from_map(map).map_err(|e| match e {
                SettingsError::Parse(text) => serde::de::Error::custom(text),
                other => serde::de::Error::custom(other),
            }),
            other => Err(serde::de::Error::custom(format!(
                "a mask must be an object, not {other}"
            ))),
        }
    }
}

/// A node's masks: the rows and the node's own Invert.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MaskSettings {
    /// The node's Invert: `m = 1 - m`.
    pub invert: bool,
    /// The rows, in the order they are drawn and combined, at most [`MAX_MASKS_PER_NODE`].
    pub items: Vec<MaskItem>,
}

// ---------------------------------------------------------------------------------------------
// Checking
// ---------------------------------------------------------------------------------------------

/// The clamping state of one mask row: whether anything was changed.
#[derive(Default)]
struct Changed(bool);

impl Changed {
    fn number(
        &mut self,
        prefix: &str,
        name: &str,
        value: &mut f64,
        range: (f64, f64),
    ) -> Result<(), SettingsError> {
        let (v, c) = clamp_to(&format!("{prefix}.{name}"), *value, range)?;
        *value = v;
        self.0 |= c;
        Ok(())
    }

    fn point(
        &mut self,
        prefix: &str,
        name: &str,
        p: &mut [f64; 2],
        range: (f64, f64),
    ) -> Result<(), SettingsError> {
        for (i, v) in p.iter_mut().enumerate() {
            self.number(prefix, &format!("{name}[{i}]"), v, range)?;
        }
        Ok(())
    }
}

impl MaskSettings {
    /// Clamps every number of every row to its range (`docs/23` section 3.5), counts the limits
    /// and returns whether anything was changed. `prefix` names the masks for an error
    /// ("nodes.items[2].masks").
    ///
    /// # Errors
    ///
    /// [`SettingsError::NotFinite`] for a number that is not a number,
    /// [`SettingsError::Graph`] for more rows or points than the limits allow.
    pub(crate) fn sanitise(&mut self, prefix: &str) -> Result<bool, SettingsError> {
        use super::graph::GraphError;
        if self.items.len() > MAX_MASKS_PER_NODE {
            return Err(GraphError::TooManyMasks.into());
        }
        let mut changed = Changed::default();
        for (i, item) in self.items.iter_mut().enumerate() {
            let p = format!("{prefix}.items[{i}]");
            changed.number(&p, "opacity", &mut item.opacity, UNIT_RANGE)?;
            match &mut item.shape {
                MaskShape::Object(o) => {
                    if o.points.len() > MAX_OBJECT_POINTS {
                        return Err(GraphError::TooManyObjectPoints.into());
                    }
                    for (j, pt) in o.points.iter_mut().enumerate() {
                        let mut xy = [pt.0, pt.1];
                        changed.point(&p, &format!("points[{j}]"), &mut xy, UNIT_RANGE)?;
                        (pt.0, pt.1) = (xy[0], xy[1]);
                    }
                    if let Some(b) = &mut o.box_ {
                        let mut lo = [b[0], b[1]];
                        let mut hi = [b[2], b[3]];
                        changed.point(&p, "box", &mut lo, UNIT_RANGE)?;
                        changed.point(&p, "box", &mut hi, UNIT_RANGE)?;
                        // A box is read corner to corner, whichever corner came first.
                        let ordered = [
                            lo[0].min(hi[0]),
                            lo[1].min(hi[1]),
                            lo[0].max(hi[0]),
                            lo[1].max(hi[1]),
                        ];
                        changed.0 |= ordered != *b;
                        *b = ordered;
                    }
                    changed.number(&p, "feather", &mut o.feather, UNIT_RANGE)?;
                    changed.number(&p, "grow", &mut o.grow, GROW_RANGE)?;
                    changed.number(&p, "refine", &mut o.refine, UNIT_RANGE)?;
                }
                MaskShape::Subject(f) | MaskShape::People(f) => {
                    changed.number(&p, "feather", &mut f.feather, UNIT_RANGE)?;
                    changed.number(&p, "grow", &mut f.grow, GROW_RANGE)?;
                    changed.number(&p, "refine", &mut f.refine, UNIT_RANGE)?;
                }
                MaskShape::Background(b) => {
                    changed.number(&p, "level", &mut b.level, UNIT_RANGE)?;
                    changed.number(&p, "softness", &mut b.softness, UNIT_RANGE)?;
                    changed.number(&p, "feather", &mut b.feather, UNIT_RANGE)?;
                    changed.number(&p, "grow", &mut b.grow, GROW_RANGE)?;
                    changed.number(&p, "refine", &mut b.refine, UNIT_RANGE)?;
                }
                MaskShape::Depth(d) => {
                    changed.number(&p, "near", &mut d.near, UNIT_RANGE)?;
                    changed.number(&p, "far", &mut d.far, UNIT_RANGE)?;
                    if d.near > d.far {
                        // Read as both at their mean (`docs/23` section 4.8), as the Brightness range's low and high.
                        let mean = (d.near + d.far) / 2.0;
                        (d.near, d.far) = (mean, mean);
                    }
                    changed.number(&p, "softness", &mut d.softness, UNIT_RANGE)?;
                    changed.number(&p, "feather", &mut d.feather, UNIT_RANGE)?;
                    changed.number(&p, "grow", &mut d.grow, GROW_RANGE)?;
                    changed.number(&p, "refine", &mut d.refine, UNIT_RANGE)?;
                }
                MaskShape::Linear(l) => {
                    changed.point(&p, "center", &mut l.center, UNIT_RANGE)?;
                    changed.number(&p, "angle_deg", &mut l.angle_deg, MASK_ANGLE_RANGE_DEG)?;
                    changed.number(&p, "feather", &mut l.feather, UNIT_RANGE)?;
                }
                MaskShape::Radial(r) => {
                    changed.point(&p, "center", &mut r.center, UNIT_RANGE)?;
                    changed.point(&p, "radii", &mut r.radii, RADII_RANGE)?;
                    changed.number(&p, "angle_deg", &mut r.angle_deg, MASK_ANGLE_RANGE_DEG)?;
                    changed.number(&p, "feather", &mut r.feather, UNIT_RANGE)?;
                }
                MaskShape::Brush(b) => {
                    if b.point_count() > MAX_BRUSH_POINTS {
                        return Err(GraphError::TooManyBrushPoints.into());
                    }
                    for (j, s) in b.strokes.iter_mut().enumerate() {
                        let sp = format!("{p}.strokes[{j}]");
                        changed.number(&sp, "size", &mut s.size, STROKE_SIZE_RANGE)?;
                        changed.number(&sp, "hardness", &mut s.hardness, UNIT_RANGE)?;
                        for (k, pt) in s.points.iter_mut().enumerate() {
                            changed.point(&sp, &format!("points[{k}]"), pt, UNIT_RANGE)?;
                        }
                    }
                    changed.number(&p, "feather", &mut b.feather, UNIT_RANGE)?;
                    changed.number(&p, "grow", &mut b.grow, GROW_RANGE)?;
                    changed.number(&p, "refine", &mut b.refine, UNIT_RANGE)?;
                }
                MaskShape::ColorRange(c) => {
                    for (k, v) in c.center.iter().enumerate() {
                        if !v.is_finite() {
                            return Err(SettingsError::NotFinite(format!("{p}.center[{k}]")));
                        }
                    }
                    let len = c.center[0].hypot(c.center[1]);
                    if len > COLOR_RANGE_CENTER_LIMIT {
                        let k = COLOR_RANGE_CENTER_LIMIT / len;
                        c.center = [c.center[0] * k, c.center[1] * k];
                        changed.0 = true;
                    }
                    c.center = [c.center[0] + 0.0, c.center[1] + 0.0];
                    changed.number(&p, "width", &mut c.width, COLOR_RANGE_WIDTH_RANGE)?;
                    changed.number(&p, "softness", &mut c.softness, UNIT_RANGE)?;
                }
                MaskShape::BrightnessRange(r) => {
                    changed.number(&p, "low", &mut r.low, UNIT_RANGE)?;
                    changed.number(&p, "high", &mut r.high, UNIT_RANGE)?;
                    if r.low > r.high {
                        // Read as both at their mean (`docs/23` section 3.3): not clamping.
                        let mean = (r.low + r.high) / 2.0;
                        (r.low, r.high) = (mean, mean);
                    }
                    changed.number(&p, "softness", &mut r.softness, UNIT_RANGE)?;
                }
            }
        }
        Ok(changed.0)
    }
}

// ---------------------------------------------------------------------------------------------
// Row patches
// ---------------------------------------------------------------------------------------------

/// Reads a field that may be absent (leave it), `null` (clear it) or a value (set it).
fn double_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// A partial of one mask row's fields (`masks.edit`, `docs/23` section 3.7). It cannot change the
/// row's kind. `strokes_append` and `strokes_keep` are for a Brush row.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskRowPatch {
    /// The row's kind: allowed only when it is the row's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Add or subtract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<MaskMode>,
    /// This mask alone is inverted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invert: Option<bool>,
    /// The eye.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// 0 to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    /// Feather.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feather: Option<f64>,
    /// Grow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grow: Option<f64>,
    /// Refine edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine: Option<f64>,
    /// An Object row's clicks, replaced whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<(f64, f64, bool)>>,
    /// An Object row's box; `null` takes it away.
    #[serde(
        default,
        rename = "box",
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub box_: Option<Option<[f64; 4]>>,
    /// A Background row's level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<f64>,
    /// A Depth row's `near`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near: Option<f64>,
    /// A Depth row's `far`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub far: Option<f64>,
    /// The softness of a Background, Depth, Colour range or Brightness range row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub softness: Option<f64>,
    /// A Linear, Radial or Colour range row's centre.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<[f64; 2]>,
    /// A Linear or Radial row's angle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle_deg: Option<f64>,
    /// A Radial row's radii.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radii: Option<[f64; 2]>,
    /// A Brush row's strokes, replaced whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes: Option<Vec<BrushStroke>>,
    /// Strokes added after the existing ones (a Brush row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes_append: Option<Vec<BrushStroke>>,
    /// Keep only the first n strokes (a Brush row; the inverse of an append).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes_keep: Option<usize>,
    /// A Colour range row's width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    /// A Brightness range row's low end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<f64>,
    /// A Brightness range row's high end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<f64>,
}

impl MaskRowPatch {
    /// The names of the fields the patch carries, in the order of the struct.
    fn names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        macro_rules! has {
            ($($f:ident => $n:expr),* $(,)?) => { $(if self.$f.is_some() { v.push($n); })* };
        }
        has!(
            kind => "kind", mode => "mode", invert => "invert", enabled => "enabled",
            opacity => "opacity", feather => "feather", grow => "grow", refine => "refine",
            points => "points", box_ => "box", level => "level", near => "near", far => "far",
            softness => "softness",
            center => "center", angle_deg => "angle_deg", radii => "radii", strokes => "strokes",
            strokes_append => "strokes_append", strokes_keep => "strokes_keep", width => "width",
            low => "low", high => "high",
        );
        v
    }

    /// Applies the patch to a row.
    ///
    /// # Errors
    ///
    /// [`SettingsError::MaskKind`] when the patch names another kind than the row's and
    /// [`SettingsError::UnknownMaskField`] for a field the row's kind does not have.
    pub(crate) fn apply(&self, row: &mut MaskItem) -> Result<(), SettingsError> {
        if let Some(kind) = &self.kind
            && kind != row.shape.kind_name()
        {
            return Err(SettingsError::MaskKind);
        }
        let own = row.shape.own_fields();
        let common = ["kind", "mode", "invert", "enabled", "opacity"];
        let brush = matches!(row.shape, MaskShape::Brush(_));
        for name in self.names() {
            let stroke_op = matches!(name, "strokes_append" | "strokes_keep");
            let ok = common.contains(&name) || own.contains(&name) || (brush && stroke_op);
            if !ok {
                let mut mask = row.shape.display_name().to_owned();
                if let Some(first) = mask.get_mut(0..1) {
                    first.make_ascii_uppercase();
                }
                return Err(SettingsError::UnknownMaskField {
                    mask,
                    field: name.to_owned(),
                });
            }
        }
        if let Some(v) = self.mode {
            row.mode = v;
        }
        if let Some(v) = self.invert {
            row.invert = v;
        }
        if let Some(v) = self.enabled {
            row.enabled = v;
        }
        if let Some(v) = self.opacity {
            row.opacity = v;
        }
        macro_rules! set {
            ($target:expr, $field:ident) => {
                if let Some(v) = self.$field.clone() {
                    $target.$field = v;
                }
            };
        }
        match &mut row.shape {
            MaskShape::Object(o) => {
                set!(o, feather);
                set!(o, grow);
                set!(o, refine);
                if let Some(v) = &self.points {
                    o.points.clone_from(v);
                }
                if let Some(v) = self.box_ {
                    o.box_ = v;
                }
            }
            MaskShape::Subject(f) | MaskShape::People(f) => {
                set!(f, feather);
                set!(f, grow);
                set!(f, refine);
            }
            MaskShape::Background(b) => {
                set!(b, level);
                set!(b, softness);
                set!(b, feather);
                set!(b, grow);
                set!(b, refine);
            }
            MaskShape::Depth(d) => {
                set!(d, near);
                set!(d, far);
                set!(d, softness);
                set!(d, feather);
                set!(d, grow);
                set!(d, refine);
            }
            MaskShape::Linear(l) => {
                set!(l, center);
                set!(l, angle_deg);
                set!(l, feather);
            }
            MaskShape::Radial(r) => {
                set!(r, center);
                set!(r, radii);
                set!(r, angle_deg);
                set!(r, feather);
            }
            MaskShape::Brush(b) => {
                set!(b, feather);
                set!(b, grow);
                set!(b, refine);
                if let Some(v) = &self.strokes {
                    b.strokes.clone_from(v);
                }
                // Keep first, then append: an undo of an append is a keep, and a redo an append.
                if let Some(n) = self.strokes_keep {
                    b.strokes.truncate(n);
                }
                if let Some(v) = &self.strokes_append {
                    b.strokes.extend(v.iter().cloned());
                }
            }
            MaskShape::ColorRange(c) => {
                set!(c, center);
                set!(c, width);
                set!(c, softness);
            }
            MaskShape::BrightnessRange(r) => {
                set!(r, low);
                set!(r, high);
                set!(r, softness);
            }
        }
        Ok(())
    }
}

/// A node's masks to change (`docs/23` section 3.7): the node's Invert, the whole list, and rows
/// by index.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasksPatch {
    /// The node's Invert.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invert: Option<bool>,
    /// The whole list of rows: add, remove or reorder rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<MaskItem>>,
    /// Row patches by the row's index (a string), applied after `items`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<std::collections::BTreeMap<String, MaskRowPatch>>,
}

impl MasksPatch {
    /// Applies the patch to a node's masks.
    ///
    /// # Errors
    ///
    /// [`SettingsError::NoMaskRow`] for an index the list does not have or that is not a number,
    /// and the errors of [`MaskRowPatch`].
    pub(crate) fn apply(&self, masks: &mut MaskSettings, node: &str) -> Result<(), SettingsError> {
        if let Some(v) = self.invert {
            masks.invert = v;
        }
        if let Some(items) = &self.items {
            masks.items.clone_from(items);
        }
        if let Some(edit) = &self.edit {
            // Row indices are numbers written as strings; apply them in numeric order.
            let mut rows: Vec<(usize, &MaskRowPatch)> = Vec::with_capacity(edit.len());
            for (key, patch) in edit {
                let index = key.parse::<usize>().map_err(|_| SettingsError::NoMaskRow {
                    node: node.to_owned(),
                    index: key.clone(),
                })?;
                rows.push((index, patch));
            }
            rows.sort_by_key(|(i, _)| *i);
            for (index, patch) in rows {
                let row = masks
                    .items
                    .get_mut(index)
                    .ok_or_else(|| SettingsError::NoMaskRow {
                        node: node.to_owned(),
                        index: index.to_string(),
                    })?;
                patch.apply(row)?;
            }
        }
        Ok(())
    }
}
