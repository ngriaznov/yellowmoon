//! The node graph a photo's settings are (`docs/23-nodes.md` section 3): the Input, the Output and
//! the correctors with their fields, the typed links between them, the rules a graph must keep
//! and the reading of a document over the default graph.
//!
//! N-02 makes the settings node-first without changing a pixel: the render still uses the Input,
//! Main (corrector 1) and the Output as it used the flat fields before. The other correctors,
//! their masks and the links are read, checked, patched, copied and written, and rendered from
//! N-03 on.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use ym_color::wb::LightSource;
use ym_looks::library::LookId;

use super::film::{
    FILM_EXPOSURE_RANGE_EV, FilmNode, GRAIN_AMOUNT_RANGE, HALATION_AMOUNT_RANGE, LIGHTS_RANGE,
    PRINT_EXPOSURE_RANGE_EV, SIZE_RANGE, STRENGTH_RANGE,
};
use super::masks::MaskSettings;
use super::{
    AutoOr, BalanceSetting, ColorBalance, ColorBalancePoint, ColorWarp, ContrastSetting, Crop,
    Curves, DecodeOptions, ExposureSetting, FieldRef, LensSettings, NoiseReductionSetting,
    OUTPUT_SHARPENING_DEFAULT, Orientation, PercentSetting, PercentValue, Rolloff,
    SaturationSetting, SettingsError, SettingsField, SharpenSetting, WhiteBalanceSetting,
    percent_document,
};

/// The most corrector nodes a photo may hold, Main included (the Input and the Output do not
/// count).
pub const MAX_CORRECTORS: usize = 64;
/// The id of Main, the corrector every photo has.
pub const MAIN_ID: u32 = 1;
/// The longest name a node may have, in characters.
pub const MAX_NAME_CHARS: usize = 32;
/// The range of a card's `pos` on the canvas, in grid units.
pub const POS_RANGE: (f64, f64) = (-1000.0, 1000.0);
/// Range of a node's exposure trim (stops).
pub const TRIM_EXPOSURE_RANGE_EV: (f64, f64) = (-4.0, 4.0);
/// Range of a node's temperature and tint trims.
pub const TRIM_RANGE: (f64, f64) = (-100.0, 100.0);
/// The name of Main.
pub const MAIN_NAME: &str = "Main";
/// The name a new corrector has.
pub const DEFAULT_NAME: &str = "Corrector";
/// The name a new Film node has.
pub const DEFAULT_FILM_NAME: &str = "Film";

// ---------------------------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------------------------

/// A rule of the graph that was broken (`docs/23` section 3.5). The texts are the plain sentences
/// of the spec.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// The Input, the Output and Main are not what they must be.
    #[error("A photo needs its input, its output and node 1, Main.")]
    MissingFixed,
    /// A node has a field its kind does not have.
    #[error("{kind} has no field called “{field}”.")]
    UnknownField {
        /// "An input", "An output" or "A corrector".
        kind: &'static str,
        /// The field.
        field: String,
    },
    /// A link goes from an input port or to an output port.
    #[error("A link goes from the output of one node to the input of another.")]
    LinkDirection,
    /// A link joins a picture to a key.
    #[error("A link joins a picture to a picture, or a key to a key.")]
    LinkPortTypes,
    /// A link names a node the photo does not have.
    #[error("A link names node {0}, which this photo does not have.")]
    UnknownNode(String),
    /// A link names a port the node does not have.
    #[error("{node} has no port called “{port}”.")]
    UnknownPort {
        /// "Node 7", "The input" or "The output".
        node: String,
        /// The port.
        port: String,
    },
    /// An input has two links.
    #[error("An input can have one link. {node} has two on its {port} input.")]
    TwoLinks {
        /// "Node 3", "The output".
        node: String,
        /// The port.
        port: String,
    },
    /// The Output's picture is not linked exactly once.
    #[error("The output needs one picture link.")]
    OutputLink,
    /// Main does not take its picture from the Input.
    #[error("Node 1, Main, must take its picture from the input.")]
    MainInput,
    /// The links make a circle.
    #[error("The nodes of this photo are linked in a circle. Remove a link.")]
    Cycle,
    /// Two nodes have one id.
    #[error("Two nodes have the id {0}.")]
    DuplicateId(String),
    /// A node of a kind this version does not know.
    #[error(
        "This settings file has a node of a kind (“{0}”) that this version of Yellowmoon does not have. Update Yellowmoon."
    )]
    UnknownKind(String),
    /// A new node gave no kind.
    #[error("Node {0} is not in the default graph, so it needs its “kind”.")]
    NeedsKind(String),
    /// More correctors than the limit.
    #[error("A photo can have up to 64 nodes.")]
    TooManyNodes,
    /// More masks in a node than the limit.
    #[error("A node can have up to 64 masks.")]
    TooManyMasks,
    /// More brush points than the limit.
    #[error("A brush mask can have up to 100,000 points.")]
    TooManyBrushPoints,
    /// More Object points than the limit.
    #[error("An Object mask can have up to 256 points.")]
    TooManyObjectPoints,
    /// A patch removes a node that cannot be removed.
    #[error("The input, the output and node 1, Main, can't be removed.")]
    CannotRemove,
    /// A patch names a node the photo does not have and gives no kind.
    #[error("The patch names node {0}, which this photo does not have.")]
    PatchUnknownNode(String),
    /// A patch gives a kind other than the node's.
    #[error("{node} is {actual}, not {given}.")]
    PatchWrongKind {
        /// "Node 7", "The input", "The output".
        node: String,
        /// "a corrector", "an input" or "an output".
        actual: &'static str,
        /// The same for the kind the patch gave.
        given: &'static str,
    },
}

// ---------------------------------------------------------------------------------------------
// Ids, kinds, ports and links
// ---------------------------------------------------------------------------------------------

/// A node's id: `"input"`, `"output"` or an integer 1 to 2³¹ − 1 (Main is 1). The order is the
/// order a graph is written in: the Input, the correctors by id, the Output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeId {
    /// The fixed Input.
    Input,
    /// A corrector.
    Corrector(u32),
    /// The fixed Output.
    Output,
}

impl NodeId {
    /// Main, corrector 1.
    pub const MAIN: NodeId = NodeId::Corrector(MAIN_ID);

    /// The kind a node with this id is **when the graph is not at hand**: a numbered node reads as a corrector.
    /// A Film node has a number too, and [`NodeGraph::kind_of`] tells the two apart.
    #[must_use]
    pub fn kind(self) -> NodeKind {
        match self {
            Self::Input => NodeKind::Input,
            Self::Corrector(_) => NodeKind::Corrector,
            Self::Output => NodeKind::Output,
        }
    }

    /// The node as a sentence starts with it: "Node 7", "The input", "The output".
    #[must_use]
    pub fn sentence(self) -> String {
        match self {
            Self::Input => "The input".to_owned(),
            Self::Output => "The output".to_owned(),
            Self::Corrector(n) => format!("Node {n}"),
        }
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => f.write_str("input"),
            Self::Output => f.write_str("output"),
            Self::Corrector(n) => write!(f, "{n}"),
        }
    }
}

impl FromStr for NodeId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "input" => Ok(Self::Input),
            "output" => Ok(Self::Output),
            digits if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => digits
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=i32::MAX as u32).contains(n))
                .map(Self::Corrector)
                .ok_or_else(|| format!("a node id must be 1 to {}, not {s}", i32::MAX)),
            other => Err(format!(
                "a node id is “input”, “output” or a number, not “{other}”"
            )),
        }
    }
}

impl Serialize for NodeId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Input => serializer.serialize_str("input"),
            Self::Output => serializer.serialize_str("output"),
            Self::Corrector(n) => serializer.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for NodeId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl Visitor<'_> for V {
            type Value = NodeId;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("“input”, “output” or a node number")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<NodeId, E> {
                v.parse().map_err(E::custom)
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<NodeId, E> {
                u32::try_from(v)
                    .ok()
                    .filter(|n| (1..=i32::MAX as u32).contains(n))
                    .map(NodeId::Corrector)
                    .ok_or_else(|| {
                        E::custom(format!("a node id must be 1 to {}, not {v}", i32::MAX))
                    })
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<NodeId, E> {
                u64::try_from(v)
                    .map_err(|_| E::custom(format!("a node id must be 1 to {}, not {v}", i32::MAX)))
                    .and_then(|v| self.visit_u64(v))
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// What a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// The fixed Input.
    Input,
    /// The fixed Output.
    Output,
    /// A corrector.
    Corrector,
    /// A Film node (`docs/24`).
    Film,
}

impl NodeKind {
    /// Every kind. The scheduler's footprint list (`pipeline::schedule::step_kind`) is an exhaustive `match` over the same
    /// kinds, and the random-graph test draws from this list (`docs/23` section 5.12).
    pub const ALL: [NodeKind; 4] = [
        NodeKind::Input,
        NodeKind::Output,
        NodeKind::Corrector,
        NodeKind::Film,
    ];

    /// "an input", "an output", "a corrector".
    #[must_use]
    pub fn with_article(self) -> &'static str {
        match self {
            Self::Input => "an input",
            Self::Output => "an output",
            Self::Corrector => "a corrector",
            Self::Film => "a film node",
        }
    }

    /// "An input", "An output", "A corrector".
    #[must_use]
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Input => "An input",
            Self::Output => "An output",
            Self::Corrector => "A corrector",
            Self::Film => "A film node",
        }
    }

    /// The kind's JSON name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Corrector => "corrector",
            Self::Film => "film",
        }
    }

    /// The kind with this JSON name.
    ///
    /// # Errors
    ///
    /// [`GraphError::UnknownKind`] for a kind this version does not have.
    pub fn parse(name: &str) -> Result<Self, GraphError> {
        match name {
            "input" => Ok(Self::Input),
            "output" => Ok(Self::Output),
            "corrector" => Ok(Self::Corrector),
            "film" => Ok(Self::Film),
            other => Err(GraphError::UnknownKind(other.to_owned())),
        }
    }
}

/// A port of a node: a picture or a key (a mask).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Port {
    /// The LogC4 picture.
    Picture,
    /// A mask.
    Key,
    /// A port name no node has (the validator refuses it).
    Other(String),
}

impl Port {
    /// The port's JSON name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Picture => "picture",
            Self::Key => "key",
            Self::Other(s) => s,
        }
    }
}

impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Port {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Port {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "picture" => Self::Picture,
            "key" => Self::Key,
            _ => Self::Other(s),
        })
    }
}

/// One end of a link: a node and one of its ports.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeRef {
    /// The node.
    pub node: NodeId,
    /// The port.
    pub port: Port,
}

impl NodeRef {
    /// The picture port of a node.
    #[must_use]
    pub fn picture(node: NodeId) -> Self {
        Self {
            node,
            port: Port::Picture,
        }
    }

    /// The key port of a node.
    #[must_use]
    pub fn key(node: NodeId) -> Self {
        Self {
            node,
            port: Port::Key,
        }
    }
}

/// A link from an output port to an input port (`docs/23` section 3.4).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeLink {
    /// The output port the link starts at.
    pub from: NodeRef,
    /// The input port it ends at.
    pub to: NodeRef,
}

impl NodeLink {
    /// A picture link from one node to another.
    #[must_use]
    pub fn picture(from: NodeId, to: NodeId) -> Self {
        Self {
            from: NodeRef::picture(from),
            to: NodeRef::picture(to),
        }
    }

    /// A key link from one node's key output to another's key input.
    #[must_use]
    pub fn key(from: NodeId, to: NodeId) -> Self {
        Self {
            from: NodeRef::key(from),
            to: NodeRef::key(to),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------------------------

/// A card's place on the canvas, `[x, y]` in grid units (`docs/23` section 3.2).
pub type NodePos = [f64; 2];

/// The Input (`docs/23` section 3.2): the raw, light and framing settings of the photo. Every field
/// has the name and the type it had when the settings were flat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct InputNode {
    /// The card's place.
    pub pos: NodePos,
    /// Light › Exposure: Auto or an absolute value, -4 to +4 EV.
    pub exposure: ExposureSetting,
    /// Colour › White balance, the light presets.
    pub light_source: LightSource,
    /// Colour › White balance: Auto or Kelvin and tint.
    pub white_balance: WhiteBalanceSetting,
    /// Pick white: the sampled point `[x, y]` in [0, 1]², measured on the photo as the camera
    /// oriented it (EXIF orientation applied, before the user's crop, straighten and
    /// `orientation`), so it keeps pointing at the same spot when those change. See
    /// [`crate::auto::display_to_sensor`].
    pub white_pick: Option<[f64; 2]>,
    /// RAW › Range: the highlight range and the low-light toe (C7).
    pub rolloff: Rolloff,
    /// RAW › Decode switches.
    pub decode: DecodeOptions,
    /// Detail › Noise reduction: Auto or luminance, colour and detail.
    pub noise_reduction: NoiseReductionSetting,
    /// Detail › Sharpening: Auto or amount and radius.
    pub sharpen: SharpenSetting,
    /// Look › Lens: the lens and its three amounts (`docs/21`); no lens by default.
    pub lens: LensSettings,
    /// Crop; `null` when the photo is not cropped.
    pub crop: Option<Crop>,
    /// Orientation after the camera's EXIF orientation.
    pub orientation: Orientation,
}

/// The place the Input has in the default graph.
pub const INPUT_POS: NodePos = [-1.0, 0.0];

impl Default for InputNode {
    fn default() -> Self {
        Self {
            pos: INPUT_POS,
            exposure: AutoOr::Auto,
            light_source: LightSource::Auto,
            white_balance: AutoOr::Auto,
            white_pick: None,
            rolloff: Rolloff::default(),
            decode: DecodeOptions::default(),
            noise_reduction: AutoOr::Auto,
            sharpen: AutoOr::Auto,
            lens: LensSettings::default(),
            crop: None,
            orientation: Orientation::Deg0,
        }
    }
}

/// The Output (`docs/23` section 3.2): the look and the display, and the sharpening at the size
/// the picture is shown or written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OutputNode {
    /// The card's place.
    pub pos: NodePos,
    /// Look tab. The user's default look (Signature out of the box; a photo a session opens gets it
    /// through [`DevelopSettings::with_look`](super::DevelopSettings::with_look)).
    pub look: LookId,
    /// 0 to 100 percent.
    pub look_intensity: f64,
    /// Detail › Sharpening › Output: the strength of the output sharpening in percent, 0 to 200,
    /// default 100 (a plain number, not Auto; 0 is off).
    pub output_sharpening: f64,
}

/// The place the Output has in the default graph.
pub const OUTPUT_POS: NodePos = [1.0, 0.0];

impl OutputNode {
    /// An Output at its defaults with this look.
    #[must_use]
    pub fn with_look(look: LookId) -> Self {
        Self {
            pos: OUTPUT_POS,
            look,
            look_intensity: 100.0,
            output_sharpening: OUTPUT_SHARPENING_DEFAULT,
        }
    }
}

impl Default for OutputNode {
    fn default() -> Self {
        Self::with_look(LookId::neutral())
    }
}

/// A corrector's trim (`docs/algorithms/06` section 2): an exposure trim in stops and a
/// temperature and tint trim. Rendered from N-03.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NodeTrim {
    /// Stops, -4 to 4.
    pub exposure_ev: f64,
    /// -100 to 100.
    pub temperature: f64,
    /// -100 to 100.
    pub tint: f64,
}

/// A corrector's grade: the settings that were top-level fields before the settings became a graph,
/// under the same names and with the same types. Auto belongs to Main: in another corrector Auto
/// reads as a manual 0 ([`NodeGrade::neutral`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NodeGrade {
    /// Light › Exposure and Look › Tone: Auto (0, or Apple Auto's contrast, `docs/19` section 5) or a
    /// manual value, -100 to 100: a tone curve on the luma about mid grey that keeps black and the
    /// top code value (docs/02 C9). A plain number reads as Manual with that value, and 0 as Auto.
    #[serde(deserialize_with = "percent_document")]
    pub contrast: ContrastSetting,
    /// -100 to 100.
    pub brightness: f64,
    /// Light › High dynamic range: Auto (0, or Apple Auto's highlights) or a manual value, -100 to
    /// 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub highlights: PercentSetting,
    /// Auto (0, or Apple Auto's shadows) or a manual value, -100 to 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub shadows: PercentSetting,
    /// -100 to 100.
    pub white: f64,
    /// Auto (0, or Apple Auto's black level) or a manual value, -100 to 100. Read like `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub black: PercentSetting,
    /// The three colour-balance wheels; the shadows and highlights wheels are Auto (`[0, 0]`, or the
    /// colour Apple Auto gave that zone) or a point.
    pub color_balance: ColorBalance,
    /// Colour › Colour balance and Look › Tone: Auto (0, or Apple Auto's saturation) or a manual
    /// value, -100 to 100; the chroma scaled about the luminance, which stays (docs/02 C9). Read like
    /// `contrast`.
    #[serde(deserialize_with = "percent_document")]
    pub saturation: SaturationSetting,
    /// Colour › Colour warp: the pins of stage C9b (docs/02); none by default.
    pub color_warp: ColorWarp,
    /// Luma, red, green and blue curves.
    pub curve: Curves,
}

impl Default for NodeGrade {
    /// Main's grade: Auto where there is Auto.
    fn default() -> Self {
        Self {
            contrast: AutoOr::Auto,
            brightness: 0.0,
            highlights: AutoOr::Auto,
            shadows: AutoOr::Auto,
            white: 0.0,
            black: AutoOr::Auto,
            color_balance: ColorBalance::default(),
            saturation: AutoOr::Auto,
            color_warp: ColorWarp::default(),
            curve: Curves::default(),
        }
    }
}

impl NodeGrade {
    /// The grade of a corrector that is not Main: a manual 0 everywhere there could be Auto.
    #[must_use]
    pub fn neutral() -> Self {
        let mut g = Self::default();
        g.make_manual();
        g
    }

    /// Reads Auto as the neutral value (a manual 0, a wheel at the centre): the rule for every
    /// corrector but Main (`docs/23` section 3.2). Not clamping, so nothing reports it.
    pub fn make_manual(&mut self) {
        let zero = || AutoOr::Manual(PercentValue { value: 0.0 });
        for setting in [
            &mut self.contrast,
            &mut self.highlights,
            &mut self.shadows,
            &mut self.black,
            &mut self.saturation,
        ] {
            if setting.is_auto() {
                *setting = zero();
            }
        }
        for wheel in [
            &mut self.color_balance.shadows,
            &mut self.color_balance.highlights,
        ] {
            if wheel.is_auto() {
                *wheel = BalanceSetting::manual_point([0.0, 0.0]);
            }
        }
    }
}

/// A corrector node, Main included (`docs/23` section 3.2). Rendered from N-03; N-02 reads, checks,
/// patches, copies and writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CorrectorNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// The trims.
    pub trim: NodeTrim,
    /// The grade.
    pub grade: NodeGrade,
    /// The masks.
    pub masks: MaskSettings,
}

impl Default for CorrectorNode {
    /// A new corrector other than Main: manual 0 everywhere, no trim, no masks.
    fn default() -> Self {
        Self {
            name: DEFAULT_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            trim: NodeTrim::default(),
            grade: NodeGrade::neutral(),
            masks: MaskSettings::default(),
        }
    }
}

impl CorrectorNode {
    /// Main as the default graph has it: Auto where there is Auto.
    #[must_use]
    pub fn main() -> Self {
        Self {
            name: MAIN_NAME.to_owned(),
            grade: NodeGrade::default(),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------------------------

/// The graph a photo's settings are (`docs/23` section 3.1): the Input, Main, the other correctors,
/// the Output and the links. The three fixed nodes always exist; `validated` checks the rest.
///
/// JSON: `{ "items": [ … ], "links": [ … ] }`, the items in the order Input, correctors by id,
/// Output, the links in the order of their ends.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeGraph {
    /// The Input.
    pub input: InputNode,
    /// Main, corrector 1.
    pub main: CorrectorNode,
    /// Every other corrector by id.
    pub extra: BTreeMap<u32, CorrectorNode>,
    /// The Film nodes by id. They share the number space of the correctors (an id is one or the other).
    pub films: BTreeMap<u32, FilmNode>,
    /// The Output.
    pub output: OutputNode,
    /// The links.
    pub links: Vec<NodeLink>,
}

impl NodeGraph {
    /// The default graph: Input → Main → Output with every field at its default and `look` on the
    /// Output.
    #[must_use]
    pub fn with_look(look: LookId) -> Self {
        Self {
            input: InputNode::default(),
            main: CorrectorNode::main(),
            extra: BTreeMap::new(),
            films: BTreeMap::new(),
            output: OutputNode::with_look(look),
            links: Self::default_links(),
        }
    }

    /// The links of the default graph: Input → Main → Output, pictures.
    #[must_use]
    pub fn default_links() -> Vec<NodeLink> {
        vec![
            NodeLink::picture(NodeId::Input, NodeId::MAIN),
            NodeLink::picture(NodeId::MAIN, NodeId::Output),
        ]
    }

    /// The corrector with this id (Main is 1).
    #[must_use]
    pub fn corrector(&self, id: u32) -> Option<&CorrectorNode> {
        if id == MAIN_ID {
            Some(&self.main)
        } else {
            self.extra.get(&id)
        }
    }

    /// The corrector with this id, to change.
    pub fn corrector_mut(&mut self, id: u32) -> Option<&mut CorrectorNode> {
        if id == MAIN_ID {
            Some(&mut self.main)
        } else {
            self.extra.get_mut(&id)
        }
    }

    /// Every corrector by id, Main first.
    pub fn correctors(&self) -> impl Iterator<Item = (u32, &CorrectorNode)> {
        std::iter::once((MAIN_ID, &self.main)).chain(self.extra.iter().map(|(id, n)| (*id, n)))
    }

    /// The number of correctors, Main included.
    #[must_use]
    pub fn corrector_count(&self) -> usize {
        1 + self.extra.len()
    }

    /// The Film node with this id.
    #[must_use]
    pub fn film(&self, id: u32) -> Option<&FilmNode> {
        self.films.get(&id)
    }

    /// The Film node with this id, to change.
    pub fn film_mut(&mut self, id: u32) -> Option<&mut FilmNode> {
        self.films.get_mut(&id)
    }

    /// The mask rows of a numbered node, whatever its kind (a corrector's or a Film node's).
    #[must_use]
    pub fn masks_of(&self, id: u32) -> Option<&MaskSettings> {
        self.corrector(id)
            .map(|c| &c.masks)
            .or_else(|| self.films.get(&id).map(|f| &f.masks))
    }

    /// The mask rows of a numbered node, to change.
    pub fn masks_of_mut(&mut self, id: u32) -> Option<&mut MaskSettings> {
        if id == MAIN_ID {
            return Some(&mut self.main.masks);
        }
        if self.extra.contains_key(&id) {
            return self.extra.get_mut(&id).map(|c| &mut c.masks);
        }
        self.films.get_mut(&id).map(|f| &mut f.masks)
    }

    /// The masks of every numbered node, by id.
    pub fn all_masks(&self) -> impl Iterator<Item = (u32, &MaskSettings)> {
        self.correctors()
            .map(|(id, c)| (id, &c.masks))
            .chain(self.films.iter().map(|(id, f)| (*id, &f.masks)))
    }

    /// The number of numbered nodes: the correctors (Main included) and the Film nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.corrector_count() + self.films.len()
    }

    /// The numbered nodes with their kinds, by id.
    #[must_use]
    pub fn numbered(&self) -> Vec<(u32, NodeKind)> {
        let mut v: Vec<(u32, NodeKind)> = self
            .correctors()
            .map(|(id, _)| (id, NodeKind::Corrector))
            .chain(self.films.keys().map(|id| (*id, NodeKind::Film)))
            .collect();
        v.sort_by_key(|(id, _)| *id);
        v
    }

    /// What kind of node `id` is in this graph, or `None` when the graph has no such node.
    #[must_use]
    pub fn kind_of(&self, id: NodeId) -> Option<NodeKind> {
        match id {
            NodeId::Input => Some(NodeKind::Input),
            NodeId::Output => Some(NodeKind::Output),
            NodeId::Corrector(n) if self.films.contains_key(&n) => Some(NodeKind::Film),
            NodeId::Corrector(n) => self.corrector(n).map(|_| NodeKind::Corrector),
        }
    }

    /// The ids of every node in written order: the Input, the numbered nodes by id, the Output.
    #[must_use]
    pub fn ids(&self) -> Vec<NodeId> {
        std::iter::once(NodeId::Input)
            .chain(
                self.numbered()
                    .into_iter()
                    .map(|(id, _)| NodeId::Corrector(id)),
            )
            .chain(std::iter::once(NodeId::Output))
            .collect()
    }

    /// Whether the graph has this node.
    #[must_use]
    pub fn has(&self, id: NodeId) -> bool {
        self.kind_of(id).is_some()
    }

    /// A node's place in the written order (its index in `items`).
    #[must_use]
    pub fn item_index(&self, id: NodeId) -> usize {
        match id {
            NodeId::Input => 0,
            NodeId::Corrector(n) => {
                1 + self
                    .numbered()
                    .into_iter()
                    .take_while(|(i, _)| *i < n)
                    .count()
            }
            NodeId::Output => 1 + self.node_count(),
        }
    }

    /// The position of a node.
    #[must_use]
    pub fn pos(&self, id: NodeId) -> Option<NodePos> {
        match id {
            NodeId::Input => Some(self.input.pos),
            NodeId::Output => Some(self.output.pos),
            NodeId::Corrector(n) => self
                .corrector(n)
                .map(|c| c.pos)
                .or_else(|| self.film(n).map(|f| f.pos)),
        }
    }

    /// Puts the nodes whose place was not given (`unplaced`) in a row by their number, at y = 0:
    /// the Input first, the correctors in id order from x = 0, the Output last.
    pub(crate) fn place(&mut self, unplaced: &BTreeSet<NodeId>) {
        if unplaced.contains(&NodeId::Input) {
            self.input.pos = INPUT_POS;
        }
        let count = self.node_count();
        for (k, (id, _)) in self.numbered().into_iter().enumerate() {
            if !unplaced.contains(&NodeId::Corrector(id)) {
                continue;
            }
            if let Some(c) = self.corrector_mut(id) {
                c.pos = [k as f64, 0.0];
            } else if let Some(f) = self.film_mut(id) {
                f.pos = [k as f64, 0.0];
            }
        }
        if unplaced.contains(&NodeId::Output) {
            self.output.pos = [count as f64, 0.0];
        }
    }

    /// Sorts the links into the written order: by the ends of each, the from-node in the order of
    /// ids, then its port, then the to-node and its port.
    pub(crate) fn sort_links(&mut self) {
        self.links.sort();
    }

    // -----------------------------------------------------------------------------------------
    // Rules
    // -----------------------------------------------------------------------------------------

    /// Checks the rules of `docs/23` section 3.5 that concern the links and the number of nodes.
    ///
    /// # Errors
    ///
    /// The first rule that is broken, as a [`GraphError`].
    pub fn check(&self) -> Result<(), GraphError> {
        if self.node_count() > MAX_CORRECTORS {
            return Err(GraphError::TooManyNodes);
        }
        let ports_out = |n: NodeId| match n {
            NodeId::Input => &[Port::Picture][..],
            NodeId::Corrector(_) => &[Port::Picture, Port::Key][..],
            NodeId::Output => &[][..],
        };
        let ports_in = |n: NodeId| match n {
            NodeId::Input => &[][..],
            NodeId::Corrector(_) => &[Port::Picture, Port::Key][..],
            NodeId::Output => &[Port::Picture][..],
        };
        for link in &self.links {
            for end in [&link.from, &link.to] {
                if !self.has(end.node) {
                    return Err(GraphError::UnknownNode(end.node.to_string()));
                }
            }
            if matches!(link.from.node, NodeId::Output) || matches!(link.to.node, NodeId::Input) {
                return Err(GraphError::LinkDirection);
            }
            if !ports_out(link.from.node).contains(&link.from.port) {
                return Err(GraphError::UnknownPort {
                    node: link.from.node.sentence(),
                    port: link.from.port.to_string(),
                });
            }
            if !ports_in(link.to.node).contains(&link.to.port) {
                return Err(GraphError::UnknownPort {
                    node: link.to.node.sentence(),
                    port: link.to.port.to_string(),
                });
            }
            if link.from.port != link.to.port {
                return Err(GraphError::LinkPortTypes);
            }
        }
        let mut into: BTreeMap<(NodeId, &Port), usize> = BTreeMap::new();
        for link in &self.links {
            let n = into.entry((link.to.node, &link.to.port)).or_insert(0);
            *n += 1;
            if *n > 1 {
                return Err(GraphError::TwoLinks {
                    node: link.to.node.sentence(),
                    port: link.to.port.to_string(),
                });
            }
        }
        if !into.contains_key(&(NodeId::Output, &Port::Picture)) {
            return Err(GraphError::OutputLink);
        }
        let main_from_input = self.links.iter().any(|l| {
            l.to == NodeRef::picture(NodeId::MAIN) && l.from == NodeRef::picture(NodeId::Input)
        });
        if !main_from_input {
            return Err(GraphError::MainInput);
        }
        if self.has_cycle() {
            return Err(GraphError::Cycle);
        }
        Ok(())
    }

    /// Whether the links, picture and key together, make a circle.
    fn has_cycle(&self) -> bool {
        let mut next: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for link in &self.links {
            next.entry(link.from.node).or_default().push(link.to.node);
        }
        // Kahn's algorithm: peel the nodes that nothing points at; what is left is on a circle.
        let mut indegree: BTreeMap<NodeId, usize> =
            self.ids().into_iter().map(|n| (n, 0)).collect();
        for to in next.values().flatten() {
            *indegree.entry(*to).or_insert(0) += 1;
        }
        let mut ready: Vec<NodeId> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(n, _)| *n)
            .collect();
        let mut seen = 0;
        while let Some(n) = ready.pop() {
            seen += 1;
            for to in next.get(&n).into_iter().flatten() {
                if let Some(d) = indegree.get_mut(to) {
                    *d -= 1;
                    if *d == 0 {
                        ready.push(*to);
                    }
                }
            }
        }
        seen < indegree.len()
    }
}

impl Serialize for NodeGraph {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        fn item<T: Serialize, E: serde::ser::Error>(
            id: NodeId,
            kind: NodeKind,
            node: &T,
        ) -> Result<Value, E> {
            let mut v = serde_json::to_value(node).map_err(E::custom)?;
            if let Value::Object(map) = &mut v {
                map.insert(
                    "id".to_owned(),
                    serde_json::to_value(id).map_err(E::custom)?,
                );
                map.insert("kind".to_owned(), Value::String(kind.name().to_owned()));
            }
            Ok(v)
        }
        let mut items = Vec::with_capacity(self.node_count() + 2);
        items.push(item(NodeId::Input, NodeKind::Input, &self.input)?);
        for (id, kind) in self.numbered() {
            match kind {
                NodeKind::Film => {
                    if let Some(f) = self.film(id) {
                        items.push(item(NodeId::Corrector(id), NodeKind::Film, f)?);
                    }
                }
                _ => {
                    if let Some(c) = self.corrector(id) {
                        items.push(item(NodeId::Corrector(id), NodeKind::Corrector, c)?);
                    }
                }
            }
        }
        items.push(item(NodeId::Output, NodeKind::Output, &self.output)?);
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("items", &items)?;
        map.serialize_entry("links", &self.links)?;
        map.end()
    }
}

// ---------------------------------------------------------------------------------------------
// Reading a document over the default graph
// ---------------------------------------------------------------------------------------------

/// The fields every node has besides its kind's own.
const COMMON_FIELDS: [&str; 3] = ["id", "kind", "pos"];

/// The fields of each kind, for the unknown-field rule.
pub(crate) fn node_fields(kind: NodeKind) -> &'static [&'static str] {
    match kind {
        NodeKind::Input => &[
            "exposure",
            "light_source",
            "white_balance",
            "white_pick",
            "rolloff",
            "decode",
            "noise_reduction",
            "sharpen",
            "lens",
            "crop",
            "orientation",
        ],
        NodeKind::Output => &["look", "look_intensity", "output_sharpening"],
        NodeKind::Corrector => &["name", "bypass", "trim", "grade", "masks"],
        NodeKind::Film => &[
            "name", "bypass", "film", "print", "halation", "grain", "strength", "masks",
        ],
    }
}

fn parse_error(e: serde_json::Error) -> SettingsError {
    SettingsError::Parse(e.to_string())
}

impl NodeGraph {
    /// Reads a graph document (`{ "items": […], "links": […] }`) over the default graph that has
    /// `default_look` on its Output. A listed node's missing fields take their defaults, the fixed
    /// three need not be listed, and `links` left out are the default links.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Graph`] for a node of a kind this version does not know, two nodes with
    /// one id, a field the node's kind does not have or a new node without a `kind`;
    /// [`SettingsError::Parse`] for anything serde refuses.
    pub fn read(value: Value, default_look: &LookId) -> Result<Self, SettingsError> {
        let Value::Object(mut doc) = value else {
            return Err(SettingsError::Parse(
                "`nodes` must be an object with `items` and `links`".to_owned(),
            ));
        };
        let items = doc.remove("items");
        let links = doc.remove("links");
        if let Some(key) = doc.keys().next() {
            return Err(SettingsError::Parse(format!(
                "unknown field `{key}` in `nodes`, expected `items` or `links`"
            )));
        }
        let mut graph = Self::with_look(default_look.clone());
        // The fixed three keep the default graph's places unless the document lists them.
        let mut unplaced: BTreeSet<NodeId> = BTreeSet::new();
        let mut placed: BTreeSet<NodeId> = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let items = match items {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a,
            Some(_) => {
                return Err(SettingsError::Parse(
                    "`items` must be a list of nodes".to_owned(),
                ));
            }
        };
        for item in items {
            let Value::Object(mut map) = item else {
                return Err(SettingsError::Parse("a node must be an object".to_owned()));
            };
            let id: NodeId = match map.remove("id") {
                Some(v) => serde_json::from_value(v).map_err(parse_error)?,
                None => {
                    return Err(SettingsError::Parse(
                        "a node needs an `id` (“input”, “output” or a number)".to_owned(),
                    ));
                }
            };
            if !seen.insert(id) {
                return Err(GraphError::DuplicateId(id.to_string()).into());
            }
            let given = match map.remove("kind") {
                Some(Value::String(k)) => Some(NodeKind::parse(&k)?),
                Some(other) => {
                    return Err(SettingsError::Parse(format!(
                        "a node's kind must be a word, not {other}"
                    )));
                }
                None => None,
            };
            let kind = match (id, given) {
                (NodeId::Input, None | Some(NodeKind::Input)) => NodeKind::Input,
                (NodeId::Output, None | Some(NodeKind::Output)) => NodeKind::Output,
                (NodeId::Corrector(_), Some(NodeKind::Corrector)) => NodeKind::Corrector,
                (NodeId::Corrector(n), Some(NodeKind::Film)) if n != MAIN_ID => NodeKind::Film,
                (NodeId::Corrector(n), None) if n == MAIN_ID => NodeKind::Corrector,
                (NodeId::Corrector(_), None) => {
                    return Err(GraphError::NeedsKind(id.to_string()).into());
                }
                _ => return Err(GraphError::MissingFixed.into()),
            };
            for key in map.keys() {
                if !COMMON_FIELDS.contains(&key.as_str())
                    && !node_fields(kind).contains(&key.as_str())
                    && is_field_of_some_kind(key)
                {
                    return Err(GraphError::UnknownField {
                        kind: kind.sentence(),
                        field: key.clone(),
                    }
                    .into());
                }
            }
            if map.contains_key("pos") {
                placed.insert(id);
            } else {
                unplaced.insert(id);
            }
            match id {
                NodeId::Input => graph.input = read_node(map)?,
                NodeId::Output => {
                    let named = map.contains_key("look");
                    let mut out: OutputNode = read_node(map)?;
                    if !named {
                        out.look = default_look.clone();
                    }
                    graph.output = out;
                }
                NodeId::Corrector(n) if kind == NodeKind::Film => {
                    let named = map.contains_key("name");
                    let mut node: FilmNode = read_node(map)?;
                    if !named {
                        node.name = DEFAULT_FILM_NAME.to_owned();
                    }
                    graph.films.insert(n, node);
                }
                NodeId::Corrector(n) => {
                    let named = map.contains_key("name");
                    let graded = map.contains_key("grade");
                    let mut node: CorrectorNode = read_node(map)?;
                    if n == MAIN_ID && !graded {
                        // Main's missing grade is Auto where there is Auto (a new corrector's own
                        // default is a manual 0, `docs/23` section 3.2).
                        node.grade = NodeGrade::default();
                    }
                    if !named {
                        node.name = if n == MAIN_ID {
                            MAIN_NAME
                        } else {
                            DEFAULT_NAME
                        }
                        .to_owned();
                    }
                    if n == MAIN_ID {
                        graph.main = node;
                    } else {
                        graph.extra.insert(n, node);
                    }
                }
            }
        }
        if let Some(v) = links {
            graph.links = serde_json::from_value(v).map_err(parse_error)?;
        }
        if (!graph.extra.is_empty() || !graph.films.is_empty()) && !placed.contains(&NodeId::Output)
        {
            // Correctors were added: the Output goes to the end of the row.
            unplaced.insert(NodeId::Output);
        }
        graph.place(&unplaced);
        Ok(graph)
    }
}

/// Whether `key` is a field of some node kind (so that a known field on the wrong kind is told
/// from a typo, which serde reports with its own words).
fn is_field_of_some_kind(key: &str) -> bool {
    [
        NodeKind::Input,
        NodeKind::Output,
        NodeKind::Corrector,
        NodeKind::Film,
    ]
    .iter()
    .any(|k| node_fields(*k).contains(&key))
}

/// Reads the fields of one node (everything but `id` and `kind`) over its defaults.
fn read_node<T: serde::de::DeserializeOwned>(map: Map<String, Value>) -> Result<T, SettingsError> {
    serde_json::from_value(Value::Object(map)).map_err(parse_error)
}

impl<'de> Deserialize<'de> for NodeGraph {
    /// A graph read over the default graph with Neutral on the Output; sessions and `ym` read
    /// documents through [`DevelopSettings::from_value`](super::DevelopSettings::from_value), which
    /// has the user's default look.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        // A `Parse` error already says "The settings can't be read:" where the document is read; the inner text
        // alone goes up, so the sentence is not said twice.
        Self::read(value, &LookId::neutral()).map_err(|e| match e {
            SettingsError::Parse(text) => de::Error::custom(text),
            other => de::Error::custom(other),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Checking and clamping (`docs/23` section 3.5)
// ---------------------------------------------------------------------------------------------

/// Collects what clamping changed.
#[derive(Default)]
pub(crate) struct Sanitiser {
    pub(crate) clamped: Vec<FieldRef>,
}

impl Sanitiser {
    fn note(&mut self, node: NodeId, field: SettingsField, changed: bool) {
        let r = FieldRef { node, field };
        if changed && !self.clamped.contains(&r) {
            self.clamped.push(r);
        }
    }
}

use super::{
    BalanceValue, EXPOSURE_RANGE_EV, KELVIN_RANGE_K, LENS_AMOUNT_RANGE, LOOK_INTENSITY_RANGE,
    NOISE_REDUCTION_RANGE, OUTPUT_SHARPENING_RANGE, PERCENT_RANGE, ROLLOFF_RANGE,
    SHARPEN_AMOUNT_RANGE, SHARPEN_RADIUS_RANGE_PX, TINT_RANGE, clamp_to, clamp_to_disc,
    normalise_crop, normalise_curve, normalise_warp,
};

fn clamp_pos(
    s: &mut Sanitiser,
    node: NodeId,
    prefix: &str,
    pos: &mut NodePos,
) -> Result<(), SettingsError> {
    let mut changed = false;
    for (i, v) in pos.iter_mut().enumerate() {
        let (x, c) = clamp_to(&format!("{prefix}.pos[{i}]"), *v, POS_RANGE)?;
        *v = x;
        changed |= c;
    }
    // A card moved off the canvas is clamped to its edge; that is not a setting in range.
    s.note(node, SettingsField::Pos, changed);
    Ok(())
}

impl InputNode {
    fn sanitise(&mut self, s: &mut Sanitiser, prefix: &str) -> Result<(), SettingsError> {
        let node = NodeId::Input;
        clamp_pos(s, node, prefix, &mut self.pos)?;
        if let AutoOr::Manual(m) = &mut self.exposure {
            let (v, c) = clamp_to(&format!("{prefix}.exposure"), m.ev, EXPOSURE_RANGE_EV)?;
            m.ev = v;
            s.note(node, SettingsField::Exposure, c);
        }
        if let AutoOr::Manual(m) = &mut self.white_balance {
            let (k, ck) = clamp_to(
                &format!("{prefix}.white_balance.kelvin"),
                m.kelvin,
                KELVIN_RANGE_K,
            )?;
            let (t, ct) = clamp_to(&format!("{prefix}.white_balance.tint"), m.tint, TINT_RANGE)?;
            (m.kelvin, m.tint) = (k, t);
            s.note(node, SettingsField::WhiteBalance, ck || ct);
        }
        if let AutoOr::Manual(m) = &mut self.sharpen {
            let (a, ca) = clamp_to(
                &format!("{prefix}.sharpen.amount"),
                m.amount,
                SHARPEN_AMOUNT_RANGE,
            )?;
            let (r, cr) = clamp_to(
                &format!("{prefix}.sharpen.radius_px"),
                m.radius_px,
                SHARPEN_RADIUS_RANGE_PX,
            )?;
            (m.amount, m.radius_px) = (a, r);
            s.note(node, SettingsField::Sharpen, ca || cr);
        }
        if let AutoOr::Manual(m) = &mut self.noise_reduction {
            let mut changed = false;
            for (name, v) in [
                ("luminance", &mut m.luminance),
                ("color", &mut m.color),
                ("detail", &mut m.detail),
            ] {
                let (x, c) = clamp_to(
                    &format!("{prefix}.noise_reduction.{name}"),
                    *v,
                    NOISE_REDUCTION_RANGE,
                )?;
                *v = x;
                changed |= c;
            }
            s.note(node, SettingsField::NoiseReduction, changed);
        }
        let mut changed = false;
        for (name, v) in [
            ("highlights", &mut self.rolloff.highlights),
            ("low_light", &mut self.rolloff.low_light),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.rolloff.{name}"), *v, ROLLOFF_RANGE)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Rolloff, changed);
        let mut changed = false;
        for (name, v) in [
            ("flares", &mut self.lens.flares),
            ("edges", &mut self.lens.edges),
            ("anamorphic", &mut self.lens.anamorphic),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.lens.{name}"), *v, LENS_AMOUNT_RANGE)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Lens, changed);
        if let Some(crop) = &mut self.crop {
            s.note(node, SettingsField::Crop, normalise_crop(crop, prefix)?);
        }
        if let Some(p) = &mut self.white_pick {
            let mut changed = false;
            for v in p.iter_mut() {
                let (x, c) = clamp_to(&format!("{prefix}.white_pick"), *v, (0.0, 1.0))?;
                *v = x;
                changed |= c;
            }
            s.note(node, SettingsField::WhitePick, changed);
        }
        Ok(())
    }
}

impl OutputNode {
    fn sanitise(&mut self, s: &mut Sanitiser, prefix: &str) -> Result<(), SettingsError> {
        let node = NodeId::Output;
        clamp_pos(s, node, prefix, &mut self.pos)?;
        let (v, c) = clamp_to(
            &format!("{prefix}.look_intensity"),
            self.look_intensity,
            LOOK_INTENSITY_RANGE,
        )?;
        self.look_intensity = v;
        s.note(node, SettingsField::LookIntensity, c);
        let (v, c) = clamp_to(
            &format!("{prefix}.output_sharpening"),
            self.output_sharpening,
            OUTPUT_SHARPENING_RANGE,
        )?;
        self.output_sharpening = v;
        s.note(node, SettingsField::OutputSharpening, c);
        Ok(())
    }
}

impl NodeTrim {
    fn sanitise(
        &mut self,
        s: &mut Sanitiser,
        node: NodeId,
        prefix: &str,
    ) -> Result<(), SettingsError> {
        let mut changed = false;
        for (name, v, range) in [
            ("exposure_ev", &mut self.exposure_ev, TRIM_EXPOSURE_RANGE_EV),
            ("temperature", &mut self.temperature, TRIM_RANGE),
            ("tint", &mut self.tint, TRIM_RANGE),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.trim.{name}"), *v, range)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Trim, changed);
        Ok(())
    }
}

impl NodeGrade {
    fn sanitise(
        &mut self,
        s: &mut Sanitiser,
        node: NodeId,
        prefix: &str,
    ) -> Result<(), SettingsError> {
        let prefix = format!("{prefix}.grade");
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
                let (v, c) = clamp_to(&format!("{prefix}.{name}"), m.value, PERCENT_RANGE)?;
                m.value = v;
                s.note(node, field, c);
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
            let (v, c) = clamp_to(&format!("{prefix}.{name}"), *value, PERCENT_RANGE)?;
            *value = v;
            s.note(node, field, c);
        }
        let mut changed = false;
        let cb = &mut self.color_balance;
        let mut points: Vec<(&'static str, &mut ColorBalancePoint)> =
            vec![("midtones", &mut cb.midtones)];
        if let AutoOr::Manual(BalanceValue { point }) = &mut cb.shadows {
            points.push(("shadows", point));
        }
        if let AutoOr::Manual(BalanceValue { point }) = &mut cb.highlights {
            points.push(("highlights", point));
        }
        for (name, point) in points {
            let (p, c) = clamp_to_disc(&format!("{prefix}.color_balance.{name}"), *point)?;
            *point = p;
            changed |= c;
        }
        s.note(node, SettingsField::ColorBalance, changed);
        s.note(
            node,
            SettingsField::ColorWarp,
            normalise_warp(&mut self.color_warp, &prefix)?,
        );
        let mut changed = false;
        for (channel, points) in [
            ("luma", &mut self.curve.luma),
            ("red", &mut self.curve.red),
            ("green", &mut self.curve.green),
            ("blue", &mut self.curve.blue),
        ] {
            changed |= normalise_curve(channel, points, &prefix)?;
        }
        s.note(node, SettingsField::Curve, changed);
        Ok(())
    }
}

impl CorrectorNode {
    fn sanitise(&mut self, s: &mut Sanitiser, id: u32, prefix: &str) -> Result<(), SettingsError> {
        let node = NodeId::Corrector(id);
        // The name is trimmed and cut, and an empty one reads as the default.
        let trimmed: String = self.name.trim().chars().take(MAX_NAME_CHARS).collect();
        let trimmed = trimmed.trim().to_owned();
        self.name = if trimmed.is_empty() {
            if id == MAIN_ID {
                MAIN_NAME
            } else {
                DEFAULT_NAME
            }
            .to_owned()
        } else {
            trimmed
        };
        clamp_pos(s, node, prefix, &mut self.pos)?;
        self.trim.sanitise(s, node, prefix)?;
        if id != MAIN_ID {
            // Auto belongs to Main (`docs/23` section 3.2): here it is a manual 0, silently.
            self.grade.make_manual();
        }
        self.grade.sanitise(s, node, prefix)?;
        let changed = self.masks.sanitise(&format!("{prefix}.masks"))?;
        s.note(node, SettingsField::Masks, changed);
        Ok(())
    }
}

impl FilmNode {
    fn sanitise(&mut self, s: &mut Sanitiser, id: u32, prefix: &str) -> Result<(), SettingsError> {
        let node = NodeId::Corrector(id);
        let trimmed: String = self.name.trim().chars().take(MAX_NAME_CHARS).collect();
        let trimmed = trimmed.trim().to_owned();
        self.name = if trimmed.is_empty() {
            DEFAULT_FILM_NAME.to_owned()
        } else {
            trimmed
        };
        clamp_pos(s, node, prefix, &mut self.pos)?;
        // The stock and the paper must be in the data (`docs/24` section 4.5). A slide takes no paper: it reads
        // as `null`, silently, like an Auto in a corrector other than Main.
        if let Some(c) = crate::film::catalogue() {
            c.check_stock(&self.film.stock)
                .map_err(SettingsError::Film)?;
            if c.is_slide(&self.film.stock) {
                self.print.paper = None;
            } else if let Some(p) = &self.print.paper {
                c.check_paper(p).map_err(SettingsError::Film)?;
            }
        }
        let (v, c) = clamp_to(
            &format!("{prefix}.film.exposure_ev"),
            self.film.exposure_ev,
            FILM_EXPOSURE_RANGE_EV,
        )?;
        self.film.exposure_ev = v;
        s.note(node, SettingsField::Film, c);
        let mut changed = false;
        for (name, v, range) in [
            (
                "exposure_ev",
                &mut self.print.exposure_ev,
                PRINT_EXPOSURE_RANGE_EV,
            ),
            ("lights.c", &mut self.print.lights.c, LIGHTS_RANGE),
            ("lights.m", &mut self.print.lights.m, LIGHTS_RANGE),
            ("lights.y", &mut self.print.lights.y, LIGHTS_RANGE),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.print.{name}"), *v, range)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Print, changed);
        let mut changed = false;
        for (name, v, range) in [
            ("amount", &mut self.halation.amount, HALATION_AMOUNT_RANGE),
            ("size", &mut self.halation.size, SIZE_RANGE),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.halation.{name}"), *v, range)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Halation, changed);
        let mut changed = false;
        for (name, v, range) in [
            ("amount", &mut self.grain.amount, GRAIN_AMOUNT_RANGE),
            ("size", &mut self.grain.size, SIZE_RANGE),
        ] {
            let (x, c) = clamp_to(&format!("{prefix}.grain.{name}"), *v, range)?;
            *v = x;
            changed |= c;
        }
        s.note(node, SettingsField::Grain, changed);
        let (v, c) = clamp_to(&format!("{prefix}.strength"), self.strength, STRENGTH_RANGE)?;
        self.strength = v;
        s.note(node, SettingsField::Strength, c);
        let changed = self.masks.sanitise(&format!("{prefix}.masks"))?;
        s.note(node, SettingsField::Masks, changed);
        Ok(())
    }
}

impl NodeGraph {
    /// Clamps and normalises every node in place, checks the limits and the rules of the links and
    /// puts the links in written order. `clamped` collects what was changed.
    ///
    /// # Errors
    ///
    /// [`SettingsError::NotFinite`] for a number that is not a number, the errors of the curves,
    /// the crop and the masks, and [`SettingsError::Graph`] for a broken rule.
    pub(crate) fn sanitise(&mut self, s: &mut Sanitiser) -> Result<(), SettingsError> {
        if self.node_count() > MAX_CORRECTORS {
            return Err(GraphError::TooManyNodes.into());
        }
        // An id is a corrector's or a Film node's, never both.
        if let Some(id) = self
            .films
            .keys()
            .find(|id| **id == MAIN_ID || self.extra.contains_key(id))
        {
            return Err(GraphError::DuplicateId(id.to_string()).into());
        }
        self.input.sanitise(s, &format!("nodes.items[{}]", 0))?;
        for (id, _) in self.numbered() {
            let index = self.item_index(NodeId::Corrector(id));
            let prefix = format!("nodes.items[{index}]");
            if let Some(c) = self.corrector_mut(id) {
                c.sanitise(s, id, &prefix)?;
            } else if let Some(f) = self.film_mut(id) {
                f.sanitise(s, id, &prefix)?;
            }
        }
        let index = self.item_index(NodeId::Output);
        self.output.sanitise(s, &format!("nodes.items[{index}]"))?;
        self.sort_links();
        self.check()?;
        Ok(())
    }
}
