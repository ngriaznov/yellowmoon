//! The mixer nodes' settings (`docs/25-mixers.md` section 2.1): Parallel, Layer, Splitter and Combiner.
//!
//! They are numbered nodes of the graph, like a corrector or a Film node: their ids share the number space and they
//! count toward the 64 nodes. They have no masks, no key ports and no Auto. A Parallel and a Layer have the picture
//! inputs `picture.1` … `picture.N` and one `picture` output; a Splitter has a `picture` input and the outputs `red`,
//! `green` and `blue`; a Combiner has the inputs `red`, `green` and `blue` and a `picture` output.

use serde::{Deserialize, Serialize};

use super::NodePos;
use super::graph::NodeKind;

/// The most picture inputs a mixer may have.
pub const MIXER_INPUTS: u8 = 8;
/// The name a new Parallel has.
pub const DEFAULT_PARALLEL_NAME: &str = "Parallel";
/// The name a new Layer has.
pub const DEFAULT_LAYER_NAME: &str = "Layer";
/// The name a new Splitter has.
pub const DEFAULT_SPLITTER_NAME: &str = "Splitter";
/// The name a new Combiner has.
pub const DEFAULT_COMBINER_NAME: &str = "Combiner";
/// Range of a Layer's `opacity`.
pub const OPACITY_RANGE: (f64, f64) = (0.0, 1.0);

/// How a Layer blends an input over the ones below it (`docs/25` section 3.3), computed in linear light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    /// The layer itself.
    #[default]
    Normal,
    /// Below plus the layer.
    Add,
    /// Below minus the layer.
    Subtract,
    /// Below times the layer.
    Multiply,
    /// The lighter of the two.
    Lighten,
    /// The darker of the two.
    Darken,
    /// The absolute difference.
    Difference,
}

impl BlendMode {
    /// Every mode, in the order of the interface's Mode menu.
    pub const ALL: [BlendMode; 7] = [
        BlendMode::Normal,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Multiply,
        BlendMode::Lighten,
        BlendMode::Darken,
        BlendMode::Difference,
    ];

    /// The mode's JSON name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Lighten => "lighten",
            Self::Darken => "darken",
            Self::Difference => "difference",
        }
    }
}

/// A Parallel node: adds the change each of its inputs makes to the picture they started from (`docs/25` section 3.2).
/// It has no controls of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ParallelNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed Parallel outputs its base.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
}

impl Default for ParallelNode {
    fn default() -> Self {
        Self {
            name: DEFAULT_PARALLEL_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
        }
    }
}

/// A Layer node: stacks its inputs and blends each over the ones below it (`docs/25` section 3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LayerNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed Layer outputs its input 1.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// How the layers blend.
    pub mode: BlendMode,
    /// How much of each layer above input 1 shows, 0 to 1.
    pub opacity: f64,
}

impl Default for LayerNode {
    fn default() -> Self {
        Self {
            name: DEFAULT_LAYER_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            mode: BlendMode::Normal,
            opacity: 1.0,
        }
    }
}

/// A Splitter node: sends the red, green and blue of a picture to three outputs (`docs/25` section 3.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SplitterNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// The card's place.
    pub pos: NodePos,
}

impl Default for SplitterNode {
    fn default() -> Self {
        Self {
            name: DEFAULT_SPLITTER_NAME.to_owned(),
            pos: [0.0, 0.0],
        }
    }
}

/// A Combiner node: joins three pictures into one, a channel from each (`docs/25` section 3.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CombinerNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// The card's place.
    pub pos: NodePos,
}

impl Default for CombinerNode {
    fn default() -> Self {
        Self {
            name: DEFAULT_COMBINER_NAME.to_owned(),
            pos: [0.0, 0.0],
        }
    }
}

/// A node of one of the four kinds of `docs/25`. The graph holds them in one map by id
/// ([`NodeGraph::mixers`](super::NodeGraph)); the match over this enum has no wildcard arm anywhere it is made, so a new
/// kind does not compile until each place says what it does.
#[derive(Debug, Clone, PartialEq)]
pub enum MixerNode {
    /// A Parallel.
    Parallel(ParallelNode),
    /// A Layer.
    Layer(LayerNode),
    /// A Splitter.
    Splitter(SplitterNode),
    /// A Combiner.
    Combiner(CombinerNode),
}

impl MixerNode {
    /// A new node of this kind at its defaults, or `None` for a kind that is not one of the four.
    #[must_use]
    pub fn new_of(kind: NodeKind) -> Option<Self> {
        match kind {
            NodeKind::Parallel => Some(Self::Parallel(ParallelNode::default())),
            NodeKind::Layer => Some(Self::Layer(LayerNode::default())),
            NodeKind::Splitter => Some(Self::Splitter(SplitterNode::default())),
            NodeKind::Combiner => Some(Self::Combiner(CombinerNode::default())),
            NodeKind::Input
            | NodeKind::Output
            | NodeKind::Corrector
            | NodeKind::Film
            | NodeKind::FaceRefinement => None,
        }
    }

    /// The node's kind.
    #[must_use]
    pub fn kind(&self) -> NodeKind {
        match self {
            Self::Parallel(_) => NodeKind::Parallel,
            Self::Layer(_) => NodeKind::Layer,
            Self::Splitter(_) => NodeKind::Splitter,
            Self::Combiner(_) => NodeKind::Combiner,
        }
    }

    /// The card's name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Parallel(n) => &n.name,
            Self::Layer(n) => &n.name,
            Self::Splitter(n) => &n.name,
            Self::Combiner(n) => &n.name,
        }
    }

    /// The card's place.
    #[must_use]
    pub fn pos(&self) -> NodePos {
        match self {
            Self::Parallel(n) => n.pos,
            Self::Layer(n) => n.pos,
            Self::Splitter(n) => n.pos,
            Self::Combiner(n) => n.pos,
        }
    }

    /// The card's place, to change.
    pub fn pos_mut(&mut self) -> &mut NodePos {
        match self {
            Self::Parallel(n) => &mut n.pos,
            Self::Layer(n) => &mut n.pos,
            Self::Splitter(n) => &mut n.pos,
            Self::Combiner(n) => &mut n.pos,
        }
    }

    /// Whether the node is bypassed (a Splitter and a Combiner have no Bypass).
    #[must_use]
    pub fn bypass(&self) -> bool {
        match self {
            Self::Parallel(n) => n.bypass,
            Self::Layer(n) => n.bypass,
            Self::Splitter(_) | Self::Combiner(_) => false,
        }
    }

    /// The node's name, to change.
    pub fn name_mut(&mut self) -> &mut String {
        match self {
            Self::Parallel(n) => &mut n.name,
            Self::Layer(n) => &mut n.name,
            Self::Splitter(n) => &mut n.name,
            Self::Combiner(n) => &mut n.name,
        }
    }

    /// The name a new node of this kind has.
    #[must_use]
    pub fn default_name(&self) -> &'static str {
        match self {
            Self::Parallel(_) => DEFAULT_PARALLEL_NAME,
            Self::Layer(_) => DEFAULT_LAYER_NAME,
            Self::Splitter(_) => DEFAULT_SPLITTER_NAME,
            Self::Combiner(_) => DEFAULT_COMBINER_NAME,
        }
    }

    /// The node as a JSON object without its `id` and `kind`.
    ///
    /// # Errors
    ///
    /// [`serde_json::Error`] when serde cannot write the node (it always can).
    pub fn to_value(&self) -> Result<serde_json::Value, serde_json::Error> {
        match self {
            Self::Parallel(n) => serde_json::to_value(n),
            Self::Layer(n) => serde_json::to_value(n),
            Self::Splitter(n) => serde_json::to_value(n),
            Self::Combiner(n) => serde_json::to_value(n),
        }
    }

    /// Reads the node of `kind` from a JSON object without its `id` and `kind`.
    ///
    /// # Errors
    ///
    /// [`serde_json::Error`] for anything serde refuses (an unknown `mode`, a field of another kind), and for a
    /// `kind` that is not one of the four.
    pub fn from_value(
        kind: NodeKind,
        value: serde_json::Value,
    ) -> Result<Option<Self>, serde_json::Error> {
        Ok(Some(match kind {
            NodeKind::Parallel => Self::Parallel(serde_json::from_value(value)?),
            NodeKind::Layer => Self::Layer(serde_json::from_value(value)?),
            NodeKind::Splitter => Self::Splitter(serde_json::from_value(value)?),
            NodeKind::Combiner => Self::Combiner(serde_json::from_value(value)?),
            NodeKind::Input
            | NodeKind::Output
            | NodeKind::Corrector
            | NodeKind::Film
            | NodeKind::FaceRefinement => {
                return Ok(None);
            }
        }))
    }
}
