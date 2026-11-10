//! The plan: a graph of nodes compiled for the per-pixel chain (`docs/23-nodes.md` section 5.2,
//! `docs/25-mixers.md` section 4.1, `docs/algorithms/06-masks.md` section 1.3).
//!
//! The settings hold a graph (`ym-core`: the Input, the correctors and Film nodes, the mixers, the
//! Splitters and Combiners, the Output and typed links). [`compile_with`] turns what the render needs of
//! it, a [`PlanGraph`] and its [`PlanJoin`]s, into a [`Plan`]: which nodes are live, the order they run
//! in, the card number of every node, the register each value lives in and, when a plan needs more than
//! [`PLAN_REGISTERS`] pictures or [`PLAN_KEYS`] keys alive at once, the split into passes with a buffer
//! for every value that crosses from one pass to the next.
//!
//! Everything here is pure and deterministic: the same graph gives the same plan whatever order its
//! nodes were listed in, on every machine. Nothing in this module knows a colour.
//!
//! # Live and dead
//!
//! A node is **live** when the Output needs it: it is on the chain of picture links that leads back
//! from the Output to the Input, or its key output feeds (through any number of key links) the key
//! input of a live node, or it is a picture ancestor of such a node. That is the closure of the
//! Output's picture source under "my picture sources" (every picture input of a mixer, a Splitter and a
//! Combiner) and "my key source". Other nodes are dead: they are in [`Plan::cards`] (with a number,
//! dimmed) and not in the steps.
//!
//! # Order and numbers
//!
//! The order is a topological order over the picture and the key links together (a validated graph
//! has no circle), with ties broken by Main first, then the card's `pos` (x, then y), then the id. A
//! card's **number** is its place in the same order over the correctors and Film nodes, dead ones
//! included, counted from 1; a mixer, a Splitter and a Combiner take their place in the order and have
//! no number ([`Card::number`] is `None`).
//!
//! # Values, registers and passes
//!
//! A step reads the pictures it is linked from (one for a corrector, a Film node and a Splitter, up to
//! [`PLAN_INPUTS`] for a mixer, three for a Combiner), may read one key, and writes the values that a
//! later step or the result reads: its own picture (a Splitter's three), its key (the node's mask `m`)
//! and, for a corrector or Film node that feeds an input above the first of a Layer, its **graded value**,
//! the picture before its mask blend. The Input's picture is the pass's input and takes no register:
//! [`PictureRef::Input`]. Every other value lives in a **register** (pictures) or a **key slot** from the
//! step that makes it to the last step of the pass that reads it, and a step may write into the register
//! its own inputs leave, so a chain of any length needs one register. A value that a later pass reads as
//! well is also written to a **buffer** when it is made, and read from it in the later passes
//! ([`PictureRef::Buffer`]).
//!
//! The compiler fills a pass with steps for as long as it fits [`PLAN_REGISTERS`] and [`PLAN_KEYS`]
//! and starts the next one when the next step would not. **A join does not split a pass**: a mixer or a
//! Combiner is a per-pixel step that reads several registers. A Parallel of eight branches and its base
//! needs nine pictures alive, so the compiler always starts a second pass there, and the join reads some
//! of its inputs from buffers.
//!
//! # The base of a Parallel
//!
//! The base of a Parallel is the immediate dominator of its output in the graph of picture values rooted
//! at the Input's picture ([`Plan::bases`], `docs/25` section 3.2). It is a fact of the graph, found here
//! from the links alone; a step says which of its inputs *is* the base ([`JoinStep::is_base`]), so the
//! chain never compares floats to find out.
//!
//! # Stops
//!
//! [`Stop::Output`] is the whole picture. [`Stop::Picture`] ends at a node's picture (a thumbnail:
//! the picture after the node, which the caller then runs through the Output) and [`Stop::Mask`] at a
//! node's mask (the mask view). Both name any node, live or dead; the plan holds the node's ancestors
//! in the full order, so for a chain it is a prefix of the full plan. A mask stop names a corrector or a
//! Film node: the four mixer kinds have no mask.
//!
//! # A picture input with no link
//!
//! A node whose picture input has no link ([`PlanNode::picture`] is `None`, a [`JoinInput::source`] is
//! `None`) reads the Input's picture: the Input is the only picture there is. (The settings validator
//! allows such a node; the spec reads it so.)

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Pictures a pass may keep alive at once (`docs/23` section 5.2).
pub const PLAN_REGISTERS: usize = 8;
/// Keys a pass may keep alive at once.
pub const PLAN_KEYS: usize = 16;
/// Picture inputs a mixer has at most (`docs/25` section 2.2).
pub const PLAN_INPUTS: usize = 8;
/// The id of Main, the corrector that comes first among equals.
pub const MAIN_NODE: u32 = 1;

/// Where a picture comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PictureSource {
    /// The Input's picture (the LogC4 picture of C8).
    Input,
    /// The picture output of a corrector, a Film node, a mixer or a Combiner.
    Node(u32),
    /// One of the three outputs of a Splitter: the node and the channel (0 red, 1 green, 2 blue).
    Split(u32, u8),
}

impl PictureSource {
    /// The node that makes the picture; `None` for the Input.
    #[must_use]
    pub fn node(self) -> Option<u32> {
        match self {
            Self::Input => None,
            Self::Node(m) | Self::Split(m, _) => Some(m),
        }
    }
}

/// A corrector (or Film node) as the compiler sees it: its place, its picture source and its key source.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanNode {
    /// The node's id (Main is [`MAIN_NODE`]).
    pub id: u32,
    /// The card's place on the canvas, `[x, y]`, used to break ties in the order.
    pub pos: [f64; 2],
    /// Where the node's picture input is linked from; `None` when it has no link (it then reads the
    /// Input's picture).
    pub picture: Option<PictureSource>,
    /// The corrector whose key output is linked to this node's key input.
    pub key: Option<u32>,
}

/// The part of the settings' graph the render needs: the correctors with their sources and the
/// picture the Output reads. The mixers, Splitters and Combiners are [`PlanJoin`]s, given to
/// [`compile_with`] beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGraph {
    /// The correctors, in any order.
    pub nodes: Vec<PlanNode>,
    /// The picture the Output's input is linked to.
    pub output: PictureSource,
}

impl PlanGraph {
    /// The default graph: Input → Main → Output.
    #[must_use]
    pub fn single_main() -> Self {
        Self {
            nodes: vec![PlanNode {
                id: MAIN_NODE,
                pos: [0.0, 0.0],
                picture: Some(PictureSource::Input),
                key: None,
            }],
            output: PictureSource::Node(MAIN_NODE),
        }
    }
}

/// The kinds of node that join or split pictures (`docs/25-mixers.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinKind {
    /// Adds the change each input makes to the base.
    Parallel,
    /// Blends each input over the ones below it.
    Layer,
    /// Sends the red, green and blue of a picture to three outputs.
    Splitter,
    /// Joins three pictures, a channel from each.
    Combiner,
}

/// One picture input of a [`PlanJoin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinInput {
    /// Where the input is linked from; `None` reads the Input's picture.
    pub source: Option<PictureSource>,
    /// For a Layer's input above the first: the input is linked from the picture output of a corrector or
    /// Film node that is not bypassed, so the Layer reads that node's picture before its own mask blend
    /// and the node's `m` as the input's alpha (`docs/25` section 3.3). Ignored elsewhere, and when the
    /// source is not a corrector or Film node of the graph.
    pub graded: bool,
}

/// A mixer, a Splitter or a Combiner as the compiler sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanJoin {
    /// The node's id.
    pub id: u32,
    /// The card's place on the canvas, `[x, y]`.
    pub pos: [f64; 2],
    /// Which kind.
    pub kind: JoinKind,
    /// The picture inputs in port order: `picture.1` to `picture.N` of a mixer (none: one input, the Input's
    /// picture), the one `picture` of a Splitter, `red`, `green` and `blue` of a Combiner.
    pub inputs: Vec<JoinInput>,
}

/// Where a plan ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stop {
    /// The picture the Output reads.
    Output,
    /// The picture of a node (a thumbnail, once it has gone through the Output). A Splitter's is its red.
    Picture(u32),
    /// The mask `m` of a node (the mask view).
    Mask(u32),
}

/// Why a graph cannot be compiled. A graph the settings validator accepted never fails, apart from
/// a stop that names a node the graph does not have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// Two nodes have one id.
    #[error("two nodes have the id {0}")]
    DuplicateNode(u32),
    /// A link or a stop names a node the graph does not have.
    #[error("the plan names node {0}, which the graph does not have")]
    UnknownNode(u32),
    /// The links make a circle.
    #[error("the nodes are linked in a circle")]
    Cycle,
    /// A mixer has more than [`PLAN_INPUTS`] inputs.
    #[error("node {0} has more than {PLAN_INPUTS} inputs")]
    TooManyInputs(u32),
    /// A mask stop names a node that has no mask (a mixer, a Splitter or a Combiner).
    #[error("node {0} has no mask")]
    NoMask(u32),
}

/// What a card shows: the node's number in the flow and whether it is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Card {
    /// The node.
    pub id: u32,
    /// Its place in the order over the correctors and Film nodes, from 1; `None` for a mixer, a Splitter
    /// or a Combiner, which have no number.
    pub number: Option<u32>,
    /// Whether the Output needs it (dead nodes are not in the steps).
    pub live: bool,
}

/// Where a step reads its picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureRef {
    /// The Input's picture (no register).
    Input,
    /// A picture register of this pass.
    Register(u8),
    /// A buffer an earlier pass wrote.
    Buffer(u16),
}

/// Where a step reads its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRef {
    /// A key slot of this pass.
    Slot(u8),
    /// A buffer an earlier pass wrote.
    Buffer(u16),
}

/// Where a step writes its picture: a register (when a step of this pass or the result reads it),
/// a buffer (when a later pass reads it), or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PictureOut {
    /// The register, when this pass reads the value.
    pub register: Option<u8>,
    /// The buffer, when a later pass reads the value.
    pub buffer: Option<u16>,
}

/// Where a step writes its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyOut {
    /// The key slot, when this pass reads the value.
    pub slot: Option<u8>,
    /// The buffer, when a later pass reads the value.
    pub buffer: Option<u16>,
}

/// What a mixer, a Splitter or a Combiner step reads and writes beyond [`Step`]'s own fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinStep {
    /// Which kind.
    pub kind: JoinKind,
    /// The picture inputs in port order, the first [`JoinStep::count`] of them. A Layer's input above the
    /// first that has an alpha holds the linked node's graded value (its picture before its mask blend);
    /// every other input holds the picture that is linked in. Entries past `count` are
    /// [`PictureRef::Input`] and unused.
    pub inputs: [PictureRef; PLAN_INPUTS],
    /// How many inputs the step has: N for a mixer, 1 for a Splitter, 3 for a Combiner.
    pub count: u8,
    /// A Parallel's base (its immediate dominator, [`Plan::bases`]).
    pub base: Option<PictureRef>,
    /// A Parallel's inputs that are the base itself, as a bit set (bit `i` for input `i`, counted from 0):
    /// the sum skips them.
    pub is_base: u8,
    /// A Layer's alpha of each input: the key `m` of the node that feeds it, to multiply by the Layer's
    /// opacity; `None` is the opacity alone. Always `None` for the first input.
    pub alphas: [Option<KeyRef>; PLAN_INPUTS],
    /// A Splitter's red, green and blue outputs (all `None` for the other kinds, whose output is
    /// [`Step::picture_out`]).
    pub channels: [Option<PictureOut>; 3],
}

impl JoinStep {
    /// The inputs the step has.
    #[must_use]
    pub fn inputs(&self) -> &[PictureRef] {
        &self.inputs[..usize::from(self.count)]
    }

    /// Whether input `i` (from 0) is the Parallel's base itself.
    #[must_use]
    pub fn input_is_base(&self, i: usize) -> bool {
        i < PLAN_INPUTS && self.is_base & (1 << i) != 0
    }
}

/// One node run on a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// The node.
    pub node: u32,
    /// The picture it reads (for a mixer, a Splitter or a Combiner: its first input, also in
    /// [`JoinStep::inputs`]).
    pub picture: PictureRef,
    /// The key it reads (its key input), when it is linked.
    pub key: Option<KeyRef>,
    /// The picture it writes; `None` when nothing reads it (a node live only for its key, or a Splitter,
    /// which writes [`JoinStep::channels`]).
    pub picture_out: Option<PictureOut>,
    /// The key it writes (its mask `m`); `None` when nothing reads it.
    pub key_out: Option<KeyOut>,
    /// The graded value it writes (a corrector or Film node's picture before its own mask blend), when a
    /// Layer reads it.
    pub graded_out: Option<PictureOut>,
    /// What a mixer, a Splitter or a Combiner reads and writes; `None` for a corrector or Film node.
    pub join: Option<JoinStep>,
}

/// What a buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    /// A LogC4 picture (three f32 planes on the device).
    Picture,
    /// A key (one f32 plane).
    Key,
}

/// A run of steps on one pixel, with the registers and key slots they share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pass {
    /// The steps, in order.
    pub steps: Vec<Step>,
    /// How many picture registers the pass uses (at most [`PLAN_REGISTERS`]).
    pub registers: usize,
    /// How many key slots the pass uses (at most [`PLAN_KEYS`]).
    pub slots: usize,
}

/// The value a plan gives back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanResult {
    /// A picture: what the Output reads, or a node's picture ([`Stop::Picture`]).
    Picture(PictureRef),
    /// A node's mask ([`Stop::Mask`]).
    Mask(KeyRef),
}

/// A compiled graph (the module notes).
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Where the plan ends.
    pub stop: Stop,
    /// Every node with its number and whether it is live, in the order over all nodes.
    pub cards: Vec<Card>,
    /// The passes (empty when the result is the Input's picture itself).
    pub passes: Vec<Pass>,
    /// What each buffer holds, by buffer number.
    pub buffers: Vec<BufferKind>,
    /// Where the plan's value is when the last pass ends.
    pub result: PlanResult,
    /// The base of every Parallel the plan runs, by the Parallel's id (`docs/25` section 3.2).
    pub bases: BTreeMap<u32, PictureSource>,
}

impl Plan {
    /// The plan of the default graph, `[Main]` in one pass (`compile(&PlanGraph::single_main(),
    /// Stop::Output)` gives the same).
    #[must_use]
    pub fn main_only() -> Self {
        Self {
            stop: Stop::Output,
            cards: vec![Card {
                id: MAIN_NODE,
                number: Some(1),
                live: true,
            }],
            passes: vec![Pass {
                steps: vec![Step {
                    node: MAIN_NODE,
                    picture: PictureRef::Input,
                    key: None,
                    picture_out: Some(PictureOut {
                        register: Some(0),
                        buffer: None,
                    }),
                    key_out: None,
                    graded_out: None,
                    join: None,
                }],
                registers: 1,
                slots: 0,
            }],
            buffers: Vec::new(),
            result: PlanResult::Picture(PictureRef::Register(0)),
            bases: BTreeMap::new(),
        }
    }

    /// Every step of every pass, in order.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.passes.iter().flat_map(|p| p.steps.iter())
    }

    /// Whether any step is a mixer, a Splitter or a Combiner (or makes a graded value for one).
    #[must_use]
    pub fn has_joins(&self) -> bool {
        self.steps()
            .any(|s| s.join.is_some() || s.graded_out.is_some())
    }

    /// For each step (in [`Plan::steps`] order), the earlier steps whose pictures it reads, taken from the plan's graph: the
    /// step's picture inputs (every input of a mixer, a Splitter or a Combiner, a Parallel's base, a Layer's graded values),
    /// and its key inputs (a key is a node's mask, which may read that node's own input, and a Layer's alphas), followed
    /// back through registers and buffers to every step they came from. Transitive, so a step lists the far ancestors as
    /// well, in step order. The scheduler uses it to give a Film node only the Film pictures it really reads (`docs/23`
    /// section 5.12), which after a Layer are the pictures of both branches.
    #[must_use]
    pub fn picture_ancestors(&self) -> Vec<Vec<usize>> {
        // P[i]: the steps whose picture output step i's picture depends on (itself included); K[i]: the same for its key.
        let mut pic: Vec<BTreeSet<usize>> = Vec::new();
        let mut key: Vec<BTreeSet<usize>> = Vec::new();
        let mut pic_buffer: BTreeMap<u16, usize> = BTreeMap::new();
        let mut key_buffer: BTreeMap<u16, usize> = BTreeMap::new();
        let mut out = Vec::new();
        for pass in &self.passes {
            let mut register: BTreeMap<u8, usize> = BTreeMap::new();
            let mut slot: BTreeMap<u8, usize> = BTreeMap::new();
            for step in &pass.steps {
                let i = pic.len();
                let mut pictures = vec![step.picture];
                let mut keys: Vec<KeyRef> = step.key.into_iter().collect();
                if let Some(j) = &step.join {
                    pictures = j.inputs().to_vec();
                    pictures.extend(j.base);
                    keys.extend(j.alphas.iter().flatten().copied());
                }
                let mut reads: BTreeSet<usize> = BTreeSet::new();
                for r in pictures {
                    let from = match r {
                        PictureRef::Input => None,
                        PictureRef::Register(r) => register.get(&r).copied(),
                        PictureRef::Buffer(b) => pic_buffer.get(&b).copied(),
                    };
                    if let Some(p) = from {
                        reads.extend(&pic[p]);
                    }
                }
                for k in keys {
                    let from = match k {
                        KeyRef::Slot(s) => slot.get(&s).copied(),
                        KeyRef::Buffer(b) => key_buffer.get(&b).copied(),
                    };
                    if let Some(k) = from {
                        reads.extend(&key[k]);
                    }
                }
                let mut made = reads.clone();
                made.insert(i);
                out.push(reads.iter().copied().collect());
                key.push(reads);
                pic.push(made);
                let channels = step
                    .join
                    .iter()
                    .flat_map(|j| j.channels.iter().copied().flatten());
                for o in step
                    .picture_out
                    .into_iter()
                    .chain(step.graded_out)
                    .chain(channels)
                {
                    if let Some(r) = o.register {
                        register.insert(r, i);
                    }
                    if let Some(b) = o.buffer {
                        pic_buffer.insert(b, i);
                    }
                }
                if let Some(o) = step.key_out {
                    if let Some(s) = o.slot {
                        slot.insert(s, i);
                    }
                    if let Some(b) = o.buffer {
                        key_buffer.insert(b, i);
                    }
                }
            }
        }
        out
    }

    /// The number of steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.passes.iter().map(|p| p.steps.len()).sum()
    }

    /// The ids of the nodes the plan runs, in order.
    #[must_use]
    pub fn node_ids(&self) -> Vec<u32> {
        self.steps().map(|s| s.node).collect()
    }

    /// The card of a node.
    #[must_use]
    pub fn card(&self, id: u32) -> Option<&Card> {
        self.cards.iter().find(|c| c.id == id)
    }

    /// Whether the plan is Main alone, reading the Input, in one pass: the plan of the default
    /// graph.
    #[must_use]
    pub fn is_main_only(&self) -> bool {
        self.stop == Stop::Output
            && self.passes.len() == 1
            && self.passes[0].steps.len() == 1
            && self.passes[0].steps[0].node == MAIN_NODE
            && self.passes[0].steps[0].picture == PictureRef::Input
            && self.passes[0].steps[0].key.is_none()
    }
}

// ---------------------------------------------------------------------------------------------
// Compiling
// ---------------------------------------------------------------------------------------------

/// Compiles a graph of correctors into a plan that ends at `stop`: [`compile_with`] without joins.
///
/// # Errors
///
/// As [`compile_with`].
pub fn compile(graph: &PlanGraph, stop: Stop) -> Result<Plan, PlanError> {
    compile_with(graph, &[], stop)
}

/// A node as the compiler sees it, whatever its kind: its place, its picture inputs in port order, its
/// key source and, for a Layer, which inputs are graded.
#[derive(Debug, Clone)]
struct Item {
    id: u32,
    pos: [f64; 2],
    /// `None` for a corrector or Film node.
    kind: Option<JoinKind>,
    pictures: Vec<PictureSource>,
    graded: Vec<bool>,
    key: Option<u32>,
}

impl Item {
    fn corrector(node: &PlanNode) -> Self {
        Self {
            id: node.id,
            pos: node.pos,
            kind: None,
            pictures: vec![node.picture.unwrap_or(PictureSource::Input)],
            graded: vec![false],
            key: node.key,
        }
    }

    fn join(join: &PlanJoin) -> Result<Self, PlanError> {
        let (mut sources, mut graded): (Vec<_>, Vec<_>) = join
            .inputs
            .iter()
            .map(|i| (i.source.unwrap_or(PictureSource::Input), i.graded))
            .unzip();
        match join.kind {
            JoinKind::Parallel | JoinKind::Layer => {
                if sources.len() > PLAN_INPUTS {
                    return Err(PlanError::TooManyInputs(join.id));
                }
                if sources.is_empty() {
                    sources.push(PictureSource::Input);
                    graded.push(false);
                }
            }
            JoinKind::Splitter => {
                sources.truncate(1);
                graded.truncate(1);
                if sources.is_empty() {
                    sources.push(PictureSource::Input);
                }
            }
            JoinKind::Combiner => {
                sources.truncate(3);
                graded.truncate(3);
                sources.resize(3, PictureSource::Input);
            }
        }
        graded.resize(sources.len(), false);
        // Only a Layer's inputs above the first are graded.
        for (i, g) in graded.iter_mut().enumerate() {
            *g &= join.kind == JoinKind::Layer && i > 0;
        }
        Ok(Self {
            id: join.id,
            pos: join.pos,
            kind: Some(join.kind),
            pictures: sources,
            graded,
            key: None,
        })
    }

    /// The nodes this one reads from, by picture or key.
    fn needs(&self) -> BTreeSet<u32> {
        self.pictures
            .iter()
            .filter_map(|s| s.node())
            .chain(self.key)
            .collect()
    }

    /// The picture outputs: three for a Splitter, one for every other node.
    fn outputs(&self) -> Vec<PictureSource> {
        if self.kind == Some(JoinKind::Splitter) {
            (0u8..3).map(|c| PictureSource::Split(self.id, c)).collect()
        } else {
            vec![PictureSource::Node(self.id)]
        }
    }

    /// The values the step reads (each once, the Input's picture left out), given a Parallel's base.
    fn reads(&self, base: Option<PictureSource>) -> Vec<Val> {
        let mut reads = Vec::new();
        let mut add = |v: Val| {
            if !reads.contains(&v) {
                reads.push(v);
            }
        };
        for (src, &graded) in self.pictures.iter().zip(&self.graded) {
            match (graded, src.node()) {
                (true, Some(n)) => {
                    add(Val::Graded(n));
                    add(Val::Key(n));
                }
                _ => {
                    if let Some(v) = Val::of(*src) {
                        add(v);
                    }
                }
            }
        }
        if let Some(v) = base.and_then(Val::of) {
            add(v);
        }
        if let Some(k) = self.key {
            add(Val::Key(k));
        }
        reads
    }
}

/// A value a step makes and later steps read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Val {
    /// A node's picture output: the node and the channel (0 for every node but a Splitter).
    Pic(u32, u8),
    /// The graded value of a corrector or Film node.
    Graded(u32),
    /// A node's key `m`.
    Key(u32),
}

impl Val {
    fn of(src: PictureSource) -> Option<Self> {
        match src {
            PictureSource::Input => None,
            PictureSource::Node(m) => Some(Self::Pic(m, 0)),
            PictureSource::Split(m, c) => Some(Self::Pic(m, c)),
        }
    }

    fn producer(self) -> u32 {
        match self {
            Self::Pic(m, _) | Self::Graded(m) | Self::Key(m) => m,
        }
    }

    fn is_key(self) -> bool {
        matches!(self, Self::Key(_))
    }
}

/// Compiles a graph into a plan that ends at `stop`: the correctors of `graph` and the mixers,
/// Splitters and Combiners in `joins`.
///
/// # Errors
///
/// [`PlanError`] for a graph with a repeated id, a link or a stop that names a node the graph does
/// not have, a circle, a mixer with more than [`PLAN_INPUTS`] inputs, or a mask stop on a node with no
/// mask. A graph that `ym-core` validated has none of them.
pub fn compile_with(graph: &PlanGraph, joins: &[PlanJoin], stop: Stop) -> Result<Plan, PlanError> {
    let mut items: BTreeMap<u32, Item> = BTreeMap::new();
    let all = graph
        .nodes
        .iter()
        .map(|n| Ok(Item::corrector(n)))
        .chain(joins.iter().map(Item::join));
    for item in all {
        let item = item?;
        let id = item.id;
        if items.insert(id, item).is_some() {
            return Err(PlanError::DuplicateNode(id));
        }
    }
    let known = |id: u32| {
        if items.contains_key(&id) {
            Ok(())
        } else {
            Err(PlanError::UnknownNode(id))
        }
    };
    for item in items.values() {
        for src in &item.pictures {
            if let Some(m) = src.node() {
                known(m)?;
            }
            if let PictureSource::Split(m, c) = src
                && *c >= 3
            {
                return Err(PlanError::UnknownNode(*m));
            }
        }
        if let Some(m) = item.key {
            known(m)?;
        }
    }
    if let Some(m) = graph.output.node() {
        known(m)?;
    }
    if let Stop::Picture(k) | Stop::Mask(k) = stop {
        known(k)?;
    }
    if let Stop::Mask(k) = stop
        && items[&k].kind.is_some()
    {
        return Err(PlanError::NoMask(k));
    }
    // A graded input reads a corrector or Film node's picture output; anything else is read as its picture.
    let correctors: BTreeSet<u32> = items
        .values()
        .filter(|i| i.kind.is_none())
        .map(|i| i.id)
        .collect();
    for item in items.values_mut() {
        for (src, g) in item.pictures.iter().zip(item.graded.iter_mut()) {
            *g &= matches!(src, PictureSource::Node(n) if correctors.contains(n));
        }
    }

    let order = topological_order(&items)?;
    let output_root = graph.output.node();
    let live = closure(&items, output_root);
    let mut number = 0u32;
    let cards = order
        .iter()
        .map(|&id| Card {
            id,
            number: items[&id].kind.is_none().then(|| {
                number += 1;
                number
            }),
            live: live.contains(&id),
        })
        .collect();
    let root = match stop {
        Stop::Output => output_root,
        Stop::Picture(k) | Stop::Mask(k) => Some(k),
    };
    let wanted = closure(&items, root);
    let steps: Vec<u32> = order.into_iter().filter(|id| wanted.contains(id)).collect();
    let bases = parallel_bases(&items, &steps);

    let compiler = Compiler::new(&items, &bases, steps, graph.output, stop);
    let (passes, buffers) = compiler.passes();
    let result = compiler.result(&passes);
    Ok(Plan {
        stop,
        cards,
        passes,
        buffers,
        result,
        bases,
    })
}

/// `root` and everything it needs: its picture sources and its key source, and theirs.
fn closure(items: &BTreeMap<u32, Item>, root: Option<u32>) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut todo: Vec<u32> = root.into_iter().collect();
    while let Some(id) = todo.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(item) = items.get(&id) else { continue };
        todo.extend(item.needs());
    }
    seen
}

/// The order in which two ready nodes run: Main first, then `pos` (x, then y), then the id.
fn precedes(a: &Item, b: &Item) -> Ordering {
    (a.id != MAIN_NODE)
        .cmp(&(b.id != MAIN_NODE))
        .then(a.pos[0].total_cmp(&b.pos[0]))
        .then(a.pos[1].total_cmp(&b.pos[1]))
        .then(a.id.cmp(&b.id))
}

/// Kahn's algorithm over the picture and key links, picking the node that [`precedes`] the others
/// among those that are ready.
fn topological_order(items: &BTreeMap<u32, Item>) -> Result<Vec<u32>, PlanError> {
    let mut waiting: BTreeMap<u32, BTreeSet<u32>> =
        items.iter().map(|(&id, n)| (id, n.needs())).collect();
    let mut order = Vec::with_capacity(items.len());
    while !waiting.is_empty() {
        let next = waiting
            .iter()
            .filter(|(_, deps)| deps.is_empty())
            .map(|(&id, _)| id)
            .min_by(|&a, &b| precedes(&items[&a], &items[&b]))
            .ok_or(PlanError::Cycle)?;
        waiting.remove(&next);
        for deps in waiting.values_mut() {
            deps.remove(&next);
        }
        order.push(next);
    }
    Ok(order)
}

/// The base of every Parallel among `steps` (a topological order): the immediate dominator of its output in
/// the graph of picture values rooted at the Input's picture. A value leads to the outputs of every node
/// that reads it on a picture input (`docs/25` section 3.2).
fn parallel_bases(items: &BTreeMap<u32, Item>, steps: &[u32]) -> BTreeMap<u32, PictureSource> {
    let mut dominators: BTreeMap<PictureSource, BTreeSet<PictureSource>> = BTreeMap::new();
    dominators.insert(PictureSource::Input, BTreeSet::from([PictureSource::Input]));
    let mut bases = BTreeMap::new();
    for id in steps {
        let item = &items[id];
        // What every path to this node's pictures goes through: the common dominators of its inputs.
        let mut common: Option<BTreeSet<PictureSource>> = None;
        for src in &item.pictures {
            let theirs = dominators.get(src).cloned().unwrap_or_default();
            common = Some(match common {
                None => theirs,
                Some(c) => c.intersection(&theirs).copied().collect(),
            });
        }
        let common = common.unwrap_or_default();
        if item.kind == Some(JoinKind::Parallel) {
            // The dominators of a node form a chain, so the deepest has the most dominators of its own.
            let deepest = common
                .iter()
                .copied()
                .max_by_key(|d| dominators.get(d).map_or(0, BTreeSet::len));
            bases.insert(*id, deepest.unwrap_or(PictureSource::Input));
        }
        for out in item.outputs() {
            let mut set = common.clone();
            set.insert(out);
            dominators.insert(out, set);
        }
    }
    bases
}

/// A buffer table that grows as passes are laid out.
#[derive(Clone, Default)]
struct Buffers {
    ids: BTreeMap<Val, u16>,
    kinds: Vec<BufferKind>,
}

impl Buffers {
    fn of(&mut self, v: Val) -> u16 {
        if let Some(&b) = self.ids.get(&v) {
            return b;
        }
        let b = self.kinds.len() as u16;
        self.kinds.push(if v.is_key() {
            BufferKind::Key
        } else {
            BufferKind::Picture
        });
        self.ids.insert(v, b);
        b
    }
}

/// Registers or key slots of one pass: the lowest free one is taken first, so the numbering is the
/// same on every machine.
struct Pool {
    free: BTreeSet<u8>,
    next: usize,
    limit: usize,
}

impl Pool {
    fn new(limit: usize) -> Self {
        Self {
            free: BTreeSet::new(),
            next: 0,
            limit,
        }
    }

    fn take(&mut self) -> Option<u8> {
        if let Some(&r) = self.free.iter().next() {
            self.free.remove(&r);
            return Some(r);
        }
        if self.next < self.limit {
            self.next += 1;
            return Some((self.next - 1) as u8);
        }
        None
    }

    fn give(&mut self, r: u8) {
        self.free.insert(r);
    }

    fn used(&self) -> usize {
        self.next
    }
}

/// The registers, key slots and buffers a pass being laid out has handed out.
struct Layout {
    pictures: Pool,
    slots: Pool,
    register_of: BTreeMap<Val, u8>,
    slot_of: BTreeMap<Val, u8>,
}

/// The values of a plan's steps: who makes them and who reads them.
struct Compiler<'a> {
    items: &'a BTreeMap<u32, Item>,
    bases: &'a BTreeMap<u32, PictureSource>,
    /// The nodes the plan runs, in order.
    steps: Vec<u32>,
    stop: Stop,
    /// The position of each node in `steps`.
    at: BTreeMap<u32, usize>,
    /// For each value, the positions of the steps that read it, in order.
    readers: BTreeMap<Val, Vec<usize>>,
    /// The picture (node and channel) that is the plan's result, when it is a node's.
    result_picture: Option<(u32, u8)>,
    /// The node whose key is the plan's result.
    result_key: Option<u32>,
    output: PictureSource,
}

impl<'a> Compiler<'a> {
    fn new(
        items: &'a BTreeMap<u32, Item>,
        bases: &'a BTreeMap<u32, PictureSource>,
        steps: Vec<u32>,
        output: PictureSource,
        stop: Stop,
    ) -> Self {
        let at: BTreeMap<u32, usize> = steps.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut readers: BTreeMap<Val, Vec<usize>> = BTreeMap::new();
        for (i, id) in steps.iter().enumerate() {
            for v in items[id].reads(bases.get(id).copied()) {
                readers.entry(v).or_default().push(i);
            }
        }
        let (result_picture, result_key) = match stop {
            Stop::Output => match output {
                PictureSource::Input => (None, None),
                PictureSource::Node(m) => (Some((m, 0)), None),
                PictureSource::Split(m, c) => (Some((m, c)), None),
            },
            Stop::Picture(k) => (Some((k, 0)), None),
            Stop::Mask(k) => (None, Some(k)),
        };
        Self {
            items,
            bases,
            steps,
            stop,
            at,
            readers,
            result_picture,
            result_key,
            output,
        }
    }

    /// The last position below `end` among `readers` of a value.
    fn last_reader(readers: Option<&Vec<usize>>, end: usize) -> Option<usize> {
        readers.and_then(|r| r.iter().copied().filter(|&i| i < end).max())
    }

    fn read_after(readers: Option<&Vec<usize>>, end: usize) -> bool {
        readers.is_some_and(|r| r.iter().any(|&i| i >= end))
    }

    fn is_result(&self, v: Val) -> bool {
        match v {
            Val::Pic(m, c) => self.result_picture == Some((m, c)),
            Val::Key(m) => self.result_key == Some(m),
            Val::Graded(_) => false,
        }
    }

    /// Where a picture value is read from in the pass starting at `begin`: its register when a step of the pass
    /// makes it, else its buffer.
    fn picture_ref(
        &self,
        v: Option<Val>,
        begin: usize,
        l: &Layout,
        buffers: &mut Buffers,
    ) -> Option<PictureRef> {
        let Some(v) = v else {
            return Some(PictureRef::Input);
        };
        if self.at[&v.producer()] >= begin {
            Some(PictureRef::Register(*l.register_of.get(&v)?))
        } else {
            Some(PictureRef::Buffer(buffers.of(v)))
        }
    }

    fn key_ref(&self, v: Val, begin: usize, l: &Layout, buffers: &mut Buffers) -> Option<KeyRef> {
        if self.at[&v.producer()] >= begin {
            Some(KeyRef::Slot(*l.slot_of.get(&v)?))
        } else {
            Some(KeyRef::Buffer(buffers.of(v)))
        }
    }

    /// Where the step at `k` writes the picture value `v`: `Some(None)` when nothing reads it, `None` when
    /// the pass has no register left.
    fn picture_out(
        &self,
        v: Val,
        end: usize,
        l: &mut Layout,
        buffers: &mut Buffers,
    ) -> Option<Option<PictureOut>> {
        let readers = self.readers.get(&v);
        if readers.is_none() && !self.is_result(v) {
            return Some(None);
        }
        let in_pass = Self::last_reader(readers, end).is_some() || self.is_result(v);
        let register = if in_pass {
            let r = l.pictures.take()?;
            l.register_of.insert(v, r);
            Some(r)
        } else {
            None
        };
        let buffer = Self::read_after(readers, end).then(|| buffers.of(v));
        Some(Some(PictureOut { register, buffer }))
    }

    fn key_out(
        &self,
        v: Val,
        end: usize,
        l: &mut Layout,
        buffers: &mut Buffers,
    ) -> Option<Option<KeyOut>> {
        let readers = self.readers.get(&v);
        if readers.is_none() && !self.is_result(v) {
            return Some(None);
        }
        let in_pass = Self::last_reader(readers, end).is_some() || self.is_result(v);
        let slot = if in_pass {
            let s = l.slots.take()?;
            l.slot_of.insert(v, s);
            Some(s)
        } else {
            None
        };
        let buffer = Self::read_after(readers, end).then(|| buffers.of(v));
        Some(Some(KeyOut { slot, buffer }))
    }

    /// Lays out the steps `begin..end` as one pass, or `None` when they need more registers or key
    /// slots than a pass has.
    fn layout(&self, begin: usize, end: usize, buffers: &mut Buffers) -> Option<Pass> {
        let mut l = Layout {
            pictures: Pool::new(PLAN_REGISTERS),
            slots: Pool::new(PLAN_KEYS),
            register_of: BTreeMap::new(),
            slot_of: BTreeMap::new(),
        };
        let mut steps = Vec::with_capacity(end - begin);
        for k in begin..end {
            let id = self.steps[k];
            let item = &self.items[&id];
            let base = self.bases.get(&id).copied();
            // What the step reads.
            let mut inputs = [PictureRef::Input; PLAN_INPUTS];
            let mut alphas = [None; PLAN_INPUTS];
            for (i, (src, &graded)) in item.pictures.iter().zip(&item.graded).enumerate() {
                match (graded, src.node()) {
                    (true, Some(n)) => {
                        inputs[i] = self.picture_ref(Some(Val::Graded(n)), begin, &l, buffers)?;
                        alphas[i] = Some(self.key_ref(Val::Key(n), begin, &l, buffers)?);
                    }
                    _ => inputs[i] = self.picture_ref(Val::of(*src), begin, &l, buffers)?,
                }
            }
            let key = match item.key {
                None => None,
                Some(m) => Some(self.key_ref(Val::Key(m), begin, &l, buffers)?),
            };
            let base_ref = match base {
                None => None,
                Some(b) => Some(self.picture_ref(Val::of(b), begin, &l, buffers)?),
            };
            // The values it was the last in this pass to read: their registers are free for its outputs.
            for v in item.reads(base) {
                if self.at[&v.producer()] >= begin
                    && Self::last_reader(self.readers.get(&v), end) == Some(k)
                    && !self.is_result(v)
                {
                    if v.is_key() {
                        if let Some(s) = l.slot_of.remove(&v) {
                            l.slots.give(s);
                        }
                    } else if let Some(r) = l.register_of.remove(&v) {
                        l.pictures.give(r);
                    }
                }
            }
            // What it writes.
            let mut channels = [None; 3];
            let mut picture_out = None;
            if item.kind == Some(JoinKind::Splitter) {
                for (c, slot) in channels.iter_mut().enumerate() {
                    *slot = self.picture_out(Val::Pic(id, c as u8), end, &mut l, buffers)?;
                }
            } else {
                picture_out = self.picture_out(Val::Pic(id, 0), end, &mut l, buffers)?;
            }
            let graded_out = if item.kind.is_none() {
                self.picture_out(Val::Graded(id), end, &mut l, buffers)?
            } else {
                None
            };
            let key_out = if item.kind.is_none() {
                self.key_out(Val::Key(id), end, &mut l, buffers)?
            } else {
                None
            };
            let join = item.kind.map(|kind| {
                let mut is_base = 0u8;
                if let Some(b) = base {
                    for (i, src) in item.pictures.iter().enumerate() {
                        if *src == b {
                            is_base |= 1 << i;
                        }
                    }
                }
                JoinStep {
                    kind,
                    inputs,
                    count: item.pictures.len() as u8,
                    base: base_ref,
                    is_base,
                    alphas,
                    channels,
                }
            });
            steps.push(Step {
                node: id,
                picture: inputs[0],
                key,
                picture_out,
                key_out,
                graded_out,
                join,
            });
        }
        Some(Pass {
            steps,
            registers: l.pictures.used(),
            slots: l.slots.used(),
        })
    }

    /// The passes: each takes as many steps as fit.
    fn passes(&self) -> (Vec<Pass>, Vec<BufferKind>) {
        let n = self.steps.len();
        let mut passes = Vec::new();
        let mut buffers = Buffers::default();
        let mut begin = 0;
        while begin < n {
            let mut end = begin + 1;
            // A single step always fits: it reads its inputs from the Input or from buffers and
            // writes at most three pictures and a key.
            let mut best = {
                let mut trial = buffers.clone();
                let pass = self.layout(begin, end, &mut trial);
                (pass, trial)
            };
            while end < n {
                let mut trial = buffers.clone();
                match self.layout(begin, end + 1, &mut trial) {
                    Some(pass) => {
                        best = (Some(pass), trial);
                        end += 1;
                    }
                    None => break,
                }
            }
            let (pass, table) = best;
            buffers = table;
            passes.extend(pass);
            begin = end;
        }
        (passes, buffers.kinds)
    }

    /// Where the plan's value is. The node that makes it is the plan's last step: every other node
    /// of the plan is an ancestor of it.
    fn result(&self, passes: &[Pass]) -> PlanResult {
        let last = passes.last().and_then(|p| p.steps.last());
        match self.stop {
            Stop::Mask(_) => PlanResult::Mask(
                last.and_then(|s| s.key_out)
                    .and_then(|o| o.slot)
                    .map_or(KeyRef::Slot(0), KeyRef::Slot),
            ),
            Stop::Output | Stop::Picture(_) => {
                let from_input = matches!(self.stop, Stop::Output)
                    && matches!(self.output, PictureSource::Input);
                if from_input {
                    PlanResult::Picture(PictureRef::Input)
                } else {
                    let channel = self.result_picture.map_or(0, |(_, c)| usize::from(c));
                    let out = last.and_then(|s| match &s.join {
                        Some(j) if j.kind == JoinKind::Splitter => j.channels[channel],
                        _ => s.picture_out,
                    });
                    PlanResult::Picture(
                        out.and_then(|o| o.register)
                            .map_or(PictureRef::Register(0), PictureRef::Register),
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, x: f64, picture: PictureSource, key: Option<u32>) -> PlanNode {
        PlanNode {
            id,
            pos: [x, 0.0],
            picture: Some(picture),
            key,
        }
    }

    #[test]
    fn the_default_graph_is_main_in_one_pass() {
        let plan = compile(&PlanGraph::single_main(), Stop::Output).unwrap();
        assert_eq!(plan, Plan::main_only());
        assert!(plan.is_main_only());
        assert_eq!(plan.node_ids(), vec![1]);
    }

    #[test]
    fn a_chain_needs_one_register() {
        let mut nodes = vec![node(1, 0.0, PictureSource::Input, None)];
        for id in 2..=6 {
            nodes.push(node(id, f64::from(id), PictureSource::Node(id - 1), None));
        }
        let graph = PlanGraph {
            nodes,
            output: PictureSource::Node(6),
        };
        let plan = compile(&graph, Stop::Output).unwrap();
        assert_eq!(plan.passes.len(), 1);
        assert_eq!(plan.passes[0].registers, 1);
        assert_eq!(plan.node_ids(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn an_output_read_straight_from_the_input_has_no_steps() {
        let graph = PlanGraph {
            nodes: vec![node(1, 0.0, PictureSource::Input, None)],
            output: PictureSource::Input,
        };
        let plan = compile(&graph, Stop::Output).unwrap();
        assert!(plan.passes.is_empty());
        assert_eq!(plan.result, PlanResult::Picture(PictureRef::Input));
        assert!(!plan.cards[0].live);
    }
}
