//! The plan: a graph of corrector nodes compiled for the per-pixel chain (`docs/23-nodes.md`
//! section 5.2, `docs/algorithms/06-masks.md` section 1.3).
//!
//! The settings hold a graph (`ym-core`: the Input, the correctors, the Output and typed links).
//! [`compile`] turns what the render needs of it, a [`PlanGraph`], into a [`Plan`]: which nodes are
//! live, the order they run in, the card number of every node, the register each value lives in and,
//! when a plan needs more than [`PLAN_REGISTERS`] pictures or [`PLAN_KEYS`] keys alive at once, the
//! split into passes with a buffer for every value that crosses from one pass to the next.
//!
//! Everything here is pure and deterministic: the same graph gives the same plan whatever order its
//! nodes were listed in, on every machine. Nothing in this module knows a colour.
//!
//! # Live and dead
//!
//! A node is **live** when the Output needs it: it is on the chain of picture links that leads back
//! from the Output to the Input, or its key output feeds (through any number of key links) the key
//! input of a live node, or it is a picture ancestor of such a node. That is the closure of the
//! Output's picture source under "my picture source" and "my key source". Other nodes are dead: they
//! are in [`Plan::cards`] (with a number, dimmed) and not in the steps.
//!
//! # Order and numbers
//!
//! The order is a topological order over the picture and the key links together (a validated graph
//! has no circle), with ties broken by Main first, then the card's `pos` (x, then y), then the id. A
//! card's **number** is its place in the same order over all nodes, dead ones included, counted from 1.
//!
//! # Values, registers and passes
//!
//! A step reads one picture and may read one key, and writes its own picture (when a later step or the
//! result needs it) and its own key (the node's mask `m`, when a later step or the result needs it).
//! The Input's picture is the pass's input and takes no register: [`PictureRef::Input`]. Every other
//! value lives in a **register** (pictures) or a **key slot** from the step that makes it to the last
//! step of the pass that reads it, and a step may write into the register its own input leaves, so a
//! chain of any length needs one register. A value that a later pass reads as well is also written to
//! a **buffer** when it is made, and read from it in the later passes ([`PictureRef::Buffer`]).
//!
//! The compiler fills a pass with steps for as long as it fits [`PLAN_REGISTERS`] and [`PLAN_KEYS`]
//! and starts the next one when the next step would not. Every graph of this version fits one pass
//! unless a hand-written file fans out into more than eight pictures alive at once; the joins of a
//! later version (Parallel, Layer, Splitter and combiner, Outside) will use the same split.
//!
//! # Stops
//!
//! [`Stop::Output`] is the whole picture. [`Stop::Picture`] ends at a node's picture (a thumbnail:
//! the picture after the node, which the caller then runs through the Output) and [`Stop::Mask`] at a
//! node's mask (the mask view). Both name any node, live or dead; the plan holds the node's ancestors
//! in the full order, so for a chain it is a prefix of the full plan.
//!
//! # A picture input with no link
//!
//! A corrector whose picture input has no link ([`PlanNode::picture`] is `None`) reads the Input's
//! picture: the Input is the only picture there is. (The settings validator allows such a node; the
//! spec does not say what it reads.)

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Pictures a pass may keep alive at once (`docs/23` section 5.2).
pub const PLAN_REGISTERS: usize = 8;
/// Keys a pass may keep alive at once.
pub const PLAN_KEYS: usize = 16;
/// The id of Main, the corrector that comes first among equals.
pub const MAIN_NODE: u32 = 1;

/// Where a picture comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PictureSource {
    /// The Input's picture (the LogC4 picture of C8).
    Input,
    /// The picture output of a corrector.
    Node(u32),
}

/// A corrector as the compiler sees it: its place, its picture source and its key source.
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
/// picture the Output reads.
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

/// Where a plan ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stop {
    /// The picture the Output reads.
    Output,
    /// The picture of a node (a thumbnail, once it has gone through the Output).
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
}

/// What a card shows: the node's number in the flow and whether it is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Card {
    /// The node.
    pub id: u32,
    /// Its place in the order over all nodes, from 1.
    pub number: u32,
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

/// One node run on a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// The node.
    pub node: u32,
    /// The picture it reads.
    pub picture: PictureRef,
    /// The key it reads (its key input), when it is linked.
    pub key: Option<KeyRef>,
    /// The picture it writes; `None` when nothing reads it (a node live only for its key).
    pub picture_out: Option<PictureOut>,
    /// The key it writes (its mask `m`); `None` when nothing reads it.
    pub key_out: Option<KeyOut>,
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
    /// Every corrector with its number and whether it is live, in the order over all nodes.
    pub cards: Vec<Card>,
    /// The passes (empty when the result is the Input's picture itself).
    pub passes: Vec<Pass>,
    /// What each buffer holds, by buffer number.
    pub buffers: Vec<BufferKind>,
    /// Where the plan's value is when the last pass ends.
    pub result: PlanResult,
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
                number: 1,
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
                }],
                registers: 1,
                slots: 0,
            }],
            buffers: Vec::new(),
            result: PlanResult::Picture(PictureRef::Register(0)),
        }
    }

    /// Every step of every pass, in order.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.passes.iter().flat_map(|p| p.steps.iter())
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

/// Compiles a graph into a plan that ends at `stop`.
///
/// # Errors
///
/// [`PlanError`] for a graph with a repeated id, a link or a stop that names a node the graph does
/// not have, or a circle. A graph that `ym-core` validated has none of them.
pub fn compile(graph: &PlanGraph, stop: Stop) -> Result<Plan, PlanError> {
    let mut by_id: BTreeMap<u32, &PlanNode> = BTreeMap::new();
    for node in &graph.nodes {
        if by_id.insert(node.id, node).is_some() {
            return Err(PlanError::DuplicateNode(node.id));
        }
    }
    let known = |id: u32| {
        if by_id.contains_key(&id) {
            Ok(())
        } else {
            Err(PlanError::UnknownNode(id))
        }
    };
    for node in &graph.nodes {
        if let Some(PictureSource::Node(m)) = node.picture {
            known(m)?;
        }
        if let Some(m) = node.key {
            known(m)?;
        }
    }
    if let PictureSource::Node(m) = graph.output {
        known(m)?;
    }
    if let Stop::Picture(k) | Stop::Mask(k) = stop {
        known(k)?;
    }

    let order = topological_order(&by_id)?;
    let output_root = match graph.output {
        PictureSource::Input => None,
        PictureSource::Node(m) => Some(m),
    };
    let live = closure(&by_id, output_root);
    let cards = order
        .iter()
        .enumerate()
        .map(|(i, &id)| Card {
            id,
            number: i as u32 + 1,
            live: live.contains(&id),
        })
        .collect();
    let root = match stop {
        Stop::Output => output_root,
        Stop::Picture(k) | Stop::Mask(k) => Some(k),
    };
    let wanted = closure(&by_id, root);
    let steps: Vec<u32> = order.into_iter().filter(|id| wanted.contains(id)).collect();

    let compiler = Compiler::new(&by_id, steps, graph.output, stop);
    let (passes, buffers) = compiler.passes();
    let result = compiler.result(&passes);
    Ok(Plan {
        stop,
        cards,
        passes,
        buffers,
        result,
    })
}

/// The picture source a node reads: its link, or the Input when it has none.
fn picture_source(node: &PlanNode) -> PictureSource {
    node.picture.unwrap_or(PictureSource::Input)
}

/// `root` and everything it needs: its picture source and its key source, and theirs.
fn closure(by_id: &BTreeMap<u32, &PlanNode>, root: Option<u32>) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut todo: Vec<u32> = root.into_iter().collect();
    while let Some(id) = todo.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = by_id.get(&id) else { continue };
        if let PictureSource::Node(m) = picture_source(node) {
            todo.push(m);
        }
        if let Some(m) = node.key {
            todo.push(m);
        }
    }
    seen
}

/// The order in which two ready nodes run: Main first, then `pos` (x, then y), then the id.
fn precedes(a: &PlanNode, b: &PlanNode) -> Ordering {
    (a.id != MAIN_NODE)
        .cmp(&(b.id != MAIN_NODE))
        .then(a.pos[0].total_cmp(&b.pos[0]))
        .then(a.pos[1].total_cmp(&b.pos[1]))
        .then(a.id.cmp(&b.id))
}

/// Kahn's algorithm over the picture and key links, picking the node that [`precedes`] the others
/// among those that are ready.
fn topological_order(by_id: &BTreeMap<u32, &PlanNode>) -> Result<Vec<u32>, PlanError> {
    let needs = |node: &PlanNode| -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        if let PictureSource::Node(m) = picture_source(node) {
            set.insert(m);
        }
        if let Some(m) = node.key {
            set.insert(m);
        }
        set
    };
    let mut waiting: BTreeMap<u32, BTreeSet<u32>> =
        by_id.iter().map(|(&id, n)| (id, needs(n))).collect();
    let mut order = Vec::with_capacity(by_id.len());
    while !waiting.is_empty() {
        let next = waiting
            .iter()
            .filter(|(_, deps)| deps.is_empty())
            .map(|(&id, _)| id)
            .min_by(|&a, &b| precedes(by_id[&a], by_id[&b]))
            .ok_or(PlanError::Cycle)?;
        waiting.remove(&next);
        for deps in waiting.values_mut() {
            deps.remove(&next);
        }
        order.push(next);
    }
    Ok(order)
}

/// A buffer table that grows as passes are laid out.
#[derive(Clone, Default)]
struct Buffers {
    pictures: BTreeMap<u32, u16>,
    keys: BTreeMap<u32, u16>,
    kinds: Vec<BufferKind>,
}

impl Buffers {
    fn picture(&mut self, node: u32) -> u16 {
        if let Some(&b) = self.pictures.get(&node) {
            return b;
        }
        let b = self.kinds.len() as u16;
        self.kinds.push(BufferKind::Picture);
        self.pictures.insert(node, b);
        b
    }

    fn key(&mut self, node: u32) -> u16 {
        if let Some(&b) = self.keys.get(&node) {
            return b;
        }
        let b = self.kinds.len() as u16;
        self.kinds.push(BufferKind::Key);
        self.keys.insert(node, b);
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

/// The values of a plan's steps: who makes them and who reads them.
struct Compiler<'a> {
    by_id: &'a BTreeMap<u32, &'a PlanNode>,
    /// The nodes the plan runs, in order.
    steps: Vec<u32>,
    stop: Stop,
    /// The position of each node in `steps`.
    at: BTreeMap<u32, usize>,
    /// For each node, the positions of the steps that read its picture, in order.
    picture_readers: BTreeMap<u32, Vec<usize>>,
    /// The same for its key.
    key_readers: BTreeMap<u32, Vec<usize>>,
    /// The node whose picture is the plan's result, when it is a node's.
    result_picture: Option<u32>,
    /// The node whose key is the plan's result.
    result_key: Option<u32>,
    output: PictureSource,
}

impl<'a> Compiler<'a> {
    fn new(
        by_id: &'a BTreeMap<u32, &'a PlanNode>,
        steps: Vec<u32>,
        output: PictureSource,
        stop: Stop,
    ) -> Self {
        let at: BTreeMap<u32, usize> = steps.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut picture_readers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut key_readers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, id) in steps.iter().enumerate() {
            let node = by_id[id];
            if let PictureSource::Node(m) = picture_source(node) {
                picture_readers.entry(m).or_default().push(i);
            }
            if let Some(m) = node.key {
                key_readers.entry(m).or_default().push(i);
            }
        }
        let (result_picture, result_key) = match stop {
            Stop::Output => match output {
                PictureSource::Input => (None, None),
                PictureSource::Node(m) => (Some(m), None),
            },
            Stop::Picture(k) => (Some(k), None),
            Stop::Mask(k) => (None, Some(k)),
        };
        Self {
            by_id,
            steps,
            stop,
            at,
            picture_readers,
            key_readers,
            result_picture,
            result_key,
            output,
        }
    }

    /// The last position below `end` among `readers` of a value made at or after `begin`.
    fn last_reader(readers: Option<&Vec<usize>>, end: usize) -> Option<usize> {
        readers.and_then(|r| r.iter().copied().filter(|&i| i < end).max())
    }

    fn read_after(readers: Option<&Vec<usize>>, end: usize) -> bool {
        readers.is_some_and(|r| r.iter().any(|&i| i >= end))
    }

    /// Lays out the steps `begin..end` as one pass, or `None` when they need more registers or key
    /// slots than a pass has.
    fn layout(&self, begin: usize, end: usize, buffers: &mut Buffers) -> Option<Pass> {
        let mut pictures = Pool::new(PLAN_REGISTERS);
        let mut slots = Pool::new(PLAN_KEYS);
        let mut register_of: BTreeMap<u32, u8> = BTreeMap::new();
        let mut slot_of: BTreeMap<u32, u8> = BTreeMap::new();
        let mut steps = Vec::with_capacity(end - begin);
        for k in begin..end {
            let id = self.steps[k];
            let node = self.by_id[&id];
            // What the step reads, and which values it was the last in this pass to read.
            let source = picture_source(node);
            let picture = match source {
                PictureSource::Input => PictureRef::Input,
                PictureSource::Node(m) if self.at[&m] >= begin => {
                    PictureRef::Register(*register_of.get(&m)?)
                }
                PictureSource::Node(m) => PictureRef::Buffer(buffers.picture(m)),
            };
            let key = match node.key {
                None => None,
                Some(m) if self.at[&m] >= begin => Some(KeyRef::Slot(*slot_of.get(&m)?)),
                Some(m) => Some(KeyRef::Buffer(buffers.key(m))),
            };
            if let PictureSource::Node(m) = source
                && self.at[&m] >= begin
                && Self::last_reader(self.picture_readers.get(&m), end) == Some(k)
                && self.result_picture != Some(m)
                && let Some(r) = register_of.remove(&m)
            {
                pictures.give(r);
            }
            if let Some(m) = node.key
                && self.at[&m] >= begin
                && Self::last_reader(self.key_readers.get(&m), end) == Some(k)
                && self.result_key != Some(m)
                && let Some(s) = slot_of.remove(&m)
            {
                slots.give(s);
            }
            // What it writes.
            let readers = self.picture_readers.get(&id);
            let picture_out = if readers.is_some() || self.result_picture == Some(id) {
                let in_pass =
                    Self::last_reader(readers, end).is_some() || self.result_picture == Some(id);
                let register = if in_pass {
                    let r = pictures.take()?;
                    register_of.insert(id, r);
                    Some(r)
                } else {
                    None
                };
                let buffer = Self::read_after(readers, end).then(|| buffers.picture(id));
                Some(PictureOut { register, buffer })
            } else {
                None
            };
            let readers = self.key_readers.get(&id);
            let key_out = if readers.is_some() || self.result_key == Some(id) {
                let in_pass =
                    Self::last_reader(readers, end).is_some() || self.result_key == Some(id);
                let slot = if in_pass {
                    let s = slots.take()?;
                    slot_of.insert(id, s);
                    Some(s)
                } else {
                    None
                };
                let buffer = Self::read_after(readers, end).then(|| buffers.key(id));
                Some(KeyOut { slot, buffer })
            } else {
                None
            };
            steps.push(Step {
                node: id,
                picture,
                key,
                picture_out,
                key_out,
            });
        }
        Some(Pass {
            steps,
            registers: pictures.used(),
            slots: slots.used(),
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
            // writes at most one picture and one key.
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
                    PlanResult::Picture(
                        last.and_then(|s| s.picture_out)
                            .and_then(|o| o.register)
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
