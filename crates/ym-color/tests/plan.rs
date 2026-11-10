//! The plan (`ym_color::plan`, `docs/23-nodes.md` section 5.2, `docs/algorithms/06-masks.md` section
//! 1.3): live and dead nodes, the order and the card numbers, registers and passes, and the stops.
//!
//! A small interpreter runs a compiled plan on numbers (a node turns a picture `x` into `1.5 x + id`
//! and a key `k` into `(x + id) / 4 + k`) and its answer is compared with the graph evaluated
//! directly, so that a wrong register, a buffer read before it is written or a value that dies too
//! early shows as a different number.

use std::collections::BTreeMap;

use ym_color::plan::{
    BufferKind, JoinInput, JoinKind, KeyRef, MAIN_NODE, PLAN_KEYS, PLAN_REGISTERS, PictureOut,
    PictureRef, PictureSource, Plan, PlanError, PlanGraph, PlanJoin, PlanNode, PlanResult, Stop,
    compile, compile_with,
};

const INPUT: f64 = 7.0;

fn node(id: u32, pos: [f64; 2], picture: PictureSource, key: Option<u32>) -> PlanNode {
    PlanNode {
        id,
        pos,
        picture: Some(picture),
        key,
    }
}

fn graph(nodes: Vec<PlanNode>, output: u32) -> PlanGraph {
    PlanGraph {
        nodes,
        output: PictureSource::Node(output),
    }
}

fn main_node() -> PlanNode {
    node(MAIN_NODE, [0.0, 0.0], PictureSource::Input, None)
}

// ---- the interpreter -------------------------------------------------------------------------

/// The value a node's picture output has, evaluated straight from the graph.
fn direct_picture(g: &PlanGraph, id: u32) -> f64 {
    let n = g.nodes.iter().find(|n| n.id == id).unwrap();
    let x = match n.picture.unwrap_or(PictureSource::Input) {
        PictureSource::Input => INPUT,
        PictureSource::Node(m) => direct_picture(g, m),
        PictureSource::Split(..) => unreachable!("no Splitter in a graph of correctors"),
    };
    1.5 * x + f64::from(id)
}

/// The node's key output, evaluated straight from the graph.
fn direct_key(g: &PlanGraph, id: u32) -> f64 {
    let n = g.nodes.iter().find(|n| n.id == id).unwrap();
    let x = match n.picture.unwrap_or(PictureSource::Input) {
        PictureSource::Input => INPUT,
        PictureSource::Node(m) => direct_picture(g, m),
        PictureSource::Split(..) => unreachable!("no Splitter in a graph of correctors"),
    };
    (x + f64::from(id)) / 4.0 + n.key.map_or(0.0, |k| direct_key(g, k))
}

/// A value read from a register or buffer; panics on one that was not written.
fn get(r: PictureRef, registers: &[f64], buffers: &[f64]) -> f64 {
    let x = match r {
        PictureRef::Input => INPUT,
        PictureRef::Register(r) => registers[usize::from(r)],
        PictureRef::Buffer(b) => buffers[usize::from(b)],
    };
    assert!(!x.is_nan(), "{r:?} was not written");
    x
}

fn get_key(k: KeyRef, slots: &[f64], buffers: &[f64]) -> f64 {
    let x = match k {
        KeyRef::Slot(s) => slots[usize::from(s)],
        KeyRef::Buffer(b) => buffers[usize::from(b)],
    };
    assert!(!x.is_nan(), "{k:?} was not written");
    x
}

/// The numbers a join makes: a Parallel is `base + sum (in - base)`, a Layer folds `acc * 0.5 + layer + alpha` (alpha 0.25
/// when the input has none), a Splitter's channel `c` is `2 x + c`, a Combiner is `r + 2 g + 3 b`.
const NO_ALPHA: f64 = 0.25;

/// Runs a plan: the registers, the key slots and the buffers of the plan, nothing else. Returns the
/// plan's result as a number.
fn run(plan: &Plan) -> f64 {
    let mut registers = [f64::NAN; PLAN_REGISTERS];
    let mut slots = [f64::NAN; PLAN_KEYS];
    let mut buffers = vec![f64::NAN; plan.buffers.len()];
    for pass in &plan.passes {
        // Registers and key slots are the pass's own: whatever an earlier pass left is gone.
        registers = [f64::NAN; PLAN_REGISTERS];
        slots = [f64::NAN; PLAN_KEYS];
        assert!(pass.registers <= PLAN_REGISTERS && pass.slots <= PLAN_KEYS);
        for step in &pass.steps {
            let write =
                |o: Option<PictureOut>, v: f64, registers: &mut [f64], buffers: &mut [f64]| {
                    if let Some(o) = o {
                        if let Some(r) = o.register {
                            registers[usize::from(r)] = v;
                        }
                        if let Some(b) = o.buffer {
                            assert_eq!(plan.buffers[usize::from(b)], BufferKind::Picture);
                            buffers[usize::from(b)] = v;
                        }
                    }
                };
            if let Some(j) = &step.join {
                let ins: Vec<f64> = j
                    .inputs()
                    .iter()
                    .map(|&r| get(r, &registers, &buffers))
                    .collect();
                assert_eq!(step.picture, j.inputs[0]);
                match j.kind {
                    JoinKind::Parallel => {
                        let base =
                            get(j.base.expect("a Parallel has a base"), &registers, &buffers);
                        let mut acc = base;
                        for (i, v) in ins.iter().enumerate() {
                            if !j.input_is_base(i) {
                                acc += v - base;
                            }
                        }
                        write(step.picture_out, acc, &mut registers, &mut buffers);
                    }
                    JoinKind::Layer => {
                        assert!(j.alphas[0].is_none());
                        let mut acc = ins[0];
                        for (i, v) in ins.iter().enumerate().skip(1) {
                            let a = j.alphas[i].map_or(NO_ALPHA, |k| get_key(k, &slots, &buffers));
                            acc = acc * 0.5 + v + a;
                        }
                        write(step.picture_out, acc, &mut registers, &mut buffers);
                    }
                    JoinKind::Splitter => {
                        assert!(step.picture_out.is_none());
                        for (c, o) in j.channels.iter().enumerate() {
                            write(*o, 2.0 * ins[0] + c as f64, &mut registers, &mut buffers);
                        }
                    }
                    JoinKind::Combiner => {
                        let v = ins[0] + 2.0 * ins[1] + 3.0 * ins[2];
                        write(step.picture_out, v, &mut registers, &mut buffers);
                    }
                }
                continue;
            }
            let x = get(step.picture, &registers, &buffers);
            let k = step.key.map(|k| get_key(k, &slots, &buffers));
            let id = f64::from(step.node);
            write(step.picture_out, 1.5 * x + id, &mut registers, &mut buffers);
            write(
                step.graded_out,
                1.5 * x + id + GRADED,
                &mut registers,
                &mut buffers,
            );
            if let Some(o) = step.key_out {
                let v = (x + id) / 4.0 + k.unwrap_or(0.0);
                if let Some(s) = o.slot {
                    slots[usize::from(s)] = v;
                }
                if let Some(b) = o.buffer {
                    assert_eq!(plan.buffers[usize::from(b)], BufferKind::Key);
                    buffers[usize::from(b)] = v;
                }
            }
        }
    }
    match plan.result {
        PlanResult::Picture(PictureRef::Input) => INPUT,
        PlanResult::Picture(PictureRef::Register(r)) => registers[usize::from(r)],
        PlanResult::Picture(PictureRef::Buffer(b)) => buffers[usize::from(b)],
        PlanResult::Mask(KeyRef::Slot(s)) => slots[usize::from(s)],
        PlanResult::Mask(KeyRef::Buffer(b)) => buffers[usize::from(b)],
    }
}

/// What a graded value adds to a corrector's picture in the interpreter.
const GRADED: f64 = 100.0;

// ---- graphs ------------------------------------------------------------------------------------

/// Main fans out to `branches` nodes (ids 2..), each read by a second node (ids 20..) whose keys
/// are chained, and the last of those is the Output. Placed so that all the first nodes run before
/// the second ones: `branches` pictures are alive at once.
fn fan(branches: u32) -> PlanGraph {
    let mut nodes = vec![main_node()];
    for i in 0..branches {
        nodes.push(node(
            2 + i,
            [1.0, f64::from(i)],
            PictureSource::Node(MAIN_NODE),
            None,
        ));
    }
    for i in 0..branches {
        nodes.push(node(
            20 + i,
            [2.0, f64::from(i)],
            PictureSource::Node(2 + i),
            (i > 0).then(|| 20 + i - 1),
        ));
    }
    graph(nodes, 20 + branches - 1)
}

// ---- tests -----------------------------------------------------------------------------------

#[test]
fn the_default_graph_is_main_in_one_pass() {
    let plan = compile(&PlanGraph::single_main(), Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 1);
    assert_eq!(plan.node_ids(), vec![MAIN_NODE]);
    assert_eq!(plan, Plan::main_only());
    assert!(plan.is_main_only());
    assert_eq!(plan.cards.len(), 1);
    assert_eq!(
        (plan.cards[0].id, plan.cards[0].number, plan.cards[0].live),
        (1, Some(1), true)
    );
    assert!(plan.buffers.is_empty());
    assert_eq!(run(&plan), direct_picture(&PlanGraph::single_main(), 1));
}

#[test]
fn the_order_is_topological_with_main_first_then_position_then_id() {
    // Main feeds four nodes that feed one more through a chain of keys; nodes 5 and 4 stand at one
    // place (the id breaks the tie), 3 stands further left, 2 further right.
    let g = graph(
        vec![
            main_node(),
            node(2, [3.0, 0.0], PictureSource::Node(1), None),
            node(3, [1.0, 1.0], PictureSource::Node(1), None),
            node(4, [1.0, 0.0], PictureSource::Node(1), None),
            node(5, [1.0, 0.0], PictureSource::Node(1), None),
            node(6, [4.0, 0.0], PictureSource::Node(1), Some(2)),
        ],
        6,
    );
    let all = compile(&g, Stop::Output).unwrap();
    // 6 needs 1 (picture) and 2 (key); 3, 4 and 5 are dead. The live order: Main, 2, 6.
    assert_eq!(all.node_ids(), vec![1, 2, 6]);
    // The cards count every node, in the order over all of them: Main, then y within x = 1, the
    // id within one place, then x = 3, then x = 4.
    let numbers: Vec<(u32, Option<u32>, bool)> =
        all.cards.iter().map(|c| (c.id, c.number, c.live)).collect();
    assert_eq!(
        numbers,
        vec![
            (1, Some(1), true),
            (4, Some(2), false),
            (5, Some(3), false),
            (3, Some(4), false),
            (2, Some(5), true),
            (6, Some(6), true)
        ]
    );
}

#[test]
fn a_node_after_its_sources_even_when_its_place_is_far_left() {
    // Node 3 stands left of node 2 but reads its picture: it still runs after it.
    let g = graph(
        vec![
            main_node(),
            node(2, [5.0, 0.0], PictureSource::Node(1), None),
            node(3, [-5.0, 0.0], PictureSource::Node(2), None),
        ],
        3,
    );
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 2, 3]);
    assert_eq!(run(&plan), direct_picture(&g, 3));
}

#[test]
fn two_files_listing_one_graph_in_different_orders_give_one_plan() {
    let g = fan(6);
    let reference = compile(&g, Stop::Output).unwrap();
    for rotate in [1, 3, 7, 12] {
        let mut shuffled = g.clone();
        shuffled.nodes.rotate_left(rotate);
        assert_eq!(compile(&shuffled, Stop::Output).unwrap(), reference);
        shuffled.nodes.reverse();
        assert_eq!(compile(&shuffled, Stop::Output).unwrap(), reference);
    }
    // The same on every call.
    assert_eq!(compile(&g, Stop::Output).unwrap(), reference);
}

#[test]
fn a_dead_branch_is_left_out() {
    // Main -> 2 -> Output; 3 reads Main and goes nowhere; 4 reads 3 and goes nowhere.
    let g = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), None),
            node(3, [1.0, 1.0], PictureSource::Node(1), None),
            node(4, [2.0, 1.0], PictureSource::Node(3), None),
        ],
        2,
    );
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 2]);
    assert_eq!(
        plan.cards
            .iter()
            .map(|c| (c.id, c.live))
            .collect::<Vec<_>>(),
        vec![(1, true), (2, true), (3, false), (4, false)]
    );
    // The plan of the graph without the dead branch is the same plan, apart from the cards.
    let without = graph(g.nodes[..2].to_vec(), 2);
    let small = compile(&without, Stop::Output).unwrap();
    assert_eq!(plan.passes, small.passes);
    assert_eq!(plan.result, small.result);
    assert_eq!(run(&plan), run(&small));
}

#[test]
fn a_dead_picture_branch_whose_key_feeds_a_live_node_is_live() {
    // Node 5 reads Main's picture and goes nowhere, but its key is node 3's key input.
    let g = graph(
        vec![
            main_node(),
            node(5, [1.0, 1.0], PictureSource::Node(1), None),
            node(3, [2.0, 0.0], PictureSource::Node(1), Some(5)),
        ],
        3,
    );
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 5, 3]);
    assert!(plan.cards.iter().all(|c| c.live));
    let step5 = plan.steps().find(|s| s.node == 5).unwrap();
    // Its mask is used and its picture is not: it writes a key and no picture.
    assert!(step5.key_out.is_some() && step5.picture_out.is_none());
    let step3 = plan.steps().find(|s| s.node == 3).unwrap();
    assert!(step3.key.is_some());
    assert_eq!(run(&plan), direct_picture(&g, 3));
    // A key that is picture-dead one node further back: 6 -> 5's key input.
    let mut deeper = g.clone();
    deeper
        .nodes
        .push(node(6, [1.0, 2.0], PictureSource::Node(1), None));
    deeper.nodes[1].key = Some(6);
    let plan = compile(&deeper, Stop::Output).unwrap();
    // Main, then 6 (5 needs its key), then 5, then 3.
    assert_eq!(plan.node_ids(), vec![1, 6, 5, 3]);
    assert_eq!(run(&plan), direct_picture(&deeper, 3));
}

#[test]
fn a_chain_of_any_length_needs_one_register() {
    let mut nodes = vec![main_node()];
    for id in 2..=40 {
        nodes.push(node(
            id,
            [f64::from(id), 0.0],
            PictureSource::Node(id - 1),
            None,
        ));
    }
    let g = graph(nodes, 40);
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 1);
    assert_eq!(plan.passes[0].registers, 1);
    assert_eq!(plan.passes[0].slots, 0);
    assert_eq!(run(&plan), direct_picture(&g, 40));
}

#[test]
fn eight_pictures_alive_at_once_fit_one_pass_and_nine_need_two() {
    // fan(n): Main and n branches whose pictures are read later. The register count is the
    // number of pictures alive at the busiest moment (the last branch reuses Main's register).
    let eight = fan(8);
    let plan = compile(&eight, Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 1, "{:?}", plan.passes[0].registers);
    assert_eq!(plan.passes[0].registers, PLAN_REGISTERS);
    assert!(plan.buffers.is_empty());
    assert_eq!(run(&plan), direct_picture(&eight, 27));

    let nine = fan(9);
    let plan = compile(&nine, Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 2, "9 pictures alive need a split");
    assert!(plan.passes.iter().all(|p| p.registers <= PLAN_REGISTERS));
    assert!(!plan.buffers.is_empty());
    assert!(
        plan.buffers
            .iter()
            .all(|k| *k == BufferKind::Picture || *k == BufferKind::Key)
    );
    // The same number from the plan and from the graph, through registers and buffers.
    assert_eq!(run(&plan), direct_picture(&nine, 28));
    // Every node runs once, in the order of the single-pass plan.
    let ids = plan.node_ids();
    assert_eq!(ids.len(), 1 + 9 + 9);
    assert_eq!(ids[0], MAIN_NODE);
    // The first pass is full and the second starts with the node that did not fit.
    assert_eq!(
        plan.passes[0].steps.len() + plan.passes[1].steps.len(),
        ids.len()
    );
    assert_eq!(plan.passes[0].registers, PLAN_REGISTERS);
    assert_eq!(plan.passes[1].steps[0].picture, PictureRef::Buffer(0));

    // A deterministic split: the same graph gives the same passes.
    assert_eq!(compile(&nine, Stop::Output).unwrap(), plan);
}

#[test]
fn the_masks_of_a_wide_graph_cross_the_split_through_buffers() {
    // The key chain of fan(n) depends on every branch's picture, so the mask of the last node is
    // a number that breaks when any branch's register or buffer is wrong.
    for n in [9, 10, 14, 17] {
        let g = fan(n);
        let last = 20 + n - 1;
        let plan = compile(&g, Stop::Mask(last)).unwrap();
        assert!(plan.passes.iter().all(|p| p.registers <= PLAN_REGISTERS));
        assert_eq!(run(&plan), direct_key(&g, last), "n = {n}");
        let pictures = compile(&g, Stop::Output).unwrap();
        assert_eq!(run(&pictures), direct_picture(&g, last), "n = {n}");
        if n >= 9 {
            assert!(pictures.passes.len() >= 2, "n = {n}");
        }
    }
}

/// `n` nodes read Main and make masks, and a chain of `n` nodes takes one each as its key input, so
/// `n` keys are alive at once (the pictures of the mask nodes are not needed).
fn keys(n: u32) -> (PlanGraph, u32) {
    let mut nodes = vec![main_node()];
    for i in 0..n {
        nodes.push(node(
            2 + i,
            [1.0, f64::from(i)],
            PictureSource::Node(MAIN_NODE),
            None,
        ));
    }
    for i in 0..n {
        let prev = if i == 0 {
            PictureSource::Node(MAIN_NODE)
        } else {
            PictureSource::Node(100 + i - 1)
        };
        nodes.push(node(100 + i, [2.0, f64::from(i)], prev, Some(2 + i)));
    }
    (graph(nodes, 100 + n - 1), 100 + n - 1)
}

#[test]
fn sixteen_keys_alive_at_once_fit_one_pass_and_seventeen_need_two() {
    let (g, last) = keys(16);
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 1);
    assert_eq!(plan.passes[0].slots, PLAN_KEYS);
    // The mask nodes have no pictures that anything reads, so the pictures need one register.
    assert_eq!(plan.passes[0].registers, 1);
    assert_eq!(run(&plan), direct_picture(&g, last));

    let (g, last) = keys(17);
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.passes.len(), 2);
    assert!(plan.passes.iter().all(|p| p.slots <= PLAN_KEYS));
    assert_eq!(plan.passes[0].slots, PLAN_KEYS);
    assert_eq!(run(&plan), direct_picture(&g, last));
    assert_eq!(
        run(&compile(&g, Stop::Mask(last)).unwrap()),
        direct_key(&g, last)
    );
    assert!(plan.buffers.contains(&BufferKind::Key));
}

#[test]
fn a_thumbnail_stop_ends_at_the_node_and_a_chain_gives_a_prefix() {
    let mut nodes = vec![main_node()];
    for id in 2..=6 {
        nodes.push(node(
            id,
            [f64::from(id), 0.0],
            PictureSource::Node(id - 1),
            None,
        ));
    }
    let g = graph(nodes, 6);
    let full = compile(&g, Stop::Output).unwrap();
    for k in 1..=6u32 {
        let stopped = compile(&g, Stop::Picture(k)).unwrap();
        assert_eq!(stopped.stop, Stop::Picture(k));
        assert_eq!(stopped.node_ids(), full.node_ids()[..k as usize].to_vec());
        assert_eq!(run(&stopped), direct_picture(&g, k));
        // The cards do not depend on the stop.
        assert_eq!(stopped.cards, full.cards);
    }
}

#[test]
fn a_stop_names_any_node_live_or_dead_and_takes_its_ancestors() {
    // 4 is dead: it reads 3 which reads Main. A thumbnail of 4 is Main, 3, 4.
    let g = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), None),
            node(3, [1.0, 1.0], PictureSource::Node(1), None),
            node(4, [2.0, 1.0], PictureSource::Node(3), None),
        ],
        2,
    );
    let plan = compile(&g, Stop::Picture(4)).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 3, 4]);
    assert_eq!(run(&plan), direct_picture(&g, 4));
    // The cards still say 4 is dead.
    assert!(!plan.card(4).unwrap().live);
    // The mask view of 4 ends at its key.
    let mask = compile(&g, Stop::Mask(4)).unwrap();
    assert_eq!(mask.node_ids(), vec![1, 3, 4]);
    assert!(matches!(mask.result, PlanResult::Mask(_)));
    assert_eq!(run(&mask), direct_key(&g, 4));
    // A mask stop writes no picture for its own node.
    assert!(mask.steps().last().unwrap().picture_out.is_none());
}

#[test]
fn a_stop_on_a_node_with_a_key_input_takes_the_key_source_too() {
    let g = graph(
        vec![
            main_node(),
            node(2, [1.0, 1.0], PictureSource::Node(1), None),
            node(3, [2.0, 0.0], PictureSource::Node(1), Some(2)),
        ],
        3,
    );
    let plan = compile(&g, Stop::Mask(3)).unwrap();
    assert_eq!(plan.node_ids(), vec![1, 2, 3]);
    assert_eq!(run(&plan), direct_key(&g, 3));
}

#[test]
fn an_output_read_from_the_input_has_no_steps_and_every_node_is_dead() {
    let g = PlanGraph {
        nodes: vec![main_node()],
        output: PictureSource::Input,
    };
    let plan = compile(&g, Stop::Output).unwrap();
    assert!(plan.passes.is_empty());
    assert_eq!(plan.result, PlanResult::Picture(PictureRef::Input));
    assert!(!plan.card(1).unwrap().live);
    assert_eq!(run(&plan), INPUT);
}

#[test]
fn a_picture_input_with_no_link_reads_the_input() {
    let mut a = node(2, [1.0, 0.0], PictureSource::Input, None);
    a.picture = None;
    let g = graph(vec![main_node(), a], 2);
    let plan = compile(&g, Stop::Output).unwrap();
    assert_eq!(plan.node_ids(), vec![2]);
    assert_eq!(plan.steps().next().unwrap().picture, PictureRef::Input);
    assert_eq!(run(&plan), 1.5 * INPUT + 2.0);
    // Main is dead here: nothing reads it.
    assert!(!plan.card(1).unwrap().live);
}

#[test]
fn a_node_may_read_the_input_beside_main() {
    // The Input's extra branch: node 2 reads the Input, node 3 joins Main's output through a key.
    let g = graph(
        vec![
            main_node(),
            node(2, [-3.0, 5.0], PictureSource::Input, None),
            node(3, [2.0, 0.0], PictureSource::Node(1), Some(2)),
        ],
        3,
    );
    let plan = compile(&g, Stop::Output).unwrap();
    // Node 2 stands left of Main, but Main comes first among the nodes that are ready: Main, 2, 3.
    assert_eq!(plan.node_ids(), vec![1, 2, 3]);
    assert_eq!(run(&plan), direct_picture(&g, 3));
    assert_eq!(
        compile(&g, Stop::Mask(3)).map(|p| run(&p)).unwrap(),
        direct_key(&g, 3)
    );
}

#[test]
fn bad_graphs_are_errors() {
    let dup = graph(vec![main_node(), main_node()], 1);
    assert_eq!(
        compile(&dup, Stop::Output),
        Err(PlanError::DuplicateNode(1))
    );
    let unknown = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(9), None),
        ],
        2,
    );
    assert_eq!(
        compile(&unknown, Stop::Output),
        Err(PlanError::UnknownNode(9))
    );
    let key = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), Some(8)),
        ],
        2,
    );
    assert_eq!(compile(&key, Stop::Output), Err(PlanError::UnknownNode(8)));
    let ok = PlanGraph::single_main();
    assert_eq!(
        compile(&ok, Stop::Picture(5)),
        Err(PlanError::UnknownNode(5))
    );
    assert_eq!(compile(&ok, Stop::Mask(5)), Err(PlanError::UnknownNode(5)));
    // A circle through a picture and a key: 2 reads 3's key and 3 reads 2's picture.
    let circle = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), Some(3)),
            node(3, [2.0, 0.0], PictureSource::Node(2), None),
        ],
        3,
    );
    assert_eq!(compile(&circle, Stop::Output), Err(PlanError::Cycle));
    // A node whose key is its own.
    let own = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), Some(2)),
        ],
        2,
    );
    assert_eq!(compile(&own, Stop::Output), Err(PlanError::Cycle));
}

#[test]
fn the_registers_are_numbered_from_zero_and_reused() {
    let g = fan(4);
    let plan = compile(&g, Stop::Output).unwrap();
    let used: std::collections::BTreeSet<u8> = plan
        .steps()
        .filter_map(|s| s.picture_out.and_then(|o| o.register))
        .collect();
    assert_eq!(
        used.iter().copied().collect::<Vec<_>>(),
        (0..used.len() as u8).collect::<Vec<_>>()
    );
    // A step whose input dies with it writes into that register.
    let chain = graph(
        vec![
            main_node(),
            node(2, [1.0, 0.0], PictureSource::Node(1), None),
        ],
        2,
    );
    let plan = compile(&chain, Stop::Output).unwrap();
    let steps: Vec<_> = plan.steps().collect();
    assert_eq!(steps[0].picture_out.unwrap().register, Some(0));
    assert_eq!(steps[1].picture, PictureRef::Register(0));
    assert_eq!(steps[1].picture_out.unwrap().register, Some(0));
}

#[test]
fn the_cards_number_every_node_from_one_in_the_same_order() {
    let g = fan(3);
    let plan = compile(&g, Stop::Output).unwrap();
    let numbers: BTreeMap<u32, u32> = plan
        .cards
        .iter()
        .map(|c| (c.id, c.number.expect("a corrector has a number")))
        .collect();
    assert_eq!(numbers.len(), 7);
    assert_eq!(numbers[&MAIN_NODE], 1);
    let mut sorted: Vec<u32> = numbers.values().copied().collect();
    sorted.sort_unstable();
    assert_eq!(sorted, (1..=7).collect::<Vec<_>>());
    // The steps run in card order.
    let order = plan.node_ids();
    let by_number: Vec<u32> = plan.cards.iter().map(|c| c.id).collect();
    assert_eq!(order, by_number);
}

// =============================================================================================
// Joins (`docs/25-mixers.md` section 4.1): mixers, Splitters and Combiners
// =============================================================================================

/// A graph with joins, and the numbers its nodes make when evaluated straight from the links.
struct Model {
    graph: PlanGraph,
    joins: Vec<PlanJoin>,
}

fn join(id: u32, x: f64, kind: JoinKind, inputs: Vec<(Option<PictureSource>, bool)>) -> PlanJoin {
    PlanJoin {
        id,
        pos: [x, 0.0],
        kind,
        inputs: inputs
            .into_iter()
            .map(|(source, graded)| JoinInput { source, graded })
            .collect(),
    }
}

/// A join whose inputs are plain pictures.
fn plain(id: u32, x: f64, kind: JoinKind, sources: &[PictureSource]) -> PlanJoin {
    join(
        id,
        x,
        kind,
        sources.iter().map(|&s| (Some(s), false)).collect(),
    )
}

fn node_at(id: u32, x: f64, from: PictureSource) -> PlanNode {
    node(id, [x, 0.0], from, None)
}

impl Model {
    fn compile(&self, stop: Stop) -> Plan {
        compile_with(&self.graph, &self.joins, stop).unwrap()
    }

    fn join(&self, id: u32) -> Option<&PlanJoin> {
        self.joins.iter().find(|j| j.id == id)
    }

    fn value(&self, bases: &BTreeMap<u32, PictureSource>, src: PictureSource) -> f64 {
        match src {
            PictureSource::Input => INPUT,
            PictureSource::Node(id) => self.output_of(bases, id),
            PictureSource::Split(id, c) => {
                let j = self.join(id).expect("a Splitter");
                let x = self.value(bases, j.inputs[0].source.unwrap_or(PictureSource::Input));
                2.0 * x + f64::from(c)
            }
        }
    }

    fn corrector(&self, id: u32) -> Option<&PlanNode> {
        self.graph.nodes.iter().find(|n| n.id == id)
    }

    fn key(&self, bases: &BTreeMap<u32, PictureSource>, id: u32) -> f64 {
        let n = self.corrector(id).unwrap();
        let x = self.value(bases, n.picture.unwrap_or(PictureSource::Input));
        (x + f64::from(id)) / 4.0 + n.key.map_or(0.0, |k| self.key(bases, k))
    }

    fn output_of(&self, bases: &BTreeMap<u32, PictureSource>, id: u32) -> f64 {
        if let Some(n) = self.corrector(id) {
            let x = self.value(bases, n.picture.unwrap_or(PictureSource::Input));
            return 1.5 * x + f64::from(id);
        }
        let j = self.join(id).expect("a node");
        let src = |i: &JoinInput| i.source.unwrap_or(PictureSource::Input);
        match j.kind {
            JoinKind::Parallel => {
                let base = bases[&id];
                let b = self.value(bases, base);
                let ins = if j.inputs.is_empty() {
                    vec![INPUT]
                } else {
                    j.inputs.iter().map(|i| self.value(bases, src(i))).collect()
                };
                let mut acc = b;
                for (i, v) in ins.iter().enumerate() {
                    let this = j.inputs.get(i).map_or(PictureSource::Input, src);
                    if this != base {
                        acc += v - b;
                    }
                }
                acc
            }
            JoinKind::Layer => {
                let first = j.inputs.first().map_or(PictureSource::Input, src);
                let mut acc = self.value(bases, first);
                for i in j.inputs.iter().skip(1) {
                    let graded = match src(i) {
                        PictureSource::Node(n) if i.graded && self.corrector(n).is_some() => {
                            Some(n)
                        }
                        _ => None,
                    };
                    let (layer, alpha) = match graded {
                        Some(n) => {
                            let x = self.value(
                                bases,
                                self.corrector(n)
                                    .unwrap()
                                    .picture
                                    .unwrap_or(PictureSource::Input),
                            );
                            (1.5 * x + f64::from(n) + GRADED, self.key(bases, n))
                        }
                        None => (self.value(bases, src(i)), NO_ALPHA),
                    };
                    acc = acc * 0.5 + layer + alpha;
                }
                acc
            }
            JoinKind::Splitter => unreachable!("a Splitter has no single output"),
            JoinKind::Combiner => {
                let v =
                    |k: usize| self.value(bases, j.inputs.get(k).map_or(PictureSource::Input, src));
                v(0) + 2.0 * v(1) + 3.0 * v(2)
            }
        }
    }

    /// The value the Output reads.
    fn direct(&self, plan: &Plan) -> f64 {
        self.value(&plan.bases, self.graph.output)
    }
}

fn model(nodes: Vec<PlanNode>, joins: Vec<PlanJoin>, output: PictureSource) -> Model {
    Model {
        graph: PlanGraph { nodes, output },
        joins,
    }
}

fn check(m: &Model) -> Plan {
    let plan = m.compile(Stop::Output);
    assert_eq!(run(&plan), m.direct(&plan), "{plan:#?}");
    plan
}

use PictureSource::{Input, Node, Split};

#[test]
fn the_base_is_the_last_picture_every_branch_shares() {
    // One branching node: Main -> Skin and Sky -> Parallel. The base is Main's picture.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![plain(4, 2.0, JoinKind::Parallel, &[Node(2), Node(3)])],
        Node(4),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&4], Node(1));
    // Neither input is the base.
    let step = plan.steps().find(|s| s.node == 4).unwrap();
    assert_eq!(step.join.unwrap().is_base, 0);

    // Branches that leave different places: Input -> Main -> A -> P.1 and Input -> C -> P.2. The base is the Input's picture.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node(3, [1.0, 1.0], Input, None),
        ],
        vec![plain(4, 2.0, JoinKind::Parallel, &[Node(2), Node(3)])],
        Node(4),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&4], Input);
    assert_eq!(
        plan.steps()
            .find(|s| s.node == 4)
            .unwrap()
            .join
            .unwrap()
            .base,
        Some(PictureRef::Input)
    );

    // Branches through a Splitter: two outputs of one Splitter share only the value that feeds it.
    let m = model(
        vec![
            main_node(),
            node_at(3, 2.0, Split(2, 0)),
            node_at(4, 2.0, Split(2, 1)),
        ],
        vec![
            plain(2, 1.0, JoinKind::Splitter, &[Node(1)]),
            plain(5, 3.0, JoinKind::Parallel, &[Node(3), Node(4)]),
        ],
        Node(5),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&5], Node(1));

    // An input that is the base: Main -> P.1 and Main -> Sky -> P.2. Its change is skipped.
    let m = model(
        vec![main_node(), node_at(2, 1.0, Node(1))],
        vec![plain(3, 2.0, JoinKind::Parallel, &[Node(1), Node(2)])],
        Node(3),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&3], Node(1));
    assert_eq!(
        plan.steps()
            .find(|s| s.node == 3)
            .unwrap()
            .join
            .unwrap()
            .is_base,
        0b01
    );

    // One input: that input is the base, so the sum is the input itself.
    let m = model(
        vec![main_node(), node_at(2, 1.0, Node(1))],
        vec![plain(3, 2.0, JoinKind::Parallel, &[Node(2)])],
        Node(3),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&3], Node(2));
    assert_eq!(
        plan.steps()
            .find(|s| s.node == 3)
            .unwrap()
            .join
            .unwrap()
            .is_base,
        0b1
    );

    // No linked input: a mixer with one input, the Input's picture, which is its own base.
    let m = model(
        vec![main_node()],
        vec![join(2, 1.0, JoinKind::Parallel, vec![])],
        Node(2),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&2], Input);
    assert_eq!(
        plan.steps()
            .find(|s| s.node == 2)
            .unwrap()
            .join
            .unwrap()
            .is_base,
        0b1
    );
}

#[test]
fn nested_mixers_find_their_bases() {
    // P1 = (A, B) from Main; P2 = (P1, C) with C from Main: the base of P2 is Main.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
            node_at(5, 3.0, Node(1)),
        ],
        vec![
            plain(4, 2.0, JoinKind::Parallel, &[Node(2), Node(3)]),
            plain(6, 4.0, JoinKind::Parallel, &[Node(4), Node(5)]),
        ],
        Node(6),
    );
    let plan = check(&m);
    assert_eq!((plan.bases[&4], plan.bases[&6]), (Node(1), Node(1)));
    // P2 = (P1, D) with D read from P1: the base of P2 is P1, and its first input is the base.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
            node_at(5, 3.0, Node(4)),
        ],
        vec![
            plain(4, 2.0, JoinKind::Parallel, &[Node(2), Node(3)]),
            plain(6, 4.0, JoinKind::Parallel, &[Node(4), Node(5)]),
        ],
        Node(6),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&6], Node(4));
    assert_eq!(
        plan.steps()
            .find(|s| s.node == 6)
            .unwrap()
            .join
            .unwrap()
            .is_base,
        0b01
    );
    // A Layer in between does not change what a branch shares: Main -> L(A, B) -> C, and Main -> D; P(C, D) has base Main.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
            node_at(5, 3.0, Node(4)),
            node(6, [3.0, 1.0], Node(1), None),
        ],
        vec![
            plain(4, 2.0, JoinKind::Layer, &[Node(2), Node(3)]),
            plain(7, 4.0, JoinKind::Parallel, &[Node(5), Node(6)]),
        ],
        Node(7),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&7], Node(1));
}

#[test]
fn a_layer_of_two_branches_is_one_pass_and_reads_the_graded_value_and_the_key() {
    // Main -> Warmth(2) -> Layer.1, Main -> Glow(3) -> Layer.2 with Glow's graded value and its key m.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![join(
            4,
            2.0,
            JoinKind::Layer,
            vec![(Some(Node(2)), false), (Some(Node(3)), true)],
        )],
        Node(4),
    );
    let plan = check(&m);
    assert_eq!(plan.passes.len(), 1);
    assert!(plan.buffers.is_empty());
    let steps: Vec<_> = plan.steps().collect();
    let glow = steps.iter().find(|s| s.node == 3).unwrap();
    assert!(glow.graded_out.is_some(), "Glow makes a graded value");
    assert!(glow.key_out.is_some(), "and its key");
    let warmth = steps.iter().find(|s| s.node == 2).unwrap();
    assert!(warmth.graded_out.is_none() && warmth.key_out.is_none());
    let layer = steps.iter().find(|s| s.node == 4).unwrap().join.unwrap();
    assert!(layer.alphas[0].is_none() && layer.alphas[1].is_some());
    // The graded value and the picture are two registers, and the Layer's second input is the graded one.
    assert_ne!(
        glow.graded_out.unwrap().register,
        glow.picture_out.and_then(|o| o.register)
    );
    // Not graded: the input is the linked picture and has no alpha; the graded value is not made.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![plain(4, 2.0, JoinKind::Layer, &[Node(2), Node(3)])],
        Node(4),
    );
    let plan = check(&m);
    assert!(
        plan.steps()
            .all(|s| s.graded_out.is_none() && s.key_out.is_none())
    );
    // A graded input whose source is not a corrector is read as its picture.
    let m = model(
        vec![main_node(), node_at(2, 1.0, Node(1))],
        vec![
            plain(3, 2.0, JoinKind::Combiner, &[Node(1), Node(2), Node(1)]),
            join(
                4,
                3.0,
                JoinKind::Layer,
                vec![(Some(Node(2)), false), (Some(Node(3)), true)],
            ),
        ],
        Node(4),
    );
    let plan = check(&m);
    assert!(plan.steps().all(|s| s.graded_out.is_none()));
    // The first input is never graded.
    let m = model(
        vec![main_node(), node_at(2, 1.0, Node(1))],
        vec![join(3, 2.0, JoinKind::Layer, vec![(Some(Node(2)), true)])],
        Node(3),
    );
    let plan = check(&m);
    assert!(plan.steps().all(|s| s.graded_out.is_none()));
}

#[test]
fn a_splitter_and_a_combiner_move_channels_between_corrector_chains() {
    // Main -> S; S.r -> R, S.g -> G, S.b -> B; Combiner(R, G, B).
    let m = model(
        vec![
            main_node(),
            node(3, [2.0, -1.0], Split(2, 0), None),
            node(4, [2.0, 0.0], Split(2, 1), None),
            node(5, [2.0, 1.0], Split(2, 2), None),
        ],
        vec![
            plain(2, 1.0, JoinKind::Splitter, &[Node(1)]),
            plain(6, 3.0, JoinKind::Combiner, &[Node(3), Node(4), Node(5)]),
        ],
        Node(6),
    );
    let plan = check(&m);
    assert_eq!(plan.passes.len(), 1);
    // Cards: the three correctors and Main are numbered, the join nodes are not.
    let cards: Vec<(u32, Option<u32>)> = plan.cards.iter().map(|c| (c.id, c.number)).collect();
    assert_eq!(
        cards,
        vec![
            (1, Some(1)),
            (2, None),
            (3, Some(2)),
            (4, Some(3)),
            (5, Some(4)),
            (6, None)
        ]
    );
    // The Splitter writes three registers and no picture of its own.
    let s = plan.steps().find(|s| s.node == 2).unwrap();
    assert!(s.picture_out.is_none());
    assert!(s.join.unwrap().channels.iter().all(Option::is_some));
    // A Splitter output may feed the Output, and a Combiner input with no link reads the Input.
    let m = model(
        vec![main_node()],
        vec![plain(2, 1.0, JoinKind::Splitter, &[Node(1)])],
        Split(2, 1),
    );
    check(&m);
    let m = model(
        vec![main_node()],
        vec![join(
            2,
            1.0,
            JoinKind::Combiner,
            vec![(Some(Node(1)), false)],
        )],
        Node(2),
    );
    check(&m);
    // A thumbnail stop at a Splitter is its red.
    let m = model(
        vec![main_node()],
        vec![plain(2, 1.0, JoinKind::Splitter, &[Node(1)])],
        Node(1),
    );
    let plan = m.compile(Stop::Picture(2));
    assert_eq!(run(&plan), 2.0 * m.value(&plan.bases, Node(1)));
}

#[test]
fn a_parallel_of_eight_branches_and_its_base_needs_two_passes_and_reads_buffers() {
    let mut nodes = vec![main_node()];
    for i in 0..8 {
        nodes.push(node(2 + i, [1.0, f64::from(i)], Node(1), None));
    }
    let sources: Vec<PictureSource> = (0..8).map(|i| Node(2 + i)).collect();
    let m = model(
        nodes,
        vec![plain(20, 2.0, JoinKind::Parallel, &sources)],
        Node(20),
    );
    let plan = check(&m);
    assert_eq!(plan.bases[&20], Node(1));
    assert_eq!(
        plan.passes.len(),
        2,
        "nine pictures alive do not fit eight registers"
    );
    let p = plan.steps().find(|s| s.node == 20).unwrap().join.unwrap();
    let buffered = p
        .inputs()
        .iter()
        .chain(&p.base)
        .filter(|r| matches!(r, PictureRef::Buffer(_)))
        .count();
    assert!(
        buffered >= 1,
        "the join reads some inputs from buffers: {p:?}"
    );
    // Seven branches and the base fit one pass.
    let nodes = std::iter::once(main_node())
        .chain((0..7).map(|i| node(2 + i, [1.0, f64::from(i)], Node(1), None)))
        .collect();
    let sources: Vec<PictureSource> = (0..7).map(|i| Node(2 + i)).collect();
    let m = model(
        nodes,
        vec![plain(20, 2.0, JoinKind::Parallel, &sources)],
        Node(20),
    );
    assert_eq!(check(&m).passes.len(), 1);
}

#[test]
fn a_dead_join_is_in_the_cards_and_not_in_the_steps() {
    // Main -> A -> Output; a Layer of (A, B) and a Splitter nobody reads are dead.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![
            plain(4, 2.0, JoinKind::Layer, &[Node(2), Node(3)]),
            plain(5, 2.0, JoinKind::Splitter, &[Node(2)]),
        ],
        Node(2),
    );
    let plan = check(&m);
    assert_eq!(plan.node_ids(), vec![1, 2]);
    let live: Vec<(u32, bool)> = plan.cards.iter().map(|c| (c.id, c.live)).collect();
    assert_eq!(
        live,
        vec![(1, true), (2, true), (3, false), (4, false), (5, false)]
    );
    // A live mixer makes the chains of all its inputs live.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(2)),
            node_at(4, 1.0, Node(1)),
        ],
        vec![plain(5, 3.0, JoinKind::Layer, &[Node(3), Node(4)])],
        Node(5),
    );
    let plan = check(&m);
    assert_eq!(plan.node_ids(), vec![1, 2, 3, 4, 5]);
    assert!(plan.cards.iter().all(|c| c.live));
    // The stop on a dead join takes its ancestors.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![plain(4, 2.0, JoinKind::Layer, &[Node(2), Node(3)])],
        Node(2),
    );
    let plan = m.compile(Stop::Picture(4));
    assert_eq!(plan.node_ids(), vec![1, 2, 3, 4]);
    assert_eq!(run(&plan), m.value(&plan.bases, Node(4)));
}

#[test]
fn the_pictures_a_node_reads_are_those_of_every_input_of_the_mixers_before_it() {
    // Main -> A, Main -> B -> Layer(A, B) -> C. C (a Film node in the real graph) reads both branches.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
            node_at(5, 3.0, Node(4)),
        ],
        vec![join(
            4,
            2.0,
            JoinKind::Layer,
            vec![(Some(Node(2)), false), (Some(Node(3)), true)],
        )],
        Node(5),
    );
    let plan = check(&m);
    let ids = plan.node_ids();
    let at = |id: u32| ids.iter().position(|&n| n == id).unwrap();
    let ancestors = plan.picture_ancestors();
    assert_eq!(ancestors.len(), ids.len());
    let of_c: Vec<u32> = ancestors[at(5)].iter().map(|&i| ids[i]).collect();
    assert_eq!(of_c, vec![1, 2, 3, 4], "both branches, the Layer and Main");
    // A and B are not each other's ancestors.
    assert_eq!(ancestors[at(2)], vec![at(1)]);
    assert_eq!(ancestors[at(3)], vec![at(1)]);
    // The Layer reads its inputs and Glow's key (itself an ancestor already).
    let of_layer: Vec<u32> = ancestors[at(4)].iter().map(|&i| ids[i]).collect();
    assert_eq!(of_layer, vec![1, 2, 3]);
    // Through a Parallel's base: Main -> A -> P.1, Main -> B -> P.2: P reads A, B and Main.
    let m = model(
        vec![
            main_node(),
            node_at(2, 1.0, Node(1)),
            node_at(3, 1.0, Node(1)),
        ],
        vec![plain(4, 2.0, JoinKind::Parallel, &[Node(2), Node(3)])],
        Node(4),
    );
    let plan = check(&m);
    let ancestors = plan.picture_ancestors();
    assert_eq!(ancestors[3], vec![0, 1, 2]);
    // Through a Splitter's channel and a Combiner.
    let m = model(
        vec![main_node(), node_at(3, 2.0, Split(2, 0))],
        vec![
            plain(2, 1.0, JoinKind::Splitter, &[Node(1)]),
            plain(
                4,
                3.0,
                JoinKind::Combiner,
                &[Node(3), Split(2, 1), Split(2, 2)],
            ),
        ],
        Node(4),
    );
    let plan = check(&m);
    let ids = plan.node_ids();
    let ancestors = plan.picture_ancestors();
    let of_combiner: Vec<u32> = ancestors[ids.len() - 1].iter().map(|&i| ids[i]).collect();
    assert_eq!(of_combiner, vec![1, 2, 3]);
}

#[test]
fn two_files_listing_one_graph_in_different_orders_give_one_plan_with_joins() {
    let nodes = vec![
        main_node(),
        node_at(2, 1.0, Node(1)),
        node_at(3, 1.0, Node(1)),
    ];
    let joins = vec![
        join(
            4,
            2.0,
            JoinKind::Layer,
            vec![(Some(Node(2)), false), (Some(Node(3)), true)],
        ),
        plain(5, 3.0, JoinKind::Splitter, &[Node(4)]),
        plain(
            6,
            4.0,
            JoinKind::Combiner,
            &[Split(5, 0), Split(5, 1), Split(5, 2)],
        ),
    ];
    let forward = model(nodes.clone(), joins.clone(), Node(6)).compile(Stop::Output);
    let mut nodes = nodes;
    nodes.reverse();
    let mut joins = joins;
    joins.reverse();
    let reversed = model(nodes, joins, Node(6)).compile(Stop::Output);
    assert_eq!(forward, reversed);
}

#[test]
fn bad_joins_are_errors() {
    let m = model(
        vec![main_node()],
        vec![plain(
            2,
            1.0,
            JoinKind::Layer,
            &(0..9).map(|_| Node(1)).collect::<Vec<_>>(),
        )],
        Node(2),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Output),
        Err(PlanError::TooManyInputs(2))
    );
    let m = model(
        vec![main_node()],
        vec![plain(2, 1.0, JoinKind::Layer, &[Node(1), Node(9)])],
        Node(2),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Output),
        Err(PlanError::UnknownNode(9))
    );
    let m = model(
        vec![main_node()],
        vec![plain(2, 1.0, JoinKind::Combiner, &[Split(1, 3)])],
        Node(2),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Output),
        Err(PlanError::UnknownNode(1))
    );
    // A join has no mask to stop at.
    let m = model(
        vec![main_node()],
        vec![plain(2, 1.0, JoinKind::Layer, &[Node(1)])],
        Node(2),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Mask(2)),
        Err(PlanError::NoMask(2))
    );
    // A node id used by a corrector and a join.
    let m = model(
        vec![main_node()],
        vec![plain(1, 1.0, JoinKind::Layer, &[Input])],
        Node(1),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Output),
        Err(PlanError::DuplicateNode(1))
    );
    // A circle through a join.
    let m = model(
        vec![node_at(1, 0.0, Node(3))],
        vec![
            plain(2, 1.0, JoinKind::Parallel, &[Node(1)]),
            plain(3, 2.0, JoinKind::Layer, &[Node(2)]),
        ],
        Node(3),
    );
    assert_eq!(
        compile_with(&m.graph, &m.joins, Stop::Output),
        Err(PlanError::Cycle)
    );
}

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// Random graphs with every kind of join, fans and keys: every one compiles, its registers hold, and the interpreter's answer
/// is the graph's direct value at the Output and at the stop of every node.
#[test]
fn random_graphs_with_joins_match_direct_evaluation() {
    let mut passes_seen = BTreeMap::new();
    let mut joins_seen = BTreeMap::new();
    for seed in 0..600u64 {
        let mut rng = Lcg(seed * 7919 + 13);
        let count = 3 + rng.below(18) as usize;
        let mut nodes = vec![main_node()];
        let mut joins: Vec<PlanJoin> = Vec::new();
        // The picture values a later node may read.
        let mut values: Vec<PictureSource> = vec![Input, Node(1)];
        let mut correctors: Vec<u32> = vec![1];
        for i in 0..count {
            let id = 2 + i as u32;
            let x = f64::from(id);
            let pick = |rng: &mut Lcg| values[rng.below(values.len() as u64) as usize];
            match rng.below(6) {
                0 | 1 => {
                    let key = (rng.below(4) == 0)
                        .then(|| correctors[rng.below(correctors.len() as u64) as usize]);
                    nodes.push(PlanNode {
                        id,
                        pos: [x, 0.0],
                        picture: Some(pick(&mut rng)),
                        key,
                    });
                    correctors.push(id);
                    values.push(Node(id));
                }
                2 | 3 => {
                    let n = 1 + rng.below(8) as usize;
                    let kind = if rng.below(2) == 0 {
                        JoinKind::Parallel
                    } else {
                        JoinKind::Layer
                    };
                    let inputs = (0..n)
                        .map(|k| {
                            let s = pick(&mut rng);
                            let graded = k > 0
                                && rng.below(2) == 0
                                && matches!(s, Node(c) if correctors.contains(&c));
                            (Some(s), graded)
                        })
                        .collect();
                    joins.push(join(id, x, kind, inputs));
                    values.push(Node(id));
                }
                4 => {
                    let s = pick(&mut rng);
                    joins.push(plain(id, x, JoinKind::Splitter, &[s]));
                    values.extend([Split(id, 0), Split(id, 1), Split(id, 2)]);
                }
                _ => {
                    let ins: Vec<_> = (0..3)
                        .map(|_| (rng.below(5) != 0).then(|| pick(&mut rng)))
                        .map(|s| (s, false))
                        .collect();
                    joins.push(join(id, x, JoinKind::Combiner, ins));
                    values.push(Node(id));
                }
            }
        }
        // The Output reads the last value that is not the Input.
        let output = *values.iter().rev().find(|v| **v != Input).unwrap();
        let m = model(nodes, joins, output);
        let plan = m.compile(Stop::Output);
        assert_eq!(run(&plan), m.direct(&plan), "seed {seed}: {plan:#?}");
        *passes_seen.entry(plan.passes.len()).or_insert(0) += 1;
        for j in &m.joins {
            *joins_seen.entry(format!("{:?}", j.kind)).or_insert(0) += 1;
        }
        // And a thumbnail stop at some node: its picture, wherever the node is.
        let target = m
            .graph
            .nodes
            .iter()
            .map(|n| n.id)
            .nth(rng.below(m.graph.nodes.len() as u64) as usize)
            .unwrap();
        let stopped = m.compile(Stop::Picture(target));
        assert_eq!(
            run(&stopped),
            m.value(&stopped.bases, Node(target)),
            "seed {seed} stop {target}"
        );
        // The cards number the correctors only.
        let numbers: Vec<u32> = plan.cards.iter().filter_map(|c| c.number).collect();
        assert_eq!(
            numbers,
            (1..=m.graph.nodes.len() as u32).collect::<Vec<_>>()
        );
    }
    assert!(joins_seen.len() == 4, "{joins_seen:?}");
    assert!(
        passes_seen.keys().any(|&p| p > 1),
        "no random graph needed a second pass: {passes_seen:?}"
    );
}
