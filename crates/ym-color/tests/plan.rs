//! The plan (`ym_color::plan`, `docs/23-nodes.md` section 5.2, `docs/algorithms/06-masks.md` section
//! 1.3): live and dead nodes, the order and the card numbers, registers and passes, and the stops.
//!
//! A small interpreter runs a compiled plan on numbers (a node turns a picture `x` into `1.5 x + id`
//! and a key `k` into `(x + id) / 4 + k`) and its answer is compared with the graph evaluated
//! directly, so that a wrong register, a buffer read before it is written or a value that dies too
//! early shows as a different number.

use std::collections::BTreeMap;

use ym_color::plan::{
    BufferKind, KeyRef, MAIN_NODE, PLAN_KEYS, PLAN_REGISTERS, PictureRef, PictureSource, Plan,
    PlanError, PlanGraph, PlanNode, PlanResult, Stop, compile,
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
    };
    1.5 * x + f64::from(id)
}

/// The node's key output, evaluated straight from the graph.
fn direct_key(g: &PlanGraph, id: u32) -> f64 {
    let n = g.nodes.iter().find(|n| n.id == id).unwrap();
    let x = match n.picture.unwrap_or(PictureSource::Input) {
        PictureSource::Input => INPUT,
        PictureSource::Node(m) => direct_picture(g, m),
    };
    (x + f64::from(id)) / 4.0 + n.key.map_or(0.0, |k| direct_key(g, k))
}

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
            let x = match step.picture {
                PictureRef::Input => INPUT,
                PictureRef::Register(r) => registers[usize::from(r)],
                PictureRef::Buffer(b) => buffers[usize::from(b)],
            };
            assert!(!x.is_nan(), "{step:?} reads a picture that was not written");
            let k = step.key.map(|k| match k {
                KeyRef::Slot(s) => slots[usize::from(s)],
                KeyRef::Buffer(b) => buffers[usize::from(b)],
            });
            assert!(
                k.is_none_or(|k| !k.is_nan()),
                "{step:?} reads a key that was not written"
            );
            let id = f64::from(step.node);
            if let Some(o) = step.picture_out {
                let v = 1.5 * x + id;
                if let Some(r) = o.register {
                    registers[usize::from(r)] = v;
                }
                if let Some(b) = o.buffer {
                    assert_eq!(plan.buffers[usize::from(b)], BufferKind::Picture);
                    buffers[usize::from(b)] = v;
                }
            }
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
        (1, 1, true)
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
    let numbers: Vec<(u32, u32, bool)> =
        all.cards.iter().map(|c| (c.id, c.number, c.live)).collect();
    assert_eq!(
        numbers,
        vec![
            (1, 1, true),
            (4, 2, false),
            (5, 3, false),
            (3, 4, false),
            (2, 5, true),
            (6, 6, true)
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
    let numbers: BTreeMap<u32, u32> = plan.cards.iter().map(|c| (c.id, c.number)).collect();
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
