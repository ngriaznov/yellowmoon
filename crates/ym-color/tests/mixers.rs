//! The mixers on the CPU reference (`docs/25-mixers.md` section 3 and 7): the formulas of the seven modes against values worked
//! by hand, a Parallel's sum, the identities that must hold bit for bit, the mask applied once, and a Splitter and a Combiner.
//!
//! The first group tests the pure functions of `ym_color::display::mix`; the second runs whole chains (a plan with joins, the
//! nodes' parameters and `ColorChain`) on camera pixels.

use std::collections::BTreeMap;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::display::mix::{self, Source};
use ym_color::display::{
    BlendMode, ChainParams, ColorChain, JoinParams, MainNode, NodeParams, NodeSet, Trim,
};
use ym_color::grade::GradeParams;
use ym_color::logc::{LogC4, logc4_decode, logc4_encode, logc4_encode_rgb};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoFrame, PhotoSpot, PixelAt};
use ym_color::plan::{
    JoinInput, JoinKind, MAIN_NODE, PictureSource, PlanGraph, PlanJoin, PlanNode, Stop,
    compile_with,
};
use ym_color::tone::ToneParams;

// ---- the formulas ----------------------------------------------------------------------------

const A: f64 = 0.18;
const B: f64 = 0.09;

fn src(linear: f64) -> Source {
    Source {
        logc4: [logc4_encode(linear); 3],
        linear: None,
    }
}

/// The result of a two-layer stack in code values, channel 0.
fn stack(mode: BlendMode, a: f64, b: f64, alpha: f64) -> f64 {
    layer1(mode, a, b, alpha)[0]
}

fn layer1(mode: BlendMode, a: f64, b: f64, alpha: f64) -> Vec3 {
    mix::layer(mode, alpha, &[src(a), src(b)], &[None, None])
}

#[test]
fn each_mode_is_its_formula_on_the_spec_table() {
    // `docs/25` section 3.3: a = 0.18 (code 0.2784), b = 0.09 (code 0.2256).
    let table = [
        (BlendMode::Normal, 0.09, 0.2256),
        (BlendMode::Add, 0.27, 0.3119),
        (BlendMode::Subtract, 0.09, 0.2256),
        (BlendMode::Multiply, 0.0162, 0.1347),
        (BlendMode::Lighten, 0.18, 0.2784),
        (BlendMode::Darken, 0.09, 0.2256),
        (BlendMode::Difference, 0.09, 0.2256),
    ];
    for (mode, linear, code) in table {
        assert!((mode.channel(A, B) - linear).abs() < 1e-15, "{mode:?}");
        let got = stack(mode, A, B, 1.0);
        // One decode and one encode: the code of the linear result, to 1e-12.
        assert!(
            (got - logc4_encode(linear)).abs() < 1e-12,
            "{mode:?}: {got}"
        );
        // And the code the table prints, to its four digits.
        assert!((got - code).abs() < 6e-5, "{mode:?}: {got} against {code}");
    }
    // At alpha 0.5 Normal gives 0.135, code 0.2557.
    let half = stack(BlendMode::Normal, A, B, 0.5);
    assert!((half - logc4_encode(0.135)).abs() < 1e-12);
    assert!((half - 0.2557).abs() < 6e-5);
}

#[test]
fn modes_keep_negative_values_with_no_clamp() {
    // (a, b, mode, linear result): a negative below, a negative layer, two negatives for Multiply, and Subtract past black.
    let cases = [
        (-0.02, 0.05, BlendMode::Add, 0.03),
        (-0.02, 0.05, BlendMode::Subtract, -0.07),
        (-0.02, 0.05, BlendMode::Lighten, 0.05),
        (-0.02, 0.05, BlendMode::Darken, -0.02),
        (-0.02, 0.05, BlendMode::Difference, 0.07),
        (0.18, -0.05, BlendMode::Normal, -0.05),
        (0.18, -0.05, BlendMode::Add, 0.13),
        (0.18, -0.05, BlendMode::Multiply, -0.009),
        (-0.01, -0.05, BlendMode::Multiply, 0.0005),
        (0.18, 0.27, BlendMode::Subtract, -0.09),
    ];
    for (a, b, mode, want) in cases {
        let got = stack(mode, a, b, 1.0);
        assert!(
            (got - logc4_encode(want)).abs() < 1e-12,
            "{mode:?} {a} {b}: {got} against the code of {want}"
        );
        assert!(got.is_finite());
    }
    // Subtract 0.18 - 0.27 = -0.09 is code -0.6333 (`docs/25` section 3.3 printed -0.7213, which is the code of -0.10).
    assert!((stack(BlendMode::Subtract, 0.18, 0.27, 1.0) + 0.6333).abs() < 6e-5);
}

#[test]
fn compositing_goes_bottom_up_with_the_w3c_rule_and_skips_alpha_zero() {
    // Three layers: Add 0.05 at alpha 0.5 over 0.18, then Multiply by 2 at alpha 1 -> (0.18 + 0.5 * 0.05) * 2.
    let inputs = [src(0.18), src(0.05), src(2.0)];
    // One mode for the whole stack: Add, and alpha 0.5 for both layers.
    let got = mix::layer(BlendMode::Add, 0.5, &inputs, &[None, None, None]);
    // acc = 0.18; + 0.5 * (0.23 - 0.18) = 0.205; then + 0.5 * (2.205 - 0.205) = 1.205
    assert!((got[0] - logc4_encode(1.205)).abs() < 1e-12, "{got:?}");
    // The key of a layer multiplies the opacity: alpha = m * opacity; a layer with m = 0 is skipped (its bits do not matter).
    let skipped = mix::layer(
        BlendMode::Add,
        0.8,
        &[src(0.18), src(f64::MAX)],
        &[None, Some(0.0)],
    );
    assert_eq!(skipped, src(0.18).logc4, "alpha 0 reads nothing");
    let weighted = mix::layer(
        BlendMode::Normal,
        0.5,
        &[src(0.18), src(0.09)],
        &[None, Some(0.5)],
    );
    assert!((weighted[0] - logc4_encode(0.18 + 0.25 * (0.09 - 0.18))).abs() < 1e-12);
}

#[test]
fn a_layer_with_nothing_on_top_is_its_first_input_bit_for_bit() {
    // Opacity 0, a single input, and keys of 0: no decode and encode.
    let odd = Source {
        logc4: [0.123_456_789_012_345_67, -0.3, 0.987_654_321_098_765_4],
        linear: None,
    };
    let top = src(0.5);
    assert_eq!(
        mix::layer(BlendMode::Add, 0.0, &[odd, top], &[None, None]),
        odd.logc4
    );
    assert_eq!(mix::layer(BlendMode::Add, 1.0, &[odd], &[None]), odd.logc4);
    assert_eq!(
        mix::layer(BlendMode::Add, 1.0, &[odd, top], &[None, Some(0.0)]),
        odd.logc4
    );
}

#[test]
fn a_layer_reads_the_inputs_own_linear_values_when_it_has_them() {
    // The Input's picture: the linear values C7 made. Normal at alpha 1 gives the encode of them: exactly the input's code.
    let linear = [0.123_456_789, 0.0001, 7.5];
    let input = Source {
        logc4: logc4_encode_rgb(linear),
        linear: Some(linear),
    };
    let got = mix::layer(BlendMode::Normal, 1.0, &[src(0.18), input], &[None, None]);
    assert_eq!(got, input.logc4);
}

#[test]
fn a_parallel_adds_each_branchs_change_to_the_base() {
    // Two branches that each lift grey by one stop (code 0.2784 to 0.3365) give 0.3947.
    let base = [logc4_encode(0.18); 3];
    let up = [logc4_encode(0.36); 3];
    let got = mix::parallel(base, &[up, up], |_| false);
    assert!((got[0] - (base[0] + 2.0 * (up[0] - base[0]))).abs() < 1e-15);
    assert!((got[0] - 0.3947).abs() < 1e-4, "{}", got[0]);
    // In port order, per channel, nothing clamped: a change that leaves the usual range stays.
    let wild = [5.0, -3.0, 0.25];
    let got = mix::parallel([0.5, 0.5, 0.5], &[wild, [0.5, 0.5, 0.75]], |_| false);
    assert_eq!(got, [5.0, -3.0, 0.5 + (0.25 - 0.5) + (0.75 - 0.5)]);
    // An input that is the base is skipped whatever its floats are.
    let got = mix::parallel(base, &[up, [9.0; 3]], |i| i == 1);
    assert!((got[0] - up[0]).abs() < 1e-15);
    // One input that is the base is the base, bit for bit.
    let odd = [0.123_456_789_012_345_67, 0.3, -0.7];
    assert_eq!(mix::parallel(odd, &[odd], |_| true), odd);
    // Branches at their identity give the base: every change is exactly 0.
    assert_eq!(mix::parallel(odd, &[odd, odd, odd], |_| false), odd);
}

#[test]
fn a_splitter_and_a_combiner_move_channels() {
    let v = [0.1, 0.2, 0.3];
    assert_eq!(mix::split(v), [[0.1; 3], [0.2; 3], [0.3; 3]]);
    let [r, g, b] = mix::split(v);
    assert_eq!(mix::combine(r, g, b), v);
    // A channel swap: red from the green output, green from the red output.
    assert_eq!(mix::combine(g, r, b), [0.2, 0.1, 0.3]);
    assert_eq!(
        logc4_decode(logc4_encode(0.18)).to_bits(),
        logc4_decode(logc4_encode(0.18)).to_bits()
    );
}

// ---- whole chains ----------------------------------------------------------------------------

fn matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * gain))
}

fn base() -> ChainParams {
    let tone = ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap();
    ChainParams::new(matrix(), tone)
}

fn pixels() -> Vec<Vec3> {
    vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
        [2.0, 1.5, 1.0],
        [0.002, 0.003, 0.004],
        [12.0, 9.0, 6.0],
        [0.0, 0.0, 0.0],
    ]
}

/// A corrector that visibly changes a picture.
fn visible(ev: f64) -> NodeParams {
    NodeParams {
        trim: Trim {
            exposure_ev: ev,
            ..Trim::default()
        },
        grade: GradeParams {
            contrast: 40.0,
            saturation: -35.0,
            ..GradeParams::default()
        },
        ..NodeParams::default()
    }
}

fn identity() -> NodeParams {
    NodeParams::default()
}

use PictureSource::{Input, Node, Split};

/// A test graph: correctors (with their parameters), joins (with theirs) and what the Output reads.
#[derive(Clone, Default)]
struct Graph {
    nodes: Vec<(PlanNode, NodeParams)>,
    joins: Vec<(PlanJoin, JoinParams)>,
}

impl Graph {
    fn corrector(mut self, id: u32, from: PictureSource, params: NodeParams) -> Self {
        self.nodes.push((
            PlanNode {
                id,
                pos: [f64::from(id), 0.0],
                picture: Some(from),
                key: None,
            },
            params,
        ));
        self
    }

    fn join(
        mut self,
        id: u32,
        kind: JoinKind,
        inputs: &[(Option<PictureSource>, bool)],
        params: JoinParams,
    ) -> Self {
        self.joins.push((
            PlanJoin {
                id,
                pos: [f64::from(id), 0.0],
                kind,
                inputs: inputs
                    .iter()
                    .map(|&(source, graded)| JoinInput { source, graded })
                    .collect(),
            },
            params,
        ));
        self
    }

    fn chain(&self, output: PictureSource, stop: Stop) -> ColorChain {
        let mut nodes = vec![PlanNode {
            id: MAIN_NODE,
            pos: [0.0, 0.0],
            picture: Some(Input),
            key: None,
        }];
        let mut others = BTreeMap::new();
        for (n, p) in &self.nodes {
            nodes.push(n.clone());
            others.insert(n.id, p.clone());
        }
        let joins: Vec<PlanJoin> = self.joins.iter().map(|(j, _)| j.clone()).collect();
        let plan = compile_with(&PlanGraph { nodes, output }, &joins, stop).unwrap();
        let mut p = base();
        p.nodes = NodeSet {
            plan,
            main: MainNode::default(),
            others,
        };
        p.joins = self.joins.iter().map(|(j, q)| (j.id, *q)).collect();
        ColorChain::new(&p).unwrap()
    }

    fn render(&self, output: PictureSource) -> ColorChain {
        self.chain(output, Stop::Output)
    }
}

fn plain() -> JoinParams {
    JoinParams::parallel(false)
}

fn layer_of(mode: BlendMode, opacity: f64) -> JoinParams {
    JoinParams::layer(false, mode, opacity)
}

/// The LogC4 picture the chain gives for a camera pixel.
fn at(chain: &ColorChain, px: Vec3) -> LogC4 {
    chain.apply_logc4_pixel(px)
}

#[test]
fn a_parallel_with_one_input_is_that_input_bit_for_bit() {
    let g = Graph::default().corrector(2, Node(1), visible(1.0)).join(
        3,
        JoinKind::Parallel,
        &[(Some(Node(2)), false)],
        plain(),
    );
    let parallel = g.render(Node(3));
    let line = g.render(Node(2));
    for px in pixels() {
        assert_eq!(at(&parallel, px), at(&line, px), "{px:?}");
    }
    // And one with no linked input is the Input's picture.
    let g = Graph::default().join(2, JoinKind::Parallel, &[], plain());
    let c = g.render(Node(2));
    for px in pixels() {
        assert_eq!(at(&c, px), c.trace(px).c8_logc4, "{px:?}");
    }
}

#[test]
fn branches_at_their_identity_give_the_base_bit_for_bit() {
    let g = Graph::default()
        .corrector(2, Node(1), identity())
        .corrector(3, Node(1), identity())
        .corrector(4, Node(1), identity())
        .join(
            5,
            JoinKind::Parallel,
            &[
                (Some(Node(2)), false),
                (Some(Node(3)), false),
                (Some(Node(4)), false),
            ],
            plain(),
        );
    let parallel = g.render(Node(5));
    let main = g.render(Node(1));
    for px in pixels() {
        assert_eq!(at(&parallel, px), at(&main, px), "{px:?}");
    }
}

#[test]
fn a_parallel_sums_the_changes_of_its_branches_on_the_base() {
    // Skin (+1 EV) and Sky (-0.5 EV) from Main: out = main + (skin - main) + (sky - main), per channel.
    let g = Graph::default()
        .corrector(2, Node(1), visible(1.0))
        .corrector(3, Node(1), visible(-0.5))
        .join(
            4,
            JoinKind::Parallel,
            &[(Some(Node(2)), false), (Some(Node(3)), false)],
            plain(),
        );
    let (p, main, skin, sky) = (
        g.render(Node(4)),
        g.render(Node(1)),
        g.render(Node(2)),
        g.render(Node(3)),
    );
    for px in pixels() {
        let (m, a, b) = (at(&main, px).0, at(&skin, px).0, at(&sky, px).0);
        let got = at(&p, px).0;
        for k in 0..3 {
            let want = m[k] + (a[k] - m[k]) + (b[k] - m[k]);
            assert_eq!(got[k].to_bits(), want.to_bits(), "{px:?} channel {k}");
        }
    }
    // The base is the Input's picture for a branch fed from the start of the graph: out = A + (C - Input).
    let g = Graph::default()
        .corrector(2, Node(1), visible(1.0))
        .corrector(3, Input, visible(0.3))
        .join(
            4,
            JoinKind::Parallel,
            &[(Some(Node(2)), false), (Some(Node(3)), false)],
            plain(),
        );
    let (p, a, c) = (g.render(Node(4)), g.render(Node(2)), g.render(Node(3)));
    for px in pixels() {
        let input = p.trace(px).c8_logc4.0;
        let (a, c) = (at(&a, px).0, at(&c, px).0);
        for k in 0..3 {
            let want = input[k] + (a[k] - input[k]) + (c[k] - input[k]);
            assert_eq!(at(&p, px).0[k].to_bits(), want.to_bits(), "{px:?}");
        }
    }
}

#[test]
fn a_bypassed_mixer_outputs_its_base_or_its_first_input() {
    let g = Graph::default()
        .corrector(2, Node(1), visible(1.0))
        .corrector(3, Node(1), visible(-0.5))
        .join(
            4,
            JoinKind::Parallel,
            &[(Some(Node(2)), false), (Some(Node(3)), false)],
            JoinParams::parallel(true),
        )
        .join(
            5,
            JoinKind::Layer,
            &[(Some(Node(2)), false), (Some(Node(3)), true)],
            JoinParams::layer(true, BlendMode::Add, 1.0),
        );
    let (parallel, layer) = (g.render(Node(4)), g.render(Node(5)));
    let (main, first) = (g.render(Node(1)), g.render(Node(2)));
    for px in pixels() {
        assert_eq!(at(&parallel, px), at(&main, px));
        assert_eq!(at(&layer, px), at(&first, px));
    }
}

#[test]
fn a_layer_at_opacity_zero_with_one_input_or_a_zero_key_is_its_first_input_bit_for_bit() {
    let two = |opacity| {
        Graph::default()
            .corrector(2, Node(1), visible(1.0))
            .corrector(3, Node(1), visible(-0.5))
            .join(
                4,
                JoinKind::Layer,
                &[(Some(Node(2)), false), (Some(Node(3)), false)],
                layer_of(BlendMode::Multiply, opacity),
            )
    };
    let g = two(0.0);
    let (layer, first) = (g.render(Node(4)), g.render(Node(2)));
    for px in pixels() {
        assert_eq!(at(&layer, px), at(&first, px), "{px:?}");
    }
    let g = Graph::default().corrector(2, Node(1), visible(1.0)).join(
        3,
        JoinKind::Layer,
        &[(Some(Node(2)), false)],
        layer_of(BlendMode::Add, 1.0),
    );
    let (layer, first) = (g.render(Node(3)), g.render(Node(2)));
    for px in pixels() {
        assert_eq!(at(&layer, px), at(&first, px), "{px:?}");
    }
}

#[test]
fn a_layer_at_normal_and_full_alpha_is_its_top_input() {
    let g = Graph::default()
        .corrector(2, Node(1), visible(1.0))
        .corrector(3, Node(1), visible(-0.5))
        .join(
            4,
            JoinKind::Layer,
            &[(Some(Node(2)), false), (Some(Node(3)), false)],
            layer_of(BlendMode::Normal, 1.0),
        );
    let (layer, top) = (g.render(Node(4)), g.render(Node(3)));
    for px in pixels() {
        let (got, want) = (at(&layer, px).0, at(&top, px).0);
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "{px:?}: {got:?} against {want:?}"
            );
        }
    }
    // The top is the Input's picture: C7 taken in linear, so the encode is C8 itself.
    let g = Graph::default().corrector(2, Node(1), visible(1.0)).join(
        3,
        JoinKind::Layer,
        &[(Some(Node(2)), false), (Some(Input), false)],
        layer_of(BlendMode::Normal, 1.0),
    );
    let layer = g.render(Node(3));
    for px in pixels() {
        assert_eq!(at(&layer, px), layer.trace(px).c8_logc4, "{px:?}");
    }
}

fn spot(x: f64, y: f64) -> PixelAt<'static> {
    Some(PhotoSpot::new([x, y], PhotoFrame::new(1000.0, 1000.0, 1.0)))
}

#[test]
fn the_mask_is_applied_once_at_a_soft_edge() {
    // Glow (node 3) has a feathered Linear mask. The Layer shows Glow's picture before its own blend, through m: the soft edge
    // is applied once, by the Layer.
    let masked = NodeParams {
        masks: NodeMasks {
            rows: vec![MaskRow::new(MaskKind::Linear {
                center: [0.5, 0.5],
                angle_deg: 0.0,
                feather: 0.5,
            })],
            invert: false,
        },
        ..visible(1.0)
    };
    let g = Graph::default()
        .corrector(2, Node(1), visible(-0.3))
        .corrector(3, Node(1), masked.clone())
        .join(
            4,
            JoinKind::Layer,
            &[(Some(Node(2)), false), (Some(Node(3)), true)],
            layer_of(BlendMode::Normal, 1.0),
        );
    let layer = g.render(Node(4));
    // Glow without its mask, and its mask on its own.
    let unmasked = Graph::default()
        .corrector(2, Node(1), visible(-0.3))
        .corrector(
            3,
            Node(1),
            NodeParams {
                masks: NodeMasks::default(),
                ..masked.clone()
            },
        );
    let (bottom, v_c) = (unmasked.render(Node(2)), unmasked.render(Node(3)));
    let mask_chain = Graph::default()
        .corrector(2, Node(1), visible(-0.3))
        .corrector(3, Node(1), masked)
        .chain(Node(3), Stop::Mask(3));
    let blended = Graph::default()
        .corrector(2, Node(1), visible(-0.3))
        .corrector(
            3,
            Node(1),
            NodeParams {
                masks: NodeMasks {
                    rows: vec![MaskRow::new(MaskKind::Linear {
                        center: [0.5, 0.5],
                        angle_deg: 0.0,
                        feather: 0.5,
                    })],
                    invert: false,
                },
                ..visible(1.0)
            },
        )
        .render(Node(3));
    let mut soft = 0;
    for px in pixels() {
        for (x, y) in [(0.5, 0.5), (0.5, 0.56), (0.5, 0.62), (0.5, 0.9), (0.5, 0.1)] {
            let m = mask_chain.mask_pixel_at(px, spot(x, y))[0];
            let got = layer.apply_logc4_pixel_at(px, spot(x, y)).0;
            let b = at(&bottom, px).0;
            let lin_b = LogC4(b).to_linear().0;
            let lin_c = at(&v_c, px).to_linear().0;
            for k in 0..3 {
                let want = logc4_encode(lin_b[k] + m * (lin_c[k] - lin_b[k]));
                assert!(
                    (got[k] - want).abs() < 1e-12,
                    "{px:?} at ({x}, {y}), m {m}: {got:?}"
                );
            }
            if m > 0.05 && m < 0.95 {
                soft += 1;
                // Not a blend of Glow's already blended picture: that would apply m twice.
                let twice = blended.apply_logc4_pixel_at(px, spot(x, y)).0;
                let lin_t = LogC4(twice).to_linear().0;
                let again = logc4_encode(lin_b[0] + m * (lin_t[0] - lin_b[0]));
                assert!((got[0] - again).abs() > 1e-6 || (lin_c[0] - lin_b[0]).abs() < 1e-9);
            }
        }
    }
    assert!(soft >= 5, "the test found {soft} pixels on the soft edge");
}

#[test]
fn a_splitter_three_identity_correctors_and_a_combiner_give_the_input_bit_for_bit() {
    let g = Graph::default()
        .join(2, JoinKind::Splitter, &[(Some(Node(1)), false)], plain())
        .corrector(3, Split(2, 0), identity())
        .corrector(4, Split(2, 1), identity())
        .corrector(5, Split(2, 2), identity())
        .join(
            6,
            JoinKind::Combiner,
            &[
                (Some(Node(3)), false),
                (Some(Node(4)), false),
                (Some(Node(5)), false),
            ],
            plain(),
        );
    let (combined, main) = (g.render(Node(6)), g.render(Node(1)));
    for px in pixels() {
        assert_eq!(at(&combined, px), at(&main, px), "{px:?}");
    }
    // Red lifted by 0.4 EV, blue lowered: only those channels change, each as the corrector made it from a grey of that channel.
    let g = Graph::default()
        .join(2, JoinKind::Splitter, &[(Some(Node(1)), false)], plain())
        .corrector(3, Split(2, 0), visible(0.4))
        .corrector(4, Split(2, 1), identity())
        .corrector(5, Split(2, 2), visible(-0.4))
        .join(
            6,
            JoinKind::Combiner,
            &[
                (Some(Node(3)), false),
                (Some(Node(4)), false),
                (Some(Node(5)), false),
            ],
            plain(),
        );
    let (combined, red, blue) = (g.render(Node(6)), g.render(Node(3)), g.render(Node(5)));
    let main = g.render(Node(1));
    for px in pixels() {
        let got = at(&combined, px).0;
        assert_eq!(got[0], at(&red, px).0[0]);
        assert_eq!(got[1], at(&main, px).0[1]);
        assert_eq!(got[2], at(&blue, px).0[2]);
    }
    // A channel swap needs no corrector at all: Combiner(green, red, blue) straight off the Splitter.
    let g = Graph::default()
        .join(2, JoinKind::Splitter, &[(Some(Node(1)), false)], plain())
        .join(
            3,
            JoinKind::Combiner,
            &[
                (Some(Split(2, 1)), false),
                (Some(Split(2, 0)), false),
                (Some(Split(2, 2)), false),
            ],
            plain(),
        );
    let (swapped, main) = (g.render(Node(3)), g.render(Node(1)));
    for px in pixels() {
        let m = at(&main, px).0;
        assert_eq!(at(&swapped, px).0, [m[1], m[0], m[2]]);
    }
}

#[test]
fn a_join_whose_inputs_leave_the_registers_matches_the_one_pass_value() {
    // A Parallel of eight branches and its base: two passes, the join reads buffers. The sum, computed straight.
    let mut g = Graph::default();
    let mut inputs = Vec::new();
    for i in 0..8u32 {
        g = g.corrector(2 + i, Node(1), visible(0.2 * f64::from(i) - 0.5));
        inputs.push((Some(Node(2 + i)), false));
    }
    let g = g.join(20, JoinKind::Parallel, &inputs, plain());
    let parallel = g.render(Node(20));
    let plan = &parallel.params().nodes.plan;
    assert_eq!(plan.passes.len(), 2);
    let main = g.render(Node(1));
    let branches: Vec<ColorChain> = (0..8).map(|i| g.render(Node(2 + i))).collect();
    for px in pixels() {
        let m = at(&main, px).0;
        let got = at(&parallel, px).0;
        for k in 0..3 {
            let mut want = m[k];
            for b in &branches {
                want += at(b, px).0[k] - m[k];
            }
            assert_eq!(got[k].to_bits(), want.to_bits(), "{px:?} channel {k}");
        }
    }
}

#[test]
fn a_chain_with_joins_is_the_same_bits_on_every_call() {
    let g = Graph::default()
        .corrector(2, Node(1), visible(1.0))
        .corrector(3, Node(1), visible(-0.5))
        .join(
            4,
            JoinKind::Layer,
            &[(Some(Node(2)), false), (Some(Node(3)), true)],
            layer_of(BlendMode::Difference, 0.6),
        )
        .join(
            5,
            JoinKind::Parallel,
            &[(Some(Node(4)), false), (Some(Node(2)), false)],
            plain(),
        );
    let c = g.render(Node(5));
    let again = g.render(Node(5));
    for px in pixels() {
        let (a, b) = (at(&c, px), at(&again, px));
        assert_eq!(a, b);
        assert_eq!(a, at(&c, px));
    }
}
