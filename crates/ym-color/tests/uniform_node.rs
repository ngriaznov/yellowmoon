//! The Colour Uniform node in the chain (`docs/32` sections 3.4, 6 and 8, CU-04): the plan step, the mask blend
//! `in + m (node(in) - in)`, the skip at the identity, the seeded Colour range row, Show mask and the step kind.

use std::collections::BTreeMap;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::display::{
    ChainParams, ColorChain, MainNode, NodeParams, NodeSet, UniformStep,
};
use ym_color::logc::LogC4;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;
use ym_color::uniform::{self, Amounts, Target};

fn matrix() -> Mat3 {
    let g = 0.25_f64.exp2();
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * g))
}

fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap()
}

fn pixels() -> Vec<Vec3> {
    vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.35, 0.25, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.0, 0.0, 0.0],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
        [2.0, 1.5, 1.0],
    ]
}

fn target() -> Target {
    Target::from_stored(26.0, 9.7, 0.0)
}

fn step(t: Option<Target>, a: Amounts) -> UniformStep {
    UniformStep {
        target: t,
        amounts: a,
    }
}

fn chain(node: NodeParams) -> ColorChain {
    let plan = compile(
        &PlanGraph {
            nodes: vec![
                PlanNode {
                    id: MAIN_NODE,
                    pos: [0.0, 0.0],
                    picture: Some(PictureSource::Input),
                    key: None,
                },
                PlanNode {
                    id: 2,
                    pos: [1.0, 0.0],
                    picture: Some(PictureSource::Node(1)),
                    key: None,
                },
            ],
            output: PictureSource::Node(2),
        },
        Stop::Output,
    )
    .unwrap();
    let mut p = ChainParams::new(matrix(), tone());
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others: BTreeMap::from([(2, node)]),
    };
    ColorChain::new(&p).unwrap()
}

fn main_only() -> ColorChain {
    ColorChain::new(&ChainParams::new(matrix(), tone())).unwrap()
}

fn node(uniform: UniformStep, masks: NodeMasks) -> NodeParams {
    NodeParams {
        uniform: Some(uniform),
        masks,
        ..NodeParams::default()
    }
}

fn live() -> UniformStep {
    step(Some(target()), Amounts::from_percent(80.0, 80.0, 40.0))
}

fn near(a: Vec3, b: Vec3, tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

#[test]
fn the_node_is_the_pull_of_the_core_on_the_value_it_reads() {
    let with = chain(node(live(), NodeMasks::default()));
    let base = main_only();
    let a = Amounts::from_percent(80.0, 80.0, 40.0);
    let mut moved = 0;
    for px in pixels() {
        let v_in = base.apply_logc4_pixel(px).0;
        let want = uniform::apply(v_in, Some(&target()), a);
        let got = with.apply_logc4_pixel(px).0;
        assert_eq!(got, want, "{px:?}");
        moved += usize::from(got != v_in);
    }
    assert!(moved > 0, "the node moved no pixel");
}

#[test]
fn a_node_at_its_identity_changes_no_bit() {
    let base = main_only();
    for uniform in [
        step(None, Amounts::from_percent(80.0, 80.0, 40.0)),
        step(Some(target()), Amounts::default()),
        step(None, Amounts::default()),
    ] {
        let c = chain(node(uniform, NodeMasks::default()));
        for px in pixels() {
            assert_eq!(c.apply_logc4_pixel(px), base.apply_logc4_pixel(px));
        }
    }
    // Bypassed.
    let bypassed = NodeParams {
        bypass: true,
        ..node(live(), NodeMasks::default())
    };
    let c = chain(bypassed);
    for px in pixels() {
        assert_eq!(c.apply_logc4_pixel(px), base.apply_logc4_pixel(px));
    }
}

/// A brightness range whose mask is strictly between 0 and 1 for some of the pixels.
fn feathered() -> NodeMasks {
    NodeMasks {
        rows: vec![MaskRow::new(MaskKind::BrightnessRange {
            low: 0.3,
            high: 0.6,
            softness: 0.5,
        })],
        invert: false,
    }
}

#[test]
fn the_mask_blends_as_in_plus_m_times_node_minus_in() {
    let masked = chain(node(live(), feathered()));
    let base = main_only();
    let a = Amounts::from_percent(80.0, 80.0, 40.0);
    let shown = {
        // Show mask: the chain that ends at the node's mask.
        let plan = compile(
            &PlanGraph {
                nodes: vec![
                    PlanNode {
                        id: MAIN_NODE,
                        pos: [0.0, 0.0],
                        picture: Some(PictureSource::Input),
                        key: None,
                    },
                    PlanNode {
                        id: 2,
                        pos: [1.0, 0.0],
                        picture: Some(PictureSource::Node(1)),
                        key: None,
                    },
                ],
                output: PictureSource::Node(2),
            },
            Stop::Mask(2),
        )
        .unwrap();
        let mut p = ChainParams::new(matrix(), tone());
        p.nodes = NodeSet {
            plan,
            main: MainNode::default(),
            others: BTreeMap::from([(2, node(live(), feathered()))]),
        };
        ColorChain::new(&p).unwrap()
    };
    let (mut partial, mut zero, mut one) = (0, 0, 0);
    for px in pixels() {
        let v_in = base.apply_logc4_pixel(px).0;
        let m = shown.apply_logc4_pixel(px).0[0];
        let full = uniform::apply(v_in, Some(&target()), a);
        let want: Vec3 = std::array::from_fn(|k| v_in[k] + m * (full[k] - v_in[k]));
        let got = masked.apply_logc4_pixel(px).0;
        assert!(near(got, want, 1e-12), "{px:?}: m {m}: {got:?} against {want:?}");
        if m == 0.0 {
            zero += 1;
            assert_eq!(got, v_in);
        } else if m == 1.0 {
            one += 1;
        } else {
            partial += 1;
        }
    }
    assert!(partial > 0 && zero + one > 0, "partial {partial}, zero {zero}, one {one}");
}

#[test]
fn the_seeded_colour_range_row_is_one_at_the_picked_chroma() {
    // The first pick centres the row on the picked chroma `[a, b]` (docs/32 section 2.3): the mask is 1 there and the
    // pixel at that chroma is a fixed point of the node, so the node leaves it as it is.
    let t = target();
    let rows = NodeMasks {
        rows: vec![MaskRow::new(MaskKind::ColorRange {
            center: [t.c1, t.c2],
            width: 0.08,
            softness: 0.5,
        })],
        invert: false,
    };
    let v = t.to_rgb();
    let m = rows.prepare().mask(None, LogC4(v), 0.3, None);
    assert!((m - 1.0).abs() < 1e-12, "m {m}");
    let out = uniform::apply(v, Some(&t), Amounts::from_percent(100.0, 100.0, 100.0));
    assert!(near(out, v, 1e-12));
}

#[test]
fn the_step_is_a_corrector_like_kind_of_its_own() {
    use ym_color::schedule::{Footprint, StepKind};
    let d = StepKind::ColorUniform.declared(0);
    assert_eq!(d.footprint, Footprint::PerPixel);
    assert_eq!(d, StepKind::Corrector.declared(0));
    assert!(StepKind::ALL.contains(&StepKind::ColorUniform));
    // The chain declares the node's step with that kind, and a live Film step (none here) stays distinct.
    let s = chain(node(live(), NodeMasks::default())).schedule(1.0, &|_| None);
    assert_eq!(s.total_reach(), 0);
}
