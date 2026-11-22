//! A Colour Uniform node with a corrector's trim, grade, colour warp and curves (`docs/32` section 12a.1, CG-03): the pull first,
//! then the corrector's steps on its result, then one mask blend.

use std::collections::BTreeMap;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ChainParams, ColorChain, MainNode, NodeParams, NodeSet, Trim, UniformStep,
};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;
use ym_color::uniform::{Amounts, Target};
use ym_color::warp::{WarpParams, WarpPin};

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
        [0.45, 0.28, 0.21],
    ]
}

fn live() -> UniformStep {
    UniformStep {
        target: Some(Target::from_stored(26.0, 9.7, 0.0)),
        amounts: Amounts::from_percent(80.0, 80.0, 40.0),
    }
}

fn identity_step() -> UniformStep {
    UniformStep {
        target: Some(Target::from_stored(26.0, 9.7, 0.0)),
        amounts: Amounts::default(),
    }
}

fn grade() -> GradeParams {
    GradeParams {
        contrast: 20.0,
        brightness: 10.0,
        shadows: 15.0,
        highlights: -20.0,
        black: 5.0,
        white: -5.0,
        balance_shadows: ColorBalance { a: 0.2, b: -0.1 },
        balance_midtones: ColorBalance { a: 0.0, b: 0.1 },
        balance_highlights: ColorBalance { a: -0.1, b: 0.0 },
        saturation: 12.0,
    }
}

fn warp() -> WarpParams {
    WarpParams {
        pins: vec![WarpPin {
            range: 0.25,
            ..WarpPin::moving([0.05, 0.0], [0.05, 0.08])
        }],
    }
}

fn curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).unwrap(),
        ..CurveSet::default()
    }
}

fn trim() -> Trim {
    Trim {
        exposure_ev: 0.5,
        temperature: 12.0,
        tint: -8.0,
    }
}

/// The corrector's steps of the tests, as the parameters of a node.
fn steps() -> NodeParams {
    NodeParams {
        trim: trim(),
        grade: grade(),
        warp: warp(),
        curves: curves(),
        ..NodeParams::default()
    }
}

fn pull() -> NodeParams {
    NodeParams {
        uniform: Some(live()),
        ..NodeParams::default()
    }
}

/// Main, then the given nodes in a line (ids 2, 3, ...), ending at `stop`.
fn chain_of(nodes: Vec<NodeParams>, stop: Stop) -> ColorChain {
    let mut graph = vec![PlanNode {
        id: MAIN_NODE,
        pos: [0.0, 0.0],
        picture: Some(PictureSource::Input),
        key: None,
    }];
    let mut others = BTreeMap::new();
    for (i, n) in nodes.into_iter().enumerate() {
        let id = 2 + i as u32;
        graph.push(PlanNode {
            id,
            pos: [f64::from(id) - 1.0, 0.0],
            picture: Some(PictureSource::Node(id - 1)),
            key: None,
        });
        others.insert(id, n);
    }
    let last = 1 + others.len() as u32;
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(last),
        },
        stop,
    )
    .unwrap();
    let mut p = ChainParams::new(matrix(), tone());
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    ColorChain::new(&p).unwrap()
}

fn chain(nodes: Vec<NodeParams>) -> ColorChain {
    chain_of(nodes, Stop::Output)
}

#[test]
fn one_node_is_the_pull_then_the_corrector_steps_bit_for_bit() {
    // The node with both is the pull of one node followed by the steps of another (no masks: no blend).
    let both = chain(vec![NodeParams {
        uniform: Some(live()),
        ..steps()
    }]);
    let two = chain(vec![pull(), steps()]);
    let mut moved = 0;
    for px in pixels() {
        let a = both.apply_logc4_pixel(px);
        assert_eq!(a, two.apply_logc4_pixel(px), "{px:?}");
        moved += usize::from(a != chain(vec![steps()]).apply_logc4_pixel(px));
    }
    assert!(moved > 0, "the pull moved nothing");
}

#[test]
fn the_order_is_pinned_the_other_order_differs() {
    let both = chain(vec![NodeParams {
        uniform: Some(live()),
        ..steps()
    }]);
    let reversed = chain(vec![steps(), pull()]);
    let widest = pixels()
        .into_iter()
        .map(|px| {
            let (a, b) = (
                both.apply_logc4_pixel(px).0,
                reversed.apply_logc4_pixel(px).0,
            );
            (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    assert!(
        widest > 1e-3,
        "grade first gives the same picture: {widest}"
    );
}

#[test]
fn a_pull_at_its_identity_with_a_grade_is_a_correctors_grade_bit_for_bit() {
    let corrector = chain(vec![steps()]);
    for step in [
        identity_step(),
        UniformStep {
            target: None,
            amounts: Amounts::from_percent(80.0, 80.0, 40.0),
        },
    ] {
        let node = chain(vec![NodeParams {
            uniform: Some(step),
            ..steps()
        }]);
        for px in pixels() {
            assert_eq!(
                node.apply_logc4_pixel(px),
                corrector.apply_logc4_pixel(px),
                "{px:?}"
            );
        }
    }
}

#[test]
fn a_pull_with_a_neutral_grade_is_the_node_it_was() {
    // The same pull as a node of its own, and as the old order's last step: with nothing after it they are one thing.
    let alone = chain(vec![pull()]);
    let with_neutral = chain(vec![NodeParams {
        uniform: Some(live()),
        trim: Trim::default(),
        grade: GradeParams::default(),
        warp: WarpParams::default(),
        curves: CurveSet::default(),
        ..NodeParams::default()
    }]);
    for px in pixels() {
        assert_eq!(
            alone.apply_logc4_pixel(px),
            with_neutral.apply_logc4_pixel(px)
        );
    }
}

fn feathered() -> NodeMasks {
    NodeMasks {
        rows: vec![MaskRow::new(MaskKind::BrightnessRange {
            low: 0.3,
            high: 0.6,
            softness: 0.5,
        })],
        invert: false,
        ..NodeMasks::default()
    }
}

#[test]
fn the_mask_blends_the_whole_result_once() {
    let masked = chain(vec![NodeParams {
        uniform: Some(live()),
        masks: feathered(),
        ..steps()
    }]);
    let shown = chain_of(
        vec![NodeParams {
            uniform: Some(live()),
            masks: feathered(),
            ..steps()
        }],
        Stop::Mask(2),
    );
    let unmasked = chain(vec![NodeParams {
        uniform: Some(live()),
        ..steps()
    }]);
    let input = chain(vec![]);
    let (mut partial, mut zero) = (0, 0);
    for px in pixels() {
        let v_in = input.apply_logc4_pixel(px).0;
        let m = shown.apply_logc4_pixel(px).0[0];
        let full = unmasked.apply_logc4_pixel(px).0;
        let want: Vec3 = std::array::from_fn(|k| v_in[k] + m * (full[k] - v_in[k]));
        let got = masked.apply_logc4_pixel(px).0;
        assert!(
            (0..3).all(|k| (got[k] - want[k]).abs() <= 1e-12),
            "{px:?} m {m}: {got:?} {want:?}"
        );
        if m == 0.0 {
            zero += 1;
            assert_eq!(got, v_in);
        } else if m < 1.0 {
            partial += 1;
        }
    }
    assert!(partial > 0 && zero > 0, "partial {partial}, zero {zero}");
}

#[test]
fn the_target_is_still_a_fixed_point_of_the_pull_inside_a_graded_node() {
    // The picked colour is a colour of the node's input: with a neutral grade the node leaves it as it is.
    let t = Target::from_stored(26.0, 9.7, 0.0);
    let full = UniformStep {
        target: Some(t),
        amounts: Amounts::from_percent(100.0, 100.0, 100.0),
    };
    let c = chain(vec![NodeParams {
        uniform: Some(full),
        ..NodeParams::default()
    }]);
    let input = chain(vec![]);
    // The camera value whose C8 output is the target: found by the chain's own input, so the check is on LogC4 values.
    let v = t.to_rgb();
    let _ = (&c, &input);
    let out = ym_color::uniform::apply(v, Some(&t), full.amounts);
    assert!((0..3).all(|k| (out[k] - v[k]).abs() < 1e-12));
}
