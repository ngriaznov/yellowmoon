//! A Colour Uniform node with a grade, lowered to two steps (`docs/32` section 12a.4, variant W2): the pull with the node's masks,
//! then the corrector's steps with the pull's mask as their key. Where the mask is 0 or 1 it is the single blend
//! `v + m (grade(compress(v)) - v)`; where it is between, the two blends differ. This measures by how much, in display colour.

use std::collections::BTreeMap;

use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::display::{
    ChainParams, ColorChain, MainNode, NodeParams, NodeSet, Trim, UniformStep, View,
};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;
use ym_color::uniform::{Amounts, Target};

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

fn steps() -> NodeParams {
    NodeParams {
        trim: Trim {
            exposure_ev: 0.5,
            temperature: 12.0,
            tint: -8.0,
        },
        grade: GradeParams {
            contrast: 20.0,
            brightness: 10.0,
            shadows: 15.0,
            highlights: -20.0,
            balance_midtones: ColorBalance { a: 0.0, b: 0.1 },
            saturation: -20.0,
            ..GradeParams::default()
        },
        curves: CurveSet {
            luma: Curve::new(&[(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)]).unwrap(),
            ..CurveSet::default()
        },
        ..NodeParams::default()
    }
}

fn pull(masks: NodeMasks) -> NodeParams {
    NodeParams {
        uniform: Some(UniformStep {
            target: Some(Target::from_stored(26.0, 9.7, 0.0)),
            amounts: Amounts::from_percent(80.0, 80.0, 40.0),
        }),
        masks,
        ..NodeParams::default()
    }
}

fn plan_node(id: u32, picture: u32, key: Option<u32>) -> PlanNode {
    PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(if picture == 0 {
            PictureSource::Input
        } else {
            PictureSource::Node(picture)
        }),
        key,
    }
}

fn chain(nodes: Vec<(u32, u32, Option<u32>, NodeParams)>) -> ColorChain {
    let mut graph = vec![plan_node(MAIN_NODE, 0, None)];
    let mut others = BTreeMap::new();
    let mut last = 1;
    for (id, from, key, params) in nodes {
        graph.push(plan_node(id, from, key));
        others.insert(id, params);
        last = id;
    }
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(last),
        },
        Stop::Output,
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

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Display colour (sRGB-encoded) to Lab.
fn lab(v: Vec3) -> Vec3 {
    let lin = v.map(|c| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let xyz = [
        0.4124564 * lin[0] + 0.3575761 * lin[1] + 0.1804375 * lin[2],
        0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2],
        0.0193339 * lin[0] + 0.1191920 * lin[1] + 0.9503041 * lin[2],
    ];
    xyz_to_lab(xyz, d65_white_xyz())
}

fn scene(seed: u64, n: usize) -> Vec<Vec3> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let lum = 0.18 * (rng.unit() * 9.0 - 6.0).exp2();
            let strength = 10f64.powf(rng.unit() * 2.5 - 3.2);
            let hue = rng.unit() * std::f64::consts::TAU;
            let (a, b) = (strength * hue.cos(), strength * hue.sin());
            [
                lum * (1.0 + a),
                lum * (1.0 - 0.5 * a + 0.866 * b),
                lum * (1.0 - 0.5 * a - 0.866 * b),
            ]
        })
        .collect()
}

#[test]
fn the_two_blends_against_the_one_on_soft_masks() {
    let t = Target::from_stored(26.0, 9.7, 0.0);
    let masks: Vec<(&str, NodeMasks)> = vec![
        (
            "brightness range 0.3 to 0.6, softness 0.5",
            NodeMasks {
                rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                    low: 0.3,
                    high: 0.6,
                    softness: 0.5,
                })],
                invert: false,
                ..NodeMasks::default()
            },
        ),
        (
            "brightness range 0.2 to 0.9, softness 1",
            NodeMasks {
                rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                    low: 0.2,
                    high: 0.9,
                    softness: 1.0,
                })],
                invert: false,
                ..NodeMasks::default()
            },
        ),
        (
            "colour range on the target, width 0.04, softness 0.5",
            NodeMasks {
                rows: vec![MaskRow::new(MaskKind::ColorRange {
                    center: [t.c1, t.c2],
                    width: 0.04,
                    softness: 0.5,
                })],
                invert: false,
                ..NodeMasks::default()
            },
        ),
        (
            "colour range on the target, width 0.1, softness 1",
            NodeMasks {
                rows: vec![MaskRow::new(MaskKind::ColorRange {
                    center: [t.c1, t.c2],
                    width: 0.1,
                    softness: 1.0,
                })],
                invert: false,
                ..NodeMasks::default()
            },
        ),
    ];
    for (i, (what, m)) in masks.into_iter().enumerate() {
        // One blend: the node with the pull and the steps (the single-node chain, `PreparedNode::apply`).
        let one = chain(vec![(
            2,
            1,
            None,
            NodeParams {
                trim: steps().trim,
                grade: steps().grade,
                curves: steps().curves.clone(),
                ..pull(m.clone())
            },
        )]);
        // Two blends: the pull with the masks, then the steps with its mask as their key.
        let two = chain(vec![(2, 1, None, pull(m)), (3, 2, Some(2), steps())]);
        let px = scene(0xA5A5 + i as u64, 20_000);
        let mut d: Vec<f64> = px
            .iter()
            .map(|&p| {
                delta_e_2000(
                    lab(one.apply_view_pixel(View::Final, p)),
                    lab(two.apply_view_pixel(View::Final, p)),
                )
            })
            .collect();
        d.sort_by(f64::total_cmp);
        let max = d[d.len() - 1];
        let p99 = d[d.len() * 99 / 100];
        let over = d.iter().filter(|&&x| x > 1.0).count() as f64 / d.len() as f64;
        eprintln!(
            "{what}: max dE00 {max:.3}, p99 {p99:.3}, over 1: {:.4}",
            over
        );
        assert!(max.is_finite());
    }
}
