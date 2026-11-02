//! The nodes on the CPU reference (`docs/23-nodes.md` sections 1.3 and 1.4, `docs/algorithms/06-masks.md`
//! section 1): the default graph is the sequence of pipeline 0.8.2, the curve runs in front of the look
//! (pipeline 0.9.0), a node's steps run in their order, nodes run in the plan's order, and a bypassed
//! node, a dead branch and a node at its identity change no bit.
//!
//! Masks are not rendered until N-10: every node's mask is 1 here.

use std::collections::BTreeMap;
use std::sync::Arc;

use ym_color::cdl::Cdl;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::Mat3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ChainParams, ColorChain, DisplayError, Frame, LookSetting, MainNode, NodeParams, NodeSet, Trim,
    View,
};
use ym_color::gamut::LinearAwg4;
use ym_color::grade::{ColorBalance, Grade, GradeParams, grade_luma};
use ym_color::logc::LogC4;
use ym_color::look::{LookInput, LookTransform, blend_logc4};
use ym_color::lut::{Lut, Lut3d};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, PlanResult, Stop, compile};
use ym_color::tone::ToneParams;
use ym_color::warp::{Warp, WarpParams, WarpPin};

// ---- helpers -----------------------------------------------------------------------------------

fn fused_matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ]
    .map(|row| row.map(|v| v * gain))
}

fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).unwrap()
}

/// Camera pixels over many stops, saturated colours, near black and highlights.
fn pixels() -> Vec<Vec3> {
    let mut v = vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.001, 0.001, 0.001],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.5, 1.0],
        [12.0, 9.0, 6.0],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
        [-0.004, 0.01, 0.02],
    ];
    // SplitMix64, seeded.
    let mut state = 0x00C0_FFEE_u64;
    let mut unit = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..60 {
        let scale = 0.18 * 2.0_f64.powf(-7.0 + 11.0 * unit());
        v.push([
            scale * (0.3 + 1.4 * unit()),
            scale * (0.3 + 1.4 * unit()),
            scale * (0.3 + 1.4 * unit()),
        ]);
    }
    v
}

fn cube(size: usize, f: impl FnMut(Vec3) -> Vec3) -> Lut {
    Lut::from_cube(Lut3d::bake(size, [0.0; 3], [1.0; 3], f).unwrap())
}

/// A LogC4 look with a CDL and a LUT of our own (the LUT is a smooth non-linear tweak).
fn look() -> Arc<LookTransform> {
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).unwrap();
    let lut = cube(17, |c| {
        [
            (c[0] * 1.05 + 0.02).min(1.0) * 0.98 + 0.01 * c[1],
            c[1] * 0.97 + 0.03 * c[2] * c[2],
            (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
        ]
    });
    Arc::new(LookTransform::new(LookInput::LogC4, Some(cdl), Some(lut)).unwrap())
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
        pins: vec![
            WarpPin {
                range: 0.25,
                ..WarpPin::moving([0.05, 0.0], [0.05, 0.08])
            },
            WarpPin::locked_at([-0.15, 0.1]),
        ],
    }
}

fn curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).unwrap(),
        r: Curve::new(&[(0.0, 0.02), (0.5, 0.52), (1.0, 1.0)]).unwrap(),
        g: Curve::identity(),
        b: Curve::new(&[(0.0, 0.0), (0.4, 0.36), (1.0, 0.97)]).unwrap(),
    }
}

fn s_curve() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.25, 0.18), (0.6, 0.66), (1.0, 1.0)]).unwrap(),
        ..CurveSet::default()
    }
}

fn base() -> ChainParams {
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = grade();
    p.warp = warp();
    p
}

fn chain(p: &ChainParams) -> ColorChain {
    ColorChain::new(p).unwrap()
}

fn max_diff(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
}

/// C1 to C8 of a camera pixel as the chain does them.
fn c8_of(p: &ChainParams, px: Vec3) -> (Vec3, LogC4) {
    let c6 = ym_color::consts::linalg::mul_vec(&p.matrix, px);
    let c7 = p.tone.apply_rgb(c6);
    (c7, LinearAwg4(c7).to_logc4())
}

/// The chain of 0.8.2 for Main alone, from the finished stage modules: C9, C9b, then C10 (the look,
/// blended), then C11 (the curves).
fn legacy_logc4(p: &ChainParams, px: Vec3) -> LogC4 {
    let (c7, c8) = c8_of(p, px);
    let c9 = Grade::new(&p.grade).apply(c8, grade_luma(LinearAwg4(c7)));
    let c9b = Warp::new(&p.warp).apply(c9);
    let c10 = match &p.look {
        Some(s) if !s.look.includes_display_rendering() => {
            s.look.apply_logc4(c9b, s.intensity_pct).unwrap()
        }
        _ => c9b,
    };
    p.curves.bake().apply(c10)
}

/// The chain of 0.9.0 for Main alone: C9, C9b, C11, then C10.
fn current_logc4(p: &ChainParams, px: Vec3) -> LogC4 {
    let (c7, c8) = c8_of(p, px);
    let c9 = Grade::new(&p.grade).apply(c8, grade_luma(LinearAwg4(c7)));
    let c9b = Warp::new(&p.warp).apply(c9);
    let c11 = p.curves.bake().apply(c9b);
    match &p.look {
        Some(s) if !s.look.includes_display_rendering() => {
            s.look.apply_logc4(c11, s.intensity_pct).unwrap()
        }
        _ => c11,
    }
}

// ---- the default graph is the chain of 0.8.2 ----------------------------------------------------

#[test]
fn the_default_graph_renders_the_bytes_of_0_8_2_when_the_curve_or_the_look_is_an_identity() {
    // The configurations of 0.8.2 whose order of C10 and C11 does not matter: an identity curve, or no
    // look that does anything (none, Neutral, an intensity of 0, a look that includes its own display
    // rendering). Every one is bit for bit the old sequence.
    let neutral = Arc::new(LookTransform::neutral());
    let mut configs: Vec<(&str, ChainParams)> = Vec::new();
    for (name, curves, look) in [
        (
            "look, identity curve",
            CurveSet::default(),
            Some(LookSetting::new(look(), 80.0)),
        ),
        (
            "look 100, identity curve",
            CurveSet::default(),
            Some(LookSetting::new(look(), 100.0)),
        ),
        ("curve, no look", curves(), None),
        (
            "curve, Neutral",
            curves(),
            Some(LookSetting::new(Arc::clone(&neutral), 100.0)),
        ),
        (
            "curve, look at 0",
            curves(),
            Some(LookSetting::new(look(), 0.0)),
        ),
        ("no curve, no look", CurveSet::default(), None),
    ] {
        let mut p = base();
        p.curves = curves;
        p.look = look;
        configs.push((name, p));
    }
    for (name, p) in &configs {
        let c = chain(p);
        for px in pixels() {
            let want = legacy_logc4(p, px);
            assert_eq!(c.apply_logc4_pixel(px), want, "{name}: {px:?}");
            // The finished picture is the display transform and the encode of that value.
            let linear = c.display_linear_from_logc4(want);
            assert_eq!(c.apply_to_display_linear(px), linear, "{name}: {px:?}");
            assert_eq!(c.apply_pixel(px), c.encode(linear), "{name}: {px:?}");
            // And the DNG is its decode.
            assert_eq!(c.apply_dng_pixel(px), want.to_linear().0, "{name}: {px:?}");
        }
    }
}

#[test]
fn a_curve_with_a_look_runs_the_curve_first() {
    // The one case that changed in 0.9.0 (docs/23 section 1.4): a curve that is not the identity with a
    // look that does something. 0.8.2 gave look(then curve); 0.9.0 gives curve(then look).
    for intensity in [100.0, 80.0, 35.0] {
        let mut p = base();
        p.curves = curves();
        p.look = Some(LookSetting::new(look(), intensity));
        let c = chain(&p);
        let mut moved = 0usize;
        let mut largest = 0.0f64;
        for px in pixels() {
            let new = current_logc4(&p, px);
            assert_eq!(c.apply_logc4_pixel(px), new, "{px:?} at {intensity} %");
            // The trace says the same: C11 is the curve of C9b, C10 the look of C11.
            let t = c.trace(px);
            let c11 = p.curves.bake().apply(t.c9b_logc4);
            assert_eq!(t.c11_logc4, c11);
            assert_eq!(
                t.c10_logc4,
                blend_logc4(c11, look().eval_logc4(c11).unwrap(), intensity)
            );
            let old = legacy_logc4(&p, px);
            let d = max_diff(new.0, old.0);
            largest = largest.max(d);
            if d > 1e-3 {
                moved += 1;
            }
        }
        assert!(
            moved > 40,
            "only {moved} of {} pixels differ from the old order at {intensity} %",
            pixels().len()
        );
        assert!(
            largest > 5e-3,
            "the orders differ by at most {largest} at {intensity} %"
        );
    }
}

#[test]
fn a_look_that_includes_display_rendering_still_runs_after_the_curve() {
    // Its path is unchanged: C12 reads the curve's output, and the look has not run at C10.
    let lut = cube(17, |c| [c[0].powf(1.1), c[1] * 0.95 + 0.02, c[2].powf(0.9)]);
    let baked = Arc::new(LookTransform::new(LookInput::LogC3ToRec709, None, Some(lut)).unwrap());
    let mut p = base();
    p.curves = curves();
    p.look = Some(LookSetting::new(baked, 100.0));
    let c = chain(&p);
    for px in pixels() {
        let t = c.trace(px);
        assert_eq!(t.c10_logc4, t.c11_logc4);
        assert_eq!(t.c11_logc4, p.curves.bake().apply(t.c9b_logc4));
        assert_eq!(c.apply_logc4_pixel(px), t.c11_logc4);
    }
}

// ---- nodes ------------------------------------------------------------------------------------

/// A corrector of a test graph.
#[derive(Clone)]
struct Spec {
    id: u32,
    pos: [f64; 2],
    picture: PictureSource,
    key: Option<u32>,
    params: NodeParams,
}

fn spec(id: u32, x: f64, from: u32, params: NodeParams) -> Spec {
    Spec {
        id,
        pos: [x, 0.0],
        picture: PictureSource::Node(from),
        key: None,
        params,
    }
}

fn trim_node(ev: f64) -> NodeParams {
    NodeParams {
        trim: Trim {
            exposure_ev: ev,
            ..Trim::default()
        },
        ..NodeParams::default()
    }
}

fn curve_node() -> NodeParams {
    NodeParams {
        curves: s_curve(),
        ..NodeParams::default()
    }
}

fn grade_node() -> NodeParams {
    NodeParams {
        grade: GradeParams {
            contrast: 35.0,
            saturation: -20.0,
            ..GradeParams::default()
        },
        ..NodeParams::default()
    }
}

/// The chain for `base` (Main's fields) and the nodes after it, the Output reading `output`.
fn nodes_chain(base: &ChainParams, nodes: Vec<Spec>, output: u32, stop: Stop) -> ColorChain {
    let mut graph = vec![PlanNode {
        id: MAIN_NODE,
        pos: [0.0, 0.0],
        picture: Some(PictureSource::Input),
        key: None,
    }];
    let mut others = BTreeMap::new();
    for s in nodes {
        graph.push(PlanNode {
            id: s.id,
            pos: s.pos,
            picture: Some(s.picture),
            key: s.key,
        });
        others.insert(s.id, s.params);
    }
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(output),
        },
        stop,
    )
    .unwrap();
    let mut p = base.clone();
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    ColorChain::new(&p).unwrap()
}

/// What one node does to a LogC4 value, from the stage modules.
fn by_hand(node: &NodeParams, v: LogC4) -> LogC4 {
    let mut v = v;
    let mut lin = v.to_linear().0;
    if let Some(g) = node.trim.gain() {
        lin = [lin[0] * g[0], lin[1] * g[1], lin[2] * g[2]];
        v = LinearAwg4(lin).to_logc4();
    }
    let c9 = Grade::new(&node.grade).apply(v, grade_luma(LinearAwg4(lin)));
    let c9b = Warp::new(&node.warp).apply(c9);
    node.curves.bake().apply(c9b)
}

/// The trace-free value after Main alone (the chain's default graph) as a LogC4 triple.
fn after_main(p: &ChainParams, px: Vec3) -> LogC4 {
    let (c7, c8) = c8_of(p, px);
    let c9 = Grade::new(&p.grade).apply(c8, grade_luma(LinearAwg4(c7)));
    p.curves.bake().apply(Warp::new(&p.warp).apply(c9))
}

#[test]
fn nodes_run_in_the_plans_order_each_with_its_own_parameters() {
    let b = base();
    let c = nodes_chain(
        &b,
        vec![
            spec(2, 1.0, 1, trim_node(1.0)),
            spec(3, 2.0, 2, curve_node()),
            spec(4, 3.0, 3, grade_node()),
        ],
        4,
        Stop::Output,
    );
    for px in pixels() {
        let main = after_main(&b, px);
        let n2 = by_hand(&trim_node(1.0), main);
        let n3 = by_hand(&curve_node(), n2);
        let n4 = by_hand(&grade_node(), n3);
        assert_eq!(c.apply_logc4_pixel(px), n4, "{px:?}");
    }
}

#[test]
fn node_order_matters() {
    // An exposure trim and an S curve do not commute.
    let b = base();
    let a_then_b = nodes_chain(
        &b,
        vec![
            spec(2, 1.0, 1, trim_node(1.0)),
            spec(3, 2.0, 2, curve_node()),
        ],
        3,
        Stop::Output,
    );
    // The same two nodes, the curve first: the ids stay, the links and places change.
    let b_then_a = nodes_chain(
        &b,
        vec![
            spec(3, 1.0, 1, curve_node()),
            spec(2, 2.0, 3, trim_node(1.0)),
        ],
        2,
        Stop::Output,
    );
    let mut differs = 0usize;
    for px in pixels() {
        let (x, y) = (
            a_then_b.apply_logc4_pixel(px),
            b_then_a.apply_logc4_pixel(px),
        );
        let main = after_main(&b, px);
        assert_eq!(x, by_hand(&curve_node(), by_hand(&trim_node(1.0), main)));
        assert_eq!(y, by_hand(&trim_node(1.0), by_hand(&curve_node(), main)));
        if max_diff(x.0, y.0) > 1e-3 {
            differs += 1;
        }
    }
    assert!(
        differs > 40,
        "the swapped nodes differ on only {differs} pixels"
    );
}

#[test]
fn a_bypassed_node_equals_no_node() {
    let b = base();
    let bypassed = NodeParams {
        bypass: true,
        trim: Trim {
            exposure_ev: 2.0,
            temperature: 60.0,
            tint: -30.0,
        },
        ..grade_node()
    };
    let with = nodes_chain(
        &b,
        vec![spec(2, 1.0, 1, bypassed), spec(3, 2.0, 2, trim_node(0.5))],
        3,
        Stop::Output,
    );
    let without = nodes_chain(&b, vec![spec(3, 2.0, 1, trim_node(0.5))], 3, Stop::Output);
    for px in pixels() {
        assert_eq!(with.trace(px).c13_encoded, without.trace(px).c13_encoded);
        assert_eq!(with.apply_logc4_pixel(px), without.apply_logc4_pixel(px));
    }
    // Main can be bypassed too: it passes the Input's picture through.
    let mut p = b.clone();
    p.nodes.main.bypass = true;
    let main_off = chain(&p);
    let mut plain = ChainParams::new(fused_matrix(), tone());
    plain.look = None;
    let plain = chain(&plain);
    for px in pixels() {
        assert_eq!(main_off.apply_logc4_pixel(px), plain.apply_logc4_pixel(px));
        // The trace reports Main's stages as unchanged.
        let t = main_off.trace(px);
        assert_eq!(t.c8_logc4, t.c9_logc4);
        assert_eq!(t.c9_logc4, t.c11_logc4);
    }
}

#[test]
fn a_node_at_its_defaults_changes_no_bit() {
    let b = base();
    let one = chain(&b);
    let mut nodes = Vec::new();
    for (i, id) in (2..=5).enumerate() {
        nodes.push(spec(
            id,
            f64::from(id),
            if i == 0 { 1 } else { id - 1 },
            NodeParams::default(),
        ));
    }
    let many = nodes_chain(&b, nodes, 5, Stop::Output);
    assert_eq!(many.params().nodes.plan.step_count(), 5);
    // A trim of zero is no trim: the sign of the zero does not matter either.
    let zero = NodeParams {
        trim: Trim {
            exposure_ev: -0.0,
            temperature: 0.0,
            tint: -0.0,
        },
        ..NodeParams::default()
    };
    let zeros = nodes_chain(&b, vec![spec(2, 1.0, 1, zero)], 2, Stop::Output);
    for px in pixels() {
        let want = one.trace(px);
        assert_eq!(many.trace(px), want);
        assert_eq!(zeros.trace(px), want);
        assert_eq!(many.apply_pixel(px), one.apply_pixel(px));
    }
}

#[test]
fn a_dead_branch_changes_nothing() {
    let b = base();
    let clean = nodes_chain(
        &b,
        vec![
            spec(2, 1.0, 1, trim_node(1.0)),
            spec(3, 2.0, 2, curve_node()),
        ],
        3,
        Stop::Output,
    );
    // Nodes 4 and 5 read Main and node 4 and go nowhere; their parameters are strong.
    let dead = nodes_chain(
        &b,
        vec![
            spec(2, 1.0, 1, trim_node(1.0)),
            spec(3, 2.0, 2, curve_node()),
            spec(4, 1.0, 1, trim_node(-3.0)),
            spec(5, 2.0, 4, grade_node()),
        ],
        3,
        Stop::Output,
    );
    assert_eq!(dead.params().nodes.plan.node_ids(), vec![1, 2, 3]);
    assert!(!dead.params().nodes.plan.card(5).unwrap().live);
    for px in pixels() {
        assert_eq!(dead.trace(px), clean.trace(px));
    }
}

#[test]
fn a_node_that_only_makes_a_mask_is_in_the_plan_and_changes_no_pixel_yet() {
    // Masks are not rendered until N-10: node 5 (strong parameters, its picture unused) feeds node 3's
    // key input; the picture is the same as without the link and without node 5.
    let b = base();
    let clean = nodes_chain(&b, vec![spec(3, 2.0, 1, trim_node(1.0))], 3, Stop::Output);
    let mut keyed = spec(3, 2.0, 1, trim_node(1.0));
    keyed.key = Some(5);
    let with_key = nodes_chain(
        &b,
        vec![spec(5, 1.0, 1, grade_node()), keyed],
        3,
        Stop::Output,
    );
    assert_eq!(with_key.params().nodes.plan.node_ids(), vec![1, 5, 3]);
    for px in pixels() {
        assert_eq!(with_key.trace(px), clean.trace(px));
    }
}

#[test]
fn a_plan_that_stops_at_a_node_gives_the_picture_after_it() {
    let b = base();
    let nodes = || {
        vec![
            spec(2, 1.0, 1, trim_node(1.0)),
            spec(3, 2.0, 2, curve_node()),
            spec(4, 3.0, 3, grade_node()),
        ]
    };
    let two = nodes_chain(&b, nodes()[..1].to_vec(), 2, Stop::Output);
    let stopped = nodes_chain(&b, nodes(), 4, Stop::Picture(2));
    let main_only = nodes_chain(&b, vec![], 1, Stop::Output);
    let stopped_at_main = nodes_chain(&b, nodes(), 4, Stop::Picture(1));
    for px in pixels() {
        assert_eq!(stopped.apply_logc4_pixel(px), two.apply_logc4_pixel(px));
        assert_eq!(
            stopped_at_main.apply_logc4_pixel(px),
            main_only.apply_logc4_pixel(px)
        );
    }
    // The mask view reads 1 (every mask is 1 until N-10) as a grey LogC4 triple.
    let mask = nodes_chain(&b, nodes(), 4, Stop::Mask(3));
    assert!(matches!(
        mask.params().nodes.plan.result,
        PlanResult::Mask(_)
    ));
}

#[test]
fn the_plan_needs_parameters_for_every_node_it_runs() {
    let b = base();
    let plan = compile(
        &PlanGraph {
            nodes: vec![
                PlanNode {
                    id: 1,
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
    let mut p = b;
    p.nodes.plan = plan;
    assert_eq!(
        ColorChain::new(&p).unwrap_err(),
        DisplayError::MissingNode(2)
    );
}

#[test]
fn a_plan_of_two_passes_renders_the_picture_of_the_chain_it_holds() {
    // Main feeds nine nodes (ids 2 to 10, exposure trims of different sizes), each read by a node of
    // a second row (ids 20 to 28) whose keys are chained; node 40 reads the picture of node 2 and
    // takes the last key. Nine pictures alive at once split the plan, and the picture the Output
    // reads, node 2's, crosses the split through a buffer: the result is the chain Main, 2, 40.
    let b = base();
    let mut nodes = Vec::new();
    for i in 0..9u32 {
        nodes.push(Spec {
            id: 2 + i,
            pos: [1.0, f64::from(i)],
            picture: PictureSource::Node(1),
            key: None,
            params: trim_node(0.1 * f64::from(i + 1)),
        });
    }
    for i in 0..9u32 {
        nodes.push(Spec {
            id: 20 + i,
            pos: [2.0, f64::from(i)],
            picture: PictureSource::Node(2 + i),
            key: (i > 0).then(|| 20 + i - 1),
            params: grade_node(),
        });
    }
    nodes.push(Spec {
        id: 40,
        pos: [3.0, 0.0],
        picture: PictureSource::Node(2),
        key: Some(28),
        params: curve_node(),
    });
    let wide = nodes_chain(&b, nodes, 40, Stop::Output);
    let plan = &wide.params().nodes.plan;
    assert_eq!(plan.node_ids().len(), 1 + 9 + 9 + 1);
    assert_eq!(plan.passes.len(), 2);
    assert!(plan.passes.iter().all(|p| p.registers <= 8));
    assert!(!plan.buffers.is_empty());
    let chain3 = nodes_chain(
        &b,
        vec![
            spec(2, 1.0, 1, trim_node(0.1)),
            spec(40, 2.0, 2, curve_node()),
        ],
        40,
        Stop::Output,
    );
    for px in pixels() {
        assert_eq!(wide.trace(px), chain3.trace(px), "{px:?}");
    }
}

// ---- the whole image ---------------------------------------------------------------------------

fn three_nodes() -> ColorChain {
    let mut b = base();
    b.curves = curves();
    b.look = Some(LookSetting::new(look(), 90.0));
    nodes_chain(
        &b,
        vec![
            spec(
                2,
                1.0,
                1,
                NodeParams {
                    trim: Trim {
                        exposure_ev: 0.5,
                        temperature: 30.0,
                        tint: -12.0,
                    },
                    ..grade_node()
                },
            ),
            spec(3, 2.0, 2, curve_node()),
            spec(4, 3.0, 3, trim_node(-0.25)),
        ],
        4,
        Stop::Output,
    )
}

fn test_image(w: usize, h: usize) -> Vec<f32> {
    let px = pixels();
    let mut v = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let p = px[(i * 7 + i / w) % px.len()];
        v.extend(p.map(|c| c as f32));
    }
    v
}

#[test]
fn three_nodes_are_deterministic_whatever_the_threads_and_the_tiles() {
    let (w, h) = (96, 80);
    let data = test_image(w, h);
    let src = Frame::interleaved(w, h, &data).unwrap();
    let c = three_nodes();
    let mut whole = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, (0, 0), &mut whole, 1)
        .unwrap();
    for threads in [0, 2, 4] {
        let mut again = vec![0_u8; 3 * w * h];
        c.render_u8(View::Final, &src, (0, 0), &mut again, threads)
            .unwrap();
        assert_eq!(again, whole, "{threads} threads");
    }
    // A tile that does not start on a multiple of the 64 px noise period.
    let (x0, y0, tw, th) = (21, 13, 50, 41);
    let mut tile_px = Vec::with_capacity(3 * tw * th);
    for y in y0..y0 + th {
        tile_px.extend_from_slice(&data[3 * (y * w + x0)..3 * (y * w + x0 + tw)]);
    }
    let tile = Frame::interleaved(tw, th, &tile_px).unwrap();
    let mut tile_out = vec![0_u8; 3 * tw * th];
    c.render_u8(View::Final, &tile, (x0, y0), &mut tile_out, 0)
        .unwrap();
    for y in 0..th {
        let a = &tile_out[3 * y * tw..3 * (y + 1) * tw];
        let b = &whole[3 * ((y0 + y) * w + x0)..3 * ((y0 + y) * w + x0 + tw)];
        assert_eq!(a, b, "row {y}");
    }
    // The nodes really did something to it.
    let mut plain = three_nodes().params().clone();
    plain.nodes = NodeSet::default();
    let plain = chain(&plain);
    let mut other = vec![0_u8; 3 * w * h];
    plain
        .render_u8(View::Final, &src, (0, 0), &mut other, 0)
        .unwrap();
    assert!(other != whole);
}

#[test]
fn the_chain_clones_and_rebuilds_with_its_nodes() {
    let c = three_nodes();
    let rebuilt = ColorChain::new(c.params()).unwrap();
    let cloned = c.clone();
    for px in pixels() {
        assert_eq!(rebuilt.trace(px), c.trace(px));
        assert_eq!(cloned.trace(px), c.trace(px));
    }
}
