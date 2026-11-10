//! X-05 (`docs/25-mixers.md` section 4.3): the mixers, the Splitter and the Combiner in the fused colour kernel against the
//! CPU reference (`ym_color::display::mix` through `ColorChain`).
//!
//! The gates are the nodes' (`docs/25` section 7): C10, the last LogC4 value of the nodes and the look, within 2e-4 in LogC4
//! (a code beyond the photo's range, where f32 holds 7 digits, within 1e-4 of its value); the encoded picture (C13) within
//! one 10-bit code. A Splitter and a Combiner move f32 values and are exact. The scene is `camera_image` (17 stops, negative
//! samples, zeros, strong highlights), so Subtract, Difference and Multiply meet negative linear values.
//!
//! Each test prints its worst errors (`--nocapture`); the numbers are in `docs/25` section 11.

mod common;

use std::collections::BTreeMap;

use common::chains::{self, camera_image, pixel};
use common::exact::assert_bit_equal;
use common::{Rng, gpus};
use ym_color::display::{
    BlendMode, ChainParams, ColorChain, JoinParams, LookSetting, MainNode, NodeParams, NodeSet,
    OutputTarget, Trim,
};
use ym_color::grade::GradeParams;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::plan::{
    self, JoinInput, JoinKind, MAIN_NODE, PictureSource, Plan, PlanGraph, PlanJoin, PlanNode,
    compile_with,
};
use ym_gpu::cpu::RgbImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// C10 gate, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;
/// Codes up to this size are held to the absolute gate; beyond it the bound is [`BEYOND_REL`] of the value (as `gpu_nodes.rs`).
const SANE_CODE: f64 = 20.0;
/// The relative bound of a LogC4 code beyond [`SANE_CODE`].
const BEYOND_REL: f64 = 1e-4;
/// A pixel whose largest camera value is above this is "bright" (2.5 stops over grey): the scene has the 0.1 % outliers and the top of a
/// 17-stop range.
const BRIGHT_CAMERA: f64 = 1.0;
/// The absolute C10 gate of a bright pixel. A Layer subtracts what it decodes: two layers of linear value 5 to 50 that agree to
/// 1.2e-6 (one ulp of the f32 code, amplified by LogC4's 10.4 per unit of relative error) differ by 6e-5, and a result near
/// black is 8.8 codes per linear unit. The normal pixels keep [`LOGC4_GATE`].
const BRIGHT_GATE: f64 = 1e-3;
/// Codes under this (linear values more than 0.06 below black) are held to [`BELOW_BLACK_REL`] of their value.
const BELOW_BLACK: f64 = -0.5;
/// The relative bound of a code under [`BELOW_BLACK`]: the f32 code's last digit amplified by a cancellation of bright layers.
const BELOW_BLACK_REL: f64 = 5e-4;

// ---- the scene and the graphs --------------------------------------------------------------------------------

/// A scene in the range a photo has, with a tenth of a percent of very bright, nearly neutral outliers (as `gpu_nodes.rs`).
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let img = camera_image(w, h, seed);
    let n = img.plane_len();
    let mut data = img.data.clone();
    for i in (0..n).filter(|i| i % 101 == 7) {
        let lum = data[n + i].abs().max(1.0);
        data[i] = common::f16_round(0.9 * lum);
        data[n + i] = lum;
        data[2 * n + i] = common::f16_round(1.1 * lum);
    }
    RgbImage { data, ..img }
}

/// A corrector with an exposure, a contrast, a saturation and (when `masked`) a soft Brightness range, so that its `m` is
/// between 0 and 1 over the scene.
fn node_params(rng: &mut Rng, masked: bool) -> NodeParams {
    NodeParams {
        masks: if masked {
            NodeMasks {
                rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                    low: rng.range(0.2, 0.4),
                    high: rng.range(0.5, 0.8),
                    softness: 0.4,
                })],
                invert: false,
            }
        } else {
            NodeMasks::default()
        },
        bypass: false,
        trim: Trim {
            exposure_ev: rng.range(-1.2, 1.2),
            temperature: rng.range(-40.0, 40.0),
            tint: 0.0,
        },
        grade: GradeParams {
            contrast: rng.range(-50.0, 50.0),
            saturation: rng.range(-40.0, 40.0),
            brightness: rng.range(-20.0, 20.0),
            ..GradeParams::default()
        },
        ..NodeParams::default()
    }
}

fn corrector(id: u32, picture: PictureSource, key: Option<u32>) -> PlanNode {
    PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(picture),
        key,
    }
}

/// A graph being built: correctors, joins, their parameters and the Output's source.
struct Graph {
    nodes: Vec<PlanNode>,
    joins: Vec<PlanJoin>,
    params: BTreeMap<u32, NodeParams>,
    jparams: BTreeMap<u32, JoinParams>,
    output: PictureSource,
}

impl Graph {
    /// Main reading the Input, with the random parameters of `main`.
    fn new(main: NodeParams) -> Self {
        Self {
            nodes: vec![corrector(MAIN_NODE, PictureSource::Input, None)],
            joins: Vec::new(),
            params: BTreeMap::from([(MAIN_NODE, main)]),
            jparams: BTreeMap::new(),
            output: PictureSource::Node(MAIN_NODE),
        }
    }

    fn corrector(&mut self, id: u32, from: PictureSource, key: Option<u32>, p: NodeParams) {
        self.nodes.push(corrector(id, from, key));
        self.params.insert(id, p);
    }

    /// A Layer whose input 1 is `inputs[0]`; an input above it is `graded` when it comes from a corrector.
    fn layer(&mut self, id: u32, inputs: &[PictureSource], mode: BlendMode, opacity: f64) {
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, &s)| JoinInput {
                source: Some(s),
                graded: i > 0
                    && matches!(s, PictureSource::Node(n) if self.params.contains_key(&n)),
            })
            .collect();
        self.join(id, JoinKind::Layer, inputs);
        self.jparams
            .insert(id, JoinParams::layer(false, mode, opacity));
    }

    fn parallel(&mut self, id: u32, inputs: &[PictureSource]) {
        let inputs = inputs
            .iter()
            .map(|&s| JoinInput {
                source: Some(s),
                graded: false,
            })
            .collect();
        self.join(id, JoinKind::Parallel, inputs);
        self.jparams.insert(id, JoinParams::parallel(false));
    }

    fn join(&mut self, id: u32, kind: JoinKind, inputs: Vec<JoinInput>) {
        self.joins.push(PlanJoin {
            id,
            pos: [f64::from(id), 0.0],
            kind,
            inputs,
        });
    }

    fn plan(&self) -> Plan {
        let g = PlanGraph {
            nodes: self.nodes.clone(),
            output: self.output,
        };
        compile_with(&g, &self.joins, plan::Stop::Output).expect("plan")
    }

    fn chain(&self, target: OutputTarget, look: Option<LookSetting>) -> ColorChain {
        let main = self.params[&MAIN_NODE].clone();
        let mut p: ChainParams = chains::base(target);
        p.look = look;
        p.grade = main.grade;
        p.warp = main.warp;
        p.curves = main.curves;
        p.joins = self.jparams.clone();
        p.nodes = NodeSet {
            plan: self.plan(),
            main: MainNode {
                bypass: main.bypass,
                trim: main.trim,
                masks: main.masks,
            },
            others: self
                .params
                .iter()
                .filter(|&(&id, _)| id != MAIN_NODE)
                .map(|(&id, n)| (id, n.clone()))
                .collect(),
        };
        ColorChain::new(&p).expect("chain")
    }
}

fn logc4_look() -> Option<LookSetting> {
    Some(LookSetting::new(chains::logc4_look(), 70.0))
}

// ---- rendering and comparing ---------------------------------------------------------------------------------

fn render(
    dev: &common::TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    outs: &[ColorOutput],
) -> Vec<ColorImage> {
    render_as(dev, chain, img, outs, false)
}

/// As [`render`]; `fast` is the view's plain arithmetic (`PreparedChain::fast`), not the export's compensated one.
fn render_as(
    dev: &common::TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    outs: &[ColorOutput],
    fast: bool,
) -> Vec<ColorImage> {
    let (chain, img, outs) = (chain.clone(), img.clone(), outs.to_vec());
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let prepared = PreparedChain::new(ctx, &chain)?.fast(fast);
            outs.iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o))
                .collect()
        })
        .expect("GPU render")
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

/// The worst `|GPU - CPU|` of C10 (LogC4, absolute between [`BELOW_BLACK`] and [`SANE_CODE`]) and of C13 (codes), and the worst
/// relative error of the codes beyond [`SANE_CODE`] and of those under [`BELOW_BLACK`].
#[derive(Debug, Clone, Copy, Default)]
struct Worst {
    c10: f64,
    bright: f64,
    c13: f64,
    beyond_rel: f64,
    below_black: f64,
}

impl Worst {
    fn assert_within_the_gates(&self, what: &str) {
        assert!(
            self.c10 <= LOGC4_GATE,
            "{what}: C10 differs by {:.3e} (gate {LOGC4_GATE:.0e})",
            self.c10
        );
        assert!(
            self.bright <= BRIGHT_GATE,
            "{what}: C10 of a bright pixel differs by {:.3e} (gate {BRIGHT_GATE:.0e})",
            self.bright
        );
        assert!(
            self.c13 <= CODE10,
            "{what}: C13 differs by {:.3e} (gate {CODE10:.1e})",
            self.c13
        );
        assert!(
            self.below_black <= BELOW_BLACK_REL,
            "{what}: codes far below black differ by {:.3e} of their value (gate {BELOW_BLACK_REL:.0e})",
            self.below_black
        );
        assert!(
            self.beyond_rel <= BEYOND_REL,
            "{what}: codes beyond the range differ by {:.3e} of their value (gate {BEYOND_REL:.0e})",
            self.beyond_rel
        );
    }
}

fn parity(dev: &common::TestGpu, chain: &ColorChain, img: &RgbImage) -> Worst {
    parity_as(dev, chain, img, false)
}

fn parity_as(dev: &common::TestGpu, chain: &ColorChain, img: &RgbImage, fast: bool) -> Worst {
    let n = img.width * img.height;
    let gpu = render_as(
        dev,
        chain,
        img,
        &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C13)],
        fast,
    );
    let mut w = Worst::default();
    for i in 0..n {
        let cam = pixel(img, i);
        // A pixel brighter than about 2.5 stops over grey (more with the nodes' exposure): its f32 codes carry a last digit that a
        // Layer's subtraction of two such layers shows (see BRIGHT_GATE).
        let bright = cam.iter().fold(0.0f64, |m, &c| m.max(c)) > BRIGHT_CAMERA;
        let t = chain.trace(cam);
        let mut beyond = false;
        for (ch, &want) in t.c10_logc4.0.iter().enumerate() {
            let d = (f64::from(planar(&gpu[0])[ch * n + i]) - want).abs();
            if want.abs() > SANE_CODE {
                beyond = true;
                w.beyond_rel = w.beyond_rel.max(d / want.abs());
            } else if want < BELOW_BLACK {
                // Far below black: Subtract and Difference of two bright, nearly equal layers cancel to a small negative
                // linear value on the steep linear segment of LogC4 (8.8 codes per unit), so the last digit of the
                // upstream f32 codes (1.2e-6 of a linear value near code 1) shows as a few 1e-4 of a code. Held to a
                // relative bound; the display renders it black (C13 is held to its gate below).
                w.below_black = w.below_black.max(d / want.abs());
            } else if bright {
                w.bright = w.bright.max(d);
            } else {
                w.c10 = w.c10.max(d);
            }
        }
        if !beyond {
            for (ch, &want) in t.c13_encoded.iter().enumerate() {
                let d = (f64::from(planar(&gpu[1])[ch * n + i]) - want).abs();
                w.c13 = w.c13.max(d);
            }
        }
    }
    w
}

fn merge(a: Worst, b: Worst) -> Worst {
    Worst {
        c10: a.c10.max(b.c10),
        bright: a.bright.max(b.bright),
        c13: a.c13.max(b.c13),
        beyond_rel: a.beyond_rel.max(b.beyond_rel),
        below_black: a.below_black.max(b.below_black),
    }
}

// ---- the Layer -------------------------------------------------------------------------------------------------

/// Main, a corrector A and a corrector B (with a soft Brightness range, so `m` is in between) and a Layer of Main, A, B:
/// input 2 is opaque but for the Opacity, input 3 has its `m` as alpha.
fn layer_graph(rng: &mut Rng, mode: BlendMode, opacity: f64) -> Graph {
    let mut g = Graph::new(node_params(rng, false));
    g.corrector(
        2,
        PictureSource::Node(MAIN_NODE),
        None,
        node_params(rng, false),
    );
    g.corrector(3, PictureSource::Node(2), None, node_params(rng, true));
    g.layer(
        10,
        &[
            PictureSource::Node(MAIN_NODE),
            PictureSource::Node(2),
            PictureSource::Node(3),
        ],
        mode,
        opacity,
    );
    g.output = PictureSource::Node(10);
    g
}

#[test]
fn every_mode_of_a_layer_matches_the_cpu() {
    // The export's arithmetic (compensated LogC4) and the view's (plain): both are held to the same gates.
    let img = scene(61, 43, 0x2505);
    for dev in gpus() {
        for fast in [false, true] {
            let way = if fast { "view" } else { "export" };
            for mode in BlendMode::ALL {
                let mut worst = Worst::default();
                for (round, opacity) in [1.0, 0.6, 0.25].into_iter().enumerate() {
                    let mut rng = Rng(0x25_0000 + round as u64 * 17 + mode as u64);
                    let g = layer_graph(&mut rng, mode, opacity);
                    assert_eq!(
                        g.plan().passes.len(),
                        1,
                        "a Layer of two branches is one pass"
                    );
                    let look = if round == 1 { logc4_look() } else { None };
                    worst = merge(
                        worst,
                        parity_as(dev, &g.chain(OutputTarget::Viewport, look), &img, fast),
                    );
                }
                eprintln!(
                    "{:?} {way}: Layer {mode:?}: C10 {:.2e}, bright {:.2e}, C13 {:.2e}, beyond (relative) {:.2e}, below black (relative) {:.2e}",
                    dev.mode,
                    worst.c10,
                    worst.bright,
                    worst.c13,
                    worst.beyond_rel,
                    worst.below_black
                );
                worst.assert_within_the_gates(&format!("{:?} {way} Layer {mode:?}", dev.mode));
            }
        }
    }
}

#[test]
fn a_layer_reads_the_graded_value_and_the_alpha_of_a_masked_input_once() {
    // Input 2 is a corrector with a Brightness range: the Layer must take its picture before its own blend and the mask once.
    // (The CPU's `mask-once` test is `crates/ym-color/tests/mixers.rs`; here the device is held to the CPU.)
    let img = scene(53, 37, 0x2506);
    for dev in gpus() {
        let mut worst = Worst::default();
        for round in 0..6u64 {
            let mut rng = Rng(0x25_1000 + round);
            let mode = BlendMode::ALL[(round % 7) as usize];
            let mut g = Graph::new(node_params(&mut rng, false));
            g.corrector(
                2,
                PictureSource::Node(MAIN_NODE),
                None,
                node_params(&mut rng, true),
            );
            // A key link: node 3 takes node 2's mask as its own key, so the alpha is a key slot of the same pass.
            g.corrector(
                3,
                PictureSource::Node(MAIN_NODE),
                Some(2),
                node_params(&mut rng, true),
            );
            g.layer(
                10,
                &[
                    PictureSource::Input,
                    PictureSource::Node(2),
                    PictureSource::Node(3),
                ],
                mode,
                0.8,
            );
            g.output = PictureSource::Node(10);
            assert!(g.plan().has_joins());
            worst = merge(worst, parity(dev, &g.chain(OutputTarget::Srgb, None), &img));
        }
        eprintln!(
            "{:?}: Layer with masked inputs and a key: C10 {:.2e}, bright {:.2e}, C13 {:.2e}, beyond {:.2e}, below black {:.2e}",
            dev.mode, worst.c10, worst.bright, worst.c13, worst.beyond_rel, worst.below_black
        );
        worst.assert_within_the_gates(&format!("{:?} masked Layer", dev.mode));
    }
}

#[test]
fn a_layer_at_opacity_zero_with_one_input_or_bypassed_is_input_1_bit_for_bit() {
    let img = scene(41, 29, 0x2507);
    for dev in gpus() {
        let mut rng = Rng(0x25_2000);
        // The bottom is a corrector's picture, so the bits are those of a registered value, not the Input's.
        let base = |rng: &mut Rng| {
            let mut g = Graph::new(node_params(rng, false));
            g.corrector(
                2,
                PictureSource::Node(MAIN_NODE),
                None,
                node_params(rng, false),
            );
            g
        };
        let mut reference = base(&mut rng);
        reference.output = PictureSource::Node(MAIN_NODE);
        let want = render(
            dev,
            &reference.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10)],
        );
        let mut cases: Vec<(&str, Graph)> = Vec::new();
        let mut zero = base(&mut rng);
        zero.params = reference.params.clone();
        zero.params.insert(2, node_params(&mut Rng(9), true));
        zero.layer(
            10,
            &[PictureSource::Node(MAIN_NODE), PictureSource::Node(2)],
            BlendMode::Multiply,
            0.0,
        );
        cases.push(("Opacity 0", zero));
        let mut one = base(&mut rng);
        one.params = reference.params.clone();
        one.layer(10, &[PictureSource::Node(MAIN_NODE)], BlendMode::Add, 1.0);
        cases.push(("one input", one));
        let mut bypassed = base(&mut rng);
        bypassed.params = reference.params.clone();
        bypassed.layer(
            10,
            &[PictureSource::Node(MAIN_NODE), PictureSource::Node(2)],
            BlendMode::Subtract,
            1.0,
        );
        bypassed.jparams.get_mut(&10).expect("layer").bypass = true;
        cases.push(("bypassed", bypassed));
        for (what, mut g) in cases {
            g.output = PictureSource::Node(10);
            let got = render(
                dev,
                &g.chain(OutputTarget::Viewport, None),
                &img,
                &[ColorOutput::Stage(Stop::C10)],
            );
            assert_bit_equal(
                &format!("{:?}: Layer, {what}", dev.mode),
                planar(&got[0]),
                planar(&want[0]),
            );
        }
    }
}

#[test]
fn a_layer_over_the_input_with_nothing_on_top_returns_the_inputs_bits() {
    // Input 1 is the Input's picture: C10 of the Layer is C8 itself, whatever the dead layers do (no decode and encode).
    let img = scene(41, 29, 0x2508);
    for dev in gpus() {
        let mut rng = Rng(0x25_3000);
        let mut g = Graph::new(node_params(&mut rng, false));
        g.corrector(2, PictureSource::Input, None, node_params(&mut rng, true));
        g.layer(
            10,
            &[PictureSource::Input, PictureSource::Node(2)],
            BlendMode::Difference,
            0.0,
        );
        g.output = PictureSource::Node(10);
        let got = render(
            dev,
            &g.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C8)],
        );
        assert_bit_equal(
            &format!("{:?}: Layer at Opacity 0 over the Input", dev.mode),
            planar(&got[0]),
            planar(&got[1]),
        );
    }
}

// ---- the Parallel ------------------------------------------------------------------------------------------------

/// Main and `branches` correctors reading Main, a Parallel of all of them: the base is Main's picture.
fn parallel_graph(rng: &mut Rng, branches: u32) -> Graph {
    let mut g = Graph::new(node_params(rng, false));
    let mut inputs = Vec::new();
    for i in 0..branches {
        let p = node_params(rng, i % 3 == 2);
        g.corrector(2 + i, PictureSource::Node(MAIN_NODE), None, p);
        inputs.push(PictureSource::Node(2 + i));
    }
    g.parallel(40, &inputs);
    g.output = PictureSource::Node(40);
    g
}

#[test]
fn a_parallel_of_1_3_and_8_inputs_matches_the_cpu_and_8_split_the_plan() {
    let img = scene(57, 41, 0x2509);
    for dev in gpus() {
        for branches in [1u32, 3, 8] {
            let mut worst = Worst::default();
            let mut passes = 0;
            for round in 0..3u64 {
                let mut rng = Rng(0x25_4000 + u64::from(branches) * 100 + round);
                let g = parallel_graph(&mut rng, branches);
                let plan = g.plan();
                passes = plan.passes.len();
                if branches == 8 {
                    assert!(
                        passes >= 2 && !plan.buffers.is_empty(),
                        "8 branches and the base need 9 pictures"
                    );
                } else {
                    assert_eq!(passes, 1);
                }
                let look = if round == 2 { logc4_look() } else { None };
                worst = merge(
                    worst,
                    parity(dev, &g.chain(OutputTarget::Viewport, look), &img),
                );
            }
            eprintln!(
                "{:?}: Parallel of {branches} in {passes} pass(es): C10 {:.2e}, C13 {:.2e}, beyond {:.2e}",
                dev.mode, worst.c10, worst.c13, worst.beyond_rel
            );
            worst.assert_within_the_gates(&format!("{:?} Parallel of {branches}", dev.mode));
        }
    }
}

#[test]
fn a_parallel_whose_branches_start_at_different_places_and_one_that_is_the_base() {
    // Input -> Main -> A -> P.1 and Input -> C -> P.2: the base is the Input's picture (`out = A + (C - Input)`).
    // Main -> P.1 and Main -> Sky -> P.2: input 1 is the base and adds nothing.
    let img = scene(49, 35, 0x250A);
    for dev in gpus() {
        let mut rng = Rng(0x25_5000);
        let mut g = Graph::new(node_params(&mut rng, false));
        g.corrector(
            2,
            PictureSource::Node(MAIN_NODE),
            None,
            node_params(&mut rng, false),
        );
        g.corrector(3, PictureSource::Input, None, node_params(&mut rng, false));
        g.parallel(40, &[PictureSource::Node(2), PictureSource::Node(3)]);
        g.output = PictureSource::Node(40);
        let different = parity(dev, &g.chain(OutputTarget::Viewport, None), &img);

        let mut h = Graph::new(node_params(&mut rng, false));
        h.corrector(
            2,
            PictureSource::Node(MAIN_NODE),
            None,
            node_params(&mut rng, true),
        );
        h.parallel(
            40,
            &[PictureSource::Node(MAIN_NODE), PictureSource::Node(2)],
        );
        h.output = PictureSource::Node(40);
        let is_base = parity(dev, &h.chain(OutputTarget::Viewport, None), &img);
        let worst = merge(different, is_base);
        eprintln!(
            "{:?}: Parallel from different places / with the base as an input: C10 {:.2e}, C13 {:.2e}",
            dev.mode, worst.c10, worst.c13
        );
        worst.assert_within_the_gates(&format!("{:?} Parallel bases", dev.mode));
    }
}

#[test]
fn a_parallel_of_identity_branches_and_a_bypassed_one_are_the_base_bit_for_bit() {
    let img = scene(41, 29, 0x250B);
    for dev in gpus() {
        let mut rng = Rng(0x25_6000);
        let mut g = Graph::new(node_params(&mut rng, false));
        // Branches at their identity (no trims, no grade): their picture is Main's, bit for bit.
        for id in 2..=4 {
            g.corrector(
                id,
                PictureSource::Node(MAIN_NODE),
                None,
                NodeParams::default(),
            );
        }
        g.parallel(
            40,
            &[
                PictureSource::Node(2),
                PictureSource::Node(3),
                PictureSource::Node(4),
            ],
        );
        let mut want_graph = Graph::new(g.params[&MAIN_NODE].clone());
        want_graph.output = PictureSource::Node(MAIN_NODE);
        let want = render(
            dev,
            &want_graph.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10)],
        );
        g.output = PictureSource::Node(40);
        let got = render(
            dev,
            &g.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10)],
        );
        assert_bit_equal(
            &format!("{:?}: Parallel of identity branches", dev.mode),
            planar(&got[0]),
            planar(&want[0]),
        );
        // Bypassed, with real branches: the base.
        let mut b = parallel_graph(&mut rng, 3);
        b.params.insert(MAIN_NODE, g.params[&MAIN_NODE].clone());
        b.jparams.get_mut(&40).expect("parallel").bypass = true;
        let got = render(
            dev,
            &b.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10)],
        );
        assert_bit_equal(
            &format!("{:?}: bypassed Parallel", dev.mode),
            planar(&got[0]),
            planar(&want[0]),
        );
    }
}

// ---- the Splitter and the Combiner --------------------------------------------------------------------------------

/// Main, a Splitter on Main, three correctors on its outputs and a Combiner of them.
fn trio_graph(red: NodeParams, green: NodeParams, blue: NodeParams, swap: bool) -> Graph {
    let mut g = Graph::new(NodeParams::default());
    g.join(
        20,
        JoinKind::Splitter,
        vec![JoinInput {
            source: Some(PictureSource::Node(MAIN_NODE)),
            graded: false,
        }],
    );
    g.corrector(21, PictureSource::Split(20, 0), None, red);
    g.corrector(22, PictureSource::Split(20, 1), None, green);
    g.corrector(23, PictureSource::Split(20, 2), None, blue);
    let from = |n: u32| JoinInput {
        source: Some(PictureSource::Node(n)),
        graded: false,
    };
    let inputs = if swap {
        vec![from(23), from(22), from(21)]
    } else {
        vec![from(21), from(22), from(23)]
    };
    g.join(30, JoinKind::Combiner, inputs);
    g.output = PictureSource::Node(30);
    g
}

#[test]
fn a_splitter_three_identity_correctors_and_a_combiner_return_the_input_bit_for_bit() {
    let img = scene(53, 37, 0x250C);
    for dev in gpus() {
        let g = trio_graph(
            NodeParams::default(),
            NodeParams::default(),
            NodeParams::default(),
            false,
        );
        let got = render(
            dev,
            &g.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C8)],
        );
        assert_bit_equal(
            &format!("{:?}: Splitter, identity correctors, Combiner", dev.mode),
            planar(&got[0]),
            planar(&got[1]),
        );
    }
}

#[test]
fn a_channel_swap_and_graded_channels_match_the_cpu_and_a_swap_moves_the_exact_values() {
    let img = scene(53, 37, 0x250D);
    for dev in gpus() {
        let mut rng = Rng(0x25_7000);
        let mut worst = Worst::default();
        for round in 0..4 {
            let g = trio_graph(
                node_params(&mut rng, false),
                node_params(&mut rng, round % 2 == 1),
                node_params(&mut rng, false),
                round >= 2,
            );
            worst = merge(
                worst,
                parity(dev, &g.chain(OutputTarget::Viewport, None), &img),
            );
        }
        eprintln!(
            "{:?}: Splitter, graded channels, Combiner: C10 {:.2e}, C13 {:.2e}, beyond {:.2e}",
            dev.mode, worst.c10, worst.c13, worst.beyond_rel
        );
        worst.assert_within_the_gates(&format!("{:?} trio", dev.mode));

        // Identity correctors and a swapped red and blue: the exact f32 values of the input's red and blue trade places.
        let g = trio_graph(
            NodeParams::default(),
            NodeParams::default(),
            NodeParams::default(),
            true,
        );
        let got = render(
            dev,
            &g.chain(OutputTarget::Viewport, None),
            &img,
            &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C8)],
        );
        let (out, c8) = (planar(&got[0]), planar(&got[1]));
        let n = img.width * img.height;
        assert_bit_equal("swap: red", &out[..n], &c8[2 * n..3 * n]);
        assert_bit_equal("swap: green", &out[n..2 * n], &c8[n..2 * n]);
        assert_bit_equal("swap: blue", &out[2 * n..3 * n], &c8[..n]);
    }
}

// ---- a join over several passes ----------------------------------------------------------------------------------

#[test]
fn a_layer_whose_graded_values_cross_a_pass_matches_the_cpu() {
    // Seven correctors read Main and a Layer takes their graded values (inputs 2 to 8) over Main; a Parallel after it reads the
    // Layer and the seven pictures again, so fifteen values are alive when the Layer runs: the plan splits, and the Layer reads
    // some of its pictures, graded values and alphas from the buffers.
    let img = scene(45, 33, 0x250E);
    for dev in gpus() {
        let mut rng = Rng(0x25_8000);
        let mut g = Graph::new(node_params(&mut rng, false));
        let mut layer_in = vec![PictureSource::Node(MAIN_NODE)];
        let mut par_in = vec![PictureSource::Node(40)];
        for i in 0..7u32 {
            g.corrector(
                2 + i,
                PictureSource::Node(MAIN_NODE),
                None,
                node_params(&mut rng, i % 2 == 0),
            );
            layer_in.push(PictureSource::Node(2 + i));
            par_in.push(PictureSource::Node(2 + i));
        }
        g.layer(40, &layer_in, BlendMode::Add, 0.7);
        g.parallel(41, &par_in);
        g.output = PictureSource::Node(41);
        let plan = g.plan();
        assert!(plan.passes.len() >= 2, "{} passes", plan.passes.len());
        assert!(!plan.buffers.is_empty());
        let worst = parity(dev, &g.chain(OutputTarget::Viewport, logc4_look()), &img);
        eprintln!(
            "{:?}: Layer of 8 with masked inputs and a Parallel in {} passes, {} buffers: C10 {:.2e}, bright {:.2e}, C13 {:.2e}, beyond {:.2e}",
            dev.mode,
            plan.passes.len(),
            plan.buffers.len(),
            worst.c10,
            worst.bright,
            worst.c13,
            worst.beyond_rel
        );
        worst.assert_within_the_gates(&format!("{:?} Layer over passes", dev.mode));
    }
}

// ---- tiles and determinism ----------------------------------------------------------------------------------------

#[test]
fn tiles_equal_the_whole_frame_and_a_repeat_render_is_identical_with_a_layer_and_a_parallel() {
    let (w, h) = (97usize, 61usize);
    let img = scene(w, h, 0x250F);
    let mut rng = Rng(0x25_9000);
    // Main, A, B (masked), a Layer of Main, A, B and a Parallel of the Layer and a corrector C that reads the Input.
    let mut g = layer_graph(&mut rng, BlendMode::Multiply, 0.8);
    g.corrector(4, PictureSource::Input, None, node_params(&mut rng, false));
    g.parallel(40, &[PictureSource::Node(10), PictureSource::Node(4)]);
    g.output = PictureSource::Node(40);
    for dev in gpus() {
        let chain = g.chain(OutputTarget::DisplayP3, logc4_look());
        let outs = [
            ColorOutput::Rgba8 { dither: true },
            ColorOutput::Stage(Stop::C13),
        ];
        let whole = render(dev, &chain, &img, &outs);
        let again = render(dev, &chain, &img, &outs);
        assert_bit_equal("repeat render", planar(&again[1]), planar(&whole[1]));
        for tile in [24usize, 33, 64] {
            let plan = TilePlan::with_halo(w, h, tile, 0);
            let (chain2, img2) = (chain.clone(), img.clone());
            let tiles = dev
                .gpu
                .run(JobSpec::new(Priority::Export), move |ctx, token| {
                    let prepared = PreparedChain::new(ctx, &chain2)?;
                    let mut collected = Vec::new();
                    for out in outs {
                        let mut tiles = Vec::new();
                        prepared.render_tiled(ctx, &img2, &plan, out, 2, token, &mut |t, im| {
                            tiles.push((t, im));
                            Ok(())
                        })?;
                        collected.push(tiles);
                    }
                    Ok(collected)
                })
                .expect("tiled render");
            let PixelData::Rgba8(whole8) = &whole[0].data else {
                panic!("RGBA8 expected")
            };
            let mut tiled8 = vec![0u8; whole8.len()];
            for (t, im) in &tiles[0] {
                let PixelData::Rgba8(d) = &im.data else {
                    panic!("RGBA8 expected")
                };
                for y in 0..t.core.h {
                    let dst = ((t.core.y + y) * w + t.core.x) * 4;
                    tiled8[dst..dst + t.core.w * 4]
                        .copy_from_slice(&d[y * t.core.w * 4..][..t.core.w * 4]);
                }
            }
            assert_eq!(&tiled8, whole8, "{:?}: 8-bit tiles of {tile}", dev.mode);
            let wholef = planar(&whole[1]);
            let mut tiledf = vec![f32::NAN; wholef.len()];
            for (t, im) in &tiles[1] {
                let d = planar(im);
                let c = t.core;
                for ch in 0..3 {
                    for y in 0..c.h {
                        for x in 0..c.w {
                            tiledf[ch * w * h + (c.y + y) * w + c.x + x] =
                                d[ch * c.w * c.h + y * c.w + x];
                        }
                    }
                }
            }
            assert_bit_equal(
                &format!("{:?}: f32 tiles of {tile}", dev.mode),
                &tiledf,
                wholef,
            );
        }
    }
}
