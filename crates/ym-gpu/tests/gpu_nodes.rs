//! N-04 (`docs/23-nodes.md` section 5.4): the plan loop in the fused colour kernel against the CPU
//! reference (`ym_color::display::ColorChain` over `ym_color::plan`).
//!
//! The gates are today's (`docs/02` section Tests): C10, the last LogC4 value of the nodes and the
//! look, within 2e-4 in LogC4; the encoded picture (C13) within one 10-bit code; the DNG stop within
//! 2e-4 of its value. Graphs of 1, 4 and 16 correctors with random trims, grades, colour warps and
//! curves; bypassed nodes and nodes at their identity (which run no code: the bits of the graph
//! without them); a dead branch; a graph that needs nine pictures at once, which the plan splits into
//! two passes with f32 buffers between them; the stops inside the nodes; tiles, determinism and
//! poison. Masks are not rendered until N-10 (every node's `m` is 1).

mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::chains::{self, camera_image, pixel};
use common::exact::{assert_bit_equal, assert_poison_neutral};
use common::{Rng, gpus};
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ChainParams, ColorChain, LookSetting, MainNode, NodeParams, NodeSet, OutputTarget, Trim,
};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::plan::{
    self, MAIN_NODE, PLAN_REGISTERS, PictureSource, Plan, PlanGraph, PlanNode, compile,
};
use ym_color::warp::{WarpParams, WarpPin};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, NodeStop, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;
use ym_gpu::tiles::TilePlan;

/// C7 to C11 and C10 gate, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;

// ---- the scene -----------------------------------------------------------------------------------------

/// `camera_image` with its very bright outliers (a tenth of a percent of the pixels, colours drawn over
/// the whole gamut up to 80) made bright and nearly neutral: a stack of 16 nodes pushes a pixel that far
/// outside the gamut, where the luminance is negative, to a LogC4 code of -160, which no photo has and where
/// f32 holds a relative 5e-5 of it at best. The scene keeps the 17 stops, the negative samples, the zero
/// and the strong highlights.
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let img = camera_image(w, h, seed);
    let n = img.plane_len();
    let mut data = img.data.clone();
    for i in (0..n).filter(|i| i % 101 == 7) {
        let lum = data[n + i].abs().max(1.0);
        // Representable in f16 like every other sample, so that every storage mode reads what the CPU sees.
        data[i] = common::f16_round(0.9 * lum);
        data[n + i] = lum;
        data[2 * n + i] = common::f16_round(1.1 * lum);
    }
    RgbImage { data, ..img }
}

// ---- random nodes ----------------------------------------------------------------------------------

/// The grade of a node: each control is drawn over `scale` of its range (the chains of many nodes use a
/// smaller `scale`, so that their stack stays in the range a photo has: a pixel pushed to a LogC4 code of
/// 5 by one node and decoded by the next is 1e22 in linear light, where f32 and f64 part ways).
fn random_grade(rng: &mut Rng, scale: f64) -> GradeParams {
    let control = |rng: &mut Rng, p: f64| {
        if rng.unit() < p {
            rng.range(-100.0, 100.0) * scale
        } else {
            0.0
        }
    };
    let balance = |rng: &mut Rng| {
        if rng.unit() < 0.4 {
            let (r, a) = (
                rng.range(0.0, 1.2) * scale,
                rng.range(0.0, std::f64::consts::TAU),
            );
            ColorBalance {
                a: r * a.cos(),
                b: r * a.sin(),
            }
        } else {
            ColorBalance { a: 0.0, b: 0.0 }
        }
    };
    GradeParams {
        contrast: control(rng, 0.5),
        brightness: control(rng, 0.5),
        shadows: control(rng, 0.4),
        highlights: control(rng, 0.4),
        black: control(rng, 0.4),
        white: control(rng, 0.4),
        balance_shadows: balance(rng),
        balance_midtones: balance(rng),
        balance_highlights: balance(rng),
        saturation: control(rng, 0.5),
    }
}

/// Half the sets have a monotone curve on one to three of the four channels through two to six points.
fn random_curves(rng: &mut Rng) -> CurveSet {
    let mut set = CurveSet::default();
    if rng.unit() < 0.5 {
        return set;
    }
    for which in 0..4 {
        if rng.unit() < 0.5 {
            continue;
        }
        let k = 2 + (rng.next_u64() % 5) as usize;
        let mut xs: Vec<f64> = (0..k).map(|_| rng.unit()).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        xs.dedup_by(|a, b| (*a - *b).abs() < 0.02);
        let mut y = 0.0;
        let pts: Vec<(f64, f64)> = xs
            .iter()
            .map(|&x| {
                y = (y + rng.range(0.0, 0.3)).min(1.0);
                (x, (x * 0.7 + y * 0.3).clamp(0.0, 1.0))
            })
            .collect();
        if let Ok(c) = Curve::new(&pts) {
            match which {
                0 => set.luma = c,
                1 => set.r = c,
                2 => set.g = c,
                _ => set.b = c,
            }
        }
    }
    set
}

/// Trims (half the nodes have some): exposure to `scale` stops either way, temperature and tint over
/// `scale` of their range.
fn random_trim(rng: &mut Rng, scale: f64) -> Trim {
    let mut t = Trim::default();
    if rng.unit() < 0.5 {
        if rng.unit() < 0.6 {
            t.exposure_ev = rng.range(-1.0, 1.0) * scale;
        }
        if rng.unit() < 0.5 {
            t.temperature = rng.range(-100.0, 100.0) * scale;
        }
        if rng.unit() < 0.5 {
            t.tint = rng.range(-100.0, 100.0) * scale;
        }
    }
    t
}

/// A colour warp of up to eight pins (about a third locked, a few still) where the colours of
/// `camera_image` are, their moves and tonal controls over `scale` of the range of `chains::random_warp`.
fn random_pins(rng: &mut Rng, scale: f64) -> WarpParams {
    let count = (rng.next_u64() % 9) as usize;
    let point = |rng: &mut Rng, reach: f64| {
        let (r, a) = (
            reach * rng.unit().sqrt(),
            rng.range(0.0, std::f64::consts::TAU),
        );
        [r * a.cos(), r * a.sin()]
    };
    let pins = (0..count)
        .map(|_| {
            let source = point(rng, 0.12);
            let range = rng.range(0.02, 0.15);
            if rng.unit() < 0.3 {
                return WarpPin {
                    range,
                    ..WarpPin::locked_at(source)
                };
            }
            let offset = point(rng, 0.15 * scale);
            let target = if rng.unit() < 0.15 {
                source
            } else {
                [source[0] + offset[0], source[1] + offset[1]]
            };
            WarpPin {
                source,
                target,
                locked: false,
                range,
                shadows: rng.range(0.0, 1.0),
                highlights: rng.range(0.0, 1.0),
                pivot_ev: rng.range(-4.0, 4.0),
                exposure_ev: if rng.unit() < 0.5 {
                    rng.range(-2.0, 2.0) * scale
                } else {
                    0.0
                },
                saturation: if rng.unit() < 0.5 {
                    rng.range(-1.0, 1.0) * scale
                } else {
                    0.0
                },
            }
        })
        .collect();
    WarpParams { pins }
}

fn random_node(rng: &mut Rng, scale: f64) -> NodeParams {
    NodeParams {
        bypass: false,
        trim: random_trim(rng, scale),
        grade: random_grade(rng, scale),
        warp: if scale >= 1.0 {
            chains::random_warp(rng)
        } else {
            random_pins(rng, scale)
        },
        curves: random_curves(rng),
    }
}

/// Random nodes for every id; `scale` 1 draws over the full range of each control.
fn random_nodes_at(
    rng: &mut Rng,
    scale: f64,
    ids: impl IntoIterator<Item = u32>,
) -> BTreeMap<u32, NodeParams> {
    ids.into_iter()
        .map(|id| (id, random_node(rng, scale)))
        .collect()
}

/// Random nodes over `0.5` of the range: a graph of a few nodes in the range a photo has.
fn random_nodes(rng: &mut Rng, ids: impl IntoIterator<Item = u32>) -> BTreeMap<u32, NodeParams> {
    random_nodes_at(rng, 0.5, ids)
}

// ---- graphs ------------------------------------------------------------------------------------------

fn node(id: u32, x: f64, picture: PictureSource, key: Option<u32>) -> PlanNode {
    PlanNode {
        id,
        pos: [x, 0.0],
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

/// Main and `n - 1` correctors in a line, the last one the Output's.
fn line(n: u32) -> PlanGraph {
    let mut nodes = vec![node(MAIN_NODE, 0.0, PictureSource::Input, None)];
    for id in 2..=n {
        nodes.push(node(id, f64::from(id), PictureSource::Node(id - 1), None));
    }
    graph(nodes, n)
}

/// Main fans out to `branches` nodes (ids 2..), each read by a second node (ids 20..) whose keys are
/// chained; the last of those is the Output. All the first nodes run before the second ones, so
/// `branches` pictures are alive at once (the graph of `crates/ym-color/tests/plan.rs`).
fn fan(branches: u32) -> PlanGraph {
    let mut nodes = vec![node(MAIN_NODE, 0.0, PictureSource::Input, None)];
    for i in 0..branches {
        nodes.push(node(
            2 + i,
            1.0 + f64::from(i) * 0.001,
            PictureSource::Node(MAIN_NODE),
            None,
        ));
    }
    for i in 0..branches {
        nodes.push(node(
            20 + i,
            2.0 + f64::from(i) * 0.001,
            PictureSource::Node(2 + i),
            (i > 0).then(|| 20 + i - 1),
        ));
    }
    graph(nodes, 20 + branches - 1)
}

/// The chain parameters of a plan with the nodes' parameters: Main's are the chain's own `grade`,
/// `warp` and `curves`, every other node's are in the node set. `look` is the Output's.
fn chain_params(
    plan: Plan,
    nodes: &BTreeMap<u32, NodeParams>,
    target: OutputTarget,
    look: Option<LookSetting>,
) -> ChainParams {
    let mut p = chains::base(target);
    p.look = look;
    let main = nodes.get(&MAIN_NODE).cloned().unwrap_or_default();
    p.grade = main.grade;
    p.warp = main.warp;
    p.curves = main.curves;
    p.nodes = NodeSet {
        plan,
        main: MainNode {
            bypass: main.bypass,
            trim: main.trim,
        },
        others: nodes
            .iter()
            .filter(|&(&id, _)| id != MAIN_NODE)
            .map(|(&id, n)| (id, n.clone()))
            .collect(),
    };
    p
}

fn chain_of(
    g: &PlanGraph,
    nodes: &BTreeMap<u32, NodeParams>,
    target: OutputTarget,
    look: Option<LookSetting>,
) -> ColorChain {
    let plan = compile(g, plan::Stop::Output).expect("plan");
    ColorChain::new(&chain_params(plan, nodes, target, look)).expect("chain")
}

fn logc4_look() -> Option<LookSetting> {
    Some(LookSetting::new(chains::logc4_look(), 70.0))
}

// ---- rendering ---------------------------------------------------------------------------------------

fn render_with(
    dev: &common::TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    node_stop: Option<NodeStop>,
    outputs: &[ColorOutput],
) -> Vec<ColorImage> {
    let (chain, img, outputs) = (chain.clone(), img.clone(), outputs.to_vec());
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut prepared = PreparedChain::new(ctx, &chain)?;
            if let Some(s) = node_stop {
                prepared = prepared.with_node_stop(s)?;
            }
            outputs
                .iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o))
                .collect()
        })
        .expect("GPU render")
}

fn render(
    dev: &common::TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    outputs: &[ColorOutput],
) -> Vec<ColorImage> {
    render_with(dev, chain, img, None, outputs)
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

/// LogC4 codes up to this size are held to the absolute gate; beyond it (an out-of-gamut pixel that a
/// stack of nodes pushed far from the photo's range: a luminance that went negative makes a code of -160)
/// f32 carries seven digits, not the 2e-4 of a code near 1, and the bound is [`BEYOND_REL`] of the value.
const SANE_CODE: f64 = 20.0;
/// The relative bound of a LogC4 code beyond [`SANE_CODE`].
const BEYOND_REL: f64 = 1e-4;

/// The largest `|GPU - CPU|` of each of the stages C9, C9b, C11 (Main's), C10, C13 and the DNG stop
/// (relative to its value, 0.01 at least) over the image, for the LogC4 codes within [`SANE_CODE`];
/// the worst relative error of the codes beyond it and how many there are.
#[derive(Debug, Clone, Copy, Default)]
struct Worst {
    c9: f64,
    c9b: f64,
    c11: f64,
    c10: f64,
    c13: f64,
    dng: f64,
    beyond: usize,
    beyond_rel: f64,
}

impl Worst {
    fn max(self, o: Worst) -> Worst {
        Worst {
            c9: self.c9.max(o.c9),
            c9b: self.c9b.max(o.c9b),
            c11: self.c11.max(o.c11),
            c10: self.c10.max(o.c10),
            c13: self.c13.max(o.c13),
            dng: self.dng.max(o.dng),
            beyond: self.beyond + o.beyond,
            beyond_rel: self.beyond_rel.max(o.beyond_rel),
        }
    }

    fn assert_within_the_gates(&self, what: &str) {
        for (name, v, gate) in [
            ("C9", self.c9, LOGC4_GATE),
            ("C9b", self.c9b, LOGC4_GATE),
            ("C11", self.c11, LOGC4_GATE),
            ("C10", self.c10, LOGC4_GATE),
            ("C13", self.c13, CODE10),
            ("DNG", self.dng, LOGC4_GATE),
            (
                "codes beyond the photo's range, relative",
                self.beyond_rel,
                BEYOND_REL,
            ),
        ] {
            assert!(
                v <= gate,
                "{what}: {name} differs by {v:.3e} (gate {gate:.1e})"
            );
        }
    }
}

fn parity(dev: &common::TestGpu, chain: &ColorChain, img: &RgbImage) -> Worst {
    let n = img.width * img.height;
    let gpu = render(
        dev,
        chain,
        img,
        &[
            ColorOutput::Stage(Stop::C9),
            ColorOutput::Stage(Stop::C9b),
            ColorOutput::Stage(Stop::C11),
            ColorOutput::Stage(Stop::C10),
            ColorOutput::Stage(Stop::C13),
            ColorOutput::Stage(Stop::DngLinearAwg4),
        ],
    );
    let mut w = Worst::default();
    for i in 0..n {
        let cam = pixel(img, i);
        let t = chain.trace(cam);
        let dng = chain.apply_dng_pixel(cam);
        let logc4 = [
            (0, t.c9_logc4.0),
            (1, t.c9b_logc4.0),
            (2, t.c11_logc4.0),
            (3, t.c10_logc4.0),
        ];
        let mut beyond = false;
        for (k, want) in logc4 {
            for (ch, &want) in want.iter().enumerate() {
                let d = (f64::from(planar(&gpu[k])[ch * n + i]) - want).abs();
                if want.abs() > SANE_CODE {
                    beyond = true;
                    w.beyond_rel = w.beyond_rel.max(d / want.abs());
                    continue;
                }
                let slot = match k {
                    0 => &mut w.c9,
                    1 => &mut w.c9b,
                    2 => &mut w.c11,
                    _ => &mut w.c10,
                };
                *slot = slot.max(d);
            }
        }
        w.beyond += usize::from(beyond);
        if beyond {
            // The encoded picture and the DNG of such a pixel are clamped or huge: not compared.
            continue;
        }
        for (ch, &dng) in dng.iter().enumerate() {
            let d = |k: usize, want: f64| (f64::from(planar(&gpu[k])[ch * n + i]) - want).abs();
            w.c13 = w.c13.max(d(4, t.c13_encoded[ch]));
            w.dng = w.dng.max(d(5, dng) / dng.abs().max(0.01));
        }
    }
    w
}

// ---- parity ----------------------------------------------------------------------------------------

#[test]
fn graphs_of_1_4_and_16_correctors_match_the_cpu() {
    let (w, h) = (53usize, 37usize);
    let img = scene(w, h, 0x0404);
    for dev in gpus() {
        for (n, rounds) in [(1u32, 8u64), (4, 8), (16, 5)] {
            let mut worst = Worst::default();
            let (mut moved, mut total) = (0usize, 0usize);
            for round in 0..rounds {
                let mut rng = Rng(0xA110_0000 + u64::from(n) * 1000 + round);
                let g = line(n);
                let scale = 1.0 / f64::from(n).sqrt();
                let nodes = random_nodes_at(&mut rng, scale, 1..=n);
                let target = OutputTarget::ALL[(round % 4) as usize];
                let look = (round % 2 == 0).then(logc4_look).flatten();
                let chain = chain_of(&g, &nodes, target, look);
                assert_eq!(chain.params().nodes.plan.node_ids().len(), n as usize);
                worst = worst.max(parity(dev, &chain, &img));
                // Not vacuous: the nodes move the picture.
                let plain = chain_of(&line(1), &BTreeMap::new(), target, None);
                for i in 0..w * h {
                    let (a, b) = (
                        chain.trace(pixel(&img, i)).c10_logc4.0,
                        plain.trace(pixel(&img, i)).c10_logc4.0,
                    );
                    moved += usize::from((0..3).any(|k| (a[k] - b[k]).abs() > 1e-3));
                    total += 1;
                }
            }
            eprintln!(
                "{:?}: {rounds} random graphs of {n} correctors, worst |GPU - CPU|: C9 {:.1e}, C9b {:.1e}, C11 {:.1e}, C10 {:.1e} (LogC4; gate {LOGC4_GATE:.0e}), C13 {:.1e} (gate {CODE10:.1e}), DNG {:.1e}; {} % of the pixels move by more than 1e-3; {} pixels with a code beyond {SANE_CODE} (worst relative {:.1e}, bound {BEYOND_REL:.0e})",
                dev.mode,
                worst.c9,
                worst.c9b,
                worst.c11,
                worst.c10,
                worst.c13,
                worst.dng,
                100 * moved / total,
                worst.beyond,
                worst.beyond_rel
            );
            worst.assert_within_the_gates(&format!("{:?} {n} correctors", dev.mode));
            assert!(
                moved * 4 > total,
                "{n}: only {moved} of {total} pixels move"
            );
        }
    }
}

#[test]
fn a_node_that_reads_a_picture_decodes_it_and_one_that_reads_the_input_does_not() {
    // Two lines of the same two nodes' parameters: in one both read the Input (the second node is
    // another reader of it, the Output's), in the other the second reads the first. The luma of C9 is
    // taken from the linear values C7 made in the first case and from the decoded picture in the second;
    // both agree with the CPU, which makes the same difference.
    let img = camera_image(41, 29, 0x1337);
    let mut rng = Rng(0x2468);
    let nodes = random_nodes(&mut rng, 1..=2);
    let reads_input = graph(
        vec![
            node(1, 0.0, PictureSource::Input, None),
            node(2, 1.0, PictureSource::Input, None),
        ],
        2,
    );
    for dev in gpus() {
        for (what, g) in [
            ("line", line(2)),
            ("both read the Input", reads_input.clone()),
        ] {
            let chain = chain_of(&g, &nodes, OutputTarget::Srgb, logc4_look());
            parity(dev, &chain, &img).assert_within_the_gates(what);
        }
    }
}

#[test]
fn bypassed_nodes_and_nodes_at_their_identity_run_no_code() {
    let img = camera_image(47, 31, 0xB1B1);
    let mut rng = Rng(0xBAD5);
    let full = random_nodes(&mut rng, 1..=6);
    // The same line without nodes 3 and 5, which are bypassed in the first chain, at their identity
    // in the second and absent from the third.
    let mut bypassed = full.clone();
    for id in [3, 5] {
        bypassed.get_mut(&id).expect("node").bypass = true;
    }
    let mut identity = full.clone();
    for id in [3, 5] {
        identity.insert(id, NodeParams::default());
    }
    let short = |ids: &[u32]| {
        let mut nodes = vec![node(ids[0], 0.0, PictureSource::Input, None)];
        for pair in ids.windows(2) {
            nodes.push(node(
                pair[1],
                f64::from(pair[1]),
                PictureSource::Node(pair[0]),
                None,
            ));
        }
        graph(nodes, *ids.last().expect("ids"))
    };
    for dev in gpus() {
        for look in [None, logc4_look()] {
            let outputs = [
                ColorOutput::Stage(Stop::C10),
                ColorOutput::Stage(Stop::C13),
                ColorOutput::Rgba8 { dither: true },
            ];
            let b = render(
                dev,
                &chain_of(&line(6), &bypassed, OutputTarget::Viewport, look.clone()),
                &img,
                &outputs,
            );
            let i = render(
                dev,
                &chain_of(&line(6), &identity, OutputTarget::Viewport, look.clone()),
                &img,
                &outputs,
            );
            let reduced = render(
                dev,
                &chain_of(
                    &short(&[1, 2, 4, 6]),
                    &full,
                    OutputTarget::Viewport,
                    look.clone(),
                ),
                &img,
                &outputs,
            );
            for k in 0..2 {
                assert_bit_equal(
                    "bypassed against absent",
                    planar(&b[k]),
                    planar(&reduced[k]),
                );
                assert_bit_equal(
                    "at the identity against absent",
                    planar(&i[k]),
                    planar(&reduced[k]),
                );
            }
            assert_eq!(b[2], reduced[2], "bypassed against absent, RGBA8");
            assert_eq!(i[2], reduced[2], "at the identity against absent, RGBA8");
            // And the CPU agrees with the device on the graphs that have them.
            parity(
                dev,
                &chain_of(&line(6), &bypassed, OutputTarget::Viewport, look.clone()),
                &img,
            )
            .assert_within_the_gates("bypassed nodes");
            parity(
                dev,
                &chain_of(&line(6), &identity, OutputTarget::Viewport, look),
                &img,
            )
            .assert_within_the_gates("nodes at their identity");
        }
    }
}

#[test]
fn a_bypassed_main_passes_the_input_on_and_its_stages_are_the_inputs() {
    let img = camera_image(37, 23, 0xC0DE);
    let mut rng = Rng(0x1);
    let mut nodes = random_nodes(&mut rng, 1..=3);
    nodes.get_mut(&1).expect("main").bypass = true;
    for dev in gpus() {
        let chain = chain_of(&line(3), &nodes, OutputTarget::Srgb, logc4_look());
        let w = parity(dev, &chain, &img);
        w.assert_within_the_gates("Main bypassed");
        // Main's C9, C9b and C11 are C8 whatever follows.
        let out = render(
            dev,
            &chain,
            &img,
            &[
                ColorOutput::Stage(Stop::C8),
                ColorOutput::Stage(Stop::C9),
                ColorOutput::Stage(Stop::C9b),
                ColorOutput::Stage(Stop::C11),
            ],
        );
        for k in 1..4 {
            assert_bit_equal("a bypassed Main's stage", planar(&out[k]), planar(&out[0]));
        }
    }
}

#[test]
fn a_dead_branch_changes_no_bit_and_the_output_may_skip_main() {
    let img = camera_image(43, 27, 0xDEAD);
    let mut rng = Rng(0xD0);
    let nodes = random_nodes(&mut rng, 1..=8);
    // Nodes 7 and 8 read Main and nothing reads them: dead.
    let mut with_dead = line(4).nodes;
    with_dead.push(node(7, 7.0, PictureSource::Node(1), None));
    with_dead.push(node(8, 8.0, PictureSource::Node(7), None));
    let with_dead = graph(with_dead, 4);
    // Node 2 reads the Input and the Output reads it: Main is dead and the trace's stages are C8.
    let skip_main = graph(
        vec![
            node(1, 0.0, PictureSource::Input, None),
            node(2, 1.0, PictureSource::Input, None),
        ],
        2,
    );
    for dev in gpus() {
        let live = chain_of(&line(4), &nodes, OutputTarget::DisplayP3, logc4_look());
        let dead = chain_of(&with_dead, &nodes, OutputTarget::DisplayP3, logc4_look());
        assert_eq!(dead.params().nodes.plan.node_ids(), vec![1, 2, 3, 4]);
        let outputs = [
            ColorOutput::Stage(Stop::C10),
            ColorOutput::Stage(Stop::C13),
            ColorOutput::Rgba8 { dither: true },
        ];
        let (a, b) = (
            render(dev, &live, &img, &outputs),
            render(dev, &dead, &img, &outputs),
        );
        assert_bit_equal("a dead branch, C10", planar(&a[0]), planar(&b[0]));
        assert_bit_equal("a dead branch, C13", planar(&a[1]), planar(&b[1]));
        assert_eq!(a[2], b[2], "a dead branch, RGBA8");

        let chain = chain_of(&skip_main, &nodes, OutputTarget::Srgb, None);
        assert_eq!(chain.params().nodes.plan.node_ids(), vec![2]);
        let w = parity(dev, &chain, &img);
        w.assert_within_the_gates("Main is dead");
        let out = render(
            dev,
            &chain,
            &img,
            &[ColorOutput::Stage(Stop::C8), ColorOutput::Stage(Stop::C9)],
        );
        assert_bit_equal(
            "Main is not in the plan: its C9 is C8",
            planar(&out[0]),
            planar(&out[1]),
        );
    }
}

// ---- several passes ----------------------------------------------------------------------------------

#[test]
fn a_graph_that_needs_nine_pictures_runs_in_two_passes_and_matches_the_cpu() {
    let img = camera_image(45, 33, 0x9999);
    for dev in gpus() {
        for branches in [8u32, 9, 10, 17] {
            let g = fan(branches);
            let plan = compile(&g, plan::Stop::Output).expect("plan");
            if branches <= 8 {
                assert_eq!(plan.passes.len(), 1, "{branches} branches");
            } else {
                assert!(
                    plan.passes.len() >= 2,
                    "{branches} branches: {} passes",
                    plan.passes.len()
                );
                assert!(plan.passes.iter().all(|p| p.registers <= PLAN_REGISTERS));
                assert!(!plan.buffers.is_empty());
            }
            let ids: Vec<u32> = plan.node_ids();
            let mut rng = Rng(0x9000 + u64::from(branches));
            let nodes = random_nodes(&mut rng, ids.iter().copied());
            let chain = chain_of(&g, &nodes, OutputTarget::Viewport, logc4_look());
            let worst = parity(dev, &chain, &img);
            eprintln!(
                "{:?}: fan({branches}) in {} pass(es), {} buffers: C10 {:.1e}, C13 {:.1e}",
                dev.mode,
                plan.passes.len(),
                plan.buffers.len(),
                worst.c10,
                worst.c13
            );
            worst.assert_within_the_gates(&format!("{:?} fan({branches})", dev.mode));
        }
    }
}

/// The CPU chain of a graph's plan that ends at `stop`.
fn chain_ending_at(
    g: &PlanGraph,
    stop: plan::Stop,
    nodes: &BTreeMap<u32, NodeParams>,
) -> ColorChain {
    let plan = compile(g, stop).expect("plan");
    ColorChain::new(&chain_params(plan, nodes, OutputTarget::Srgb, None)).expect("chain")
}

#[test]
fn a_stop_after_a_node_is_the_cpu_plan_that_ends_there_and_a_mask_stop_is_grey_one() {
    let (w, h) = (31usize, 19usize);
    let img = camera_image(w, h, 0x5707);
    let n = w * h;
    for dev in gpus() {
        for g in [line(5), fan(9)] {
            let plan = compile(&g, plan::Stop::Output).expect("plan");
            let mut rng = Rng(0x57_0000 + plan.step_count() as u64);
            let nodes = random_nodes(&mut rng, plan.node_ids());
            let full = chain_of(&g, &nodes, OutputTarget::Srgb, None);
            let mut worst = 0.0f64;
            for id in plan.node_ids() {
                let reference = chain_ending_at(&g, plan::Stop::Picture(id), &nodes);
                let got = render_with(
                    dev,
                    &full,
                    &img,
                    Some(NodeStop::Picture(id)),
                    &[
                        ColorOutput::Stage(Stop::C13),
                        ColorOutput::Rgba8 { dither: false },
                    ],
                );
                for i in 0..n {
                    let want = reference.apply_pixel(pixel(&img, i));
                    for (ch, &want) in want.iter().enumerate() {
                        worst = worst.max((f64::from(planar(&got[0])[ch * n + i]) - want).abs());
                    }
                }
                // The 8-bit picture is the same stop quantised.
                let PixelData::Rgba8(bytes) = &got[1].data else {
                    panic!("RGBA8 expected")
                };
                for i in 0..n {
                    for ch in 0..3 {
                        let f = f64::from(planar(&got[0])[ch * n + i]);
                        let q = (f * 255.0).round();
                        assert!(
                            (f64::from(bytes[4 * i + ch]) - q).abs() <= 1.0,
                            "node {id}: RGBA8 {} against {q}",
                            bytes[4 * i + ch]
                        );
                    }
                }
                // The mask of a node is 1 until N-10: grey one, exactly, in every output.
                let mask = render_with(
                    dev,
                    &full,
                    &img,
                    Some(NodeStop::Mask(id)),
                    &[
                        ColorOutput::Stage(Stop::C13),
                        ColorOutput::Rgba8 { dither: true },
                        ColorOutput::U16 { bits: 10 },
                    ],
                );
                assert!(
                    planar(&mask[0]).iter().all(|&v| v == 1.0),
                    "node {id}'s mask"
                );
                let PixelData::Rgba8(bytes) = &mask[1].data else {
                    panic!("RGBA8 expected")
                };
                assert!(bytes.iter().all(|&b| b == 255), "node {id}'s mask in RGBA8");
                let PixelData::U16Planar(words) = &mask[2].data else {
                    panic!("U16 expected")
                };
                assert!(words.iter().all(|&v| v == 1023), "node {id}'s mask in U16");
            }
            eprintln!(
                "{:?}: stops after each of {} nodes against the CPU plans that end there: worst {worst:.1e} (gate {CODE10:.1e})",
                dev.mode,
                plan.step_count()
            );
            assert!(worst <= CODE10, "{worst}");
            // A stop does not change the other reading points, and a node the plan does not run has none.
            let stopped = render_with(
                dev,
                &full,
                &img,
                Some(NodeStop::Picture(MAIN_NODE)),
                &[ColorOutput::Stage(Stop::C10)],
            );
            let plain = render(dev, &full, &img, &[ColorOutput::Stage(Stop::C10)]);
            assert_bit_equal(
                "a node stop leaves C10 alone",
                planar(&stopped[0]),
                planar(&plain[0]),
            );
        }
        let dead = line(3);
        let mut with_dead = dead.nodes.clone();
        with_dead.push(node(9, 9.0, PictureSource::Node(1), None));
        let g = graph(with_dead, 3);
        let nodes = random_nodes(&mut Rng(1), 1..=3);
        let chain = chain_of(&g, &nodes, OutputTarget::Srgb, None);
        let refused = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                Ok(PreparedChain::new(ctx, &chain)?
                    .with_node_stop(NodeStop::Picture(9))
                    .is_err())
            });
        assert!(
            refused.expect("device"),
            "a dead node is not a step of the plan"
        );
    }
}

// ---- tiles, determinism, residency --------------------------------------------------------------------

#[test]
fn tiles_equal_the_whole_frame_with_4_nodes_and_with_two_passes() {
    let (w, h) = (97usize, 61usize);
    let img = camera_image(w, h, 0x7117);
    let mut rng = Rng(0x4444);
    let line4 = (line(4), random_nodes(&mut rng, 1..=4));
    let fan9 = (fan(9), random_nodes(&mut rng, (1..=10).chain(20..=28)));
    for dev in gpus() {
        for (what, (g, nodes)) in [("4 nodes", &line4), ("two passes", &fan9)] {
            let chain = chain_of(g, nodes, OutputTarget::DisplayP3, logc4_look());
            let whole = render(
                dev,
                &chain,
                &img,
                &[
                    ColorOutput::Rgba8 { dither: true },
                    ColorOutput::Stage(Stop::C13),
                ],
            );
            for tile in [24usize, 33, 64] {
                let plan = TilePlan::with_halo(w, h, tile, 0);
                let (chain2, img2) = (chain.clone(), img.clone());
                let tiles = dev
                    .gpu
                    .run(JobSpec::new(Priority::Export), move |ctx, token| {
                        let prepared = PreparedChain::new(ctx, &chain2)?;
                        let mut collected = Vec::new();
                        for out in [
                            ColorOutput::Rgba8 { dither: true },
                            ColorOutput::Stage(Stop::C13),
                        ] {
                            let mut tiles = Vec::new();
                            prepared.render_tiled(
                                ctx,
                                &img2,
                                &plan,
                                out,
                                2,
                                token,
                                &mut |t, im| {
                                    tiles.push((t, im));
                                    Ok(())
                                },
                            )?;
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
                assert_eq!(&tiled8, whole8, "{what}: 8-bit tiles of {tile}");
                let wholef = planar(&whole[1]);
                let mut tiledf = vec![f32::NAN; wholef.len()];
                for (t, im) in &tiles[1] {
                    let d = planar(im);
                    let c: Rect = t.core;
                    for ch in 0..3 {
                        for y in 0..c.h {
                            for x in 0..c.w {
                                tiledf[ch * w * h + (c.y + y) * w + c.x + x] =
                                    d[ch * c.w * c.h + y * c.w + x];
                            }
                        }
                    }
                }
                assert_bit_equal(&format!("{what}: f32 tiles of {tile}"), &tiledf, wholef);
            }
        }
    }
}

#[test]
fn a_repeat_render_is_identical_and_no_buffer_is_read_before_it_is_written() {
    let img = camera_image(59, 43, 0x4E4E);
    let mut rng = Rng(0x3333);
    let nodes = random_nodes(&mut rng, (1..=10).chain(20..=28));
    let g = fan(9);
    for (what, g, nodes) in [
        ("16 nodes", line(16), random_nodes(&mut Rng(0x16), 1..=16)),
        ("two passes", g, nodes),
    ] {
        let chain = chain_of(&g, &nodes, OutputTarget::Rec2020, logc4_look());
        // The poisoned test device fills every buffer with NaN; a plain one with zeros: the same bits
        // from both means the kernel read only what it, or the host, had written.
        let run = |gpu: &ym_gpu::device::Gpu| {
            let (chain, img) = (chain.clone(), img.clone());
            let once = move |ctx: &ym_gpu::device::GpuContext| {
                let p = PreparedChain::new(ctx, &chain)?;
                p.render(ctx, &img, (0, 0), ColorOutput::Stage(Stop::C13))
            };
            let (a, b) = gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    Ok((once(ctx)?, once(ctx)?))
                })
                .expect("render");
            assert_bit_equal(&format!("{what}: a repeat render"), planar(&a), planar(&b));
            planar(&a).to_vec()
        };
        assert_poison_neutral(what, run);
    }
}

#[test]
fn a_window_too_big_for_the_pass_buffers_is_cut_into_bands_and_gives_the_same_bits() {
    // 9 pictures alive: the pass buffers are 3 + 3 + ... planes of the window. A limit of a few rows of
    // them makes bands; the bands of a pointwise chain are exact, the dither included.
    let (w, h) = (45usize, 33usize);
    let img = camera_image(w, h, 0xBA4D);
    let mut rng = Rng(0xB4);
    let g = fan(9);
    let plan = compile(&g, plan::Stop::Output).expect("plan");
    let nodes = random_nodes(&mut rng, plan.node_ids());
    let chain = chain_of(&g, &nodes, OutputTarget::Srgb, logc4_look());
    let outputs = [
        ColorOutput::Rgba8 { dither: true },
        ColorOutput::Stage(Stop::C13),
    ];
    let dev = gpus().first();
    let Some(dev) = dev else { return };
    let whole = render(dev, &chain, &img, &outputs);
    let (c2, i2) = (chain.clone(), img.clone());
    let banded = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut out = Vec::new();
            // Bands of one, two, seven and thirty-two rows of the planes (`rows` times four bytes times
            // the planes a pixel needs, and the plan of 19 nodes needs a handful).
            for rows in [1u64, 2, 7, 32] {
                let planes = u64::from(PreparedChain::new(ctx, &c2)?.pass_buffer_planes());
                let p = PreparedChain::new(ctx, &c2)?
                    .with_pass_buffer_limit(rows * 4 * planes * w as u64);
                out.push(vec![
                    p.render(ctx, &i2, (0, 0), ColorOutput::Rgba8 { dither: true })?,
                    p.render(ctx, &i2, (0, 0), ColorOutput::Stage(Stop::C13))?,
                ]);
            }
            Ok(out)
        })
        .expect("banded render");
    for (k, got) in banded.iter().enumerate() {
        assert_eq!(got[0], whole[0], "RGBA8 in bands (case {k})");
        assert_bit_equal("f32 in bands", planar(&got[1]), planar(&whole[1]));
    }
}

#[test]
fn the_device_entry_points_write_a_multi_pass_plan_at_their_offsets() {
    // The export's path: a window of a device image to a device image at an offset, and RGBA8 into a
    // buffer, the source with a letterbox plane. Same bits as the host wrapper, nothing else touched.
    let (w, h) = (53usize, 29usize);
    let img = camera_image(w, h, 0x0FF5);
    let g = fan(9);
    let plan = compile(&g, plan::Stop::Output).expect("plan");
    let nodes = random_nodes(&mut Rng(0x0FF), plan.node_ids());
    let chain = chain_of(&g, &nodes, OutputTarget::Srgb, logc4_look());
    let Some(dev) = gpus().first() else { return };
    let (img2, chain2) = (img.clone(), chain.clone());
    let (whole, slab) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let prepared = PreparedChain::new(ctx, &chain2)?;
            let whole = prepared.render(ctx, &img2, (0, 0), ColorOutput::Stage(Stop::C13))?;
            let src = GpuImage::upload_rgb(ctx, "multi-pass test input", &img2)?;
            let dst = GpuImage::alloc(ctx, "slab", 80, 40, 3, img2.place)?;
            let mut passes = PassList::new(ctx, "slab");
            for (x0, wid) in [(0usize, 21usize), (21, 32)] {
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    Rect::new(x0, 0, wid, h),
                    Stop::C13,
                    &dst,
                    (10 + x0, 5),
                )?;
            }
            let id = dst.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            Ok((planar(&whole).to_vec(), dst.take(&done, id)?))
        })
        .expect("device");
    for c in 0..3 {
        for y in 0..40usize {
            for x in 0..80usize {
                let got = slab[c * 80 * 40 + y * 80 + x];
                if (10..10 + w).contains(&x) && (5..5 + h).contains(&y) {
                    let want = whole[c * w * h + (y - 5) * w + (x - 10)];
                    assert_eq!(got.to_bits(), want.to_bits(), "plane {c} at ({x}, {y})");
                } else {
                    assert!(got.is_nan(), "plane {c} at ({x}, {y}) was written: {got}");
                }
            }
        }
    }
}

// ---- timing -------------------------------------------------------------------------------------------

#[test]
fn the_colour_pass_of_1_4_and_16_nodes_on_lavapipe_is_timed() {
    // Not a gate: the numbers of `docs/23` section 15 (lavapipe is correct and slow, `docs/17` section 12).
    let (w, h) = (512usize, 384usize);
    let img = camera_image(w, h, 0x71E5);
    let Some(dev) = gpus().first() else { return };
    let mut line_ms = Vec::new();
    for n in [1u32, 4, 16] {
        let nodes = random_nodes(&mut Rng(0x7100 + u64::from(n)), 1..=n);
        let chain = chain_of(&line(n), &nodes, OutputTarget::Viewport, logc4_look());
        let (c, i) = (chain.clone(), img.clone());
        let ms = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let p = PreparedChain::new(ctx, &c)?;
                // The first frame compiles the pipeline (once per device and process); then the best of five.
                let cold = Instant::now();
                p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                if n == 1 {
                    eprintln!(
                        "the first frame of the process, pipeline included: {:.0} ms",
                        cold.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(best)
            })
            .expect("render");
        line_ms.push((n, ms));
    }
    let two_passes = {
        let g = fan(9);
        let plan = compile(&g, plan::Stop::Output).expect("plan");
        let nodes = random_nodes(&mut Rng(0x7109), plan.node_ids());
        let chain = chain_of(&g, &nodes, OutputTarget::Viewport, logc4_look());
        let (c, i) = (chain.clone(), img.clone());
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let p = PreparedChain::new(ctx, &c)?;
                p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(best)
            })
            .expect("render")
    };
    eprintln!(
        "colour pass on lavapipe, {w}x{h} to RGBA8, random graphs: {}; fan(9) in two passes (19 nodes): {two_passes:.1} ms",
        line_ms
            .iter()
            .map(|(n, ms)| format!("{n} node(s) {ms:.1} ms"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn the_default_graph_is_timed() {
    // Not a gate: Main alone, with nothing on and with everything on (grade, colour warp, curves, a look),
    // 512 x 384 to RGBA8 on lavapipe, the best of seven frames. The numbers of `docs/23` section 15 compare
    // them with the kernel before the plan loop.
    let (w, h) = (512usize, 384usize);
    let img = camera_image(w, h, 0x71E5);
    let Some(dev) = gpus().first() else { return };
    let mut all_on = chains::full(OutputTarget::Viewport);
    all_on.warp = chains::warp_pins();
    let mut out = Vec::new();
    for (name, params) in [
        ("nothing on", chains::base(OutputTarget::Viewport)),
        ("everything on", all_on),
    ] {
        let chain = ColorChain::new(&params).expect("chain");
        let (c, i) = (chain.clone(), img.clone());
        let ms = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let p = PreparedChain::new(ctx, &c)?;
                p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let t = Instant::now();
                    p.render(ctx, &i, (0, 0), ColorOutput::Rgba8 { dither: true })?;
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(best)
            })
            .expect("render");
        out.push(format!("{name} {ms:.1} ms"));
    }
    eprintln!(
        "default graph, {w}x{h} to RGBA8 on lavapipe: {}",
        out.join(", ")
    );
}
