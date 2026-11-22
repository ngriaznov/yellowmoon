//! CU-05 (`docs/32-color-uniform.md` sections 3.10, 6 and 8): the Colour Uniform step in the fused colour kernel against
//! `ym_color::uniform::apply`, through the CPU chain (`ColorChain::trace`).
//!
//! The gate is the grade's: C10, the last LogC4 value of the nodes and the look, within 2e-4 in LogC4 (the measured error is
//! printed and written into `docs/STATUS.md`), and the encoded picture within one 10-bit code. Every axis alone and together,
//! a target at 0 degrees, at 180 degrees and near grey (Saturation 2), amounts at 100 %, a node after a node, negative codes,
//! chromas from 1e-4 to 0.6 so that the neutral guard and the antipode fade are on the picture; the identity cases (no target,
//! amounts 0, bypassed, a pixel with mask 0) are exact; tiles equal the whole frame, a repeat render is identical and no buffer
//! is read before it is written.

mod common;

use std::collections::BTreeMap;

use common::chains::{self, pixel};
use common::exact::{assert_bit_equal, assert_poison_neutral};
use common::{Rng, f16_round, gpus};
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ChainParams, ColorChain, MainNode, NodeParams, NodeSet, OutputTarget, Trim, UniformStep,
};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop as PlanStop, compile};
use ym_color::uniform::{Amounts, Target};
use ym_color::warp::{WarpParams, WarpPin};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, NodeStop, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::tiles::TilePlan;

/// The grade's gate in LogC4 (`docs/13`, `docs/17` C10).
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;
/// Codes up to this size are held to the absolute gate (as `gpu_nodes.rs`).
const SANE_CODE: f64 = 20.0;

/// A picture whose chroma runs from 1e-4 to 0.6 of the linear value at every hue (so the LogC4 chroma is from a grey to a vivid
/// colour and the pixels fall on every part of the antipode fade), over 14 stops, with a few negative samples and an exact zero.
fn chroma_image(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for i in 0..w * h {
        let lum = 0.18 * rng.range(-9.0, 4.0).exp2();
        let strength = (10f64).powf(rng.range(-4.0, -0.2));
        let hue = rng.range(0.0, std::f64::consts::TAU);
        let (a, b) = (strength * hue.cos(), strength * hue.sin());
        let mut px = [
            lum * (1.0 + a),
            lum * (1.0 - 0.5 * a + 0.866 * b),
            lum * (1.0 - 0.5 * a - 0.866 * b),
        ];
        if i % 29 == 3 {
            px[2] = -0.003 * rng.unit();
        }
        if i == 0 {
            px = [0.0; 3];
        }
        if i == 1 {
            px = [0.18; 3];
        }
        for c in px {
            rgb.push(f16_round(c as f32));
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

fn plan_node(id: u32, picture: PictureSource) -> PlanNode {
    PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(picture),
        key: None,
    }
}

/// Main, then one Colour Uniform node per entry of `nodes` in a line; the last is the Output's.
fn chain_of(nodes: &[NodeParams], target: OutputTarget) -> ColorChain {
    let mut graph = vec![plan_node(MAIN_NODE, PictureSource::Input)];
    let mut others = BTreeMap::new();
    for (k, n) in nodes.iter().enumerate() {
        let id = 2 + k as u32;
        graph.push(plan_node(id, PictureSource::Node(id - 1)));
        others.insert(id, n.clone());
    }
    let last = 1 + nodes.len() as u32;
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(last.max(MAIN_NODE)),
        },
        PlanStop::Output,
    )
    .expect("plan");
    let mut p: ChainParams = chains::base(target);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    ColorChain::new(&p).expect("chain")
}

fn step(hue: f64, sat: f64, ev: f64, amounts: (f64, f64, f64)) -> UniformStep {
    UniformStep {
        target: Some(Target::from_stored(hue, sat, ev)),
        amounts: Amounts::from_percent(amounts.0, amounts.1, amounts.2),
    }
}

fn node(u: UniformStep) -> NodeParams {
    NodeParams {
        uniform: Some(u),
        ..NodeParams::default()
    }
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

fn render(
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

/// The device's C10 and C13 against the CPU chain: the worst absolute error of the LogC4 codes within [`SANE_CODE`], the worst
/// relative error of the others, the worst C13 error, and how many pixels the node moved by more than 1e-3 (against `base`).
struct Measured {
    c10: f64,
    beyond_rel: f64,
    c13: f64,
    moved: usize,
}

fn measure(
    dev: &common::TestGpu,
    chain: &ColorChain,
    base: &ColorChain,
    img: &RgbImage,
) -> Measured {
    let n = img.width * img.height;
    let got = render(
        dev,
        chain,
        img,
        None,
        &[ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C13)],
    );
    let mut m = Measured {
        c10: 0.0,
        beyond_rel: 0.0,
        c13: 0.0,
        moved: 0,
    };
    for i in 0..n {
        let cam = pixel(img, i);
        let t = chain.trace(cam);
        let b = base.trace(cam);
        let mut beyond = false;
        let mut moved = false;
        for ch in 0..3 {
            let want = t.c10_logc4.0[ch];
            let d = (f64::from(planar(&got[0])[ch * n + i]) - want).abs();
            if want.abs() > SANE_CODE {
                beyond = true;
                m.beyond_rel = m.beyond_rel.max(d / want.abs());
            } else {
                m.c10 = m.c10.max(d);
            }
            moved |= (want - b.c10_logc4.0[ch]).abs() > 1e-3;
        }
        m.moved += usize::from(moved);
        if !beyond {
            for ch in 0..3 {
                let d = (f64::from(planar(&got[1])[ch * n + i]) - t.c13_encoded[ch]).abs();
                m.c13 = m.c13.max(d);
            }
        }
    }
    m
}

/// The cases of the gate: every axis alone and together, a target at 0 and at 180 degrees, near grey (Saturation 2), 100 %,
/// the lightness at the ends of its range.
fn cases() -> Vec<(&'static str, UniformStep)> {
    vec![
        ("hue only", step(26.0, 9.7, 0.0, (80.0, 0.0, 0.0))),
        ("saturation only", step(26.0, 9.7, 0.0, (0.0, 80.0, 0.0))),
        ("lightness only", step(26.0, 9.7, 0.3, (0.0, 0.0, 80.0))),
        ("all three", step(26.0, 9.7, -0.03, (60.0, 60.0, 40.0))),
        (
            "target at 0 degrees",
            step(0.0, 12.0, 0.5, (100.0, 70.0, 30.0)),
        ),
        (
            "target at 180 degrees",
            step(180.0, 15.0, -1.0, (100.0, 100.0, 100.0)),
        ),
        (
            "target near grey",
            step(300.0, 2.0, 0.0, (100.0, 100.0, 0.0)),
        ),
        (
            "target at 359.99 degrees",
            step(359.99, 20.0, 2.0, (90.0, 50.0, 60.0)),
        ),
        (
            "full on every axis",
            step(120.0, 30.0, 1.0, (100.0, 100.0, 100.0)),
        ),
        ("a faint pull", step(210.0, 6.0, 0.0, (5.0, 5.0, 5.0))),
    ]
}

#[test]
fn the_device_matches_the_cpu_on_every_axis_and_every_target() {
    let img = chroma_image(97, 71, 0xC505);
    let base = chain_of(&[], OutputTarget::Srgb);
    for dev in gpus() {
        let (mut worst, mut worst_rel, mut worst13) = (0.0f64, 0.0f64, 0.0f64);
        for (what, u) in cases() {
            let chain = chain_of(&[node(u)], OutputTarget::Srgb);
            let m = measure(dev, &chain, &base, &img);
            eprintln!(
                "{:?}: {what}: C10 {:.2e} (beyond the range {:.1e}, relative), C13 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.beyond_rel, m.c13, m.moved
            );
            assert!(m.moved > 50, "{what}: the node moved {} pixels", m.moved);
            assert!(m.c10 <= LOGC4_GATE, "{what}: C10 differs by {:.3e}", m.c10);
            assert!(m.beyond_rel <= 1e-4, "{what}: {:.3e}", m.beyond_rel);
            assert!(m.c13 <= CODE10, "{what}: C13 differs by {:.3e}", m.c13);
            worst = worst.max(m.c10);
            worst_rel = worst_rel.max(m.beyond_rel);
            worst13 = worst13.max(m.c13);
        }
        eprintln!(
            "{:?}: Colour Uniform, all cases: worst C10 {worst:.2e} (gate {LOGC4_GATE:.0e}), worst beyond-range relative {worst_rel:.1e}, worst C13 {worst13:.2e} (gate {CODE10:.1e})",
            dev.mode
        );
    }
}

#[test]
fn a_node_after_a_node_matches_the_cpu() {
    let img = chroma_image(61, 47, 0xC506);
    let base = chain_of(&[], OutputTarget::Srgb);
    let two = [
        node(step(26.0, 9.7, 0.0, (70.0, 70.0, 30.0))),
        node(step(200.0, 14.0, 0.5, (100.0, 40.0, 50.0))),
    ];
    let chain = chain_of(&two, OutputTarget::DisplayP3);
    for dev in gpus() {
        let m = measure(dev, &chain, &base, &img);
        eprintln!(
            "{:?}: a node after a node: C10 {:.2e}, C13 {:.2e}",
            dev.mode, m.c10, m.c13
        );
        assert!(m.moved > 50);
        assert!(m.c10 <= LOGC4_GATE, "C10 differs by {:.3e}", m.c10);
        assert!(m.c13 <= CODE10, "C13 differs by {:.3e}", m.c13);
    }
}

/// A grey ramp and black keep chroma 0 on the device (no NaN from `atan2(0, 0)`), and the node's lightness moves them.
#[test]
fn greys_and_black_stay_finite_and_grey_on_the_device() {
    let w = 64usize;
    let mut rgb = Vec::new();
    for i in 0..w {
        let v = if i == 0 {
            0.0
        } else {
            f16_round((0.18 * (-8.0 + 12.0 * i as f64 / w as f64).exp2()) as f32)
        };
        rgb.extend([v, v, v]);
    }
    let img = RgbImage::from_interleaved(w, 1, &rgb).expect("image");
    let chain = chain_of(
        &[node(step(26.0, 9.7, 0.5, (100.0, 100.0, 60.0)))],
        OutputTarget::Srgb,
    );
    for dev in gpus() {
        let got = render(dev, &chain, &img, None, &[ColorOutput::Stage(Stop::C10)]);
        let d = planar(&got[0]);
        assert!(d.iter().all(|v| v.is_finite()), "{:?}", dev.mode);
        for i in 0..w {
            let t = chain.trace(pixel(&img, i));
            for ch in 0..3 {
                let want = t.c10_logc4.0[ch];
                assert!(
                    (f64::from(d[ch * w + i]) - want).abs() <= LOGC4_GATE,
                    "pixel {i} channel {ch}: {} against {want}",
                    d[ch * w + i]
                );
            }
        }
    }
}

/// A node with no target, with all amounts 0 or bypassed runs no code: the bits of the graph without it.
#[test]
fn a_node_at_its_identity_changes_no_bit_on_the_device() {
    let img = chroma_image(53, 41, 0xC507);
    let base = chain_of(&[node(UniformStep::default())], OutputTarget::Srgb);
    let live = step(26.0, 9.7, 0.0, (80.0, 80.0, 40.0));
    let identity = [
        (
            "no target",
            UniformStep {
                target: None,
                ..live
            },
        ),
        (
            "all amounts 0",
            UniformStep {
                amounts: Amounts::default(),
                ..live
            },
        ),
    ];
    let outs = [
        ColorOutput::Stage(Stop::C10),
        ColorOutput::Stage(Stop::C13),
        ColorOutput::Rgba8 { dither: true },
    ];
    for dev in gpus() {
        let want = render(dev, &base, &img, None, &outs);
        for (what, u) in &identity {
            let got = render(
                dev,
                &chain_of(&[node(*u)], OutputTarget::Srgb),
                &img,
                None,
                &outs,
            );
            for (a, b) in got.iter().zip(&want) {
                match (&a.data, &b.data) {
                    (PixelData::F32Planar(x), PixelData::F32Planar(y)) => {
                        assert_bit_equal(what, x, y);
                    }
                    (PixelData::Rgba8(x), PixelData::Rgba8(y)) => assert_eq!(x, y, "{what}"),
                    _ => panic!("unexpected pixel data"),
                }
            }
        }
        let bypassed = NodeParams {
            bypass: true,
            ..node(live)
        };
        let got = render(
            dev,
            &chain_of(&[bypassed], OutputTarget::Srgb),
            &img,
            None,
            &outs,
        );
        assert_bit_equal("bypassed", planar(&got[0]), planar(&want[0]));
    }
}

/// A pixel with the node's mask at 0 has the input's bits (the early out), one with the mask at 1 the node's own, and the
/// mask is the node's `m` in between.
#[test]
fn a_pixel_with_mask_0_is_the_inputs_bits_and_one_with_mask_1_the_nodes() {
    let img = chroma_image(89, 57, 0xC508);
    let n = img.width * img.height;
    let masked = NodeParams {
        masks: NodeMasks {
            rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                low: 0.3,
                high: 0.55,
                softness: 0.2,
            })],
            invert: false,
            ..NodeMasks::default()
        },
        ..node(step(26.0, 9.7, 0.0, (80.0, 80.0, 40.0)))
    };
    let unmasked = node(step(26.0, 9.7, 0.0, (80.0, 80.0, 40.0)));
    let (chain, plain, without) = (
        chain_of(&[masked], OutputTarget::Srgb),
        chain_of(&[unmasked], OutputTarget::Srgb),
        chain_of(&[], OutputTarget::Srgb),
    );
    for dev in gpus() {
        let m = render(
            dev,
            &chain,
            &img,
            Some(NodeStop::Mask(2)),
            &[ColorOutput::Stage(Stop::C13)],
        );
        let masks = planar(&m[0]);
        let outs = [ColorOutput::Stage(Stop::C10)];
        let got = render(dev, &chain, &img, None, &outs);
        let full = render(dev, &plain, &img, None, &outs);
        let none = render(dev, &without, &img, None, &outs);
        let (mut zeros, mut ones) = (0, 0);
        for (i, &mask) in masks.iter().enumerate().take(n) {
            // C13 of a mask stop is the grey code of `m`; the mask is read from the same stop on every channel.
            for ch in 0..3 {
                let (g, f, b) = (
                    planar(&got[0])[ch * n + i],
                    planar(&full[0])[ch * n + i],
                    planar(&none[0])[ch * n + i],
                );
                if mask == 0.0 {
                    assert_eq!(g.to_bits(), b.to_bits(), "pixel {i}: mask 0");
                } else if mask == 1.0 {
                    assert_eq!(g.to_bits(), f.to_bits(), "pixel {i}: mask 1");
                }
            }
            zeros += usize::from(mask == 0.0);
            ones += usize::from(mask == 1.0);
        }
        assert!(zeros > 20 && ones > 20, "{zeros} zeros, {ones} ones");
    }
}

/// Tiles of any size equal the whole frame bit for bit, a repeat render is identical, and a poisoned device gives the same bits.
#[test]
fn tiles_and_repeats_are_the_same_bits_with_the_node() {
    let (w, h) = (97usize, 61usize);
    let img = chroma_image(w, h, 0xC509);
    let nodes = [
        node(step(26.0, 9.7, 0.0, (80.0, 70.0, 40.0))),
        node(step(200.0, 14.0, 0.5, (100.0, 40.0, 50.0))),
    ];
    let chain = chain_of(&nodes, OutputTarget::DisplayP3);
    for dev in gpus() {
        let whole = render(
            dev,
            &chain,
            &img,
            None,
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
            assert_eq!(&tiled8, whole8, "8-bit tiles of {tile}");
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
            assert_bit_equal(&format!("f32 tiles of {tile}"), &tiledf, wholef);
        }
    }
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
        assert_bit_equal("a repeat render", planar(&a), planar(&b));
        planar(&a).to_vec()
    };
    assert_poison_neutral("Colour Uniform", run);
}

// ---- CG-04 (`docs/32` section 12a.4): the pull and the corrector's steps in one node --------------------------------------

fn graded_grade() -> GradeParams {
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

fn graded_warp() -> WarpParams {
    WarpParams {
        pins: vec![WarpPin {
            range: 0.25,
            ..WarpPin::moving([0.05, 0.0], [0.05, 0.08])
        }],
    }
}

fn graded_curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).expect("curve"),
        ..CurveSet::default()
    }
}

fn graded_trim() -> Trim {
    Trim {
        exposure_ev: 0.5,
        temperature: 12.0,
        tint: -8.0,
    }
}

/// The corrector's steps of a graded node: `which` names the ones that are on.
fn steps_of(which: &str) -> NodeParams {
    let on = |k: &str| which == "all" || which == k;
    NodeParams {
        trim: if on("trim") {
            graded_trim()
        } else {
            Trim::default()
        },
        grade: if on("grade") {
            graded_grade()
        } else {
            GradeParams::default()
        },
        warp: if on("warp") {
            graded_warp()
        } else {
            WarpParams::default()
        },
        curves: if on("curves") {
            graded_curves()
        } else {
            CurveSet::default()
        },
        ..NodeParams::default()
    }
}

/// A graded Colour Uniform node as the plan lowers it (variant W2, `docs/32` section 12a.4): the pull with the node's masks as node
/// 2, then the corrector's steps as node 3, which take node 2's mask `m` through their key input.
fn lowered(pull: NodeParams, steps: NodeParams, target: OutputTarget) -> ColorChain {
    let mut graph = vec![
        plan_node(MAIN_NODE, PictureSource::Input),
        plan_node(2, PictureSource::Node(1)),
    ];
    graph.push(PlanNode {
        key: Some(2),
        ..plan_node(3, PictureSource::Node(2))
    });
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(3),
        },
        PlanStop::Output,
    )
    .expect("plan");
    let mut p: ChainParams = chains::base(target);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others: BTreeMap::from([(2, pull), (3, steps)]),
    };
    ColorChain::new(&p).expect("chain")
}

/// The pull with each of the corrector's steps alone and with all of them, lowered to two steps, against the CPU chain.
#[test]
fn a_pull_with_each_grade_step_matches_the_cpu() {
    let img = chroma_image(97, 71, 0xC609);
    let base = chain_of(&[], OutputTarget::Srgb);
    let live = step(26.0, 9.7, -0.03, (60.0, 60.0, 40.0));
    for dev in gpus() {
        let mut worst = 0.0f64;
        for which in ["trim", "grade", "warp", "curves", "all"] {
            let chain = lowered(node(live), steps_of(which), OutputTarget::Srgb);
            let m = measure(dev, &chain, &base, &img);
            eprintln!(
                "{:?}: pull and {which}: C10 {:.2e} (beyond the range {:.1e}), C13 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.beyond_rel, m.c13, m.moved
            );
            assert!(m.moved > 50, "{which}: moved {}", m.moved);
            assert!(m.c10 <= LOGC4_GATE, "{which}: C10 differs by {:.3e}", m.c10);
            assert!(m.beyond_rel <= 1e-4, "{which}: {:.3e}", m.beyond_rel);
            assert!(m.c13 <= CODE10, "{which}: C13 differs by {:.3e}", m.c13);
            worst = worst.max(m.c10);
        }
        eprintln!(
            "{:?}: pull and grade, worst C10 {worst:.2e} (gate {LOGC4_GATE:.0e})",
            dev.mode
        );
    }
}

/// A graded node whose pull is at its identity is a corrector's grade, and the order is the CPU's.
#[test]
fn a_lowered_node_with_no_pull_keeps_a_correctors_bits_and_the_order_is_the_pulls_first() {
    let img = chroma_image(53, 41, 0xC60A);
    let live = step(26.0, 9.7, 0.0, (80.0, 80.0, 40.0));
    let outs = [ColorOutput::Stage(Stop::C10), ColorOutput::Stage(Stop::C13)];
    for dev in gpus() {
        let corrector = chain_of(&[steps_of("all")], OutputTarget::Srgb);
        let no_pull = lowered(
            node(UniformStep {
                amounts: Amounts::default(),
                ..live
            }),
            steps_of("all"),
            OutputTarget::Srgb,
        );
        let a = render(dev, &corrector, &img, None, &outs);
        let b = render(dev, &no_pull, &img, None, &outs);
        assert_bit_equal("no pull", planar(&b[0]), planar(&a[0]));
        let both = render(
            dev,
            &lowered(node(live), steps_of("all"), OutputTarget::Srgb),
            &img,
            None,
            &outs,
        );
        let reversed = render(
            dev,
            &chain_of(&[steps_of("all"), node(live)], OutputTarget::Srgb),
            &img,
            None,
            &outs,
        );
        let n = img.width * img.height;
        let widest = (0..3 * n)
            .map(|i| f64::from((planar(&both[0])[i] - planar(&reversed[0])[i]).abs()))
            .fold(0.0, f64::max);
        assert!(
            widest > 1e-3,
            "{:?}: the other order gives the same picture ({widest})",
            dev.mode
        );
    }
}

/// A masked graded node (soft mask: the two steps share its `m`) after a node, and a repeat render.
#[test]
fn a_masked_lowered_node_after_a_node_matches_the_cpu() {
    let img = chroma_image(61, 47, 0xC60B);
    let base = chain_of(&[], OutputTarget::Srgb);
    let masked = NodeParams {
        masks: NodeMasks {
            rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                low: 0.3,
                high: 0.55,
                softness: 0.2,
            })],
            invert: false,
            ..NodeMasks::default()
        },
        ..node(step(200.0, 14.0, 0.5, (100.0, 40.0, 50.0)))
    };
    let first = node(step(26.0, 9.7, 0.0, (70.0, 70.0, 30.0)));
    // Main, the first node, the masked pull, its grade.
    let mut graph = vec![
        plan_node(MAIN_NODE, PictureSource::Input),
        plan_node(2, PictureSource::Node(1)),
        plan_node(3, PictureSource::Node(2)),
    ];
    graph.push(PlanNode {
        key: Some(3),
        ..plan_node(4, PictureSource::Node(3))
    });
    let plan = compile(
        &PlanGraph {
            nodes: graph,
            output: PictureSource::Node(4),
        },
        PlanStop::Output,
    )
    .expect("plan");
    let mut p: ChainParams = chains::base(OutputTarget::DisplayP3);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others: BTreeMap::from([(2, first), (3, masked), (4, steps_of("all"))]),
    };
    let chain = ColorChain::new(&p).expect("chain");
    for dev in gpus() {
        let m = measure(dev, &chain, &base, &img);
        eprintln!(
            "{:?}: a masked lowered node after a node: C10 {:.2e}, C13 {:.2e}",
            dev.mode, m.c10, m.c13
        );
        assert!(m.moved > 50);
        assert!(m.c10 <= LOGC4_GATE, "C10 differs by {:.3e}", m.c10);
        assert!(m.c13 <= CODE10, "C13 differs by {:.3e}", m.c13);
        let whole = render(dev, &chain, &img, None, &[ColorOutput::Stage(Stop::C10)]);
        let again = render(dev, &chain, &img, None, &[ColorOutput::Stage(Stop::C10)]);
        assert_bit_equal("a repeat", planar(&again[0]), planar(&whole[0]));
    }
}
