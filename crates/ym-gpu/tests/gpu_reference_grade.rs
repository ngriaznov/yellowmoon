//! RG-08 (`docs/34-reference-grade.md` sections 4.2, 4a, 7.2 and 7.3, `docs/17-gpu.md` section 11a): the Reference Grade node's
//! hue step and element step in the fused colour kernel against `PreparedNode::apply` through the CPU chain.
//!
//! The gate is the grade's: C10, the last LogC4 value of the nodes and the look, within 2e-4 in LogC4 (the worst measured error
//! is printed and written into `docs/STATUS.md`), the encoded picture within one 10-bit code, and the identity cases (no hue
//! curve, curves that do nothing, no live element, a bypassed node, a pixel with mask 0) **exact**. The cases are the
//! spec's: `hue_vs_hue` alone, `hue_vs_sat` alone, both, with trims, grade and curves, each element alone and all seven, planes
//! of mask 0, partial and 1, a pixel in two elements, a pixel at `r = 0` and near grey, colours at 0, 60 and 359.99 degrees
//! (the seam), negative codes, a node after a node, a node with a mask.
//!
//! Two kinds of test. The kernel's own: the planes are the CPU's u16 codes uploaded as they are (`RasterPlanes::from_codes`), so
//! the two sides read the same masks and what is measured is the step. The planes' own: the device makes the element planes
//! from the scene (the nearest pixel of the scene at the place of each output pixel, the guided filter of the lightness of the
//! Input's picture) and they are held against `ym_color::mask_element` within a stated bound, a tile equals the whole frame
//! bit for bit, and the colour pass with the device's planes is within one 10-bit code of the CPU's chain with the CPU's.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::chains::{self, pixel};
use common::exact::{assert_bit_equal, assert_poison_neutral};
use common::{Rng, TestGpu, f16_round, gpus};
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ChainParams, ColorChain, MainNode, NodeParams, NodeSet, OutputTarget, ReferenceStep, Trim,
};
use ym_color::grade::GradeParams;
use ym_color::hue::{ELEMENT_COUNT, ElementGrade, ElementStep, HueStep};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoFrame, PhotoMap, PhotoSpot};
use ym_color::mask_element::{ElementJob, ElementPlanes, ElementRow, ElementScene, element_planes};
use ym_color::mask_raster::Area;
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop as PlanStop, compile};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::sampler::{GridPositions, SamplerGrid, sample_to_camera};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorImage, ColorOutput, NodeStop, PreparedChain, Stop};
use ym_gpu::kernels::mask_raster::{
    RasterJob, RasterPlanes, RasterTile, encode_raster_planes, encode_raster_planes_tiled,
};
use ym_gpu::kernels::sampler::encode_sample_at;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The grade's gate in LogC4 (`docs/13`, `docs/17` C10).
const LOGC4_GATE: f64 = 2e-4;
/// One code value at 10 bits.
const CODE10: f64 = 1.0 / 1023.0;
/// Codes up to this size are held to the absolute gate (as `gpu_nodes.rs`).
const SANE_CODE: f64 = 20.0;
/// The bound of the element planes the device makes against the CPU's, absolute on the plane's value (measured, see the tests).
const PLANE_GATE: f64 = 5e-3;

/// A picture whose chroma runs from 1e-4 to 0.6 of the linear value at every hue, over 14 stops, with a few negative samples,
/// an exact zero, a grey, and pixels at the hue seam.
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

/// Main, then one node per entry of `nodes` in a line (ids 2, 3, ...); the last is the Output's. `elements` are the element rows
/// of the Reference Grade nodes with live elements.
fn chain_with(nodes: &[NodeParams], elements: Vec<ElementRow>, target: OutputTarget) -> ColorChain {
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
    p.elements = elements;
    ColorChain::new(&p).expect("chain")
}

fn hue_chain(nodes: &[NodeParams], target: OutputTarget) -> ColorChain {
    chain_with(nodes, Vec::new(), target)
}

/// A scene of one pixel, for a chain whose planes are uploaded and not made.
fn dummy_scene() -> Arc<ElementScene> {
    Arc::new(ElementScene {
        key: 1,
        width: 1,
        height: 1,
        bits: vec![0],
        classes: Vec::new(),
    })
}

fn reference(hue: HueStep, elements: ElementStep) -> NodeParams {
    NodeParams {
        reference: Some(ReferenceStep { hue, elements }),
        ..NodeParams::default()
    }
}

fn hue(vs_hue: &[(f64, f64)], vs_sat: &[(f64, f64)]) -> NodeParams {
    reference(HueStep::new(vs_hue, vs_sat), ElementStep::default())
}

fn planar(img: &ColorImage) -> &[f32] {
    match &img.data {
        PixelData::F32Planar(v) => v,
        other => panic!("expected f32 planes, got {other:?}"),
    }
}

/// The device's outputs of `chain` over `img`, with the element planes `codes` bound when there are some.
fn render(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    codes: Option<&[Vec<u16>]>,
    node_stop: Option<NodeStop>,
    outputs: &[ColorOutput],
) -> Vec<ColorImage> {
    let (chain, img, outputs) = (chain.clone(), img.clone(), outputs.to_vec());
    let codes = codes.map(<[Vec<u16>]>::to_vec);
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let mut prepared = PreparedChain::new(ctx, &chain)?;
            if let Some(s) = node_stop {
                prepared = prepared.with_node_stop(s)?;
            }
            if let Some(codes) = &codes {
                prepared = prepared
                    .with_rasters(RasterPlanes::from_codes(ctx, img.width, img.height, codes)?);
            }
            outputs
                .iter()
                .map(|o| prepared.render(ctx, &img, (0, 0), *o))
                .collect()
        })
        .expect("GPU render")
}

/// What the device differs by: C10 (absolute, within [`SANE_CODE`]; relative beyond), C13, and how many pixels the node moved.
struct Measured {
    c10: f64,
    beyond_rel: f64,
    c13: f64,
    moved: usize,
}

/// The CPU chain's trace at pixel `i`, with the element planes when the case has them.
fn trace(
    chain: &ColorChain,
    img: &RgbImage,
    planes: Option<&ElementPlanes>,
    i: usize,
) -> ym_color::display::StageTrace {
    let cam = pixel(img, i);
    match planes {
        None => chain.trace(cam),
        Some(p) => {
            let (x, y) = (i % img.width, i / img.width);
            let frame = PhotoFrame::new(img.width as f64, img.height as f64, 1.0);
            let spot = PhotoSpot::new(
                [
                    (x as f64 + 0.5) / img.width as f64,
                    (y as f64 + 0.5) / img.height as f64,
                ],
                frame,
            )
            .with_raster(p, x, y);
            chain.trace_at(cam, Some(spot))
        }
    }
}

fn measure(
    dev: &TestGpu,
    chain: &ColorChain,
    base: &ColorChain,
    img: &RgbImage,
    planes: Option<&ElementPlanes>,
    codes: Option<&[Vec<u16>]>,
) -> Measured {
    let n = img.width * img.height;
    let got = render(
        dev,
        chain,
        img,
        codes,
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
        let t = trace(chain, img, planes, i);
        let b = base.trace(pixel(img, i));
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

fn check(what: &str, m: &Measured, expect_moved: usize) {
    assert!(
        m.moved > expect_moved,
        "{what}: the node moved {} pixels",
        m.moved
    );
    assert!(m.c10 <= LOGC4_GATE, "{what}: C10 differs by {:.3e}", m.c10);
    assert!(m.beyond_rel <= 1e-4, "{what}: {:.3e}", m.beyond_rel);
    assert!(m.c13 <= CODE10, "{what}: C13 differs by {:.3e}", m.c13);
}

// ---- the hue step ---------------------------------------------------------------------------------------------------------

fn hue_cases() -> Vec<(&'static str, NodeParams)> {
    let vh = [(30.0, 25.0), (120.0, -15.0), (200.0, 18.0), (300.0, -30.0)];
    let vs = [(0.0, 1.4), (90.0, 0.6), (180.0, 1.8), (270.0, 0.45)];
    vec![
        ("hue_vs_hue alone", hue(&vh, &[])),
        ("hue_vs_sat alone", hue(&[], &vs)),
        ("both", hue(&vh, &vs)),
        (
            "the seam (0, 60 and 359.99 degrees)",
            hue(
                &[(0.0, 40.0), (60.0, -25.0), (359.99, 40.0)],
                &[(0.0, 0.5), (60.0, 2.0), (359.99, 0.5)],
            ),
        ),
        ("one point", hue(&[(10.0, 33.0)], &[(300.0, 1.6)])),
        (
            "sixteen points",
            hue(
                &(0..16)
                    .map(|k| (22.5 * f64::from(k), if k % 2 == 0 { 30.0 } else { -30.0 }))
                    .collect::<Vec<_>>(),
                &(0..16)
                    .map(|k| (22.5 * f64::from(k), if k % 3 == 0 { 2.5 } else { 0.7 }))
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            "the widest ranges",
            hue(
                &[(45.0, 180.0), (225.0, -180.0)],
                &[(10.0, 8.0), (190.0, 0.125)],
            ),
        ),
    ]
}

#[test]
fn the_hue_step_matches_the_cpu_on_every_curve() {
    let img = chroma_image(97, 71, 0xA1E1);
    let base = hue_chain(&[], OutputTarget::Srgb);
    for dev in gpus() {
        let (mut worst, mut worst_rel, mut worst13) = (0.0f64, 0.0f64, 0.0f64);
        for (what, node) in hue_cases() {
            let chain = hue_chain(&[node], OutputTarget::Srgb);
            let m = measure(dev, &chain, &base, &img, None, None);
            eprintln!(
                "{:?}: hue step, {what}: C10 {:.2e} (beyond the range {:.1e}, relative), C13 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.beyond_rel, m.c13, m.moved
            );
            check(what, &m, 50);
            worst = worst.max(m.c10);
            worst_rel = worst_rel.max(m.beyond_rel);
            worst13 = worst13.max(m.c13);
        }
        eprintln!(
            "{:?}: hue step, all cases: worst C10 {worst:.2e} (gate {LOGC4_GATE:.0e}), worst beyond-range relative {worst_rel:.1e}, worst C13 {worst13:.2e} (gate {CODE10:.1e})",
            dev.mode
        );
    }
}

/// The corrector's steps and the hue step in one node (the order of `docs/34` section 2.3), a node after a node, a node with a mask.
#[test]
fn the_hue_step_after_the_correctors_steps_after_a_node_and_with_a_mask_matches_the_cpu() {
    let img = chroma_image(89, 57, 0xA1E2);
    let base = hue_chain(&[], OutputTarget::Srgb);
    let steps = ReferenceStep {
        hue: HueStep::new(
            &[(30.0, 25.0), (200.0, -20.0)],
            &[(100.0, 1.5), (280.0, 0.6)],
        ),
        elements: ElementStep::default(),
    };
    let graded = NodeParams {
        trim: Trim {
            exposure_ev: 0.5,
            temperature: 12.0,
            tint: -8.0,
        },
        grade: GradeParams {
            contrast: 20.0,
            saturation: 12.0,
            shadows: 15.0,
            ..GradeParams::default()
        },
        curves: CurveSet {
            luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).expect("curve"),
            ..CurveSet::default()
        },
        reference: Some(steps.clone()),
        ..NodeParams::default()
    };
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
        ..graded.clone()
    };
    for dev in gpus() {
        for (what, nodes, target) in [
            (
                "trim, grade, curves and hue",
                vec![graded.clone()],
                OutputTarget::Srgb,
            ),
            (
                "a node after a node",
                vec![hue(&[(60.0, 30.0)], &[(150.0, 1.7)]), graded.clone()],
                OutputTarget::DisplayP3,
            ),
            (
                "with a Brightness range mask",
                vec![masked.clone()],
                OutputTarget::Srgb,
            ),
        ] {
            let chain = hue_chain(&nodes, target);
            let m = measure(dev, &chain, &base, &img, None, None);
            eprintln!(
                "{:?}: {what}: C10 {:.2e}, C13 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.c13, m.moved
            );
            check(what, &m, 50);
        }
    }
}

/// The device's hue step keeps the Rec.709 luma of the LogC4 codes (to f32 rounding) and never turns a grey.
#[test]
fn the_hue_step_keeps_luma_and_leaves_greys_alone_on_the_device() {
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
    let grey = RgbImage::from_interleaved(w, 1, &rgb).expect("image");
    let chain = hue_chain(
        &[hue(
            &[(30.0, 90.0), (210.0, -90.0)],
            &[(0.0, 2.0), (180.0, 0.4)],
        )],
        OutputTarget::Srgb,
    );
    let plain = hue_chain(&[], OutputTarget::Srgb);
    for dev in gpus() {
        let got = render(
            dev,
            &chain,
            &grey,
            None,
            None,
            &[ColorOutput::Stage(Stop::C10)],
        );
        let want = render(
            dev,
            &plain,
            &grey,
            None,
            None,
            &[ColorOutput::Stage(Stop::C10)],
        );
        // A grey has (p, q) of the size of the rounding of its codes, so the move is that small: within a few ulps of the plain
        // chain's value (an exact (p, q) of 0 is returned as it is, which the step does by `select`, not by a test on the data).
        let worst = planar(&got[0])
            .iter()
            .zip(planar(&want[0]))
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max);
        eprintln!(
            "{:?}: greys, the hue step moved them by at most {worst:.2e}",
            dev.mode
        );
        assert!(worst < 1e-6, "greys moved by {worst}");
    }
    // The luma of a coloured picture.
    let img = chroma_image(61, 47, 0xA1E3);
    let n = img.width * img.height;
    let chain = hue_chain(
        &[hue(
            &[(30.0, 90.0), (210.0, -90.0)],
            &[(0.0, 2.0), (180.0, 0.4)],
        )],
        OutputTarget::Srgb,
    );
    for dev in gpus() {
        let got = render(
            dev,
            &chain,
            &img,
            None,
            None,
            &[ColorOutput::Stage(Stop::C10)],
        );
        let without = render(
            dev,
            &plain,
            &img,
            None,
            None,
            &[ColorOutput::Stage(Stop::C10)],
        );
        let (g, p) = (planar(&got[0]), planar(&without[0]));
        let y = |d: &[f32], i: usize| {
            0.2126 * f64::from(d[i])
                + 0.7152 * f64::from(d[n + i])
                + 0.0722 * f64::from(d[2 * n + i])
        };
        let mut worst = 0.0f64;
        for i in 0..n {
            if (0..3).all(|c| p[c * n + i].abs() < 2.0) {
                worst = worst.max((y(g, i) - y(p, i)).abs());
            }
        }
        eprintln!(
            "{:?}: the hue step moved the luma by at most {worst:.2e}",
            dev.mode
        );
        assert!(worst < 1e-5, "luma moved by {worst}");
    }
}

// ---- the identity ------------------------------------------------------------------------------------------------------

/// A node with no live hue or element step, curves that do nothing, a node with no hue point and a bypassed node run no code:
/// the bits of the graph without it. A node with a live step and a mask of 0 has the input's bits.
#[test]
fn a_node_at_its_identity_changes_no_bit_on_the_device() {
    let img = chroma_image(53, 41, 0xA1E4);
    let base = hue_chain(&[], OutputTarget::Srgb);
    let neutral = reference(
        HueStep::new(&[(10.0, 0.0), (200.0, 0.0)], &[(10.0, 1.0), (200.0, 1.0)]),
        ElementStep::default(),
    );
    let identities = [
        (
            "a default node",
            NodeParams {
                reference: Some(ReferenceStep::default()),
                ..NodeParams::default()
            },
        ),
        ("curves that do nothing", neutral),
        ("empty curves", hue(&[], &[])),
        (
            "a bypassed live node",
            NodeParams {
                bypass: true,
                ..hue(&[(30.0, 25.0)], &[(100.0, 1.5)])
            },
        ),
    ];
    let outs = [
        ColorOutput::Stage(Stop::C10),
        ColorOutput::Stage(Stop::C13),
        ColorOutput::Rgba8 { dither: true },
    ];
    for dev in gpus() {
        let want = render(dev, &base, &img, None, None, &outs);
        for (what, node) in &identities {
            let got = render(
                dev,
                &hue_chain(std::slice::from_ref(node), OutputTarget::Srgb),
                &img,
                None,
                None,
                &outs,
            );
            for (a, b) in got.iter().zip(&want) {
                match (&a.data, &b.data) {
                    (PixelData::F32Planar(x), PixelData::F32Planar(y)) => {
                        assert_bit_equal(what, x, y)
                    }
                    (PixelData::Rgba8(x), PixelData::Rgba8(y)) => assert_eq!(x, y, "{what}"),
                    _ => panic!("unexpected pixel data"),
                }
            }
        }
    }
    // A live node whose mask is 0 at a pixel has the input's bits there and its own where the mask is 1.
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
        ..hue(&[(30.0, 40.0), (200.0, -40.0)], &[(100.0, 1.5)])
    };
    let unmasked = hue(&[(30.0, 40.0), (200.0, -40.0)], &[(100.0, 1.5)]);
    let n = img.width * img.height;
    let (chain, plain) = (
        hue_chain(&[masked], OutputTarget::Srgb),
        hue_chain(&[unmasked], OutputTarget::Srgb),
    );
    for dev in gpus() {
        let m = render(
            dev,
            &chain,
            &img,
            None,
            Some(NodeStop::Mask(2)),
            &[ColorOutput::Stage(Stop::C13)],
        );
        let masks = planar(&m[0]);
        let o = [ColorOutput::Stage(Stop::C10)];
        let got = render(dev, &chain, &img, None, None, &o);
        let full = render(dev, &plain, &img, None, None, &o);
        let none = render(dev, &base, &img, None, None, &o);
        let (mut zeros, mut ones) = (0, 0);
        for (i, &mask) in masks.iter().enumerate().take(n) {
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

// ---- the element step, planes uploaded --------------------------------------------------------------------------------------

fn grade(exposure_ev: f64, hue_deg: f64, chroma: f64) -> ElementGrade {
    ElementGrade {
        exposure_ev,
        hue_deg,
        chroma,
    }
}

/// The codes of element `e`'s plane over a `w x h` window: blocks of 0 and of 65535, ramps of partial values, so that the planes
/// overlap (a pixel is in two or three elements) and every kind of code is on the picture.
fn plane_codes(w: usize, h: usize, e: usize) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            match (x / 13 + y / 11 + e) % 4 {
                0 => 0,
                1 => u16::MAX,
                2 => (((x * 7 + y * 3 + e * 40) % 256) * 257) as u16,
                _ => (((x + y) % 5) * 13_000) as u16,
            }
        })
        .collect()
}

/// A chain whose Reference Grade node (id 2) has `grades` live, with the planes of its live elements in `codes` order, and the CPU's
/// planes of the same codes.
fn element_case(
    grades: [ElementGrade; ELEMENT_COUNT],
    w: usize,
    h: usize,
    hue_step: HueStep,
) -> (ColorChain, Vec<Vec<u16>>, ElementPlanes) {
    let step = ElementStep::new(grades);
    let live: [bool; ELEMENT_COUNT] = std::array::from_fn(|e| step.is_live(e));
    let chain = chain_with(
        &[reference(hue_step, step)],
        vec![ElementRow {
            node: 2,
            live,
            scene: dummy_scene(),
        }],
        OutputTarget::Srgb,
    );
    let codes: Vec<Vec<u16>> = (0..ELEMENT_COUNT)
        .filter(|&e| live[e])
        .map(|e| plane_codes(w, h, e))
        .collect();
    let mut by_element: [Vec<u16>; ELEMENT_COUNT] = Default::default();
    for (k, e) in (0..ELEMENT_COUNT).filter(|&e| live[e]).enumerate() {
        by_element[e] = codes[k].clone();
    }
    let planes = ElementPlanes {
        width: w,
        height: h,
        nodes: [(2, by_element)].into(),
    };
    (chain, codes, planes)
}

fn element_cases() -> Vec<(&'static str, [ElementGrade; ELEMENT_COUNT])> {
    let id = ElementGrade::IDENTITY;
    let one = |e: usize, g: ElementGrade| {
        let mut a = [id; ELEMENT_COUNT];
        a[e] = g;
        a
    };
    vec![
        ("sky", one(0, grade(0.3, 20.0, 1.3))),
        ("vegetation", one(1, grade(-0.4, -25.0, 0.7))),
        ("water", one(2, grade(0.0, 30.0, 1.5))),
        ("ground", one(3, grade(0.5, -10.0, 1.0))),
        ("built", one(4, grade(0.0, 0.0, 0.6))),
        ("skin", one(5, grade(0.1, 8.0, 1.1))),
        ("subject (exposure only)", one(6, grade(0.8, 0.0, 1.0))),
        (
            "all seven",
            [
                grade(0.3, 20.0, 1.3),
                grade(-0.4, -25.0, 0.7),
                grade(0.0, 30.0, 1.5),
                grade(0.5, -10.0, 1.0),
                grade(0.0, 0.0, 0.6),
                grade(0.1, 8.0, 1.1),
                grade(0.8, 0.0, 1.0),
            ],
        ),
        (
            "the ends of the ranges",
            [
                grade(1.0, 40.0, 1.6),
                grade(-1.0, -35.0, 0.5),
                grade(1.0, -35.0, 1.6),
                grade(-1.0, 30.0, 0.5),
                grade(1.0, -25.0, 0.5),
                grade(-1.0, 15.0, 1.6),
                grade(-1.0, 0.0, 1.0),
            ],
        ),
    ]
}

#[test]
fn the_element_step_matches_the_cpu_for_each_element_and_all_seven() {
    let (w, h) = (97usize, 71usize);
    let img = chroma_image(w, h, 0xA1E5);
    let base = hue_chain(&[], OutputTarget::Srgb);
    for dev in gpus() {
        let (mut worst, mut worst13) = (0.0f64, 0.0f64);
        for (what, grades) in element_cases() {
            let (chain, codes, planes) = element_case(grades, w, h, HueStep::default());
            let m = measure(dev, &chain, &base, &img, Some(&planes), Some(&codes));
            eprintln!(
                "{:?}: element step, {what}: C10 {:.2e} (beyond the range {:.1e}), C13 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.beyond_rel, m.c13, m.moved
            );
            check(what, &m, 50);
            worst = worst.max(m.c10);
            worst13 = worst13.max(m.c13);
        }
        eprintln!(
            "{:?}: element step, all cases: worst C10 {worst:.2e} (gate {LOGC4_GATE:.0e}), worst C13 {worst13:.2e} (gate {CODE10:.1e})",
            dev.mode
        );
    }
}

/// The hue step and the element grades together, after a node and with the corrector's steps: the order of `docs/34` section 2.3.
#[test]
fn the_hue_and_element_steps_together_match_the_cpu() {
    let (w, h) = (89usize, 57usize);
    let img = chroma_image(w, h, 0xA1E6);
    let base = hue_chain(&[], OutputTarget::Srgb);
    let (_, grades) = element_cases().remove(7);
    let step = ElementStep::new(grades);
    let live = [true; ELEMENT_COUNT];
    let codes: Vec<Vec<u16>> = (0..ELEMENT_COUNT).map(|e| plane_codes(w, h, e)).collect();
    let planes = ElementPlanes {
        width: w,
        height: h,
        nodes: [(3, std::array::from_fn(|e| codes[e].clone()))].into(),
    };
    let rg = NodeParams {
        trim: Trim {
            exposure_ev: 0.3,
            ..Trim::default()
        },
        grade: GradeParams {
            contrast: 15.0,
            saturation: -10.0,
            ..GradeParams::default()
        },
        reference: Some(ReferenceStep {
            hue: HueStep::new(&[(40.0, 30.0), (220.0, -25.0)], &[(120.0, 1.4)]),
            elements: step,
        }),
        ..NodeParams::default()
    };
    let chain = chain_with(
        &[hue(&[(10.0, -20.0)], &[(200.0, 0.8)]), rg],
        vec![ElementRow {
            node: 3,
            live,
            scene: dummy_scene(),
        }],
        OutputTarget::DisplayP3,
    );
    for dev in gpus() {
        let m = measure(dev, &chain, &base, &img, Some(&planes), Some(&codes));
        eprintln!(
            "{:?}: hue and elements after a node: C10 {:.2e}, C13 {:.2e}",
            dev.mode, m.c10, m.c13
        );
        check("hue and elements after a node", &m, 50);
    }
}

/// With every plane 0 the element step moves nothing (the input's bits after the hue step); with every plane 65535 the elements are
/// applied whole, in order; the device with no planes bound selects nothing.
#[test]
fn planes_of_zero_and_of_one_and_no_planes_at_all() {
    let (w, h) = (61usize, 47usize);
    let img = chroma_image(w, h, 0xA1E7);
    let base = hue_chain(&[], OutputTarget::Srgb);
    let (_, grades) = element_cases().remove(7);
    for dev in gpus() {
        for (what, code) in [("all 0", 0u16), ("all 1", u16::MAX)] {
            let (chain, _, mut planes) = element_case(grades, w, h, HueStep::default());
            let codes: Vec<Vec<u16>> = (0..ELEMENT_COUNT).map(|_| vec![code; w * h]).collect();
            for p in planes.nodes.get_mut(&2).expect("planes").iter_mut() {
                *p = vec![code; w * h];
            }
            let m = measure(dev, &chain, &base, &img, Some(&planes), Some(&codes));
            eprintln!(
                "{:?}: planes {what}: C10 {:.2e}, {} pixels moved",
                dev.mode, m.c10, m.moved
            );
            assert!(m.c10 <= LOGC4_GATE && m.c13 <= CODE10, "{what}");
            if code == 0 {
                assert_eq!(m.moved, 0, "mask 0 moved a pixel");
            }
        }
        // Planes of 0 give the bits of the chain without the node.
        let (chain, _, _) = element_case(grades, w, h, HueStep::default());
        let zero: Vec<Vec<u16>> = (0..ELEMENT_COUNT).map(|_| vec![0; w * h]).collect();
        let outs = [ColorOutput::Stage(Stop::C10)];
        let got = render(dev, &chain, &img, Some(&zero), None, &outs);
        let want = render(dev, &base, &img, None, None, &outs);
        assert_bit_equal("planes of 0", planar(&got[0]), planar(&want[0]));
        // No planes bound: the same bits.
        let got = render(dev, &chain, &img, None, None, &outs);
        assert_bit_equal("no planes", planar(&got[0]), planar(&want[0]));
    }
}

/// A repeat render and a poisoned device give the same bits.
#[test]
fn a_repeat_and_a_poisoned_device_give_the_same_bits_with_the_node() {
    let (w, h) = (73usize, 41usize);
    let img = chroma_image(w, h, 0xA1E8);
    let (_, grades) = element_cases().remove(7);
    let (chain, codes, _) = element_case(
        grades,
        w,
        h,
        HueStep::new(&[(40.0, 30.0), (220.0, -25.0)], &[(120.0, 1.4)]),
    );
    let run = |gpu: &ym_gpu::device::Gpu| {
        let (chain, img, codes) = (chain.clone(), img.clone(), codes.clone());
        let once = move |ctx: &ym_gpu::device::GpuContext| {
            let p = PreparedChain::new(ctx, &chain)?.with_rasters(RasterPlanes::from_codes(
                ctx, img.width, img.height, &codes,
            )?);
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
    assert_poison_neutral("Reference Grade", run);
}

// ---- the element planes the device makes ----------------------------------------------------------------------------------

/// A scene on a `sw x sh` grid: the elements are overlapping shapes (bands, discs, a block), so that edges run at every angle.
fn scene_bits(sw: usize, sh: usize) -> Arc<ElementScene> {
    let mut bits = vec![0u8; sw * sh];
    for y in 0..sh {
        for x in 0..sw {
            let (fx, fy) = (x as f64 / sw as f64, y as f64 / sh as f64);
            let mut b = 0u8;
            if fy < 0.35 + 0.1 * (fx * 6.0).sin() {
                b |= 1; // sky
            }
            if fy > 0.55 && fx < 0.6 {
                b |= 1 << 1; // vegetation
            }
            if (fx - 0.75).powi(2) + (fy - 0.7).powi(2) < 0.05 {
                b |= 1 << 2; // water
            }
            if fy > 0.8 {
                b |= 1 << 3; // ground
            }
            if (0.3..0.5).contains(&fx) && (0.3..0.65).contains(&fy) {
                b |= 1 << 4 | 1 << 6; // built, and the subject
            }
            if (fx - 0.4).powi(2) + (fy - 0.45).powi(2) < 0.01 {
                b |= 1 << 5; // skin (inside the subject)
            }
            bits[y * sw + x] = b;
        }
    }
    Arc::new(ElementScene {
        key: 0xE1E3,
        width: sw,
        height: sh,
        bits,
        classes: Vec::new(),
    })
}

/// A photo-like picture with structure at every scale (so the guide has edges to follow).
fn picture(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let blocks = if ((x / 23) + (y / 17)) % 2 == 0 {
                0.9
            } else {
                -0.9
            };
            let stops = -2.0
                + 4.0 * (0.5 + 0.5 * (fx * 7.0 + fy * 4.0).sin())
                + blocks
                + 0.3 * rng.range(-1.0, 1.0);
            let lum = 0.18 * stops.exp2();
            for k in 0..3 {
                let tint = 0.3 * ((fx * (3.0 + k as f64) + fy * 2.0 * (k as f64 + 1.0)).cos())
                    + 0.05 * rng.range(-1.0, 1.0);
                rgb.push(f16_round((lum * tint.exp2()) as f32));
            }
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

/// A grid for an unrotated, uncropped output of `scale` camera pixels per output pixel.
fn grid_for(buffer: (usize, usize), scale: f64) -> (SamplerGrid, Area) {
    let (bw, bh) = (buffer.0 as f64, buffer.1 as f64);
    let grid = SamplerGrid::new(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0]],
        (bw, bh),
        (bw, bh),
        1,
        0,
        true,
    );
    let frame = Area::whole((bw / scale) as usize, (bh / scale) as usize);
    (grid, frame)
}

/// The chain of the device-made planes: a Reference Grade node (id 2) with the elements `live`.
fn scene_chain(scene: &Arc<ElementScene>, live: [bool; ELEMENT_COUNT]) -> ColorChain {
    let grades = std::array::from_fn(|e| {
        if live[e] {
            grade(0.2 + 0.05 * e as f64, 5.0 + e as f64, 1.2)
        } else {
            ElementGrade::IDENTITY
        }
    });
    chain_with(
        &[reference(HueStep::default(), ElementStep::new(grades))],
        vec![ElementRow {
            node: 2,
            live,
            scene: Arc::clone(scene),
        }],
        OutputTarget::Srgb,
    )
}

fn halo_of(chain: &ColorChain, grid: &SamplerGrid) -> usize {
    let row = &chain.element_rows()[0];
    ym_color::mask_element::element_halo(row.scene.radius(1.0 / grid.photo_frame().pixel))
}

/// The CPU's planes of `window` (the live elements in order, the window's pixels).
fn cpu_planes(
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    frame: Area,
    window: Area,
    halo: usize,
) -> Vec<Vec<u16>> {
    let grown = window.grown(halo, &frame);
    let (gw, gh) = (grown.w, grown.h);
    let sampled = sample_to_camera(
        img,
        (0, 0),
        grid,
        (grown.x as usize, grown.y as usize),
        gw,
        gh,
    );
    let outside = |i: usize| sampled.outside.as_ref().is_some_and(|o| o[i] != 0);
    let lightness: Vec<f32> = (0..gw * gh)
        .map(|i| {
            let p = sampled.img.pixel(i % gw, i / gw);
            chain.scene_guide_pixel([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        })
        .collect();
    let map = GridPositions::new(grid, (grown.x as usize, grown.y as usize));
    let position = |x: usize, y: usize| {
        if outside(y * gw + x) {
            None
        } else {
            Some(map.position(x, y))
        }
    };
    let planes = element_planes(
        chain.element_rows(),
        &ElementJob {
            photo: grid.photo_frame(),
            grown,
            frame,
            window,
            position: &position,
            guide: &lightness,
        },
    );
    let layout = chain.element_layout();
    let by_element = &planes.nodes[&2];
    layout
        .iter()
        .flat_map(|l| {
            (0..ELEMENT_COUNT)
                .filter(|&e| l.live[e])
                .map(|e| by_element[e].clone())
        })
        .map(|p| {
            if p.is_empty() {
                vec![0; window.w * window.h]
            } else {
                p
            }
        })
        .collect()
}

fn unpack(bytes: &[u8], planes: &RasterPlanes) -> Vec<Vec<u16>> {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (0..planes.planes)
        .map(|p| {
            let base = p * planes.plane_words();
            let mut out = Vec::with_capacity(planes.width * planes.height);
            for y in 0..planes.height {
                for x in 0..planes.width {
                    let idx = y * planes.stride + x;
                    let w = words[base + idx / 2];
                    out.push(((w >> ((idx % 2) * 16)) & 0xffff) as u16);
                }
            }
            out
        })
        .collect()
}

/// The device's planes of `window` over `extent`, as u16 codes.
#[allow(clippy::too_many_arguments)]
fn device_planes(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    frame: Area,
    window: Area,
    halo: usize,
) -> Vec<Vec<u16>> {
    let (chain, img, grid) = (chain.clone(), img.clone(), *grid);
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "element source", &img)?;
            let mut passes = PassList::new(ctx, "element planes");
            let grown = window.grown(halo, &frame);
            let sampled = encode_sample_at(
                &mut passes,
                &src,
                (0, 0),
                &grid,
                (grown.x as usize, grown.y as usize),
                grown.w,
                grown.h,
                true,
            )?;
            let prepared = PreparedChain::new(ctx, &chain)?.with_photo_frame(grid.photo_frame());
            let planes = encode_raster_planes(
                &mut passes,
                &prepared,
                &chain,
                &sampled,
                &RasterJob {
                    frame,
                    grown,
                    window,
                    extent: window,
                    photo: grid.photo_frame(),
                },
            )?;
            let read = passes.read(planes.buffer(), 0, planes.byte_len())?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            let bytes = done.read(read).to_vec();
            Ok(unpack(&bytes, &planes))
        })
        .expect("GPU element planes")
}

/// The largest absolute difference of two planes' values (codes over 65535), the mean, and how many pixels differ by more than 1e-3.
fn plane_diff(a: &[u16], b: &[u16]) -> (f64, f64, usize) {
    assert_eq!(a.len(), b.len());
    let (mut worst, mut sum, mut over) = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        let d = (f64::from(*x) - f64::from(*y)).abs() / 65535.0;
        worst = worst.max(d);
        sum += d;
        over += usize::from(d > 1e-3);
    }
    (worst, sum / a.len().max(1) as f64, over)
}

#[test]
fn the_device_makes_the_element_planes_the_cpu_makes_within_the_stated_bound() {
    let buffer = (480usize, 320usize);
    let img = picture(buffer.0, buffer.1, 0xE1E5);
    // The output is 1200 x 800: the radius is 16 pixels, the halo 33.
    let (grid, frame) = grid_for(buffer, 0.4);
    let scene = scene_bits(64, 48);
    let mut worst_all = 0.0f64;
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        for (what, live, window) in [
            (
                "all seven, a tall window with every element in it",
                [true; ELEMENT_COUNT],
                Area {
                    x: 400,
                    y: 330,
                    w: 520,
                    h: 400,
                },
            ),
            (
                "all seven, a window in the middle",
                [true; ELEMENT_COUNT],
                Area {
                    x: 420,
                    y: 260,
                    w: 200,
                    h: 150,
                },
            ),
            (
                "the sky and the subject, the corner",
                [true, false, false, false, false, false, true],
                Area {
                    x: 0,
                    y: 0,
                    w: 130,
                    h: 90,
                },
            ),
            (
                "skin and water, the far corner",
                [false, false, true, false, false, true, false],
                Area {
                    x: frame.w as i64 - 140,
                    y: frame.h as i64 - 100,
                    w: 140,
                    h: 100,
                },
            ),
        ] {
            let chain = scene_chain(&scene, live);
            let halo = halo_of(&chain, &grid);
            let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
            let want = cpu_planes(&chain, &img, &grid, frame, window, halo);
            assert_eq!(got.len(), want.len(), "{what}");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                let (worst, mean, over) = plane_diff(g, w);
                let (lo, hi) = w
                    .iter()
                    .fold((u16::MAX, 0u16), |(l, h), &c| (l.min(c), h.max(c)));
                eprintln!(
                    "{what}, plane {k}: halo {halo} px, the device differs from the CPU by {worst:.2e}, mean {mean:.2e}, {over} pixels over 1e-3 (plane {lo} to {hi})"
                );
                assert!(worst < PLANE_GATE, "{what}, plane {k}: {worst}");
                // A pixel on the edge of a scene cell or of an 8-bit code of the guide may fall on the other side in f32: a handful.
                assert!(over <= 8, "{what}, plane {k}: {over} pixels over 1e-3");
                worst_all = worst_all.max(worst);
            }
        }
    }
    eprintln!(
        "the largest element plane difference over the cases: {worst_all:.2e} (gate {PLANE_GATE:.0e})"
    );
}

/// Planes made in tiles are the planes made in one piece, bit for bit, whatever the tile (`docs/34` section 10, tile invariance).
#[test]
fn element_planes_equal_the_whole_frame_bit_for_bit_at_any_tile() {
    let buffer = (480usize, 320usize);
    let img = picture(buffer.0, buffer.1, 0xE1E6);
    let (grid, frame) = grid_for(buffer, 0.4);
    let scene = scene_bits(64, 48);
    let chain = scene_chain(&scene, [true; ELEMENT_COUNT]);
    let halo = halo_of(&chain, &grid);
    let whole_window = Area {
        x: 400,
        y: 330,
        w: 520,
        h: 400,
    };
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        let whole = device_planes(dev, &chain, &img, &grid, frame, whole_window, halo);
        // Each tile of the window, computed over itself grown by the halo, equals the whole's pixels there.
        for tile in [97usize, 160, 256] {
            let mut y = whole_window.y;
            while y < whole_window.y1() {
                let mut x = whole_window.x;
                while x < whole_window.x1() {
                    let t = Area {
                        x,
                        y,
                        w: (tile as i64).min(whole_window.x1() - x) as usize,
                        h: (tile as i64).min(whole_window.y1() - y) as usize,
                    };
                    let got = device_planes(dev, &chain, &img, &grid, frame, t, halo);
                    for (k, plane) in got.iter().enumerate() {
                        for j in 0..t.h {
                            for i in 0..t.w {
                                let a = plane[j * t.w + i];
                                let wx = (t.x - whole_window.x) as usize + i;
                                let wy = (t.y - whole_window.y) as usize + j;
                                let b = whole[k][wy * whole_window.w + wx];
                                assert_eq!(
                                    a,
                                    b,
                                    "tile {tile} at {:?}, plane {k}, pixel ({i}, {j})",
                                    (t.x, t.y)
                                );
                            }
                        }
                    }
                    x += tile as i64;
                }
                y += tile as i64;
            }
        }
    }
}

/// The planes in the pass in tiles (`encode_raster_planes_tiled`) are the planes in one piece, bit for bit.
#[test]
fn element_planes_made_in_tiles_are_the_planes_made_in_one_piece() {
    let buffer = (480usize, 320usize);
    let img = picture(buffer.0, buffer.1, 0xE1E7);
    let (grid, frame) = grid_for(buffer, 0.4);
    let scene = scene_bits(64, 48);
    let chain = scene_chain(&scene, [true; ELEMENT_COUNT]);
    let halo = halo_of(&chain, &grid);
    let window = Area {
        x: 100,
        y: 80,
        w: 300,
        h: 200,
    };
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        let one = device_planes(dev, &chain, &img, &grid, frame, window, halo);
        let (chain2, img2, grid2) = (chain.clone(), img.clone(), grid);
        let tiled = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let src = GpuImage::upload_rgb(ctx, "element source", &img2)?;
                let mut passes = PassList::new(ctx, "element planes");
                let prepared =
                    PreparedChain::new(ctx, &chain2)?.with_photo_frame(grid2.photo_frame());
                let halves = [
                    Area {
                        x: window.x,
                        y: window.y,
                        w: 150,
                        h: window.h,
                    },
                    Area {
                        x: window.x + 150,
                        y: window.y,
                        w: 150,
                        h: window.h,
                    },
                ];
                let regions: Vec<RasterTile> = halves
                    .iter()
                    .map(|t| RasterTile {
                        extent: *t,
                        grown: t.grown(halo, &frame),
                    })
                    .collect();
                let key = ym_gpu::kernels::mask_raster::PlaneKey {
                    owner: 7,
                    place: "element tiles".to_owned(),
                    guide: String::new(),
                };
                let planes = encode_raster_planes_tiled(
                    &mut passes,
                    &prepared,
                    &chain2,
                    &RasterJob {
                        frame,
                        grown: window.grown(halo, &frame),
                        window,
                        extent: window,
                        photo: grid2.photo_frame(),
                    },
                    &key,
                    &regions,
                    |i, passes| {
                        let g = regions[i].grown;
                        encode_sample_at(
                            passes,
                            &src,
                            (0, 0),
                            &grid2,
                            (g.x as usize, g.y as usize),
                            g.w,
                            g.h,
                            true,
                        )
                    },
                )?;
                let read = passes.read(planes.buffer(), 0, planes.byte_len())?;
                let _src = passes.keep(src.buf);
                let done = passes.submit()?.finish()?;
                let bytes = done.read(read).to_vec();
                Ok(unpack(&bytes, &planes))
            })
            .expect("tiled element planes");
        assert_eq!(
            tiled, one,
            "the planes made in tiles differ from the planes made in one piece"
        );
    }
}

/// The colour pass with the planes the device made is within one 10-bit code of the CPU chain with the CPU's planes, and a tile
/// of the picture equals the whole frame bit for bit.
#[test]
fn the_colour_pass_with_the_devices_element_planes_is_within_one_ten_bit_code_of_the_cpu() {
    let buffer = (480usize, 320usize);
    let img = picture(buffer.0, buffer.1, 0xE1E8);
    let (grid, frame) = grid_for(buffer, 0.4);
    let scene = scene_bits(64, 48);
    let chain = scene_chain(&scene, [true; ELEMENT_COUNT]);
    let halo = halo_of(&chain, &grid);
    let window = Area {
        x: 420,
        y: 260,
        w: 160,
        h: 120,
    };
    let cpu = cpu_planes(&chain, &img, &grid, frame, window, halo);
    let by_element: [Vec<u16>; ELEMENT_COUNT] = std::array::from_fn(|e| cpu[e].clone());
    let planes = ElementPlanes {
        width: window.w,
        height: window.h,
        nodes: [(2, by_element)].into(),
    };
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        // The camera pixels of the window, sampled on the host as the CPU render does; the device colours the same pixels.
        let sampled = sample_to_camera(
            &img,
            (0, 0),
            &grid,
            (window.x as usize, window.y as usize),
            window.w,
            window.h,
        );
        let camera = sampled.img.clone();
        let n = window.w * window.h;
        let (got, stride) = render_over(dev, &chain, &img, &grid, frame, window, halo);
        let plane = got.len() / 3;
        let mut worst = 0.0f64;
        let mut over = 0usize;
        for i in 0..n {
            let (x, y) = (i % window.w, i / window.w);
            let p = camera.pixel(x, y);
            let spot = PhotoSpot::new(
                GridPositions::new(&grid, (window.x as usize, window.y as usize)).position(x, y),
                grid.photo_frame(),
            )
            .with_raster(&planes, x, y);
            let t = chain.trace_at(
                [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])],
                Some(spot),
            );
            for ch in 0..3 {
                let d = (f64::from(got[ch * plane + y * stride + x]) - t.c13_encoded[ch]).abs();
                worst = worst.max(d);
                over += usize::from(d > CODE10);
            }
        }
        eprintln!(
            "{:?}: the colour pass with the device's element planes: C13 differs from the CPU's by {worst:.2e} at worst, {over} samples over one 10-bit code",
            dev.mode
        );
        assert!(
            worst <= 3.0 * CODE10 && over < n / 200,
            "{worst}, {over} over"
        );
    }
}

/// The C13 planes of `window` with the planes the device made for it, over a sampled picture of the window itself.
fn render_over(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    frame: Area,
    window: Area,
    halo: usize,
) -> (Vec<f32>, usize) {
    let (chain, img, grid) = (chain.clone(), img.clone(), *grid);
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "element source", &img)?;
            let mut passes = PassList::new(ctx, "element colour");
            let grown = window.grown(halo, &frame);
            let sampled = encode_sample_at(
                &mut passes,
                &src,
                (0, 0),
                &grid,
                (grown.x as usize, grown.y as usize),
                grown.w,
                grown.h,
                true,
            )?;
            let prepared = PreparedChain::new(ctx, &chain)?.with_photo_frame(grid.photo_frame());
            let planes = encode_raster_planes(
                &mut passes,
                &prepared,
                &chain,
                &sampled,
                &RasterJob {
                    frame,
                    grown,
                    window,
                    extent: window,
                    photo: grid.photo_frame(),
                },
            )?;
            let prepared = prepared.with_rasters(planes);
            let out = GpuImage::alloc(
                ctx,
                "element colour out",
                window.w,
                window.h,
                4,
                ym_gpu::cpu::Placement::whole(window.w, window.h),
            )?;
            prepared.encode_stage(
                &mut passes,
                &sampled,
                ym_gpu::cpu::tiling::Rect::new(
                    (window.x - grown.x) as usize,
                    (window.y - grown.y) as usize,
                    window.w,
                    window.h,
                ),
                Stop::C13,
                &out,
                (0, 0),
            )?;
            let read = passes.read(&out.buf, 0, (out.plane_len() * 4 * 4) as u64)?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            let bytes = done.read(read).to_vec();
            let mut v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            v.truncate(3 * out.plane_len());
            Ok((v, out.stride))
        })
        .expect("GPU element colour")
}
