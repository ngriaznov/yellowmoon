//! N-10 (`docs/23-nodes.md` sections 3.4, 4 and 5.4, `docs/algorithms/06-masks.md` sections 3 and 4): the mask
//! shapes and ranges, the combination of the rows, the key links and the mask blend in the fused colour
//! kernel against the CPU reference (`ym_color::mask`, `ym_color::display::ColorChain`).
//!
//! The sampler writes each pixel's place in the photo frame (planes 5 and 6 of its output, `encode_sample_at`)
//! and the colour pass reads it: the pictures here go through both, over grids that crop, straighten, turn,
//! mirror, shrink and bend, and the CPU reference gets the same positions from `GridPositions`. The gates are
//! the colour kernel's: 2e-4 in LogC4 for the blended picture (for nodes whose effect is at most 0.25 in
//! LogC4, `docs/23` section 11) and one 10-bit code after C13; the mask itself within 5e-4.

mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::chains;
use common::exact::{assert_bit_equal, assert_poison_neutral};
use common::{TestGpu, gpus};
use ym_color::display::{
    ChainParams, ColorChain, LookSetting, MainNode, NodeParams, NodeSet, OutputTarget, Trim,
};
use ym_color::grade::GradeParams;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoMap, PhotoSpot, PixelAt};
use ym_color::plan::{self, MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop as PlanStop};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::sampler::{
    GridPositions, SamplerGrid, SamplerWarp, photo_positions, sample_to_camera,
};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::device::StorageMode;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::color::{NodeStop, PreparedChain, Stop};
use ym_gpu::kernels::sampler::encode_sample_at;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// C10, the blended picture, in LogC4 code values.
const LOGC4_GATE: f64 = 2e-4;
/// One 10-bit code after C13.
const CODE10: f64 = 1.0 / 1023.0;
/// The mask `m` itself.
const MASK_GATE: f64 = 5e-4;

// ---- the scene and the geometry ------------------------------------------------------------------------

/// A photo-like scene: positive camera values over about eight stops that vary smoothly with a texture of noise,
/// colours around neutral (so that a Colour range at the origin and a Brightness range in the middle each take a
/// good part of it), no outlier. The stages of a node at the photo's range stay where f32 is as exact as the
/// 2e-4 gate asks (`gpu_nodes.rs` keeps the wild scene for the plain nodes; a bicubic sample next to a value of
/// 60 can be negative, and its LogC4 code is then -37000).
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = common::Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let stops =
                -3.5 + 6.5 * (0.5 + 0.5 * (fx * 5.0 + fy * 3.0).sin()) + 0.8 * rng.range(-1.0, 1.0);
            let lum = 0.18 * stops.exp2();
            for k in 0..3 {
                // Each channel a little off the luminance, by up to 0.6 stop, smoothly and by noise.
                let tint = 0.35 * ((fx * (3.0 + k as f64) + fy * 2.0 * (k as f64 + 1.0)).cos())
                    + 0.25 * rng.range(-1.0, 1.0);
                rgb.push(common::f16_round((lum * tint.exp2()) as f32));
            }
        }
    }
    RgbImage::from_interleaved(w, h, &rgb).expect("image")
}

fn swaps(exif: u16, turns: u8) -> bool {
    ((5..=8).contains(&exif)) != (turns % 2 == 1)
}

/// A grid for an output pixel `q` -> canvas `R(-angle) (scale q + t - c) + c` (`c` the canvas centre).
fn grid_of(
    (bw, bh): (usize, usize),
    exif: u16,
    turns: u8,
    angle_deg: f64,
    scale: f64,
    t: (f64, f64),
) -> SamplerGrid {
    let (cw, ch) = if swaps(exif, turns) {
        (bh, bw)
    } else {
        (bw, bh)
    };
    let (s, c) = (-angle_deg).to_radians().sin_cos();
    let (cx, cy) = (cw as f64 * 0.5, ch as f64 * 0.5);
    let (tx, ty) = (t.0 * cw as f64, t.1 * ch as f64);
    SamplerGrid::new(
        [
            [c * scale, -s * scale, c * (tx - cx) - s * (ty - cy) + cx],
            [s * scale, c * scale, s * (tx - cx) + c * (ty - cy) + cy],
        ],
        (cw as f64, ch as f64),
        (bw as f64, bh as f64),
        exif,
        turns,
        SamplerGrid::is_unrotated_angle(angle_deg),
    )
}

/// The attachment's bend (data version 5), at `extra` times its strength, fitted so that the canvas's edge maps
/// inside the photo.
fn bend(extra: f64, canvas: (f64, f64)) -> SamplerWarp {
    const X: [f64; 5] = [-0.0402, -0.0494, 0.0069, 0.0128, 0.0329];
    const Y: [f64; 5] = [-0.1157, -0.0196, 0.0294, 0.0297, -0.0529];
    let mut w = SamplerWarp::new(
        X.map(|c| c * extra),
        Y.map(|c| c * extra),
        -0.03,
        0.01,
        1.0,
        3,
    );
    let inside = |w: &SamplerWarp| {
        (0..=100).all(|i| {
            let t = f64::from(i) / 100.0;
            [
                (t * canvas.0, 0.0),
                (t * canvas.0, canvas.1),
                (0.0, t * canvas.1),
                (canvas.0, t * canvas.1),
            ]
            .into_iter()
            .all(|(x, y)| {
                let (px, py) = w.canvas_to_photo(canvas, x, y);
                px >= 1.0 && px <= canvas.0 - 1.0 && py >= 1.0 && py <= canvas.1 - 1.0
            })
        })
    };
    let mut zoom = 1.0;
    while zoom < 4.0 {
        w.zoom = zoom;
        if inside(&w) {
            break;
        }
        zoom += 0.03125;
    }
    w
}

/// The grids the parity runs over, each with the output size to render: an exact copy, a crop, a straightened and
/// scaled photo, a photo turned and mirrored, a shrink that needs the box kernel, and one through the lens's bend.
fn grids(buffer: (usize, usize)) -> Vec<(&'static str, SamplerGrid, usize, usize)> {
    let (bw, bh) = buffer;
    let dim = |exif, turns, scale: f64| {
        let (cw, ch) = if swaps(exif, turns) {
            (bh, bw)
        } else {
            (bw, bh)
        };
        (
            ((cw as f64 / scale) as usize).clamp(8, 90),
            ((ch as f64 / scale) as usize).clamp(8, 70),
        )
    };
    let mut v = Vec::new();
    for (name, exif, turns, angle, scale, t) in [
        ("exact", 1u16, 0u8, 0.0, 1.0, (0.0, 0.0)),
        ("crop and shift", 1, 0, 0.0, 1.0, (0.1, 0.12)),
        ("straightened, magnified", 1, 0, 7.5, 0.8, (0.0, 0.0)),
        ("turned and mirrored", 6, 1, 0.0, 1.0, (0.0, 0.0)),
        ("mirrored, straightened", 7, 2, -12.0, 1.1, (0.05, -0.05)),
        ("shrunk by 2.3 (box)", 1, 0, 0.0, 2.3, (0.0, 0.0)),
        ("shrunk, turned, tilted (box)", 8, 3, 19.0, 1.7, (0.0, 0.0)),
    ] {
        let (w, h) = dim(exif, turns, scale);
        v.push((name, grid_of(buffer, exif, turns, angle, scale, t), w, h));
    }
    let (w, h) = dim(1, 0, 1.0);
    let warped = grid_of(buffer, 1, 0, 0.0, 1.0, (0.0, 0.0))
        .with_warp(Some(bend(1.0, (bw as f64, bh as f64))));
    v.push(("lens bend", warped, w, h));
    let (w, h) = dim(6, 1, 1.4);
    let warped = grid_of(buffer, 6, 1, 3.0, 1.4, (0.0, 0.0))
        .with_warp(Some(bend(0.5, (bh as f64, bw as f64))));
    v.push(("lens bend, turned, tilted", warped, w, h));
    v
}

// ---- nodes and plans ------------------------------------------------------------------------------------

fn linear(center: [f64; 2], angle_deg: f64, feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Linear {
        center,
        angle_deg,
        feather,
    })
}

fn radial(center: [f64; 2], radii: [f64; 2], angle_deg: f64, feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Radial {
        center,
        radii,
        angle_deg,
        feather,
    })
}

fn colour(center: [f64; 2], width: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::ColorRange {
        center,
        width,
        softness,
    })
}

fn brightness(low: f64, high: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::BrightnessRange {
        low,
        high,
        softness,
    })
}

fn masks(rows: Vec<MaskRow>) -> NodeMasks {
    NodeMasks {
        rows,
        invert: false,
    }
}

fn subtract(row: MaskRow) -> MaskRow {
    MaskRow {
        subtract: true,
        ..row
    }
}

fn invert(row: MaskRow) -> MaskRow {
    MaskRow {
        invert: true,
        ..row
    }
}

fn opacity(row: MaskRow, o: f64) -> MaskRow {
    MaskRow { opacity: o, ..row }
}

/// A node whose effect on the picture is visible and at most about 0.25 in LogC4: an exposure trim, a contrast
/// and a saturation.
fn effect() -> NodeParams {
    NodeParams {
        trim: Trim {
            exposure_ev: 0.5,
            ..Trim::default()
        },
        grade: GradeParams {
            contrast: 30.0,
            saturation: -30.0,
            ..GradeParams::default()
        },
        ..NodeParams::default()
    }
}

fn effect_with(m: NodeMasks) -> NodeParams {
    NodeParams {
        masks: m,
        ..effect()
    }
}

fn node(id: u32, picture: PictureSource, key: Option<u32>) -> PlanNode {
    PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(picture),
        key,
    }
}

/// A graph with its parameters: the nodes' wiring, the Output's source and the stop.
#[derive(Clone)]
struct Scenario {
    nodes: Vec<PlanNode>,
    params: BTreeMap<u32, NodeParams>,
    output: u32,
}

impl Scenario {
    /// Main at its defaults and node 2 after it with these parameters.
    fn after_main(p: NodeParams) -> Self {
        Self {
            nodes: vec![
                node(MAIN_NODE, PictureSource::Input, None),
                node(2, PictureSource::Node(MAIN_NODE), None),
            ],
            params: [(2, p)].into(),
            output: 2,
        }
    }

    /// Main at its defaults and node 2 reading the Input itself (the luma of C9 and of a Brightness range is
    /// taken from the linear values C7 made).
    fn on_input(p: NodeParams) -> Self {
        let mut s = Self::after_main(p);
        s.nodes[1].picture = Some(PictureSource::Input);
        s
    }

    fn chain(&self, stop: PlanStop, look: Option<LookSetting>) -> ColorChain {
        let plan = plan::compile(
            &PlanGraph {
                nodes: self.nodes.clone(),
                output: PictureSource::Node(self.output),
            },
            stop,
        )
        .expect("plan");
        let mut p: ChainParams = chains::base(OutputTarget::Srgb);
        p.look = look;
        // Main's grade, warp and curves are the chain's own fields; its trim, bypass and masks are the node set's.
        let main = self.params.get(&MAIN_NODE).cloned().unwrap_or_default();
        p.grade = main.grade;
        p.warp = main.warp;
        p.curves = main.curves;
        p.nodes = NodeSet {
            plan,
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

// ---- rendering --------------------------------------------------------------------------------------------

/// The device's pictures of a chain over a window of a grid: the sampler writes the colour and, when the chain's
/// shapes read them, the places in the photo; the colour pass colours it. Planar f32 of the stage (three planes
/// and the letterbox mask).
#[allow(clippy::too_many_arguments)] // a fixture: every piece of the job
fn gpu_stage(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    off: (usize, usize),
    (w, h): (usize, usize),
    stop: Stop,
    node_stop: Option<NodeStop>,
) -> Vec<f32> {
    let (chain, img, grid) = (chain.clone(), img.clone(), *grid);
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "masks source", &img)?;
            let mut passes = PassList::new(ctx, "masks");
            let positions = chain.uses_photo_position();
            let sampled = encode_sample_at(&mut passes, &src, (0, 0), &grid, off, w, h, positions)?;
            let mut prepared = PreparedChain::new(ctx, &chain)?;
            if positions {
                prepared = prepared.with_photo_frame(grid.photo_frame());
            }
            if let Some(s) = node_stop {
                prepared = prepared.with_node_stop(s)?;
            }
            let dst = GpuImage::alloc(
                ctx,
                "masks out",
                w,
                h,
                sampled.planes.min(4),
                ym_gpu::cpu::Placement::whole(w, h),
            )?;
            prepared.encode_stage(&mut passes, &sampled, Rect::whole(w, h), stop, &dst, (0, 0))?;
            let read = dst.encode_read(&mut passes)?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            dst.take(&done, read)
        })
        .expect("GPU render")
}

/// The CPU reference over the same window: the sampled camera pixels (the CPU port of the sampler) and each
/// pixel's place in the photo frame.
struct Reference {
    cam: Vec<[f64; 3]>,
    at: Vec<PixelAt<'static>>,
    /// The pixels outside the photo (the letterbox): the device does not colour them.
    outside: Vec<bool>,
    n: usize,
}

fn reference(
    img: &RgbImage,
    grid: &SamplerGrid,
    off: (usize, usize),
    (w, h): (usize, usize),
) -> Reference {
    let sampled = sample_to_camera(img, (0, 0), grid, off, w, h);
    let n = w * h;
    let map = GridPositions::new(grid, off);
    let frame = grid.photo_frame();
    let mut cam = Vec::with_capacity(n);
    let mut at = Vec::with_capacity(n);
    for i in 0..n {
        let p = sampled.img.pixel(i % w, i / w);
        cam.push([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
        at.push(Some(PhotoSpot::new(map.position(i % w, i / w), frame)));
    }
    let outside = (0..n)
        .map(|i| sampled.outside.as_ref().is_some_and(|o| o[i] != 0))
        .collect();
    Reference {
        cam,
        at,
        outside,
        n,
    }
}

/// The largest differences of the device from the CPU over a window: the blended picture (C10), the encoded
/// picture and the mask. Pixels outside the photo are not compared.
#[derive(Debug, Clone, Copy, Default)]
struct Worst {
    c10: f64,
    c13: f64,
    mask: f64,
    /// The largest change a node made, for the test to be sure it is not vacuous (LogC4).
    effect: f64,
    /// How many pixels the mask changed by more than 1e-3 (the blend is not all 0 or all 1).
    partial: usize,
    /// The pixel of the largest C10 difference.
    c10_at: Option<WorstPixel>,
}

/// A pixel of the largest C10 difference: its index, the camera triple, the device's and the CPU's C10 and
/// the CPU's mask.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // read through `Debug`, in the message of a failed gate
struct WorstPixel {
    index: usize,
    camera: [f64; 3],
    device: [f32; 3],
    cpu: [f64; 3],
    mask: f64,
}

impl Worst {
    fn max(self, o: Self) -> Self {
        Self {
            c10: self.c10.max(o.c10),
            c13: self.c13.max(o.c13),
            mask: self.mask.max(o.mask),
            effect: self.effect.max(o.effect),
            partial: self.partial + o.partial,
            c10_at: if o.c10 > self.c10 {
                o.c10_at
            } else {
                self.c10_at
            },
        }
    }

    /// One line of the numbers, for the log (`docs/23` section 15 quotes them).
    fn report(&self, what: &str) {
        eprintln!(
            "{what}: worst |GPU - CPU| C10 {:.2e}, C13 {:.2e}, m {:.2e}",
            self.c10, self.c13, self.mask
        );
    }

    fn assert_within_the_gates(&self, what: &str) {
        assert!(
            self.c10 <= LOGC4_GATE,
            "{what}: C10 differs by {:.3e} (gate {LOGC4_GATE:.0e}); at {:?}",
            self.c10,
            self.c10_at
        );
        assert!(
            self.c13 <= CODE10,
            "{what}: C13 differs by {:.3e} (gate {CODE10:.1e})",
            self.c13
        );
        assert!(
            self.mask <= MASK_GATE,
            "{what}: m differs by {:.3e} (gate {MASK_GATE:.0e})",
            self.mask
        );
    }
}

/// Compares the device with the CPU for a scenario over a window of a grid.
#[allow(clippy::too_many_arguments)] // a fixture: every piece of the comparison
fn parity(
    dev: &TestGpu,
    scenario: &Scenario,
    mask_of: u32,
    look: Option<LookSetting>,
    img: &RgbImage,
    grid: &SamplerGrid,
    off: (usize, usize),
    size: (usize, usize),
) -> Worst {
    let chain = scenario.chain(PlanStop::Output, look);
    let mask_chain = scenario.chain(PlanStop::Mask(mask_of), None);
    let plain = {
        let mut s = scenario.clone();
        for p in s.params.values_mut() {
            p.masks = NodeMasks::default();
        }
        s.nodes.iter_mut().for_each(|n| n.key = None);
        s.chain(PlanStop::Output, None)
    };
    let r = reference(img, grid, off, size);
    let g10 = gpu_stage(dev, &chain, img, grid, off, size, Stop::C10, None);
    let g13 = gpu_stage(dev, &chain, img, grid, off, size, Stop::C13, None);
    let gm = gpu_stage(
        dev,
        &chain,
        img,
        grid,
        off,
        size,
        Stop::C13,
        Some(NodeStop::Mask(mask_of)),
    );
    let mut w = Worst::default();
    let n = r.n;
    for i in 0..n {
        if r.outside[i] {
            continue;
        }
        let t = chain.trace_at(r.cam[i], r.at[i]);
        let m = mask_chain.apply_logc4_pixel_at(r.cam[i], r.at[i]).0[0];
        let unmasked = plain.apply_logc4_pixel_at(r.cam[i], r.at[i]).0;
        for ch in 0..3 {
            let d10 = (f64::from(g10[ch * n + i]) - t.c10_logc4.0[ch]).abs();
            if d10 > w.c10 {
                w.c10_at = Some(WorstPixel {
                    index: i,
                    camera: r.cam[i],
                    device: [g10[i], g10[n + i], g10[2 * n + i]],
                    cpu: t.c10_logc4.0,
                    mask: m,
                });
            }
            w.c10 = w.c10.max(d10);
            w.c13 = w
                .c13
                .max((f64::from(g13[ch * n + i]) - t.c13_encoded[ch]).abs());
            w.mask = w.mask.max((f64::from(gm[ch * n + i]) - m).abs());
            w.effect = w.effect.max((unmasked[ch] - t.c8_logc4.0[ch]).abs());
        }
        w.partial += usize::from(m > 1e-3 && m < 1.0 - 1e-3);
    }
    w
}

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

// ---- parity by kind --------------------------------------------------------------------------------------

#[test]
fn every_kind_matches_the_cpu_over_every_geometry() {
    let buffer = (97usize, 71usize);
    let img = scene(buffer.0, buffer.1, 0x5EED);
    let kinds: Vec<(&str, Vec<MaskRow>)> = vec![
        ("Linear", vec![linear([0.45, 0.5], 25.0, 0.35)]),
        ("Linear, feather 0", vec![linear([0.5, 0.4], -70.0, 0.0)]),
        ("Radial", vec![radial([0.5, 0.45], [0.35, 0.2], 30.0, 0.4)]),
        (
            "Radial, feather 0",
            vec![radial([0.4, 0.55], [0.3, 0.15], -50.0, 0.0)],
        ),
        ("Colour range", vec![colour([0.0, 0.0], 0.04, 0.5)]),
        (
            "Colour range, off centre",
            vec![colour([0.06, -0.03], 0.05, 0.2)],
        ),
        ("Brightness range", vec![brightness(0.3, 0.7, 0.4)]),
        (
            "Brightness range, softness 0",
            vec![brightness(0.25, 0.6, 0.0)],
        ),
    ];
    for dev in f32_devices() {
        let mut per_kind: BTreeMap<&str, Worst> = BTreeMap::new();
        for (what, rows) in &kinds {
            for (gname, grid, w, h) in grids(buffer) {
                for scenario in [
                    Scenario::after_main(effect_with(masks(rows.clone()))),
                    Scenario::on_input(effect_with(masks(rows.clone()))),
                ] {
                    let worst =
                        parity(dev, &scenario, 2, logc4_look(), &img, &grid, (0, 0), (w, h));
                    worst.assert_within_the_gates(&format!("{what} over {gname}"));
                    assert!(
                        worst.effect <= 0.5,
                        "{what}: the node moves {}",
                        worst.effect
                    );
                    let e = per_kind.entry(what).or_default();
                    *e = e.max(worst);
                }
            }
        }
        for (what, w) in &per_kind {
            eprintln!(
                "{:?} {what}: worst |GPU - CPU| over {} grids x 2 readings: C10 {:.2e} (gate {LOGC4_GATE:.0e}), C13 {:.2e} (gate {CODE10:.1e}), m {:.2e} (gate {MASK_GATE:.0e}); the node moves the picture by up to {:.3} LogC4, {} pixels have 0 < m < 1",
                dev.mode,
                grids(buffer).len(),
                w.c10,
                w.c13,
                w.mask,
                w.effect,
                w.partial
            );
            assert!(w.partial > 20, "{what}: the mask is all 0 or all 1");
        }
    }
}

#[test]
fn inverted_subtracted_and_combined_rows_match_the_cpu() {
    let buffer = (89usize, 61usize);
    let img = scene(buffer.0, buffer.1, 0xC0B1);
    let rad = || radial([0.5, 0.5], [0.3, 0.22], 15.0, 0.4);
    let lin = || linear([0.5, 0.55], 40.0, 0.3);
    let col = || colour([0.0, 0.0], 0.05, 0.6);
    let bri = || brightness(0.2, 0.75, 0.3);
    let cases: Vec<(&str, NodeMasks)> = vec![
        ("an inverted Radial", masks(vec![invert(rad())])),
        (
            "an inverted Linear at 60 %",
            masks(vec![opacity(invert(lin()), 0.6)]),
        ),
        ("an inverted Colour range", masks(vec![invert(col())])),
        ("an inverted Brightness range", masks(vec![invert(bri())])),
        ("a lone subtracted Radial", masks(vec![subtract(rad())])),
        (
            "a lone subtracted Colour range",
            masks(vec![subtract(col())]),
        ),
        ("Radial plus Linear (screen)", masks(vec![rad(), lin()])),
        ("Radial minus Linear", masks(vec![rad(), subtract(lin())])),
        (
            "Radial plus Linear minus a half Radial",
            masks(vec![
                rad(),
                lin(),
                subtract(opacity(radial([0.3, 0.4], [0.2, 0.2], 0.0, 0.5), 0.5)),
            ]),
        ),
        ("Radial times Colour range", masks(vec![rad(), col()])),
        (
            "Radial times two ranges, one subtracted",
            masks(vec![rad(), col(), subtract(opacity(bri(), 0.7))]),
        ),
        (
            "ranges alone, added",
            masks(vec![opacity(col(), 0.5), opacity(bri(), 0.5)]),
        ),
        (
            "the node's Invert on a combination",
            NodeMasks {
                rows: vec![rad(), subtract(lin()), col()],
                invert: true,
            },
        ),
        (
            "the node's Invert alone",
            NodeMasks {
                rows: vec![],
                invert: true,
            },
        ),
        ("no rows", NodeMasks::default()),
        (
            "a raster kind (selects nothing)",
            masks(vec![MaskRow::new(MaskKind::Raster { plane: 0 })]),
        ),
        (
            "an inverted raster kind",
            masks(vec![invert(MaskRow::new(MaskKind::Raster { plane: 0 }))]),
        ),
        (
            "nine rows of every kind",
            masks(vec![
                rad(),
                subtract(opacity(lin(), 0.4)),
                opacity(col(), 0.8),
                subtract(opacity(bri(), 0.3)),
                opacity(radial([0.7, 0.3], [0.15, 0.1], -20.0, 0.6), 0.9),
                invert(opacity(linear([0.3, 0.6], -50.0, 0.2), 0.5)),
                MaskRow::new(MaskKind::Raster { plane: 0 }),
                colour([0.03, 0.02], 0.08, 0.4),
                brightness(0.1, 0.5, 0.6),
            ]),
        ),
    ];
    let (_, grid, w, h) = grids(buffer).swap_remove(2);
    for dev in f32_devices() {
        let mut all = Worst::default();
        for (what, m) in &cases {
            for scenario in [
                Scenario::after_main(effect_with(m.clone())),
                Scenario::on_input(effect_with(m.clone())),
            ] {
                let worst = parity(dev, &scenario, 2, None, &img, &grid, (0, 0), (w, h));
                worst.assert_within_the_gates(what);
                all = all.max(worst);
            }
        }
        all.report(&format!(
            "{:?} {} combinations of rows",
            dev.mode,
            cases.len()
        ));
    }
}

#[test]
fn main_may_have_masks_too_and_a_bypassed_main_still_makes_its_key() {
    let buffer = (79usize, 57usize);
    let img = scene(buffer.0, buffer.1, 0x3A1);
    let (_, grid, w, h) = grids(buffer).swap_remove(2);
    let masked_main = NodeParams {
        masks: masks(vec![
            radial([0.5, 0.5], [0.3, 0.2], 20.0, 0.4),
            subtract(linear([0.6, 0.5], 30.0, 0.2)),
        ]),
        ..effect()
    };
    for dev in f32_devices() {
        // Main with a grade and masks, node 2 after it.
        let mut s = Scenario::after_main(effect_with(masks(vec![colour([0.0, 0.0], 0.06, 0.5)])));
        s.params.insert(MAIN_NODE, masked_main.clone());
        let w1 = parity(dev, &s, 1, logc4_look(), &img, &grid, (0, 0), (w, h));
        w1.assert_within_the_gates("Main with masks");
        w1.report(&format!("{:?} Main with masks", dev.mode));
        assert!(w1.partial > 20, "Main's mask is all 0 or all 1");
        parity(dev, &s, 2, None, &img, &grid, (0, 0), (w, h))
            .assert_within_the_gates("Main with masks, node 2's mask");
        // Main bypassed: its picture is the Input's, its key output is still its mask.
        let mut bypassed = masked_main.clone();
        bypassed.bypass = true;
        let mut s = Scenario {
            nodes: vec![
                node(MAIN_NODE, PictureSource::Input, None),
                node(2, PictureSource::Input, Some(MAIN_NODE)),
            ],
            params: [(2, effect())].into(),
            output: 2,
        };
        s.params.insert(MAIN_NODE, bypassed);
        parity(dev, &s, 2, logc4_look(), &img, &grid, (0, 0), (w, h))
            .assert_within_the_gates("a bypassed Main's key");
    }
}

#[test]
fn a_node_with_many_rows_has_no_cap() {
    // The settings allow 64 rows a node and the kernel loops over any number.
    let buffer = (73usize, 51usize);
    let img = scene(buffer.0, buffer.1, 0x64);
    let mut rows = Vec::new();
    for i in 0..64u32 {
        let t = f64::from(i);
        let row = match i % 4 {
            0 => radial(
                [
                    0.1 + 0.8 * ((t * 0.37) % 1.0),
                    0.1 + 0.8 * ((t * 0.61) % 1.0),
                ],
                [0.08 + 0.02 * (t % 5.0), 0.06],
                10.0 * t,
                0.5,
            ),
            1 => linear([0.2 + 0.6 * ((t * 0.23) % 1.0), 0.5], 17.0 * t, 0.3),
            2 => colour([0.02 * (t % 7.0) - 0.06, 0.0], 0.06, 0.5),
            _ => brightness(0.1 + 0.01 * (t % 9.0), 0.6 + 0.02 * (t % 11.0), 0.5),
        };
        rows.push(MaskRow {
            subtract: i % 5 == 4,
            opacity: 0.3 + 0.05 * f64::from(i % 8),
            ..row
        });
    }
    let (_, grid, w, h) = grids(buffer).swap_remove(1);
    for dev in f32_devices() {
        let scenario = Scenario::after_main(effect_with(masks(rows.clone())));
        let worst = parity(dev, &scenario, 2, None, &img, &grid, (0, 0), (w, h));
        worst.assert_within_the_gates("64 rows");
        worst.report(&format!("{:?} one node of 64 rows", dev.mode));
    }
}

// ---- key links ----------------------------------------------------------------------------------------------

#[test]
fn a_key_link_carries_the_final_mask_and_a_dead_branch_feeds_a_live_node() {
    let buffer = (83usize, 59usize);
    let img = scene(buffer.0, buffer.1, 0x4B);
    let (_, grid, w, h) = grids(buffer).swap_remove(2);
    // Node 2 makes a mask only (nothing reads its picture); node 3 is keyed by it and has a Radial subtracted
    // from the key and a Colour range of its own. Node 2's Invert and a Bypass are part of "final".
    let key_masks = |bypass: bool, invert_node: bool| NodeParams {
        bypass,
        masks: NodeMasks {
            rows: vec![
                linear([0.5, 0.5], 30.0, 0.4),
                opacity(colour([0.0, 0.0], 0.06, 0.5), 0.9),
            ],
            invert: invert_node,
        },
        ..effect()
    };
    let keyed = |bypass: bool, invert_node: bool, own: Vec<MaskRow>| Scenario {
        nodes: vec![
            node(MAIN_NODE, PictureSource::Input, None),
            node(2, PictureSource::Input, None),
            node(3, PictureSource::Node(MAIN_NODE), Some(2)),
        ],
        params: [
            (2, key_masks(bypass, invert_node)),
            (3, effect_with(masks(own))),
        ]
        .into(),
        output: 3,
    };
    for dev in f32_devices() {
        for (what, scenario) in [
            ("the key alone", keyed(false, false, vec![])),
            ("the key, node 2 bypassed", keyed(true, false, vec![])),
            ("the key, node 2 inverted", keyed(false, true, vec![])),
            (
                "the key minus a Radial",
                keyed(
                    false,
                    false,
                    vec![subtract(radial([0.5, 0.5], [0.2, 0.15], 0.0, 0.5))],
                ),
            ),
            (
                "the key plus a Linear (screen)",
                keyed(
                    false,
                    false,
                    vec![opacity(linear([0.3, 0.5], 80.0, 0.3), 0.5)],
                ),
            ),
            (
                "the key times a Brightness range",
                keyed(false, false, vec![brightness(0.2, 0.7, 0.3)]),
            ),
            ("the key with the node's Invert", {
                let mut s = keyed(false, false, vec![]);
                s.params.get_mut(&3).expect("node 3").masks.invert = true;
                s
            }),
        ] {
            let plan = scenario
                .chain(PlanStop::Output, None)
                .params()
                .nodes
                .plan
                .node_ids();
            assert_eq!(plan, vec![1, 2, 3], "{what}: node 2 is live for its key");
            let worst = parity(dev, &scenario, 3, logc4_look(), &img, &grid, (0, 0), (w, h));
            worst.assert_within_the_gates(what);
            // Node 2's mask on its own is what the stop at node 2 shows.
            parity(dev, &scenario, 2, None, &img, &grid, (0, 0), (w, h))
                .assert_within_the_gates(&format!("{what}, node 2's mask"));
            eprintln!(
                "{:?} {what}: C10 {:.2e}, C13 {:.2e}, m {:.2e}, {} pixels with 0 < m < 1",
                dev.mode, worst.c10, worst.c13, worst.mask, worst.partial
            );
        }
    }
}

#[test]
fn keys_cross_the_split_between_passes_on_the_device() {
    // Main fans out to nine nodes (2..=10), each read by one of 20..=28 whose keys are chained: nine pictures alive
    // at once need two passes with buffers between them, the keys crossing in buffers too.
    let buffer = (67usize, 49usize);
    let img = scene(buffer.0, buffer.1, 0x9A55);
    let (_, grid, w, h) = grids(buffer).swap_remove(1);
    let fan = 9u32;
    let mut nodes = vec![node(MAIN_NODE, PictureSource::Input, None)];
    let mut params = BTreeMap::new();
    for i in 0..fan {
        nodes.push(PlanNode {
            pos: [1.0 + f64::from(i) * 0.001, 0.0],
            ..node(2 + i, PictureSource::Node(MAIN_NODE), None)
        });
        params.insert(
            2 + i,
            NodeParams {
                trim: Trim {
                    exposure_ev: 0.05 * f64::from(i + 1),
                    ..Trim::default()
                },
                ..NodeParams::default()
            },
        );
    }
    for i in 0..fan {
        nodes.push(PlanNode {
            pos: [2.0 + f64::from(i) * 0.001, 0.0],
            ..node(
                20 + i,
                PictureSource::Node(2 + i),
                (i > 0).then(|| 20 + i - 1),
            )
        });
        // Some of the nodes are bypassed: their pictures are their inputs, their keys are still their masks.
        params.insert(
            20 + i,
            NodeParams {
                bypass: i % 2 == 1 && i < fan - 1,
                ..effect_with(masks(vec![radial(
                    [0.15 + 0.08 * f64::from(i), 0.5],
                    [0.25, 0.2],
                    10.0 * f64::from(i),
                    0.5,
                )]))
            },
        );
    }
    let scenario = Scenario {
        nodes,
        params,
        output: 20 + fan - 1,
    };
    let chain = scenario.chain(PlanStop::Output, None);
    assert!(chain.params().nodes.plan.passes.len() >= 2, "two passes");
    for dev in f32_devices() {
        let worst = parity(
            dev,
            &scenario,
            20 + fan - 1,
            None,
            &img,
            &grid,
            (0, 0),
            (w, h),
        );
        worst.assert_within_the_gates("keys across passes");
        worst.report(&format!("{:?} nine chained keys over two passes", dev.mode));
        // And a stop in the middle of the second pass.
        parity(dev, &scenario, 24, None, &img, &grid, (0, 0), (w, h))
            .assert_within_the_gates("the mask of a node in the second pass");
    }
}

#[test]
fn seventeen_keys_alive_at_once_cross_the_split_in_buffers() {
    // Seventeen nodes read Main and make masks only (every other one bypassed: their pictures are their inputs,
    // their keys are still their masks); a chain of seventeen takes one key each. Sixteen key slots fit a pass, so
    // the seventeenth key crosses to the second pass in a plane of the pass buffers.
    let buffer = (61usize, 45usize);
    let img = scene(buffer.0, buffer.1, 0x17C);
    let (_, grid, w, h) = grids(buffer).swap_remove(2);
    let n = 17u32;
    let mut nodes = vec![node(MAIN_NODE, PictureSource::Input, None)];
    let mut params = BTreeMap::new();
    for i in 0..n {
        nodes.push(PlanNode {
            pos: [1.0, f64::from(i)],
            ..node(2 + i, PictureSource::Node(MAIN_NODE), None)
        });
        let t = f64::from(i);
        let row = match i % 3 {
            0 => radial([0.1 + 0.05 * t, 0.5], [0.3, 0.2], 7.0 * t, 0.5),
            1 => linear([0.5, 0.2 + 0.04 * t], 11.0 * t, 0.3),
            _ => colour([0.01 * (t % 5.0) - 0.02, 0.0], 0.08, 0.5),
        };
        params.insert(
            2 + i,
            NodeParams {
                bypass: i % 2 == 0,
                ..effect_with(masks(vec![row]))
            },
        );
    }
    for i in 0..n {
        let prev = if i == 0 {
            PictureSource::Node(MAIN_NODE)
        } else {
            PictureSource::Node(100 + i - 1)
        };
        nodes.push(PlanNode {
            pos: [2.0, f64::from(i)],
            ..node(100 + i, prev, Some(2 + i))
        });
        params.insert(
            100 + i,
            NodeParams {
                trim: Trim {
                    exposure_ev: 0.06,
                    ..Trim::default()
                },
                ..NodeParams::default()
            },
        );
    }
    let last = 100 + n - 1;
    let scenario = Scenario {
        nodes,
        params,
        output: last,
    };
    let plan = scenario
        .chain(PlanStop::Output, None)
        .params()
        .nodes
        .plan
        .clone();
    assert!(plan.passes.len() >= 2, "{} passes", plan.passes.len());
    assert!(
        plan.buffers.contains(&plan::BufferKind::Key),
        "a key crosses the split"
    );
    for dev in f32_devices() {
        let worst = parity(dev, &scenario, last, None, &img, &grid, (0, 0), (w, h));
        worst.assert_within_the_gates("seventeen keys");
        worst.report(&format!(
            "{:?} seventeen keys, one in a plane of the pass buffers",
            dev.mode
        ));
        // The mask of a node whose key crossed.
        parity(dev, &scenario, 2 + n - 1, None, &img, &grid, (0, 0), (w, h))
            .assert_within_the_gates("the seventeenth mask node");
        parity(
            dev,
            &scenario,
            100 + n - 2,
            None,
            &img,
            &grid,
            (0, 0),
            (w, h),
        )
        .assert_within_the_gates("the second last of the chain");
    }
}

#[test]
fn nodes_without_masks_or_keys_keep_their_own_bits() {
    // A node with no rows, no Invert and no key takes no blend: the same bits as a chain without masks at all.
    let buffer = (61usize, 43usize);
    let img = scene(buffer.0, buffer.1, 0xB17);
    let (_, grid, w, h) = grids(buffer).swap_remove(0);
    for dev in f32_devices() {
        let plain = Scenario::after_main(effect());
        let empty = Scenario::after_main(effect_with(NodeMasks::default()));
        let a = gpu_stage(
            dev,
            &plain.chain(PlanStop::Output, logc4_look()),
            &img,
            &grid,
            (0, 0),
            (w, h),
            Stop::C10,
            None,
        );
        let b = gpu_stage(
            dev,
            &empty.chain(PlanStop::Output, logc4_look()),
            &img,
            &grid,
            (0, 0),
            (w, h),
            Stop::C10,
            None,
        );
        assert_bit_equal("a node with no mask rows", &a, &b);
        // A node whose rows are all selecting (a Radial covering the photo, m exactly 1) gives the node's bits.
        let all = Scenario::after_main(effect_with(masks(vec![radial(
            [0.5, 0.5],
            [1.0, 1.0],
            0.0,
            0.0,
        )])));
        let c = gpu_stage(
            dev,
            &all.chain(PlanStop::Output, logc4_look()),
            &img,
            &grid,
            (0, 0),
            (w, h),
            Stop::C10,
            None,
        );
        let n = w * h;
        let mut equal = 0usize;
        for i in 0..n {
            equal += usize::from((0..3).all(|k| a[k * n + i] == c[k * n + i]));
        }
        assert!(
            equal * 2 > n,
            "a mask of 1 gives the node's own bits (the pixels where it is exactly 1): {equal} of {n}"
        );
    }
}

// ---- tiles, determinism ------------------------------------------------------------------------------------

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let buffer = (101usize, 75usize);
    let img = scene(buffer.0, buffer.1, 0x71E5);
    let scenario = Scenario::after_main(effect_with(masks(vec![
        radial([0.5, 0.5], [0.35, 0.25], 25.0, 0.3),
        subtract(linear([0.6, 0.5], -35.0, 0.2)),
        colour([0.0, 0.0], 0.07, 0.5),
        brightness(0.15, 0.85, 0.4),
    ])));
    let chain = scenario.chain(PlanStop::Output, logc4_look());
    for dev in f32_devices() {
        for (name, grid, w, h) in grids(buffer) {
            for stop in [Stop::C10, Stop::C13] {
                let whole = gpu_stage(dev, &chain, &img, &grid, (0, 0), (w, h), stop, None);
                let n = w * h;
                for tile in [17usize, 40] {
                    let mut put = vec![0.0f32; 4 * n];
                    let mut ty = 0;
                    while ty < h {
                        let th = tile.min(h - ty);
                        let mut tx = 0;
                        while tx < w {
                            let tw = tile.min(w - tx);
                            let part =
                                gpu_stage(dev, &chain, &img, &grid, (tx, ty), (tw, th), stop, None);
                            let m = tw * th;
                            for c in 0..4 {
                                for y in 0..th {
                                    for x in 0..tw {
                                        put[c * n + (ty + y) * w + tx + x] =
                                            part[c * m + y * tw + x];
                                    }
                                }
                            }
                            tx += tw;
                        }
                        ty += th;
                    }
                    assert_bit_equal(&format!("{name}, tiles of {tile}, {stop:?}"), &whole, &put);
                }
            }
        }
    }
}

#[test]
fn the_position_planes_of_a_tile_are_the_wholes_and_the_cpus() {
    // The sampler's places in the photo: tile and whole frame equal bit for bit, the host's planes for a picture it
    // sampled itself equal to within f32, and the device within 1e-5 of the photo of the CPU's f64.
    let buffer = (101usize, 75usize);
    let img = scene(buffer.0, buffer.1, 0x9051);
    for dev in f32_devices() {
        let mut worst = 0.0f64;
        for (name, grid, w, h) in grids(buffer) {
            let read = |off: (usize, usize), size: (usize, usize)| {
                let (grid, img) = (grid, img.clone());
                dev.gpu
                    .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                        let src = GpuImage::upload_rgb(ctx, "src", &img)?;
                        let mut passes = PassList::new(ctx, "positions");
                        let out = encode_sample_at(
                            &mut passes,
                            &src,
                            (0, 0),
                            &grid,
                            off,
                            size.0,
                            size.1,
                            true,
                        )?;
                        let read = out.encode_read(&mut passes)?;
                        let _src = passes.keep(src.buf);
                        let done = passes.submit()?.finish()?;
                        out.take(&done, read)
                    })
                    .expect("GPU sample")
            };
            let whole = read((0, 0), (w, h));
            let n = w * h;
            assert_eq!(whole.len(), 6 * n);
            // A tile's planes.
            let (tx, ty, tw, th) = (w / 3, h / 4, w / 2, h / 2);
            let tile = read((tx, ty), (tw, th));
            let m = tw * th;
            for p in [4usize, 5] {
                for y in 0..th {
                    for x in 0..tw {
                        assert_eq!(
                            tile[p * m + y * tw + x].to_bits(),
                            whole[p * n + (ty + y) * w + tx + x].to_bits(),
                            "{name}: plane {p} at ({x}, {y}) of the tile"
                        );
                    }
                }
            }
            // The CPU's positions.
            let cpu = photo_positions(&grid, (0, 0), w, h);
            let sampled = sample_to_camera(&img, (0, 0), &grid, (0, 0), w, h);
            for i in 0..n {
                if sampled.outside.as_ref().is_some_and(|o| o[i] != 0) {
                    continue;
                }
                for p in 0..2 {
                    let d = (f64::from(whole[(4 + p) * n + i]) - f64::from(cpu[p * n + i])).abs();
                    worst = worst.max(d);
                    assert!(
                        d < 1e-5,
                        "{name}: position {p} of pixel {i} differs by {d:e}"
                    );
                }
            }
        }
        eprintln!(
            "{:?} the device's photo-frame positions against the CPU's f64: worst {worst:.2e} of the photo (about {:.1e} px of a 6000 px frame)",
            dev.mode,
            worst * 6000.0
        );
    }
}

#[test]
fn a_repeat_render_is_identical_and_poison_changes_nothing() {
    let buffer = (59usize, 41usize);
    let img = scene(buffer.0, buffer.1, 0xD37);
    let scenario = Scenario {
        nodes: vec![
            node(MAIN_NODE, PictureSource::Input, None),
            node(2, PictureSource::Input, None),
            node(3, PictureSource::Node(MAIN_NODE), Some(2)),
        ],
        params: [
            (
                2,
                effect_with(masks(vec![
                    radial([0.5, 0.5], [0.3, 0.2], 10.0, 0.3),
                    colour([0.0, 0.0], 0.06, 0.5),
                ])),
            ),
            (
                3,
                effect_with(masks(vec![subtract(linear([0.5, 0.5], 20.0, 0.2))])),
            ),
        ]
        .into(),
        output: 3,
    };
    let chain = scenario.chain(PlanStop::Output, logc4_look());
    let (_, grid, w, h) = grids(buffer).swap_remove(2);
    for dev in f32_devices() {
        let a = gpu_stage(dev, &chain, &img, &grid, (0, 0), (w, h), Stop::C13, None);
        let b = gpu_stage(dev, &chain, &img, &grid, (0, 0), (w, h), Stop::C13, None);
        assert_bit_equal("a repeat render", &a, &b);
    }
    // The same job on a poisoned device and on a plain one.
    let (c, g) = (chain.clone(), grid);
    let im = img.clone();
    let run = move |gpu: &ym_gpu::device::Gpu| {
        let (chain, grid, img) = (c.clone(), g, im.clone());
        gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "src", &img)?;
            let mut passes = PassList::new(ctx, "poison");
            let sampled = encode_sample_at(&mut passes, &src, (0, 0), &grid, (0, 0), w, h, true)?;
            let prepared = PreparedChain::new(ctx, &chain)?.with_photo_frame(grid.photo_frame());
            let dst = GpuImage::alloc(ctx, "out", w, h, 4, ym_gpu::cpu::Placement::whole(w, h))?;
            prepared.encode_stage(
                &mut passes,
                &sampled,
                Rect::whole(w, h),
                Stop::C13,
                &dst,
                (0, 0),
            )?;
            let read = dst.encode_read(&mut passes)?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            dst.take(&done, read)
        })
        .expect("GPU render")
    };
    let _ = assert_poison_neutral("masks and a key", run);
}

// ---- resolution independence and the photo frame ------------------------------------------------------------

/// The mask of rows over a `w x h` render of a `bw x bh` photo, `k` photo pixels to an output pixel, on the CPU
/// (the render's own grid: the photo scaled by `1/k`).
fn cpu_mask_plane(rows: &[MaskRow], buffer: (usize, usize), k: f64) -> (Vec<f64>, usize, usize) {
    let (bw, bh) = buffer;
    let (w, h) = ((bw as f64 / k) as usize, (bh as f64 / k) as usize);
    let grid = grid_of(buffer, 1, 0, 0.0, k, (0.0, 0.0));
    let prepared = masks(rows.to_vec()).prepare();
    let map = GridPositions::new(&grid, (0, 0));
    let frame = grid.photo_frame();
    let grey = ym_color::logc::LogC4([0.3; 3]);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let spot = Some(PhotoSpot::new(map.position(x, y), frame));
            out.push(prepared.mask(None, grey, 0.3, spot));
        }
    }
    (out, w, h)
}

/// The mask of `rows` at k = 4 against the mask at k = 1 reduced by 4 x 4: the largest and the mean difference.
fn reduced_against_full(rows: &[MaskRow], buffer: (usize, usize)) -> (f64, f64) {
    let (full, fw, fh) = cpu_mask_plane(rows, buffer, 1.0);
    let (small, sw, sh) = cpu_mask_plane(rows, buffer, 4.0);
    assert_eq!((fw / 4, fh / 4), (sw, sh));
    let (mut largest, mut sum) = (0.0f64, 0.0f64);
    for y in 0..sh {
        for x in 0..sw {
            let mut acc = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    acc += full[(4 * y + j) * fw + 4 * x + i];
                }
            }
            let d = (acc / 16.0 - small[y * sw + x]).abs();
            largest = largest.max(d);
            sum += d;
        }
    }
    (largest, sum / (sw * sh) as f64)
}

#[test]
fn a_reduced_view_agrees_with_the_export_reduced() {
    // docs/23 section 11: the same mask at k = 4 and k = 1 (the larger render reduced by the same box) within
    // largest 0.1 and mean 0.005 (the starting numbers; they are not loosened). Feathered shapes, then hard
    // edges (Feather 0), whose ramp is the minimum one: a ramp of MIN_RAMP_PIXELS centred on the edge.
    let buffer = (1200usize, 800usize);
    let cases: Vec<(&str, Vec<MaskRow>)> = vec![
        ("Linear, feather 0.2", vec![linear([0.5, 0.5], 20.0, 0.2)]),
        (
            "Linear, feather 0.05",
            vec![linear([0.4, 0.5], -35.0, 0.05)],
        ),
        (
            "Radial, feather 0.3",
            vec![radial([0.5, 0.5], [0.3, 0.2], 25.0, 0.3)],
        ),
        (
            "Radial, feather 0.08",
            vec![radial([0.45, 0.55], [0.25, 0.15], -40.0, 0.08)],
        ),
        (
            "Radial minus Linear, feathered",
            vec![
                radial([0.5, 0.5], [0.35, 0.3], 0.0, 0.2),
                subtract(linear([0.5, 0.5], 60.0, 0.1)),
            ],
        ),
        (
            "two Radials, a small one",
            vec![
                radial([0.3, 0.3], [0.04, 0.03], 0.0, 0.5),
                radial([0.7, 0.6], [0.2, 0.2], 0.0, 0.5),
            ],
        ),
        ("Linear, feather 0", vec![linear([0.4137, 0.5], -35.0, 0.0)]),
        (
            "Linear, feather 0, 20 degrees",
            vec![linear([0.5, 0.5], 20.0, 0.0)],
        ),
        (
            "Radial, feather 0",
            vec![radial([0.45, 0.55], [0.25, 0.15], -40.0, 0.0)],
        ),
        (
            "Radial, feather 0, a circle",
            vec![radial([0.5, 0.5], [0.3, 0.3], 0.0, 0.0)],
        ),
        (
            "Radial minus Linear, both feather 0",
            vec![
                radial([0.5, 0.5], [0.35, 0.3], 0.0, 0.0),
                subtract(linear([0.5, 0.5], 60.0, 0.0)),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (what, rows) in cases {
        let (largest, mean) = reduced_against_full(&rows, buffer);
        eprintln!(
            "{what}: the mask at k = 4 against the mask at k = 1 reduced by 4 x 4: largest {largest:.4}, mean {mean:.5} (starting bounds 0.1 and 0.005)"
        );
        if largest > 0.1 || mean > 0.005 {
            failures.push(format!("{what}: largest {largest:.4}, mean {mean:.5}"));
        }
    }
    assert!(
        failures.is_empty(),
        "preview against export is over its starting bounds (largest 0.1, mean 0.005), not loosened: {failures:?}"
    );
}

#[test]
fn hard_edges_agree_at_two_scales_whatever_the_angle_and_the_place() {
    // The same gate as above over many placements of a hard edge: a Linear at ten angles and three offsets (a
    // 1200 x 800 photo has pixel phases that no one placement covers), and Radials of different shapes, a circle
    // and a sliver among them (the ramp across an ellipse's outline is as wide at the end of its long axis as at
    // the end of its short one).
    let buffer = (1200usize, 800usize);
    let (mut worst_linear, mut worst_radial) = (0.0f64, 0.0f64);
    let mut worst_mean = 0.0f64;
    let mut failures = Vec::new();
    for angle in [0.0, 5.0, 10.0, 20.0, 30.0, 35.0, 45.0, 60.0, 80.0, 90.0] {
        for c in [0.4137, 0.5, 0.5333] {
            let (largest, mean) = reduced_against_full(&[linear([c, 0.5], angle, 0.0)], buffer);
            worst_linear = worst_linear.max(largest);
            worst_mean = worst_mean.max(mean);
            if largest > 0.1 || mean > 0.005 {
                failures.push(format!(
                    "Linear at {angle} degrees through {c}: largest {largest:.4}, mean {mean:.5}"
                ));
            }
        }
    }
    for (c, r, a) in [
        ([0.45, 0.55], [0.25, 0.15], -40.0),
        ([0.5, 0.5], [0.3, 0.2], 0.0),
        ([0.37, 0.41], [0.1, 0.07], 25.0),
        ([0.6, 0.5], [0.04, 0.03], 70.0),
        ([0.5, 0.5], [0.35, 0.3], 12.0),
        ([0.5, 0.5], [0.3, 0.3], 0.0),
        ([0.5, 0.5], [0.45, 0.06], 33.0),
    ] {
        let (largest, mean) = reduced_against_full(&[radial(c, r, a, 0.0)], buffer);
        worst_radial = worst_radial.max(largest);
        worst_mean = worst_mean.max(mean);
        if largest > 0.1 || mean > 0.005 {
            failures.push(format!(
                "Radial {r:?} at {a} degrees: largest {largest:.4}, mean {mean:.5}"
            ));
        }
    }
    eprintln!(
        "hard edges, k = 4 against k = 1 reduced: Linear (30 placements) largest {worst_linear:.4}, Radial (7 placements) largest {worst_radial:.4}, worst mean {worst_mean:.5} (starting bounds 0.1 and 0.005)"
    );
    assert!(
        failures.is_empty(),
        "a hard edge previews differently from how it exports, over the starting bounds (not loosened): {failures:?}"
    );
}

/// A flat picture with a bright dot at `(cx, cy)` of the oriented photo (fractions), sigma `s` of its long edge:
/// `buffer` in the sensor's orientation `exif`.
fn dot_scene(buffer: (usize, usize), exif: u16, c: [f64; 2], sigma: f64) -> RgbImage {
    let (bw, bh) = buffer;
    let (ow, oh) = if (5..=8).contains(&exif) {
        (bh as f64, bw as f64)
    } else {
        (bw as f64, bh as f64)
    };
    let long = ow.max(oh);
    let mut img = RgbImage::new(bw, bh);
    for y in 0..bh {
        for x in 0..bw {
            // The sensor pixel's centre as an oriented fraction.
            let s = [(x as f64 + 0.5) / bw as f64, (y as f64 + 0.5) / bh as f64];
            let (m, t) = SamplerGrid::new(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                (bw as f64, bh as f64),
                (bw as f64, bh as f64),
                exif,
                0,
                true,
            )
            .sensor_to_oriented_map();
            let o = [
                m[0][0] * s[0] + m[0][1] * s[1] + t[0],
                m[1][0] * s[0] + m[1][1] * s[1] + t[1],
            ];
            let d2 = ((o[0] - c[0]) * ow / long).powi(2) + ((o[1] - c[1]) * oh / long).powi(2);
            let v = 0.02 + 0.8 * (-d2 / (2.0 * sigma * sigma)).exp();
            img.set_pixel(x, y, [v as f32, v as f32, v as f32]);
        }
    }
    img
}

#[test]
fn a_radial_stays_on_its_photo_feature_through_every_geometry() {
    // docs/23 section 3.6: a photo cropped, straightened, turned or bent keeps a mask on the same feature. The
    // scene is flat with a dot at the Radial's centre; the centroid of the mask over the render is the centroid of
    // the dot's brightness over the render, to within one output pixel.
    let buffer = (240usize, 160usize);
    let centre = [0.3, 0.62];
    for (exif, turns, angle, scale, t, warp, name) in [
        (1u16, 0u8, 0.0, 1.0, (0.0, 0.0), false, "as shot"),
        (1, 0, 0.0, 1.0, (0.12, 0.08), false, "cropped"),
        (1, 0, 9.0, 1.0, (0.0, 0.0), false, "straightened"),
        (1, 1, 0.0, 1.0, (0.0, 0.0), false, "turned"),
        (
            6,
            0,
            0.0,
            1.0,
            (0.0, 0.0),
            false,
            "the camera's orientation 6",
        ),
        (
            6,
            3,
            -7.0,
            1.3,
            (0.05, 0.0),
            false,
            "orientation 6, turned back, tilted, shrunk",
        ),
        (3, 2, 14.0, 1.25, (0.0, 0.0), false, "turned over, tilted"),
        (1, 0, 0.0, 1.0, (0.0, 0.0), true, "bent"),
        (
            8,
            1,
            5.0,
            1.0,
            (0.0, 0.0),
            true,
            "orientation 8, turned, tilted, bent",
        ),
    ] {
        let img = dot_scene(buffer, exif, centre, 0.03);
        let mut grid = grid_of(buffer, exif, turns, angle, scale, t);
        if warp {
            grid = grid.with_warp(Some(bend(1.0, (grid.canvas.0, grid.canvas.1))));
        }
        let (w, h) = (
            ((grid.canvas.0 / scale) as usize).clamp(8, 300),
            ((grid.canvas.1 / scale) as usize).clamp(8, 300),
        );
        let sampled = sample_to_camera(&img, (0, 0), &grid, (0, 0), w, h);
        let map = GridPositions::new(&grid, (0, 0));
        let frame = grid.photo_frame();
        let radial_row = vec![radial(centre, [0.06, 0.06], 0.0, 0.5)];
        let prepared = masks(radial_row).prepare();
        let grey = ym_color::logc::LogC4([0.3; 3]);
        // Centroids in output pixels: the dot's brightness above the floor, and the mask.
        let (mut dot, mut mk) = ([0.0f64; 3], [0.0f64; 3]);
        for y in 0..h {
            for x in 0..w {
                if sampled.outside.as_ref().is_some_and(|o| o[y * w + x] != 0) {
                    continue;
                }
                let b = f64::from(sampled.img.pixel(x, y)[0]) - 0.02;
                let spot = Some(PhotoSpot::new(map.position(x, y), frame));
                let m = prepared.mask(None, grey, 0.3, spot);
                for (acc, wgt) in [(&mut dot, b), (&mut mk, m)] {
                    acc[0] += wgt;
                    acc[1] += wgt * (x as f64 + 0.5);
                    acc[2] += wgt * (y as f64 + 0.5);
                }
            }
        }
        assert!(
            dot[0] > 1.0 && mk[0] > 1.0,
            "{name}: the dot or the mask is out of view"
        );
        let (dx, dy) = (dot[1] / dot[0], dot[2] / dot[0]);
        let (mx, my) = (mk[1] / mk[0], mk[2] / mk[0]);
        let off = (dx - mx).hypot(dy - my);
        eprintln!(
            "{name}: the dot at ({dx:.2}, {dy:.2}), the mask at ({mx:.2}, {my:.2}) px: {off:.3} px apart"
        );
        assert!(off < 1.0, "{name}: the mask is {off:.2} px off the dot");
    }
}

// ---- cost -------------------------------------------------------------------------------------------------

#[test]
fn four_nodes_of_four_masks_are_timed_on_lavapipe() {
    let buffer = (512usize, 384usize);
    let img = scene(buffer.0, buffer.1, 0x71A3);
    let grid = grid_of(buffer, 1, 0, 0.0, 1.0, (0.0, 0.0));
    let rows = |i: u32| {
        let t = f64::from(i);
        vec![
            radial([0.3 + 0.1 * t, 0.5], [0.3, 0.2], 10.0 * t, 0.4),
            subtract(linear([0.5, 0.5 + 0.05 * t], 20.0 + 15.0 * t, 0.3)),
            colour([0.0, 0.0], 0.06 + 0.01 * t, 0.5),
            brightness(0.1, 0.8, 0.4),
        ]
    };
    let mut nodes = vec![node(MAIN_NODE, PictureSource::Input, None)];
    let mut params = BTreeMap::new();
    for id in 2..=5u32 {
        nodes.push(node(id, PictureSource::Node(id - 1), None));
        params.insert(id, effect_with(masks(rows(id))));
    }
    let masked = Scenario {
        nodes: nodes.clone(),
        params: params.clone(),
        output: 5,
    };
    let plain = Scenario {
        nodes,
        params: params
            .into_iter()
            .map(|(id, p)| {
                (
                    id,
                    NodeParams {
                        masks: NodeMasks::default(),
                        ..p
                    },
                )
            })
            .collect(),
        output: 5,
    };
    for dev in f32_devices() {
        let time = |s: &Scenario| {
            let chain = s.chain(PlanStop::Output, logc4_look());
            let mut best = f64::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                let _ = gpu_stage(dev, &chain, &img, &grid, (0, 0), buffer, Stop::C13, None);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            best
        };
        let (a, b) = (time(&plain), time(&masked));
        eprintln!(
            "{:?}: 4 nodes, 512 x 384 through the sampler and the colour pass on lavapipe: no masks {a:.1} ms, 4 masks a node (Radial, subtracted Linear, Colour range, Brightness range) {b:.1} ms",
            dev.mode
        );
    }
}
