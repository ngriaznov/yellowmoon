//! N-13 (`docs/23-nodes.md` sections 4.3, 5.5 and 11, `docs/algorithms/06-masks.md` section 5): the raster masks'
//! pass on the device against its CPU reference (`ym_color::mask_raster`), and the colour pass reading the planes.
//!
//! The pass makes, for every raster row of a chain, a u16 plane over a window of the output grid: the row's
//! photo-frame plane resampled through the sampler's positions, then Grow, Feather and the guided filter of
//! Refine edges. The gates: the plane within a stated bound of the CPU's (*starting* 4e-4 absolute), one 10-bit
//! code after C13, **tiles equal the whole frame bit for bit** with their halos, a repeat render and a poisoned
//! device give the same bytes, and a Brush dab stays on its photo feature through every geometry.
//!
//! The scenes are windows of large exports: a small camera image sampled at a scale of 0.2 is an output of
//! 3000 px along the long edge, so the sizes of Feather and Refine edges (fractions of the long edge) are those of
//! a real export, and the windows computed are a tile of it.

mod common;

use std::sync::Arc;

use common::{TestGpu, gpus};
use ym_color::display::{
    ChainParams, ColorChain, Frame, Layout, MainNode, NodeParams, NodeSet, OutputTarget, Trim, View,
};
use ym_color::grade::GradeParams;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoMap};
use ym_color::mask_raster::{
    Area, BrushStrokes, CpuPlanes, PlaneJob, RasterRow, RasterSet, RasterSource, Stroke,
};
use ym_color::plan::{self, MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop as PlanStop};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::sampler::{GridPositions, SamplerGrid, SamplerWarp, sample_to_camera};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::color::{PreparedChain, Stop};
use ym_gpu::kernels::mask_raster::{RasterJob, RasterPlanes, encode_raster_planes};
use ym_gpu::kernels::sampler::encode_sample_at;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

use common::chains;

/// The starting bound of the raster planes, absolute, on the plane's value (`docs/23` section 11).
const PLANE_GATE: f64 = 4e-4;
/// One 10-bit code after C13.
const CODE10: f64 = 1.0 / 1023.0;

// ---- the scene -----------------------------------------------------------------------------------

/// A photo-like scene with structure at every scale (so that the guide has edges to follow).
fn scene(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = common::Rng(seed);
    let mut rgb = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 / w as f64, y as f64 / h as f64);
            let blocks = if ((x / 23) + (y / 17)) % 2 == 0 {
                0.9
            } else {
                -0.9
            };
            let stops = -3.0
                + 5.0 * (0.5 + 0.5 * (fx * 7.0 + fy * 4.0).sin())
                + blocks
                + 0.4 * rng.range(-1.0, 1.0);
            let lum = 0.18 * stops.exp2();
            for k in 0..3 {
                let tint = 0.3 * ((fx * (3.0 + k as f64) + fy * 2.0 * (k as f64 + 1.0)).cos())
                    + 0.1 * rng.range(-1.0, 1.0);
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

/// The attachment's bend (data version 5) fitted so that the canvas's edge maps inside the photo.
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

// ---- the chain -----------------------------------------------------------------------------------

fn brush(points: &[[f64; 2]], size: f64, hardness: f64, erase: bool) -> Stroke {
    Stroke {
        size,
        hardness,
        erase,
        points: points.to_vec(),
    }
}

/// Strokes with edges at several scales: a fat soft stroke, a thin hard one across it, an eraser.
fn strokes() -> Arc<BrushStrokes> {
    Arc::new(BrushStrokes::new(vec![
        brush(&[[0.15, 0.3], [0.45, 0.55], [0.8, 0.4]], 0.16, 0.3, false),
        brush(&[[0.2, 0.7], [0.5, 0.35], [0.85, 0.7]], 0.05, 1.0, false),
        brush(&[[0.45, 0.5], [0.5, 0.45]], 0.04, 0.8, true),
    ]))
}

fn row(grow: f64, feather: f64, refine: f64) -> RasterRow {
    RasterRow {
        source: RasterSource::Brush(strokes()),
        grow,
        feather,
        refine,
    }
}

/// Main and node 2 behind it, node 2 with an exposure trim, a contrast and a saturation and one raster row of
/// the set for each plane (added); the effect is at most about 0.25 in LogC4.
fn chain_with(set: RasterSet) -> ColorChain {
    let plan = plan::compile(
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
                    picture: Some(PictureSource::Node(MAIN_NODE)),
                    key: None,
                },
            ],
            output: PictureSource::Node(2),
        },
        PlanStop::Output,
    )
    .expect("plan");
    let rows = (0..set.rows.len())
        .map(|plane| MaskRow::new(MaskKind::Raster { plane }))
        .collect();
    let mut p: ChainParams = chains::base(OutputTarget::Srgb);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others: [(
            2,
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
                masks: NodeMasks {
                    rows,
                    invert: false,
                },
                ..NodeParams::default()
            },
        )]
        .into(),
    };
    p.rasters = set;
    ColorChain::new(&p).expect("chain")
}

// ---- the planes: the reference and the device ------------------------------------------------------

/// The CPU reference of the planes of `window`, over `grown`.
fn cpu_planes(
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    frame: Area,
    window: Area,
    halo: usize,
) -> CpuPlanes {
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
    let photo = grid.photo_frame();
    let u_px = 1.0 / photo.pixel;
    let outside = |i: usize| sampled.outside.as_ref().is_some_and(|o| o[i] != 0);
    let guide: Option<Vec<f64>> = chain.rasters().needs_guide(u_px).then(|| {
        (0..gw * gh)
            .map(|i| {
                if outside(i) {
                    0.0
                } else {
                    let p = sampled.img.pixel(i % gw, i / gw);
                    chain.guide_pixel([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
                }
            })
            .collect()
    });
    let map = GridPositions::new(grid, (grown.x as usize, grown.y as usize));
    let position = |x: usize, y: usize| {
        if outside(y * gw + x) {
            None
        } else {
            Some(map.position(x, y))
        }
    };
    chain.rasters().planes(&PlaneJob {
        photo,
        grown,
        frame,
        window,
        position: &position,
        guide: guide.as_deref(),
    })
}

/// The device's planes of `window` as u16 codes, one `Vec` for each plane (the window's pixels, no padding).
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
            let src = GpuImage::upload_rgb(ctx, "raster source", &img)?;
            let mut passes = PassList::new(ctx, "raster");
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
                    photo: grid.photo_frame(),
                },
            )?;
            let read = passes.read(planes.buffer(), 0, planes.byte_len())?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            let bytes = done.read(read).to_vec();
            Ok(unpack(&bytes, &planes))
        })
        .expect("GPU raster planes")
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

/// The largest absolute difference of two planes' values (codes over 65535) and the mean.
fn plane_diff(a: &[u16], b: &[u16]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut worst, mut sum) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (f64::from(*x) - f64::from(*y)).abs() / 65535.0;
        worst = worst.max(d);
        sum += d;
    }
    (worst, sum / a.len().max(1) as f64)
}

/// A large export seen from a small camera image: the output is `buffer / scale` pixels along its long edge.
fn export_grid(buffer: (usize, usize), scale: f64) -> (SamplerGrid, Area) {
    let grid = grid_of(buffer, 1, 0, 0.0, scale, (0.0, 0.0));
    let frame = Area::whole(
        (buffer.0 as f64 / scale) as usize,
        (buffer.1 as f64 / scale) as usize,
    );
    (grid, frame)
}

// ---- parity --------------------------------------------------------------------------------------

#[test]
fn the_device_planes_are_the_cpu_planes_within_the_stated_bound() {
    let buffer = (480usize, 320usize);
    let img = scene(buffer.0, buffer.1, 0x5EED);
    let (grid, frame) = export_grid(buffer, 0.2); // 2400 x 1600 output
    let window = Area {
        x: 1000,
        y: 700,
        w: 150,
        h: 110,
    };
    let mut worst_all = 0.0f64;
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        for (name, grow, feather, refine) in [
            ("resample only", 0.0, 0.0, 0.0),
            ("grown", 0.5, 0.0, 0.0),
            ("eroded", -0.4, 0.0, 0.0),
            ("feathered", 0.0, 0.5, 0.0),
            ("refined (the window as it is)", 0.0, 0.0, 0.3),
            ("refined, subsampled", 0.0, 0.0, 1.0),
            ("grown, feathered and refined", 0.3, 0.3, 0.6),
            ("all, subsampled", 0.4, 0.5, 1.0),
        ] {
            let set = RasterSet {
                rows: vec![row(grow, feather, refine)],
            };
            let chain = chain_with(set);
            let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
            let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
            let want = cpu_planes(&chain, &img, &grid, frame, window, halo);
            let (worst, mean) = plane_diff(&got[0], &want.planes[0]);
            eprintln!(
                "{name}: halo {halo} px, the device differs from the CPU by {worst:.2e}, mean {mean:.2e}"
            );
            assert!(worst < PLANE_GATE, "{name}: {worst}");
            worst_all = worst_all.max(worst);
            // The planes are not empty (the test has something to find).
            let (lo, hi) = want.planes[0]
                .iter()
                .fold((u16::MAX, 0u16), |(l, h), &c| (l.min(c), h.max(c)));
            assert!(hi - lo > 8000, "{name}: the plane is flat ({lo} to {hi})");
        }
    }
    eprintln!("the largest difference over the cases: {worst_all:.2e}");
}

#[test]
fn the_planes_agree_through_every_geometry() {
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0xFEED);
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        for (name, exif, turns, angle, scale, t, warp) in [
            ("as shot", 1u16, 0u8, 0.0, 0.3, (0.0, 0.0), false),
            ("cropped", 1, 0, 0.0, 0.3, (0.12, 0.08), false),
            ("straightened", 1, 0, 9.0, 0.3, (0.0, 0.0), false),
            ("turned and mirrored", 6, 1, 0.0, 0.3, (0.0, 0.0), false),
            (
                "mirrored, straightened",
                7,
                2,
                -12.0,
                0.35,
                (0.05, -0.05),
                false,
            ),
            ("shrunk by 2.3 (box)", 1, 0, 0.0, 2.3, (0.0, 0.0), false),
            ("bent", 1, 0, 0.0, 0.3, (0.0, 0.0), true),
            ("bent, turned, tilted", 8, 1, 5.0, 0.3, (0.0, 0.0), true),
        ] {
            let mut grid = grid_of(buffer, exif, turns, angle, scale, t);
            if warp {
                grid = grid.with_warp(Some(bend(1.0, (grid.canvas.0, grid.canvas.1))));
            }
            let frame = Area::whole(
                (grid.canvas.0 / scale) as usize,
                (grid.canvas.1 / scale) as usize,
            );
            let window = Area {
                x: (frame.w / 3) as i64,
                y: (frame.h / 3) as i64,
                w: 120.min(frame.w / 3),
                h: 90.min(frame.h / 3),
            };
            let chain = chain_with(RasterSet {
                rows: vec![row(0.2, 0.3, 0.6)],
            });
            let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
            let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
            let want = cpu_planes(&chain, &img, &grid, frame, window, halo);
            let (worst, mean) = plane_diff(&got[0], &want.planes[0]);
            eprintln!("{name}: halo {halo} px, {worst:.2e}, mean {mean:.2e}");
            assert!(worst < PLANE_GATE, "{name}: {worst}");
        }
    }
}

#[test]
fn a_window_at_the_frame_edge_is_cut_to_the_frame_like_the_cpu_does() {
    let buffer = (480usize, 320usize);
    let img = scene(buffer.0, buffer.1, 0xED6E);
    let (grid, frame) = export_grid(buffer, 0.2);
    let chain = chain_with(RasterSet {
        rows: vec![row(0.3, 0.4, 0.8)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        for window in [
            Area {
                x: 0,
                y: 0,
                w: 100,
                h: 80,
            },
            Area {
                x: frame.w as i64 - 100,
                y: frame.h as i64 - 80,
                w: 100,
                h: 80,
            },
            Area {
                x: 0,
                y: frame.h as i64 / 2,
                w: 90,
                h: 60,
            },
        ] {
            let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
            let want = cpu_planes(&chain, &img, &grid, frame, window, halo);
            let (worst, _) = plane_diff(&got[0], &want.planes[0]);
            assert!(worst < PLANE_GATE, "{window:?}: {worst}");
        }
    }
}

#[test]
fn the_empty_corners_of_a_straightened_photo_are_zero_before_the_steps_and_agree_with_the_cpu() {
    // A photo straightened by 12 degrees and shrunk has empty corners (pixels outside the photo, flagged by the
    // sampler): the resample gives them 0, the guide is 0 there, and the steps run over them as the CPU's do.
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0xC0F5);
    let grid = grid_of(buffer, 1, 0, 12.0, 0.8, (0.0, 0.0));
    let frame = Area::whole(
        (grid.canvas.0 / 0.8) as usize,
        (grid.canvas.1 / 0.8) as usize,
    );
    let chain = chain_with(RasterSet {
        rows: vec![RasterRow {
            source: RasterSource::Brush(Arc::new(BrushStrokes::new(vec![brush(
                &[[0.5, 0.5], [0.02, 0.02]],
                0.9,
                1.0,
                false,
            )]))),
            grow: 0.2,
            feather: 0.3,
            refine: 0.7,
        }],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        let got = device_planes(dev, &chain, &img, &grid, frame, frame, halo);
        let want = cpu_planes(&chain, &img, &grid, frame, frame, halo);
        let (worst, _) = plane_diff(&got[0], &want.planes[0]);
        assert!(worst < PLANE_GATE, "{worst}");
        // The corners really are outside, and the brush covers them in the photo's frame (a stroke to the corner),
        // so the zero there is the letterbox's and not the brush's: the resample is zero at the very corner pixel.
        let sampled = sample_to_camera(&img, (0, 0), &grid, (0, 0), frame.w, frame.h);
        let outside = sampled.outside.as_ref().expect("empty corners");
        assert!(outside.iter().any(|&o| o != 0));
        assert!(outside[0] != 0 && want.planes[0][0] < 1500 && got[0][0] < 1500);
    }
}

#[test]
fn a_tile_coloured_in_two_row_windows_reads_the_planes_of_its_own_rows() {
    // The export tail colours a tile in the pieces that two slabs share: the planes are seen through the rows of
    // each (`RasterPlanes::rows`), and the pieces together are the tile coloured whole.
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x2B0C);
    let (grid, frame) = export_grid(buffer, 0.5);
    let window = Area {
        x: 150,
        y: 90,
        w: 91,
        h: 75,
    };
    let chain = chain_with(RasterSet {
        rows: vec![row(0.3, 0.3, 0.5)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let (c, im, g) = (chain.clone(), img.clone(), grid);
    let (whole, split) = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "src", &im)?;
            let mut passes = PassList::new(ctx, "rows");
            let grown = window.grown(halo, &frame);
            let sampled = encode_sample_at(
                &mut passes,
                &src,
                (0, 0),
                &g,
                (grown.x as usize, grown.y as usize),
                grown.w,
                grown.h,
                true,
            )?;
            let prepared = PreparedChain::new(ctx, &c)?.with_photo_frame(g.photo_frame());
            let planes = encode_raster_planes(
                &mut passes,
                &prepared,
                &c,
                &sampled,
                &RasterJob {
                    frame,
                    grown,
                    window,
                    photo: g.photo_frame(),
                },
            )?;
            let at = ((window.x - grown.x) as usize, (window.y - grown.y) as usize);
            let alloc = |label: &str, h: usize| {
                GpuImage::alloc(
                    ctx,
                    label,
                    window.w,
                    h,
                    4,
                    ym_gpu::cpu::Placement::whole(window.w, h),
                )
            };
            let whole = alloc("whole", window.h)?;
            prepared.with_rasters(planes.clone()).encode_stage(
                &mut passes,
                &sampled,
                Rect::new(at.0, at.1, window.w, window.h),
                Stop::C13,
                &whole,
                (0, 0),
            )?;
            let (top_rows, bottom_rows) = (31usize, window.h - 31);
            let (top, bottom) = (alloc("top", top_rows)?, alloc("bottom", bottom_rows)?);
            prepared
                .with_rasters(planes.rows(0, top_rows))
                .encode_stage(
                    &mut passes,
                    &sampled,
                    Rect::new(at.0, at.1, window.w, top_rows),
                    Stop::C13,
                    &top,
                    (0, 0),
                )?;
            prepared
                .with_rasters(planes.rows(top_rows, bottom_rows))
                .encode_stage(
                    &mut passes,
                    &sampled,
                    Rect::new(at.0, at.1 + top_rows, window.w, bottom_rows),
                    Stop::C13,
                    &bottom,
                    (0, 0),
                )?;
            let reads = (
                whole.encode_read(&mut passes)?,
                top.encode_read(&mut passes)?,
                bottom.encode_read(&mut passes)?,
            );
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            Ok((
                whole.take(&done, reads.0)?,
                (top.take(&done, reads.1)?, bottom.take(&done, reads.2)?),
            ))
        })
        .expect("render");
    let (w, h) = (window.w, window.h);
    let (top_rows, bottom_rows) = (31usize, h - 31);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let a = whole[c * w * h + y * w + x];
                let b = if y < top_rows {
                    split.0[c * w * top_rows + y * w + x]
                } else {
                    split.1[c * w * bottom_rows + (y - top_rows) * w + x]
                };
                assert_eq!(a.to_bits(), b.to_bits(), "plane {c} pixel ({x}, {y})");
            }
        }
    }
}

#[test]
fn several_rows_make_several_planes() {
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x6A55);
    let (grid, frame) = export_grid(buffer, 0.25);
    let window = Area {
        x: 400,
        y: 250,
        w: 130,
        h: 100,
    };
    let set = RasterSet {
        rows: vec![row(0.0, 0.0, 0.0), row(0.0, 0.5, 0.0), row(-0.3, 0.2, 0.9)],
    };
    let chain = chain_with(set);
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
        let want = cpu_planes(&chain, &img, &grid, frame, window, halo);
        assert_eq!(got.len(), 3);
        for (p, (g, w)) in got.iter().zip(&want.planes).enumerate() {
            let (worst, _) = plane_diff(g, w);
            assert!(worst < PLANE_GATE, "plane {p}: {worst}");
        }
        assert_ne!(got[0], got[1]);
    }
}

// ---- tiles ---------------------------------------------------------------------------------------

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit_halos_included() {
    let buffer = (360usize, 240usize);
    let img = scene(buffer.0, buffer.1, 0x71E5);
    let (grid, frame) = export_grid(buffer, 0.5); // 720 x 480
    for (name, rows) in [
        ("grow, feather", vec![row(0.5, 0.4, 0.0)]),
        ("refine", vec![row(0.0, 0.0, 0.7)]),
        (
            "all, two rows",
            vec![row(0.3, 0.3, 0.5), row(-0.2, 0.0, 1.0)],
        ),
    ] {
        let chain = chain_with(RasterSet { rows });
        let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
        assert!(halo >= 3, "{name}: {halo}");
        for dev in gpus() {
            if dev.mode.is_f16() {
                continue;
            }
            let whole = device_planes(dev, &chain, &img, &grid, frame, frame, halo);
            for tile in [96usize, 211] {
                let mut y = 0;
                while y < frame.h {
                    let mut x = 0;
                    while x < frame.w {
                        let window = Area {
                            x: x as i64,
                            y: y as i64,
                            w: tile.min(frame.w - x),
                            h: tile.min(frame.h - y),
                        };
                        let got = device_planes(dev, &chain, &img, &grid, frame, window, halo);
                        for (p, plane) in got.iter().enumerate() {
                            for j in 0..window.h {
                                for i in 0..window.w {
                                    let a = plane[j * window.w + i];
                                    let b = whole[p][(y + j) * frame.w + x + i];
                                    assert_eq!(
                                        a, b,
                                        "{name}: tile {tile} at ({x}, {y}), plane {p}, pixel ({i}, {j})"
                                    );
                                }
                            }
                        }
                        x += tile;
                    }
                    y += tile;
                }
            }
        }
    }
}

#[test]
fn a_halo_one_pixel_short_changes_a_tile_on_the_device_too() {
    let buffer = (360usize, 240usize);
    let img = scene(buffer.0, buffer.1, 0x71E5);
    let (grid, frame) = export_grid(buffer, 0.5);
    let chain = chain_with(RasterSet {
        rows: vec![row(0.5, 0.4, 0.0)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let window = Area {
        x: 300,
        y: 200,
        w: 80,
        h: 60,
    };
    let whole = device_planes(dev, &chain, &img, &grid, frame, frame, halo);
    let short = device_planes(dev, &chain, &img, &grid, frame, window, halo - 1);
    let differs = (0..window.h).any(|j| {
        (0..window.w).any(|i| short[0][j * window.w + i] != whole[0][(200 + j) * frame.w + 300 + i])
    });
    assert!(differs, "a halo one short must show");
}

#[test]
fn a_repeat_and_a_poisoned_device_give_the_same_bytes() {
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x9015);
    let (grid, frame) = export_grid(buffer, 0.25);
    let window = Area {
        x: 300,
        y: 200,
        w: 120,
        h: 90,
    };
    let chain = chain_with(RasterSet {
        rows: vec![row(0.3, 0.3, 0.8)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let a = device_planes(dev, &chain, &img, &grid, frame, window, halo);
    let b = device_planes(dev, &chain, &img, &grid, frame, window, halo);
    assert_eq!(a, b);
    // The tests' devices poison every buffer with NaN; a device that does not gives the same bytes.
    let plain = common::start_unpoisoned(ym_gpu::device::StoragePreference::Auto);
    if let Some(plain) = plain {
        let plain = TestGpu {
            mode: plain.info().storage_mode,
            gpu: plain,
        };
        let c = device_planes(&plain, &chain, &img, &grid, frame, window, halo);
        assert_eq!(a, c, "poisoned and plain devices differ");
    }
}

// ---- the colour pass reads the planes ------------------------------------------------------------

/// The device's C13 stage of a window with the raster planes bound, and the CPU's with its own planes.
fn colour_both(
    dev: &TestGpu,
    chain: &ColorChain,
    img: &RgbImage,
    grid: &SamplerGrid,
    frame: Area,
    window: Area,
) -> (Vec<f32>, Vec<f32>) {
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let (c, im, g) = (chain.clone(), img.clone(), *grid);
    let device = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "raster source", &im)?;
            let mut passes = PassList::new(ctx, "raster colour");
            let grown = window.grown(halo, &frame);
            let sampled = encode_sample_at(
                &mut passes,
                &src,
                (0, 0),
                &g,
                (grown.x as usize, grown.y as usize),
                grown.w,
                grown.h,
                true,
            )?;
            let prepared = PreparedChain::new(ctx, &c)?.with_photo_frame(g.photo_frame());
            let planes = encode_raster_planes(
                &mut passes,
                &prepared,
                &c,
                &sampled,
                &RasterJob {
                    frame,
                    grown,
                    window,
                    photo: g.photo_frame(),
                },
            )?;
            let prepared = prepared.with_rasters(planes);
            let dst = GpuImage::alloc(
                ctx,
                "raster out",
                window.w,
                window.h,
                4,
                ym_gpu::cpu::Placement::whole(window.w, window.h),
            )?;
            // The colour pass over the tile's part of the sampled picture.
            let at = ((window.x - grown.x) as usize, (window.y - grown.y) as usize);
            prepared.encode_stage(
                &mut passes,
                &sampled,
                Rect::new(at.0, at.1, window.w, window.h),
                Stop::C13,
                &dst,
                (0, 0),
            )?;
            let read = dst.encode_read(&mut passes)?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            dst.take(&done, read)
        })
        .expect("GPU colour");
    // The CPU: the window's pixels, the planes the reference made.
    let planes = cpu_planes(chain, img, grid, frame, window, halo);
    let sampled = sample_to_camera(
        img,
        (0, 0),
        grid,
        (window.x as usize, window.y as usize),
        window.w,
        window.h,
    );
    let map = GridPositions::new(grid, (window.x as usize, window.y as usize));
    let photo = grid.photo_frame();
    let f = Frame::planar(window.w, window.h, &sampled.img.data)
        .expect("frame")
        .with_photo(&map, photo)
        .with_rasters(&planes);
    let mut cpu = vec![0.0f32; 3 * window.w * window.h];
    chain
        .render_encoded_f32(View::Final, &f, Layout::Planar, &mut cpu, 1)
        .expect("render");
    (device, cpu)
}

#[test]
fn the_colour_pass_with_the_planes_is_within_one_ten_bit_code_of_the_cpu() {
    let buffer = (480usize, 320usize);
    let img = scene(buffer.0, buffer.1, 0xC010);
    let (grid, frame) = export_grid(buffer, 0.25);
    let window = Area {
        x: 400,
        y: 250,
        w: 160,
        h: 120,
    };
    for (name, rows) in [
        ("a plain dab", vec![row(0.0, 0.0, 0.0)]),
        ("all three", vec![row(0.3, 0.3, 0.7)]),
        ("two rows", vec![row(0.0, 0.2, 0.0), row(-0.4, 0.0, 0.5)]),
    ] {
        let chain = chain_with(RasterSet { rows });
        for dev in gpus() {
            if dev.mode.is_f16() {
                continue;
            }
            let (device, cpu) = colour_both(dev, &chain, &img, &grid, frame, window);
            let n = window.w * window.h;
            let mut worst = 0.0f64;
            for c in 0..3 {
                for i in 0..n {
                    let (a, b) = (f64::from(device[c * n + i]), f64::from(cpu[c * n + i]));
                    worst = worst.max((a - b).abs());
                }
            }
            eprintln!("{name}: C13 device against CPU {worst:.2e} (gate {CODE10:.2e})");
            assert!(worst < CODE10, "{name}: {worst}");
        }
    }
}

#[test]
fn the_colour_pass_of_a_tile_with_its_planes_equals_the_whole_frame_bit_for_bit() {
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x7117);
    let (grid, frame) = export_grid(buffer, 0.5); // 480 x 320
    let chain = chain_with(RasterSet {
        rows: vec![row(0.3, 0.3, 0.6), row(-0.2, 0.0, 0.0)],
    });
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
        let (whole, _) = colour_both(dev, &chain, &img, &grid, frame, frame);
        let n = frame.w * frame.h;
        for window in [
            Area {
                x: 0,
                y: 0,
                w: 130,
                h: 97,
            },
            Area {
                x: 200,
                y: 120,
                w: 151,
                h: 111,
            },
            Area {
                x: 350,
                y: 250,
                w: 130,
                h: 70,
            },
        ] {
            let (tile, _) = colour_both(dev, &chain, &img, &grid, frame, window);
            let m = window.w * window.h;
            for c in 0..3 {
                for j in 0..window.h {
                    for i in 0..window.w {
                        let a = tile[c * m + j * window.w + i];
                        let b = whole
                            [c * n + (window.y as usize + j) * frame.w + window.x as usize + i];
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{window:?} plane {c} pixel ({i}, {j})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_chain_with_raster_rows_and_no_planes_selects_nothing() {
    // The row is an enabled row in the fold with no plane to read: its `s` is 0, as on the CPU (a frame with no
    // planes), so the node acts nowhere; with the planes bound it acts where the dab is.
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x0123);
    let (grid, frame) = export_grid(buffer, 0.5);
    let window = Area {
        x: 100,
        y: 80,
        w: 60,
        h: 50,
    };
    let chain = chain_with(RasterSet {
        rows: vec![row(0.0, 0.0, 0.0)],
    });
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let (c, im, g) = (chain.clone(), img.clone(), grid);
    let unbound = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let src = GpuImage::upload_rgb(ctx, "src", &im)?;
            let mut passes = PassList::new(ctx, "none");
            let sampled = encode_sample_at(
                &mut passes,
                &src,
                (0, 0),
                &g,
                (window.x as usize, window.y as usize),
                window.w,
                window.h,
                true,
            )?;
            let prepared = PreparedChain::new(ctx, &c)?.with_photo_frame(g.photo_frame());
            let dst = GpuImage::alloc(
                ctx,
                "out",
                window.w,
                window.h,
                4,
                ym_gpu::cpu::Placement::whole(window.w, window.h),
            )?;
            prepared.encode_stage(
                &mut passes,
                &sampled,
                Rect::whole(window.w, window.h),
                Stop::C13,
                &dst,
                (0, 0),
            )?;
            let read = dst.encode_read(&mut passes)?;
            let _src = passes.keep(src.buf);
            let done = passes.submit()?.finish()?;
            dst.take(&done, read)
        })
        .expect("render");
    // The CPU, given no planes.
    let sampled = sample_to_camera(
        &img,
        (0, 0),
        &grid,
        (window.x as usize, window.y as usize),
        window.w,
        window.h,
    );
    let map = GridPositions::new(&grid, (window.x as usize, window.y as usize));
    let f = Frame::planar(window.w, window.h, &sampled.img.data)
        .expect("frame")
        .with_photo(&map, grid.photo_frame());
    let mut cpu = vec![0.0f32; 3 * window.w * window.h];
    chain
        .render_encoded_f32(View::Final, &f, Layout::Planar, &mut cpu, 1)
        .expect("render");
    let n = window.w * window.h;
    let worst = (0..3 * n)
        .map(|i| f64::from((unbound[i] - cpu[i]).abs()))
        .fold(0.0, f64::max);
    assert!(worst < CODE10, "{worst}");
    // And the planes matter: bound, the node acts where the dab is.
    let (bound, _) = colour_both(dev, &chain, &img, &grid, frame, window);
    assert_ne!(unbound[..3 * n], bound[..3 * n]);
}

// ---- bands of a plan of several passes -----------------------------------------------------------

/// Main fans out to nine nodes (nine pictures alive: two passes) and nine more join them through key links; the
/// ones after Main in the first and the middle of the second row each have a raster row of the set.
fn fan_chain(set: RasterSet) -> ColorChain {
    let node = |id: u32, picture, key| PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(picture),
        key,
    };
    let mut nodes = vec![node(MAIN_NODE, PictureSource::Input, None)];
    for i in 0..9u32 {
        nodes.push(node(2 + i, PictureSource::Node(MAIN_NODE), None));
    }
    for i in 0..9u32 {
        nodes.push(node(
            20 + i,
            PictureSource::Node(2 + i),
            (i > 0).then(|| 19 + i),
        ));
    }
    let plan = plan::compile(
        &PlanGraph {
            nodes,
            output: PictureSource::Node(28),
        },
        PlanStop::Output,
    )
    .expect("plan");
    assert!(plan.passes.len() >= 2, "the plan has several passes");
    let effect = |ev: f64, rows: Vec<MaskRow>| NodeParams {
        trim: Trim {
            exposure_ev: ev,
            ..Trim::default()
        },
        masks: NodeMasks {
            rows,
            invert: false,
        },
        ..NodeParams::default()
    };
    let mut others = std::collections::BTreeMap::new();
    for i in 0..9u32 {
        others.insert(2 + i, effect(0.05 * f64::from(i + 1), vec![]));
        let rows = match i {
            0 => vec![MaskRow::new(MaskKind::Raster { plane: 0 })],
            5 => vec![MaskRow::new(MaskKind::Raster { plane: 1 })],
            _ => vec![],
        };
        others.insert(20 + i, effect(0.3, rows));
    }
    let mut p: ChainParams = chains::base(OutputTarget::Srgb);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    p.rasters = set;
    ColorChain::new(&p).expect("chain")
}

#[test]
fn bands_of_a_plan_of_several_passes_read_the_planes_of_their_own_rows() {
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0xBA4D);
    let (grid, frame) = export_grid(buffer, 0.5);
    let window = Area {
        x: 100,
        y: 60,
        w: 77,
        h: 61,
    };
    let chain = fan_chain(RasterSet {
        rows: vec![row(0.2, 0.2, 0.0), row(0.0, 0.3, 0.4)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let render = |limit: Option<u64>, bind: bool| {
        let (c, im, g) = (chain.clone(), img.clone(), grid);
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let src = GpuImage::upload_rgb(ctx, "src", &im)?;
                let mut passes = PassList::new(ctx, "bands");
                let grown = window.grown(halo, &frame);
                let sampled = encode_sample_at(
                    &mut passes,
                    &src,
                    (0, 0),
                    &g,
                    (grown.x as usize, grown.y as usize),
                    grown.w,
                    grown.h,
                    true,
                )?;
                let mut prepared = PreparedChain::new(ctx, &c)?.with_photo_frame(g.photo_frame());
                let planes = encode_raster_planes(
                    &mut passes,
                    &prepared,
                    &c,
                    &sampled,
                    &RasterJob {
                        frame,
                        grown,
                        window,
                        photo: g.photo_frame(),
                    },
                )?;
                if bind {
                    prepared = prepared.with_rasters(planes);
                }
                if let Some(bytes) = limit {
                    prepared = prepared.with_pass_buffer_limit(bytes);
                }
                let dst = GpuImage::alloc(
                    ctx,
                    "out",
                    window.w,
                    window.h,
                    4,
                    ym_gpu::cpu::Placement::whole(window.w, window.h),
                )?;
                let at = ((window.x - grown.x) as usize, (window.y - grown.y) as usize);
                prepared.encode_stage(
                    &mut passes,
                    &sampled,
                    Rect::new(at.0, at.1, window.w, window.h),
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
    let whole = render(None, true);
    // The planes act: the same plan with none bound is another picture.
    assert!(render(None, false) != whole);
    // Bands of 1, 7 and 32 rows: the plan's pass buffers are four f32 planes of a row at most.
    let planes_per_pixel = {
        let probe = fan_chain(RasterSet {
            rows: vec![row(0.0, 0.0, 0.0), row(0.0, 0.0, 0.0)],
        });
        let dev_chain = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                Ok(PreparedChain::new(ctx, &probe)?.pass_buffer_planes())
            })
            .expect("prepared");
        u64::from(dev_chain)
    };
    assert!(planes_per_pixel > 0);
    for rows in [1u64, 7, 32] {
        let banded = render(
            Some(rows * 4 * planes_per_pixel * window.w.next_multiple_of(2) as u64),
            true,
        );
        assert!(
            banded
                .iter()
                .zip(&whole)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "bands of {rows} rows differ from the whole window"
        );
    }
}

// ---- preview against export ----------------------------------------------------------------------

#[test]
fn the_same_mask_at_two_scales_agrees_after_reducing_the_larger() {
    // The export (k = 1) and a preview four times smaller (k = 4) of the same 1200 x 800 photo: the Brush row with
    // Grow, Feather and Refine edges at both, the larger plane reduced by 4 x 4 boxes, against the smaller.
    let buffer = (1200usize, 800usize);
    let img = scene(buffer.0, buffer.1, 0x5CA1);
    let rows = vec![row(0.3, 0.4, 0.6)];
    let make = |scale: f64| {
        let grid = grid_of(buffer, 1, 0, 0.0, scale, (0.0, 0.0));
        let frame = Area::whole(
            (buffer.0 as f64 / scale) as usize,
            (buffer.1 as f64 / scale) as usize,
        );
        let chain = chain_with(RasterSet { rows: rows.clone() });
        let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
        let planes = cpu_planes(&chain, &img, &grid, frame, frame, halo);
        (planes, frame)
    };
    let (full, f1) = make(1.0);
    let (small, f4) = make(4.0);
    assert_eq!((f1.w, f1.h), (1200, 800));
    assert_eq!((f4.w, f4.h), (300, 200));
    let (mut largest, mut sum) = (0.0f64, 0.0f64);
    for y in 0..f4.h {
        for x in 0..f4.w {
            let mut m = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    m += f64::from(full.planes[0][(4 * y + j) * f1.w + 4 * x + i]) / 65535.0;
                }
            }
            m /= 16.0;
            let d = (m - f64::from(small.planes[0][y * f4.w + x]) / 65535.0).abs();
            largest = largest.max(d);
            sum += d;
        }
    }
    let mean = sum / (f4.w * f4.h) as f64;
    eprintln!(
        "a Brush row with Grow 0.3, Feather 0.4, Refine 0.6 at k = 4 and k = 1: largest {largest:.4}, mean {mean:.5}"
    );
    assert!(largest < 0.1, "{largest}");
    assert!(mean < 0.005, "{mean}");
}

// ---- geometry ------------------------------------------------------------------------------------

/// A flat scene with a dot at `centre` (fractions of the oriented photo).
fn dot_scene((bw, bh): (usize, usize), exif: u16, centre: [f64; 2], sigma: f64) -> RgbImage {
    let mut img = RgbImage::new(bw, bh);
    let (ow, oh) = if (5..=8).contains(&exif) {
        (bh as f64, bw as f64)
    } else {
        (bw as f64, bh as f64)
    };
    let long = ow.max(oh);
    for y in 0..bh {
        for x in 0..bw {
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
            let d2 =
                ((o[0] - centre[0]) * ow / long).powi(2) + ((o[1] - centre[1]) * oh / long).powi(2);
            let v = 0.02 + 0.8 * (-d2 / (2.0 * sigma * sigma)).exp();
            img.set_pixel(x, y, [v as f32, v as f32, v as f32]);
        }
    }
    img
}

#[test]
fn a_brush_dab_stays_on_its_photo_feature_through_every_geometry() {
    // docs/23 section 3.6: a photo cropped, straightened, turned or bent keeps a mask on the same feature. The
    // scene is flat with a dot at the dab; the centroid of the raster plane over the render is the centroid of the
    // dot over the render, to within one output pixel.
    let buffer = (240usize, 160usize);
    let centre = [0.3, 0.62];
    let set = RasterSet {
        rows: vec![RasterRow {
            source: RasterSource::Brush(Arc::new(BrushStrokes::new(vec![brush(
                &[centre],
                0.08,
                0.6,
                false,
            )]))),
            grow: 0.0,
            feather: 0.0,
            refine: 0.0,
        }],
    };
    let chain = chain_with(set);
    for dev in gpus() {
        if dev.mode.is_f16() {
            continue;
        }
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
            let img = dot_scene(buffer, exif, centre, 0.035);
            let mut grid = grid_of(buffer, exif, turns, angle, scale, t);
            if warp {
                grid = grid.with_warp(Some(bend(1.0, (grid.canvas.0, grid.canvas.1))));
            }
            let (w, h) = (
                ((grid.canvas.0 / scale) as usize).clamp(8, 300),
                ((grid.canvas.1 / scale) as usize).clamp(8, 300),
            );
            let frame = Area::whole(w, h);
            let planes = device_planes(dev, &chain, &img, &grid, frame, frame, 0);
            let sampled = sample_to_camera(&img, (0, 0), &grid, (0, 0), w, h);
            let (mut dot, mut mk) = ([0.0f64; 3], [0.0f64; 3]);
            for y in 0..h {
                for x in 0..w {
                    if sampled.outside.as_ref().is_some_and(|o| o[y * w + x] != 0) {
                        continue;
                    }
                    let b = f64::from(sampled.img.pixel(x, y)[0]) - 0.02;
                    let m = f64::from(planes[0][y * w + x]) / 65535.0;
                    for (acc, wgt) in [(&mut dot, b), (&mut mk, m)] {
                        acc[0] += wgt;
                        acc[1] += wgt * (x as f64 + 0.5);
                        acc[2] += wgt * (y as f64 + 0.5);
                    }
                }
            }
            assert!(
                dot[0] > 1.0 && mk[0] > 1.0,
                "{name}: the dot or the dab is out of view"
            );
            let (dx, dy) = (dot[1] / dot[0], dot[2] / dot[0]);
            let (mx, my) = (mk[1] / mk[0], mk[2] / mk[0]);
            let off = (dx - mx).hypot(dy - my);
            eprintln!(
                "{name}: the dot at ({dx:.2}, {dy:.2}), the dab at ({mx:.2}, {my:.2}) px: {off:.3} px apart"
            );
            assert!(off < 1.0, "{name}: the dab is {off:.2} px off the dot");
        }
    }
}

// ---- cost ----------------------------------------------------------------------------------------

/// Not a gate: the cost of the raster pass of one 1024 px tile of a 24 MP export (6000 x 4000) on whatever device
/// the tests run on (lavapipe in CI), with two raster rows at the settings of a heavy use: Grow, Feather and Refine
/// edges at 40 %, and a second row with Feather and Refine edges at 100 %. The tile is computed over itself grown
/// by the halo; an export is 24 of them (6 x 4).
#[test]
fn the_raster_pass_of_a_24_mp_export_tile_is_timed() {
    use std::time::Instant;
    let buffer = (240usize, 160usize);
    let img = scene(buffer.0, buffer.1, 0x7144);
    let (grid, frame) = export_grid(buffer, 0.04); // 6000 x 4000
    assert_eq!((frame.w, frame.h), (6000, 4000));
    let chain = chain_with(RasterSet {
        rows: vec![row(0.4, 0.4, 0.4), row(0.0, 1.0, 1.0)],
    });
    let halo = chain.rasters().halo(1.0 / grid.photo_frame().pixel);
    let window = Area {
        x: 2500,
        y: 1500,
        w: 1024,
        h: 1024,
    };
    let dev = &gpus()[0];
    if dev.mode.is_f16() {
        return;
    }
    let grown = window.grown(halo, &frame);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = device_planes(dev, &chain, &img, &grid, frame, window, halo);
        best = best.min(t.elapsed().as_secs_f64());
    }
    eprintln!(
        "TIMING: two raster rows on a 1024 px tile of a 24 MP export, halo {halo} px, {} x {} grown: {best:.2} s a tile, {:.0} s for the 24 tiles",
        grown.w,
        grown.h,
        best * 24.0
    );
}
