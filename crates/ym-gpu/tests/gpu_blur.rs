//! The Blur node's scaffold on the device (`docs/33-blur.md` section 6, BL-05; `kernels::blur_stack`) against its CPU twins
//! (`cpu::blur`, in `f64`): no type's kernel is here, so the passes every type shares are tested alone and with **stand-in
//! kernels written in this file** (an identity kernel that copies the weighted planes, and a cell kernel that reads one level of
//! the pyramid back to the pixels), which is the seam a type plugs into (`BlurKernel`).
//!
//! * the passes alone: prep (decode, the weight, the weighted planes), the pyramid at absolute alignment (cells cut by the
//!   frame's edge included), the nearest and the bilinear read-back, and the blend with the encode and its exact `m = 0` and
//!   `m = 1` branches, each within 2e-4 of its twin (relative for the linear-light planes, absolute in LogC4 for the pictures);
//! * the node in a stack: the picture a Blur node makes after a prefix chain and a mask chain, and the whole chain at C10 with
//!   the colour kernel's `N_BLUR` step, against the CPU's chain given the twin's picture;
//! * tiles equal the whole frame bit for bit, a repeat and a run without poison are the same bits;
//! * a type with no kernel is refused with its own name, and the scheduler's reach grows the tile.
#![allow(clippy::needless_range_loop)]

mod common;
mod film_common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::chains;
use common::{TestGpu, gpus, start_unpoisoned};
use film_common::image_at;
use ym_color::blur::{BlurKind, BlurParams};
use ym_color::display::{
    ChainParams, ColorChain, FilmStep, Frame as ColorFrame, Layout, MainNode, NodeParams, NodeSet,
    OutputTarget, Trim,
};
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoFrame, PhotoMap, RasterPlanes};
use ym_color::mask_raster::Area;
use ym_color::plan::{self, MAIN_NODE, PictureSource, PlanGraph, PlanNode, compile};
use ym_gpu::cpu::Placement;
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::blur as twin;
use ym_gpu::cpu::blur::CellGrid;
use ym_gpu::cpu::tiling::Rect as GRect;
use ym_gpu::device::{Gpu, GpuContext, GpuError, StorageMode, StoragePreference};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::blur_stack::{
    BlurKernel, BlurWork, Pyramid, SumsImage, encode_blend, encode_blur_picture, encode_prep,
    encode_pyramid, encode_upsample,
};
use ym_gpu::kernels::color::{PreparedChain, Stop};
use ym_gpu::kernels::copy::encode_copy_rect;
use ym_gpu::kernels::film_stack::{FilmNodeSpec, FilmStack, NodeWork};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The gate of `docs/33` section 3.7 for the scaffold's own passes: relative for the planes in linear light, absolute in LogC4
/// for the pictures.
const GATE: f64 = 2e-4;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

// ---- the scene ---------------------------------------------------------------------------------------------

const FRAME: (usize, usize) = (300, 200);

/// A LogC4 value at the absolute pixel `(x, y)`: a gradient, a bright disc (code about 0.9) and a dark square below the
/// LogC4 toe (negative codes), a function of the place alone so that every window of it agrees with every other.
fn logc4_at(x: i64, y: i64) -> [f64; 3] {
    let (u, v) = (x as f64 / 300.0, y as f64 / 200.0);
    let mut c = [0.25 + 0.3 * u, 0.3 + 0.2 * v, 0.2 + 0.25 * (1.0 - u)];
    let r = ((u - 0.35).powi(2) + (1.5 * (v - 0.4)).powi(2)).sqrt();
    if r < 0.12 {
        c = [0.9, 0.88, 0.85];
    }
    if (200..230).contains(&x) && (120..150).contains(&y) {
        c = [-0.02, -0.015, -0.01];
    }
    c
}

/// The node's mask at `(x, y)`: 0 on the left (x < 75), 1 on the right (x > 175), a ramp between, and a few values the clean-up must handle (NaN,
/// below 0 and above 1).
fn mask_at(x: i64, y: i64) -> f64 {
    if (x, y) == (97, 51) {
        return f64::NAN;
    }
    if (x, y) == (98, 51) {
        return -0.5;
    }
    if (x, y) == (99, 51) {
        return 1.5;
    }
    ((x as f64 - 75.0) / 100.0).clamp(0.0, 1.0)
}

fn picture(area: Area) -> Vec<f32> {
    let n = area.w * area.h;
    let mut d = vec![0.0f32; 3 * n];
    for y in 0..area.h {
        for x in 0..area.w {
            let c = logc4_at(area.x + x as i64, area.y + y as i64);
            for k in 0..3 {
                d[k * n + y * area.w + x] = c[k] as f32;
            }
        }
    }
    d
}

fn mask(area: Area) -> Vec<f32> {
    let mut d = vec![0.0f32; area.w * area.h];
    for y in 0..area.h {
        for x in 0..area.w {
            d[y * area.w + x] = mask_at(area.x + x as i64, area.y + y as i64) as f32;
        }
    }
    d
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn upload(ctx: &GpuContext, label: &str, area: Area, planes: usize, d: &[f32]) -> GpuImage {
    GpuImage::upload(
        ctx,
        label,
        area.w,
        area.h,
        planes,
        d,
        Placement::whole(area.w, area.h),
    )
    .expect("upload")
}

/// The largest relative difference, `|a - b| / max(|b|, floor)`.
fn rel_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs() / y.abs().max(1e-3))
        .fold(0.0, f64::max)
}

fn abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

// ---- the passes alone --------------------------------------------------------------------------------------

#[test]
fn prep_decodes_and_weighs_like_its_twin() {
    let read = Area {
        x: 40,
        y: 30,
        w: 173,
        h: 121,
    };
    let (pic, msk) = (picture(read), mask(read));
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (highlights, respect) in [(0.0, true), (0.0, false), (60.0, true), (100.0, false)] {
            let (p, m) = (pic.clone(), msk.clone());
            let got: Vec<f32> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "blur prep");
                    let picture = upload(ctx, "picture", read, 3, &p);
                    let mask = upload(ctx, "mask", read, 1, &m);
                    let dst = GpuImage::alloc(
                        ctx,
                        "dst",
                        read.w,
                        read.h,
                        4,
                        Placement::whole(read.w, read.h),
                    )?;
                    encode_prep(
                        &mut passes,
                        &picture,
                        &mask,
                        &dst,
                        read,
                        (highlights, respect),
                    )?;
                    let r = dst.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    dst.take(&done, r)
                })
                .expect("prep");
            let want = twin::prep(
                &f64s(&pic),
                &f64s(&msk),
                read.w,
                read.h,
                highlights,
                respect,
            );
            let d = rel_diff(&got, &want);
            println!(
                "prep, Highlights {highlights}, respect {respect}: largest relative difference {d:.2e}"
            );
            assert!(d < GATE, "Highlights {highlights}, respect {respect}: {d}");
            worst = worst.max(d);
        }
    }
    println!("prep: worst {worst:.2e}");
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_pyramid_is_the_twins_inside_the_frame_and_at_its_edge() {
    let frame = Area::whole(FRAME.0, FRAME.1);
    let inner = Area {
        x: 23,
        y: 17,
        w: 201,
        h: 143,
    };
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for read in [inner, frame] {
            let (pic, msk) = (picture(read), mask(read));
            let (p, m) = (pic.clone(), msk.clone());
            let levels = 4u32;
            let got: Vec<(CellGrid, Vec<f32>)> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "blur pyramid");
                    let picture = upload(ctx, "picture", read, 3, &p);
                    let mask = upload(ctx, "mask", read, 1, &m);
                    let pyr = encode_pyramid(
                        ctx,
                        &mut passes,
                        (&picture, &mask),
                        (read, frame),
                        (0.0, true),
                        levels,
                    )?;
                    let reads: Vec<_> = pyr
                        .levels
                        .iter()
                        .map(|l| l.img.encode_read(&mut passes).map(|r| (l.grid, r)))
                        .collect::<Result<_, _>>()?;
                    let done = passes.submit()?.finish()?;
                    reads
                        .into_iter()
                        .zip(&pyr.levels)
                        .map(|((g, r), l)| l.img.take(&done, r).map(|v| (g, v)))
                        .collect()
                })
                .expect("pyramid");
            let base = twin::prep(&f64s(&pic), &f64s(&msk), read.w, read.h, 0.0, true);
            let g0 = CellGrid::of(read, frame, 0).expect("grid");
            let want = twin::pyramid(&base, g0, frame, levels);
            assert_eq!(got.len(), want.len(), "the levels {read:?}");
            assert_eq!(got.len(), levels as usize + 1);
            for ((gg, gv), (wg, wv)) in got.iter().zip(&want) {
                assert_eq!(gg, wg);
                let d = rel_diff(gv, wv);
                println!(
                    "pyramid over {read:?}, level {}: {} x {} cells, largest relative difference {d:.2e}",
                    gg.level, gg.w, gg.h
                );
                assert!(d < GATE, "level {}: {d}", gg.level);
                worst = worst.max(d);
            }
        }
    }
    println!("pyramid: worst {worst:.2e}");
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_read_back_is_the_twins_nearest_and_bilinear() {
    let frame = Area::whole(FRAME.0, FRAME.1);
    let read = Area {
        x: 16,
        y: 16,
        w: 220,
        h: 150,
    };
    let window = Area {
        x: 50,
        y: 40,
        w: 101,
        h: 77,
    };
    let mut worst = 0.0f64;
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (level, bilinear) in [(2u32, false), (2, true), (3, true), (4, true)] {
            let (pic, msk) = (picture(read), mask(read));
            let (p, m) = (pic.clone(), msk.clone());
            let got: Vec<f32> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "blur upsample");
                    let picture = upload(ctx, "picture", read, 3, &p);
                    let mask = upload(ctx, "mask", read, 1, &m);
                    let pyr = encode_pyramid(
                        ctx,
                        &mut passes,
                        (&picture, &mask),
                        (read, frame),
                        (0.0, true),
                        level,
                    )?;
                    let cells = pyr.level(level).expect("level");
                    let dst = GpuImage::alloc(
                        ctx,
                        "dst",
                        window.w,
                        window.h,
                        4,
                        Placement::whole(window.w, window.h),
                    )?;
                    encode_upsample(&mut passes, cells, &dst, window, bilinear)?;
                    let r = dst.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    dst.take(&done, r)
                })
                .expect("upsample");
            let base = twin::prep(&f64s(&pic), &f64s(&msk), read.w, read.h, 0.0, true);
            let g0 = CellGrid::of(read, frame, 0).expect("grid");
            let levels = twin::pyramid(&base, g0, frame, level);
            let (grid, planes) = &levels[level as usize];
            let want = twin::upsample(planes, grid, window, bilinear);
            let d = rel_diff(&got, &want);
            println!(
                "read-back of level {level}, bilinear {bilinear}: largest relative difference {d:.2e}"
            );
            assert!(d < GATE, "level {level}, bilinear {bilinear}: {d}");
            worst = worst.max(d);
        }
    }
    println!("read-back: worst {worst:.2e}");
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_blend_is_the_twins_and_its_exact_branches_are_exact() {
    let read = Area {
        x: 30,
        y: 20,
        w: 200,
        h: 140,
    };
    let made = Area {
        x: 60,
        y: 40,
        w: 130,
        h: 90,
    };
    let (pic, msk) = (picture(read), mask(read));
    // Sums that are not the identity: N and D of a blurred copy (a shifted picture), with weights of 0 to 1 and a few of 0.
    let n = made.w * made.h;
    let mut sums = vec![0.0f64; 4 * n];
    for y in 0..made.h {
        for x in 0..made.w {
            let c = logc4_decode_rgb(logc4_at(made.x + x as i64 + 3, made.y + y as i64 - 2));
            let d = if (x + y) % 17 == 0 {
                0.0
            } else {
                0.25 + 0.75 * ((x * 7 + y * 3) % 10) as f64 / 10.0
            };
            for k in 0..3 {
                sums[k * n + y * made.w + x] = c[k] * d;
            }
            sums[3 * n + y * made.w + x] = d;
        }
    }
    let sums32: Vec<f32> = sums.iter().map(|&v| v as f32).collect();
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let (p, m, s) = (pic.clone(), msk.clone(), sums32.clone());
        let got: Vec<f32> = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "blur blend");
                let picture = upload(ctx, "picture", read, 3, &p);
                let mask = upload(ctx, "mask", read, 1, &m);
                let sums = SumsImage {
                    img: upload(ctx, "sums", made, 4, &s),
                    area: made,
                };
                let out = GpuImage::alloc(
                    ctx,
                    "out",
                    made.w,
                    made.h,
                    3,
                    Placement::whole(made.w, made.h),
                )?;
                encode_blend(&mut passes, (&picture, &mask), read, &sums, &out)?;
                let r = out.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                out.take(&done, r)
            })
            .expect("blend");
        // The twin over the f32 sums the device was given.
        let want = twin::blend(&f64s(&pic), &f64s(&msk), read, &f64s(&sums32), made);
        let d = abs_diff(&got, &want);
        println!("blend: largest difference {d:.2e} in LogC4");
        assert!(d < GATE, "{d}");
        // m = 0 (and NaN, and a negative mask) is the input's bits; m = 1 is the encode of N / (D + eps).
        let (mut zeros, mut ones) = (0, 0);
        for y in 0..made.h {
            for x in 0..made.w {
                let (ax, ay) = (made.x + x as i64, made.y + y as i64);
                let m = mask_at(ax, ay);
                let m = if m.is_nan() { 0.0 } else { m.clamp(0.0, 1.0) };
                for k in 0..3 {
                    let g = got[k * n + y * made.w + x];
                    if m == 0.0 {
                        zeros += 1;
                        let i = (ay - read.y) as usize * read.w + (ax - read.x) as usize;
                        assert_eq!(g.to_bits(), pic[k * read.w * read.h + i].to_bits());
                    } else if m == 1.0 {
                        ones += 1;
                        let i = y * made.w + x;
                        let q = f64::from(sums32[3 * n + i]) + ym_gpu::cpu::blur::BLEND_EPS;
                        let lin = [
                            f64::from(sums32[i]) / q,
                            f64::from(sums32[n + i]) / q,
                            f64::from(sums32[2 * n + i]) / q,
                        ];
                        assert!(
                            (f64::from(g) - logc4_encode_rgb(lin)[k]).abs() < GATE,
                            "m = 1 at ({ax}, {ay})"
                        );
                    }
                }
            }
        }
        assert!(zeros > 0 && ones > 0);
        println!("blend: {zeros} samples at m = 0 are the input's bits, {ones} at m = 1");
    }
    assert!(seen || gpus().is_empty());
}

// ---- the stand-in kernels ----------------------------------------------------------------------------------

/// The identity: `N = omega X` and `D = omega` of the pixel itself (the weighted planes copied over the made area).
#[derive(Debug)]
struct Identity;

impl BlurKernel for Identity {
    fn name(&self) -> &'static str {
        "Identity"
    }
    fn levels(&self) -> u32 {
        0
    }
    fn encode(
        &self,
        _ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError> {
        let read = src.read_area();
        let rect = GRect::new(
            (out.area.x - read.x) as usize,
            (out.area.y - read.y) as usize,
            out.area.w,
            out.area.h,
        );
        encode_copy_rect(passes, &src.levels[0].img, rect, &out.img, (0, 0))
    }
}

/// The cell kernel: the level `L` of the pyramid read back bilinearly (a coarse grid with no blur of its own): the type
/// kernels' shape, with the pyramid and the read-back, and nothing of their weights.
#[derive(Debug)]
struct Cells(u32);

impl BlurKernel for Cells {
    fn name(&self) -> &'static str {
        "Cells"
    }
    fn levels(&self) -> u32 {
        self.0
    }
    fn encode(
        &self,
        _ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError> {
        let level = src
            .level(self.0)
            .ok_or_else(|| GpuError::InvalidInput("the pyramid has no such level".to_owned()))?;
        encode_upsample(passes, level, &out.img, out.area, true)
    }
}

fn params(respect: bool) -> BlurParams {
    BlurParams {
        kind: BlurKind::Soft { radius: 10.0 },
        respect_mask_edges: respect,
    }
}

fn work(kernel: Option<Arc<dyn BlurKernel>>, mask: Arc<ColorChain>, reach: usize) -> BlurWork {
    BlurWork {
        params: params(true),
        mask,
        reach,
        kernel,
    }
}

// ---- a node on its own: the picture ------------------------------------------------------------------------

/// The picture a node makes over `made` from `read`, with the stand-in kernel, read back.
fn node_picture(
    gpu: &Gpu,
    kernel: Arc<dyn BlurKernel>,
    respect: bool,
    (read, made): (Area, Area),
) -> Vec<f32> {
    let (pic, msk) = (picture(read), mask(read));
    let frame = Area::whole(FRAME.0, FRAME.1);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur node");
        let picture = upload(ctx, "picture", read, 3, &pic);
        let mask = upload(ctx, "mask", read, 1, &msk);
        let w = BlurWork {
            params: BlurParams {
                respect_mask_edges: respect,
                ..params(true)
            },
            mask: Arc::new(unused_chain()),
            reach: 0,
            kernel: Some(kernel),
        };
        let out =
            encode_blur_picture(ctx, &mut passes, &w, (&picture, &mask), (read, made, frame))?;
        let r = out.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        out.img.take(&done, r)
    })
    .expect("node picture")
}

/// A chain no pass of these tests runs (the node's mask is made by the caller there).
fn unused_chain() -> ColorChain {
    ColorChain::new(&chains::base(OutputTarget::Srgb)).expect("chain")
}

#[test]
fn an_identity_kernel_gives_the_input_and_the_cell_kernel_the_twins_picture() {
    let read = Area {
        x: 20,
        y: 20,
        w: 240,
        h: 160,
    };
    let made = Area {
        x: 70,
        y: 60,
        w: 120,
        h: 70,
    };
    let (pic, msk) = (picture(read), mask(read));
    let frame = Area::whole(FRAME.0, FRAME.1);
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for respect in [true, false] {
            // Identity: the node's output is its input (D X over D), within the f32 round trip, and exactly where m = 0.
            let got = node_picture(&dev.gpu, Arc::new(Identity), respect, (read, made));
            let n = made.w * made.h;
            let mut worst = 0.0f64;
            for y in 0..made.h {
                for x in 0..made.w {
                    let (ax, ay) = (made.x + x as i64, made.y + y as i64);
                    let i = (ay - read.y) as usize * read.w + (ax - read.x) as usize;
                    for k in 0..3 {
                        let input = pic[k * read.w * read.h + i];
                        let g = got[k * n + y * made.w + x];
                        worst = worst.max(f64::from((g - input).abs()));
                    }
                }
            }
            println!(
                "identity, respect {respect}: largest difference from the input {worst:.2e} in LogC4"
            );
            assert!(worst < GATE, "respect {respect}: {worst}");
            // The cell kernel against the twin's chain: prep, pyramid, read-back, blend.
            for level in [2u32, 3] {
                let got = node_picture(&dev.gpu, Arc::new(Cells(level)), respect, (read, made));
                let base = twin::prep(&f64s(&pic), &f64s(&msk), read.w, read.h, 0.0, respect);
                let g0 = CellGrid::of(read, frame, 0).expect("grid");
                let levels = twin::pyramid(&base, g0, frame, level);
                let (grid, planes) = &levels[level as usize];
                let sums = twin::upsample(planes, grid, made, true);
                let want = twin::blend(&f64s(&pic), &f64s(&msk), read, &sums, made);
                let d = abs_diff(&got, &want);
                println!(
                    "cells at level {level}, respect {respect}: picture within {d:.2e} of the twin's, in LogC4"
                );
                assert!(d < GATE, "level {level}, respect {respect}: {d}");
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn a_type_with_no_kernel_is_refused_with_its_name() {
    let read = Area {
        x: 0,
        y: 0,
        w: 64,
        h: 64,
    };
    let frame = Area::whole(FRAME.0, FRAME.1);
    let kinds = [
        (BlurKind::Soft { radius: 20.0 }, "Soft"),
        (
            BlurKind::Motion {
                length: 20.0,
                angle_deg: 10.0,
            },
            "Motion",
        ),
        (
            BlurKind::Zoom {
                amount: 20.0,
                center: [0.5, 0.5],
            },
            "Zoom",
        ),
    ];
    for dev in f32_devices() {
        for (kind, name) in kinds {
            let (pic, msk) = (picture(read), mask(read));
            let r = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let mut passes = PassList::new(ctx, "blur refused");
                    let picture = upload(ctx, "picture", read, 3, &pic);
                    let mask = upload(ctx, "mask", read, 1, &msk);
                    let w = BlurWork {
                        params: BlurParams {
                            kind,
                            respect_mask_edges: true,
                        },
                        mask: Arc::new(unused_chain()),
                        reach: 0,
                        kernel: ym_gpu::kernels::blur_stack::device_kernel(&BlurParams {
                            kind,
                            respect_mask_edges: true,
                        }),
                    };
                    encode_blur_picture(
                        ctx,
                        &mut passes,
                        &w,
                        (&picture, &mask),
                        (read, read, frame),
                    )
                    .map(|_| ())
                });
            let Err(e) = r else {
                panic!("{name} ran on the device without a kernel");
            };
            let text = e.to_string();
            assert!(
                text.contains(&format!("a {name} Blur node is not on the device yet")),
                "{text}"
            );
        }
    }
}

// ---- the node in a stack -----------------------------------------------------------------------------------

fn node(id: u32, x: f64, picture: PictureSource) -> PlanNode {
    PlanNode {
        id,
        pos: [x, 0.0],
        picture: Some(picture),
        key: None,
    }
}

/// Main, the Blur node 2, a corrector 3.
fn graph() -> PlanGraph {
    PlanGraph {
        nodes: vec![
            node(MAIN_NODE, 0.0, PictureSource::Input),
            node(2, 1.0, PictureSource::Node(MAIN_NODE)),
            node(3, 2.0, PictureSource::Node(2)),
        ],
        output: PictureSource::Node(3),
    }
}

fn chain_params(g: &PlanGraph, stop: plan::Stop) -> ChainParams {
    let mut p = chains::base(OutputTarget::Srgb);
    let mut others: BTreeMap<u32, NodeParams> = BTreeMap::new();
    // The node's mask: a range on the brightness of its input, soft, so that `m` takes every value between 0 and 1 over the
    // scene's gradient and the bright disc.
    others.insert(
        2,
        NodeParams {
            masks: NodeMasks {
                rows: vec![MaskRow::new(MaskKind::BrightnessRange {
                    low: 0.3,
                    high: 0.75,
                    softness: 0.6,
                })],
                invert: false,
            },
            ..NodeParams::default()
        },
    );
    others.insert(
        3,
        NodeParams {
            trim: Trim {
                exposure_ev: 0.4,
                ..Trim::default()
            },
            ..NodeParams::default()
        },
    );
    p.nodes = NodeSet {
        plan: compile(g, stop).expect("plan"),
        main: MainNode::default(),
        others,
    };
    p.films = BTreeMap::from([(
        2,
        FilmStep {
            strength: 1.0,
            bypass: false,
            face: false,
            blur: true,
        },
    )]);
    p
}

/// The chain of the render, the node's prefix chain and its mask chain (what `ym-core`'s `film_pass` builds).
fn chains_of() -> (Arc<ColorChain>, Arc<ColorChain>, Arc<ColorChain>) {
    let g = graph();
    let whole = ColorChain::new(&chain_params(&g, plan::Stop::Output)).expect("chain");
    let make = |stop| {
        let mut p = chain_params(&g, stop);
        p.look = None;
        p.films.get_mut(&2).expect("film").bypass = true;
        Arc::new(ColorChain::new(&p).expect("chain"))
    };
    (
        Arc::new(whole),
        make(plan::Stop::Picture(2)),
        make(plan::Stop::Mask(2)),
    )
}

fn stack_of(
    prefix: &Arc<ColorChain>,
    mask: &Arc<ColorChain>,
    kernel: Arc<dyn BlurKernel>,
    reach: usize,
) -> FilmStack {
    FilmStack::new(
        Area::whole(FRAME.0, FRAME.1),
        vec![FilmNodeSpec {
            node: 2,
            prefix: Arc::clone(prefix),
            work: NodeWork::Blur(Box::new(work(Some(kernel), Arc::clone(mask), reach))),
            reads: None,
        }],
    )
}

/// The camera RGB of the photo over `rect` (absolute pixels), planar, linear: a function of the place alone.
fn camera(rect: Area) -> RgbImage {
    let aspect = FRAME.1 as f64 / FRAME.0 as f64;
    let (w, h) = (rect.w, rect.h);
    let mut data = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let v = image_at(
                ((rect.x + x as i64) as f64 + 0.5) / FRAME.0 as f64,
                ((rect.y + y as i64) as f64 + 0.5) / FRAME.0 as f64,
                aspect,
            );
            let lin = logc4_decode_rgb(v);
            for c in 0..3 {
                data[c * w * h + y * w + x] = lin[c] as f32;
            }
        }
    }
    RgbImage::from_planar(w, h, data).expect("image")
}

/// The picture the stack makes over `need` and the area it covers, read back, planar LogC4.
fn stack_picture(gpu: &Gpu, stack: &FilmStack, need: Area) -> (Area, Vec<f32>) {
    let stack = stack.clone();
    let wide_area = stack.read_area(need);
    let wide_img = camera(wide_area);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "blur stack");
        let wide = GpuImage::upload_rgb(ctx, "wide", &wide_img)?;
        let pictures = stack.encode(ctx, &mut passes, &wide, wide_area, need, None, None)?;
        let p = pictures
            .get(2)
            .cloned()
            .ok_or_else(|| GpuError::InvalidInput("no picture".to_owned()))?;
        let r = p.img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let _ = wide;
        let area = Area {
            x: i64::from(p.rect.x0),
            y: i64::from(p.rect.y0),
            w: p.rect.width() as usize,
            h: p.rect.height() as usize,
        };
        p.img.take(&done, r).map(|v| (area, v))
    })
    .expect("stack picture")
}

/// What a chain makes of `area` of the camera picture on the CPU, planar f32 (all three planes).
fn cpu_chain(chain: &ColorChain, area: Area, films: &dyn RasterPlanes) -> Vec<f32> {
    let cam = camera(area);
    let places = Places {
        origin: (area.x, area.y),
    };
    let frame = ColorFrame::planar(area.w, area.h, &cam.data)
        .expect("frame")
        .with_photo(&places, photo_frame())
        .with_rasters(films);
    let mut out = vec![0.0f32; 3 * area.w * area.h];
    chain
        .render_logc4_f32(&frame, Layout::Interleaved, &mut out, 1)
        .expect("cpu chain");
    let n = area.w * area.h;
    let mut planar = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            planar[c * n + i] = out[i * 3 + c];
        }
    }
    planar
}

struct Places {
    origin: (i64, i64),
}

impl PhotoMap for Places {
    fn position(&self, x: usize, y: usize) -> [f64; 2] {
        [
            ((self.origin.0 + x as i64) as f64 + 0.5) / FRAME.0 as f64,
            ((self.origin.1 + y as i64) as f64 + 0.5) / FRAME.0 as f64,
        ]
    }
}

fn photo_frame() -> PhotoFrame {
    PhotoFrame::new(FRAME.0 as f64, FRAME.1 as f64, 1.0 / FRAME.0 as f64)
}

struct NoFilms;

impl RasterPlanes for NoFilms {
    fn value(&self, _plane: usize, _x: usize, _y: usize) -> f64 {
        0.0
    }
}

/// The twin's picture over `made`: the CPU's prefix and mask chains over `read`, then the twin passes with a cell kernel.
fn twin_picture(
    prefix: &ColorChain,
    mask: &ColorChain,
    level: u32,
    respect: bool,
    (read, made): (Area, Area),
) -> Vec<f64> {
    let frame = Area::whole(FRAME.0, FRAME.1);
    let pic = cpu_chain(prefix, read, &NoFilms);
    let m = cpu_chain(mask, read, &NoFilms);
    let n = read.w * read.h;
    let m = f64s(&m[..n]);
    let base = twin::prep(&f64s(&pic), &m, read.w, read.h, 0.0, respect);
    let g0 = CellGrid::of(read, frame, 0).expect("grid");
    let levels = twin::pyramid(&base, g0, frame, level);
    let (grid, planes) = &levels[level as usize];
    let sums = twin::upsample(planes, grid, made, true);
    twin::blend(&f64s(&pic), &m, read, &sums, made)
}

const REACH: usize = 20;

#[test]
fn a_blur_node_in_a_stack_is_the_twin_and_the_colour_pass_takes_its_picture_as_it_is() {
    let (whole, prefix, mask_chain) = chains_of();
    let need = Area {
        x: 70,
        y: 40,
        w: 131,
        h: 97,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let stack = stack_of(&prefix, &mask_chain, Arc::new(Cells(2)), REACH);
        let (area, got) = stack_picture(&dev.gpu, &stack, need);
        assert_eq!(area, need);
        let read = stack.read_area(need);
        let want = twin_picture(&prefix, &mask_chain, 2, true, (read, need));
        let d = abs_diff(&got, &want);
        println!("the node's picture in a stack: largest difference {d:.2e} in LogC4");
        assert!(d < GATE, "{d}");
        // The test is not vacuous: the node's mask takes values between 0 and 1 and the picture moves from the input where it
        // does (the cell kernel at level 2 is a visible blur of the scene's edges).
        let input = cpu_chain(&prefix, need, &NoFilms);
        let moved = abs_diff(&got, &f64s(&input));
        let mm = cpu_chain(&mask_chain, need, &NoFilms);
        let n = need.w * need.h;
        let partial = mm[..n].iter().filter(|&&m| m > 0.01 && m < 0.99).count();
        println!(
            "the picture differs from its input by up to {moved:.3} in LogC4; {partial} of {n} pixels have a partial mask"
        );
        assert!(moved > 0.01 && partial > n / 20);

        // The whole chain at C10 on the device against the CPU's chain given the twin's picture as the node's finished
        // picture (`FilmStep::blur`: where `m > 0` the picture as it is, where `m = 0` the input).
        let want32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        let pics = [(2u32, need, want32)];
        let planes = FilmPictures {
            origin: (need.x, need.y),
            done: &pics,
        };
        let cpu = cpu_chain(&whole, need, &planes);
        let (stack, chain) = (stack.clone(), Arc::clone(&whole));
        let wide_area = stack.read_area(need);
        let (wide_img, tile_img) = (camera(wide_area), camera(need));
        let dev_out: Vec<f32> = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "blur chain");
                let wide = GpuImage::upload_rgb(ctx, "wide", &wide_img)?;
                let pictures =
                    stack.encode(ctx, &mut passes, &wide, wide_area, need, None, None)?;
                let src = GpuImage::upload_rgb(ctx, "tile", &tile_img)?;
                let prepared =
                    PreparedChain::new(ctx, &chain)?.with_film(pictures, (need.x, need.y));
                let out = GpuImage::alloc(
                    ctx,
                    "out",
                    need.w,
                    need.h,
                    3,
                    Placement::whole(need.w, need.h),
                )?;
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    GRect::whole(need.w, need.h),
                    Stop::C10,
                    &out,
                    (0, 0),
                )?;
                let r = out.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                let _ = (wide, src);
                out.take(&done, r)
            })
            .expect("chain");
        let d = abs_diff(&dev_out, &f64s(&cpu));
        println!("the chain at C10: largest difference {d:.2e} in LogC4");
        assert!(d < GATE, "{d}");
    }
    assert!(seen || gpus().is_empty());
}

/// The pictures made so far, as the chain's Film planes (absolute pixels).
struct FilmPictures<'a> {
    origin: (i64, i64),
    done: &'a [(u32, Area, Vec<f32>)],
}

impl RasterPlanes for FilmPictures<'_> {
    fn value(&self, _plane: usize, _x: usize, _y: usize) -> f64 {
        0.0
    }

    fn film(&self, node: u32, x: usize, y: usize) -> Option<[f32; 3]> {
        let (_, a, data) = self.done.iter().find(|(n, ..)| *n == node)?;
        let (ax, ay) = (self.origin.0 + x as i64, self.origin.1 + y as i64);
        if ax < a.x || ay < a.y || ax >= a.x1() || ay >= a.y1() {
            return None;
        }
        let n = a.w * a.h;
        let i = (ay - a.y) as usize * a.w + (ax - a.x) as usize;
        Some([data[i], data[n + i], data[2 * n + i]])
    }
}

#[test]
fn tiles_a_repeat_and_a_clean_pool_are_the_same_bits() {
    let (_, prefix, mask_chain) = chains_of();
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        // The cell kernel at level 3 reads cells of 8 pixels: the reach holds them (REACH = 20 >= 2 * 8).
        let stack = stack_of(&prefix, &mask_chain, Arc::new(Cells(3)), REACH);
        let whole = Area::whole(FRAME.0, FRAME.1);
        let (_, all) = stack_picture(&dev.gpu, &stack, whole);
        for need in [
            Area {
                x: 0,
                y: 0,
                w: 97,
                h: 61,
            },
            Area {
                x: 133,
                y: 71,
                w: 97,
                h: 61,
            },
            Area {
                x: 203,
                y: 139,
                w: 97,
                h: 61,
            },
            Area {
                x: 41,
                y: 9,
                w: 211,
                h: 150,
            },
        ] {
            let (area, tile) = stack_picture(&dev.gpu, &stack, need);
            assert_eq!(area, need);
            let (n, wn) = (area.w * area.h, FRAME.0 * FRAME.1);
            for c in 0..3 {
                for y in 0..area.h {
                    for x in 0..area.w {
                        let g = tile[c * n + y * area.w + x];
                        let w = all[c * wn + (area.y as usize + y) * FRAME.0 + area.x as usize + x];
                        assert_eq!(g.to_bits(), w.to_bits(), "{need:?} at ({x}, {y}) plane {c}");
                    }
                }
            }
        }
        // A repeat, and a run on a device without the pool's poison.
        let (_, again) = stack_picture(&dev.gpu, &stack, whole);
        assert!(
            all.iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        if let Some(clean) = start_unpoisoned(StoragePreference::Auto) {
            let (_, other) = stack_picture(&clean, &stack, whole);
            assert!(
                all.iter()
                    .zip(&other)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_reach_grows_the_tile_the_scheduler_cuts() {
    use ym_gpu::kernels::mask_raster::fit_tile;
    let (_, prefix, mask_chain) = chains_of();
    // A node whose reach is 880 pixels (Soft at Radius 100 on a 50 MP photo): the stack's schedule grows a tile by it and
    // `fit_tile` keeps the grown window inside the design binding.
    let frame = (8600usize, 5700usize);
    let stack = FilmStack::new(
        Area::whole(frame.0, frame.1),
        vec![FilmNodeSpec {
            node: 2,
            prefix,
            work: NodeWork::Blur(Box::new(work(Some(Arc::new(Identity)), mask_chain, 880))),
            reads: None,
        }],
    );
    assert_eq!(stack.reach_of(2), Some(880));
    let schedule = stack.schedule();
    assert_eq!(schedule.total_reach(), 880);
    let tile = fit_tile(&schedule, frame, 2048);
    let mid = Area {
        x: ((frame.0 - tile) / 2) as i64,
        y: ((frame.1 - tile) / 2) as i64,
        w: tile,
        h: tile,
    };
    let grown = stack.read_area(mid);
    println!(
        "Soft at 880 px: tile {tile}, grown to {} x {}",
        grown.w, grown.h
    );
    assert_eq!((grown.w, grown.h), (tile + 2 * 880, tile + 2 * 880));
    assert!((64..=2048).contains(&tile));
    // Blur planes are at most four f32 planes of the grown window: inside what the raster pass already holds.
    assert!(
        GpuImage::buffer_bytes(grown.w, grown.h, 4)
            <= ym_gpu::kernels::mask_raster::max_binding_bytes(grown.w, grown.h)
    );
}
