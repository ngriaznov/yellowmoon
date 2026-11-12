//! The Film nodes of a plan on the device (`docs/24-film.md` section 3.9c, `docs/17-gpu.md` section 11a): the stack that
//! makes their pictures before the colour pass (`kernels::film_stack`) and the colour kernel's Film step that blends them,
//! against the CPU reference (`ym-film`'s `FilmFrame::run_tile` over the prefix chain, and `ColorChain` with the pictures
//! as `RasterPlanes::film`).
//!
//! * the pictures of one node and of two in a line, on tiles, within the bound of `gpu_film_spatial.rs` (2e-4 in LogC4);
//! * the whole chain (Main, a Film node at Strength 70, a corrector after it) at C10 within 2e-4 in LogC4 and at C13 within
//!   one 10-bit code, in the whole window and in bands of rows (a plan buffer limit that cuts the window);
//! * tiles equal the whole frame bit for bit (halos and grain in), a repeat and a run without poison are the same bits, the
//!   far corner of an 8,658 x 5,772 frame is the same for any tile, and the uploaded tables are kept (a change of exposure
//!   uploads nothing).
#![allow(clippy::needless_range_loop)]

mod common;
mod film_common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::chains;
use common::{TestGpu, gpus, start_unpoisoned};
use film_common::*;
use ym_color::display::{
    ChainParams, ColorChain, FilmStep, Frame as ColorFrame, Layout, MainNode, NodeParams, NodeSet,
    OutputTarget, Trim,
};
use ym_color::logc::logc4_decode_rgb;
use ym_color::mask::RasterPlanes;
use ym_color::mask_raster::Area;
use ym_color::plan::{self, MAIN_NODE, PictureSource, PlanGraph, PlanNode, compile};
use ym_film::Output;
use ym_film::spatial::{self as sp, FilmFrame, Frame, Plane, Rect, RunOptions, SpatialSettings};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::tiling::Rect as GRect;
use ym_gpu::device::{Gpu, StorageMode, StoragePreference};
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::color::{PreparedChain, Stop};
use ym_gpu::kernels::film::FilmImage;
use ym_gpu::kernels::film_stack::{
    FilmNodeSpec, FilmPictures, FilmStack, FilmTables, FilmWork, NodeWork,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

/// The Film nodes' pictures and the chain around them, in LogC4.
const LOGC4_GATE: f64 = 2e-4;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

// ---- the scene ---------------------------------------------------------------------------------------------

/// The camera RGB of the photo over `rect` (absolute pixels), planar: the test image of `film_common` decoded to linear
/// light, a function of the absolute place only, so every window of it agrees with every other.
fn camera(frame: (usize, usize), rect: Area) -> RgbImage {
    let aspect = frame.1 as f64 / frame.0 as f64;
    let (w, h) = (rect.w, rect.h);
    let mut data = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let v = image_at(
                ((rect.x + x as i64) as f64 + 0.5) / frame.0 as f64,
                ((rect.y + y as i64) as f64 + 0.5) / frame.0 as f64,
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

// ---- the chains --------------------------------------------------------------------------------------------

fn node(id: u32, x: f64, picture: PictureSource) -> PlanNode {
    PlanNode {
        id,
        pos: [x, 0.0],
        picture: Some(picture),
        key: None,
    }
}

/// Main, the Film nodes `films` (ids 2..) in a line, then a corrector (the next id) when `corrector`, the Output's.
fn graph(films: u32, corrector: bool) -> PlanGraph {
    let mut nodes = vec![node(MAIN_NODE, 0.0, PictureSource::Input)];
    let mut last = MAIN_NODE;
    for i in 0..films {
        let id = 2 + i;
        nodes.push(node(id, f64::from(id), PictureSource::Node(last)));
        last = id;
    }
    if corrector {
        let id = 2 + films;
        nodes.push(node(id, f64::from(id), PictureSource::Node(last)));
        last = id;
    }
    PlanGraph {
        nodes,
        output: PictureSource::Node(last),
    }
}

fn params(g: &PlanGraph, stop: plan::Stop, strengths: &[f64], corrector: bool) -> ChainParams {
    let mut p = chains::base(OutputTarget::Srgb);
    let mut others: BTreeMap<u32, NodeParams> = BTreeMap::new();
    let mut films = BTreeMap::new();
    for (i, &strength) in strengths.iter().enumerate() {
        let id = 2 + i as u32;
        others.insert(id, NodeParams::default());
        films.insert(
            id,
            FilmStep {
                strength,
                bypass: false,
                face: false,
            },
        );
    }
    if corrector {
        others.insert(
            2 + strengths.len() as u32,
            NodeParams {
                trim: Trim {
                    exposure_ev: 0.4,
                    ..Trim::default()
                },
                ..NodeParams::default()
            },
        );
    }
    p.nodes = NodeSet {
        plan: compile(g, stop).expect("plan"),
        main: MainNode::default(),
        others,
    };
    p.films = films;
    p
}

/// The chain of the render, and for each Film node the prefix chain that makes its input.
fn chains_of(strengths: &[f64], corrector: bool) -> (Arc<ColorChain>, Vec<(u32, Arc<ColorChain>)>) {
    let g = graph(strengths.len() as u32, corrector);
    let whole =
        ColorChain::new(&params(&g, plan::Stop::Output, strengths, corrector)).expect("chain");
    let prefixes = (0..strengths.len() as u32)
        .map(|i| {
            let id = 2 + i;
            let mut p = params(&g, plan::Stop::Picture(id), strengths, corrector);
            p.look = None;
            p.films.get_mut(&id).expect("film").bypass = true;
            (id, Arc::new(ColorChain::new(&p).expect("prefix")))
        })
        .collect();
    (Arc::new(whole), prefixes)
}

fn stack_of(
    films: &[(&str, Output, SpatialSettings)],
    prefixes: &[(u32, Arc<ColorChain>)],
    frame: Frame,
) -> (FilmStack, Vec<FilmFrame>) {
    let frames: Vec<FilmFrame> = films.iter().map(|(id, o, s)| film(id, *o, *s)).collect();
    let steps = frames
        .iter()
        .zip(prefixes)
        .map(|(f, (id, prefix))| {
            let data = colour_data(f.baked());
            let scalars = data.scalars.clone();
            FilmNodeSpec {
                node: *id,
                prefix: Arc::clone(prefix),
                work: NodeWork::Film(Box::new(FilmWork {
                    plan: Arc::new(plan_data(f, frame)),
                    scalars,
                    tables: FilmTables {
                        key: vec![Arc::as_ptr(f.baked().t3b()) as usize, *id as usize],
                        keep: vec![
                            Arc::clone(f.baked().t3b()) as Arc<dyn std::any::Any + Send + Sync>
                        ],
                        build: Arc::new(move || data.clone()),
                    },
                })),
                reads: None,
            }
        })
        .collect();
    (
        FilmStack::new(Area::whole(frame.width, frame.height), steps),
        frames,
    )
}

// ---- the device --------------------------------------------------------------------------------------------

/// The pictures the stack makes over `need`, read back: per node the rectangle and planar LogC4.
fn device_pictures(
    gpu: &Gpu,
    stack: &FilmStack,
    frame: (usize, usize),
    need: Area,
) -> Vec<(u32, Area, Vec<f32>)> {
    let stack = stack.clone();
    let wide_area = stack.read_area(need);
    let wide_img = camera(frame, wide_area);
    let ids: Vec<u32> = stack.steps().iter().map(|s| s.node).collect();
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "film stack");
        let wide = GpuImage::upload_rgb(ctx, "wide", &wide_img)?;
        let pictures = stack.encode(ctx, &mut passes, &wide, wide_area, need, None, None)?;
        let reads: Vec<_> = ids
            .iter()
            .filter_map(|&n| pictures.get(n).map(|p| (n, p.clone())))
            .map(|(n, p)| p.img.encode_read(&mut passes).map(|r| (n, p, r)))
            .collect::<Result<_, _>>()?;
        let done = passes.submit()?.finish()?;
        let _ = wide;
        reads
            .into_iter()
            .map(|(n, p, r)| {
                let area = Area {
                    x: i64::from(p.rect.x0),
                    y: i64::from(p.rect.y0),
                    w: p.rect.width() as usize,
                    h: p.rect.height() as usize,
                };
                p.img.take(&done, r).map(|v| (n, area, v))
            })
            .collect()
    })
    .expect("pictures")
}

/// What the colour pass makes of `window` of a chain with the pictures of `stack` over `need`, at C10, as planar f32 of
/// the window; the plan buffer's limit cuts the window into bands of `band_rows` rows when given.
fn device_chain(
    gpu: &Gpu,
    stack: &FilmStack,
    chain: &Arc<ColorChain>,
    frame: (usize, usize),
    need: Area,
    stop: Stop,
    band_bytes: Option<u64>,
) -> Vec<f32> {
    let (stack, chain) = (stack.clone(), Arc::clone(chain));
    let wide_area = stack.read_area(need);
    let wide_img = camera(frame, wide_area);
    let tile_img = camera(frame, need);
    gpu.run(JobSpec::new(Priority::Viewport), move |ctx, _| {
        let mut passes = PassList::new(ctx, "film chain");
        let wide = GpuImage::upload_rgb(ctx, "wide", &wide_img)?;
        let pictures = stack.encode(ctx, &mut passes, &wide, wide_area, need, None, None)?;
        let src = GpuImage::upload_rgb(ctx, "tile", &tile_img)?;
        let mut prepared = PreparedChain::new(ctx, &chain)?.with_film(pictures, (need.x, need.y));
        if let Some(bytes) = band_bytes {
            prepared = prepared.with_pass_buffer_limit(bytes);
        }
        let out = GpuImage::alloc(
            ctx,
            "out",
            need.w,
            need.h,
            3,
            ym_gpu::cpu::Placement::whole(need.w, need.h),
        )?;
        prepared.encode_stage(
            &mut passes,
            &src,
            GRect::whole(need.w, need.h),
            stop,
            &out,
            (0, 0),
        )?;
        let read = out.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        let _ = (wide, src);
        out.take(&done, read)
    })
    .expect("chain")
}

// ---- the CPU reference -------------------------------------------------------------------------------------

/// The pictures of `frames` over `need` on the CPU: the prefix chain over what the node reads, then `run_tile`, in the
/// plan's order, as `film_pass` does.
fn cpu_pictures(
    frames: &[FilmFrame],
    prefixes: &[(u32, Arc<ColorChain>)],
    frame: Frame,
    need: Area,
) -> Vec<(u32, Area, Vec<f32>)> {
    let extent = Area::whole(frame.width, frame.height);
    let n = frames.len();
    let halo = |i: usize| frames[i].halos(frame).total();
    let mut made = vec![Area::default(); n];
    let mut read = vec![Area::default(); n];
    let mut want = need.intersect(&extent);
    for i in (0..n).rev() {
        made[i] = want;
        read[i] = want.grown(halo(i), &extent);
        want = read[i];
    }
    let mut done: Vec<(u32, Area, Vec<f32>)> = Vec::new();
    for i in 0..n {
        let a = read[i];
        let cam = camera((frame.width, frame.height), a);
        let planes = Pictures {
            origin: (a.x, a.y),
            done: &done,
        };
        let places = Places {
            origin: (a.x, a.y),
            size: (frame.width as f64, frame.width as f64),
        };
        let cframe = ColorFrame::planar(a.w, a.h, &cam.data)
            .expect("frame")
            .with_photo(&places, photo_frame())
            .with_rasters(&planes);
        let mut logc4 = vec![0.0f32; 3 * a.w * a.h];
        prefixes[i]
            .1
            .render_logc4_f32(&cframe, Layout::Interleaved, &mut logc4, 1)
            .expect("prefix");
        let input = Plane {
            rect: Rect {
                x0: a.x as usize,
                y0: a.y as usize,
                x1: a.x1() as usize,
                y1: a.y1() as usize,
            },
            data: logc4,
        };
        let m = made[i];
        let tile = Rect {
            x0: m.x as usize,
            y0: m.y as usize,
            x1: m.x1() as usize,
            y1: m.y1() as usize,
        };
        let out = frames[i]
            .run_tile(frame, tile, &input, RunOptions::default())
            .expect("run_tile");
        done.push((prefixes[i].0, m, planar_of(&out)));
    }
    done
}

/// Interleaved (`ym-film`) to planar.
fn planar_of(p: &Plane) -> Vec<f32> {
    let n = p.rect.width() * p.rect.height();
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = p.data[i * 3 + c];
        }
    }
    out
}

/// The places of a window's pixels in the photo (the Film step reads its picture only where a frame knows its places, as
/// `film_pass` gives them): the absolute pixel over the frame.
struct Places {
    origin: (i64, i64),
    size: (f64, f64),
}

impl ym_color::mask::PhotoMap for Places {
    fn position(&self, x: usize, y: usize) -> [f64; 2] {
        [
            ((self.origin.0 + x as i64) as f64 + 0.5) / self.size.0,
            ((self.origin.1 + y as i64) as f64 + 0.5) / self.size.1,
        ]
    }
}

fn photo_frame() -> ym_color::mask::PhotoFrame {
    ym_color::mask::PhotoFrame::new(
        FRAME.width as f64,
        FRAME.height as f64,
        1.0 / FRAME.width as f64,
    )
}

/// The pictures made so far, as the chain's Film planes (absolute pixels).
struct Pictures<'a> {
    origin: (i64, i64),
    done: &'a [(u32, Area, Vec<f32>)],
}

impl RasterPlanes for Pictures<'_> {
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

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

// ---- the tests ---------------------------------------------------------------------------------------------

fn settings() -> SpatialSettings {
    SpatialSettings {
        grain_amount: 1.5,
        grain_seed: 3,
        halation_amount: 2.0,
        ..SpatialSettings::default()
    }
}

const FRAME: Frame = Frame {
    width: 640,
    height: 427,
};

fn two() -> Vec<(&'static str, Output, SpatialSettings)> {
    vec![
        ("kodak_portra_400", Output::Srgb, settings()),
        (
            "fujifilm_velvia_100",
            Output::Srgb,
            SpatialSettings {
                grain_seed: 9,
                ..settings()
            },
        ),
    ]
}

#[test]
fn the_pictures_of_one_node_and_of_two_in_a_line_are_the_cpus() {
    let need = Area {
        x: 150,
        y: 90,
        w: 301,
        h: 200,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for n in [1usize, 2] {
            let films = &two()[..n];
            let (_, prefixes) = chains_of(&vec![1.0; n], false);
            let (stack, frames) = stack_of(films, &prefixes, FRAME);
            let got = device_pictures(&dev.gpu, &stack, (FRAME.width, FRAME.height), need);
            let want = cpu_pictures(&frames, &prefixes, FRAME, need);
            assert_eq!(got.len(), n);
            for ((gn, ga, gv), (wn, wa, wv)) in got.iter().zip(&want) {
                assert_eq!((gn, ga), (wn, wa));
                let d = max_diff(gv, wv);
                println!(
                    "{n} node(s), node {gn}: picture {ga:?}, largest difference {d:.2e} in LogC4"
                );
                assert!(d < LOGC4_GATE, "node {gn}: {d}");
            }
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_chain_with_a_film_node_and_a_corrector_after_it_is_the_cpus() {
    let need = Area {
        x: 150,
        y: 90,
        w: 301,
        h: 200,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for (strengths, corrector) in [
            (vec![0.7], true),
            (vec![1.0, 0.6], true),
            (vec![0.5], false),
        ] {
            let films = &two()[..strengths.len()];
            let (chain, prefixes) = chains_of(&strengths, corrector);
            let (stack, frames) = stack_of(films, &prefixes, FRAME);
            let pics = cpu_pictures(&frames, &prefixes, FRAME, need);
            let planes = Pictures {
                origin: (need.x, need.y),
                done: &pics,
            };
            let cam = camera((FRAME.width, FRAME.height), need);
            let places = Places {
                origin: (need.x, need.y),
                size: (FRAME.width as f64, FRAME.width as f64),
            };
            let cframe = ColorFrame::planar(need.w, need.h, &cam.data)
                .expect("frame")
                .with_photo(&places, photo_frame())
                .with_rasters(&planes);
            let mut want = vec![0.0f32; 3 * need.w * need.h];
            chain
                .render_logc4_f32(&cframe, Layout::Interleaved, &mut want, 1)
                .expect("cpu chain");
            let n = need.w * need.h;
            let mut want_planar = vec![0.0f32; 3 * n];
            for i in 0..n {
                for c in 0..3 {
                    want_planar[c * n + i] = want[i * 3 + c];
                }
            }
            let frame = (FRAME.width, FRAME.height);
            let whole = device_chain(&dev.gpu, &stack, &chain, frame, need, Stop::C10, None);
            let d = max_diff(&whole, &want_planar);
            println!(
                "{strengths:?} corrector {corrector}: C10 largest difference {d:.2e} in LogC4"
            );
            assert!(d < LOGC4_GATE, "{strengths:?}: {d}");
            // The window cut into bands of rows by the plan buffer's limit: the same bits.
            let band = 4 * (3 * strengths.len() as u64 + 3) * need.w as u64 * 37;
            let banded = device_chain(&dev.gpu, &stack, &chain, frame, need, Stop::C10, Some(band));
            assert!(
                whole
                    .iter()
                    .zip(&banded)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{strengths:?}: bands differ from the whole window"
            );
        }
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn tiles_a_repeat_and_a_clean_pool_are_the_same_bits() {
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let (_, prefixes) = chains_of(&[1.0, 1.0], false);
        let (stack, _) = stack_of(&two(), &prefixes, FRAME);
        let frame = (FRAME.width, FRAME.height);
        let whole = Area::whole(FRAME.width, FRAME.height);
        let all = device_pictures(&dev.gpu, &stack, frame, whole);
        // Two tile sizes at several origins, grain and the halos of both nodes in.
        for (x, y, w, h) in [
            (0i64, 0i64, 97usize, 61usize),
            (233, 120, 97, 61),
            (300, 200, 301, 200),
            (543, 366, 97, 61),
        ] {
            let tile = Area { x, y, w, h };
            let part = device_pictures(&dev.gpu, &stack, frame, tile);
            for ((_, pa, pv), (_, wa, wv)) in part.iter().zip(&all) {
                let (pn, wn) = (pa.w * pa.h, wa.w * wa.h);
                // The tile's last node's picture is the tile; the first's is wider: compare where they overlap.
                let ov = pa.intersect(wa);
                for c in 0..3 {
                    for yy in ov.y..ov.y1() {
                        for xx in ov.x..ov.x1() {
                            let a = pv[c * pn + (yy - pa.y) as usize * pa.w + (xx - pa.x) as usize];
                            let b = wv[c * wn + (yy - wa.y) as usize * wa.w + (xx - wa.x) as usize];
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "tile {tile:?} at ({xx},{yy}) c{c}"
                            );
                        }
                    }
                }
            }
        }
        let again = device_pictures(&dev.gpu, &stack, frame, whole);
        for ((_, _, a), (_, _, b)) in all.iter().zip(&again) {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "a repeat differs"
            );
        }
        if dev.mode == StorageMode::F32
            && let Some(clean) = start_unpoisoned(StoragePreference::Auto)
        {
            let plain = device_pictures(&clean, &stack, frame, whole);
            for ((_, _, a), (_, _, b)) in all.iter().zip(&plain) {
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "a clean pool differs"
                );
            }
        }
        println!("tiles 97x61 and 301x200 at four origins, a repeat and a clean pool: bit for bit");
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_far_corner_of_an_8658_by_5772_frame_is_the_same_for_any_tile() {
    let frame = Frame {
        width: 8658,
        height: 5772,
    };
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        let (_, prefixes) = chains_of(&[1.0], false);
        let (stack, _) = stack_of(&two()[..1], &prefixes, frame);
        let f = (frame.width, frame.height);
        let small = Area {
            x: 8618,
            y: 5739,
            w: 40,
            h: 33,
        };
        let big = Area {
            x: 8447,
            y: 5622,
            w: 211,
            h: 150,
        };
        let a = device_pictures(&dev.gpu, &stack, f, small);
        let b = device_pictures(&dev.gpu, &stack, f, big);
        let (_, aa, av) = &a[0];
        let (_, ba, bv) = &b[0];
        let ov = aa.intersect(ba);
        assert_eq!(*aa, small);
        for c in 0..3 {
            for y in ov.y..ov.y1() {
                for x in ov.x..ov.x1() {
                    let p = av[c * aa.w * aa.h + (y - aa.y) as usize * aa.w + (x - aa.x) as usize];
                    let q = bv[c * ba.w * ba.h + (y - ba.y) as usize * ba.w + (x - ba.x) as usize];
                    assert_eq!(p.to_bits(), q.to_bits(), "({x},{y}) c{c}");
                }
            }
        }
        println!("far corner: bit for bit");
    }
    assert!(seen || gpus().is_empty());
}

#[test]
fn the_tables_are_uploaded_once_and_a_change_of_scalars_uploads_nothing() {
    // A device of its own: the tests of this binary run at the same time on the shared ones, and the tables are kept
    // per device.
    let Some(own) = start_unpoisoned(StoragePreference::Auto) else {
        return;
    };
    {
        let dev = &own;
        let (_, prefixes) = chains_of(&[1.0], false);
        let (stack, _) = stack_of(&two()[..1], &prefixes, FRAME);
        let f = (FRAME.width, FRAME.height);
        let need = Area {
            x: 100,
            y: 100,
            w: 64,
            h: 48,
        };
        let held = |gpu: &Gpu| {
            gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
                Ok(ym_gpu::kernels::film_stack::tables_held(ctx))
            })
            .expect("held")
        };
        assert_eq!(held(dev), 0);
        let a = device_pictures(dev, &stack, f, need);
        assert_eq!(held(dev), 1, "the tables are kept");
        // Another exposure: the scalars differ, the tables are the same ones.
        let mut other = stack.steps()[0].clone();
        let NodeWork::Film(film) = &mut other.work else {
            panic!("a Film node");
        };
        film.scalars.log10_gain += 0.3;
        let stack2 = FilmStack::new(stack.frame(), vec![other]);
        let b = device_pictures(dev, &stack2, f, need);
        assert_eq!(held(dev), 1, "a change of exposure uploads nothing");
        assert_ne!(a[0].2, b[0].2, "and the picture is another");
        // The pool lets go of them when asked.
        dev.run(JobSpec::new(Priority::Viewport), |ctx, _| {
            ym_gpu::kernels::color::release_tables(ctx);
            Ok(())
        })
        .expect("release");
        assert_eq!(held(dev), 0);
    }
}

#[test]
fn a_node_with_a_picture_missing_is_refused_not_guessed() {
    let (whole, prefixes) = chains_of(&[1.0], false);
    let (stack, _) = stack_of(&two()[..1], &prefixes, FRAME);
    let _ = stack;
    for dev in f32_devices() {
        let chain = Arc::clone(&whole);
        let err = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "refused");
                let src =
                    GpuImage::upload_rgb(ctx, "tile", &camera((64, 48), Area::whole(64, 48)))?;
                let out =
                    GpuImage::alloc(ctx, "out", 64, 48, 3, ym_gpu::cpu::Placement::whole(64, 48))?;
                // No pictures at all: the chain runs a Film node and has none for it.
                let prepared = PreparedChain::new(ctx, &chain)?;
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    GRect::whole(64, 48),
                    Stop::C10,
                    &out,
                    (0, 0),
                )?;
                Ok(())
            })
            .expect_err("no pictures");
        assert!(err.to_string().contains("no pictures"), "{err}");
        // Nothing of the photo under the window: empty pictures, zeros, and no error.
        let chain = Arc::clone(&whole);
        dev.gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let mut passes = PassList::new(ctx, "empty");
                let src =
                    GpuImage::upload_rgb(ctx, "tile", &camera((64, 48), Area::whole(64, 48)))?;
                let out =
                    GpuImage::alloc(ctx, "out", 64, 48, 3, ym_gpu::cpu::Placement::whole(64, 48))?;
                let prepared = PreparedChain::new(ctx, &chain)?
                    .with_film(Arc::new(FilmPictures::empty()), (0, 0));
                prepared.encode_stage(
                    &mut passes,
                    &src,
                    GRect::whole(64, 48),
                    Stop::C10,
                    &out,
                    (0, 0),
                )?;
                passes.submit()?.finish()?;
                Ok(())
            })
            .expect("empty pictures");
    }
    let _ = (FilmImage::alloc, sp::Frame::px_per_um);
}
