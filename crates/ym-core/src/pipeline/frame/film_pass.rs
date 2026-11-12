//! The Film node's passes on the CPU reference path (`docs/24-film.md` sections 2.6, 3.9 and 3.9a, `docs/23`
//! section 5.2).
//!
//! A Film node is not per-pixel: halation and the coupler diffusion spread light, and grain needs the pixel's
//! place. The colour chain therefore runs a Film node as a *step that reads a picture the render made before the
//! colour pass* ([`RasterPlanes::film`]) and blends it by the node's mask and Strength, like a corrector's
//! result. This module makes those pictures.
//!
//! For one area `need` of the photo (a view, or a tile of an export) and every live Film node, in the plan's
//! order:
//!
//! 1. the **input** of the node is the chain with its plan stopped at the node's input and no look
//!    ([`ColorChain::render_logc4_f32`]; the node itself bypassed, so its picture is what it reads), run over
//!    `need` grown by the node's halo and by the halos of the Film nodes after it;
//! 2. [`FilmFrame::run_tile`] makes the node's picture over the area the chain after it needs;
//! 3. the picture is kept as a [`FilmPlane`]; the next node's input already sees it.
//!
//! Everything is in **absolute pixels of the photo's extent** (`docs/algorithms/06` section 5.2: the coordinates
//! of an export of the same size), so a tile equals the same pixels of the whole frame bit for bit: the grain's
//! hash and the halation's frame edge are functions of those coordinates and never of the tile.
//!
//! **The device route (F-08b).** [`FilmRender::device_stack`] hands the same steps, as plain data, to
//! `ym_gpu::kernels::film_stack::FilmStack`: the prefix chain, the node's passes over the photo's extent and its baked
//! tables (named by their identity, so the device keeps them and a change of exposure uploads nothing). The areas are the
//! same ones ([`FilmRender::read_area`], [`FilmRender::total_halo`]), and the pictures are made on the device
//! (`docs/24` section 3.9c). A device error on a frame falls back to the CPU path of this module, as the other kernels do.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use ym_color::display::{ColorChain, Frame as ColorFrame, Layout, OutputTarget};
use ym_color::mask::RasterPlanes;
use ym_color::mask_raster::{Area, CpuPlanes};
use ym_color::plan::Stop;
use ym_color::schedule::{Schedule, StepKind};
use ym_film::spatial::{FilmFrame, Frame as FilmFrameExtent, Plane, Rect as FilmRect, RunOptions};
use ym_gpu::cpu::Rect;
use ym_gpu::cpu::sampler::{GridPositions, Sampled, SamplerGrid};
use ym_gpu::kernels::film_stack::{FilmNodeSpec, FilmStack, NodeWork};

use super::face_pass::{FacePart, FaceWork, ProtectInput, union};
use super::{Job, RenderError};
use crate::film::FilmEngine;
use crate::pipeline::color::node_plan;

/// The picture one Film node made over an area, LogC4 / AWG4, interleaved f32; or the pictures a Face Refinement node made over
/// its regions (`face_pass`), in which a pixel the retouch left as it was is `NaN` and reads as no picture.
#[derive(Debug, Clone)]
pub(crate) struct FilmPlane {
    parts: Vec<Arc<FacePart>>,
}

impl FilmPlane {
    fn of_film(area: Area, data: Vec<f32>) -> Self {
        Self {
            parts: vec![Arc::new(FacePart { area, data })],
        }
    }

    fn at(&self, ax: i64, ay: i64) -> Option<[f32; 3]> {
        for part in &self.parts {
            let area = part.area;
            if ax < area.x || ay < area.y || ax >= area.x1() || ay >= area.y1() {
                continue;
            }
            let i = ((ay - area.y) as usize * area.w + (ax - area.x) as usize) * 3;
            let v = [part.data[i], part.data[i + 1], part.data[i + 2]];
            // `NaN`: the retouch left this pixel as it was.
            return (!v[0].is_nan()).then_some(v);
        }
        None
    }
}

/// The pictures of the Film nodes of a chain over an area.
#[derive(Debug, Clone, Default)]
pub(crate) struct FilmPlanes {
    nodes: BTreeMap<u32, FilmPlane>,
}

/// What a frame being coloured reads besides its camera RGB: the raster planes (over their own window, in
/// absolute pixels) and the Film pictures, with the absolute place of the frame's pixel (0, 0).
#[derive(Debug, Clone, Default)]
pub(crate) struct OwnedPlanes {
    pub rasters: Option<(CpuPlanes, Area)>,
    pub films: FilmPlanes,
    /// The absolute pixel of the frame's first pixel.
    pub origin: (i64, i64),
}

impl RasterPlanes for OwnedPlanes {
    fn value(&self, plane: usize, x: usize, y: usize) -> f64 {
        self.view().value(plane, x, y)
    }

    fn film(&self, node: u32, x: usize, y: usize) -> Option<[f32; 3]> {
        self.view().film(node, x, y)
    }
}

impl OwnedPlanes {
    fn view(&self) -> ChainPlanes<'_> {
        ChainPlanes {
            rasters: self.rasters.as_ref().map(|(p, a)| (p, *a)),
            films: &self.films,
            origin: self.origin,
        }
    }
}

/// [`OwnedPlanes`] by reference, for the chains the passes run over their own areas.
pub(crate) struct ChainPlanes<'a> {
    rasters: Option<(&'a CpuPlanes, Area)>,
    films: &'a FilmPlanes,
    origin: (i64, i64),
}

impl RasterPlanes for ChainPlanes<'_> {
    fn value(&self, plane: usize, x: usize, y: usize) -> f64 {
        let Some((planes, area)) = self.rasters else {
            return 0.0;
        };
        let (ax, ay) = (self.origin.0 + x as i64, self.origin.1 + y as i64);
        if ax < area.x || ay < area.y || ax >= area.x1() || ay >= area.y1() {
            return 0.0;
        }
        planes.value(plane, (ax - area.x) as usize, (ay - area.y) as usize)
    }

    fn film(&self, node: u32, x: usize, y: usize) -> Option<[f32; 3]> {
        let plane = self.films.nodes.get(&node)?;
        plane.at(self.origin.0 + x as i64, self.origin.1 + y as i64)
    }
}

/// The camera RGB the passes read: a sampled picture, where it is in the photo's extent, and the geometry that
/// places its pixels in the photo (for the mask shapes upstream of a Film node).
pub(crate) struct Cover<'a> {
    pub sampled: &'a Sampled,
    /// The area of the photo's extent the picture covers.
    pub area: Area,
    pub grid: Option<&'a SamplerGrid>,
    /// The grid pixel of the picture's first pixel.
    pub grid_off: (usize, usize),
}

/// What makes a node's picture.
enum StepWork {
    /// A Film node's passes.
    Film(Box<FilmFrame>),
    /// A Face Refinement node's retouch of its regions.
    Face(Box<FaceWork>),
}

struct FilmStepRun {
    id: u32,
    work: StepWork,
    halo: usize,
    prefix: Arc<ColorChain>,
    /// The Film nodes whose pictures this one reads: its ancestors in the plan's graph.
    reads: Vec<u32>,
}

/// The Film nodes of one chain, ready to make their pictures over areas of one photo extent.
pub(crate) struct FilmRender {
    frame: Area,
    steps: Vec<FilmStepRun>,
    threads: usize,
}

fn film_error(e: &impl std::fmt::Display) -> RenderError {
    RenderError::Color(e.to_string())
}

fn engine_for(job: &Job<'_>) -> Result<Arc<FilmEngine>, RenderError> {
    let root = match job.options().data_root.clone() {
        Some(r) => r,
        None => crate::data_dir::find_data_dir().map_err(|e| film_error(&e))?,
    };
    FilmEngine::shared(&root).map_err(|e| film_error(&e))
}

fn threads_of(job: &Job<'_>) -> usize {
    match job.options().threads {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        n => n,
    }
}

impl FilmRender {
    /// The Film nodes of `grade`'s chain over a photo extent `frame` (absolute pixels, normally from `(0, 0)`),
    /// for a render into `target`. `None` when the chain has no Film node that does anything.
    ///
    /// # Errors
    ///
    /// [`RenderError::Color`] when the film data is missing or damaged.
    pub(crate) fn build(
        job: &Job<'_>,
        chain: &ColorChain,
        frame: Area,
        target: OutputTarget,
    ) -> Result<Option<Self>, RenderError> {
        if !chain.has_films() {
            return Ok(None);
        }
        let _span = ym_gpu::perf::span(ym_gpu::perf::Stage::FilmBuild);
        let settings = &job.src.settings;
        let extent = FilmFrameExtent {
            width: frame.w,
            height: frame.h,
        };
        let mut engine = None;
        let mut steps = Vec::new();
        let film_reads = chain.film_reads();
        for id in chain.film_nodes() {
            let (work, halo) = if let Some(node) = settings.nodes.film(id) {
                // A bypassed node and one at Strength 0 pass their input through: nothing to make.
                if node.is_inert() {
                    continue;
                }
                if engine.is_none() {
                    engine = Some(engine_for(job)?);
                }
                let Some(engine) = engine.as_ref() else {
                    continue;
                };
                let film = engine.frame_for(node, target).map_err(|e| film_error(&e))?;
                let halo = film.halos(extent).total();
                (StepWork::Film(Box::new(film)), halo)
            } else if let Some(node) = settings.nodes.face_refinement(id) {
                // A Face Refinement node: its regions at this frame's scale. It reads nothing beyond them (`widen` grows what a
                // render asks for to the regions it meets), so its reach is 0.
                if node.is_inert() {
                    continue;
                }
                let Some(entry) = job.faces_entry() else {
                    continue;
                };
                let Some(work) = FaceWork::build(
                    &entry,
                    node,
                    frame,
                    crate::pipeline::color::protect_row(node),
                ) else {
                    continue;
                };
                (StepWork::Face(Box::new(work)), 0)
            } else {
                continue;
            };
            let mut params = chain.params().clone();
            params.nodes.plan = node_plan(&settings.nodes, Stop::Picture(id))?;
            params.look = None;
            if let Some(step) = params.films.get_mut(&id) {
                step.bypass = true;
            }
            let prefix = Arc::new(ColorChain::new(&params).map_err(|e| film_error(&e))?);
            steps.push(FilmStepRun {
                id,
                work,
                halo,
                prefix,
                reads: film_reads.get(&id).cloned().unwrap_or_default(),
            });
        }
        if steps.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            frame,
            steps,
            threads: threads_of(job),
        }))
    }

    /// The reach of a Film node's picture (its halo), or `None` for a node that is not one of this render's.
    pub(crate) fn reach_of(&self, node: u32) -> Option<usize> {
        self.steps.iter().find(|s| s.id == node).map(|s| s.halo)
    }

    /// The scheduler's schedule of the Film nodes alone (`ym_color::schedule`): their made and read areas.
    pub(crate) fn schedule(&self) -> Schedule {
        // Each node reads the pictures of its ancestors in the plan's graph, not every earlier node's.
        let deps = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (0..i)
                    .filter(|&j| s.reads.contains(&self.steps[j].id))
                    .collect()
            })
            .collect();
        Schedule::new(self.steps.iter().map(|s| StepKind::Film.declared(s.halo)))
            .with_picture_deps(deps)
    }

    /// The pixels of the photo the Film nodes read beyond the pixels they make: the sum of their halos (a node
    /// after another reads the other's picture over its own halo).
    pub(crate) fn total_halo(&self) -> usize {
        self.schedule().total_reach()
    }

    /// The Film and Face Refinement nodes as plain data for the device (`ym_gpu::kernels::film_stack`): the prefix chain, and for
    /// a Film node the passes over the photo's extent, the scalars and the tables, for a Face Refinement node its regions with
    /// the CPU's plan of each (`docs/27` S-13).
    ///
    /// # Errors
    ///
    /// [`RenderError::Mask`] when the face library refuses a region.
    pub(crate) fn device_stack(&self) -> Result<FilmStack, RenderError> {
        let _span = ym_gpu::perf::span(ym_gpu::perf::Stage::FilmBuild);
        let extent = FilmFrameExtent {
            width: self.frame.w,
            height: self.frame.h,
        };
        let mut steps = Vec::with_capacity(self.steps.len());
        for s in &self.steps {
            let work = match &s.work {
                StepWork::Film(film) => NodeWork::Film {
                    plan: Arc::new(crate::film::device_plan(film, extent)),
                    scalars: crate::film::device_scalars(film.baked()),
                    tables: crate::film::device_tables(film.baked()),
                },
                StepWork::Face(face) => NodeWork::Face(Arc::new(face.device_node()?)),
            };
            steps.push(FilmNodeSpec {
                node: s.id,
                prefix: Arc::clone(&s.prefix),
                work,
                reads: Some(s.reads.clone()),
            });
        }
        Ok(FilmStack::new(self.frame, steps))
    }

    /// The area a render must ask for to get `need` of the pictures: `need` grown to hold, whole, every face region that the
    /// picture of a Face Refinement node is demanded over, and the reach of the protection plane around it. A render that
    /// makes the pictures of an area (a view, a tile of an export) uses this area where it would use `need`.
    ///
    /// The demand of a node is the area the nodes after it read of its picture: `need` for the last, `need` grown by a Film
    /// node's halo for a node before it (`docs/27` S-11b2: a halo that cut a region left that part unretouched). A region
    /// that joins the area makes a wider demand upstream, so the growth goes on to a fixed point (at most one round per
    /// region); the protection plane's reach is added at the end and widens no demand.
    pub(crate) fn widen(&self, need: Area) -> Area {
        let has_face = self
            .steps
            .iter()
            .any(|s| matches!(s.work, StepWork::Face(_)));
        if !has_face || need.is_empty() {
            return need;
        }
        let schedule = self.schedule();
        let mut core = need;
        let mut joined: Vec<(usize, usize)> = Vec::new();
        loop {
            let made = schedule.infer(core, &self.frame).made;
            let mut grew = false;
            for (i, s) in self.steps.iter().enumerate() {
                let StepWork::Face(face) = &s.work else {
                    continue;
                };
                for ri in face.needed_in(&made[i]) {
                    if joined.contains(&(i, ri)) {
                        continue;
                    }
                    joined.push((i, ri));
                    let next = union(core, face.region(ri));
                    grew |= next != core;
                    core = next;
                }
            }
            if !grew {
                break;
            }
        }
        joined
            .iter()
            .fold(core, |out, &(i, ri)| match &self.steps[i].work {
                StepWork::Face(face) => union(out, face.region_with_protect(ri)),
                StepWork::Film(_) => out,
            })
    }

    /// The area the first Film node reads for `need`: `need` grown by every halo, cut to the photo's extent.
    pub(crate) fn read_area(&self, need: Area) -> Area {
        self.schedule().infer(self.widen(need), &self.frame).source
    }

    /// The pictures of the Film nodes over `need` (cut to the photo's extent). `rasters` are the raster planes
    /// of the chain over their window; `cover` must hold [`FilmRender::read_area`] of `need`.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] and the errors of the chains and of the passes.
    pub(crate) fn planes(
        &self,
        job: &mut Job<'_>,
        need: Area,
        cover: &Cover<'_>,
        rasters: Option<(&CpuPlanes, Area)>,
    ) -> Result<FilmPlanes, RenderError> {
        // What the pixels asked for demand of each node's picture, before the area grows to hold whole regions.
        let demanded = self.schedule().infer(need, &self.frame).made;
        let need = self.widen(need);
        let areas = self.schedule().infer(need, &self.frame);
        let (made, read, want) = (areas.made, areas.read, areas.source);
        if want.is_empty() {
            return Ok(FilmPlanes::default());
        }
        if !cover.area.contains(&want) {
            return Err(RenderError::Color(
                "The picture does not cover what the Film node reads.".to_owned(),
            ));
        }
        let mut done = FilmPlanes::default();
        let extent = FilmFrameExtent {
            width: self.frame.w,
            height: self.frame.h,
        };
        for (i, step) in self.steps.iter().enumerate() {
            job.check()?;
            if let StepWork::Face(face) = &step.work {
                let parts = self.face_parts(
                    job,
                    step,
                    face,
                    (made[i], demanded[i]),
                    cover,
                    rasters,
                    &done,
                )?;
                done.nodes.insert(step.id, FilmPlane { parts });
                continue;
            }
            let StepWork::Film(film) = &step.work else {
                continue;
            };
            let a = read[i];
            let (w, h) = (a.w, a.h);
            let (ox, oy) = ((a.x - cover.area.x) as usize, (a.y - cover.area.y) as usize);
            let sub = cover.sampled.img.extract(Rect::new(ox, oy, w, h));
            let positions = cover
                .grid
                .filter(|_| step.prefix.uses_photo_position())
                .map(|g| {
                    (
                        GridPositions::new(g, (cover.grid_off.0 + ox, cover.grid_off.1 + oy)),
                        g.photo_frame(),
                    )
                });
            let view = ChainPlanes {
                rasters,
                films: &done,
                origin: (a.x, a.y),
            };
            let mut frame = ColorFrame::planar(w, h, &sub.data).map_err(|e| film_error(&e))?;
            if let Some((map, photo)) = &positions {
                frame = frame.with_photo(map, *photo);
            }
            let frame = frame.with_rasters(&view);
            let mut logc4 = vec![0.0f32; 3 * w * h];
            step.prefix
                .render_logc4_f32(&frame, Layout::Interleaved, &mut logc4, self.threads)
                .map_err(|e| film_error(&e))?;
            job.check()?;
            let input = Plane {
                rect: FilmRect {
                    x0: a.x as usize,
                    y0: a.y as usize,
                    x1: a.x1() as usize,
                    y1: a.y1() as usize,
                },
                data: logc4,
            };
            let m = made[i];
            let tile = FilmRect {
                x0: m.x as usize,
                y0: m.y as usize,
                x1: m.x1() as usize,
                y1: m.y1() as usize,
            };
            let out = film
                .run_tile(
                    extent,
                    tile,
                    &input,
                    RunOptions {
                        threads: self.threads,
                        poison: false,
                    },
                )
                .map_err(|e| film_error(&e))?;
            done.nodes.insert(step.id, FilmPlane::of_film(m, out.data));
        }
        Ok(done)
    }

    /// The pictures of a Face Refinement node's regions that meet `made`: for each, the prefix chain over the region (the
    /// node's input), then the retouch of it (`face_pass`). A region met by an earlier tile of the same export is not made again.
    #[allow(clippy::too_many_arguments)]
    fn face_parts(
        &self,
        job: &mut Job<'_>,
        step: &FilmStepRun,
        face: &FaceWork,
        (made, demanded): (Area, Area),
        cover: &Cover<'_>,
        rasters: Option<(&CpuPlanes, Area)>,
        done: &FilmPlanes,
    ) -> Result<Vec<Arc<FacePart>>, RenderError> {
        let mut parts = Vec::new();
        for ri in face.regions_in(&made) {
            if let Some(kept) = face.kept(ri) {
                parts.push(kept);
                continue;
            }
            job.check()?;
            let a = face.region(ri);
            if !cover.area.contains(&a) {
                // `FilmRender::widen` holds whole every region the pixels demanded of this node meet; this is a region that
                // only the margin of the area meets (the protection plane's reach), whose pixels no output reads.
                if a.intersect(&demanded).is_empty() {
                    tracing::debug!(
                        ?a,
                        "a face region outside what the render demands is not made"
                    );
                } else {
                    // Never expected: `widen` holds whole every region the demand meets.
                    tracing::warn!(
                        ?a,
                        ?demanded,
                        "a face region is cut by the picture the node reads; it is left unretouched"
                    );
                    job.note_face_region_cut();
                }
                continue;
            }
            let prefix_started = Instant::now();
            let (w, h) = (a.w, a.h);
            let (ox, oy) = ((a.x - cover.area.x) as usize, (a.y - cover.area.y) as usize);
            let sub = cover.sampled.img.extract(Rect::new(ox, oy, w, h));
            let positions = cover
                .grid
                .filter(|_| step.prefix.uses_photo_position())
                .map(|g| {
                    (
                        GridPositions::new(g, (cover.grid_off.0 + ox, cover.grid_off.1 + oy)),
                        g.photo_frame(),
                    )
                });
            let view = ChainPlanes {
                rasters,
                films: done,
                origin: (a.x, a.y),
            };
            let mut frame = ColorFrame::planar(w, h, &sub.data).map_err(|e| film_error(&e))?;
            if let Some((map, photo)) = &positions {
                frame = frame.with_photo(map, *photo);
            }
            let frame = frame.with_rasters(&view);
            let mut logc4 = vec![0.0f32; 3 * w * h];
            step.prefix
                .render_logc4_f32(&frame, Layout::Interleaved, &mut logc4, self.threads)
                .map_err(|e| film_error(&e))?;
            job.check()?;
            tracing::debug!(
                w,
                h,
                prefix_ms = prefix_started.elapsed().as_secs_f64() * 1000.0,
                "the prefix chain over a face region"
            );
            let guide = |rgb: [f64; 3]| step.prefix.guide_pixel(rgb);
            let protect = cover.grid.and_then(|grid| {
                face.protect_mask(
                    ri,
                    &ProtectInput {
                        grid,
                        grid_off: cover.grid_off,
                        area: cover.area,
                        planar: &cover.sampled.img.data,
                        outside: cover.sampled.outside.as_deref(),
                        guide: &guide,
                    },
                )
            });
            parts.push(face.part(ri, &logc4, protect.as_ref())?);
        }
        Ok(parts)
    }
}
