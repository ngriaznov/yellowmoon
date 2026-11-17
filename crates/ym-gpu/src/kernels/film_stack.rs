//! The Film nodes of a plan on the device (`docs/24-film.md` section 3.9c, `docs/17-gpu.md` section 11a), and the Face
//! Refinement nodes, which the colour kernel runs as Film steps too (`docs/27` S-13).
//!
//! A Film node is not per pixel, so the colour kernel cannot run it in its loop. The render makes the node's picture
//! **before the colour pass**, from the colour chain stopped at the node's input, and the loop's Film step blends that
//! picture by the node's mask times its Strength (`N_FILM` in `color.wgsl`). This module is that render, on the device:
//! for each live Film node in the plan's order
//!
//! 1. the node's *prefix* chain (the plan stopped at the node's input, no look, the node bypassed:
//!    [`FilmNodeSpec::prefix`]) colours the camera RGB the node reads, over the area the node reads ([`Stop::C10`], LogC4 /
//!    AWG4 f32);
//! 2. [`encode_film_tile`] makes the node's LogC4 over the area the chain after it needs, in absolute pixels of the photo's
//!    extent, so that a tile equals the same pixels of the whole frame bit for bit (`docs/24` section 3.9a);
//! 3. the picture is kept for the next node's prefix chain and for the final colour pass
//!    ([`PreparedChain::with_film`], which copies the part under its window into the first planes of the plan buffer).
//!
//! The areas are the CPU's (`ym-core`'s `film_pass`): `made[i]` is what the nodes after `i` read of it, `read[i]` is
//! `made[i]` grown by the node's halo, and the first node reads `need` grown by every halo, cut to the photo's extent.
//!
//! **Plain data.** `ym-gpu` takes no `ym-film` type: the caller (`ym-core`) converts the baked tables and the plan and
//! hands a [`FilmTables`] that builds the tables on demand. The uploaded tables are kept per device ([`FilmDevice`] in a
//! small LRU, dropped by [`release_tables`](super::color::release_tables)), by a key the caller makes from the identity of
//! its baked tables, so a change of exposure, print exposure or coupler amount (the scalars) uploads nothing.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

use crate::cpu::sampler::SamplerGrid;
use ym_color::display::ColorChain;
use ym_color::mask::PhotoFrame;
use ym_color::mask_raster::{Area, RasterRow, RasterSet};
use ym_color::schedule::{Schedule, StepKind, union_area};

use super::color::{PreparedChain, Stop};
use super::face::{
    FacePlan, FaceRegion, KeptRegion, Protect, ToneSource, make_region, place_region,
};
use super::film::{
    FRect, FilmColourData, FilmDevice, FilmImage, FilmPlan, FilmScalars, encode_film_tile, halos,
};
use super::mask_raster::{PlaneKey, RasterJob, RasterPlanes, encode_raster_planes_over};
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::{Counters, PassList};

/// The pictures the Film nodes made, by node.
#[derive(Debug, Default)]
pub struct FilmPictures {
    images: Vec<(u32, Arc<FilmImage>)>,
}

impl FilmPictures {
    /// No picture: nothing of the photo is under the window (the colour pass reads zeros for every Film node, and no pixel
    /// of the window is the photo's).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// The picture of `node`.
    #[must_use]
    pub fn get(&self, node: u32) -> Option<&Arc<FilmImage>> {
        self.images.iter().find(|(n, _)| *n == node).map(|(_, i)| i)
    }

    /// The number of pictures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether there is no picture.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// How the tables of a Film node are found on the device and built when they are not there.
#[derive(Clone)]
pub struct FilmTables {
    /// What names the uploaded tables: the identity of the caller's baked tables (their addresses) and the output. The
    /// scalars are not in it.
    pub key: Vec<usize>,
    /// What keeps `key` true: the baked tables themselves, held by the cache entry so that an address cannot be reused
    /// by other tables while the entry lives.
    pub keep: Vec<Arc<dyn Any + Send + Sync>>,
    /// Builds the plain data on a miss (the 3D tables are large: they are converted only then).
    pub build: Arc<dyn Fn() -> FilmColourData + Send + Sync>,
}

impl std::fmt::Debug for FilmTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilmTables")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// What the prefix chains of the Film nodes need to make the planes of their raster mask rows.
#[derive(Debug, Clone, Copy)]
pub struct FilmRasters<'a> {
    /// The photo's extent: the planes' window sums are cut to it.
    pub frame: Area,
    /// What the planes are kept under on the device (a view's frames), or `None` to keep none (an export tile).
    pub key: Option<&'a PlaneKey>,
}

/// A Film node's work: its passes over the photo's extent and its tables.
#[derive(Debug, Clone)]
pub struct FilmWork {
    /// The passes of the node over the photo's extent.
    pub plan: Arc<FilmPlan>,
    /// The scalars of the chain (exposure, print shift, coupler amount, the output's matrix).
    pub scalars: FilmScalars,
    /// The tables.
    pub tables: FilmTables,
}

/// What makes a node's picture.
#[derive(Debug, Clone)]
pub enum NodeWork {
    /// A Film node.
    Film(Box<FilmWork>),
    /// A Face Refinement node: the retouch of the regions around the faces (`kernels::face`).
    Face(Arc<FaceNode>),
}

/// One region of a Face Refinement node's picture.
#[derive(Debug, Clone)]
pub struct FaceRegionSpec {
    /// The region, in absolute pixels of the photo's extent.
    pub area: Area,
    /// What the CPU's plan says the faces do there.
    pub plan: Arc<FacePlan>,
    /// How the blemish gain meets the picture's tone there.
    pub tone: Arc<ToneSource>,
    /// The picture the region made in this render, once a tile has made it: a later tile of the same render copies it. A fresh
    /// slot for each render ([`KeptRegion::new`]); a slot is never shared by two renders, whose inputs may differ.
    pub kept: Arc<KeptRegion>,
}

/// A Face Refinement node over a photo's extent: its regions, and the raster row of its protection plane.
#[derive(Debug, Clone)]
pub struct FaceNode {
    /// The regions, in reading order. They do not overlap.
    pub regions: Vec<FaceRegionSpec>,
    /// The protection plane's raster row (a Brush row's strokes with its Grow, Feather and Refine), when something is painted.
    pub protect: Option<RasterRow>,
    /// The reach of the protection plane's edges around a region, in pixels of the extent.
    pub protect_halo: usize,
}

impl FaceNode {
    /// The regions of the node that meet `made` (the area its picture is demanded over).
    fn meeting(&self, made: &Area) -> impl Iterator<Item = usize> + '_ {
        let made = *made;
        (0..self.regions.len()).filter(move |&i| !self.regions[i].area.intersect(&made).is_empty())
    }

    /// The regions that meet `made` and that no earlier tile of this render made: the ones a render must hold whole. A region a
    /// tile made is kept (`FaceRegionSpec::kept`) and copied by the tiles after it, which read nothing of it, so they are not
    /// grown to hold it (the CPU keeps a region the same way). The tiles of an export run one after the other, each submitted
    /// before the next is planned, so what is kept when a tile is planned is the same wherever it is asked.
    fn needed_in(&self, made: &Area) -> impl Iterator<Item = usize> + '_ {
        self.meeting(made)
            .filter(|&i| self.regions[i].kept.get().is_none())
    }
}

/// One Film or Face Refinement node of a plan, ready to run on the device.
#[derive(Debug, Clone)]
pub struct FilmNodeSpec {
    /// The node.
    pub node: u32,
    /// The colour chain stopped at the node's input, with no look and the node bypassed.
    pub prefix: Arc<ColorChain>,
    /// What makes the picture.
    pub work: NodeWork,
    /// The Film nodes whose pictures the node reads: its ancestors in the plan's graph (`ColorChain::film_reads`).
    /// `None`: every earlier node of the stack (a line).
    pub reads: Option<Vec<u32>>,
}

impl FilmNodeSpec {
    /// The pixels the node reads beyond the pixels it makes: the sum of a Film node's three halos; 0 for a Face Refinement
    /// node, which reads whole regions (`FilmStack::widen` grows what a render asks for to hold them).
    #[must_use]
    pub fn halo(&self) -> usize {
        match &self.work {
            NodeWork::Film(film) => halos(&film.plan).iter().sum(),
            NodeWork::Face(_) => 0,
        }
    }
}

/// The Film nodes of a plan over a photo's extent: what a render needs to make their pictures on the device.
#[derive(Debug, Clone)]
pub struct FilmStack {
    frame: Area,
    steps: Vec<FilmNodeSpec>,
    fast: bool,
}

impl FilmStack {
    /// The Film nodes `steps` (in the plan's order) of a photo whose extent is `frame` (absolute pixels).
    #[must_use]
    pub fn new(frame: Area, steps: Vec<FilmNodeSpec>) -> Self {
        Self {
            frame,
            steps,
            fast: false,
        }
    }

    /// The same nodes for a **view**: the prefix chains, the Film passes and the colour pass run the plain f32 arithmetic of
    /// before F-08d (`PreparedChain::fast`, `FilmDevice::fast`). The default is the compensated arithmetic of the export.
    #[must_use]
    pub fn fast(mut self, fast: bool) -> Self {
        self.fast = fast;
        self
    }

    /// Whether the nodes run the plain f32 arithmetic of a view.
    #[must_use]
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// The nodes.
    #[must_use]
    pub fn steps(&self) -> &[FilmNodeSpec] {
        &self.steps
    }

    /// The photo's extent.
    #[must_use]
    pub fn frame(&self) -> Area {
        self.frame
    }

    /// The reach of a node's picture (its halo), or `None` for a node this stack does not hold.
    #[must_use]
    pub fn reach_of(&self, node: u32) -> Option<usize> {
        self.steps
            .iter()
            .find(|s| s.node == node)
            .map(FilmNodeSpec::halo)
    }

    /// The scheduler's schedule of `chain` with this stack's nodes as its Film steps (`ym_color::schedule`): the areas of
    /// every spatial step come from it, for the CPU and the device alike.
    #[must_use]
    pub fn schedule_of(
        chain: &ColorChain,
        grid: &SamplerGrid,
        films: Option<&FilmStack>,
    ) -> Schedule {
        chain.schedule(1.0 / grid.photo_frame().pixel, &|id| {
            films.and_then(|f| f.reach_of(id))
        })
    }

    /// The schedule of the Film nodes alone (no mask planes): their made and read areas.
    #[must_use]
    pub fn schedule(&self) -> Schedule {
        let schedule = Schedule::new(self.steps.iter().map(|s| StepKind::Film.declared(s.halo())));
        if self.steps.iter().all(|s| s.reads.is_none()) {
            return schedule;
        }
        // Each node reads the pictures of its ancestors only (parallel branches do not read each other's).
        let deps = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| match &s.reads {
                Some(ids) => (0..i)
                    .filter(|&j| ids.contains(&self.steps[j].node))
                    .collect(),
                None => (0..i).collect(),
            })
            .collect();
        schedule.with_picture_deps(deps)
    }

    /// The area a render must ask for to get `need` of the pictures: `need` grown to hold, whole, every face region that the
    /// picture of a Face Refinement node is demanded over, and the reach of the protection plane around it (`docs/27` S-13;
    /// `ym-core`'s `FilmRender::widen` is the same rule on the CPU). The demand of a node is the area the nodes after it read of
    /// its picture, `need` for the last and `need` grown by a Film node's halo for one before it; a region that joins the area
    /// makes a wider demand upstream, so the growth goes on to a fixed point (at most one round per region); the protection
    /// plane's reach is added at the end and widens no demand. `need` itself when the stack has no Face Refinement node.
    #[must_use]
    pub fn widen(&self, need: Area) -> Area {
        let faces = |s: &FilmNodeSpec| match &s.work {
            NodeWork::Face(f) => Some(Arc::clone(f)),
            NodeWork::Film(_) => None,
        };
        if need.is_empty() || !self.steps.iter().any(|s| faces(s).is_some()) {
            return need;
        }
        let schedule = self.schedule();
        let mut core = need;
        let mut joined: Vec<(usize, usize)> = Vec::new();
        loop {
            let made = schedule.infer(core, &self.frame).made;
            let mut grew = false;
            for (i, s) in self.steps.iter().enumerate() {
                let Some(face) = faces(s) else {
                    continue;
                };
                for ri in face.needed_in(&made[i]) {
                    if joined.contains(&(i, ri)) {
                        continue;
                    }
                    joined.push((i, ri));
                    let next = union_area(core, face.regions[ri].area);
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
            .fold(core, |out, &(i, ri)| match faces(&self.steps[i]) {
                Some(face) => union_area(
                    out,
                    face.regions[ri].area.grown(face.protect_halo, &self.frame),
                ),
                None => out,
            })
    }

    /// The area the first node reads for `need`: `need` widened (see [`FilmStack::widen`]) and grown by every halo, cut to
    /// the photo's extent.
    #[must_use]
    pub fn read_area(&self, need: Area) -> Area {
        self.schedule().infer(self.widen(need), &self.frame).source
    }

    /// The area of the sampled picture a tile `need` of the photo needs: [`FilmStack::read_area`] for a stack, or the
    /// schedule's own source for a chain with none.
    #[must_use]
    pub fn grown_for(
        schedule: &Schedule,
        films: Option<&FilmStack>,
        frame: &Area,
        need: Area,
    ) -> Area {
        schedule
            .infer(films.map_or(need, |f| f.widen(need)), frame)
            .source
    }

    /// The pictures of the nodes over `need` (cut to the photo's extent), as passes of `passes`. `wide` is the sampled
    /// picture the first node reads (three, four or six planes; six when the chain's mask shapes need the pixels'
    /// places), `wide_area` the absolute pixels it covers (it must hold [`FilmStack::read_area`] of `need`), and `photo`
    /// the render's photo frame when `wide` has the position planes.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when `wide` does not cover what the first node reads, when a node's prefix chain has a
    /// raster mask row and `rasters` or `photo` is missing, or for a plan the kernels refuse; the pool's and the
    /// passes' errors.
    ///
    /// The raster rows before a node make their planes over the area the node reads (`read[i]`), from `wide`, which is
    /// grown by the masks' receptive field as well (`rasters.frame` is the photo's extent). A pixel's plane does not
    /// depend on the window it is made in, so these equal the CPU's planes over the widest window.
    #[allow(clippy::too_many_arguments)] // the render's own pieces: areas, picture, photo frame, rasters
    pub fn encode(
        &self,
        ctx: &GpuContext,
        passes: &mut PassList<'_>,
        wide: &GpuImage,
        wide_area: Area,
        need: Area,
        photo: Option<PhotoFrame>,
        rasters: Option<FilmRasters<'_>>,
    ) -> Result<Arc<FilmPictures>, GpuError> {
        let n = self.steps.len();
        let areas = self.schedule().infer(self.widen(need), &self.frame);
        let (made, read) = (areas.made, areas.read);
        let want = areas.source;
        if want.is_empty() {
            return Ok(Arc::new(FilmPictures::empty()));
        }
        if !wide_area.contains(&want) {
            return Err(GpuError::InvalidInput(format!(
                "film: the sampled picture {wide_area:?} does not cover the {want:?} the Film nodes read"
            )));
        }
        crate::perf::note_film(n, want.w, want.h);
        let mut done: Vec<(u32, Arc<FilmImage>)> = Vec::with_capacity(n);
        for (i, step) in self.steps.iter().enumerate() {
            let a = read[i];
            let prefix_span = crate::perf::span(crate::perf::Stage::FilmPrefix);
            let mut prefix = PreparedChain::new(ctx, &step.prefix)?.fast(self.fast);
            if let (6, Some(frame)) = (wide.planes, photo) {
                prefix = prefix.with_photo_frame(frame);
            }
            if !prefix.live_films().is_empty() {
                prefix = prefix.with_film(
                    Arc::new(FilmPictures {
                        images: done.clone(),
                    }),
                    (a.x, a.y),
                );
            }
            // The raster rows before the node: planes over the area the node reads.
            let planes = if step.prefix.uses_rasters() {
                let (Some(r), Some(frame)) = (rasters, photo.filter(|_| wide.planes == 6)) else {
                    return Err(GpuError::InvalidInput(
                        "film: a raster mask row before a Film node needs the photo frame and the six-plane picture"
                            .to_owned(),
                    ));
                };
                Some(encode_raster_planes_over(
                    passes,
                    &prefix,
                    &step.prefix,
                    wide,
                    &RasterJob {
                        frame: r.frame,
                        grown: wide_area,
                        window: a,
                        extent: a,
                        photo: frame,
                    },
                    r.key,
                )?)
            } else {
                None
            };
            if let Some(p) = planes {
                prefix = prefix.with_rasters(p);
            }
            // The node's input: LogC4 / AWG4 over the area it reads, in the planes the stage writes (a fourth one is the
            // letterbox mask; the Film kernels read the first three by the image's plane length).
            let input = GpuImage::alloc(
                ctx,
                "film input",
                a.w,
                a.h,
                wide.planes.min(4),
                Placement::whole(a.w, a.h),
            )?;
            let window = Rect::new(
                (a.x - wide_area.x) as usize,
                (a.y - wide_area.y) as usize,
                a.w,
                a.h,
            );
            prefix.encode_stage(passes, wide, window, Stop::C10, &input, (0, 0))?;
            drop(prefix_span);
            let input = FilmImage {
                img: input,
                rect: frect(a),
                shadow: None,
            };
            let out = match &step.work {
                NodeWork::Film(film) => {
                    let dev = {
                        let _span = crate::perf::span(crate::perf::Stage::FilmTables);
                        device_tables(ctx, &film.tables)?
                            .with_scalars(film.scalars.clone())
                            .fast(self.fast)
                    };
                    let _span = crate::perf::span(crate::perf::Stage::FilmPasses);
                    encode_film_tile(ctx, passes, &dev, &film.plan, &input, frect(made[i]))?
                }
                NodeWork::Face(face) => {
                    let _span = crate::perf::span(crate::perf::Stage::FilmPasses);
                    let protect = match (&face.protect, photo.filter(|_| wide.planes == 6), rasters)
                    {
                        (Some(row), Some(frame), Some(r)) => Some(protect_planes(
                            passes, &prefix, step, row, wide, wide_area, a, frame, r,
                        )?),
                        (Some(_), ..) => {
                            return Err(GpuError::InvalidInput(
                                "film: a protection plane needs the photo frame, the six-plane picture and the raster frame"
                                    .to_owned(),
                            ));
                        }
                        (None, ..) => None,
                    };
                    encode_face_node(ctx, passes, face, &input, made[i], protect.as_ref())?
                }
            };
            let FilmImage { img, .. } = input;
            let _ = passes.keep(img.buf);
            done.push((step.node, Arc::new(out)));
        }
        Ok(Arc::new(FilmPictures { images: done }))
    }
}

/// The raster planes of a Face Refinement node's protection row over the area the node reads, by the same passes as the
/// raster rows of a chain: the row alone in a chain of the node's prefix (`docs/27` S-13).
#[allow(clippy::too_many_arguments)] // the render's own pieces: passes, chains, pictures, areas
fn protect_planes(
    passes: &mut PassList<'_>,
    prefix: &PreparedChain,
    step: &FilmNodeSpec,
    row: &RasterRow,
    wide: &GpuImage,
    wide_area: Area,
    read: Area,
    photo: PhotoFrame,
    rasters: FilmRasters<'_>,
) -> Result<RasterPlanes, GpuError> {
    let mut params = step.prefix.params().clone();
    params.rasters = RasterSet {
        rows: vec![row.clone()],
    };
    let chain = ColorChain::new(&params)
        .map_err(|e| GpuError::InvalidInput(format!("film: the protection row's chain: {e}")))?;
    encode_raster_planes_over(
        passes,
        prefix,
        &chain,
        wide,
        &RasterJob {
            frame: rasters.frame,
            grown: wide_area,
            window: read,
            extent: read,
            photo,
        },
        None,
    )
}

/// The picture of a Face Refinement node over `made`: zeros, and the retouch of every region of the node that `made` meets
/// (`kernels::face`), from the node's input `input` (the prefix chain's LogC4 over the area the node reads). A region that does
/// not lie in what the node read is left out: only the margin of the area meets it, and no pixel reads its picture.
fn encode_face_node(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    face: &FaceNode,
    input: &FilmImage,
    made: Area,
    protect: Option<&RasterPlanes>,
) -> Result<FilmImage, GpuError> {
    let read = Area {
        x: i64::from(input.rect.x0),
        y: i64::from(input.rect.y0),
        w: input.rect.width() as usize,
        h: input.rect.height() as usize,
    };
    let out = FilmImage::alloc(ctx, "face picture", frect(made))?;
    passes.encoder().clear_buffer(&out.img.buf, 0, None);
    for ri in face.meeting(&made) {
        let region = &face.regions[ri];
        let r = region.area;
        // A region another tile of this render made is copied; else it is made here and kept for the tiles after.
        let picture = match region.kept.get() {
            Some(picture) => picture,
            None => {
                if !read.contains(&r) {
                    tracing::debug!(
                        ?r,
                        ?read,
                        "a face region outside what the node read is not made"
                    );
                    continue;
                }
                let protect = match protect {
                    Some(planes) => Protect::Raster {
                        planes,
                        plane: 0,
                        origin: ((r.x - read.x) as i32, (r.y - read.y) as i32),
                    },
                    None => Protect::None,
                };
                make_region(
                    ctx,
                    passes,
                    &FaceRegion {
                        plan: &region.plan,
                        tone: &region.tone,
                        protect,
                    },
                    &input.img,
                    ((r.x - read.x) as usize, (r.y - read.y) as usize),
                    Some(&region.kept),
                )?
            }
        };
        place_region(
            ctx,
            passes,
            &picture,
            &out.img,
            (r.x - made.x, r.y - made.y),
        )?;
    }
    Ok(out)
}

fn frect(a: Area) -> FRect {
    FRect {
        x0: a.x as i32,
        y0: a.y as i32,
        x1: a.x1() as i32,
        y1: a.y1() as i32,
    }
}

// ---------------------------------------------------------------------------------------------------------
// The tables on the device
// ---------------------------------------------------------------------------------------------------------

/// How many uploaded sets of tables a device keeps (a set is tens of megabytes at the finest T3b′).
const KEPT: usize = 4;

type Entry = (Vec<usize>, Vec<Arc<dyn Any + Send + Sync>>, FilmDevice);

thread_local! {
    /// The tables of the devices whose thread this is: one list per live device, the most recently used last.
    static DEVICES: RefCell<Vec<(Weak<Counters>, Vec<Entry>)>> = const { RefCell::new(Vec::new()) };
}

/// The device tables for `tables` (uploaded on a miss).
fn device_tables(ctx: &GpuContext, tables: &FilmTables) -> Result<FilmDevice, GpuError> {
    let id = Arc::as_ptr(ctx.counters_shared());
    let hit = DEVICES.with(|cell| {
        let mut devices = cell.borrow_mut();
        devices.retain(|(owner, _)| owner.strong_count() > 0);
        let list = devices.iter_mut().find(|(owner, _)| owner.as_ptr() == id)?;
        let at = list.1.iter().position(|e| e.0 == tables.key)?;
        let entry = list.1.remove(at);
        let dev = entry.2.clone();
        list.1.push(entry);
        Some(dev)
    });
    if let Some(dev) = hit {
        return Ok(dev);
    }
    let data = (tables.build)();
    let dev = FilmDevice::upload(ctx, &data)?;
    DEVICES.with(|cell| {
        let mut devices = cell.borrow_mut();
        let at = match devices.iter().position(|(owner, _)| owner.as_ptr() == id) {
            Some(at) => at,
            None => {
                devices.push((Arc::downgrade(ctx.counters_shared()), Vec::new()));
                devices.len() - 1
            }
        };
        let list = &mut devices[at].1;
        list.push((tables.key.clone(), tables.keep.clone(), dev.clone()));
        while list.len() > KEPT {
            list.remove(0);
        }
    });
    Ok(dev)
}

/// Drops the Film tables of `ctx`'s device (their buffers go back to the pool). Called with the colour kernel's own
/// tables when the pool is out of budget.
pub(super) fn release(ctx: &GpuContext) {
    let id = Arc::as_ptr(ctx.counters_shared());
    DEVICES.with(|cell| {
        cell.borrow_mut()
            .retain(|(owner, _)| owner.as_ptr() != id && owner.strong_count() > 0);
    });
}

/// How many sets of Film tables `ctx`'s device holds (for tests).
#[must_use]
pub fn tables_held(ctx: &GpuContext) -> usize {
    let id = Arc::as_ptr(ctx.counters_shared());
    DEVICES.with(|cell| {
        cell.borrow()
            .iter()
            .find(|(owner, _)| owner.as_ptr() == id)
            .map_or(0, |(_, l)| l.len())
    })
}
