//! The Film nodes of a plan on the device (`docs/24-film.md` section 3.9c, `docs/17-gpu.md` section 11a).
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

use ym_color::display::ColorChain;
use ym_color::mask::PhotoFrame;
use ym_color::mask_raster::Area;

use super::color::{PreparedChain, Stop};
use super::film::{
    FRect, FilmColourData, FilmDevice, FilmImage, FilmPlan, FilmScalars, encode_film_tile, halos,
};
use super::mask_raster::{PlaneKey, RasterJob, encode_raster_planes_over};
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

/// One Film node of a plan, ready to run on the device.
#[derive(Debug, Clone)]
pub struct FilmNodeSpec {
    /// The node.
    pub node: u32,
    /// The colour chain stopped at the node's input, with no look and the node bypassed.
    pub prefix: Arc<ColorChain>,
    /// The passes of the node over the photo's extent.
    pub plan: Arc<FilmPlan>,
    /// The scalars of the chain (exposure, print shift, coupler amount, the output's matrix).
    pub scalars: FilmScalars,
    /// The tables.
    pub tables: FilmTables,
}

impl FilmNodeSpec {
    /// The pixels the node reads beyond the pixels it makes: the sum of its three halos.
    #[must_use]
    pub fn halo(&self) -> usize {
        halos(&self.plan).iter().sum()
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

    /// The pixels the nodes read beyond the pixels they make: the sum of their halos (a node after another reads the
    /// other's picture over its own halo).
    #[must_use]
    pub fn total_halo(&self) -> usize {
        self.steps.iter().map(FilmNodeSpec::halo).sum()
    }

    /// The area the first node reads for `need`: `need` grown by every halo, cut to the photo's extent.
    #[must_use]
    pub fn read_area(&self, need: Area) -> Area {
        need.intersect(&self.frame)
            .grown(self.total_halo(), &self.frame)
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
        let mut made = vec![Area::default(); n];
        let mut read = vec![Area::default(); n];
        let mut want = need.intersect(&self.frame);
        for i in (0..n).rev() {
            made[i] = want;
            read[i] = want.grown(self.steps[i].halo(), &self.frame);
            want = read[i];
        }
        if want.is_empty() {
            return Ok(Arc::new(FilmPictures::empty()));
        }
        if !wide_area.contains(&read[0]) {
            return Err(GpuError::InvalidInput(format!(
                "film: the sampled picture {wide_area:?} does not cover the {:?} the first Film node reads",
                read[0]
            )));
        }
        crate::perf::note_film(n, read[0].w, read[0].h);
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
            let dev = {
                let _span = crate::perf::span(crate::perf::Stage::FilmTables);
                device_tables(ctx, &step.tables)?
                    .with_scalars(step.scalars.clone())
                    .fast(self.fast)
            };
            let out = {
                let _span = crate::perf::span(crate::perf::Stage::FilmPasses);
                encode_film_tile(ctx, passes, &dev, &step.plan, &input, frect(made[i]))?
            };
            let FilmImage { img, .. } = input;
            let _ = passes.keep(img.buf);
            done.push((step.node, Arc::new(out)));
        }
        Ok(Arc::new(FilmPictures { images: done }))
    }
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
