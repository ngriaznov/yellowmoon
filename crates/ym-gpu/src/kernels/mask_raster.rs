//! The raster masks' pass on the device (`docs/23-nodes.md` sections 4.3 and 5.5, `docs/algorithms/06-masks.md`
//! section 5): for each raster row of a chain, the resample of the row's photo-frame plane `S0` to the output grid,
//! Grow, Feather, the guided filter of Refine edges and the quantising to the u16 planes that the colour pass reads
//! (binding 7 of `color.wgsl`). The CPU reference is `ym_color::mask_raster` in `f64`; this is the same arithmetic
//! in `f32`, one dispatch of `mask_raster.wgsl` for each step (`parity: tests/gpu_raster_masks.rs`).
//!
//! # What goes in
//!
//! The pass works on a **window of the output grid** ([`RasterJob`]): `grown`, the tile grown by the receptive
//! field of the planes ([`RasterSet::halo`]) and cut to the frame; `window`, the tile, the part whose planes are
//! returned; `frame`, the whole output image, which every window sum is cut to. It reads the sampler's image of
//! `grown` (six planes: the colour, the letterbox mask and the pixels' places in the photo, `encode_sample_at`),
//! and, when a row refines its edges, the Input's luma over `grown` ([`Stop::Guide`], which the colour kernel
//! writes from the same image), and a source plane `S0` for every row (u16 codes on the device, uploaded once and
//! kept in a small cache keyed by what made the plane).
//!
//! # What comes out
//!
//! [`RasterPlanes`]: one u16 plane for every row, over `window`, two pixels to a word, in one buffer. The colour
//! pass binds it ([`PreparedChain::with_rasters`]) and reads a raster row's `s` as the code over 65535 at its
//! pixel. A tile of a frame equals the whole frame bit for bit (every window fold is a blocked fold with absolute
//! block origins, cut to the frame; `docs/17` D4 and D5).
//!
//! # Memory
//!
//! The pass keeps its f32 planes in one scratch buffer: seven planes of `grown` for the guided filter at full
//! size (the mask, a temporary, the two planes that a fold's scans write, and three of the means) and, for the
//! subsampled form, four of `grown` and eight of the coarse grid. [`max_binding_bytes`] says how big that is; a tile whose grown window passes the
//! design binding is made smaller by the caller (tiles are exact, so the size does not change a pixel).

use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use ym_color::display::ColorChain;
use ym_color::mask::PhotoFrame;
use ym_color::mask_raster::{Area, RasterSet, RasterSteps, SourcePlane};

use super::color::{PreparedChain, Stop};
use super::shader::KernelId;
use super::util::{STORAGE_OUT, even, storage_buffer};
use crate::cpu::{Placement, tiling::Rect};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::perf::{self, Stage};
use crate::pool::PooledBuffer;
use crate::stream::{Counters, PassList};
use crate::tiles::DESIGN_BINDING_BYTES;

/// Planes of the pass's f32 scratch at full size for a row that refines its edges.
const FULL_SLOTS: usize = 7;
/// Planes of the coarse grid of the subsampled guided filter.
const COARSE_SLOTS: usize = 8;

/// The uniform block of `mask_raster.wgsl` (`RU`), 240 bytes: fifteen `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct RasterU {
    op: [u32; 4],
    pa: [i32; 4],
    pb: [i32; 4],
    pc: [i32; 4],
    pd: [i32; 4],
    org_a: [i32; 4],
    org_d: [i32; 4],
    frame: [i32; 4],
    win: [i32; 4],
    ins: [i32; 4],
    outp: [u32; 4],
    samp: [u32; 4],
    s0: [u32; 4],
    f: [f32; 4],
    cg: [i32; 4],
}

/// The ops of `mask_raster.wgsl`.
#[derive(Clone, Copy)]
enum Op {
    Resample = 0,
    Scan = 1,
    Mul = 2,
    Coef = 3,
    Block = 4,
    Up = 5,
    Final = 6,
    Quant = 7,
    Comb = 8,
}

/// How a window fold combines (`mode` of the SCAN and COMB ops).
#[derive(Clone, Copy)]
enum Fold {
    Mean = 0,
    Max = 1,
    Min = 2,
}

/// A plane of the scratch buffer: where it starts (in elements), its size and its row stride.
#[derive(Clone, Copy)]
struct Slot {
    off: i32,
    w: i32,
    h: i32,
}

impl Slot {
    fn vec(self) -> [i32; 4] {
        [self.off, self.w, self.h, self.w]
    }
}

/// The windows of a raster pass, in absolute pixels of the output grid.
#[derive(Debug, Clone, Copy)]
pub struct RasterJob {
    /// The whole output image: window sums are cut to it.
    pub frame: Area,
    /// The tile grown by the receptive field and cut to the frame: the area the sampled image covers.
    pub grown: Area,
    /// The tile: the part of `grown` the planes are made for.
    pub window: Area,
    /// The area the returned planes cover: `window`, or an area around it whose pixels outside `window` have no
    /// plane and read 0 (a view whose frame has paper in it: the photo ends inside the picture the colour pass
    /// makes). `window` must lie inside it.
    pub extent: Area,
    /// The photo's aspect and the size of one output pixel in units of its long edge.
    pub photo: PhotoFrame,
}

/// The u16 raster planes of one colour window on the device: what the colour kernel binds at 7.
#[derive(Clone)]
pub struct RasterPlanes {
    pub(crate) buf: Arc<PooledBuffer>,
    /// The width in pixels of the part of the planes a colour pass reads.
    pub width: usize,
    /// The height in pixels of that part.
    pub height: usize,
    /// The row stride of a plane in elements (the width rounded up to an even number).
    pub stride: usize,
    /// The number of planes.
    pub planes: usize,
    /// The row of the planes where that part starts (0 for the whole planes; see [`RasterPlanes::rows`]).
    pub(crate) row0: usize,
    /// The rows of the planes the buffer holds.
    pub(crate) rows_held: usize,
}

impl std::fmt::Debug for RasterPlanes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RasterPlanes")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("planes", &self.planes)
            .finish_non_exhaustive()
    }
}

impl RasterPlanes {
    /// Words in one plane (two u16 codes to a word).
    #[must_use]
    pub fn plane_words(&self) -> usize {
        self.stride * self.rows_held / 2
    }

    /// The same planes seen through rows `first .. first + count` of the window: what a colour pass over
    /// that part of the window reads (an export tile that two slabs share is coloured in two windows).
    ///
    /// # Panics
    ///
    /// When the rows are not inside the planes.
    #[must_use]
    pub fn rows(&self, first: usize, count: usize) -> Self {
        assert!(
            self.row0 + first + count <= self.rows_held,
            "rows {first}..{} of planes of {} rows",
            first + count,
            self.height
        );
        Self {
            buf: Arc::clone(&self.buf),
            height: count,
            row0: self.row0 + first,
            ..*self
        }
    }

    /// The buffer that holds the planes (`planes * plane_words` words), for reading them back.
    #[must_use]
    pub fn buffer(&self) -> &Arc<PooledBuffer> {
        &self.buf
    }

    /// Bytes of the planes' buffer.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        (self.planes * self.plane_words() * 4) as u64
    }

    /// The row of the planes where the part a colour pass reads starts.
    #[must_use]
    pub fn first_row(&self) -> usize {
        self.row0
    }
}

/// The window an export tile of `w x h` pixels at `off` of a `frame`-sized output is sampled, guided and
/// coloured over: the tile's area as the scheduler infers it (`ym_color::schedule`), which is the tile grown by every
/// spatial step's reach (the raster planes' and the Film nodes') and cut to the frame. For a chain with no spatial step it
/// is the tile.
#[must_use]
pub fn grown_tile(
    schedule: &ym_color::schedule::Schedule,
    frame: (usize, usize),
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Area {
    let tile = Area {
        x: off.0 as i64,
        y: off.1 as i64,
        w,
        h,
    };
    schedule
        .infer(tile, &Area::whole(frame.0, frame.1))
        .source
}

/// The largest storage binding a raster pass over a `grown` window makes: the f32 scratch (seven planes at full
/// size) or the sampler's six-plane image, whichever is larger (K4 of `docs/17-gpu.md`).
#[must_use]
pub fn max_binding_bytes(grown_w: usize, grown_h: usize) -> u64 {
    let pixels = even(grown_w) as u64 * grown_h as u64;
    (FULL_SLOTS as u64 * 4).max(24) * pixels
}

/// The largest tile side (pixels) whose grown window fits one design binding for a halo of `halo` pixels.
#[must_use]
pub fn largest_tile_for(halo: usize, wanted: usize) -> usize {
    let mut tile = wanted;
    while tile > 64 && max_binding_bytes(tile + 2 * halo, tile + 2 * halo) > DESIGN_BINDING_BYTES {
        tile = (tile * 3 / 4).max(64);
    }
    tile
}

// ---------------------------------------------------------------------------------------------
// The source planes on the device
// ---------------------------------------------------------------------------------------------

/// A source plane held on the device.
struct SourceEntry {
    key: (u64, usize, usize),
    buf: Arc<PooledBuffer>,
    bytes: u64,
}

thread_local! {
    /// The uploaded source planes of the devices whose thread this is, most recently used last.
    static SOURCES: RefCell<Vec<(Weak<Counters>, Vec<SourceEntry>)>> = const { RefCell::new(Vec::new()) };
}

/// How many source planes a device keeps at least (8 MiB each at 2048 px); a plan with more live raster rows
/// keeps one for each, within [`kept_bytes_cap`].
const SOURCES_KEPT: usize = 8;

/// The most bytes of kept planes (sources, and the raster planes of [`PlaneKey`]) a device holds: 128 MiB, and an
/// eighth of the pool's budget (`docs/17-gpu.md` section 6: the resident caches take a fraction of it).
fn kept_bytes_cap(ctx: &GpuContext) -> u64 {
    (128u64 << 20).min(ctx.pool().budget() / 8)
}

/// The device copy of `plane` (u16 codes, two to a word), uploaded the first time and kept.
fn source_buffer(
    ctx: &GpuContext,
    plane: &SourcePlane,
    live: usize,
) -> Result<Arc<PooledBuffer>, GpuError> {
    let key = (plane.key, plane.width, plane.height);
    SOURCES.with(|cell| {
        let mut slots = cell.borrow_mut();
        slots.retain(|(owner, _)| owner.strong_count() > 0);
        let id = Arc::as_ptr(ctx.counters_shared());
        let at = match slots.iter().position(|(owner, _)| owner.as_ptr() == id) {
            Some(i) => i,
            None => {
                slots.push((Arc::downgrade(ctx.counters_shared()), Vec::new()));
                slots.len() - 1
            }
        };
        let held = &mut slots[at].1;
        if let Some(i) = held.iter().position(|e| e.key == key) {
            let entry = held.remove(i);
            let buf = Arc::clone(&entry.buf);
            held.push(entry);
            return Ok(buf);
        }
        let bytes: Vec<u8> = plane.data.iter().flat_map(|c| c.to_le_bytes()).collect();
        let buf = Arc::new(storage_buffer(ctx, "raster source plane", &bytes)?);
        held.push(SourceEntry {
            key,
            buf: Arc::clone(&buf),
            bytes: bytes.len() as u64,
        });
        // At least eight planes, and one for every live raster row; beyond eight, within the byte cap.
        let keep = SOURCES_KEPT.max(live);
        let cap = kept_bytes_cap(ctx);
        let mut total: u64 = held.iter().map(|e| e.bytes).sum();
        while held.len() > keep || (total > cap && held.len() > SOURCES_KEPT) {
            total -= held.remove(0).bytes;
        }
        Ok(buf)
    })
}

/// Drops the source planes this device holds, and the raster planes it keeps ([`release_planes`]); their buffers go
/// back to the pool.
pub fn release_sources(ctx: &GpuContext) {
    SOURCES.with(|cell| {
        let id = Arc::as_ptr(ctx.counters_shared());
        cell.borrow_mut()
            .retain(|(owner, _)| owner.as_ptr() != id && owner.strong_count() > 0);
    });
    release_planes(ctx);
}

// ---------------------------------------------------------------------------------------------
// The raster planes kept on the device
// ---------------------------------------------------------------------------------------------

/// What names the raster planes of a view's rows (`docs/23-nodes.md` section 5.6): a row's plane is the same on
/// every frame whose [`PlaneKey`] and whose row (source, Grow, Feather and Refine edges) are the same, and a slider of a
/// node's grade, an Opacity, a Mode, an Invert, the eye or Bypass is none of them: those act in the colour pass,
/// after the plane.
///
/// The caller names what the pass cannot see; the pass adds what it can (the windows, the photo's frame, the
/// size of an output pixel, and the row).
#[derive(Debug, Clone)]
pub struct PlaneKey {
    /// The photo the planes belong to; [`forget_planes_of`] drops them when it leaves the renderer.
    pub owner: u64,
    /// Where every pixel of the windows is in the photo: the raw parameters that make the pixels' places (the
    /// lens), the geometry (crop, turn, straighten), the view's map to the final image, its size and scale.
    pub place: String,
    /// What the guide of Refine edges is made of: the Input's luma of the sampled picture (the raw group's
    /// parameters and the colour group: white balance, exposure, the roll-offs). Only the rows that refine
    /// their edges carry it.
    pub guide: String,
}

/// A kept plane: one row's u16 codes over the `extent`, as the packed buffer of [`RasterPlanes`] holds them.
struct PlaneEntry {
    owner: u64,
    key: String,
    buf: Arc<PooledBuffer>,
    bytes: u64,
}

/// The planes one device keeps, most recently used last, and those a submission that has not been made yet will fill.
struct DevicePlanes {
    device: Weak<Counters>,
    held: Vec<PlaneEntry>,
    /// Planes recorded by the list with this id: served to that list only (Show mask Overlay makes the rows of
    /// its second chain from the first chain's), and kept when the list is submitted. A list that is dropped
    /// leaves its planes here, where the next list's lookup throws them away: they were never filled.
    staged: Vec<(u64, PlaneEntry)>,
}

static PLANES: Mutex<Vec<DevicePlanes>> = Mutex::new(Vec::new());

fn with_planes<R>(device: &Arc<Counters>, f: impl FnOnce(&mut DevicePlanes) -> R) -> R {
    let mut all = PLANES.lock().unwrap_or_else(PoisonError::into_inner);
    all.retain(|d| d.device.strong_count() > 0);
    let at = match all
        .iter()
        .position(|d| std::ptr::eq(d.device.as_ptr(), Arc::as_ptr(device)))
    {
        Some(i) => i,
        None => {
            all.push(DevicePlanes {
                device: Arc::downgrade(device),
                held: Vec::new(),
                staged: Vec::new(),
            });
            all.len() - 1
        }
    };
    f(&mut all[at])
}

/// The kept plane `key` of `owner`, or the one the list `list` recorded.
fn plane_lookup(ctx: &GpuContext, list: u64, owner: u64, key: &str) -> Option<Arc<PooledBuffer>> {
    with_planes(ctx.counters_shared(), |d| {
        d.staged.retain(|(id, _)| *id == list);
        if let Some((_, e)) = d
            .staged
            .iter()
            .find(|(_, e)| e.owner == owner && e.key == key)
        {
            return Some(Arc::clone(&e.buf));
        }
        let i = d
            .held
            .iter()
            .position(|e| e.owner == owner && e.key == key)?;
        let e = d.held.remove(i);
        let buf = Arc::clone(&e.buf);
        d.held.push(e);
        Some(buf)
    })
}

/// Keeps the staged planes of `list` now that it was submitted, oldest planes out beyond `cap` bytes.
fn plane_commit(device: &Weak<Counters>, list: u64, cap: u64) {
    let Some(device) = device.upgrade() else {
        return;
    };
    with_planes(&device, |d| {
        let staged = std::mem::take(&mut d.staged);
        for (id, e) in staged {
            if id != list {
                continue;
            }
            d.held.retain(|h| !(h.owner == e.owner && h.key == e.key));
            d.held.push(e);
        }
        let mut total: u64 = d.held.iter().map(|e| e.bytes).sum();
        while total > cap && !d.held.is_empty() {
            total -= d.held.remove(0).bytes;
        }
    });
}

/// Drops the raster planes this device keeps (their buffers go back to the pool). `release_tables` and
/// [`release_sources`] call it: a device that lets go of what it holds lets go of these.
pub fn release_planes(ctx: &GpuContext) {
    let id = Arc::as_ptr(ctx.counters_shared());
    let mut all = PLANES.lock().unwrap_or_else(PoisonError::into_inner);
    all.retain(|d| d.device.strong_count() > 0 && !std::ptr::eq(d.device.as_ptr(), id));
}

/// Drops the kept raster planes of the photo `owner` on every device: it left the renderer's photos.
pub fn forget_planes_of(owner: u64) {
    let mut all = PLANES.lock().unwrap_or_else(PoisonError::into_inner);
    for d in all.iter_mut() {
        d.held.retain(|e| e.owner != owner);
        d.staged.retain(|(_, e)| e.owner != owner);
    }
}

/// Drops the kept raster planes of every photo on every device.
pub fn forget_all_planes() {
    PLANES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}

/// The bytes of raster planes this device keeps (a test's and a diagnosis's number; the pool ledger counts them).
#[must_use]
pub fn kept_plane_bytes(ctx: &GpuContext) -> u64 {
    with_planes(ctx.counters_shared(), |d| {
        d.held.iter().map(|e| e.bytes).sum()
    })
}

// ---------------------------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------------------------

/// Everything the dispatches of one row share.
struct Rec<'p, 'a> {
    passes: &'p mut PassList<'a>,
    work: Arc<PooledBuffer>,
    s0: Arc<PooledBuffer>,
    samp: &'p GpuImage,
    guide: Option<&'p GpuImage>,
    out: Arc<PooledBuffer>,
    base: RasterU,
    /// The offsets (elements) of the two scratch planes that a fold's scans write: the prefixes and the suffixes.
    scans: [i32; 2],
}

impl Rec<'_, '_> {
    fn dispatch(&mut self, u: RasterU, size: (usize, usize), pairs: bool) -> Result<(), GpuError> {
        let (w, h) = (size.0 as u32, size.1 as u32);
        let gx = if pairs { w.div_ceil(2) } else { w }.div_ceil(16).max(1);
        let gy = h.div_ceil(16).max(1);
        let ub = self.passes.uniform("mask raster uniform", &u)?;
        let guide = self.guide.map_or(&self.samp.buf, |g| &g.buf);
        let label = match u.op[0] {
            0 => "mask raster resample",
            1 | 8 => "mask raster fold",
            2 => "mask raster mul",
            3 => "mask raster coef",
            4 => "mask raster block",
            5 | 6 => "mask raster final",
            _ => "mask raster quantise",
        };
        self.passes.dispatch(
            KernelId::MaskRaster,
            label,
            &[
                (0, ub.binding()),
                (1, self.work.binding()),
                (2, self.s0.binding()),
                (3, self.samp.buf.binding()),
                (4, guide.binding()),
                (5, self.out.binding()),
            ],
            (gx, gy, 1),
        )
    }

    /// A window fold of plane `a` along an axis into plane `d` (same size and origin), as the two ops of a blocked fold
    /// (`mask_raster.wgsl`): SCAN writes the prefix and suffix scans of every block of `2 radius + 1` samples, COMB joins a
    /// suffix to a prefix. `a_guide` says plane A is the guide.
    #[allow(clippy::too_many_arguments)] // the ops' own fields
    fn fold(
        &mut self,
        a: Slot,
        d: Slot,
        origin: (i32, i32),
        frame: [i32; 4],
        horizontal: bool,
        mode: Fold,
        radius: usize,
        a_guide: bool,
    ) -> Result<(), GpuError> {
        let mut u = self.base;
        u.op = [
            Op::Scan as u32,
            mode as u32,
            u32::from(!horizontal),
            radius as u32,
        ];
        u.pa = a.vec();
        if a_guide {
            u.pa = [0, a.w, a.h, self.guide_stride()];
        }
        let scan = |off: i32| [off, a.w, a.h, a.w];
        u.pb = scan(self.scans[0]);
        u.pc = scan(self.scans[1]);
        u.org_a = [origin.0, origin.1, i32::from(a_guide), 0];
        u.frame = frame;
        // The blocks of the plane's samples that the frame leaves: counted from the one that holds the first.
        let (len, org, f0, f1) = if horizontal {
            (a.w, origin.0, frame[0], frame[2])
        } else {
            (a.h, origin.1, frame[1], frame[3])
        };
        let (lo, hi) = (i64::from(org.max(f0)), i64::from((org + len).min(f1)));
        if lo < hi {
            let w = 2 * radius as i64 + 1;
            let blocks = (floor_div(hi - 1, w) - floor_div(lo, w) + 1) as usize;
            let lines = if horizontal { a.h } else { a.w } as usize;
            self.dispatch(u, (lines, blocks), false)?;
        }
        u.op[0] = Op::Comb as u32;
        u.pd = d.vec();
        self.dispatch(u, (a.w as usize, a.h as usize), false)
    }

    fn guide_stride(&self) -> i32 {
        self.guide.map_or(0, |g| g.stride as i32)
    }

    /// A separable window mean (a horizontal fold, then a vertical one) of `a` into `d`, through the scratch plane `t`.
    #[allow(clippy::too_many_arguments)]
    fn mean(
        &mut self,
        a: Slot,
        t: Slot,
        d: Slot,
        origin: (i32, i32),
        frame: [i32; 4],
        radius: usize,
        a_guide: bool,
    ) -> Result<(), GpuError> {
        self.fold(a, t, origin, frame, true, Fold::Mean, radius, a_guide)?;
        self.fold(t, d, origin, frame, false, Fold::Mean, radius, false)
    }

    /// MUL: `d = a * b` (either may be the guide).
    fn mul(
        &mut self,
        a: Slot,
        a_guide: bool,
        b: Slot,
        b_guide: bool,
        d: Slot,
    ) -> Result<(), GpuError> {
        let mut u = self.base;
        u.op[0] = Op::Mul as u32;
        let gs = self.guide_stride();
        u.pa = if a_guide { [0, a.w, a.h, gs] } else { a.vec() };
        u.pb = if b_guide { [0, b.w, b.h, gs] } else { b.vec() };
        u.pd = d.vec();
        u.org_a[2] = i32::from(a_guide) | (i32::from(b_guide) << 1);
        self.dispatch(u, (a.w as usize, a.h as usize), false)
    }
}

/// `floor(a / b)` for a positive `b`.
fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// `ceil(a / b)` for a positive `b`.
fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// The sampler's image the pass reads: one the caller made, or one the pass made (and keeps alive).
enum Wide<'s> {
    Borrowed(&'s GpuImage),
    Owned(GpuImage),
}

impl Wide<'_> {
    fn image(&self) -> &GpuImage {
        match self {
            Wide::Borrowed(i) => i,
            Wide::Owned(i) => i,
        }
    }
}

/// Records the raster planes of `chain` for the window `job` on the device and returns them.
///
/// `sampled` is the sampler's six-plane image of `job.grown` (`encode_sample_at` with positions): its pixel `(0, 0)`
/// is the grid's `(grown.x, grown.y)`. `prepared` is the chain prepared for the same photo frame
/// ([`PreparedChain::with_photo_frame`]); it makes the Input's luma, the guide of Refine edges, from the sampled
/// image when a row refines its edges. The planes cover `job.extent` (`job.window`, which must lie inside `job.grown`, with 0 around it when the extent is larger).
///
/// This makes every row's plane and keeps none: an export tile reads nothing another pass made
/// ([`encode_raster_planes_kept`] is the viewport's).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the chain has no raster row, the windows are inconsistent, the image is not
/// six planes of `job.grown`, or the f32 scratch would pass one storage binding; the pool's errors.
pub fn encode_raster_planes(
    passes: &mut PassList<'_>,
    prepared: &PreparedChain,
    chain: &ColorChain,
    sampled: &GpuImage,
    job: &RasterJob,
) -> Result<RasterPlanes, GpuError> {
    encode_rows(passes, prepared, chain, job, None, |_| {
        Ok(Wide::Borrowed(sampled))
    })
}

/// [`encode_raster_planes`] for a view's frames, which keeps what it makes: each row's quantised plane stays on the
/// device under its [`PlaneKey`] and its row's source, Grow, Feather and Refine edges, and the next frame with the
/// same key copies it instead of making it again (`docs/23` section 5.6). `wide` makes the sampler's image of
/// `job.grown`; it is called only when some row has no kept plane, and so is the guide.
///
/// A plane is kept when the list is submitted, and a list that is dropped keeps none. The planes of a frame are
/// bit for bit the planes the same pass makes with nothing kept ([`crate::device::Gpu::set_planes_kept`]).
///
/// # Errors
///
/// As [`encode_raster_planes`]; `wide`'s errors.
pub fn encode_raster_planes_kept<'p, W>(
    passes: &mut PassList<'p>,
    prepared: &PreparedChain,
    chain: &ColorChain,
    job: &RasterJob,
    key: &PlaneKey,
    wide: W,
) -> Result<RasterPlanes, GpuError>
where
    W: FnOnce(&mut PassList<'p>) -> Result<GpuImage, GpuError>,
{
    encode_rows(passes, prepared, chain, job, Some(key), |p| {
        wide(p).map(Wide::Owned)
    })
}

/// [`encode_raster_planes`] over a sampled picture the caller already holds, which keeps what it makes when `key` is given
/// ([`encode_raster_planes_kept`]'s cache) and nothing when it is `None`. The Film nodes' prefix chains use it: each makes
/// its rows' planes over the area the node reads, from the one wider picture the first node reads.
///
/// # Errors
///
/// As [`encode_raster_planes`].
pub fn encode_raster_planes_over(
    passes: &mut PassList<'_>,
    prepared: &PreparedChain,
    chain: &ColorChain,
    sampled: &GpuImage,
    job: &RasterJob,
    key: Option<&PlaneKey>,
) -> Result<RasterPlanes, GpuError> {
    encode_rows(passes, prepared, chain, job, key, |_| {
        Ok(Wide::Borrowed(sampled))
    })
}

/// The text a row's plane is kept under: the pass's windows and the photo's frame, the caller's place, the row, and
/// (for a row that refines its edges) the guide's.
fn row_key(
    key: &PlaneKey,
    job: &RasterJob,
    row: &ym_color::mask_raster::RasterRow,
    guided: bool,
) -> String {
    let mut k = String::with_capacity(key.place.len() + 256);
    let _ = write!(
        k,
        "{:?}|{:?}|{:?}|{:?}|{:x}|{:x}|{:x}|{:x}|{:x}|{:x}|{}",
        job.frame,
        job.grown,
        job.window,
        job.extent,
        job.photo.aspect[0].to_bits(),
        job.photo.aspect[1].to_bits(),
        job.photo.pixel.to_bits(),
        row.source.key(),
        row.grow.to_bits(),
        row.feather.to_bits(),
        row.refine.to_bits(),
    );
    k.push('|');
    k.push_str(&key.place);
    if guided {
        k.push('|');
        k.push_str(&key.guide);
    }
    k
}

fn encode_rows<'p, 's, W>(
    passes: &mut PassList<'p>,
    prepared: &PreparedChain,
    chain: &ColorChain,
    job: &RasterJob,
    cache: Option<&PlaneKey>,
    wide: W,
) -> Result<RasterPlanes, GpuError>
where
    W: FnOnce(&mut PassList<'p>) -> Result<Wide<'s>, GpuError>,
{
    let ctx = passes.context();
    let set: &RasterSet = chain.rasters();
    if set.is_empty() {
        return Err(GpuError::InvalidInput(
            "the chain has no raster row".to_owned(),
        ));
    }
    let g = job.grown;
    let w = job.window;
    let e = job.extent;
    if g.is_empty() || w.is_empty() || !g.contains(&w) || !job.frame.contains(&g) || !e.contains(&w)
    {
        return Err(GpuError::InvalidInput(format!(
            "the raster windows are inconsistent: frame {:?}, grown {g:?}, window {w:?}",
            job.frame
        )));
    }
    let u_px = 1.0 / job.photo.pixel;
    let steps: Vec<RasterSteps> = set.steps(u_px);
    let (gw, gh) = (g.w as i32, g.h as i32);
    let n = g.w * g.h;
    if max_binding_bytes(g.w, g.h) > DESIGN_BINDING_BYTES {
        return Err(GpuError::InvalidInput(format!(
            "a raster pass over {} x {} pixels needs more than the design binding",
            g.w, g.h
        )));
    }

    // The output.
    let stride = even(e.w);
    let plane_words = stride * e.h / 2;
    let plane_bytes = (plane_words * 4) as u64;
    let out_bytes = (set.rows.len() * plane_words * 4) as u64;
    let list = passes.id();
    let cache = cache.filter(|_| !ctx.counters_shared().planes_kept_off());
    let usage = if cache.is_some() {
        STORAGE_OUT | wgpu::BufferUsages::COPY_DST
    } else {
        STORAGE_OUT
    };
    let out = Arc::new(
        ctx.pool()
            .acquire("raster planes", out_bytes.max(4), usage)?,
    );
    passes.hold(Arc::clone(&out));

    // The rows that have a kept plane are copied into place; the others are made.
    let keys: Vec<Option<String>> = set
        .rows
        .iter()
        .zip(&steps)
        .map(|(row, st)| cache.map(|k| row_key(k, job, row, st.guided.is_some())))
        .collect();
    let mut todo: Vec<usize> = Vec::with_capacity(set.rows.len());
    let mut hits = 0u64;
    for (index, key) in keys.iter().enumerate() {
        let kept = cache
            .zip(key.as_deref())
            .and_then(|(k, text)| plane_lookup(ctx, list, k.owner, text));
        match kept {
            Some(buf) => {
                passes.encoder().copy_buffer_to_buffer(
                    &buf,
                    0,
                    &out,
                    index as u64 * plane_bytes,
                    plane_bytes,
                );
                passes.hold(buf);
                hits += 1;
            }
            None => todo.push(index),
        }
    }
    ctx.counters_shared().count_planes(todo.len() as u64, hits);
    let plane = RasterPlanes {
        buf: Arc::clone(&out),
        width: e.w,
        height: e.h,
        stride,
        planes: set.rows.len(),
        row0: 0,
        rows_held: e.h,
    };
    if todo.is_empty() {
        return Ok(plane);
    }

    let wide_image = wide(passes)?;
    let sampled = wide_image.image();
    if sampled.planes != 6 || (sampled.width, sampled.height) != (g.w, g.h) {
        return Err(GpuError::InvalidInput(format!(
            "the raster pass reads the six-plane image of the grown window ({} x {}), not {} planes of {} x {}",
            g.w, g.h, sampled.planes, sampled.width, sampled.height
        )));
    }
    perf::note_raster(todo.len(), g.w, g.h);

    // The guide.
    let guide = if todo.iter().any(|&i| steps[i].guided.is_some()) {
        let _span = perf::span(Stage::GuideRecord);
        let image = GpuImage::alloc(ctx, "raster guide", g.w, g.h, 4, Placement::whole(g.w, g.h))?;
        prepared.encode_stage(
            passes,
            sampled,
            Rect::new(0, 0, g.w, g.h),
            Stop::Guide,
            &image,
            (0, 0),
        )?;
        Some(image)
    } else {
        None
    };

    // The scratch: the largest any row needs. Two planes of the mask and a temporary (`p`, `t`), then the two planes
    // that a fold's scans write (as big as the biggest plane that is folded), then the planes of the guided filter.
    let coarse = |s: usize| {
        let s = s as i64;
        let cw = ceil_div(g.x + g.w as i64, s) - floor_div(g.x, s);
        let ch = ceil_div(g.y + g.h as i64, s) - floor_div(g.y, s);
        (cw as usize, ch as usize)
    };
    let made = || todo.iter().map(|&i| &steps[i]);
    let scan_len = made()
        .filter_map(|st| st.guided.as_ref().filter(|gf| gf.step > 1))
        .map(|gf| {
            let (cw, ch) = coarse(gf.step);
            cw * ch
        })
        .fold(n, usize::max);
    let rest = 2 * n + 2 * scan_len;
    let need = made()
        .map(|st| match &st.guided {
            None => rest,
            Some(gf) if gf.step == 1 => rest + (FULL_SLOTS - 4) * n,
            Some(gf) => {
                let (cw, ch) = coarse(gf.step);
                rest + COARSE_SLOTS * cw * ch
            }
        })
        .max()
        .unwrap_or(rest);
    let work = passes.scratch("raster work", (need * 4) as u64)?;
    let scans = [(2 * n) as i32, (2 * n + scan_len) as i32];

    let frame_fine = [
        job.frame.x as i32,
        job.frame.y as i32,
        job.frame.x1() as i32,
        job.frame.y1() as i32,
    ];
    let origin = (g.x as i32, g.y as i32);
    let win = [
        (e.x - g.x) as i32,
        (e.y - g.y) as i32,
        e.w as i32,
        e.h as i32,
    ];
    let ins = [
        (w.x - g.x) as i32,
        (w.y - g.y) as i32,
        (w.x1() - g.x) as i32,
        (w.y1() - g.y) as i32,
    ];
    let slot = |k: usize| Slot {
        off: (k * n) as i32,
        w: gw,
        h: gh,
    };

    let mut record = perf::span(Stage::RasterRecord);
    // The source's time is its own span's, not the record's.
    let mut source_ms = 0.0;
    let cap = kept_bytes_cap(ctx);
    let mut staged = false;
    for &index in &todo {
        let (row, st) = (&set.rows[index], &steps[index]);
        let started = perf::enabled().then(Instant::now);
        let src: Arc<SourcePlane> = row.source.plane_among(job.photo.aspect, set.rows.len());
        let made = started.map(|t| t.elapsed().as_secs_f64() * 1000.0);
        let started = perf::enabled().then(Instant::now);
        let s0 = source_buffer(ctx, &src, set.rows.len())?;
        let uploaded = started.map(|t| t.elapsed().as_secs_f64() * 1000.0);
        if let (Some(made), Some(uploaded)) = (made, uploaded) {
            perf::record(Stage::Source, made);
            perf::record(Stage::SourceUpload, uploaded);
            source_ms += made + uploaded;
        }
        passes.hold(Arc::clone(&s0));
        let taps = ym_color::mask_raster::resample_taps(src.width.max(src.height), u_px);
        let shrink = src.width.max(src.height) as f64 / u_px;
        let base = RasterU {
            samp: [sampled.stride as u32, sampled.plane_len() as u32, 0, 0],
            s0: [src.width as u32, src.height as u32, taps as u32, 0],
            f: [0.0, shrink as f32, 0.0, 0.0],
            outp: [stride as u32, plane_words as u32, index as u32, 0],
            win,
            ins,
            frame: frame_fine,
            ..RasterU::default()
        };
        let mut rec = Rec {
            passes,
            work: Arc::clone(&work),
            s0,
            samp: sampled,
            guide: guide.as_ref(),
            out: Arc::clone(&out),
            base,
            scans,
        };
        let (p, t) = (slot(0), slot(1));
        // 2. The resample.
        {
            let mut u = rec.base;
            u.op[0] = Op::Resample as u32;
            u.pd = p.vec();
            rec.dispatch(u, (g.w, g.h), false)?;
        }
        // 3. Grow.
        if st.grow != 0 {
            let mode = if st.grow > 0 { Fold::Max } else { Fold::Min };
            let r = st.grow.unsigned_abs() as usize;
            rec.fold(p, t, origin, frame_fine, true, mode, r, false)?;
            rec.fold(t, p, origin, frame_fine, false, mode, r, false)?;
        }
        // 4. Feather.
        if let Some(boxes) = st.feather {
            for r in boxes {
                rec.mean(p, t, p, origin, frame_fine, r, false)?;
            }
        }
        // 5. Refine edges.
        match &st.guided {
            None => {
                let mut u = rec.base;
                u.op[0] = Op::Quant as u32;
                u.pa = p.vec();
                u.org_a = [0; 4];
                rec.dispatch(u, (e.w, e.h), true)?;
            }
            Some(gf) if gf.step == 1 => {
                // Seven planes of the grown window: p, t and the two scan planes, then three more. The order of the
                // means keeps at most four of the means alive at once and reuses p's plane once p is spent.
                let more = |k: usize| Slot {
                    off: (rest + k * n) as i32,
                    w: gw,
                    h: gh,
                };
                let (s4, s5, s6) = (more(0), more(1), more(2));
                let r = gf.samples;
                // mean(guide * p) into s4, through s5 as the product.
                rec.mul(slot(0), true, p, false, s5)?;
                rec.mean(s5, t, s4, origin, frame_fine, r, false)?;
                // mean(p) into s5 (the product is spent). p is spent after it.
                rec.mean(p, t, s5, origin, frame_fine, r, false)?;
                // mean(guide^2) into s6, through p's plane as the product.
                rec.mul(slot(0), true, slot(0), true, p)?;
                rec.mean(p, t, s6, origin, frame_fine, r, false)?;
                // mean(guide) into p's plane. The guide is the first fold's input.
                rec.fold(slot(0), t, origin, frame_fine, true, Fold::Mean, r, true)?;
                rec.fold(t, p, origin, frame_fine, false, Fold::Mean, r, false)?;
                // a (in place of s6) and b (in place of s4) from the means of the guide (p), of the mask (s5), of
                // the guide squared (s6) and of the guide times the mask (s4).
                {
                    let mut u = rec.base;
                    u.op[0] = Op::Coef as u32;
                    u.pa = p.vec();
                    u.pb = s5.vec();
                    u.pc = [s6.off, s4.off, 0, gw];
                    u.f[0] = gf.eps as f32;
                    rec.dispatch(u, (g.w, g.h), false)?;
                }
                // a-bar into p's place, b-bar into s5's.
                rec.mean(s6, t, p, origin, frame_fine, r, false)?;
                rec.mean(s4, t, s5, origin, frame_fine, r, false)?;
                let mut u = rec.base;
                u.op[0] = Op::Final as u32;
                u.pa = p.vec();
                u.pb = s5.vec();
                u.pc = [0, 0, 0, rec.guide_stride()];
                rec.dispatch(u, (e.w, e.h), true)?;
            }
            Some(gf) => {
                let s = gf.step as i64;
                let (cw, ch) = coarse(gf.step);
                let cn = cw * ch;
                let c0 = rest;
                let cs = |k: usize| Slot {
                    off: (c0 + k * cn) as i32,
                    w: cw as i32,
                    h: ch as i32,
                };
                let c_origin = (floor_div(g.x, s) as i32, floor_div(g.y, s) as i32);
                let c_frame = [
                    floor_div(job.frame.x, s) as i32,
                    floor_div(job.frame.y, s) as i32,
                    ceil_div(job.frame.x1(), s) as i32,
                    ceil_div(job.frame.y1(), s) as i32,
                ];
                // The block means of the guide and the mask.
                for (src_slot, guide_in, dst) in [(slot(0), true, cs(0)), (p, false, cs(1))] {
                    let mut u = rec.base;
                    u.op[0] = Op::Block as u32;
                    u.pa = if guide_in {
                        [0, gw, gh, rec.guide_stride()]
                    } else {
                        src_slot.vec()
                    };
                    u.pd = dst.vec();
                    u.org_a = [origin.0, origin.1, i32::from(guide_in), 0];
                    u.org_d = [c_origin.0, c_origin.1, gf.step as i32, 0];
                    u.frame = frame_fine;
                    rec.dispatch(u, (cw, ch), false)?;
                }
                let (c_g, c_p, c_t, c_mg, c_mp, c_mgg, c_mgp, c_t2) =
                    (cs(0), cs(1), cs(2), cs(3), cs(4), cs(5), cs(6), cs(7));
                let r = gf.samples;
                rec.mean(c_g, c_t, c_mg, c_origin, c_frame, r, false)?;
                rec.mean(c_p, c_t, c_mp, c_origin, c_frame, r, false)?;
                rec.mul(c_g, false, c_g, false, c_t2)?;
                rec.mean(c_t2, c_t, c_mgg, c_origin, c_frame, r, false)?;
                rec.mul(c_g, false, c_p, false, c_t2)?;
                rec.mean(c_t2, c_t, c_mgp, c_origin, c_frame, r, false)?;
                {
                    let mut u = rec.base;
                    u.op[0] = Op::Coef as u32;
                    u.pa = c_mg.vec();
                    u.pb = c_mp.vec();
                    u.pc = [c_mgg.off, c_mgp.off, 0, cw as i32];
                    u.f[0] = gf.eps as f32;
                    rec.dispatch(u, (cw, ch), false)?;
                }
                rec.mean(c_mgg, c_t, c_mg, c_origin, c_frame, r, false)?;
                rec.mean(c_mgp, c_t, c_mp, c_origin, c_frame, r, false)?;
                let mut u = rec.base;
                u.op[0] = Op::Up as u32;
                u.pa = c_mg.vec();
                u.pb = c_mp.vec();
                u.pc = [0, 0, 0, rec.guide_stride()];
                u.org_a = [origin.0, origin.1, 0, 0];
                u.org_d = [c_origin.0, c_origin.1, gf.step as i32, 0];
                u.frame = c_frame;
                u.cg = [c_origin.0, c_origin.1, cw as i32, ch as i32];
                rec.dispatch(u, (e.w, e.h), true)?;
            }
        }
        // Keep the row's plane: a copy of it, filled by this list and served once the list is submitted. A device
        // that cannot hold the copy (the budget) keeps none, and the frame is the same.
        if let (Some(k), Some(text)) = (cache, keys[index].as_ref())
            && plane_bytes <= cap
            && let Ok(kept) = ctx.pool().acquire(
                "raster plane (kept)",
                plane_bytes.max(4),
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            )
        {
            passes.encoder().copy_buffer_to_buffer(
                &out,
                index as u64 * plane_bytes,
                &kept,
                0,
                plane_bytes,
            );
            let entry = PlaneEntry {
                owner: k.owner,
                key: text.clone(),
                buf: Arc::new(kept),
                bytes: plane_bytes,
            };
            with_planes(ctx.counters_shared(), |d| d.staged.push((list, entry)));
            staged = true;
        }
    }
    if let Some(r) = record.as_mut() {
        r.subtract(source_ms);
    }
    drop(record);
    if staged {
        let device = Arc::downgrade(ctx.counters_shared());
        passes.on_submit(move || plane_commit(&device, list, cap));
    }
    if let Wide::Owned(image) = wide_image {
        // The picture must outlive the submission.
        let _ = passes.keep(image.buf);
    }
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::MaskRaster);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("RU_"))
            .expect("the uniform RU_");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("RU_ is not a struct");
        };
        let fields = [
            ("op", offset_of!(RasterU, op)),
            ("pa", offset_of!(RasterU, pa)),
            ("pb", offset_of!(RasterU, pb)),
            ("pc", offset_of!(RasterU, pc)),
            ("pd", offset_of!(RasterU, pd)),
            ("org_a", offset_of!(RasterU, org_a)),
            ("org_d", offset_of!(RasterU, org_d)),
            ("frame", offset_of!(RasterU, frame)),
            ("win", offset_of!(RasterU, win)),
            ("ins", offset_of!(RasterU, ins)),
            ("outp", offset_of!(RasterU, outp)),
            ("samp", offset_of!(RasterU, samp)),
            ("s0", offset_of!(RasterU, s0)),
            ("f", offset_of!(RasterU, f)),
            ("cg", offset_of!(RasterU, cg)),
        ];
        assert_eq!(members.len(), fields.len());
        for (m, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(m.name.as_deref(), Some(name));
            assert_eq!(m.offset as usize, offset, "{name}");
        }
        assert_eq!(*span as usize, size_of::<RasterU>());
    }

    #[test]
    fn a_grown_window_that_fits_the_binding_is_found_for_every_halo() {
        for halo in [0, 100, 300, 600, 900] {
            let tile = largest_tile_for(halo, 1024);
            assert!(tile >= 64);
            if tile > 64 {
                assert!(
                    max_binding_bytes(tile + 2 * halo, tile + 2 * halo) <= DESIGN_BINDING_BYTES,
                    "halo {halo}: tile {tile}"
                );
            }
        }
        assert_eq!(largest_tile_for(0, 1024), 1024);
    }
}
