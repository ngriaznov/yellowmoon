//! Whole-image levels: a pan shows a picture that is already rendered (G-13, `docs/11-ui.md` section
//! "Pans", `docs/17-gpu.md` section "Levels").
//!
//! # What a level is
//!
//! A **level** is the whole final image rendered once at one exact zoom `z`, finished (Film, the
//! colour chain, output sharpening, C13 with its dither, the Display P3 encode) and kept on the device
//! as RGBA8 in strips of whole rows. It is the picture [`Renderer::viewport_frame_to`] would draw for
//! a window of the photo, made for every window at once: a pan at that zoom copies the rectangle the
//! cell shows out of the strips into a cell-sized device buffer (a [`DeviceFrame`], so the native
//! viewport is unchanged) and renders nothing.
//!
//! Levels are at the *exact* zoom and not at the raw group's scale `k`: Film grain, halation, the
//! colour chain's non-linearity and output sharpening all work on the view's own pixel grid, so a
//! level at `k` resampled to `z` would not be the picture a window of that zoom shows. A level at `z`
//! is, window for window, the picture the window path makes, **bit for bit** on the same device when
//! the window starts on a whole pixel of the level (the viewport snaps its origin to whole device
//! pixels, [`ViewportState::view_to_final`](crate::pipeline::ViewportState::view_to_final)) and the
//! window's dither is counted from its place in the level ([`ViewSpec::dither_origin`]): the dither
//! sticks to the image, not to the screen, which differs from the screen-anchored dither by one code
//! at most.
//!
//! # When there is one
//!
//! * [`RenderOptions::levels`] is on, the request is a device frame ([`FrameOutput::Device`]) of the
//!   photo as it is (not the Before view, not a mask view), the device runs the view stages, and the
//!   photo does not fit the cell (`is_fit`) at a zoom up to [`LEVEL_MAX_ZOOM`] (150 %). Fit is served
//!   by the frame cache already; above 150 % a window reads a small part of the 100 % picture.
//! * The level is no larger than a share of the device's budget ([`Renderer`]'s `budget`,
//!   `gpu_bytes / 8` for all levels of a photo) and at most [`LEVELS_PER_PHOTO`] of them are kept
//!   (the zoom on screen and the one before it, so toggling between two presets reuses both).
//! * The window lies inside the level: a photo narrower than the cell has a letterbox, which a level
//!   does not hold, and that window is rendered as before.
//!
//! # The key
//!
//! The store is keyed by the *settings key*: the raw group's parameters, the grade plan's key (which
//! holds the node graph, the masks, the Film nodes, the look, the display transform and the **keys of
//! the Background, Object, Subject and People planes**, so a plane that arrives after the settings
//! is a new key), the geometry, the lens key and the Output Sharpening amount. A request whose
//! settings key differs from the store's frees every level and cancels the fill in flight; within a
//! settings key a level is named by its zoom (`f64::to_bits`).
//!
//! # The fill
//!
//! After a window is rendered with no level to serve it, [`Renderer::viewport_frame_to`] records
//! what it wants ([`LevelAsk`]). A short-lived thread waits [`LEVEL_DEBOUNCE`] for the key (settings
//! and zoom) to stay the same, so that a slider drag, which changes it every few milliseconds, starts
//! nothing; then the level is filled by **one scheduler job per band** at [`Priority::Fill`], below
//! every viewport frame and tile job (a running job is never pre-empted, so a frame waits for one
//! band at most; the bands are at most [`FILL_BAND_PX`] pixels). Each band is a view of its own,
//! made by the same passes as a band of a cut view (`banded`): its dither counts from its first row,
//! its output-sharpening window is the whole level's cut to its rows. The bands nearest the rows on
//! screen go first, and a window whose rows are all filled is served while the rest fills.
//!
//! The fill gives up quietly (the window path goes on as before) when the device cannot cut the level
//! into bands, runs a band out of memory, or makes one on the CPU: a level is never made of two
//! devices.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use ym_gpu::device::Gpu;
use ym_gpu::pool::PooledBuffer;
use ym_gpu::queue::{CancelToken, Priority};

use super::device::{self, Bounds, CropJob, DraftJob, GatherJob, RowCopy, ViewRoute, ViewStages};
use super::{
    Flow, FrameOutput, OutputTarget, RenderError, Rgba8, Shape, StageWindow, ViewSpec, band_split,
    band_window, magnified_band_window, raster_grown, scale_for_zoom, view_sharpening, whole,
    whole_window,
};
use crate::pipeline::color::PhotoSource;
use crate::pipeline::geometry::Affine;
use crate::pipeline::graph::{ComputeDevice, Job, PhotoGraph, Renderer};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::kernels::level_draft::DraftMap;

/// The largest zoom a level is kept for (150 %): above it a window reads a small part of the 100 %
/// picture and renders in about the time a pan would take to copy.
pub const LEVEL_MAX_ZOOM: f64 = 1.5;

/// How long the key of the wanted level (settings and zoom) must stay the same before the fill
/// starts.
pub const LEVEL_DEBOUNCE: Duration = Duration::from_millis(300);

/// The largest band of a fill, in pixels: a viewport frame that arrives while a band renders waits
/// for it, so a band is a few frames' worth of work, not a photo's.
pub const FILL_BAND_PX: usize = 2_000_000;

/// How many levels of one photo are kept: the zoom on screen and the one before it.
pub const LEVELS_PER_PHOTO: usize = 2;

/// Where the window the cell shows lies in a level of the zoom on screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LevelGeometry {
    /// The zoom.
    pub zoom: f64,
    /// The size of the level: the final image times the zoom, in whole pixels.
    pub size: (usize, usize),
    /// The window's first pixel in the level.
    pub origin: (usize, usize),
}

/// The level geometry of a view, if the zoom has levels and the view is a window inside one:
/// `view_to_final` is `scale_translate(1/z, 1/z, n/z, m/z)` for whole `n`, `m`, and the cell
/// `(dw, dh)` at that origin lies inside the image's `round(image * z)` pixels.
pub(crate) fn geometry_of(
    view_to_final: &Affine,
    zoom: f64,
    image: (usize, usize),
    cell: (usize, usize),
    is_fit: bool,
) -> Option<LevelGeometry> {
    if is_fit || !zoom.is_finite() || zoom <= 0.0 || zoom > LEVEL_MAX_ZOOM {
        return None;
    }
    let size = (
        (image.0 as f64 * zoom).round() as usize,
        (image.1 as f64 * zoom).round() as usize,
    );
    if size.0 == 0 || size.1 == 0 {
        return None;
    }
    let (ox, oy) = view_to_final.apply(0.0, 0.0);
    let whole = |v: f64| {
        let n = v * zoom;
        ((n - n.round()).abs() < 1e-6 && n.round() >= 0.0).then(|| n.round() as usize)
    };
    let origin = (whole(ox)?, whole(oy)?);
    (origin.0 + cell.0 <= size.0 && origin.1 + cell.1 <= size.1).then_some(LevelGeometry {
        zoom,
        size,
        origin,
    })
}

/// What a request that has a level geometry needs from the store.
#[derive(Debug, Clone)]
pub(crate) struct LevelAsk {
    /// The settings key (module notes).
    pub settings_key: String,
    /// Where the window is.
    pub geometry: LevelGeometry,
    /// The cell.
    pub cell: (usize, usize),
    /// The bytes all levels of the photo may take.
    pub cap: u64,
}

impl LevelAsk {
    fn zoom_bits(&self) -> u64 {
        self.geometry.zoom.to_bits()
    }

    fn level_bytes(&self) -> u64 {
        4 * self.geometry.size.0 as u64 * self.geometry.size.1 as u64
    }
}

/// A strip a window reads: its first row, its rows, its buffer.
type StripRef = (usize, usize, Arc<PooledBuffer>);

/// A frame made from a level: the buffer, and the device it is on.
pub(crate) type Cropped = (Arc<PooledBuffer>, Weak<Gpu>);

/// A band of whole rows of a level, on the device.
struct Strip {
    rows: usize,
    buffer: Arc<PooledBuffer>,
}

/// One level: planned bands and the strips made so far.
struct Level {
    zoom_bits: u64,
    size: (usize, usize),
    /// `(first row, rows)` of every band.
    bands: Vec<(usize, usize)>,
    /// The strips made, by first row.
    strips: BTreeMap<usize, Strip>,
    gpu: Weak<Gpu>,
    used: u64,
}

impl Level {
    fn reserved(&self) -> u64 {
        4 * self.size.0 as u64 * self.size.1 as u64
    }

    fn bytes(&self) -> u64 {
        self.strips.values().map(|s| s.buffer.size()).sum()
    }

    fn complete(&self) -> bool {
        self.strips.len() == self.bands.len()
    }
}

/// How a level stands, for the log line and the tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelStatus {
    /// No level of this zoom, and none being made.
    None,
    /// Waiting for the settings and zoom to hold still.
    Waiting,
    /// `done` of `total` bands are made.
    Filling {
        /// Bands made.
        done: usize,
        /// Bands in all (0 until the level is cut).
        total: usize,
    },
    /// Every band is made: a pan renders nothing.
    Ready,
}

/// The cut of a level and what its bands share, found once by the first job of a fill.
#[derive(Clone)]
struct FillPlan {
    /// `(first row, rows)` of every band.
    bands: Vec<(usize, usize)>,
    /// The window output sharpening works on in the whole level's frame (`None`: no stage).
    whole_box: Option<Option<Bounds>>,
    /// Where the 100 % picture under a magnified level starts.
    whole_origin: Option<(usize, usize)>,
    halo: usize,
}

/// The fill in flight.
struct Fill {
    id: u64,
    zoom_bits: u64,
    plan: Option<Arc<FillPlan>>,
    /// Band indices still to make, nearest the rows on screen first.
    todo: Vec<usize>,
    /// The token of the job that is queued or running: cancelled with the fill.
    token: Option<CancelToken>,
}

/// What a request wanted and has not got, waiting for the key to hold still.
struct Want {
    ask: LevelAsk,
    src: PhotoSource,
    deadline: Instant,
}

/// The levels of one photo (a field of its graph's state; module notes).
#[derive(Default)]
pub(crate) struct LevelStore {
    settings_key: String,
    levels: Vec<Level>,
    fill: Option<Fill>,
    want: Option<Want>,
    /// A thread is waiting to start a fill.
    timer: bool,
    /// Zooms whose level the device cannot cut or make (until the key changes).
    refused: Vec<u64>,
    tick: u64,
}

static NEXT_FILL: AtomicU64 = AtomicU64::new(1);

impl LevelStore {
    /// Brings the store to `settings_key`: levels of another key are freed and the fill ends.
    pub(crate) fn reconcile(&mut self, settings_key: &str) {
        if self.settings_key == settings_key {
            return;
        }
        self.clear();
        self.settings_key = settings_key.to_owned();
    }

    /// Frees every level and ends the fill and the wait for one.
    pub(crate) fn clear(&mut self) {
        self.cancel_fill();
        self.levels.clear();
        self.want = None;
        self.refused.clear();
    }

    fn cancel_fill(&mut self) {
        if let Some(fill) = self.fill.take()
            && let Some(token) = fill.token
        {
            token.cancel();
        }
    }

    /// The bytes the strips made so far take on the device.
    pub(crate) fn bytes(&self) -> usize {
        self.levels.iter().map(Level::bytes).sum::<u64>() as usize
    }

    /// How many levels the store holds (partly made ones included).
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    /// How the level of `zoom` stands.
    pub(crate) fn status(&self, zoom: f64) -> LevelStatus {
        let bits = zoom.to_bits();
        if let Some(level) = self.levels.iter().find(|l| l.zoom_bits == bits) {
            if level.complete() {
                return LevelStatus::Ready;
            }
            return LevelStatus::Filling {
                done: level.strips.len(),
                total: level.bands.len(),
            };
        }
        if self.fill.as_ref().is_some_and(|f| f.zoom_bits == bits) {
            return LevelStatus::Filling { done: 0, total: 0 };
        }
        if self
            .want
            .as_ref()
            .is_some_and(|w| w.ask.zoom_bits() == bits)
        {
            return LevelStatus::Waiting;
        }
        LevelStatus::None
    }

    /// The strips under the rows `y0..y1` of the level of `zoom` on `gpu`, when every one of those rows is made:
    /// `(level width, [(first row, rows, buffer)])`.
    fn cover(
        &mut self,
        zoom_bits: u64,
        gpu: &Arc<Gpu>,
        y0: usize,
        y1: usize,
    ) -> Option<(usize, Vec<StripRef>)> {
        self.tick += 1;
        let tick = self.tick;
        let level = self
            .levels
            .iter_mut()
            .find(|l| l.zoom_bits == zoom_bits && Weak::ptr_eq(&l.gpu, &Arc::downgrade(gpu)))?;
        let mut parts = Vec::new();
        let mut y = y0;
        while y < y1 {
            let (&row0, strip) = level.strips.range(..=y).next_back()?;
            if y >= row0 + strip.rows {
                return None;
            }
            parts.push((row0, strip.rows, Arc::clone(&strip.buffer)));
            y = row0 + strip.rows;
        }
        level.used = tick;
        Some((level.size.0, parts))
    }

    /// Records that a request wants the level of `ask`; `true` when a thread has to be started to wait for the key to hold
    /// still. Nothing is recorded for a level that is made, being made or refused.
    fn want(&mut self, ask: LevelAsk, src: &PhotoSource, debounce: Instant) -> bool {
        if self.settings_key != ask.settings_key || self.refused.contains(&ask.zoom_bits()) {
            return false;
        }
        // A fill whose job was cancelled (navigation, a closed session) is over.
        if self
            .fill
            .as_ref()
            .is_some_and(|f| f.token.as_ref().is_some_and(CancelToken::is_cancelled))
        {
            self.fill = None;
        }
        let bits = ask.zoom_bits();
        let made = self
            .levels
            .iter()
            .any(|l| l.zoom_bits == bits && l.complete());
        if made || self.fill.as_ref().is_some_and(|f| f.zoom_bits == bits) {
            return false;
        }
        // The same level asked for again keeps its deadline; another one (a new zoom) moves it.
        let same = self
            .want
            .as_ref()
            .is_some_and(|w| w.ask.zoom_bits() == bits && w.ask.settings_key == ask.settings_key);
        if !same {
            self.want = Some(Want {
                ask,
                src: src.clone(),
                deadline: debounce,
            });
        }
        !std::mem::replace(&mut self.timer, true)
    }

    /// The fill the wait ended in: the wanted level when its key is still the store's.
    fn start_fill(&mut self, want: &Want) -> Option<FillTicket> {
        let ask = &want.ask;
        if self.settings_key != ask.settings_key {
            return None;
        }
        if self
            .levels
            .iter()
            .any(|l| l.zoom_bits == ask.zoom_bits() && l.complete())
        {
            return None;
        }
        // Whatever was waiting is answered by this fill.
        self.want = None;
        // The same level being made already (its job is alive): nothing to start.
        if self.fill.as_ref().is_some_and(|f| {
            f.zoom_bits == ask.zoom_bits()
                && !f.token.as_ref().is_some_and(CancelToken::is_cancelled)
        }) {
            return None;
        }
        // Another fill in flight (for another zoom) ends: a fill at a time.
        self.cancel_fill();
        let id = NEXT_FILL.fetch_add(1, Ordering::Relaxed);
        self.fill = Some(Fill {
            id,
            zoom_bits: ask.zoom_bits(),
            plan: None,
            todo: Vec::new(),
            token: None,
        });
        Some(FillTicket {
            id,
            ask: ask.clone(),
        })
    }

    fn fill_is(&self, id: u64) -> bool {
        self.fill.as_ref().is_some_and(|f| f.id == id)
    }

    /// The cut of the fill `id`, once it is made; creates the level (evicting older ones to stay inside the cap).
    fn set_plan(
        &mut self,
        ticket: &FillTicket,
        plan: FillPlan,
        gpu: &Arc<Gpu>,
        centre_row: usize,
    ) -> bool {
        if !self.fill_is(ticket.id) {
            return false;
        }
        let ask = &ticket.ask;
        let bits = ask.zoom_bits();
        let need = ask.level_bytes();
        // A level of this zoom made for another device is no use.
        self.levels
            .retain(|l| l.zoom_bits != bits || Weak::ptr_eq(&l.gpu, &Arc::downgrade(gpu)));
        if !self.levels.iter().any(|l| l.zoom_bits == bits) {
            // Older levels go (least recently used first) until the new one fits the cap and the count.
            loop {
                let held: u64 = self.levels.iter().map(Level::reserved).sum();
                if self.levels.len() < LEVELS_PER_PHOTO && held + need <= ask.cap {
                    break;
                }
                let Some(oldest) = self
                    .levels
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, l)| l.used)
                    .map(|(i, _)| i)
                else {
                    break;
                };
                self.levels.remove(oldest);
            }
            if need > ask.cap {
                return false;
            }
            self.tick += 1;
            self.levels.push(Level {
                zoom_bits: bits,
                size: ask.geometry.size,
                bands: plan.bands.clone(),
                strips: BTreeMap::new(),
                gpu: Arc::downgrade(gpu),
                used: self.tick,
            });
        }
        let Some(level) = self.levels.iter().find(|l| l.zoom_bits == bits) else {
            return false;
        };
        // The bands not made yet, those under the rows on screen first.
        let mut todo: Vec<usize> = (0..plan.bands.len())
            .filter(|i| !level.strips.contains_key(&plan.bands[*i].0))
            .collect();
        let distance = |i: &usize| {
            let (row0, rows) = plan.bands[*i];
            if (row0..row0 + rows).contains(&centre_row) {
                0
            } else {
                centre_row
                    .abs_diff(row0)
                    .min(centre_row.abs_diff(row0 + rows))
            }
        };
        todo.sort_by_key(distance);
        if let Some(fill) = self.fill.as_mut() {
            fill.plan = Some(Arc::new(plan));
            fill.todo = todo;
        }
        true
    }

    /// The next band of the fill `id` to make: `(band index, plan)`.
    fn next_band(&mut self, id: u64) -> Option<(usize, Arc<FillPlan>)> {
        let fill = self.fill.as_mut().filter(|f| f.id == id)?;
        let plan = fill.plan.clone()?;
        if fill.todo.is_empty() {
            return None;
        }
        Some((fill.todo[0], plan))
    }

    /// Keeps the strip of band `index` of the fill `id`. `true` when the fill goes on.
    fn put(&mut self, id: u64, index: usize, buffer: Arc<PooledBuffer>) -> bool {
        let Some(fill) = self.fill.as_mut().filter(|f| f.id == id) else {
            return false;
        };
        let Some(plan) = fill.plan.clone() else {
            return false;
        };
        let bits = fill.zoom_bits;
        fill.todo.retain(|i| *i != index);
        let more = !fill.todo.is_empty();
        let (row0, rows) = plan.bands[index];
        if let Some(level) = self.levels.iter_mut().find(|l| l.zoom_bits == bits) {
            level.strips.insert(row0, Strip { rows, buffer });
        } else {
            self.fill = None;
            return false;
        }
        if !more {
            self.fill = None;
        }
        more
    }

    /// The fill `id` ends: the level it was making is kept as far as it got, or dropped when `drop_level`.
    fn end_fill(&mut self, id: u64, refuse: Option<u64>) {
        if !self.fill_is(id) {
            return;
        }
        self.fill = None;
        if let Some(bits) = refuse {
            self.levels.retain(|l| l.zoom_bits != bits);
            self.refused.push(bits);
        }
    }

    fn set_token(&mut self, id: u64, token: CancelToken) {
        if let Some(fill) = self.fill.as_mut().filter(|f| f.id == id) {
            fill.token = Some(token);
        } else {
            token.cancel();
        }
    }
}

/// What a fill job needs to know of its fill.
#[derive(Clone)]
pub(crate) struct FillTicket {
    id: u64,
    ask: LevelAsk,
}

// ---------------------------------------------------------------------------------------------
// The key
// ---------------------------------------------------------------------------------------------

/// The settings key of a view (module notes): everything that makes the finished picture of the photo except the view.
fn settings_key(
    job: &mut Job<'_>,
    grade: &crate::pipeline::color::GradePlan,
    sharpened: bool,
) -> Result<String, RenderError> {
    let params = job.params()?;
    let geometry = job.geometry()?;
    let mut key = format!("{:?}|{:?}|{:?}", params.key(), grade.key, geometry);
    if sharpened {
        key.push_str(&format!(
            "|os{:?}",
            job.src.settings.output().output_sharpening
        ));
    }
    if let Some(lens) = job.lens_key()? {
        key.push_str(&format!("|{lens}"));
    }
    Ok(key)
}

/// The level a viewport frame asks for, and the store brought to its settings key; `None` when the frame has no
/// level (module notes). Called before the frame is rendered, with the frame's `spec` already holding the window's dither
/// origin.
pub(crate) fn ask(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    geometry: Option<LevelGeometry>,
    mask_view: bool,
) -> Result<Option<LevelAsk>, RenderError> {
    if !job.options().levels || spec.output != FrameOutput::Device || spec.before || mask_view {
        return Ok(None);
    }
    let Some(gpu) = job.view_gpu()? else {
        return Ok(None);
    };
    let stages = ViewStages::of(gpu.stages);
    let grade = job.grade(spec.target, true)?;
    let sharpening = view_sharpening(job, spec)?;
    // A level is made by the device from the sampler to the RGBA8 frame, or not at all.
    let key = settings_key(job, &grade, sharpening.is_some())?;
    job.g.with_levels(|store| store.reconcile(&key));
    let runs_on_device = stages.color && (sharpening.is_none() || stages.tail);
    let Some(geometry) = geometry.filter(|_| runs_on_device && spec.cache) else {
        return Ok(None);
    };
    Ok(Some(LevelAsk {
        settings_key: key,
        geometry,
        cell: (spec.out_w, spec.out_h),
        cap: job.options().budget.gpu_bytes / 8,
    }))
}

// ---------------------------------------------------------------------------------------------
// Serving a pan
// ---------------------------------------------------------------------------------------------

/// The copies that put the rectangle `size` pixels at `origin` of a level `width` pixels wide together, from the strips that cover
/// its rows (`parts`: first row, rows, buffer), into a buffer of `size.0 * size.1` pixels.
fn row_copies(
    parts: &[StripRef],
    width: usize,
    origin: (usize, usize),
    size: (usize, usize),
) -> Vec<RowCopy> {
    let ((x0, y0), (dw, dh)) = (origin, size);
    let mut copies: Vec<RowCopy> = Vec::new();
    let full_width = x0 == 0 && dw == width;
    let mut y = y0;
    for (row0, rows, buffer) in parts {
        let end = (row0 + rows).min(y0 + dh);
        if full_width {
            copies.push(RowCopy {
                src: Arc::clone(buffer),
                src_at: 4 * ((y - row0) * width) as u64,
                dst_at: 4 * ((y - y0) * dw) as u64,
                len: 4 * ((end - y) * dw) as u64,
            });
        } else {
            for r in y..end {
                copies.push(RowCopy {
                    src: Arc::clone(buffer),
                    src_at: 4 * ((r - row0) * width + x0) as u64,
                    dst_at: 4 * ((r - y0) * dw) as u64,
                    len: 4 * dw as u64,
                });
            }
        }
        y = end;
    }
    copies
}

/// The window of `ask` copied out of its level, when the level has every row of it: nothing is sampled, coloured or
/// sharpened. The caller makes the frame (and its report) of it.
pub(crate) fn crop(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    ask: &LevelAsk,
) -> Result<Option<Cropped>, RenderError> {
    let Some(view_gpu) = job.view_gpu()? else {
        return Ok(None);
    };
    let (x0, y0) = ask.geometry.origin;
    let (dw, dh) = ask.cell;
    let bits = ask.zoom_bits();
    let Some((width, parts)) = job
        .g
        .with_levels(|store| store.cover(bits, &view_gpu.gpu, y0, y0 + dh))
    else {
        return Ok(None);
    };
    let copies = row_copies(&parts, width, (x0, y0), (dw, dh));
    let route = ViewRoute::new(job, view_gpu, spec.gpu_priority);
    let data = CropJob {
        copies,
        bytes: 4 * (dw * dh) as u64,
    };
    let result = route.run(data, device::crop_job);
    let rgba = match result {
        Ok(rgba) => rgba,
        Err(ym_gpu::device::GpuError::Cancelled) => return Err(RenderError::Cancelled),
        // A device that cannot copy rows renders the window as before; the next frame tries again. One that has no room for
        // the frame gives its levels up.
        Err(e) => {
            if matches!(
                e,
                ym_gpu::device::GpuError::OutOfBudget { .. }
                    | ym_gpu::device::GpuError::BufferTooLarge { .. }
            ) {
                job.g.with_levels(LevelStore::clear);
            }
            tracing::debug!(error = %e, "a level could not be cropped; the window is rendered");
            return Ok(None);
        }
    };
    let Rgba8::Device(buffer) = rgba else {
        return Ok(None);
    };
    job.runs.level_hits += 1;
    Ok(Some((buffer, route.gpu_ref())))
}

// ---------------------------------------------------------------------------------------------
// The draft of a zoom
// ---------------------------------------------------------------------------------------------

/// The most a draft gathers into one buffer: the part of the level the view covers.
const DRAFT_REGION_BYTES: u64 = 128 << 20;

/// How far apart, as a factor, the zoom of a level and the zoom it drafts may be.
const DRAFT_MAX_RATIO: f64 = 8.0;

impl LevelStore {
    /// The complete level on `gpu` nearest in zoom to `zoom` (in the ratio of the zoom, a larger one before a smaller at a tie:
    /// shrinking a picture is sharper than growing it), not the level of `zoom` itself: `(zoom bits, size)`.
    fn nearest(&self, gpu: &Arc<Gpu>, zoom: f64) -> Option<(u64, (usize, usize))> {
        self.levels
            .iter()
            .filter(|l| l.complete() && Weak::ptr_eq(&l.gpu, &Arc::downgrade(gpu)))
            .filter(|l| l.zoom_bits != zoom.to_bits())
            .map(|l| (f64::from_bits(l.zoom_bits), l))
            .filter(|(z, _)| (1.0 / DRAFT_MAX_RATIO..=DRAFT_MAX_RATIO).contains(&(z / zoom)))
            .min_by(|(a, _), (b, _)| {
                let (da, db) = ((a / zoom).ln().abs(), (b / zoom).ln().abs());
                da.total_cmp(&db).then(b.total_cmp(a))
            })
            .map(|(_, l)| (l.zoom_bits, l.size))
    }
}

/// A draft of the view `spec`: the picture of a whole-image level of a nearby zoom (any complete one of the photo's settings),
/// resampled to this view in one device job and shown at once; the caller marks it not sharp, so that the exact frame is
/// asked for next. Only for the first request of a round (`coarse_only`: the shell asks again for the sharp frame when this
/// one is not), for a view wholly inside the photo (a letterbox is painted by the exact frame), and only when the exact frame
/// is not in the frame cache.
pub(crate) fn draft(
    job: &mut Job<'_>,
    spec: &ViewSpec,
    image: (usize, usize),
    mask_view: bool,
) -> Result<Option<Cropped>, RenderError> {
    if !job.options().levels
        || spec.output != FrameOutput::Device
        || !spec.cache
        || !spec.coarse_only
        || spec.before
        || mask_view
    {
        return Ok(None);
    }
    // A view that is a scale and a translation, wholly inside the photo.
    let m = &spec.view_to_final.m;
    let step = m[0][0];
    if !(step.is_finite() && step > 0.0)
        || (m[1][1] - step).abs() > 1e-12
        || m[0][1] != 0.0
        || m[1][0] != 0.0
    {
        return Ok(None);
    }
    let (ox, oy) = (m[0][2], m[1][2]);
    let (ex, ey) = (ox + step * spec.out_w as f64, oy + step * spec.out_h as f64);
    const EDGE: f64 = 1e-6;
    if ox < -EDGE || oy < -EDGE || ex > image.0 as f64 + EDGE || ey > image.1 as f64 + EDGE {
        return Ok(None);
    }
    let Some(view_gpu) = job.view_gpu()? else {
        return Ok(None);
    };
    let grade = job.grade(spec.target, true)?;
    let sharpening = view_sharpening(job, spec)?;
    let key = settings_key(job, &grade, sharpening.is_some())?;
    // The exact frame, if it has been made, is better and as cheap.
    for coarse_only in [true, false] {
        let seen = ViewSpec {
            coarse_only,
            ..spec.clone()
        };
        let key = super::frame_key(job, &seen, &grade, sharpening.as_ref())?;
        if job.g.cached_frame(&key).is_some_and(|f| f.sharp) {
            return Ok(None);
        }
    }
    let zoom = 1.0 / step;
    let gpu = Arc::clone(&view_gpu.gpu);
    let Some((bits, size)) = job.g.with_levels(|s| {
        (s.settings_key == key)
            .then(|| s.nearest(&gpu, zoom))
            .flatten()
    }) else {
        return Ok(None);
    };
    // The part of the level the view covers, and the map from the frame's pixels to it.
    let z0 = f64::from_bits(bits);
    let (tx, ty) = (ox * z0, oy * z0);
    let scale = step * z0;
    let span = |t: f64, n: usize, limit: usize| {
        let lo = ((t.floor() as i64) - 1).clamp(0, limit as i64) as usize;
        let hi = (((t + scale * n as f64).ceil() as i64) + 1).clamp(0, limit as i64) as usize;
        (lo, hi)
    };
    let ((rx0, rx1), (ry0, ry1)) = (span(tx, spec.out_w, size.0), span(ty, spec.out_h, size.1));
    if rx1 <= rx0 || ry1 <= ry0 {
        return Ok(None);
    }
    let region = (rx1 - rx0, ry1 - ry0);
    let route = ViewRoute::new(job, view_gpu, spec.gpu_priority);
    if 4 * (region.0 * region.1) as u64 > route.frame_bytes.min(DRAFT_REGION_BYTES) {
        return Ok(None);
    }
    let Some((width, parts)) = job.g.with_levels(|s| s.cover(bits, &gpu, ry0, ry1)) else {
        return Ok(None);
    };
    let data = DraftJob {
        copies: row_copies(&parts, width, (rx0, ry0), region),
        region,
        map: DraftMap {
            sx: scale as f32,
            tx: (tx - rx0 as f64) as f32,
            sy: scale as f32,
            ty: (ty - ry0 as f64) as f32,
        },
        size: (spec.out_w, spec.out_h),
    };
    let rgba = match route.run(data, device::draft_job) {
        Ok(rgba) => rgba,
        Err(ym_gpu::device::GpuError::Cancelled) => return Err(RenderError::Cancelled),
        Err(e) => {
            tracing::debug!(error = %e, "a zoom could not be drafted; the exact frame is rendered");
            return Ok(None);
        }
    };
    let Rgba8::Device(buffer) = rgba else {
        return Ok(None);
    };
    job.runs.level_drafts += 1;
    Ok(Some((buffer, route.gpu_ref())))
}

// ---------------------------------------------------------------------------------------------
// Filling
// ---------------------------------------------------------------------------------------------

/// The full-resolution raw tiles a fill job makes before it renders a band that needs them: a band of a level at 50 % or more
/// reads the photo at full resolution, and making every tile of the photo inside one job would hold up the viewport for as long
/// as it takes to demosaic the photo. Each job makes this many, and the band is rendered by the job that finds them all made.
const WARM_TILES_PER_STEP: usize = 1;

/// Makes up to [`WARM_TILES_PER_STEP`] of the raw tiles the pass `read` samples that are not made yet; `true` when it made any.
fn warm_step(job: &mut Job<'_>, read: &ViewSpec) -> Result<bool, RenderError> {
    let plan = job.plan()?;
    if plan.reduced || plan.effective_scale(read.k) != 1 {
        return Ok(false);
    }
    let grid = job
        .geometry()?
        .at_full_resolution()
        .grid_from((plan.out_w, plan.out_h), read.view_to_final);
    let Some((x0, y0, x1, y1)) = grid.source_bounds(0, 0, read.out_w, read.out_h, 3) else {
        return Ok(false);
    };
    let rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
    let t = job.options().tile_px.max(64);
    let mut made = 0;
    for ty in rect.y / t..=(rect.y1() - 1) / t {
        for tx in rect.x / t..=(rect.x1() - 1) / t {
            let cell = rect.intersect(&Rect::new(tx * t, ty * t, t, t));
            if cell.w == 0 || cell.h == 0 || job.lens_tiles_ready(cell)? {
                continue;
            }
            job.check()?;
            job.lens_warm_tiles(cell)?;
            made += 1;
            if made == WARM_TILES_PER_STEP {
                return Ok(true);
            }
        }
    }
    Ok(made > 0)
}

/// What a fill job came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// A band is made and more are to come.
    More,
    /// A few raw tiles are made for the next band, which is not rendered yet.
    Warm,
    /// The level is complete.
    Done,
    /// The fill ended (cancelled, stale, or the device cannot make the level).
    Stop,
}

/// The view that is the whole level of `ask`.
fn level_spec(r: &Renderer, ask: &LevelAsk) -> ViewSpec {
    let z = ask.geometry.zoom;
    ViewSpec {
        out_w: ask.geometry.size.0,
        out_h: ask.geometry.size.1,
        view_to_final: Affine::scale_translate(1.0 / z, 1.0 / z, 0.0, 0.0),
        k: scale_for_zoom(z),
        target: OutputTarget::Viewport,
        before: false,
        coarse_only: false,
        cache: false,
        background: r.options().background,
        gpu_priority: Priority::Fill,
        dither_origin: (0, 0),
        output: FrameOutput::Device,
    }
}

/// Makes the next band of the fill of `ticket`: one scheduler job's worth.
pub(crate) fn fill_step(
    r: &Renderer,
    src: &PhotoSource,
    ticket: &FillTicket,
    token: &CancelToken,
) -> Step {
    match fill_band(r, src, ticket, token) {
        Ok(step) => step,
        Err(RenderError::Cancelled) => Step::Stop,
        Err(e) => {
            tracing::debug!(error = %e, "a level fill stopped");
            Step::Stop
        }
    }
}

fn fill_band(
    r: &Renderer,
    src: &PhotoSource,
    ticket: &FillTicket,
    token: &CancelToken,
) -> Result<Step, RenderError> {
    let ask = &ticket.ask;
    let Some(graph) = r.existing_graph(src.id) else {
        return Ok(Step::Stop);
    };
    let alive = |g: &PhotoGraph| g.with_levels(|s| s.fill_is(ticket.id));
    if !alive(&graph) {
        return Ok(Step::Stop);
    }
    let mut job = Job::new(r, src, token)?.with_gpu_priority(Priority::Fill);
    fn end(job: &Job<'_>, ticket: &FillTicket, refuse: bool) -> Result<Step, RenderError> {
        job.g
            .with_levels(|s| s.end_fill(ticket.id, refuse.then(|| ticket.ask.zoom_bits())));
        Ok(Step::Stop)
    }
    let Some(view_gpu) = job.view_gpu()? else {
        return end(&job, ticket, false);
    };
    let spec = level_spec(r, ask);
    let grade = job.grade(spec.target, true)?;
    let sharpening = view_sharpening(&mut job, &spec)?;
    // The settings the ticket was made for are still the photo's (a plane that arrived since is a new key).
    if settings_key(&mut job, &grade, sharpening.is_some())? != ask.settings_key {
        return end(&job, ticket, false);
    }
    let stages = ViewStages::of(view_gpu.stages);
    if !(stages.color && (sharpening.is_none() || stages.tail)) {
        return end(&job, ticket, true);
    }
    let gpu = Arc::clone(&view_gpu.gpu);
    let route = ViewRoute::new(&job, view_gpu, Priority::Fill);
    let image = job.final_size()?;

    // The cut, found by the first job of the fill.
    let next = job.g.with_levels(|s| s.next_band(ticket.id));
    let (index, plan) = match next {
        Some(next) => next,
        None => {
            let already = job.g.with_levels(|s| {
                s.fill
                    .as_ref()
                    .is_some_and(|f| f.id == ticket.id && f.plan.is_some())
            });
            if already {
                return Ok(Step::Done);
            }
            let chains: [&ym_color::display::ColorChain; 1] = [&grade.chain];
            let shape = band_split(&mut job, &spec, sharpening.as_ref(), &route, &chains)?;
            tracing::debug!(?shape, level = %format!("{}x{}", spec.out_w, spec.out_h), "the cut of a level");
            let rows = match shape {
                Some(Shape::Rows(rows)) => rows,
                Some(Shape::Whole) => spec.out_h,
                // Tiles, or not even those: the window path makes these views.
                Some(Shape::Tiles(..)) | None => return end(&job, ticket, true),
            };
            let rows = rows.min((FILL_BAND_PX / spec.out_w.max(1)).max(1)).max(1);
            let whole_box = sharpening
                .as_ref()
                .map(|sh| whole_window(&mut job, &spec, sh))
                .transpose()?;
            let frame_of = |v: &ViewSpec| {
                sharpening
                    .as_ref()
                    .and_then(|sh| sh.frame_geometry(&v.view_to_final, (v.out_w, v.out_h), image))
            };
            let plan = FillPlan {
                bands: (0..spec.out_h)
                    .step_by(rows)
                    .map(|y| (y, rows.min(spec.out_h - y)))
                    .collect(),
                whole_box,
                whole_origin: frame_of(&spec).and_then(|g| g.origin),
                halo: sharpening.as_ref().map_or(0, |s| s.halo),
            };
            // The rows on screen go first: the middle of the window the level was asked for.
            let centre = ask.geometry.origin.1 + ask.cell.1 / 2;
            if !job
                .g
                .with_levels(|s| s.set_plan(ticket, plan, &gpu, centre))
            {
                return end(&job, ticket, true);
            }
            match job.g.with_levels(|s| s.next_band(ticket.id)) {
                Some(next) => next,
                None => return Ok(Step::Done),
            }
        }
    };

    job.check()?;
    let (row0, rows) = plan.bands[index];
    let band = ViewSpec {
        out_h: rows,
        view_to_final: Affine::scale_translate(1.0, 1.0, 0.0, row0 as f64)
            .then(&spec.view_to_final),
        dither_origin: (spec.dither_origin.0, spec.dither_origin.1 + row0),
        ..spec.clone()
    };
    let frame_of = |v: &ViewSpec| {
        sharpening
            .as_ref()
            .and_then(|sh| sh.frame_geometry(&v.view_to_final, (v.out_w, v.out_h), image))
    };
    let window = match (plan.whole_box, plan.whole_origin) {
        (Some(b), Some(origin)) => {
            StageWindow::Of(magnified_band_window(b, origin, frame_of(&band).as_ref()))
        }
        (Some(b), None) => StageWindow::Of(band_window(b, row0, rows, plan.halo)),
        (None, _) => StageWindow::Own,
    };
    // The raw tiles the band reads (its frame, grown by the Film nodes' reach), a few at a time.
    let read = match frame_of(&band) {
        Some(g) => ViewSpec {
            out_w: g.width,
            out_h: g.height,
            view_to_final: g.view_to_final,
            ..band.clone()
        },
        None => band.clone(),
    };
    let read = raster_grown(&mut job, &read, &[&grade.chain])?.unwrap_or(read);
    if warm_step(&mut job, &read)? {
        job.finish(1, ComputeDevice::Gpu, spec.target);
        return Ok(Step::Warm);
    }
    let done = match whole(
        &mut job,
        &band,
        &grade,
        sharpening.as_ref(),
        Some(&route),
        window,
    )? {
        Flow::Done(done) if done.device == ComputeDevice::Gpu || done.paper => done,
        // A band the CPU made: a level is never made of two devices.
        Flow::Done(_) | Flow::Cpu => return end(&job, ticket, true),
    };
    let buffer = match done.rgba {
        Rgba8::Device(buffer) => buffer,
        // A band of paper only is painted on the host: it goes up as it is.
        Rgba8::Host(bytes) => {
            let data = GatherJob {
                bytes: bytes.len() as u64,
                parts: vec![Rgba8::Host(bytes)],
            };
            match route.run(data, device::gather_job) {
                Ok(Rgba8::Device(buffer)) => buffer,
                Err(ym_gpu::device::GpuError::Cancelled) => return Err(RenderError::Cancelled),
                _ => return end(&job, ticket, true),
            }
        }
    };
    job.runs.level_bands += 1;
    let more = job.g.with_levels(|s| s.put(ticket.id, index, buffer));
    let graph = Arc::clone(&job.g);
    // The band's stages are counted in the photo's totals like any request's.
    job.finish(done.scale, ComputeDevice::Gpu, spec.target);
    let _ = graph;
    Ok(if more { Step::More } else { Step::Done })
}

impl Renderer {
    /// Records that a viewport frame of `src` had no level to be served from; the level is filled in the background once the
    /// key has held still for [`LEVEL_DEBOUNCE`].
    pub(crate) fn want_level(&self, src: &PhotoSource, ask: LevelAsk) {
        let Some(graph) = self.existing_graph(src.id) else {
            return;
        };
        let deadline = Instant::now() + LEVEL_DEBOUNCE;
        if !graph.with_levels(|s| s.want(ask, src, deadline)) {
            return;
        }
        let r = self.clone();
        let spawned = std::thread::Builder::new()
            .name("ym-level-wait".to_owned())
            .spawn(move || r.level_timer(&graph));
        if spawned.is_err() {
            // No thread, no fill; the window path serves as before.
            if let Some(graph) = self.existing_graph(src.id) {
                graph.with_levels(|s| {
                    s.timer = false;
                    s.want = None;
                });
            }
        }
    }

    /// The wait of [`Renderer::want_level`]: sleeps until the wanted level's deadline, then starts its fill.
    fn level_timer(&self, graph: &Arc<PhotoGraph>) {
        enum Next {
            End,
            Sleep(Duration),
            Fire(Box<(FillTicket, PhotoSource)>),
        }
        loop {
            let next = graph.with_levels(|s| match s.want.take() {
                None => {
                    s.timer = false;
                    Next::End
                }
                Some(w) => {
                    let now = Instant::now();
                    if w.deadline > now {
                        let wait = w.deadline - now;
                        s.want = Some(w);
                        Next::Sleep(wait)
                    } else {
                        s.timer = false;
                        match s.start_fill(&w) {
                            Some(ticket) => Next::Fire(Box::new((ticket, w.src))),
                            None => Next::End,
                        }
                    }
                }
            });
            match next {
                Next::End => return,
                Next::Sleep(d) => std::thread::sleep(d.min(Duration::from_millis(50))),
                Next::Fire(fire) => {
                    let (ticket, src) = *fire;
                    self.submit_fill(src, ticket);
                    return;
                }
            }
        }
    }

    /// Queues the next band of a fill at [`Priority::Fill`]; the job queues the one after it.
    fn submit_fill(&self, src: PhotoSource, ticket: FillTicket) {
        let r = self.clone();
        let id = src.id;
        let again = (src.clone(), ticket.clone());
        let handle = self
            .scheduler()
            .submit(Priority::Fill, Some(id), move |token| {
                let step = fill_step(&r, &src, &ticket, token);
                match step {
                    Step::More | Step::Warm => r.submit_fill(src, ticket),
                    Step::Done | Step::Stop => {
                        if let Some(g) = r.existing_graph(src.id) {
                            g.with_levels(|s| {
                                if step == Step::Stop {
                                    s.end_fill(ticket.id, None);
                                }
                            });
                        }
                    }
                }
            });
        if let Some(g) = self.existing_graph(again.0.id) {
            g.with_levels(|s| s.set_token(again.1.id, handle.token().clone()));
        }
    }

    /// Fills the level `ask` names on the calling thread, at most `steps` bands (tests, and a caller that wants the level now).
    /// A level left part made is kept as far as it got.
    pub(crate) fn fill_level_now(
        &self,
        src: &PhotoSource,
        ask: &LevelAsk,
        token: &CancelToken,
        steps: usize,
    ) -> Step {
        let Some(graph) = self.existing_graph(src.id) else {
            return Step::Stop;
        };
        let want = Want {
            ask: ask.clone(),
            src: src.clone(),
            deadline: Instant::now(),
        };
        let Some(ticket) = graph.with_levels(|s| s.start_fill(&want)) else {
            return Step::Stop;
        };
        let mut last = Step::More;
        let mut bands = 0;
        while bands < steps {
            last = fill_step(self, src, &ticket, token);
            match last {
                Step::More => bands += 1,
                Step::Warm => {}
                Step::Done | Step::Stop => break,
            }
        }
        graph.with_levels(|s| {
            if last != Step::Done {
                s.end_fill(ticket.id, None);
            }
        });
        last
    }
}
