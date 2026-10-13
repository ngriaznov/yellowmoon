//! The render graph: the stage groups of `docs/01` with their caches, keyed by the inputs of each
//! group so that a settings change recomputes only the groups it reaches.
//!
//! A [`Renderer`] holds one [`PhotoGraph`] per photo (the most recently used few, see
//! [`MemoryBudget::max_photos`]). Everything that computes goes through a `Job`, which bundles the
//! renderer, the photo's graph, the settings snapshot ([`PhotoSource`]) and a cancel token, and
//! counts what ran so that [`RenderReport::runs`] can say, for one request, which groups did work.
//!
//! | Cache | Holds | Key |
//! |---|---|---|
//! | `frame` | the area copied out of the readout | the photo |
//! | `plan` | levels, vignetting profile, lens maps and their auto crop | `lens_corrections`, `chromatic_aberration` |
//! | `sigma` | the half-resolution noise sigma | the noise model and the gains `g0` |
//! | `groups` | the raw group's image at 1/k | [`RawParams`] (all of `decode.*`, noise reduction, sharpening, `g0`, noise) and k |
//! | `full` | the prepared full-resolution chain | [`RawParams`] |
//! | `tiles` | 512 px tiles of the full-resolution raw group | [`RawParams`] and the tile |
//! | `color` | the fused matrix and C7 | [`ColorKey`] |
//! | `grade` | the prepared `ColorChain` | [`GradeKey`] |
//! | `frames` | finished viewport frames | everything above plus geometry and the cell |
//!
//! The estimates' version (`PhotoView::estimates_version`) is deliberately not part of any key: the
//! keys hold the numbers the group reads, so a burst median that moves only the exposure recomputes
//! the colour group and leaves the raw group alone.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Instant;

use serde::Serialize;
use ym_export::ColorSpace;
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::device::{Gpu, GpuInfo};
use ym_gpu::queue::CancelToken;

use super::color::{
    ColorKey, ColorPlan, DisplayChoice, GradeKey, GradePlan, GradeRequest, LookProvider,
    PhotoSource, color_key, color_plan, grade_plan,
};
use super::geometry::Geometry;
use super::raw::{
    FrameData, FullChain, RawParams, RawParamsKey, RawPlan, SigmaPlane, reduced_bayer, reduced_rgb,
    sigma_plane,
};
use super::{FrameZoom, NeuralStatus, RenderError, RenderPath, RenderStats};
use crate::scheduler::{ByteLru, JobScheduler, MemoryBudget, MemoryLedger};
use crate::session::PhotoId;
use ym_color::display::OutputTarget;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a [`Renderer`] needs to know about the machine and the preferences.
#[derive(Clone)]
pub struct RenderOptions {
    /// Threads of the per-pixel colour pass; 0 means every core. (The raw stages use rayon's
    /// global pool; a test or the CLI sets the thread count by installing a pool.)
    pub threads: usize,
    /// Edge of the tiles the 100 % view is made of, in pixels.
    pub tile_px: usize,
    /// Edge of the tiles a full-resolution render works in, in pixels.
    pub full_tile_px: usize,
    /// Memory limits.
    pub budget: MemoryBudget,
    /// Whether the learned demosaic can run.
    pub neural: NeuralStatus,
    /// The GPU, when there is one: the colour pass of viewport frames and thumbnails runs on it.
    pub gpu: Option<Arc<Gpu>>,
    /// The colour of the letterbox around the photo (the paper colour).
    pub background: [u8; 3],
    /// Longest edge of a thumbnail in pixels (`docs/01`: 240).
    pub thumbnail_px: usize,
}

impl std::fmt::Debug for RenderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderOptions")
            .field("threads", &self.threads)
            .field("tile_px", &self.tile_px)
            .field("full_tile_px", &self.full_tile_px)
            .field("budget", &self.budget)
            .field("neural", &self.neural)
            .field("gpu", &self.gpu.as_ref().map(|g| g.info().name.clone()))
            .finish_non_exhaustive()
    }
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            tile_px: 512,
            full_tile_px: 1024,
            budget: MemoryBudget::detect(),
            neural: NeuralStatus::unavailable(),
            gpu: None,
            background: [255, 255, 255],
            thumbnail_px: 240,
        }
    }
}

/// Where the per-pixel colour stages ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDevice {
    /// The CPU reference path.
    Cpu,
    /// The wgpu device.
    Gpu,
}

/// How many times each group did work (for one request, or since the graph was made).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StageRuns {
    /// The area copied out of the readout.
    pub frame_data: u64,
    /// Raw-group images at a reduced scale (R1, R3 to R9 at 1/k).
    pub raw_groups: u64,
    /// Noise sigma planes.
    pub sigma_planes: u64,
    /// Prepared full-resolution chains.
    pub full_chains: u64,
    /// Full-resolution tiles (R1, R3 to R9 over 512 px).
    pub tiles: u64,
    /// Colour plans (C1 to C8).
    pub color_plans: u64,
    /// Grade plans (C9 to C13).
    pub grade_plans: u64,
    /// Per-pixel colour passes over a sampled image.
    pub color_passes: u64,
    /// Finished frames served from the frame cache.
    pub frame_hits: u64,
}

impl StageRuns {
    /// The number of times the raw stages (R1, R3 to R9) did work: groups, chains and tiles.
    #[must_use]
    pub fn raw_work(&self) -> u64 {
        self.raw_groups + self.full_chains + self.tiles
    }

    fn add(&mut self, other: &StageRuns) {
        self.frame_data += other.frame_data;
        self.raw_groups += other.raw_groups;
        self.sigma_planes += other.sigma_planes;
        self.full_chains += other.full_chains;
        self.tiles += other.tiles;
        self.color_plans += other.color_plans;
        self.grade_plans += other.grade_plans;
        self.color_passes += other.color_passes;
        self.frame_hits += other.frame_hits;
    }
}

/// Milliseconds the groups spent on one request (zero for a group that was cached).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct StageTimings {
    /// Copying the area out of the readout.
    pub frame_data_ms: f64,
    /// The noise sigma plane.
    pub sigma_ms: f64,
    /// The raw group at a reduced scale (R1, R3 to R9), or the full-resolution tiles.
    pub raw_ms: f64,
    /// The colour and grade plans (C1 to C13 parameters, curves, LUTs).
    pub plans_ms: f64,
    /// Sampling the raw group through the geometry.
    pub sample_ms: f64,
    /// The per-pixel colour pass (CPU or GPU) and the readback.
    pub pass_ms: f64,
}

/// Running totals of a graph's [`StageRuns`].
#[derive(Debug, Default)]
pub struct StageCounters(Mutex<StageRuns>);

impl StageCounters {
    /// The totals so far.
    #[must_use]
    pub fn snapshot(&self) -> StageRuns {
        *lock(&self.0)
    }

    fn add(&self, runs: &StageRuns) {
        lock(&self.0).add(runs);
    }
}

/// What one request did, for the status text and for tests.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderReport {
    /// Milliseconds the request took.
    pub ms: f64,
    /// The demosaic path.
    pub path: RenderPath,
    /// The output colour space.
    pub space: ColorSpace,
    /// Where the colour pass ran.
    pub device: ComputeDevice,
    /// The reduction of the raw group that was used (1 = full resolution).
    pub scale: usize,
    /// What did work for this request.
    pub runs: StageRuns,
    /// How long each group took (`tracing` logs the same numbers at debug level).
    pub timings: StageTimings,
    /// Plain-words notes: why the classical path ran, a look that was replaced by Neutral, lens
    /// tables that were ignored.
    pub notes: Vec<String>,
    /// The zoom of the frame, for a viewport frame ([`Renderer::viewport_frame`] sets it, also on a
    /// frame served from the cache); `None` for every other request.
    pub zoom: Option<FrameZoom>,
}

impl RenderReport {
    /// The `render_stats` event payload.
    #[must_use]
    pub fn stats(&self) -> RenderStats {
        RenderStats {
            ms: self.ms,
            path: self.path,
            space: self.space,
            zoom_percent: self.zoom.map(|z| z.zoom * 100.0),
            fit_percent: self.zoom.map(|z| z.fit_zoom * 100.0),
        }
    }
}

#[derive(Default)]
struct GraphState {
    /// The fingerprint of the decoded image the caches belong to.
    fingerprint: Option<u64>,
    plan: Option<((bool, bool), Arc<RawPlan>)>,
    sigma: Option<(Vec<u64>, Arc<SigmaPlane>)>,
    groups: Option<ByteLru<(RawParamsKey, usize), Arc<RgbImage>>>,
    full: Option<(RawParamsKey, Arc<FullChain>)>,
    tiles: Option<ByteLru<(RawParamsKey, usize, usize), Arc<RgbImage>>>,
    color: Option<Arc<ColorPlan>>,
    grade: Option<ByteLru<GradeKey, Arc<GradePlan>>>,
    frames: Option<ByteLru<String, Arc<super::frame::RenderedFrame>>>,
    stats: Option<RenderStats>,
}

/// The stage caches of one photo.
pub struct PhotoGraph {
    id: PhotoId,
    state: Mutex<GraphState>,
    counters: StageCounters,
    budget: MemoryBudget,
}

impl std::fmt::Debug for PhotoGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoGraph")
            .field("id", &self.id)
            .field("runs", &self.counters.snapshot())
            .finish_non_exhaustive()
    }
}

impl PhotoGraph {
    fn new(id: PhotoId, budget: MemoryBudget) -> Self {
        Self {
            id,
            state: Mutex::new(GraphState::default()),
            counters: StageCounters::default(),
            budget,
        }
    }

    /// The photo.
    #[must_use]
    pub fn id(&self) -> PhotoId {
        self.id
    }

    /// How many times each group has done work since the graph was made.
    #[must_use]
    pub fn counters(&self) -> StageRuns {
        self.counters.snapshot()
    }

    /// Bytes the caches hold.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let st = lock(&self.state);
        st.sigma.as_ref().map_or(0, |(_, s)| s.bytes())
            + st.groups.as_ref().map_or(0, ByteLru::used)
            + st.full.as_ref().map_or(0, |(_, f)| f.bytes())
            + st.tiles.as_ref().map_or(0, ByteLru::used)
            + st.frames.as_ref().map_or(0, ByteLru::used)
    }

    /// Drops the full-resolution intermediates (tiles and the prepared chain), keeping the small
    /// reduced images: called when the photo is no longer the one being looked at.
    pub fn release_full_resolution(&self) {
        let mut st = lock(&self.state);
        st.full = None;
        st.tiles = None;
    }

    /// Drops every cache unless it was made for the image with this fingerprint.
    fn adopt(&self, fingerprint: u64) {
        let mut st = lock(&self.state);
        if st.fingerprint != Some(fingerprint) {
            *st = GraphState {
                fingerprint: Some(fingerprint),
                ..GraphState::default()
            };
        }
    }

    /// A finished frame with this key, if one is cached.
    pub(crate) fn cached_frame(&self, key: &str) -> Option<Arc<super::frame::RenderedFrame>> {
        let mut st = lock(&self.state);
        st.frames.as_mut()?.get(&key.to_owned()).cloned()
    }

    /// Keeps a finished frame.
    pub(crate) fn store_frame(&self, key: String, frame: Arc<super::frame::RenderedFrame>) {
        let bytes = frame.rgba.len();
        let budget = (self.budget.cpu_bytes / 32) as usize;
        lock(&self.state)
            .frames
            .get_or_insert_with(|| ByteLru::new(budget))
            .insert(key, frame, bytes);
    }

    fn group_budget(&self) -> usize {
        (self.budget.cpu_bytes / 6) as usize
    }

    fn tile_budget(&self) -> usize {
        (self.budget.cpu_bytes / 6) as usize
    }
}

struct Inner {
    /// The photo the user is looking at (for [`Renderer::navigate`]).
    current: Mutex<Option<PhotoId>>,
    looks: Arc<dyn LookProvider>,
    display: RwLock<DisplayChoice>,
    options: RenderOptions,
    graphs: Mutex<Vec<(PhotoId, Arc<PhotoGraph>)>>,
    /// Photos with an id up to this one belong to a session that was closed (0: none).
    retired_through: AtomicU32,
    scheduler: JobScheduler,
    ledger: Arc<MemoryLedger>,
}

/// The render engine (see the module documentation of [`crate::pipeline`]). Cheap to clone.
#[derive(Clone)]
pub struct Renderer {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl Renderer {
    /// A renderer that takes its looks from `looks`.
    ///
    /// # Errors
    ///
    /// [`RenderError::Thread`] when the operating system cannot start the render thread.
    pub fn new(looks: Arc<dyn LookProvider>, options: RenderOptions) -> Result<Self, RenderError> {
        let scheduler = JobScheduler::new("yellowmoon-render").map_err(|e| {
            tracing::error!(error = %e, "the render thread could not start");
            RenderError::Thread(e.to_string())
        })?;
        let ledger = MemoryLedger::new(options.budget.cpu_bytes);
        Ok(Self {
            inner: Arc::new(Inner {
                current: Mutex::new(None),
                looks,
                display: RwLock::new(DisplayChoice::Aces2),
                options,
                graphs: Mutex::new(Vec::new()),
                retired_through: AtomicU32::new(0),
                scheduler,
                ledger,
            }),
        })
    }

    /// A renderer with only the Neutral look and default options (tests and tools).
    ///
    /// # Errors
    ///
    /// [`RenderError::Thread`] when the operating system cannot start the render thread.
    pub fn neutral_only() -> Result<Self, RenderError> {
        Self::new(
            Arc::new(super::color::NeutralOnly),
            RenderOptions::default(),
        )
    }

    /// The options.
    #[must_use]
    pub fn options(&self) -> &RenderOptions {
        &self.inner.options
    }

    /// The job scheduler (priorities, navigation).
    #[must_use]
    pub fn scheduler(&self) -> &JobScheduler {
        &self.inner.scheduler
    }

    /// The memory ledger of full-resolution jobs.
    #[must_use]
    pub fn ledger(&self) -> &Arc<MemoryLedger> {
        &self.inner.ledger
    }

    /// Where looks come from.
    #[must_use]
    pub fn looks(&self) -> &Arc<dyn LookProvider> {
        &self.inner.looks
    }

    /// Chooses the display rendering (Preferences). Cached grade plans of the other choice stop
    /// matching and are rebuilt on the next request.
    pub fn set_display(&self, choice: DisplayChoice) {
        *self
            .inner
            .display
            .write()
            .unwrap_or_else(PoisonError::into_inner) = choice;
    }

    /// The display rendering in effect.
    #[must_use]
    pub fn display(&self) -> DisplayChoice {
        self.inner
            .display
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The GPU in use, if any.
    #[must_use]
    pub fn gpu_info(&self) -> Option<GpuInfo> {
        self.inner.options.gpu.as_ref().map(|g| g.info().clone())
    }

    /// The graph of a photo, made on first use. The least recently used graphs beyond
    /// [`MemoryBudget::max_photos`] are dropped, except the graph of the photo being looked at
    /// ([`Renderer::navigate`]): the thumbnails of a folder that is still developing touch every
    /// photo in turn and would otherwise push the current photo's images out one thumbnail at a
    /// time.
    #[must_use]
    pub fn graph(&self, id: PhotoId) -> Arc<PhotoGraph> {
        let mut graphs = lock(&self.inner.graphs);
        if let Some(i) = graphs.iter().position(|(p, _)| *p == id) {
            let e = graphs.remove(i);
            graphs.push(e);
        } else {
            let g = Arc::new(PhotoGraph::new(id, self.inner.options.budget));
            graphs.push((id, g));
        }
        let max = self.inner.options.budget.max_photos.max(1);
        if graphs.len() > max {
            let current = *lock(&self.inner.current);
            while graphs.len() > max {
                // The oldest graph that is neither the one asked for (the last) nor the current.
                let last = graphs.len() - 1;
                let Some(victim) = graphs[..last].iter().position(|(p, _)| Some(*p) != current)
                else {
                    break;
                };
                graphs.remove(victim);
            }
        }
        Arc::clone(&graphs[graphs.len() - 1].1)
    }

    /// Records the photo being looked at and returns the one it replaced, when that was another.
    pub(crate) fn note_current(&self, photo: PhotoId) -> Option<PhotoId> {
        let mut cur = lock(&self.inner.current);
        let old = cur.replace(photo);
        old.filter(|o| *o != photo)
    }

    /// The graph of a photo if it has one (without making one).
    pub(crate) fn existing_graph(&self, id: PhotoId) -> Option<Arc<PhotoGraph>> {
        lock(&self.inner.graphs)
            .iter()
            .find(|(p, _)| *p == id)
            .map(|(_, g)| Arc::clone(g))
    }

    /// Forgets a photo's caches (it was closed).
    pub fn forget(&self, id: PhotoId) {
        lock(&self.inner.graphs).retain(|(p, _)| *p != id);
    }

    /// Forgets everything (the session closed).
    pub fn forget_all(&self) {
        lock(&self.inner.graphs).clear();
    }

    /// The session closed: forgets every photo's caches, as [`Renderer::forget_all`] does, and
    /// from now on refuses viewport, tile, thumbnail, histogram and readout requests for the
    /// photos up to `last`, the highest id the closed session handed out (ids keep counting across
    /// sessions, so a photo of the next session is above it). A request that was still on its way
    /// when the session closed (a command that had read the photo before the close and queued its
    /// job after the scheduler's cancel) would otherwise run, and make the caches of a photo that
    /// is gone. **Full-resolution renders for exports are not refused**: they hold their own
    /// snapshots and go on after the close (`docs/11`).
    pub fn retire_through(&self, last: PhotoId) {
        self.inner
            .retired_through
            .fetch_max(last.0, Ordering::SeqCst);
        self.forget_all();
    }

    /// Whether a photo belongs to a session that was closed ([`Renderer::retire_through`]).
    #[must_use]
    pub fn is_retired(&self, id: PhotoId) -> bool {
        id.0 <= self.inner.retired_through.load(Ordering::SeqCst)
    }

    /// The runs of a photo's graph so far (zero for a photo with no graph).
    #[must_use]
    pub fn counters(&self, id: PhotoId) -> StageRuns {
        lock(&self.inner.graphs)
            .iter()
            .find(|(p, _)| *p == id)
            .map(|(_, g)| g.counters())
            .unwrap_or_default()
    }

    /// Bytes held by all graphs.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        lock(&self.inner.graphs)
            .iter()
            .map(|(_, g)| g.bytes())
            .sum()
    }

    /// The `render_stats` of the last frame rendered for a photo.
    #[must_use]
    pub fn last_stats(&self, id: PhotoId) -> Option<RenderStats> {
        let g = lock(&self.inner.graphs)
            .iter()
            .find(|(p, _)| *p == id)
            .map(|(_, g)| Arc::clone(g))?;
        lock(&g.state).stats
    }

    pub(crate) fn note_stats(&self, id: PhotoId, stats: RenderStats) {
        if self.is_retired(id) {
            return;
        }
        let g = self.graph(id);
        lock(&g.state).stats = Some(stats);
    }
}

/// One request: the renderer, the photo's graph, the settings snapshot and the bookkeeping.
pub(crate) struct Job<'a> {
    pub r: &'a Renderer,
    pub g: Arc<PhotoGraph>,
    pub src: &'a PhotoSource,
    pub cancel: &'a CancelToken,
    pub runs: StageRuns,
    pub timings: StageTimings,
    pub notes: Vec<String>,
    started: Instant,
    params: Option<RawParams>,
    /// The decoded area of this request (see [`Job::frame`]).
    frame: Option<Arc<FrameData>>,
    /// A full-resolution chain built for this request only (`keep = false`), so that a render
    /// that works tile by tile prepares it once.
    transient_full: Option<(RawParamsKey, Arc<FullChain>)>,
}

impl<'a> Job<'a> {
    /// A request for the viewport, a tile, a thumbnail, the histogram or a readout. A photo of a
    /// closed session ([`Renderer::retire_through`]) is refused as cancelled.
    pub fn new(
        r: &'a Renderer,
        src: &'a PhotoSource,
        cancel: &'a CancelToken,
    ) -> Result<Self, RenderError> {
        if r.is_retired(src.id) {
            return Err(RenderError::Cancelled);
        }
        Ok(Self::for_export(r, src, cancel))
    }

    /// A request that is not refused for a photo of a closed session: an export, which goes on
    /// with its snapshot.
    pub fn for_export(r: &'a Renderer, src: &'a PhotoSource, cancel: &'a CancelToken) -> Self {
        let g = r.graph(src.id);
        g.adopt(src.fingerprint);
        Self {
            r,
            g,
            src,
            cancel,
            runs: StageRuns::default(),
            timings: StageTimings::default(),
            notes: Vec::new(),
            started: Instant::now(),
            params: None,
            frame: None,
            transient_full: None,
        }
    }

    pub fn check(&self) -> Result<(), RenderError> {
        self.cancel.check().map_err(|_| RenderError::Cancelled)
    }

    /// Adds a plain-words note to this request's report, once.
    fn note(&mut self, text: &str) {
        if !self.notes.iter().any(|n| n == text) {
            self.notes.push(text.to_owned());
        }
    }

    /// Adds this request's runs to the graph's totals and builds the report.
    pub fn finish(
        mut self,
        scale: usize,
        device: ComputeDevice,
        target: OutputTarget,
    ) -> RenderReport {
        self.g.counters.add(&self.runs);
        let neural = &self.r.inner.options.neural;
        if self.src.settings.decode.neural_demosaic
            && !neural.available
            && let Some(reason) = &neural.reason
        {
            let reason = reason.clone();
            self.note(&reason);
        }
        let report = RenderReport {
            ms: self.started.elapsed().as_secs_f64() * 1000.0,
            path: RenderPath::Classical,
            space: color_space_of(target),
            device,
            scale,
            runs: self.runs,
            timings: self.timings,
            notes: self.notes,
            zoom: None,
        };
        tracing::debug!(
            photo = %self.src.id,
            ms = report.ms,
            scale = report.scale,
            runs = ?report.runs,
            timings = ?report.timings,
            "render"
        );
        report
    }

    /// The numbers the raw group reads from the settings and the estimates.
    pub fn params(&mut self) -> Result<RawParams, RenderError> {
        if let Some(p) = &self.params {
            return Ok(*p);
        }
        let s = &self.src.settings;
        let e = &self.src.estimates;
        let g0 = e
            .initial_gains()
            .map_err(|err| RenderError::Color(err.to_string()))?;
        let nr = match s.noise_reduction.manual() {
            Some(m) => *m,
            None => e.noise_reduction,
        };
        let sh = match s.sharpen.manual() {
            Some(m) => *m,
            None => e.sharpen,
        };
        let iso = self
            .src
            .raw
            .meta
            .iso
            .filter(|i| *i > 0)
            .map_or(100.0, |i| i as f32);
        let p = RawParams {
            g0: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
            neural_demosaic: s.decode.neural_demosaic,
            highlight_recovery: s.decode.highlight_recovery,
            lens_corrections: s.decode.lens_corrections,
            chromatic_aberration: s.decode.chromatic_aberration,
            noise_reduction: NoiseReduction {
                luminance: nr.luminance as f32,
                color: nr.color as f32,
                detail: nr.detail as f32,
            },
            sharpen: SharpenParams {
                amount: sh.amount as f32,
                radius_px: sh.radius_px as f32,
            },
            iso,
            noise: e.noise.model,
        };
        self.params = Some(p);
        Ok(p)
    }

    /// The decode group: the photo's area, cut out of the readout once per request and dropped
    /// with it. It is not cached in the graph: a copy kept per photo would hold a second decoded
    /// mosaic beyond the three the session keeps (`docs/01`: at most three decoded mosaics).
    pub fn frame(&mut self) -> Result<Arc<FrameData>, RenderError> {
        if let Some(f) = &self.frame {
            return Ok(Arc::clone(f));
        }
        let t = Instant::now();
        let frame = Arc::new(FrameData::new(&self.src.raw)?);
        self.runs.frame_data += 1;
        self.timings.frame_data_ms += t.elapsed().as_secs_f64() * 1000.0;
        self.frame = Some(Arc::clone(&frame));
        Ok(frame)
    }

    /// The plan for the lens switches in effect.
    pub fn plan(&mut self) -> Result<Arc<RawPlan>, RenderError> {
        let params = self.params()?;
        let key = (params.lens_corrections, params.chromatic_aberration);
        let cached = lock(&self.g.state)
            .plan
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, p)| Arc::clone(p));
        if let Some(plan) = cached {
            // The notes belong to every request that uses the plan, not only the one that made it.
            for n in &plan.notes {
                self.note(n);
            }
            return Ok(plan);
        }
        let frame = self.frame()?;
        let plan = Arc::new(RawPlan::new(&self.src.raw, &frame, &params)?);
        for n in &plan.notes {
            self.note(n);
        }
        lock(&self.g.state).plan = Some((key, Arc::clone(&plan)));
        Ok(plan)
    }

    /// The noise sigma plane (mosaic photos).
    pub fn sigma(&mut self) -> Result<Arc<SigmaPlane>, RenderError> {
        let params = self.params()?;
        let key = params.sigma_key();
        if let Some((k, s)) = &lock(&self.g.state).sigma
            && *k == key
        {
            return Ok(Arc::clone(s));
        }
        let frame = self.frame()?;
        let plan = self.plan()?;
        let t = Instant::now();
        let sigma = Arc::new(sigma_plane(&frame, &plan, &params, self.cancel)?);
        self.runs.sigma_planes += 1;
        self.timings.sigma_ms += t.elapsed().as_secs_f64() * 1000.0;
        lock(&self.g.state).sigma = Some((key, Arc::clone(&sigma)));
        Ok(sigma)
    }

    /// The raw group at 1/`k` as an image. Mosaic photos need `k >= 2` (full resolution goes
    /// through [`Job::region`]); camera-processed photos accept any `k`.
    pub fn group(&mut self, k: usize) -> Result<Arc<RgbImage>, RenderError> {
        let params = self.params()?;
        let key = (params.key(), k);
        {
            let mut st = lock(&self.g.state);
            let groups = st
                .groups
                .get_or_insert_with(|| ByteLru::new(self.g.group_budget()));
            if let Some(img) = groups.get(&key) {
                return Ok(Arc::clone(img));
            }
        }
        let frame = self.frame()?;
        let plan = self.plan()?;
        let img = if frame.reduced {
            let t = Instant::now();
            let img = reduced_rgb(&frame, &plan, &params, k, self.cancel)?;
            self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
            img
        } else {
            debug_assert!(k >= 2, "full resolution of a mosaic goes through tiles");
            let sigma = self.sigma()?;
            let t = Instant::now();
            let img = reduced_bayer(&frame, &plan, &params, &sigma, k, self.cancel)?;
            self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
            img
        };
        self.runs.raw_groups += 1;
        let img = Arc::new(img);
        let bytes = img.data.len() * 4;
        lock(&self.g.state)
            .groups
            .get_or_insert_with(|| ByteLru::new(self.g.group_budget()))
            .insert(key, Arc::clone(&img), bytes);
        Ok(img)
    }

    /// The prepared full-resolution chain (mosaic photos).
    pub fn full(&mut self, keep: bool) -> Result<Arc<FullChain>, RenderError> {
        let params = self.params()?;
        let key = params.key();
        if let Some((k, f)) = &lock(&self.g.state).full
            && *k == key
        {
            return Ok(Arc::clone(f));
        }
        if let Some((k, f)) = &self.transient_full
            && *k == key
        {
            return Ok(Arc::clone(f));
        }
        let frame = self.frame()?;
        let plan = self.plan()?;
        let sigma = self.sigma()?;
        let t = Instant::now();
        let full = Arc::new(FullChain::new(&frame, &plan, &params, &sigma, self.cancel)?);
        self.runs.full_chains += 1;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        if keep {
            lock(&self.g.state).full = Some((key, Arc::clone(&full)));
        } else {
            self.transient_full = Some((key, Arc::clone(&full)));
        }
        Ok(full)
    }

    /// One 512 px tile of the full-resolution raw group.
    fn tile(&mut self, tx: usize, ty: usize, keep: bool) -> Result<Arc<RgbImage>, RenderError> {
        let params = self.params()?;
        let key = (params.key(), tx, ty);
        if keep {
            let mut st = lock(&self.g.state);
            let tiles = st
                .tiles
                .get_or_insert_with(|| ByteLru::new(self.g.tile_budget()));
            if let Some(t) = tiles.get(&key) {
                return Ok(Arc::clone(t));
            }
        }
        self.check()?;
        let plan = self.plan()?;
        let frame = self.frame()?;
        let full = self.full(keep)?;
        let t = self.r.inner.options.tile_px.max(64);
        let rect = Rect::new(
            tx * t,
            ty * t,
            t.min(plan.out_w - tx * t),
            t.min(plan.out_h - ty * t),
        );
        let t = Instant::now();
        let img = Arc::new(full.run_tile(&frame, rect)?);
        self.runs.tiles += 1;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        if keep {
            let bytes = img.data.len() * 4;
            lock(&self.g.state)
                .tiles
                .get_or_insert_with(|| ByteLru::new(self.g.tile_budget()))
                .insert(key, Arc::clone(&img), bytes);
        }
        Ok(img)
    }

    /// The full-resolution raw group over `rect` (coordinates of the lens-cropped image), made of
    /// cached tiles when `keep` is set.
    pub fn region(&mut self, rect: Rect, keep: bool) -> Result<RgbImage, RenderError> {
        let plan = self.plan()?;
        if rect.is_empty() || rect.x1() > plan.out_w || rect.y1() > plan.out_h {
            return Err(RenderError::Image(
                "The region is outside the photo.".to_owned(),
            ));
        }
        if plan.reduced {
            let whole = self.group(1)?;
            return Ok(whole.extract(rect));
        }
        if !keep {
            // One call over the whole rectangle: the stage halos are paid once, not per tile.
            self.check()?;
            let frame = self.frame()?;
            let full = self.full(false)?;
            let t = Instant::now();
            let img = full.run_tile(&frame, rect)?;
            self.runs.tiles += 1;
            self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
            return Ok(img);
        }
        let t = self.r.inner.options.tile_px.max(64);
        let mut out = RgbImage::new(rect.w, rect.h);
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                let tile = self.tile(tx, ty, keep)?;
                let tr = Rect::new(tx * t, ty * t, tile.width, tile.height);
                let ov = tr.intersect(&rect);
                out.paste(
                    &tile,
                    Rect::new(ov.x - tr.x, ov.y - tr.y, ov.w, ov.h),
                    ov.x - rect.x,
                    ov.y - rect.y,
                );
            }
        }
        Ok(out)
    }

    /// Whether every tile (or the whole image, for camera-processed photos) that `rect` needs is
    /// cached.
    pub fn tiles_ready(&mut self, rect: Rect) -> Result<bool, RenderError> {
        let plan = self.plan()?;
        let params = self.params()?;
        let key = params.key();
        let st = lock(&self.g.state);
        if plan.reduced {
            return Ok(st.groups.as_ref().is_some_and(|g| g.contains(&(key, 1))));
        }
        let t = self.r.inner.options.tile_px.max(64);
        let Some(tiles) = st.tiles.as_ref() else {
            return Ok(false);
        };
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                if !tiles.contains(&(key.clone(), tx, ty)) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// The colour plan (C1 to C8).
    pub fn color(&mut self) -> Result<Arc<ColorPlan>, RenderError> {
        let key: ColorKey = color_key(self.src);
        if let Some(c) = &lock(&self.g.state).color
            && c.key == key
        {
            return Ok(Arc::clone(c));
        }
        let t = Instant::now();
        let plan = Arc::new(color_plan(self.src)?);
        self.runs.color_plans += 1;
        self.timings.plans_ms += t.elapsed().as_secs_f64() * 1000.0;
        lock(&self.g.state).color = Some(Arc::clone(&plan));
        Ok(plan)
    }

    /// The grade plan (C9 to C13) for an output.
    pub fn grade(
        &mut self,
        target: OutputTarget,
        dither: bool,
    ) -> Result<Arc<GradePlan>, RenderError> {
        let color = self.color()?;
        let display = self.r.display();
        let req = GradeRequest {
            target,
            dither,
            display: &display,
            looks: self.r.inner.looks.as_ref(),
        };
        // Build the plan's key without building the chain: the key is part of the plan, so the
        // cheap way to know whether a cached plan matches is to build the key text first.
        let wanted = super::color::grade_key(self.src, &color, &req);
        let cached = {
            let mut st = lock(&self.g.state);
            let cache = st.grade.get_or_insert_with(|| ByteLru::new(8));
            cache.get(&wanted).map(Arc::clone)
        };
        if let Some(plan) = cached {
            // A look that was replaced by Neutral says so on every frame, not only the first.
            for w in &plan.warnings {
                self.note(w);
            }
            return Ok(plan);
        }
        let t = Instant::now();
        let plan = Arc::new(grade_plan(self.src, &color, &req)?);
        self.runs.grade_plans += 1;
        self.timings.plans_ms += t.elapsed().as_secs_f64() * 1000.0;
        for w in &plan.warnings {
            self.note(w);
        }
        lock(&self.g.state)
            .grade
            .get_or_insert_with(|| ByteLru::new(8))
            .insert(plan.key.clone(), Arc::clone(&plan), 1);
        Ok(plan)
    }

    /// The geometry of the settings on this file.
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.src.exif_orientation(), &self.src.settings)
    }

    /// The size of the final image at full resolution: the lens-cropped image, turned, straightened
    /// and cropped.
    pub fn final_size(&mut self) -> Result<(usize, usize), RenderError> {
        let plan = self.plan()?;
        Ok(self.geometry().output_size(plan.out_w, plan.out_h))
    }
}

/// The colour space a target is encoded in.
pub(crate) fn color_space_of(target: OutputTarget) -> ColorSpace {
    match target {
        OutputTarget::Viewport | OutputTarget::DisplayP3 => ColorSpace::DisplayP3,
        OutputTarget::Srgb => ColorSpace::Srgb,
        OutputTarget::Rec2020 => ColorSpace::Rec2020,
    }
}
