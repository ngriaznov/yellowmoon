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
//! | `display` | the display-linear frame of a view before output sharpening | everything above plus geometry and the cell, not the sharpening amount |
//! | `frames` | finished viewport frames | everything above plus geometry, the cell and the output sharpening amount |
//!
//! The estimates' version (`PhotoView::estimates_version`) is deliberately not part of any key: the
//! keys hold the numbers the group reads, so a burst median that moves only the exposure recomputes
//! the colour group and leaves the raw group alone.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Instant;

use serde::Serialize;
use ym_export::{ColorSpace, DetailStat};
use ym_gpu::chain::StageRange;
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::sharpen::diffraction::DiffractionParams;
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::device::{Gpu, GpuError, GpuInfo};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

use super::color::{
    ColorKey, ColorPlan, DisplayChoice, GradeKey, GradePlan, GradeRequest, LookProvider,
    PhotoSource, color_key, color_plan, grade_plan,
};
use super::geometry::Geometry;
use super::raw::{
    CaptureMethod, FrameData, FullChain, GpuRoute, RawParams, RawParamsKey, RawPlan, SigmaPlane,
    diffraction_params, noise_model_is_usable, reduced_bayer, reduced_rgb, sigma_plane,
    usable_noise_model,
};
use super::{FrameZoom, NeuralStatus, RenderError, RenderPath, RenderStats};
use crate::app::gpu::GpuStages;
use crate::scheduler::{ByteLru, JobScheduler, MemoryBudget, MemoryLedger};
use crate::session::PhotoId;
use ym_color::display::OutputTarget;

/// The note of a render whose demosaic method is not the adaptive ratio (the test-only switch of
/// `RenderOptions::demosaic`): the device runs that one only.
const METHOD_NOT_ON_DEVICE_NOTE: &str = "The graphics card runs the adaptive ratio demosaic only, so the CPU ran the one that was asked for.";

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The default of [`RenderOptions::view_band_bytes`]: 128 MiB, the storage binding the GPU plan
/// designs every device pass for (`docs/gpu-plan.md` N-1, K4).
pub const VIEW_BAND_BYTES: u64 = 128 << 20;

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
    /// It is opened by [`crate::app::gpu::open_gpu`] and nothing else, and it can be given to a
    /// renderer that exists already ([`Renderer::set_gpu`]: `ym` opens the device while photos
    /// decode).
    pub gpu: Option<Arc<Gpu>>,
    /// The stages the device may run (`Auto` runs [`GpuStages::SIGNED_OFF`], `On` and `Require`
    /// run all of them). All of them by default: a renderer given a device and no policy holds
    /// nothing back. The raw chain ([`GpuStages::RAW_CHAIN`], the R6 demosaic of full-resolution
    /// tiles) consults it; the wiring tasks of the GPU wave consult it one stage at a time, and a
    /// stage that is off keeps its host boundary.
    pub gpu_stages: GpuStages,
    /// Whether a device error is a failure of the photo (`GpuPolicy::Require`, `--require-gpu`)
    /// instead of a fall-back to the CPU. Off by default.
    pub gpu_strict: bool,
    /// The largest storage binding, in bytes, that one device pass of a view may need: a four-plane
    /// f32 frame takes `16 * width * rows` of it. A view whose frame needs more (the Metal limit is
    /// 2 GiB, lavapipe's 128 MiB, and the plan designs for the lower, `docs/gpu-plan.md` N-1) is cut
    /// into horizontal bands that are rendered one after the other and put together, which gives
    /// the same pixels bit for bit (every device stage of a view is tile-exact). A constant of the
    /// renderer and not the device's limit, so that the cut does not depend on the machine.
    /// [`VIEW_BAND_BYTES`] by default.
    pub view_band_bytes: u64,
    /// The colour of the letterbox around the photo (the paper colour).
    pub background: [u8; 3],
    /// Longest edge of a thumbnail in pixels (`docs/01`: 240).
    pub thumbnail_px: usize,
    /// The R6 demosaic method of the full-resolution render (the 100 % view and every export). An
    /// internal switch with no settings field and no UI: [`DEFAULT_METHOD`], the adaptive ratio
    /// demosaic (the owner's decision of 2026-10-13, `docs/05` section R6), unless a test or an
    /// evaluation tool sets another (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 10).
    /// The reduced renders (fit view, thumbnails) bin the mosaic and do not demosaic. It is a
    /// property of the renderer, fixed when it is made, so it is not part of the cache keys.
    ///
    /// Builds with the `force-demosaic` feature (tests only, never a shipped build) read the
    /// initial value from the environment variable `YM_FORCE_DEMOSAIC`
    /// ([`forced_demosaic`]); other builds always start at [`DEFAULT_METHOD`].
    pub demosaic: Method,
}

/// The method the test-only switch asks for: the value of `YM_FORCE_DEMOSAIC` (a
/// [`Method::name`], such as `adaptive_ratio`), or `None` when the variable is not set.
///
/// Only builds with the `force-demosaic` cargo feature have this switch. The feature is off in
/// every normal build and is switched on only by the dev-dependencies of the test crates that
/// start `ym` as a child process (`crates/ym-cli`) and by `cargo build -p ym-cli --features
/// force-demosaic` for the separately built parity binary. Without it this function is a constant
/// `None` and nothing reads the environment.
///
/// # Panics
///
/// With the feature on, when the variable is set to something that is not a method name: a typo in
/// a test setup must not run the default method unnoticed.
#[cfg(feature = "force-demosaic")]
#[must_use]
pub fn forced_demosaic() -> Option<Method> {
    let value = std::env::var_os("YM_FORCE_DEMOSAIC")?;
    let name = value.to_string_lossy();
    let method = Method::from_name(&name).unwrap_or_else(|| {
        panic!(
            "YM_FORCE_DEMOSAIC={name:?} is not a demosaic method (one of {:?})",
            Method::ALL.map(Method::name)
        )
    });
    Some(method)
}

/// The method the test-only switch asks for. This build has no switch (the `force-demosaic`
/// feature is off): always `None`.
#[cfg(not(feature = "force-demosaic"))]
#[must_use]
pub const fn forced_demosaic() -> Option<Method> {
    None
}

impl std::fmt::Debug for RenderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderOptions")
            .field("threads", &self.threads)
            .field("tile_px", &self.tile_px)
            .field("full_tile_px", &self.full_tile_px)
            .field("budget", &self.budget)
            .field("neural", &self.neural)
            .field("demosaic", &self.demosaic)
            .field("gpu", &self.gpu.as_ref().map(|g| g.info().name.clone()))
            .field("gpu_stages", &self.gpu_stages)
            .field("gpu_strict", &self.gpu_strict)
            .field("view_band_bytes", &self.view_band_bytes)
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
            gpu_stages: GpuStages::ALL,
            gpu_strict: false,
            view_band_bytes: VIEW_BAND_BYTES,
            background: [255, 255, 255],
            thumbnail_px: 240,
            demosaic: forced_demosaic().unwrap_or(DEFAULT_METHOD),
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

/// Where one range of raw stages ran (`docs/gpu-plan.md` N-6: a render can be both, so the report
/// has one value per range, next to the single `device` of the colour pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeDevice {
    /// The stages (`R6` for the demosaic).
    pub range: StageRange,
    /// Where they ran for the tiles of this request.
    pub device: ComputeDevice,
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
    /// Output-sharpening passes (the last stage: sharpen the display-linear frame of a view and
    /// encode it).
    pub output_passes: u64,
    /// Display-linear frames served from the display cache (a sharpening pass that did not need to
    /// sample or run the colour chain again).
    pub display_hits: u64,
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
        self.output_passes += other.output_passes;
        self.display_hits += other.display_hits;
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
    /// The output-sharpening stage and the final encode (zero when the frame has none).
    pub output_ms: f64,
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
    /// Where the raw stages that can run on the device ran, per range, for the full-resolution
    /// tiles of this request. Empty when the request ran no such tile (a fit view, a thumbnail, a
    /// request served from a cache) or the renderer has no device for them. A request that fell
    /// back to the CPU part-way says [`ComputeDevice::Cpu`] and carries the note.
    pub ranges: Vec<RangeDevice>,
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
    /// Display-linear frames of views, before output sharpening (`frame::DisplayFrame`).
    display: Option<ByteLru<String, Arc<super::frame::DisplayFrame>>>,
    stats: Option<RenderStats>,
}

/// The stage caches of one photo.
pub struct PhotoGraph {
    id: PhotoId,
    state: Mutex<GraphState>,
    counters: StageCounters,
    budget: MemoryBudget,
    /// A device error ended the GPU route of this photo's raw chain: from then on it renders on the
    /// CPU (`docs/gpu-plan.md` 4.9). The caches may hold tiles of both devices, which differ within
    /// the tolerances of the kernels; an export restarts the photo wholly on the CPU.
    raw_gpu_failed: AtomicBool,
    /// A device error ended the GPU route of this photo's view stages (the sampler, the colour pass,
    /// output sharpening and C13 of frames, thumbnails and look tiles): from then on the CPU makes
    /// its frames, quietly (`docs/gpu-plan.md` 4.9).
    view_gpu_failed: AtomicBool,
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
            raw_gpu_failed: AtomicBool::new(false),
            view_gpu_failed: AtomicBool::new(false),
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
            + st.display.as_ref().map_or(0, ByteLru::used)
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

    /// A display-linear frame (a view before output sharpening) with this key, if one is cached.
    pub(crate) fn cached_display(&self, key: &str) -> Option<Arc<super::frame::DisplayFrame>> {
        let mut st = lock(&self.state);
        st.display.as_mut()?.get(&key.to_owned()).cloned()
    }

    /// Keeps a display-linear frame. Only the newest few are worth keeping (a slider drag asks for
    /// the same one again and again), so the budget is that of the finished frames.
    pub(crate) fn store_display(&self, key: String, frame: Arc<super::frame::DisplayFrame>) {
        let bytes = frame.bytes();
        let budget = (self.budget.cpu_bytes / 32) as usize;
        lock(&self.state)
            .display
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
    /// The options, replaced as a whole when the GPU or the memory budget is attached later
    /// ([`Renderer::set_gpu`], [`Renderer::set_budget`]); readers take the `Arc`.
    options: RwLock<Arc<RenderOptions>>,
    graphs: Mutex<Vec<(PhotoId, Arc<PhotoGraph>)>>,
    /// The detail statistic of the latest thumbnail of each photo (`thumbs.rs`), for the size
    /// estimate of the Export panel. A few bytes per photo, so unlike the graphs it is not capped.
    thumbnail_details: Mutex<HashMap<PhotoId, DetailStat>>,
    /// Photos with an id up to this one belong to a session that was closed (0: none).
    retired_through: AtomicU32,
    scheduler: JobScheduler,
    ledger: Arc<MemoryLedger>,
    /// A shader of the raw chain failed to compile on this machine: the device does not run the raw
    /// chain for the rest of the process (`docs/gpu-plan.md` 4.9, "shader: disable that kernel
    /// family for the process").
    raw_gpu_disabled: AtomicBool,
    /// A shader of the view stages failed to compile on this machine: the device does not run them
    /// for the rest of the process.
    view_gpu_disabled: AtomicBool,
}

/// The render engine (see the module documentation of [`crate::pipeline`]). Cheap to clone.
#[derive(Clone)]
pub struct Renderer {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("options", &self.options())
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
                options: RwLock::new(Arc::new(options)),
                graphs: Mutex::new(Vec::new()),
                thumbnail_details: Mutex::new(HashMap::new()),
                retired_through: AtomicU32::new(0),
                scheduler,
                ledger,
                raw_gpu_disabled: AtomicBool::new(false),
                view_gpu_disabled: AtomicBool::new(false),
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

    /// The options in force now: what the renderer was made with, with the GPU and the memory
    /// budget it was given since ([`Renderer::set_gpu`], [`Renderer::set_budget`]). A snapshot: it
    /// does not change under the caller.
    #[must_use]
    pub fn options(&self) -> Arc<RenderOptions> {
        Arc::clone(
            &self
                .inner
                .options
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Replaces the options with a copy changed by `change`.
    fn update_options(&self, change: impl FnOnce(&mut RenderOptions)) {
        let mut options = self
            .inner
            .options
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let mut next = (**options).clone();
        change(&mut next);
        *options = Arc::new(next);
    }

    /// Gives the renderer a GPU (or takes it away with `None`) and the stages it may run. `ym`
    /// opens the device in the background while photos decode and calls this before it queues
    /// the export, so that an export is wholly on the device or wholly on the CPU: it is for the
    /// start of a run, not for the middle of a render: a render that is running reads the options
    /// again at each stage ([`Renderer::options`] is a snapshot of the moment it is called), so one
    /// that straddles the call may see the device from the next stage on.
    pub fn set_gpu(&self, gpu: Option<Arc<Gpu>>, stages: GpuStages) {
        self.update_options(|o| {
            o.gpu = gpu;
            o.gpu_stages = stages;
        });
    }

    /// Makes a device error a failure of the photo (`GpuPolicy::Require`) or a fall-back to the CPU
    /// (every other policy). Like [`set_gpu`](Self::set_gpu) it is for the start of a run.
    pub fn set_gpu_strict(&self, strict: bool) {
        self.update_options(|o| o.gpu_strict = strict);
    }

    /// Cancels the queued GPU jobs of the viewport and tile renders of `left`, the photo the person
    /// moved away from, and sets the token of the one that runs. Without it a queued job keeps the
    /// photo's decoded mosaic alive behind it until it is popped. Thumbnail and export jobs of the
    /// device are untouched (`Scheduler::cancel_navigation`). Returns how many were cancelled.
    pub(crate) fn cancel_gpu_navigation(&self, left: PhotoId) -> usize {
        self.options().gpu.as_ref().map_or(0, |g| {
            g.scheduler().cancel_navigation(Some(u64::from(left.0)))
        })
    }

    /// Changes the memory budget: the ledger of full-resolution renders follows at once, graphs
    /// made from now on get the new figures (a graph keeps the budget it was made with). The
    /// unified-memory rule lowers the CPU budget when the device turns out to share the RAM
    /// ([`MemoryBudget::for_unified_memory`]).
    pub fn set_budget(&self, budget: MemoryBudget) {
        self.inner.ledger.set_budget(budget.cpu_bytes);
        self.update_options(|o| o.budget = budget);
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
        self.options().gpu.as_ref().map(|g| g.info().clone())
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
            let g = Arc::new(PhotoGraph::new(id, self.options().budget));
            graphs.push((id, g));
        }
        let max = self.options().budget.max_photos.max(1);
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
        lock(&self.inner.thumbnail_details).remove(&id);
    }

    /// Forgets everything (the session closed).
    pub fn forget_all(&self) {
        lock(&self.inner.graphs).clear();
        lock(&self.inner.thumbnail_details).clear();
    }

    /// The detail statistic of the latest thumbnail rendered for a photo
    /// ([`Renderer::thumbnail_png`]: the photo as it is developed now, at most 240 px, in sRGB),
    /// `None` while the photo has no thumbnail (or one too small to measure). The Export panel's
    /// size estimate reads it (`docs/09` Size estimate). A thumbnail of a look tile does not
    /// count: it is the photo through another look.
    #[must_use]
    pub fn thumbnail_detail(&self, id: PhotoId) -> Option<DetailStat> {
        lock(&self.inner.thumbnail_details).get(&id).copied()
    }

    /// Records the statistic of a photo's new thumbnail. A thumbnail too small to measure removes
    /// the old figure, which belongs to a different picture. A photo of a closed session is not
    /// recorded.
    pub(crate) fn note_thumbnail_detail(&self, id: PhotoId, detail: Option<DetailStat>) {
        let mut details = lock(&self.inner.thumbnail_details);
        // Checked under the lock: `retire_through` marks the photos retired and then clears the map
        // under the same lock, so a thumbnail that finishes while the session closes either sees
        // the mark or is cleared after it, and never leaves a figure behind.
        if self.is_retired(id) {
            return;
        }
        match detail {
            Some(d) => {
                details.insert(id, d);
            }
            None => {
                details.remove(&id);
            }
        }
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

/// The stages of a viewport frame, a thumbnail or a look tile that can run on the device: the
/// geometry sampler, the colour chain, output sharpening and C13.
pub(crate) const VIEW_STAGES: [GpuStages; 4] = [
    GpuStages::SAMPLER,
    GpuStages::COLOR,
    GpuStages::OUTPUT_SHARPEN,
    GpuStages::ENCODE,
];

/// The device a request runs its view stages on and the stages it may run there
/// ([`Job::view_gpu`]).
#[derive(Clone)]
pub(crate) struct ViewGpu {
    /// The device.
    pub gpu: Arc<Gpu>,
    /// Which view stages it runs: a non-empty subset of [`VIEW_STAGES`].
    pub stages: GpuStages,
    /// A device error fails the request (`Require`) instead of falling back to the CPU.
    pub strict: bool,
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
    /// The options of the moment the request began. A render reads them from here and not from
    /// [`Renderer::options`] again, so a device attached while an export runs ([`Renderer::set_gpu`])
    /// cannot put the tiles of one export on two devices.
    opts: Arc<RenderOptions>,
    /// The priority of the device jobs of this request. Every job of one request has the same
    /// priority (an export `Export`, any other request `Viewport` unless the caller says), so that
    /// cancelling a photo's navigation jobs reaches all of the request's queued jobs or none.
    gpu_priority: Priority,
    /// An export restarts the whole photo on the CPU after a device error, instead of finishing the
    /// tiles that are left on the CPU (never a mix of devices in one export, `docs/gpu-plan.md` 4.9).
    restartable: bool,
    /// The device error that asked for such a restart ([`Job::take_gpu_restart`]).
    restart: Option<String>,
    /// Where the raw stages ran, per range, for the tiles of this request.
    raw_ranges: Vec<RangeDevice>,
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
        let mut job = Self::for_export(r, src, cancel);
        job.gpu_priority = Priority::Viewport;
        job.restartable = false;
        Ok(job)
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
            opts: r.options(),
            gpu_priority: Priority::Export,
            restartable: true,
            restart: None,
            raw_ranges: Vec::new(),
        }
    }

    /// The device jobs of this request run at `priority` (the request's own: `TileNeural` for the
    /// 100 % tiles a viewport prefetches, `Thumbnail` for thumbnails and look tiles).
    #[must_use]
    pub fn with_gpu_priority(mut self, priority: Priority) -> Self {
        self.gpu_priority = priority;
        self
    }

    /// The options this request runs with: a snapshot taken when it began.
    pub fn options(&self) -> &RenderOptions {
        &self.opts
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
        let neural = &self.opts.neural;
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
            ranges: self.raw_ranges,
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

    /// The controls of R8 diffraction deconvolution for this photo: `Some` when the internal
    /// switch ([`PhotoSource::capture_method`]) asks for it and the photo has an f-number and a
    /// pixel pitch (`docs/algorithms/03-diffraction-deconvolution.md` section 2.2), else `None`
    /// (Richardson-Lucy).
    fn diffraction_params(&self) -> Option<DiffractionParams> {
        if self.src.capture_method != CaptureMethod::Diffraction {
            return None;
        }
        let raw = &self.src.raw;
        diffraction_params(
            &raw.meta,
            self.src.sensor.as_ref(),
            raw.data.width,
            raw.data.height,
            raw.is_reduced(),
        )
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
        // A noise model that cannot be used (a number that is not finite, a negative gain or
        // variance, a variance that overflows) is read as zero noise for the whole raw group, so
        // that the sigma map, the denoise, R8 and the adaptive ratio demosaic all see the same
        // usable model and the photo renders instead of coming out black or being refused
        // (`docs/algorithms/INTEGRATION.md` section 10). The estimator of `ym-noise` never makes
        // such a model; this guards against corrupt data. The log carries no file name.
        if !noise_model_is_usable(&e.noise.model) {
            tracing::warn!("the noise model of the photo is not usable; rendering with zero noise");
        }
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
            noise: usable_noise_model(&e.noise.model),
            diffraction: self.diffraction_params(),
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
        let full = Arc::new(FullChain::with_method(
            &frame,
            &plan,
            &params,
            &sigma,
            self.opts.demosaic,
            self.cancel,
        )?);
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
        let t = self.opts.tile_px.max(64);
        let rect = Rect::new(
            tx * t,
            ty * t,
            t.min(plan.out_w - tx * t),
            t.min(plan.out_h - ty * t),
        );
        let t = Instant::now();
        let img = Arc::new(self.run_full_tile(&full, &frame, rect)?);
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
            let img = self.run_full_tile(&full, &frame, rect)?;
            self.runs.tiles += 1;
            self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
            return Ok(img);
        }
        let t = self.opts.tile_px.max(64);
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

    /// The route to the device for the raw chain of this request, or `None` when the CPU runs it:
    /// there is no device, the stage is not on ([`GpuStages::RAW_CHAIN`]), a device error ended the
    /// route for this photo or for the process, the device is lost, or the demosaic method is not the
    /// one the device runs (a test switch). Reads the options of the request's snapshot.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] when the device is lost, or an earlier device error ended the route, and
    /// the policy is strict.
    fn raw_route(&mut self) -> Result<Option<GpuRoute>, RenderError> {
        let Some(gpu) = self.opts.gpu.clone() else {
            return Ok(None);
        };
        if !self.opts.gpu_stages.contains(GpuStages::RAW_CHAIN) {
            return Ok(None);
        }
        if self.g.raw_gpu_failed.load(Ordering::SeqCst)
            || self.r.inner.raw_gpu_disabled.load(Ordering::SeqCst)
        {
            // An earlier device error ended the route. A strict policy (`Require`) must not turn
            // that into a quiet CPU render of the next photo: it fails that photo too.
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(
                    "an earlier error ended the use of the graphics card, and the strict setting does not let the CPU take over"
                        .to_owned(),
                ));
            }
            return Ok(None);
        }
        if gpu.is_lost() {
            let why = gpu
                .lost_reason()
                .unwrap_or_else(|| "the device was lost".to_owned());
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(why));
            }
            self.note(&format!(
                "The graphics card stopped working, so the CPU did: {why}."
            ));
            return Ok(None);
        }
        if self.opts.demosaic != Method::AdaptiveRatio {
            self.note(METHOD_NOT_ON_DEVICE_NOTE);
            return Ok(None);
        }
        Ok(Some(GpuRoute {
            gpu,
            range: StageRange::DEMOSAIC,
            spec: JobSpec::new(self.gpu_priority).owned_by(u64::from(self.src.id.0)),
            token: self.cancel.clone(),
        }))
    }

    /// Notes where the raw range ran, once per device (the report lists each range once).
    fn record_range(&mut self, range: StageRange, device: ComputeDevice) {
        match self.raw_ranges.iter_mut().find(|r| r.range == range) {
            // A request that ran on both says CPU: the report is about the picture, and a mixed
            // request is a fall-back.
            Some(r) if r.device != device => r.device = ComputeDevice::Cpu,
            Some(_) => {}
            None => self.raw_ranges.push(RangeDevice { range, device }),
        }
    }

    /// The device for the view stages of this request, or `None` when the CPU makes the frame:
    /// there is no device, none of the view stages is on ([`VIEW_STAGES`]), a device error ended the
    /// route for this photo or for the process, or the device is lost. Reads the options of the
    /// request's snapshot.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] when the device is lost, or an earlier device error ended the route, and
    /// the policy is strict.
    pub(crate) fn view_gpu(&mut self) -> Result<Option<ViewGpu>, RenderError> {
        let Some(gpu) = self.opts.gpu.clone() else {
            return Ok(None);
        };
        let stages = VIEW_STAGES
            .into_iter()
            .filter(|s| self.opts.gpu_stages.contains(*s))
            .fold(GpuStages::NONE, GpuStages::union);
        if stages.is_empty() {
            return Ok(None);
        }
        if self.g.view_gpu_failed.load(Ordering::SeqCst)
            || self.r.inner.view_gpu_disabled.load(Ordering::SeqCst)
        {
            // As for the raw chain: a strict policy must not turn an earlier error into a quiet CPU
            // frame of the next request.
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(
                    "an earlier error ended the use of the graphics card, and the strict setting does not let the CPU take over"
                        .to_owned(),
                ));
            }
            return Ok(None);
        }
        if gpu.is_lost() {
            let why = gpu
                .lost_reason()
                .unwrap_or_else(|| "the device was lost".to_owned());
            if self.opts.gpu_strict {
                return Err(RenderError::Gpu(why));
            }
            self.note(&format!(
                "The graphics card stopped working, so the CPU did: {why}."
            ));
            return Ok(None);
        }
        Ok(Some(ViewGpu {
            gpu,
            stages,
            strict: self.opts.gpu_strict,
        }))
    }

    /// A view stage failed on the device with `error`. A shader that does not compile switches the
    /// view stages off for the process. A strict policy fails the request; any other policy notes
    /// why the CPU makes this frame, and ends the route for the photo (view caches may then hold
    /// frames of both devices, which differ within the tolerances of the kernels; viewing only,
    /// `docs/gpu-plan.md` 4.9).
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] under a strict policy.
    pub(crate) fn view_gpu_failed(&mut self, error: &GpuError) -> Result<(), RenderError> {
        let reason = error.to_string();
        tracing::warn!(error = %reason, "the device failed on a view; the CPU takes over");
        if matches!(error, GpuError::Shader(_)) {
            self.r
                .inner
                .view_gpu_disabled
                .store(true, Ordering::SeqCst);
        }
        if self.opts.gpu_strict {
            return Err(RenderError::Gpu(reason));
        }
        self.g.view_gpu_failed.store(true, Ordering::SeqCst);
        self.note(&format!(
            "The graphics card could not render this frame, so the CPU did: {reason}."
        ));
        Ok(())
    }

    /// Runs one tile of the full-resolution chain: the stages the renderer runs on the device go
    /// there (the route above), everything else on the CPU. A device error ends the route for the
    /// photo, never for one tile:
    ///
    /// * strict policy (`Require`): the error fails the render ([`RenderError::Gpu`]);
    /// * an export (`restartable`): the render asks to start again on the CPU
    ///   ([`Job::take_gpu_restart`]), so that one file is wholly device or wholly CPU. It asks the
    ///   same when its first tiles ran on the device and the route went away between two tiles
    ///   (the device was lost, or another request ended the photo's or the process's use of it);
    /// * any other request (a 100 % tile, a viewport region): this tile runs on the CPU, with a note,
    ///   and so does every later request for the photo. The tile cache may then hold tiles of both
    ///   devices, which differ within the tolerances of the kernels (viewing only).
    ///
    /// A cancelled render stays cancelled.
    fn run_full_tile(
        &mut self,
        full: &FullChain,
        frame: &FrameData,
        rect: Rect,
    ) -> Result<RgbImage, RenderError> {
        let expected =
            self.opts.gpu.is_some() && self.opts.gpu_stages.contains(GpuStages::RAW_CHAIN);
        let route = self.raw_route()?;
        if route.is_none()
            && expected
            && self.restartable
            && self
                .raw_ranges
                .iter()
                .any(|r| r.device == ComputeDevice::Gpu)
        {
            // The first tiles of this export ran on the device and the route has gone away since:
            // another request failed on the device (the flags above belong to the photo and to the
            // process, not to this export), or the device was lost between two tiles. The rest of
            // the file must not be made on the CPU: start again on the CPU, as for an error of the
            // device in one of the tiles.
            let reason = "the graphics card was no longer available while the photo was rendering"
                .to_owned();
            self.raw_ranges.clear();
            self.restart = Some(reason.clone());
            return Err(RenderError::Gpu(reason));
        }
        if let Some(route) = route {
            match full.run_tile_gpu(frame, rect, &route) {
                Ok(img) => {
                    self.record_range(route.range, ComputeDevice::Gpu);
                    return Ok(img);
                }
                Err(GpuError::Cancelled) => return Err(RenderError::Cancelled),
                Err(e) => {
                    let reason = e.to_string();
                    tracing::warn!(error = %reason, "the device failed on a tile; the CPU takes over");
                    if matches!(e, GpuError::Shader(_)) {
                        self.r.inner.raw_gpu_disabled.store(true, Ordering::SeqCst);
                    }
                    if self.opts.gpu_strict {
                        return Err(RenderError::Gpu(reason));
                    }
                    self.g.raw_gpu_failed.store(true, Ordering::SeqCst);
                    if self.restartable {
                        self.raw_ranges.clear();
                        self.restart = Some(reason.clone());
                        return Err(RenderError::Gpu(reason));
                    }
                    self.note(&format!(
                        "The graphics card could not render this frame, so the CPU did: {reason}."
                    ));
                }
            }
        }
        if expected {
            self.record_range(StageRange::DEMOSAIC, ComputeDevice::Cpu);
        }
        full.run_tile(frame, rect)
    }

    /// Whether the last error was a device error that asks the export to start again on the CPU
    /// (see [`Job::run_full_tile`]). Clears the request, drops what the device had made and adds the
    /// note. The photo's route is already off, so the restart runs on the CPU from the first tile.
    pub fn take_gpu_restart(&mut self) -> bool {
        let Some(reason) = self.restart.take() else {
            return false;
        };
        self.raw_ranges.clear();
        self.note(&format!(
            "The graphics card could not render this photo, so the CPU did: {reason}."
        ));
        true
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
        let t = self.opts.tile_px.max(64);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A build without the `force-demosaic` feature has no switch: nothing reads the environment
    /// and a renderer made from the default options demosaics with the default method.
    #[cfg(not(feature = "force-demosaic"))]
    #[test]
    fn without_the_test_feature_nothing_can_force_the_method() {
        assert_eq!(forced_demosaic(), None);
        assert_eq!(RenderOptions::default().demosaic, DEFAULT_METHOD);
    }

    /// With the feature on and the variable unset (the normal state of every test run), the
    /// options still start at the default method.
    #[cfg(feature = "force-demosaic")]
    #[test]
    fn the_test_feature_changes_nothing_while_the_variable_is_unset() {
        if std::env::var_os("YM_FORCE_DEMOSAIC").is_none() {
            assert_eq!(forced_demosaic(), None);
            assert_eq!(RenderOptions::default().demosaic, DEFAULT_METHOD);
        }
    }
}
