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
//! | `plan` | levels, vignetting profile, lens maps and their auto crop; one for the final picture and one for the Before view (the same plan without the vignetting profile) | `lens_corrections`, `chromatic_aberration`, and whether R3 runs |
//! | `sigma` | the half-resolution noise sigma | the noise model and the gains `g0` |
//! | `groups` | the raw group's image at 1/k | [`RawParams`] (all of `decode.*`, whether R3 runs, noise reduction, sharpening, `g0`, noise) and k |
//! | `full` | the prepared full-resolution chain; one for the final picture and one for the Before view | [`RawParams`] |
//! | `tiles` | 512 px tiles of the full-resolution raw group | [`RawParams`] and the tile |
//! | `color` | the fused matrix and C7 | [`ColorKey`] |
//! | `grade` | the prepared `ColorChain` | [`GradeKey`] |
//! | `display` | the display-linear frame of a view before output sharpening | everything above plus geometry and the cell, not the sharpening amount |
//! | `frames` | finished viewport frames | everything above plus geometry, the cell and the output sharpening amount |
//!
//! The estimates' version (`PhotoView::estimates_version`) is deliberately not part of any key: the
//! keys hold the numbers the group reads, so a burst median that moves only the exposure recomputes
//! the colour group and leaves the raw group alone.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::Instant;

use serde::Serialize;
use ym_export::{ColorSpace, DetailStat};
use ym_gpu::chain::StageRange;
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::sharpen::diffraction::DiffractionParams;
use ym_gpu::cpu::{Rect, RgbImage};
use ym_gpu::device::{Gpu, GpuContext, GpuError, GpuInfo};
use ym_gpu::image::{GpuImage, RowBands};
use ym_gpu::kernels::copy::Piece;
use ym_gpu::kernels::prep::DeviceRaw;
use ym_gpu::queue::{CancelToken, JobHandle, JobSpec, Priority};

use super::color::{
    ColorKey, ColorPlan, DisplayChoice, GradeKey, GradePlan, GradeRequest, LookProvider,
    PhotoSource, color_key, color_plan, grade_plan,
};
use super::geometry::Geometry;
use super::raw::{
    CaptureMethod, DEVICE_RANGE, FRONT_RANGE, FrameData, FullChain, GpuRoute, RawParams,
    RawParamsKey, RawPlan, SigmaPlane, diffraction_params, has_vignetting_profile,
    noise_model_is_usable, reduced_bayer, reduced_params, reduced_rgb, sigma_plane,
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

/// How many photos keep their raw words on the device at a time: the one being looked at and the one
/// before it (`docs/17-gpu.md` section 7). The words of a 50 MP photo are 100 MB; a photo that is not
/// among the two is uploaded again when it is next rendered.
pub const RESIDENT_PHOTOS: usize = 2;

/// The default of [`RenderOptions::view_band_bytes`]: 128 MiB, the storage binding every device
/// pass is designed for (`docs/17-gpu.md` section 4, K4).
pub const VIEW_BAND_BYTES: u64 = 128 << 20;

/// What [`Renderer::prefetch_raw`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawPrefetch {
    /// The photo's raw words were uploaded.
    Uploaded,
    /// They were on the device already.
    Resident,
    /// Nothing was done: there is no device for the raw front, the file has no mosaic, two photos are
    /// resident already, or the upload failed.
    Skipped,
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
    /// It is opened by [`crate::app::gpu::open_gpu`] and nothing else, and it can be given to a
    /// renderer that exists already ([`Renderer::set_gpu`]: `ym` opens the device while photos
    /// decode).
    pub gpu: Option<Arc<Gpu>>,
    /// The stages the device may run (`Auto` runs [`GpuStages::SIGNED_OFF`], `On` and `Require`
    /// run all of them). All of them by default: a renderer given a device and no policy holds
    /// nothing back. The raw chain ([`GpuStages::RAW_CHAIN`], R6 to R8 of full-resolution
    /// tiles: the demosaic, the denoise, R7 and capture sharpening) consults it; the other ranges
    /// consult it one stage at a time, and a stage that is off keeps its host boundary.
    pub gpu_stages: GpuStages,
    /// Whether a device error is a failure of the photo (`GpuPolicy::Require`, `--require-gpu`)
    /// instead of a fall-back to the CPU. Off by default.
    pub gpu_strict: bool,
    /// The largest storage binding, in bytes, that one device pass of a view may need: a four-plane
    /// f32 frame takes `16 * width * rows` of it. A view whose frame needs more (the Metal limit is
    /// 2 GiB, lavapipe's 128 MiB, and every kernel is designed for the lower, `docs/17-gpu.md`
    /// section 4) is cut into horizontal bands that are rendered one after the other and put
    /// together, which gives the same pixels bit for bit (every device stage of a view is
    /// tile-exact). A constant of the renderer and not the device's limit, so that the cut does not
    /// depend on the machine. [`VIEW_BAND_BYTES`] by default.
    pub view_band_bytes: u64,
    /// Whether the caches of the raw group (the reduced groups and the tiles of the 100 % view) keep
    /// what the device made where it is, on the device, for the sampler to read (`docs/17-gpu.md`
    /// section 3, `Backing`): no readback after the raw chain and no upload before the sampler. Has an
    /// effect only with a device that runs the sampler ([`GpuStages::SAMPLER`]). Off keeps host
    /// images; the frames are the same either way, bit for bit (the sampler reads the
    /// same samples), and the tests compare the two. On by default.
    pub resident_caches: bool,
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
            .field("resident_caches", &self.resident_caches)
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
            resident_caches: true,
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

/// Where one range of raw stages ran (`docs/17-gpu.md` section 9: a render can be both, so the report
/// has one value per range, next to the single `device` of the colour pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeDevice {
    /// The stages (`R1..R9` for the whole chain of the device: [`FRONT_RANGE`], with the raw front
    /// on; `R6..R8` for the raw group behind a CPU front: [`DEVICE_RANGE`]; `R6` is the demosaic
    /// alone).
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
    /// Of those, the ones the device made from the photo's resident raw words (`GpuReduced`).
    pub device_groups: u64,
    /// Times the photo's raw words were put on the device (once per photo while it stays among the
    /// [`RESIDENT_PHOTOS`] most recent ones).
    pub raw_uploads: u64,
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
        self.device_groups += other.device_groups;
        self.raw_uploads += other.raw_uploads;
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

/// A cached raw plan with the lens switches (`lens_corrections`, `chromatic_aberration`) it was made for.
type PlanSlot = ((bool, bool), Arc<RawPlan>);

#[derive(Default)]
struct GraphState {
    /// The fingerprint of the decoded image the caches belong to.
    fingerprint: Option<u64>,
    /// The raw plan, with R3 (index 0, the final picture's) and without it (index 1, the Before view's:
    /// [`RawPlan::without_vignetting`]), each with the lens switches it was made for.
    plan: [Option<PlanSlot>; 2],
    sigma: Option<(Vec<u64>, Arc<SigmaPlane>)>,
    groups: Option<ByteLru<(RawParamsKey, usize), Backing>>,
    /// The prepared full-resolution chain, with R3 (index 0) and without it (index 1, the Before view's):
    /// one of each, so that flipping to Before and back prepares neither again, and a settings edit
    /// still replaces the one chain it changes.
    full: [Option<(RawParamsKey, Arc<FullChain>)>; 2],
    /// The photo's raw words on the device (`GpuReduced` and the range R1..R9 read them), uploaded
    /// once per photo; released with the photo's place among the [`RESIDENT_PHOTOS`].
    resident: Option<Arc<DeviceRaw>>,
    tiles: Option<ByteLru<(RawParamsKey, usize, usize), Backing>>,
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
    /// CPU (`docs/17-gpu.md` section 9). The caches may hold tiles of both devices, which differ within
    /// the tolerances of the kernels; an export restarts the photo wholly on the CPU.
    raw_gpu_failed: AtomicBool,
    /// A device error ended the GPU route of this photo's view stages (the sampler, the colour pass,
    /// output sharpening and C13 of frames, thumbnails and look tiles): from then on the CPU makes
    /// its frames, quietly (`docs/17-gpu.md` section 9).
    view_gpu_failed: AtomicBool,
    /// When the photo's raw words were last used on the device, in ticks of the renderer's clock
    /// (the order of the two-photo LRU).
    resident_tick: AtomicU64,
    /// Held while the raw words are uploaded, so that two requests for one photo upload once.
    upload_gate: Mutex<()>,
    /// Exports that are rendering the photo: while there is one, the raw words stay on the device
    /// whatever else the renderer uploads (an export asks for them at every tile).
    pins: AtomicU32,
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
            resident_tick: AtomicU64::new(0),
            upload_gate: Mutex::new(()),
            pins: AtomicU32::new(0),
        }
    }

    /// The photo's raw words on the device, when they are there.
    fn resident(&self) -> Option<Arc<DeviceRaw>> {
        lock(&self.state).resident.clone()
    }

    /// Lets go of the photo's raw words (a request that is using them keeps its own reference until
    /// it is done).
    fn drop_resident(&self) {
        lock(&self.state).resident = None;
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
            + st.full
                .iter()
                .flatten()
                .map(|(_, f)| f.bytes())
                .sum::<usize>()
            + st.tiles.as_ref().map_or(0, ByteLru::used)
            + st.frames.as_ref().map_or(0, ByteLru::used)
            + st.display.as_ref().map_or(0, ByteLru::used)
    }

    /// Drops the full-resolution intermediates (tiles and the prepared chain), keeping the small
    /// reduced images: called when the photo is no longer the one being looked at.
    pub fn release_full_resolution(&self) {
        let mut st = lock(&self.state);
        st.full = [None, None];
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

    /// Keeps a finished frame (on the host, or kept on the device: either way the bytes count).
    pub(crate) fn store_frame(&self, key: String, frame: Arc<super::frame::RenderedFrame>) {
        let bytes = frame.bytes();
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

    /// The bytes the reduced groups may take. A cache that keeps its images on the device counts
    /// device bytes, and takes at most an eighth of the device's budget as well (the rest is for the
    /// raw words, the working buffers of a render and the other cache).
    fn group_budget(&self, on_device: bool) -> usize {
        self.cache_budget(on_device)
    }

    /// The bytes the tiles of the 100 % view may take (as [`PhotoGraph::group_budget`]).
    fn tile_budget(&self, on_device: bool) -> usize {
        self.cache_budget(on_device)
    }

    fn cache_budget(&self, on_device: bool) -> usize {
        let host = self.budget.cpu_bytes / 6;
        if on_device {
            host.min(self.budget.gpu_bytes / 8) as usize
        } else {
            host as usize
        }
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
    /// Photos taken out of the filmstrip while the session goes on ([`Renderer::retire_photos`]); a photo
    /// put back ([`Renderer::unretire_photos`]) leaves the set.
    removed: Mutex<HashSet<PhotoId>>,
    scheduler: JobScheduler,
    ledger: Arc<MemoryLedger>,
    /// A shader of the raw chain failed to compile on this machine: the device does not run the raw
    /// chain for the rest of the process (`docs/17-gpu.md` section 9, "Shader: that kernel family
    /// is off for the process").
    raw_gpu_disabled: AtomicBool,
    /// The clock of the two-photo LRU of resident raw words ([`PhotoGraph::resident_tick`]).
    resident_clock: AtomicU64,
    /// The photo the person looked at before the current one: its raw words are the second to go.
    previous: Mutex<Option<PhotoId>>,
    /// A shader of the view stages failed to compile on this machine: the device does not run them
    /// for the rest of the process.
    view_gpu_disabled: AtomicBool,
    /// Held while a prefetch of raw words counts the photos that are resident and uploads, so that
    /// two prefetches do not both find room for the last place ([`Renderer::prefetch_raw`]).
    prefetch_gate: Mutex<()>,
    /// A shader of the export tail (the sampler, colour, resize, sharpening and C13 kernels as the
    /// tail runs them, and the DNG's device colour stage) failed to compile on this machine: the
    /// device does not run the tail for the rest of the process (`docs/17-gpu.md` section 9,
    /// "Shader: that kernel family is off for the process"). The photo that met the error was already made
    /// again on the CPU; without the flag every later export would try the same shader and fail the
    /// same way before it fell back.
    tail_gpu_disabled: AtomicBool,
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
                removed: Mutex::new(HashSet::new()),
                scheduler,
                ledger,
                raw_gpu_disabled: AtomicBool::new(false),
                resident_clock: AtomicU64::new(0),
                previous: Mutex::new(None),
                view_gpu_disabled: AtomicBool::new(false),
                prefetch_gate: Mutex::new(()),
                tail_gpu_disabled: AtomicBool::new(false),
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
        let old = lock(&self.inner.current).replace(photo);
        if let Some(left) = old.filter(|o| *o != photo) {
            *lock(&self.inner.previous) = Some(left);
        }
        // The photo left and the photo arrived are the two the device keeps raw words for; any other
        // photo's words (a thumbnail's) are let go now.
        for id in old.into_iter().chain([photo]) {
            if let Some(g) = self.existing_graph(id) {
                self.touch_resident(&g);
            }
        }
        self.trim_resident(RESIDENT_PHOTOS);
        old.filter(|o| *o != photo)
    }

    /// Whether `id` is the photo being looked at or the one looked at before it: the two whose raw
    /// words the device keeps whatever else is uploaded.
    fn is_protected(&self, id: PhotoId) -> bool {
        Some(id) == *lock(&self.inner.current) || Some(id) == *lock(&self.inner.previous)
    }

    /// Marks the photo's raw words as the most recently used.
    fn touch_resident(&self, g: &PhotoGraph) {
        let tick = self.inner.resident_clock.fetch_add(1, Ordering::SeqCst) + 1;
        g.resident_tick.store(tick, Ordering::SeqCst);
    }

    /// Lets go of the raw words of every photo but the `keep` most recently used ones. The photo being
    /// looked at ([`Renderer::navigate`]) counts as the most recent, the one before it as the next. The device frees nothing at once
    /// (the buffers go back to its pool, which gives them to the next allocation or trims them under
    /// pressure).
    pub(crate) fn trim_resident(&self, keep: usize) {
        let graphs: Vec<Arc<PhotoGraph>> = lock(&self.inner.graphs)
            .iter()
            .map(|(_, g)| Arc::clone(g))
            .collect();
        let current = *lock(&self.inner.current);
        let previous = *lock(&self.inner.previous);
        let mut held: Vec<(u64, Arc<PhotoGraph>)> = graphs
            .into_iter()
            .filter(|g| g.pins.load(Ordering::SeqCst) == 0 && g.resident().is_some())
            .map(|g| {
                // The photo looked at and the one looked at before it come first, whatever the
                // thumbnails of a folder have touched since.
                let tick = if Some(g.id) == current {
                    u64::MAX
                } else if Some(g.id) == previous {
                    u64::MAX - 1
                } else {
                    g.resident_tick.load(Ordering::SeqCst)
                };
                (tick, g)
            })
            .collect();
        held.sort_by_key(|(tick, _)| std::cmp::Reverse(*tick));
        for (_, g) in held.into_iter().skip(keep) {
            g.drop_resident();
        }
    }

    /// Whether a shader of the export tail failed to compile on this machine, and the device does
    /// not run the tail for the rest of the process (see [`Renderer::export_tail_enabled`]).
    #[must_use]
    pub fn export_tail_disabled(&self) -> bool {
        self.inner.tail_gpu_disabled.load(Ordering::SeqCst)
    }

    /// Switches the export tail off for the process: `error` is a shader failure of one of its
    /// kernels. Any other error is a property of the export (a buffer that does not fit, a device
    /// that was lost), not of the machine, and leaves the tail on.
    pub(crate) fn note_export_tail_error(&self, error: &GpuError) {
        if matches!(error, GpuError::Shader(_)) {
            tracing::warn!(error = %error, "a shader of the export tail failed; the device does not run it again in this process");
            self.inner.tail_gpu_disabled.store(true, Ordering::SeqCst);
        }
    }

    /// Puts the raw words of `src` on the device ahead of the person getting there (the neighbours of
    /// the photo being looked at), as a job of [`Priority::Thumbnail`], so that the first frame of the
    /// photo does not wait for the upload. A prefetch never makes room: it uploads only while fewer
    /// than [`RESIDENT_PHOTOS`] photos have their words on the device, so that the photo being
    /// looked at and the one before it keep theirs and at most two are ever uploaded.
    ///
    /// Best effort: no device, the raw front off, a camera-processed file, a failed upload and a
    /// strict policy's refusal are all [`RawPrefetch::Skipped`] (the frame that needs the words
    /// uploads them itself, and reports any error).
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`] when `cancel` was set or the photo belongs to a closed session;
    /// [`RenderError::Image`] for a mosaic that cannot be cut.
    pub fn prefetch_raw(
        &self,
        src: &PhotoSource,
        cancel: &CancelToken,
    ) -> Result<RawPrefetch, RenderError> {
        let mut job = Job::new(self, src, cancel)?;
        job.gpu_priority = Priority::Thumbnail;
        let Ok(Some(gpu)) = job.raw_device(GpuStages::RAW_FRONT) else {
            return Ok(RawPrefetch::Skipped);
        };
        if job.g.resident().is_some() {
            self.touch_resident(&job.g);
            return Ok(RawPrefetch::Resident);
        }
        let _gate = lock(&self.inner.prefetch_gate);
        if self.resident_photos() >= RESIDENT_PHOTOS {
            return Ok(RawPrefetch::Skipped);
        }
        let frame = job.frame()?;
        if frame.reduced {
            return Ok(RawPrefetch::Skipped);
        }
        let route = job.device_route(gpu);
        match job.resident(&route, &frame) {
            Ok(_) => Ok(RawPrefetch::Uploaded),
            Err(GpuError::Cancelled) => Err(RenderError::Cancelled),
            Err(e) => {
                tracing::debug!(error = %e, "the raw words of a neighbour could not be prefetched");
                Ok(RawPrefetch::Skipped)
            }
        }
    }

    /// Queues [`Renderer::prefetch_raw`] at [`Priority::Thumbnail`]. Moving on to another photo does
    /// not cancel it (the photo it loads is likely the next one).
    #[must_use]
    pub fn spawn_raw_prefetch(
        &self,
        src: PhotoSource,
    ) -> JobHandle<Result<RawPrefetch, RenderError>> {
        let r = self.clone();
        let id = src.id;
        self.scheduler()
            .submit(Priority::Thumbnail, Some(id), move |token| {
                r.prefetch_raw(&src, token)
            })
    }

    /// How many photos have their raw words on the device.
    #[must_use]
    pub fn resident_photos(&self) -> usize {
        lock(&self.inner.graphs)
            .iter()
            .filter(|(_, g)| g.resident().is_some())
            .count()
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
        // Every id up to `last` is refused now; the single ones set aside by a removal are included.
        lock(&self.inner.removed).clear();
        self.forget_all();
    }

    /// Photos were taken out of the session while it goes on (Remove from filmstrip): refuses viewport, tile, thumbnail,
    /// histogram and readout requests for them from now on ([`Renderer::is_retired`]), cancels the ones queued or running
    /// (device jobs included), forgets what the renderer cached for them and lets go of the photo being looked at when it is
    /// one of them. A request that was still on its way (a command that read its photo before the removal and queues its
    /// job after the cancel) is refused as cancelled, so nothing is cached for a photo that is gone. **Full-resolution
    /// renders for exports are not touched**: they hold their own snapshots and finish the photos the export was started
    /// with ([`Renderer::retire_through`] does the same for a closed session). [`Renderer::unretire_photos`] undoes it.
    pub fn retire_photos(&self, ids: &[PhotoId]) {
        if ids.is_empty() {
            return;
        }
        lock(&self.inner.removed).extend(ids.iter().copied());
        // Marked first, cancelled after: a request that gets past the cancel meets the mark.
        self.scheduler().cancel_photos(ids);
        if let Some(gpu) = self.options().gpu.clone() {
            let owners: Vec<u64> = ids.iter().map(|id| u64::from(id.0)).collect();
            gpu.scheduler().cancel_owners(&owners);
        }
        for id in ids {
            self.forget(*id);
        }
        for slot in [&self.inner.current, &self.inner.previous] {
            let mut slot = lock(slot);
            if slot.is_some_and(|p| ids.contains(&p)) {
                *slot = None;
            }
        }
    }

    /// Photos put back into the session (Undo of a removal): requests for them are served again.
    pub fn unretire_photos(&self, ids: &[PhotoId]) {
        lock(&self.inner.removed).retain(|id| !ids.contains(id));
    }

    /// Whether a photo belongs to a session that was closed ([`Renderer::retire_through`]) or was taken out of the
    /// filmstrip ([`Renderer::retire_photos`]).
    #[must_use]
    pub fn is_retired(&self, id: PhotoId) -> bool {
        id.0 <= self.inner.retired_through.load(Ordering::SeqCst)
            || lock(&self.inner.removed).contains(&id)
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
}

/// What a cache of the raw group holds for one key (`docs/17-gpu.md` section 3): a host image, or the
/// buffers the device made it in, which stay on the device for the sampler to read.
///
/// An entry is one or the other and never both: a consumer that needs host pixels (the CPU colour
/// pass, a fall-back after a device error) reads the device's back and the entry becomes a host one
/// ([`Job::group`]), so the cache's byte count is the memory the entry takes where it is.
#[derive(Clone)]
pub(crate) enum Backing {
    /// A host image.
    Cpu(Arc<RgbImage>),
    /// Device buffers, in pieces.
    Gpu(Arc<ResidentImage>),
}

impl Backing {
    /// Bytes the entry takes, where it is.
    fn bytes(&self) -> usize {
        match self {
            Self::Cpu(image) => image.data.len() * 4,
            Self::Gpu(resident) => resident.device_bytes(),
        }
    }

    /// Width and height of the image in pixels.
    pub(crate) fn size(&self) -> (usize, usize) {
        match self {
            Self::Cpu(image) => (image.width, image.height),
            Self::Gpu(resident) => (resident.width, resident.height),
        }
    }
}

/// An image of the raw group (three planes of camera RGB) on the device: the bands of a reduced group,
/// or a tile of the full-resolution group.
pub(crate) struct ResidentImage {
    pieces: Vec<Piece>,
    width: usize,
    height: usize,
    /// The device the buffers belong to. A reader on another device (the renderer was given a new one)
    /// reads the host copy instead.
    gpu: Weak<Gpu>,
}

impl ResidentImage {
    /// The bands of a group made on `gpu`.
    fn of_bands(bands: &RowBands, gpu: &Arc<Gpu>) -> Self {
        Self {
            pieces: bands.pieces(),
            width: bands.width,
            height: bands.height,
            gpu: Arc::downgrade(gpu),
        }
    }

    /// A tile made on `gpu`.
    fn of_tile(image: Arc<GpuImage>, gpu: &Arc<Gpu>) -> Self {
        Self {
            width: image.width,
            height: image.height,
            pieces: vec![Piece { at: (0, 0), image }],
            gpu: Arc::downgrade(gpu),
        }
    }

    /// The buffers, with the place of each in the image.
    pub(crate) fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// Whether the buffers are on this device.
    pub(crate) fn is_on(&self, gpu: &Weak<Gpu>) -> bool {
        Weak::ptr_eq(&self.gpu, gpu)
    }

    fn device_bytes(&self) -> usize {
        self.pieces
            .iter()
            .map(|p| p.image.byte_len() as usize)
            .sum()
    }

    /// Reads the image back (a job on the device thread).
    fn read(&self, ctx: &GpuContext) -> Result<RgbImage, GpuError> {
        let (w, h) = (self.width, self.height);
        let mut out = RgbImage::new(w, h);
        for p in &self.pieces {
            let (pw, ph) = (p.image.width, p.image.height);
            let part = p.image.read(ctx)?;
            for c in 0..3 {
                for y in 0..ph {
                    let dst = c * w * h + (p.at.1 + y) * w + p.at.0;
                    let src = c * pw * ph + y * pw;
                    out.data[dst..dst + pw].copy_from_slice(&part[src..src + pw]);
                }
            }
        }
        Ok(out)
    }
}

/// Where a cached image came from: the key to find it again by, and the way to make it on the host
/// when the device cannot hand it over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartSource {
    /// The raw group at 1/k.
    Group(usize),
    /// The tile `(tx, ty)` of the full-resolution group.
    Tile(usize, usize),
}

/// A cached image on the device and its place in the buffer a view samples from.
#[derive(Clone)]
pub(crate) struct ResidentPart {
    pub image: Arc<ResidentImage>,
    /// Column and row of the image's first pixel in the buffer.
    pub at: (usize, usize),
    pub source: PartSource,
}

/// The pixels a view samples from.
#[derive(Clone)]
pub(crate) enum Pixels {
    /// A host image; its pixel (0, 0) is the view's `origin`.
    Host(Arc<RgbImage>),
    /// Images on the device that together cover `rect` (buffer pixels) and may reach beyond it.
    Resident {
        parts: Vec<ResidentPart>,
        rect: Rect,
    },
}

impl Pixels {
    /// The rectangle of the buffer the pixels cover, `origin` being where a host image's first pixel
    /// is (the device's pieces know their own place).
    pub(crate) fn rect(&self, origin: (usize, usize)) -> Rect {
        match self {
            Self::Host(image) => Rect::new(origin.0, origin.1, image.width, image.height),
            Self::Resident { rect, .. } => *rect,
        }
    }
}

/// A tile of the full-resolution raw group: the host's image, or one the device made and left where it
/// is (`docs/17-gpu.md` section 3), for the stages that follow on the device to read.
pub enum TileImage {
    /// Camera RGB on the host.
    Host(RgbImage),
    /// Camera RGB on the device, three planes. The buffers go back to the device's pool when the
    /// last owner lets go.
    Device(Arc<GpuImage>),
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
    /// tiles that are left on the CPU (never a mix of devices in one export, `docs/17-gpu.md` section 9).
    restartable: bool,
    /// The device error that asked for such a restart ([`Job::take_gpu_restart`]).
    restart: Option<String>,
    /// A restart on the CPU was taken ([`Job::take_gpu_restart`]): this request's picture is the
    /// CPU's, made again after a device error ([`Job::gpu_restarted`]).
    gpu_restarted: bool,
    /// Where the raw stages ran, per range, for the tiles of this request.
    raw_ranges: Vec<RangeDevice>,
    /// Whether R8 capture sharpening runs for this request. Off for the renders of a DNG, which
    /// carries no sharpening ([`Job::without_capture_sharpening`]).
    capture_sharpening: bool,
    /// The photo's raw words could not be put on the device for this request: its tiles run R6..R8
    /// behind the CPU's front ([`DEVICE_RANGE`]) from the first one to the last, never a mix.
    front_failed: bool,
    /// This request pins the photo's raw words on the device while it lives ([`Job::for_export`]).
    pinned: bool,
}

impl Drop for Job<'_> {
    /// An export lets go of the photo's raw words when it is done (a batch of exports would
    /// otherwise leave the words of two photos on the device behind it), unless another export is
    /// still rendering the photo or the photo is the one being looked at: the person goes on
    /// editing it.
    fn drop(&mut self) {
        let was_export = self.pinned;
        self.unpin();
        if was_export
            && self.g.pins.load(Ordering::SeqCst) == 0
            && *lock(&self.r.inner.current) != Some(self.src.id)
        {
            self.g.drop_resident();
        }
        // An export that rendered a photo no one can look at any more (it was taken out of the filmstrip, or its session
        // was closed) leaves no cache behind: the graph it made is of no use to anything else.
        if was_export && self.g.pins.load(Ordering::SeqCst) == 0 && self.r.is_retired(self.src.id) {
            self.r.forget(self.src.id);
        }
    }
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
        job.unpin();
        Ok(job)
    }

    /// A request that is not refused for a photo of a closed session: an export, which goes on
    /// with its snapshot.
    pub fn for_export(r: &'a Renderer, src: &'a PhotoSource, cancel: &'a CancelToken) -> Self {
        let g = r.graph(src.id);
        g.adopt(src.fingerprint);
        g.pins.fetch_add(1, Ordering::SeqCst);
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
            gpu_restarted: false,
            raw_ranges: Vec::new(),
            capture_sharpening: true,
            front_failed: false,
            pinned: true,
        }
    }

    /// The raw group of this request runs without R8 capture sharpening (neither Richardson-Lucy nor
    /// diffraction deconvolution), on the CPU and on the device alike (both read the chain that is
    /// made from [`Job::params`]). The renders of a DNG use it: `docs/09`, "What a DNG carries".
    /// The demosaic's own shadow step stays on. Call before the first use of [`Job::params`].
    #[must_use]
    pub fn without_capture_sharpening(mut self) -> Self {
        self.capture_sharpening = false;
        self.params = None;
        self
    }

    /// The device jobs of this request run at `priority` (the request's own: `TileNeural` for the
    /// 100 % tiles a viewport prefetches, `Thumbnail` for thumbnails and look tiles).
    #[must_use]
    pub fn with_gpu_priority(mut self, priority: Priority) -> Self {
        self.gpu_priority = priority;
        self
    }

    /// This request runs with `opts` instead of the renderer's options of the moment it began. The
    /// export that falls back to the CPU after a device error uses it with no device, so that the
    /// whole photo is made on the CPU reference path and nothing is left sticky on the photo.
    /// Call before anything reads [`Job::options`].
    #[must_use]
    pub fn with_options(mut self, opts: Arc<RenderOptions>) -> Self {
        self.opts = opts;
        self
    }

    /// The options this request runs with: a snapshot taken when it began.
    pub fn options(&self) -> &RenderOptions {
        &self.opts
    }

    /// Lets go of the pin of [`Job::for_export`] (a request for a view does not pin the photo).
    fn unpin(&mut self) {
        if self.pinned {
            self.g.pins.fetch_sub(1, Ordering::SeqCst);
            self.pinned = false;
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
            ranges: std::mem::take(&mut self.raw_ranges),
            scale,
            runs: self.runs,
            timings: self.timings,
            notes: std::mem::take(&mut self.notes),
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
            // The Before view shows the photo without the lens vignetting correction, as the camera's own
            // picture does (`PhotoSource::for_before_view`); every other render has R3. A photo with no
            // vignetting table to apply (none in the file, the lens switch off, a camera-processed file)
            // is the same with and without R3, so its Before view shares the final picture's raw front
            // and its caches: flipping to Before costs no raw work there.
            vignetting: !(self.src.before_view
                && has_vignetting_profile(&self.src.raw, s.decode.lens_corrections)),
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
        let p = if self.capture_sharpening {
            p
        } else {
            p.without_capture_sharpening()
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

    /// The plan for the lens switches in effect, with R3 or, for the Before view, without it. The
    /// Before view's plan is made from the final picture's when that is cached (the same plan without the
    /// vignetting profile, [`RawPlan::without_vignetting`]: no pass over the pixels), and from the file
    /// otherwise; both ways give the plan [`RawPlan::new`] makes for parameters without R3.
    pub fn plan(&mut self) -> Result<Arc<RawPlan>, RenderError> {
        let params = self.params()?;
        let key = (params.lens_corrections, params.chromatic_aberration);
        let slot = usize::from(!params.vignetting);
        let (cached, with_r3) = {
            let st = lock(&self.g.state);
            let hit = |i: usize| {
                st.plan[i]
                    .as_ref()
                    .filter(|(k, _)| *k == key)
                    .map(|(_, p)| Arc::clone(p))
            };
            (hit(slot), hit(0))
        };
        if let Some(plan) = cached {
            // The notes belong to every request that uses the plan, not only the one that made it.
            for n in &plan.notes {
                self.note(n);
            }
            return Ok(plan);
        }
        let plan = match with_r3 {
            Some(full) if slot == 1 => Arc::new(full.without_vignetting()),
            _ => {
                let frame = self.frame()?;
                Arc::new(RawPlan::new(&self.src.raw, &frame, &params)?)
            }
        };
        for n in &plan.notes {
            self.note(n);
        }
        lock(&self.g.state).plan[slot] = Some((key, Arc::clone(&plan)));
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

    /// Whether the caches of the raw group keep their images on the device: there is a device that
    /// runs the sampler, and [`RenderOptions::resident_caches`] is on.
    fn keeps_on_device(&self) -> bool {
        self.opts.resident_caches
            && self.opts.gpu.is_some()
            && self.opts.gpu_stages.contains(GpuStages::SAMPLER)
    }

    /// The raw group at 1/`k` as a host image. Mosaic photos need `k >= 2` (full resolution goes
    /// through [`Job::region`]); camera-processed photos accept any `k`. A group that is on the device
    /// is read back, and the cache keeps the host image from then on.
    pub fn group(&mut self, k: usize) -> Result<Arc<RgbImage>, RenderError> {
        match self.group_backing(k)? {
            Backing::Cpu(image) => Ok(image),
            Backing::Gpu(resident) => self.demote(PartSource::Group(k), &resident),
        }
    }

    /// The raw group at 1/`k` as the cache holds it: on the device when the raw chain made it there
    /// and the caches keep their images there ([`Job::keeps_on_device`]), else a host image.
    pub(crate) fn group_backing(&mut self, k: usize) -> Result<Backing, RenderError> {
        let params = self.params()?;
        let key = (params.key(), k);
        {
            let budget = self.g.group_budget(self.keeps_on_device());
            let mut st = lock(&self.g.state);
            let groups = st.groups.get_or_insert_with(|| ByteLru::new(budget));
            if let Some(entry) = groups.get(&key) {
                return Ok(entry.clone());
            }
        }
        let frame = self.frame()?;
        let plan = self.plan()?;
        let made = if frame.reduced {
            let t = Instant::now();
            let img = reduced_rgb(&frame, &plan, &params, k, self.cancel)?;
            self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
            Backing::Cpu(Arc::new(img))
        } else {
            debug_assert!(k >= 2, "full resolution of a mosaic goes through tiles");
            match self.group_on_device(&frame, &plan, &params, k)? {
                Some(made) => {
                    self.runs.device_groups += 1;
                    made
                }
                None => Backing::Cpu(Arc::new(self.group_on_host(&frame, &plan, &params, k)?)),
            }
        };
        self.runs.raw_groups += 1;
        let (bytes, budget) = (made.bytes(), self.g.group_budget(self.keeps_on_device()));
        lock(&self.g.state)
            .groups
            .get_or_insert_with(|| ByteLru::new(budget))
            .insert(key, made.clone(), bytes);
        Ok(made)
    }

    /// The raw group of a mosaic photo at 1/`k` on the CPU reference path.
    fn group_on_host(
        &mut self,
        frame: &Arc<FrameData>,
        plan: &RawPlan,
        params: &RawParams,
        k: usize,
    ) -> Result<RgbImage, RenderError> {
        let sigma = self.sigma()?;
        let t = Instant::now();
        let img = reduced_bayer(frame, plan, params, &sigma, k, self.cancel)?;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        Ok(img)
    }

    /// The raw group at 1/`k` made on the device from the photo's resident raw words (`GpuReduced`:
    /// R1, R4, R5, the binning, the vignetting gain, the sigma, the light denoise and R9 at 1/k) as a
    /// series of jobs ([`GpuRoute::reduced_resident`]), or `None` when the CPU makes it: there is no
    /// device, the raw front is off ([`GpuStages::RAW_FRONT`]), a device error ended the route, or the
    /// device cannot take the photo's parameters. The sigma plane is made on the device too, so the
    /// host's plane ([`Job::sigma`]) is not built for this group.
    ///
    /// The group is left on the device when the caches keep their images there
    /// ([`Job::keeps_on_device`]), else it is read back.
    ///
    /// A device error falls back to the CPU for this group with a note; one that is not about this
    /// request's size also ends the route for the photo (as for a tile). A strict policy fails the
    /// request.
    fn group_on_device(
        &mut self,
        frame: &Arc<FrameData>,
        plan: &RawPlan,
        params: &RawParams,
        k: usize,
    ) -> Result<Option<Backing>, RenderError> {
        // An export makes full-resolution tiles and never a reduced group.
        if self.restartable {
            return Ok(None);
        }
        let Some(gpu) = self.raw_device(GpuStages::RAW_FRONT)? else {
            return Ok(None);
        };
        let Some(device_params) = reduced_params(frame, plan, params, k)? else {
            return Ok(None);
        };
        let route = GpuRoute {
            gpu: Arc::clone(&gpu),
            range: FRONT_RANGE,
            spec: JobSpec::new(self.gpu_priority).owned_by(u64::from(self.src.id.0)),
            token: self.cancel.clone(),
            resident: None,
        };
        let keep = self.keeps_on_device();
        let t = Instant::now();
        let result = self
            .resident(&route, frame)
            .and_then(|raw| -> Result<Backing, GpuError> {
                if keep {
                    let bands = route.reduced_resident(&raw, device_params)?;
                    Ok(Backing::Gpu(Arc::new(ResidentImage::of_bands(
                        &bands, &gpu,
                    ))))
                } else {
                    route
                        .reduced(&raw, device_params)
                        .map(|i| Backing::Cpu(Arc::new(i)))
                }
            });
        match result {
            Ok(made) => {
                self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
                Ok(Some(made))
            }
            Err(GpuError::Cancelled) => Err(RenderError::Cancelled),
            Err(e) => {
                self.group_gpu_failed(&e)?;
                Ok(None)
            }
        }
    }

    /// The host image of a cached image that is on the device: read back, and the cache entry made a
    /// host one (an entry is on one side only, [`Backing`]). When the device cannot hand it over (it
    /// is gone, it failed, it is not the one the image is on), the CPU makes the image again, with a
    /// note, and the cache keeps that.
    ///
    /// # Errors
    ///
    /// [`RenderError::Cancelled`]; a strict policy's [`RenderError::Gpu`]; the CPU path's errors.
    fn demote(
        &mut self,
        source: PartSource,
        resident: &Arc<ResidentImage>,
    ) -> Result<Arc<RgbImage>, RenderError> {
        let read = match resident.gpu.upgrade() {
            Some(gpu) if !gpu.is_lost() => {
                let route = self.device_route(gpu);
                let image = Arc::clone(resident);
                match route.run_job(move |ctx, _| image.read(ctx)) {
                    Ok(img) => Some(img),
                    Err(GpuError::Cancelled) => return Err(RenderError::Cancelled),
                    Err(e) => {
                        self.group_gpu_failed(&e)?;
                        None
                    }
                }
            }
            _ => {
                if self.opts.gpu_strict {
                    return Err(RenderError::Gpu(
                        "the graphics card that holds the image is no longer there".to_owned(),
                    ));
                }
                None
            }
        };
        let image = match read {
            Some(img) => Arc::new(img),
            None => Arc::new(self.make_on_host(source)?),
        };
        self.keep_host(source, &image);
        Ok(image)
    }

    /// The pixels as one host image whose pixel (0, 0) is the first of [`Pixels::rect`]: the host
    /// image itself, or the device's images read back ([`Job::demote`]: each cache entry becomes a
    /// host one) and put together.
    ///
    /// # Errors
    ///
    /// As [`Job::demote`].
    pub(crate) fn host_pixels(&mut self, pixels: &Pixels) -> Result<Arc<RgbImage>, RenderError> {
        let (parts, rect) = match pixels {
            Pixels::Host(image) => return Ok(Arc::clone(image)),
            Pixels::Resident { parts, rect } => (parts, *rect),
        };
        let mut hosts = Vec::with_capacity(parts.len());
        for part in parts {
            hosts.push(self.demote(part.source, &part.image)?);
        }
        if let ([part], [host]) = (parts.as_slice(), hosts.as_slice())
            && part.at == (rect.x, rect.y)
            && (host.width, host.height) == (rect.w, rect.h)
        {
            return Ok(Arc::clone(host));
        }
        let mut out = RgbImage::new(rect.w, rect.h);
        for (part, host) in parts.iter().zip(&hosts) {
            let at = Rect::new(part.at.0, part.at.1, host.width, host.height);
            let ov = at.intersect(&rect);
            if ov.is_empty() {
                continue;
            }
            out.paste(
                host,
                Rect::new(ov.x - at.x, ov.y - at.y, ov.w, ov.h),
                ov.x - rect.x,
                ov.y - rect.y,
            );
        }
        Ok(Arc::new(out))
    }

    /// The route for the readback of an image: the request's priority and token.
    fn device_route(&self, gpu: Arc<Gpu>) -> GpuRoute {
        GpuRoute {
            gpu,
            range: FRONT_RANGE,
            spec: JobSpec::new(self.gpu_priority).owned_by(u64::from(self.src.id.0)),
            token: self.cancel.clone(),
            resident: None,
        }
    }

    /// Makes an image of the raw group again on the host (a group on the CPU path, a tile as
    /// [`Job::run_full_tile`] makes it).
    fn make_on_host(&mut self, source: PartSource) -> Result<RgbImage, RenderError> {
        match source {
            PartSource::Group(k) => {
                let params = self.params()?;
                let frame = self.frame()?;
                let plan = self.plan()?;
                self.group_on_host(&frame, &plan, &params, k)
            }
            PartSource::Tile(tx, ty) => {
                let (full, frame, rect) = self.tile_inputs(tx, ty, true)?;
                self.run_full_tile(&full, &frame, rect)
            }
        }
    }

    /// Puts a host image in the place of the cache entry of `source`.
    fn keep_host(&mut self, source: PartSource, image: &Arc<RgbImage>) {
        let Ok(params) = self.params() else { return };
        let key = params.key();
        let bytes = image.data.len() * 4;
        let mut st = lock(&self.g.state);
        match source {
            PartSource::Group(k) => {
                if let Some(groups) = st.groups.as_mut() {
                    groups.insert((key, k), Backing::Cpu(Arc::clone(image)), bytes);
                }
            }
            PartSource::Tile(tx, ty) => {
                if let Some(tiles) = st.tiles.as_mut() {
                    tiles.insert((key, tx, ty), Backing::Cpu(Arc::clone(image)), bytes);
                }
            }
        }
    }

    /// A reduced group failed on the device with `error`: the policy of [`Job::run_full_tile`] for a
    /// request that can fall back on its own. A shader that does not compile switches the raw chain
    /// off for the process; a strict policy fails the request; any other policy notes why the CPU
    /// makes this group. An error about the request's size (a buffer that does not fit, a frame the
    /// device refuses) is a property of the request and not of the photo, and the next one tries the
    /// device again.
    fn group_gpu_failed(&mut self, error: &GpuError) -> Result<(), RenderError> {
        let reason = error.to_string();
        tracing::warn!(error = %reason, "the device failed on a reduced group; the CPU takes over");
        if matches!(error, GpuError::Shader(_)) {
            self.r.inner.raw_gpu_disabled.store(true, Ordering::SeqCst);
        }
        if self.opts.gpu_strict {
            return Err(RenderError::Gpu(reason));
        }
        if !matches!(
            error,
            GpuError::OutOfBudget { .. }
                | GpuError::BufferTooLarge { .. }
                | GpuError::InvalidInput(_)
        ) {
            self.g.raw_gpu_failed.store(true, Ordering::SeqCst);
        }
        self.note(&format!(
            "The graphics card could not render this frame, so the CPU did: {reason}."
        ));
        Ok(())
    }

    /// The prepared full-resolution chain (mosaic photos).
    pub fn full(&mut self, keep: bool) -> Result<Arc<FullChain>, RenderError> {
        let params = self.params()?;
        let key = params.key();
        let slot = usize::from(!params.vignetting);
        if let Some((k, f)) = &lock(&self.g.state).full[slot]
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
            lock(&self.g.state).full[slot] = Some((key, Arc::clone(&full)));
        } else {
            self.transient_full = Some((key, Arc::clone(&full)));
        }
        Ok(full)
    }

    /// The chain, the decoded area and the rectangle of the tile `(tx, ty)` of the full-resolution
    /// raw group.
    fn tile_inputs(
        &mut self,
        tx: usize,
        ty: usize,
        keep: bool,
    ) -> Result<(Arc<FullChain>, Arc<FrameData>, Rect), RenderError> {
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
        Ok((full, frame, rect))
    }

    /// One 512 px tile of the full-resolution raw group as a host image (a tile on the device is read
    /// back, and the cache keeps the host image from then on).
    fn tile(&mut self, tx: usize, ty: usize, keep: bool) -> Result<Arc<RgbImage>, RenderError> {
        match self.tile_backing(tx, ty, keep)? {
            Backing::Cpu(image) => Ok(image),
            Backing::Gpu(resident) => self.demote(PartSource::Tile(tx, ty), &resident),
        }
    }

    /// One 512 px tile of the full-resolution raw group as the cache holds it: left on the device
    /// when the raw chain made it there and the caches keep their images there
    /// ([`Job::keeps_on_device`]), else a host image. A tile that is not kept (`keep` false) is
    /// always a host image.
    fn tile_backing(&mut self, tx: usize, ty: usize, keep: bool) -> Result<Backing, RenderError> {
        let params = self.params()?;
        let key = (params.key(), tx, ty);
        let on_device = self.keeps_on_device();
        if keep {
            let budget = self.g.tile_budget(on_device);
            let mut st = lock(&self.g.state);
            let tiles = st.tiles.get_or_insert_with(|| ByteLru::new(budget));
            if let Some(entry) = tiles.get(&key) {
                return Ok(entry.clone());
            }
        }
        self.check()?;
        let (full, frame, rect) = self.tile_inputs(tx, ty, keep)?;
        let t = Instant::now();
        let made = if keep && on_device {
            match self.run_full_tile_any(&full, &frame, rect, true)? {
                TileImage::Device(image) => {
                    let gpu = self.opts.gpu.clone().ok_or_else(|| {
                        RenderError::Image(
                            "A tile was left on a graphics card that is not attached.".to_owned(),
                        )
                    })?;
                    Backing::Gpu(Arc::new(ResidentImage::of_tile(image, &gpu)))
                }
                TileImage::Host(image) => Backing::Cpu(Arc::new(image)),
            }
        } else {
            Backing::Cpu(Arc::new(self.run_full_tile(&full, &frame, rect)?))
        };
        self.runs.tiles += 1;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        if keep {
            let (bytes, budget) = (made.bytes(), self.g.tile_budget(on_device));
            lock(&self.g.state)
                .tiles
                .get_or_insert_with(|| ByteLru::new(budget))
                .insert(key, made.clone(), bytes);
        }
        Ok(made)
    }

    /// Makes the tiles of the 100 % view that cover `rect` and leaves them in the cache where they
    /// are made (on the device when the caches keep their images there), bringing no pixel to the
    /// host: the work ahead of a view ([`Renderer::prepare_tiles`]).
    pub fn warm_tiles(&mut self, rect: Rect) -> Result<(), RenderError> {
        let plan = self.plan()?;
        if rect.is_empty() || rect.x1() > plan.out_w || rect.y1() > plan.out_h {
            return Err(RenderError::Image(
                "The region is outside the photo.".to_owned(),
            ));
        }
        if plan.reduced {
            return self.group_backing(1).map(|_| ());
        }
        let t = self.opts.tile_px.max(64);
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                self.tile_backing(tx, ty, true)?;
            }
        }
        Ok(())
    }

    /// The pixels a view of the 100 % path samples from: the full-resolution raw group over `rect`,
    /// out of the cached tiles. When every tile is on the device they stay there, and the sampler
    /// reads them in place ([`Pixels::Resident`]); else the host assembles the region
    /// ([`Job::region`]).
    pub(crate) fn region_pixels(&mut self, rect: Rect) -> Result<Pixels, RenderError> {
        let plan = self.plan()?;
        if !self.keeps_on_device()
            || plan.reduced
            || rect.is_empty()
            || rect.x1() > plan.out_w
            || rect.y1() > plan.out_h
        {
            return self.region(rect, true).map(|r| Pixels::Host(Arc::new(r)));
        }
        let t = self.opts.tile_px.max(64);
        let mut parts = Vec::new();
        for ty in rect.y / t..=(rect.y1() - 1) / t {
            for tx in rect.x / t..=(rect.x1() - 1) / t {
                self.check()?;
                match self.tile_backing(tx, ty, true)? {
                    Backing::Gpu(image) => parts.push(ResidentPart {
                        image,
                        at: (tx * t, ty * t),
                        source: PartSource::Tile(tx, ty),
                    }),
                    // A tile the CPU made: the host assembles the region (the tiles cached on the
                    // device are read back as they are pasted).
                    Backing::Cpu(_) => {
                        return self.region(rect, true).map(|r| Pixels::Host(Arc::new(r)));
                    }
                }
            }
        }
        Ok(Pixels::Resident { parts, rect })
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

    /// The full-resolution raw group over `rect` for an export's colour stage, left on the device when
    /// the raw chain made it there ([`TileImage`]): one call over the whole rectangle (the stage halos
    /// are paid once), nothing cached, no readback and no upload between the raw chain and the sampler
    /// that follows. A tile the CPU made, or the device made behind the CPU's front, comes back as a
    /// host image, as [`Job::region`] gives it.
    pub fn region_resident(&mut self, rect: Rect) -> Result<TileImage, RenderError> {
        let plan = self.plan()?;
        if rect.is_empty() || rect.x1() > plan.out_w || rect.y1() > plan.out_h {
            return Err(RenderError::Image(
                "The region is outside the photo.".to_owned(),
            ));
        }
        if plan.reduced {
            return self.region(rect, false).map(TileImage::Host);
        }
        self.check()?;
        let frame = self.frame()?;
        let full = self.full(false)?;
        let t = Instant::now();
        let img = self.run_full_tile_any(&full, &frame, rect, true)?;
        self.runs.tiles += 1;
        self.timings.raw_ms += t.elapsed().as_secs_f64() * 1000.0;
        Ok(img)
    }

    /// The device for the raw stages of a request and the stage that lets it run them, or `None` when
    /// the CPU runs them: there is no device, `stage` is not on, a device error ended the route for
    /// this photo or for the process, or the device is lost. Reads the options of the request's
    /// snapshot.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] when the device is lost, or an earlier device error ended the route, and
    /// the policy is strict.
    fn raw_device(&mut self, stage: GpuStages) -> Result<Option<Arc<Gpu>>, RenderError> {
        let Some(gpu) = self.opts.gpu.clone() else {
            return Ok(None);
        };
        if !self.opts.gpu_stages.contains(stage) {
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
        Ok(Some(gpu))
    }

    /// The route to the device for the raw chain of this request, or `None` when the CPU runs it:
    /// there is no device, the stage is not on ([`GpuStages::RAW_CHAIN`]), a device error ended the
    /// route for this photo or for the process, the device is lost, or the demosaic method is not the
    /// one the device runs (a test switch). Reads the options of the request's snapshot.
    ///
    /// The route runs R1 to R9 ([`FRONT_RANGE`]) when the raw front is on too
    /// ([`GpuStages::RAW_FRONT`]) and the photo's raw words are on the device ([`Job::resident`]
    /// uploads them on first use); when they cannot be put there the route runs R6 to R8
    /// ([`DEVICE_RANGE`]) behind the CPU's front, for the rest of the request.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] when the device is lost, or an earlier device error ended the route, and
    /// the policy is strict; also when the upload fails under a strict policy.
    fn raw_route(&mut self) -> Result<Option<GpuRoute>, RenderError> {
        let Some(gpu) = self.raw_device(GpuStages::RAW_CHAIN)? else {
            return Ok(None);
        };
        if self.opts.demosaic != Method::AdaptiveRatio {
            self.note(METHOD_NOT_ON_DEVICE_NOTE);
            return Ok(None);
        }
        let mut route = GpuRoute {
            gpu,
            range: DEVICE_RANGE,
            spec: JobSpec::new(self.gpu_priority).owned_by(u64::from(self.src.id.0)),
            token: self.cancel.clone(),
            resident: None,
        };
        if self.opts.gpu_stages.contains(GpuStages::RAW_FRONT) && !self.front_failed {
            let frame = self.frame()?;
            match self.resident(&route, &frame) {
                Ok(raw) => {
                    route.range = FRONT_RANGE;
                    route.resident = Some(raw);
                }
                Err(GpuError::Cancelled) => return Err(RenderError::Cancelled),
                Err(e) => {
                    let reason = e.to_string();
                    tracing::warn!(error = %reason, "the photo's raw words could not be put on the device; the CPU runs the raw front");
                    if self.opts.gpu_strict {
                        return Err(RenderError::Gpu(reason));
                    }
                    if self.restartable
                        && self
                            .raw_ranges
                            .iter()
                            .any(|r| r.range == FRONT_RANGE && r.device == ComputeDevice::Gpu)
                    {
                        // Earlier tiles of this export ran the front on the device: the rest must
                        // not run behind the CPU's front (one file, one device). Start again on
                        // the CPU, as for a device error in a tile: the photo's raw route is off, so
                        // that a render which goes on in this very request (a DNG, an export without
                        // the tail) makes every tile again on the CPU and not on R6..R8 behind the
                        // CPU's front, which is not the CPU reference.
                        self.raw_ranges.clear();
                        self.front_failed = true;
                        self.g.raw_gpu_failed.store(true, Ordering::SeqCst);
                        self.restart = Some(reason.clone());
                        return Err(RenderError::Gpu(reason));
                    }
                    self.front_failed = true;
                }
            }
        }
        Ok(Some(route))
    }

    /// The range this request expects its tiles to run on the device (what a CPU fall-back is
    /// recorded against): the whole chain when the raw front is on, whether or not the photo's words
    /// could be put on the device for this request (the tiles that did run behind the CPU's front say
    /// [`DEVICE_RANGE`] themselves).
    fn expected_range(&self) -> StageRange {
        if self.opts.gpu_stages.contains(GpuStages::RAW_FRONT) {
            FRONT_RANGE
        } else {
            DEVICE_RANGE
        }
    }

    /// The photo's raw words on the device: uploaded on first use as a job of `route`, kept in the
    /// photo's graph, and let go of with the photo's place among the [`RESIDENT_PHOTOS`] most recent
    /// ones (navigation, closing). Two requests for one photo upload once.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when the request's token was set; the upload's errors (a frame that
    /// does not fit the device, the pool's budget).
    fn resident(
        &mut self,
        route: &GpuRoute,
        frame: &Arc<FrameData>,
    ) -> Result<Arc<DeviceRaw>, GpuError> {
        if let Some(raw) = self.g.resident() {
            self.r.touch_resident(&self.g);
            return Ok(raw);
        }
        let g = Arc::clone(&self.g);
        let _gate = lock(&g.upload_gate);
        if let Some(raw) = g.resident() {
            // Another request uploaded while this one waited at the gate.
            self.r.touch_resident(&g);
            return Ok(raw);
        }
        // Room for one more, when this is a photo the person is on: of the photos that already have
        // theirs on the device, the newest stays. (Any other photo is let go of after its upload if the
        // two the person is on are there; nothing is made room for.)
        if self.r.is_protected(self.src.id) {
            self.r.trim_resident(RESIDENT_PHOTOS - 1);
        }
        let raw = route.upload(frame)?;
        {
            let mut st = lock(&g.state);
            if st.fingerprint == Some(self.src.fingerprint) {
                st.resident = Some(Arc::clone(&raw));
            }
        }
        self.r.touch_resident(&g);
        // A photo that is neither the one looked at nor the one before it, uploaded while those two
        // are there (a thumbnail of the folder), serves this request and is let go of at once.
        self.r.trim_resident(RESIDENT_PHOTOS);
        self.runs.raw_uploads += 1;
        Ok(raw)
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
        Ok(Some(ViewGpu { gpu, stages }))
    }

    /// A view stage failed on the device with `error`. A shader that does not compile switches the
    /// view stages off for the process. A strict policy fails the request; any other policy notes
    /// why the CPU makes this frame, and ends the route for the photo (view caches may then hold
    /// frames of both devices, which differ within the tolerances of the kernels; viewing only,
    /// `docs/17-gpu.md` section 9). A buffer that does not fit (out of budget, too large) is a property
    /// of the request and not of the photo: the CPU makes this frame, and the next one tries the
    /// device again.
    ///
    /// # Errors
    ///
    /// [`RenderError::Gpu`] under a strict policy.
    pub(crate) fn view_gpu_failed(&mut self, error: &GpuError) -> Result<(), RenderError> {
        let reason = error.to_string();
        tracing::warn!(error = %reason, "the device failed on a view; the CPU takes over");
        if matches!(error, GpuError::Shader(_)) {
            self.r.inner.view_gpu_disabled.store(true, Ordering::SeqCst);
        }
        if self.opts.gpu_strict {
            return Err(RenderError::Gpu(reason));
        }
        if !matches!(
            error,
            GpuError::OutOfBudget { .. } | GpuError::BufferTooLarge { .. }
        ) {
            self.g.view_gpu_failed.store(true, Ordering::SeqCst);
        }
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
        match self.run_full_tile_any(full, frame, rect, false)? {
            TileImage::Host(image) => Ok(image),
            TileImage::Device(_) => Err(RenderError::Image(
                "A tile was left on the graphics card where the host's was asked for.".to_owned(),
            )),
        }
    }

    /// [`run_full_tile`](Self::run_full_tile) that may leave the tile on the device (`device`): the
    /// range R1..R9 from the photo's resident words ends there ([`FullChain::run_tile_gpu_device`]);
    /// every other way a tile is made (the CPU, the range R6..R8 behind the CPU's front) gives a host
    /// image. The policy is the same: a device error ends the route for the photo, an export asks to
    /// start again on the CPU, a cancelled render stays cancelled.
    fn run_full_tile_any(
        &mut self,
        full: &FullChain,
        frame: &FrameData,
        rect: Rect,
        device: bool,
    ) -> Result<TileImage, RenderError> {
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
            let made = if device && route.range == FRONT_RANGE {
                full.run_tile_gpu_device(rect, &route)
                    .map(|image| TileImage::Device(Arc::new(image)))
            } else {
                full.run_tile_gpu(frame, rect, &route).map(TileImage::Host)
            };
            match made {
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
            self.record_range(self.expected_range(), ComputeDevice::Cpu);
        }
        full.run_tile(frame, rect).map(TileImage::Host)
    }

    /// Whether the last error was a device error that asks the export to start again on the CPU
    /// (see [`Job::run_full_tile`]). Clears the request, drops what the device had made and adds the
    /// note. The photo's route is already off, so the restart runs on the CPU from the first tile.
    pub fn take_gpu_restart(&mut self) -> bool {
        let Some(reason) = self.restart.take() else {
            return false;
        };
        self.raw_ranges.clear();
        self.gpu_restarted = true;
        self.note(&format!(
            "The graphics card could not render this photo, so the CPU did: {reason}."
        ));
        true
    }

    /// Whether this request was made again on the CPU after a device error
    /// ([`Job::take_gpu_restart`] took a restart): the typed form of the note that says so, for an
    /// export that is more than one render ([`FullImage::fell_back_to_cpu`](super::frame::FullImage::fell_back_to_cpu)).
    pub(crate) fn gpu_restarted(&self) -> bool {
        self.gpu_restarted
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
