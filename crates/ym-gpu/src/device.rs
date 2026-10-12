//! Device ownership: one thread owns the wgpu `Device` and `Queue`; adapter selection, feature
//! detection, requested limits and the pipeline-cache hook (`docs/01-architecture.md` sections
//! Scheduling and threads and GPU tiling).
//!
//! # Ownership model
//!
//! [`GpuContext`] holds the `wgpu::Device`, the `wgpu::Queue`, the [`BufferPool`] and the compiled
//! kernels. It is created **on** the device thread and never leaves it: every piece of GPU work is a
//! closure submitted through [`Gpu::submit`] (priority queue, cancellation tokens, see
//! [`crate::queue`]) and run there with `&mut GpuContext`. A `Gpu` handle is cheap to share between
//! threads; dropping the last handle shuts the thread down.
//!
//! # Adapter selection
//!
//! 1. The backends come from `GpuConfig::backends` or, when that is `None`, from the `WGPU_BACKEND`
//!    environment variable (all backends if it is unset).
//! 2. A high-performance adapter is requested first; if wgpu offers none, the fallback (software)
//!    adapter is tried, then any adapter that enumerates. A CPU adapter (Mesa lavapipe) is therefore
//!    accepted, which is what CI uses, and is reported as such in [`GpuInfo`].
//!
//! # Storage mode
//!
//! Image intermediates are f16 in memory and f32 in registers (`docs/01`, GPU tiling). How the f16
//! is stored depends on the device ([`StorageMode`]): native `array<f16>` when `SHADER_F16` is
//! available, else `array<u32>` holding two halves each, packed with `pack2x16float`, else plain
//! f32. The choice is made once at start-up and written into every generated shader.
//!
//! # Pipeline cache
//!
//! `GpuConfig::pipeline_cache_dir` is the hook for the wgpu pipeline cache (the only thing that may
//! live under `~/Library/Caches/app.yellowmoon/`, `docs/01` section Persistence; the integrator
//! passes [`default_pipeline_cache_dir`]). When the adapter supports `PIPELINE_CACHE` (Vulkan today)
//! the cache is loaded at start and written back, atomically, by [`GpuContext::save_pipeline_cache`].
//! No image data is ever written.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::kernels::Kernels;
use crate::pool::{BufferPool, MemoryBudget};
use crate::queue::{Cancelled, JobError, JobHandle, JobSpec, Scheduler, SpawnError};

/// Errors from device start-up and from GPU work.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No adapter could be opened. The text lists what was tried.
    #[error("no usable GPU adapter: {0}")]
    NoAdapter(String),
    /// The adapter refused the device request.
    #[error("the GPU device could not be created: {0}")]
    RequestDevice(String),
    /// A shader failed to compile or a pipeline failed validation.
    #[error("GPU shader error: {0}")]
    Shader(String),
    /// An allocation would exceed the GPU memory budget even after freeing unused buffers.
    #[error(
        "GPU memory budget exceeded: {requested} bytes requested, {allocated} of {budget} bytes in use"
    )]
    OutOfBudget {
        /// Bytes asked for.
        requested: u64,
        /// Bytes currently allocated (in use plus pooled).
        allocated: u64,
        /// The budget.
        budget: u64,
    },
    /// A single buffer is larger than the adapter allows.
    #[error("a {requested} byte buffer is larger than the adapter's limit of {limit} bytes")]
    BufferTooLarge {
        /// Bytes asked for.
        requested: u64,
        /// The limit that applies.
        limit: u64,
    },
    /// Reading a result back from the GPU failed.
    #[error("GPU readback failed: {0}")]
    Readback(String),
    /// The inputs of a kernel are inconsistent.
    #[error("invalid kernel input: {0}")]
    InvalidInput(String),
    /// The job was cancelled before or while it ran.
    #[error("the GPU job was cancelled")]
    Cancelled,
    /// The device thread has shut down.
    #[error("the GPU thread has shut down")]
    Shutdown,
    /// The job panicked on the device thread.
    #[error("the GPU job panicked: {0}")]
    Panicked(String),
    /// The display chain could not be prepared.
    #[error("colour chain: {0}")]
    Chain(String),
    /// A CPU reference stage rejected its inputs.
    #[error("CPU reference: {0}")]
    Cpu(String),
}

impl From<JobError> for GpuError {
    fn from(e: JobError) -> Self {
        match e {
            JobError::Cancelled => GpuError::Cancelled,
            JobError::Shutdown => GpuError::Shutdown,
            JobError::Panicked(m) => GpuError::Panicked(m),
            JobError::WouldDeadlock => GpuError::InvalidInput(
                "a GPU job waited for another GPU job from the GPU thread; submit it and return \
                 its handle instead"
                    .to_string(),
            ),
        }
    }
}

impl From<Cancelled> for GpuError {
    fn from(_: Cancelled) -> Self {
        GpuError::Cancelled
    }
}

impl From<crate::cpu::CpuError> for GpuError {
    fn from(e: crate::cpu::CpuError) -> Self {
        GpuError::Cpu(e.to_string())
    }
}

/// How f16 image data is held in GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageMode {
    /// `array<f16>` buffers (`SHADER_F16`).
    F16Native,
    /// `array<u32>` buffers, two f16 per word, packed with `pack2x16float` /
    /// `unpack2x16float` (the "f32 storage with f16 packing" of `docs/01`).
    F16Packed,
    /// `array<f32>` buffers: no f16 at all (twice the memory; used when neither of the above works).
    F32,
}

impl StorageMode {
    /// Bytes one image sample occupies in memory.
    pub const fn bytes_per_sample(self) -> u64 {
        match self {
            StorageMode::F16Native | StorageMode::F16Packed => 2,
            StorageMode::F32 => 4,
        }
    }

    /// Stable lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            StorageMode::F16Native => "f16-native",
            StorageMode::F16Packed => "f16-packed",
            StorageMode::F32 => "f32",
        }
    }

    /// Whether samples are stored as 16-bit floats (so storage rounds values to f16).
    pub const fn is_f16(self) -> bool {
        !matches!(self, StorageMode::F32)
    }
}

/// Which storage mode to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StoragePreference {
    /// The best the device supports: native f16, else packed f16, else f32.
    #[default]
    Auto,
    /// A specific mode; start-up fails if the device cannot do it.
    Force(StorageMode),
}

/// Settings for [`Gpu::start`].
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// Backends to consider. `None` reads `WGPU_BACKEND` (all backends when unset).
    pub backends: Option<wgpu::Backends>,
    /// Power preference for the first adapter request.
    pub power_preference: wgpu::PowerPreference,
    /// Storage mode preference.
    pub storage: StoragePreference,
    /// GPU memory budget in bytes. `None` uses [`MemoryBudget::DEFAULT_BYTES`] (4 GiB) scaled down
    /// to the adapter's limits.
    pub memory_budget_bytes: Option<u64>,
    /// Folder for the wgpu pipeline cache (`None`: no cache).
    pub pipeline_cache_dir: Option<PathBuf>,
    /// Debug label for the device.
    pub label: String,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            storage: StoragePreference::Auto,
            memory_budget_bytes: None,
            pipeline_cache_dir: None,
            label: "yellowmoon".to_string(),
        }
    }
}

/// `~/Library/Caches/app.yellowmoon/` on macOS (the only place the pipeline cache may live,
/// `docs/01` section Persistence), `$XDG_CACHE_HOME/app.yellowmoon` or `~/.cache/app.yellowmoon`
/// elsewhere. `None` when no home folder is known.
pub fn default_pipeline_cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    if cfg!(target_os = "macos") {
        Some(home.join("Library/Caches/app.yellowmoon"))
    } else if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        Some(PathBuf::from(x).join("app.yellowmoon"))
    } else {
        Some(home.join(".cache/app.yellowmoon"))
    }
}

/// The limits of the opened device that matter to the kernels, for `ym doctor` and tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuLimits {
    /// Largest buffer in bytes.
    pub max_buffer_size: u64,
    /// Largest storage buffer binding in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Largest 2D texture side.
    pub max_texture_dimension_2d: u32,
    /// Storage buffers one shader stage may bind.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Workgroups per dispatch dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Bytes of workgroup memory.
    pub max_compute_workgroup_storage_size: u32,
}

/// What `ym doctor` prints about the GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    /// Adapter name.
    pub name: String,
    /// Backend (`Vulkan`, `Metal`, ...).
    pub backend: String,
    /// Device type (`DiscreteGpu`, `IntegratedGpu`, `Cpu`, ...).
    pub device_type: String,
    /// Driver name.
    pub driver: String,
    /// Driver version text.
    pub driver_info: String,
    /// PCI vendor id.
    pub vendor: u32,
    /// PCI device id.
    pub device: u32,
    /// Whether this is a software rasteriser (Mesa lavapipe, WARP, ...): speeds are meaningless.
    pub is_software: bool,
    /// `SHADER_F16` is available (native `f16` in shaders).
    pub shader_f16: bool,
    /// `pack2x16float` is available (f16 packing in f32 registers).
    pub f16_packing: bool,
    /// The wgpu pipeline cache is available and in use.
    pub pipeline_cache: bool,
    /// The storage mode the kernels use.
    pub storage_mode: StorageMode,
    /// The memory budget in bytes.
    pub memory_budget_bytes: u64,
    /// Device limits.
    pub limits: GpuLimits,
}

impl fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GPU: {} ({}, {}){}",
            self.name,
            self.backend,
            self.device_type,
            if self.is_software { " [software]" } else { "" }
        )?;
        writeln!(f, "Driver: {} {}", self.driver, self.driver_info)?;
        writeln!(
            f,
            "f16: native {}, packing {}; storage {}",
            self.shader_f16,
            self.f16_packing,
            self.storage_mode.name()
        )?;
        writeln!(
            f,
            "Memory budget: {:.2} GiB; pipeline cache: {}",
            self.memory_budget_bytes as f64 / f64::from(1u32 << 30),
            if self.pipeline_cache { "on" } else { "off" }
        )?;
        write!(
            f,
            "Limits: buffer {} MiB, storage binding {} MiB, texture {} px, {} storage buffers, {} invocations",
            self.limits.max_buffer_size >> 20,
            self.limits.max_storage_buffer_binding_size >> 20,
            self.limits.max_texture_dimension_2d,
            self.limits.max_storage_buffers_per_shader_stage,
            self.limits.max_compute_invocations_per_workgroup
        )
    }
}

/// The pipeline cache of one device and the file it is persisted to.
struct PipelineCacheState {
    cache: wgpu::PipelineCache,
    path: PathBuf,
}

/// Everything the device thread owns. Created on that thread by [`GpuContext::open`].
pub struct GpuContext {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu queue.
    pub queue: wgpu::Queue,
    info: GpuInfo,
    pool: BufferPool,
    kernels: Kernels,
    cache: Option<PipelineCacheState>,
}

impl fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuContext")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

fn limits_of(l: &wgpu::Limits) -> GpuLimits {
    GpuLimits {
        max_buffer_size: l.max_buffer_size,
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
        max_texture_dimension_2d: l.max_texture_dimension_2d,
        max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage,
        max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
        max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
        max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
    }
}

fn describe_adapter(a: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?}, {:?})", a.name, a.backend, a.device_type)
}

/// Picks an adapter: high-performance first, then the fallback adapter, then anything.
async fn pick_adapter(
    instance: &wgpu::Instance,
    config: &GpuConfig,
    backends: wgpu::Backends,
) -> Result<wgpu::Adapter, GpuError> {
    let mut tried = Vec::new();
    for force_fallback_adapter in [false, true] {
        let options = wgpu::RequestAdapterOptions {
            power_preference: config.power_preference,
            force_fallback_adapter,
            compatible_surface: None,
            ..Default::default()
        };
        match instance.request_adapter(&options).await {
            Ok(a) => return Ok(a),
            Err(e) => tried.push(format!("request(fallback={force_fallback_adapter}): {e}")),
        }
    }
    let all = instance.enumerate_adapters(backends).await;
    tried.push(format!("{} adapters enumerated", all.len()));
    all.into_iter()
        .next()
        .ok_or_else(|| GpuError::NoAdapter(tried.join("; ")))
}

impl GpuContext {
    /// Opens the best adapter and creates the device, queue, pool and (lazy) kernels.
    ///
    /// Must be called on the thread that will own the context ([`Gpu::start`] does so).
    ///
    /// # Errors
    ///
    /// [`GpuError::NoAdapter`] when nothing opens, [`GpuError::RequestDevice`] when the adapter
    /// rejects the request or a forced [`StorageMode`] is not available.
    pub fn open(config: &GpuConfig) -> Result<GpuContext, GpuError> {
        let descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        let descriptor = match config.backends {
            Some(b) => wgpu::InstanceDescriptor {
                backends: b,
                ..descriptor
            },
            None => descriptor.with_env(),
        };
        let backends = descriptor.backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(pick_adapter(&instance, config, backends))?;
        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();
        let downlevel = adapter.get_downlevel_capabilities();
        let shader_f16 = adapter_features.contains(wgpu::Features::SHADER_F16);
        let f16_packing = downlevel
            .flags
            .contains(wgpu::DownlevelFlags::SHADER_F16_IN_F32);

        let storage_mode = match config.storage {
            StoragePreference::Auto => {
                if shader_f16 {
                    StorageMode::F16Native
                } else if f16_packing {
                    StorageMode::F16Packed
                } else {
                    StorageMode::F32
                }
            }
            StoragePreference::Force(mode) => {
                let ok = match mode {
                    StorageMode::F16Native => shader_f16,
                    StorageMode::F16Packed => f16_packing,
                    StorageMode::F32 => true,
                };
                if !ok {
                    return Err(GpuError::RequestDevice(format!(
                        "storage mode {} is not available on {}",
                        mode.name(),
                        describe_adapter(&adapter_info)
                    )));
                }
                mode
            }
        };

        let mut required_features = wgpu::Features::empty();
        if storage_mode == StorageMode::F16Native {
            required_features |= wgpu::Features::SHADER_F16;
        }
        let want_cache = config.pipeline_cache_dir.is_some()
            && adapter_features.contains(wgpu::Features::PIPELINE_CACHE)
            && wgpu::util::pipeline_cache_key(&adapter_info).is_some();
        if want_cache {
            required_features |= wgpu::Features::PIPELINE_CACHE;
        }

        // Ask for what the adapter has in the three limits that matter for 50 MP tiles, and keep
        // everything else at the (downlevel-safe) defaults.
        let defaults = wgpu::Limits::default();
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size.min(1 << 32),
            max_storage_buffer_binding_size: adapter_limits
                .max_storage_buffer_binding_size
                .min(1 << 31)
                .max(defaults.max_storage_buffer_binding_size),
            ..defaults
        };
        let required_limits = clamp_limits_to(&required_limits, &adapter_limits);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(&config.label),
            required_features,
            required_limits,
            ..Default::default()
        }))
        .map_err(|e| GpuError::RequestDevice(e.to_string()))?;

        let limits = device.limits();
        let budget = MemoryBudget::for_limits(config.memory_budget_bytes, &limits);
        let pool = BufferPool::new(
            device.clone(),
            budget.bytes,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        );

        let cache = if want_cache {
            config
                .pipeline_cache_dir
                .as_deref()
                .and_then(|dir| open_pipeline_cache(&device, &adapter_info, dir))
        } else {
            None
        };

        let info = GpuInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            driver: adapter_info.driver.clone(),
            driver_info: adapter_info.driver_info.clone(),
            vendor: adapter_info.vendor,
            device: adapter_info.device,
            is_software: adapter_info.device_type == wgpu::DeviceType::Cpu,
            shader_f16,
            f16_packing,
            pipeline_cache: cache.is_some(),
            storage_mode,
            memory_budget_bytes: budget.bytes,
            limits: limits_of(&limits),
        };
        tracing::info!(adapter = %info.name, backend = %info.backend, storage = info.storage_mode.name(), "GPU device opened");
        let kernels = Kernels::new(storage_mode);
        Ok(GpuContext {
            device,
            queue,
            info,
            pool,
            kernels,
            cache,
        })
    }

    /// What was opened.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// The storage mode in use.
    pub fn storage_mode(&self) -> StorageMode {
        self.info.storage_mode
    }

    /// The buffer pool with its byte accounting.
    pub fn pool(&self) -> &BufferPool {
        &self.pool
    }

    /// The kernels (pipelines are compiled on first use).
    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// Compiles every kernel now instead of on first use. A shader as large as the colour chain
    /// takes a while to compile on a cold start (on Metal there is no pipeline cache to help), which
    /// would land on the first viewport frame and break the 16 ms budget of `docs/05`; call this
    /// from a job at start-up, then [`save_pipeline_cache`](Self::save_pipeline_cache).
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] for the first kernel that does not compile.
    pub fn warm_up(&self) -> Result<(), GpuError> {
        for id in crate::kernels::KernelId::ALL {
            self.kernels.pipeline(self, id)?;
        }
        Ok(())
    }

    /// The wgpu pipeline cache, if one is in use.
    pub(crate) fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref().map(|c| &c.cache)
    }

    /// Blocks until all submitted work has finished.
    pub fn wait_idle(&self) {
        // A poll failure means the device is lost; the next operation reports it.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Writes the pipeline cache to its file (atomically). Returns whether anything was written.
    pub fn save_pipeline_cache(&self) -> bool {
        let Some(state) = &self.cache else {
            return false;
        };
        let Some(data) = state.cache.get_data() else {
            return false;
        };
        write_atomically(&state.path, &data).is_ok()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        self.save_pipeline_cache();
    }
}

/// Lowers every limit in `want` that the adapter cannot meet to the adapter's value, so the device
/// request cannot fail on a limit we only asked for opportunistically.
fn clamp_limits_to(want: &wgpu::Limits, have: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: want.max_buffer_size.min(have.max_buffer_size),
        max_storage_buffer_binding_size: want
            .max_storage_buffer_binding_size
            .min(have.max_storage_buffer_binding_size),
        max_compute_workgroups_per_dimension: want
            .max_compute_workgroups_per_dimension
            .min(have.max_compute_workgroups_per_dimension),
        max_storage_buffers_per_shader_stage: want
            .max_storage_buffers_per_shader_stage
            .min(have.max_storage_buffers_per_shader_stage),
        max_compute_invocations_per_workgroup: want
            .max_compute_invocations_per_workgroup
            .min(have.max_compute_invocations_per_workgroup),
        max_compute_workgroup_size_x: want
            .max_compute_workgroup_size_x
            .min(have.max_compute_workgroup_size_x),
        max_compute_workgroup_size_y: want
            .max_compute_workgroup_size_y
            .min(have.max_compute_workgroup_size_y),
        max_compute_workgroup_storage_size: want
            .max_compute_workgroup_storage_size
            .min(have.max_compute_workgroup_storage_size),
        ..want.clone()
    }
}

fn open_pipeline_cache(
    device: &wgpu::Device,
    info: &wgpu::AdapterInfo,
    dir: &Path,
) -> Option<PipelineCacheState> {
    let key = wgpu::util::pipeline_cache_key(info)?;
    let path = dir.join(key);
    let data = std::fs::read(&path).ok();
    // SAFETY: `data` is either absent or a file this function wrote from `PipelineCache::get_data`
    // (same adapter vendor/device, encoded in the file name); wgpu additionally validates the
    // header, and `fallback: true` makes an invalid or stale file start an empty cache instead.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("yellowmoon pipeline cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    Some(PipelineCacheState { cache, path })
}

fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Unique per process and per call, so two devices (or threads) saving at once cannot write
    // through the same temporary file.
    let tmp = path.with_extension(format!(
        "tmp{}-{}",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    let result = std::fs::write(&tmp, data).and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err() {
        // Do not leave a partial temporary file behind in the cache folder.
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// A handle to the device thread. Cheap to share (`Arc` it); dropping it shuts the thread down.
pub struct Gpu {
    scheduler: Scheduler<GpuContext>,
    info: GpuInfo,
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu").field("info", &self.info).finish()
    }
}

impl Gpu {
    /// Starts the device thread and opens the device on it.
    ///
    /// # Errors
    ///
    /// Whatever [`GpuContext::open`] reports.
    pub fn start(config: GpuConfig) -> Result<Gpu, GpuError> {
        let (scheduler, info) = Scheduler::spawn("yellowmoon-gpu", move || {
            let ctx = GpuContext::open(&config)?;
            let info = ctx.info().clone();
            Ok::<_, GpuError>((ctx, info))
        })
        .map_err(|e| match e {
            SpawnError::Init(e) => e,
            SpawnError::Thread(io) => {
                GpuError::RequestDevice(format!("the GPU thread could not start: {io}"))
            }
        })?;
        Ok(Gpu { scheduler, info })
    }

    /// Adapter, features, limits and storage mode.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// The scheduler in front of the device thread (cancellation, shutdown, queue length).
    pub fn scheduler(&self) -> &Scheduler<GpuContext> {
        &self.scheduler
    }

    /// Queues `job` for the device thread. Higher [`crate::queue::Priority`] runs first; equal
    /// priorities run in submission order.
    pub fn submit<R, F>(&self, spec: JobSpec, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &crate::queue::CancelToken) -> R + Send + 'static,
    {
        self.scheduler.submit(spec, job)
    }

    /// Runs `job` on the device thread at `spec` and waits for the result.
    ///
    /// # Errors
    ///
    /// The job's own error, or [`GpuError::Cancelled`], [`GpuError::Shutdown`] or
    /// [`GpuError::Panicked`].
    pub fn run<R, F>(&self, spec: JobSpec, job: F) -> Result<R, GpuError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &crate::queue::CancelToken) -> Result<R, GpuError>
            + Send
            + 'static,
    {
        self.submit(spec, job).wait()?
    }

    /// Stops the thread after the running job; pending jobs resolve with [`GpuError::Shutdown`].
    pub fn shutdown(&self) {
        self.scheduler.shutdown();
    }
}

/// Whether the environment demands a GPU (`YM_REQUIRE_GPU=1`, the `gpu-lavapipe` CI job).
pub fn gpu_required_by_env() -> bool {
    std::env::var("YM_REQUIRE_GPU").is_ok_and(|v| !v.is_empty() && v != "0")
}
