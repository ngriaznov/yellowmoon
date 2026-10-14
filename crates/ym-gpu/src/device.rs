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
//!    accepted, which is what CI uses, and is reported as such in [`GpuInfo`]. Whether a software
//!    adapter is *used* is the integrator's policy, not this module's.
//!
//! # Storage mode
//!
//! Image planes that cross a kernel boundary are **f32 in memory** in every storage mode
//! (`docs/gpu-plan.md` X-3: the f16 errors measured 7.9e-4 to 6e-3 against the CPU reference, and the
//! raw-algorithm kernels already require f32). [`StoragePreference::Auto`] therefore resolves to
//! [`StorageMode::F32`] on every device. The two f16 modes remain for the legacy kernels and their
//! tests: [`StoragePreference::Force`] with [`StorageMode::F16Native`] (`array<f16>`, needs
//! `SHADER_F16`) or [`StorageMode::F16Packed`] (`array<u32>` holding two halves, packed with
//! `pack2x16float`). The choice is made once at start-up and written into every generated shader.
//!
//! # Health: lost and tainted devices
//!
//! An uncaptured wgpu error (a validation error nobody wrapped in an error scope, an out-of-memory
//! from the driver) taints the device, and a device-lost callback marks it lost. Both are recorded in
//! a [`DeviceHealth`] shared with the pool and the handle: [`Gpu::is_lost`] reports either, the next
//! allocation, submission or readback of a job fails with [`GpuError::DeviceLost`], and so does a
//! readback that takes longer than [`GpuConfig::readback_timeout`] (30 s). Rebuilding the device (at
//! most twice, then the GPU is off for the process) is the integrator's job
//! (`docs/gpu-plan.md` 4.9).
//!
//! # Pipeline cache
//!
//! `GpuConfig::pipeline_cache_dir` is the hook for the wgpu pipeline cache (the only thing that may
//! live under `~/Library/Caches/app.yellowmoon/`, `docs/01` section Persistence; the integrator
//! passes [`default_pipeline_cache_dir`]). When the adapter supports `PIPELINE_CACHE` (Vulkan today)
//! the cache is loaded at start and written back, atomically, by [`GpuContext::save_pipeline_cache`].
//! No image data is ever written.
//!
//! # Poison
//!
//! With [`GpuConfig::poison`] (or `YM_GPU_POISON=1`) every buffer the pool hands out is first filled
//! with NaN, so a kernel that reads what it did not write (or what an earlier job left in a recycled
//! buffer) produces NaN instead of a plausible number. The test helpers switch it on for every test
//! device; see [`crate::pool`].
//!
//! # Sharing the device with burn (feature `ml`)
//!
//! [`GpuContext`] keeps the `Instance` and the `Adapter` it opened, and [`Gpu::wgpu_setup`] hands out
//! clones with the device and the queue, the shape burn's wgpu backend imports. With the cargo
//! feature `ml` the device is requested the way cubecl requests one (every adapter feature except
//! `MAPPABLE_PRIMARY_BUFFERS` and the experimental ones, the adapter's limits) so that burn can use
//! it; without the feature nothing changes (`docs/gpu-plan.md` section 8).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crate::kernels::{KernelSet, Kernels};
use crate::pool::{BufferPool, MemoryBudget, PoolOptions};
use crate::queue::{CancelToken, Cancelled, JobError, JobHandle, JobSpec, Scheduler, SpawnError};
use crate::stream::{CounterSnapshot, Counters};

mod fault;

pub(crate) use fault::FaultState;
#[cfg(feature = "fault-injection")]
pub use fault::GpuFault;

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
    /// The device was lost or can no longer be trusted: it was destroyed, the driver reset it, wgpu
    /// reported an error nobody had captured, or a readback did not finish in time. Every later job
    /// on this device fails the same way; the integrator rebuilds the device or falls back to the CPU.
    #[error("the GPU device was lost: {0}")]
    DeviceLost(String),
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

/// How image data is held in GPU memory. [`StorageMode::F32`] is what the device uses unless a test
/// forces an f16 mode.
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
    /// f32 on every device ([`StorageMode::F32`]): image planes are f32 in memory (`docs/gpu-plan.md`
    /// X-3). The f16 modes are reached only through [`StoragePreference::Force`].
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
    /// Fill every buffer the pool hands out with NaN before the caller sees it (see the module
    /// notes). Slow; for tests and for hunting a stale read. The default is `true` when the
    /// environment variable `YM_GPU_POISON` is set to anything but `0` or empty, else `false`.
    pub poison: bool,
    /// Request `TIMESTAMP_QUERY` when the adapter has it, so that [`PassList`](crate::stream::PassList)s
    /// can report the GPU time of each pass ([`PassTimings`](crate::stream::PassTimings)). Off by
    /// default: nothing is timed in a normal run.
    pub timestamps: bool,
    /// The longest a readback waits for the GPU. When it expires the device is tainted and the
    /// readback fails with [`GpuError::DeviceLost`] (`docs/gpu-plan.md` 4.9). 30 s by default.
    pub readback_timeout: Duration,
    /// A fault to arm when the device opens (feature `fault-injection`; `YM_GPU_FAULT` is read when
    /// this is `None`). [`Gpu::set_fault`] arms one later.
    #[cfg(feature = "fault-injection")]
    pub fault: Option<GpuFault>,
}

/// How long a readback waits for the GPU by default.
pub const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Whether `YM_GPU_POISON` asks for poison mode.
pub fn poison_requested_by_env() -> bool {
    std::env::var("YM_GPU_POISON").is_ok_and(|v| !v.is_empty() && v != "0")
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
            poison: poison_requested_by_env(),
            timestamps: false,
            readback_timeout: READBACK_TIMEOUT,
            #[cfg(feature = "fault-injection")]
            fault: None,
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
    /// The GPU shares memory with the CPU (`DeviceType::IntegratedGpu`: Metal on Apple silicon sets
    /// it exactly when the machine has unified memory). The memory budget is then carved out of the
    /// RAM the CPU side also uses ([`MemoryBudget::for_unified_memory`]).
    pub unified_memory: bool,
    /// Timestamps were asked for ([`GpuConfig::timestamps`]) and the device has `TIMESTAMP_QUERY`, so
    /// pass lists report GPU times.
    pub timestamp_query: bool,
    /// Poison mode is on: every buffer is filled with NaN when handed out.
    pub poison: bool,
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
            "Memory budget: {:.2} GiB{}; pipeline cache: {}; timestamps: {}",
            self.memory_budget_bytes as f64 / f64::from(1u32 << 30),
            if self.unified_memory {
                " (unified memory)"
            } else {
                ""
            },
            if self.pipeline_cache { "on" } else { "off" },
            if self.timestamp_query { "on" } else { "off" }
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

/// What has gone wrong with a device, shared by the device thread, the buffer pool and the [`Gpu`]
/// handle (see the module notes). Cheap to read; the first reason recorded is the one kept.
#[derive(Debug, Default)]
pub struct DeviceHealth {
    lost: AtomicBool,
    tainted: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl DeviceHealth {
    /// The device was lost: destroyed, reset by the driver, or lost by an injected fault.
    pub fn lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// The device can no longer be trusted although it is not lost: wgpu reported an error that no
    /// error scope captured, or a readback timed out.
    pub fn tainted(&self) -> bool {
        self.tainted.load(Ordering::Acquire)
    }

    /// Lost or tainted: results from this device must not be used any more.
    pub fn is_unusable(&self) -> bool {
        self.lost() || self.tainted()
    }

    /// Why the device is unusable (the first reason recorded), if it is.
    pub fn reason(&self) -> Option<String> {
        self.reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn record(&self, why: String) {
        let mut slot = self.reason.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(why);
        }
    }

    /// Marks the device lost.
    pub(crate) fn mark_lost(&self, why: String) {
        self.record(why);
        self.lost.store(true, Ordering::Release);
    }

    /// Marks the device tainted.
    pub(crate) fn mark_tainted(&self, why: String) {
        self.record(why);
        self.tainted.store(true, Ordering::Release);
    }

    /// `Err(DeviceLost)` once the device is lost or tainted.
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] with the first recorded reason.
    pub fn check(&self) -> Result<(), GpuError> {
        if self.is_unusable() {
            Err(GpuError::DeviceLost(
                self.reason()
                    .unwrap_or_else(|| "no reason recorded".to_string()),
            ))
        } else {
            Ok(())
        }
    }
}

/// The wgpu objects of the opened device, the shape burn's wgpu backend imports (`WgpuSetup` of
/// `cubecl-wgpu`: instance, adapter, device, queue, backend). All of them are cheap clones of the
/// handles the device thread owns; see [`Gpu::wgpu_setup`].
#[derive(Clone, Debug)]
pub struct WgpuSetup {
    /// The instance the adapter came from.
    pub instance: wgpu::Instance,
    /// The adapter the device was opened on.
    pub adapter: wgpu::Adapter,
    /// The device.
    pub device: wgpu::Device,
    /// The queue.
    pub queue: wgpu::Queue,
    /// The backend of the adapter.
    pub backend: wgpu::Backend,
}

/// What the [`Gpu`] handle shares with the context on the device thread.
#[derive(Clone)]
struct Shared {
    health: Arc<DeviceHealth>,
    counters: Arc<Counters>,
    #[cfg_attr(not(feature = "fault-injection"), allow(dead_code))]
    fault: Arc<FaultState>,
    setup: WgpuSetup,
}

/// Everything the device thread owns. Created on that thread by [`GpuContext::open`].
pub struct GpuContext {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu queue.
    pub queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    info: GpuInfo,
    pool: BufferPool,
    kernels: Kernels,
    cache: Option<PipelineCacheState>,
    health: Arc<DeviceHealth>,
    counters: Arc<Counters>,
    fault: Arc<FaultState>,
    readback_timeout: Duration,
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
            // f32 on every device (module notes, `docs/gpu-plan.md` X-3).
            StoragePreference::Auto => StorageMode::F32,
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
        // Timestamps are optional and only requested when asked for (K6 of `docs/gpu-plan.md`).
        let timestamp_query =
            config.timestamps && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamp_query {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
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
        let (required_features, required_limits) =
            ml_request(required_features, required_limits, &adapter);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(&config.label),
            required_features,
            required_limits,
            ..Default::default()
        }))
        .map_err(|e| GpuError::RequestDevice(e.to_string()))?;

        // Errors nobody captured taint the device instead of panicking the device thread; a lost
        // device is recorded the same way. Both are read by the pool, the readbacks and `Gpu`.
        let health = Arc::new(DeviceHealth::default());
        {
            let h = Arc::clone(&health);
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let why = format!("wgpu reported an uncaptured error: {e}");
                tracing::error!("{why}");
                h.mark_tainted(why);
            }));
            let h = Arc::clone(&health);
            device.set_device_lost_callback(move |reason, message| {
                let why = format!("device lost ({reason:?}): {message}");
                tracing::warn!("{why}");
                h.mark_lost(why);
            });
        }
        let fault = Arc::new(FaultState::default());
        #[cfg(feature = "fault-injection")]
        match config.fault {
            Some(f) => fault.arm(Some(f)),
            None => fault.arm_from_env(),
        }
        let counters = Arc::new(Counters::default());

        let limits = device.limits();
        let budget = MemoryBudget::for_limits(config.memory_budget_bytes, &limits);
        let pool = BufferPool::with_options(
            device.clone(),
            budget.bytes,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
            PoolOptions {
                health: Some(Arc::clone(&health)),
                fault: Some(Arc::clone(&fault)),
                poison: config.poison.then(|| queue.clone()),
            },
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
            unified_memory: adapter_info.device_type == wgpu::DeviceType::IntegratedGpu,
            // Recorded only when asked for, also on a device that has the feature for another reason
            // (the `ml` request takes every adapter feature).
            timestamp_query: config.timestamps
                && device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            poison: config.poison,
            storage_mode,
            memory_budget_bytes: budget.bytes,
            limits: limits_of(&limits),
        };
        tracing::info!(adapter = %info.name, backend = %info.backend, storage = info.storage_mode.name(), "GPU device opened");
        let kernels = Kernels::new(storage_mode);
        Ok(GpuContext {
            device,
            queue,
            instance,
            adapter,
            info,
            pool,
            kernels,
            cache,
            health,
            counters,
            fault,
            readback_timeout: config.readback_timeout,
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

    /// The instance the device was opened from (kept for sharing the device, module notes).
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// The adapter the device was opened on.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Lost or tainted state of this device.
    pub fn health(&self) -> &DeviceHealth {
        &self.health
    }

    /// The shared health flags, for values (a pending submission) that outlive a borrow of the
    /// context.
    pub(crate) fn health_arc(&self) -> Arc<DeviceHealth> {
        Arc::clone(&self.health)
    }

    /// Delivers pending device callbacks and fails when the device is lost or tainted. Jobs run
    /// through [`Gpu::run`] start with this check; a kernel that allocates, submits or reads back
    /// through the pool and the stream helpers checks the flags again.
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] with the recorded reason.
    pub fn check_alive(&self) -> Result<(), GpuError> {
        // A destroyed device only reports itself lost at the next poll.
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.health.check()
    }

    /// Submits command buffers: checks the device first and counts the submission
    /// ([`GpuContext::counters`]).
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] when the device is lost or tainted.
    pub fn submit<I>(&self, commands: I) -> Result<wgpu::SubmissionIndex, GpuError>
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        self.health.check()?;
        self.counters.count_submission();
        Ok(self.queue.submit(commands))
    }

    /// A snapshot of the upload, readback, submission and dispatch counters. Subtract two
    /// snapshots ([`CounterSnapshot::since`]) to count what a region of code did.
    pub fn counters(&self) -> CounterSnapshot {
        self.counters.snapshot()
    }

    /// The shared counters, for the helpers that count (`crate::stream`).
    pub(crate) fn counters_shared(&self) -> &Arc<Counters> {
        &self.counters
    }

    /// How long a readback waits for the GPU ([`GpuConfig::readback_timeout`]).
    pub fn readback_timeout(&self) -> Duration {
        self.readback_timeout
    }

    /// Whether GPU pass timestamps are on ([`GpuConfig::timestamps`] and the adapter has them).
    pub fn timestamps_enabled(&self) -> bool {
        self.info.timestamp_query
    }

    /// Whether poison mode is on ([`GpuConfig::poison`]).
    pub fn poison(&self) -> bool {
        self.info.poison
    }

    /// The hook the chains call once per tile: with the `fault-injection` feature and a `tile@N`
    /// fault armed, the `N`th call fails; otherwise it does nothing.
    ///
    /// # Errors
    ///
    /// [`GpuError::Readback`] for the injected fault.
    pub fn inject_tile_fault(&self) -> Result<(), GpuError> {
        self.fault.on_tile()
    }

    /// The fault state, for the kernels' pipeline hook.
    pub(crate) fn fault(&self) -> &FaultState {
        &self.fault
    }

    /// Compiles every real kernel now instead of on first use. A shader as large as the colour chain
    /// takes a while to compile on a cold start (on Metal there is no pipeline cache to help), which
    /// would land on the first viewport frame and break the 16 ms budget of `docs/05`; call this
    /// from a job at start-up, then [`save_pipeline_cache`](Self::save_pipeline_cache). Same as
    /// `warm_up_set(KernelSet::All)`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] for the first kernel that does not compile.
    pub fn warm_up(&self) -> Result<(), GpuError> {
        self.warm_up_set(KernelSet::All)
    }

    /// Compiles the kernels of `set` now (the app warms [`KernelSet::View`] first, `ym` the sets
    /// it needs while photos decode). Placeholder kernels, registered before their implementation
    /// exists, are skipped.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] for the first kernel that does not compile.
    pub fn warm_up_set(&self, set: KernelSet) -> Result<(), GpuError> {
        for id in set.kernels().filter(|id| !id.is_placeholder()) {
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
        if self.health.is_unusable() {
            return false;
        }
        let Some(data) = state.cache.get_data() else {
            return false;
        };
        write_atomically(&state.path, &data).is_ok()
    }

    /// The pieces the [`Gpu`] handle shares.
    fn shared(&self) -> Shared {
        Shared {
            health: Arc::clone(&self.health),
            counters: Arc::clone(&self.counters),
            fault: Arc::clone(&self.fault),
            setup: WgpuSetup {
                instance: self.instance.clone(),
                adapter: self.adapter.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                backend: self.adapter.get_info().backend,
            },
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        self.save_pipeline_cache();
    }
}

/// With the `ml` feature the device is requested the way cubecl requests one, so burn can import
/// it: every adapter feature except `MAPPABLE_PRIMARY_BUFFERS` and the experimental ones (a device
/// cannot hold those without the `unsafe` opt-in), and exactly the adapter's limits
/// (`docs/gpu-plan.md` section 8).
#[cfg(feature = "ml")]
fn ml_request(
    _features: wgpu::Features,
    _limits: wgpu::Limits,
    adapter: &wgpu::Adapter,
) -> (wgpu::Features, wgpu::Limits) {
    (
        adapter.features()
            - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
            - wgpu::Features::all_experimental_mask(),
        adapter.limits(),
    )
}

/// Without the `ml` feature the request is the plain one.
#[cfg(not(feature = "ml"))]
fn ml_request(
    features: wgpu::Features,
    limits: wgpu::Limits,
    _adapter: &wgpu::Adapter,
) -> (wgpu::Features, wgpu::Limits) {
    (features, limits)
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
    shared: Shared,
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu").field("info", &self.info).finish()
    }
}

/// Wraps a job so that, on the device thread, the armed fault (feature `fault-injection`) gets its
/// chance before the job body runs. A normal build adds nothing.
fn wrapped<R, F>(job: F) -> impl FnOnce(&mut GpuContext, &CancelToken) -> R + Send + 'static
where
    F: FnOnce(&mut GpuContext, &CancelToken) -> R + Send + 'static,
{
    move |ctx, token| {
        ctx.fault.on_job(&ctx.health, &ctx.device, token);
        job(ctx, token)
    }
}

/// Wraps a fallible job so that it starts with [`GpuContext::check_alive`].
fn checked<R, F>(job: F) -> impl FnOnce(&mut GpuContext, &CancelToken) -> Result<R, GpuError>
where
    F: FnOnce(&mut GpuContext, &CancelToken) -> Result<R, GpuError>,
{
    move |ctx, token| {
        ctx.check_alive()?;
        job(ctx, token)
    }
}

impl Gpu {
    /// Starts the device thread and opens the device on it.
    ///
    /// # Errors
    ///
    /// Whatever [`GpuContext::open`] reports.
    pub fn start(config: GpuConfig) -> Result<Gpu, GpuError> {
        let (scheduler, (info, shared)) = Scheduler::spawn("yellowmoon-gpu", move || {
            let ctx = GpuContext::open(&config)?;
            let info = ctx.info().clone();
            let shared = ctx.shared();
            Ok::<_, GpuError>((ctx, (info, shared)))
        })
        .map_err(|e| match e {
            SpawnError::Init(e) => e,
            SpawnError::Thread(io) => {
                GpuError::RequestDevice(format!("the GPU thread could not start: {io}"))
            }
        })?;
        Ok(Gpu {
            scheduler,
            info,
            shared,
        })
    }

    /// Adapter, features, limits and storage mode.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// The scheduler in front of the device thread (cancellation, shutdown, queue length).
    pub fn scheduler(&self) -> &Scheduler<GpuContext> {
        &self.scheduler
    }

    /// Whether the device is lost or tainted (see the module notes): destroyed, reset by the driver,
    /// an uncaptured wgpu error, or a readback that timed out. Delivers pending device callbacks
    /// first, so a device destroyed a moment ago reports itself.
    pub fn is_lost(&self) -> bool {
        // `Device::poll` is callable from any thread; it only delivers callbacks here.
        let _ = self.shared.setup.device.poll(wgpu::PollType::Poll);
        self.shared.health.is_unusable()
    }

    /// Why the device is lost or tainted, if it is.
    pub fn lost_reason(&self) -> Option<String> {
        self.shared.health.reason()
    }

    /// A snapshot of the upload, readback, submission and dispatch counters
    /// ([`GpuContext::counters`]).
    pub fn counters(&self) -> CounterSnapshot {
        self.shared.counters.snapshot()
    }

    /// Clones of the instance, adapter, device and queue, in the shape burn's wgpu backend imports
    /// (feature `ml` makes the device a burn-compatible one; see the module notes).
    pub fn wgpu_setup(&self) -> WgpuSetup {
        self.shared.setup.clone()
    }

    /// Arms a fault (or clears the armed one) for the fallback tests; see [`GpuFault`].
    #[cfg(feature = "fault-injection")]
    pub fn set_fault(&self, fault: Option<GpuFault>) {
        self.shared.fault.arm(fault);
    }

    /// Queues `job` for the device thread. Higher [`crate::queue::Priority`] runs first; equal
    /// priorities run in submission order.
    pub fn submit<R, F>(&self, spec: JobSpec, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &CancelToken) -> R + Send + 'static,
    {
        self.scheduler.submit(spec, wrapped(job))
    }

    /// Like [`Gpu::submit`] with a cancel token the caller already owns (a render's own), shared by
    /// every job of the render: setting it skips the jobs that have not started and is seen by the
    /// one that runs ([`Scheduler::submit_with_token`]).
    pub fn submit_with_token<R, F>(&self, spec: JobSpec, token: CancelToken, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &CancelToken) -> R + Send + 'static,
    {
        self.scheduler.submit_with_token(spec, token, wrapped(job))
    }

    /// Runs `job` on the device thread at `spec` and waits for the result. The job starts with
    /// [`GpuContext::check_alive`], so a lost device reports [`GpuError::DeviceLost`] at once.
    ///
    /// # Errors
    ///
    /// The job's own error, or [`GpuError::Cancelled`], [`GpuError::Shutdown`],
    /// [`GpuError::DeviceLost`] or [`GpuError::Panicked`].
    pub fn run<R, F>(&self, spec: JobSpec, job: F) -> Result<R, GpuError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &CancelToken) -> Result<R, GpuError> + Send + 'static,
    {
        self.submit(spec, checked(job)).wait()?
    }

    /// [`Gpu::run`] with a cancel token the caller owns ([`Gpu::submit_with_token`]).
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`]; [`GpuError::Cancelled`] when the token was set before the job started.
    pub fn run_with_token<R, F>(
        &self,
        spec: JobSpec,
        token: CancelToken,
        job: F,
    ) -> Result<R, GpuError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GpuContext, &CancelToken) -> Result<R, GpuError> + Send + 'static,
    {
        self.submit_with_token(spec, token, checked(job)).wait()?
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
