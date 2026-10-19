//! The one place that opens the graphics device: the policy, the stages it may run, and the slot
//! the application and `ym` both get back (`docs/17-gpu.md` section 1).
//!
//! * [`GpuPolicy`] says what a person or a developer asked for: `Auto` (the default: a hardware
//!   adapter, and only the stages signed off, [`GpuStages::SIGNED_OFF`]), `On` (any adapter,
//!   software ones too, every stage; CI and development), `Off` (the CPU reference; `ym render
//!   --cpu`) and `Require` (like `On`, strict: no silent fallback, `ym render --require-gpu`).
//!   Precedence is **flag, then the environment variable `YELLOWMOON_GPU`, then `Auto`**
//!   ([`GpuPolicy::resolve`]).
//! * [`open_gpu`] (or a [`GpuOpener`] for the knobs tests and the application need) turns a policy
//!   into a [`GpuSlot`]: the device when one is to be used, the reason when not, the adapter that was
//!   seen, the stages and the memory budgets that apply. **Nothing else in the application, the
//!   command line or the core creates a compute device** (`tests/gpu_policy.rs` scans the sources
//!   for it). The native viewport of the Mac app draws with this device (`docs/17-gpu.md` section
//!   16) and opens a presentation device of its own, which draws frames and computes nothing, only
//!   when the core has none.
//! * The device is asked for through an [`AdapterProbe`], so the decision (four policies against a
//!   hardware adapter, a software adapter and none) is tested without a graphics card.
//!
//! The renderer consults [`GpuStages`] one stage at a time (`RenderOptions::gpu_stages`).

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use ym_gpu::device::{Gpu, GpuConfig, GpuError, GpuInfo};
use ym_gpu::kernels::KernelId;
use ym_gpu::queue::{JobHandle, JobSpec, Priority};

use crate::pipeline::RenderOptions;
use crate::scheduler::{MemoryBudget, total_ram_bytes};

/// The environment variable that sets the policy when no flag does.
pub const GPU_ENV: &str = "YELLOWMOON_GPU";

// ---------------------------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------------------------

/// The parts of the image pipeline that can run on the device, as a small bit set. Each member is a
/// range that can be switched on and off on its own; a member that is off keeps the host boundary
/// it has today (`docs/17-gpu.md` section 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuStages(u8);

impl GpuStages {
    /// No stage.
    pub const NONE: Self = Self(0);
    /// The raw chain: the adaptive ratio demosaic, the bilateral denoise, R7 and capture sharpening
    /// of every full-resolution tile (R6 to R8). The stages in front of it and after it are
    /// [`GpuStages::RAW_FRONT`].
    pub const RAW_CHAIN: Self = Self(1 << 0);
    /// The geometry sampler (orientation, straighten, crop, resize to the view).
    pub const SAMPLER: Self = Self(1 << 1);
    /// The colour chain, C1 to C12.
    pub const COLOR: Self = Self(1 << 2);
    /// Output sharpening, the last stage before the encode.
    pub const OUTPUT_SHARPEN: Self = Self(1 << 3);
    /// C13: the transfer function, the dither and the quantisation.
    pub const ENCODE: Self = Self(1 << 4);
    /// The export tail: resize and bands for the writers.
    pub const EXPORT_TAIL: Self = Self(1 << 5);
    /// The raw front and the lens: R1 normalise, R3 vignetting, R4 gains, R5 highlights and R5b
    /// lateral CA of every full-resolution tile (with [`GpuStages::RAW_CHAIN`] the whole of
    /// R1 to R9 runs on the device from the photo's resident raw words, uploaded once), R9 lens
    /// resample, and the reduced fit-view group (the noise sigma plane, the R5 chroma offsets, the
    /// `k x k` binning, the denoise and R9 at 1/k) that never builds a full-resolution frame on the
    /// host. Signed off on 2026-10-15 ([`GpuStages::SIGNED_OFF`]).
    pub const RAW_FRONT: Self = Self(1 << 6);
    /// Every stage.
    pub const ALL: Self = Self(0b111_1111);

    /// What `Auto` runs on the device: the stages signed off, that is, the ones whose parity, tile
    /// and determinism suites pass on the Metal CI runners (`.github/workflows/mac-gpu.yml`), the
    /// rule of STATUS question 14. **The one constant to widen** when a stage joins.
    /// Signed off on 2026-10-14 (Mykyta), after every suite passed on macOS 14 and 15 (Metal CI run
    /// 38557455494, commit 6d277b8): the raw chain (R6 to R8), the sampler, the colour chain,
    /// output sharpening and C13. On 2026-10-15, by the same rule, after Metal CI run 38694506557
    /// (commit 4c5939f): the export tail and the raw front, so `Auto` on a hardware
    /// adapter now runs every stage. Before 2026-10-14 only the colour pass was signed off (since
    /// M0-07).
    pub const SIGNED_OFF: Self = Self(
        Self::RAW_CHAIN.0
            | Self::SAMPLER.0
            | Self::COLOR.0
            | Self::OUTPUT_SHARPEN.0
            | Self::ENCODE.0
            | Self::EXPORT_TAIL.0
            | Self::RAW_FRONT.0,
    );

    /// The stages of an export that run on the device in this version: the raw chain (R6 to R8 of
    /// every full-resolution tile: demosaic, denoise, R7 and capture sharpening) and the
    /// raw front (R1 to R5b and R9 of every tile), and the export tail: the sampler, the colour
    /// chain, the Lanczos-3 resize, output sharpening and C13 of the written picture, which the CPU
    /// writers pull as encoded bands. Decides whether `ym` warms the kernels up while photos
    /// decode; **a stage that exports run on the device is added here**.
    pub const USED_BY_EXPORTS: Self =
        Self(Self::RAW_CHAIN.0 | Self::RAW_FRONT.0 | Self::EXPORT_TAIL.0);

    /// Whether every stage of `other` is in this set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the two sets share a stage.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The stages of either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether there is no stage.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The names of the stages in the set, in the order of the members (`raw`, `sampler`,
    /// `colour`, `sharpening`, `encode`, `export`, `raw front`): what `ym doctor` prints.
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        [
            (Self::RAW_CHAIN, "raw"),
            (Self::SAMPLER, "sampler"),
            (Self::COLOR, "colour"),
            (Self::OUTPUT_SHARPEN, "sharpening"),
            (Self::ENCODE, "encode"),
            (Self::EXPORT_TAIL, "export"),
            (Self::RAW_FRONT, "raw front"),
        ]
        .into_iter()
        .filter(|(stage, _)| self.contains(*stage))
        .map(|(_, name)| name)
        .collect()
    }
}

impl std::ops::BitOr for GpuStages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl fmt::Display for GpuStages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&self.names().join(", "))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------------------------

/// What the person (or the developer) asked of the graphics device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GpuPolicy {
    /// The default. A hardware adapter, never a software one (Mesa lavapipe, WARP), and only the
    /// stages in [`GpuStages::SIGNED_OFF`]; otherwise the CPU.
    #[default]
    Auto,
    /// Any adapter, software ones included, every stage; the CPU when none opens. CI and
    /// development.
    On,
    /// Never: the CPU reference (`ym render --cpu`; the golden images are made this way).
    Off,
    /// Like `On`, but strict: no silent fallback. `ym` stops with exit code 4 when no device opens
    /// at the start (`--require-gpu`).
    Require,
}

/// A command-line flag that decides the policy before the environment is looked at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFlag {
    /// `--cpu`: [`GpuPolicy::Off`].
    Cpu,
    /// `--require-gpu`: [`GpuPolicy::Require`].
    RequireGpu,
}

impl PolicyFlag {
    /// The flag as it is typed.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::Cpu => "--cpu",
            Self::RequireGpu => "--require-gpu",
        }
    }
}

/// Where a [`GpuPolicy`] came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicySource {
    /// A command-line flag.
    Flag(PolicyFlag),
    /// The environment variable [`GPU_ENV`].
    Environment,
    /// Nothing was asked: [`GpuPolicy::Auto`].
    Default,
}

/// The policy in force, where it came from and a complaint about a setting that could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyChoice {
    /// The policy.
    pub policy: GpuPolicy,
    /// What chose it.
    pub source: PolicySource,
    /// Set when the environment holds something that is not a policy: the policy is then `Auto`,
    /// and the caller says so (a warning on stderr, a line in the log).
    pub warning: Option<String>,
}

/// The text of [`GPU_ENV`] is not one of the policies.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{GPU_ENV}={value} is not a graphics setting. Use auto, on, off or require.")]
pub struct UnknownPolicy {
    /// What was found.
    pub value: String,
}

impl GpuPolicy {
    /// Every policy.
    pub const ALL: [Self; 4] = [Self::Auto, Self::On, Self::Off, Self::Require];

    /// The word for the policy: `auto`, `on`, `off` or `require`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
            Self::Require => "require",
        }
    }

    /// Reads a policy word, in any case: `auto`, `on`, `off`, `require`, and the `1` and `0` that
    /// the variable accepted before the policies had names (`1` is `on`, `0` is `off`).
    ///
    /// # Errors
    ///
    /// [`UnknownPolicy`] for anything else.
    pub fn parse(text: &str) -> Result<Self, UnknownPolicy> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "1" => Ok(Self::On),
            "off" | "0" => Ok(Self::Off),
            "require" => Ok(Self::Require),
            _ => Err(UnknownPolicy {
                value: text.trim().to_owned(),
            }),
        }
    }

    /// The policy for a run: a flag wins over the environment, the environment over the default.
    /// `env` is the value of [`GPU_ENV`] (`None`, or an empty text, when it is not set). A value
    /// that is not a policy is not an error: the policy is `Auto` and the complaint is returned
    /// for the caller to print, so a typo in a variable never stops a run.
    #[must_use]
    pub fn resolve(flag: Option<PolicyFlag>, env: Option<&str>) -> PolicyChoice {
        if let Some(flag) = flag {
            return PolicyChoice {
                policy: match flag {
                    PolicyFlag::Cpu => Self::Off,
                    PolicyFlag::RequireGpu => Self::Require,
                },
                source: PolicySource::Flag(flag),
                warning: None,
            };
        }
        match env.map(str::trim).filter(|v| !v.is_empty()) {
            None => PolicyChoice {
                policy: Self::Auto,
                source: PolicySource::Default,
                warning: None,
            },
            Some(value) => match Self::parse(value) {
                Ok(policy) => PolicyChoice {
                    policy,
                    source: PolicySource::Environment,
                    warning: None,
                },
                Err(e) => PolicyChoice {
                    policy: Self::Auto,
                    source: PolicySource::Default,
                    warning: Some(e.to_string()),
                },
            },
        }
    }

    /// [`GpuPolicy::resolve`] with the real environment.
    #[must_use]
    pub fn from_environment(flag: Option<PolicyFlag>) -> PolicyChoice {
        let env = match std::env::var(GPU_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            // Not UTF-8: a value nobody can mean as a policy.
            Err(std::env::VarError::NotUnicode(raw)) => Some(raw.to_string_lossy().into_owned()),
        };
        Self::resolve(flag, env.as_deref())
    }

    /// The stages this policy runs on a device it uses.
    #[must_use]
    pub const fn stages(self) -> GpuStages {
        match self {
            Self::Off => GpuStages::NONE,
            Self::Auto => GpuStages::SIGNED_OFF,
            Self::On | Self::Require => GpuStages::ALL,
        }
    }

    /// Whether a software adapter (Mesa lavapipe, WARP) may be used. `Auto` never uses one: the CPU
    /// path is the reference, and faster than a software device's first pass.
    #[must_use]
    pub const fn accepts_software(self) -> bool {
        matches!(self, Self::On | Self::Require)
    }

    /// Whether a missing device is a failure rather than a fallback.
    #[must_use]
    pub const fn is_strict(self) -> bool {
        matches!(self, Self::Require)
    }
}

impl fmt::Display for GpuPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for PolicyChoice {
    /// `auto (default)`, `on (YELLOWMOON_GPU)`, `off (--cpu)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.source {
            PolicySource::Flag(flag) => write!(f, "{} ({})", self.policy, flag.text()),
            PolicySource::Environment => write!(f, "{} ({GPU_ENV})", self.policy),
            PolicySource::Default => write!(f, "{} (default)", self.policy),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------------------------

/// What [`open_gpu`] needs from an opened device: a description of the adapter, and a way to close
/// it again when the policy turns it down.
pub trait GpuDevice: Send + Sync + 'static {
    /// The adapter, its limits and its storage mode.
    fn info(&self) -> &GpuInfo;
    /// Stops the device and waits until it is gone.
    fn close(&self);
}

impl GpuDevice for Gpu {
    fn info(&self) -> &GpuInfo {
        Gpu::info(self)
    }

    fn close(&self) {
        self.shutdown();
    }
}

/// The way a device is asked for. The system's is [`SystemProbe`]; a test passes one that answers
/// "hardware", "software" or "nothing" without a graphics card.
pub trait AdapterProbe {
    /// The device it opens.
    type Device: GpuDevice;

    /// Opens a device with `config`, or says why none opens.
    ///
    /// # Errors
    ///
    /// [`GpuError`] when no adapter opens.
    fn open(&self, config: GpuConfig) -> Result<Self::Device, GpuError>;
}

/// The real graphics stack, through `ym_gpu::device::Gpu::start`: **the only call that creates a
/// compute device in the application, the command line and the core**.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemProbe;

impl AdapterProbe for SystemProbe {
    type Device = Gpu;

    fn open(&self, config: GpuConfig) -> Result<Gpu, GpuError> {
        // A driver that panics is a GPU that cannot be used, not a reason to stop.
        std::panic::catch_unwind(|| Gpu::start(config)).unwrap_or_else(|_| {
            Err(GpuError::NoAdapter(
                "the graphics driver stopped unexpectedly".to_owned(),
            ))
        })
    }
}

/// Looks at the adapter a device would open on (the same choice as [`open_gpu`]), and closes the
/// device again. For `ym doctor`, which reports the adapter whatever the policy says.
///
/// # Errors
///
/// The reason no device opens.
pub fn probe_adapter() -> Result<GpuInfo, GpuError> {
    let device = SystemProbe.open(GpuConfig::default())?;
    let info = device.info().clone();
    device.close();
    Ok(info)
}

/// Whether the adapter's memory is the machine's RAM (`GpuInfo::unified_memory`: wgpu's Metal
/// backend reports `DeviceType::IntegratedGpu` exactly when the device has unified memory). The
/// rule of [`MemoryBudget::for_unified_memory`] applies to such an adapter.
#[must_use]
pub fn is_unified_memory(info: &GpuInfo) -> bool {
    info.unified_memory
}

// ---------------------------------------------------------------------------------------------
// The slot
// ---------------------------------------------------------------------------------------------

/// Why the CPU is used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuReason {
    /// The policy is `Off`.
    Disabled,
    /// `Auto` found only a software adapter (Mesa lavapipe, WARP).
    SoftwareAdapter,
    /// No adapter opened; the text says why.
    NoAdapter(String),
}

/// What [`open_gpu`] came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuOutcome {
    /// The device is used for the stages of [`GpuSlot::stages`].
    Using,
    /// The CPU is used.
    Cpu(CpuReason),
    /// `Require` and no device opened: nothing should be rendered. The text says why.
    Unavailable(String),
}

/// The result of [`open_gpu`]: the device when it is to be used, and everything the front ends and
/// `ym doctor` say about it.
pub struct GpuSlot<D: GpuDevice = Gpu> {
    policy: GpuPolicy,
    outcome: GpuOutcome,
    gpu: Option<Arc<D>>,
    adapter: Option<GpuInfo>,
    budget: MemoryBudget,
    shared_memory: bool,
}

impl<D: GpuDevice> fmt::Debug for GpuSlot<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuSlot")
            .field("policy", &self.policy)
            .field("outcome", &self.outcome)
            .field("adapter", &self.adapter.as_ref().map(|a| &a.name))
            .field("stages", &self.stages())
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl<D: GpuDevice> GpuSlot<D> {
    fn new(policy: GpuPolicy, outcome: GpuOutcome, budget: MemoryBudget) -> Self {
        Self {
            policy,
            outcome,
            gpu: None,
            adapter: None,
            budget,
            shared_memory: false,
        }
    }

    /// A slot that uses the CPU because the policy is `Off`; nothing was opened.
    #[must_use]
    pub fn off(budget: MemoryBudget) -> Self {
        Self::new(GpuPolicy::Off, GpuOutcome::Cpu(CpuReason::Disabled), budget)
    }

    /// The policy that was asked for.
    #[must_use]
    pub fn policy(&self) -> GpuPolicy {
        self.policy
    }

    /// What came of it.
    #[must_use]
    pub fn outcome(&self) -> &GpuOutcome {
        &self.outcome
    }

    /// The device, when it is to be used.
    #[must_use]
    pub fn gpu(&self) -> Option<&Arc<D>> {
        self.gpu.as_ref()
    }

    /// Whether the device is used.
    #[must_use]
    pub fn is_using(&self) -> bool {
        matches!(self.outcome, GpuOutcome::Using)
    }

    /// The stages that run on the device: those of the policy when it is used, else none.
    #[must_use]
    pub fn stages(&self) -> GpuStages {
        if self.is_using() {
            self.policy.stages()
        } else {
            GpuStages::NONE
        }
    }

    /// The adapter that was seen, also when the policy turned it down (a software adapter under
    /// `Auto`). `None` when nothing opened or nothing was tried.
    #[must_use]
    pub fn adapter(&self) -> Option<&GpuInfo> {
        self.adapter.as_ref()
    }

    /// The memory budgets that apply: the ones asked for, with the unified-memory rule applied when
    /// the adapter shares the RAM.
    #[must_use]
    pub fn budget(&self) -> MemoryBudget {
        self.budget
    }

    /// Whether the device's memory is the machine's RAM, so that the budgets were adjusted.
    #[must_use]
    pub fn shares_memory(&self) -> bool {
        self.shared_memory
    }

    /// `Require` and no device: the reason, in plain words. `None` otherwise.
    #[must_use]
    pub fn failure(&self) -> Option<&str> {
        match &self.outcome {
            GpuOutcome::Unavailable(reason) => Some(reason),
            _ => None,
        }
    }

    /// The reason there is no usable device when one was wanted, for a warning: `Some` for an
    /// `Auto` or `On` run that found no adapter at all. A software adapter under `Auto` is a
    /// decision, not a problem, and has none.
    #[must_use]
    pub fn missing_device_reason(&self) -> Option<&str> {
        match &self.outcome {
            GpuOutcome::Cpu(CpuReason::NoAdapter(reason)) | GpuOutcome::Unavailable(reason) => {
                Some(reason)
            }
            _ => None,
        }
    }

    /// Where the image processing runs for this slot, for the report: `gpu` or `cpu`.
    #[must_use]
    pub fn device_word(&self) -> &'static str {
        if self.is_using() { "gpu" } else { "cpu" }
    }
}

impl GpuSlot<Gpu> {
    /// Puts the device, its stages and the budgets into `options`, for a [`crate::app::Core`] that
    /// is not built yet.
    pub fn apply_to(&self, options: &mut RenderOptions) {
        options.budget = self.budget;
        options.gpu = self.gpu.clone();
        options.gpu_stages = self.stages();
        options.gpu_strict = self.policy.is_strict();
    }

    /// Starts the kernel compile on the device as an `Export`-priority job, cancellable between
    /// kernels, so that a run that ends early does not wait for it. `None` when no device is used
    /// or the export render uses no stage on it yet ([`GpuStages::USED_BY_EXPORTS`]).
    #[must_use]
    pub fn warm_up_for_exports(&self) -> Option<JobHandle<Result<usize, GpuError>>> {
        self.warm_up_for(GpuStages::USED_BY_EXPORTS)
    }

    /// [`GpuSlot::warm_up_for_exports`] for the stages `used` by the work to come: `None` when no
    /// device is used or none of those stages runs on it, else the warm-up job.
    #[must_use]
    pub fn warm_up_for(&self, used: GpuStages) -> Option<JobHandle<Result<usize, GpuError>>> {
        let gpu = self.gpu.as_ref()?;
        self.stages().intersects(used).then(|| warm_up(gpu))
    }
}

/// Compiles every kernel on the device thread, as a job of `Export` priority (it queues behind
/// nothing and the render jobs of an export queue behind it), checking its token between kernels:
/// cancelling the handle, or shutting the device down, stops it after the kernel in progress.
/// Resolves to the number of kernels compiled.
pub fn warm_up(gpu: &Gpu) -> JobHandle<Result<usize, GpuError>> {
    gpu.submit(JobSpec::new(Priority::Export), |ctx, token| {
        let mut compiled = 0;
        for id in KernelId::ALL {
            token.check()?;
            ctx.kernels().pipeline(ctx, id)?;
            compiled += 1;
        }
        Ok(compiled)
    })
}

// ---------------------------------------------------------------------------------------------
// Opening
// ---------------------------------------------------------------------------------------------

/// Opens the device for `policy` with the system's graphics stack and no pipeline cache.
/// [`GpuOpener`] has the knobs.
#[must_use]
pub fn open_gpu(policy: GpuPolicy, budget: MemoryBudget) -> GpuSlot {
    GpuOpener::new(policy, budget).open()
}

/// Opens the device for a policy; see [`open_gpu`].
pub struct GpuOpener<P: AdapterProbe = SystemProbe> {
    policy: GpuPolicy,
    budget: MemoryBudget,
    pipeline_cache_dir: Option<PathBuf>,
    ram_bytes: Option<u64>,
    timestamps: bool,
    probe: P,
}

impl GpuOpener<SystemProbe> {
    /// An opener for `policy` that starts from `budget` (what the machine has with separate
    /// memories, [`MemoryBudget::detect`]), uses the real graphics stack and reads the RAM from the
    /// operating system.
    #[must_use]
    pub fn new(policy: GpuPolicy, budget: MemoryBudget) -> Self {
        Self {
            policy,
            budget,
            pipeline_cache_dir: None,
            ram_bytes: total_ram_bytes(),
            timestamps: false,
            probe: SystemProbe,
        }
    }
}

impl<P: AdapterProbe> GpuOpener<P> {
    /// Asks another probe for the device (a test double).
    #[must_use]
    pub fn with_probe<Q: AdapterProbe>(self, probe: Q) -> GpuOpener<Q> {
        GpuOpener {
            policy: self.policy,
            budget: self.budget,
            pipeline_cache_dir: self.pipeline_cache_dir,
            ram_bytes: self.ram_bytes,
            timestamps: self.timestamps,
            probe,
        }
    }

    /// The folder of the wgpu pipeline cache (the application passes
    /// `ym_gpu::device::default_pipeline_cache_dir()`; `ym` passes none).
    #[must_use]
    pub fn with_pipeline_cache_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.pipeline_cache_dir = dir;
        self
    }

    /// The machine's RAM for the unified-memory rule (`None`: not known).
    #[must_use]
    pub fn with_ram_bytes(mut self, ram_bytes: Option<u64>) -> Self {
        self.ram_bytes = ram_bytes;
        self
    }

    /// Asks for GPU pass timings from timestamp queries (`Gpu::take_pass_times`), where the adapter has
    /// them. Off by default: nothing is timed in a normal run. `ym doctor --bench` turns it on.
    #[must_use]
    pub fn with_timestamps(mut self, on: bool) -> Self {
        self.timestamps = on;
        self
    }

    fn config(&self, budget: &MemoryBudget) -> GpuConfig {
        GpuConfig {
            memory_budget_bytes: Some(budget.gpu_bytes),
            pipeline_cache_dir: self.pipeline_cache_dir.clone(),
            timestamps: self.timestamps,
            ..GpuConfig::default()
        }
    }

    /// What there is to say when no device opened.
    fn without_device(&self, budget: MemoryBudget, error: &GpuError) -> GpuSlot<P::Device> {
        let reason = error.to_string();
        tracing::info!(policy = %self.policy, %reason, "no usable GPU: using the CPU");
        let outcome = if self.policy.is_strict() {
            GpuOutcome::Unavailable(reason)
        } else {
            GpuOutcome::Cpu(CpuReason::NoAdapter(reason))
        };
        GpuSlot::new(self.policy, outcome, budget)
    }

    /// Opens the device the policy asks for, or says why not.
    ///
    /// * `Off`: nothing is opened.
    /// * `Auto`: a software adapter is closed again and the CPU is used.
    /// * `On`, `Require`: whatever opens is used.
    ///
    /// On an adapter that shares the RAM the budgets follow
    /// [`MemoryBudget::for_unified_memory`]; when that changes the GPU budget the device is
    /// closed and opened again with the new one (the pool is sized when the device is created).
    #[must_use]
    pub fn open(&self) -> GpuSlot<P::Device> {
        let policy = self.policy;
        let mut budget = self.budget;
        if policy == GpuPolicy::Off {
            return GpuSlot::off(budget);
        }
        let mut device = match self.probe.open(self.config(&budget)) {
            Ok(device) => device,
            Err(e) => return self.without_device(budget, &e),
        };
        let info = device.info().clone();

        if info.is_software && !policy.accepts_software() {
            device.close();
            tracing::info!(adapter = %info.name, "software GPU only: using the CPU");
            let mut slot =
                GpuSlot::new(policy, GpuOutcome::Cpu(CpuReason::SoftwareAdapter), budget);
            slot.adapter = Some(info);
            return slot;
        }

        let shared_memory = is_unified_memory(&info);
        let mut info = info;
        if shared_memory {
            let unified = budget.for_unified_memory(self.ram_bytes);
            if unified.gpu_bytes != budget.gpu_bytes {
                // Closed *and dropped* before the second is asked for: the device thread is gone
                // after `close`, but the wgpu device lives as long as the value does, and on an
                // 8 GB Mac two of them must not exist at once.
                device.close();
                drop(device);
                budget = unified;
                device = match self.probe.open(self.config(&budget)) {
                    Ok(device) => device,
                    Err(e) => return self.without_device(budget, &e),
                };
                info = device.info().clone();
            } else {
                budget = unified;
            }
        }
        tracing::info!(
            adapter = %info.name,
            backend = %info.backend,
            policy = %policy,
            stages = %policy.stages(),
            gpu_budget_mib = budget.gpu_bytes >> 20,
            "GPU ready"
        );
        GpuSlot {
            policy,
            outcome: GpuOutcome::Using,
            gpu: Some(Arc::new(device)),
            adapter: Some(info),
            budget,
            shared_memory,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ym_gpu::device::{GpuLimits, StorageMode};

    use super::*;

    const GIB: u64 = 1 << 30;

    /// A description of an adapter, built field by field in this one place.
    fn info(name: &str, device_type: &str, software: bool) -> GpuInfo {
        GpuInfo {
            name: name.to_owned(),
            backend: "Metal".to_owned(),
            device_type: device_type.to_owned(),
            driver: String::new(),
            driver_info: String::new(),
            vendor: 0,
            device: 0,
            is_software: software,
            shader_f16: true,
            f16_packing: true,
            pipeline_cache: false,
            unified_memory: device_type == "IntegratedGpu",
            timestamp_query: false,
            poison: false,
            storage_mode: StorageMode::F32,
            memory_budget_bytes: 0,
            limits: GpuLimits {
                max_buffer_size: 1 << 32,
                max_storage_buffer_binding_size: 1 << 31,
                max_texture_dimension_2d: 16384,
                max_storage_buffers_per_shader_stage: 10,
                max_compute_workgroups_per_dimension: 65535,
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroup_storage_size: 32768,
            },
        }
    }

    /// What the test adapter is like.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Adapter {
        /// A GPU with its own memory.
        Hardware,
        /// A GPU that shares the RAM (Apple silicon).
        Unified,
        /// Mesa lavapipe, WARP.
        Software,
        /// Nothing opens.
        None,
    }

    struct FakeDevice {
        info: GpuInfo,
        closed: Arc<AtomicUsize>,
        /// Devices that exist (opened and not dropped yet): the wgpu device of a real one lives as
        /// long as the value does, whatever `close` has stopped.
        alive: Arc<AtomicUsize>,
    }

    impl GpuDevice for FakeDevice {
        fn info(&self) -> &GpuInfo {
            &self.info
        }

        fn close(&self) {
            self.closed.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Drop for FakeDevice {
        fn drop(&mut self) {
            self.alive.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// A probe that records how it was asked. Clones share the records.
    #[derive(Clone)]
    struct FakeProbe {
        adapter: Adapter,
        opened: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
        /// Devices that exist now, and the most that ever existed at one time.
        alive: Arc<AtomicUsize>,
        most_alive: Arc<AtomicUsize>,
        budgets: Arc<Mutex<Vec<Option<u64>>>>,
        /// The request (counted from 1) that finds no adapter although the others do.
        fail_request: Option<usize>,
    }

    impl FakeProbe {
        fn new(adapter: Adapter) -> Self {
            Self {
                adapter,
                opened: Arc::default(),
                closed: Arc::default(),
                alive: Arc::default(),
                most_alive: Arc::default(),
                budgets: Arc::default(),
                fail_request: None,
            }
        }

        /// Devices opened and not closed.
        fn open_devices(&self) -> usize {
            self.opened.load(Ordering::SeqCst) - self.closed.load(Ordering::SeqCst)
        }
    }

    impl AdapterProbe for FakeProbe {
        type Device = FakeDevice;

        fn open(&self, config: GpuConfig) -> Result<FakeDevice, GpuError> {
            let request = {
                let mut budgets = self.budgets.lock().expect("budgets");
                budgets.push(config.memory_budget_bytes);
                budgets.len()
            };
            if self.fail_request == Some(request) {
                return Err(GpuError::NoAdapter("the driver went away".to_owned()));
            }
            let info = match self.adapter {
                Adapter::Hardware => info("Test discrete GPU", "DiscreteGpu", false),
                Adapter::Unified => info("Test Apple GPU", "IntegratedGpu", false),
                Adapter::Software => info("llvmpipe", "Cpu", true),
                Adapter::None => {
                    return Err(GpuError::NoAdapter("nothing enumerated".to_owned()));
                }
            };
            // Counted when a device exists; a request that finds none opens nothing.
            self.opened.fetch_add(1, Ordering::SeqCst);
            let alive = self.alive.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_alive.fetch_max(alive, Ordering::SeqCst);
            Ok(FakeDevice {
                info,
                closed: Arc::clone(&self.closed),
                alive: Arc::clone(&self.alive),
            })
        }
    }

    fn open_with(policy: GpuPolicy, adapter: Adapter) -> (GpuSlot<FakeDevice>, FakeProbe) {
        let probe = FakeProbe::new(adapter);
        let slot = GpuOpener::new(policy, MemoryBudget::for_ram(Some(16 * GIB)))
            .with_ram_bytes(Some(16 * GIB))
            .with_probe(probe.clone())
            .open();
        (slot, probe)
    }

    /// What a slot comes to, for the table: (uses the device, stages, devices still open).
    fn summary(slot: &GpuSlot<FakeDevice>, probe: &FakeProbe) -> (bool, GpuStages, usize) {
        (slot.is_using(), slot.stages(), probe.open_devices())
    }

    // --- the matrix: four policies against a hardware adapter, a software one and none ------

    #[test]
    fn the_policy_matrix_hardware_adapter() {
        let hw = |policy| {
            let (slot, probe) = open_with(policy, Adapter::Hardware);
            (summary(&slot, &probe), slot)
        };
        let ((used, stages, open), slot) = hw(GpuPolicy::Auto);
        assert_eq!((used, stages, open), (true, GpuStages::SIGNED_OFF, 1));
        assert_eq!(slot.outcome(), &GpuOutcome::Using);
        let ((used, stages, open), _) = hw(GpuPolicy::On);
        assert_eq!((used, stages, open), (true, GpuStages::ALL, 1));
        let ((used, stages, open), slot) = hw(GpuPolicy::Require);
        assert_eq!((used, stages, open), (true, GpuStages::ALL, 1));
        assert!(slot.failure().is_none());
        let ((used, stages, open), slot) = hw(GpuPolicy::Off);
        assert_eq!((used, stages, open), (false, GpuStages::NONE, 0));
        assert_eq!(slot.outcome(), &GpuOutcome::Cpu(CpuReason::Disabled));
    }

    #[test]
    fn the_policy_matrix_software_adapter() {
        // Auto never turns a software adapter on: it is closed again and the CPU is used.
        let (slot, probe) = open_with(GpuPolicy::Auto, Adapter::Software);
        assert_eq!(summary(&slot, &probe), (false, GpuStages::NONE, 0));
        assert_eq!(slot.outcome(), &GpuOutcome::Cpu(CpuReason::SoftwareAdapter));
        assert_eq!(slot.adapter().map(|a| a.name.as_str()), Some("llvmpipe"));
        assert!(
            slot.missing_device_reason().is_none(),
            "a decision, not a problem"
        );
        assert!(slot.gpu().is_none());
        // On and Require take it (CI runs on it).
        for policy in [GpuPolicy::On, GpuPolicy::Require] {
            let (slot, probe) = open_with(policy, Adapter::Software);
            assert_eq!(
                summary(&slot, &probe),
                (true, GpuStages::ALL, 1),
                "{policy}"
            );
            assert!(slot.gpu().is_some());
        }
        // Off does not even look.
        let (slot, probe) = open_with(GpuPolicy::Off, Adapter::Software);
        assert_eq!(summary(&slot, &probe), (false, GpuStages::NONE, 0));
        assert!(
            probe.budgets.lock().expect("b").is_empty(),
            "Off asks for nothing"
        );
    }

    #[test]
    fn the_policy_matrix_no_adapter() {
        for policy in [GpuPolicy::Auto, GpuPolicy::On] {
            let (slot, probe) = open_with(policy, Adapter::None);
            assert_eq!(
                summary(&slot, &probe),
                (false, GpuStages::NONE, 0),
                "{policy}"
            );
            assert!(
                matches!(slot.outcome(), GpuOutcome::Cpu(CpuReason::NoAdapter(r)) if r.contains("nothing enumerated")),
                "{policy}: {:?}",
                slot.outcome()
            );
            assert!(slot.failure().is_none(), "{policy} falls back");
            assert!(slot.missing_device_reason().is_some());
        }
        // Require is strict: no silent fallback, the reason is kept.
        let (slot, probe) = open_with(GpuPolicy::Require, Adapter::None);
        assert_eq!(summary(&slot, &probe), (false, GpuStages::NONE, 0));
        assert!(
            matches!(slot.outcome(), GpuOutcome::Unavailable(r) if r.contains("nothing enumerated"))
        );
        assert!(
            slot.failure()
                .is_some_and(|r| r.contains("no usable GPU adapter"))
        );
        let (slot, probe) = open_with(GpuPolicy::Off, Adapter::None);
        assert_eq!(summary(&slot, &probe), (false, GpuStages::NONE, 0));
        assert!(
            probe.budgets.lock().expect("b").is_empty(),
            "Off asks for nothing"
        );
    }

    #[test]
    fn the_report_word_follows_the_outcome() {
        for (policy, adapter, word) in [
            (GpuPolicy::Auto, Adapter::Hardware, "gpu"),
            (GpuPolicy::Auto, Adapter::Software, "cpu"),
            (GpuPolicy::On, Adapter::Software, "gpu"),
            (GpuPolicy::Off, Adapter::Hardware, "cpu"),
            (GpuPolicy::Require, Adapter::None, "cpu"),
        ] {
            let (slot, _) = open_with(policy, adapter);
            assert_eq!(slot.device_word(), word, "{policy} / {adapter:?}");
        }
    }

    // --- the budgets --------------------------------------------------------------------------

    #[test]
    fn the_device_is_opened_with_the_gpu_budget_of_the_machine() {
        // A GPU with its own memory: the budget as asked, one open.
        let (slot, probe) = open_with(GpuPolicy::On, Adapter::Hardware);
        assert_eq!(*probe.budgets.lock().expect("b"), vec![Some(4 * GIB)]);
        assert_eq!(slot.budget().gpu_bytes, 4 * GIB);
        assert_eq!(
            slot.budget().cpu_bytes,
            6 * GIB,
            "separate memories leave the CPU alone"
        );
        assert!(!slot.shares_memory());
    }

    #[test]
    fn a_unified_adapter_gets_the_unified_budgets_and_is_opened_again_only_when_they_differ() {
        for (ram, opens, gpu, cpu) in [
            // 16 GB: the GPU share is the default 4 GiB, so one open; the CPU budget is what 0.6 of
            // the RAM leaves (`min(cpu, 0.6 * RAM - GPU)`, `MemoryBudget::for_unified_memory`).
            (16, vec![Some(4 * GIB)], 4 * GIB, 16 * GIB * 3 / 5 - 4 * GIB),
            // 8 GB: 2 GiB differs from the 4 GiB the first open asked for, so a second open.
            (
                8,
                vec![Some(4 * GIB), Some(2 * GIB)],
                2 * GIB,
                8 * GIB * 3 / 5 - 2 * GIB,
            ),
            (64, vec![Some(4 * GIB)], 4 * GIB, 6 * GIB),
        ] {
            let probe = FakeProbe::new(Adapter::Unified);
            let slot = GpuOpener::new(GpuPolicy::Auto, MemoryBudget::for_ram(Some(ram * GIB)))
                .with_ram_bytes(Some(ram * GIB))
                .with_probe(probe.clone())
                .open();
            assert_eq!(*probe.budgets.lock().expect("b"), opens, "{ram} GB");
            assert_eq!(slot.budget().gpu_bytes, gpu, "{ram} GB: GPU");
            assert_eq!(slot.budget().cpu_bytes, cpu, "{ram} GB: CPU");
            assert!(slot.shares_memory());
            assert!(slot.is_using());
            // Exactly one device is left open: the first was closed before the second started.
            assert_eq!(probe.open_devices(), 1, "{ram} GB");
            // And the first was gone, not only stopped, when the second was asked for: two wgpu
            // devices at once is what an 8 GB Mac cannot spare.
            assert_eq!(probe.alive.load(Ordering::SeqCst), 1, "{ram} GB: alive");
            assert_eq!(
                probe.most_alive.load(Ordering::SeqCst),
                1,
                "{ram} GB: the first device was still alive when the second was opened"
            );
        }
    }

    #[test]
    fn a_second_open_that_fails_leaves_no_device_and_follows_the_policy() {
        // 8 GB: the first device is closed for the second, which then finds nothing.
        for (policy, strict) in [
            (GpuPolicy::Auto, false),
            (GpuPolicy::On, false),
            (GpuPolicy::Require, true),
        ] {
            let mut probe = FakeProbe::new(Adapter::Unified);
            probe.fail_request = Some(2);
            let slot = GpuOpener::new(policy, MemoryBudget::for_ram(Some(8 * GIB)))
                .with_ram_bytes(Some(8 * GIB))
                .with_probe(probe.clone())
                .open();
            assert_eq!(probe.budgets.lock().expect("b").len(), 2, "{policy}");
            assert!(!slot.is_using() && slot.gpu().is_none(), "{policy}");
            assert_eq!(slot.stages(), GpuStages::NONE, "{policy}");
            assert_eq!(slot.failure().is_some(), strict, "{policy}");
            assert!(
                slot.missing_device_reason()
                    .is_some_and(|r| r.contains("the driver went away")),
                "{policy}: {:?}",
                slot.outcome()
            );
            assert_eq!(
                probe.alive.load(Ordering::SeqCst),
                0,
                "{policy}: nothing left open"
            );
        }
    }

    #[test]
    fn a_unified_machine_whose_ram_is_unknown_keeps_the_asked_budget() {
        let probe = FakeProbe::new(Adapter::Unified);
        let slot = GpuOpener::new(GpuPolicy::On, MemoryBudget::default())
            .with_ram_bytes(None)
            .with_probe(probe)
            .open();
        assert_eq!(slot.budget(), MemoryBudget::default());
    }

    #[test]
    fn the_pipeline_cache_folder_reaches_the_device_request() {
        struct Spy(Mutex<Option<PathBuf>>);
        impl AdapterProbe for &Spy {
            type Device = FakeDevice;
            fn open(&self, config: GpuConfig) -> Result<FakeDevice, GpuError> {
                *self.0.lock().expect("spy") = config.pipeline_cache_dir;
                Ok(FakeDevice {
                    info: info("x", "DiscreteGpu", false),
                    closed: Arc::default(),
                    alive: Arc::new(AtomicUsize::new(1)),
                })
            }
        }
        let spy = Spy(Mutex::new(None));
        let _ = GpuOpener::new(GpuPolicy::On, MemoryBudget::default())
            .with_pipeline_cache_dir(Some(PathBuf::from("/cache")))
            .with_probe(&spy)
            .open();
        assert_eq!(
            spy.0.lock().expect("spy").as_deref(),
            Some(std::path::Path::new("/cache"))
        );
    }

    // --- flag, environment, default -----------------------------------------------------------

    #[test]
    fn the_policy_words_are_read_in_any_case() {
        for (text, policy) in [
            ("auto", GpuPolicy::Auto),
            ("AUTO", GpuPolicy::Auto),
            (" On ", GpuPolicy::On),
            ("1", GpuPolicy::On),
            ("off", GpuPolicy::Off),
            ("0", GpuPolicy::Off),
            ("Require", GpuPolicy::Require),
        ] {
            assert_eq!(GpuPolicy::parse(text), Ok(policy), "{text:?}");
        }
        for text in ["", "yes", "true", "gpu", "2"] {
            assert!(GpuPolicy::parse(text).is_err(), "{text:?}");
        }
        for policy in GpuPolicy::ALL {
            assert_eq!(GpuPolicy::parse(policy.name()), Ok(policy));
        }
    }

    #[test]
    fn a_flag_beats_the_environment_and_the_environment_beats_the_default() {
        use PolicyFlag::{Cpu, RequireGpu};
        use PolicySource::{Default, Environment, Flag};
        let r = GpuPolicy::resolve;
        // Nothing asked.
        let c = r(None, None);
        assert_eq!(
            (c.policy, c.source, c.warning),
            (GpuPolicy::Auto, Default, None)
        );
        assert_eq!(r(None, Some("")).policy, GpuPolicy::Auto, "empty is unset");
        assert_eq!(r(None, Some("  ")).source, Default);
        // The environment alone.
        for (text, policy) in [
            ("auto", GpuPolicy::Auto),
            ("on", GpuPolicy::On),
            ("off", GpuPolicy::Off),
            ("require", GpuPolicy::Require),
            ("1", GpuPolicy::On),
        ] {
            let c = r(None, Some(text));
            assert_eq!((c.policy, c.source), (policy, Environment), "{text}");
        }
        // A flag beats the environment, whatever it says.
        for env in [
            None,
            Some("auto"),
            Some("on"),
            Some("off"),
            Some("require"),
            Some("nonsense"),
        ] {
            let c = r(Some(Cpu), env);
            assert_eq!((c.policy, c.source), (GpuPolicy::Off, Flag(Cpu)), "{env:?}");
            assert!(c.warning.is_none(), "a flag does not read the variable");
            let c = r(Some(RequireGpu), env);
            assert_eq!(
                (c.policy, c.source),
                (GpuPolicy::Require, Flag(RequireGpu)),
                "{env:?}"
            );
        }
    }

    #[test]
    fn a_setting_that_is_not_a_policy_is_auto_with_a_complaint() {
        let c = GpuPolicy::resolve(None, Some("yes"));
        assert_eq!(c.policy, GpuPolicy::Auto);
        assert_eq!(c.source, PolicySource::Default);
        assert_eq!(
            c.warning.as_deref(),
            Some("YELLOWMOON_GPU=yes is not a graphics setting. Use auto, on, off or require.")
        );
    }

    #[test]
    fn the_choice_says_where_it_came_from() {
        let text = |flag, env| GpuPolicy::resolve(flag, env).to_string();
        assert_eq!(text(None, None), "auto (default)");
        assert_eq!(text(None, Some("on")), "on (YELLOWMOON_GPU)");
        assert_eq!(text(Some(PolicyFlag::Cpu), None), "off (--cpu)");
        assert_eq!(
            text(Some(PolicyFlag::RequireGpu), Some("off")),
            "require (--require-gpu)"
        );
    }

    // --- stages -------------------------------------------------------------------------------

    #[test]
    fn auto_runs_what_is_signed_off_and_on_and_require_run_everything() {
        assert_eq!(GpuPolicy::Auto.stages(), GpuStages::SIGNED_OFF);
        assert_eq!(GpuPolicy::On.stages(), GpuStages::ALL);
        assert_eq!(GpuPolicy::Require.stages(), GpuStages::ALL);
        assert_eq!(GpuPolicy::Off.stages(), GpuStages::NONE);
        // Signed off on 2026-10-14 (the raw chain and the view stages) and on 2026-10-15 (the
        // export tail and the raw front): every stage.
        assert_eq!(
            GpuStages::SIGNED_OFF.names(),
            [
                "raw",
                "sampler",
                "colour",
                "sharpening",
                "encode",
                "export",
                "raw front"
            ]
        );
        assert_eq!(GpuStages::SIGNED_OFF, GpuStages::ALL);
    }

    #[test]
    fn the_stage_set_is_a_set() {
        let all = [
            GpuStages::RAW_CHAIN,
            GpuStages::SAMPLER,
            GpuStages::COLOR,
            GpuStages::OUTPUT_SHARPEN,
            GpuStages::ENCODE,
            GpuStages::EXPORT_TAIL,
            GpuStages::RAW_FRONT,
        ];
        let mut union = GpuStages::NONE;
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert!(!a.intersects(*b), "{a} and {b} share a bit");
            }
            union = union | *a;
        }
        assert_eq!(union, GpuStages::ALL);
        assert!(GpuStages::NONE.is_empty() && !GpuStages::ALL.is_empty());
        assert_eq!(GpuStages::default(), GpuStages::NONE);
        assert_eq!(GpuStages::NONE.to_string(), "none");
        assert_eq!(
            GpuStages::ALL.to_string(),
            "raw, sampler, colour, sharpening, encode, export, raw front"
        );
        assert!(GpuStages::ALL.contains(GpuStages::NONE));
    }

    #[test]
    fn the_raw_chain_is_the_stage_of_an_export_on_the_device() {
        // The raw chain: the R6 demosaic of every export tile; the raw front: R1 to R5b and R9 of
        // every tile.
        assert!(GpuStages::USED_BY_EXPORTS.contains(GpuStages::RAW_CHAIN));
        assert!(GpuStages::USED_BY_EXPORTS.contains(GpuStages::RAW_FRONT));
        // The export tail: the sampler, colour chain, resize, sharpening and C13 of the written
        // picture.
        assert!(GpuStages::USED_BY_EXPORTS.contains(GpuStages::EXPORT_TAIL));
    }

    // --- the real stack -----------------------------------------------------------------------

    /// What the real graphics stack of this machine says, or `None` when it has no adapter (the CI
    /// jobs that mean to have one set `YM_REQUIRE_GPU=1`).
    fn real_device() -> Option<Arc<Gpu>> {
        match SystemProbe.open(GpuConfig::default()) {
            Ok(gpu) => Some(Arc::new(gpu)),
            Err(e) => {
                assert!(
                    !ym_gpu::device::gpu_required_by_env(),
                    "YM_REQUIRE_GPU is set but: {e}"
                );
                eprintln!("skipped: no GPU adapter ({e})");
                None
            }
        }
    }

    #[test]
    fn the_real_stack_follows_the_policy() {
        let Some(device) = real_device() else {
            return;
        };
        let software = device.info().is_software;
        device.close();
        drop(device);
        let budget = MemoryBudget::detect();

        let auto = open_gpu(GpuPolicy::Auto, budget);
        assert_eq!(
            auto.is_using(),
            !software,
            "Auto uses a hardware adapter only"
        );
        let on = open_gpu(GpuPolicy::On, budget);
        assert!(on.is_using());
        assert_eq!(on.stages(), GpuStages::ALL);
        let off = open_gpu(GpuPolicy::Off, budget);
        assert!(!off.is_using() && off.gpu().is_none() && off.adapter().is_none());
        let required = open_gpu(GpuPolicy::Require, budget);
        assert!(required.is_using());
        // The device the slot holds is a working one, and it can be handed to a renderer.
        let mut options = RenderOptions::default();
        on.apply_to(&mut options);
        assert!(options.gpu.is_some());
        assert_eq!(options.gpu_stages, GpuStages::ALL);
        assert_eq!(options.budget, on.budget());
    }

    #[test]
    fn warming_up_compiles_every_kernel_and_stops_when_cancelled() {
        let Some(device) = real_device() else {
            return;
        };
        let handle = warm_up(&device);
        // Cancelled at once: either it never started or it stops after the kernel in progress.
        handle.cancel();
        match handle.wait() {
            Ok(Ok(n)) => assert!(n <= KernelId::ALL.len()),
            Ok(Err(GpuError::Cancelled)) | Err(_) => {}
            Ok(Err(e)) => panic!("warm-up failed: {e}"),
        }
        // Not cancelled: all of them, and a second run finds them compiled.
        let compiled = warm_up(&device).wait().expect("job").expect("compiled");
        assert_eq!(compiled, KernelId::ALL.len());
        let again = warm_up(&device).wait().expect("job").expect("compiled");
        assert_eq!(again, KernelId::ALL.len());
    }

    #[test]
    fn exports_warm_up_only_the_stages_they_run_on_the_device() {
        let slot = open_gpu(GpuPolicy::On, MemoryBudget::default());
        if !slot.is_using() {
            assert!(
                !ym_gpu::device::gpu_required_by_env(),
                "{:?}",
                slot.outcome()
            );
            return;
        }
        // The raw chain is on the device for an export under `On`, so `ym` warms the kernels up
        // while the photos develop; cancelled at once here (the job itself is tested above).
        let started = slot
            .warm_up_for_exports()
            .expect("the raw chain is on the device");
        started.cancel();
        let _ = started.wait();
        // `Auto` runs the signed-off stages; since 2026-10-15 they are every stage, so an export
        // under `Auto` on a hardware device warms the kernels of the raw chain, the raw front and
        // the export tail up too.
        let auto = open_gpu(GpuPolicy::Auto, MemoryBudget::default());
        if auto.is_using() {
            let started = auto
                .warm_up_for_exports()
                .expect("the export stages are signed off");
            started.cancel();
            let _ = started.wait();
        }
    }

    #[test]
    fn the_warm_up_starts_for_a_stage_the_device_runs_and_for_nothing_else() {
        // The branch taken when `GpuStages::USED_BY_EXPORTS` holds a stage the device runs: it
        // must start the job (and only then).
        let slot = open_gpu(GpuPolicy::On, MemoryBudget::default());
        if !slot.is_using() {
            assert!(
                !ym_gpu::device::gpu_required_by_env(),
                "{:?}",
                slot.outcome()
            );
            return;
        }
        assert!(slot.warm_up_for(GpuStages::NONE).is_none(), "nothing used");
        // Started; it is cancelled at once, because the job itself (every kernel compiled, the
        // cancel honoured) is the subject of the test above and a full compile costs seconds on a
        // real GPU.
        let started = slot.warm_up_for(GpuStages::EXPORT_TAIL).expect("started");
        started.cancel();
        match started.wait() {
            Ok(Ok(n)) => assert!(n <= KernelId::ALL.len()),
            Ok(Err(GpuError::Cancelled)) | Err(_) => {}
            Ok(Err(e)) => panic!("warm-up failed: {e}"),
        }
        // A slot with no device has nothing to warm up, whatever is used.
        let off: GpuSlot = GpuSlot::off(MemoryBudget::default());
        assert!(off.warm_up_for(GpuStages::ALL).is_none());
        assert!(off.warm_up_for_exports().is_none());
    }
}
