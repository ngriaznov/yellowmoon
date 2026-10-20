//! The job scheduler: priorities, cancellation tokens and memory budgets (`docs/01-architecture.md`
//! section Scheduling and threads).
//!
//! * **Priorities.** Viewport frames run before visible-tile passes, those before thumbnails, those
//!   before export ([`Priority`]). A job that is running is never pre-empted.
//! * **Two workers.** The first runs everything the person is waiting for (viewport, tile and
//!   thumbnail jobs) one at a time in priority order. The second runs export jobs and **gives way**
//!   to the first: an export job starts only when no foreground job is queued or running, and a
//!   long one (a full-resolution render) calls [`JobScheduler::wait_for_foreground`] between its
//!   tiles, so a viewport frame asked for during an export waits for the tile in progress at most,
//!   not for the whole export (`docs/09`: "Exports keep running while the user browses"). One queue
//!   with one worker could not do that: it never pre-empts, so a single export job would hold up
//!   every frame for as long as it takes (36 s for 50 MP on four cores). While an export tile
//!   runs, a foreground job shares the cores with it; that overlap is bounded by one tile.
//! * **Cancellation.** Every job carries a token. Moving to another photo ([`JobScheduler::navigate`])
//!   cancels the viewport and tile jobs of the photo being left, pending and running, and the jobs
//!   queued with [`JobScheduler::submit_superseded`] (the Look tab's tiles and the histogram: panel
//!   requests that nobody wants once the person has moved on) for that photo, letting go of what
//!   they hold at once; the filmstrip's thumbnails and export jobs are never cancelled by
//!   navigation. An export stops only through its own handle.
//! * **Memory.** At most [`MemoryBudget::max_mosaics`] decoded mosaics (the current photo and its
//!   neighbours, kept by the session) and [`MemoryBudget::max_photos`] render graphs. The render
//!   caches are byte-bounded LRU maps ([`ByteLru`]); a full-resolution render reserves its working
//!   set in the [`MemoryLedger`] first and is refused, rather than started, when the budget would be
//!   exceeded. Budget defaults follow `docs/01`: 6 GiB of CPU memory scaled down to half the
//!   machine's RAM, 4 GiB of GPU memory ([`MemoryBudget::gpu_bytes`], the budget the GPU context is
//!   opened with by [`crate::app::gpu::open_gpu`]). On a machine whose GPU shares the RAM (Apple
//!   silicon) the two come out of one pool: [`MemoryBudget::for_unified_memory`].
//!
//! The queues are `ym_gpu::queue::Scheduler`, the same one that fronts the GPU thread, so the
//! ordering and cancellation rules are the ones its tests pin; this module adds the photo-aware
//! navigation rule, the export's yielding and the memory accounting. Each job is data-parallel
//! inside (rayon, scoped threads), so a second *foreground* worker would only compete for the same
//! cores.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use ym_gpu::queue::Scheduler;
pub use ym_gpu::queue::{CancelToken, Cancelled, JobError, JobHandle, JobSpec, Priority};

use crate::session::PhotoId;

/// Default CPU memory budget, `docs/01`: 6 GiB.
pub const DEFAULT_CPU_BUDGET_BYTES: u64 = 6 << 30;
/// Default GPU memory budget, `docs/01`: 4 GiB (`ym_gpu::pool::MemoryBudget::DEFAULT_BYTES`).
pub const DEFAULT_GPU_BUDGET_BYTES: u64 = 4 << 30;
/// The least CPU budget any machine gets.
pub const MIN_CPU_BUDGET_BYTES: u64 = 512 << 20;

/// What the core may keep in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Decoded mosaics kept at once (`docs/01`: the current photo, the previous and the next). The
    /// session owns the mosaics ([`crate::session::MOSAIC_CACHE_LEN`]); the renderer keeps none, so
    /// this is the figure the session is held to and what the tests check it against.
    pub max_mosaics: usize,
    /// Photos whose render caches are kept at once.
    pub max_photos: usize,
    /// CPU bytes for render caches and full-resolution working sets.
    pub cpu_bytes: u64,
    /// GPU bytes: the budget to open the GPU context with (`GpuConfig`); the renderer does not
    /// create the context, the application does, so this is the number it passes on.
    pub gpu_bytes: u64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_mosaics: crate::session::MOSAIC_CACHE_LEN,
            max_photos: 3,
            cpu_bytes: DEFAULT_CPU_BUDGET_BYTES,
            gpu_bytes: DEFAULT_GPU_BUDGET_BYTES,
        }
    }
}

/// A GPU that shares the RAM gets this share of it, as a divisor: `RAM / 4` (`docs/17-gpu.md`
/// section 8, STATUS question 16).
pub const UNIFIED_GPU_RAM_DIVISOR: u64 = 4;
/// The least GPU budget on a machine whose GPU shares the RAM.
pub const UNIFIED_GPU_MIN_BYTES: u64 = 512 << 20;
/// The most GPU budget on such a machine: the same 4 GiB as everywhere.
pub const UNIFIED_GPU_MAX_BYTES: u64 = DEFAULT_GPU_BUDGET_BYTES;
/// What the CPU and GPU budgets of a unified-memory machine may add up to, as a share of the RAM
/// (3 / 5 = 0.6). The rule of [`MemoryBudget::for_unified_memory`] meets it for every machine from
/// 4 GiB up; the unit tests assert it.
pub const UNIFIED_TOTAL_RAM_SHARE: (u64, u64) = (3, 5);

/// What the CPU may take of `ram` bytes when the GPU takes `gpu_bytes` of the same memory:
/// [`UNIFIED_TOTAL_RAM_SHARE`] of the RAM minus the GPU's share (zero when that is negative).
fn shared_cpu_limit(ram: u64, gpu_bytes: u64) -> u64 {
    let (num, den) = UNIFIED_TOTAL_RAM_SHARE;
    (u128::from(ram) * u128::from(num) / u128::from(den))
        .saturating_sub(u128::from(gpu_bytes))
        .try_into()
        .unwrap_or(u64::MAX)
}

impl MemoryBudget {
    /// The defaults scaled down to the machine: the CPU budget is at most half of the installed RAM.
    /// The GPU budget stays at its default (a GPU with its own memory, or no GPU); a GPU that shares
    /// the RAM is accounted for by [`MemoryBudget::for_unified_memory`] once the device is known.
    #[must_use]
    pub fn detect() -> Self {
        Self::for_ram(total_ram_bytes())
    }

    /// [`MemoryBudget::detect`] for a machine with `ram_bytes` of RAM (`None`: not known).
    #[must_use]
    pub fn for_ram(ram_bytes: Option<u64>) -> Self {
        let mut b = Self::default();
        if let Some(total) = ram_bytes {
            b.cpu_bytes = b.cpu_bytes.min(total / 2).max(MIN_CPU_BUDGET_BYTES);
        }
        b
    }

    /// The budgets of a machine whose GPU shares the RAM with the CPU (`DeviceType::IntegratedGpu`:
    /// Metal on Apple silicon, an integrated GPU elsewhere), given the budgets it would have with
    /// separate memories and its `ram_bytes`.
    ///
    /// The GPU gets `clamp(RAM / 4, 512 MiB, 4 GiB)` and the CPU budget becomes
    /// `min(cpu, 0.6 * RAM - gpu)` (never below [`MIN_CPU_BUDGET_BYTES`]), so the two together stay
    /// under 0.6 of the RAM ([`UNIFIED_TOTAL_RAM_SHARE`]) and the CPU keeps everything the bound
    /// leaves it: 8 GB gives 2 GiB and 2.8 GiB, 16 GB 4 GiB and 5.6 GiB, 64 GB 4 GiB and 6 GiB. With
    /// the RAM unknown nothing changes.
    ///
    /// *History.* The first form of the rule lowered the CPU budget by the whole GPU amount, which
    /// cut every Apple-silicon Mac to 2 GiB of CPU budget while only the colour pass used the device
    /// (the graph caches fall to a third and the finished frames to a third with it). The GPU rule is unchanged; the CPU rule only claims what the 0.6
    /// bound still allows. Proposal pending the owner's answer (STATUS question 16); the whole rule
    /// is this function and the constants above.
    #[must_use]
    pub fn for_unified_memory(self, ram_bytes: Option<u64>) -> Self {
        let Some(ram) = ram_bytes else {
            return self;
        };
        let gpu_bytes =
            (ram / UNIFIED_GPU_RAM_DIVISOR).clamp(UNIFIED_GPU_MIN_BYTES, UNIFIED_GPU_MAX_BYTES);
        Self {
            gpu_bytes,
            cpu_bytes: self
                .cpu_bytes
                .min(shared_cpu_limit(ram, gpu_bytes))
                .max(MIN_CPU_BUDGET_BYTES),
            ..self
        }
    }

    /// The same budget with another CPU limit.
    #[must_use]
    pub fn with_cpu_bytes(mut self, bytes: u64) -> Self {
        self.cpu_bytes = bytes;
        self
    }
}

/// Installed RAM in bytes, where the operating system says (Linux `/proc/meminfo`, macOS `sysctl`).
#[must_use]
pub fn total_ram_bytes() -> Option<u64> {
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb * 1024);
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// A request that would take the CPU memory over its budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{requested} bytes requested, {used} of {budget} in use")]
pub struct OverBudget {
    /// Bytes asked for.
    pub requested: u64,
    /// Bytes reserved already.
    pub used: u64,
    /// The budget.
    pub budget: u64,
}

/// Counts the bytes that full-resolution jobs hold while they run.
#[derive(Debug)]
pub struct MemoryLedger {
    budget: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl MemoryLedger {
    /// A ledger for `budget` bytes.
    #[must_use]
    pub fn new(budget: u64) -> Arc<Self> {
        Arc::new(Self {
            budget: AtomicU64::new(budget),
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        })
    }

    /// Reserves `bytes` until the returned guard is dropped.
    ///
    /// # Errors
    ///
    /// [`OverBudget`] when the reservation would exceed the budget; nothing is reserved.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation, OverBudget> {
        let mut cur = self.used.load(Ordering::SeqCst);
        loop {
            let next = cur.saturating_add(bytes);
            let budget = self.budget();
            if next > budget {
                return Err(OverBudget {
                    requested: bytes,
                    used: cur,
                    budget,
                });
            }
            match self
                .used
                .compare_exchange(cur, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::SeqCst);
                    return Ok(Reservation {
                        ledger: Arc::clone(self),
                        bytes,
                    });
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Bytes reserved now.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::SeqCst)
    }

    /// The most bytes reserved at once.
    #[must_use]
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::SeqCst)
    }

    /// The budget.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget.load(Ordering::SeqCst)
    }

    /// Changes the budget (the unified-memory rule lowers it once the GPU is known). Bytes that are
    /// reserved already stay reserved; only later requests are held to the new figure.
    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::SeqCst);
    }
}

/// Bytes reserved in a [`MemoryLedger`]; released when dropped.
#[derive(Debug)]
pub struct Reservation {
    ledger: Arc<MemoryLedger>,
    bytes: u64,
}

impl Reservation {
    /// The reserved size.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// A least-recently-used map bounded by the bytes its values hold. Small (a handful of entries):
/// a vector, newest last.
#[derive(Debug)]
pub struct ByteLru<K, V> {
    entries: Vec<(K, V, usize)>,
    budget: usize,
    used: usize,
}

impl<K: PartialEq, V> ByteLru<K, V> {
    /// An empty map that holds at most `budget` bytes (the newest entry is always kept, even when it
    /// alone is larger).
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            used: 0,
        }
    }

    /// The value for `key`, marking it most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let i = self.entries.iter().position(|(k, _, _)| k == key)?;
        let e = self.entries.remove(i);
        self.entries.push(e);
        self.entries.last().map(|(_, v, _)| v)
    }

    /// Whether `key` is present (does not touch the order).
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.entries.iter().any(|(k, _, _)| k == key)
    }

    /// Inserts `value` that holds `bytes` bytes, replacing an equal key, then evicts the least
    /// recently used entries until the budget holds.
    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        if let Some(i) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.used -= self.entries.remove(i).2;
        }
        self.entries.push((key, value, bytes));
        self.used += bytes;
        while self.used > self.budget && self.entries.len() > 1 {
            self.used -= self.entries.remove(0).2;
        }
    }

    /// Removes every entry for which `drop_it` is true.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        let mut used = 0;
        self.entries.retain(|(k, v, b)| {
            let keep_it = keep(k, v);
            if keep_it {
                used += b;
            }
            keep_it
        });
        self.used = used;
    }

    /// Removes everything.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }

    /// Bytes held.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Changes the budget and evicts to fit.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        while self.used > self.budget && self.entries.len() > 1 {
            self.used -= self.entries.remove(0).2;
        }
    }
}

/// How often an export waiting for the foreground checks whether it is idle.
const YIELD_POLL: Duration = Duration::from_millis(2);

/// Counts a foreground job from the moment it is queued until it has finished, been cancelled or
/// been dropped unrun (the guard travels inside the job's closure, which the queue drops either
/// way).
struct LoadGuard(Arc<AtomicUsize>);

impl LoadGuard {
    fn new(load: &Arc<AtomicUsize>) -> Self {
        load.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(load))
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A queued job that moving to another photo supersedes (see [`JobScheduler::submit_superseded`]).
struct Supersedable {
    /// The photo the job is for.
    owner: u64,
    /// The job has finished or been dropped unrun: there is nothing left to cancel.
    finished: Arc<AtomicBool>,
    token: CancelToken,
    /// Drops the job's closure, and what it captured (the photo's decoded mosaic), if it has not
    /// started.
    discard: Box<dyn Fn() + Send + Sync>,
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Travels inside the closure of a superseded-able job and ends the job's entry in the registry
/// when the closure is dropped, however that comes about: the job ran (a panic included), or the
/// queue dropped it unrun (the session closed, the scheduler shut down, its handle was cancelled).
/// An unrun job's closure is gone with the queue's copy of it, but the registry's `discard` still
/// holds the job's slot, and with it what the job captured (the photo's decoded mosaic): the guard
/// empties the slot so that nothing outlives the job.
struct Release<F> {
    finished: Arc<AtomicBool>,
    slot: Arc<Mutex<Option<F>>>,
}

impl<F> Drop for Release<F> {
    fn drop(&mut self) {
        drop(locked(&self.slot).take());
        self.finished.store(true, Ordering::SeqCst);
    }
}

/// Runs render jobs in priority order (see the module documentation).
pub struct JobScheduler {
    /// Viewport, tile and thumbnail jobs, one at a time.
    foreground: Scheduler<()>,
    /// Export jobs, which give way to the foreground.
    background: Scheduler<()>,
    /// Foreground jobs queued or running.
    load: Arc<AtomicUsize>,
    /// The photo being looked at, for [`JobScheduler::navigate`] (0: none).
    current: AtomicU64,
    /// Queued panel requests that navigation supersedes.
    superseded: Mutex<Vec<Supersedable>>,
}

impl std::fmt::Debug for JobScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobScheduler")
            .field("pending", &self.pending())
            .field("foreground_load", &self.foreground_load())
            .finish()
    }
}

fn start_worker(name: &str) -> std::io::Result<Scheduler<()>> {
    Scheduler::<()>::spawn::<(), std::convert::Infallible, _>(name, || Ok(((), ())))
        .map(|(scheduler, ())| scheduler)
        .map_err(|e| match e {
            ym_gpu::queue::SpawnError::Thread(io) => io,
            ym_gpu::queue::SpawnError::Init(never) => match never {},
        })
}

impl JobScheduler {
    /// Starts the worker threads (`<name>` for the foreground, `<name>-export` for exports).
    ///
    /// # Errors
    ///
    /// The operating system's error when a thread cannot start.
    pub fn new(name: &str) -> std::io::Result<Self> {
        let foreground = start_worker(name)?;
        let background = start_worker(&format!("{name}-export"))?;
        Ok(Self {
            foreground,
            background,
            load: Arc::new(AtomicUsize::new(0)),
            current: AtomicU64::new(0),
            superseded: Mutex::new(Vec::new()),
        })
    }

    /// Queues a job. `owner` is the photo the job renders; navigation cancels the viewport and tile
    /// jobs of the photo it leaves.
    ///
    /// A [`Priority::Export`] job goes to the export worker and starts when the foreground is idle
    /// (it is called at once, with a token that may already be cancelled, if the scheduler shuts
    /// down or the handle is cancelled while it waits). A long export job should call
    /// [`JobScheduler::wait_for_foreground`] between its pieces.
    pub fn submit<R, F>(&self, priority: Priority, owner: Option<PhotoId>, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&CancelToken) -> R + Send + 'static,
    {
        let mut spec = JobSpec::new(priority);
        if let Some(o) = owner {
            spec = spec.owned_by(u64::from(o.0));
        }
        if priority == Priority::Export {
            let load = Arc::clone(&self.load);
            self.background.submit(spec, move |(), token| {
                // A cancelled wait still runs the job: it sees its token set at its first check
                // and returns its own cancellation result.
                let _ = wait_until_idle(&load, token);
                job(token)
            })
        } else {
            let guard = LoadGuard::new(&self.load);
            self.foreground.submit(spec, move |(), token| {
                let _guard = guard;
                job(token)
            })
        }
    }

    /// Queues a job for `owner` that moving on from that photo supersedes: a request of a panel (a
    /// look tile, the histogram) for the photo on screen, which nobody wants once the person has
    /// stepped to the next one. [`JobScheduler::navigate`] away from `owner` takes the job out of
    /// the queue's hands: its closure, and the decoded mosaic it holds, are dropped at once (a
    /// queued job otherwise pins its photo's mosaic until its turn comes, and stepping through
    /// photos faster than the tiles render piled up mosaics and wasted the render worker on
    /// pictures nobody sees), and the handle resolves with [`JobError::Cancelled`], as for a
    /// viewport job of the photo left (with `superseded()` instead, in the instant between the
    /// worker taking the job and running it). A job that is already running has its token set and
    /// decides when to stop.
    ///
    /// Like the viewport and tile jobs, only the photo being left is affected: a request for the
    /// photo being moved to, which may arrive before the move is recorded, is never cancelled.
    pub fn submit_superseded<R, F, G>(
        &self,
        priority: Priority,
        owner: PhotoId,
        job: F,
        superseded: G,
    ) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&CancelToken) -> R + Send + 'static,
        G: FnOnce() -> R + Send + 'static,
    {
        let slot = Arc::new(Mutex::new(Some(job)));
        let finished = Arc::new(AtomicBool::new(false));
        let handle = {
            let release = Release {
                finished: Arc::clone(&finished),
                slot: Arc::clone(&slot),
            };
            self.submit(priority, Some(owner), move |token| {
                // The whole guard moves in (a type with a destructor is captured whole anyway).
                let release = release;
                let job = locked(&release.slot).take();
                let result = match job {
                    Some(job) if !token.is_cancelled() => job(token),
                    _ => superseded(),
                };
                // The job is over, and so is its entry in the registry.
                drop(release);
                result
            })
        };
        let mut registry = locked(&self.superseded);
        registry.retain(|e| !e.finished.load(Ordering::SeqCst));
        registry.push(Supersedable {
            owner: u64::from(owner.0).max(1),
            finished,
            token: handle.token().clone(),
            discard: Box::new(move || drop(locked(&slot).take())),
        });
        drop(registry);
        handle
    }

    /// Supersedes the panel requests of `owner`; returns how many were still waiting or running.
    fn supersede(&self, owner: u64) -> usize {
        let doomed: Vec<Supersedable> = {
            let mut registry = locked(&self.superseded);
            let (doomed, kept) = std::mem::take(&mut *registry)
                .into_iter()
                .filter(|e| !e.finished.load(Ordering::SeqCst))
                .partition(|e| e.owner == owner);
            *registry = kept;
            doomed
        };
        let mut n = 0;
        for e in doomed {
            if !e.token.is_cancelled() {
                n += 1;
            }
            e.token.cancel();
            (e.discard)();
        }
        n
    }

    /// Blocks while a viewport, tile or thumbnail job is queued or running, and returns when the
    /// foreground is idle: the point where a long export job may take its next piece.
    ///
    /// Must not be called from a foreground job (it would wait for itself).
    ///
    /// # Errors
    ///
    /// [`Cancelled`] as soon as `cancel` fires.
    pub fn wait_for_foreground(&self, cancel: &CancelToken) -> Result<(), Cancelled> {
        wait_until_idle(&self.load, cancel)
    }

    /// Foreground jobs queued or running now.
    #[must_use]
    pub fn foreground_load(&self) -> usize {
        self.load.load(Ordering::SeqCst)
    }

    /// Records that the user moved to `photo` and cancels the viewport and tile jobs of the photo
    /// they left, and its panel requests ([`JobScheduler::submit_superseded`]). Returns how many
    /// jobs were cancelled. Filmstrip thumbnail and export jobs are untouched.
    pub fn navigate(&self, photo: PhotoId) -> usize {
        let new = u64::from(photo.0).max(1);
        let old = self.current.swap(new, Ordering::SeqCst);
        if old == 0 || old == new {
            return 0;
        }
        self.supersede(old) + self.foreground.cancel_navigation(Some(old))
    }

    /// Photos left the session (removed from the filmstrip): cancels every viewport, tile, thumbnail, histogram, readout
    /// and look-tile job of theirs, queued or running, and the panel requests waiting to be superseded, so that nothing
    /// is rendered for them and no queued job keeps their decoded mosaic alive. **Export jobs are left alone**: an export
    /// holds its own snapshot of each photo and finishes the photos it was started with. Returns how many jobs were
    /// cancelled.
    pub fn cancel_photos(&self, photos: &[PhotoId]) -> usize {
        let owners: Vec<u64> = photos.iter().map(|p| u64::from(p.0).max(1)).collect();
        // The queue first: a look tile or histogram request that has not started is dropped there with its closure
        // (and the mosaic it holds), so the registry of superseded-able requests finds it finished and does not count it again.
        let queued = self.foreground.cancel_owners(&owners);
        queued + owners.iter().map(|&o| self.supersede(o)).sum::<usize>()
    }

    /// Cancels every pending job and sets the token of the running ones (the session closed).
    pub fn cancel_all(&self) -> usize {
        self.foreground.cancel_all() + self.background.cancel_all()
    }

    /// Cancels the viewport, tile and thumbnail jobs, pending and running, and leaves the export
    /// jobs alone: what closing the session does (an export that was queued keeps going).
    pub fn cancel_foreground(&self) -> usize {
        self.foreground.cancel_all()
    }

    /// Jobs waiting, not counting the ones that run.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.foreground.pending() + self.background.pending()
    }

    /// Stops the workers after the running jobs; pending jobs resolve with [`JobError::Shutdown`].
    pub fn shutdown(&self) {
        self.foreground.shutdown();
        self.background.shutdown();
    }
}

fn wait_until_idle(load: &AtomicUsize, cancel: &CancelToken) -> Result<(), Cancelled> {
    loop {
        cancel.check()?;
        if load.load(Ordering::SeqCst) == 0 {
            return Ok(());
        }
        std::thread::sleep(YIELD_POLL);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, mpsc};
    use std::time::Duration;

    use super::*;

    /// Blocks the worker until the returned sender is used.
    fn block(s: &JobScheduler) -> (mpsc::Sender<()>, JobHandle<()>) {
        let (tx, rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let h = s.submit(Priority::Viewport, None, move |_| {
            ready_tx.send(()).expect("ready");
            rx.recv().expect("gate");
        });
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("started");
        (tx, h)
    }

    #[test]
    fn jobs_run_viewport_then_tiles_then_thumbnails_then_export() {
        let s = JobScheduler::new("test-prio").expect("scheduler");
        let (gate, gate_job) = block(&s);
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (name, p) in [
            ("export", Priority::Export),
            ("thumbnail", Priority::Thumbnail),
            ("tile", Priority::TileNeural),
            ("viewport", Priority::Viewport),
        ] {
            let order = Arc::clone(&order);
            handles.push(s.submit(p, Some(PhotoId(1)), move |_| {
                order.lock().expect("order").push(name);
            }));
        }
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        for h in handles {
            h.wait().expect("job");
        }
        assert_eq!(
            *order.lock().expect("order"),
            vec!["viewport", "tile", "thumbnail", "export"]
        );
    }

    #[test]
    fn navigation_cancels_viewport_and_tile_jobs_of_the_photo_left_but_never_exports() {
        let s = JobScheduler::new("test-nav").expect("scheduler");
        assert_eq!(s.navigate(PhotoId(1)), 0, "the first photo cancels nothing");
        let (gate, gate_job) = block(&s);
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (name, p, owner) in [
            ("viewport-1", Priority::Viewport, 1),
            ("tile-1", Priority::TileNeural, 1),
            ("thumbnail-1", Priority::Thumbnail, 1),
            ("export-1", Priority::Export, 1),
            ("viewport-2", Priority::Viewport, 2),
        ] {
            let ran = Arc::clone(&ran);
            handles.push((
                name,
                s.submit(p, Some(PhotoId(owner)), move |_| {
                    ran.lock().expect("ran").push(name);
                }),
            ));
        }
        let cancelled = s.navigate(PhotoId(2));
        assert_eq!(cancelled, 2, "the viewport and tile jobs of photo 1");
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        let mut results = Vec::new();
        for (name, h) in handles {
            results.push((name, h.wait()));
        }
        for (name, r) in &results {
            match *name {
                "viewport-1" | "tile-1" => assert_eq!(r, &Err(JobError::Cancelled), "{name}"),
                _ => assert!(r.is_ok(), "{name} must still run: {r:?}"),
            }
        }
        let ran = ran.lock().expect("ran").clone();
        assert!(ran.contains(&"export-1") && ran.contains(&"thumbnail-1"));
        assert!(!ran.contains(&"viewport-1") && !ran.contains(&"tile-1"));
    }

    #[test]
    fn cancelling_photos_takes_every_foreground_job_of_theirs_but_no_export() {
        let s = JobScheduler::new("test-gone").expect("scheduler");
        let (gate, gate_job) = block(&s);
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (name, p, owner) in [
            ("viewport-1", Priority::Viewport, 1),
            ("thumbnail-1", Priority::Thumbnail, 1),
            ("thumbnail-2", Priority::Thumbnail, 2),
            ("export-1", Priority::Export, 1),
            ("thumbnail-3", Priority::Thumbnail, 3),
        ] {
            let ran = Arc::clone(&ran);
            handles.push((
                name,
                s.submit(p, Some(PhotoId(owner)), move |_| {
                    ran.lock().expect("ran").push(name);
                }),
            ));
        }
        // A panel request that waits to be superseded (a look tile) goes too, and lets go of what it holds.
        let tile = s.submit_superseded(Priority::Thumbnail, PhotoId(2), |_| "ran", || "superseded");
        let cancelled = s.cancel_photos(&[PhotoId(1), PhotoId(2)]);
        assert_eq!(
            cancelled, 4,
            "viewport-1, thumbnail-1, thumbnail-2 and the look tile of photo 2"
        );
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        for (name, h) in handles {
            let r = h.wait();
            match name {
                "viewport-1" | "thumbnail-1" | "thumbnail-2" => {
                    assert_eq!(r, Err(JobError::Cancelled), "{name}");
                }
                _ => assert!(r.is_ok(), "{name} must still run: {r:?}"),
            }
        }
        assert_eq!(tile.wait(), Err(JobError::Cancelled));
        assert_eq!(*ran.lock().expect("ran"), vec!["thumbnail-3", "export-1"]);
        assert_eq!(s.cancel_photos(&[]), 0);
    }

    #[test]
    fn a_running_viewport_job_sees_its_token_set_by_navigation() {
        let s = JobScheduler::new("test-run").expect("scheduler");
        s.navigate(PhotoId(7));
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let h = s.submit(Priority::TileNeural, Some(PhotoId(7)), move |token| {
            started_tx.send(()).expect("started");
            stop_rx.recv().expect("stop");
            token.is_cancelled()
        });
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("running");
        assert_eq!(s.navigate(PhotoId(8)), 1);
        stop_tx.send(()).expect("stop");
        assert_eq!(h.wait(), Ok(true));
    }

    #[test]
    fn an_export_is_cancelled_only_through_its_own_handle() {
        let s = JobScheduler::new("test-export").expect("scheduler");
        s.navigate(PhotoId(1));
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let h = s.submit(Priority::Export, Some(PhotoId(1)), move |token| {
            started_tx.send(()).expect("started");
            stop_rx.recv().expect("stop");
            token.is_cancelled()
        });
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("running");
        s.navigate(PhotoId(2));
        s.navigate(PhotoId(3));
        h.cancel();
        stop_tx.send(()).expect("stop");
        assert_eq!(h.wait(), Ok(true), "only its own handle set the token");
    }

    #[test]
    fn an_export_piece_waits_for_the_foreground_and_goes_on_when_it_is_idle() {
        let s = Arc::new(JobScheduler::new("test-yield").expect("scheduler"));
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let (piece_done_tx, piece_done_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        // An export of three pieces that looks at the foreground before each one.
        let export = {
            let (s2, log) = (Arc::clone(&s), Arc::clone(&log));
            s.submit(Priority::Export, Some(PhotoId(1)), move |token| {
                for piece in 0..3 {
                    s2.wait_for_foreground(token).expect("not cancelled");
                    log.lock().expect("log").push(format!("piece {piece}"));
                    if piece == 0 {
                        piece_done_tx.send(()).expect("first piece");
                        go_rx.recv().expect("go");
                    }
                }
            })
        };
        piece_done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the first piece ran on an idle foreground");
        // A viewport job arrives while the export is between pieces and is held open.
        let (up_tx, up_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let viewport = {
            let log = Arc::clone(&log);
            s.submit(Priority::Viewport, Some(PhotoId(1)), move |_| {
                up_tx.send(()).expect("up");
                release_rx.recv().expect("release");
                log.lock().expect("log").push("viewport".to_owned());
            })
        };
        up_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the foreground job started at once, with the export mid-way");
        go_tx.send(()).expect("go");
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(
            *log.lock().expect("log"),
            vec!["piece 0".to_owned()],
            "the next piece waits while the viewport job runs"
        );
        release_tx.send(()).expect("release");
        viewport.wait().expect("viewport");
        export.wait().expect("export");
        assert_eq!(
            *log.lock().expect("log"),
            vec!["piece 0", "viewport", "piece 1", "piece 2"]
        );
    }

    #[test]
    fn the_foreground_load_counts_queued_and_running_jobs_and_always_comes_back_to_zero() {
        let s = JobScheduler::new("test-load").expect("scheduler");
        s.navigate(PhotoId(1));
        let (gate, gate_job) = block(&s);
        assert_eq!(s.foreground_load(), 1);
        let queued = s.submit(Priority::Viewport, Some(PhotoId(1)), |_| ());
        let cancelled = s.submit(Priority::TileNeural, Some(PhotoId(1)), |_| ());
        let panics = s.submit(Priority::Thumbnail, Some(PhotoId(2)), |_| -> () {
            panic!("a failing job")
        });
        assert_eq!(s.foreground_load(), 4);
        // Navigation cancels the viewport and tile jobs of photo 1 before they run.
        assert_eq!(s.navigate(PhotoId(2)), 2);
        assert_eq!(s.foreground_load(), 2, "a cancelled job is not load");
        assert_eq!(queued.wait(), Err(JobError::Cancelled));
        assert_eq!(cancelled.wait(), Err(JobError::Cancelled));
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        assert!(matches!(panics.wait(), Err(JobError::Panicked(_))));
        assert_eq!(s.foreground_load(), 0, "a panic does not leak load");
        // With nothing queued an export is not held up at all.
        let h = s.submit(Priority::Export, None, |_| 7);
        assert_eq!(h.wait(), Ok(7));
    }

    #[test]
    fn an_export_waiting_for_the_foreground_runs_with_a_cancelled_token_when_cancelled() {
        let s = JobScheduler::new("test-cancel-wait").expect("scheduler");
        let (gate, gate_job) = block(&s);
        let h = s.submit(Priority::Export, None, |token| token.is_cancelled());
        std::thread::sleep(Duration::from_millis(30));
        h.cancel();
        // The job is called, sees its token, and reports it, though the foreground is still busy.
        assert_eq!(h.wait(), Ok(true));
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        // The wait itself reports the cancellation to a job that polls it.
        let token = CancelToken::new();
        let s = Arc::new(s);
        let (up_tx, up_rx) = mpsc::channel::<()>();
        let (down_tx, down_rx) = mpsc::channel::<()>();
        let (gate2, gate_job2) = block(&s);
        let waiter = {
            let (s2, t) = (Arc::clone(&s), token.clone());
            std::thread::spawn(move || {
                up_tx.send(()).expect("up");
                let r = s2.wait_for_foreground(&t);
                down_tx.send(()).expect("down");
                r
            })
        };
        up_rx.recv_timeout(Duration::from_secs(5)).expect("up");
        std::thread::sleep(Duration::from_millis(20));
        assert!(down_rx.try_recv().is_err(), "still waiting");
        token.cancel();
        down_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the wait ends when the token fires");
        assert_eq!(waiter.join().expect("thread"), Err(Cancelled));
        gate2.send(()).expect("open the gate");
        gate_job2.wait().expect("gate job");
    }

    #[test]
    fn the_ledger_refuses_what_does_not_fit_and_releases_on_drop() {
        let l = MemoryLedger::new(100);
        let a = l.try_reserve(60).expect("fits");
        assert_eq!(l.used(), 60);
        let err = l.try_reserve(60).expect_err("over budget");
        assert_eq!((err.requested, err.used, err.budget), (60, 60, 100));
        let b = l.try_reserve(40).expect("exactly fits");
        assert_eq!(l.used(), 100);
        drop(a);
        assert_eq!(l.used(), 40);
        drop(b);
        assert_eq!((l.used(), l.peak()), (0, 100));
    }

    #[test]
    fn the_lru_evicts_oldest_first_and_always_keeps_the_newest() {
        let mut c: ByteLru<u32, &str> = ByteLru::new(100);
        c.insert(1, "a", 40);
        c.insert(2, "b", 40);
        assert_eq!(c.get(&1), Some(&"a")); // 1 is now the most recent
        c.insert(3, "c", 40); // over budget: evicts 2
        assert!(c.contains(&1) && c.contains(&3) && !c.contains(&2));
        c.insert(4, "huge", 500); // alone over budget: everything older goes
        assert_eq!((c.len(), c.used()), (1, 500));
        c.insert(4, "again", 10); // replacing frees the old size
        assert_eq!(c.used(), 10);
    }

    #[test]
    fn a_superseded_able_job_dropped_unrun_lets_go_of_what_it_holds_whoever_cancelled_it() {
        // A queued panel request holds its photo's decoded mosaic. Navigation lets go of it at once
        // (tested with the renderer); closing the session, shutting down and cancelling the handle
        // drop the job unrun as well, and the scheduler's own list of such jobs must not keep what
        // the job captured.
        let s = JobScheduler::new("test-unrun").expect("scheduler");
        s.navigate(PhotoId(1));
        let (gate, gate_job) = block(&s);
        let held = Arc::new(());
        let queue = |s: &JobScheduler| {
            let held = Arc::clone(&held);
            s.submit_superseded(
                Priority::Thumbnail,
                PhotoId(1),
                move |_| Arc::strong_count(&held),
                || 0,
            )
        };

        // The session closes: the queue drops every pending job.
        let closed = queue(&s);
        assert_eq!(Arc::strong_count(&held), 2);
        s.cancel_foreground();
        assert_eq!(closed.wait(), Err(JobError::Cancelled));
        assert_eq!(
            Arc::strong_count(&held),
            1,
            "a job dropped by the session closing let go"
        );
        assert!(
            locked(&s.superseded)
                .iter()
                .all(|e| e.finished.load(Ordering::SeqCst)),
            "and its entry is over, so that the next call prunes it"
        );

        // The caller gives up on the handle: the job is dropped when the worker reaches it.
        let given_up = queue(&s);
        given_up.cancel();
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        assert_eq!(given_up.wait(), Err(JobError::Cancelled));
        // The worker answers the handle first and drops the job after, on its own thread.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&held) != 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(Arc::strong_count(&held), 1, "a job cancelled by its handle");

        // The scheduler shuts down with a job waiting (the shutdown joins the worker, which is
        // held by the gate until another thread opens it).
        let (gate, gate_job) = block(&s);
        let shut = queue(&s);
        let opener = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            gate.send(()).expect("open the gate");
        });
        s.shutdown();
        opener.join().expect("opener");
        gate_job.wait().expect("gate job");
        assert_eq!(shut.wait(), Err(JobError::Shutdown));
        assert_eq!(Arc::strong_count(&held), 1, "a job dropped by shutdown");
    }

    #[test]
    fn the_list_of_superseded_able_jobs_does_not_grow_with_jobs_that_are_over() {
        let s = JobScheduler::new("test-list").expect("scheduler");
        s.navigate(PhotoId(1));
        for _ in 0..50 {
            s.submit_superseded(Priority::Thumbnail, PhotoId(1), |_| 1, || 0)
                .wait()
                .expect("job");
        }
        // One more call prunes the finished ones; a job queued behind a gate is the only live one.
        let (gate, gate_job) = block(&s);
        let waiting = s.submit_superseded(Priority::Thumbnail, PhotoId(1), |_| 1, || 0);
        assert!(
            locked(&s.superseded).len() <= 2,
            "{} entries",
            locked(&s.superseded).len()
        );
        gate.send(()).expect("open the gate");
        gate_job.wait().expect("gate job");
        waiting.wait().expect("job");
    }

    #[test]
    fn the_budget_follows_the_spec_defaults() {
        let b = MemoryBudget::default();
        assert_eq!(b.max_mosaics, 3);
        assert_eq!(b.cpu_bytes, 6 << 30);
        assert_eq!(b.gpu_bytes, 4 << 30);
        let d = MemoryBudget::detect();
        assert!(d.cpu_bytes <= b.cpu_bytes && d.cpu_bytes >= MIN_CPU_BUDGET_BYTES);
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn a_gpu_with_its_own_memory_leaves_the_two_budgets_apart() {
        // A discrete adapter (or none): `detect` as it always was.
        for (ram, cpu) in [(8, 4), (16, 6), (64, 6)] {
            let b = MemoryBudget::for_ram(Some(ram * GIB));
            assert_eq!(b.cpu_bytes, cpu * GIB, "{ram} GB");
            assert_eq!(b.gpu_bytes, 4 * GIB, "{ram} GB");
        }
        // Unknown RAM: the defaults, and the unified rule changes nothing without a RAM figure.
        assert_eq!(MemoryBudget::for_ram(None), MemoryBudget::default());
        assert_eq!(
            MemoryBudget::default().for_unified_memory(None),
            MemoryBudget::default()
        );
    }

    #[test]
    fn a_gpu_that_shares_the_ram_takes_a_quarter_of_it_and_the_cpu_keeps_what_the_bound_leaves() {
        // 8, 16 and 64 GB unified machines (docs/17-gpu.md section 8): the CPU budget is
        // min(its own, 0.6 x RAM - GPU), not its own minus the whole GPU amount (which cut every
        // Mac to 2 GiB).
        for (ram, gpu, cpu) in [
            (8, 2 * GIB, 8 * GIB * 3 / 5 - 2 * GIB),
            (16, 4 * GIB, 16 * GIB * 3 / 5 - 4 * GIB),
            (64, 4 * GIB, 6 * GIB),
        ] {
            let b = MemoryBudget::for_ram(Some(ram * GIB)).for_unified_memory(Some(ram * GIB));
            assert_eq!(b.gpu_bytes, gpu, "{ram} GB: GPU");
            assert_eq!(b.cpu_bytes, cpu, "{ram} GB: CPU");
            assert_eq!(b.max_mosaics, MemoryBudget::default().max_mosaics);
            assert_eq!(b.max_photos, MemoryBudget::default().max_photos);
        }
    }

    #[test]
    fn on_a_unified_machine_cpu_and_gpu_together_stay_under_six_tenths_of_the_ram() {
        let (num, den) = UNIFIED_TOTAL_RAM_SHARE;
        assert_eq!((num, den), (3, 5), "the bound is 0.6");
        // Every RAM size from 4 GiB to 1 TiB in steps of 1 GiB, and some that are not round.
        let sizes = (4..=1024u64).map(|g| g * GIB).chain([
            (6 * GIB) + (123 << 20),
            (12 * GIB) - 1,
            24 * GIB + 7,
        ]);
        for ram in sizes {
            let b = MemoryBudget::for_ram(Some(ram)).for_unified_memory(Some(ram));
            assert!(
                (b.cpu_bytes + b.gpu_bytes) * den <= ram * num,
                "{ram} bytes of RAM: CPU {} + GPU {} is over {num}/{den}",
                b.cpu_bytes,
                b.gpu_bytes
            );
            assert!(b.gpu_bytes >= UNIFIED_GPU_MIN_BYTES && b.gpu_bytes <= UNIFIED_GPU_MAX_BYTES);
            assert!(b.cpu_bytes >= MIN_CPU_BUDGET_BYTES);
            // The CPU gives up only what the bound makes it: it keeps its own budget when that
            // fits, and otherwise takes exactly what 0.6 of the RAM leaves after the GPU.
            let own = MemoryBudget::for_ram(Some(ram)).cpu_bytes;
            assert!(b.cpu_bytes <= own, "{ram} bytes of RAM");
            assert!(
                b.cpu_bytes == own || b.cpu_bytes == ram * num / den - b.gpu_bytes,
                "{ram} bytes of RAM: CPU {} is neither its own {own} nor the bound's remainder",
                b.cpu_bytes
            );
        }
    }

    #[test]
    fn a_unified_mac_with_plenty_of_ram_keeps_most_of_its_cpu_budget() {
        // The literal rule (lowering the CPU budget by the whole GPU amount) left every Apple-silicon
        // Mac 2 GiB of CPU budget while only the colour pass used the device.
        for ram in [16u64, 24, 32, 64, 128] {
            let b = MemoryBudget::for_ram(Some(ram * GIB)).for_unified_memory(Some(ram * GIB));
            assert!(b.cpu_bytes > 2 * GIB + GIB / 2, "{ram} GB: {}", b.cpu_bytes);
        }
        let b = MemoryBudget::for_ram(Some(24 * GIB)).for_unified_memory(Some(24 * GIB));
        assert_eq!(
            b.cpu_bytes,
            6 * GIB,
            "24 GB and up: the CPU default is untouched"
        );
    }

    #[test]
    fn the_unified_gpu_budget_has_a_floor_and_a_ceiling() {
        let tiny = MemoryBudget::for_ram(Some(GIB)).for_unified_memory(Some(GIB));
        assert_eq!(tiny.gpu_bytes, UNIFIED_GPU_MIN_BYTES, "1 GB: the floor");
        assert_eq!(
            tiny.cpu_bytes, MIN_CPU_BUDGET_BYTES,
            "and the CPU keeps its own floor"
        );
        let huge = MemoryBudget::for_ram(Some(1024 * GIB)).for_unified_memory(Some(1024 * GIB));
        assert_eq!(huge.gpu_bytes, UNIFIED_GPU_MAX_BYTES, "1 TB: the ceiling");
    }

    #[test]
    fn the_ledger_follows_a_new_budget() {
        let ledger = MemoryLedger::new(100);
        let first = ledger.try_reserve(60).expect("fits");
        ledger.set_budget(50);
        assert_eq!(ledger.budget(), 50);
        let err = ledger.try_reserve(1).expect_err("over the lowered budget");
        assert_eq!((err.used, err.budget), (60, 50));
        drop(first);
        assert_eq!(ledger.used(), 0, "what was reserved is released as before");
        ledger.try_reserve(50).expect("fits the new budget");
    }
}
