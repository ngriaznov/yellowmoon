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
//!   cancels the viewport and tile jobs of the photo being left, pending and running; thumbnail and
//!   export jobs are never cancelled by navigation. An export stops only through its own handle.
//! * **Memory.** At most [`MemoryBudget::max_mosaics`] decoded mosaics (the current photo and its
//!   neighbours, kept by the session) and [`MemoryBudget::max_photos`] render graphs. The render
//!   caches are byte-bounded LRU maps ([`ByteLru`]); a full-resolution render reserves its working
//!   set in the [`MemoryLedger`] first and is refused, rather than started, when the budget would be
//!   exceeded. Budget defaults follow `docs/01`: 6 GiB of CPU memory scaled down to half the
//!   machine's RAM, 4 GiB of GPU memory (the caller gives the GPU context its own budget, see
//!   [`MemoryBudget::gpu_bytes`]).
//!
//! The queues are `ym_gpu::queue::Scheduler`, the same one that fronts the GPU thread, so the
//! ordering and cancellation rules are the ones its tests pin; this module adds the photo-aware
//! navigation rule, the export's yielding and the memory accounting. Each job is data-parallel
//! inside (rayon, scoped threads), so a second *foreground* worker would only compete for the same
//! cores.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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

impl MemoryBudget {
    /// The defaults scaled down to the machine: the CPU budget is at most half of the installed RAM.
    #[must_use]
    pub fn detect() -> Self {
        let mut b = Self::default();
        if let Some(total) = total_ram_bytes() {
            b.cpu_bytes = b.cpu_bytes.min(total / 2).max(MIN_CPU_BUDGET_BYTES);
        }
        b
    }

    /// The same budget with another CPU limit.
    #[must_use]
    pub fn with_cpu_bytes(mut self, bytes: u64) -> Self {
        self.cpu_bytes = bytes;
        self
    }
}

/// Installed RAM in bytes, where the operating system says (Linux `/proc/meminfo`, macOS `sysctl`).
fn total_ram_bytes() -> Option<u64> {
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
    budget: u64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl MemoryLedger {
    /// A ledger for `budget` bytes.
    #[must_use]
    pub fn new(budget: u64) -> Arc<Self> {
        Arc::new(Self {
            budget,
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
            if next > self.budget {
                return Err(OverBudget {
                    requested: bytes,
                    used: cur,
                    budget: self.budget,
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
        self.budget
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
    /// they left. Returns how many jobs were cancelled. Thumbnail and export jobs are untouched.
    pub fn navigate(&self, photo: PhotoId) -> usize {
        let new = u64::from(photo.0).max(1);
        let old = self.current.swap(new, Ordering::SeqCst);
        if old == 0 || old == new {
            return 0;
        }
        self.foreground.cancel_navigation(Some(old))
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
    fn the_budget_follows_the_spec_defaults() {
        let b = MemoryBudget::default();
        assert_eq!(b.max_mosaics, 3);
        assert_eq!(b.cpu_bytes, 6 << 30);
        assert_eq!(b.gpu_bytes, 4 << 30);
        let d = MemoryBudget::detect();
        assert!(d.cpu_bytes <= b.cpu_bytes && d.cpu_bytes >= MIN_CPU_BUDGET_BYTES);
    }
}
