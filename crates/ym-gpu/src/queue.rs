//! The priority queue in front of the device thread (`docs/01-architecture.md` section Scheduling
//! and threads): viewport frames > visible-tile neural passes > thumbnails > export, with
//! cancellation tokens.
//!
//! [`Scheduler`] owns one worker thread. The worker creates a context `C` (for the GPU: the wgpu
//! device, queue and pools) on its own thread, then runs submitted closures against `&mut C` one at a
//! time. The queue does not know about wgpu, so ordering, cancellation and shutdown are tested here
//! without a GPU.
//!
//! # Rules
//!
//! * **Order.** The highest [`Priority`] runs first; equal priorities run in submission order.
//!   A running job is never pre-empted (it may poll its [`CancelToken`] between tiles), so work
//!   that must not hold up the viewport (an export) is cut into jobs, one per tile row or per tile
//!   ([`TilePlan::tile_rows`](crate::tiles::TilePlan::tile_rows)); the queue then lets a viewport
//!   job in between them.
//! * **Cancellation.** Every job has a [`CancelToken`]. Cancelling a job that has not started
//!   resolves its handle with [`JobError::Cancelled`] and the closure never runs: the worker drops
//!   every cancelled job still in the queue before it runs the next one, so what such a closure holds
//!   (the device buffers of a cancelled render) is let go of then, not when the job's own turn would
//!   have come. Cancelling a running job only sets the flag; the closure decides when to stop.
//!   [`Scheduler::cancel_navigation`] (moving to another photo) cancels the pending and running
//!   *viewport* and *tile* jobs of that photo; thumbnail and export jobs are never cancelled by
//!   navigation. An export is cancelled only through its own token (the user's "cancel").
//! * **Shutdown.** [`Scheduler::shutdown`] lets the running job finish, resolves every pending job
//!   with [`JobError::Shutdown`], stops the thread and joins it. Submitting afterwards resolves
//!   immediately with `Shutdown`. Dropping the scheduler does the same. No lock is held while a job
//!   runs or while a result is sent, so a job may submit further jobs; waiting for a job from inside
//!   a job on the same scheduler would deadlock (the one thread that could run it is the one
//!   waiting), so [`JobHandle::wait`] on the worker thread returns [`JobError::WouldDeadlock`]
//!   instead of hanging unless the result is already there.
//! * **Panics** in a job are caught: the handle gets [`JobError::Panicked`] and the worker carries
//!   on.

use std::any::Any;
use std::collections::BinaryHeap;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;

/// Job priority; a higher value runs first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Export: full-resolution frames for a file on disk. Never cancelled by navigation.
    Export = 0,
    /// Thumbnails after open and on look changes. Not cancelled by navigation.
    Thumbnail = 1,
    /// Neural demosaic passes for the visible tiles at 100 %.
    TileNeural = 2,
    /// Viewport frames: the picture the user is looking at.
    Viewport = 3,
}

impl Priority {
    /// Whether moving to another photo cancels jobs of this priority.
    pub const fn cancelled_by_navigation(self) -> bool {
        matches!(self, Priority::Viewport | Priority::TileNeural)
    }
}

/// Describes a job: its priority and the photo it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobSpec {
    /// Queue priority.
    pub priority: Priority,
    /// The owner (a photo id) used by [`Scheduler::cancel_navigation`]; `None` matches only a
    /// navigation that cancels every owner.
    pub owner: Option<u64>,
}

impl JobSpec {
    /// A job of `priority` with no owner.
    pub const fn new(priority: Priority) -> Self {
        Self {
            priority,
            owner: None,
        }
    }

    /// The same job belonging to `owner`.
    #[must_use]
    pub const fn owned_by(mut self, owner: u64) -> Self {
        self.owner = Some(owner);
        self
    }
}

/// A flag a job polls to learn it was cancelled. Cloning shares the flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

/// The error of [`CancelToken::check`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

impl CancelToken {
    /// A fresh, not-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancels the job.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether the job was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// `Err(Cancelled)` once cancelled, for use with `?` between tiles.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when the token was cancelled.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Why a job produced no result.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum JobError {
    /// The job was cancelled before it ran.
    #[error("the job was cancelled")]
    Cancelled,
    /// The scheduler shut down before the job ran.
    #[error("the scheduler has shut down")]
    Shutdown,
    /// The job panicked; the text is the panic message.
    #[error("the job panicked: {0}")]
    Panicked(String),
    /// A job on the worker thread waited for a job that had not finished, which would never end
    /// because the worker is the thread that has to run it. Submit and return instead.
    #[error("a job waited for another job from the worker thread, which would never finish")]
    WouldDeadlock,
}

/// What the worker tells a queued job to do.
enum Dispatch<'a, C> {
    Run(&'a mut C),
    Cancelled,
    Shutdown,
}

type Job<C> = Box<dyn for<'a> FnOnce(Dispatch<'a, C>, &CancelToken) + Send>;

struct Entry<C> {
    priority: Priority,
    seq: u64,
    owner: Option<u64>,
    token: CancelToken,
    job: Job<C>,
}

impl<C> PartialEq for Entry<C> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}
impl<C> Eq for Entry<C> {}
impl<C> PartialOrd for Entry<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<C> Ord for Entry<C> {
    /// Max-heap order: higher priority first, then the earlier submission.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct Running {
    priority: Priority,
    owner: Option<u64>,
    token: CancelToken,
}

struct State<C> {
    heap: BinaryHeap<Entry<C>>,
    next_seq: u64,
    shutdown: bool,
    running: Option<Running>,
}

struct Inner<C> {
    state: Mutex<State<C>>,
    wake: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The result of a submitted job, to wait on.
pub struct JobHandle<R> {
    rx: mpsc::Receiver<Result<R, JobError>>,
    token: CancelToken,
    /// The thread that runs the jobs, to refuse a wait that cannot end.
    worker: ThreadId,
}

impl<R> fmt::Debug for JobHandle<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobHandle")
            .field("cancelled", &self.token.is_cancelled())
            .finish()
    }
}

impl<R> JobHandle<R> {
    /// Cancels the job (see the module rules).
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// The job's cancellation token.
    pub fn token(&self) -> &CancelToken {
        &self.token
    }

    /// Waits for the result.
    ///
    /// # Errors
    ///
    /// [`JobError`] when the job was cancelled before it ran, the scheduler shut down, or the job
    /// panicked; [`JobError::WouldDeadlock`] when called on the worker thread for a job that has
    /// not finished.
    pub fn wait(self) -> Result<R, JobError> {
        if thread::current().id() == self.worker {
            // Only a result that is already there can be returned; blocking would wait for a job
            // the blocked thread itself has to run.
            return match self.rx.try_recv() {
                Ok(r) => r,
                Err(mpsc::TryRecvError::Empty) => Err(JobError::WouldDeadlock),
                Err(mpsc::TryRecvError::Disconnected) => Err(JobError::Shutdown),
            };
        }
        self.rx.recv().unwrap_or(Err(JobError::Shutdown))
    }

    /// Waits up to `timeout`; `None` if the job has not finished by then.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Result<R, JobError>> {
        match self.rx.recv_timeout(timeout) {
            Ok(r) => Some(r),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => Some(Err(JobError::Shutdown)),
        }
    }

    /// The result if the job has already finished.
    pub fn try_wait(&self) -> Option<Result<R, JobError>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(JobError::Shutdown)),
        }
    }
}

/// Why [`Scheduler::spawn`] failed.
#[derive(Debug)]
pub enum SpawnError<E> {
    /// The context constructor failed.
    Init(E),
    /// The operating system refused to start the thread.
    Thread(std::io::Error),
}

/// One worker thread owning a context `C`, fed by a priority queue.
pub struct Scheduler<C: 'static> {
    inner: Arc<Inner<C>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    worker_id: ThreadId,
}

impl<C: 'static> fmt::Debug for Scheduler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scheduler")
            .field("pending", &self.pending())
            .field("shut_down", &self.is_shut_down())
            .finish()
    }
}

fn panic_message(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

impl<C: 'static> Scheduler<C> {
    /// Starts the worker thread. `init` runs **on** the new thread and builds the context (and a
    /// value `T` handed back to the caller); the context never leaves that thread.
    ///
    /// # Errors
    ///
    /// [`SpawnError::Init`] with `init`'s error, or [`SpawnError::Thread`] if the thread cannot start.
    pub fn spawn<T, E, F>(name: &str, init: F) -> Result<(Scheduler<C>, T), SpawnError<E>>
    where
        F: FnOnce() -> Result<(C, T), E> + Send + 'static,
        T: Send + 'static,
        E: Send + 'static,
    {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                heap: BinaryHeap::new(),
                next_seq: 0,
                shutdown: false,
                running: None,
            }),
            wake: Condvar::new(),
        });
        let (ready_tx, ready_rx) = mpsc::channel::<Result<T, E>>();
        let worker_inner = Arc::clone(&inner);
        let handle = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let (mut ctx, extra) = match init() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(extra));
                worker_loop(&worker_inner, &mut ctx);
                // `ctx` is dropped here, on the thread that created it.
            })
            .map_err(SpawnError::Thread)?;
        let worker_id = handle.thread().id();
        match ready_rx.recv() {
            Ok(Ok(extra)) => Ok((
                Scheduler {
                    inner,
                    thread: Mutex::new(Some(handle)),
                    worker_id,
                },
                extra,
            )),
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(SpawnError::Init(e))
            }
            Err(_) => {
                let _ = handle.join();
                Err(SpawnError::Thread(std::io::Error::other(
                    "the worker thread ended during start-up",
                )))
            }
        }
    }

    /// Queues `job` with a fresh [`CancelToken`]. See the module rules for ordering and
    /// cancellation.
    pub fn submit<R, F>(&self, spec: JobSpec, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut C, &CancelToken) -> R + Send + 'static,
    {
        self.submit_with_token(spec, CancelToken::new(), job)
    }

    /// Queues `job` with a token the caller already owns, normally the render's own
    /// ([`JobSpec`] stays `Copy`, so the token is a separate argument).
    ///
    /// Every job of one render, one export or one navigation can share a single token: setting it
    /// skips every job of the group that has not started (the worker finds the token set and
    /// resolves the job with [`JobError::Cancelled`] without running it), and a job that is running
    /// sees the flag through the `&CancelToken` it is given, at its next check. The shared token is
    /// the one [`JobHandle::token`] returns, so [`JobHandle::cancel`] on one handle of the group
    /// cancels the whole group, and [`Scheduler::cancel_navigation`] and [`Scheduler::cancel_all`]
    /// set it for every job they reach (a viewport or tile job of a navigation, any job for
    /// `cancel_all`). Use a fresh token per independent job ([`Scheduler::submit`]) when that is
    /// not wanted.
    pub fn submit_with_token<R, F>(&self, spec: JobSpec, token: CancelToken, job: F) -> JobHandle<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut C, &CancelToken) -> R + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel::<Result<R, JobError>>(1);
        let boxed: Job<C> = Box::new(move |dispatch, token| match dispatch {
            Dispatch::Run(ctx) => {
                let r = catch_unwind(AssertUnwindSafe(|| job(ctx, token)));
                let _ = tx.send(r.map_err(|p| JobError::Panicked(panic_message(p))));
            }
            Dispatch::Cancelled => {
                let _ = tx.send(Err(JobError::Cancelled));
            }
            Dispatch::Shutdown => {
                let _ = tx.send(Err(JobError::Shutdown));
            }
        });
        let handle = JobHandle {
            rx,
            token: token.clone(),
            worker: self.worker_id,
        };
        let rejected = {
            let mut st = lock(&self.inner.state);
            if st.shutdown {
                Some(boxed)
            } else {
                let seq = st.next_seq;
                st.next_seq += 1;
                st.heap.push(Entry {
                    priority: spec.priority,
                    seq,
                    owner: spec.owner,
                    token: token.clone(),
                    job: boxed,
                });
                None
            }
        };
        match rejected {
            Some(job) => job(Dispatch::Shutdown, &token),
            None => self.inner.wake.notify_one(),
        }
        handle
    }

    /// Cancels the viewport and tile jobs of `owner` (or of every owner when `None`), pending and
    /// running. Pending ones resolve at once with [`JobError::Cancelled`]; a running one only has its
    /// token set. Thumbnail and export jobs are untouched. Returns how many jobs were cancelled.
    pub fn cancel_navigation(&self, owner: Option<u64>) -> usize {
        let matches = |p: Priority, o: Option<u64>| {
            p.cancelled_by_navigation() && (owner.is_none() || o == owner)
        };
        let (cancelled, running) = {
            let mut st = lock(&self.inner.state);
            let entries = std::mem::take(&mut st.heap).into_vec();
            let (hit, keep): (Vec<_>, Vec<_>) = entries
                .into_iter()
                .partition(|e| matches(e.priority, e.owner));
            st.heap = keep.into_iter().collect();
            let running = match &st.running {
                Some(r) if matches(r.priority, r.owner) => {
                    r.token.cancel();
                    1
                }
                _ => 0,
            };
            (hit, running)
        };
        let n = cancelled.len() + running;
        for e in cancelled {
            e.token.cancel();
            (e.job)(Dispatch::Cancelled, &e.token);
        }
        n
    }

    /// Cancels the jobs of `owners` that are not export jobs, pending and running: the viewport, tile
    /// and thumbnail jobs of photos that left the session. Pending ones resolve at once with
    /// [`JobError::Cancelled`]; a running one only has its token set. Export jobs are left to their
    /// own token (an export holds its own snapshots and finishes the photos it was started with).
    /// Returns how many jobs were cancelled.
    pub fn cancel_owners(&self, owners: &[u64]) -> usize {
        let matches = |p: Priority, o: Option<u64>| {
            p != Priority::Export && o.is_some_and(|o| owners.contains(&o))
        };
        let (cancelled, running) = {
            let mut st = lock(&self.inner.state);
            let entries = std::mem::take(&mut st.heap).into_vec();
            let (hit, keep): (Vec<_>, Vec<_>) = entries
                .into_iter()
                .partition(|e| matches(e.priority, e.owner));
            st.heap = keep.into_iter().collect();
            let running = match &st.running {
                Some(r) if matches(r.priority, r.owner) => {
                    r.token.cancel();
                    1
                }
                _ => 0,
            };
            (hit, running)
        };
        let n = cancelled.len() + running;
        for e in cancelled {
            e.token.cancel();
            (e.job)(Dispatch::Cancelled, &e.token);
        }
        n
    }

    /// Cancels every pending job of any priority (running jobs get their token set). Used when the
    /// session closes; navigation uses [`Scheduler::cancel_navigation`].
    pub fn cancel_all(&self) -> usize {
        let (cancelled, running) = {
            let mut st = lock(&self.inner.state);
            let entries = std::mem::take(&mut st.heap).into_vec();
            let running = match &st.running {
                Some(r) => {
                    r.token.cancel();
                    1
                }
                None => 0,
            };
            (entries, running)
        };
        let n = cancelled.len() + running;
        for e in cancelled {
            e.token.cancel();
            (e.job)(Dispatch::Cancelled, &e.token);
        }
        n
    }

    /// Number of jobs waiting (not counting the running one).
    pub fn pending(&self) -> usize {
        lock(&self.inner.state).heap.len()
    }

    /// Whether [`shutdown`](Self::shutdown) has been called.
    pub fn is_shut_down(&self) -> bool {
        lock(&self.inner.state).shutdown
    }

    /// Stops accepting jobs, resolves the pending ones with [`JobError::Shutdown`], lets the running
    /// job finish and joins the worker. Idempotent. Called from the worker thread itself (a job that
    /// owns the last handle) it does not join.
    pub fn shutdown(&self) {
        let (pending, running_token) = {
            let mut st = lock(&self.inner.state);
            st.shutdown = true;
            let pending = std::mem::take(&mut st.heap).into_vec();
            (pending, st.running.as_ref().map(|r| r.token.clone()))
        };
        if let Some(t) = running_token {
            t.cancel();
        }
        self.inner.wake.notify_all();
        for e in pending {
            e.token.cancel();
            (e.job)(Dispatch::Shutdown, &e.token);
        }
        if thread::current().id() == self.worker_id {
            return;
        }
        let handle = lock(&self.thread).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl<C: 'static> Drop for Scheduler<C> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Takes the jobs whose token is set out of the queue, leaving the others in their order.
fn take_cancelled<C>(heap: &mut BinaryHeap<Entry<C>>) -> Vec<Entry<C>> {
    if !heap.iter().any(|e| e.token.is_cancelled()) {
        return Vec::new();
    }
    let (cancelled, kept): (Vec<_>, Vec<_>) = std::mem::take(heap)
        .into_vec()
        .into_iter()
        .partition(|e| e.token.is_cancelled());
    *heap = BinaryHeap::from(kept);
    cancelled
}

fn worker_loop<C>(inner: &Inner<C>, ctx: &mut C) {
    loop {
        // Jobs cancelled before they started are resolved, and their closures dropped, before
        // anything else runs (outside the lock, like every dispatch).
        let cancelled = take_cancelled(&mut lock(&inner.state).heap);
        for e in cancelled {
            let token = e.token.clone();
            (e.job)(Dispatch::Cancelled, &token);
        }
        let (entry, shutting_down) = {
            let mut st = lock(&inner.state);
            loop {
                if let Some(e) = st.heap.pop() {
                    st.running = Some(Running {
                        priority: e.priority,
                        owner: e.owner,
                        token: e.token.clone(),
                    });
                    break (e, st.shutdown);
                }
                if st.shutdown {
                    st.running = None;
                    return;
                }
                st = inner.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let token = entry.token.clone();
        if shutting_down {
            (entry.job)(Dispatch::Shutdown, &token);
        } else if token.is_cancelled() {
            (entry.job)(Dispatch::Cancelled, &token);
        } else {
            (entry.job)(Dispatch::Run(ctx), &token);
        }
        lock(&inner.state).running = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A scheduler whose context is a log of what ran.
    fn scheduler() -> Scheduler<Vec<&'static str>> {
        Scheduler::spawn("test-worker", || Ok::<_, ()>((Vec::new(), ())))
            .expect("spawn")
            .0
    }

    /// Blocks the worker until the returned sender is used (or dropped), so the queue can be filled.
    fn gate(s: &Scheduler<Vec<&'static str>>) -> (mpsc::Sender<()>, JobHandle<()>) {
        let (open, wait) = mpsc::channel::<()>();
        let started = mpsc::channel::<()>();
        let started_tx = started.0;
        let h = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            let _ = started_tx.send(());
            let _ = wait.recv();
        });
        started
            .1
            .recv_timeout(Duration::from_secs(10))
            .expect("gate started");
        (open, h)
    }

    const WAIT: Duration = Duration::from_secs(10);

    #[test]
    fn runs_by_priority_then_submission_order() {
        let s = scheduler();
        let (open, gate_job) = gate(&s);
        let mut handles = Vec::new();
        for (name, p) in [
            ("export1", Priority::Export),
            ("thumb1", Priority::Thumbnail),
            ("view1", Priority::Viewport),
            ("tile1", Priority::TileNeural),
            ("view2", Priority::Viewport),
            ("export2", Priority::Export),
            ("tile2", Priority::TileNeural),
            ("thumb2", Priority::Thumbnail),
        ] {
            handles.push(
                s.submit(JobSpec::new(p), move |log: &mut Vec<&'static str>, _| {
                    log.push(name);
                }),
            );
        }
        assert_eq!(s.pending(), 8);
        open.send(()).expect("open gate");
        gate_job.wait().expect("gate job");
        for h in handles {
            h.wait().expect("job");
        }
        let log = s
            .submit(JobSpec::new(Priority::Export), |log, _| log.clone())
            .wait()
            .expect("log");
        assert_eq!(
            log,
            [
                "view1", "view2", "tile1", "tile2", "thumb1", "thumb2", "export1", "export2"
            ]
        );
    }

    #[test]
    fn a_cancelled_pending_job_never_runs() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let ran = Arc::new(AtomicUsize::new(0));
        let r2 = Arc::clone(&ran);
        let victim = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            r2.fetch_add(1, Ordering::SeqCst);
        });
        victim.cancel();
        let r3 = Arc::clone(&ran);
        let survivor = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            r3.fetch_add(10, Ordering::SeqCst);
        });
        open.send(()).expect("open");
        g.wait().expect("gate");
        assert_eq!(victim.wait(), Err(JobError::Cancelled));
        survivor.wait().expect("survivor");
        assert_eq!(ran.load(Ordering::SeqCst), 10);
    }

    /// Sets its flag when dropped: stands for a buffer that a job's closure holds.
    struct Held(Arc<AtomicBool>);

    impl Drop for Held {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_cancelled_job_lets_go_of_what_it_holds_before_a_higher_priority_job_runs() {
        // A cancelled export job that is still queued under a viewport job is dropped first, so
        // what its closure captured (the device buffers of a cancelled render) is released before
        // the viewport job runs, not when the export job's own turn comes.
        let s = scheduler();
        let (open, g) = gate(&s);
        let dropped = Arc::new(AtomicBool::new(false));
        let held = Held(Arc::clone(&dropped));
        let export = s.submit(JobSpec::new(Priority::Export), move |_, _| {
            let _keep = &held;
        });
        export.cancel();
        let seen = Arc::clone(&dropped);
        let viewport = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            seen.load(Ordering::SeqCst)
        });
        open.send(()).expect("open");
        g.wait().expect("gate");
        assert!(
            viewport.wait().expect("viewport"),
            "the cancelled job still held its closure when the viewport job ran"
        );
        assert_eq!(export.wait(), Err(JobError::Cancelled));
    }

    #[test]
    fn navigation_cancels_viewport_and_tile_jobs_of_that_owner_only() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let ran = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let mk = |name: &'static str, p: Priority, owner: u64| {
            let ran = Arc::clone(&ran);
            s.submit(JobSpec::new(p).owned_by(owner), move |_, _| {
                lock(&ran).push(name);
            })
        };
        let view_a = mk("view_a", Priority::Viewport, 1);
        let tile_a = mk("tile_a", Priority::TileNeural, 1);
        let thumb_a = mk("thumb_a", Priority::Thumbnail, 1);
        let export_a = mk("export_a", Priority::Export, 1);
        let view_b = mk("view_b", Priority::Viewport, 2);
        let n = s.cancel_navigation(Some(1));
        assert_eq!(n, 2, "only the pending viewport and tile jobs of photo 1");
        // Resolved at once, without waiting for the gate.
        assert_eq!(view_a.wait_timeout(WAIT), Some(Err(JobError::Cancelled)));
        assert_eq!(tile_a.wait_timeout(WAIT), Some(Err(JobError::Cancelled)));
        assert_eq!(s.pending(), 3);
        open.send(()).expect("open");
        g.wait().expect("gate");
        for h in [thumb_a, export_a, view_b] {
            h.wait().expect("kept job");
        }
        assert_eq!(*lock(&ran), ["view_b", "thumb_a", "export_a"]);
    }

    #[test]
    fn cancelling_owners_takes_every_job_but_exports_of_those_owners_only() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let ran = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let mk = |name: &'static str, p: Priority, owner: Option<u64>| {
            let ran = Arc::clone(&ran);
            let spec = owner.map_or(JobSpec::new(p), |o| JobSpec::new(p).owned_by(o));
            s.submit(spec, move |_, _| {
                lock(&ran).push(name);
            })
        };
        let view_a = mk("view_a", Priority::Viewport, Some(1));
        let thumb_a = mk("thumb_a", Priority::Thumbnail, Some(1));
        let export_a = mk("export_a", Priority::Export, Some(1));
        let thumb_b = mk("thumb_b", Priority::Thumbnail, Some(2));
        let thumb_c = mk("thumb_c", Priority::Thumbnail, Some(3));
        let unowned = mk("unowned", Priority::Thumbnail, None);
        let n = s.cancel_owners(&[1, 2]);
        assert_eq!(n, 3, "the viewport and thumbnail jobs of photos 1 and 2");
        for h in [view_a, thumb_a, thumb_b] {
            assert_eq!(h.wait_timeout(WAIT), Some(Err(JobError::Cancelled)));
        }
        assert_eq!(
            s.pending(),
            3,
            "an export, another photo's job, an unowned job"
        );
        open.send(()).expect("open");
        g.wait().expect("gate");
        for h in [export_a, thumb_c, unowned] {
            h.wait().expect("kept job");
        }
        assert_eq!(*lock(&ran), ["thumb_c", "unowned", "export_a"]);
        assert_eq!(s.cancel_owners(&[]), 0);
    }

    #[test]
    fn cancelling_owners_sets_the_token_of_a_running_thumbnail_but_not_an_export() {
        let s = scheduler();
        for (priority, expect_cancelled) in [(Priority::Thumbnail, true), (Priority::Export, false)]
        {
            let (started_tx, started_rx) = mpsc::channel::<()>();
            let (go_tx, go_rx) = mpsc::channel::<()>();
            let h = s.submit(
                JobSpec::new(priority).owned_by(7),
                move |_, token: &CancelToken| {
                    let _ = started_tx.send(());
                    let _ = go_rx.recv();
                    token.is_cancelled()
                },
            );
            started_rx.recv_timeout(WAIT).expect("started");
            s.cancel_owners(&[7]);
            go_tx.send(()).expect("go");
            assert_eq!(h.wait().expect("result"), expect_cancelled, "{priority:?}");
        }
    }

    #[test]
    fn navigation_sets_the_token_of_a_running_viewport_job_but_not_an_export() {
        let s = scheduler();
        for (priority, expect_cancelled) in [(Priority::Viewport, true), (Priority::Export, false)]
        {
            let (started_tx, started_rx) = mpsc::channel::<()>();
            let (go_tx, go_rx) = mpsc::channel::<()>();
            let h = s.submit(
                JobSpec::new(priority).owned_by(7),
                move |_, token: &CancelToken| {
                    let _ = started_tx.send(());
                    let _ = go_rx.recv();
                    token.is_cancelled()
                },
            );
            started_rx.recv_timeout(WAIT).expect("started");
            s.cancel_navigation(Some(7));
            go_tx.send(()).expect("go");
            assert_eq!(h.wait().expect("result"), expect_cancelled, "{priority:?}");
        }
    }

    #[test]
    fn an_export_is_cancelled_only_through_its_own_token() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let export = s.submit(JobSpec::new(Priority::Export).owned_by(3), |_, _| ());
        // Navigation sets the token of the running viewport gate job and touches nothing else.
        assert_eq!(s.cancel_navigation(None), 1);
        assert_eq!(s.pending(), 1, "the export is still queued");
        assert!(!export.token().is_cancelled());
        export.cancel();
        open.send(()).expect("open");
        g.wait().expect("gate");
        assert_eq!(export.wait(), Err(JobError::Cancelled));
    }

    #[test]
    fn shutdown_resolves_pending_jobs_and_joins_without_deadlock() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let pending: Vec<_> = (0..5)
            .map(|_| s.submit(JobSpec::new(Priority::Export), |_, _| ()))
            .collect();
        // The running gate job must be allowed to finish for shutdown to join.
        let opener = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            let _ = open.send(());
        });
        s.shutdown();
        opener.join().expect("opener");
        g.wait().expect("the running job finished");
        for h in pending {
            assert_eq!(h.wait_timeout(WAIT), Some(Err(JobError::Shutdown)));
        }
        // After shutdown new jobs are refused immediately, and shutdown is idempotent.
        let late = s.submit(JobSpec::new(Priority::Viewport), |_, _| ());
        assert_eq!(late.wait_timeout(WAIT), Some(Err(JobError::Shutdown)));
        s.shutdown();
        assert!(s.is_shut_down());
    }

    #[test]
    fn dropping_a_busy_scheduler_does_not_deadlock() {
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let t = thread::spawn(move || {
            let s = scheduler();
            let _handles: Vec<_> = (0..50)
                .map(|i| {
                    s.submit(JobSpec::new(Priority::Thumbnail), move |_, _| {
                        thread::sleep(Duration::from_millis(1));
                        i
                    })
                })
                .collect();
            drop(s);
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(WAIT)
            .expect("drop returned within the timeout");
        t.join().expect("thread");
    }

    #[test]
    fn a_panicking_job_does_not_kill_the_worker() {
        let s = scheduler();
        let bad = s.submit(JobSpec::new(Priority::Viewport), |_, _| -> u32 {
            panic!("boom")
        });
        match bad.wait() {
            Err(JobError::Panicked(m)) => assert!(m.contains("boom"), "{m}"),
            other => panic!("expected a panic error, got {other:?}"),
        }
        let ok = s.submit(JobSpec::new(Priority::Viewport), |_, _| 5u32);
        assert_eq!(ok.wait(), Ok(5));
    }

    #[test]
    fn waiting_for_another_job_from_the_worker_thread_is_an_error_not_a_hang() {
        let s = Arc::new(scheduler());
        let s2 = Arc::clone(&s);
        let outer = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            // The inner job can only run once this one returns: waiting would never end.
            let inner = s2.submit(JobSpec::new(Priority::Export), |log, _| log.push("inner"));
            inner.wait()
        });
        assert_eq!(
            outer.wait_timeout(WAIT).expect("the outer job returned"),
            Ok(Err(JobError::WouldDeadlock))
        );
        // A result that is already there is still returned.
        let s3 = Arc::clone(&s);
        let outer = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            let inner = s3.submit(JobSpec::new(Priority::Export), |_, _| ());
            inner.cancel();
            s3.cancel_all();
            inner.wait()
        });
        assert_eq!(
            outer.wait_timeout(WAIT).expect("the outer job returned"),
            Ok(Err(JobError::Cancelled))
        );
    }

    #[test]
    fn a_job_can_submit_another_job() {
        let s = Arc::new(scheduler());
        let s2 = Arc::clone(&s);
        let outer = s.submit(JobSpec::new(Priority::Viewport), move |_, _| {
            // Submitting is fine; waiting here would deadlock the single worker.
            s2.submit(JobSpec::new(Priority::Export), |log, _| log.push("inner"))
        });
        let inner = outer.wait().expect("outer");
        inner.wait().expect("inner");
    }

    #[test]
    fn the_context_is_built_on_the_worker_thread_and_init_errors_come_back() {
        let main_id = thread::current().id();
        let (s, built_on) = Scheduler::<Vec<&'static str>>::spawn("ctx-thread", || {
            Ok::<_, ()>((Vec::new(), thread::current().id()))
        })
        .expect("spawn");
        assert_ne!(built_on, main_id);
        let ran_on = s
            .submit(JobSpec::new(Priority::Viewport), |_, _| {
                thread::current().id()
            })
            .wait()
            .expect("job");
        assert_eq!(ran_on, built_on);
        let failed = Scheduler::<()>::spawn("failing", || Err::<((), ()), _>("no device"));
        assert!(matches!(failed, Err(SpawnError::Init("no device"))));
    }

    /// Several threads submit, cancel, navigate and shut down at once: every handle must resolve
    /// (to a result, `Cancelled` or `Shutdown`), none may hang and the worker must not die.
    #[test]
    fn concurrent_submit_cancel_and_shutdown_resolve_every_handle() {
        for round in 0..6 {
            let s = Arc::new(scheduler());
            let threads: Vec<_> = (0..4u64)
                .map(|t| {
                    let s = Arc::clone(&s);
                    thread::spawn(move || {
                        let mut handles = Vec::new();
                        for i in 0..150u64 {
                            let priority = match (t + i) % 4 {
                                0 => Priority::Export,
                                1 => Priority::Thumbnail,
                                2 => Priority::TileNeural,
                                _ => Priority::Viewport,
                            };
                            let h = s.submit(
                                JobSpec::new(priority).owned_by(i % 5),
                                move |log: &mut Vec<&'static str>, token| {
                                    log.push("ran");
                                    token.is_cancelled()
                                },
                            );
                            if i % 7 == 0 {
                                h.cancel();
                            }
                            if i % 31 == 0 {
                                s.cancel_navigation(Some(i % 5));
                            }
                            if i % 97 == 0 {
                                s.cancel_all();
                            }
                            handles.push(h);
                        }
                        handles
                            .into_iter()
                            .map(|h| match h.wait_timeout(Duration::from_secs(30)) {
                                Some(Ok(_) | Err(JobError::Cancelled | JobError::Shutdown)) => 1,
                                other => panic!("a job did not resolve properly: {other:?}"),
                            })
                            .sum::<usize>()
                    })
                })
                .collect();
            if round % 2 == 0 {
                // Shut down while the other threads are still submitting.
                thread::sleep(Duration::from_millis(1));
                s.shutdown();
            }
            let resolved: usize = threads.into_iter().map(|t| t.join().expect("thread")).sum();
            assert_eq!(resolved, 4 * 150, "round {round}");
        }
    }

    #[test]
    fn jobs_sharing_a_token_are_all_skipped_when_it_is_set_and_others_still_run() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let ran = Arc::new(AtomicUsize::new(0));
        let shared = CancelToken::new();
        let mk = |token: CancelToken, p: Priority| {
            let ran = Arc::clone(&ran);
            s.submit_with_token(JobSpec::new(p), token, move |_, _| {
                ran.fetch_add(1, Ordering::SeqCst);
            })
        };
        let a = mk(shared.clone(), Priority::Viewport);
        let b = mk(shared.clone(), Priority::Export);
        let other = mk(CancelToken::new(), Priority::Thumbnail);
        assert!(
            !a.token().is_cancelled() && !b.token().is_cancelled(),
            "nothing is cancelled yet"
        );
        // The handle's token is the shared one: cancelling through either handle cancels both.
        b.cancel();
        assert!(shared.is_cancelled() && a.token().is_cancelled());
        open.send(()).expect("open");
        g.wait().expect("gate");
        assert_eq!(a.wait(), Err(JobError::Cancelled));
        assert_eq!(b.wait(), Err(JobError::Cancelled));
        other.wait().expect("the job with its own token ran");
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_running_job_sees_the_shared_token_and_navigation_sets_it_for_the_whole_render() {
        let s = scheduler();
        let token = CancelToken::new();
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let seen = s.submit_with_token(
            JobSpec::new(Priority::Viewport).owned_by(4),
            token.clone(),
            move |_, t| {
                let _ = started_tx.send(());
                let _ = go_rx.recv();
                t.is_cancelled()
            },
        );
        started_rx.recv_timeout(WAIT).expect("started");
        // A queued sibling of the same render, at a priority navigation does not cancel on its own.
        let sibling = s.submit_with_token(
            JobSpec::new(Priority::Export).owned_by(4),
            token.clone(),
            |log: &mut Vec<&'static str>, _| log.push("sibling"),
        );
        assert_eq!(s.cancel_navigation(Some(4)), 1, "the running viewport job");
        assert!(token.is_cancelled(), "the render's token is set");
        go_tx.send(()).expect("go");
        assert_eq!(
            seen.wait(),
            Ok(true),
            "the running job sees the shared token"
        );
        assert_eq!(
            sibling.wait_timeout(WAIT),
            Some(Err(JobError::Cancelled)),
            "the queued sibling is skipped because it shares the token"
        );
        let log = s
            .submit(JobSpec::new(Priority::Export), |log, _| log.clone())
            .wait()
            .expect("log");
        assert!(log.is_empty(), "{log:?}");
    }

    #[test]
    fn cancel_all_clears_the_queue() {
        let s = scheduler();
        let (open, g) = gate(&s);
        let hs: Vec<_> = [Priority::Export, Priority::Thumbnail, Priority::Viewport]
            .into_iter()
            .map(|p| s.submit(JobSpec::new(p), |_, _| ()))
            .collect();
        assert_eq!(s.cancel_all(), 4, "three pending and the running gate job");
        for h in hs {
            assert_eq!(h.wait_timeout(WAIT), Some(Err(JobError::Cancelled)));
        }
        open.send(()).expect("open");
        g.wait().expect("gate finishes");
    }
}
