//! The cache of what a program is and whether it can answer, and the probe that fills it (`docs/22`
//! section 3.3): reads never wait for a process. `refresh` starts one probe on a background thread and
//! returns; one probe runs at a time and a call within the minimum interval of the last probe's end
//! does nothing; `settle` waits; `on_change` hears of a change.

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Called when the cached reading changes (the same type as `ym_core::auto_provider::ChangeListener`).
pub type Listener = Arc<dyn Fn() + Send + Sync>;

/// How much a probe asks of the program. `agy`'s sign-in check is the only probe known to make its
/// program contact a network, so the start-up probe is [`Level::Light`] (up to its version) and only a
/// [`Level::Full`] one, asked for by `refresh`, runs the sign-in check (`docs/22` D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Finding the program and its version.
    Light,
    /// Everything, the sign-in check included.
    Full,
}

type Job<T> = Box<dyn FnOnce() -> T + Send>;

struct State<T> {
    current: T,
    /// The level of the probe that runs now.
    running: Option<Level>,
    /// A full probe asked for while a light one ran: it follows at once.
    pending: Option<Job<T>>,
    /// When the last full probe ended.
    last_end: Option<Instant>,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    idle: Condvar,
    listeners: Mutex<Vec<Listener>>,
    min_interval: Duration,
}

/// The cached reading of one program with the machinery around it. Cheap to clone; every clone is the
/// same cache.
pub struct Prober<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Clone for Prober<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<T: Clone + PartialEq + Send + 'static> Prober<T> {
    /// A cache holding `initial`, whose full probes are at least `min_interval` apart.
    #[must_use]
    pub fn new(initial: T, min_interval: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    current: initial,
                    running: None,
                    pending: None,
                    last_end: None,
                }),
                idle: Condvar::new(),
                listeners: Mutex::new(Vec::new()),
                min_interval,
            }),
        }
    }

    /// The last reading. Never waits for a process.
    #[must_use]
    pub fn current(&self) -> T {
        lock(&self.shared.state).current.clone()
    }

    /// Whether a probe runs now.
    #[must_use]
    #[cfg_attr(not(any(test, feature = "test-support")), allow(dead_code))]
    pub fn is_running(&self) -> bool {
        lock(&self.shared.state).running.is_some()
    }

    /// Replaces the reading (a request's outcome: a success renews `available`, a sign-out is
    /// `signed_out`) and calls the listeners when it differs.
    pub fn set(&self, value: T) {
        let changed = {
            let mut state = lock(&self.shared.state);
            let changed = state.current != value;
            state.current = value;
            changed
        };
        if changed {
            self.notify();
        }
    }

    /// Starts `probe` on a background thread and returns at once; `true` when it was started or queued.
    ///
    /// Does nothing while a probe of the same or a higher level runs, and, for a full probe, within the
    /// minimum interval of the last full probe's end. A full probe asked for while a light one runs
    /// follows it at once. The result becomes the reading; the listeners are called when it differs
    /// from the last one.
    pub fn refresh<F>(&self, level: Level, probe: F) -> bool
    where
        F: FnOnce() -> T + Send + 'static,
    {
        {
            let mut state = lock(&self.shared.state);
            match (state.running, level) {
                (Some(Level::Full), _) | (Some(Level::Light), Level::Light) => return false,
                (Some(Level::Light), Level::Full) => {
                    state.pending = Some(Box::new(probe));
                    return true;
                }
                (None, _) => {}
            }
            if level == Level::Full
                && state
                    .last_end
                    .is_some_and(|end| end.elapsed() < self.shared.min_interval)
            {
                return false;
            }
            state.running = Some(level);
        }
        let this = self.clone();
        let first: Job<T> = Box::new(probe);
        let spawned = std::thread::Builder::new()
            .name("ym-agents-probe".to_owned())
            .spawn(move || this.work(level, first));
        if spawned.is_err() {
            lock(&self.shared.state).running = None;
            self.shared.idle.notify_all();
            return false;
        }
        true
    }

    /// The probe thread: runs `first`, then a full probe that was queued behind it. The listeners hear
    /// of a change before `settle` returns, so a caller that settled reads what they read.
    fn work(&self, level: Level, first: Job<T>) {
        let mut job = Some((level, first));
        while let Some((level, probe)) = job.take() {
            // A panic in a probe must not leave `running` set: the next refresh would never start.
            let result = std::panic::catch_unwind(AssertUnwindSafe(probe));
            let changed = {
                let mut state = lock(&self.shared.state);
                let changed = match result {
                    Ok(value) => {
                        let changed = state.current != value;
                        state.current = value;
                        changed
                    }
                    Err(_) => {
                        tracing::warn!("a probe of an Auto provider panicked");
                        false
                    }
                };
                if level == Level::Full {
                    state.last_end = Some(Instant::now());
                }
                changed
            };
            if changed {
                self.notify();
            }
            {
                let mut state = lock(&self.shared.state);
                match state.pending.take() {
                    Some(next) => {
                        job = Some((Level::Full, next));
                        state.running = Some(Level::Full);
                    }
                    None => state.running = None,
                }
            }
            if job.is_none() {
                self.shared.idle.notify_all();
            }
        }
    }

    /// Waits until no probe runs or `timeout` is up.
    pub fn settle(&self, timeout: Duration) {
        let state = lock(&self.shared.state);
        let _ = self
            .shared
            .idle
            .wait_timeout_while(state, timeout, |s| s.running.is_some())
            .unwrap_or_else(PoisonError::into_inner);
    }

    /// Registers a listener (kept for the life of the cache).
    pub fn on_change(&self, listener: Listener) {
        lock(&self.shared.listeners).push(listener);
    }

    /// Calls the listeners, outside every lock of ours: a listener reads the cache again.
    fn notify(&self) {
        let listeners: Vec<Listener> = lock(&self.shared.listeners).clone();
        for listener in listeners {
            listener();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn refresh_returns_at_once_and_the_probe_fills_the_cache() {
        let prober = Prober::new(0_u32, Duration::ZERO);
        let started = Instant::now();
        assert!(prober.refresh(Level::Full, || {
            std::thread::sleep(Duration::from_millis(200));
            7
        }));
        assert!(started.elapsed() < Duration::from_millis(100));
        assert_eq!(prober.current(), 0);
        prober.settle(Duration::from_secs(5));
        assert_eq!(prober.current(), 7);
    }

    #[test]
    fn calls_while_a_probe_runs_or_within_the_interval_do_nothing() {
        let prober = Prober::new(0_u32, Duration::from_secs(60));
        let runs = Arc::new(AtomicUsize::new(0));
        let probe = |runs: &Arc<AtomicUsize>| {
            let runs = Arc::clone(runs);
            move || {
                runs.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(100));
                1
            }
        };
        assert!(prober.refresh(Level::Full, probe(&runs)));
        assert!(!prober.refresh(Level::Full, probe(&runs)), "one probe at a time");
        prober.settle(Duration::from_secs(5));
        assert!(!prober.refresh(Level::Full, probe(&runs)), "within the minimum interval");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn listeners_hear_of_a_change_once_and_not_of_a_repeat() {
        let prober = Prober::new(0_u32, Duration::ZERO);
        let heard = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&heard);
        prober.on_change(Arc::new(move || {
            h.fetch_add(1, Ordering::SeqCst);
        }));
        prober.refresh(Level::Full, || 5);
        prober.settle(Duration::from_secs(5));
        assert_eq!(heard.load(Ordering::SeqCst), 1);
        prober.refresh(Level::Full, || 5);
        prober.settle(Duration::from_secs(5));
        assert_eq!(heard.load(Ordering::SeqCst), 1, "the same reading is not a change");
        prober.set(5);
        assert_eq!(heard.load(Ordering::SeqCst), 1);
        prober.set(6);
        assert_eq!(heard.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_panicking_probe_leaves_the_cache_usable() {
        let prober = Prober::new(1_u32, Duration::ZERO);
        prober.refresh(Level::Full, || panic!("a bug in a probe"));
        prober.settle(Duration::from_secs(5));
        assert!(!prober.is_running());
        assert_eq!(prober.current(), 1);
        assert!(prober.refresh(Level::Full, || 2));
        prober.settle(Duration::from_secs(5));
        assert_eq!(prober.current(), 2);
    }

    #[test]
    fn settle_gives_up_at_its_timeout() {
        let prober = Prober::new(0_u32, Duration::ZERO);
        prober.refresh(Level::Full, || {
            std::thread::sleep(Duration::from_millis(600));
            1
        });
        let started = Instant::now();
        prober.settle(Duration::from_millis(100));
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(prober.is_running());
        prober.settle(Duration::from_secs(5));
        assert_eq!(prober.current(), 1);
    }
}
