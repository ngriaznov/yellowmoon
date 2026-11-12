//! Memory pressure: what the core lets go of when the computer is short of memory, and the poll that notices
//! (`docs/26-hardening.md` P4-4, P4-5, H-14; `docs/17-gpu.md` section 8).
//!
//! **The level.** macOS says how it feels about memory with `sysctl kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning,
//! 4 critical ([`PressureLevel`]). A [`PressureSource`] reads it; the real one ([`system_pressure_source`]) exists only on macOS
//! and everywhere else always says normal. Tests give a source that says what they want.
//!
//! **What goes at each level** ([`Core::trim_caches`], the one place that frees anything for pressure):
//!
//! | level | frees |
//! |---|---|
//! | normal | nothing |
//! | warning | the cached mosaics of the photos next to the one on screen; the finished-frame caches of every photo that is not on screen; the models |
//! | critical | the same, then the device's resident raw words of the other photos and the pool's free buffers |
//!
//! Warning drops caches through the two ways the core already has and adds no third: the ledger's reclaimers
//! ([`MemoryLedger::reclaim`], H-13: the session's mosaics and the renderer's raw groups and sigma planes, other photos' only)
//! and [`Renderer::release_off_screen_frames`] (the frames the ledger does not count). The models unload through
//! [`Unloadable`](crate::pipeline::Unloadable) (`pipeline/resident.rs`), and one that is running an inference unloads when the
//! inference ends. **The photo on screen keeps every cache it has.**
//!
//! **What else critical does** ([`Core::apply_pressure`], called by the poll with each reading): the optional background work
//! stops, and starts again when the level falls below critical. That is the decode pool's prefetches and thumbnail refreshes
//! and the shell's thumbnail refresher, through one [`BackgroundGate`](crate::scheduler::BackgroundGate). A photo's development,
//! the viewport and exports go on.
//!
//! **The notice.** On the way into critical the core says once, through [`Core::on_memory_notice`], that background work is
//! paused ([`MEMORY_NOTICE_SENTENCE`]); it says the notice is gone when the level is normal again. Warning shows nothing, and
//! a level that goes from critical to warning and back leaves the notice as it is: **once per stay above normal**.
//!
//! **The poll.** [`Core::start_memory_watch`] runs one thread that reads the level every [`POLL_INTERVAL`], acts on it, and
//! sweeps the models for the idle unload. The app starts it; `ym` does not (see `docs/26` As built, H-14). The thread ends with
//! [`Core::shutdown`] or when the core is dropped.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use crate::pipeline::{Renderer, Unload, Unloadable};
use crate::scheduler::{BackgroundGate, Reclaim};
use crate::session::{PhotoId, Session};

/// How often the poll reads the level: every two seconds.
pub const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// The words of the notice that shows while memory is critical. The core owns them so the app and the mock say the same.
pub const MEMORY_NOTICE_SENTENCE: &str = "Your Mac is short of memory, so Yellowmoon paused background work. Close other apps to speed it up again.";

/// How short of memory the computer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    /// Plenty of memory (sysctl 1).
    Normal,
    /// The system is compressing and starting to page (sysctl 2).
    Warning,
    /// The system is about to kill processes (sysctl 4).
    Critical,
}

impl PressureLevel {
    /// The level of a `kern.memorystatus_vm_pressure_level` value: 4 and above is critical, 2 and 3 warning, anything else
    /// (1, 0, a negative number) normal.
    #[must_use]
    pub fn from_sysctl(value: i64) -> Self {
        match value {
            v if v >= 4 => Self::Critical,
            2 | 3 => Self::Warning,
            _ => Self::Normal,
        }
    }
}

/// Where the poll reads the level: a function the core calls every [`POLL_INTERVAL`]. Cloning shares it.
#[derive(Clone)]
pub struct PressureSource(Arc<dyn Fn() -> PressureLevel + Send + Sync>);

impl PressureSource {
    /// A source that is `read`.
    pub fn new(read: impl Fn() -> PressureLevel + Send + Sync + 'static) -> Self {
        Self(Arc::new(read))
    }

    /// Reads the level now.
    #[must_use]
    pub fn read(&self) -> PressureLevel {
        (self.0)()
    }
}

impl std::fmt::Debug for PressureSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PressureSource")
    }
}

/// The level of this computer: `sysctl kern.memorystatus_vm_pressure_level` on macOS (an error reads as normal), always
/// normal elsewhere.
#[must_use]
pub fn system_pressure_source() -> PressureSource {
    PressureSource::new(system_level)
}

#[cfg(target_os = "macos")]
fn system_level() -> PressureLevel {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    // SAFETY: the name is a NUL-terminated literal; `value` and `size` are live, `size` is the size of `value`, and no new
    // value is passed (null pointer, zero length), so the call only reads.
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut value).cast::<libc::c_void>(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        PressureLevel::from_sysctl(i64::from(value))
    } else {
        PressureLevel::Normal
    }
}

#[cfg(not(target_os = "macos"))]
fn system_level() -> PressureLevel {
    PressureLevel::Normal
}

/// What [`Core::trim_caches`] let go of.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrimReport {
    /// What each registered model did: its name and the outcome ([`Unload::InUse`]: it unloads when its inference ends).
    pub models: Vec<(&'static str, Unload)>,
    /// How many photos that are not on screen had their finished-frame caches dropped.
    pub off_screen_photos: usize,
    /// Bytes the device's pool gave back (critical only; 0 without a device).
    pub gpu_bytes: u64,
}

/// What [`Core::apply_pressure`] did with one reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PressureStep {
    /// The level read.
    pub level: PressureLevel,
    /// Whether the level was higher than the one before, so the caches were trimmed.
    pub trimmed: bool,
    /// Whether the background work is paused now.
    pub paused: bool,
    /// The notice changed: `Some(true)` it was shown, `Some(false)` it went away, `None` it stayed as it was.
    pub notice: Option<bool>,
}

type NoticeListener = Arc<dyn Fn(Option<&str>) + Send + Sync>;

/// The state the pressure rules need, shared by the core and the poll thread (which holds it, never the core).
pub(crate) struct Guard {
    pub(crate) renderer: Renderer,
    pub(crate) session: Arc<Session>,
    pub(crate) gate: Arc<BackgroundGate>,
    models: Mutex<Vec<Arc<dyn Unloadable>>>,
    state: Mutex<Observed>,
    listeners: Mutex<Vec<NoticeListener>>,
    polls: AtomicU64,
}

#[derive(Default)]
struct Observed {
    level: Option<PressureLevel>,
    notice_shown: bool,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Guard {
    pub(crate) fn new(
        renderer: Renderer,
        session: Arc<Session>,
        gate: Arc<BackgroundGate>,
    ) -> Self {
        Self {
            renderer,
            session,
            gate,
            models: Mutex::new(Vec::new()),
            state: Mutex::new(Observed::default()),
            listeners: Mutex::new(Vec::new()),
            polls: AtomicU64::new(0),
        }
    }

    pub(crate) fn register(&self, model: Arc<dyn Unloadable>) {
        let mut models = lock(&self.models);
        if !models.iter().any(|m| Arc::ptr_eq(m, &model)) {
            models.push(model);
        }
    }

    pub(crate) fn on_notice(&self, f: NoticeListener) {
        lock(&self.listeners).push(f);
    }

    /// Frees what `level` lists (see the module documentation).
    pub(crate) fn trim(&self, level: PressureLevel) -> TrimReport {
        let mut report = TrimReport::default();
        if level == PressureLevel::Normal {
            return report;
        }
        let keep: Option<PhotoId> = self.session.current();
        // The caches the ledger counts, of the photos other than the one on screen: the session's cached mosaics (the neighbours')
        // and the renderer's raw groups and sigma planes. The reclaimers of H-13 are the one way these are freed.
        self.renderer.ledger().reclaim(Reclaim { all: false, keep });
        // The frames the ledger does not count.
        report.off_screen_photos = self.renderer.release_off_screen_frames(keep);
        // The models: now, or when the inference that holds them ends.
        let models: Vec<Arc<dyn Unloadable>> = lock(&self.models).clone();
        for model in models {
            report.models.push((model.name(), model.trim()));
        }
        if level == PressureLevel::Critical {
            report.gpu_bytes = self.renderer.trim_gpu(keep);
        }
        report
    }

    /// Unloads the models that have been idle, and those whose unload was waiting for an inference to end. Returns how many
    /// were unloaded.
    pub(crate) fn sweep(&self) -> usize {
        let models: Vec<Arc<dyn Unloadable>> = lock(&self.models).clone();
        models
            .iter()
            .filter(|m| m.sweep() == Unload::Unloaded)
            .count()
    }

    /// Acts on one reading of the level: trims when it rose, pauses at critical and resumes below it, shows and hides the
    /// notice.
    pub(crate) fn apply(&self, level: PressureLevel) -> PressureStep {
        self.polls.fetch_add(1, Ordering::SeqCst);
        let (rose, notice) = {
            let mut st = lock(&self.state);
            let rose = st.level.unwrap_or(PressureLevel::Normal) < level;
            st.level = Some(level);
            let notice = match level {
                PressureLevel::Critical if !st.notice_shown => {
                    st.notice_shown = true;
                    Some(true)
                }
                PressureLevel::Normal if st.notice_shown => {
                    st.notice_shown = false;
                    Some(false)
                }
                _ => None,
            };
            (rose, notice)
        };
        if rose {
            self.trim(level);
        }
        // Paused before the notice is told, resumed before it goes: what the sentence says is true while it shows.
        let paused = level == PressureLevel::Critical;
        self.gate.set_paused(paused);
        if let Some(shown) = notice {
            let listeners: Vec<NoticeListener> = lock(&self.listeners).clone();
            for listener in listeners {
                listener(shown.then_some(MEMORY_NOTICE_SENTENCE));
            }
        }
        PressureStep {
            level,
            trimmed: rose,
            paused,
            notice,
        }
    }

    /// How many readings have been applied.
    pub(crate) fn polls(&self) -> u64 {
        self.polls.load(Ordering::SeqCst)
    }
}

/// Ends the poll thread: stops it between two readings and waits for it.
pub(crate) struct Watch {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watch {
    /// Starts the thread: it reads `source` at once and then every `interval`, applies each reading to `guard` and sweeps the
    /// models.
    pub(crate) fn start(
        guard: Arc<Guard>,
        source: PressureSource,
        interval: Duration,
    ) -> std::io::Result<Self> {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("ym-memory".to_owned())
            .spawn(move || {
                loop {
                    guard.apply(source.read());
                    guard.sweep();
                    let (stopped, wake) = &*flag;
                    let mut stopped_now = lock(stopped);
                    if !*stopped_now {
                        stopped_now = wake
                            .wait_timeout(stopped_now, interval)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0;
                    }
                    if *stopped_now {
                        return;
                    }
                }
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    /// Stops the thread and waits for it. Idempotent.
    pub(crate) fn stop(&mut self) {
        {
            let (stopped, wake) = &*self.stop;
            *lock(stopped) = true;
            wake.notify_all();
        }
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            tracing::error!("the memory watch ended with a panic");
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sysctl_values_map_to_levels() {
        for (value, level) in [
            (1, PressureLevel::Normal),
            (2, PressureLevel::Warning),
            (4, PressureLevel::Critical),
            (0, PressureLevel::Normal),
            (3, PressureLevel::Warning),
            (8, PressureLevel::Critical),
            (-1, PressureLevel::Normal),
        ] {
            assert_eq!(PressureLevel::from_sysctl(value), level, "value {value}");
        }
        assert!(PressureLevel::Normal < PressureLevel::Warning);
        assert!(PressureLevel::Warning < PressureLevel::Critical);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_default_source_says_normal_off_the_mac() {
        assert_eq!(system_pressure_source().read(), PressureLevel::Normal);
    }
}
