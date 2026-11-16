//! Models that load on first use and are let go when idle or when memory is short (`docs/26-hardening.md` P4-4, H-14).
//!
//! A [`ModelSlot`] holds one model's weights: `Untried` until something asks for them, `Ready` while they are in memory,
//! `Damaged` for the rest of the run when they could not be read. It adds two things to what `ObjectModel` and `DepthModel`
//! did before:
//!
//! * **Idle unload.** A model that nobody has asked for during [`MODEL_IDLE`] (five minutes) goes back to `Untried`; the next
//!   use reads the weights again (about half a second for SAM). The time comes from a [`Clock`] the slot is given, so a test
//!   moves it by hand and never sleeps.
//! * **Trim.** [`ModelSlot::trim`] is what a memory warning asks for: unload now.
//!
//! **A model in use is never unloaded.** The only way to the weights is [`ModelSlot::get`], which clones the `Arc` *while it
//! holds the slot's lock*; an inference holds that clone until it is done. An unload decides under the same lock, and does it
//! only when the slot's own `Arc` is the only one (`Arc::strong_count == 1`). Nothing can raise the count meanwhile (a clone
//! needs the lock), so a count of one means no inference is running on the weights and none can start before the lock is
//! released, at which point the state is `Untried` and the next `get` loads. When the count is more than one the slot stays
//! `Ready`, remembers that an unload was asked for (a trim) and counts the model as used just now (so the idle time starts when
//! it was last seen busy); the next sweep ([`ModelSlot::sweep`], called every poll tick of the core's memory watch) unloads it
//! once the inference has let go.
//!
//! A model that cannot be loaded again (made from weights already in memory, as tests do) is **pinned**: it never unloads.
//!
//! Any other model joins by implementing [`Unloadable`] (usually by delegating to its slot) and registering with
//! [`Core::register_unloadable`](crate::app::Core::register_unloadable).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// How long a model may go without being used before it is unloaded: five minutes (`docs/26` P4-4).
pub const MODEL_IDLE: Duration = Duration::from_secs(300);

/// A reading of a monotonic clock: the time since some fixed moment. Models read it to stamp their use; a test gives them a
/// clock it moves by hand.
pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// The clock of a run: the time since the first time it was asked for.
#[must_use]
pub fn monotonic_clock() -> Clock {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    Arc::new(|| ORIGIN.get_or_init(Instant::now).elapsed())
}

/// What an unload did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unload {
    /// Nothing was in memory (never loaded, already unloaded, or damaged).
    NotLoaded,
    /// The weights are in memory and stay: not idle long enough, or pinned (they cannot be read again).
    Kept,
    /// An inference holds the weights. The unload is remembered and happens at the first sweep after it lets go.
    InUse,
    /// The weights were let go.
    Unloaded,
}

/// A model the core can unload: what a memory warning and the idle sweep act on.
pub trait Unloadable: Send + Sync {
    /// A short name for the log.
    fn name(&self) -> &'static str;
    /// Unloads the weights if they have been idle for the model's idle time, or if an unload was asked for while they were busy.
    fn sweep(&self) -> Unload;
    /// Unloads the weights now (a memory warning), or as soon as the inference that holds them ends.
    fn trim(&self) -> Unload;
    /// Whether the weights are in memory now.
    fn loaded(&self) -> bool;
}

/// Why [`ModelSlot::get`] gave no weights.
#[derive(Debug, PartialEq, Eq)]
pub enum SlotError<E> {
    /// An earlier load failed: the model is gone for the rest of the run.
    Damaged,
    /// This load failed (and the slot is `Damaged` from now on).
    Load(E),
}

enum State<T> {
    /// Not in memory (never asked for, or unloaded).
    Untried,
    /// In memory.
    Ready {
        value: Arc<T>,
        /// When `get` last handed it out, or when a sweep last found it busy.
        last_used: Duration,
        /// The weights cannot be read again, so they are never unloaded.
        pinned: bool,
        /// A trim found the weights busy: unload them at the first sweep that finds them free.
        trim_pending: bool,
    },
    /// Could not be read.
    Damaged,
}

/// The weights of one model and the rule for keeping them (see the module documentation).
pub struct ModelSlot<T> {
    state: Mutex<State<T>>,
    clock: Clock,
    idle: Duration,
    loads: AtomicU64,
    unloads: AtomicU64,
}

impl<T> std::fmt::Debug for ModelSlot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSlot")
            .field("loaded", &self.loaded())
            .field("loads", &self.loads())
            .finish_non_exhaustive()
    }
}

impl<T> ModelSlot<T> {
    /// A slot with nothing in it: the first [`get`](Self::get) loads. Idle time [`MODEL_IDLE`], the run's own clock.
    #[must_use]
    pub fn lazy() -> Self {
        Self::lazy_with(monotonic_clock(), MODEL_IDLE)
    }

    /// [`ModelSlot::lazy`] with the clock and the idle time given (tests).
    #[must_use]
    pub fn lazy_with(clock: Clock, idle: Duration) -> Self {
        Self {
            state: Mutex::new(State::Untried),
            clock,
            idle,
            loads: AtomicU64::new(0),
            unloads: AtomicU64::new(0),
        }
    }

    /// A slot that holds `value` for good: it is never unloaded, because it cannot be read again.
    #[must_use]
    pub fn pinned(value: Arc<T>) -> Self {
        let slot = Self::lazy();
        *slot.lock() = State::Ready {
            value,
            last_used: Duration::ZERO,
            pinned: true,
            trim_pending: false,
        };
        slot
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a load has failed: the model is gone for the rest of the run.
    #[must_use]
    pub fn damaged(&self) -> bool {
        matches!(*self.lock(), State::Damaged)
    }

    /// Whether the weights are in memory now.
    #[must_use]
    pub fn loaded(&self) -> bool {
        matches!(*self.lock(), State::Ready { .. })
    }

    /// How many times `load` has run successfully.
    #[must_use]
    pub fn loads(&self) -> u64 {
        self.loads.load(Ordering::SeqCst)
    }

    /// How many times the weights have been unloaded.
    #[must_use]
    pub fn unloads(&self) -> u64 {
        self.unloads.load(Ordering::SeqCst)
    }

    /// The weights, calling `load` first when they are not in memory. The caller keeps the `Arc` for as long as it works with
    /// the weights: that is what keeps the slot from unloading them. Counts as a use now.
    ///
    /// `load` runs with the slot's lock held, so two callers never read the weights twice.
    ///
    /// # Errors
    ///
    /// [`SlotError::Damaged`] when an earlier load failed, [`SlotError::Load`] with `load`'s error when this one does.
    pub fn get<E>(&self, load: impl FnOnce() -> Result<T, E>) -> Result<Arc<T>, SlotError<E>> {
        let mut st = self.lock();
        match &mut *st {
            State::Ready {
                value, last_used, ..
            } => {
                *last_used = (self.clock)();
                return Ok(Arc::clone(value));
            }
            State::Damaged => return Err(SlotError::Damaged),
            State::Untried => {}
        }
        match load() {
            Ok(value) => {
                let value = Arc::new(value);
                self.loads.fetch_add(1, Ordering::SeqCst);
                *st = State::Ready {
                    value: Arc::clone(&value),
                    last_used: (self.clock)(),
                    pinned: false,
                    trim_pending: false,
                };
                Ok(value)
            }
            Err(e) => {
                *st = State::Damaged;
                Err(SlotError::Load(e))
            }
        }
    }

    /// Unloads the weights when they have been idle for the idle time, or when an earlier [`trim`](Self::trim) found them busy
    /// and they are free now. A busy slot counts as used just now.
    pub fn sweep(&self) -> Unload {
        self.unload(false)
    }

    /// Unloads the weights now, or as soon as the inference that holds them ends (a memory warning).
    pub fn trim(&self) -> Unload {
        self.unload(true)
    }

    fn unload(&self, now: bool) -> Unload {
        let mut st = self.lock();
        let at = (self.clock)();
        let State::Ready {
            value,
            last_used,
            pinned,
            trim_pending,
        } = &mut *st
        else {
            return Unload::NotLoaded;
        };
        if *pinned {
            return Unload::Kept;
        }
        let wanted = now || *trim_pending || at.saturating_sub(*last_used) >= self.idle;
        if !wanted {
            return Unload::Kept;
        }
        // Every clone is made under this lock, so the count cannot rise while we look: one means nobody is using the weights.
        if Arc::strong_count(value) > 1 {
            *trim_pending = true;
            *last_used = at;
            return Unload::InUse;
        }
        *st = State::Untried;
        self.unloads.fetch_add(1, Ordering::SeqCst);
        Unload::Unloaded
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::*;

    /// A clock a test moves by hand.
    #[derive(Clone)]
    struct Hand(Arc<AtomicU64>);

    impl Hand {
        fn new() -> Self {
            Self(Arc::new(AtomicU64::new(0)))
        }
        fn clock(&self) -> Clock {
            let t = Arc::clone(&self.0);
            Arc::new(move || Duration::from_secs(t.load(Ordering::SeqCst)))
        }
        fn advance(&self, secs: u64) {
            self.0.fetch_add(secs, Ordering::SeqCst);
        }
    }

    fn slot(hand: &Hand) -> ModelSlot<u32> {
        ModelSlot::lazy_with(hand.clock(), Duration::from_secs(300))
    }

    fn load_ok() -> Result<u32, String> {
        Ok(7)
    }

    #[test]
    fn a_model_idle_for_five_minutes_unloads_and_the_next_use_loads_again() {
        let hand = Hand::new();
        let slot = slot(&hand);
        drop(slot.get(load_ok).expect("loads"));
        assert_eq!((slot.loads(), slot.loaded()), (1, true));
        hand.advance(299);
        assert_eq!(slot.sweep(), Unload::Kept, "not idle long enough");
        assert!(slot.loaded());
        hand.advance(1);
        assert_eq!(slot.sweep(), Unload::Unloaded);
        assert!(!slot.loaded());
        assert_eq!(slot.sweep(), Unload::NotLoaded);
        let again = slot.get(load_ok).expect("loads again");
        assert_eq!(*again, 7);
        assert_eq!(slot.loads(), 2, "the next use read the weights again");
    }

    #[test]
    fn a_use_during_the_idle_time_starts_it_again() {
        let hand = Hand::new();
        let slot = slot(&hand);
        drop(slot.get(load_ok).expect("loads"));
        hand.advance(200);
        drop(slot.get(load_ok).expect("a use"));
        hand.advance(200);
        // 400 s since the load, 200 s since the use.
        assert_eq!(slot.sweep(), Unload::Kept);
        assert_eq!(slot.loads(), 1);
        hand.advance(100);
        assert_eq!(slot.sweep(), Unload::Unloaded);
    }

    #[test]
    fn a_model_in_use_is_kept_and_unloaded_at_the_first_sweep_after_it_is_free() {
        let hand = Hand::new();
        let slot = slot(&hand);
        let held = slot.get(load_ok).expect("loads");
        // Idle by the clock, and a trim on top: both find the weights busy.
        hand.advance(1000);
        assert_eq!(slot.sweep(), Unload::InUse);
        assert_eq!(slot.trim(), Unload::InUse);
        assert!(slot.loaded());
        assert_eq!(*held, 7, "the inference still has its weights");
        drop(held);
        // The inference is over: no idle time has passed since the busy sweep, yet the trim that was asked for is done.
        assert_eq!(slot.sweep(), Unload::Unloaded);
        assert_eq!(slot.unloads(), 1);
    }

    #[test]
    fn trim_unloads_at_once_and_a_pinned_model_never_unloads() {
        let hand = Hand::new();
        let slot = slot(&hand);
        drop(slot.get(load_ok).expect("loads"));
        assert_eq!(slot.trim(), Unload::Unloaded);
        assert_eq!(slot.trim(), Unload::NotLoaded);

        let pinned = ModelSlot::pinned(Arc::new(9u32));
        assert_eq!(pinned.trim(), Unload::Kept);
        assert_eq!(pinned.sweep(), Unload::Kept);
        assert!(pinned.loaded());
        assert_eq!(*pinned.get(|| Err::<u32, ()>(())).expect("still there"), 9);
    }

    #[test]
    fn a_failed_load_takes_the_model_away_for_the_run() {
        let hand = Hand::new();
        let slot = slot(&hand);
        assert_eq!(
            slot.get(|| Err::<u32, &str>("bad weights")).unwrap_err(),
            SlotError::Load("bad weights")
        );
        assert!(slot.damaged());
        assert_eq!(
            slot.get(load_ok).unwrap_err(),
            SlotError::Damaged,
            "no second try"
        );
        assert_eq!(slot.loads(), 0);
        assert_eq!(slot.sweep(), Unload::NotLoaded);
    }

    #[test]
    fn an_unload_never_frees_weights_an_inference_is_using() {
        // Many threads use the slot while another trims it as fast as it can: every user sees the weights it was given until it
        // lets go, and every trim either unloads a free model or leaves a busy one alone.
        let slot = Arc::new(ModelSlot::<Vec<u32>>::lazy());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let trimmer = {
            let (slot, stop) = (Arc::clone(&slot), Arc::clone(&stop));
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    slot.trim();
                }
            })
        };
        let users: Vec<_> = (0..4)
            .map(|_| {
                let slot = Arc::clone(&slot);
                std::thread::spawn(move || {
                    for _ in 0..2000 {
                        let w = slot.get(|| Ok::<_, ()>(vec![1; 64])).expect("weights");
                        // The weights stay whole while this clone lives.
                        assert_eq!(w.iter().sum::<u32>(), 64);
                        assert!(Arc::strong_count(&w) > 1);
                    }
                })
            })
            .collect();
        for u in users {
            u.join().expect("a user");
        }
        stop.store(true, Ordering::SeqCst);
        trimmer.join().expect("the trimmer");
    }
}
