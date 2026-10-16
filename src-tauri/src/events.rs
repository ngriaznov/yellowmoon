//! Events from the core to the webview.
//!
//! Everything the shell sends goes through [`Emitter`], so the command handlers and the viewport
//! can be tested without a webview: [`TauriEmitter`] sends the events to the window, `Recorder` (tests only)
//! keeps them for a test. The event names and payloads are the ones of `docs/01` §IPC and
//! `src/api/types.ts` (`ApiEvents`), declared in [`crate::wire`].
//!
//! # Coalescing
//!
//! Each event [`TauriEmitter`] sends costs the main thread a message that runs a script in the
//! webview (`evaluateJavaScript`). Two kinds come in bursts the page cannot use: `render_stats`
//! (one per presented frame, up to the display rate while a slider moves) and `photo_progress`
//! (four or five per photo, thousands when a folder of hundreds of photos opens). They go through
//! [`Throttle`], which sends at most one `render_stats` and one batch of `photo_progress` per
//! [`COALESCE_PERIOD`] (about 30 Hz), from one thread of its own:
//!
//! * `render_stats` is latest-wins: the page only shows the last frame's numbers.
//! * `photo_progress` is latest per photo while the pending event of the photo is only "a stage
//!   started" (the next event of the photo replaces it); an event that says a stage finished
//!   (`done`) or failed (`error`) is never dropped, nor moved past a later event of its photo, so
//!   what the page ends up knowing about every photo is what it was told last.
//! * An event that arrives after a quiet period goes out at once; only the ones that follow it
//!   within the period wait.
//!
//! Every other event is sent as it comes, and the names and payloads are unchanged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use tauri::Runtime;
use tauri_specta::Event;

use crate::wire::{
    AppError, DownloadFound, ErrorEvent, ExportBusy, ExportProgress, LooksChanged,
    LooksImportProgress, PhotoId, PhotoProgress, RenderStats, SettingsChanged, ThumbnailReady,
    ViewportFrame,
};

/// The least time between two `render_stats` events, and between two batches of `photo_progress`:
/// 30 a second.
pub const COALESCE_PERIOD: Duration = Duration::from_millis(33);

/// An event for the webview.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    /// `photo_progress`.
    PhotoProgress(PhotoProgress),
    /// `thumbnail_ready`.
    ThumbnailReady(ThumbnailReady),
    /// `render_stats`.
    RenderStats(RenderStats),
    /// `export_progress`.
    ExportProgress(ExportProgress),
    /// `looks_changed`.
    LooksChanged,
    /// `error`.
    Error(AppError),
    /// `looks_import_progress`.
    LooksImportProgress(LooksImportProgress),
    /// `download_found`.
    DownloadFound(DownloadFound),
    /// `settings_changed` (addition): after a Pick white click, and for each photo a paste changed.
    SettingsChanged(PhotoId),
    /// `export_busy` (addition).
    ExportBusy(bool),
    /// `viewport_frame` (addition, fallback viewport).
    ViewportFrame(ViewportFrame),
}

impl AppEvent {
    /// The name of the event on the bridge.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::PhotoProgress(_) => "photo_progress",
            Self::ThumbnailReady(_) => "thumbnail_ready",
            Self::RenderStats(_) => "render_stats",
            Self::ExportProgress(_) => "export_progress",
            Self::LooksChanged => "looks_changed",
            Self::Error(_) => "error",
            Self::LooksImportProgress(_) => "looks_import_progress",
            Self::DownloadFound(_) => "download_found",
            Self::SettingsChanged(_) => "settings_changed",
            Self::ExportBusy(_) => "export_busy",
            Self::ViewportFrame(_) => "viewport_frame",
        }
    }
}

/// Where events go. Called from any thread, including the core's worker threads while they hold
/// internal locks, so an implementation must not block and must not call back into the core.
pub trait Emitter: Send + Sync + 'static {
    /// Delivers one event. A webview that is gone loses it; nothing else happens.
    fn emit(&self, event: AppEvent);
}

/// Sends events to the app's windows.
pub struct TauriEmitter<R: Runtime> {
    app: tauri::AppHandle<R>,
    /// Where `render_stats` and `photo_progress` go (see the module documentation); `None` if its
    /// thread did not start, and then they are sent as they come.
    throttle: Option<Throttle>,
}

impl<R: Runtime> TauriEmitter<R> {
    /// An emitter for the windows of `app`.
    #[must_use]
    pub fn new(app: tauri::AppHandle<R>) -> Self {
        let sender = app.clone();
        let throttle = Throttle::new(COALESCE_PERIOD, move |event| send(&sender, event))
            .map_err(|e| tracing::warn!(error = %e, "events are sent one by one"))
            .ok();
        Self { app, throttle }
    }
}

/// Sends one event to the windows.
fn send<R: Runtime>(app: &tauri::AppHandle<R>, event: AppEvent) {
    let name = event.name();
    let sent = match event {
        AppEvent::PhotoProgress(e) => e.emit(app),
        AppEvent::ThumbnailReady(e) => e.emit(app),
        AppEvent::RenderStats(e) => e.emit(app),
        AppEvent::ExportProgress(e) => e.emit(app),
        AppEvent::LooksChanged => LooksChanged.emit(app),
        AppEvent::Error(e) => ErrorEvent::from(e).emit(app),
        AppEvent::LooksImportProgress(e) => e.emit(app),
        AppEvent::DownloadFound(e) => e.emit(app),
        AppEvent::SettingsChanged(photo_id) => SettingsChanged { photo_id }.emit(app),
        AppEvent::ExportBusy(busy) => ExportBusy { busy }.emit(app),
        AppEvent::ViewportFrame(e) => e.emit(app),
    };
    if let Err(e) = sent {
        tracing::debug!(event = name, error = %e, "could not send an event");
    }
}

impl<R: Runtime> Emitter for TauriEmitter<R> {
    fn emit(&self, event: AppEvent) {
        match (&self.throttle, event) {
            (Some(throttle), AppEvent::RenderStats(stats)) => throttle.stats(stats),
            (Some(throttle), AppEvent::PhotoProgress(progress)) => throttle.progress(progress),
            (_, event) => send(&self.app, event),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Coalescing
// ---------------------------------------------------------------------------------------------

/// An event that says a stage finished or failed: the page needs to hear it.
fn is_terminal(p: &PhotoProgress) -> bool {
    p.done || p.error.is_some()
}

/// What is waiting to be sent, and when each kind was last sent. No clock and no thread: the
/// caller says what time it is, which is how the tests drive it.
#[derive(Debug)]
struct Merger {
    period: Duration,
    stats: Option<RenderStats>,
    stats_sent: Option<Instant>,
    /// In the order the events arrived; at most one event of a photo that is not terminal, and it is
    /// the last of the photo.
    progress: VecDeque<PhotoProgress>,
    progress_sent: Option<Instant>,
}

impl Merger {
    fn new(period: Duration) -> Self {
        Self {
            period,
            stats: None,
            stats_sent: None,
            progress: VecDeque::new(),
            progress_sent: None,
        }
    }

    /// Keeps `stats` in place of the one waiting; true if none was waiting.
    fn push_stats(&mut self, stats: RenderStats) -> bool {
        self.stats.replace(stats).is_none()
    }

    /// Keeps `event`; true if no event was waiting. The last event of the photo that is waiting
    /// is replaced when it only said a stage started; otherwise `event` goes behind it.
    fn push_progress(&mut self, event: PhotoProgress) -> bool {
        let was_empty = self.progress.is_empty();
        match self.progress.iter_mut().rev().find(|p| p.id == event.id) {
            Some(waiting) if !is_terminal(waiting) => *waiting = event,
            _ => self.progress.push_back(event),
        }
        was_empty
    }

    /// Whether a kind that was last sent at `sent` may be sent at `now`.
    fn ready(&self, sent: Option<Instant>, now: Instant) -> bool {
        sent.is_none_or(|at| now >= at + self.period)
    }

    /// The events that are due at `now`, in the order to send them; they are no longer waiting.
    fn take_due(&mut self, now: Instant) -> Vec<AppEvent> {
        let mut due = Vec::new();
        if self.stats.is_some() && self.ready(self.stats_sent, now) {
            due.extend(self.stats.take().map(AppEvent::RenderStats));
            self.stats_sent = Some(now);
        }
        if !self.progress.is_empty() && self.ready(self.progress_sent, now) {
            due.extend(self.progress.drain(..).map(AppEvent::PhotoProgress));
            self.progress_sent = Some(now);
        }
        due
    }

    /// When the first of the events that wait is due (`now` if one is due already); `None` if
    /// nothing waits.
    fn next_due(&self, now: Instant) -> Option<Instant> {
        let stats = self
            .stats
            .is_some()
            .then(|| self.stats_sent.map_or(now, |at| at + self.period));
        let progress = (!self.progress.is_empty())
            .then(|| self.progress_sent.map_or(now, |at| at + self.period));
        stats.into_iter().chain(progress).min()
    }

    /// Everything that waits, due or not (the app is ending).
    fn take_all(&mut self) -> Vec<AppEvent> {
        let mut all: Vec<AppEvent> = self
            .stats
            .take()
            .map(AppEvent::RenderStats)
            .into_iter()
            .collect();
        all.extend(self.progress.drain(..).map(AppEvent::PhotoProgress));
        all
    }
}

struct ThrottleShared {
    merger: Mutex<Merger>,
    wake: Condvar,
    shutdown: AtomicBool,
}

impl ThrottleShared {
    fn merger(&self) -> MutexGuard<'_, Merger> {
        self.merger.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Sends `render_stats` and `photo_progress` at most once per period from a thread of its own (see
/// the module documentation). Pushing never waits for a send: it takes a lock for the time of a
/// copy, and wakes the thread only when nothing was waiting.
pub struct Throttle {
    shared: Arc<ThrottleShared>,
}

impl Throttle {
    /// Starts the thread. `deliver` is called on it, one event after the other, in the order the
    /// events must be sent.
    ///
    /// # Errors
    ///
    /// The error of the system when the thread cannot start.
    pub fn new(
        period: Duration,
        deliver: impl Fn(AppEvent) + Send + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(ThrottleShared {
            merger: Mutex::new(Merger::new(period)),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("ym-events".to_owned())
            .spawn(move || Self::run(&worker, &deliver))?;
        Ok(Self { shared })
    }

    /// `render_stats`: the newest one waiting goes out when its turn comes.
    pub fn stats(&self, stats: RenderStats) {
        if self.shared.merger().push_stats(stats) {
            self.shared.wake.notify_one();
        }
    }

    /// `photo_progress`: see the module documentation for what is merged.
    pub fn progress(&self, event: PhotoProgress) {
        if self.shared.merger().push_progress(event) {
            self.shared.wake.notify_one();
        }
    }

    fn run(shared: &ThrottleShared, deliver: &impl Fn(AppEvent)) {
        let mut merger = shared.merger();
        loop {
            if shared.shutdown.load(Ordering::SeqCst) {
                let rest = merger.take_all();
                drop(merger);
                rest.into_iter().for_each(deliver);
                return;
            }
            let now = Instant::now();
            let due = merger.take_due(now);
            if !due.is_empty() {
                // Not under the lock: a send may take a while, and pushing must not wait for it.
                drop(merger);
                due.into_iter().for_each(deliver);
                merger = shared.merger();
                continue;
            }
            merger = match merger.next_due(now) {
                Some(at) => {
                    shared
                        .wake
                        .wait_timeout(merger, at.saturating_duration_since(now))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
                None => shared
                    .wake
                    .wait(merger)
                    .unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

impl Drop for Throttle {
    /// Stops the thread after it has sent what waits.
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // Under the lock, so that the thread is either before its check or already waiting.
        drop(self.shared.merger());
        self.shared.wake.notify_all();
    }
}

/// Keeps every event, for tests.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct Recorder {
    events: std::sync::Mutex<Vec<AppEvent>>,
}

#[cfg(test)]
impl Recorder {
    /// An empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything recorded so far.
    #[must_use]
    pub fn events(&self) -> Vec<AppEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Waits until an event for which `pick` returns a value arrives, up to `timeout`.
    #[must_use]
    pub fn wait_for<T>(
        &self,
        timeout: std::time::Duration,
        mut pick: impl FnMut(&AppEvent) -> Option<T>,
    ) -> Option<T> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(found) = self.events().iter().find_map(&mut pick) {
                return Some(found);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
impl Emitter for Recorder {
    fn emit(&self, event: AppEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{ColorSpace, Num, PhotoStage, RenderPath};
    use std::sync::mpsc;

    fn stats(ms: f64) -> RenderStats {
        RenderStats {
            ms: Num(ms),
            path: RenderPath::Classical,
            space: ColorSpace::DisplayP3,
            zoom_percent: None,
            fit_percent: None,
        }
    }

    fn stats_ms(e: &AppEvent) -> f64 {
        match e {
            AppEvent::RenderStats(s) => s.ms.0,
            other => panic!("not render_stats: {other:?}"),
        }
    }

    fn progress(id: u32, stage: PhotoStage, done: bool) -> PhotoProgress {
        PhotoProgress {
            id: PhotoId(id),
            stage,
            done,
            error: None,
        }
    }

    fn failed(id: u32, stage: PhotoStage) -> PhotoProgress {
        PhotoProgress {
            error: Some("Could not be read.".to_owned()),
            ..progress(id, stage, true)
        }
    }

    fn progress_of(events: Vec<AppEvent>) -> Vec<PhotoProgress> {
        events
            .into_iter()
            .map(|e| match e {
                AppEvent::PhotoProgress(p) => p,
                other => panic!("not photo_progress: {other:?}"),
            })
            .collect()
    }

    const PERIOD: Duration = Duration::from_millis(33);

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn render_stats_are_latest_wins_and_go_out_once_per_period() {
        let t0 = Instant::now();
        let mut m = Merger::new(PERIOD);
        // After a quiet period the first one is due at once, and only the last of a burst is kept.
        for i in 0..1000 {
            assert_eq!(
                m.push_stats(stats(f64::from(i))),
                i == 0,
                "only the first wakes the thread"
            );
        }
        let due = m.take_due(t0);
        assert_eq!(due.len(), 1);
        assert_eq!(stats_ms(&due[0]), 999.0);
        // The next one waits for the period, and the thread is told when.
        m.push_stats(stats(1000.0));
        assert!(m.take_due(t0 + ms(10)).is_empty());
        assert_eq!(m.next_due(t0 + ms(10)), Some(t0 + PERIOD));
        assert!(m.take_due(t0 + ms(32)).is_empty());
        let due = m.take_due(t0 + ms(33));
        assert_eq!(due.len(), 1);
        assert_eq!(stats_ms(&due[0]), 1000.0);
        assert_eq!(m.next_due(t0 + ms(33)), None, "nothing waits");

        // A second of frames, one a millisecond, taken whenever the thread looks (every 5 ms).
        let mut m = Merger::new(PERIOD);
        let mut sent: Vec<(Duration, f64)> = Vec::new();
        for step in 0..1000_u64 {
            m.push_stats(stats(step as f64));
            if step % 5 == 0 {
                for e in m.take_due(t0 + ms(step)) {
                    sent.push((ms(step), stats_ms(&e)));
                }
            }
        }
        for e in m.take_due(t0 + ms(2000)) {
            sent.push((ms(2000), stats_ms(&e)));
        }
        assert!(
            sent.len() <= 1000 / 33 + 2,
            "{} sends for 1000 frames",
            sent.len()
        );
        assert!(sent.len() >= 20, "{} sends", sent.len());
        assert!(
            sent.windows(2).all(|w| w[1].0 >= w[0].0 + PERIOD),
            "two sends closer than the period: {sent:?}"
        );
        assert!(
            sent.windows(2).all(|w| w[1].1 > w[0].1),
            "an older value after a newer one"
        );
        assert_eq!(
            sent.last().map(|s| s.1),
            Some(999.0),
            "the last value arrives"
        );
    }

    #[test]
    fn progress_that_only_says_a_stage_started_is_replaced_and_a_finished_stage_is_kept() {
        use PhotoStage::{Decode, Estimate, Probe, Thumbnail};
        let mut m = Merger::new(PERIOD);
        let script = [
            progress(1, Probe, false),
            progress(1, Decode, false),
            progress(1, Decode, true),
            progress(1, Estimate, false),
            progress(1, Estimate, true),
            progress(1, Thumbnail, false),
            progress(1, Thumbnail, true),
        ];
        for (i, e) in script.into_iter().enumerate() {
            assert_eq!(m.push_progress(e), i == 0);
        }
        let sent = progress_of(m.take_due(Instant::now()));
        let stages: Vec<_> = sent.iter().map(|p| (p.stage, p.done)).collect();
        assert_eq!(
            stages,
            [(Decode, true), (Estimate, true), (Thumbnail, true)],
            "every finished stage arrives, the started ones are replaced"
        );

        // A failure is kept, and a photo that is developed again afterwards (a re-run) is told
        // after it, never before.
        let mut m = Merger::new(PERIOD);
        m.push_progress(progress(2, Decode, false));
        m.push_progress(failed(2, Decode));
        m.push_progress(progress(2, Decode, false));
        m.push_progress(progress(2, Decode, false));
        let sent = progress_of(m.take_due(Instant::now()));
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert!(sent[0].error.is_some() && sent[1].error.is_none() && !sent[1].done);

        // Ready, then developing again: in that order.
        let mut m = Merger::new(PERIOD);
        m.push_progress(progress(3, Thumbnail, true));
        m.push_progress(progress(3, Decode, false));
        let sent = progress_of(m.take_due(Instant::now()));
        assert_eq!(
            sent.iter().map(|p| (p.stage, p.done)).collect::<Vec<_>>(),
            [(Thumbnail, true), (Decode, false)]
        );
    }

    #[test]
    fn progress_of_other_photos_does_not_merge() {
        use PhotoStage::Decode;
        let mut m = Merger::new(PERIOD);
        for id in 1..=5 {
            m.push_progress(progress(id, Decode, false));
        }
        let sent = progress_of(m.take_due(Instant::now()));
        assert_eq!(
            sent.iter().map(|p| p.id.0).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
    }

    /// A small deterministic generator (the tests do not need `rand`).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, below: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % below
        }
    }

    #[test]
    fn whatever_the_order_nothing_terminal_is_dropped_and_the_last_word_is_the_last_sent() {
        use PhotoStage::{Decode, Estimate, Probe, Thumbnail};
        let stages = [Probe, Decode, Estimate, Thumbnail];
        let t0 = Instant::now();
        for seed in 0..300_u64 {
            let mut rng = Lcg(seed * 7919 + 1);
            let mut m = Merger::new(PERIOD);
            let mut pushed: Vec<PhotoProgress> = Vec::new();
            let mut sent: Vec<PhotoProgress> = Vec::new();
            let mut now = t0;
            for _ in 0..rng.next(60) + 1 {
                let id = u32::try_from(rng.next(4)).expect("small") + 1;
                let stage = stages[usize::try_from(rng.next(4)).expect("small")];
                let event = match rng.next(6) {
                    0 => failed(id, stage),
                    1 | 2 => progress(id, stage, true),
                    _ => progress(id, stage, false),
                };
                pushed.push(event.clone());
                m.push_progress(event);
                now += ms(rng.next(40));
                if rng.next(3) == 0 {
                    sent.extend(progress_of(m.take_due(now)));
                }
            }
            sent.extend(progress_of(m.take_all()));

            for id in 1..=4 {
                let pushed: Vec<_> = pushed.iter().filter(|p| p.id.0 == id).collect();
                let sent: Vec<_> = sent.iter().filter(|p| p.id.0 == id).collect();
                // Every terminal event arrives, in the order it was pushed.
                let terminal = |v: &[&PhotoProgress]| -> Vec<PhotoProgress> {
                    v.iter()
                        .filter(|p| is_terminal(p))
                        .map(|p| (*p).clone())
                        .collect()
                };
                assert_eq!(
                    terminal(&sent),
                    terminal(&pushed),
                    "seed {seed}, photo {id}"
                );
                // What is sent is some of what was pushed, in order, ending with the last of it.
                let mut at = 0;
                for s in &sent {
                    at +=
                        pushed[at..].iter().position(|p| p == s).unwrap_or_else(|| {
                            panic!("seed {seed}: {s:?} was never pushed in order")
                        }) + 1;
                }
                assert_eq!(sent.last(), pushed.last(), "seed {seed}, photo {id}");
            }
        }
    }

    /// Records what a [`Throttle`] delivers, with the time it did.
    fn recording(period: Duration) -> (Throttle, mpsc::Receiver<(Instant, AppEvent)>) {
        let (tx, rx) = mpsc::channel();
        let throttle = Throttle::new(period, move |e| {
            let _ = tx.send((Instant::now(), e));
        })
        .expect("the thread starts");
        (throttle, rx)
    }

    #[test]
    fn a_thousand_render_stats_make_a_few_sends_and_the_last_one_arrives() {
        let period = ms(40);
        let (throttle, rx) = recording(period);
        let started = Instant::now();
        for i in 0..1000 {
            throttle.stats(stats(f64::from(i)));
        }
        let mut got: Vec<(Instant, f64)> = Vec::new();
        while got.last().map(|g| g.1) != Some(999.0) {
            let (at, e) = rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the last value arrives");
            got.push((at, stats_ms(&e)));
        }
        let elapsed = got.last().expect("one").0 - started;
        assert!(
            got.len() as u128 <= elapsed.as_millis() / period.as_millis() + 2,
            "{} sends in {elapsed:?}",
            got.len()
        );
        assert!(got.len() < 20, "{} sends for 1000 frames", got.len());
        assert!(got.windows(2).all(|w| w[1].1 > w[0].1), "{got:?}");
        // Nothing follows.
        assert!(rx.recv_timeout(ms(150)).is_err());
    }

    #[test]
    fn an_event_after_a_quiet_period_is_not_held_back() {
        // A period far longer than anything a loaded machine adds, so that "at once" and "after the
        // period" cannot be mistaken for each other.
        let (throttle, rx) = recording(Duration::from_secs(2));
        let asked = Instant::now();
        throttle.stats(stats(1.0));
        let (at, _) = rx.recv_timeout(Duration::from_secs(5)).expect("sent");
        assert!(
            at - asked < Duration::from_secs(1),
            "the first one waited {:?}",
            at - asked
        );
        // The next one, within the period, waits for it.
        throttle.stats(stats(2.0));
        let (later, e) = rx.recv_timeout(Duration::from_secs(10)).expect("sent");
        assert_eq!(stats_ms(&e), 2.0);
        assert!(
            later - at >= Duration::from_secs(1),
            "{:?} after the first",
            later - at
        );
    }

    #[test]
    fn terminal_photo_states_always_arrive_when_hundreds_of_photos_report_at_once() {
        use PhotoStage::{Decode, Estimate, Probe, Thumbnail};
        let (throttle, rx) = recording(ms(20));
        let photos = 300_u32;
        // Four threads of a decode pool report their photos, each stage twice.
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let throttle = &throttle;
                scope.spawn(move || {
                    for id in (1..=photos).filter(|id| id % 4 == worker) {
                        for stage in [Probe, Decode, Estimate, Thumbnail] {
                            throttle.progress(progress(id, stage, false));
                            throttle.progress(progress(id, stage, true));
                        }
                    }
                });
            }
        });
        let pushed = usize::try_from(photos).expect("small") * 8;
        let mut got: Vec<PhotoProgress> = Vec::new();
        let ready = |got: &[PhotoProgress]| {
            (1..=photos).all(|id| {
                got.iter()
                    .any(|p| p.id.0 == id && p.stage == Thumbnail && p.done)
            })
        };
        while !ready(&got) {
            let (_, e) = rx
                .recv_timeout(Duration::from_secs(10))
                .expect("every photo is reported ready");
            got.extend(progress_of(vec![e]));
        }
        for id in 1..=photos {
            let done: Vec<_> = got
                .iter()
                .filter(|p| p.id.0 == id && p.done)
                .map(|p| p.stage)
                .collect();
            assert_eq!(done, [Probe, Decode, Estimate, Thumbnail], "photo {id}");
            let last = got.iter().rev().find(|p| p.id.0 == id).expect("some");
            assert!(last.done && last.stage == Thumbnail, "photo {id}: {last:?}");
        }
        assert!(
            got.len() < pushed,
            "{} sends for {pushed} events: nothing was merged",
            got.len()
        );
    }

    #[test]
    fn what_waits_is_sent_when_the_throttle_is_dropped() {
        let (throttle, rx) = recording(Duration::from_secs(60));
        // The first of each kind goes out at once; the second waits for a minute.
        throttle.stats(stats(1.0));
        throttle.progress(progress(1, PhotoStage::Decode, false));
        for _ in 0..2 {
            rx.recv_timeout(Duration::from_secs(5))
                .expect("the first of each");
        }
        throttle.stats(stats(2.0));
        throttle.progress(progress(1, PhotoStage::Thumbnail, true));
        assert!(rx.recv_timeout(ms(100)).is_err(), "still waiting");
        drop(throttle);
        let mut rest = Vec::new();
        while let Ok((_, e)) = rx.recv_timeout(Duration::from_secs(5)) {
            rest.push(e);
        }
        assert_eq!(rest.len(), 2, "{rest:?}");
        assert!(
            rest.iter()
                .any(|e| matches!(e, AppEvent::RenderStats(s) if s.ms.0 == 2.0))
        );
        assert!(rest.iter().any(
            |e| matches!(e, AppEvent::PhotoProgress(p) if p.stage == PhotoStage::Thumbnail && p.done)
        ));
    }
}
