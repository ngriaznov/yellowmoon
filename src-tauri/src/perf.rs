//! Performance instrumentation: where the time goes when the interface feels sluggish.
//!
//! On in development builds, and in a release build with `YM_PERF=1` (`YM_PERF=0` turns it off
//! again). Everything goes to the local log (`target: "ym::perf"`, `docs/13-testing.md` §Performance
//! logging); nothing is sent anywhere (hard rule 7). With it off, every hook is a flag check.
//!
//! Three things are measured:
//!
//! * **The main thread.** [`watch_main_thread`] posts a closure to it every 4 ms and times how long
//!   it waited. A wait of 33 ms or more is a missed frame at 30 Hz and is logged at once; the
//!   histogram of all waits is logged every 5 s.
//! * **The viewport.** [`viewport`] counts the requests the worker gets, the rounds it runs, the
//!   frames it presents and the rounds that were cancelled, and keeps the time from a request to
//!   the frame that shows it (p50 and p95), the time the native sink waits for a drawable, the
//!   time it spends uploading a frame and the time it takes to record and submit its pass. The
//!   native sink also counts how it drew each frame ([`Drawn`]): straight from the core's device
//!   (zero-copy, no upload) or uploaded from the host.
//! * **Handlers.** `viewport_set_rect` (it runs on the main thread) and the `ym-frame` handler are
//!   timed with [`timer`].
//!
//! One line per 5 s window, only when something happened in it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// How often the counters and the main-thread histogram are logged.
pub const REPORT_EVERY: Duration = Duration::from_secs(5);

/// How often the watchdog posts a closure to the main thread.
const PROBE_EVERY: Duration = Duration::from_millis(4);

/// How long the watchdog waits for the main thread before it gives up on one probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// A main-thread wait at or above this is logged when it happens.
pub const STALL: Duration = Duration::from_millis(33);

/// A handler on the main thread that takes at least this long is logged when it happens.
const SLOW_HANDLER: Duration = Duration::from_millis(2);

/// The upper edges, in milliseconds, of the buckets of the main-thread histogram; the last bucket
/// is everything slower.
const EDGES_MS: [u128; 5] = [4, 8, 16, 33, 100];

/// How many request-to-present samples a window keeps (more are dropped, not the earliest).
const MAX_SAMPLES: usize = 4096;

/// The target of every line this module logs.
const TARGET: &str = "ym::perf";

/// Whether instrumentation is on.
#[must_use]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled_by(
            std::env::var("YM_PERF").ok().as_deref(),
            cfg!(debug_assertions),
        )
    })
}

/// `YM_PERF` decides when it is set (anything but empty and `0` is on); otherwise development
/// builds are on and release builds off.
fn enabled_by(variable: Option<&str>, development_build: bool) -> bool {
    match variable {
        Some(value) => !(value.is_empty() || value == "0"),
        None => development_build,
    }
}

/// Whether the page is told to measure itself (`src/perf.ts`). Only when `YM_PERF` asks for it by
/// name, unlike [`enabled`]: a development build logs from Rust by default, but the page's own
/// measuring wraps the command bridge and watches every animation frame, and a person running
/// `tauri dev` did not ask for that. The Mac viewport job starts the app with `YM_PERF=1` and has
/// no address to add `?perf` to and no `localStorage` to fill, so the shell sets the flag for it.
#[must_use]
pub fn page_requested() -> bool {
    page_requested_by(std::env::var("YM_PERF").ok().as_deref())
}

/// `YM_PERF` set to anything but empty and `0` (the same reading as [`enabled_by`]).
fn page_requested_by(variable: Option<&str>) -> bool {
    variable.is_some_and(|value| !(value.is_empty() || value == "0"))
}

/// What the page reads before its first script runs: `perfRequested` in `src/perf.ts` takes
/// `window.__YM_PERF__ === true` as a request to measure.
const PAGE_FLAG_SCRIPT: &str = "window.__YM_PERF__ = true;";

/// A plugin with no commands whose only job is an initialisation script: Tauri runs it in the main
/// frame of every webview before the page's own scripts (the window of `tauri.conf.json`
/// included, which a script added after the window exists would be too late for). It sets the
/// flag [`page_requested`] stands for; add it to the builder only when that is true.
#[must_use]
pub fn page_plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("ym-perf")
        .js_init_script(PAGE_FLAG_SCRIPT)
        .build()
}

// ---------------------------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------------------------

/// What a [`Timer`] measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timed {
    /// The `viewport_set_rect` command, on the main thread.
    SetRect,
    /// A request of the `ym-frame` scheme, from the handler being called (main thread) to the
    /// response being ready (blocking thread): the wait for a thread and the work of the handler.
    FrameHandler,
    /// The native sink asking the surface for a drawable (it waits for the display).
    Acquire,
    /// The native sink uploading a frame to its texture.
    Upload,
    /// The native sink recording its pass into the drawable and submitting it (with the present).
    Pass,
}

impl Timed {
    /// Every kind, in the order of the log line.
    const ALL: [Self; 5] = [
        Self::SetRect,
        Self::FrameHandler,
        Self::Acquire,
        Self::Upload,
        Self::Pass,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::SetRect => "set_rect",
            Self::FrameHandler => "ym-frame",
            Self::Acquire => "acquire",
            Self::Upload => "upload",
            Self::Pass => "pass",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::SetRect => 0,
            Self::FrameHandler => 1,
            Self::Acquire => 2,
            Self::Upload => 3,
            Self::Pass => 4,
        }
    }
}

/// How the native sink drew a frame.
// Only the native sink (macOS) draws.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drawn {
    /// Straight from the buffer the core's device wrote it to: no upload, no copy through the
    /// processor.
    ZeroCopy,
    /// From host bytes, uploaded to a texture first.
    Uploaded,
}

/// Count, total and worst of a kind of timing, for lock-free recording.
#[derive(Debug)]
struct Stat {
    count: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl Stat {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
        }
    }

    fn add(&self, d: Duration) {
        let ns = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn take(&self) -> StatWindow {
        StatWindow {
            count: self.count.swap(0, Ordering::Relaxed),
            total_ns: self.total_ns.swap(0, Ordering::Relaxed),
            max_ns: self.max_ns.swap(0, Ordering::Relaxed),
        }
    }
}

/// One window of a [`Stat`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StatWindow {
    /// How many were timed.
    pub count: u64,
    /// Their total, in nanoseconds.
    pub total_ns: u64,
    /// The longest, in nanoseconds.
    pub max_ns: u64,
}

impl StatWindow {
    /// The mean, in milliseconds (0 for none).
    #[must_use]
    pub fn mean_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1e6
        }
    }

    /// The longest, in milliseconds.
    #[must_use]
    pub fn max_ms(&self) -> f64 {
        self.max_ns as f64 / 1e6
    }
}

/// The viewport's counters. Lock-free except for the request-to-present samples. The shared
/// instance ([`viewport`]) records only when [`enabled`]; an instance made with
/// [`Counters::new`] always records (tests).
#[derive(Debug)]
pub struct Counters {
    gated: bool,
    requests: AtomicU64,
    rounds: AtomicU64,
    presented: AtomicU64,
    cancelled: AtomicU64,
    tiles_cancelled: AtomicU64,
    zero_copy: AtomicU64,
    uploaded: AtomicU64,
    latencies_us: Mutex<Vec<u32>>,
    timed: [Stat; 5],
}

impl Counters {
    const fn make(gated: bool) -> Self {
        Self {
            gated,
            requests: AtomicU64::new(0),
            rounds: AtomicU64::new(0),
            presented: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            tiles_cancelled: AtomicU64::new(0),
            zero_copy: AtomicU64::new(0),
            uploaded: AtomicU64::new(0),
            latencies_us: Mutex::new(Vec::new()),
            timed: [
                Stat::new(),
                Stat::new(),
                Stat::new(),
                Stat::new(),
                Stat::new(),
            ],
        }
    }

    /// Counters that always record.
    #[must_use]
    pub const fn new() -> Self {
        Self::make(false)
    }

    fn live(&self) -> bool {
        !self.gated || enabled()
    }

    /// The webview asked for a frame (an input, a settings patch, a cell change).
    pub fn request(&self) {
        if self.live() {
            self.requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The worker started a round.
    pub fn round(&self) {
        if self.live() {
            self.rounds.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A frame reached the sink; `latency` is the time since the oldest request it answers.
    pub fn present(&self, latency: Option<Duration>) {
        if !self.live() {
            return;
        }
        self.presented.fetch_add(1, Ordering::Relaxed);
        if let Some(latency) = latency {
            let mut samples = self
                .latencies_us
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if samples.len() < MAX_SAMPLES {
                samples.push(u32::try_from(latency.as_micros()).unwrap_or(u32::MAX));
            }
        }
    }

    /// A round ended without a frame because it was cancelled.
    pub fn cancelled(&self) {
        if self.live() {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A sharp-tile pass was given up for a newer request (expected, and cheap).
    pub fn tiles_cancelled(&self) {
        if self.live() {
            self.tiles_cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The native sink drew a frame, the way `how` says.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn drawn(&self, how: Drawn) {
        if self.live() {
            match how {
                Drawn::ZeroCopy => &self.zero_copy,
                Drawn::Uploaded => &self.uploaded,
            }
            .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Adds one timing.
    pub fn time(&self, kind: Timed, d: Duration) {
        if self.live() {
            self.timed[kind.index()].add(d);
        }
    }

    /// Takes the counts since the last call and starts a new window.
    pub fn take(&self) -> Window {
        let mut latency_us = std::mem::take(
            &mut *self
                .latencies_us
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        latency_us.sort_unstable();
        Window {
            requests: self.requests.swap(0, Ordering::Relaxed),
            rounds: self.rounds.swap(0, Ordering::Relaxed),
            presented: self.presented.swap(0, Ordering::Relaxed),
            cancelled: self.cancelled.swap(0, Ordering::Relaxed),
            tiles_cancelled: self.tiles_cancelled.swap(0, Ordering::Relaxed),
            zero_copy: self.zero_copy.swap(0, Ordering::Relaxed),
            uploaded: self.uploaded.swap(0, Ordering::Relaxed),
            latency_us,
            timed: Timed::ALL.map(|kind| self.timed[kind.index()].take()),
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

static VIEWPORT: Counters = Counters::make(true);

/// The viewport's counters, shared by the controller, the sink and the handlers.
#[must_use]
pub fn viewport() -> &'static Counters {
    &VIEWPORT
}

/// What [`Counters::take`] returns: one window of counts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Window {
    /// Requests for a frame.
    pub requests: u64,
    /// Rounds the worker ran.
    pub rounds: u64,
    /// Frames presented.
    pub presented: u64,
    /// Rounds that ended without a frame, cancelled.
    pub cancelled: u64,
    /// Sharp-tile passes given up for a newer request.
    pub tiles_cancelled: u64,
    /// Frames the native sink drew straight from the core's device ([`Drawn::ZeroCopy`]).
    pub zero_copy: u64,
    /// Frames the native sink uploaded from the host first ([`Drawn::Uploaded`]).
    pub uploaded: u64,
    /// Request-to-present times in microseconds, sorted.
    pub latency_us: Vec<u32>,
    /// Timings, in the order of [`Timed::ALL`].
    pub timed: [StatWindow; 5],
}

impl Window {
    /// Nothing happened in the window.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.requests == 0
            && self.rounds == 0
            && self.presented == 0
            && self.cancelled == 0
            && self.tiles_cancelled == 0
            && self.zero_copy == 0
            && self.uploaded == 0
            && self.timed.iter().all(|t| t.count == 0)
    }

    /// The request-to-present time at `percent` (1 to 100), in milliseconds, by the nearest-rank
    /// rule; `None` when there are no samples.
    #[must_use]
    pub fn latency_ms(&self, percent: usize) -> Option<f64> {
        let n = self.latency_us.len();
        if n == 0 {
            return None;
        }
        let rank = (percent * n).div_ceil(100).clamp(1, n);
        Some(f64::from(self.latency_us[rank - 1]) / 1000.0)
    }

    /// The window as one log line.
    #[must_use]
    pub fn line(&self) -> String {
        let mut line = format!(
            "viewport: requests {}, rounds {}, presented {}, cancelled {}, tiles cancelled {}",
            self.requests, self.rounds, self.presented, self.cancelled, self.tiles_cancelled
        );
        match (self.latency_ms(50), self.latency_ms(95)) {
            (Some(p50), Some(p95)) => {
                line.push_str(&format!(
                    ", request to present p50 {p50:.1} ms p95 {p95:.1} ms (n {})",
                    self.latency_us.len()
                ));
            }
            _ => line.push_str(", request to present none"),
        }
        // Only a native sink draws; the fallback viewport's lines leave this out.
        if self.zero_copy > 0 || self.uploaded > 0 {
            line.push_str(&format!(
                ", drawn zero-copy {} uploaded {}",
                self.zero_copy, self.uploaded
            ));
        }
        for (kind, stat) in Timed::ALL.into_iter().zip(&self.timed) {
            if stat.count > 0 {
                line.push_str(&format!(
                    ", {} mean {:.2} ms max {:.2} ms (n {})",
                    kind.name(),
                    stat.mean_ms(),
                    stat.max_ms(),
                    stat.count
                ));
            }
        }
        line
    }
}

/// Logs the counters of the window so far and starts a new one, if anything happened in it.
/// Returns whether anything did.
pub fn log_viewport_window() -> bool {
    let window = viewport().take();
    if window.is_idle() {
        return false;
    }
    tracing::info!(target: TARGET, "{}", window.line());
    true
}

/// Times a stretch of code: the timing is recorded when the value is dropped. Does nothing when
/// instrumentation is off. A [`Timed::SetRect`] of 2 ms or more is logged at once, because that
/// one runs on the main thread.
#[must_use = "the timing is taken when the timer is dropped"]
pub struct Timer {
    kind: Timed,
    start: Option<Instant>,
}

/// Starts timing `kind`.
pub fn timer(kind: Timed) -> Timer {
    Timer {
        kind,
        start: enabled().then(Instant::now),
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let took = start.elapsed();
        viewport().time(self.kind, took);
        if self.kind == Timed::SetRect && took >= SLOW_HANDLER {
            tracing::warn!(
                target: TARGET,
                ms = took.as_secs_f64() * 1000.0,
                "viewport_set_rect held the main thread"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The main thread
// ---------------------------------------------------------------------------------------------

/// How long the main thread took to answer, as a histogram with the edges of [`EDGES_MS`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LagHistogram {
    counts: [u32; 6],
    worst: Duration,
}

impl LagHistogram {
    /// An empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one wait; returns true when it counts as a stall ([`STALL`] or more).
    pub fn record(&mut self, lag: Duration) -> bool {
        let bucket = EDGES_MS
            .iter()
            .position(|&edge| lag.as_millis() < edge)
            .unwrap_or(EDGES_MS.len());
        self.counts[bucket] += 1;
        self.worst = self.worst.max(lag);
        lag >= STALL
    }

    /// How many waits there were.
    #[cfg(test)]
    #[must_use]
    pub fn total(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// The longest wait.
    #[must_use]
    pub fn worst(&self) -> Duration {
        self.worst
    }

    /// The waits in each bucket: below each edge of [`EDGES_MS`], then the slower ones.
    #[cfg(test)]
    #[must_use]
    pub fn counts(&self) -> [u32; 6] {
        self.counts
    }

    /// The histogram as one log line.
    #[must_use]
    pub fn line(&self) -> String {
        let [a, b, c, d, e, f] = self.counts;
        format!(
            "main thread: waits <4 ms {a}, <8 ms {b}, <16 ms {c}, <33 ms {d}, <100 ms {e}, slower {f}; worst {:.1} ms",
            self.worst.as_secs_f64() * 1000.0
        )
    }
}

/// Starts the watchdog thread: every 4 ms it asks the main thread to run a closure and times how
/// long that took, which is how long the page and the menu would have waited for the main thread.
/// A wait of 33 ms or more is logged at once (`warn`); every 5 s the histogram and the viewport
/// counters are logged (not when nothing happened, so a window left open on an idle photo does not
/// fill the log). The thread ends when the app does.
pub fn watch_main_thread<R: tauri::Runtime>(app: tauri::AppHandle<R>) {
    let started = std::thread::Builder::new()
        .name("ym-perf-main".to_owned())
        .spawn(move || {
            let mut histogram = LagHistogram::new();
            let mut report_at = Instant::now() + REPORT_EVERY;
            loop {
                std::thread::sleep(PROBE_EVERY);
                let (tx, rx) = std::sync::mpsc::channel();
                let posted = Instant::now();
                if app
                    .run_on_main_thread(move || {
                        let _ = tx.send(());
                    })
                    .is_err()
                {
                    return;
                }
                if rx.recv_timeout(PROBE_TIMEOUT).is_ok() {
                    let lag = posted.elapsed();
                    if histogram.record(lag) {
                        tracing::warn!(
                            target: TARGET,
                            lag_ms = u64::try_from(lag.as_millis()).unwrap_or(u64::MAX),
                            "the main thread stalled"
                        );
                    }
                }
                if Instant::now() >= report_at {
                    // The viewport line first: it is the one that says whether the app was in use.
                    let busy = log_viewport_window();
                    if busy || histogram.worst() >= Duration::from_millis(8) {
                        tracing::info!(target: TARGET, "{}", histogram.line());
                    }
                    histogram = LagHistogram::new();
                    report_at = Instant::now() + REPORT_EVERY;
                }
            }
        });
    if let Err(e) = started {
        tracing::warn!(target: TARGET, error = %e, "the main-thread watchdog did not start");
    } else {
        tracing::info!(target: TARGET, "performance logging is on");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn the_variable_decides_and_the_build_is_the_default() {
        assert!(enabled_by(None, true));
        assert!(!enabled_by(None, false));
        assert!(enabled_by(Some("1"), false));
        assert!(enabled_by(Some("yes"), false));
        assert!(
            !enabled_by(Some("0"), true),
            "YM_PERF=0 turns a debug build off"
        );
        assert!(!enabled_by(Some(""), true));
    }

    #[test]
    fn the_page_measures_only_when_the_variable_asks_for_it_by_name() {
        // Unlike `enabled_by`, a development build does not turn the page's measuring on.
        assert!(!page_requested_by(None));
        assert!(!page_requested_by(Some("")));
        assert!(!page_requested_by(Some("0")));
        assert!(page_requested_by(Some("1")));
        assert!(page_requested_by(Some("yes")));
    }

    #[test]
    fn the_plugin_hands_the_page_its_flag_before_the_page_runs() {
        use tauri::plugin::Plugin;
        let plugin = page_plugin::<tauri::Wry>();
        let script = plugin
            .initialization_script_2()
            .expect("the plugin has an initialisation script");
        assert_eq!(script.script, "window.__YM_PERF__ = true;");
        assert!(script.for_main_frame_only, "the page, not its frames");
        // `src/perf.ts` reads exactly this name, and only the value `true`.
        let reader = include_str!("../../src/perf.ts");
        assert!(reader.contains("__YM_PERF__"), "src/perf.ts reads the flag");
        assert!(reader.contains("shell === true"));
    }

    #[test]
    fn the_plugin_registers_in_a_real_builder() {
        let app = tauri::test::mock_builder()
            .plugin(page_plugin())
            .build(tauri::test::mock_context(tauri::test::noop_assets()));
        assert!(app.is_ok(), "{:?}", app.err());
    }

    #[test]
    fn a_window_counts_and_the_next_one_starts_empty() {
        let counters = Counters::new();
        for _ in 0..5 {
            counters.request();
        }
        counters.round();
        counters.round();
        counters.present(Some(ms(20)));
        counters.present(None);
        counters.cancelled();
        counters.tiles_cancelled();
        counters.time(Timed::Upload, ms(2));
        counters.time(Timed::Upload, ms(4));
        counters.drawn(Drawn::ZeroCopy);
        counters.drawn(Drawn::ZeroCopy);
        counters.drawn(Drawn::Uploaded);
        let window = counters.take();
        assert_eq!(
            (window.requests, window.rounds, window.presented),
            (5, 2, 2)
        );
        assert_eq!((window.cancelled, window.tiles_cancelled), (1, 1));
        assert_eq!((window.zero_copy, window.uploaded), (2, 1));
        assert_eq!(
            window.latency_us,
            vec![20_000],
            "a frame without a request has no latency"
        );
        let upload = window.timed[Timed::Upload.index()];
        assert_eq!(
            (upload.count, upload.total_ns, upload.max_ns),
            (2, 6_000_000, 4_000_000)
        );
        assert!((upload.mean_ms() - 3.0).abs() < 1e-9);
        assert!(!window.is_idle());

        let again = counters.take();
        assert!(again.is_idle(), "{again:?}");
        assert_eq!(again.latency_ms(50), None);

        // A window in which the sink only drew is not idle.
        counters.drawn(Drawn::Uploaded);
        assert!(!counters.take().is_idle());
    }

    #[test]
    fn percentiles_are_by_nearest_rank() {
        let counters = Counters::new();
        // 1 to 100 ms, one of each, put in out of order.
        for n in (1..=100u64).rev() {
            counters.present(Some(ms(n)));
        }
        let window = counters.take();
        assert_eq!(window.latency_ms(50), Some(50.0));
        assert_eq!(window.latency_ms(95), Some(95.0));
        assert_eq!(window.latency_ms(100), Some(100.0));
        assert_eq!(window.latency_ms(1), Some(1.0));

        let one = Counters::new();
        one.present(Some(ms(7)));
        let one = one.take();
        assert_eq!(
            (one.latency_ms(50), one.latency_ms(95)),
            (Some(7.0), Some(7.0))
        );
    }

    #[test]
    fn a_window_reads_as_one_line() {
        let counters = Counters::new();
        counters.request();
        counters.round();
        counters.present(Some(ms(12)));
        counters.time(Timed::Acquire, ms(6));
        counters.time(Timed::Pass, ms(1));
        counters.drawn(Drawn::ZeroCopy);
        let line = counters.take().line();
        for part in [
            "requests 1",
            "rounds 1",
            "presented 1",
            "cancelled 0",
            "p50 12.0 ms",
            "p95 12.0 ms",
            "drawn zero-copy 1 uploaded 0",
            "acquire mean 6.00 ms max 6.00 ms (n 1)",
            "pass mean 1.00 ms max 1.00 ms (n 1)",
        ] {
            assert!(line.contains(part), "{part} is not in {line}");
        }
        assert!(
            !line.contains("upload mean"),
            "a kind that was not timed is left out: {line}"
        );
        assert!(
            !Counters::new().take().line().contains("drawn"),
            "a sink that drew nothing (the fallback viewport) is left out"
        );
        assert!(!line.contains('\n'));
        assert!(
            Counters::new()
                .take()
                .line()
                .contains("request to present none")
        );
    }

    #[test]
    fn the_samples_of_a_window_are_bounded() {
        let counters = Counters::new();
        for _ in 0..MAX_SAMPLES + 100 {
            counters.present(Some(ms(1)));
        }
        let window = counters.take();
        assert_eq!(window.presented, (MAX_SAMPLES + 100) as u64);
        assert_eq!(window.latency_us.len(), MAX_SAMPLES);
    }

    #[test]
    fn the_histogram_buckets_waits_and_flags_stalls() {
        let mut h = LagHistogram::new();
        let waits = [0, 3, 4, 7, 8, 15, 16, 32, 33, 99, 100, 5_000];
        let stalls: Vec<bool> = waits.iter().map(|&w| h.record(ms(w))).collect();
        assert_eq!(h.counts(), [2, 2, 2, 2, 2, 2]);
        assert_eq!(h.total(), 12);
        assert_eq!(h.worst(), ms(5_000));
        assert_eq!(
            stalls,
            [
                false, false, false, false, false, false, false, false, true, true, true, true
            ]
        );
        let line = h.line();
        assert!(
            line.contains("<4 ms 2") && line.contains("slower 2"),
            "{line}"
        );
        assert!(line.contains("worst 5000.0 ms"), "{line}");
    }

    #[test]
    fn a_timer_that_is_off_records_nothing_and_one_that_is_on_records_once() {
        let off = Timer {
            kind: Timed::SetRect,
            start: None,
        };
        drop(off);
        // The shared counters are on in a test build; only the count is checked, and only that
        // it went up (other tests of this process may time things too).
        let before = viewport().take();
        drop(before);
        {
            let _t = timer(Timed::FrameHandler);
            std::thread::sleep(ms(2));
        }
        let after = viewport().take();
        if enabled() {
            let stat = after.timed[Timed::FrameHandler.index()];
            assert!(stat.count >= 1 && stat.max_ns >= 1_000_000, "{stat:?}");
        } else {
            assert_eq!(after.timed[Timed::FrameHandler.index()].count, 0);
        }
    }
}
