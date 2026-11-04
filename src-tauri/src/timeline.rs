//! The timeline of a slider tick, from the hand to the screen (U-02, `docs/01-architecture.md` §Perf log).
//!
//! With `YM_PERF=1` the page stamps each settings patch it sends (`src/perfTimeline.ts`: when the pointer moved, when the command was
//! sent, when the reply came, when the page was painted) and reports them once a second in one `perf_report` command. The shell stamps
//! the same patch where it passes: `update_settings` arriving and leaving, the viewport asking for a frame because of it, the sink
//! handing the frame to the display. One number joins the two records: the position of the `update_settings` call in the page's life
//! (`n`), which is also the position of the call in the shell's count (`base` + `n`; the page's first report says what the shell's
//! count was when the page began). Both sides stamp the system clock in microseconds since the Unix epoch, so the stamps compare.
//!
//! The "presented" stamp is the moment the sink returned from `present`, with the drawable queued to the display (the sink uses wgpu,
//! which does not hand out Metal's drawable, so Metal's presented handler is out of reach); the picture is on the glass at most one
//! refresh later (`Fifo`, one frame of latency).
//!
//! One `tick` line per patch, one `click` line per tab, node view and menu opening, and a `window` line every 5 s (the main-thread
//! watchdog of `perf` writes it). Nothing leaves the machine (hard rule 7); with the switch off every hook is a flag check.

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

/// The target of every line (the same as `perf`).
const TARGET: &str = "ym::perf";

/// How many ticks the shell remembers, as many as the page's ring.
const RING: usize = 512;

/// How many requests for a frame wait for their frame at most.
const PENDING: usize = 256;

/// A frame later than this is over the budget of a 120 Hz display; later than [`BUDGET_60_MS`], of a 60 Hz one.
const BUDGET_120_MS: f64 = 8.3;
const BUDGET_60_MS: f64 = 16.7;

/// Microseconds since the Unix epoch.
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

/// One patch as the shell saw it. A stamp that did not happen is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShellTick {
    /// The position of the `update_settings` call in the shell's count.
    seq: u64,
    ipc_in_us: u64,
    ipc_out_us: Option<u64>,
    /// The viewport asked for a frame because of the patch.
    request_us: Option<u64>,
    present_called_us: Option<u64>,
    presented_us: Option<u64>,
}

/// What the page reports once a second (`src/perfTimeline.ts`). Times are milliseconds since the Unix epoch.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct PageReport {
    /// Names this page's life; a reloaded page has another.
    pub page: String,
    /// The first report of a page: the shell's count of `update_settings` calls is the page's zero.
    #[serde(default)]
    pub start: bool,
    /// Pointer moves seen since the last report, and animation frames the page drew while a gesture was held.
    #[serde(default)]
    pub pointer_moves: u64,
    #[serde(default)]
    pub ticks: Vec<PageTick>,
    #[serde(default)]
    pub clicks: Vec<PageClick>,
}

/// One `update_settings` call as the page saw it.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct PageTick {
    /// The position of the call in the page's life, from 1.
    pub n: u64,
    /// The release, not a frame of the drag.
    #[serde(default, rename = "final")]
    pub is_final: bool,
    /// The first pointer move the patch answers; `null` for a patch that no pointer made (a key, a typed value).
    pub t_input: Option<f64>,
    pub t_sent: f64,
    pub t_reply: Option<f64>,
    pub t_painted: Option<f64>,
}

/// One click of a tab, the node view or a menu, as the page saw it.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct PageClick {
    pub kind: String,
    #[serde(default)]
    pub label: String,
    pub t_input: f64,
    pub t_painted: Option<f64>,
}

/// The window the 5 s line is made from.
#[derive(Debug, Default)]
struct Aggregate {
    input_to_presented_ms: Vec<f64>,
    pointer_moves: u64,
    frames_shown: u64,
    unmatched: u64,
}

#[derive(Debug, Default)]
struct Inner {
    /// How many `update_settings` calls arrived.
    calls: u64,
    /// The shell's count when the current page began, and that page.
    page: Option<(String, u64)>,
    ticks: VecDeque<ShellTick>,
    /// Requests for a frame that have no frame yet, in microseconds.
    pending: Vec<u64>,
    round_started_us: u64,
    last_present_called_us: u64,
    window: Aggregate,
}

/// The shell's side of the timeline.
#[derive(Debug, Default)]
pub struct Timeline {
    inner: Mutex<Inner>,
}

/// The stamp `update_settings` holds while it works; dropping it stamps the way out.
#[must_use = "the way out is stamped when this is dropped"]
pub struct TickGuard {
    seq: u64,
}

impl Timeline {
    /// An empty timeline.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                calls: 0,
                page: None,
                ticks: VecDeque::new(),
                pending: Vec::new(),
                round_started_us: 0,
                last_present_called_us: 0,
                window: Aggregate {
                    input_to_presented_ms: Vec::new(),
                    pointer_moves: 0,
                    frames_shown: 0,
                    unmatched: 0,
                },
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `update_settings` arrived at `at_us`.
    pub fn begin_update_at(&self, at_us: u64) -> TickGuard {
        let mut inner = self.lock();
        inner.calls += 1;
        let seq = inner.calls;
        if inner.ticks.len() >= RING {
            inner.ticks.pop_front();
        }
        inner.ticks.push_back(ShellTick {
            seq,
            ipc_in_us: at_us,
            ipc_out_us: None,
            request_us: None,
            present_called_us: None,
            presented_us: None,
        });
        TickGuard { seq }
    }

    /// `update_settings` is leaving at `at_us`: the first request for a frame since it arrived is the patch's.
    pub fn end_update_at(&self, guard: &TickGuard, at_us: u64) {
        let mut inner = self.lock();
        let pending = inner.pending.clone();
        if let Some(tick) = inner.ticks.iter_mut().rev().find(|t| t.seq == guard.seq) {
            tick.ipc_out_us = Some(at_us);
            tick.request_us = pending.into_iter().find(|&r| r >= tick.ipc_in_us);
        }
    }

    /// The viewport was asked for a frame at `at_us`.
    pub fn request_at(&self, at_us: u64) {
        let mut inner = self.lock();
        if inner.pending.len() >= PENDING {
            inner.pending.remove(0);
        }
        inner.pending.push(at_us);
    }

    /// The viewport worker began a round at `at_us`; the frame it presents answers the requests made before.
    pub fn round_at(&self, at_us: u64) {
        self.lock().round_started_us = at_us;
    }

    /// The native sink is about to hand the drawable to the display.
    pub fn present_called_at(&self, at_us: u64) {
        self.lock().last_present_called_us = at_us;
    }

    /// A frame reached the display (see the module note) at `at_us`.
    pub fn presented_at(&self, at_us: u64) {
        let mut inner = self.lock();
        inner.window.frames_shown += 1;
        let round = inner.round_started_us;
        let called = inner.last_present_called_us;
        for tick in inner.ticks.iter_mut().rev() {
            if tick.presented_us.is_some() {
                // Older ticks were answered by older frames.
                break;
            }
            if tick.request_us.is_some_and(|r| r <= round) {
                tick.presented_us = Some(at_us);
                tick.present_called_us = (called >= round).then_some(called);
            }
        }
        inner.pending.retain(|&r| r > round);
    }

    /// Joins a report of the page with the shell's stamps. Returns the lines to log (one per tick and per click).
    pub fn merge(&self, report: &PageReport) -> Vec<String> {
        let mut inner = self.lock();
        if report.start || inner.page.as_ref().is_none_or(|(id, _)| *id != report.page) {
            let base = if report.start { inner.calls } else { 0 };
            inner.page = Some((report.page.clone(), base));
        }
        let base = inner.page.as_ref().map_or(0, |(_, base)| *base);
        inner.window.pointer_moves += report.pointer_moves;
        let mut lines = Vec::new();
        for tick in &report.ticks {
            let seq = base + tick.n;
            let shell = inner.ticks.iter().rev().find(|t| t.seq == seq).copied();
            let Some(shell) = shell else {
                inner.window.unmatched += 1;
                lines.push(format!(
                    "tick seq={seq} unmatched (the shell's ring no longer holds it, or the page began before the shell was counting)"
                ));
                continue;
            };
            let (line, input_to_presented) = tick_line(seq, tick, &shell);
            if let Some(ms) = input_to_presented {
                inner.window.input_to_presented_ms.push(ms);
            }
            lines.push(line);
        }
        for click in &report.clicks {
            lines.push(click_line(click));
        }
        lines
    }

    /// Takes the window so far and starts a new one; `None` when nothing happened in it.
    pub fn take_window_line(&self) -> Option<String> {
        let mut inner = self.lock();
        let window = std::mem::take(&mut inner.window);
        window_line(&window)
    }
}

/// Milliseconds between two stamps in microseconds, or `None` when one is missing.
fn between(from_us: Option<u64>, to_us: Option<u64>) -> Option<f64> {
    let (from, to) = (from_us?, to_us?);
    Some((to as f64 - from as f64) / 1000.0)
}

/// Milliseconds between two page stamps (milliseconds since the epoch).
fn page_between(from: Option<f64>, to: Option<f64>) -> Option<f64> {
    Some(to? - from?)
}

fn field(name: &str, ms: Option<f64>) -> String {
    ms.map_or_else(|| format!("{name}=-"), |ms| format!("{name}={ms:.1}"))
}

/// The line for one patch and its input-to-presented time.
fn tick_line(seq: u64, page: &PageTick, shell: &ShellTick) -> (String, Option<f64>) {
    let sent_us = Some((page.t_sent * 1000.0) as u64);
    let input_us = page.t_input.map(|t| (t * 1000.0) as u64);
    let reply_us = page.t_reply.map(|t| (t * 1000.0) as u64);
    let input_to_presented = between(input_us.or(sent_us), shell.presented_us);
    let line = format!(
        "tick seq={seq} final={} {} {} {} {} {} {} {} {} {} (ms)",
        page.is_final,
        field(
            "input_to_sent",
            page_between(page.t_input, Some(page.t_sent))
        ),
        field("ipc_in", between(sent_us, Some(shell.ipc_in_us))),
        field("handle", between(Some(shell.ipc_in_us), shell.ipc_out_us)),
        field("ipc_back", between(shell.ipc_out_us, reply_us)),
        field(
            "request_to_present_called",
            between(shell.request_us, shell.present_called_us)
        ),
        field(
            "request_to_presented",
            between(shell.request_us, shell.presented_us)
        ),
        field(
            "page_painted",
            page_between(page.t_input.or(Some(page.t_sent)), page.t_painted)
        ),
        field("input_to_presented", input_to_presented),
        field("sent_to_presented", between(sent_us, shell.presented_us)),
    );
    (line, input_to_presented)
}

fn click_line(click: &PageClick) -> String {
    format!(
        "click kind={} label={:?} {} (ms)",
        click.kind,
        click.label,
        field(
            "to_painted",
            page_between(Some(click.t_input), click.t_painted)
        )
    )
}

/// The value at `percent` (1 to 100) of a sorted list, by the nearest-rank rule.
fn rank(sorted: &[f64], percent: usize) -> f64 {
    let n = sorted.len();
    let at = (percent * n).div_ceil(100).clamp(1, n);
    sorted[at - 1]
}

fn window_line(window: &Aggregate) -> Option<String> {
    if window.input_to_presented_ms.is_empty() && window.pointer_moves == 0 && window.unmatched == 0
    {
        return None;
    }
    let mut sorted = window.input_to_presented_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let spread = if sorted.is_empty() {
        "input_to_presented none".to_owned()
    } else {
        format!(
            "input_to_presented p50 {:.1} p95 {:.1} max {:.1} ms, over {BUDGET_120_MS} ms {}, over {BUDGET_60_MS} ms {}",
            rank(&sorted, 50),
            rank(&sorted, 95),
            sorted[sorted.len() - 1],
            sorted.iter().filter(|&&ms| ms > BUDGET_120_MS).count(),
            sorted.iter().filter(|&&ms| ms > BUDGET_60_MS).count(),
        )
    };
    Some(format!(
        "window 5s: ticks {}, {spread}, pointer events {}, frames shown {}{}",
        sorted.len(),
        window.pointer_moves,
        window.frames_shown,
        if window.unmatched > 0 {
            format!(", unmatched {}", window.unmatched)
        } else {
            String::new()
        }
    ))
}

static TIMELINE: Timeline = Timeline::new();

/// The shared timeline.
#[must_use]
pub fn timeline() -> &'static Timeline {
    &TIMELINE
}

/// `update_settings` arrived. `None` with the switch off.
pub fn begin_update() -> Option<TickGuard> {
    crate::perf::enabled().then(|| timeline().begin_update_at(now_us()))
}

/// `update_settings` is leaving.
pub fn end_update(guard: Option<TickGuard>) {
    if let Some(guard) = guard {
        timeline().end_update_at(&guard, now_us());
    }
}

/// The viewport was asked for a frame.
pub fn request() {
    if crate::perf::enabled() {
        timeline().request_at(now_us());
    }
}

/// The viewport worker began a round.
pub fn round() {
    if crate::perf::enabled() {
        timeline().round_at(now_us());
    }
}

/// The native sink is about to hand the drawable to the display.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn present_called() {
    if crate::perf::enabled() {
        timeline().present_called_at(now_us());
    }
}

/// A frame reached the sink and was handed over.
pub fn presented() {
    if crate::perf::enabled() {
        timeline().presented_at(now_us());
    }
}

/// Handles the page's report: logs a line per tick and click. Anything that is not a report is dropped with a note.
pub fn report(json: &str) {
    if !crate::perf::enabled() {
        return;
    }
    match serde_json::from_str::<PageReport>(json) {
        Ok(report) => {
            for line in timeline().merge(&report) {
                tracing::info!(target: TARGET, "{line}");
            }
        }
        Err(e) => tracing::warn!(target: TARGET, error = %e, "the page's report could not be read"),
    }
}

/// Logs the window since the last call, if anything happened in it. Returns whether it did.
pub fn log_window() -> bool {
    match timeline().take_window_line() {
        Some(line) => {
            tracing::info!(target: TARGET, "{line}");
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(json: &str) -> PageReport {
        serde_json::from_str(json).expect("a report")
    }

    #[test]
    fn a_patch_is_followed_from_the_hand_to_the_display() {
        let t = Timeline::new();
        // The page's first report: the shell has counted nothing yet.
        t.merge(&report(r#"{"page":"a","start":true}"#));
        let guard = t.begin_update_at(1_000_000);
        t.request_at(1_002_000);
        t.end_update_at(&guard, 1_003_000);
        t.round_at(1_004_000);
        t.present_called_at(1_010_000);
        t.presented_at(1_011_000);
        let lines = t.merge(&report(
            r#"{"page":"a","pointer_moves":3,"ticks":[{"n":1,"final":false,"t_input":995.0,"t_sent":998.0,"t_reply":1004.0,"t_painted":1020.0}]}"#,
        ));
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert!(line.starts_with("tick seq=1 final=false"), "{line}");
        assert!(line.contains("input_to_sent=3.0"), "{line}");
        assert!(line.contains("ipc_in=2.0"), "{line}");
        assert!(line.contains("handle=3.0"), "{line}");
        assert!(line.contains("ipc_back=1.0"), "{line}");
        assert!(line.contains("request_to_presented=9.0"), "{line}");
        assert!(line.contains("input_to_presented=16.0"), "{line}");
        assert!(line.contains("page_painted=25.0"), "{line}");
        let window = t.take_window_line().expect("a window line");
        assert!(
            window.contains("ticks 1, input_to_presented p50 16.0"),
            "{window}"
        );
        assert!(window.contains("over 8.3 ms 1, over 16.7 ms 0"), "{window}");
        assert!(
            window.contains("pointer events 3, frames shown 1"),
            "{window}"
        );
        assert_eq!(t.take_window_line(), None, "the next window starts empty");
    }

    #[test]
    fn a_frame_answers_only_the_requests_made_before_its_round() {
        let t = Timeline::new();
        t.merge(&report(r#"{"page":"a","start":true}"#));
        let first = t.begin_update_at(100);
        t.request_at(110);
        t.end_update_at(&first, 120);
        t.round_at(130);
        // A second patch arrives while the round draws: its request is after the round's start.
        let second = t.begin_update_at(140);
        t.request_at(150);
        t.end_update_at(&second, 160);
        t.presented_at(200);
        {
            let inner = t.lock();
            assert_eq!(inner.ticks[0].presented_us, Some(200));
            assert_eq!(
                inner.ticks[1].presented_us, None,
                "its frame is the next one"
            );
        }
        t.round_at(210);
        t.presented_at(250);
        assert_eq!(t.lock().ticks[1].presented_us, Some(250));
    }

    #[test]
    fn a_reloaded_page_counts_from_the_shells_count_at_its_start() {
        let t = Timeline::new();
        t.merge(&report(r#"{"page":"a","start":true}"#));
        for i in 0..3 {
            let guard = t.begin_update_at(1000 + i);
            t.end_update_at(&guard, 1001 + i);
        }
        // The page reloads: its first call is the shell's fourth.
        t.merge(&report(r#"{"page":"b","start":true}"#));
        let guard = t.begin_update_at(5000);
        t.end_update_at(&guard, 5001);
        let lines = t.merge(&report(r#"{"page":"b","ticks":[{"n":1,"t_sent":4.0}]}"#));
        assert!(lines[0].starts_with("tick seq=4 "), "{}", lines[0]);
        assert!(!lines[0].contains("unmatched"));
    }

    #[test]
    fn a_patch_the_ring_forgot_is_said_so() {
        let t = Timeline::new();
        t.merge(&report(r#"{"page":"a","start":true}"#));
        let lines = t.merge(&report(r#"{"page":"a","ticks":[{"n":9,"t_sent":1.0}]}"#));
        assert!(lines[0].contains("unmatched"), "{}", lines[0]);
        assert!(
            t.take_window_line()
                .expect("a line")
                .contains("unmatched 1")
        );
    }

    #[test]
    fn a_click_has_its_own_line() {
        let t = Timeline::new();
        let lines = t.merge(&report(
            r#"{"page":"a","clicks":[{"kind":"tab","label":"Colour","t_input":10.0,"t_painted":71.5}]}"#,
        ));
        assert_eq!(
            lines,
            vec![r#"click kind=tab label="Colour" to_painted=61.5 (ms)"#.to_owned()]
        );
    }

    #[test]
    fn the_ring_keeps_the_newest_patches() {
        let t = Timeline::new();
        for i in 0..(RING as u64 + 5) {
            let guard = t.begin_update_at(i);
            t.end_update_at(&guard, i);
        }
        let inner = t.lock();
        assert_eq!(inner.ticks.len(), RING);
        assert_eq!(inner.ticks.front().map(|t| t.seq), Some(6));
    }

    #[test]
    fn a_report_that_is_not_json_is_refused_quietly() {
        assert!(serde_json::from_str::<PageReport>("not json").is_err());
        // Missing optional parts are fine.
        let r = report(r#"{"page":"x"}"#);
        assert!(r.ticks.is_empty() && !r.start);
    }
}
