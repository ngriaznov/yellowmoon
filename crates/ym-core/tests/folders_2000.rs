//! A folder of 2000 files (`docs/26-hardening.md` P3-3, task H-10). Ignored: it writes 4000 small
//! synthetic files and develops them twice (about a minute). Run it with
//!
//! ```text
//! cargo test -p ym-core --release --test folders_2000 -- --ignored --nocapture
//! ```
//!
//! and it prints, for a folder of 2000 files with 2000 different capture times and for one of 2000
//! files with the same capture time (one burst): the open time, the time until every photo is
//! Ready, the slowest command a person's clicks would have meant while the photos developed, and
//! the peak resident memory of the process so far.
//!
//! Asserts: the slowest command is under the bound of the 500-file test of the shell (100 ms;
//! `YM_PERF_SHELL_MAX_MS` moves it, as the weekly job does for a shared runner) and the peak memory
//! is under [`RSS_BOUND_MIB`].
mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use common::*;
use ym_core::session::{OpenOptions, PhotoStatus, Session};
use ym_raw::testgen::ArwBuilder;

const COUNT: usize = 2000;

/// The bound on peak memory of the process after both runs, in MiB. First run on the sandbox (Linux
/// x86_64, 4 vCPU, 2000 files of 192 x 128 pixels, twice): 116 MiB. The bound is three times that,
/// so only a growth per photo (a leak, a cache without a limit) fails it. A Mac with real files
/// would use far more for the pixels; this bounds the session's bookkeeping.
const RSS_BOUND_MIB: u64 = 350;

fn bound() -> Duration {
    std::env::var("YM_PERF_SHELL_MAX_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .map_or(Duration::from_millis(100), Duration::from_millis)
}

/// Peak resident memory of this process so far, in MiB.
#[cfg(unix)]
fn peak_rss_mib() -> u64 {
    // SAFETY: `getrusage` fills the struct it is given; a zeroed one is a valid value to pass.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: the pointer is to a live `rusage`.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(status, 0, "getrusage failed");
    let max = u64::try_from(usage.ru_maxrss).unwrap_or(0);
    // Bytes on macOS, kilobytes on Linux.
    if cfg!(target_os = "macos") {
        max / (1024 * 1024)
    } else {
        max / 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_mib() -> u64 {
    0
}

/// Writes `COUNT` files of 192 x 128 pixels. `distinct_times`: every file has its own capture
/// time, else all share one.
fn write_folder(dir: &Path, distinct_times: bool) {
    let (w, h) = (192_u32, 128_u32);
    let scene = Scene::new(w, h, light("sony-ilce-7rm4", 5200.0), Layout::Ramp, 0.4)
        .noise(2e-4, 1e-5)
        .seed(7);
    let mosaic = scene.mosaic();
    for k in 0..COUNT {
        let time = if distinct_times {
            let s = k % 60;
            let m = (k / 60) % 60;
            let hr = (k / 3600) % 24;
            format!("2026:10:04 {hr:02}:{m:02}:{s:02}")
        } else {
            "2026:10:04 17:42:10".to_owned()
        };
        ArwBuilder::new("ILCE-7RM4", w, h)
            .black([512; 4])
            .white([16_300; 3])
            .wb(Some(scene.as_shot_wb()))
            .iso(400)
            .exposure_time(1, 125)
            .capture_time(&time, "00", None)
            .write_to(&dir.join(format!("DSC{:05}.ARW", k + 1)), &mosaic)
            .expect("write the ARW");
    }
}

fn run(label: &str, distinct_times: bool) {
    let dir = tempfile::tempdir().expect("dir");
    write_folder(dir.path(), distinct_times);
    let session = Session::new(base_env());

    let started = Instant::now();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    let open_time = started.elapsed();
    assert_eq!(outcome.added.len(), COUNT);

    // What a person does while the photos develop: look around, move, pick.
    let ids = outcome.added;
    let mut slowest = Duration::ZERO;
    let mut calls = 0_u32;
    let mut i = 0_usize;
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        for call in 0..3 {
            let t = Instant::now();
            match call {
                0 => {
                    let _ = session.state();
                }
                1 => {
                    let _ = session.set_current(ids[i % COUNT]);
                }
                _ => {
                    let _ = session.set_picked(ids[(i * 3) % COUNT], i.is_multiple_of(2));
                }
            }
            slowest = slowest.max(t.elapsed());
            calls += 1;
        }
        i += 1;
        let ready = session
            .state()
            .photos
            .iter()
            .all(|p| p.status == PhotoStatus::Ready);
        if ready {
            break;
        }
        assert!(Instant::now() < deadline, "the photos did not all develop");
        std::thread::sleep(Duration::from_millis(5));
    }
    let all_ready = started.elapsed();
    let rss = peak_rss_mib();
    eprintln!(
        "{label}: open {open_time:?}; all Ready after {all_ready:?}; {calls} commands, the slowest \
         took {slowest:?}; peak RSS so far {rss} MiB"
    );
    let limit = bound();
    assert!(
        slowest < limit,
        "{label}: a command took {slowest:?} while photos were developing (the bound is {limit:?})"
    );
    assert!(
        rss <= RSS_BOUND_MIB,
        "{label}: peak memory {rss} MiB is over {RSS_BOUND_MIB} MiB"
    );
}

/// One test for both folders, one after the other: they would share the process (and its memory
/// figure) if they ran side by side.
#[test]
#[ignore = "writes 4000 synthetic photos and develops them (about a minute)"]
fn two_thousand_files_leave_the_session_responsive() {
    run("2000 files, 2000 capture times", true);
    run("2000 files, one capture time (one burst)", false);
}
