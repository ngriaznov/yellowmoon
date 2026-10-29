//! The process runner (`docs/22` section 5), against the two test doubles in their raw mode.

mod common;

use std::ffi::OsString;
use std::time::{Duration, Instant};

#[cfg(unix)]
use common::pids_in;
use common::{FAKE_CLAUDE, assert_tree_dead, leftovers, read_runs, wait_for_slow};
use tempfile::TempDir;
use ym_agents::process::{
    CancelSource, EnvProfile, Outcome, RunSpec, Workdir, run, run_with, sweep_stale_in,
};

fn raw(scenario: &str, extra: &[(&str, &std::path::Path)]) -> RunSpec {
    let dir = std::env::temp_dir();
    let mut vars: Vec<(OsString, OsString)> = vec![("YM_FAKE_SCENARIO".into(), scenario.into())];
    for (name, value) in extra {
        vars.push(((*name).into(), value.as_os_str().to_owned()));
    }
    RunSpec::new(FAKE_CLAUDE, dir, EnvProfile::Claude)
        .args(["--ym-fake-raw"])
        .extra_env(vars)
}

#[test]
fn a_program_that_ends_by_itself_is_exited() {
    let out = run(raw("signed_in", &[]));
    assert_eq!(out.outcome, Outcome::Exited);
    assert_eq!(out.exit_code, Some(0));
    assert!(out.success());
    assert_eq!(out.stdout_text().trim(), "raw ok");
}

#[test]
fn a_program_that_cannot_be_started_is_spawn_failed() {
    let spec = RunSpec::new(
        std::env::temp_dir().join("ym-no-such-program"),
        std::env::temp_dir(),
        EnvProfile::Inherit,
    );
    let out = run(spec);
    assert!(
        matches!(out.outcome, Outcome::SpawnFailed(ref why) if !why.is_empty()),
        "{out:?}"
    );
}

#[test]
fn a_slow_program_is_killed_together_with_its_grandchild() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let started = Instant::now();
    let out =
        run(raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)]).timeout(Duration::from_millis(1500)));
    assert_eq!(out.outcome, Outcome::TimedOut, "{out:?}");
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
    assert_tree_dead(&pidfile);
}

#[test]
fn cancelling_ends_a_running_request_within_three_seconds() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let source = CancelSource::new();
    let spec = raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)])
        .timeout(Duration::from_secs(60))
        .cancel(source.token());
    let worker = std::thread::spawn(move || run(spec));
    wait_for_slow(&pidfile);
    let cancelled_at = Instant::now();
    source.cancel();
    let out = worker.join().unwrap();
    assert_eq!(out.outcome, Outcome::Cancelled, "{out:?}");
    assert!(
        cancelled_at.elapsed() < Duration::from_secs(3),
        "{:?}",
        cancelled_at.elapsed()
    );
    assert_tree_dead(&pidfile);
}

#[test]
fn a_program_that_writes_before_it_reads_does_not_deadlock() {
    // It writes 2 MB to standard output before it reads a byte of the 300 KB we send.
    let dir = TempDir::new().unwrap();
    let record = dir.path().to_path_buf();
    let spec = raw("backpressure", &[("YM_FAKE_RECORD", &record)])
        .stdin(vec![b'a'; 300 * 1024])
        .timeout(Duration::from_secs(30));
    let out = run(spec);
    assert_eq!(out.outcome, Outcome::Exited, "{out:?}");
    assert!(out.stdout.len() > 2 * 1024 * 1024);
    assert!(out.stdout_text().trim_end().ends_with("done"));
    assert_eq!(
        std::fs::read_to_string(record.join("stdin.len")).unwrap(),
        (300 * 1024).to_string()
    );
}

#[test]
fn output_beyond_the_cap_kills_the_program() {
    let out = run(raw("huge", &[]).timeout(Duration::from_secs(60)));
    assert_eq!(out.outcome, Outcome::OutputTooLarge);
    assert!(out.stdout.len() <= 4 * 1024 * 1024);
    assert_eq!(out.exit_code, None);
}

#[test]
fn a_line_beyond_its_cap_kills_the_program() {
    // 1 KB lines are fine under a cap of 512 bytes per line? No: they are not.
    let mut spec = raw("huge", &[]).timeout(Duration::from_secs(60));
    spec.line_cap = 512;
    let out = run(spec);
    assert_eq!(out.outcome, Outcome::OutputTooLarge);
}

#[test]
fn a_program_that_exits_without_reading_its_input_is_not_an_error() {
    // 4 MB into a pipe that nobody reads: the write fails with a broken pipe, which is ignored.
    let spec = raw("early_exit", &[]).stdin(vec![0; 4 * 1024 * 1024]);
    let out = run(spec);
    assert_eq!(out.outcome, Outcome::Exited, "{out:?}");
    assert_eq!(out.exit_code, Some(0));
}

#[test]
fn the_environment_has_the_switches() {
    let dir = TempDir::new().unwrap();
    let record = dir.path().to_path_buf();
    let out = run(raw("signed_in", &[("YM_FAKE_RECORD", &record)]));
    assert!(out.success());
    let env = std::fs::read_to_string(record.join("env.txt")).unwrap();
    for line in [
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
        "DISABLE_TELEMETRY=1",
        "DISABLE_ERROR_REPORTING=1",
        "DISABLE_AUTOUPDATER=1",
        "NO_COLOR=1",
    ] {
        assert!(env.lines().any(|l| l == line), "{line} missing in\n{env}");
    }
}

#[test]
fn agy_gets_no_color_and_nothing_else() {
    let dir = TempDir::new().unwrap();
    let record = dir.path().to_path_buf();
    let mut spec = raw("signed_in", &[("YM_FAKE_RECORD", &record)]);
    spec.env = EnvProfile::Agy;
    assert!(run(spec).success());
    let env = std::fs::read_to_string(record.join("env.txt")).unwrap();
    assert!(env.lines().any(|l| l == "NO_COLOR=1"), "{env}");
    assert!(
        env.lines()
            .any(|l| l == "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=unset"),
        "{env}"
    );
    assert!(env.lines().any(|l| l == "DISABLE_TELEMETRY=unset"), "{env}");
}

#[cfg(windows)]
#[test]
fn no_console_window_is_made_on_windows() {
    // CREATE_NO_WINDOW gives the child a console of its own, or none: at most the child is attached.
    let dir = TempDir::new().unwrap();
    let record = dir.path().to_path_buf();
    assert!(run(raw("signed_in", &[("YM_FAKE_RECORD", &record)])).success());
    let env = std::fs::read_to_string(record.join("env.txt")).unwrap();
    let line = env
        .lines()
        .find(|l| l.starts_with("CONSOLE_PROCESSES="))
        .unwrap();
    let n: u32 = line
        .trim_start_matches("CONSOLE_PROCESSES=")
        .parse()
        .unwrap();
    assert!(n <= 1, "the child shares a console with {n} processes");
}

#[test]
fn a_panic_after_the_spawn_kills_the_tree() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let spec = raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)]).timeout(Duration::from_secs(60));
    let waiting = pidfile.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_with(spec, || {
            wait_for_slow(&waiting);
            panic!("the caller's code failed after the spawn");
        });
    }));
    assert!(result.is_err());
    assert_tree_dead(&pidfile);
}

#[test]
fn no_folder_is_left_after_a_panic_and_the_sweep_is_careful() {
    let parent = TempDir::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        let folder = Workdir::new_in(parent.path()).unwrap();
        assert!(folder.cwd().is_dir());
        panic!("a request failed");
    });
    assert!(result.is_err());
    assert_eq!(leftovers(parent.path()), 0);

    // The sweep removes an old `ym-auto-*` folder and leaves a newer one and any other folder alone.
    let old = parent.path().join("ym-auto-old");
    let other = parent.path().join("keep");
    std::fs::create_dir(&old).unwrap();
    std::fs::create_dir(&other).unwrap();
    assert_eq!(
        sweep_stale_in(parent.path(), Duration::from_secs(600)),
        0,
        "nothing is ten minutes old"
    );
    assert!(old.exists());
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(sweep_stale_in(parent.path(), Duration::from_millis(10)), 1);
    assert!(!old.exists() && other.exists());
    let _ = read_runs(parent.path());
}

// ----- cancel() tells the children to stop before it returns -----

/// Whether `pid` is gone within `within` (a zombie is gone).
#[cfg(unix)]
fn gone_within(pid: &str, within: Duration) -> bool {
    let until = Instant::now() + within;
    while Instant::now() < until {
        if !common::pid_running(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    !common::pid_running(pid)
}

#[cfg(unix)]
#[test]
fn cancel_tells_the_children_to_stop_before_it_returns() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let source = CancelSource::new();
    let spec = raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)])
        .timeout(Duration::from_secs(60))
        .cancel(source.token());
    // The run's own thread is held in the hook that runs after the start, so it never polls: only
    // `cancel` itself can reach the child.
    let (release, held) = std::sync::mpsc::channel::<()>();
    let waiting = pidfile.clone();
    let worker = std::thread::spawn(move || {
        run_with(spec, || {
            wait_for_slow(&waiting);
            let _ = held.recv_timeout(Duration::from_secs(30));
        })
    });
    wait_for_slow(&pidfile);
    let pids = pids_in(&pidfile);
    assert_eq!(pids.len(), 2, "{pids:?}");
    assert!(pids.iter().all(|p| common::pid_running(p)));

    source.cancel();
    // SIGTERM reached the group at once; the double does not handle it, so both are gone (they stay
    // zombies, which nobody has reaped, and a zombie is not running).
    for pid in &pids {
        assert!(
            gone_within(pid, Duration::from_secs(2)),
            "pid {pid} is still running after cancel() returned"
        );
    }
    release.send(()).unwrap();
    let out = worker.join().unwrap();
    assert_eq!(out.outcome, Outcome::Cancelled, "{out:?}");
    assert_tree_dead(&pidfile);
}

#[test]
fn a_token_cancelled_before_the_start_starts_nothing() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let source = CancelSource::new();
    let token = source.token();
    source.cancel();
    let out = run(raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)]).cancel(token));
    assert_eq!(out.outcome, Outcome::Cancelled, "{out:?}");
    std::thread::sleep(Duration::from_millis(300));
    assert!(!pidfile.exists(), "the program was started");
    // A token made after the cancel is a new run's: it starts.
    let out = run(raw("signed_in", &[]).cancel(source.token()));
    assert!(out.success(), "{out:?}");
}

#[cfg(unix)]
#[test]
fn a_cancel_during_a_run_is_a_cancel_not_an_exit() {
    // The child dies of the SIGTERM that `cancel` sent; the run must not read that as the program having
    // ended by itself (which would be a failure of the request, counted by the breaker).
    for _ in 0..5 {
        let dir = TempDir::new().unwrap();
        let pidfile = dir.path().join("pids");
        let source = CancelSource::new();
        let spec = raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)])
            .timeout(Duration::from_secs(60))
            .cancel(source.token());
        let worker = std::thread::spawn(move || run(spec));
        wait_for_slow(&pidfile);
        source.cancel();
        let out = worker.join().unwrap();
        assert_eq!(out.outcome, Outcome::Cancelled, "{out:?}");
        assert_eq!(out.exit_code, None);
    }
}

#[cfg(unix)]
#[test]
fn a_stopped_program_is_ended_at_once_when_its_time_is_up() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let started = Instant::now();
    let spec = raw("slow", &[("YM_FAKE_PIDFILE", &pidfile)]).timeout(Duration::from_millis(1500));
    let worker = std::thread::spawn(move || run(spec));
    wait_for_slow(&pidfile);
    // Stopped as a terminal's job control stops a background group. A SIGTERM to a stopped group is
    // not acted on until it is continued; without SIGCONT it would wait for the SIGKILL, 2 s later.
    let leader = pids_in(&pidfile)[0].clone();
    let stopped = std::process::Command::new("sh")
        .args(["-c", &format!("kill -s STOP -- -{leader}")])
        .status()
        .unwrap();
    assert!(stopped.success());
    let out = worker.join().unwrap();
    assert_eq!(out.outcome, Outcome::TimedOut, "{out:?}");
    assert!(
        started.elapsed() < Duration::from_millis(2600),
        "took {:?}: the SIGTERM waited for the SIGKILL",
        started.elapsed()
    );
    assert_tree_dead(&pidfile);
}

#[cfg(unix)]
#[test]
fn a_program_that_ignores_sigterm_is_ended_by_the_sigkill_after_the_grace() {
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("pids");
    let started = Instant::now();
    let out = run(
        raw("slow_deaf", &[("YM_FAKE_PIDFILE", &pidfile)]).timeout(Duration::from_millis(1000))
    );
    assert_eq!(out.outcome, Outcome::TimedOut, "{out:?}");
    // 1 s of time, then 2 s of grace in which the double does not leave.
    assert!(
        started.elapsed() >= Duration::from_millis(2900),
        "{:?}",
        started.elapsed()
    );
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
    assert_tree_dead(&pidfile);
}

// ----- a session of its own -----

#[cfg(target_os = "linux")]
#[test]
fn a_run_can_lead_a_session_of_its_own() {
    // The shell reads its own process, group and session ids from /proc.
    let script = "read pid comm state ppid pgrp sid rest < /proc/$$/stat; echo \"$$ $pgrp $sid\"";
    let ids = |new_session: bool| -> (u32, u32, u32) {
        let mut spec =
            RunSpec::new("/bin/sh", std::env::temp_dir(), EnvProfile::Inherit).args(["-c", script]);
        spec.new_session = new_session;
        let out = run(spec);
        assert!(out.success(), "{out:?}");
        let text = out.stdout_text();
        let n: Vec<u32> = text
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        (n[0], n[1], n[2])
    };
    let (pid, group, session) = ids(true);
    assert_eq!(
        (group, session),
        (pid, pid),
        "no controlling terminal: its own session"
    );
    let (pid, group, session) = ids(false);
    assert_eq!(group, pid, "its own group");
    assert_ne!(session, pid, "but the session it was started in");
}
