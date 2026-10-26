//! The process runner (`docs/22` section 5), against the two test doubles in their raw mode.

mod common;

use std::ffi::OsString;
use std::time::{Duration, Instant};

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
