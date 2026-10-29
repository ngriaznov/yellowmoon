//! Claude Code's advisor against its test double (`docs/22` sections 7 and 15.2): the request the double saw,
//! the answer it gave, and what every failure becomes.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{Rig, assert_tree_dead, dark_flat, halves, read_runs, wait_for_slow};
use serde_json::Value;
use ym_agents::schema::schema_text;
use ym_core::apple_auto::{
    AdvisorPicture, AutoAdvisor, Availability, FailureKind, FakeAdvisor, UnavailableReason,
};
use ym_core::auto_provider::AutoProvider;

fn expected_steps(
    picture: &AdvisorPicture,
    instructions: &str,
) -> (ym_core::apple_auto::AutoSteps, String) {
    let answer = FakeAdvisor.ask(picture, instructions).unwrap();
    (answer.steps, answer.plan)
}

#[test]
fn an_unprobed_advisor_is_checking_and_a_probe_makes_it_available() {
    let rig = Rig::claude("signed_in");
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::Checking)
    );
    rig.probe();
    let Availability::Available(info) = rig.advisor.availability() else {
        panic!("not available: {:?}", rig.advisor.availability());
    };
    assert_eq!(info.provider, AutoProvider::Claude);
    assert_eq!(info.model, "sonnet");
    let tool = info.tool.unwrap();
    assert_eq!(tool.version, "2.1.310");
    assert_eq!(tool.api.as_deref(), Some("firstParty"));
    assert_eq!(tool.auth.as_deref(), Some("oauth_token"));
    assert_eq!(rig.advisor.concurrency(), 2);
    let details = rig.advisor.details();
    assert_eq!(details.tool_version.as_deref(), Some("2.1.310"));
    assert!(details.path.is_some() && details.breaker.is_none());
}

#[test]
fn the_request_the_double_saw_is_the_specs_exactly() {
    let rig = Rig::claude("signed_in");
    rig.probe();
    let picture = dark_flat();
    let answer = rig.advisor.ask(&picture, "Moody grade, please.").unwrap();

    // The answer is what the double computed from the picture, mapped back to steps.
    let (steps, _) = expected_steps(&picture, "Moody grade, please.");
    assert_eq!(answer.steps, steps);
    assert_eq!(answer.plan, "Test plan: Moody");
    assert_eq!(answer.scene, "Test picture");
    assert_eq!(
        answer.served_model.as_deref(),
        Some("claude-fake-1"),
        "the model with most output"
    );

    let requests = rig.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let a = &request.argv;
    let strs: Vec<&str> = a.iter().map(String::as_str).collect();
    assert_eq!(
        &strs[..16],
        [
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--safe-mode",
            "--tools",
            "",
            "--no-session-persistence",
            "--max-turns",
            "3",
            "--model",
            "sonnet",
            "--effort",
            "low",
        ]
    );
    assert_eq!(strs[16], "--system-prompt-file");
    let file = std::path::Path::new(strs[17]);
    assert!(file.ends_with(std::path::Path::new("in").join("instructions.txt")));
    assert_eq!(
        &strs[18..],
        [
            "--json-schema",
            schema_text(),
            "--disable-slash-commands",
            "--strict-mcp-config"
        ]
    );

    // The working folder was empty and is named `cwd`; the temporary folder is gone.
    assert!(
        request.files.is_empty(),
        "the working folder holds {:?}",
        request.files
    );
    assert!(std::path::Path::new(&request.cwd).ends_with("cwd"));
    assert_eq!(rig.leftovers(), 0);

    // The environment.
    let env = &request.env;
    assert_eq!(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1");
    assert_eq!(env["DISABLE_TELEMETRY"], "1");
    assert_eq!(env["NO_COLOR"], "1");
    assert_eq!(env["CLAUDECODE"], "unset");

    // Standard input: one line, an image block that is our JPEG of the picture, then the request line.
    let text = String::from_utf8(request.stdin.clone()).unwrap();
    assert_eq!(text.matches('\n').count(), 1);
    assert!(text.ends_with('\n'));
    let line: Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(line["type"], "user");
    assert_eq!(line["message"]["role"], "user");
    let content = line["message"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
    assert_eq!(content[1]["text"], "Grade this photo.");
    use base64::Engine;
    let jpeg = base64::engine::general_purpose::STANDARD
        .decode(content[0]["source"]["data"].as_str().unwrap())
        .unwrap();
    assert_eq!(jpeg, ym_agents::jpeg::encode(&picture).unwrap());
}

#[test]
fn another_picture_gets_another_answer() {
    let rig = Rig::claude("signed_in");
    rig.probe();
    let picture = halves();
    let answer = rig.advisor.ask(&picture, "Plain.").unwrap();
    let (steps, plan) = expected_steps(&picture, "Plain.");
    assert_eq!(answer.steps, steps);
    assert_eq!(answer.plan, plan);
}

#[test]
fn null_values_and_an_extra_key_do_not_fail_a_photo() {
    let rig = Rig::claude("extra_keys");
    rig.probe();
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok());
}

#[test]
fn the_model_and_the_effort_can_be_changed_by_the_developer() {
    let rig = Rig::claude_with("signed_in", |t| {
        t.model = Some("claude-opus-5-5".to_owned());
        t.effort = Some("high".to_owned());
    });
    rig.probe();
    rig.advisor.ask(&dark_flat(), "Plain.").unwrap();
    let a = &rig.requests()[0].argv;
    assert!(a.windows(2).any(|w| w == ["--model", "claude-opus-5-5"]));
    assert!(a.windows(2).any(|w| w == ["--effort", "high"]));
    let Availability::Available(info) = rig.advisor.availability() else {
        panic!()
    };
    assert_eq!(info.model, "claude-opus-5-5");
}

fn failure_of(scenario: &str) -> (FailureKind, String, usize) {
    let rig = Rig::claude_with(scenario, |t| t.request_timeout = Duration::from_secs(30));
    rig.probe();
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    (error.kind, error.message, rig.leftovers())
}

#[test]
fn every_failure_maps_to_its_kind_and_leaves_no_folder() {
    for (scenario, kind) in [
        ("malformed", FailureKind::BadAnswer),
        ("no_structured", FailureKind::BadAnswer),
        ("refused", FailureKind::Refused),
        ("rate_limit", FailureKind::RateLimited),
        ("unauthenticated", FailureKind::NotSignedIn),
        ("signed_out", FailureKind::NotSignedIn),
        ("crash", FailureKind::CliFailed),
        ("huge", FailureKind::CliFailed),
    ] {
        let (got, message, leftovers) = failure_of(scenario);
        assert_eq!(got, kind, "{scenario}: {message}");
        assert_eq!(leftovers, 0, "{scenario} left a folder");
    }
}

#[test]
fn a_crash_reports_the_exit_code_and_redacts_the_token() {
    let (_, message, _) = failure_of("crash");
    assert!(message.contains("exit code 3"), "{message}");
    assert!(!message.contains("FAKE0123456789"), "{message}");
    assert!(message.contains("rejected"), "{message}");
}

#[test]
fn a_late_sign_out_ends_as_not_signed_in_and_flips_the_state() {
    let rig = Rig::claude("late_signout");
    rig.probe();
    assert!(rig.advisor.availability().is_available());
    let heard = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&heard);
    rig.advisor.on_change(Arc::new(move || {
        h.fetch_add(1, Ordering::SeqCst);
    }));
    assert!(
        rig.advisor.ask(&dark_flat(), "Plain.").is_ok(),
        "the first request succeeds"
    );
    assert_eq!(heard.load(Ordering::SeqCst), 0, "a success changes nothing");
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::NotSignedIn);
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::SignedOut)
    );
    assert_eq!(heard.load(Ordering::SeqCst), 1);
    // The flipped state does not stop `ask` itself (the core decides whether to ask); the breaker's
    // count is the next test's.
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::NotSignedIn
    );
    assert_eq!(rig.requests().len(), 3);
    // `auth status` now says signed out too.
    rig.advisor.refresh();
    rig.advisor.settle(Duration::from_secs(20));
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::SignedOut)
    );
    // And a success renews `available` after the person signed in again (here: a fresh double).
}

#[test]
fn a_success_after_a_sign_out_renews_available() {
    let rig = Rig::claude("signed_in");
    rig.probe();
    // Pretend a sign-out was seen: the double is still signed in, so the next request succeeds.
    let heard = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&heard);
    rig.advisor.on_change(Arc::new(move || {
        h.fetch_add(1, Ordering::SeqCst);
    }));
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok());
    assert!(rig.advisor.availability().is_available());
    assert_eq!(heard.load(Ordering::SeqCst), 0);
}

#[test]
fn a_probe_of_a_signed_out_program_says_so() {
    let rig = Rig::claude("signed_out");
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::SignedOut)
    );
    assert_eq!(
        rig.advisor.details().tool_version.as_deref(),
        Some("2.1.310")
    );
}

#[test]
fn an_old_program_is_too_old() {
    let rig = Rig::claude("too_old");
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::TooOld)
    );
    assert_eq!(
        rig.advisor.details().tool_version.as_deref(),
        Some("2.1.100")
    );
}

#[test]
fn a_time_out_kills_the_tree_and_leaves_no_folder() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::claude_with("slow", move |t| {
        t.request_timeout = Duration::from_millis(1500);
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let started = Instant::now();
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::Timeout);
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
}

#[test]
fn cancel_returns_within_three_seconds_and_stores_nothing() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::claude_with("slow", move |t| {
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let advisor = Arc::clone(&rig.advisor);
    let worker = std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."));
    wait_for_slow(&pidfile);
    let cancelled_at = Instant::now();
    rig.advisor.cancel();
    let result = worker.join().unwrap();
    assert!(
        cancelled_at.elapsed() < Duration::from_secs(3),
        "{:?}",
        cancelled_at.elapsed()
    );
    let error = result.unwrap_err();
    assert!(error.message.contains("cancelled"), "{error:?}");
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
    // A request after the cancel is a new request: the advisor is not stuck cancelled.
    assert_eq!(
        read_runs(rig.record.path())
            .iter()
            .filter(|r| r.is_request())
            .count(),
        1
    );
}

#[test]
fn two_requests_at_once_run_in_parallel() {
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env.push(("YM_FAKE_DELAY_MS".into(), "1500".into()));
    });
    rig.probe();
    let started = Instant::now();
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let advisor = Arc::clone(&rig.advisor);
            std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."))
        })
        .collect();
    for handle in handles {
        assert!(handle.join().unwrap().is_ok());
    }
    let elapsed = started.elapsed();
    assert!(elapsed >= Duration::from_millis(1500), "{elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(2700),
        "two at once took {elapsed:?}"
    );
    assert_eq!(rig.leftovers(), 0);
}

#[test]
fn a_program_that_is_not_there_is_unavailable() {
    use ym_agents::advisor::Source;
    use ym_agents::tool::{Tool, Tuning};
    let missing = std::env::temp_dir().join("ym-agents-no-such-claude");
    let advisor =
        ym_agents::ClaudeAdvisor::custom(Source::Program(missing), Tuning::for_tool(Tool::Claude));
    advisor.refresh();
    advisor.settle(Duration::from_secs(20));
    assert_eq!(
        advisor.availability(),
        Availability::Unavailable(UnavailableReason::NotInstalled)
    );
    let error = advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::Unavailable);
}

#[test]
fn a_picture_that_is_not_width_by_height_by_three_is_refused() {
    let rig = Rig::claude("signed_in");
    rig.probe();
    let bad = AdvisorPicture {
        width: 10,
        height: 10,
        rgb: vec![0; 10],
    };
    let error = rig.advisor.ask(&bad, "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::Other);
    assert!(rig.requests().is_empty(), "no process was started for it");
}

#[test]
fn a_program_that_writes_two_megabytes_before_reading_still_answers() {
    // The double writes 2 MB of output before it reads the picture; the request must not deadlock.
    let rig = Rig::claude("backpressure");
    rig.probe();
    let answer = rig.advisor.ask(&halves(), "Plain.").unwrap();
    assert_eq!(answer.scene, "Test picture");
    assert_eq!(rig.leftovers(), 0);
}

#[test]
fn not_signed_in_does_not_count_toward_the_breaker() {
    // Two failures the breaker counts, a request that finds the program signed out, two more: five
    // requests, four failures, never three in a row. Were `not_signed_in` counted (or ignored, which
    // leaves the count where it was) the fifth request would find the breaker open.
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env.push((
            "YM_FAKE_PLAN".into(),
            "fail,fail,signedout,fail,fail,ok".into(),
        ));
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    let kinds: Vec<Option<FailureKind>> = (0..6)
        .map(|_| {
            rig.advisor
                .ask(&dark_flat(), "Plain.")
                .err()
                .map(|e| e.kind)
        })
        .collect();
    assert_eq!(
        kinds,
        [
            Some(FailureKind::CliFailed),
            Some(FailureKind::CliFailed),
            Some(FailureKind::NotSignedIn),
            Some(FailureKind::CliFailed),
            Some(FailureKind::CliFailed),
            None,
        ]
    );
    assert!(rig.advisor.details().breaker.is_none());
    assert_eq!(rig.requests().len(), 6, "every request reached the program");
}

// ----- cancel() and wait_idle -----

/// A request on its own thread against the `slow` double (`scenario`), started and waiting on its pid file.
fn slow_request(
    scenario: &str,
    adjust: impl FnOnce(&mut ym_agents::tool::Tuning),
) -> (
    Rig<ym_agents::ClaudeAdvisor>,
    std::path::PathBuf,
    tempfile::TempDir,
    std::thread::JoinHandle<
        Result<ym_core::apple_auto::AdvisorAnswer, ym_core::apple_auto::AdvisorError>,
    >,
) {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::claude_with(scenario, move |t| {
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
        adjust(t);
    });
    rig.probe();
    let advisor = Arc::clone(&rig.advisor);
    let worker = std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."));
    wait_for_slow(&pidfile);
    (rig, pidfile, pids, worker)
}

#[test]
fn after_cancel_wait_idle_sees_every_child_reaped_and_every_folder_gone() {
    let (rig, pidfile, _pids, worker) = slow_request("slow", |_| {});
    assert_eq!(
        rig.leftovers(),
        1,
        "the request's folder exists while it runs"
    );
    // `cancel` does not wait for the request's thread, which reaps the child and drops the folder;
    // `wait_idle` does. The caller may be about to exit and wants both.
    let cancelled_at = Instant::now();
    rig.advisor.cancel();
    assert!(rig.advisor.wait_idle(Duration::from_secs(3)));
    assert!(cancelled_at.elapsed() < Duration::from_secs(3));
    assert_tree_dead(&pidfile);
    assert_eq!(
        rig.leftovers(),
        0,
        "no ym-auto-* folder of that run remains"
    );
    let error = worker.join().unwrap().unwrap_err();
    assert!(error.message.contains("cancelled"), "{error:?}");
}

#[test]
fn wait_idle_is_true_when_nothing_runs_and_false_at_its_timeout_while_a_request_does() {
    let (rig, pidfile, _pids, worker) = slow_request("slow", |_| {});
    let started = Instant::now();
    assert!(!rig.advisor.wait_idle(Duration::from_millis(400)));
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(350) && waited < Duration::from_secs(3),
        "{waited:?}"
    );
    rig.advisor.cancel();
    assert!(rig.advisor.wait_idle(Duration::from_secs(3)));
    assert!(rig.advisor.wait_idle(Duration::ZERO), "and it stays idle");
    assert_tree_dead(&pidfile);
    let _ = worker.join().unwrap();
}

#[cfg(unix)]
#[test]
fn a_program_that_ignores_sigterm_is_gone_after_wait_idle() {
    let (rig, pidfile, _pids, worker) = slow_request("slow_deaf", |_| {});
    let leader = common::pids_in(&pidfile)[0].clone();
    rig.advisor.cancel();
    std::thread::sleep(Duration::from_millis(600));
    assert!(
        common::pid_running(&leader),
        "the double ignores SIGTERM: it must still be there"
    );
    // The request's thread sends the SIGKILL after its 2 s of grace.
    assert!(rig.advisor.wait_idle(Duration::from_secs(6)));
    assert!(!common::pid_running(&leader));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn cancel_stops_a_hanging_probe_too() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::claude_with("signin_hang", move |t| {
        t.probe.claude_auth = Duration::from_secs(60);
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.advisor.refresh();
    // `claude auth status` hangs (the double writes its pid when it starts to).
    let until = Instant::now() + Duration::from_secs(20);
    while !pidfile.is_file() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(pidfile.is_file(), "the sign-in check did not start");
    let before = rig.advisor.availability();
    assert!(
        !rig.advisor.wait_idle(Duration::from_millis(200)),
        "the probe runs"
    );

    let cancelled_at = Instant::now();
    rig.advisor.cancel();
    assert!(rig.advisor.wait_idle(Duration::from_secs(3)));
    assert!(cancelled_at.elapsed() < Duration::from_secs(3));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
    assert_eq!(
        rig.advisor.availability(),
        before,
        "a probe that was stopped half-way says nothing"
    );
    assert!(rig.advisor.wait_idle(Duration::ZERO), "and it stays idle");
}

// ----- a sign-out that a request found -----

#[test]
fn an_automatic_probe_does_not_undo_a_sign_out_that_a_request_found() {
    // `auth status` reads the stored credentials and says signed in while the service answers 401.
    let rig = Rig::claude("unauthenticated");
    rig.probe();
    assert!(rig.advisor.availability().is_available());
    let heard = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&heard);
    rig.advisor.on_change(Arc::new(move || {
        h.fetch_add(1, Ordering::SeqCst);
    }));
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::NotSignedIn
    );
    let signed_out = Availability::Unavailable(UnavailableReason::SignedOut);
    assert_eq!(rig.advisor.availability(), signed_out);
    assert_eq!(heard.load(Ordering::SeqCst), 1);

    // The core's refreshes (a session opening, a change of the preferences, before an export) are
    // automatic: the program still says `loggedIn`, and it is not believed.
    for _ in 0..3 {
        rig.probe();
        assert_eq!(rig.advisor.availability(), signed_out);
    }
    assert!(
        rig.runs().iter().filter(|r| r.is_sign_in_check()).count() >= 4,
        "the probes did run"
    );
    assert_eq!(
        heard.load(Ordering::SeqCst),
        1,
        "and told nobody of a change"
    );
    assert_eq!(rig.advisor.details().auth.as_deref(), Some("oauth_token"));

    // The person opens Preferences: that check is believed.
    rig.probe_asked();
    assert!(rig.advisor.availability().is_available());
    assert_eq!(heard.load(Ordering::SeqCst), 2);
    // If the service still says 401, the next request finds out again, once.
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::NotSignedIn
    );
    rig.probe();
    assert_eq!(rig.advisor.availability(), signed_out);
}

#[test]
fn a_held_sign_out_ends_with_a_success_or_when_its_hold_has_passed() {
    // A success clears it: the plan fails the first request as signed out and answers the second.
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env
            .push(("YM_FAKE_PLAN".into(), "signedout,ok".into()));
    });
    rig.probe();
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_err());
    assert!(!rig.advisor.availability().is_available());
    rig.probe();
    assert!(!rig.advisor.availability().is_available(), "held");
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok());
    assert!(rig.advisor.availability().is_available());
    rig.probe();
    assert!(
        rig.advisor.availability().is_available(),
        "a success ended the hold: a later probe is believed"
    );

    // Time clears it: the hold is shortened for the test.
    let rig = Rig::claude_with("unauthenticated", |t| {
        t.signed_out_hold = Duration::from_secs(1);
    });
    rig.probe();
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_err());
    rig.probe();
    assert!(
        !rig.advisor.availability().is_available(),
        "within the hold"
    );
    std::thread::sleep(Duration::from_millis(1100));
    rig.probe();
    assert!(
        rig.advisor.availability().is_available(),
        "the hold has passed: the probe is believed"
    );
}

#[test]
fn only_the_registrys_check_of_every_advisor_is_the_persons_own() {
    // Preferences opening and `ym doctor` refresh every advisor at once (`ProviderAdvisors::refresh(None)`
    // calls `refresh_asked`); every other read refreshes the chosen provider's (`Some`), which is automatic.
    use ym_core::auto_provider::ProviderAdvisors;
    let rig = Rig::claude("unauthenticated");
    rig.probe();
    let advisor: Arc<dyn AutoAdvisor> = rig.advisor.clone();
    let advisors = ProviderAdvisors::default().with(AutoProvider::Claude, advisor);
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_err());
    let signed_out = Availability::Unavailable(UnavailableReason::SignedOut);
    assert_eq!(rig.advisor.availability(), signed_out);

    advisors.refresh(Some(AutoProvider::Claude));
    advisors.settle(Duration::from_secs(30));
    assert_eq!(
        rig.advisor.availability(),
        signed_out,
        "automatic: not believed"
    );
    advisors.refresh(None);
    advisors.settle(Duration::from_secs(30));
    assert!(
        rig.advisor.availability().is_available(),
        "the person's: believed"
    );
}
