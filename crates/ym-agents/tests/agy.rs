//! Antigravity's advisor against its test double (`docs/22` sections 8 and 15.2).

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Rig, assert_tree_dead, dark_flat, halves, wait_for_slow};
use ym_agents::schema::schema_text;
use ym_core::apple_auto::{
    AdvisorPicture, AutoAdvisor, Availability, FailureKind, FakeAdvisor, UnavailableReason,
};
use ym_core::auto_provider::AutoProvider;

#[test]
fn a_probe_makes_it_available_and_it_takes_one_request_at_a_time() {
    let rig = Rig::agy("signed_in");
    rig.probe();
    let Availability::Available(info) = rig.advisor.availability() else {
        panic!("not available: {:?}", rig.advisor.availability());
    };
    assert_eq!(info.provider, AutoProvider::Agy);
    assert_eq!(info.model, "default");
    assert_eq!(info.tool.unwrap().version, "1.4.7");
    assert_eq!(rig.advisor.concurrency(), 1);
}

#[test]
fn the_request_the_double_saw_is_the_specs_exactly() {
    let rig = Rig::agy("signed_in");
    rig.probe();
    let picture = dark_flat();
    let answer = rig.advisor.ask(&picture, "Moody grade, please.").unwrap();

    let expected = FakeAdvisor.ask(&picture, "Moody grade, please.").unwrap();
    assert_eq!(answer.steps, expected.steps);
    assert_eq!(answer.plan, "Test plan: Moody");
    assert_eq!(answer.served_model, None, "the envelope names no model");

    let requests = rig.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let a: Vec<&str> = request.argv.iter().map(String::as_str).collect();
    assert_eq!(a[0], "-p");
    let prompt = a[1];
    assert!(prompt.starts_with(
        "You will grade one photograph. Follow these instructions.\n\nMoody grade, please.\n\n"
    ));
    assert!(prompt.contains("@photo.jpg"));
    assert!(prompt.contains("Look only at that file."));
    assert!(prompt.ends_with(schema_text()));
    assert_eq!(&a[2..5], ["--output-format", "json", "--json-schema"]);
    assert!(std::path::Path::new(a[5]).ends_with(std::path::Path::new("in").join("schema.json")));
    assert_eq!(
        &a[6..],
        [
            "--effort",
            "low",
            "--print-timeout",
            "110s",
            "--disable-slash-commands",
            "--sandbox"
        ]
    );
    assert!(
        !a.contains(&"--model"),
        "no model unless a developer names one"
    );
    assert!(
        !a.iter()
            .any(|x| x.contains("dangerously") || *x == "--continue")
    );

    // The working folder held exactly the photograph; the temporary folder is gone.
    assert_eq!(request.files, ["photo.jpg"]);
    assert!(std::path::Path::new(&request.cwd).ends_with("cwd"));
    assert_eq!(rig.leftovers(), 0);

    // Only NO_COLOR is changed for agy.
    assert_eq!(request.env["NO_COLOR"], "1");
    assert_eq!(
        request.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"],
        "unset"
    );
}

#[test]
fn a_named_model_is_passed_and_another_picture_gets_another_answer() {
    let rig = Rig::agy_with("signed_in", |t| t.model = Some("gemini-x".to_owned()));
    rig.probe();
    let picture = halves();
    let answer = rig.advisor.ask(&picture, "Plain.").unwrap();
    assert_eq!(
        answer.steps,
        FakeAdvisor.ask(&picture, "Plain.").unwrap().steps
    );
    assert!(
        rig.requests()[0]
            .argv
            .windows(2)
            .any(|w| w == ["--model", "gemini-x"])
    );
    let Availability::Available(info) = rig.advisor.availability() else {
        panic!()
    };
    assert_eq!(info.model, "gemini-x");
}

#[test]
fn a_denied_action_and_extra_keys_do_not_fail_a_photo() {
    for scenario in ["denied", "extra_keys"] {
        let rig = Rig::agy(scenario);
        rig.probe();
        assert!(
            rig.advisor.ask(&dark_flat(), "Plain.").is_ok(),
            "{scenario}"
        );
    }
}

fn failure_of(scenario: &str) -> (FailureKind, String, usize) {
    let rig = Rig::agy_with(scenario, |t| t.request_timeout = Duration::from_secs(30));
    rig.probe();
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    (error.kind, error.message, rig.leftovers())
}

#[test]
fn every_failure_maps_to_its_kind_and_leaves_no_folder() {
    for (scenario, kind) in [
        ("malformed", FailureKind::BadAnswer),
        ("no_structured", FailureKind::BadAnswer),
        ("rate_limit", FailureKind::RateLimited),
        ("unauthenticated", FailureKind::NotSignedIn),
        ("signed_out", FailureKind::NotSignedIn),
        ("exit1", FailureKind::CliFailed),
        ("exit2", FailureKind::CliFailed),
        ("status_error", FailureKind::CliFailed),
        ("print_timeout", FailureKind::CliFailed),
        ("crash", FailureKind::CliFailed),
        ("huge", FailureKind::CliFailed),
    ] {
        let (got, message, leftovers) = failure_of(scenario);
        assert_eq!(got, kind, "{scenario}: {message}");
        assert_eq!(leftovers, 0, "{scenario} left a folder");
    }
}

#[test]
fn the_messages_say_what_went_wrong() {
    assert!(
        failure_of("exit2")
            .1
            .contains("didn't accept Yellowmoon's options")
    );
    assert!(
        failure_of("status_error")
            .1
            .contains("the model is unavailable")
    );
    let crash = failure_of("crash").1;
    assert!(
        crash.contains("exit code 3") && !crash.contains("FAKE0123456789"),
        "{crash}"
    );
}

#[test]
fn the_programs_own_time_limit_is_a_timeout() {
    // INTERRUPTED after our own clock passed its limit means the program's `--print-timeout` fired.
    let rig = Rig::agy_with("print_timeout", |t| {
        t.agy_own_limit = Duration::ZERO;
    });
    rig.probe();
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::Timeout
    );
}

#[test]
fn a_late_sign_out_ends_as_not_signed_in_and_flips_the_state() {
    let rig = Rig::agy("late_signout");
    rig.probe();
    assert!(rig.advisor.availability().is_available());
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok());
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::NotSignedIn);
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::SignedOut)
    );
}

#[test]
fn a_time_out_kills_the_tree_and_cancel_returns_within_three_seconds() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");

    // Time out.
    let p = pidfile.clone();
    let rig = Rig::agy_with("slow", move |t| {
        t.request_timeout = Duration::from_millis(1500);
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::Timeout);
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);

    // Cancel.
    let pidfile2 = pids.path().join("pids2");
    let p = pidfile2.clone();
    let rig = Rig::agy_with("slow", move |t| {
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let advisor = Arc::clone(&rig.advisor);
    let worker = std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."));
    wait_for_slow(&pidfile2);
    let cancelled_at = Instant::now();
    rig.advisor.cancel();
    let error = worker.join().unwrap().unwrap_err();
    assert!(cancelled_at.elapsed() < Duration::from_secs(3));
    assert!(error.message.contains("cancelled"));
    assert_tree_dead(&pidfile2);
    assert_eq!(rig.leftovers(), 0);
}

#[test]
fn a_picture_that_is_not_width_by_height_by_three_is_refused() {
    let rig = Rig::agy("signed_in");
    rig.probe();
    let bad = AdvisorPicture {
        width: 4,
        height: 4,
        rgb: vec![0; 5],
    };
    assert_eq!(
        rig.advisor.ask(&bad, "Plain.").unwrap_err().kind,
        FailureKind::Other
    );
    assert!(rig.requests().is_empty());
}

#[test]
fn a_program_that_writes_two_megabytes_before_it_answers_still_answers() {
    let rig = Rig::agy("backpressure");
    rig.probe();
    assert!(rig.advisor.ask(&halves(), "Plain.").is_ok());
}

#[test]
fn after_cancel_wait_idle_sees_every_child_reaped_and_every_folder_gone() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::agy_with("slow", move |t| {
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let advisor = Arc::clone(&rig.advisor);
    let worker = std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."));
    wait_for_slow(&pidfile);
    assert_eq!(rig.leftovers(), 1, "the folder with the copy of the photo");
    assert!(
        !rig.advisor.wait_idle(Duration::from_millis(300)),
        "a request runs"
    );
    rig.advisor.cancel();
    assert!(rig.advisor.wait_idle(Duration::from_secs(3)));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0, "the photo's copy is gone");
    assert!(
        worker
            .join()
            .unwrap()
            .unwrap_err()
            .message
            .contains("cancelled")
    );
}

#[cfg(unix)]
#[test]
fn a_program_that_ignores_sigterm_is_gone_after_wait_idle() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::agy_with("slow_deaf", move |t| {
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.probe();
    let advisor = Arc::clone(&rig.advisor);
    let worker = std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain."));
    wait_for_slow(&pidfile);
    let leader = common::pids_in(&pidfile)[0].clone();
    rig.advisor.cancel();
    std::thread::sleep(Duration::from_millis(600));
    assert!(common::pid_running(&leader), "it ignores SIGTERM");
    assert!(rig.advisor.wait_idle(Duration::from_secs(6)));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn cancel_stops_a_hanging_sign_in_check_too() {
    let pids = tempfile::TempDir::new().unwrap();
    let pidfile = pids.path().join("pids");
    let p = pidfile.clone();
    let rig = Rig::agy_with("signin_hang", move |t| {
        t.probe.agy_usage = Duration::from_secs(60);
        t.extra_env
            .push(("YM_FAKE_PIDFILE".into(), p.into_os_string()));
    });
    rig.advisor.refresh();
    let until = Instant::now() + Duration::from_secs(20);
    while !pidfile.is_file() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(pidfile.is_file(), "the sign-in check did not start");
    let before = rig.advisor.availability();
    let cancelled_at = Instant::now();
    rig.advisor.cancel();
    assert!(rig.advisor.wait_idle(Duration::from_secs(3)));
    assert!(cancelled_at.elapsed() < Duration::from_secs(3));
    assert_tree_dead(&pidfile);
    assert_eq!(rig.leftovers(), 0);
    assert_eq!(rig.advisor.availability(), before);
}

#[test]
fn an_automatic_probe_does_not_undo_a_sign_out_that_a_request_found() {
    // `agy -p /usage` exits 0 while the requests answer 401.
    let rig = Rig::agy("unauthenticated");
    rig.probe();
    assert!(rig.advisor.availability().is_available());
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::NotSignedIn
    );
    let signed_out = Availability::Unavailable(UnavailableReason::SignedOut);
    assert_eq!(rig.advisor.availability(), signed_out);
    rig.probe();
    rig.probe();
    assert_eq!(rig.advisor.availability(), signed_out);
    rig.probe_asked();
    assert!(rig.advisor.availability().is_available());
}
