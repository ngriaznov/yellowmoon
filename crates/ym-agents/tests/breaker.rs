//! The breaker (`docs/22` section 9.2) through the real advisors: three failures in a row open it, an open
//! breaker starts no process, one trial goes through after the cool-down.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Rig, dark_flat};
use ym_core::apple_auto::{AutoAdvisor, FailureKind};

fn fails(rig: &Rig<ym_agents::ClaudeAdvisor>, n: usize) {
    for i in 0..n {
        let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
        assert_eq!(error.kind, FailureKind::CliFailed, "failure {i}: {error:?}");
    }
}

#[test]
fn three_failures_open_it_and_an_open_breaker_starts_no_process() {
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env.push(("YM_FAKE_FAILS".into(), "100".into()));
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    assert!(rig.advisor.details().breaker.is_none());
    fails(&rig, 3);
    assert_eq!(rig.requests().len(), 3);

    let started = Instant::now();
    let error = rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err();
    assert_eq!(error.kind, FailureKind::Unavailable);
    assert!(
        error
            .message
            .starts_with("stopped after three failures in a row: "),
        "{error:?}"
    );
    assert!(
        error.message.contains("Claude Code reported an error"),
        "{error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "refused at once"
    );
    assert_eq!(rig.requests().len(), 3, "the record shows no new run");

    let sentence = rig.advisor.details().breaker.unwrap();
    assert_eq!(
        sentence,
        "Claude Code failed three times in a row (Claude Code reported an error). \
         Photos it doesn't answer use Default Auto."
    );
    // A refresh does not clear it.
    rig.probe();
    assert!(rig.advisor.details().breaker.is_some());
}

#[test]
fn after_the_cooldown_one_trial_goes_through_and_a_success_closes_it() {
    let rig = Rig::claude_with("signed_in", |t| {
        // The first three requests fail, the fourth (the trial) succeeds.
        t.extra_env.push(("YM_FAKE_FAILS".into(), "3".into()));
        t.breaker_cooldown = Duration::from_millis(400);
    });
    rig.probe();
    fails(&rig, 3);
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::Unavailable
    );
    std::thread::sleep(Duration::from_millis(500));
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok(), "the trial");
    assert!(
        rig.advisor.details().breaker.is_none(),
        "a success closes it"
    );
    assert!(rig.advisor.ask(&dark_flat(), "Plain.").is_ok());
    assert_eq!(rig.requests().len(), 5);
}

#[test]
fn a_failed_trial_opens_it_again() {
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env.push(("YM_FAKE_FAILS".into(), "4".into()));
        t.breaker_cooldown = Duration::from_millis(400);
    });
    rig.probe();
    fails(&rig, 3);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::CliFailed,
        "the trial fails"
    );
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::Unavailable
    );
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        rig.advisor.ask(&dark_flat(), "Plain.").is_ok(),
        "the next trial succeeds"
    );
}

#[test]
fn a_bad_answer_resets_the_count() {
    // Two failures, then answers the breaker does not count against the program: never three in a row.
    let rig = Rig::claude_with("malformed", |t| {
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    for _ in 0..6 {
        assert_eq!(
            rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
            FailureKind::BadAnswer
        );
    }
    assert!(rig.advisor.details().breaker.is_none());
}

#[test]
fn requests_at_once_share_the_breaker() {
    let rig = Rig::claude_with("signed_in", |t| {
        t.extra_env.push(("YM_FAKE_FAILS".into(), "100".into()));
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    // Two at once (Claude Code's concurrency), both fail; the third failure opens it.
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let advisor = Arc::clone(&rig.advisor);
            std::thread::spawn(move || advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind)
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), FailureKind::CliFailed);
    }
    assert!(
        rig.advisor.details().breaker.is_none(),
        "two failures are not three"
    );
    fails(&rig, 1);
    assert!(rig.advisor.details().breaker.is_some());
    assert_eq!(
        rig.advisor.ask(&dark_flat(), "Plain.").unwrap_err().kind,
        FailureKind::Unavailable
    );
}
