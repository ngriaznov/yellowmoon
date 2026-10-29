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

/// What each of `n` requests ends as: `None` for an answer, else the failure's kind.
fn outcomes<A: AutoAdvisor>(advisor: &A, n: usize) -> Vec<Option<FailureKind>> {
    (0..n)
        .map(|_| advisor.ask(&dark_flat(), "Plain.").err().map(|e| e.kind))
        .collect()
}

fn claude_with_plan(plan: &str) -> Rig<ym_agents::ClaudeAdvisor> {
    let plan = plan.to_owned();
    let rig = Rig::claude_with("signed_in", move |t| {
        t.extra_env.push(("YM_FAKE_PLAN".into(), plan.into()));
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    rig
}

fn agy_with_plan(plan: &str) -> Rig<ym_agents::AgyAdvisor> {
    let plan = plan.to_owned();
    let rig = Rig::agy_with("signed_in", move |t| {
        t.extra_env.push(("YM_FAKE_PLAN".into(), plan.into()));
        t.breaker_cooldown = Duration::from_secs(60);
    });
    rig.probe();
    rig
}

#[test]
fn an_answer_the_breaker_does_not_count_resets_the_count() {
    // Two failures, one answer that is not the program's fault, two failures, then an answer: five
    // requests with four failures in all and never three in a row, so the breaker stays closed and the
    // sixth request goes through. Were the middle one counted as a failure, or merely ignored, the
    // fifth request would find the breaker open.
    let cli = Some(FailureKind::CliFailed);
    for (step, kind) in [
        ("bad", FailureKind::BadAnswer),
        ("refused", FailureKind::Refused),
        ("signedout", FailureKind::NotSignedIn),
    ] {
        let rig = claude_with_plan(&format!("fail,fail,{step},fail,fail,ok"));
        assert_eq!(
            outcomes(&*rig.advisor, 6),
            [cli, cli, Some(kind), cli, cli, None],
            "claude, {step}"
        );
        assert!(rig.advisor.details().breaker.is_none(), "claude, {step}");
        assert_eq!(
            rig.requests().len(),
            6,
            "claude, {step}: all reached the program"
        );
    }
    // The agy double has no refusal.
    for (step, kind) in [
        ("bad", FailureKind::BadAnswer),
        ("signedout", FailureKind::NotSignedIn),
    ] {
        let rig = agy_with_plan(&format!("fail,fail,{step},fail,fail,ok"));
        assert_eq!(
            outcomes(&*rig.advisor, 6),
            [cli, cli, Some(kind), cli, cli, None],
            "agy, {step}"
        );
        assert!(rig.advisor.details().breaker.is_none(), "agy, {step}");
    }
}

#[test]
fn the_same_requests_with_a_counted_failure_in_the_middle_open_it() {
    // The control of the test above: the middle request is a failure the breaker counts (a failure, a
    // rate limit), so the third in a row opens it and the fourth request is refused without a process.
    let cli = Some(FailureKind::CliFailed);
    for (step, kind) in [
        ("fail", FailureKind::CliFailed),
        ("rate", FailureKind::RateLimited),
    ] {
        let rig = claude_with_plan(&format!("fail,fail,{step},fail,fail,ok"));
        assert_eq!(
            outcomes(&*rig.advisor, 6),
            [
                cli,
                cli,
                Some(kind),
                Some(FailureKind::Unavailable),
                Some(FailureKind::Unavailable),
                Some(FailureKind::Unavailable)
            ],
            "claude, {step}"
        );
        assert!(rig.advisor.details().breaker.is_some(), "claude, {step}");
        assert_eq!(
            rig.requests().len(),
            3,
            "claude, {step}: the open breaker starts nothing"
        );

        let rig = agy_with_plan(&format!("fail,fail,{step},fail,fail,ok"));
        assert_eq!(
            outcomes(&*rig.advisor, 4),
            [cli, cli, Some(kind), Some(FailureKind::Unavailable)],
            "agy, {step}"
        );
        assert!(rig.advisor.details().breaker.is_some(), "agy, {step}");
    }
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
