//! The probes (`docs/22` section 3.3): a read never waits for a process, calls are coalesced, `settle`
//! waits, a change is heard once, a hanging probe ends at its limit, and `agy`'s sign-in check runs only
//! when asked for.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{Rig, install, known_path, locations_in};
use tempfile::TempDir;
use ym_agents::ClaudeAdvisor;
use ym_agents::advisor::Source;
use ym_agents::tool::{Tool, Tuning};
use ym_core::apple_auto::{AutoAdvisor, Availability, UnavailableReason};

#[test]
fn refresh_never_blocks_and_a_hanging_probe_ends_at_its_limit() {
    let rig = Rig::claude_with("probe_hang", |t| {
        t.probe.version = Duration::from_millis(800)
    });
    let started = Instant::now();
    rig.advisor.refresh();
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "refresh returned at once"
    );
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::Checking),
        "the read does not wait"
    );
    rig.advisor.settle(Duration::from_secs(20));
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(700),
        "settle waited for the probe: {took:?}"
    );
    assert!(took < Duration::from_secs(10), "{took:?}");
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::ProbeFailed)
    );
}

#[test]
fn a_hanging_sign_in_check_is_probe_failed_at_its_limit() {
    let rig = Rig::claude_with("signin_hang", |t| {
        t.probe.claude_auth = Duration::from_millis(600)
    });
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::ProbeFailed)
    );
    // The version was found, so it is known for the hint.
    assert_eq!(
        rig.advisor.details().tool_version.as_deref(),
        Some("2.1.310")
    );
}

#[test]
fn calls_within_the_interval_are_coalesced() {
    let rig = Rig::claude_with("signed_in", |t| t.min_interval = Duration::from_secs(60));
    rig.probe();
    let probes =
        |rig: &Rig<ClaudeAdvisor>| rig.runs().iter().filter(|r| r.is_sign_in_check()).count();
    assert_eq!(probes(&rig), 1);
    rig.probe();
    rig.probe();
    assert_eq!(probes(&rig), 1, "one probe within the minimum interval");
}

#[test]
fn settle_waits_and_a_listener_hears_one_change_not_a_repeat() {
    let rig = Rig::claude("signed_in");
    let heard = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&heard);
    rig.advisor.on_change(Arc::new(move || {
        h.fetch_add(1, Ordering::SeqCst);
    }));
    rig.probe();
    assert_eq!(heard.load(Ordering::SeqCst), 1, "checking became available");
    rig.probe();
    assert_eq!(
        heard.load(Ordering::SeqCst),
        1,
        "the same reading is not a change"
    );
    assert_eq!(
        rig.runs().iter().filter(|r| r.is_sign_in_check()).count(),
        2,
        "it did probe again"
    );
}

#[test]
fn a_listener_may_read_the_availability_without_deadlock() {
    let rig = Rig::claude("signed_in");
    let advisor = Arc::clone(&rig.advisor);
    let seen = Arc::new(std::sync::Mutex::new(None));
    let s = Arc::clone(&seen);
    rig.advisor.on_change(Arc::new(move || {
        *s.lock().unwrap() = Some(advisor.availability());
    }));
    rig.probe();
    assert!(
        seen.lock()
            .unwrap()
            .as_ref()
            .is_some_and(Availability::is_available)
    );
}

#[test]
fn agys_sign_in_check_runs_only_when_asked_for() {
    let rig = Rig::agy("signed_in");
    // The start-up probe: up to the version. The state is `checking`; the version is known.
    rig.advisor.start();
    rig.advisor.settle(Duration::from_secs(20));
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::Checking)
    );
    assert_eq!(rig.advisor.details().tool_version.as_deref(), Some("1.4.7"));
    assert!(
        rig.runs().iter().all(|r| !r.is_sign_in_check()),
        "no sign-in check yet"
    );
    assert_eq!(rig.runs().len(), 1);

    // Asked for: now it runs.
    rig.probe();
    assert!(rig.advisor.availability().is_available());
    assert_eq!(
        rig.runs().iter().filter(|r| r.is_sign_in_check()).count(),
        1
    );
}

#[test]
fn a_full_probe_asked_for_during_the_light_one_follows_it() {
    let rig = Rig::agy("signed_in");
    rig.advisor.start();
    // Asked for at once, while the light probe may still be running.
    rig.advisor.refresh();
    rig.advisor.settle(Duration::from_secs(30));
    assert!(
        rig.advisor.availability().is_available(),
        "{:?}",
        rig.advisor.availability()
    );
}

#[test]
fn a_signed_out_agy_is_signed_out_and_an_old_one_is_too_old() {
    let rig = Rig::agy("signed_out");
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::SignedOut)
    );
    let rig = Rig::agy("too_old");
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::TooOld)
    );
    assert_eq!(rig.advisor.details().tool_version.as_deref(), Some("1.0.9"));
    // A sign-in check that hangs ends at its limit.
    let rig = Rig::agy_with("signin_hang", |t| {
        t.probe.agy_usage = Duration::from_millis(600)
    });
    rig.probe();
    assert_eq!(
        rig.advisor.availability(),
        Availability::Unavailable(UnavailableReason::ProbeFailed)
    );
}

#[test]
fn not_installed_is_searched_again_by_refresh() {
    let home = TempDir::new().unwrap();
    let mut tuning = Tuning::for_tool(Tool::Claude);
    tuning.min_interval = Duration::ZERO;
    let source = Source::System(locations_in(home.path()));
    let advisor = ClaudeAdvisor::custom(source, tuning);
    advisor.refresh();
    advisor.settle(Duration::from_secs(20));
    assert_eq!(
        advisor.availability(),
        Availability::Unavailable(UnavailableReason::NotInstalled)
    );

    // The person installs it and opens Preferences.
    let path = known_path(Tool::Claude, home.path());
    install(
        Tool::Claude,
        path.parent().unwrap(),
        path.file_name().unwrap().to_str().unwrap(),
    );
    advisor.refresh();
    advisor.settle(Duration::from_secs(20));
    assert!(
        advisor.availability().is_available(),
        "{:?}",
        advisor.availability()
    );
    assert_eq!(advisor.details().path, Some(path));
}

#[test]
fn a_start_of_either_kind_makes_no_request() {
    // The first probe of both advisors (`StartProbe::Now`) finds the program and asks its version; it
    // sends no picture, and for `agy` it does not even make the program contact its service.
    let claude = Rig::claude("signed_in");
    claude.advisor.start();
    claude.advisor.settle(Duration::from_secs(20));
    assert!(claude.requests().is_empty());
    assert_eq!(claude.runs().len(), 2, "claude: --version, auth status");
    assert!(claude.advisor.availability().is_available());

    let agy = Rig::agy("signed_in");
    agy.advisor.start();
    agy.advisor.settle(Duration::from_secs(20));
    assert!(agy.requests().is_empty());
    assert!(
        agy.runs().iter().all(|r| r.is_version()),
        "agy: --version only"
    );
    assert_eq!(agy.runs().len(), 1);
    assert_eq!(agy.advisor.details().tool_version.as_deref(), Some("1.4.7"));
}
