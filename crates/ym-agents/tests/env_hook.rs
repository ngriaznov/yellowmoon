//! The test hook `YELLOWMOON_TEST_AUTO_PROVIDERS` (`docs/22` section 15.3) through `system_advisors`.
//!
//! One test in this file, because it sets an environment variable of this process.

mod common;

use std::time::Duration;

use common::{FAKE_CLAUDE, dark_flat};
use ym_agents::{StartProbe, TEST_PROVIDERS_ENV, system_advisors};
use ym_core::apple_auto::{AutoAdvisor, Availability, FakeAdvisor, UnavailableReason};
use ym_core::auto_provider::AutoProvider;

fn advisors(
    value: Option<&str>,
) -> (
    std::sync::Arc<dyn AutoAdvisor>,
    std::sync::Arc<dyn AutoAdvisor>,
) {
    // SAFETY: this is the only test of this binary; no other thread reads the environment.
    unsafe {
        match value {
            Some(v) => std::env::set_var(TEST_PROVIDERS_ENV, v),
            None => std::env::remove_var(TEST_PROVIDERS_ENV),
        }
    }
    let list = system_advisors(StartProbe::Never);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].0, AutoProvider::Claude);
    assert_eq!(list[1].0, AutoProvider::Agy);
    (list[0].1.clone(), list[1].1.clone())
}

fn reason(advisor: &dyn AutoAdvisor) -> Option<UnavailableReason> {
    match advisor.availability() {
        Availability::Unavailable(reason) => Some(reason),
        Availability::Available(_) => None,
    }
}

#[test]
fn the_hook_chooses_what_the_providers_are() {
    // `none`: neither is installed, and no program is looked for.
    let (claude, agy) = advisors(Some("none"));
    assert_eq!(reason(&*claude), Some(UnavailableReason::NotInstalled));
    assert_eq!(reason(&*agy), Some(UnavailableReason::NotInstalled));

    // A list: states are in-process fakes; a provider not listed is not installed.
    let (claude, agy) = advisors(Some("claude=available,agy=signed_out"));
    let Availability::Available(info) = claude.availability() else {
        panic!("claude is not available");
    };
    assert_eq!(info.provider, AutoProvider::Claude);
    assert_eq!(info.model, "Test Claude Code");
    assert_eq!(info.tool.as_ref().unwrap().version, "0.0.0-test");
    assert_eq!(reason(&*agy), Some(UnavailableReason::SignedOut));
    assert_eq!(agy.details().tool_version.as_deref(), Some("0.0.0-test"));
    // The fake answers like FakeAdvisor, with no process.
    let picture = dark_flat();
    assert_eq!(
        claude.ask(&picture, "Plain.").unwrap().steps,
        FakeAdvisor.ask(&picture, "Plain.").unwrap().steps
    );
    let (claude, agy) = advisors(Some("agy=too_old"));
    assert_eq!(reason(&*claude), Some(UnavailableReason::NotInstalled));
    assert_eq!(reason(&*agy), Some(UnavailableReason::TooOld));
    let (_, agy) = advisors(Some("agy=unavailable"));
    assert_eq!(reason(&*agy), Some(UnavailableReason::ProbeFailed));
    let (_, agy) = advisors(Some("agy=checking"));
    assert_eq!(reason(&*agy), Some(UnavailableReason::Checking));

    // `exec:` is the real advisor over that program: checking until a refresh, then what the double says.
    let (claude, _) = advisors(Some(&format!("claude=exec:{FAKE_CLAUDE}")));
    assert_eq!(
        reason(&*claude),
        Some(UnavailableReason::Checking),
        "no probe was started"
    );
    claude.refresh();
    claude.settle(Duration::from_secs(30));
    assert!(
        claude.availability().is_available(),
        "{:?}",
        claude.availability()
    );
    assert!(claude.details().path.is_some());

    // Anything else: an error on standard error, and the provider is not installed.
    let (claude, agy) = advisors(Some("claude=maybe,nonsense"));
    assert_eq!(reason(&*claude), Some(UnavailableReason::NotInstalled));
    assert_eq!(reason(&*agy), Some(UnavailableReason::NotInstalled));

    // Unset: the real advisors, which have probed nothing (`StartProbe::Never`).
    let (claude, agy) = advisors(None);
    assert_eq!(reason(&*claude), Some(UnavailableReason::Checking));
    assert_eq!(reason(&*agy), Some(UnavailableReason::Checking));
    assert_eq!((claude.concurrency(), agy.concurrency()), (2, 1));
}
