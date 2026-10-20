//! Apple Auto's bridge to Apple's on-device model (`docs/19-apple-auto.md`, sections 1, 4, 7 and 11).
//!
//! [`system_advisor`] gives the app and `ym` the [`AutoAdvisor`] of this computer: [`AppleAdvisor`],
//! which asks Apple Intelligence's on-device model through Foundation Models (`swift/AppleAuto.swift`)
//! where it can and reports why not everywhere else, or, when the tests ask for one with
//! `YELLOWMOON_TEST_APPLE_AUTO`, [`FakeAdvisor`] (`fake`) or an advisor that is never available
//! (`none`).
//!
//! The Swift is compiled only for Apple silicon Macs with the macOS 27 SDK or later (`build.rs`);
//! elsewhere this crate is a stub that reports `not_a_mac`, `intel_mac` or `built_without_support`.
//! The binaries that depend on it link FoundationModels weakly when the bridge is in (`build.rs`
//! explains why and how), so they still start on macOS 14 to 26.
//!
//! The model runs on this Mac and nothing is sent anywhere (hard rule 7). Answers are not stored
//! (hard rule 4); the core keeps them in the session's memory. Retries are the core's.
//!
//! `examples/probe.rs` asks the model about a synthetic picture: `cargo run -p ym-apple --release
//! --example probe`.

#![warn(missing_docs)]

// Without the bridge only the tests use the conversions; they are tested on every platform.
#[cfg_attr(not(ym_apple_bridge), allow(dead_code))]
mod abi;
#[cfg(ym_apple_bridge)]
mod bridge;
#[cfg(test)]
mod source_tests;

use std::sync::Arc;

use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorPicture, AutoAdvisor, Availability, FailureKind,
    FakeAdvisor, UnavailableReason,
};

/// The environment variable that selects the test advisor (`docs/19` section 11), like
/// `YELLOWMOON_GPU` selects the device.
pub const TEST_ADVISOR_ENV: &str = "YELLOWMOON_TEST_APPLE_AUTO";

/// The value of [`TEST_ADVISOR_ENV`] that selects [`FakeAdvisor`] on any platform.
pub const TEST_ADVISOR_FAKE: &str = "fake";

/// The value of [`TEST_ADVISOR_ENV`] that selects [`NoAdvisor`]: Apple Auto is not available, for
/// the reason a build without the bridge gives on this platform. Tests that expect the rule set it,
/// so that they give the same answer on a Mac with Apple Intelligence.
pub const TEST_ADVISOR_NONE: &str = "none";

/// Why a build without the Swift bridge has no Apple Auto on this platform: `not_a_mac`,
/// `intel_mac`, or `built_without_support` on Apple silicon.
pub const NO_BRIDGE_REASON: UnavailableReason = if cfg!(not(target_os = "macos")) {
    UnavailableReason::NotAMac
} else if cfg!(not(target_arch = "aarch64")) {
    UnavailableReason::IntelMac
} else {
    UnavailableReason::BuiltWithoutSupport
};

/// Whether this build carries the Swift bridge (Apple silicon, built with the macOS 27 SDK or
/// later). Without it [`AppleAdvisor`] is never available.
pub const BRIDGE_BUILT: bool = cfg!(ym_apple_bridge);

/// The advisor the app and `ym` hand to the core: [`FakeAdvisor`] when
/// `YELLOWMOON_TEST_APPLE_AUTO=fake` is set, [`NoAdvisor`] when it is `none`, else [`AppleAdvisor`].
#[must_use]
pub fn system_advisor() -> Arc<dyn AutoAdvisor> {
    let test_advisor = std::env::var(TEST_ADVISOR_ENV).ok();
    advisor_for(test_advisor.as_deref())
}

/// The advisor for a value of [`TEST_ADVISOR_ENV`] (`None` when it is not set).
fn advisor_for(test_advisor: Option<&str>) -> Arc<dyn AutoAdvisor> {
    match test_advisor {
        Some(TEST_ADVISOR_FAKE) => Arc::new(FakeAdvisor),
        Some(TEST_ADVISOR_NONE) => Arc::new(NoAdvisor),
        _ => Arc::new(AppleAdvisor),
    }
}

/// An advisor that is never available, for [`NO_BRIDGE_REASON`]: what the tests that expect the
/// rule use (`YELLOWMOON_TEST_APPLE_AUTO=none`), whatever this computer could do.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoAdvisor;

impl AutoAdvisor for NoAdvisor {
    fn availability(&self) -> Availability {
        Availability::Unavailable(NO_BRIDGE_REASON)
    }

    fn ask(
        &self,
        _picture: &AdvisorPicture,
        _instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        Err(AdvisorError::new(
            FailureKind::Unavailable,
            format!(
                "Apple Auto is not available on this computer ({}).",
                NO_BRIDGE_REASON.words()
            ),
        ))
    }
}

/// Apple Intelligence's on-device model, through Foundation Models (`docs/19` sections 1 and 4).
///
/// [`availability`](AutoAdvisor::availability) is cheap (property reads, no model call).
/// [`ask`](AutoAdvisor::ask) blocks the calling thread for one request, at most
/// [`REQUEST_TIMEOUT_MS`](ym_core::apple_auto::REQUEST_TIMEOUT_MS); call it from the session's Apple
/// Auto worker, never from the main thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppleAdvisor;

impl AutoAdvisor for AppleAdvisor {
    fn availability(&self) -> Availability {
        platform::availability()
    }

    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        platform::ask(picture, instructions)
    }
}

/// With the Swift bridge: Apple's model answers.
#[cfg(ym_apple_bridge)]
mod platform {
    pub(crate) use crate::bridge::{ask, availability};
}

/// Without the Swift bridge: never available, for the reason this build gives.
#[cfg(not(ym_apple_bridge))]
mod platform {
    use ym_core::apple_auto::{
        AdvisorAnswer, AdvisorError, AdvisorPicture, AutoAdvisor, Availability,
    };

    use crate::NoAdvisor;

    pub(crate) fn availability() -> Availability {
        NoAdvisor.availability()
    }

    pub(crate) fn ask(
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        NoAdvisor.ask(picture, instructions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_core::apple_auto::{DEFAULT_INSTRUCTIONS, FailureKind, UnavailableReason};

    fn picture(value: u8) -> AdvisorPicture {
        AdvisorPicture {
            width: 4,
            height: 3,
            rgb: vec![value; 4 * 3 * 3],
        }
    }

    #[test]
    fn fake_selects_the_test_advisor() {
        let advisor = advisor_for(Some("fake"));
        assert_eq!(
            advisor.availability(),
            Availability::Available(FakeAdvisor::info())
        );
        assert_eq!(
            advisor.ask(&picture(40), DEFAULT_INSTRUCTIONS),
            FakeAdvisor.ask(&picture(40), DEFAULT_INSTRUCTIONS)
        );
        let answer = advisor.ask(&picture(40), "Grade it warm.").unwrap();
        assert_eq!(
            (
                answer.steps.exposure,
                answer.steps.contrast,
                answer.steps.blacks
            ),
            (Some(1), Some(1), None)
        );
        assert_eq!(answer.plan, "Test plan: Grade");
    }

    #[test]
    fn none_is_never_available() {
        let advisor = advisor_for(Some("none"));
        assert_eq!(
            advisor.availability(),
            Availability::Unavailable(NO_BRIDGE_REASON)
        );
        assert_eq!(
            advisor
                .ask(&picture(40), DEFAULT_INSTRUCTIONS)
                .unwrap_err()
                .kind,
            FailureKind::Unavailable
        );
    }

    #[test]
    fn anything_else_selects_apple() {
        for value in [
            None,
            Some(""),
            Some("FAKE"),
            Some("off"),
            Some(" fake"),
            Some("NONE"),
        ] {
            let advisor = advisor_for(value);
            assert_ne!(
                advisor.availability(),
                Availability::Available(FakeAdvisor::info()),
                "{value:?}"
            );
        }
    }

    /// The reasons that only a build without the bridge gives.
    const STATIC_REASONS: [UnavailableReason; 3] = [
        UnavailableReason::NotAMac,
        UnavailableReason::IntelMac,
        UnavailableReason::BuiltWithoutSupport,
    ];

    #[test]
    fn apple_reports_the_reason_of_this_platform() {
        let availability = AppleAdvisor.availability();
        println!("Apple Auto on this computer: {availability:?}");
        if cfg!(ym_apple_bridge) {
            // A Mac that can ask: available, or a reason Apple's model gives (a virtual Mac has no
            // Apple Intelligence). Never one of the build's own reasons.
            if let Availability::Unavailable(reason) = availability {
                assert!(!STATIC_REASONS.contains(&reason), "{reason:?}");
            }
        } else if cfg!(not(target_os = "macos")) {
            assert_eq!(
                availability,
                Availability::Unavailable(UnavailableReason::NotAMac)
            );
        } else if cfg!(not(target_arch = "aarch64")) {
            assert_eq!(
                availability,
                Availability::Unavailable(UnavailableReason::IntelMac)
            );
        } else {
            assert_eq!(
                availability,
                Availability::Unavailable(UnavailableReason::BuiltWithoutSupport)
            );
        }
        assert_eq!(BRIDGE_BUILT, cfg!(ym_apple_bridge));
    }

    #[test]
    fn asking_where_apple_auto_is_not_available_fails_as_unavailable() {
        if AppleAdvisor.availability().is_available() {
            // A Mac with Apple Intelligence: the probe example and the owner's checks cover it.
            return;
        }
        let error = AppleAdvisor
            .ask(&picture(128), DEFAULT_INSTRUCTIONS)
            .unwrap_err();
        assert_eq!(error.kind, FailureKind::Unavailable, "{error}");
    }

    #[test]
    fn a_malformed_picture_never_reaches_the_model() {
        let bad = AdvisorPicture {
            width: 4,
            height: 3,
            rgb: vec![0; 5],
        };
        let error = AppleAdvisor.ask(&bad, DEFAULT_INSTRUCTIONS).unwrap_err();
        // Without the bridge every request is unavailable; with it the picture is refused first.
        let expected = if cfg!(ym_apple_bridge) {
            FailureKind::Other
        } else {
            FailureKind::Unavailable
        };
        assert_eq!(error.kind, expected, "{error}");
    }
}
