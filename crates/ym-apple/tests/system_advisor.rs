//! `ym_apple::system_advisor()` as the app and `ym` call it, without the test advisor.

use ym_core::apple_auto::{
    AdvisorPicture, Availability, DEFAULT_INSTRUCTIONS, FailureKind, UnavailableReason,
};

#[test]
fn the_system_advisor_reports_this_computer() {
    if std::env::var_os(ym_apple::TEST_ADVISOR_ENV).is_some() {
        // The test advisor was asked for; the crate's own tests cover it.
        return;
    }
    let advisor = ym_apple::system_advisor();
    let availability = advisor.availability();
    println!("Apple Auto here: {availability:?}");
    let expected_reason = if !cfg!(target_os = "macos") {
        Some(UnavailableReason::NotAMac)
    } else if !cfg!(target_arch = "aarch64") {
        Some(UnavailableReason::IntelMac)
    } else if !ym_apple::BRIDGE_BUILT {
        Some(UnavailableReason::BuiltWithoutSupport)
    } else {
        // Apple's model decides; the xcode-27 runner (a virtual Mac) is expected to say
        // device_not_eligible or apple_intelligence_off.
        None
    };
    if let Some(reason) = expected_reason {
        assert_eq!(availability, Availability::Unavailable(reason));
    }
    if !availability.is_available() {
        let picture = AdvisorPicture {
            width: 2,
            height: 2,
            rgb: vec![128; 12],
        };
        let error = advisor.ask(&picture, DEFAULT_INSTRUCTIONS).unwrap_err();
        assert_eq!(error.kind, FailureKind::Unavailable, "{error}");
    }
}
