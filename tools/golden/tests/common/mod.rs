//! Helpers the integration tests of the golden runner share.

use std::ffi::OsStr;

/// Whether an environment value switches a `YM_REQUIRE_*` demand on: set, not empty, not `0`.
/// (`YM_REQUIRE_SYNTHETIC=0` and `YM_REQUIRE_SYNTHETIC=` mean "not required", as in every other test
/// that reads the variable: `docs/13-testing.md`.)
pub fn demand_is_on(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether the environment demands the synthetic set (`YM_REQUIRE_SYNTHETIC`, set by the
/// `synthetic-set` CI job): a missing set, or a missing `ym`, is then a failure of the test. Without
/// it the test says why it was skipped and passes, so a developer's machine without the generated
/// files stays green.
pub fn synthetic_required() -> bool {
    demand_is_on(std::env::var_os("YM_REQUIRE_SYNTHETIC").as_deref())
}
