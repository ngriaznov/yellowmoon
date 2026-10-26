//! Helpers for the tests of other crates (feature `test-support`): where the two test doubles are
//! (`docs/22` sections 15.2 and 15.3).

use std::path::PathBuf;

use crate::tool::Tool;

/// The variable that makes a missing test double a failure instead of a skip. The CI job sets it.
pub const REQUIRE_ENV: &str = "YM_REQUIRE_FAKE_CLI";

/// Where the test double of `tool` is: next to the test executable's profile folder
/// (`target/debug/ym-fake-claude`, `.exe` on Windows). The file may not exist: `cargo test -p ym-agents`
/// (or a workspace build) builds it.
#[must_use]
pub fn fake_cli_path(tool: Tool) -> PathBuf {
    let name = match tool {
        Tool::Claude => "ym-fake-claude",
        Tool::Agy => "ym-fake-agy",
    };
    let exe = std::env::current_exe().unwrap_or_default();
    // A test executable is in `<profile>/deps/`; a binary run by hand is in `<profile>/`.
    let mut dir = exe.parent().map(PathBuf::from).unwrap_or_default();
    if dir.file_name().is_some_and(|n| n == "deps") {
        dir.pop();
    }
    dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// [`fake_cli_path`] when the double exists. When it does not: a panic if `YM_REQUIRE_FAKE_CLI` is
/// set, otherwise a printed `SKIPPED` line and `None`, as the repository's other tests that need a
/// tool do.
#[must_use]
pub fn require_fake_cli(tool: Tool) -> Option<PathBuf> {
    let path = fake_cli_path(tool);
    if path.is_file() {
        return Some(path);
    }
    let message = format!(
        "the test double {} is not built (run `cargo test -p ym-agents` first)",
        path.display()
    );
    assert!(
        std::env::var_os(REQUIRE_ENV).is_none(),
        "{message}, and {REQUIRE_ENV} is set"
    );
    eprintln!("SKIPPED: {message}");
    None
}
