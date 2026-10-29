//! The advisors of this computer (`docs/22` section 15.1).

use std::sync::Arc;

use ym_core::apple_auto::AutoAdvisor;
use ym_core::auto_provider::AutoProvider;

use crate::agy::AgyAdvisor;
use crate::claude::ClaudeAdvisor;
use crate::env_hook;

/// The environment variable of the test hook (`docs/22` section 15.3), like `YELLOWMOON_TEST_APPLE_AUTO`.
pub const TEST_PROVIDERS_ENV: &str = "YELLOWMOON_TEST_AUTO_PROVIDERS";

/// Whether [`system_advisors`] starts the first probe.
///
/// Both front ends pass [`StartProbe::Never`] (owner, 2026-10-29: a program is started only when
/// grading needs it, `docs/22` section 3.3): the core asks the chosen program to check when photos are
/// opened or an export starts, and every program when Preferences opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartProbe {
    /// Each advisor is already probing in the background when it is returned (Claude Code fully;
    /// Antigravity up to its version, because its sign-in check is the one probe known to make its
    /// program contact a network, so it runs only when asked for). **No shipped front end uses it
    /// any more**: it would start a program at every launch whatever the person chose.
    Now,
    /// The app and `ym`: no probe starts, so a run that does not use a program never starts one; the
    /// core asks the program it uses (or Preferences, every program) to check, when it needs to.
    Never,
}

/// The advisors for Claude Code and Antigravity (`docs/22` section 15.1).
///
/// With [`TEST_PROVIDERS_ENV`] unset these are [`ClaudeAdvisor::system`] and [`AgyAdvisor::system`], real
/// discovery included; with it set, the advisors the hook describes (in-process fakes, or the real
/// advisors over a test double; `start` has no effect on the fakes). Leftover `ym-auto-*` folders of a
/// crash are swept first (`docs/22` section 2.1).
#[must_use]
pub fn system_advisors(start: StartProbe) -> Vec<(AutoProvider, Arc<dyn AutoAdvisor>)> {
    if let Some(value) = std::env::var(TEST_PROVIDERS_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        return env_hook::advisors(&value);
    }
    crate::process::sweep_stale();
    let claude = ClaudeAdvisor::system();
    let agy = AgyAdvisor::system();
    if start == StartProbe::Now {
        claude.start();
        agy.start();
    }
    vec![
        (AutoProvider::Claude, claude as Arc<dyn AutoAdvisor>),
        (AutoProvider::Agy, agy as Arc<dyn AutoAdvisor>),
    ]
}
