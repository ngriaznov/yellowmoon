//! The two programs Yellowmoon can start (`docs/22` sections 3 and 4): what they are called, the oldest
//! version each is used from, and the limits of its probes.

use std::ffi::OsString;
use std::time::Duration;

use semver::Version;
use ym_core::auto_provider::{
    AGY_TIMEOUT_S, AutoProvider, BREAKER_COOLDOWN_S, BREAKER_FAILURES, CLAUDE_TIMEOUT_S,
    PROBE_MIN_INTERVAL_S,
};

/// One of the person's own command-line programs that can grade a photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    /// Claude Code (`claude`).
    Claude,
    /// Antigravity (`agy`).
    Agy,
}

/// The first Claude Code version with `--safe-mode` (`docs/22` section 3.2).
pub const CLAUDE_MIN_VERSION: Version = Version::new(2, 1, 169);

/// The first `agy` version with the schema rules and exit code 3 (`docs/22` section 3.2).
pub const AGY_MIN_VERSION: Version = Version::new(1, 2, 14);

impl Tool {
    /// Both programs, Claude Code first.
    pub const ALL: [Tool; 2] = [Tool::Claude, Tool::Agy];

    /// The name of the executable without an extension: `claude`, `agy`.
    #[must_use]
    pub const fn program(self) -> &'static str {
        match self {
            Tool::Claude => "claude",
            Tool::Agy => "agy",
        }
    }

    /// The name of the executable's file on this operating system (`claude.exe` on Windows).
    #[must_use]
    pub fn file_name(self) -> String {
        format!("{}{}", self.program(), std::env::consts::EXE_SUFFIX)
    }

    /// What the person sees: `Claude Code`, `Antigravity`.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Tool::Claude => "Claude Code",
            Tool::Agy => "Antigravity",
        }
    }

    /// The provider this program is.
    #[must_use]
    pub const fn provider(self) -> AutoProvider {
        match self {
            Tool::Claude => AutoProvider::Claude,
            Tool::Agy => AutoProvider::Agy,
        }
    }

    /// The oldest version that is used.
    #[must_use]
    pub const fn min_version(self) -> Version {
        match self {
            Tool::Claude => CLAUDE_MIN_VERSION,
            Tool::Agy => AGY_MIN_VERSION,
        }
    }
}

/// How long each kind of probe may take before its process tree is killed and the answer is
/// `probe_failed` (`docs/22` section 3.2). The tests shorten them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeLimits {
    /// The login-shell lookup, per attempt.
    pub shell: Duration,
    /// A `--version` run.
    pub version: Duration,
    /// `claude auth status`.
    pub claude_auth: Duration,
    /// `agy -p /usage --output-format json`.
    pub agy_usage: Duration,
}

impl Default for ProbeLimits {
    fn default() -> Self {
        Self {
            shell: Duration::from_secs(3),
            version: Duration::from_secs(5),
            claude_auth: Duration::from_secs(10),
            agy_usage: Duration::from_secs(15),
        }
    }
}

/// Every number and switch the tests turn, with the spec's values as the default: the time limits, the
/// breaker, and the variables added to a child's environment (the scenario of a test double).
#[derive(Debug, Clone)]
pub struct Tuning {
    /// The probes' time limits.
    pub probe: ProbeLimits,
    /// The hard stop of one request: 90 s for Claude Code, 120 s for `agy`.
    pub request_timeout: Duration,
    /// `agy`: how long our clock must have run for a `INTERRUPTED` envelope to be read as its own
    /// time limit having fired (105 s, `docs/22` section 8.3).
    pub agy_own_limit: Duration,
    /// The minimum time between two full probes started by `refresh` (5 s).
    pub min_interval: Duration,
    /// Failures in a row that open the breaker (3).
    pub breaker_failures: u32,
    /// How long the open breaker refuses requests (60 s).
    pub breaker_cooldown: Duration,
    /// Variables set in every child after the usual changes (a test double's scenario); none in
    /// production.
    pub extra_env: Vec<(OsString, OsString)>,
    /// The model to ask for, instead of `YELLOWMOON_CLAUDE_MODEL` / `YELLOWMOON_AGY_MODEL` and the
    /// default.
    pub model: Option<String>,
    /// The effort to ask for, instead of `YELLOWMOON_CLAUDE_EFFORT` / `YELLOWMOON_AGY_EFFORT` and `low`.
    pub effort: Option<String>,
}

impl Tuning {
    /// The spec's values for `tool`.
    #[must_use]
    pub fn for_tool(tool: Tool) -> Self {
        let seconds = match tool {
            Tool::Claude => CLAUDE_TIMEOUT_S,
            Tool::Agy => AGY_TIMEOUT_S,
        };
        Self {
            probe: ProbeLimits::default(),
            request_timeout: Duration::from_secs(u64::from(seconds)),
            agy_own_limit: Duration::from_secs(105),
            min_interval: Duration::from_secs(PROBE_MIN_INTERVAL_S),
            breaker_failures: BREAKER_FAILURES,
            breaker_cooldown: Duration::from_secs(BREAKER_COOLDOWN_S),
            extra_env: Vec::new(),
            model: None,
            effort: None,
        }
    }

    /// Adds a variable to every child's environment.
    #[must_use]
    pub fn with_env(mut self, name: &str, value: &str) -> Self {
        self.extra_env.push((name.into(), value.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_core::auto_provider::{AGY_MIN_VERSION as CORE_AGY, CLAUDE_MIN_VERSION as CORE_CLAUDE};

    #[test]
    fn the_minimum_versions_are_the_cores() {
        assert_eq!(Tool::Claude.min_version().to_string(), CORE_CLAUDE);
        assert_eq!(Tool::Agy.min_version().to_string(), CORE_AGY);
        assert_eq!(Tuning::for_tool(Tool::Claude).request_timeout, Duration::from_secs(90));
        assert_eq!(Tuning::for_tool(Tool::Agy).request_timeout, Duration::from_secs(120));
    }

    #[test]
    fn names_and_minimum_versions() {
        assert_eq!(Tool::Claude.program(), "claude");
        assert_eq!(Tool::Agy.program(), "agy");
        assert_eq!(Tool::Claude.min_version().to_string(), "2.1.169");
        assert_eq!(Tool::Agy.min_version().to_string(), "1.2.14");
        assert_eq!(Tool::Claude.display_name(), "Claude Code");
        assert_eq!(Tool::Agy.display_name(), "Antigravity");
    }
}
