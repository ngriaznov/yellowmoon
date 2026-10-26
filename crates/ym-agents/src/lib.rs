//! Auto providers that run the person's own Claude Code (`claude`) and Antigravity (`agy`) programs
//! (`docs/22-auto-providers.md`).
//!
//! [`system_advisors`] gives the app and `ym` the two [`AutoAdvisor`]s of this computer: [`ClaudeAdvisor`]
//! and [`AgyAdvisor`]. Each starts the person's own program as a child process, hands it a 768 px picture
//! of the photo and the instructions, and reads its answer; the program talks to its own service. This
//! crate opens no network connection, uses no HTTP client (`tests/no_network.rs` checks its
//! dependency tree) and never touches the person's credentials (hard rule 7).
//!
//! The pieces: `discovery` finds a program (the known install folders, `PATH`, the login shell),
//! `process` is the one place a child is started (a process group or a job object, caps, a deadline, a
//! cancel token, redaction), `probe` and `advisor` hold the cached availability and the probes that fill
//! it without ever blocking the caller, `breaker` stops asking a program that keeps failing, `jpeg` and
//! `schema` make the request, `claude` and `agy` are the two advisors, and `env_hook` serves the test
//! hook `YELLOWMOON_TEST_AUTO_PROVIDERS`. The two test doubles, `ym-fake-claude` and `ym-fake-agy`, are
//! binaries of this crate; they are not part of any shipped build.

#![warn(missing_docs)]

/// Declares the internal modules: private normally, public (and hidden from the documentation) with the
/// `test-support` feature, so that this crate's integration tests can reach them.
macro_rules! internal_modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "test-support")]
            #[doc(hidden)]
            pub mod $name;
            #[cfg(not(feature = "test-support"))]
            mod $name;
        )*
    };
}

internal_modules!(
    advisor, agy, breaker, claude, discovery, env_hook, jpeg, probe, process, schema, tool,
);

mod system;
#[cfg(feature = "test-support")]
pub mod testing;

pub use agy::AgyAdvisor;
pub use claude::ClaudeAdvisor;
pub use system::{StartProbe, TEST_PROVIDERS_ENV, system_advisors};
pub use tool::Tool;

#[allow(unused_imports)]
use ym_core::apple_auto::AutoAdvisor;
