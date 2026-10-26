//! The test hook `YELLOWMOON_TEST_AUTO_PROVIDERS` (`docs/22` section 15.3), like
//! `YELLOWMOON_TEST_APPLE_AUTO` for Apple: the tests of `ym` and the app choose what the two providers
//! are without a real program being looked for.
//!
//! | Value | Effect |
//! |---|---|
//! | unset (or empty) | the real advisors with real discovery |
//! | `none` | no program is looked for; Claude Code and Antigravity are `not_installed` |
//! | `claude=available,agy=signed_out` | a provider that is not listed is `not_installed`; the value is a state (`available`, `signed_out`, `not_installed`, `too_old`, `unavailable`, `checking`), an in-process `FakeProvider` (no process, deterministic answers), or `exec:<absolute path>`, the real advisor over that program with discovery skipped |
//! | anything else | printed once on standard error as an error; the provider is `not_installed` |

use std::path::PathBuf;
use std::sync::Arc;

use ym_core::apple_auto::{AutoAdvisor, Availability, UnavailableReason};
use ym_core::auto_provider::{AdvisorDetails, AutoProvider, FakeProvider};

use crate::agy::AgyAdvisor;
use crate::claude::ClaudeAdvisor;
use crate::tool::Tool;

/// What the hook says about one provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// An in-process fake in this state.
    State(Availability),
    /// The real advisor over this program.
    Exec(PathBuf),
}

/// The hook's value, read: an entry for each provider that has one, and what to say about the rest.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Parsed {
    /// The Claude Code entry.
    pub claude: Option<Entry>,
    /// The Antigravity entry.
    pub agy: Option<Entry>,
    /// What was wrong with the value: one line each.
    pub errors: Vec<String>,
}

/// The state a name stands for, or `None` when it is not one.
fn state(name: &str) -> Option<Availability> {
    let reason = match name {
        "not_installed" => UnavailableReason::NotInstalled,
        "too_old" => UnavailableReason::TooOld,
        "signed_out" => UnavailableReason::SignedOut,
        "unavailable" => UnavailableReason::ProbeFailed,
        "checking" => UnavailableReason::Checking,
        _ => return None,
    };
    Some(Availability::Unavailable(reason))
}

/// Reads the hook's value (not unset and not empty).
#[must_use]
pub fn parse(value: &str) -> Parsed {
    let mut parsed = Parsed::default();
    if value.trim() == "none" {
        return parsed;
    }
    for item in value.split(',') {
        let item = item.trim();
        let Some((name, what)) = item.split_once('=') else {
            parsed
                .errors
                .push(format!("{item:?} is not provider=value"));
            continue;
        };
        let entry = if let Some(path) = what.trim().strip_prefix("exec:") {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                parsed
                    .errors
                    .push(format!("exec: needs an absolute path, not {path:?}"));
                continue;
            }
            Entry::Exec(path)
        } else if what.trim() == "available" {
            // The model is the fake's; the entry is made for the provider below.
            Entry::State(Availability::Unavailable(UnavailableReason::Checking))
        } else if let Some(availability) = state(what.trim()) {
            Entry::State(availability)
        } else {
            parsed
                .errors
                .push(format!("{:?} is not a state or exec:<path>", what.trim()));
            continue;
        };
        // `available` is spelled out again so that the placeholder above is never used.
        let entry = if what.trim() == "available" {
            Entry::State(Availability::Available(FakeProvider::info_for(
                AutoProvider::Claude,
            )))
        } else {
            entry
        };
        match name.trim() {
            "claude" => parsed.claude = Some(entry),
            "agy" => parsed.agy = Some(entry),
            other => parsed
                .errors
                .push(format!("{other:?} is not claude or agy")),
        }
    }
    parsed
}

/// The advisor an entry makes for `tool`.
fn advisor(tool: Tool, entry: Option<Entry>) -> Arc<dyn AutoAdvisor> {
    let provider = tool.provider();
    match entry {
        Some(Entry::Exec(path)) => match tool {
            Tool::Claude => ClaudeAdvisor::with_program(path),
            Tool::Agy => AgyAdvisor::with_program(path),
        },
        Some(Entry::State(Availability::Available(_))) => Arc::new(
            FakeProvider::available(provider).with_details(fake_details()),
        ),
        Some(Entry::State(availability)) => {
            let details = match availability {
                Availability::Unavailable(UnavailableReason::NotInstalled) => {
                    AdvisorDetails::default()
                }
                _ => fake_details(),
            };
            Arc::new(FakeProvider::new(provider, availability).with_details(details))
        }
        None => Arc::new(FakeProvider::new(
            provider,
            Availability::Unavailable(UnavailableReason::NotInstalled),
        )),
    }
}

/// What a fake that "found" a program says about it.
fn fake_details() -> AdvisorDetails {
    AdvisorDetails {
        tool_version: Some("0.0.0-test".to_owned()),
        ..AdvisorDetails::default()
    }
}

/// The two advisors for a value of the hook.
#[must_use]
pub fn advisors(value: &str) -> Vec<(AutoProvider, Arc<dyn AutoAdvisor>)> {
    let parsed = parse(value);
    for error in &parsed.errors {
        eprintln!("YELLOWMOON_TEST_AUTO_PROVIDERS: {error}");
    }
    vec![
        (AutoProvider::Claude, advisor(Tool::Claude, parsed.claude)),
        (AutoProvider::Agy, advisor(Tool::Agy, parsed.agy)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_lists_nothing() {
        let parsed = parse("none");
        assert_eq!(parsed, Parsed::default());
    }

    #[test]
    fn a_list_names_states_and_programs() {
        let parsed = parse("claude=available, agy=signed_out");
        assert!(matches!(
            parsed.claude,
            Some(Entry::State(Availability::Available(_)))
        ));
        assert_eq!(
            parsed.agy,
            Some(Entry::State(Availability::Unavailable(
                UnavailableReason::SignedOut
            )))
        );
        assert!(parsed.errors.is_empty());
        let exec = if cfg!(windows) {
            "claude=exec:C:\\tools\\ym-fake-claude.exe"
        } else {
            "claude=exec:/tools/ym-fake-claude"
        };
        assert!(matches!(parse(exec).claude, Some(Entry::Exec(_))));
        for (text, state) in [
            ("agy=too_old", UnavailableReason::TooOld),
            ("agy=unavailable", UnavailableReason::ProbeFailed),
            ("agy=checking", UnavailableReason::Checking),
            ("agy=not_installed", UnavailableReason::NotInstalled),
        ] {
            assert_eq!(
                parse(text).agy,
                Some(Entry::State(Availability::Unavailable(state)))
            );
        }
    }

    #[test]
    fn anything_else_is_an_error() {
        for bad in ["claude", "claude=maybe", "gemini=available", "claude=exec:relative/path"] {
            let parsed = parse(bad);
            assert_eq!(parsed.errors.len(), 1, "{bad}");
            assert!(parsed.claude.is_none() && parsed.agy.is_none(), "{bad}");
        }
    }
}
