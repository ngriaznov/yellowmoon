//! What the two advisors share (`docs/22` sections 3, 9): the cached reading of the program, the probes
//! that fill it, the breaker, cancelling, and the skeleton of one request.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, Availability, FailureKind, UnavailableReason,
};
use ym_core::auto_provider::{
    AdvisorDetails, CLAUDE_DEFAULT_MODEL, CLI_EFFORT, ChangeListener, ToolInfo, failure_words,
};

use crate::breaker::{Breaker, Record};
use crate::discovery::{Discovery, Locations, check_program, locate};
use crate::probe::{Level, Prober};
use crate::process::{CancelSource, CancelToken, EnvProfile, Outcome, RunSpec, Workdir, run};
use crate::tool::{Tool, Tuning};

/// Where a program comes from.
#[derive(Debug, Clone)]
pub enum Source {
    /// Look for it (`docs/22` section 4).
    System(Locations),
    /// This program: discovery is skipped (`with_program`, `exec:`).
    Program(PathBuf),
}

/// What is known about one program and whether it can answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// Whether it can answer, and for the model's sake which model it is asked for.
    pub availability: Availability,
    /// Where it was found.
    pub path: Option<PathBuf>,
    /// Its version, and how it is signed in, when known.
    pub tool: Option<ToolInfo>,
}

impl Reading {
    /// A program that cannot answer, for `reason`, with nothing known about it.
    #[must_use]
    pub fn unavailable(reason: UnavailableReason) -> Self {
        Self {
            availability: Availability::Unavailable(reason),
            path: None,
            tool: None,
        }
    }
}

/// How a request failed, before the breaker and the sign-out rule have looked at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fail {
    /// The request was cancelled: nothing is stored.
    Cancelled,
    /// A failure of this kind with this message.
    Failed(FailureKind, String),
}

impl Fail {
    /// A failure of `kind` with `message`.
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self::Failed(kind, message.into())
    }
}

/// The state shared by an advisor's requests and probes.
pub struct Core {
    tool: Tool,
    source: Source,
    tuning: Tuning,
    prober: Prober<Reading>,
    breaker: Breaker,
    cancel: CancelSource,
}

impl Core {
    /// A core that has probed nothing yet: `checking`.
    #[must_use]
    pub fn new(tool: Tool, source: Source, tuning: Tuning) -> Self {
        let min_interval = tuning.min_interval;
        let breaker = Breaker::with(tuning.breaker_failures, tuning.breaker_cooldown);
        Self {
            tool,
            source,
            tuning,
            prober: Prober::new(Reading::unavailable(UnavailableReason::Checking), min_interval),
            breaker,
            cancel: CancelSource::new(),
        }
    }

    /// The tuning in use.
    #[must_use]
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// The model asked for: the tuning's, the developer variable's, else the default (`docs/22` D18).
    #[must_use]
    pub fn model_alias(&self) -> String {
        let (var, default) = match self.tool {
            Tool::Claude => ("YELLOWMOON_CLAUDE_MODEL", Some(CLAUDE_DEFAULT_MODEL)),
            Tool::Agy => ("YELLOWMOON_AGY_MODEL", None),
        };
        self.tuning
            .model
            .clone()
            .or_else(|| env_text(var))
            .or_else(|| default.map(str::to_owned))
            .unwrap_or_else(|| "default".to_owned())
    }

    /// Whether the model was named (by the tuning or the developer variable): `agy` passes `--model`
    /// only then.
    #[must_use]
    pub fn model_named(&self) -> Option<String> {
        let var = match self.tool {
            Tool::Claude => "YELLOWMOON_CLAUDE_MODEL",
            Tool::Agy => "YELLOWMOON_AGY_MODEL",
        };
        self.tuning.model.clone().or_else(|| env_text(var))
    }

    /// The effort asked for: the tuning's, the developer variable's, else `low`.
    #[must_use]
    pub fn effort(&self) -> String {
        let var = match self.tool {
            Tool::Claude => "YELLOWMOON_CLAUDE_EFFORT",
            Tool::Agy => "YELLOWMOON_AGY_EFFORT",
        };
        self.tuning
            .effort
            .clone()
            .or_else(|| env_text(var))
            .unwrap_or_else(|| CLI_EFFORT.to_owned())
    }

    /// The `AdvisorInfo` of an available program.
    fn info(&self, tool: ToolInfo) -> AdvisorInfo {
        AdvisorInfo::cli(self.tool.provider(), self.model_alias(), tool)
    }

    /// The last reading's availability (`AutoAdvisor::availability`): never waits for a process.
    #[must_use]
    pub fn availability(&self) -> Availability {
        self.prober.current().availability
    }

    /// What the advisor knows besides its availability (`AutoAdvisor::details`).
    #[must_use]
    pub fn details(&self) -> AdvisorDetails {
        let reading = self.prober.current();
        let breaker = self.breaker.open_reason().map(|last| {
            format!(
                "{} failed three times in a row ({last}). Photos it doesn't answer use \
                 Yellowmoon's own Auto.",
                self.tool.display_name()
            )
        });
        AdvisorDetails {
            path: reading.path,
            tool_version: reading.tool.as_ref().map(|t| t.version.clone()),
            api: reading.tool.as_ref().and_then(|t| t.api.clone()),
            auth: reading.tool.as_ref().and_then(|t| t.auth.clone()),
            breaker,
        }
    }

    /// Starts a probe of `level` in the background; see [`Prober::refresh`].
    pub fn refresh(&self, level: Level) {
        let (tool, source, tuning) = (self.tool, self.source.clone(), self.tuning.clone());
        let model = self.model_alias();
        self.prober.refresh(level, move || {
            probe_tool(tool, &source, &tuning, level, &model)
        });
    }

    /// Waits until no probe runs or `timeout` is up.
    pub fn settle(&self, timeout: Duration) {
        self.prober.settle(timeout);
    }

    /// Ends the requests that run now.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Registers the listener of availability changes.
    pub fn on_change(&self, listener: ChangeListener) {
        self.prober.on_change(listener);
    }

    /// A request found the program signed out: the cached state says so at once (`docs/22` section 3.3).
    fn mark_signed_out(&self) {
        let mut reading = self.prober.current();
        reading.availability = Availability::Unavailable(UnavailableReason::SignedOut);
        self.prober.set(reading);
    }

    /// A request answered: the cached state is `available` again.
    fn mark_available(&self) {
        let mut reading = self.prober.current();
        if reading.availability.is_available() {
            return;
        }
        let Some(tool) = reading.tool.clone() else {
            return;
        };
        reading.availability = Availability::Available(self.info(tool));
        self.prober.set(reading);
    }

    /// The skeleton of one request (`docs/22` sections 9.2 and 3.3): the breaker's say, the program's
    /// path, then `request` with the path and a cancel token; its outcome is recorded by the breaker,
    /// a sign-out flips the cached state, and a failure becomes an [`AdvisorError`].
    pub fn execute(
        &self,
        request: impl FnOnce(&Path, CancelToken) -> Result<AdvisorAnswer, Fail>,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        let provider = self.tool.provider();
        let ticket = self.breaker.check().map_err(|last| {
            AdvisorError::new(
                FailureKind::Unavailable,
                format!("stopped after three failures in a row: {last}"),
            )
        })?;
        let Some(path) = self.prober.current().path else {
            self.breaker.finish(ticket, Record::Abandoned);
            return Err(AdvisorError::new(
                FailureKind::Unavailable,
                format!("{} was not found", self.tool.display_name()),
            ));
        };
        match request(&path, self.cancel.token()) {
            Ok(answer) => {
                self.breaker.finish(ticket, Record::Success);
                self.mark_available();
                Ok(answer)
            }
            Err(Fail::Cancelled) => {
                self.breaker.finish(ticket, Record::Abandoned);
                // Every cancel is followed by a session or instructions change: the core drops this.
                Err(AdvisorError::new(
                    FailureKind::Other,
                    "the request was cancelled",
                ))
            }
            Err(Fail::Failed(kind, message)) => {
                let record = match kind {
                    FailureKind::RateLimited | FailureKind::Timeout | FailureKind::CliFailed => {
                        Record::Failure(failure_words(provider, kind).to_owned())
                    }
                    FailureKind::BadAnswer | FailureKind::Refused | FailureKind::NotSignedIn => {
                        Record::Neutral
                    }
                    _ => Record::Abandoned,
                };
                self.breaker.finish(ticket, record);
                if kind == FailureKind::NotSignedIn {
                    self.mark_signed_out();
                }
                Err(AdvisorError::new(kind, message))
            }
        }
    }
}

/// A developer variable's value, when it is set and not empty.
fn env_text(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Finds the program and checks it as far as `level` asks (`docs/22` section 3.2): found, version, and
/// for Claude Code `claude auth status`; for `agy` the sign-in check only at [`Level::Full`].
#[must_use]
pub fn probe_tool(
    tool: Tool,
    source: &Source,
    tuning: &Tuning,
    level: Level,
    model: &str,
) -> Reading {
    let discovery = match source {
        Source::System(locations) => locate(tool, locations, tuning),
        Source::Program(path) => check_program(tool, path, tuning),
    };
    let info = |version: &semver::Version| ToolInfo {
        version: version.to_string(),
        api: None,
        auth: None,
    };
    match discovery {
        Discovery::NotInstalled => Reading::unavailable(UnavailableReason::NotInstalled),
        Discovery::UnsafeLauncher => Reading::unavailable(UnavailableReason::UnsafeLauncher),
        Discovery::TooOld { path, version } => Reading {
            availability: Availability::Unavailable(UnavailableReason::TooOld),
            path: Some(path),
            tool: Some(info(&version)),
        },
        Discovery::ProbeFailed { path } => Reading {
            availability: Availability::Unavailable(UnavailableReason::ProbeFailed),
            path: Some(path),
            tool: None,
        },
        Discovery::Found { path, version } => match tool {
            Tool::Claude => claude_sign_in(&path, &version, tuning, model),
            Tool::Agy if level == Level::Light => Reading {
                availability: Availability::Unavailable(UnavailableReason::Checking),
                path: Some(path),
                tool: Some(info(&version)),
            },
            Tool::Agy => agy_sign_in(&path, &version, tuning, model),
        },
    }
}

/// `claude auth status`: JSON on standard output, exit 0 and `"loggedIn": true` when signed in, exit 1
/// and `false` when not (`docs/22` section 7.1). Only `loggedIn`, `authMethod` and `apiProvider` are
/// read.
fn claude_sign_in(path: &Path, version: &semver::Version, tuning: &Tuning, model: &str) -> Reading {
    let mut tool = ToolInfo {
        version: version.to_string(),
        api: None,
        auth: None,
    };
    let failed = |tool: ToolInfo| Reading {
        availability: Availability::Unavailable(UnavailableReason::ProbeFailed),
        path: Some(path.to_path_buf()),
        tool: Some(tool),
    };
    let Ok(workdir) = Workdir::new() else {
        return failed(tool);
    };
    let output = run(
        RunSpec::new(path, workdir.cwd(), EnvProfile::Claude)
            .args(["auth", "status"])
            .timeout(tuning.probe.claude_auth)
            .extra_env(tuning.extra_env.clone()),
    );
    if output.outcome != Outcome::Exited {
        return failed(tool);
    }
    let Ok(status) = serde_json::from_slice::<Value>(&output.stdout) else {
        return failed(tool);
    };
    tool.api = status
        .get("apiProvider")
        .and_then(Value::as_str)
        .map(str::to_owned);
    tool.auth = status
        .get("authMethod")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match status.get("loggedIn").and_then(Value::as_bool) {
        Some(true) if output.exit_code == Some(0) => Reading {
            availability: Availability::Available(AdvisorInfo::cli(
                Tool::Claude.provider(),
                model,
                tool.clone(),
            )),
            path: Some(path.to_path_buf()),
            tool: Some(tool),
        },
        Some(false) => Reading {
            availability: Availability::Unavailable(UnavailableReason::SignedOut),
            path: Some(path.to_path_buf()),
            tool: Some(tool),
        },
        _ => failed(tool),
    }
}

/// `agy -p /usage --output-format json` in a new empty folder (`docs/22` section 8.1): exit 0 with
/// output is signed in; `authentication required` in the output or on standard error is signed out;
/// anything else is `probe_failed`. This is the one check known to make the program contact its service.
fn agy_sign_in(path: &Path, version: &semver::Version, tuning: &Tuning, model: &str) -> Reading {
    let tool = ToolInfo {
        version: version.to_string(),
        api: None,
        auth: None,
    };
    let reading = |availability: Availability| Reading {
        availability,
        path: Some(path.to_path_buf()),
        tool: Some(tool.clone()),
    };
    let failed = || reading(Availability::Unavailable(UnavailableReason::ProbeFailed));
    let Ok(workdir) = Workdir::new() else {
        return failed();
    };
    let output = run(
        RunSpec::new(path, workdir.cwd(), EnvProfile::Agy)
            .args(["-p", "/usage", "--output-format", "json"])
            .timeout(tuning.probe.agy_usage)
            .extra_env(tuning.extra_env.clone()),
    );
    if output.outcome != Outcome::Exited {
        return failed();
    }
    let text = format!("{}\n{}", output.stdout_text(), String::from_utf8_lossy(&output.stderr))
        .to_lowercase();
    if text.contains("authentication required") {
        return reading(Availability::Unavailable(UnavailableReason::SignedOut));
    }
    if output.exit_code == Some(0) && !output.stdout.iter().all(u8::is_ascii_whitespace) {
        return reading(Availability::Available(AdvisorInfo::cli(
            Tool::Agy.provider(),
            model,
            tool.clone(),
        )));
    }
    failed()
}
