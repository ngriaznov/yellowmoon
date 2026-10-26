//! Auto providers (`docs/22-auto-providers.md`): who grades a photo's Auto. Four of them: Yellowmoon's
//! own Auto (the rule, `docs/02`), Apple Intelligence's on-device model (`docs/19`), and the two
//! command-line programs the person installed and signed in to themselves, Claude Code and
//! Antigravity (`agy`), which Yellowmoon starts as child processes (`ym-agents`).
//!
//! This module is the platform-free contract between the core, the app and `ym`: the provider enum
//! and its constants, the choice a run makes, the consent a person gives, the state of each provider
//! and the status Preferences shows ([`AutoStatus`]), the registry of advisors ([`ProviderAdvisors`])
//! and the words every front end uses ([`unavailable_hint`], [`unavailable_words`],
//! [`failure_words`], [`cloud_notice`]). The advisors themselves are
//! [`AutoAdvisor`](crate::apple_auto::AutoAdvisor)s: `ym-apple` is one, `ym-agents` has two more.
//!
//! **A word on names** (`docs/22` decision D3). In the code and on the wire the grading is still
//! called "apple" (`apple_auto`, `AppleAuto`, `AutoEstimates::apple`, `apple_steps`, the `apple`
//! stage): in those names "apple" means "the chosen provider's grading", and
//! [`AdvisorInfo::provider`](crate::apple_auto::AdvisorInfo::provider) says whose. Everything new
//! uses the neutral names of this module.
//!
//! Nothing here persists (hard rule 4) except what the preferences hold ([`AutoConsent`] and the
//! chosen provider).

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::apple_auto::{
    Activity, AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, AutoAdvisor, Availability,
    DEFAULT_INSTRUCTIONS, FailureKind, FakeAdvisor, INSTRUCTIONS_MAX_CHARS, InstructionsId,
    PROMPT_VERSION, TABLE_VERSION, UnavailableReason, effective_instructions,
};

/// The version of the prompt Claude Code is asked with (`docs/22` D19). Part of "version" under hard
/// rule 8: change it whenever the words or the request change.
pub const CLAUDE_PROMPT_VERSION: u32 = 1;

/// The version of the prompt `agy` is asked with (`docs/22` D19).
pub const AGY_PROMPT_VERSION: u32 = 1;

/// The version of the answer's JSON Schema (`docs/22` section 6.2). Changing the schema, or the
/// picture's encoding ([`IMAGE_ENCODING`]), changes this or a prompt version.
pub const SCHEMA_VERSION: u32 = 1;

/// How the picture is encoded for the two programs, as reports name it (`docs/22` section 6.1):
/// baseline JPEG, 4:4:4, quality 92, at most 768 px on the long edge.
pub const IMAGE_ENCODING: &str = "jpeg444q92@768";

/// The oldest Claude Code Yellowmoon runs: the first version with `--safe-mode` (`docs/22` section
/// 3.2).
pub const CLAUDE_MIN_VERSION: &str = "2.1.169";

/// The oldest `agy` Yellowmoon runs: the first version with the schema rules and exit code 3
/// (`docs/22` section 3.2).
pub const AGY_MIN_VERSION: &str = "1.2.14";

/// The model Claude Code is asked for unless `YELLOWMOON_CLAUDE_MODEL` says otherwise (`docs/22`
/// section 7.2, provisional until the blind A/B of section 17).
pub const CLAUDE_DEFAULT_MODEL: &str = "sonnet";

/// The effort both programs are asked for unless a developer variable says otherwise (`docs/22`
/// D18).
pub const CLI_EFFORT: &str = "low";

/// Claude Code's hard stop for one request, in seconds (`docs/22` section 1).
pub const CLAUDE_TIMEOUT_S: u32 = 90;

/// `agy`'s hard stop for one request, in seconds (`docs/22` section 1).
pub const AGY_TIMEOUT_S: u32 = 120;

/// How many failures in a row (`rate_limited`, `timeout`, `cli_failed`) open an advisor's breaker
/// (`docs/22` D5, section 9.2).
pub const BREAKER_FAILURES: u32 = 3;

/// How long an open breaker answers `unavailable` at once, in seconds, before one trial request goes
/// through (`docs/22` section 9.2).
pub const BREAKER_COOLDOWN_S: u64 = 60;

/// The shortest time between two probes of a program, in seconds (`docs/22` section 3.3).
pub const PROBE_MIN_INTERVAL_S: u64 = 5;

/// The listener an advisor calls when its availability changed (`AutoAdvisor::on_change`, `docs/22`
/// section 3.3). It runs on the advisor's probe thread, so it must not block.
pub type ChangeListener = Arc<dyn Fn() + Send + Sync>;

/// Who grades a photo's Auto (`docs/22` section 1). JSON: `"yellowmoon"`, `"apple"`, `"claude"`,
/// `"agy"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoProvider {
    /// Yellowmoon's own Auto: the rule (`docs/02`). Needs no advisor and sends nothing anywhere.
    Yellowmoon,
    /// Apple Intelligence's on-device model through `ym-apple` (`docs/19`).
    Apple,
    /// The person's own `claude` program (Claude Code).
    Claude,
    /// The person's own `agy` program (Antigravity). Marked Experimental.
    Agy,
}

impl AutoProvider {
    /// Every provider, in the order of the dropdown.
    pub const ALL: [Self; 4] = [Self::Yellowmoon, Self::Apple, Self::Claude, Self::Agy];

    /// The wire name: `yellowmoon`, `apple`, `claude` or `agy`.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::Yellowmoon => "yellowmoon",
            Self::Apple => "apple",
            Self::Claude => "claude",
            Self::Agy => "agy",
        }
    }

    /// The name in the interface and in sentences: "Yellowmoon's own Auto", "Apple Intelligence",
    /// "Claude Code", "Antigravity".
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Yellowmoon => "Yellowmoon's own Auto",
            Self::Apple => "Apple Intelligence",
            Self::Claude => "Claude Code",
            Self::Agy => "Antigravity",
        }
    }

    /// Whether the picture leaves the computer through a program (Claude Code and Antigravity): the
    /// providers that need the person's consent (`docs/22` section 10.2).
    #[must_use]
    pub fn is_cloud(self) -> bool {
        matches!(self, Self::Claude | Self::Agy)
    }

    /// The hard stop of one request, in seconds: 0 (none), 30, 90, 120.
    #[must_use]
    pub fn timeout_s(self) -> u32 {
        match self {
            Self::Yellowmoon => 0,
            Self::Apple => crate::apple_auto::REQUEST_TIMEOUT_MS / 1000,
            Self::Claude => CLAUDE_TIMEOUT_S,
            Self::Agy => AGY_TIMEOUT_S,
        }
    }

    /// How many requests the advisor takes at once by default: 1, 1, 2, 1 (the advisors report their
    /// own, `AutoAdvisor::concurrency`).
    #[must_use]
    pub fn concurrency(self) -> usize {
        match self {
            Self::Yellowmoon | Self::Apple | Self::Agy => 1,
            Self::Claude => 2,
        }
    }

    /// The version of the prompt: 0 (none), [`PROMPT_VERSION`], [`CLAUDE_PROMPT_VERSION`],
    /// [`AGY_PROMPT_VERSION`].
    #[must_use]
    pub fn prompt_version(self) -> u32 {
        match self {
            Self::Yellowmoon => 0,
            Self::Apple => PROMPT_VERSION,
            Self::Claude => CLAUDE_PROMPT_VERSION,
            Self::Agy => AGY_PROMPT_VERSION,
        }
    }

    /// The place of the provider's answers in a photo (`Photo::apple_results`) and of its advisor in
    /// [`ProviderAdvisors`]: `None` for Yellowmoon, `Some(0)` Apple, `Some(1)` Claude Code, `Some(2)`
    /// Antigravity.
    #[must_use]
    pub fn slot(self) -> Option<usize> {
        match self {
            Self::Yellowmoon => None,
            Self::Apple => Some(0),
            Self::Claude => Some(1),
            Self::Agy => Some(2),
        }
    }

    /// The provider whose [`AutoProvider::slot`] is `slot`.
    #[must_use]
    pub fn from_slot(slot: usize) -> Option<Self> {
        [Self::Apple, Self::Claude, Self::Agy].get(slot).copied()
    }
}

impl fmt::Display for AutoProvider {
    /// The wire name ([`AutoProvider::id`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for AutoProvider {
    type Err = String;

    /// Reads a provider's wire name in any case (`ym --auto-provider Claude`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|p| p.id().eq_ignore_ascii_case(s.trim()))
            .ok_or_else(|| {
                format!("\"{s}\" is not an Auto provider. Use yellowmoon, apple, claude or agy.")
            })
    }
}

/// How a run chooses its provider (`docs/22` section 10.3). Replaces the old `AppleAutoMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoChoice {
    /// The saved `auto_provider` decides (the app, and `ym` without `--auto-provider`).
    #[default]
    FromPreference,
    /// This provider, for this run (`ym --auto-provider p`; `--apple-auto on` is `Provider(Apple)`,
    /// `off` is `Provider(Yellowmoon)`). Naming a cloud provider is the consent for the run.
    Provider(AutoProvider),
}

impl AutoChoice {
    /// The provider chosen for the run: the preference for [`AutoChoice::FromPreference`], else the
    /// named one.
    #[must_use]
    pub fn chosen(self, preference: AutoProvider) -> AutoProvider {
        match self {
            Self::FromPreference => preference,
            Self::Provider(p) => p,
        }
    }

    /// Whether the run named its provider (which is the consent for a cloud provider).
    #[must_use]
    pub fn is_explicit(self) -> bool {
        matches!(self, Self::Provider(_))
    }
}

/// Whether the person agreed to send photos through each program (`docs/22` section 10.2). Saved in
/// `preferences.json` as `auto_consent`. JSON: `{ "claude": false, "agy": false }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoConsent {
    /// Claude Code.
    pub claude: bool,
    /// Antigravity.
    pub agy: bool,
}

impl AutoConsent {
    /// Whether `provider` may be used without the run naming it: true for Yellowmoon's own Auto and
    /// Apple Intelligence (nothing leaves the computer), the saved flag for the two programs.
    #[must_use]
    pub fn get(&self, provider: AutoProvider) -> bool {
        match provider {
            AutoProvider::Yellowmoon | AutoProvider::Apple => true,
            AutoProvider::Claude => self.claude,
            AutoProvider::Agy => self.agy,
        }
    }

    /// The same consent with `provider`'s flag set; a provider that needs none is left alone.
    #[must_use]
    pub fn with(mut self, provider: AutoProvider, agreed: bool) -> Self {
        match provider {
            AutoProvider::Claude => self.claude = agreed,
            AutoProvider::Agy => self.agy = agreed,
            AutoProvider::Yellowmoon | AutoProvider::Apple => {}
        }
        self
    }
}

/// What a program says about itself (`docs/22` sections 3.2 and 7.1): kept with the probe and with an
/// answer's [`AdvisorInfo`]. JSON: `{ "version", "api", "auth" }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The version the program printed (`2.1.310`).
    pub version: String,
    /// Where its requests go (`claude auth status`: `apiProvider`, `firstParty` for Anthropic); `None`
    /// for `agy` and when the program did not say.
    pub api: Option<String>,
    /// How it is signed in (`authMethod`); `None` for `agy` and when it did not say.
    pub auth: Option<String>,
}

/// Where a provider stands (`docs/22` section 3.1). JSON, snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderState {
    /// It can answer now.
    Available,
    /// No usable program was found (also: no advisor was registered).
    NotInstalled,
    /// The program runs but its version is below the minimum.
    TooOld,
    /// It runs and is not signed in.
    SignedOut,
    /// It cannot answer, for a reason in [`Availability::Unavailable`] (a failed check, a script that
    /// cannot be run safely, Apple's reasons).
    Unavailable,
    /// The first probe has not finished, or `agy`'s sign-in check has not been asked for yet.
    Checking,
}

impl ProviderState {
    /// The state of a reading: available; `not_installed`, `too_old`, `signed_out` and `checking` to
    /// themselves; every other reason to [`ProviderState::Unavailable`].
    #[must_use]
    pub fn of(availability: &Availability) -> Self {
        match availability {
            Availability::Available(_) => Self::Available,
            Availability::Unavailable(UnavailableReason::NotInstalled) => Self::NotInstalled,
            Availability::Unavailable(UnavailableReason::TooOld) => Self::TooOld,
            Availability::Unavailable(UnavailableReason::SignedOut) => Self::SignedOut,
            Availability::Unavailable(UnavailableReason::Checking) => Self::Checking,
            Availability::Unavailable(_) => Self::Unavailable,
        }
    }
}

/// What an advisor can say besides its availability (`AutoAdvisor::details`, `docs/22` sections 3.3,
/// 9.2 and 13.3). The path is the person's own and is never written to a file or a report; only
/// `ym doctor` prints it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdvisorDetails {
    /// Where the program was found.
    pub path: Option<PathBuf>,
    /// The version it printed, also when it is too old or signed out.
    pub tool_version: Option<String>,
    /// Where its requests go (`firstParty`, ...).
    pub api: Option<String>,
    /// How it is signed in.
    pub auth: Option<String>,
    /// The breaker's sentence while it is open (`docs/22` section 9.2), `None` otherwise.
    pub breaker: Option<String>,
}

/// One provider in [`AutoStatus`] (`docs/22` section 11). JSON, snake_case; the path is not in it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderStatus {
    /// Which provider.
    pub provider: AutoProvider,
    /// Where it stands.
    pub state: ProviderState,
    /// Why it is not available; `None` when it is.
    pub reason: Option<UnavailableReason>,
    /// Which model answers, with the instructions in use, when it is available.
    pub info: Option<AdvisorInfo>,
    /// The program's version, also when it is too old or signed out.
    pub tool_version: Option<String>,
    /// Where the program was found. Not on the app's wire and never serialised here: `ym doctor` reads
    /// it from the struct.
    #[serde(skip)]
    pub path: Option<PathBuf>,
    /// Whether the person agreed to send photos through it ([`AutoConsent::get`]).
    pub consent: bool,
    /// Whether the interface marks it Experimental (Antigravity).
    pub experimental: bool,
    /// Whether the dropdown lists it (`docs/22` section 3.1).
    pub listed: bool,
    /// Whether the dropdown lets the person choose it now: it is available.
    pub selectable: bool,
}

/// What the app's Preferences and `ym doctor` show about Auto (`docs/22` section 11), built by
/// `Session::auto_status` from what the advisors last said. Replaces the old `AppleAutoStatus`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoStatus {
    /// Every provider, in [`AutoProvider::ALL`] order.
    pub providers: Vec<ProviderStatus>,
    /// The saved choice (`auto_provider`).
    pub preference: AutoProvider,
    /// How this run chooses ([`AutoChoice`]).
    pub choice: AutoChoice,
    /// The provider chosen for this run: the preference, or the one the run named.
    pub chosen: AutoProvider,
    /// The provider that acts now: the chosen one when it can answer (and its consent is saved or the
    /// run named it), else Yellowmoon's own Auto.
    pub effective: AutoProvider,
    /// Whether a provider other than Yellowmoon's own Auto acts (`effective` is not `yellowmoon`).
    pub active: bool,
    /// Whether the dropdown and the instructions show: Apple Intelligence is available, or Claude Code
    /// or Antigravity is listed (`docs/22` section 3.1).
    pub visible: bool,
    /// The saved consent.
    pub consent: AutoConsent,
    /// [`TABLE_VERSION`].
    pub table: u32,
    /// Which instructions the models read: default or edited, and their short digest.
    #[serde(flatten)]
    pub instructions: InstructionsId,
    /// The person's own instructions, `""` when they use the default.
    pub own_instructions: String,
    /// [`DEFAULT_INSTRUCTIONS`], for Preferences' text field and its Reset to default.
    pub default_instructions: String,
    /// [`INSTRUCTIONS_MAX_CHARS`].
    pub instructions_max_chars: usize,
}

/// What [`AutoStatus::build`] needs: the session's switches and what each advisor last said.
#[derive(Debug, Clone)]
pub struct StatusInput<'a> {
    /// Whether the session was given any advisor at all (none: the golden tool, most tests).
    pub has_advisors: bool,
    /// For each slot (Apple, Claude Code, Antigravity), the advisor's last availability and its
    /// details; `None` when no advisor is registered there or it has not been read.
    pub readings: [Option<(Availability, AdvisorDetails)>; 3],
    /// The saved choice.
    pub preference: AutoProvider,
    /// How the run chooses.
    pub choice: AutoChoice,
    /// The saved consent.
    pub consent: AutoConsent,
    /// The person's own instructions, `""` for the default.
    pub own_instructions: &'a str,
}

impl AutoStatus {
    /// Builds the status from the cached readings; no process runs and no advisor is asked.
    #[must_use]
    pub fn build(input: &StatusInput<'_>) -> Self {
        let text = effective_instructions(input.own_instructions);
        let providers: Vec<ProviderStatus> = AutoProvider::ALL
            .into_iter()
            .map(|p| provider_status(p, input, text))
            .collect();
        let chosen = input.choice.chosen(input.preference);
        let availability = chosen
            .slot()
            .and_then(|s| input.readings[s].as_ref().map(|(a, _)| a));
        let activity = Activity::decide(
            input.has_advisors,
            chosen,
            input.choice.is_explicit(),
            &input.consent,
            availability,
        );
        let effective = if activity.is_active() {
            chosen
        } else {
            AutoProvider::Yellowmoon
        };
        let listed = |p: AutoProvider| providers.iter().any(|s| s.provider == p && s.listed);
        let visible = listed(AutoProvider::Apple)
            || listed(AutoProvider::Claude)
            || listed(AutoProvider::Agy);
        Self {
            providers,
            preference: input.preference,
            choice: input.choice,
            chosen,
            effective,
            active: effective != AutoProvider::Yellowmoon,
            visible,
            consent: input.consent,
            table: TABLE_VERSION,
            instructions: InstructionsId::of(text),
            own_instructions: input.own_instructions.to_owned(),
            default_instructions: DEFAULT_INSTRUCTIONS.to_owned(),
            instructions_max_chars: INSTRUCTIONS_MAX_CHARS,
        }
    }

    /// The status of one provider.
    #[must_use]
    pub fn provider(&self, provider: AutoProvider) -> Option<&ProviderStatus> {
        self.providers.iter().find(|s| s.provider == provider)
    }
}

/// The [`ProviderStatus`] of one provider (`docs/22` section 3.1).
fn provider_status(
    provider: AutoProvider,
    input: &StatusInput<'_>,
    instructions: &str,
) -> ProviderStatus {
    let consent = input.consent.get(provider);
    let experimental = provider == AutoProvider::Agy;
    let Some(slot) = provider.slot() else {
        // Yellowmoon's own Auto: always there, needs nothing.
        return ProviderStatus {
            provider,
            state: ProviderState::Available,
            reason: None,
            info: None,
            tool_version: None,
            path: None,
            consent,
            experimental,
            listed: true,
            selectable: true,
        };
    };
    let Some((availability, details)) = &input.readings[slot] else {
        // No advisor registered, or none read yet: as if it were not installed (`Activity::decide`).
        return ProviderStatus {
            provider,
            state: ProviderState::NotInstalled,
            reason: Some(UnavailableReason::NotInstalled),
            info: None,
            tool_version: None,
            path: None,
            consent,
            experimental,
            listed: false,
            selectable: false,
        };
    };
    let state = ProviderState::of(availability);
    let (reason, info) = match availability {
        Availability::Available(info) => (None, Some(info.clone().with_instructions(instructions))),
        Availability::Unavailable(reason) => (Some(*reason), None),
    };
    let listed = match provider {
        AutoProvider::Claude | AutoProvider::Agy => matches!(
            state,
            ProviderState::Available
                | ProviderState::TooOld
                | ProviderState::SignedOut
                | ProviderState::Unavailable
        ),
        _ => state == ProviderState::Available,
    };
    let tool_version = details.tool_version.clone().or_else(|| {
        info.as_ref()
            .and_then(|i| i.tool.as_ref().map(|t| t.version.clone()))
    });
    ProviderStatus {
        provider,
        state,
        reason,
        info,
        tool_version,
        path: details.path.clone(),
        consent,
        experimental,
        listed,
        selectable: state == ProviderState::Available,
    }
}

/// The registry of advisors a session holds: one optional slot each for Apple Intelligence, Claude
/// Code and Antigravity (`docs/22` decision D4). Yellowmoon's own Auto has none. Cheap to clone.
#[derive(Clone, Default)]
pub struct ProviderAdvisors {
    slots: [Option<Arc<dyn AutoAdvisor>>; 3],
}

impl fmt::Debug for ProviderAdvisors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let present: Vec<&str> = AutoProvider::ALL
            .into_iter()
            .filter(|p| self.get(*p).is_some())
            .map(AutoProvider::id)
            .collect();
        f.debug_struct("ProviderAdvisors")
            .field("providers", &present)
            .finish()
    }
}

impl ProviderAdvisors {
    /// The same registry with `advisor` as `provider`'s. Yellowmoon's own Auto takes none: that call
    /// is ignored (and logged).
    #[must_use]
    pub fn with(mut self, provider: AutoProvider, advisor: Arc<dyn AutoAdvisor>) -> Self {
        match provider.slot() {
            Some(slot) => self.slots[slot] = Some(advisor),
            None => tracing::warn!("an advisor was given for Yellowmoon's own Auto; it is ignored"),
        }
        self
    }

    /// The advisor of `provider`, if one is registered.
    #[must_use]
    pub fn get(&self, provider: AutoProvider) -> Option<&Arc<dyn AutoAdvisor>> {
        provider.slot().and_then(|s| self.slots[s].as_ref())
    }

    /// Whether no advisor is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// The largest `concurrency()` of the advisors: how many workers a session starts. 1 when empty.
    #[must_use]
    pub fn max_concurrency(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|a| a.concurrency().max(1))
            .max()
            .unwrap_or(1)
    }

    /// Asks the advisor of `only`, or every advisor, to read its availability again. Never blocks
    /// (`AutoAdvisor::refresh`).
    pub fn refresh(&self, only: Option<AutoProvider>) {
        for p in AutoProvider::ALL {
            if only.is_some_and(|o| o != p) {
                continue;
            }
            if let Some(advisor) = self.get(p) {
                advisor.refresh();
            }
        }
    }

    /// Waits until no advisor probes, or `timeout` has passed in all (`ym` only).
    pub fn settle(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        for advisor in self.slots.iter().flatten() {
            advisor.settle(deadline.saturating_duration_since(Instant::now()));
        }
    }

    /// Cancels the running requests of every advisor (`AutoAdvisor::cancel`).
    pub fn cancel_all(&self) {
        for advisor in self.slots.iter().flatten() {
            advisor.cancel();
        }
    }

    /// Registers `listener` on every advisor (`AutoAdvisor::on_change`).
    pub fn on_change(&self, listener: ChangeListener) {
        for advisor in self.slots.iter().flatten() {
            advisor.on_change(Arc::clone(&listener));
        }
    }
}

/// `Terminal`, or "a terminal window" on Windows, for the hints (`docs/22` section 3.4).
fn terminal(os: &str) -> &'static str {
    if os == "windows" {
        "a terminal window"
    } else {
        "Terminal"
    }
}

/// The command that starts a program, for the hints.
fn command(provider: AutoProvider) -> &'static str {
    match provider {
        AutoProvider::Agy => "agy",
        _ => "claude",
    }
}

/// The hint Preferences shows under a provider that is listed but not usable, and `ym` prints as its
/// warning (`docs/22` section 3.4): a sentence with its action. Empty for the reasons that show no
/// hint (`not_installed`, `checking`, `not_consented`) and for Yellowmoon's own Auto. `tool_version`
/// is the version the program printed; `os` is `std::env::consts::OS` (on `"windows"` the word
/// "Terminal" is "a terminal window"). For Apple Intelligence it is the reason in words
/// ([`UnavailableReason::words`]).
#[must_use]
pub fn unavailable_hint(
    provider: AutoProvider,
    reason: UnavailableReason,
    tool_version: Option<&str>,
    os: &str,
) -> String {
    match provider {
        AutoProvider::Yellowmoon => String::new(),
        AutoProvider::Apple => reason.words().to_owned(),
        AutoProvider::Claude | AutoProvider::Agy => {
            let (name, cmd, t) = (provider.name(), command(provider), terminal(os));
            match reason {
                UnavailableReason::SignedOut => {
                    format!("{name} is installed but not signed in. Open {t}, run {cmd} and sign in.")
                }
                UnavailableReason::TooOld => {
                    let version = tool_version.map_or(String::new(), |v| format!(" {v}"));
                    format!("{name}{version} is too old. Update it: open {t} and run {cmd} update.")
                }
                UnavailableReason::ProbeFailed => format!(
                    "{name} didn't answer its check. Open {t}, run {cmd} to see if it starts, then open Preferences again."
                ),
                UnavailableReason::UnsafeLauncher => format!(
                    "Yellowmoon found {name} only as a script it can't run safely. Install the native version of {name}."
                ),
                _ => String::new(),
            }
        }
    }
}

/// The clause `ym doctor` and `ym info` put in "Not available (<clause>)" (`docs/22` section 3.4):
/// "Claude Code isn't signed in", "Antigravity 1.0.9 is too old", ... For Apple Intelligence it is
/// [`UnavailableReason::words`]. Empty for Yellowmoon's own Auto.
#[must_use]
pub fn unavailable_words(
    provider: AutoProvider,
    reason: UnavailableReason,
    tool_version: Option<&str>,
) -> String {
    match provider {
        AutoProvider::Yellowmoon => String::new(),
        AutoProvider::Apple => reason.words().to_owned(),
        AutoProvider::Claude | AutoProvider::Agy => {
            let name = provider.name();
            match reason {
                UnavailableReason::SignedOut => format!("{name} isn't signed in"),
                UnavailableReason::TooOld => {
                    let version = tool_version.map_or(String::new(), |v| format!(" {v}"));
                    format!("{name}{version} is too old")
                }
                UnavailableReason::ProbeFailed => format!("{name} didn't answer its check"),
                UnavailableReason::UnsafeLauncher => {
                    format!("{name} was found only as a script that Yellowmoon can't run safely")
                }
                UnavailableReason::NotInstalled => format!("{name} isn't installed"),
                UnavailableReason::Checking => "Yellowmoon is still checking".to_owned(),
                UnavailableReason::NotConsented => {
                    format!("you haven't agreed to send photos to {name} yet")
                }
                other => other.words().to_owned(),
            }
        }
    }
}

/// A failure in words, as `ym info` prints "Failed (<words>)" and the logs say it (`docs/22` section
/// 9.1). Apple Intelligence's are [`FailureKind::words`]; the two programs have their own for the
/// kinds they use, and the kinds they do not use read as Apple's.
#[must_use]
pub fn failure_words(provider: AutoProvider, kind: FailureKind) -> &'static str {
    match (provider, kind) {
        (AutoProvider::Claude, FailureKind::Unavailable) => "Claude Code stopped being available",
        (AutoProvider::Agy, FailureKind::Unavailable) => "Antigravity stopped being available",
        (AutoProvider::Claude, FailureKind::RateLimited) => "Claude Code reported a rate limit",
        (AutoProvider::Agy, FailureKind::RateLimited) => "Antigravity reported a rate limit",
        (AutoProvider::Claude, FailureKind::Timeout) => "no answer within 90 seconds",
        (AutoProvider::Agy, FailureKind::Timeout) => "no answer within 120 seconds",
        (AutoProvider::Claude, FailureKind::Refused) => "Claude refused",
        (AutoProvider::Agy, FailureKind::Refused) => "Antigravity refused",
        (AutoProvider::Claude, FailureKind::NotSignedIn) => "Claude Code isn't signed in",
        (AutoProvider::Agy, FailureKind::NotSignedIn) => "Antigravity isn't signed in",
        (AutoProvider::Claude, FailureKind::CliFailed) => "Claude Code reported an error",
        (AutoProvider::Agy, FailureKind::CliFailed) => "Antigravity reported an error",
        (_, kind) => kind.words(),
    }
}

/// What `ym` says on stderr, once per run, when a cloud provider acts (`docs/22` section 13.2):
/// "Claude Code will send a 768 px picture of each photo and the instructions to Anthropic." For
/// Claude Code the destination is Anthropic only when `api` is `firstParty`; any other (or unknown)
/// way of using it reads "the service your Claude Code is set up to use". `None` for the providers
/// that send nothing.
#[must_use]
pub fn cloud_notice(provider: AutoProvider, api: Option<&str>) -> Option<String> {
    match provider {
        AutoProvider::Claude => {
            let to = if api == Some("firstParty") {
                "Anthropic"
            } else {
                "the service your Claude Code is set up to use"
            };
            Some(format!(
                "Claude Code will send a 768 px picture of each photo and the instructions to {to}."
            ))
        }
        AutoProvider::Agy => Some(
            "Antigravity (agy) will send a 768 px picture of each photo and the instructions to Google."
                .to_owned(),
        ),
        AutoProvider::Yellowmoon | AutoProvider::Apple => None,
    }
}

// ---------------------------------------------------------------------------------------------
// FakeProvider
// ---------------------------------------------------------------------------------------------

/// A scripted advisor for the tests of every crate (`docs/22` section 11): it answers like
/// [`FakeAdvisor`] (from the picture's brightness and spread), reports a given availability, and can be
/// told a concurrency, a delay and a failure. It starts no process. `ym-agents` hands it out for
/// `YELLOWMOON_TEST_AUTO_PROVIDERS=claude=available` (`docs/22` section 15.3).
pub struct FakeProvider {
    provider: AutoProvider,
    availability: Mutex<Availability>,
    concurrency: usize,
    delay: Duration,
    failure: Mutex<Option<(FailureKind, String)>>,
    served_model: Mutex<Option<String>>,
    details: Mutex<AdvisorDetails>,
    listeners: Mutex<Vec<ChangeListener>>,
    asks: AtomicUsize,
    running: AtomicUsize,
    peak: AtomicUsize,
    /// Bumped by `cancel`: an `ask` that waits in its delay when it changes ends at once.
    cancel_generation: Mutex<u64>,
    cancel_wake: Condvar,
}

impl fmt::Debug for FakeProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeProvider")
            .field("provider", &self.provider)
            .field("concurrency", &self.concurrency)
            .field("delay", &self.delay)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FakeProvider {
    /// The [`AdvisorInfo`] a fake of `provider` reports when it is available: Apple's is
    /// [`FakeAdvisor::info`]; Claude Code's and Antigravity's name the model `Test <name>` and the
    /// tool version `0.0.0-test`.
    #[must_use]
    pub fn info_for(provider: AutoProvider) -> AdvisorInfo {
        match provider {
            AutoProvider::Claude | AutoProvider::Agy => AdvisorInfo::cli(
                provider,
                format!("Test {}", provider.name()),
                ToolInfo {
                    version: "0.0.0-test".to_owned(),
                    api: None,
                    auth: None,
                },
            ),
            AutoProvider::Apple | AutoProvider::Yellowmoon => FakeAdvisor::info(),
        }
    }

    /// A fake of `provider` that reports `availability`, takes one request at a time and answers at
    /// once.
    #[must_use]
    pub fn new(provider: AutoProvider, availability: Availability) -> Self {
        Self {
            provider,
            availability: Mutex::new(availability),
            concurrency: provider.concurrency(),
            delay: Duration::ZERO,
            failure: Mutex::new(None),
            served_model: Mutex::new(None),
            details: Mutex::new(AdvisorDetails::default()),
            listeners: Mutex::new(Vec::new()),
            asks: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            cancel_generation: Mutex::new(0),
            cancel_wake: Condvar::new(),
        }
    }

    /// A fake of `provider` that is available ([`FakeProvider::info_for`]).
    #[must_use]
    pub fn available(provider: AutoProvider) -> Self {
        Self::new(provider, Availability::Available(Self::info_for(provider)))
    }

    /// The number of requests it takes at once.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// How long each request takes (the wait is cut short by `cancel`).
    #[must_use]
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Makes every request fail with `kind` and `message`.
    #[must_use]
    pub fn with_failure(self, kind: FailureKind, message: &str) -> Self {
        self.fail_with(Some((kind, message.to_owned())));
        self
    }

    /// The model id every answer says it came from (`AdvisorAnswer::served_model`).
    #[must_use]
    pub fn with_served_model(self, model: &str) -> Self {
        *lock(&self.served_model) = Some(model.to_owned());
        self
    }

    /// What `details()` reports.
    #[must_use]
    pub fn with_details(self, details: AdvisorDetails) -> Self {
        *lock(&self.details) = details;
        self
    }

    /// Sets (or clears, with `None`) the failure every request gets.
    pub fn fail_with(&self, failure: Option<(FailureKind, String)>) {
        *lock(&self.failure) = failure;
    }

    /// Changes what it reports as its availability and, when that differs from before, calls the
    /// listeners (as a probe that found a change does).
    pub fn set_availability(&self, availability: Availability) {
        {
            let mut current = lock(&self.availability);
            if *current == availability {
                return;
            }
            *current = availability;
        }
        let listeners: Vec<ChangeListener> = lock(&self.listeners).clone();
        for l in listeners {
            l();
        }
    }

    /// The number of calls of `ask` so far.
    #[must_use]
    pub fn asks(&self) -> usize {
        self.asks.load(Ordering::SeqCst)
    }

    /// The most `ask` calls that ran at the same moment.
    #[must_use]
    pub fn peak_concurrency(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// Waits `self.delay`, or until `cancel` is called.
    fn wait_delay(&self) {
        if self.delay.is_zero() {
            return;
        }
        let deadline = Instant::now() + self.delay;
        let mut generation = lock(&self.cancel_generation);
        let start = *generation;
        while *generation == start {
            let now = Instant::now();
            if now >= deadline {
                return;
            }
            generation = self
                .cancel_wake
                .wait_timeout(generation, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

impl AutoAdvisor for FakeProvider {
    fn availability(&self) -> Availability {
        lock(&self.availability).clone()
    }

    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        self.asks.fetch_add(1, Ordering::SeqCst);
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.wait_delay();
        let result = match lock(&self.failure).clone() {
            Some((kind, message)) => Err(AdvisorError::new(kind, message)),
            None => FakeAdvisor.ask(picture, instructions).map(|mut answer| {
                answer.served_model = lock(&self.served_model).clone();
                answer
            }),
        };
        self.running.fetch_sub(1, Ordering::SeqCst);
        result
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn cancel(&self) {
        *lock(&self.cancel_generation) += 1;
        self.cancel_wake.notify_all();
    }

    fn on_change(&self, listener: ChangeListener) {
        lock(&self.listeners).push(listener);
    }

    fn details(&self) -> AdvisorDetails {
        lock(&self.details).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apple_auto::{AdvisorPicture, UnavailableReason as R};

    fn picture() -> AdvisorPicture {
        AdvisorPicture {
            width: 4,
            height: 3,
            rgb: vec![128; 36],
        }
    }

    fn available(p: AutoProvider) -> Availability {
        Availability::Available(FakeProvider::info_for(p))
    }

    #[test]
    fn providers_have_their_constants() {
        let names: Vec<_> = AutoProvider::ALL.iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            [
                "Yellowmoon's own Auto",
                "Apple Intelligence",
                "Claude Code",
                "Antigravity"
            ]
        );
        let ids: Vec<_> = AutoProvider::ALL.iter().map(|p| p.id()).collect();
        assert_eq!(ids, ["yellowmoon", "apple", "claude", "agy"]);
        assert_eq!(
            AutoProvider::ALL.map(AutoProvider::is_cloud),
            [false, false, true, true]
        );
        assert_eq!(
            AutoProvider::ALL.map(AutoProvider::timeout_s),
            [0, 30, 90, 120]
        );
        assert_eq!(
            AutoProvider::ALL.map(AutoProvider::concurrency),
            [1, 1, 2, 1]
        );
        assert_eq!(
            AutoProvider::ALL.map(AutoProvider::prompt_version),
            [0, 3, 1, 1]
        );
        assert_eq!(
            AutoProvider::ALL.map(AutoProvider::slot),
            [None, Some(0), Some(1), Some(2)]
        );
        for p in AutoProvider::ALL {
            assert_eq!(p.slot().and_then(AutoProvider::from_slot), p.slot().map(|_| p));
            assert_eq!(
                serde_json::to_value(p).unwrap(),
                serde_json::json!(p.id()),
                "{p:?}"
            );
            assert_eq!(p.id().parse::<AutoProvider>(), Ok(p));
        }
        assert_eq!("CLAUDE".parse::<AutoProvider>(), Ok(AutoProvider::Claude));
        assert!("gemini".parse::<AutoProvider>().is_err());
    }

    #[test]
    fn a_choice_follows_the_preference_unless_it_names_a_provider() {
        let pref = AutoProvider::Claude;
        assert_eq!(AutoChoice::FromPreference.chosen(pref), AutoProvider::Claude);
        assert_eq!(
            AutoChoice::Provider(AutoProvider::Apple).chosen(pref),
            AutoProvider::Apple
        );
        assert!(!AutoChoice::FromPreference.is_explicit());
        assert!(AutoChoice::Provider(AutoProvider::Agy).is_explicit());
        assert_eq!(
            serde_json::to_value(AutoChoice::FromPreference).unwrap(),
            serde_json::json!("from_preference")
        );
    }

    #[test]
    fn consent_is_needed_only_for_the_two_programs() {
        let none = AutoConsent::default();
        assert!(none.get(AutoProvider::Yellowmoon) && none.get(AutoProvider::Apple));
        assert!(!none.get(AutoProvider::Claude) && !none.get(AutoProvider::Agy));
        let c = none.with(AutoProvider::Claude, true);
        assert!(c.get(AutoProvider::Claude) && !c.get(AutoProvider::Agy));
        assert_eq!(c.with(AutoProvider::Apple, false), c);
        assert_eq!(
            serde_json::to_value(c).unwrap(),
            serde_json::json!({ "claude": true, "agy": false })
        );
        let read: AutoConsent = serde_json::from_str(r#"{ "agy": true }"#).unwrap();
        assert_eq!(read, AutoConsent { claude: false, agy: true });
    }

    #[test]
    fn a_reading_maps_to_a_state() {
        let state = |r: R| ProviderState::of(&Availability::Unavailable(r));
        assert_eq!(ProviderState::of(&available(AutoProvider::Claude)), ProviderState::Available);
        assert_eq!(state(R::NotInstalled), ProviderState::NotInstalled);
        assert_eq!(state(R::TooOld), ProviderState::TooOld);
        assert_eq!(state(R::SignedOut), ProviderState::SignedOut);
        assert_eq!(state(R::Checking), ProviderState::Checking);
        for r in [
            R::ProbeFailed,
            R::UnsafeLauncher,
            R::NotAMac,
            R::AppleIntelligenceOff,
            R::NoImageInput,
            R::NotConsented,
        ] {
            assert_eq!(state(r), ProviderState::Unavailable, "{r:?}");
        }
    }

    fn input<'a>(
        readings: [Option<(Availability, AdvisorDetails)>; 3],
        preference: AutoProvider,
        consent: AutoConsent,
    ) -> StatusInput<'a> {
        StatusInput {
            has_advisors: readings.iter().any(Option::is_some),
            readings,
            preference,
            choice: AutoChoice::FromPreference,
            consent,
            own_instructions: "",
        }
    }

    fn reading(a: Availability) -> Option<(Availability, AdvisorDetails)> {
        Some((a, AdvisorDetails::default()))
    }

    #[test]
    fn the_dropdown_shows_when_any_provider_is_there_to_choose() {
        // Nothing registered: hidden, and Yellowmoon's own Auto acts.
        let s = AutoStatus::build(&input([None, None, None], AutoProvider::Apple, AutoConsent::default()));
        assert!(!s.visible && !s.active);
        assert_eq!(s.effective, AutoProvider::Yellowmoon);
        assert_eq!(s.providers.len(), 4);
        let y = s.provider(AutoProvider::Yellowmoon).unwrap();
        assert!(y.listed && y.selectable && y.consent);
        let c = s.provider(AutoProvider::Claude).unwrap();
        assert_eq!(
            (c.state, c.reason, c.listed, c.selectable),
            (ProviderState::NotInstalled, Some(R::NotInstalled), false, false)
        );
        // Apple alone, available: visible.
        let s = AutoStatus::build(&input(
            [reading(available(AutoProvider::Apple)), None, None],
            AutoProvider::Apple,
            AutoConsent::default(),
        ));
        assert!(s.visible && s.active);
        assert_eq!(s.effective, AutoProvider::Apple);
        let a = s.provider(AutoProvider::Apple).unwrap();
        assert!(a.listed && a.selectable && a.info.is_some());
        // Apple unavailable: its row is not listed, and nothing else is: hidden.
        let s = AutoStatus::build(&input(
            [
                reading(Availability::Unavailable(R::NotAMac)),
                reading(Availability::Unavailable(R::NotInstalled)),
                None,
            ],
            AutoProvider::Apple,
            AutoConsent::default(),
        ));
        assert!(!s.visible && !s.active);
        assert!(!s.provider(AutoProvider::Apple).unwrap().listed);
        // Only a signed-out program: visible, listed, not selectable.
        let s = AutoStatus::build(&input(
            [
                reading(Availability::Unavailable(R::NotAMac)),
                Some((
                    Availability::Unavailable(R::SignedOut),
                    AdvisorDetails {
                        tool_version: Some("2.1.310".to_owned()),
                        path: Some(PathBuf::from("/x/claude")),
                        ..AdvisorDetails::default()
                    },
                )),
                None,
            ],
            AutoProvider::Apple,
            AutoConsent::default(),
        ));
        assert!(s.visible && !s.active);
        let c = s.provider(AutoProvider::Claude).unwrap();
        assert_eq!(
            (c.state, c.reason, c.listed, c.selectable),
            (ProviderState::SignedOut, Some(R::SignedOut), true, false)
        );
        assert_eq!(c.tool_version.as_deref(), Some("2.1.310"));
        assert_eq!(c.path.as_deref(), Some(std::path::Path::new("/x/claude")));
        // The path never reaches JSON.
        assert!(serde_json::to_string(c).unwrap().find("/x/claude").is_none());
        // A checking program is not listed yet.
        let s = AutoStatus::build(&input(
            [None, None, reading(Availability::Unavailable(R::Checking))],
            AutoProvider::Apple,
            AutoConsent::default(),
        ));
        assert!(!s.visible);
        assert!(s.provider(AutoProvider::Agy).unwrap().experimental);
    }

    #[test]
    fn a_cloud_choice_acts_only_with_its_consent() {
        let readings = || {
            [
                None,
                reading(available(AutoProvider::Claude)),
                None,
            ]
        };
        let s = AutoStatus::build(&input(readings(), AutoProvider::Claude, AutoConsent::default()));
        assert_eq!((s.chosen, s.effective, s.active), (AutoProvider::Claude, AutoProvider::Yellowmoon, false));
        assert!(s.visible);
        let consent = AutoConsent::default().with(AutoProvider::Claude, true);
        let s = AutoStatus::build(&input(readings(), AutoProvider::Claude, consent));
        assert_eq!((s.effective, s.active), (AutoProvider::Claude, true));
        // A run that names the provider is the consent.
        let mut i = input(readings(), AutoProvider::Yellowmoon, AutoConsent::default());
        i.choice = AutoChoice::Provider(AutoProvider::Claude);
        let s = AutoStatus::build(&i);
        assert_eq!((s.preference, s.chosen, s.effective), (AutoProvider::Yellowmoon, AutoProvider::Claude, AutoProvider::Claude));
        // The instructions are the models' default unless the person wrote their own.
        assert_eq!(s.instructions, InstructionsId::default());
        assert_eq!(
            s.provider(AutoProvider::Claude).unwrap().info.as_ref().unwrap().instructions,
            InstructionsId::default()
        );
        i.own_instructions = "Moody.";
        let s = AutoStatus::build(&i);
        assert_eq!(s.instructions, InstructionsId::of("Moody."));
        assert_eq!(s.own_instructions, "Moody.");
        assert_eq!(
            s.provider(AutoProvider::Claude).unwrap().info.as_ref().unwrap().instructions,
            InstructionsId::of("Moody.")
        );
    }

    #[test]
    fn the_hints_and_clauses_are_the_words_of_the_spec() {
        use AutoProvider::{Agy, Claude};
        let hint = |p, r, v: Option<&str>, os| unavailable_hint(p, r, v, os);
        assert_eq!(
            hint(Claude, R::SignedOut, None, "macos"),
            "Claude Code is installed but not signed in. Open Terminal, run claude and sign in."
        );
        assert_eq!(
            hint(Claude, R::SignedOut, None, "windows"),
            "Claude Code is installed but not signed in. Open a terminal window, run claude and sign in."
        );
        assert_eq!(
            hint(Claude, R::TooOld, Some("2.1.100"), "linux"),
            "Claude Code 2.1.100 is too old. Update it: open Terminal and run claude update."
        );
        assert_eq!(
            hint(Agy, R::TooOld, Some("1.0.9"), "windows"),
            "Antigravity 1.0.9 is too old. Update it: open a terminal window and run agy update."
        );
        assert_eq!(
            hint(Claude, R::ProbeFailed, None, "macos"),
            "Claude Code didn't answer its check. Open Terminal, run claude to see if it starts, then open Preferences again."
        );
        assert_eq!(
            hint(Agy, R::SignedOut, None, "linux"),
            "Antigravity is installed but not signed in. Open Terminal, run agy and sign in."
        );
        assert_eq!(
            hint(Agy, R::ProbeFailed, None, "windows"),
            "Antigravity didn't answer its check. Open a terminal window, run agy to see if it starts, then open Preferences again."
        );
        assert_eq!(
            hint(Claude, R::UnsafeLauncher, None, "windows"),
            "Yellowmoon found Claude Code only as a script it can't run safely. Install the native version of Claude Code."
        );
        for r in [R::NotInstalled, R::Checking, R::NotConsented] {
            assert_eq!(hint(Claude, r, None, "macos"), "", "{r:?}");
        }
        assert_eq!(
            unavailable_words(Claude, R::SignedOut, None),
            "Claude Code isn't signed in"
        );
        assert_eq!(
            unavailable_words(Claude, R::TooOld, Some("2.1.100")),
            "Claude Code 2.1.100 is too old"
        );
        assert_eq!(unavailable_words(Claude, R::TooOld, None), "Claude Code is too old");
        assert_eq!(
            unavailable_words(Agy, R::ProbeFailed, None),
            "Antigravity didn't answer its check"
        );
        assert_eq!(
            unavailable_words(Claude, R::UnsafeLauncher, None),
            "Claude Code was found only as a script that Yellowmoon can't run safely"
        );
        assert_eq!(unavailable_words(Agy, R::NotInstalled, None), "Antigravity isn't installed");
        assert_eq!(
            unavailable_words(Claude, R::Checking, None),
            "Yellowmoon is still checking"
        );
        assert_eq!(
            unavailable_words(Agy, R::NotConsented, None),
            "you haven't agreed to send photos to Antigravity yet"
        );
        assert_eq!(
            unavailable_words(AutoProvider::Apple, R::AppleIntelligenceOff, None),
            "Apple Intelligence is off in System Settings"
        );
    }

    #[test]
    fn failures_have_words_per_provider() {
        use AutoProvider::{Agy, Apple, Claude};
        assert_eq!(failure_words(Apple, FailureKind::Timeout), "no answer within 30 seconds");
        assert_eq!(failure_words(Claude, FailureKind::Timeout), "no answer within 90 seconds");
        assert_eq!(failure_words(Agy, FailureKind::Timeout), "no answer within 120 seconds");
        assert_eq!(failure_words(Claude, FailureKind::RateLimited), "Claude Code reported a rate limit");
        assert_eq!(failure_words(Apple, FailureKind::RateLimited), "macOS asked Yellowmoon to wait");
        assert_eq!(failure_words(Claude, FailureKind::Refused), "Claude refused");
        assert_eq!(failure_words(Agy, FailureKind::Refused), "Antigravity refused");
        assert_eq!(failure_words(Agy, FailureKind::NotSignedIn), "Antigravity isn't signed in");
        assert_eq!(failure_words(Claude, FailureKind::CliFailed), "Claude Code reported an error");
        assert_eq!(failure_words(Claude, FailureKind::Unavailable), "Claude Code stopped being available");
        assert_eq!(
            failure_words(Apple, FailureKind::Unavailable),
            "Apple Intelligence stopped being available"
        );
        for p in [Apple, Claude, Agy] {
            assert_eq!(failure_words(p, FailureKind::BadAnswer), "the answer could not be read");
            assert_eq!(failure_words(p, FailureKind::Other), "an error");
        }
    }

    #[test]
    fn the_cloud_notice_says_where_the_picture_goes() {
        assert_eq!(
            cloud_notice(AutoProvider::Claude, Some("firstParty")).as_deref(),
            Some("Claude Code will send a 768 px picture of each photo and the instructions to Anthropic.")
        );
        assert_eq!(
            cloud_notice(AutoProvider::Claude, Some("bedrock")).as_deref(),
            Some("Claude Code will send a 768 px picture of each photo and the instructions to the service your Claude Code is set up to use.")
        );
        assert_eq!(
            cloud_notice(AutoProvider::Claude, None).as_deref(),
            cloud_notice(AutoProvider::Claude, Some("vertex")).as_deref()
        );
        assert_eq!(
            cloud_notice(AutoProvider::Agy, None).as_deref(),
            Some("Antigravity (agy) will send a 768 px picture of each photo and the instructions to Google.")
        );
        assert_eq!(cloud_notice(AutoProvider::Apple, None), None);
        assert_eq!(cloud_notice(AutoProvider::Yellowmoon, None), None);
    }

    #[test]
    fn the_registry_holds_three_slots() {
        let none = ProviderAdvisors::default();
        assert!(none.is_empty());
        assert_eq!(none.max_concurrency(), 1);
        let claude = Arc::new(FakeProvider::available(AutoProvider::Claude));
        let apple = Arc::new(FakeProvider::available(AutoProvider::Apple));
        let advisors = ProviderAdvisors::default()
            .with(AutoProvider::Claude, claude.clone())
            .with(AutoProvider::Apple, apple)
            // Yellowmoon's own Auto takes no advisor.
            .with(AutoProvider::Yellowmoon, claude);
        assert!(!advisors.is_empty());
        assert!(advisors.get(AutoProvider::Yellowmoon).is_none());
        assert!(advisors.get(AutoProvider::Agy).is_none());
        assert!(advisors.get(AutoProvider::Apple).is_some());
        assert_eq!(advisors.max_concurrency(), 2);
        assert_eq!(format!("{advisors:?}"), "ProviderAdvisors { providers: [\"apple\", \"claude\"] }");
        // `on_change` reaches every advisor.
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);
        advisors.on_change(Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }));
        let fake = FakeProvider::available(AutoProvider::Claude);
        let advisors = ProviderAdvisors::default().with(AutoProvider::Claude, Arc::new(fake));
        let c = Arc::clone(&count);
        advisors.on_change(Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }));
        advisors.refresh(None);
        advisors.refresh(Some(AutoProvider::Agy));
        advisors.settle(Duration::from_millis(10));
        advisors.cancel_all();
    }

    #[test]
    fn the_fake_provider_answers_fails_and_tells_about_changes() {
        let fake = FakeProvider::available(AutoProvider::Claude).with_served_model("claude-test-1");
        assert_eq!(fake.concurrency(), 2);
        let answer = fake.ask(&picture(), "Grade it").unwrap();
        assert_eq!(answer.served_model.as_deref(), Some("claude-test-1"));
        assert_eq!(answer.plan, "Test plan: Grade");
        assert_eq!(fake.asks(), 1);
        match fake.availability() {
            Availability::Available(info) => {
                assert_eq!(info.provider, AutoProvider::Claude);
                assert_eq!(info.model, "Test Claude Code");
                assert_eq!(info.tool.unwrap().version, "0.0.0-test");
                assert_eq!(info.os, "");
                assert_eq!(info.prompt, 1);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(FakeProvider::info_for(AutoProvider::Agy).model, "Test Antigravity");
        assert_eq!(FakeProvider::info_for(AutoProvider::Apple), FakeAdvisor::info());
        // A scripted failure.
        fake.fail_with(Some((FailureKind::RateLimited, "busy".to_owned())));
        let e = fake.ask(&picture(), "x").unwrap_err();
        assert_eq!((e.kind, e.message.as_str()), (FailureKind::RateLimited, "busy"));
        fake.fail_with(None);
        assert!(fake.ask(&picture(), "x").is_ok());
        // The listeners hear of a change, once.
        let heard = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&heard);
        fake.on_change(Arc::new(move || {
            h.fetch_add(1, Ordering::SeqCst);
        }));
        let signed_out = Availability::Unavailable(R::SignedOut);
        fake.set_availability(signed_out.clone());
        fake.set_availability(signed_out.clone());
        assert_eq!(heard.load(Ordering::SeqCst), 1);
        assert_eq!(fake.availability(), signed_out);
    }

    #[test]
    fn the_fake_provider_runs_requests_in_parallel_and_cancel_cuts_the_delay() {
        let fake = Arc::new(
            FakeProvider::available(AutoProvider::Claude)
                .with_concurrency(3)
                .with_delay(Duration::from_secs(30)),
        );
        let started = Instant::now();
        let threads: Vec<_> = (0..3)
            .map(|_| {
                let fake = Arc::clone(&fake);
                std::thread::spawn(move || fake.ask(&picture(), "x"))
            })
            .collect();
        while fake.peak_concurrency() < 3 {
            assert!(started.elapsed() < Duration::from_secs(10), "no three at once");
            std::thread::sleep(Duration::from_millis(5));
        }
        fake.cancel();
        for t in threads {
            assert!(t.join().unwrap().is_ok());
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(fake.peak_concurrency(), 3);
    }
}
