//! What every command that develops photos needs: the core with the CLI's options, the Ctrl-C
//! flag, the graphics device, opened by `ym_core::app::gpu::open_gpu` like the app's, and the Auto
//! providers' advisors (Apple's `ym_apple::system_advisor`, the app's too, `docs/19`, and the two
//! programs' `ym_agents::system_advisors`, `docs/22`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Duration;

use ym_agents::StartProbe;
use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreError, CoreOptions};
use ym_core::apple_auto::UnavailableReason;
use ym_core::auto_provider::{
    AutoChoice, AutoProvider, AutoStatus, ProviderAdvisors, ProviderState, cloud_notice,
    unavailable_hint,
};
use ym_core::export_queue::{ExportEvent, ExportSink};
use ym_core::preferences::PreferencesStore;
use ym_core::scheduler::MemoryBudget;
use ym_core::session::SessionError;
use ym_gpu::device::GpuInfo;

use crate::exit::{self, Failure};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
/// The running command looks at [`interrupted`] itself and cleans up after itself.
static POLLING: AtomicBool = AtomicBool::new(false);
/// The advisors of the run, so that Ctrl-C can kill the programs they started: a child is not
/// stopped by its parent's death on Unix (`docs/22` section 9.4).
static RUNNING_ADVISORS: OnceLock<ProviderAdvisors> = OnceLock::new();

/// Tells the Ctrl-C handler which advisors the run has, so that a signal kills the programs they
/// started (`docs/22` section 9.4). The first registration wins: a process has one run.
pub fn register_advisors(advisors: &ProviderAdvisors) {
    let _ = RUNNING_ADVISORS.set(advisors.clone());
}

/// Installs the Ctrl-C handler (and SIGTERM and SIGHUP).
///
/// What the first signal does depends on the command. One that calls [`handle_interrupts`] (an
/// export, an import, a contact sheet) is asked to stop: it finishes the current tile, deletes the
/// partial file and exits 130. Everywhere else (waiting for a list on stdin, opening the library,
/// `doctor`) nothing would ever look at the flag and Ctrl-C would seem dead until it was pressed
/// again, so the process ends at once with 130: nothing is half-written there. A second signal
/// always ends the process at once.
pub fn install_interrupt_handler() {
    let result = ctrlc::set_handler(|| {
        // First of all: the programs a request started must not outlive `ym`.
        if let Some(advisors) = RUNNING_ADVISORS.get() {
            advisors.cancel_all();
        }
        let again = INTERRUPTED.swap(true, Ordering::SeqCst);
        if again || !POLLING.load(Ordering::SeqCst) {
            std::process::exit(i32::from(exit::INTERRUPTED));
        }
    });
    if let Err(error) = result {
        tracing::warn!(%error, "could not install the Ctrl-C handler");
    }
}

/// Says that the command checks [`interrupted`] from here on and stops cleanly when it is set, so
/// the first Ctrl-C only sets the flag.
pub fn handle_interrupts() {
    POLLING.store(true, Ordering::SeqCst);
}

/// Whether Ctrl-C was pressed.
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// The failure a [`CoreError`] stands for. A missing data folder is a usage error with the message
/// of `docs/01` (exit code 2), and so is a profile id that does not exist.
pub fn core_failure(error: &CoreError) -> Failure {
    match error {
        CoreError::Data(e) => Failure::usage(e.to_string()),
        CoreError::Session(SessionError::Data(message)) => Failure::usage(message.clone()),
        other => Failure::failed(other.to_string()),
    }
}

/// How the core is built for a run.
#[derive(Debug, Clone, Default)]
pub struct CoreSettings {
    /// Photos developed at the same time (`--jobs`); `None` is the app's number.
    pub jobs: Option<u32>,
    /// A camera profile that replaces the detected one (`--profile`).
    pub forced_profile: Option<String>,
    /// The Auto providers (`docs/22`): `Some(choice)` hands the core the advisors of this computer,
    /// as the app does, with the choice of `--auto-provider` ([`AutoChoice::FromPreference`] without
    /// it); `None` hands it none, which is Yellowmoon's own Auto (`doctor --bench`, which times the
    /// render path only).
    pub auto: Option<AutoChoice>,
    /// Instructions for the model for this run instead of the `apple_auto_instructions` preference
    /// (`--auto-instructions FILE`, read by [`read_instructions`]).
    pub auto_instructions: Option<String>,
}

/// How long `ym` waits for the program it is going to use to say whether it can answer (`docs/22`
/// section 13.1).
pub const AUTO_SETTLE: Duration = Duration::from_secs(30);

/// The advisors of this computer, as the app builds them (`docs/22` section 13.1): Apple's, and the
/// two programs' without a first probe (`StartProbe::Never`), so a run that does not use a program
/// never starts one.
#[must_use]
pub fn system_provider_advisors() -> ProviderAdvisors {
    let mut advisors =
        ProviderAdvisors::default().with(AutoProvider::Apple, ym_apple::system_advisor());
    for (provider, advisor) in ym_agents::system_advisors(StartProbe::Never) {
        advisors = advisors.with(provider, advisor);
    }
    advisors
}

/// Reads `--auto-instructions FILE` (`docs/19` section 8, `docs/22` section 13.1): the file's text as
/// UTF-8, at most 2000 characters (`ym_core::apple_auto::INSTRUCTIONS_MAX_CHARS`). The file's final
/// line break (`\n` or `\r\n`, which text editors add) is not part of the instructions, so a file
/// holding the default text is the default. An empty file is the default instructions.
///
/// # Errors
///
/// A usage [`Failure`] (exit code 2) when the file cannot be read or is too long.
pub fn read_instructions(path: &std::path::Path) -> Result<String, Failure> {
    let mut text = std::fs::read_to_string(path).map_err(|e| {
        Failure::usage(format!(
            "Can't read the instructions from {}: {e}.",
            path.display()
        ))
    })?;
    if text.ends_with('\n') {
        text.pop();
        if text.ends_with('\r') {
            text.pop();
        }
    }
    ym_core::apple_auto::check_instructions(&text).map_err(Failure::usage)?;
    Ok(text)
}

/// The core and the channel its export queue reports on.
pub struct Runtime {
    /// The core: session, look library, preferences, renderer and export queue.
    pub core: Core,
    /// The export queue's events.
    pub events: mpsc::Receiver<ExportEvent>,
    /// What the run knows about the Auto providers for its notices ([`Runtime::auto_notices`]).
    auto_watch: Mutex<AutoWatch>,
}

/// What a run knows about the Auto providers for its notices, each of which it prints once at most
/// (`docs/22` section 13.2).
#[derive(Debug, Clone, Copy, Default)]
struct AutoWatch {
    /// How the run chose its provider.
    choice: AutoChoice,
    /// The run wants a provider other than Yellowmoon's own Auto to act: it named one, or the saved
    /// choice is one whose consent is saved.
    wanted: bool,
    /// The chosen provider was available when it was last read.
    available: bool,
    /// The "not available" sentence has been printed.
    noticed: bool,
    /// The sentence of the cloud provider has been printed.
    cloud_noticed: bool,
    /// The breaker's sentence has been printed.
    breaker_noticed: bool,
}

impl Runtime {
    /// Opens the core without thumbnails: the CLI shows no pictures, and an export waits for any
    /// thumbnail work that is queued.
    ///
    /// With `settings.auto` the advisors of this computer go to the core; when the provider the run
    /// will use is Claude Code or Antigravity (named, or saved with its consent), it is probed and
    /// waited for (up to [`AUTO_SETTLE`]) before the core opens, so no run starts with `checking`
    /// and no run touches a program it does not use.
    ///
    /// # Errors
    ///
    /// The [`Failure`] of [`core_failure`].
    pub fn open(settings: &CoreSettings) -> Result<Self, Failure> {
        let (sink, events) = ExportSink::channel();
        // Finding the data folder and the app folder is about how Yellowmoon is installed and
        // configured (`YELLOWMOON_DATA`, `YELLOWMOON_HOME`, a home folder), so it is code 2.
        let mut options =
            CoreOptions::from_environment().map_err(|e| Failure::usage(e.to_string()))?;
        options.thumbnails = false;
        options.export_events = sink;
        if let Some(jobs) = settings.jobs {
            options.decode_threads = jobs.max(1) as usize;
        }
        options.forced_profile.clone_from(&settings.forced_profile);
        if let Some(choice) = settings.auto {
            // The app's advisors: Apple's model where it can look at photos, the reason it cannot
            // everywhere else, the test advisor with `YELLOWMOON_TEST_APPLE_AUTO=fake`; the two
            // programs, or what `YELLOWMOON_TEST_AUTO_PROVIDERS` says they are.
            let advisors = system_provider_advisors();
            let prefs = PreferencesStore::open(&options.app_root).get();
            let chosen = choice.chosen(prefs.auto_provider);
            if chosen.is_cloud() && (choice.is_explicit() || prefs.auto_consent.get(chosen)) {
                advisors.refresh(Some(chosen));
                advisors.settle(AUTO_SETTLE);
            }
            // Ctrl-C kills the programs a request started.
            register_advisors(&advisors);
            options.advisors = advisors;
            options.auto_choice = choice;
        }
        options
            .apple_auto_instructions
            .clone_from(&settings.auto_instructions);
        let core = Core::open(options).map_err(|e| core_failure(&e))?;
        let choice = settings.auto.unwrap_or_default();
        Ok(Self::over(core, events, choice))
    }

    /// A runtime over a core that is already open.
    fn over(core: Core, events: mpsc::Receiver<ExportEvent>, choice: AutoChoice) -> Self {
        Self {
            core,
            events,
            auto_watch: Mutex::new(AutoWatch {
                choice,
                ..AutoWatch::default()
            }),
        }
    }

    fn watch(&self) -> std::sync::MutexGuard<'_, AutoWatch> {
        self.auto_watch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The Auto status as the advisors last said it, without asking any program to probe again.
    #[must_use]
    pub fn auto_status(&self) -> AutoStatus {
        self.core.session.auto_status_cached()
    }

    /// What the run says on stderr once, after the core opens (`docs/22` section 13.2), each as a
    /// warning (`--quiet` hides them):
    ///
    /// * the chosen provider is not available, but only when the run **named** it (`--auto-provider`,
    ///   `--apple-auto on`): a run that only follows the saved choice says nothing, as the app hides
    ///   its row;
    /// * a cloud provider acts: where the picture goes ([`cloud_notice`]).
    ///
    /// Call it once: it also notes whether the run wants a provider and whether it is available, for
    /// [`Runtime::auto_notice_before_export`].
    #[must_use]
    pub fn auto_notices(&self) -> Vec<String> {
        let status = self.auto_status();
        let mut watch = self.watch();
        let chosen = status.chosen;
        watch.wanted = chosen != AutoProvider::Yellowmoon
            && (watch.choice.is_explicit() || status.consent.get(chosen));
        watch.available = chosen_state(&status) == ProviderState::Available;
        let mut notices = Vec::new();
        if watch.choice.is_explicit()
            && chosen != AutoProvider::Yellowmoon
            && status.effective != chosen
        {
            watch.noticed = true;
            notices.push(unavailable_notice(&status, chosen));
        }
        if status.effective.is_cloud() && !watch.cloud_noticed {
            watch.cloud_noticed = true;
            let api = status
                .provider(status.effective)
                .and_then(|p| p.info.as_ref())
                .and_then(|i| i.tool.as_ref())
                .and_then(|t| t.api.as_deref());
            notices.extend(cloud_notice(status.effective, api));
        }
        notices
    }

    /// Reads the chosen provider's availability again just before the export starts (`docs/19`
    /// section 1, `docs/22` section 3.3), with the same refresh the app's `Core::export_start` makes
    /// (and the same wait of up to 5 s while it still reads `checking`), so the export's photos wait
    /// for the answers of a provider that is available now and use the rule where it no longer is.
    /// Returns the sentence of [`Runtime::auto_notices`] when the run wanted a provider (named, or
    /// saved with its consent), it was available and is not any more, and the run has not said so yet:
    /// once per run.
    #[must_use]
    pub fn auto_notice_before_export(&self) -> Option<String> {
        if matches!(
            self.core.session.refresh_apple_auto(),
            Some(ym_core::apple_auto::Availability::Unavailable(
                UnavailableReason::Checking
            ))
        ) {
            self.core
                .settle_auto(ym_core::app::AUTO_SETTLE_BEFORE_EXPORT);
        }
        let status = self.auto_status();
        let mut watch = self.watch();
        let now = chosen_state(&status) == ProviderState::Available;
        let lost = watch.wanted && watch.available && !now && !watch.noticed;
        watch.available = now;
        if !lost {
            return None;
        }
        watch.noticed = true;
        Some(unavailable_notice(&status, status.chosen))
    }

    /// The breaker's sentence (`docs/22` section 9.2), once, when an advisor stopped asking after three
    /// failures in a row during the run: "Claude Code failed three times in a row (<the last failure in
    /// words>). Photos it doesn't answer use Yellowmoon's own Auto." Read it after the run's waits.
    #[must_use]
    pub fn breaker_notice(&self) -> Option<String> {
        let mut watch = self.watch();
        if watch.breaker_noticed {
            return None;
        }
        let sentence = [AutoProvider::Claude, AutoProvider::Agy]
            .into_iter()
            .find_map(|p| self.core.session.advisor_details(p)?.breaker)?;
        watch.breaker_noticed = true;
        Some(sentence)
    }

    /// Whether the run named its provider (`--auto-provider`, `--apple-auto`).
    #[must_use]
    pub fn explicit_choice(&self) -> bool {
        self.watch().choice.is_explicit()
    }

    /// The provider whose Auto the run asks, for the progress line ("Asking Claude Code").
    #[must_use]
    pub fn asking(&self) -> AutoProvider {
        self.core.session.chosen_provider()
    }

    /// Waits until the session has developed every photo and, when Apple Auto acts, until every
    /// developed photo has its answer or its failure (`docs/19` section 6), calling `tick` about
    /// every tenth of a second. Returns false when Ctrl-C came first.
    pub fn wait_developed(&self, mut tick: impl FnMut(Waiting)) -> bool {
        use ym_core::session::PhotoStatus;
        handle_interrupts();
        loop {
            if self.core.session.wait_idle(Duration::from_millis(100)) {
                return true;
            }
            if interrupted() {
                return false;
            }
            let state = self.core.session.state();
            let total = state.photos.len();
            let done = |p: &&ym_core::session::PhotoSummary| {
                matches!(p.status, PhotoStatus::Ready | PhotoStatus::Skipped { .. })
            };
            let ready = state.photos.iter().filter(done).count();
            // Every photo developed and the session still busy: the provider's answers, when it acts.
            let answers = (ready == total && self.core.session.apple_auto_active()).then(|| {
                let developed: Vec<_> = state
                    .photos
                    .iter()
                    .filter(|p| matches!(p.status, PhotoStatus::Ready))
                    .collect();
                let settled = developed
                    .iter()
                    .filter(|p| {
                        self.core
                            .session
                            .estimates(p.id)
                            .is_some_and(|e| e.apple.is_settled())
                    })
                    .count();
                (settled, developed.len())
            });
            tick(Waiting {
                ready,
                total,
                answers,
            });
        }
    }
}

/// How far [`Runtime::wait_developed`] has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waiting {
    /// Photos developed (or that cannot be developed).
    pub ready: usize,
    /// Photos open.
    pub total: usize,
    /// While every photo is developed and a provider acts: the photos that have its answer (or a
    /// failure) and the photos developed.
    pub answers: Option<(usize, usize)>,
}

/// Where the provider chosen for the run stands.
fn chosen_state(status: &AutoStatus) -> ProviderState {
    status
        .provider(status.chosen)
        .map_or(ProviderState::NotInstalled, |p| p.state)
}

/// The sentence a run that named a provider prints where it is not available (`docs/22` section 13.2).
/// Apple Intelligence: "Apple Auto is not available on this computer (<reason in words>). Auto uses
/// Yellowmoon's own settings." The two programs: "Claude Code is not available on this computer (it
/// isn't installed). Auto uses Yellowmoon's own settings.", or for a program that is there but cannot
/// be used the hint of `docs/22` section 3.4 followed by " Auto uses Yellowmoon's own settings.".
#[must_use]
pub fn unavailable_notice(status: &AutoStatus, provider: AutoProvider) -> String {
    let provider_status = status.provider(provider);
    let reason = provider_status
        .and_then(|p| p.reason)
        .unwrap_or(UnavailableReason::NotInstalled);
    let version = provider_status.and_then(|p| p.tool_version.as_deref());
    let hint = unavailable_hint(provider, reason, version, std::env::consts::OS);
    if provider.is_cloud() && !hint.is_empty() && reason != UnavailableReason::NotInstalled {
        return format!("{hint} Auto uses Yellowmoon's own settings.");
    }
    let what = match provider {
        AutoProvider::Apple => "Apple Auto",
        other => other.name(),
    };
    format!(
        "{what} is not available on this computer ({}). Auto uses Yellowmoon's own settings.",
        reason.words()
    )
}

/// The graphics device being opened on another thread, so that the answer is ready by the time the
/// photos are developed, and the kernels compile in the background if an export will use them
/// (`GpuSlot::warm_up_for_exports`, an `Export`-priority job that gives way to the first render
/// job and stops when the device shuts down).
///
/// Dropping it waits for the open to finish: leaving the process while a graphics driver is still
/// starting up on another thread can crash the driver's exit handlers, and a run that failed
/// early would then end with a signal instead of its exit code.
#[derive(Debug)]
pub struct BackgroundGpu(Option<std::thread::JoinHandle<GpuSlot>>);

impl BackgroundGpu {
    /// Starts opening the device for `policy`. If the thread cannot be started, nothing is ever
    /// said about the GPU.
    pub fn start(policy: GpuPolicy, budget: MemoryBudget) -> Self {
        Self(
            std::thread::Builder::new()
                .name("ym-gpu-open".to_owned())
                .spawn(move || {
                    let slot = open_gpu(policy, budget);
                    // The handle is not kept: the job runs on the device thread, its result is
                    // only a count, and a failing kernel shows again when its stage runs.
                    drop(slot.warm_up_for_exports());
                    slot
                })
                .ok(),
        )
    }

    /// Waits for the answer. `None` when there is none (the thread could not start or panicked).
    pub fn finish(&mut self) -> Option<GpuSlot> {
        self.0.take().and_then(|handle| handle.join().ok())
    }
}

impl Drop for BackgroundGpu {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// The one-line description of an opened GPU: `Apple M1 (Metal, IntegratedGpu)`.
pub fn gpu_line(info: &GpuInfo) -> String {
    format!(
        "{} ({}, {}){}",
        info.name,
        info.backend,
        info.device_type,
        if info.is_software { ", software" } else { "" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_core::apple_auto::Availability;
    use ym_core::auto_provider::{AdvisorDetails, AutoConsent, StatusInput};
    use ym_core::data_dir::DataDirError;

    #[test]
    fn a_missing_data_folder_is_a_usage_error_with_the_message_of_docs_01() {
        let failure = core_failure(&CoreError::Data(DataDirError::Missing));
        assert_eq!(failure.code, exit::USAGE);
        assert_eq!(
            failure.message,
            "Yellowmoon's data folder is missing. Reinstall the app."
        );
    }

    /// A status where the advisors say these things (Apple, Claude Code, Antigravity), the choice and
    /// the consent as given.
    fn status_with(
        readings: [Option<Availability>; 3],
        choice: AutoChoice,
        preference: AutoProvider,
        consent: AutoConsent,
    ) -> AutoStatus {
        AutoStatus::build(&StatusInput {
            has_advisors: true,
            readings: readings.map(|r| {
                r.map(|a| {
                    (
                        a,
                        AdvisorDetails {
                            tool_version: Some("2.1.100".to_owned()),
                            ..AdvisorDetails::default()
                        },
                    )
                })
            }),
            preference,
            choice,
            consent,
            own_instructions: "",
        })
    }

    #[test]
    fn the_notice_of_a_named_provider_that_is_not_available_says_why_in_words() {
        use UnavailableReason as R;
        let status = |reason: R| {
            status_with(
                [
                    Some(Availability::Unavailable(R::AppleIntelligenceOff)),
                    Some(Availability::Unavailable(reason)),
                    None,
                ],
                AutoChoice::FromPreference,
                AutoProvider::Apple,
                AutoConsent::default(),
            )
        };
        let s = status(R::NotInstalled);
        assert_eq!(
            unavailable_notice(&s, AutoProvider::Apple),
            "Apple Auto is not available on this computer (Apple Intelligence is off in System Settings). Auto uses Yellowmoon's own settings."
        );
        assert_eq!(
            unavailable_notice(&s, AutoProvider::Claude),
            "Claude Code is not available on this computer (it isn't installed). Auto uses Yellowmoon's own settings."
        );
        // Not installed at all (no advisor registered): the same words.
        assert_eq!(
            unavailable_notice(&s, AutoProvider::Agy),
            "Antigravity is not available on this computer (it isn't installed). Auto uses Yellowmoon's own settings."
        );
        let signed_out = status(R::SignedOut);
        let hint = unavailable_notice(&signed_out, AutoProvider::Claude);
        assert!(
            hint.starts_with("Claude Code is installed but not signed in. Open ")
                && hint.ends_with(" Auto uses Yellowmoon's own settings."),
            "{hint}"
        );
        let old = unavailable_notice(&status(R::TooOld), AutoProvider::Claude);
        assert!(old.starts_with("Claude Code 2.1.100 is too old. Update it: "), "{old}");
        // A reason with no hint falls back to the sentence with the reason in words.
        assert_eq!(
            unavailable_notice(&status(R::Checking), AutoProvider::Claude),
            "Claude Code is not available on this computer (Yellowmoon is still checking). Auto uses Yellowmoon's own settings."
        );
    }

    /// An advisor whose availability the test switches.
    struct Switch(AtomicBool);

    impl ym_core::apple_auto::AutoAdvisor for Switch {
        fn availability(&self) -> Availability {
            if self.0.load(Ordering::SeqCst) {
                Availability::Available(ym_core::apple_auto::AdvisorInfo::new("Switch", "test"))
            } else {
                Availability::Unavailable(UnavailableReason::AppleIntelligenceOff)
            }
        }

        fn ask(
            &self,
            _picture: &ym_core::apple_auto::AdvisorPicture,
            _instructions: &str,
        ) -> Result<ym_core::apple_auto::AdvisorAnswer, ym_core::apple_auto::AdvisorError> {
            Ok(ym_core::apple_auto::AdvisorAnswer {
                steps: ym_core::apple_auto::AutoSteps::keep(),
                scene: String::new(),
                plan: String::new(),
                served_model: None,
            })
        }
    }

    /// A runtime over a core with the switchable advisor as Apple's, in a new app folder (the
    /// preference Apple Intelligence), as `Runtime::open` makes it for `ym render` with `choice`.
    fn runtime_with(
        advisor: &std::sync::Arc<Switch>,
        choice: AutoChoice,
        app: &std::path::Path,
    ) -> Runtime {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let (sink, events) = ExportSink::channel();
        let mut options = CoreOptions::new(data, app.to_path_buf());
        options.thumbnails = false;
        options.export_events = sink;
        options.decode_threads = 1;
        options.advisors =
            ProviderAdvisors::default().with(AutoProvider::Apple, advisor.clone());
        options.auto_choice = choice;
        Runtime::over(Core::open(options).expect("core"), events, choice)
    }

    #[test]
    fn a_provider_is_read_again_before_the_export_and_its_loss_is_said_once() {
        let sentence = "Apple Auto is not available on this computer (Apple Intelligence is off in System Settings). Auto uses Yellowmoon's own settings.";
        let apple = AutoChoice::Provider(AutoProvider::Apple);
        let app = tempfile::tempdir().expect("app folder");
        let advisor = std::sync::Arc::new(Switch(AtomicBool::new(true)));
        // The saved choice is Apple (a new app folder), no option: available at the start, then gone
        // before the export. The export's re-read sees it (the session now uses the rule) and the
        // run says so, once.
        let rt = runtime_with(&advisor, AutoChoice::FromPreference, app.path());
        assert!(rt.auto_notices().is_empty());
        assert!(rt.core.session.apple_auto_active());
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.auto_notice_before_export().as_deref(), Some(sentence));
        assert!(!rt.core.session.apple_auto_active(), "the session was told");
        assert_eq!(rt.auto_notice_before_export(), None, "once per run");
        // `--auto-provider apple` where it is not available: said after the core opens, not again.
        let rt = runtime_with(&advisor, apple, app.path());
        assert_eq!(rt.auto_notices(), vec![sentence.to_owned()]);
        assert_eq!(rt.auto_notice_before_export(), None);
        // The same, available at the start and gone before the export.
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(&advisor, apple, app.path());
        assert!(rt.auto_notices().is_empty());
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.auto_notice_before_export().as_deref(), Some(sentence));
        // Available again before the export: nothing to say, and the session uses it again.
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AutoChoice::FromPreference, app.path());
        assert!(rt.auto_notices().is_empty());
        advisor.0.store(false, Ordering::SeqCst);
        rt.core.session.refresh_apple_auto();
        advisor.0.store(true, Ordering::SeqCst);
        assert_eq!(rt.auto_notice_before_export(), None);
        assert!(rt.core.session.apple_auto_active());
        // Never available without the option: nothing is said, the app hides its row.
        advisor.0.store(false, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AutoChoice::FromPreference, app.path());
        assert!(rt.auto_notices().is_empty());
        assert_eq!(rt.auto_notice_before_export(), None);
        // Yellowmoon's own Auto for the run: nothing is wanted, nothing is said.
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(
            &advisor,
            AutoChoice::Provider(AutoProvider::Yellowmoon),
            app.path(),
        );
        assert!(rt.auto_notices().is_empty());
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.auto_notice_before_export(), None);
    }

    #[test]
    fn a_profile_that_does_not_exist_is_a_usage_error_and_other_failures_are_not() {
        let unknown = core_failure(&CoreError::Session(SessionError::Data(
            "There is no camera profile called x.".to_owned(),
        )));
        assert_eq!(unknown.code, exit::USAGE);
        let other = core_failure(&CoreError::Render(ym_core::pipeline::RenderError::Shutdown));
        assert_eq!(other.code, exit::SOME_FAILED);
    }
}
