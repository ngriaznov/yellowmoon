//! What every command that develops photos needs: the core with the CLI's options, the Ctrl-C
//! flag, the graphics device, opened by `ym_core::app::gpu::open_gpu` like the app's, and Apple
//! Auto's advisor (`ym_apple::system_advisor`, the app's too, `docs/19`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreError, CoreOptions};
use ym_core::apple_auto::{AppleAutoMode, UnavailableReason};
use ym_core::export_queue::{ExportEvent, ExportSink};
use ym_core::scheduler::MemoryBudget;
use ym_core::session::SessionError;
use ym_gpu::device::GpuInfo;

use crate::exit::{self, Failure};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
/// The running command looks at [`interrupted`] itself and cleans up after itself.
static POLLING: AtomicBool = AtomicBool::new(false);

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
    /// Apple Auto (`docs/19`): `Some(mode)` hands the core this computer's advisor, as the app does,
    /// with the mode of `--apple-auto` ([`AppleAutoMode::FromPreference`] without it); `None` hands it
    /// none, which is Apple Auto off (`doctor --bench`, which times the render path only).
    pub apple_auto: Option<AppleAutoMode>,
    /// Instructions for Apple Auto's model for this run instead of the `apple_auto_instructions`
    /// preference (`--apple-auto-instructions FILE`, read by [`read_instructions`]).
    pub apple_auto_instructions: Option<String>,
}

/// Reads `--apple-auto-instructions FILE` (`docs/19` section 8): the file's text as UTF-8, at most
/// 2000 characters (`ym_core::apple_auto::INSTRUCTIONS_MAX_CHARS`). The file's final line break
/// (`\n` or `\r\n`, which text editors add) is not part of the instructions, so a file holding the
/// default text is the default. An empty file is the default instructions.
///
/// # Errors
///
/// A usage [`Failure`] (exit code 2) when the file cannot be read or is too long.
pub fn read_instructions(path: &std::path::Path) -> Result<String, Failure> {
    let mut text = std::fs::read_to_string(path).map_err(|e| {
        Failure::usage(format!(
            "Can't read the instructions for Apple Intelligence from {}: {e}.",
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
    /// What the run knows about Apple Auto for its notice ([`Runtime::apple_auto_notice`]).
    apple_watch: Mutex<AppleAutoWatch>,
}

/// What a run knows about Apple Auto for its notice, which it prints once at most (`docs/19`
/// section 8).
#[derive(Debug, Clone, Copy, Default)]
struct AppleAutoWatch {
    /// The run wants Apple Auto: `--apple-auto on`, or no option and the preference on.
    wanted: bool,
    /// Apple Auto was available when it was last read.
    available: bool,
    /// The notice has been printed.
    noticed: bool,
}

impl Runtime {
    /// Opens the core without thumbnails: the CLI shows no pictures, and an export waits for any
    /// thumbnail work that is queued.
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
        if let Some(mode) = settings.apple_auto {
            // The app's advisor: Apple's model where it can look at photos, the reason it cannot
            // everywhere else, the test advisor with `YELLOWMOON_TEST_APPLE_AUTO=fake`.
            options.advisor = Some(ym_apple::system_advisor());
            options.apple_auto = mode;
        }
        options
            .apple_auto_instructions
            .clone_from(&settings.apple_auto_instructions);
        let core = Core::open(options).map_err(|e| core_failure(&e))?;
        Ok(Self::over(core, events))
    }

    /// A runtime over a core that is already open.
    fn over(core: Core, events: mpsc::Receiver<ExportEvent>) -> Self {
        Self {
            core,
            events,
            apple_watch: Mutex::new(AppleAutoWatch::default()),
        }
    }

    /// What `--apple-auto on` says where Apple Auto is not available (`docs/19` section 8): one
    /// warning per run, and the run goes on with Yellowmoon's own Auto and the same exit code.
    /// `None` when the run did not ask for it or it is available (the availability is read again).
    /// Call it once, after the core opens: it also notes whether the run wants Apple Auto and
    /// whether it is available, for [`Runtime::apple_auto_notice_before_export`].
    #[must_use]
    pub fn apple_auto_notice(&self, mode: AppleAutoMode) -> Option<String> {
        let status = self.core.apple_auto_status();
        let mut watch = self.watch();
        watch.wanted = match mode {
            AppleAutoMode::On => true,
            AppleAutoMode::FromPreference => status.preference,
            AppleAutoMode::Off => false,
        };
        watch.available = status.available;
        if mode != AppleAutoMode::On || status.available {
            return None;
        }
        watch.noticed = true;
        Some(unavailable_notice(
            status.reason.unwrap_or(UnavailableReason::Unavailable),
        ))
    }

    /// Reads Apple Auto's availability again just before the export starts (`docs/19` section 1),
    /// with the same refresh the app's `Core::export_start` makes, so the export's photos wait for
    /// the answers of an Apple Auto that is available now and use the rule where it no longer is.
    /// Returns the notice of [`Runtime::apple_auto_notice`] when the run wanted Apple Auto (`--apple-auto
    /// on`, or the preference on), it was available and is not any more, and the run has not said
    /// so yet: once per run.
    #[must_use]
    pub fn apple_auto_notice_before_export(&self) -> Option<String> {
        let status = self.core.apple_auto_status();
        let mut watch = self.watch();
        let lost = watch.wanted && watch.available && !status.available && !watch.noticed;
        watch.available = status.available;
        if !lost {
            return None;
        }
        watch.noticed = true;
        Some(unavailable_notice(
            status.reason.unwrap_or(UnavailableReason::Unavailable),
        ))
    }

    fn watch(&self) -> std::sync::MutexGuard<'_, AppleAutoWatch> {
        self.apple_watch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
            // Every photo developed and the session still busy: Apple Auto's answers, when it acts.
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
    /// While every photo is developed and Apple Auto acts: the photos that have Apple's answer (or a
    /// failure) and the photos developed.
    pub answers: Option<(usize, usize)>,
}

/// The sentence `--apple-auto on` prints where Apple Auto is not available.
#[must_use]
pub fn unavailable_notice(reason: UnavailableReason) -> String {
    format!(
        "Apple Auto is not available on this computer ({}). Auto uses Yellowmoon's own settings.",
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

    #[test]
    fn the_apple_auto_notice_names_the_reason_in_words() {
        assert_eq!(
            unavailable_notice(UnavailableReason::AppleIntelligenceOff),
            "Apple Auto is not available on this computer (Apple Intelligence is off in System Settings). Auto uses Yellowmoon's own settings."
        );
    }

    /// An advisor whose availability the test switches.
    struct Switch(AtomicBool);

    impl ym_core::apple_auto::AutoAdvisor for Switch {
        fn availability(&self) -> ym_core::apple_auto::Availability {
            if self.0.load(Ordering::SeqCst) {
                ym_core::apple_auto::Availability::Available(ym_core::apple_auto::AdvisorInfo::new(
                    "Switch", "test",
                ))
            } else {
                ym_core::apple_auto::Availability::Unavailable(
                    UnavailableReason::AppleIntelligenceOff,
                )
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
            })
        }
    }

    /// A runtime over a core with the switchable advisor, in a new app folder (the preference on),
    /// as `Runtime::open` makes it for `ym render` with `mode`.
    fn runtime_with(
        advisor: &std::sync::Arc<Switch>,
        mode: AppleAutoMode,
        app: &std::path::Path,
    ) -> Runtime {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let (sink, events) = ExportSink::channel();
        let mut options = CoreOptions::new(data, app.to_path_buf());
        options.thumbnails = false;
        options.export_events = sink;
        options.decode_threads = 1;
        options.advisor = Some(advisor.clone());
        options.apple_auto = mode;
        Runtime::over(Core::open(options).expect("core"), events)
    }

    #[test]
    fn apple_auto_is_read_again_before_the_export_and_its_loss_is_said_once() {
        let words = unavailable_notice(UnavailableReason::AppleIntelligenceOff);
        let app = tempfile::tempdir().expect("app folder");
        let advisor = std::sync::Arc::new(Switch(AtomicBool::new(true)));
        // The preference on (a new app folder), no option: available at the start, then gone
        // before the export. The export's re-read sees it (the session now uses the rule) and the
        // run says so, once.
        let rt = runtime_with(&advisor, AppleAutoMode::FromPreference, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::FromPreference), None);
        assert!(rt.core.session.apple_auto_active());
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.apple_auto_notice_before_export(), Some(words.clone()));
        assert!(!rt.core.session.apple_auto_active(), "the session was told");
        assert_eq!(rt.apple_auto_notice_before_export(), None, "once per run");
        // `--apple-auto on` where it is not available: said after the core opens, not again.
        let rt = runtime_with(&advisor, AppleAutoMode::On, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::On), Some(words.clone()));
        assert_eq!(rt.apple_auto_notice_before_export(), None);
        // `--apple-auto on`, available at the start and gone before the export.
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AppleAutoMode::On, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::On), None);
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.apple_auto_notice_before_export(), Some(words));
        // Available again before the export: nothing to say, and the session uses it again.
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AppleAutoMode::FromPreference, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::FromPreference), None);
        advisor.0.store(false, Ordering::SeqCst);
        rt.core.session.refresh_apple_auto();
        advisor.0.store(true, Ordering::SeqCst);
        assert_eq!(rt.apple_auto_notice_before_export(), None);
        assert!(rt.core.session.apple_auto_active());
        // Never available without the option, or off for the run: nothing is said.
        advisor.0.store(false, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AppleAutoMode::FromPreference, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::FromPreference), None);
        assert_eq!(rt.apple_auto_notice_before_export(), None);
        advisor.0.store(true, Ordering::SeqCst);
        let rt = runtime_with(&advisor, AppleAutoMode::Off, app.path());
        assert_eq!(rt.apple_auto_notice(AppleAutoMode::Off), None);
        advisor.0.store(false, Ordering::SeqCst);
        assert_eq!(rt.apple_auto_notice_before_export(), None);
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
