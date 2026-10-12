//! What every command that develops photos needs: the core with the CLI's options, the Ctrl-C
//! flag and the GPU probe.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use ym_core::app::{Core, CoreError, CoreOptions};
use ym_core::export_queue::{ExportEvent, ExportSink};
use ym_core::session::SessionError;
use ym_gpu::device::{Gpu, GpuConfig, GpuInfo};

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
}

/// The core and the channel its export queue reports on.
pub struct Runtime {
    /// The core: session, look library, preferences, renderer and export queue.
    pub core: Core,
    /// The export queue's events.
    pub events: mpsc::Receiver<ExportEvent>,
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
        let core = Core::open(options).map_err(|e| core_failure(&e))?;
        Ok(Self { core, events })
    }

    /// Waits until the session has developed every photo, calling `tick(photos_ready, photos)`
    /// about every tenth of a second. Returns false when Ctrl-C came first.
    pub fn wait_developed(&self, mut tick: impl FnMut(usize, usize)) -> bool {
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
            let ready = state
                .photos
                .iter()
                .filter(|p| {
                    matches!(
                        p.status,
                        ym_core::session::PhotoStatus::Ready
                            | ym_core::session::PhotoStatus::Skipped { .. }
                    )
                })
                .count();
            tick(ready, total);
        }
    }
}

/// Opens the GPU the way the app does and reports what it found.
///
/// A software adapter (Mesa lavapipe, WARP) counts: the device is usable, and `GpuInfo` says it is
/// software. The device is closed again when the returned value is dropped.
///
/// # Errors
///
/// The reason there is no usable GPU, in plain words.
pub fn probe_gpu() -> Result<GpuInfo, String> {
    let gpu = Gpu::start(GpuConfig::default()).map_err(|e| e.to_string())?;
    let info = gpu.info().clone();
    gpu.shutdown();
    Ok(info)
}

/// A GPU probe running on another thread, so the answer is ready by the time the photos are
/// developed.
///
/// Dropping it waits for the probe to finish: leaving the process while a graphics driver is still
/// starting up on another thread can crash the driver's exit handlers, and a run that failed
/// early would then end with a signal instead of its exit code.
#[derive(Debug)]
pub struct BackgroundProbe(Option<std::thread::JoinHandle<Result<GpuInfo, String>>>);

impl BackgroundProbe {
    /// Starts probing. If the thread cannot be started, nothing is ever said about the GPU.
    pub fn start() -> Self {
        Self(
            std::thread::Builder::new()
                .name("ym-gpu-probe".to_owned())
                .spawn(|| {
                    // A driver that panics is a GPU that cannot be used, not a reason to stop.
                    std::panic::catch_unwind(probe_gpu).unwrap_or_else(|_| {
                        Err("the graphics driver stopped unexpectedly".to_owned())
                    })
                })
                .ok(),
        )
    }

    /// Waits for the answer. `None` when there is none (the thread could not start or panicked).
    pub fn finish(&mut self) -> Option<Result<GpuInfo, String>> {
        self.0.take().and_then(|handle| handle.join().ok())
    }
}

impl Drop for BackgroundProbe {
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
    fn a_profile_that_does_not_exist_is_a_usage_error_and_other_failures_are_not() {
        let unknown = core_failure(&CoreError::Session(SessionError::Data(
            "There is no camera profile called x.".to_owned(),
        )));
        assert_eq!(unknown.code, exit::USAGE);
        let other = core_failure(&CoreError::Render(ym_core::pipeline::RenderError::Shutdown));
        assert_eq!(other.code, exit::SOME_FAILED);
    }
}
