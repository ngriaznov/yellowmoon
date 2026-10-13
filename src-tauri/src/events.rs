//! Events from the core to the webview.
//!
//! Everything the shell sends goes through [`Emitter`], so the command handlers and the viewport
//! can be tested without a webview: [`TauriEmitter`] sends the events to the window, `Recorder` (tests only)
//! keeps them for a test. The event names and payloads are the ones of `docs/01` §IPC and
//! `src/api/types.ts` (`ApiEvents`), declared in [`crate::wire`].

use tauri::Runtime;
use tauri_specta::Event;

use crate::wire::{
    AppError, DownloadFound, ErrorEvent, ExportBusy, ExportProgress, LooksChanged,
    LooksImportProgress, PhotoId, PhotoProgress, RenderStats, SettingsChanged, ThumbnailReady,
    ViewportFrame,
};

/// An event for the webview.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    /// `photo_progress`.
    PhotoProgress(PhotoProgress),
    /// `thumbnail_ready`.
    ThumbnailReady(ThumbnailReady),
    /// `render_stats`.
    RenderStats(RenderStats),
    /// `export_progress`.
    ExportProgress(ExportProgress),
    /// `looks_changed`.
    LooksChanged,
    /// `error`.
    Error(AppError),
    /// `looks_import_progress`.
    LooksImportProgress(LooksImportProgress),
    /// `download_found`.
    DownloadFound(DownloadFound),
    /// `settings_changed` (addition).
    SettingsChanged(PhotoId),
    /// `export_busy` (addition).
    ExportBusy(bool),
    /// `viewport_frame` (addition, fallback viewport).
    ViewportFrame(ViewportFrame),
}

impl AppEvent {
    /// The name of the event on the bridge.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::PhotoProgress(_) => "photo_progress",
            Self::ThumbnailReady(_) => "thumbnail_ready",
            Self::RenderStats(_) => "render_stats",
            Self::ExportProgress(_) => "export_progress",
            Self::LooksChanged => "looks_changed",
            Self::Error(_) => "error",
            Self::LooksImportProgress(_) => "looks_import_progress",
            Self::DownloadFound(_) => "download_found",
            Self::SettingsChanged(_) => "settings_changed",
            Self::ExportBusy(_) => "export_busy",
            Self::ViewportFrame(_) => "viewport_frame",
        }
    }
}

/// Where events go. Called from any thread, including the core's worker threads while they hold
/// internal locks, so an implementation must not block and must not call back into the core.
pub trait Emitter: Send + Sync + 'static {
    /// Delivers one event. A webview that is gone loses it; nothing else happens.
    fn emit(&self, event: AppEvent);
}

/// Sends events to the app's windows.
pub struct TauriEmitter<R: Runtime> {
    app: tauri::AppHandle<R>,
}

impl<R: Runtime> TauriEmitter<R> {
    /// An emitter for the windows of `app`.
    #[must_use]
    pub fn new(app: tauri::AppHandle<R>) -> Self {
        Self { app }
    }
}

impl<R: Runtime> Emitter for TauriEmitter<R> {
    fn emit(&self, event: AppEvent) {
        let name = event.name();
        let sent = match event {
            AppEvent::PhotoProgress(e) => e.emit(&self.app),
            AppEvent::ThumbnailReady(e) => e.emit(&self.app),
            AppEvent::RenderStats(e) => e.emit(&self.app),
            AppEvent::ExportProgress(e) => e.emit(&self.app),
            AppEvent::LooksChanged => LooksChanged.emit(&self.app),
            AppEvent::Error(e) => ErrorEvent::from(e).emit(&self.app),
            AppEvent::LooksImportProgress(e) => e.emit(&self.app),
            AppEvent::DownloadFound(e) => e.emit(&self.app),
            AppEvent::SettingsChanged(photo_id) => SettingsChanged { photo_id }.emit(&self.app),
            AppEvent::ExportBusy(busy) => ExportBusy { busy }.emit(&self.app),
            AppEvent::ViewportFrame(e) => e.emit(&self.app),
        };
        if let Err(e) = sent {
            tracing::debug!(event = name, error = %e, "could not send an event");
        }
    }
}

/// Keeps every event, for tests.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct Recorder {
    events: std::sync::Mutex<Vec<AppEvent>>,
}

#[cfg(test)]
impl Recorder {
    /// An empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything recorded so far.
    #[must_use]
    pub fn events(&self) -> Vec<AppEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Waits until an event for which `pick` returns a value arrives, up to `timeout`.
    #[must_use]
    pub fn wait_for<T>(
        &self,
        timeout: std::time::Duration,
        mut pick: impl FnMut(&AppEvent) -> Option<T>,
    ) -> Option<T> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(found) = self.events().iter().find_map(&mut pick) {
                return Some(found);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
impl Emitter for Recorder {
    fn emit(&self, event: AppEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }
}
