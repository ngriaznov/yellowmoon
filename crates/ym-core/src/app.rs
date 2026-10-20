//! [`Core`]: the session, the renderer, the look library, the preferences and the export queue wired
//! together, with one method per IPC command of `docs/01-architecture.md`.
//!
//! The Tauri application builds one [`Core`] at start and forwards each command to it; the CLI builds
//! one per run. What the parts need from each other is settled here once:
//!
//! * the **renderer** reads looks through the look library and shows the display rendering the
//!   preferences choose;
//! * new photos get the **default look** from the preferences (or Neutral when that look is gone);
//! * a change of the default look, the display rendering or the look list reaches the session and
//!   the renderer, and then the application's `looks_changed` listener;
//! * an **export** snapshots the photos and the preference "Remove location from exports" when it is
//!   queued, and runs through `export_photo`, the function the CLI calls too;
//! * **Apple Auto** (`docs/19`) acts when the caller gives an advisor ([`CoreOptions::advisor`]: the
//!   app and `ym` give `ym_apple::system_advisor()`, every other caller none), the preference or the
//!   run's mode ([`CoreOptions::apple_auto`], [`Core::set_apple_auto_mode`]) is on and the advisor is
//!   available; the availability is read again when a session opens, when the status is asked
//!   ([`Core::apple_auto_status`]), when the preferences change and before each export.
//!
//! Nothing here persists anything but the preferences and the looks (`docs/01` section
//! Persistence).
//!
//! The graphics device is opened in [`gpu`] and nowhere else: [`gpu::open_gpu`] turns a
//! [`gpu::GpuPolicy`] into a [`gpu::GpuSlot`], and [`Core::attach_gpu`] gives the slot to the
//! renderer.

pub mod gpu;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::apple_auto::{AppleAutoMode, AppleAutoStatus, AutoAdvisor, check_instructions};

use ym_color::look::LookInput;
use ym_looks::import::Progress;
use ym_looks::library::LookId;

use crate::cli_preview::{CliPreview, cli_preview};
use crate::data_dir::{DataDirError, find_data_dir};
use crate::export::ExportContext;
use crate::export_queue::{
    ExportEstimate, ExportJob, ExportJobId, ExportQueue, ExportSink, JobSummary, QueueError,
};
use crate::looks::{
    DisplayLutEntry, ImportRequest, LookEntry, LookService, LooksError, LooksImportResult,
};
use crate::pipeline::{AdvisorPictures, RenderError, RenderOptions, Renderer};
use crate::preferences::{
    DisplayRendering, LastUsedFolders, Preferences, PreferencesError, PreferencesPatch,
    PreferencesStore, own_instructions,
};
use crate::session::{
    AppleAutoEnv, ApplyScope, EventSink, PasteReport, PhotoId, RemoveOutcome, RestoreOutcome,
    Session, SessionEnv, SessionError, default_decode_threads,
};
use crate::settings::SettingsPatch;

/// How long [`Core::shutdown`] waits for a cancelled export to delete its partial file. A cancel is
/// noticed between render tiles and writer bands (a few tenths of a second at 24 MP); this is the
/// bound for a tile that takes much longer, after which the application quits anyway.
pub const SHUTDOWN_EXPORT_WAIT: Duration = Duration::from_secs(5);

/// Why a command failed. [`CoreError::scope`] is the `AppError.scope` the webview shows it under.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// The shipped `data/` folder is missing.
    #[error(transparent)]
    Data(#[from] DataDirError),
    /// A session command failed.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// A look command failed.
    #[error(transparent)]
    Looks(#[from] LooksError),
    /// The preferences could not be changed or saved.
    #[error(transparent)]
    Preferences(#[from] PreferencesError),
    /// An export could not be queued.
    #[error(transparent)]
    Export(#[from] QueueError),
    /// The render engine could not start.
    #[error(transparent)]
    Render(#[from] RenderError),
}

impl CoreError {
    /// The `ErrorScope` of `src/api/types.ts` this error belongs to.
    #[must_use]
    pub fn scope(&self) -> &'static str {
        match self {
            Self::Data(_) => "app",
            Self::Session(_) => "open",
            Self::Looks(_) => "looks",
            Self::Preferences(_) => "preferences",
            Self::Export(_) => "export",
            Self::Render(_) => "render",
        }
    }
}

/// How a [`Core`] is built.
pub struct CoreOptions {
    /// The shipped `data/` folder ([`find_data_dir`]).
    pub data_root: PathBuf,
    /// The app folder: `preferences.json` and `looks/` live in it
    /// (`ym_looks::library::default_root`; `YELLOWMOON_HOME` replaces it).
    pub app_root: PathBuf,
    /// Where the session's events go (`photo_progress`, `thumbnail_ready`).
    pub session_events: EventSink,
    /// Where the export queue's events go (`export_progress`, busy or idle).
    pub export_events: ExportSink,
    /// The render engine's options (threads, tile sizes, the GPU).
    pub render: RenderOptions,
    /// Threads of the decode pool.
    pub decode_threads: usize,
    /// Forces a camera profile for every photo (the CLI's `--profile`).
    pub forced_profile: Option<String>,
    /// The home folder, for the export fallback `~/Pictures/Yellowmoon export`. `None`: `$HOME`.
    pub home: Option<PathBuf>,
    /// Render a thumbnail for every photo that finishes developing (`thumbnail_ready`). The
    /// application wants them; the CLI does not (it would spend the render worker on pictures
    /// nobody sees, and an export waits for such work).
    pub thumbnails: bool,
    /// What looks at the photos for Apple Auto (`docs/19`): the app and `ym` give
    /// `ym_apple::system_advisor()`; every other caller (tests, the golden tool, examples) gives none,
    /// which is Apple Auto off and today's bytes everywhere.
    pub advisor: Option<Arc<dyn AutoAdvisor>>,
    /// How this run decides on Apple Auto: the preference ([`AppleAutoMode::FromPreference`], the
    /// app's and `ym`'s default) or `ym --apple-auto on|off`.
    pub apple_auto: AppleAutoMode,
    /// Instructions for Apple Auto's model for this run instead of the `apple_auto_instructions`
    /// preference (`ym --apple-auto-instructions FILE`, `docs/19` section 8); `None` follows the
    /// preference. Checked by [`Core::open`].
    pub apple_auto_instructions: Option<String>,
}

impl CoreOptions {
    /// The options of this machine: the data folder and the app folder as the app and the CLI find
    /// them, no event listeners.
    ///
    /// # Errors
    ///
    /// [`CoreError::Data`] when the `data/` folder is missing, [`CoreError::Looks`] when there is no
    /// home folder for the app folder.
    pub fn from_environment() -> Result<Self, CoreError> {
        let data_root = find_data_dir()?;
        let app_root = ym_looks::library::default_root().map_err(LooksError::from)?;
        Ok(Self::new(data_root, app_root))
    }

    /// Options over the given folders, with no event listeners.
    #[must_use]
    pub fn new(data_root: PathBuf, app_root: PathBuf) -> Self {
        let render = RenderOptions {
            neural: crate::pipeline::NeuralStatus::probe(Some(&data_root)),
            ..RenderOptions::default()
        };
        Self {
            data_root,
            app_root,
            session_events: EventSink::none(),
            export_events: ExportSink::none(),
            render,
            decode_threads: default_decode_threads(),
            forced_profile: None,
            home: None,
            thumbnails: true,
            advisor: None,
            apple_auto: AppleAutoMode::FromPreference,
            apple_auto_instructions: None,
        }
    }
}

/// Everything the application and the CLI work with (see the module documentation).
pub struct Core {
    /// The preferences (`preferences.json`).
    pub preferences: Arc<PreferencesStore>,
    /// The look library.
    pub looks: Arc<LookService>,
    /// The render engine.
    pub renderer: Renderer,
    /// The photos opened in this run.
    pub session: Arc<Session>,
    /// The export queue.
    pub exports: ExportQueue,
    data_root: PathBuf,
    /// Apple Auto's instructions for this run when they override the preference
    /// ([`CoreOptions::apple_auto_instructions`]).
    instructions_override: Option<String>,
}

impl std::fmt::Debug for Core {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Core")
            .field("data_root", &self.data_root)
            .finish_non_exhaustive()
    }
}

impl Core {
    /// Builds the core: opens the preferences and the look library, starts the render engine, the
    /// decode pool and the export queue. Nothing is written to disk.
    ///
    /// # Errors
    ///
    /// [`CoreError::Session`] when the data folder cannot be read (profiles, noise priors),
    /// [`CoreError::Looks`], [`CoreError::Render`].
    pub fn open(options: CoreOptions) -> Result<Self, CoreError> {
        let instructions_override = match &options.apple_auto_instructions {
            Some(text) => {
                check_instructions(text).map_err(PreferencesError::Invalid)?;
                Some(own_instructions(text))
            }
            None => None,
        };
        let preferences = Arc::new(PreferencesStore::open(&options.app_root));
        let own = instructions_override
            .clone()
            .unwrap_or_else(|| preferences.get().apple_auto_instructions);
        let looks = Arc::new(LookService::open(
            &options.app_root,
            &options.data_root,
            Arc::clone(&preferences),
        )?);
        let renderer = Renderer::new(looks.provider(), options.render)?;
        renderer.set_display(looks.display_choice());

        let mut env = SessionEnv::load(&options.data_root)?
            .with_events(options.session_events)
            .with_default_look(looks.effective_default())
            .with_decode_threads(options.decode_threads);
        if options.thumbnails {
            env = env.with_thumbnails(renderer.thumbnail_renderer());
        }
        if let Some(profile) = &options.forced_profile {
            env = env.with_forced_profile(profile)?;
        }
        if let Some(advisor) = options.advisor {
            // The model's pictures come from a renderer of their own, without the device, so they
            // are the CPU reference path's bytes whatever the GPU policy (`docs/19` section 3).
            let pictures = AdvisorPictures::new(env.reference_look.clone(), &renderer.options())?;
            env = env.with_apple_auto(AppleAutoEnv {
                preference: preferences.get().apple_auto,
                mode: options.apple_auto,
                instructions: own.clone(),
                ..AppleAutoEnv::new(advisor, Arc::new(pictures))
            });
        }
        let session = Arc::new(Session::new(env));
        // The preference is known to the session with or without an advisor (the status says it).
        session.set_apple_auto_preference(preferences.get().apple_auto);
        session.set_apple_auto_instructions(&own);

        let ctx = Arc::new(
            ExportContext::new(renderer.clone())
                .with_home(options.home.clone().or_else(ym_export::naming::home_dir)),
        );
        let exports = ExportQueue::new(ctx, options.export_events);

        // The first listener on `looks_changed` keeps the session and the renderer in step; the
        // application's own listener (to tell the webview) is added after it.
        {
            let looks_weak = Arc::downgrade(&looks);
            let session = Arc::clone(&session);
            let renderer = renderer.clone();
            looks.on_change(move || {
                if let Some(looks) = looks_weak.upgrade() {
                    session.set_default_look(looks.effective_default());
                    renderer.set_display(looks.display_choice());
                }
            });
        }
        Ok(Self {
            preferences,
            looks,
            renderer,
            session,
            exports,
            data_root: options.data_root,
            instructions_override,
        })
    }

    /// The shipped data folder.
    #[must_use]
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// Gives the renderer the device of `slot`, the stages it may run and the memory budgets that
    /// go with it (the unified-memory rule lowers the CPU budget on a machine whose GPU shares the
    /// RAM). `ym` opens the device in the background while photos decode and attaches it here
    /// before it queues the export, so that one export is wholly on the device or wholly on the
    /// CPU: call it before the first render of the run, not during one. A slot that uses the CPU
    /// takes any device away (the renderer then uses the CPU for everything).
    pub fn attach_gpu(&self, slot: &gpu::GpuSlot) {
        if slot.budget() != self.renderer.options().budget {
            self.renderer.set_budget(slot.budget());
        }
        self.renderer.set_gpu(slot.gpu().cloned(), slot.stages());
        self.renderer.set_gpu_strict(slot.policy().is_strict());
    }

    /// Registers the application's listener for `looks_changed`.
    pub fn on_looks_changed(&self, f: impl Fn() + Send + Sync + 'static) {
        self.looks.on_change(f);
    }

    /// Closes the session (⌘W): drops the photos and cancels the viewport, tile, thumbnail,
    /// histogram and look-tile renders, queued and running, then forgets what the renderer cached
    /// for them and refuses any such request for the closed session's photos that is still on its
    /// way (a command that read its photo before the close and queues its job after the cancel).
    /// Exports that were queued keep going; they hold their own snapshots.
    pub fn close_session(&self) {
        let last = self.session.last_photo_id();
        self.session.close();
        self.renderer.scheduler().cancel_foreground();
        match last {
            Some(last) => self.renderer.retire_through(last),
            None => self.renderer.forget_all(),
        }
    }

    /// `remove_photos`: takes photos out of the filmstrip ([`Session::remove`]) and stops the renderer working for them:
    /// their viewport, tile, thumbnail, histogram and look-tile renders are cancelled, queued and running, the render caches
    /// they filled are dropped, and requests for them are refused from now on, so nothing is cached for a photo that is
    /// gone. **The files on disk are not touched** (hard rule 5). Exports that were queued keep going with the snapshots
    /// they hold: a photo removed while its export runs is still exported, with the settings it had when the export was
    /// started.
    ///
    /// What was taken out is kept for [`Core::restore_photos`] until the next removal or the end of the session.
    pub fn remove_photos(&self, ids: &[PhotoId]) -> RemoveOutcome {
        let outcome = self.session.remove(ids);
        self.renderer.retire_photos(&outcome.removed);
        outcome
    }

    /// `restore_photos`: Undo of the last [`Core::remove_photos`] ([`Session::restore_removed`]): the photos come back
    /// where they were, with their settings, and the renderer serves them again.
    pub fn restore_photos(&self) -> RestoreOutcome {
        // Served again before they are in the session, so that the first request for one is not refused.
        self.renderer.unretire_photos(&self.session.restorable());
        self.session.restore_removed()
    }

    /// Quitting: cancels exports and renders, and waits (up to [`SHUTDOWN_EXPORT_WAIT`]) for the
    /// export queue to stop.
    ///
    /// The process ends as soon as this returns, so a worker that was still writing would never
    /// get to delete its temporary file, and a hidden `.yellowmoon-*.tmp` of tens or hundreds of
    /// megabytes would stay in the person's export folder for good. A cancel is noticed between
    /// render tiles and writer bands, a fraction of a second away.
    pub fn shutdown(&self) {
        self.exports.cancel_all();
        self.close_session();
        self.renderer.scheduler().cancel_all();
        if !self.exports.wait_idle(SHUTDOWN_EXPORT_WAIT) {
            tracing::warn!(
                "an export did not stop within {} s; quitting anyway",
                SHUTDOWN_EXPORT_WAIT.as_secs()
            );
        }
    }

    // --- Preferences ----------------------------------------------------------------------

    /// `preferences_get`.
    #[must_use]
    pub fn preferences_get(&self) -> Preferences {
        self.preferences.get()
    }

    /// `preferences_set`: validates the patch (the default look must exist and be usable, a display
    /// LUT must be imported), saves it, and brings the session and the renderer in step.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`] with "That look isn't in the library." or "File missing or changed.
    /// Import again." or "Those display LUTs aren't imported…", [`CoreError::Preferences`] when the
    /// file cannot be saved. Nothing changes on an error.
    pub fn preferences_set(&self, patch: &PreferencesPatch) -> Result<Preferences, CoreError> {
        let next = self.looks.preferences_set(patch)?;
        self.session
            .set_default_look(self.looks.effective_default());
        self.renderer.set_display(self.looks.display_choice());
        // Apple Auto follows its preference at once (off: every photo back to the rule; on: the
        // photos without an answer are asked), with the availability read again; new instructions
        // ask every photo again (unless this run overrides them).
        self.session.set_apple_auto_preference(next.apple_auto);
        if self.instructions_override.is_none() {
            self.session
                .set_apple_auto_instructions(&next.apple_auto_instructions);
        }
        Ok(next)
    }

    // --- Apple Auto -----------------------------------------------------------------------

    /// Apple Auto's status for the app's Preferences (whether to show the row) and `ym doctor`, with
    /// the availability read again (`docs/19` sections 1, 2 and 8). Without an advisor it is not
    /// available and has no reason.
    #[must_use]
    pub fn apple_auto_status(&self) -> AppleAutoStatus {
        self.session.apple_auto_status()
    }

    /// Sets how this run decides on Apple Auto (`ym --apple-auto on|off`; the app keeps
    /// [`AppleAutoMode::FromPreference`]). Takes effect at once, as the preference does.
    pub fn set_apple_auto_mode(&self, mode: AppleAutoMode) {
        self.session.set_apple_auto_mode(mode);
    }

    // --- Looks ------------------------------------------------------------------------------

    /// `looks_list`. Looks at the library folder first (a stat per look and the manifest), so an
    /// import made by the CLI shows up and a look whose file has gone is marked missing; if that
    /// changed the list, `looks_changed` fires before this returns.
    #[must_use]
    pub fn looks_list(&self) -> Vec<LookEntry> {
        self.looks.reload();
        self.looks.list()
    }

    /// `looks_display_luts`.
    #[must_use]
    pub fn looks_display_luts(&self) -> Vec<DisplayLutEntry> {
        self.looks.reload();
        self.looks.display_luts()
    }

    /// `looks_import`: a zip, a folder or a `.cube` file; `input` answers the generic question.
    /// Remembers the folder it came from.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`].
    pub fn looks_import(
        &self,
        path: &Path,
        input: Option<LookInput>,
        progress: &mut dyn FnMut(Progress<'_>),
    ) -> Result<LooksImportResult, CoreError> {
        let result = self.looks.import(
            path,
            &ImportRequest {
                input,
                cancel: None,
            },
            progress,
        )?;
        if let Some(folder) = path.parent() {
            self.remember_folder(|f| f.looks_import = Some(folder.to_path_buf()));
        }
        Ok(result)
    }

    /// `looks_remove`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`].
    pub fn looks_remove(&self, id: &LookId) -> Result<(), CoreError> {
        self.looks.remove(id)?;
        Ok(())
    }

    /// `looks_set_default`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`].
    pub fn looks_set_default(&self, id: &LookId) -> Result<(), CoreError> {
        self.looks.set_default(id)?;
        self.session
            .set_default_look(self.looks.effective_default());
        Ok(())
    }

    /// `apply_look`: copies a look and its intensity to This photo, the Picked photos or All.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`] for a look that is not in the library or whose file is gone.
    pub fn apply_look(
        &self,
        scope: ApplyScope,
        id: &LookId,
        intensity: f64,
    ) -> Result<Vec<PhotoId>, CoreError> {
        Ok(self.looks.apply_look(&self.session, scope, id, intensity)?)
    }

    /// `copy_settings`: a photo's settings as a patch to paste onto other photos
    /// ([`SettingsPatch::copied_from`]).
    ///
    /// # Errors
    ///
    /// [`CoreError::Session`] for a photo that is unknown, failed or not developed yet.
    pub fn copy_settings(&self, id: PhotoId) -> Result<SettingsPatch, CoreError> {
        Ok(self.session.copy_settings(id)?)
    }

    /// `paste_settings`: applies a copied patch to the photos in `ids` ([`Session::paste_settings`]).
    /// A look in the patch whose file is gone, or that is not in the library any more, is refused,
    /// as `apply_look` refuses it: the photos would not draw.
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`] for such a look, [`CoreError::Session`] for a patch that cannot be applied.
    pub fn paste_settings(
        &self,
        ids: &[PhotoId],
        patch: &SettingsPatch,
    ) -> Result<PasteReport, CoreError> {
        if let Some(look) = &patch.look {
            self.looks.check_usable(look)?;
        }
        Ok(self.session.paste_settings(ids, patch)?)
    }

    // --- Export -----------------------------------------------------------------------------

    /// `export_estimate`.
    #[must_use]
    pub fn export_estimate(&self, job: &ExportJob) -> ExportEstimate {
        self.exports.estimate(&self.session, job)
    }

    /// `export_start`: snapshots the photos and the preference "Remove location from exports" and
    /// queues the job. Remembers the destination folder when one was chosen. Looks at the library
    /// folder first, as [`Core::looks_list`] does (and fires `looks_changed` if that changed the
    /// list), so a look or display LUT whose file has gone is noticed now.
    ///
    /// The session need not have finished developing: the burst median exposure and the pooled
    /// noise of a photo depend on the photos of its burst and of its noise pool, and a photo whose
    /// turn comes before they are final waits for them (not for thumbnails, prefetching or
    /// viewport work) and reads them then, so the file is the one `ym render` writes after
    /// waiting for the whole session (`ExportQueue::start`).
    ///
    /// # Errors
    ///
    /// [`CoreError::Looks`] when the display LUT chosen in the preferences is gone,
    /// [`CoreError::Export`] for the rest (`QueueError::Developing` when photos of the scope have
    /// no estimates yet).
    pub fn export_start(&self, job: &ExportJob) -> Result<ExportJobId, CoreError> {
        // Look at the library folder first: a look or display LUT whose file has gone since the
        // last look at it is noticed now, and a look the CLI imported is known.
        self.looks.reload();
        // A display LUT the user chose and that has since gone is an error, not ACES 2 in its place.
        self.looks.check_display_rendering()?;
        // Apple Auto may have become available, or stopped being, since the last look (`docs/19`
        // section 1); the export's photos then wait for their answers.
        self.session.refresh_apple_auto();
        let remove_location = self.preferences.get().remove_location_from_exports;
        let id = self.exports.start(&self.session, job, remove_location)?;
        if let Some(folder) = &job.options.destination {
            self.remember_folder(|f| f.export = Some(folder.clone()));
        }
        Ok(id)
    }

    /// `export_cancel`.
    pub fn export_cancel(&self, id: ExportJobId) {
        self.exports.cancel(id);
    }

    /// How a job stands (counts, the files written, the folders): "N photos saved to {folder}.".
    #[must_use]
    pub fn export_summary(&self, id: ExportJobId) -> Option<JobSummary> {
        self.exports.summary(id)
    }

    /// What "Show in Finder" opens for a job.
    #[must_use]
    pub fn export_reveal_path(&self, id: ExportJobId) -> Option<PathBuf> {
        self.exports.reveal_path(id)
    }

    /// Whether an export is queued or running: "Restart to update" waits while this is true.
    #[must_use]
    pub fn export_busy(&self) -> bool {
        self.exports.is_busy()
    }

    /// `cli_preview`: the `ym render …` command for a job.
    #[must_use]
    pub fn cli_preview(&self, job: &ExportJob) -> String {
        self.cli_preview_detail(job).command
    }

    /// [`Core::cli_preview`] with the number of photos the command leaves adjustments out for.
    #[must_use]
    pub fn cli_preview_detail(&self, job: &ExportJob) -> CliPreview {
        cli_preview(
            &self.session,
            job,
            Some(&self.looks),
            self.exports.context().home(),
        )
    }

    /// Remembers a folder in the preferences. A failure to save is logged, never shown: a folder
    /// not being remembered is no reason to fail the command that used it.
    fn remember_folder(&self, set: impl FnOnce(&mut LastUsedFolders)) {
        // One read-modify-write under the store's lock: two commands remembering different
        // folders at once (an export and an import) keep both.
        if let Err(e) = self.preferences.modify(|prefs| {
            set(&mut prefs.last_used_folders);
            Ok(())
        }) {
            tracing::warn!(kind = e.kind(), "could not remember the folder");
            tracing::debug!(error = %e, "could not remember the folder: details");
        }
    }
}

/// Makes a [`DisplayRendering`] from a display LUT id, for callers that build patches.
#[must_use]
pub fn arri_display(id: LookId) -> DisplayRendering {
    DisplayRendering::ArriDisplay { id }
}
