//! [`AppApi`]: one method per command of `docs/01` §IPC (and the additions of `docs/11`), over
//! `ym-core`, with no Tauri types in sight.
//!
//! The Tauri commands in [`crate::commands`] are one-line wrappers that run these methods on a
//! worker thread, so everything here is tested without a webview (`tests/` of this crate and the
//! tests at the foot of this file) against synthetic ARW files. Events leave through the
//! [`Emitter`] the API was built with.
//!
//! State: the [`Core`] (session, renderer, look library, preferences, export queue), the
//! [`Viewport`] controller and the thumbnail refresher, the Downloads watcher and the place of the
//! command-line tool. Nothing here persists anything: the core writes the preferences and the
//! look library, the application the log (`crate::logging`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::{ExportEvent, ExportJob as CoreExportJob, ExportSink};
use ym_core::pipeline::PhotoSource;
use ym_core::session::{EventSink, OpenOptions, PhotoId as CorePhotoId, SessionEvent};
use ym_gpu::queue::{CancelToken, Priority};
use ym_looks::import::Stage;
use ym_looks::import::watch::WatchConfig;
use ym_looks::library::LookId as CoreLookId;

use crate::cli_link::{CliLink, Elevate};
use crate::events::{AppEvent, Emitter};
use crate::thumbs::Thumbnails;
use crate::viewport::{FrameSink, FrameStore, Viewport, core_id};
use crate::wire::{
    AppError, AppInfo, ApplyScope, CliLinkResult, CliLinkStatus, DisplayLut, ErrorScope,
    ExportEstimate, ExportJob, ExportJobId, HistogramData, Look, LookId, LookInput,
    LooksImportProgress, LooksImportResult, PhotoId, PhotoProgress, PixelSample, Preferences,
    PreferencesPatch, ResolvedSettings, SessionState, SettingsPatch, SourceRecognition,
    ThumbnailReady, ViewportBackendKind, ViewportInfo, ViewportInput, ViewportMode, ViewportRect,
    WatchStatus, convert,
};
use crate::wizard::{Downloads, recognise};

/// How the application is started: where its data and its files are.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// The shipped `data/` folder.
    pub data_root: PathBuf,
    /// The app folder (`preferences.json` and `looks/`).
    pub app_root: PathBuf,
    /// The home folder, for the export fallback and the Downloads watcher; `None`: `$HOME`.
    pub home: Option<PathBuf>,
    /// Use the GPU for the colour pass when there is one.
    pub gpu: bool,
    /// Render a thumbnail for every photo that finishes developing.
    pub thumbnails: bool,
    /// Threads of the decode pool; `None`: the core's default.
    pub decode_threads: Option<usize>,
    /// The Downloads watcher's settings.
    pub watch: WatchConfig,
}

impl AppConfig {
    /// The folders of this machine, as the app and the CLI find them. The GPU is used on macOS and
    /// wherever `YELLOWMOON_GPU=on`; `YELLOWMOON_GPU=off` turns it off.
    ///
    /// # Errors
    ///
    /// [`AppError`] with "Yellowmoon's data folder is missing. Reinstall the app." when there is no
    /// data folder.
    pub fn from_environment() -> Result<Self, AppError> {
        let options = CoreOptions::from_environment().map_err(AppError::from)?;
        let gpu = match std::env::var("YELLOWMOON_GPU").as_deref() {
            Ok("off" | "0") => false,
            Ok("on" | "1") => true,
            _ => cfg!(target_os = "macos"),
        };
        Ok(Self {
            data_root: options.data_root,
            app_root: options.app_root,
            home: None,
            gpu,
            thumbnails: true,
            decode_threads: None,
            watch: WatchConfig::default(),
        })
    }
}

/// The command handlers (see the module documentation).
pub struct AppApi {
    core: Arc<Core>,
    emitter: Arc<dyn Emitter>,
    viewport: Viewport,
    thumbs: Thumbnails,
    downloads: Downloads,
    watch: WatchConfig,
    cli: Option<CliLink>,
    elevate: Box<dyn Elevate + Send + Sync>,
    home: Option<PathBuf>,
    /// The frames and thumbnails the webview loads from the `ym-frame` scheme.
    frames: Arc<FrameStore>,
    /// Orders "queue an export" against "restart the app": see [`AppApi::begin_restart`].
    lifecycle: Mutex<Lifecycle>,
}

/// Where the application is in its life, for [`AppApi::begin_restart`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lifecycle {
    Running,
    /// A restart into an update has begun: no export is queued from now on.
    Restarting,
}

/// The `AppError` for a session command that failed.
fn session_error(e: ym_core::session::SessionError) -> AppError {
    AppError::from(e)
}

fn job_error(e: ym_gpu::queue::JobError) -> AppError {
    use ym_gpu::queue::JobError;
    match e {
        JobError::Cancelled => AppError::new(ErrorScope::Render, "The render was cancelled."),
        other => {
            tracing::error!(error = %other, "a render job failed");
            AppError::new(
                ErrorScope::Render,
                "The photo could not be drawn because of an unexpected problem. Try again.",
            )
        }
    }
}

fn look_id(id: &LookId) -> Result<CoreLookId, AppError> {
    CoreLookId::parse(&id.0).map_err(|_| {
        AppError::new(
            ErrorScope::Looks,
            "That look isn't in the library. Pick another look.",
        )
    })
}

/// The fields of a patch that change what a thumbnail shows.
fn touches_thumbnail(patch: &SettingsPatch) -> bool {
    patch.look.is_some()
        || patch.look_intensity.is_some()
        || patch.crop.is_some()
        || patch.orientation.is_some()
}

impl AppApi {
    /// Builds the core and the viewport and starts the background threads.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the data folder cannot be read or a thread cannot start.
    pub fn start(
        config: AppConfig,
        emitter: Arc<dyn Emitter>,
        sink: Arc<dyn FrameSink>,
        frames: Arc<FrameStore>,
        elevate: Box<dyn Elevate + Send + Sync>,
        cli: Option<CliLink>,
    ) -> Result<Arc<Self>, AppError> {
        let viewport = Viewport::new(Arc::clone(&emitter), sink);
        let shared = viewport.shared();

        let session_events = {
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            let frames = Arc::clone(&frames);
            EventSink::new(move |event| match event {
                SessionEvent::PhotoProgress(p) => {
                    // The estimates are in: the photo can be drawn (the thumbnail comes after).
                    // A photo that gave up has nothing to draw, which the cell must show too.
                    if p.done
                        && (p.error.is_some() || p.stage == ym_core::session::PhotoStage::Estimate)
                    {
                        shared.photo_ready(p.id);
                    }
                    match convert::<_, PhotoProgress>(&p) {
                        Ok(p) => emitter.emit(AppEvent::PhotoProgress(p)),
                        Err(e) => tracing::error!(error = %e, "photo_progress did not convert"),
                    }
                }
                SessionEvent::ThumbnailReady(t) => {
                    // The picture stays in the shell; the webview gets its address.
                    let url = frames.put_thumbnail(t.id.0, t.png_bytes);
                    emitter.emit(AppEvent::ThumbnailReady(ThumbnailReady {
                        id: PhotoId(t.id.0),
                        url,
                    }));
                }
                SessionEvent::EstimatesChanged { id } => shared.estimates_changed(id),
            })
        };
        let export_events = {
            let emitter = Arc::clone(&emitter);
            ExportSink::new(move |event| match event {
                ExportEvent::Progress(p) => match convert(&p) {
                    Ok(p) => emitter.emit(AppEvent::ExportProgress(p)),
                    Err(e) => tracing::error!(error = %e, "export_progress did not convert"),
                },
                ExportEvent::Busy(busy) => emitter.emit(AppEvent::ExportBusy(busy)),
                // `File` is for the command line; the enum may grow.
                _ => {}
            })
        };

        let mut options = CoreOptions::new(config.data_root.clone(), config.app_root.clone());
        options.session_events = session_events;
        options.export_events = export_events;
        options.home = config.home.clone();
        options.thumbnails = config.thumbnails;
        if let Some(threads) = config.decode_threads {
            options.decode_threads = threads;
        }
        if config.gpu {
            options.render.gpu = start_gpu();
        }
        let core = Arc::new(Core::open(options).map_err(AppError::from)?);
        {
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            core.on_looks_changed(move || {
                emitter.emit(AppEvent::LooksChanged);
                shared.invalidate();
            });
        }
        viewport.attach(&core)?;
        let thumbs = Thumbnails::new(Arc::clone(&emitter), Arc::clone(&frames));
        thumbs.attach(&core).map_err(AppError::internal)?;
        Ok(Arc::new(Self {
            core,
            emitter,
            viewport,
            thumbs,
            downloads: Downloads::default(),
            watch: config.watch,
            cli,
            elevate,
            home: config.home.or_else(ym_export::naming::home_dir),
            frames,
            lifecycle: Mutex::new(Lifecycle::Running),
        }))
    }

    /// The core, for the tests.
    #[cfg(test)]
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.core
    }

    /// The viewport controller, for the tests.
    #[cfg(test)]
    pub(crate) fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// The thumbnail refresher, for the tests.
    #[cfg(test)]
    pub(crate) fn thumbnails(&self) -> &Thumbnails {
        &self.thumbs
    }

    /// Stops what runs in the background (quitting): exports and renders are cancelled and partial
    /// files are removed.
    pub fn shutdown(&self) {
        self.downloads.set(
            false,
            None,
            self.core.looks.packages(),
            &self.watch,
            Arc::clone(&self.emitter),
        );
        self.core.shutdown();
    }

    // --- Session ------------------------------------------------------------------------------

    /// `open_paths`: starts a session or adds photos. Folders are read one level deep.
    ///
    /// # Errors
    ///
    /// [`AppError`] with "No ARW files in what you dropped." when nothing is an ARW file, or when
    /// what was given is not there.
    pub fn open_paths(&self, paths: &[String]) -> Result<SessionState, AppError> {
        let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        let outcome = self.core.session.open(&paths, OpenOptions::default());
        // Something that is not there is not "no ARW files": say so first.
        if outcome.added.is_empty() && outcome.already_open == 0 && !outcome.missing.is_empty() {
            return Err(AppError::new(
                ErrorScope::Open,
                "That file isn't there any more. Check that its drive is connected, then open it again.",
            ));
        }
        if let Some(message) = outcome.message() {
            return Err(AppError::new(ErrorScope::Open, message));
        }
        self.remember_open_folder(&paths);
        self.sync_current();
        self.session_state()
    }

    /// Remembers the folder the first path came from (`last_used_folders.open`). A failure to save
    /// is logged: not remembering a folder is no reason to refuse to open photos.
    fn remember_open_folder(&self, paths: &[PathBuf]) {
        let Some(first) = paths.first() else { return };
        let folder = if first.is_dir() {
            first.clone()
        } else {
            first.parent().map(Path::to_path_buf).unwrap_or_default()
        };
        if folder.as_os_str().is_empty() {
            return;
        }
        if let Err(e) = self.core.preferences.modify(|prefs| {
            prefs.last_used_folders.open = Some(folder);
            Ok(())
        }) {
            tracing::warn!(error = %e, "could not remember the folder");
        }
    }

    /// `session_state`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes of the core and of the bridge disagree (a bug).
    pub fn session_state(&self) -> Result<SessionState, AppError> {
        convert(&self.core.session.state())
    }

    /// `close_session`: forgets the photos and stops what is developing; exports keep going.
    pub fn close_session(&self) {
        self.core.close_session();
        self.thumbs.clear();
        self.frames.clear_thumbnails();
        self.viewport.set_photo(None);
    }

    /// `thumbnails()` (addition): the address of the thumbnail of every photo that has one, to get
    /// them again after the webview reloaded (the core announces each thumbnail once, with
    /// `thumbnail_ready`). Only addresses: the pictures stay here.
    #[must_use]
    pub fn thumbnails_now(&self) -> Vec<ThumbnailReady> {
        self.frames
            .thumbnail_urls()
            .into_iter()
            .map(|(id, url)| ThumbnailReady {
                id: PhotoId(id),
                url,
            })
            .collect()
    }

    /// `set_current`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that is not in the session.
    pub fn set_current(&self, id: PhotoId) -> Result<(), AppError> {
        self.core
            .session
            .set_current(core_id(id))
            .map_err(session_error)?;
        self.sync_current();
        Ok(())
    }

    /// `set_picked`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that is not in the session.
    pub fn set_picked(&self, id: PhotoId, picked: bool) -> Result<(), AppError> {
        self.core
            .session
            .set_picked(core_id(id), picked)
            .map_err(session_error)
    }

    /// Tells the renderer and the viewport which photo is current.
    fn sync_current(&self) {
        let current = self.core.session.current();
        if let Some(id) = current {
            self.core.renderer.navigate(id);
        }
        self.viewport.set_photo(current);
    }

    // --- Settings -----------------------------------------------------------------------------

    /// `get_settings`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown or unfinished photo.
    pub fn get_settings(&self, id: PhotoId) -> Result<ResolvedSettings, AppError> {
        convert(
            &self
                .core
                .session
                .settings(core_id(id))
                .map_err(session_error)?,
        )
    }

    /// `update_settings`: applies a patch and returns the settings resolved.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the patch cannot be applied (a Pick white spot that is too bright, a bad
    /// curve); nothing changes then.
    pub fn update_settings(
        &self,
        id: PhotoId,
        patch: &SettingsPatch,
    ) -> Result<ResolvedSettings, AppError> {
        let core_patch: ym_core::settings::SettingsPatch = convert(patch)?;
        let resolved = self
            .core
            .session
            .update_settings(core_id(id), &core_patch)
            .map_err(session_error)?;
        self.settings_changed(core_id(id), touches_thumbnail(patch));
        convert(&resolved)
    }

    /// A photo's settings changed: draw it again if it is on screen, and refresh its thumbnail
    /// when the change shows in one.
    fn settings_changed(&self, id: CorePhotoId, thumbnail: bool) {
        if self.core.session.current() == Some(id) {
            self.viewport.invalidate();
        }
        if thumbnail {
            self.thumbs.refresh([id]);
        }
    }

    /// `apply_look`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a look that is not in the library or whose file is gone.
    pub fn apply_look(
        &self,
        scope: ApplyScope,
        look: &LookId,
        intensity: f64,
    ) -> Result<(), AppError> {
        let id = look_id(look)?;
        let changed = self
            .core
            .apply_look(convert(&scope)?, &id, intensity)
            .map_err(AppError::from)?;
        if let Some(current) = self.core.session.current()
            && changed.contains(&current)
        {
            self.viewport.invalidate();
        }
        self.thumbs.refresh(changed);
        Ok(())
    }

    /// `rerun_auto`.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the photo cannot be decoded again.
    pub fn rerun_auto(&self, id: PhotoId) -> Result<ResolvedSettings, AppError> {
        let resolved = self
            .core
            .session
            .rerun_auto(core_id(id))
            .map_err(session_error)?;
        self.settings_changed(core_id(id), false);
        convert(&resolved)
    }

    /// `histogram`: the photo as it is rendered now. It is asked for after every frame, so it runs
    /// as a thumbnail-priority job: a viewport frame that is wanted meanwhile goes first, as it
    /// does for the Look tab's tiles.
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown or unfinished photo.
    pub fn histogram(&self, id: PhotoId) -> Result<HistogramData, AppError> {
        let id = core_id(id);
        let src = PhotoSource::from_session(&self.core.session, id).map_err(session_error)?;
        let renderer = self.core.renderer.clone();
        let histogram = self
            .core
            .renderer
            .scheduler()
            .submit(Priority::Thumbnail, Some(id), move |token| {
                renderer.histogram(&src, token)
            })
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        convert(&histogram)
    }

    // --- Viewport -----------------------------------------------------------------------------

    /// Which backend draws the photo.
    #[must_use]
    pub fn viewport_info(&self) -> ViewportInfo {
        ViewportInfo {
            backend: self.viewport.kind(),
        }
    }

    /// `viewport_set_rect`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn viewport_set_rect(&self, rect: &ViewportRect) -> Result<(), AppError> {
        self.viewport.set_rect(rect)
    }

    /// `viewport_input`. A Pick white click is converted to a point of the photo, set as the
    /// photo's `white_pick`, and announced with `settings_changed`; the other inputs move the view.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the spot clicked cannot be used as a white.
    pub fn viewport_input(&self, event: &ViewportInput) -> Result<(), AppError> {
        match *event {
            ViewportInput::PickWhite { x, y } => self.pick_white(x.0, y.0),
            _ => self.viewport.input(event),
        }
    }

    fn pick_white(&self, x: f64, y: f64) -> Result<(), AppError> {
        let (photo, state) = self.viewport.snapshot_synced();
        let Some(photo) = photo else { return Ok(()) };
        let src = PhotoSource::from_session(&self.core.session, photo).map_err(session_error)?;
        let point = self
            .core
            .renderer
            .pick_point_at(&src, &state, (x, y), &CancelToken::new())
            .map_err(AppError::from)?;
        // A click on the paper around the photo picks nothing.
        let Some(point) = point else { return Ok(()) };
        let patch = ym_core::settings::SettingsPatch {
            white_pick: Some(Some(point)),
            ..ym_core::settings::SettingsPatch::default()
        };
        self.core
            .session
            .update_settings(photo, &patch)
            .map_err(session_error)?;
        self.settings_changed(photo, false);
        self.emitter
            .emit(AppEvent::SettingsChanged(PhotoId(photo.0)));
        Ok(())
    }

    /// `viewport_set_mode`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn viewport_set_mode(&self, mode: &ViewportMode) -> Result<(), AppError> {
        self.viewport.set_mode(mode)
    }

    /// `set_before`.
    pub fn set_before(&self, before: bool) {
        self.viewport.set_before(before);
    }

    /// `sample_pixel`: display values under a point of the cell, in CSS pixels from its corner.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the point is outside the photo (on the paper around it) or no photo is open.
    pub fn sample_pixel(&self, x: f64, y: f64) -> Result<PixelSample, AppError> {
        let outside = || AppError::new(ErrorScope::Viewport, "That point is outside the photo.");
        let (photo, state) = self.viewport.snapshot_synced();
        let photo = photo.ok_or_else(outside)?;
        let src = PhotoSource::from_session(&self.core.session, photo).map_err(session_error)?;
        let sample = self
            .core
            .renderer
            .sample_pixel(&src, &state, (x, y), &CancelToken::new())
            .map_err(AppError::from)?;
        convert(&sample.ok_or_else(outside)?)
    }

    // --- Looks --------------------------------------------------------------------------------

    /// `looks_list`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn looks_list(&self) -> Result<Vec<Look>, AppError> {
        convert(&self.core.looks_list())
    }

    /// `look_thumbnail`: the photo as it is developed now through one look. The PNG stays here
    /// and is served on the `ym-frame` scheme, like the filmstrip's thumbnails; the answer is its
    /// address.
    ///
    /// # Errors
    ///
    /// [`AppError`] with scope `looks` and "File missing or changed. Import again." when the look's
    /// file is gone.
    pub fn look_thumbnail(&self, photo: PhotoId, look: &LookId) -> Result<String, AppError> {
        let id = look_id(look)?;
        let src =
            PhotoSource::from_session(&self.core.session, core_id(photo)).map_err(session_error)?;
        let png = self
            .core
            .renderer
            .spawn_look_thumbnail(src, id)
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        // The session closed while the tile was rendering: nobody wants the picture, and the
        // store was just emptied.
        self.core
            .session
            .view(core_id(photo))
            .map_err(session_error)?;
        Ok(self.frames.put_look_tile(photo.0, &look.0, png))
    }

    /// `looks_import`: a zip, a folder or a `.cube` file; `input` answers the generic question for
    /// `.cube` files that match no ARRI package. Progress arrives as `looks_import_progress`.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the importer's sentence (not a look package, the question unanswered, …).
    pub fn looks_import(
        &self,
        path: &str,
        input: Option<LookInput>,
    ) -> Result<LooksImportResult, AppError> {
        let core_input: Option<ym_color::look::LookInput> = match input {
            Some(i) => Some(convert(&i)?),
            None => None,
        };
        let emitter = Arc::clone(&self.emitter);
        let source = path.to_owned();
        let mut progress = move |p: ym_looks::import::Progress<'_>| {
            // The scan before the copy is the same files read once; the bar follows the copy.
            if p.stage == Stage::Copy {
                emitter.emit(AppEvent::LooksImportProgress(LooksImportProgress {
                    path: source.clone(),
                    done: u32::try_from(p.done).unwrap_or(u32::MAX),
                    total: u32::try_from(p.total).unwrap_or(u32::MAX),
                    file: p.file.map(str::to_owned),
                }));
            }
        };
        let result = self
            .core
            .looks_import(Path::new(path), core_input, &mut progress)
            .map_err(AppError::from)?;
        convert(&result)
    }

    /// `looks_recognise`: what a zip, a folder or a `.cube` file is, before anything is copied.
    #[must_use]
    pub fn looks_recognise(&self, path: &str) -> SourceRecognition {
        recognise(self.core.looks.packages(), Path::new(path))
    }

    /// `looks_display_luts`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn looks_display_luts(&self) -> Result<Vec<DisplayLut>, AppError> {
        convert(&self.core.looks_display_luts())
    }

    /// `looks_watch_downloads`: starts or stops the Downloads watcher.
    #[must_use]
    pub fn looks_watch_downloads(&self, enabled: bool) -> WatchStatus {
        let folder = self.home.as_ref().map(|h| h.join("Downloads"));
        self.downloads.set(
            enabled,
            folder.as_deref(),
            self.core.looks.packages(),
            &self.watch,
            Arc::clone(&self.emitter),
        )
    }

    /// `looks_set_default`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a look that is not in the library or whose file is gone.
    pub fn looks_set_default(&self, id: &LookId) -> Result<(), AppError> {
        self.core
            .looks_set_default(&look_id(id)?)
            .map_err(AppError::from)
    }

    /// `looks_remove`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a look that is not imported.
    pub fn looks_remove(&self, id: &LookId) -> Result<(), AppError> {
        self.core
            .looks_remove(&look_id(id)?)
            .map_err(AppError::from)
    }

    /// The folder "Show Looks Folder" opens: `looks/` of the app folder, made if it is not there yet.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the folder cannot be made.
    pub fn looks_folder(&self) -> Result<PathBuf, AppError> {
        let folder = self.core.looks.app_root().join("looks");
        std::fs::create_dir_all(&folder).map_err(|e| {
            AppError::new(
                ErrorScope::Looks,
                format!(
                    "The looks folder could not be opened ({e}). Check that you can write to it."
                ),
            )
        })?;
        Ok(folder)
    }

    // --- Export -------------------------------------------------------------------------------

    /// `export_estimate`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn export_estimate(&self, job: &ExportJob) -> Result<ExportEstimate, AppError> {
        let job: CoreExportJob = convert(job)?;
        convert(&self.core.export_estimate(&job))
    }

    /// `export_start`: snapshots the photos and queues the job; progress arrives as `export_progress`.
    ///
    /// The session need not be done developing. A burst's shared exposure and the pooled noise
    /// model depend on other photos, so the core makes a photo's turn wait until the photos it
    /// depends on have their estimates (not for thumbnails, prefetching or viewport frames) and
    /// then reads them, and the file is the one `ym render` writes for the same photos (`docs/01`
    /// Session, "Estimates the photos share").
    ///
    /// # Errors
    ///
    /// [`AppError`] with a plain sentence: no photos in the scope, photos of the scope that have
    /// no estimates yet ("Some photos are still being developed…"), a DNG of a look with its own
    /// display rendering, a display LUT that is gone.
    pub fn export_start(&self, job: &ExportJob) -> Result<ExportJobId, AppError> {
        let job: CoreExportJob = convert(job)?;
        // Held while the job is queued: a restart that has begun (and found no export) is not
        // followed by one that would be cut off by it.
        let lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if *lifecycle == Lifecycle::Restarting {
            return Err(AppError::new(
                ErrorScope::Export,
                "Yellowmoon is restarting to finish an update. Export again once it has opened.",
            ));
        }
        let id = self.core.export_start(&job).map_err(AppError::from)?;
        Ok(ExportJobId(u32::try_from(id).unwrap_or(u32::MAX)))
    }

    /// Begins a restart into an installed update (`docs/12`), if no export is queued or running.
    ///
    /// The restart the Tauri way (`AppHandle::restart`) cannot be vetoed: it asks the run loop to
    /// exit with `RESTART_EXIT_CODE`, and `ExitRequestApi::prevent_exit` is ignored for that code,
    /// so the question "An export is still running…" never gets its chance and the exit cancels the
    /// export and deletes its partial file. The webview used to call the process plugin's
    /// `relaunch` directly, guarded only by what it knew of the queue. The restart is now a
    /// command of the app (`restart_app`) that asks here first; nothing else can restart the app
    /// (`process:allow-restart` is not in the capabilities). Once this returns `Ok`, `export_start`
    /// refuses, so no export is queued between the check and the exit.
    ///
    /// # Errors
    ///
    /// [`AppError`] with "An export is still running…" while the queue is busy.
    pub fn begin_restart(&self) -> Result<(), AppError> {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if self.core.export_busy() {
            // Scope `export`: the webview tells this refusal (the update stays "ready") from a
            // restart that failed.
            return Err(AppError::new(
                ErrorScope::Export,
                "An export is still running. Restart after it finishes.",
            ));
        }
        *lifecycle = Lifecycle::Restarting;
        Ok(())
    }

    /// `export_cancel`.
    pub fn export_cancel(&self, id: ExportJobId) {
        self.core.export_cancel(u64::from(id.0));
    }

    /// `cli_preview`: the `ym render …` command for a job.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn cli_preview(&self, job: &ExportJob) -> Result<String, AppError> {
        let job: CoreExportJob = convert(job)?;
        Ok(self.core.cli_preview(&job))
    }

    /// Whether an export is queued or running: "Update ready" waits while this is true.
    #[must_use]
    pub fn export_busy(&self) -> bool {
        self.core.export_busy()
    }

    /// What "Show in Finder" opens for a job: the file when it wrote exactly one, else its first
    /// destination folder. The frontend names the path itself (`reveal_in_finder`), so only the
    /// tests ask the core.
    #[cfg(test)]
    #[must_use]
    pub fn export_reveal_path(&self, id: ExportJobId) -> Option<PathBuf> {
        self.core.export_reveal_path(u64::from(id.0))
    }

    // --- Preferences, About and the command-line tool -----------------------------------------

    /// `preferences_get`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn preferences_get(&self) -> Result<Preferences, AppError> {
        convert(&self.core.preferences_get())
    }

    /// `preferences_set`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a default look that isn't in the library, a display LUT that isn't
    /// imported, or a file that cannot be saved; nothing changes then.
    pub fn preferences_set(&self, patch: &PreferencesPatch) -> Result<Preferences, AppError> {
        let patch: ym_core::preferences::PreferencesPatch = convert(patch)?;
        let next = self.core.preferences_set(&patch).map_err(AppError::from)?;
        // The display rendering may have changed what the viewport shows.
        self.viewport.invalidate();
        convert(&next)
    }

    /// `install_cli_link`.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the tool is not in the app, a different file is in the way, or nothing can
    /// be written.
    pub fn install_cli_link(&self) -> Result<CliLinkResult, AppError> {
        let cli = self.cli.as_ref().ok_or_else(|| {
            AppError::new(
                ErrorScope::App,
                "Yellowmoon could not find the command-line tool. Reinstall the app.",
            )
        })?;
        let path = cli.install(&*self.elevate)?;
        Ok(CliLinkResult {
            path: path.to_string_lossy().into_owned(),
        })
    }

    /// `cli_link_status`.
    #[must_use]
    pub fn cli_link_status(&self) -> CliLinkStatus {
        self.cli.as_ref().map_or(
            CliLinkStatus {
                installed: false,
                path: None,
            },
            CliLink::status,
        )
    }

    /// The `ym` that ships with the app, for `ym doctor`.
    #[must_use]
    pub fn cli_source(&self) -> Option<&Path> {
        self.cli.as_ref().map(|c| c.source.as_path())
    }

    /// `app_info`.
    #[must_use]
    pub fn app_info(&self) -> AppInfo {
        AppInfo {
            version: ym_core::version::app_version().to_owned(),
            pipeline_version: ym_core::version::PIPELINE_VERSION.to_owned(),
        }
    }

    /// The backend the viewport draws with.
    #[must_use]
    pub fn backend(&self) -> ViewportBackendKind {
        self.viewport.kind()
    }
}

/// Starts the GPU context for the colour pass, if there is a usable adapter. A software adapter
/// (Mesa lavapipe) is not used unless `YELLOWMOON_GPU=on` asks for it: the CPU path is faster than
/// its first pass and is the reference.
fn start_gpu() -> Option<Arc<ym_gpu::device::Gpu>> {
    let config = ym_gpu::device::GpuConfig {
        pipeline_cache_dir: ym_gpu::device::default_pipeline_cache_dir(),
        ..ym_gpu::device::GpuConfig::default()
    };
    match ym_gpu::device::Gpu::start(config) {
        Ok(gpu) => {
            let forced = std::env::var("YELLOWMOON_GPU").is_ok_and(|v| v == "on" || v == "1");
            if gpu.info().is_software && !forced {
                tracing::info!(adapter = %gpu.info().name, "software GPU only: using the CPU for the colour pass");
                return None;
            }
            tracing::info!(adapter = %gpu.info().name, backend = %gpu.info().backend, "GPU ready");
            Some(Arc::new(gpu))
        }
        Err(e) => {
            tracing::info!(error = %e, "no usable GPU: using the CPU for the colour pass");
            None
        }
    }
}

#[cfg(test)]
mod tests;
