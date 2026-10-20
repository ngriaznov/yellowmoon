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

use ym_core::app::gpu::{GpuOpener, GpuPolicy, GpuSlot};
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::AutoAdvisor;
use ym_core::export_queue::{ExportEvent, ExportJob as CoreExportJob, ExportSink};
use ym_core::pipeline::PhotoSource;
use ym_core::session::{
    EventSink, OpenOptions, PhotoId as CorePhotoId, SessionError, SessionEvent,
};
use ym_gpu::device::Gpu;
use ym_gpu::queue::CancelToken;
use ym_looks::import::Stage;
use ym_looks::import::watch::WatchConfig;
use ym_looks::library::LookId as CoreLookId;

use crate::cli_link::{CliLink, Elevate};
use crate::events::{AppEvent, Emitter};
use crate::thumbs::Thumbnails;
use crate::viewport::{FrameSink, FrameStore, Viewport, core_id};
use crate::wire::{
    AppError, AppInfo, AppleAutoStatus, ApplyScope, ChromaSample, ChromaScatter, CliLinkResult,
    CliLinkStatus, DisplayLut, ErrorScope, ExportEstimate, ExportJob, ExportJobId, HistogramData,
    Look, LookId, LookInput, LooksImportProgress, LooksImportResult, OpenSummary, PasteReport,
    PhotoId, PhotoProgress, PixelSample, Preferences, PreferencesPatch, ResolvedSettings,
    SessionChange, SessionState, SettingsPatch, SourceRecognition, ThumbnailReady, ViewportInfo,
    ViewportInput, ViewportMode, ViewportRect, WatchStatus, convert,
};
use crate::wizard::{Downloads, recognise};

/// The paper of the interface as 8-bit sRGB codes: `--paper` of the dark theme in `src/styles/tokens.css` (#0f0f10). It is the
/// letterbox the core paints around a fitted or cropped photo in viewport frames, thumbnails and look tiles, so that a photo that does
/// not fill its cell sits on the colour of the page and of the window behind it (`native.rs`, `PAPER_SRGB`). Only views have a
/// letterbox; an export has none, so no exported file depends on it. The command line keeps the core's default (white), the page its
/// contact sheets are laid on. `src/styles/native-viewport.node.test.ts` and a test of this module hold the colour to `--paper`.
pub(crate) const PAPER_RGB8: [u8; 3] = [0x0f, 0x0f, 0x10];

/// How the application is started: where its data and its files are.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// The shipped `data/` folder.
    pub data_root: PathBuf,
    /// The app folder (`preferences.json` and `looks/`).
    pub app_root: PathBuf,
    /// The home folder, for the export fallback and the Downloads watcher; `None`: `$HOME`.
    pub home: Option<PathBuf>,
    /// Open the graphics device as `YELLOWMOON_GPU` says (`auto`, the default: a hardware adapter
    /// only and the stages signed off; `on`, `off`, `require`; [`GpuPolicy`]). `false`: never open
    /// one, whatever the variable says (the tests).
    pub gpu: bool,
    /// Render a thumbnail for every photo that finishes developing.
    pub thumbnails: bool,
    /// Threads of the decode pool; `None`: the core's default.
    pub decode_threads: Option<usize>,
    /// The Downloads watcher's settings.
    pub watch: WatchConfig,
    /// What looks at each photo for Apple Auto (`docs/19` sections 11 and 12): `ym_apple`'s system
    /// advisor in the app (Apple's on-device model where it is available, the fake with
    /// `YELLOWMOON_TEST_APPLE_AUTO=fake`); `None` in the tests unless they give one, which is the
    /// same as unavailable.
    pub advisor: Option<Advisor>,
}

/// An Apple Auto advisor for [`AppConfig`].
#[derive(Clone)]
pub struct Advisor(pub Arc<dyn AutoAdvisor>);

impl std::fmt::Debug for Advisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Advisor")
    }
}

impl AppConfig {
    /// The folders of this machine, as the app and the CLI find them. The GPU follows
    /// `YELLOWMOON_GPU` (`auto` by default, on any system; `off` turns it off), the policy `ym`
    /// reads too.
    ///
    /// # Errors
    ///
    /// [`AppError`] with "Yellowmoon's data folder is missing. Reinstall the app." when there is no
    /// data folder.
    pub fn from_environment() -> Result<Self, AppError> {
        let options = CoreOptions::from_environment().map_err(AppError::from)?;
        let gpu = GpuPolicy::from_environment(None).policy != GpuPolicy::Off;
        Ok(Self {
            data_root: options.data_root,
            app_root: options.app_root,
            home: None,
            gpu,
            thumbnails: true,
            decode_threads: None,
            watch: WatchConfig::default(),
            advisor: Some(Advisor(ym_apple::system_advisor())),
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
    /// The graphics device is opened first, and `make_sink` makes the viewport's sink with it (the
    /// device the core renders with, `None` when the core uses the CPU): the native viewport draws
    /// with that device, so a frame the core makes there is drawn from there
    /// (`docs/01` §Native viewport).
    ///
    /// # Errors
    ///
    /// [`AppError`] when the data folder cannot be read or a thread cannot start.
    pub fn start(
        config: AppConfig,
        emitter: Arc<dyn Emitter>,
        make_sink: impl FnOnce(Option<&Arc<Gpu>>) -> Arc<dyn FrameSink>,
        frames: Arc<FrameStore>,
        elevate: Box<dyn Elevate + Send + Sync>,
        cli: Option<CliLink>,
    ) -> Result<Arc<Self>, AppError> {
        let mut options = CoreOptions::new(config.data_root.clone(), config.app_root.clone());
        // The device is opened here and nowhere else (`ym_core::app::gpu`), as `ym` opens it: the
        // same policy, the same adapter, the same budgets. The viewport's sink is made with it.
        let slot = open_app_gpu(config.gpu, options.render.budget);
        let sink = make_sink(slot.gpu());
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
                // A burst's median moved, Apple Auto answered or failed, or its preference changed
                // (`docs/19` section 6): the viewport draws the photo again, and the webview reads its
                // settings again, so the Automatic card and the sliders' detents follow.
                SessionEvent::EstimatesChanged { id } => {
                    shared.estimates_changed(id);
                    emitter.emit(AppEvent::SettingsChanged(PhotoId(id.0)));
                }
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

        options.session_events = session_events;
        options.export_events = export_events;
        options.home = config.home.clone();
        options.thumbnails = config.thumbnails;
        options.advisor = config.advisor.as_ref().map(|a| Arc::clone(&a.0));
        if let Some(threads) = config.decode_threads {
            options.decode_threads = threads;
        }
        slot.apply_to(&mut options.render);
        // The ground of the interface is dark: the letterbox of a viewport frame, a thumbnail and a look tile is that colour, as the
        // window behind the webview is (`native.rs`). Exports have no letterbox, so no file changes.
        options.render.background = PAPER_RGB8;
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
        let mut state = self.session_state()?;
        state.opened = Some(OpenSummary {
            added: u32::try_from(outcome.added.len()).unwrap_or(u32::MAX),
            already_open: u32::try_from(outcome.already_open).unwrap_or(u32::MAX),
        });
        Ok(state)
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
            tracing::warn!(kind = e.kind(), "could not remember the folder");
            tracing::debug!(error = %e, "could not remember the folder: details");
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

    /// `remove_photos`: takes photos out of the filmstrip ([`ym_core::app::Core::remove_photos`]). The files on disk are not
    /// touched. The photo on screen hands over to the next one (the previous one when the last was removed) or, with no photo
    /// left, the viewport shows nothing. The photos' thumbnails are set aside for Undo ([`AppApi::restore_photos`]), the burst
    /// frames that stay get new estimates, a `settings_changed` event and a new thumbnail. An id that is not in the session is
    /// ignored; the answer lists the photos that were removed. Exports that were queued keep going with their snapshots.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes of the core and of the bridge disagree (a bug).
    pub fn remove_photos(&self, ids: &[PhotoId]) -> Result<SessionChange, AppError> {
        let core_ids: Vec<CorePhotoId> = ids.iter().map(|id| core_id(*id)).collect();
        let outcome = self.core.remove_photos(&core_ids);
        if !outcome.removed.is_empty() {
            self.thumbs.forget(&outcome.removed);
            let gone: Vec<u32> = outcome.removed.iter().map(|id| id.0).collect();
            self.frames.set_aside_photos(&gone);
            self.sync_current();
            self.estimates_moved(&outcome.estimates_changed);
        }
        Ok(SessionChange {
            photo_ids: outcome.removed.iter().map(|id| PhotoId(id.0)).collect(),
            already_open: 0,
            session: self.session_state()?,
        })
    }

    /// `restore_photos`: Undo of the last `remove_photos` ([`ym_core::app::Core::restore_photos`]): the photos come back where
    /// they were with their settings, their thumbnails are announced again at once, and the photo on screen is the one that was
    /// on screen when the removal took it away, unless the person has moved on. Answers an empty change when there is nothing
    /// to put back.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes of the core and of the bridge disagree (a bug).
    pub fn restore_photos(&self) -> Result<SessionChange, AppError> {
        let outcome = self.core.restore_photos();
        if !outcome.restored.is_empty() {
            let back: Vec<u32> = outcome.restored.iter().map(|id| id.0).collect();
            for (id, url) in self.frames.bring_back_photos(&back) {
                self.emitter.emit(AppEvent::ThumbnailReady(ThumbnailReady {
                    id: PhotoId(id),
                    url,
                }));
            }
            self.sync_current();
            self.estimates_moved(&outcome.estimates_changed);
        }
        Ok(SessionChange {
            photo_ids: outcome.restored.iter().map(|id| PhotoId(id.0)).collect(),
            already_open: u32::try_from(outcome.already_open).unwrap_or(u32::MAX),
            session: self.session_state()?,
        })
    }

    /// Photos that stay in the session had their shared estimates made again (a burst lost or regained a frame): they get a new
    /// thumbnail. The webview is told to read their settings again, which show the exposure and the Apple Auto values the burst now
    /// shares, by the event sink: the core sent each of them `estimates_changed` before the command returned, which the sink turns
    /// into `settings_changed` for every cause (an Apple Auto answer, a burst's median, a removal).
    fn estimates_moved(&self, ids: &[CorePhotoId]) {
        self.thumbs.refresh(ids.iter().copied());
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

    /// `copy_settings`: the settings of a photo as a patch to paste onto other photos: everything
    /// but the crop, the orientation and the Pick white point, Auto kept as Auto, a white balance
    /// that came from a picked white as the manual Kelvin and tint it resolved to
    /// (`ym_core::settings::SettingsPatch::copied_from`). Nothing is stored here: the webview keeps
    /// the patch for as long as the app runs.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that is unknown, failed or not developed yet.
    pub fn copy_settings(&self, id: PhotoId) -> Result<SettingsPatch, AppError> {
        let patch = self
            .core
            .copy_settings(core_id(id))
            .map_err(AppError::from)?;
        convert(&patch)
    }

    /// `paste_settings`: applies a copied patch to every photo in `ids` in one call under one lock
    /// (`ym_core::session::Session::paste_settings`), so a paste onto five hundred photos is one
    /// command and not a thousand. A photo that is still developing or failed is skipped. The
    /// viewport draws again if the current photo changed, every photo that changed gets a new
    /// thumbnail (queued at thumbnail priority, behind what the viewport needs) and a
    /// `settings_changed` event, and the answer carries each photo's change as an undo step.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a look that is gone from the library (as `apply_look` refuses it) and for a
    /// patch that cannot be applied; nothing changes then.
    pub fn paste_settings(
        &self,
        ids: &[PhotoId],
        patch: &SettingsPatch,
    ) -> Result<PasteReport, AppError> {
        let core_patch: ym_core::settings::SettingsPatch = convert(patch)?;
        let core_ids: Vec<CorePhotoId> = ids.iter().map(|id| core_id(*id)).collect();
        let report = self
            .core
            .paste_settings(&core_ids, &core_patch)
            .map_err(AppError::from)?;
        let changed: Vec<CorePhotoId> = report
            .pasted
            .iter()
            .filter(|p| !(p.before.is_empty() && p.after.is_empty()))
            .map(|p| p.id)
            .collect();
        if let Some(current) = self.core.session.current()
            && changed.contains(&current)
        {
            self.viewport.invalidate();
        }
        self.thumbs.refresh(changed.iter().copied());
        for id in &changed {
            self.emitter.emit(AppEvent::SettingsChanged(PhotoId(id.0)));
        }
        convert(&report)
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
        let histogram = self
            .core
            .renderer
            .spawn_histogram(src)
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        convert(&histogram)
    }

    /// `chroma_scatter`: the photo's colours at the colour warp's input, for the plane of the Colour
    /// warp card (`Renderer::chroma_scatter`: the histogram's view, at most 2048 points). Like the
    /// histogram it is a thumbnail-priority job that a move to another photo cancels. A photo that
    /// is still being developed has no points yet (an empty list, not an error: the plane is drawn
    /// without them and asks again after the next `render_stats`).
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown or failed photo and a cancelled render.
    pub fn chroma_scatter(&self, id: PhotoId) -> Result<ChromaScatter, AppError> {
        let id = core_id(id);
        let src = match PhotoSource::from_session(&self.core.session, id) {
            Ok(src) => src,
            Err(SessionError::NotReady(_)) => return Ok(ChromaScatter { points: Vec::new() }),
            Err(e) => return Err(session_error(e)),
        };
        let scatter = self
            .core
            .renderer
            .spawn_chroma_scatter(src)
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        convert(&scatter)
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

    /// `sample_chroma`: the chroma at the colour warp's input under a point of the cell, in CSS
    /// pixels from its corner as `sample_pixel` and Pick white take it (`Renderer::sample_chroma`:
    /// the pixel the readout reads, through the chain to C9, in the final picture also while the
    /// Before view is shown). `None` (JSON `null`) on the paper around the photo, for a point that is
    /// not a number and when no photo is shown.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that cannot be read or rendered.
    pub fn sample_chroma(&self, x: f64, y: f64) -> Result<Option<ChromaSample>, AppError> {
        let (photo, state) = self.viewport.snapshot_synced();
        let Some(photo) = photo else { return Ok(None) };
        let src = PhotoSource::from_session(&self.core.session, photo).map_err(session_error)?;
        let sample = self
            .core
            .renderer
            .sample_chroma(&src, &state, (x, y), &CancelToken::new())
            .map_err(AppError::from)?;
        sample.map(|s| convert(&s)).transpose()
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
    /// no estimates yet ("Some photos are still being developed…"), a display LUT that is gone.
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

    /// `apple_auto_status`: whether Apple Auto is available on this computer, read again now, and
    /// whether it acts (`docs/19` sections 1 and 2). Preferences asks when it opens and shows its
    /// row only while Apple Auto is available.
    ///
    /// # Errors
    ///
    /// [`AppError::internal`] when the answer does not cross the bridge (a bug).
    pub fn apple_auto_status(&self) -> Result<AppleAutoStatus, AppError> {
        convert(&self.core.apple_auto_status())
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

    /// The backend the viewport draws with. Only the development smoke test (`YM_SMOKE`, compiled
    /// out of release builds) asks for it, so it is compiled out there too; the bridge's
    /// [`AppApi::viewport_info`] is the same answer for the page.
    #[cfg(debug_assertions)]
    #[must_use]
    pub fn backend(&self) -> crate::wire::ViewportBackendKind {
        self.viewport.kind()
    }
}

/// The graphics policy of the application: `YELLOWMOON_GPU`, else `auto` (the application has no
/// flags). A value that is not a policy is logged and read as `auto`.
fn gpu_policy() -> GpuPolicy {
    let choice = GpuPolicy::from_environment(None);
    if let Some(warning) = &choice.warning {
        tracing::warn!("{warning}");
    }
    choice.policy
}

/// Opens the GPU for the colour pass (and the stages to come) as the policy says. A software
/// adapter (Mesa lavapipe) is left alone under `auto`: the CPU path is faster than its first pass
/// and is the reference. `enabled` is false in the tests, which never open a device.
fn open_app_gpu(enabled: bool, budget: ym_core::scheduler::MemoryBudget) -> GpuSlot {
    if !enabled {
        return GpuSlot::off(budget);
    }
    let slot = GpuOpener::new(gpu_policy(), budget)
        .with_pipeline_cache_dir(ym_gpu::device::default_pipeline_cache_dir())
        .open();
    // `require` is strict, and a window has no exit code to say so with: the application goes on
    // with the CPU, loudly, rather than quietly looking as if the device were in use.
    if let Some(reason) = slot.failure() {
        tracing::warn!(
            "YELLOWMOON_GPU=require, but no graphics device opened ({reason}); using the CPU"
        );
    }
    slot
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use ym_core::app::gpu::GpuOutcome;

    #[test]
    fn a_disabled_app_opens_nothing() {
        let slot = open_app_gpu(false, ym_core::scheduler::MemoryBudget::default());
        assert_eq!(slot.policy(), GpuPolicy::Off);
        assert!(slot.gpu().is_none() && slot.adapter().is_none());
        assert!(matches!(slot.outcome(), GpuOutcome::Cpu(_)));
        let mut options = ym_core::pipeline::RenderOptions::default();
        slot.apply_to(&mut options);
        assert!(options.gpu.is_none());
        assert!(options.gpu_stages.is_empty());
    }

    #[test]
    fn the_app_reads_the_policy_the_way_ym_does() {
        // The same function, the same words, the same default (`ym_core::app::gpu` has the table).
        for (env, policy) in [
            (None, GpuPolicy::Auto),
            (Some("auto"), GpuPolicy::Auto),
            (Some("on"), GpuPolicy::On),
            (Some("1"), GpuPolicy::On),
            (Some("off"), GpuPolicy::Off),
            (Some("0"), GpuPolicy::Off),
            (Some("require"), GpuPolicy::Require),
            (Some("typo"), GpuPolicy::Auto),
        ] {
            assert_eq!(GpuPolicy::resolve(None, env).policy, policy, "{env:?}");
        }
    }
}

#[cfg(test)]
mod tests;
