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
use ym_core::auto_provider::ProviderAdvisors;
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
use crate::node_thumbs::NodeThumbnailer;
use crate::thumbs::Thumbnails;
use crate::viewport::{FrameSink, FrameStore, Viewport, core_id};
use crate::wire::{
    AppError, AppInfo, ApplyScope, AutoStatus, ChromaSample, ChromaScatter, ChromaTap,
    CliLinkResult, CliLinkStatus, CropAspect, CropSuggestion, DepthFar, DepthState, ErrorScope,
    ExportEstimate, ExportJob, ExportJobId, FaceBox, FacePhase, FaceState, FocusPlaneResult,
    FocusReason, HistogramData, LensInfo, Look, LookId, LookInput, LooksImportProgress,
    LooksImportResult, MapFrom, MaskFindPrompt, MaskFindResult, MaskKind, MaskView, NodeId, Num,
    ObjectState, OpenProgress, OpenSummary, Orientation, PasteReport, PhotoId, PhotoProgress,
    PixelSample, Preferences, PreferencesPatch, ResolvedSettings, SessionChange, SessionState,
    SettingsPatch, SourceRecognition, ThumbnailReady, ViewportInfo, ViewportInput, ViewportMode,
    ViewportRect, VisionState, WatchStatus, convert,
};
use crate::wizard::{Downloads, recognise};

/// The wire form of the core's [`FaceState`](ym_core::pipeline::FaceState) for a photo: the phase, the numbered faces and the sentence.
fn face_state_of(photo: PhotoId, state: &ym_core::pipeline::FaceState) -> FaceState {
    FaceState {
        photo_id: photo,
        phase: match state.phase {
            ym_core::pipeline::FacePhase::Finding => FacePhase::Finding,
            ym_core::pipeline::FacePhase::Ready => FacePhase::Ready,
            ym_core::pipeline::FacePhase::Failed => FacePhase::Failed,
            ym_core::pipeline::FacePhase::ModelsMissing => FacePhase::ModelsMissing,
        },
        faces: state
            .faces
            .iter()
            .map(|f| FaceBox {
                number: f.number,
                x: Num(f.x),
                y: Num(f.y),
                width: Num(f.width),
                height: Num(f.height),
            })
            .collect(),
        message: state.message.clone(),
    }
}

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
    /// What looks at each photo for Auto (`docs/19` sections 11 and 12, `docs/22` section 12.6): in
    /// the app `ym_apple`'s system advisor (Apple's on-device model where it is available, the fake
    /// with `YELLOWMOON_TEST_APPLE_AUTO=fake`) and the two programs' `ym_agents::system_advisors`
    /// (Claude Code and Antigravity, **not started**: `LAUNCH_PROBE`; the test hook
    /// `YELLOWMOON_TEST_AUTO_PROVIDERS` swaps them for fakes). Empty in the tests unless they give
    /// some, which is the same as every provider unavailable.
    pub advisors: ProviderAdvisors,
    /// What makes the Subject and People masks (`docs/23` sections 3.10 and 4.5): in the app `ym_apple`'s system Vision
    /// (this Mac's Vision framework, or what `YELLOWMOON_TEST_VISION` says); `None` in the tests unless they give one,
    /// which is a computer that cannot make the masks.
    pub vision: Option<Arc<dyn ym_core::vision::VisionProvider>>,
    /// The Depth model (`docs/23` section 4.8): `None` in the app, which finds it in the data folder
    /// (`models/depth`, `ym_core::pipeline::DepthModel::probe`); the tests give a fake one so that Depth rows work without the
    /// weights.
    pub depth: Option<Arc<ym_core::pipeline::DepthModel>>,
}

/// What the app starts when it launches: **no probe**. The programs' advisors are built with
/// `StartProbe::Never`, so launching the app runs neither `claude` nor `agy` (owner, 2026-10-29:
/// "only instantiate CLIs when we need grading and immediately close them after finish"; `docs/22`
/// sections 0 and 3.3). The chosen program is checked when photos are opened or an export starts
/// (`Session::open`, `Core::export_start`), and every program when Preferences opens
/// (`Core::auto_status`).
pub(crate) const LAUNCH_PROBE: ym_agents::StartProbe = ym_agents::StartProbe::Never;

/// The advisors of this computer: Apple Intelligence's, and the two programs' (`docs/22` section 12.6),
/// none of them started ([`LAUNCH_PROBE`]). Apple's availability is an operating-system call, not a
/// program, and is read as before.
fn system_advisors() -> ProviderAdvisors {
    use ym_core::auto_provider::AutoProvider;
    let mut advisors =
        ProviderAdvisors::default().with(AutoProvider::Apple, ym_apple::system_advisor());
    for (provider, advisor) in ym_agents::system_advisors(LAUNCH_PROBE) {
        advisors = advisors.with(provider, advisor);
    }
    advisors
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
            advisors: system_advisors(),
            vision: Some(ym_apple::system_vision()),
            depth: None,
        })
    }
}

/// The command handlers (see the module documentation).
pub struct AppApi {
    core: Arc<Core>,
    emitter: Arc<dyn Emitter>,
    viewport: Viewport,
    thumbs: Thumbnails,
    /// The node thumbnails of the node view (`node_thumbnails`).
    node_thumbs: NodeThumbnailer,
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
    patch.nodes.values().any(|node| {
        node.look.is_some()
            || node.look_intensity.is_some()
            || node.crop.is_some()
            || node.orientation.is_some()
            || node.lens.is_some()
    })
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
        options.advisors = config.advisors.clone();
        options.render.vision = config.vision.clone().map(ym_core::vision::VisionMasks::new);
        options.render.depth_model = config.depth.clone();
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
        {
            // A provider's availability changed (a program found, signed in or out, a probe ended):
            // the webview asks `auto_status()` again (`docs/22` section 12.6). It runs on the probe's
            // thread, so it only sends the event.
            let emitter = Arc::clone(&emitter);
            core.on_auto_changed(move || emitter.emit(AppEvent::AutoStatusChanged));
        }
        {
            // SAM's encoder started or stopped working for a photo: the rows say "Getting ready…" while it does, and a photo
            // that has its features now is drawn again (an Object row selects nothing until then).
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            core.renderer
                .set_object_listener(Arc::new(move |id, phase| {
                    let preparing = phase == ym_core::pipeline::ObjectPhase::Preparing;
                    emitter.emit(AppEvent::ObjectState(ObjectState {
                        photo_id: PhotoId(id.0),
                        preparing,
                    }));
                    if !preparing {
                        shared.estimates_changed(id);
                    }
                }));
        }
        {
            // The worker that makes the Subject and People masks of the photo being looked at (`docs/23` section 4.5): while
            // it works the viewport is busy and finding (the rows say "Finding…", `viewport_busy.finding`); when it ends the
            // photo is drawn again with the masks, and the page learns which kinds found nothing ("Nothing found").
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            core.renderer
                .set_vision_listener(Arc::new(move |id, phase| match phase {
                    ym_core::pipeline::VisionPhase::Started => shared.masks_finding(true),
                    ym_core::pipeline::VisionPhase::Finished { nothing_found } => {
                        shared.masks_finding(false);
                        match convert::<_, Vec<MaskKind>>(&nothing_found) {
                            Ok(nothing_found) => emitter.emit(AppEvent::VisionState(VisionState {
                                photo_id: PhotoId(id.0),
                                nothing_found,
                            })),
                            Err(e) => tracing::error!(error = %e, "vision_state did not convert"),
                        }
                        shared.estimates_changed(id);
                    }
                }));
        }
        {
            // The worker that makes the depth map of the photo being looked at (`docs/23` section 4.8): while it runs the
            // viewport is busy and finding (the Depth rows with the eye on say "Finding…", `viewport_busy.finding`); when it
            // ends the photo is drawn again with the mask, and the page learns whether the map failed (`depth_state`).
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            core.renderer
                .set_depth_listener(Arc::new(move |id, phase| match phase {
                    ym_core::pipeline::DepthPhase::Started => shared.masks_finding(true),
                    ym_core::pipeline::DepthPhase::Finished { failed } => {
                        shared.masks_finding(false);
                        emitter.emit(AppEvent::DepthState(DepthState {
                            photo_id: PhotoId(id.0),
                            failed,
                        }));
                        shared.estimates_changed(id);
                    }
                }));
        }
        {
            // The worker that finds the faces of the photo being looked at (`docs/27` decision 9): while it runs the viewport is busy and
            // finding (`viewport_busy.finding`, the Face Refinement node says "Finding faces…"); the page learns where the faces are
            // (`face_state`) when it starts and when it ends, and the photo is drawn again with the retouch.
            let emitter = Arc::clone(&emitter);
            let shared = Arc::clone(&shared);
            core.renderer
                .set_faces_listener(Arc::new(move |id, event| match event {
                    ym_core::pipeline::FaceEvent::Started => {
                        shared.masks_finding(true);
                        emitter.emit(AppEvent::FaceState(face_state_of(
                            PhotoId(id.0),
                            &ym_core::pipeline::FaceState::finding(),
                        )));
                    }
                    ym_core::pipeline::FaceEvent::Finished(state) => {
                        shared.masks_finding(false);
                        emitter.emit(AppEvent::FaceState(face_state_of(PhotoId(id.0), &state)));
                        shared.estimates_changed(id);
                    }
                }));
        }
        if let Some(masks) = core.renderer.vision_masks() {
            // Ask the computer once, off the thread that opens the app: the first Add mask menu then finds the answer
            // there (`mask_kinds` waits for the same probe otherwise). It runs each Vision request on a 64 x 64 picture.
            let spawned = std::thread::Builder::new()
                .name("yellowmoon-vision-probe".to_owned())
                .spawn(move || {
                    let _ = masks.support();
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "the Vision probe could not start early");
            }
        }
        viewport.attach(&core)?;
        let thumbs = Thumbnails::new(Arc::clone(&emitter), Arc::clone(&frames));
        thumbs.attach(&core).map_err(AppError::internal)?;
        let node_thumbs = NodeThumbnailer::new(Arc::clone(&emitter), Arc::clone(&frames));
        node_thumbs.attach(&core).map_err(AppError::internal)?;
        Ok(Arc::new(Self {
            core,
            emitter,
            viewport,
            thumbs,
            node_thumbs,
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

    /// The maker of node thumbnails, for the tests.
    #[cfg(test)]
    pub(crate) fn node_thumbnailer(&self) -> &NodeThumbnailer {
        &self.node_thumbs
    }

    /// Stops what runs in the background (quitting): exports and renders are cancelled, partial
    /// files are removed and the programs Auto started are killed (`Core::shutdown` calls
    /// `cancel_auto`, `docs/22` section 9.4: a child is not stopped by its parent's death on Unix).
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
    /// [`AppError`] with "No ARW files in what you dropped." when nothing is an ARW file, with
    /// "Yellowmoon can't read the folder {name}. …" when a folder could not be read (`docs/26`
    /// H-02), or when what was given is not there.
    pub fn open_paths(&self, paths: &[String]) -> Result<SessionState, AppError> {
        let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        // Progress goes out as `open_progress` (at most one event per 50 files), so a long read of a
        // folder of thousands shows "Reading {done} of {total} files…" and can be cancelled.
        let emitter = &self.emitter;
        let outcome = self.core.session.open_with_progress(
            &paths,
            OpenOptions::default(),
            &|p: ym_core::session::OpenProgress| {
                emitter.emit(AppEvent::OpenProgress(OpenProgress {
                    done: u32::try_from(p.done).unwrap_or(u32::MAX),
                    total: u32::try_from(p.total).unwrap_or(u32::MAX),
                }));
            },
        );
        // Something that is not there is not "no ARW files": say so first. A folder that could not be
        // read is not "no ARW files" either: `message` names it.
        if outcome.added.is_empty()
            && outcome.already_open == 0
            && outcome.unreadable.is_empty()
            && !outcome.missing.is_empty()
        {
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
            cancelled: outcome.cancelled,
        });
        Ok(state)
    }

    /// `cancel_open`: stops the `open_paths` or `open_dialog` call that is reading files. The files
    /// read so far stay in the session; the others are dropped (`docs/26` P3-3).
    pub fn cancel_open(&self) {
        self.core.session.cancel_open();
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
        self.node_thumbs.clear();
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

    /// `thumbnails_in_view(ids)` (`docs/26` H-11): the page shows these photos (the filmstrip draws
    /// only the ones in view). Answers with the address of the thumbnail the shell holds for each, and
    /// counts them as used. A photo whose thumbnail the shell let go of to keep its budget
    /// ([`crate::viewport::THUMBNAIL_BUDGET_BYTES`]) gets a new one at the thumbnails' priority, announced
    /// with `thumbnail_ready` like any other.
    #[must_use]
    pub fn thumbnails_in_view(&self, ids: &[PhotoId]) -> Vec<ThumbnailReady> {
        let ids: Vec<u32> = ids.iter().map(|id| id.0).collect();
        let (held, lost) = self.frames.thumbnails_in_view(&ids);
        if !lost.is_empty() {
            self.thumbs.refresh(lost.into_iter().map(CorePhotoId));
        }
        held.into_iter()
            .map(|(id, url)| ThumbnailReady {
                id: PhotoId(id),
                url,
            })
            .collect()
    }

    /// `node_thumbnails` (`docs/23` sections 5.7 and 9): asks for the thumbnails of every node of a
    /// photo. It answers at once; the pictures that changed since the webview was told come as a
    /// `node_thumbnails_ready` event, and are served on the `ym-frame` scheme. A photo that cannot be
    /// drawn (not developed yet, skipped, gone) gets none.
    pub fn node_thumbnails(&self, photo: PhotoId) {
        self.node_thumbs.request(core_id(photo));
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
            self.node_thumbs.forget(&outcome.removed);
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
            // A photo with an Object row is made ready for the person's clicks (`docs/23` section 4.4), and one with a
            // Subject or People row gets its masks made in the background (section 4.5).
            self.core.prepare_masks(id);
        }
        self.frames.set_current_photo(current.map(|id| id.0));
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
        // The core also asks for SAM's features in the background when the photo has an Object row (`docs/23` section 4.4).
        let resolved = self
            .core
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

    /// `histogram(photo_id, at)`: the signal at a node as it is rendered now (`docs/01`): the input of
    /// its curve, or the Input's output for `"input"`; left out (`None`) it is Main's, the photo as
    /// it is rendered now. A node the photo does not have and the Output are refused in words. It is
    /// asked for after every frame, so it runs as a thumbnail-priority job: a viewport frame that is
    /// wanted meanwhile goes first, as it does for the Look tab's tiles.
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown or unfinished photo, a node that cannot be read and a
    /// cancelled render.
    pub fn histogram(&self, id: PhotoId, at: Option<NodeId>) -> Result<HistogramData, AppError> {
        let id = core_id(id);
        let src = PhotoSource::from_session(&self.core.session, id).map_err(session_error)?;
        let histogram = self
            .core
            .renderer
            .spawn_histogram_at(src, at.map(|n| n.0))
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        convert(&histogram)
    }

    /// `chroma_scatter(photo_id, at)`: the photo's colours at the input of a node's colour warp, for the
    /// plane of the Colour warp card (`Renderer::chroma_scatter_at`: the histogram's view, at most 2048
    /// points); left out (`None`) it is Main's. Like the histogram it is a thumbnail-priority job that
    /// a move to another photo cancels. A photo that is still being developed has no points yet (an
    /// empty list, not an error: the plane is drawn without them and asks again after the next
    /// `render_stats`).
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown or failed photo, a node that cannot be read and a cancelled
    /// render.
    pub fn chroma_scatter(
        &self,
        id: PhotoId,
        at: Option<NodeId>,
    ) -> Result<ChromaScatter, AppError> {
        let id = core_id(id);
        let src = match PhotoSource::from_session(&self.core.session, id) {
            Ok(src) => src,
            Err(SessionError::NotReady(_)) => return Ok(ChromaScatter { points: Vec::new() }),
            Err(e) => return Err(session_error(e)),
        };
        let scatter = self
            .core
            .renderer
            .spawn_chroma_scatter_at(src, at.map(|n| n.0))
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        convert(&scatter)
    }

    /// `suggest_crop`: Auto crop's proposal for a photo at an aspect and orientation
    /// (`docs/20-auto-crop.md` §3). The first call for a photo renders its picture and analyses it
    /// on the CPU; later calls only make the frames.
    ///
    /// # Errors
    ///
    /// [`AppError`] for an unknown, failed or developing photo, and when the picture cannot be
    /// made.
    pub fn suggest_crop(
        &self,
        id: PhotoId,
        aspect: CropAspect,
        orientation: Orientation,
    ) -> Result<CropSuggestion, AppError> {
        let suggestion = self
            .core
            .session
            .suggest_crop(core_id(id), convert(&aspect)?, convert(&orientation)?)
            .map_err(session_error)?;
        convert(&suggestion)
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
        let patch = ym_core::settings::SettingsPatch::input(ym_core::settings::NodePatch {
            white_pick: Some(Some(point)),
            ..ym_core::settings::NodePatch::default()
        });
        self.core
            .session
            .update_settings(photo, &patch)
            .map_err(session_error)?;
        self.settings_changed(photo, false);
        self.emitter
            .emit(AppEvent::SettingsChanged(PhotoId(photo.0)));
        Ok(())
    }

    /// `viewport_set_mode(mode, mask_view)`.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn viewport_set_mode(
        &self,
        mode: &ViewportMode,
        mask_view: Option<&MaskView>,
    ) -> Result<(), AppError> {
        self.viewport.set_mode(mode, mask_view)
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

    /// `sample_chroma(x, y, at, tap)`: the chroma under a point of the cell, in CSS pixels from its
    /// corner as `sample_pixel` and Pick white take it (`Renderer::sample_chroma_at`: the pixel the
    /// readout reads, through the chain to the stage `tap` names, in the final picture also while the
    /// Before view is shown). `tap` left out (`None`) or `warp_input` reads the input of the node's
    /// colour warp, after its trim and grade (Pick colour of the Colour warp); `node_input` reads the
    /// picture the node takes in, before its trim and grade (Pick colour of a Colour range). `at`
    /// left out (`None`) is Main's. `None` (JSON `null`) on the paper around the photo, for a point
    /// that is not a number and when no photo is shown.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that cannot be read or rendered, and a node that cannot be read.
    pub fn sample_chroma(
        &self,
        x: f64,
        y: f64,
        at: Option<NodeId>,
        tap: Option<ChromaTap>,
    ) -> Result<Option<ChromaSample>, AppError> {
        let (photo, state) = self.viewport.snapshot_synced();
        let Some(photo) = photo else { return Ok(None) };
        let src = PhotoSource::from_session(&self.core.session, photo).map_err(session_error)?;
        let tap = match tap {
            None | Some(ChromaTap::WarpInput) => ym_core::pipeline::ChromaTap::WarpInput,
            Some(ChromaTap::NodeInput) => ym_core::pipeline::ChromaTap::NodeInput,
        };
        let sample = self
            .core
            .renderer
            .sample_chroma_at(
                &src,
                at.map(|n| n.0),
                tap,
                &state,
                (x, y),
                &CancelToken::new(),
            )
            .map_err(AppError::from)?;
        sample.map(|s| convert(&s)).transpose()
    }

    /// `viewport_map_points(points, from)` (`docs/23` sections 3.6 and 9): maps points between the photo frame
    /// and the cell, for the handles and outlines the page draws over the photo (`Renderer::map_points`).
    /// With `photo` the points are `[x, y]` in the photo frame and come back as CSS pixels from the cell's
    /// corner (they may lie outside the cell); with `cell` they are CSS pixels and come back as photo-frame
    /// fractions, `null` for a point on the paper. The answer has one entry per point, in order; a point that
    /// is not a number, and every point while no photo is shown, is `null`.
    ///
    /// # Errors
    ///
    /// [`AppError`] for a photo that cannot be read or prepared.
    pub fn viewport_map_points(
        &self,
        points: &[[Num; 2]],
        from: MapFrom,
    ) -> Result<Vec<Option<[Num; 2]>>, AppError> {
        let (photo, state) = self.viewport.snapshot_synced();
        let Some(photo) = photo else {
            return Ok(vec![None; points.len()]);
        };
        let src = PhotoSource::from_session(&self.core.session, photo).map_err(session_error)?;
        let flat: Vec<[f64; 2]> = points.iter().map(|[x, y]| [x.0, y.0]).collect();
        let from = match from {
            MapFrom::Photo => ym_core::pipeline::MapFrom::Photo,
            MapFrom::Cell => ym_core::pipeline::MapFrom::Cell,
        };
        let mapped = self
            .core
            .renderer
            .map_points(&src, &state, from, &flat, &CancelToken::new())
            .map_err(AppError::from)?;
        Ok(mapped
            .into_iter()
            .map(|p| p.map(|[x, y]| [Num(x), Num(y)]))
            .collect())
    }

    /// `mask_kinds()` (`docs/23` section 9): the kinds of mask this machine can make now, in the order of the
    /// Add mask menu (`ym_core::settings::mask_kinds`). The menu lists exactly these.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn mask_kinds(&self) -> Result<Vec<MaskKind>, AppError> {
        convert(&self.core.mask_kinds())
    }

    /// `mask_find(photo_id, node, mask, prompt)` (`docs/23` section 9): what the Object row `mask` of `node` becomes with the
    /// person's click or box, or that nothing was found there. Nothing is changed: the page sends a found answer as one row
    /// patch. It waits for the encoder when it is still running (`Core::mask_find`), so it runs off the command thread.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the scope `mask` for an Object model that is missing, a mask that is not an Object row and a row
    /// that has all its points; with the scope of the photo's errors for a photo that cannot be read.
    pub fn mask_find(
        &self,
        photo: PhotoId,
        node: u32,
        mask: u32,
        prompt: MaskFindPrompt,
    ) -> Result<MaskFindResult, AppError> {
        let prompt = match prompt {
            MaskFindPrompt::Point { x, y, positive } => ym_core::pipeline::ObjectPrompt::Point {
                x: x.0,
                y: y.0,
                positive,
            },
            MaskFindPrompt::Box { x0, y0, x1, y1 } => ym_core::pipeline::ObjectPrompt::Box {
                x0: x0.0,
                y0: y0.0,
                x1: x1.0,
                y1: y1.0,
            },
        };
        let found = self
            .core
            .mask_find(
                core_id(photo),
                node,
                mask as usize,
                prompt,
                &CancelToken::new(),
            )
            .map_err(AppError::from)?;
        Ok(match found {
            ym_core::pipeline::MaskFound::Found {
                points,
                box_,
                took_ms,
            } => MaskFindResult::Found {
                points: points
                    .into_iter()
                    .map(|(x, y, positive)| (Num(x), Num(y), positive))
                    .collect(),
                box_: box_.map(|b| b.map(Num)),
                took_ms: Num(took_ms),
            },
            ym_core::pipeline::MaskFound::NothingFound(_) => MaskFindResult::Nothing,
        })
    }

    /// `sample_depth(photo_id, x, y)` (`docs/23` section 9, Pick depth): the far limit for the depth at the photo-frame point
    /// (`x`, `y` are shares of the photo's width and height), the depth plus 0.1 and at most 1, or `None` when no depth
    /// map can be made. Nothing is changed: the page writes the answer as one row patch. It waits for the depth worker when it
    /// is still running (`Core::sample_depth`), so it runs off the command thread.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the scope `mask` for a missing Depth model and a point that is not a number; with the scope of the
    /// photo's errors for a photo that cannot be read.
    pub fn sample_depth(
        &self,
        photo: PhotoId,
        x: Num,
        y: Num,
    ) -> Result<Option<DepthFar>, AppError> {
        let far = self
            .core
            .sample_depth(core_id(photo), x.0, y.0, &CancelToken::new())
            .map_err(AppError::from)?;
        Ok(far.map(|far| DepthFar { far: Num(far) }))
    }

    /// `focus_plane(photo_id)` (`docs/23` section 9, Focus plane): the far limit for the plane the camera focused on, from
    /// the maker note's focus point and the median depth in its box; without a limit, why (`no_focus_point` is answered at once,
    /// without waiting for the depth map). Nothing is changed: the page writes the limit as one row patch.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the scope `mask` for a missing Depth model; with the scope of the photo's errors for a photo that
    /// cannot be read.
    pub fn focus_plane(&self, photo: PhotoId) -> Result<FocusPlaneResult, AppError> {
        let answer = self
            .core
            .focus_plane(core_id(photo), &CancelToken::new())
            .map_err(AppError::from)?;
        Ok(match answer {
            ym_core::pipeline::FocusPlane::Far { far, exact_centre } => FocusPlaneResult {
                far: Some(Num(far)),
                exact_centre,
                reason: None,
            },
            ym_core::pipeline::FocusPlane::None(why) => FocusPlaneResult {
                far: None,
                exact_centre: false,
                reason: Some(match why {
                    ym_core::pipeline::FocusReason::NoFocusPoint => FocusReason::NoFocusPoint,
                    ym_core::pipeline::FocusReason::MapFailed => FocusReason::MapFailed,
                }),
            },
        })
    }

    /// `retry_depth(photo_id)` (`docs/23` section 4.8, "Try again"): starts the depth map of a photo again after it failed.
    /// It returns at once: the worker runs in the background, the viewport is busy and finding meanwhile, and `depth_state`
    /// says how it ended (`failed` null when the map is made, the sentence when it failed again). Nothing is changed in the
    /// photo's settings. Nothing happens for a photo whose map did not fail, one that is not the photo being looked at and a
    /// copy of Yellowmoon with no Depth model.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the scope of the photo's errors for a photo that is unknown or not developed yet.
    pub fn retry_depth(&self, photo: PhotoId) -> Result<(), AppError> {
        self.core
            .retry_depth(core_id(photo))
            .map_err(AppError::from)
    }

    /// `face_state(photo_id)` (`docs/27` section 10, S-11b): where the faces of the photo are, for the panel of a Face Refinement
    /// node: `finding`, `ready` with the numbered faces and their boxes in the oriented photo frame (none is a valid answer),
    /// `failed` with the sentence, or `models_missing`. It changes nothing and starts nothing (the core finds the faces in the
    /// background when a photo with the node is the one looked at); `face_state` events say when it changes.
    ///
    /// # Errors
    ///
    /// [`AppError`] with the scope of the photo's errors for a photo that is unknown or not developed yet.
    pub fn face_state(&self, photo: PhotoId) -> Result<FaceState, AppError> {
        let state = self
            .core
            .face_state(core_id(photo))
            .map_err(AppError::from)?;
        Ok(face_state_of(photo, &state))
    }

    /// `face_crop`: the picture of face `number` (1-based) for the Faces card. The PNG stays here and is served on the `ym-frame`
    /// scheme like a look tile (the same store, its budget and its photo-by-photo forgetting); the answer is its address. It runs
    /// at the thumbnails' priority and moving to another photo cancels it.
    ///
    /// # Errors
    ///
    /// [`AppError`] with scope `render` and a plain sentence before the faces are known and for a number the photo does not have.
    pub fn face_crop(&self, photo: PhotoId, number: u32) -> Result<String, AppError> {
        let src =
            PhotoSource::from_session(&self.core.session, core_id(photo)).map_err(session_error)?;
        let png = self
            .core
            .renderer
            .spawn_face_crop(src, number)
            .wait()
            .map_err(job_error)?
            .map_err(AppError::from)?;
        self.core
            .session
            .view(core_id(photo))
            .map_err(session_error)?;
        // A look tile's store keeps it: the key names the face, so a picture of the same face replaces the older.
        Ok(self
            .frames
            .put_look_tile(photo.0, &format!("face:{number}"), png))
    }

    // --- Lens ---------------------------------------------------------------------------------

    /// `lens_list`: the lenses of the Lens card's menu in menu order (`docs/21` sections 1 and 10);
    /// empty when the lens data could not be loaded.
    ///
    /// # Errors
    ///
    /// [`AppError`] only if the shapes disagree (a bug).
    pub fn lens_list(&self) -> Result<Vec<LensInfo>, AppError> {
        convert(&self.core.lens_list())
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

    /// `film_catalogue`: the stocks and papers the Film node's pickers list (`names.json` through the core).
    ///
    /// # Errors
    ///
    /// [`AppError`] with scope `render` and "Yellowmoon's film data is missing. Reinstall the app." when the list cannot be read.
    pub fn film_catalogue(&self) -> Result<crate::wire::FilmCatalogue, AppError> {
        let c = ym_core::film_tiles::film_catalogue().ok_or_else(|| {
            AppError::new(
                ErrorScope::Render,
                "Yellowmoon's film data is missing. Reinstall the app.",
            )
        })?;
        Ok(crate::wire::FilmCatalogue {
            stocks: c
                .stocks
                .into_iter()
                .map(|s| crate::wire::FilmStockInfo {
                    id: s.id,
                    name: s.name,
                    kind: s.kind.to_owned(),
                    group: s.group,
                    iso: s.iso,
                    matching_paper: s.matching_paper,
                })
                .collect(),
            papers: c
                .papers
                .into_iter()
                .map(|p| crate::wire::FilmPaperInfo {
                    id: p.id,
                    name: p.name,
                })
                .collect(),
            profile_version: c.profile_version,
            credit: c.credit,
            disclaimer: c.disclaimer,
            trademarks: c.trademarks,
        })
    }

    /// `film_tile`: the photo as it is developed now with the stock of Film node `node` replaced by `stock`. The PNG stays here
    /// and is served on the `ym-frame` scheme like a look tile (the same store, its budget and its photo-by-photo forgetting); the
    /// answer is its address. It runs at the thumbnails' priority and moving to another photo cancels it.
    ///
    /// # Errors
    ///
    /// [`AppError`] with scope `render` for a node the photo does not have, a stock that is not offered, or a cancelled render.
    pub fn film_tile(&self, photo: PhotoId, node: u32, stock: &str) -> Result<String, AppError> {
        let src =
            PhotoSource::from_session(&self.core.session, core_id(photo)).map_err(session_error)?;
        let png =
            ym_core::film_tiles::spawn_film_tile(&self.core.renderer, src, node, stock.to_owned())
                .wait()
                .map_err(job_error)?
                .map_err(AppError::from)?;
        self.core
            .session
            .view(core_id(photo))
            .map_err(session_error)?;
        // A look tile's store keeps it: the key names the node and the stock, so a tile for the same pair replaces the older.
        Ok(self
            .frames
            .put_look_tile(photo.0, &format!("film:{node}:{stock}"), png))
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
    /// no estimates yet ("Some photos are still being developed…").
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
    /// [`AppError`] for a default look that isn't in the library or a file that cannot be saved;
    /// nothing changes then.
    pub fn preferences_set(&self, patch: &PreferencesPatch) -> Result<Preferences, AppError> {
        let patch: ym_core::preferences::PreferencesPatch = convert(patch)?;
        let next = self.core.preferences_set(&patch).map_err(AppError::from)?;
        convert(&next)
    }

    /// `auto_status`: the four providers and which one grades now (`docs/22` sections 11 and 12.6).
    /// Every advisor is asked to read its availability again, in the background, and the answer is
    /// what they last said: nothing here waits for a program. Only Preferences opening asks for it
    /// (the person looking at the providers is the request: `agy`'s sign-in check, which contacts
    /// its service, runs then and never otherwise, `docs/22` section 3.3 and D7), and the row is
    /// shown only while `visible` or while the saved choice is a program.
    ///
    /// # Errors
    ///
    /// [`AppError::internal`] when the answer does not cross the bridge (a bug).
    pub fn auto_status(&self) -> Result<AutoStatus, AppError> {
        convert(&self.core.auto_status())
    }

    /// `auto_status_cached`: the same status as each advisor last said it, and no advisor is asked
    /// to probe (`docs/22` section 12.6). The webview reads this after it changes the choice, the
    /// consent or the instructions and on `auto_status_changed`, so typing instructions never makes
    /// a program contact its service.
    ///
    /// # Errors
    ///
    /// [`AppError::internal`] when the answer does not cross the bridge (a bug).
    pub fn auto_status_cached(&self) -> Result<AutoStatus, AppError> {
        convert(&self.core.auto_status_cached())
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
            os: std::env::consts::OS.to_owned(),
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
    // GPU pass times (timestamp queries, where the adapter has them) when `YM_PERF` is on: the `gpu passes`
    // line of `ym::perf`. Off, nothing is timed.
    let slot = GpuOpener::new(gpu_policy(), budget)
        .with_pipeline_cache_dir(ym_gpu::device::default_pipeline_cache_dir())
        .with_timestamps(crate::perf::enabled())
        .open();
    // `require` is strict, and a window has no exit code to say so with: the application goes on
    // with the CPU, loudly, rather than quietly looking as if the device were in use.
    if let Some(reason) = slot.failure() {
        tracing::warn!(
            "YELLOWMOON_GPU=require, but no graphics device opened ({reason}); using the CPU"
        );
    }
    // Every kernel compiles in the background now, one job each at `Export` priority (a frame outranks
    // them), so that no viewer frame waits for a shader compile on first use (the eye of a mask row cost
    // 210 ms on Metal for the tint kernel). The jobs run on; the handles are not needed.
    drop(slot.warm_up_for_viewer());
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
