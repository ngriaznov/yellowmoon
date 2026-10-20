//! The in-memory `Session`: the photos opened in this run, the current photo, the picked set and
//! each photo's settings and automatic estimates. Nothing is persisted (`docs/01-architecture.md`
//! section Session; hard rule 4 of `AGENTS.md`).
//!
//! # Opening files
//!
//! [`Session::open`] collects the ARW files in what it is given (folders are walked for `*.ARW`,
//! case-insensitively, one level deep unless [`OpenOptions::recursive`]; other files and `.`-files
//! such as macOS's `._DSC0001.ARW` are ignored), probes them in parallel (`ym_raw::metadata::probe`,
//! well under 20 ms each), registers a photo for each and returns. A file that fails the probe is a
//! photo that is *Skipped* with the plain-words reason of `docs/04` (the UI's CAN'T OPEN). Then one
//! background job per photo runs on the decode pool:
//!
//! ```text
//! decode -> automatic estimates -> thumbnail (the hook of part 2) -> Ready
//! ```
//!
//! The pool has `min(available_parallelism - 2, 6)` threads (at least one) and never touches the
//! GPU. It takes the current photo's job first, so a photo the user is waiting for is not behind
//! 500 others. A failing file never blocks the others: its job ends in *Skipped*, and a panic in a
//! job is caught and reported the same way.
//!
//! # Progress events
//!
//! Each photo emits [`PhotoProgress`] events, in this order per photo (photos interleave):
//!
//! ```text
//! probe(done) | decode(start) decode(done) | estimate(start) estimate(done)
//!             | thumbnail(start) [ThumbnailReady] thumbnail(done)
//! ```
//!
//! `thumbnail(done)` is *Ready*. A failure ends the photo's sequence with one event that carries
//! `error` (its stage, `done = true`): *Skipped*. [`PhotoStatus`] is what `session_state` shows:
//! Waiting, Developing at a stage, Ready or Skipped with the reason. Events go to the
//! [`EventSink`]; they are emitted while an internal lock is held (that is what makes "no event
//! after `close` returns" true), so a sink must not call back into the session. Forward them to a
//! channel ([`EventSink::channel`]) or to the webview.
//!
//! # Memory
//!
//! At most [`MOSAIC_CACHE_LEN`] decoded mosaics are kept: the current photo and its neighbours. A
//! mosaic outside that window is dropped when its job is done, and [`Session::raw_image`] decodes
//! again when something needs one (a render, Pick white, Re-run). Moving the current photo queues
//! the new window for decoding; a queued prefetch is dropped unless its photo is still in the window
//! when it runs. Callers that ask for a photo while it is being decoded wait for that decode and
//! share its image: one decode per photo at a time.
//!
//! # Removing photos
//!
//! [`Session::remove`] takes photos out of the filmstrip (the files on disk are never touched) and [`Session::restore_removed`]
//! puts the last removal back where it was, with the photos' settings, picks and estimates exactly as they were. The queued
//! jobs of removed photos are dropped, a running one stops at its next stage, and no event about a removed photo is sent once
//! `remove` returns (each photo carries an `incarnation` that jobs copy, so the job of a photo that was removed and put back
//! cannot report as the new one). The shared estimates below follow the set of photos that is left.
//!
//! # Shared estimates
//!
//! Two parts of the estimates depend on the other photos: burst frames share the median exposure
//! (`docs/02`, step 6) and photos of the same body, ISO, bit depth and compression pool their
//! noise estimate (`docs/06`). The session keeps each photo's own values and serves the shared ones
//! from [`Session::view`] / [`Session::estimates`]. Both are deterministic functions of the set of
//! photos, never of which job finished first. When a burst's median moves, the photos whose
//! exposure changed get [`SessionEvent::EstimatesChanged`].

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ym_color::consts::Vec3;
use ym_color::exposure::{BurstFrame, burst_groups, share_burst_ev};
use ym_color::look::{LookTransform, ParametricLook};
use ym_looks::library::LookId;
use ym_noise::estimate::PoolKey;
use ym_noise::priors::{NoisePrior, PriorStore};
use ym_profiles::profile::Profile;
use ym_profiles::registry::ProfileStore;
use ym_raw::decode::RawImage;
use ym_raw::error::RawError;
use ym_raw::metadata::RawMetadata;

use crate::auto::{
    AutoError, AutoEstimates, AutoInputs, IlluminantModels, NoisePool, POOL_MAX_FILES, burst_frame,
    meta_image_size, noise_prior, pick_white_sample, pool_key,
};
use crate::camera::{CameraCard, CameraInfo, PhotoCamera};
use crate::data_dir::{DataDirError, find_data_dir};
use crate::settings::{DevelopSettings, ResolvedSettings, SettingsError, SettingsPatch};

/// How many decoded mosaics the session keeps: the current photo, the previous and the next
/// (`docs/01` section Scheduling and threads).
pub const MOSAIC_CACHE_LEN: usize = 3;

/// The message of the Start screen when a drop holds no ARW file (`docs/11` section Start).
pub const NO_ARW_MESSAGE: &str = "No ARW files in what you dropped.";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One run at a time per key. The first caller for a key runs the work (the leader); callers that
/// arrive while it runs wait for its result and share it, instead of repeating the work.
///
/// A leader that fails, or panics, leaves the waiters with nothing: each looks again and may lead
/// a run of its own, so every caller still gets its own error.
struct SingleFlight<K, V> {
    flights: Mutex<HashMap<K, Arc<Flight<V>>>>,
}

/// A run in progress and, once it is over, what came of it.
struct Flight<V> {
    outcome: Mutex<Outcome<V>>,
    done: Condvar,
}

enum Outcome<V> {
    Running,
    Done(V),
    Failed,
}

/// What a caller of [`SingleFlight::join`] is to do.
enum Role<'a, K: std::hash::Hash + Eq + Copy, V: Clone> {
    /// Another caller is running it: [`Flight::wait`] for the result.
    Wait(Arc<Flight<V>>),
    /// This caller runs it and ends it with [`Lead::finish`].
    Lead(Lead<'a, K, V>),
}

/// The leader's end of a run. Dropped without [`Lead::finish`] (an error, a panic), it ends the run
/// as failed, so the waiters never wait for ever.
struct Lead<'a, K: std::hash::Hash + Eq + Copy, V: Clone> {
    owner: &'a SingleFlight<K, V>,
    key: K,
    flight: Arc<Flight<V>>,
    ended: bool,
}

impl<K: std::hash::Hash + Eq + Copy, V: Clone> SingleFlight<K, V> {
    fn new() -> Self {
        Self {
            flights: Mutex::new(HashMap::new()),
        }
    }

    /// Joins the run for `key`, or starts one.
    fn join(&self, key: K) -> Role<'_, K, V> {
        let mut flights = lock(&self.flights);
        if let Some(flight) = flights.get(&key) {
            return Role::Wait(Arc::clone(flight));
        }
        let flight = Arc::new(Flight {
            outcome: Mutex::new(Outcome::Running),
            done: Condvar::new(),
        });
        flights.insert(key, Arc::clone(&flight));
        Role::Lead(Lead {
            owner: self,
            key,
            flight,
            ended: false,
        })
    }
}

impl<V: Clone> Flight<V> {
    /// Waits for the leader; `None` when it failed.
    fn wait(&self) -> Option<V> {
        let mut outcome = lock(&self.outcome);
        loop {
            match &*outcome {
                Outcome::Running => {
                    outcome = self
                        .done
                        .wait(outcome)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                Outcome::Done(value) => return Some(value.clone()),
                Outcome::Failed => return None,
            }
        }
    }
}

impl<K: std::hash::Hash + Eq + Copy, V: Clone> Lead<'_, K, V> {
    /// Ends the run with its result (`None` for a failure) and wakes the waiters.
    fn finish(mut self, value: Option<V>) {
        self.end(value);
    }

    fn end(&mut self, value: Option<V>) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        *lock(&self.flight.outcome) = value.map_or(Outcome::Failed, Outcome::Done);
        self.flight.done.notify_all();
        let mut flights = lock(&self.owner.flights);
        if flights
            .get(&self.key)
            .is_some_and(|f| Arc::ptr_eq(f, &self.flight))
        {
            flights.remove(&self.key);
        }
    }
}

impl<K: std::hash::Hash + Eq + Copy, V: Clone> Drop for Lead<'_, K, V> {
    fn drop(&mut self) {
        self.end(None);
    }
}

// ---------------------------------------------------------------------------------------------
// Identifiers, status and events (the wire shapes of src/api/types.ts)
// ---------------------------------------------------------------------------------------------

/// Identifies a photo for the lifetime of the session only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhotoId(pub u32);

impl std::fmt::Display for PhotoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stages of the background job that develops a freshly opened photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhotoStage {
    /// The metadata probe.
    Probe,
    /// Decoding the raw data.
    Decode,
    /// The automatic estimates.
    Estimate,
    /// The thumbnail render.
    Thumbnail,
}

/// A photo's place in the development pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhotoStatus {
    /// Probed, waiting for a decode thread.
    Waiting,
    /// A stage is running.
    Developing {
        /// The stage in flight.
        stage: PhotoStage,
    },
    /// Developed: estimates and thumbnail are there.
    Ready,
    /// Can't be developed; the plain-words reason.
    Skipped {
        /// What went wrong and what to do next.
        reason: String,
    },
}

/// `photo_progress { id, stage, done, error }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotoProgress {
    /// The photo.
    pub id: PhotoId,
    /// The stage that started or finished.
    pub stage: PhotoStage,
    /// True when the stage has finished. A photo is ready when `thumbnail` is done.
    pub done: bool,
    /// The plain-words reason when the photo can't be developed (it becomes Skipped).
    pub error: Option<String>,
}

/// `thumbnail_ready { id, png_bytes }`: a PNG of at most 240 px on its long edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThumbnailReady {
    /// The photo.
    pub id: PhotoId,
    /// The PNG file.
    pub png_bytes: Vec<u8>,
}

/// What the session tells its listeners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    /// A development stage started or finished.
    PhotoProgress(PhotoProgress),
    /// A thumbnail was rendered.
    ThumbnailReady(ThumbnailReady),
    /// The estimates a photo resolves against changed because other photos finished (the burst
    /// median moved): re-render it and its thumbnail.
    EstimatesChanged {
        /// The photo.
        id: PhotoId,
    },
}

impl SessionEvent {
    /// The name of the event on the bridge (`photo_progress`, `thumbnail_ready`,
    /// `estimates_changed`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::PhotoProgress(_) => "photo_progress",
            Self::ThumbnailReady(_) => "thumbnail_ready",
            Self::EstimatesChanged { .. } => "estimates_changed",
        }
    }
}

/// Where session events go: a callback the app or the CLI provides.
#[derive(Clone)]
pub struct EventSink(Arc<dyn Fn(SessionEvent) + Send + Sync>);

impl EventSink {
    /// A sink that calls `f` for every event. `f` runs while the session holds an internal lock:
    /// it must be quick and must not call back into the session.
    pub fn new(f: impl Fn(SessionEvent) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// A sink that drops every event.
    #[must_use]
    pub fn none() -> Self {
        Self::new(|_| {})
    }

    /// A sink that sends every event into a channel, and the receiving end.
    #[must_use]
    pub fn channel() -> (Self, mpsc::Receiver<SessionEvent>) {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        (
            Self::new(move |e| {
                let _ = lock(&tx).send(e);
            }),
            rx,
        )
    }

    fn send(&self, event: SessionEvent) {
        (self.0)(event);
    }
}

impl std::fmt::Debug for EventSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EventSink")
    }
}

// ---------------------------------------------------------------------------------------------
// Summaries
// ---------------------------------------------------------------------------------------------

/// What the UI needs to know about one photo without asking for its settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhotoSummary {
    /// The photo.
    pub id: PhotoId,
    /// File name with its extension: `DSC04204.ARW`.
    pub file_name: String,
    /// The detected body and the profile used.
    pub camera: CameraInfo,
    /// ISO, 0 when the file does not say.
    pub iso: u32,
    /// Exposure time in seconds (0.008 for 1/125), 0 when unknown.
    pub shutter_s: f64,
    /// F-number, 0 when unknown.
    pub aperture_f: f64,
    /// Capture time as local ISO 8601 without a zone (`2026-10-04T17:42:10`); empty when unknown.
    pub captured_at: String,
    /// Width in pixels after the camera's orientation.
    pub width_px: u32,
    /// Height in pixels after the camera's orientation.
    pub height_px: u32,
    /// Where the photo is in development.
    pub status: PhotoStatus,
    /// Picked for the batch actions.
    pub picked: bool,
    /// Lossless M or S: the camera already processed it, so colour emulation is approximate.
    pub reduced: bool,
}

/// The session as the webview mirrors it (`session_state`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionState {
    /// Name of the folder the photos came from, for the header cell.
    pub folder_name: Option<String>,
    /// The photos in filmstrip order.
    pub photos: Vec<PhotoSummary>,
    /// Index of the current photo; null when the session is empty.
    pub current_index: Option<usize>,
    /// Ids of the picked photos.
    pub picked: Vec<PhotoId>,
    /// Where Export saves by default: `<source folder> export` next to the source folder.
    pub default_export_folder: Option<String>,
}

/// "Apply to" on the Look card: which photos a command acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyScope {
    /// The current photo.
    This,
    /// The picked photos.
    Picked,
    /// Every photo.
    All,
}

/// What [`Session::paste_settings`] did to one photo: the fields it changed, as a pair of patches
/// the caller can keep as one undo step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PastedPhoto {
    /// The photo.
    pub id: PhotoId,
    /// The changed fields as they were (what Undo sends). Empty when the photo already had the
    /// pasted settings.
    pub before: SettingsPatch,
    /// The same fields as they are now (what Redo sends).
    pub after: SettingsPatch,
}

/// What [`Session::paste_settings`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PasteReport {
    /// The photos that took the settings, in the order they were asked for.
    pub pasted: Vec<PastedPhoto>,
    /// The photos left out: still developing, failed or not in the session.
    pub skipped: Vec<PhotoId>,
}

/// What [`Session::remove`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemoveOutcome {
    /// The photos taken out of the session, in filmstrip order. An id that was not in the session is not listed; when
    /// none was, this is empty and nothing changed.
    pub removed: Vec<PhotoId>,
    /// The current photo afterwards: the one it was, or when that was removed the next photo that stays (the previous one
    /// when the last was removed); `None` when no photo is left.
    pub current: Option<PhotoId>,
    /// Photos that stay whose served estimates moved because of the photos that left (the burst median exposure, and for
    /// the current photo the pooled noise). Each one was sent [`SessionEvent::EstimatesChanged`].
    pub estimates_changed: Vec<PhotoId>,
}

/// What [`Session::restore_removed`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestoreOutcome {
    /// The photos put back, in filmstrip order.
    pub restored: Vec<PhotoId>,
    /// Photos of the removal that stayed out because their file was added again meanwhile.
    pub already_open: usize,
    /// The current photo afterwards.
    pub current: Option<PhotoId>,
    /// Photos that were in the session whose served estimates moved because the restored photos joined their burst or
    /// their noise pool. Each one was sent [`SessionEvent::EstimatesChanged`].
    pub estimates_changed: Vec<PhotoId>,
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a session command failed. The text is plain words with a next step where there is one.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// No photo has this id.
    #[error("That photo isn't in the session any more.")]
    UnknownPhoto(PhotoId),
    /// The photo's estimates are not there yet.
    #[error("This photo is still being developed. Try again in a moment.")]
    NotReady(PhotoId),
    /// The photo could not be developed; the text is the reason it was skipped with.
    #[error("{0}")]
    Skipped(String),
    /// A decode failed (when a mosaic had to be read again).
    #[error(transparent)]
    Decode(#[from] RawError),
    /// The automatic estimates failed.
    #[error(transparent)]
    Estimate(#[from] AutoError),
    /// A settings value or patch was refused.
    #[error(transparent)]
    Settings(#[from] SettingsError),
    /// The data folder could not be found or read.
    #[error("{0}")]
    Data(String),
}

impl From<DataDirError> for SessionError {
    fn from(e: DataDirError) -> Self {
        Self::Data(e.to_string())
    }
}

// ---------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------

/// What a rendered thumbnail is made from (part 2 implements [`ThumbnailRenderer`]).
pub struct ThumbnailInput<'a> {
    /// The photo.
    pub id: PhotoId,
    /// The decoded raw data.
    pub raw: &'a Arc<RawImage>,
    /// The camera match.
    pub camera: &'a PhotoCamera,
    /// The profile the photo uses.
    pub profile: &'a Arc<Profile>,
    /// The estimates (burst and pooled values applied).
    pub estimates: &'a Arc<AutoEstimates>,
    /// The photo's settings (the defaults when it is first developed).
    pub settings: &'a DevelopSettings,
    /// True when the job was cancelled (the session closed): stop early.
    pub cancelled: &'a dyn Fn() -> bool,
}

/// The thumbnail hook: renders the quarter-resolution classical preview of a freshly developed
/// photo and returns it as a PNG of at most 240 px on its long edge. Implemented by the render
/// graph (part 2); without one the thumbnail stage is a no-op and a photo is Ready after its
/// estimates.
pub trait ThumbnailRenderer: Send + Sync {
    /// Renders one thumbnail. An `Err` is the plain-words reason the photo is Skipped with.
    ///
    /// # Errors
    ///
    /// Whatever the renderer reports.
    fn render(&self, input: &ThumbnailInput<'_>) -> Result<Vec<u8>, String>;
}

/// Everything a session needs from outside: the loaded data, the event sink and the knobs.
#[derive(Clone)]
pub struct SessionEnv {
    /// Camera profiles and the registry.
    pub profiles: Arc<ProfileStore>,
    /// Noise priors by body.
    pub priors: Arc<PriorStore>,
    /// Illuminant estimator models by profile (the as-shot fallback where there is none).
    pub illuminants: Arc<IlluminantModels>,
    /// Where progress events go.
    pub events: EventSink,
    /// The thumbnail hook, if the render graph is attached.
    pub thumbnails: Option<Arc<dyn ThumbnailRenderer>>,
    /// The user's default look: the look of every photo opened from now on.
    pub default_look: LookId,
    /// The look the white lift is sized for: Signature, read from `<data>/looks/builtin/signature.json`
    /// by [`SessionEnv::load`] (`None`, the identity look, when the file is missing or unusable). It
    /// does not follow the user's default look: the lift of a photo is what the picture needs on top
    /// of the Signature look (`docs/02` C12 step 1b).
    pub reference_look: Option<Arc<LookTransform>>,
    /// Threads of the decode pool.
    pub decode_threads: usize,
    /// A profile id that replaces the detected one (the CLI's `--profile`).
    pub forced_profile: Option<String>,
}

/// The decode pool size: `min(available_parallelism - 2, 6)`, at least one.
#[must_use]
pub fn default_decode_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
    cpus.saturating_sub(2).clamp(1, 6)
}

/// The Signature look the white lift is sized for: `<data_root>/looks/builtin/signature.json`, baked
/// like any built-in look. `None` when there is no such file or it cannot be used (the lift is then
/// sized for the bare ACES 2 picture).
fn load_reference_look(data_root: &Path) -> Option<Arc<LookTransform>> {
    let text = std::fs::read_to_string(
        data_root
            .join("looks")
            .join("builtin")
            .join("signature.json"),
    )
    .ok()?;
    let look = ParametricLook::from_json(&text).ok()?.to_transform().ok()?;
    Some(Arc::new(look))
}

impl SessionEnv {
    /// Loads the profiles, the noise priors and the illuminant models from a data folder
    /// (`<data_root>/cameras`, `<data_root>/models/illum`, ...).
    ///
    /// # Errors
    ///
    /// [`SessionError::Data`] when the profiles or priors cannot be read (a missing illuminant
    /// model is not an error: that profile uses the as-shot fallback).
    pub fn load(data_root: &Path) -> Result<Self, SessionError> {
        let profiles =
            ProfileStore::load(data_root).map_err(|e| SessionError::Data(e.to_string()))?;
        let priors = PriorStore::load_dir(&data_root.join("cameras").join("noise"))
            .map_err(|e| SessionError::Data(e.to_string()))?;
        let illuminants = IlluminantModels::load(data_root, &profiles);
        Ok(Self {
            profiles: Arc::new(profiles),
            priors: Arc::new(priors),
            illuminants: Arc::new(illuminants),
            events: EventSink::none(),
            thumbnails: None,
            // No look library is attached yet, so the identity look; `Core::open` sets the user's
            // default look (Signature out of the box) with `with_default_look`.
            default_look: LookId::neutral(),
            reference_look: load_reference_look(data_root),
            decode_threads: default_decode_threads(),
            forced_profile: None,
        })
    }

    /// [`SessionEnv::load`] from the data folder of the running process (`docs/01` section Finding
    /// `data/`).
    ///
    /// # Errors
    ///
    /// [`SessionError::Data`] when the folder is missing or unreadable.
    pub fn from_data_dir() -> Result<Self, SessionError> {
        Self::load(&find_data_dir()?)
    }

    /// Sets where events go.
    #[must_use]
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Attaches the thumbnail renderer.
    #[must_use]
    pub fn with_thumbnails(mut self, renderer: Arc<dyn ThumbnailRenderer>) -> Self {
        self.thumbnails = Some(renderer);
        self
    }

    /// Sets the default look of new photos.
    #[must_use]
    pub fn with_default_look(mut self, look: LookId) -> Self {
        self.default_look = look;
        self
    }

    /// Sets the number of decode threads (at least one).
    #[must_use]
    pub fn with_decode_threads(mut self, threads: usize) -> Self {
        self.decode_threads = threads.max(1);
        self
    }

    /// Forces a camera profile for every photo (`ym render --profile`).
    ///
    /// # Errors
    ///
    /// [`SessionError::Data`] with the valid ids when `id` names no profile.
    pub fn with_forced_profile(mut self, id: &str) -> Result<Self, SessionError> {
        match self.profiles.profile_by_name(id.trim()) {
            Some(p) => {
                self.forced_profile = Some(p.id.to_string());
                Ok(self)
            }
            None => Err(SessionError::Data(format!(
                "There is no camera profile called {id}. The profiles are: {}.",
                self.profiles
                    .profile_ids()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Opening: collecting files
// ---------------------------------------------------------------------------------------------

/// How [`Session::open`] reads folders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenOptions {
    /// Walk subfolders too (the CLI's `--recursive`). The app opens one folder level.
    pub recursive: bool,
}

/// What [`Session::open`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenOutcome {
    /// The photos added, in filmstrip order.
    pub added: Vec<PhotoId>,
    /// ARW files that were already in the session (same file) and so were not added again. A file
    /// counts once, however often the call listed it.
    pub already_open: usize,
    /// Files and folder entries that were ignored: not ARW, or hidden `.` files.
    pub ignored: usize,
    /// Paths that do not exist.
    pub missing: Vec<PathBuf>,
}

impl OpenOutcome {
    /// True when what was given held no ARW file at all: the Start screen says
    /// [`NO_ARW_MESSAGE`].
    #[must_use]
    pub fn no_arw(&self) -> bool {
        self.added.is_empty() && self.already_open == 0
    }

    /// The message for the Start screen, when there is one.
    #[must_use]
    pub fn message(&self) -> Option<&'static str> {
        self.no_arw().then_some(NO_ARW_MESSAGE)
    }
}

#[derive(Debug, Default)]
struct Collected {
    files: Vec<PathBuf>,
    ignored: usize,
    missing: Vec<PathBuf>,
    /// The only folder that was given, if exactly one folder and nothing else was.
    single_folder: Option<PathBuf>,
}

fn is_arw(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("arw"))
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
}

fn walk(dir: &Path, recursive: bool, out: &mut Collected) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = read.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let is_dir = meta.is_dir();
        let is_file = meta.is_file() || (meta.file_type().is_symlink() && path.is_file());
        if is_hidden(&path) {
            out.ignored += 1;
        } else if is_dir {
            if recursive {
                walk(&path, recursive, out);
            }
        } else if is_file && is_arw(&path) {
            out.files.push(path);
        } else {
            out.ignored += 1;
        }
    }
}

fn collect(paths: &[PathBuf], recursive: bool) -> Collected {
    let mut out = Collected::default();
    for path in paths {
        if path.is_dir() {
            walk(path, recursive, &mut out);
        } else if path.is_file() {
            if is_arw(path) {
                out.files.push(path.clone());
            } else {
                out.ignored += 1;
            }
        } else {
            out.missing.push(path.clone());
        }
    }
    if let [only] = paths
        && only.is_dir()
    {
        out.single_folder = Some(named_folder(only));
    }
    out
}

/// The folder all of the given photos are in, if they share one.
fn common_parent(st: &State, ids: &[PhotoId]) -> Option<PathBuf> {
    let mut parents = ids
        .iter()
        .filter_map(|id| st.photo(*id))
        .map(|p| p.path.parent().map(Path::to_path_buf));
    let first = parents.next()??;
    parents.all(|p| p.as_ref() == Some(&first)).then_some(first)
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// `path` made absolute against the working directory without resolving symbolic links (the
/// photo keeps the name the user gave it). A path that cannot be made absolute is kept.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A folder the user gave, in a form that has a name: `.` and `..` have none, so they are
/// resolved (`ym render .` names its export folder after the folder it is run in).
fn named_folder(path: &Path) -> PathBuf {
    let abs = absolute(path);
    if abs.file_name().is_some() {
        abs
    } else {
        canonical(path)
    }
}

// ---------------------------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------------------------

/// The Pick white sample that belongs to `white_pick`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickedWhite {
    /// The point, as in `DevelopSettings::white_pick`.
    pub point: [f64; 2],
    /// The 5 x 5 median of the g0-balanced camera RGB there ([`pick_white_sample`]).
    pub sample: Vec3,
}

struct Photo {
    id: PhotoId,
    path: PathBuf,
    /// The canonical form of `path`, the key of the set of files already open (`State::known`).
    canon: PathBuf,
    /// Counts the times the photo was taken out of the session ([`Session::remove`]). A job carries the number the photo
    /// had when it was queued, so the job of a photo that was removed and put back stops: the photo has a new job.
    incarnation: u32,
    summary: PhotoSummary,
    meta: Option<Arc<RawMetadata>>,
    camera: Option<PhotoCamera>,
    profile: Option<Arc<Profile>>,
    pool_key: Option<PoolKey>,
    burst: Option<BurstFrame>,
    /// The photo's own estimates.
    own: Option<Arc<AutoEstimates>>,
    /// The estimates served: own, with the burst median and the pooled noise.
    effective: Option<Arc<AutoEstimates>>,
    /// Bumped whenever `own` is replaced (the pool of a group is rebuilt only when this changed).
    own_gen: u64,
    /// The version of the noise pool the effective noise was built from.
    noise_version: u64,
    settings: DevelopSettings,
    pick: Option<PickedWhite>,
    /// Bumped whenever the effective estimates change; the render graph's cache key.
    version: u64,
}

struct GroupPool {
    /// Bumped when the pool is rebuilt from a different set of estimates.
    version: u64,
    pool: NoisePool,
    /// The members the pool was built from: photo id and `own_gen`.
    signature: Vec<(PhotoId, u64)>,
}

#[derive(Default)]
struct MosaicCache {
    entries: Vec<(PhotoId, Arc<RawImage>)>,
    max_len: usize,
}

/// A photo taken out of the session, and what it takes to put it back.
struct RemovedPhoto {
    /// Its place in the filmstrip before the removal.
    place: usize,
    photo: Photo,
    picked: bool,
}

/// The photos of the last removal, kept in memory for [`Session::restore_removed`] until the next removal, until the
/// session is closed, or until files are opened into the emptied session (a new session).
struct Removal {
    /// In filmstrip order.
    photos: Vec<RemovedPhoto>,
    /// The current photo before the removal.
    was_current: Option<PhotoId>,
    /// The current photo after it: the one `was_current` handed over to.
    successor: Option<PhotoId>,
    /// The session's folder before the removal, for a session that is empty when the photos come back.
    folder: Option<PathBuf>,
    mixed_folders: bool,
}

struct State {
    photos: Vec<Photo>,
    index: HashMap<PhotoId, usize>,
    next_id: u32,
    current: Option<usize>,
    picked: BTreeSet<PhotoId>,
    known: HashSet<PathBuf>,
    folder: Option<PathBuf>,
    mixed_folders: bool,
    pools: HashMap<PoolKey, GroupPool>,
    cache: MosaicCache,
    default_look: LookId,
    /// What [`Session::restore_removed`] puts back.
    removal: Option<Removal>,
}

impl State {
    fn new(default_look: LookId) -> Self {
        Self {
            photos: Vec::new(),
            index: HashMap::new(),
            next_id: 1,
            current: None,
            picked: BTreeSet::new(),
            known: HashSet::new(),
            folder: None,
            mixed_folders: false,
            pools: HashMap::new(),
            cache: MosaicCache::default(),
            default_look,
            removal: None,
        }
    }

    /// Forgets every photo. Ids keep counting: a command or job that outlives the close names an
    /// id that no longer exists, instead of a different photo of the next session.
    fn clear(&mut self) {
        let look = self.default_look.clone();
        let next_id = self.next_id;
        *self = Self::new(look);
        self.next_id = next_id;
    }

    fn photo(&self, id: PhotoId) -> Option<&Photo> {
        self.index.get(&id).map(|&i| &self.photos[i])
    }

    /// Whether the photo is in the session (and, with `incarnation`, has not been taken out and put back since).
    fn has_photo(&self, id: PhotoId, incarnation: Option<u32>) -> bool {
        self.photo(id)
            .is_some_and(|p| incarnation.is_none_or(|n| p.incarnation == n))
    }

    /// Rebuilds the id-to-place map after photos came out of the list or went into it.
    fn reindex(&mut self) {
        self.index = self
            .photos
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id, i))
            .collect();
    }

    /// The photos of the window around the current one that are developed and whose mosaic is not cached: what a
    /// move of the current photo queues for decoding.
    fn prefetch_candidates(&self) -> Vec<PhotoId> {
        self.window()
            .into_iter()
            .map(|w| &self.photos[w])
            .filter(|p| p.effective.is_some() && self.cache_get(p.id).is_none())
            .map(|p| p.id)
            .collect()
    }

    /// The indices of the photos whose mosaic is worth keeping: current, previous, next.
    fn window(&self) -> Vec<usize> {
        let Some(cur) = self.current else {
            return Vec::new();
        };
        let lo = cur.saturating_sub(1);
        let hi = (cur + 1).min(self.photos.len().saturating_sub(1));
        (lo..=hi).collect()
    }

    fn in_window(&self, id: PhotoId) -> bool {
        self.index
            .get(&id)
            .is_some_and(|i| self.window().contains(i))
    }

    fn cache_get(&self, id: PhotoId) -> Option<Arc<RawImage>> {
        self.cache
            .entries
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, r)| r.clone())
    }

    /// Keeps `raw` when its photo is in the window; evicts what fell out of it.
    fn cache_offer(&mut self, id: PhotoId, raw: Arc<RawImage>) {
        self.cache_trim();
        if self.in_window(id) && self.cache_get(id).is_none() {
            self.cache.entries.push((id, raw));
            self.cache.max_len = self.cache.max_len.max(self.cache.entries.len());
        }
    }

    fn cache_trim(&mut self) {
        let window: HashSet<PhotoId> = self
            .window()
            .into_iter()
            .map(|i| self.photos[i].id)
            .collect();
        self.cache.entries.retain(|(id, _)| window.contains(id));
    }

    /// Recomputes the burst median exposure of every photo with estimates and, for the photos
    /// whose pooled noise is out of date, the pooled noise. Returns the photos whose served
    /// estimates changed in a way a viewer should react to: a different exposure, or a different
    /// noise model on the current photo.
    fn refresh_effective(&mut self, priors: &PriorStore) -> Vec<PhotoId> {
        let with: Vec<usize> = (0..self.photos.len())
            .filter(|&i| self.photos[i].own.is_some())
            .collect();
        let frames: Vec<BurstFrame> = with
            .iter()
            .filter_map(|&i| self.photos[i].burst.clone())
            .collect();
        let evs: Vec<f64> = with
            .iter()
            .filter_map(|&i| self.photos[i].own.as_ref().map(|o| o.exposure.own_ev))
            .collect();
        let shared = if frames.len() == with.len() {
            share_burst_ev(&frames, &evs)
        } else {
            evs
        };
        let current = self.current;
        let mut changed = Vec::new();
        for (slot, &i) in with.iter().enumerate() {
            let ev = shared.get(slot).copied().unwrap_or(0.0);
            if let Some(id) = self.refresh_photo(i, Some(ev), priors, current == Some(i)) {
                changed.push(id);
            }
        }
        changed
    }

    /// Rebuilds one photo's served estimates when the burst median or the pooled noise moved;
    /// returns its id when a viewer should react. Cheap when nothing moved: the pooled model is
    /// refitted only for a photo whose pool was rebuilt since.
    fn refresh_photo(
        &mut self,
        i: usize,
        shared_ev: Option<f64>,
        priors: &PriorStore,
        is_current: bool,
    ) -> Option<PhotoId> {
        let own = self.photos[i].own.clone()?;
        let old = self.photos[i].effective.clone();
        let ev = shared_ev
            .unwrap_or_else(|| old.as_ref().map_or(own.exposure.own_ev, |e| e.exposure.ev));
        let group_version = self.photos[i]
            .pool_key
            .as_ref()
            .and_then(|k| self.pools.get(k))
            .map(|g| g.version);
        let noise_stale =
            old.is_none() || group_version.is_some_and(|v| v != self.photos[i].noise_version);
        let ev_moved = old
            .as_ref()
            .is_none_or(|o| (o.exposure.ev - ev).abs() > 1e-12);
        if !noise_stale && !ev_moved {
            return None;
        }
        // The noise: pooled when the photo has a pool, else its own; unchanged when only the
        // exposure moved.
        let mut noise_version = self.photos[i].noise_version;
        let noise = if noise_stale {
            let pooled = match (self.photos[i].pool_key.clone(), self.photos[i].meta.clone()) {
                (Some(key), Some(meta)) => self.pools.get_mut(&key).map(|group| {
                    noise_version = group.version;
                    group
                        .pool
                        .result_for(&key, &own.noise, &noise_prior(priors, &meta))
                }),
                _ => None,
            };
            pooled.unwrap_or_else(|| own.noise.clone())
        } else {
            old.as_ref()
                .map_or_else(|| own.noise.clone(), |o| o.noise.clone())
        };
        let noise_moved = old.as_ref().is_some_and(|o| o.noise.model != noise.model);
        let mut next = if (ev - own.exposure.own_ev).abs() < 1e-12 {
            (*own).clone()
        } else {
            own.with_shared_ev(ev)
        };
        next = next.with_noise(noise);
        let photo = &mut self.photos[i];
        photo.noise_version = noise_version;
        let changed = old.is_none() || ev_moved || noise_moved;
        if changed {
            photo.effective = Some(Arc::new(next));
            photo.version += 1;
        }
        let react = old.is_some() && (ev_moved || (noise_moved && is_current));
        react.then_some(photo.id)
    }

    /// Whether the estimates of photo `id` that depend on other photos are final: every photo that
    /// can still move them has finished its own estimates (or failed and never will).
    ///
    /// Two things are shared. The burst median exposure depends on the frames of the photo's burst,
    /// which are known from the metadata alone: the groups are made over **every** photo with a
    /// capture record, so a frame that has not been estimated yet is still a member (grouping only
    /// the estimated photos can split a burst and join it again when the missing frame arrives;
    /// leaving frames out only ever splits a group, so a group whose members are all resolved is
    /// final in the served numbers too). The pooled noise depends on the photos of the pool key
    /// (model, ISO, bit depth, compression), of which the first sixteen estimated ones in id order
    /// build the pool: all of them must be resolved, which is stricter than needed for a group of
    /// more than sixteen and simple. A photo with no ISO has no pool.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`], and [`SessionError::Skipped`] for a photo that cannot be
    /// developed. A photo with no estimates of its own yet is `Ok(false)`.
    fn estimates_final(&self, id: PhotoId) -> Result<bool, SessionError> {
        let i = *self.index.get(&id).ok_or(SessionError::UnknownPhoto(id))?;
        let photo = &self.photos[i];
        if let PhotoStatus::Skipped { reason } = &photo.summary.status {
            return Err(SessionError::Skipped(reason.clone()));
        }
        if photo.own.is_none() {
            return Ok(false);
        }
        let resolved =
            |p: &Photo| p.own.is_some() || matches!(p.summary.status, PhotoStatus::Skipped { .. });
        if photo.burst.is_some() {
            let members: Vec<usize> = (0..self.photos.len())
                .filter(|&j| self.photos[j].burst.is_some())
                .collect();
            let frames: Vec<BurstFrame> = members
                .iter()
                .filter_map(|&j| self.photos[j].burst.clone())
                .collect();
            let groups = burst_groups(&frames);
            let mine = members.iter().position(|&j| j == i).map(|k| groups[k]);
            if members
                .iter()
                .zip(&groups)
                .any(|(&j, &g)| Some(g) == mine && !resolved(&self.photos[j]))
            {
                return Ok(false);
            }
        }
        if let Some(key) = &photo.pool_key
            && self
                .photos
                .iter()
                .any(|p| p.pool_key.as_ref() == Some(key) && !resolved(p))
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Rebuilds the noise pool of a photo's group when the set of estimates in it changed; the
    /// group's photos are re-served on their next refresh.
    fn rebuild_pool(&mut self, key: &PoolKey, priors: &PriorStore) {
        let mut members: Vec<(PhotoId, u64)> = self
            .photos
            .iter()
            .filter(|p| p.pool_key.as_ref() == Some(key))
            .filter(|p| p.own.as_ref().is_some_and(|o| o.noise.estimate.is_some()))
            .map(|p| (p.id, p.own_gen))
            .collect();
        members.sort_by_key(|m| m.0);
        members.truncate(POOL_MAX_FILES);
        if self.pools.get(key).is_some_and(|g| g.signature == members) {
            return;
        }
        let built: Vec<(Arc<AutoEstimates>, NoisePrior)> = members
            .iter()
            .filter_map(|(id, _)| {
                let p = self.photo(*id)?;
                Some((p.own.clone()?, noise_prior(priors, p.meta.as_ref()?)))
            })
            .collect();
        let pool = NoisePool::build(key, built.iter().map(|(o, p)| (&o.noise, p)));
        let version = self.pools.get(key).map_or(1, |g| g.version + 1);
        self.pools.insert(
            key.clone(),
            GroupPool {
                version,
                pool,
                signature: members,
            },
        );
    }

    /// Takes the photos in `doomed` out of the filmstrip in one step (see [`Session::remove`]): the photos, with their
    /// places and picks, are set aside as the session's [`Removal`] (the files on disk are not touched), the current photo hands over to the next photo that stays (the
    /// previous one when the last was removed), their mosaics leave the cache, and the estimates the remaining photos
    /// share (the burst median, the pooled noise) are made again from the photos that are left. Returns the ids taken out,
    /// in filmstrip order, and the photos whose served estimates moved; `None` when none of the ids is in the session
    /// (nothing changes, and a removal set aside earlier stays).
    fn take_out(
        &mut self,
        doomed: &HashSet<PhotoId>,
        priors: &PriorStore,
    ) -> Option<(Vec<PhotoId>, Vec<PhotoId>)> {
        if !self.photos.iter().any(|p| doomed.contains(&p.id)) {
            return None;
        }
        let was_current = self.current.map(|i| self.photos[i].id);
        let successor = match self.current {
            Some(cur) if doomed.contains(&self.photos[cur].id) => {
                let stays = |i: &usize| !doomed.contains(&self.photos[*i].id);
                (cur + 1..self.photos.len())
                    .find(stays)
                    .or_else(|| (0..cur).rev().find(stays))
                    .map(|i| self.photos[i].id)
            }
            _ => was_current,
        };
        let mut taken = Vec::new();
        let mut kept = Vec::with_capacity(self.photos.len());
        for (place, mut photo) in std::mem::take(&mut self.photos).into_iter().enumerate() {
            if doomed.contains(&photo.id) {
                photo.incarnation += 1;
                self.known.remove(&photo.canon);
                let picked = self.picked.remove(&photo.id);
                taken.push(RemovedPhoto {
                    place,
                    photo,
                    picked,
                });
            } else {
                kept.push(photo);
            }
        }
        self.photos = kept;
        self.reindex();
        self.current = successor.and_then(|id| self.index.get(&id).copied());
        self.cache.entries.retain(|(id, _)| !doomed.contains(id));
        self.cache_trim();
        // The burst median and the pooled noise are functions of the set of photos: made again from the ones left.
        let keys: HashSet<PoolKey> = taken
            .iter()
            .filter_map(|t| t.photo.pool_key.clone())
            .collect();
        for key in &keys {
            self.rebuild_pool(key, priors);
        }
        let changed = self.refresh_effective(priors);
        let removed = taken.iter().map(|t| t.photo.id).collect();
        self.removal = Some(Removal {
            photos: taken,
            was_current,
            successor,
            folder: self.folder.clone(),
            mixed_folders: self.mixed_folders,
        });
        Some((removed, changed))
    }

    /// Puts the photos of the last removal back (see [`Session::restore_removed`]); `None` when there is none.
    fn put_back(&mut self, priors: &PriorStore) -> Option<PutBack> {
        let removal = self.removal.take()?;
        let was_empty = self.photos.is_empty();
        let current_id = self.current.map(|i| self.photos[i].id);
        let mut restored = Vec::new();
        let mut already_open = 0;
        let mut redevelop = Vec::new();
        for RemovedPhoto {
            place,
            mut photo,
            picked,
        } in removal.photos
        {
            // The same file was added again meanwhile: that photo is the one in the session.
            if self.known.contains(&photo.canon) {
                already_open += 1;
                continue;
            }
            if !matches!(
                photo.summary.status,
                PhotoStatus::Ready | PhotoStatus::Skipped { .. }
            ) {
                // Its development was cut short: it starts again.
                photo.summary.status = PhotoStatus::Waiting;
                photo.own = None;
                photo.effective = None;
                redevelop.push(photo.id);
            }
            self.known.insert(photo.canon.clone());
            if picked {
                self.picked.insert(photo.id);
            }
            restored.push(photo.id);
            let at = place.min(self.photos.len());
            self.photos.insert(at, photo);
        }
        self.reindex();
        // The photo on screen stays, unless the removal had taken it away and the person has not moved since: then it
        // comes back to it.
        self.current = current_id.and_then(|id| self.index.get(&id).copied());
        let back_to = removal
            .was_current
            .filter(|c| restored.contains(c) && current_id == removal.successor);
        if let Some(c) = back_to {
            self.current = self.index.get(&c).copied();
        }
        if self.current.is_none() && !self.photos.is_empty() {
            self.current = Some(0);
        }
        if !restored.is_empty() {
            if was_empty {
                self.folder = removal.folder;
                self.mixed_folders = removal.mixed_folders;
            } else {
                let source = common_parent(self, &restored);
                if source.is_none() || source != self.folder {
                    self.mixed_folders = true;
                }
            }
        }
        self.cache_trim();
        let keys: HashSet<PoolKey> = restored
            .iter()
            .filter_map(|id| self.photo(*id)?.pool_key.clone())
            .collect();
        for key in &keys {
            self.rebuild_pool(key, priors);
        }
        let changed = self
            .refresh_effective(priors)
            .into_iter()
            .filter(|c| !restored.contains(c))
            .collect();
        Some(PutBack {
            restored,
            already_open,
            redevelop,
            changed,
        })
    }
}

/// What [`State::put_back`] did.
struct PutBack {
    restored: Vec<PhotoId>,
    already_open: usize,
    /// The restored photos whose development had not finished: they need a job again.
    redevelop: Vec<PhotoId>,
    changed: Vec<PhotoId>,
}

// ---------------------------------------------------------------------------------------------
// The decode pool
// ---------------------------------------------------------------------------------------------

/// How many times `update_settings` applies its patch again after a conflicting command before it
/// stops waiting for a quiet moment.
const MAX_COMMIT_ATTEMPTS: u32 = 64;

/// Whether a commit of settings waits for the photo's settings to be what the caller started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Commit {
    /// A patch: only if no other command changed the settings meanwhile.
    IfUnchanged,
    /// A replacement: whatever is there.
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKind {
    /// Decode, estimate, thumbnail.
    Develop,
    /// Decode again so a neighbour of the current photo is cached.
    Prefetch,
}

#[derive(Debug, Clone, Copy)]
struct Job {
    epoch: u64,
    photo: PhotoId,
    kind: JobKind,
    /// The photo's [`Photo::incarnation`] when the job was queued.
    incarnation: u32,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    running: usize,
    shutdown: bool,
}

const NO_PHOTO: u32 = u32::MAX;

struct Inner {
    env: SessionEnv,
    state: Mutex<State>,
    /// Held while an event is checked and sent, and by `close` while it bumps the epoch.
    emit_lock: Mutex<()>,
    epoch: AtomicU64,
    shutdown: AtomicBool,
    current_hint: AtomicU32,
    queue: Mutex<Queue>,
    work: Condvar,
    idle: Condvar,
    /// Jobs started, for tests and `doctor`-style diagnostics.
    jobs_started: AtomicU64,
    /// Files decoded by [`Session::raw_image`] (not by the development jobs), for tests.
    raw_decodes: AtomicU64,
    /// The decodes of [`Session::raw_image`] in progress: callers that ask for a photo that is
    /// being decoded wait for that decode instead of reading the file again.
    decoding: SingleFlight<PhotoId, Arc<RawImage>>,
}

/// The session: photos in memory, the current photo, the picked set, the decode pool.
///
/// All methods take `&self`; the session can be shared between threads (`Arc<Session>`). Dropping
/// it stops the pool.
pub struct Session {
    inner: Arc<Inner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

/// A photo as the render graph and the export queue need it: everything immutable about the
/// photo, plus a snapshot of its settings and estimates.
#[derive(Debug, Clone)]
pub struct PhotoView {
    /// The photo.
    pub id: PhotoId,
    /// The original file (never written).
    pub path: PathBuf,
    /// The probe's metadata.
    pub meta: Arc<RawMetadata>,
    /// The camera match.
    pub camera: PhotoCamera,
    /// The profile the photo uses.
    pub profile: Arc<Profile>,
    /// The photo's settings.
    pub settings: DevelopSettings,
    /// The estimates, with the burst median and the pooled noise.
    pub estimates: Arc<AutoEstimates>,
    /// The Pick white sample, when `settings.white_pick` is set.
    pub pick: Option<PickedWhite>,
    /// Changes whenever the estimates or the Pick white sample change.
    pub estimates_version: u64,
}

impl PhotoView {
    /// The Pick white sample to give the colour chain: the one that belongs to the current
    /// `settings.white_pick`.
    #[must_use]
    pub fn picked_white(&self) -> Option<Vec3> {
        self.pick
            .filter(|p| Some(p.point) == self.settings.white_pick)
            .map(|p| p.sample)
    }

    /// The settings resolved against the estimates.
    ///
    /// # Errors
    ///
    /// [`SettingsError`] when the colour chain cannot be built.
    pub fn resolved(&self) -> Result<ResolvedSettings, SettingsError> {
        ResolvedSettings::resolve(
            self.id,
            &self.settings,
            &self.estimates,
            &self.profile,
            self.picked_white(),
        )
    }
}

impl Session {
    /// A session with an empty filmstrip and a running decode pool.
    #[must_use]
    pub fn new(env: SessionEnv) -> Self {
        let threads = env.decode_threads.max(1);
        let default_look = env.default_look.clone();
        let inner = Arc::new(Inner {
            env,
            state: Mutex::new(State::new(default_look)),
            emit_lock: Mutex::new(()),
            epoch: AtomicU64::new(1),
            shutdown: AtomicBool::new(false),
            current_hint: AtomicU32::new(NO_PHOTO),
            queue: Mutex::new(Queue::default()),
            work: Condvar::new(),
            idle: Condvar::new(),
            jobs_started: AtomicU64::new(0),
            raw_decodes: AtomicU64::new(0),
            decoding: SingleFlight::new(),
        });
        let workers: Vec<JoinHandle<()>> = (0..threads)
            .filter_map(|n| {
                let inner = Arc::clone(&inner);
                std::thread::Builder::new()
                    .name(format!("ym-decode-{n}"))
                    .spawn(move || inner.worker())
                    .map_err(|e| tracing::error!(error = %e, "could not start a decode thread"))
                    .ok()
            })
            .collect();
        Self {
            inner,
            workers: Mutex::new(workers),
        }
    }

    /// The environment the session was built with.
    #[must_use]
    pub fn env(&self) -> &SessionEnv {
        &self.inner.env
    }

    // --- Opening -----------------------------------------------------------------------------

    /// Opens files and folders, adding to the photos already open (see the module documentation).
    /// Relative paths are taken against the working directory and stored as absolute paths; a file
    /// that another `open` registered while this one probed is counted as already open.
    ///
    /// Returns once every file has been probed and the background jobs are queued; the photos
    /// then develop while the caller carries on. The new photos are in the filmstrip in the order
    /// of their paths (case-insensitive); the first photo of an empty session becomes current.
    pub fn open(&self, paths: &[PathBuf], options: OpenOptions) -> OpenOutcome {
        let inner = &self.inner;
        let epoch = inner.epoch.load(Ordering::SeqCst);
        let collected = collect(paths, options.recursive);
        let mut outcome = OpenOutcome {
            ignored: collected.ignored,
            missing: collected.missing,
            ..OpenOutcome::default()
        };
        // New files only, sorted, each file once.
        let mut files: Vec<(PathBuf, PathBuf)> = {
            let st = lock(&inner.state);
            let mut seen: HashSet<PathBuf> = HashSet::new();
            let mut fresh = Vec::new();
            for path in collected.files {
                let canon = canonical(&path);
                // A file listed twice in one call (a folder and a file inside it) is one file:
                // the repeat is neither added nor counted as already open.
                if !seen.insert(canon.clone()) {
                    continue;
                }
                if st.known.contains(&canon) {
                    outcome.already_open += 1;
                } else {
                    fresh.push((absolute(&path), canon));
                }
            }
            fresh
        };
        files.sort_by_key(|(p, _)| p.to_string_lossy().to_lowercase());
        // Probe in parallel; the result order is the file order.
        let probed: Vec<Result<RawMetadata, RawError>> = files
            .par_iter()
            .map(|(path, _)| ym_raw::metadata::probe(path))
            .collect();

        let mut events = Vec::new();
        let mut develop = Vec::new();
        {
            let mut st = lock(&inner.state);
            if inner.epoch.load(Ordering::SeqCst) != epoch {
                return outcome; // closed while probing
            }
            let was_empty = st.photos.is_empty();
            // Files opened into an emptied session start a new one: what the last removal kept is not offered any more.
            if was_empty && !files.is_empty() {
                st.removal = None;
            }
            for ((path, canon), probe) in files.into_iter().zip(probed) {
                // Another `open` of the same file (a folder dropped twice in a row) may have
                // registered it while this one was probing.
                if st.known.contains(&canon) {
                    outcome.already_open += 1;
                    continue;
                }
                let id = PhotoId(st.next_id);
                st.next_id += 1;
                let photo = inner.new_photo(id, path, canon.clone(), &st.default_look, probe);
                events.push(PhotoProgress {
                    id,
                    stage: PhotoStage::Probe,
                    done: true,
                    error: match &photo.summary.status {
                        PhotoStatus::Skipped { reason } => Some(reason.clone()),
                        _ => None,
                    },
                });
                if photo.meta.is_some() {
                    develop.push(id);
                }
                st.known.insert(canon);
                let idx = st.photos.len();
                st.index.insert(id, idx);
                st.photos.push(photo);
                outcome.added.push(id);
            }
            if !outcome.added.is_empty() {
                let source = collected
                    .single_folder
                    .clone()
                    .or_else(|| common_parent(&st, &outcome.added));
                if was_empty {
                    st.current = Some(0);
                    inner
                        .current_hint
                        .store(st.photos[0].id.0, Ordering::SeqCst);
                    st.folder = source;
                    st.mixed_folders = st.folder.is_none();
                } else if source.is_none() || source != st.folder {
                    st.mixed_folders = true;
                }
            }
        }
        for e in events {
            inner.emit(epoch, SessionEvent::PhotoProgress(e));
        }
        inner.enqueue(epoch, &develop, JobKind::Develop);
        outcome
    }

    // --- Reading -----------------------------------------------------------------------------

    /// The session as the webview mirrors it (`session_state`).
    #[must_use]
    pub fn state(&self) -> SessionState {
        let st = lock(&self.inner.state);
        let photos = st
            .photos
            .iter()
            .map(|p| {
                let mut s = p.summary.clone();
                s.picked = st.picked.contains(&p.id);
                s
            })
            .collect();
        // An emptied session (every photo was removed) names no folder; the next `open` sets it afresh.
        let folder = st
            .folder
            .as_ref()
            .filter(|_| !st.mixed_folders && !st.photos.is_empty());
        let folder_name = folder
            .and_then(|f| f.file_name())
            .map(|n| n.to_string_lossy().into_owned());
        let default_export_folder = folder.and_then(|f| {
            let name = f.file_name()?.to_string_lossy().into_owned();
            let parent = f.parent()?;
            Some(
                parent
                    .join(format!("{name} export"))
                    .to_string_lossy()
                    .into_owned(),
            )
        });
        SessionState {
            folder_name,
            photos,
            current_index: st.current,
            picked: st.picked.iter().copied().collect(),
            default_export_folder,
        }
    }

    /// The ids of the photos in filmstrip order.
    #[must_use]
    pub fn photo_ids(&self) -> Vec<PhotoId> {
        lock(&self.inner.state)
            .photos
            .iter()
            .map(|p| p.id)
            .collect()
    }

    /// The summary of one photo.
    #[must_use]
    pub fn summary(&self, id: PhotoId) -> Option<PhotoSummary> {
        let st = lock(&self.inner.state);
        st.photo(id).map(|p| {
            let mut s = p.summary.clone();
            s.picked = st.picked.contains(&id);
            s
        })
    }

    /// The status of one photo.
    #[must_use]
    pub fn status(&self, id: PhotoId) -> Option<PhotoStatus> {
        lock(&self.inner.state)
            .photo(id)
            .map(|p| p.summary.status.clone())
    }

    /// The current photo.
    #[must_use]
    pub fn current(&self) -> Option<PhotoId> {
        let st = lock(&self.inner.state);
        st.current.map(|i| st.photos[i].id)
    }

    /// The picked photos, in filmstrip order.
    #[must_use]
    pub fn picked(&self) -> Vec<PhotoId> {
        let st = lock(&self.inner.state);
        st.photos
            .iter()
            .map(|p| p.id)
            .filter(|id| st.picked.contains(id))
            .collect()
    }

    /// The Camera card of a photo (RAW tab).
    #[must_use]
    pub fn camera_card(&self, id: PhotoId) -> Option<CameraCard> {
        lock(&self.inner.state)
            .photo(id)
            .and_then(|p| p.camera.as_ref().map(PhotoCamera::card))
    }

    /// The camera match of a photo.
    #[must_use]
    pub fn camera(&self, id: PhotoId) -> Option<PhotoCamera> {
        lock(&self.inner.state)
            .photo(id)
            .and_then(|p| p.camera.clone())
    }

    /// A photo's settings and estimates, as the render graph and the export queue read them.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`], [`SessionError::Skipped`] for a photo that cannot be
    /// developed and [`SessionError::NotReady`] while its estimates are not there yet.
    pub fn view(&self, id: PhotoId) -> Result<PhotoView, SessionError> {
        let mut st = lock(&self.inner.state);
        let i = *st.index.get(&id).ok_or(SessionError::UnknownPhoto(id))?;
        if let PhotoStatus::Skipped { reason } = &st.photos[i].summary.status {
            return Err(SessionError::Skipped(reason.clone()));
        }
        // Serve the current pooled noise.
        let stale = {
            let p = &st.photos[i];
            p.pool_key
                .as_ref()
                .and_then(|k| st.pools.get(k))
                .is_some_and(|g| g.version != p.noise_version)
        };
        if stale {
            st.refresh_photo(i, None, &self.inner.env.priors, false);
        }
        let p = &st.photos[i];
        let (Some(meta), Some(camera), Some(profile), Some(estimates)) = (
            p.meta.clone(),
            p.camera.clone(),
            p.profile.clone(),
            p.effective.clone(),
        ) else {
            return Err(SessionError::NotReady(id));
        };
        Ok(PhotoView {
            id,
            path: p.path.clone(),
            meta,
            camera,
            profile,
            settings: p.settings.clone(),
            estimates,
            pick: p.pick,
            estimates_version: p.version,
        })
    }

    /// The estimates of a photo (see [`Session::view`]).
    #[must_use]
    pub fn estimates(&self, id: PhotoId) -> Option<Arc<AutoEstimates>> {
        self.view(id).ok().map(|v| v.estimates)
    }

    /// Whether the automatic estimates of a photo that depend on other photos, the burst median
    /// exposure and the pooled noise (`docs/02` Auto exposure step 6, `docs/06` Batch pooling), are
    /// final: every photo that can still change them has been estimated or has failed. The command
    /// line waits for the whole session to go idle before it exports, so its files carry final
    /// estimates; the export queue uses this to give a job started earlier the same ones, waiting
    /// only for the photos its photos depend on and not for thumbnails, prefetching or viewport work.
    ///
    /// Opening more photos or re-running a photo's automatic estimates can make final estimates
    /// move again; this answers for the photos open now.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`] when the photo is gone (the session was closed) and
    /// [`SessionError::Skipped`] for a photo that cannot be developed. `Ok(false)` while the photo
    /// has no estimates of its own yet.
    pub fn estimates_final(&self, id: PhotoId) -> Result<bool, SessionError> {
        lock(&self.inner.state).estimates_final(id)
    }

    /// `get_settings`: a photo's settings resolved against its estimates.
    ///
    /// # Errors
    ///
    /// As [`Session::view`], and [`SettingsError`] when the colour chain cannot be built.
    pub fn settings(&self, id: PhotoId) -> Result<ResolvedSettings, SessionError> {
        Ok(self.view(id)?.resolved()?)
    }

    // --- Removing photos ---------------------------------------------------------------------

    /// Takes photos out of the session (Remove from filmstrip): they leave the filmstrip, the picks and the caches, and
    /// their edits go with them (nothing is saved per photo, hard rule 4). **The files on disk are not touched** (hard
    /// rule 5); the session only forgets them.
    ///
    /// - Queued development and prefetch jobs for these photos are dropped, and a running one stops at its next stage; no
    ///   event about them is sent once this returns (a late `thumbnail_ready` or `photo_progress` is dropped).
    /// - Their decoded mosaics leave the cache.
    /// - The current photo hands over to the next photo that stays, or to the previous one when the last was removed; with
    ///   no photo left the session is empty and the start screen shows.
    /// - The estimates the other photos share are made again from the photos that are left: the burst median exposure of
    ///   the remaining frames of a burst, and the pooled noise of a noise group. The photos whose served estimates moved
    ///   get [`SessionEvent::EstimatesChanged`] and are listed in the outcome.
    /// - What was taken out is kept in memory for [`Session::restore_removed`] (their settings, estimates and place), until
    ///   the next removal that takes something out, [`Session::close`], or files opened into the emptied session (a new
    ///   session begins). Ids that are not in the session are ignored.
    ///
    /// The render cache of the photos is the renderer's: [`Core::remove_photos`](crate::app::Core::remove_photos) clears it
    /// and cancels their renders. Exports queued earlier keep the photos they were started with (their snapshots).
    pub fn remove(&self, ids: &[PhotoId]) -> RemoveOutcome {
        let inner = &self.inner;
        let doomed: HashSet<PhotoId> = ids.iter().copied().collect();
        // Held for the whole removal: a job that reports after this has the lock finds its photo gone and sends nothing.
        let emit = lock(&inner.emit_lock);
        let epoch = inner.epoch.load(Ordering::SeqCst);
        let (outcome, prefetch) = {
            let mut st = lock(&inner.state);
            let Some((removed, changed)) = st.take_out(&doomed, &inner.env.priors) else {
                return RemoveOutcome::default();
            };
            let current = st.current.map(|i| st.photos[i].id);
            inner
                .current_hint
                .store(current.map_or(NO_PHOTO, |c| c.0), Ordering::SeqCst);
            let outcome = RemoveOutcome {
                removed,
                current,
                estimates_changed: changed,
            };
            (outcome, st.prefetch_candidates())
        };
        if inner.live(epoch) {
            for &id in &outcome.estimates_changed {
                inner.env.events.send(SessionEvent::EstimatesChanged { id });
            }
        }
        drop(emit);
        {
            let mut q = lock(&inner.queue);
            q.jobs.retain(|j| !doomed.contains(&j.photo));
            if q.jobs.is_empty() && q.running == 0 {
                inner.idle.notify_all();
            }
        }
        inner.enqueue(epoch, &prefetch, JobKind::Prefetch);
        outcome
    }

    /// Puts back the photos of the last [`Session::remove`], where they were in the filmstrip, with their settings, picks
    /// and estimates exactly as they were; the photo that was current is current again when the removal took it away and
    /// the person has not moved to another since. A photo whose development was cut short starts again. A photo whose file
    /// was added to the session again meanwhile stays out (it is already there). Does nothing, and returns an empty
    /// outcome, when there is nothing to put back: after [`Session::close`], after a second removal took the place of the
    /// first, or when it was restored already.
    pub fn restore_removed(&self) -> RestoreOutcome {
        let inner = &self.inner;
        let emit = lock(&inner.emit_lock);
        let epoch = inner.epoch.load(Ordering::SeqCst);
        let (outcome, redevelop, prefetch) = {
            let mut st = lock(&inner.state);
            let Some(back) = st.put_back(&inner.env.priors) else {
                return RestoreOutcome::default();
            };
            let current = st.current.map(|i| st.photos[i].id);
            inner
                .current_hint
                .store(current.map_or(NO_PHOTO, |c| c.0), Ordering::SeqCst);
            let outcome = RestoreOutcome {
                restored: back.restored,
                already_open: back.already_open,
                current,
                estimates_changed: back.changed,
            };
            (outcome, back.redevelop, st.prefetch_candidates())
        };
        if inner.live(epoch) {
            for &id in &outcome.estimates_changed {
                inner.env.events.send(SessionEvent::EstimatesChanged { id });
            }
        }
        drop(emit);
        inner.enqueue(epoch, &redevelop, JobKind::Develop);
        inner.enqueue(epoch, &prefetch, JobKind::Prefetch);
        outcome
    }

    /// The photos [`Session::restore_removed`] would put back, in filmstrip order of the removal.
    #[must_use]
    pub fn restorable(&self) -> Vec<PhotoId> {
        lock(&self.inner.state)
            .removal
            .as_ref()
            .map(|r| r.photos.iter().map(|p| p.photo.id).collect())
            .unwrap_or_default()
    }

    // --- Navigation and picking --------------------------------------------------------------

    /// Makes `id` the current photo. The cache keeps the mosaics of the new current photo and its
    /// neighbours; the ones that are not cached yet are queued for decoding.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`].
    pub fn set_current(&self, id: PhotoId) -> Result<(), SessionError> {
        let inner = &self.inner;
        let epoch = inner.epoch.load(Ordering::SeqCst);
        let prefetch: Vec<PhotoId> = {
            let mut st = lock(&inner.state);
            let i = *st.index.get(&id).ok_or(SessionError::UnknownPhoto(id))?;
            st.current = Some(i);
            inner.current_hint.store(id.0, Ordering::SeqCst);
            st.cache_trim();
            st.prefetch_candidates()
        };
        inner.enqueue(epoch, &prefetch, JobKind::Prefetch);
        Ok(())
    }

    /// Marks or unmarks a photo as picked.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`].
    pub fn set_picked(&self, id: PhotoId, picked: bool) -> Result<(), SessionError> {
        let mut st = lock(&self.inner.state);
        if !st.index.contains_key(&id) {
            return Err(SessionError::UnknownPhoto(id));
        }
        if picked {
            st.picked.insert(id);
        } else {
            st.picked.remove(&id);
        }
        Ok(())
    }

    /// Picks every photo that can be exported (not the skipped ones).
    pub fn pick_all(&self) {
        let mut st = lock(&self.inner.state);
        let ids: Vec<PhotoId> = st
            .photos
            .iter()
            .filter(|p| !matches!(p.summary.status, PhotoStatus::Skipped { .. }))
            .map(|p| p.id)
            .collect();
        st.picked.extend(ids);
    }

    /// Clears the picked set.
    pub fn clear_picks(&self) {
        lock(&self.inner.state).picked.clear();
    }

    /// The photos a command with this scope acts on, in filmstrip order. Skipped photos are left
    /// out.
    #[must_use]
    pub fn scope_ids(&self, scope: ApplyScope) -> Vec<PhotoId> {
        let st = lock(&self.inner.state);
        st.photos
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.summary.status, PhotoStatus::Skipped { .. }))
            .filter(|(i, p)| match scope {
                ApplyScope::This => st.current == Some(*i),
                ApplyScope::Picked => st.picked.contains(&p.id),
                ApplyScope::All => true,
            })
            .map(|(_, p)| p.id)
            .collect()
    }

    // --- Settings ----------------------------------------------------------------------------

    /// `update_settings`: applies a patch to a photo's settings and returns them resolved.
    ///
    /// Numbers out of range are clamped; a patch that cannot be applied (not a number, a bad
    /// curve, a Pick white spot that is too bright or too dark) changes nothing and returns the
    /// error. A white pick samples the photo's pixels, which decodes the photo again when its
    /// mosaic is not cached.
    ///
    /// Commands may arrive from several threads (the webview sends one per slider move, and a
    /// Pick white takes a while): the patch is applied to the settings the photo has when it is
    /// committed, so two commands on different fields both take effect, whatever their order.
    ///
    /// # Errors
    ///
    /// [`SessionError`] as above, and as [`Session::view`]; [`SessionError::UnknownPhoto`] when
    /// the session was closed meanwhile.
    pub fn update_settings(
        &self,
        id: PhotoId,
        patch: &SettingsPatch,
    ) -> Result<ResolvedSettings, SessionError> {
        let epoch = self.inner.epoch.load(Ordering::SeqCst);
        let mut sample = None;
        let mut attempt = 0;
        loop {
            attempt += 1;
            let view = self.view(id)?;
            let current = view.resolved()?;
            let mut next = view.settings.clone();
            next.apply_patch(patch, Some(&current.seeds()))?;
            // Another command may change this photo's settings while this one works (a Pick
            // white decodes the photo): then the patch is applied again to what is there now.
            // Every retry means another command succeeded; the cap only guards against a loop
            // that something unforeseen would make endless, and then the last attempt wins.
            let how = if attempt < MAX_COMMIT_ATTEMPTS {
                Commit::IfUnchanged
            } else {
                Commit::Always
            };
            if let Some(resolved) = self.commit_settings(epoch, &view, next, how, &mut sample)? {
                return Ok(resolved);
            }
        }
    }

    /// Replaces a photo's settings wholesale (the CLI's `--settings`, a golden manifest): values
    /// are clamped like a patch's.
    ///
    /// # Errors
    ///
    /// As [`Session::update_settings`].
    pub fn replace_settings(
        &self,
        id: PhotoId,
        settings: DevelopSettings,
    ) -> Result<ResolvedSettings, SessionError> {
        let epoch = self.inner.epoch.load(Ordering::SeqCst);
        let view = self.view(id)?;
        let (next, _clamped) = settings.validated()?;
        let mut sample = None;
        // Replacing is the last word: it never has to retry.
        self.commit_settings(epoch, &view, next, Commit::Always, &mut sample)?
            .ok_or(SessionError::UnknownPhoto(id))
    }

    /// Resolves `next` against the photo and stores it. `Ok(None)` means `how` asked for the
    /// settings to be unchanged since `view` and they were not: the caller starts again from the
    /// current ones. `sample` keeps a Pick white sample between attempts (it depends on the
    /// photo's initial gains only).
    fn commit_settings(
        &self,
        epoch: u64,
        view: &PhotoView,
        next: DevelopSettings,
        how: Commit,
        sample: &mut Option<([f64; 3], PickedWhite)>,
    ) -> Result<Option<ResolvedSettings>, SessionError> {
        let id = view.id;
        let mut pick = view.pick;
        match next.white_pick {
            None => pick = None,
            Some(point) => {
                if pick.map(|p| p.point) != Some(point) {
                    let g0 = view
                        .estimates
                        .initial_gains()
                        .map_err(|e| SettingsError::Chain(e.to_string()))?;
                    pick = match *sample {
                        Some((g, picked)) if g == g0 && picked.point == point => Some(picked),
                        _ => {
                            let raw = self.raw_image(id)?;
                            let value = pick_white_sample(&raw, point, g0).ok_or_else(|| {
                                SettingsError::WhitePick(
                                    "That spot is outside the photo. Pick a grey or white area."
                                        .to_owned(),
                                )
                            })?;
                            let picked = PickedWhite {
                                point,
                                sample: value,
                            };
                            *sample = Some((g0, picked));
                            Some(picked)
                        }
                    };
                }
            }
        }
        let picked_white = pick
            .filter(|p| Some(p.point) == next.white_pick)
            .map(|p| p.sample);
        let resolved =
            ResolvedSettings::resolve(id, &next, &view.estimates, &view.profile, picked_white)?;
        let mut st = lock(&self.inner.state);
        if self.inner.epoch.load(Ordering::SeqCst) != epoch {
            return Err(SessionError::UnknownPhoto(id));
        }
        let i = *st.index.get(&id).ok_or(SessionError::UnknownPhoto(id))?;
        let p = &mut st.photos[i];
        if how == Commit::IfUnchanged && p.settings != view.settings {
            return Ok(None);
        }
        p.settings = next;
        if p.pick != pick {
            // The sample is part of what the colour chain renders from.
            p.version += 1;
        }
        p.pick = pick;
        Ok(Some(resolved))
    }

    /// `apply_look`: copies a look and its intensity to the photos in `scope` (This photo, Picked,
    /// All). Other settings are not touched. Returns the photos that changed.
    ///
    /// # Errors
    ///
    /// [`SessionError::Settings`] when the intensity is not a number.
    pub fn apply_look(
        &self,
        scope: ApplyScope,
        look: &LookId,
        intensity: f64,
    ) -> Result<Vec<PhotoId>, SessionError> {
        if !intensity.is_finite() {
            return Err(SettingsError::NotFinite("look_intensity").into());
        }
        let intensity = intensity.clamp(
            crate::settings::LOOK_INTENSITY_RANGE.0,
            crate::settings::LOOK_INTENSITY_RANGE.1,
        );
        let ids = self.scope_ids(scope);
        let mut st = lock(&self.inner.state);
        let mut changed = Vec::new();
        for id in ids {
            if let Some(&i) = st.index.get(&id) {
                let s = &mut st.photos[i].settings;
                if s.look != *look || s.look_intensity != intensity {
                    s.look = look.clone();
                    s.look_intensity = intensity;
                    changed.push(id);
                }
            }
        }
        Ok(changed)
    }

    /// `copy_settings`: the settings of a photo as a patch to paste onto other photos
    /// ([`SettingsPatch::copied_from`]): everything but the crop, the orientation and the Pick
    /// white point, Auto kept as Auto, a white balance that came from a picked white as the
    /// manual Kelvin and tint it resolved to. Nothing is stored: the caller keeps the patch.
    ///
    /// # Errors
    ///
    /// As [`Session::settings`].
    pub fn copy_settings(&self, id: PhotoId) -> Result<SettingsPatch, SessionError> {
        Ok(SettingsPatch::copied_from(&self.settings(id)?))
    }

    /// `paste_settings`: applies `patch` (what [`Session::copy_settings`] made) to every photo in
    /// `ids`, in one call under one lock, so a hundred photos are one change and not a hundred
    /// commands. The crop, the orientation and the Pick white point are never touched, whatever
    /// the patch says. A photo that has no estimates yet (still developing), failed, or is not in
    /// the session is skipped. A photo's own automatic values stay its own: only the settings
    /// change, nothing is rendered here (the caller draws the current photo and the thumbnails
    /// again). Each id counts once, in the order given.
    ///
    /// The patch is checked against every photo before any is changed, so a patch that cannot be
    /// applied changes nothing.
    ///
    /// # Errors
    ///
    /// [`SessionError::Settings`] for a patch that cannot be applied (a value that is not a
    /// number, a partial manual value for a photo whose field is Auto: `copy_settings` makes
    /// whole ones).
    pub fn paste_settings(
        &self,
        ids: &[PhotoId],
        patch: &SettingsPatch,
    ) -> Result<PasteReport, SessionError> {
        let mut patch = patch.clone();
        patch.crop = None;
        patch.orientation = None;
        patch.white_pick = None;
        let mut st = lock(&self.inner.state);
        let mut report = PasteReport::default();
        let mut seen = HashSet::new();
        let mut apply: Vec<(usize, DevelopSettings, Vec<crate::settings::SettingsField>)> =
            Vec::new();
        for &id in ids {
            if !seen.insert(id) {
                continue;
            }
            let ready = st.index.get(&id).copied().filter(|&i| {
                let p = &st.photos[i];
                !matches!(p.summary.status, PhotoStatus::Skipped { .. })
                    && p.meta.is_some()
                    && p.camera.is_some()
                    && p.profile.is_some()
                    && p.effective.is_some()
            });
            let Some(i) = ready else {
                report.skipped.push(id);
                continue;
            };
            let mut next = st.photos[i].settings.clone();
            let done = next.apply_patch(&patch, None)?;
            apply.push((i, next, done.changed));
        }
        for (i, next, changed) in apply {
            let p = &mut st.photos[i];
            report.pasted.push(PastedPhoto {
                id: p.id,
                before: SettingsPatch::of_fields(&p.settings, &changed),
                after: SettingsPatch::of_fields(&next, &changed),
            });
            p.settings = next;
        }
        Ok(report)
    }

    /// Sets the default look: the look of photos opened from now on (a preference).
    pub fn set_default_look(&self, look: LookId) {
        lock(&self.inner.state).default_look = look;
    }

    /// `rerun_auto`: recomputes a photo's automatic estimates from its pixels and sets exposure,
    /// white balance (the light preset, Kelvin and tint and a picked white) and noise reduction
    /// back to Auto. Runs on the calling thread and decodes the photo again when its mosaic is not
    /// cached.
    ///
    /// # Errors
    ///
    /// As [`Session::view`], and [`SessionError::Decode`] or [`SessionError::Estimate`].
    pub fn rerun_auto(&self, id: PhotoId) -> Result<ResolvedSettings, SessionError> {
        let epoch = self.inner.epoch.load(Ordering::SeqCst);
        let view = self.view(id)?;
        let raw = self.raw_image(id)?;
        let model = self.inner.env.illuminants.get(view.profile.id.as_str());
        let estimates = AutoEstimates::compute_with_look(
            &raw,
            &AutoInputs {
                profile: &view.profile,
                illuminant_model: model.map(|m| &**m),
                priors: &self.inner.env.priors,
            },
            view.camera.matched.white_scale,
            self.inner.env.reference_look.as_deref(),
        )?;
        let estimates = Arc::new(estimates);
        let changed = {
            let mut st = lock(&self.inner.state);
            if self.inner.epoch.load(Ordering::SeqCst) != epoch {
                return Err(SessionError::UnknownPhoto(id));
            }
            let i = *st.index.get(&id).ok_or(SessionError::UnknownPhoto(id))?;
            let key = st.photos[i].pool_key.clone();
            {
                let p = &mut st.photos[i];
                p.own = Some(estimates.clone());
                p.own_gen += 1;
                p.effective = Some(estimates);
                p.settings.reset_automatic();
                p.pick = None;
                p.version += 1;
            }
            if let Some(key) = key {
                st.rebuild_pool(&key, &self.inner.env.priors);
            }
            st.refresh_effective(&self.inner.env.priors)
        };
        for other in changed.into_iter().filter(|c| *c != id) {
            self.inner.emit_photo(
                epoch,
                other,
                None,
                SessionEvent::EstimatesChanged { id: other },
            );
        }
        self.settings(id)
    }

    // --- Pixels ------------------------------------------------------------------------------

    /// The decoded raw image of a photo: from the cache when it is there, else decoded now (and
    /// cached when the photo is in the window of the current photo and its neighbours).
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownPhoto`], [`SessionError::Skipped`] and [`SessionError::Decode`].
    pub fn raw_image(&self, id: PhotoId) -> Result<Arc<RawImage>, SessionError> {
        self.inner.raw_image(id)
    }

    /// The ids whose mosaics are cached, in cache order (for tests and diagnostics).
    #[must_use]
    pub fn cached_ids(&self) -> Vec<PhotoId> {
        lock(&self.inner.state)
            .cache
            .entries
            .iter()
            .map(|(id, _)| *id)
            .collect()
    }

    /// The most mosaics the cache ever held at once (for tests).
    #[must_use]
    pub fn max_cached(&self) -> usize {
        lock(&self.inner.state).cache.max_len
    }

    // --- Lifecycle ---------------------------------------------------------------------------

    /// Jobs that are queued or running.
    #[must_use]
    pub fn pending_jobs(&self) -> usize {
        let q = lock(&self.inner.queue);
        q.jobs.len() + q.running
    }

    /// Jobs that have started since the session was created (for tests).
    #[must_use]
    pub fn jobs_started(&self) -> u64 {
        self.inner.jobs_started.load(Ordering::SeqCst)
    }

    /// Files [`Session::raw_image`] has decoded since the session was created, prefetching
    /// included (the development jobs decode on their own and are not counted). For tests.
    #[must_use]
    pub fn raw_decodes(&self) -> u64 {
        self.inner.raw_decodes.load(Ordering::SeqCst)
    }

    /// Waits until no job is queued or running; false when `timeout` ran out first.
    #[must_use]
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut q = lock(&self.inner.queue);
        while !q.jobs.is_empty() || q.running > 0 {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            q = self
                .inner
                .idle
                .wait_timeout(q, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }

    /// The highest photo id the session has handed out, across closes (`None` before the first
    /// photo). Ids only count up, so every photo of a later session is above it: what a renderer
    /// needs to tell the photos of a closed session from the next one's.
    #[must_use]
    pub fn last_photo_id(&self) -> Option<PhotoId> {
        let next = lock(&self.inner.state).next_id;
        (next > 1).then(|| PhotoId(next - 1))
    }

    /// Closes the session (`close_session`): cancels every background job, forgets every photo,
    /// settings and cache. Jobs that are running stop at their next stage; their results are
    /// discarded and no event is sent for them after this returns. The session can be opened
    /// again.
    pub fn close(&self) {
        let inner = &self.inner;
        {
            let _emit = lock(&inner.emit_lock);
            inner.epoch.fetch_add(1, Ordering::SeqCst);
        }
        {
            let mut q = lock(&inner.queue);
            q.jobs.clear();
            if q.running == 0 {
                inner.idle.notify_all();
            }
        }
        inner.current_hint.store(NO_PHOTO, Ordering::SeqCst);
        lock(&inner.state).clear();
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.close();
        {
            let mut q = lock(&self.inner.queue);
            q.shutdown = true;
        }
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.inner.work.notify_all();
        let handles = std::mem::take(&mut *lock(&self.workers));
        let me = std::thread::current().id();
        for h in handles {
            if h.thread().id() != me {
                let _ = h.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Inner: events, jobs
// ---------------------------------------------------------------------------------------------

impl Inner {
    fn live(&self, epoch: u64) -> bool {
        !self.shutdown.load(Ordering::SeqCst) && self.epoch.load(Ordering::SeqCst) == epoch
    }

    /// Sends an event unless the job is cancelled. Returns whether the job is still live.
    fn emit(&self, epoch: u64, event: SessionEvent) -> bool {
        let _guard = lock(&self.emit_lock);
        if !self.live(epoch) {
            return false;
        }
        self.env.events.send(event);
        true
    }

    /// Sends an event about photo `id` unless the job is cancelled or the photo is not in the session any more (it was
    /// removed, [`Session::remove`]); with `incarnation`, also unless it was removed and put back since the job was queued.
    /// Returns whether the job is still live for the photo.
    fn emit_photo(
        &self,
        epoch: u64,
        id: PhotoId,
        incarnation: Option<u32>,
        event: SessionEvent,
    ) -> bool {
        let _guard = lock(&self.emit_lock);
        if !self.live(epoch) || !lock(&self.state).has_photo(id, incarnation) {
            return false;
        }
        self.env.events.send(event);
        true
    }

    /// Records a stage change in the photo's status and sends the event, as one step. False (and nothing
    /// sent) when the job is cancelled or its photo has left the session since the job was queued.
    fn progress(&self, job: &Job, stage: PhotoStage, done: bool, error: Option<String>) -> bool {
        let id = job.photo;
        let _guard = lock(&self.emit_lock);
        if !self.live(job.epoch) {
            return false;
        }
        {
            let mut st = lock(&self.state);
            let Some(&i) = st.index.get(&id) else {
                return false;
            };
            let p = &mut st.photos[i];
            if p.incarnation != job.incarnation {
                return false;
            }
            if let Some(reason) = &error {
                p.summary.status = PhotoStatus::Skipped {
                    reason: reason.clone(),
                };
            } else if stage == PhotoStage::Thumbnail && done {
                p.summary.status = PhotoStatus::Ready;
            } else if !done {
                p.summary.status = PhotoStatus::Developing { stage };
            }
        }
        self.env
            .events
            .send(SessionEvent::PhotoProgress(PhotoProgress {
                id,
                stage,
                done,
                error,
            }));
        true
    }

    fn new_photo(
        &self,
        id: PhotoId,
        path: PathBuf,
        canon: PathBuf,
        default_look: &LookId,
        probe: Result<RawMetadata, RawError>,
    ) -> Photo {
        let file_name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let settings = DevelopSettings::with_look(default_look.clone());
        match probe {
            Ok(meta) => {
                let camera = PhotoCamera::new(
                    &self.env.profiles,
                    &meta,
                    self.env.forced_profile.as_deref(),
                );
                let profile = self.env.profiles.resolve(&camera.matched).clone();
                let (w, h) = meta_image_size(&meta);
                let summary = PhotoSummary {
                    id,
                    file_name,
                    camera: camera.info(),
                    iso: meta.iso.unwrap_or(0),
                    shutter_s: meta.exposure_time_s.unwrap_or(0.0),
                    aperture_f: meta.f_number.unwrap_or(0.0),
                    captured_at: meta
                        .capture_time
                        .map(|t| {
                            let s = t.to_iso8601();
                            s.get(..19).unwrap_or(&s).to_owned()
                        })
                        .unwrap_or_default(),
                    width_px: u32::try_from(w).unwrap_or(u32::MAX),
                    height_px: u32::try_from(h).unwrap_or(u32::MAX),
                    status: PhotoStatus::Waiting,
                    picked: false,
                    reduced: camera.reduced,
                };
                Photo {
                    id,
                    path,
                    canon,
                    incarnation: 0,
                    summary,
                    burst: Some(burst_frame(&meta)),
                    // Files that do not say their ISO cannot be told apart by it: each keeps its
                    // own noise estimate rather than pooling with files of another gain.
                    pool_key: meta.iso.filter(|iso| *iso > 0).map(|_| pool_key(&meta)),
                    meta: Some(Arc::new(meta)),
                    camera: Some(camera),
                    profile: Some(profile),
                    own: None,
                    effective: None,
                    noise_version: 0,
                    own_gen: 0,
                    settings,
                    pick: None,
                    version: 0,
                }
            }
            Err(error) => Photo {
                id,
                path,
                canon,
                incarnation: 0,
                summary: PhotoSummary {
                    id,
                    file_name,
                    camera: CameraInfo::unknown(),
                    iso: 0,
                    shutter_s: 0.0,
                    aperture_f: 0.0,
                    captured_at: String::new(),
                    width_px: 0,
                    height_px: 0,
                    status: PhotoStatus::Skipped {
                        reason: error.to_string(),
                    },
                    picked: false,
                    reduced: false,
                },
                meta: None,
                camera: None,
                profile: None,
                pool_key: None,
                burst: None,
                own: None,
                effective: None,
                noise_version: 0,
                own_gen: 0,
                settings,
                pick: None,
                version: 0,
            },
        }
    }

    fn enqueue(&self, epoch: u64, ids: &[PhotoId], kind: JobKind) {
        if ids.is_empty() {
            return;
        }
        // A photo that has left the session since is not queued; the number a photo has now is the one its job carries.
        let queued: Vec<(PhotoId, u32)> = {
            let st = lock(&self.state);
            ids.iter()
                .filter_map(|&id| st.photo(id).map(|p| (id, p.incarnation)))
                .collect()
        };
        let mut q = lock(&self.queue);
        for (photo, incarnation) in queued {
            let dup = q.jobs.iter().any(|j| j.photo == photo && j.kind == kind);
            if !dup {
                q.jobs.push_back(Job {
                    epoch,
                    photo,
                    kind,
                    incarnation,
                });
            }
        }
        drop(q);
        self.work.notify_all();
    }

    /// The body of a decode thread.
    fn worker(self: &Arc<Self>) {
        loop {
            let job = {
                let mut q = lock(&self.queue);
                loop {
                    if q.shutdown {
                        return;
                    }
                    if let Some(job) = self.take_job(&mut q) {
                        q.running += 1;
                        break job;
                    }
                    q = self.work.wait(q).unwrap_or_else(PoisonError::into_inner);
                }
            };
            if self.live(job.epoch) {
                self.jobs_started.fetch_add(1, Ordering::SeqCst);
                let run =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job.kind {
                        JobKind::Develop => self.develop(&job),
                        JobKind::Prefetch => self.prefetch(job.photo),
                    }));
                if let Err(payload) = run {
                    let message = payload
                        .downcast_ref::<&str>()
                        .map(|m| (*m).to_owned())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_default();
                    tracing::error!(photo = job.photo.0, panic = %message, "a decode job panicked");
                    // Only a development job decides a photo's fate; a failed prefetch just
                    // means the next render decodes the file again. Reporting must not take the
                    // thread down either (a sink that panics), or the pool would shrink.
                    if job.kind == JobKind::Develop {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.progress(
                                &job,
                                PhotoStage::Decode,
                                true,
                                Some(
                                    "Yellowmoon hit a problem it didn't expect with this file. Skip it and open the others."
                                        .to_owned(),
                                ),
                            )
                        }));
                    }
                }
            }
            let mut q = lock(&self.queue);
            q.running -= 1;
            if q.jobs.is_empty() && q.running == 0 {
                self.idle.notify_all();
            }
        }
    }

    /// The next job: the current photo's development first, then the oldest development, then a
    /// prefetch (the current photo's first).
    fn take_job(&self, q: &mut Queue) -> Option<Job> {
        let current = self.current_hint.load(Ordering::SeqCst);
        let pos = q
            .jobs
            .iter()
            .position(|j| j.photo.0 == current && j.kind == JobKind::Develop)
            .or_else(|| q.jobs.iter().position(|j| j.kind == JobKind::Develop))
            .or_else(|| q.jobs.iter().position(|j| j.photo.0 == current))
            .or_else(|| (!q.jobs.is_empty()).then_some(0))?;
        q.jobs.remove(pos)
    }

    /// Decodes a photo so that its mosaic is cached, if it still belongs in the cache. The job
    /// was queued when the photo was in the window of the current photo; stepping through photos
    /// faster than they decode leaves jobs for photos the person has already passed, and
    /// decoding those would only be thrown away again (`cache_offer` keeps nothing outside the
    /// window) while the photo they reached waits for the same cores.
    fn prefetch(&self, id: PhotoId) {
        let wanted = {
            let st = lock(&self.state);
            st.in_window(id) && st.cache_get(id).is_none()
        };
        if wanted {
            let _ = self.raw_image(id);
        }
    }

    /// Decode, estimate, thumbnail for one photo.
    fn develop(&self, job: &Job) {
        let (epoch, id) = (job.epoch, job.photo);
        let snapshot = {
            let st = lock(&self.state);
            st.photo(id)
                .filter(|p| p.incarnation == job.incarnation)
                .and_then(|p| {
                    Some((
                        p.path.clone(),
                        p.camera.clone()?,
                        p.profile.clone()?,
                        p.settings.clone(),
                    ))
                })
        };
        let Some((path, camera, profile, settings)) = snapshot else {
            return;
        };
        let fail = |stage: PhotoStage, reason: String| {
            self.progress(job, stage, true, Some(reason));
        };

        // Decode.
        if !self.progress(job, PhotoStage::Decode, false, None) {
            return;
        }
        let started = Instant::now();
        let raw = match ym_raw::decode::decode(&path) {
            Ok(raw) => Arc::new(raw),
            Err(e) => return fail(PhotoStage::Decode, e.to_string()),
        };
        tracing::debug!(
            photo = id.0,
            ms = started.elapsed().as_millis() as u64,
            "decoded"
        );
        if !self.progress(job, PhotoStage::Decode, true, None)
            || !self.progress(job, PhotoStage::Estimate, false, None)
        {
            return;
        }

        // Estimates.
        let started = Instant::now();
        let model = self.env.illuminants.get(profile.id.as_str());
        let estimates = match AutoEstimates::compute_with_look(
            &raw,
            &AutoInputs {
                profile: &profile,
                illuminant_model: model.map(|m| &**m),
                priors: &self.env.priors,
            },
            camera.matched.white_scale,
            self.env.reference_look.as_deref(),
        ) {
            Ok(e) => Arc::new(e),
            Err(e) => return fail(PhotoStage::Estimate, e.to_string()),
        };
        tracing::debug!(
            photo = id.0,
            ms = started.elapsed().as_millis() as u64,
            "estimated"
        );
        let (changed, served) = {
            let mut st = lock(&self.state);
            if !self.live(epoch) {
                return;
            }
            let Some(&i) = st.index.get(&id) else {
                return;
            };
            if st.photos[i].incarnation != job.incarnation {
                return;
            }
            let key = st.photos[i].pool_key.clone();
            {
                let p = &mut st.photos[i];
                p.own = Some(estimates.clone());
                p.own_gen += 1;
                p.effective = Some(estimates);
                p.version += 1;
            }
            if let Some(key) = key {
                st.rebuild_pool(&key, &self.env.priors);
            }
            let changed = st.refresh_effective(&self.env.priors);
            st.cache_offer(id, raw.clone());
            let served = st.photos[i].effective.clone();
            (changed, served)
        };
        for other in changed.into_iter().filter(|c| *c != id) {
            self.emit_photo(
                epoch,
                other,
                None,
                SessionEvent::EstimatesChanged { id: other },
            );
        }
        let Some(served) = served else { return };
        if !self.progress(job, PhotoStage::Estimate, true, None)
            || !self.progress(job, PhotoStage::Thumbnail, false, None)
        {
            return;
        }

        // Thumbnail, with the settings the photo has now (the look may have been applied while
        // it was being decoded).
        if let Some(renderer) = &self.env.thumbnails {
            let settings = lock(&self.state)
                .photo(id)
                .map_or(settings, |p| p.settings.clone());
            let cancelled = || !self.live(epoch);
            let started = Instant::now();
            let result = renderer.render(&ThumbnailInput {
                id,
                raw: &raw,
                camera: &camera,
                profile: &profile,
                estimates: &served,
                settings: &settings,
                cancelled: &cancelled,
            });
            match result {
                Ok(png_bytes) => {
                    tracing::debug!(
                        photo = id.0,
                        ms = started.elapsed().as_millis() as u64,
                        "thumbnail"
                    );
                    if !self.emit_photo(
                        epoch,
                        id,
                        Some(job.incarnation),
                        SessionEvent::ThumbnailReady(ThumbnailReady { id, png_bytes }),
                    ) {
                        return;
                    }
                }
                Err(reason) => return fail(PhotoStage::Thumbnail, reason),
            }
        }
        self.progress(job, PhotoStage::Thumbnail, true, None);
    }

    fn raw_image(&self, id: PhotoId) -> Result<Arc<RawImage>, SessionError> {
        // Before anything is read: a session closed while the file is decoded must not receive
        // the image.
        let epoch = self.epoch.load(Ordering::SeqCst);
        loop {
            let path = {
                let st = lock(&self.state);
                let p = st.photo(id).ok_or(SessionError::UnknownPhoto(id))?;
                if let PhotoStatus::Skipped { reason } = &p.summary.status {
                    return Err(SessionError::Skipped(reason.clone()));
                }
                if let Some(raw) = st.cache_get(id) {
                    return Ok(raw);
                }
                p.path.clone()
            };
            // One decode per photo at a time: the viewport, the histogram, a tile for each look and
            // the prefetch all ask at once, and each holds a whole mosaic (122 MB at 61 MP) while
            // it works. The others wait for the leader's image and share it.
            let lead = match self.decoding.join(id) {
                Role::Wait(flight) => match flight.wait() {
                    Some(raw) => return Ok(raw),
                    // The leader failed: look again, and lead (and fail with our own error).
                    None => continue,
                },
                Role::Lead(lead) => lead,
            };
            // A decode that finished between our look at the cache and our turn to lead.
            if let Some(raw) = lock(&self.state).cache_get(id) {
                lead.finish(Some(raw.clone()));
                return Ok(raw);
            }
            self.raw_decodes.fetch_add(1, Ordering::SeqCst);
            let decoded = Arc::new(ym_raw::decode::decode(&path)?);
            let raw = {
                let mut st = lock(&self.state);
                // The development job may have cached it while we read the file: one image.
                let raw = st.cache_get(id).unwrap_or(decoded);
                if self.epoch.load(Ordering::SeqCst) == epoch {
                    st.cache_offer(id, raw.clone());
                }
                raw
            };
            lead.finish(Some(raw.clone()));
            return Ok(raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    /// Runs `work` for `key` the way `Inner::raw_image` does: lead it, or wait for the leader's
    /// result and lead again if the leader failed.
    fn run(
        flights: &SingleFlight<u32, Arc<String>>,
        key: u32,
        work: impl Fn() -> Option<Arc<String>>,
    ) -> Option<Arc<String>> {
        loop {
            match flights.join(key) {
                Role::Wait(flight) => match flight.wait() {
                    Some(value) => return Some(value),
                    None => continue,
                },
                Role::Lead(lead) => {
                    let value = work();
                    lead.finish(value.clone());
                    return value;
                }
            }
        }
    }

    #[test]
    fn callers_of_one_key_share_the_result_of_one_run() {
        // The leader takes a while, so that every other caller arrives while it runs.
        let flights = Arc::new(SingleFlight::new());
        let runs = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (flights, runs, barrier) = (
                    Arc::clone(&flights),
                    Arc::clone(&runs),
                    Arc::clone(&barrier),
                );
                std::thread::spawn(move || {
                    barrier.wait();
                    run(&flights, 7, || {
                        runs.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(400));
                        Some(Arc::new("decoded".to_owned()))
                    })
                })
            })
            .collect();
        let results: Vec<Arc<String>> = threads
            .into_iter()
            .map(|t| t.join().expect("thread").expect("a result"))
            .collect();
        assert_eq!(runs.load(Ordering::SeqCst), 1, "one run for eight callers");
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
        // The run is over: the next caller starts a new one, and another key is independent.
        run(&flights, 7, || {
            runs.fetch_add(1, Ordering::SeqCst);
            Some(Arc::new("again".to_owned()))
        });
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert!(
            lock(&flights.flights).is_empty(),
            "nothing is left in flight"
        );
    }

    #[test]
    fn a_leader_that_fails_or_panics_does_not_leave_the_others_waiting() {
        let flights = Arc::new(SingleFlight::<u32, Arc<String>>::new());
        // A failure: the waiter leads its own run, which succeeds.
        let Role::Lead(lead) = flights.join(1) else {
            panic!("the first caller leads");
        };
        let waiter = {
            let flights = Arc::clone(&flights);
            std::thread::spawn(move || run(&flights, 1, || Some(Arc::new("mine".to_owned()))))
        };
        std::thread::sleep(Duration::from_millis(100));
        lead.finish(None);
        assert_eq!(
            waiter
                .join()
                .expect("thread")
                .as_deref()
                .map(String::as_str),
            Some("mine")
        );

        // A panic: the lead is dropped while unwinding and ends the run as failed.
        let panicked = {
            let flights = Arc::clone(&flights);
            std::thread::spawn(move || {
                let Role::Lead(_lead) = flights.join(2) else {
                    panic!("the first caller leads");
                };
                panic!("the decode panicked");
            })
        };
        assert!(panicked.join().is_err());
        let value = run(&flights, 2, || Some(Arc::new("after".to_owned())));
        assert_eq!(value.as_deref().map(String::as_str), Some("after"));
        assert!(lock(&flights.flights).is_empty());
    }
}

#[cfg(test)]
mod removal_tests {
    //! The estimates a burst shares, after a frame leaves it and when it comes back. A synthetic Sony frame never makes
    //! the highlight guard lower its exposure, so the frames of a burst all have the camera's gain and share it whatever
    //! leaves; here the frames' own exposures are set to different numbers (as the guard would make them) to see the
    //! median follow the set of photos.

    use super::*;
    use ym_raw::testgen::ArwBuilder;

    fn write_frame(dir: &Path, name: &str, second: u32) {
        let (w, h) = (128u32, 96u32);
        ArwBuilder::new("ILCE-7RM4", w, h)
            .black([512; 4])
            .white([16300; 3])
            .wb(Some([2048, 1024, 1024, 1536]))
            .exposure_comp(0, 10)
            .iso(400)
            .capture_time(&format!("2026:10:04 12:03:{second:02}"), "00", None)
            .write_to(&dir.join(name), &vec![4000u16; (w * h) as usize])
            .expect("write a synthetic ARW");
    }

    /// Gives the photos (in filmstrip order) these exposures as their own, and shares them again.
    fn set_own_evs(session: &Session, evs: &[f64]) {
        let mut st = lock(&session.inner.state);
        for (i, ev) in evs.iter().enumerate() {
            let mut own = (**st.photos[i].own.as_ref().expect("estimated")).clone();
            own.exposure.own_ev = *ev;
            own.exposure.ev = *ev;
            own.color.ev_auto = *ev;
            st.photos[i].own = Some(Arc::new(own));
            st.photos[i].own_gen += 1;
        }
        st.refresh_effective(&session.inner.env.priors);
    }

    #[test]
    fn a_burst_shares_the_median_of_the_frames_that_are_left_and_gets_the_old_one_back_with_undo() {
        let dir = tempfile::tempdir().expect("dir");
        // Five frames a second apart: one burst.
        for i in 0..5 {
            write_frame(dir.path(), &format!("f{i}.ARW"), 10 + i);
        }
        let (sink, rx) = EventSink::channel();
        let env = SessionEnv::load(&find_data_dir().expect("data"))
            .expect("env")
            .with_decode_threads(2)
            .with_events(sink);
        let session = Session::new(env);
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        assert!(session.wait_idle(Duration::from_secs(60)));
        let ids = session.photo_ids();
        assert_eq!(ids.len(), 5);

        // Own exposures 0, 1, 2, 3 and 10 stops: the burst median is 2.
        let evs = [0.0, 1.0, 2.0, 3.0, 10.0];
        set_own_evs(&session, &evs);
        let shared = |id: PhotoId| session.estimates(id).expect("estimates").exposure.ev;
        let own = |id: PhotoId| session.estimates(id).expect("estimates").exposure.own_ev;
        for id in &ids {
            assert_eq!(shared(*id), 2.0, "{id}");
        }
        let _ = rx.try_iter().count();

        // The two darkest frames leave: the median of 2, 3 and 10 is 3.
        let out = session.remove(&[ids[0], ids[1]]);
        assert_eq!(out.removed, vec![ids[0], ids[1]]);
        for id in &ids[2..] {
            assert_eq!(shared(*id), 3.0, "{id}");
        }
        assert_eq!(own(ids[4]), 10.0, "a frame keeps its own exposure");
        // The three frames that stayed are told, and nobody else is.
        assert_eq!(out.estimates_changed, vec![ids[2], ids[3], ids[4]]);
        let told: Vec<PhotoId> = rx
            .try_iter()
            .filter_map(|e| match e {
                SessionEvent::EstimatesChanged { id } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(told, out.estimates_changed);
        // What the render reads carries the shared value.
        let est = session.estimates(ids[2]).expect("estimates");
        assert_eq!(est.color.ev_auto, 3.0);

        // A burst with one frame left is that frame's own exposure.
        let out = session.remove(&[ids[2], ids[3]]);
        assert_eq!(out.estimates_changed, vec![ids[4]]);
        assert_eq!(shared(ids[4]), 10.0);

        // Undo brings the frames back, and the median with them: the photo that stayed is told once more.
        let _ = rx.try_iter().count();
        let back = session.restore_removed();
        assert_eq!(back.restored, vec![ids[2], ids[3]]);
        assert_eq!(back.estimates_changed, vec![ids[4]]);
        assert_eq!(shared(ids[4]), 3.0);
        assert_eq!(shared(ids[2]), 3.0);
        // The restored frames are in the burst again; none of them is "told" about its own return.
        let told: Vec<PhotoId> = rx
            .try_iter()
            .filter_map(|e| match e {
                SessionEvent::EstimatesChanged { id } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(told, vec![ids[4]]);
    }

    #[test]
    fn the_burst_of_two_frames_that_lost_one_is_alone_and_the_other_burst_is_not_told() {
        let dir = tempfile::tempdir().expect("dir");
        // A burst of three, and a lone frame half a minute later.
        for (name, second) in [("a.ARW", 10), ("b.ARW", 11), ("c.ARW", 12), ("z.ARW", 50)] {
            write_frame(dir.path(), name, second);
        }
        let env = SessionEnv::load(&find_data_dir().expect("data"))
            .expect("env")
            .with_decode_threads(1);
        let session = Session::new(env);
        session.open(&[dir.path().to_path_buf()], OpenOptions::default());
        assert!(session.wait_idle(Duration::from_secs(60)));
        let ids = session.photo_ids();
        set_own_evs(&session, &[0.0, 4.0, 8.0, 1.0]);
        let shared = |id: PhotoId| session.estimates(id).expect("estimates").exposure.ev;
        assert_eq!([shared(ids[0]), shared(ids[3])], [4.0, 1.0]);
        // The lone frame leaves: the burst is as it was and nobody is told.
        let out = session.remove(&[ids[3]]);
        assert!(out.estimates_changed.is_empty(), "{out:?}");
        assert_eq!(shared(ids[0]), 4.0);
        // A frame of the burst leaves: the two left share their median.
        let out = session.remove(&[ids[0]]);
        assert_eq!(out.estimates_changed, vec![ids[1], ids[2]]);
        assert_eq!(shared(ids[1]), 6.0);
    }
}
