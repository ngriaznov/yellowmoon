//! The one function that exports a photo: [`export_photo`] (`docs/09-export.md`: "App and CLI share
//! one code path, so the same inputs and settings give byte-identical files").
//!
//! The app's export queue ([`crate::export_queue`]) and the command-line tool (`ym render`) both
//! call it with an [`ExportPhoto`] (a photo as the render sees it, settings included), the
//! [`ExportParams`] of the job and an [`ExportControl`] (cancel and progress). Everything that
//! decides the bytes is in here or below it: the render, the metadata, the file name, the writer.
//! The front ends only choose the photos and the settings, so a job exported through the app and
//! the same job through the CLI are the same bytes (`tests/export.rs` checks it, across threads and
//! across processes).
//!
//! # What an export does
//!
//! 1. **Checks** that the look (and the imported display LUT, if one is chosen) can be loaded: an
//!    export never quietly falls back to Neutral the way the viewport does, because the file's XMP
//!    says which look made it.
//! 2. **Names** the file from the template and resolves the destination folder (`docs/09` Options).
//!    With "If the file exists: Skip" and a file in the way nothing is rendered at all.
//! 3. **Renders** at full resolution, tile by tile, on the render scheduler's export worker, which
//!    gives way to the viewport between tiles ([`Renderer::spawn_full_reporting`]):
//!    * JPEG, TIFF and PNG: C1 to C12 in the primaries of the chosen colour space with the
//!      camera's and the user's orientation, straighten and crop baked in
//!      ([`FullTarget::DisplayLinear`]). The writer resizes in linear light (Lanczos-3) when a long
//!      edge is asked for, **sharpens the picture at the size it writes** (output sharpening,
//!      `DevelopSettings::output_sharpening`, default 100 %, `docs/algorithms/04`: the screen
//!      preset at the scale of the written long edge over the rendered one; 0 is no stage and the
//!      bytes of an export without it) and does C13 (the transfer function and, for 8 bit, the
//!      dither).
//!    * DNG (**no sharpening of any kind**: both renders run with R8 capture sharpening off, and
//!      output sharpening is never applied): C1 to C11 decoded to linear AWG4 in the sensor's
//!      orientation ([`FullTarget::DngLinearAwg4`]) and the sRGB preview of what the user sees
//!      ([`FullTarget::DngPreview`]). The `Orientation` tag carries the camera's and the user's
//!      turns, the default crop tags the user's crop, so other editors can undo both. With a look
//!      that carries its own display rendering (LogC3 to Rec.709), or an imported display LUT, the
//!      pixels are the scene values ACES 2 turns into that picture, so the DNG is close to, not the
//!      same as, the picture of the app (`docs/09`, "What a DNG carries").
//! 4. **Writes** through `ym_export::write_export`: a temporary file in the destination folder, then
//!    a rename; a cancel or an error deletes the temporary file.
//!
//! # Metadata
//!
//! EXIF comes from the probe's [`RawMetadata`] (never copied from the
//! file), with the GPS position left out when the preference "Remove location from exports" is on.
//! The XMP carries the `ym:` properties of `docs/09`: the pipeline version, the camera profile with
//! its version and how it was matched, the look with its SHA-256 and intensity, the display
//! transform, the models (none yet: the classical demosaic has none) and the **pinned settings** as
//! JSON ([`ResolvedSettings::pinned_settings`]: exposure, sharpening and noise reduction as the
//! numbers they resolved to, white balance as it was set). All dates are the capture time, never
//! "now".
//!
//! # Time and size estimates
//!
//! [`ExportSpeed`] keeps the render and encode speed in seconds per megapixel, starting from the
//! numbers measured in `docs/01` and `docs/09` and replaced by what this machine measures as photos
//! are exported; [`estimate`] turns it and `ym_export::estimate_bytes_with_detail` into the "≈ 120 MB"
//! and the seconds of the Export panel. The size of a file depends on how much fine detail the
//! picture carries, so each photo brings the detail statistic of its newest thumbnail
//! ([`Renderer::thumbnail_detail`](crate::pipeline::Renderer::thumbnail_detail)); a photo without a
//! thumbnail yet is estimated as a picture of typical detail.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use ym_color::display::OutputTarget;
use ym_export::dng::{CropRect, DngReport, FittedProfile, compose_orientation};
use ym_export::exif::{CaptureTime, ExifInfo, Gps};
use ym_export::naming::{self, NameFields, Placement};
use ym_export::request::{BandRequest, PictureStages, write_export_bands, write_export_with};
use ym_export::xmp::{XmpInfo, YmInfo};
use ym_export::{
    ColorSpace, DetailStat, DngInputs, ExportDestination, ExportFormat, ExportHooks, ExportOptions,
    ExportRequest, FormatKind, IfExists, ImageRef, Metadata, PixelEncoding,
    estimate_bytes_with_detail, write_export,
};
use ym_gpu::queue::{CancelToken, JobError};
use ym_looks::library::LookId;
use ym_profiles::detect::MatchKind;
use ym_raw::decode::RawImage;
use ym_raw::metadata::RawMetadata;

use crate::auto::meta_image_size;
use crate::camera::PhotoCamera;
use crate::pipeline::frame::TailRequest;
use crate::pipeline::{
    ComputeDevice, DisplayChoice, FullImage, FullRequest, FullTarget, PhotoSource, RenderError,
    RenderReport, Renderer,
};
use crate::session::{PhotoId, PhotoView, Session, SessionError};
use crate::settings::{DevelopSettings, ResolvedSettings, SettingsError};
use crate::version::{PIPELINE_VERSION, app_version};

/// `<source folder> export` next to the source folder of a photo, spelled the way the path gives
/// it (symbolic links are not resolved). The one rule of the Export panel, the command `ym render`
/// is shown as and the files themselves; see [`ym_export::naming::default_destination`].
pub use ym_export::naming::default_destination;

/// The note on a photo that was numbered because an earlier photo of the same job has the name its
/// template gives (see [`WrittenFiles`]).
pub const SAME_NAME_IN_JOB_NOTE: &str = "Another photo of this export has the same file name, so this one was saved with a number instead of replacing it.";

/// Why a photo could not be exported. The text is plain words with a next step.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The job was cancelled; the partial file was removed.
    #[error("The export was cancelled and the partial file was removed.")]
    Cancelled,
    /// The look or the display LUT cannot be used ("File missing or changed. Import again.").
    #[error("{0}")]
    Look(String),
    /// The photo could not be rendered.
    #[error(transparent)]
    Render(RenderError),
    /// The file could not be written.
    #[error(transparent)]
    Write(ym_export::ExportError),
    /// The photo is not in the session, is still being developed or was skipped.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// The settings could not be resolved.
    #[error(transparent)]
    Settings(#[from] SettingsError),
    /// The options cannot be used.
    #[error("{0}")]
    Options(String),
    /// The photo's file could not be decoded again.
    #[error("{0}")]
    Decode(String),
}

impl From<ym_export::ExportError> for ExportError {
    fn from(e: ym_export::ExportError) -> Self {
        match e {
            ym_export::ExportError::Cancelled => Self::Cancelled,
            other => Self::Write(other),
        }
    }
}

impl From<RenderError> for ExportError {
    fn from(e: RenderError) -> Self {
        match e {
            RenderError::Cancelled => Self::Cancelled,
            RenderError::Look(m) => Self::Look(m),
            other => Self::Render(other),
        }
    }
}

impl ExportError {
    /// Whether the export was cancelled rather than failed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// A short machine-readable kind, for the CLI's JSON (`"error": { "kind": … }`). A photo that a
    /// strict policy (`--require-gpu`) refused to render on the CPU after a device error is `gpu`.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Look(_) => "look",
            Self::Render(RenderError::Gpu(_)) => "gpu",
            Self::Render(_) => "render",
            Self::Write(_) => "write",
            Self::Session(SessionError::Skipped(_)) => "unsupported",
            Self::Session(_) => "session",
            Self::Settings(_) | Self::Options(_) => "usage",
            Self::Decode(_) => "decode",
        }
    }

    /// The CLI exit code: 3 for a look that is missing, 2 for a mistake in what was asked, else 1.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Look(_) => 3,
            Self::Options(_) | Self::Settings(_) => 2,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Speed
// ---------------------------------------------------------------------------------------------

/// Starting render speed in seconds per megapixel of output: 17.44 s for a 24 MP frame, the CPU
/// render measured in `docs/01` (section Preview strategy, "Measured"). Replaced by the measured
/// speed of this machine after the first exported photo.
pub const RENDER_S_PER_MP_PRIOR: f64 = 17.44 / 24.0;

/// Starting encode speed per format in seconds per megapixel of output, from the 24 MP release
/// timings in `docs/09` (review notes: JPEG q92 5.9 s, TIFF 16-bit LZW 10 s, PNG 16-bit 9.9 s);
/// DNG (uncompressed 16-bit strips and a JPEG preview) is taken as the mean of the three.
pub const ENCODE_S_PER_MP_PRIOR: [f64; 4] = [5.9 / 24.0, 10.0 / 24.0, 9.9 / 24.0, 0.35];

fn format_index(kind: FormatKind) -> usize {
    match kind {
        FormatKind::Jpeg => 0,
        FormatKind::Tiff => 1,
        FormatKind::Png => 2,
        FormatKind::Dng => 3,
    }
}

/// What one device (the CPU path or the export tail on the graphics card) is measured at.
#[derive(Debug, Clone, Copy)]
struct DeviceSpeed {
    render_s_per_mp: f64,
    render_samples: u32,
    encode_s_per_mp: [f64; 4],
    encode_samples: [u32; 4],
    /// The time of a whole raster export of each format per megapixel of the rendered picture, the
    /// writing inside it: on the device the writer pulls the picture while the tail renders it, so
    /// the two overlap and the encode must not be added again. Only a format that has been
    /// measured this way is estimated from it; the others keep `render + encode`. Always empty for
    /// the CPU path (it renders the whole frame first and encodes it after).
    combined_s_per_mp: [f64; 4],
    combined_samples: [u32; 4],
}

impl DeviceSpeed {
    const PRIOR: Self = Self {
        render_s_per_mp: RENDER_S_PER_MP_PRIOR,
        render_samples: 0,
        encode_s_per_mp: ENCODE_S_PER_MP_PRIOR,
        encode_samples: [0; 4],
        combined_s_per_mp: [0.0; 4],
        combined_samples: [0; 4],
    };
}

#[derive(Debug, Clone, Copy)]
struct SpeedState {
    /// The CPU reference path, and the device path.
    sets: [DeviceSpeed; 2],
}

fn device_index(device: ComputeDevice) -> usize {
    match device {
        ComputeDevice::Cpu => 0,
        ComputeDevice::Gpu => 1,
    }
}

/// How fast this machine renders and encodes, in seconds per megapixel, for the CPU path and for
/// the export tail on the graphics card, each measured on its own.
///
/// Starts from the numbers measured on the development machine ([`RENDER_S_PER_MP_PRIOR`],
/// [`ENCODE_S_PER_MP_PRIOR`]) for both: no speed of the graphics card has been measured yet (the
/// Mac checks of `docs/STATUS.md`, M0-H4, do), so the estimate for the device path starts as the
/// CPU's. Every exported photo then replaces the numbers of the path it took: the first sample
/// outright, later ones as a running mean that weighs the newest half. A photo that fell back to
/// the CPU teaches the CPU numbers and leaves the device's alone.
///
/// Every reader names the path it reads ([`ComputeDevice`]); nothing holds a "current path" (the
/// estimate used to select it as a side effect, which a second reader could not see). The caller
/// that estimates picks the path the renderer would take
/// ([`Renderer::export_tail_enabled`](crate::pipeline::Renderer::export_tail_enabled)).
#[derive(Debug)]
pub struct ExportSpeed {
    state: Mutex<SpeedState>,
}

impl Default for ExportSpeed {
    fn default() -> Self {
        Self::new()
    }
}

/// Samples smaller than this say nothing about speed (start-up costs dominate).
const MIN_SAMPLE_MP: f64 = 0.05;

impl ExportSpeed {
    /// The starting speed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SpeedState {
                sets: [DeviceSpeed::PRIOR; 2],
            }),
        }
    }

    fn st(&self) -> std::sync::MutexGuard<'_, SpeedState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn blend(old: f64, samples: u32, new: f64) -> f64 {
        if samples == 0 {
            new
        } else {
            0.5 * old + 0.5 * new
        }
    }

    /// Records that rendering `megapixels` took `seconds` on the CPU path.
    pub fn record_render(&self, megapixels: f64, seconds: f64) {
        self.record_render_on(ComputeDevice::Cpu, megapixels, seconds);
    }

    /// Records that rendering `megapixels` took `seconds` on `device`.
    pub fn record_render_on(&self, device: ComputeDevice, megapixels: f64, seconds: f64) {
        if megapixels >= MIN_SAMPLE_MP && seconds.is_finite() && seconds > 0.0 {
            let mut st = self.st();
            let set = &mut st.sets[device_index(device)];
            set.render_s_per_mp = Self::blend(
                set.render_s_per_mp,
                set.render_samples,
                seconds / megapixels,
            );
            set.render_samples += 1;
        }
    }

    /// Records that encoding `megapixels` of `format` took `seconds` on the CPU path.
    pub fn record_encode(&self, format: FormatKind, megapixels: f64, seconds: f64) {
        if megapixels >= MIN_SAMPLE_MP && seconds.is_finite() && seconds > 0.0 {
            let i = format_index(format);
            let mut st = self.st();
            let set = &mut st.sets[0];
            set.encode_s_per_mp[i] = Self::blend(
                set.encode_s_per_mp[i],
                set.encode_samples[i],
                seconds / megapixels,
            );
            set.encode_samples[i] += 1;
        }
    }

    /// Records that a photo of `megapixels` (the rendered picture) took `seconds` in all on the
    /// device path as a file of `format`, rendering and writing together (the writer pulls the
    /// picture while the device makes it, so the two cannot be told apart). Only `format` is
    /// estimated from it from now on: the other formats keep their own numbers, since the time of
    /// a slow encoder (PNG) says nothing about a fast one (JPEG).
    pub fn record_overlapped(&self, format: FormatKind, megapixels: f64, seconds: f64) {
        if megapixels >= MIN_SAMPLE_MP && seconds.is_finite() && seconds > 0.0 {
            let i = format_index(format);
            let mut st = self.st();
            let set = &mut st.sets[1];
            set.combined_s_per_mp[i] = Self::blend(
                set.combined_s_per_mp[i],
                set.combined_samples[i],
                seconds / megapixels,
            );
            set.combined_samples[i] += 1;
        }
    }

    /// The render speed of `device` now, in seconds per megapixel.
    #[must_use]
    pub fn render_s_per_mp(&self, device: ComputeDevice) -> f64 {
        self.st().sets[device_index(device)].render_s_per_mp
    }

    /// The encode speed of a format on `device` now, in seconds per megapixel.
    #[must_use]
    pub fn encode_s_per_mp(&self, device: ComputeDevice, format: FormatKind) -> f64 {
        self.st().sets[device_index(device)].encode_s_per_mp[format_index(format)]
    }

    /// Seconds per megapixel of the rendered picture of a whole export of `format` on `device`, the
    /// writing inside it, when that has been measured ([`record_overlapped`](Self::record_overlapped));
    /// `None` otherwise (the CPU path never has one).
    #[must_use]
    pub fn combined_s_per_mp(&self, device: ComputeDevice, format: FormatKind) -> Option<f64> {
        let set = &self.st().sets[device_index(device)];
        let i = format_index(format);
        (set.combined_samples[i] > 0).then_some(set.combined_s_per_mp[i])
    }

    /// Whether any photo has been measured yet (otherwise the numbers are the starting ones).
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.st()
            .sets
            .iter()
            .any(|s| s.render_samples > 0 || s.combined_samples.iter().any(|&n| n > 0))
    }

    /// Seconds to export one photo on `device`: `render_mp` megapixels to render (twice for a DNG,
    /// which renders the raw pixels and the preview) and `out_mp` to encode. A format that has been
    /// measured whole on the device (render and writing overlapped) is `render_mp` at that speed.
    #[must_use]
    pub fn seconds_for(
        &self,
        device: ComputeDevice,
        render_mp: f64,
        out_mp: f64,
        format: FormatKind,
    ) -> f64 {
        let st = self.st();
        let set = &st.sets[device_index(device)];
        let i = format_index(format);
        if set.combined_samples[i] > 0 {
            return render_mp * set.combined_s_per_mp[i];
        }
        let renders = if format == FormatKind::Dng { 2.0 } else { 1.0 };
        renders * render_mp * set.render_s_per_mp + out_mp * set.encode_s_per_mp[i]
    }
}

// ---------------------------------------------------------------------------------------------
// Context, photo, parameters
// ---------------------------------------------------------------------------------------------

/// What an export needs besides the photo: the renderer (which knows the looks and the display
/// rendering), the folder used when the default destination cannot be written, the app version
/// and the DNG profile.
#[derive(Debug)]
pub struct ExportContext {
    renderer: Renderer,
    home: Option<PathBuf>,
    app_version: String,
    dng_profile: FittedProfile,
    speed: ExportSpeed,
}

impl ExportContext {
    /// A context over `renderer`. The home folder is `$HOME` and the version the application's.
    #[must_use]
    pub fn new(renderer: Renderer) -> Self {
        Self {
            renderer,
            home: naming::home_dir(),
            app_version: app_version().to_owned(),
            // The tone curve and look table `tools/dngfit` fitted to ACES 2 (`data/dng/profile.json`,
            // embedded in `ym-export`), and the `BaselineExposure` offset they were fitted with.
            // The app and the CLI both come through here, so their DNGs are the same bytes.
            dng_profile: FittedProfile::embedded().clone(),
            speed: ExportSpeed::new(),
        }
    }

    /// Replaces the home folder (where `Pictures/Yellowmoon export` is). `None`: no fallback.
    #[must_use]
    pub fn with_home(mut self, home: Option<PathBuf>) -> Self {
        self.home = home;
        self
    }

    /// Replaces the version written into `xmp:CreatorTool`.
    #[must_use]
    pub fn with_app_version(mut self, version: impl Into<String>) -> Self {
        self.app_version = version.into();
        self
    }

    /// Replaces the DNG profile (the fitted tone curve and look table, and their `BaselineExposure`
    /// offset). The default is the profile embedded in `ym-export` ([`FittedProfile::embedded`]).
    #[must_use]
    pub fn with_dng_profile(mut self, profile: FittedProfile) -> Self {
        self.dng_profile = profile;
        self
    }

    /// The renderer.
    #[must_use]
    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    /// The home folder used for the fallback destination.
    #[must_use]
    pub fn home(&self) -> Option<&Path> {
        self.home.as_deref()
    }

    /// The measured speed of this machine (for estimates).
    #[must_use]
    pub fn speed(&self) -> &ExportSpeed {
        &self.speed
    }
}

/// What the queue keeps of a photo when a job is queued: its settings, estimates, camera and
/// metadata at that moment, without the decoded image (which is read when the photo's turn comes).
#[derive(Debug, Clone)]
pub struct PhotoSnapshot {
    /// The session's view of the photo.
    pub view: PhotoView,
    /// The settings resolved against the estimates, for the XMP.
    pub resolved: ResolvedSettings,
}

impl PhotoSnapshot {
    /// Snapshots a photo of the session.
    ///
    /// # Errors
    ///
    /// [`ExportError::Session`] for an unknown, skipped or not yet developed photo.
    pub fn take(session: &Session, id: PhotoId) -> Result<Self, ExportError> {
        Self::of_view(session.view(id)?)
    }

    /// Snapshots a view.
    ///
    /// # Errors
    ///
    /// [`ExportError::Settings`] when the settings cannot be resolved.
    pub fn of_view(view: PhotoView) -> Result<Self, ExportError> {
        let resolved = view.resolved()?;
        Ok(Self { view, resolved })
    }

    /// The snapshot with the estimates of `fresh` (the session's view of the same photo, taken
    /// now) in place of the ones it was taken with. The settings, the Pick white sample and
    /// everything else stay as they were when the job was queued; only what the session shares
    /// between photos (the burst median exposure, the pooled noise) is read again, so a job
    /// queued while the photos it depends on were still developing exports with the numbers the
    /// command line would have used.
    ///
    /// # Errors
    ///
    /// [`ExportError::Settings`] when the settings cannot be resolved against the new estimates.
    pub fn with_estimates_of(&self, fresh: PhotoView) -> Result<Self, ExportError> {
        let view = PhotoView {
            estimates: fresh.estimates,
            estimates_version: fresh.estimates_version,
            ..self.view.clone()
        };
        Self::of_view(view)
    }

    /// The file name the template gives this photo as number `seq` of a job.
    #[must_use]
    pub fn planned_file_name(&self, options: &ExportOptions, seq: u32) -> String {
        let v = &self.view;
        planned_file_name_of(&v.path, &v.meta, &v.camera, &v.settings.look, options, seq)
    }

    /// The snapshot with the decoded image attached.
    #[must_use]
    pub fn with_raw(self, raw: Arc<RawImage>) -> ExportPhoto {
        ExportPhoto {
            id: self.view.id,
            path: self.view.path.clone(),
            meta: Arc::clone(&self.view.meta),
            camera: self.view.camera.clone(),
            resolved: self.resolved,
            source: PhotoSource::from_view(&self.view, raw),
        }
    }
}

/// A photo ready to export: the render's source (decoded image, profile, estimates, settings) and
/// what the metadata and the file name need.
#[derive(Debug, Clone)]
pub struct ExportPhoto {
    /// The photo in its session.
    pub id: PhotoId,
    /// The original file (never written).
    pub path: PathBuf,
    /// The probe's metadata, for EXIF.
    pub meta: Arc<RawMetadata>,
    /// The camera match, for the XMP and `{camera}`.
    pub camera: PhotoCamera,
    /// The settings resolved against the estimates, for the XMP.
    pub resolved: ResolvedSettings,
    /// The render's input; its `settings` are the ones exported.
    pub source: PhotoSource,
}

impl ExportPhoto {
    /// A photo of the session with its current settings. Decodes the file again when the session
    /// no longer holds its mosaic.
    ///
    /// # Errors
    ///
    /// [`ExportError::Session`] (unknown, skipped, still developing, or undecodable) and
    /// [`ExportError::Settings`].
    pub fn from_session(session: &Session, id: PhotoId) -> Result<Self, ExportError> {
        let snapshot = PhotoSnapshot::take(session, id)?;
        let raw = session.raw_image(id)?;
        Ok(snapshot.with_raw(raw))
    }

    /// The look the photo is exported with.
    #[must_use]
    pub fn look(&self) -> &LookId {
        &self.source.settings.look
    }
}

/// What a file is on disk, whatever name it is reached by.
#[cfg(unix)]
type FileKey = (u64, u64);
/// What a file is on disk, whatever name it is reached by.
#[cfg(not(unix))]
type FileKey = PathBuf;

/// The identity of the file at `path`: its device and inode where there are such (a name that
/// differs only in case on a case-insensitive volume reaches the same file), else its canonical
/// path. `None` when there is no file.
fn file_key(path: &Path) -> Option<FileKey> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path).ok()?;
        Some((meta.dev(), meta.ino()))
    }
    #[cfg(not(unix))]
    {
        std::fs::canonicalize(path).ok()
    }
}

/// The files one job has written so far.
///
/// "If the file exists: Replace" and "Skip" look at the disk, and a file the same job wrote a moment
/// ago is on the disk too. Two photos of a job that map to one name (two cards that both hold
/// `DSC00001.ARW`, or a template with `{date}` and no `{seq}`) would otherwise replace each other
/// with both reported as saved, or have the second skipped as "already there". [`export_photo`]
/// asks this set before it plans a photo's file: a name that this job has already written is
/// numbered as "Keep both" numbers it, whatever the job's choice is. A file that was there before
/// the job is still replaced or skipped as asked.
///
/// Clones share one set, so one is made per job and handed to every photo of it through
/// [`ExportParams::written`].
#[derive(Debug, Clone, Default)]
pub struct WrittenFiles(Arc<Mutex<std::collections::HashSet<FileKey>>>);

impl WrittenFiles {
    /// An empty set for a new job.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the file now at `path` is one this job has written (also when `path` spells its name
    /// in another case on a volume that does not tell the two apart).
    #[must_use]
    pub fn contains(&self, path: &Path) -> bool {
        file_key(path).is_some_and(|key| {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains(&key)
        })
    }

    /// Records the file at `path`, which has just been written.
    pub fn insert(&self, path: &Path) {
        if let Some(key) = file_key(path) {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key);
        }
    }

    /// What "If the file exists" means for `file_name` in `dir`: the job's choice, except that a
    /// name this job has already written is always numbered ([`IfExists::KeepBoth`]).
    #[must_use]
    pub fn if_exists_for(&self, dir: &Path, file_name: &str, chosen: IfExists) -> IfExists {
        if chosen != IfExists::KeepBoth && self.contains(&dir.join(file_name)) {
            IfExists::KeepBoth
        } else {
            chosen
        }
    }
}

impl PartialEq for WrittenFiles {
    /// Two handles are equal when they are the same set.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for WrittenFiles {}

/// The parameters of one photo's export that come from the job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportParams {
    /// Format, quality, bits, compression, colour space, size, destination, name template and what
    /// to do when the file exists.
    pub options: ExportOptions,
    /// The preference "Remove location from exports".
    pub remove_location: bool,
    /// The number of this photo in the job, 1-based: the `{seq}` of the name template.
    pub seq: u32,
    /// The files the job has written so far (one set for all photos of a job), so that a photo
    /// never replaces, or is skipped for, a file its own job made. `None` for a single export.
    pub written: Option<WrittenFiles>,
}

impl ExportParams {
    /// Parameters for the first photo of a job.
    #[must_use]
    pub fn new(options: ExportOptions) -> Self {
        Self {
            options,
            remove_location: false,
            seq: 1,
            written: None,
        }
    }

    /// Shares the job's set of written files (see [`WrittenFiles`]).
    #[must_use]
    pub fn written(mut self, written: WrittenFiles) -> Self {
        self.written = Some(written);
        self
    }

    /// Sets the number of the photo in the job.
    #[must_use]
    pub fn seq(mut self, seq: u32) -> Self {
        self.seq = seq;
        self
    }

    /// Sets "Remove location from exports".
    #[must_use]
    pub fn remove_location(mut self, remove: bool) -> Self {
        self.remove_location = remove;
        self
    }
}

/// Cancel and progress for one export.
pub struct ExportControl<'a> {
    /// Cancelling stops the export after the current tile (rendering) or band (encoding) and
    /// deletes the partial file.
    pub cancel: CancelToken,
    /// Called with the fraction done, 0 to 1, from any thread; not always in order.
    pub progress: Option<&'a (dyn Fn(f32) + Sync)>,
}

impl ExportControl<'_> {
    /// No cancellation, no progress.
    #[must_use]
    pub fn none() -> ExportControl<'static> {
        ExportControl {
            cancel: CancelToken::new(),
            progress: None,
        }
    }
}

/// Where the time of an export went.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExportTimings {
    /// Rendering the pixels (for a DNG, both renders), in milliseconds.
    pub render_ms: f64,
    /// Resizing, encoding and writing, in milliseconds. With the export tail on the device the
    /// writer pulls the picture while it is rendered, so this includes the time spent waiting for
    /// the render, and the two overlap.
    pub encode_ms: f64,
}

/// What [`export_photo`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportReport {
    /// The file written, or the existing file when it was skipped.
    pub path: PathBuf,
    /// What happened to the file.
    pub placement: Placement,
    /// "If the file exists: Skip" and it did: nothing was rendered or written.
    pub skipped: bool,
    /// Bytes written (0 for a skip).
    pub bytes: u64,
    /// Width of the written image (0 for a skip).
    pub width: u32,
    /// Height of the written image (0 for a skip).
    pub height: u32,
    /// Size of the rendered picture before any resize.
    pub rendered: (u32, u32),
    /// DNG scaling and clipping details.
    pub dng: Option<DngReport>,
    /// Where the time went.
    pub timings: ExportTimings,
    /// [`ComputeDevice::Gpu`] when any stage of this photo's export ran on the graphics card (the
    /// raw chain of a tile, the export tail), [`ComputeDevice::Cpu`] when none did: also for a photo
    /// that started on the device and was made again on the CPU after an error, whose `notes` say
    /// why. A photo that was skipped (nothing was rendered) says `Cpu`.
    pub device: ComputeDevice,
    /// Plain-words notes from the render (a classical demosaic, lens tables that were ignored).
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------------------------------------
// Names, destination, metadata
// ---------------------------------------------------------------------------------------------

/// The fields of the file name template for a photo, from what a snapshot holds.
#[must_use]
pub fn name_fields_of(
    path: &Path,
    meta: &RawMetadata,
    camera: &PhotoCamera,
    look: &LookId,
    seq: u32,
) -> NameFields {
    NameFields {
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        look: look.slug().to_owned(),
        date: meta
            .capture_time
            .map(|t| format!("{:04}-{:02}-{:02}", t.year, t.month, t.day)),
        seq,
        camera: naming::slugify(&camera.matched.body),
    }
}

/// The file name (with extension) the template gives a photo, before any "Keep both" number.
#[must_use]
pub fn planned_file_name_of(
    path: &Path,
    meta: &RawMetadata,
    camera: &PhotoCamera,
    look: &LookId,
    options: &ExportOptions,
    seq: u32,
) -> String {
    format!(
        "{}.{}",
        naming::file_stem(
            &options.name_template,
            &name_fields_of(path, meta, camera, look, seq)
        ),
        options.format.extension()
    )
}

/// The fields of the file name template for a photo.
#[must_use]
pub fn name_fields(photo: &ExportPhoto, seq: u32) -> NameFields {
    name_fields_of(&photo.path, &photo.meta, &photo.camera, photo.look(), seq)
}

/// The file name (with extension) the template gives a photo, before any "Keep both" number.
#[must_use]
pub fn planned_file_name(photo: &ExportPhoto, options: &ExportOptions, seq: u32) -> String {
    planned_file_name_of(
        &photo.path,
        &photo.meta,
        &photo.camera,
        photo.look(),
        options,
        seq,
    )
}

/// The EXIF content for a photo (`docs/09` Metadata): camera, lens, exposure, capture time and,
/// unless `remove_location`, the GPS position.
#[must_use]
pub fn exif_of(meta: &RawMetadata, remove_location: bool) -> ExifInfo {
    let gps = if remove_location {
        None
    } else {
        meta.gps.as_ref().and_then(|g| {
            Some(Gps {
                latitude_deg: g.latitude_deg?,
                longitude_deg: g.longitude_deg?,
                altitude_m: g.altitude_m,
            })
        })
    };
    ExifInfo {
        make: Some(meta.make.clone()).filter(|s| !s.is_empty()),
        model: Some(meta.model.clone()).filter(|s| !s.is_empty()),
        lens_make: None,
        lens_model: meta.lens_model.clone(),
        exposure_time_s: meta.exposure_time_s,
        f_number: meta.f_number,
        iso: meta.iso,
        focal_length_mm: meta.focal_length_mm,
        focal_length_35mm: meta.focal_length_35mm,
        exposure_bias_ev: meta.exposure_comp_ev,
        capture_time: meta.capture_time.map(|t| CaptureTime {
            year: t.year,
            month: t.month,
            day: t.day,
            hour: t.hour,
            minute: t.minute,
            second: t.second,
            millisecond: t.subsec_ms,
            utc_offset_minutes: t.utc_offset_minutes,
        }),
        gps,
    }
}

/// How the profile was matched, as `ym:ProfileMatch` says it: `exact`, `sensor family`, `default`
/// (or `forced` for the CLI's `--profile`).
#[must_use]
pub fn profile_match_text(kind: &MatchKind) -> &'static str {
    match kind {
        MatchKind::Exact => "exact",
        MatchKind::SensorFamily { .. } => "sensor family",
        MatchKind::Default => "default",
        MatchKind::Forced { .. } => "forced",
    }
}

/// The id of the display transform that makes the picture: `aces2-sdr100-rec709lim` for sRGB,
/// `aces2-sdr100-p3lim` for Display P3, `aces2-sdr100-p3lim-rec2020` for Rec.2020 (the limiting
/// primaries and, where it differs from them, the encoding), or the id of the imported ARRI
/// display LUT (`arri-display:logc4:rec709-g24`).
#[must_use]
pub fn display_transform_id(choice: &DisplayChoice, space: ColorSpace) -> String {
    match choice {
        DisplayChoice::ArriLut(id) => id.to_string(),
        DisplayChoice::Aces2 => match space {
            ColorSpace::Srgb => "aces2-sdr100-rec709lim".to_owned(),
            ColorSpace::DisplayP3 => "aces2-sdr100-p3lim".to_owned(),
            ColorSpace::Rec2020 => "aces2-sdr100-p3lim-rec2020".to_owned(),
        },
    }
}

fn output_target(space: ColorSpace) -> OutputTarget {
    match space {
        ColorSpace::Srgb => OutputTarget::Srgb,
        ColorSpace::DisplayP3 => OutputTarget::DisplayP3,
        ColorSpace::Rec2020 => OutputTarget::Rec2020,
    }
}

fn rounded_percent(x: f64) -> u32 {
    x.round().clamp(0.0, 100.0) as u32
}

// ---------------------------------------------------------------------------------------------
// The export
// ---------------------------------------------------------------------------------------------

/// Shares of one photo's progress: rendering and the rest.
const RASTER_RENDER_SHARE: f32 = 0.75;
/// A DNG renders twice (raw pixels, then the preview) and then writes.
const DNG_RAW_SHARE: f32 = 0.35;
const DNG_PREVIEW_SHARE: f32 = 0.70;

fn report(control: &ExportControl<'_>, fraction: f32) {
    if let Some(p) = control.progress {
        p(fraction.clamp(0.0, 1.0));
    }
}

/// Renders one target on the export worker and waits for it, passing the render's tile progress
/// on (scaled into `lo..hi` of the photo's progress) and its own cancellation to the job. With
/// `cpu_reference` the render uses no device at all (what an export falls back to).
fn render(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    target: FullTarget,
    control: &ExportControl<'_>,
    (lo, hi): (f32, f32),
    cpu_reference: bool,
) -> Result<FullImage, ExportError> {
    control.cancel.check().map_err(|_| ExportError::Cancelled)?;
    let tiles = Arc::new(AtomicU32::new(0));
    let sink = {
        let tiles = Arc::clone(&tiles);
        Arc::new(move |f: f32| tiles.store(f.to_bits(), Ordering::Relaxed))
    };
    let handle = if cpu_reference {
        ctx.renderer
            .spawn_full_cpu(photo.source.clone(), FullRequest { target }, sink)
    } else {
        ctx.renderer
            .spawn_full_reporting(photo.source.clone(), FullRequest { target }, sink)
    };
    let mut last = f32::NAN;
    loop {
        if control.cancel.is_cancelled() {
            handle.cancel();
        }
        match handle.wait_timeout(Duration::from_millis(20)) {
            None => {
                let f = f32::from_bits(tiles.load(Ordering::Relaxed));
                if f != last {
                    last = f;
                    report(control, lo + (hi - lo) * f);
                }
            }
            Some(Ok(Ok(image))) => {
                report(control, hi);
                return Ok(image);
            }
            Some(Ok(Err(e))) => return Err(e.into()),
            Some(Err(JobError::Cancelled)) => return Err(ExportError::Cancelled),
            Some(Err(JobError::Shutdown)) => return Err(RenderError::Shutdown.into()),
            Some(Err(e)) => {
                return Err(ExportError::Render(RenderError::Image(format!(
                    "This photo can't be rendered: {e}."
                ))));
            }
        }
    }
}

/// Exports one photo: the single code path of the app and the CLI (see the module documentation).
///
/// # Errors
///
/// [`ExportError::Look`] for a look or display LUT that cannot be loaded, or that was replaced,
/// removed or switched while the photo was rendering ([`CHANGED_DURING_EXPORT_MESSAGE`]; nothing is
/// written under a look the pixels were not made with),
/// [`ExportError::Render`] (memory, an unusable photo, a panic
/// below this function, which is caught and logged),
/// [`ExportError::Write`] (the folder, the disk, the original in the way), and
/// [`ExportError::Cancelled`] after `control.cancel` fired, with the partial file removed. A photo
/// whose file exists with "Skip" is not an error: the report says `skipped`.
pub fn export_photo(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    params: &ExportParams,
    control: &ExportControl<'_>,
) -> Result<ExportReport, ExportError> {
    // A bug below this function (a writer, a look provider) is this photo's error, not an
    // unwinding queue thread that leaves the panel waiting for a file that never comes. The
    // writers delete their temporary file when they unwind.
    catch_unwind(AssertUnwindSafe(|| {
        export_photo_unguarded(ctx, photo, params, control)
    }))
    .unwrap_or_else(|payload| {
        let what = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        tracing::error!(photo = %photo.id, panic = %what, "an export panicked");
        Err(ExportError::Render(RenderError::Image(
            "This photo can't be exported because of an unexpected problem. Try again, or export it in another format."
                .to_owned(),
        )))
    })
}

/// The message of a photo whose look or display rendering changed while it was exporting.
pub const CHANGED_DURING_EXPORT_MESSAGE: &str =
    "The look or the display rendering changed while this photo was exporting. Export it again.";

/// What the render reads from the look library and from Preferences › Display rendering, for
/// noticing that either changed between the start of an export and the file being written: the
/// file's XMP says which look and transform made it, so it must say it truthfully.
#[derive(Debug, PartialEq, Eq)]
struct RenderInputs {
    /// The look's fingerprint (the SHA-256 of its LUT, or its id for a built-in look).
    look: String,
    /// The display rendering in effect.
    display: DisplayChoice,
    /// The fingerprint of the chosen display LUT.
    display_lut: Option<String>,
}

/// The look and display rendering as they are now. An error when either cannot be loaded: an
/// export never falls back.
fn render_inputs(ctx: &ExportContext, photo: &ExportPhoto) -> Result<RenderInputs, ExportError> {
    let looks = ctx.renderer.looks();
    let look = looks
        .look(photo.look())
        .map_err(|e| ExportError::Look(e.to_string()))?;
    let display = ctx.renderer.display();
    let display_lut = match &display {
        DisplayChoice::ArriLut(id) => Some(
            looks
                .display_transform(id)
                .map_err(|e| ExportError::Look(e.to_string()))?
                .1,
        ),
        DisplayChoice::Aces2 => None,
    };
    Ok(RenderInputs {
        look: look.fingerprint,
        display,
        display_lut,
    })
}

fn export_photo_unguarded(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    params: &ExportParams,
    control: &ExportControl<'_>,
) -> Result<ExportReport, ExportError> {
    let started = Instant::now();
    let options = &params.options;
    options
        .validate()
        .map_err(|e| ExportError::Options(e.to_string()))?;
    control.cancel.check().map_err(|_| ExportError::Cancelled)?;

    // Progress never goes back, also when a photo is made again on the CPU after the device failed.
    let best = AtomicU32::new(0);
    let monotone = |f: f32| {
        let f = f.clamp(0.0, 1.0);
        if f > f32::from_bits(best.load(Ordering::Relaxed)) {
            best.store(f.to_bits(), Ordering::Relaxed);
            if let Some(p) = control.progress {
                p(f);
            }
        }
    };
    let control = &ExportControl {
        cancel: control.cancel.clone(),
        progress: Some(&monotone),
    };

    // 1. The look and the display rendering must be usable: an export does not fall back.
    let looks = ctx.renderer.looks();
    let inputs = render_inputs(ctx, photo)?;
    let display = inputs.display.clone();
    let is_dng = options.format == FormatKind::Dng;
    let look_sha256 = looks.look_sha256(photo.look());
    // Checked again once the pixels are rendered (which takes seconds to minutes).
    let still_the_same = || -> Result<(), ExportError> {
        if render_inputs(ctx, photo)? == inputs {
            Ok(())
        } else {
            Err(ExportError::Look(CHANGED_DURING_EXPORT_MESSAGE.to_owned()))
        }
    };

    // 2. The file name and the folder; a skipped file costs nothing.
    let fields = name_fields(photo, params.seq);
    let mut dest =
        ExportDestination::resolve(options, &fields, Some(&photo.path), ctx.home.as_deref())?;
    // A name this job has already written is numbered whatever "If the file exists" says: Replace
    // would overwrite the job's own earlier photo and Skip would take it for a file from before.
    let mut notes: Vec<String> = Vec::new();
    if let Some(written) = &params.written {
        let file_name = format!("{}.{}", dest.file_stem, options.format.extension());
        let chosen = dest.if_exists;
        dest.if_exists = written.if_exists_for(&dest.dir, &file_name, chosen);
        if dest.if_exists != chosen {
            notes.push(SAME_NAME_IN_JOB_NOTE.to_owned());
        }
    }
    if let Some(placement) = naming::plan_output(
        &dest.dir,
        &dest.file_stem,
        options.format.extension(),
        dest.if_exists,
        Some(&photo.path),
    )? {
        report(control, 1.0);
        return Ok(ExportReport {
            path: placement.path().to_path_buf(),
            skipped: matches!(placement, Placement::Skipped(_)),
            placement,
            bytes: 0,
            width: 0,
            height: 0,
            rendered: (0, 0),
            dng: None,
            timings: ExportTimings::default(),
            device: ComputeDevice::Cpu,
            notes,
        });
    }

    // 3. The metadata every format carries.
    let camera = &photo.camera;
    let space = if is_dng {
        ColorSpace::Srgb
    } else {
        options.color_space
    };
    let exif = exif_of(&photo.meta, params.remove_location);
    let metadata = Metadata {
        xmp: XmpInfo {
            app_version: ctx.app_version.clone(),
            capture_time: exif.capture_time,
            ym: YmInfo {
                pipeline_version: Some(PIPELINE_VERSION.to_owned()),
                profile: Some(photo.source.profile.id.to_string()),
                profile_version: Some(photo.source.profile.version.to_string()),
                profile_match: Some(profile_match_text(&camera.matched.how).to_owned()),
                look: Some(photo.look().to_string()),
                look_sha256,
                look_intensity_pct: Some(rounded_percent(photo.source.settings.look_intensity)),
                display_transform: Some(display_transform_id(&display, space)),
                models: None,
                // Which Auto made the values, and what Apple Auto said (`docs/19` section 9).
                auto: Some(photo.source.estimates.source().name().to_owned()),
                apple_auto: photo.source.estimates.apple.xmp_json(),
                settings: serde_json::to_value(photo.resolved.pinned_settings()).ok(),
            },
        },
        exif,
    };

    // 4. Render and write.
    let cancel_flag = || control.cancel.is_cancelled();
    let progress_fn = |f: f32| report(control, f);
    let hooks = ExportHooks::new(Some(&cancel_flag), Some(&progress_fn));
    let render_ms;
    let outcome;
    let rendered;
    let encode_started;
    let device;
    // The time of a photo made on the device covers the render and the writing together.
    let mut overlapped = false;
    if is_dng {
        // A DNG is two renders, the raw pixels and the preview, and one file is wholly device or
        // wholly CPU (`docs/17-gpu.md` section 9). A device error that fires once is spent by the render
        // it hits, so the other render would go on running on the device: when the raw render fell
        // back to the CPU the preview is made on the CPU reference path, and when only the preview
        // fell back the raw pixels are made again on it (`FullImage::fell_back_to_cpu`).
        let mut raw = render(
            ctx,
            photo,
            FullTarget::DngLinearAwg4,
            control,
            (0.0, DNG_RAW_SHARE),
            false,
        )?;
        let preview = render(
            ctx,
            photo,
            FullTarget::DngPreview,
            control,
            (DNG_RAW_SHARE, DNG_PREVIEW_SHARE),
            raw.fell_back_to_cpu,
        )?;
        // What the discarded device render took of the time.
        let mut discarded_ms = 0.0;
        if preview.fell_back_to_cpu && !raw.fell_back_to_cpu {
            discarded_ms = raw.report.ms;
            raw = render(
                ctx,
                photo,
                FullTarget::DngLinearAwg4,
                control,
                // No progress to report: the share of the raw pixels is long reported.
                (DNG_PREVIEW_SHARE, DNG_PREVIEW_SHARE),
                true,
            )?;
        }
        render_ms = discarded_ms + raw.report.ms + preview.report.ms;
        device = if device_of(&raw.report) == ComputeDevice::Gpu
            || device_of(&preview.report) == ComputeDevice::Gpu
        {
            ComputeDevice::Gpu
        } else {
            ComputeDevice::Cpu
        };
        // The numbers of the path the estimate reads: the colour stage on the device (the export
        // tail's) or the CPU reference, whatever ran the raw chain (`RenderReport::device`, not
        // `device_of`, which also counts the raw ranges).
        ctx.speed.record_render_on(
            raw.report.device,
            megapixels(raw.width, raw.height),
            raw.report.ms / 1000.0,
        );
        ctx.speed.record_render_on(
            preview.report.device,
            megapixels(preview.width, preview.height),
            preview.report.ms / 1000.0,
        );
        for n in raw.report.notes.iter().chain(&preview.report.notes) {
            if !notes.contains(n) {
                notes.push(n.clone());
            }
        }
        rendered = (raw.width as u32, raw.height as u32);
        let orientation = compose_orientation(
            photo.source.exif_orientation(),
            photo.source.settings.orientation.degrees(),
        )
        .unwrap_or(1);
        let crop = raw.window.map(|w| CropRect {
            x: w.x as u32,
            y: w.y as u32,
            width: w.width as u32,
            height: w.height as u32,
        });
        still_the_same()?;
        let raw_ref = ImageRef::new(
            raw.width as u32,
            raw.height as u32,
            &raw.rgb,
            PixelEncoding::SceneLinearAwg4,
        )?;
        let preview_ref = ImageRef::new(
            preview.width as u32,
            preview.height as u32,
            &preview.rgb,
            PixelEncoding::DisplayLinear,
        )?;
        let request = ExportRequest {
            pixels: raw_ref,
            format: ExportFormat::Dng,
            color_space: ColorSpace::Srgb,
            size: ym_export::Size::Full,
            metadata: &metadata,
            remove_location: params.remove_location,
            dng: Some(DngInputs {
                preview: preview_ref,
                orientation,
                crop,
                baseline_exposure: ctx.dng_profile.baseline_exposure(),
                profile: &ctx.dng_profile.profile,
            }),
        };
        encode_started = Instant::now();
        outcome = write_export(&request, &dest, &hooks.stage(DNG_PREVIEW_SHARE, 1.0))?;
    } else {
        // Output sharpening (Detail › Sharpening › Output) runs right after the Lanczos-3 resize, on
        // the picture at the size it is written. Never for a DNG.
        let stages =
            PictureStages::output_sharpening(photo.source.settings.output_sharpening as f32);
        // With the export tail on the device the picture is never a host frame: the device makes
        // the encoded rows and the writer pulls them (`pipeline::frame::export_tail`). A device
        // error ends that attempt, and the whole photo is made again on the CPU reference path.
        let mut cpu_reference = false;
        let mut on_device = None;
        if let Some(why) = ctx.renderer.export_tail_refused() {
            return Err(ExportError::Render(RenderError::Gpu(why)));
        }
        if ctx.renderer.export_tail_enabled() {
            let began = Instant::now();
            let request = TailRequest {
                target: output_target(options.color_space),
                space: options.color_space,
                size: options.size,
                output_sharpening_pct: stages.output_sharpening_pct,
                bits: bit_depth_of(options.format_spec()),
            };
            match export_on_device(
                ctx,
                photo,
                request,
                &metadata,
                params,
                &dest,
                &hooks,
                control,
                &still_the_same,
            ) {
                Ok(done) => on_device = Some((done, began)),
                Err(ExportError::Render(RenderError::Gpu(reason)))
                    if !ctx.renderer.options().gpu_strict =>
                {
                    tracing::warn!(photo = %photo.id, error = %reason, "the export tail failed on the device; the CPU takes over");
                    notes.push(format!(
                        "The graphics card could not render this photo, so the CPU did: {reason}."
                    ));
                    cpu_reference = true;
                }
                Err(e) => return Err(e),
            }
        }
        if let Some((done, began)) = on_device {
            if let Some(render_report) = &done.report {
                render_ms = render_report.ms;
                rendered = done.rendered;
                device = ComputeDevice::Gpu;
                overlapped = true;
                for n in &render_report.notes {
                    if !notes.contains(n) {
                        notes.push(n.clone());
                    }
                }
            } else {
                // Skipped after the render had started (see `export_on_device`): nothing was
                // rendered to the end, as for a photo that is skipped before it starts.
                render_ms = 0.0;
                rendered = (0, 0);
                device = ComputeDevice::Cpu;
            }
            outcome = done.outcome;
            encode_started = began;
        } else {
            let image = render(
                ctx,
                photo,
                FullTarget::DisplayLinear(output_target(options.color_space)),
                control,
                (0.0, RASTER_RENDER_SHARE),
                cpu_reference,
            )?;
            render_ms = image.report.ms;
            device = device_of(&image.report);
            // The CPU reference's numbers, also when its raw chain ran on the device: the estimate
            // reads this set unless the export tail is on (`estimate`).
            ctx.speed.record_render_on(
                image.report.device,
                megapixels(image.width, image.height),
                image.report.ms / 1000.0,
            );
            for n in &image.report.notes {
                if !notes.contains(n) {
                    notes.push(n.clone());
                }
            }
            still_the_same()?;
            rendered = (image.width as u32, image.height as u32);
            let pixels = ImageRef::new(
                image.width as u32,
                image.height as u32,
                &image.rgb,
                PixelEncoding::DisplayLinear,
            )?;
            let request = ExportRequest::new(pixels, options, &metadata)
                .remove_location(params.remove_location);
            encode_started = Instant::now();
            outcome = write_export_with(
                &request,
                &dest,
                &hooks.stage(RASTER_RENDER_SHARE, 1.0),
                &stages,
            )?;
        }
    }
    let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
    let out_mp = megapixels(outcome.width as usize, outcome.height as usize);
    if overlapped {
        // The whole export (render and writing overlap) per megapixel of the rendered picture,
        // for this format only. The render time of such an export is not recorded as the device's
        // render speed: the render waits for a writer that is slower (PNG), so it would hold the
        // writer's time.
        ctx.speed.record_overlapped(
            options.format,
            megapixels(rendered.0 as usize, rendered.1 as usize),
            encode_ms / 1000.0,
        );
    } else {
        ctx.speed
            .record_encode(options.format, out_mp, encode_ms / 1000.0);
    }
    report(control, 1.0);
    if let Some(dng) = &outcome.dng {
        // `docs/09`: the clipped fractions go in the export log.
        tracing::info!(
            photo = %photo.id,
            x_white = dng.x_white,
            baseline_exposure_ev = dng.baseline_exposure_ev,
            negative_clipped_fraction = dng.negative_clipped_fraction,
            highlight_clipped_fraction = dng.highlight_clipped_fraction,
            "DNG scaling"
        );
    }
    tracing::info!(
        photo = %photo.id,
        format = options.format.display_name(),
        bytes = outcome.bytes,
        render_ms,
        encode_ms,
        device = ?device,
        total_ms = started.elapsed().as_secs_f64() * 1000.0,
        "exported"
    );
    let skipped = matches!(outcome.placement, Placement::Skipped(_));
    if !skipped && let Some(written) = &params.written {
        written.insert(outcome.path());
    }
    Ok(ExportReport {
        path: outcome.path().to_path_buf(),
        skipped,
        placement: outcome.placement,
        bytes: outcome.bytes,
        width: outcome.width,
        height: outcome.height,
        rendered,
        dng: outcome.dng,
        timings: ExportTimings {
            render_ms,
            encode_ms,
        },
        device,
        notes,
    })
}

/// What a raster export that the device made gives back.
struct DeviceExport {
    outcome: ym_export::ExportOutcome,
    /// The render's report; `None` for a photo that was skipped after the render had started.
    report: Option<RenderReport>,
    rendered: (u32, u32),
}

/// The depth of a raster export's samples.
fn bit_depth_of(format: ExportFormat) -> ym_export::BitDepth {
    match format {
        ExportFormat::Tiff { bits, .. } | ExportFormat::Png { bits } => bits,
        ExportFormat::Jpeg { .. } | ExportFormat::Dng => ym_export::BitDepth::Eight,
    }
}

/// Where the stages of a render ran: the device when its colour pass or any raw range did.
fn device_of(report: &RenderReport) -> ComputeDevice {
    if report.device == ComputeDevice::Gpu
        || report.ranges.iter().any(|r| r.device == ComputeDevice::Gpu)
    {
        ComputeDevice::Gpu
    } else {
        ComputeDevice::Cpu
    }
}

/// Makes a raster export with the export tail on the device: the render sends encoded rows, the
/// writer pulls them into the file. Returns [`RenderError::Gpu`] (inside [`ExportError::Render`])
/// for a device error, with the temporary file already removed, so that the caller can make the
/// photo again on the CPU.
#[allow(clippy::too_many_arguments)] // the export's state, passed once
fn export_on_device(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    request: TailRequest,
    metadata: &Metadata,
    params: &ExportParams,
    dest: &ExportDestination,
    hooks: &ExportHooks<'_>,
    control: &ExportControl<'_>,
    still_the_same: &dyn Fn() -> Result<(), ExportError>,
) -> Result<DeviceExport, ExportError> {
    let options = &params.options;
    let mut bands =
        ctx.renderer
            .start_export_tail(photo.source.clone(), request, &control.cancel)?;
    let rendered = bands.rendered();
    // Nothing is written under a look that changed while the picture rendered: checked when the last
    // rows are asked for, before the writer can finish the file.
    bands.check_before_last_rows(|| {
        still_the_same().map_err(|e| match e {
            ExportError::Look(message) => RenderError::Look(message),
            other => RenderError::Image(other.to_string()),
        })
    });
    let band_request = BandRequest {
        format: options.format_spec(),
        color_space: options.color_space,
        metadata,
        remove_location: params.remove_location,
    };
    let written = write_export_bands(&band_request, &mut bands, dest, hooks);
    let failure = bands.take_failure();
    let finished = bands.finish();
    match written {
        Ok(outcome) => {
            let report = match finished {
                Ok(report) => Some(report),
                // A file appeared under the name between the check at the start and the writer's
                // own: with "Skip" the writer pulled no row and left the file alone, and it is
                // `bands` that stopped the render. The photo is skipped like any other, not cancelled.
                Err(RenderError::Cancelled)
                    if matches!(outcome.placement, Placement::Skipped(_)) =>
                {
                    None
                }
                Err(e) => return Err(e.into()),
            };
            Ok(DeviceExport {
                outcome,
                report,
                rendered,
            })
        }
        Err(e) => {
            // The render's own error, or the look that changed, is the reason when there is one: the
            // writer only saw that its rows stopped coming.
            if let Some(f) = failure {
                return Err(f.into());
            }
            Err(e.into())
        }
    }
}

fn megapixels(w: usize, h: usize) -> f64 {
    w as f64 * h as f64 / 1e6
}

// ---------------------------------------------------------------------------------------------
// Estimates
// ---------------------------------------------------------------------------------------------

/// The size in pixels of the picture a photo exports at full size, worked out from the probe alone
/// (the camera's crop, the orientation, the user's turn and crop; the lens correction's few percent
/// of auto crop and the straightening are not counted). For an estimate, not for the render.
#[must_use]
pub fn estimated_output_size(meta: &RawMetadata, settings: &DevelopSettings) -> (u32, u32) {
    let (mut w, mut h) = meta_image_size(meta);
    if matches!(settings.orientation.degrees(), 90 | 270) {
        std::mem::swap(&mut w, &mut h);
    }
    if let Some(crop) = &settings.crop {
        w = ((w as f64 * crop.rect.width).round() as usize).max(1);
        h = ((h as f64 * crop.rect.height).round() as usize).max(1);
    }
    (w as u32, h as u32)
}

/// What `export_estimate` answers for a set of photos (`ExportEstimate` in `src/api/types.ts`,
/// without the two counts the queue adds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Total size of the files in bytes.
    pub bytes: u64,
    /// Time to export them in seconds.
    pub seconds: f64,
}

/// Estimates the files of `photos` for `options`: each photo's probe metadata, settings and the
/// detail statistic of its thumbnail (`None` while it has none: the estimate then assumes a
/// picture of typical detail).
#[must_use]
pub fn estimate<'a>(
    ctx: &ExportContext,
    photos: impl IntoIterator<Item = (&'a RawMetadata, &'a DevelopSettings, Option<DetailStat>)>,
    options: &ExportOptions,
) -> Estimate {
    let format = options.format_spec();
    // The numbers of the path the export would take: the device's when the renderer runs the tail.
    let device = if ctx.renderer.export_tail_enabled() {
        ComputeDevice::Gpu
    } else {
        ComputeDevice::Cpu
    };
    let mut bytes = 0u64;
    let mut seconds = 0.0;
    for (meta, settings, detail) in photos {
        let (w, h) = estimated_output_size(meta, settings);
        bytes += estimate_bytes_with_detail(w, h, format, options.size, detail);
        let (ow, oh) = if options.format == FormatKind::Dng {
            (w, h)
        } else {
            options.size.output_dimensions(w, h)
        };
        seconds += ctx.speed.seconds_for(
            device,
            megapixels(w as usize, h as usize),
            megapixels(ow as usize, oh as usize),
            options.format,
        );
    }
    Estimate { bytes, seconds }
}

/// The destination folder an export of `photo` would use, worked out without touching the disk:
/// the folder the options name, else `<source folder> export`, else `~/Pictures/Yellowmoon export`.
#[must_use]
pub fn planned_destination(
    ctx: &ExportContext,
    photo: &Path,
    options: &ExportOptions,
) -> Option<PathBuf> {
    options
        .destination
        .clone()
        .or_else(|| default_destination(photo))
        .or_else(|| ctx.home().map(naming::fallback_destination))
}

/// Whether "Keep both" would add a number to this name, and which: the first free of `name.ext`,
/// `name 2.ext`, `name 3.ext` in `dir`. A prediction for the panel; the writer decides atomically.
#[must_use]
pub fn predicted_name_in(dir: &Path, file_name: &str, if_exists: IfExists) -> String {
    if if_exists != IfExists::KeepBoth || !dir.join(file_name).exists() {
        return file_name.to_owned();
    }
    let (stem, ext) = file_name
        .rsplit_once('.')
        .map_or((file_name, ""), |(s, e)| (s, e));
    for n in 2..10_000 {
        let candidate = if ext.is_empty() {
            format!("{stem} {n}")
        } else {
            format!("{stem} {n}.{ext}")
        };
        if !dir.join(&candidate).exists() {
            return candidate;
        }
    }
    file_name.to_owned()
}
