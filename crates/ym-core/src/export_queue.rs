//! The export queue: jobs of photos with their settings snapshotted at queue time, one photo at a
//! time at full resolution with the next decode overlapping, progress events, cancel
//! (`docs/09-export.md` section Queue, `docs/01-architecture.md` IPC commands `export_estimate`,
//! `export_start`, `export_cancel` and the event `export_progress`).
//!
//! # Jobs
//!
//! [`ExportQueue::start`] takes the photos of a scope (This photo, Picked, All) and snapshots each
//! one's settings, estimates and metadata **now** ([`PhotoSnapshot`]): editing a photo afterwards
//! does not change a queued job, and neither does closing the session. The decoded image is not
//! part of the snapshot; it is read when the photo's turn comes (from the session's mosaic cache if
//! it is there, else from the file).
//!
//! One worker thread runs the jobs in order, and the photos of a job in order, each through
//! [`export_photo`] (the same function the CLI calls). While a photo renders, the next one is
//! decoded on another thread. The renders themselves run on the render scheduler's export worker,
//! which waits for the viewport (browsing and editing stay responsive while an export runs).
//!
//! # Events
//!
//! [`ExportEvent::Progress`] carries the `export_progress` payload of `src/api/types.ts`, in the
//! order the panel's `applyProgress` expects:
//!
//! * when a photo starts: `{ file, done: i, total, file_percent: 0 }` (`file` is the name the
//!   template gives, with the "Keep both" number the folder will need);
//! * while it renders and encodes: the same with a larger `file_percent` (whole percents, each at
//!   most once);
//! * a photo that failed or was skipped is reported once, as it finishes, with the new count:
//!   `{ file, done: i + 1, total, error }` or `{ …, skipped: true }`;
//! * the last event of a job has `done == total` (for a photo that was written it is
//!   `{ file, done: total, file_percent: 100 }`).
//!
//! [`ExportEvent::File`] carries what `export_photo` reported for a photo (its [`ExportReport`], or
//! why it failed and with which CLI exit code, and the milliseconds its decode took) just before the
//! `Progress` event that counts the photo: the CLI prints one JSON object per file from it, and
//! gets the decode-while-rendering overlap of the app for free.
//!
//! A cancelled job sends nothing more: the panel that asked for the cancel closes the run itself.
//! [`ExportEvent::Busy`] says when the queue becomes busy or idle, for the updater (below).
//!
//! # Names
//!
//! A name that an earlier photo of the same job has written is numbered ("Keep both") whatever the
//! job's "If the file exists" says: Replace would overwrite that photo and Skip would take it for a
//! file from before the job ([`WrittenFiles`]). A file that was there before the job is still
//! replaced or skipped as asked.
//!
//! # Cancel
//!
//! [`ExportQueue::cancel`] stops after the current tile (rendering) or band (encoding), deletes the
//! partial file and skips the rest of the job. Files already finished stay.
//!
//! # The updater
//!
//! `docs/12`: "An update never interrupts an export." [`ExportQueue::is_busy`] is true from the
//! moment a job is queued until the last one has finished or been cancelled; the application keeps
//! "Restart to update" disabled while it is, and [`ExportEvent::Busy`] tells it when that changes.
//!
//! # After the job
//!
//! [`ExportQueue::summary`] keeps the last jobs' results: counts, the files written and the
//! destination folders ("N photos saved to {folder}."), and [`ExportQueue::reveal_path`] is what
//! "Show in Finder" opens.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use ym_export::ExportOptions;
use ym_gpu::queue::CancelToken;
use ym_raw::decode::RawImage;

use crate::export::{
    ExportContext, ExportControl, ExportError, ExportParams, ExportReport, PhotoSnapshot,
    WrittenFiles, estimate, export_photo, planned_destination, predicted_name_in,
};
use crate::session::{ApplyScope, PhotoId, PhotoStatus, Session, SessionError};
use crate::settings::DevelopSettings;

/// Identifies an export job for the life of the process (`ExportJobId` in `src/api/types.ts`).
pub type ExportJobId = u64;

/// What to export and how (`ExportJob` in `src/api/types.ts`). Settings are snapshotted when the
/// job is queued.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportJob {
    /// This photo, the picked photos, or all.
    pub scope: ApplyScope,
    /// Format, quality, bits, compression, colour space, size, destination, name template and what
    /// to do when the file exists.
    pub options: ExportOptions,
}

/// The size and time of a job, and two facts the panel shows (`ExportEstimate` in
/// `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportEstimate {
    /// Total size of the files in bytes.
    pub bytes: u64,
    /// Time the job will take, in seconds.
    pub seconds: f64,
    /// Photos in the job with adjustments beyond the look, which the command line preview cannot
    /// carry.
    pub adjusted_photos: usize,
    /// A DNG of this job is close to, not the same as, the picture in the app: a photo in the job uses a
    /// look with its own display rendering (LogC3 to Rec.709) at an intensity above 0, or ARRI's display
    /// rendering is chosen in Preferences. A DNG then holds the scene values that ACES 2 turns into that
    /// picture (`docs/09`, "What a DNG carries"), which Adobe's renderer shows approximately.
    pub dng_approximate: bool,
}

/// The `export_progress` event (`ExportProgress` in `src/api/types.ts`; see the module
/// documentation for the order of events).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportProgress {
    /// The job.
    pub job_id: ExportJobId,
    /// The output file name now being written.
    pub file: String,
    /// Files finished.
    pub done: usize,
    /// Files in the job.
    pub total: usize,
    /// Progress of `file`, 0 to 100.
    pub file_percent: Option<f64>,
    /// Why `file` failed, in plain words.
    pub error: Option<String>,
    /// `file` already existed and "If it exists" is Skip.
    pub skipped: bool,
}

/// A photo of a job that has been dealt with, for the CLI's one JSON object per file.
#[derive(Debug, Clone, PartialEq)]
pub struct FileDone {
    /// The job.
    pub job_id: ExportJobId,
    /// The photo.
    pub photo: PhotoId,
    /// Its position in the job, from 0.
    pub index: usize,
    /// Photos in the job.
    pub total: usize,
    /// Milliseconds spent reading the file (it overlapped the render of the photo before it).
    pub decode_ms: f64,
    /// What `export_photo` reported, or why it failed.
    pub outcome: Result<ExportReport, FileFailure>,
}

/// Why a photo of a job could not be exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFailure {
    /// A short kind (`look`, `render`, `write`, `decode`, `unsupported`; see `ExportError::kind`).
    pub kind: &'static str,
    /// What went wrong, in plain words.
    pub message: String,
    /// The CLI exit code this failure stands for (3 for a look that is gone).
    pub exit_code: i32,
}

/// What the queue tells the application.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExportEvent {
    /// `export_progress`.
    Progress(ExportProgress),
    /// A photo was exported, skipped or failed: sent just before the `Progress` event that counts
    /// it. Not sent for a cancelled photo.
    File(FileDone),
    /// The queue became busy (`true`) or idle (`false`): the updater's "wait for the queue".
    Busy(bool),
}

/// Where the queue's events go. Runs on the worker thread, so a callback must be quick; forward to
/// a channel or to the webview.
#[derive(Clone)]
pub struct ExportSink(Arc<dyn Fn(ExportEvent) + Send + Sync>);

impl ExportSink {
    /// A sink that calls `f` for every event.
    pub fn new(f: impl Fn(ExportEvent) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// A sink that drops every event.
    #[must_use]
    pub fn none() -> Self {
        Self::new(|_| {})
    }

    /// A sink and the channel the events arrive on (tests, the CLI).
    #[must_use]
    pub fn channel() -> (Self, mpsc::Receiver<ExportEvent>) {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        (
            Self::new(move |e| {
                let _ = tx.lock().unwrap_or_else(PoisonError::into_inner).send(e);
            }),
            rx,
        )
    }

    /// Delivers an event. A callback that panics (a webview that is gone) loses that event and
    /// nothing else: the queue's thread and the job it runs go on.
    fn send(&self, event: ExportEvent) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.0)(event))).is_err() {
            tracing::error!("an export event listener panicked");
        }
    }
}

impl std::fmt::Debug for ExportSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExportSink")
    }
}

/// Why a job could not be queued.
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    /// The scope holds no photo.
    #[error("{0}")]
    NoPhotos(&'static str),
    /// The options cannot be used.
    #[error("{0}")]
    Options(String),
    /// Photos of the scope have no estimates yet, so there is nothing to snapshot. (A photo whose
    /// estimates exist but may still move, because a photo of its burst or its noise pool is not
    /// done, is queued and waits.)
    #[error("{}", developing_message(*.0))]
    Developing(usize),
    /// A photo could not be snapshotted (it was skipped, or is gone).
    #[error(transparent)]
    Photo(#[from] ExportError),
    /// The queue was shut down.
    #[error("Yellowmoon is shutting down.")]
    Shutdown,
    /// The queue's thread could not be started.
    #[error(
        "Yellowmoon could not start its export thread ({0}). Close other apps and open it again."
    )]
    Thread(String),
}

/// Where a job is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Waiting for the jobs before it.
    Queued,
    /// Its photos are being exported.
    Running,
    /// Every photo was dealt with (written, skipped or failed).
    Finished,
    /// Cancelled before it finished.
    Cancelled,
}

/// A photo of a job that could not be exported.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileError {
    /// The file name it would have had.
    pub file: String,
    /// What went wrong, in plain words.
    pub message: String,
}

/// How a job stands (see [`ExportQueue::summary`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobSummary {
    /// The job.
    pub id: ExportJobId,
    /// Where it is.
    pub state: JobState,
    /// Photos in the job.
    pub total: usize,
    /// Photos dealt with.
    pub done: usize,
    /// Photos written.
    pub saved: usize,
    /// Photos skipped because the file exists.
    pub skipped: usize,
    /// Photos that failed.
    pub failed: usize,
    /// The files written, in order. A photo skipped because its file exists writes nothing and is
    /// counted in `skipped`, not listed here (the CLI prints the same: a skipped file prints
    /// nothing, `docs/10`).
    pub files: Vec<PathBuf>,
    /// The folders of the job's files, each once: those of the files written and also those of the
    /// files that were skipped, so that "Show in Finder" opens the destination of a job that wrote
    /// nothing because everything was already there.
    pub folders: Vec<PathBuf>,
    /// The photos that failed, with the reasons.
    pub errors: Vec<FileError>,
}

impl JobSummary {
    /// What "Show in Finder" opens: the file when the job wrote exactly one, else the first folder
    /// (a job that wrote nothing, every file being there already, opens its destination).
    #[must_use]
    pub fn reveal_path(&self) -> Option<PathBuf> {
        match self.files.as_slice() {
            [one] => Some(one.clone()),
            _ => self.folders.first().cloned(),
        }
    }
}

struct Item {
    snapshot: PhotoSnapshot,
    /// The estimates in the snapshot were final when the job was queued (`Session::estimates_final`):
    /// nothing to wait for. When false, the photo's turn waits for them and reads them again.
    settled: bool,
}

struct Job {
    id: ExportJobId,
    options: ExportOptions,
    remove_location: bool,
    items: Vec<Item>,
    cancel: CancelToken,
    session: Arc<Session>,
}

struct State {
    pending: VecDeque<Job>,
    running: Option<(ExportJobId, CancelToken)>,
    summaries: VecDeque<JobSummary>,
    next_id: ExportJobId,
    shutdown: bool,
    /// `Busy(true)` has been sent and `Busy(false)` has not: the worker alone sends both, so the
    /// events come in order and before the first progress event.
    announced_busy: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    idle: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How many finished jobs the queue remembers.
const KEPT_SUMMARIES: usize = 16;

/// How often a job that waits for the estimates of its photo to become final looks again.
const SETTLE_POLL: Duration = Duration::from_millis(25);

/// The sentence for [`QueueError::Developing`].
fn developing_message(count: usize) -> &'static str {
    if count == 1 {
        "One photo is still being developed. Wait until it is ready, then export again."
    } else {
        "Some photos are still being developed. Wait until they are ready, then export again."
    }
}

/// The export queue (see the module documentation).
pub struct ExportQueue {
    ctx: Arc<ExportContext>,
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// Why the worker thread could not start, when it could not.
    thread_error: Option<String>,
}

impl std::fmt::Debug for ExportQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportQueue")
            .field("busy", &self.is_busy())
            .finish_non_exhaustive()
    }
}

impl ExportQueue {
    /// A queue that exports through `ctx` and reports to `sink`. Starts its worker thread.
    #[must_use]
    pub fn new(ctx: Arc<ExportContext>, sink: ExportSink) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                pending: VecDeque::new(),
                running: None,
                summaries: VecDeque::new(),
                next_id: 1,
                shutdown: false,
                announced_busy: false,
            }),
            wake: Condvar::new(),
            idle: Condvar::new(),
        });
        let spawned = {
            let (ctx, sink, shared) = (Arc::clone(&ctx), sink.clone(), Arc::clone(&shared));
            std::thread::Builder::new()
                .name("ym-export-queue".to_owned())
                .spawn(move || worker_loop(&ctx, &sink, &shared))
        };
        let (worker, thread_error) = match spawned {
            Ok(handle) => (Some(handle), None),
            Err(e) => {
                tracing::error!(error = %e, "the export queue thread could not start");
                (None, Some(e.to_string()))
            }
        };
        Self {
            ctx,
            shared,
            worker: Mutex::new(worker),
            thread_error,
        }
    }

    /// The context the queue exports through.
    #[must_use]
    pub fn context(&self) -> &Arc<ExportContext> {
        &self.ctx
    }

    // --- Estimate -------------------------------------------------------------------------

    /// `export_estimate`: the size and time of a job, how many of its photos have adjustments
    /// beyond the look, and whether a DNG is out of the question. Photos that are not developed
    /// yet are left out.
    #[must_use]
    pub fn estimate(&self, session: &Session, job: &ExportJob) -> ExportEstimate {
        let views: Vec<_> = session
            .scope_ids(job.scope)
            .into_iter()
            .filter_map(|id| session.view(id).ok())
            .collect();
        // The detail of each photo's newest thumbnail makes the size estimate: a smooth frame and a
        // busy one of the same pixel count differ by a factor of three or more in a JPEG.
        let renderer = self.ctx.renderer();
        let est = estimate(
            &self.ctx,
            views
                .iter()
                .map(|v| (&*v.meta, &v.settings, renderer.thumbnail_detail(v.id))),
            &job.options,
        );
        let settings: Vec<&DevelopSettings> = views.iter().map(|v| &v.settings).collect();
        let (command_look, command_intensity) = crate::cli_preview::command_look(&settings);
        let command_output_sharpening = crate::cli_preview::command_output_sharpening(&settings);
        let looks = self.ctx.renderer().looks();
        ExportEstimate {
            bytes: est.bytes,
            seconds: est.seconds,
            adjusted_photos: crate::cli_preview::adjusted_photos_with(
                &settings,
                &command_look,
                command_intensity,
                command_output_sharpening,
            ),
            dng_approximate: matches!(
                self.ctx.renderer().display(),
                crate::pipeline::DisplayChoice::ArriLut(_)
            ) || views.iter().any(|v| {
                v.settings.look_intensity > 0.0
                    && looks
                        .look(&v.settings.look)
                        .is_ok_and(|l| l.transform.includes_display_rendering())
            }),
        }
    }

    // --- Queueing -------------------------------------------------------------------------

    /// `export_start`: snapshots the photos of the job's scope and queues the job. Returns at once;
    /// progress arrives as [`ExportEvent::Progress`]. `remove_location` is the preference "Remove
    /// location from exports" as it is now.
    ///
    /// The burst median exposure and the pooled noise depend on the other photos of the session
    /// (`docs/01` Session). A photo whose estimates are not final yet (a photo of its burst or of
    /// its noise pool is still being developed) is queued all the same: when its turn comes the job
    /// waits until they are (`Session::estimates_final`) and then takes them, so the file is the
    /// one `ym render` writes for the same photos. The wait is for the photos it depends on, not for
    /// thumbnails, prefetching or viewport work.
    ///
    /// # Errors
    ///
    /// [`QueueError::NoPhotos`] for an empty scope, [`QueueError::Options`] for options the writers refuse,
    /// [`QueueError::Developing`] when photos of the scope have no estimates yet and
    /// [`QueueError::Photo`] for a photo that cannot be exported (nothing is queued).
    pub fn start(
        &self,
        session: &Arc<Session>,
        job: &ExportJob,
        remove_location: bool,
    ) -> Result<ExportJobId, QueueError> {
        if let Some(e) = &self.thread_error {
            return Err(QueueError::Thread(e.clone()));
        }
        job.options
            .validate()
            .map_err(|e| QueueError::Options(e.to_string()))?;
        let ids = session.scope_ids(job.scope);
        if ids.is_empty() {
            // The one photo asked for is open but cannot be developed: say why, not "open a photo".
            if job.scope == ApplyScope::This {
                let state = session.state();
                if let Some(PhotoStatus::Skipped { reason }) = state
                    .current_index
                    .and_then(|i| state.photos.get(i))
                    .map(|p| &p.status)
                {
                    return Err(QueueError::Photo(ExportError::Session(
                        SessionError::Skipped(reason.clone()),
                    )));
                }
            }
            return Err(QueueError::NoPhotos(match job.scope {
                ApplyScope::This => "There is no photo to export. Open a photo first.",
                ApplyScope::Picked => {
                    "No photos are picked. Pick some photos, or choose All, then export again."
                }
                ApplyScope::All => "There are no photos to export. Open some photos first.",
            }));
        }
        let mut items = Vec::with_capacity(ids.len());
        let mut developing = 0;
        for id in ids {
            // Asked before the snapshot is taken: estimates that are final now stay final, those
            // that are not are read again when the photo's turn comes.
            let settled = session.estimates_final(id).unwrap_or(true);
            match PhotoSnapshot::take(session, id) {
                Ok(snapshot) => items.push(Item { snapshot, settled }),
                Err(ExportError::Session(SessionError::NotReady(_))) => developing += 1,
                Err(e) => return Err(e.into()),
            }
        }
        if developing > 0 {
            return Err(QueueError::Developing(developing));
        }
        // Apple Auto asks about these photos first, after the one on screen, each with the frames of
        // its burst that its export also waits for (`docs/19` section 6).
        let waiting: Vec<PhotoId> = items
            .iter()
            .filter(|item| !item.settled)
            .map(|item| item.snapshot.view.id)
            .collect();
        session.prioritise_apple_auto(&waiting);
        let mut st = lock(&self.shared.state);
        if st.shutdown {
            return Err(QueueError::Shutdown);
        }
        let id = st.next_id;
        st.next_id += 1;
        st.summaries.push_back(JobSummary {
            id,
            state: JobState::Queued,
            total: items.len(),
            done: 0,
            saved: 0,
            skipped: 0,
            failed: 0,
            files: Vec::new(),
            folders: Vec::new(),
            errors: Vec::new(),
        });
        while st.summaries.len() > KEPT_SUMMARIES {
            st.summaries.pop_front();
        }
        st.pending.push_back(Job {
            id,
            options: job.options.clone(),
            remove_location,
            items,
            cancel: CancelToken::new(),
            session: Arc::clone(session),
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(id)
    }

    /// `export_cancel`: stops the job after the current tile and deletes the partial file. A job
    /// that has not started is dropped. Does nothing for a job that has finished.
    pub fn cancel(&self, id: ExportJobId) {
        let mut st = lock(&self.shared.state);
        if let Some((running, token)) = &st.running
            && *running == id
        {
            token.cancel();
            return;
        }
        if let Some(i) = st.pending.iter().position(|j| j.id == id) {
            st.pending.remove(i);
            if let Some(s) = st.summaries.iter_mut().find(|s| s.id == id) {
                s.state = JobState::Cancelled;
            }
            drop(st);
            self.shared.idle.notify_all();
        }
    }

    /// Cancels every job (the application is quitting).
    pub fn cancel_all(&self) {
        let ids: Vec<ExportJobId> = {
            let st = lock(&self.shared.state);
            st.pending
                .iter()
                .map(|j| j.id)
                .chain(st.running.as_ref().map(|(id, _)| *id))
                .collect()
        };
        for id in ids {
            self.cancel(id);
        }
    }

    /// Whether a job is queued or running: the updater waits while this is true.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        let st = lock(&self.shared.state);
        st.running.is_some() || !st.pending.is_empty()
    }

    /// Waits until the queue is idle; false if `timeout` passed first.
    #[must_use]
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let st = lock(&self.shared.state);
        let (st, _) = self
            .shared
            .idle
            .wait_timeout_while(st, timeout, |s| {
                s.running.is_some() || !s.pending.is_empty()
            })
            .unwrap_or_else(PoisonError::into_inner);
        st.running.is_none() && st.pending.is_empty()
    }

    /// How a job stands, for as long as the queue remembers it (the last sixteen).
    #[must_use]
    pub fn summary(&self, id: ExportJobId) -> Option<JobSummary> {
        lock(&self.shared.state)
            .summaries
            .iter()
            .find(|s| s.id == id)
            .cloned()
    }

    /// What "Show in Finder" opens for a job: its one file, else its first destination folder.
    #[must_use]
    pub fn reveal_path(&self, id: ExportJobId) -> Option<PathBuf> {
        self.summary(id).and_then(|s| s.reveal_path())
    }
}

impl Drop for ExportQueue {
    fn drop(&mut self) {
        {
            let mut st = lock(&self.shared.state);
            st.shutdown = true;
            if let Some((_, token)) = &st.running {
                token.cancel();
            }
            for j in &st.pending {
                j.cancel.cancel();
            }
        }
        self.shared.wake.notify_all();
        if let Some(handle) = lock(&self.worker).take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------------------------

fn worker_loop(ctx: &Arc<ExportContext>, sink: &ExportSink, shared: &Arc<Shared>) {
    loop {
        let (job, announce) = {
            let mut st = lock(&shared.state);
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(job) = st.pending.pop_front() {
                    st.running = Some((job.id, job.cancel.clone()));
                    if let Some(s) = st.summaries.iter_mut().find(|s| s.id == job.id) {
                        s.state = JobState::Running;
                    }
                    let announce = !st.announced_busy;
                    st.announced_busy = true;
                    break (job, announce);
                }
                st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        if announce {
            sink.send(ExportEvent::Busy(true));
        }
        // A panic in one job must not take the queue down with it.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(ctx, sink, shared, &job);
        }));
        if outcome.is_err() {
            tracing::error!(job = job.id, "an export job panicked");
            // The panel waits for the last event of the job; without one it would show a file at
            // some percent for ever. (The sink may be what panicked, so this is guarded too.)
            let total = job.items.len();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sink.send(ExportEvent::Progress(ExportProgress {
                    job_id: job.id,
                    file: String::new(),
                    done: total,
                    total,
                    file_percent: None,
                    error: Some(
                        "The export stopped because of an unexpected problem. Try it again."
                            .to_owned(),
                    ),
                    skipped: false,
                }));
            }));
        }
        let idle = {
            let mut st = lock(&shared.state);
            st.running = None;
            let cancelled = job.cancel.is_cancelled() || outcome.is_err();
            if let Some(s) = st.summaries.iter_mut().find(|s| s.id == job.id) {
                s.state = if cancelled && s.done < s.total {
                    JobState::Cancelled
                } else {
                    JobState::Finished
                };
            }
            let idle = st.pending.is_empty();
            if idle {
                st.announced_busy = false;
            }
            idle
        };
        if idle {
            sink.send(ExportEvent::Busy(false));
            shared.idle.notify_all();
        }
    }
}

/// Reads a photo's decoded image: from the session's cache when it is there, else from the file.
/// Returns the image and the milliseconds it took.
fn load_raw(session: &Session, snapshot: &PhotoSnapshot) -> Result<(Arc<RawImage>, f64), String> {
    let started = std::time::Instant::now();
    let raw = match session.raw_image(snapshot.view.id) {
        Ok(raw) => raw,
        Err(SessionError::UnknownPhoto(_)) => {
            // The session was closed or the photo removed meanwhile: the job goes on from the file.
            Arc::new(ym_raw::decode::decode(&snapshot.view.path).map_err(|e| e.to_string())?)
        }
        Err(e) => return Err(e.to_string()),
    };
    Ok((raw, started.elapsed().as_secs_f64() * 1000.0))
}

/// How waiting for the final estimates of a photo ended.
enum Settled {
    /// The job was cancelled meanwhile.
    Cancelled,
    /// The snapshot with the final estimates in it.
    Fresh(Box<PhotoSnapshot>),
    /// The photo cannot be asked about any more (the session was closed, which stops development,
    /// or the photo was skipped): the job goes on with the snapshot it was queued with.
    AsQueued,
}

/// Waits until the estimates of the snapshot's photo are final in `session` and returns the
/// snapshot with them. Looks again every [`SETTLE_POLL`], so a cancel is noticed within that.
fn wait_for_final_estimates(
    session: &Session,
    cancel: &CancelToken,
    snapshot: &PhotoSnapshot,
) -> Settled {
    let id = snapshot.view.id;
    loop {
        if cancel.is_cancelled() {
            return Settled::Cancelled;
        }
        match session.estimates_final(id) {
            Ok(true) => break,
            Ok(false) => std::thread::sleep(SETTLE_POLL),
            Err(e) => {
                tracing::info!(photo = id.0, reason = %e, "exporting with the estimates the job was queued with");
                return Settled::AsQueued;
            }
        }
    }
    match session
        .view(id)
        .map_err(ExportError::from)
        .and_then(|fresh| snapshot.with_estimates_of(fresh))
    {
        Ok(fresh) => Settled::Fresh(Box::new(fresh)),
        Err(e) => {
            tracing::warn!(photo = id.0, error = %e, "the final estimates could not be read; exporting with the queued ones");
            Settled::AsQueued
        }
    }
}

fn update_summary(shared: &Shared, id: ExportJobId, f: impl FnOnce(&mut JobSummary)) {
    if let Some(s) = lock(&shared.state)
        .summaries
        .iter_mut()
        .find(|s| s.id == id)
    {
        f(s);
    }
}

fn run_job(ctx: &Arc<ExportContext>, sink: &ExportSink, shared: &Arc<Shared>, job: &Job) {
    let total = job.items.len();
    // The files this job has written: a later photo with the same name is numbered, whatever
    // "If the file exists" says, instead of replacing or being skipped for an earlier photo.
    let written = WrittenFiles::new();
    let spawn_load = |i: usize| {
        let session = Arc::clone(&job.session);
        let snapshot = job.items[i].snapshot.clone();
        std::thread::Builder::new()
            .name("ym-export-decode".to_owned())
            .spawn(move || load_raw(&session, &snapshot))
    };
    let mut next_load = (total > 0).then(|| spawn_load(0));
    for (i, item) in job.items.iter().enumerate() {
        let loaded = next_load.take();
        if job.cancel.is_cancelled() {
            // Wait for the decode thread so nothing outlives the job.
            if let Some(Ok(h)) = loaded {
                let _ = h.join();
            }
            return;
        }
        // The next photo decodes while this one renders.
        if i + 1 < total {
            next_load = Some(spawn_load(i + 1));
        }
        let raw = match loaded {
            Some(Ok(handle)) => handle
                .join()
                .unwrap_or_else(|_| Err("The photo could not be read.".to_owned())),
            Some(Err(e)) => Err(format!("The photo could not be read: {e}.")),
            None => Err("The photo could not be read.".to_owned()),
        };
        let decode_ms = raw.as_ref().map_or(0.0, |(_, ms)| *ms);

        let snapshot = &item.snapshot;
        let dest_dir = planned_destination(ctx, &snapshot.view.path, &job.options);
        // The name needs the metadata only, so it is known before the photo is loaded.
        let predicted = {
            let name = snapshot.planned_file_name(&job.options, i as u32 + 1);
            match &dest_dir {
                Some(dir) => predicted_name_in(
                    dir,
                    &name,
                    written.if_exists_for(dir, &name, job.options.if_exists),
                ),
                None => name,
            }
        };
        let event = |done: usize,
                     file: &str,
                     percent: Option<f64>,
                     error: Option<String>,
                     skipped: bool| {
            sink.send(ExportEvent::Progress(ExportProgress {
                job_id: job.id,
                file: file.to_owned(),
                done,
                total,
                file_percent: percent,
                error,
                skipped,
            }));
        };
        event(i, &predicted, Some(0.0), None, false);

        // A job queued while the photos this one depends on were still developing waits here for
        // their estimates, then reads the shared ones (burst median, pooled noise) again.
        let refreshed = if item.settled {
            None
        } else {
            match wait_for_final_estimates(&job.session, &job.cancel, &item.snapshot) {
                Settled::Cancelled => {
                    if let Some(Ok(h)) = next_load.take() {
                        let _ = h.join();
                    }
                    return;
                }
                Settled::Fresh(snapshot) => Some(*snapshot),
                Settled::AsQueued => None,
            }
        };
        let snapshot = refreshed.as_ref().unwrap_or(&item.snapshot);

        let result = raw.map_err(ExportError::Decode).and_then(|(raw, _)| {
            let photo = snapshot.clone().with_raw(raw);
            let last = Mutex::new(0u32);
            let report = |f: f32| {
                // A cancelled job says nothing more, also from a writer thread that is finishing.
                if job.cancel.is_cancelled() {
                    return;
                }
                let pct = (f64::from(f) * 100.0).floor().clamp(0.0, 99.0) as u32;
                let mut l = lock(&last);
                if pct > *l {
                    *l = pct;
                    // Sent while the lock is held: two writer threads reporting at once cannot
                    // swap two percentages on the way to the panel.
                    event(i, &predicted, Some(f64::from(pct)), None, false);
                }
            };
            export_photo(
                ctx,
                &photo,
                &ExportParams {
                    options: job.options.clone(),
                    remove_location: job.remove_location,
                    seq: i as u32 + 1,
                    written: Some(written.clone()),
                },
                &ExportControl {
                    cancel: job.cancel.clone(),
                    progress: Some(&report),
                },
            )
        });

        match result {
            Ok(report) => {
                let name = report
                    .path
                    .file_name()
                    .map_or_else(|| predicted.clone(), |n| n.to_string_lossy().into_owned());
                update_summary(shared, job.id, |s| {
                    s.done += 1;
                    if report.skipped {
                        s.skipped += 1;
                    } else {
                        s.saved += 1;
                        s.files.push(report.path.clone());
                    }
                    if let Some(dir) = report.path.parent()
                        && !s.folders.iter().any(|f| f == dir)
                    {
                        s.folders.push(dir.to_path_buf());
                    }
                });
                sink.send(ExportEvent::File(FileDone {
                    job_id: job.id,
                    photo: snapshot.view.id,
                    index: i,
                    total,
                    decode_ms,
                    outcome: Ok(report.clone()),
                }));
                if report.skipped {
                    event(i + 1, &name, Some(100.0), None, true);
                } else if i + 1 == total {
                    event(total, &name, Some(100.0), None, false);
                }
            }
            Err(ExportError::Cancelled) => {
                // Nothing more is reported for a cancelled job.
                return;
            }
            Err(e) => {
                let message = e.to_string();
                update_summary(shared, job.id, |s| {
                    s.done += 1;
                    s.failed += 1;
                    s.errors.push(FileError {
                        file: predicted.clone(),
                        message: message.clone(),
                    });
                });
                // The log can end up in a bug report (`docs/01`): the kind of failure and the job at
                // warn level, the file name and the message (which names the folder) at debug.
                tracing::warn!(job = job.id, kind = e.kind(), "export failed");
                tracing::debug!(job = job.id, file = %predicted, error = %message, "export failure details");
                sink.send(ExportEvent::File(FileDone {
                    job_id: job.id,
                    photo: snapshot.view.id,
                    index: i,
                    total,
                    decode_ms,
                    outcome: Err(FileFailure {
                        kind: e.kind(),
                        message: message.clone(),
                        exit_code: e.exit_code(),
                    }),
                }));
                event(i + 1, &predicted, None, Some(message), false);
            }
        }
    }
}
