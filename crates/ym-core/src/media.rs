//! Slow and hung media (`docs/26-hardening.md` P1-7, P3-4, task H-15).
//!
//! A file on a sleeping network share or a pulled drive can block `open`, `pread` or `stat` inside
//! the kernel for as long as the volume stays away, and a blocked system call cannot be cancelled.
//! [`Media`] is the one place the session touches files that can do that: it runs each probe and
//! each whole-file read on a **detached thread** of its own and waits for the answer for at most a
//! time limit. A late answer is dropped (the thread ends when its call finally returns); the photo
//! is `Skipped` with [`NO_ANSWER_MESSAGE`] and the caller carries on.
//!
//! # Limits
//!
//! The limits are starting values chosen by reasoning, not measured: `docs/26` MC-2 (a Mac check
//! on a slept SMB share) confirms or changes them.
//!
//! * [`PROBE_LIMIT`], 15 s. A probe reads a few kilobytes (about 120 `pread` calls, at most
//!   64 MiB). A spun-down hard disk answers in 5 to 10 s and a share that is waking up in 10 to 20 s,
//!   so 15 s lets most sleeping media wake and still returns `open_paths` within a minute for a dead
//!   share.
//! * [`READ_LIMIT`], 60 s, for a develop job's whole read and decode of one file (a 61 MP file is
//!   about 120 MB: at 2 MB/s from a poor share that is the limit, and the decode itself is
//!   seconds). It is long on purpose: a job that fails costs the photo, and a person can open the
//!   file again.
//!
//! # How many threads can hang
//!
//! A thread that did not answer in time is **hung** until its call returns, perhaps never. To keep
//! a dead volume from taking a thread for each of 2000 files, the hung threads are counted per
//! folder and in all: when [`MAX_HUNG_PER_FOLDER`] threads of one folder are hung, the next files of
//! that folder get the same answer **at once**, without a thread (the folder evidently does not
//! answer); the same when [`MAX_HUNG_TOTAL`] are hung in all. A folder that answers is never
//! refused because another one hung. So the threads this module has alive are at most
//! `MAX_HUNG_TOTAL` hung ones plus the callers that wait at that moment (one per rayon thread for
//! probes, at most the decode pool for reads). At 100 files that all hang in one folder, at most
//! `MAX_HUNG_PER_FOLDER` plus the callers in flight threads are ever started; the rest are refused
//! without one.
//!
//! # A file that went away or changed (`docs/26` P1-4, H-04)
//!
//! The probe keeps a [`FileStamp`] (length and modification time) of the file it read. Every later
//! whole-file read ([`Media::decode`]) compares the file's stamp with it with one `stat` **inside
//! the same guarded call** as the read, so a dead share cannot hang that `stat` outside the time
//! limit. A file that is gone reads as `NotFound` (the catalogue sentence of `ym_raw::error`); a
//! file whose length or modification time differs is [`ReadFailure::Changed`], and the caller shows
//! [`CHANGED_MESSAGE`].
//!
//! The stamp does not see a rewrite of the same length within the file system's time granularity
//! (1 s on HFS+ and many SMB shares, 2 s on FAT). No hash is kept: that would mean reading the
//! whole file at open.
//!
//! The listing of a folder (`read_dir`) and the canonical path of a file are not guarded: a share
//! that hangs there still hangs `open_paths`. That is for MC-2 to show.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use ym_raw::decode::RawImage;
use ym_raw::error::RawError;
use ym_raw::metadata::RawMetadata;

/// What a photo says when its file did not answer in time.
pub const NO_ANSWER_MESSAGE: &str =
    "This file didn't answer. Check the drive or the network connection, then open it again.";

/// What a photo says when its file was replaced or edited after it was opened.
pub const CHANGED_MESSAGE: &str =
    "This file changed after it was opened. Remove it from the filmstrip and open it again.";

/// How long one probe may take. A starting value; see the module documentation (MC-2).
pub const PROBE_LIMIT: Duration = Duration::from_secs(15);

/// How long a develop job's read of one file may take. A starting value; see the module
/// documentation (MC-2).
pub const READ_LIMIT: Duration = Duration::from_secs(60);

/// Hung threads allowed for one folder before its next files are refused without a thread.
pub const MAX_HUNG_PER_FOLDER: usize = 8;

/// Hung threads allowed in all before every next file is refused without a thread.
pub const MAX_HUNG_TOTAL: usize = 32;

/// How often a waiting caller looks at its cancel flag.
const POLL: Duration = Duration::from_millis(25);

/// How a guarded call ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Guarded<T> {
    /// The call answered in time.
    Done(T),
    /// It did not answer in time, or its folder already has too many hung threads.
    NoAnswer,
    /// The caller was cancelled while it waited.
    Cancelled,
}

/// What `stat` said about a file when it was probed: its length and modification time. Two stamps
/// that differ mean the file is not the one that was opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    /// The length in bytes.
    pub len: u64,
    /// The modification time, when the file system has one.
    pub modified: Option<std::time::SystemTime>,
}

impl FileStamp {
    /// The stamp of the file at `path` (one `stat`, following symlinks).
    ///
    /// # Errors
    ///
    /// The error of `stat` (`NotFound` for a file that is gone).
    pub fn of(path: &Path) -> io::Result<Self> {
        let meta = std::fs::metadata(path)?;
        Ok(Self {
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

/// Why a whole-file read gave no image.
#[derive(Debug)]
pub enum ReadFailure {
    /// The file could not be checked or decoded (`NotFound` when it is gone).
    Raw(RawError),
    /// The file's length or modification time differs from the probe's: [`CHANGED_MESSAGE`].
    Changed,
}

impl ReadFailure {
    /// True when the file is gone (`NotFound`) or differs from the probe: the photo is skipped
    /// for it. Any other failure is the read's own and leaves the photo as it was.
    #[must_use]
    pub fn is_gone_or_changed(&self) -> bool {
        match self {
            Self::Changed => true,
            Self::Raw(RawError::Io(e)) => e.kind() == io::ErrorKind::NotFound,
            Self::Raw(_) => false,
        }
    }

    /// The sentence a photo is skipped with.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Raw(error) => error.to_string(),
            Self::Changed => CHANGED_MESSAGE.to_owned(),
        }
    }
}

/// How a file is probed (the standard one is `ym_raw::metadata::probe`).
pub type ProbeFn = Arc<dyn Fn(&Path) -> Result<RawMetadata, RawError> + Send + Sync>;
/// How a file is read and decoded (the standard one is `ym_raw::decode::decode`).
pub type DecodeFn = Arc<dyn Fn(&Path) -> Result<RawImage, RawError> + Send + Sync>;
/// How a path is made canonical (the standard one is `std::fs::canonicalize`).
pub type CanonicalFn = Arc<dyn Fn(&Path) -> io::Result<PathBuf> + Send + Sync>;

#[derive(Debug, Default)]
struct Hung {
    total: usize,
    by_folder: HashMap<PathBuf, usize>,
}

#[derive(Debug, Default)]
struct Counters {
    hung: Mutex<Hung>,
    started: AtomicU64,
}

/// The session's access to files that may not answer. The standard value ([`Media::new`]) uses the
/// real calls and the limits above; the fields are public so that a test can slow a call down or
/// shorten a limit without a slow drive (the same kind of seam as the folder lister of
/// `session.rs`).
#[derive(Clone)]
pub struct Media {
    /// How long a probe may take.
    pub probe_limit: Duration,
    /// How long a develop job's read of one file may take.
    pub read_limit: Duration,
    /// The probe.
    pub probe: ProbeFn,
    /// The whole-file read and decode.
    pub decode: DecodeFn,
    /// The canonical path.
    pub canonicalize: CanonicalFn,
    counters: Arc<Counters>,
}

impl std::fmt::Debug for Media {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Media")
            .field("probe_limit", &self.probe_limit)
            .field("read_limit", &self.read_limit)
            .finish_non_exhaustive()
    }
}

impl Default for Media {
    fn default() -> Self {
        Self::new()
    }
}

impl Media {
    /// The real calls with the standard limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            probe_limit: PROBE_LIMIT,
            read_limit: READ_LIMIT,
            probe: Arc::new(|path| ym_raw::metadata::probe(path)),
            decode: Arc::new(|path| ym_raw::decode::decode(path)),
            canonicalize: Arc::new(|path| std::fs::canonicalize(path)),
            counters: Arc::default(),
        }
    }

    /// The canonical form of `path`, or `path` itself when it cannot be made (not guarded).
    #[must_use]
    pub fn canonical(&self, path: &Path) -> PathBuf {
        (self.canonicalize)(path).unwrap_or_else(|_| path.to_path_buf())
    }

    /// Probes `path` with the probe limit. `cancelled` is looked at every 25 ms while waiting.
    ///
    /// The answer carries the file's [`FileStamp`], read just before the probe (so a file that
    /// changes while it is probed is seen as changed later); `None` when the `stat` failed.
    pub fn probe(
        &self,
        path: &Path,
        cancelled: &dyn Fn() -> bool,
    ) -> Guarded<Result<(RawMetadata, Option<FileStamp>), RawError>> {
        let call = Arc::clone(&self.probe);
        let owned = path.to_path_buf();
        self.guarded(path, self.probe_limit, cancelled, move || {
            let stamp = FileStamp::of(&owned).ok();
            call(&owned).map(|meta| (meta, stamp))
        })
    }

    /// Reads and decodes `path` with the read limit. With a `stamp`, the file is first compared
    /// with it (one `stat` inside the same guarded call): a file that is gone is
    /// [`ReadFailure::Raw`] with `NotFound`, a file that differs is [`ReadFailure::Changed`], and
    /// nothing is decoded for either.
    pub fn decode(
        &self,
        path: &Path,
        stamp: Option<FileStamp>,
        cancelled: &dyn Fn() -> bool,
    ) -> Guarded<Result<RawImage, ReadFailure>> {
        let call = Arc::clone(&self.decode);
        let owned = path.to_path_buf();
        self.guarded(path, self.read_limit, cancelled, move || {
            if let Some(expected) = stamp {
                match FileStamp::of(&owned) {
                    Ok(now) if now == expected => {}
                    Ok(_) => return Err(ReadFailure::Changed),
                    Err(error) => return Err(ReadFailure::Raw(RawError::Io(error))),
                }
            }
            call(&owned).map_err(ReadFailure::Raw)
        })
    }

    /// Threads that did not answer in time and have not returned yet.
    #[must_use]
    pub fn hung_threads(&self) -> usize {
        self.hung().total
    }

    /// Threads this value has started in all (finished, hung or running): what a test bounds.
    #[must_use]
    pub fn threads_started(&self) -> u64 {
        self.counters.started.load(Ordering::SeqCst)
    }

    fn hung(&self) -> std::sync::MutexGuard<'_, Hung> {
        self.counters
            .hung
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `work` on a detached thread and waits for it for `limit`.
    fn guarded<T: Send + 'static>(
        &self,
        path: &Path,
        limit: Duration,
        cancelled: &dyn Fn() -> bool,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Guarded<T> {
        let folder = path.parent().map(Path::to_path_buf).unwrap_or_default();
        {
            let hung = self.hung();
            if hung.total >= MAX_HUNG_TOTAL
                || hung.by_folder.get(&folder).copied().unwrap_or(0) >= MAX_HUNG_PER_FOLDER
            {
                return Guarded::NoAnswer;
            }
        }
        // `state`: (the caller gave up, the thread returned). Whoever sets the second of the two
        // takes the thread out of the hung count, so the count is right however they race.
        let state = Arc::new(Mutex::new((false, false)));
        let (send, receive) = channel();
        let thread_state = Arc::clone(&state);
        let thread_counters = Arc::clone(&self.counters);
        let thread_folder = folder.clone();
        self.counters.started.fetch_add(1, Ordering::SeqCst);
        let spawned = std::thread::Builder::new()
            .name("ym-media".to_owned())
            .spawn(move || {
                let answer = work();
                // Nobody waits any more: the answer is dropped.
                let _ = send.send(answer);
                let mut s = thread_state.lock().unwrap_or_else(PoisonError::into_inner);
                s.1 = true;
                if s.0 {
                    release(&thread_counters, &thread_folder);
                }
            });
        if let Err(error) = spawned {
            tracing::error!(%error, "could not start a thread to read a file");
            return Guarded::NoAnswer;
        }
        let give_up = |outcome| {
            let mut s = state.lock().unwrap_or_else(PoisonError::into_inner);
            s.0 = true;
            if !s.1 {
                let mut hung = self.hung();
                hung.total += 1;
                *hung.by_folder.entry(folder.clone()).or_insert(0) += 1;
            }
            outcome
        };
        let deadline = Instant::now() + limit;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return give_up(Guarded::NoAnswer);
            }
            if cancelled() {
                return give_up(Guarded::Cancelled);
            }
            match receive.recv_timeout(POLL.min(deadline - now)) {
                Ok(answer) => return Guarded::Done(answer),
                Err(RecvTimeoutError::Timeout) => {}
                // The thread ended without an answer (a panic in the call).
                Err(RecvTimeoutError::Disconnected) => return Guarded::NoAnswer,
            }
        }
    }
}

/// Takes one thread of `folder` out of the hung count.
fn release(counters: &Counters, folder: &Path) {
    let mut hung = counters.hung.lock().unwrap_or_else(PoisonError::into_inner);
    hung.total = hung.total.saturating_sub(1);
    if let Some(n) = hung.by_folder.get_mut(folder) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            hung.by_folder.remove(folder);
        }
    }
}
