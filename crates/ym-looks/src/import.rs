//! The importer behind the wizard and `ym looks import`: zips, folders, `.cube` files and ARRI Look
//! Files (`docs/08-looks.md` sections Wizard, Any other .cube and ARRI Look Files).
//!
//! Two steps, matching the wizard:
//!
//! 1. [`inspect`] (wizard step 3, "Recognise") reads the source without writing anything: it lists
//!    the `.cube` files and look files (`.alf4`, `.alf4c`, `.aml`), hashes and validates each with
//!    [`ym_color::cube`] or [`ym_color::alf`], and runs the recognition rules
//!    ([`crate::recognize`]). The result says what each file is and explains anything wrong in
//!    plain words.
//! 2. [`import`] (step 4, "Import") copies only those files into the library, grouped as ARRI
//!    groups them, hashes each, and writes the manifest. [`import_path`] does both for the CLI.
//!
//! What import guarantees:
//!
//! * **Originals are read-only** and the copies are byte-for-byte identical (the SHA-256 recorded
//!   in the manifest is that of the source bytes). A look file is kept as it came and read again
//!   when the look is built, so its CDL and LUT always are the file's.
//! * **All or nothing.** Files are first written to hidden temporary names, then renamed into
//!   place, then the manifest is replaced atomically. A failure or a cancel before the manifest is
//!   written removes what was staged and every new file that was already renamed into place. The
//!   one thing a failure in the last two steps can leave behind is an ARRI file that ARRI
//!   re-released and that had already replaced the old copy: the list still records the old
//!   checksum, so `verify` reports that look as changed and importing again repairs it. All of it
//!   happens under `looks/.lock` with the manifest re-read after the lock is taken, so concurrent
//!   imports from the app and the CLI never lose each other's entries.
//! * **Idempotent.** Importing the same files again changes nothing: same id and same hash is
//!   "unchanged". A file that went missing or was altered in the library is written again.
//! * **Validation.** A file must parse as a 3D `.cube` LUT (a 1D shaper in front is fine), size 2
//!   to 129, all values finite, or as an ARRI Look File whose checksum matches and whose look
//!   Yellowmoon can use. Anything else is rejected with a reason, and the rest still imports.
//! * **One look, one entry.** The LogC4 library comes as 3D LUTs and as `.alf4` look files and both
//!   give the same ids. Importing the look files after the `.cube` version leaves the `.cube` (it
//!   has 65 points where the look files have 33); importing the `.cube` version after the look
//!   files replaces them; and when one import holds both (a folder with both libraries), the
//!   `.cube` files are planned first, so the look files are left out the same way.
//! * **The same bytes are never imported twice.** A file whose SHA-256 (with the same input) is
//!   already in the library is "unchanged" whatever it is called now: no `Mine-2` for a renamed
//!   copy. If an ARRI package claims a file that an earlier import filed under "Your LUTs", that
//!   entry becomes the ARRI look in place (see `plan`).

mod plan;
mod source;
pub mod trash;
pub mod watch;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use ym_color::cube::{self, CubeError};
use ym_color::look::LookInput;

use crate::library::{Library, LookFormat, LookId, cube_error_text};
use crate::lock::{DEFAULT_TIMEOUT, LibraryLock, LockError};
use crate::manifest::{
    Manifest, ManifestError, commit_staged, hash_file, sha256_hex, stage_write, sync_dir,
};
use crate::recognize::{
    CubeSummary, FileFacts, LookFileFacts, Outcome, PackageDb, Recognition, Verdict, recognize,
};
use plan::{EntryData, Slot, Spec, SpecKind, apply, assign};
use source::{ReadFailure, Source};

/// Which phase a [`Progress`] report belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Reading, hashing and validating the source's `.cube` files ([`inspect`]).
    Scan,
    /// Copying files into the library ([`import`]).
    Copy,
    /// Re-hashing the library ([`Library::verify`]).
    Verify,
}

/// A progress report: `done` of `total` files, for a progress bar "per file count".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress<'a> {
    /// The phase.
    pub stage: Stage,
    /// Files finished.
    pub done: usize,
    /// Files in the phase.
    pub total: usize,
    /// The file about to be worked on, when there is one.
    pub file: Option<&'a str>,
}

/// A progress callback. Called from the thread doing the work, so a UI should forward the
/// numbers (for example as an event) rather than draw from inside it.
pub type ProgressFn<'a> = &'a mut dyn FnMut(Progress<'_>);

/// A flag the wizard's Cancel button sets. Clones share the flag.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks the running work to stop at the next file.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether [`CancelToken::cancel`] was called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Settings for [`import`].
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// The answer to the generic question for files no known package claims: what the LUT expects
    /// as input. Ignored for files that were recognized.
    pub input: Option<LookInput>,
    /// Refuse to import when a package has fewer files than it should
    /// ([`Verdict::Incomplete`]). Default off: the wizard gates on the recognition itself.
    pub require_complete: bool,
    /// The import time to record (RFC 3339), for reproducible manifests in tests. Default: now.
    pub imported_at: Option<String>,
    /// How long to wait for another import to finish. Default two minutes.
    pub lock_timeout: Duration,
    /// Lets the caller stop an import between files.
    pub cancel: Option<CancelToken>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            input: None,
            require_complete: false,
            imported_at: None,
            lock_timeout: DEFAULT_TIMEOUT,
            cancel: None,
        }
    }
}

/// Why an import or inspection failed. The messages are written for the person using the app.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// The chosen file or folder is not there.
    #[error("{} was not found. Choose it again.", .0.display())]
    SourceMissing(PathBuf),
    /// Not a look file, a folder or a zip.
    #[error(
        "{0} is not a look file (.cube, .alf4, .alf4c or .aml), a folder or a zip. Choose one of those."
    )]
    UnsupportedSource(String),
    /// The zip could not be opened.
    #[error("This zip cannot be opened. {0}. Download it again.")]
    BadZip(String),
    /// A folder with an absurd number of files.
    #[error("This folder holds far too many files to be a set of looks.")]
    TooManyFiles,
    /// More LUT data than any set of looks holds.
    #[error("The .cube files here add up to more than 4 GB, far more than a set of looks.")]
    TooLarge,
    /// Nothing in the source can be imported; the text says why and what to do.
    #[error("{0}")]
    NothingToImport(String),
    /// Some files need the generic question answered first.
    #[error("Say what these LUTs expect as input (LogC4, LogC3, or LogC3 with Rec.709 baked in): {}.", .files.join(", "))]
    NeedsInput {
        /// The files that need an answer (the first few).
        files: Vec<String>,
    },
    /// `require_complete` was set and a package is missing files.
    #[error("{0}")]
    Incomplete(String),
    /// A file is not what [`inspect`] saw.
    #[error("{0} changed while it was being imported. Import again.")]
    SourceChanged(String),
    /// The library lock could not be taken.
    #[error(transparent)]
    Lock(#[from] LockError),
    /// The manifest could not be read or written.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// A file operation failed.
    #[error("Could not read or write {}: {source}", .path.display())]
    Io {
        /// The file or folder.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// The caller cancelled.
    #[error("The import was cancelled. Nothing was changed.")]
    Cancelled,
}

impl ImportError {
    /// A CLI exit code: 2 for mistakes in what was asked (docs/10), 1 for everything else.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::SourceMissing(_) | Self::UnsupportedSource(_) | Self::NeedsInput { .. } => 2,
            _ => 1,
        }
    }
}

/// A file that was not imported, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RejectedFile {
    /// Path inside the source.
    pub rel_path: String,
    /// File name.
    pub name: String,
    /// A sentence saying why.
    pub reason: String,
}

/// A look or display LUT touched by an import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImportedItem {
    /// Its id in the library.
    pub id: LookId,
    /// Display name.
    pub name: String,
    /// File, relative to the `looks` folder.
    pub file: String,
    /// Whether it is a display LUT rather than a look.
    pub display_lut: bool,
}

/// What an import did.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct ImportReport {
    /// New looks and display LUTs.
    pub added: Vec<ImportedItem>,
    /// Entries whose file was replaced, repaired, or whose input answer changed.
    pub updated: Vec<ImportedItem>,
    /// Entries that were already there with the same content: nothing was written.
    pub unchanged: Vec<ImportedItem>,
    /// Files that could not be used.
    pub rejected: Vec<RejectedFile>,
    /// The person's own looks that an ARRI package has claimed: (the id they had, the ARRI id
    /// they have now). The old id is gone from the library, so whatever still names it (the
    /// default look, a script's `--look`) has to follow. Display LUTs are not listed: a claimed
    /// own LUT that becomes a display LUT is not a look any more.
    pub renamed: Vec<(LookId, LookId)>,
    /// Things worth telling the user, for example a damaged looks list that was set aside.
    pub notes: Vec<String>,
}

impl ImportReport {
    /// New and updated entries.
    #[must_use]
    pub fn imported_count(&self) -> usize {
        self.added.len() + self.updated.len()
    }

    /// Every entry the source maps to, however it was handled.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.added.len() + self.updated.len() + self.unchanged.len()
    }
}

fn check_cancel(cancel: Option<&CancelToken>) -> Result<(), ImportError> {
    if cancel.is_some_and(CancelToken::is_cancelled) {
        Err(ImportError::Cancelled)
    } else {
        Ok(())
    }
}

/// One file on its way to a worker: what was read (or why it could not be).
struct Job {
    index: usize,
    rel_path: String,
    size: u64,
    /// What the file name says it is.
    format: LookFormat,
    bytes: Result<Vec<u8>, ReadFailure>,
}

/// What a `.cube` file is: its summary, or why it cannot be used.
fn analyze_cube(bytes: &[u8]) -> Result<CubeSummary, String> {
    match cube::parse_bytes(bytes) {
        Ok(file) => match file.lut3d.as_ref() {
            Some(cube3d) => Ok(CubeSummary {
                title: file.title.clone(),
                lut_size: cube3d.size(),
                has_shaper: file.lut1d.is_some(),
            }),
            None => Err(cube_error_text(&CubeError::NoThreeDimensionalLut)),
        },
        Err(e) => Err(cube_error_text(&e)),
    }
}

/// Hashes and validates one file as a 3D `.cube` or an ARRI Look File. Pure; runs on a worker
/// thread.
fn analyze(job: Job) -> (usize, FileFacts) {
    let Job {
        index,
        rel_path,
        size,
        format,
        bytes,
    } = job;
    let facts = match bytes {
        Err(failure) => FileFacts {
            rel_path,
            size,
            sha256: String::new(),
            format,
            check: Err(failure.sentence()),
            look_file: None,
        },
        Ok(bytes) => {
            let sha256 = sha256_hex(&bytes);
            let size = bytes.len() as u64;
            if format.is_cube() {
                FileFacts {
                    rel_path,
                    size,
                    sha256,
                    format,
                    check: analyze_cube(&bytes),
                    look_file: None,
                }
            } else {
                // The container is what the bytes say, whatever the name does.
                match ym_color::alf::parse(&bytes) {
                    Ok(look) => FileFacts {
                        rel_path,
                        size,
                        sha256,
                        format: LookFormat::from_kind(look.kind),
                        check: Ok(CubeSummary {
                            title: None,
                            lut_size: look.lut.size(),
                            has_shaper: false,
                        }),
                        look_file: Some(LookFileFacts {
                            input: look.input,
                            display_transforms_ignored: look.display_transforms_ignored,
                        }),
                    },
                    Err(e) => FileFacts {
                        rel_path,
                        size,
                        sha256,
                        format,
                        check: Err(e.to_string()),
                        look_file: None,
                    },
                }
            }
        }
    };
    (index, facts)
}

/// Worker threads for hashing and parsing: one per core, up to eight, never more than files.
fn worker_count(files: usize) -> usize {
    std::thread::available_parallelism()
        .map_or(2, std::num::NonZero::get)
        .clamp(1, 8)
        .min(files.max(1))
}

/// Wizard step 3: reads `path` (a `.cube` file, a folder or a zip) without writing anything,
/// validates every `.cube` file and recognises the packages.
///
/// The source is read on the calling thread, one file at a time; hashing and parsing, which
/// dominate for 65-point cubes, run on a small pool of worker threads. `progress` is called only
/// on the calling thread.
///
/// # Errors
///
/// [`ImportError::SourceMissing`], [`ImportError::UnsupportedSource`], [`ImportError::BadZip`],
/// [`ImportError::TooLarge`], [`ImportError::Io`] or [`ImportError::Cancelled`]. A source that
/// holds nothing usable is not an error here: the returned [`Recognition`] explains it.
pub fn inspect(
    path: &Path,
    db: &PackageDb,
    cancel: Option<&CancelToken>,
    progress: ProgressFn<'_>,
) -> Result<Recognition, ImportError> {
    let mut source = Source::open(path)?;
    let total = source.candidates.len();
    let workers = worker_count(total);
    let facts = thread::scope(|scope| -> Result<Vec<FileFacts>, ImportError> {
        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(workers);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (done_tx, done_rx) = mpsc::channel::<(usize, FileFacts)>();
        for _ in 0..workers {
            let (jobs, done) = (Arc::clone(&job_rx), done_tx.clone());
            scope.spawn(move || {
                loop {
                    let next = jobs.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    let Ok(job) = next else { break };
                    if done.send(analyze(job)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(done_tx);

        let mut slots: Vec<Option<FileFacts>> = (0..total).map(|_| None).collect();
        let mut finished = 0_usize;
        for index in 0..total {
            check_cancel(cancel)?;
            let (rel_path, size, format) = {
                let c = &source.candidates[index];
                (c.rel_path.clone(), c.size, c.format)
            };
            progress(Progress {
                stage: Stage::Scan,
                done: finished,
                total,
                file: Some(&rel_path),
            });
            let bytes = source.read(index);
            if job_tx
                .send(Job {
                    index,
                    rel_path,
                    size,
                    format,
                    bytes,
                })
                .is_err()
            {
                break;
            }
            while let Ok(got) = done_rx.try_recv() {
                slots[got.0] = Some(got.1);
                finished += 1;
            }
        }
        drop(job_tx);
        for got in done_rx {
            slots[got.0] = Some(got.1);
            finished += 1;
            progress(Progress {
                stage: Stage::Scan,
                done: finished,
                total,
                file: None,
            });
        }
        check_cancel(cancel)?;
        Ok(slots
            .into_iter()
            .enumerate()
            .map(|(i, slot)| {
                slot.unwrap_or_else(|| FileFacts {
                    rel_path: source.candidates[i].rel_path.clone(),
                    size: source.candidates[i].size,
                    sha256: String::new(),
                    format: source.candidates[i].format,
                    check: Err("It could not be checked.".to_string()),
                    look_file: None,
                })
            })
            .collect())
    })?;
    progress(Progress {
        stage: Stage::Scan,
        done: total,
        total,
        file: None,
    });
    Ok(recognize(db, &source.info, &facts, &source.ignored))
}

fn now_rfc3339() -> String {
    let now = OffsetDateTime::now_utc();
    let now = now.replace_nanosecond(0).unwrap_or(now);
    now.format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

/// A temporary file written by phase 1 and not yet renamed into place.
struct Staged {
    temp: PathBuf,
    dest: PathBuf,
    existed_before: bool,
}

/// Whether the file is there and is the one that was imported (its SHA-256 is `sha256`).
fn intact(path: &Path, sha256: &str) -> bool {
    hash_file(path).ok().as_deref() == Some(sha256)
}

/// Most renamed looks a note names one by one.
const MAX_RENAMED_NOTED: usize = 5;

fn discard(staged: &[Staged]) {
    for s in staged {
        let _ = std::fs::remove_file(&s.temp);
    }
}

/// Wizard step 4: copies the files [`inspect`] recognized (and, with `options.input`, the ones it
/// asked about) into the library and records them in the manifest. `library` is refreshed to
/// match.
///
/// # Errors
///
/// [`ImportError::NothingToImport`] when the source holds nothing usable,
/// [`ImportError::NeedsInput`] when files need an answer that `options.input` does not give,
/// [`ImportError::Incomplete`] (with `require_complete`), [`ImportError::SourceChanged`],
/// [`ImportError::Cancelled`], and lock, manifest or file errors. On any error no new look is
/// added and the list of looks is unchanged (see the module documentation for the one repairable
/// exception).
pub fn import(
    library: &mut Library,
    db: &PackageDb,
    path: &Path,
    recognition: &Recognition,
    options: &ImportOptions,
    progress: ProgressFn<'_>,
) -> Result<ImportReport, ImportError> {
    // What can be done at all?
    if recognition.verdict() == Verdict::Blocked {
        let message = recognition
            .problems()
            .next()
            .or_else(|| recognition.issues.first())
            .map_or_else(
                || "There is nothing to import.".to_string(),
                |i| i.message.clone(),
            );
        return Err(ImportError::NothingToImport(message));
    }
    if options.require_complete && recognition.verdict() == Verdict::Incomplete {
        let message = recognition
            .issues
            .iter()
            .find(|i| i.kind == crate::recognize::IssueKind::Incomplete)
            .map_or_else(
                || "A package is incomplete.".to_string(),
                |i| i.message.clone(),
            );
        return Err(ImportError::Incomplete(message));
    }
    if recognition.needs_answer() && options.input.is_none() {
        return Err(ImportError::NeedsInput {
            files: recognition
                .files
                .iter()
                .filter(|f| f.needs_answer())
                .take(8)
                .map(|f| f.name.clone())
                .collect(),
        });
    }

    let mut report = ImportReport::default();
    let mut items: Vec<(usize, Spec)> = Vec::new();
    for (i, file) in recognition.files.iter().enumerate() {
        if let Outcome::Rejected { reason } = &file.outcome {
            report.rejected.push(RejectedFile {
                rel_path: file.rel_path.clone(),
                name: file.name.clone(),
                reason: reason.clone(),
            });
        } else if let Some(spec) =
            Spec::from_outcome(&file.outcome, &file.rel_path, options.input, file.format)
        {
            items.push((i, spec));
        }
    }

    // ARRI's `.cube` files are planned before its look files and before the person's own files, so
    // that a look file meets the `.cube` of the same look (the `.cube` wins) and an ARRI file can
    // claim an own LUT of an earlier import but never one of this import. The order within each
    // kind is the order of the list.
    items.sort_by_key(|(_, spec)| match (spec.arri, spec.format.is_cube()) {
        (true, true) => 0_u8,
        (true, false) => 1,
        (false, _) => 2,
    });

    // Locate every item in the source again; it must be what `inspect` saw. A zip may hold one
    // name twice (with other bytes), and a source lists those in the same order both times, so
    // the n-th file of the recognition with a name is the n-th candidate with that name.
    let mut source = Source::open(path)?;
    let mut candidates_named: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, c) in source.candidates.iter().enumerate() {
        candidates_named.entry(c.rel_path.as_str()).or_default().push(i);
    }
    let mut seen_named: HashMap<&str, usize> = HashMap::new();
    let mut occurrence: Vec<usize> = Vec::with_capacity(recognition.files.len());
    for file in &recognition.files {
        let n = seen_named.entry(file.rel_path.as_str()).or_default();
        occurrence.push(*n);
        *n += 1;
    }
    let located: Vec<usize> = items
        .iter()
        .map(|(i, _)| {
            let rel = &recognition.files[*i].rel_path;
            candidates_named
                .get(rel.as_str())
                .and_then(|named| named.get(occurrence[*i]))
                .copied()
                .ok_or_else(|| ImportError::SourceChanged(rel.clone()))
        })
        .collect::<Result<_, _>>()?;
    drop(candidates_named);

    let looks_dir = library.looks_dir();
    let _lock = LibraryLock::acquire(&looks_dir, options.lock_timeout)?;
    let mut manifest = match Manifest::load(&looks_dir) {
        Ok(m) => m,
        Err(ManifestError::Damaged { path: damaged, .. }) => {
            let moved =
                library
                    .damaged_manifest_quarantined()
                    .map_err(|source| ImportError::Io {
                        path: damaged.clone(),
                        source,
                    })?;
            report.notes.push(format!(
                "The list of imported looks was damaged. It was kept as {} and a new list was started.",
                moved.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ));
            Manifest::default()
        }
        Err(e) => return Err(e.into()),
    };

    let imported_at = options.imported_at.clone().unwrap_or_else(now_rfc3339);
    let paths = library.paths().clone();
    let mut staged: Vec<Staged> = Vec::new();
    let mut staged_dests: HashSet<PathBuf> = HashSet::new();
    // Files that another file of the same look has replaced (a look file by its `.cube`, a `.cube`
    // that is gone or altered by its look file, an own LUT by the ARRI package that claims it);
    // deleted once the list is saved.
    let mut replaced_files: Vec<String> = Vec::new();
    // Looks whose look file a `.cube` has replaced, and looks whose damaged `.cube` a look file
    // has replaced: the two notes that say so.
    let mut upgraded = 0_usize;
    let mut repaired = 0_usize;
    // Looks left as the `.cube` that is already there instead of a look file: from an earlier
    // import, and from this one (a folder that holds both libraries).
    let mut kept_cube = 0_usize;
    let mut kept_cube_here = 0_usize;
    // Files that are copies of LUTs already in the library under other names.
    let mut same_bytes = 0_usize;
    // Entries of the person's own that an ARRI package has claimed.
    let mut claimed = 0_usize;
    // Ids this import has already settled, so a later file never replaces an earlier one.
    let mut touched: HashSet<LookId> = HashSet::new();
    let total = items.len();

    // Phase 1: read, verify, decide, stage.
    let phase1 = (|| -> Result<(), ImportError> {
        for (n, ((file_index, spec), source_index)) in items.iter().zip(&located).enumerate() {
            check_cancel(options.cancel.as_ref())?;
            let recognized = &recognition.files[*file_index];
            progress(Progress {
                stage: Stage::Copy,
                done: n,
                total,
                file: Some(&recognized.rel_path),
            });
            let bytes = source
                .read(*source_index)
                .map_err(|_| ImportError::SourceChanged(recognized.rel_path.clone()))?;
            let sha256 = sha256_hex(&bytes);
            if sha256 != recognized.sha256 {
                return Err(ImportError::SourceChanged(recognized.rel_path.clone()));
            }
            let mut slot = assign(&manifest, spec, &sha256, &touched);
            // The look is imported as a `.cube` from an earlier import, but that file is gone or
            // altered: it is no reason to keep the entry as it is, so the look file takes its
            // place. (A `.cube` of this very import is staged and not on disk yet, and is fine.)
            if let Slot::Superseded {
                id,
                file,
                same_import: false,
                own_file,
            } = &slot
                && !manifest
                    .look(id)
                    .is_some_and(|entry| intact(&paths.file(&entry.file), &entry.sha256))
            {
                slot = Slot::Repair {
                    id: id.clone(),
                    file: own_file.clone(),
                    old_file: file.clone(),
                };
            }
            let (id, file, is_new) = match &slot {
                Slot::Add { id, file } | Slot::Claim { id, file, .. } => {
                    (id.clone(), file.clone(), true)
                }
                Slot::Replace { id, file }
                | Slot::Existing { id, file }
                | Slot::Superseded { id, file, .. }
                | Slot::Same { id, file, .. }
                | Slot::Upgrade { id, file, .. }
                | Slot::Repair { id, file, .. } => (id.clone(), file.clone(), false),
            };
            // The look is already imported as a `.cube`: that one stays, nothing is written.
            if let Slot::Superseded { same_import, .. } = &slot {
                touched.insert(id.clone());
                if *same_import {
                    kept_cube_here += 1;
                } else {
                    kept_cube += 1;
                }
                report.unchanged.push(ImportedItem {
                    id,
                    name: spec.name.clone(),
                    file,
                    display_lut: false,
                });
                continue;
            }
            // These bytes are in the library already, under another name or group: the entry
            // stays as it is (a missing or altered file is written again, like for any entry).
            if let Slot::Same {
                name, display_lut, ..
            } = &slot
            {
                let dest = paths.file(&file);
                // A file staged by this import is not on disk yet and is not damaged either.
                let repair = !staged_dests.contains(&dest)
                    && hash_file(&dest).ok().as_deref() != Some(sha256.as_str());
                if repair && staged_dests.insert(dest.clone()) {
                    let existed_before = dest.exists();
                    if let Some(parent) = dest.parent() {
                        std::fs::create_dir_all(parent).map_err(|source| ImportError::Io {
                            path: parent.to_path_buf(),
                            source,
                        })?;
                    }
                    let temp = stage_write(&dest, &bytes).map_err(|source| ImportError::Io {
                        path: dest.clone(),
                        source,
                    })?;
                    staged.push(Staged {
                        temp,
                        dest,
                        existed_before,
                    });
                }
                same_bytes += 1;
                let item = ImportedItem {
                    id,
                    name: name.clone(),
                    file,
                    display_lut: *display_lut,
                };
                if repair {
                    report.updated.push(item);
                } else {
                    report.unchanged.push(item);
                }
                continue;
            }
            touched.insert(id.clone());
            let mut replaces = None;
            match &slot {
                Slot::Upgrade { old_file, .. } => {
                    replaced_files.push(old_file.clone());
                    upgraded += 1;
                }
                Slot::Repair { old_file, .. } => {
                    replaced_files.push(old_file.clone());
                    repaired += 1;
                }
                Slot::Claim { old, .. } => {
                    replaced_files.push(old.file.clone());
                    replaces = Some(old.id.clone());
                    claimed += 1;
                    if matches!(spec.kind, SpecKind::Look { .. }) {
                        report.renamed.push((old.id.clone(), id.clone()));
                    }
                }
                _ => {}
            }
            let dest = paths.file(&file);
            let write = match &slot {
                Slot::Add { .. }
                | Slot::Replace { .. }
                | Slot::Upgrade { .. }
                | Slot::Repair { .. }
                | Slot::Claim { .. } => true,
                // Already imported: write again only if the library's copy is gone or altered.
                Slot::Existing { .. } => hash_file(&dest).ok().as_deref() != Some(sha256.as_str()),
                Slot::Superseded { .. } | Slot::Same { .. } => false,
            };
            if write && staged_dests.insert(dest.clone()) {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|source| ImportError::Io {
                        path: parent.to_path_buf(),
                        source,
                    })?;
                }
                let existed_before = dest.exists();
                let temp = stage_write(&dest, &bytes).map_err(|source| ImportError::Io {
                    path: dest.clone(),
                    source,
                })?;
                staged.push(Staged {
                    temp,
                    dest: dest.clone(),
                    existed_before,
                });
            }
            let changed = apply(
                &mut manifest,
                &EntryData {
                    spec,
                    id: id.clone(),
                    file: file.clone(),
                    sha256: sha256.clone(),
                    lut_size: recognized.lut_size.unwrap_or(0),
                    imported: &imported_at,
                    url: Some(db.page_url()),
                    replaces: replaces.as_ref(),
                },
            );
            let item = ImportedItem {
                id,
                name: spec.name.clone(),
                file,
                display_lut: matches!(spec.kind, SpecKind::Display { .. }),
            };
            if matches!(slot, Slot::Claim { .. }) {
                // Not a new look: the person's entry became ARRI's.
                report.updated.push(item);
            } else if is_new {
                report.added.push(item);
            } else if write
                || changed
                || matches!(
                    slot,
                    Slot::Replace { .. } | Slot::Upgrade { .. } | Slot::Repair { .. }
                )
            {
                report.updated.push(item);
            } else {
                report.unchanged.push(item);
            }
        }
        Ok(())
    })();
    if let Err(e) = phase1 {
        discard(&staged);
        return Err(e);
    }

    // Phase 2: move the staged files into place.
    let mut committed: Vec<&Staged> = Vec::new();
    for s in &staged {
        if let Err(source) = commit_staged(&s.temp, &s.dest) {
            // Roll back files that did not exist before, and drop the rest of the temporaries.
            for done in &committed {
                if !done.existed_before {
                    let _ = std::fs::remove_file(&done.dest);
                }
            }
            discard(&staged);
            return Err(ImportError::Io {
                path: s.dest.clone(),
                source,
            });
        }
        committed.push(s);
    }
    let dirs: HashSet<&Path> = staged.iter().filter_map(|s| s.dest.parent()).collect();
    for dir in dirs {
        sync_dir(dir);
    }

    // Phase 3: the manifest, atomically.
    manifest.sort();
    // Nothing new or changed (a repeat import): leave the manifest file alone.
    let changed_anything = !(report.added.is_empty() && report.updated.is_empty());
    if changed_anything && let Err(e) = manifest.save(&looks_dir) {
        for done in &committed {
            if !done.existed_before {
                let _ = std::fs::remove_file(&done.dest);
            }
        }
        return Err(e.into());
    }
    // The look files that a `.cube` has replaced, and the own LUTs that ARRI's package has
    // claimed (their ARRI copy is written), are not in the list any more.
    for old in &replaced_files {
        if !manifest
            .looks
            .iter()
            .map(|l| l.file.as_str())
            .chain(manifest.display_luts.iter().map(|d| d.file.as_str()))
            .any(|f| f.eq_ignore_ascii_case(old))
        {
            let path = paths.file(old);
            if std::fs::remove_file(&path).is_ok()
                && let Some(parent) = path.parent()
            {
                sync_dir(parent);
            }
        }
    }
    if kept_cube > 0 {
        report.notes.push(format!(
            "{} already imported from the 3D-LUT version, which is kept: it has more points than the look file.",
            if kept_cube == 1 {
                "1 look was".to_string()
            } else {
                format!("{kept_cube} looks were")
            }
        ));
    }
    if kept_cube_here > 0 {
        report.notes.push(if kept_cube_here == 1 {
            "1 look came both as a 3D LUT and as a look file. The 3D LUT is kept: it has more points."
                .to_string()
        } else {
            format!(
                "{kept_cube_here} looks came both as 3D LUTs and as look files. The 3D LUTs are kept: they have more points."
            )
        });
    }
    if claimed > 0 {
        report.notes.push(if claimed == 1 {
            "1 of your own LUTs is a file from ARRI's library, so it moved to the ARRI looks. It is not listed twice."
                .to_string()
        } else {
            format!(
                "{claimed} of your own LUTs are files from ARRI's library, so they moved to the ARRI looks. They are not listed twice."
            )
        });
    }
    if !report.renamed.is_empty() {
        // The ids are what scripts name (`--look user:warm`), so they are said, a few of them.
        let shown: Vec<String> = report
            .renamed
            .iter()
            .take(MAX_RENAMED_NOTED)
            .map(|(old, new)| format!("{old} is now {new}"))
            .collect();
        let more = report.renamed.len() - shown.len();
        report.notes.push(format!(
            "{}{}. A script or a setting that names the old id needs the new one.",
            shown.join(", "),
            if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            }
        ));
    }
    if same_bytes > 0 {
        report.notes.push(if same_bytes == 1 {
            "1 file is the same as a LUT that is already imported under another name, so it was not added again."
                .to_string()
        } else {
            format!(
                "{same_bytes} files are the same as LUTs that are already imported under other names, so they were not added again."
            )
        });
    }
    if upgraded > 0 {
        report.notes.push(format!(
            "{} replaced by the 3D-LUT version, which has more points than the look file.",
            if upgraded == 1 {
                "1 look was".to_string()
            } else {
                format!("{upgraded} looks were")
            }
        ));
    }
    if repaired > 0 {
        report.notes.push(if repaired == 1 {
            "The 3D LUT of 1 look was missing or had changed, so that look is now the look file. Import the 3D-LUT version again to get the 3D LUT back."
                .to_string()
        } else {
            format!(
                "The 3D LUTs of {repaired} looks were missing or had changed, so those looks are now the look files. Import the 3D-LUT version again to get the 3D LUTs back."
            )
        });
    }
    progress(Progress {
        stage: Stage::Copy,
        done: total,
        total,
        file: None,
    });
    library.install_manifest(manifest);
    Ok(report)
}

/// [`inspect`] then [`import`]: what `ym looks import` does.
///
/// # Errors
///
/// As [`inspect`] and [`import`].
pub fn import_path(
    library: &mut Library,
    db: &PackageDb,
    path: &Path,
    options: &ImportOptions,
    progress: ProgressFn<'_>,
) -> Result<(Recognition, ImportReport), ImportError> {
    let recognition = inspect(path, db, options.cancel.as_ref(), &mut *progress)?;
    let report = import(library, db, path, &recognition, options, progress)?;
    Ok((recognition, report))
}
