//! The look library: built-in and imported looks, lookup by id, slug or name, default look, lazy LUT loading (`docs/08-looks.md` sections Look model and Library folder).
//!
//! A [`Library`] is a view of one app folder (`~/Library/Application Support/app.yellowmoon` on a
//! Mac, see [`default_root`]):
//!
//! ```text
//! <root>/looks/
//!   manifest.json                    imported looks and display LUTs, with SHA-256 per file
//!   .lock                            held while importing or removing
//!   arri/<group_no>-<group_slug>/    imported ARRI look files (copied, never modified)
//!   arri-display/                    imported ARRI display LUTs
//!   user/                            other imported .cube files
//! ```
//!
//! Built-in looks come from a data folder (`data/looks/builtin/*.json`, parametric look files as in `docs/08-looks.md`); Neutral
//! exists even without one. Imported looks come from `manifest.json`. A look whose file is gone or
//! whose checksum changed stays in the list, disabled, with [`DISABLED_MESSAGE`].
//!
//! LUT files are read lazily: [`Library::transform`] loads a look's `.cube`, checks its SHA-256
//! against the manifest, parses it with [`ym_color::cube`] and keeps the result in memory.
//!
//! Reading needs no lock. Anything that changes the library (import, [`Library::remove`]) takes
//! `looks/.lock` and re-reads the manifest after it holds the lock, so two processes never lose
//! each other's updates (see [`crate::lock`]).

mod builtin;
mod ids;
mod model;
mod resolve;
mod verify;

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::Lut;

pub use builtin::BuiltinProblem;
pub use ids::{
    IdError, IdKind, LookId, MAX_FILE_STEM_BYTES, MAX_FILE_STEM_CHARS, MAX_SLUG_CHARS,
    is_valid_slug, numbered_slug, sanitize_file_stem, slugify, truncate_to_bytes,
};
pub use model::{
    DISABLED_MESSAGE, DisplayInput, DisplayLut, LoadedLook, Look, LookFormat, LookGroup,
    LookLibrary, LookStatus, LutLoadError, LutRef, cube_error_text,
};
pub use resolve::{Candidate, EXIT_LOOK_MISSING, EXIT_USAGE, ResolveError, resolve};
pub use verify::{VerifyEntry, VerifyOutcome, VerifyReport};

use crate::import::{Progress, Stage};
use crate::lock::{DEFAULT_TIMEOUT, LibraryLock, LockError};
use crate::manifest::{Manifest, ManifestError, ManifestGroup, hash_file, quarantine, sync_dir};

/// Environment variable that replaces the app folder (tests, development, portable use).
pub const HOME_ENV: &str = "YELLOWMOON_HOME";
/// Name of the app folder inside the platform's data folder.
pub const APP_FOLDER: &str = "app.yellowmoon";
/// Name of the folder inside the app folder that holds the library.
pub const LOOKS_FOLDER: &str = "looks";

/// Which platform's folder conventions [`root_from`] follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// `~/Library/Application Support/app.yellowmoon`.
    MacOs,
    /// `$XDG_DATA_HOME/app.yellowmoon`, else `~/.local/share/app.yellowmoon`.
    Unix,
    /// `%APPDATA%\app.yellowmoon`.
    Windows,
}

impl Platform {
    /// The platform this build runs on.
    #[must_use]
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOs
        } else if cfg!(windows) {
            Self::Windows
        } else {
            Self::Unix
        }
    }
}

/// The app folder, from an environment lookup (so tests need not touch the real environment).
///
/// `YELLOWMOON_HOME`, when set and not empty, is used as the app folder itself. Otherwise the
/// platform's data folder plus `app.yellowmoon`. A relative `XDG_DATA_HOME` is ignored, as the
/// XDG specification says.
///
/// # Errors
///
/// [`LibraryError::NoHome`] when neither `YELLOWMOON_HOME` nor a home folder is available.
pub fn root_from(
    get: impl Fn(&str) -> Option<OsString>,
    platform: Platform,
) -> Result<PathBuf, LibraryError> {
    let non_empty = |key: &str| get(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(home) = non_empty(HOME_ENV) {
        return Ok(home);
    }
    let from_home = || non_empty("HOME").ok_or(LibraryError::NoHome);
    let root = match platform {
        Platform::MacOs => from_home()?
            .join("Library")
            .join("Application Support")
            .join(APP_FOLDER),
        Platform::Unix => match non_empty("XDG_DATA_HOME").filter(|p| p.is_absolute()) {
            Some(xdg) => xdg.join(APP_FOLDER),
            None => from_home()?.join(".local").join("share").join(APP_FOLDER),
        },
        Platform::Windows => non_empty("APPDATA")
            .ok_or(LibraryError::NoHome)?
            .join(APP_FOLDER),
    };
    Ok(root)
}

/// The app folder for this machine: `~/Library/Application Support/app.yellowmoon` on a Mac,
/// `$XDG_DATA_HOME/app.yellowmoon` (or `~/.local/share/app.yellowmoon`) on Linux for development,
/// overridden by `YELLOWMOON_HOME`.
///
/// # Errors
///
/// [`LibraryError::NoHome`] when no home folder can be found.
pub fn default_root() -> Result<PathBuf, LibraryError> {
    root_from(|key| std::env::var_os(key), Platform::current())
}

/// Why a library operation failed.
#[derive(Debug, thiserror::Error)]
pub enum LibraryError {
    /// No `YELLOWMOON_HOME` and no home folder.
    #[error(
        "Yellowmoon cannot find your home folder, so it cannot find its looks. Set YELLOWMOON_HOME to a folder to use instead."
    )]
    NoHome,
    /// The manifest could not be used.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// The library lock could not be taken.
    #[error(transparent)]
    Lock(#[from] LockError),
    /// The id names no imported look or display LUT.
    #[error("There is no imported look with the id {0}.")]
    UnknownLook(LookId),
    /// Built-in looks cannot be removed.
    #[error("{0} is built in and cannot be removed.")]
    BuiltIn(LookId),
    /// A file operation failed.
    #[error("Could not change {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: io::Error,
    },
}

/// Why a look could not be loaded for use.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// No look has this id.
    #[error("There is no look with the id {0}.")]
    NotFound(LookId),
    /// The look's file is missing or changed.
    #[error("{id}: {DISABLED_MESSAGE}")]
    Unavailable {
        /// The look.
        id: LookId,
        /// Missing or changed.
        status: LookStatus,
    },
    /// The file no longer parses as a 3D `.cube` LUT or an ARRI Look File.
    #[error("{id}: the LUT file is not usable: {reason}")]
    Invalid {
        /// The look.
        id: LookId,
        /// Why.
        reason: String,
    },
    /// The file could not be read.
    #[error("{id}: the LUT file could not be read: {source}")]
    Unreadable {
        /// The look.
        id: LookId,
        /// What the system said.
        #[source]
        source: io::Error,
    },
    /// A built-in look could not be built.
    #[error("{id}: {reason}")]
    Builtin {
        /// The look.
        id: LookId,
        /// Why.
        reason: String,
    },
}

impl LoadError {
    /// The CLI exit code: a look that cannot be used counts as missing (3).
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        EXIT_LOOK_MISSING
    }

    fn from_lut(id: &LookId, error: LutLoadError) -> Self {
        match error {
            LutLoadError::Missing => Self::Unavailable {
                id: id.clone(),
                status: LookStatus::Missing,
            },
            LutLoadError::Changed { .. } => Self::Unavailable {
                id: id.clone(),
                status: LookStatus::Changed,
            },
            LutLoadError::Unreadable(source) => Self::Unreadable {
                id: id.clone(),
                source,
            },
            LutLoadError::Invalid(reason) | LutLoadError::InvalidLookFile(reason) => {
                Self::Invalid {
                    id: id.clone(),
                    reason,
                }
            }
        }
    }
}

/// Something wrong with the library that did not stop it opening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryProblem {
    /// A built-in look file was skipped.
    Builtin(BuiltinProblem),
    /// `manifest.json` is damaged or from a newer version: no imported looks are listed.
    Manifest(String),
}

/// What [`Library::remove`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removed {
    /// The look that was removed.
    pub id: LookId,
    /// Its file, relative to the `looks` folder.
    pub file: String,
    /// Whether the file was deleted (it may already have been gone).
    pub file_deleted: bool,
}

/// The paths of one library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryPaths {
    root: PathBuf,
}

impl LibraryPaths {
    /// Paths under an app folder.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The app folder.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/looks`.
    #[must_use]
    pub fn looks_dir(&self) -> PathBuf {
        self.root.join(LOOKS_FOLDER)
    }

    /// `<root>/looks/manifest.json`.
    #[must_use]
    pub fn manifest_path(&self) -> PathBuf {
        Manifest::path(&self.looks_dir())
    }

    /// The absolute path of a manifest `file` entry.
    #[must_use]
    pub fn file(&self, relative: &str) -> PathBuf {
        let mut path = self.looks_dir();
        path.extend(relative.split('/'));
        path
    }
}

#[derive(Default)]
struct Cache {
    looks: HashMap<LookId, (String, Arc<LookTransform>)>,
    displays: HashMap<LookId, (String, Arc<Lut>)>,
}

/// The look library (see the module documentation).
pub struct Library {
    paths: LibraryPaths,
    builtin_dir: Option<PathBuf>,
    builtin: builtin::BuiltinSet,
    manifest: Manifest,
    manifest_problem: Option<String>,
    looks: Vec<Look>,
    display_luts: Vec<DisplayLut>,
    cache: Mutex<Cache>,
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Library")
            .field("root", &self.paths.root)
            .field("looks", &self.looks.len())
            .field("display_luts", &self.display_luts.len())
            .finish()
    }
}

impl Library {
    /// Opens the library under `root` (the app folder; see [`default_root`]). `builtin_dir` is
    /// `data/looks/builtin`, or `None` to have Neutral only.
    ///
    /// Opening never fails because of a bad manifest: a damaged or too-new `manifest.json` leaves
    /// the built-in looks working and is reported by [`Library::problems`]. Nothing is created on
    /// disk.
    ///
    /// # Errors
    ///
    /// Currently never; the `Result` keeps room for failures that should stop the app.
    pub fn open(
        root: impl Into<PathBuf>,
        builtin_dir: Option<&Path>,
    ) -> Result<Self, LibraryError> {
        let mut library = Self {
            paths: LibraryPaths::new(root),
            builtin_dir: builtin_dir.map(Path::to_path_buf),
            builtin: builtin::BuiltinSet::default(),
            manifest: Manifest::default(),
            manifest_problem: None,
            looks: Vec::new(),
            display_luts: Vec::new(),
            cache: Mutex::new(Cache::default()),
        };
        library.reload();
        Ok(library)
    }

    /// Re-reads the built-in files and the manifest and re-checks that every file exists. Picks up
    /// what another process (the CLI, the app) changed, and `signature.json` when it appears.
    pub fn reload(&mut self) {
        self.builtin = builtin::load(self.builtin_dir.as_deref());
        // The built-in files may have been edited: bake them again on next use.
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .looks
            .retain(|id, _| is_imported(id));
        match Manifest::load(&self.paths.looks_dir()) {
            Ok(manifest) => {
                self.manifest = manifest;
                self.manifest_problem = None;
            }
            Err(e) => {
                self.manifest = Manifest::default();
                self.manifest_problem = Some(e.to_string());
            }
        }
        self.rebuild();
    }

    /// Rebuilds the look lists from the builtin set and the manifest, with a stat-only check of
    /// each file, and drops cached LUTs whose checksum is no longer the manifest's.
    fn rebuild(&mut self) {
        let mut looks: Vec<Look> = self
            .builtin
            .items
            .iter()
            .map(|b| Look {
                id: b.id.clone(),
                name: b.name.clone(),
                group: LookGroup::BuiltIn,
                input: LookInput::LogC4,
                cdl: None,
                lut: None,
                source: None,
                status: LookStatus::Ok,
            })
            .collect();

        let mut imported: Vec<Look> = self
            .manifest
            .looks
            .iter()
            .map(|m| {
                let path = self.paths.file(&m.file);
                let status = if path.is_file() {
                    LookStatus::Ok
                } else {
                    LookStatus::Missing
                };
                Look {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    group: match &m.group {
                        ManifestGroup::Arri { no, name } => LookGroup::Arri {
                            group_no: *no,
                            group_name: name.clone(),
                        },
                        ManifestGroup::User => LookGroup::User,
                    },
                    input: m.input,
                    cdl: None,
                    lut: Some(LutRef {
                        path,
                        sha256: m.sha256.clone(),
                        lut_size: m.lut_size,
                        format: m.format,
                    }),
                    source: m.source.clone(),
                    status,
                }
            })
            .collect();
        imported.sort_by_cached_key(|l| {
            let (rank, group_no) = match &l.group {
                LookGroup::BuiltIn => (0, 0),
                LookGroup::Arri { group_no, .. } => (1, *group_no),
                LookGroup::User => (2, 0),
            };
            (rank, group_no, l.name.to_lowercase(), l.id.clone())
        });
        looks.extend(imported);
        self.looks = looks;

        self.display_luts = self
            .manifest
            .display_luts
            .iter()
            .map(|m| {
                let path = self.paths.file(&m.file);
                let status = if path.is_file() {
                    LookStatus::Ok
                } else {
                    LookStatus::Missing
                };
                DisplayLut {
                    id: m.id.clone(),
                    input: m.input,
                    target: m.target.clone(),
                    lut: LutRef {
                        path,
                        sha256: m.sha256.clone(),
                        lut_size: 0,
                        format: LookFormat::Cube,
                    },
                    status,
                }
            })
            .collect();

        // Forget cached LUTs that no longer match the manifest (removed, replaced or re-imported).
        let known: HashMap<&LookId, &str> = self
            .manifest
            .looks
            .iter()
            .map(|m| (&m.id, m.sha256.as_str()))
            .chain(
                self.manifest
                    .display_luts
                    .iter()
                    .map(|m| (&m.id, m.sha256.as_str())),
            )
            .collect();
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .looks
            .retain(|id, (sha, _)| !is_imported(id) || known.get(id) == Some(&sha.as_str()));
        cache
            .displays
            .retain(|id, (sha, _)| known.get(id) == Some(&sha.as_str()));
        drop(cache);
        self.evict_disabled();
    }

    /// Drops cached LUTs of looks and display LUTs that are disabled (file missing or changed), so
    /// that a disabled look is never applied from memory: the next load reads the file again,
    /// checks its checksum and fails with [`LoadError::Unavailable`] if it is still wrong.
    fn evict_disabled(&self) {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let disabled: HashSet<&LookId> = self
            .looks
            .iter()
            .filter(|l| !l.status.is_ok())
            .map(|l| &l.id)
            .chain(
                self.display_luts
                    .iter()
                    .filter(|d| !d.status.is_ok())
                    .map(|d| &d.id),
            )
            .collect();
        cache.looks.retain(|id, _| !disabled.contains(id));
        cache.displays.retain(|id, _| !disabled.contains(id));
    }

    /// The app folder this library lives in.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.paths.root()
    }

    /// The library's paths.
    #[must_use]
    pub fn paths(&self) -> &LibraryPaths {
        &self.paths
    }

    /// The `looks` folder.
    #[must_use]
    pub fn looks_dir(&self) -> PathBuf {
        self.paths.looks_dir()
    }

    /// Every look in display order: built in (Neutral, Signature if present, others), then ARRI
    /// looks by group number and name, then the user's own by name. Disabled looks are included.
    #[must_use]
    pub fn looks(&self) -> &[Look] {
        &self.looks
    }

    /// Imported ARRI display LUTs, in manifest order.
    #[must_use]
    pub fn display_luts(&self) -> &[DisplayLut] {
        &self.display_luts
    }

    /// The look with this id.
    #[must_use]
    pub fn get(&self, id: &LookId) -> Option<&Look> {
        self.looks.iter().find(|l| &l.id == id)
    }

    /// The display LUT with this id.
    #[must_use]
    pub fn display_lut_info(&self, id: &LookId) -> Option<&DisplayLut> {
        self.display_luts.iter().find(|d| &d.id == id)
    }

    /// Finds a look from a `--look` argument: full id, slug after the colon, or display name,
    /// case-insensitive. See [`resolve`].
    ///
    /// # Errors
    ///
    /// [`ResolveError::Ambiguous`] (CLI exit code 2) or [`ResolveError::NotFound`] (exit code 3).
    pub fn resolve(&self, query: &str) -> Result<&Look, ResolveError> {
        resolve(&self.looks, query)
    }

    /// Anything wrong that did not stop the library opening.
    #[must_use]
    pub fn problems(&self) -> Vec<LibraryProblem> {
        let mut out: Vec<LibraryProblem> = self
            .builtin
            .problems
            .iter()
            .cloned()
            .map(LibraryProblem::Builtin)
            .collect();
        if let Some(message) = &self.manifest_problem {
            out.push(LibraryProblem::Manifest(message.clone()));
        }
        out
    }

    /// The look ready to apply, loaded on first use and kept in memory. An imported look's file is
    /// read, checked against its SHA-256 and parsed; a built-in parametric look is baked into a
    /// 65-point cube (`ym_color::lut::BAKE_SIZE`).
    ///
    /// # Errors
    ///
    /// [`LoadError::NotFound`] for an unknown id; [`LoadError::Unavailable`] when the file is
    /// missing or changed (CLI exit code 3); [`LoadError::Invalid`] and [`LoadError::Unreadable`]
    /// for a file that cannot be used.
    pub fn transform(&self, id: &LookId) -> Result<Arc<LookTransform>, LoadError> {
        let look = self
            .get(id)
            .ok_or_else(|| LoadError::NotFound(id.clone()))?;
        let key = look
            .lut
            .as_ref()
            .map(|l| l.sha256.clone())
            .unwrap_or_default();
        if let Some((sha, hit)) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .looks
            .get(id)
            && *sha == key
        {
            return Ok(Arc::clone(hit));
        }
        let transform = match &look.lut {
            Some(lut_ref) => {
                let loaded = lut_ref
                    .load_look()
                    .map_err(|e| LoadError::from_lut(id, e))?;
                // An ARRI Look File says what it is made for; the list must agree, or the look
                // would be applied in the wrong encoding.
                if loaded.input.is_some_and(|input| input != look.input) {
                    return Err(LoadError::Invalid {
                        id: id.clone(),
                        reason: "the look file is not the kind of look it was imported as. Import it again.".to_string(),
                    });
                }
                // A look file's own CDL goes in front of its LUT; a `.cube` has none.
                LookTransform::new(look.input, loaded.cdl.or(look.cdl), Some(loaded.lut)).map_err(
                    |e| LoadError::Invalid {
                        id: id.clone(),
                        reason: e.to_string(),
                    },
                )?
            }
            None => {
                let builtin = self
                    .builtin
                    .items
                    .iter()
                    .find(|b| &b.id == id)
                    .ok_or_else(|| LoadError::NotFound(id.clone()))?;
                builtin.transform().map_err(|reason| LoadError::Builtin {
                    id: id.clone(),
                    reason,
                })?
            }
        };
        let transform = Arc::new(transform);
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .looks
            .insert(id.clone(), (key, Arc::clone(&transform)));
        Ok(transform)
    }

    /// An imported display LUT, loaded on first use (checksum-checked) and kept in memory.
    ///
    /// # Errors
    ///
    /// As [`Library::transform`].
    pub fn display_lut(&self, id: &LookId) -> Result<Arc<Lut>, LoadError> {
        let info = self
            .display_lut_info(id)
            .ok_or_else(|| LoadError::NotFound(id.clone()))?;
        if let Some((sha, hit)) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .displays
            .get(id)
            && *sha == info.lut.sha256
        {
            return Ok(Arc::clone(hit));
        }
        let lut = Arc::new(info.lut.load().map_err(|e| LoadError::from_lut(id, e))?);
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .displays
            .insert(id.clone(), (info.lut.sha256.clone(), Arc::clone(&lut)));
        Ok(lut)
    }

    /// Re-reads the manifest, re-hashes every imported file and updates each look's
    /// [`LookStatus`]: missing and changed looks become disabled.
    ///
    /// # Errors
    ///
    /// [`LibraryError::Manifest`] when the list of imported looks is damaged or was written by a
    /// newer version: there is nothing to check against, and saying "all fine" for a list that
    /// cannot be read would be wrong. File problems are not errors; they are results in the report.
    pub fn verify(
        &mut self,
        progress: &mut dyn FnMut(Progress<'_>),
    ) -> Result<VerifyReport, LibraryError> {
        self.reload();
        if self.manifest_problem.is_some() {
            // `reload` keeps opening from failing; here the typed error is wanted. (If the list
            // was repaired between the two reads, read it again and carry on.)
            Manifest::load(&self.paths.looks_dir())?;
            self.reload();
        }
        let total = self.manifest.looks.len() + self.manifest.display_luts.len();
        let mut report = VerifyReport::default();
        let mut statuses: HashMap<LookId, LookStatus> = HashMap::new();
        let files = self
            .manifest
            .looks
            .iter()
            .map(|m| (&m.id, &m.file, &m.sha256))
            .chain(
                self.manifest
                    .display_luts
                    .iter()
                    .map(|m| (&m.id, &m.file, &m.sha256)),
            );
        for (done, (id, file, expected)) in files.enumerate() {
            progress(Progress {
                stage: Stage::Verify,
                done,
                total,
                file: Some(file),
            });
            let (outcome, status) = match hash_file(&self.paths.file(file)) {
                Ok(actual) if &actual == expected => (VerifyOutcome::Ok, LookStatus::Ok),
                Ok(actual) => (
                    VerifyOutcome::Changed {
                        expected: expected.clone(),
                        actual,
                    },
                    LookStatus::Changed,
                ),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    (VerifyOutcome::Missing, LookStatus::Missing)
                }
                Err(e) => (
                    VerifyOutcome::Unreadable {
                        reason: e.to_string(),
                    },
                    LookStatus::Missing,
                ),
            };
            statuses.insert(id.clone(), status);
            report.entries.push(VerifyEntry {
                id: id.clone(),
                file: file.clone(),
                outcome,
            });
        }
        progress(Progress {
            stage: Stage::Verify,
            done: total,
            total,
            file: None,
        });
        for look in &mut self.looks {
            if let Some(status) = statuses.get(&look.id) {
                look.status = *status;
            }
        }
        for display in &mut self.display_luts {
            if let Some(status) = statuses.get(&display.id) {
                display.status = *status;
            }
        }
        self.evict_disabled();
        Ok(report)
    }

    /// Removes an imported look or display LUT: takes the library lock, drops its manifest entry
    /// (atomically), then deletes its file and its group folder if that is now empty.
    ///
    /// # Errors
    ///
    /// [`LibraryError::BuiltIn`] for a built-in look, [`LibraryError::UnknownLook`] for an id that
    /// is not imported, and lock, manifest or file errors.
    pub fn remove(&mut self, id: &LookId) -> Result<Removed, LibraryError> {
        if matches!(id.kind(), IdKind::BuiltIn) {
            return Err(LibraryError::BuiltIn(id.clone()));
        }
        let looks_dir = self.paths.looks_dir();
        let _lock = LibraryLock::acquire(&looks_dir, DEFAULT_TIMEOUT)?;
        let mut manifest = Manifest::load(&looks_dir)?;
        let file = if let Some(i) = manifest.looks.iter().position(|l| &l.id == id) {
            manifest.looks.remove(i).file
        } else if let Some(i) = manifest.display_luts.iter().position(|l| &l.id == id) {
            manifest.display_luts.remove(i).file
        } else {
            return Err(LibraryError::UnknownLook(id.clone()));
        };
        manifest.save(&looks_dir)?;

        let still_used = manifest
            .looks
            .iter()
            .map(|l| &l.file)
            .chain(manifest.display_luts.iter().map(|d| &d.file))
            .any(|f| f.eq_ignore_ascii_case(&file));
        let path = self.paths.file(&file);
        let mut file_deleted = false;
        if !still_used {
            match fs::remove_file(&path) {
                Ok(()) => file_deleted = true,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(source) => return Err(LibraryError::Io { path, source }),
            }
            if let Some(parent) = path.parent() {
                remove_group_folder_if_empty(parent, &looks_dir);
                sync_dir(parent);
            }
        }
        self.manifest = manifest;
        self.manifest_problem = None;
        self.rebuild();
        Ok(Removed {
            id: id.clone(),
            file,
            file_deleted,
        })
    }

    /// Installs a manifest that was just written by an import and refreshes the lists.
    pub(crate) fn install_manifest(&mut self, manifest: Manifest) {
        self.manifest = manifest;
        self.manifest_problem = None;
        self.rebuild();
    }

    /// Starts over from an empty list after a damaged manifest was moved aside.
    pub(crate) fn damaged_manifest_quarantined(&mut self) -> io::Result<PathBuf> {
        quarantine(&self.paths.looks_dir())
    }
}

fn is_imported(id: &LookId) -> bool {
    matches!(id.kind(), IdKind::Arri { .. } | IdKind::User)
}

/// Removes `dir` (a group folder such as `arri/3000-film`) when it is empty. Only folders below
/// `looks/arri/` go; `looks/arri`, `looks/arri-display` and `looks/user` stay.
fn remove_group_folder_if_empty(dir: &Path, looks_dir: &Path) {
    let Ok(relative) = dir.strip_prefix(looks_dir) else {
        return;
    };
    let depth = relative.components().count();
    if depth >= 2 {
        // `remove_dir` fails on a folder that still has files, which is what we want.
        let _ = fs::remove_dir(dir);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;

    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), OsString::from(v)))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn root_follows_platform_conventions() {
        let mac = root_from(env(&[("HOME", "/Users/me")]), Platform::MacOs).unwrap();
        assert_eq!(
            mac,
            PathBuf::from("/Users/me/Library/Application Support/app.yellowmoon")
        );
        let linux = root_from(env(&[("HOME", "/home/me")]), Platform::Unix).unwrap();
        assert_eq!(linux, PathBuf::from("/home/me/.local/share/app.yellowmoon"));
        let xdg = root_from(
            env(&[("HOME", "/home/me"), ("XDG_DATA_HOME", "/data")]),
            Platform::Unix,
        )
        .unwrap();
        assert_eq!(xdg, PathBuf::from("/data/app.yellowmoon"));
        // A relative XDG_DATA_HOME is ignored.
        let relative = root_from(
            env(&[("HOME", "/home/me"), ("XDG_DATA_HOME", "data")]),
            Platform::Unix,
        )
        .unwrap();
        assert_eq!(
            relative,
            PathBuf::from("/home/me/.local/share/app.yellowmoon")
        );
        let win = root_from(
            env(&[("APPDATA", "C:\\Users\\me\\AppData")]),
            Platform::Windows,
        )
        .unwrap();
        assert!(win.ends_with(APP_FOLDER));
    }

    #[test]
    fn yellowmoon_home_overrides_everything() {
        for platform in [Platform::MacOs, Platform::Unix, Platform::Windows] {
            let root = root_from(
                env(&[(HOME_ENV, "/tmp/ym-test"), ("HOME", "/home/me")]),
                platform,
            )
            .unwrap();
            assert_eq!(root, PathBuf::from("/tmp/ym-test"));
        }
        // An empty override is ignored.
        let root = root_from(env(&[(HOME_ENV, ""), ("HOME", "/home/me")]), Platform::Unix).unwrap();
        assert!(root.ends_with(APP_FOLDER));
    }

    #[test]
    fn no_home_is_an_error() {
        assert!(matches!(
            root_from(env(&[]), Platform::Unix),
            Err(LibraryError::NoHome)
        ));
        assert!(matches!(
            root_from(env(&[]), Platform::MacOs),
            Err(LibraryError::NoHome)
        ));
    }

    #[test]
    fn fresh_library_has_neutral_and_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        let library = Library::open(&root, None).unwrap();
        let ids: Vec<_> = library.looks().iter().map(|l| l.id.to_string()).collect();
        assert_eq!(ids, vec!["builtin:neutral"]);
        assert!(!root.exists(), "opening must not create folders");
        assert!(library.problems().is_empty());
        let neutral = library.resolve("neutral").unwrap();
        assert!(neutral.is_enabled());
        assert!(library.transform(&neutral.id).unwrap().is_identity());
    }

    #[test]
    fn damaged_manifest_does_not_break_built_in_looks() {
        let dir = tempfile::tempdir().unwrap();
        let looks = dir.path().join("looks");
        fs::create_dir_all(&looks).unwrap();
        fs::write(looks.join("manifest.json"), "{ broken").unwrap();
        let library = Library::open(dir.path(), None).unwrap();
        assert_eq!(library.looks().len(), 1);
        assert!(matches!(
            library.problems().as_slice(),
            [LibraryProblem::Manifest(_)]
        ));
    }

    #[test]
    fn unknown_ids_are_not_found() {
        let library = Library::open(tempfile::tempdir().unwrap().path(), None).unwrap();
        let err = library.transform(&LookId::user("nope")).unwrap_err();
        assert!(matches!(err, LoadError::NotFound(_)));
        assert_eq!(err.exit_code(), EXIT_LOOK_MISSING);
    }
}
