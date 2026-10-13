//! The look library as the app and the CLI use it: listing, importing, removing, the default look
//! and the display rendering, with a `looks_changed` notification (`docs/08-looks.md`,
//! `docs/01-architecture.md` IPC commands `looks_list`, `looks_import`, `looks_set_default`,
//! `looks_remove`, `apply_look`).
//!
//! # What lives where
//!
//! The library itself (manifest, files, hashes, import, recognition) is `ym-looks`; this module wraps
//! one [`Library`] behind a lock so that the render graph can read it while the UI changes it, and
//! ties it to the two preferences that belong to looks:
//!
//! * the **default look** (`Preferences::default_look`): the look new photos get. A default that has
//!   been removed or whose file is gone is not used; new photos get Neutral until another is chosen
//!   ([`LookService::effective_default`]).
//! * the **display rendering** (`Preferences::display_rendering`): ACES 2 or an imported ARRI display
//!   LUT ([`LookService::display_choice`]).
//!
//! # Missing looks
//!
//! A look whose file is gone or whose hash changed stays in the list, flagged `missing`, and cannot
//! be applied or made the default: both answer "File missing or changed. Import again."
//! ([`LooksError::Missing`]).
//!
//! # Concurrency
//!
//! Reads (the renderer asking for a look) take the read lock. An import or a removal works on a
//! second [`Library`] over the same folder, under the library's own `looks/.lock`, and only then
//! takes the write lock for the moment it takes to re-read the manifest, so a slow import (or a
//! CLI import holding the file lock) never stalls a viewport frame. The CLI and the app can change
//! the library at the same time; [`LookService::reload`] picks up what the other one did.
//!
//! [`LookService::on_change`] registers a callback that runs after the list of looks changed (an
//! import, a removal, a new default, a reload that found something different): the `looks_changed`
//! event. Callbacks run on the thread that made the change, outside every lock.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use serde::{Deserialize, Serialize};
use ym_color::look::LookInput;
use ym_looks::import::{CancelToken, ImportError, ImportOptions, ImportReport, Progress};
use ym_looks::library::{
    DISABLED_MESSAGE, DisplayInput, DisplayLut, Library, LibraryError, LibraryProblem, LoadError,
    Look, LookGroup, LookId, ResolveError, VerifyReport,
};
use ym_looks::manifest::hash_file;
use ym_looks::recognize::PackageDb;

use crate::pipeline::color::{DisplayChoice, LookProblem, LookProvider, ResolvedLook};
use crate::preferences::{
    DisplayRendering, NEUTRAL_LOOK, Preferences, PreferencesError, PreferencesPatch,
    PreferencesStore,
};
use crate::session::{ApplyScope, PhotoId, Session, SessionError};

/// Why a look command failed. The text is plain words with a next step.
#[derive(Debug, thiserror::Error)]
pub enum LooksError {
    /// The library could not be opened or changed.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// An import failed.
    #[error(transparent)]
    Import(#[from] ImportError),
    /// A look could not be loaded.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// A `--look` argument did not name exactly one look.
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    /// The preferences could not be saved.
    #[error(transparent)]
    Preferences(#[from] PreferencesError),
    /// A session command failed.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// No look has this id.
    #[error("That look isn't in the library. Pick another look.")]
    NotFound(LookId),
    /// The look's file is gone or changed.
    #[error("{}", DISABLED_MESSAGE)]
    Missing(LookId),
    /// The display LUT is not imported.
    #[error("Those display LUTs aren't imported. Import ARRI's display LUTs first.")]
    NoDisplayLut(LookId),
}

impl LooksError {
    /// The CLI exit code: 3 for a look that is missing, 2 for a usage mistake, 1 otherwise.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotFound(_) | Self::Missing(_) | Self::Load(_) => 3,
            Self::Resolve(e) => e.exit_code(),
            Self::Import(e) => e.exit_code(),
            Self::NoDisplayLut(_) => 2,
            _ => 1,
        }
    }
}

/// Where a look is listed (`LookGroup` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LookGroupInfo {
    /// Shipped with the app.
    BuiltIn,
    /// One of ARRI's groups, with its number and name.
    Arri {
        /// The four-digit group number.
        no: u32,
        /// The group's name ("Film").
        name: String,
    },
    /// Imported by the user.
    User,
}

impl From<&LookGroup> for LookGroupInfo {
    fn from(g: &LookGroup) -> Self {
        match g {
            LookGroup::BuiltIn => Self::BuiltIn,
            LookGroup::Arri {
                group_no,
                group_name,
            } => Self::Arri {
                no: *group_no,
                name: group_name.clone(),
            },
            LookGroup::User => Self::User,
        }
    }
}

/// One entry of the look library (`Look` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookEntry {
    /// `builtin:neutral`, `arri:3000:film`, `user:warm`.
    pub id: LookId,
    /// The display name.
    pub name: String,
    /// Where it is listed.
    pub group: LookGroupInfo,
    /// What its LUT expects as input.
    pub input: LookInput,
    /// The user's default look (the DEFAULT tag).
    pub is_default: bool,
    /// The look's file is gone or its hash changed: shown disabled.
    pub missing: bool,
}

/// An imported display LUT (`DisplayLut` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplayLutEntry {
    /// `arri-display:logc4:rec709-g24`.
    pub id: LookId,
    /// A readable name ("Rec.709 · Gamma 2.4").
    pub name: String,
    /// What it expects as input.
    pub input: LookInput,
    /// The display it renders to (`rec709-g24`).
    pub target: String,
}

/// What an import reports (`LooksImportResult` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LooksImportResult {
    /// Looks added or replaced by this import; empty when everything was already there.
    pub added: Vec<LookEntry>,
    /// Display LUTs added (ARRI's display rendering, not looks).
    pub display_luts_added: usize,
    /// Files whose hash matched a look or display LUT that was already imported.
    pub unchanged: usize,
}

/// How an import is run.
#[derive(Debug, Clone, Default)]
pub struct ImportRequest {
    /// The answer to the generic question for `.cube` files that no known package claims.
    pub input: Option<LookInput>,
    /// Lets the wizard's Cancel button stop the import between files.
    pub cancel: Option<CancelToken>,
}

/// A readable name for a display LUT target: `rec709-g24` is "Rec.709 · Gamma 2.4".
#[must_use]
pub fn display_lut_name(target: &str) -> String {
    let word = |w: &str| -> String {
        match w {
            "rec709" | "bt709" => "Rec.709".to_owned(),
            "rec2020" | "bt2020" => "Rec.2020".to_owned(),
            "g22" => "Gamma 2.2".to_owned(),
            "g24" => "Gamma 2.4".to_owned(),
            "g26" => "Gamma 2.6".to_owned(),
            "p3" => "P3".to_owned(),
            "dcip3" => "DCI-P3".to_owned(),
            "d65" => "D65".to_owned(),
            "d60" => "D60".to_owned(),
            "hlg" => "HLG".to_owned(),
            "pq" => "PQ".to_owned(),
            "srgb" => "sRGB".to_owned(),
            other => match other.strip_suffix("nits") {
                Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
                    format!("{n} nits")
                }
                _ => {
                    let mut chars = other.chars();
                    chars.next().map_or_else(String::new, |c| {
                        c.to_uppercase().collect::<String>() + chars.as_str()
                    })
                }
            },
        }
    };
    let words: Vec<String> = target
        .split('-')
        .filter(|w| !w.is_empty())
        .map(word)
        .collect();
    if words.is_empty() {
        target.to_owned()
    } else {
        words.join(" · ")
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The library as a [`LookProvider`], with the SHA-256 of built-in looks added: a built-in look has
/// no imported file, so the hash of its description (`data/looks/builtin/<slug>.json`) stands in
/// for it in an export's `ym:LookSHA256`.
struct LibraryProvider {
    library: Arc<RwLock<Library>>,
    builtin_dir: Option<PathBuf>,
}

impl LookProvider for LibraryProvider {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        self.library.look(id)
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        self.library.display_transform(id)
    }

    fn look_sha256(&self, id: &LookId) -> Option<String> {
        if let Some(sha) = self.library.look_sha256(id) {
            return Some(sha);
        }
        let dir = self.builtin_dir.as_ref()?;
        if id.prefix() != "builtin" {
            return None;
        }
        hash_file(&dir.join(format!("{}.json", id.slug()))).ok()
    }
}

type Listener = Arc<dyn Fn() + Send + Sync>;

/// What a look or display LUT is from outside: its id, whether it can be used, and the SHA-256 of
/// its file (so a look replaced by another file under the same id is a change too).
type ListedFile = (LookId, bool, Option<String>);

/// What a change of the library looks like from outside: which looks and display LUTs there are,
/// whether each can be used, which file each is, and which look is the default.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Listing {
    looks: Vec<ListedFile>,
    display_luts: Vec<ListedFile>,
    default: LookId,
}

/// The look library with its preferences and change notification (see the module documentation).
pub struct LookService {
    app_root: PathBuf,
    builtin_dir: Option<PathBuf>,
    library: Arc<RwLock<Library>>,
    packages: PackageDb,
    prefs: Arc<PreferencesStore>,
    listeners: Mutex<Vec<Listener>>,
}

impl std::fmt::Debug for LookService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LookService")
            .field("app_root", &self.app_root)
            .finish_non_exhaustive()
    }
}

impl LookService {
    /// Opens the library of the app folder `app_root` (the folder `preferences.json` is in; see
    /// `ym_looks::library::default_root`), with the built-in looks and the package list of
    /// `data_root`. Nothing is created on disk. A package list that cannot be read is logged and
    /// replaced by an empty one (every file then gets the generic question).
    ///
    /// # Errors
    ///
    /// [`LooksError::Library`] (currently never).
    pub fn open(
        app_root: &Path,
        data_root: &Path,
        prefs: Arc<PreferencesStore>,
    ) -> Result<Self, LooksError> {
        let builtin_dir = data_root.join("looks").join("builtin");
        let builtin_dir = builtin_dir.is_dir().then_some(builtin_dir);
        let library = Library::open(app_root, builtin_dir.as_deref())?;
        for problem in library.problems() {
            // The text of a problem names the looks list in the person's folder, and the log can
            // end up in a bug report (`docs/01`): the kind at warn level, the text at debug.
            let kind = match &problem {
                LibraryProblem::Builtin(_) => "builtin",
                LibraryProblem::Manifest(_) => "manifest",
            };
            tracing::warn!(kind, "the look library has a problem");
            tracing::debug!(?problem, "the look library has a problem: details");
        }
        let packages = match PackageDb::load_from_data_dir(data_root) {
            Ok(db) => db,
            Err(e) => {
                tracing::warn!("the list of ARRI packages could not be read");
                tracing::debug!(error = %e, "the list of ARRI packages could not be read: details");
                PackageDb::empty()
            }
        };
        Ok(Self {
            app_root: app_root.to_path_buf(),
            builtin_dir,
            library: Arc::new(RwLock::new(library)),
            packages,
            prefs,
            listeners: Mutex::new(Vec::new()),
        })
    }

    /// The preferences the default look and the display rendering are kept in.
    #[must_use]
    pub fn preferences(&self) -> &Arc<PreferencesStore> {
        &self.prefs
    }

    /// The shared library (the render graph reads looks through [`LookService::provider`]).
    #[must_use]
    pub fn library(&self) -> &Arc<RwLock<Library>> {
        &self.library
    }

    /// The package list of `data/looks/arri-packages.json` (for the wizard's recognition step).
    #[must_use]
    pub fn packages(&self) -> &PackageDb {
        &self.packages
    }

    /// The app folder the library lives in.
    #[must_use]
    pub fn app_root(&self) -> &Path {
        &self.app_root
    }

    /// Where the render graph gets looks and display LUTs from.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn LookProvider> {
        Arc::new(LibraryProvider {
            library: Arc::clone(&self.library),
            builtin_dir: self.builtin_dir.clone(),
        })
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Library> {
        self.library.read().unwrap_or_else(PoisonError::into_inner)
    }

    // --- Listing --------------------------------------------------------------------------

    /// `looks_list`: every look in display order (built in, ARRI by group, the user's own), the
    /// default one flagged, the ones whose file is gone marked `missing`.
    #[must_use]
    pub fn list(&self) -> Vec<LookEntry> {
        let default = self.effective_default();
        self.read()
            .looks()
            .iter()
            .map(|l| entry_of(l, &default))
            .collect()
    }

    /// `looks_display_luts`: the imported display LUTs.
    #[must_use]
    pub fn display_luts(&self) -> Vec<DisplayLutEntry> {
        self.read()
            .display_luts()
            .iter()
            .map(display_entry)
            .collect()
    }

    /// The look new photos get: the preference if that look exists and its file is there, else
    /// Neutral.
    #[must_use]
    pub fn effective_default(&self) -> LookId {
        let wanted = self.prefs.get().default_look;
        let usable = self.read().get(&wanted).is_some_and(Look::is_enabled);
        if usable {
            wanted
        } else {
            LookId::builtin("neutral")
        }
    }

    /// The display rendering in effect: the chosen ARRI display LUT if it is still imported and
    /// its file is there, else ACES 2.
    #[must_use]
    pub fn display_choice(&self) -> DisplayChoice {
        match self.prefs.get().display_rendering {
            DisplayRendering::Aces2 => DisplayChoice::Aces2,
            DisplayRendering::ArriDisplay { id } => {
                let usable = self
                    .read()
                    .display_lut_info(&id)
                    .is_some_and(|d| d.status.is_ok());
                if usable {
                    DisplayChoice::ArriLut(id)
                } else {
                    DisplayChoice::Aces2
                }
            }
        }
    }

    /// Finds the look a `--look` argument means (full id, slug or name; see
    /// `ym_looks::library::resolve`).
    ///
    /// # Errors
    ///
    /// [`LooksError::Resolve`] (exit code 2 for an ambiguous name, 3 for none).
    pub fn resolve(&self, query: &str) -> Result<LookId, LooksError> {
        Ok(self.read().resolve(query)?.id.clone())
    }

    /// Checks that a look exists and can be used.
    ///
    /// # Errors
    ///
    /// [`LooksError::NotFound`] or [`LooksError::Missing`].
    pub fn check_usable(&self, id: &LookId) -> Result<(), LooksError> {
        match self.read().get(id) {
            None => Err(LooksError::NotFound(id.clone())),
            Some(l) if !l.is_enabled() => Err(LooksError::Missing(id.clone())),
            Some(_) => Ok(()),
        }
    }

    // --- Changes --------------------------------------------------------------------------

    /// Registers a callback for `looks_changed` (see the module documentation).
    pub fn on_change(&self, f: impl Fn() + Send + Sync + 'static) {
        lock(&self.listeners).push(Arc::new(f));
    }

    fn notify(&self) {
        let listeners: Vec<Listener> = lock(&self.listeners).clone();
        for l in listeners {
            l();
        }
    }

    /// What `looks_changed` compares: ids, whether usable, and the default.
    fn fingerprint(&self) -> Listing {
        let lib = self.read();
        Listing {
            looks: lib
                .looks()
                .iter()
                .map(|l| {
                    (
                        l.id.clone(),
                        l.is_enabled(),
                        l.lut.as_ref().map(|r| r.sha256.clone()),
                    )
                })
                .collect(),
            display_luts: lib
                .display_luts()
                .iter()
                .map(|d| (d.id.clone(), d.status.is_ok(), Some(d.lut.sha256.clone())))
                .collect(),
            default: self.prefs.get().default_look,
        }
    }

    /// Re-reads the manifest (the CLI or another app window may have imported something) and
    /// notifies when the list differs from before. Returns whether it did.
    pub fn reload(&self) -> bool {
        // The picture before looking at the disk: the default look is in it, and the CLI changes it
        // through the preferences file, so the preferences are read after this point.
        let before = self.fingerprint();
        self.prefs.reload();
        self.library
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .reload();
        let changed = before != self.fingerprint();
        if changed {
            self.notify();
        }
        changed
    }

    /// Re-hashes every imported file and updates which looks are missing or changed.
    ///
    /// # Errors
    ///
    /// [`LooksError::Library`] when the manifest is damaged or from a newer version.
    pub fn verify(&self) -> Result<VerifyReport, LooksError> {
        let before = self.fingerprint();
        let report = self
            .library
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .verify(&mut |_| {})?;
        if before != self.fingerprint() {
            self.notify();
        }
        Ok(report)
    }

    fn private_library(&self) -> Result<Library, LooksError> {
        Ok(Library::open(&self.app_root, self.builtin_dir.as_deref())?)
    }

    /// `looks_import`: imports a zip, a folder or a `.cube` file with the importer of the wizard
    /// and `ym looks import` (same recognition rules, checksums and library folder), then
    /// refreshes the library and notifies. Re-importing the same files adds nothing.
    ///
    /// # Errors
    ///
    /// [`LooksError::Import`]: the source is missing or not a look package, a `.cube` needs the
    /// input question answered (`request.input`), another import holds the lock for too long, or
    /// the import was cancelled. Nothing is added on an error.
    pub fn import(
        &self,
        path: &Path,
        request: &ImportRequest,
        progress: &mut dyn FnMut(Progress<'_>),
    ) -> Result<LooksImportResult, LooksError> {
        let mut lib = self.private_library()?;
        let options = ImportOptions {
            input: request.input,
            cancel: request.cancel.clone(),
            ..ImportOptions::default()
        };
        let (_, report) =
            ym_looks::import::import_path(&mut lib, &self.packages, path, &options, progress)?;
        self.reload();
        Ok(self.result_of(&report))
    }

    fn result_of(&self, report: &ImportReport) -> LooksImportResult {
        let default = self.effective_default();
        let lib = self.read();
        let added = report
            .added
            .iter()
            .chain(&report.updated)
            .filter(|i| !i.display_lut)
            .filter_map(|i| lib.get(&i.id))
            .map(|l| entry_of(l, &default))
            .collect();
        LooksImportResult {
            added,
            display_luts_added: report
                .added
                .iter()
                .chain(&report.updated)
                .filter(|i| i.display_lut)
                .count(),
            unchanged: report.unchanged.len(),
        }
    }

    /// `looks_remove`: removes an imported look or display LUT and its file. If it was the default
    /// look, new photos get Neutral again; if it was the chosen display rendering, ACES 2 is used
    /// again.
    ///
    /// # Errors
    ///
    /// [`LooksError::Library`]: a built-in look, an id that is not imported, or a file that cannot
    /// be deleted.
    pub fn remove(&self, id: &LookId) -> Result<(), LooksError> {
        let mut lib = self.private_library()?;
        lib.remove(id)?;
        let prefs = self.prefs.get();
        let mut patch = PreferencesPatch::default();
        if &prefs.default_look == id {
            patch.default_look = Some(LookId::builtin("neutral"));
        }
        if matches!(&prefs.display_rendering, DisplayRendering::ArriDisplay { id: d } if d == id) {
            patch.display_rendering = Some(DisplayRendering::Aces2);
        }
        if patch != PreferencesPatch::default() {
            // Failing to save a preference must not undo a removal that has happened.
            if let Err(e) = self.prefs.update(&patch) {
                tracing::warn!(
                    kind = e.kind(),
                    "could not reset a preference after removing {id}"
                );
                tracing::debug!(error = %e, "could not reset a preference: details");
            }
        }
        // The removal is done; the shared library re-reads it and `looks_changed` fires.
        self.library
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .reload();
        self.notify();
        Ok(())
    }

    /// `looks_set_default`: makes a look the default for photos opened from now on.
    ///
    /// # Errors
    ///
    /// [`LooksError::NotFound`] or [`LooksError::Missing`] (a look whose file is gone cannot be the
    /// default), or the preferences could not be saved.
    pub fn set_default(&self, id: &LookId) -> Result<(), LooksError> {
        self.check_usable(id)?;
        let before = self.prefs.get().default_look;
        self.prefs
            .update(&PreferencesPatch::default_look(id.clone()))?;
        if &before != id {
            self.notify();
        }
        Ok(())
    }

    /// Chooses the display rendering (`Preferences › Display rendering`).
    ///
    /// # Errors
    ///
    /// [`LooksError::NoDisplayLut`] when the display LUT is not imported or its file is gone, or the
    /// preferences could not be saved.
    pub fn set_display_rendering(&self, choice: DisplayRendering) -> Result<(), LooksError> {
        if let DisplayRendering::ArriDisplay { id } = &choice {
            let usable = self
                .read()
                .display_lut_info(id)
                .is_some_and(|d| d.status.is_ok());
            if !usable {
                return Err(LooksError::NoDisplayLut(id.clone()));
            }
        }
        self.prefs.update(&PreferencesPatch {
            display_rendering: Some(choice),
            ..PreferencesPatch::default()
        })?;
        Ok(())
    }

    /// Checks that the display rendering the preferences ask for can be used: ACES 2 always can; an
    /// ARRI display LUT must be imported and its file there. [`LookService::display_choice`] falls
    /// back to ACES 2 for the viewport; an export calls this first so that it does not quietly
    /// render with a transform the user did not choose.
    ///
    /// # Errors
    ///
    /// [`LooksError::NoDisplayLut`] when the LUT is not imported, [`LooksError::Missing`] when its
    /// file is gone or changed.
    pub fn check_display_rendering(&self) -> Result<(), LooksError> {
        let DisplayRendering::ArriDisplay { id } = self.prefs.get().display_rendering else {
            return Ok(());
        };
        match self.read().display_lut_info(&id) {
            None => Err(LooksError::NoDisplayLut(id)),
            Some(d) if !d.status.is_ok() => Err(LooksError::Missing(id)),
            Some(_) => Ok(()),
        }
    }

    /// Checks the parts of a preferences value that depend on the library: the default look exists
    /// and is usable, the display LUT is imported.
    ///
    /// # Errors
    ///
    /// The error [`LookService::check_usable`] or [`LookService::set_display_rendering`] would give.
    pub fn validate_preferences(&self, patch: &PreferencesPatch) -> Result<(), LooksError> {
        if let Some(id) = &patch.default_look {
            self.check_usable(id)?;
        }
        if let Some(DisplayRendering::ArriDisplay { id }) = &patch.display_rendering {
            let usable = self
                .read()
                .display_lut_info(id)
                .is_some_and(|d| d.status.is_ok());
            if !usable {
                return Err(LooksError::NoDisplayLut(id.clone()));
            }
        }
        Ok(())
    }

    /// `preferences_set` for the preferences that involve looks: validates them against the
    /// library, saves the patch and, if the default look changed, notifies.
    ///
    /// # Errors
    ///
    /// As [`LookService::validate_preferences`], and [`PreferencesError`] from saving.
    pub fn preferences_set(&self, patch: &PreferencesPatch) -> Result<Preferences, LooksError> {
        self.validate_preferences(patch)?;
        let before = self.prefs.get().default_look;
        let next = self.prefs.update(patch)?;
        if next.default_look != before {
            self.notify();
        }
        Ok(next)
    }

    /// `apply_look`: copies a look and its intensity to the photos in `scope` (This photo, Picked,
    /// All). A look whose file is gone is refused.
    ///
    /// # Errors
    ///
    /// [`LooksError::NotFound`], [`LooksError::Missing`], or the session's error. Returns the photos
    /// that changed.
    pub fn apply_look(
        &self,
        session: &Session,
        scope: ApplyScope,
        id: &LookId,
        intensity: f64,
    ) -> Result<Vec<PhotoId>, LooksError> {
        self.check_usable(id)?;
        Ok(session.apply_look(scope, id, intensity)?)
    }
}

fn entry_of(look: &Look, default: &LookId) -> LookEntry {
    LookEntry {
        id: look.id.clone(),
        name: look.name.clone(),
        group: LookGroupInfo::from(&look.group),
        input: look.input,
        is_default: &look.id == default,
        missing: !look.is_enabled(),
    }
}

fn display_entry(d: &DisplayLut) -> DisplayLutEntry {
    DisplayLutEntry {
        id: d.id.clone(),
        name: display_lut_name(&d.target),
        input: match d.input {
            DisplayInput::LogC4 => LookInput::LogC4,
            DisplayInput::LogC3 => LookInput::LogC3Ei800,
        },
        target: d.target.clone(),
    }
}

/// The id of Neutral, for callers that need a fallback look.
#[must_use]
pub fn neutral() -> LookId {
    LookId::parse(NEUTRAL_LOOK).unwrap_or_else(|_| LookId::builtin("neutral"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_lut_names_read_as_words() {
        assert_eq!(display_lut_name("rec709-g24"), "Rec.709 · Gamma 2.4");
        assert_eq!(display_lut_name("p3-d65"), "P3 · D65");
        assert_eq!(display_lut_name("rec2020"), "Rec.2020");
        assert_eq!(
            display_lut_name("rec2020-pq-1000nits"),
            "Rec.2020 · PQ · 1000 nits"
        );
        assert_eq!(display_lut_name("hlg"), "HLG");
        assert_eq!(display_lut_name("mystery-view"), "Mystery · View");
        assert_eq!(display_lut_name(""), "");
    }

    #[test]
    fn the_group_and_entry_json_follow_the_typescript_types() {
        let groups = [
            (
                LookGroupInfo::BuiltIn,
                serde_json::json!({ "kind": "built_in" }),
            ),
            (
                LookGroupInfo::Arri {
                    no: 3000,
                    name: "Film".into(),
                },
                serde_json::json!({ "kind": "arri", "no": 3000, "name": "Film" }),
            ),
            (LookGroupInfo::User, serde_json::json!({ "kind": "user" })),
        ];
        for (info, json) in groups {
            assert_eq!(serde_json::to_value(&info).expect("json"), json);
        }
        let entry = LookEntry {
            id: LookId::builtin("neutral"),
            name: "Neutral".into(),
            group: LookGroupInfo::BuiltIn,
            input: LookInput::LogC3ToRec709,
            is_default: true,
            missing: false,
        };
        assert_eq!(
            serde_json::to_value(&entry).expect("json"),
            serde_json::json!({
                "id": "builtin:neutral", "name": "Neutral", "group": { "kind": "built_in" },
                "input": "logc3_to_rec709", "is_default": true, "missing": false
            })
        );
    }
}
