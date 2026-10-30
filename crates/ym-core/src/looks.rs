//! The look library as the app and the CLI use it: listing, importing, removing, the default look
//! with a `looks_changed` notification (`docs/08-looks.md`,
//! `docs/01-architecture.md` IPC commands `looks_list`, `looks_import`, `looks_set_default`,
//! `looks_remove`, `apply_look`).
//!
//! # What lives where
//!
//! The library itself (manifest, files, hashes, import, recognition) is `ym-looks`; this module wraps
//! one [`Library`] behind a lock so that the render graph can read it while the UI changes it, and
//! ties it to the preference that belongs to looks, the **default look**
//! (`Preferences::default_look`): the look new photos get, Signature out of the box. A default that
//! has been removed or whose file is gone is not used; new photos get Signature again (Neutral when
//! the built-in Signature file is missing too) until another is chosen
//! ([`LookService::effective_default`]).
//!
//! There is no choice of display rendering: Yellowmoon always renders with its own. ARRI's display
//! LUTs are not imported (`docs/02-colour-pipeline.md`, C12), and a library that an older version
//! filled with them is read without them.
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
    DISABLED_MESSAGE, Library, LibraryError, LibraryProblem, LoadError, Look, LookFormat,
    LookGroup, LookId, LookLibrary, ResolveError, VerifyReport,
};
use ym_looks::manifest::hash_file;
use ym_looks::recognize::PackageDb;

use crate::pipeline::color::{LookProblem, LookProvider, ResolvedLook};
use crate::preferences::{
    NEUTRAL_LOOK, Preferences, PreferencesError, PreferencesPatch, PreferencesStore,
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
}

impl LooksError {
    /// The CLI exit code: 3 for a look that is missing, 2 for a usage mistake, 1 otherwise.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotFound(_) | Self::Missing(_) | Self::Load(_) => 3,
            Self::Resolve(e) => e.exit_code(),
            Self::Import(e) => e.exit_code(),
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
    /// The library it is listed under: its section of the Look tab and the LIBRARY column of
    /// `ym looks list` (`Look::library`, `docs/08-looks.md`).
    pub library: LookLibrary,
    /// The kind of file the look is kept as; `None` for a built-in look, which has no file.
    pub format: Option<LookFormat>,
    /// Points per axis of the look's 3D table (65, 33, …); `None` for a built-in look.
    pub lut_size: Option<usize>,
    /// The user's default look (the DEFAULT tag).
    pub is_default: bool,
    /// The look's file is gone or its hash changed: shown disabled.
    pub missing: bool,
}

/// What an import reports (`LooksImportResult` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LooksImportResult {
    /// Looks added or replaced by this import; empty when everything was already there.
    pub added: Vec<LookEntry>,
    /// Files whose hash matched a look that was already imported.
    pub unchanged: usize,
    /// Things worth telling the person, in plain words: an `.alf4c`'s display transforms are not
    /// used; ARRI's display LUTs in the source were left out; looks that stayed or became the 3D-LUT
    /// version of a library; a damaged list that was set aside. Empty (and left out of the JSON)
    /// when there is nothing to say.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// One sentence for each file of the source that could not be used (damaged, made for another
    /// display, …) while the others were imported, as the recogniser words them ("Torn.alf4
    /// cannot be used. Its checksum does not match, so the file is damaged. Download it again."),
    /// at most 12 and then how many more. The webview names these files on the Recognise step
    /// before the import, so they are not sent to it; `ym looks import` prints them as errors and
    /// exits 1 (`docs/10-cli.md`). Empty (and left out of the JSON) when nothing was refused.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refusals: Vec<String>,
}

/// How an import is run.
#[derive(Debug, Clone, Default)]
pub struct ImportRequest {
    /// The answer to the generic question for `.cube` files that no known package claims.
    pub input: Option<LookInput>,
    /// Lets the wizard's Cancel button stop the import between files.
    pub cancel: Option<CancelToken>,
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

/// What a look is from outside: its id, whether it can be used, and the SHA-256 of its file (so a
/// look replaced by another file under the same id is a change too).
type ListedFile = (LookId, bool, Option<String>);

/// What a change of the library looks like from outside: which looks there are, whether each can
/// be used, which file each is, and which look is the default.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Listing {
    looks: Vec<ListedFile>,
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

    /// The preferences the default look is kept in.
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

    /// Where the render graph gets looks from.
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

    /// The look new photos get: the preference if that look exists and its file is there, else
    /// Signature (the default out of the box), else Neutral, which exists in code even when the
    /// data folder has no look files.
    #[must_use]
    pub fn effective_default(&self) -> LookId {
        let wanted = self.prefs.get().default_look;
        let lib = self.read();
        let usable = |id: &LookId| lib.get(id).is_some_and(Look::is_enabled);
        if usable(&wanted) {
            return wanted;
        }
        let factory = LookId::factory_default();
        if usable(&factory) {
            factory
        } else {
            LookId::neutral()
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
        let (recognition, report) =
            ym_looks::import::import_path(&mut lib, &self.packages, path, &options, progress)?;
        let followed = self.follow_renames(&report);
        self.reload();
        let mut result = self.result_of(&report);
        // What the recognition found worth saying (the display transforms of an `.alf4c` that are
        // not used, ARRI's display LUTs that were left out, and the files that were refused while
        // the rest came in, each with its reason), then what the import did.
        use ym_looks::recognize::IssueKind;
        let said = |kind: IssueKind| {
            recognition
                .issues
                .iter()
                .filter(move |i| i.kind == kind)
                .map(|i| i.message.clone())
        };
        result.refusals = said(IssueKind::Rejected).collect();
        result.notes = said(IssueKind::DisplayTransformsIgnored)
            .chain(said(IssueKind::DisplayRenderings))
            .chain(report.notes.iter().cloned())
            .chain(followed)
            .collect();
        Ok(result)
    }

    /// The default look is one of the person's own looks that an ARRI package has just claimed
    /// (its id changed): the preference follows it, so new photos and `ym render` keep using the
    /// look the person chose. Returns the sentence that says so.
    fn follow_renames(&self, report: &ImportReport) -> Option<String> {
        let default = self.prefs.get().default_look;
        let (_, now) = report.renamed.iter().find(|(old, _)| *old == default)?;
        let patch = PreferencesPatch {
            default_look: Some(now.clone()),
            ..PreferencesPatch::default()
        };
        // Failing to save a preference must not undo an import that has happened.
        if let Err(e) = self.prefs.update(&patch) {
            tracing::warn!(
                kind = e.kind(),
                "could not keep the default look after its id changed: {e}"
            );
            return None;
        }
        Some(format!(
            "Your default look {default} is now {now}, so new photos still start with it."
        ))
    }

    fn result_of(&self, report: &ImportReport) -> LooksImportResult {
        let default = self.effective_default();
        let lib = self.read();
        let added = report
            .added
            .iter()
            .chain(&report.updated)
            .filter_map(|i| lib.get(&i.id))
            .map(|l| entry_of(l, &default))
            .collect();
        LooksImportResult {
            added,
            unchanged: report.unchanged.len(),
            notes: Vec::new(),
            refusals: Vec::new(),
        }
    }

    /// `looks_remove`: removes an imported look and its file. If it was the default look, new photos
    /// get Signature again (the default out of the box).
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
            patch.default_look = Some(LookId::factory_default());
        }
        if patch != PreferencesPatch::default() {
            // Failing to save a preference must not undo a removal that has happened.
            if let Err(e) = self.prefs.update(&patch) {
                tracing::warn!(
                    kind = e.kind(),
                    "could not reset the default look after removing {id}"
                );
                tracing::debug!(error = %e, "could not reset the default look: details");
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

    /// Checks the parts of a preferences value that depend on the library: the default look exists
    /// and is usable.
    ///
    /// # Errors
    ///
    /// The error [`LookService::check_usable`] would give.
    pub fn validate_preferences(&self, patch: &PreferencesPatch) -> Result<(), LooksError> {
        if let Some(id) = &patch.default_look {
            self.check_usable(id)?;
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
        library: look.library(),
        format: look.format(),
        lut_size: look.lut_size(),
        is_default: &look.id == default,
        missing: !look.is_enabled(),
    }
}

/// The id of Neutral, for callers that need a fallback look.
#[must_use]
pub fn neutral() -> LookId {
    LookId::parse(NEUTRAL_LOOK).unwrap_or_else(|_| LookId::neutral())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            library: LookLibrary::BuiltIn,
            format: None,
            lut_size: None,
            is_default: true,
            missing: false,
        };
        assert_eq!(
            serde_json::to_value(&entry).expect("json"),
            serde_json::json!({
                "id": "builtin:neutral", "name": "Neutral", "group": { "kind": "built_in" },
                "input": "logc3_to_rec709", "library": "built_in", "format": null,
                "lut_size": null, "is_default": true, "missing": false
            })
        );
        let imported = LookEntry {
            id: LookId::arri(3000, "film"),
            name: "Film".into(),
            group: LookGroupInfo::Arri {
                no: 3000,
                name: "Film".into(),
            },
            input: LookInput::LogC3ToRec709,
            library: LookLibrary::ArriAlf2,
            format: Some(LookFormat::Aml),
            lut_size: Some(33),
            is_default: false,
            missing: false,
        };
        assert_eq!(
            serde_json::to_value(&imported).expect("json"),
            serde_json::json!({
                "id": "arri:3000:film", "name": "Film",
                "group": { "kind": "arri", "no": 3000, "name": "Film" },
                "input": "logc3_to_rec709", "library": "arri_alf2", "format": "aml",
                "lut_size": 33, "is_default": false, "missing": false
            })
        );
    }
}
