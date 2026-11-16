//! [`PresetService`]: the presets folder tied to the look library and the session (`docs/30` section 7), and the wire types of
//! its answers (section 8). `Core::preset_*` forwards to it, one method for each IPC command.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use ym_looks::library::LookId;

use super::store::{PresetListing, PresetStore, StoredPreset};
use super::{
    MAX_PRESET_BYTES, Preset, PresetError, PresetGroup, PresetLook, changed_groups,
    changed_groups_of_settings,
};
use crate::looks::LookService;
use crate::session::{PastedPhoto, PhotoId, Session};
use crate::settings::DevelopSettings;

/// A preset as `presets_list` shows it (`PresetEntry` of `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetEntry {
    /// The name, which is the identity (ignoring case).
    pub name: String,
    /// The groups it holds, in the order of `docs/30` section 2.
    pub groups: Vec<PresetGroup>,
    /// The look it names, with the `look` group; `null` without.
    pub look: Option<PresetLook>,
    /// The look is not in the library, or its file is gone or changed.
    pub look_missing: bool,
    /// Applying the preset to the asked photo would change nothing (the ring on its tile); false for no photo.
    pub matches: bool,
    /// The first 12 hex digits of the file's SHA-256: the thumbnail's key, new when the preset is saved again.
    pub revision: String,
}

/// A file of the folder that could not be used, listed so that it can be deleted (`DamagedPreset`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DamagedPreset {
    /// The file's name in the folder.
    pub file: String,
    /// Why, in the closed list of `docs/30` section 6.
    pub reason: String,
}

/// What `presets_list` answers: the presets, the damaged files and the folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetList {
    /// The presets, sorted by name ignoring case.
    pub presets: Vec<PresetEntry>,
    /// The files that could not be used.
    pub damaged: Vec<DamagedPreset>,
    /// The folder, for "Show in Finder" and the write error.
    pub folder: String,
}

/// Where a preset is made from (`preset_inspect`, `preset_save`): `{ kind: "photo", photo_id }` or `{ kind: "file", path }`
/// (a settings file; an exported photo too from P-06).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PresetSource {
    /// A photo of the session.
    Photo {
        /// The photo.
        photo_id: PhotoId,
    },
    /// A file.
    File {
        /// Its path.
        path: PathBuf,
    },
}

/// What the Save dialog needs to open (`preset_inspect`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetInspect {
    /// The groups to tick (`docs/30` section 4.1).
    pub changed: Vec<PresetGroup>,
    /// The source is an export that has no `ym:AutoFields` (`docs/30` section 12; always false until P-06).
    pub older_export: bool,
    /// The name of the source's look, when it is in the library.
    pub look: Option<String>,
}

/// What an application of a preset did: the fields of `PasteReport` and the look that was left out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetApplyReport {
    /// The photos that took the preset, in the order asked for, each with its undo pair.
    pub pasted: Vec<PastedPhoto>,
    /// The photos left out: still developing, failed or not in the session.
    pub skipped: Vec<PhotoId>,
    /// The name of the preset's look when it is not in the library (`missing_look_sentence` words it); `null` otherwise.
    pub missing_look: Option<String>,
}

/// A name that exists and was not replaced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportExists {
    /// The preset's name.
    pub name: String,
    /// The file it came from.
    pub path: String,
}

/// A file that was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportRefused {
    /// The file's name.
    pub file: String,
    /// The sentence of `docs/30` section 5.
    pub reason: String,
}

/// What an import did (`ImportResult` of `src/api/types.ts`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportOutcome {
    /// The names of the presets added.
    pub added: Vec<String>,
    /// The names of the presets replaced (only with `replace`).
    pub replaced: Vec<String>,
    /// The names that exist and were not replaced, with the file, so that the page can ask.
    pub exists: Vec<ImportExists>,
    /// The files that were refused.
    pub refused: Vec<ImportRefused>,
}

impl ImportOutcome {
    fn merge(&mut self, other: ImportOutcome) {
        self.added.extend(other.added);
        self.replaced.extend(other.replaced);
        self.exists.extend(other.exists);
        self.refused.extend(other.refused);
    }
}

type Listener = Arc<dyn Fn() + Send + Sync>;

/// The presets folder behind the look library and the session (see the module documentation).
pub struct PresetService {
    store: PresetStore,
    looks: Arc<LookService>,
    session: Arc<Session>,
    listeners: Mutex<Vec<Listener>>,
    /// What the last look at the folder found: the names with their revisions and the damaged files.
    seen: Mutex<Option<Vec<String>>>,
}

impl std::fmt::Debug for PresetService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresetService")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn fingerprint(listing: &PresetListing) -> Vec<String> {
    listing
        .presets
        .iter()
        .map(|s| format!("{}\u{0}{}", s.preset.name, s.revision))
        .chain(
            listing
                .damaged
                .iter()
                .map(|d| format!("damaged\u{0}{}\u{0}{}", d.file, d.reason)),
        )
        .collect()
}

impl PresetService {
    /// The presets of the application folder `app_root`, with the look checks of `looks` and the photos of `session`. Writes
    /// nothing.
    #[must_use]
    pub fn open(app_root: &Path, looks: Arc<LookService>, session: Arc<Session>) -> Self {
        Self::with_store(PresetStore::open(app_root), looks, session)
    }

    /// The service over a store the caller made (a test's fake Trash).
    #[must_use]
    pub fn with_store(store: PresetStore, looks: Arc<LookService>, session: Arc<Session>) -> Self {
        Self {
            store,
            looks,
            session,
            listeners: Mutex::new(Vec::new()),
            seen: Mutex::new(None),
        }
    }

    /// The store under the service.
    #[must_use]
    pub fn store(&self) -> &PresetStore {
        &self.store
    }

    /// Registers a callback for `presets_changed`. It runs after any change the service made, and when a listing found a
    /// different folder than the last one (a preset made by `ym`), outside every lock.
    pub fn on_change(&self, f: impl Fn() + Send + Sync + 'static) {
        lock(&self.listeners).push(Arc::new(f));
    }

    fn notify(&self) {
        let listeners: Vec<Listener> = lock(&self.listeners).clone();
        for l in listeners {
            l();
        }
    }

    /// Remembers what the folder holds after a change the service made, and tells the listeners.
    fn changed(&self) {
        *lock(&self.seen) = Some(fingerprint(&self.store.list()));
        self.notify();
    }

    fn look_usable(&self, preset: &Preset) -> bool {
        preset
            .look
            .as_ref()
            .is_none_or(|l| self.looks.check_usable(&l.id).is_ok())
    }

    fn look_name(&self, id: &LookId) -> Option<String> {
        self.looks
            .list()
            .into_iter()
            .find(|l| &l.id == id)
            .map(|l| l.name)
    }

    fn entry(&self, stored: &StoredPreset, photo: Option<&DevelopSettings>) -> PresetEntry {
        let usable = self.look_usable(&stored.preset);
        PresetEntry {
            name: stored.preset.name.clone(),
            groups: stored.preset.groups.clone(),
            look: stored.preset.look.clone(),
            look_missing: !usable,
            matches: photo.is_some_and(|s| stored.preset.matches(s, usable)),
            revision: stored.revision.clone(),
        }
    }

    fn settings_of(&self, photo: Option<PhotoId>) -> Option<DevelopSettings> {
        photo
            .and_then(|id| self.session.settings(id).ok())
            .map(|r| r.settings)
    }

    // --- Listing ----------------------------------------------------------------------------

    /// `presets_list`: reads the folder again, and adds to each preset whether applying it to the photo would change nothing
    /// ([`Preset::matches`]; false for `None` or a photo that cannot be asked). Nothing is stored. When the folder holds
    /// something other than the last listing found, `presets_changed` fires before this returns.
    #[must_use]
    pub fn list(&self, photo: Option<PhotoId>) -> PresetList {
        let listing = self.store.list();
        let now = fingerprint(&listing);
        let differs = {
            let mut seen = lock(&self.seen);
            let differs = seen.as_ref().is_some_and(|s| *s != now);
            *seen = Some(now);
            differs
        };
        if differs {
            self.notify();
        }
        let settings = self.settings_of(photo);
        PresetList {
            presets: listing
                .presets
                .iter()
                .map(|s| self.entry(s, settings.as_ref()))
                .collect(),
            damaged: listing.damaged,
            folder: self.store.folder().display().to_string(),
        }
    }

    // --- Making -----------------------------------------------------------------------------

    fn settings_from_file(&self, path: &Path) -> Result<DevelopSettings, PresetError> {
        let file = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let meta = std::fs::metadata(path).map_err(|source| PresetError::Read {
            file: file.clone(),
            source,
        })?;
        if meta.len() > 16 * MAX_PRESET_BYTES as u64 {
            return Err(PresetError::TooLarge);
        }
        let text = std::fs::read_to_string(path).map_err(|source| {
            // Bytes that are not text (a photo) are not a settings file, which is not a read error.
            if source.kind() == std::io::ErrorKind::InvalidData {
                PresetError::SettingsFile("the file is not text".to_owned())
            } else {
                PresetError::Read { file, source }
            }
        })?;
        DevelopSettings::from_json(&text, &self.looks.effective_default())
            .map(|(s, _)| s)
            .map_err(|e| PresetError::SettingsFile(e.to_string()))
    }

    fn make(
        &self,
        name: &str,
        groups: &[PresetGroup],
        source: &PresetSource,
    ) -> Result<Preset, PresetError> {
        match source {
            PresetSource::Photo { photo_id } => {
                let resolved = self.session.settings(*photo_id)?;
                let look = self.look_name(&resolved.settings.output().look);
                Preset::from_photo(name, &resolved, groups, look.as_deref())
            }
            PresetSource::File { path } => {
                let settings = self.settings_from_file(path)?;
                let look = self.look_name(&settings.output().look);
                Preset::from_settings(name, &settings, groups, look.as_deref())
            }
        }
    }

    /// `preset_inspect`: the groups the Save dialog ticks for the source, and the source's look's name.
    ///
    /// # Errors
    ///
    /// [`PresetError::Session`] for a photo that cannot be asked, the file's errors for a file.
    pub fn inspect(&self, source: &PresetSource) -> Result<PresetInspect, PresetError> {
        let defaults = DevelopSettings::with_look(self.looks.effective_default());
        let (changed, look) = match source {
            PresetSource::Photo { photo_id } => {
                let resolved = self.session.settings(*photo_id)?;
                (
                    changed_groups(&resolved, &defaults),
                    self.look_name(&resolved.settings.output().look),
                )
            }
            PresetSource::File { path } => {
                let settings = self.settings_from_file(path)?;
                (
                    changed_groups_of_settings(&settings, &defaults),
                    self.look_name(&settings.output().look),
                )
            }
        };
        Ok(PresetInspect {
            changed,
            older_export: false,
            look,
        })
    }

    /// `preset_save`: makes the preset from `source` (`docs/30` section 4.1) and writes it. Without `replace` a name that exists
    /// is refused.
    ///
    /// # Errors
    ///
    /// [`PresetError`].
    pub fn save(
        &self,
        name: &str,
        groups: &[PresetGroup],
        source: &PresetSource,
        replace: bool,
    ) -> Result<PresetEntry, PresetError> {
        let preset = self.make(name, groups, source)?;
        let stored = self.store.save(&preset, replace)?;
        self.changed();
        let photo = match source {
            PresetSource::Photo { photo_id } => self.settings_of(Some(*photo_id)),
            PresetSource::File { .. } => None,
        };
        Ok(self.entry(&stored, photo.as_ref()))
    }

    /// `preset_update`: the photo's values again for the preset's own groups, under the same name.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Session`], [`PresetError::Write`].
    pub fn update(&self, name: &str, photo_id: PhotoId) -> Result<PresetEntry, PresetError> {
        let current = self.store.get(name)?;
        let source = PresetSource::Photo { photo_id };
        let preset = self.make(&current.name, &current.groups, &source)?;
        let stored = self.store.save(&preset, true)?;
        self.changed();
        Ok(self.entry(&stored, self.settings_of(Some(photo_id)).as_ref()))
    }

    /// `preset_rename`: refuses a name that exists, case aside from its own.
    ///
    /// # Errors
    ///
    /// [`PresetError`].
    pub fn rename(&self, name: &str, new_name: &str) -> Result<PresetEntry, PresetError> {
        let stored = self.store.rename(name, new_name)?;
        self.changed();
        Ok(self.entry(&stored, None))
    }

    /// `preset_remove`: moves the preset's file to the Trash.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Trash`].
    pub fn remove(&self, name: &str) -> Result<(), PresetError> {
        self.store.remove(name)?;
        self.changed();
        Ok(())
    }

    /// `preset_remove_file`: moves a damaged file to the Trash, by its file name.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Trash`].
    pub fn remove_file(&self, file: &str) -> Result<(), PresetError> {
        self.store.remove_file(file)?;
        self.changed();
        Ok(())
    }

    /// `presets_import`: reads each file into the folder. A name that exists is not replaced unless `replace`.
    #[must_use]
    pub fn import(&self, paths: &[PathBuf], replace: bool) -> ImportOutcome {
        let mut outcome = ImportOutcome::default();
        for path in paths {
            outcome.merge(self.store.import(path, replace));
        }
        if !outcome.added.is_empty() || !outcome.replaced.is_empty() {
            self.changed();
        }
        outcome
    }

    /// `preset_export`: writes the preset's file to `dest`, overwriting it.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Write`].
    pub fn export(&self, name: &str, dest: &Path) -> Result<(), PresetError> {
        self.store.export(name, dest)
    }

    /// `preset_reveal`: the preset's file.
    #[must_use]
    pub fn path_of(&self, name: &str) -> Option<PathBuf> {
        self.store.path_of(name)
    }

    // --- Applying ---------------------------------------------------------------------------

    /// `preset_apply`: loads the preset, drops the look when the library has none, and gives its patch to
    /// [`Session::paste_settings`] once: one call for many photos under one lock, atomic, with one undo pair for each photo.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Session`] for a patch that cannot be applied (nothing is changed).
    pub fn apply(
        &self,
        name: &str,
        photo_ids: &[PhotoId],
    ) -> Result<PresetApplyReport, PresetError> {
        let preset = self.store.get(name)?;
        let patch = preset.patch_for(&DevelopSettings::default(), self.look_usable(&preset));
        let report = self.session.paste_settings(photo_ids, &patch.patch)?;
        Ok(PresetApplyReport {
            pasted: report.pasted,
            skipped: report.skipped,
            missing_look: patch.missing_look,
        })
    }

    /// The settings of a preset's tile (`docs/30` section 8): the photo's own with the preset applied.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Settings`].
    pub fn thumbnail_settings(
        &self,
        name: &str,
        settings: &DevelopSettings,
    ) -> Result<DevelopSettings, PresetError> {
        let preset = self.store.get(name)?;
        preset.applied_to(settings, self.look_usable(&preset))
    }
}
