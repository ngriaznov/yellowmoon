//! The `presets/` folder (`docs/30` section 6): one `<safe name>.ympreset` file for each preset, written atomically under
//! `presets/.lock`, read without a lock.
//!
//! Nothing else is written: no index, no cache, no thumbnails. A folder that is not there is an empty list, and nothing is
//! created until the first write.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};
use ym_looks::import::trash::{TrashError, move_to_trash};
use ym_looks::lock::{DEFAULT_TIMEOUT, LibraryLock, LockError};
use ym_looks::manifest::atomic_write;

use super::service::{DamagedPreset, ImportExists, ImportOutcome, ImportRefused};
use super::{
    COULDNT_READ_SENTENCE, DUPLICATE_NAME_SENTENCE, FILE_EXTENSION, MAX_PRESET_BYTES,
    NOT_A_PRESET_SENTENCE, Preset, PresetError, same_name,
};

/// The folder's name inside the application folder, beside `looks/`.
pub const FOLDER_NAME: &str = "presets";

/// The longest stem of a file name, in bytes.
const MAX_STEM_BYTES: usize = 200;

type TrashFn = dyn Fn(&Path) -> Result<(), TrashError> + Send + Sync;
type WriteFn = dyn Fn(&Path, &[u8]) -> io::Result<()> + Send + Sync;

/// A preset as the folder holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredPreset {
    /// The preset.
    pub preset: Preset,
    /// The file's name inside the folder (`Warm evening.ympreset`): a label, the name in the file is the truth.
    pub file: String,
    /// The first 12 hex digits of the file's SHA-256: the thumbnail key, and what tells a preset that was saved again.
    pub revision: String,
}

/// What [`PresetStore::list`] found: the presets sorted by name (ignoring case, then by file name) and the files that could
/// not be used, with the reason, so that they can be deleted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PresetListing {
    /// The presets that read.
    pub presets: Vec<StoredPreset>,
    /// The files that did not.
    pub damaged: Vec<DamagedPreset>,
}

/// The presets folder.
pub struct PresetStore {
    folder: PathBuf,
    lock_timeout: Duration,
    trash: Arc<TrashFn>,
    write: Arc<WriteFn>,
}

impl std::fmt::Debug for PresetStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresetStore")
            .field("folder", &self.folder)
            .finish_non_exhaustive()
    }
}

fn revision_of(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn is_preset_file_name(name: &str) -> bool {
    !name.starts_with('.')
        && Path::new(name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(FILE_EXTENSION))
}

fn stem_of(file: &str) -> &str {
    Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file)
}

impl PresetStore {
    /// The store of the application folder `app_root` (the folder `preferences.json` and `looks/` are in; `YELLOWMOON_HOME`
    /// replaces it): its `presets/` folder. Nothing is created.
    #[must_use]
    pub fn open(app_root: &Path) -> Self {
        Self {
            folder: app_root.join(FOLDER_NAME),
            lock_timeout: DEFAULT_TIMEOUT,
            trash: Arc::new(move_to_trash),
            write: Arc::new(atomic_write),
        }
    }

    /// Waits at most `timeout` for `presets/.lock` (the default is the look library's two minutes).
    #[must_use]
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Moves files to Trash through `trash` instead of the system's (a test's fake Trash).
    #[must_use]
    pub fn with_trash(
        mut self,
        trash: impl Fn(&Path) -> Result<(), TrashError> + Send + Sync + 'static,
    ) -> Self {
        self.trash = Arc::new(trash);
        self
    }

    /// Writes files through `write` instead of [`atomic_write`] (a test's failing disk).
    #[must_use]
    pub fn with_writer(
        mut self,
        write: impl Fn(&Path, &[u8]) -> io::Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.write = Arc::new(write);
        self
    }

    /// The `presets/` folder.
    #[must_use]
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    // --- Names ------------------------------------------------------------------------------

    /// The name made safe for every file system (`docs/30` section 6): `/ \ : * ? " < > |` and control characters become `-`,
    /// runs of spaces, and of dashes, are kept to one, leading and trailing spaces and dots are cut, a name that is empty after
    /// that becomes `preset`, and the stem is cut to 200 bytes at a character boundary. The extension is not added.
    #[must_use]
    pub fn safe_file_name(name: &str) -> String {
        let mut out = String::new();
        let mut last = None;
        for c in name.chars() {
            let c = if c.is_control() || "/\\:*?\"<>|".contains(c) {
                '-'
            } else {
                c
            };
            if (c == ' ' || c == '-') && last == Some(c) {
                continue;
            }
            last = Some(c);
            out.push(c);
        }
        let trim = |s: &str| s.trim_matches(|c| c == ' ' || c == '.').to_owned();
        let mut stem = trim(&out);
        if stem.len() > MAX_STEM_BYTES {
            let mut end = MAX_STEM_BYTES;
            while !stem.is_char_boundary(end) {
                end -= 1;
            }
            stem.truncate(end);
            stem = trim(&stem);
        }
        if stem.is_empty() {
            "preset".to_owned()
        } else {
            stem
        }
    }

    /// A file name for a new preset: the safe name, with ` 2`, ` 3` … after the stem while another file (of any kind, ignoring
    /// case, as a case-insensitive disk would) has it.
    fn new_file_name(&self, name: &str) -> String {
        let taken: Vec<String> = fs::read_dir(&self.folder)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| e.file_name().to_str().map(str::to_lowercase))
                    .collect()
            })
            .unwrap_or_default();
        let stem = Self::safe_file_name(name);
        let mut n = 1;
        loop {
            let file = if n == 1 {
                format!("{stem}.{FILE_EXTENSION}")
            } else {
                format!("{stem} {n}.{FILE_EXTENSION}")
            };
            if !taken.contains(&file.to_lowercase()) {
                return file;
            }
            n += 1;
        }
    }

    // --- Reading ----------------------------------------------------------------------------

    /// Reads the folder (every time: a preset made by `ym` shows up). A file that cannot be read is listed as damaged with the
    /// reason, never stops the others and is never rewritten; of two files with one name the first by file name is used and
    /// the other is damaged ("Another preset has this name."). A read takes no lock: a file is whole or absent.
    #[must_use]
    pub fn list(&self) -> PresetListing {
        let mut files: Vec<String> = fs::read_dir(&self.folder)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.metadata().is_ok_and(|m| m.is_file()))
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| is_preset_file_name(n))
                    .collect()
            })
            .unwrap_or_default();
        files.sort_by_cached_key(|n| (n.to_lowercase(), n.clone()));
        let mut listing = PresetListing::default();
        for file in files {
            let path = self.folder.join(&file);
            match read_file(&path) {
                Ok((preset, bytes)) => {
                    if listing
                        .presets
                        .iter()
                        .any(|s| same_name(&s.preset.name, &preset.name))
                    {
                        listing.damaged.push(DamagedPreset {
                            file,
                            reason: DUPLICATE_NAME_SENTENCE.to_owned(),
                        });
                    } else {
                        listing.presets.push(StoredPreset {
                            revision: revision_of(&bytes),
                            preset,
                            file,
                        });
                    }
                }
                Err(e) => listing.damaged.push(DamagedPreset {
                    file,
                    reason: reason_of(&e),
                }),
            }
        }
        listing
            .presets
            .sort_by_cached_key(|s| (s.preset.name.to_lowercase(), s.file.clone()));
        listing
    }

    /// The preset with this name (ignoring case).
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`].
    pub fn get(&self, name: &str) -> Result<Preset, PresetError> {
        self.get_stored(name).map(|s| s.preset)
    }

    /// The preset with this name (ignoring case), with its file and revision.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`].
    pub fn get_stored(&self, name: &str) -> Result<StoredPreset, PresetError> {
        self.list()
            .presets
            .into_iter()
            .find(|s| same_name(&s.preset.name, name))
            .ok_or_else(|| PresetError::NotFound(name.to_owned()))
    }

    /// The path of the preset's file.
    #[must_use]
    pub fn path_of(&self, name: &str) -> Option<PathBuf> {
        self.get_stored(name).ok().map(|s| self.folder.join(s.file))
    }

    /// Finds the preset an argument means (`docs/30` section 11): a path to a `.ympreset` file (an argument with a `/` or
    /// ending `.ympreset`) is read without importing it; a name matches ignoring case, exactly first and then as a prefix.
    ///
    /// # Errors
    ///
    /// [`PresetError::PathNotFound`] (exit 3) for a path that is not a file, [`PresetError::NotFound`] (exit 3) for a name that
    /// matches nothing, [`PresetError::Ambiguous`] (exit 2) for several matches, and the file's own errors.
    pub fn resolve(&self, query: &str) -> Result<Preset, PresetError> {
        let lower = query.to_lowercase();
        if query.contains('/') || lower.ends_with(&format!(".{FILE_EXTENSION}")) {
            let path = Path::new(query);
            if !path.is_file() {
                return Err(PresetError::PathNotFound(query.to_owned()));
            }
            return read_file(path).map(|(preset, _)| preset);
        }
        let presets = self.list().presets;
        if let Some(s) = presets
            .iter()
            .find(|s| s.preset.name.to_lowercase() == lower)
        {
            return Ok(s.preset.clone());
        }
        let mut found: Vec<&StoredPreset> = presets
            .iter()
            .filter(|s| s.preset.name.to_lowercase().starts_with(&lower))
            .collect();
        match found.len() {
            0 => Err(PresetError::NotFound(query.to_owned())),
            1 => Ok(found.remove(0).preset.clone()),
            _ => Err(PresetError::Ambiguous {
                query: query.to_owned(),
                names: found.iter().map(|s| s.preset.name.clone()).collect(),
            }),
        }
    }

    // --- Writing ----------------------------------------------------------------------------

    fn lock(&self) -> Result<LibraryLock, PresetError> {
        LibraryLock::acquire(&self.folder, self.lock_timeout).map_err(|e| match e {
            LockError::Timeout { .. } => PresetError::Busy,
            LockError::Io { source, .. } => self.write_error(source),
        })
    }

    fn write_error(&self, source: io::Error) -> PresetError {
        PresetError::Write {
            folder: self.folder.display().to_string(),
            source,
        }
    }

    fn write_file(&self, file: &str, text: &str) -> Result<(), PresetError> {
        (self.write)(&self.folder.join(file), text.as_bytes()).map_err(|e| self.write_error(e))
    }

    /// Writes `preset` under the lock. Without `replace` a name that exists (ignoring case) is refused; with it the file of that
    /// name is written over (an atomic rename: the old file is not kept). Returns what is stored now.
    ///
    /// # Errors
    ///
    /// [`PresetError::Exists`] ("A preset called Warm evening already exists."), [`PresetError::Write`] ("Yellowmoon couldn't
    /// save the preset to {folder}. … Nothing was changed."), [`PresetError::Busy`], [`PresetError::TooLarge`].
    pub fn save(&self, preset: &Preset, replace: bool) -> Result<StoredPreset, PresetError> {
        let text = preset.to_json()?;
        let _lock = self.lock()?;
        self.save_locked(preset, &text, replace).map(|(s, _)| s)
    }

    /// [`PresetStore::save`] with the lock held; also says whether a preset was replaced.
    fn save_locked(
        &self,
        preset: &Preset,
        text: &str,
        replace: bool,
    ) -> Result<(StoredPreset, bool), PresetError> {
        let existing = self
            .list()
            .presets
            .into_iter()
            .find(|s| same_name(&s.preset.name, &preset.name));
        let (file, replaced) = match existing {
            Some(_) if !replace => return Err(PresetError::Exists(preset.name.clone())),
            Some(s) => (s.file, true),
            None => (self.new_file_name(&preset.name), false),
        };
        self.write_file(&file, text)?;
        Ok((
            StoredPreset {
                preset: preset.clone(),
                revision: revision_of(text.as_bytes()),
                file,
            },
            replaced,
        ))
    }

    /// Gives the preset `name` the name `new_name`: the new file is written and then the old one removed, under one lock. A name
    /// that another preset has (ignoring case) is refused; a change of case alone is allowed.
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Exists`], [`PresetError::InvalidName`], [`PresetError::Write`].
    pub fn rename(&self, name: &str, new_name: &str) -> Result<StoredPreset, PresetError> {
        let new_name = super::normalise_name(new_name)?;
        if !self.folder.is_dir() {
            return Err(PresetError::NotFound(name.to_owned()));
        }
        let _lock = self.lock()?;
        let listing = self.list();
        let current = listing
            .presets
            .iter()
            .find(|s| same_name(&s.preset.name, name))
            .ok_or_else(|| PresetError::NotFound(name.to_owned()))?;
        if listing
            .presets
            .iter()
            .any(|s| s.file != current.file && same_name(&s.preset.name, &new_name))
        {
            return Err(PresetError::Exists(new_name));
        }
        let mut preset = current.preset.clone();
        preset.name = new_name;
        let text = preset.to_json()?;
        // The same file name (a change of case, or a name that is made safe to the same stem) is written over.
        let stem = Self::safe_file_name(&preset.name).to_lowercase();
        let file = if stem == stem_of(&current.file).to_lowercase() {
            current.file.clone()
        } else {
            self.new_file_name(&preset.name)
        };
        self.write_file(&file, &text)?;
        if file != current.file {
            // The new file is whole before the old one goes; a failure to remove it leaves two files, and the list says so.
            let _ = fs::remove_file(self.folder.join(&current.file));
        }
        Ok(StoredPreset {
            revision: revision_of(text.as_bytes()),
            preset,
            file,
        })
    }

    /// Moves the preset's file to the Trash (the person can take it back).
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Trash`] (a machine without a Trash: the file stays).
    pub fn remove(&self, name: &str) -> Result<(), PresetError> {
        if !self.folder.is_dir() {
            return Err(PresetError::NotFound(name.to_owned()));
        }
        let _lock = self.lock()?;
        let stored = self.get_stored(name)?;
        (self.trash)(&self.folder.join(stored.file))?;
        Ok(())
    }

    /// Moves a damaged file of the folder to the Trash, by its file name (`presets_list`'s `damaged`).
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`] for a name that is not a preset file of the folder, [`PresetError::Trash`].
    pub fn remove_file(&self, file: &str) -> Result<(), PresetError> {
        if file.contains(['/', '\\']) || !is_preset_file_name(file) || !self.folder.is_dir() {
            return Err(PresetError::NotFound(file.to_owned()));
        }
        let _lock = self.lock()?;
        let path = self.folder.join(file);
        if !path.is_file() {
            return Err(PresetError::NotFound(file.to_owned()));
        }
        (self.trash)(&path)?;
        Ok(())
    }

    /// Reads a `.ympreset` file into the folder. A file the core refuses is `refused` with its sentence; a name that exists
    /// (ignoring case) is not replaced unless `replace` and is listed in `exists`, so that the page can ask. The answer has the
    /// one entry for this file.
    #[must_use]
    pub fn import(&self, path: &Path, replace: bool) -> ImportOutcome {
        let file = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let mut outcome = ImportOutcome::default();
        let (preset, _) = match read_file(path) {
            Ok(read) => read,
            Err(e) => {
                outcome.refused.push(ImportRefused {
                    file,
                    reason: reason_of(&e),
                });
                return outcome;
            }
        };
        let result = preset.to_json().and_then(|text| {
            let _lock = self.lock()?;
            self.save_locked(&preset, &text, replace)
        });
        match result {
            Ok((_, true)) => outcome.replaced.push(preset.name),
            Ok((_, false)) => outcome.added.push(preset.name),
            Err(PresetError::Exists(name)) => outcome.exists.push(ImportExists {
                name,
                path: path.display().to_string(),
            }),
            Err(e) => outcome.refused.push(ImportRefused {
                file,
                reason: e.to_string(),
            }),
        }
        outcome
    }

    /// Writes the preset's file to `dest` (atomically; an existing destination is overwritten, like `cp`).
    ///
    /// # Errors
    ///
    /// [`PresetError::NotFound`], [`PresetError::Write`].
    pub fn export(&self, name: &str, dest: &Path) -> Result<(), PresetError> {
        let text = self.get(name)?.to_json()?;
        (self.write)(dest, text.as_bytes()).map_err(|source| PresetError::Write {
            folder: dest
                .parent()
                .map_or_else(|| dest.display().to_string(), |p| p.display().to_string()),
            source,
        })
    }
}

/// Reads and parses one preset file: its preset and its bytes.
fn read_file(path: &Path) -> Result<(Preset, Vec<u8>), PresetError> {
    let read_error = |source| PresetError::Read {
        file: path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        source,
    };
    let meta = fs::metadata(path).map_err(read_error)?;
    if meta.len() > MAX_PRESET_BYTES as u64 {
        return Err(PresetError::TooLarge);
    }
    let bytes = fs::read(path).map_err(read_error)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| PresetError::NotAPreset)?;
    Ok((Preset::from_json(text)?, bytes))
}

/// The closed list of reasons a file is damaged (`docs/30` section 6).
fn reason_of(e: &PresetError) -> String {
    match e {
        PresetError::Read { .. } => COULDNT_READ_SENTENCE.to_owned(),
        PresetError::Settings(_) | PresetError::NotAPreset => NOT_A_PRESET_SENTENCE.to_owned(),
        other => other.to_string(),
    }
}
