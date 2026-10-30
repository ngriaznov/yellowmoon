//! `preferences.json` under the app data folder: the default look, the export defaults and the
//! other persisted choices. This list is exhaustive (`docs/01-architecture.md`
//! section Persistence, hard rule 4 of `AGENTS.md`): nothing about a photo is ever saved.
//!
//! # Where
//!
//! `<app folder>/preferences.json`, where the app folder is `~/Library/Application Support/app.yellowmoon`
//! on a Mac and the equivalent `ym_looks::library::default_root()` elsewhere (`$XDG_DATA_HOME/app.yellowmoon`
//! on Linux); `YELLOWMOON_HOME` replaces it. The look library lives in the same folder, so the app
//! and the CLI share both.
//!
//! # What the file does when it is wrong
//!
//! Reading never fails and never crashes the app:
//!
//! * a missing file is the defaults (and is not created until something is changed);
//! * a field that is missing takes its default; a field this version does not know is ignored,
//!   without a warning (a file written by an earlier version may hold `display_rendering`, the
//!   choice of ARRI's display LUT that Yellowmoon no longer offers: it is read as nothing and left
//!   out of the file the next time a preference changes);
//! * a field with a value that does not fit (a JPEG quality of 12, a look id that is not an id) takes
//!   its default and the rest of the file is kept;
//! * a file that is not JSON, or not an object, is all defaults.
//!
//! Each of these logs a warning (`tracing`) and is kept in [`PreferencesStore::warnings`]. The file
//! is replaced on the next change.
//!
//! # Writing
//!
//! Atomically: a temporary file in the same folder, `fsync`, rename, `fsync` of the folder
//! ([`ym_looks::manifest::atomic_write`]). A change re-reads the file first, so the app and the CLI
//! changing different fields at about the same time do not undo each other.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::apple_auto::{DEFAULT_INSTRUCTIONS, check_instructions};
use crate::auto_provider::{AutoConsent, AutoProvider};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use ym_export::ExportOptions;
use ym_looks::library::{LibraryError, LookId, default_root};
use ym_looks::manifest::atomic_write;

/// The name of the file inside the app folder.
pub const FILE_NAME: &str = "preferences.json";

/// The id of the look new photos get out of the box: Signature (owner's decision of 2026-10-14,
/// `docs/08-looks.md`). `ym_looks::library::LookId::factory_default` is the same id as a value.
pub const DEFAULT_LOOK: &str = "builtin:signature";

/// The id of Neutral, the identity look. It exists in code whatever the data folder holds, so it is
/// the look to fall back on when the default look cannot be used.
pub const NEUTRAL_LOOK: &str = "builtin:neutral";

/// Folders remembered between runs (`LastUsedFolders` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LastUsedFolders {
    /// Where Open files or Open folder last looked.
    pub open: Option<PathBuf>,
    /// Where an export last saved to.
    pub export: Option<PathBuf>,
    /// Where the look import last looked.
    pub looks_import: Option<PathBuf>,
}

/// Everything that persists besides imported looks (`Preferences` in `src/api/types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    /// The look new photos get ([`DEFAULT_LOOK`], Signature, out of the box).
    pub default_look: LookId,
    /// Format, quality, bits, compression, colour space, size, destination, name template and
    /// what to do when the file exists.
    pub export_defaults: ExportOptions,
    /// "Remove location from exports".
    pub remove_location_from_exports: bool,
    /// "Check for updates automatically" (`docs/12`).
    pub check_for_updates_automatically: bool,
    /// "Watch Downloads during import" (`docs/08`, wizard step 2).
    pub watch_downloads_during_import: bool,
    /// The folders used last.
    pub last_used_folders: LastUsedFolders,
    /// "Auto grading" (`docs/22` section 10.1): who grades each photo's Auto, Apple Intelligence by
    /// default. A provider that is not available (or not agreed to, [`Preferences::auto_consent`]) at
    /// run time acts as Default Auto, and the choice stays saved. A file from before
    /// providers has `apple_auto` instead, which reads as this field (`false` is `yellowmoon`) and is
    /// never written again.
    pub auto_provider: AutoProvider,
    /// Whether the person agreed to send photos through Claude Code and through Antigravity
    /// (`docs/22` section 10.2). Both false by default; saved together with the choice that needs
    /// it, and a patch cannot choose a cloud provider without it.
    pub auto_consent: AutoConsent,
    /// "Instructions" for the Auto grading (`docs/19` section 2, one text for every provider,
    /// `docs/22` D1): the person's own instructions for the model, at most
    /// [`crate::apple_auto::INSTRUCTIONS_MAX_CHARS`] characters; `""` (absent, empty or only
    /// white space) is the default, [`crate::apple_auto::DEFAULT_INSTRUCTIONS`]. Saved as `""` when
    /// it equals the default.
    pub apple_auto_instructions: String,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            default_look: LookId::factory_default(),
            export_defaults: ExportOptions::default(),
            remove_location_from_exports: false,
            check_for_updates_automatically: true,
            watch_downloads_during_import: false,
            last_used_folders: LastUsedFolders::default(),
            auto_provider: AutoProvider::Apple,
            auto_consent: AutoConsent::default(),
            apple_auto_instructions: String::new(),
        }
    }
}

/// A partial update of the preferences (`PreferencesPatch` in `src/api/types.ts`: every field
/// replaces the whole field it names; an absent field is left alone). A field this version does not
/// have is an error, so a typo is not a silent no-op.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PreferencesPatch {
    /// Replaces `default_look`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_look: Option<LookId>,
    /// Replaces `export_defaults`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_defaults: Option<ExportOptions>,
    /// Replaces `remove_location_from_exports`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remove_location_from_exports: Option<bool>,
    /// Replaces `check_for_updates_automatically`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub check_for_updates_automatically: Option<bool>,
    /// Replaces `watch_downloads_during_import`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watch_downloads_during_import: Option<bool>,
    /// Replaces `last_used_folders`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_folders: Option<LastUsedFolders>,
    /// Replaces `auto_provider`. Choosing Claude Code or Antigravity needs its `auto_consent` to be
    /// true after the patch ([`PreferencesPatch::check_against`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_provider: Option<AutoProvider>,
    /// Replaces the whole `auto_consent`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_consent: Option<AutoConsent>,
    /// Replaces `apple_auto_instructions` (`""` or the default text: the default). Checked by
    /// [`PreferencesPatch::check`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apple_auto_instructions: Option<String>,
}

impl PreferencesPatch {
    /// Applies the patch to `prefs`; true when something changed.
    pub fn apply(&self, prefs: &mut Preferences) -> bool {
        let before = prefs.clone();
        if let Some(v) = &self.default_look {
            prefs.default_look = v.clone();
        }
        if let Some(v) = &self.export_defaults {
            prefs.export_defaults = v.clone();
        }
        if let Some(v) = self.remove_location_from_exports {
            prefs.remove_location_from_exports = v;
        }
        if let Some(v) = self.check_for_updates_automatically {
            prefs.check_for_updates_automatically = v;
        }
        if let Some(v) = self.watch_downloads_during_import {
            prefs.watch_downloads_during_import = v;
        }
        if let Some(v) = &self.last_used_folders {
            prefs.last_used_folders = v.clone();
        }
        if let Some(v) = self.auto_provider {
            prefs.auto_provider = v;
        }
        if let Some(v) = self.auto_consent {
            prefs.auto_consent = v;
        }
        if let Some(v) = &self.apple_auto_instructions {
            prefs.apple_auto_instructions = own_instructions(v);
        }
        *prefs != before
    }

    /// Checks the values the core refuses rather than saves, those that can be judged from the patch
    /// alone: instructions longer than [`crate::apple_auto::INSTRUCTIONS_MAX_CHARS`] characters. The
    /// consent rule needs the saved preferences too ([`PreferencesPatch::check_against`]).
    ///
    /// # Errors
    ///
    /// [`PreferencesError::Invalid`] with the message to show.
    pub fn check(&self) -> Result<(), PreferencesError> {
        if let Some(text) = &self.apple_auto_instructions {
            check_instructions(text).map_err(PreferencesError::Invalid)?;
        }
        Ok(())
    }

    /// [`PreferencesPatch::check`] and the consent rule of `docs/22` section 10.2: a patch that makes
    /// `auto_provider` Claude Code or Antigravity is refused when `auto_consent` for it is not true
    /// after the patch is applied to `current` ("Say yes to sending photos to Claude Code first.").
    /// Choosing Default Auto or Apple Intelligence is always allowed, available or not.
    /// [`PreferencesStore::update`] calls this against the file's current content.
    ///
    /// # Errors
    ///
    /// [`PreferencesError::Invalid`] with the message to show.
    pub fn check_against(&self, current: &Preferences) -> Result<(), PreferencesError> {
        self.check()?;
        let Some(provider) = self.auto_provider.filter(|p| p.is_cloud()) else {
            return Ok(());
        };
        let consent = self.auto_consent.unwrap_or(current.auto_consent);
        if consent.get(provider) {
            Ok(())
        } else {
            Err(PreferencesError::Invalid(format!(
                "Say yes to sending photos to {} first.",
                provider.name()
            )))
        }
    }

    /// A patch that sets only the default look.
    #[must_use]
    pub fn default_look(id: LookId) -> Self {
        Self {
            default_look: Some(id),
            ..Self::default()
        }
    }
}

/// The person's own instructions as they are saved: `""` for text that is empty, only white space or
/// the default, else the text as given.
#[must_use]
pub fn own_instructions(text: &str) -> String {
    if text.trim().is_empty() || text == DEFAULT_INSTRUCTIONS {
        String::new()
    } else {
        text.to_owned()
    }
}

/// Why preferences could not be changed or saved. The text is plain words with a next step.
#[derive(Debug, thiserror::Error)]
pub enum PreferencesError {
    /// No home folder, so no place for the file.
    #[error(transparent)]
    NoHome(#[from] LibraryError),
    /// The file could not be written.
    #[error(
        "Couldn't save your preferences to {}: {source}. Check that the folder is writable and has free space.",
        path.display()
    )]
    Write {
        /// The file.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// The preferences cannot be written as JSON (a folder name that is not valid text).
    #[error("Couldn't save your preferences: {0}")]
    Encode(String),
    /// A value that cannot be used; nothing was changed.
    #[error("{0}")]
    Invalid(String),
}

impl PreferencesError {
    /// A short machine-readable kind, for the log: its text names the preferences file, whose path
    /// holds the user's name.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NoHome(_) => "no_home",
            Self::Write { .. } => "write",
            Self::Encode(_) => "encode",
            Self::Invalid(_) => "invalid",
        }
    }
}

/// A preferences file after reading: the values and what was wrong with the file.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedPreferences {
    /// The preferences, with the default for everything that was missing or unusable.
    pub preferences: Preferences,
    /// One plain-words line for every problem found. Empty for a good file.
    pub warnings: Vec<String>,
}

/// Reads one field of the top-level object, leaving it out (default) with a warning when its value
/// does not fit.
fn take<T: DeserializeOwned>(
    map: &mut Map<String, Value>,
    key: &str,
    warnings: &mut Vec<String>,
) -> Option<T> {
    let value = map.remove(key)?;
    match serde_json::from_value::<T>(value) {
        Ok(v) => Some(v),
        Err(e) => {
            warnings.push(format!(
                "Preference \"{key}\" has a value that can't be used ({e}), so its default is used."
            ));
            None
        }
    }
}

/// Reads a struct whose fields all have defaults, field by field: the fields that fit are kept and
/// each one that does not takes its default, with a warning.
fn take_struct<T>(map: &mut Map<String, Value>, key: &str, warnings: &mut Vec<String>) -> Option<T>
where
    T: Default + Serialize + DeserializeOwned,
{
    let value = map.remove(key)?;
    let Value::Object(fields) = value else {
        warnings.push(format!(
            "Preference \"{key}\" is not a group of settings, so its defaults are used."
        ));
        return None;
    };
    if let Ok(whole) = serde_json::from_value::<T>(Value::Object(fields.clone())) {
        return Some(whole);
    }
    // Something in it does not fit: put the fields on top of the defaults one by one.
    let Ok(Value::Object(mut merged)) = serde_json::to_value(T::default()) else {
        return Some(T::default());
    };
    for (name, value) in fields {
        let Some(slot) = merged.get_mut(&name) else {
            continue; // a field this version does not have
        };
        let before = std::mem::replace(slot, value);
        if serde_json::from_value::<T>(Value::Object(merged.clone())).is_err() {
            warnings.push(format!(
                "Preference \"{key}.{name}\" has a value that can't be used, so its default is used."
            ));
            merged.insert(name, before);
        }
    }
    Some(serde_json::from_value(Value::Object(merged)).unwrap_or_default())
}

/// Reads `auto_consent` (`docs/22` section 10.1): a missing key is false, keys of other names are
/// dropped, a value that is not a boolean is false with a warning, and a value that is not an object
/// is all false with a warning.
fn parse_consent(value: Value, warnings: &mut Vec<String>) -> AutoConsent {
    let Value::Object(fields) = value else {
        warnings.push(
            "Preference \"auto_consent\" is not a group of settings, so no consent is used."
                .to_owned(),
        );
        return AutoConsent::default();
    };
    let mut read = |name: &str| match fields.get(name) {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => {
            warnings.push(format!(
                "Preference \"auto_consent.{name}\" has a value that can't be used, so its default is used."
            ));
            false
        }
    };
    AutoConsent {
        claude: read("claude"),
        agy: read("agy"),
    }
}

/// Reads the text of a preferences file (see the module documentation for what is forgiven).
#[must_use]
pub fn parse(text: &str) -> ParsedPreferences {
    let mut warnings = Vec::new();
    let value: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            warnings.push(format!(
                "The preferences file is damaged ({e}), so the defaults are used. It is replaced the next time a preference changes."
            ));
            return ParsedPreferences {
                preferences: Preferences::default(),
                warnings,
            };
        }
    };
    let Value::Object(mut map) = value else {
        warnings.push(
            "The preferences file is not a set of preferences, so the defaults are used. It is replaced the next time a preference changes."
                .to_owned(),
        );
        return ParsedPreferences {
            preferences: Preferences::default(),
            warnings,
        };
    };
    let mut p = Preferences::default();
    if let Some(v) = take(&mut map, "default_look", &mut warnings) {
        p.default_look = v;
    }
    if let Some(v) = take_struct(&mut map, "export_defaults", &mut warnings) {
        p.export_defaults = v;
    }
    if let Some(v) = take(&mut map, "remove_location_from_exports", &mut warnings) {
        p.remove_location_from_exports = v;
    }
    if let Some(v) = take(&mut map, "check_for_updates_automatically", &mut warnings) {
        p.check_for_updates_automatically = v;
    }
    if let Some(v) = take(&mut map, "watch_downloads_during_import", &mut warnings) {
        p.watch_downloads_during_import = v;
    }
    // The provider: a valid `auto_provider` wins; a file from before providers has `apple_auto`
    // (`false` is Default Auto, `true` or absent is Apple); `apple_auto` is never written
    // again (`docs/22` section 10.1, decision D15).
    let legacy = map.remove("apple_auto");
    let provider = match map.remove("auto_provider") {
        None => None,
        Some(value) => match serde_json::from_value::<AutoProvider>(value) {
            Ok(v) => Some(v),
            Err(_) => {
                warnings.push(
                    "Preference \"auto_provider\" can't be used, so the default is used."
                        .to_owned(),
                );
                None
            }
        },
    };
    p.auto_provider = match (provider, legacy) {
        (Some(v), _) => v,
        (None, Some(Value::Bool(false))) => AutoProvider::Yellowmoon,
        (None, Some(Value::Bool(true)) | None) => AutoProvider::Apple,
        (None, Some(_)) => {
            warnings.push(
                "Preference \"apple_auto\" has a value that can't be used, so its default is used."
                    .to_owned(),
            );
            AutoProvider::Apple
        }
    };
    if let Some(v) = map.remove("auto_consent") {
        p.auto_consent = parse_consent(v, &mut warnings);
    }
    if let Some(v) = take::<String>(&mut map, "apple_auto_instructions", &mut warnings) {
        // Longer than the model takes (a file edited by hand): the default, with a warning.
        match check_instructions(&v) {
            Ok(()) => p.apple_auto_instructions = own_instructions(&v),
            Err(e) => warnings.push(format!(
                "Preference \"apple_auto_instructions\" can't be used ({e}), so the default instructions are used."
            )),
        }
    }
    if let Some(v) = take_struct(&mut map, "last_used_folders", &mut warnings) {
        p.last_used_folders = v;
    }
    // A long edge outside what the writers accept would make every export fail: keep the rest.
    if let Err(e) = p.export_defaults.validate() {
        warnings.push(format!(
            "The saved export size can't be used ({e}), so the default size is used."
        ));
        p.export_defaults.size = ExportOptions::default().size;
    }
    ParsedPreferences {
        preferences: p,
        warnings,
    }
}

/// The preferences file of one app folder, in memory and on disk.
///
/// Cheap to share (`Arc<PreferencesStore>`); every method takes `&self`.
#[derive(Debug)]
pub struct PreferencesStore {
    path: PathBuf,
    state: Mutex<Loaded>,
}

#[derive(Debug)]
struct Loaded {
    prefs: Preferences,
    warnings: Vec<String>,
}

fn lock(m: &Mutex<Loaded>) -> MutexGuard<'_, Loaded> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads `path`: the defaults when the file is not there.
fn read_file(path: &Path) -> ParsedPreferences {
    match std::fs::read(path) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => parse(&text),
            Err(_) => ParsedPreferences {
                preferences: Preferences::default(),
                warnings: vec![
                    "The preferences file is damaged (it is not text), so the defaults are used. It is replaced the next time a preference changes."
                        .to_owned(),
                ],
            },
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ParsedPreferences {
            preferences: Preferences::default(),
            warnings: Vec::new(),
        },
        Err(e) => ParsedPreferences {
            preferences: Preferences::default(),
            warnings: vec![format!(
                "The preferences file could not be read ({e}), so the defaults are used."
            )],
        },
    }
}

/// Logs what was wrong with the file. The path holds the user's name and a bug report may carry the
/// log (`docs/01`), so it goes at debug level only.
fn log_warnings(path: &Path, warnings: &[String]) {
    for w in warnings {
        tracing::warn!("{w}");
        tracing::debug!(file = %path.display(), "the preferences file in question");
    }
}

impl PreferencesStore {
    /// Opens the preferences of the app folder `root` (see [`default_root`]). Reads the file now;
    /// nothing is created on disk.
    #[must_use]
    pub fn open(root: &Path) -> Self {
        let path = root.join(FILE_NAME);
        let parsed = read_file(&path);
        log_warnings(&path, &parsed.warnings);
        Self {
            path,
            state: Mutex::new(Loaded {
                prefs: parsed.preferences,
                warnings: parsed.warnings,
            }),
        }
    }

    /// Opens the preferences of this machine: `~/Library/Application Support/app.yellowmoon` on a
    /// Mac, `YELLOWMOON_HOME` when it is set.
    ///
    /// # Errors
    ///
    /// [`PreferencesError::NoHome`] when there is no home folder.
    pub fn open_default() -> Result<Self, PreferencesError> {
        Ok(Self::open(&default_root()?))
    }

    /// The preferences file (it may not exist yet).
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The preferences now.
    #[must_use]
    pub fn get(&self) -> Preferences {
        lock(&self.state).prefs.clone()
    }

    /// What was wrong with the file when it was last read, in plain words.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        lock(&self.state).warnings.clone()
    }

    /// Reads the file again (another process, such as the CLI, may have changed it). Returns the
    /// preferences.
    pub fn reload(&self) -> Preferences {
        let parsed = read_file(&self.path);
        let mut st = lock(&self.state);
        // A damaged file is reported once, not at every look at it.
        if parsed.warnings != st.warnings {
            log_warnings(&self.path, &parsed.warnings);
        }
        st.prefs = parsed.preferences;
        st.warnings = parsed.warnings;
        st.prefs.clone()
    }

    /// `preferences_set`: applies a patch, saves the file and returns the new preferences. A patch
    /// that changes nothing writes nothing.
    ///
    /// # Errors
    ///
    /// [`PreferencesError::Invalid`] for export defaults the writers cannot use, for too long
    /// instructions and for a cloud provider without its consent,
    /// [`PreferencesError::Write`] or [`PreferencesError::Encode`] when the file cannot be saved.
    /// Nothing changes on an error.
    pub fn update(&self, patch: &PreferencesPatch) -> Result<Preferences, PreferencesError> {
        patch.check()?;
        self.modify(|prefs| {
            // The consent rule reads the file's current consent, which a patch without
            // `auto_consent` leaves as it is.
            patch.check_against(prefs)?;
            patch.apply(prefs);
            Ok(())
        })
    }

    /// Changes the preferences with `change` (which sees the file's current content, not the
    /// copy this process read at start) and saves them.
    ///
    /// # Errors
    ///
    /// As [`PreferencesStore::update`], and whatever `change` returns.
    pub fn modify(
        &self,
        change: impl FnOnce(&mut Preferences) -> Result<(), PreferencesError>,
    ) -> Result<Preferences, PreferencesError> {
        let mut st = lock(&self.state);
        // Start from the file, so a change made by another process meanwhile is kept.
        let on_disk = read_file(&self.path);
        let mut next = if self.path.exists() {
            on_disk.preferences
        } else {
            st.prefs.clone()
        };
        let before = next.clone();
        change(&mut next)?;
        next.export_defaults.validate().map_err(|e| {
            PreferencesError::Invalid(format!("The export options can't be used: {e}"))
        })?;
        if next == before && on_disk.warnings.is_empty() {
            // Nothing to write; take what the file says in case another process changed it.
            st.prefs = next.clone();
            return Ok(next);
        }
        self.write(&next)?;
        st.prefs = next.clone();
        st.warnings.clear();
        Ok(next)
    }

    fn write(&self, prefs: &Preferences) -> Result<(), PreferencesError> {
        let mut text = serde_json::to_string_pretty(prefs)
            .map_err(|e| PreferencesError::Encode(e.to_string()))?;
        text.push('\n');
        let io_error = |source| PreferencesError::Write {
            path: self.path.clone(),
            source,
        };
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(io_error)?;
        }
        atomic_write(&self.path, text.as_bytes()).map_err(io_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_export::{BitDepth, ColorSpace, FormatKind, IfExists, Size};

    fn root() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp dir")
    }

    #[test]
    fn defaults_are_the_documented_ones() {
        let p = Preferences::default();
        assert_eq!(p.default_look.as_str(), DEFAULT_LOOK);
        assert_eq!(DEFAULT_LOOK, "builtin:signature");
        assert_eq!(NEUTRAL_LOOK, "builtin:neutral");
        assert_eq!(p.export_defaults, ExportOptions::default());
        assert!(!p.remove_location_from_exports);
        assert!(p.check_for_updates_automatically);
        assert!(!p.watch_downloads_during_import);
        assert_eq!(p.last_used_folders, LastUsedFolders::default());
        assert_eq!(
            p.auto_provider,
            AutoProvider::Apple,
            "Apple Intelligence is the default where it is available (docs/22 O3)"
        );
        assert_eq!(p.auto_consent, AutoConsent::default());
        assert_eq!(p.apple_auto_instructions, "");
    }

    #[test]
    fn the_provider_migrates_from_the_old_switch() {
        let provider = |text: &str| parse(text).preferences.auto_provider;
        // Nothing, or the old switch on: Apple Intelligence. The old switch off: Default Auto.
        assert_eq!(provider("{}"), AutoProvider::Apple);
        assert_eq!(
            provider(r#"{ "remove_location_from_exports": true }"#),
            AutoProvider::Apple
        );
        assert_eq!(provider(r#"{ "apple_auto": true }"#), AutoProvider::Apple);
        assert_eq!(
            provider(r#"{ "apple_auto": false }"#),
            AutoProvider::Yellowmoon
        );
        // A valid `auto_provider` wins over the old switch, which is ignored.
        for (text, want) in [
            (
                r#"{ "auto_provider": "claude", "apple_auto": false }"#,
                AutoProvider::Claude,
            ),
            (r#"{ "auto_provider": "agy" }"#, AutoProvider::Agy),
            (
                r#"{ "auto_provider": "yellowmoon", "apple_auto": true }"#,
                AutoProvider::Yellowmoon,
            ),
            (
                r#"{ "auto_provider": "apple", "apple_auto": "yes" }"#,
                AutoProvider::Apple,
            ),
        ] {
            let parsed = parse(text);
            assert_eq!(parsed.preferences.auto_provider, want, "{text}");
            assert!(parsed.warnings.is_empty(), "{text}: {:?}", parsed.warnings);
        }
        // A name that is not a provider is read as absent, with the warning; then the old switch decides.
        let bad = parse(
            r#"{ "auto_provider": "gemini", "apple_auto": false, "remove_location_from_exports": true }"#,
        );
        assert_eq!(bad.preferences.auto_provider, AutoProvider::Yellowmoon);
        assert!(bad.preferences.remove_location_from_exports);
        assert_eq!(
            bad.warnings,
            ["Preference \"auto_provider\" can't be used, so the default is used."]
        );
        let worse = parse(r#"{ "auto_provider": 7 }"#);
        assert_eq!(worse.preferences.auto_provider, AutoProvider::Apple);
        assert_eq!(worse.warnings.len(), 1);
        // An old switch that is not a switch costs only that value.
        let odd = parse(r#"{ "apple_auto": "yes", "remove_location_from_exports": true }"#);
        assert!(odd.preferences.remove_location_from_exports);
        assert_eq!(odd.preferences.auto_provider, AutoProvider::Apple);
        assert_eq!(odd.warnings.len(), 1);
        // The next save writes `auto_provider` and `auto_consent` and drops `apple_auto`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(FILE_NAME),
            r#"{ "apple_auto": false, "check_for_updates_automatically": false }"#,
        )
        .unwrap();
        let store = PreferencesStore::open(dir.path());
        assert_eq!(store.get().auto_provider, AutoProvider::Yellowmoon);
        store
            .update(&PreferencesPatch {
                watch_downloads_during_import: Some(true),
                ..PreferencesPatch::default()
            })
            .unwrap();
        let saved: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(FILE_NAME)).unwrap())
                .unwrap();
        assert!(saved.get("apple_auto").is_none(), "{saved}");
        assert_eq!(saved["auto_provider"], "yellowmoon");
        assert_eq!(
            saved["auto_consent"],
            serde_json::json!({ "claude": false, "agy": false })
        );
        assert_eq!(saved["check_for_updates_automatically"], false);
        assert_eq!(
            PreferencesStore::open(dir.path()).get().auto_provider,
            AutoProvider::Yellowmoon
        );
    }

    #[test]
    fn consent_reads_forgivingly() {
        let consent = |text: &str| parse(text).preferences.auto_consent;
        assert_eq!(consent("{}"), AutoConsent::default());
        assert_eq!(
            consent(r#"{ "auto_consent": { "claude": true, "agy": true } }"#),
            AutoConsent {
                claude: true,
                agy: true
            }
        );
        // A missing key is false; keys of other names are dropped.
        let parsed = parse(r#"{ "auto_consent": { "claude": true, "gemini": true } }"#);
        assert_eq!(
            parsed.preferences.auto_consent,
            AutoConsent {
                claude: true,
                agy: false
            }
        );
        assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
        // A value that is not a boolean is false, with a warning; the other key is kept.
        let parsed = parse(r#"{ "auto_consent": { "claude": "yes", "agy": true } }"#);
        assert_eq!(
            parsed.preferences.auto_consent,
            AutoConsent {
                claude: false,
                agy: true
            }
        );
        assert_eq!(parsed.warnings.len(), 1, "{:?}", parsed.warnings);
        // A value that is not a group is all false, with a warning.
        let parsed = parse(r#"{ "auto_consent": true, "remove_location_from_exports": true }"#);
        assert_eq!(parsed.preferences.auto_consent, AutoConsent::default());
        assert!(parsed.preferences.remove_location_from_exports);
        assert_eq!(parsed.warnings.len(), 1);
        // A file with a cloud choice and no consent is kept as it is.
        let kept = parse(r#"{ "auto_provider": "claude" }"#);
        assert_eq!(kept.preferences.auto_provider, AutoProvider::Claude);
        assert_eq!(kept.preferences.auto_consent, AutoConsent::default());
        assert!(kept.warnings.is_empty());
    }

    #[test]
    fn a_patch_cannot_choose_a_cloud_provider_without_its_consent() {
        let dir = tempfile::tempdir().unwrap();
        let store = PreferencesStore::open(dir.path());
        let choose = |p: AutoProvider| PreferencesPatch {
            auto_provider: Some(p),
            ..PreferencesPatch::default()
        };
        // Refused: no consent saved, none in the patch; nothing is written.
        for (p, name) in [
            (AutoProvider::Claude, "Claude Code"),
            (AutoProvider::Agy, "Antigravity"),
        ] {
            let err = store.update(&choose(p)).expect_err("refused");
            assert!(
                matches!(&err, PreferencesError::Invalid(m) if *m == format!("Say yes to sending photos to {name} first.")),
                "{err}"
            );
        }
        assert!(!dir.path().join(FILE_NAME).exists());
        // Refused too when the patch's consent is for the other program.
        let wrong = PreferencesPatch {
            auto_provider: Some(AutoProvider::Claude),
            auto_consent: Some(AutoConsent {
                claude: false,
                agy: true,
            }),
            ..PreferencesPatch::default()
        };
        assert!(store.update(&wrong).is_err());
        // Accepted with its consent in the same patch: one atomic save.
        let both = PreferencesPatch {
            auto_provider: Some(AutoProvider::Claude),
            auto_consent: Some(AutoConsent {
                claude: true,
                agy: false,
            }),
            ..PreferencesPatch::default()
        };
        let saved = store.update(&both).unwrap();
        assert_eq!(saved.auto_provider, AutoProvider::Claude);
        assert_eq!(
            saved.auto_consent,
            AutoConsent {
                claude: true,
                agy: false
            }
        );
        // With the consent saved, a patch that names only the provider is accepted at once.
        store.update(&choose(AutoProvider::Yellowmoon)).unwrap();
        assert_eq!(
            store
                .update(&choose(AutoProvider::Claude))
                .unwrap()
                .auto_provider,
            AutoProvider::Claude
        );
        // Antigravity is still refused: its consent was never given.
        assert!(store.update(&choose(AutoProvider::Agy)).is_err());
        assert_eq!(store.get().auto_provider, AutoProvider::Claude);
        // Default Auto and Apple Intelligence are always allowed, consent or not.
        let fresh = PreferencesStore::open(tempfile::tempdir().unwrap().path());
        for p in [AutoProvider::Yellowmoon, AutoProvider::Apple] {
            assert!(fresh.update(&choose(p)).is_ok(), "{p:?}");
        }
        // The patch alone is judged by `check_against`, not by `check`.
        assert!(choose(AutoProvider::Agy).check().is_ok());
        assert!(
            choose(AutoProvider::Agy)
                .check_against(&Preferences::default())
                .is_err()
        );
        // A patch with the old switch is an error, so the webview and `ym` never send it.
        assert!(serde_json::from_str::<PreferencesPatch>(r#"{ "apple_auto": false }"#).is_err());
        let patch: PreferencesPatch = serde_json::from_str(
            r#"{ "auto_provider": "agy", "auto_consent": { "claude": false, "agy": true } }"#,
        )
        .unwrap();
        assert!(patch.check_against(&Preferences::default()).is_ok());
    }

    #[test]
    fn apple_auto_instructions_are_the_default_unless_the_person_wrote_their_own() {
        use crate::apple_auto::DEFAULT_INSTRUCTIONS;
        // Absent, empty or white space: the default, saved as "".
        for text in [
            r#"{}"#,
            r#"{ "apple_auto_instructions": "" }"#,
            r#"{ "apple_auto_instructions": "  \n" }"#,
        ] {
            let parsed = parse(text);
            assert_eq!(parsed.preferences.apple_auto_instructions, "", "{text}");
            assert!(parsed.warnings.is_empty(), "{text}");
        }
        // Edited: kept as written.
        let edited = parse(r#"{ "apple_auto_instructions": "Moody and cool." }"#);
        assert_eq!(
            edited.preferences.apple_auto_instructions,
            "Moody and cool."
        );
        // The default text itself reads as the default.
        let same = serde_json::json!({ "apple_auto_instructions": DEFAULT_INSTRUCTIONS });
        assert_eq!(
            parse(&same.to_string()).preferences.apple_auto_instructions,
            ""
        );
        // Too long (a file edited by hand), or not text: the default, with a warning; the rest is kept.
        let long = serde_json::json!({
            "apple_auto_instructions": "x".repeat(2001),
            "auto_provider": "yellowmoon",
        });
        let parsed = parse(&long.to_string());
        assert_eq!(parsed.preferences.apple_auto_instructions, "");
        assert_eq!(parsed.preferences.auto_provider, AutoProvider::Yellowmoon);
        assert_eq!(parsed.warnings.len(), 1);
        assert!(
            parsed.warnings[0].contains("At most 2000"),
            "{:?}",
            parsed.warnings
        );
        let wrong = parse(r#"{ "apple_auto_instructions": 7 }"#);
        assert_eq!(wrong.preferences.apple_auto_instructions, "");
        assert_eq!(wrong.warnings.len(), 1);
        // A patch sets them, the default text clears them, and one that is too long is refused.
        let mut p = Preferences::default();
        let set = PreferencesPatch {
            apple_auto_instructions: Some("Warm and soft.".to_owned()),
            ..PreferencesPatch::default()
        };
        assert!(set.check().is_ok());
        assert!(set.apply(&mut p));
        assert_eq!(p.apple_auto_instructions, "Warm and soft.");
        let clear = PreferencesPatch {
            apple_auto_instructions: Some(DEFAULT_INSTRUCTIONS.to_owned()),
            ..PreferencesPatch::default()
        };
        assert!(clear.apply(&mut p));
        assert_eq!(p.apple_auto_instructions, "");
        let too_long = PreferencesPatch {
            apple_auto_instructions: Some("é".repeat(2001)),
            ..PreferencesPatch::default()
        };
        assert!(matches!(
            too_long.check(),
            Err(PreferencesError::Invalid(m)) if m.contains("2001 characters")
        ));
        let just = PreferencesPatch {
            apple_auto_instructions: Some("é".repeat(2000)),
            ..PreferencesPatch::default()
        };
        assert!(just.check().is_ok());
        // The store refuses it and saves nothing.
        let dir = tempfile::tempdir().unwrap();
        let store = PreferencesStore::open(dir.path());
        assert!(store.update(&too_long).is_err());
        assert!(!dir.path().join(FILE_NAME).exists());
        assert_eq!(
            store.update(&set).unwrap().apple_auto_instructions,
            "Warm and soft."
        );
        assert_eq!(
            PreferencesStore::open(dir.path())
                .get()
                .apple_auto_instructions,
            "Warm and soft."
        );
    }

    #[test]
    fn the_json_has_the_fields_of_the_typescript_type() {
        let json = serde_json::to_value(Preferences::default()).expect("json");
        assert_eq!(
            json,
            serde_json::json!({
                "default_look": "builtin:signature",
                "export_defaults": {
                    "format": "jpeg", "jpeg_quality": 92, "bit_depth": 16, "tiff_compression": "lzw",
                    "color_space": "srgb", "size": { "kind": "full" }, "destination": null,
                    "name_template": "{name}_{look}", "if_exists": "keep_both"
                },
                "remove_location_from_exports": false,
                "check_for_updates_automatically": true,
                "watch_downloads_during_import": false,
                "last_used_folders": { "open": null, "export": null, "looks_import": null },
                "auto_provider": "apple",
                "auto_consent": { "claude": false, "agy": false },
                "apple_auto_instructions": ""
            })
        );
    }

    /// Yellowmoon no longer offers ARRI's display LUT as the rendering (owner's decision of
    /// 2026-10-29). A file saved by an earlier version still holds the choice: it is read as
    /// nothing, with no warning, and the next change leaves it out of the file.
    #[test]
    fn a_saved_choice_of_arris_display_rendering_is_ignored_and_dropped_on_the_next_change() {
        let dir = root();
        let path = dir.path().join(FILE_NAME);
        std::fs::write(
            &path,
            r#"{
                "default_look": "arri:3000:film",
                "display_rendering": { "kind": "arri_display", "id": "arri-display:logc4:rec709-g24" },
                "check_for_updates_automatically": false
            }"#,
        )
        .expect("write");
        let before = std::fs::read(&path).expect("read");
        let store = PreferencesStore::open(dir.path());
        assert!(store.warnings().is_empty(), "{:?}", store.warnings());
        let read = store.get();
        assert_eq!(read.default_look.as_str(), "arri:3000:film");
        assert!(!read.check_for_updates_automatically);
        // Reading changes nothing on disk.
        assert_eq!(std::fs::read(&path).expect("read"), before);

        let next = store
            .update(&PreferencesPatch {
                remove_location_from_exports: Some(true),
                ..PreferencesPatch::default()
            })
            .expect("saved");
        assert!(next.remove_location_from_exports);
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        assert!(saved.get("display_rendering").is_none(), "{saved}");
        assert_eq!(saved["default_look"], "arri:3000:film");
        // The patch type has no such field any more: a script that still sends one is told.
        let patch: Result<PreferencesPatch, _> = serde_json::from_str(
            r#"{ "display_rendering": { "kind": "arri_display", "id": "arri-display:logc4:rec709-g24" } }"#,
        );
        assert!(patch.is_err());
    }

    #[test]
    fn a_missing_file_is_the_defaults_and_creates_nothing() {
        let dir = root();
        let store = PreferencesStore::open(dir.path());
        assert_eq!(store.get(), Preferences::default());
        assert!(store.warnings().is_empty());
        assert!(!store.path().exists());
        assert_eq!(std::fs::read_dir(dir.path()).expect("dir").count(), 0);
    }

    #[test]
    fn a_change_is_saved_and_read_back() {
        let dir = root();
        let store = PreferencesStore::open(dir.path());
        let patch = PreferencesPatch {
            check_for_updates_automatically: Some(false),
            export_defaults: Some(ExportOptions {
                format: FormatKind::Tiff,
                bit_depth: BitDepth::Eight,
                color_space: ColorSpace::DisplayP3,
                size: Size::LongEdge { px: 3000 },
                if_exists: IfExists::Skip,
                destination: Some(PathBuf::from("/tmp/out")),
                ..ExportOptions::default()
            }),
            last_used_folders: Some(LastUsedFolders {
                open: Some(PathBuf::from("/Volumes/A1")),
                ..LastUsedFolders::default()
            }),
            ..PreferencesPatch::default()
        };
        let next = store.update(&patch).expect("saved");
        assert!(!next.check_for_updates_automatically);
        assert!(store.path().is_file());
        let again = PreferencesStore::open(dir.path());
        assert_eq!(again.get(), next);
        assert_eq!(
            again.get().export_defaults.size,
            Size::LongEdge { px: 3000 }
        );
        // Only the file, no temporary leftovers.
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .expect("dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![FILE_NAME.to_owned()]);
    }

    #[test]
    fn a_patch_that_changes_nothing_writes_nothing() {
        let dir = root();
        let store = PreferencesStore::open(dir.path());
        store
            .update(&PreferencesPatch {
                check_for_updates_automatically: Some(true),
                ..PreferencesPatch::default()
            })
            .expect("no change");
        assert!(!store.path().exists());
    }

    #[test]
    fn missing_and_unknown_fields_are_forgiven() {
        let p = parse(r#"{ "remove_location_from_exports": true, "from_the_future": [1, 2, 3] }"#);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
        assert!(p.preferences.remove_location_from_exports);
        assert_eq!(
            p.preferences.default_look,
            Preferences::default().default_look
        );
        let q = parse(r#"{ "export_defaults": { "format": "png", "colour": "x" } }"#);
        assert!(q.warnings.is_empty(), "{:?}", q.warnings);
        assert_eq!(q.preferences.export_defaults.format, FormatKind::Png);
    }

    #[test]
    fn a_bad_value_costs_only_that_value() {
        let p = parse(
            r#"{
                "default_look": "not a look id",
                "check_for_updates_automatically": false,
                "export_defaults": { "format": "tiff", "jpeg_quality": 12, "bit_depth": 8 },
                "last_used_folders": { "open": 5, "export": "/tmp/x" }
            }"#,
        );
        assert_eq!(
            p.preferences.default_look,
            Preferences::default().default_look
        );
        assert!(!p.preferences.check_for_updates_automatically);
        assert_eq!(p.preferences.export_defaults.format, FormatKind::Tiff);
        assert_eq!(p.preferences.export_defaults.bit_depth, BitDepth::Eight);
        assert_eq!(p.preferences.export_defaults.jpeg_quality.get(), 92);
        assert_eq!(p.preferences.last_used_folders.open, None);
        assert_eq!(
            p.preferences.last_used_folders.export,
            Some(PathBuf::from("/tmp/x"))
        );
        assert_eq!(p.warnings.len(), 3, "{:?}", p.warnings);
        assert!(p.warnings.iter().any(|w| w.contains("jpeg_quality")));
    }

    #[test]
    fn a_damaged_file_is_the_defaults_with_a_warning_and_is_replaced_on_the_next_change() {
        let dir = root();
        for junk in [
            &b"{ not json"[..],
            b"",
            b"[1, 2]",
            b"\"text\"",
            &[0xff, 0xfe, 0x00, 0x80],
        ] {
            std::fs::write(dir.path().join(FILE_NAME), junk).expect("write junk");
            let store = PreferencesStore::open(dir.path());
            assert_eq!(store.get(), Preferences::default(), "{junk:?}");
            assert_eq!(store.warnings().len(), 1, "{junk:?}");
        }
        let store = PreferencesStore::open(dir.path());
        store
            .update(&PreferencesPatch {
                watch_downloads_during_import: Some(true),
                ..PreferencesPatch::default()
            })
            .expect("saved over the junk");
        assert!(store.warnings().is_empty());
        let again = PreferencesStore::open(dir.path());
        assert!(again.get().watch_downloads_during_import);
        assert!(again.warnings().is_empty());
    }

    /// Runs `f` and returns what it logged at info level and above (the application's default).
    fn logged(f: impl FnOnce()) -> String {
        use std::sync::{Arc, Mutex, PoisonError};

        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::clone(&buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || Sink(Arc::clone(&shared)))
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = buffer.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[test]
    fn the_warning_about_a_damaged_file_does_not_put_its_path_in_the_log() {
        // The path holds the user's name, and a bug report may carry the log (`docs/01`).
        let dir = root();
        std::fs::write(dir.path().join(FILE_NAME), b"{ not json").expect("write junk");
        let log = logged(|| {
            let store = PreferencesStore::open(dir.path());
            assert_eq!(store.warnings().len(), 1);
        });
        assert!(
            log.contains("The preferences file is damaged"),
            "the warning is logged: {log:?}"
        );
        let folder = dir.path().display().to_string();
        assert!(!log.contains(&folder), "the folder reached the log: {log}");
        assert!(!log.contains(FILE_NAME), "the file reached the log: {log}");
    }

    #[test]
    fn export_defaults_the_writers_refuse_are_refused() {
        let dir = root();
        let store = PreferencesStore::open(dir.path());
        let bad = PreferencesPatch {
            export_defaults: Some(ExportOptions {
                size: Size::LongEdge { px: 3 },
                ..ExportOptions::default()
            }),
            ..PreferencesPatch::default()
        };
        let err = store.update(&bad).expect_err("refused");
        assert!(matches!(err, PreferencesError::Invalid(_)), "{err}");
        assert_eq!(store.get(), Preferences::default());
        assert!(!store.path().exists());
        // And a file that holds such a size reads as the default size.
        let p = parse(r#"{ "export_defaults": { "size": { "kind": "long_edge", "px": 3 } } }"#);
        assert_eq!(p.preferences.export_defaults.size, Size::Full);
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn a_change_made_by_another_process_is_not_undone() {
        let dir = root();
        let app = PreferencesStore::open(dir.path());
        let cli = PreferencesStore::open(dir.path());
        cli.update(&PreferencesPatch::default_look(
            LookId::parse("user:warm").expect("id"),
        ))
        .expect("cli");
        app.update(&PreferencesPatch {
            remove_location_from_exports: Some(true),
            ..PreferencesPatch::default()
        })
        .expect("app");
        let now = PreferencesStore::open(dir.path()).get();
        assert_eq!(now.default_look.as_str(), "user:warm");
        assert!(now.remove_location_from_exports);
        // The app's own copy sees it after a reload.
        assert_eq!(app.get().default_look.as_str(), "user:warm");
    }

    #[test]
    fn a_patch_with_a_typo_is_an_error() {
        let bad = serde_json::from_value::<PreferencesPatch>(
            serde_json::json!({ "remove_location_from_export": true }),
        );
        assert!(bad.is_err());
        let ok = serde_json::from_value::<PreferencesPatch>(
            serde_json::json!({ "remove_location_from_exports": true }),
        )
        .expect("patch");
        assert_eq!(ok.remove_location_from_exports, Some(true));
    }

    #[test]
    fn the_file_lives_in_the_app_folder_next_to_the_looks() {
        use std::ffi::OsString;
        use ym_looks::library::{LibraryPaths, Platform, root_from};
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| OsString::from(*v))
            }
        };
        let mac = root_from(env(&[("HOME", "/Users/me")]), Platform::MacOs).expect("mac");
        assert_eq!(
            PreferencesStore::open(&mac).path(),
            Path::new("/Users/me/Library/Application Support/app.yellowmoon/preferences.json")
        );
        let linux = root_from(env(&[("HOME", "/home/me")]), Platform::Unix).expect("linux");
        assert_eq!(
            PreferencesStore::open(&linux).path(),
            Path::new("/home/me/.local/share/app.yellowmoon/preferences.json")
        );
        // YELLOWMOON_HOME replaces the folder, for the looks and the preferences alike.
        let home = root_from(
            env(&[("YELLOWMOON_HOME", "/tmp/ym"), ("HOME", "/Users/me")]),
            Platform::MacOs,
        )
        .expect("override");
        assert_eq!(
            PreferencesStore::open(&home).path(),
            Path::new("/tmp/ym/preferences.json")
        );
        assert_eq!(
            LibraryPaths::new(&home).looks_dir(),
            Path::new("/tmp/ym/looks")
        );
    }

    #[test]
    fn a_damaged_file_is_reported_once_by_a_reload() {
        let dir = root();
        std::fs::write(dir.path().join(FILE_NAME), "{ nope").expect("junk");
        let store = PreferencesStore::open(dir.path());
        assert_eq!(store.warnings().len(), 1);
        for _ in 0..3 {
            store.reload();
            assert_eq!(store.warnings().len(), 1);
        }
        std::fs::write(dir.path().join(FILE_NAME), "{}").expect("fixed");
        store.reload();
        assert!(store.warnings().is_empty());
    }

    #[test]
    fn a_write_failure_changes_nothing() {
        let dir = root();
        // A file where the app folder should be: the folder cannot be created.
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"x").expect("blocker");
        let store = PreferencesStore::open(&blocker);
        let err = store
            .update(&PreferencesPatch {
                remove_location_from_exports: Some(true),
                ..PreferencesPatch::default()
            })
            .expect_err("cannot write");
        assert!(matches!(err, PreferencesError::Write { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("Check that the folder is writable")
        );
        assert!(!store.get().remove_location_from_exports);
    }
}
