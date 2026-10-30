//! `manifest.json` model: imported looks with SHA-256 per file; atomic writes (`docs/08-looks.md` section Library folder).
//!
//! The manifest lists what the user has imported; it holds no preferences (the default look lives in
//! `preferences.json`). Layout, exactly as in the spec:
//!
//! ```json
//! {
//!   "version": 1,
//!   "looks": [
//!     { "id": "arri:3000:film", "name": "Film", "group": { "kind": "arri", "no": 3000, "name": "Film" },
//!       "input": "logc4", "file": "arri/3000-film/Film.cube", "sha256": "…", "lut_size": 65,
//!       "source": { "package": "look-library-logc4", "url": "https://www.arri.com/…",
//!                   "imported": "2026-11-02T10:14:00Z", "original_name": "Film.cube" } }
//!   ]
//! }
//! ```
//!
//! A manifest written by an older version may also carry a `display_luts` list (ARRI's display LUTs,
//! which Yellowmoon no longer imports or uses). It is read without complaint, ignored, and left out
//! of the next file this build writes; the files it names stay on disk.
//!
//! Two additions the spec leaves open: a user LUT's group is `{ "kind": "user" }`, and `package`
//! and `url` are left out of `source` for files that did not come from a known package.
//!
//! Every write goes through [`atomic_write`] (write a temporary file in the same folder, `fsync`,
//! rename, `fsync` the folder), so a crash or a second process never sees half a file.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ym_color::cube::{MAX_3D_SIZE, MIN_3D_SIZE};
use ym_color::look::LookInput;

use crate::library::{IdKind, LookFormat, LookId};

/// The manifest format version this build reads and writes.
pub const MANIFEST_VERSION: u32 = 1;
/// File name of the manifest inside the `looks` folder.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Largest manifest that will be read, in bytes.
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;

/// Where a look came from (`docs/08-looks.md` section Look model).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// The id of the ARRI package it was recognized as (`look-library-logc4`), when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    /// The page the package is downloaded from, for ARRI files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// When it was imported, RFC 3339 in UTC.
    pub imported: String,
    /// The file name it had when imported.
    pub original_name: String,
}

/// The group of an imported look.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ManifestGroup {
    /// One of ARRI's look groups, for example `{ "kind": "arri", "no": 3000, "name": "Film" }`.
    Arri {
        /// The group number from the package's folder name.
        no: u32,
        /// The group's name.
        name: String,
    },
    /// Imported by the user ("Your LUTs").
    User,
}

/// One imported look.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestLook {
    /// `arri:<group>:<slug>` or `user:<slug>`.
    pub id: LookId,
    /// Display name.
    pub name: String,
    /// The group it is listed under.
    pub group: ManifestGroup,
    /// What the LUT expects as input.
    pub input: LookInput,
    /// Path relative to the `looks` folder, with `/` separators.
    pub file: String,
    /// What kind of file it is: `cube` (also when the field is absent: manifests written before
    /// ARRI Look Files were read), `alf4`, `alf4c` or `aml`.
    #[serde(default, skip_serializing_if = "LookFormat::is_cube")]
    pub format: LookFormat,
    /// SHA-256 of the file, lower-case hex.
    pub sha256: String,
    /// Points per axis of the 3D table.
    pub lut_size: usize,
    /// Where it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
}

/// The whole manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Always [`MANIFEST_VERSION`].
    pub version: u32,
    /// Imported looks.
    #[serde(default)]
    pub looks: Vec<ManifestLook>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            looks: Vec::new(),
        }
    }
}

/// Why the manifest could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// The file could not be read or written.
    #[error("Could not access the looks list {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: io::Error,
    },
    /// The file is not a valid manifest.
    #[error("The looks list {path} is damaged: {reason}")]
    Damaged {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
    /// A newer Yellowmoon wrote it.
    #[error(
        "The looks list {path} was written by a newer version of Yellowmoon (format {found}). Update Yellowmoon to use it."
    )]
    NewerVersion {
        /// The file.
        path: PathBuf,
        /// The format version found.
        found: u64,
    },
}

impl Manifest {
    /// The manifest path inside a `looks` folder.
    #[must_use]
    pub fn path(looks_dir: &Path) -> PathBuf {
        looks_dir.join(MANIFEST_FILE)
    }

    /// Reads the manifest from a `looks` folder. A missing file is an empty manifest.
    ///
    /// # Errors
    ///
    /// [`ManifestError`] when the file cannot be read, is damaged or is from a newer version.
    pub fn load(looks_dir: &Path) -> Result<Self, ManifestError> {
        let path = Self::path(looks_dir);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => return Err(ManifestError::Io { path, source }),
        };
        let mut bytes = Vec::new();
        file.take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| ManifestError::Io {
                path: path.clone(),
                source,
            })?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(ManifestError::Damaged {
                path,
                reason: "it is far larger than a looks list should be".to_string(),
            });
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| ManifestError::Damaged {
            path: path.clone(),
            reason: "it is not text".to_string(),
        })?;
        Self::parse(text, &path)
    }

    /// Parses manifest text. `path` is only used in error messages.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Damaged`] or [`ManifestError::NewerVersion`].
    pub fn parse(text: &str, path: &Path) -> Result<Self, ManifestError> {
        let damaged = |reason: String| ManifestError::Damaged {
            path: path.to_path_buf(),
            reason,
        };
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| damaged(e.to_string()))?;
        match value.get("version").and_then(serde_json::Value::as_u64) {
            Some(v) if v == u64::from(MANIFEST_VERSION) => {}
            Some(v) if v > u64::from(MANIFEST_VERSION) => {
                return Err(ManifestError::NewerVersion {
                    path: path.to_path_buf(),
                    found: v,
                });
            }
            _ => return Err(damaged("it has no usable \"version\"".to_string())),
        }
        let manifest: Self = serde_json::from_value(value).map_err(|e| damaged(e.to_string()))?;
        manifest.validate().map_err(damaged)?;
        Ok(manifest)
    }

    /// Checks what the loader relies on: unique ids of the right kind, relative file paths that
    /// stay inside the library, well-formed hashes and LUT sizes.
    fn validate(&self) -> Result<(), String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut check_common = |id: &LookId, file: &str, sha256: &str| -> Result<(), String> {
            if !seen.insert(id.as_str().to_lowercase()) {
                return Err(format!("the id {id} appears twice"));
            }
            if !is_safe_relative_path(file) {
                return Err(format!("{id} has an unsafe file path {file:?}"));
            }
            if !is_sha256_hex(sha256) {
                return Err(format!("{id} has a malformed checksum"));
            }
            Ok(())
        };
        for look in &self.looks {
            check_common(&look.id, &look.file, &look.sha256)?;
            if !matches!(look.id.kind(), IdKind::Arri { .. } | IdKind::User) {
                return Err(format!("{} cannot be an imported look", look.id));
            }
            if !(MIN_3D_SIZE..=MAX_3D_SIZE).contains(&look.lut_size) {
                return Err(format!("{} has an impossible LUT size", look.id));
            }
        }
        Ok(())
    }

    /// Orders entries by id so the file is stable between runs.
    pub fn sort(&mut self) {
        self.looks.sort_by(|a, b| a.id.cmp(&b.id));
    }

    /// The manifest as pretty JSON with a final newline.
    #[must_use]
    pub fn to_json(&self) -> String {
        // Serialising plain data cannot fail; the fallback keeps the signature infallible.
        let mut text = serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string());
        text.push('\n');
        text
    }

    /// Writes the manifest atomically into a `looks` folder (created if needed).
    ///
    /// # Errors
    ///
    /// [`ManifestError::Io`] when the write fails.
    pub fn save(&self, looks_dir: &Path) -> Result<(), ManifestError> {
        let path = Self::path(looks_dir);
        fs::create_dir_all(looks_dir).map_err(|source| ManifestError::Io {
            path: looks_dir.to_path_buf(),
            source,
        })?;
        atomic_write(&path, self.to_json().as_bytes())
            .map_err(|source| ManifestError::Io { path, source })
    }

    /// The imported look with this id (compared without regard to case).
    #[must_use]
    pub fn look(&self, id: &LookId) -> Option<&ManifestLook> {
        self.looks
            .iter()
            .find(|l| l.id.as_str().eq_ignore_ascii_case(id.as_str()))
    }
}

/// Moves a damaged `manifest.json` aside as `manifest.json.damaged-<n>` so a fresh list can be
/// started without losing what was there. Returns the new path.
///
/// # Errors
///
/// The I/O error from the rename.
pub fn quarantine(looks_dir: &Path) -> io::Result<PathBuf> {
    let from = Manifest::path(looks_dir);
    for n in 1_u32.. {
        let to = looks_dir.join(format!("{MANIFEST_FILE}.damaged-{n}"));
        if !to.exists() {
            fs::rename(&from, &to)?;
            return Ok(to);
        }
    }
    Err(io::Error::other("no free name for the damaged looks list"))
}

/// Whether `file` is a relative path with `/` separators that cannot leave the library folder.
#[must_use]
pub fn is_safe_relative_path(file: &str) -> bool {
    if file.is_empty() || file.contains('\\') || file.contains('\0') {
        return false;
    }
    let path = Path::new(file);
    !path.is_absolute() && path.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Whether `text` is 64 lower-case hex digits.
#[must_use]
pub fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// SHA-256 of `bytes` as lower-case hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// SHA-256 of a file, read in 1 MiB pieces.
///
/// # Errors
///
/// The I/O error from opening or reading the file.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to a new temporary file next to `dest` and `fsync`s it. The temporary file is
/// hidden (`.<name>.tmp-<pid>-<n>`) and is the caller's to rename with [`commit_staged`] or delete.
///
/// # Errors
///
/// The I/O error from creating or writing the file; nothing is left behind on failure.
pub fn stage_write(dest: &Path, bytes: &[u8]) -> io::Result<PathBuf> {
    let dir = dest.parent().unwrap_or_else(|| Path::new("."));
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = dir.join(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    match result {
        Ok(()) => Ok(temp),
        Err(e) => {
            let _ = fs::remove_file(&temp);
            Err(e)
        }
    }
}

/// Renames a file made by [`stage_write`] over `dest` (atomic on one volume).
///
/// # Errors
///
/// The I/O error from the rename; the temporary file is removed on failure.
pub fn commit_staged(temp: &Path, dest: &Path) -> io::Result<()> {
    match fs::rename(temp, dest) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(temp);
            Err(e)
        }
    }
}

/// `fsync`s a folder so renames inside it survive a crash. Failures are ignored: some platforms
/// cannot open a folder for syncing, and the data itself is already on disk.
pub fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}

/// Atomic file write: temporary file in the same folder, `fsync`, rename, `fsync` the folder.
///
/// # Errors
///
/// The I/O error from any step; the destination is either the old file or the new one, never half.
pub fn atomic_write(dest: &Path, bytes: &[u8]) -> io::Result<()> {
    let temp = stage_write(dest, bytes)?;
    commit_staged(&temp, dest)?;
    if let Some(dir) = dest.parent() {
        sync_dir(dir);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            version: 1,
            looks: vec![ManifestLook {
                id: LookId::arri(3000, "film"),
                name: "Film".into(),
                group: ManifestGroup::Arri {
                    no: 3000,
                    name: "Film".into(),
                },
                input: LookInput::LogC4,
                file: "arri/3000-film/Film.cube".into(),
                format: LookFormat::Cube,
                sha256: "a".repeat(64),
                lut_size: 65,
                source: Some(Source {
                    package: Some("look-library-logc4".into()),
                    url: Some("https://example.invalid/".into()),
                    imported: "2026-11-02T10:14:00Z".into(),
                    original_name: "Film.cube".into(),
                }),
            }],
        }
    }

    #[test]
    fn json_matches_the_documented_shape() {
        let json = sample().to_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["version"], 1);
        let look = &value["looks"][0];
        assert_eq!(look["id"], "arri:3000:film");
        assert_eq!(look["group"]["kind"], "arri");
        assert_eq!(look["group"]["no"], 3000);
        assert_eq!(look["input"], "logc4");
        assert_eq!(look["source"]["package"], "look-library-logc4");
        assert!(value.get("display_luts").is_none());
        assert!(json.ends_with("}\n"));
    }

    #[test]
    fn a_list_with_display_luts_from_an_older_version_is_read_and_the_list_is_dropped_on_save() {
        // What an older version wrote when the person had imported ARRI's display LUTs. The entries
        // are not checked (they may name anything), nothing is reported, and the next write leaves
        // them out.
        let mut value: serde_json::Value = serde_json::from_str(&sample().to_json()).unwrap();
        value["display_luts"] = serde_json::json!([
            { "id": "arri-display:logc4:rec709-g24", "input": "logc4", "target": "rec709-g24",
              "file": "arri-display/Rec709.cube", "sha256": "b".repeat(64) },
            { "id": "not even an id", "file": "../../outside.cube" }
        ]);
        let old = serde_json::to_string_pretty(&value).unwrap();
        let read = Manifest::parse(&old, Path::new("m.json")).unwrap();
        assert_eq!(read, sample());
        assert!(!read.to_json().contains("display_luts"));
        assert!(!read.to_json().contains("arri-display"));
    }

    #[test]
    fn the_format_of_a_look_file_is_recorded_and_a_cube_has_none() {
        let mut m = sample();
        // A list of cubes is what 0.1.1 wrote: no `format` key at all.
        assert!(!m.to_json().contains("format"));
        m.looks[0].format = LookFormat::Alf4;
        m.looks[0].file = "arri/3000-film/Film.alf4".into();
        m.looks[0].lut_size = 33;
        let json = m.to_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["looks"][0]["format"], "alf4");
        let back = Manifest::parse(&json, Path::new("m.json")).unwrap();
        assert_eq!(back, m);
        for (format, word) in [
            (LookFormat::Alf4c, "alf4c"),
            (LookFormat::Aml, "aml"),
            (LookFormat::Cube, "cube"),
        ] {
            let text = serde_json::to_string(&format).unwrap();
            assert_eq!(text, format!("\"{word}\""));
        }
        // A list written before look files were read has no format: every entry is a cube.
        let old = json.replace("\"format\": \"alf4\",", "");
        assert!(!old.contains("format"));
        let back = Manifest::parse(&old, Path::new("m.json")).unwrap();
        assert_eq!(back.looks[0].format, LookFormat::Cube);
        // A format this build does not know is a damaged list, not a silent cube.
        let odd = json.replace("\"alf4\"", "\"alf9\"");
        assert!(matches!(
            Manifest::parse(&odd, Path::new("m.json")),
            Err(ManifestError::Damaged { .. })
        ));
    }

    #[test]
    fn round_trips_through_a_folder() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Manifest::load(dir.path()).unwrap(), Manifest::default());
        let m = sample();
        m.save(dir.path()).unwrap();
        assert_eq!(Manifest::load(dir.path()).unwrap(), m);
        // No temporary files remain.
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![MANIFEST_FILE.to_string()]);
    }

    #[test]
    fn user_group_and_optional_source_fields() {
        let json = r#"{"version":1,"looks":[{"id":"user:a","name":"A","group":{"kind":"user"},
            "input":"logc3_ei800","file":"user/a.cube","sha256":"0000000000000000000000000000000000000000000000000000000000000000",
            "lut_size":33,"source":{"imported":"2026-01-01T00:00:00Z","original_name":"a.cube"}}]}"#;
        let m = Manifest::parse(json, Path::new("m.json")).unwrap();
        assert_eq!(m.looks[0].group, ManifestGroup::User);
        assert_eq!(m.looks[0].input, LookInput::LogC3Ei800);
        let again = m.to_json();
        assert!(!again.contains("\"package\""));
        assert!(!again.contains("\"url\""));
    }

    #[test]
    fn rejects_damage_and_unsafe_paths() {
        let p = Path::new("m.json");
        assert!(matches!(
            Manifest::parse("{ nope", p),
            Err(ManifestError::Damaged { .. })
        ));
        assert!(matches!(
            Manifest::parse("{}", p),
            Err(ManifestError::Damaged { .. })
        ));
        assert!(matches!(
            Manifest::parse(r#"{"version":2}"#, p),
            Err(ManifestError::NewerVersion { found: 2, .. })
        ));
        let mut m = sample();
        m.looks[0].file = "../../etc/passwd".into();
        assert!(matches!(
            Manifest::parse(&m.to_json(), p),
            Err(ManifestError::Damaged { .. })
        ));
        let mut m = sample();
        m.looks[0].sha256 = "xyz".into();
        assert!(Manifest::parse(&m.to_json(), p).is_err());
        let mut m = sample();
        m.looks.push(m.looks[0].clone());
        assert!(Manifest::parse(&m.to_json(), p).is_err());
        let mut m = sample();
        m.looks[0].id = LookId::builtin("neutral");
        assert!(Manifest::parse(&m.to_json(), p).is_err());
    }

    #[test]
    fn safe_paths() {
        assert!(is_safe_relative_path("arri/3000-film/a.cube"));
        assert!(!is_safe_relative_path("/abs/a.cube"));
        assert!(!is_safe_relative_path("a/../b.cube"));
        assert!(!is_safe_relative_path("a\\b.cube"));
        assert!(!is_safe_relative_path(""));
        assert!(!is_safe_relative_path("./a.cube"));
    }

    #[test]
    fn atomic_write_replaces_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.bin");
        atomic_write(&dest, b"one").unwrap();
        atomic_write(&dest, b"two").unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"two");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        assert_eq!(hash_file(&dest).unwrap(), sha256_hex(b"two"));
        assert!(is_sha256_hex(&sha256_hex(b"")));
    }

    #[test]
    fn quarantine_keeps_the_old_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(Manifest::path(dir.path()), "garbage").unwrap();
        let moved = quarantine(dir.path()).unwrap();
        assert!(moved.exists());
        assert!(!Manifest::path(dir.path()).exists());
        fs::write(Manifest::path(dir.path()), "garbage 2").unwrap();
        let moved2 = quarantine(dir.path()).unwrap();
        assert_ne!(moved, moved2);
    }
}
