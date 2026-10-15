//! `data/looks/arri-packages.json`: what the five ARRI packages look like (`docs/08-looks.md` section Recognition rules).
//!
//! The file holds patterns and fingerprints only, never LUT data:
//!
//! ```json
//! {
//!   "version": 1,
//!   "note": "...",
//!   "page_url": "https://www.arri.com/en/learn/camera-systems/image-science/look-files",
//!   "display_names": ["gamma24", "rec709", "p3", "rec2020", "hlg", "pq"],
//!   "packages": [
//!     {
//!       "id": "look-library-logc4",
//!       "title": "Look Library · LogC4",
//!       "arri_title": "Look Library for LogC4 (3D-LUTs)",  // as on ARRI's page; optional
//!       "role": "looks",                      // "looks" or "display_luts"
//!       "input": "logc4",                     // "logc4", "logc3" or "logc3_to_rec709"
//!       "download_mb": 181,
//!       "zip_name": [ { "all_of": ["look", "logc4"], "none_of": ["logc3"] } ],
//!       "expected": { "cube_files": 87, "groups": 9, "lut_sizes": [65] },
//!       "markers": {
//!         "encoding": { "any_of": ["logc4"], "none_of": ["logc3"] },
//!         "display_rendering": "absent",      // "required", "absent" or "any"
//!         "group_folders": "required"
//!       },
//!       "fingerprints": [ { "path": "3000 Film/x.cube", "size": 8123456, "lut_size": 65, "sha256": "…" } ]
//!     }
//!   ]
//! }
//! ```
//!
//! All name patterns are compared in folded form (lower case, letters and digits only), so
//! `"LogC4"`, `"logc-4"` and `"LOGC_4"` are the same.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ym_color::look::LookInput;

use super::names::fold_alnum;
use crate::library::DisplayInput;
use crate::manifest::is_sha256_hex;

/// The format version of `arri-packages.json` this build reads.
pub const PACKAGES_VERSION: u32 = 1;
/// Where the file lives below the data folder.
pub const PACKAGES_PATH: &str = "looks/arri-packages.json";
/// The page that ARRI publishes the packages on.
pub const ARRI_LOOK_FILES_URL: &str =
    "https://www.arri.com/en/learn/camera-systems/image-science/look-files";

fn default_display_names() -> Vec<String> {
    [
        "gamma24", "rec709", "bt709", "p3", "dcip3", "rec2020", "bt2020", "hlg", "pq", "srgb",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// What a package holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageRole {
    /// Looks: creative LUTs, listed in the Look tab.
    Looks,
    /// Display LUTs: ARRI's display rendering, chosen in the display setting.
    DisplayLuts,
}

/// The encoding a package's LUTs expect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageInput {
    /// LogC4 / AWG4.
    #[serde(rename = "logc4")]
    LogC4,
    /// LogC3 EI 800 / AWG3.
    #[serde(rename = "logc3")]
    LogC3,
    /// LogC3 in, Rec.709 out (looks only).
    #[serde(rename = "logc3_to_rec709")]
    LogC3ToRec709,
}

/// Whether a feature must, must not, or may be present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Presence {
    /// Either way.
    #[default]
    Any,
    /// Must be present.
    Required,
    /// Must not be present.
    Absent,
}

impl Presence {
    /// Whether `present` satisfies the requirement.
    #[must_use]
    pub fn allows(self, present: bool) -> bool {
        match self {
            Self::Any => true,
            Self::Required => present,
            Self::Absent => !present,
        }
    }
}

/// A name pattern over folded text: every `all_of` word, at least one `any_of` word (when there
/// are any) and no `none_of` word must be contained.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamePattern {
    /// All of these must appear.
    #[serde(default)]
    pub all_of: Vec<String>,
    /// At least one of these must appear (ignored when empty).
    #[serde(default)]
    pub any_of: Vec<String>,
    /// None of these may appear.
    #[serde(default)]
    pub none_of: Vec<String>,
}

impl NamePattern {
    /// Whether the pattern says anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.all_of.is_empty() && self.any_of.is_empty() && self.none_of.is_empty()
    }

    /// Whether already-folded text matches.
    #[must_use]
    pub fn matches_folded(&self, folded: &str) -> bool {
        let has = |w: &String| folded.contains(&fold_alnum(w));
        self.all_of.iter().all(has)
            && (self.any_of.is_empty() || self.any_of.iter().any(has))
            && !self.none_of.iter().any(has)
    }

    /// Whether `text` matches (it is folded first).
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        self.matches_folded(&fold_alnum(text))
    }
}

/// How many files and what LUT sizes a package should have.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expected {
    /// `.cube` files in the package.
    #[serde(default)]
    pub cube_files: Option<usize>,
    /// Look groups (informational).
    #[serde(default)]
    pub groups: Option<usize>,
    /// 3D sizes the package's LUTs have; empty when not known.
    #[serde(default)]
    pub lut_sizes: Vec<usize>,
}

/// What the files of a package carry in their names and headers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Markers {
    /// Words that name the encoding, searched in the source name, the file's path and its `TITLE`.
    #[serde(default)]
    pub encoding: NamePattern,
    /// Whether the file names a display rendering (`-to-` plus `Rec709`, `P3`, …).
    #[serde(default)]
    pub display_rendering: Presence,
    /// Whether the file sits in a folder named with a four-digit group number.
    #[serde(default)]
    pub group_folders: Presence,
}

/// One file of a package, identified by its hash. Names, sizes and hashes only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// Path inside the package, with `/` separators.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// 3D points per axis.
    pub lut_size: usize,
    /// SHA-256, lower-case hex.
    pub sha256: String,
}

/// One of ARRI's packages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Package {
    /// Stable id, used as `source.package` in the manifest (`look-library-logc4`).
    pub id: String,
    /// Name shown in the wizard (`Look Library · LogC4`).
    pub title: String,
    /// The title of the download on ARRI's page (`Look Library for LogC4 (3D-LUTs)`), for the words
    /// that tell the person what to download. [`Package::download_title`] falls back to `title`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arri_title: Option<String>,
    /// Looks or display LUTs.
    pub role: PackageRole,
    /// The encoding the LUTs expect.
    pub input: PackageInput,
    /// Download size in megabytes, for the wizard's total.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_mb: Option<u32>,
    /// Patterns for the zip's file name; any one matching is enough. Used by the Downloads watcher.
    #[serde(default)]
    pub zip_name: Vec<NamePattern>,
    /// Counts and LUT sizes.
    #[serde(default)]
    pub expected: Expected,
    /// Name and header markers of the package's files.
    #[serde(default)]
    pub markers: Markers,
    /// What the fingerprint tool saw (zip name, size, counts); informational, never read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<serde_json::Value>,
    /// One entry per file; empty until `tools/looks/fingerprint.py` has been run (HUMAN M3-H2).
    #[serde(default)]
    pub fingerprints: Vec<Fingerprint>,
}

impl Package {
    /// The look input for a looks package.
    #[must_use]
    pub fn look_input(&self) -> Option<LookInput> {
        if self.role != PackageRole::Looks {
            return None;
        }
        Some(match self.input {
            PackageInput::LogC4 => LookInput::LogC4,
            PackageInput::LogC3 => LookInput::LogC3Ei800,
            PackageInput::LogC3ToRec709 => LookInput::LogC3ToRec709,
        })
    }

    /// The display input for a display-LUT package.
    #[must_use]
    pub fn display_input(&self) -> Option<DisplayInput> {
        if self.role != PackageRole::DisplayLuts {
            return None;
        }
        match self.input {
            PackageInput::LogC4 => Some(DisplayInput::LogC4),
            PackageInput::LogC3 => Some(DisplayInput::LogC3),
            PackageInput::LogC3ToRec709 => None,
        }
    }

    /// The name ARRI gives the download, so the person can find it on ARRI's page.
    #[must_use]
    pub fn download_title(&self) -> &str {
        self.arri_title.as_deref().unwrap_or(&self.title)
    }

    /// Whether the package has been fingerprinted.
    #[must_use]
    pub fn is_fingerprinted(&self) -> bool {
        !self.fingerprints.is_empty()
    }

    /// Whether a zip file name looks like this package's download.
    #[must_use]
    pub fn matches_zip_name(&self, file_name: &str) -> bool {
        let folded = fold_alnum(file_name.strip_suffix(".zip").unwrap_or(file_name));
        self.zip_name
            .iter()
            .any(|p| !p.is_empty() && p.matches_folded(&folded))
    }
}

/// Why `arri-packages.json` could not be used.
#[derive(Debug, thiserror::Error)]
pub enum PackageDbError {
    /// The file could not be read.
    #[error("Could not read the ARRI package list {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: io::Error,
    },
    /// The file is not valid.
    #[error("The ARRI package list is not valid: {0}")]
    Invalid(String),
}

/// The contents of `arri-packages.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackageDb {
    /// Always [`PACKAGES_VERSION`].
    pub version: u32,
    /// Free text kept with the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The page the packages are downloaded from; recorded as `source.url` of imported files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_url: Option<String>,
    /// Display-rendering words that follow `-to-` in file names (folded when compared).
    #[serde(default = "default_display_names")]
    pub display_names: Vec<String>,
    /// The packages.
    #[serde(default)]
    pub packages: Vec<Package>,
}

impl Default for PackageDb {
    fn default() -> Self {
        Self::empty()
    }
}

impl PackageDb {
    /// A list with no packages: everything gets the generic question.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            version: PACKAGES_VERSION,
            note: None,
            page_url: None,
            display_names: default_display_names(),
            packages: Vec::new(),
        }
    }

    /// Parses and validates the JSON text.
    ///
    /// # Errors
    ///
    /// [`PackageDbError::Invalid`] for bad JSON, another version, duplicate ids, a display package
    /// with a Rec.709-baked input, or a malformed fingerprint.
    pub fn from_json(text: &str) -> Result<Self, PackageDbError> {
        let db: Self =
            serde_json::from_str(text).map_err(|e| PackageDbError::Invalid(e.to_string()))?;
        db.validate()?;
        Ok(db)
    }

    /// Reads the file at `path`.
    ///
    /// # Errors
    ///
    /// [`PackageDbError::Io`] or the errors of [`PackageDb::from_json`].
    pub fn load(path: &Path) -> Result<Self, PackageDbError> {
        let text = fs::read_to_string(path).map_err(|source| PackageDbError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json(&text)
    }

    /// Reads `looks/arri-packages.json` from the data folder (`docs/01-architecture.md`
    /// section Finding data/).
    ///
    /// # Errors
    ///
    /// As [`PackageDb::load`].
    pub fn load_from_data_dir(data_dir: &Path) -> Result<Self, PackageDbError> {
        Self::load(&data_dir.join(PACKAGES_PATH))
    }

    fn validate(&self) -> Result<(), PackageDbError> {
        let bad = |m: String| Err(PackageDbError::Invalid(m));
        if self.version != PACKAGES_VERSION {
            return bad(format!(
                "version {} is not supported (expected {PACKAGES_VERSION})",
                self.version
            ));
        }
        for (i, package) in self.packages.iter().enumerate() {
            if package.id.trim().is_empty() {
                return bad(format!("package {i} has no id"));
            }
            if self.packages[..i].iter().any(|p| p.id == package.id) {
                return bad(format!("package id {} appears twice", package.id));
            }
            if package.role == PackageRole::DisplayLuts
                && package.input == PackageInput::LogC3ToRec709
            {
                return bad(format!(
                    "{} is a display-LUT package; \"logc3_to_rec709\" is for looks only",
                    package.id
                ));
            }
            for fp in &package.fingerprints {
                if !is_sha256_hex(&fp.sha256) {
                    return bad(format!(
                        "{}: fingerprint of {:?} has a malformed sha256",
                        package.id, fp.path
                    ));
                }
            }
        }
        Ok(())
    }

    /// The package with this id.
    #[must_use]
    pub fn package(&self, id: &str) -> Option<&Package> {
        self.packages.iter().find(|p| p.id == id)
    }

    /// The packages a downloaded zip's file name could be.
    #[must_use]
    pub fn packages_for_zip_name(&self, file_name: &str) -> Vec<&Package> {
        self.packages
            .iter()
            .filter(|p| p.matches_zip_name(file_name))
            .collect()
    }

    /// The wizard's "Download total": the sizes of the packages with these ids, in megabytes.
    /// Unknown ids and packages without a recorded size count as zero.
    #[must_use]
    pub fn download_total_mb(&self, ids: &[String]) -> u32 {
        self.packages
            .iter()
            .filter(|p| ids.contains(&p.id))
            .filter_map(|p| p.download_mb)
            .sum()
    }

    /// The page ARRI publishes the packages on.
    #[must_use]
    pub fn page_url(&self) -> &str {
        self.page_url.as_deref().unwrap_or(ARRI_LOOK_FILES_URL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(all: &[&str], any: &[&str], none: &[&str]) -> NamePattern {
        let own = |s: &[&str]| s.iter().map(|x| (*x).to_string()).collect();
        NamePattern {
            all_of: own(all),
            any_of: own(any),
            none_of: own(none),
        }
    }

    #[test]
    fn patterns_compare_folded_text() {
        let p = pattern(&["look", "logc4"], &[], &["logc3"]);
        assert!(p.matches("ARRI_Look_Library_LogC4.zip"));
        assert!(p.matches("look library (LogC-4)"));
        assert!(!p.matches("Look Library LogC3"));
        assert!(!p.matches("Display LUTs LogC4"));
        let any = pattern(&[], &["p3", "rec2020"], &[]);
        assert!(any.matches("x_P3_y"));
        assert!(!any.matches("x_y"));
        assert!(NamePattern::default().is_empty());
    }

    #[test]
    fn presence_rules() {
        assert!(Presence::Any.allows(true) && Presence::Any.allows(false));
        assert!(Presence::Required.allows(true) && !Presence::Required.allows(false));
        assert!(!Presence::Absent.allows(true) && Presence::Absent.allows(false));
    }

    const MINI: &str = r#"{
        "version": 1, "note": "keep me",
        "packages": [
          { "id": "a", "title": "A", "role": "looks", "input": "logc3_to_rec709",
            "zip_name": [ { "all_of": ["look", "logc3", "rec709"] } ],
            "expected": { "cube_files": 3, "lut_sizes": [33] } },
          { "id": "b", "title": "B", "role": "display_luts", "input": "logc4" }
        ] }"#;

    #[test]
    fn parses_the_documented_shape() {
        let db = PackageDb::from_json(MINI).unwrap();
        assert_eq!(db.note.as_deref(), Some("keep me"));
        assert_eq!(db.packages.len(), 2);
        assert_eq!(db.packages[0].look_input(), Some(LookInput::LogC3ToRec709));
        assert_eq!(db.packages[0].display_input(), None);
        assert_eq!(db.packages[1].display_input(), Some(DisplayInput::LogC4));
        assert!(!db.packages[0].is_fingerprinted());
        assert!(db.display_names.contains(&"hlg".to_string()));
        assert_eq!(db.page_url(), ARRI_LOOK_FILES_URL);
        assert_eq!(
            db.packages_for_zip_name("ARRI Look Library LogC3 to Rec709.zip")
                .len(),
            1
        );
        assert!(db.packages_for_zip_name("holiday.zip").is_empty());
        assert_eq!(db.download_total_mb(&["a".to_string(), "b".to_string()]), 0);
    }

    #[test]
    fn download_total_adds_the_chosen_packages() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/looks/arri-packages.json");
        let db = PackageDb::load(&path).unwrap();
        let ids = |list: &[&str]| list.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert_eq!(db.download_total_mb(&ids(&["look-library-logc4"])), 190);
        assert_eq!(
            db.download_total_mb(&ids(&["look-library-logc4", "display-luts-logc4"])),
            190 + 32
        );
        assert_eq!(db.download_total_mb(&ids(&["nope"])), 0);
    }

    #[test]
    fn rejects_inconsistent_lists() {
        let dup = MINI.replace("\"id\": \"b\"", "\"id\": \"a\"");
        assert!(PackageDb::from_json(&dup).is_err());
        let wrong = MINI.replace(
            "\"role\": \"display_luts\", \"input\": \"logc4\"",
            "\"role\": \"display_luts\", \"input\": \"logc3_to_rec709\"",
        );
        assert!(PackageDb::from_json(&wrong).is_err());
        let version = MINI.replace("\"version\": 1", "\"version\": 7");
        assert!(PackageDb::from_json(&version).is_err());
        let bad_hash = MINI.replace(
            "\"expected\"",
            "\"fingerprints\": [{\"path\":\"x\",\"size\":1,\"lut_size\":2,\"sha256\":\"nope\"}], \"expected\"",
        );
        assert!(PackageDb::from_json(&bad_hash).is_err());
        assert!(PackageDb::from_json("{").is_err());
    }
}
