//! `tools/golden/expected.json`: the hash of every golden picture, committed with the code.
//!
//! The references themselves (16-bit PNGs under `testdata/golden/refs/`) are not in git. What is in git is the SHA-256
//! of each picture's pixels, so a run needs no stored picture to know that nothing changed (`golden check` passes an
//! identical hash at once); the picture is needed only to say *how much* a changed file moved. Every `golden update`
//! that changes an existing picture has to carry a note, which is kept in `log`: that is the "explanation" the gate
//! asks for (`docs/13`).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// What is expected of one golden picture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedEntry {
    /// SHA-256 of the pixels (`Rgb16::pixel_sha256`).
    pub pixels_sha256: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pipeline version of the binary that made the picture.
    pub pipeline_version: String,
}

/// One explained update.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// The pipeline version of the binary that made the new pictures.
    pub pipeline_version: String,
    /// Why (`golden update --note`).
    pub note: String,
    /// Existing pictures whose pixels changed.
    pub changed: usize,
    /// Pictures added.
    pub added: usize,
    /// The largest ΔE00 any changed picture moved by (0 when none had a reference to compare with).
    pub max_delta_e_2000: f64,
}

/// The contents of `expected.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Expected {
    /// Format version.
    pub schema_version: u32,
    /// What the file is.
    pub description: String,
    /// Picture id to its expected hash.
    #[serde(default)]
    pub entries: BTreeMap<String, ExpectedEntry>,
    /// Explained updates, oldest first.
    #[serde(default)]
    pub log: Vec<LogEntry>,
}

const DESCRIPTION: &str = "Expected pixel hashes of the golden pictures (docs/13-testing.md, section Golden images). Written by \
`golden update`, which needs a --note for every change to an existing picture; the note is kept in `log`. The pictures \
themselves live in testdata/golden/refs (git-ignored).";

impl Default for Expected {
    fn default() -> Self {
        Self {
            schema_version: 1,
            description: DESCRIPTION.to_owned(),
            entries: BTreeMap::new(),
            log: Vec::new(),
        }
    }
}

impl Expected {
    /// Reads `path`; a missing file is an empty set.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let e: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            e.schema_version == 1,
            "{} has schema version {}, this tool reads 1",
            path.display(),
            e.schema_version
        );
        Ok(e)
    }

    /// Writes `path` (sorted keys, indented, a final newline), replacing it atomically.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_empty_and_a_saved_one_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("expected.json");
        assert!(Expected::load(&path).unwrap().entries.is_empty());
        let mut e = Expected::default();
        e.entries.insert(
            "b".into(),
            ExpectedEntry {
                pixels_sha256: "ab".into(),
                width: 4,
                height: 2,
                pipeline_version: "0.1.0".into(),
            },
        );
        e.entries.insert(
            "a".into(),
            ExpectedEntry {
                pixels_sha256: "cd".into(),
                width: 8,
                height: 4,
                pipeline_version: "0.1.0".into(),
            },
        );
        e.log.push(LogEntry {
            pipeline_version: "0.1.0".into(),
            note: "first".into(),
            changed: 0,
            added: 2,
            max_delta_e_2000: 0.0,
        });
        e.save(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("}\n"));
        assert!(
            text.find("\"a\"").unwrap() < text.find("\"b\"").unwrap(),
            "keys are sorted"
        );
        assert_eq!(Expected::load(&path).unwrap(), e);
    }

    #[test]
    fn another_schema_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.json");
        fs::write(&path, r#"{"schema_version":2,"description":"x"}"#).unwrap();
        assert!(Expected::load(&path).is_err());
    }
}
