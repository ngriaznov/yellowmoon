//! The golden manifest (`tools/golden/manifest.json`) and the choice of entries a run takes.
//!
//! One entry per input file. The synthetic entries are written by `python3 -m training.synth.make_scenes
//! --write-manifest tools/golden/manifest.json` and never edited by hand; entries of the other tiers (`cc0`, `free`,
//! `eval`) are left alone by that command and come with the fetch scripts. `settings` is the `ym:Settings` document
//! the render uses (`{}` is everything Auto). The manifest's `expected_hash` and `reference` fields stay empty for
//! generated entries: the runner keeps its hashes in `expected.json` and its images under `testdata/golden/refs`
//! (`docs/13-testing.md`, section Golden images).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::Deserialize;
use serde_json::Value;

/// One golden file.
#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    /// Unique id, also the name of the reference image (`/` becomes a folder).
    pub id: String,
    /// `synthetic`, `synthetic-eval`, `cc0`, `free` or `eval`.
    pub tier: String,
    /// `default`, `full` or `eval` for generated entries; empty for the others.
    #[serde(default)]
    pub set: String,
    /// The input file, relative to `testdata/`.
    pub file: String,
    /// The scene's ground truth (generated entries), relative to `testdata/`.
    #[serde(default)]
    pub sidecar: Option<String>,
    /// The `ym:Settings` document for this file; `{}` (the default) is everything Auto.
    #[serde(default = "no_settings")]
    pub settings: Value,
    /// The ARW variant of a generated file (`lossless-l`, `craw`, ...).
    #[serde(default)]
    pub variant: Option<String>,
    /// The scene's light token (`D6500`, `bb2850`, ...).
    #[serde(default)]
    pub light: Option<String>,
    /// The camera model.
    #[serde(default)]
    pub body: Option<String>,
    /// The CFA phase of a generated file (`RGGB`, `GRBG`, ...).
    #[serde(default)]
    pub cfa: Option<String>,
    /// The bit depth of a generated file's mosaic (14, or 12 for the 12-bit variants).
    #[serde(default)]
    pub bits: Option<u32>,
}

fn no_settings() -> Value {
    Value::Object(serde_json::Map::new())
}

/// The manifest: its entries, in file order.
#[derive(Clone, Debug)]
pub struct Manifest {
    /// Every entry.
    pub entries: Vec<Entry>,
}

impl Manifest {
    /// Reads and checks the manifest at `path`: the ids must be unique and the settings must be objects.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let doc: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let list = doc
            .get("entries")
            .cloned()
            .with_context(|| format!("{} has no `entries`", path.display()))?;
        let entries: Vec<Entry> = serde_json::from_value(list)
            .with_context(|| format!("reading the entries of {}", path.display()))?;
        let mut seen = std::collections::BTreeSet::new();
        for e in &entries {
            if !seen.insert(e.id.as_str()) {
                bail!("{}: the id {:?} appears twice", path.display(), e.id);
            }
            if !e.settings.is_object() {
                bail!(
                    "{}: the settings of {:?} are not an object",
                    path.display(),
                    e.id
                );
            }
            if e.id.is_empty() || e.id.starts_with('/') || e.id.split('/').any(|p| p == "..") {
                bail!(
                    "{}: the id {:?} is not a safe relative name",
                    path.display(),
                    e.id
                );
            }
        }
        Ok(Self { entries })
    }
}

/// Which entries a run takes (`--set`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SetChoice {
    /// The default synthetic set (tier `synthetic`, set `default`): what a plain `tools/fetch/synthetic.sh` writes.
    Synthetic,
    /// All the generated files of tier `synthetic` (also the full-size scene and the rest of the light x body product).
    SyntheticFull,
    /// The evaluation scene with the CIE samples (tier `synthetic-eval`, local only).
    SyntheticEval,
    /// The CC0 tier (RawDB and raw.pixls.us files).
    Cc0,
    /// The free-licence tier (Play Raw files, with attribution).
    Tier2,
    /// The copyrighted evaluation tier (local only, never in CI).
    Tier3,
    /// Every entry of the manifest.
    All,
}

impl SetChoice {
    /// Whether `entry` belongs to this set.
    #[must_use]
    pub fn includes(self, entry: &Entry) -> bool {
        match self {
            Self::Synthetic => {
                entry.tier == "synthetic" && (entry.set.is_empty() || entry.set == "default")
            }
            Self::SyntheticFull => entry.tier == "synthetic",
            Self::SyntheticEval => entry.tier == "synthetic-eval",
            Self::Cc0 => entry.tier == "cc0",
            Self::Tier2 => entry.tier == "free",
            Self::Tier3 => entry.tier == "eval",
            Self::All => true,
        }
    }
}

/// The entries of `manifest` in `set` whose id contains every one of `filters` (all of them: `--filter D6500 --filter
/// ILCE-1` narrows).
#[must_use]
pub fn select<'a>(manifest: &'a Manifest, set: SetChoice, filters: &[String]) -> Vec<&'a Entry> {
    manifest
        .entries
        .iter()
        .filter(|e| set.includes(e) && filters.iter().all(|f| e.id.contains(f.as_str())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, tier: &str, set: &str) -> Entry {
        Entry {
            id: id.to_owned(),
            tier: tier.to_owned(),
            set: set.to_owned(),
            file: format!("{id}.ARW"),
            sidecar: None,
            settings: no_settings(),
            variant: None,
            light: None,
            body: None,
            cfa: None,
            bits: None,
        }
    }

    #[test]
    fn sets_pick_their_tiers() {
        let m = Manifest {
            entries: vec![
                entry("a", "synthetic", "default"),
                entry("b", "synthetic", "full"),
                entry("c", "synthetic-eval", "eval"),
                entry("d", "cc0", ""),
                entry("e", "free", ""),
                entry("f", "eval", ""),
            ],
        };
        let ids = |s: SetChoice| {
            select(&m, s, &[])
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(SetChoice::Synthetic), ["a"]);
        assert_eq!(ids(SetChoice::SyntheticFull), ["a", "b"]);
        assert_eq!(ids(SetChoice::SyntheticEval), ["c"]);
        assert_eq!(ids(SetChoice::Cc0), ["d"]);
        assert_eq!(ids(SetChoice::Tier2), ["e"]);
        assert_eq!(ids(SetChoice::Tier3), ["f"]);
        assert_eq!(ids(SetChoice::All).len(), 6);
    }

    #[test]
    fn filters_all_have_to_match() {
        let m = Manifest {
            entries: vec![
                entry("synthetic/chart__D6500__ILCE-1", "synthetic", "default"),
                entry("synthetic/chart__D6500__ILCE-7M3", "synthetic", "default"),
                entry("synthetic/chart__bb2850__ILCE-1", "synthetic", "default"),
            ],
        };
        let got = select(
            &m,
            SetChoice::Synthetic,
            &["D6500".to_owned(), "ILCE-1".to_owned()],
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, "synthetic/chart__D6500__ILCE-1");
    }

    #[test]
    fn the_committed_manifest_loads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("manifest.json");
        let m = Manifest::load(&path).unwrap();
        assert!(!m.entries.is_empty());
        let default_set = select(&m, SetChoice::Synthetic, &[]);
        assert_eq!(
            default_set.len(),
            61,
            "the default synthetic set has 61 files"
        );
        assert!(
            default_set
                .iter()
                .all(|e| e.settings.as_object().is_some_and(|o| o.is_empty()))
        );
    }

    #[test]
    fn duplicate_and_unsafe_ids_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let write = |body: &str| {
            let p = dir.path().join("m.json");
            fs::write(&p, body).unwrap();
            Manifest::load(&p)
        };
        let one = |id: &str| format!(r#"{{"id":"{id}","tier":"cc0","file":"x.ARW"}}"#);
        assert!(write(&format!(r#"{{"entries":[{},{}]}}"#, one("a"), one("a"))).is_err());
        assert!(write(&format!(r#"{{"entries":[{}]}}"#, one("../x"))).is_err());
        assert!(write(&format!(r#"{{"entries":[{}]}}"#, one("/x"))).is_err());
        assert!(write(r#"{"entries":[{"id":"a","tier":"cc0","file":"x","settings":3}]}"#).is_err());
        assert!(write(&format!(r#"{{"entries":[{}]}}"#, one("ok/a"))).is_ok());
        assert!(write("{}").is_err());
    }
}
