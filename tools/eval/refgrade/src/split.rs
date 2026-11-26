//! The local evaluation set: `split.json`, and the seeded sample of held-out photos for the owner's eyes.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

use crate::measure::Rng;

/// The seed of the owner's sample of held-out photos.
pub const OWNER_SAMPLE_SEED: u64 = 20_261_124;
/// How many held-out photos go to the owner.
pub const OWNER_SAMPLE_LEN: usize = 12;

/// One photo of the split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitEntry {
    /// The raw file.
    pub raw: PathBuf,
    /// `dev` or `test`.
    pub set: String,
    /// The content tags.
    pub tags: Vec<String>,
    /// The first content tag, for the per-tag tables.
    pub primary_tag: String,
    /// The folder of the label maps (`<labels>/labels/photo__<stem>.png`), when the file gives one.
    pub labels: Option<PathBuf>,
}

impl SplitEntry {
    /// The photo's name in the records: the raw file's stem.
    #[must_use]
    pub fn name(&self) -> String {
        self.raw
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// The label map's file, when the split names a folder for it.
    #[must_use]
    pub fn label_file(&self) -> Option<PathBuf> {
        self.labels
            .as_ref()
            .map(|d| d.join("labels").join(format!("photo__{}.png", self.name())))
    }
}

/// Reads `split.json`: an object with an `items` (or `photos`) array, or an array, of entries with `raw` (or `path`), `set`,
/// `tags` and optionally `primary_tag` and `labels`.
///
/// # Errors
///
/// A file that cannot be read or has no list of photos.
pub fn read_split(path: &Path) -> Result<Vec<SplitEntry>> {
    let v: Value = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("{}", path.display()))?,
    )?;
    parse_split(&v)
}

/// The entries of a parsed split.
///
/// # Errors
///
/// No list of photos, or an entry with no `raw` path.
pub fn parse_split(v: &Value) -> Result<Vec<SplitEntry>> {
    let arr = v
        .get("items")
        .or_else(|| v.get("photos"))
        .and_then(Value::as_array)
        .or_else(|| v.as_array())
        .ok_or_else(|| anyhow!("split.json: expected an array of photos"))?;
    let mut out = Vec::new();
    for e in arr {
        let raw = e
            .get("raw")
            .or_else(|| e.get("path"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("split.json: an entry has no `raw` path"))?;
        let set = e
            .get("set")
            .and_then(Value::as_str)
            .unwrap_or("dev")
            .to_string();
        let tags: Vec<String> = e
            .get("tags")
            .and_then(Value::as_array)
            .map(|t| {
                t.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let primary_tag = e
            .get("primary_tag")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| tags.first().cloned())
            .unwrap_or_else(|| "untagged".into());
        let labels = e.get("labels").and_then(Value::as_str).map(PathBuf::from);
        out.push(SplitEntry {
            raw: PathBuf::from(raw),
            set,
            tags,
            primary_tag,
            labels,
        });
    }
    Ok(out)
}

/// A seeded shuffle (Fisher-Yates) of the indices `0..n`.
#[must_use]
pub fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// The names of the held-out photos whose renders go to the owner: a seeded sample of [`OWNER_SAMPLE_LEN`] of the `test` entries,
/// in the order of the split.
#[must_use]
pub fn owner_sample(entries: &[SplitEntry]) -> Vec<String> {
    let test: Vec<&SplitEntry> = entries.iter().filter(|e| e.set == "test").collect();
    let mut picked: Vec<usize> = shuffled(test.len(), OWNER_SAMPLE_SEED)
        .into_iter()
        .take(OWNER_SAMPLE_LEN)
        .collect();
    picked.sort_unstable();
    picked.into_iter().map(|k| test[k].name()).collect()
}

/// The reference files of a folder (`jpg`, `jpeg`, `png`, `tif`, `tiff`), sorted by name.
///
/// # Errors
///
/// A folder that cannot be read.
pub fn reference_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("{}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| {
                matches!(
                    x.to_string_lossy().to_ascii_lowercase().as_str(),
                    "jpg" | "jpeg" | "png" | "tif" | "tiff"
                )
            })
        })
        .collect();
    v.sort();
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(i: usize, set: &str) -> Value {
        json!({ "raw": format!("/x/p{i:03}.ARW"), "set": set, "tags": ["landscape", "water"],
                "labels": "/x/labels" })
    }

    #[test]
    fn reads_the_items_with_the_first_tag_as_primary() {
        let v = json!({ "method": "m", "items": [entry(1, "dev"), { "raw": "/y/q.ARW", "set": "test",
            "primary_tag": "people", "tags": ["x"] }] });
        let e = parse_split(&v).expect("split");
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].primary_tag, "landscape");
        assert_eq!(e[0].name(), "p001");
        assert_eq!(
            e[0].label_file(),
            Some(PathBuf::from("/x/labels/labels/photo__p001.png"))
        );
        assert_eq!(e[1].primary_tag, "people");
        assert_eq!(e[1].label_file(), None);
        assert!(parse_split(&json!({"nothing": 1})).is_err());
        assert!(parse_split(&json!([{ "set": "dev" }])).is_err());
    }

    #[test]
    fn the_owner_sample_is_seeded_distinct_and_only_held_out() {
        let items: Vec<Value> = (0..40)
            .map(|i| entry(i, if i % 3 == 0 { "dev" } else { "test" }))
            .collect();
        let e = parse_split(&Value::Array(items)).expect("split");
        let a = owner_sample(&e);
        assert_eq!(a.len(), 12);
        assert_eq!(a, owner_sample(&e));
        let mut d = a.clone();
        d.dedup();
        assert_eq!(d.len(), 12);
        for name in &a {
            let i: usize = name[1..].parse().expect("number");
            assert_ne!(i % 3, 0, "{name} is a dev photo");
        }
        // A permutation, whatever the seed.
        let mut s = shuffled(50, 7);
        s.sort_unstable();
        assert_eq!(s, (0..50).collect::<Vec<_>>());
    }
}
