//! Helpers shared by the integration tests of `ym-profiles`.
#![allow(dead_code)] // each test file uses a subset

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_profiles::registry::ProfileStore;

/// `<workspace>/data`, the shipped data root.
pub fn data_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The shipped store (loaded once per test binary).
pub fn store() -> &'static ProfileStore {
    static STORE: std::sync::OnceLock<ProfileStore> = std::sync::OnceLock::new();
    STORE.get_or_init(|| ProfileStore::load(data_root()).expect("the shipped data loads"))
}

pub fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// One row of the table "Which profile each camera gets" in `docs/03-camera-profiles.md`:
/// model ID -> (profile id, `exact` | `sensor_family` | `default`).
pub fn docs_table() -> BTreeMap<u16, (String, String)> {
    let text = std::fs::read_to_string(repo_root().join("docs/03-camera-profiles.md"))
        .expect("docs/03 exists");
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let cells: Vec<&str> = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        if cells.len() != 3 || !cells[1].starts_with("sony-") {
            continue;
        }
        let kind = match cells[2] {
            "exact" => "exact",
            "same sensor" => "sensor_family",
            "default" => "default",
            other => panic!("unknown match cell {other:?} in docs/03"),
        };
        // Model IDs are the numbers in parentheses in the first cell: "ILCE-7M3 A7 III (363)".
        let mut rest = cells[0];
        while let Some(open) = rest.find('(') {
            let after = &rest[open + 1..];
            let close = after.find(')').expect("closing parenthesis");
            if let Ok(id) = after[..close].parse::<u16>() {
                out.insert(id, (cells[1].to_string(), kind.to_string()));
            }
            rest = &after[close + 1..];
        }
    }
    out
}
