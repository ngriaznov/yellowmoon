//! Built-in looks: Neutral in code, everything else from `data/looks/builtin/*.json` (`docs/08-looks.md` section Built-in looks).
//!
//! A built-in look is a parametric look file (`yellowmoon.parametric-look/1`, parsed and baked by
//! [`ym_color::look::ParametricLook`]): a luma tone curve, a CDL and hue-vs-sat, hue-vs-hue and
//! luma-vs-sat curves in LogC4, baked into a 65-point cube when the look is first used. The id is
//! `builtin:<file stem>`. `neutral.json` is the identity; Neutral exists even when the file or the
//! whole folder is missing. `signature.json` is the look new photos get out of the box (owner's
//! decisions of 2026-10-14 and 2026-10-19: a LogC4 luma curve and CDL saturation 1.45). It is
//! read like any other file here: listed when it is present, not listed when it is not, and a
//! folder without it leaves the default look at Neutral (`LookService::effective_default` in
//! `ym-core`).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use ym_color::look::{LookTransform, ParametricLook};

use super::ids::{LookId, slugify};

/// What a built-in look is made of.
#[derive(Debug, Clone)]
pub(crate) enum BuiltinKind {
    /// The identity.
    Identity,
    /// A parametric look, baked on first use.
    Parametric(Box<ParametricLook>),
}

/// A built-in look.
#[derive(Debug, Clone)]
pub(crate) struct Builtin {
    pub(crate) id: LookId,
    pub(crate) name: String,
    pub(crate) kind: BuiltinKind,
}

impl Builtin {
    /// Builds the transform (bakes the cube for a parametric look).
    pub(crate) fn transform(&self) -> Result<LookTransform, String> {
        match &self.kind {
            BuiltinKind::Identity => Ok(LookTransform::neutral()),
            BuiltinKind::Parametric(look) => look.to_transform().map_err(|e| e.to_string()),
        }
    }
}

/// A built-in look file that was skipped, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinProblem {
    /// The file.
    pub file: PathBuf,
    /// Why it was not loaded.
    pub reason: String,
}

/// The built-in looks found, and the files that were skipped.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuiltinSet {
    pub(crate) items: Vec<Builtin>,
    pub(crate) problems: Vec<BuiltinProblem>,
}

const NEUTRAL_SLUG: &str = "neutral";
const SIGNATURE_SLUG: &str = "signature";

fn neutral_default() -> Builtin {
    Builtin {
        id: LookId::builtin(NEUTRAL_SLUG),
        name: "Neutral".to_string(),
        kind: BuiltinKind::Identity,
    }
}

/// Loads the built-in looks from `dir` (when there is one). Never fails: a file that does not
/// parse is skipped and reported in [`BuiltinSet::problems`].
pub(crate) fn load(dir: Option<&Path>) -> BuiltinSet {
    let mut set = BuiltinSet::default();
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(dir) = dir
        && let Ok(entries) = fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_json = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"));
            if is_json && path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();

    let mut neutral: Option<Builtin> = None;
    // Files are visited in path order, so when two names make the same slug the first one that
    // loads wins and the other is reported. Two looks must never share an id.
    let mut slugs: HashSet<String> = HashSet::new();
    for path in files {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let slug = slugify(&stem);
        let mut problem = |reason: String| {
            set.problems.push(BuiltinProblem {
                file: path.clone(),
                reason,
            });
        };
        if slug.is_empty() {
            problem("the file name has no letters or digits".to_string());
            continue;
        }
        if slugs.contains(&slug) {
            problem(format!(
                "another file already makes the look builtin:{slug}, so this one was ignored"
            ));
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                problem(format!("could not be read: {e}"));
                continue;
            }
        };
        let look = match ParametricLook::from_json(&text) {
            Ok(l) => l,
            Err(e) => {
                problem(e.to_string());
                continue;
            }
        };
        slugs.insert(slug.clone());
        let id = LookId::builtin(&slug);
        if slug == NEUTRAL_SLUG {
            if !look.is_identity() {
                problem("Neutral must be the identity look, so the file was ignored".to_string());
                continue;
            }
            neutral = Some(Builtin {
                id,
                name: look.name().to_string(),
                kind: BuiltinKind::Identity,
            });
            continue;
        }
        set.items.push(Builtin {
            id,
            name: look.name().to_string(),
            kind: BuiltinKind::Parametric(Box::new(look)),
        });
    }

    // Order: Neutral, Signature, then the rest by id.
    set.items.push(neutral.unwrap_or_else(neutral_default));
    set.items.sort_by_key(|b| {
        let rank = match b.id.slug() {
            NEUTRAL_SLUG => 0,
            SIGNATURE_SLUG => 1,
            _ => 2,
        };
        (rank, b.id.clone())
    });
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) {
        fs::write(dir.join(name), text).unwrap();
    }

    const NEUTRAL: &str = r#"{"schema":"yellowmoon.parametric-look/1","name":"Neutral"}"#;

    #[test]
    fn neutral_exists_without_any_files() {
        let set = load(None);
        assert_eq!(set.items.len(), 1);
        assert_eq!(set.items[0].id.as_str(), "builtin:neutral");
        assert!(set.items[0].transform().unwrap().is_identity());

        let dir = tempfile::tempdir().unwrap();
        let set = load(Some(dir.path()));
        assert_eq!(set.items.len(), 1);
        let missing = load(Some(&dir.path().join("nope")));
        assert_eq!(missing.items.len(), 1);
        assert!(missing.problems.is_empty());
    }

    #[test]
    fn signature_is_listed_only_when_the_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "neutral.json", NEUTRAL);
        let before: Vec<_> = load(Some(dir.path()))
            .items
            .iter()
            .map(|b| b.id.to_string())
            .collect();
        assert_eq!(before, vec!["builtin:neutral"]);

        write(
            dir.path(),
            "signature.json",
            r#"{"schema":"yellowmoon.parametric-look/1","name":"Test Signature",
                "cdl":{"slope":[1.05,1.0,0.95],"offset":[0,0,0],"power":[1,1,1],"saturation":1.1},
                "hue_vs_sat":[{"hue_deg":30,"value":1.2}]}"#,
        );
        write(
            dir.path(),
            "aaa-extra.json",
            r#"{"schema":"yellowmoon.parametric-look/1","name":"Extra","cdl":{"slope":[1,1,1],"offset":[0.01,0,0],"power":[1,1,1],"saturation":1}}"#,
        );
        let set = load(Some(dir.path()));
        let ids: Vec<_> = set.items.iter().map(|b| b.id.to_string()).collect();
        assert_eq!(
            ids,
            vec!["builtin:neutral", "builtin:signature", "builtin:aaa-extra"]
        );
        assert_eq!(set.items[1].name, "Test Signature");
        let transform = set.items[1].transform().unwrap();
        assert!(!transform.is_identity());
        assert_eq!(
            transform.lut().unwrap().cube.size(),
            ym_color::lut::BAKE_SIZE
        );
    }

    #[test]
    fn bad_files_are_reported_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "broken.json", "{ not json");
        write(
            dir.path(),
            "neutral.json",
            r#"{"schema":"yellowmoon.parametric-look/1","name":"Neutral","cdl":{"slope":[2,1,1],"offset":[0,0,0],"power":[1,1,1],"saturation":1}}"#,
        );
        write(dir.path(), "notes.txt", "ignored");
        let set = load(Some(dir.path()));
        assert_eq!(set.problems.len(), 2);
        // Neutral is still the identity, from code.
        assert_eq!(set.items.len(), 1);
        assert!(set.items[0].transform().unwrap().is_identity());
    }
}
