//! Decoder oracle (`docs/04-raw-decoding.md` section Tests, `docs/13-testing.md`).
//!
//! Test-only. Links rawler (LGPL-2.1) to compare our decoder's output with an independent one.
//! It must never become a dependency of a shipped crate.
//!
//! The comparison itself is not implemented yet. For now the tool walks the corpus for `*.ARW`
//! files and writes a JSON report. With no files the report is empty and the exit code is 0. With
//! files present it lists them as `not_compared` and exits with code 2, so a CI job cannot pass
//! on a comparison that did not happen.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;

/// Compares ym-raw decodes with rawler on a corpus of ARW files.
#[derive(Debug, Parser)]
#[command(name = "oracle", version)]
struct Args {
    /// Folder holding the corpus; searched recursively for `*.ARW` (case-insensitive).
    #[arg(long, value_name = "DIR")]
    corpus: PathBuf,
    /// Where to write the JSON report.
    #[arg(long, value_name = "FILE")]
    report: PathBuf,
}

fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let files = find_arw_files(&args.corpus)
        .with_context(|| format!("reading corpus folder {}", args.corpus.display()))?;

    let entries: Vec<_> = files
        .iter()
        .map(|path| json!({ "path": path.display().to_string(), "status": "not_compared" }))
        .collect();
    let report = json!({
        "corpus": args.corpus.display().to_string(),
        "implemented": false,
        "files": entries,
        "summary": { "total": files.len(), "compared": 0, "mismatched": 0 },
    });
    if let Some(parent) = args.report.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", args.report.display()))?;

    if files.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("Not implemented yet.");
        Ok(ExitCode::from(2))
    }
}

/// Returns every `*.ARW` file below `dir`, sorted, matching the extension case-insensitively.
fn find_arw_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("arw"))
            {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_arw_files_recursively_and_ignores_other_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path();
        let nested = dir.join("a/b");
        fs::create_dir_all(&nested).expect("create test folders");
        for name in [
            "one.ARW",
            "two.arw",
            "three.txt",
            "a/four.Arw",
            "a/b/five.jpg",
        ] {
            fs::write(dir.join(name), b"").expect("create test file");
        }

        let names: Vec<String> = find_arw_files(dir)
            .expect("walk corpus")
            .iter()
            .map(|p| {
                p.strip_prefix(dir)
                    .expect("under corpus")
                    .display()
                    .to_string()
            })
            .collect();

        assert_eq!(names, ["a/four.Arw", "one.ARW", "two.arw"]);
    }
}
