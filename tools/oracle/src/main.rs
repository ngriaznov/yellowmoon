//! Decoder oracle (`docs/04-raw-decoding.md` section Tests, `docs/13-testing.md`).
//!
//! Test-only. Links rawler (LGPL-2.1) and runs LibRaw (through `rawpy_compare.py`) to compare our
//! decoder's output with two independent ones. It must never become a dependency of a shipped
//! crate.
//!
//! ```text
//! oracle --corpus testdata/arw --report oracle-report.json     # compare every *.ARW below the folder
//! oracle dump FILE.ARW --out raw.bin                           # our decode as little-endian u16
//! ```
//!
//! # What is compared
//!
//! * **Pixels.** `ym-raw`'s `data` against LibRaw's `raw_image` (undithered, the whole sensor) and
//!   against rawler's decode. Lossless variants must be identical. rawler dithers cRAW output
//!   with a pseudo-random offset of at most a quarter of the local curve step, so for cRAW the
//!   comparison with rawler is "within the dither interval"; LibRaw's cRAW output is exact.
//!   Lossless M and S are compared with LibRaw's pseudo-raw RGB (the chroma upsampling and white
//!   balance steps that LibRaw does not have switched off) and only reported for rawler, whose
//!   YCbCr matrix differs.
//! * **Levels.** Black level, white level and white balance, per colour.
//!
//! Exit codes: 0 every comparison passed, 1 at least one mismatch, 2 a comparison could not run
//! (an oracle is missing or failed to start, or LibRaw refused every file that ym-raw decoded) or
//! the corpus is unreadable. With an empty corpus the report is empty and the exit code is 0.

mod compare;
mod libraw;
mod rawler_side;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use ym_raw::decode::{ChromaUpsampling, DecodeOptions, YcbcrOptions, decode_with};

/// Compares ym-raw decodes with rawler and LibRaw on a corpus of ARW files.
#[derive(Debug, Parser)]
#[command(name = "oracle", version)]
struct Args {
    /// Folder holding the corpus; searched recursively for `*.ARW` (case-insensitive).
    #[arg(long, value_name = "DIR", requires = "report")]
    corpus: Option<PathBuf>,
    /// Where to write the JSON report.
    #[arg(long, value_name = "FILE", requires = "corpus")]
    report: Option<PathBuf>,
    /// The Python interpreter that has rawpy installed.
    #[arg(long, default_value = "python3")]
    python: String,
    /// The LibRaw dump script (default: `rawpy_compare.py` next to this crate).
    #[arg(long, value_name = "FILE")]
    script: Option<PathBuf>,
    /// Do not compare with LibRaw.
    #[arg(long)]
    no_libraw: bool,
    /// Do not compare Lossless L tiles with libjpeg (pylibjpeg-libjpeg; skipped when missing).
    #[arg(long)]
    no_libjpeg: bool,
    /// Do not compare with rawler.
    #[arg(long)]
    no_rawler: bool,
    /// Keep LibRaw's pixel dumps in this folder (default: a temporary folder that is removed).
    #[arg(long, value_name = "DIR")]
    keep: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Writes our decode of one file as little-endian u16 samples plus a JSON sidecar.
    Dump {
        /// The ARW file.
        file: PathBuf,
        /// Output file for the samples; the sidecar is `<out>.json`.
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        /// Lossless M/S: replicate chroma instead of interpolating it (what LibRaw does).
        #[arg(long)]
        replicate_chroma: bool,
        /// Lossless M/S: leave the camera's white balance in the data.
        #[arg(long)]
        keep_white_balance: bool,
    },
}

fn main() -> Result<ExitCode> {
    let args = Args::parse();
    match args.command {
        Some(Command::Dump {
            ref file,
            ref out,
            replicate_chroma,
            keep_white_balance,
        }) => {
            dump(file, out, replicate_chroma, keep_white_balance)?;
            Ok(ExitCode::SUCCESS)
        }
        None => {
            let (Some(corpus), Some(report)) = (args.corpus.as_deref(), args.report.as_deref())
            else {
                bail!("give --corpus DIR --report FILE, or a subcommand (see --help)");
            };
            run_corpus(&args, corpus, report)
        }
    }
}

/// `oracle dump`.
fn dump(file: &Path, out: &Path, replicate: bool, keep_wb: bool) -> Result<()> {
    let options = DecodeOptions {
        ycbcr: YcbcrOptions {
            chroma: if replicate {
                ChromaUpsampling::Replicate
            } else {
                ChromaUpsampling::Linear
            },
            undo_white_balance: !keep_wb,
        },
    };
    let image =
        decode_with(file, &options).with_context(|| format!("decoding {}", file.display()))?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let bytes: Vec<u8> = image
        .data
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    let sidecar = compare::sidecar(file, &image, &bytes);
    let mut sidecar_path = out.as_os_str().to_owned();
    sidecar_path.push(".json");
    fs::write(
        &sidecar_path,
        serde_json::to_string_pretty(&sidecar)? + "\n",
    )
    .context("writing the JSON sidecar")?;
    println!("wrote {} and its sidecar", out.display());
    Ok(())
}

/// `oracle --corpus DIR --report FILE`.
fn run_corpus(args: &Args, corpus: &Path, report_path: &Path) -> Result<ExitCode> {
    let files = find_arw_files(corpus)
        .with_context(|| format!("reading corpus folder {}", corpus.display()))?;

    let libraw_runner = if args.no_libraw || files.is_empty() {
        None
    } else {
        match libraw::Runner::new(
            libraw::Tool::LibRaw,
            &args.python,
            args.script.clone(),
            args.keep.clone(),
        ) {
            Ok(runner) => Some(runner),
            Err(e) => {
                // A comparison that cannot run is a failure, never a pass.
                eprintln!("LibRaw is not available: {e:#}");
                return Ok(ExitCode::from(2));
            }
        }
    };
    // libjpeg is an extra oracle for the lossless tiles; it is used when installed.
    let libjpeg_runner = if args.no_libjpeg || files.is_empty() {
        None
    } else {
        match libraw::Runner::new(libraw::Tool::Libjpeg, &args.python, None, args.keep.clone()) {
            Ok(runner) => Some(runner),
            Err(e) => {
                eprintln!("libjpeg oracle skipped: {e:#}");
                None
            }
        }
    };
    let mut entries: Vec<Value> = Vec::new();
    let mut tally = compare::Tally::default();
    for path in &files {
        let entry = compare::compare_file(
            path,
            libraw_runner.as_ref(),
            libjpeg_runner.as_ref(),
            !args.no_rawler,
            &mut tally,
        );
        eprintln!("{}", compare::one_line(&entry));
        entries.push(entry);
    }

    let report = json!({
        "corpus": corpus.display().to_string(),
        "implemented": true,
        "libraw": libraw_runner.as_ref().map(|r| r.version().to_owned()),
        "libjpeg": libjpeg_runner.as_ref().map(|r| r.version().to_owned()),
        "rawler": if args.no_rawler { Value::Null } else { json!("0.8") },
        "files": entries,
        "summary": tally.summary(files.len()),
    });
    if let Some(parent) = report_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(report_path, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", report_path.display()))?;
    eprintln!("{}", tally.table());

    if tally.mismatched() > 0 {
        return Ok(ExitCode::from(1));
    }
    // A comparison that could not run is a failure, never a pass: if LibRaw was asked and refused
    // every file we decoded (a broken rawpy install, a changed output format), nothing was
    // compared and the green result would mean nothing.
    if libraw_runner.is_some() && tally.ours_decoded() > 0 && tally.libraw_compared() == 0 {
        eprintln!(
            "LibRaw could not decode any of the {} files that ym-raw decoded: nothing was compared",
            tally.ours_decoded()
        );
        return Ok(ExitCode::from(2));
    }
    Ok(ExitCode::SUCCESS)
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
