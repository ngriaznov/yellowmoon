//! `golden`: golden-image tooling (`docs/13-testing.md`, section Golden images).
//!
//! `golden synth` writes the synthetic ARW set from the scenes of `training/synth`. The runner that renders the golden
//! set through the CLI and compares it with stored references (CIEDE2000) is a later task: running `golden` without a
//! subcommand prints a notice and exits with code 2.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use golden::synth::{self, SynthOptions, Variant};

/// Golden-image tooling.
#[derive(Debug, Parser)]
#[command(name = "golden", version)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Write ARW files from the synthetic scenes and check that each decodes back to its mosaic.
    Synth(SynthArgs),
}

#[derive(Debug, clap::Args)]
struct SynthArgs {
    /// Folder with the scenes written by `python3 -m training.synth.make_scenes`.
    #[arg(long, default_value = "testdata/synthetic/scenes")]
    scenes: PathBuf,
    /// Folder for the ARW files (created if missing).
    #[arg(long, default_value = "testdata/synthetic/arw")]
    out: PathBuf,
    /// Variants to write for every scene they fit: uncompressed14, packed12, craw, lossless-l, lossless-ms
    /// (comma separated). Default: the variants each scene's sidecar recommends.
    #[arg(long, value_delimiter = ',')]
    variants: Vec<Variant>,
    /// Only scenes whose id contains this text (repeatable).
    #[arg(long)]
    only: Vec<String>,
    /// Skip decoding each file again and comparing it with the mosaic.
    #[arg(long)]
    no_verify: bool,
    /// Worker threads (default: up to four).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.0} kB", b / 1e3)
    }
}

fn run_synth(a: SynthArgs) -> anyhow::Result<ExitCode> {
    let started = std::time::Instant::now();
    let opts = SynthOptions {
        variants: if a.variants.is_empty() {
            None
        } else {
            Some(a.variants)
        },
        only: a.only,
        no_verify: a.no_verify,
        jobs: a.jobs,
    };
    let report = synth::run(&a.scenes, &a.out, &opts)?;
    println!("{:<86} {:>9}  round trip", "file", "size");
    for w in &report.written {
        let mark = if w.unchanged { " (unchanged)" } else { "" };
        println!("{:<86} {:>9}  {}{mark}", w.file, human(w.bytes), w.verify);
    }
    for (scene, variant, reason) in &report.skipped {
        eprintln!("skipped {scene}__{variant}: {reason}");
    }
    for (file, message) in &report.failures {
        eprintln!("FAILED {file}: {message}");
    }
    let total: u64 = report.written.iter().map(|w| w.bytes).sum();
    println!(
        "{} files, {} in {}, {} skipped, {} failed, {:.0} s",
        report.written.len(),
        human(total),
        a.out.display(),
        report.skipped.len(),
        report.failures.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(if report.failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn main() -> anyhow::Result<ExitCode> {
    match Args::parse().command {
        Some(Command::Synth(a)) => run_synth(a),
        None => {
            eprintln!(
                "Not implemented yet: the render and compare runner is a later task. See `golden synth --help`."
            );
            Ok(ExitCode::from(2))
        }
    }
}
