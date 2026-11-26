//! `refgrade-eval`: run the product's Reference Grade fit on the local evaluation set, merge shards, compare two runs.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use refgrade_eval::compare::compare;
use refgrade_eval::report::{merge, read_records, write_mask_summary};
use refgrade_eval::run::{RunOpts, SceneMode, default_renders, run};
use refgrade_eval::split::{owner_sample, read_split};

#[derive(Parser)]
#[command(
    name = "refgrade-eval",
    about = "The Reference Grade evaluation tool (docs/34 section 10.1)"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fits every photo of a set to every reference and measures the harms and the look transfer.
    Run {
        /// The local `split.json`.
        #[arg(long)]
        split: PathBuf,
        /// The folder of reference stills.
        #[arg(long)]
        references: PathBuf,
        /// Where the shard's files go.
        #[arg(long)]
        out: PathBuf,
        /// `dev`, `test` or `all`.
        #[arg(long, default_value = "dev")]
        set: String,
        /// This process's share, `i/n`: the photos whose position in the list is `i` modulo `n`.
        #[arg(long, default_value = "0/1")]
        shard: String,
        /// Threads of this process.
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Keep only these photos (file stems, separated by commas).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Stop after this many photos (a trial).
        #[arg(long)]
        max_photos: Option<usize>,
        /// The `data/` folder (default: as `ym` finds it, or `YELLOWMOON_DATA`).
        #[arg(long)]
        data: Option<PathBuf>,
        /// Carry on a run that was stopped: skip the fits already in `records.jsonl`.
        #[arg(long)]
        resume: bool,
        /// Do not compare the product's element map with the split's label maps.
        #[arg(long)]
        no_mask_check: bool,
        /// Do not render each photo's first fit through the real pipeline for the audit.
        #[arg(long)]
        no_audit: bool,
        /// The ablation of `docs/34` section 16.10 (dev only): `product` (the default), `poc-labels` (A1: the label maps of the proof
        /// of concept for the photos and the references) or `default-render` (A2: the segmenter on the photo's default render).
        #[arg(long, default_value = "product")]
        scene: String,
        /// With `--scene poc-labels`: the folder of the references' label maps.
        #[arg(long)]
        reference_labels: Option<PathBuf>,
    },
    /// Joins the shards of a run and writes `metrics.json`, `summary.json`, `summary.md` and the mask summary.
    Merge {
        /// Where the joined files go.
        #[arg(long)]
        out: PathBuf,
        /// The run's name in the summary.
        #[arg(long, default_value = "product")]
        name: String,
        /// The shard folders.
        runs: Vec<PathBuf>,
    },
    /// Compares a new run with a baseline under the three acceptance rules.
    Compare {
        /// The baseline's `metrics.json` (or its folder).
        #[arg(long)]
        baseline: PathBuf,
        /// The new run's `metrics.json` (or its folder).
        #[arg(long)]
        new: PathBuf,
        /// Only the fits of this set (default: all).
        #[arg(long)]
        set: Option<String>,
        /// Where `compare.md` and `compare.json` go (default: print only).
        #[arg(long)]
        out: Option<PathBuf>,
        /// The baseline's name in the tables.
        #[arg(long, default_value = "baseline")]
        baseline_name: String,
        /// The new run's name in the tables.
        #[arg(long, default_value = "new")]
        new_name: String,
    },
    /// Renders each photo's default (no node) through the real pipeline as a png, for a side-by-side check of the photos' own pictures.
    Renders {
        /// The local `split.json`.
        #[arg(long)]
        split: PathBuf,
        /// Where the pngs go.
        #[arg(long)]
        out: PathBuf,
        /// `dev`, `test` or `all`.
        #[arg(long, default_value = "dev")]
        set: String,
        /// Threads of this process.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Keep only these photos (file stems, separated by commas).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// The `data/` folder.
        #[arg(long)]
        data: Option<PathBuf>,
    },
    /// Prints the seeded sample of held-out photos that go to the owner (the numbers on the held-out set stay aggregate).
    Sample {
        /// The local `split.json`.
        #[arg(long)]
        split: PathBuf,
    },
}

fn parse_shard(s: &str) -> Result<(usize, usize)> {
    let (i, n) = s.split_once('/').context("--shard takes i/n")?;
    let (i, n): (usize, usize) = (i.parse()?, n.parse()?);
    if n == 0 || i >= n {
        bail!("--shard i/n needs i < n");
    }
    Ok((i, n))
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Run {
            split,
            references,
            out,
            set,
            shard,
            threads,
            only,
            max_photos,
            data,
            resume,
            no_mask_check,
            no_audit,
            scene,
            reference_labels,
        } => run(&RunOpts {
            split,
            set,
            references,
            out,
            shard: parse_shard(&shard)?,
            threads,
            only,
            max_photos,
            data,
            resume,
            mask_check: !no_mask_check,
            audit: !no_audit,
            scene: SceneMode::parse(&scene)?,
            reference_labels,
        }),
        Cmd::Merge { out, name, runs } => {
            let n = merge(&runs, &name, &out)?;
            eprintln!("{n} fits merged into {}", out.display());
            if let Some(p) = write_mask_summary(&runs, &out)? {
                eprintln!(
                    "mask check of {p} photos in {}",
                    out.join("masks.md").display()
                );
            }
            Ok(())
        }
        Cmd::Compare {
            baseline,
            new,
            set,
            out,
            baseline_name,
            new_name,
        } => {
            let v = compare(
                &read_records(&baseline)?,
                &read_records(&new)?,
                set.as_deref(),
                &baseline_name,
                &new_name,
            );
            println!("{}", v.markdown);
            if let Some(out) = out {
                fs::create_dir_all(&out)?;
                fs::write(out.join("compare.md"), &v.markdown)?;
                fs::write(
                    out.join("compare.json"),
                    serde_json::to_string_pretty(&v.json)?,
                )?;
            }
            // The exit code is the verdict: a run that fails a line is reported, not hidden.
            if v.passes_rg10() {
                Ok(())
            } else {
                std::process::exit(3)
            }
        }
        Cmd::Renders {
            split,
            out,
            set,
            threads,
            only,
            data,
        } => default_renders(&RunOpts {
            split,
            set,
            references: PathBuf::new(),
            out,
            shard: (0, 1),
            threads,
            only,
            max_photos: None,
            data,
            resume: false,
            mask_check: false,
            audit: false,
            scene: SceneMode::Product,
            reference_labels: None,
        }),
        Cmd::Sample { split } => {
            for name in owner_sample(&read_split(&split)?) {
                println!("{name}");
            }
            Ok(())
        }
    }
}
