//! `golden`: golden-image tooling (`docs/13-testing.md`, section Golden images).
//!
//! * `golden check`: render the golden files through `ym render --json --settings ...` and compare them with the stored
//!   pictures (ΔE00 per pixel on a 1/4 downsample: median, p99, max). Exit 1 on an unexplained change above the tolerance.
//! * `golden update --note "why"`: store new pictures and hashes (a note is required when an existing picture changes).
//! * `golden truth`: the end-to-end colour of the product against the ground truth of the synthetic scenes (a report).
//! * `golden synth`: write the synthetic ARW set from the scenes of `training/synth`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use golden::manifest::{Entry, Manifest, SetChoice, select};
use golden::runner::{self, CheckReport, Config, Metric, Status, UpdateKind, UpdateReport};
use golden::synth::{self, SynthOptions, Variant};
use golden::truth;
use golden::ymrun::{YmBinary, find_ym, workspace_root};

/// Golden-image tooling.
#[derive(Debug, Parser)]
#[command(name = "golden", version)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Render the golden files with `ym` and compare them with the stored pictures. Exit 0: nothing changed beyond the
    /// tolerance; 1: something did (or a file failed to render); 2: usage or setup error; 3: nothing was checked.
    Check(CheckArgs),
    /// Render the golden files and store the pictures and their hashes (`expected.json`). A change to an existing
    /// picture needs `--note`.
    Update(UpdateArgs),
    /// Compare the colour of the product with the ground truth of the synthetic scenes (a report, no gate).
    Truth(TruthArgs),
    /// Write ARW files from the synthetic scenes and check that each decodes back to its mosaic.
    Synth(SynthArgs),
}

/// What every runner command shares.
#[derive(Debug, clap::Args)]
struct Common {
    /// Which entries: synthetic (the default set), synthetic-full, synthetic-eval, cc0, tier2, tier3 or all.
    #[arg(long, value_enum, default_value = "synthetic")]
    set: SetChoice,
    /// Only entries whose id contains this text (repeatable; all of them must match).
    #[arg(long)]
    filter: Vec<String>,
    /// The manifest (default: tools/golden/manifest.json).
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// The test data folder (default: $YM_TESTDATA, else testdata/).
    #[arg(long)]
    testdata: Option<PathBuf>,
    /// The `ym` binary (default: $YM_BIN, else the newer of target/release/ym and target/debug/ym).
    #[arg(long)]
    ym: Option<PathBuf>,
    /// The data folder for `ym` ($YELLOWMOON_DATA); not needed for a debug build in the workspace.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Files rendered at the same time (default: half the cores).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    /// No progress lines on stderr.
    #[arg(long)]
    quiet: bool,
    /// Write the full result as JSON to this file.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct RunnerFiles {
    /// The reference pictures (default: <testdata>/golden/refs).
    #[arg(long)]
    refs: Option<PathBuf>,
    /// The expected hashes (default: tools/golden/expected.json).
    #[arg(long)]
    expected: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct CheckArgs {
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    files: RunnerFiles,
    /// The ΔE00 above which a changed picture fails.
    #[arg(long, default_value_t = 0.5)]
    tolerance: f64,
    /// The statistic held against the tolerance: max (default) or p99.
    #[arg(long, value_enum, default_value = "max")]
    metric: Metric,
    /// Give a picture that is identical to its expectation and has no reference file one (fills the CI cache).
    #[arg(long)]
    write_missing_refs: bool,
    /// Fail when an input file is not on disk instead of skipping it.
    #[arg(long)]
    require_all: bool,
    /// Pass (exit 0) when no file could be checked at all.
    #[arg(long)]
    allow_empty: bool,
}

#[derive(Debug, clap::Args)]
struct UpdateArgs {
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    files: RunnerFiles,
    /// Why the pictures changed (required when an existing picture changes; kept in expected.json).
    #[arg(long)]
    note: Option<String>,
    /// Drop expected entries that the manifest no longer lists.
    #[arg(long)]
    prune: bool,
}

#[derive(Debug, clap::Args)]
struct TruthArgs {
    #[command(flatten)]
    common: Common,
    /// Print every patch of every scene, not only the table.
    #[arg(long)]
    patches: bool,
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

/// The paths and the selected entries every runner command needs.
struct Setup {
    manifest: Manifest,
    selected_ids: Vec<String>,
    ym: YmBinary,
    testdata: PathBuf,
    refs: PathBuf,
    expected: PathBuf,
    jobs: usize,
    quiet: bool,
}

fn setup(common: &Common, files: Option<&RunnerFiles>) -> Result<Setup> {
    let root = workspace_root();
    let manifest_path = common
        .manifest
        .clone()
        .unwrap_or_else(|| root.join("tools/golden/manifest.json"));
    let manifest = Manifest::load(&manifest_path)?;
    let selected_ids: Vec<String> = select(&manifest, common.set, &common.filter)
        .iter()
        .map(|e| e.id.clone())
        .collect();
    let testdata = common
        .testdata
        .clone()
        .or_else(|| std::env::var_os("YM_TESTDATA").map(PathBuf::from))
        .unwrap_or_else(|| root.join("testdata"));
    let refs = files
        .and_then(|f| f.refs.clone())
        .unwrap_or_else(|| testdata.join("golden/refs"));
    let expected = files
        .and_then(|f| f.expected.clone())
        .unwrap_or_else(|| root.join("tools/golden/expected.json"));
    let ym = YmBinary {
        path: find_ym(common.ym.as_deref())?,
        data: common.data.clone(),
    };
    let jobs = if common.jobs == 0 {
        std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1))
    } else {
        common.jobs
    };
    Ok(Setup {
        manifest,
        selected_ids,
        ym,
        testdata,
        refs,
        expected,
        jobs,
        quiet: common.quiet,
    })
}

impl Setup {
    fn entries(&self) -> Vec<&Entry> {
        self.manifest
            .entries
            .iter()
            .filter(|e| self.selected_ids.contains(&e.id))
            .collect()
    }

    fn context(&self, tolerance: f64, metric: Metric) -> Config {
        Config {
            ym: self.ym.clone(),
            testdata: self.testdata.clone(),
            refs: self.refs.clone(),
            expected: self.expected.clone(),
            jobs: self.jobs,
            tolerance,
            metric,
            quiet: self.quiet,
        }
    }
}

fn write_report<T: serde::Serialize>(path: Option<&Path>, value: &T) -> Result<()> {
    if let Some(path) = path {
        let mut text = serde_json::to_string_pretty(value)?;
        text.push('\n');
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn print_check(report: &CheckReport) {
    println!(
        "{:<16} {:>7} {:>7} {:>7} {:>6}  id",
        "status", "median", "p99", "max", "sec"
    );
    for o in &report.outcomes {
        let (m, p, x) =
            o.delta_e_2000
                .as_ref()
                .map_or(("-".to_owned(), "-".to_owned(), "-".to_owned()), |d| {
                    (
                        format!("{:.3}", d.median),
                        format!("{:.3}", d.p99),
                        format!("{:.3}", d.max),
                    )
                });
        println!(
            "{:<16} {m:>7} {p:>7} {x:>7} {:>6.1}  {}",
            o.status.label(),
            o.seconds,
            o.id
        );
    }
    for o in report.outcomes.iter().filter(|o| !o.message.is_empty()) {
        if o.status != Status::Identical {
            println!("  {}: {}", o.id, o.message);
        }
    }
    let n = report.outcomes.len();
    println!(
        "{n} files: {} identical, {} within the tolerance of ΔE00 {} ({}), {} failed, {} not on disk",
        report.count(Status::Identical),
        report.count(Status::Within),
        report.tolerance,
        report.metric,
        report.outcomes.iter().filter(|o| o.status.failed()).count(),
        report.count(Status::Missing),
    );
}

fn run_check(a: CheckArgs) -> Result<ExitCode> {
    let s = setup(&a.common, Some(&a.files))?;
    let ctx = s.context(a.tolerance, a.metric);
    let entries = s.entries();
    if entries.is_empty() {
        eprintln!("golden: no entry of the manifest matches --set and --filter");
        return Ok(ExitCode::from(if a.allow_empty { 0 } else { 3 }));
    }
    let report = runner::check(&ctx, &entries, a.write_missing_refs)?;
    print_check(&report);
    write_report(a.common.report.as_deref(), &report)?;
    let mut failed = report.failed();
    if a.require_all && report.count(Status::Missing) > 0 {
        eprintln!("golden: --require-all: some input files are not on disk");
        failed = true;
    }
    if failed {
        return Ok(ExitCode::from(1));
    }
    if report.checked() == 0 {
        eprintln!("golden: nothing was checked: no input file of the selection is on disk");
        return Ok(ExitCode::from(if a.allow_empty { 0 } else { 3 }));
    }
    Ok(ExitCode::SUCCESS)
}

fn print_update(report: &UpdateReport) {
    for o in &report.outcomes {
        let kind = match o.kind {
            UpdateKind::Added => "added",
            UpdateKind::Unchanged => "unchanged",
            UpdateKind::Changed => "CHANGED",
        };
        match &o.delta_e_2000 {
            Some(d) => println!(
                "{kind:<10} {}  (ΔE00 median {:.3}, p99 {:.3}, max {:.3})",
                o.id, d.median, d.p99, d.max
            ),
            None => println!("{kind:<10} {}", o.id),
        }
    }
    for id in &report.missing {
        println!("not on disk {id}");
    }
    for id in &report.pruned {
        println!("pruned     {id}");
    }
    println!(
        "{} added, {} changed, {} unchanged, {} not on disk, {} pruned; expected.json {}",
        report.count(UpdateKind::Added),
        report.count(UpdateKind::Changed),
        report.count(UpdateKind::Unchanged),
        report.missing.len(),
        report.pruned.len(),
        if report.written {
            "written"
        } else {
            "not changed"
        }
    );
}

fn run_update(a: UpdateArgs) -> Result<ExitCode> {
    let s = setup(&a.common, Some(&a.files))?;
    let ctx = s.context(0.5, Metric::Max);
    let entries = s.entries();
    if entries.is_empty() {
        eprintln!("golden: no entry of the manifest matches --set and --filter");
        return Ok(ExitCode::from(3));
    }
    let report = runner::update(&ctx, &s.manifest, &entries, a.note.as_deref(), a.prune)?;
    print_update(&report);
    write_report(a.common.report.as_deref(), &report)?;
    if report.outcomes.is_empty() {
        eprintln!("golden: nothing was rendered: no input file of the selection is on disk");
        return Ok(ExitCode::from(3));
    }
    Ok(ExitCode::SUCCESS)
}

fn run_truth(a: TruthArgs) -> Result<ExitCode> {
    let s = setup(&a.common, None)?;
    let all = s.entries();
    let present: Vec<&Entry> = all
        .iter()
        .copied()
        .filter(|e| s.testdata.join(&e.file).is_file() && e.sidecar.is_some())
        .collect();
    let scenes = truth::one_per_scene(&present);
    if scenes.is_empty() {
        eprintln!(
            "golden: no scene with a sidecar and a file on disk matches (run tools/fetch/synthetic.sh)"
        );
        return Ok(ExitCode::from(3));
    }
    let mut results = Vec::new();
    let mut failures = Vec::new();
    for (i, entry) in scenes.iter().enumerate() {
        if !s.quiet {
            eprintln!("[{}/{}] {}", i + 1, scenes.len(), entry.id);
        }
        match truth::scene_result(&s.ym, &s.testdata, entry) {
            Ok(r) => results.push(r),
            Err(e) => failures.push(format!("{}: {e:#}", entry.id)),
        }
    }
    println!("{}", truth::format_table(&results));
    println!(
        "ΔE00 of the 190 chart patches against the ALEXA-estimate truth, after exposure scaling by the 18 % grey patch:"
    );
    println!(
        "med, p95, max in scene-linear AWG4 (the colour chain up to the look and curve); b.* the baseline, Sony camera RGB"
    );
    println!(
        "through the same ARRI matrix with no sensor map; d.* the render after the ACES 2 display transform on both sides;"
    );
    println!("neutr the median of the neutral ramp.");
    let lights = truth::pool(&results, |r| {
        r.body == "ILCE-7RM4" && r.iso == 400 && r.light != "mixed"
    });
    let all_scenes = truth::pool(&results, |_| true);
    for (name, pooled) in [
        ("A7R IV, ISO 400, single lights", &lights),
        ("every scene", &all_scenes),
    ] {
        if let (Some(r), Some(b), Some(d)) = (pooled.render, pooled.baseline, pooled.render_display)
        {
            println!(
                "pooled, {name} ({} scenes, {} patches): render median {:.2} p95 {:.2} max {:.2}; baseline median {:.2} p95 {:.2} max {:.2}; \
                 after the display transform median {:.2} p95 {:.2} max {:.2}",
                pooled.scenes,
                r.count,
                r.median,
                r.p95,
                r.max,
                b.median,
                b.p95,
                b.max,
                d.median,
                d.p95,
                d.max
            );
        }
    }
    if a.patches {
        for r in &results {
            println!("\n{}", r.id);
            for p in &r.patches {
                println!(
                    "  {:<22} {:<9} render {:6.2}  baseline {:6.2}{}",
                    p.name,
                    p.group,
                    p.render,
                    p.baseline,
                    if p.clipped {
                        "  (clipped, not counted)"
                    } else {
                        ""
                    }
                );
            }
        }
    }
    write_report(
        a.common.report.as_deref(),
        &serde_json::json!({ "scenes": results, "pooled_a7r4_iso400_single_lights": lights, "pooled_all": all_scenes, "failures": failures }),
    )?;
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAILED {f}");
        }
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args.command {
        Command::Synth(a) => run_synth(a),
        Command::Check(a) => run_check(a),
        Command::Update(a) => run_update(a),
        Command::Truth(a) => run_truth(a),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("golden: {e:#}");
            // A refused update (a failed render, a change without a note) is a failed run; the rest is setup.
            ExitCode::from(if e.downcast_ref::<runner::Refused>().is_some() {
                1
            } else {
                2
            })
        }
    }
}
