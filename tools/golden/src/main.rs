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
use clap::{Parser, Subcommand, ValueEnum};
use golden::manifest::{Entry, Manifest, SetChoice, select};
use golden::runner::{
    self, CheckReport, Config, GpuGate, Metric, Reference, Status, UpdateKind, UpdateReport,
};
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
    /// Render real photos on the CPU and, strictly, on the graphics device, and hold the two pictures against the
    /// device gates (the owner photos of CI, a folder of local files; no manifest, no expected.json).
    DeviceCheck(DeviceCheckArgs),
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
    /// Only the entries whose id is listed in this file (one id per line, `#` starts a comment), after `--set` and
    /// `--filter`: `tools/golden/gpu-subset.txt` is the handful of files a push checks on the device.
    #[arg(long)]
    subset: Option<PathBuf>,
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
    /// Where the pictures are made: `cpu` (the default: the reference the hashes of expected.json pin) or `gpu`
    /// (`ym render --require-gpu`: a picture that the device did not make is a failure, and the pictures are held
    /// against the CPU's by the gates below instead of by the hash and the tolerance).
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,
    /// With `--device gpu`: what the device picture is compared with. `stored`: the reference pictures (the CI cache);
    /// `render`: a CPU render of the same run and machine (a Mac, whose CPU path is not the one expected.json was
    /// written on).
    #[arg(long, value_enum, default_value = "stored")]
    reference: Reference,
    /// With `--device gpu`: which set of gates is the starting point: `calibrated` (the default: twice the first run
    /// on lavapipe) or `plan` (the first proposal, made before any GPU picture was measured). The four options below
    /// replace single values.
    #[arg(long, value_enum, default_value = "calibrated")]
    gpu_gates: GateSet,
    /// With `--device gpu`: the median ΔE00 may not exceed this.
    #[arg(long)]
    gpu_median: Option<f64>,
    /// With `--device gpu`: the 99th percentile of the ΔE00 may not exceed this.
    #[arg(long)]
    gpu_p99: Option<f64>,
    /// With `--device gpu`: the maximum ΔE00 may not exceed this (`--tolerance` is for the CPU mode).
    #[arg(long)]
    gpu_max: Option<f64>,
    /// With `--device gpu`: the share of the full-resolution samples that differ by more than 8 codes of 16 bits may not
    /// exceed this.
    #[arg(long)]
    gpu_codes_share: Option<f64>,
}

/// `--gpu-gates`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum GateSet {
    /// Twice the first lavapipe run ([`GpuGate::CALIBRATED`]).
    Calibrated,
    /// The first proposal ([`GpuGate::PLAN`]).
    Plan,
}

#[derive(Debug, clap::Args)]
struct DeviceCheckArgs {
    /// The raw files.
    #[arg(required = true)]
    files: Vec<PathBuf>,
    /// The `ym` binary (default: $YM_BIN, else the newer of target/release/ym and target/debug/ym).
    #[arg(long)]
    ym: Option<PathBuf>,
    /// The data folder for `ym` ($YELLOWMOON_DATA); not needed for a debug build in the workspace.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Scale the long edge of both pictures to this many pixels (the raw chain still runs at full resolution, so the
    /// device and the CPU do the same work; the comparison is quick). 0 compares the full-size pictures.
    #[arg(long, default_value_t = 1500)]
    long_edge: u32,
    /// Files rendered at the same time (default 1: a real photo is large).
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Which set of gates is the starting point: `calibrated` or `plan`.
    #[arg(long, value_enum, default_value = "calibrated")]
    gpu_gates: GateSet,
    /// The median ΔE00 may not exceed this.
    #[arg(long)]
    gpu_median: Option<f64>,
    /// The 99th percentile of the ΔE00 may not exceed this.
    #[arg(long)]
    gpu_p99: Option<f64>,
    /// The maximum ΔE00 may not exceed this.
    #[arg(long)]
    gpu_max: Option<f64>,
    /// The share of the full-resolution samples more than 8 codes of 16 bits off may not exceed this.
    #[arg(long)]
    gpu_codes_share: Option<f64>,
    /// No progress lines on stderr.
    #[arg(long)]
    quiet: bool,
    /// Write the full result as JSON to this file.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// `--device`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
    /// The CPU reference.
    Cpu,
    /// The graphics device, strictly.
    Gpu,
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

/// The ids of a subset file: one per line, blank lines and `#` comments ignored. An id the manifest does not have is
/// an error (a typo would otherwise shrink the subset without a word).
fn read_subset(path: &Path, manifest: &Manifest) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the subset {}", path.display()))?;
    let ids = parse_subset(&text);
    anyhow::ensure!(!ids.is_empty(), "{} lists no id", path.display());
    for id in &ids {
        anyhow::ensure!(
            manifest.entries.iter().any(|e| &e.id == id),
            "{}: the manifest has no entry {id:?}",
            path.display()
        );
    }
    Ok(ids)
}

fn parse_subset(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
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
    let mut selected_ids: Vec<String> = select(&manifest, common.set, &common.filter)
        .iter()
        .map(|e| e.id.clone())
        .collect();
    if let Some(path) = &common.subset {
        let wanted = read_subset(path, &manifest)?;
        // An id that `--set` leaves out would be dropped without a word, and a push would check fewer files than the
        // subset file lists; `--filter` narrows on purpose and is not held to this.
        let in_set: Vec<&str> = select(&manifest, common.set, &[])
            .iter()
            .map(|e| e.id.as_str())
            .collect();
        if let Some(id) = wanted.iter().find(|id| !in_set.contains(&id.as_str())) {
            anyhow::bail!(
                "{}: {id:?} is in the manifest but not in the selected set (--set), so it would be left out of the subset",
                path.display()
            );
        }
        selected_ids.retain(|id| wanted.contains(id));
    }
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
    if let Some(gpu) = &report.gpu {
        print_gpu_summary(report, gpu);
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

/// The lines of a GPU run: the adapter, the gates, and the worst of every statistic over the files.
fn print_gpu_summary(report: &CheckReport, gpu: &runner::GpuReport) {
    let adapter = gpu.adapter.as_ref().map_or_else(
        || "unknown adapter".to_owned(),
        |a| format!("{} ({})", a.name, a.backend),
    );
    let device_gpu = report
        .outcomes
        .iter()
        .filter(|o| o.device.as_deref() == Some("gpu"))
        .count();
    let rendered = report
        .outcomes
        .iter()
        .filter(|o| o.status != Status::Missing)
        .count();
    println!(
        "GPU run on {adapter}, compared with the {} CPU picture: {device_gpu} of {rendered} rendered files say device gpu",
        match gpu.reference {
            Reference::Stored => "stored",
            Reference::Render => "freshly rendered",
        }
    );
    let worst = |f: fn(&runner::Outcome) -> Option<f64>| {
        report.outcomes.iter().filter_map(f).fold(0.0_f64, f64::max)
    };
    let (median, p99, max) = (
        worst(|o| o.delta_e_2000.map(|d| d.median)),
        worst(|o| o.delta_e_2000.map(|d| d.p99)),
        worst(|o| o.delta_e_2000.map(|d| d.max)),
    );
    let share = worst(|o| o.codes.map(|c| c.share_over));
    let codes = report
        .outcomes
        .iter()
        .filter_map(|o| o.codes.map(|c| c.max))
        .max()
        .unwrap_or(0);
    println!(
        "worst over the files: ΔE00 median {median:.4} (gate {}), p99 {p99:.4} (gate {}), max {max:.4} (gate {}); samples more than {} codes off {share:.2e} (gate {:.0e}), largest difference {codes} codes",
        gpu.gate.median, gpu.gate.p99, gpu.gate.max, gpu.gate.codes_threshold, gpu.gate.codes_share
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
    let report = if a.device == DeviceArg::Gpu {
        if a.write_missing_refs {
            eprintln!(
                "golden: --write-missing-refs stores CPU pictures and cannot be combined with --device gpu"
            );
            return Ok(ExitCode::from(2));
        }
        let gate = gate_of(
            a.gpu_gates,
            a.gpu_median,
            a.gpu_p99,
            a.gpu_max,
            a.gpu_codes_share,
        );
        // The adapter first: a machine without one would otherwise fail every file with the same message.
        let adapter = match adapter_or_exit(&s.ym, s.quiet) {
            Ok(adapter) => adapter,
            Err(code) => return Ok(code),
        };
        let mut report = runner::check_gpu(&ctx, &gate, a.reference, &entries)?;
        if let Some(gpu) = report.gpu.as_mut() {
            gpu.adapter = Some(adapter);
        }
        report
    } else {
        runner::check(&ctx, &entries, a.write_missing_refs)?
    };
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

/// The gates of the options: the chosen set with single values replaced.
fn gate_of(
    set: GateSet,
    median: Option<f64>,
    p99: Option<f64>,
    max: Option<f64>,
    share: Option<f64>,
) -> GpuGate {
    let base = match set {
        GateSet::Calibrated => GpuGate::CALIBRATED,
        GateSet::Plan => GpuGate::PLAN,
    };
    GpuGate {
        median: median.unwrap_or(base.median),
        p99: p99.unwrap_or(base.p99),
        max: max.unwrap_or(base.max),
        codes_share: share.unwrap_or(base.codes_share),
        ..base
    }
}

/// Finds the adapter, or says why not (exit 2 for the caller).
fn adapter_or_exit(ym: &YmBinary, quiet: bool) -> Result<golden::ymrun::Adapter, ExitCode> {
    let scratch = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("golden: creating a scratch folder: {e}");
            return Err(ExitCode::from(2));
        }
    };
    match ym.adapter(scratch.path()) {
        Ok(Some(adapter)) => {
            if !quiet {
                eprintln!(
                    "golden: GPU run on {} ({}{})",
                    adapter.name,
                    adapter.backend,
                    if adapter.software { ", software" } else { "" }
                );
            }
            Ok(adapter)
        }
        Ok(None) => {
            eprintln!(
                "golden: --device gpu: `ym doctor` finds no graphics adapter (is WGPU_BACKEND set? \
                 is Mesa lavapipe installed?), so nothing was rendered"
            );
            Err(ExitCode::from(2))
        }
        Err(e) => {
            eprintln!("golden: --device gpu: {e:#}");
            Err(ExitCode::from(2))
        }
    }
}

fn run_device_check(a: DeviceCheckArgs) -> Result<ExitCode> {
    let ym = YmBinary {
        path: find_ym(a.ym.as_deref())?,
        data: a.data,
    };
    let adapter = match adapter_or_exit(&ym, a.quiet) {
        Ok(adapter) => adapter,
        Err(code) => return Ok(code),
    };
    let entries: Vec<Entry> = a
        .files
        .iter()
        .map(|f| {
            let stem = f
                .file_stem()
                .map_or_else(|| "photo".to_owned(), |s| s.to_string_lossy().into_owned());
            Entry {
                id: format!("local/{stem}"),
                tier: "local".to_owned(),
                set: String::new(),
                file: f.to_string_lossy().into_owned(),
                sidecar: None,
                settings: serde_json::json!({}),
                variant: None,
                light: None,
                body: None,
                cfa: None,
                bits: None,
            }
        })
        .collect();
    for e in &entries {
        if !Path::new(&e.file).is_file() {
            eprintln!("golden: {} is not a file", e.file);
            return Ok(ExitCode::from(2));
        }
    }
    let refs: Vec<&Entry> = entries.iter().collect();
    let gate = gate_of(
        a.gpu_gates,
        a.gpu_median,
        a.gpu_p99,
        a.gpu_max,
        a.gpu_codes_share,
    );
    let ctx = Config {
        ym,
        // Every entry's file is an absolute or working-directory path; `join` keeps an absolute one as it is.
        testdata: std::env::current_dir().context("the working directory")?,
        refs: PathBuf::new(),
        expected: PathBuf::new(),
        jobs: a.jobs.max(1),
        tolerance: gate.max,
        metric: Metric::Max,
        quiet: a.quiet,
    };
    let output = if a.long_edge == 0 {
        golden::ymrun::Output::Png16
    } else {
        golden::ymrun::Output::Png16LongEdge(a.long_edge)
    };
    let mut report = runner::check_gpu_as(&ctx, &gate, Reference::Render, output, &refs)?;
    if let Some(gpu) = report.gpu.as_mut() {
        gpu.adapter = Some(adapter);
    }
    print_check(&report);
    write_report(a.report.as_deref(), &report)?;
    Ok(if report.failed() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match args.command {
        Command::Synth(a) => run_synth(a),
        Command::Check(a) => run_check(a),
        Command::Update(a) => run_update(a),
        Command::Truth(a) => run_truth(a),
        Command::DeviceCheck(a) => run_device_check(a),
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
