//! `dngfit`: fits the DNG profile of the baked-look export and checks the DNG gates (`docs/09-export.md`).
//!
//! * `dngfit fit`: fit the tone curve, the `BaselineExposure` offset and the look table to the cine rendering with Adobe's renderer
//!   (`dng_validate`) and write `data/dng/profile.json`.
//! * `dngfit eval`: how close Adobe's render with a profile is to the cine rendering on the test sets.
//! * `dngfit gate`: the DNG gates on the golden set (`dng_validate` passes; render against our JPEG, ΔE00 median).
//! * `dngfit validate`: `dng_validate` on DNG files.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use dngfit::adobe::Adobe;
use dngfit::check::{self, GateOptions, MEDIAN_LIMIT, Reprofile};
use dngfit::fit::{self, FitConfig, SetResult};
use dngfit::profile_file::{self, ProfileFile};
use golden::manifest::{Manifest, SetChoice, select};
use golden::ymrun::{YmBinary, find_ym, workspace_root};

/// Fits the DNG profile and checks the DNG gates.
#[derive(Debug, Parser)]
#[command(name = "dngfit", version)]
struct Args {
    /// Adobe's `dng_validate` (default: $DNG_VALIDATE, else testdata/dng_sdk/dng_validate; build it with
    /// tools/dng/build_dng_validate.sh).
    #[arg(long, global = true)]
    dng_validate: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fit the profile to the cine rendering and write it (default: data/dng/profile.json).
    Fit(FitArgs),
    /// Evaluate a profile file against the cine rendering on the test sets.
    Eval(EvalArgs),
    /// The DNG gates on the golden set: dng_validate passes, and its render matches our JPEG.
    Gate(GateArgs),
    /// Run dng_validate on DNG files and print what it complains about.
    Validate {
        /// The DNG files.
        files: Vec<PathBuf>,
    },
}

#[derive(Debug, clap::Args)]
struct FitArgs {
    /// Where to write the profile.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Also write the fit report (JSON) here.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Hue, saturation and value divisions of the look table (at most 36,32,16).
    #[arg(long, default_value = "36,16,16", value_delimiter = ',')]
    dims: Vec<u32>,
    /// The scene-linear AWG4 value at the top of the tone curve; BaselineExposure's offset is -log2 of it.
    #[arg(long, default_value_t = 16.0)]
    x_min: f64,
    /// Points of the tone curve.
    #[arg(long, default_value_t = 64)]
    tone_points: usize,
    /// Samples per grid cell and direction for the least-squares fit of the table.
    #[arg(long, default_value_t = 3)]
    density: usize,
    /// Weight of the pull towards the node-exact solution.
    #[arg(long, default_value_t = 1e-4)]
    pull: f64,
    /// Conjugate-gradient iterations.
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    /// The synthetic scenes (testdata/synthetic/scenes) for the golden-patch test set; skipped when missing.
    #[arg(long)]
    scenes: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct EvalArgs {
    /// The profile (default: data/dng/profile.json).
    #[arg(long)]
    profile: Option<PathBuf>,
    /// The synthetic scenes for the golden-patch test set.
    #[arg(long)]
    scenes: Option<PathBuf>,
    /// Write the result as JSON here.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct GateArgs {
    /// Write every DNG again with this profile before checking it (an error for a `ym` that already writes a fitted
    /// profile). Without it the default is data/dng/profile.json for the DNGs that carry the identity profile.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Check the DNGs exactly as `ym` writes them. Without `--profile` or this option a DNG with the identity profile is
    /// written again with data/dng/profile.json and one that already carries a fitted profile is checked as it is.
    #[arg(long, conflicts_with = "profile")]
    as_written: bool,
    /// Fail (exit 1) when a golden file of the selection is not on disk, instead of leaving it out.
    #[arg(long)]
    require_all: bool,
    /// Check these ARW files instead of golden entries (auto settings; for local evaluation of real files).
    #[arg(long = "files", num_args = 1.., value_name = "ARW")]
    files: Vec<PathBuf>,
    /// Which golden entries (see `golden check --help`).
    #[arg(long, value_enum, default_value = "synthetic")]
    set: SetChoice,
    /// Only entries whose id contains this text (repeatable).
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
    /// The data folder for `ym` ($YELLOWMOON_DATA), for a release build outside the workspace.
    #[arg(long)]
    data: Option<PathBuf>,
    /// The limit for the median ΔE00 of every file.
    #[arg(long, default_value_t = MEDIAN_LIMIT)]
    limit: f64,
    /// Also compare with the 16-bit PNG render.
    #[arg(long)]
    png: bool,
    /// Files processed at the same time (default: half the cores).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    /// Write the result as JSON here.
    #[arg(long)]
    report: Option<PathBuf>,
    /// No progress lines.
    #[arg(long)]
    quiet: bool,
}

fn default_profile_path() -> PathBuf {
    workspace_root().join("data/dng/profile.json")
}

fn default_scenes() -> Option<PathBuf> {
    let root = std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("testdata"));
    let dir = root.join("synthetic/scenes");
    dir.is_dir().then_some(dir)
}

fn print_sets(results: &[SetResult]) {
    println!(
        "{:<62} {:>6} {:>7} {:>7} {:>7} {:>7}",
        "set (ΔE00 of Adobe's render against the cine rendering)",
        "n",
        "median",
        "p95",
        "p99",
        "max"
    );
    for r in results {
        println!(
            "{:<62} {:>6} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            r.name, r.stats.count, r.stats.median, r.stats.p95, r.stats.p99, r.stats.max
        );
    }
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn run_fit(adobe: &Adobe, a: FitArgs) -> Result<ExitCode> {
    if a.dims.len() != 3 {
        bail!("--dims takes three numbers: hue, saturation, value");
    }
    let cfg = FitConfig {
        x_min: a.x_min,
        tone_points: a.tone_points,
        tone_iterations: 8,
        dims: [a.dims[0], a.dims[1], a.dims[2]],
    };
    cfg.validate()?;
    let started = Instant::now();
    let (mut curve, tone, tone_report) = fit::fit_tone(adobe, &cfg)?;
    fit::round_curve(&mut curve);
    eprintln!(
        "tone curve: {} points, neutral ΔE00 median {:.4} max {:.4} ({:.1} s)",
        curve.len(),
        tone_report.median_delta_e,
        tone_report
            .max_delta_e_by_iteration
            .last()
            .copied()
            .unwrap_or(0.0),
        started.elapsed().as_secs_f64()
    );
    let (mut table, look_report) = fit::fit_look(&cfg, &tone, a.density, a.pull, a.iterations);
    fit::round_table(&mut table);
    eprintln!(
        "look table: {} x {} x {}, {} samples, {:.1} s in all",
        cfg.dims[0],
        cfg.dims[1],
        cfg.dims[2],
        look_report.least_squares.samples,
        started.elapsed().as_secs_f64()
    );
    let profile = ym_export::dng::DngProfile {
        tone_curve: curve,
        look_table: table,
    };
    profile.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    let scenes = a.scenes.or_else(default_scenes);
    let sets = fit::test_sets(scenes.as_deref());
    let results = fit::evaluate_sets(adobe, &profile, cfg.offset_ev(), &sets)?;
    print_sets(&results);
    let mut summary = serde_json::json!({
        "tool": "dngfit",
        "dng_validate": adobe.version(),
        "config": {
            "x_min": cfg.x_min,
            "fit_offset_ev": cfg.offset_ev(),
            "tone_points": cfg.tone_points,
            "tone_iterations": cfg.tone_iterations,
            "look_dims": cfg.dims,
            "look_encoding": 1,
            "sample_density": a.density,
            "pull": a.pull,
            "cg_iterations": a.iterations,
        },
        "tone": tone_report,
        "look": look_report,
        "delta_e_2000_against_cine_via_dng_validate": results,
    });
    dngfit::profile_file::round_numbers(&mut summary, 6);
    let file = ProfileFile::from_profile(
        &profile,
        cfg.offset_ev(),
        "the cine rendering, sRGB (cine-rec709)",
        summary.clone(),
    );
    let out = a.out.unwrap_or_else(default_profile_path);
    profile_file::save(&file, &out)?;
    eprintln!("wrote {}", out.display());
    if let Some(r) = &a.report {
        write_json(r, &summary)?;
    }
    Ok(ExitCode::SUCCESS)
}

fn run_eval(adobe: &Adobe, a: EvalArgs) -> Result<ExitCode> {
    let path = a.profile.unwrap_or_else(default_profile_path);
    let file = profile_file::load(&path)?;
    let scenes = a.scenes.or_else(default_scenes);
    let sets = fit::test_sets(scenes.as_deref());
    let results = fit::evaluate_sets(adobe, &file.to_profile(), file.fit_offset_ev, &sets)?;
    print_sets(&results);
    if let Some(r) = &a.report {
        write_json(r, &results)?;
    }
    Ok(ExitCode::SUCCESS)
}

fn run_gate(adobe: &Adobe, a: GateArgs) -> Result<ExitCode> {
    let root = workspace_root();
    let manifest = Manifest::load(
        &a.manifest
            .clone()
            .unwrap_or_else(|| root.join("tools/golden/manifest.json")),
    )?;
    let testdata = a
        .testdata
        .clone()
        .or_else(|| std::env::var_os("YM_TESTDATA").map(PathBuf::from))
        .unwrap_or_else(|| root.join("testdata"));
    let ad_hoc: Vec<golden::manifest::Entry> = a
        .files
        .iter()
        .map(|f| golden::manifest::Entry {
            id: f.file_name().map_or_else(
                || f.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            tier: "local".to_owned(),
            set: String::new(),
            file: std::fs::canonicalize(f)
                .unwrap_or_else(|_| f.clone())
                .to_string_lossy()
                .into_owned(),
            sidecar: None,
            settings: serde_json::json!({}),
            variant: None,
            light: None,
            body: None,
            cfa: None,
            bits: None,
        })
        .collect();
    let all: Vec<&golden::manifest::Entry> = if ad_hoc.is_empty() {
        select(&manifest, a.set, &a.filter)
    } else {
        ad_hoc.iter().collect()
    };
    let (entries, absent): (Vec<_>, Vec<_>) = all
        .into_iter()
        .partition(|e| testdata.join(&e.file).is_file());
    let not_on_disk: Vec<String> = absent.iter().map(|e| e.id.clone()).collect();
    if !ad_hoc.is_empty() && !not_on_disk.is_empty() {
        bail!("not a file: {}", not_on_disk.join(", "));
    }
    if !not_on_disk.is_empty() {
        eprintln!(
            "dngfit: {} golden file(s) of the selection are not on disk and are not checked (tools/fetch/synthetic.sh): {}",
            not_on_disk.len(),
            not_on_disk.join(", ")
        );
    }
    if entries.is_empty() {
        eprintln!("dngfit: no golden file of the selection is on disk (tools/fetch/synthetic.sh)");
        return Ok(ExitCode::from(3));
    }
    let reprofile = if a.as_written {
        Reprofile::Never
    } else if let Some(path) = &a.profile {
        Reprofile::Always(profile_file::load(path)?)
    } else {
        Reprofile::IfIdentity(profile_file::load(&default_profile_path())?)
    };
    let opts = GateOptions {
        ym: YmBinary {
            path: find_ym(a.ym.as_deref())?,
            data: a.data.clone(),
        },
        testdata,
        adobe: adobe.clone(),
        reprofile,
        jobs: if a.jobs == 0 {
            std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1))
        } else {
            a.jobs
        },
        with_png: a.png,
        median_limit: a.limit,
        quiet: a.quiet,
    };
    let mut report = check::run(&opts, &entries);
    report.not_on_disk = not_on_disk;
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>9}  id",
        "gate", "median", "p99", "max", "size"
    );
    for f in &report.files {
        let (m, p, x) = f
            .vs_jpeg
            .map_or(("-".to_owned(), "-".to_owned(), "-".to_owned()), |s| {
                (
                    format!("{:.3}", s.median),
                    format!("{:.3}", s.p99),
                    format!("{:.3}", s.max),
                )
            });
        println!(
            "{:<8} {m:>8} {p:>8} {x:>8} {:>9}  {}",
            if f.passed { "pass" } else { "FAIL" },
            format!("{}x{}", f.width, f.height),
            f.id
        );
        for r in &f.reasons {
            println!("    {r}");
        }
        if let Some(s) = f.vs_png {
            println!(
                "    against the 16-bit PNG: median {:.3} p99 {:.3} max {:.3}",
                s.median, s.p99, s.max
            );
        }
    }
    println!(
        "{} files ({} DNGs): {} passed; median of the per-file medians {:.3}, worst file median {:.3}, limit {}",
        report.files.len(),
        report.dng,
        report.files.iter().filter(|f| f.passed).count(),
        report.median_of_medians().unwrap_or(f64::NAN),
        report.worst_median().unwrap_or(f64::NAN),
        report.median_limit
    );
    if let Some(r) = &a.report {
        write_json(r, &report)?;
    }
    if a.require_all && !report.not_on_disk.is_empty() {
        eprintln!("dngfit: --require-all: some golden files are not on disk");
        return Ok(ExitCode::from(1));
    }
    Ok(if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn run_validate(adobe: &Adobe, files: &[PathBuf]) -> Result<ExitCode> {
    if files.is_empty() {
        bail!("give at least one DNG file");
    }
    let mut failed = false;
    for (file, result) in check::validate_files(adobe, files) {
        match result {
            Ok(problems) if problems.is_empty() => println!("ok       {}", file.display()),
            Ok(problems) => {
                failed = true;
                println!("PROBLEMS {}", file.display());
                for p in problems {
                    println!("    {p}");
                }
            }
            Err(e) => {
                failed = true;
                println!("FAILED   {}: {e:#}", file.display());
            }
        }
    }
    Ok(if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = Adobe::new(args.dng_validate.as_deref()).and_then(|adobe| match args.command {
        Command::Fit(a) => run_fit(&adobe, a),
        Command::Eval(a) => run_eval(&adobe, a),
        Command::Gate(a) => run_gate(&adobe, a),
        Command::Validate { files } => run_validate(&adobe, &files),
    });
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("dngfit: {e:#}");
            ExitCode::from(2)
        }
    }
}
