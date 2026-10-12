//! `golden check` and `golden update`: render the golden files through `ym`, compare with the stored pictures.
//!
//! ```text
//! ym render --json --settings <entry settings> --format png --bits 16 <file>
//!    -> 16-bit sRGB PNG -> pixel hash
//!         hash == expected.json            identical                          (pass, nothing else needed)
//!         hash differs, reference picture  ΔE00 per pixel on a 1/4 downsample (median, p99, max)
//!              max <= tolerance            within tolerance                   (pass, reported)
//!              max >  tolerance            exceeded                           (FAIL)
//!         hash differs, no reference       no reference                       (FAIL: cannot say by how much)
//! ```
//!
//! A change that is meant goes in with `golden update --note "why"`: it replaces the pictures and the hashes and keeps
//! the note in `expected.json`, so the next check is identical again. That is how "no unexplained ΔE00 > 0.5 change"
//! (`docs/13`) is enforced: an unexplained change leaves the old hash in git and fails the check.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Serialize;

use crate::expected::{Expected, ExpectedEntry, LogEntry};
use crate::image::{DOWNSAMPLE, DeltaStats, Rgb16, delta_e_stats, read_png16, write_png16};
use crate::manifest::{Entry, Manifest};
use crate::ymrun::{Output, YmBinary};

/// A run that did what it was asked but must not go on: a render that failed or a change that has no note. The binary
/// exits with 1 for it (a setup problem, such as a missing `ym`, exits with 2).
#[derive(Debug)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// Which statistic of the per-pixel ΔE00 is held against the tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Metric {
    /// The largest difference of any pixel of the downsample (the default: "changes by ΔE00 > 0.5").
    Max,
    /// The 99th percentile.
    P99,
}

impl Metric {
    fn value(self, s: &DeltaStats) -> f64 {
        match self {
            Self::Max => s.max,
            Self::P99 => s.p99,
        }
    }
}

/// Everything a run needs besides the entries.
#[derive(Clone, Debug)]
pub struct Config {
    /// The binary under test.
    pub ym: YmBinary,
    /// `testdata/`: input files are relative to it.
    pub testdata: PathBuf,
    /// The reference pictures.
    pub refs: PathBuf,
    /// `expected.json`.
    pub expected: PathBuf,
    /// Entries rendered at the same time.
    pub jobs: usize,
    /// The ΔE00 above which a changed picture fails.
    pub tolerance: f64,
    /// The statistic compared with the tolerance.
    pub metric: Metric,
    /// No progress lines on stderr.
    pub quiet: bool,
}

impl Config {
    /// Checks the setting that would silently turn the gate off: a tolerance that is not a number (every comparison
    /// with it is false, so every change would pass) or is negative (every change would fail).
    ///
    /// # Errors
    ///
    /// A message that names the setting.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.tolerance.is_finite() && self.tolerance >= 0.0,
            "the tolerance must be a number of at least 0, not {}",
            self.tolerance
        );
        Ok(())
    }
}

/// How one file came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The pixels are the expected ones.
    Identical,
    /// The pixels changed by less than the tolerance.
    Within,
    /// The pixels changed by more than the tolerance.
    Exceeded,
    /// The pixels differ and there is no reference picture to say by how much.
    NoReference,
    /// The reference picture is not the one `expected.json` describes (a cache from another version).
    StaleReference,
    /// The picture has another size.
    SizeChanged,
    /// `expected.json` has no entry for the file.
    NotExpected,
    /// `ym` failed.
    RenderFailed,
    /// The input file is not on disk.
    Missing,
}

impl Status {
    /// Whether this outcome fails the run.
    #[must_use]
    pub fn failed(self) -> bool {
        !matches!(self, Self::Identical | Self::Within | Self::Missing)
    }

    /// The word in the table.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Identical => "identical",
            Self::Within => "within",
            Self::Exceeded => "EXCEEDED",
            Self::NoReference => "NO REFERENCE",
            Self::StaleReference => "STALE REFERENCE",
            Self::SizeChanged => "SIZE CHANGED",
            Self::NotExpected => "NOT EXPECTED",
            Self::RenderFailed => "RENDER FAILED",
            Self::Missing => "missing",
        }
    }
}

/// One file's result.
#[derive(Clone, Debug, Serialize)]
pub struct Outcome {
    /// The manifest id.
    pub id: String,
    /// How it came out.
    pub status: Status,
    /// The hash of the picture rendered now.
    pub pixels_sha256: Option<String>,
    /// How far the picture moved from the reference.
    pub delta_e_2000: Option<DeltaStats>,
    /// The pipeline version `ym` reported.
    pub pipeline_version: Option<String>,
    /// What to read when the status is not a pass.
    pub message: String,
    /// Wall time of the render.
    pub seconds: f64,
}

/// The result of `golden check`.
#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    /// The tolerance used.
    pub tolerance: f64,
    /// The statistic compared with it.
    pub metric: &'static str,
    /// One outcome per selected entry, in manifest order.
    pub outcomes: Vec<Outcome>,
}

impl CheckReport {
    /// The number of outcomes with this status.
    #[must_use]
    pub fn count(&self, status: Status) -> usize {
        self.outcomes.iter().filter(|o| o.status == status).count()
    }

    /// Whether any outcome fails the run.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.outcomes.iter().any(|o| o.status.failed())
    }

    /// How many files were actually rendered and compared.
    #[must_use]
    pub fn checked(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.status != Status::Missing)
            .count()
    }
}

/// The path of the reference picture of `id` under `refs`.
#[must_use]
pub fn reference_path(refs: &Path, id: &str) -> PathBuf {
    refs.join(format!("{id}.png"))
}

/// Runs `f` on every item with up to `jobs` threads, returning the results in the order of `items`.
fn run_parallel<T: Sync, R: Send>(
    items: &[T],
    jobs: usize,
    f: impl Fn(usize, &T) -> R + Sync,
) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..jobs.clamp(1, items.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(item) = items.get(i) else { break };
                    let r = f(i, item);
                    if let Ok(mut guard) = results.lock() {
                        guard[i] = Some(r);
                    }
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect()
}

struct Rendered {
    picture: Rgb16,
    pipeline_version: String,
    seconds: f64,
}

fn render_picture(ctx: &Config, entry: &Entry) -> Result<Rendered> {
    let input = ctx.testdata.join(&entry.file);
    let scratch = tempfile::tempdir().context("creating a scratch folder")?;
    let started = Instant::now();
    let rendered = ctx.ym.render(
        &input,
        &entry.settings,
        Output::Png16,
        &scratch.path().join("out"),
        scratch.path(),
    )?;
    let seconds = started.elapsed().as_secs_f64();
    let picture = read_png16(&rendered.output)?;
    Ok(Rendered {
        picture,
        pipeline_version: rendered.pipeline_version(),
        seconds,
    })
}

fn progress(ctx: &Config, done: &AtomicUsize, total: usize, id: &str, what: &str) {
    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
    if !ctx.quiet {
        eprintln!("[{n}/{total}] {id}: {what}");
    }
}

/// Compares a picture that differs from its expectation with the reference picture.
fn compare_with_reference(
    ctx: &Config,
    entry: &Entry,
    expected: &ExpectedEntry,
    picture: &Rgb16,
) -> (Status, Option<DeltaStats>, String) {
    let path = reference_path(&ctx.refs, &entry.id);
    if !path.is_file() {
        return (
            Status::NoReference,
            None,
            format!(
                "the pixels differ from expected.json and {} does not exist, so the change cannot be measured; \
                 restore the reference pictures (CI cache) or explain the change with `golden update --note`",
                path.display()
            ),
        );
    }
    let reference = match read_png16(&path) {
        Ok(r) => r,
        Err(e) => return (Status::NoReference, None, format!("{e:#}")),
    };
    if reference.pixel_sha256() != expected.pixels_sha256 {
        return (
            Status::StaleReference,
            None,
            format!(
                "{} is not the picture expected.json describes (another version's reference?); \
                 delete testdata/golden/refs or run `golden update`",
                path.display()
            ),
        );
    }
    match delta_e_stats(&reference, picture) {
        Ok(stats) => {
            let value = ctx.metric.value(&stats);
            let (status, verb) = if value > ctx.tolerance {
                (Status::Exceeded, "moved by more than")
            } else {
                (Status::Within, "moved by no more than")
            };
            (
                status,
                Some(stats),
                format!(
                    "the picture {verb} the tolerance of ΔE00 {} (median {:.3}, p99 {:.3}, max {:.3}); \
                     if this is intended, run `golden update --note \"why\"` and bump pipeline_version",
                    ctx.tolerance, stats.median, stats.p99, stats.max
                ),
            )
        }
        Err(e) => (Status::SizeChanged, None, format!("{e:#}")),
    }
}

fn check_one(
    ctx: &Config,
    entry: &Entry,
    expected: &Expected,
    write_missing_refs: bool,
) -> Outcome {
    let mut outcome = Outcome {
        id: entry.id.clone(),
        status: Status::Missing,
        pixels_sha256: None,
        delta_e_2000: None,
        pipeline_version: None,
        message: String::new(),
        seconds: 0.0,
    };
    let input = ctx.testdata.join(&entry.file);
    if !input.is_file() {
        outcome.message = format!(
            "{} is not on disk (fetch or generate it first)",
            input.display()
        );
        return outcome;
    }
    let rendered = match render_picture(ctx, entry) {
        Ok(r) => r,
        Err(e) => {
            outcome.status = Status::RenderFailed;
            outcome.message = format!("{e:#}");
            return outcome;
        }
    };
    let hash = rendered.picture.pixel_sha256();
    outcome.pixels_sha256 = Some(hash.clone());
    outcome.pipeline_version = Some(rendered.pipeline_version.clone());
    outcome.seconds = rendered.seconds;
    let Some(want) = expected.entries.get(&entry.id) else {
        outcome.status = Status::NotExpected;
        outcome.message =
            "expected.json has no entry for this file; run `golden update`".to_owned();
        return outcome;
    };
    if want.pixels_sha256 == hash {
        outcome.status = Status::Identical;
        // Same pixels: the difference is exactly zero everywhere.
        outcome.delta_e_2000 = Some(DeltaStats {
            pixels: (rendered.picture.width / DOWNSAMPLE) * (rendered.picture.height / DOWNSAMPLE),
            median: 0.0,
            p99: 0.0,
            mean: 0.0,
            max: 0.0,
        });
        if write_missing_refs {
            let path = reference_path(&ctx.refs, &entry.id);
            if !path.exists()
                && let Err(e) = write_png16(&path, &rendered.picture)
            {
                outcome.message = format!("could not write the reference: {e:#}");
            }
        }
        return outcome;
    }
    let (status, stats, message) = compare_with_reference(ctx, entry, want, &rendered.picture);
    outcome.status = status;
    outcome.delta_e_2000 = stats;
    outcome.message = message;
    if status == Status::Within || status == Status::Exceeded {
        // The pixels moved although the pipeline version did not.
        if want.pipeline_version == rendered.pipeline_version {
            outcome.message.push_str(
                " Note: pipeline_version is unchanged, but the pixels are not (docs/13).",
            );
        }
    }
    outcome
}

/// `golden check`: renders every entry and compares it with `expected.json` and the reference pictures.
///
/// With `write_missing_refs`, a picture that is identical to its expectation and has no reference file gets one (a
/// CI run on the main branch fills the cache this way).
///
/// # Errors
///
/// Only when `expected.json` cannot be read; a failing file is an [`Outcome`], not an error.
pub fn check(ctx: &Config, entries: &[&Entry], write_missing_refs: bool) -> Result<CheckReport> {
    ctx.validate()?;
    let expected = Expected::load(&ctx.expected)?;
    let done = AtomicUsize::new(0);
    let outcomes = run_parallel(entries, ctx.jobs, |_, entry| {
        let o = check_one(ctx, entry, &expected, write_missing_refs);
        progress(ctx, &done, entries.len(), &o.id, o.status.label());
        o
    });
    Ok(CheckReport {
        tolerance: ctx.tolerance,
        metric: match ctx.metric {
            Metric::Max => "max",
            Metric::P99 => "p99",
        },
        outcomes,
    })
}

/// What `golden update` did to one file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateKind {
    /// No entry before.
    Added,
    /// The pixels are the ones already expected.
    Unchanged,
    /// The pixels changed.
    Changed,
}

/// One file's part of an update.
#[derive(Clone, Debug, Serialize)]
pub struct UpdateOutcome {
    /// The manifest id.
    pub id: String,
    /// What changed.
    pub kind: UpdateKind,
    /// How far a changed picture moved from the old reference (when there was one).
    pub delta_e_2000: Option<DeltaStats>,
}

/// The result of `golden update`.
#[derive(Clone, Debug, Serialize)]
pub struct UpdateReport {
    /// One entry per file that was rendered.
    pub outcomes: Vec<UpdateOutcome>,
    /// Entries removed because the manifest no longer lists them (`--prune`).
    pub pruned: Vec<String>,
    /// Files that were not on disk and were left alone.
    pub missing: Vec<String>,
    /// Whether `expected.json` was written.
    pub written: bool,
}

impl UpdateReport {
    /// How many files have this kind of outcome.
    #[must_use]
    pub fn count(&self, kind: UpdateKind) -> usize {
        self.outcomes.iter().filter(|o| o.kind == kind).count()
    }
}

/// `golden update`: renders every entry and stores the pictures and their hashes.
///
/// A change to an existing picture needs `note`; nothing is written otherwise. `prune` also drops the entries of
/// `expected.json` that `manifest` no longer lists.
///
/// # Errors
///
/// When a render fails (nothing is written), when a change has no note, or when a file cannot be written.
pub fn update(
    ctx: &Config,
    manifest: &Manifest,
    entries: &[&Entry],
    note: Option<&str>,
    prune: bool,
) -> Result<UpdateReport> {
    let mut expected = Expected::load(&ctx.expected)?;
    let mut missing = Vec::new();
    let mut to_render = Vec::new();
    for e in entries {
        if ctx.testdata.join(&e.file).is_file() {
            to_render.push(*e);
        } else {
            missing.push(e.id.clone());
        }
    }
    let done = AtomicUsize::new(0);
    let renders = run_parallel(&to_render, ctx.jobs, |_, entry| {
        let r = render_picture(ctx, entry);
        progress(
            ctx,
            &done,
            to_render.len(),
            &entry.id,
            if r.is_ok() { "rendered" } else { "FAILED" },
        );
        r
    });
    let mut failures = Vec::new();
    let mut pictures = Vec::new();
    for (entry, r) in to_render.iter().zip(renders) {
        match r {
            Ok(r) => pictures.push((*entry, r)),
            Err(e) => failures.push(format!("{}: {e:#}", entry.id)),
        }
    }
    if !failures.is_empty() {
        return Err(Refused(format!(
            "{} file(s) did not render, nothing was written:\n  {}",
            failures.len(),
            failures.join("\n  ")
        ))
        .into());
    }

    let mut outcomes = Vec::new();
    for (entry, r) in &pictures {
        let hash = r.picture.pixel_sha256();
        let (kind, delta) = match expected.entries.get(&entry.id) {
            None => (UpdateKind::Added, None),
            Some(old) if old.pixels_sha256 == hash => (UpdateKind::Unchanged, None),
            Some(old) => {
                let delta = read_png16(&reference_path(&ctx.refs, &entry.id))
                    .ok()
                    .filter(|reference| reference.pixel_sha256() == old.pixels_sha256)
                    .and_then(|reference| delta_e_stats(&reference, &r.picture).ok());
                (UpdateKind::Changed, delta)
            }
        };
        outcomes.push(UpdateOutcome {
            id: entry.id.clone(),
            kind,
            delta_e_2000: delta,
        });
    }
    let changed = outcomes
        .iter()
        .filter(|o| o.kind == UpdateKind::Changed)
        .count();
    let added = outcomes
        .iter()
        .filter(|o| o.kind == UpdateKind::Added)
        .count();
    if changed > 0 && note.is_none_or(|n| n.trim().is_empty()) {
        let list: Vec<String> = outcomes
            .iter()
            .filter(|o| o.kind == UpdateKind::Changed)
            .map(|o| match &o.delta_e_2000 {
                Some(d) => format!(
                    "{} (ΔE00 median {:.3}, p99 {:.3}, max {:.3})",
                    o.id, d.median, d.p99, d.max
                ),
                None => format!("{} (no old reference picture to measure against)", o.id),
            })
            .collect();
        return Err(Refused(format!(
            "{changed} existing picture(s) changed; say why with `golden update --note \"...\"` (docs/13: no unexplained \
             change). Nothing was written.\n  {}",
            list.join("\n  ")
        ))
        .into());
    }

    for (entry, r) in &pictures {
        let hash = r.picture.pixel_sha256();
        let path = reference_path(&ctx.refs, &entry.id);
        let stored_ok = read_png16(&path)
            .map(|p| p.pixel_sha256() == hash)
            .unwrap_or(false);
        if !stored_ok {
            write_png16(&path, &r.picture)?;
        }
        expected.entries.insert(
            entry.id.clone(),
            ExpectedEntry {
                pixels_sha256: hash,
                width: u32::try_from(r.picture.width).unwrap_or(u32::MAX),
                height: u32::try_from(r.picture.height).unwrap_or(u32::MAX),
                pipeline_version: r.pipeline_version.clone(),
            },
        );
    }
    let mut pruned = Vec::new();
    if prune {
        let known: std::collections::BTreeSet<&str> =
            manifest.entries.iter().map(|e| e.id.as_str()).collect();
        let stale: Vec<String> = expected
            .entries
            .keys()
            .filter(|k| !known.contains(k.as_str()))
            .cloned()
            .collect();
        for id in stale {
            expected.entries.remove(&id);
            // Best effort: the picture of a file that is no longer in the manifest.
            let _ = std::fs::remove_file(reference_path(&ctx.refs, &id));
            pruned.push(id);
        }
    }
    // JSON has no infinity (serde_json writes `null`, which `expected.json` could not read back): a picture whose
    // difference is not a number is recorded as the largest finite value.
    let max_de = outcomes
        .iter()
        .filter_map(|o| o.delta_e_2000.as_ref())
        .map(|d| d.max)
        .fold(0.0, f64::max)
        .min(f64::MAX);
    let written = changed > 0 || added > 0 || !pruned.is_empty();
    if changed > 0 || added > 0 {
        let version = pictures
            .first()
            .map_or_else(|| "unknown".to_owned(), |(_, r)| r.pipeline_version.clone());
        expected.log.push(LogEntry {
            pipeline_version: version,
            note: note
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .unwrap_or("new pictures")
                .to_owned(),
            changed,
            added,
            max_delta_e_2000: max_de,
        });
    }
    if written {
        expected.save(&ctx.expected)?;
    }
    Ok(UpdateReport {
        outcomes,
        pruned,
        missing,
        written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_results_keep_their_order() {
        let items: Vec<usize> = (0..40).collect();
        let out = run_parallel(&items, 5, |i, v| {
            std::thread::sleep(std::time::Duration::from_millis(((40 - i) % 3) as u64));
            v * 2
        });
        assert_eq!(out, (0..40).map(|v| v * 2).collect::<Vec<_>>());
        assert!(run_parallel::<usize, usize>(&[], 3, |_, v| *v).is_empty());
    }

    #[test]
    fn statuses_say_whether_they_fail() {
        assert!(!Status::Identical.failed());
        assert!(!Status::Within.failed());
        assert!(!Status::Missing.failed());
        for s in [
            Status::Exceeded,
            Status::NoReference,
            Status::StaleReference,
            Status::SizeChanged,
            Status::NotExpected,
            Status::RenderFailed,
        ] {
            assert!(s.failed(), "{s:?}");
        }
    }

    #[test]
    fn a_tolerance_that_is_not_a_number_is_refused() {
        let config = |tolerance: f64| Config {
            ym: YmBinary {
                path: PathBuf::from("ym"),
                data: None,
            },
            testdata: PathBuf::new(),
            refs: PathBuf::new(),
            expected: PathBuf::new(),
            jobs: 1,
            tolerance,
            metric: Metric::Max,
            quiet: true,
        };
        assert!(config(0.5).validate().is_ok());
        assert!(config(0.0).validate().is_ok());
        for bad in [f64::NAN, -0.1, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(config(bad).validate().is_err(), "{bad}");
            let entries: [&Entry; 0] = [];
            assert!(check(&config(bad), &entries, false).is_err(), "{bad}");
        }
    }

    #[test]
    fn reference_paths_follow_the_id() {
        let p = reference_path(Path::new("/r"), "synthetic/chart__x");
        assert_eq!(p, Path::new("/r/synthetic/chart__x.png"));
    }
}
