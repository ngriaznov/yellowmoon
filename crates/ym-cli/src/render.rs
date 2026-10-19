//! `ym render`: develop and export (`docs/10-cli.md`).
//!
//! The command opens the photos in a session, develops them, gives each its settings
//! ([`crate::develop`]) and hands the whole set to the core's export queue, the same queue the app
//! uses, so the files are the bytes the app would write (`ym_core::export::export_photo` is the
//! only code that makes a file). Results arrive as the queue finishes each photo; the command
//! prints them in that order.
//!
//! Apple Auto (`docs/19`) is the app's too: the core gets this computer's advisor and the
//! `apple_auto` preference decides unless `--apple-auto on|off` says otherwise. When it acts, the
//! photos are developed and answered before their settings are applied (`Session::wait_idle` waits
//! for the answers), so the JSON and the files carry Apple's values.
//!
//! stdout carries data only: the path of every file written, or one JSON object per photo with
//! `--json`. Everything for a person (progress, warnings, the summary) goes to stderr.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use ym_core::app::gpu::{
    CpuReason, GPU_ENV, GpuOutcome, GpuPolicy, GpuSlot, PolicyChoice, PolicySource, open_gpu,
};
use ym_core::export_queue::{ExportEvent, ExportJob, FileDone, QueueError};
use ym_core::looks::{LookService, LooksError};
use ym_core::pipeline::{ComputeDevice, DisplayChoice};
use ym_core::preferences::DisplayRendering;
use ym_core::session::{ApplyScope, OpenOptions, PhotoId, PhotoStatus};
use ym_export::{ExportOptions, FormatKind, IfExists, JpegQuality, Size};
use ym_looks::library::LookId;
use ym_looks::library::ResolveError;
use ym_raw::layout::DecodeKind;

use crate::cli::{AppleAutoArg, RenderArgs};
use crate::console::{Console, bar, duration_text, photos, shorten};
use crate::develop::{LoadedSettings, Overrides, SettingsFile};
use crate::exit::{self, Failure};
use crate::inputs::{self, Expanded, Problem};
use crate::report::{AppleAutoField, Object, develop_fields, render_error, render_ok};
use crate::runtime::{
    BackgroundGpu, CoreSettings, Runtime, gpu_line, interrupted, read_instructions,
};
use crate::text::{data_path, line_text, lossy, path_text};

/// The longest file name the progress line shows. With the bar and the counts the line is then
/// about 70 columns, which fits the 80 of a default terminal window; a longer line would wrap and
/// every redraw would leave a copy of itself behind.
const PROGRESS_NAME_CHARS: usize = 32;

/// Runs `ym render` and returns the exit code.
pub fn run(args: &RenderArgs, console: &mut Console) -> u8 {
    match run_checked(args, console) {
        Ok(code) => code,
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// The failure a look error stands for.
pub fn look_failure(error: &LooksError) -> Failure {
    let code = u8::try_from(error.exit_code()).unwrap_or(exit::SOME_FAILED);
    match error {
        // The library's own text names the long command; the command here is `ym`.
        LooksError::Resolve(ResolveError::NotFound { query }) => Failure::new(
            exit::LOOK_MISSING,
            format!("No look called \"{query}\". Run `ym looks list` to see the ids and names."),
        ),
        other => Failure::new(code, other.to_string()),
    }
}

/// Finds the look `--look` or `looks default` names, and checks that its file is there.
///
/// # Errors
///
/// Exit code 2 for a name that matches several looks, 3 for one that matches none or whose file is
/// gone.
pub fn resolve_look(looks: &LookService, query: &str) -> Result<LookId, Failure> {
    let id = looks.resolve(query).map_err(|e| look_failure(&e))?;
    looks.check_usable(&id).map_err(|e| look_failure(&e))?;
    Ok(id)
}

/// `--display`: which display rendering this run uses.
fn resolve_display(looks: &LookService, query: &str) -> Result<DisplayChoice, Failure> {
    let word = query.trim();
    if word.eq_ignore_ascii_case("aces2") {
        return Ok(DisplayChoice::Aces2);
    }
    let luts = looks.display_luts();
    let id = if word.eq_ignore_ascii_case("arri") {
        let chosen = match looks.preferences().get().display_rendering {
            DisplayRendering::ArriDisplay { id } => {
                luts.iter().find(|l| l.id == id).map(|l| l.id.clone())
            }
            DisplayRendering::Aces2 => None,
        };
        match (chosen, luts.as_slice()) {
            (Some(id), _) => id,
            (None, [only]) => only.id.clone(),
            (None, []) => {
                return Err(Failure::new(
                    exit::LOOK_MISSING,
                    "No ARRI display LUTs are imported. Import ARRI's look package first (ym looks import <package>).",
                ));
            }
            (None, many) => {
                let list = many
                    .iter()
                    .map(|l| format!("{} ({})", l.id, l.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(Failure::usage(format!(
                    "Several ARRI display LUTs are imported: {list}. Choose one in the app's Preferences, or name it: --display <id>."
                )));
            }
        }
    } else {
        let id = LookId::parse(word).map_err(|_| {
            Failure::usage(format!(
                "`{word}` is not a display rendering. Use aces2, arri, or the id of an imported display LUT."
            ))
        })?;
        if luts.iter().all(|l| l.id != id) {
            return Err(Failure::new(
                exit::LOOK_MISSING,
                format!("There is no imported display LUT with the id {id}."),
            ));
        }
        id
    };
    // The file must still be there and make a usable transform.
    looks
        .provider()
        .display_transform(&id)
        .map_err(|e| Failure::new(exit::LOOK_MISSING, e.to_string()))?;
    Ok(DisplayChoice::ArriLut(id))
}

/// The export options of a run: the table of `docs/09` as defaults, the user's format preference,
/// then the options of the command line.
fn export_options(
    args: &RenderArgs,
    preferred_format: FormatKind,
) -> Result<ExportOptions, Failure> {
    let mut o = ExportOptions {
        format: args.format.map_or(preferred_format, Into::into),
        ..ExportOptions::default()
    };
    if let Some(q) = args.quality {
        o.jpeg_quality = JpegQuality::new(q).map_err(|e| Failure::usage(e.to_string()))?;
    }
    if let Some(bits) = args.bits {
        o.bit_depth = bits;
    }
    if let Some(c) = args.compression {
        o.tiff_compression = c.into();
    }
    if let Some(c) = args.color_space {
        o.color_space = c.into();
    }
    if let Some(px) = args.long_edge {
        o.size = Size::LongEdge { px };
    }
    if let Some(dir) = &args.out {
        o.destination = Some(std::path::absolute(dir).unwrap_or_else(|_| dir.clone()));
    }
    if let Some(name) = &args.name {
        o.name_template.clone_from(name);
    }
    o.if_exists = if args.skip_existing {
        IfExists::Skip
    } else if args.overwrite {
        IfExists::Replace
    } else {
        IfExists::KeepBoth
    };
    o.validate().map_err(|e| Failure::usage(e.to_string()))?;
    Ok(o)
}

/// Options that were given but do not apply to the chosen format.
fn inapplicable(args: &RenderArgs, format: FormatKind) -> Vec<&'static str> {
    let mut out = Vec::new();
    if args.quality.is_some() && format != FormatKind::Jpeg {
        out.push("--quality (JPEG only)");
    }
    if args.bits.is_some() && !matches!(format, FormatKind::Tiff | FormatKind::Png) {
        out.push("--bits (TIFF and PNG only)");
    }
    if args.compression.is_some() && format != FormatKind::Tiff {
        out.push("--compression (TIFF only)");
    }
    if format == FormatKind::Dng {
        if args.color_space.is_some() {
            out.push("--colour-space (not for DNG)");
        }
        if args.long_edge.is_some() {
            out.push("--long-edge (not for DNG)");
        }
        if args.output_sharpening.is_some() {
            out.push("--output-sharpening (not for DNG)");
        }
        // A DNG carries no sharpening of any kind (`docs/09`): R8 is off for both of its renders.
        if args.sharpen.is_some() {
            out.push("--sharpen (not for DNG)");
        }
    }
    out
}

/// One input file and what became of it.
struct Plan {
    /// The photo as an absolute path.
    input: PathBuf,
    /// The session's id for it, while it can still be exported.
    photo: Option<PhotoId>,
    /// `camera`, `look`, `auto` and `settings` once the photo is developed.
    known: Option<Object>,
    /// How it ended.
    result: Option<Outcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Written,
    Skipped,
    Failed { kind: String },
}

/// A stdout that notices when its reader is gone and then stays quiet.
struct Stdout {
    open: bool,
}

impl Stdout {
    fn line(&mut self, text: &str) {
        if !self.open {
            return;
        }
        let mut out = io::stdout().lock();
        if writeln!(out, "{text}").and_then(|()| out.flush()).is_err() {
            // The reader went away (`| head -n 1`). The files are the product: keep exporting.
            self.open = false;
        }
    }

    /// The path of a file that was written, as a line of data: exact for a pipe, escaped on a
    /// terminal (`crate::text::data_path`).
    fn path(&mut self, path: &Path) {
        self.line(&data_path(path, io::stdout().is_terminal()));
    }
}

/// What `--require-gpu` (or `YELLOWMOON_GPU=require`) says when no device opens.
fn no_gpu_message(choice: &PolicyChoice, reason: &str) -> String {
    let asked = match choice.source {
        PolicySource::Flag(flag) => format!("{} was given", flag.text()),
        PolicySource::Environment | PolicySource::Default => {
            format!("{GPU_ENV}={} is set", choice.policy)
        }
    };
    format!("No usable GPU was found ({reason}). {asked}, so nothing was rendered.")
}

/// What is said about the graphics device once it is open: the adapter and the stages on a line of
/// detail, a software adapter that `Auto` left alone, and a warning when none was found.
fn report_gpu(console: &mut Console, slot: &GpuSlot) {
    let adapter = slot.adapter().map(gpu_line).unwrap_or_default();
    match slot.outcome() {
        GpuOutcome::Using => console.detail(&format!("GPU: {adapter}; stages: {}", slot.stages())),
        GpuOutcome::Cpu(CpuReason::SoftwareAdapter) => console.detail(&format!(
            "GPU: {} is a software adapter, so the CPU is used (YELLOWMOON_GPU=on uses it).",
            slot.adapter().map(|a| a.name.as_str()).unwrap_or_default()
        )),
        GpuOutcome::Cpu(CpuReason::NoAdapter(reason)) | GpuOutcome::Unavailable(reason) => {
            console.warn("No usable GPU was found. Rendering on the CPU, which is slower.");
            console.detail(&format!("GPU: {reason}"));
        }
        GpuOutcome::Cpu(CpuReason::Disabled) => {}
    }
}

/// Where the pictures of one photo were made, for `device` in the JSON: what the export reported
/// ([`ym_core::export::ExportReport::device`], the path the photo actually took), not what the policy allowed. A
/// photo that the device failed on and the CPU made again says `cpu` and carries the note ("The
/// graphics card could not render this photo, so the CPU did: …"), so a strict job that reads both
/// cannot be fooled by a silent fallback.
const fn device_name(device: ComputeDevice) -> &'static str {
    match device {
        ComputeDevice::Gpu => "gpu",
        ComputeDevice::Cpu => "cpu",
    }
}

fn read_settings_text(path: &Path) -> Result<String, Failure> {
    if path.as_os_str() == "-" {
        let mut text = String::new();
        io::Read::read_to_string(&mut io::stdin().lock(), &mut text)
            .map_err(|e| Failure::usage(format!("Couldn't read the settings from stdin: {e}.")))?;
        Ok(text)
    } else {
        std::fs::read_to_string(path).map_err(|e| {
            Failure::usage(format!(
                "Couldn't read the settings file {}: {e}.",
                path_text(path)
            ))
        })
    }
}

fn decode_kind_unsupported(kind: DecodeKind) -> bool {
    matches!(kind, DecodeKind::Arw6 | DecodeKind::Unknown { .. })
}

/// Why a photo the session skipped could not be developed, as a JSON `kind`.
fn skip_kind(path: &Path) -> &'static str {
    match ym_raw::metadata::probe(path) {
        Err(error) => error.kind(),
        Ok(meta) if decode_kind_unsupported(meta.decode) => "unsupported",
        Ok(_) => "decode",
    }
}

fn run_checked(args: &RenderArgs, console: &mut Console) -> Result<u8, Failure> {
    let inputs_from_stdin = args.inputs.iter().any(|p| p.as_os_str() == "-");
    let settings_from_stdin = args.settings.as_deref() == Some(Path::new("-"));
    if inputs_from_stdin && settings_from_stdin {
        return Err(Failure::usage(
            "Both the photo list (-) and --settings - read stdin. Give the settings as a file.",
        ));
    }
    let raw_inputs = if inputs_from_stdin {
        if io::stdin().is_terminal() {
            console.info("Reading file paths from stdin, one per line. End with Ctrl-D.");
        }
        inputs::with_stdin(&args.inputs, io::stdin().lock())
            .map_err(|e| Failure::usage(format!("Couldn't read the file list from stdin: {e}.")))?
    } else {
        args.inputs.clone()
    };

    // The data folder, the preferences, the look library and Apple Auto's advisor.
    let apple_auto = AppleAutoArg::mode(args.apple_auto);
    let apple_auto_instructions = args
        .apple_auto_instructions
        .as_deref()
        .map(read_instructions)
        .transpose()?;
    let rt = Runtime::open(&CoreSettings {
        jobs: args.jobs,
        forced_profile: args.profile.clone(),
        apple_auto: Some(apple_auto),
        apple_auto_instructions,
    })?;
    let core = &rt.core;
    if let Some(notice) = rt.apple_auto_notice(apple_auto) {
        console.warn(&notice);
    }

    // The graphics policy: `--cpu` or `--require-gpu`, else `YELLOWMOON_GPU`, else Auto, the same
    // choice the app makes (`ym_core::app::gpu`). A policy that needs a device is checked first: a
    // machine that can't meet it renders nothing.
    let choice = GpuPolicy::from_environment(args.policy_flag());
    if let Some(warning) = &choice.warning {
        console.warn(warning);
    }
    let budget = core.renderer.options().budget;
    let mut gpu_slot: Option<GpuSlot> = None;
    if choice.policy.is_strict() {
        let slot = open_gpu(choice.policy, budget);
        if let Some(reason) = slot.failure() {
            return Err(Failure::new(exit::NO_GPU, no_gpu_message(&choice, reason)));
        }
        // As `BackgroundGpu` does for the other policies: the kernels an export will use compile
        // while the photos develop (`GpuStages::USED_BY_EXPORTS`: the raw chain).
        drop(slot.warm_up_for_exports());
        gpu_slot = Some(slot);
    } else if choice.policy == GpuPolicy::Off {
        // Nothing to open, but the renderer is told so all the same, as the app tells it: the
        // same policy leaves the same options (no device, no stage) in both front ends.
        gpu_slot = Some(GpuSlot::off(budget));
    }

    // Display rendering and look.
    match args.display.as_deref() {
        Some(query) => core
            .renderer
            .set_display(resolve_display(&core.looks, query)?),
        None => core
            .looks
            .check_display_rendering()
            .map_err(|e| look_failure(&e))?,
    }
    let look = args
        .look
        .as_deref()
        .map(|q| resolve_look(&core.looks, q))
        .transpose()?;
    let overrides = Overrides::from_args(args, look);

    // The settings file.
    let default_look = core.looks.effective_default();
    let settings_file = match &args.settings {
        Some(path) => {
            let LoadedSettings { file, clamped } =
                SettingsFile::parse(&read_settings_text(path)?, &default_look)
                    .map_err(Failure::usage)?;
            for (entry, fields) in clamped {
                let whom = if entry.is_empty() {
                    "The settings file".to_owned()
                } else {
                    format!("The settings for {entry}")
                };
                console.warn(&format!(
                    "{whom} had out-of-range values for {}; they were limited to their range.",
                    fields.join(", ")
                ));
            }
            Some(file)
        }
        None => None,
    };

    // The export options.
    let preferred = core.preferences.get().export_defaults.format;
    let options = export_options(args, preferred)?;
    for flag in inapplicable(args, options.format) {
        console.warn(&format!(
            "{flag}: ignored for {} files.",
            options.format.display_name()
        ));
    }

    // The photos.
    let Expanded {
        files,
        problems,
        ignored,
    } = inputs::expand(&raw_inputs, args.recursive);
    let mut stdout = Stdout { open: true };
    let mut counts = Counts::default();
    for Problem {
        path,
        kind,
        message,
    } in &problems
    {
        report_failure(console, &mut stdout, args.json, path, None, kind, message);
        counts.failed.push((*kind).to_owned());
    }
    if files.is_empty() {
        console.error(if ignored > 0 || !problems.is_empty() {
            "No ARW files were found. Give ARW files, or folders that hold them."
        } else {
            "There are no files to render. Give ARW files, or folders that hold them."
        });
        return Ok(exit::NO_INPUT);
    }
    if let Some(file) = &settings_file {
        for unused in file.unused_entries(&files) {
            console.warn(&format!(
                "The settings file has an entry for {unused}, which is not one of the photos."
            ));
        }
    }

    // Everything that could be a usage mistake has been checked. Open the GPU while the photos
    // develop (it is wanted before the export starts), unless the CPU was asked for or a strict
    // policy opened it already.
    let mut gpu_open = (gpu_slot.is_none() && choice.policy != GpuPolicy::Off)
        .then(|| BackgroundGpu::start(choice.policy, budget));

    let mut plans: Vec<Plan> = Vec::with_capacity(files.len());
    for file in &files {
        let outcome = core
            .session
            .open(std::slice::from_ref(file), OpenOptions::default());
        plans.push(Plan {
            input: file.clone(),
            photo: outcome.added.first().copied(),
            known: None,
            result: None,
        });
    }
    let total = plans.len();
    let developed = rt.wait_developed(|w| {
        // Apple Auto's answers come after the photos are developed (`docs/19` section 6).
        let (label, done, n) = match w.answers {
            Some((answered, n)) => ("Asking Apple Intelligence", answered, n),
            None => ("Developing", w.ready, w.total),
        };
        console.progress(format!(
            "{label} {} {done}/{n}",
            bar(done as f64 / n.max(1) as f64, 20)
        ));
    });
    console.finish_progress();
    if !developed {
        core.close_session();
        console.info("Interrupted.");
        return Ok(exit::INTERRUPTED);
    }

    // Settings for each photo; photos that cannot be developed end here.
    let mut by_photo: HashMap<PhotoId, usize> = HashMap::new();
    for (i, plan) in plans.iter_mut().enumerate() {
        let input = &plan.input;
        let Some(id) = plan.photo else {
            plan.result = Some(Outcome::Failed {
                kind: "open".into(),
            });
            counts.failed.push("open".into());
            report_failure(
                console,
                &mut stdout,
                args.json,
                input,
                None,
                "open",
                "This file couldn't be opened. Check that it exists and try again.",
            );
            continue;
        };
        if let Some(PhotoStatus::Skipped { reason }) = core.session.status(id) {
            let kind = skip_kind(&plan.input);
            plan.result = Some(Outcome::Failed { kind: kind.into() });
            plan.photo = None;
            counts.failed.push(kind.into());
            report_failure(console, &mut stdout, args.json, input, None, kind, &reason);
            continue;
        }
        match apply_settings(&rt, id, &plan.input, settings_file.as_ref(), &overrides) {
            Ok((known, warning)) => {
                if let Some(warning) = warning {
                    console.warn(&format!("{}: {warning}", path_text(input)));
                }
                plan.known = Some(known);
                by_photo.insert(id, i);
            }
            Err(message) => {
                plan.result = Some(Outcome::Failed {
                    kind: "settings".into(),
                });
                plan.photo = None;
                counts.failed.push("settings".into());
                report_failure(
                    console,
                    &mut stdout,
                    args.json,
                    input,
                    None,
                    "settings",
                    &message,
                );
            }
        }
    }
    if by_photo.is_empty() {
        summarise(
            console,
            args.json,
            &counts,
            &Summary::default(),
            total,
            false,
        );
        return Ok(final_code(&counts, false));
    }

    // The device is open by now (or was never wanted). Say what it is, and give it to the core
    // before the export is queued, so that the whole export is on the device or on the CPU.
    let gpu_slot = gpu_slot.or_else(|| gpu_open.as_mut().and_then(BackgroundGpu::finish));
    if let Some(slot) = &gpu_slot {
        report_gpu(console, slot);
        core.attach_gpu(slot);
    }
    // Export.
    let remove_location = core.preferences.get().remove_location_from_exports;
    // The photos that are going to be exported are the picked ones: a photo whose settings were
    // refused above is still in the session, and must not be exported with its defaults.
    core.session.clear_picks();
    for id in by_photo.keys() {
        if let Err(error) = core.session.set_picked(*id, true) {
            return Err(Failure::failed(error.to_string()));
        }
    }
    let job = ExportJob {
        scope: ApplyScope::Picked,
        options: options.clone(),
    };
    // Apple Auto's availability is read again before the export starts, as the app's
    // `Core::export_start` does (`docs/19` section 1); an Apple Auto the run wanted that has gone
    // meanwhile is said once.
    if let Some(notice) = rt.apple_auto_notice_before_export() {
        console.warn(&notice);
    }
    let started = Instant::now();
    if let Err(error) = core.exports.start(&core.session, &job, remove_location) {
        return Err(match error {
            QueueError::Options(_) => Failure::usage(error.to_string()),
            other => Failure::failed(other.to_string()),
        });
    }

    let mut summary = Summary::default();
    let mut cancelled = false;
    let mut idle_seen = false;
    let mut notes: Vec<String> = Vec::new();
    loop {
        if interrupted() && !cancelled {
            cancelled = true;
            core.exports.cancel_all();
        }
        match rt.events.recv_timeout(Duration::from_millis(100)) {
            Ok(ExportEvent::File(done)) => {
                handle_file(
                    done,
                    &mut plans,
                    &by_photo,
                    args.json,
                    console,
                    &mut stdout,
                    &mut counts,
                    &mut summary,
                    &mut notes,
                );
            }
            Ok(ExportEvent::Progress(p)) => {
                let fraction =
                    (p.done as f64 + p.file_percent.unwrap_or(0.0) / 100.0) / p.total.max(1) as f64;
                console.progress(format!(
                    "Exporting {} {}/{}  {}",
                    bar(fraction, 20),
                    (p.done + 1).min(p.total),
                    p.total,
                    shorten(&line_text(&p.file), PROGRESS_NAME_CHARS)
                ));
            }
            Ok(ExportEvent::Busy(false)) | Err(RecvTimeoutError::Disconnected) => break,
            Ok(_) => {}
            Err(RecvTimeoutError::Timeout) => {
                // `Busy(false)` normally ends the run. A job that Ctrl-C removed before the
                // worker took it up sends no `Busy` event at all, and waiting for one would hang.
                // The queue is busy from `start` until its worker has finished, so an idle queue
                // with an empty channel means everything has been seen; one more pass picks up
                // an event that arrived between the two looks.
                if !core.exports.is_busy() {
                    if idle_seen {
                        break;
                    }
                    idle_seen = true;
                }
            }
        }
    }
    console.finish_progress();

    // A photo with no result: the job stopped (Ctrl-C) or something went wrong below us.
    if !cancelled {
        for plan in plans
            .iter_mut()
            .filter(|p| p.photo.is_some() && p.result.is_none())
        {
            let message = "The export stopped before this photo was done. Try it again.";
            counts.failed.push("render".into());
            report_failure(
                console,
                &mut stdout,
                args.json,
                &plan.input,
                plan.known.as_ref(),
                "render",
                message,
            );
            plan.result = Some(Outcome::Failed {
                kind: "render".into(),
            });
        }
    }
    summary.elapsed = started.elapsed().as_secs_f64();
    summarise(console, args.json, &counts, &summary, total, cancelled);
    if !args.json || console.is_verbose() {
        for note in &notes {
            console.info(&format!("Note: {note}"));
        }
    }
    Ok(final_code(&counts, cancelled))
}

/// Gives a photo its settings: the file's, then the options'. Returns what `--json` knows of it
/// and, when an option asked for more than the settings allow, a sentence saying what was done
/// instead (the options are checked against their ranges before anything runs, but
/// `--exposure-offset` depends on the photo: its automatic exposure plus the offset can pass 4).
fn apply_settings(
    rt: &Runtime,
    id: PhotoId,
    path: &Path,
    file: Option<&SettingsFile>,
    overrides: &Overrides,
) -> Result<(Object, Option<String>), String> {
    let session = &rt.core.session;
    let mut resolved = session.settings(id).map_err(|e| e.to_string())?;
    if let Some(settings) = file.and_then(|f| f.for_file(path)) {
        resolved = session
            .replace_settings(id, settings.clone())
            .map_err(|e| e.to_string())?;
    }
    let mut warning = None;
    if !overrides.is_empty() {
        let auto_ev = resolved.estimates.exposure_ev;
        let patch = overrides.patch(auto_ev);
        if !patch.is_empty() {
            resolved = session
                .update_settings(id, &patch)
                .map_err(|e| e.to_string())?;
        }
        if let (Some(offset), None) = (overrides.exposure_offset, overrides.exposure) {
            let wanted = auto_ev + offset;
            let got = resolved.values.exposure_ev;
            if (got - wanted).abs() > 1e-9 {
                warning = Some(format!(
                    "the exposure is limited to {got:+.2} EV (automatic {auto_ev:+.2} plus {offset:+.2} would be {wanted:+.2})."
                ));
            }
        }
    }
    let view = session.view(id).map_err(|e| e.to_string())?;
    Ok((
        develop_fields(&view, &resolved, AppleAutoField::WhenActive),
        warning,
    ))
}

#[derive(Debug, Default)]
struct Counts {
    /// The kind of every failure.
    failed: Vec<String>,
}

#[derive(Debug, Default)]
struct Summary {
    written: usize,
    skipped: usize,
    folders: Vec<PathBuf>,
    elapsed: f64,
}

/// Prints one failure: a JSON object with `--json`, else a line on stderr.
fn report_failure(
    console: &mut Console,
    stdout: &mut Stdout,
    json: bool,
    input: &Path,
    known: Option<&Object>,
    kind: &str,
    message: &str,
) {
    if json {
        stdout.line(&render_error(&lossy(input), known, kind, message).to_line());
    } else {
        console.error(&format!("{}: {message}", path_text(input)));
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the loop's state, passed once per file"
)]
fn handle_file(
    done: FileDone,
    plans: &mut [Plan],
    by_photo: &HashMap<PhotoId, usize>,
    json: bool,
    console: &mut Console,
    stdout: &mut Stdout,
    counts: &mut Counts,
    summary: &mut Summary,
    notes: &mut Vec<String>,
) {
    let Some(&index) = by_photo.get(&done.photo) else {
        return;
    };
    let plan = &mut plans[index];
    let input = &plan.input;
    match done.outcome {
        Ok(report) => {
            for note in &report.notes {
                if !notes.contains(note) {
                    notes.push(note.clone());
                }
            }
            if report.skipped {
                summary.skipped += 1;
                plan.result = Some(Outcome::Skipped);
                console.detail(&format!(
                    "{}: {} already exists, skipped.",
                    path_text(input),
                    path_text(&report.path)
                ));
            } else {
                summary.written += 1;
                plan.result = Some(Outcome::Written);
                if let Some(dir) = report.path.parent()
                    && !summary.folders.iter().any(|f| f == dir)
                {
                    summary.folders.push(dir.to_path_buf());
                }
                console.detail(&format!(
                    "{}: {}x{} in {} (read {}, render {}, encode {}).",
                    path_text(input),
                    report.width,
                    report.height,
                    duration_text((report.timings.render_ms + report.timings.encode_ms) / 1000.0),
                    duration_text(done.decode_ms / 1000.0),
                    duration_text(report.timings.render_ms / 1000.0),
                    duration_text(report.timings.encode_ms / 1000.0),
                ));
            }
            if json {
                let status = if report.skipped { "skipped" } else { "ok" };
                // `device` joins what is known of the photo, so it comes after `settings` and
                // before `timings_ms`: where the picture was made (`docs/10`); a file that was
                // skipped was not rendered, so it has none (`null`).
                let mut known = plan.known.clone().unwrap_or_default();
                known.insert(
                    "device",
                    (!report.skipped).then_some(device_name(report.device)),
                );
                stdout.line(
                    &render_ok(
                        &lossy(input),
                        &lossy(&report.path),
                        status,
                        &known,
                        (
                            done.decode_ms,
                            report.timings.render_ms,
                            report.timings.encode_ms,
                        ),
                        &report.notes,
                    )
                    .to_line(),
                );
            } else if !report.skipped {
                stdout.path(&report.path);
            }
        }
        Err(failure) => {
            plan.result = Some(Outcome::Failed {
                kind: failure.kind.to_owned(),
            });
            counts.failed.push(failure.kind.to_owned());
            report_failure(
                console,
                stdout,
                json,
                input,
                plan.known.as_ref(),
                failure.kind,
                &failure.message,
            );
        }
    }
}

/// The last lines on stderr.
fn summarise(
    console: &mut Console,
    json: bool,
    counts: &Counts,
    summary: &Summary,
    total: usize,
    interrupted: bool,
) {
    let mut parts: Vec<String> = Vec::new();
    if interrupted {
        parts.push(format!(
            "Interrupted: {} of {} exported. Nothing half-written is left behind.",
            summary.written + summary.skipped,
            photos(total)
        ));
    } else if summary.written > 0 {
        let place = match summary.folders.as_slice() {
            [one] => format!(" to {}", path_text(one)),
            [] => String::new(),
            many => format!(" to {} folders", many.len()),
        };
        parts.push(format!(
            "Exported {}{place} in {}.",
            photos(summary.written),
            duration_text(summary.elapsed)
        ));
    }
    if summary.skipped > 0 {
        parts.push(if summary.skipped == 1 {
            "1 photo already existed and was skipped.".to_owned()
        } else {
            format!(
                "{} already existed and were skipped.",
                photos(summary.skipped)
            )
        });
    }
    if !counts.failed.is_empty() {
        parts.push(format!("{} failed.", photos(counts.failed.len())));
    }
    if parts.is_empty() {
        return;
    }
    // A program reading JSON has the objects; the sentence is for a person.
    if !json || console.is_verbose() || interrupted {
        console.info(&parts.join(" "));
    }
}

/// The exit code of a run that got as far as exporting.
fn final_code(counts: &Counts, interrupted: bool) -> u8 {
    if interrupted {
        exit::INTERRUPTED
    } else if counts.failed.is_empty() {
        exit::OK
    } else if counts.failed.iter().all(|k| k == "look") {
        exit::LOOK_MISSING
    } else {
        exit::SOME_FAILED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_export::{BitDepth, ColorSpace, TiffCompression};

    #[test]
    fn exit_codes_follow_the_failures() {
        let none = Counts::default();
        assert_eq!(final_code(&none, false), exit::OK);
        assert_eq!(final_code(&none, true), exit::INTERRUPTED);
        let look = Counts {
            failed: vec!["look".into(), "look".into()],
        };
        assert_eq!(final_code(&look, false), exit::LOOK_MISSING);
        let mixed = Counts {
            failed: vec!["look".into(), "render".into()],
        };
        assert_eq!(final_code(&mixed, false), exit::SOME_FAILED);
        let one = Counts {
            failed: vec!["not_arw".into()],
        };
        assert_eq!(final_code(&one, false), exit::SOME_FAILED);
    }

    #[test]
    fn a_missing_look_name_gets_the_ym_command_in_its_message() {
        let error = LooksError::Resolve(ResolveError::NotFound {
            query: "warm".into(),
        });
        let failure = look_failure(&error);
        assert_eq!(failure.code, exit::LOOK_MISSING);
        assert!(
            failure.message.contains("`ym looks list`"),
            "{}",
            failure.message
        );
        assert!(!failure.message.contains("yellowmoon"));
    }

    #[test]
    fn options_that_do_not_apply_to_the_format_are_named() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from([
            "ym",
            "render",
            "a.ARW",
            "--format",
            "dng",
            "--quality",
            "80",
            "--long-edge",
            "1000",
            "--bits",
            "8",
            "--compression",
            "zip",
            "--sharpen",
            "50,1.2",
        ])
        .expect("parses");
        let crate::cli::Command::Render(args) = cli.command else {
            panic!("render");
        };
        let flags = inapplicable(&args, FormatKind::Dng);
        assert_eq!(flags.len(), 5, "{flags:?}");
        assert!(flags.contains(&"--sharpen (not for DNG)"));
        assert!(!inapplicable(&args, FormatKind::Jpeg).contains(&"--sharpen (not for DNG)"));
        assert!(inapplicable(&args, FormatKind::Jpeg).contains(&"--bits (TIFF and PNG only)"));
    }

    #[test]
    fn export_options_start_from_the_table_of_docs_09() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["ym", "render", "a.ARW"]).expect("parses");
        let crate::cli::Command::Render(args) = cli.command else {
            panic!("render");
        };
        let o = export_options(&args, FormatKind::Tiff).expect("options");
        assert_eq!(o.format, FormatKind::Tiff, "the preferred format");
        assert_eq!(o.jpeg_quality.get(), 92);
        assert_eq!(o.bit_depth, BitDepth::Sixteen);
        assert_eq!(o.tiff_compression, TiffCompression::Lzw);
        assert_eq!(o.color_space, ColorSpace::Srgb);
        assert_eq!(o.size, Size::Full);
        assert_eq!(o.if_exists, IfExists::KeepBoth);
        assert_eq!(o.name_template, "{name}_{look}");
    }
}
