//! `ym doctor`: the diagnostics that go into a bug report (`docs/10-cli.md`).
//!
//! The report says what it includes and nothing else: the version, the GPU, the render path,
//! whether Apple Auto is available (and which model answers, `docs/19` section 8), the model weights
//! with their hashes, the camera profiles, how many looks are imported and the folders Yellowmoon
//! uses. No photo, file name, location or look name is in it.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use ym_core::app::gpu::{
    CpuReason, GpuOutcome, GpuPolicy, GpuStages, PolicyChoice, PolicySource, open_gpu,
    probe_adapter,
};
use ym_core::apple_auto::{
    Activity, AppleAutoMode, AppleAutoStatus, Availability, DEFAULT_INSTRUCTIONS,
    INSTRUCTIONS_MAX_CHARS, InstructionsId, PROMPT_VERSION, TABLE_VERSION, effective_instructions,
};
use ym_core::data_dir::find_data_dir;
use ym_core::looks::{LookGroupInfo, LookService};
use ym_core::pipeline::NeuralStatus;
use ym_core::preferences::PreferencesStore;
use ym_core::scheduler::MemoryBudget;
use ym_core::version::{PIPELINE_VERSION, app_version};
use ym_gpu::device::GpuInfo;
use ym_profiles::registry::ProfileStore;

pub mod bench;

use crate::cli::{DoctorArgs, version_text};
use crate::console::Console;
use crate::exit::{self, Failure};
use crate::text::{json_line, line_text, lossy, path_text};

/// The sentence at the top of the report: what it includes, and what it leaves out.
pub const INCLUDES: &str = "This report lists the version, the GPU, the render path, whether Apple Auto is available, the model weights and their hashes, the camera profiles, how many looks are imported, and the folders Yellowmoon uses. It does not include photos, file names, locations or look names.";

/// A model file shipped in `data/models/`.
struct ModelFile {
    /// Path inside `data/models/`, with `/`.
    name: String,
    sha256: String,
}

/// What the graphics policy comes to on this machine: the setting and where it came from, whether
/// the device is used, and the budgets that follow.
struct GpuPlan {
    choice: PolicyChoice,
    /// Whether the device would be used for the stages below.
    uses_gpu: bool,
    /// Why not, in plain words (`None` when it is used or the policy is `off`).
    reason: Option<String>,
    /// The stages that would run on the device.
    stages: GpuStages,
    /// The budgets the run would have.
    budget: MemoryBudget,
    /// The adapter shares the machine's RAM.
    shared_memory: bool,
}

/// Everything the report says.
struct Report {
    gpu: Result<GpuInfo, String>,
    plan: GpuPlan,
    neural: NeuralStatus,
    apple_auto: AppleAutoStatus,
    weights: Vec<ModelFile>,
    illuminant_models: Vec<ModelFile>,
    profiles: Vec<(String, String, String)>,
    looks_built_in: usize,
    looks_imported: usize,
    looks_missing: usize,
    display_luts: usize,
    preferences_path: PathBuf,
    preferences_exist: bool,
    preference_warnings: usize,
    looks_folder: PathBuf,
    data_folder: PathBuf,
}

/// Runs `ym doctor` and returns the exit code.
pub fn run(args: &DoctorArgs, console: &mut Console) -> u8 {
    if let Some(path) = &args.bench {
        return run_bench(path, args, console);
    }
    match build() {
        Ok(report) => {
            let mut out = io::stdout().lock();
            let written = if args.json {
                writeln!(out, "{}", json_line(&to_json(&report)))
            } else {
                write!(out, "{}", to_text(&report))
            }
            .and_then(|()| out.flush());
            match written {
                Ok(()) => exit::OK,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => exit::OK,
                Err(e) => {
                    console.error(&format!("Couldn't write the report ({e})."));
                    exit::SOME_FAILED
                }
            }
        }
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// `ym doctor --bench <file>`: times the render path on one photo (see [`bench`](mod@bench)).
fn run_bench(path: &Path, args: &DoctorArgs, console: &mut Console) -> u8 {
    match bench::run(path, args.repeat) {
        Ok(report) => {
            let mut out = io::stdout().lock();
            let written = if args.json {
                writeln!(out, "{}", json_line(&report))
            } else {
                write!(out, "{}", bench::to_text(&report))
            }
            .and_then(|()| out.flush());
            match written {
                Ok(()) => exit::OK,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => exit::OK,
                Err(e) => {
                    console.error(&format!("Couldn't write the report ({e})."));
                    exit::SOME_FAILED
                }
            }
        }
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

fn hash_models(dir: &Path, skip: &[&str]) -> Vec<ModelFile> {
    let mut files = Vec::new();
    collect_models(dir, dir, skip, &mut files);
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files
}

fn collect_models(root: &Path, dir: &Path, skip: &[&str], out: &mut Vec<ModelFile>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if skip.contains(&name.as_str()) || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_models(root, &path, skip, out);
        } else if path
            .extension()
            .is_some_and(|e| matches!(e.to_str(), Some("json" | "safetensors" | "onnx" | "bin")))
            && let Ok(sha256) = ym_looks::manifest::hash_file(&path)
        {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(ModelFile {
                name: relative,
                sha256,
            });
        }
    }
}

fn build() -> Result<Report, Failure> {
    let data_folder = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    let app_root = ym_looks::library::default_root().map_err(|e| Failure::usage(e.to_string()))?;
    let models = data_folder.join("models");
    // The illuminant models of `models/illum/` are the shipped estimator parameters; everything
    // else under `models/` (the demosaic network, `jdd/`) is a weight file.
    let all = hash_models(&models, &["metrics"]);
    let (illuminant_models, weights): (Vec<_>, Vec<_>) =
        all.into_iter().partition(|m| m.name.starts_with("illum/"));

    let store = ProfileStore::load(&data_folder)
        .map_err(|e| Failure::failed(format!("The camera profiles can't be read: {e}")))?;
    let profiles = store
        .profiles()
        .map(|p| {
            (
                p.id.to_string(),
                p.version.to_string(),
                p.display_name.clone(),
            )
        })
        .collect();

    let prefs = Arc::new(PreferencesStore::open(&app_root));
    let preferences_path = prefs.path().to_path_buf();
    let preference_warnings = prefs.warnings().len();
    let service = LookService::open(&app_root, &data_folder, Arc::clone(&prefs))
        .map_err(|e| Failure::failed(e.to_string()))?;
    let looks = service.list();
    let (gpu, plan) = gpu_report();
    Ok(Report {
        gpu,
        plan,
        neural: NeuralStatus::probe(Some(&data_folder)),
        apple_auto: apple_auto_status(prefs.get().apple_auto, &prefs.get().apple_auto_instructions),
        weights,
        illuminant_models,
        profiles,
        looks_built_in: looks
            .iter()
            .filter(|l| matches!(l.group, LookGroupInfo::BuiltIn))
            .count(),
        looks_imported: looks
            .iter()
            .filter(|l| !matches!(l.group, LookGroupInfo::BuiltIn))
            .count(),
        looks_missing: looks.iter().filter(|l| l.missing).count(),
        display_luts: service.display_luts().len(),
        preferences_exist: preferences_path.exists(),
        preferences_path,
        preference_warnings,
        looks_folder: service
            .library()
            .read()
            .map_or_else(|p| p.into_inner().looks_dir(), |lib| lib.looks_dir()),
        data_folder,
    })
}

/// Opens the device the way a run would (`YELLOWMOON_GPU`, else Auto) and says what it found.
/// The adapter is reported whatever the policy does with it: a software adapter that `Auto`
/// leaves alone, and the one on a machine where the setting is `off`, are still listed.
fn gpu_report() -> (Result<GpuInfo, String>, GpuPlan) {
    let choice = GpuPolicy::from_environment(None);
    let slot = open_gpu(choice.policy, MemoryBudget::detect());
    let gpu = match (slot.adapter(), slot.missing_device_reason()) {
        (Some(info), _) => Ok(info.clone()),
        (None, Some(reason)) => Err(reason.to_owned()),
        // `off` opened nothing; look anyway, the report is for finding out what is there.
        (None, None) => probe_adapter().map_err(|e| e.to_string()),
    };
    let reason = match slot.outcome() {
        GpuOutcome::Using | GpuOutcome::Cpu(CpuReason::Disabled) => None,
        GpuOutcome::Cpu(CpuReason::SoftwareAdapter) => {
            Some("only a software adapter was found; YELLOWMOON_GPU=on uses it".to_owned())
        }
        GpuOutcome::Cpu(CpuReason::NoAdapter(reason)) | GpuOutcome::Unavailable(reason) => {
            Some(format!("no adapter opened ({reason})"))
        }
    };
    let plan = GpuPlan {
        uses_gpu: slot.is_using(),
        reason,
        stages: slot.stages(),
        budget: slot.budget(),
        shared_memory: slot.shares_memory(),
        choice,
    };
    (gpu, plan)
}

/// Apple Auto on this computer, as `Core::apple_auto_status` gives it to the app's Preferences
/// (`docs/19` sections 1, 2 and 8), without opening a session: the advisor the app and `ym` use
/// (`ym_apple::system_advisor`, read now), the `apple_auto` preference and the person's own
/// instructions (`apple_auto_instructions`, `""` for the default).
fn apple_auto_status(preference: bool, own_instructions: &str) -> AppleAutoStatus {
    let availability = ym_apple::system_advisor().availability();
    let active = Activity::decide(
        true,
        AppleAutoMode::FromPreference,
        preference,
        Some(&availability),
    )
    .is_active();
    let text = effective_instructions(own_instructions);
    let (available, reason, info) = match availability {
        Availability::Available(info) => (true, None, Some(info.with_instructions(text))),
        Availability::Unavailable(reason) => (false, Some(reason), None),
    };
    AppleAutoStatus {
        available,
        reason,
        info,
        preference,
        mode: AppleAutoMode::FromPreference,
        active,
        prompt: PROMPT_VERSION,
        table: TABLE_VERSION,
        instructions: InstructionsId::of(text),
        own_instructions: own_instructions.to_owned(),
        default_instructions: DEFAULT_INSTRUCTIONS.to_owned(),
        instructions_max_chars: INSTRUCTIONS_MAX_CHARS,
    }
}

/// The `apple_auto` object of `doctor --json`: `{ "available", "reason", "model", "os",
/// "preference", "prompt", "table", "instructions", "instructions_sha256" }` (`docs/19` section 8).
fn apple_auto_value(s: &AppleAutoStatus) -> Value {
    json!({
        "available": s.available,
        "reason": s.reason.and_then(|r| serde_json::to_value(r).ok()),
        "model": s.info.as_ref().map(|i| i.model.as_str()),
        "os": s.info.as_ref().map(|i| i.os.as_str()),
        "preference": s.preference,
        "prompt": s.prompt,
        "table": s.table,
        "instructions": s.instructions.instructions,
        "instructions_sha256": s.instructions.instructions_sha256,
    })
}

/// The `Apple Auto` row: `available · Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) ·
/// prompt 3 · table 2` (with `· edited instructions (<digest>)` when the person wrote their own, and
/// `· turned off in Preferences` when the preference is off), or `not available (this is not a
/// Mac)`.
fn apple_auto_text(s: &AppleAutoStatus) -> String {
    match (&s.info, s.reason) {
        (Some(info), _) if s.available => {
            let mut text = format!("available · {}", line_text(&info.version_string()));
            if !s.preference {
                text.push_str(" · turned off in Preferences");
            }
            text
        }
        (_, Some(reason)) => format!("not available ({})", reason.words()),
        _ => "not available".to_owned(),
    }
}

fn source_word(source: PolicySource) -> &'static str {
    match source {
        PolicySource::Flag(_) => "flag",
        PolicySource::Environment => "environment",
        PolicySource::Default => "default",
    }
}

fn gpu_value(gpu: &Result<GpuInfo, String>) -> Value {
    match gpu {
        Ok(i) => json!({
            "available": true,
            "name": i.name,
            "backend": i.backend,
            "device_type": i.device_type,
            "software": i.is_software,
            "driver": i.driver,
            "driver_info": i.driver_info,
            "shader_f16": i.shader_f16,
            "storage_mode": i.storage_mode.name(),
            "memory_budget_bytes": i.memory_budget_bytes,
            "limits": {
                "max_buffer_size": i.limits.max_buffer_size,
                "max_storage_buffer_binding_size": i.limits.max_storage_buffer_binding_size,
                "max_texture_dimension_2d": i.limits.max_texture_dimension_2d,
                "max_storage_buffers_per_shader_stage": i.limits.max_storage_buffers_per_shader_stage,
                "max_compute_workgroups_per_dimension": i.limits.max_compute_workgroups_per_dimension,
                "max_compute_invocations_per_workgroup": i.limits.max_compute_invocations_per_workgroup,
                "max_compute_workgroup_storage_size": i.limits.max_compute_workgroup_storage_size,
            },
        }),
        Err(reason) => json!({ "available": false, "reason": reason }),
    }
}

fn model_values(files: &[ModelFile]) -> Value {
    Value::Array(
        files
            .iter()
            .map(|m| json!({ "file": m.name, "sha256": m.sha256 }))
            .collect(),
    )
}

fn to_json(r: &Report) -> Value {
    json!({
        "includes": INCLUDES,
        "version": app_version(),
        "version_text": format!("ym {}", version_text()),
        "pipeline_version": PIPELINE_VERSION,
        "system": { "os": std::env::consts::OS, "arch": std::env::consts::ARCH },
        "gpu": gpu_value(&r.gpu),
        "gpu_policy": {
            "setting": r.plan.choice.policy.name(),
            "source": source_word(r.plan.choice.source),
            "warning": r.plan.choice.warning,
            "uses_gpu": r.plan.uses_gpu,
            "reason": r.plan.reason,
            "stages": r.plan.stages.names(),
            "auto_stages": GpuPolicy::Auto.stages().names(),
            "all_stages": GpuStages::ALL.names(),
        },
        "memory_budget": {
            "cpu_bytes": r.plan.budget.cpu_bytes,
            "gpu_bytes": r.plan.budget.gpu_bytes,
            "shared_memory": r.plan.shared_memory,
        },
        "render_path": {
            "demosaic": if r.neural.available { "neural" } else { "classical" },
            "neural_available": r.neural.available,
            "reason": r.neural.reason,
            "device": "cpu",
        },
        "apple_auto": apple_auto_value(&r.apple_auto),
        "model_weights": model_values(&r.weights),
        "illuminant_models": model_values(&r.illuminant_models),
        "profiles": r.profiles.iter().map(|(id, version, name)| json!({ "id": id, "version": version, "name": name })).collect::<Vec<_>>(),
        "looks": {
            "built_in": r.looks_built_in,
            "imported": r.looks_imported,
            "missing": r.looks_missing,
            "display_luts": r.display_luts,
        },
        "preferences": {
            "path": lossy(&r.preferences_path),
            "exists": r.preferences_exist,
            "problems": r.preference_warnings,
        },
        "looks_folder": lossy(&r.looks_folder),
        "data_folder": lossy(&r.data_folder),
    })
}

/// The `GPU policy` row: the setting, where it came from, and what it comes to here.
fn policy_text(plan: &GpuPlan) -> String {
    let mut text = plan.choice.to_string();
    if plan.uses_gpu {
        // The stages the policy lets the device run; which of them actually run there is
        // `GpuStages`'s business, so this says "may", not "does".
        text.push_str(&format!(": uses the GPU, which may run {}", plan.stages));
    } else {
        text.push_str(": uses the CPU");
        if let Some(reason) = &plan.reason {
            text.push_str(&format!(", {reason}"));
        }
    }
    if let Some(warning) = &plan.choice.warning {
        text.push_str(&format!(". {warning}"));
    }
    text
}

fn short(hash: &str) -> &str {
    &hash[..hash.len().min(16)]
}

fn to_text(r: &Report) -> String {
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut row = |label: &str, value: String| rows.push((label.to_owned(), value));
    row("Version", format!("ym {}", version_text()));
    row(
        "System",
        format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
    );
    match &r.gpu {
        Ok(i) => {
            row(
                "GPU",
                format!(
                    "{} ({}, {}{})",
                    i.name,
                    i.backend,
                    i.device_type,
                    if i.is_software { ", software" } else { "" }
                ),
            );
            row(
                "GPU driver",
                format!("{} {}", i.driver, i.driver_info).trim().to_owned(),
            );
            row(
                "GPU storage",
                format!(
                    "{} (native f16: {}), budget {:.1} GiB",
                    i.storage_mode.name(),
                    if i.shader_f16 { "yes" } else { "no" },
                    i.memory_budget_bytes as f64 / f64::from(1u32 << 30)
                ),
            );
            row(
                "GPU limits",
                format!(
                    "buffer {} MiB, storage binding {} MiB, texture {} px, {} storage buffers, {} invocations",
                    i.limits.max_buffer_size >> 20,
                    i.limits.max_storage_buffer_binding_size >> 20,
                    i.limits.max_texture_dimension_2d,
                    i.limits.max_storage_buffers_per_shader_stage,
                    i.limits.max_compute_invocations_per_workgroup
                ),
            );
        }
        Err(reason) => row("GPU", format!("none, CPU fallback ({reason})")),
    }
    row("GPU policy", policy_text(&r.plan));
    row(
        "GPU stages",
        format!(
            "{} for Auto; On and Require run {}",
            GpuPolicy::Auto.stages(),
            GpuStages::ALL
        ),
    );
    row(
        "Memory budget",
        format!(
            "CPU {:.1} GiB, GPU {:.1} GiB{}",
            r.plan.budget.cpu_bytes as f64 / f64::from(1u32 << 30),
            r.plan.budget.gpu_bytes as f64 / f64::from(1u32 << 30),
            if r.plan.shared_memory {
                " (the GPU shares the memory of the CPU)"
            } else {
                ""
            }
        ),
    );
    row(
        "Render path",
        if r.neural.available {
            "neural demosaic".to_owned()
        } else {
            format!(
                "classical demosaic on the CPU. {}",
                r.neural.reason.as_deref().unwrap_or("")
            )
            .trim()
            .to_owned()
        },
    );
    row("Apple Auto", apple_auto_text(&r.apple_auto));
    if r.weights.is_empty() {
        row("Model weights", "none installed".to_owned());
    } else {
        for (i, m) in r.weights.iter().enumerate() {
            row(
                if i == 0 { "Model weights" } else { "" },
                format!("{}  sha256 {}", m.name, short(&m.sha256)),
            );
        }
    }
    for (i, m) in r.illuminant_models.iter().enumerate() {
        row(
            if i == 0 { "Illuminant models" } else { "" },
            format!(
                "{}  sha256 {}",
                m.name.trim_start_matches("illum/"),
                short(&m.sha256)
            ),
        );
    }
    for (i, (id, version, _)) in r.profiles.iter().enumerate() {
        row(
            if i == 0 { "Camera profiles" } else { "" },
            format!("{id} {version}"),
        );
    }
    row(
        "Looks",
        format!(
            "{} built in, {} imported{}, {} display {}",
            r.looks_built_in,
            r.looks_imported,
            if r.looks_missing > 0 {
                format!(" ({} need importing again)", r.looks_missing)
            } else {
                String::new()
            },
            r.display_luts,
            if r.display_luts == 1 { "LUT" } else { "LUTs" }
        ),
    );
    row(
        "Preferences",
        format!(
            "{}{}",
            path_text(&r.preferences_path),
            if r.preferences_exist {
                if r.preference_warnings > 0 {
                    ", has problems and was read with defaults where needed"
                } else {
                    ""
                }
            } else {
                " (not created yet)"
            }
        ),
    );
    row("Looks folder", path_text(&r.looks_folder));
    row("Data folder", path_text(&r.data_folder));

    let width = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    let mut text = format!("{INCLUDES}\n\n");
    for (label, value) in rows {
        text.push_str(&format!("{label:<width$}  {value}\n"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_report_says_what_it_includes_and_what_it_leaves_out() {
        assert!(INCLUDES.contains("does not include photos, file names, locations"));
    }

    fn status(available: bool, preference: bool, own: &str) -> AppleAutoStatus {
        let text = effective_instructions(own);
        let info = ym_core::apple_auto::AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)")
            .with_instructions(text);
        AppleAutoStatus {
            available,
            reason: (!available).then_some(ym_core::apple_auto::UnavailableReason::MacosTooOld),
            info: available.then_some(info),
            preference,
            mode: AppleAutoMode::FromPreference,
            active: available && preference,
            prompt: PROMPT_VERSION,
            table: TABLE_VERSION,
            instructions: InstructionsId::of(text),
            own_instructions: own.to_owned(),
            default_instructions: DEFAULT_INSTRUCTIONS.to_owned(),
            instructions_max_chars: INSTRUCTIONS_MAX_CHARS,
        }
    }

    #[test]
    fn the_apple_auto_row_names_the_model_or_why_not() {
        assert_eq!(
            apple_auto_text(&status(true, true, "")),
            "available · Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2"
        );
        assert!(
            apple_auto_text(&status(true, false, ""))
                .ends_with("table 2 · turned off in Preferences")
        );
        let edited = InstructionsId::of("Moody.").instructions_sha256;
        assert_eq!(
            apple_auto_text(&status(true, true, "Moody.")),
            format!(
                "available · Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2 · edited instructions ({edited})"
            )
        );
        assert_eq!(
            apple_auto_text(&status(false, true, "")),
            "not available (it needs macOS 27 or later)"
        );
        let sha = InstructionsId::default().instructions_sha256;
        assert_eq!(
            apple_auto_value(&status(false, true, "")),
            json!({ "available": false, "reason": "macos_too_old", "model": null, "os": null,
                    "preference": true, "prompt": 3, "table": 2,
                    "instructions": "default", "instructions_sha256": sha })
        );
        assert_eq!(
            apple_auto_value(&status(true, false, "Moody.")),
            json!({ "available": true, "reason": null, "model": "AFM 3 Core", "os": "27.0.1 (27A404)",
                    "preference": false, "prompt": 3, "table": 2,
                    "instructions": "edited", "instructions_sha256": edited })
        );
    }

    #[test]
    fn hashes_are_shortened_for_the_text_only() {
        assert_eq!(short("0123456789abcdef0123"), "0123456789abcdef");
        assert_eq!(short("abc"), "abc");
    }
}
