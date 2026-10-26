//! `ym doctor`: the diagnostics that go into a bug report (`docs/10-cli.md`).
//!
//! The report says what it includes and nothing else: the version, the GPU, the render path,
//! whether Apple Intelligence, Claude Code and Antigravity are available (which model answers and,
//! for the two programs, where they were found, `docs/19` section 8, `docs/22` section 13.3), the
//! model weights with their hashes, the camera profiles, how many looks are imported and the folders
//! Yellowmoon uses. No photo, file name, location or look name is in it.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use ym_core::app::gpu::{
    CpuReason, GpuOutcome, GpuPolicy, GpuStages, PolicyChoice, PolicySource, open_gpu,
    probe_adapter,
};
use ym_core::apple_auto::{PROMPT_VERSION, TABLE_VERSION, UnavailableReason};
use ym_core::auto_provider::{
    AGY_MIN_VERSION, AdvisorDetails, AutoChoice, AutoProvider, AutoStatus, CLAUDE_DEFAULT_MODEL,
    CLAUDE_MIN_VERSION, IMAGE_ENCODING, ProviderState, ProviderStatus, SCHEMA_VERSION, StatusInput,
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
pub const INCLUDES: &str = "This report lists the version, the GPU, the render path, whether Apple Intelligence, Claude Code and Antigravity are available (with the places where Claude Code and Antigravity were found), the model weights and their hashes, the camera profiles, how many looks are imported, and the folders Yellowmoon uses. It does not include photos, file names, locations or look names.";

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

/// The Auto providers as the report says them: the status the app's Preferences would show, and what each
/// advisor said besides (how the program is signed in).
struct AutoReport {
    status: AutoStatus,
    /// By [`AutoProvider::slot`].
    details: [Option<AdvisorDetails>; 3],
}

/// Everything the report says.
struct Report {
    gpu: Result<GpuInfo, String>,
    plan: GpuPlan,
    neural: NeuralStatus,
    auto: AutoReport,
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
        auto: auto_report(&prefs.get()),
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

/// The Auto providers on this computer, as `Core::auto_status` gives them to the app's Preferences
/// (`docs/22` sections 11 and 13.3), without opening a session: the advisors the app and `ym` use
/// (`ym_apple::system_advisor`, `ym_agents::system_advisors`), every one asked to read its
/// availability now and waited for (up to 30 s: a program that hangs ends at its own probe limit), the
/// saved choice and consent, and the person's own instructions (`""` for the default).
fn auto_report(prefs: &ym_core::preferences::Preferences) -> AutoReport {
    let advisors = crate::runtime::system_provider_advisors();
    crate::runtime::register_advisors(&advisors);
    advisors.refresh(None);
    advisors.settle(crate::runtime::AUTO_SETTLE);
    let details: [Option<AdvisorDetails>; 3] = std::array::from_fn(|slot| {
        AutoProvider::from_slot(slot)
            .and_then(|p| advisors.get(p))
            .map(|a| a.details())
    });
    let readings = std::array::from_fn(|slot| {
        let advisor = AutoProvider::from_slot(slot).and_then(|p| advisors.get(p))?;
        Some((advisor.availability(), advisor.details()))
    });
    let status = AutoStatus::build(&StatusInput {
        has_advisors: true,
        readings,
        preference: prefs.auto_provider,
        choice: AutoChoice::FromPreference,
        consent: prefs.auto_consent,
        own_instructions: &prefs.apple_auto_instructions,
    });
    AutoReport { status, details }
}

/// The model a provider is asked for until a request has answered: the alias or slug (`docs/22`
/// section 13.3). A developer's variable replaces the default (`docs/22` D18).
fn asked_model(provider: AutoProvider) -> Option<String> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    match provider {
        AutoProvider::Claude => {
            Some(var("YELLOWMOON_CLAUDE_MODEL").unwrap_or_else(|| CLAUDE_DEFAULT_MODEL.to_owned()))
        }
        AutoProvider::Agy => {
            Some(var("YELLOWMOON_AGY_MODEL").unwrap_or_else(|| "default".to_owned()))
        }
        AutoProvider::Yellowmoon | AutoProvider::Apple => None,
    }
}

/// The `auto` object of `doctor --json` (`docs/22` section 13.3): the saved choice, the consent, the
/// provider in effect, the instructions' digest and each provider. Keys are sorted. `path` is the
/// person's own and appears here only, never in a report of a photo.
fn auto_value(r: &AutoReport) -> Value {
    let s = &r.status;
    let mut providers = serde_json::Map::new();
    for status in &s.providers {
        let reason = status.reason.and_then(|r| serde_json::to_value(r).ok());
        let mut entry = json!({ "state": status.state, "reason": reason });
        if let (Value::Object(map), Some(slot)) = (&mut entry, status.provider.slot()) {
            let details = r.details[slot].as_ref();
            let info = status.info.as_ref();
            match status.provider {
                AutoProvider::Apple => {
                    map.insert("model".to_owned(), json!(info.map(|i| i.model.as_str())));
                    map.insert("os".to_owned(), json!(info.map(|i| i.os.as_str())));
                    map.insert("prompt".to_owned(), json!(PROMPT_VERSION));
                }
                _ => {
                    map.insert("path".to_owned(), json!(status.path.as_deref().map(lossy)));
                    map.insert("tool_version".to_owned(), json!(status.tool_version));
                    map.insert(
                        "model".to_owned(),
                        json!(
                            info.map(|i| i.model.clone())
                                .or_else(|| asked_model(status.provider))
                        ),
                    );
                    let api = info
                        .and_then(|i| i.tool.as_ref())
                        .and_then(|t| t.api.clone())
                        .or_else(|| details.and_then(|d| d.api.clone()));
                    let auth = info
                        .and_then(|i| i.tool.as_ref())
                        .and_then(|t| t.auth.clone())
                        .or_else(|| details.and_then(|d| d.auth.clone()));
                    map.insert("api".to_owned(), json!(api));
                    map.insert("auth".to_owned(), json!(auth));
                    map.insert("prompt".to_owned(), json!(status.provider.prompt_version()));
                    map.insert("schema".to_owned(), json!(SCHEMA_VERSION));
                    map.insert("image".to_owned(), json!(IMAGE_ENCODING));
                }
            }
        }
        providers.insert(status.provider.id().to_owned(), entry);
    }
    json!({
        "preference": s.preference.id(),
        "effective": s.effective.id(),
        "consent": { "claude": s.consent.claude, "agy": s.consent.agy },
        "instructions": s.instructions.instructions,
        "instructions_sha256": s.instructions.instructions_sha256,
        "table": s.table,
        "providers": providers,
    })
}

/// The `apple_auto` object of `doctor --json`, **kept one version, deprecated**, computed for the Apple
/// provider only (`docs/19` section 8, `docs/22` section 13.3): `{ "available", "reason", "model",
/// "os", "preference", "prompt", "table", "instructions", "instructions_sha256" }`, where
/// `preference` says whether the saved choice is Apple Intelligence.
fn apple_auto_value(s: &AutoStatus) -> Value {
    let apple = s.provider(AutoProvider::Apple);
    let info = apple.and_then(|a| a.info.as_ref());
    json!({
        "available": apple.is_some_and(|a| a.state == ProviderState::Available),
        "reason": apple.and_then(|a| a.reason).and_then(|r| serde_json::to_value(r).ok()),
        "model": info.map(|i| i.model.as_str()),
        "os": info.map(|i| i.os.as_str()),
        "preference": s.preference == AutoProvider::Apple,
        "prompt": PROMPT_VERSION,
        "table": TABLE_VERSION,
        "instructions": s.instructions.instructions,
        "instructions_sha256": s.instructions.instructions_sha256,
    })
}

/// The `Auto grading` row: `Claude Code (saved in Preferences)`, or `Default Auto`, with
/// ` · not agreed to yet; naming it with --auto-provider is enough for ym` when it is a program whose
/// consent is not saved.
fn auto_grading_text(s: &AutoStatus) -> String {
    let mut text = if s.preference == AutoProvider::Yellowmoon {
        s.preference.name().to_owned()
    } else {
        format!("{} (saved in Preferences)", s.preference.name())
    };
    if s.preference.is_cloud() && !s.consent.get(s.preference) {
        text.push_str(" · not agreed to yet; naming it with --auto-provider is enough for ym");
    }
    text
}

/// The row of one provider (`docs/22` section 13.3): `available · Apple Intelligence · AFM 3 Core ·
/// macOS 27.0.1 (27A404) · prompt 3 · table 2` (with ` · edited instructions (<digest>)` when the person
/// wrote their own), `available · Claude Code 2.1.310 · sonnet · prompt 1 · schema 1 · table 2 ·
/// /Users/me/.local/bin/claude`, `installed but not signed in · 1.4.7 · <path>`, `too old (2.1.100;
/// 2.1.169 or later is needed) · <path>`, `didn't answer its check · <path>`, `found only as a script
/// that Yellowmoon can't run safely`, `not installed`, `still checking`, or for Apple Intelligence `not
/// available (this is not a Mac)`.
fn provider_text(p: &ProviderStatus) -> String {
    let path = p.path.as_deref().map(path_text);
    let version = p.tool_version.as_deref().map(line_text);
    let mut parts: Vec<String> = Vec::new();
    match (p.provider, p.state) {
        (AutoProvider::Yellowmoon, _) => return "always available".to_owned(),
        (_, ProviderState::Available) => {
            if let Some(info) = &p.info {
                parts.push(format!("available · {}", line_text(&info.version_string())));
            }
            parts.extend(path);
        }
        (AutoProvider::Apple, _) => {
            return match p.reason {
                Some(reason) => format!("not available ({})", reason.words()),
                None => "not available".to_owned(),
            };
        }
        (_, ProviderState::SignedOut) => {
            parts.push("installed but not signed in".to_owned());
            parts.extend(version);
            parts.extend(path);
        }
        (provider, ProviderState::TooOld) => {
            let min = if provider == AutoProvider::Agy {
                AGY_MIN_VERSION
            } else {
                CLAUDE_MIN_VERSION
            };
            let have = version.map_or(String::new(), |v| format!("{v}; "));
            parts.push(format!("too old ({have}{min} or later is needed)"));
            parts.extend(path);
        }
        (_, ProviderState::NotInstalled) => parts.push("not installed".to_owned()),
        (_, ProviderState::Checking) => parts.push("still checking".to_owned()),
        (_, ProviderState::Unavailable) => {
            if p.reason == Some(UnavailableReason::UnsafeLauncher) {
                parts.push("found only as a script that Yellowmoon can't run safely".to_owned());
            } else {
                parts.push("didn't answer its check".to_owned());
                parts.extend(path);
            }
        }
    }
    parts.join(" · ")
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
        "apple_auto": apple_auto_value(&r.auto.status),
        "auto": auto_value(&r.auto),
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
    row("Auto grading", auto_grading_text(&r.auto.status));
    for p in &r.auto.status.providers {
        if p.provider != AutoProvider::Yellowmoon {
            let label = if p.provider == AutoProvider::Agy {
                "Antigravity (agy)"
            } else {
                p.provider.name()
            };
            row(label, provider_text(p));
        }
    }
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

    use ym_core::apple_auto::{AdvisorInfo, Availability, InstructionsId};
    use ym_core::auto_provider::{AutoConsent, ToolInfo};

    fn claude_info() -> AdvisorInfo {
        AdvisorInfo::cli(
            AutoProvider::Claude,
            "sonnet",
            ToolInfo {
                version: "2.1.310".to_owned(),
                api: Some("firstParty".to_owned()),
                auth: Some("oauth_token".to_owned()),
            },
        )
    }

    fn details(version: &str, path: &str) -> AdvisorDetails {
        AdvisorDetails {
            path: Some(PathBuf::from(path)),
            tool_version: Some(version.to_owned()),
            ..AdvisorDetails::default()
        }
    }

    /// A report where Apple is `apple`, Claude Code `claude` and Antigravity `agy`.
    fn report(
        apple: Availability,
        claude: (Availability, AdvisorDetails),
        agy: (Availability, AdvisorDetails),
        preference: AutoProvider,
        consent: AutoConsent,
        own: &str,
    ) -> AutoReport {
        let details = [None, Some(claude.1.clone()), Some(agy.1.clone())];
        let status = AutoStatus::build(&StatusInput {
            has_advisors: true,
            readings: [
                Some((apple, AdvisorDetails::default())),
                Some(claude),
                Some(agy),
            ],
            preference,
            choice: AutoChoice::FromPreference,
            consent,
            own_instructions: own,
        });
        AutoReport { status, details }
    }

    fn row(r: &AutoReport, p: AutoProvider) -> String {
        provider_text(r.status.provider(p).unwrap())
    }

    #[test]
    fn the_provider_rows_name_the_model_the_program_and_where_it_is() {
        use UnavailableReason as R;
        let apple = Availability::Available(AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)"));
        let ok = (
            Availability::Available(claude_info()),
            details("2.1.310", "/Users/me/.local/bin/claude"),
        );
        let out = |reason, d: AdvisorDetails| (Availability::Unavailable(reason), d);
        let r = report(
            apple.clone(),
            ok.clone(),
            out(R::SignedOut, details("1.4.7", "/Users/me/.local/bin/agy")),
            AutoProvider::Claude,
            AutoConsent {
                claude: true,
                agy: false,
            },
            "",
        );
        assert_eq!(
            row(&r, AutoProvider::Apple),
            "available · Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2"
        );
        assert_eq!(
            row(&r, AutoProvider::Claude),
            "available · Claude Code 2.1.310 · sonnet · prompt 1 · schema 1 · table 2 · /Users/me/.local/bin/claude"
        );
        assert_eq!(
            row(&r, AutoProvider::Agy),
            "installed but not signed in · 1.4.7 · /Users/me/.local/bin/agy"
        );
        assert_eq!(
            auto_grading_text(&r.status),
            "Claude Code (saved in Preferences)"
        );
        // Edited instructions are in the version string, before the path.
        let edited = InstructionsId::of("Moody.").instructions_sha256;
        let r = report(
            apple,
            ok,
            out(R::NotInstalled, AdvisorDetails::default()),
            AutoProvider::Claude,
            AutoConsent::default(),
            "Moody.",
        );
        assert_eq!(
            row(&r, AutoProvider::Claude),
            format!(
                "available · Claude Code 2.1.310 · sonnet · prompt 1 · schema 1 · table 2 · edited instructions ({edited}) · /Users/me/.local/bin/claude"
            )
        );
        assert_eq!(row(&r, AutoProvider::Agy), "not installed");
        // A saved program that was not agreed to yet says how `ym` can use it all the same.
        assert_eq!(
            auto_grading_text(&r.status),
            "Claude Code (saved in Preferences) · not agreed to yet; naming it with --auto-provider is enough for ym"
        );
        // The other states.
        let gone = Availability::Unavailable(R::NotAMac);
        let r = report(
            gone,
            out(R::TooOld, details("2.1.100", "/opt/claude")),
            out(R::ProbeFailed, details("1.4.7", "/opt/agy")),
            AutoProvider::Yellowmoon,
            AutoConsent::default(),
            "",
        );
        assert_eq!(
            row(&r, AutoProvider::Apple),
            "not available (this is not a Mac)"
        );
        assert_eq!(
            row(&r, AutoProvider::Claude),
            "too old (2.1.100; 2.1.169 or later is needed) · /opt/claude"
        );
        assert_eq!(
            row(&r, AutoProvider::Agy),
            "didn't answer its check · /opt/agy"
        );
        assert_eq!(auto_grading_text(&r.status), "Default Auto");
        let r = report(
            Availability::Unavailable(R::NotAMac),
            out(R::UnsafeLauncher, AdvisorDetails::default()),
            out(R::Checking, AdvisorDetails::default()),
            AutoProvider::Apple,
            AutoConsent::default(),
            "",
        );
        assert_eq!(
            row(&r, AutoProvider::Claude),
            "found only as a script that Yellowmoon can't run safely"
        );
        assert_eq!(row(&r, AutoProvider::Agy), "still checking");
    }

    #[test]
    fn the_auto_object_lists_every_provider_with_sorted_keys_and_the_path_here_only() {
        use UnavailableReason as R;
        let ok = (
            Availability::Available(claude_info()),
            details("2.1.310", "/Users/me/.local/bin/claude"),
        );
        let r = report(
            Availability::Unavailable(R::NotAMac),
            ok,
            (
                Availability::Unavailable(R::SignedOut),
                details("1.4.7", "/Users/me/.local/bin/agy"),
            ),
            AutoProvider::Claude,
            AutoConsent {
                claude: true,
                agy: false,
            },
            "",
        );
        let sha = InstructionsId::default().instructions_sha256;
        assert_eq!(
            auto_value(&r),
            json!({
                "preference": "claude", "effective": "claude",
                "consent": { "claude": true, "agy": false },
                "instructions": "default", "instructions_sha256": sha, "table": 2,
                "providers": {
                    "yellowmoon": { "state": "available", "reason": null },
                    "apple": { "state": "unavailable", "reason": "not_a_mac", "model": null, "os": null, "prompt": 3 },
                    "claude": { "state": "available", "reason": null, "path": "/Users/me/.local/bin/claude",
                                "tool_version": "2.1.310", "model": "sonnet", "api": "firstParty", "auth": "oauth_token",
                                "prompt": 1, "schema": 1, "image": "jpeg444q92@768" },
                    "agy": { "state": "signed_out", "reason": "signed_out", "path": "/Users/me/.local/bin/agy",
                             "tool_version": "1.4.7", "model": "default", "api": null, "auth": null,
                             "prompt": 1, "schema": 1, "image": "jpeg444q92@768" },
                }
            })
        );
        // The old object stays for one version, Apple only.
        assert_eq!(
            apple_auto_value(&r.status),
            json!({ "available": false, "reason": "not_a_mac", "model": null, "os": null,
                    "preference": false, "prompt": 3, "table": 2,
                    "instructions": "default", "instructions_sha256": sha })
        );
    }

    #[test]
    fn the_old_apple_auto_object_says_whether_apple_is_the_saved_choice() {
        let r = report(
            Availability::Available(AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)")),
            (
                Availability::Unavailable(UnavailableReason::NotInstalled),
                AdvisorDetails::default(),
            ),
            (
                Availability::Unavailable(UnavailableReason::NotInstalled),
                AdvisorDetails::default(),
            ),
            AutoProvider::Apple,
            AutoConsent::default(),
            "Moody.",
        );
        let edited = InstructionsId::of("Moody.").instructions_sha256;
        assert_eq!(
            apple_auto_value(&r.status),
            json!({ "available": true, "reason": null, "model": "AFM 3 Core", "os": "27.0.1 (27A404)",
                    "preference": true, "prompt": 3, "table": 2,
                    "instructions": "edited", "instructions_sha256": edited })
        );
    }

    #[test]
    fn hashes_are_shortened_for_the_text_only() {
        assert_eq!(short("0123456789abcdef0123"), "0123456789abcdef");
        assert_eq!(short("abc"), "abc");
    }
}
