//! `ym doctor --bench <file.ARW>`: what the render path costs on this machine, for one photo (`docs/10-cli.md`,
//! `docs/17-gpu.md` section 15, the Mac checklist of `tools/dev/mac-gpu-check.sh`).
//!
//! The report follows the graphics policy in force (`YELLOWMOON_GPU`, else `auto`), so the same command measures the
//! device path (`YELLOWMOON_GPU=on` or `require`) and the CPU reference (`YELLOWMOON_GPU=off`). It times
//!
//! * the **compile** of each kernel set on the device (`view`, `raw`, `tail`): the first launch on a machine with
//!   an empty shader cache is the slow one,
//! * the **decode** of the file,
//! * the **views**: the fit view in a 1440 x 960 and a 2880 x 1920 cell (cold and with the frame cache off), the
//!   first sharp frame at 100 % in a 1440 x 960 cell, a colour-only edit, a thumbnail, each with the wall time of
//!   its stages as the renderer reports it,
//! * the **exports**: full-size JPEG, 16-bit TIFF and 16-bit PNG, through the export queue the app and `ym render`
//!   share, with the time of the render and of the encode and the device the report says made the picture,
//! * the **repeat**: the same JPEG (long edge 2000 px) `--repeat` times, whose bytes must be equal (the 10 identical
//!   exports of the Mac checklist),
//! * the **peaks**: the host ledger and the device pool,
//! * the **GPU pass times** from timestamp queries, per pass label, where the adapter has them (Metal, Vulkan on
//!   hardware; none on a software adapter or the CPU path).
//!
//! The report names no file and no folder: the photo is identified by its camera model and size only.

use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use ym_core::app::gpu::{GpuOpener, GpuPolicy, GpuSlot};
use ym_core::export_queue::{ExportEvent, ExportJob};
use ym_core::pipeline::{
    FrameOptions, PhotoSource, RenderReport, RenderedFrame, StageTimings, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{ApplyScope, OpenOptions, PhotoStatus};
use ym_core::version::{PIPELINE_VERSION, app_version};
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists, Size, TiffCompression};
use ym_gpu::device::Gpu;
use ym_gpu::kernels::KernelSet;
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

use crate::exit::{self, Failure};
use crate::runtime::{CoreSettings, Runtime, interrupted};

/// The version of the shape of the report, for `tools/dev/mac-gpu-check.sh`.
pub const SCHEMA: u32 = 1;

/// How many exports the repeat check makes when `--repeat` is not given.
pub const DEFAULT_REPEAT: u32 = 10;

/// The long edge of the exports of the repeat check, in pixels: large enough for several tiles and bands.
const REPEAT_EDGE_PX: u32 = 2000;

/// A folder under the system's temporary folder that is removed again.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self, Failure> {
        let dir = std::env::temp_dir().join(format!("ym-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| {
            Failure::failed(format!(
                "Couldn't make a folder for the bench exports: {e}."
            ))
        })?;
        Ok(Self(dir))
    }

    fn sub(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Rounded to a tenth of a millisecond, so that the report is short.
fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn stage_values(t: &StageTimings) -> Value {
    json!({
        "frame_data": r1(t.frame_data_ms),
        "sigma": r1(t.sigma_ms),
        "raw": r1(t.raw_ms),
        "plans": r1(t.plans_ms),
        "sample": r1(t.sample_ms),
        "pass": r1(t.pass_ms),
        "output": r1(t.output_ms),
    })
}

fn device_word(report: &RenderReport) -> &'static str {
    match report.device {
        ym_core::pipeline::ComputeDevice::Gpu => "gpu",
        ym_core::pipeline::ComputeDevice::Cpu => "cpu",
    }
}

/// FNV-1a, 64 bit: enough to tell exports apart in a report.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// Fails with the exit code of Ctrl-C when it was pressed. Once the photo is developed
/// ([`Runtime::wait_developed`] says the command looks at the flag) the first Ctrl-C only sets it, so every phase of the
/// bench looks at it; the scratch folder and the device go with the return.
fn check_interrupted() -> Result<(), Failure> {
    if interrupted() {
        Err(Failure::new(exit::INTERRUPTED, "Interrupted.".to_owned()))
    } else {
        Ok(())
    }
}

/// The GPU time per pass label since the last call, as a list sorted by total time, longest first.
fn take_passes(gpu: Option<&Gpu>) -> Value {
    let Some(gpu) = gpu else {
        return json!([]);
    };
    let mut rows: Vec<_> = gpu.take_pass_times().into_iter().collect();
    rows.sort_by(|a, b| b.1.total_ms.total_cmp(&a.1.total_ms));
    Value::Array(
        rows.into_iter()
            .map(|(label, t)| {
                json!({
                    "label": label,
                    "count": t.count,
                    "total_ms": r1(t.total_ms),
                    "max_ms": r1(t.max_ms),
                })
            })
            .collect(),
    )
}

/// What a view phase reports.
struct Phase<'a> {
    gpu: Option<&'a Gpu>,
}

impl Phase<'_> {
    fn frame(
        &self,
        name: &str,
        cold: (f64, &RenderedFrame),
        warm: Option<(f64, &RenderedFrame)>,
    ) -> Value {
        let mut v = json!({
            "name": name,
            "cold_ms": r1(cold.0),
            "device": device_word(&cold.1.report),
            "scale": cold.1.report.scale,
            "tiles": cold.1.report.runs.tiles,
            "stages_ms": stage_values(&cold.1.report.timings),
            "notes": cold.1.report.notes,
        });
        if let Some((w, f)) = warm {
            v["warm_ms"] = json!(r1(w));
            v["warm_stages_ms"] = stage_values(&f.report.timings);
        }
        v["gpu_passes"] = take_passes(self.gpu);
        v
    }
}

/// Runs the bench on `path` and returns the report.
///
/// # Errors
///
/// A [`Failure`] when the file cannot be developed, `require` finds no device, or an export fails.
pub fn run(path: &Path, repeat: u32) -> Result<Value, Failure> {
    let choice = GpuPolicy::from_environment(None);
    let slot: GpuSlot = GpuOpener::new(choice.policy, MemoryBudget::detect())
        .with_timestamps(true)
        .open();
    if let Some(reason) = slot.failure() {
        return Err(Failure::new(
            exit::NO_GPU,
            format!(
                "No usable GPU was found ({reason}). The setting is {}, so nothing was timed.",
                choice.policy
            ),
        ));
    }
    // No advisor: the bench times the render path, and a call to Apple's model is not part of it
    // (Apple Auto off, `docs/19`).
    let rt = Runtime::open(&CoreSettings {
        jobs: Some(1),
        forced_profile: None,
        apple_auto: None,
        apple_auto_instructions: None,
    })?;
    rt.core.attach_gpu(&slot);
    let gpu = slot.gpu().cloned();
    let gpu_ref = gpu.as_deref();

    // The kernels, set by set, on a device that has compiled none of them.
    let mut compile = serde_json::Map::new();
    if let Some(g) = gpu_ref {
        for set in [KernelSet::View, KernelSet::Raw, KernelSet::Tail] {
            let t = Instant::now();
            g.run(JobSpec::new(Priority::Export), move |ctx, _| {
                ctx.warm_up_set(set)
            })
            .map_err(|e| {
                Failure::failed(format!("The {} kernels do not compile: {e}.", set.name()))
            })?;
            compile.insert(set.name().to_owned(), json!(r1(ms(t.elapsed()))));
        }
    }

    // Decode and develop.
    let started = Instant::now();
    let outcome = rt
        .core
        .session
        .open(&[path.to_path_buf()], OpenOptions::default());
    let Some(&id) = outcome.added.first() else {
        return Err(Failure::failed(
            "That file is not a photo Yellowmoon can open. Try an ARW file.".to_owned(),
        ));
    };
    if !rt.wait_developed(|_| {}) {
        return Err(Failure::new(exit::INTERRUPTED, "Interrupted.".to_owned()));
    }
    let decode_ms = ms(started.elapsed());
    let state = rt.core.session.state();
    if let Some(summary) = state.photos.iter().find(|p| p.id == id)
        && let PhotoStatus::Skipped { reason } = &summary.status
    {
        return Err(Failure::failed(format!(
            "That photo can't be developed: {reason}"
        )));
    }
    let view = rt
        .core
        .session
        .view(id)
        .map_err(|e| Failure::failed(e.to_string()))?;
    let raw = rt
        .core
        .session
        .raw_image(id)
        .map_err(|e| Failure::failed(e.to_string()))?;
    let (model, raw_w, raw_h) = (view.meta.model.clone(), view.meta.width, view.meta.height);
    let src = PhotoSource::from_view(&view, raw);
    let r = &rt.core.renderer;
    let cancel = CancelToken::new();
    let phase = Phase { gpu: gpu_ref };
    // Whatever the compile loop and the open left in the log is not a view.
    let _ = take_passes(gpu_ref);

    let render_error = |e: ym_core::pipeline::RenderError| Failure::failed(e.to_string());
    let frame = |s: &PhotoSource, st: &ViewportState| -> Result<(f64, RenderedFrame), Failure> {
        check_interrupted()?;
        let t = Instant::now();
        let f = r
            .viewport_frame(
                s,
                st,
                &FrameOptions {
                    no_cache: true,
                    ..FrameOptions::default()
                },
                &cancel,
            )
            .map_err(render_error)?;
        Ok((ms(t.elapsed()), f))
    };

    let mut views: Vec<Value> = Vec::new();
    for (name, w, h) in [
        ("fit, 1440 x 960 cell", 1440.0, 960.0),
        ("fit, 2880 x 1920 cell (retina)", 2880.0, 1920.0),
    ] {
        let st = ViewportState::new(ViewportRect::cell(w, h, 1.0));
        r.forget(src.id);
        let cold = frame(&src, &st)?;
        let warm = frame(&src, &st)?;
        views.push(phase.frame(name, (cold.0, &cold.1), Some((warm.0, &warm.1))));
    }
    // The first sharp frame at 100 %: the tiles of a 1440 x 960 cell, cold and with their raw work done.
    {
        let image = r.final_size(&src, &cancel).map_err(render_error)?;
        let mut st = ViewportState::new(ViewportRect::cell(1440.0, 960.0, 1.0));
        st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
        r.forget(src.id);
        let cold = frame(&src, &st)?;
        let warm = frame(&src, &st)?;
        views.push(phase.frame(
            "100 %, 1440 x 960 cell",
            (cold.0, &cold.1),
            Some((warm.0, &warm.1)),
        ));
    }
    // A colour-only edit: the raw group is cached, only the colour and grade stages run.
    {
        let st = ViewportState::new(ViewportRect::cell(1440.0, 960.0, 1.0));
        let base = frame(&src, &st)?;
        let mut s = src.settings.clone();
        let contrast = s.contrast.resolve(0.0);
        s.contrast = ym_core::settings::PercentSetting::manual_value(if contrast > 50.0 {
            5.0
        } else {
            contrast + 20.0
        });
        let edited = frame(&src.with_settings(s), &st)?;
        let mut v = phase.frame(
            "colour-only edit, 1440 x 960 cell",
            (edited.0, &edited.1),
            Some((base.0, &base.1)),
        );
        v["raw_work"] = json!(edited.1.report.runs.raw_work());
        views.push(v);
    }
    // The thumbnail.
    {
        check_interrupted()?;
        r.forget(src.id);
        let t = Instant::now();
        let png = r.thumbnail_png(&src, &cancel).map_err(render_error)?;
        views.push(json!({
            "name": "thumbnail",
            "cold_ms": r1(ms(t.elapsed())),
            "bytes": png.len(),
            "gpu_passes": take_passes(gpu_ref),
        }));
    }

    // Exports through the queue.
    let scratch = Scratch::new()?;
    let exports = vec![
        export_row(&rt, "jpeg", &scratch, ExportOptions::default(), gpu_ref)?,
        export_row(
            &rt,
            "tiff 16",
            &scratch,
            ExportOptions {
                format: FormatKind::Tiff,
                bit_depth: BitDepth::Sixteen,
                tiff_compression: TiffCompression::Lzw,
                ..ExportOptions::default()
            },
            gpu_ref,
        )?,
        export_row(
            &rt,
            "png 16",
            &scratch,
            ExportOptions {
                format: FormatKind::Png,
                bit_depth: BitDepth::Sixteen,
                ..ExportOptions::default()
            },
            gpu_ref,
        )?,
    ];

    // The same export many times: the bytes must not move.
    let mut digests: Vec<String> = Vec::new();
    let mut identical = true;
    let mut repeat_ms: Vec<f64> = Vec::new();
    for i in 0..repeat {
        check_interrupted()?;
        let dir = scratch.sub(&format!("repeat-{i}"));
        let opts = ExportOptions {
            size: Size::LongEdge { px: REPEAT_EDGE_PX },
            ..ExportOptions::default()
        };
        let (done, wall) = export_once(&rt, opts, &dir)?;
        let bytes = std::fs::read(&done.path)
            .map_err(|e| Failure::failed(format!("Couldn't read back a bench export: {e}.")))?;
        let digest = format!("{:016x}", fnv(&bytes));
        if digests.first().is_some_and(|first| *first != digest) {
            identical = false;
        }
        digests.push(digest);
        repeat_ms.push(wall);
        let _ = std::fs::remove_dir_all(&dir);
    }
    let repeat_report = json!({
        "runs": repeat,
        "edge_px": REPEAT_EDGE_PX,
        "identical": identical,
        "digest": digests.first(),
        "wall_ms": repeat_ms.iter().map(|v| r1(*v)).collect::<Vec<_>>(),
    });

    // Peaks.
    let pool = match gpu_ref {
        Some(g) => g
            .run(JobSpec::new(Priority::Viewport), |ctx, _| {
                Ok(ctx.pool().stats())
            })
            .ok(),
        None => None,
    };
    let counters = gpu_ref.map(|g| g.counters());
    let adapter = slot.adapter();
    Ok(json!({
        "bench": SCHEMA,
        "includes": "Timings of one photo on this machine. It names no file and no folder.",
        "version": app_version(),
        "pipeline_version": PIPELINE_VERSION,
        "system": { "os": std::env::consts::OS, "arch": std::env::consts::ARCH },
        "gpu": {
            "policy": choice.policy.name(),
            "source": super::source_word(choice.source),
            "uses_gpu": slot.is_using(),
            "stages": slot.stages().names(),
            "adapter": adapter.map(|a| json!({
                "name": a.name, "backend": a.backend, "device_type": a.device_type,
                "software": a.is_software, "storage_mode": a.storage_mode.name(),
                "timestamp_query": a.timestamp_query,
                "memory_budget_bytes": a.memory_budget_bytes,
            })),
        },
        "photo": {
            "camera": model,
            "width": raw_w,
            "height": raw_h,
            "megapixels": r1(f64::from(raw_w) * f64::from(raw_h) / 1e6),
        },
        "compile_ms": Value::Object(compile),
        "decode_ms": r1(decode_ms),
        "views": views,
        "exports": exports,
        "repeat": repeat_report,
        "memory": {
            "budget": {
                "cpu_bytes": slot.budget().cpu_bytes,
                "gpu_bytes": slot.budget().gpu_bytes,
                "shared_memory": slot.shares_memory(),
            },
            "host_ledger_peak_bytes": r.ledger().peak(),
            "device_pool_peak_in_use_bytes": pool.map(|p| p.peak_in_use_bytes),
            "device_pool_peak_allocated_bytes": pool.map(|p| p.peak_allocated_bytes),
        },
        "device_counters": counters.map(|c| json!({
            "uploads": c.uploads, "uploaded_bytes": c.uploaded_bytes,
            "readbacks": c.readbacks, "readback_bytes": c.readback_bytes,
            "submissions": c.submissions, "dispatches": c.dispatches,
        })),
    }))
}

/// One export through the queue to `dir`: the report of the file and the wall time of the job.
fn export_once(
    rt: &Runtime,
    mut options: ExportOptions,
    dir: &Path,
) -> Result<(ym_core::export::ExportReport, f64), Failure> {
    options.destination = Some(dir.to_path_buf());
    options.if_exists = IfExists::Replace;
    let job = ExportJob {
        scope: ApplyScope::All,
        options,
    };
    let started = Instant::now();
    rt.core
        .exports
        .start(&rt.core.session, &job, false)
        .map_err(|e| Failure::failed(e.to_string()))?;
    let mut report = None;
    let mut cancelled = false;
    let mut last_event = Instant::now();
    loop {
        // Ctrl-C ends the queue's job at the next tile; the loop then waits for the queue to say it is idle.
        if interrupted() && !cancelled {
            cancelled = true;
            rt.core.exports.cancel_all();
        }
        match rt.events.recv_timeout(Duration::from_millis(100)) {
            Ok(ExportEvent::File(done)) => {
                last_event = Instant::now();
                report = Some(done.outcome.map_err(|f| Failure::failed(f.message)));
            }
            Ok(ExportEvent::Busy(false)) | Err(RecvTimeoutError::Disconnected) => break,
            Ok(_) => last_event = Instant::now(),
            Err(RecvTimeoutError::Timeout) => {
                if last_event.elapsed() > Duration::from_secs(900) {
                    break;
                }
            }
        }
    }
    if cancelled {
        return Err(Failure::new(exit::INTERRUPTED, "Interrupted.".to_owned()));
    }
    let wall = ms(started.elapsed());
    match report {
        Some(Ok(r)) => Ok((r, wall)),
        Some(Err(f)) => Err(f),
        None => Err(Failure::failed(
            "The bench export did not finish. Try again.".to_owned(),
        )),
    }
}

fn export_row(
    rt: &Runtime,
    name: &str,
    scratch: &Scratch,
    options: ExportOptions,
    gpu: Option<&Gpu>,
) -> Result<Value, Failure> {
    let dir = scratch.sub(&name.replace(' ', "-"));
    let (r, wall) = export_once(rt, options, &dir)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(json!({
        "name": name,
        "wall_ms": r1(wall),
        "render_ms": r1(r.timings.render_ms),
        "encode_ms": r1(r.timings.encode_ms),
        "bytes": r.bytes,
        "width": r.width,
        "height": r.height,
        "device": match r.device {
            ym_core::pipeline::ComputeDevice::Gpu => "gpu",
            ym_core::pipeline::ComputeDevice::Cpu => "cpu",
        },
        "notes": r.notes,
        "gpu_passes": take_passes(gpu),
    }))
}

/// The report as text: the same facts in short lines.
#[must_use]
pub fn to_text(v: &Value) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let f = |p: &str| v.pointer(p).and_then(Value::as_f64).unwrap_or(f64::NAN);
    let s = |p: &str| {
        v.pointer(p)
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned()
    };
    let _ = writeln!(out, "{}", v["includes"].as_str().unwrap_or(""));
    let _ = writeln!(
        out,
        "ym {} (pipeline {}) on {} {}",
        s("/version"),
        s("/pipeline_version"),
        s("/system/os"),
        s("/system/arch")
    );
    let _ = writeln!(
        out,
        "GPU policy: {} ({}); adapter: {} ({}, {}{}); timestamps: {}",
        s("/gpu/policy"),
        s("/gpu/source"),
        s("/gpu/adapter/name"),
        s("/gpu/adapter/backend"),
        s("/gpu/adapter/device_type"),
        if v.pointer("/gpu/adapter/software").and_then(Value::as_bool) == Some(true) {
            ", software"
        } else {
            ""
        },
        v.pointer("/gpu/adapter/timestamp_query")
            .and_then(Value::as_bool)
            .map_or("none", |b| if b { "on" } else { "off" })
    );
    let _ = writeln!(
        out,
        "Photo: {} {} x {} ({:.1} MP); decode {:.0} ms",
        s("/photo/camera"),
        f("/photo/width"),
        f("/photo/height"),
        f("/photo/megapixels"),
        f("/decode_ms")
    );
    if let Some(c) = v["compile_ms"].as_object() {
        // In the order the sets are warmed (the JSON object itself is sorted by name).
        let list: Vec<String> = ["view", "raw", "tail"]
            .iter()
            .filter_map(|k| c.get(*k).map(|t| (k, t)))
            .map(|(k, t)| format!("{k} {:.0} ms", t.as_f64().unwrap_or(f64::NAN)))
            .collect();
        if !list.is_empty() {
            let _ = writeln!(out, "Kernel compile: {}", list.join(", "));
        }
    }
    for view in v["views"].as_array().into_iter().flatten() {
        let _ = write!(
            out,
            "{:<34} cold {:>8.1} ms",
            view["name"].as_str().unwrap_or("?"),
            view["cold_ms"].as_f64().unwrap_or(f64::NAN)
        );
        if let Some(w) = view["warm_ms"].as_f64() {
            let _ = write!(out, ", warm {w:>8.1} ms");
        }
        if let Some(d) = view["device"].as_str() {
            let _ = write!(out, " [{d}]");
        }
        let _ = writeln!(out);
    }
    for e in v["exports"].as_array().into_iter().flatten() {
        let _ = writeln!(
            out,
            "Export {:<8} {:>9.0} ms (render {:.0}, encode {:.0}), {} x {}, {} bytes [{}]",
            e["name"].as_str().unwrap_or("?"),
            e["wall_ms"].as_f64().unwrap_or(f64::NAN),
            e["render_ms"].as_f64().unwrap_or(f64::NAN),
            e["encode_ms"].as_f64().unwrap_or(f64::NAN),
            e["width"],
            e["height"],
            e["bytes"],
            e["device"].as_str().unwrap_or("?")
        );
    }
    let _ = writeln!(
        out,
        "Repeat: {} exports of the long edge {} px, bytes {}",
        v["repeat"]["runs"],
        v["repeat"]["edge_px"],
        if v["repeat"]["identical"].as_bool() == Some(true) {
            "identical"
        } else {
            "DIFFER"
        }
    );
    let _ = writeln!(
        out,
        "Peak memory: host ledger {:.0} MiB; device pool in use {} MiB, allocated {} MiB",
        f("/memory/host_ledger_peak_bytes") / 1_048_576.0,
        v.pointer("/memory/device_pool_peak_in_use_bytes")
            .and_then(Value::as_u64)
            .map_or_else(|| "-".to_owned(), |b| (b >> 20).to_string()),
        v.pointer("/memory/device_pool_peak_allocated_bytes")
            .and_then(Value::as_u64)
            .map_or_else(|| "-".to_owned(), |b| (b >> 20).to_string())
    );
    // The heaviest GPU passes over the whole run, when the adapter timed them.
    let mut totals: std::collections::BTreeMap<String, (u64, f64)> =
        std::collections::BTreeMap::new();
    let mut collect = |rows: &Value| {
        for p in rows.as_array().into_iter().flatten() {
            let t = totals
                .entry(p["label"].as_str().unwrap_or("?").to_owned())
                .or_default();
            t.0 += p["count"].as_u64().unwrap_or(0);
            t.1 += p["total_ms"].as_f64().unwrap_or(0.0);
        }
    };
    for part in ["views", "exports"] {
        for item in v[part].as_array().into_iter().flatten() {
            collect(&item["gpu_passes"]);
        }
    }
    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    if !rows.is_empty() {
        let _ = writeln!(out, "GPU passes by timestamp queries (the longest):");
        for (label, (count, total)) in rows.into_iter().take(12) {
            let _ = writeln!(out, "  {label:<40} {count:>6} x  {total:>10.1} ms");
        }
    }
    out
}
