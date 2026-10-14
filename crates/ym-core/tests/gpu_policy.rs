//! The graphics policy end to end (`docs/gpu-plan.md` task G-01): one place opens the device, a
//! renderer can be given the device and the memory budget after it is made, and nothing about the
//! default render changes in this version. The policy matrix itself (four policies against a
//! hardware adapter, a software one and none, flag, environment and default) is in
//! `ym_core::app::gpu`'s unit tests.
//!
//! The tests that need an adapter are skipped, loudly, when none opens, unless `YM_REQUIRE_GPU=1`
//! (the `gpu-lavapipe` job, the Metal job) turns that into a failure.
mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::app::gpu::{GpuOpener, GpuPolicy, GpuStages, open_gpu};
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{FullRequest, FullTarget, NeutralOnly, RenderOptions, Renderer};
use ym_core::scheduler::MemoryBudget;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const GIB: u64 = 1 << 30;

/// The device of this machine, or `None` (said loudly) when there is no adapter.
fn device(test: &str) -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TEST SKIPPED in gpu_policy::{test}: {e}\n"
            );
            None
        }
    }
}

// --- one place opens the device ---------------------------------------------------------------

/// The source files of the application, the command line, the core and the tools, below `root`.
fn sources(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(root) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            // Test folders may start a device of their own; the rule is about shipped code.
            if !matches!(
                name.as_str(),
                "tests" | "target" | "node_modules" | "examples"
            ) {
                sources(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_code_outside_open_gpu_creates_a_device() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    for crate_dir in std::fs::read_dir(workspace.join("crates"))
        .expect("crates")
        .flatten()
    {
        // `ym-gpu` is the device layer itself: `Gpu::start` and `GpuContext::open` are its own.
        if crate_dir.file_name() != "ym-gpu" {
            sources(&crate_dir.path().join("src"), &mut files);
        }
    }
    sources(&workspace.join("src-tauri/src"), &mut files);
    for tool in std::fs::read_dir(workspace.join("tools"))
        .expect("tools")
        .flatten()
    {
        sources(&tool.path().join("src"), &mut files);
    }
    assert!(files.len() > 100, "the scan found {} files", files.len());

    // What creates a compute device (or an instance to find one with).
    let needles = [
        "Gpu::start(",
        "GpuContext::open(",
        ".request_device(",
        ".request_adapter(",
        "wgpu::Instance::new(",
    ];
    // The one allowed use of each, with the reason.
    let allowed = |file: &str, needle: &str| {
        // The only place: `SystemProbe` calls `Gpu::start`.
        (file.ends_with("crates/ym-core/src/app/gpu.rs") && needle == "Gpu::start(")
            // The Mac's native viewport owns a *presentation* device: it draws the RGBA8 frames
            // the core made into the window's surface and computes nothing (`docs/gpu-plan.md`
            // section 6, "Native viewport (Mac)": unchanged).
            || file.ends_with("src-tauri/src/native.rs")
    };
    let mut offences = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("read a source file");
        let name = file.to_string_lossy().replace('\\', "/");
        for (n, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for needle in needles {
                if line.contains(needle) && !allowed(&name, needle) {
                    offences.push(format!("{name}:{}: {}", n + 1, code));
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "a device is created outside ym_core::app::gpu::open_gpu:\n{}",
        offences.join("\n")
    );
    // And the allowed place really is the one the application and `ym` use.
    let api = std::fs::read_to_string(workspace.join("src-tauri/src/api.rs")).expect("api.rs");
    assert!(
        !api.contains("fn start_gpu"),
        "the app's own opener is gone"
    );
    let cli = std::fs::read_to_string(workspace.join("crates/ym-cli/src/runtime.rs"))
        .expect("runtime.rs");
    assert!(!cli.contains("fn probe_gpu"), "the CLI's own probe is gone");
}

// --- a renderer takes its device afterwards ----------------------------------------------------

#[test]
fn a_renderer_takes_the_device_the_stages_and_the_budget_after_it_is_made() {
    let Some(gpu) = device("a_renderer_takes_the_device") else {
        return;
    };
    let r = Renderer::neutral_only().expect("render thread");
    assert!(r.gpu_info().is_none());
    let before = r.options();
    assert!(before.gpu.is_none());
    // Nothing is held back by default: a device given without a policy runs every stage.
    assert_eq!(before.gpu_stages, GpuStages::ALL);

    r.set_gpu(Some(Arc::clone(&gpu)), GpuStages::SIGNED_OFF);
    assert_eq!(r.gpu_info().map(|i| i.name), Some(gpu.info().name.clone()));
    assert_eq!(r.options().gpu_stages, GpuStages::SIGNED_OFF);
    // An options snapshot taken earlier does not change under its holder.
    assert!(before.gpu.is_none());

    let lowered = MemoryBudget::default().with_cpu_bytes(GIB);
    r.set_budget(lowered);
    assert_eq!(r.options().budget, lowered);
    assert_eq!(
        r.ledger().budget(),
        GIB,
        "the ledger of full-size renders follows"
    );
    // Everything else the renderer was made with is as it was.
    assert_eq!(r.options().tile_px, before.tile_px);
    assert_eq!(r.options().full_tile_px, before.full_tile_px);
    assert_eq!(r.options().demosaic, before.demosaic);
    assert!(
        r.options().gpu.is_some(),
        "setting the budget keeps the device"
    );

    r.set_gpu(None, GpuStages::NONE);
    assert!(r.gpu_info().is_none());
    assert_eq!(
        r.options().budget,
        lowered,
        "and the device keeps the budget"
    );
}

#[test]
fn core_attach_gpu_gives_the_renderer_the_slot_and_its_budgets() {
    let app = tempfile::tempdir().expect("app root");
    let core = Core::open(CoreOptions::new(data_root(), app.path().to_path_buf())).expect("core");
    let machine = core.renderer.options().budget;
    assert!(core.renderer.gpu_info().is_none());

    // A slot that uses the CPU leaves the renderer on the CPU and its budget as it was.
    core.attach_gpu(&open_gpu(GpuPolicy::Off, machine));
    assert!(core.renderer.gpu_info().is_none());
    assert_eq!(core.renderer.options().budget, machine);
    assert!(core.renderer.options().gpu_stages.is_empty());

    // `On` takes whatever adapter there is (a software one in CI).
    let slot = GpuOpener::new(GpuPolicy::On, machine).open();
    if !slot.is_using() {
        let reason = slot
            .missing_device_reason()
            .unwrap_or("no reason")
            .to_owned();
        assert!(
            !gpu_required_by_env(),
            "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {reason}"
        );
        eprintln!("* GPU TEST SKIPPED in gpu_policy::core_attach_gpu: {reason}");
        return;
    }
    core.attach_gpu(&slot);
    assert_eq!(
        core.renderer.gpu_info().map(|i| i.name),
        slot.adapter().map(|i| i.name.clone())
    );
    assert_eq!(core.renderer.options().gpu_stages, GpuStages::ALL);
    assert_eq!(core.renderer.options().budget, slot.budget());
    assert_eq!(core.renderer.ledger().budget(), slot.budget().cpu_bytes);

    // `Auto` keeps to what is signed off, and a software adapter is not used at all.
    let auto = open_gpu(GpuPolicy::Auto, machine);
    core.attach_gpu(&auto);
    if auto.is_using() {
        assert_eq!(core.renderer.options().gpu_stages, GpuStages::SIGNED_OFF);
    } else {
        assert!(core.renderer.gpu_info().is_none());
        assert!(auto.adapter().is_none_or(|a| a.is_software));
    }
}

// --- the default render does not change --------------------------------------------------------

#[test]
fn attaching_the_device_changes_no_byte_of_a_full_render() {
    let Some(gpu) = device("attaching_the_device_changes_no_byte") else {
        return;
    };
    // The export render is the CPU reference in this version, whatever device is attached; the
    // pixels of a full render with a device attached are those of one without (`AGENTS.md` hard
    // rule 8; `golden check` pins the same thing on the whole synthetic set).
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(300, 200, white, Layout::Patchwork, 1.2)
        .noise(3e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let request = FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    };
    let cancel = CancelToken::new();
    let render = |r: &Renderer| {
        r.render_full(&src, &request, &cancel)
            .expect("full render")
            .rgb
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<u32>>()
    };
    let cpu =
        Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let reference = render(&cpu);
    assert!(!reference.is_empty());
    for stages in [GpuStages::NONE, GpuStages::SIGNED_OFF, GpuStages::ALL] {
        let r =
            Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
        r.set_gpu(Some(Arc::clone(&gpu)), stages);
        assert_eq!(render(&r), reference, "stages: {stages}");
    }
}
