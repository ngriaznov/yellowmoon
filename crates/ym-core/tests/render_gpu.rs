//! The GPU in the render path (Mesa lavapipe in CI) against the CPU reference.
//!
//! * The colour pass of viewport frames and thumbnails: the frames agree within the parity
//!   tolerances of `ym-gpu`.
//! * G-03, the raw chain: with [`GpuStages::RAW_CHAIN`] the R6 demosaic of every full-resolution tile
//!   (the 100 % view, the export) runs on the device, one device job per tile with the render's own
//!   cancel token and priority. A 100 % frame and a full photo agree with the CPU render within the
//!   tolerances of the kernel; a forced demosaic method, a stage that is off and a policy that holds
//!   the device back keep the CPU; a device error (the faults of `ym-gpu`'s `fault-injection`
//!   feature, run with `--features fault-injection`) re-renders the whole photo on the CPU, bit for
//!   bit the CPU render, with the note; a strict policy (`Require`) fails the photo instead; a
//!   cancelled render and a navigation leave the pool at its baseline.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` (the `gpu-lavapipe` job) turns
//! that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource,
    RenderError, RenderOptions, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::PhotoId;
use ym_core::settings::DevelopSettings;
use ym_gpu::chain::StageRange;
use ym_gpu::cpu::demosaic::Method;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobError};

const MODEL: &str = "ILCE-7RM4";

fn gpu() -> Option<Arc<Gpu>> {
    // Poison mode: every pool buffer starts as NaN, so a kernel that reads what it did not write
    // fails the comparison with the CPU instead of passing by luck.
    match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in render_gpu: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

#[test]
fn gpu_and_cpu_frames_agree_within_the_parity_tolerance() {
    let Some(gpu) = gpu() else { return };
    let info = gpu.info().clone();
    eprintln!("GPU render tests on:\n{info}");
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(900, 600, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let cpu =
        Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread");
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    assert!(on_gpu.gpu_info().is_some());
    let cancel = CancelToken::new();
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    // Plain, then with grade controls, a curve and a straighten.
    let variants: Vec<(&str, PhotoSource)> = vec![
        ("plain", src.clone()),
        (
            "graded",
            edit(&src, |s| {
                s.contrast = 30.0;
                s.brightness = 10.0;
                s.saturation = 20.0;
                s.color_balance.highlights = [0.4, -0.3];
                s.curve.luma = vec![[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]];
                s.curve.blue = vec![[0.0, 0.05], [1.0, 0.95]];
                s.crop = Some(ym_core::settings::Crop {
                    rect: ym_core::settings::NormRect {
                        x: 0.05,
                        y: 0.05,
                        width: 0.9,
                        height: 0.9,
                    },
                    angle_deg: 4.0,
                    aspect: ym_core::settings::CropAspect::Free,
                });
            }),
        ),
    ];
    for (name, v) in variants {
        let a = cpu
            .viewport_frame(&v, &st, &FrameOptions::default(), &cancel)
            .expect("cpu frame");
        let b = on_gpu
            .viewport_frame(&v, &st, &FrameOptions::default(), &cancel)
            .expect("gpu frame");
        assert_eq!(a.report.device, ComputeDevice::Cpu);
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "{name}: {:?}",
            b.report.notes
        );
        assert_eq!((a.width, a.height), (b.width, b.height));
        let (mut worst, mut sum) = (0u8, 0u64);
        for (x, y) in a.rgba.iter().zip(&b.rgba) {
            let d = x.abs_diff(*y);
            worst = worst.max(d);
            sum += u64::from(d);
        }
        let mean = sum as f64 / a.rgba.len() as f64;
        eprintln!("{name}: worst difference {worst} codes, mean {mean:.4}");
        // One code of 8 bits at most (the GPU's f16 storage and its own dither lookup agree with
        // the CPU's to better than the 1e-3 gate of ym-gpu), almost always zero.
        assert!(worst <= 2, "{name}: worst difference {worst} codes");
        assert!(mean < 0.25, "{name}: mean difference {mean}");
    }
    // The Before view runs on the CPU whatever is attached.
    let mut before = st.clone();
    before.before = true;
    let b = on_gpu
        .viewport_frame(&src, &before, &FrameOptions::default(), &cancel)
        .expect("before");
    assert_eq!(b.report.device, ComputeDevice::Cpu);
    // Thumbnails go through the same pass.
    let a = cpu.thumbnail_png(&src, &cancel).expect("thumbnail");
    let b = on_gpu
        .thumbnail_png(&src, &cancel)
        .expect("thumbnail on gpu");
    assert_eq!(&a[..8], &b[..8]);
}

// ---------------------------------------------------------------------------------------------
// The raw chain on the device (G-03)
// ---------------------------------------------------------------------------------------------

/// A scene of 1.2 Mpixel: with 512 px export tiles a full render is six device jobs. A neutral
/// patchwork with noise, so that it has edges, flat patches and no clipping (tie-poor content is not
/// the point here: the comparison with the CPU is a statistic, and the picture a whole).
fn scene() -> PhotoSource {
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(1300, 900, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
    Developed::new(&scene, MODEL).source()
}

fn renderer_with(
    gpu: Option<Arc<Gpu>>,
    stages: GpuStages,
    strict: bool,
    method: Method,
) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: stages,
            gpu_strict: strict,
            demosaic: method,
            full_tile_px: 512,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn on_device(gpu: &Arc<Gpu>) -> Renderer {
    renderer_with(
        Some(Arc::clone(gpu)),
        GpuStages::ALL,
        false,
        Method::AdaptiveRatio,
    )
}

fn on_cpu() -> Renderer {
    renderer_with(None, GpuStages::NONE, false, Method::AdaptiveRatio)
}

fn render_full(r: &Renderer, src: &PhotoSource) -> Result<FullImage, RenderError> {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
}

/// How two display-linear pictures differ: the largest difference, the share of samples over 1e-3
/// (a tie flip of the demosaic and what the later stages make of it) and the mean.
fn difference(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut worst, mut over, mut sum) = (0.0f64, 0usize, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite(), "a sample is not finite");
        let d = f64::from((x - y).abs());
        worst = worst.max(d);
        over += usize::from(d > 1e-3);
        sum += d;
    }
    (worst, over as f64 / a.len() as f64, sum / a.len() as f64)
}

/// The bytes in use in the device's pool (what a finished or cancelled render must hand back).
fn pool_in_use(gpu: &Gpu) -> u64 {
    gpu.run(
        ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
        |ctx, _| Ok(ctx.pool().stats().in_use_bytes),
    )
    .expect("pool stats")
}

fn device_of(report: &ym_core::pipeline::RenderReport) -> Vec<(String, ComputeDevice)> {
    report
        .ranges
        .iter()
        .map(|r| (r.range.label(), r.device))
        .collect()
}

#[test]
fn a_full_render_and_a_100_percent_frame_on_the_device_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let (cpu, dev) = (on_cpu(), on_device(&gpu));
    let a = render_full(&cpu, &src).expect("cpu render");
    let before = pool_in_use(&gpu);
    let b = render_full(&dev, &src).expect("device render");
    assert_eq!(
        pool_in_use(&gpu),
        before,
        "the render hands its buffers back"
    );
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert!(a.report.ranges.is_empty(), "no device, no range");
    assert_eq!(
        device_of(&b.report),
        vec![("R6".to_owned(), ComputeDevice::Gpu)],
        "{:?}",
        b.report.notes
    );
    assert_eq!(b.report.runs.tiles, a.report.runs.tiles);
    assert!(b.report.runs.tiles >= 6, "{}", b.report.runs.tiles);
    let (worst, over, mean) = difference(&a.rgb, &b.rgb);
    eprintln!(
        "full render, device against CPU: worst {worst:e}, share over 1e-3 {over:e}, mean {mean:e}"
    );
    // Twice the first measured values (worst 4.8e-4, share over 1e-3 zero, mean 2.6e-8).
    assert!(
        worst < 1e-3 && over <= 1e-5 && mean < 1e-6,
        "{worst} {over} {mean}"
    );

    // The 100 % view: the tiles of 512 px, cached, then sampled and coloured. Two baselines: the CPU
    // renderer, and a renderer with the device for the colour pass only (the raw chain on the CPU),
    // which isolates what the raw chain on the device changes from what the GPU colour pass and the
    // CPU output sharpening already do (`docs/05`: the stage's limiter flips on noiseless content).
    let colour_only = renderer_with(
        Some(Arc::clone(&gpu)),
        GpuStages::COLOR,
        false,
        Method::AdaptiveRatio,
    );
    let image = (a.width, a.height);
    let mut st = ViewportState::new(ViewportRect::cell(700.0, 500.0, 1.0));
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let cancel = CancelToken::new();
    let frame = |r: &Renderer| {
        r.viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .expect("frame")
    };
    let (fa, fb, fc) = (frame(&cpu), frame(&dev), frame(&colour_only));
    assert_eq!(fa.report.scale, 1, "full-resolution tiles");
    assert_eq!(
        device_of(&fb.report),
        vec![("R6".to_owned(), ComputeDevice::Gpu)]
    );
    assert!(fc.report.ranges.is_empty());
    assert_eq!(fb.report.runs.tiles, fa.report.runs.tiles);
    let codes = |x: &ym_core::pipeline::RenderedFrame, y: &ym_core::pipeline::RenderedFrame| {
        let (mut worst, mut over, mut sum) = (0u8, 0usize, 0u64);
        for (p, q) in x.rgba.iter().zip(&y.rgba) {
            let d = p.abs_diff(*q);
            worst = worst.max(d);
            over += usize::from(d > 1);
            sum += u64::from(d);
        }
        (
            worst,
            over as f64 / x.rgba.len() as f64,
            sum as f64 / x.rgba.len() as f64,
        )
    };
    let (w1, o1, m1) = codes(&fa, &fb);
    let (w2, o2, m2) = codes(&fc, &fb);
    eprintln!("100 % frame, device against CPU: worst {w1} codes, over 1 code {o1:e}, mean {m1:e}");
    eprintln!(
        "100 % frame, device against the CPU raw chain with the same colour pass: worst {w2} codes, over 1 code {o2:e}, mean {m2:e}"
    );
    // Twice the first measured values (against the CPU: no sample over 1 code, mean 5.1e-5 codes;
    // against the CPU raw chain with the same colour pass: 2.1e-6 over 1 code, worst 6 codes in
    // three samples where the output sharpening's limiter flips on a flipped demosaic, mean 3.1e-5).
    assert!(o1 <= 1e-5 && m1 < 1e-4, "{o1} {m1}");
    assert!(o2 <= 1e-5 && m2 < 1e-4 && w2 <= 12, "{o2} {m2} {w2}");
}

#[test]
fn the_device_runs_the_raw_chain_only_when_the_stage_and_the_method_allow_it() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let cpu = render_full(&on_cpu(), &src).expect("cpu render");
    // The stage is off (what `Auto` runs today, the colour pass): the CPU runs R6, no range.
    let off = renderer_with(
        Some(Arc::clone(&gpu)),
        GpuStages::SIGNED_OFF,
        false,
        Method::AdaptiveRatio,
    );
    let jobs = gpu.counters().submissions;
    let a = render_full(&off, &src).expect("render");
    assert_eq!(a.rgb, cpu.rgb, "the CPU path is the CPU path");
    assert!(a.report.ranges.is_empty(), "{:?}", a.report.ranges);
    assert_eq!(gpu.counters().submissions, jobs, "the device was not asked");
    // A forced method that the device does not run: the CPU, with a note, and the same bits as the
    // CPU renderer with that method.
    let forced = renderer_with(Some(Arc::clone(&gpu)), GpuStages::ALL, false, Method::Menon);
    let menon_cpu = renderer_with(None, GpuStages::NONE, false, Method::Menon);
    let a = render_full(&forced, &src).expect("render");
    let b = render_full(&menon_cpu, &src).expect("render");
    assert_eq!(a.rgb, b.rgb);
    assert_eq!(
        device_of(&a.report),
        vec![("R6".to_owned(), ComputeDevice::Cpu)]
    );
    assert!(
        a.report
            .notes
            .iter()
            .any(|n| n.contains("adaptive ratio demosaic only")),
        "{:?}",
        a.report.notes
    );
    assert_eq!(gpu.counters().submissions, jobs);
    // The device is attached later: renders from then on use it, and a render in progress would
    // have kept its snapshot (the options are read once per render).
    let late = renderer_with(None, GpuStages::NONE, false, Method::AdaptiveRatio);
    late.set_gpu(Some(Arc::clone(&gpu)), GpuStages::ALL);
    let a = render_full(&late, &src).expect("render");
    assert_eq!(
        device_of(&a.report),
        vec![("R6".to_owned(), ComputeDevice::Gpu)]
    );
    late.set_gpu(None, GpuStages::NONE);
    let b = render_full(&late, &src).expect("render");
    assert!(b.report.ranges.is_empty());
    assert_eq!(b.rgb, cpu.rgb);
    assert_eq!(StageRange::DEMOSAIC.label(), "R6");
}

#[test]
fn two_renders_and_renders_on_other_thread_counts_are_byte_identical() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let dev = on_device(&gpu);
    let first = render_full(&dev, &src).expect("render");
    let second = render_full(&dev, &src).expect("render");
    assert_eq!(first.rgb, second.rgb, "twice on one renderer");
    // A fresh renderer and a device that did other work in between (A, B, A: pool history).
    let other = Developed::new(
        &Scene::new(640, 480, light("sony-ilce-7rm4", 3200.0), Layout::Ramp, 0.4).noise(2e-4, 2e-6),
        MODEL,
    )
    .source();
    let _ = render_full(&on_device(&gpu), &other).expect("render another photo");
    let third = render_full(&on_device(&gpu), &src).expect("render");
    assert_eq!(first.rgb, third.rgb, "after another photo");
    // The CPU stages use rayon: one thread and four give the same picture.
    for threads in [1usize, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        let img = pool.install(|| render_full(&on_device(&gpu), &src).expect("render"));
        assert_eq!(first.rgb, img.rgb, "{threads} thread(s)");
    }
}

#[test]
fn cancelling_an_export_stops_within_a_tile_and_hands_the_buffers_back() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let dev = on_device(&gpu);
    let before = pool_in_use(&gpu);
    let jobs = gpu.counters().submissions;
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handle = {
        let seen = Arc::clone(&seen);
        dev.spawn_full_reporting(
            src.clone(),
            FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            Arc::new(move |_| {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }),
        )
    };
    // Cancel as soon as the first tile is reported.
    let started = std::time::Instant::now();
    while seen.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        assert!(
            started.elapsed().as_secs() < 120,
            "the export never started"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    handle.cancel();
    let done_at_cancel = seen.load(std::sync::atomic::Ordering::SeqCst);
    let result = handle.wait().expect("the job resolved");
    assert!(matches!(result, Err(RenderError::Cancelled)), "{result:?}");
    let tiles_done = seen.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        tiles_done <= done_at_cancel + 1,
        "{tiles_done} tiles reported after the cancel at {done_at_cancel}"
    );
    assert!(tiles_done < 6, "the export stopped before the last tile");
    assert_eq!(
        pool_in_use(&gpu),
        before,
        "a cancelled export leaves nothing resident"
    );
    let ran = gpu.counters().submissions - jobs;
    assert!(
        ran <= 2 * (tiles_done as u64 + 1) + 4,
        "{ran} submissions for {tiles_done} tiles"
    );
}

#[test]
fn navigating_away_cancels_the_queued_device_jobs_of_the_photo_left() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    // Two photos of two sessions have the same id; the renderer tells photos apart by it.
    let mut src = src;
    src.id = PhotoId(1);
    let mut other = Developed::new(
        &Scene::new(640, 480, light("sony-ilce-7rm4", 3200.0), Layout::Ramp, 0.4).noise(2e-4, 2e-6),
        MODEL,
    )
    .source();
    other.id = PhotoId(2);
    let dev = on_device(&gpu);
    dev.navigate(src.id);
    // Hold the device thread, then ask for the 100 % tiles of the photo: its first device job
    // queues behind the gate.
    let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let gate = gpu.submit(
        ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
        move |_, _| {
            ready_tx.send(()).expect("ready");
            gate_rx.recv().expect("gate");
        },
    );
    ready_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the device thread took the gate job");
    let mut st = ViewportState::new(ViewportRect::cell(500.0, 400.0, 1.0));
    st.set_mode(ViewportMode::Percent { value: 100.0 }, (1300, 900));
    let tiles = dev.spawn_tiles(src.clone(), st);
    // Wait until the render's device job is queued behind the gate.
    let started = std::time::Instant::now();
    while gpu.scheduler().pending() == 0 {
        assert!(started.elapsed().as_secs() < 60, "no device job was queued");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    dev.navigate(other.id);
    // The render resolves now, with the gate still closed: the queued device job was cancelled, not
    // left to be skipped when it is popped.
    let result = tiles
        .wait_timeout(std::time::Duration::from_secs(30))
        .expect("the render resolved while the device thread was still held");
    assert!(
        matches!(result, Err(JobError::Cancelled))
            || matches!(result, Ok(Err(RenderError::Cancelled))),
        "{result:?}"
    );
    assert_eq!(
        gpu.scheduler().pending(),
        0,
        "nothing of the photo is left queued"
    );
    gate_tx.send(()).expect("open the gate");
    gate.wait().expect("gate job");
}

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
    use ym_gpu::device::GpuFault;

    const FAULTS: [(&str, GpuFault); 5] = [
        ("out of budget", GpuFault::OutOfBudget),
        ("shader", GpuFault::Shader),
        ("lost device", GpuFault::Lost),
        ("failing tile", GpuFault::FailTile(2)),
        ("failing first tile", GpuFault::FailTile(1)),
    ];

    #[test]
    fn a_device_error_renders_the_whole_photo_on_the_cpu_with_the_note() {
        let src = scene();
        let cpu = render_full(&on_cpu(), &src).expect("cpu render");
        for (name, fault) in FAULTS {
            let Some(gpu) = gpu() else { return };
            let before = pool_in_use(&gpu);
            let dev = on_device(&gpu);
            gpu.set_fault(Some(fault));
            let img = render_full(&dev, &src).unwrap_or_else(|e| panic!("{name}: {e}"));
            gpu.set_fault(None);
            assert_eq!(img.rgb, cpu.rgb, "{name}: the CPU render, bit for bit");
            assert_eq!(
                device_of(&img.report),
                vec![("R6".to_owned(), ComputeDevice::Cpu)],
                "{name}"
            );
            assert!(
                img.report.notes.iter().any(|n| n
                    .starts_with("The graphics card could not render this photo, so the CPU did")),
                "{name}: {:?}",
                img.report.notes
            );
            if !matches!(fault, GpuFault::Lost) {
                assert_eq!(
                    pool_in_use(&gpu),
                    before,
                    "{name}: the ledger is back at its baseline"
                );
            }
            // The photo renders on the CPU from then on, quietly: no second attempt, no new note.
            let jobs = gpu.counters().submissions;
            let again = render_full(&dev, &src).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(again.rgb, cpu.rgb, "{name}: second render");
            assert!(
                !again
                    .report
                    .notes
                    .iter()
                    .any(|n| n.contains("could not render")),
                "{name}: {:?}",
                again.report.notes
            );
            assert_eq!(
                gpu.counters().submissions,
                jobs,
                "{name}: the device was not asked again"
            );
        }
    }

    #[test]
    fn a_shader_that_does_not_compile_turns_the_raw_chain_off_for_the_process() {
        let Some(gpu) = gpu() else { return };
        let (a, b) = (scene(), scene());
        let dev = on_device(&gpu);
        gpu.set_fault(Some(GpuFault::Shader));
        let first = render_full(&dev, &a).expect("render");
        gpu.set_fault(None);
        assert!(
            first
                .report
                .notes
                .iter()
                .any(|n| n.contains("could not render"))
        );
        // Another photo of the same renderer: no attempt on the device any more.
        let jobs = gpu.counters().submissions;
        let second = render_full(&dev, &b).expect("render");
        assert_eq!(gpu.counters().submissions, jobs);
        assert_eq!(
            device_of(&second.report),
            vec![("R6".to_owned(), ComputeDevice::Cpu)]
        );
    }

    #[test]
    fn a_strict_policy_fails_the_photo_instead_of_falling_back() {
        let src = scene();
        for (name, fault) in FAULTS {
            let Some(gpu) = gpu() else { return };
            let strict = renderer_with(
                Some(Arc::clone(&gpu)),
                GpuStages::ALL,
                true,
                Method::AdaptiveRatio,
            );
            gpu.set_fault(Some(fault));
            let result = render_full(&strict, &src);
            gpu.set_fault(None);
            assert!(
                matches!(result, Err(RenderError::Gpu(_))),
                "{name}: {:?}",
                result.map(|i| i.report.notes)
            );
        }
    }

    /// The ways the device route of a photo can go away between two tiles of an export that has
    /// already put its first tile on the device.
    #[derive(Clone, Copy, Debug)]
    enum RouteLost {
        /// A viewport tile of the same photo fails on the device, which ends the photo's route.
        ViewportTileFails,
        /// The device is lost between two tiles.
        DeviceLost,
    }

    #[test]
    fn an_export_whose_route_goes_away_between_tiles_starts_again_on_the_cpu() {
        // Never a mix of devices in one file (`docs/gpu-plan.md` 4.9): the route is decided by flags
        // that other requests can set while an export is under way, so the export itself must notice
        // that its first tiles ran on the device and the rest cannot, and render the whole photo on
        // the CPU. The event happens inside the export's own progress callback, that is between two
        // tiles, so the test is not a race.
        let src = scene();
        let cpu = render_full(&on_cpu(), &src).expect("cpu render");
        for how in [RouteLost::ViewportTileFails, RouteLost::DeviceLost] {
            let Some(gpu) = gpu() else { return };
            let dev = on_device(&gpu);
            let mut st = ViewportState::new(ViewportRect::cell(700.0, 500.0, 1.0));
            st.set_mode(ViewportMode::Percent { value: 100.0 }, (1300, 900));
            let first = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let progress = {
                let (gpu, dev, src) = (Arc::clone(&gpu), dev.clone(), src.clone());
                Arc::new(move |_: f32| {
                    if !first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        return;
                    }
                    match how {
                        RouteLost::ViewportTileFails => {
                            gpu.set_fault(Some(GpuFault::OutOfBudget));
                            let frame = dev.viewport_frame(
                                &src,
                                &st,
                                &FrameOptions::default(),
                                &CancelToken::new(),
                            );
                            gpu.set_fault(None);
                            assert!(frame.is_ok(), "the frame falls back to the CPU: {frame:?}");
                        }
                        RouteLost::DeviceLost => {
                            gpu.set_fault(Some(GpuFault::Lost));
                            let _ = gpu.run(
                                ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
                                |_, _| Ok(()),
                            );
                            assert!(gpu.is_lost(), "the fault lost the device");
                        }
                    }
                })
            };
            let handle = dev.spawn_full_reporting(
                src.clone(),
                FullRequest {
                    target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                },
                progress,
            );
            let img = handle
                .wait()
                .expect("the job resolved")
                .unwrap_or_else(|e| panic!("{how:?}: {e}"));
            assert!(
                img.rgb == cpu.rgb,
                "{how:?}: the CPU render, bit for bit (a mix of the two devices differs in the tiles the device made)"
            );
            assert_eq!(
                device_of(&img.report),
                vec![("R6".to_owned(), ComputeDevice::Cpu)],
                "{how:?}"
            );
            assert!(
                img.report.notes.iter().any(|n| n
                    .starts_with("The graphics card could not render this photo, so the CPU did")),
                "{how:?}: {:?}",
                img.report.notes
            );
        }
    }

    #[test]
    fn a_strict_policy_does_not_fall_back_quietly_once_the_raw_chain_is_off_for_the_process() {
        // A shader that does not compile fails the photo under `Require` and switches the raw chain off
        // for the process. The next photo must fail too: rendering it on the CPU without a word would
        // be the silent fall-back that `Require` exists to forbid.
        let Some(gpu) = gpu() else { return };
        let strict = renderer_with(
            Some(Arc::clone(&gpu)),
            GpuStages::ALL,
            true,
            Method::AdaptiveRatio,
        );
        gpu.set_fault(Some(GpuFault::Shader));
        let first = render_full(&strict, &scene());
        gpu.set_fault(None);
        assert!(
            matches!(first, Err(RenderError::Gpu(_))),
            "{:?}",
            first.map(|i| i.report.notes)
        );
        let jobs = gpu.counters().submissions;
        let second = render_full(&strict, &scene());
        assert!(
            matches!(second, Err(RenderError::Gpu(_))),
            "{:?}",
            second.map(|i| (i.report.ranges, i.report.notes))
        );
        assert_eq!(gpu.counters().submissions, jobs, "the device was not asked");
    }

    #[test]
    fn a_cancel_fault_cancels_the_render_and_leaves_nothing_resident() {
        let Some(gpu) = gpu() else { return };
        let src = scene();
        let dev = on_device(&gpu);
        let before = pool_in_use(&gpu);
        gpu.set_fault(Some(GpuFault::CancelAtJob(2)));
        let result = render_full(&dev, &src);
        gpu.set_fault(None);
        assert!(
            matches!(result, Err(RenderError::Cancelled)),
            "{:?}",
            result.map(|i| i.report)
        );
        assert_eq!(pool_in_use(&gpu), before);
        // A fresh render with a fresh token works, on the device.
        let again = render_full(&dev, &src).expect("render");
        assert_eq!(
            device_of(&again.report),
            vec![("R6".to_owned(), ComputeDevice::Gpu)]
        );
    }

    #[test]
    fn a_viewport_tile_that_fails_is_made_on_the_cpu_with_a_note() {
        let Some(gpu) = gpu() else { return };
        let src = scene();
        let dev = on_device(&gpu);
        // The baseline: the CPU raw chain with the same GPU colour pass.
        let colour_only = renderer_with(
            Some(Arc::clone(&gpu)),
            GpuStages::COLOR,
            false,
            Method::AdaptiveRatio,
        );
        let mut st = ViewportState::new(ViewportRect::cell(700.0, 500.0, 1.0));
        st.set_mode(ViewportMode::Percent { value: 100.0 }, (1300, 900));
        let cancel = CancelToken::new();
        let baseline = colour_only
            .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .expect("baseline frame");
        // The first tile fails: every tile of the frame is made on the CPU, so the frame is the
        // baseline byte for byte.
        gpu.set_fault(Some(GpuFault::FailTile(1)));
        let f = dev
            .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
            .expect("the frame is made");
        gpu.set_fault(None);
        assert!(
            f.report
                .notes
                .iter()
                .any(|n| n
                    .starts_with("The graphics card could not render this frame, so the CPU did")),
            "{:?}",
            f.report.notes
        );
        assert_eq!(f.rgba, baseline.rgba);
        assert_eq!(
            device_of(&f.report),
            vec![("R6".to_owned(), ComputeDevice::Cpu)]
        );
    }
}
