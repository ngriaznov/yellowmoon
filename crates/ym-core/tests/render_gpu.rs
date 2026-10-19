//! The GPU in the render path (Mesa lavapipe in CI) against the CPU reference.
//!
//! * The colour pass of viewport frames and thumbnails: the frames agree within the parity
//!   tolerances of `ym-gpu`.
//! * The raw chain: with [`GpuStages::RAW_CHAIN`] the R6 demosaic of every full-resolution tile
//!   (the 100 % view, the export) runs on the device, one device job per tile with the render's own
//!   cancel token and priority. A 100 % frame and a full photo agree with the CPU render within the
//!   tolerances of the kernel; a forced demosaic method, a stage that is off and a policy that holds
//!   the device back keep the CPU; a device error (the faults of `ym-gpu`'s `fault-injection`
//!   feature, run with `--features fault-injection`) re-renders the whole photo on the CPU, bit for
//!   bit the CPU render, with the note; a strict policy (`Require`) fails the photo instead; a
//!   cancelled render and a navigation leave the pool at its baseline.
//!
//! * The raw front: `GpuStages::ALL` includes [`GpuStages::RAW_FRONT`], so the same tiles run
//!   R1 to R9 (`R1..R9`) from the photo's raw words, uploaded once per photo; `gpu_reduced.rs` has the
//!   tests of the front itself and of the fit-view group. The tests below that name a range say
//!   `R1..R9` for that reason; the R6 to R8 route behind the CPU's front is what `RAW_CHAIN` alone
//!   runs.
//!
//! * The view stages: with the members of [`GpuStages`] for the sampler, the colour chain,
//!   output sharpening and C13, a viewport frame, a thumbnail and a look tile are one device job each
//!   (sampler, C1 to C12, stage, C13, one readback), a move of the Output slider is one small job on
//!   the resident display-linear frame, and a view too big for one storage binding is cut into bands
//!   with the same pixels. See `mod view` below.
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
                s.contrast = ym_core::settings::PercentSetting::manual_value(30.0);
                s.brightness = 10.0;
                s.saturation = ym_core::settings::PercentSetting::manual_value(20.0);
                s.color_balance.highlights =
                    ym_core::settings::BalanceSetting::manual_point([0.4, -0.3]);
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
    // The Before view is the colour kernel's RGBA8 stop, on the device.
    let mut before = st.clone();
    before.before = true;
    let a = cpu
        .viewport_frame(&src, &before, &FrameOptions::default(), &cancel)
        .expect("cpu before");
    let b = on_gpu
        .viewport_frame(&src, &before, &FrameOptions::default(), &cancel)
        .expect("before");
    assert_eq!(a.report.device, ComputeDevice::Cpu);
    assert_eq!(b.report.device, ComputeDevice::Gpu, "{:?}", b.report.notes);
    let worst = a
        .rgba
        .iter()
        .zip(&b.rgba)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    assert!(worst <= 2, "the Before view differs by {worst} codes");
    // Thumbnails go through the same pass.
    let a = cpu.thumbnail_png(&src, &cancel).expect("thumbnail");
    let b = on_gpu
        .thumbnail_png(&src, &cancel)
        .expect("thumbnail on gpu");
    assert_eq!(&a[..8], &b[..8]);
}

// ---------------------------------------------------------------------------------------------
// The raw chain on the device
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
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)],
        "{:?}",
        b.report.notes
    );
    assert_eq!(b.report.runs.tiles, a.report.runs.tiles);
    assert!(b.report.runs.tiles >= 6, "{}", b.report.runs.tiles);
    let (worst, over, mean) = difference(&a.rgb, &b.rgb);
    eprintln!(
        "full render, device against CPU: worst {worst:e}, share over 1e-3 {over:e}, mean {mean:e}"
    );
    // Twice the first measured values (worst 4.8e-4, share over 1e-3 zero, mean 2.6e-8), with the worst
    // allowance raised for a gain on the highlights: the one tie-flipped demosaic sample of the
    // patchwork (the single worst pixel and its two neighbours, at a noise texture of 4e-3 between
    // neighbours) is multiplied by the local gain of what acts on its tones: the white lift (C12 step
    // 1b; this scene is high-key, a lift of 1.04 stops) and, since pipeline 0.4.0, the highlight range
    // of C7 alongside it, whose slope above the knee is more than 1. Measured on lavapipe: 1.41e-3 with
    // the lift (pipeline 0.3.0, and 0.4.0 with the range: share over 1e-3 8.5e-7, mean 6.3e-8), 8.4e-4
    // without it (8.38e-4 with the range alone). The allowance stays at 2e-3. The stages' own parity is
    // gated in `ym-gpu` (`gpu_color_white_lift`, `gpu_color_highlight_range`).
    assert!(
        worst < 2e-3 && over <= 1e-5 && mean < 1e-6,
        "{worst} {over} {mean}"
    );

    // The 100 % view: the tiles of 512 px, cached, then sampled and coloured. Two baselines: the CPU
    // renderer, and a renderer with the device for the view stages only (the raw chain on the CPU),
    // which isolates what the raw chain on the device changes from what the sampler, the colour pass
    // and output sharpening on the device already do (`docs/05`: the stage's limiter flips on
    // noiseless content).
    let colour_only = renderer_with(
        Some(Arc::clone(&gpu)),
        view::VIEW,
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
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)]
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

/// The two renders of a DNG carry no R8 capture sharpening on the device either (`docs/09`, "What a DNG
/// carries"): the device chain is made from the same parameters as the CPU's, and a photo whose R8 is
/// switched off by its setting is the same render on the device as the DNG's preview, bit for bit.
#[test]
fn the_renders_of_a_dng_have_no_capture_sharpening_on_the_device_and_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let off = edit(&src, |s| {
        s.sharpen = ym_core::settings::AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 0.0,
            radius_px: 1.0,
        });
    });
    // Every stage but the export tail: with `GpuStages::EXPORT_TAIL` the colour stage of the DNG's
    // two renders runs on the device too (`tests/gpu_export.rs` compares it with the CPU's),
    // while the display render of a host frame colours on the CPU, so the two would differ by the
    // colour stage's f32 rounding and not be equal bit for bit. This test is about R8.
    let no_tail = GpuStages::RAW_CHAIN
        .union(GpuStages::RAW_FRONT)
        .union(GpuStages::SAMPLER)
        .union(GpuStages::COLOR)
        .union(GpuStages::OUTPUT_SHARPEN)
        .union(GpuStages::ENCODE);
    let (cpu, dev) = (
        on_cpu(),
        renderer_with(
            Some(Arc::clone(&gpu)),
            no_tail,
            false,
            Method::AdaptiveRatio,
        ),
    );
    let full = |r: &Renderer, s: &PhotoSource, target: FullTarget| {
        r.render_full(s, &FullRequest { target }, &CancelToken::new())
            .expect("a render")
    };
    let display = FullTarget::DisplayLinear(OutputTarget::Srgb);

    // The control: on the device R8 is on for a picture of the other formats, and it moves this scene.
    let dev_display_on = full(&dev, &src, display);
    let dev_display_off = full(&dev, &off, display);
    assert_eq!(
        device_of(&dev_display_on.report),
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)]
    );
    let (effect, _, _) = difference(&dev_display_on.rgb, &dev_display_off.rgb);
    assert!(effect > 1e-3, "R8 moves the picture by {effect}");

    // The DNG's preview on the device ran the device range and is the render with R8 off, bit for bit.
    let dev_preview = full(&dev, &src, FullTarget::DngPreview);
    assert_eq!(
        device_of(&dev_preview.report),
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)],
        "{:?}",
        dev_preview.report.notes
    );
    assert_eq!(dev_preview.rgb, dev_display_off.rgb);
    // The raw pixels do not depend on the setting.
    let dev_raw = full(&dev, &src, FullTarget::DngLinearAwg4);
    assert_eq!(dev_raw.rgb, full(&dev, &off, FullTarget::DngLinearAwg4).rgb);

    // And the device agrees with the CPU within the kernels' tolerances (`docs/05`), for both.
    let cpu_preview = full(&cpu, &src, FullTarget::DngPreview);
    let (worst, over, mean) = difference(&cpu_preview.rgb, &dev_preview.rgb);
    eprintln!(
        "DNG preview, device against CPU: worst {worst:e}, share over 1e-3 {over:e}, mean {mean:e}"
    );
    // Twice the first measured values (worst 8.3e-4, no sample over 1e-3, mean 4.5e-8): a tie flip of
    // the demosaic and what the later stages make of it.
    assert!(
        worst < 2e-3 && over <= 2e-5 && mean < 1e-6,
        "{worst} {over} {mean}"
    );
    let cpu_raw = full(&cpu, &src, FullTarget::DngLinearAwg4);
    let (worst, over, mean) = difference(&cpu_raw.rgb, &dev_raw.rgb);
    eprintln!(
        "DNG pixels, device against CPU: worst {worst:e}, share over 1e-3 {over:e}, mean {mean:e}"
    );
    // The pixels are scene-linear (a wider range than the display's): twice the first measured values
    // (worst 1.7e-3, share over 1e-3 8.5e-7, mean 7.7e-8).
    assert!(
        worst < 4e-3 && over <= 2e-5 && mean < 1e-6,
        "{worst} {over} {mean}"
    );
}

#[test]
fn the_device_runs_the_raw_chain_only_when_the_stage_and_the_method_allow_it() {
    let Some(gpu) = gpu() else { return };
    let src = scene();
    let cpu = render_full(&on_cpu(), &src).expect("cpu render");
    // The stage is off (the colour pass alone, what `Auto` ran before 2026-10-14): the CPU runs
    // R6, no range.
    let off = renderer_with(
        Some(Arc::clone(&gpu)),
        GpuStages::COLOR,
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
        vec![("R1..R9".to_owned(), ComputeDevice::Cpu)]
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
        vec![("R1..R9".to_owned(), ComputeDevice::Gpu)]
    );
    late.set_gpu(None, GpuStages::NONE);
    let b = render_full(&late, &src).expect("render");
    assert!(b.report.ranges.is_empty());
    assert_eq!(b.rgb, cpu.rgb);
    assert_eq!(StageRange::DEMOSAIC.label(), "R6");
}

/// Small and odd photos through the whole render: a frame below one export tile, odd sides (a device
/// image pads its rows to an even stride), a strip, and the smallest the synthetic writer makes. The
/// device must neither panic nor refuse them, and must agree with the CPU render (the content is
/// tie-poor, so a plumbing error moves the mean, a tie flip only a few samples).
#[test]
fn small_and_odd_photos_render_on_the_device_like_on_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4500.0);
    for (w, h) in [
        (66u32, 50u32),
        (131, 97),
        (34, 18),
        (512, 9),
        (7, 301),
        (250, 250),
    ] {
        let scene = Scene::new(w, h, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
        let src = Developed::new(&scene, MODEL).source();
        let a = render_full(&on_cpu(), &src).unwrap_or_else(|e| panic!("{w}x{h} on the CPU: {e}"));
        let b = render_full(&on_device(&gpu), &src)
            .unwrap_or_else(|e| panic!("{w}x{h} on the device: {e}"));
        assert_eq!((a.width, a.height), (b.width, b.height), "{w}x{h}");
        assert_eq!(
            device_of(&b.report),
            vec![("R1..R9".to_owned(), ComputeDevice::Gpu)],
            "{w}x{h}: {:?}",
            b.report.notes
        );
        let (worst, _, mean) = difference(&a.rgb, &b.rgb);
        eprintln!("{w}x{h}: worst {worst:e}, mean {mean:e}");
        // The bounds of the full render above (measured here: worst 1.1e-7 to 6.9e-7, mean 1.4e-8
        // to 2.2e-8): a plumbing error moves the mean by orders of magnitude.
        assert!(
            mean < 1e-6 && worst < 1e-3,
            "{w}x{h}: worst {worst}, mean {mean}"
        );
    }
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
    // The tile job that was in flight when the export was cancelled is abandoned, not waited for
    // (`GpuRoute::run_job`): it holds the tile's buffers, and with the raw front the photo's resident
    // words, until the device thread pops it (a queued job whose token is set is skipped and dropped)
    // or finishes it. The ledger probe runs at Viewport priority, ahead of an Export job that has not
    // been popped yet, so under load it can see those buffers once. A leak never goes away: wait for
    // the baseline for up to ten seconds, then require it.
    let settle = std::time::Instant::now();
    while pool_in_use(&gpu) != before && settle.elapsed().as_secs() < 10 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
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

/// A photo a strict policy refused to render on the CPU is the export error kind `gpu` (the JSON of
/// `ym render` and `docs/10`); every other render error stays `render`.
#[test]
fn a_device_error_of_a_strict_render_is_the_export_error_kind_gpu() {
    use ym_core::export::ExportError;
    assert_eq!(
        ExportError::Render(RenderError::Gpu("lost".into())).kind(),
        "gpu"
    );
    assert_eq!(
        ExportError::Render(RenderError::Image("bad".into())).kind(),
        "render"
    );
    assert_eq!(ExportError::Render(RenderError::Memory).kind(), "render");
}

// ---------------------------------------------------------------------------------------------
// The view stages on the device
// ---------------------------------------------------------------------------------------------

mod view {
    use super::*;
    use ym_color::consts::linalg;
    use ym_color::deltae::{d65_white_xyz, delta_e_2000, percentile, xyz_to_lab};
    use ym_color::encode::srgb_decode;
    use ym_core::pipeline::{RenderedFrame, StageRuns};
    use ym_core::settings::{Crop, CropAspect, NormRect, Orientation};

    /// What a frame, a thumbnail and a look tile run on the device: the sampler, the colour chain,
    /// output sharpening and C13. The raw chain stays on the CPU, so a comparison with the CPU
    /// renderer isolates the view stages.
    pub const VIEW: GpuStages = GpuStages::SAMPLER
        .union(GpuStages::COLOR)
        .union(GpuStages::OUTPUT_SHARPEN)
        .union(GpuStages::ENCODE);

    /// A note that reaches the CI log even when the test passes: written to the process's own stderr
    /// (the harness captures `eprintln!`, not this) and, on GitHub Actions, as an annotation. The
    /// Apple runners' first numbers are read from these.
    pub fn ci_note(text: &str) {
        let mut err = std::io::stderr();
        let _ = writeln!(err, "[view stages] {text}");
        if std::env::var_os("GITHUB_ACTIONS").is_some() {
            let _ = writeln!(err, "::notice title=View stages::{text}");
        }
    }

    /// The largest difference in codes a device frame may have from the CPU's: the bound of the
    /// colour pass's parity test above (`gpu_and_cpu_frames_agree_within_the_parity_tolerance`),
    /// twice the one code measured on lavapipe with the whole view on the device. The same on every
    /// adapter: a Metal column comes only with the Apple runners' numbers (the `[view stages]` CI notes).
    pub const CODE_BOUND: u8 = 2;

    /// The largest difference in codes a sharpened device frame may have from the CPU's where output
    /// sharpening's halo limiter flipped a rank (see `device_frames_equal_the_cpu_frames_within_the_golden_metric`):
    /// twice the 5 codes first measured on lavapipe.
    pub const HALO_FLIP_CODE_BOUND: u8 = 10;

    /// The share of samples over one code a sharpened device frame may have where output
    /// sharpening's halo limiter flipped ranks: twice the 1.214e-5 measured on lavapipe with
    /// pipeline 0.4.0 (the highlight range and the white lift together; the bound for every other
    /// frame is 1e-5). Owner's decision of 2026-10-17 (`docs/STATUS.md`).
    pub const HALO_FLIP_OVER_BOUND: f64 = 2.5e-5;

    pub fn picture(w: u32, h: u32) -> PhotoSource {
        let white = light("sony-ilce-7rm4", 4500.0);
        let scene = Scene::new(w, h, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
        Developed::new(&scene, MODEL).source()
    }

    pub fn device_renderer(gpu: &Arc<Gpu>, stages: GpuStages) -> Renderer {
        renderer_with(Some(Arc::clone(gpu)), stages, false, Method::AdaptiveRatio)
    }

    pub fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
        r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
    }

    pub fn fresh(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
        let options = FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        };
        r.viewport_frame(src, st, &options, &CancelToken::new())
            .expect("frame")
    }

    pub fn cell(w: f64, h: f64, scale: f64) -> ViewportState {
        ViewportState::new(ViewportRect::cell(w, h, scale))
    }

    pub fn at_100(w: f64, h: f64, image: (usize, usize)) -> ViewportState {
        let mut st = cell(w, h, 1.0);
        st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
        st
    }

    fn with_amount(src: &PhotoSource, pct: f64) -> PhotoSource {
        edit(src, |s| s.output_sharpening = pct)
    }

    /// The comparison of the golden runner (`tools/golden/src/image.rs`): CIEDE2000 per pixel on a
    /// 1/4 downsample (a box filter in linear light, the samples read as sRGB), and its median, 99th
    /// percentile and maximum.
    pub fn golden(a: &RenderedFrame, b: &RenderedFrame) -> (f64, f64, f64) {
        assert_eq!((a.width, a.height), (b.width, b.height));
        let to_xyz = ym_color::gamut::matrices().rec709_to_xyz;
        let white = d65_white_xyz();
        let table: Vec<f64> = (0..=255u32)
            .map(|v| srgb_decode(f64::from(v) / 255.0))
            .collect();
        let lab = |f: &RenderedFrame| -> Vec<[f64; 3]> {
            let (w, h) = (f.width / 4, f.height / 4);
            let mut out = Vec::with_capacity(w * h);
            for by in 0..h {
                for bx in 0..w {
                    let mut sum = [0.0f64; 3];
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let i = ((by * 4 + dy) * f.width + bx * 4 + dx) * 4;
                            for (c, s) in sum.iter_mut().enumerate() {
                                *s += table[usize::from(f.rgba[i + c])];
                            }
                        }
                    }
                    let linear = sum.map(|v| v / 16.0);
                    out.push(xyz_to_lab(linalg::mul_vec(&to_xyz, linear), white));
                }
            }
            out
        };
        let d: Vec<f64> = lab(a)
            .iter()
            .zip(&lab(b))
            .map(|(x, y)| delta_e_2000(*x, *y))
            .collect();
        assert!(!d.is_empty() && d.iter().all(|v| v.is_finite()));
        (
            percentile(&d, 50.0).expect("median"),
            percentile(&d, 99.0).expect("p99"),
            d.iter().copied().fold(0.0, f64::max),
        )
    }

    /// The largest difference in codes, the share of samples more than one code apart, the mean.
    pub fn codes(a: &RenderedFrame, b: &RenderedFrame) -> (u8, f64, f64) {
        assert_eq!((a.width, a.height), (b.width, b.height));
        let (mut worst, mut over, mut sum) = (0u8, 0usize, 0u64);
        for (x, y) in a.rgba.iter().zip(&b.rgba) {
            let d = x.abs_diff(*y);
            worst = worst.max(d);
            over += usize::from(d > 1);
            sum += u64::from(d);
        }
        let n = a.rgba.len() as f64;
        (worst, over as f64 / n, sum as f64 / n)
    }

    /// Whether the frame's notes say nothing about the graphics card (the classical-demosaic note
    /// is on every frame until a network exists).
    pub fn no_gpu_note(f: &RenderedFrame) -> bool {
        !f.report.notes.iter().any(|n| n.contains("graphics card"))
    }

    /// Bytes in use in the device's pool (what a finished or dropped frame must hand back).
    pub fn in_use(gpu: &Gpu) -> u64 {
        super::pool_in_use(gpu)
    }

    fn crop_and_straighten(s: &mut DevelopSettings) {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.05,
                y: 0.08,
                width: 0.88,
                height: 0.84,
            },
            angle_deg: 5.0,
            aspect: CropAspect::Free,
        });
    }

    #[test]
    fn device_frames_equal_the_cpu_frames_within_the_golden_metric() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let token = CancelToken::new();
        let (cpu, dev) = (on_cpu(), device_renderer(&gpu, VIEW));
        let image = cpu.final_size(&src, &token).expect("final size");
        // Fit, a retina cell (2x device pixels), the 100 % view, a straightened crop with paper in
        // its corners, a rotated photo, 200 % sharpening, no sharpening, and a magnified view (the
        // colour pass on the device, the stage and the magnification on the CPU).
        type Edit = fn(&mut DevelopSettings);
        let cases: Vec<(&str, ViewportState, Edit)> = vec![
            ("fit", cell(300.0, 200.0, 1.0), |_| {}),
            ("retina", cell(300.0, 200.0, 2.0), |_| {}),
            ("100 %", at_100(700.0, 500.0, image), |_| {}),
            ("crop and straighten", cell(300.0, 200.0, 1.0), |s| {
                crop_and_straighten(s);
            }),
            ("rotated", cell(300.0, 200.0, 1.0), |s| {
                s.orientation = Orientation::Deg90;
            }),
            ("amount 200", cell(300.0, 200.0, 1.0), |s| {
                s.output_sharpening = 200.0;
            }),
            ("amount 0", cell(300.0, 200.0, 1.0), |s| {
                s.output_sharpening = 0.0;
            }),
            (
                "magnified",
                {
                    let mut st = cell(300.0, 200.0, 1.0);
                    st.set_mode(ViewportMode::Percent { value: 300.0 }, image);
                    st
                },
                |_| {},
            ),
        ];
        for (name, st, change) in cases {
            let src = edit(&src, change);
            let a = frame(&cpu, &src, &st);
            let b = frame(&dev, &src, &st);
            assert_eq!(a.report.device, ComputeDevice::Cpu);
            assert_eq!(
                b.report.device,
                ComputeDevice::Gpu,
                "{name}: {:?}",
                b.report.notes
            );
            assert!(no_gpu_note(&b), "{name}: {:?}", b.report.notes);
            assert_eq!(b.report.timings.sample_ms, 0.0, "{name}: the CPU sampled");
            assert_eq!((a.width, a.height), (b.width, b.height), "{name}");
            let (median, p99, max) = golden(&a, &b);
            let (worst, over, mean) = codes(&a, &b);
            let line = format!(
                "{name}: ΔE00 median {median:.4} p99 {p99:.4} max {max:.4}; worst {worst} codes, over 1 code {over:e}, mean {mean:e}"
            );
            eprintln!("{line}");
            ci_note(&format!("{} {line}", gpu.info().name));
            // The first proposal of golden gates (`docs/17-gpu.md` section 12), which a whole GPU
            // render must meet.
            assert!(
                median <= 0.02 && p99 <= 0.05 && max <= 0.5,
                "{name}: {median} {p99} {max}"
            );
            // A sample past `CODE_BOUND` is allowed only where output sharpening's halo limiter
            // flipped a rank (`docs/05`, the output sharpening row of the parity table: up to 8.3e-3
            // display-linear within 8 px of a flipped pixel): the same frame without output
            // sharpening must then be within `CODE_BOUND`, so the colour pass and the sampler never
            // hide behind it. First seen with pipeline 0.4.0 (the highlight range of C7 changed the
            // patches' levels), on lavapipe at 100 %: 5 codes on six samples at one edge of a dark
            // patch (codes 18 to 23), with the amount at 100 only (one code at 0 and at 200, and one
            // code with the range at 0 or 1, so not the range itself); the bound is twice that. The
            // white lift alongside the range (this scene's lift is 1.04 stops) leaves it at 5 codes, 1
            // without output sharpening.
            // The same holds for the share of samples over one code (`HALO_FLIP_OVER_BOUND`): with
            // the white lift alongside the range, 1.214e-5 at 100 % on lavapipe against the 1e-5 of
            // every other frame, 4.3e-6 with the lift off.
            let (worst_allowed, over_allowed) = if (worst > CODE_BOUND || over > 1e-5)
                && src.settings.output_sharpening > 0.0
            {
                let unsharpened = edit(&src, |s| s.output_sharpening = 0.0);
                let (bare, bare_over, _) = codes(
                    &frame(&cpu, &unsharpened, &st),
                    &frame(&dev, &unsharpened, &st),
                );
                assert!(
                    bare <= CODE_BOUND && bare_over <= 1e-5,
                    "{name}: {bare} codes, over 1 code {bare_over:e} without output sharpening"
                );
                eprintln!(
                    "{name}: {worst} codes, over 1 code {over:e} from the halo limiter's rank flips; {bare} codes, {bare_over:e} without output sharpening"
                );
                (HALO_FLIP_CODE_BOUND, HALO_FLIP_OVER_BOUND)
            } else {
                (CODE_BOUND, 1e-5)
            };
            assert!(
                worst <= worst_allowed && mean < 0.25,
                "{name}: {worst} codes, mean {mean}"
            );
            // The view stages alone come in far below them: twice the first values measured on
            // lavapipe (ΔE00 max 0.082, one code at worst, none over one code, mean 7.1e-5 codes).
            // The same on every adapter; a Metal column comes only with the Apple runners' numbers
            // (the `[view stages]` notes above).
            assert!(
                max <= 0.17 && worst <= worst_allowed && over <= over_allowed && mean < 1.5e-4,
                "{name}: max {max}, {worst} codes, over {over}, mean {mean}"
            );
        }
    }

    #[test]
    fn a_move_of_the_slider_is_one_small_job_with_no_sampler_and_no_colour_pass() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let r = device_renderer(&gpu, VIEW);
        let st = cell(576.0, 384.0, 1.0);
        let (w, h) = (576usize, 384usize);
        let base = gpu.counters();
        let before = in_use(&gpu);
        let first = frame(&r, &src, &st);
        let cold = gpu.counters().since(base);
        eprintln!("cold frame: {cold:?}, runs {:?}", first.report.runs);
        assert_eq!(first.report.device, ComputeDevice::Gpu);
        assert_eq!(
            (
                first.report.runs.color_passes,
                first.report.runs.output_passes,
                first.report.runs.display_hits
            ),
            (1, 1, 0)
        );
        // One device job for the whole frame: one submission, and one readback, of the RGBA8 frame.
        assert_eq!(cold.submissions, 1, "{cold:?}");
        assert_eq!(
            (cold.readbacks, cold.readback_bytes),
            (1, (w * h * 4) as u64)
        );
        // The display-linear frame (the view and the stage's border, four f32 planes) stays resident
        // and is on the pool's ledger, and in the cache's byte count.
        let resident = in_use(&gpu) - before;
        let display_bytes = 16 * (w as u64 + 34) * (h as u64 + 34);
        assert!(resident >= display_bytes, "{resident} < {display_bytes}");
        assert!(r.cached_bytes() as u64 >= display_bytes);

        // The slider moves: the resident frame is sharpened and encoded again. Nothing is sampled
        // or coloured, nothing but a few uniform blocks is uploaded, and one small job runs.
        let c1 = gpu.counters();
        let moved = frame(&r, &with_amount(&src, 60.0), &st);
        let slide = gpu.counters().since(c1);
        eprintln!("slider move: {slide:?}, runs {:?}", moved.report.runs);
        assert_eq!(
            moved.report.runs,
            StageRuns {
                output_passes: 1,
                display_hits: 1,
                ..StageRuns::default()
            }
        );
        let t = moved.report.timings;
        assert_eq!((t.sample_ms, t.pass_ms), (0.0, 0.0));
        assert!(t.output_ms > 0.0, "the move reports its time: {t:?}");
        assert_eq!(slide.submissions, 1, "{slide:?}");
        assert_eq!(
            (slide.readbacks, slide.readback_bytes),
            (1, (w * h * 4) as u64)
        );
        assert!(
            slide.uploaded_bytes < 16 << 10,
            "a slider move uploaded {} bytes",
            slide.uploaded_bytes
        );
        assert_eq!(moved.report.device, ComputeDevice::Gpu);
        assert_eq!(
            in_use(&gpu) - before,
            resident,
            "the move made no new resident frame"
        );

        // Back to the first value: the finished frame, no device work at all.
        let c2 = gpu.counters();
        let again = frame(&r, &src, &st);
        assert_eq!(again.report.runs.frame_hits, 1);
        assert_eq!(gpu.counters().since(c2).submissions, 0);
        assert_eq!(again.rgba, first.rgba);

        // Zero is no stage: the plain road (the colour pass to C13 in one dispatch), no output pass,
        // no raw work.
        let off = frame(&r, &with_amount(&src, 0.0), &st);
        assert_eq!(
            (off.report.runs.output_passes, off.report.runs.color_passes),
            (0, 1)
        );
        assert_eq!(off.report.runs.raw_work(), 0);

        // The photo's caches let go of the frame: the pool gets its buffers back.
        r.forget(src.id);
        assert!(
            in_use(&gpu) - before < resident,
            "the resident frame outlived its photo's caches"
        );
    }

    struct CubeLook {
        transform: Arc<ym_color::look::LookTransform>,
    }

    impl CubeLook {
        fn new(points: usize) -> Self {
            use ym_color::look::{LookInput, LookTransform};
            use ym_color::lut::{Lut, Lut3d};
            let cube = Lut3d::bake(points, [0.0; 3], [1.0; 3], |c| {
                [c[0] * 0.95 + 0.02, c[1], (c[2] * 1.04).min(1.0)]
            })
            .expect("cube");
            let transform = LookTransform::new(LookInput::LogC4, None, Some(Lut::from_cube(cube)))
                .expect("look");
            Self {
                transform: Arc::new(transform),
            }
        }
    }

    impl ym_core::pipeline::LookProvider for CubeLook {
        fn look(
            &self,
            id: &ym_looks::library::LookId,
        ) -> Result<ym_core::pipeline::ResolvedLook, ym_core::pipeline::LookProblem> {
            match id.as_str() {
                "builtin:neutral" => NeutralOnly.look(id),
                "user:cube" => Ok(ym_core::pipeline::ResolvedLook {
                    transform: Arc::clone(&self.transform),
                    fingerprint: "cube-v1".to_owned(),
                }),
                _ => Err(ym_core::pipeline::LookProblem(
                    "File missing or changed. Import again.".to_owned(),
                )),
            }
        }

        fn display_transform(
            &self,
            id: &ym_looks::library::LookId,
        ) -> Result<(ym_color::display::DisplayTransform, String), ym_core::pipeline::LookProblem>
        {
            NeutralOnly.display_transform(id)
        }
    }

    #[test]
    fn a_colour_only_edit_uploads_no_look_cube_and_runs_no_raw_stage() {
        let Some(gpu) = gpu() else { return };
        // A 65-point look cube is 4.4 MB; the camera RGB of this view is 0.7 MB.
        let cube_bytes = 65u64.pow(3) * 16;
        let r = Renderer::new(
            Arc::new(CubeLook::new(65)),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: VIEW,
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        let look = ym_looks::library::LookId::parse("user:cube").expect("id");
        let src = edit(&picture(600, 400), |s| s.look = look.clone());
        let st = cell(300.0, 200.0, 1.0);
        let c0 = gpu.counters();
        let cold = frame(&r, &src, &st);
        let uploaded_cold = gpu.counters().since(c0).uploaded_bytes;
        assert_eq!(
            cold.report.device,
            ComputeDevice::Gpu,
            "{:?}",
            cold.report.notes
        );
        assert!(
            uploaded_cold >= cube_bytes,
            "the cube went up once: {uploaded_cold}"
        );
        assert!(cold.report.runs.raw_work() > 0);
        // Edits of the grade: a control, a curve, the look's intensity, white balance.
        type Edit = Box<dyn Fn(&mut DevelopSettings)>;
        let edits: Vec<(&str, Edit)> = vec![
            (
                "contrast",
                Box::new(|s| s.contrast = ym_core::settings::PercentSetting::manual_value(30.0)),
            ),
            (
                "saturation",
                Box::new(|s| s.saturation = ym_core::settings::PercentSetting::manual_value(-25.0)),
            ),
            (
                "curve",
                Box::new(|s| s.curve.luma = vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]),
            ),
            ("look intensity", Box::new(|s| s.look_intensity = 60.0)),
        ];
        for (name, change) in edits {
            let edited = edit(&src, |s| change(s));
            let c = gpu.counters();
            let f = frame(&r, &edited, &st);
            let up = gpu.counters().since(c);
            eprintln!("{name}: {up:?}");
            assert_eq!(f.report.device, ComputeDevice::Gpu, "{name}");
            // Colour and grade work and no raw stage; the cube stays where it is.
            assert_eq!(f.report.runs.raw_work(), 0, "{name}: {:?}", f.report.runs);
            assert_eq!(f.report.timings.raw_ms, 0.0, "{name}");
            assert!(
                up.uploaded_bytes < cube_bytes / 2,
                "{name}: {} bytes uploaded, the cube is {cube_bytes}",
                up.uploaded_bytes
            );
            assert_eq!(up.submissions, 1, "{name}: {up:?}");
        }
    }

    #[test]
    fn a_frame_queued_behind_other_device_work_stops_at_once_when_it_is_cancelled_or_left() {
        let Some(gpu) = gpu() else { return };
        let mut src = picture(900, 600);
        src.id = PhotoId(1);
        let mut other = picture(640, 480);
        other.id = PhotoId(2);
        let r = device_renderer(&gpu, VIEW);
        r.navigate(src.id);
        let before = in_use(&gpu);
        // Hold the device thread with a job of our own; the frame's device job queues behind it.
        let hold = |gpu: &Arc<Gpu>| {
            let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
            let job = gpu.submit(
                ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
                move |_, _| {
                    ready_tx.send(()).expect("ready");
                    gate_rx.recv().expect("gate");
                },
            );
            ready_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the device thread took the gate job");
            (gate_tx, job)
        };
        let wait_queued = |gpu: &Gpu| {
            let started = std::time::Instant::now();
            while gpu.scheduler().pending() == 0 {
                assert!(started.elapsed().as_secs() < 60, "no device job was queued");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        let st = cell(300.0, 200.0, 1.0);

        // 1. The render's own token (a newer request, a closed session): the render resolves with
        // the gate still closed, and the queued job is skipped when it is popped.
        let (gate, job) = hold(&gpu);
        let handle = r.spawn_viewport_frame(src.clone(), st.clone(), FrameOptions::default());
        wait_queued(&gpu);
        handle.cancel();
        let result = handle
            .wait_timeout(std::time::Duration::from_secs(30))
            .expect("the render resolved while the device thread was still held");
        assert!(
            matches!(result, Err(JobError::Cancelled))
                || matches!(result, Ok(Err(RenderError::Cancelled))),
            "{result:?}"
        );
        gate.send(()).expect("open the gate");
        job.wait().expect("gate job");

        // 2. Moving to another photo: the device job of the photo left is dropped from the queue at
        // once (it holds the photo's camera RGB), not left to be skipped when it is popped.
        let (gate, job) = hold(&gpu);
        let handle = r.spawn_viewport_frame(src.clone(), st.clone(), FrameOptions::default());
        wait_queued(&gpu);
        r.navigate(other.id);
        let result = handle
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
        gate.send(()).expect("open the gate");
        job.wait().expect("gate job");

        // The device is as it was, and a fresh request works.
        let f = frame(&r, &other, &st);
        assert_eq!(f.report.device, ComputeDevice::Gpu);
        r.forget(other.id);
        r.forget(src.id);
        assert!(
            in_use(&gpu) <= before + (1 << 20),
            "the ledger returns to its baseline"
        );
    }

    #[test]
    fn every_subset_of_the_view_stages_renders_the_picture_of_the_cpu() {
        let Some(gpu) = gpu() else { return };
        let src = picture(600, 400);
        let tilted = edit(&src, crop_and_straighten);
        let cpu = on_cpu();
        let st = cell(300.0, 200.0, 1.0);
        let members = [
            GpuStages::SAMPLER,
            GpuStages::COLOR,
            GpuStages::OUTPUT_SHARPEN,
            GpuStages::ENCODE,
        ];
        for mask in 0u8..16 {
            let stages = members
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .fold(GpuStages::NONE, |acc, (_, m)| acc.union(*m));
            let (sampler, colour) = (mask & 1 != 0, mask & 2 != 0);
            for (name, photo) in [
                ("sharpened", src.clone()),
                ("tilted", tilted.clone()),
                ("plain", with_amount(&src, 0.0)),
            ] {
                let r = device_renderer(&gpu, stages);
                let jobs = gpu.counters().submissions;
                let a = fresh(&cpu, &photo, &st);
                let b = fresh(&r, &photo, &st);
                let ran = gpu.counters().submissions - jobs;
                // The colour pass decides where the frame is said to be made; the sampler alone
                // still uses the device, and a tail without them does not.
                let want = if colour {
                    ComputeDevice::Gpu
                } else {
                    ComputeDevice::Cpu
                };
                assert_eq!(
                    b.report.device, want,
                    "{name} {stages}: {:?}",
                    b.report.notes
                );
                assert_eq!(ran > 0, sampler || colour, "{name} {stages}: {ran} jobs");
                assert!(no_gpu_note(&b), "{name} {stages}: {:?}", b.report.notes);
                assert_eq!(
                    b.report.runs.color_passes, 1,
                    "{name} {stages}: {:?}",
                    b.report.runs
                );
                assert_eq!(
                    b.report.runs.output_passes, a.report.runs.output_passes,
                    "{name} {stages}"
                );
                let (median, p99, max) = golden(&a, &b);
                let (worst, _, mean) = codes(&a, &b);
                assert!(
                    median <= 0.02
                        && p99 <= 0.05
                        && max <= 0.5
                        && worst <= CODE_BOUND
                        && mean < 0.25,
                    "{name} {stages}: ΔE00 {median} {p99} {max}, {worst} codes, mean {mean}"
                );
                if !sampler && !colour {
                    assert_eq!(
                        b.rgba, a.rgba,
                        "{name} {stages}: no device, the CPU's bytes"
                    );
                }
            }
        }
    }

    #[test]
    fn a_view_cut_into_bands_has_the_pixels_of_the_view_in_one_piece() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let whole = device_renderer(&gpu, VIEW);
        let banded = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: VIEW,
                // About 110 rows of a 576 px view with the stage's border (16 bytes a pixel).
                view_band_bytes: 1_400_000,
                full_tile_px: 512,
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        let image = whole.final_size(&src, &CancelToken::new()).expect("size");
        let tilted = edit(&src, crop_and_straighten);
        let cases: Vec<(&str, PhotoSource, ViewportState)> = vec![
            ("fit", src.clone(), cell(576.0, 384.0, 1.0)),
            ("odd cell", src.clone(), cell(577.0, 383.0, 1.0)),
            ("retina", src.clone(), cell(300.0, 200.0, 2.0)),
            ("tilted crop", tilted.clone(), cell(576.0, 384.0, 1.0)),
            ("plain", with_amount(&src, 0.0), cell(576.0, 384.0, 1.0)),
            ("before", src.clone(), {
                let mut st = cell(576.0, 384.0, 1.0);
                st.before = true;
                st
            }),
            ("100 %", src.clone(), at_100(640.0, 480.0, image)),
        ];
        // A large photo seen small: the reduced images (1/4 and 1/2) the views read are larger than
        // the band size too, so the bands share them out and each uploads only its own part.
        let large = picture(1800, 1200);
        let cases = cases.into_iter().chain([
            ("fit of 1/4", large.clone(), cell(400.0, 267.0, 1.0)),
            ("fit of 1/2", large.clone(), cell(700.0, 467.0, 1.0)),
            (
                "fit of 1/2, plain",
                with_amount(&large, 0.0),
                cell(700.0, 467.0, 1.0),
            ),
        ]);
        for (name, photo, st) in cases {
            let a = fresh(&whole, &photo, &st);
            let jobs = gpu.counters().submissions;
            let b = fresh(&banded, &photo, &st);
            let ran = gpu.counters().submissions - jobs;
            assert_eq!(
                b.report.device,
                ComputeDevice::Gpu,
                "{name}: {:?}",
                b.report.notes
            );
            eprintln!("{name}: {ran} device jobs");
            assert!(ran >= 2, "{name}: the view was cut into {ran} jobs");
            assert!(
                a.rgba == b.rgba,
                "{name}: the bands differ from the view in one piece ({:?})",
                codes(&a, &b)
            );
        }
        // The display-linear frames of the bands are cached one by one: a move of the slider runs
        // the tail of each band and nothing else.
        let st = cell(576.0, 384.0, 1.0);
        let first = frame(&banded, &src, &st);
        let moved = frame(&banded, &with_amount(&src, 40.0), &st);
        let bands = first.report.runs.output_passes;
        assert!(bands >= 3, "{bands}");
        assert_eq!(
            moved.report.runs,
            StageRuns {
                output_passes: bands,
                display_hits: bands,
                ..StageRuns::default()
            }
        );
        let want = fresh(&whole, &with_amount(&src, 40.0), &st);
        assert!(moved.rgba == want.rgba);
    }

    /// A renderer with `stages` on the device whose device passes hold at most `bytes` per binding.
    pub fn small_binding(gpu: &Arc<Gpu>, stages: GpuStages, bytes: u64) -> Renderer {
        Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                gpu: Some(Arc::clone(gpu)),
                gpu_stages: stages,
                view_band_bytes: bytes,
                full_tile_px: 512,
                ..RenderOptions::default()
            },
        )
        .expect("render thread")
    }

    #[test]
    fn a_move_of_the_slider_gives_the_frame_a_fresh_render_gives_every_time() {
        // The resident display-linear frame is read by every move and written by none: whatever order
        // the amounts come in, the frame of a move is the frame of a render from scratch.
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let r = device_renderer(&gpu, VIEW);
        let reference = device_renderer(&gpu, VIEW);
        let tilted = edit(&src, crop_and_straighten);
        for (name, photo, st) in [
            ("fit", src.clone(), cell(576.0, 384.0, 1.0)),
            ("odd cell with paper", tilted, cell(577.0, 383.0, 1.0)),
        ] {
            let _ = frame(&r, &photo, &st);
            for amount in [60.0, 61.0, 150.0, 200.0, 10.0, 0.5, 99.0] {
                let moved = frame(&r, &with_amount(&photo, amount), &st);
                assert_eq!(moved.report.runs.display_hits, 1, "{name} {amount}");
                assert_eq!(moved.report.runs.color_passes, 0, "{name} {amount}");
                let want = fresh(&reference, &with_amount(&photo, amount), &st);
                assert!(
                    moved.rgba == want.rgba,
                    "{name}: the move to {amount} % differs from a fresh render ({:?})",
                    codes(&moved, &want)
                );
            }
        }
    }

    #[test]
    fn a_frame_over_one_binding_keeps_the_stages_that_are_on_whichever_they_are() {
        // Any set of view members (`Auto` ran the colour pass alone until 2026-10-14; the others are
        // what `On` adds): whichever members are on, a frame larger than one storage binding is cut and
        // made by the device, with the pixels of the frame in one piece, and not quietly given to
        // the CPU (which would be a CPU frame, slow and unannounced, for a big display).
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let tilted = edit(&src, crop_and_straighten);
        let members = [
            GpuStages::COLOR,
            GpuStages::COLOR.union(GpuStages::SAMPLER),
            GpuStages::COLOR
                .union(GpuStages::OUTPUT_SHARPEN)
                .union(GpuStages::ENCODE),
            VIEW,
        ];
        for stages in members {
            let whole = small_binding(&gpu, stages, ym_core::pipeline::graph::VIEW_BAND_BYTES);
            // About 110 rows of a 576 px view (16 bytes a pixel).
            let cut = small_binding(&gpu, stages, 1_400_000);
            for (name, photo) in [
                ("sharpened", src.clone()),
                ("tilted crop", tilted.clone()),
                ("plain", with_amount(&src, 0.0)),
            ] {
                // The second cell is tall: the photo has paper above and below it, and some bands
                // of the cut are nothing but paper.
                for (w, h) in [(576.0, 384.0), (300.0, 520.0)] {
                    let st = cell(w, h, 1.0);
                    let a = fresh(&whole, &photo, &st);
                    let jobs = gpu.counters().submissions;
                    let b = fresh(&cut, &photo, &st);
                    let ran = gpu.counters().submissions - jobs;
                    let what = format!("{name} {w}x{h} {stages}");
                    assert_eq!(
                        b.report.device,
                        ComputeDevice::Gpu,
                        "{what}: {:?}",
                        b.report.notes
                    );
                    assert!(ran >= 2, "{what}: {ran} device jobs");
                    assert!(b.rgba == a.rgba, "{what}: {:?}", codes(&a, &b));
                }
            }
        }
    }

    #[test]
    fn bands_of_a_tilted_photo_with_empty_corners_have_the_pixels_of_the_view_in_one_piece() {
        // The stage works on the bounding box of the photo's pixels and sharpens the empty corners of
        // a straightened photo against black: a band must see the same box as the view does, or the
        // pixels at the slanted edge differ between the cut view and the view in one piece. The
        // portrait cell has bands that are nothing but paper (a landscape photo in a tall cell).
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let whole = device_renderer(&gpu, VIEW);
        let cpu = on_cpu();
        // (angle, cell, bytes a binding holds). A band of a rotated view reads the bounding box of a
        // rotated strip, which is as wide as the photo and `width * tan(angle)` tall even for one
        // row, so the bytes must hold that.
        let cases: [(f64, (f64, f64), u64); 5] = [
            (4.0, (576.0, 384.0), 1_400_000),
            (12.0, (576.0, 384.0), 3_000_000),
            (-9.0, (576.0, 384.0), 3_000_000),
            (4.0, (300.0, 520.0), 1_400_000),
            (-12.0, (300.0, 520.0), 1_400_000),
        ];
        for (angle, (w, h), bytes) in cases {
            let tilted = edit(&src, |s| {
                s.crop = Some(Crop {
                    rect: NormRect {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    angle_deg: angle,
                    aspect: CropAspect::Free,
                });
            });
            let cut = small_binding(&gpu, VIEW, bytes);
            let st = cell(w, h, 1.0);
            let a = fresh(&whole, &tilted, &st);
            let jobs = gpu.counters().submissions;
            let b = fresh(&cut, &tilted, &st);
            let ran = gpu.counters().submissions - jobs;
            let what = format!("{angle} degrees, {w}x{h}");
            assert_eq!(
                b.report.device,
                ComputeDevice::Gpu,
                "{what}: {:?}",
                b.report.notes
            );
            assert!(ran >= 2, "{what}: {ran} device jobs");
            let c = fresh(&cpu, &tilted, &st);
            eprintln!(
                "{what}: {ran} jobs, bands against one piece {:?}, one piece against the CPU {:?}",
                codes(&a, &b),
                codes(&a, &c)
            );
            assert!(
                a.rgba == b.rgba,
                "{what}: the bands differ from the view in one piece {:?}",
                codes(&a, &b)
            );
        }
    }

    #[test]
    fn a_resident_frame_is_never_served_to_another_device() {
        // A device that replaces another one (`Renderer::set_gpu`) used to be allocated where the
        // first one was (the first run of this test on the first implementation: the same address),
        // and the resident display-linear frame the photo's caches still held was bound in the new
        // device's passes: a validation error that lost a healthy device and gave the frame to the
        // CPU. A frame belongs to the device that made it, whatever the next one's address.
        let Some(first) = gpu() else { return };
        let src = picture(600, 400);
        let st = cell(300.0, 200.0, 1.0);
        let r = device_renderer(&first, VIEW);
        let _ = frame(&r, &src, &st);
        r.set_gpu(None, GpuStages::NONE);
        drop(first);
        for i in 0..8 {
            let amount = 60.0 + f64::from(i);
            let Some(next) = gpu() else { return };
            r.set_gpu(Some(Arc::clone(&next)), VIEW);
            let moved = frame(&r, &with_amount(&src, amount), &st);
            assert_eq!(
                moved.report.device,
                ComputeDevice::Gpu,
                "{:?}",
                moved.report.notes
            );
            assert!(no_gpu_note(&moved), "{:?}", moved.report.notes);
            assert!(!next.is_lost(), "the new device was lost");
            assert_eq!(
                moved.report.runs.display_hits, 0,
                "the frame of another device"
            );
            let want = fresh(
                &device_renderer(&next, VIEW),
                &with_amount(&src, amount),
                &st,
            );
            assert!(moved.rgba == want.rgba);
            // The next device takes the place of this one.
            r.set_gpu(None, GpuStages::NONE);
            drop(next);
        }
    }

    #[test]
    fn tiny_and_odd_cells_render_on_the_device_like_on_the_cpu() {
        let Some(gpu) = gpu() else { return };
        let src = picture(300, 200);
        let (cpu, dev) = (on_cpu(), device_renderer(&gpu, VIEW));
        let image = cpu.final_size(&src, &CancelToken::new()).expect("size");
        let cells: [(f64, f64); 9] = [
            (1.0, 1.0),
            (1.0, 37.0),
            (41.0, 1.0),
            (2.0, 2.0),
            (3.0, 5.0),
            (17.0, 9.0),
            (33.0, 33.0),
            (64.0, 1.0),
            (7.0, 120.0),
        ];
        for (w, h) in cells {
            for scale in [1.0, 2.0] {
                for (name, st) in [
                    ("fit", cell(w, h, scale)),
                    ("100 %", {
                        let mut st = cell(w, h, scale);
                        st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
                        st
                    }),
                    ("before", {
                        let mut st = cell(w, h, scale);
                        st.before = true;
                        st
                    }),
                ] {
                    for amount in [100.0, 0.0] {
                        let photo = with_amount(&src, amount);
                        let a = fresh(&cpu, &photo, &st);
                        let b = fresh(&dev, &photo, &st);
                        let what = format!("{w}x{h} @{scale} {name} amount {amount}");
                        assert_eq!(b.report.device, ComputeDevice::Gpu, "{what}");
                        assert!(no_gpu_note(&b), "{what}: {:?}", b.report.notes);
                        assert_eq!((a.width, a.height), (b.width, b.height), "{what}");
                        let (worst, _, _) = codes(&a, &b);
                        assert!(worst <= CODE_BOUND, "{what}: {worst} codes");
                    }
                }
            }
        }
    }

    #[test]
    fn a_thumbnail_and_a_look_tile_run_at_thumbnail_priority_behind_the_viewport() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let r = device_renderer(&gpu, VIEW);
        let st = cell(300.0, 200.0, 1.0);
        // Warm up everything that is not a device job (the raw group, the plans), so that the two
        // requests below go straight to the device.
        let _ = frame(&r, &src, &st);
        let _ = r
            .thumbnail_png(&src, &CancelToken::new())
            .expect("thumbnail");
        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        let hold = gpu.submit(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |_, _| {
                ready_tx.send(()).expect("ready");
                gate_rx.recv().expect("gate");
            },
        );
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the device thread took the gate job");
        let order = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
        let queued = |n: usize| {
            let started = std::time::Instant::now();
            while gpu.scheduler().pending() < n {
                assert!(started.elapsed().as_secs() < 60, "device jobs not queued");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        // A thumbnail first, then a viewport frame (of a size nothing has cached), then a look tile.
        let thumb = {
            let (r, src, order) = (r.clone(), src.clone(), Arc::clone(&order));
            std::thread::spawn(move || {
                r.thumbnail_png(&src, &CancelToken::new())
                    .expect("thumbnail");
                order.lock().expect("order").push("thumbnail");
            })
        };
        queued(1);
        let viewport = {
            let (r, src, order) = (r.clone(), src.clone(), Arc::clone(&order));
            std::thread::spawn(move || {
                let st = cell(310.0, 207.0, 1.0);
                frame(&r, &src, &st);
                order.lock().expect("order").push("viewport");
            })
        };
        queued(2);
        gate_tx.send(()).expect("open the gate");
        hold.wait().expect("gate job");
        thumb.join().expect("thumbnail thread");
        viewport.join().expect("viewport thread");
        assert_eq!(
            *order.lock().expect("order"),
            ["viewport", "thumbnail"],
            "the viewport's job ran first although it was queued second"
        );
    }

    #[test]
    fn the_histogram_sample_comes_from_the_device_and_is_counted_on_the_cpu() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let cpu = on_cpu();
        let dev = device_renderer(&gpu, VIEW);
        let token = CancelToken::new();
        let a = cpu.histogram(&src, &token).expect("cpu histogram");
        let jobs = gpu.counters().submissions;
        let b = dev.histogram(&src, &token).expect("device histogram");
        assert_eq!(
            gpu.counters().submissions - jobs,
            1,
            "the sample was made by one device job"
        );
        // The device sampler is the CPU's within 5e-6 of a sample: a count can change only where a
        // sample sits on a bin edge.
        let mut worst = 0.0f64;
        for (x, y) in [(&a.red, &b.red), (&a.green, &b.green), (&a.blue, &b.blue)] {
            assert_eq!(x.len(), y.len());
            for (p, q) in x.iter().zip(y) {
                worst = worst.max((p - q).abs());
            }
        }
        eprintln!("histogram, device against CPU: worst bin difference {worst}");
        assert!(worst <= 0.02, "{worst}");
    }

    #[test]
    fn a_strict_policy_and_the_off_policy_keep_to_their_word() {
        let Some(gpu) = gpu() else { return };
        let src = picture(600, 400);
        let st = cell(300.0, 200.0, 1.0);
        // Off: a device is attached but no stage is on it: the CPU's bytes, and the device is not asked.
        let off = device_renderer(&gpu, GpuStages::NONE);
        let jobs = gpu.counters().submissions;
        let a = frame(&off, &src, &st);
        let b = frame(&on_cpu(), &src, &st);
        assert_eq!(a.rgba, b.rgba);
        assert_eq!(gpu.counters().submissions, jobs);
        assert_eq!(a.report.device, ComputeDevice::Cpu);
        // The thumbnail and the look tile too.
        let t = off
            .thumbnail_png(&src, &CancelToken::new())
            .expect("thumbnail");
        assert_eq!(
            t,
            on_cpu()
                .thumbnail_png(&src, &CancelToken::new())
                .expect("thumbnail")
        );
        assert_eq!(gpu.counters().submissions, jobs);
    }

    #[test]
    fn two_renders_and_a_render_after_another_photo_are_byte_identical() {
        let Some(gpu) = gpu() else { return };
        let src = picture(900, 600);
        let other = picture(640, 480);
        let r = device_renderer(&gpu, VIEW);
        let st = cell(576.0, 384.0, 1.0);
        let first = fresh(&r, &src, &st);
        let second = fresh(&r, &src, &st);
        assert!(first.rgba == second.rgba, "twice");
        // Another photo in between (pool history), on a fresh renderer.
        let _ = fresh(&device_renderer(&gpu, VIEW), &other, &st);
        let third = fresh(&device_renderer(&gpu, VIEW), &src, &st);
        assert!(first.rgba == third.rgba, "after another photo");
        // A thumbnail and a look tile.
        let token = CancelToken::new();
        let t1 = r.thumbnail_png(&src, &token).expect("thumbnail");
        let t2 = device_renderer(&gpu, VIEW)
            .thumbnail_png(&src, &token)
            .expect("thumbnail");
        assert_eq!(t1, t2);
    }
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
                vec![("R1..R9".to_owned(), ComputeDevice::Cpu)],
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
            vec![("R1..R9".to_owned(), ComputeDevice::Cpu)]
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
        // Never a mix of devices in one file (`docs/17-gpu.md` section 9): the route is decided by flags
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
                vec![("R1..R9".to_owned(), ComputeDevice::Cpu)],
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
            vec![("R1..R9".to_owned(), ComputeDevice::Gpu)]
        );
    }

    #[test]
    fn a_viewport_tile_that_fails_is_made_on_the_cpu_with_a_note() {
        let Some(gpu) = gpu() else { return };
        let src = scene();
        let dev = on_device(&gpu);
        // The baseline: the CPU raw chain with the same view stages on the device.
        let colour_only = renderer_with(
            Some(Arc::clone(&gpu)),
            view::VIEW,
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
            vec![("R1..R9".to_owned(), ComputeDevice::Cpu)]
        );
    }

    mod views {
        //! A device error in the view stages: the frame, the thumbnail and the histogram are made on
        //! the CPU, bit for bit what a renderer without a device makes, with the note; a strict
        //! policy fails the request instead.
        use super::super::view::{VIEW, cell, device_renderer, fresh, no_gpu_note, picture};
        use super::*;

        #[test]
        fn a_device_error_is_answered_by_the_cpu_with_the_note() {
            let src = picture(600, 400);
            let st = cell(300.0, 200.0, 1.0);
            let token = CancelToken::new();
            let want = fresh(&on_cpu(), &src, &st);
            let thumb = on_cpu().thumbnail_png(&src, &token).expect("cpu thumbnail");
            let hist = on_cpu().histogram(&src, &token).expect("cpu histogram");
            // (name, fault, whether the route is closed for the photo from then on)
            let faults = [
                ("out of budget", GpuFault::OutOfBudget, false),
                ("shader", GpuFault::Shader, true),
                ("lost device", GpuFault::Lost, true),
            ];
            for (name, fault, closed) in faults {
                let Some(gpu) = gpu() else { return };
                let r = device_renderer(&gpu, VIEW);
                let before = pool_in_use(&gpu);
                gpu.set_fault(Some(fault));
                let f = fresh(&r, &src, &st);
                // The thumbnail and the histogram of the same photo meet the same fault.
                let t = r.thumbnail_png(&src, &token).expect("thumbnail");
                let h = r.histogram(&src, &token).expect("histogram");
                gpu.set_fault(None);
                assert_eq!(f.report.device, ComputeDevice::Cpu, "{name}");
                assert!(f.rgba == want.rgba, "{name}: the CPU frame, bit for bit");
                assert_eq!(t, thumb, "{name}: the CPU thumbnail");
                assert_eq!(h, hist, "{name}: the CPU histogram");
                assert!(
                    f.report.notes.iter().any(|n| n.starts_with(
                        "The graphics card could not render this frame, so the CPU did"
                    )),
                    "{name}: {:?}",
                    f.report.notes
                );
                if !matches!(fault, GpuFault::Lost) {
                    assert_eq!(pool_in_use(&gpu), before, "{name}: the ledger is back");
                }
                // The next request: a buffer that did not fit is a property of that request and the
                // device is asked again; a shader that does not compile or a lost device closes the
                // route, quietly, for the photo (and, for a shader, the process).
                let jobs = gpu.counters().submissions;
                let again = fresh(&r, &src, &st);
                if closed {
                    assert_eq!(again.report.device, ComputeDevice::Cpu, "{name}");
                    assert_eq!(gpu.counters().submissions, jobs, "{name}: not asked again");
                    assert!(again.rgba == want.rgba, "{name}");
                } else {
                    assert_eq!(again.report.device, ComputeDevice::Gpu, "{name}");
                    assert!(no_gpu_note(&again), "{name}: {:?}", again.report.notes);
                }
            }
        }

        #[test]
        fn a_strict_policy_fails_the_request_instead_of_falling_back() {
            let src = picture(600, 400);
            let st = cell(300.0, 200.0, 1.0);
            for (name, fault) in [
                ("out of budget", GpuFault::OutOfBudget),
                ("shader", GpuFault::Shader),
                ("lost device", GpuFault::Lost),
            ] {
                let Some(gpu) = gpu() else { return };
                let strict =
                    renderer_with(Some(Arc::clone(&gpu)), VIEW, true, Method::AdaptiveRatio);
                gpu.set_fault(Some(fault));
                let frame =
                    strict.viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new());
                let thumb = strict.thumbnail_png(&src, &CancelToken::new());
                gpu.set_fault(None);
                assert!(
                    matches!(frame, Err(RenderError::Gpu(_))),
                    "{name}: {frame:?}"
                );
                assert!(
                    matches!(thumb, Err(RenderError::Gpu(_))),
                    "{name}: {thumb:?}"
                );
            }
        }

        #[test]
        fn a_device_error_in_a_cut_view_is_answered_by_the_whole_frame_on_the_cpu() {
            // The view is cut into bands; the device refuses (the first band's buffers): no band of
            // the device is mixed into the frame, the frame is the CPU's bit for bit, with the note,
            // and a strict policy fails the request.
            use super::super::view::small_binding;
            let src = picture(600, 400);
            let st = cell(300.0, 200.0, 1.0);
            let want = fresh(&on_cpu(), &src, &st);
            for (name, fault) in [
                ("out of budget", GpuFault::OutOfBudget),
                ("lost device", GpuFault::Lost),
            ] {
                let Some(gpu) = gpu() else { return };
                let cut = small_binding(&gpu, VIEW, 400_000);
                let strict = Renderer::new(
                    Arc::new(NeutralOnly),
                    RenderOptions {
                        gpu: Some(Arc::clone(&gpu)),
                        gpu_stages: VIEW,
                        gpu_strict: true,
                        view_band_bytes: 400_000,
                        ..RenderOptions::default()
                    },
                )
                .expect("render thread");
                let before = pool_in_use(&gpu);
                gpu.set_fault(Some(fault));
                let f = fresh(&cut, &src, &st);
                let refused = strict.viewport_frame(
                    &src,
                    &st,
                    &FrameOptions {
                        no_cache: true,
                        ..FrameOptions::default()
                    },
                    &CancelToken::new(),
                );
                gpu.set_fault(None);
                assert_eq!(f.report.device, ComputeDevice::Cpu, "{name}");
                assert!(f.rgba == want.rgba, "{name}: the CPU frame, bit for bit");
                assert!(
                    f.report.notes.iter().any(|n| n.starts_with(
                        "The graphics card could not render this frame, so the CPU did"
                    )),
                    "{name}: {:?}",
                    f.report.notes
                );
                assert!(
                    matches!(refused, Err(RenderError::Gpu(_))),
                    "{name}: {refused:?}"
                );
                if !matches!(fault, GpuFault::Lost) {
                    assert_eq!(pool_in_use(&gpu), before, "{name}: the ledger is back");
                }
            }
        }

        #[test]
        fn a_cancelled_job_leaves_nothing_resident_and_the_next_request_works() {
            let Some(gpu) = gpu() else { return };
            let src = picture(600, 400);
            let st = cell(300.0, 200.0, 1.0);
            let r = device_renderer(&gpu, VIEW);
            let before = pool_in_use(&gpu);
            gpu.set_fault(Some(GpuFault::CancelAtJob(1)));
            let result = r.viewport_frame(&src, &st, &FrameOptions::default(), &CancelToken::new());
            gpu.set_fault(None);
            assert!(matches!(result, Err(RenderError::Cancelled)), "{result:?}");
            assert_eq!(pool_in_use(&gpu), before);
            let again = fresh(&r, &src, &st);
            assert_eq!(again.report.device, ComputeDevice::Gpu);
            assert!(no_gpu_note(&again), "{:?}", again.report.notes);
        }
    }
}
