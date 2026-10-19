//! The device frame of the native viewport (`docs/17-gpu.md` section 16).
//!
//! A viewport frame asked for with [`FrameOutput::Device`] is not read back: the pool buffer the view
//! job's kernels wrote is handed over as a [`ym_core::pipeline::DeviceFrame`], and the native
//! viewport draws it from there. These tests read such a frame back and hold it to the frame the same
//! renderer reads back for [`FrameOutput::Host`], **bit for bit**: the same cells, zooms, the Before
//! view, a frame without output sharpening, a tilted crop with paper in its corners, a move of the
//! Output slider (the tail over the resident display-linear frame), a view cut into bands (gathered
//! on the device, paper bands included) and a magnified view (made on the CPU, so a host frame
//! whatever was asked). They also check that the frame counts against the device's pool while it is
//! held and goes back when it is dropped, and that the frame cache serves a device frame again.
//!
//! On Mesa lavapipe in CI (`gpu-lavapipe`, `YM_REQUIRE_GPU=1`) and on Metal (`mac-gpu.yml` runs every
//! test of `ym-core`). Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1`.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FrameOutput, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{Crop, CropAspect, DevelopSettings, NormRect};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

const MODEL: &str = "ILCE-7RM4";

fn gpu() -> Option<Arc<Gpu>> {
    // Poison mode: every pool buffer starts as NaN, so a pixel of the device frame that no kernel
    // wrote shows up as a difference instead of passing by luck.
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
                "\n* GPU TESTS SKIPPED in device_frames: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// Every stage on the device, as the app runs under `auto` on a Mac; `band_bytes` cuts views into
/// bands of that many bytes (the default when `None`).
fn renderer(gpu: Option<&Arc<Gpu>>, band_bytes: Option<u64>) -> Renderer {
    let defaults = RenderOptions::default();
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: gpu.cloned(),
            gpu_stages: if gpu.is_some() {
                GpuStages::ALL
            } else {
                GpuStages::NONE
            },
            full_tile_px: 512,
            view_band_bytes: band_bytes.unwrap_or(defaults.view_band_bytes),
            ..defaults
        },
    )
    .expect("render thread")
}

fn picture(w: u32, h: u32) -> Developed {
    let white = light("sony-ilce-7rm4", 4500.0);
    let scene = Scene::new(w, h, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
    Developed::new(&scene, MODEL)
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn with_amount(src: &PhotoSource, pct: f64) -> PhotoSource {
    edit(src, |s| s.output_sharpening = pct)
}

fn tilted(src: &PhotoSource) -> PhotoSource {
    edit(src, |s| {
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
    })
}

fn cell(w: f64, h: f64, scale: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, scale))
}

fn at(percent: f64, w: f64, h: f64, image: (usize, usize)) -> ViewportState {
    let mut st = cell(w, h, 1.0);
    st.set_mode(ViewportMode::Percent { value: percent }, image);
    st
}

fn before(mut st: ViewportState) -> ViewportState {
    st.before = true;
    st
}

/// A frame made without the caches (the frame cache, the display cache), so that each request runs
/// its own device jobs.
fn fresh(
    r: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
    output: FrameOutput,
) -> RenderedFrame {
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    r.viewport_frame_to(src, st, &options, output, &CancelToken::new())
        .expect("frame")
}

/// A frame through the caches.
fn cached(
    r: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
    output: FrameOutput,
) -> RenderedFrame {
    r.viewport_frame_to(
        src,
        st,
        &FrameOptions::default(),
        output,
        &CancelToken::new(),
    )
    .expect("frame")
}

/// The bytes in use in the device's pool.
fn in_use(gpu: &Gpu) -> u64 {
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        Ok(ctx.pool().stats().in_use_bytes)
    })
    .expect("pool stats")
}

/// The first pixel where two frames differ, for the failure message.
fn first_difference(a: &[u8], b: &[u8], width: usize) -> String {
    if a.len() != b.len() {
        return format!("{} bytes against {}", a.len(), b.len());
    }
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) => {
            let p = i / 4;
            format!(
                "first at pixel ({}, {}): {:?} against {:?}",
                p % width,
                p / width,
                &a[4 * p..4 * p + 4],
                &b[4 * p..4 * p + 4]
            )
        }
        None => "none".to_owned(),
    }
}

/// Holds a frame asked for on the device to the host frame of the same request: the same size, the
/// same report of where it ran, and the same bytes once read back. Returns whether the frame stayed
/// on the device.
fn assert_same_pixels(name: &str, host: &RenderedFrame, device: &RenderedFrame) -> bool {
    assert_eq!(
        (host.width, host.height),
        (device.width, device.height),
        "{name}"
    );
    assert!(
        host.on_device.is_none(),
        "{name}: a host request reads back"
    );
    assert_eq!(host.rgba.len(), 4 * host.width * host.height, "{name}");
    assert_eq!(host.report.device, device.report.device, "{name}");
    assert_eq!(host.sharp, device.sharp, "{name}");
    let kept = device.on_device.is_some();
    if kept {
        assert!(device.rgba.is_empty(), "{name}: nothing was read back");
        assert_eq!(
            device.bytes(),
            4 * device.width * device.height,
            "{name}: the device frame is the frame's bytes"
        );
    }
    let bytes = device.host_rgba().expect("read the device frame back");
    assert!(
        bytes.as_ref() == host.rgba.as_slice(),
        "{name}: the device frame is not the host frame ({})",
        first_difference(&bytes, &host.rgba, host.width)
    );
    assert!(
        !device
            .report
            .notes
            .iter()
            .any(|n| n.contains("graphics card")),
        "{name}: {:?}",
        device.report.notes
    );
    kept
}

#[test]
fn a_device_frame_has_the_bytes_of_the_frame_read_back() {
    let Some(gpu) = gpu() else { return };
    let photo = picture(900, 600);
    let src = photo.source();
    let r = renderer(Some(&gpu), None);
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let cases: Vec<(&str, PhotoSource, ViewportState, bool)> = vec![
        // (name, photo, state, whether the device makes the whole frame)
        ("fit", src.clone(), cell(576.0, 384.0, 1.0), true),
        ("odd cell", src.clone(), cell(577.0, 383.0, 1.0), true),
        ("retina", src.clone(), cell(300.0, 200.0, 2.0), true),
        ("tall cell", src.clone(), cell(300.0, 420.0, 1.0), true),
        ("50 %", src.clone(), at(50.0, 400.0, 300.0, image), true),
        ("100 %", src.clone(), at(100.0, 400.0, 300.0, image), true),
        ("before", src.clone(), before(cell(576.0, 384.0, 1.0)), true),
        (
            "before at 100 %",
            src.clone(),
            before(at(100.0, 400.0, 300.0, image)),
            true,
        ),
        (
            "no sharpening",
            with_amount(&src, 0.0),
            cell(576.0, 384.0, 1.0),
            true,
        ),
        (
            "200 % sharpening",
            with_amount(&src, 200.0),
            cell(576.0, 384.0, 1.0),
            true,
        ),
        ("tilted crop", tilted(&src), cell(576.0, 384.0, 1.0), true),
        // Above 100 % the stage runs on the 100 % picture and the CPU magnifies it: a host frame
        // whatever was asked, the same bytes.
        ("200 %", src.clone(), at(200.0, 400.0, 300.0, image), false),
    ];
    for (name, photo, st, on_device) in cases {
        let host = fresh(&r, &photo, &st, FrameOutput::Host);
        let device = fresh(&r, &photo, &st, FrameOutput::Device);
        let kept = assert_same_pixels(name, &host, &device);
        assert_eq!(kept, on_device, "{name}: {:?}", device.report);
        if on_device {
            assert_eq!(device.report.device, ComputeDevice::Gpu, "{name}");
        }
        eprintln!(
            "{name}: {} x {}, {}",
            device.width,
            device.height,
            if kept { "kept on the device" } else { "host" }
        );
    }
}

#[test]
fn a_view_cut_into_bands_is_gathered_on_the_device_with_the_same_bytes() {
    let Some(gpu) = gpu() else { return };
    let photo = picture(900, 600);
    let src = photo.source();
    // About 110 rows of a 576 px view with the stage's border (16 bytes a pixel).
    let r = renderer(Some(&gpu), Some(1_400_000));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let cases: Vec<(&str, PhotoSource, ViewportState)> = vec![
        ("fit", src.clone(), cell(576.0, 384.0, 1.0)),
        ("odd cell", src.clone(), cell(577.0, 383.0, 1.0)),
        ("plain", with_amount(&src, 0.0), cell(576.0, 384.0, 1.0)),
        ("before", src.clone(), before(cell(576.0, 384.0, 1.0))),
        ("100 %", src.clone(), at(100.0, 640.0, 480.0, image)),
        // A tall cell: the photo has paper above and below it, and the bands that are nothing but
        // paper are painted on the host and uploaded into the gathered frame.
        ("tall cell", src.clone(), cell(300.0, 1400.0, 1.0)),
        ("tilted crop, tall", tilted(&src), cell(300.0, 1400.0, 1.0)),
    ];
    for (name, photo, st) in cases {
        let submissions = gpu.counters().submissions;
        let host = fresh(&r, &photo, &st, FrameOutput::Host);
        let jobs = gpu.counters().submissions - submissions;
        assert!(jobs >= 2, "{name}: the view was cut into {jobs} jobs");
        let device = fresh(&r, &photo, &st, FrameOutput::Device);
        assert!(
            assert_same_pixels(name, &host, &device),
            "{name}: the bands were gathered on the device"
        );
    }
}

#[test]
fn a_move_of_the_output_slider_keeps_its_frame_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let photo = picture(900, 600);
    let src = photo.source();
    let st = cell(576.0, 384.0, 1.0);
    // Two renderers that see the same requests through their caches: the first frame makes the
    // display-linear frame resident, the moves of the slider run the tail over it.
    let (on_host, on_device) = (renderer(Some(&gpu), None), renderer(Some(&gpu), None));
    // (amount, what serves the frame): the first makes the display-linear frame, the next two run
    // the tail over it, no sharpening is a plain pass, and the first amount again is a frame the
    // cache kept.
    for (amount, served) in [
        (100.0, "made"),
        (40.0, "tail"),
        (160.0, "tail"),
        (0.0, "plain"),
        (100.0, "cache"),
    ] {
        let name = format!("slider at {amount} ({served})");
        let photo = with_amount(&src, amount);
        let host = cached(&on_host, &photo, &st, FrameOutput::Host);
        let device = cached(&on_device, &photo, &st, FrameOutput::Device);
        assert_eq!(host.report.runs, device.report.runs, "{name}");
        let runs = device.report.runs;
        match served {
            "tail" => assert_eq!(runs.display_hits, 1, "{name}: {runs:?}"),
            "cache" => assert_eq!(runs.frame_hits, 1, "{name}: {runs:?}"),
            _ => assert_eq!((runs.display_hits, runs.frame_hits), (0, 0), "{name}"),
        }
        assert!(assert_same_pixels(&name, &host, &device), "{name}");
    }
}

#[test]
fn the_frame_cache_serves_a_device_frame_again_and_keeps_host_and_device_frames_apart() {
    let Some(gpu) = gpu() else { return };
    let photo = picture(900, 600);
    let src = photo.source();
    let r = renderer(Some(&gpu), None);
    let st = cell(576.0, 384.0, 1.0);
    let first = cached(&r, &src, &st, FrameOutput::Device);
    let again = cached(&r, &src, &st, FrameOutput::Device);
    assert_eq!(again.report.runs.frame_hits, 1);
    assert!(first.on_device.is_some());
    assert_eq!(
        first.on_device, again.on_device,
        "the same buffer, not a copy"
    );
    // A host request for the same view is not served the device frame (nor the other way round).
    let host = cached(&r, &src, &st, FrameOutput::Host);
    assert_eq!(host.report.runs.frame_hits, 0);
    assert!(host.on_device.is_none());
    assert_same_pixels("cached", &host, &again);
    let host_again = cached(&r, &src, &st, FrameOutput::Host);
    assert_eq!(host_again.report.runs.frame_hits, 1);
    assert!(host_again.on_device.is_none());
}

#[test]
fn a_device_frame_counts_against_the_pool_while_it_is_held() {
    let Some(gpu) = gpu() else { return };
    let photo = picture(900, 600);
    let src = photo.source();
    let r = renderer(Some(&gpu), None);
    let st = cell(576.0, 384.0, 1.0);
    // The first frame leaves what the caches keep on the device (the raw group): the baseline.
    drop(fresh(&r, &src, &st, FrameOutput::Host));
    let baseline = in_use(&gpu);
    let frame = fresh(&r, &src, &st, FrameOutput::Device);
    let bytes = frame.on_device.as_ref().expect("on the device").byte_len();
    let held = in_use(&gpu);
    assert!(
        held >= baseline + bytes,
        "the frame's buffer is in use while it is held: {held} against {baseline} + {bytes}"
    );
    // A clone is the same buffer, not another.
    let clone = frame.clone();
    assert_eq!(in_use(&gpu), held);
    drop(frame);
    assert_eq!(in_use(&gpu), held, "the clone still holds it");
    drop(clone);
    assert_eq!(
        in_use(&gpu),
        baseline,
        "the buffer went back to the pool with the last frame"
    );
}

#[test]
fn without_a_device_a_request_for_a_device_frame_gets_the_host_frame() {
    // No graphics card needed: the CPU makes the frame, whatever was asked.
    let photo = picture(300, 200);
    let src = photo.source();
    let r = renderer(None, None);
    let st = cell(192.0, 128.0, 1.0);
    let host = fresh(&r, &src, &st, FrameOutput::Host);
    let device = fresh(&r, &src, &st, FrameOutput::Device);
    assert_eq!(device.report.device, ComputeDevice::Cpu);
    assert!(device.on_device.is_none());
    assert!(host.rgba == device.rgba);
    assert_eq!(device.bytes(), 4 * 192 * 128);
}
