//! The Film node on the device (`docs/24-film.md` section 3.9c, `docs/17-gpu.md` sections 9 and 11a; Mesa lavapipe in CI).
//!
//! A graph with a Film node, a mask, halation and grain is made on the device for the viewport, the export tail and the
//! DNG's colour stage, and is held to the CPU reference of `render_film.rs`:
//!
//! * **Device against CPU.** The viewport frame, the export (PNG, 16 bit) and the DNG's colour stage agree with the CPU's
//!   within the bound of the other device-against-CPU tests (the picture's colour: one 10-bit code after C13, so 64 codes of
//!   a 16-bit file; an 8-bit frame: one code), with two Film nodes in a line and a corrector after them as well.
//! * **Tiles and determinism.** The export is the same bytes for any tile size, twice, and with the pool poisoned (every
//!   device buffer starts as NaN) or not.
//! * **The device route is the device's.** The frame and the export say `device = gpu` and carry no "graphics card" note.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{
    ExportContext, ExportControl, ExportParams, ExportPhoto, ExportReport, export_photo,
};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource,
    RenderOptions, RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn gpu(poison: bool) -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        poison,
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
                "\n* GPU TESTS SKIPPED in render_film_device: {e}\n"
            );
            None
        }
    }
}

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn renderer(gpu: Option<Arc<Gpu>>, tile: usize) -> Renderer {
    let stages = if gpu.is_some() {
        GpuStages::ALL
    } else {
        GpuStages::NONE
    };
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: stages,
            full_tile_px: tile,
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn settings_of(doc: Value) -> DevelopSettings {
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// Input, Main, a Film node with these fields, Output.
fn one_film(film: Value) -> DevelopSettings {
    let mut node = json!({ "id": 2, "kind": "film", "pos": [1.0, 0.0] });
    node.as_object_mut()
        .unwrap()
        .extend(film.as_object().unwrap().clone());
    settings_of(json!({ "nodes": { "items": [ { "id": 1 }, node ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } }))
}

/// Halation, grain, a Linear mask and Strength below 100.
fn busy() -> Value {
    json!({
        "grain": { "amount": 150, "size": 100, "seed": 7 },
        "halation": { "amount": 200, "size": 120 },
        "strength": 80,
        "masks": { "items": [
            { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 }
        ] },
    })
}

/// Two Film nodes in a line, then a corrector.
fn two_films() -> DevelopSettings {
    settings_of(json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "film", "pos": [1, 0], "film": { "stock": "kodak_portra_400" },
          "grain": { "amount": 120, "seed": 2 }, "halation": { "amount": 250 } },
        { "id": 3, "kind": "film", "pos": [2, 0], "film": { "stock": "fujifilm_velvia_100" },
          "grain": { "amount": 80, "seed": 3 }, "strength": 60,
          "masks": { "items": [ { "kind": "radial", "center": [0.4, 0.5], "radii": [0.3, 0.2], "feather": 0.5 } ] } },
        { "id": 4, "kind": "corrector", "pos": [3, 0], "trim": { "exposure_ev": 0.4 } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": 4, "port": "picture" } },
        { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } }))
}

fn view(zoom_100: bool) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(
        src,
        st,
        &FrameOptions {
            no_cache: true,
            ..FrameOptions::default()
        },
        &CancelToken::new(),
    )
    .expect("frame")
}

/// The largest difference of two 8-bit frames, and the pixels more than one code apart.
fn frame_diff(a: &RenderedFrame, b: &RenderedFrame) -> (u8, usize, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut max, mut over, mut sum) = (0u8, 0usize, 0u64);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        over += usize::from(d > 1);
        sum += u64::from(d);
    }
    (max, over, sum as f64 / a.rgba.len() as f64)
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[film on the device] {text}");
}

#[test]
fn the_viewport_frame_is_the_devices_and_agrees_with_the_cpu() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let device = renderer(Some(gpu), 256);
    let cpu = renderer(None, 256);
    for (what, settings) in [
        ("one node", one_film(busy())),
        ("two nodes and a corrector", two_films()),
    ] {
        let src = d.source().with_settings(settings);
        for zoom_100 in [false, true] {
            let st = view(zoom_100);
            let got = frame(&device, &src, &st);
            let want = frame(&cpu, &src, &st);
            let at = format!("{what}, 100 % {zoom_100}");
            assert_eq!(got.report.device, ComputeDevice::Gpu, "{at}");
            assert!(
                !got.report.notes.iter().any(|n| n.contains("graphics card")),
                "{at}: {:?}",
                got.report.notes
            );
            assert!(!got.report.device_refused, "{at}");
            assert_eq!(want.report.device, ComputeDevice::Cpu, "{at}");
            let (max, over, mean) = frame_diff(&got, &want);
            say(&format!(
                "view, {at}: largest {max} codes, {over} pixel values over 1, mean {mean:.4}"
            ));
            assert!(max <= 2 && over < got.rgba.len() / 2000, "{at}: {max}, {over}");
            // The node is in the picture: the same photo without it is not this frame.
            let plain = frame(&device, &d.source(), &st);
            assert_ne!(plain.rgba, got.rgba, "{at}");
        }
    }
}

/// A PNG's samples, 16 bit.
fn decode_png16(path: &Path) -> (usize, usize, Vec<u32>) {
    let bytes = std::fs::read(path).expect("read the export");
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    let codes = buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
        .collect();
    (info.width as usize, info.height as usize, codes)
}

fn export(
    d: &Developed,
    settings: &DevelopSettings,
    r: Renderer,
    out: &Path,
) -> (ExportReport, Vec<u32>) {
    let ctx = ExportContext::new(r).with_home(Some(out.to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = photo.source.with_settings(settings.clone());
    let opts = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let report = export_photo(&ctx, &photo, &ExportParams::new(opts), &ExportControl::none())
        .expect("export");
    let (_, _, codes) = decode_png16(&report.path);
    (report, codes)
}

fn code_diff(a: &[u32], b: &[u32]) -> (u32, f64) {
    assert_eq!(a.len(), b.len());
    let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    let mean = a.iter().zip(b).map(|(x, y)| f64::from(x.abs_diff(*y))).sum::<f64>() / a.len() as f64;
    (max, mean)
}

/// One 10-bit code after C13, in the codes of a 16-bit file.
const TEN_BIT_CODE: u32 = 64;

#[test]
fn the_export_tail_runs_the_film_node_on_the_device_and_agrees_with_the_cpu() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    for (what, settings) in [
        ("one node", one_film(busy())),
        ("two nodes and a corrector", two_films()),
    ] {
        let home = tempfile::tempdir().expect("dir");
        let (dev_report, dev) = export(&d, &settings, renderer(Some(Arc::clone(&gpu)), 256), home.path());
        let (cpu_report, cpu) = export(&d, &settings, renderer(None, 256), home.path());
        assert_eq!(dev_report.device, ComputeDevice::Gpu, "{what}");
        assert_eq!(cpu_report.device, ComputeDevice::Cpu, "{what}");
        assert!(
            !dev_report.notes.iter().any(|n| n.contains("graphics card")),
            "{what}: {:?}",
            dev_report.notes
        );
        let (max, mean) = code_diff(&dev, &cpu);
        say(&format!(
            "export, {what}: largest {max} of 65535 (a 10-bit code is {TEN_BIT_CODE}), mean {mean:.2}"
        ));
        assert!(max <= TEN_BIT_CODE, "{what}: {max}");
    }
}

#[test]
fn tiles_poison_and_repeats_do_not_change_a_bit_on_the_device() {
    let Some(poisoned) = gpu(true) else { return };
    let Some(plain) = gpu(false) else { return };
    let d = photo();
    let settings = one_film(busy());
    let home = tempfile::tempdir().expect("dir");
    let (_, whole) = export(&d, &settings, renderer(Some(Arc::clone(&poisoned)), 1024), home.path());
    for tile in [97usize, 160, 256] {
        let (_, got) = export(&d, &settings, renderer(Some(Arc::clone(&poisoned)), tile), home.path());
        assert_eq!(got, whole, "tile {tile}");
    }
    let (_, again) = export(&d, &settings, renderer(Some(Arc::clone(&poisoned)), 1024), home.path());
    assert_eq!(again, whole, "a repeat");
    let (_, clean) = export(&d, &settings, renderer(Some(plain), 160), home.path());
    assert_eq!(clean, whole, "a device with a clean pool");
}

fn dng_image(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DngLinearAwg4,
        },
        &CancelToken::new(),
    )
    .expect("render")
}

#[test]
fn the_colour_stage_of_a_dng_with_a_film_node_runs_on_the_device_and_agrees_with_the_cpu() {
    let Some(gpu) = gpu(true) else { return };
    let d = photo();
    let src = d.source().with_settings(one_film(busy()));
    let device = dng_image(&renderer(Some(gpu), 256), &src);
    let cpu = dng_image(&renderer(None, 256), &src);
    assert_eq!(device.report.device, ComputeDevice::Gpu);
    assert_eq!(cpu.report.device, ComputeDevice::Cpu);
    let (mut max, mut sum) = (0.0f64, 0.0f64);
    for (a, b) in device.rgb.iter().zip(&cpu.rgb) {
        let diff = f64::from((a - b).abs()) / f64::from(b.abs()).max(0.05);
        max = max.max(diff);
        sum += diff;
    }
    say(&format!(
        "DNG linear AWG4: largest relative {max:.2e}, mean {:.2e}",
        sum / device.rgb.len() as f64
    ));
    assert!(max < 2e-2, "{max}");
    let _ = OutputTarget::Srgb;
}

// ---- the fallback (`docs/17-gpu.md` section 9) ---------------------------------------------------------------

/// `--features fault-injection` (without it these tests are empty).
#[cfg(feature = "fault-injection")]
mod fallback {
    use super::*;
    use ym_gpu::device::GpuFault;

    fn bytes(path: &Path) -> Vec<u8> {
        std::fs::read(path).expect("read")
    }

    fn png_options(out: &Path) -> ExportOptions {
        ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Sixteen,
            destination: Some(out.to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        }
    }

    /// A frame our own check refuses is the CPU's, with the note, and the next frame is the device's: a Film node is
    /// no different from the other kernels, and the viewport's pill says CPU for the frame that was refused
    /// (`RenderReport::device_refused`).
    #[test]
    fn a_refused_film_frame_costs_one_cpu_frame_and_leaves_the_route_open() {
        let Some(gpu) = gpu(true) else { return };
        let d = photo();
        let src = d.source().with_settings(one_film(busy()));
        let r = renderer(Some(Arc::clone(&gpu)), 256);
        let st = view(false);
        let device = frame(&r, &src, &st);
        assert_eq!(device.report.device, ComputeDevice::Gpu);
        gpu.set_fault(Some(GpuFault::FailInvalid(1)));
        let refused = frame(&r, &src, &st);
        gpu.set_fault(None);
        assert_eq!(refused.report.device, ComputeDevice::Cpu);
        assert!(refused.report.device_refused);
        assert!(
            refused
                .report
                .notes
                .iter()
                .any(|n| n.contains("invalid kernel input")),
            "{:?}",
            refused.report.notes
        );
        assert!(!r.raw_on_cpu(src.id), "the route is still open");
        let (max, ..) = frame_diff(&refused, &device);
        assert!(max <= 3, "{max} codes");
        let next = frame(&r, &src, &st);
        assert_eq!(next.report.device, ComputeDevice::Gpu);
        assert!(!next.report.device_refused);
    }

    /// A device error in the export makes the whole photo again on the CPU: the CPU export, bit for bit, with the note.
    #[test]
    fn a_device_error_on_a_film_export_makes_the_photo_again_on_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu(true) else { return };
        let d = photo();
        let settings = one_film(busy());
        let home = tempfile::tempdir().expect("dir");
        let cpu_dir = tempfile::tempdir().expect("dir");
        let ctx = ExportContext::new(renderer(None, 256)).with_home(Some(home.path().to_path_buf()));
        let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
        photo.source = photo.source.with_settings(settings);
        let cpu = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(png_options(cpu_dir.path())),
            &ExportControl::none(),
        )
        .expect("cpu export");
        for fault in [GpuFault::Shader, GpuFault::OutOfBudget, GpuFault::Lost] {
            // A device that is lost stays lost: each case gets a device of its own.
            let own = if matches!(fault, GpuFault::Lost) {
                Arc::new(Gpu::start(GpuConfig::default()).expect("device"))
            } else {
                Arc::clone(&gpu)
            };
            let ctx = ExportContext::new(renderer(Some(Arc::clone(&own)), 256))
                .with_home(Some(home.path().to_path_buf()));
            let out = tempfile::tempdir().expect("dir");
            own.set_fault(Some(fault));
            let done = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(png_options(out.path())),
                &ExportControl::none(),
            );
            own.set_fault(None);
            let done = done.unwrap_or_else(|e| panic!("{fault:?}: {e}"));
            assert_eq!(done.device, ComputeDevice::Cpu, "{fault:?}");
            assert!(
                done.notes.iter().any(|n| n
                    .contains("The graphics card could not render this photo, so the CPU did")),
                "{fault:?}: {:?}",
                done.notes
            );
            assert_eq!(
                bytes(&done.path),
                bytes(&cpu.path),
                "{fault:?}: the fallback is not the CPU export"
            );
        }
    }
}

/// Times of the device route on lavapipe (`cargo test --release -p ym-core --test render_film_device -- --ignored
/// --nocapture timings`), to set against the CPU's in `render_film.rs`: the first tables (baked and uploaded), a view of
/// the fit size, and a full-resolution export of 24 MP, with a Film node at its defaults (halation and grain on).
#[test]
#[ignore = "a measurement, not a check"]
fn timings() {
    use std::time::Instant;
    let Some(gpu) = gpu(false) else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(6000, 4000, white, Layout::Uniform, 0.12)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    let r = renderer(Some(gpu), 1024);
    let st = ViewportState::new(ViewportRect::cell(1600.0, 1066.0, 1.0));
    let timed = |s: &PhotoSource| {
        let t = Instant::now();
        let f = r
            .viewport_frame(
                s,
                &st,
                &FrameOptions {
                    no_cache: true,
                    ..FrameOptions::default()
                },
                &CancelToken::new(),
            )
            .expect("frame");
        (t.elapsed().as_secs_f64(), f.width, f.height, f.report.device)
    };
    let (plain, w, h, _) = timed(&src);
    let (plain2, ..) = timed(&src);
    let filmed = src.with_settings(one_film(json!({})));
    let (cold, _, _, dev) = timed(&filmed);
    let (warm_same, ..) = timed(&filmed);
    let (seed, ..) = timed(&src.with_settings(one_film(json!({ "grain": { "seed": 1 } }))));
    let (exposure, ..) = timed(&src.with_settings(one_film(json!({ "film": { "exposure_ev": 0.5 } }))));
    say(&format!(
        "view {w}x{h} ({dev:?}): without the node {plain:.2} s ({plain2:.2} s again), first with it (tables baked and uploaded) {cold:.2} s, again {warm_same:.2} s, new seed {seed:.2} s, new exposure {exposure:.2} s"
    ));
    for (what, settings) in [("with a Film node", one_film(json!({}))), ("without it", one_film(json!({ "bypass": true })))] {
        let home = tempfile::tempdir().expect("dir");
        let ctx = ExportContext::new(renderer(gpu_of(&r), 1024)).with_home(Some(home.path().to_path_buf()));
        let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
        photo.source = photo.source.with_settings(settings);
        let opts = ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Eight,
            destination: Some(home.path().to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        };
        let t = Instant::now();
        let report = export_photo(&ctx, &photo, &ExportParams::new(opts), &ExportControl::none())
            .expect("export");
        say(&format!(
            "export {}x{} ({:.1} MP) {what}: {:.1} s ({:?})",
            report.rendered.0,
            report.rendered.1,
            f64::from(report.rendered.0) * f64::from(report.rendered.1) / 1e6,
            t.elapsed().as_secs_f64(),
            report.device
        ));
    }
}

fn gpu_of(r: &Renderer) -> Option<Arc<Gpu>> {
    r.options().gpu.clone()
}
