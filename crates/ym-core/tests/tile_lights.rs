//! Tiles of an export equal the whole frame bit for bit on a scene of bright, hard-edged lights (`docs/17-gpu.md` D4
//! and D5), with the default settings and no mask. The lights are discs at 0.45 of the sensor's range on a dark ground.
mod common;

use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions,
    Renderer,
};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// Discs of 1.5 % of the width, `cols` x `rows` of them, at `level` of the sensor's range, on a dark ground.
fn lights(w: u32, h: u32, cols: usize, rows: usize, level: f64) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.02).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + level * (scene.white_level - scene.black)) as u16;
    let radius = 0.015 * f64::from(w);
    for r in 0..rows {
        for c in 0..cols {
            let fx = (c as f64 + 0.5) / cols as f64;
            let fy = 0.1 + 0.8 * (r as f64 + 0.5) / rows as f64;
            let (cx, cy) = (fx * f64::from(w), fy * f64::from(h));
            for y in 0..h {
                for x in 0..w {
                    let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                    if d <= radius {
                        mosaic[(y * w + x) as usize] = code;
                    }
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn renderer(gpu: Option<Arc<Gpu>>, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            data_root: Some(data_root()),
            full_tile_px: tile,
            gpu,
            gpu_stages: GpuStages::ALL,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn export(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

/// The pixels where two images differ, with the largest difference.
fn differences(a: &FullImage, b: &FullImage) -> (usize, f32, Vec<(usize, usize, usize)>) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut n = 0;
    let mut worst = 0.0f32;
    let mut at = Vec::new();
    for (i, (x, y)) in a.rgb.iter().zip(&b.rgb).enumerate() {
        if x.to_bits() != y.to_bits() {
            n += 1;
            worst = worst.max((x - y).abs());
            if at.len() < 12 {
                let p = i / 3;
                at.push((p % a.width, p / a.width, i % 3));
            }
        }
    }
    (n, worst, at)
}

/// The raw stages of the export ran on the device (the colour stage of a display export is on the host).
fn on_the_device(image: &FullImage) {
    let ranges = &image.report.ranges;
    assert!(
        !ranges.is_empty() && ranges.iter().all(|r| r.device == ComputeDevice::Gpu),
        "the raw stages did not run on the device: {ranges:?} {:?}",
        image.report.notes
    );
}

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            eprintln!("\n* GPU TESTS SKIPPED in tile_lights: {e}\n");
            None
        }
    }
}

/// Tiles of 100, 128 and 256 px against the whole frame on the CPU, for lights at 0.45 of the range and for lights
/// that clip. The default settings, no mask. The colour guided filter of R7 (the gate of `false_color`) used box means
/// with running sums, whose rounding depends on where a row or a column begins: a tile differed from the whole frame
/// in 8 pixels at the edge of a light, by about 1e-8.
#[test]
fn cpu_tiles_equal_the_whole_frame_on_bright_lights() {
    for level in [0.45, 1.0] {
        let d = lights(640, 480, 5, 4, level);
        let src = d.source();
        let whole = export(&renderer(None, 4096), &src);
        for tile in [100, 128, 256] {
            let tiled = export(&renderer(None, tile), &src);
            let (n, worst, at) = differences(&whole, &tiled);
            eprintln!(
                "[tile {tile}, lights at {level}] {n} values differ, worst {worst:e}, first at {at:?}"
            );
            assert_eq!(
                n, 0,
                "tiles of {tile} px differ from the whole frame (lights at {level})"
            );
        }
    }
}

/// The same on the device (Mesa lavapipe in CI), with the raw stages R1 to R9 on it.
#[test]
fn device_tiles_equal_the_whole_frame_on_bright_lights() {
    let Some(gpu) = gpu() else { return };
    let d = lights(640, 480, 5, 4, 0.45);
    let src = d.source();
    let whole = export(&renderer(Some(Arc::clone(&gpu)), 4096), &src);
    on_the_device(&whole);
    for tile in [100, 128, 256] {
        let tiled = export(&renderer(Some(Arc::clone(&gpu)), tile), &src);
        on_the_device(&tiled);
        let (n, worst, at) = differences(&whole, &tiled);
        eprintln!("[device tile {tile}] {n} values differ, worst {worst:e}, first at {at:?}");
        assert_eq!(
            n, 0,
            "device tiles of {tile} px differ from the whole frame"
        );
    }
}
