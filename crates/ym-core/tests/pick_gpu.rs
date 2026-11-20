//! Pick colour of a Colour Uniform node on the device (CU-fix): the pick asks for a 7 x 7 square (`radius_px` 3), which
//! `Renderer::sample_chroma_square` reads through `point_readings` as a 7 x 7 view. On the owner's Mac the pick does
//! nothing and shows no error, while the Colour range and Colour warp picks (1 x 1) work. Linux (CPU, lavapipe) reads
//! the square fine, so these tests are meant to run on a Metal runner (`mac-gpu.yml`) and print what the device gave.
//!
//! * Every reading inside the photo must be `Some`, for the radius 0 and the radius 3, in a fit view, a zoomed-in view
//!   and at the 2 x pixel ratio of a Retina screen.
//! * The device's `a`, `b` and `tone` match the CPU renderer's within `CHROMA_TOL` / `TONE_TOL`.
//! * Every reading, device and CPU, is printed; a `None` prints the zoom, the step and the side the view had.
//!
//! `ViewSpec` and its `sampled.outside` marker are `pub(crate)`, so the `outside` flag itself cannot be read from a
//! test: the `None` line gives the zoom, `step` and `side` instead (`uniform_pick_view_outside_flag` is skipped).
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::json;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ChromaSample, ChromaTap, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, NodeId};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PAPER: [u8; 3] = [40, 60, 90];
/// The cell in CSS pixels: a square one, so the 3:2 photo has paper above and below (y 50 to 250 in a fit view).
const CELL: f64 = 300.0;
/// The Colour Uniform node.
const NODE: u32 = 2;
/// Device against CPU: the chroma plane's a, b (about 0.1 across a photo) and the tone (LogC4 codes, 0 to 1).
const CHROMA_TOL: f64 = 0.01;
const TONE_TOL: f64 = 0.01;

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

/// `Input -> 1 -> 2 (Colour Uniform, no target) -> Output`.
fn uniform_settings() -> DevelopSettings {
    let link = |from: serde_json::Value, to: serde_json::Value| json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } });
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": {
        "items": [{ "id": 1 }, { "id": NODE, "kind": "color_uniform", "pos": [2.0, 0.0] }],
        "links": [
            link(json!("input"), json!(1)),
            link(json!(1), json!(NODE)),
            link(json!(NODE), json!("output")),
        ],
    } }))
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn renderer(gpu: Option<&Arc<Gpu>>) -> Renderer {
    let options = RenderOptions {
        gpu: gpu.cloned(),
        gpu_stages: if gpu.is_some() {
            GpuStages::ALL
        } else {
            GpuStages::NONE
        },
        background: PAPER,
        ..RenderOptions::default()
    };
    Renderer::new(Arc::new(NeutralOnly), options).expect("render thread")
}

fn gpu() -> Option<Arc<Gpu>> {
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
                "\n* GPU TESTS SKIPPED in pick_gpu: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// A view of the cell: its name, the state and the points to read (CSS pixels, inside the photo and away from its edges).
struct Case {
    name: &'static str,
    state: ViewportState,
    points: Vec<(f64, f64)>,
}

fn grid(xs: [f64; 3], ys: [f64; 3]) -> Vec<(f64, f64)> {
    ys.iter()
        .flat_map(|&y| xs.iter().map(move |&x| (x, y)))
        .collect()
}

fn cases() -> Vec<Case> {
    let state = |scale: f64, mode: ViewportMode| {
        let mut s = ViewportState::new(ViewportRect::cell(CELL, CELL, scale));
        s.mode = mode;
        s
    };
    // In the fit view the photo is 300 x 200 at y 50 to 250; a zoomed view fills the cell with the photo.
    let fit_points = grid([60.0, 150.0, 240.0], [90.0, 150.0, 210.0]);
    let zoom_points = grid([60.0, 150.0, 240.0], [60.0, 150.0, 240.0]);
    vec![
        Case {
            name: "fit, scale 1",
            state: state(1.0, ViewportMode::Fit),
            points: fit_points.clone(),
        },
        Case {
            name: "zoom 400 %, scale 1",
            state: state(1.0, ViewportMode::Percent { value: 400.0 }),
            points: zoom_points.clone(),
        },
        Case {
            name: "fit, Retina scale 2",
            state: state(2.0, ViewportMode::Fit),
            points: fit_points,
        },
        Case {
            name: "zoom 200 %, Retina scale 2",
            state: state(2.0, ViewportMode::Percent { value: 200.0 }),
            points: zoom_points,
        },
    ]
}

type Reading = Option<ChromaSample>;

fn read(
    r: &Renderer,
    src: &PhotoSource,
    radius: usize,
    state: &ViewportState,
    at: (f64, f64),
) -> Reading {
    r.sample_chroma_square(
        src,
        Some(NodeId::Corrector(NODE)),
        ChromaTap::NodeInput,
        radius,
        state,
        at,
        &CancelToken::new(),
    )
    .expect("a render without an error")
}

fn show(r: Reading) -> String {
    match r {
        Some(c) => format!("({:+.5}, {:+.5}) tone {:.5}", c.a, c.b, c.tone),
        None => "None".to_string(),
    }
}

/// The zoom, the step and the side `point_readings` used for a read that came back `None` (`step = 1 / zoom`, 1 for r = 0).
fn view_of(src: &PhotoSource, state: &ViewportState, radius: usize) -> String {
    let side = 2 * radius + 1;
    // The final image is the photo's auto-cropped size; the scene is 480 x 320, so the zoom is read from the state.
    let image = (src_size(src).0, src_size(src).1);
    let zoom = state.zoom(image);
    let step = if radius == 0 { 1.0 } else { 1.0 / zoom };
    format!(
        "zoom {zoom:.4}, step {step:.4}, side {side} (image {image:?}, outside flag not reachable from a test)"
    )
}

fn src_size(src: &PhotoSource) -> (usize, usize) {
    // The sensor size of the scene; a lens crop or an auto crop would change it by a few pixels at most.
    let _ = src;
    (480, 320)
}

#[test]
fn uniform_pick_readings_on_the_device_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let (device, cpu) = (renderer(Some(&g)), renderer(None));
    let d = photo();
    let src = d.source().with_settings(uniform_settings());
    let (mut missing, mut off, mut total) = (Vec::new(), Vec::new(), 0usize);
    for case in cases() {
        for radius in [0usize, 3] {
            for &at in &case.points {
                let (on_device, on_cpu) = (
                    read(&device, &src, radius, &case.state, at),
                    read(&cpu, &src, radius, &case.state, at),
                );
                total += 1;
                eprintln!(
                    "{} r={radius} at ({:.0}, {:.0}): device {} | cpu {}",
                    case.name,
                    at.0,
                    at.1,
                    show(on_device),
                    show(on_cpu)
                );
                let label = format!("{} r={radius} at {at:?}", case.name);
                if on_device.is_none() {
                    eprintln!(
                        "  NONE on the device: {label}: {}",
                        view_of(&src, &case.state, radius)
                    );
                    missing.push(label.clone());
                }
                if on_cpu.is_none() {
                    eprintln!(
                        "  NONE on the CPU: {label}: {}",
                        view_of(&src, &case.state, radius)
                    );
                }
                if let (Some(a), Some(b)) = (on_device, on_cpu) {
                    let (da, db, dt) = (
                        (a.a - b.a).abs(),
                        (a.b - b.b).abs(),
                        (a.tone - b.tone).abs(),
                    );
                    if da > CHROMA_TOL || db > CHROMA_TOL || dt > TONE_TOL {
                        off.push(format!("{label}: da {da:.5} db {db:.5} dtone {dt:.5}"));
                    }
                }
            }
        }
    }
    eprintln!(
        "uniform_pick: {total} reads, {} None on the device, {} off the CPU",
        missing.len(),
        off.len()
    );
    assert!(
        missing.is_empty(),
        "the device read None inside the photo: {missing:#?}"
    );
    assert!(
        off.is_empty(),
        "the device differs from the CPU beyond {CHROMA_TOL} / {TONE_TOL}: {off:#?}"
    );
}

#[test]
fn uniform_pick_square_of_three_is_the_mean_near_the_single_pixel() {
    // A sanity check on the device alone: in a flat patch the 7 x 7 square and the 1 x 1 pixel agree closely, so a
    // square that read the wrong place (a flipped or shifted view) would show up as a large gap.
    let Some(g) = gpu() else { return };
    let device = renderer(Some(&g));
    let d = photo();
    let src = d.source().with_settings(uniform_settings());
    let case = &cases()[0];
    // The middle of a 50 px patch of the fit view (patches start at x 0 and y 50, 6 x 4 of 50 px).
    for (x, y) in [(25.5, 75.5), (125.5, 125.5), (225.5, 175.5)] {
        let one = read(&device, &src, 0, &case.state, (x, y));
        let seven = read(&device, &src, 3, &case.state, (x, y));
        eprintln!("patch at ({x}, {y}): r0 {} | r3 {}", show(one), show(seven));
        let (one, seven) = (
            one.expect("r = 0 reads on the device"),
            seven.expect("r = 3 reads on the device"),
        );
        assert!(
            (one.a - seven.a).abs() < 0.03 && (one.b - seven.b).abs() < 0.03,
            "r0 and r3 disagree in a flat patch at ({x}, {y})"
        );
    }
}
