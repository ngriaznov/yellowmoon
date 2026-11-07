//! Pick colour of a Colour range (`docs/23-nodes.md` sections 4.2, 7.6 and 9, task N-12 follow-up): a click
//! on the photo sets the row's centre to the chroma of the clicked pixel **at the node's input**
//! (`ChromaTap::NodeInput`), the colour the range selects on, so that in a Mask only frame `m` at the clicked
//! pixel is near 1 whatever the node's own trim and grade do.
//!
//! * The node under test has a strong trim and a strong grade (warmer, tinted, more saturated, more
//!   contrast): its warp's input (`ChromaTap::WarpInput`, the colours its pins take hold of) is far from its
//!   input in the chroma plane, and a centre picked there selects nothing at the pixel that was clicked.
//! * The same on the CPU and on the device (Mesa lavapipe in CI): the chroma is read from the CPU chain either
//!   way, and `m` is drawn by the renderer under test.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ChromaSample, ChromaTap, ComputeDevice, FrameOptions, MaskStyle, MaskView, NeutralOnly,
    PhotoSource, RenderOptions, RenderedFrame, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, NodeId};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
/// The paper: nothing in the scene or in a mask is this colour.
const PAPER: [u8; 3] = [40, 60, 90];
/// The cell: a square one, so that the 3:2 photo (300 x 200 in it) has paper above and below.
const CELL: f64 = 300.0;
/// The node under test.
const NODE: u32 = 2;
/// The width and softness of the row: a centre that is 0.02 off the pixel's chroma is already at the edge, and one that
/// is 0.04 off selects nothing (the ramp is at least `RANGE_RAMP_MIN`, 0.02).
const WIDTH: f64 = 0.02;
const SOFTNESS: f64 = 0.0;

/// 6 x 4 patches of 50 px in the cell; the photo's rows start at y = 50. Patch `k = row * 6 + column`: the
/// coloured ones are those with `k % 5 == 3` (3, 8, 13, 18, 23), the others are neutral at different levels.
fn patch_centre(k: usize) -> (f64, f64) {
    let (column, row) = (k % 6, k / 6);
    (50.0 * column as f64 + 25.5, 50.0 + 50.0 * row as f64 + 25.5)
}

const COLOURED: [usize; 5] = [3, 8, 13, 18, 23];
/// The brightest neutral patches (the dark ones are noisy in chroma at this level, and a single pixel is read).
const NEUTRAL: [usize; 3] = [10, 16, 17];

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn link(from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } })
}

/// The node's fields: a strong trim and a strong grade, so that its warp's input is far from its input.
fn busy() -> Value {
    json!({
        "trim": { "exposure_ev": 0.6, "temperature": -100.0, "tint": -100.0 },
        "grade": {
            "contrast": { "kind": "manual", "value": 40.0 },
            "saturation": { "kind": "manual", "value": 100.0 },
            "highlights": { "kind": "manual", "value": 0.0 },
            "shadows": { "kind": "manual", "value": 0.0 },
            "black": { "kind": "manual", "value": 0.0 },
            "color_balance": {
                "shadows": { "kind": "manual", "point": [0.0, 0.0] },
                "highlights": { "kind": "manual", "point": [0.0, 0.0] }
            }
        }
    })
}

fn colour_range(center: [f64; 2]) -> Value {
    json!({
        "kind": "color_range", "mode": "add", "center": center,
        "width": WIDTH, "softness": SOFTNESS
    })
}

/// `Input → 1 → 2 → Output`: node 2 has `body` and, when `center` is given, one Colour range row there.
fn graph(body: &Value, center: Option<[f64; 2]>) -> DevelopSettings {
    let mut node = json!({ "id": NODE, "kind": "corrector", "pos": [1.0, 0.0] });
    if let (Some(o), Some(b)) = (node.as_object_mut(), body.as_object()) {
        o.extend(b.clone());
    }
    if let Some(center) = center {
        node["masks"] = json!({ "invert": false, "items": [colour_range(center)] });
    }
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": {
        "items": [{ "id": 1 }, node],
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

fn options(gpu: Option<&Arc<Gpu>>) -> RenderOptions {
    RenderOptions {
        gpu: gpu.cloned(),
        gpu_stages: if gpu.is_some() {
            GpuStages::ALL
        } else {
            GpuStages::NONE
        },
        background: PAPER,
        ..RenderOptions::default()
    }
}

fn renderer(gpu: Option<&Arc<Gpu>>) -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options(gpu)).expect("render thread")
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
                "\n* GPU TESTS SKIPPED in range_pick: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn cell() -> ViewportState {
    ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0))
}

fn mask_only() -> ViewportState {
    let mut st = cell();
    st.mask_view = Some(MaskView {
        node: NODE,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
    });
    st
}

/// `m` of a Mask only frame at a cell point (the grey code over 255).
fn m_at(f: &RenderedFrame, (x, y): (f64, f64)) -> f64 {
    let i = (y as usize * f.width + x as usize) * 4;
    let [r, g, b] = [f.rgba[i], f.rgba[i + 1], f.rgba[i + 2]];
    assert!(
        r == g && g == b,
        "the mask is grey at {x}, {y}: {r} {g} {b}"
    );
    f64::from(r) / 255.0
}

/// What a click at `point` reads at `tap` of node 2, and the Mask only frame of the photo with a Colour range
/// row whose centre is that.
struct Picked {
    chroma: ChromaSample,
    frame: RenderedFrame,
}

/// The click lands on `clicked` (the photo as it is when the person clicks) and the row is then set to what was
/// read, on a copy of the photo with the row and no other change.
fn pick(r: &Renderer, clicked: &PhotoSource, point: (f64, f64), tap: ChromaTap) -> Picked {
    let chroma = r
        .sample_chroma_at(
            clicked,
            Some(NodeId::Corrector(NODE)),
            tap,
            &cell(),
            point,
            &CancelToken::new(),
        )
        .expect("a sample")
        .expect("inside the photo");
    let picked = clicked.with_settings(graph(&busy(), Some([chroma.a, chroma.b])));
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let frame = r
        .viewport_frame(&picked, &mask_only(), &options, &CancelToken::new())
        .expect("a frame");
    Picked { chroma, frame }
}

/// The photo with node 2 busy and a new Colour range row at the middle of the plane: what a click starts from.
fn source(d: &Developed) -> PhotoSource {
    d.source().with_settings(graph(&busy(), Some([0.0, 0.0])))
}

/// The same without the row: nothing blends node 2's trim and grade into the picture it makes.
fn source_without_the_row(d: &Developed) -> PhotoSource {
    d.source().with_settings(graph(&busy(), None))
}

/// The claim: with the node's input as the place to read, the picked centre selects the clicked pixel, and the warp's
/// input (the reading of the Colour warp's Pick colour) does not, on a node with a strong trim and grade.
fn the_picked_centre_selects_the_clicked_pixel(r: &Renderer, what: &str) {
    let d = photo();
    let (src, bare) = (source(&d), source_without_the_row(&d));
    let (mut least_input, mut most_warp, mut apart) = (1.0f64, 0.0f64, 0.0f64);
    for k in COLOURED.into_iter().chain(NEUTRAL) {
        let point = patch_centre(k);
        // The click as the page makes it: the row is there, at the middle of the plane.
        let at_input = pick(r, &src, point, ChromaTap::NodeInput);
        let m_input = m_at(&at_input.frame, point);
        // Today's reading, for the Colour warp's Pick colour: after the node's trim and grade. Without a row in the way
        // it is the plain C9 of the node; with one, the row's own mask blends the node's effect into it (where the new
        // row at the middle of the plane selects, the effect is in the reading; where it does not, it is not).
        let at_warp = pick(r, &bare, point, ChromaTap::WarpInput);
        let m_warp = m_at(&at_warp.frame, point);
        let with_row = pick(r, &src, point, ChromaTap::WarpInput);
        let m_with_row = m_at(&with_row.frame, point);
        let (da, db) = (
            at_input.chroma.a - at_warp.chroma.a,
            at_input.chroma.b - at_warp.chroma.b,
        );
        eprintln!(
            "{what}: patch {k:2}: node input ({:+.4}, {:+.4}) m {m_input:.3}; warp input ({:+.4}, {:+.4}) m {m_warp:.3}, with the row in the way m {m_with_row:.3}",
            at_input.chroma.a, at_input.chroma.b, at_warp.chroma.a, at_warp.chroma.b
        );
        least_input = least_input.min(m_input);
        most_warp = most_warp.max(m_warp);
        apart = apart.max(da.hypot(db));
        assert!(
            m_input > 0.95,
            "{what}: patch {k}: the centre picked at the node's input selects the pixel (m = {m_input})"
        );
        assert!(
            m_warp < 0.2,
            "{what}: patch {k}: the centre picked at the warp's input misses it (m = {m_warp}), the two readings being {:.3} apart",
            da.hypot(db)
        );
        // In the page's own flow the new row sits at the middle of the plane: every colour near neutral is inside
        // it, the node's effect is in the reading, and the centre misses the pixel that was clicked.
        if NEUTRAL.contains(&k) {
            assert!(
                m_with_row < 0.2,
                "{what}: patch {k}: the warp's input with the new row in the way misses a neutral pixel (m = {m_with_row})"
            );
        }
        // The row selects the clicked colour and not everything: a patch of the other kind is out.
        let other = patch_centre(if COLOURED.contains(&k) { 10 } else { 13 });
        let m_other = m_at(&at_input.frame, other);
        assert!(
            m_other < 0.5,
            "{what}: patch {k}: the row keeps out a patch of another colour (m = {m_other})"
        );
    }
    eprintln!(
        "{what}: m at the clicked pixel: node input at least {least_input:.3}; warp input at most {most_warp:.3} (the two readings up to {apart:.3} apart in the chroma plane)"
    );
    assert!(
        apart > 3.0 * WIDTH,
        "the node moves the colour well outside the row's width ({apart})"
    );
}

#[test]
fn on_the_cpu_a_picked_centre_selects_the_clicked_pixel() {
    the_picked_centre_selects_the_clicked_pixel(&renderer(None), "CPU");
}

#[test]
fn on_the_device_a_picked_centre_selects_the_clicked_pixel() {
    let Some(gpu) = gpu() else { return };
    let r = renderer(Some(&gpu));
    // The frame is the device's.
    let d = photo();
    let src = source(&d);
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let frame = r
        .viewport_frame(&src, &mask_only(), &options, &CancelToken::new())
        .expect("a frame");
    assert_eq!(frame.report.device, ComputeDevice::Gpu);
    the_picked_centre_selects_the_clicked_pixel(&r, "device");
}

#[test]
fn the_cpu_and_the_device_pick_the_same_centre_and_draw_the_same_mask() {
    let Some(gpu) = gpu() else { return };
    let (on_gpu, cpu) = (renderer(Some(&gpu)), renderer(None));
    let d = photo();
    let src = source(&d);
    let mut worst = 0u8;
    for k in COLOURED {
        let point = patch_centre(k);
        let (a, b) = (
            pick(&cpu, &src, point, ChromaTap::NodeInput),
            pick(&on_gpu, &src, point, ChromaTap::NodeInput),
        );
        // The reading is made on the CPU chain from the same camera pixel; the sampler's colour differs by the
        // device's f32 against the CPU's f64.
        assert!(
            (a.chroma.a - b.chroma.a).abs() < 1e-6 && (a.chroma.b - b.chroma.b).abs() < 1e-6,
            "patch {k}: {:?} against {:?}",
            a.chroma,
            b.chroma
        );
        assert_eq!(b.frame.report.device, ComputeDevice::Gpu);
        for (x, y) in a.frame.rgba.iter().zip(&b.frame.rgba) {
            worst = worst.max(x.abs_diff(*y));
        }
    }
    eprintln!("the mask of a picked centre, device against CPU: at most {worst} codes");
    assert!(worst <= 1, "{worst} codes");
}
