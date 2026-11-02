//! The mask view of the viewport (`docs/23-nodes.md` section 5.8, task N-11): Show mask, rendered by the core.
//!
//! * **Mask only** is the frame's own plan compiled to the selected node's mask: the code value after C13 is
//!   `m`, grey, with the paper where the photo is not.
//! * **Overlay** is the photo as it is now with the node's `m` tinted in the mask accent after output
//!   sharpening and before C13: `out = display + 0.55 m (accent - display)` in display-linear light.
//! * A node that is not connected to the Output (a dead node) has a mask but no picture: Mask only shows its
//!   mask, Overlay tints the photo as it is.
//! * The mask view belongs to the viewport: the Before view, the pixel readout, an export and the caches that
//!   other views share never show it.
//!
//! Device and CPU agree within what the photo's own device and CPU frames do (`render_nodes.rs`); the mask
//! itself, which depends on the pixel's place and not on the photo, within one 8-bit code.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_color::encode::OutputSpace;
use ym_color::encode::srgb_decode;
use ym_color::gamut::LinearRec709;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly,
    PhotoSource, RenderOptions, RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
/// The paper: nothing in the scene or in a mask is this colour.
const PAPER: [u8; 3] = [40, 60, 90];
/// The cell: a square one, so that the 3:2 photo has paper above and below it.
const CELL: f64 = 300.0;
/// The photo's rows in the cell (200 of 300, centred).
const ROWS: std::ops::Range<usize> = 50..250;

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

/// A node-first document of these items and links.
fn doc(items: Vec<Value>, links: Vec<Value>) -> DevelopSettings {
    serde_json::from_value::<DevelopSettings>(
        json!({ "nodes": { "items": items, "links": links } }),
    )
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn link(from: Value, to: Value) -> Value {
    json!({ "from": { "node": from, "port": "picture" }, "to": { "node": to, "port": "picture" } })
}

/// Input, these ids in a line, Output.
fn line(ids: &[u32]) -> Vec<Value> {
    let mut links = Vec::new();
    let mut from = json!("input");
    for id in ids {
        links.push(link(from, json!(id)));
        from = json!(id);
    }
    links.push(link(from, json!("output")));
    links
}

fn radial(center: [f64; 2], radii: [f64; 2], feather: f64) -> Value {
    json!({ "kind": "radial", "mode": "add", "center": center, "radii": radii, "feather": feather })
}

fn node(id: u32, body: Value, rows: Vec<Value>) -> Value {
    let mut n = json!({ "id": id, "kind": "corrector", "pos": [f64::from(id), 0.0] });
    if let (Some(o), Some(b)) = (n.as_object_mut(), body.as_object()) {
        o.extend(b.clone());
    }
    if !rows.is_empty() {
        n["masks"] = json!({ "invert": false, "items": rows });
    }
    n
}

fn trim(ev: f64) -> Value {
    json!({ "trim": { "exposure_ev": ev } })
}

/// Main with a Radial of these radii at the middle of the photo.
fn main_radial() -> DevelopSettings {
    doc(
        vec![node(
            1,
            json!({}),
            vec![radial([0.5, 0.5], [0.3, 0.2], 0.4)],
        )],
        line(&[1]),
    )
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn options(gpu: Option<&Arc<Gpu>>, band_bytes: Option<u64>) -> RenderOptions {
    let defaults = RenderOptions::default();
    RenderOptions {
        gpu: gpu.cloned(),
        gpu_stages: if gpu.is_some() {
            GpuStages::ALL
        } else {
            GpuStages::NONE
        },
        background: PAPER,
        view_band_bytes: band_bytes.unwrap_or(defaults.view_band_bytes),
        ..defaults
    }
}

fn renderer(gpu: Option<&Arc<Gpu>>, band_bytes: Option<u64>) -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options(gpu, band_bytes)).expect("render thread")
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
                "\n* GPU TESTS SKIPPED in mask_view: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn view(style: Option<MaskStyle>, node: u32) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
    st.mask_view = style.map(|style| MaskView { node, style });
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

/// A frame made without the caches, so that the request runs its own passes.
fn fresh(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    r.viewport_frame(src, st, &options, &CancelToken::new())
        .expect("frame")
}

fn px(f: &RenderedFrame, x: usize, y: usize) -> [u8; 3] {
    let i = (y * f.width + x) * 4;
    [f.rgba[i], f.rgba[i + 1], f.rgba[i + 2]]
}

/// The largest and the mean difference between two frames, in 8-bit codes.
fn differ(a: &RenderedFrame, b: &RenderedFrame) -> (u8, f64) {
    assert_eq!(a.rgba.len(), b.rgba.len());
    let (mut worst, mut sum) = (0u8, 0u64);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        worst = worst.max(d);
        sum += u64::from(d);
    }
    (worst, sum as f64 / a.rgba.len() as f64)
}

fn is_paper(f: &RenderedFrame, x: usize, y: usize) -> bool {
    px(f, x, y) == PAPER
}

fn all_paper_above_and_below(f: &RenderedFrame) -> bool {
    (0..f.width).all(|x| {
        is_paper(f, x, 0) && is_paper(f, x, 20) && is_paper(f, x, 280) && is_paper(f, x, 299)
    })
}

/// `m` of a Mask only frame at a pixel (the grey code over 255).
fn m_at(mask_only: &RenderedFrame, x: usize, y: usize) -> f64 {
    let [r, g, b] = px(mask_only, x, y);
    assert!(
        r == g && g == b,
        "the mask is grey at ({x}, {y}): {r} {g} {b}"
    );
    f64::from(r) / 255.0
}

/// The spec's weight (`docs/23` section 5.8), written out again here.
const WEIGHT: f64 = 0.55;

/// The spec's accent, `#FF6FD2`, as the Display P3 linear colour that looks like it: the three sRGB codes
/// decoded to linear Rec.709 and converted to the viewport's primaries (written out again here).
fn accent() -> [f64; 3] {
    LinearRec709([255.0, 111.0, 210.0].map(|c| srgb_decode(c / 255.0)))
        .to_display_p3()
        .0
}

/// What an Overlay pixel should be, from the plain frame's code and the mask's, in codes.
fn expected_overlay(plain: [u8; 3], m: f64) -> [f64; 3] {
    let space = OutputSpace::DisplayP3;
    let accent = accent();
    let mut out = [0.0; 3];
    for c in 0..3 {
        let d = space.decode_channel(f64::from(plain[c]) / 255.0);
        let lit = d + WEIGHT * m * (accent[c] - d);
        out[c] = space.encode_channel(lit).clamp(0.0, 1.0) * 255.0;
    }
    out
}

/// The largest and the mean distance between an Overlay frame and the formula applied to the plain frame, over
/// the photo's pixels.
fn formula_error(
    plain: &RenderedFrame,
    mask_only: &RenderedFrame,
    overlay: &RenderedFrame,
) -> (f64, f64) {
    let (mut worst, mut sum, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in ROWS {
        for x in 0..plain.width {
            let want = expected_overlay(px(plain, x, y), m_at(mask_only, x, y));
            let got = px(overlay, x, y);
            for c in 0..3 {
                let d = (f64::from(got[c]) - want[c]).abs();
                worst = worst.max(d);
                sum += d;
                n += 1;
            }
        }
    }
    (worst, sum / n as f64)
}

// ---- Mask only -------------------------------------------------------------------------------------

#[test]
fn mask_only_is_the_mask_as_grey_code_values_with_the_paper_around_it() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    let r = renderer(None, None);
    let f = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    assert_eq!((f.width, f.height), (300, 300));
    assert!(all_paper_above_and_below(&f), "the paper stays paper");
    // The middle of the Radial is 1, its far side is 0, and the mask falls between.
    let centre = m_at(&f, 150, 150);
    let corner = m_at(&f, 3, 52);
    assert!(centre > 0.985, "the middle is {centre}");
    assert!(corner < 0.01, "the corner is {corner}");
    let along: Vec<f64> = [150, 190, 215, 235, 255, 285]
        .iter()
        .map(|&x| m_at(&f, x, 150))
        .collect();
    eprintln!("m along the row from the middle: {along:.3?}");
    for pair in along.windows(2) {
        assert!(pair[0] + 0.02 >= pair[1], "m falls outwards: {along:?}");
    }
    assert!(along[1] > 0.5 && along[5] < 0.05, "{along:?}");
}

#[test]
fn mask_only_does_not_depend_on_the_photo_or_on_a_grade() {
    // Grading a node never moves its own mask.
    let d = photo();
    let r = renderer(None, None);
    let st = view(Some(MaskStyle::MaskOnly), 1);
    let plain = frame(&r, &with(&d.source(), main_radial()), &st);
    let graded = doc(
        vec![node(
            1,
            json!({ "trim": { "exposure_ev": 2.0, "temperature": 40.0 } }),
            vec![radial([0.5, 0.5], [0.3, 0.2], 0.4)],
        )],
        line(&[1]),
    );
    let regraded = frame(&r, &with(&d.source(), graded), &st);
    assert_eq!(plain.rgba, regraded.rgba);
}

#[test]
fn a_node_without_a_mask_shows_one_everywhere() {
    let d = photo();
    let src = with(
        &d.source(),
        doc(vec![node(1, json!({}), vec![])], line(&[1])),
    );
    let r = renderer(None, None);
    let f = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    assert!(all_paper_above_and_below(&f));
    for y in ROWS {
        for x in 0..f.width {
            assert_eq!(px(&f, x, y), [255, 255, 255], "({x}, {y})");
        }
    }
}

#[test]
fn a_dead_node_shows_its_mask_and_the_photo_does_not_change() {
    // Node 2 is not connected to the Output; it has a Radial. Main is the whole graph that renders.
    let d = photo();
    let live = doc(vec![node(1, json!({}), vec![])], line(&[1]));
    let with_dead = doc(
        vec![
            node(1, json!({}), vec![]),
            node(2, trim(1.0), vec![radial([0.5, 0.5], [0.3, 0.2], 0.4)]),
        ],
        line(&[1]),
    );
    let r = renderer(None, None);
    let live_src = with(&d.source(), live);
    let dead_src = with(&d.source(), with_dead);
    let plain = frame(&r, &live_src, &view(None, 1));
    // Without a mask view the dead node changes no bit.
    assert_eq!(plain.rgba, frame(&r, &dead_src, &view(None, 1)).rgba);
    // Mask only shows its mask: the same as the Radial would be on a live node.
    let dead_mask = frame(&r, &dead_src, &view(Some(MaskStyle::MaskOnly), 2));
    let live_mask = frame(
        &r,
        &with(&d.source(), main_radial()),
        &view(Some(MaskStyle::MaskOnly), 1),
    );
    assert_eq!(dead_mask.rgba, live_mask.rgba);
    assert!(m_at(&dead_mask, 150, 150) > 0.98);
    // Overlay tints the photo as it is (the Output's), with that mask.
    let overlay = frame(&r, &dead_src, &view(Some(MaskStyle::Overlay), 2));
    let (worst, mean) = formula_error(&plain, &dead_mask, &overlay);
    eprintln!(
        "a dead node's overlay against the formula: largest {worst:.2} codes, mean {mean:.3}"
    );
    assert!(worst <= 3.0 && mean < 1.0, "{worst} {mean}");
    assert_ne!(overlay.rgba, plain.rgba, "the tint shows");
}

// ---- Overlay ---------------------------------------------------------------------------------------

#[test]
fn overlay_is_the_formula_on_the_photo_as_it_is() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    let r = renderer(None, None);
    let plain = frame(&r, &src, &view(None, 1));
    let mask = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    let overlay = frame(&r, &src, &view(Some(MaskStyle::Overlay), 1));
    assert!(all_paper_above_and_below(&overlay));
    let (worst, mean) = formula_error(&plain, &mask, &overlay);
    eprintln!("overlay against the formula (CPU): largest {worst:.2} codes, mean {mean:.3}");
    assert!(worst <= 3.0 && mean < 1.0, "{worst} {mean}");
    // Where the mask is 0 the photo is its own (the tint is under half a code); in the middle it is tinted.
    let outside = px(&overlay, 3, 52);
    let was = px(&plain, 3, 52);
    for c in 0..3 {
        assert!(outside[c].abs_diff(was[c]) <= 1, "{outside:?} {was:?}");
    }
    let (tinted, was) = (px(&overlay, 150, 150), px(&plain, 150, 150));
    assert_ne!(tinted, was);
    // The accent is pink: red and blue go up against green.
    assert!(
        i32::from(tinted[0]) + i32::from(tinted[2]) > i32::from(was[0]) + i32::from(was[2]),
        "{tinted:?} {was:?}"
    );
}

#[test]
fn overlay_on_a_node_with_no_mask_tints_the_whole_photo_and_a_later_grade_shows_under_it() {
    let d = photo();
    let r = renderer(None, None);
    let main_only = with(
        &d.source(),
        doc(vec![node(1, json!({}), vec![])], line(&[1])),
    );
    let later = with(
        &d.source(),
        doc(
            vec![node(1, json!({}), vec![]), node(2, trim(1.5), vec![])],
            line(&[1, 2]),
        ),
    );
    for src in [&main_only, &later] {
        let plain = frame(&r, src, &view(None, 1));
        let mask = frame(&r, src, &view(Some(MaskStyle::MaskOnly), 1));
        let overlay = frame(&r, src, &view(Some(MaskStyle::Overlay), 1));
        let (worst, mean) = formula_error(&plain, &mask, &overlay);
        assert!(worst <= 3.0 && mean < 1.0, "{worst} {mean}");
        // Every pixel of the photo is tinted by 0.55 of the way to the accent.
        for (x, y) in [(10, 60), (150, 150), (290, 240)] {
            assert_eq!(m_at(&mask, x, y), 1.0);
            assert_ne!(px(&overlay, x, y), px(&plain, x, y), "({x}, {y})");
        }
    }
    // Node 2 comes after the selected node and still brightens the picture under the tint.
    let under = |src: &PhotoSource| {
        let f = frame(&r, src, &view(Some(MaskStyle::Overlay), 1));
        let [r, g, b] = px(&f, 60, 150);
        i32::from(r) + i32::from(g) + i32::from(b)
    };
    assert!(under(&later) > under(&main_only) + 20);
}

#[test]
fn overlay_without_output_sharpening_is_the_formula_too() {
    let d = photo();
    let mut s = main_radial();
    s.output_mut().output_sharpening = 0.0;
    let src = with(&d.source(), s);
    let r = renderer(None, None);
    let plain = frame(&r, &src, &view(None, 1));
    let mask = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    let overlay = frame(&r, &src, &view(Some(MaskStyle::Overlay), 1));
    let (worst, mean) = formula_error(&plain, &mask, &overlay);
    assert!(worst <= 3.0 && mean < 1.0, "{worst} {mean}");
    assert_ne!(overlay.rgba, plain.rgba);
}

// ---- A view and nothing else -----------------------------------------------------------------------

#[test]
fn the_mask_view_is_never_in_the_before_view_the_readout_or_an_export() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    let r = renderer(None, None);
    let image = r
        .final_size(&src, &CancelToken::new())
        .expect("the final size");
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        // Before: the Before view, with or without a mask view, is the same frame.
        let mut with_mask = view(Some(style), 1);
        with_mask.before = true;
        let mut without = view(None, 1);
        without.before = true;
        assert_eq!(
            frame(&r, &src, &with_mask).rgba,
            frame(&r, &src, &without).rgba,
            "{style:?}: the Before view"
        );
        // The pixel readout reads the photo.
        for css in [(150.0, 150.0), (60.0, 80.0)] {
            let a = r
                .sample_pixel(&src, &view(Some(style), 1), css, &CancelToken::new())
                .expect("a reading");
            let b = r
                .sample_pixel(&src, &view(None, 1), css, &CancelToken::new())
                .expect("a reading");
            assert_eq!(a, b, "{style:?}: the readout at {css:?}");
        }
    }
    // An export (the full-resolution render) after the mask views is the export without them.
    let export = |r: &Renderer| {
        r.render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("render")
        .rgb
    };
    let before = export(&r);
    let _ = frame(&r, &src, &view(Some(MaskStyle::Overlay), 1));
    let _ = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    assert_eq!(export(&r), before, "the exports are unchanged");
    // And a view at a zoom is the same photo: the thumbnail path and the viewport share no mask state.
    let thumb = r
        .thumbnail_png(&src, &CancelToken::new())
        .expect("thumbnail");
    let _ = frame(&r, &src, &view(Some(MaskStyle::Overlay), 1));
    assert_eq!(
        r.thumbnail_png(&src, &CancelToken::new())
            .expect("thumbnail"),
        thumb,
        "thumbnails are unchanged"
    );
    let _ = image;
}

#[test]
fn the_frame_cache_keeps_the_photo_the_overlay_and_the_mask_apart() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    let r = renderer(None, None);
    let plain = frame(&r, &src, &view(None, 1));
    let overlay = frame(&r, &src, &view(Some(MaskStyle::Overlay), 1));
    let mask = frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1));
    assert_ne!(plain.rgba, overlay.rgba);
    assert_ne!(plain.rgba, mask.rgba);
    assert_ne!(overlay.rgba, mask.rgba);
    // A second round (served from the caches where it can be) gives the same three frames.
    assert_eq!(plain.rgba, frame(&r, &src, &view(None, 1)).rgba);
    assert_eq!(
        overlay.rgba,
        frame(&r, &src, &view(Some(MaskStyle::Overlay), 1)).rgba
    );
    assert_eq!(
        mask.rgba,
        frame(&r, &src, &view(Some(MaskStyle::MaskOnly), 1)).rgba
    );
    // Another node is another mask.
    let two = with(
        &d.source(),
        doc(
            vec![
                node(1, json!({}), vec![radial([0.5, 0.5], [0.3, 0.2], 0.4)]),
                node(2, json!({}), vec![radial([0.2, 0.5], [0.1, 0.1], 0.2)]),
            ],
            line(&[1, 2]),
        ),
    );
    let one = frame(&r, &two, &view(Some(MaskStyle::MaskOnly), 1));
    let other = frame(&r, &two, &view(Some(MaskStyle::MaskOnly), 2));
    assert_ne!(one.rgba, other.rgba);
}

#[test]
fn a_repeat_render_and_a_new_renderer_give_the_same_bytes() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        let st = view(Some(style), 1);
        let a = fresh(&renderer(None, None), &src, &st);
        let b = fresh(&renderer(None, None), &src, &st);
        let r = renderer(None, None);
        assert_eq!(a.rgba, b.rgba, "{style:?}");
        assert_eq!(a.rgba, fresh(&r, &src, &st).rgba, "{style:?}");
        assert_eq!(a.rgba, fresh(&r, &src, &st).rgba, "{style:?}: a repeat");
    }
}

#[test]
fn a_mask_view_of_a_node_the_photo_does_not_have_is_the_photo() {
    let d = photo();
    let src = with(&d.source(), main_radial());
    let r = renderer(None, None);
    let plain = frame(&r, &src, &view(None, 1));
    for style in [MaskStyle::Overlay, MaskStyle::MaskOnly] {
        assert_eq!(frame(&r, &src, &view(Some(style), 7)).rgba, plain.rgba);
    }
}

#[test]
fn a_photo_that_is_turned_and_cropped_keeps_the_mask_on_its_subject() {
    // The Radial follows the photo through a turn (docs/23 section 3.6): the same node turned by a quarter is
    // the upright mask turned.
    let d = photo();
    let upright = with(
        &d.source(),
        doc(
            vec![node(
                1,
                json!({}),
                vec![radial([0.3, 0.4], [0.12, 0.1], 0.3)],
            )],
            line(&[1]),
        ),
    );
    let r = renderer(None, None);
    let f = frame(&r, &upright, &view(Some(MaskStyle::MaskOnly), 1));
    // The centre of the Radial is at (0.3, 0.4) of the photo: pixel (90, 50 + 80).
    assert!(m_at(&f, 90, 130) > 0.95, "{}", m_at(&f, 90, 130));
    assert!(m_at(&f, 210, 130) < 0.02);
}

// ---- The device ------------------------------------------------------------------------------------

#[test]
fn the_device_makes_the_mask_and_the_overlay_as_the_cpu_does() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = with(&d.source(), main_radial());
    let on_gpu = renderer(Some(&gpu), None);
    let cpu = renderer(None, None);
    let mask_view = view(Some(MaskStyle::MaskOnly), 1);
    let overlay_view = view(Some(MaskStyle::Overlay), 1);

    let mask_dev = fresh(&on_gpu, &src, &mask_view);
    let mask_cpu = fresh(&cpu, &src, &mask_view);
    assert_eq!(mask_dev.report.device, ComputeDevice::Gpu);
    let (worst, mean) = differ(&mask_dev, &mask_cpu);
    eprintln!("Mask only, device against CPU: largest {worst} codes, mean {mean:.4}");
    assert!(worst <= 1, "Mask only: {worst} codes");

    let over_dev = fresh(&on_gpu, &src, &overlay_view);
    let over_cpu = fresh(&cpu, &src, &overlay_view);
    assert_eq!(over_dev.report.device, ComputeDevice::Gpu);
    let (worst, mean) = differ(&over_dev, &over_cpu);
    // The photo under the tint: the device's own frame against the CPU's, which `render_nodes.rs` bounds at 2.
    let plain_dev = fresh(&on_gpu, &src, &view(None, 1));
    let plain_cpu = fresh(&cpu, &src, &view(None, 1));
    let (plain_worst, plain_mean) = differ(&plain_dev, &plain_cpu);
    eprintln!(
        "Overlay, device against CPU: largest {worst} codes, mean {mean:.4} (the photo alone: {plain_worst}, {plain_mean:.4})"
    );
    assert!(
        worst <= plain_worst.max(1) + 1 && mean < 0.25,
        "{worst} {mean}"
    );
    // The device's overlay is the formula on the device's own plain frame.
    let (formula_worst, formula_mean) = formula_error(&plain_dev, &mask_dev, &over_dev);
    eprintln!(
        "Overlay against the formula (device): largest {formula_worst:.2}, mean {formula_mean:.3}"
    );
    assert!(formula_worst <= 3.0 && formula_mean < 1.0);
    assert_ne!(over_dev.rgba, plain_dev.rgba);
}

#[test]
fn the_device_mask_view_of_a_dead_node_and_of_a_node_with_no_mask() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let on_gpu = renderer(Some(&gpu), None);
    let cpu = renderer(None, None);
    let dead = with(
        &d.source(),
        doc(
            vec![
                node(1, json!({}), vec![]),
                node(2, trim(1.0), vec![radial([0.5, 0.5], [0.3, 0.2], 0.4)]),
            ],
            line(&[1]),
        ),
    );
    let none = with(
        &d.source(),
        doc(
            vec![node(1, json!({}), vec![]), node(2, trim(1.5), vec![])],
            line(&[1, 2]),
        ),
    );
    for (what, src, node_id) in [("dead", &dead, 2), ("no mask", &none, 1)] {
        for style in [MaskStyle::MaskOnly, MaskStyle::Overlay] {
            let st = view(Some(style), node_id);
            let (a, b) = (fresh(&on_gpu, src, &st), fresh(&cpu, src, &st));
            let (worst, mean) = differ(&a, &b);
            eprintln!("{what}, {style:?}: device against CPU largest {worst}, mean {mean:.4}");
            assert!(
                worst <= 2 && mean < 0.25,
                "{what} {style:?}: {worst} {mean}"
            );
        }
    }
}

#[test]
fn a_view_cut_into_bands_is_the_view_in_one_piece() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = with(&d.source(), main_radial());
    let whole = renderer(Some(&gpu), None);
    // Bands of a few dozen rows: 300 x 300 x 16 bytes is 1.44 MB.
    let cut = renderer(Some(&gpu), Some(300_000));
    for style in [MaskStyle::MaskOnly, MaskStyle::Overlay] {
        let st = view(Some(style), 1);
        let a = fresh(&whole, &src, &st);
        let b = fresh(&cut, &src, &st);
        assert_eq!(a.report.device, ComputeDevice::Gpu);
        assert_eq!(b.report.device, ComputeDevice::Gpu);
        assert_eq!(a.rgba, b.rgba, "{style:?}: the bands are the whole");
    }
}

#[test]
fn a_magnified_view_takes_the_overlay_too() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = with(&d.source(), main_radial());
    let on_gpu = renderer(Some(&gpu), None);
    let cpu = renderer(None, None);
    let image = on_gpu
        .final_size(&src, &CancelToken::new())
        .expect("the final size");
    for percent in [100.0, 200.0] {
        for style in [MaskStyle::MaskOnly, MaskStyle::Overlay] {
            let mut st = ViewportState::new(ViewportRect::cell(CELL, CELL, 1.0));
            st.set_mode(ViewportMode::Percent { value: percent }, image);
            st.mask_view = Some(MaskView { node: 1, style });
            let (a, b) = (fresh(&on_gpu, &src, &st), fresh(&cpu, &src, &st));
            let (worst, mean) = differ(&a, &b);
            eprintln!("{percent} %, {style:?}: device against CPU largest {worst}, mean {mean:.4}");
            assert!(
                worst <= 3 && mean < 0.35,
                "{percent} {style:?}: {worst} {mean}"
            );
            let plain = {
                let mut p = st.clone();
                p.mask_view = None;
                fresh(&on_gpu, &src, &p)
            };
            assert_ne!(
                a.rgba, plain.rgba,
                "{percent} {style:?}: the mask view shows"
            );
        }
    }
}

#[test]
fn a_device_without_the_tail_makes_the_overlay_on_the_cpu() {
    // Only the colour pass on the device: the overlay is made whole on the CPU, and it is the CPU's frame.
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = with(&d.source(), main_radial());
    let partial = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu_stages: GpuStages::COLOR,
            ..options(Some(&gpu), None)
        },
    )
    .expect("render thread");
    let st = view(Some(MaskStyle::Overlay), 1);
    let a = fresh(&partial, &src, &st);
    assert_eq!(a.report.device, ComputeDevice::Cpu);
    let b = fresh(&renderer(None, None), &src, &st);
    let (worst, _) = differ(&a, &b);
    assert!(worst <= 2, "{worst}");
}
