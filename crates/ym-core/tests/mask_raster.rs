//! Raster masks through the render graph (`docs/23-nodes.md` sections 4.3 and 5.5, `docs/algorithms/06`
//! section 5), with Brush as the first source (N-13): a Brush row acts where it is painted and nowhere else, it
//! stays on its photo feature when the photo is turned, cropped or bent, tiles of an export equal the whole frame
//! bit for bit with the halos of Grow, Feather and Refine edges, the app and the device agree, and a raster
//! row of a kind this computer cannot make fails the export with the `mask` error and is never "selects nothing".
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, MaskStyle, MaskView,
    NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
    unmakeable_mask,
};
use ym_core::settings::{DevelopSettings, MaskKind, mask_kinds, mask_unavailable};
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(480, 320, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn graph(nodes: Vec<Value>) -> DevelopSettings {
    let mut items = vec![json!({ "id": 1 })];
    items.extend(nodes);
    let doc = json!({ "nodes": { "items": items, "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// A Brush row: one stroke through these points.
fn brush(size: f64, hardness: f64, points: &[[f64; 2]], extra: Value) -> Value {
    let mut row = json!({
        "kind": "brush", "mode": "add",
        "strokes": [{ "size": size, "hardness": hardness, "erase": false, "points": points }],
    });
    if let (Some(r), Some(e)) = (row.as_object_mut(), extra.as_object()) {
        r.extend(e.clone());
    }
    row
}

/// Node 2: an exposure trim of a stop behind these mask rows.
fn node_2(rows: Vec<Value>) -> Value {
    json!({
        "id": 2, "kind": "corrector", "pos": [1.0, 0.0],
        "trim": { "exposure_ev": 1.0 },
        "masks": { "invert": false, "items": rows },
    })
}

fn with(src: &PhotoSource, s: DevelopSettings) -> PhotoSource {
    src.with_settings(s)
}

fn renderer_with_tile(tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn display(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

fn block(img: &FullImage, x0: f64, y0: f64, w: f64, h: f64) -> [f64; 3] {
    let (iw, ih) = (img.width as f64, img.height as f64);
    let (bx, by) = ((x0 * iw) as usize, (y0 * ih) as usize);
    let (bw, bh) = ((w * iw) as usize, (h * ih) as usize);
    let mut sum = [0.0f64; 3];
    for y in by..by + bh {
        for x in bx..bx + bw {
            for (c, s) in sum.iter_mut().enumerate() {
                *s += f64::from(img.rgb[(y * img.width + x) * 3 + c]);
            }
        }
    }
    let n = (bw * bh) as f64;
    sum.map(|v| v / n)
}

// ---- the kinds -----------------------------------------------------------------------------------

#[test]
fn brush_and_background_are_kinds_this_computer_can_make_and_the_others_of_the_raster_family_are_not_yet()
 {
    let kinds = mask_kinds();
    assert_eq!(
        kinds,
        [
            MaskKind::Background,
            MaskKind::Linear,
            MaskKind::Radial,
            MaskKind::Brush,
            MaskKind::ColorRange,
            MaskKind::BrightnessRange
        ]
    );
    let missing: Vec<MaskKind> = mask_unavailable().into_iter().map(|(k, _)| k).collect();
    assert_eq!(
        missing,
        [MaskKind::Object, MaskKind::Subject, MaskKind::People]
    );
    // Every kind is in exactly one of the two lists.
    for k in MaskKind::ALL {
        assert_ne!(kinds.contains(&k), missing.contains(&k), "{k:?}");
    }
    // The reasons' wire names.
    let wire: Vec<(&str, &str)> = mask_unavailable()
        .into_iter()
        .map(|(k, why)| (k.label(), why.wire()))
        .collect();
    let mac = cfg!(target_os = "macos");
    assert_eq!(wire[0], ("Object", "not_supported"));
    assert_eq!(wire[1].1, if mac { "not_supported" } else { "not_a_mac" });
    assert_eq!(wire.len(), 3, "Object, Subject and People: {wire:?}");
}

#[test]
fn a_raster_row_of_a_kind_that_cannot_be_made_is_a_mask_error_never_an_empty_mask() {
    for (row, sentence) in [
        (
            json!({ "kind": "object", "points": [[0.5, 0.5, true]] }),
            "Node 2 has an Object mask, and this version of Yellowmoon cannot make one yet. Remove the mask, or update Yellowmoon.",
        ),
    ] {
        let s = graph(vec![node_2(vec![row])]);
        assert_eq!(unmakeable_mask(&s).as_deref(), Some(sentence));
    }
    if !cfg!(target_os = "macos") {
        let s = graph(vec![node_2(vec![json!({ "kind": "subject" })])]);
        assert_eq!(
            unmakeable_mask(&s).as_deref(),
            Some(
                "Node 2 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask."
            )
        );
        let s = graph(vec![node_2(vec![json!({ "kind": "people" })])]);
        assert_eq!(
            unmakeable_mask(&s).as_deref(),
            Some(
                "Node 2 has a People mask, and this computer cannot make one. Use a Mac that supports People masks, or remove the mask."
            )
        );
    }
    // A row with the eye off, a dead node's row and a kind that can be made are no problem.
    let off = json!({ "kind": "object", "enabled": false });
    assert_eq!(unmakeable_mask(&graph(vec![node_2(vec![off])])), None);
    let ok = brush(0.05, 0.5, &[[0.5, 0.5]], json!({}));
    assert_eq!(unmakeable_mask(&graph(vec![node_2(vec![ok])])), None);
    // Background is made from the photo's own blur map (N-15).
    let ok = json!({ "kind": "background" });
    assert_eq!(unmakeable_mask(&graph(vec![node_2(vec![ok])])), None);
    assert_eq!(unmakeable_mask(&DevelopSettings::default()), None);
}

// ---- Brush on the CPU reference ------------------------------------------------------------------

#[test]
fn a_brush_dab_acts_where_it_is_painted_and_nowhere_else() {
    let d = photo();
    let src = d.source();
    let r = renderer_with_tile(1024);
    let dab = with(
        &src,
        graph(vec![node_2(vec![brush(
            0.2,
            1.0,
            &[[0.5, 0.5]],
            json!({}),
        )])]),
    );
    let none = with(&src, graph(vec![node_2(vec![])]));
    let empty = with(
        &src,
        graph(vec![node_2(vec![
            json!({ "kind": "brush", "strokes": [] }),
        ])]),
    );
    let a = display(&r, &dab);
    let b = display(&r, &none);
    let e = display(&r, &empty);
    // A Brush row with no stroke selects nothing: the node changes nothing at all.
    let main_only = display(
        &r,
        &with(
            &src,
            graph(vec![
                json!({ "id": 2, "kind": "corrector", "pos": [1.0, 0.0] }),
            ]),
        ),
    );
    assert_eq!(e.rgb, main_only.rgb);
    // At the dab's centre the node's picture, far from it the node's input.
    let centre = |img: &FullImage| block(img, 0.47, 0.47, 0.06, 0.06);
    let corner = |img: &FullImage| block(img, 0.02, 0.02, 0.1, 0.1);
    assert_eq!(corner(&a), corner(&main_only));
    let (inside, full) = (centre(&a), centre(&b));
    assert!(
        (inside[1] - full[1]).abs() < 1e-3 * full[1].abs().max(1.0),
        "the dab reaches the whole node effect at its centre: {inside:?} against {full:?}"
    );
    assert_ne!(centre(&a), centre(&main_only));
    // The painted disc is a disc: D = 0.2 of the long edge, 96 px on a 480 px photo.
    let w = a.width;
    let (cx, cy) = (w as f64 / 2.0, a.height as f64 / 2.0);
    let changed = |x: usize, y: usize| {
        let i = (y * w + x) * 3;
        a.rgb[i + 1] != main_only.rgb[i + 1]
    };
    let mut n = 0usize;
    for y in 0..a.height {
        for x in 0..w {
            if changed(x, y) {
                n += 1;
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                assert!(d < 51.0, "a pixel {d:.1} px from the dab's centre changed");
            }
        }
    }
    let disc = std::f64::consts::PI * 48.0 * 48.0;
    assert!(
        (n as f64 - disc).abs() / disc < 0.1,
        "{n} pixels against {disc:.0}"
    );
}

#[test]
fn tiles_of_two_sizes_equal_the_whole_frame_with_the_halo_of_grow_feather_and_refine_edges() {
    let d = photo();
    let src = d.source();
    let rows = vec![
        brush(
            0.12,
            0.4,
            &[[0.2, 0.3], [0.45, 0.55], [0.8, 0.4]],
            json!({ "grow": 0.4, "feather": 0.3, "refine": 0.6 }),
        ),
        json!({ "kind": "brush", "mode": "subtract", "strokes": [
            { "size": 0.05, "hardness": 1.0, "erase": false, "points": [[0.45, 0.55]] }
        ], "feather": 0.1 }),
    ];
    let s = with(&src, graph(vec![node_2(rows)]));
    let whole = display(&renderer_with_tile(4096), &s);
    for tile in [64, 100, 192] {
        let tiled = display(&renderer_with_tile(tile), &s);
        assert_eq!((tiled.width, tiled.height), (whole.width, whole.height));
        assert_eq!(
            tiled.rgb, whole.rgb,
            "tiles of {tile} px differ from the whole frame"
        );
    }
    // The raster rows act (the same node with the rows removed is another picture).
    let plain = display(
        &renderer_with_tile(4096),
        &with(&src, graph(vec![node_2(vec![])])),
    );
    assert_ne!(plain.rgb, whole.rgb);
}

#[test]
fn a_turned_photo_is_the_upright_photo_turned_by_hand_with_a_brush_on_it() {
    let d = photo();
    let src = d.source();
    let r = renderer_with_tile(1024);
    let rows = vec![brush(
        0.15,
        0.5,
        &[[0.3, 0.4], [0.5, 0.45]],
        json!({ "feather": 0.2 }),
    )];
    let upright = with(&src, graph(vec![node_2(rows)]));
    let mut turned = upright.settings.clone();
    turned.input_mut().orientation = ym_core::settings::Orientation::Deg90;
    let a = display(&r, &upright);
    let b = display(&r, &with(&src, turned));
    assert_eq!((b.width, b.height), (a.height, a.width));
    let (w, h) = (a.width, a.height);
    let mut worst = 0.0f32;
    for y in 0..b.height {
        for x in 0..b.width {
            for c in 0..3 {
                let t = b.rgb[(y * b.width + x) * 3 + c];
                let u = a.rgb[((h - 1 - x) * w + y) * 3 + c];
                worst = worst.max((t - u).abs());
            }
        }
    }
    eprintln!("a turned photo against the upright one turned by hand, with a Brush: {worst:e}");
    assert!(worst < 2e-4, "{worst}");
}

#[test]
fn the_app_and_the_cpu_render_the_same_viewport_frame_with_a_brush_row() {
    let d = photo();
    let src = d.source();
    let r = renderer_with_tile(1024);
    let rows = vec![brush(
        0.2,
        0.6,
        &[[0.4, 0.5], [0.6, 0.5]],
        json!({ "feather": 0.15 }),
    )];
    let masked = with(&src, graph(vec![node_2(rows)]));
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let frame = |s: &PhotoSource| {
        r.viewport_frame(s, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
            .rgba
    };
    let with_brush = frame(&masked);
    let without = frame(&with(&src, graph(vec![node_2(vec![])])));
    assert_ne!(with_brush, without, "the brush acts in the viewport");
    // And a second frame is the same bytes.
    assert_eq!(with_brush, frame(&masked));
}

// ---- the device ----------------------------------------------------------------------------------

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
                "\n* GPU TESTS SKIPPED in mask_raster: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

fn device_renderer(gpu: Arc<Gpu>, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

#[test]
fn the_device_makes_the_same_frames_as_the_cpu_with_a_brush_row() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let on_gpu = device_renderer(gpu, 1024);
    let cpu = renderer_with_tile(1024);
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    let frame_of = |r: &Renderer, s: &PhotoSource| {
        r.viewport_frame(s, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
    };
    for (what, extra) in [
        ("a plain dab", json!({})),
        ("grown", json!({ "grow": 0.5 })),
        ("feathered", json!({ "feather": 0.4 })),
        ("refined", json!({ "refine": 0.8 })),
        (
            "all three",
            json!({ "grow": 0.3, "feather": 0.3, "refine": 0.7 }),
        ),
    ] {
        let s = with(
            &src,
            graph(vec![node_2(vec![brush(
                0.25,
                0.5,
                &[[0.35, 0.45], [0.6, 0.55]],
                extra,
            )])]),
        );
        let device = frame_of(&on_gpu, &s);
        assert_eq!(device.report.device, ComputeDevice::Gpu, "{what}");
        let reference = frame_of(&cpu, &s);
        let (mut worst, mut sum) = (0u8, 0u64);
        for (x, y) in device.rgba.iter().zip(&reference.rgba) {
            let d = x.abs_diff(*y);
            worst = worst.max(d);
            sum += u64::from(d);
        }
        let mean = sum as f64 / device.rgba.len() as f64;
        eprintln!("{what}: the device differs from the CPU by {worst} codes, mean {mean:.4}");
        assert!(
            worst <= 3 && mean < 0.3,
            "{what}: {worst} codes, mean {mean}"
        );
    }
}

// ---- the exports on the device -----------------------------------------------------------------------

fn tail_stages() -> GpuStages {
    GpuStages::RAW_CHAIN.union(GpuStages::EXPORT_TAIL)
}

fn stage_renderer(gpu: Option<Arc<Gpu>>, stages: GpuStages, tile: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: stages,
            full_tile_px: tile,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn png_export(r: Renderer, d: &Developed, s: &DevelopSettings) -> (Vec<u8>, ComputeDevice) {
    d.session
        .replace_settings(d.id, s.clone())
        .expect("settings");
    let photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    let home = tempfile::tempdir().expect("home");
    let out = tempfile::tempdir().expect("out");
    let ctx = ExportContext::new(r).with_home(Some(home.path().to_path_buf()));
    let report = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Sixteen,
            destination: Some(out.path().to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        }),
        &ExportControl::none(),
    )
    .expect("export");
    assert!(
        !report.notes.iter().any(|n| n.contains("graphics card")),
        "{:?}",
        report.notes
    );
    (
        std::fs::read(&report.path).expect("the file"),
        report.device,
    )
}

fn png_codes(bytes: &[u8]) -> Vec<u16> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes.to_vec()));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn the_export_tail_with_a_brush_row_is_the_cpu_export_whatever_the_tile() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let rows = vec![
        brush(
            0.14,
            0.4,
            &[[0.2, 0.3], [0.45, 0.55], [0.8, 0.4]],
            json!({ "grow": 0.4, "feather": 0.3, "refine": 0.6 }),
        ),
        brush(0.06, 1.0, &[[0.7, 0.7]], json!({ "mode": "subtract" })),
    ];
    let s = graph(vec![node_2(rows)]);
    let (cpu, cpu_device) = png_export(stage_renderer(None, GpuStages::NONE, 256), &d, &s);
    assert_ne!(cpu_device, ComputeDevice::Gpu);
    let (small, small_device) = png_export(
        stage_renderer(Some(Arc::clone(&gpu)), tail_stages(), 96),
        &d,
        &s,
    );
    let (large, large_device) = png_export(
        stage_renderer(Some(Arc::clone(&gpu)), tail_stages(), 512),
        &d,
        &s,
    );
    assert_eq!(small_device, ComputeDevice::Gpu);
    assert_eq!(large_device, ComputeDevice::Gpu);
    // Tiles of any size give the same file, bit for bit.
    assert!(small == large, "tiles of 96 and 512 px differ");
    // The device and the CPU agree to within a few 16-bit codes.
    let (a, b) = (png_codes(&cpu), png_codes(&small));
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    eprintln!("a Brush export on the device against the CPU: at most {worst} of 65535");
    assert!(worst < 300, "{worst}");
    // The brush acts.
    let (plain, _) = png_export(
        stage_renderer(Some(gpu), tail_stages(), 512),
        &d,
        &graph(vec![node_2(vec![])]),
    );
    assert!(plain != small);
}

#[test]
fn the_dng_colour_stage_with_a_brush_row_is_tile_exact_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let rows = vec![brush(
        0.2,
        0.5,
        &[[0.3, 0.4], [0.6, 0.5]],
        json!({ "grow": 0.3, "feather": 0.2, "refine": 0.5 }),
    )];
    let s = with(&src, graph(vec![node_2(rows)]));
    let render = |r: &Renderer| {
        r.render_full(
            &s,
            &FullRequest {
                target: FullTarget::DngLinearAwg4,
            },
            &CancelToken::new(),
        )
        .expect("render")
    };
    let small = render(&stage_renderer(Some(Arc::clone(&gpu)), tail_stages(), 80));
    let large = render(&stage_renderer(Some(Arc::clone(&gpu)), tail_stages(), 700));
    let cpu = render(&stage_renderer(None, GpuStages::NONE, 200));
    assert_eq!(small.report.device, ComputeDevice::Gpu);
    assert_eq!(small.rgb, large.rgb, "device tiles of 80 and 700 px differ");
    let mut worst = 0.0f64;
    for (x, y) in cpu.rgb.iter().zip(&small.rgb) {
        worst = worst.max(f64::from((x - y).abs()) / f64::from(x.abs().max(1.0)));
    }
    eprintln!("a Brush row in the DNG's colour stage, device against CPU: {worst:e}");
    assert!(worst < 2e-3, "{worst:e}");
}

// ---- the mask view -------------------------------------------------------------------------------

#[test]
fn the_mask_view_shows_the_brush_plane_on_the_cpu_and_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let d = photo();
    let src = d.source();
    let cpu = renderer_with_tile(1024);
    let on_gpu = device_renderer(gpu, 1024);
    // A soft round dab at the middle of the photo, Feather and Grow on it.
    let s = with(
        &src,
        graph(vec![node_2(vec![brush(
            0.3,
            0.2,
            &[[0.5, 0.5]],
            json!({ "feather": 0.2, "grow": 0.1 }),
        )])]),
    );
    let st = |style| {
        let mut st = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
        st.mask_view = Some(MaskView { node: 2, style });
        st
    };
    let only = st(MaskStyle::MaskOnly);
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let frame = |r: &Renderer, st: &ViewportState| {
        r.viewport_frame(&s, st, &options, &CancelToken::new())
            .expect("frame")
    };
    let a = frame(&cpu, &only);
    let b = frame(&on_gpu, &only);
    assert_eq!(b.report.device, ComputeDevice::Gpu);
    // The photo is 3:2 in a square cell: rows 50 to 250. The code value is `m`: 255 at the dab's centre, 0 far away.
    let px =
        |f: &ym_core::pipeline::RenderedFrame, x: usize, y: usize| f.rgba[(y * f.width + x) * 4];
    for f in [&a, &b] {
        assert!(
            px(f, 150, 150) >= 252,
            "the dab's centre is {}",
            px(f, 150, 150)
        );
        assert_eq!(px(f, 10, 60), 0);
        assert_eq!(px(f, 290, 240), 0);
    }
    let worst = a
        .rgba
        .iter()
        .zip(&b.rgba)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    eprintln!("the mask view of a Brush row, device against CPU: {worst} codes");
    assert!(worst <= 1, "{worst}");
    // The overlay tints the photo where the dab is, on both.
    let overlay = st(MaskStyle::Overlay);
    let (oa, ob) = (frame(&cpu, &overlay), frame(&on_gpu, &overlay));
    let plain = {
        let mut p = ViewportState::new(ViewportRect::cell(300.0, 300.0, 1.0));
        p.mask_view = None;
        frame(&cpu, &p)
    };
    assert_ne!(oa.rgba, plain.rgba);
    let worst = oa
        .rgba
        .iter()
        .zip(&ob.rgba)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    assert!(worst <= 3, "{worst}");
}

// ---- views see the photo beyond their edge --------------------------------------------------------------

/// A viewport state at 100 % whose cell is `w x h` device pixels and whose top left corner is the pixel
/// `(ox, oy)` of the final image (the origin of a 100 % view is a whole pixel), showing node 2's mask when `mask`.
fn view_100(
    image: (usize, usize),
    (ox, oy): (usize, usize),
    (w, h): (usize, usize),
    mask: bool,
) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(w as f64, h as f64, 1.0));
    st.mode = ym_core::pipeline::ViewportMode::Percent { value: 100.0 };
    st.centre = [
        (ox as f64 + w as f64 * 0.5) / image.0 as f64,
        (oy as f64 + h as f64 * 0.5) / image.1 as f64,
    ];
    if mask {
        st.mask_view = Some(MaskView {
            node: 2,
            style: MaskStyle::MaskOnly,
        });
    }
    st
}

/// The largest difference in codes and the mean of the colour channels of the pixels of `view` against the same
/// pixels of `whole` (the view's `(0, 0)` is `at` in it), over the rectangle `(x0, y0, x1, y1)` of the view.
fn crop_difference(
    whole: &ym_core::pipeline::RenderedFrame,
    view: &ym_core::pipeline::RenderedFrame,
    at: (usize, usize),
    (x0, y0, x1, y1): (usize, usize, usize, usize),
) -> (u8, f64) {
    let (mut worst, mut sum, mut n) = (0u8, 0u64, 0u64);
    for y in y0..y1 {
        for x in x0..x1 {
            for c in 0..3 {
                let a = view.rgba[(y * view.width + x) * 4 + c];
                let b = whole.rgba[((at.1 + y) * whole.width + at.0 + x) * 4 + c];
                let d = a.abs_diff(b);
                worst = worst.max(d);
                sum += u64::from(d);
                n += 1;
            }
        }
    }
    (worst, sum as f64 / n as f64)
}

/// A photo with a wide, soft, refined dab, so that the masks' receptive field is tens of pixels.
fn soft_dab_photo() -> (Developed, serde_json::Value) {
    (
        photo(),
        brush(
            0.22,
            0.5,
            &[[0.5, 0.5]],
            json!({ "grow": 0.5, "feather": 1.0, "refine": 1.0 }),
        ),
    )
}

#[test]
fn a_zoomed_view_sees_the_photo_beyond_its_edge_like_the_export_of_that_size() {
    let (d, dab) = soft_dab_photo();
    let src = d.source();
    let s = with(&src, graph(vec![node_2(vec![dab])]));
    let gpu = gpu();
    let cpu = renderer_with_tile(1024);
    let cancel = CancelToken::new();
    let image = cpu.final_size(&s, &cancel).expect("size");
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    // The reach of the masks at this size, which the view must see beyond its edge.
    let device = gpu.map(|g| device_renderer(g, 1024));
    for mask in [true, false] {
        for (name, r) in [("CPU", Some(&cpu)), ("device", device.as_ref())] {
            let Some(r) = r else { continue };
            let frame =
                |st: &ViewportState| r.viewport_frame(&s, st, &options, &cancel).expect("frame");
            // The export of this size: the whole photo at 100 %.
            let whole = frame(&view_100(image, (0, 0), image, mask));
            // A view of the right-hand part whose left edge is inside the dab's feather (the dab is at 240 px).
            let (ox, oy, w, h) = (205, 60, 220, 200);
            let view = frame(&view_100(image, (ox, oy), (w, h), mask));
            if name == "device" {
                assert_eq!(view.report.device, ComputeDevice::Gpu);
            }
            // Close to the dab, the edge of the view is inside the feather: the old frame, which ended at the view,
            // differed by tens of codes there.
            let (worst, mean) = crop_difference(&whole, &view, (ox, oy), (0, 0, w, h));
            eprintln!(
                "{name}, {}: a 100 % view against the export, {worst} codes, mean {mean:.5}",
                if mask { "the mask" } else { "the picture" }
            );
            // The dither counts its texels from the view's own corner, so a pixel can differ by a code of it.
            let (gate, gate_mean) = if mask { (1, 0.3) } else { (2, 0.5) };
            assert!(
                worst <= gate && mean < gate_mean,
                "{name}: {worst} codes, mean {mean}"
            );
        }
    }
}

#[test]
fn two_views_panned_by_a_few_pixels_agree_where_they_overlap() {
    let (d, dab) = soft_dab_photo();
    let src = d.source();
    let s = with(&src, graph(vec![node_2(vec![dab])]));
    let gpu = gpu();
    let cpu = renderer_with_tile(1024);
    let cancel = CancelToken::new();
    let image = cpu.final_size(&s, &cancel).expect("size");
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let device = gpu.map(|g| device_renderer(g, 1024));
    for mask in [true, false] {
        for (name, r) in [("CPU", Some(&cpu)), ("device", device.as_ref())] {
            let Some(r) = r else { continue };
            let frame =
                |st: &ViewportState| r.viewport_frame(&s, st, &options, &cancel).expect("frame");
            let (w, h) = (180, 150);
            let a = frame(&view_100(image, (200, 70), (w, h), mask));
            for (dx, dy) in [(3usize, 0usize), (0, 5), (7, 4)] {
                let b = frame(&view_100(image, (200 + dx, 70 + dy), (w, h), mask));
                // `b` shows the pixels of `a` moved by (dx, dy): its (0, 0) is a's (dx, dy).
                let (worst, mean) = crop_difference(&a, &b, (dx, dy), (0, 0, w - dx, h - dy));
                eprintln!(
                    "{name}, {}: panned by ({dx}, {dy}), {worst} codes, mean {mean:.5}",
                    if mask { "the mask" } else { "the picture" }
                );
                // The dither counts its texels from the view's own corner, so a pixel can differ by a code of it.
                let (gate, gate_mean) = if mask { (1, 0.3) } else { (2, 0.5) };
                assert!(
                    worst <= gate && mean < gate_mean,
                    "{name}: {worst} codes, mean {mean}"
                );
            }
        }
    }
}

// ---- views the device cuts into bands ------------------------------------------------------------------

/// A device renderer whose view bands are `band_bytes` of one binding.
fn banded_renderer(gpu: Arc<Gpu>, band_bytes: u64) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            view_band_bytes: band_bytes,
            full_tile_px: 1024,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

/// The largest difference in codes and the mean of two frames of one size.
fn frame_difference(
    a: &ym_core::pipeline::RenderedFrame,
    b: &ym_core::pipeline::RenderedFrame,
) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    crop_difference(a, b, (0, 0), (0, 0, a.width, a.height))
}

#[test]
fn the_bands_of_a_view_with_a_brush_row_are_the_view_in_one_piece_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let (d, dab) = soft_dab_photo();
    let src = d.source();
    let s = with(&src, graph(vec![node_2(vec![dab])]));
    let one_piece = device_renderer(Arc::clone(&gpu), 1024);
    // About 160 rows of a 480 px view with its border and the masks' reach (16 bytes a pixel, two thirds of the bytes
    // for the six planes of a chain that reads the photo's places).
    let banded = banded_renderer(Arc::clone(&gpu), 2_300_000);
    let cpu = renderer_with_tile(1024);
    let cancel = CancelToken::new();
    let image = cpu.final_size(&s, &cancel).expect("size");
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    for (what, st) in [
        ("the picture", view_100(image, (0, 0), image, false)),
        ("the mask", view_100(image, (0, 0), image, true)),
        (
            "a panned 100 % view",
            view_100(image, (100, 40), (300, 260), false),
        ),
    ] {
        let frame = |r: &Renderer| r.viewport_frame(&s, &st, &options, &cancel).expect("frame");
        let a = frame(&one_piece);
        assert_eq!(a.report.device, ComputeDevice::Gpu, "{what}");
        let jobs = gpu.counters().submissions;
        let b = frame(&banded);
        let ran = gpu.counters().submissions - jobs;
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "{what}: {:?}",
            b.report.notes
        );
        assert!(ran >= 2, "{what}: the view was cut into {ran} jobs");
        let (worst, mean) = frame_difference(&a, &b);
        eprintln!(
            "{what}: {ran} device jobs, the bands against the view in one piece: {worst} codes, mean {mean:.5}"
        );
        assert!(
            worst <= 1 && mean < 0.01,
            "{what}: {worst} codes, mean {mean}"
        );
        // And the CPU makes it within the device's gates.
        let c = frame(&cpu);
        let (worst, mean) = frame_difference(&b, &c);
        eprintln!("{what}: the bands against the CPU: {worst} codes, mean {mean:.5}");
        assert!(
            worst <= 3 && mean < 0.3,
            "{what}: {worst} codes, mean {mean}"
        );
    }
}

#[test]
fn a_retina_size_view_with_a_brush_row_is_made_on_the_device_in_bands() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(2880, 1920, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let src = d.source();
    // A dab the size of a face, softly feathered and refined: the masks reach tens of pixels at this size.
    let rows = vec![brush(
        0.2,
        0.4,
        &[[0.35, 0.5], [0.6, 0.45]],
        json!({ "grow": 0.3, "feather": 0.3, "refine": 0.3 }),
    )];
    let s = with(&src, graph(vec![node_2(rows)]));
    let cpu = renderer_with_tile(1024);
    let on_gpu = device_renderer(Arc::clone(&gpu), 1024);
    let cancel = CancelToken::new();
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    // The cell is the photo at 100 %: 2880 x 1920 device pixels, 88 MB of frame, which passes the band size of a chain
    // with raster masks.
    let st = ViewportState::new(ViewportRect::cell(2880.0, 1920.0, 1.0));
    let frame = |r: &Renderer| r.viewport_frame(&s, &st, &options, &cancel).expect("frame");
    let jobs = gpu.counters().submissions;
    let device = frame(&on_gpu);
    let ran = gpu.counters().submissions - jobs;
    assert_eq!(
        device.report.device,
        ComputeDevice::Gpu,
        "{:?}",
        device.report.notes
    );
    assert!(ran >= 2, "the view was cut into {ran} jobs");
    let reference = frame(&cpu);
    assert_eq!(reference.report.device, ComputeDevice::Cpu);
    let (worst, mean) = frame_difference(&device, &reference);
    eprintln!(
        "a retina-size view: {ran} device jobs, the device against the CPU: {worst} codes, mean {mean:.5}"
    );
    assert!(worst <= 3 && mean < 0.3, "{worst} codes, mean {mean}");
    // Its bands are the same bytes with the cut elsewhere.
    let finer = banded_renderer(gpu, 60_000_000);
    let other = frame(&finer);
    assert_eq!(other.report.device, ComputeDevice::Gpu);
    let (worst, mean) = frame_difference(&device, &other);
    eprintln!("a retina-size view cut elsewhere: {worst} codes, mean {mean:.6}");
    assert!(worst <= 1 && mean < 0.01, "{worst} codes, mean {mean}");
}
