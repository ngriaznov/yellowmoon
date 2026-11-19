//! The Blur node through the render graph on the CPU reference (`docs/33-blur.md` sections 3.8, 6 and 8, BL-04): the node is a
//! spatial step on the Film node's route, its pixels are the same bits for every tile size and every thread count, the
//! view and the export agree at another scale, an empty mask costs nothing, and the device refuses a plan with a live
//! node and the CPU makes the frame with its reason (until BL-05).
mod common;

use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    FrameOptions, FullImage, FullRequest, FullTarget, MaskStyle, MaskView, NeutralOnly,
    PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.4)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

/// Main, then a Blur node with `body` (Input, Main, the node, Output).
fn graph(body: &Value) -> DevelopSettings {
    let mut node = json!({ "id": 2, "kind": "blur", "pos": [1.0, 0.0] });
    if let (Some(n), Some(b)) = (node.as_object_mut(), body.as_object()) {
        n.extend(b.clone());
    }
    let doc = json!({ "nodes": { "items": [ { "id": 1 }, node ], "links": [
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

fn renderer(tile: usize, threads: usize) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            threads,
            full_tile_px: tile,
            gpu: None,
            gpu_stages: GpuStages::NONE,
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn full(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        },
        &CancelToken::new(),
    )
    .expect("render")
}

fn bits(img: &FullImage) -> Vec<u32> {
    img.rgb.iter().map(|v| v.to_bits()).collect()
}

/// One body for each type, with a mask for some: small amounts (the picture is 640 px wide).
fn bodies() -> Vec<Value> {
    let linear = json!({ "items": [
        { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 } ] });
    vec![
        json!({ "type": "soft", "soft": { "radius": 25 } }),
        json!({ "type": "soft", "soft": { "radius": 25 }, "masks": linear }),
        json!({ "type": "lens", "lens": { "radius": 30, "blades": 6, "rotation_deg": 15, "highlights": 60 },
            "masks": linear }),
        json!({ "type": "lens", "lens": { "radius": 30 }, "respect_mask_edges": false, "masks": linear }),
        json!({ "type": "motion", "motion": { "length": 40, "angle_deg": 30 }, "masks": linear }),
        json!({ "type": "zoom", "zoom": { "amount": 50, "center": [0.3, 0.6] }, "masks": linear }),
    ]
}

/// Every type, with and without a mask: a tile with its halo is the frame (`docs/33` section 3.8), for tiles of 97, 160, 256 px and
/// 1, 3 and 4 threads, and a repeat is the same bits.
#[test]
fn every_type_is_the_same_bits_whatever_the_tiles_and_the_threads() {
    let d = photo();
    let plain = bits(&full(&renderer(1024, 4), &d.source()));
    let reference = renderer(1024, 4);
    for body in bodies() {
        let src = d.source().with_settings(graph(&body));
        let whole = bits(&full(&reference, &src));
        assert_ne!(whole, plain, "the node changes the picture: {body}");
        for (tile, threads) in [(97, 1), (160, 3), (256, 4)] {
            let got = bits(&full(&renderer(tile, threads), &src));
            assert!(got == whole, "tile {tile}, {threads} threads: {body}");
        }
        assert!(bits(&full(&reference, &src)) == whole, "a repeat: {body}");
    }
}

/// A node that is bypassed or at amount 0 changes no bit, and a node whose mask selects a corner changes only the corner (every
/// other tile has `m = 0` in its own pixels and is skipped).
#[test]
fn an_inert_node_and_an_empty_mask_change_no_bit() {
    let d = photo();
    let r = renderer(160, 3);
    let plain = bits(&full(&r, &d.source()));
    // A node that selects a few pixels of the corner only: everywhere else the picture is the input's bits.
    let corner = json!({ "items": [
        { "kind": "radial", "center": [0.0, 0.0], "radii": [0.01, 0.01], "feather": 0.0 } ] });
    for body in [
        json!({ "bypass": true }),
        json!({ "soft": { "radius": 0 } }),
        json!({ "type": "zoom", "zoom": { "amount": 0 } }),
        json!({ "type": "motion", "motion": { "length": 0 } }),
        json!({ "type": "lens", "lens": { "radius": 0 } }),
    ] {
        let got = bits(&full(&r, &d.source().with_settings(graph(&body))));
        assert!(got == plain, "{body}");
    }
    let body = json!({ "type": "soft", "soft": { "radius": 40 }, "masks": corner });
    let got = bits(&full(&r, &d.source().with_settings(graph(&body))));
    let mut moved = 0;
    for y in 0..420 {
        for x in 0..640 {
            let same = (0..3).all(|c| got[(y * 640 + x) * 3 + c] == plain[(y * 640 + x) * 3 + c]);
            if x >= 40 || y >= 40 {
                assert!(same, "({x}, {y}) is outside the mask");
            }
            moved += usize::from(!same);
        }
    }
    assert!(moved > 0, "the mask selects the corner");
}

/// The picture is the same at another scale (`docs/33` section 8, resolution independence): a view at half the size of the
/// export, against the export reduced by two, with the node on and with it off; the node adds to the difference no more than
/// the starting bound (mean 1 code, the largest 10 in sRGB codes).
#[test]
fn a_view_and_the_export_agree_at_another_scale() {
    let d = photo();
    let r = renderer(1024, 4);
    let (w, h) = (640usize, 420usize);
    let to_code = |v: f32| -> f64 {
        let v = f64::from(v).clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        e * 255.0
    };
    let measure = |src: &PhotoSource| -> (f64, f64) {
        let exp = full(&r, src);
        assert_eq!((exp.width, exp.height), (w, h));
        let st = ViewportState::new(ViewportRect::cell(320.0, 210.0, 1.0));
        let f = r
            .viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame");
        let (vw, vh) = (f.width, f.height);
        let (mut sum, mut worst, mut n) = (0.0f64, 0.0f64, 0usize);
        // The view is the export reduced by two where it holds the photo (inside a margin of the border).
        for y in 8..vh - 8 {
            for x in 8..vw - 8 {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (ex, ey) = (
                            ((2 * x + dx) * w / (2 * vw)).min(w - 1),
                            ((2 * y + dy) * h / (2 * vh)).min(h - 1),
                        );
                        acc += to_code(exp.rgb[(ey * w + ex) * 3 + c]);
                    }
                    let view = f64::from(f.rgba[(y * vw + x) * 4 + c]);
                    let e = (view - acc / 4.0).abs();
                    sum += e;
                    worst = worst.max(e);
                    n += 1;
                }
            }
        }
        (sum / n as f64, worst)
    };
    let off = measure(&d.source().with_settings(graph(&json!({ "bypass": true }))));
    let body = json!({ "type": "soft", "soft": { "radius": 30 } });
    let on = measure(&d.source().with_settings(graph(&body)));
    eprintln!(
        "view against export: node off mean {:.3} largest {:.1}; on mean {:.3} largest {:.1}",
        off.0, off.1, on.0, on.1
    );
    assert!(on.0 <= off.0 + 1.0, "mean {} against {}", on.0, off.0);
    assert!(on.1 <= off.1 + 10.0, "largest {} against {}", on.1, off.1);
}

/// A type with no device kernel (BL-06 to BL-09 add them one by one; Lens, BL-07, has one) is refused by the device by its name
/// and the CPU makes the frame, with its reason; a type with one runs on the device with no reason; a node at its identity keeps
/// the device either way.
#[test]
fn the_device_refuses_a_type_with_no_kernel_and_runs_one_that_has_it() {
    let gpu = match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            assert!(
                !ym_gpu::device::gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            eprintln!("\n* GPU TESTS SKIPPED in render_blur: {e}\n");
            return;
        }
    };
    let d = photo();
    let device = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let st = ViewportState::new(ViewportRect::cell(320.0, 210.0, 1.0));
    let frame = |src: &PhotoSource| {
        device
            .viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
            .expect("frame")
    };
    let off = frame(&d.source().with_settings(graph(&json!({ "bypass": true }))));
    // A node at its identity keeps the device and says nothing.
    assert!(!off.report.device_refused && off.report.cpu_reason.is_none());
    for (name, label, body) in [
        (
            "soft",
            "Soft",
            json!({ "type": "soft", "soft": { "radius": 30 } }),
        ),
        (
            "lens",
            "Lens",
            json!({ "type": "lens", "lens": { "radius": 30 } }),
        ),
        (
            "motion",
            "Motion",
            json!({ "type": "motion", "motion": { "length": 30 } }),
        ),
        (
            "zoom",
            "Zoom",
            json!({ "type": "zoom", "zoom": { "amount": 30 } }),
        ),
    ] {
        let settings = graph(&body);
        let params = ym_core::pipeline::color::blur_params_of(
            settings.nodes.blur(2).expect("the blur node"),
        );
        let live = d.source().with_settings(settings);
        let has_kernel = ym_gpu::kernels::blur_stack::device_kernel(&params).is_some();
        let got = frame(&live);
        if has_kernel {
            assert!(
                !got.report.device_refused && got.report.cpu_reason.is_none(),
                "{name}: the device runs it: {:?}",
                got.report.cpu_reason
            );
        } else {
            assert!(
                got.report.device_refused,
                "{name}: the device refused the plan"
            );
            assert!(
                got.report.cpu_reason.as_deref().is_some_and(
                    |r| r.contains(&format!("a {label} Blur node is not on the device yet"))
                ),
                "{name}: the reason: {:?}",
                got.report.cpu_reason
            );
        }
        // The frame is the picture with the node in it (not the identity).
        assert_ne!(got.rgba, off.rgba, "{name}");
    }
}

/// Show mask on a Blur node does not run the blur (`docs/33` section 6): the plan ends at `m`, so the view is the same whatever
/// the node blurs with.
#[test]
fn show_mask_does_not_run_the_blur() {
    let d = photo();
    let r = renderer(1024, 4);
    let mut st = ViewportState::new(ViewportRect::cell(320.0, 210.0, 1.0));
    st.mask_view = Some(MaskView {
        node: 2,
        style: MaskStyle::MaskOnly,
        row: None,
        depth: false,
        face: None,
    });
    let masks = json!({ "items": [
        { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 } ] });
    let frame = |body: &Value| {
        r.viewport_frame(
            &d.source().with_settings(graph(body)),
            &st,
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame")
        .rgba
    };
    // The grey is `m`: another radius, another type, and the switch change nothing in it.
    let soft = frame(&json!({ "type": "soft", "soft": { "radius": 80 }, "masks": masks }));
    let other = frame(&json!({ "type": "soft", "soft": { "radius": 10 }, "masks": masks }));
    let lens = frame(
        &json!({ "type": "lens", "lens": { "radius": 80 }, "respect_mask_edges": false, "masks": masks }),
    );
    assert!(soft == other && soft == lens);
}
