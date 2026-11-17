//! The time the Face Refinement node takes (`docs/27` decision 13): a preview frame with one and with two faces after a slider
//! move of the node and after one of a corrector upstream, the first analysis, and an export of a 24 MP photo. Ignored: run it in a
//! release build to measure, `YELLOWMOON_DATA=$PWD/data cargo test -p ym-core --release --test face_timing -- --ignored --nocapture`.
//! Needs the weights (`tools/fetch/mediapipe.sh`) and the public portraits (`tools/fetch/mediapipe.sh --photos`); the faces of
//! the portraits are enlarged to the size they have in a 24 MP photo.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use common::*;
use serde_json::{Value, json};
use ym_color::display::OutputTarget;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    FaceModel, FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions,
    Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The photo enlarged (bilinear) to `w x h`, as a raw file.
fn big(id: &str, w: u32, h: u32) -> Option<Developed> {
    let path = repo()
        .join("testdata/mediapipe/photos")
        .join(format!("{id}.jpg"));
    if !path.is_file() {
        println!("SKIPPED: no photo {}", path.display());
        return None;
    }
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(std::fs::read(&path).ok()?));
    let rgb = d.decode().ok()?;
    let info = d.info()?;
    let (sw, sh) = (usize::from(info.width), usize::from(info.height));
    let white = light(PROFILE, 5500.0);
    let scene = Scene::new(w, h, white, Layout::Uniform, 0.5).noise(0.0, 0.0);
    let peak = white.iter().copied().fold(0.0, f64::max);
    let lut: Vec<f64> = (0..256)
        .map(|v| {
            let e = f64::from(v) / 255.0;
            if e <= 0.04045 {
                e / 12.92
            } else {
                ((e + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let range = scene.white_level - scene.black;
    // Fit the photo in the frame, centred, by its height; the rest is grey.
    let scale = f64::from(h) / sh as f64;
    let (fw, off) = (sw as f64 * scale, (f64::from(w) - sw as f64 * scale) / 2.0);
    let mut mosaic = vec![0u16; (w * h) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let c = [0usize, 1, 1, 2][(y & 1) * 2 + (x & 1)];
            let (fx, fy) = ((x as f64 - off) / scale, y as f64 / scale);
            let lin = if x as f64 >= off && (x as f64) < off + fw {
                let (ix, iy) = ((fx as usize).min(sw - 2), (fy as usize).min(sh - 2));
                let (tx, ty) = (fx - ix as f64, fy - iy as f64);
                let at = |dx: usize, dy: usize| {
                    lut[usize::from(rgb[((iy + dy) * sw + ix + dx) * 3 + c])]
                };
                (at(0, 0) * (1.0 - tx) + at(1, 0) * tx) * (1.0 - ty)
                    + (at(0, 1) * (1.0 - tx) + at(1, 1) * tx) * ty
            } else {
                0.2
            };
            let v = (0.9 * lin * white[c] / peak).clamp(0.0, 1.0);
            mosaic[y * w as usize + x] = (scene.black + v * range).round() as u16;
        }
    }
    let dir = tempfile::tempdir().ok()?;
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .ok()?;
    Some(Developed::open(dir))
}

fn graph(face: Value, corrector_ev: f64) -> DevelopSettings {
    let link = |a: Value, b: Value| json!({ "from": { "node": a, "port": "picture" }, "to": { "node": b, "port": "picture" } });
    let mut face = face;
    face["id"] = json!(3);
    face["kind"] = json!("face_refinement");
    face["pos"] = json!([2.0, 0.0]);
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": {
        "items": [ { "id": 1 },
            { "id": 2, "kind": "corrector", "pos": [1.0, 0.0], "trim": { "exposure_ev": corrector_ev } }, face ],
        "links": [ link(json!("input"), json!(1)), link(json!(1), json!(2)), link(json!(2), json!(3)), link(json!(3), json!("output")) ] } }))
    .expect("a document")
    .validated()
    .expect("a valid graph")
    .0
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn frame(r: &Renderer, src: &PhotoSource, zoom_100: bool) -> f64 {
    let mut st = ViewportState::new(ViewportRect::cell(1600.0, 1000.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    let t = Instant::now();
    let _ = r
        .viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame");
    ms(t)
}

#[test]
#[ignore = "a measurement"]
fn the_node_costs() {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    println!("{threads} threads");
    let model = FaceModel::probe(&data_root()).expect("the face models");
    for (id, w, h, faces) in [
        ("mp-portrait", 6000, 4000, 1),
        ("mp-two-people", 6000, 4000, 2),
    ] {
        let Some(d) = big(id, w, h) else { continue };
        let src = d.source();
        let r = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                face_model: Some(Arc::clone(&model)),
                data_root: Some(data_root()),
                ..RenderOptions::default()
            },
        )
        .expect("render thread");
        r.navigate(src.id);
        let at = |face: Value, ev: f64| src.with_settings(graph(face, ev));
        let inert = json!({ "all": { "blemish": 0, "dark_circles": 0, "even_skin": 0 } });
        let base = at(inert.clone(), 0.0);
        // The frames the viewer draws before the faces are known (the node passes through), then the first analysis.
        let _ = frame(&r, &base, false);
        let first_no_faces = frame(&r, &at(json!({}), 0.0), false);
        let t = Instant::now();
        let found = r
            .faces_found(&at(json!({}), 0.0), &CancelToken::new())
            .expect("faces");
        let analysis = ms(t);
        println!("== {id} ({w} x {h}): {} faces ==", found.len());
        assert_eq!(found.len(), faces);
        let timings = r.faces_timings(src.id).expect("timings");
        println!(
            "first analysis {analysis:.0} ms: picture {:.0}, finder {:.0}, masks and maps {:.0}",
            timings.picture_ms, timings.find_ms, timings.prepare_ms
        );
        println!("a frame before the faces are known {first_no_faces:.0} ms");
        // Fit-size frames (the cell is 1600 x 1000).
        for (what, zoom) in [("fit", false), ("100 %", true)] {
            let _ = frame(&r, &at(json!({}), 0.0), zoom);
            let mut base_moves = Vec::new();
            for i in 1..=3 {
                base_moves.push(frame(
                    &r,
                    &at(inert.clone(), 0.1 * f64::from(i) + 0.05),
                    zoom,
                ));
            }
            let mut node_moves = Vec::new();
            let mut upstream_moves = Vec::new();
            for i in 1..=3 {
                node_moves.push(frame(
                    &r,
                    &at(json!({ "all": { "blemish": 30, "dark_circles": 30, "even_skin": f64::from(20 + i * 10) } }), 0.0),
                    zoom,
                ));
                upstream_moves.push(frame(&r, &at(json!({}), 0.1 * f64::from(i)), zoom));
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "{what}: the same upstream moves with the node at 0: {:.0} ms; after a slider of the node {:.0} ms; after a corrector upstream {:.0} ms",
                mean(&base_moves),
                mean(&node_moves),
                mean(&upstream_moves)
            );
        }
        assert_eq!(r.faces_analyses(src.id), 1, "no slider ran a network");
        // An export of the 24 MP photo, with the node and without.
        let target = FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Srgb),
        };
        let t = Instant::now();
        let _ = r
            .render_full(&base, &target, &CancelToken::new())
            .expect("export");
        let plain = ms(t);
        let t = Instant::now();
        let _ = r
            .render_full(&at(json!({}), 0.0), &target, &CancelToken::new())
            .expect("export");
        let with = ms(t);
        println!(
            "export of 24 MP: without the node {plain:.0} ms, with it {with:.0} ms (its own analysis included, kept: {})",
            r.faces_analyses(src.id) > 1
        );
        if let Ok(tile) = std::env::var("YM_TIMING_TILE") {
            let tile: usize = tile.parse().expect("a tile");
            let bigger = Renderer::new(
                Arc::new(NeutralOnly),
                RenderOptions {
                    face_model: Some(Arc::clone(&model)),
                    data_root: Some(data_root()),
                    full_tile_px: tile,
                    ..RenderOptions::default()
                },
            )
            .expect("render thread");
            let t = Instant::now();
            let _ = bigger
                .render_full(&base, &target, &CancelToken::new())
                .expect("export");
            let plain = ms(t);
            let t = Instant::now();
            let _ = bigger
                .render_full(&at(json!({}), 0.0), &target, &CancelToken::new())
                .expect("export");
            println!(
                "tile {tile}: export without the node {plain:.0} ms, with it {:.0} ms",
                ms(t)
            );
        }
    }
}

/// The same measurements with the face step on the device (S-13): the frames of a viewport and the export tail, set against the
/// CPU numbers of `the_node_costs`. Run it with the same command (`--ignored --nocapture the_node_costs_on_the_device`); on a
/// machine with no graphics card Mesa's lavapipe (`WGPU_BACKEND=vulkan`) is a software device, which says nothing about a GPU.
#[test]
#[ignore = "a measurement"]
fn the_node_costs_on_the_device() {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let gpu = Arc::new(Gpu::start(GpuConfig::default()).expect("a device"));
    println!("{threads} threads; device: {}", gpu.info());
    let model = FaceModel::probe(&data_root()).expect("the face models");
    for (id, w, h, faces) in [
        ("mp-portrait", 6000, 4000, 1),
        ("mp-two-people", 6000, 4000, 2),
    ] {
        let Some(d) = big(id, w, h) else { continue };
        let src = d.source();
        let make = |gpu: Option<Arc<Gpu>>| {
            Renderer::new(
                Arc::new(NeutralOnly),
                RenderOptions {
                    face_model: Some(Arc::clone(&model)),
                    data_root: Some(data_root()),
                    gpu_stages: if gpu.is_some() {
                        GpuStages::ALL
                    } else {
                        GpuStages::NONE
                    },
                    gpu,
                    ..RenderOptions::default()
                },
            )
            .expect("render thread")
        };
        let r = make(Some(Arc::clone(&gpu)));
        r.navigate(src.id);
        let at = |face: Value, ev: f64| src.with_settings(graph(face, ev));
        let inert = json!({ "all": { "blemish": 0, "dark_circles": 0, "even_skin": 0 } });
        let live = |i: i32| {
            json!({ "all": { "blemish": 30, "dark_circles": 30, "even_skin": f64::from(20 + i * 10),
                                              "skin_contrast": 20, "teeth_whitening": 30 } })
        };
        let base = at(inert.clone(), 0.0);
        let _ = frame(&r, &base, false);
        let _ = r
            .faces_found(&at(json!({}), 0.0), &CancelToken::new())
            .expect("faces");
        println!("== {id} ({w} x {h}), {faces} face(s), on the device ==");
        for (what, zoom) in [("fit", false), ("100 %", true)] {
            // The view itself is warmed first (the raw group of a 100 % view, the colour kernel), without the node; the first frame
            // with the node then builds the plan of each region and compiles the face kernel, and the rest are warm.
            let _ = frame(&r, &base, zoom);
            let first = frame(&r, &at(live(0), 0.0), zoom);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let base_moves: Vec<f64> = (1..=3)
                .map(|i| frame(&r, &at(inert.clone(), 0.1 * f64::from(i) + 0.05), zoom))
                .collect();
            let node_moves: Vec<f64> = (1..=3)
                .map(|i| frame(&r, &at(live(i), 0.0), zoom))
                .collect();
            let upstream: Vec<f64> = (1..=3)
                .map(|i| frame(&r, &at(live(1), 0.1 * f64::from(i)), zoom))
                .collect();
            println!(
                "{what}: first frame with the node {first:.0} ms; the same upstream moves with the node at 0: {:.0} ms; after a slider of the node {:.0} ms; after a corrector upstream {:.0} ms",
                mean(&base_moves),
                mean(&node_moves),
                mean(&upstream)
            );
        }
        // The export tail of the 24 MP photo (PNG, 16 bit), with the node and without, on the device.
        let out = tempfile::tempdir().expect("dir");
        let export = |r: Renderer, s: PhotoSource| {
            let ctx = ExportContext::new(r).with_home(Some(out.path().to_path_buf()));
            let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
            photo.source = s;
            let t = Instant::now();
            let done = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(ExportOptions {
                    format: FormatKind::Png,
                    bit_depth: BitDepth::Sixteen,
                    color_space: ColorSpace::Srgb,
                    destination: Some(out.path().to_path_buf()),
                    if_exists: IfExists::Replace,
                    ..ExportOptions::default()
                }),
                &ExportControl::none(),
            )
            .expect("export");
            (ms(t), done.device)
        };
        let (plain, dev_a) = export(make(Some(Arc::clone(&gpu))), base.clone());
        let (with, dev_b) = export(make(Some(Arc::clone(&gpu))), at(live(2), 0.0));
        println!(
            "export of 24 MP on the device ({dev_a:?}, {dev_b:?}): without the node {plain:.0} ms, with it {with:.0} ms (its own analysis included)"
        );
    }
}
