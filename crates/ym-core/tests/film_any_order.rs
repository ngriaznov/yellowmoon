//! A Film node in any place of the graph is made on the device (F-08g; `docs/24-film.md` section 3.9c, `docs/17-gpu.md`
//! section 9). On the owner's Mac every frame and export of a photo with a raster mask (Brush, Background, Object, Subject,
//! People) on a node before a Film node fell back to the CPU (`invalid kernel input: film: a raster mask row upstream of a
//! Film node ...`), and the Before view of a photo with a Film node was refused once (`the plan runs a Film node and the job
//! has no Film passes for it`). No valid graph with a Film node may be refused by the device.
//!
//! The sweep builds graphs with a masked node before the Film node, on it, after it, between two Film nodes, behind a key link,
//! with Bypass and Strength 0, and checks each:
//!
//! * the viewport frame (Fit and 100 %; the photo, the Before view, Overlay and Mask only on the masked node) is the device's,
//!   is not refused, carries no "graphics card" note, and matches the CPU's within the Film bounds of `render_film_device.rs`
//!   (an 8-bit frame: 2 codes, and few pixels beyond one);
//! * the export (PNG, 16 bit) is the device's and matches the CPU's within one 10-bit code; at another tile size it is the
//!   same bytes (raster cases);
//! * the thumbnail, the node thumbnails, a stock tile, the histogram and the histogram at a node make no device refusal (the
//!   log has no refusal line) and run device work.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, MaskStyle, MaskView, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_core::vision::{FakeVision, VisionMasks};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
/// One 10-bit code after C13, in the codes of a 16-bit file.
const TEN_BIT_CODE: u32 = 64;

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
                "\n* GPU TESTS SKIPPED in film_any_order: {e}\n"
            );
            None
        }
    }
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
            vision: Some(VisionMasks::new(Arc::new(FakeVision::new()))),
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

// ---- graphs ------------------------------------------------------------------------------------------------

/// One node after Main: a corrector or a Film node, with mask rows, and what else its JSON carries.
struct Spec {
    film: bool,
    rows: Vec<Value>,
    extra: Value,
}

fn corrector(rows: Vec<Value>) -> Spec {
    Spec {
        film: false,
        rows,
        extra: json!({}),
    }
}

fn film(rows: Vec<Value>) -> Spec {
    Spec {
        film: true,
        rows,
        extra: json!({}),
    }
}

fn row(kind: &str, extra: Value) -> Value {
    let mut r = json!({ "kind": kind, "mode": "add" });
    if let (Some(a), Some(b)) = (r.as_object_mut(), extra.as_object()) {
        a.extend(b.clone());
    }
    r
}

fn brush() -> Value {
    row(
        "brush",
        json!({ "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.35, 0.45], [0.6, 0.55]] }],
                "feather": 0.3, "refine": 0.5 }),
    )
}
fn background() -> Value {
    row("background", json!({}))
}
fn subject() -> Value {
    row("subject", json!({}))
}
fn linear() -> Value {
    row(
        "linear",
        json!({ "center": [0.5, 0.5], "angle_deg": 20.0, "feather": 0.3 }),
    )
}
fn radial() -> Value {
    row(
        "radial",
        json!({ "center": [0.5, 0.5], "radii": [0.3, 0.2], "angle_deg": 0.0, "feather": 0.3 }),
    )
}
fn colour_range() -> Value {
    row(
        "color_range",
        json!({ "center": [0.05, 0.0], "width": 0.2, "softness": 0.5 }),
    )
}

/// Input, Main, the nodes in a line (ids 2, 3, ...), Output. `keys` are key links `(from, to)` between the nodes' ids.
fn line(specs: &[Spec], keys: &[(u32, u32)]) -> DevelopSettings {
    let mut items = vec![json!({ "id": 1 })];
    let mut links =
        vec![json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } })];
    let mut prev = json!(1);
    for (i, s) in specs.iter().enumerate() {
        let id = 2 + i as u32;
        let mut n = if s.film {
            json!({ "id": id, "kind": "film", "pos": [id, 0.0],
                    "film": { "exposure_ev": 0.3 },
                    "grain": { "amount": 100, "seed": id },
                    "halation": { "amount": 200, "size": 120 },
                    "strength": 85 })
        } else {
            json!({ "id": id, "kind": "corrector", "pos": [id, 0.0], "trim": { "exposure_ev": 0.8 } })
        };
        if !s.rows.is_empty() {
            n["masks"] = json!({ "invert": false, "items": s.rows });
        }
        if let (Some(a), Some(b)) = (n.as_object_mut(), s.extra.as_object()) {
            a.extend(b.clone());
        }
        items.push(n);
        links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": id, "port": "picture" } }));
        prev = json!(id);
    }
    links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
    for (a, b) in keys {
        links.push(json!({ "from": { "node": a, "port": "key" }, "to": { "node": b, "port": "key" } }));
    }
    serde_json::from_value::<DevelopSettings>(json!({ "nodes": { "items": items, "links": links } }))
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// A case: its graph, the node the mask view looks at (when it has masks), and whether the export is checked too.
struct Case {
    name: String,
    settings: DevelopSettings,
    masked: Option<u32>,
    export: bool,
    /// Whether the CPU's frame is a reference (Subject masks are made by a background worker: no).
    compare: bool,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let kinds: [(&str, fn() -> Value, bool); 5] = [
        ("brush", brush, true),
        ("background", background, true),
        ("linear", linear, false),
        ("colour range", colour_range, false),
        ("subject", subject, false),
    ];
    for (k, make, export) in kinds {
        let compare = k != "subject";
        let mut add = |place: &str, specs: Vec<Spec>, masked: u32, export: bool| {
            out.push(Case {
                name: format!("{k} {place}"),
                settings: line(&specs, &[]),
                masked: Some(masked),
                export: export && compare,
                compare,
            });
        };
        add(
            "on a corrector before the Film node",
            vec![corrector(vec![make()]), film(vec![])],
            2,
            export,
        );
        add(
            "on the Film node",
            vec![film(vec![make()])],
            2,
            export,
        );
        add(
            "on a corrector after the Film node",
            vec![film(vec![]), corrector(vec![make()])],
            3,
            export,
        );
        add(
            "between two Film nodes",
            vec![film(vec![]), corrector(vec![make()]), film(vec![])],
            3,
            false,
        );
        add(
            "before and on the Film node and after it",
            vec![
                corrector(vec![make()]),
                film(vec![make()]),
                corrector(vec![make()]),
            ],
            2,
            false,
        );
    }
    // Two raster rows on the node before the Film node, one of them guided; a radial on the Film node.
    out.push(Case {
        name: "brush and background before a Film node with a radial on it".into(),
        settings: line(
            &[
                corrector(vec![brush(), background()]),
                film(vec![radial()]),
            ],
            &[],
        ),
        masked: Some(2),
        export: true,
        compare: true,
    });
    // A key link from the masked node to the Film node (the Film node's own mask reads it).
    out.push(Case {
        name: "a key link from a brush node to the Film node".into(),
        settings: line(&[corrector(vec![brush()]), film(vec![])], &[(2, 3)]),
        masked: Some(2),
        export: false,
        compare: true,
    });
    // A Film node that does nothing (Bypass, Strength 0) after a masked node.
    for (what, extra) in [
        ("bypassed", json!({ "bypass": true })),
        ("at Strength 0", json!({ "strength": 0 })),
    ] {
        out.push(Case {
            name: format!("a brush node, then a Film node {what}"),
            settings: line(
                &[
                    corrector(vec![brush()]),
                    Spec {
                        film: true,
                        rows: vec![],
                        extra,
                    },
                ],
                &[],
            ),
            masked: Some(2),
            export: true,
            compare: true,
        });
    }
    // No mask at all, two Film nodes and a corrector after.
    out.push(Case {
        name: "two Film nodes and a corrector, no mask".into(),
        settings: line(&[film(vec![]), film(vec![]), corrector(vec![])], &[]),
        masked: None,
        export: false,
        compare: true,
    });
    out
}

// ---- frames ------------------------------------------------------------------------------------------------

fn view(zoom_100: bool, before: bool, mask: Option<(u32, MaskStyle, Option<u32>)>) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    if zoom_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st.before = before;
    st.mask_view = mask.map(|(node, style, row)| MaskView { node, style, row });
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

fn frame_diff(a: &RenderedFrame, b: &RenderedFrame) -> (u8, usize) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut max, mut over) = (0u8, 0usize);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        over += usize::from(d > 1);
    }
    (max, over)
}

fn decode_png16(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read the export");
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
        .collect()
}

fn export(
    d: &Developed,
    settings: &DevelopSettings,
    r: Renderer,
    out: &Path,
) -> (ComputeDevice, Vec<String>, Vec<u32>) {
    let ctx = ExportContext::new(r).with_home(Some(out.to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = photo.source.with_settings(settings.clone());
    let opts = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        color_space: ColorSpace::Srgb,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let report = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(opts),
        &ExportControl::none(),
    )
    .expect("export");
    (report.device, report.notes.clone(), decode_png16(&report.path))
}

/// The lines of the log that say the device refused or failed something.
fn refusals(log: &str) -> Vec<String> {
    log.lines()
        .filter(|l| {
            l.contains("refused by an input check")
                || l.contains("failed on the device")
                || l.contains("failed in the export tail")
                || l.contains("the device failed")
        })
        .map(str::to_owned)
        .collect()
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[film any order] {text}");
}

#[test]
fn no_graph_with_a_film_node_is_refused_by_the_device() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let cpu = renderer(None, 256);
    let mut problems: Vec<String> = Vec::new();
    let mut worst_view = (0u8, 0usize);
    let mut worst_export = 0u32;
    for case in cases() {
        // A renderer of its own: a refusal is logged as a warning once per photo.
        let device = renderer(Some(Arc::clone(&gpu)), 256);
        let src = d.source().with_settings(case.settings.clone());
        let log_before = captured_log().len();
        let mut states: Vec<(String, ViewportState)> = Vec::new();
        for zoom_100 in [false, true] {
            states.push((format!("100 % {zoom_100}"), view(zoom_100, false, None)));
            states.push((format!("Before, 100 % {zoom_100}"), view(zoom_100, true, None)));
            if let Some(node) = case.masked {
                for (style, name) in [(MaskStyle::Overlay, "Overlay"), (MaskStyle::MaskOnly, "Mask only")] {
                    states.push((
                        format!("{name}, 100 % {zoom_100}"),
                        view(zoom_100, false, Some((node, style, None))),
                    ));
                }
                states.push((
                    format!("Overlay row 0, 100 % {zoom_100}"),
                    view(zoom_100, false, Some((node, MaskStyle::Overlay, Some(0)))),
                ));
            }
        }
        // The first frame of a Background row makes its blur map; the frames compared are the ones after it.
        let _ = frame(&device, &src, &states[0].1);
        for (what, st) in &states {
            let at = format!("{}: {what}", case.name);
            let got = frame(&device, &src, st);
            if got.report.device != ComputeDevice::Gpu {
                problems.push(format!("{at}: the frame is {:?}", got.report.device));
            }
            if got.report.device_refused {
                problems.push(format!("{at}: the device refused the frame"));
            }
            for n in got.report.notes.iter().filter(|n| n.contains("graphics card")) {
                problems.push(format!("{at}: note {n}"));
            }
            if case.compare {
                let _ = frame(&cpu, &src, st);
                let want = frame(&cpu, &src, st);
                let (max, over) = frame_diff(&got, &want);
                worst_view = (worst_view.0.max(max), worst_view.1.max(over));
                if max > 2 || over >= got.rgba.len() / 2000 {
                    problems.push(format!("{at}: against the CPU {max} codes, {over} values over 1"));
                }
            }
        }
        // The thumbnail, the node thumbnails, a stock tile and the histograms.
        let cancel = CancelToken::new();
        let counters = gpu.counters();
        device.thumbnail_png(&src, &cancel).expect("thumbnail");
        device.node_thumbnails(&src, &cancel).expect("node thumbnails");
        if let Some(film_node) = case.settings.nodes.films.keys().next().copied() {
            ym_core::film_tiles::film_tile_png(&device, &src, film_node, "kodak_portra_400", &cancel)
                .expect("stock tile");
        }
        device.histogram(&src, &cancel).expect("histogram");
        device.histogram_at(&src, None, &cancel).expect("histogram at");
        if gpu.counters().since(counters).dispatches == 0 {
            problems.push(format!("{}: the thumbnails ran no device work", case.name));
        }
        // The export, and its tiles.
        if case.export {
            let home = tempfile::tempdir().expect("dir");
            let (dev_on, notes, dev) = export(&d, &case.settings, renderer(Some(Arc::clone(&gpu)), 256), home.path());
            if dev_on != ComputeDevice::Gpu {
                problems.push(format!("{}: the export is {dev_on:?}", case.name));
            }
            if notes.iter().any(|n| n.contains("graphics card")) {
                problems.push(format!("{}: export notes {notes:?}", case.name));
            }
            let (_, _, cpu_codes) = export(&d, &case.settings, renderer(None, 256), home.path());
            let max = dev.iter().zip(&cpu_codes).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
            worst_export = worst_export.max(max);
            if max > TEN_BIT_CODE {
                problems.push(format!("{}: the export differs from the CPU's by {max} of 65535", case.name));
            }
            for tile in [97usize, 1024] {
                let (_, _, other) = export(&d, &case.settings, renderer(Some(Arc::clone(&gpu)), tile), home.path());
                if other != dev {
                    problems.push(format!("{}: the export at tile {tile} is other bytes", case.name));
                }
            }
        }
        let log = captured_log();
        for line in refusals(&log[log_before.min(log.len())..]) {
            problems.push(format!("{}: log: {line}", case.name));
        }
        say(&format!("{}: done", case.name));
    }
    say(&format!(
        "worst view {} codes, {} values over 1; worst export {worst_export} of 65535",
        worst_view.0, worst_view.1
    ));
    assert!(problems.is_empty(), "{problems:#?}");
}
