//! How fast a viewer frame with a Film node is, on the device (F-08e; `docs/24-film.md` section 3.9c "Speed").
//!
//! * `a_viewer_frame_with_a_film_node_is_cheap_by_the_counters`: counters, not wall time (stable on Mesa lavapipe in CI).
//!   (1) A viewer frame runs no colour or Film pass in the compensated arithmetic of F-08d (`precise.wgsl`), which made
//!   the passes about twice as dear and is for exports: the counter `precise_dispatches` is 0 for every viewer frame and
//!   positive for an export of the same photo. (2) Once a Film frame has been made, a second frame of the same photo after
//!   a change of the Film node's exposure, of the Main node's exposure or of a node after the Film node compiles no
//!   pipeline, bakes no table, uploads no Film table (a few hundred KiB at most: the uniforms and the node records), and
//!   makes one submission and one readback.
//! * `measure` (ignored): the table of the F-08e report, on a real file. `FILM_SPEED_ARW` names the file,
//!   `FILM_SPEED_DPR` the device pixel ratio of the window (1 or 2).
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::film::FilmEngine;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";

/// The Film engine and its baker are the process's (`FilmEngine::shared`), so their counters count every test that renders a
/// Film node in this binary. The tests that read them one frame at a time run one after the other: with two tests on two
/// threads (the Mac CI's `--test-threads=2`), the other test's first export bakes the Rec.2020/sRGB tables of its own output
/// inside a measured frame, "a table was baked" for a frame that baked nothing (CI on c581338, all three Mac jobs).
static ENGINE: Mutex<()> = Mutex::new(());

fn engine_alone() -> std::sync::MutexGuard<'static, ()> {
    ENGINE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig {
        timestamps: std::env::var("FILM_SPEED_LOG").is_ok(),
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
                "\n* GPU TESTS SKIPPED in film_speed: {e}\n"
            );
            None
        }
    }
}

fn renderer(gpu: Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
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

/// Input, Main (exposure `main_ev`), an optional Film node (exposure `film_ev`), a corrector (exposure `after_ev`), Output.
fn graph(film: bool, main_ev: f64, film_ev: f64, after_ev: f64) -> DevelopSettings {
    let mut items = vec![
        json!({ "id": 1, "trim": { "exposure_ev": main_ev } }),
        json!({ "id": 3, "kind": "corrector", "pos": [2, 0], "trim": { "exposure_ev": after_ev } }),
    ];
    let mut links = Vec::new();
    let mut prev = json!(1);
    links.push(json!({ "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } }));
    if film {
        items.push(
            json!({ "id": 2, "kind": "film", "pos": [1, 0], "film": { "exposure_ev": film_ev } }),
        );
        links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": 2, "port": "picture" } }));
        prev = json!(2);
    }
    links.push(json!({ "from": { "node": prev, "port": "picture" }, "to": { "node": 3, "port": "picture" } }));
    links.push(json!({ "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } }));
    settings_of(json!({ "nodes": { "items": items, "links": links } }))
}

/// Input, Main, a corrector with a Brush mask (Refine edges on), a Film node, a corrector, Output: a raster mask row
/// upstream of the Film node (F-08g).
fn graph_raster(main_ev: f64, film_ev: f64, after_ev: f64) -> DevelopSettings {
    let link = |a: Value, b: Value| json!({ "from": { "node": a, "port": "picture" }, "to": { "node": b, "port": "picture" } });
    settings_of(json!({ "nodes": { "items": [
        { "id": 1, "trim": { "exposure_ev": main_ev } },
        { "id": 4, "kind": "corrector", "pos": [1, 0], "trim": { "exposure_ev": 0.8 },
          "masks": { "invert": false, "items": [ { "kind": "brush", "mode": "add",
            "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.35, 0.45], [0.6, 0.55]] }],
            "feather": 0.3, "refine": 0.5 } ] } },
        { "id": 2, "kind": "film", "pos": [2, 0], "film": { "exposure_ev": film_ev } },
        { "id": 3, "kind": "corrector", "pos": [3, 0], "trim": { "exposure_ev": after_ev } },
    ], "links": [
        link(json!("input"), json!(1)), link(json!(1), json!(4)), link(json!(4), json!(2)),
        link(json!(2), json!(3)), link(json!(3), json!("output")),
    ] } }))
}

fn view(w: f64, h: f64, dpr: f64, percent_100: bool) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(w, h, dpr));
    if percent_100 {
        st.mode = ViewportMode::Percent { value: 100.0 };
    }
    st
}

struct Seen {
    ms: f64,
    size: (usize, usize),
    uploads: u64,
    uploaded_bytes: u64,
    pipelines: u64,
    dispatches: u64,
    submissions: u64,
    readbacks: u64,
    builds: (u64, u64),
    baked: u64,
    /// Which tables: T1, curves, T3a, T3b.
    baked_kinds: [u64; 4],
    precise: u64,
}

fn timed(r: &Renderer, gpu: &Gpu, src: &PhotoSource, st: &ViewportState) -> Seen {
    let engine = FilmEngine::shared(&data_root()).expect("engine");
    let before = gpu.counters();
    let built = engine.built();
    let baked = engine.baker().built();
    let t = Instant::now();
    let f = r
        .viewport_frame(
            src,
            st,
            &FrameOptions {
                no_cache: true,
                ..FrameOptions::default()
            },
            &CancelToken::new(),
        )
        .expect("frame");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(f.report.device, ComputeDevice::Gpu);
    assert!(!f.report.device_refused);
    let d = gpu.counters().since(before);
    let now = engine.built();
    Seen {
        ms,
        size: (f.width, f.height),
        uploads: d.uploads,
        uploaded_bytes: d.uploaded_bytes,
        pipelines: d.pipelines,
        dispatches: d.dispatches,
        submissions: d.submissions,
        readbacks: d.readbacks,
        builds: (now.0 - built.0, now.1 - built.1),
        precise: d.precise_dispatches,
        baked_kinds: {
            let b = engine.baker().built();
            [
                b.t1 - baked.t1,
                b.curves - baked.curves,
                b.t3a - baked.t3a,
                b.t3b - baked.t3b,
            ]
        },
        baked: {
            let b = engine.baker().built();
            (b.t1 - baked.t1)
                + (b.curves - baked.curves)
                + (b.t3a - baked.t3a)
                + (b.t3b - baked.t3b)
        },
    }
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[film speed] {text}");
}

fn row(what: &str, s: &Seen) {
    say(&format!(
        "{what:<34} {:>8.1} ms  {}x{}  uploads {} / {:.2} MiB  pipelines {}  dispatches {}  submissions {}  readbacks {}  chains+models built {:?}  bakes {} (t1, curves, t3a, t3b {:?})",
        s.ms,
        s.size.0,
        s.size.1,
        s.uploads,
        s.uploaded_bytes as f64 / 1_048_576.0,
        s.pipelines,
        s.dispatches,
        s.submissions,
        s.readbacks,
        s.builds,
        s.baked,
        s.baked_kinds
    ));
}

fn open_real() -> Developed {
    let path = std::env::var("FILM_SPEED_ARW").unwrap_or_else(|_| {
        format!(
            "{}/../../testdata/lens-photos/14077__DSC5565.ARW",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let dir = tempfile::tempdir().expect("dir");
    std::fs::copy(&path, dir.path().join("DSC00001.ARW")).expect("copy the ARW");
    Developed::open(dir)
}

fn small() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

#[test]
fn a_viewer_frame_with_a_film_node_is_cheap_by_the_counters() {
    let _alone = engine_alone();
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(Arc::clone(&gpu));
    let base = d.source();
    let st = view(300.0, 200.0, 1.0, false);
    let precise = |gpu: &Gpu, src: &PhotoSource| -> u64 {
        let before = gpu.counters();
        let _ = timed(&r, gpu, src, &st);
        gpu.counters().since(before).precise_dispatches
    };
    // (1) Views are plain f32; the first Film frame also compiles and uploads what it needs.
    let film = base.with_settings(graph(true, 0.0, 0.0, 0.0));
    assert_eq!(precise(&gpu, &base), 0, "a frame without a Film node");
    assert_eq!(precise(&gpu, &film), 0, "the first frame with a Film node");
    // (2) Warm frames: a change of the Film node, of the Main node and of a node after the Film node.
    for (what, settings) in [
        ("the same frame", graph(true, 0.0, 0.0, 0.0)),
        ("a Film exposure", graph(true, 0.0, 0.5, 0.0)),
        ("a Main exposure", graph(true, 0.4, 0.5, 0.0)),
        ("a node after the Film node", graph(true, 0.4, 0.5, 0.3)),
    ] {
        let s = timed(&r, &gpu, &base.with_settings(settings), &st);
        row(what, &s);
        assert_eq!(s.pipelines, 0, "{what}: a pipeline was compiled");
        assert_eq!(
            s.baked, 0,
            "{what}: a table was baked (t1, curves, t3a, t3b {:?})",
            s.baked_kinds
        );
        assert!(
            s.uploaded_bytes < 1 << 20,
            "{what}: {} bytes were uploaded (the Film tables are tens of MiB)",
            s.uploaded_bytes
        );
        assert_eq!(s.submissions, 1, "{what}: the frame is one submission");
        assert_eq!(s.readbacks, 1, "{what}: the frame is one readback");
        assert_eq!(s.precise, 0, "{what}: a compensated pass ran in a view");
    }
    // The export of the same photo is the compensated arithmetic (F-08d): the counter sees it.
    let home = tempfile::tempdir().expect("dir");
    let ctx =
        ExportContext::new(renderer(Arc::clone(&gpu))).with_home(Some(home.path().to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = photo.source.with_settings(graph(true, 0.0, 0.0, 0.0));
    let opts = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Eight,
        destination: Some(home.path().to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let before = gpu.counters();
    let report = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(opts),
        &ExportControl::none(),
    )
    .expect("export");
    assert_eq!(report.device, ComputeDevice::Gpu);
    assert!(
        gpu.counters().since(before).precise_dispatches > 0,
        "the export runs the compensated passes"
    );
}

/// A raster mask row before the Film node (F-08g): the frame is the device's, and once one has been made a change of the
/// Film node, of Main or of a node after the Film node compiles nothing, bakes nothing and uploads no table: the raster
/// planes are copied from the ones kept on the device, and the planes of the Film node's prefix chain too.
#[test]
fn a_viewer_frame_with_a_raster_mask_before_the_film_node_is_cheap_by_the_counters() {
    let _alone = engine_alone();
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(Arc::clone(&gpu));
    let base = d.source();
    for percent_100 in [false, true] {
        let st = view(300.0, 200.0, 1.0, percent_100);
        let first = timed(
            &r,
            &gpu,
            &base.with_settings(graph_raster(0.0, 0.0, 0.0)),
            &st,
        );
        row("first raster frame", &first);
        let mut plain_ms = f64::MAX;
        for _ in 0..3 {
            let s = timed(
                &r,
                &gpu,
                &base.with_settings(graph(true, 0.0, 0.0, 0.0)),
                &st,
            );
            plain_ms = plain_ms.min(s.ms);
        }
        say(&format!(
            "100 % {percent_100}: a Film frame without the raster mask {plain_ms:.1} ms"
        ));
        for (what, settings) in [
            ("the same frame", graph_raster(0.0, 0.0, 0.0)),
            ("a Film exposure", graph_raster(0.0, 0.5, 0.0)),
            ("a Main exposure", graph_raster(0.4, 0.5, 0.0)),
            ("a node after the Film node", graph_raster(0.4, 0.5, 0.3)),
        ] {
            let s = timed(&r, &gpu, &base.with_settings(settings), &st);
            row(what, &s);
            assert_eq!(s.pipelines, 0, "{what}: a pipeline was compiled");
            assert_eq!(
                s.baked, 0,
                "{what}: a table was baked (t1, curves, t3a, t3b {:?})",
                s.baked_kinds
            );
            assert!(
                s.uploaded_bytes < 1 << 20,
                "{what}: {} bytes were uploaded",
                s.uploaded_bytes
            );
            assert_eq!(s.submissions, 1, "{what}: the frame is one submission");
            assert_eq!(s.readbacks, 1, "{what}: the frame is one readback");
            assert_eq!(s.precise, 0, "{what}: a compensated pass ran in a view");
        }
    }
}

#[test]
#[ignore = "a measurement, not a check"]
fn measure() {
    let _ = captured_log();
    ym_core::perf::set_enabled(true);
    let Some(gpu) = gpu() else { return };
    let d = open_real();
    let _ = MODEL;
    let r = renderer(Arc::clone(&gpu));
    let dpr: f64 = std::env::var("FILM_SPEED_DPR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let base = d.source();
    for (name, st) in [
        ("fit 1440x900", view(1440.0, 900.0, dpr, false)),
        ("100 %", view(1440.0, 900.0, dpr, true)),
    ] {
        if std::env::var("FILM_SPEED_VIEW").is_ok_and(|v| !name.contains(&v)) {
            continue;
        }
        say(&format!("==== {name}, device pixel ratio {dpr}"));
        let plain = base.with_settings(graph(false, 0.0, 0.0, 0.0));
        row("no film, cold", &timed(&r, &gpu, &plain, &st));
        row("no film, warm", &timed(&r, &gpu, &plain, &st));
        row(
            "no film, main exposure",
            &timed(
                &r,
                &gpu,
                &base.with_settings(graph(false, 0.3, 0.0, 0.0)),
                &st,
            ),
        );
        let film = base.with_settings(graph(true, 0.0, 0.0, 0.0));
        row("film, cold", &timed(&r, &gpu, &film, &st));
        row("film, warm same", &timed(&r, &gpu, &film, &st));
        row(
            "film, film exposure",
            &timed(
                &r,
                &gpu,
                &base.with_settings(graph(true, 0.0, 0.5, 0.0)),
                &st,
            ),
        );
        row(
            "film, main exposure",
            &timed(
                &r,
                &gpu,
                &base.with_settings(graph(true, 0.4, 0.5, 0.0)),
                &st,
            ),
        );
        row(
            "film, node after",
            &timed(
                &r,
                &gpu,
                &base.with_settings(graph(true, 0.4, 0.5, 0.3)),
                &st,
            ),
        );
        row(
            "film, node after again",
            &timed(
                &r,
                &gpu,
                &base.with_settings(graph(true, 0.4, 0.5, 0.6)),
                &st,
            ),
        );
    }
    for line in captured_log()
        .lines()
        .filter(|l| l.contains("ym::perf") && std::env::var("FILM_SPEED_LOG").is_ok())
    {
        say(line.split("ym::perf:").nth(1).unwrap_or(line).trim());
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A hash of every viewer frame of a few graphs (with and without Film nodes), for comparing two builds of the device
/// code: a view returns to the bits it had before F-08d. Run on both commits and compare the lines.
#[test]
#[ignore = "a tool: prints hashes of the viewer frames"]
fn view_hashes() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    let d = Developed::new(&scene, MODEL);
    let r = renderer(Arc::clone(&gpu));
    let busy = json!({
        "grain": { "amount": 150, "size": 100, "seed": 7 },
        "halation": { "amount": 200, "size": 120 },
        "strength": 80,
        "masks": { "items": [ { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 } ] },
    });
    let mut one = json!({ "id": 2, "kind": "film", "pos": [1.0, 0.0] });
    one.as_object_mut()
        .unwrap()
        .extend(busy.as_object().unwrap().clone());
    let one_film = settings_of(json!({ "nodes": { "items": [ { "id": 1 }, one ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } },
    ] } }));
    let slide = settings_of(json!({ "nodes": { "items": [
        { "id": 1 },
        { "id": 2, "kind": "film", "pos": [1, 0], "film": { "stock": "fujifilm_velvia_100" } },
        { "id": 3, "kind": "corrector", "pos": [2, 0], "trim": { "exposure_ev": 0.4 } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
        { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } }));
    let base = d.source();
    for (what, settings) in [
        ("default", None),
        ("no film, exposure", Some(graph(false, 0.3, 0.0, 0.2))),
        ("film, plain", Some(graph(true, 0.0, 0.0, 0.0))),
        ("film, exposures", Some(graph(true, 0.3, 0.5, 0.2))),
        ("film, busy", Some(one_film)),
        ("slide and corrector", Some(slide)),
    ] {
        let src = match settings {
            Some(s) => base.with_settings(s),
            None => base.clone(),
        };
        for percent in [false, true] {
            let f = r
                .viewport_frame(
                    &src,
                    &view(300.0, 200.0, 1.0, percent),
                    &FrameOptions {
                        no_cache: true,
                        ..FrameOptions::default()
                    },
                    &CancelToken::new(),
                )
                .expect("frame");
            say(&format!(
                "HASH {what}, 100 % {percent}: {:016x} ({}x{}, {:?})",
                fnv(&f.rgba),
                f.width,
                f.height,
                f.report.device
            ));
        }
    }
}
