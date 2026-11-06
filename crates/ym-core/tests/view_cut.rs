//! How a zoomed view with raster masks and a Film node is cut for the device (G-12; `docs/23-nodes.md` section 5.12,
//! `docs/17-gpu.md` section 8).
//!
//! * `measure` (ignored): the owner's case on a real 24 MP photo (`VIEW_CUT_ARW`, by default a 24 MP file of
//!   `testdata/lens-photos`): a 2080x1387 viewport at 57 %, 92 %, 106 % and 116 %, raster rows and a Film node. Prints, for
//!   each, the frame time, the pieces (`color_passes` of the `ym::perf` line), the uploads, the dispatches and the
//!   submissions. `VIEW_CUT_BYTES` caps the cut (`view_band_bytes`), `VIEW_CUT_ZOOMS` names the zooms, `VIEW_CUT_HOST=1` keeps the raw group on the host (the camera RGB of every piece is uploaded: the owner's Mac), `VIEW_CUT_STAGES=all` runs every stage
//!   (`auto`, the stages `Auto` runs on a Mac, by default).
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;
use std::time::Instant;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in view_cut: {e}\n"
            );
            None
        }
    }
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[view cut] {text}");
}

fn settings_of(doc: Value) -> DevelopSettings {
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

/// Input, Main, a corrector with two raster rows (Brush, refined), a Film node, a corrector, Output: the owner's graph
/// (the Background row needs a model; a Brush row has the same footprint).
fn owner_graph() -> DevelopSettings {
    // The halation's size sets the Film node's reach: 145 gives about 650 pixels of the photo, the owner's log.
    let halation_size: u32 = std::env::var("VIEW_CUT_HALATION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(145);
    let link = |a: Value, b: Value| json!({ "from": { "node": a, "port": "picture" }, "to": { "node": b, "port": "picture" } });
    settings_of(json!({ "nodes": { "items": [
        { "id": 1, "trim": { "exposure_ev": 0.0 } },
        { "id": 4, "kind": "corrector", "pos": [1, 0], "trim": { "exposure_ev": 0.8 },
          "masks": { "invert": false, "items": [
            { "kind": "brush", "mode": "add",
              "strokes": [{ "size": 0.2, "hardness": 0.5, "erase": false, "points": [[0.35, 0.45], [0.6, 0.55]] }],
              "feather": 0.3, "refine": 0.5 },
            { "kind": "brush", "mode": "add",
              "strokes": [{ "size": 0.3, "hardness": 0.5, "erase": false, "points": [[0.2, 0.3], [0.4, 0.7]] }],
              "feather": 0.3, "refine": 0.5 } ] } },
        { "id": 2, "kind": "film", "pos": [2, 0], "film": { "exposure_ev": 0.0 },
          "halation": { "size": halation_size } },
        { "id": 3, "kind": "corrector", "pos": [3, 0], "trim": { "exposure_ev": 0.0 } },
    ], "links": [
        link(json!("input"), json!(1)), link(json!(1), json!(4)), link(json!(4), json!(2)),
        link(json!(2), json!(3)), link(json!(3), json!("output")),
    ] } }))
}

fn open_real() -> Developed {
    let path = std::env::var("VIEW_CUT_ARW").unwrap_or_else(|_| {
        format!(
            "{}/../../testdata/lens-photos/14077__DSC5565.ARW",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let dir = tempfile::tempdir().expect("dir");
    std::fs::copy(&path, dir.path().join("DSC00001.ARW")).expect("copy the ARW");
    Developed::open(dir)
}

fn view(w: f64, h: f64, percent: f64) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(w, h, 1.0));
    st.mode = ViewportMode::Percent { value: percent };
    st
}

fn renderer(gpu: &Arc<Gpu>, band_bytes: Option<u64>) -> Renderer {
    let defaults = RenderOptions::default();
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: if std::env::var("VIEW_CUT_STAGES").is_ok_and(|v| v == "all") {
                GpuStages::ALL
            } else {
                GpuStages::SIGNED_OFF
            },
            data_root: Some(data_root()),
            view_band_bytes: band_bytes.unwrap_or(defaults.view_band_bytes),
            resident_caches: std::env::var("VIEW_CUT_HOST").is_err(),
            ..defaults
        },
    )
    .expect("render thread")
}

/// What one frame cost.
struct Seen {
    ms: f64,
    size: (usize, usize),
    pieces: u64,
    windows: String,
    uploads: u64,
    uploaded_mib: f64,
    dispatches: u64,
    submissions: u64,
    peak_mib: f64,
    device: ComputeDevice,
    reason: Option<String>,
}

/// The number after `key=` in the last `frame` line of the `ym::perf` log since `from` bytes.
fn perf_number(log: &str, key: &str) -> u64 {
    log.lines()
        .rev()
        .find(|l| l.contains("ym::perf: frame"))
        .and_then(|l| l.split(&format!("{key}=")).nth(1))
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// The word after `key=` in the last `frame` line of the `ym::perf` log.
fn perf_word(log: &str, key: &str) -> String {
    log.lines()
        .rev()
        .find(|l| l.contains("ym::perf: frame"))
        .and_then(|l| l.split(&format!("{key}=")).nth(1))
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or("?")
        .to_owned()
}

fn timed(r: &Renderer, gpu: &Gpu, src: &PhotoSource, st: &ViewportState) -> Seen {
    let before = gpu.counters();
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        ctx.pool().reset_peaks();
        Ok(())
    })
    .expect("reset");
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
    let d = gpu.counters().since(before);
    Seen {
        ms,
        size: (f.width, f.height),
        pieces: perf_number(&captured_log(), "color_passes"),
        windows: format!(
            "raster {} film {}",
            perf_word(&captured_log(), "raster_window"),
            perf_word(&captured_log(), "film_window")
        ),
        uploads: d.uploads,
        uploaded_mib: d.uploaded_bytes as f64 / 1_048_576.0,
        dispatches: d.dispatches,
        submissions: d.submissions,
        peak_mib: gpu
            .run(JobSpec::new(Priority::Viewport), |ctx, _| {
                Ok(ctx.pool().stats().peak_in_use_bytes)
            })
            .expect("stats") as f64
            / 1_048_576.0,
        device: f.report.device,
        reason: f.report.cpu_reason.clone(),
    }
}

fn row(what: &str, s: &Seen) {
    say(&format!(
        "{what:<26} {:>8.1} ms  {}x{}  pieces {:>3}  uploads {} / {:.1} MiB  dispatches {}  submissions {}  peak in use {:.0} MiB  {}  {:?}{}",
        s.ms,
        s.size.0,
        s.size.1,
        s.pieces,
        s.uploads,
        s.uploaded_mib,
        s.dispatches,
        s.submissions,
        s.peak_mib,
        s.windows,
        s.device,
        s.reason
            .as_ref()
            .map(|r| format!("  ({r})"))
            .unwrap_or_default(),
    ));
}

#[test]
#[ignore = "a measurement, not a check"]
fn measure() {
    let _ = captured_log();
    ym_core::perf::set_enabled(true);
    let Some(gpu) = gpu() else { return };
    let d = open_real();
    let bytes = std::env::var("VIEW_CUT_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let r = renderer(&gpu, bytes);
    let base = d.source();
    let zooms: Vec<f64> = std::env::var("VIEW_CUT_ZOOMS")
        .unwrap_or_else(|_| "57,92,106,116".to_owned())
        .split(',')
        .filter_map(|z| z.trim().parse().ok())
        .collect();
    let src = base.with_settings(owner_graph());
    for zoom in zooms {
        let st = view(2080.0, 1387.0, zoom);
        // The first frame of a zoom builds the raw group, the tables and the pipelines; the second is what a
        // person sees on the next move.
        let cold = timed(&r, &gpu, &src, &st);
        row(&format!("{zoom} % cold"), &cold);
        let warm = timed(&r, &gpu, &src, &st);
        row(&format!("{zoom} % warm"), &warm);
    }
}

// ---- the cut (G-12) -------------------------------------------------------------------------------------------------

const MODEL: &str = "ILCE-7RM4";

fn small() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn frame_of(
    r: &Renderer,
    src: &PhotoSource,
    st: &ViewportState,
) -> ym_core::pipeline::RenderedFrame {
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

fn limited(gpu: &Arc<Gpu>, cap: u64, limits: Option<ym_core::pipeline::ViewLimits>) -> Renderer {
    let defaults = RenderOptions::default();
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::ALL,
            data_root: Some(data_root()),
            view_band_bytes: cap,
            view_limits: limits,
            ..defaults
        },
    )
    .expect("render thread")
}

fn worst_code(a: &ym_core::pipeline::RenderedFrame, b: &ym_core::pipeline::RenderedFrame) -> u8 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.rgba
        .iter()
        .zip(b.rgba.iter())
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// The owner's graph (a raster row and a Film node) at zooms around 100 % and at three places of the photo, with the
/// device's limits cut to a few hundred KB: every view is made on the device ("the route must be the device's": a piece the
/// cut judged to fit is a piece whose buffers fit), within a code of the view in one piece, and a cut is never more than 16
/// pieces unless no cut of 16 fits.
#[test]
fn a_cut_view_with_raster_rows_and_a_film_node_stays_on_the_device_at_any_zoom_and_place() {
    let _ = captured_log();
    ym_core::perf::set_enabled(true);
    let Some(gpu) = gpu() else { return };
    let d = small();
    let src = d.source().with_settings(owner_graph());
    let whole = limited(&gpu, u64::MAX, None);
    let cut = limited(&gpu, 2_000_000, None);
    for percent in [57.0, 90.0, 100.0, 106.0, 116.0, 130.0] {
        for centre in [[0.5, 0.5], [0.1, 0.15], [0.9, 0.85]] {
            let mut st = view(300.0, 200.0, percent);
            st.centre = centre;
            let a = frame_of(&whole, &src, &st);
            let pieces_whole = perf_number(&captured_log(), "color_passes");
            let b = frame_of(&cut, &src, &st);
            let pieces = perf_number(&captured_log(), "color_passes");
            let what = format!("{percent} % at {centre:?}");
            assert_eq!(
                a.report.device,
                ComputeDevice::Gpu,
                "{what}: {:?}",
                a.report.notes
            );
            assert_eq!(
                b.report.device,
                ComputeDevice::Gpu,
                "{what} cut: {:?}",
                b.report.notes
            );
            assert!(
                !b.report.device_refused,
                "{what}: {:?}",
                b.report.cpu_reason
            );
            assert!(pieces >= pieces_whole, "{what}: {pieces} pieces");
            let worst = worst_code(&a, &b);
            say(&format!(
                "{what}: whole {pieces_whole} piece(s), cut {pieces} pieces, worst {worst} codes"
            ));
            assert!(worst <= 1, "{what}: the pieces differ by {worst} codes");
        }
    }
}

/// A device with a binding of a gigabyte and the 4 GiB budget the app gives the Mac cuts nothing the owner looks at: the
/// owner's geometry (a 2080 x 1387 viewport of a 24 MP photo at 50 to 150 %, a Film reach of 655 pixels and the raster
/// masks' 75) is one piece. The binding cannot be had on lavapipe, so this is the cut's arithmetic: the limits, the sizes
/// the route takes from them and the wider pictures the scheduler infers for the owner's views.
#[test]
fn the_owners_views_are_one_piece_at_a_gigabyte() {
    use ym_core::pipeline::ViewLimits;
    const MIB: u64 = 1 << 20;
    let mac = ViewLimits {
        binding_bytes: 1024 * MIB,
        budget_bytes: 4096 * MIB,
    };
    let (view_w, view_h, reach) = (2080u64, 1387u64, 655 + 75);
    for percent in [50u64, 57, 70, 87, 92, 100, 106, 116, 120, 150] {
        // Below 100 % the picture is the view itself and the reach counts in view pixels; above, the 100 % picture under
        // the view (`view / zoom` pixels) and the reach in photo pixels.
        let (w, h) = if percent <= 100 {
            (view_w, view_h)
        } else {
            (view_w * 100 / percent, view_h * 100 / percent)
        };
        let reach_px = if percent <= 100 {
            reach * percent / 100
        } else {
            reach
        };
        let wide = 16 * (w + 2 * reach_px) * (h + 2 * reach_px);
        let raster_binding = 28 * (w + 2 * reach_px) * (h + 2 * reach_px);
        assert!(
            wide <= mac.picture_bytes(u64::MAX) && raster_binding <= mac.binding_bytes,
            "{percent} %: {wide} bytes of wider picture against {} (and {raster_binding} against the binding)",
            mac.picture_bytes(u64::MAX)
        );
    }
}
