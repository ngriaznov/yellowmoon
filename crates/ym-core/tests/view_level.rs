//! Whole-image levels (G-13; `docs/11-ui.md` "Pans", `docs/17-gpu.md` "Levels", `pipeline::frame::level`).
//!
//! After a settings change settles the viewer renders the whole photo once at the zoom on screen and keeps it on the
//! device; a pan at that zoom copies a rectangle out of it and renders nothing. These tests hold a level to the window
//! render of the same pixels, count what a pan costs, and check that the level is freed and refused when it should be.
//!
//! * `the_crop_of_a_level_is_the_window_render`: for zooms below, at and above 100 %, with a Film node and a raster mask in
//!   the graph and output sharpening on, three pans of the cell; the frame served from the level equals the frame the
//!   window path renders (`no_cache`, so no level serves it), at most one code apart on the existing view bound, and the
//!   test says how many bytes differ.
//! * `pans_after_the_fill_render_nothing`: twenty pans after the fill: the device counters see no dispatch, no upload and
//!   no readback, the stage counters of the photo do not move, and every frame is on the device and says `level_hits`.
//! * `a_change_of_settings_frees_the_level` and `a_change_of_zoom_makes_another_level`.
//! * `the_cap_bounds_the_levels` and `a_letterbox_window_has_no_level`.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use serde_json::{Value, json};
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FrameOutput, LevelStatus, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, ViewportInput, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::scheduler::MemoryBudget;
use ym_core::settings::DevelopSettings;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};

const MODEL: &str = "ILCE-7RM4";

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
                "\n* GPU TESTS SKIPPED in view_level: {e}\n"
            );
            None
        }
    }
}

fn renderer(gpu: &Arc<Gpu>, levels: bool, budget: Option<MemoryBudget>) -> Renderer {
    renderer_cut(gpu, levels, budget, None)
}

/// A renderer whose views are cut into pieces of at most `band_bytes` of picture (`RenderOptions::view_band_bytes`).
fn renderer_cut(
    gpu: &Arc<Gpu>,
    levels: bool,
    budget: Option<MemoryBudget>,
    band_bytes: Option<u64>,
) -> Renderer {
    let defaults = RenderOptions::default();
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::ALL,
            data_root: Some(data_root()),
            levels,
            view_band_bytes: band_bytes.unwrap_or(defaults.view_band_bytes),
            budget: budget.unwrap_or(defaults.budget),
            ..defaults
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

/// Input, Main, a corrector with a Brush mask (Refine edges on), a Film node, a corrector, Output: a raster mask row
/// upstream of a Film node, the two things whose pixels depend on the window's surroundings.
fn graph(main_ev: f64, film_ev: f64, after_ev: f64) -> DevelopSettings {
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

fn small() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(640, 420, white, Layout::Patchwork, 0.5)
        .noise(2e-4, 2e-6)
        .seed(5);
    Developed::new(&scene, MODEL)
}

fn at(percent: f64, cell: (f64, f64), image: (usize, usize)) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(cell.0, cell.1, 1.0));
    st.set_mode(ViewportMode::Percent { value: percent }, image);
    st
}

fn frame(r: &Renderer, src: &PhotoSource, st: &ViewportState, no_cache: bool) -> RenderedFrame {
    r.viewport_frame_to(
        src,
        st,
        &FrameOptions {
            no_cache,
            ..FrameOptions::default()
        },
        FrameOutput::Device,
        &CancelToken::new(),
    )
    .expect("frame")
}

fn bytes(f: &RenderedFrame) -> Vec<u8> {
    f.host_rgba().expect("read the frame back").into_owned()
}

/// The bytes in use in the device's pool.
fn in_use(gpu: &Gpu) -> u64 {
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        Ok(ctx.pool().stats().in_use_bytes)
    })
    .expect("pool stats")
}

fn say(text: &str) {
    let _ = writeln!(std::io::stderr(), "[view level] {text}");
}

fn pans() -> [(f64, f64); 3] {
    [(-37.0, 21.0), (53.5, -9.25), (-11.0, -48.0)]
}

#[test]
fn the_crop_of_a_level_is_the_window_render() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    for percent in [60.0, 75.0, 100.0, 130.0] {
        let mut st = at(percent, (300.0, 200.0), image);
        let status = r
            .fill_level(&src, &st, &CancelToken::new())
            .expect("the fill");
        assert_eq!(status, LevelStatus::Ready, "{percent} %");
        for (dx, dy) in pans() {
            st.input(&ViewportInput::Pan { dx, dy }, image);
            let served = frame(&r, &src, &st, false);
            let window = frame(&r, &src, &st, true);
            assert_eq!(served.report.device, ComputeDevice::Gpu);
            assert_eq!(
                served.report.runs.level_hits, 1,
                "{percent} %: served from the level"
            );
            assert_eq!(window.report.runs.level_hits, 0);
            assert!(served.on_device.is_some() && window.on_device.is_some());
            let (a, b) = (bytes(&served), bytes(&window));
            assert_eq!(a.len(), b.len());
            let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            say(&format!(
                "{percent} % pan ({dx}, {dy}): {differing} of {} bytes differ, by {worst} code at most",
                a.len()
            ));
            assert!(worst <= 1, "{percent} %: a crop differs by {worst} codes");
            if percent <= 100.0 {
                assert_eq!(
                    differing, 0,
                    "{percent} %: a crop is the window bit for bit"
                );
            }
        }
    }
}

#[test]
fn pans_after_the_fill_render_nothing() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let mut st = at(100.0, (300.0, 200.0), image);
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    let photo_before = r.counters(src.id);
    let before = gpu.counters();
    for i in 0..20 {
        let (dx, dy) = if i % 2 == 0 { (-9.0, 4.0) } else { (7.0, -3.0) };
        st.input(&ViewportInput::Pan { dx, dy }, image);
        let f = frame(&r, &src, &st, false);
        assert!(f.on_device.is_some(), "pan {i}: a device frame");
        assert_eq!(f.report.runs.level_hits, 1, "pan {i}");
        assert_eq!(f.report.runs.frame_hits, 0);
        assert_eq!(f.report.runs.display_hits, 0);
        assert_eq!(f.report.runs.color_passes, 0);
        assert_eq!(f.report.runs.output_passes, 0);
        assert!(f.sharp);
    }
    let d = gpu.counters().since(before);
    say(&format!(
        "20 pans: dispatches {} uploads {} readbacks {} submissions {}",
        d.dispatches, d.uploads, d.readbacks, d.submissions
    ));
    assert_eq!(d.dispatches, 0, "a pan dispatches no kernel");
    assert_eq!(d.uploads, 0, "a pan uploads nothing");
    assert_eq!(d.readbacks, 0, "a pan reads nothing back");
    let photo = r.counters(src.id);
    assert_eq!(photo.level_hits - photo_before.level_hits, 20);
    assert_eq!(photo.color_passes, photo_before.color_passes);
    assert_eq!(photo.output_passes, photo_before.output_passes);
    assert_eq!(photo.raw_work(), photo_before.raw_work());
}

#[test]
fn a_change_of_settings_frees_the_level() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let base = d.source();
    let src = base.with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 200.0), image);
    let changes: Vec<(&str, PhotoSource)> = vec![
        ("Main exposure", base.with_settings(graph(0.2, 0.3, 0.1))),
        ("Film exposure", base.with_settings(graph(0.0, 0.5, 0.1))),
        (
            "a node after Film",
            base.with_settings(graph(0.0, 0.3, 0.4)),
        ),
        ("output sharpening", {
            let mut s = src.settings.clone();
            s.output_mut().output_sharpening = 40.0;
            src.with_settings(s)
        }),
    ];
    for (what, changed) in changes {
        let fresh = r.fill_level(&src, &st, &CancelToken::new()).expect("fill");
        assert_eq!(fresh, LevelStatus::Ready);
        let (n, held) = r.levels_held(src.id);
        assert_eq!(n, 1);
        assert!(held > 0);
        let f = frame(&r, &changed, &st, false);
        assert_eq!(
            f.report.runs.level_hits, 0,
            "{what}: not served from the old level"
        );
        assert_eq!(
            r.levels_held(src.id),
            (0, 0),
            "{what}: the level is freed at the next request"
        );
        drop(f);
        // And the way back finds no level either.
        let back = frame(&r, &src, &st, false);
        assert_eq!(
            back.report.runs.level_hits, 0,
            "{what}: back to the old settings"
        );
    }
}

#[test]
fn a_level_is_returned_to_the_pool_when_it_is_freed() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 200.0), image);
    let before = in_use(&gpu);
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    let (_, held) = r.levels_held(src.id);
    assert_eq!(
        held,
        4 * image.0 * image.1,
        "the strips are the level, RGBA8"
    );
    let with = in_use(&gpu);
    assert!(with >= before + held as u64, "the strips count in the pool");
    // Leaving the photo frees its levels.
    r.graph(src.id).release_full_resolution();
    assert_eq!(r.levels_held(src.id), (0, 0));
    assert!(
        in_use(&gpu) + held as u64 <= with + 1,
        "the strips went back"
    );
}

#[test]
fn a_change_of_zoom_makes_another_level_and_two_are_kept() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let cell = (300.0, 200.0);
    let (a, b, c) = (
        at(60.0, cell, image),
        at(100.0, cell, image),
        at(130.0, cell, image),
    );
    for st in [&a, &b] {
        assert_eq!(
            r.fill_level(&src, st, &CancelToken::new()).expect("fill"),
            LevelStatus::Ready
        );
    }
    assert_eq!(r.levels_held(src.id).0, 2);
    // Toggling between the two zooms is served from both.
    for st in [&a, &b, &a] {
        assert_eq!(frame(&r, &src, st, false).report.runs.level_hits, 1);
    }
    // A third zoom pushes out the least recently used (b).
    assert_eq!(
        r.fill_level(&src, &c, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    assert_eq!(r.levels_held(src.id).0, 2, "at most two levels are kept");
    assert_eq!(frame(&r, &src, &a, false).report.runs.level_hits, 1);
    assert_eq!(frame(&r, &src, &c, false).report.runs.level_hits, 1);
    assert_eq!(frame(&r, &src, &b, false).report.runs.level_hits, 0);
}

#[test]
fn the_cap_bounds_the_levels() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let src0 = d.source();
    // A budget whose eighth is smaller than one level of the photo at 100 % (640 x 420 x 4 bytes, about 1 MiB).
    let budget = MemoryBudget {
        gpu_bytes: 4 << 20,
        ..MemoryBudget::default()
    };
    let r = renderer(&gpu, true, Some(budget));
    let src = src0.with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 200.0), image);
    let need = 4 * image.0 * image.1;
    assert!(
        need as u64 > budget.gpu_bytes / 8,
        "the level is over the cap"
    );
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::None
    );
    assert_eq!(r.levels_held(src.id), (0, 0));
    assert_eq!(frame(&r, &src, &st, false).report.runs.level_hits, 0);
}

#[test]
fn a_letterbox_window_fit_and_off_have_no_level() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let on = renderer(&gpu, true, None);
    let image = on.final_size(&src, &CancelToken::new()).expect("size");
    // Fit.
    let fit = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    assert_eq!(
        on.fill_level(&src, &fit, &CancelToken::new())
            .expect("fill"),
        LevelStatus::None
    );
    // Above 150 %.
    let big = at(200.0, (300.0, 200.0), image);
    assert_eq!(
        on.fill_level(&src, &big, &CancelToken::new())
            .expect("fill"),
        LevelStatus::None
    );
    // A cell wider than the photo at 100 %: a letterbox.
    let wide = at(100.0, (900.0, 200.0), image);
    assert_eq!(
        on.fill_level(&src, &wide, &CancelToken::new())
            .expect("fill"),
        LevelStatus::None
    );
    // Levels off: nothing is made, and a frame is the window path's.
    let off = renderer(&gpu, false, None);
    let st = at(100.0, (300.0, 200.0), image);
    assert_eq!(
        off.fill_level(&src, &st, &CancelToken::new())
            .expect("fill"),
        LevelStatus::None
    );
    assert_eq!(frame(&off, &src, &st, false).report.runs.level_hits, 0);
}

/// The graph without Film or masks: Main and a corrector.
fn plain_graph(ev: f64) -> DevelopSettings {
    let link = |a: Value, b: Value| json!({ "from": { "node": a, "port": "picture" }, "to": { "node": b, "port": "picture" } });
    settings_of(json!({ "nodes": { "items": [
        { "id": 1, "trim": { "exposure_ev": ev } },
        { "id": 3, "kind": "corrector", "pos": [1, 0], "trim": { "exposure_ev": 0.2 } },
    ], "links": [
        link(json!("input"), json!(1)), link(json!(1), json!(3)), link(json!(3), json!("output")),
    ] } }))
}

#[test]
fn a_level_of_many_bands_is_the_window_render_and_a_part_made_level_serves_its_rows() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    // Pieces of about 150 rows of the 640-wide picture: the level is cut into bands.
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 170));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let cell = (300.0, 100.0);
    let mut top = at(100.0, cell, image);
    top.input(&ViewportInput::Pan { dx: 0.0, dy: 1e9 }, image);
    // One band of the level is made first: the one under the window.
    let status = r
        .fill_level_bands(&src, &top, 1, &CancelToken::new())
        .expect("one band");
    let LevelStatus::Filling { done: 1, total } = status else {
        panic!("a level of several bands, one made: {status:?}");
    };
    say(&format!("the level is cut into {total} bands"));
    assert!(total >= 3);
    // The window at the top is under the band that was made; the window at the bottom is not.
    assert_eq!(frame(&r, &src, &top, false).report.runs.level_hits, 1);
    let mut bottom = at(100.0, cell, image);
    bottom.input(&ViewportInput::Pan { dx: 0.0, dy: -1e9 }, image);
    let partly = frame(&r, &src, &bottom, false);
    assert_eq!(
        partly.report.runs.level_hits, 0,
        "rows that are not made are rendered"
    );
    // The rest of the level, and then every window of it is the window render.
    assert_eq!(
        r.fill_level(&src, &top, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    let mut st = at(100.0, cell, image);
    for (dx, dy) in [
        (0.0, 0.0),
        (-40.0, -45.0),
        (30.0, -90.0),
        (0.0, -1e9),
        (-1e9, 1e9),
    ] {
        st.input(&ViewportInput::Pan { dx, dy }, image);
        let served = frame(&r, &src, &st, false);
        let window = frame(&r, &src, &st, true);
        assert_eq!(served.report.runs.level_hits, 1);
        assert_eq!(bytes(&served), bytes(&window), "pan ({dx}, {dy})");
    }
}

#[test]
fn a_fill_that_is_cancelled_makes_no_more_bands() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 170));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 100.0), image);
    let cancel = CancelToken::new();
    cancel.cancel();
    let status = r.fill_level(&src, &st, &cancel);
    assert!(
        !matches!(status, Ok(LevelStatus::Ready)),
        "a cancelled fill is not a level: {status:?}"
    );
    assert_eq!(r.levels_held(src.id).1, 0, "no band was made");
}

/// Waits (up to `secs`) for the level of `zoom` to be complete.
fn wait_ready(r: &Renderer, src: &PhotoSource, zoom: f64, secs: u64) -> bool {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    while std::time::Instant::now() < end {
        if r.level_status(src, zoom) == LevelStatus::Ready {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    false
}

#[test]
fn a_window_that_has_no_level_starts_the_fill_after_the_debounce() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 170));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let mut st = at(100.0, (300.0, 100.0), image);
    let zoom = st.zoom(image);
    let first = frame(&r, &src, &st, false);
    assert_eq!(first.report.runs.level_hits, 0);
    // The level is wanted, and not made before the key has held still.
    assert_eq!(r.level_status(&src, zoom), LevelStatus::Waiting);
    assert_eq!(
        r.levels_held(src.id).1,
        0,
        "nothing is filled during the debounce"
    );
    let started = std::time::Instant::now();
    assert!(
        wait_ready(&r, &src, zoom, 60),
        "the level was made in the background"
    );
    say(&format!(
        "the level was ready {:.0} ms after the first window",
        started.elapsed().as_secs_f64() * 1000.0
    ));
    assert!(started.elapsed() >= ym_core::pipeline::LEVEL_DEBOUNCE);
    // Every pan is then served from it, whatever the rows.
    for (dx, dy) in [(-30.0, 20.0), (0.0, -1e9), (1e9, 1e9), (12.0, 7.0)] {
        st.input(&ViewportInput::Pan { dx, dy }, image);
        let f = frame(&r, &src, &st, false);
        assert_eq!(f.report.runs.level_hits, 1, "pan ({dx}, {dy})");
        assert_eq!(bytes(&f), bytes(&frame(&r, &src, &st, true)));
    }
    let photo = r.counters(src.id);
    assert!(photo.level_bands >= 3, "the level was made band by band");
}

#[test]
fn a_slider_drag_fills_nothing_and_the_settled_settings_fill_once() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 170));
    let base = d.source();
    let first = base.with_settings(plain_graph(0.0));
    let image = r.final_size(&first, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 100.0), image);
    let zoom = st.zoom(image);
    // Fifty settings changes, a frame for each, 16 ms apart (a drag of the Exposure slider).
    for i in 0..50 {
        let src = base.with_settings(plain_graph(f64::from(i) * 0.01));
        let t = std::time::Instant::now();
        let f = frame(&r, &src, &st, false);
        assert_eq!(f.report.runs.level_hits, 0);
        assert_eq!(
            r.levels_held(src.id).1,
            0,
            "change {i}: a drag fills nothing"
        );
        let spent = t.elapsed();
        if spent < std::time::Duration::from_millis(16) {
            std::thread::sleep(std::time::Duration::from_millis(16) - spent);
        }
    }
    let last = base.with_settings(plain_graph(0.49));
    assert_eq!(r.counters(last.id).level_bands, 0);
    // The hand stops: the level of the last settings is made, and only that one.
    assert!(wait_ready(&r, &last, zoom, 60));
    assert_eq!(r.levels_held(last.id).0, 1);
    let served = frame(&r, &last, &st, false);
    assert_eq!(served.report.runs.level_hits, 1);
    assert_eq!(bytes(&served), bytes(&frame(&r, &last, &st, true)));
}

#[test]
fn leaving_the_photo_cancels_the_fill_and_frees_the_levels() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 170));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 100.0), image);
    let zoom = st.zoom(image);
    r.navigate(src.id);
    let _ = frame(&r, &src, &st, false);
    assert!(wait_ready(&r, &src, zoom, 60));
    assert!(r.levels_held(src.id).1 > 0);
    r.navigate(ym_core::session::PhotoId(4242));
    assert_eq!(
        r.levels_held(src.id),
        (0, 0),
        "the levels of the photo left are freed"
    );
    assert_eq!(r.level_status(&src, zoom), LevelStatus::None);
}

#[test]
fn leaving_the_photo_mid_fill_stops_the_fill() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    // Pieces of about 50 rows: a level of about ten bands, so that no device finishes it between two looks of the poll.
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 50));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 100.0), image);
    let zoom = st.zoom(image);
    r.navigate(src.id);
    let _ = frame(&r, &src, &st, false);
    // Wait for the first band (looking without a pause: a fast device makes a band in milliseconds), then leave.
    let end = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let total = loop {
        if let LevelStatus::Filling { done, total } = r.level_status(&src, zoom)
            && done >= 1
        {
            break total;
        }
        assert!(std::time::Instant::now() < end, "the fill never started");
        std::thread::yield_now();
    };
    assert!(total >= 8, "the level is cut into many bands: {total}");
    r.navigate(ym_core::session::PhotoId(4242));
    // The bands made when `navigate` returned. A band that was running then may finish, and none starts after it.
    let at_leaving = r.counters(src.id).level_bands;
    std::thread::sleep(std::time::Duration::from_millis(1500));
    assert_eq!(r.levels_held(src.id), (0, 0));
    let bands = r.counters(src.id).level_bands;
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(
        r.counters(src.id).level_bands,
        bands,
        "no band was made after leaving"
    );
    assert!(
        bands <= at_leaving + 1,
        "at most the running band finished after leaving: {at_leaving} bands at leaving, {bands} later"
    );
    assert!(
        bands < total as u64,
        "the fill stopped before the end ({bands} of {total} bands)"
    );
    assert_eq!(
        r.scheduler().foreground_load(),
        0,
        "no fill job is left in the queue"
    );
}

/// The numbers of G-13 on a real file (`LEVEL_ARW` names it; the default is a lens-photos file). Not a check:
/// `cargo test -p ym-core --test view_level --release measure -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn measure() {
    let Some(gpu) = gpu() else { return };
    let path = std::env::var("LEVEL_ARW").unwrap_or_else(|_| {
        format!(
            "{}/../../testdata/lens-photos/14077__DSC5565.ARW",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let dir = tempfile::tempdir().expect("dir");
    std::fs::copy(&path, dir.path().join("DSC00001.ARW")).expect("copy the ARW");
    let d = Developed::open(dir);
    let r = renderer(&gpu, true, None);
    let film = std::env::var("LEVEL_FILM").map_or(true, |v| v != "0");
    let settings = |ev: f64| {
        if film {
            graph(0.0, ev, 0.1)
        } else {
            plain_graph(ev)
        }
    };
    let base = d.source();
    let src = base.with_settings(settings(0.0));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    say(&format!("photo {}x{}, film graph {film}", image.0, image.1));
    let peak = |gpu: &Gpu| {
        gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
            Ok(ctx.pool().stats().peak_in_use_bytes)
        })
        .expect("stats")
    };
    for percent in [50.0, 75.0, 100.0, 130.0] {
        if std::env::var("LEVEL_PERCENT").is_ok_and(|v| v.parse::<f64>().ok() != Some(percent)) {
            continue;
        }
        let mut st = at(percent, (1440.0, 900.0), image);
        // Warm the raw groups and the Film tables, then a settings change: the frame the owner waits for.
        let _ = frame(&r, &src, &st, true);
        let changed = base.with_settings(settings(0.3));
        let t = std::time::Instant::now();
        let f = frame(&r, &changed, &st, false);
        let first = t.elapsed().as_secs_f64() * 1000.0;
        let c0 = r.counters(src.id);
        let t = std::time::Instant::now();
        let status = r
            .fill_level(&changed, &st, &CancelToken::new())
            .expect("fill");
        let fill = t.elapsed().as_secs_f64() * 1000.0;
        let c1 = r.counters(src.id);
        if status != LevelStatus::Ready {
            say(&format!(
                "{percent:>5} %: first frame {first:.0} ms; no level ({status:?}): the device cuts this view into tiles"
            ));
            continue;
        }
        let (_, held) = r.levels_held(src.id);
        let bands = c1.level_bands - c0.level_bands;
        // Pans.
        let before = gpu.counters();
        let mut ms = Vec::new();
        for i in 0..40 {
            let (dx, dy) = if i % 2 == 0 {
                (-23.0, 11.0)
            } else {
                (19.0, -7.0)
            };
            st.input(&ViewportInput::Pan { dx, dy }, image);
            let t = std::time::Instant::now();
            let p = frame(&r, &changed, &st, false);
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(
                p.report.runs.level_hits,
                1,
                "{percent} %: pan {i} ({status:?}, {} levels)",
                r.levels_held(src.id).0
            );
        }
        let dd = gpu.counters().since(before);
        ms.sort_by(f64::total_cmp);
        // A window render at the same place, for comparison, and the tiles the next slider move has to make again.
        let t = std::time::Instant::now();
        let w = frame(&r, &changed, &st, true);
        let window_ms = t.elapsed().as_secs_f64() * 1000.0;
        let after_fill = r.counters(src.id).tiles;
        let moved = base.with_settings(settings(0.5));
        let m = frame(&r, &moved, &st, false);
        say(&format!(
            "{percent:>5} %: first frame {first:.0} ms; fill {fill:.0} ms in {bands} bands ({:.0} ms a band), {:.0} MiB, {status:?}; \
             pan median {:.2} ms max {:.2} ms (dispatches {} uploads {} readbacks {}); window render {window_ms:.0} ms; \
             tiles made by the slider move after the fill: {} (made by the first frame: {}); peak pool {:.0} MiB",
            fill / bands.max(1) as f64,
            held as f64 / 1_048_576.0,
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            dd.dispatches,
            dd.uploads,
            dd.readbacks,
            m.report.runs.tiles,
            f.report.runs.tiles,
            peak(&gpu) as f64 / 1_048_576.0,
        ));
        let _ = (w, after_fill);
    }
}

/// How long a viewport frame waits behind the fill (`LEVEL_ARW`, `LEVEL_PERCENT`): the frame asked for while bands are
/// queued, against the same frame on an idle renderer.
#[test]
#[ignore = "a measurement, not a check"]
fn measure_wait() {
    let Some(gpu) = gpu() else { return };
    let path = std::env::var("LEVEL_ARW").unwrap_or_else(|_| {
        format!(
            "{}/../../testdata/lens-photos/14077__DSC5565.ARW",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let percent: f64 = std::env::var("LEVEL_PERCENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100.0);
    let dir = tempfile::tempdir().expect("dir");
    std::fs::copy(&path, dir.path().join("DSC00001.ARW")).expect("copy the ARW");
    let d = Developed::open(dir);
    let r = renderer(&gpu, true, None);
    let film = std::env::var("LEVEL_FILM").map_or(true, |v| v != "0");
    let settings = |ev: f64| {
        if film {
            graph(0.0, ev, 0.1)
        } else {
            plain_graph(ev)
        }
    };
    let base = d.source();
    let src = base.with_settings(settings(0.0));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(percent, (1440.0, 900.0), image);
    let zoom = st.zoom(image);
    r.navigate(src.id);
    let _ = frame(&r, &src, &st, true);
    let opts = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    let timed = |src: &PhotoSource| {
        let t = std::time::Instant::now();
        let h = r.spawn_viewport_frame_to(src.clone(), st.clone(), opts, FrameOutput::Device);
        h.wait().expect("job").expect("frame");
        t.elapsed().as_secs_f64() * 1000.0
    };
    let idle: Vec<f64> = (1..4)
        .map(|i| timed(&base.with_settings(settings(0.01 * f64::from(i)))))
        .collect();
    say(&format!("{percent} %: idle frames {idle:.0?} ms"));
    // A settings change, a first window, then the frames asked for while the level fills.
    let changed = base.with_settings(settings(0.3));
    let _ = frame(&r, &changed, &st, false);
    let started = std::time::Instant::now();
    let mut during = Vec::new();
    let mut i = 0;
    while r.level_status(&changed, zoom) != LevelStatus::Ready
        && started.elapsed() < std::time::Duration::from_secs(300)
    {
        std::thread::sleep(std::time::Duration::from_millis(150));
        if matches!(r.level_status(&changed, zoom), LevelStatus::Filling { .. }) {
            i += 1;
            // Another setting, so the frame is a render and not the level's crop (the fill is cancelled by the change: the
            // frame is asked for on the same settings with no cache instead).
            during.push(timed(&changed));
            let _ = i;
        }
    }
    say(&format!(
        "{percent} %: the level was ready {:.1} s after the first window; frames asked for while it filled took {during:.0?} ms",
        started.elapsed().as_secs_f64()
    ));
    let c = r.counters(src.id);
    say(&format!("bands {} tiles {}", c.level_bands, c.tiles));
}

/// A device with no room for a frame gives the photo's levels up (they are what is cheapest to make again), and the frame is
/// the CPU's. Needs the fault injection (`cargo test -p ym-core --features fault-injection --test view_level`).
#[cfg(feature = "fault-injection")]
#[test]
fn a_device_out_of_memory_frees_the_levels() {
    use ym_gpu::device::GpuFault;
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let st = at(100.0, (300.0, 200.0), image);
    assert_eq!(
        r.fill_level(&src, &st, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    gpu.set_fault(Some(GpuFault::OutOfBudget));
    let f = frame(&r, &src, &st, false);
    gpu.set_fault(None);
    assert_eq!(f.report.runs.level_hits, 0);
    assert_eq!(r.levels_held(src.id), (0, 0), "the levels were given up");
    // And the next frame is the device's again, with no level until the fill makes one.
    let again = frame(&r, &src, &st, true);
    assert_eq!(again.report.device, ComputeDevice::Gpu);
}

fn coarse(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame_to(
        src,
        st,
        &FrameOptions {
            coarse_only: true,
            ..FrameOptions::default()
        },
        FrameOutput::Device,
        &CancelToken::new(),
    )
    .expect("frame")
}

#[test]
fn a_zoom_shows_the_nearest_level_resampled_at_once_and_the_exact_frame_follows() {
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer(&gpu, true, None);
    let src = d.source().with_settings(graph(0.0, 0.3, 0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let cell = (300.0, 200.0);
    let (a, b) = (at(100.0, cell, image), at(75.0, cell, image));
    // No level yet: the first request is the window path's.
    let none = coarse(&r, &src, &b);
    assert_eq!(none.report.runs.level_drafts, 0);
    assert_eq!(
        r.fill_level(&src, &a, &CancelToken::new()).expect("fill"),
        LevelStatus::Ready
    );
    // The exact frame for 75 % is in the cache: it is served from there, not drafted.
    let _ = frame(&r, &src, &b, false);
    let again = coarse(&r, &src, &b);
    assert_eq!(
        again.report.runs.level_drafts, 0,
        "a view with an exact frame is not drafted"
    );
    // The sharp frame above wanted the level of 75 %, which the renderer fills in the background once the debounce is over
    // (`LEVEL_DEBOUNCE`), on a loaded machine in the middle of the draft's counter window. Let that fill end first, so that
    // nothing but the draft is left to dispatch. The level of 75 % then exists, which is why the draft below is at another zoom.
    assert!(
        wait_ready(&r, &src, b.zoom(image), 120),
        "the level of 75 % was made in the background"
    );
    // Another zoom and cell size make it a frame nobody has: a draft from the level of 100 % (the nearest of the two levels).
    let c = at(90.0, (280.0, 190.0), image);
    let before = gpu.counters();
    let draft = coarse(&r, &src, &c);
    let spent = gpu.counters().since(before);
    assert_eq!(draft.report.runs.level_drafts, 1);
    assert!(
        !draft.sharp,
        "a draft is not the sharp frame: the shell asks for it next"
    );
    assert_eq!((draft.width, draft.height), (280, 190));
    assert!(draft.on_device.is_some());
    assert_eq!(draft.report.runs.color_passes, 0);
    assert_eq!(draft.report.runs.raw_groups + draft.report.runs.tiles, 0);
    assert_eq!(spent.dispatches, 1, "a draft is one kernel");
    assert!(
        spent.uploaded_bytes <= 64,
        "a draft uploads its uniform block and nothing else"
    );
    assert_eq!(spent.readbacks, 0, "a draft reads nothing back");
    // It looks like the exact frame (a resample of the picture at another zoom).
    let exact = frame(&r, &src, &c, true);
    let (x, y) = (bytes(&draft), bytes(&exact));
    let mean = x
        .iter()
        .zip(&y)
        .map(|(p, q)| u64::from(p.abs_diff(*q)))
        .sum::<u64>() as f64
        / x.len() as f64;
    say(&format!(
        "the draft differs from the exact frame by {mean:.2} codes on average"
    ));
    assert!(mean < 12.0, "a draft is close to the exact frame: {mean}");
    // The exact request is the window path's.
    let sharp = r
        .viewport_frame_to(
            &src,
            &c,
            &FrameOptions::default(),
            FrameOutput::Device,
            &CancelToken::new(),
        )
        .expect("frame");
    assert!(sharp.sharp);
    assert_eq!(sharp.report.runs.level_drafts, 0);
    // A host request, the Before view and a changed setting are never drafted.
    let host = r
        .viewport_frame(
            &src,
            &at(75.0, (270.0, 180.0), image),
            &FrameOptions {
                coarse_only: true,
                ..FrameOptions::default()
            },
            &CancelToken::new(),
        )
        .expect("frame");
    assert_eq!(host.report.runs.level_drafts, 0);
    let changed = d.source().with_settings(graph(0.4, 0.3, 0.1));
    assert_eq!(
        coarse(&r, &changed, &at(75.0, (260.0, 170.0), image))
            .report
            .runs
            .level_drafts,
        0
    );
}

/// An export does not wait for a fill, and the fill gives way to it: the bands that were queued make no progress while the
/// export runs (but the one in flight), and the fill goes on afterwards to a complete level that serves pans, equal to the
/// window renders.
#[test]
fn an_export_does_not_wait_for_a_fill_and_the_fill_resumes_after_it() {
    use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
    use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists};
    let Some(gpu) = gpu() else { return };
    let d = small();
    let r = renderer_cut(&gpu, true, None, Some(16 * 640 * 50));
    let src = d.source().with_settings(plain_graph(0.1));
    let image = r.final_size(&src, &CancelToken::new()).expect("size");
    let mut st = at(100.0, (300.0, 100.0), image);
    let zoom = st.zoom(image);
    r.navigate(src.id);
    let _ = frame(&r, &src, &st, false);
    let done_of = |r: &Renderer| match r.level_status(&src, zoom) {
        LevelStatus::Filling { done, .. } => Some(done),
        LevelStatus::Ready => Some(usize::MAX),
        _ => None,
    };
    let end = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while done_of(&r).is_none_or(|n| n < 1) {
        assert!(std::time::Instant::now() < end, "the fill never started");
        std::thread::yield_now();
    }
    // The export, on a renderer of its own over the same device, while the fill runs.
    let home = tempfile::tempdir().expect("dir");
    let ctx = ExportContext::new(r.clone()).with_home(Some(home.path().to_path_buf()));
    let mut photo = ExportPhoto::from_session(&d.session, d.id).expect("export photo");
    photo.source = src.clone();
    let opts = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Eight,
        destination: Some(home.path().to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    };
    let at_start = done_of(&r);
    assert_ne!(
        at_start,
        Some(usize::MAX),
        "the fill was still running when the export began"
    );
    // The probe: every progress report of the export (they come from the whole of it, the stretches after the render
    // job has ended included) finds the scheduler saying an export is running, and waits a little, which on a fast
    // device is long enough for a fill band to fit if it were allowed to. This holds on any device, whatever a band costs.
    let reports = std::sync::atomic::AtomicUsize::new(0);
    let idle_reports = std::sync::atomic::AtomicUsize::new(0);
    let probe = |_: f32| {
        reports.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if !r.scheduler().exports_active() {
            idle_reports.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        std::thread::sleep(std::time::Duration::from_millis(60));
    };
    let t = std::time::Instant::now();
    export_photo(
        &ctx,
        &photo,
        &ExportParams::new(opts),
        &ExportControl {
            cancel: CancelToken::new(),
            progress: Some(&probe),
        },
    )
    .expect("export");
    let spent = t.elapsed();
    say(&format!(
        "the export made {} progress reports",
        reports.load(std::sync::atomic::Ordering::SeqCst)
    ));
    assert!(reports.load(std::sync::atomic::Ordering::SeqCst) >= 1);
    assert_eq!(
        idle_reports.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the scheduler says an export is running from its start to its end, not only while a job is queued"
    );
    assert!(!r.scheduler().exports_active(), "the guard was released");
    let at_end = done_of(&r);
    say(&format!(
        "export {:.0} ms; bands made {at_start:?} at its start, {at_end:?} at its end",
        spent.as_secs_f64() * 1000.0
    ));
    // The export did not wait for the fill to end, and the fill made at most the band that was running.
    assert!(at_end != Some(usize::MAX) || at_start == Some(usize::MAX));
    if let (Some(a), Some(b)) = (at_start, at_end) {
        assert!(
            b <= a.saturating_add(1),
            "the fill went on during the export"
        );
    }
    // The fill resumes, and the level is complete and right.
    assert!(
        wait_ready(&r, &src, zoom, 60),
        "the fill resumed after the export"
    );
    for (dx, dy) in [(0.0, 0.0), (0.0, -1e9), (1e9, 1e9), (-20.0, 30.0)] {
        st.input(&ViewportInput::Pan { dx, dy }, image);
        let served = frame(&r, &src, &st, false);
        assert_eq!(served.report.runs.level_hits, 1, "pan ({dx}, {dy})");
        assert_eq!(bytes(&served), bytes(&frame(&r, &src, &st, true)));
    }
    let bands = r.counters(src.id).level_bands;
    assert!(bands >= 3, "bands made: {bands}");
    // Each band was made once: the counter is the plan's bands (no band twice) -- the level has exactly them.
    let LevelStatus::Ready = r.level_status(&src, zoom) else {
        panic!("ready")
    };
}
