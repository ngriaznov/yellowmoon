//! The lens in the render graph (`docs/21` sections 3, 9 and 10; `docs/01` Render graph): L1 to L3
//! between the raw group and the colour group, on the CPU, for every picture the app and `ym` make.
//!
//! * with all three amounts at 0 nothing changes: the bytes of a render without the field, and no
//!   lens counter moves;
//! * a lens changes the fit view, the 100 % view, the thumbnail and the export, and the Before view
//!   has none;
//! * a slider move re-runs the lens passes and nothing under them (the planes are made once per
//!   photo);
//! * the 100 % view's lens tiles and the export's regions are the whole-frame `ym_lens::apply`
//!   (`ym-lens` guarantees that any tiling gives the bits of the whole frame);
//! * the view stages on the device take the lens's host images and agree with the CPU.
//!
//! The synthetic photos are the ones the other render tests use (`tests/common`) and the lens data are
//! the shipped `data/lenses`.
mod common;

use std::io::Write as _;
use std::sync::{Arc, OnceLock};

use common::*;
use ym_color::display::OutputTarget;
use ym_color::encode::OutputSpace;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource,
    RenderOptions, RenderedFrame, Renderer, StageRuns, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, LensKind, LensSettings, SettingsPatch};
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists};
use ym_gpu::cpu::Rect;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensLibrary, LensPlan, LensPlanes,
    PixelRect,
};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// Vintage with all three sliders well above 0: every stage of the lens has something to do.
const VINTAGE: LensSettings = LensSettings {
    kind: LensKind::Vintage,
    flares: 60.0,
    edges: 50.0,
    anamorphic: 50.0,
};

fn library() -> Arc<LensLibrary> {
    static LIBRARY: OnceLock<Arc<LensLibrary>> = OnceLock::new();
    Arc::clone(LIBRARY.get_or_init(|| {
        Arc::new(LensLibrary::load(&data_root()).expect("the shipped lens data load"))
    }))
}

fn options() -> RenderOptions {
    RenderOptions {
        lens: Some(library()),
        data_root: Some(data_root()),
        ..RenderOptions::default()
    }
}

fn cpu() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), options()).expect("render thread")
}

/// The device, in poison mode. Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1`.
fn gpu() -> Option<Arc<Gpu>> {
    // A job of the lens kernels is heavy for a software rasteriser: the device's 30 s hang limit would
    // fire on a loaded development machine.
    match Gpu::start(GpuConfig {
        poison: true,
        readback_timeout: std::time::Duration::from_secs(900),
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
                "\n* GPU TESTS SKIPPED in render_lens: {e}\n"
            );
            None
        }
    }
}

/// A renderer whose view stages (sampler, colour chain, output sharpening, C13) run on `gpu`; the lens
/// runs on the host and hands the device its images.
fn device(gpu: &Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::SAMPLER
                .union(GpuStages::COLOR)
                .union(GpuStages::OUTPUT_SHARPEN)
                .union(GpuStages::ENCODE),
            ..options()
        },
    )
    .expect("render thread")
}

/// A renderer whose lens runs on the device too (L2 and L3 kernels), with the view stages; the raw group is
/// the CPU's and the lens uploads it.
fn device_lens(gpu: &Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::SAMPLER
                .union(GpuStages::COLOR)
                .union(GpuStages::OUTPUT_SHARPEN)
                .union(GpuStages::ENCODE)
                .union(GpuStages::LENS),
            ..options()
        },
    )
    .expect("render thread")
}

/// A renderer with every stage on the device: the raw group is made there and the lens reads it where it is.
fn device_everything(gpu: &Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::ALL,
            ..options()
        },
    )
    .expect("render thread")
}

/// Patches of many levels below the flares' knee (half the clip level), with a little noise: edges for
/// the field optics; and three small lights the sensor clipped, for the flares (a photo's lights are
/// small and few, and a big bright patch would be hundreds of sources).
fn developed(w: u32, h: u32) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Patchwork, 0.35).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let scale = f64::from(w) / 800.0;
    for (fx, fy, radius) in [(0.22, 0.30, 6.0), (0.71, 0.24, 9.0), (0.58, 0.78, 4.0)] {
        let (cx, cy, r) = (fx * f64::from(w), fy * f64::from(h), radius * scale);
        for y in 0..h {
            for x in 0..w {
                let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                if d <= r {
                    mosaic[(y * w + x) as usize] = scene.white_level as u16;
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn with_lens(src: &PhotoSource, lens: LensSettings) -> PhotoSource {
    edit(src, |s| s.lens = lens)
}

fn cell(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn at_100_percent(w: f64, h: f64, image: (usize, usize)) -> ViewportState {
    let mut st = cell(w, h);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    st
}

fn frame(r: &Renderer, src: &PhotoSource, state: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, state, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

fn fresh_frame(r: &Renderer, src: &PhotoSource, state: &ViewportState) -> RenderedFrame {
    let options = FrameOptions {
        no_cache: true,
        ..FrameOptions::default()
    };
    r.viewport_frame(src, state, &options, &CancelToken::new())
        .expect("frame")
}

fn export(r: &Renderer, src: &PhotoSource) -> FullImage {
    r.render_full(
        src,
        &FullRequest {
            target: FullTarget::DisplayLinear(OutputTarget::Viewport),
        },
        &CancelToken::new(),
    )
    .expect("full render")
}

/// The largest difference between two frames, in codes, over every sample.
fn worst_difference(a: &RenderedFrame, b: &RenderedFrame) -> i32 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.rgba
        .iter()
        .zip(&b.rgba)
        .map(|(&x, &y)| (i32::from(x) - i32::from(y)).abs())
        .max()
        .unwrap_or(0)
}

/// The lens counters of a set of runs.
fn lens_work(runs: &StageRuns) -> (u64, u64, u64) {
    (runs.lens_planes, runs.lens_groups, runs.lens_tiles)
}

/// A photo without sharpening on the output (the finished frame is then the colour pass alone).
fn plain(src: &PhotoSource) -> PhotoSource {
    edit(src, |s| s.output_sharpening = 0.0)
}

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// All three amounts at 0, for every lens: the fit view, the 100 % view, the thumbnail and the export
/// are the bytes of a render without the field, and no lens counter moves.
fn zero_amounts_are_the_render_without_the_field_with(r: &Renderer) {
    let d = developed(800, 600);
    let src = d.source();
    assert_eq!(src.settings.lens, LensSettings::default());
    let fit = cell(400.0, 300.0);
    let tiles = at_100_percent(560.0, 360.0, (800, 600));
    let token = CancelToken::new();
    let reference = (
        fresh_frame(r, &src, &fit).rgba,
        fresh_frame(r, &src, &tiles).rgba,
        export(r, &src).rgb,
        r.thumbnail_png(&src, &token).expect("thumbnail"),
    );
    for kind in LensKind::ALL {
        let zero = with_lens(
            &src,
            LensSettings {
                kind,
                flares: 0.0,
                edges: 0.0,
                anamorphic: 0.0,
            },
        );
        let f = fresh_frame(r, &zero, &fit);
        let t = fresh_frame(r, &zero, &tiles);
        let e = export(r, &zero);
        assert!(f.rgba == reference.0, "{kind}: the fit view moved");
        assert!(t.rgba == reference.1, "{kind}: the 100 % view moved");
        assert!(e.rgb == reference.2, "{kind}: the export moved");
        assert!(
            r.thumbnail_png(&zero, &token).expect("thumbnail") == reference.3,
            "{kind}: the thumbnail moved"
        );
        for runs in [&f.report.runs, &t.report.runs, &e.report.runs] {
            assert_eq!(lens_work(runs), (0, 0, 0), "{kind}: {runs:?}");
        }
        assert_eq!(f.report.timings.lens_ms, 0.0, "{kind}");
        assert!(f.report.lens.is_none() && t.report.lens.is_none() && e.report.lens.is_none());
        assert!(
            !f.report
                .notes
                .iter()
                .chain(&e.report.notes)
                .any(|n| n.contains("lens")),
            "{kind}: {:?} {:?}",
            f.report.notes,
            e.report.notes
        );
    }
    let total = r.counters(src.id);
    assert_eq!(lens_work(&total), (0, 0, 0), "{total:?}");
}

#[test]
fn zero_amounts_are_the_render_without_the_field() {
    zero_amounts_are_the_render_without_the_field_with(&cpu());
}

#[test]
fn zero_amounts_are_the_render_without_the_field_on_the_device() {
    let Some(gpu) = gpu() else { return };
    zero_amounts_are_the_render_without_the_field_with(&device(&gpu));
}

// ---------------------------------------------------------------------------------------------
// A lens changes the picture; a slider move re-runs the lens only
// ---------------------------------------------------------------------------------------------

fn a_lens_changes_the_fit_view_and_a_slider_move_re_runs_only_the_lens_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let state = cell(576.0, 384.0);
    let before = fresh_frame(r, &src, &state);

    // The first lens frame: the raw group and the planes are made once, the lens runs on the group.
    let lensed_src = with_lens(&src, VINTAGE);
    let first = frame(r, &lensed_src, &state);
    assert!(first.sharp);
    let runs = first.report.runs;
    assert_eq!(
        lens_work(&runs),
        (1, 1, 0),
        "planes and one group, no tile: {runs:?}"
    );
    // The raw group under the lens was made for the frame without it: it is not made again.
    assert_eq!(runs.raw_groups, 0, "{runs:?}");
    assert!(first.report.timings.lens_ms > 0.0);
    let worst = worst_difference(&before, &first);
    assert!(
        worst >= 4,
        "the lens moved the picture by {worst} codes only"
    );
    let lens = first.report.lens.expect("the lens report");
    assert!(
        lens.reshaped || lens.reason.is_some(),
        "the report says why when it did not reshape: {lens:?}"
    );
    // The stats the webview gets carry it.
    assert_eq!(
        first.report.stats().lens.map(|l| l.reshaped),
        Some(lens.reshaped)
    );

    // The same request again: the finished frame is served from the cache and says the same.
    let again = frame(r, &lensed_src, &state);
    assert_eq!(again.rgba, first.rgba);
    assert_eq!(again.report.runs.frame_hits, 1);
    assert_eq!(lens_work(&again.report.runs), (0, 0, 0));
    assert_eq!(
        again.report.lens, first.report.lens,
        "a cached frame forgets the report"
    );

    // A slider move: new pixels, from the cached raw group and the cached planes. Only the lens pass
    // and the colour pass run.
    for (name, moved) in [
        (
            "Flares",
            LensSettings {
                flares: 61.0,
                ..VINTAGE
            },
        ),
        (
            "Edges",
            LensSettings {
                edges: 51.0,
                ..VINTAGE
            },
        ),
        (
            "Anamorphic",
            LensSettings {
                anamorphic: 51.0,
                ..VINTAGE
            },
        ),
    ] {
        let f = frame(r, &with_lens(&src, moved), &state);
        let runs = f.report.runs;
        assert_eq!(runs.raw_work(), 0, "{name}: no raw stage ran: {runs:?}");
        assert_eq!(runs.sigma_planes, 0, "{name}: {runs:?}");
        assert_eq!(
            runs.lens_planes, 0,
            "{name}: the planes were made again: {runs:?}"
        );
        assert_eq!(runs.lens_groups, 1, "{name}: {runs:?}");
        assert_eq!(runs.color_plans + runs.grade_plans, 0, "{name}: {runs:?}");
        assert_ne!(f.rgba, first.rgba, "{name}: the picture did not move");
    }
    // The totals: the planes were made once for all of it.
    let total = r.counters(src.id);
    assert_eq!(total.lens_planes, 1, "{total:?}");
    assert_eq!(total.lens_groups, 4, "{total:?}");

    // A colour edit does not run the lens again: the group's key does not hold it.
    let warmer = edit(&with_lens(&src, VINTAGE), |s| s.brightness = 20.0);
    let f = frame(r, &warmer, &state);
    assert_eq!(lens_work(&f.report.runs), (0, 0, 0), "{:?}", f.report.runs);
    assert_eq!(f.report.runs.raw_work(), 0);
    assert_eq!(f.report.lens, first.report.lens);

    // Back to no lens: the frame without it is what it was, from the cache of the raw group.
    let off = fresh_frame(r, &src, &state);
    assert!(off.rgba == before.rgba);
    assert_eq!(lens_work(&off.report.runs), (0, 0, 0));
    assert!(off.report.lens.is_none());
}

#[test]
fn a_lens_changes_the_fit_view_and_a_slider_move_re_runs_only_the_lens() {
    a_lens_changes_the_fit_view_and_a_slider_move_re_runs_only_the_lens_with(&cpu());
}

#[test]
fn a_lens_changes_the_fit_view_and_a_slider_move_re_runs_only_the_lens_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_lens_changes_the_fit_view_and_a_slider_move_re_runs_only_the_lens_with(&device(&gpu));
}

#[test]
fn the_fit_view_on_the_device_is_the_cpu_fit_view_of_the_lens_group() {
    let Some(gpu) = gpu() else { return };
    let d = developed(1200, 800);
    let src = with_lens(&plain(&d.source()), VINTAGE);
    let state = cell(576.0, 384.0);
    let on_cpu = fresh_frame(&cpu(), &src, &state);
    let on_device = fresh_frame(&device(&gpu), &src, &state);
    // The raw group is the CPU's in both (the raw chain stays on the CPU), the lens is the host's, and
    // the colour pass differs by the tolerances of the kernels.
    let worst = worst_difference(&on_cpu, &on_device);
    let _ = writeln!(
        std::io::stderr(),
        "[lens] fit view, CPU against device: {worst} codes"
    );
    assert!(
        worst <= 2,
        "the device colour pass differs by {worst} codes"
    );
    assert_eq!(
        on_cpu.report.runs.lens_groups,
        on_device.report.runs.lens_groups
    );
}

/// The lens's field stage `Z = L2(L1(X))` does not read Flares and its flare terms do not read Edges: a
/// Flares move finds `Z` cached and makes the flare terms, an Edges move makes `Z` and finds the flare terms,
/// an Anamorphic move makes both, and whichever way a frame was reached its pixels are the pixels of a cold
/// renderer (the caches change what runs, never what comes out).
fn a_slider_move_runs_only_the_stages_that_read_it_with(
    r: &Renderer,
    cold: &dyn Fn() -> Renderer,
    device: bool,
) {
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let state = cell(576.0, 384.0);
    // (field stages, flare terms, groups) a request made.
    let stages = |f: &RenderedFrame| {
        (
            f.report.runs.lens_fields,
            f.report.runs.lens_flares,
            f.report.runs.lens_groups,
        )
    };
    let first = frame(r, &with_lens(&src, VINTAGE), &state);
    assert_eq!(stages(&first), (1, 1, 1), "{:?}", first.report.runs);
    let mut previous = first.rgba.clone();
    for flares in [61.0, 20.0, 100.0] {
        let moved = LensSettings { flares, ..VINTAGE };
        let f = frame(r, &with_lens(&src, moved), &state);
        let runs = f.report.runs;
        assert_eq!(
            (stages(&f), runs.lens_planes),
            ((0, 1, 1), 0),
            "Flares {flares}: the flare terms alone: {runs:?}"
        );
        assert_ne!(
            f.rgba, previous,
            "Flares {flares}: the picture did not move"
        );
        let reference = fresh_frame(&cold(), &with_lens(&src, moved), &state);
        assert!(
            f.rgba == reference.rgba,
            "Flares {flares}: the cached field stage changed the pixels"
        );
        previous = f.rgba.clone();
    }
    // Edges is read by the field stage and not by the flare terms (the device does not keep those: it
    // makes them again in a kernel that costs next to nothing).
    let edges = LensSettings {
        edges: 51.0,
        ..VINTAGE
    };
    let f = frame(r, &with_lens(&src, edges), &state);
    assert_eq!(
        stages(&f),
        (1, u64::from(device), 1),
        "Edges: {:?}",
        f.report.runs
    );
    let reference = fresh_frame(&cold(), &with_lens(&src, edges), &state);
    assert!(
        f.rgba == reference.rgba,
        "Edges: pixels differ from a cold run"
    );
    // Anamorphic is read by both.
    let anamorphic = LensSettings {
        anamorphic: 51.0,
        ..VINTAGE
    };
    let f = frame(r, &with_lens(&src, anamorphic), &state);
    assert_eq!(stages(&f), (1, 1, 1), "Anamorphic: {:?}", f.report.runs);
    let reference = fresh_frame(&cold(), &with_lens(&src, anamorphic), &state);
    assert!(
        f.rgba == reference.rgba,
        "Anamorphic: pixels differ from a cold run"
    );
    // Back to a value every stage has seen: nothing runs in the lens.
    let again = fresh_frame(r, &with_lens(&src, VINTAGE), &state);
    assert_eq!(
        (lens_work(&again.report.runs), stages(&again)),
        ((0, 0, 0), (0, 0, 0)),
        "{:?}",
        again.report.runs
    );
    // Flares at 0 with the field stage on: the lens's output is the field stage's, made once.
    let off = LensSettings {
        flares: 0.0,
        ..VINTAGE
    };
    let f = frame(r, &with_lens(&src, off), &state);
    assert_eq!(stages(&f), (0, 0, 1), "{:?}", f.report.runs);
    let reference = fresh_frame(&cold(), &with_lens(&src, off), &state);
    assert!(
        f.rgba == reference.rgba,
        "Flares 0: pixels differ from a cold run"
    );
    // Flares alone has no field stage at all; Edges and Anamorphic at 0 are its other sliders.
    let only = LensSettings {
        kind: LensKind::Vintage,
        flares: 40.0,
        edges: 0.0,
        anamorphic: 0.0,
    };
    let f = frame(r, &with_lens(&src, only), &state);
    assert_eq!(stages(&f), (0, 1, 1), "{:?}", f.report.runs);
    let reference = fresh_frame(&cold(), &with_lens(&src, only), &state);
    assert!(
        f.rgba == reference.rgba,
        "Flares alone: pixels differ from a cold run"
    );
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it() {
    a_slider_move_runs_only_the_stages_that_read_it_with(&cpu(), &cpu, false);
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_slider_move_runs_only_the_stages_that_read_it_with(&device(&gpu), &|| device(&gpu), false);
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_with_the_lens_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_slider_move_runs_only_the_stages_that_read_it_with(
        &device_lens(&gpu),
        &|| device_lens(&gpu),
        true,
    );
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_with_everything_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_slider_move_runs_only_the_stages_that_read_it_with(
        &device_everything(&gpu),
        &|| device_everything(&gpu),
        true,
    );
}

/// The same at 100 %: the lens tiles of a Flares move are the cached field tiles plus new flare terms, those
/// of an Edges move new field tiles plus the cached flare terms.
fn a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with(
    r: &Renderer,
    cold: &dyn Fn() -> Renderer,
    device: bool,
) {
    let d = developed(800, 600);
    let src = plain(&d.source());
    let st = at_100_percent(500.0, 400.0, (800, 600));
    let first = frame(r, &with_lens(&src, VINTAGE), &st);
    let runs = first.report.runs;
    assert!(runs.lens_tiles >= 1, "{runs:?}");
    assert_eq!(
        runs.lens_fields, runs.lens_tiles,
        "one field tile per lens tile: {runs:?}"
    );
    assert_eq!(
        runs.lens_flares, runs.lens_tiles,
        "one flare tile per lens tile: {runs:?}"
    );
    let moved = LensSettings {
        flares: 61.0,
        ..VINTAGE
    };
    let f = frame(r, &with_lens(&src, moved), &st);
    let runs = f.report.runs;
    assert_eq!(runs.lens_fields, 0, "the field tiles are reused: {runs:?}");
    assert_eq!(runs.lens_flares, runs.lens_tiles, "{runs:?}");
    assert!(runs.lens_tiles >= 1, "{runs:?}");
    let reference = fresh_frame(&cold(), &with_lens(&src, moved), &st);
    assert!(
        f.rgba == reference.rgba,
        "the cached field tiles changed the pixels"
    );
    let moved = LensSettings {
        edges: 51.0,
        ..VINTAGE
    };
    let f = frame(r, &with_lens(&src, moved), &st);
    let runs = f.report.runs;
    assert_eq!(runs.lens_fields, runs.lens_tiles, "{runs:?}");
    if device {
        assert_eq!(runs.lens_flares, runs.lens_tiles, "{runs:?}");
    } else {
        assert_eq!(runs.lens_flares, 0, "the flare tiles are reused: {runs:?}");
    }
    let reference = fresh_frame(&cold(), &with_lens(&src, moved), &st);
    assert!(
        f.rgba == reference.rgba,
        "an Edges move: pixels differ from a cold run"
    );
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_on_tiles() {
    a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with(&cpu(), &cpu, false);
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with_the_lens_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with(
        &device_lens(&gpu),
        &|| device_lens(&gpu),
        true,
    );
}

#[test]
fn a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with_everything_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_slider_move_runs_only_the_stages_that_read_it_on_tiles_with(
        &device_everything(&gpu),
        &|| device_everything(&gpu),
        true,
    );
}

/// The first lens request is the planes' request: a request after a change of lens (the planes' key
/// changes) makes them again, an amount does not.
#[test]
fn the_planes_are_made_again_for_another_lens_and_not_for_an_amount() {
    let r = cpu();
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let state = cell(576.0, 384.0);
    let a = frame(&r, &with_lens(&src, VINTAGE), &state).report.runs;
    assert_eq!(a.lens_planes, 1);
    let b = frame(
        &r,
        &with_lens(
            &src,
            LensSettings {
                flares: 20.0,
                ..VINTAGE
            },
        ),
        &state,
    )
    .report
    .runs;
    assert_eq!(b.lens_planes, 0);
    let c = frame(
        &r,
        &with_lens(
            &src,
            LensSettings {
                kind: LensKind::Petzval,
                ..VINTAGE
            },
        ),
        &state,
    )
    .report
    .runs;
    assert_eq!(c.lens_planes, 1, "another lens, other planes: {c:?}");
    // Going back to the first lens makes the planes again (one set is kept), but its frame is cached.
    let d2 = frame(&r, &with_lens(&src, VINTAGE), &state).report.runs;
    assert_eq!(d2.frame_hits, 1, "{d2:?}");
    assert_eq!(r.counters(src.id).lens_planes, 2);
}

/// The planes key leaves the amounts out, but the blur map is built only for a plan that asks for
/// reshaping: planes made for Flares alone do not serve a plan with Anamorphic, and are made again
/// (`LensPlanes::serves`); planes with the map serve a plan that does not need it.
fn planes_without_the_blur_map_are_made_again_when_a_slider_needs_it_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let state = cell(576.0, 384.0);
    let flares = LensSettings {
        kind: LensKind::Vintage,
        flares: 50.0,
        edges: 0.0,
        anamorphic: 0.0,
    };
    let a = frame(r, &with_lens(&src, flares), &state);
    assert_eq!(a.report.runs.lens_planes, 1);
    assert_eq!(
        a.report.lens.and_then(|l| l.reason),
        Some(ym_lens::ReshapeReason::NotAsked)
    );
    // Anamorphic asks for the bokeh: the cached planes lack the map, so they are made again, and the
    // render works (no error from `apply`).
    let anamorphic = LensSettings {
        anamorphic: 40.0,
        ..flares
    };
    let b = frame(r, &with_lens(&src, anamorphic), &state);
    assert_eq!(b.report.runs.lens_planes, 1, "{:?}", b.report.runs);
    assert_eq!(b.report.runs.lens_groups, 1);
    assert_ne!(a.rgba, b.rgba);
    assert_ne!(
        b.report.lens.and_then(|l| l.reason),
        Some(ym_lens::ReshapeReason::NotAsked),
        "the planes were made with the blur map"
    );
    // The planes with the map serve the plan that does not need it, and the report is the plan's.
    let c = frame(
        r,
        &with_lens(
            &src,
            LensSettings {
                flares: 51.0,
                ..flares
            },
        ),
        &state,
    );
    assert_eq!(c.report.runs.lens_planes, 0, "{:?}", c.report.runs);
    assert_eq!(
        c.report.lens.and_then(|l| l.reason),
        Some(ym_lens::ReshapeReason::NotAsked)
    );
    // And the other sliders of the plan that needs the map keep them.
    let d2 = frame(
        r,
        &with_lens(
            &src,
            LensSettings {
                anamorphic: 41.0,
                ..flares
            },
        ),
        &state,
    );
    assert_eq!(d2.report.runs.lens_planes, 0, "{:?}", d2.report.runs);
    // An export of a plan that needs the map makes its own, with it.
    let e = export(r, &with_lens(&src, anamorphic));
    assert_eq!(e.report.runs.lens_planes, 1);
}

#[test]
fn planes_without_the_blur_map_are_made_again_when_a_slider_needs_it() {
    planes_without_the_blur_map_are_made_again_when_a_slider_needs_it_with(&cpu());
}

// ---------------------------------------------------------------------------------------------
// The 100 % view and the export
// ---------------------------------------------------------------------------------------------

/// The lens reference for a photo: the plan and the planes `ym_lens` makes from the raw group over the
/// whole frame (the test frames are under 1024 px, so the lens plane is the frame itself), and the
/// whole-frame `apply` of it.
struct Reference {
    raw: ym_gpu::cpu::RgbImage,
    out: Vec<f32>,
    width: usize,
    height: usize,
    /// The plan's L4 map for the frame (a canvas the size of the frame, no turns).
    warp: ym_lens::LensWarp,
}

fn reference_of(r: &Renderer, src: &PhotoSource, lens: LensSettings, axis: Axis) -> Reference {
    let token = CancelToken::new();
    let (w, h) = r.final_size(src, &token).expect("size");
    // The camera orientation of the synthetic file is the sensor's, so a quarter turn swaps the sides.
    let (w, h) = if axis == Axis::Y { (h, w) } else { (w, h) };
    let raw = r
        .camera_region(src, Rect::whole(w, h), false, &token)
        .expect("the raw group");
    let camera =
        CameraResponse::from_curves(&src.profile.load_curves(&data_root()).expect("the curves"))
            .expect("the camera");
    let g0 = src.estimates.initial_gains().expect("g0");
    let frame = FrameInfo {
        width: w,
        height: h,
        axis,
        f_number: src.raw.meta.f_number.filter(|f| f.is_finite() && *f > 0.0),
        clip: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
    };
    assert!(
        src.raw.meta.focus.is_none(),
        "the synthetic file has no focus point"
    );
    let plan = LensPlan::new(&library(), &LensAmounts::from(lens), frame, &camera).expect("plan");
    let image = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &raw.data,
    };
    assert_eq!(
        plan.lens_plane_scale(),
        1,
        "a test frame is its own lens plane"
    );
    let planes = LensPlanes::build(&plan, &image, None, &|| false).expect("planes");
    let out = ym_lens::apply(&plan, &planes, 1, &image, PixelRect::new(0, 0, w, h)).expect("apply");
    let warp = plan.warp(w as f64, h as f64);
    Reference {
        raw,
        out,
        width: w,
        height: h,
        warp,
    }
}

impl Reference {
    /// What the geometry sampler reads at the canvas position `(x, y)` (continuous, pixel centres at
    /// `i + 0.5`) of an unturned, unstraightened view of the frame: the lens output at the photo
    /// position L4's inverse map gives, by Catmull-Rom (the sampler's kernel for a bent grid).
    fn sampled(&self, x: f64, y: f64) -> [f32; 3] {
        let n = self.width * self.height;
        let (px, py) = self.warp.canvas_to_photo(x, y);
        let (px, py) = (
            px.clamp(0.0, self.width as f64),
            py.clamp(0.0, self.height as f64),
        );
        std::array::from_fn(|c| {
            ym_gpu::cpu::resample::sample_catmull_rom(
                &self.out[c * n..(c + 1) * n],
                self.width,
                self.height,
                px,
                py,
            )
        })
    }
}

/// An export tile (the raw region grown by the halo, one call) is the whole-frame `apply`, bit for bit,
/// for rectangles that cross the 512 px lines of the tiles and the borders of the frame.
fn an_export_region_is_the_whole_frame_apply_with(r: &Renderer, axis: Axis) {
    // 640 x 540: under 1024 px (the lens plane is the frame) and past the 512 px line both ways.
    let d = developed(640, 540);
    let mut src = with_lens(&d.source(), VINTAGE);
    if axis == Axis::Y {
        src = edit(&src, |s| {
            s.orientation = ym_core::settings::Orientation::Deg90
        });
    }
    let reference = reference_of(r, &src, VINTAGE, axis);
    let (w, h) = (reference.width, reference.height);
    let token = CancelToken::new();
    let different_from_raw = reference
        .raw
        .data
        .iter()
        .zip(&reference.out)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        different_from_raw > reference.out.len() / 4,
        "the lens changed only {different_from_raw} samples"
    );
    let n = w * h;
    for rect in [
        Rect::whole(w, h),
        Rect::new(37, 11, 300, 200),
        Rect::new(480, 90, 130, 380),
        Rect::new(w - 70, h - 40, 70, 40),
        Rect::new(0, 0, 64, 64),
    ] {
        let got = r.camera_region(&src, rect, true, &token).expect("region");
        assert_eq!((got.width, got.height), (rect.w, rect.h));
        // The image keeps its place in the frame.
        assert_eq!((got.place.x0, got.place.y0), (rect.x, rect.y));
        assert_eq!((got.place.frame_width, got.place.frame_height), (w, h));
        let mut worst = 0.0f32;
        for c in 0..3 {
            for y in 0..rect.h {
                for x in 0..rect.w {
                    let a = got.data[c * rect.w * rect.h + y * rect.w + x];
                    let b = reference.out[c * n + (rect.y + y) * w + rect.x + x];
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert_eq!(
            worst, 0.0,
            "{rect:?} differs from the whole-frame apply by {worst}"
        );
    }
}

#[test]
fn an_export_region_is_the_whole_frame_apply() {
    an_export_region_is_the_whole_frame_apply_with(&cpu(), Axis::X);
}

#[test]
fn an_export_region_of_a_turned_photo_uses_the_canvas_horizontal_as_the_axis() {
    // A quarter turn: the canvas's horizontal is the buffer's y.
    an_export_region_is_the_whole_frame_apply_with(&cpu(), Axis::Y);
}

/// The 100 % view reads the same lens: `camera_rgb_at` (the raw group's output at a pixel of the view,
/// before C1) is the whole-frame `apply` at that pixel, on both sides of the tile lines.
fn the_100_percent_view_samples_the_whole_frame_apply_with(r: &Renderer) {
    let d = developed(800, 600);
    let src = with_lens(&d.source(), VINTAGE);
    let reference = reference_of(r, &src, VINTAGE, Axis::X);
    let st = at_100_percent(800.0, 600.0, (800, 600));
    let token = CancelToken::new();
    for (x, y) in [
        (3, 5),
        (100, 100),
        (511, 300),
        (512, 300),
        (513, 301),
        (700, 590),
        (400, 511),
        (400, 512),
    ] {
        let got = r
            .camera_rgb_at(&src, &st, (x as f64 + 0.5, y as f64 + 0.5), &token)
            .expect("reading")
            .expect("inside the photo");
        // The photo's lens bends the picture (Anamorphic and Edges are above 0): the view reads the
        // lens output at the position the map gives, not at the pixel itself.
        let want = reference.sampled(x as f64 + 0.5, y as f64 + 0.5);
        for c in 0..3 {
            assert!(
                (got[c] - f64::from(want[c])).abs() <= 1e-6 * f64::from(want[c]).abs().max(1.0),
                "pixel ({x}, {y}) channel {c}: {} against {}",
                got[c],
                want[c]
            );
        }
    }
    // The readouts made lens tiles (not a lens group), and no more of them than the pixels need.
    let total = r.counters(src.id);
    assert_eq!(total.lens_groups, 0, "{total:?}");
    assert!(total.lens_tiles >= 2 && total.lens_tiles <= 4, "{total:?}");
    assert_eq!(total.lens_planes, 1, "{total:?}");
}

#[test]
fn the_100_percent_view_samples_the_whole_frame_apply() {
    the_100_percent_view_samples_the_whole_frame_apply_with(&cpu());
}

#[test]
fn the_100_percent_view_on_the_device_samples_the_whole_frame_apply() {
    let Some(gpu) = gpu() else { return };
    the_100_percent_view_samples_the_whole_frame_apply_with(&device(&gpu));
}

/// The 100 % view and the export show one picture: the view's frame is the export's pixels, encoded,
/// within the one code the dither moves a value by.
fn the_100_percent_view_is_the_picture_the_export_writes_with(r: &Renderer) {
    let d = developed(800, 600);
    let src = with_lens(&plain(&d.source()), VINTAGE);
    let full = export(r, &src);
    assert_eq!((full.width, full.height), (800, 600));
    assert!(lens_work(&full.report.runs).2 > 0, "{:?}", full.report.runs);
    assert_eq!(full.report.runs.lens_planes, 1);
    let st = at_100_percent(800.0, 600.0, (800, 600));
    let f = frame(r, &src, &st);
    assert_eq!(f.report.scale, 1, "the full-resolution tiles");
    assert!(f.report.runs.lens_tiles >= 4, "{:?}", f.report.runs);
    let mut worst = 0.0f64;
    for y in 0..600 {
        for x in 0..800 {
            let i = (y * 800 + x) * 3;
            let enc = OutputSpace::DisplayP3
                .encode_clamped([full.rgb[i], full.rgb[i + 1], full.rgb[i + 2]].map(f64::from));
            for (c, e) in enc.iter().enumerate() {
                let got = f64::from(f.rgba[(y * 800 + x) * 4 + c]);
                worst = worst.max((got - e * 255.0).abs());
            }
        }
    }
    let _ = writeln!(
        std::io::stderr(),
        "[lens] 100 % view against the export: worst {worst:.4} codes"
    );
    assert!(
        worst <= 1.0 + 1e-6,
        "the view and the export differ by {worst} codes"
    );
    // The picture is not the one without the lens.
    let off = frame(r, &plain(&d.source()), &st);
    assert!(worst_difference(&off, &f) >= 4);
}

#[test]
fn the_100_percent_view_is_the_picture_the_export_writes() {
    the_100_percent_view_is_the_picture_the_export_writes_with(&cpu());
}

#[test]
fn the_100_percent_view_on_the_device_is_the_picture_the_export_writes() {
    let Some(gpu) = gpu() else { return };
    the_100_percent_view_is_the_picture_the_export_writes_with(&device(&gpu));
}

/// A coarse frame (the tiles are not ready) stays coarse until the lens tiles exist, and the work
/// ahead (`prepare_tiles`) makes them, so the next frame is sharp.
fn the_work_ahead_makes_the_lens_tiles_with(r: &Renderer) {
    let d = developed(800, 600);
    let src = with_lens(&plain(&d.source()), VINTAGE);
    let st = at_100_percent(560.0, 360.0, (800, 600));
    let coarse = FrameOptions {
        coarse_only: true,
        ..FrameOptions::default()
    };
    let token = CancelToken::new();
    let first = r.viewport_frame(&src, &st, &coarse, &token).expect("frame");
    assert!(!first.sharp, "nothing is ready yet");
    // The coarse frame is the lens too (the fit-size group went through it).
    assert!(
        first.report.runs.lens_groups >= 1,
        "{:?}",
        first.report.runs
    );
    let made = r.prepare_tiles(&src, &st, &token).expect("tiles");
    assert!(made > 0);
    let sharp = r.viewport_frame(&src, &st, &coarse, &token).expect("frame");
    assert!(
        sharp.sharp,
        "the lens tiles are ready, so the frame is sharp"
    );
    assert_eq!(sharp.report.runs.lens_tiles, 0, "{:?}", sharp.report.runs);
    assert_eq!(sharp.report.runs.raw_work(), 0, "{:?}", sharp.report.runs);
    // The same request without the lens had its raw tiles cached by the work ahead too, and a lens-free
    // photo's readiness is the raw tiles' (the old rule).
    let off = plain(&d.source());
    let first = r.viewport_frame(&off, &st, &coarse, &token).expect("frame");
    assert!(first.sharp, "the raw tiles under the lens tiles are cached");
}

#[test]
fn the_work_ahead_makes_the_lens_tiles() {
    the_work_ahead_makes_the_lens_tiles_with(&cpu());
}

#[test]
fn the_work_ahead_makes_the_lens_tiles_on_the_device() {
    let Some(gpu) = gpu() else { return };
    the_work_ahead_makes_the_lens_tiles_with(&device(&gpu));
}

/// Releasing a photo's full-resolution intermediates drops its lens tiles, and keeps its planes.
#[test]
fn releasing_the_full_resolution_drops_the_lens_tiles_and_keeps_the_planes() {
    let r = cpu();
    let d = developed(800, 600);
    let src = with_lens(&plain(&d.source()), VINTAGE);
    let st = at_100_percent(560.0, 360.0, (800, 600));
    frame(&r, &src, &st);
    let graph = r.graph(src.id);
    let held = graph.bytes();
    graph.release_full_resolution();
    assert!(graph.bytes() < held);
    let f = fresh_frame(&r, &src, &st);
    assert_eq!(
        f.report.runs.lens_planes, 0,
        "the planes stayed: {:?}",
        f.report.runs
    );
    assert!(f.report.runs.lens_tiles > 0, "{:?}", f.report.runs);
}

// ---------------------------------------------------------------------------------------------
// Before, thumbnails, look tiles
// ---------------------------------------------------------------------------------------------

fn the_before_view_has_no_lens_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let mut state = cell(576.0, 384.0);
    state.before = true;
    let a = fresh_frame(r, &src, &state);
    let b = fresh_frame(r, &with_lens(&src, VINTAGE), &state);
    assert_eq!(a.rgba, b.rgba, "the Before view moved with the lens");
    assert_eq!(lens_work(&b.report.runs), (0, 0, 0), "{:?}", b.report.runs);
    assert!(b.report.lens.is_none());
    // A 100 % Before view neither.
    let mut st = at_100_percent(560.0, 360.0, (1200, 800));
    st.before = true;
    let c = fresh_frame(r, &src, &st);
    let e = fresh_frame(r, &with_lens(&src, VINTAGE), &st);
    assert_eq!(c.rgba, e.rgba);
    assert_eq!(lens_work(&e.report.runs), (0, 0, 0));
    assert_eq!(lens_work(&r.counters(src.id)), (0, 0, 0));
    // The readout of the Before view neither.
    let at = r
        .sample_pixel(
            &with_lens(&src, VINTAGE),
            &st,
            (100.5, 100.5),
            &CancelToken::new(),
        )
        .expect("sample");
    let none = r
        .sample_pixel(&src, &st, (100.5, 100.5), &CancelToken::new())
        .expect("sample");
    assert_eq!(at, none);
}

#[test]
fn the_before_view_has_no_lens() {
    the_before_view_has_no_lens_with(&cpu());
}

#[test]
fn the_before_view_on_the_device_has_no_lens() {
    let Some(gpu) = gpu() else { return };
    the_before_view_has_no_lens_with(&device(&gpu));
}

#[test]
fn a_thumbnail_and_a_look_tile_show_the_lens() {
    let r = cpu();
    let d = developed(1200, 800);
    let src = d.source();
    let token = CancelToken::new();
    let thumbnail = r.thumbnail_png(&src, &token).expect("thumbnail");
    let lensed = with_lens(&src, VINTAGE);
    let with = r.thumbnail_png(&lensed, &token).expect("thumbnail");
    assert_ne!(thumbnail, with, "the lens did not reach the thumbnail");
    assert_eq!(
        r.thumbnail_png(&lensed, &token).expect("thumbnail"),
        with,
        "the same request gave another thumbnail"
    );
    let neutral = ym_looks::library::LookId::builtin("neutral");
    let tile = r.look_thumbnail_png(&src, &neutral, &token).expect("tile");
    let tile_with = r
        .look_thumbnail_png(&lensed, &neutral, &token)
        .expect("tile");
    assert_ne!(tile, tile_with, "the lens did not reach the look tile");
    let total = r.counters(src.id);
    assert_eq!(
        total.lens_planes, 1,
        "one set of planes for all of them: {total:?}"
    );
    assert!(total.lens_groups >= 1, "{total:?}");
}

// ---------------------------------------------------------------------------------------------
// The report and the missing data
// ---------------------------------------------------------------------------------------------

#[test]
fn the_report_says_whether_the_bokeh_was_reshaped_and_a_plan_without_the_ask_says_so() {
    let r = cpu();
    let d = developed(800, 600);
    let src = plain(&d.source());
    let state = cell(400.0, 300.0);
    // Only Flares: bokeh reshaping was not asked for.
    let flares = LensSettings {
        kind: LensKind::Vintage,
        flares: 50.0,
        edges: 0.0,
        anamorphic: 0.0,
    };
    let f = frame(&r, &with_lens(&src, flares), &state);
    let lens = f.report.lens.expect("a report");
    assert!(!lens.reshaped);
    assert_eq!(lens.reason, Some(ym_lens::ReshapeReason::NotAsked));
    let stats = f.report.stats().lens.expect("stats");
    assert!(!stats.reshaped);
    assert_eq!(
        stats.reason,
        Some(ym_core::pipeline::LensReshapeReason::NotAsked)
    );
    // Anamorphic asks for it: the synthetic patchwork has no out-of-focus background, so the report
    // says why it was left alone.
    let f = frame(&r, &with_lens(&src, VINTAGE), &state);
    let lens = f.report.lens.expect("a report");
    assert!(!lens.reshaped, "{lens:?}");
    assert_eq!(lens.reason, Some(ym_lens::ReshapeReason::LittleBlur));
    assert_eq!(
        f.report.stats().lens.and_then(|l| l.reason),
        Some(ym_core::pipeline::LensReshapeReason::LittleBlur)
    );
}

#[test]
fn without_the_lens_data_a_lens_setting_renders_as_no_lens_and_says_so() {
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            lens: None,
            data_root: Some(data_root()),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let d = developed(800, 600);
    let src = plain(&d.source());
    let state = cell(400.0, 300.0);
    let a = fresh_frame(&r, &src, &state);
    let b = fresh_frame(&r, &with_lens(&src, VINTAGE), &state);
    assert_eq!(a.rgba, b.rgba);
    assert!(
        b.report.notes.iter().any(|n| n.contains("lens")),
        "{:?}",
        b.report.notes
    );
    assert_eq!(lens_work(&b.report.runs), (0, 0, 0));
    assert!(b.report.lens.is_none());
}

#[test]
fn a_camera_whose_curves_cannot_be_read_gets_a_neutral_lens_and_a_note() {
    let r = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            lens: Some(library()),
            data_root: None,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let d = developed(800, 600);
    let src = plain(&d.source());
    let state = cell(400.0, 300.0);
    let f = fresh_frame(&r, &with_lens(&src, VINTAGE), &state);
    assert!(
        f.report.notes.iter().any(|n| n.contains("colour curves")),
        "{:?}",
        f.report.notes
    );
    assert_eq!(f.report.runs.lens_groups, 1);
    let off = fresh_frame(&r, &src, &state);
    assert_ne!(f.rgba, off.rgba, "the lens still acts");
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn a_lens_render_is_the_same_bytes_whatever_was_looked_at_first() {
    let d = developed(800, 600);
    let src = with_lens(&d.source(), VINTAGE);
    // One renderer exports cold, the other after its viewer filled the caches with the lens's groups,
    // tiles and planes.
    let cold = cpu();
    let warm = cpu();
    let st = at_100_percent(560.0, 360.0, (800, 600));
    frame(&warm, &src, &st);
    frame(&warm, &src, &cell(400.0, 300.0));
    let a = export(&cold, &src);
    let b = export(&warm, &src);
    assert!(a.rgb == b.rgb, "the export depends on what the viewer did");
    assert_eq!(a.report.runs.lens_planes, 1);
    assert_eq!(
        b.report.runs.lens_planes, 1,
        "an export makes its own planes"
    );
    // And the same twice.
    assert!(export(&cold, &src).rgb == a.rgb);
}

// ---------------------------------------------------------------------------------------------
// The lens on the device
// ---------------------------------------------------------------------------------------------

/// A night scene of defocused lights (discs of 18 px on a dark ground): the blur the detector finds, so the
/// bokeh reshaping acts.
fn bokeh_developed(w: u32, h: u32) -> Developed {
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.02).noise(2e-4, 1e-6);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.45 * (scene.white_level - scene.black)) as u16;
    let radius = 0.015 * f64::from(w);
    for fy in [0.18, 0.4, 0.62, 0.84] {
        for fx in [0.1, 0.26, 0.42, 0.58, 0.74, 0.9] {
            let (cx, cy) = (fx * f64::from(w), fy * f64::from(h));
            for y in 0..h {
                for x in 0..w {
                    let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                    if d <= radius {
                        mosaic[(y * w + x) as usize] = code;
                    }
                }
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

/// The CPU lens and the device lens make the same picture of the fit view and of the 100 % view, within the
/// tolerances of the kernels and of the colour pass; the device lens runs kernels (the counters say so) and
/// reads nothing back that the CPU lens would read.
fn the_device_lens_matches_the_cpu_lens_with(device: &Renderer, label: &str) {
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let lensed = with_lens(&src, VINTAGE);
    let reference = cpu();
    for (name, state) in [
        ("fit view", cell(576.0, 384.0)),
        ("100 % view", at_100_percent(500.0, 400.0, (1200, 800))),
    ] {
        let want = fresh_frame(&reference, &lensed, &state);
        let got = fresh_frame(device, &lensed, &state);
        let worst = worst_difference(&want, &got);
        let _ = writeln!(
            std::io::stderr(),
            "[lens] {label}, {name}: CPU lens against device lens {worst} codes"
        );
        assert!(
            worst <= 2,
            "{label}, {name}: the device lens differs by {worst} codes"
        );
        assert_eq!(
            lens_work(&got.report.runs).0,
            lens_work(&want.report.runs).0,
            "{label}, {name}: the planes are the CPU's in both"
        );
        assert!(
            got.report.runs.lens_groups + got.report.runs.lens_tiles >= 1,
            "{label}, {name}: {:?}",
            got.report.runs
        );
    }
}

#[test]
fn the_lens_on_the_device_matches_the_cpu_lens() {
    let Some(gpu) = gpu() else { return };
    the_device_lens_matches_the_cpu_lens_with(&device_lens(&gpu), "lens on the device");
}

#[test]
fn the_lens_on_the_device_after_the_raw_group_on_the_device_matches_the_cpu_lens() {
    let Some(gpu) = gpu() else { return };
    the_device_lens_matches_the_cpu_lens_with(&device_everything(&gpu), "everything on the device");
}

/// Bokeh reshaping (L1) runs on the host and L2 and L3 on the device: the picture is the CPU lens's.
#[test]
fn the_reshaping_on_the_host_feeds_the_lens_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let d = bokeh_developed(1200, 800);
    let src = plain(&d.source());
    let lensed = with_lens(
        &src,
        LensSettings {
            kind: LensKind::Vintage,
            flares: 40.0,
            edges: 60.0,
            anamorphic: 100.0,
        },
    );
    let state = cell(576.0, 384.0);
    let want = fresh_frame(&cpu(), &lensed, &state);
    let lens = want.report.lens.expect("the lens report");
    assert!(
        lens.reshaped,
        "the scene must have bokeh for the detector: {lens:?}"
    );
    for (label, renderer) in [
        ("lens on the device", device_lens(&gpu)),
        ("everything on the device", device_everything(&gpu)),
    ] {
        let got = fresh_frame(&renderer, &lensed, &state);
        assert_eq!(got.report.lens, want.report.lens, "{label}");
        let worst = worst_difference(&want, &got);
        let _ = writeln!(
            std::io::stderr(),
            "[lens] {label}, reshaped bokeh: CPU lens against device lens {worst} codes"
        );
        assert!(
            worst <= 2,
            "{label}: the device lens differs by {worst} codes"
        );
    }
}

/// With the raw group and the lens on the device the group never comes back to the host for the lens:
/// a Flares move reads back the finished frame and nothing else. (The CPU lens reads the group back.)
#[test]
fn a_slider_move_on_the_device_reads_back_the_frame_and_not_the_group() {
    let Some(gpu) = gpu() else { return };
    let d = developed(1200, 800);
    let src = plain(&d.source());
    let state = cell(576.0, 384.0);
    let frame_bytes = 576 * 384 * 4;
    let on_device = device_everything(&gpu);
    let first = frame(&on_device, &with_lens(&src, VINTAGE), &state);
    assert!(first.sharp);
    let before = gpu.counters();
    let moved = frame(
        &on_device,
        &with_lens(
            &src,
            LensSettings {
                flares: 61.0,
                ..VINTAGE
            },
        ),
        &state,
    );
    let read = gpu.counters().since(before);
    assert_eq!(moved.report.runs.lens_fields, 0, "{:?}", moved.report.runs);
    assert!(
        read.readback_bytes <= 2 * frame_bytes,
        "a Flares move read {} bytes back (the frame is {frame_bytes})",
        read.readback_bytes
    );
    // The CPU lens on a raw group made on the device reads the group (3 planes of 600 x 400) back first.
    let cpu_lens = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            gpu_stages: GpuStages::ALL.difference(GpuStages::LENS),
            ..options()
        },
    )
    .expect("render thread");
    let before = gpu.counters();
    let _ = frame(&cpu_lens, &with_lens(&src, VINTAGE), &state);
    let read_cpu = gpu.counters().since(before);
    assert!(
        read_cpu.readback_bytes > 2 * frame_bytes,
        "the CPU lens read only {} bytes back",
        read_cpu.readback_bytes
    );
}

/// All three amounts at 0: no lens pass at all on the device either, and the bytes of a render without the
/// field.
#[test]
fn zero_amounts_are_the_render_without_the_field_with_the_lens_on_the_device() {
    let Some(gpu) = gpu() else { return };
    zero_amounts_are_the_render_without_the_field_with(&device_lens(&gpu));
    zero_amounts_are_the_render_without_the_field_with(&device_everything(&gpu));
}

/// The codes of a 16-bit PNG.
fn png16(bytes: &[u8]) -> (usize, usize, Vec<u32>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes.to_vec()));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    let codes = buf[..info.buffer_size()]
        .chunks_exact(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
        .collect();
    (info.width as usize, info.height as usize, codes)
}

/// An export through the export tail with the lens on the device (the kernels read the raw tiles where the raw
/// chain left them and hand the sampler the lens tiles) is the CPU export within the kernels' tolerances, and
/// it is made wholly on the device.
#[test]
fn an_export_with_the_lens_on_the_device_is_the_cpu_export_within_tolerance() {
    let Some(gpu) = gpu() else { return };
    let d = developed(800, 600);
    let patch: SettingsPatch = serde_json::from_str(
        r#"{ "lens": { "kind": "vintage", "flares": 60, "edges": 50, "anamorphic": 50 } }"#,
    )
    .expect("patch");
    d.session.update_settings(d.id, &patch).expect("settings");
    let photo = ExportPhoto::from_session(&d.session, d.id).expect("photo");
    let home = tempfile::tempdir().expect("home");
    let run = |renderer: Renderer| {
        let out = tempfile::tempdir().expect("out");
        let ctx = ExportContext::new(renderer).with_home(Some(home.path().to_path_buf()));
        let opts = ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Sixteen,
            destination: Some(out.path().to_path_buf()),
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
        let bytes = std::fs::read(&report.path).expect("read");
        (report, png16(&bytes))
    };
    let (want_report, (w, h, want)) = run(cpu());
    assert_eq!(want_report.device, ComputeDevice::Cpu);
    let on_device = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            gpu_stages: GpuStages::ALL,
            full_tile_px: 256,
            ..options()
        },
    )
    .expect("render thread");
    let (report, (w2, h2, got)) = run(on_device);
    assert_eq!(report.device, ComputeDevice::Gpu, "{:?}", report.notes);
    assert!(
        !report.notes.iter().any(|n| n.contains("graphics card")),
        "{:?}",
        report.notes
    );
    assert_eq!((w, h), (w2, h2));
    let (mut worst, mut over) = (0u32, 0usize);
    for (a, b) in want.iter().zip(&got) {
        let dif = a.abs_diff(*b);
        worst = worst.max(dif);
        over += usize::from(dif > 256);
    }
    let _ = writeln!(
        std::io::stderr(),
        "[lens] export tail, CPU against device: worst {worst} of 65535, {over} of {} samples over 256",
        want.len()
    );
    assert!(worst <= 2500, "the device export differs by {worst} codes");
    assert!(
        (over as f64) <= 1e-3 * want.len() as f64,
        "{over} samples differ by more than 256 codes"
    );
}

/// D7 for the lens kernels (`docs/17-gpu.md`): viewport frames of the same photo, lens included, rendered from
/// inside the export (at every progress report, so between its tiles) change not a byte of the export, and
/// the frames equal the frame rendered alone.
#[test]
fn viewport_jobs_between_the_lens_tiles_of_an_export_change_nothing() {
    let Some(gpu) = gpu() else { return };
    let d = developed(800, 600);
    let patch: SettingsPatch = serde_json::from_str(
        r#"{ "lens": { "kind": "vintage", "flares": 60, "edges": 50, "anamorphic": 50 } }"#,
    )
    .expect("patch");
    d.session.update_settings(d.id, &patch).expect("settings");
    let photo = ExportPhoto::from_session(&d.session, d.id).expect("photo");
    let src = d.source();
    let state = cell(576.0, 384.0);
    let home = tempfile::tempdir().expect("home");
    let renderer = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(&gpu)),
            gpu_stages: GpuStages::ALL,
            full_tile_px: 256,
            ..options()
        },
    )
    .expect("render thread");
    let ctx = ExportContext::new(renderer).with_home(Some(home.path().to_path_buf()));
    let export_once = |progress: Option<&(dyn Fn(f32) + Sync)>| {
        let out = tempfile::tempdir().expect("out");
        let opts = ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Sixteen,
            destination: Some(out.path().to_path_buf()),
            if_exists: IfExists::Replace,
            ..ExportOptions::default()
        };
        let report = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(opts),
            &ExportControl {
                cancel: CancelToken::new(),
                progress,
            },
        )
        .expect("export");
        assert_eq!(report.device, ComputeDevice::Gpu, "{:?}", report.notes);
        std::fs::read(&report.path).expect("read")
    };
    let alone = fresh_frame(ctx.renderer(), &src, &state).rgba;
    let reference = export_once(None);
    let injected: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let between = |_: f32| {
        if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
            let f = fresh_frame(ctx.renderer(), &src, &state).rgba;
            injected.lock().expect("lock").push(f);
        }
    };
    let bytes = export_once(Some(&between));
    assert!(
        bytes == reference,
        "viewport jobs between the tiles changed the export"
    );
    let injected = injected.into_inner().expect("lock");
    assert!(!injected.is_empty(), "no progress was reported");
    for f in &injected {
        assert!(
            *f == alone,
            "a frame rendered during the export differs from the frame rendered alone"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Timing (not part of the suite)
// ---------------------------------------------------------------------------------------------

/// Prints one timing line: the wall time and, when the request's report is given, what it ran and how
/// the time divides between the raw stages and the lens.
fn say_timing(what: &str, t: std::time::Instant, report: Option<&ym_core::pipeline::RenderReport>) {
    let _ = writeln!(
        std::io::stderr(),
        "[timing] {what}: {:.0} ms{}",
        t.elapsed().as_secs_f64() * 1000.0,
        report.map_or(String::new(), |r| format!(
            " (lens planes {}, groups {}, tiles {}, lens {:.0} ms; raw groups {}, tiles {}, raw {:.0} ms)",
            r.runs.lens_planes,
            r.runs.lens_groups,
            r.runs.lens_tiles,
            r.timings.lens_ms,
            r.runs.raw_groups,
            r.runs.tiles,
            r.timings.raw_ms
        ))
    );
}

/// The cost of the lens on the views of a 24 MP photo on the CPU, printed as it goes. Run by hand:
/// `cargo test -p ym-core --test render_lens -- --ignored --nocapture lens_timing_of_the_views_on_24_megapixels`.
/// The numbers depend on the machine's load and on the profile the tests are built with; the test
/// asserts nothing about them.
#[test]
#[ignore = "takes minutes: a 24 MP photo on the CPU"]
fn lens_timing_of_the_views_on_24_megapixels() {
    let t = std::time::Instant::now();
    let d = developed(6000, 4000);
    say_timing("the 24 MP synthetic file, written and developed", t, None);
    let src = d.source();
    let lensed = with_lens(&src, VINTAGE);
    let r = cpu();
    let fit = cell(1500.0, 1000.0);
    let view = |what: &str, src: &PhotoSource, state: &ViewportState| {
        let t = std::time::Instant::now();
        let f = fresh_frame(&r, src, state);
        say_timing(what, t, Some(&f.report));
    };
    view("fit view, no lens (cold: the raw group at 1/4)", &src, &fit);
    view(
        "fit view, Vintage 60/50/50, first (planes + lens group)",
        &lensed,
        &fit,
    );
    let flares = with_lens(
        &src,
        LensSettings {
            flares: 61.0,
            ..VINTAGE
        },
    );
    view("fit view, a Flares move (lens group)", &flares, &fit);
    let edges = with_lens(
        &src,
        LensSettings {
            edges: 51.0,
            ..VINTAGE
        },
    );
    view("fit view, an Edges move (lens group)", &edges, &fit);
    view("fit view, back to no lens (cached raw group)", &src, &fit);
    let st = at_100_percent(1500.0, 1000.0, (6000, 4000));
    view("100 % view of 1500 x 1000, no lens (raw tiles)", &src, &st);
    view(
        "100 % view, Vintage 60/50/50 (lens tiles; raw tiles cached)",
        &lensed,
        &st,
    );
    view("100 % view, a Flares move (lens tiles)", &flares, &st);
}

/// The cost of an export with and without the lens on a 24 MP photo on the CPU (see
/// [`lens_timing_of_the_views_on_24_megapixels`]).
#[test]
#[ignore = "takes minutes: a 24 MP photo on the CPU"]
fn lens_timing_of_the_export_on_24_megapixels() {
    let d = developed(6000, 4000);
    let src = d.source();
    let r = cpu();
    let t = std::time::Instant::now();
    let e = export(&r, &src);
    say_timing("export, no lens", t, Some(&e.report));
    let t = std::time::Instant::now();
    let e = export(&r, &with_lens(&src, VINTAGE));
    say_timing("export, Vintage 60/50/50", t, Some(&e.report));
}
