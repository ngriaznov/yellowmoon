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
use ym_core::pipeline::{
    FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions,
    RenderedFrame, Renderer, StageRuns, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{DevelopSettings, LensKind, LensSettings};
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

/// Patches of many levels, the brightest above the sensor's clip, with a little noise: edges for the
/// field optics and light for the flares.
fn developed(w: u32, h: u32) -> Developed {
    let scene =
        Scene::new(w, h, light(PROFILE, 5500.0), Layout::Patchwork, 1.6).noise(2e-4, 1e-6);
    Developed::new(&scene, MODEL)
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
            !f.report.notes.iter().chain(&e.report.notes).any(|n| n.contains("lens")),
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
    assert!(runs.raw_work() > 0);
    assert!(first.report.timings.lens_ms > 0.0);
    let worst = worst_difference(&before, &first);
    assert!(worst >= 4, "the lens moved the picture by {worst} codes only");
    let lens = first.report.lens.expect("the lens report");
    assert!(
        lens.reshaped || lens.reason.is_some(),
        "the report says why when it did not reshape: {lens:?}"
    );
    // The stats the webview gets carry it.
    assert_eq!(first.report.stats().lens.map(|l| l.reshaped), Some(lens.reshaped));

    // The same request again: the finished frame is served from the cache and says the same.
    let again = frame(r, &lensed_src, &state);
    assert_eq!(again.rgba, first.rgba);
    assert_eq!(again.report.runs.frame_hits, 1);
    assert_eq!(lens_work(&again.report.runs), (0, 0, 0));
    assert_eq!(again.report.lens, first.report.lens, "a cached frame forgets the report");

    // A slider move: new pixels, from the cached raw group and the cached planes. Only the lens pass
    // and the colour pass run.
    for (name, moved) in [
        ("Flares", LensSettings { flares: 61.0, ..VINTAGE }),
        ("Edges", LensSettings { edges: 51.0, ..VINTAGE }),
        ("Anamorphic", LensSettings { anamorphic: 51.0, ..VINTAGE }),
    ] {
        let f = frame(r, &with_lens(&src, moved), &state);
        let runs = f.report.runs;
        assert_eq!(runs.raw_work(), 0, "{name}: no raw stage ran: {runs:?}");
        assert_eq!(runs.sigma_planes, 0, "{name}: {runs:?}");
        assert_eq!(runs.lens_planes, 0, "{name}: the planes were made again: {runs:?}");
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
    let _ = writeln!(std::io::stderr(), "[lens] fit view, CPU against device: {worst} codes");
    assert!(worst <= 2, "the device colour pass differs by {worst} codes");
    assert_eq!(on_cpu.report.runs.lens_groups, on_device.report.runs.lens_groups);
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
        &with_lens(&src, LensSettings { flares: 20.0, ..VINTAGE }),
        &state,
    )
    .report
    .runs;
    assert_eq!(b.lens_planes, 0);
    let c = frame(
        &r,
        &with_lens(&src, LensSettings { kind: LensKind::Petzval, ..VINTAGE }),
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
}

fn reference_of(r: &Renderer, src: &PhotoSource, lens: LensSettings, axis: Axis) -> Reference {
    let token = CancelToken::new();
    let (w, h) = r.final_size(src, &token).expect("size");
    // The camera orientation of the synthetic file is the sensor's, so a quarter turn swaps the sides.
    let (w, h) = if axis == Axis::Y { (h, w) } else { (w, h) };
    let raw = r
        .camera_region(src, Rect::whole(w, h), false, &token)
        .expect("the raw group");
    let camera = CameraResponse::from_curves(
        &src.profile.load_curves(&data_root()).expect("the curves"),
    )
    .expect("the camera");
    let g0 = src.estimates.initial_gains().expect("g0");
    let frame = FrameInfo {
        width: w,
        height: h,
        axis,
        f_number: src.raw.meta.f_number.filter(|f| f.is_finite() && *f > 0.0),
        clip: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
    };
    assert!(src.raw.meta.focus.is_none(), "the synthetic file has no focus point");
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
    assert_eq!(plan.lens_plane_scale(), 1, "a test frame is its own lens plane");
    let planes = LensPlanes::build(&plan, &image, None, &|| false).expect("planes");
    let out = ym_lens::apply(&plan, &planes, 1, &image, PixelRect::new(0, 0, w, h)).expect("apply");
    Reference {
        raw,
        out,
        width: w,
        height: h,
    }
}

impl Reference {
    fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let n = self.width * self.height;
        let i = y * self.width + x;
        [self.out[i], self.out[n + i], self.out[2 * n + i]]
    }
}

/// An export tile (the raw region grown by the halo, one call) is the whole-frame `apply`, bit for bit,
/// for rectangles that cross the 512 px lines of the tiles and the borders of the frame.
fn an_export_region_is_the_whole_frame_apply_with(r: &Renderer, axis: Axis) {
    let d = developed(800, 600);
    let mut src = with_lens(&d.source(), VINTAGE);
    if axis == Axis::Y {
        src = edit(&src, |s| s.orientation = ym_core::settings::Orientation::Deg90);
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
        assert_eq!(worst, 0.0, "{rect:?} differs from the whole-frame apply by {worst}");
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
    for (x, y) in [(3, 5), (100, 100), (511, 300), (512, 300), (513, 301), (700, 590), (400, 511), (400, 512)] {
        let got = r
            .camera_rgb_at(&src, &st, (x as f64 + 0.5, y as f64 + 0.5), &token)
            .expect("reading")
            .expect("inside the photo");
        let want = reference.pixel(x, y);
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
            for c in 0..3 {
                let got = f64::from(f.rgba[(y * 800 + x) * 4 + c]);
                worst = worst.max((got - enc[c] * 255.0).abs());
            }
        }
    }
    let _ = writeln!(
        std::io::stderr(),
        "[lens] 100 % view against the export: worst {worst:.4} codes"
    );
    assert!(worst <= 1.0 + 1e-6, "the view and the export differ by {worst} codes");
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
    assert!(first.report.runs.lens_groups >= 1, "{:?}", first.report.runs);
    let made = r.prepare_tiles(&src, &st, &token).expect("tiles");
    assert!(made > 0);
    let sharp = r.viewport_frame(&src, &st, &coarse, &token).expect("frame");
    assert!(sharp.sharp, "the lens tiles are ready, so the frame is sharp");
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
    assert_eq!(f.report.runs.lens_planes, 0, "the planes stayed: {:?}", f.report.runs);
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
        .sample_pixel(&with_lens(&src, VINTAGE), &st, (100.5, 100.5), &CancelToken::new())
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
    let tile_with = r.look_thumbnail_png(&lensed, &neutral, &token).expect("tile");
    assert_ne!(tile, tile_with, "the lens did not reach the look tile");
    let total = r.counters(src.id);
    assert_eq!(total.lens_planes, 1, "one set of planes for all of them: {total:?}");
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
    assert_eq!(stats.reason, Some(ym_core::pipeline::LensReshapeReason::NotAsked));
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
    assert!(b.report.notes.iter().any(|n| n.contains("lens")), "{:?}", b.report.notes);
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
    assert_eq!(b.report.runs.lens_planes, 1, "an export makes its own planes");
    // And the same twice.
    assert!(export(&cold, &src).rgb == a.rgb);
}
