//! Output sharpening in the render graph (`docs/01` Render graph, `docs/algorithms/04`; the owner's
//! decision of 2026-10-13): the last stage of every display-referred picture. Moving the Output
//! slider re-runs only that stage; a view at 100 % shows the picture the export writes; a pan, a
//! letterbox and the Before view change nothing they should not.
//!
//! Every test runs twice: on the CPU (`Renderer::neutral_only`, the reference) and on the device
//! (`*_on_the_device`: the sampler, the colour chain, output sharpening and C13 on Mesa lavapipe,
//! G-05; the raw chain stays on the CPU so that the difference is the view stages alone). The two must
//! meet the same assertions: the same run counts, the same agreement with the export within a code.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_color::encode::OutputSpace;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, NeutralOnly, PhotoSource, RenderOptions, RenderedFrame,
    Renderer, StageRuns, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::DevelopSettings;
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::resample::sample_catmull_rom;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

/// The device, in poison mode (every pool buffer starts as NaN, so a kernel that reads what it did
/// not write fails the comparison). Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1`
/// turns that into a failure.
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
                "\n* GPU TESTS SKIPPED in render_sharpen: {e}\n"
            );
            None
        }
    }
}

/// A renderer whose view stages (sampler, colour chain, output sharpening, C13) run on `gpu`.
fn device(gpu: &Arc<Gpu>) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(Arc::clone(gpu)),
            gpu_stages: GpuStages::SAMPLER
                .union(GpuStages::COLOR)
                .union(GpuStages::OUTPUT_SHARPEN)
                .union(GpuStages::ENCODE),
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn developed(w: u32, h: u32) -> Developed {
    // Clean patches with a little noise: hard edges for the stage to work on.
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-6);
    Developed::new(&scene, MODEL)
}

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn with_amount(src: &PhotoSource, pct: f64) -> PhotoSource {
    edit(src, |s| s.output_sharpening = pct)
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

fn cell(w: f64, h: f64) -> ViewportState {
    ViewportState::new(ViewportRect::cell(w, h, 1.0))
}

fn at_100_percent(w: f64, h: f64, image: (usize, usize)) -> ViewportState {
    let mut st = cell(w, h);
    st.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    st
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

/// Mean absolute difference between neighbouring pixels along a row: a measure of local contrast.
fn local_contrast(f: &RenderedFrame) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for y in 0..f.height {
        for x in 1..f.width {
            let (a, b) = (
                &f.rgba[(y * f.width + x - 1) * 4..][..3],
                &f.rgba[(y * f.width + x) * 4..][..3],
            );
            sum += (0..3)
                .map(|c| (f64::from(a[c]) - f64::from(b[c])).abs())
                .sum::<f64>();
            n += 1;
        }
    }
    sum / n as f64
}

fn moving_the_slider_re_runs_only_the_stage_and_zero_runs_no_stage_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = d.source();
    assert_eq!(src.settings.output_sharpening, 100.0);
    let state = cell(576.0, 384.0);
    let runs = |src: &PhotoSource| frame(r, src, &state).report.runs;

    let first = runs(&src);
    assert_eq!(
        (first.output_passes, first.color_passes, first.display_hits),
        (1, 1, 0),
        "{first:?}"
    );

    // The slider moves: the display-linear frame under the stage is served from its cache, so
    // nothing is sampled, coloured or demosaiced again; only the stage and the encode run.
    let moved = runs(&with_amount(&src, 60.0));
    assert_eq!(
        moved,
        StageRuns {
            output_passes: 1,
            display_hits: 1,
            ..StageRuns::default()
        }
    );
    // Back to a value that was shown: the finished frame is cached.
    assert_eq!(
        runs(&src),
        StageRuns {
            frame_hits: 1,
            ..StageRuns::default()
        }
    );

    // Zero is no stage: the plain road (sample and colour pass), no output pass. A plain frame
    // has nothing for the stage to read, so the way back to a sharpened one colours it again once.
    let off = runs(&with_amount(&src, 0.0));
    assert_eq!(
        (off.output_passes, off.color_passes, off.display_hits),
        (0, 1, 0),
        "{off:?}"
    );
    assert_eq!(off.raw_work(), 0, "no raw stage ran: {off:?}");
    // Below a tenth of a percent the stage has no amount either (spec 04: A < 0.001).
    let tiny = runs(&with_amount(&src, 0.05));
    assert_eq!(tiny.output_passes, 0, "{tiny:?}");
    // The display-linear frame of 60 % is still cached from before: no colour pass.
    let again = runs(&with_amount(&src, 45.0));
    assert_eq!(
        (again.output_passes, again.display_hits, again.color_passes),
        (1, 1, 0),
        "{again:?}"
    );
    // The totals say the same: the raw group ran once for all of it.
    assert_eq!(r.counters(src.id).raw_groups, 1);
}

#[test]
fn moving_the_slider_re_runs_only_the_stage_and_zero_runs_no_stage() {
    moving_the_slider_re_runs_only_the_stage_and_zero_runs_no_stage_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn moving_the_slider_re_runs_only_the_stage_and_zero_runs_no_stage_on_the_device() {
    let Some(gpu) = gpu() else { return };
    moving_the_slider_re_runs_only_the_stage_and_zero_runs_no_stage_with(&device(&gpu));
}

fn the_amount_changes_the_picture_and_zero_is_the_unsharpened_one_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = d.source();
    let state = cell(576.0, 384.0);
    let off = fresh_frame(r, &with_amount(&src, 0.0), &state);
    let half = fresh_frame(r, &with_amount(&src, 50.0), &state);
    let full = fresh_frame(r, &with_amount(&src, 100.0), &state);
    let double = fresh_frame(r, &with_amount(&src, 200.0), &state);
    // More amount, more local contrast, monotonically.
    let c: Vec<f64> = [&off, &half, &full, &double]
        .iter()
        .map(|f| local_contrast(f))
        .collect();
    assert!(c[0] < c[1] && c[1] < c[2] && c[2] < c[3], "{c:?}");
    // The stage moves values a little on average (the screen preset at a fit scale is gentle; the
    // few pixels on a hard edge of the synthetic patchwork move more).
    let mean = off
        .rgba
        .iter()
        .zip(&full.rgba)
        .map(|(&a, &b)| f64::from((i32::from(a) - i32::from(b)).abs()))
        .sum::<f64>()
        / off.rgba.len() as f64;
    assert!(
        (0.01..3.0).contains(&mean),
        "mean change {mean} codes, {c:?}"
    );
    assert!(worst_difference(&off, &full) >= 1);
    // The same request twice is the same bytes.
    assert!(
        fresh_frame(r, &src, &state).rgba == full.rgba,
        "the same request gave other pixels"
    );
}

#[test]
fn the_amount_changes_the_picture_and_zero_is_the_unsharpened_one() {
    the_amount_changes_the_picture_and_zero_is_the_unsharpened_one_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn the_amount_changes_the_picture_and_zero_is_the_unsharpened_one_on_the_device() {
    let Some(gpu) = gpu() else { return };
    the_amount_changes_the_picture_and_zero_is_the_unsharpened_one_with(&device(&gpu));
}

fn the_before_view_is_not_sharpened_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = d.source();
    let mut state = cell(576.0, 384.0);
    state.before = true;
    let a = frame(r, &with_amount(&src, 0.0), &state);
    let b = frame(r, &with_amount(&src, 200.0), &state);
    assert_eq!(a.rgba, b.rgba);
    assert_eq!(a.report.runs.output_passes + b.report.runs.output_passes, 0);
}

#[test]
fn the_before_view_is_not_sharpened() {
    the_before_view_is_not_sharpened_with(&Renderer::neutral_only().expect("render thread"));
}

#[test]
fn the_before_view_is_not_sharpened_on_the_device() {
    let Some(gpu) = gpu() else { return };
    the_before_view_is_not_sharpened_with(&device(&gpu));
}

/// `slack` is what the renderer's own arithmetic may add to the one code the dither moves a value by:
/// 0 for the CPU, which is the reference (first measured 0.9984), and for the device the f32
/// accumulation of the stage on a hard edge.
fn a_view_at_100_percent_is_the_sharpened_picture_the_export_writes_with(r: &Renderer, slack: f64) {
    let d = developed(800, 600);
    let src = d.source();
    let full = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Viewport),
            },
            &CancelToken::new(),
        )
        .expect("full render");
    assert_eq!((full.width, full.height), (800, 600));
    // What the export does to those pixels: the stage at s = 1, then C13.
    let sharpened = OutputSharpen::screen()
        .with_amount_pct(100.0)
        .apply_interleaved(800, 600, &full.rgb, 1.0)
        .expect("sharpen");
    let st = at_100_percent(800.0, 600.0, (800, 600));
    let f = frame(r, &src, &st);
    assert_eq!(f.report.scale, 1, "the full-resolution tiles");
    assert_eq!(f.report.runs.output_passes, 1);
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0, 0);
    let mut over_one = 0usize;
    let mut moved = 0usize;
    for y in 0..600 {
        for x in 0..800 {
            let i = (y * 800 + x) * 3;
            let lin = [sharpened[i], sharpened[i + 1], sharpened[i + 2]].map(f64::from);
            let enc = OutputSpace::DisplayP3.encode_clamped(lin);
            let plain = OutputSpace::DisplayP3
                .encode_clamped([full.rgb[i], full.rgb[i + 1], full.rgb[i + 2]].map(f64::from));
            for c in 0..3 {
                let got = f64::from(f.rgba[(y * 800 + x) * 4 + c]);
                let d = (got - enc[c] * 255.0).abs();
                if d > worst {
                    (worst, worst_at) = (d, (x, y, c));
                }
                over_one += usize::from(d > 1.0);
                if (enc[c] - plain[c]).abs() * 255.0 > 1.5 {
                    moved += 1;
                }
            }
        }
    }
    let _ = writeln!(
        std::io::stderr(),
        "[G-05] 100 % view against the export: worst {worst:.4} codes at {worst_at:?}, {over_one} samples over 1.0, {moved} samples moved by the stage"
    );
    assert!(
        worst <= 1.0 + slack,
        "the view and the export differ by {worst} codes (the dither moves a value by less than one)"
    );
    // A shift of the whole picture could hide under the slack; rank-count flips touch a handful
    // of samples (measured: 0 on the CPU, 2 on lavapipe, 5 on Metal).
    assert!(
        over_one <= 50,
        "{over_one} samples differ from the export by more than one code"
    );
    assert!(
        moved > 100,
        "the stage did change the picture ({moved} samples)"
    );
}

#[test]
fn a_view_at_100_percent_is_the_sharpened_picture_the_export_writes() {
    a_view_at_100_percent_is_the_sharpened_picture_the_export_writes_with(
        &Renderer::neutral_only().expect("render thread"),
        0.0,
    );
}

#[test]
fn a_view_at_100_percent_is_the_sharpened_picture_the_export_writes_on_the_device() {
    let Some(gpu) = gpu() else { return };
    let dev = device(&gpu);
    // The bound is twice the excess over the CPU's one code, per adapter kind. Software (lavapipe):
    // first measured 1.1011 codes at one sample of 1.44 million (0.3.0: 1.0967, two samples), so
    // 0.2. Hardware (Metal, 498a078, macos-14 and macos-15 alike): 1.2039 codes at five samples, so
    // 0.41. The excess is not the stage (device against CPU on the same input: 3e-7 linear) but its
    // halo limiter's rank counts, which flip on last-bit differences between the device's f32
    // colour chain and the CPU's f64 one: one ulp of input noise in half the samples moves the CPU
    // stage against itself by 0.197 codes. It lands over one code only where the dither is near
    // its extreme.
    let slack = if gpu.info().is_software { 0.2 } else { 0.41 };
    a_view_at_100_percent_is_the_sharpened_picture_the_export_writes_with(&dev, slack);
    // And the device's frame is the CPU's frame within the codes of the colour pass's parity test
    // (two; measured: one).
    let d = developed(800, 600);
    let src = d.source();
    let cpu = Renderer::neutral_only().expect("render thread");
    let st = at_100_percent(800.0, 600.0, (800, 600));
    let (a, b) = (frame(&cpu, &src, &st), frame(&dev, &src, &st));
    let worst = worst_difference(&a, &b);
    let _ = writeln!(
        std::io::stderr(),
        "[G-05] {}: the 100 % view on the device differs from the CPU's by {worst} codes",
        gpu.info().name
    );
    assert!(worst <= 2);
}

fn a_pan_at_100_percent_is_a_window_of_the_whole_picture_with(r: &Renderer) {
    // The view is grown by the stage's receptive field, so what it shows is what the whole
    // picture shows there: no seam where the cell ends in the middle of the photo.
    let d = developed(800, 600);
    let src = d.source();
    let image = (800, 600);
    let whole = frame(r, &src, &at_100_percent(800.0, 600.0, image));
    for centre in [[0.5, 0.5], [0.3, 0.62], [0.8, 0.25], [0.0, 0.0], [1.0, 1.0]] {
        let mut st = at_100_percent(240.0, 180.0, image);
        st.centre = centre;
        let part = frame(r, &src, &st);
        // The window's origin in the picture (whole pixels at 100 %).
        let (ox, oy) = st.view_to_final(image).apply(0.5, 0.5);
        let (ox, oy) = ((ox - 0.5).round() as usize, (oy - 0.5).round() as usize);
        let mut worst = 0;
        for y in 0..part.height {
            for x in 0..part.width {
                for c in 0..3 {
                    let a = i32::from(part.rgba[(y * part.width + x) * 4 + c]);
                    let b = i32::from(whole.rgba[((y + oy) * 800 + x + ox) * 4 + c]);
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert!(
            worst <= 1,
            "centre {centre:?}: window at ({ox}, {oy}) differs from the whole picture by {worst} codes"
        );
    }
}

#[test]
fn a_pan_at_100_percent_is_a_window_of_the_whole_picture() {
    a_pan_at_100_percent_is_a_window_of_the_whole_picture_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn a_pan_at_100_percent_is_a_window_of_the_whole_picture_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_pan_at_100_percent_is_a_window_of_the_whole_picture_with(&device(&gpu));
}

fn a_letterbox_changes_nothing_inside_the_photo_with(r: &Renderer) {
    // 800 x 600 in a 1000 x 300 cell at Fit: half size, with 300 px of paper on each side. The
    // stage works on the photo's own pixels and mirrors at its edge, as it does at the edge of an
    // export, so the photo is what the same view without a letterbox shows.
    let d = developed(800, 600);
    let src = d.source();
    let wide = frame(r, &src, &cell(1000.0, 300.0));
    let exact = frame(r, &src, &cell(400.0, 300.0));
    assert_eq!((wide.width, wide.height), (1000, 300));
    assert_eq!((exact.width, exact.height), (400, 300));
    let background = r.options().background;
    for y in 0..300 {
        for x in 0..1000 {
            let p = &wide.rgba[(y * 1000 + x) * 4..][..4];
            if !(300..700).contains(&x) {
                assert_eq!(&p[..3], &background, "letterbox at ({x}, {y})");
            }
        }
    }
    let mut worst = 0;
    for y in 0..300 {
        for x in 0..400 {
            for c in 0..3 {
                let a = i32::from(wide.rgba[(y * 1000 + x + 300) * 4 + c]);
                let b = i32::from(exact.rgba[(y * 400 + x) * 4 + c]);
                worst = worst.max((a - b).abs());
            }
        }
    }
    assert!(worst <= 1, "the photo differs by {worst} codes");
    // And it really is the sharpened picture there.
    let plain = frame(r, &with_amount(&src, 0.0), &cell(400.0, 300.0));
    assert!(worst_difference(&plain, &exact) >= 1);
}

#[test]
fn a_letterbox_changes_nothing_inside_the_photo() {
    a_letterbox_changes_nothing_inside_the_photo_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn a_letterbox_changes_nothing_inside_the_photo_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_letterbox_changes_nothing_inside_the_photo_with(&device(&gpu));
}

fn thumbnails_and_look_tiles_are_sharpened_at_their_own_scale_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = d.source();
    let token = CancelToken::new();
    let on = r.thumbnail_png(&src, &token).expect("thumbnail");
    let off = r
        .thumbnail_png(&with_amount(&src, 0.0), &token)
        .expect("thumbnail");
    assert_ne!(on, off, "the thumbnail follows the slider");
    // Deterministic.
    assert_eq!(on, r.thumbnail_png(&src, &token).expect("again"));
    let tile = r
        .look_thumbnail_png(&src, &ym_looks::library::LookId::builtin("neutral"), &token)
        .expect("tile");
    assert_eq!(tile, on, "a look tile is the thumbnail through its look");
    // The scale of a 240 px thumbnail of a 1200 px picture.
    assert!((scale_for_edges(240, 1200) - 0.2).abs() < 1e-6);
}

#[test]
fn thumbnails_and_look_tiles_are_sharpened_at_their_own_scale() {
    thumbnails_and_look_tiles_are_sharpened_at_their_own_scale_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn thumbnails_and_look_tiles_are_sharpened_at_their_own_scale_on_the_device() {
    let Some(gpu) = gpu() else { return };
    thumbnails_and_look_tiles_are_sharpened_at_their_own_scale_with(&device(&gpu));
}

fn a_view_magnified_above_100_percent_is_the_sharpened_export_picture_made_bigger_with(
    r: &Renderer,
) {
    // The stage runs on the picture at 100 % (the pixels an export holds, s = 1) and the sharpened
    // picture is magnified: the view at 200 % is the sharpened export, Catmull-Rom filtered.
    let d = developed(800, 600);
    let src = d.source();
    let full = r
        .render_full(
            &src,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Viewport),
            },
            &CancelToken::new(),
        )
        .expect("full render");
    let sharpened = OutputSharpen::screen()
        .with_amount_pct(100.0)
        .apply_interleaved(800, 600, &full.rgb, 1.0)
        .expect("sharpen");
    let planes: Vec<Vec<f32>> = (0..3)
        .map(|c| sharpened.chunks_exact(3).map(|p| p[c]).collect())
        .collect();
    let image = (800, 600);
    for (zoom, centre) in [
        (200.0, [0.5, 0.5]),
        (400.0, [0.3, 0.7]),
        (125.0, [0.8, 0.2]),
    ] {
        let mut st = cell(400.0, 300.0);
        st.set_mode(ViewportMode::Percent { value: zoom }, image);
        st.centre = centre;
        let f = frame(r, &src, &st);
        assert_eq!(f.report.runs.output_passes, 1, "{zoom} %");
        let to_final = st.view_to_final(image);
        let mut worst = 0.0f64;
        for y in (0..300).step_by(3) {
            for x in (0..400).step_by(3) {
                let (fx, fy) = to_final.apply(x as f64 + 0.5, y as f64 + 0.5);
                let lin: [f64; 3] = std::array::from_fn(|c| {
                    f64::from(sample_catmull_rom(&planes[c], 800, 600, fx, fy))
                });
                let enc = OutputSpace::DisplayP3.encode_clamped(lin);
                for (c, expected) in enc.iter().enumerate() {
                    let got = f64::from(f.rgba[(y * 400 + x) * 4 + c]);
                    worst = worst.max((got - expected * 255.0).abs());
                }
            }
        }
        assert!(
            worst <= 1.0,
            "{zoom} %: the view differs from the sharpened export magnified by {worst} codes"
        );
        // And it is not the plain magnified picture.
        let plain = frame(r, &with_amount(&src, 0.0), &st);
        assert!(worst_difference(&plain, &f) >= 1, "{zoom} %");
    }
}

#[test]
fn a_view_magnified_above_100_percent_is_the_sharpened_export_picture_made_bigger() {
    a_view_magnified_above_100_percent_is_the_sharpened_export_picture_made_bigger_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn a_view_magnified_above_100_percent_is_the_sharpened_export_picture_made_bigger_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_view_magnified_above_100_percent_is_the_sharpened_export_picture_made_bigger_with(&device(
        &gpu,
    ));
}

fn moving_the_slider_on_a_magnified_view_re_runs_only_the_stage_with(r: &Renderer) {
    let d = developed(800, 600);
    let src = d.source();
    let mut st = cell(400.0, 300.0);
    st.set_mode(ViewportMode::Percent { value: 300.0 }, (800, 600));
    let first = frame(r, &src, &st).report.runs;
    assert_eq!(
        (first.output_passes, first.color_passes),
        (1, 1),
        "{first:?}"
    );
    let moved = frame(r, &with_amount(&src, 50.0), &st).report.runs;
    assert_eq!(
        moved,
        StageRuns {
            output_passes: 1,
            display_hits: 1,
            ..StageRuns::default()
        }
    );
}

#[test]
fn moving_the_slider_on_a_magnified_view_re_runs_only_the_stage() {
    moving_the_slider_on_a_magnified_view_re_runs_only_the_stage_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn moving_the_slider_on_a_magnified_view_re_runs_only_the_stage_on_the_device() {
    let Some(gpu) = gpu() else { return };
    moving_the_slider_on_a_magnified_view_re_runs_only_the_stage_with(&device(&gpu));
}

fn a_magnified_view_pans_and_letterboxes_like_the_whole_picture_with(r: &Renderer) {
    // 200 x 150 in an 800 x 300 cell: the photo is shown at 200 % (400 x 300) with 200 px of paper
    // on each side. The photo is what the same view without paper shows, and the paper is paper.
    let d = developed(200, 150);
    let src = d.source();
    let wide = frame(r, &src, &cell(800.0, 300.0));
    let exact = frame(r, &src, &cell(400.0, 300.0));
    let background = r.options().background;
    for y in 0..300 {
        for x in 0..800 {
            if !(200..600).contains(&x) {
                assert_eq!(
                    &wide.rgba[(y * 800 + x) * 4..][..3],
                    &background,
                    "paper at ({x}, {y})"
                );
            }
        }
    }
    let mut worst = 0;
    for y in 0..300 {
        for x in 0..400 {
            for c in 0..3 {
                let a = i32::from(wide.rgba[(y * 800 + x + 200) * 4 + c]);
                let b = i32::from(exact.rgba[(y * 400 + x) * 4 + c]);
                worst = worst.max((a - b).abs());
            }
        }
    }
    assert!(worst <= 1, "the photo differs by {worst} codes");
    assert!(wide.report.runs.output_passes == 1);
}

#[test]
fn a_magnified_view_pans_and_letterboxes_like_the_whole_picture() {
    a_magnified_view_pans_and_letterboxes_like_the_whole_picture_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn a_magnified_view_pans_and_letterboxes_like_the_whole_picture_on_the_device() {
    let Some(gpu) = gpu() else { return };
    a_magnified_view_pans_and_letterboxes_like_the_whole_picture_with(&device(&gpu));
}

fn the_tiles_of_a_magnified_sharpened_view_are_prepared_with_it_with(r: &Renderer) {
    let d = developed(1200, 800);
    let src = d.source();
    let image = r.final_size(&src, &CancelToken::new()).expect("final size");
    let mut st = cell(576.0, 384.0);
    st.set_mode(ViewportMode::Percent { value: 200.0 }, image);
    let coarse_only = FrameOptions {
        coarse_only: true,
        no_cache: false,
    };
    let ask = || {
        r.viewport_frame(&src, &st, &coarse_only, &CancelToken::new())
            .expect("frame")
    };
    let coarse = ask();
    assert!(!coarse.sharp);
    assert_eq!(
        (coarse.report.runs.tiles, coarse.report.runs.full_chains),
        (0, 0)
    );
    let made = r
        .prepare_tiles(&src, &st, &CancelToken::new())
        .expect("tiles");
    assert!(made >= 1);
    let sharp = ask();
    assert!(sharp.sharp, "the tiles the frame needs were prepared");
    assert_eq!(sharp.report.runs.tiles, 0);
    assert_eq!(sharp.report.scale, 1);
}
#[test]
fn the_tiles_of_a_magnified_sharpened_view_are_prepared_with_it() {
    the_tiles_of_a_magnified_sharpened_view_are_prepared_with_it_with(
        &Renderer::neutral_only().expect("render thread"),
    );
}

#[test]
fn the_tiles_of_a_magnified_sharpened_view_are_prepared_with_it_on_the_device() {
    let Some(gpu) = gpu() else { return };
    the_tiles_of_a_magnified_sharpened_view_are_prepared_with_it_with(&device(&gpu));
}
