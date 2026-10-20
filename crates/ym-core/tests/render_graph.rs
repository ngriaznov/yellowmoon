//! The render graph's cache rules (`docs/01` section Render graph): a settings change invalidates
//! only the groups after the first one it affects. Each test reads the stage runs of one request.
//!
//! The tests that read the run counts of one request run twice: on the CPU (the reference) and with
//! the view stages on the device (`*_on_the_device`: the sampler, the colour chain, output
//! sharpening and C13 on Mesa lavapipe; the raw chain stays on the CPU). The device road counts what
//! the CPU road counts, stage for stage.
mod common;

use std::io::Write as _;
use std::sync::Arc;

use common::*;
use ym_core::app::gpu::GpuStages;
use ym_core::pipeline::{
    FrameOptions, NeutralOnly, PhotoSource, RenderOptions, Renderer, StageRuns, ViewportMode,
    ViewportRect, ViewportState,
};
use ym_core::scheduler::MemoryBudget;
use ym_core::settings::{Crop, CropAspect, DevelopSettings, NormRect, Orientation};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

/// The device, in poison mode, or `None` (loudly) when no adapter opens, unless `YM_REQUIRE_GPU=1`
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
                "\n* GPU TESTS SKIPPED in render_graph: {e}\n"
            );
            None
        }
    }
}

/// A 1200 x 800 patchwork photo, developed, with a renderer that has only the Neutral look.
struct Rig {
    developed: Developed,
    renderer: Renderer,
    state: ViewportState,
}

impl Rig {
    fn new() -> Self {
        Self::with_renderer(Renderer::neutral_only().expect("render thread"))
    }

    /// The same rig with the view stages on the device.
    fn on_device() -> Option<Self> {
        let gpu = gpu()?;
        Some(Self::with_renderer(
            Renderer::new(
                Arc::new(NeutralOnly),
                RenderOptions {
                    gpu: Some(gpu),
                    gpu_stages: GpuStages::SAMPLER
                        .union(GpuStages::COLOR)
                        .union(GpuStages::OUTPUT_SHARPEN)
                        .union(GpuStages::ENCODE),
                    ..RenderOptions::default()
                },
            )
            .expect("render thread"),
        ))
    }

    fn with_renderer(renderer: Renderer) -> Self {
        let scene = Scene::new(
            1200,
            800,
            light("sony-ilce-7rm4", 5500.0),
            Layout::Patchwork,
            0.4,
        )
        .noise(2e-4, 1e-6);
        Self {
            developed: Developed::new(&scene, "ILCE-7RM4"),
            renderer,
            // A little under half the photo: the fit zoom must not sit on the 50 % boundary between the
            // half-size and the full-size image, or switching the lens corrections off (the photo
            // loses its auto-cropped border, a pixel or two) tips the view over to full-size tiles.
            state: ViewportState::new(ViewportRect::cell(576.0, 384.0, 1.0)),
        }
    }

    fn source(&self) -> PhotoSource {
        self.developed.source()
    }

    /// What one request did.
    fn runs(&self, src: &PhotoSource) -> StageRuns {
        self.runs_with(src, &self.state, FrameOptions::default())
    }

    fn runs_with(&self, src: &PhotoSource, state: &ViewportState, o: FrameOptions) -> StageRuns {
        self.renderer
            .viewport_frame(src, state, &o, &CancelToken::new())
            .expect("frame")
            .report
            .runs
    }
}

/// One edit of the settings, with its name.
type Edit = Box<dyn Fn(&mut DevelopSettings)>;

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn none() -> StageRuns {
    StageRuns::default()
}

fn the_first_frame_runs_every_group_once_and_the_second_is_a_cache_hit_with(rig: Rig) {
    let src = rig.source();
    let first = rig.runs(&src);
    assert_eq!(
        first,
        StageRuns {
            frame_data: 1,
            raw_groups: 1,
            sigma_planes: 1,
            color_plans: 1,
            grade_plans: 1,
            color_passes: 1,
            // Output sharpening (default 100 %) is the last stage of every finished frame.
            output_passes: 1,
            ..none()
        }
    );
    // The same request again: the finished frame is served from the cache.
    assert_eq!(
        rig.runs(&src),
        StageRuns {
            frame_hits: 1,
            ..none()
        }
    );
    // Without the frame cache only the per-pixel pass and the output stage run again.
    let again = rig.runs_with(
        &src,
        &rig.state,
        FrameOptions {
            no_cache: true,
            ..Default::default()
        },
    );
    assert_eq!(
        again,
        StageRuns {
            color_passes: 1,
            output_passes: 1,
            ..none()
        }
    );
    // The totals of the graph add up.
    let total = rig.renderer.counters(src.id);
    assert_eq!(
        (total.raw_groups, total.color_plans, total.grade_plans),
        (1, 1, 1)
    );
}

#[test]
fn the_first_frame_runs_every_group_once_and_the_second_is_a_cache_hit() {
    the_first_frame_runs_every_group_once_and_the_second_is_a_cache_hit_with(Rig::new());
}

#[test]
fn the_first_frame_runs_every_group_once_and_the_second_is_a_cache_hit_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    the_first_frame_runs_every_group_once_and_the_second_is_a_cache_hit_with(rig);
}

fn a_curve_change_does_not_rerun_demosaic_or_the_color_group_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let curved = edit(&src, |s| {
        s.curve.luma = vec![[0.0, 0.0], [0.5, 0.62], [1.0, 1.0]];
    });
    let r = rig.runs(&curved);
    assert_eq!(r.raw_work(), 0, "no raw stage ran: {r:?}");
    assert_eq!(r.sigma_planes, 0);
    assert_eq!(
        r.color_plans, 0,
        "the colour group is untouched by the curve"
    );
    assert_eq!((r.grade_plans, r.color_passes), (1, 1));
}

#[test]
fn a_curve_change_does_not_rerun_demosaic_or_the_color_group() {
    a_curve_change_does_not_rerun_demosaic_or_the_color_group_with(Rig::new());
}

#[test]
fn a_curve_change_does_not_rerun_demosaic_or_the_color_group_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_curve_change_does_not_rerun_demosaic_or_the_color_group_with(rig);
}

fn every_grade_control_and_the_look_only_rebuild_the_grade_group_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let edits: Vec<(&str, Edit)> = vec![
        (
            "contrast",
            Box::new(|s| s.contrast = ym_core::settings::PercentSetting::manual_value(20.0)),
        ),
        ("brightness", Box::new(|s| s.brightness = -20.0)),
        (
            "highlights",
            Box::new(|s| s.highlights = ym_core::settings::PercentSetting::manual_value(30.0)),
        ),
        (
            "shadows",
            Box::new(|s| s.shadows = ym_core::settings::PercentSetting::manual_value(30.0)),
        ),
        ("white", Box::new(|s| s.white = 10.0)),
        (
            "black",
            Box::new(|s| s.black = ym_core::settings::PercentSetting::manual_value(10.0)),
        ),
        (
            "saturation",
            Box::new(|s| s.saturation = ym_core::settings::PercentSetting::manual_value(25.0)),
        ),
        (
            "color_balance",
            Box::new(|s| {
                s.color_balance.shadows =
                    ym_core::settings::BalanceSetting::manual_point([0.3, -0.2])
            }),
        ),
        (
            "color_warp",
            Box::new(|s| {
                s.color_warp.pins =
                    vec![ym_color::warp::WarpPin::moving([0.1, 0.02], [0.06, 0.08]).into()];
            }),
        ),
        ("look_intensity", Box::new(|s| s.look_intensity = 40.0)),
        (
            "curve red",
            Box::new(|s| s.curve.red = vec![[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]]),
        ),
    ];
    for (name, f) in edits {
        let r = rig.runs(&edit(&src, f));
        assert_eq!(r.raw_work(), 0, "{name}: {r:?}");
        assert_eq!(r.color_plans, 0, "{name}: {r:?}");
        assert_eq!(r.grade_plans, 1, "{name}: {r:?}");
    }
}

#[test]
fn every_grade_control_and_the_look_only_rebuild_the_grade_group() {
    every_grade_control_and_the_look_only_rebuild_the_grade_group_with(Rig::new());
}

#[test]
fn every_grade_control_and_the_look_only_rebuild_the_grade_group_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    every_grade_control_and_the_look_only_rebuild_the_grade_group_with(rig);
}

fn a_white_balance_change_does_not_rerun_r3_to_r9_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let edits: Vec<(&str, Edit)> = vec![
        (
            "kelvin",
            Box::new(|s| {
                s.white_balance =
                    ym_core::settings::AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
                        kelvin: 3200.0,
                        tint: 10.0,
                    })
            }),
        ),
        (
            "exposure",
            Box::new(|s| {
                s.exposure =
                    ym_core::settings::AutoOr::Manual(ym_core::settings::ExposureValue { ev: 1.25 })
            }),
        ),
        (
            "light source",
            Box::new(|s| s.light_source = ym_color::wb::LightSource::Tungsten),
        ),
        ("highlight range", Box::new(|s| s.rolloff.highlights = 0.9)),
        ("low-light toe", Box::new(|s| s.rolloff.low_light = 0.1)),
    ];
    for (name, f) in edits {
        let r = rig.runs(&edit(&src, f));
        assert_eq!(r.raw_work(), 0, "{name}: {r:?}");
        assert_eq!(r.sigma_planes, 0, "{name}");
        assert_eq!(r.color_plans, 1, "{name}: {r:?}");
        assert_eq!(r.grade_plans, 1, "{name}: {r:?}");
        assert_eq!(r.color_passes, 1, "{name}");
    }
}

#[test]
fn a_white_balance_change_does_not_rerun_r3_to_r9() {
    a_white_balance_change_does_not_rerun_r3_to_r9_with(Rig::new());
}

#[test]
fn a_white_balance_change_does_not_rerun_r3_to_r9_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_white_balance_change_does_not_rerun_r3_to_r9_with(rig);
}

fn a_decode_switch_noise_reduction_or_sharpening_reruns_the_raw_group_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let edits: Vec<(&str, Edit)> = vec![
        (
            "highlight recovery",
            Box::new(|s| s.decode.highlight_recovery = false),
        ),
        (
            "lens corrections",
            Box::new(|s| s.decode.lens_corrections = false),
        ),
        (
            "chromatic aberration",
            Box::new(|s| s.decode.chromatic_aberration = false),
        ),
        (
            "neural demosaic",
            Box::new(|s| s.decode.neural_demosaic = false),
        ),
        (
            "noise reduction",
            Box::new(|s| {
                s.noise_reduction =
                    ym_core::settings::AutoOr::Manual(ym_core::settings::NoiseReductionValue {
                        luminance: 70.0,
                        color: 20.0,
                        detail: 10.0,
                    })
            }),
        ),
        (
            "sharpen",
            Box::new(|s| {
                s.sharpen = ym_core::settings::AutoOr::Manual(ym_core::settings::SharpenValue {
                    amount: 200.0,
                    radius_px: 1.2,
                })
            }),
        ),
    ];
    for (name, f) in edits {
        let r = rig.runs(&edit(&src, f));
        assert_eq!(r.raw_groups, 1, "{name} must re-run the raw group: {r:?}");
        assert_eq!(
            r.color_plans, 0,
            "{name}: the colour group does not depend on it"
        );
        assert!(
            r.frame_data <= 1,
            "{name}: the area is copied at most once per request"
        );
    }
    // Switching a switch back finds the earlier group still cached.
    let r = rig.runs(&src);
    assert_eq!(r.raw_work(), 0, "the original group is still there: {r:?}");
}

#[test]
fn a_decode_switch_noise_reduction_or_sharpening_reruns_the_raw_group() {
    a_decode_switch_noise_reduction_or_sharpening_reruns_the_raw_group_with(Rig::new());
}

#[test]
fn a_decode_switch_noise_reduction_or_sharpening_reruns_the_raw_group_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_decode_switch_noise_reduction_or_sharpening_reruns_the_raw_group_with(rig);
}

fn crop_straighten_and_orientation_re_run_nothing_but_the_pixel_pass_with(rig: Rig) {
    let src = rig.source();
    // A cell of 330 x 220 (Fit is 27.5 %) at a fixed 30 % zoom: every edit below stays in the
    // 1/2-resolution bucket (above 25 %, up to 50 %), so the same raw group serves them all.
    let mut state = ViewportState::new(ViewportRect::cell(330.0, 220.0, 1.0));
    state.set_mode(ViewportMode::Percent { value: 30.0 }, (1200, 800));
    rig.runs_with(&src, &state, FrameOptions::default());
    let edits: Vec<(&str, Edit)> = vec![
        (
            "orientation",
            Box::new(|s| s.orientation = Orientation::Deg90),
        ),
        (
            "crop",
            Box::new(|s| {
                s.crop = Some(Crop {
                    rect: NormRect {
                        x: 0.1,
                        y: 0.1,
                        width: 0.6,
                        height: 0.7,
                    },
                    angle_deg: 0.0,
                    aspect: CropAspect::Free,
                })
            }),
        ),
        (
            "straighten",
            Box::new(|s| {
                s.crop = Some(Crop {
                    rect: NormRect::FULL,
                    angle_deg: 3.5,
                    aspect: CropAspect::Original,
                })
            }),
        ),
    ];
    for (name, f) in edits {
        let r = rig.runs_with(&edit(&src, f), &state, FrameOptions::default());
        assert_eq!(
            r,
            StageRuns {
                color_passes: 1,
                output_passes: 1,
                ..none()
            },
            "{name}: only the sampling, the colour pass and the output stage run"
        );
    }
}

#[test]
fn crop_straighten_and_orientation_re_run_nothing_but_the_pixel_pass() {
    crop_straighten_and_orientation_re_run_nothing_but_the_pixel_pass_with(Rig::new());
}

#[test]
fn crop_straighten_and_orientation_re_run_nothing_but_the_pixel_pass_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    crop_straighten_and_orientation_re_run_nothing_but_the_pixel_pass_with(rig);
}

fn a_crop_that_needs_a_finer_image_runs_the_raw_group_once_at_the_new_scale_with(rig: Rig) {
    let src = rig.source();
    // Fit in a 600 x 400 cell is 50 %: the half-resolution group. Cropping to a third of the
    // photo shows 3x the pixel density, so the full-resolution tiles are needed: that is the
    // zoom changing, not the crop invalidating a group.
    assert_eq!(rig.runs(&src).raw_groups, 1);
    let cropped = edit(&src, |s| {
        s.crop = Some(Crop {
            rect: NormRect {
                x: 0.0,
                y: 0.0,
                width: 0.4,
                height: 0.4,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
        })
    });
    let r = rig.runs(&cropped);
    assert_eq!((r.raw_groups, r.sigma_planes), (0, 0), "{r:?}");
    assert!(r.full_chains == 1 && r.tiles >= 1, "{r:?}");
    // The same crop again is a frame-cache hit; going back to the whole photo finds the group.
    assert_eq!(
        rig.runs(&cropped),
        StageRuns {
            frame_hits: 1,
            ..none()
        }
    );
    assert_eq!(
        rig.runs(&src),
        StageRuns {
            frame_hits: 1,
            ..none()
        }
    );
}

#[test]
fn a_crop_that_needs_a_finer_image_runs_the_raw_group_once_at_the_new_scale() {
    a_crop_that_needs_a_finer_image_runs_the_raw_group_once_at_the_new_scale_with(Rig::new());
}

#[test]
fn a_crop_that_needs_a_finer_image_runs_the_raw_group_once_at_the_new_scale_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_crop_that_needs_a_finer_image_runs_the_raw_group_once_at_the_new_scale_with(rig);
}

fn a_shared_exposure_from_the_burst_moves_only_the_color_group_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let moved = {
        let mut s = src.clone();
        s.estimates = std::sync::Arc::new(
            src.estimates
                .with_shared_ev(src.estimates.exposure.ev + 0.4),
        );
        s
    };
    let r = rig.runs(&moved);
    assert_eq!(r.raw_work(), 0, "{r:?}");
    assert_eq!((r.color_plans, r.grade_plans), (1, 1), "{r:?}");
}

#[test]
fn a_shared_exposure_from_the_burst_moves_only_the_color_group() {
    a_shared_exposure_from_the_burst_moves_only_the_color_group_with(Rig::new());
}

#[test]
fn a_shared_exposure_from_the_burst_moves_only_the_color_group_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_shared_exposure_from_the_burst_moves_only_the_color_group_with(rig);
}

fn a_pooled_noise_model_reruns_the_raw_group_but_not_the_decoded_area_with(rig: Rig) {
    let src = rig.source();
    rig.runs(&src);
    let mut noise = src.estimates.noise.clone();
    noise.model.a *= 1.5;
    let moved = {
        let mut s = src.clone();
        s.estimates = std::sync::Arc::new(src.estimates.with_noise(noise));
        s
    };
    let r = rig.runs(&moved);
    assert_eq!(
        (r.sigma_planes, r.raw_groups, r.frame_data),
        (1, 1, 1),
        "{r:?}"
    );
}

#[test]
fn a_pooled_noise_model_reruns_the_raw_group_but_not_the_decoded_area() {
    a_pooled_noise_model_reruns_the_raw_group_but_not_the_decoded_area_with(Rig::new());
}

#[test]
fn a_pooled_noise_model_reruns_the_raw_group_but_not_the_decoded_area_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_pooled_noise_model_reruns_the_raw_group_but_not_the_decoded_area_with(rig);
}

fn at_100_percent_tiles_are_cached_across_pans_and_color_edits_with(rig: Rig) {
    let src = rig.source();
    let mut state = rig.state.clone();
    let image = rig
        .renderer
        .final_size(&src, &CancelToken::new())
        .expect("size");
    assert_eq!(image, (1200, 800), "no lens tags: the whole frame");
    state.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let first = rig.runs_with(&src, &state, FrameOptions::default());
    // A 600 x 400 cell of 512 px tiles over a 1200 x 800 photo, centred: 2 x 2 tiles (or 3 x 2).
    assert!(first.tiles >= 4, "{first:?}");
    assert_eq!((first.full_chains, first.raw_groups), (1, 0), "{first:?}");
    // A colour-only edit finds every tile, and the full chain, cached.
    let edited = edit(&src, |s| {
        s.contrast = ym_core::settings::PercentSetting::manual_value(15.0)
    });
    let r = rig.runs_with(&edited, &state, FrameOptions::default());
    assert_eq!((r.tiles, r.full_chains, r.raw_work()), (0, 0, 0), "{r:?}");
    // A small pan inside the same tiles computes nothing new either.
    let mut panned = state.clone();
    panned.centre = [0.52, 0.5];
    let r = rig.runs_with(&src, &panned, FrameOptions::default());
    assert_eq!((r.tiles, r.full_chains), (0, 0), "{r:?}");
    // A raw-group change drops the tiles: they are recomputed.
    let r = rig.runs_with(
        &edit(&src, |s| s.decode.highlight_recovery = false),
        &state,
        FrameOptions::default(),
    );
    assert!(r.tiles >= 4 && r.full_chains == 1, "{r:?}");
}

#[test]
fn at_100_percent_tiles_are_cached_across_pans_and_color_edits() {
    at_100_percent_tiles_are_cached_across_pans_and_color_edits_with(Rig::new());
}

#[test]
fn at_100_percent_tiles_are_cached_across_pans_and_color_edits_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    at_100_percent_tiles_are_cached_across_pans_and_color_edits_with(rig);
}

fn a_coarse_frame_never_computes_tiles_and_says_it_is_not_sharp_with(rig: Rig) {
    let src = rig.source();
    let mut state = rig.state.clone();
    let image = rig
        .renderer
        .final_size(&src, &CancelToken::new())
        .expect("size");
    state.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let coarse = rig
        .renderer
        .viewport_frame(
            &src,
            &state,
            &FrameOptions {
                coarse_only: true,
                no_cache: false,
            },
            &CancelToken::new(),
        )
        .expect("coarse frame");
    assert!(!coarse.sharp);
    assert_eq!(
        (coarse.report.runs.tiles, coarse.report.runs.full_chains),
        (0, 0)
    );
    assert_eq!(
        coarse.report.scale, 2,
        "served from the half-resolution image"
    );
    // The tiles are prepared as their own job; the next frame is sharp and finds them.
    let made = rig
        .renderer
        .prepare_tiles(&src, &state, &CancelToken::new())
        .expect("tiles");
    assert!(made >= 4);
    let sharp = rig
        .renderer
        .viewport_frame(
            &src,
            &state,
            &FrameOptions {
                coarse_only: true,
                no_cache: false,
            },
            &CancelToken::new(),
        )
        .expect("sharp frame");
    assert!(sharp.sharp);
    assert_eq!((sharp.report.runs.tiles, sharp.report.scale), (0, 1));
}

#[test]
fn a_coarse_frame_never_computes_tiles_and_says_it_is_not_sharp() {
    a_coarse_frame_never_computes_tiles_and_says_it_is_not_sharp_with(Rig::new());
}

#[test]
fn a_coarse_frame_never_computes_tiles_and_says_it_is_not_sharp_on_the_device() {
    let Some(rig) = Rig::on_device() else { return };
    a_coarse_frame_never_computes_tiles_and_says_it_is_not_sharp_with(rig);
}

#[test]
fn the_renderer_keeps_the_graphs_of_the_most_recent_photos_only() {
    let scene = Scene::new(
        400,
        300,
        light("sony-ilce-7rm4", 5500.0),
        Layout::Uniform,
        0.3,
    );
    let dir = tempfile::tempdir().expect("dir");
    for i in 1..=3 {
        scene.write(dir.path(), &format!("DSC0000{i}.ARW"), "ILCE-7RM4");
    }
    let session = ym_core::session::Session::new(base_env());
    let out = session.open(
        &[dir.path().to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    settle(&session);
    let renderer = Renderer::new(
        std::sync::Arc::new(ym_core::pipeline::NeutralOnly),
        RenderOptions {
            budget: MemoryBudget {
                max_photos: 2,
                ..MemoryBudget::default()
            },
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let state = ViewportState::new(ViewportRect::cell(200.0, 150.0, 1.0));
    let mut sources = Vec::new();
    for id in &out.added {
        let view = session.view(*id).expect("view");
        let raw = session.raw_image(*id).expect("raw");
        sources.push(PhotoSource::from_view(&view, raw));
    }
    for s in &sources {
        renderer
            .viewport_frame(s, &state, &FrameOptions::default(), &CancelToken::new())
            .expect("frame");
    }
    assert_eq!(
        renderer.counters(sources[0].id),
        StageRuns::default(),
        "the oldest graph is gone"
    );
    assert!(renderer.counters(sources[2].id).raw_groups >= 1);
    assert!(renderer.cached_bytes() > 0);
    // Going back to the first photo rebuilds its graph.
    let r = renderer
        .viewport_frame(
            &sources[0],
            &state,
            &FrameOptions::default(),
            &CancelToken::new(),
        )
        .expect("frame")
        .report
        .runs;
    assert_eq!(r.raw_groups, 1);
}

fn leaving_a_photo_frees_its_full_resolution_intermediates_and_keeps_the_small_images_with(
    rig: Rig,
) {
    let src = rig.source();
    rig.renderer.navigate(src.id);
    // The fit image (1/2 resolution) and the 100 % tiles with their prepared chain.
    rig.runs(&src);
    let mut state = rig.state.clone();
    state.set_mode(ViewportMode::Percent { value: 100.0 }, (1200, 800));
    rig.runs_with(&src, &state, FrameOptions::default());
    let with_tiles = rig.renderer.graph(src.id).bytes();
    // Moving to another photo (any id) releases the tiles and the sigma map of the chain.
    rig.renderer
        .navigate(ym_core::session::PhotoId(src.id.0 + 100));
    let after = rig.renderer.graph(src.id).bytes();
    assert!(after < with_tiles / 2, "{after} bytes left of {with_tiles}");
    // The small image is still there: the fit view needs no raw work; the 100 % view needs the
    // chain and its tiles again.
    assert_eq!(rig.runs(&src).raw_work(), 0);
    let r = rig.runs_with(
        &src,
        &state,
        FrameOptions {
            no_cache: true,
            ..Default::default()
        },
    );
    assert!(r.full_chains == 1 && r.tiles >= 4, "{r:?}");
}

#[test]
fn leaving_a_photo_frees_its_full_resolution_intermediates_and_keeps_the_small_images() {
    leaving_a_photo_frees_its_full_resolution_intermediates_and_keeps_the_small_images_with(
        Rig::new(),
    );
}

#[test]
fn leaving_a_photo_frees_its_full_resolution_intermediates_and_keeps_the_small_images_on_the_device()
 {
    let Some(rig) = Rig::on_device() else { return };
    leaving_a_photo_frees_its_full_resolution_intermediates_and_keeps_the_small_images_with(rig);
}

#[test]
fn a_photo_source_comes_straight_from_the_session() {
    let rig = Rig::new();
    let a = PhotoSource::from_session(&rig.developed.session, rig.developed.id).expect("source");
    assert_eq!(a.id, rig.developed.id);
    assert_eq!(a.settings, rig.source().settings);
    assert!(
        PhotoSource::from_session(&rig.developed.session, ym_core::session::PhotoId(999)).is_err()
    );
}
