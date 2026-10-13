//! A photo whose noise model cannot be used still renders (`docs/algorithms/INTEGRATION.md`
//! section 10).
//!
//! The adaptive ratio demosaic is the default of R6 and refuses a noise model with a negative or
//! non-finite number, or one whose variance overflows at white. The render never does: the photo
//! is processed as if its noise were unknown (the zero model), and the other stages that read the
//! model, the sigma map above all, read the same zero model instead of a `NaN` that would blacken
//! the picture. These tests inject such a model into the estimates of a developed photo and render
//! it through the paths a user reaches: the full-resolution render (an export) and the 100 % view.
mod common;

use std::sync::Arc;

use common::*;
use ym_color::display::OutputTarget;
use ym_core::pipeline::{
    FrameOptions, FullRequest, FullTarget, PhotoSource, Renderer, ViewportMode, ViewportRect,
    ViewportState,
};
use ym_gpu::queue::CancelToken;
use ym_noise::model::NoiseModel;

const MODEL: &str = "ILCE-7RM4";

/// Noise models that cannot be used, and whose every number the raw group reads: not a number,
/// infinite, a negative gain, a negative variance, a black-level offset that is not a number.
fn unusable_models() -> Vec<(&'static str, NoiseModel)> {
    vec![
        ("NaN gain", NoiseModel::new(f64::NAN, 3e-6)),
        ("infinite gain", NoiseModel::new(f64::INFINITY, 3e-6)),
        ("negative gain", NoiseModel::new(-4e-4, 3e-6)),
        ("negative variance", NoiseModel::new(4e-4, -1e-5)),
        ("NaN offset", {
            let mut m = NoiseModel::new(4e-4, 3e-6);
            m.delta[2] = f64::NAN;
            m
        }),
    ]
}

fn photo() -> Developed {
    let white = light("sony-ilce-7rm4", 4800.0);
    let scene = Scene::new(300, 200, white, Layout::Patchwork, 1.0)
        .noise(4e-4, 3e-6)
        .seed(3);
    Developed::new(&scene, MODEL)
}

/// `base` with the noise model of its estimates replaced.
fn with_model(base: &PhotoSource, model: NoiseModel) -> PhotoSource {
    let mut noise = base.estimates.noise.clone();
    noise.model = model;
    let mut src = base.clone();
    src.estimates = Arc::new(base.estimates.with_noise(noise));
    src
}

fn mean_f32(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64
}

#[test]
fn an_unusable_noise_model_gives_a_picture_of_the_photo_at_full_resolution() {
    let d = photo();
    let base = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let request = FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    };
    let reference = r
        .render_full(&base, &request, &CancelToken::new())
        .expect("a photo with its own noise model renders");
    assert!(reference.rgb.iter().all(|v| v.is_finite()));
    let want = mean_f32(&reference.rgb);
    assert!(want > 0.01, "the reference is not a blank: {want}");
    for (name, model) in unusable_models() {
        let image = r
            .render_full(&with_model(&base, model), &request, &CancelToken::new())
            .unwrap_or_else(|e| panic!("{name}: the render failed: {e}"));
        assert_eq!(
            (image.width, image.height),
            (reference.width, reference.height),
            "{name}"
        );
        assert!(
            image.rgb.iter().all(|v| v.is_finite()),
            "{name}: a non-finite sample"
        );
        // A picture of the photo, not a black frame from a NaN sigma map: the mean is the usual
        // one to within a few percent (the noise model only moves the denoise and the toe).
        let got = mean_f32(&image.rgb);
        assert!(
            (got / want - 1.0).abs() < 0.05,
            "{name}: mean {got} against {want}"
        );
    }
}

/// A model whose numbers are finite but whose variance overflows at white (`a = b = 1e300`) is
/// unusable too: the render does not fail and has no non-finite sample.
///
/// Not asserted: that the picture is the photo. The colour plan (`pipeline/color.rs`, C7's toe)
/// reads the read-noise variance `b` of the estimates directly and floors the toe at its square
/// root, so a `b` of 1e300 still blackens the frame there. The estimator never makes such a
/// model.
#[test]
fn a_noise_model_that_overflows_at_white_does_not_fail_the_render() {
    let d = photo();
    let base = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let request = FullRequest {
        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
    };
    let image = r
        .render_full(
            &with_model(&base, NoiseModel::new(1e300, 1e300)),
            &request,
            &CancelToken::new(),
        )
        .expect("the render does not fail");
    assert!(image.rgb.iter().all(|v| v.is_finite()));
}

#[test]
fn an_unusable_noise_model_gives_a_picture_of_the_photo_in_the_100_percent_view() {
    let d = photo();
    let base = d.source();
    let r = Renderer::neutral_only().expect("render thread");
    let cancel = CancelToken::new();
    let image = r.final_size(&base, &cancel).expect("size");
    let mut view = ViewportState::new(ViewportRect::cell(200.0, 150.0, 1.0));
    view.set_mode(ViewportMode::Percent { value: 100.0 }, image);
    let frame = |src: &PhotoSource| {
        r.viewport_frame(src, &view, &FrameOptions::default(), &cancel)
            .map(|f| f.rgba)
    };
    let mean_u8 = |rgba: &[u8]| {
        rgba.chunks_exact(4)
            .flat_map(|p| p[..3].iter().map(|v| f64::from(*v)))
            .sum::<f64>()
            / (rgba.len() / 4 * 3) as f64
    };
    let reference = frame(&base).expect("a photo with its own noise model renders");
    let want = mean_u8(&reference);
    assert!(want > 5.0, "the reference is not a blank: {want}");
    for (name, model) in unusable_models() {
        let got = frame(&with_model(&base, model))
            .unwrap_or_else(|e| panic!("{name}: the 100 % view failed: {e}"));
        assert_eq!(got.len(), reference.len(), "{name}");
        let mean = mean_u8(&got);
        assert!(
            (mean / want - 1.0).abs() < 0.05,
            "{name}: mean {mean} against {want}"
        );
    }
}
