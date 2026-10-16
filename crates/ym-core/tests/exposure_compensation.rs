//! The exposure compensation reaches the picture in full (`docs/02-colour-pipeline.md`, Auto exposure, the interim
//! rule of 2026-10-14): the raw of a shot taken with `EC` stops of compensation holds the scene `2^EC` brighter, the
//! exposure `EV_auto` is the camera's constant gain whatever the compensation, so the picture is exactly `2^EC` brighter
//! than the same scene shot without. Synthetic ARW scenes through the real decoder, estimates and renderer.
mod common;

use common::*;
use ym_color::camera_default::CameraTone;
use ym_color::encode::srgb_decode;
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_core::pipeline::{
    FrameOptions, PhotoSource, RenderedFrame, Renderer, ViewportRect, ViewportState,
};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A photo of a flat neutral field of `level`, shot with `ec_tenths` tenths of a stop of compensation.
fn shot(level: f64, ec_tenths: i32) -> Developed {
    let scene = Scene::new(600, 400, light(PROFILE, 5500.0), Layout::Uniform, level);
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("DSC00001.ARW");
    scene
        .builder(MODEL)
        .exposure_comp(ec_tenths, 10)
        .write_to(&path, &scene.mosaic())
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn before_view(r: &Renderer, src: &PhotoSource) -> RenderedFrame {
    let mut state = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    state.before = true;
    r.viewport_frame(src, &state, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

/// The scene-linear value of the exposed signal that the Before view shows, centre green: the camera default
/// renders the exposed signal through the camera tone curve and the sRGB transfer, so the code value is decoded and
/// the curve inverted (it is monotone; bisection).
fn exposed_signal(frame: &RenderedFrame) -> f64 {
    let (w, h) = (frame.width / 4, frame.height / 4);
    let g = mean_rgba(
        frame,
        frame.width / 2 - w / 2,
        frame.height / 2 - h / 2,
        w,
        h,
    )[1];
    let shown = srgb_decode(g / 255.0);
    let tone = CameraTone::FITTED;
    let (mut lo, mut hi) = (0.0_f64, tone.white());
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if tone.apply(mid) < shown {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn a_scene_shot_with_ec_stops_of_compensation_renders_two_to_the_ec_brighter() {
    let renderer = Renderer::neutral_only().expect("render thread");
    let base = shot(0.04, 0);
    let base_est = base.session.view(base.id).expect("view").estimates;
    assert_eq!(base_est.exposure.own_ev, CAMERA_GAIN_EV);
    let base_signal = exposed_signal(&before_view(&renderer, &base.source()));
    for ec in [-1.0_f64, 1.0, 2.0] {
        // The camera that was told to expose `ec` stops more exposed the scene 2^ec brighter: its raw is brighter by
        // that factor and its file says so.
        let bright = shot(0.04 * ec.exp2(), (ec * 10.0) as i32);
        let view = bright.session.view(bright.id).expect("view");
        assert!(
            (view.meta.exposure_comp_ev.expect("EC") - ec).abs() < 1e-9,
            "the file carries the compensation"
        );
        // Exactly the same exposure: no metering, and no compensation term.
        assert_eq!(view.estimates.exposure.own_ev, CAMERA_GAIN_EV, "EC {ec}");
        assert_eq!(view.estimates.exposure.own_ev, base_est.exposure.own_ev);
        // The reported key statistic follows the raw by 2^EC (noise-free flat field: 0.2 %).
        let key =
            view.estimates.exposure.meter.mean_luminance / base_est.exposure.meter.mean_luminance;
        assert!(
            (key / ec.exp2() - 1.0).abs() < 2e-3,
            "EC {ec}: the raw is {key} times brighter"
        );
        // And the picture: the exposed signal of the Before view (8-bit codes through the camera tone curve, a step of
        // 0.03 to 0.05 stop) is 2^EC brighter, the whole of the compensation.
        let signal = exposed_signal(&before_view(&renderer, &bright.source()));
        let stops = (signal / base_signal).log2();
        assert!(
            (stops - ec).abs() < 0.08,
            "EC {ec}: the picture is {stops} stops brighter"
        );
    }
}
