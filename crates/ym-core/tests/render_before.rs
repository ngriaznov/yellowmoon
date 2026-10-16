//! The Before view shows the photo as the camera would render it by default (`docs/02-colour-pipeline.md`,
//! section The Before view): the raw stages as for the final picture, then the camera's as-shot white
//! balance, the body's D65 colour matrix, the camera gain and the camera tone curve, and nothing else.
//!
//! * it is bit-identical across any edit of the user's settings (light, colour, look, tone, curve, white
//!   lift, detail), on the CPU and on the device;
//! * a scene the camera got right (as-shot white balance of the true light) renders R = G = B, for bodies
//!   with different matrices;
//! * middle grey after the camera gain lands on sRGB code 119.7;
//! * a body without a matrix of its own (not in the table) gets the A7R IV's;
//! * the final picture is not the Before view, and the CPU and the device agree on it.
//!
//! Synthetic scenes through the real decoder, estimates and renderer.
mod common;

use std::sync::Arc;

use common::*;
use ym_color::camera_default::CameraDefault;
use ym_color::encode::srgb_encode;
use ym_color::exposure::CAMERA_GAIN_EV;
use ym_core::pipeline::color::camera_default_of;
use ym_core::pipeline::{
    ComputeDevice, FrameOptions, NeutralOnly, PhotoSource, RenderOptions, RenderedFrame, Renderer,
    ViewportRect, ViewportState,
};
use ym_core::settings::{AutoOr, DevelopSettings};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;

/// A camera white (green = 1) whose as-shot multipliers are exact in the file's 1/1024 units:
/// `[2, 1, 1.25]`.
const WHITE: [f64; 3] = [0.5, 1.0, 0.8];

fn edit(src: &PhotoSource, f: impl FnOnce(&mut DevelopSettings)) -> PhotoSource {
    let mut s = src.settings.clone();
    f(&mut s);
    src.with_settings(s)
}

fn cell() -> ViewportState {
    ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0))
}

fn view(r: &Renderer, src: &PhotoSource, before: bool) -> RenderedFrame {
    let mut st = cell();
    st.before = before;
    r.viewport_frame(src, &st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

fn cpu() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread")
}

fn device() -> Option<Renderer> {
    match Gpu::start(GpuConfig::default()) {
        Ok(g) => Some(
            Renderer::new(
                Arc::new(NeutralOnly),
                RenderOptions {
                    gpu: Some(Arc::new(g)),
                    ..RenderOptions::default()
                },
            )
            .expect("render thread"),
        ),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            eprintln!("* GPU part of render_before skipped: {e}");
            None
        }
    }
}

/// A neutral patchwork shot by a camera that got the light right.
fn patchwork(model: &str) -> PhotoSource {
    let scene = Scene::new(600, 400, WHITE, Layout::Patchwork, 0.4);
    Developed::new(&scene, model).source()
}

/// A flat neutral field of scene luminance `level` shot by a camera that got the light right.
fn flat(model: &str, level: f64) -> PhotoSource {
    let scene = Scene::new(600, 400, WHITE, Layout::Uniform, level);
    Developed::new(&scene, model).source()
}

fn centre(frame: &RenderedFrame) -> [f64; 3] {
    let (w, h) = (frame.width / 4, frame.height / 4);
    mean_rgba(
        frame,
        frame.width / 2 - w / 2,
        frame.height / 2 - h / 2,
        w,
        h,
    )
}

/// Every edit the person can make that is not a decode switch, a lens correction or a crop.
/// One named edit of the settings.
type Edit = (&'static str, Box<dyn Fn(&mut DevelopSettings)>);

fn edits() -> Vec<Edit> {
    vec![
        (
            "exposure",
            Box::new(|s| s.exposure = AutoOr::manual_ev(2.0)),
        ),
        (
            "white balance",
            Box::new(|s| s.white_balance = AutoOr::manual_kelvin(3200.0, 20.0)),
        ),
        ("contrast", Box::new(|s| s.contrast = 60.0)),
        ("brightness", Box::new(|s| s.brightness = -40.0)),
        ("highlights", Box::new(|s| s.highlights = -50.0)),
        ("shadows", Box::new(|s| s.shadows = 50.0)),
        ("black", Box::new(|s| s.black = 30.0)),
        ("white", Box::new(|s| s.white = -30.0)),
        ("white offset", Box::new(|s| s.white_offset_ev = 2.0)),
        ("saturation", Box::new(|s| s.saturation = 80.0)),
        ("look intensity", Box::new(|s| s.look_intensity = 20.0)),
        (
            "curve",
            Box::new(|s| {
                s.curve.luma = vec![[0.0, 0.1], [0.3, 0.2], [0.7, 0.9], [1.0, 1.0]];
                s.curve.red = vec![[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]];
            }),
        ),
        ("roll-off", Box::new(|s| s.rolloff.highlights = 40.0)),
        (
            "noise reduction",
            Box::new(|s| {
                s.noise_reduction = AutoOr::Manual(ym_core::settings::NoiseReductionValue {
                    luminance: 90.0,
                    color: 90.0,
                    detail: 10.0,
                })
            }),
        ),
        (
            "sharpening",
            Box::new(|s| {
                s.sharpen = AutoOr::Manual(ym_core::settings::SharpenValue {
                    amount: 250.0,
                    radius_px: 2.0,
                })
            }),
        ),
        (
            "output sharpening",
            Box::new(|s| s.output_sharpening = 200.0),
        ),
    ]
}

#[test]
fn the_before_view_is_bit_identical_across_every_edit() {
    let src = patchwork("ILCE-7M3");
    let mut renderers = vec![("cpu", cpu())];
    renderers.extend(device().map(|d| ("device", d)));
    for (name, r) in &renderers {
        let reference = view(r, &src, true);
        for (what, f) in edits() {
            let edited = edit(&src, |s| f(s));
            let frame = view(r, &edited, true);
            assert!(
                frame.rgba == reference.rgba,
                "{name}: the Before view changed with {what}"
            );
        }
        // And the final picture does change with them: the test would be vacuous otherwise.
        let fin = view(r, &src, false);
        assert_ne!(
            fin.rgba, reference.rgba,
            "{name}: Before is not the final picture"
        );
        let moved = view(r, &edit(&src, |s| s.contrast = 60.0), false);
        assert_ne!(
            moved.rgba, fin.rgba,
            "{name}: contrast moves the final picture"
        );
    }
}

#[test]
fn an_as_shot_neutral_renders_neutral_for_bodies_with_different_matrices() {
    let r = cpu();
    // An A-mount body, an NEX, an old and a new full-frame body, the A1 (its profile has a white scale, which the
    // Before view must not apply), and one that is not in the table.
    for model in [
        "DSLR-A300",
        "NEX-7",
        "ILCE-7M2",
        "ILCE-7RM5",
        "ILCE-1",
        "ILCE-9999",
    ] {
        let src = flat(model, 0.09);
        let c = centre(&view(&r, &src, true));
        assert!(
            (c[0] - c[1]).abs() <= 1.0 && (c[1] - c[2]).abs() <= 1.0,
            "{model}: {c:?}"
        );
        // And it is not black or white: the scene sits at about +0.3 stop over middle grey.
        assert!(c[1] > 60.0 && c[1] < 200.0, "{model}: {c:?}");
    }
}

#[test]
fn middle_grey_after_the_camera_gain_lands_on_code_119_7() {
    let r = cpu();
    // The scene's neutral pixel is `0.9 L` in green (white 1), the camera gain is 2^1.11: a flat field of
    // `L = 0.18 / (0.9 * 2^1.11)` is middle grey after the gain.
    let level = 0.18 / (0.9 * CAMERA_GAIN_EV.exp2());
    let src = flat("ILCE-7M4", level);
    let c = centre(&view(&r, &src, true));
    for v in c {
        assert!((v - 119.7).abs() <= 0.5, "{c:?}");
    }
    // The tone curve's own value, to be sure the number is not an accident of the scene.
    let want = srgb_encode(0.186_751) * 255.0;
    assert!((want - 119.7).abs() < 0.05);
}

#[test]
fn a_body_without_a_matrix_of_its_own_gets_the_a7r_iv() {
    // ILCE-7M5 and ILCE-7RM6 have no entry (docs: color-matrices.json, `excluded`); a made-up id and
    // string are not in the table either. All use the A7R IV's matrix: the same camera default as an
    // A7R IV file of the same scene.
    let reference = patchwork("ILCE-7RM4");
    let g0 = |src: &PhotoSource| src.estimates.initial_gains().expect("gains");
    let want = camera_default_of(&reference, g0(&reference));
    for model in ["ILCE-7M5", "ILCE-7RM6", "ILCE-1M2", "ILCE-9999"] {
        let src = patchwork(model);
        let got = camera_default_of(&src, g0(&src));
        assert_eq!(got, want, "{model}");
    }
    // A body with a matrix of its own differs.
    let other = patchwork("ILCE-7M3");
    assert_ne!(camera_default_of(&other, g0(&other)), want);
}

#[test]
fn the_camera_default_comes_from_the_file_and_the_body_only() {
    let src = patchwork("ILCE-7M3");
    let g0 = src.estimates.initial_gains().expect("gains");
    let cd = camera_default_of(&src, g0);
    // No setting enters it: the same for every edit.
    for (what, f) in edits() {
        let edited = edit(&src, |s| f(s));
        assert_eq!(camera_default_of(&edited, g0), cd, "{what}");
    }
    // The as-shot white of the file is neutral through it.
    let gs = [2.0, 1.0, 1.25];
    let cam_in = [g0[0] / gs[0], g0[1] / gs[1], g0[2] / gs[2]];
    let lin = cd.scene_linear_rec709(cam_in);
    let gain = CAMERA_GAIN_EV.exp2();
    for c in lin {
        assert!((c - gain).abs() < 1e-9, "{lin:?}");
    }
    // A photo with no as-shot white balance keeps our estimate (the identity balance).
    let mut bare = src.clone();
    let mut raw = (*bare.raw).clone();
    raw.wb_as_shot = None;
    bare.raw = Arc::new(raw);
    let kept = camera_default_of(&bare, g0);
    let table = ym_profiles::color_matrices::ColorMatrices::shipped();
    let m = table.lookup("SONY", "ILCE-7M3", Some(363)).body;
    assert_eq!(
        kept,
        CameraDefault::new(&m.xyz_d65_to_camera, g0, g0).expect("camera default")
    );
}

#[test]
fn the_device_and_the_cpu_agree_on_the_before_view() {
    let Some(on_device) = device() else {
        return;
    };
    let on_cpu = cpu();
    let src = patchwork("ILCE-7M3");
    for target_before in [true, false] {
        let a = view(&on_cpu, &src, target_before);
        let b = view(&on_device, &src, target_before);
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "the colour pass ran on the device"
        );
        assert_eq!((a.width, a.height), (b.width, b.height));
        let mut worst = 0i32;
        let mut over = 0usize;
        for (x, y) in a.rgba.iter().zip(&b.rgba) {
            let d = (i32::from(*x) - i32::from(*y)).abs();
            worst = worst.max(d);
            over += usize::from(d > 1);
        }
        // f32 against f64 moves a value across a dither threshold now and then; never by more than a code
        // or two, and rarely.
        assert!(
            worst <= 2 && over * 1000 <= a.rgba.len(),
            "before = {target_before}: worst {worst}, {over} of {} bytes more than a code off",
            a.rgba.len()
        );
    }
}

#[test]
fn the_pixel_readout_follows_the_before_view() {
    let r = cpu();
    let src = patchwork("ILCE-7M3");
    let mut st = cell();
    st.before = true;
    let frame = view(&r, &src, true);
    for (x, y) in [(75.0, 75.0), (175.0, 125.0), (225.0, 175.0)] {
        let s = r
            .sample_pixel(&src, &st, (x, y), &CancelToken::new())
            .expect("sample")
            .expect("inside");
        let m = mean_rgba(&frame, x as usize - 3, y as usize - 3, 7, 7);
        for (c, v) in [s.r, s.g, s.b].into_iter().enumerate() {
            assert!(
                (f64::from(v) - m[c]).abs() <= 3.0,
                "({x},{y}) channel {c}: {v} vs {:.1}",
                m[c]
            );
        }
        // The readout also ignores the settings.
        let edited = edit(&src, |s| s.contrast = 70.0);
        let t = r
            .sample_pixel(&edited, &st, (x, y), &CancelToken::new())
            .expect("sample")
            .expect("inside");
        assert_eq!((s.r, s.g, s.b), (t.r, t.g, t.b));
    }
}
