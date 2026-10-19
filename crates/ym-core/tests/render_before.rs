//! The Before view shows the photo as the camera would render it by default (`docs/02-colour-pipeline.md`,
//! section The Before view): the raw stages as for the final picture, then the camera's as-shot white
//! balance, the body's D65 colour matrix, the camera gain and the camera tone curve, and nothing else.
//!
//! * it is bit-identical across any edit of the user's settings (light, colour, look, tone, curve, the
//!   highlight range, white lift, detail), on the CPU and on the device;
//! * a scene the camera got right (as-shot white balance of the true light) renders R = G = B, for bodies
//!   with different matrices;
//! * middle grey after the camera gain lands on sRGB code 119.7;
//! * a body without a matrix of its own (not in the table) gets the A7R IV's;
//! * the final picture is not the Before view, and the CPU and the device agree on it;
//! * the Before view leaves out the lens vignetting correction (R3), as the camera's own picture does
//!   (owner's decision 2026-10-16): at the frame corners it is not brightened, the raw front it needs
//!   is its own (never mixed with the final picture's caches), and the CPU and the device agree on it;
//! * the Detail settings do not reach it (same decision round).
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
    ViewportMode, ViewportRect, ViewportState,
};
use ym_core::settings::{AutoOr, DevelopSettings};
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;
use ym_raw::testgen::LensTables;

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
        (
            "contrast",
            Box::new(|s| s.contrast = ym_core::settings::PercentSetting::manual_value(60.0)),
        ),
        ("brightness", Box::new(|s| s.brightness = -40.0)),
        (
            "highlights",
            Box::new(|s| s.highlights = ym_core::settings::PercentSetting::manual_value(-50.0)),
        ),
        (
            "shadows",
            Box::new(|s| s.shadows = ym_core::settings::PercentSetting::manual_value(50.0)),
        ),
        (
            "black",
            Box::new(|s| s.black = ym_core::settings::PercentSetting::manual_value(30.0)),
        ),
        ("white", Box::new(|s| s.white = -30.0)),
        ("white offset", Box::new(|s| s.white_offset_ev = 2.0)),
        (
            "saturation",
            Box::new(|s| s.saturation = ym_core::settings::PercentSetting::manual_value(80.0)),
        ),
        ("look intensity", Box::new(|s| s.look_intensity = 20.0)),
        (
            "curve",
            Box::new(|s| {
                s.curve.luma = vec![[0.0, 0.1], [0.3, 0.2], [0.7, 0.9], [1.0, 1.0]];
                s.curve.red = vec![[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]];
            }),
        ),
        ("highlight range", Box::new(|s| s.rolloff.highlights = 0.0)),
        ("low light", Box::new(|s| s.rolloff.low_light = 1.0)),
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
        let moved = view(
            r,
            &edit(&src, |s| {
                s.contrast = ym_core::settings::PercentSetting::manual_value(60.0)
            }),
            false,
        );
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
    for model in ["ILCE-7M5", "ILCE-7RM6", "ILCE-9999"] {
        let src = patchwork(model);
        let got = camera_default_of(&src, g0(&src));
        assert_eq!(got, want, "{model}");
    }
    // A body with a matrix of its own differs.
    let other = patchwork("ILCE-7M3");
    assert_ne!(camera_default_of(&other, g0(&other)), want);
}

/// A patchwork shot by `model` with the Sony model ID `id` written in the maker note.
fn patchwork_with_id(model: &str, id: u16) -> PhotoSource {
    let scene = Scene::new(600, 400, WHITE, Layout::Patchwork, 0.4);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder(model)
        .model_id(id)
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    Developed::open(dir).source()
}

#[test]
fn the_a1_ii_uses_the_a1s_colours_and_the_a3000_the_generic_e_mount_ones() {
    // Owner's decision 2026-10-16 (`docs/STATUS.md`, question 29). The camera default of a file depends on
    // its as-shot white and the body's matrix, so equal scenes on bodies that share a matrix give equal
    // camera defaults, and a body with a matrix of its own gives another.
    let g0 = |src: &PhotoSource| src.estimates.initial_gains().expect("gains");
    let default_of = |src: &PhotoSource| camera_default_of(src, g0(src));
    let a1 = default_of(&patchwork("ILCE-1"));
    let a7r4 = default_of(&patchwork("ILCE-7RM4"));
    // The A1 II, by its ID (400, written by `patchwork`) and by its Model string alone.
    assert_eq!(default_of(&patchwork("ILCE-1M2")), a1);
    assert_ne!(default_of(&patchwork("ILCE-1M2")), a7r4);
    let by_string = {
        let scene = Scene::new(600, 400, WHITE, Layout::Patchwork, 0.4);
        let dir = tempfile::tempdir().expect("dir");
        scene
            .builder("ILCE-1M2")
            .without_model_id()
            .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        Developed::open(dir).source()
    };
    assert_eq!(default_of(&by_string), a1);
    // The A3000 (ID 302, and the Model string alone) and the A3500 (the same ID): the NEX-5N's numbers.
    let nex_5n = default_of(&patchwork("NEX-5N"));
    assert_ne!(nex_5n, a7r4);
    assert_eq!(default_of(&patchwork_with_id("ILCE-3000", 302)), nex_5n);
    assert_eq!(default_of(&patchwork("ILCE-3000")), nex_5n);
    assert_eq!(default_of(&patchwork_with_id("ILCE-3500", 302)), nex_5n);
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
        let edited = edit(&src, |s| {
            s.contrast = ym_core::settings::PercentSetting::manual_value(70.0)
        });
        let t = r
            .sample_pixel(&edited, &st, (x, y), &CancelToken::new())
            .expect("sample")
            .expect("inside");
        assert_eq!((s.r, s.g, s.b), (t.r, t.g, t.b));
    }
}

// ---------------------------------------------------------------------------------------------
// No vignetting correction in the Before view (owner's decision 2026-10-16, STATUS question 29)
// ---------------------------------------------------------------------------------------------

/// The vignetting table of the photos below: 16 knots up to a raw value of 3150.
fn vignette_knots() -> Vec<i16> {
    (0..16).map(|i| (i * i * 14) as i16).collect()
}

/// A flat field, middle grey after the camera gain, darkened by a lens with a known vignetting table
/// (raw = flat x G(r)), and with a distortion table when asked, as the camera writes them. The final
/// picture corrects the darkening (corner = centre); the camera's own picture, and so the Before view,
/// does not.
fn vignetted(with_distortion: bool) -> PhotoSource {
    let (w, h) = (800usize, 600usize);
    let knots = vignette_knots();
    let mut tag = vec![16i16];
    tag.extend(&knots);
    let profile = VignetteProfile::from_tag(&knots).expect("profile");
    let level = 0.18 / (0.9 * CAMERA_GAIN_EV.exp2());
    let scene = Scene::new(w as u32, h as u32, WHITE, Layout::Uniform, level);
    let mut mosaic = scene.mosaic();
    let half_diagonal = 0.5 * (w as f64).hypot(h as f64);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 + 0.5 - w as f64 / 2.0;
            let dy = y as f64 + 0.5 - h as f64 / 2.0;
            let g = profile.attenuation_at(dx.hypot(dy) / half_diagonal);
            let i = y * w + x;
            mosaic[i] = (scene.black + (f64::from(mosaic[i]) - scene.black) * g).round() as u16;
        }
    }
    let distortion = with_distortion.then(|| {
        let mut d = vec![16i16];
        d.extend((0..16).map(|i| 40 + 25 * i)); // barrel correction up to about 2.7 %
        d
    });
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder("ILCE-7M3")
        .lens_tables(LensTables {
            vignetting: Some(tag),
            chromatic_aberration: None,
            distortion,
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write");
    Developed::open(dir).source()
}

/// What the lens leaves at the very corner of `vignetted`, as the relative illumination `G`.
fn corner_attenuation() -> f64 {
    let profile = VignetteProfile::from_tag(&vignette_knots()).expect("profile");
    profile.attenuation_at(0.985)
}

/// A fit view of 300 x 225 CSS pixels: the 800 x 600 photos below (and their distortion-cropped size) are
/// shown at 1/2 or less, so the raw group at 1/2 is what the view reads.
fn fit_state(before: bool) -> ViewportState {
    let mut st = ViewportState::new(ViewportRect::cell(300.0, 225.0, 1.0));
    st.before = before;
    st
}

fn frame_in(r: &Renderer, src: &PhotoSource, st: &ViewportState) -> RenderedFrame {
    r.viewport_frame(src, st, &FrameOptions::default(), &CancelToken::new())
        .expect("frame")
}

/// The green code in a patch at the top-left corner over the one at the centre of the frame (patches of
/// 6 % of the frame's size).
fn corner_over_centre(f: &RenderedFrame) -> f64 {
    let (pw, ph) = (f.width * 6 / 100, f.height * 6 / 100);
    let corner = mean_rgba(f, 1, 1, pw, ph)[1];
    let mid = mean_rgba(f, f.width / 2 - pw / 2, f.height / 2 - ph / 2, pw, ph)[1];
    corner / mid
}

fn both_renderers() -> Vec<(&'static str, Renderer)> {
    let mut v = vec![("cpu", cpu())];
    v.extend(device().map(|d| ("device", d)));
    v
}

#[test]
fn before_at_the_corners_is_not_brightened_by_the_correction() {
    // The code at the corner after the camera curve, were the lens darkening left in: the flat field is
    // middle grey (0.18) after the camera gain, the corner is `G` of it.
    let tone = ym_color::camera_default::CameraTone::FITTED;
    let code = |x: f64| 255.0 * srgb_encode(tone.apply(x));
    let expected = code(0.18 * corner_attenuation()) / code(0.18);
    for with_distortion in [false, true] {
        let src = vignetted(with_distortion);
        for (name, r) in both_renderers() {
            let fin = corner_over_centre(&frame_in(&r, &src, &fit_state(false)));
            let bef_frame = frame_in(&r, &src, &fit_state(true));
            let bef = corner_over_centre(&bef_frame);
            // The final picture corrects the darkening; the Before view does not.
            assert!(
                (fin - 1.0).abs() < 0.02,
                "{name}, distortion {with_distortion}: final corner/centre {fin:.3}"
            );
            assert!(
                bef < 0.975,
                "{name}, distortion {with_distortion}: Before corner/centre {bef:.3}, not darker than the centre"
            );
            // And it is the darkening the camera's curve gives it (the corner patch spans a few radii).
            assert!(
                (bef - expected).abs() < 0.03,
                "{name}, distortion {with_distortion}: Before corner/centre {bef:.3}, expected about {expected:.3}"
            );
            eprintln!(
                "[Before] {name}, distortion {with_distortion}: corner/centre final {fin:.3}, Before {bef:.3} (expected {expected:.3})"
            );
            // The geometry is the final picture's: the same size, with the distortion crop.
            let cancel = CancelToken::new();
            let size = r.final_size(&src, &cancel).expect("size");
            assert_eq!(
                r.final_size(&src.for_before_view(), &cancel).expect("size"),
                size
            );
            assert_eq!(size.0 < 800, with_distortion, "{size:?}");
            if !with_distortion {
                // With only a vignetting table in the file, the lens switch does one thing, R3, so
                // the Before view is bit-identical with the switch on and off. (A decode switch
                // changes the Before view where it moves pixels, as `docs/02` says.)
                let off = edit(&src, |s| s.decode.lens_corrections = false);
                let bef_off = frame_in(&r, &off, &fit_state(true));
                assert!(
                    bef_frame.rgba == bef_off.rgba,
                    "{name}: the Before view still has R3"
                );
                // The final picture does change with the switch: the test is not vacuous.
                let fin_off = frame_in(&r, &off, &fit_state(false));
                let fin_on = frame_in(&r, &src, &fit_state(false));
                assert!(
                    fin_off.rgba != fin_on.rgba,
                    "{name}: R3 moves the final picture"
                );
            }
        }
    }
}

/// The largest difference in 8-bit codes between two frames, and how many bytes differ by more than one.
fn worst_codes(a: &RenderedFrame, b: &RenderedFrame) -> (i32, usize) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let (mut worst, mut over) = (0i32, 0usize);
    for (x, y) in a.rgba.iter().zip(&b.rgba) {
        let d = (i32::from(*x) - i32::from(*y)).abs();
        worst = worst.max(d);
        over += usize::from(d > 1);
    }
    (worst, over)
}

#[test]
fn the_device_and_the_cpu_agree_on_the_before_view_without_the_vignetting_correction() {
    let Some(on_device) = device() else {
        return;
    };
    let on_cpu = cpu();
    let src = vignetted(true);
    let (w, h) = on_cpu
        .final_size(&src.for_before_view(), &CancelToken::new())
        .expect("size");
    for (zoom, name) in [
        (None, "fit (the reduced group)"),
        (Some(100.0), "100 % (tiles)"),
    ] {
        let mut st = fit_state(true);
        if let Some(value) = zoom {
            st.set_mode(ViewportMode::Percent { value }, (w, h));
        }
        let a = frame_in(&on_cpu, &src, &st);
        let b = frame_in(&on_device, &src, &st);
        assert_eq!(b.report.device, ComputeDevice::Gpu);
        assert_eq!(a.report.scale == 1, zoom.is_some());
        let (worst, over) = worst_codes(&a, &b);
        eprintln!(
            "[Before] CPU against device, {name}: worst {worst} codes, {over} of {} bytes more than a code off",
            a.rgba.len()
        );
        // The same bound as the final picture's parity test above: f32 against f64 moves a value across a
        // dither threshold now and then, never by more than two codes, and rarely.
        assert!(
            worst <= 2 && over * 1000 <= a.rgba.len(),
            "{name}: worst {worst}, {over} of {} bytes",
            a.rgba.len()
        );
        // The corner is not brightened on either.
        for f in [&a, &b] {
            if zoom.is_none() {
                assert!(corner_over_centre(f) < 0.975, "{name}");
            }
        }
    }
}

#[test]
fn flipping_to_before_never_mixes_the_raw_fronts() {
    let src = vignetted(true);
    let (w, h) = cpu().final_size(&src, &CancelToken::new()).expect("size");
    for zoom in [None, Some(100.0)] {
        for (name, r) in both_renderers() {
            let mut fin = fit_state(false);
            let mut bef = fit_state(true);
            if let Some(value) = zoom {
                fin.set_mode(ViewportMode::Percent { value }, (w, h));
                bef.set_mode(ViewportMode::Percent { value }, (w, h));
            }
            let label = format!("{name}, zoom {zoom:?}");
            let f1 = frame_in(&r, &src, &fin);
            // The Before view makes its own raw front: a group (or the chain and tiles) of its own, and
            // not a second sigma plane (the noise does not depend on R3).
            let b1 = frame_in(&r, &src, &bef);
            let runs = b1.report.runs;
            if zoom.is_none() {
                assert_eq!(
                    runs.raw_groups, 1,
                    "{label}: Before makes its own raw group"
                );
            } else {
                assert_eq!(
                    runs.full_chains, 1,
                    "{label}: Before prepares its own chain"
                );
                assert!(runs.tiles > 0, "{label}: and its own tiles");
            }
            assert_eq!(runs.sigma_planes, 0, "{label}: the sigma plane is shared");
            assert!(
                f1.rgba != b1.rgba,
                "{label}: Before is not the final picture"
            );
            // Flipping back and forth does no raw work, and each picture is what it was.
            let f2 = frame_in(&r, &src, &fin);
            let b2 = frame_in(&r, &src, &bef);
            for (again, first, what) in [(&f2, &f1, "final"), (&b2, &b1, "Before")] {
                let runs = again.report.runs;
                assert_eq!(
                    (
                        runs.raw_groups,
                        runs.full_chains,
                        runs.tiles,
                        runs.sigma_planes
                    ),
                    (0, 0, 0, 0),
                    "{label}: {what} again made raw work"
                );
                assert!(again.rgba == first.rgba, "{label}: {what} changed");
            }
            // An edit of Detail, which reaches the raw group of the final picture, does not disturb the
            // Before view's (it ignores Detail), and a fresh renderer makes the same Before view: nothing
            // was mixed. (`the_detail_settings_do_not_reach_the_before_view` shows that Detail does move
            // the final picture of a noisy photo; this flat field has no noise to reduce.)
            let edited = edit(&src, |s| {
                s.noise_reduction = AutoOr::Manual(ym_core::settings::NoiseReductionValue {
                    luminance: 80.0,
                    color: 50.0,
                    detail: 30.0,
                });
            });
            let b3 = frame_in(&r, &edited, &bef);
            assert!(
                b3.rgba == b1.rgba,
                "{label}: Detail reached the Before view"
            );
            assert_eq!(
                b3.report.runs.raw_groups + b3.report.runs.tiles,
                0,
                "{label}"
            );
            let fresh = if name == "cpu" {
                cpu()
            } else {
                device().expect("device")
            };
            let b4 = frame_in(&fresh, &src, &bef);
            assert!(
                b4.rgba == b1.rgba,
                "{label}: a fresh renderer makes another Before view"
            );
            assert!(frame_in(&fresh, &src, &fin).rgba == f1.rgba, "{label}");
        }
    }
}

#[test]
fn a_photo_with_no_vignetting_to_apply_shares_its_raw_front_with_the_before_view() {
    // No table in the file, or the lens switch off: R3 does nothing either way, so the Before view reads the
    // final picture's raw group, chain and tiles, and flipping to it makes no raw work.
    let scene = Scene::new(800, 600, WHITE, Layout::Patchwork, 0.4);
    let dir = tempfile::tempdir().expect("dir");
    scene
        .builder("ILCE-7M3")
        .lens_tables(LensTables {
            vignetting: None,
            chromatic_aberration: None,
            distortion: None,
        })
        .write_to(&dir.path().join("DSC00001.ARW"), &scene.mosaic())
        .expect("write");
    let without_table = Developed::open(dir).source();
    let switched_off = edit(&vignetted(false), |s| s.decode.lens_corrections = false);
    let (w, h) = (800, 600);
    for (what, src) in [("no table", &without_table), ("switch off", &switched_off)] {
        for zoom in [None, Some(100.0)] {
            for (name, r) in both_renderers() {
                let mut fin = fit_state(false);
                let mut bef = fit_state(true);
                if let Some(value) = zoom {
                    fin.set_mode(ViewportMode::Percent { value }, (w, h));
                    bef.set_mode(ViewportMode::Percent { value }, (w, h));
                }
                let label = format!("{what}, {name}, zoom {zoom:?}");
                let f = frame_in(&r, src, &fin);
                let b = frame_in(&r, src, &bef);
                let runs = b.report.runs;
                assert_eq!(
                    (
                        runs.raw_groups,
                        runs.full_chains,
                        runs.tiles,
                        runs.sigma_planes
                    ),
                    (0, 0, 0, 0),
                    "{label}: the Before view made raw work of its own"
                );
                assert!(f.rgba != b.rgba, "{label}: Before is not the final picture");
            }
        }
    }
}

#[test]
fn the_detail_settings_do_not_reach_the_before_view() {
    // Owner's decision 2026-10-16 (STATUS question 29): the Before view reads Detail at Auto whatever the
    // settings say. A noisy photo, so that Detail moves the final picture (the test is not vacuous), at
    // fit (noise reduction) and at 100 % (noise reduction and capture sharpening).
    let scene = Scene::new(600, 400, WHITE, Layout::Patchwork, 0.4)
        .noise(2e-4, 2e-6)
        .seed(7);
    let src = Developed::new(&scene, "ILCE-7M3").source();
    let strong = edit(&src, |s| {
        s.noise_reduction = AutoOr::Manual(ym_core::settings::NoiseReductionValue {
            luminance: 100.0,
            color: 100.0,
            detail: 0.0,
        });
        s.sharpen = AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 300.0,
            radius_px: 2.5,
        });
    });
    let none = edit(&src, |s| {
        s.noise_reduction = AutoOr::Manual(ym_core::settings::NoiseReductionValue {
            luminance: 0.0,
            color: 0.0,
            detail: 100.0,
        });
        s.sharpen = AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 0.0,
            radius_px: 0.5,
        });
    });
    for (name, r) in both_renderers() {
        let (w, h) = r.final_size(&src, &CancelToken::new()).expect("size");
        for zoom in [None, Some(100.0)] {
            let state = |before: bool| {
                let mut st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
                st.before = before;
                if let Some(value) = zoom {
                    st.set_mode(ViewportMode::Percent { value }, (w, h));
                }
                st
            };
            let reference = frame_in(&r, &src, &state(true));
            for (what, edited) in [("strong", &strong), ("none", &none)] {
                let b = frame_in(&r, edited, &state(true));
                assert!(
                    b.rgba == reference.rgba,
                    "{name}, zoom {zoom:?}: Detail ({what}) reached the Before view"
                );
            }
            let moved = frame_in(&r, &strong, &state(false));
            let plain = frame_in(&r, &src, &state(false));
            assert!(
                moved.rgba != plain.rgba,
                "{name}, zoom {zoom:?}: Detail moves the final picture"
            );
        }
    }
}

#[test]
fn a_source_made_for_the_before_view_says_so() {
    let src = patchwork("ILCE-7M3");
    assert!(!src.before_view);
    let strong = edit(&src, |s| {
        s.contrast = ym_core::settings::PercentSetting::manual_value(40.0);
        s.sharpen = AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 300.0,
            radius_px: 2.5,
        });
    });
    let b = strong.for_before_view();
    assert!(b.before_view);
    // Detail is at Auto; everything else of the settings is as it was (no setting reaches the view anyway).
    assert!(b.settings.noise_reduction.is_auto() && b.settings.sharpen.is_auto());
    assert_eq!(
        b.settings.contrast,
        ym_core::settings::PercentSetting::manual_value(40.0)
    );
    // The flag survives an edit of the settings, and the original is not marked.
    assert!(
        edit(&b, |s| s.contrast =
            ym_core::settings::PercentSetting::manual_value(10.0))
        .before_view
    );
    assert!(!strong.before_view);
    // The camera default is the same for both: it comes from the file and the body.
    let g0 = src.estimates.initial_gains().expect("gains");
    assert_eq!(camera_default_of(&b, g0), camera_default_of(&src, g0));
}
