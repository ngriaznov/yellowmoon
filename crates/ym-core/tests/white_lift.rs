//! The white lift in the core (`docs/02` C12 step 1b, D-02): the automatic estimate of a photo, the
//! `white_offset_ev` setting, the render, the DNG and the device.
//!
//! * a bright photo gets a lift, a low-key one gets none, and the estimate is the same whatever the
//!   number of decode threads;
//! * the setting is an offset on the automatic amount: 0 is automatic, a negative offset of the
//!   amount (or any beyond) gives the render of no lift, bit for bit;
//! * the panel's numbers (`ResolvedSettings`, the `auto` summary) say what the render did;
//! * the DNG's pixels through ACES 2 with the placement are the app's picture, with the lift in them;
//! * an imported ARRI display LUT has no lift;
//! * the lift works alongside C7's highlight range: it is sized through the default range, so the
//!   user's `rolloff` never re-sizes it, and both stages are in the render;
//! * the device agrees with the CPU on a lifted photo.
//!
//! Synthetic scenes only; the looks are our own.
mod common;

use std::sync::Arc;

use common::dng_looks::{DISPLAY_LUT_ID, Looks};
use common::*;
use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::consts::linalg;
use ym_color::display::{OutputTarget, aces2_input_matrix};
use ym_core::pipeline::{
    ComputeDevice, DisplayChoice, FrameOptions, FullImage, FullRequest, FullTarget, NeutralOnly,
    PhotoSource, RenderOptions, Renderer, ViewportRect, ViewportState,
};
use ym_core::session::{OpenOptions, Session};
use ym_core::settings::{DevelopSettings, ResolvedSettings};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::CancelToken;
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A patchwork whose brightest neutral patch is about +2.3 stops over middle grey at the camera's
/// exposure: a high-key photo to the estimator.
fn bright_scene() -> Scene {
    Scene::new(420, 280, light(PROFILE, 5000.0), Layout::Patchwork, 0.45).noise(2e-4, 2e-6)
}

/// A patchwork whose brightest patch is below +1 stop.
fn dark_scene() -> Scene {
    Scene::new(420, 280, light(PROFILE, 5000.0), Layout::Patchwork, 0.08).noise(2e-4, 2e-6)
}

fn with_offset(src: &PhotoSource, offset_ev: f64) -> PhotoSource {
    let mut s: DevelopSettings = src.settings.clone();
    s.white_offset_ev = offset_ev;
    src.with_settings(s)
}

fn cpu_renderer() -> Renderer {
    Renderer::new(Arc::new(NeutralOnly), RenderOptions::default()).expect("render thread")
}

fn render(r: &Renderer, src: &PhotoSource, target: FullTarget) -> FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("a render")
}

fn display(r: &Renderer, src: &PhotoSource) -> FullImage {
    render(r, src, FullTarget::DisplayLinear(OutputTarget::Srgb))
}

#[test]
fn a_bright_photo_gets_a_lift_and_a_dark_one_does_not() {
    let bright = Developed::new(&bright_scene(), MODEL).source();
    let w = bright.estimates.white;
    assert!(
        w.amount_ev > 0.1 && w.amount_ev <= 2.0,
        "a bright photo's lift: {w:?}"
    );
    assert!(w.anchor_ev > 1.0 && w.high_key > 0.3, "{w:?}");
    assert!(w.code_after > w.code_before, "{w:?}");

    let dark = Developed::new(&dark_scene(), MODEL).source();
    assert_eq!(
        dark.estimates.white.amount_ev, 0.0,
        "{:?}",
        dark.estimates.white
    );
    assert_eq!(dark.estimates.white.high_key, 0.0);
    // No lift in effect either: the settings panel says 0.
    assert_eq!(
        ym_core::pipeline::color::white_lift(&dark).amount_ev(),
        0.0,
        "a low-key photo renders without the stage"
    );
}

#[test]
fn the_estimate_is_the_same_whatever_the_decode_threads() {
    let dir = tempfile::tempdir().expect("dir");
    let path = bright_scene().write(dir.path(), "DSC00001.ARW", MODEL);
    let mut estimates = Vec::new();
    for threads in [1usize, 2, 4] {
        let mut env = base_env();
        env.decode_threads = threads;
        let session = Session::new(env);
        let out = session.open(std::slice::from_ref(&path), OpenOptions::default());
        settle(&session);
        let view = session.view(out.added[0]).expect("view");
        estimates.push(view.estimates.white);
    }
    assert!(estimates[0].amount_ev > 0.0);
    assert!(estimates.windows(2).all(|p| p[0] == p[1]), "{estimates:?}");
    // and from the bare function of the decoded file, as often as asked
    let (_, a, _) = estimates_of_file(&path);
    let (_, b, _) = estimates_of_file(&path);
    assert_eq!(a.white, b.white);
}

#[test]
fn the_offset_adds_to_the_automatic_amount_and_the_render_follows_it() {
    let src = Developed::new(&bright_scene(), MODEL).source();
    let auto = src.estimates.white.amount_ev;
    assert!(auto > 0.1);
    let r = cpu_renderer();
    // Offset 0 is the automatic amount.
    assert_eq!(src.settings.white_offset_ev, 0.0);
    let lift = ym_core::pipeline::color::white_lift(&src);
    assert_eq!(lift.amount_ev(), auto);
    assert_eq!(lift.anchor_ev(), src.estimates.white.anchor_ev);
    let at_auto = display(&r, &src);
    assert_eq!(at_auto.rgb, display(&r, &with_offset(&src, 0.0)).rgb);

    // Minus the automatic amount, and anything beyond it, is the render with no lift at all.
    let none = display(&r, &with_offset(&src, -3.0));
    for offset in [-auto, -auto - 0.25, -2.0] {
        let s = with_offset(&src, offset);
        assert!(
            ym_core::pipeline::color::white_lift(&s).is_identity(),
            "{offset}"
        );
        assert_eq!(display(&r, &s).rgb, none.rgb, "offset {offset}");
    }
    // The lift is brighter than none at the top of the photo, and equal where the tones are low.
    let (mut brighter, mut equal) = (0usize, 0usize);
    for (a, n) in at_auto.rgb.chunks_exact(3).zip(none.rgb.chunks_exact(3)) {
        if a[1] > n[1] {
            brighter += 1;
        } else if a == n {
            equal += 1;
        }
        assert!(a[1] >= n[1] - 1e-6, "a lift never darkens a pixel");
    }
    let pixels = at_auto.rgb.len() / 3;
    assert!(brighter > pixels / 50, "{brighter} of {pixels} brighter");
    assert!(equal > pixels / 4, "{equal} of {pixels} untouched");
    // A bigger offset lifts more: the top of the picture rises with it.
    let max_green = |img: &FullImage| img.rgb.chunks_exact(3).map(|p| p[1]).fold(0.0f32, f32::max);
    let more = display(&r, &with_offset(&src, 0.6));
    assert!(max_green(&more) > max_green(&at_auto));
    assert!(max_green(&at_auto) > max_green(&none));
    // A low-key photo can be given a lift by hand: the offset alone is the amount.
    let dark = Developed::new(&dark_scene(), MODEL).source();
    let given = with_offset(&dark, 0.7);
    assert!((ym_core::pipeline::color::white_lift(&given).amount_ev() - 0.7).abs() < 1e-12);
}

#[test]
fn the_panel_numbers_say_what_the_render_did() {
    let d = Developed::new(&bright_scene(), MODEL);
    let src = d.source();
    let resolve = |s: &PhotoSource| {
        ResolvedSettings::resolve(s.id, &s.settings, &s.estimates, &s.profile, None)
            .expect("resolved")
    };
    let auto = src.estimates.white.amount_ev;
    let rs = resolve(&src);
    assert_eq!(
        rs.estimates.white_ev, auto,
        "the Automatic card's White row"
    );
    assert_eq!(rs.values.white_offset_ev, 0.0);
    assert_eq!(rs.values.white_ev, auto);
    let rs = resolve(&with_offset(&src, 0.4));
    assert_eq!(
        rs.estimates.white_ev, auto,
        "the automatic amount does not move"
    );
    assert!((rs.values.white_ev - (auto + 0.4)).abs() < 1e-12);
    let rs = resolve(&with_offset(&src, -auto - 1.0));
    assert_eq!(rs.values.white_ev, 0.0, "never below 0");
    // The JSON of `ym render --json` and of the bridge carries them.
    let json = serde_json::to_value(&rs).expect("json");
    assert!(json["estimates"]["white_ev"].is_number());
    assert!(json["values"]["white_offset_ev"].is_number());
    assert!(json["settings"]["white_offset_ev"].is_number());
    // A patch from the panel changes the field and nothing else, and the session serves it.
    let patch = ym_core::settings::SettingsPatch {
        white_offset_ev: Some(-0.2),
        ..ym_core::settings::SettingsPatch::default()
    };
    let rs = d.session.update_settings(d.id, &patch).expect("the patch");
    assert!((rs.settings.white_offset_ev + 0.2).abs() < 1e-12);
    assert!((rs.values.white_ev - (auto - 0.2)).abs() < 1e-12);
}

#[test]
fn the_users_highlight_range_does_not_resize_the_lift() {
    // The estimate is made once per open through C7 at its defaults (the highlight range at 1): RAW ›
    // Range is the user's, and moving it moves C7, never the lift's two numbers.
    let src = Developed::new(&bright_scene(), MODEL).source();
    let lift = ym_core::pipeline::color::white_lift(&src);
    assert!(lift.amount_ev() > 0.1, "{lift:?}");
    let r = cpu_renderer();
    let mut pictures = Vec::new();
    for (highlights, low_light) in [(0.0, 0.5), (0.5, 0.0), (1.0, 1.0)] {
        let mut s: DevelopSettings = src.settings.clone();
        s.rolloff.highlights = highlights;
        s.rolloff.low_light = low_light;
        let edited = src.with_settings(s);
        assert_eq!(ym_core::pipeline::color::white_lift(&edited), lift);
        pictures.push(display(&r, &edited).rgb);
    }
    // and the two stages both act: the toe moves the picture with the lift in it
    assert_ne!(pictures[0], pictures[1]);
}

/// What Adobe's renderer is made to show: the DNG's pixels with the placement through ACES 2, limited to
/// P3-D65 (the red chroma step in the pixels uses the P3-D65 tables, `docs/09`; so the picture to compare
/// with is the app's Display P3 one).
fn the_dng_through_aces2(raw: &FullImage) -> Vec<f64> {
    let t = Aces2Transform::shared(Aces2Output::DisplayP3);
    let m = aces2_input_matrix();
    raw.rgb
        .chunks_exact(3)
        .flat_map(|p| {
            let awg4 = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            t.transform_ap0_linear(linalg::mul_vec(&m, awg4))
        })
        .collect()
}

#[test]
fn the_dngs_pixels_are_the_apps_scene_values_before_aces_2_with_the_lift_in_them() {
    let base = Developed::new(&bright_scene(), MODEL).source();
    // Sharpening off: the app's picture then has none either, as a DNG never has.
    let mut s = base.settings.clone();
    s.sharpen = ym_core::settings::AutoOr::Manual(ym_core::settings::SharpenValue {
        amount: 0.0,
        radius_px: 1.0,
    });
    let src = base.with_settings(s);
    let r = cpu_renderer();
    let lifted_raw = render(&r, &src, FullTarget::DngLinearAwg4);
    let picture = render(&r, &src, FullTarget::DisplayLinear(OutputTarget::DisplayP3));
    let shown = the_dng_through_aces2(&lifted_raw);
    let mut worst = 0.0f64;
    for (p, s) in picture.rgb.chunks_exact(3).zip(shown.chunks_exact(3)) {
        for k in 0..3 {
            worst = worst.max((f64::from(p[k]) - s[k]).abs());
        }
    }
    eprintln!("the DNG through ACES 2 against the picture, lifted: {worst:.2e}");
    assert!(worst < 1e-5, "{worst}");
    // The lift is in the DNG: against the DNG of no lift the bright pixels are brighter by the gain
    // and the dark ones are the same bits.
    let none = with_offset(&src, -3.0);
    let plain_raw = render(&r, &none, FullTarget::DngLinearAwg4);
    let (mut higher, mut same) = (0usize, 0usize);
    for (a, b) in lifted_raw
        .rgb
        .chunks_exact(3)
        .zip(plain_raw.rgb.chunks_exact(3))
    {
        if a == b {
            same += 1;
        } else {
            // never darker; just over the knee the gain is within an ulp of 1, so one channel may
            // round up while the others keep their bits
            assert!((0..3).all(|k| a[k] >= b[k]), "{a:?} against {b:?}");
            // all three channels by one factor: the ratios of the pixel are kept
            let g = f64::from(a[1]) / f64::from(b[1]);
            for k in [0, 2] {
                let want = f64::from(b[k]) * g;
                assert!(
                    (f64::from(a[k]) - want).abs() <= 2e-6 * want.abs().max(1.0),
                    "{a:?} {b:?}"
                );
            }
            higher += 1;
        }
    }
    let n = lifted_raw.rgb.len() / 3;
    assert!(
        higher > n / 50 && same > n / 4,
        "{higher} higher, {same} the same, of {n}"
    );
    // The preview of a DNG is the app's picture with the lift.
    assert_eq!(render(&r, &src, FullTarget::DngPreview).rgb, picture.rgb);
}

#[test]
fn an_imported_display_lut_keeps_arris_own_highlights() {
    let src = Developed::new(&bright_scene(), MODEL).source();
    let r = Renderer::new(Arc::new(Looks), RenderOptions::default()).expect("render thread");
    r.set_display(DisplayChoice::ArriLut(
        LookId::parse(DISPLAY_LUT_ID).expect("id"),
    ));
    let with = display(&r, &src);
    let without = display(&r, &with_offset(&src, -3.0));
    assert_eq!(
        with.rgb, without.rgb,
        "the display LUT's picture has no lift"
    );
    for target in [FullTarget::DngLinearAwg4, FullTarget::DngPreview] {
        assert_eq!(
            render(&r, &src, target).rgb,
            render(&r, &with_offset(&src, -3.0), target).rgb,
            "{target:?}"
        );
    }
    // Back on ACES 2 the lift is there again.
    r.set_display(DisplayChoice::Aces2);
    assert_ne!(
        display(&r, &src).rgb,
        display(&r, &with_offset(&src, -3.0)).rgb
    );
}

#[test]
fn the_device_agrees_with_the_cpu_on_a_lifted_photo() {
    let gpu = match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no adapter opened: {e}"
            );
            eprintln!("\n* GPU TESTS SKIPPED in white_lift: {e}\n");
            return;
        }
    };
    let src = Developed::new(&bright_scene(), MODEL).source();
    assert!(src.estimates.white.amount_ev > 0.1);
    let cpu = cpu_renderer();
    let on_gpu = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu: Some(gpu),
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let cancel = CancelToken::new();
    let st = ViewportState::new(ViewportRect::cell(300.0, 200.0, 1.0));
    for (name, s) in [
        ("automatic", src.clone()),
        ("offset +0.5", with_offset(&src, 0.5)),
        ("offset -3 (none)", with_offset(&src, -3.0)),
    ] {
        let a = cpu
            .viewport_frame(&s, &st, &FrameOptions::default(), &cancel)
            .expect("cpu frame");
        let b = on_gpu
            .viewport_frame(&s, &st, &FrameOptions::default(), &cancel)
            .expect("gpu frame");
        assert_eq!(a.report.device, ComputeDevice::Cpu);
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "{name}: {:?}",
            b.report.notes
        );
        let worst = a
            .rgba
            .iter()
            .zip(&b.rgba)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        eprintln!("{name}: GPU against CPU, worst difference {worst} codes");
        assert!(worst <= 2, "{name}: {worst} codes");
    }
    // And the lift moves the frame on the device as it does on the CPU.
    let lifted = on_gpu
        .viewport_frame(&src, &st, &FrameOptions::default(), &cancel)
        .expect("frame");
    let none = on_gpu
        .viewport_frame(
            &with_offset(&src, -3.0),
            &st,
            &FrameOptions::default(),
            &cancel,
        )
        .expect("frame");
    assert_ne!(lifted.rgba, none.rgba);
}
