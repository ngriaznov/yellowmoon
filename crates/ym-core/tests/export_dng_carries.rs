//! What a DNG carries (`docs/09`, "What a DNG carries"): no sharpening of any kind in its pixels and its
//! preview, and the same picture as the app's through the profile the DNG embeds.
//!
//! * R8 capture sharpening (Richardson-Lucy or diffraction deconvolution) is off for both renders of a
//!   DNG, whatever the photo's Sharpening setting says, and stays on for JPEG, TIFF and PNG (decision of
//!   2026-10-14; output sharpening was never applied to a DNG). The renders of a DNG do not leak into
//!   the others through the caches.
//! * For every look the DNG's pixels, through ACES 2 with the display placement (the transform the fitted
//!   profile emulates, with the placement in `BaselineExposure`), are the picture the app makes: for a
//!   log-to-log look because the DNG holds the C11 output as it is, for a look with its own display
//!   rendering (LogC3 to Rec.709) and for an imported display LUT because the DNG holds the scene values
//!   that ACES 2 turns into that picture.
mod common;

use std::path::Path;
use std::sync::Arc;

use common::dng_looks::{DISPLAY_LUT_ID, Looks};
use common::*;
use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::consts::linalg;
use ym_color::display::{DISPLAY_PLACEMENT_EV, DNG_SCENE_LIMIT, OutputTarget, aces2_input_matrix};
use ym_core::export::{ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    DisplayChoice, FullImage, FullRequest, FullTarget, LookProvider, NeutralOnly, PhotoSource,
    RenderOptions, Renderer,
};
use ym_core::session::{OpenOptions, Session};
use ym_core::settings::{AutoOr, DevelopSettings, SharpenValue};
use ym_export::{BitDepth, ExportOptions, FormatKind, TiffCompression};
use ym_gpu::queue::CancelToken;
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A patchwork with edges, flat patches and a little noise: what capture sharpening acts on.
fn detailed_scene() -> Scene {
    Scene::new(420, 280, light(PROFILE, 5000.0), Layout::Patchwork, 0.45).noise(2e-4, 2e-6)
}

fn source_of(scene: &Scene) -> PhotoSource {
    Developed::new(scene, MODEL).source()
}

fn with_sharpening(src: &PhotoSource, amount: f64, radius_px: f64) -> PhotoSource {
    let mut s: DevelopSettings = src.settings.clone();
    s.sharpen = AutoOr::Manual(SharpenValue { amount, radius_px });
    src.with_settings(s)
}

fn cpu_renderer(looks: Arc<dyn LookProvider>) -> Renderer {
    Renderer::new(looks, RenderOptions::default()).expect("render thread")
}

fn render(r: &Renderer, src: &PhotoSource, target: FullTarget) -> FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("a render")
}

/// The largest absolute difference between two renders of the same size.
fn max_diff(a: &FullImage, b: &FullImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.rgb
        .iter()
        .zip(&b.rgb)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------------------------
// No sharpening in a DNG
// ---------------------------------------------------------------------------------------------

#[test]
fn the_renders_of_a_dng_have_no_capture_sharpening_whatever_the_setting_says() {
    let src = source_of(&detailed_scene());
    let r = cpu_renderer(Arc::new(NeutralOnly));
    let off = with_sharpening(&src, 0.0, 1.0);
    let strong = with_sharpening(&src, 250.0, 2.2);

    // The control: for a picture of the other formats R8 is on, and it changes this scene by far more
    // than a rounding error (otherwise the equalities below would prove nothing).
    let plain = |s: &PhotoSource| render(&r, s, FullTarget::DisplayLinear(OutputTarget::Srgb));
    let (auto_picture, off_picture, strong_picture) = (plain(&src), plain(&off), plain(&strong));
    let effect = max_diff(&auto_picture, &off_picture);
    assert!(
        effect > 1e-3,
        "capture sharpening changes this scene by {effect}"
    );
    assert!(max_diff(&strong_picture, &auto_picture) > 1e-4);

    // The raw pixels of a DNG: the same bits for every amount (R8 is off in all of them).
    let raw = |s: &PhotoSource| render(&r, s, FullTarget::DngLinearAwg4);
    let (raw_auto, raw_off, raw_strong) = (raw(&src), raw(&off), raw(&strong));
    assert_eq!(raw_auto.rgb, raw_off.rgb, "automatic sharpening or none");
    assert_eq!(raw_strong.rgb, raw_off.rgb, "strong sharpening or none");

    // The preview: the same, and the same bits as the display render with R8 switched off by the
    // setting, the one other way to make this picture (the chain, tiles and geometry are the same).
    let preview = |s: &PhotoSource| render(&r, s, FullTarget::DngPreview);
    let (prev_auto, prev_strong) = (preview(&src), preview(&strong));
    assert_eq!(prev_auto.rgb, prev_strong.rgb);
    assert_eq!(
        prev_auto.rgb, off_picture.rgb,
        "the DNG preview is the render with R8 off"
    );
    assert_ne!(prev_auto.rgb, auto_picture.rgb);
}

#[test]
fn the_dng_drops_r8_and_keeps_the_demosaic_as_it_is() {
    // The adaptive ratio demosaic's own shadow step (`shadow_sharpening`) is part of the demosaic, not of
    // R8: the renderer leaves it at its default, on, and the switch that takes R8 out of a DNG does not
    // reach it. What a DNG holds is therefore the demosaic's picture: another demosaic gives other pixels.
    assert!(ym_gpu::cpu::demosaic::adaptive_ratio::Params::default().shadow_sharpening);
    let src = source_of(&detailed_scene());
    let r = cpu_renderer(Arc::new(NeutralOnly));
    let bilinear = Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            demosaic: ym_gpu::cpu::demosaic::Method::Bilinear,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let ours = render(&r, &src, FullTarget::DngLinearAwg4);
    let other = render(&bilinear, &src, FullTarget::DngLinearAwg4);
    assert!(max_diff(&ours, &other) > 1e-3);
}

fn export_options(out: &Path, format: FormatKind) -> ExportOptions {
    ExportOptions {
        format,
        destination: Some(out.to_path_buf()),
        bit_depth: BitDepth::Sixteen,
        tiff_compression: TiffCompression::None,
        ..ExportOptions::default()
    }
}

#[test]
fn a_dng_export_leaves_the_other_formats_as_they_were() {
    let dir = tempfile::tempdir().expect("source folder");
    detailed_scene().write(dir.path(), "DSC00001.ARW", MODEL);
    let session = Session::new(base_env());
    let opened = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1);
    settle(&session);
    let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("export photo");
    let (home, out) = (
        tempfile::tempdir().expect("home"),
        tempfile::tempdir().expect("out"),
    );
    let ctx = export_context(home.path());
    let export = |photo: &ExportPhoto, format: FormatKind, name: &str| {
        let mut o = export_options(&out.path().join(name), format);
        o.if_exists = ym_export::IfExists::Replace;
        let report = export_photo(&ctx, photo, &ExportParams::new(o), &ExportControl::none())
            .expect("export");
        std::fs::read(report.path).expect("read the export")
    };
    let tiff_before = export(&photo, FormatKind::Tiff, "before");
    let _dng = export(&photo, FormatKind::Dng, "dng");
    let tiff_after = export(&photo, FormatKind::Tiff, "after");
    assert!(
        tiff_before == tiff_after,
        "a TIFF made after a DNG is the TIFF made before it"
    );
    // And a raster export still carries R8 (decision of 2026-10-14: only the DNG lost it): the same
    // photo with the sharpening set to 0 is another TIFF.
    let mut unsharpened = photo.clone();
    unsharpened.source = with_sharpening(&photo.source, 0.0, 1.0);
    let tiff_off = export(&unsharpened, FormatKind::Tiff, "off");
    assert!(tiff_before != tiff_off, "R8 is still on for a TIFF");
}

// ---------------------------------------------------------------------------------------------
// The DNG's pixels are the app's picture through ACES 2
// ---------------------------------------------------------------------------------------------

fn with_look(src: &PhotoSource, look: &str, intensity: f64) -> PhotoSource {
    let mut s = src.settings.clone();
    s.look = LookId::parse(look).expect("look id");
    s.look_intensity = intensity;
    src.with_settings(s)
}

/// What Adobe's renderer is made to show: the DNG's scene-linear AWG4 pixels, with the display placement
/// (carried by `BaselineExposure`), through ACES 2 for sRGB. Display-linear Rec.709.
fn the_dng_through_aces2(raw: &FullImage) -> Vec<f64> {
    let t = Aces2Transform::shared(Aces2Output::Srgb);
    let m = aces2_input_matrix();
    raw.rgb
        .chunks_exact(3)
        .flat_map(|p| {
            let awg4 = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            t.transform_ap0_linear(linalg::mul_vec(&m, awg4))
        })
        .collect()
}

/// How the DNG's pixels, through ACES 2, differ from the app's picture where the picture is inside the
/// range ACES 2 reaches (display values to 0.9, the DNG pixel below the profile's limit): the largest
/// difference and the number of pixels compared.
fn dng_against_picture(raw: &FullImage, picture: &FullImage) -> (f64, usize) {
    assert_eq!((raw.width, raw.height), (picture.width, picture.height));
    let shown = the_dng_through_aces2(raw);
    let limit = DNG_SCENE_LIMIT / DISPLAY_PLACEMENT_EV.exp2();
    let (mut worst, mut n) = (0.0_f64, 0usize);
    for (i, (p, raw_px)) in picture
        .rgb
        .chunks_exact(3)
        .zip(raw.rgb.chunks_exact(3))
        .enumerate()
    {
        let in_range = p.iter().all(|&c| (0.0..=0.9).contains(&c))
            && raw_px.iter().all(|&c| f64::from(c) < limit * 0.99);
        if !in_range {
            continue;
        }
        for k in 0..3 {
            worst = worst.max((shown[3 * i + k] - f64::from(p[k])).abs());
        }
        n += 1;
    }
    (worst, n)
}

#[test]
fn for_every_look_the_dngs_pixels_through_aces2_are_the_apps_picture() {
    // The photo's R8 is set to 0, so the app's picture has none either (a DNG never has any).
    let src = with_sharpening(&source_of(&detailed_scene()), 0.0, 1.0);
    let r = cpu_renderer(Arc::new(Looks));
    for (name, look, intensity) in [
        ("neutral", "builtin:neutral", 100.0),
        ("a log-to-log look", "user:warm", 100.0),
        ("a log-to-log look at 40 %", "user:warm", 40.0),
        ("LogC3 to Rec.709", "user:rec709", 100.0),
        ("LogC3 to Rec.709 at 60 %", "user:rec709", 60.0),
        ("LogC3 to Rec.709 at 0 %", "user:rec709", 0.0),
    ] {
        let s = with_look(&src, look, intensity);
        let raw = render(&r, &s, FullTarget::DngLinearAwg4);
        let picture = render(&r, &s, FullTarget::DisplayLinear(OutputTarget::Srgb));
        let (worst, n) = dng_against_picture(&raw, &picture);
        eprintln!("{name}: {n} pixels compared, the DNG through ACES 2 differs by {worst:.2e}");
        assert!(n > raw.width * raw.height / 10, "{name}: {n} pixels");
        // f32 pixels through a transform that is steep in the shadows: measured 9e-8 of display light.
        assert!(worst < 1e-5, "{name}: {worst}");
        // The preview is the picture of the app (sRGB, display-linear), not the DNG's own render.
        let preview = render(&r, &s, FullTarget::DngPreview);
        assert_eq!(preview.rgb, picture.rgb, "{name}: the preview");
    }
}

#[test]
fn only_a_display_rendering_of_arri_changes_the_dng_pixels_of_a_look_of_ours() {
    // The DNG of a log-to-log look is the C11 output decoded from LogC4: nothing in it depends on the
    // look's display rendering. A look with its own rendering at 0 % is our C12 exactly (the same DNG
    // as Neutral), and an imported display LUT is the one other thing that moves the pixels.
    let src = source_of(&detailed_scene());
    let r = cpu_renderer(Arc::new(Looks));
    let neutral = with_look(&src, "builtin:neutral", 100.0);
    let zero = with_look(&src, "user:rec709", 0.0);
    assert_eq!(
        render(&r, &neutral, FullTarget::DngLinearAwg4).rgb,
        render(&r, &zero, FullTarget::DngLinearAwg4).rgb,
        "a look with its own rendering at 0 % is Neutral"
    );
    for look in ["builtin:neutral", "user:warm"] {
        let s = with_look(&src, look, 100.0);
        let ours = render(&r, &s, FullTarget::DngLinearAwg4);
        r.set_display(DisplayChoice::ArriLut(
            LookId::parse(DISPLAY_LUT_ID).expect("id"),
        ));
        let theirs = render(&r, &s, FullTarget::DngLinearAwg4);
        r.set_display(DisplayChoice::Aces2);
        assert!(max_diff(&ours, &theirs) > 1e-2, "{look}");
        assert_eq!(
            render(&r, &s, FullTarget::DngLinearAwg4).rgb,
            ours.rgb,
            "{look}: back to ACES 2 gives the same pixels"
        );
    }
}

#[test]
fn an_imported_display_lut_goes_into_the_dng_through_the_same_inverse() {
    let src = with_sharpening(&source_of(&detailed_scene()), 0.0, 1.0);
    let r = cpu_renderer(Arc::new(Looks));
    r.set_display(DisplayChoice::ArriLut(
        LookId::parse(DISPLAY_LUT_ID).expect("id"),
    ));
    let raw = render(&r, &src, FullTarget::DngLinearAwg4);
    let picture = render(&r, &src, FullTarget::DisplayLinear(OutputTarget::Srgb));
    let (worst, n) = dng_against_picture(&raw, &picture);
    eprintln!("display LUT: {n} pixels compared, the DNG through ACES 2 differs by {worst:.2e}");
    assert!(n > raw.width * raw.height / 4, "{n}");
    assert!(worst < 1e-5, "{worst}");
    // The values stay inside what the profile covers: no channel above the profile's limit.
    let limit = (DNG_SCENE_LIMIT / DISPLAY_PLACEMENT_EV.exp2()) as f32;
    assert!(raw.rgb.iter().all(|&c| c <= limit * (1.0 + 1e-6)));
}

#[test]
fn the_scene_limit_of_the_inverse_is_the_top_of_the_profile_the_dng_embeds() {
    // `DNG_SCENE_LIMIT` is where the inverse stops the scene values, and where Adobe's renderer ends the profile's
    // tone curve after `BaselineExposure`: the floor of the white level of the embedded profile.
    let fitted = ym_export::dng::FittedProfile::embedded();
    assert_eq!(
        f64::from(fitted.baseline_exposure().white_floor()),
        DNG_SCENE_LIMIT
    );
    // And with the placement the largest AWG4 channel the inverse gives is `limit / 2^k`, the stored headroom a user
    // who lowers the exposure in the other editor has above it (`docs/09`).
    assert!((DNG_SCENE_LIMIT / DISPLAY_PLACEMENT_EV.exp2() - 8.9).abs() < 0.05);
}
