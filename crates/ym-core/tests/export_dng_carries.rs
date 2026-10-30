//! What a DNG carries (`docs/09`, "What a DNG carries"): no sharpening of any kind in its pixels and its
//! preview, and the same picture as the app's through the profile the DNG embeds.
//!
//! * R8 capture sharpening (Richardson-Lucy or diffraction deconvolution) is off for both renders of a
//!   DNG, whatever the photo's Sharpening setting says, and stays on for JPEG, TIFF and PNG (decision of
//!   2026-10-14; output sharpening was never applied to a DNG). The renders of a DNG do not leak into
//!   the others through the caches.
//! * For every look the DNG's pixels, through the cine rendering (the transform the fitted profile emulates),
//!   are the picture the app makes: for a log-to-log look because the DNG holds the C11 output as it is, for
//!   a look with its own display rendering (LogC3 to Rec.709) because the DNG holds the scene values that
//!   the cine rendering turns into that picture.
mod common;

use std::path::Path;
use std::sync::Arc;

use common::dng_looks::Looks;
use common::*;
use ym_color::display::cine::CineRendering;
use ym_color::display::{DNG_SCENE_LIMIT, OutputTarget};
use ym_color::gamut::LinearAwg4;
use ym_core::export::{ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{
    FullImage, FullRequest, FullTarget, LookProvider, NeutralOnly, PhotoSource, RenderOptions,
    Renderer,
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
// The DNG's pixels are the app's picture through the cine rendering
// ---------------------------------------------------------------------------------------------

fn with_look(src: &PhotoSource, look: &str, intensity: f64) -> PhotoSource {
    let mut s = src.settings.clone();
    s.look = LookId::parse(look).expect("look id");
    s.look_intensity = intensity;
    src.with_settings(s)
}

/// What Adobe's renderer is made to show: the DNG's scene-linear AWG4 pixels through the cine rendering.
/// Display-linear in the output's primaries.
///
/// `ours` is a DNG of the app's own rendering: its pixels are the C11 output as it is, so the picture to
/// compare with is the app's Display P3 one, through the Display P3 rendering. Otherwise the pixels are the
/// inverse of a display rendering that is not ours (a look with its own rendering above 0 %), made with the
/// sRGB rendering (`DngDisplayInverse`), and the picture is the sRGB one.
fn the_dng_through_the_rendering(raw: &FullImage, ours: bool) -> Vec<f64> {
    let rendering = CineRendering::new(picture_target(ours));
    raw.rgb
        .chunks_exact(3)
        .flat_map(|p| {
            let awg4 = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            rendering.forward(LinearAwg4(awg4))
        })
        .collect()
}

/// The output the picture to compare a DNG with is rendered for (see [`the_dng_through_the_rendering`]).
fn picture_target(ours: bool) -> OutputTarget {
    if ours {
        OutputTarget::DisplayP3
    } else {
        OutputTarget::Srgb
    }
}

/// How the DNG's pixels, through the cine rendering, differ from the app's picture where the picture is inside
/// the range the inverse reaches (display values from 2e-6 to 0.9, a colour the rendering can make, the DNG
/// pixel below the profile's limit): the largest difference and the number of pixels compared.
fn dng_against_picture(raw: &FullImage, picture: &FullImage, ours: bool) -> (f64, usize) {
    assert_eq!((raw.width, raw.height), (picture.width, picture.height));
    let shown = the_dng_through_the_rendering(raw, ours);
    let rendering = CineRendering::new(picture_target(ours));
    let limit = DNG_SCENE_LIMIT;
    let (mut worst, mut n) = (0.0_f64, 0usize);
    for (i, (p, raw_px)) in picture
        .rgb
        .chunks_exact(3)
        .zip(raw.rgb.chunks_exact(3))
        .enumerate()
    {
        let in_range = p.iter().all(|&c| (2e-6..=0.9).contains(&c))
            && raw_px.iter().all(|&c| f64::from(c) < limit * 0.99);
        if !in_range {
            continue;
        }
        // A display colour the rendering cannot make is replaced by the nearest one it can (`docs/02`, Inverse).
        if !ours
            && rendering
                .inverse([p[0], p[1], p[2]].map(f64::from))
                .chroma_fraction
                < 1.0
        {
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
fn for_every_look_the_dngs_pixels_through_the_rendering_are_the_apps_picture() {
    // The photo's R8 is set to 0, so the app's picture has none either (a DNG never has any).
    let src = with_sharpening(&source_of(&detailed_scene()), 0.0, 1.0);
    let r = cpu_renderer(Arc::new(Looks));
    // The last field: the picture is our own rendering, not the inverse of a display rendering that is not ours.
    for (name, look, intensity, ours) in [
        ("neutral", "builtin:neutral", 100.0, true),
        ("a log-to-log look", "user:warm", 100.0, true),
        ("a log-to-log look at 40 %", "user:warm", 40.0, true),
        ("LogC3 to Rec.709", "user:rec709", 100.0, false),
        ("LogC3 to Rec.709 at 60 %", "user:rec709", 60.0, false),
        ("LogC3 to Rec.709 at 0 %", "user:rec709", 0.0, true),
    ] {
        let s = with_look(&src, look, intensity);
        let raw = render(&r, &s, FullTarget::DngLinearAwg4);
        let picture = render(&r, &s, FullTarget::DisplayLinear(picture_target(ours)));
        let (worst, n) = dng_against_picture(&raw, &picture, ours);
        eprintln!(
            "{name}: {n} pixels compared, the DNG through the rendering differs by {worst:.2e}"
        );
        assert!(n > raw.width * raw.height / 10, "{name}: {n} pixels");
        // f32 pixels through a transform that is steep in the shadows, and the inverse's residual of 1e-7.
        assert!(worst < 2e-5, "{name}: {worst}");
        // The preview is the picture of the app (sRGB, display-linear), not the DNG's own render.
        let preview = render(&r, &s, FullTarget::DngPreview);
        let srgb = render(&r, &s, FullTarget::DisplayLinear(OutputTarget::Srgb));
        assert_eq!(preview.rgb, srgb.rgb, "{name}: the preview");
    }
}

#[test]
fn only_a_look_with_its_own_display_rendering_changes_the_dng_pixels_through_the_inverse() {
    // The DNG of a log-to-log look is the C11 output decoded from LogC4: nothing in it depends on a
    // display rendering. A look with its own rendering at 0 % is our C12 exactly (the same DNG as
    // Neutral); above 0 % its picture goes into the DNG through the inverse of the cine rendering.
    let src = source_of(&detailed_scene());
    let r = cpu_renderer(Arc::new(Looks));
    let neutral = with_look(&src, "builtin:neutral", 100.0);
    let zero = with_look(&src, "user:rec709", 0.0);
    let neutral_pixels = render(&r, &neutral, FullTarget::DngLinearAwg4);
    assert_eq!(
        neutral_pixels.rgb,
        render(&r, &zero, FullTarget::DngLinearAwg4).rgb,
        "a look with its own rendering at 0 % is Neutral"
    );
    let own = with_look(&src, "user:rec709", 100.0);
    let own_pixels = render(&r, &own, FullTarget::DngLinearAwg4);
    assert!(max_diff(&neutral_pixels, &own_pixels) > 1e-2);
    // Rendering again gives the same pixels: nothing about the render depends on what came before.
    assert_eq!(
        render(&r, &own, FullTarget::DngLinearAwg4).rgb,
        own_pixels.rgb
    );
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
    // The largest AWG4 channel the inverse gives is the limit itself (there is no placement to divide out).
    assert_eq!(DNG_SCENE_LIMIT, 16.0);
}
