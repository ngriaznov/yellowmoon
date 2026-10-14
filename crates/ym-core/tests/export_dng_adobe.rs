//! A DNG of a look with its own display rendering, through Adobe's own renderer (`dng_validate` of the
//! DNG SDK, `tools/dng/build_dng_validate.sh`): how close to the app's picture is what Adobe's pipeline
//! shows with the profile fitted to ACES 2? (`docs/09`, "What a DNG carries".) The look is a LogC3 to
//! Rec.709 look of our own (`common::dng_looks`, a filmic tone curve with a little colour; never an ARRI
//! file). The numbers are measured and printed, and the asserts are only sanity bounds: "close", not a
//! gate (the DNG gate of `docs/09` is the median ΔE00 of 1.5 against the JPEG, `tools/dngfit`).
//!
//! Needs `dng_validate` (`$DNG_VALIDATE` or `testdata/dng_sdk/dng_validate`); without it the test says
//! so and passes, unless `YM_REQUIRE_EXTERNAL_TOOLS` is set.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::dng_looks::Looks;
use common::tiffmini::Tiff;
use common::*;
use ym_color::consts::linalg;
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_core::export::{ExportContext, ExportControl, ExportParams, ExportPhoto, export_photo};
use ym_core::pipeline::{RenderOptions, Renderer};
use ym_core::session::{OpenOptions, Session};
use ym_core::settings::{AutoOr, SettingsPatch, SharpenManualPatch};
use ym_export::{BitDepth, ExportOptions, FormatKind, TiffCompression};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

fn dng_validate_binary() -> Option<PathBuf> {
    let from_env = std::env::var_os("DNG_VALIDATE").map(PathBuf::from);
    let default = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/dng_sdk/dng_validate");
    from_env.into_iter().chain([default]).find(|p| p.is_file())
}

/// Renders a DNG with `dng_validate` to 16-bit sRGB and returns the code values, 0 to 1, interleaved.
fn adobe_render(binary: &Path, dng: &Path, dir: &Path) -> (usize, usize, Vec<f64>) {
    let base = dir.join("adobe");
    let out = std::process::Command::new(binary)
        .args(["-v", "-16", "-cs1", "-tif"])
        .arg(&base)
        .arg(dng)
        .output()
        .expect("run dng_validate");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "dng_validate failed:\n{text}");
    assert!(
        text.contains("Validation complete")
            && !text.contains("Warning")
            && !text.contains("Error"),
        "dng_validate complained:\n{text}"
    );
    let bytes = std::fs::read(base.with_extension("tif")).expect("the render");
    let tiff = Tiff::new(&bytes).expect("a TIFF");
    let (w, h, px) = tiff.uncompressed_u16_rgb(tiff.ifd0());
    (w, h, px.iter().map(|&v| f64::from(v) / 65535.0).collect())
}

/// CIEDE2000 of two sRGB pictures (code values, interleaved), one value per pixel.
fn delta_e(a: &[f64], b: &[f64]) -> Vec<f64> {
    let lab = |c: &[f64]| {
        let linear = [srgb_decode(c[0]), srgb_decode(c[1]), srgb_decode(c[2])];
        xyz_to_lab(
            linalg::mul_vec(&ym_color::gamut::matrices().rec709_to_xyz, linear),
            d65_white_xyz(),
        )
    };
    a.chunks_exact(3)
        .zip(b.chunks_exact(3))
        .map(|(p, q)| delta_e_2000(lab(p), lab(q)))
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn summary(de: &[f64]) -> (f64, f64, f64, f64) {
    let mut v = de.to_vec();
    v.sort_by(f64::total_cmp);
    (
        percentile(&v, 0.5),
        percentile(&v, 0.95),
        percentile(&v, 0.99),
        v[v.len() - 1],
    )
}

fn decode_jpeg(bytes: &[u8]) -> Vec<f64> {
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes));
    d.decode()
        .expect("a JPEG")
        .iter()
        .map(|&v| f64::from(v) / 255.0)
        .collect()
}

#[test]
fn adobes_renderer_shows_a_look_with_its_own_display_rendering_close_to_the_apps_picture() {
    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!("SKIPPED: dng_validate not built and $DNG_VALIDATE not set");
        return;
    };
    let src = tempfile::tempdir().expect("source folder");
    Scene::new(600, 400, light(PROFILE, 5200.0), Layout::Patchwork, 0.45)
        .noise(2e-4, 2e-6)
        .write(src.path(), "DSC00001.ARW", MODEL);
    let session = Session::new(base_env());
    let opened = session.open(&[src.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1);
    settle(&session);
    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(
        Renderer::new(Arc::new(Looks), RenderOptions::default()).expect("render thread"),
    )
    .with_home(Some(home.path().to_path_buf()));

    let mut results = Vec::new();
    for (name, look) in [
        ("Neutral", "builtin:neutral"),
        ("LogC3 to Rec.709", "user:rec709"),
    ] {
        let out = tempfile::tempdir().expect("out");
        let export = |patch: SettingsPatch, format: FormatKind| {
            session
                .update_settings(opened.added[0], &patch)
                .expect("settings");
            let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("photo");
            let options = ExportOptions {
                format,
                destination: Some(out.path().to_path_buf()),
                bit_depth: BitDepth::Sixteen,
                tiff_compression: TiffCompression::None,
                if_exists: ym_export::IfExists::Replace,
                name_template: format!("{{name}}-{format:?}"),
                ..ExportOptions::default()
            };
            let report = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(options),
                &ExportControl::none(),
            )
            .expect("export");
            std::fs::read(report.path).expect("read")
        };
        let look = SettingsPatch {
            look: Some(LookId::parse(look).expect("look id")),
            look_intensity: Some(100.0),
            ..SettingsPatch::default()
        };
        // The DNG and the JPEG as the app writes them (R8 and output sharpening at their defaults) ...
        let dng = export(look.clone(), FormatKind::Dng);
        let jpeg = export(look.clone(), FormatKind::Jpeg);
        // ... and the same picture without any sharpening, which isolates what the DNG model costs.
        let unsharpened = SettingsPatch {
            sharpen: Some(AutoOr::Manual(SharpenManualPatch {
                amount: Some(0.0),
                radius_px: Some(1.0),
            })),
            output_sharpening: Some(0.0),
            ..look
        };
        let tiff = export(unsharpened, FormatKind::Tiff);
        let tiff_ifd = Tiff::new(&tiff).expect("a TIFF");
        let (tw, th, t16) = tiff_ifd.uncompressed_u16_rgb(tiff_ifd.ifd0());
        let t16: Vec<f64> = t16.iter().map(|&v| f64::from(v) / 65535.0).collect();

        let dng_path = out.path().join("photo.dng");
        std::fs::write(&dng_path, &dng).expect("write the DNG");
        let (w, h, adobe) = adobe_render(&binary, &dng_path, out.path());
        assert_eq!((w, h), (tw, th), "{name}: the sizes");
        let against_tiff = summary(&delta_e(&adobe, &t16));
        let against_jpeg = summary(&delta_e(&adobe, &decode_jpeg(&jpeg)));
        eprintln!(
            "{name}: Adobe's render of the DNG against the unsharpened 16-bit TIFF: median {:.3}, p95 {:.3}, p99 {:.3}, max {:.2}; against the JPEG as written: median {:.3}, p95 {:.3}, p99 {:.3}, max {:.2}",
            against_tiff.0,
            against_tiff.1,
            against_tiff.2,
            against_tiff.3,
            against_jpeg.0,
            against_jpeg.1,
            against_jpeg.2,
            against_jpeg.3
        );
        results.push((name, against_tiff, against_jpeg));
    }
    for (name, tiff, jpeg) in results {
        // Sanity bounds, not a gate: "close" is under the DNG gate's 1.5 (against the JPEG, which has the
        // sharpening on) and under 1.0 against the unsharpened picture.
        assert!(tiff.0 < 1.0, "{name}: median against the TIFF {}", tiff.0);
        assert!(jpeg.0 < 1.5, "{name}: median against the JPEG {}", jpeg.0);
    }
}

/// The highlights of a look that reaches display white (found in review). A display colour near white has a scene value
/// above the top of the DNG profile, and scaling that scene value down with its ratios kept turned a slightly tinted
/// highlight into a strong cast in Adobe's render (a pale blue became a saturated one; display (1, 1, 0.95) came back
/// as (1, 0.99, 0.63) in the app's own arithmetic, CIE76 20). The display colour is dimmed instead, so a highlight keeps
/// its tint and loses a few percent of light. Measured on the pixels whose picture is at or above sRGB code 0.9 in some
/// channel (the highlights), Adobe's render of the DNG against the app's unsharpened picture.
#[test]
fn adobes_renderer_keeps_the_tint_of_the_highlights_of_a_look_that_reaches_display_white() {
    let Some(binary) = dng_validate_binary() else {
        assert!(
            std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
            "dng_validate is required: run tools/dng/build_dng_validate.sh"
        );
        eprintln!("SKIPPED: dng_validate not built and $DNG_VALIDATE not set");
        return;
    };
    let src = tempfile::tempdir().expect("source folder");
    // A bright patchwork: its coloured patches are tinted near-whites through the bright look.
    Scene::new(600, 400, light(PROFILE, 5200.0), Layout::Patchwork, 0.9)
        .noise(2e-4, 2e-6)
        .write(src.path(), "DSC00001.ARW", MODEL);
    let session = Session::new(base_env());
    let opened = session.open(&[src.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1);
    settle(&session);
    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(
        Renderer::new(Arc::new(Looks), RenderOptions::default()).expect("render thread"),
    )
    .with_home(Some(home.path().to_path_buf()));
    let out = tempfile::tempdir().expect("out");
    let export = |patch: SettingsPatch, format: FormatKind| {
        session
            .update_settings(opened.added[0], &patch)
            .expect("settings");
        let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("photo");
        let options = ExportOptions {
            format,
            destination: Some(out.path().to_path_buf()),
            bit_depth: BitDepth::Sixteen,
            tiff_compression: TiffCompression::None,
            if_exists: ym_export::IfExists::Replace,
            name_template: format!("{{name}}-{format:?}"),
            ..ExportOptions::default()
        };
        let report = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(options),
            &ExportControl::none(),
        )
        .expect("export");
        std::fs::read(report.path).expect("read")
    };
    let patch = SettingsPatch {
        look: Some(LookId::parse("user:rec709bright").expect("look id")),
        look_intensity: Some(100.0),
        sharpen: Some(AutoOr::Manual(SharpenManualPatch {
            amount: Some(0.0),
            radius_px: Some(1.0),
        })),
        output_sharpening: Some(0.0),
        ..SettingsPatch::default()
    };
    let dng = export(patch.clone(), FormatKind::Dng);
    let tiff = export(patch, FormatKind::Tiff);
    let tiff_ifd = Tiff::new(&tiff).expect("a TIFF");
    let (tw, th, t16) = tiff_ifd.uncompressed_u16_rgb(tiff_ifd.ifd0());
    let picture: Vec<f64> = t16.iter().map(|&v| f64::from(v) / 65535.0).collect();
    let dng_path = out.path().join("photo.dng");
    std::fs::write(&dng_path, &dng).expect("write the DNG");
    let (w, h, adobe) = adobe_render(&binary, &dng_path, out.path());
    assert_eq!((w, h), (tw, th));

    let all = delta_e(&adobe, &picture);
    let highlights: Vec<f64> = picture
        .chunks_exact(3)
        .zip(&all)
        .filter(|(p, _)| p.iter().copied().fold(0.0, f64::max) >= 0.9)
        .map(|(_, &d)| d)
        .collect();
    assert!(
        highlights.len() > w * h / 20,
        "the scene has highlights: {} of {} pixels",
        highlights.len(),
        w * h
    );
    let (a, hl) = (summary(&all), summary(&highlights));
    eprintln!(
        "bright LogC3 to Rec.709 look: Adobe's render of the DNG against the unsharpened TIFF: all {} pixels median {:.3}, p95 {:.3}, p99 {:.3}, max {:.2}; the {} highlight pixels median {:.3}, p95 {:.3}, p99 {:.3}, max {:.2}",
        all.len(),
        a.0,
        a.1,
        a.2,
        a.3,
        highlights.len(),
        hl.0,
        hl.1,
        hl.2,
        hl.3
    );
    // The loss of a white (display 1.0 comes out at 0.9435) and the profile's own fit: measured p95 0.70, p99 0.76, worst
    // 1.00. The first version, which scaled the scene value down, measured p95 3.0, p99 3.3 and worst 3.6 on this scene.
    assert!(hl.0 < 0.5, "highlights: median {}", hl.0);
    assert!(
        hl.1 < 1.5 && hl.2 < 1.5,
        "highlights: p95 {}, p99 {}",
        hl.1,
        hl.2
    );
}
