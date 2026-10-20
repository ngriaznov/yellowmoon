//! `export_photo`: the one export function of the app and the CLI. What a file holds (pixels,
//! metadata, geometry), that two exports are the same bytes whatever thread, context or process
//! made them, the file name rules, cancel, and that nothing but the export file is written.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use common::tiffmini::{Tiff, xmp_of, xmp_property};
use common::*;
use ym_color::cdl::Cdl;
use ym_color::display::OutputTarget;
use ym_color::encode::{OutputSpace, encode_quantized};
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_core::export::{
    ExportContext, ExportControl, ExportError, ExportParams, ExportPhoto, ExportReport,
    default_destination, export_photo, planned_destination,
};
use ym_core::pipeline::{
    FullRequest, FullTarget, LookProblem, LookProvider, NeutralOnly, Renderer, ResolvedLook,
};
use ym_core::session::{PhotoId, Session};
use ym_core::settings::{Crop, CropAspect, NormRect, Orientation, SettingsPatch};
use ym_export::naming::Placement;
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists, Size, TiffCompression};
use ym_gpu::cpu::output_sharpen::OutputSharpen;
use ym_gpu::queue::CancelToken;
use ym_looks::library::LookId;
use ym_raw::testgen::{ArwBuilder, GpsSpec};

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
const W: u32 = 600;
const H: u32 = 400;

fn scene() -> Scene {
    sized_scene(W, H)
}

fn sized_scene(w: u32, h: u32) -> Scene {
    Scene::new(w, h, light(PROFILE, 5500.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-5)
}

/// A synthetic photo with a capture time, a lens, exposure data and (optionally) a position.
fn write_photo(dir: &Path, name: &str, tweak: impl FnOnce(ArwBuilder) -> ArwBuilder) -> PathBuf {
    write_sized(dir, name, &scene(), tweak)
}

fn write_sized(
    dir: &Path,
    name: &str,
    scene: &Scene,
    tweak: impl FnOnce(ArwBuilder) -> ArwBuilder,
) -> PathBuf {
    let builder = scene
        .builder(MODEL)
        .capture_time("2026:10:04 12:03:44", "07", Some("+02:00"))
        .lens_model(Some("FE 24-70mm F2.8 GM II"))
        .f_number(28, 10)
        .focal_length(35, 1)
        .exposure_time(1, 250);
    let path = dir.join(name);
    tweak(builder)
        .write_to(&path, &scene.mosaic())
        .expect("write the synthetic ARW");
    path
}

/// A session with one developed photo; `_dir` keeps a temporary source folder alive.
struct Opened {
    session: Session,
    id: PhotoId,
    _dir: Option<tempfile::TempDir>,
}

/// Opens the one ARW in a temporary folder.
fn open_dir(dir: tempfile::TempDir) -> Opened {
    let mut o = open_in_place(dir.path());
    o._dir = Some(dir);
    o
}

/// Opens the one ARW in a folder, in place.
fn open_in_place(dir: &Path) -> Opened {
    let session = Session::new(base_env());
    let out = session.open(
        &[dir.to_path_buf()],
        ym_core::session::OpenOptions::default(),
    );
    assert_eq!(out.added.len(), 1, "one photo added");
    settle(&session);
    Opened {
        id: out.added[0],
        session,
        _dir: None,
    }
}

fn photo_of(d: &Opened) -> ExportPhoto {
    ExportPhoto::from_session(&d.session, d.id).expect("export photo")
}

fn options_in(out: &Path, format: FormatKind) -> ExportOptions {
    ExportOptions {
        format,
        destination: Some(out.to_path_buf()),
        ..ExportOptions::default()
    }
}

fn run(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    options: ExportOptions,
) -> Result<ExportReport, ExportError> {
    export_photo(
        ctx,
        photo,
        &ExportParams::new(options),
        &ExportControl::none(),
    )
}

fn export_ok(ctx: &ExportContext, photo: &ExportPhoto, options: ExportOptions) -> ExportReport {
    run(ctx, photo, options).expect("export")
}

fn tiff16(out: &Path) -> ExportOptions {
    ExportOptions {
        bit_depth: BitDepth::Sixteen,
        tiff_compression: TiffCompression::None,
        ..options_in(out, FormatKind::Tiff)
    }
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read the export")
}

fn edit(session: &Session, id: PhotoId, patch: SettingsPatch) {
    session.update_settings(id, &patch).expect("settings");
}

fn output_sharpening(pct: f64) -> SettingsPatch {
    SettingsPatch {
        output_sharpening: Some(pct),
        ..SettingsPatch::default()
    }
}

fn crop_of(x: f64, y: f64, w: f64, h: f64) -> SettingsPatch {
    SettingsPatch {
        crop: Some(Some(Crop {
            rect: NormRect {
                x,
                y,
                width: w,
                height: h,
            },
            angle_deg: 0.0,
            aspect: CropAspect::Free,
        })),
        ..SettingsPatch::default()
    }
}

// ---------------------------------------------------------------------------------------------
// What a file holds
// ---------------------------------------------------------------------------------------------

#[test]
fn a_16_bit_tiff_is_the_full_render_encoded_without_a_dither() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);

    let report = export_ok(&ctx, &photo, tiff16(out.path()));
    assert_eq!(report.path, out.path().join("DSC00001_neutral.tif"));
    assert_eq!((report.width, report.height), (W, H));
    assert!(report.bytes > 0 && !report.skipped);
    assert!(matches!(report.placement, Placement::Written(_)));

    // The file's samples are exactly the render's display-linear values, sharpened by the Output
    // slider's default (100 %, at the full size: scale 1), through C13.
    let rendered = ctx
        .renderer()
        .render_full(
            &photo.source,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("render");
    assert_eq!(photo.source.settings.output_sharpening, 100.0);
    let sharpened = OutputSharpen::screen()
        .with_amount_pct(100.0)
        .apply_interleaved(rendered.width, rendered.height, &rendered.rgb, 1.0)
        .expect("sharpen");
    assert_ne!(sharpened, rendered.rgb, "the stage changes this patchwork");
    let bytes = read(&report.path);
    let tiff = Tiff::new(&bytes).expect("a TIFF");
    let (w, h, samples) = tiff.uncompressed_u16_rgb(tiff.ifd0());
    assert_eq!((w, h), (W as usize, H as usize));
    assert_eq!((rendered.width, rendered.height), (w, h));
    for (i, px) in sharpened.chunks_exact(3).enumerate() {
        let lin = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
        let want = encode_quantized(lin, OutputSpace::Srgb, 16);
        assert_eq!(&samples[3 * i..3 * i + 3], &want, "pixel {i}");
    }
}

#[test]
fn with_the_output_slider_at_zero_the_file_is_the_render_encoded_as_it_was_before_the_stage() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    edit(&d.session, d.id, output_sharpening(0.0));
    let photo = photo_of(&d);
    let report = export_ok(&ctx, &photo, tiff16(out.path()));
    let rendered = ctx
        .renderer()
        .render_full(
            &photo.source,
            &FullRequest {
                target: FullTarget::DisplayLinear(OutputTarget::Srgb),
            },
            &CancelToken::new(),
        )
        .expect("render");
    let bytes = read(&report.path);
    let tiff = Tiff::new(&bytes).expect("a TIFF");
    let (_, _, samples) = tiff.uncompressed_u16_rgb(tiff.ifd0());
    for (i, px) in rendered.rgb.chunks_exact(3).enumerate() {
        let lin = [f64::from(px[0]), f64::from(px[1]), f64::from(px[2])];
        let want = encode_quantized(lin, OutputSpace::Srgb, 16);
        assert_eq!(&samples[3 * i..3 * i + 3], &want, "pixel {i}");
    }
    // The settings the file records say so.
    let xmp = xmp_of(&bytes).expect("xmp");
    let settings = xmp_property(&xmp, "ym:Settings")
        .expect("settings")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");
    let json: serde_json::Value = serde_json::from_str(&settings).expect("JSON");
    assert_eq!(json["output_sharpening"], serde_json::json!(0.0));
}

#[test]
fn output_sharpening_is_recorded_in_the_xmp_settings() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    edit(&d.session, d.id, output_sharpening(35.0));
    let report = export_ok(&ctx, &photo_of(&d), tiff16(out.path()));
    let bytes = read(&report.path);
    let xmp = xmp_of(&bytes).expect("xmp");
    let settings = xmp_property(&xmp, "ym:Settings")
        .expect("settings")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");
    let json: serde_json::Value = serde_json::from_str(&settings).expect("JSON");
    assert_eq!(json["output_sharpening"], serde_json::json!(35.0));
}

#[test]
fn every_format_and_color_space_writes_a_file_of_the_right_size() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    for (format, ext) in [
        (FormatKind::Jpeg, "jpg"),
        (FormatKind::Tiff, "tif"),
        (FormatKind::Png, "png"),
        (FormatKind::Dng, "dng"),
    ] {
        for space in ColorSpace::ALL {
            if format == FormatKind::Dng && space != ColorSpace::Srgb {
                continue;
            }
            let sub = out.path().join(format!("{ext}-{space:?}"));
            let options = ExportOptions {
                color_space: space,
                ..options_in(&sub, format)
            };
            let report = export_ok(&ctx, &photo, options);
            assert_eq!(report.path.extension().and_then(|e| e.to_str()), Some(ext));
            assert_eq!((report.width, report.height), (W, H), "{ext} {space:?}");
            let bytes = read(&report.path);
            assert_eq!(bytes.len() as u64, report.bytes);
            match format {
                FormatKind::Jpeg => {
                    let mut dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&bytes));
                    let pixels = dec.decode().expect("a JPEG");
                    let info = dec.info().expect("info");
                    assert_eq!(
                        (usize::from(info.width), usize::from(info.height)),
                        (600, 400)
                    );
                    assert_eq!(pixels.len(), 600 * 400 * 3);
                }
                FormatKind::Png => {
                    let mut reader = png::Decoder::new(std::io::Cursor::new(&bytes))
                        .read_info()
                        .expect("a PNG");
                    let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
                    let info = reader.next_frame(&mut buf).expect("frame");
                    assert_eq!((info.width, info.height), (W, H));
                    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
                }
                FormatKind::Tiff | FormatKind::Dng => {
                    assert!(Tiff::new(&bytes).is_some(), "{ext}");
                }
            }
        }
    }
}

#[test]
fn the_xmp_records_how_the_file_was_made() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    // A tweak the XMP must say: a manual exposure and a saturation.
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            saturation: Some(ym_core::settings::PercentSetting::manual_value(15.0)),
            ..SettingsPatch::exposure_ev(0.5)
        },
    );
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path()).with_app_version("9.8.7");
    let photo = photo_of(&d);
    for format in [
        FormatKind::Jpeg,
        FormatKind::Tiff,
        FormatKind::Png,
        FormatKind::Dng,
    ] {
        let report = export_ok(
            &ctx,
            &photo,
            options_in(&out.path().join(format!("{format:?}")), format),
        );
        let bytes = read(&report.path);
        let xmp = xmp_of(&bytes).unwrap_or_else(|| panic!("an XMP packet in {format:?}"));
        assert_eq!(
            xmp_property(&xmp, "ym:PipelineVersion").as_deref(),
            Some(ym_core::version::PIPELINE_VERSION)
        );
        assert_eq!(xmp_property(&xmp, "ym:Profile").as_deref(), Some(PROFILE));
        assert_eq!(
            xmp_property(&xmp, "ym:ProfileVersion").as_deref(),
            Some(photo.source.profile.version.to_string().as_str())
        );
        assert_eq!(
            xmp_property(&xmp, "ym:ProfileMatch").as_deref(),
            Some("exact")
        );
        assert_eq!(
            xmp_property(&xmp, "ym:Look").as_deref(),
            Some("builtin:neutral")
        );
        assert_eq!(
            xmp_property(&xmp, "ym:LookIntensity").as_deref(),
            Some("100")
        );
        // The neutral-only provider has no file hash for any look.
        assert_eq!(xmp_property(&xmp, "ym:LookSHA256"), None);
        assert_eq!(
            xmp_property(&xmp, "ym:DisplayTransform").as_deref(),
            Some("aces2-sdr100-rec709lim")
        );
        assert_eq!(xmp_property(&xmp, "ym:Models"), None, "no network ran");
        // The capture time, never "now".
        assert!(xmp.contains("2026-10-04T12:03:44"), "{format:?}: {xmp}");
        assert!(xmp.contains("Yellowmoon 9.8.7"));
        // The settings are the pinned ones: the manual exposure is there, Auto exposure would
        // have been written as the number it resolved to.
        let json = xmp_property(&xmp, "ym:Settings").expect("settings");
        let value: serde_json::Value =
            serde_json::from_str(&json.replace("&quot;", "\"").replace("&amp;", "&"))
                .expect("JSON");
        assert_eq!(
            value,
            serde_json::to_value(photo.resolved.pinned_settings()).expect("json")
        );
        assert_eq!(
            value["saturation"],
            serde_json::json!({ "kind": "manual", "value": 15.0 })
        );
        assert_eq!(value["exposure"]["kind"], "manual");
        // Which Auto made it (docs/19 section 9): the rule, and no Apple Auto property.
        assert_eq!(xmp_property(&xmp, "ym:Auto").as_deref(), Some("rule"));
        assert_eq!(xmp_property(&xmp, "ym:AppleAuto"), None);
    }
}

#[test]
fn the_display_transform_follows_the_color_space() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    for (space, id) in [
        (ColorSpace::Srgb, "aces2-sdr100-rec709lim"),
        (ColorSpace::DisplayP3, "aces2-sdr100-p3lim"),
        (ColorSpace::Rec2020, "aces2-sdr100-p3lim-rec2020"),
    ] {
        let report = export_ok(
            &ctx,
            &photo,
            ExportOptions {
                color_space: space,
                ..options_in(&out.path().join(id), FormatKind::Jpeg)
            },
        );
        let xmp = xmp_of(&read(&report.path)).expect("xmp");
        assert_eq!(
            xmp_property(&xmp, "ym:DisplayTransform").as_deref(),
            Some(id)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Identical bytes
// ---------------------------------------------------------------------------------------------

/// The export a child process makes for `exports_are_byte_identical_across_processes`. Does
/// nothing unless the parent test set the folder.
#[test]
fn export_child_process() {
    let Some(work) = std::env::var_os("YM_EXPORT_CHILD_DIR") else {
        return;
    };
    let work = PathBuf::from(work);
    let d = open_in_place(&work.join("input"));
    let photo = photo_of(&d);
    let home = tempfile::tempdir().expect("home");
    let ctx = export_context(home.path());
    for (name, format) in [("a", FormatKind::Jpeg), ("b", FormatKind::Dng)] {
        let report = run(
            &ctx,
            &photo,
            options_in(&work.join("child").join(name), format),
        )
        .expect("export");
        assert!(report.bytes > 0);
    }
}

#[test]
fn exports_are_byte_identical_across_threads_contexts_and_processes() {
    let work = tempfile::tempdir().expect("work");
    let input_dir = work.path().join("input");
    std::fs::create_dir(&input_dir).expect("input dir");
    write_photo(&input_dir, "DSC00001.ARW", |b| b);

    // Settings that are not the defaults, exported on three threads with three contexts (each with
    // its own render engine) at the same time.
    let d = open_in_place(&input_dir);
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            contrast: Some(ym_core::settings::PercentSetting::manual_value(12.0)),
            ..crop_of(0.1, 0.1, 0.7, 0.6)
        },
    );
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let photo = photo_of(&d);
    let handles: Vec<_> = (0..3)
        .map(|n| {
            let photo = photo.clone();
            let home = home.path().to_path_buf();
            let out = out.path().join(format!("t{n}"));
            std::thread::spawn(move || {
                let ctx = export_context(&home);
                let jpeg = export_ok(&ctx, &photo, options_in(&out.join("jpg"), FormatKind::Jpeg));
                let dng = export_ok(&ctx, &photo, options_in(&out.join("dng"), FormatKind::Dng));
                (read(&jpeg.path), read(&dng.path))
            })
        })
        .collect();
    let results: Vec<(Vec<u8>, Vec<u8>)> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    for (jpeg, dng) in &results {
        assert_eq!(jpeg, &results[0].0, "JPEG bytes differ between threads");
        assert_eq!(dng, &results[0].1, "DNG bytes differ between threads");
    }

    // A second process: this test binary, running `export_child_process` on the same file with
    // the default settings. Compare with an export of the defaults made here.
    let exe = std::env::current_exe().expect("exe");
    let status = std::process::Command::new(exe)
        .args(["--exact", "export_child_process", "--nocapture"])
        .env("YM_EXPORT_CHILD_DIR", work.path())
        .status()
        .expect("spawn the child");
    assert!(status.success());
    let d2 = open_in_place(&input_dir);
    let ctx = export_context(home.path());
    let p2 = photo_of(&d2);
    let here_jpeg = export_ok(
        &ctx,
        &p2,
        options_in(&out.path().join("here-jpg"), FormatKind::Jpeg),
    );
    let here_dng = export_ok(
        &ctx,
        &p2,
        options_in(&out.path().join("here-dng"), FormatKind::Dng),
    );
    let child_jpeg = read(
        &work
            .path()
            .join("child")
            .join("a")
            .join("DSC00001_neutral.jpg"),
    );
    let child_dng = read(
        &work
            .path()
            .join("child")
            .join("b")
            .join("DSC00001_neutral.dng"),
    );
    assert_eq!(
        read(&here_jpeg.path),
        child_jpeg,
        "JPEG bytes differ between processes"
    );
    assert_eq!(
        read(&here_dng.path),
        child_dng,
        "DNG bytes differ between processes"
    );
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

#[test]
fn a_crop_is_a_window_on_the_uncropped_export() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());

    // With the output sharpening off the window is the same pixels everywhere. (The crop origin is
    // rounded to a whole pixel.)
    edit(&d.session, d.id, output_sharpening(0.0));
    let full = export_ok(&ctx, &photo_of(&d), tiff16(&out.path().join("full")));
    edit(&d.session, d.id, crop_of(0.25, 0.125, 0.5, 0.5));
    let cropped = export_ok(&ctx, &photo_of(&d), tiff16(&out.path().join("crop")));
    assert_eq!((cropped.width, cropped.height), (300, 200));
    let window = |full: &ExportReport, cropped: &ExportReport, margin: usize| {
        let (fb, cb) = (read(&full.path), read(&cropped.path));
        let (ft, ct) = (Tiff::new(&fb).expect("t"), Tiff::new(&cb).expect("t"));
        let (_, _, full_px) = ft.uncompressed_u16_rgb(ft.ifd0());
        let (cw, ch, crop_px) = ct.uncompressed_u16_rgb(ct.ifd0());
        let (x0, y0) = (150usize, 50usize);
        let mut same = 0usize;
        let mut differ = 0usize;
        for y in 0..ch {
            for x in 0..cw {
                let a = &crop_px[3 * (y * cw + x)..][..3];
                let b = &full_px[3 * ((y + y0) * W as usize + x + x0)..][..3];
                let inner = x >= margin && y >= margin && x + margin < cw && y + margin < ch;
                if inner {
                    assert_eq!(a, b, "({x},{y})");
                    same += 1;
                } else if a != b {
                    differ += 1;
                }
            }
        }
        (same, differ)
    };
    let (same, differ) = window(&full, &cropped, 0);
    assert_eq!((same, differ), (300 * 200, 0));

    // With the stage on, the crop's own edge is the picture's edge and the stage mirrors there:
    // the window is still the same pixels further from that edge than the stage's receptive field
    // (21 px is its bound for the screen preset), and the rim may differ.
    edit(&d.session, d.id, output_sharpening(100.0));
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            crop: Some(None),
            ..SettingsPatch::default()
        },
    );
    let full = export_ok(&ctx, &photo_of(&d), tiff16(&out.path().join("full-sharp")));
    edit(&d.session, d.id, crop_of(0.25, 0.125, 0.5, 0.5));
    let cropped = export_ok(&ctx, &photo_of(&d), tiff16(&out.path().join("crop-sharp")));
    let (same, differ) = window(&full, &cropped, 21);
    assert_eq!(same, (300 - 42) * (200 - 42));
    assert!(
        differ > 0,
        "the rim of a sharpened crop mirrors at its own edge"
    );
}

#[test]
fn the_cameras_and_the_users_turns_are_baked_into_a_jpeg() {
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    for (exif, turn, want) in [
        (1u16, Orientation::Deg0, (600u32, 400u32)),
        (6, Orientation::Deg0, (400, 600)),
        (8, Orientation::Deg90, (600, 400)),
        (1, Orientation::Deg90, (400, 600)),
        (3, Orientation::Deg180, (600, 400)),
    ] {
        let src = tempfile::tempdir().expect("dir");
        write_photo(src.path(), "DSC00001.ARW", |b| b.orientation(exif));
        let d = open_dir(src);
        edit(
            &d.session,
            d.id,
            SettingsPatch {
                orientation: Some(turn),
                ..SettingsPatch::default()
            },
        );
        let report = export_ok(
            &ctx,
            &photo_of(&d),
            options_in(
                &out.path().join(format!("{exif}-{}", turn.degrees())),
                FormatKind::Jpeg,
            ),
        );
        assert_eq!(
            (report.width, report.height),
            want,
            "exif {exif} turn {}",
            turn.degrees()
        );
        // The EXIF orientation of the file is reset: the pixels are upright.
        let bytes = read(&report.path);
        let at = common::tiffmini::find(&bytes, b"Exif\0\0").expect("an Exif segment") + 6;
        let tiff = Tiff::new(&bytes[at..]).expect("exif TIFF");
        assert_eq!(tiff.uint(tiff.ifd0(), 274), Some(1), "orientation tag");
    }
}

#[test]
fn a_dng_keeps_the_crop_and_the_turns_in_tags_and_the_pixels_unturned() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b.orientation(6));
    let d = open_dir(src);
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            orientation: Some(Orientation::Deg90),
            ..crop_of(0.0, 0.0, 0.5, 0.5)
        },
    );
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let report = export_ok(&ctx, &photo_of(&d), options_in(out.path(), FormatKind::Dng));
    // The raw IFD is the sensor's 600 x 400, uncropped and unturned.
    assert_eq!(report.rendered, (W, H));
    let bytes = read(&report.path);
    let tiff = Tiff::new(&bytes).expect("a DNG");
    let ifd0 = tiff.ifd0();
    // Camera orientation 6 (90 clockwise) plus the user's 90 clockwise: 180, TIFF value 3.
    assert_eq!(tiff.uint(ifd0, 274), Some(3), "Orientation tag");
    let subs = tiff.sub_ifds(ifd0);
    let raw = subs
        .iter()
        .copied()
        .find(|&s| tiff.uint(s, 262) == Some(34892))
        .expect("a LinearRaw IFD");
    assert_eq!(tiff.uint(raw, 256), Some(W));
    assert_eq!(tiff.uint(raw, 257), Some(H));
    // The crop is in the default crop tags, in raw pixels. The user's frame is the top-left
    // quarter of the turned picture; the picture is the sensor turned by 180 degrees in all, so
    // in the sensor's pixels that is the bottom-right quarter.
    let origin = tiff.uints(&tiff.find(raw, 50719).expect("DefaultCropOrigin"));
    let size = tiff.uints(&tiff.find(raw, 50720).expect("DefaultCropSize"));
    assert_eq!(origin, vec![300, 200]);
    assert_eq!(size, vec![300, 200]);
    // The preview is the user's picture, cropped and in the same unturned orientation.
    let preview = subs
        .iter()
        .copied()
        .find(|&s| tiff.uint(s, 262) != Some(34892))
        .expect("a preview IFD");
    assert_eq!(tiff.uint(preview, 256), Some(300));
    assert_eq!(tiff.uint(preview, 257), Some(200));
}

#[test]
fn a_dng_with_a_straighten_angle_bakes_the_angle_and_the_crop_into_the_pixels() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            crop: Some(Some(Crop {
                rect: NormRect {
                    x: 0.1,
                    y: 0.1,
                    width: 0.6,
                    height: 0.6,
                },
                angle_deg: 3.0,
                aspect: CropAspect::Free,
            })),
            ..SettingsPatch::default()
        },
    );
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let report = export_ok(&ctx, &photo_of(&d), options_in(out.path(), FormatKind::Dng));
    assert_eq!(report.rendered, (360, 240));
    let bytes = read(&report.path);
    let tiff = Tiff::new(&bytes).expect("a DNG");
    assert_eq!(tiff.uint(tiff.ifd0(), 274), Some(1), "Orientation tag");
    let raw = tiff
        .sub_ifds(tiff.ifd0())
        .into_iter()
        .find(|&s| tiff.uint(s, 262) == Some(34892))
        .expect("a LinearRaw IFD");
    assert_eq!(
        (tiff.uint(raw, 256), tiff.uint(raw, 257)),
        (Some(360), Some(240))
    );
    // A DNG has no rotated crop: nothing is left to undo.
    let origin = tiff.uints(&tiff.find(raw, 50719).expect("DefaultCropOrigin"));
    let size = tiff.uints(&tiff.find(raw, 50720).expect("DefaultCropSize"));
    assert_eq!(origin, vec![0, 0]);
    assert_eq!(size, vec![360, 240]);
}

#[test]
fn a_long_edge_resizes_after_the_render() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    let small = export_ok(
        &ctx,
        &photo,
        ExportOptions {
            size: Size::LongEdge { px: 300 },
            ..tiff16(&out.path().join("small"))
        },
    );
    assert_eq!((small.width, small.height), (300, 200));
    assert_eq!(small.rendered, (W, H), "the render is at full size");
    // Never upscaled.
    let big = export_ok(
        &ctx,
        &photo,
        ExportOptions {
            size: Size::LongEdge { px: 5000 },
            ..tiff16(&out.path().join("big"))
        },
    );
    assert_eq!((big.width, big.height), (W, H));
}

// ---------------------------------------------------------------------------------------------
// Metadata: EXIF and location
// ---------------------------------------------------------------------------------------------

#[test]
fn exif_comes_from_the_probe_and_location_can_be_left_out() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| {
        b.gps(GpsSpec {
            latitude_deg: 38.7223,
            longitude_deg: -9.1393,
            altitude_m: 12.0,
        })
    });
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    assert!(
        photo.meta.gps.is_some(),
        "the synthetic photo has a position"
    );

    let has_gps = |path: &Path| -> bool {
        let bytes = read(path);
        let at = common::tiffmini::find(&bytes, b"Exif\0\0").expect("exif") + 6;
        let tiff = Tiff::new(&bytes[at..]).expect("exif TIFF");
        tiff.find(tiff.ifd0(), 34853).is_some()
    };
    let with = export_ok(
        &ctx,
        &photo,
        options_in(&out.path().join("with"), FormatKind::Jpeg),
    );
    assert!(has_gps(&with.path));

    let without = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(options_in(&out.path().join("without"), FormatKind::Jpeg))
            .remove_location(true),
        &ExportControl::none(),
    )
    .expect("export");
    assert!(!has_gps(&without.path));

    // The same photo without a position gives the same bytes as the one whose position was removed.
    let src2 = tempfile::tempdir().expect("dir");
    write_photo(src2.path(), "DSC00001.ARW", |b| b);
    let d2 = open_dir(src2);
    let plain = export_ok(
        &ctx,
        &photo_of(&d2),
        options_in(&out.path().join("plain"), FormatKind::Jpeg),
    );
    assert_eq!(read(&plain.path), read(&without.path));

    // The camera, lens and exposure are in the Exif.
    let bytes = read(&with.path);
    for needle in [&b"ILCE-7RM4"[..], b"FE 24-70mm F2.8 GM II", b"SONY"] {
        assert!(
            common::tiffmini::find(&bytes, needle).is_some(),
            "{}",
            String::from_utf8_lossy(needle)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// File names and "if the file exists"
// ---------------------------------------------------------------------------------------------

#[test]
fn the_name_template_and_if_the_file_exists() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    let base = ExportOptions {
        name_template: "{name}_{look}_{date}_{camera}_{seq}".into(),
        ..options_in(out.path(), FormatKind::Jpeg)
    };
    let first = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(base.clone()).seq(7),
        &ExportControl::none(),
    )
    .expect("export");
    let name = first
        .path
        .file_name()
        .expect("name")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        name,
        "DSC00001_neutral_2026-10-04_sony-ilce-7rm4_007.jpg".replace(
            "sony-ilce-7rm4",
            &ym_export::naming::slugify(&photo.camera.matched.body)
        )
    );

    // Keep both (the default) adds a number.
    let second = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(base.clone()).seq(7),
        &ExportControl::none(),
    )
    .expect("export");
    assert_eq!(
        second.path.file_name().expect("n").to_string_lossy(),
        name.replace(".jpg", " 2.jpg")
    );
    assert!(matches!(second.placement, Placement::Written(_)));

    // Replace overwrites the first.
    let before = read(&first.path);
    std::fs::write(&first.path, b"old").expect("scribble");
    let replaced = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(ExportOptions {
            if_exists: IfExists::Replace,
            ..base.clone()
        })
        .seq(7),
        &ExportControl::none(),
    )
    .expect("export");
    assert_eq!(replaced.path, first.path);
    assert!(matches!(replaced.placement, Placement::Replaced(_)));
    assert_eq!(read(&first.path), before);

    // Skip leaves the file alone and does no work.
    std::fs::write(&first.path, b"keep me").expect("scribble");
    let skipped = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(ExportOptions {
            if_exists: IfExists::Skip,
            ..base
        })
        .seq(7),
        &ExportControl::none(),
    )
    .expect("export");
    assert!(skipped.skipped);
    assert_eq!(skipped.path, first.path);
    assert_eq!(skipped.bytes, 0);
    assert_eq!(skipped.timings.render_ms, 0.0);
    assert_eq!(read(&first.path), b"keep me");
    assert_eq!(names_in(out.path()).len(), 2);
}

#[test]
fn the_default_destination_is_next_to_the_source_folder() {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Lisbon");
    std::fs::create_dir(&folder).expect("folder");
    write_photo(&folder, "DSC00001.ARW", |b| b);
    let d = open_in_place(&folder);
    let home = tempfile::tempdir().expect("home");
    let ctx = export_context(home.path());
    let report = export_ok(
        &ctx,
        &photo_of(&d),
        ExportOptions {
            destination: None,
            ..ExportOptions::default()
        },
    );
    assert_eq!(
        report.path,
        root.path()
            .join("Lisbon export")
            .join("DSC00001_neutral.jpg")
    );
    // The original is untouched and nothing was made in the home folder.
    assert_eq!(names_in(&folder), vec!["DSC00001.ARW".to_owned()]);
    assert!(names_in(home.path()).is_empty());
}

/// A folder reached through a symbolic link: how a Mac reaches its temporary folder (`/var` is a
/// link to `/private/var`) and how a photos disk is often linked into `~/Pictures`. `real` holds
/// what is on the disk, `links` the link `via` to it.
#[cfg(unix)]
fn linked_root() -> (tempfile::TempDir, tempfile::TempDir, PathBuf) {
    let real = tempfile::tempdir().expect("real");
    let links = tempfile::tempdir().expect("links");
    let via = links.path().join("via");
    std::os::unix::fs::symlink(real.path(), &via).expect("symlink");
    (real, links, via)
}

#[cfg(unix)]
#[test]
fn the_default_destination_keeps_the_path_the_photo_was_opened_by() {
    // The person opened `via/Lisbon`; `via` is a link. The folder is called `via/Lisbon export`
    // everywhere: in the panel (the session's default), in the plan, in the file's path. It is not
    // `<what via points to>/Lisbon export`, which they would not recognise (on a Mac that is
    // `/private/var/…` for a folder in `/var/…`).
    let (real, _links, via) = linked_root();
    let folder = via.join("Lisbon");
    std::fs::create_dir(&folder).expect("folder");
    write_photo(&folder, "DSC00001.ARW", |b| b);
    let d = open_in_place(&folder);
    let home = tempfile::tempdir().expect("home");
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    let options = ExportOptions {
        destination: None,
        ..ExportOptions::default()
    };
    let expected = via.join("Lisbon export");

    assert_eq!(
        d.session
            .state()
            .default_export_folder
            .as_deref()
            .map(Path::new),
        Some(expected.as_path())
    );
    assert_eq!(
        planned_destination(&ctx, &photo.path, &options),
        Some(expected.clone())
    );
    let report = export_ok(&ctx, &photo, options);
    assert_eq!(report.path, expected.join("DSC00001_neutral.jpg"));
    // It is the same place on the disk, behind the link.
    assert!(
        real.path()
            .join("Lisbon export")
            .join("DSC00001_neutral.jpg")
            .is_file()
    );
}

#[cfg(unix)]
#[test]
fn a_source_folder_that_is_a_link_gets_its_export_folder_next_to_the_link() {
    // `Lisbon` is a link to a folder with another name somewhere else. The export folder is named
    // after what the person sees and sits next to it, as the panel says, not next to the target.
    let root = tempfile::tempdir().expect("root");
    let elsewhere = tempfile::tempdir().expect("elsewhere");
    let target = elsewhere.path().join("2024-05-trip");
    std::fs::create_dir(&target).expect("target");
    write_photo(&target, "DSC00001.ARW", |b| b);
    let folder = root.path().join("Lisbon");
    std::os::unix::fs::symlink(&target, &folder).expect("symlink");
    let d = open_in_place(&folder);
    let home = tempfile::tempdir().expect("home");
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    let options = ExportOptions {
        destination: None,
        ..ExportOptions::default()
    };
    let expected = root.path().join("Lisbon export");

    assert_eq!(
        d.session
            .state()
            .default_export_folder
            .as_deref()
            .map(Path::new),
        Some(expected.as_path())
    );
    assert_eq!(
        planned_destination(&ctx, &photo.path, &options),
        Some(expected.clone())
    );
    let report = export_ok(&ctx, &photo, options);
    assert_eq!(report.path, expected.join("DSC00001_neutral.jpg"));
    assert!(!elsewhere.path().join("2024-05-trip export").exists());
    // The original, behind the link, is untouched.
    assert_eq!(names_in(&target), vec!["DSC00001.ARW".to_owned()]);
}

#[test]
fn the_default_destination_is_named_after_the_folder_the_path_gives() {
    // Nothing here needs the disk: the path says it.
    assert_eq!(
        default_destination(Path::new("/Volumes/A1/Lisbon/DSC04204.ARW")),
        Some(PathBuf::from("/Volumes/A1/Lisbon export"))
    );
    // `.` components are dropped; the name of the folder is not looked up.
    assert_eq!(
        default_destination(Path::new("/Volumes/./A1/Lisbon/./DSC04204.ARW")),
        Some(PathBuf::from("/Volumes/A1/Lisbon export"))
    );
    // A file at the root of a volume has no folder to sit next to.
    assert_eq!(default_destination(Path::new("/DSC04204.ARW")), None);
    assert_eq!(default_destination(Path::new("")), None);
}

#[cfg(unix)]
#[test]
fn the_default_destination_does_not_mangle_a_folder_name_that_is_not_utf_8() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    // Only the path is looked at, so this holds on every unix, whatever its file system allows.
    let photo = PathBuf::from(OsStr::from_bytes(b"/Volumes/A1/Li\xffsbon/DSC04204.ARW"));
    let expected = PathBuf::from(OsStr::from_bytes(b"/Volumes/A1/Li\xffsbon export"));
    assert_eq!(default_destination(&photo), Some(expected));
}

#[test]
fn a_folder_without_a_name_in_the_path_takes_the_name_it_resolves_to() {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Lisbon");
    let inner = folder.join("inner");
    std::fs::create_dir_all(&inner).expect("folders");
    // `inner/..` is `Lisbon`; the path does not say so.
    let by_dots = inner.join("..").join("DSC04204.ARW");
    let canonical = std::fs::canonicalize(&folder).expect("canonical");
    assert_eq!(
        default_destination(&by_dots),
        Some(canonical.parent().expect("parent").join("Lisbon export"))
    );
    // No folder part: the working directory (the tests run in the crate's folder).
    let cwd = std::env::current_dir().expect("cwd");
    let name = cwd
        .file_name()
        .expect("named")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        default_destination(Path::new("DSC04204.ARW")),
        Some(cwd.parent().expect("parent").join(format!("{name} export")))
    );
}

// ---------------------------------------------------------------------------------------------
// Looks the export cannot use
// ---------------------------------------------------------------------------------------------

/// Neutral, a display-referred look (LogC3 to Rec.709) and a look that went missing.
struct TestLooks;

impl LookProvider for TestLooks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        match id.as_str() {
            "builtin:neutral" => NeutralOnly.look(id),
            "user:warm" => Ok(ResolvedLook {
                transform: Arc::new(
                    LookTransform::new(
                        LookInput::LogC4,
                        Some(Cdl::new([1.25, 1.0, 0.8], [0.0; 3], [1.0; 3], 1.0).expect("cdl")),
                        None,
                    )
                    .expect("look"),
                ),
                fingerprint: "warm-v1".to_owned(),
            }),
            "user:rec709" => Ok(ResolvedLook {
                transform: Arc::new(
                    LookTransform::new(
                        LookInput::LogC3ToRec709,
                        None,
                        Some(Lut::from_cube(Lut3d::identity(3).expect("cube"))),
                    )
                    .expect("look"),
                ),
                fingerprint: "rec709-v1".to_owned(),
            }),
            _ => Err(LookProblem(
                "File missing or changed. Import again.".to_owned(),
            )),
        }
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        Err(LookProblem(format!(
            "The display LUT {id} is not in the library."
        )))
    }

    fn look_sha256(&self, id: &LookId) -> Option<String> {
        (id.as_str() == "user:warm").then(|| "ab".repeat(32))
    }
}

fn ctx_with_looks(home: &Path) -> ExportContext {
    let renderer = Renderer::new(
        Arc::new(TestLooks),
        ym_core::pipeline::RenderOptions::default(),
    )
    .expect("render thread");
    ExportContext::new(renderer).with_home(Some(home.to_path_buf()))
}

fn photo_with_look(d: &Opened, look: &str) -> ExportPhoto {
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            look: Some(LookId::parse(look).expect("look id")),
            ..SettingsPatch::default()
        },
    );
    photo_of(d)
}

#[test]
fn the_look_and_its_hash_are_recorded() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = ctx_with_looks(home.path());
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            look_intensity: Some(60.0),
            ..SettingsPatch::default()
        },
    );
    let photo = photo_with_look(&d, "user:warm");
    let report = export_ok(&ctx, &photo, options_in(out.path(), FormatKind::Jpeg));
    assert_eq!(
        report.path.file_name().expect("n").to_string_lossy(),
        "DSC00001_warm.jpg"
    );
    let xmp = xmp_of(&read(&report.path)).expect("xmp");
    assert_eq!(xmp_property(&xmp, "ym:Look").as_deref(), Some("user:warm"));
    assert_eq!(
        xmp_property(&xmp, "ym:LookSHA256").as_deref(),
        Some("ab".repeat(32).as_str())
    );
    assert_eq!(
        xmp_property(&xmp, "ym:LookIntensity").as_deref(),
        Some("60")
    );
}

#[test]
fn a_dng_is_made_for_a_look_with_its_own_display_rendering() {
    // Decision of 2026-10-14: the DNG is allowed, approximately (`docs/09`, "What a DNG carries");
    // `tests/export_dng_display.rs` measures how close it is.
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = ctx_with_looks(home.path());
    let photo = photo_with_look(&d, "user:rec709");
    let dng = run(&ctx, &photo, options_in(out.path(), FormatKind::Dng)).expect("a DNG");
    assert!(dng.dng.is_some(), "the report says what the DNG scaled to");
    assert_eq!(
        names_in(out.path()),
        vec!["DSC00001_rec709.dng".to_owned()],
        "the DNG is the one file made"
    );
    let ok = run(&ctx, &photo, tiff16(out.path()));
    assert!(ok.is_ok(), "{ok:?}");
}

#[test]
fn a_look_that_went_missing_is_an_error_not_a_neutral_export() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = ctx_with_looks(home.path());
    let photo = photo_with_look(&d, "arri:3000:gone");
    let err = run(&ctx, &photo, options_in(out.path(), FormatKind::Jpeg)).expect_err("refused");
    assert!(matches!(err, ExportError::Look(_)), "{err}");
    assert_eq!(err.to_string(), "File missing or changed. Import again.");
    assert_eq!(err.exit_code(), 3);
    assert!(names_in(out.path()).is_empty());
}

// ---------------------------------------------------------------------------------------------
// Cancel
// ---------------------------------------------------------------------------------------------

/// Exports `photo` and cancels as soon as the photo's progress reaches `at`; returns the result.
fn export_cancelled_at(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    options: ExportOptions,
    at: f32,
) -> (Result<ExportReport, ExportError>, bool) {
    let token = CancelToken::new();
    let reached = AtomicBool::new(false);
    let progress = |f: f32| {
        if f >= at {
            reached.store(true, Ordering::SeqCst);
            token.cancel();
        }
    };
    let result = export_photo(
        ctx,
        photo,
        &ExportParams::new(options),
        &ExportControl {
            cancel: token.clone(),
            progress: Some(&progress),
        },
    );
    (result, reached.load(Ordering::SeqCst))
}

#[test]
fn cancelling_before_the_start_does_nothing_and_leaves_no_file() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let token = CancelToken::new();
    token.cancel();
    let err = export_photo(
        &ctx,
        &photo_of(&d),
        &ExportParams::new(options_in(out.path(), FormatKind::Jpeg)),
        &ExportControl {
            cancel: token,
            progress: None,
        },
    )
    .expect_err("cancelled");
    assert!(err.is_cancelled(), "{err}");
    assert!(names_in(out.path()).is_empty());
}

#[test]
fn cancelling_between_render_tiles_stops_the_render() {
    // 1100 x 700 is two tiles wide at the default 1024 px: the first tile's progress is the
    // cancel point, the second tile is never rendered.
    let src = tempfile::tempdir().expect("dir");
    write_sized(src.path(), "DSC00001.ARW", &sized_scene(1100, 700), |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    let (result, reached) =
        export_cancelled_at(&ctx, &photo, options_in(out.path(), FormatKind::Jpeg), 0.3);
    assert!(reached, "the first tile reported its progress");
    let err = result.expect_err("cancelled");
    assert!(err.is_cancelled(), "{err}");
    assert!(
        names_in(out.path()).is_empty(),
        "no file, no temporary file"
    );
    // A full export of the same photo after the cancel is fine (the render engine is not stuck).
    assert!(run(&ctx, &photo, options_in(out.path(), FormatKind::Jpeg)).is_ok());
}

#[test]
fn cancelling_while_encoding_deletes_the_partial_file() {
    // 2000 x 1400 is three bands of about a million pixels for the writer: cancelling after the
    // render, at the first band, leaves the temporary file to be deleted.
    let src = tempfile::tempdir().expect("dir");
    write_sized(src.path(), "DSC00001.ARW", &sized_scene(2000, 1400), |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    for format in [FormatKind::Jpeg, FormatKind::Tiff] {
        let (result, reached) =
            export_cancelled_at(&ctx, &photo, options_in(out.path(), format), 0.8);
        assert!(reached, "{format:?}: the encoder reported progress");
        let err = result.expect_err("cancelled");
        assert!(err.is_cancelled(), "{format:?}: {err}");
        assert!(
            names_in(out.path()).is_empty(),
            "{format:?}: no partial file or temporary file is left"
        );
    }
}

#[test]
fn cancelling_a_dng_between_its_two_renders_leaves_no_file() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let (result, reached) = export_cancelled_at(
        &ctx,
        &photo_of(&d),
        options_in(out.path(), FormatKind::Dng),
        0.3,
    );
    assert!(reached);
    assert!(result.expect_err("cancelled").is_cancelled());
    assert!(names_in(out.path()).is_empty());
}

#[test]
fn progress_reaches_one_and_never_goes_backwards_by_much() {
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| b);
    let d = open_dir(src);
    let (home, out) = (
        tempfile::tempdir().expect("h"),
        tempfile::tempdir().expect("o"),
    );
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    for format in [FormatKind::Jpeg, FormatKind::Dng] {
        let seen = std::sync::Mutex::new(Vec::<f32>::new());
        let progress = |f: f32| seen.lock().expect("lock").push(f);
        export_photo(
            &ctx,
            &photo,
            &ExportParams::new(options_in(&out.path().join(format!("{format:?}")), format)),
            &ExportControl {
                cancel: CancelToken::new(),
                progress: Some(&progress),
            },
        )
        .expect("export");
        let seen = seen.into_inner().expect("lock");
        assert!(!seen.is_empty());
        assert!(
            (seen.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-6,
            "{format:?}: {seen:?}"
        );
        assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)));
    }
}

// ---------------------------------------------------------------------------------------------
// Nothing else is written
// ---------------------------------------------------------------------------------------------

/// Every file under `root` with its size, sorted.
fn tree(root: &Path) -> Vec<(String, u64)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let rel = p
                    .strip_prefix(root)
                    .expect("under root")
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.sort();
    v
}

#[test]
fn an_export_writes_the_export_file_and_nothing_else() {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Shoot");
    std::fs::create_dir(&folder).expect("folder");
    write_photo(&folder, "DSC00001.ARW", |b| b);
    let original = read(&folder.join("DSC00001.ARW"));
    let home = tempfile::tempdir().expect("home");
    let out = tempfile::tempdir().expect("out");
    let before_root = tree(root.path());
    let before_home = tree(home.path());

    let d = open_in_place(&folder);
    edit(&d.session, d.id, SettingsPatch::exposure_ev(0.3));
    let ctx = export_context(home.path());
    let report = export_ok(
        &ctx,
        &photo_of(&d),
        options_in(out.path(), FormatKind::Tiff),
    );

    assert_eq!(
        tree(root.path()),
        before_root,
        "the source folder is untouched"
    );
    assert_eq!(
        read(&folder.join("DSC00001.ARW")),
        original,
        "the original is untouched"
    );
    assert_eq!(tree(home.path()), before_home, "nothing in the home folder");
    assert_eq!(tree(out.path()).len(), 1);
    assert_eq!(
        names_in(out.path()),
        vec![
            report
                .path
                .file_name()
                .expect("n")
                .to_string_lossy()
                .into_owned()
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// For external tools
// ---------------------------------------------------------------------------------------------

/// Writes one rotated and cropped photo in every format into `$YM_EXPORT_SAMPLES`, for exiftool,
/// LibRaw and ImageMagick to read (`tools/dng/check_dng.py`). Not a test of anything by itself.
#[test]
#[ignore = "writes files for external tools; set YM_EXPORT_SAMPLES to a folder"]
fn samples_for_external_tools() {
    let Some(out) = std::env::var_os("YM_EXPORT_SAMPLES") else {
        eprintln!("YM_EXPORT_SAMPLES is not set");
        return;
    };
    let out = PathBuf::from(out);
    let src = tempfile::tempdir().expect("dir");
    write_photo(src.path(), "DSC00001.ARW", |b| {
        b.orientation(6).gps(GpsSpec {
            latitude_deg: 38.7223,
            longitude_deg: -9.1393,
            altitude_m: 12.0,
        })
    });
    let d = open_dir(src);
    edit(
        &d.session,
        d.id,
        SettingsPatch {
            orientation: Some(Orientation::Deg90),
            ..crop_of(0.0, 0.0, 0.5, 0.5)
        },
    );
    let home = tempfile::tempdir().expect("h");
    let ctx = export_context(home.path());
    let photo = photo_of(&d);
    for format in [
        FormatKind::Jpeg,
        FormatKind::Tiff,
        FormatKind::Png,
        FormatKind::Dng,
    ] {
        let report = export_ok(&ctx, &photo, options_in(&out, format));
        eprintln!("{}", report.path.display());
    }
    for space in [ColorSpace::DisplayP3, ColorSpace::Rec2020] {
        let report = export_ok(
            &ctx,
            &photo,
            ExportOptions {
                color_space: space,
                name_template: format!("{{name}}_{space:?}"),
                ..options_in(&out, FormatKind::Jpeg)
            },
        );
        eprintln!("{}", report.path.display());
    }
}

// ---------------------------------------------------------------------------------------------
// The estimate's speeds, one set per path (CPU reference and export tail on the device)
// ---------------------------------------------------------------------------------------------

mod speed {
    use ym_core::export::{ENCODE_S_PER_MP_PRIOR, ExportSpeed, RENDER_S_PER_MP_PRIOR};
    use ym_core::pipeline::ComputeDevice;
    use ym_export::FormatKind;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn both_paths_start_at_the_measured_numbers_of_the_cpu() {
        let s = ExportSpeed::new();
        assert!(!s.is_measured());
        for device in [ComputeDevice::Cpu, ComputeDevice::Gpu] {
            assert!(
                close(s.render_s_per_mp(device), RENDER_S_PER_MP_PRIOR),
                "{device:?}"
            );
            assert!(close(
                s.encode_s_per_mp(device, FormatKind::Jpeg),
                ENCODE_S_PER_MP_PRIOR[0]
            ));
            // 24 MP rendered and written at full size.
            let expect = 24.0 * RENDER_S_PER_MP_PRIOR + 24.0 * ENCODE_S_PER_MP_PRIOR[0];
            assert!(
                close(s.seconds_for(device, 24.0, 24.0, FormatKind::Jpeg), expect),
                "{device:?}"
            );
            assert_eq!(s.combined_s_per_mp(device, FormatKind::Jpeg), None);
        }
    }

    #[test]
    fn a_path_learns_only_from_the_photos_it_made() {
        let s = ExportSpeed::new();
        // A photo made on the card: 10 MP in 5 s.
        s.record_render_on(ComputeDevice::Gpu, 10.0, 5.0);
        assert!(s.is_measured());
        assert!(
            close(s.render_s_per_mp(ComputeDevice::Cpu), RENDER_S_PER_MP_PRIOR),
            "the CPU's numbers are untouched by the card's photo"
        );
        assert!(
            close(s.render_s_per_mp(ComputeDevice::Gpu), 0.5),
            "the first sample replaces the prior"
        );
        // The next photo fell back to the CPU: 10 MP in 20 s teaches the CPU path only.
        s.record_render_on(ComputeDevice::Cpu, 10.0, 20.0);
        assert!(
            close(s.render_s_per_mp(ComputeDevice::Gpu), 0.5),
            "the card keeps its own numbers"
        );
        assert!(close(s.render_s_per_mp(ComputeDevice::Cpu), 2.0));
        // Later samples are a running mean that weighs the newest half.
        s.record_render_on(ComputeDevice::Cpu, 10.0, 10.0);
        assert!(close(s.render_s_per_mp(ComputeDevice::Cpu), 1.5));
        // Too small a sample says nothing.
        s.record_render_on(ComputeDevice::Cpu, 0.01, 100.0);
        assert!(close(s.render_s_per_mp(ComputeDevice::Cpu), 1.5));
    }

    #[test]
    fn a_photo_the_writer_pulled_while_the_card_made_it_costs_no_encode_on_top() {
        let s = ExportSpeed::new();
        let gpu = ComputeDevice::Gpu;
        // 24 MP written as a JPEG in 6 s in all, render and writing together.
        s.record_overlapped(FormatKind::Jpeg, 24.0, 6.0);
        assert!(s.is_measured());
        assert!(close(s.seconds_for(gpu, 24.0, 24.0, FormatKind::Jpeg), 6.0));
        // The same photo at half the size out: the time is the render's, the writing is inside it.
        assert!(close(s.seconds_for(gpu, 24.0, 6.0, FormatKind::Jpeg), 6.0));
        // The CPU path still adds its encode.
        let cpu = 24.0 * RENDER_S_PER_MP_PRIOR + 24.0 * ENCODE_S_PER_MP_PRIOR[0];
        assert!(close(
            s.seconds_for(ComputeDevice::Cpu, 24.0, 24.0, FormatKind::Jpeg),
            cpu
        ));
        // A second overlapped photo is averaged with the first.
        s.record_overlapped(FormatKind::Jpeg, 24.0, 10.0);
        assert!(close(s.seconds_for(gpu, 24.0, 24.0, FormatKind::Jpeg), 8.0));
    }

    /// One overlapped sample is about one format: a JPEG measured on the card says nothing about a
    /// PNG (its deflate is several times slower) or a TIFF, and nothing about a DNG, which does not
    /// overlap at all.
    #[test]
    fn an_overlapped_sample_is_about_its_own_format_only() {
        let s = ExportSpeed::new();
        let gpu = ComputeDevice::Gpu;
        s.record_overlapped(FormatKind::Jpeg, 24.0, 6.0);
        let before = |f: FormatKind| {
            24.0 * RENDER_S_PER_MP_PRIOR
                + 24.0
                    * ENCODE_S_PER_MP_PRIOR[match f {
                        FormatKind::Jpeg => 0,
                        FormatKind::Tiff => 1,
                        FormatKind::Png => 2,
                        FormatKind::Dng => 3,
                    }]
        };
        for f in [FormatKind::Tiff, FormatKind::Png] {
            assert_eq!(s.combined_s_per_mp(gpu, f), None, "{f:?}");
            assert!(
                close(s.seconds_for(gpu, 24.0, 24.0, f), before(f)),
                "{f:?} still adds its own encode to the render"
            );
        }
        // The DNG is two renders and its writing after them, on the card as on the CPU.
        let dng = s.seconds_for(gpu, 24.0, 24.0, FormatKind::Dng);
        assert!(close(
            dng,
            2.0 * 24.0 * RENDER_S_PER_MP_PRIOR + 24.0 * ENCODE_S_PER_MP_PRIOR[3]
        ));
        // A PNG sample then sets the PNG number and leaves the JPEG one alone.
        s.record_overlapped(FormatKind::Png, 24.0, 12.0);
        assert!(close(s.seconds_for(gpu, 24.0, 24.0, FormatKind::Png), 12.0));
        assert!(close(s.seconds_for(gpu, 24.0, 24.0, FormatKind::Jpeg), 6.0));
        // The render speed of the card is not the combined one.
        assert!(close(s.render_s_per_mp(gpu), RENDER_S_PER_MP_PRIOR));
    }

    #[test]
    fn a_dng_counts_two_renders_on_either_path() {
        let s = ExportSpeed::new();
        s.record_render_on(ComputeDevice::Gpu, 10.0, 5.0);
        let gpu = ComputeDevice::Gpu;
        let dng = s.seconds_for(gpu, 10.0, 10.0, FormatKind::Dng);
        let encode = 10.0 * s.encode_s_per_mp(gpu, FormatKind::Dng);
        assert!(close(dng, 2.0 * 10.0 * 0.5 + encode), "{dng}");
    }
}
