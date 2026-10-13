//! Real files (`testdata/real/files`, local evaluation only, never committed or copied): every one
//! exports through the core at full resolution. Skipped when the folder is absent. The exports
//! are slow (a 24 MP frame takes tens of seconds in a test build), so the tests are `#[ignore]`d:
//!
//! ```text
//! cargo test -p ym-core --test export_real -- --ignored --nocapture
//! ```
//!
//! The synthetic set (`tools/fetch/synthetic.sh` into `testdata/synthetic/arw`, or `$YM_TESTDATA`)
//! has its own ignored test. A missing set is a skip with a message, and a failure when
//! `YM_REQUIRE_SYNTHETIC` is set (not empty, not `0`), as in every test that reads the set.
mod common;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::tiffmini::{Tiff, xmp_of, xmp_property};
use common::*;
use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::{ExportEvent, ExportJob, ExportSink};
use ym_core::session::{ApplyScope, OpenOptions};
use ym_export::{ExportOptions, FormatKind};

fn files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/real/files");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn core_with_events() -> (
    Core,
    std::sync::mpsc::Receiver<ExportEvent>,
    tempfile::TempDir,
) {
    let app = tempfile::tempdir().expect("app root");
    let (sink, rx) = ExportSink::channel();
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.export_events = sink;
    options.home = Some(app.path().join("home"));
    (Core::open(options).expect("core"), rx, app)
}

fn wait_idle(rx: &std::sync::mpsc::Receiver<ExportEvent>) {
    let deadline = Instant::now() + Duration::from_secs(900);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if rx.recv_timeout(left).expect("the export finished in time") == ExportEvent::Busy(false) {
            return;
        }
    }
}

fn job(format: FormatKind, out: &Path) -> ExportJob {
    ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            format,
            destination: Some(out.to_path_buf()),
            tiff_compression: ym_export::TiffCompression::Zip,
            ..ExportOptions::default()
        },
    }
}

#[test]
#[ignore = "full-resolution exports of real files take minutes; run with --ignored"]
fn every_real_file_exports_a_jpeg_at_the_camera_size() {
    let files = files();
    if files.is_empty() {
        eprintln!("export_real: testdata/real/files is absent, skipped");
        return;
    }
    let (core, rx, _app) = core_with_events();
    let out = tempfile::tempdir().expect("out");
    let added = core.session.open(&files, OpenOptions::default()).added;
    assert!(core.session.wait_idle(Duration::from_secs(300)));
    let est = core.export_estimate(&job(FormatKind::Jpeg, out.path()));
    let started = Instant::now();
    let id = core
        .export_start(&job(FormatKind::Jpeg, out.path()))
        .expect("queued");
    wait_idle(&rx);
    let total = started.elapsed();
    let summary = core.export_summary(id).expect("summary");
    assert_eq!(summary.failed, 0, "{:?}", summary.errors);
    assert_eq!(summary.saved, files.len());

    let mut bytes_total = 0u64;
    for (photo, path) in added.iter().zip(&summary.files) {
        let view = core.session.view(*photo).expect("view");
        let data = std::fs::read(path).expect("read");
        bytes_total += data.len() as u64;
        let mut dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&data));
        let pixels = dec.decode().expect("a JPEG");
        let info = dec.info().expect("info");
        let (w, h) = (usize::from(info.width), usize::from(info.height));
        // The picture is the camera's crop, upright, less the lens correction's auto crop.
        let (mw, mh) = ym_core::auto::meta_image_size(&view.meta);
        assert!(
            w <= mw && w * 100 >= mw * 90,
            "{}: {w} x {h} against {mw} x {mh}",
            view.path.display()
        );
        assert!(
            h <= mh && h * 100 >= mh * 90,
            "{}: {w} x {h} against {mw} x {mh}",
            view.path.display()
        );
        let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / pixels.len() as f64;
        assert!(
            (30.0..230.0).contains(&mean),
            "{}: mean code {mean}",
            view.path.display()
        );
        let xmp = xmp_of(&data).expect("xmp");
        assert_eq!(
            xmp_property(&xmp, "ym:Profile").as_deref(),
            Some(view.camera.matched.profile.to_string().as_str())
        );
        eprintln!(
            "{:<24} {w:>5} x {h:<5} {:>6.1} MB  mean code {mean:>6.1}  profile {} ({})",
            view.path.file_name().expect("name").to_string_lossy(),
            data.len() as f64 / 1e6,
            view.camera.matched.profile,
            xmp_property(&xmp, "ym:ProfileMatch").unwrap_or_default(),
        );
    }
    eprintln!(
        "{} files in {:.1} s; {:.1} MB in all against an estimate of {:.1} MB and {:.1} s",
        files.len(),
        total.as_secs_f64(),
        bytes_total as f64 / 1e6,
        est.bytes as f64 / 1e6,
        est.seconds
    );
}

#[test]
#[ignore = "full-resolution exports of real files take minutes; run with --ignored"]
fn a_real_file_exports_every_format_and_the_exports_repeat_exactly() {
    let files = files();
    let Some(path) = files
        .iter()
        .find(|p| p.to_string_lossy().contains("A300"))
        .or(files.first())
    else {
        eprintln!("export_real: testdata/real/files is absent, skipped");
        return;
    };
    let (core, rx, _app) = core_with_events();
    core.session
        .open(std::slice::from_ref(path), OpenOptions::default());
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    let mut first = Vec::new();
    for format in [
        FormatKind::Jpeg,
        FormatKind::Tiff,
        FormatKind::Png,
        FormatKind::Dng,
    ] {
        let (a, b) = (
            tempfile::tempdir().expect("a"),
            tempfile::tempdir().expect("b"),
        );
        for dir in [&a, &b] {
            let started = Instant::now();
            core.export_start(&job(format, dir.path())).expect("queued");
            wait_idle(&rx);
            eprintln!("{format:?}: {:.1} s", started.elapsed().as_secs_f64());
        }
        let name = names_in(a.path()).pop().expect("a file");
        let (x, y) = (
            std::fs::read(a.path().join(&name)).expect("a"),
            std::fs::read(b.path().join(&name)).expect("b"),
        );
        assert_eq!(x, y, "{format:?}: two exports of the same photo differ");
        first.push((format, name, x));
    }
    // The DNG is a DNG, with the camera's orientation in its tag.
    let (_, _, dng) = first
        .iter()
        .find(|(f, _, _)| *f == FormatKind::Dng)
        .expect("dng");
    let tiff = Tiff::new(dng).expect("a DNG");
    assert!(tiff.find(tiff.ifd0(), 50706).is_some(), "DNGVersion");
    assert!(tiff.sub_ifds(tiff.ifd0()).len() >= 2);
}

/// The detail statistic the size estimate takes from the app's 240 px thumbnails against the one the
/// estimate was calibrated on, the full-size render box-averaged to 240 px (`docs/09` Size estimate:
/// "The app's own 240 px thumbnails were not measured; re-check them when the call is wired").
#[test]
#[ignore = "full-resolution exports of real files take minutes; run with --ignored --nocapture"]
fn the_detail_statistic_of_the_apps_thumbnails_matches_the_full_size_render() {
    use ym_export::DetailStat;
    let files = files();
    if files.is_empty() {
        eprintln!("export_real: testdata/real/files is absent, skipped");
        return;
    }
    let (core, rx, _app) = core_with_events();
    let out = tempfile::tempdir().expect("out");
    let added = core.session.open(&files, OpenOptions::default()).added;
    assert!(core.session.wait_idle(Duration::from_secs(300)));
    // 16-bit sRGB PNGs at full size: the renders the estimate was calibrated on.
    let id = core
        .export_start(&job(FormatKind::Png, out.path()))
        .expect("queued");
    wait_idle(&rx);
    let summary = core.export_summary(id).expect("summary");
    assert_eq!(summary.failed, 0, "{:?}", summary.errors);
    let mut log_ratios = Vec::new();
    for (photo, path) in added.iter().zip(&summary.files) {
        let thumbnail = core
            .renderer
            .thumbnail_detail(*photo)
            .expect("the thumbnail of a developed photo has a statistic");
        let decoder = png::Decoder::new(std::io::Cursor::new(std::fs::read(path).expect("read")));
        let mut reader = decoder.read_info().expect("a PNG");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
        let info = reader.next_frame(&mut buf).expect("frame");
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        let encoded: Vec<f32> = buf[..info.buffer_size()]
            .chunks_exact(2)
            .map(|b| f32::from(u16::from_be_bytes([b[0], b[1]])) / 65535.0)
            .collect();
        let full =
            DetailStat::measure_encoded_f32(info.width, info.height, &encoded).expect("measurable");
        let ratio = thumbnail.gradient() / full.gradient();
        eprintln!(
            "{:<24} thumbnail {:.5}  full-size render {:.5}  ratio {ratio:.3}",
            path.file_name().expect("name").to_string_lossy(),
            thumbnail.gradient(),
            full.gradient()
        );
        assert!(
            (0.75..1.33).contains(&ratio),
            "the thumbnail's statistic is {ratio:.2} times the render's"
        );
        log_ratios.push(ratio.ln());
    }
    let geometric_mean = (log_ratios.iter().sum::<f64>() / log_ratios.len() as f64).exp();
    eprintln!(
        "{} files, geometric mean of the ratios {geometric_mean:.3} (a JPEG estimate moves by that to the power 0.46)",
        log_ratios.len()
    );
}

/// Whether the environment demands the synthetic set: `YM_REQUIRE_SYNTHETIC` set to something other
/// than empty or `0`. A missing set is then a failure; otherwise it is a skip with a message. The
/// same rule in every test that reads the set (`docs/13-testing.md`).
fn synthetic_required() -> bool {
    demand_is_on(std::env::var_os("YM_REQUIRE_SYNTHETIC").as_deref())
}

/// Whether the value of a `YM_REQUIRE_*` variable switches the demand on: set, not empty, not `0`.
fn demand_is_on(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn a_demand_is_on_when_the_variable_is_set_to_something_but_nothing_or_zero() {
    use std::ffi::OsStr;
    assert!(!demand_is_on(None));
    assert!(!demand_is_on(Some(OsStr::new(""))));
    assert!(!demand_is_on(Some(OsStr::new("0"))));
    assert!(demand_is_on(Some(OsStr::new("1"))));
}

/// The folder `tools/fetch/synthetic.sh` writes the ARW files to: `synthetic/arw` inside
/// `testdata` (`$YM_TESTDATA` when it is set and not empty, else `testdata/` of the workspace).
fn synthetic_arw_dir_in(testdata_env: Option<std::ffi::OsString>) -> PathBuf {
    testdata_env
        .filter(|v| !v.is_empty())
        .map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata"),
            PathBuf::from,
        )
        .join("synthetic/arw")
}

/// The ARW files directly inside `dir`, sorted. (`manifest.json` sits there too, and the scenes
/// are in the folder next to it.)
fn arw_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// The synthetic golden set (`tools/fetch/synthetic.sh` into `testdata/synthetic/arw`, local only):
/// one scene per raw variant (uncompressed 12 and 14 bit, packed, cRAW, Lossless L in all its
/// layouts, Lossless M and S) and body.
fn synthetic_files() -> Vec<PathBuf> {
    arw_files_in(&synthetic_arw_dir_in(std::env::var_os("YM_TESTDATA")))
}

#[test]
fn the_synthetic_folder_is_where_the_generator_writes_it() {
    // The location logic, without touching the environment.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/synthetic/arw");
    assert_eq!(synthetic_arw_dir_in(None), root);
    assert_eq!(synthetic_arw_dir_in(Some("".into())), root);
    assert_eq!(
        synthetic_arw_dir_in(Some("/somewhere".into())),
        Path::new("/somewhere/synthetic/arw")
    );
    // Only ARW files directly in the folder count (any case of the extension).
    let dir = tempfile::tempdir().expect("folder");
    for name in ["b.ARW", "a.arw", "manifest.json", "c.txt"] {
        std::fs::write(dir.path().join(name), b"x").expect("write");
    }
    std::fs::create_dir(dir.path().join("sub.ARW")).expect("folder named like a file");
    let found: Vec<String> = arw_files_in(dir.path())
        .iter()
        .map(|p| p.file_name().expect("name").to_string_lossy().into_owned())
        .collect();
    assert_eq!(found, ["a.arw", "b.ARW"]);
}

#[test]
fn the_generated_synthetic_set_is_found() {
    // With the set generated (tools/fetch/synthetic.sh) the helper that the export test uses sees
    // every ARW file of the folder; a stale path would see none and let that test skip itself.
    let dir = synthetic_arw_dir_in(std::env::var_os("YM_TESTDATA"));
    let on_disk = arw_files_in(&dir).len();
    if on_disk == 0 {
        assert!(
            !synthetic_required(),
            "no ARW files in {} and YM_REQUIRE_SYNTHETIC is set; run tools/fetch/synthetic.sh first",
            dir.display()
        );
        eprintln!(
            "export_real: no ARW files in {}, skipped (YM_REQUIRE_SYNTHETIC=1 makes this a failure)",
            dir.display()
        );
        return;
    }
    assert_eq!(synthetic_files().len(), on_disk);
}

#[test]
#[ignore = "exports every file of the synthetic set at full size as JPEG and as DNG; run with --ignored"]
fn every_synthetic_golden_file_exports_as_jpeg_and_dng() {
    let files = synthetic_files();
    if files.is_empty() {
        let dir = synthetic_arw_dir_in(std::env::var_os("YM_TESTDATA"));
        assert!(
            !synthetic_required(),
            "no ARW files in {} and YM_REQUIRE_SYNTHETIC is set; run tools/fetch/synthetic.sh first",
            dir.display()
        );
        eprintln!(
            "export_real: no ARW files in {}, skipped (YM_REQUIRE_SYNTHETIC=1 makes this a failure)",
            dir.display()
        );
        return;
    }
    let (core, rx, _app) = core_with_events();
    let out = tempfile::tempdir().expect("out");
    core.session.open(&files, OpenOptions::default());
    assert!(core.session.wait_idle(Duration::from_secs(600)));
    let state = core.session.state();
    let ready = state
        .photos
        .iter()
        .filter(|p| matches!(p.status, ym_core::session::PhotoStatus::Ready))
        .count();
    eprintln!(
        "{} of {} synthetic files are ready",
        ready,
        state.photos.len()
    );
    let started = Instant::now();
    let jpeg = core
        .export_start(&job(FormatKind::Jpeg, &out.path().join("jpg")))
        .expect("queued");
    wait_idle(&rx);
    let dng = core
        .export_start(&job(FormatKind::Dng, &out.path().join("dng")))
        .expect("queued");
    wait_idle(&rx);
    let (j, d) = (
        core.export_summary(jpeg).expect("j"),
        core.export_summary(dng).expect("d"),
    );
    eprintln!(
        "{} files, JPEG {} saved {} failed, DNG {} saved {} failed, in {:.1} s",
        files.len(),
        j.saved,
        j.failed,
        d.saved,
        d.failed,
        started.elapsed().as_secs_f64()
    );
    assert_eq!(
        (j.failed, d.failed),
        (0, 0),
        "{:?} {:?}",
        j.errors,
        d.errors
    );
    assert_eq!((j.saved, d.saved), (ready, ready));
}
