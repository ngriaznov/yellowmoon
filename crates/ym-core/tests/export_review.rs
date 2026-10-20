//! Independent review of the export queue, the shared export function, the look service and the
//! preferences (`docs/09`, `docs/01` Export, preferences and looks): each test pins a defect the
//! review found (a panic, a look that changes under a render, `looks_changed` that did not fire, a
//! skipped photo's message) or a property it checked end to end (a DNG's crop window, preview and
//! Orientation tag against the baked JPEG for every orientation; the settings in the XMP; bytes
//! that do not depend on the thread count; names; hemispheres). Kept apart from the implementer's
//! tests so that what the review found stays visible.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::*;
use ym_color::look::LookInput;
use ym_color::lut::Lut3d;
use ym_core::app::{Core, CoreOptions};
use ym_core::export::{
    ExportContext, ExportControl, ExportError, ExportParams, ExportPhoto, export_photo,
};
use ym_core::export_queue::{
    ExportEvent, ExportJob, ExportQueue, ExportSink, FileDone, FileFailure,
};
use ym_core::looks::{ImportRequest, LookService};
use ym_core::pipeline::{
    LookProblem, LookProvider, NeutralOnly, RenderOptions, Renderer, ResolvedLook,
};
use ym_core::preferences::{PreferencesPatch, PreferencesStore};
use ym_core::session::{ApplyScope, OpenOptions, PhotoId, Session};
use ym_core::settings::SettingsPatch;
use ym_export::{ExportOptions, FormatKind};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// `n` small photos in a folder `Shoot` of a new temporary root.
fn shoot(n: usize) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Shoot");
    std::fs::create_dir(&folder).expect("folder");
    for k in 0..n {
        let scene = Scene::new(
            300,
            200,
            light(PROFILE, 4800.0 + 400.0 * k as f64),
            Layout::Patchwork,
            0.35,
        )
        .noise(2e-4, 1e-5)
        .seed(k as u64 + 1);
        scene
            .builder(MODEL)
            .capture_time("2026:10:04 12:03:44", "07", Some("+02:00"))
            .write_to(
                &folder.join(format!("DSC{:05}.ARW", k + 1)),
                &scene.mosaic(),
            )
            .expect("write");
    }
    (root, folder)
}

fn write_cube(dir: &Path, name: &str, size: usize) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(
        &path,
        Lut3d::identity(size).expect("cube").to_cube_string(name),
    )
    .expect("write");
    path
}

fn jpeg_job(scope: ApplyScope, out: &Path) -> ExportJob {
    ExportJob {
        scope,
        options: ExportOptions {
            destination: Some(out.to_path_buf()),
            ..ExportOptions::default()
        },
    }
}

struct Rig {
    core: Core,
    app: tempfile::TempDir,
    events: std::sync::mpsc::Receiver<ExportEvent>,
    _home: tempfile::TempDir,
}

fn rig() -> Rig {
    let app = tempfile::tempdir().expect("app root");
    let home = tempfile::tempdir().expect("home");
    let (sink, events) = ExportSink::channel();
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.export_events = sink;
    options.decode_threads = 2;
    options.home = Some(home.path().to_path_buf());
    Rig {
        core: Core::open(options).expect("core"),
        app,
        events,
        _home: home,
    }
}

/// Everything the queue reports until it is idle again.
fn until_idle(rx: &std::sync::mpsc::Receiver<ExportEvent>) -> Vec<ExportEvent> {
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut all = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let e = rx.recv_timeout(left).expect("the queue went idle");
        let done = e == ExportEvent::Busy(false);
        all.push(e);
        if done {
            return all;
        }
    }
}

fn file_events(events: &[ExportEvent]) -> Vec<&FileDone> {
    events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::File(f) => Some(f),
            _ => None,
        })
        .collect()
}

fn open_all(core: &Core, folder: &Path) -> Vec<PhotoId> {
    let out = core
        .session
        .open(&[folder.to_path_buf()], OpenOptions::default());
    assert!(core.session.wait_idle(Duration::from_secs(120)));
    out.added
}

// ---------------------------------------------------------------------------------------------
// A look whose file is gone is not used, however warm the library's cache is
// ---------------------------------------------------------------------------------------------

#[test]
fn a_look_whose_file_is_gone_is_refused_by_an_export_even_after_it_was_used() {
    let r = rig();
    let src = tempfile::tempdir().expect("src");
    let imported = r
        .core
        .looks_import(
            &write_cube(src.path(), "Warm.cube", 5),
            Some(LookInput::LogC4),
            &mut |_| {},
        )
        .expect("import");
    let warm = imported.added[0].id.clone();
    let (_root, folder) = shoot(1);
    open_all(&r.core, &folder);
    r.core
        .apply_look(ApplyScope::All, &warm, 100.0)
        .expect("apply");

    // The first export loads the look (and the library keeps it in memory).
    let first = tempfile::tempdir().expect("out");
    r.core
        .export_start(&jpeg_job(ApplyScope::All, first.path()))
        .expect("queued");
    let events = until_idle(&r.events);
    assert!(file_events(&events)[0].outcome.is_ok());

    // The file goes; the library notices (a reload, a verify, or the Look tab being opened).
    let stored = r.app.path().join("looks").join("user");
    for e in std::fs::read_dir(&stored).expect("dir").flatten() {
        std::fs::remove_file(e.path()).expect("rm");
    }
    assert!(
        r.core
            .looks_list()
            .iter()
            .any(|l| l.id == warm && l.missing)
    );

    let second = tempfile::tempdir().expect("out");
    let started = r
        .core
        .export_start(&jpeg_job(ApplyScope::All, second.path()));
    let wrote = if started.is_ok() {
        let events = until_idle(&r.events);
        let f = file_events(&events)[0];
        match &f.outcome {
            Ok(_) => true,
            Err(FileFailure {
                kind, exit_code, ..
            }) => {
                assert_eq!((*kind, *exit_code), ("look", 3));
                false
            }
        }
    } else {
        false
    };
    assert!(!wrote, "a look whose file is gone must not be exported");
    assert!(
        std::fs::read_dir(second.path())
            .expect("dir")
            .next()
            .is_none(),
        "no file for a look that is gone"
    );
}

// ---------------------------------------------------------------------------------------------
// looks_changed: a default changed by the CLI, a look replaced by a new file
// ---------------------------------------------------------------------------------------------

fn service_in(root: &Path) -> LookService {
    LookService::open(root, &data_root(), Arc::new(PreferencesStore::open(root))).expect("service")
}

fn import(service: &LookService, path: &Path) {
    service
        .import(
            path,
            &ImportRequest {
                input: Some(LookInput::LogC4),
                cancel: None,
            },
            &mut |_| {},
        )
        .expect("import");
}

#[test]
fn a_default_look_chosen_by_the_cli_reaches_the_app_through_a_reload() {
    let root = tempfile::tempdir().expect("root");
    let src = tempfile::tempdir().expect("src");
    let app = service_in(root.path());
    import(&app, &write_cube(src.path(), "Warm.cube", 5));
    let changes = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changes);
    app.on_change(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(app.effective_default().as_str(), "builtin:signature");

    // `ym looks default user:warm` in another process: the preferences file changes.
    let cli = service_in(root.path());
    cli.set_default(&LookId::parse("user:warm").expect("id"))
        .expect("cli default");

    assert!(app.reload(), "the default look changed");
    assert_eq!(changes.load(Ordering::SeqCst), 1);
    assert_eq!(app.effective_default().as_str(), "user:warm");
    assert!(
        app.list()
            .iter()
            .any(|l| l.id.as_str() == "user:warm" && l.is_default)
    );
    assert!(!app.reload(), "and nothing more");
    assert_eq!(changes.load(Ordering::SeqCst), 1);
}

#[test]
fn a_look_replaced_by_another_file_under_the_same_id_fires_looks_changed() {
    let root = tempfile::tempdir().expect("root");
    let src = tempfile::tempdir().expect("src");
    let app = service_in(root.path());
    import(&app, &write_cube(src.path(), "Warm.cube", 5));
    let changes = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changes);
    app.on_change(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    let id = LookId::parse("user:warm").expect("id");
    let before = app.provider().look_sha256(&id).expect("sha");

    // An ARRI package updated by an import in another process replaces a look's file and the
    // hash the manifest records; id, name and the file being there stay as they were.
    let looks = root.path().join("looks");
    let stored = looks.join("user").join("Warm.cube");
    std::fs::copy(write_cube(src.path(), "Other.cube", 9), &stored).expect("replace the file");
    let sha = ym_looks::manifest::hash_file(&stored).expect("hash");
    let manifest_path = looks.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("manifest")).expect("json");
    manifest["looks"][0]["sha256"] = serde_json::Value::String(sha.clone());
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("json"),
    )
    .expect("manifest");

    assert!(app.reload(), "a different file is a change");
    assert_eq!(changes.load(Ordering::SeqCst), 1);
    let after = app.provider().look_sha256(&id).expect("sha");
    assert_ne!(before, after);
    assert_eq!(after, sha);
    assert!(!app.reload());
}

// ---------------------------------------------------------------------------------------------
// A panic while exporting one photo is that photo's error
// ---------------------------------------------------------------------------------------------

/// Neutral for every look; the SHA-256 of one look panics (a bug somewhere below the export).
struct PanickingLooks;

impl LookProvider for PanickingLooks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        NeutralOnly.look(&LookId::builtin("neutral")).map(|mut l| {
            l.fingerprint = format!("{id}");
            l
        })
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }

    fn look_sha256(&self, id: &LookId) -> Option<String> {
        assert_ne!(id.as_str(), "user:boom", "a bug in a look provider");
        None
    }
}

#[test]
fn a_panic_while_exporting_a_photo_fails_that_photo_and_the_job_goes_on() {
    let (_root, folder) = shoot(2);
    let session = Arc::new(Session::new(base_env()));
    let out = session.open(&[folder], OpenOptions::default());
    settle(&session);
    session
        .update_settings(
            out.added[0],
            &SettingsPatch {
                look: Some(LookId::parse("user:boom").expect("id")),
                ..SettingsPatch::default()
            },
        )
        .expect("look");
    let renderer = Renderer::new(Arc::new(PanickingLooks), RenderOptions::default()).expect("r");
    let home = tempfile::tempdir().expect("home");
    let ctx = Arc::new(ExportContext::new(renderer).with_home(Some(home.path().to_path_buf())));
    let (sink, rx) = ExportSink::channel();
    let queue = ExportQueue::new(ctx, sink);
    let dest = tempfile::tempdir().expect("dest");
    queue
        .start(&session, &jpeg_job(ApplyScope::All, dest.path()), false)
        .expect("queued");
    let events = until_idle(&rx);

    let files = file_events(&events);
    assert_eq!(files.len(), 2, "both photos were dealt with: {events:?}");
    let failure = files[0].outcome.as_ref().expect_err("the first failed");
    assert_eq!(failure.exit_code, 1);
    assert!(!failure.message.is_empty());
    assert!(files[1].outcome.is_ok(), "the second photo was exported");
    // The panel is told: the failed file, then the last event with done == total.
    let progress: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::Progress(p) => Some(p),
            _ => None,
        })
        .collect();
    assert!(progress.iter().any(|p| p.error.is_some() && p.done == 1));
    let last = progress.last().expect("progress");
    assert_eq!((last.done, last.total), (2, 2));
    let names = names_in(dest.path());
    assert_eq!(
        names,
        vec!["DSC00002_neutral.jpg".to_owned()],
        "no partial file"
    );
    // The same through the shared function: an error, not an unwinding thread.
    let photo = ExportPhoto::from_session(&session, out.added[0]).expect("photo");
    let err = export_photo(
        queue.context(),
        &photo,
        &ExportParams::new(jpeg_job(ApplyScope::All, dest.path()).options),
        &ExportControl::none(),
    )
    .expect_err("an error");
    assert_eq!(err.exit_code(), 1);
}

// ---------------------------------------------------------------------------------------------
// The look and the display rendering are the ones the file says
// ---------------------------------------------------------------------------------------------

/// A look whose file changes while the photo renders: the first answer is `v1`, later ones `v2`.
struct ChangingLook {
    calls: AtomicUsize,
}

impl LookProvider for ChangingLook {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        NeutralOnly.look(&LookId::builtin("neutral")).map(|mut l| {
            l.fingerprint = format!("{id}-{}", if n == 0 { "v1" } else { "v2" });
            l
        })
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

#[test]
fn a_look_that_changes_while_the_photo_renders_is_not_written_under_the_old_name() {
    let (_root, folder) = shoot(1);
    let session = Session::new(base_env());
    let out = session.open(&[folder], OpenOptions::default());
    settle(&session);
    let renderer = Renderer::new(
        Arc::new(ChangingLook {
            calls: AtomicUsize::new(0),
        }),
        RenderOptions::default(),
    )
    .expect("r");
    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(renderer).with_home(Some(home.path().to_path_buf()));
    let dest = tempfile::tempdir().expect("dest");
    let photo = ExportPhoto::from_session(&session, out.added[0]).expect("photo");
    let err = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(jpeg_job(ApplyScope::All, dest.path()).options),
        &ExportControl::none(),
    )
    .expect_err("the look changed under the render");
    assert!(matches!(err, ExportError::Look(_)), "{err}");
    assert_eq!(err.exit_code(), 3);
    assert!(names_in(dest.path()).is_empty(), "no file");
}

// ---------------------------------------------------------------------------------------------
// Bytes do not depend on the number of threads
// ---------------------------------------------------------------------------------------------

#[test]
fn export_bytes_do_not_depend_on_the_number_of_threads() {
    let (_root, folder) = shoot(1);
    let session = Session::new(base_env());
    let out = session.open(&[folder], OpenOptions::default());
    settle(&session);
    session
        .update_settings(
            out.added[0],
            &SettingsPatch {
                contrast: Some(ym_core::settings::PercentSetting::manual_value(15.0)),
                saturation: Some(ym_core::settings::PercentSetting::manual_value(10.0)),
                ..SettingsPatch::default()
            },
        )
        .expect("settings");
    let photo = ExportPhoto::from_session(&session, out.added[0]).expect("photo");
    let home = tempfile::tempdir().expect("home");
    let dest = tempfile::tempdir().expect("dest");
    let mut results: Vec<Vec<Vec<u8>>> = Vec::new();
    for threads in [1usize, 4] {
        let renderer = Renderer::new(
            Arc::new(NeutralOnly),
            RenderOptions {
                threads,
                ..RenderOptions::default()
            },
        )
        .expect("r");
        let ctx = ExportContext::new(renderer).with_home(Some(home.path().to_path_buf()));
        let mut files = Vec::new();
        for format in [
            FormatKind::Jpeg,
            FormatKind::Tiff,
            FormatKind::Png,
            FormatKind::Dng,
        ] {
            let options = ExportOptions {
                format,
                destination: Some(dest.path().join(format!("{threads}-{format:?}"))),
                ..ExportOptions::default()
            };
            let report = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(options),
                &ExportControl::none(),
            )
            .expect("export");
            files.push(std::fs::read(&report.path).expect("read"));
        }
        results.push(files);
    }
    for (i, format) in ["JPEG", "TIFF", "PNG", "DNG"].iter().enumerate() {
        assert_eq!(
            results[0][i], results[1][i],
            "{format} differs between 1 and 4 render threads"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------------------------

#[test]
fn preferences_changed_by_two_threads_at_once_keep_both_changes() {
    let root = tempfile::tempdir().expect("root");
    let store = Arc::new(PreferencesStore::open(root.path()));
    let handles: Vec<_> = (0..2)
        .map(|k| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for n in 0..20 {
                    let patch = if k == 0 {
                        PreferencesPatch {
                            check_for_updates_automatically: Some(n % 2 == 1),
                            ..PreferencesPatch::default()
                        }
                    } else {
                        PreferencesPatch {
                            watch_downloads_during_import: Some(n % 2 == 0),
                            ..PreferencesPatch::default()
                        }
                    };
                    store.update(&patch).expect("saved");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread");
    }
    let now = PreferencesStore::open(root.path()).get();
    assert!(
        now.check_for_updates_automatically,
        "thread 0 ended on odd 19"
    );
    assert!(
        !now.watch_downloads_during_import,
        "thread 1 ended on odd 19"
    );
}

// ---------------------------------------------------------------------------------------------
// A DNG's crop window, preview and Orientation tag agree with the JPEG for every orientation and turn
// ---------------------------------------------------------------------------------------------

/// The Pearson correlation of two equally long series.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

/// Applies a TIFF `Orientation` value to an image stored as `w` x `h` interleaved RGB: what a
/// viewer shows. Returns the displayed width, height and pixels.
fn displayed(orientation: u32, w: usize, h: usize, rgb: &[u8]) -> (usize, usize, Vec<u8>) {
    let (dw, dh) = if orientation >= 5 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        for x in 0..dw {
            // Where in the stored image this displayed pixel comes from.
            let (sx, sy) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                8 => (w - 1 - y, x),
                _ => (x, y),
            };
            out[3 * (y * dw + x)..][..3].copy_from_slice(&rgb[3 * (sy * w + sx)..][..3]);
        }
    }
    (dw, dh, out)
}

#[test]
fn a_dngs_window_preview_and_orientation_tag_agree_with_the_baked_export_for_every_orientation_and_turn()
 {
    use common::tiffmini::Tiff;
    use ym_core::settings::{AutoOr, Crop, CropAspect, NormRect, Orientation, SharpenManualPatch};

    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(Renderer::neutral_only().expect("renderer"))
        .with_home(Some(home.path().to_path_buf()));
    for exif in 1u16..=8 {
        let folder = tempfile::tempdir().expect("folder");
        let scene =
            Scene::new(600, 400, light(PROFILE, 5200.0), Layout::Patchwork, 0.4).noise(2e-4, 1e-5);
        scene
            .builder(MODEL)
            .orientation(exif)
            .write_to(&folder.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        let session = Session::new(base_env());
        let out = session.open(&[folder.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        for turn in [
            Orientation::Deg0,
            Orientation::Deg90,
            Orientation::Deg180,
            Orientation::Deg270,
        ] {
            session
                .update_settings(
                    out.added[0],
                    &SettingsPatch {
                        // A DNG carries no sharpening (docs/09), so the JPEG it is compared with
                        // has none either: this test is about where the pixels are, not how sharp.
                        sharpen: Some(AutoOr::Manual(SharpenManualPatch {
                            amount: Some(0.0),
                            radius_px: Some(1.0),
                        })),
                        output_sharpening: Some(0.0),
                        orientation: Some(turn),
                        crop: Some(Some(Crop {
                            rect: NormRect {
                                x: 0.1,
                                y: 0.3,
                                width: 0.5,
                                height: 0.4,
                            },
                            angle_deg: 0.0,
                            aspect: CropAspect::Free,
                        })),
                        ..SettingsPatch::default()
                    },
                )
                .expect("settings");
            let photo = ExportPhoto::from_session(&session, out.added[0]).expect("photo");
            let dest = tempfile::tempdir().expect("dest");
            let report = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(ExportOptions {
                    format: FormatKind::Dng,
                    destination: Some(dest.path().to_path_buf()),
                    ..ExportOptions::default()
                }),
                &ExportControl::none(),
            )
            .expect("dng");
            let bytes = std::fs::read(&report.path).expect("read");
            let tiff = Tiff::new(&bytes).expect("a DNG");
            let subs = tiff.sub_ifds(tiff.ifd0());
            let raw_ifd = subs
                .iter()
                .copied()
                .find(|&s| tiff.uint(s, 262) == Some(34892))
                .expect("raw");
            let preview_ifd = subs
                .iter()
                .copied()
                .find(|&s| tiff.uint(s, 262) != Some(34892))
                .expect("preview");
            let (rw, _rh, raw) = tiff.uncompressed_u16_rgb(raw_ifd);
            let origin = tiff.uints(&tiff.find(raw_ifd, 50719).expect("origin"));
            let size = tiff.uints(&tiff.find(raw_ifd, 50720).expect("size"));
            let (ox, oy, cw, ch) = (
                origin[0] as usize,
                origin[1] as usize,
                size[0] as usize,
                size[1] as usize,
            );
            let at = tiff.uints(&tiff.find(preview_ifd, 273).expect("strip"))[0] as usize;
            let len = tiff.uints(&tiff.find(preview_ifd, 279).expect("count"))[0] as usize;
            let mut dec = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&bytes[at..at + len]));
            let rgb = dec.decode().expect("the preview is a JPEG");
            let info = dec.info().expect("info");
            let (pw, ph) = (usize::from(info.width), usize::from(info.height));
            assert_eq!(
                (pw, ph),
                (cw, ch),
                "exif {exif} turn {}: the preview has the size of the window",
                turn.degrees()
            );

            // Green of the raw window (log of scene-linear) against green of the preview, under
            // the four ways the window could be laid over the preview.
            let g_raw = |x: usize, y: usize| {
                f64::from(raw[3 * ((oy + y) * rw + ox + x) + 1])
                    .max(1.0)
                    .ln()
            };
            let g_prev = |x: usize, y: usize| f64::from(rgb[3 * (y * pw + x) + 1]);
            let preview: Vec<f64> = (0..ph)
                .flat_map(|y| (0..pw).map(move |x| (x, y)))
                .map(|(x, y)| g_prev(x, y))
                .collect();
            let laid = |flip_x: bool, flip_y: bool| -> f64 {
                let window: Vec<f64> = (0..ch)
                    .flat_map(|y| (0..cw).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        g_raw(
                            if flip_x { cw - 1 - x } else { x },
                            if flip_y { ch - 1 - y } else { y },
                        )
                    })
                    .collect();
                correlation(&window, &preview)
            };
            let straight = laid(false, false);
            let others = [laid(true, false), laid(false, true), laid(true, true)];
            assert!(
                straight > 0.95,
                "exif {exif} turn {}: the window and the preview agree ({straight:.3})",
                turn.degrees()
            );
            for o in others {
                assert!(
                    straight > o + 0.1,
                    "exif {exif} turn {}: the window is the right way up ({straight:.3} vs {o:.3})",
                    turn.degrees()
                );
            }

            // The JPEG of the same photo has the camera's and the user's turns baked in. A viewer
            // that applies the DNG's Orientation tag to the (unturned) preview must show the same
            // picture.
            let tag = tiff.uint(tiff.ifd0(), 274).expect("Orientation tag");
            let (dw, dh, shown) = displayed(tag, pw, ph, &rgb);
            let jpeg_report = export_photo(
                &ctx,
                &photo,
                &ExportParams::new(ExportOptions {
                    destination: Some(dest.path().to_path_buf()),
                    ..ExportOptions::default()
                }),
                &ExportControl::none(),
            )
            .expect("jpeg");
            let jpeg_bytes = std::fs::read(&jpeg_report.path).expect("read");
            let mut jd = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg_bytes[..]));
            let jrgb = jd.decode().expect("a JPEG");
            let jinfo = jd.info().expect("info");
            assert_eq!(
                (usize::from(jinfo.width), usize::from(jinfo.height)),
                (dw, dh),
                "exif {exif} turn {}: Orientation tag {tag} turns the preview to the size of the JPEG",
                turn.degrees()
            );
            let a: Vec<f64> = shown
                .iter()
                .skip(1)
                .step_by(3)
                .map(|&v| f64::from(v))
                .collect();
            let b: Vec<f64> = jrgb
                .iter()
                .skip(1)
                .step_by(3)
                .map(|&v| f64::from(v))
                .collect();
            let c = correlation(&a, &b);
            assert!(
                c > 0.99,
                "exif {exif} turn {}: the DNG shown with Orientation {tag} is the JPEG ({c:.4})",
                turn.degrees()
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The XMP says enough to make the file again
// ---------------------------------------------------------------------------------------------

#[test]
fn the_settings_in_a_files_xmp_make_the_same_file_again() {
    use common::tiffmini::{xmp_of, xmp_property};
    use ym_core::settings::DevelopSettings;

    let (_root, folder) = shoot(1);
    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(Renderer::neutral_only().expect("renderer"))
        .with_home(Some(home.path().to_path_buf()));
    let export = |settings: Option<DevelopSettings>, out: &Path| -> Vec<u8> {
        let session = Session::new(base_env());
        let opened = session.open(std::slice::from_ref(&folder), OpenOptions::default());
        settle(&session);
        if let Some(s) = settings {
            session
                .replace_settings(opened.added[0], s)
                .expect("settings");
        }
        let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("photo");
        let report = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(jpeg_job(ApplyScope::All, out).options),
            &ExportControl::none(),
        )
        .expect("export");
        std::fs::read(report.path).expect("read")
    };
    let (a, b) = (
        tempfile::tempdir().expect("a"),
        tempfile::tempdir().expect("b"),
    );
    // Automatic development: exposure, sharpening and noise reduction are Auto in the photo.
    let first = export(None, a.path());
    let xmp = xmp_of(&first).expect("xmp");
    let json = xmp_property(&xmp, "ym:Settings")
        .expect("settings")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");
    let (settings, clamped) =
        DevelopSettings::from_json(&json, &LookId::builtin("neutral")).expect("the XMP's own JSON");
    assert!(
        clamped.is_empty(),
        "nothing in the file's settings is out of range: {clamped:?}"
    );
    let second = export(Some(settings), b.path());
    assert_eq!(
        first, second,
        "the settings a file records give the same file again"
    );
}

// ---------------------------------------------------------------------------------------------
// Closing the session
// ---------------------------------------------------------------------------------------------

#[test]
fn photos_opened_after_the_session_was_closed_get_their_thumbnails() {
    use ym_core::session::{EventSink, SessionEvent};

    let app = tempfile::tempdir().expect("app root");
    let (sink, events) = EventSink::channel();
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.session_events = sink;
    options.decode_threads = 2;
    let core = Core::open(options).expect("core");
    let (_root, folder) = shoot(2);
    let thumbnails = |n: usize| -> Vec<PhotoId> {
        let mut ids = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        while ids.len() < n {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if let SessionEvent::ThumbnailReady(t) =
                events.recv_timeout(left).expect("a thumbnail in time")
            {
                ids.push(t.id);
            }
        }
        ids
    };

    let first = open_all(&core, &folder);
    let mut got = thumbnails(2);
    got.sort();
    assert_eq!(got, first);

    // ⌘W, then the same folder again: the render queues are not left cancelled.
    core.close_session();
    let second = open_all(&core, &folder);
    assert!(second.iter().all(|id| !first.contains(id)), "new ids");
    let mut got = thumbnails(2);
    got.sort();
    assert_eq!(got, second);
}

// ---------------------------------------------------------------------------------------------
// Location
// ---------------------------------------------------------------------------------------------

#[test]
fn the_hemispheres_of_a_position_survive_the_export() {
    use common::tiffmini::{Tiff, find};
    use ym_raw::testgen::GpsSpec;

    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(Renderer::neutral_only().expect("renderer"))
        .with_home(Some(home.path().to_path_buf()));
    for (lat, lon, lat_ref, lon_ref) in [
        (38.7223, -9.1393, b'N', b'W'),
        (-33.8688, 151.2093, b'S', b'E'),
    ] {
        let folder = tempfile::tempdir().expect("folder");
        let scene = Scene::new(300, 200, light(PROFILE, 5200.0), Layout::Patchwork, 0.4);
        scene
            .builder(MODEL)
            .gps(GpsSpec {
                latitude_deg: lat,
                longitude_deg: lon,
                altitude_m: 10.0,
            })
            .write_to(&folder.path().join("DSC00001.ARW"), &scene.mosaic())
            .expect("write");
        let session = Session::new(base_env());
        let opened = session.open(&[folder.path().to_path_buf()], OpenOptions::default());
        settle(&session);
        let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("photo");
        let dest = tempfile::tempdir().expect("dest");
        let report = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(jpeg_job(ApplyScope::All, dest.path()).options),
            &ExportControl::none(),
        )
        .expect("export");
        let bytes = std::fs::read(&report.path).expect("read");
        let at = find(&bytes, b"Exif\0\0").expect("exif") + 6;
        let tiff = Tiff::new(&bytes[at..]).expect("exif TIFF");
        let gps = tiff.uint(tiff.ifd0(), 34853).expect("a GPS directory") as usize;
        let text = |tag: u16| tiff.data(&tiff.find(gps, tag).expect("tag"))[0];
        assert_eq!(text(1), lat_ref, "latitude {lat}");
        assert_eq!(text(3), lon_ref, "longitude {lon}");
    }
}

// ---------------------------------------------------------------------------------------------
// A photo that cannot be developed
// ---------------------------------------------------------------------------------------------

#[test]
fn exporting_the_one_photo_that_cannot_be_developed_says_why() {
    let folder = tempfile::tempdir().expect("folder");
    std::fs::write(folder.path().join("BROKEN.ARW"), b"not a raw file").expect("write");
    let session = Arc::new(Session::new(base_env()));
    let opened = session.open(&[folder.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1);
    settle(&session);
    let reason = match &session.state().photos[0].status {
        ym_core::session::PhotoStatus::Skipped { reason } => reason.clone(),
        other => panic!("the file should have been skipped: {other:?}"),
    };

    let home = tempfile::tempdir().expect("home");
    let (sink, _rx) = ExportSink::channel();
    let ctx = Arc::new(
        ExportContext::new(Renderer::neutral_only().expect("renderer"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let queue = ExportQueue::new(ctx, sink);
    let dest = tempfile::tempdir().expect("dest");
    let err = queue
        .start(&session, &jpeg_job(ApplyScope::This, dest.path()), false)
        .expect_err("nothing to export");
    assert_eq!(err.to_string(), reason, "the reason the file was skipped");
    // All and Picked have nothing to export either, and say so.
    let err = queue
        .start(&session, &jpeg_job(ApplyScope::All, dest.path()), false)
        .expect_err("nothing to export");
    assert!(err.to_string().contains("no photos"), "{err}");
    assert!(names_in(dest.path()).is_empty());
}

// ---------------------------------------------------------------------------------------------
// Odd names
// ---------------------------------------------------------------------------------------------

#[test]
fn any_name_template_gives_a_file_inside_the_destination_folder() {
    let (_root, folder) = shoot(1);
    let session = Session::new(base_env());
    let opened = session.open(&[folder], OpenOptions::default());
    settle(&session);
    let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("photo");
    let home = tempfile::tempdir().expect("home");
    let ctx = ExportContext::new(Renderer::neutral_only().expect("renderer"))
        .with_home(Some(home.path().to_path_buf()));
    let long = "x".repeat(1000);
    let templates = [
        "",
        "   ",
        "...",
        ".",
        "..",
        "{unknown}",
        "{name}/../escape",
        "..\\..\\escape",
        "/absolute/{name}",
        "a\u{0}b",
        "{name}\n{look}",
        "CON",
        "{name}{name}{name}{name}",
        "{seq}{seq}",
        "日本語_{camera}",
        long.as_str(),
    ];
    for (n, template) in templates.iter().enumerate() {
        let dest = tempfile::tempdir().expect("dest");
        let outcome = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(ExportOptions {
                name_template: (*template).to_owned(),
                destination: Some(dest.path().to_path_buf()),
                ..ExportOptions::default()
            }),
            &ExportControl::none(),
        );
        match outcome {
            Ok(report) => {
                assert_eq!(
                    report.path.parent(),
                    Some(dest.path()),
                    "template #{n} {template:?} stays in the folder"
                );
                let name = report.path.file_name().expect("name").to_string_lossy();
                assert!(name.len() <= 255, "template #{n}: {} bytes", name.len());
                assert!(
                    !name.starts_with('.') || name.len() > 4,
                    "template #{n}: {name}"
                );
                assert_eq!(names_in(dest.path()).len(), 1, "template #{n}");
            }
            Err(e) => {
                // A refusal is fine, with words, and nothing was written outside the folder.
                assert!(!e.to_string().is_empty(), "template #{n}");
                assert!(names_in(dest.path()).is_empty(), "template #{n}");
            }
        }
        assert!(
            names_in(home.path()).is_empty(),
            "template #{n} {template:?}: nothing in the home folder"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Several threads at once
// ---------------------------------------------------------------------------------------------

#[test]
fn starting_cancelling_and_asking_from_several_threads_never_sticks() {
    let (_root, folder) = shoot(2);
    let session = Arc::new(Session::new(base_env()));
    session.open(&[folder], OpenOptions::default());
    settle(&session);
    let home = tempfile::tempdir().expect("home");
    let ctx = Arc::new(
        ExportContext::new(Renderer::neutral_only().expect("renderer"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let (sink, rx) = ExportSink::channel();
    let queue = Arc::new(ExportQueue::new(ctx, sink));
    let out = tempfile::tempdir().expect("out");

    let workers: Vec<_> = (0..4)
        .map(|t| {
            let (queue, session) = (Arc::clone(&queue), Arc::clone(&session));
            let out = out.path().join(format!("t{t}"));
            std::thread::spawn(move || {
                let mut started = Vec::new();
                for n in 0..4 {
                    let scope = if n % 2 == 0 {
                        ApplyScope::All
                    } else {
                        ApplyScope::This
                    };
                    let id = queue
                        .start(&session, &jpeg_job(scope, &out), false)
                        .expect("queued");
                    started.push(id);
                    let _ = queue.is_busy();
                    let _ = queue.summary(id);
                    if (t + n) % 3 == 0 {
                        queue.cancel(id);
                    }
                    let _ = queue.estimate(&session, &jpeg_job(scope, &out));
                }
                started
            })
        })
        .collect();
    let ids: Vec<u64> = workers
        .into_iter()
        .flat_map(|w| w.join().expect("thread"))
        .collect();
    assert_eq!(ids.len(), 16);
    assert!(
        queue.wait_idle(Duration::from_secs(120)),
        "every job finished or was cancelled"
    );
    assert!(!queue.is_busy());
    // Every job ended one way or the other, and the busy flag went on and off in pairs.
    for id in &ids {
        let s = queue.summary(*id).expect("summary");
        assert!(
            matches!(
                s.state,
                ym_core::export_queue::JobState::Finished
                    | ym_core::export_queue::JobState::Cancelled
            ),
            "job {id}: {:?}",
            s.state
        );
    }
    let mut depth = 0i32;
    // (`Busy(false)` is sent just after the queue reports idle.)
    while let Ok(e) = rx.recv_timeout(Duration::from_millis(500)) {
        match e {
            ExportEvent::Busy(true) => {
                depth += 1;
                assert_eq!(depth, 1, "busy twice in a row");
            }
            ExportEvent::Busy(false) => {
                depth -= 1;
                assert_eq!(depth, 0, "idle twice in a row");
            }
            _ => {}
        }
    }
    assert_eq!(depth, 0);
}

// ---------------------------------------------------------------------------------------------
// A listener that panics
// ---------------------------------------------------------------------------------------------

#[test]
fn a_listener_that_panics_loses_that_event_and_nothing_else() {
    let (_root, folder) = shoot(2);
    let session = Arc::new(Session::new(base_env()));
    session.open(&[folder], OpenOptions::default());
    settle(&session);
    let home = tempfile::tempdir().expect("home");
    let ctx = Arc::new(
        ExportContext::new(Renderer::neutral_only().expect("renderer"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let (tx, rx) = std::sync::mpsc::channel();
    let tx = std::sync::Mutex::new(tx);
    let panicked = std::sync::atomic::AtomicBool::new(false);
    let sink = ExportSink::new(move |e| {
        // The first progress event of the job takes the listener down; the rest are delivered.
        if matches!(e, ExportEvent::Progress(_)) && !panicked.swap(true, Ordering::SeqCst) {
            panic!("the webview is gone");
        }
        let _ = tx.lock().expect("lock").send(e);
    });
    let queue = ExportQueue::new(ctx, sink);
    let dest = tempfile::tempdir().expect("dest");
    queue
        .start(&session, &jpeg_job(ApplyScope::All, dest.path()), false)
        .expect("queued");
    let events = until_idle(&rx);
    assert_eq!(file_events(&events).len(), 2, "both photos were exported");
    assert_eq!(names_in(dest.path()).len(), 2);
    let last = events
        .iter()
        .filter_map(|e| match e {
            ExportEvent::Progress(p) => Some(p),
            _ => None,
        })
        .next_back()
        .expect("progress after the panic");
    assert_eq!(
        (last.done, last.total),
        (2, 2),
        "the job still ends with its last event"
    );
    assert!(!queue.is_busy());
}

// ---------------------------------------------------------------------------------------------
// Remembered folders
// ---------------------------------------------------------------------------------------------

#[test]
fn folders_remembered_by_two_commands_at_once_are_both_kept() {
    let r = rig();
    let (_root, shoot_folder) = shoot(1);
    open_all(&r.core, &shoot_folder);
    let imports = tempfile::tempdir().expect("imports");
    let cube = write_cube(imports.path(), "Same.cube", 5);
    let core = &r.core;
    let last_export = std::sync::Mutex::new(PathBuf::new());
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..8 {
                core.looks_import(&cube, Some(LookInput::LogC4), &mut |_| {})
                    .expect("import");
            }
        });
        scope.spawn(|| {
            for n in 0..8 {
                let dest = imports.path().join(format!("export-{n}"));
                core.export_start(&jpeg_job(ApplyScope::This, &dest))
                    .expect("queued");
                *last_export.lock().expect("lock") = dest;
            }
        });
    });
    let _ = until_idle_or_timeout(&r.events);
    let folders = r.core.preferences_get().last_used_folders;
    assert_eq!(folders.looks_import.as_deref(), Some(imports.path()));
    assert_eq!(
        folders.export.as_deref(),
        Some(last_export.lock().expect("lock").as_path())
    );
}

/// Drains the export events until the queue reports idle, or 30 s pass.
fn until_idle_or_timeout(rx: &std::sync::mpsc::Receiver<ExportEvent>) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok(ExportEvent::Busy(false)) => return true,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
}
