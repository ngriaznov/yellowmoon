//! `Core`: the pieces wired together, one method per IPC command. That a whole open, edit, export
//! cycle writes only the export file (hard rule 4 of `AGENTS.md`), that the preferences reach the
//! session, the renderer and the export, and the shape of the wire types against `src/api/types.ts`.
mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::tiffmini::{Tiff, find, xmp_of, xmp_property};
use common::*;
use serde_json::Value;
use ym_color::look::LookInput;
use ym_color::lut::Lut3d;
use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::{ExportEstimate, ExportEvent, ExportJob, ExportProgress, ExportSink};
use ym_core::looks::{DisplayLutEntry, LookEntry, LooksImportResult};
use ym_core::preferences::{DisplayRendering, Preferences, PreferencesPatch};
use ym_core::session::{ApplyScope, OpenOptions, PhotoId};
use ym_core::settings::SettingsPatch;
use ym_export::{ExportOptions, FormatKind};
use ym_looks::library::LookId;
use ym_raw::testgen::GpsSpec;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

struct Rig {
    core: Core,
    app: tempfile::TempDir,
    home: tempfile::TempDir,
    events: std::sync::mpsc::Receiver<ExportEvent>,
}

fn rig() -> Rig {
    let app = tempfile::tempdir().expect("app root");
    let home = tempfile::tempdir().expect("home");
    let (sink, events) = ExportSink::channel();
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.export_events = sink;
    options.decode_threads = 2;
    options.home = Some(home.path().to_path_buf());
    let core = Core::open(options).expect("core");
    Rig {
        core,
        app,
        home,
        events,
    }
}

/// `n` photos in a new folder `Shoot` inside a temporary root.
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
            .gps(GpsSpec {
                latitude_deg: 38.7,
                longitude_deg: -9.1,
                altitude_m: 10.0,
            })
            .write_to(
                &folder.join(format!("DSC{:05}.ARW", k + 1)),
                &scene.mosaic(),
            )
            .expect("write");
    }
    (root, folder)
}

fn open_all(core: &Core, folder: &Path) -> Vec<PhotoId> {
    let out = core
        .session
        .open(&[folder.to_path_buf()], OpenOptions::default());
    assert!(
        core.session.wait_idle(Duration::from_secs(120)),
        "the photos developed"
    );
    out.added
}

fn wait_export(rig: &Rig) {
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if rig.events.recv_timeout(left).expect("export finished") == ExportEvent::Busy(false) {
            return;
        }
    }
}

/// Every file under `root`, relative, with its size.
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

fn jpeg_job(scope: ApplyScope, destination: Option<&Path>) -> ExportJob {
    ExportJob {
        scope,
        options: ExportOptions {
            destination: destination.map(Path::to_path_buf),
            ..ExportOptions::default()
        },
    }
}

fn write_cube(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(
        &path,
        Lut3d::identity(5).expect("cube").to_cube_string(name),
    )
    .expect("write");
    path
}

// ---------------------------------------------------------------------------------------------
// Nothing else persists
// ---------------------------------------------------------------------------------------------

#[test]
fn opening_the_core_writes_nothing() {
    let r = rig();
    assert!(tree(r.app.path()).is_empty());
    assert!(tree(r.home.path()).is_empty());
    assert_eq!(r.core.preferences_get(), Preferences::default());
}

#[test]
fn a_whole_open_edit_export_cycle_writes_only_the_export_file() {
    let r = rig();
    let (root, folder) = shoot(2);
    let before_source = tree(root.path());
    let originals: Vec<Vec<u8>> = (1..=2)
        .map(|k| std::fs::read(folder.join(format!("DSC{k:05}.ARW"))).expect("read"))
        .collect();

    let ids = open_all(&r.core, &folder);
    r.core
        .session
        .update_settings(
            ids[0],
            &SettingsPatch {
                saturation: Some(ym_core::settings::PercentSetting::manual_value(25.0)),
                ..SettingsPatch::exposure_ev(0.4)
            },
        )
        .expect("edit");
    r.core.session.set_picked(ids[0], true).expect("pick");
    r.core.session.rerun_auto(ids[1]).expect("re-run");
    r.core
        .apply_look(ApplyScope::All, &LookId::builtin("neutral"), 80.0)
        .expect("apply look");
    // The default destination: next to the source folder.
    let job = jpeg_job(ApplyScope::All, None);
    let est = r.core.export_estimate(&job);
    assert!(est.bytes > 0);
    let cli = r.core.cli_preview(&job);
    assert!(cli.starts_with("ym render "), "{cli}");
    let id = r.core.export_start(&job).expect("queued");
    wait_export(&r);
    let summary = r.core.export_summary(id).expect("summary");
    assert_eq!((summary.saved, summary.failed), (2, 0));

    // The only new files: the exports, in `Shoot export` next to the source folder.
    let after_source = tree(root.path());
    let new: Vec<_> = after_source
        .iter()
        .filter(|f| !before_source.contains(f))
        .map(|(n, _)| n.clone())
        .collect();
    assert_eq!(
        new,
        vec![
            "Shoot export/DSC00001_neutral.jpg".to_owned(),
            "Shoot export/DSC00002_neutral.jpg".to_owned()
        ]
    );
    for k in 1..=2 {
        assert_eq!(
            std::fs::read(folder.join(format!("DSC{k:05}.ARW"))).expect("read"),
            originals[k - 1],
            "the original is untouched"
        );
    }
    // Nothing in the app folder (no preferences were changed: the destination was the default)
    // and nothing in the home folder.
    assert!(tree(r.app.path()).is_empty(), "{:?}", tree(r.app.path()));
    assert!(tree(r.home.path()).is_empty(), "{:?}", tree(r.home.path()));
}

#[test]
fn choosing_a_destination_remembers_the_folder_in_the_preferences_and_nothing_else() {
    let r = rig();
    let (_root, folder) = shoot(1);
    open_all(&r.core, &folder);
    let out = tempfile::tempdir().expect("out");
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(out.path())))
        .expect("queued");
    wait_export(&r);
    let files: Vec<String> = tree(r.app.path()).into_iter().map(|(n, _)| n).collect();
    assert_eq!(files, vec!["preferences.json".to_owned()]);
    assert_eq!(
        r.core.preferences_get().last_used_folders.export,
        Some(out.path().to_path_buf())
    );
    // The file is the exhaustive list of docs/01: the keys of Preferences, nothing per photo.
    let json: Value = serde_json::from_str(
        &std::fs::read_to_string(r.app.path().join("preferences.json")).expect("read"),
    )
    .expect("json");
    let keys: BTreeSet<_> = json.as_object().expect("object").keys().cloned().collect();
    let want: BTreeSet<String> = [
        "default_look",
        "display_rendering",
        "export_defaults",
        "remove_location_from_exports",
        "check_for_updates_automatically",
        "watch_downloads_during_import",
        "last_used_folders",
        // "Auto with Apple Intelligence", the eighth (the owner's agreement of 2026-10-19, docs/19).
        "apple_auto",
        // "Instructions for Apple Intelligence", the ninth (the owner's request of 2026-10-19).
        "apple_auto_instructions",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(keys, want);
}

// ---------------------------------------------------------------------------------------------
// Preferences reach the session, the renderer and the export
// ---------------------------------------------------------------------------------------------

#[test]
fn the_default_look_preference_reaches_photos_opened_afterwards() {
    let r = rig();
    let src = tempfile::tempdir().expect("src");
    let imported = r
        .core
        .looks_import(
            &write_cube(src.path(), "Warm.cube"),
            Some(LookInput::LogC4),
            &mut |_| {},
        )
        .expect("import");
    let warm = imported.added[0].id.clone();
    // Remembered: the folder it was imported from.
    assert_eq!(
        r.core.preferences_get().last_used_folders.looks_import,
        Some(src.path().to_path_buf())
    );

    let looks_changed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&looks_changed);
    r.core.on_looks_changed(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    let prefs = r
        .core
        .preferences_set(&PreferencesPatch::default_look(warm.clone()))
        .expect("set");
    assert_eq!(prefs.default_look, warm);
    assert_eq!(looks_changed.load(Ordering::SeqCst), 1);
    assert!(
        r.core
            .looks_list()
            .iter()
            .find(|l| l.id == warm)
            .expect("l")
            .is_default
    );

    let (_root, folder) = shoot(1);
    let ids = open_all(&r.core, &folder);
    assert_eq!(
        r.core.session.settings(ids[0]).expect("s").settings.look,
        warm
    );

    // `looks_set_default` does the same.
    r.core
        .looks_set_default(&LookId::builtin("neutral"))
        .expect("default");
    assert_eq!(looks_changed.load(Ordering::SeqCst), 2);
    let (_root2, folder2) = shoot(1);
    let more = r
        .core
        .session
        .open(&[folder2], OpenOptions::default())
        .added;
    assert!(r.core.session.wait_idle(Duration::from_secs(60)));
    assert_eq!(
        r.core
            .session
            .settings(more[0])
            .expect("s")
            .settings
            .look
            .as_str(),
        "builtin:neutral"
    );
    // A look that does not exist cannot become the default, and the error says where it belongs.
    let err = r
        .core
        .preferences_set(&PreferencesPatch::default_look(
            LookId::parse("user:nope").expect("id"),
        ))
        .expect_err("unknown");
    assert_eq!(err.scope(), "looks");
    assert_eq!(
        err.to_string(),
        "That look isn't in the library. Pick another look."
    );
}

#[test]
fn remove_location_is_taken_from_the_preferences_when_the_job_is_queued() {
    let r = rig();
    let (_root, folder) = shoot(1);
    open_all(&r.core, &folder);
    let has_gps = |path: &Path| -> bool {
        let bytes = std::fs::read(path).expect("read");
        let at = find(&bytes, b"Exif\0\0").expect("exif") + 6;
        let tiff = Tiff::new(&bytes[at..]).expect("exif TIFF");
        tiff.find(tiff.ifd0(), 34853).is_some()
    };
    let (a, b) = (
        tempfile::tempdir().expect("a"),
        tempfile::tempdir().expect("b"),
    );
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(a.path())))
        .expect("queued");
    wait_export(&r);
    assert!(has_gps(&a.path().join("DSC00001_signature.jpg")));

    r.core
        .preferences_set(&PreferencesPatch {
            remove_location_from_exports: Some(true),
            ..PreferencesPatch::default()
        })
        .expect("set");
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(b.path())))
        .expect("queued");
    wait_export(&r);
    assert!(!has_gps(&b.path().join("DSC00001_signature.jpg")));
}

/// The look of a photo nobody has touched is Signature (owner's decision of 2026-10-14); Neutral is
/// still listed, still the identity, and still one click away.
#[test]
fn a_new_session_gets_signature_and_neutral_is_still_the_identity() {
    let r = rig();
    let list = r.core.looks_list();
    assert_eq!(
        list.iter()
            .take(2)
            .map(|l| l.id.as_str())
            .collect::<Vec<_>>(),
        ["builtin:neutral", "builtin:signature"]
    );
    assert_eq!(
        list.iter()
            .filter(|l| l.is_default)
            .map(|l| l.id.as_str())
            .collect::<Vec<_>>(),
        ["builtin:signature"]
    );
    assert_eq!(
        r.core.preferences_get().default_look.as_str(),
        "builtin:signature"
    );

    let (_root, folder) = shoot(1);
    let ids = open_all(&r.core, &folder);
    let opened = r.core.session.settings(ids[0]).expect("settings").settings;
    assert_eq!(opened.look.as_str(), "builtin:signature");
    assert_eq!(opened.look_intensity, 100.0);

    // The export says which look made it, and Signature changes the picture.
    let (signature_dir, neutral_dir) = (
        tempfile::tempdir().expect("a"),
        tempfile::tempdir().expect("b"),
    );
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(signature_dir.path())))
        .expect("queued");
    wait_export(&r);
    r.core
        .apply_look(ApplyScope::All, &LookId::neutral(), 100.0)
        .expect("apply");
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(neutral_dir.path())))
        .expect("queued");
    wait_export(&r);
    let signature = std::fs::read(signature_dir.path().join("DSC00001_signature.jpg"))
        .expect("the Signature export");
    let neutral =
        std::fs::read(neutral_dir.path().join("DSC00001_neutral.jpg")).expect("the Neutral export");
    assert_ne!(signature, neutral, "Signature is not the identity");
    for (bytes, look) in [
        (&signature, "builtin:signature"),
        (&neutral, "builtin:neutral"),
    ] {
        assert_eq!(
            xmp_property(&xmp_of(bytes).expect("xmp"), "ym:Look").as_deref(),
            Some(look)
        );
    }
}

#[test]
fn the_display_rendering_preference_chooses_the_transform_and_is_written_to_the_xmp() {
    let r = rig();
    let src = tempfile::tempdir().expect("src");
    let folder = src.path().join("LogC4 Display LUTs");
    std::fs::create_dir_all(&folder).expect("folder");
    write_cube(&folder, "LogC4_to_Gamma24_Rec709.cube");
    let imported = r
        .core
        .looks_import(&folder, None, &mut |_| {})
        .expect("import display LUTs");
    assert_eq!(imported.display_luts_added, 1);
    let lut = r.core.looks_display_luts()[0].id.clone();

    let (_root, shoot_folder) = shoot(1);
    open_all(&r.core, &shoot_folder);
    let (aces, arri) = (
        tempfile::tempdir().expect("a"),
        tempfile::tempdir().expect("b"),
    );
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(aces.path())))
        .expect("queued");
    wait_export(&r);

    r.core
        .preferences_set(&PreferencesPatch {
            display_rendering: Some(DisplayRendering::ArriDisplay { id: lut.clone() }),
            ..PreferencesPatch::default()
        })
        .expect("set");
    assert_eq!(
        r.core.renderer.display(),
        ym_core::pipeline::DisplayChoice::ArriLut(lut.clone())
    );
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(arri.path())))
        .expect("queued");
    wait_export(&r);
    let name = "DSC00001_signature.jpg";
    let (bytes_aces, bytes_arri) = (
        std::fs::read(aces.path().join(name)).expect("aces"),
        std::fs::read(arri.path().join(name)).expect("arri"),
    );
    assert_ne!(
        bytes_aces, bytes_arri,
        "another display transform, another picture"
    );
    assert_eq!(
        xmp_property(&xmp_of(&bytes_aces).expect("xmp"), "ym:DisplayTransform").as_deref(),
        Some("aces2-sdr100-rec709lim")
    );
    assert_eq!(
        xmp_property(&xmp_of(&bytes_arri).expect("xmp"), "ym:DisplayTransform").as_deref(),
        Some(lut.as_str())
    );
    // Neutral is a built-in look with a description file: its hash is recorded.
    let sha = xmp_property(&xmp_of(&bytes_aces).expect("xmp"), "ym:LookSHA256").expect("sha");
    assert_eq!(sha.len(), 64);

    // The LUT's file vanishes. Once the library notices (a verify, a reload), the export refuses
    // instead of quietly using ACES 2 under a preference that says the LUT.
    let stored = r.app.path().join("looks").join("arri-display");
    for e in std::fs::read_dir(&stored).expect("dir").flatten() {
        std::fs::remove_file(e.path()).expect("rm");
    }
    r.core.looks.verify().expect("verify");
    let c = tempfile::tempdir().expect("c");
    let err = r
        .core
        .export_start(&jpeg_job(ApplyScope::All, Some(c.path())))
        .expect_err("refused");
    assert_eq!(err.scope(), "looks");
    assert_eq!(err.to_string(), "File missing or changed. Import again.");
    assert!(tree(c.path()).is_empty());
    // The viewport meanwhile uses ACES 2, and choosing ACES 2 again makes exports work.
    assert_eq!(
        r.core.renderer.display(),
        ym_core::pipeline::DisplayChoice::Aces2
    );
    r.core
        .preferences_set(&PreferencesPatch {
            display_rendering: Some(DisplayRendering::Aces2),
            ..PreferencesPatch::default()
        })
        .expect("set");
    r.core
        .export_start(&jpeg_job(ApplyScope::All, Some(c.path())))
        .expect("queued");
    wait_export(&r);
    assert_eq!(names_in(c.path()).len(), 1);
}

// ---------------------------------------------------------------------------------------------
// Looks and exports together
// ---------------------------------------------------------------------------------------------

#[test]
fn a_look_with_its_own_display_rendering_makes_a_dng_approximate_and_blocks_nothing() {
    let r = rig();
    let src = tempfile::tempdir().expect("src");
    let imported = r
        .core
        .looks_import(
            &write_cube(src.path(), "Rec709 Baked.cube"),
            Some(LookInput::LogC3ToRec709),
            &mut |_| {},
        )
        .expect("import");
    let baked = imported.added[0].id.clone();
    assert_eq!(imported.added[0].input, LookInput::LogC3ToRec709);

    let (_root, folder) = shoot(2);
    let ids = open_all(&r.core, &folder);
    let dng = ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            format: FormatKind::Dng,
            destination: None,
            ..ExportOptions::default()
        },
    };
    assert!(!r.core.export_estimate(&dng).dng_approximate);
    r.core.session.set_current(ids[1]).expect("current");
    r.core
        .apply_look(ApplyScope::This, &baked, 100.0)
        .expect("apply");
    assert!(
        r.core.export_estimate(&dng).dng_approximate,
        "one photo has the look"
    );
    // Not for the photo that doesn't have it.
    let first_only = ExportJob {
        scope: ApplyScope::Picked,
        ..dng.clone()
    };
    r.core.session.set_picked(ids[0], true).expect("pick");
    assert!(!r.core.export_estimate(&first_only).dng_approximate);
    // Not at an intensity of 0, where the picture is our own rendering.
    r.core
        .apply_look(ApplyScope::This, &baked, 0.0)
        .expect("apply");
    assert!(!r.core.export_estimate(&dng).dng_approximate);
    r.core
        .apply_look(ApplyScope::This, &baked, 100.0)
        .expect("apply");

    // The DNG is queued and made, as the other formats are.
    let dng_out = tempfile::tempdir().expect("out");
    let dng_job = ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            format: FormatKind::Dng,
            destination: Some(dng_out.path().to_path_buf()),
            ..ExportOptions::default()
        },
    };
    let id = r.core.export_start(&dng_job).expect("queued");
    wait_export(&r);
    let s = r.core.export_summary(id).expect("summary");
    assert_eq!((s.saved, s.failed), (2, 0), "{:?}", s.errors);
    // A JPEG of both photos: the baked look carries its own rendering.
    let out = tempfile::tempdir().expect("out");
    let id = r
        .core
        .export_start(&jpeg_job(ApplyScope::All, Some(out.path())))
        .expect("queued");
    wait_export(&r);
    let s = r.core.export_summary(id).expect("summary");
    assert_eq!((s.saved, s.failed), (2, 0), "{:?}", s.errors);
    assert!(names_in(out.path()).contains(&"DSC00002_rec709-baked.jpg".to_owned()));
}

#[test]
fn the_cli_preview_of_a_real_session() {
    let r = rig();
    let (_root, folder) = shoot(3);
    let ids = open_all(&r.core, &folder);
    let out = folder.parent().expect("root").join("Shoot export");

    // All three photos are every ARW in their folder: the folder is the input.
    let all = r.core.cli_preview_detail(&jpeg_job(ApplyScope::All, None));
    assert_eq!(
        all.command,
        format!(
            "ym render {} \\\n  --look signature \\\n  --format jpeg \\\n  --quality 92 \\\n  --colour-space srgb \\\n  --out {}",
            folder.display(),
            ym_core::cli_preview::shell_quote(&out.to_string_lossy())
        )
    );
    assert_eq!(all.adjusted_photos, 0);
    assert_eq!(all.note(), None);

    // One photo: its file. Adjustments beyond the look are counted, a different look as well.
    r.core
        .session
        .update_settings(
            ids[0],
            &SettingsPatch {
                contrast: Some(ym_core::settings::PercentSetting::manual_value(20.0)),
                ..SettingsPatch::default()
            },
        )
        .expect("edit");
    let this = r.core.cli_preview_detail(&jpeg_job(ApplyScope::This, None));
    assert!(
        this.command.starts_with(&format!(
            "ym render {} \\\n",
            folder.join("DSC00001.ARW").display()
        )),
        "{}",
        this.command
    );
    assert_eq!(this.adjusted_photos, 1);
    let all = r.core.cli_preview_detail(&jpeg_job(ApplyScope::All, None));
    assert_eq!(all.adjusted_photos, 1);
    assert_eq!(
        all.note().as_deref(),
        Some("1 photo has adjustments this command doesn't include.")
    );
    assert_eq!(
        r.core
            .export_estimate(&jpeg_job(ApplyScope::All, None))
            .adjusted_photos,
        1
    );

    // Two of three: the files, not the folder.
    r.core.session.set_picked(ids[0], true).expect("pick");
    r.core.session.set_picked(ids[2], true).expect("pick");
    let picked = r.core.cli_preview(&jpeg_job(ApplyScope::Picked, None));
    assert!(
        picked.starts_with(&format!(
            "ym render {} {} \\\n",
            folder.join("DSC00001.ARW").display(),
            folder.join("DSC00003.ARW").display()
        )),
        "{picked}"
    );
}

#[cfg(unix)]
#[test]
fn the_cli_preview_names_the_folders_as_the_session_was_opened_with_them() {
    // A session opened through a symbolic link (on a Mac, `/var` is a link to `/private/var`):
    // the command, the panel's default folder and the files all use the name the person gave.
    let r = rig();
    let (root, folder) = shoot(2);
    let links = tempfile::tempdir().expect("links");
    let via = links.path().join("via");
    std::os::unix::fs::symlink(root.path(), &via).expect("symlink");
    let opened = via.join("Shoot");
    let ids = open_all(&r.core, &opened);
    assert_eq!(ids.len(), 2);
    let out = via.join("Shoot export");

    let state = r.core.session.state();
    assert_eq!(
        state.default_export_folder.as_deref().map(Path::new),
        Some(out.as_path())
    );
    let all = r.core.cli_preview_detail(&jpeg_job(ApplyScope::All, None));
    assert_eq!(
        all.command,
        format!(
            "ym render {} \\\n  --look signature \\\n  --format jpeg \\\n  --quality 92 \\\n  --colour-space srgb \\\n  --out {}",
            opened.display(),
            ym_core::cli_preview::shell_quote(&out.to_string_lossy())
        )
    );

    // The files go where the command says (the same place on the disk as the real folder).
    r.core
        .export_start(&jpeg_job(ApplyScope::All, None))
        .expect("queued");
    wait_export(&r);
    assert_eq!(
        names_in(&out),
        vec![
            "DSC00001_signature.jpg".to_owned(),
            "DSC00002_signature.jpg".to_owned()
        ]
    );
    assert!(folder.parent().expect("root").join("Shoot export").is_dir());
}

// ---------------------------------------------------------------------------------------------
// The wire types against src/api/types.ts
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf()
}

/// The field names of `export interface <name> { ... }` in `src/api/types.ts`.
fn ts_interface_keys(source: &str, name: &str) -> BTreeSet<String> {
    let start = source
        .find(&format!("export interface {name} {{"))
        .unwrap_or_else(|| panic!("interface {name} in types.ts"));
    let body = &source[start..];
    let end = body.find("\n}").expect("end of the interface");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.strip_prefix("  ")?;
            if line.starts_with(' ') || line.starts_with('/') || line.starts_with('*') {
                return None;
            }
            let (key, _) = line.split_once(':')?;
            Some(key.trim_end_matches('?').to_owned())
        })
        .collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().expect("object").keys().cloned().collect()
}

#[test]
fn the_export_preferences_and_looks_types_have_the_fields_of_the_typescript_interfaces() {
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let check = |name: &str, value: Value| {
        assert_eq!(
            ts_interface_keys(&source, name),
            keys(&value),
            "interface {name} against its Rust type"
        );
    };
    let options = ExportOptions::default();
    check(
        "ExportOptions",
        serde_json::to_value(&options).expect("json"),
    );
    check(
        "ExportJob",
        serde_json::to_value(ExportJob {
            scope: ApplyScope::All,
            options,
        })
        .expect("json"),
    );
    check(
        "ExportEstimate",
        serde_json::to_value(ExportEstimate {
            bytes: 1,
            seconds: 1.0,
            adjusted_photos: 0,
            dng_approximate: false,
        })
        .expect("json"),
    );
    check(
        "ExportProgress",
        serde_json::to_value(ExportProgress {
            job_id: 1,
            file: String::new(),
            done: 0,
            total: 1,
            file_percent: None,
            error: None,
            skipped: false,
        })
        .expect("json"),
    );
    let prefs = Preferences::default();
    // `apple_auto` (docs/19 section 2) is a field on both sides.
    assert!(ts_interface_keys(&source, "Preferences").contains("apple_auto"));
    check("Preferences", serde_json::to_value(&prefs).expect("json"));
    check(
        "LastUsedFolders",
        serde_json::to_value(&prefs.last_used_folders).expect("json"),
    );
    check(
        "Look",
        serde_json::to_value(LookEntry {
            id: LookId::builtin("neutral"),
            name: "Neutral".into(),
            group: ym_core::looks::LookGroupInfo::BuiltIn,
            input: LookInput::LogC4,
            library: ym_looks::library::LookLibrary::BuiltIn,
            format: None,
            lut_size: None,
            is_default: true,
            missing: false,
        })
        .expect("json"),
    );
    check(
        "DisplayLut",
        serde_json::to_value(DisplayLutEntry {
            id: LookId::parse("arri-display:logc4:rec709-g24").expect("id"),
            name: "x".into(),
            input: LookInput::LogC4,
            target: "rec709-g24".into(),
        })
        .expect("json"),
    );
    check(
        "LooksImportResult",
        serde_json::to_value(LooksImportResult {
            added: vec![],
            display_luts_added: 0,
            unchanged: 0,
            notes: vec![],
            refusals: vec![],
        })
        .expect("json"),
    );
    // And the union members spelled the way the frontend spells them.
    assert!(source.contains("\"aces2\"") && source.contains("\"arri_display\""));
    assert_eq!(
        serde_json::to_value(DisplayRendering::Aces2).expect("json"),
        serde_json::json!({ "kind": "aces2" })
    );
}

#[test]
fn the_apple_auto_types_have_the_fields_of_the_typescript_interfaces() {
    // `apple_auto_status` and the photo's Apple Auto state (`docs/19` sections 10 and 12) as the
    // webview reads them.
    use ym_core::apple_auto::{AdvisorAnswer, AdvisorInfo, AppleAuto, AutoSteps};
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let check = |name: &str, value: Value| {
        assert_eq!(
            ts_interface_keys(&source, name),
            keys(&value),
            "interface {name} against its Rust type"
        );
    };
    let r = rig();
    let status = r.core.apple_auto_status();
    check(
        "AppleAutoStatus",
        serde_json::to_value(&status).expect("json"),
    );
    let info = AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)");
    check("AppleAutoInfo", serde_json::to_value(&info).expect("json"));
    let answer = AdvisorAnswer {
        steps: AutoSteps::keep(),
        scene: "A street".into(),
        plan: "Soft and warm".into(),
    };
    check(
        "AppleAutoAnswer",
        serde_json::to_value(&answer).expect("json"),
    );
    check(
        "AppleAutoSteps",
        serde_json::to_value(answer.steps).expect("json"),
    );
    // The union is tagged by `state`, with the variants `types.ts` lists.
    let answered = serde_json::to_value(AppleAuto::Answered { answer, info }).expect("json");
    assert_eq!(answered["state"], serde_json::json!("answered"));
    for state in ["off", "unavailable", "pending", "answered", "failed"] {
        assert!(
            source.contains(&format!("{{ state: \"{state}\"")),
            "AppleAuto's {state} in types.ts"
        );
    }
}

#[test]
fn the_core_and_its_parts_can_be_shared_between_threads() {
    // The application keeps the core in its managed state and calls it from every command.
    fn shareable<T: Send + Sync + 'static>() {}
    shareable::<Core>();
    shareable::<ym_core::export_queue::ExportQueue>();
    shareable::<ym_core::export::ExportContext>();
    shareable::<ym_core::looks::LookService>();
    shareable::<ym_core::preferences::PreferencesStore>();
}

#[test]
fn a_look_imported_by_the_cli_appears_in_the_list_and_looks_changed_fires_once() {
    let r = rig();
    let changed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changed);
    r.core.on_looks_changed(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    let before = r.core.looks_list().len();
    // The CLI is another process over the same folders: here another service over them.
    let prefs = Arc::new(ym_core::preferences::PreferencesStore::open(r.app.path()));
    let cli = ym_core::looks::LookService::open(r.app.path(), &data_root(), prefs).expect("cli");
    let src = tempfile::tempdir().expect("src");
    cli.import(
        &write_cube(src.path(), "FromCli.cube"),
        &ym_core::looks::ImportRequest {
            input: Some(LookInput::LogC4),
            cancel: None,
        },
        &mut |_| {},
    )
    .expect("import");
    assert_eq!(changed.load(Ordering::SeqCst), 0);
    let list = r.core.looks_list();
    assert_eq!(list.len(), before + 1);
    assert!(list.iter().any(|l| l.id.as_str() == "user:fromcli"));
    assert_eq!(changed.load(Ordering::SeqCst), 1);
    // A second listing finds nothing new and stays quiet.
    let _ = r.core.looks_list();
    assert_eq!(changed.load(Ordering::SeqCst), 1);
}

#[test]
fn quitting_while_an_export_is_writing_leaves_no_partial_file() {
    // The process ends right after `shutdown` returns, so the worker must have taken its
    // temporary file away by then ("the file being written is deleted", `docs/09`).
    let r = rig();
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Shoot");
    std::fs::create_dir(&folder).expect("folder");
    Scene::new(2400, 1600, light(PROFILE, 5200.0), Layout::Patchwork, 0.35)
        .noise(2e-4, 1e-5)
        .write(&folder, "DSC00001.ARW", MODEL);
    open_all(&r.core, &folder);
    let out = tempfile::tempdir().expect("out");
    let id = r
        .core
        .export_start(&ExportJob {
            scope: ApplyScope::All,
            options: ExportOptions {
                format: FormatKind::Png,
                bit_depth: ym_export::BitDepth::Sixteen,
                destination: Some(out.path().to_path_buf()),
                ..ExportOptions::default()
            },
        })
        .expect("queued");

    // Wait for the write phase: the render is over and the encoder has its temporary file.
    let has_partial = || {
        names_in(out.path())
            .iter()
            .any(|n| n.starts_with(".yellowmoon-"))
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while !has_partial() {
        assert!(
            std::time::Instant::now() < deadline,
            "the export never began to write"
        );
        assert!(r.core.exports.is_busy(), "the export ended before it wrote");
        std::thread::sleep(Duration::from_millis(1));
    }

    r.core.shutdown();
    assert_eq!(
        names_in(out.path()),
        Vec::<String>::new(),
        "no partial file and no finished one is left when shutdown returns"
    );
    assert!(!r.core.exports.is_busy());
    assert_eq!(
        r.core.exports.summary(id).map(|s| s.saved),
        Some(0),
        "the photo was not saved"
    );
}
