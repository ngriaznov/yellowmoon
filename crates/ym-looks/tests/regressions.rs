//! Regression tests for defects found in the independent review of `ym-looks`. Each test names
//! the behaviour that was wrong. All LUTs are synthetic; see `support`.

mod support;

use std::fs;
use std::sync::mpsc;
use std::time::Duration;

use support::{
    Kind, cube_text, no_progress, open_library, package_files, small_db, tree, write_zip,
};
use ym_color::look::LookInput;
use ym_looks::import::watch::{DownloadsWatcher, WatchConfig, ZipNameMatcher};
use ym_looks::import::{ImportOptions, import, import_path, inspect};
use ym_looks::library::{Library, LoadError, LookId, LookStatus};
use ym_looks::manifest::sha256_hex;
use ym_looks::recognize::{Fingerprint, IssueKind, Outcome, PackageStatus, Verdict};

fn options() -> ImportOptions {
    ImportOptions {
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        input: Some(LookInput::LogC4),
        ..ImportOptions::default()
    }
}

/// A disabled look must not be applied, even when its LUT was loaded earlier in the session: the
/// spec says a look with a missing or changed file is shown disabled and `--look` on it fails.
#[test]
fn a_look_loaded_earlier_is_refused_once_its_file_is_gone_or_changed() {
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    let db = small_db(1);
    for (name, seed) in [("Gone", 1), ("Edited", 2)] {
        let path = work.path().join(format!("{name}.cube"));
        fs::write(&path, cube_text(5, seed, name)).unwrap();
        import_path(&mut library, &db, &path, &options(), &mut no_progress()).unwrap();
    }
    let gone = LookId::user("gone");
    let edited = LookId::user("edited");
    // Both are loaded (and so cached) while their files are fine.
    library.transform(&gone).unwrap();
    library.transform(&edited).unwrap();

    let gone_path = library
        .get(&gone)
        .unwrap()
        .lut
        .as_ref()
        .unwrap()
        .path
        .clone();
    let edited_path = library
        .get(&edited)
        .unwrap()
        .lut
        .as_ref()
        .unwrap()
        .path
        .clone();
    fs::remove_file(&gone_path).unwrap();
    let mut text = fs::read_to_string(&edited_path).unwrap();
    text.push_str("# edited\n");
    fs::write(&edited_path, text).unwrap();

    library.verify(&mut no_progress()).unwrap();
    assert_eq!(library.get(&gone).unwrap().status, LookStatus::Missing);
    assert_eq!(library.get(&edited).unwrap().status, LookStatus::Changed);
    for id in [&gone, &edited] {
        let err = library
            .transform(id)
            .expect_err("a disabled look must not load");
        assert!(matches!(err, LoadError::Unavailable { .. }), "{err}");
        assert_eq!(err.exit_code(), 3);
    }

    // Healing the files and verifying again brings them back.
    fs::write(&gone_path, cube_text(5, 1, "Gone")).unwrap();
    let original = cube_text(5, 2, "Edited");
    fs::write(&edited_path, original).unwrap();
    library.verify(&mut no_progress()).unwrap();
    assert!(library.transform(&gone).is_ok());
    assert!(library.transform(&edited).is_ok());
}

/// The same holds for a plain `reload` (what the app does when the CLI changed the folder): a file
/// that is gone is `Missing`, and the look must not keep working from memory.
#[test]
fn reload_also_evicts_looks_whose_file_vanished() {
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    let path = work.path().join("Vanish.cube");
    fs::write(&path, cube_text(5, 1, "Vanish")).unwrap();
    import_path(
        &mut library,
        &small_db(1),
        &path,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    let id = LookId::user("vanish");
    library.transform(&id).unwrap();
    fs::remove_file(library.get(&id).unwrap().lut.as_ref().unwrap().path.clone()).unwrap();
    library.reload();
    assert_eq!(library.get(&id).unwrap().status, LookStatus::Missing);
    assert!(matches!(
        library.transform(&id),
        Err(LoadError::Unavailable { .. })
    ));
}

/// A Rec.709-baked LogC3 look is display-referred. Applying it as a log-to-log look would be
/// wrong, so when the *zip's name* says "LogC3 to Rec709" the files must be recognised as that
/// package even if the individual file names do not repeat it.
#[test]
fn the_zip_name_can_carry_the_display_rendering() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    // Same 3-point LUT size as the log-to-log LogC3 package: only the zip name tells them apart.
    let entries: Vec<(String, Vec<u8>)> = package_files(Kind::LookLogC3, 3)
        .into_iter()
        .map(|(path, text)| (path, text.into_bytes()))
        .collect();
    let zip = work.path().join(Kind::LookLogC3Rec709.zip_name());
    write_zip(&zip, &entries);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(
        recognition.verdict(),
        Verdict::Ready,
        "{:?}",
        recognition.issues
    );
    assert_eq!(recognition.packages.len(), 1);
    assert_eq!(recognition.packages[0].id, "look-library-logc3-rec709");
    for file in &recognition.files {
        match &file.outcome {
            Outcome::ArriLook { input, .. } => assert_eq!(*input, LookInput::LogC3ToRec709),
            other => panic!("{other:?}"),
        }
    }
}

/// ... and the plain LogC3 zip name must still give the log-to-log package.
#[test]
fn the_plain_logc3_zip_name_is_still_log_to_log() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let entries: Vec<(String, Vec<u8>)> = package_files(Kind::LookLogC3, 3)
        .into_iter()
        .map(|(path, text)| (path, text.into_bytes()))
        .collect();
    let zip = work.path().join(Kind::LookLogC3.zip_name());
    write_zip(&zip, &entries);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.packages[0].id, "look-library-logc3");
    assert_eq!(recognition.verdict(), Verdict::Ready);
}

/// Two files of a package may have identical content (the same LUT under two names). The package
/// is still complete when both are present, both are kept, and when one is missing the missing
/// path is named.
#[test]
fn fingerprints_with_identical_content_count_each_file() {
    let work = tempfile::tempdir().unwrap();
    let mut db = small_db(3);
    let text = cube_text(5, 7, "same");
    let other = cube_text(5, 8, "other");
    let paths = [
        "1000 Colour/Look_A_LogC4.cube",
        "3000 Film/Look_B_LogC4.cube",
        "3000 Film/Look_C_LogC4.cube",
    ];
    let contents = [&text, &text, &other];
    let package = db
        .packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc4")
        .unwrap();
    package.fingerprints = paths
        .iter()
        .zip(contents)
        .map(|(path, content)| Fingerprint {
            path: (*path).to_string(),
            size: content.len() as u64,
            lut_size: 5,
            sha256: sha256_hex(content.as_bytes()),
        })
        .collect();

    // Everything there, under names that say nothing.
    let entries: Vec<(String, Vec<u8>)> = contents
        .iter()
        .enumerate()
        .map(|(i, c)| (format!("x{i}.cube"), c.as_bytes().to_vec()))
        .collect();
    let zip = work.path().join("download.zip");
    write_zip(&zip, &entries);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(
        recognition.verdict(),
        Verdict::Ready,
        "{:?}",
        recognition.issues
    );
    assert_eq!(recognition.packages[0].status, PackageStatus::Complete);
    assert_eq!(recognition.packages[0].found, 3);
    // Each file is named after a *different* fingerprint, so neither duplicate is lost.
    let mut named: Vec<String> = recognition
        .files
        .iter()
        .map(|f| match &f.outcome {
            Outcome::ArriLook {
                group_no,
                name,
                file_name,
                ..
            } => format!("{group_no}/{name}/{file_name}"),
            other => panic!("{other:?}"),
        })
        .collect();
    named.sort();
    named.dedup();
    assert_eq!(named.len(), 3, "{named:?}");
    let mut library = open_library(&work.path().join("app"));
    let report = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &ImportOptions::default(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 3, "{report:?}");

    // The duplicate content present once only: the second path is reported missing.
    let short = work.path().join("short.zip");
    write_zip(&short, &entries[1..]);
    let recognition = inspect(&short, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Incomplete);
    let finding = &recognition.packages[0];
    assert_eq!((finding.found, finding.expected), (2, Some(3)));
    assert_eq!(finding.missing.len(), 1, "{:?}", finding.missing);
    let problem = recognition
        .issues
        .iter()
        .find(|i| i.kind == IssueKind::Incomplete)
        .unwrap();
    assert!(problem.message.contains("Missing:"), "{}", problem.message);
}

/// Two built-in files whose names make the same slug would give two looks the same id.
#[test]
fn built_in_files_with_the_same_slug_are_not_both_listed() {
    let data = tempfile::tempdir().unwrap();
    let json = r#"{"schema":"yellowmoon.parametric-look/1","name":"Warm",
        "cdl":{"slope":[1.05,1,1],"offset":[0,0,0],"power":[1,1,1],"saturation":1}}"#;
    fs::write(data.path().join("Warm Look.json"), json).unwrap();
    fs::write(data.path().join("warm-look.json"), json).unwrap();
    let root = tempfile::tempdir().unwrap();
    let library = Library::open(root.path(), Some(data.path())).unwrap();
    let ids: Vec<String> = library.looks().iter().map(|l| l.id.to_string()).collect();
    let unique: std::collections::BTreeSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), unique.len(), "duplicate ids: {ids:?}");
    assert_eq!(library.problems().len(), 1, "{:?}", library.problems());
}

/// If the Downloads folder cannot be read for a moment (access withdrawn, volume unmounted), zips
/// that were already reported or were there at the start must not be reported as new when it
/// comes back.
#[test]
fn a_folder_that_disappears_for_a_moment_does_not_re_report_old_downloads() {
    let parent = tempfile::tempdir().unwrap();
    let folder = parent.path().join("Downloads");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("ARRI_Look_Library_LogC4.zip"), b"already there").unwrap();
    let (tx, rx) = mpsc::channel();
    let config = WatchConfig {
        poll_interval: Duration::from_millis(20),
        stable_for: Duration::from_millis(300),
        report_existing: false,
    };
    let matcher = ZipNameMatcher::new(&small_db(1), None);
    let watcher = DownloadsWatcher::start(&folder, matcher, &config, move |found| {
        let _ = tx.send(found);
    })
    .unwrap();
    std::thread::sleep(Duration::from_millis(150));
    let away = parent.path().join("Downloads-away");
    fs::rename(&folder, &away).unwrap();
    // Long enough for several failed listings.
    std::thread::sleep(Duration::from_millis(250));
    assert!(watcher.last_error().is_some());
    fs::rename(&away, &folder).unwrap();
    assert!(
        rx.recv_timeout(Duration::from_millis(1200)).is_err(),
        "an old download was reported as new after the folder came back"
    );
    assert!(watcher.last_error().is_none());
    watcher.stop();
}

/// Sanity: the tree helper used above sees no leftovers after an import.
#[test]
fn imports_leave_no_temporary_files() {
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    let path = work.path().join("One.cube");
    fs::write(&path, cube_text(5, 1, "One")).unwrap();
    import_path(
        &mut library,
        &small_db(1),
        &path,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    assert!(
        tree(&work.path().join("app"))
            .iter()
            .all(|f| !f.contains(".tmp-"))
    );
}

/// `looks verify` on a looks list that cannot be read must say so, not report "all fine".
#[test]
fn verify_refuses_to_call_an_unreadable_list_clean() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    fs::create_dir_all(root.join("looks")).unwrap();

    fs::write(root.join("looks/manifest.json"), "{ not json").unwrap();
    let mut library = open_library(&root);
    let err = library.verify(&mut no_progress()).unwrap_err();
    assert!(
        matches!(&err, ym_looks::library::LibraryError::Manifest(_)),
        "{err}"
    );
    assert!(err.to_string().contains("damaged"), "{err}");

    fs::write(root.join("looks/manifest.json"), r#"{"version": 2}"#).unwrap();
    let err = library.verify(&mut no_progress()).unwrap_err();
    assert!(err.to_string().contains("newer version"), "{err}");

    // A repaired list verifies again.
    fs::write(root.join("looks/manifest.json"), r#"{"version": 1}"#).unwrap();
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

/// File names are limited to 255 bytes. A look whose name is long in a script with multi-byte
/// letters (the stem was capped by characters only) must still import, hidden temporary name and
/// encoding suffix included.
#[test]
fn very_long_non_latin_names_still_import() {
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    // 80 three-byte characters: 240 bytes plus ".cube" is a legal 245-byte file name.
    let name = "\u{8272}".repeat(80);
    let path = work.path().join(format!("{name}.cube"));
    fs::write(&path, cube_text(5, 1, "long")).unwrap();
    let (_, report) = import_path(
        &mut library,
        &small_db(1),
        &path,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 1, "{report:?}");
    let file = &report.added[0].file;
    let leaf = file.rsplit('/').next().unwrap();
    assert!(leaf.len() <= 200, "{} bytes: {leaf}", leaf.len());
    assert!(library.paths().file(file).exists());
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    assert!(
        tree(&work.path().join("app"))
            .iter()
            .all(|f| !f.contains(".tmp-"))
    );
}

/// Two looks of one package whose names differ only in punctuation get the same id and the same
/// file name. They must both be kept (one under a numbered id), and importing the package again
/// must not swap their contents.
#[test]
fn looks_with_colliding_names_in_one_package_are_both_kept() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("ARRI_Look_Library_LogC4");
    let dir = src.join("3000 Film");
    fs::create_dir_all(&dir).unwrap();
    let a = cube_text(5, 1, "a");
    let b = cube_text(5, 2, "b");
    fs::write(dir.join("Warm!_LogC4.cube"), &a).unwrap();
    fs::write(dir.join("Warm?_LogC4.cube"), &b).unwrap();
    let db = small_db(2);
    let root = work.path().join("app");
    let mut library = open_library(&root);
    let (_, report) = import_path(&mut library, &db, &src, &options(), &mut no_progress()).unwrap();
    assert_eq!(report.added.len(), 2, "{report:?}");
    assert!(report.updated.is_empty(), "{report:?}");
    let read = |item: &ym_looks::import::ImportedItem| {
        fs::read_to_string(library.paths().file(&item.file)).unwrap()
    };
    let mut contents: Vec<String> = report.added.iter().map(read).collect();
    contents.sort();
    let mut want = vec![a, b];
    want.sort();
    assert_eq!(contents, want, "both files must be in the library");

    // The same package again: nothing changes, nothing is swapped.
    let manifest_before = fs::read(library.paths().manifest_path()).unwrap();
    let (_, again) = import_path(&mut library, &db, &src, &options(), &mut no_progress()).unwrap();
    assert!(
        again.added.is_empty() && again.updated.is_empty(),
        "{again:?}"
    );
    assert_eq!(again.unchanged.len(), 2);
    assert_eq!(
        fs::read(library.paths().manifest_path()).unwrap(),
        manifest_before
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

/// The name of the folder the user picked is theirs, not ARRI's: a date-stamped or numbered
/// download folder must not turn into a look group (and so stop display LUTs being recognised).
#[test]
fn a_folder_named_with_a_year_is_not_a_look_group() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    for root_name in ["2026-10-12 ARRI downloads", "2026 Display LUTs", "2026"] {
        let src = work.path().join(root_name);
        fs::create_dir_all(&src).unwrap();
        for (path, text) in package_files(Kind::DisplayLogC4, 3) {
            fs::write(src.join(path), text).unwrap();
        }
        let recognition = inspect(&src, &db, None, &mut no_progress()).unwrap();
        // Still recognised as the display package (and so not taken for a look group): its three
        // files are display LUTs, which have nothing to import.
        assert_eq!(
            recognition.verdict(),
            Verdict::Blocked,
            "{root_name}: {:?}",
            recognition.issues
        );
        assert_eq!(recognition.display_lut_count(), 3, "{root_name}");
        assert!(
            recognition.files.iter().all(|f| matches!(
                &f.outcome,
                Outcome::ArriDisplay { package, .. } if package == "display-luts-logc4"
            )),
            "{root_name}: {:?}",
            recognition.files
        );
        // Paths are relative to the chosen folder, as inside a zip.
        assert!(
            recognition.files.iter().all(|f| !f.rel_path.contains('/')),
            "{:?}",
            recognition
                .files
                .iter()
                .map(|f| &f.rel_path)
                .collect::<Vec<_>>()
        );
    }
}
