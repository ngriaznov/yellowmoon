//! Library tests: built-in looks, listing, lazy loading, verify after tampering, removal and
//! resolution of `--look` arguments. All LUTs are synthetic.

mod support;

use std::fs;
use std::path::Path;
use std::sync::Arc;

use support::{Kind, cube_text, no_progress, open_library, small_db, tree, write_package_zip};
use ym_color::logc::LogC4;
use ym_color::look::{LookInput, LookTransform};
use ym_looks::import::{ImportOptions, import_path};
use ym_looks::library::{
    DISABLED_MESSAGE, EXIT_LOOK_MISSING, EXIT_USAGE, Library, LibraryError, LoadError, LookGroup,
    LookId, LookStatus, ResolveError, VerifyOutcome,
};
use ym_looks::recognize::PackageDb;

fn options() -> ImportOptions {
    ImportOptions {
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

/// A library with a LogC4 and a LogC3 look package, a LogC4 display package and two user LUTs.
fn populated(work: &Path) -> (Library, PackageDb) {
    let db = small_db(4);
    let mut library = open_library(&work.join("app"));
    for kind in [Kind::LookLogC4, Kind::LookLogC3, Kind::DisplayLogC4] {
        let zip = write_package_zip(work, kind, 4);
        import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    }
    for (name, seed) in [("Zed Look", 1), ("Alpha Look", 2)] {
        let path = work.join(format!("{name}.cube"));
        fs::write(&path, cube_text(5, seed, name)).unwrap();
        let opts = ImportOptions {
            input: Some(LookInput::LogC4),
            ..options()
        };
        import_path(&mut library, &db, &path, &opts, &mut no_progress()).unwrap();
    }
    (library, db)
}

/// The app keeps one library behind a lock and shares transforms between render threads.
#[test]
fn the_library_and_its_results_can_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Library>();
    send_sync::<LookTransform>();
    send_sync::<ym_looks::recognize::Recognition>();
    send_sync::<ym_looks::recognize::PackageDb>();
    fn send<T: Send>() {}
    send::<ym_looks::import::ImportReport>();
    send::<ym_looks::import::CancelToken>();
}

fn ids(library: &Library) -> Vec<String> {
    library.looks().iter().map(|l| l.id.to_string()).collect()
}

#[test]
fn the_shipped_builtin_folder_lists_neutral_then_signature() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/builtin");
    assert!(data.join("neutral.json").exists());
    assert!(data.join("signature.json").exists());
    let dir = tempfile::tempdir().unwrap();
    let library = Library::open(dir.path(), Some(&data)).unwrap();
    assert_eq!(ids(&library), vec!["builtin:neutral", "builtin:signature"]);
    assert!(library.problems().is_empty(), "{:?}", library.problems());

    let neutral = library.resolve("Neutral").unwrap();
    assert_eq!(neutral.name, "Neutral");
    assert!(matches!(neutral.group, LookGroup::BuiltIn));
    assert!(library.transform(&neutral.id).unwrap().is_identity());

    // Signature is the look new photos get; it is built in, a LogC4 look, and not the identity.
    assert_eq!(LookId::factory_default().as_str(), "builtin:signature");
    assert_eq!(LookId::neutral().as_str(), "builtin:neutral");
    let signature = library.resolve("signature").unwrap();
    assert_eq!(signature.id, LookId::factory_default());
    assert_eq!(signature.name, "Signature");
    assert!(matches!(signature.group, LookGroup::BuiltIn));
    assert_eq!(signature.input, LookInput::LogC4);
    let transform = library.transform(&signature.id).unwrap();
    assert!(!transform.is_identity());
    assert_eq!(
        transform.lut().unwrap().cube.size(),
        ym_color::lut::BAKE_SIZE
    );
    // Mid grey stays where it is; a warm colour is moved.
    let grey = LogC4([0.278_395_836_5; 3]);
    let out = transform.eval_logc4(grey).unwrap().0;
    assert!((out[0] - grey.0[0]).abs() < 1e-4, "{out:?}");
    let warm = LogC4([0.5, 0.4, 0.3]);
    assert_ne!(transform.eval_logc4(warm).unwrap(), warm);
    // The same cube as the one ym-color bakes from the file.
    let text = std::fs::read_to_string(data.join("signature.json")).unwrap();
    let direct = ym_color::look::ParametricLook::from_json(&text)
        .unwrap()
        .to_transform()
        .unwrap();
    assert_eq!(*transform, direct);
}

#[test]
fn signature_is_picked_up_when_present_and_not_listed_when_absent() {
    let data = tempfile::tempdir().unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/builtin/neutral.json"),
        data.path().join("neutral.json"),
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let mut library = Library::open(root.path(), Some(data.path())).unwrap();
    assert_eq!(ids(&library), vec!["builtin:neutral"]);
    assert!(library.resolve("signature").is_err());

    fs::write(
        data.path().join("signature.json"),
        r#"{"schema":"yellowmoon.parametric-look/1","name":"Test Signature",
            "cdl":{"slope":[1.04,1.0,0.96],"offset":[0.0,0.0,0.0],"power":[1.0,1.0,1.0],"saturation":1.08},
            "hue_vs_sat":[{"hue_deg":30,"value":1.15},{"hue_deg":210,"value":0.9}],
            "luma_vs_sat":[{"luma":0.2,"value":0.85},{"luma":0.7,"value":1.05}]}"#,
    )
    .unwrap();
    library.reload();
    assert_eq!(ids(&library), vec!["builtin:neutral", "builtin:signature"]);
    let signature = library.resolve("signature").unwrap().clone();
    assert_eq!(signature.name, "Test Signature");
    assert_eq!(signature.input, LookInput::LogC4);
    let transform = library.transform(&signature.id).unwrap();
    // Baked to a cube of ym-color's bake size, and it does something.
    assert_eq!(
        transform.lut().unwrap().cube.size(),
        ym_color::lut::BAKE_SIZE
    );
    let v = LogC4([0.5, 0.4, 0.3]);
    assert_ne!(transform.eval_logc4(v).unwrap(), v);

    fs::remove_file(data.path().join("signature.json")).unwrap();
    library.reload();
    assert_eq!(ids(&library), vec!["builtin:neutral"]);
}

#[test]
fn looks_are_listed_built_in_first_then_arri_by_group_then_yours() {
    let work = tempfile::tempdir().unwrap();
    let (library, _) = populated(work.path());
    let list = ids(&library);
    assert_eq!(list[0], "builtin:neutral");
    let arri: Vec<_> = library
        .looks()
        .iter()
        .filter_map(|l| match &l.group {
            LookGroup::Arri { group_no, .. } => Some(*group_no),
            _ => None,
        })
        .collect();
    assert_eq!(arri.len(), 8);
    assert!(arri.windows(2).all(|w| w[0] <= w[1]), "{arri:?}");
    let tail: Vec<_> = list.iter().rev().take(2).rev().cloned().collect();
    assert_eq!(tail, vec!["user:alpha-look", "user:zed-look"]);
    assert_eq!(library.display_luts().len(), 4);
}

#[test]
fn verify_reports_missing_and_changed_files_and_disables_those_looks() {
    let work = tempfile::tempdir().unwrap();
    let (mut library, _) = populated(work.path());
    let clean = library.verify(&mut no_progress()).unwrap();
    assert!(clean.is_clean());
    assert_eq!(clean.checked(), 8 + 4 + 2);
    assert_eq!(clean.ok_count(), clean.checked());

    let missing = LookId::user("zed-look");
    let changed = LookId::user("alpha-look");
    let changed_display = library.display_luts()[0].id.clone();
    let missing_path = library
        .get(&missing)
        .unwrap()
        .lut
        .as_ref()
        .unwrap()
        .path
        .clone();
    let changed_path = library
        .get(&changed)
        .unwrap()
        .lut
        .as_ref()
        .unwrap()
        .path
        .clone();
    let original = fs::read(&changed_path).unwrap();
    fs::remove_file(&missing_path).unwrap();
    fs::write(
        &changed_path,
        [original.as_slice(), b"\n# edited\n"].concat(),
    )
    .unwrap();
    let display_path = library.display_luts()[0].lut.path.clone();
    fs::write(&display_path, "tampered").unwrap();

    let report = library.verify(&mut no_progress()).unwrap();
    assert!(!report.is_clean());
    assert_eq!(report.missing(), vec![&missing]);
    let mut changed_ids = report.changed();
    changed_ids.sort();
    let mut want = vec![&changed, &changed_display];
    want.sort();
    assert_eq!(changed_ids, want);
    assert!(report.entries.iter().any(|e| e.id == changed
        && matches!(&e.outcome, VerifyOutcome::Changed { expected, actual } if expected != actual)));
    assert_eq!(report.problems().count(), 3);

    // Both are listed, disabled, with the documented words; the rest are unaffected.
    for id in [&missing, &changed] {
        let look = library.get(id).unwrap();
        assert!(!look.is_enabled());
        assert_eq!(
            look.disabled_reason(),
            Some("File missing or changed. Import again.")
        );
        assert_eq!(look.disabled_reason(), Some(DISABLED_MESSAGE));
    }
    assert_eq!(library.get(&missing).unwrap().status, LookStatus::Missing);
    assert_eq!(library.get(&changed).unwrap().status, LookStatus::Changed);
    assert_eq!(
        library.display_lut_info(&changed_display).unwrap().status,
        LookStatus::Changed
    );
    assert_eq!(
        library.looks().iter().filter(|l| !l.is_enabled()).count(),
        2,
        "only the tampered looks are disabled"
    );
    // Loading refuses them too, with exit code 3 for the CLI.
    let err = library.transform(&missing).unwrap_err();
    assert!(
        matches!(
            &err,
            LoadError::Unavailable {
                status: LookStatus::Missing,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(err.exit_code(), EXIT_LOOK_MISSING);
    assert!(
        err.to_string()
            .contains("File missing or changed. Import again.")
    );
    let err = library.transform(&changed).unwrap_err();
    assert!(
        matches!(
            &err,
            LoadError::Unavailable {
                status: LookStatus::Changed,
                ..
            }
        ),
        "{err}"
    );
    assert!(library.display_lut(&changed_display).is_err());
    let healthy = library
        .looks()
        .iter()
        .find(|l| l.id.prefix() == "arri")
        .unwrap()
        .id
        .clone();
    assert!(library.transform(&healthy).is_ok());

    // A new process opening the folder sees the deleted file at once (a cheap check) and
    // refuses to *load* the altered one (it re-hashes what it reads).
    let fresh = open_library(&work.path().join("app"));
    assert_eq!(fresh.get(&missing).unwrap().status, LookStatus::Missing);
    assert!(fresh.transform(&changed).is_err());

    // Put the files back and everything is well again.
    fs::write(&changed_path, &original).unwrap();
    fs::write(&missing_path, cube_text(5, 1, "Zed Look")).unwrap();
    let healed = library.verify(&mut no_progress()).unwrap();
    // The display LUT is still tampered with; the two looks are fine.
    assert_eq!(healed.problems().count(), 1);
    assert!(library.get(&missing).unwrap().is_enabled());
    assert!(library.get(&changed).unwrap().is_enabled());
    assert!(library.transform(&changed).is_ok());
}

#[test]
fn look_arguments_resolve_by_id_slug_or_name() {
    let work = tempfile::tempdir().unwrap();
    let (mut library, _) = populated(work.path());
    // Full id, slug and name all find the same look.
    for query in ["user:alpha-look", "alpha-look", "Alpha Look", "ALPHA LOOK"] {
        assert_eq!(
            library.resolve(query).unwrap().id.as_str(),
            "user:alpha-look",
            "{query}"
        );
    }
    assert_eq!(library.resolve("builtin:neutral").unwrap().name, "Neutral");
    assert_eq!(
        library.resolve("neutral").unwrap().id.as_str(),
        "builtin:neutral"
    );

    // The LogC4 and LogC3 libraries have looks of the same name: the name is ambiguous, the
    // slug and the id are not.
    let err = library.resolve("Look 01").unwrap_err();
    match &err {
        ResolveError::Ambiguous { matches, .. } => {
            let ids: Vec<_> = matches.iter().map(|m| m.id.as_str()).collect();
            assert_eq!(ids, vec!["arri:1000:look-01", "arri:1000:look-01-logc3"]);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(err.exit_code(), EXIT_USAGE);
    assert!(err.to_string().contains("arri:1000:look-01-logc3"), "{err}");
    assert_eq!(
        library.resolve("look-01").unwrap().id.as_str(),
        "arri:1000:look-01"
    );
    assert_eq!(
        library.resolve("1000:look-01-logc3").unwrap().id.as_str(),
        "arri:1000:look-01-logc3"
    );
    assert_eq!(
        library.resolve("arri:1000:look-01-logc3").unwrap().input,
        LookInput::LogC3Ei800
    );

    // Missing.
    let err = library.resolve("does-not-exist").unwrap_err();
    assert!(matches!(err, ResolveError::NotFound { .. }));
    assert_eq!(err.exit_code(), EXIT_LOOK_MISSING);

    // A look that lost its file still resolves; asking for it then fails with exit code 3.
    let path = library
        .resolve("zed-look")
        .unwrap()
        .lut
        .as_ref()
        .unwrap()
        .path
        .clone();
    fs::remove_file(path).unwrap();
    library.verify(&mut no_progress()).unwrap();
    let look = library.resolve("zed-look").unwrap();
    assert!(!look.is_enabled());
    let id = look.id.clone();
    assert_eq!(library.transform(&id).unwrap_err().exit_code(), 3);
}

#[test]
fn luts_load_lazily_and_stay_cached_until_their_file_changes() {
    let work = tempfile::tempdir().unwrap();
    let (mut library, db) = populated(work.path());
    let id = LookId::user("alpha-look");
    let first = library.transform(&id).unwrap();
    let second = library.transform(&id).unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "the second call must not parse the file again"
    );
    // A reload keeps the cached transform: the checksum is the same.
    library.reload();
    assert!(Arc::ptr_eq(&first, &library.transform(&id).unwrap()));
    // The cached data survives the file going away (it is in memory) until a reload notices.
    // Replace the content by importing a different file under the same name: a new entry, the
    // old one untouched.
    let path = work.path().join("Alpha Look.cube");
    fs::write(&path, cube_text(5, 77, "Alpha Look v2")).unwrap();
    let opts = ImportOptions {
        input: Some(LookInput::LogC4),
        ..options()
    };
    import_path(&mut library, &db, &path, &opts, &mut no_progress()).unwrap();
    assert!(Arc::ptr_eq(&first, &library.transform(&id).unwrap()));
    let newer = library.transform(&LookId::user("alpha-look-2")).unwrap();
    assert!(!Arc::ptr_eq(&first, &newer));
}

#[test]
fn imported_looks_apply_exactly_the_way_ym_color_says() {
    let work = tempfile::tempdir().unwrap();
    let (library, _) = populated(work.path());
    let text = cube_text(5, 2, "Alpha Look");
    let reference_lut = ym_color::cube::parse_lut(&text).unwrap();
    let reference = LookTransform::new(LookInput::LogC4, None, Some(reference_lut)).unwrap();
    let ours = library.transform(&LookId::user("alpha-look")).unwrap();
    for v in [
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.9, 0.4, 0.05],
        [0.0, 1.0, 0.5],
    ] {
        let v = LogC4(v);
        for pct in [0.0, 37.5, 100.0] {
            assert_eq!(
                ours.apply_logc4(v, pct).unwrap(),
                reference.apply_logc4(v, pct).unwrap(),
                "{v:?} at {pct}%"
            );
        }
    }
    // The LogC3 look runs through ym-color's LogC3 path.
    let c3 = library.resolve("arri:1000:look-01-logc3").unwrap();
    let transform = library.transform(&c3.id).unwrap();
    assert_eq!(transform.input(), LookInput::LogC3Ei800);
    let out = transform.eval_logc4(LogC4([0.4, 0.4, 0.4])).unwrap();
    assert!(out.0.iter().all(|c| c.is_finite()));
}

#[test]
fn removing_a_look_deletes_its_entry_file_and_empty_group_folder() {
    let work = tempfile::tempdir().unwrap();
    let (mut library, _) = populated(work.path());
    let id = LookId::arri(3000, "look-02");
    let file = library.paths().file("arri/3000-film/Look_02_LogC4.cube");
    assert!(file.exists());
    assert!(library.get(&id).is_some());
    let removed = library.remove(&id).unwrap();
    assert_eq!(removed.file, "arri/3000-film/Look_02_LogC4.cube");
    assert!(removed.file_deleted);
    assert!(!file.exists());
    assert!(library.get(&id).is_none());
    // 3000-film also holds the LogC3 look of the same group, so the folder stays ...
    assert!(library.looks_dir().join("arri/3000-film").exists());
    // ... until that is removed too.
    library
        .remove(&LookId::arri(3000, "look-02-logc3"))
        .unwrap();
    assert!(!library.looks_dir().join("arri/3000-film").exists());
    assert!(library.looks_dir().join("arri").exists());

    // Another process sees the removal.
    let fresh = open_library(&work.path().join("app"));
    assert!(fresh.get(&id).is_none());
    assert_eq!(fresh.looks().len(), library.looks().len());

    // Display LUTs go the same way.
    let display = library.display_luts()[0].id.clone();
    library.remove(&display).unwrap();
    assert_eq!(library.display_luts().len(), 3);

    // Errors.
    assert!(matches!(
        library.remove(&id),
        Err(LibraryError::UnknownLook(_))
    ));
    assert!(matches!(
        library.remove(&LookId::builtin("neutral")),
        Err(LibraryError::BuiltIn(_))
    ));
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    assert!(
        tree(&library.looks_dir())
            .iter()
            .all(|f| !f.contains(".tmp-"))
    );
}
