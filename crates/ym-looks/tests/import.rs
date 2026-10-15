//! Importer tests: every package kind, the failure paths, idempotence, validation and safety.
//! All LUTs are synthetic; see `support`.

mod support;

use std::fs;
use std::path::Path;

use support::{
    Kind, cube_text, no_progress, open_library, package_files, small_db, tree, write_package_zip,
    write_zip,
};
use ym_color::look::LookInput;
use ym_looks::import::{
    CancelToken, ImportError, ImportOptions, Progress, Stage, import, import_path, inspect,
};
use ym_looks::library::{DisplayInput, LookGroup, LookId, LookStatus};
use ym_looks::manifest::sha256_hex;
use ym_looks::recognize::{
    Evidence, Fingerprint, IssueKind, Outcome, PackageStatus, Severity, Verdict,
};

fn options() -> ImportOptions {
    ImportOptions {
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

fn with_input(input: LookInput) -> ImportOptions {
    ImportOptions {
        input: Some(input),
        ..options()
    }
}

fn leftovers(dir: &Path) -> Vec<String> {
    tree(dir)
        .into_iter()
        .filter(|p| p.contains(".tmp-"))
        .collect()
}

#[test]
fn happy_path_for_every_package_kind() {
    for kind in Kind::ALL {
        let work = tempfile::tempdir().unwrap();
        let root = work.path().join("app");
        let zip = write_package_zip(work.path(), kind, 4);
        let db = small_db(4);
        let mut library = open_library(&root);

        let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
        assert_eq!(
            recognition.verdict(),
            Verdict::Ready,
            "{kind:?}: {:?}",
            recognition.issues
        );
        assert_eq!(recognition.packages.len(), 1, "{kind:?}");
        let finding = &recognition.packages[0];
        assert_eq!(finding.id, kind.package_id());
        assert_eq!((finding.found, finding.expected), (4, Some(4)));
        assert_eq!(finding.status, PackageStatus::Complete);
        assert_eq!(finding.evidence, Evidence::Pattern);
        assert_eq!(finding.lut_sizes, vec![kind.lut_size()]);
        // The ALF4 files, the PDF and the macOS junk were set aside, not read as LUTs.
        assert_eq!(recognition.ignored.alf4, 4);
        assert_eq!(recognition.ignored.pdf, 1);
        assert_eq!(recognition.ignored.macos_metadata, 4);
        assert!(
            recognition
                .issues
                .iter()
                .all(|i| i.severity != Severity::Problem)
        );

        let report = import(
            &mut library,
            &db,
            &zip,
            &recognition,
            &options(),
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), 4, "{kind:?}");
        assert!(
            report.updated.is_empty() && report.unchanged.is_empty() && report.rejected.is_empty()
        );

        // The copies are byte-for-byte what was in the zip.
        let looks_dir = library.looks_dir();
        let mut expected: Vec<String> = package_files(kind, 4)
            .iter()
            .map(|(_, text)| sha256_hex(text.as_bytes()))
            .collect();
        let mut copied: Vec<String> = report
            .added
            .iter()
            .map(|item| sha256_hex(&fs::read(looks_dir.join(&item.file)).unwrap()))
            .collect();
        expected.sort();
        copied.sort();
        assert_eq!(
            copied, expected,
            "{kind:?}: copies differ from the originals"
        );

        if kind.is_display() {
            assert!(library.looks().iter().all(|l| l.id.prefix() == "builtin"));
            assert_eq!(library.display_luts().len(), 4);
            let input = if kind == Kind::DisplayLogC4 {
                DisplayInput::LogC4
            } else {
                DisplayInput::LogC3
            };
            let first = &library.display_luts()[0];
            assert_eq!(first.input, input);
            assert!(
                first
                    .id
                    .as_str()
                    .starts_with(&format!("arri-display:{}:", input.slug()))
            );
            assert!(
                library
                    .display_luts()
                    .iter()
                    .any(|d| d.target == "rec709-g24"),
                "{:?}",
                library
                    .display_luts()
                    .iter()
                    .map(|d| &d.target)
                    .collect::<Vec<_>>()
            );
            assert!(
                tree(&looks_dir)
                    .iter()
                    .any(|f| f.starts_with("arri-display/"))
            );
            // And a display LUT loads and samples.
            let lut = library.display_lut(&first.id).unwrap();
            assert_eq!(lut.cube.size(), kind.lut_size());
        } else {
            let imported: Vec<_> = library
                .looks()
                .iter()
                .filter(|l| l.id.prefix() == "arri")
                .collect();
            assert_eq!(imported.len(), 4);
            let expected_input = match kind {
                Kind::LookLogC4 => LookInput::LogC4,
                Kind::LookLogC3 => LookInput::LogC3Ei800,
                _ => LookInput::LogC3ToRec709,
            };
            for look in &imported {
                assert_eq!(look.input, expected_input);
                assert!(
                    matches!(&look.group, LookGroup::Arri { group_no, .. } if [1000, 3000, 4000].contains(group_no))
                );
                let source = look.source.as_ref().unwrap();
                assert_eq!(source.package.as_deref(), Some(kind.package_id()));
                assert_eq!(source.url.as_deref(), Some(db.page_url()));
                assert_eq!(source.imported, "2026-11-02T10:14:00Z");
                assert!(look.is_enabled());
            }
            // Layout: arri/<group_no>-<group_slug>/<file>.cube
            let files = tree(&looks_dir);
            assert!(
                files.iter().any(|f| f.starts_with("arri/3000-film/")),
                "{files:?}"
            );
            assert!(
                files.iter().any(|f| f.starts_with("arri/1000-colour/")),
                "{files:?}"
            );
            // Ids carry the encoding for the older packages so the three libraries never collide.
            let suffix = match kind {
                Kind::LookLogC4 => "",
                Kind::LookLogC3 => "-logc3",
                _ => "-logc3-rec709",
            };
            assert!(
                imported
                    .iter()
                    .any(|l| l.id.as_str() == format!("arri:1000:look-01{suffix}")),
                "{:?}",
                imported
                    .iter()
                    .map(|l| l.id.to_string())
                    .collect::<Vec<_>>()
            );
            // Each look loads lazily and has the encoding it was imported with.
            let transform = library.transform(&imported[0].id).unwrap();
            assert_eq!(transform.input(), expected_input);
            assert_eq!(
                transform.includes_display_rendering(),
                kind == Kind::LookLogC3Rec709
            );
        }
        assert!(library.verify(&mut no_progress()).unwrap().is_clean());
        assert!(leftovers(&root).is_empty());
        // A fresh process (a second Library over the same folder) sees the same thing.
        let again = open_library(&root);
        assert_eq!(again.looks().len(), library.looks().len());
        assert_eq!(again.display_luts().len(), library.display_luts().len());
    }
}

#[test]
fn inspecting_many_files_in_parallel_keeps_order_and_content() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("many");
    fs::create_dir_all(&src).unwrap();
    let mut expected = Vec::new();
    for i in 0..60_u32 {
        // Different sizes so the workers finish out of order.
        let size = [2, 3, 5, 9][(i % 4) as usize];
        let text = cube_text(size, i + 1, &format!("look {i}"));
        let name = format!("look_{i:02}.cube");
        fs::write(src.join(&name), &text).unwrap();
        expected.push((name.clone(), sha256_hex(text.as_bytes()), size));
    }
    // One damaged file in the middle.
    fs::write(src.join("look_30b.cube"), "LUT_3D_SIZE 3\n0 0 0\n").unwrap();
    expected.push((
        "look_30b.cube".to_string(),
        sha256_hex(b"LUT_3D_SIZE 3\n0 0 0\n"),
        0,
    ));
    expected.sort();

    let mut counts = Vec::new();
    let mut record = |p: Progress<'_>| counts.push((p.done, p.total));
    let recognition = inspect(&src, &small_db(1), None, &mut record).unwrap();
    assert_eq!(recognition.files.len(), 61);
    for (file, (path, sha, size)) in recognition.files.iter().zip(&expected) {
        assert_eq!(&file.rel_path, path);
        assert_eq!(&file.sha256, sha, "{path}");
        assert_eq!(file.lut_size, (*size != 0).then_some(*size), "{path}");
    }
    assert_eq!(recognition.rejected_count(), 1);
    assert_eq!(counts.last(), Some(&(61, 61)));
    assert!(counts.iter().all(|c| c.1 == 61 && c.0 <= 61));
    let done: Vec<usize> = counts.iter().map(|c| c.0).collect();
    assert!(
        done.windows(2).all(|w| w[0] <= w[1]),
        "progress never goes backwards: {done:?}"
    );
}

#[test]
fn progress_reports_file_counts_for_the_wizard() {
    let work = tempfile::tempdir().unwrap();
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 5);
    let db = small_db(5);
    let mut library = open_library(&work.path().join("app"));
    let mut seen: Vec<(Stage, usize, usize)> = Vec::new();
    let mut record = |p: Progress<'_>| seen.push((p.stage, p.done, p.total));
    import_path(&mut library, &db, &zip, &options(), &mut record).unwrap();
    let scan: Vec<_> = seen.iter().filter(|s| s.0 == Stage::Scan).collect();
    let copy: Vec<_> = seen.iter().filter(|s| s.0 == Stage::Copy).collect();
    assert_eq!(scan.first().map(|s| (s.1, s.2)), Some((0, 5)));
    assert_eq!(scan.last().map(|s| (s.1, s.2)), Some((5, 5)));
    assert_eq!(copy.first().map(|s| (s.1, s.2)), Some((0, 5)));
    assert_eq!(copy.last().map(|s| (s.1, s.2)), Some((5, 5)));
    assert!(copy.windows(2).all(|w| w[0].1 <= w[1].1), "monotonic");
}

#[test]
fn alf4_only_zip_is_explained_and_nothing_is_written() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let zip = work.path().join("ARRI_Look_Library_LogC4.zip");
    write_zip(
        &zip,
        &[
            ("3000 Film/Look_01.alf4".to_string(), b"binary".to_vec()),
            ("3000 Film/Look_02.alf4".to_string(), b"binary".to_vec()),
            (
                "__MACOSX/3000 Film/._Look_01.alf4".to_string(),
                b"x".to_vec(),
            ),
            ("Guide.pdf".to_string(), b"%PDF".to_vec()),
        ],
    );
    let db = small_db(2);
    let mut library = open_library(&root);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Blocked);
    assert_eq!(recognition.ignored.alf4, 2);
    let issue = &recognition.issues[0];
    assert_eq!(issue.kind, IssueKind::NoCubeFiles);
    assert_eq!(
        issue.message,
        "These are ARRI look files for the camera (.alf4). Yellowmoon needs the 3D-LUT version: \
         download 'Look Library for LogC4 (3D-LUTs)' from ARRI's look page."
    );
    let err = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &options(),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ImportError::NothingToImport(m) if m == &issue.message),
        "{err}"
    );
    assert!(
        tree(&root).is_empty(),
        "a failed import must leave no trace: {:?}",
        tree(&root)
    );
}

#[test]
fn a_zip_with_no_cube_files_and_no_alf_files_is_explained() {
    let work = tempfile::tempdir().unwrap();
    let zip = work.path().join("holiday.zip");
    write_zip(&zip, &[("photo.jpg".to_string(), vec![1, 2, 3])]);
    let recognition = inspect(&zip, &small_db(1), None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Blocked);
    assert!(recognition.issues[0].message.contains("no .cube files"));
    assert_eq!(recognition.ignored.other, 1);
}

#[test]
fn a_package_from_somewhere_else_gets_a_plain_explanation_and_the_question() {
    let work = tempfile::tempdir().unwrap();
    let zip = work.path().join("ARRI_ALEXA35_REVEAL_LUTs.zip");
    write_zip(
        &zip,
        &[
            (
                "REVEAL/Warm.cube".to_string(),
                cube_text(5, 1, "Warm").into_bytes(),
            ),
            (
                "REVEAL/Cool.cube".to_string(),
                cube_text(5, 2, "Cool").into_bytes(),
            ),
        ],
    );
    let db = small_db(2);
    let root = work.path().join("app");
    let mut library = open_library(&root);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::NeedsAnswer);
    assert!(recognition.packages.is_empty());
    let kinds: Vec<_> = recognition.issues.iter().map(|i| i.kind).collect();
    assert!(kinds.contains(&IssueKind::UnknownArriPackage), "{kinds:?}");
    assert!(kinds.contains(&IssueKind::NeedsAnswer));
    let message = &recognition
        .issues
        .iter()
        .find(|i| i.kind == IssueKind::UnknownArriPackage)
        .unwrap()
        .message;
    assert!(message.contains("another camera"), "{message}");

    // Without an answer nothing happens; with one they become the user's own LUTs.
    let err = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &options(),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ImportError::NeedsInput { files } if files.len() == 2),
        "{err}"
    );
    assert_eq!(err.exit_code(), 2);
    assert!(tree(&root).is_empty());
    let report = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 2);
    assert!(library.looks().iter().any(|l| l.id.as_str() == "user:warm"));
}

#[test]
fn a_package_with_a_missing_file_is_reported_incomplete() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    // The list expects 4 looks; the zip holds 3.
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 3);
    let db = small_db(4);
    let mut library = open_library(&root);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Incomplete);
    let finding = &recognition.packages[0];
    assert_eq!((finding.found, finding.expected), (3, Some(4)));
    assert_eq!(finding.status, PackageStatus::Incomplete);
    let problem = recognition.problems().next().unwrap();
    assert_eq!(problem.kind, IssueKind::Incomplete);
    assert!(
        problem.message.contains("should have 4 files"),
        "{}",
        problem.message
    );
    assert!(problem.message.contains("has 3"), "{}", problem.message);
    assert!(
        problem.message.contains("Download it again"),
        "{}",
        problem.message
    );

    // A caller that insists on complete packages is refused, and nothing is written ...
    let strict = ImportOptions {
        require_complete: true,
        ..options()
    };
    let err = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &strict,
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(matches!(err, ImportError::Incomplete(_)), "{err}");
    assert!(tree(&root).is_empty());
    // ... while the default imports what is there and reports it.
    let report = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 3);
}

#[test]
fn a_chosen_file_that_is_not_there_is_an_error_with_exit_code_2() {
    let work = tempfile::tempdir().unwrap();
    let err = inspect(
        &work.path().join("gone.zip"),
        &small_db(1),
        None,
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(matches!(err, ImportError::SourceMissing(_)), "{err}");
    assert_eq!(err.exit_code(), 2);
    assert!(err.to_string().contains("gone.zip"));
}

#[test]
fn unsupported_and_damaged_sources_are_refused_politely() {
    let work = tempfile::tempdir().unwrap();
    let text = work.path().join("notes.txt");
    fs::write(&text, "hello").unwrap();
    let err = inspect(&text, &small_db(1), None, &mut no_progress()).unwrap_err();
    assert!(
        matches!(&err, ImportError::UnsupportedSource(n) if n == "notes.txt"),
        "{err}"
    );
    assert_eq!(err.exit_code(), 2);

    let bad_zip = work.path().join("cut-off.zip");
    let full = {
        let p = work.path().join("full.zip");
        write_zip(
            &p,
            &[("a.cube".to_string(), cube_text(3, 1, "a").into_bytes())],
        );
        fs::read(&p).unwrap()
    };
    fs::write(&bad_zip, &full[..full.len() / 2]).unwrap();
    let err = inspect(&bad_zip, &small_db(1), None, &mut no_progress()).unwrap_err();
    assert!(matches!(err, ImportError::BadZip(_)), "{err}");
    assert!(err.to_string().contains("Download it again"), "{err}");
}

#[test]
fn corrupt_and_invalid_cubes_are_rejected_with_reasons_and_the_rest_imports() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let src = work.path().join("mixed");
    fs::create_dir_all(&src).unwrap();
    let good = cube_text(5, 1, "good");
    fs::write(src.join("good.cube"), &good).unwrap();
    // Cut off in the middle of the table.
    fs::write(src.join("truncated.cube"), &good[..good.len() / 2]).unwrap();
    // Not a LUT at all (a saved web page, say).
    fs::write(src.join("page.cube"), "<html><body>Not found</body></html>").unwrap();
    // 1D only.
    fs::write(src.join("one-d.cube"), "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n").unwrap();
    // Size outside 2 to 129.
    fs::write(src.join("huge.cube"), "LUT_3D_SIZE 130\n0 0 0\n").unwrap();
    // A value that is not finite.
    fs::write(
        src.join("nan.cube"),
        "LUT_3D_SIZE 2\n0 0 0\nnan 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n",
    )
    .unwrap();
    // Binary junk.
    fs::write(src.join("binary.cube"), [0xff_u8, 0xfe, 0x00, 0x01]).unwrap();
    // Empty.
    fs::write(src.join("empty.cube"), "").unwrap();

    let db = small_db(1);
    let mut library = open_library(&root);
    let recognition = inspect(&src, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.rejected_count(), 7);
    assert_eq!(recognition.needs_answer_count(), 1);
    let report = import(
        &mut library,
        &db,
        &src,
        &recognition,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 1);
    assert_eq!(report.added[0].id.as_str(), "user:good");
    assert_eq!(report.rejected.len(), 7);
    let reason = |name: &str| {
        report
            .rejected
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("{name} not rejected"))
            .reason
            .clone()
    };
    assert!(
        reason("truncated.cube").contains("data rows"),
        "{}",
        reason("truncated.cube")
    );
    assert!(
        reason("page.cube").contains("unknown keyword"),
        "{}",
        reason("page.cube")
    );
    assert!(
        reason("one-d.cube").contains("1D LUT"),
        "{}",
        reason("one-d.cube")
    );
    assert!(
        reason("huge.cube").contains("130"),
        "{}",
        reason("huge.cube")
    );
    assert!(
        reason("nan.cube").contains("not finite"),
        "{}",
        reason("nan.cube")
    );
    assert!(
        reason("binary.cube").contains("not a text file"),
        "{}",
        reason("binary.cube")
    );
    assert!(
        reason("empty.cube").contains("LUT_3D_SIZE"),
        "{}",
        reason("empty.cube")
    );
    // Only the good file reached the library.
    let files = tree(&library.looks_dir());
    assert!(files.contains(&"user/good.cube".to_string()), "{files:?}");
    assert_eq!(files.iter().filter(|f| f.ends_with(".cube")).count(), 1);
}

#[test]
fn a_source_of_only_bad_cubes_is_blocked_with_the_first_reason() {
    let work = tempfile::tempdir().unwrap();
    let bad = work.path().join("bad.cube");
    fs::write(&bad, "LUT_3D_SIZE 2\nnot numbers here\n").unwrap();
    let db = small_db(1);
    let mut library = open_library(&work.path().join("app"));
    let recognition = inspect(&bad, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Blocked);
    assert_eq!(recognition.issues[0].kind, IssueKind::AllRejected);
    let err = import(
        &mut library,
        &db,
        &bad,
        &recognition,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ImportError::NothingToImport(m) if m.contains("bad.cube")),
        "{err}"
    );
}

#[test]
fn reimporting_the_same_package_changes_nothing() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 6);
    let db = small_db(6);
    let mut library = open_library(&root);
    let (_, first) = import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    assert_eq!(first.added.len(), 6);
    let manifest_before = fs::read(library.paths().manifest_path()).unwrap();
    let files_before = tree(&root);

    // Later, from another run of the program, with a different clock.
    let mut library = open_library(&root);
    let later = ImportOptions {
        imported_at: Some("2027-01-01T00:00:00Z".to_string()),
        ..ImportOptions::default()
    };
    let (_, second) = import_path(&mut library, &db, &zip, &later, &mut no_progress()).unwrap();
    assert!(
        second.added.is_empty() && second.updated.is_empty(),
        "{second:?}"
    );
    assert_eq!(second.unchanged.len(), 6);
    assert_eq!(
        fs::read(library.paths().manifest_path()).unwrap(),
        manifest_before
    );
    assert_eq!(tree(&root), files_before);
    assert_eq!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .count(),
        6
    );
    assert!(leftovers(&root).is_empty());
}

#[test]
fn importing_the_package_as_a_folder_after_the_zip_is_also_idempotent() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let zip = write_package_zip(work.path(), Kind::LookLogC3, 4);
    let unpacked = work.path().join("ARRI_Look_Library_LogC3");
    for (path, text) in package_files(Kind::LookLogC3, 4) {
        let dest = unpacked.join(&path);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(dest, text).unwrap();
    }
    let db = small_db(4);
    let mut library = open_library(&root);
    import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    let (recognition, report) =
        import_path(&mut library, &db, &unpacked, &options(), &mut no_progress()).unwrap();
    assert_eq!(recognition.packages[0].id, "look-library-logc3");
    assert_eq!(report.unchanged.len(), 4, "{report:?}");
    assert!(report.added.is_empty());
}

#[test]
fn reimport_repairs_a_missing_or_altered_copy() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 4);
    let db = small_db(4);
    let mut library = open_library(&root);
    import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    let files: Vec<String> = tree(&library.looks_dir())
        .into_iter()
        .filter(|f| f.ends_with(".cube"))
        .collect();
    fs::remove_file(library.looks_dir().join(&files[0])).unwrap();
    fs::write(library.looks_dir().join(&files[1]), "tampered").unwrap();
    let report = library.verify(&mut no_progress()).unwrap();
    assert!(!report.is_clean());

    let (_, again) = import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    assert_eq!(again.updated.len(), 2, "{again:?}");
    assert_eq!(again.unchanged.len(), 2);
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    assert!(library.looks().iter().all(|l| l.status == LookStatus::Ok));
}

#[test]
fn a_changed_arri_file_replaces_its_entry_in_place() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let db = small_db(2);
    let mut library = open_library(&root);
    let v1 = write_package_zip(work.path(), Kind::LookLogC4, 2);
    import_path(&mut library, &db, &v1, &options(), &mut no_progress()).unwrap();
    // ARRI re-releases one look: same name, new content.
    let v2_dir = work.path().join("v2");
    fs::create_dir_all(&v2_dir).unwrap();
    let mut entries: Vec<(String, Vec<u8>)> = package_files(Kind::LookLogC4, 2)
        .into_iter()
        .map(|(p, t)| (p, t.into_bytes()))
        .collect();
    entries[0].1 = cube_text(5, 999, "LogC4 revised").into_bytes();
    let v2 = v2_dir.join(Kind::LookLogC4.zip_name());
    write_zip(&v2, &entries);
    let (_, report) = import_path(&mut library, &db, &v2, &options(), &mut no_progress()).unwrap();
    assert_eq!(report.updated.len(), 1, "{report:?}");
    assert_eq!(report.unchanged.len(), 1);
    assert_eq!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .count(),
        2
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

#[test]
fn single_cube_with_each_answer_to_the_generic_question() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let db = small_db(1);
    let mut library = open_library(&root);
    for (i, (name, input)) in [
        ("Warm Glow", LookInput::LogC4),
        ("Old School", LookInput::LogC3Ei800),
        ("Video Look", LookInput::LogC3ToRec709),
    ]
    .into_iter()
    .enumerate()
    {
        let path = work.path().join(format!("{name}.cube"));
        fs::write(&path, cube_text(5, i as u32 + 1, name)).unwrap();
        let (recognition, report) = import_path(
            &mut library,
            &db,
            &path,
            &with_input(input),
            &mut no_progress(),
        )
        .unwrap();
        assert!(recognition.needs_answer());
        assert_eq!(report.added.len(), 1);
        let look = library.get(&report.added[0].id).unwrap();
        assert_eq!(look.name, name);
        assert_eq!(look.input, input);
        assert!(matches!(look.group, LookGroup::User));
        assert!(library.transform(&look.id).is_ok());
        assert_eq!(
            library
                .transform(&look.id)
                .unwrap()
                .includes_display_rendering(),
            input == LookInput::LogC3ToRec709
        );
    }
    assert_eq!(
        library
            .looks()
            .iter()
            .map(|l| l.id.to_string())
            .collect::<Vec<_>>(),
        vec![
            "builtin:neutral",
            "user:old-school",
            "user:video-look",
            "user:warm-glow"
        ]
    );
    assert!(tree(&library.looks_dir()).contains(&"user/Warm Glow.cube".to_string()));
}

#[test]
fn a_single_cube_without_an_answer_asks_for_one() {
    let work = tempfile::tempdir().unwrap();
    let path = work.path().join("mine.cube");
    fs::write(&path, cube_text(5, 1, "mine")).unwrap();
    let mut library = open_library(&work.path().join("app"));
    let err = import_path(
        &mut library,
        &small_db(1),
        &path,
        &options(),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ImportError::NeedsInput { files } if files == &["mine.cube"]),
        "{err}"
    );
    assert!(err.to_string().contains("LogC3 with Rec.709 baked in"));
}

#[test]
fn answering_differently_later_updates_the_input_and_new_content_gets_a_new_entry() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(1);
    let mut library = open_library(&work.path().join("app"));
    let path = work.path().join("look.cube");
    fs::write(&path, cube_text(5, 1, "look")).unwrap();
    import_path(
        &mut library,
        &db,
        &path,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    let (_, report) = import_path(
        &mut library,
        &db,
        &path,
        &with_input(LookInput::LogC3Ei800),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.updated.len(), 1, "{report:?}");
    assert_eq!(
        library.get(&LookId::user("look")).unwrap().input,
        LookInput::LogC3Ei800
    );
    // Same name, different content: the user's first LUT is not overwritten.
    fs::write(&path, cube_text(5, 2, "look v2")).unwrap();
    let (_, report) = import_path(
        &mut library,
        &db,
        &path,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 1);
    assert_eq!(report.added[0].id.as_str(), "user:look-2");
    assert_eq!(report.added[0].file, "user/look-2.cube");
    assert_eq!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "user")
            .count(),
        2
    );
}

#[test]
fn folders_are_walked_and_junk_is_skipped() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("My LUTs");
    fs::create_dir_all(src.join("sub/deeper")).unwrap();
    fs::create_dir_all(src.join("__MACOSX")).unwrap();
    fs::write(src.join("a.cube"), cube_text(3, 1, "a")).unwrap();
    fs::write(src.join("sub/b.CUBE"), cube_text(3, 2, "b")).unwrap();
    fs::write(src.join("sub/deeper/c.cube"), cube_text(3, 3, "c")).unwrap();
    fs::write(src.join("sub/._b.cube"), b"apple double").unwrap();
    fs::write(src.join("__MACOSX/a.cube"), b"junk").unwrap();
    fs::write(src.join("readme.pdf"), b"%PDF").unwrap();
    fs::write(src.join(".DS_Store"), b"x").unwrap();
    let db = small_db(1);
    let mut library = open_library(&work.path().join("app"));
    let (recognition, report) = import_path(
        &mut library,
        &db,
        &src,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(recognition.files.len(), 3);
    assert_eq!(report.added.len(), 3);
    assert_eq!(recognition.ignored.pdf, 1);
    assert!(recognition.ignored.macos_metadata >= 3);
}

#[test]
fn originals_are_never_written_to() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("a.cube"), cube_text(3, 1, "a")).unwrap();
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 3);
    let snapshot = |path: &Path| -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut stack = vec![path.to_path_buf()];
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                stack.extend(fs::read_dir(&p).unwrap().map(|e| e.unwrap().path()));
            } else {
                let bytes = fs::read(&p).unwrap();
                let mtime = format!("{:?}", fs::metadata(&p).unwrap().modified().unwrap());
                out.push((format!("{} {mtime}", p.display()), sha256_hex(&bytes)));
            }
        }
        out.sort();
        out
    };
    let before_dir = snapshot(&src);
    let before_zip = snapshot(&zip);
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    import_path(
        &mut library,
        &db,
        &src,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    assert_eq!(snapshot(&src), before_dir);
    assert_eq!(snapshot(&zip), before_zip);
}

#[test]
fn hostile_zip_entries_are_ignored_and_nothing_escapes_the_library() {
    let work = tempfile::tempdir().unwrap();
    let zip = work.path().join("evil.zip");
    write_zip(
        &zip,
        &[
            (
                "../escape.cube".to_string(),
                cube_text(3, 1, "x").into_bytes(),
            ),
            (
                "/absolute.cube".to_string(),
                cube_text(3, 2, "y").into_bytes(),
            ),
            (
                "ok/fine.cube".to_string(),
                cube_text(3, 3, "z").into_bytes(),
            ),
        ],
    );
    let root = work.path().join("app");
    let mut library = open_library(&root);
    let (recognition, report) = import_path(
        &mut library,
        &small_db(1),
        &zip,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    // The zip writer may already have turned "/absolute.cube" into a relative name; "../escape.cube"
    // always survives as written and is refused.
    assert!(
        recognition.ignored.unsafe_entries >= 1,
        "{:?}",
        recognition.ignored
    );
    assert!(
        report.added.iter().any(|i| i.id.as_str() == "user:fine"),
        "{report:?}"
    );
    assert!(!work.path().join("escape.cube").exists());
    assert!(
        tree(&root).iter().all(|f| f.starts_with("looks/")),
        "{:?}",
        tree(&root)
    );
}

#[test]
fn a_file_name_cannot_steer_the_destination() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    fs::create_dir_all(&src).unwrap();
    // Characters that are awkward on some file systems.
    fs::write(src.join("a:b*c?.cube"), cube_text(3, 1, "x")).unwrap();
    let root = work.path().join("app");
    let mut library = open_library(&root);
    let (_, report) = import_path(
        &mut library,
        &small_db(1),
        &src,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added[0].file, "user/a_b_c_.cube");
    assert!(root.join("looks/user/a_b_c_.cube").exists());
}

#[test]
fn cancelling_leaves_the_library_untouched() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 6);
    let db = small_db(6);
    let mut library = open_library(&root);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    let token = CancelToken::new();
    let opts = ImportOptions {
        cancel: Some(token.clone()),
        ..options()
    };
    let mut cancel_after_three = |p: Progress<'_>| {
        if p.stage == Stage::Copy && p.done == 3 {
            token.cancel();
        }
    };
    let err = import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &opts,
        &mut cancel_after_three,
    )
    .unwrap_err();
    assert!(matches!(err, ImportError::Cancelled), "{err}");
    let left: Vec<_> = tree(&root)
        .into_iter()
        .filter(|f| f.ends_with(".cube") || f.contains(".tmp-") || f.ends_with("manifest.json"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
    assert!(library.looks().iter().all(|l| l.id.prefix() == "builtin"));
}

#[test]
fn a_source_that_changed_since_it_was_inspected_is_refused() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("a.cube"), cube_text(3, 1, "a")).unwrap();
    let db = small_db(1);
    let mut library = open_library(&work.path().join("app"));
    let recognition = inspect(&src, &db, None, &mut no_progress()).unwrap();
    fs::write(src.join("a.cube"), cube_text(3, 2, "a, edited")).unwrap();
    let err = import(
        &mut library,
        &db,
        &src,
        &recognition,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ImportError::SourceChanged(p) if p.ends_with("a.cube")),
        "{err}"
    );
    fs::remove_file(src.join("a.cube")).unwrap();
    let err = import(
        &mut library,
        &db,
        &src,
        &recognition,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(matches!(err, ImportError::SourceChanged(_)), "{err}");
}

#[test]
fn fingerprints_identify_a_package_whatever_its_files_are_called() {
    let work = tempfile::tempdir().unwrap();
    let mut db = small_db(3);
    // Fingerprint a synthetic LogC4 package as the HUMAN step would, then hand over a zip whose
    // names say nothing at all.
    let originals = package_files(Kind::LookLogC4, 3);
    let package = db
        .packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc4")
        .unwrap();
    package.fingerprints = originals
        .iter()
        .map(|(path, text)| Fingerprint {
            path: path.clone(),
            size: text.len() as u64,
            lut_size: 5,
            sha256: sha256_hex(text.as_bytes()),
        })
        .collect();
    let zip = work.path().join("download (1).zip");
    let renamed: Vec<(String, Vec<u8>)> = originals
        .iter()
        .enumerate()
        .map(|(i, (_, text))| (format!("x{i}.cube"), text.clone().into_bytes()))
        .collect();
    write_zip(&zip, &renamed);

    let root = work.path().join("app");
    let mut library = open_library(&root);
    let recognition = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert_eq!(
        recognition.verdict(),
        Verdict::Ready,
        "{:?}",
        recognition.issues
    );
    assert_eq!(recognition.packages[0].evidence, Evidence::Fingerprint);
    assert!(recognition.files.iter().all(|f| matches!(
        f.outcome,
        Outcome::ArriLook {
            evidence: Evidence::Fingerprint,
            ..
        }
    )));
    import(
        &mut library,
        &db,
        &zip,
        &recognition,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    // Group and name come from the fingerprinted path, not from the renamed files.
    assert!(library.get(&LookId::arri(1000, "look-01")).is_some());
    assert!(library.get(&LookId::arri(3000, "look-02")).is_some());
    // ... and so does the file name in the library.
    let files = tree(&library.looks_dir());
    assert!(
        files.contains(&"arri/1000-colour/Look_01_LogC4.cube".to_string()),
        "{files:?}"
    );

    // One file short: the missing one is named.
    let short = work.path().join("short.zip");
    write_zip(&short, &renamed[..2]);
    let recognition = inspect(&short, &db, None, &mut no_progress()).unwrap();
    assert_eq!(recognition.verdict(), Verdict::Incomplete);
    assert_eq!(
        recognition.packages[0].missing,
        vec![originals[2].0.clone()]
    );
    assert!(
        recognition
            .problems()
            .next()
            .unwrap()
            .message
            .contains("Look_03_LogC4.cube")
    );
}

#[test]
fn a_damaged_manifest_is_set_aside_and_a_new_list_started() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    fs::create_dir_all(root.join("looks")).unwrap();
    fs::write(root.join("looks/manifest.json"), "{ not json").unwrap();
    let mut library = open_library(&root);
    assert_eq!(library.problems().len(), 1);
    let path = work.path().join("a.cube");
    fs::write(&path, cube_text(3, 1, "a")).unwrap();
    let (_, report) = import_path(
        &mut library,
        &small_db(1),
        &path,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 1);
    assert_eq!(report.notes.len(), 1);
    assert!(
        report.notes[0].contains("manifest.json.damaged-1"),
        "{:?}",
        report.notes
    );
    assert!(root.join("looks/manifest.json.damaged-1").exists());
    assert!(library.problems().is_empty());
    assert_eq!(open_library(&root).looks().len(), 2);
}

#[test]
fn a_manifest_from_a_newer_version_is_never_overwritten() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    fs::create_dir_all(root.join("looks")).unwrap();
    let newer = r#"{"version": 2, "looks": []}"#;
    fs::write(root.join("looks/manifest.json"), newer).unwrap();
    let mut library = open_library(&root);
    let path = work.path().join("a.cube");
    fs::write(&path, cube_text(3, 1, "a")).unwrap();
    let err = import_path(
        &mut library,
        &small_db(1),
        &path,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(matches!(err, ImportError::Manifest(_)), "{err}");
    assert!(err.to_string().contains("newer version"), "{err}");
    assert_eq!(
        fs::read_to_string(root.join("looks/manifest.json")).unwrap(),
        newer
    );
}

#[test]
fn a_shaper_in_front_of_a_cube_is_accepted() {
    let work = tempfile::tempdir().unwrap();
    let path = work.path().join("shaped.cube");
    let mut text = String::from("TITLE \"shaped\"\nLUT_1D_SIZE 4\nLUT_3D_SIZE 2\n");
    for i in 0..4 {
        let v = f64::from(i) / 3.0;
        text.push_str(&format!("{v} {v} {v}\n"));
    }
    for b in 0..2 {
        for g in 0..2 {
            for r in 0..2 {
                text.push_str(&format!("{r} {g} {b}\n"));
            }
        }
    }
    fs::write(&path, text).unwrap();
    let mut library = open_library(&work.path().join("app"));
    let (_, report) = import_path(
        &mut library,
        &small_db(1),
        &path,
        &with_input(LookInput::LogC4),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 1);
    let lut = library.transform(&report.added[0].id).unwrap();
    assert!(lut.lut().unwrap().shaper.is_some());
}

#[test]
fn the_documented_manifest_shape_is_what_lands_on_disk() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let db = small_db(2);
    let mut library = open_library(&root);
    for kind in [Kind::LookLogC4, Kind::DisplayLogC4] {
        let zip = write_package_zip(work.path(), kind, 2);
        import_path(&mut library, &db, &zip, &options(), &mut no_progress()).unwrap();
    }
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(library.paths().manifest_path()).unwrap())
            .unwrap();
    assert_eq!(value["version"], 1);
    let look = &value["looks"][0];
    assert_eq!(look["group"]["kind"], "arri");
    assert!(look["group"]["no"].is_u64());
    assert!(look["group"]["name"].is_string());
    assert_eq!(look["input"], "logc4");
    assert!(look["file"].as_str().unwrap().starts_with("arri/"));
    assert_eq!(look["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(look["lut_size"], 5);
    assert_eq!(look["source"]["package"], "look-library-logc4");
    assert_eq!(
        look["source"]["url"],
        "https://www.arri.com/en/learn/camera-systems/image-science/look-files"
    );
    assert_eq!(look["source"]["imported"], "2026-11-02T10:14:00Z");
    assert!(
        look["source"]["original_name"]
            .as_str()
            .unwrap()
            .ends_with(".cube")
    );
    let display = &value["display_luts"][0];
    assert_eq!(display["input"], "logc4");
    assert!(
        display["id"]
            .as_str()
            .unwrap()
            .starts_with("arri-display:logc4:")
    );
    assert_eq!(
        display["target"].as_str().unwrap(),
        display["id"].as_str().unwrap().rsplit(':').next().unwrap()
    );
    assert!(
        display["file"]
            .as_str()
            .unwrap()
            .starts_with("arri-display/")
    );
}

/// The whole LogC3 look library at the real size and with the real patterns (87 files of 33
/// points, ~100 MB of text). Slow, so it is skipped by default; run with
/// `cargo test -p ym-looks --release -- --ignored full_size`.
#[test]
#[ignore = "writes about 100 MB of synthetic LUTs; run with --ignored to time the real-size case"]
fn full_size_logc3_package_with_the_real_patterns() {
    let work = tempfile::tempdir().unwrap();
    let real = {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json");
        ym_looks::recognize::PackageDb::load(&path).unwrap()
    };
    let entries: Vec<(String, Vec<u8>)> = (0..87)
        .map(|i| {
            let group = ["1000 Colour", "2000 Mood", "3000 Film"][i % 3];
            (
                format!("{group}/Look_{i:02}_LogC3.cube"),
                cube_text(33, i as u32 + 3, &format!("LogC3 Look {i}")).into_bytes(),
            )
        })
        .collect();
    let zip = work.path().join("ARRI_Look_Library_LogC3.zip");
    write_zip(&zip, &entries);
    let mut library = open_library(&work.path().join("app"));
    let total_bytes: usize = entries.iter().map(|(_, b)| b.len()).sum();
    let start = std::time::Instant::now();
    let recognition = inspect(&zip, &real, None, &mut no_progress()).unwrap();
    let inspected = start.elapsed();
    let report = import(
        &mut library,
        &real,
        &zip,
        &recognition,
        &options(),
        &mut no_progress(),
    )
    .unwrap();
    eprintln!(
        "87 x 33-point cubes ({} MB of text): inspect {:?}, import {:?}",
        total_bytes / 1_000_000,
        inspected,
        start.elapsed() - inspected
    );
    assert_eq!(
        recognition.verdict(),
        Verdict::Ready,
        "{:?}",
        recognition.issues
    );
    assert_eq!(recognition.packages[0].id, "look-library-logc3");
    assert_eq!(report.added.len(), 87);
}
