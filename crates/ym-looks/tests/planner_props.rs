//! Property tests for the importer's planning: whatever the file and folder names are, an import
//! either succeeds and leaves a library that opens without problems and verifies clean, or fails
//! without leaving anything behind. Names are random Unicode (no `/` or NUL); the cases use a
//! fixed seed, so every run sees the same ones.

mod support;

use std::collections::BTreeSet;
use std::fs;

use proptest::prelude::*;
use support::{cube_text, no_progress, open_library, small_db, tree};
use ym_color::look::LookInput;
use ym_looks::import::{ImportOptions, import_path};
use ym_looks::library::LookGroup;
use ym_looks::recognize::Verdict;

fn options() -> ImportOptions {
    ImportOptions {
        input: Some(LookInput::LogC4),
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

/// A file or folder name part: any characters but `/` and NUL.
fn name_part() -> impl Strategy<Value = String> {
    "[^/\\x00]{1,40}"
}

fn check_library(root: &std::path::Path, expected_looks: usize) -> Result<(), TestCaseError> {
    let mut library = open_library(root);
    prop_assert!(library.problems().is_empty(), "{:?}", library.problems());
    let imported: Vec<_> = library
        .looks()
        .iter()
        .filter(|l| l.id.prefix() != "builtin")
        .cloned()
        .collect();
    prop_assert_eq!(imported.len(), expected_looks);
    for look in &imported {
        let lut = look.lut.as_ref().unwrap();
        prop_assert!(lut.path.is_file(), "{}", lut.path.display());
        let leaf = lut.path.file_name().unwrap().to_string_lossy().into_owned();
        prop_assert!(leaf.len() <= 200, "{} bytes", leaf.len());
        prop_assert!(!look.name.trim().is_empty(), "blank name for {}", look.id);
    }
    let report = library.verify(&mut no_progress()).unwrap();
    prop_assert!(report.is_clean(), "{:?}", report.entries);
    prop_assert!(tree(root).iter().all(|f| !f.contains(".tmp-")));
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x594d_4c4f_4f4b_5334),
        ..ProptestConfig::default()
    })]

    /// The user's own `.cube` files, with any names (and several with the same name in different
    /// folders): all import, none overwrites another.
    #[test]
    fn any_user_file_names_import_cleanly(
        stems in proptest::collection::vec(name_part(), 1..5),
        folders in proptest::collection::vec(name_part(), 1..3),
    ) {
        let work = tempfile::tempdir().unwrap();
        let src = work.path().join("src");
        let mut written = 0_usize;
        // The same name twice would be one file on disk.
        let stems: Vec<String> = stems.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
        let folders: Vec<String> = folders.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
        for (i, stem) in stems.iter().enumerate() {
            for (j, folder) in folders.iter().enumerate() {
                let dir = src.join(folder);
                // Some random names cannot be created on this file system; skip those.
                if fs::create_dir_all(&dir).is_err() {
                    continue;
                }
                // A leading "L": names starting "._" are macOS metadata, which the importer skips on purpose.
                let path = dir.join(format!("L{stem}.cube"));
                if fs::write(&path, cube_text(3, (i * 7 + j) as u32 + 1, "t")).is_ok() {
                    written += 1;
                }
            }
        }
        prop_assume!(written > 0);
        let root = work.path().join("app");
        let mut library = open_library(&root);
        let result = import_path(&mut library, &small_db(1), &src, &options(), &mut no_progress());
        let (_, report) = result.map_err(|e| TestCaseError::fail(e.to_string()))?;
        // Files that were the same bytes under the same id count once; the rest each got an entry.
        prop_assert_eq!(report.added.len() + report.unchanged.len(), written);
        check_library(&root, report.added.len())?;
        // Importing again changes nothing.
        let (_, again) = import_path(&mut library, &small_db(1), &src, &options(), &mut no_progress())
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(again.added.is_empty() && again.updated.is_empty(), "{again:?}");
    }

    /// A package laid out like ARRI's (four-digit group folders, encoding in the zip or folder
    /// name), with random group and look names: groups and names survive, ids are unique, files
    /// stay inside the library.
    #[test]
    fn any_group_and_look_names_import_cleanly(
        groups in proptest::collection::vec(name_part(), 1..3),
        looks in proptest::collection::vec(name_part(), 1..4),
    ) {
        let work = tempfile::tempdir().unwrap();
        let src = work.path().join("ARRI_Look_Library_LogC4");
        let mut written = 0_usize;
        let looks: Vec<String> = looks.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
        for (g, group) in groups.iter().enumerate() {
            let dir = src.join(format!("{} {group}", 1000 + g * 1000));
            if fs::create_dir_all(&dir).is_err() {
                continue;
            }
            for (l, look) in looks.iter().enumerate() {
                let path = dir.join(format!("L{look}_LogC4.cube"));
                if fs::write(&path, cube_text(5, (g * 13 + l) as u32 + 1, "t")).is_ok() {
                    written += 1;
                }
            }
        }
        prop_assume!(written > 0);
        let root = work.path().join("app");
        let mut library = open_library(&root);
        let result = import_path(&mut library, &small_db(written), &src, &options(), &mut no_progress());
        let (_, report) = result.map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert_eq!(report.added.len() + report.unchanged.len(), written);
        check_library(&root, report.added.len())?;
        for look in library.looks() {
            if let LookGroup::Arri { group_no, .. } = &look.group {
                prop_assert!(*group_no % 1000 == 0 && *group_no >= 1000, "{group_no}");
            }
        }
    }
}

// ---- Recognition on arbitrary facts ---------------------------------------------------------

use ym_looks::library::LookFormat;
use ym_looks::recognize::{
    CubeSummary, FileFacts, Fingerprint, Ignored, LookFileFacts, Outcome, SourceInfo, SourceKind,
    recognize,
};

/// A path made of pieces that matter to the rules (group folders, encodings, display words) and
/// pieces that do not.
fn path_strategy() -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        Just("3000 Film".to_string()),
        Just("1000".to_string()),
        Just("LogC4".to_string()),
        Just("LogC3_to_Rec709".to_string()),
        Just("to".to_string()),
        Just("Gamma24_Rec709".to_string()),
        Just("P3 D65".to_string()),
        Just("ARRI".to_string()),
        Just("Film Warm".to_string()),
        "[ -~]{1,12}",
    ];
    (
        proptest::collection::vec(piece.clone(), 0..3),
        piece,
        prop_oneof![Just(".cube"), Just(".CUBE")],
    )
        .prop_map(|(dirs, name, ext)| {
            let mut path = dirs.join("/");
            if !path.is_empty() {
                path.push('/');
            }
            path.push_str(&name.replace('/', "-"));
            path.push_str(ext);
            path
        })
}

fn facts_strategy() -> impl Strategy<Value = FileFacts> {
    (
        path_strategy(),
        0_u8..6,
        prop_oneof![
            Just(None),
            Just(Some("LogC4 look".to_string())),
            "[ -~]{0,20}".prop_map(Some)
        ],
        prop_oneof![Just(5_usize), Just(3), Just(33), Just(65), 2_usize..130],
        any::<bool>(),
        prop_oneof![
            Just(LookFormat::Cube),
            Just(LookFormat::Cube),
            Just(LookFormat::Alf4),
            Just(LookFormat::Alf4c),
            Just(LookFormat::Aml)
        ],
    )
        .prop_map(
            |(rel_path, hash, title, lut_size, valid, format)| FileFacts {
                // A look file has a look-file name, and no TITLE.
                rel_path: if format.is_cube() {
                    rel_path
                } else {
                    rel_path.replace(".cube", &format!(".{}", format.extension()))
                },
                size: 1000,
                sha256: format!("{hash:02x}").repeat(32),
                format,
                check: if valid {
                    Ok(CubeSummary {
                        title: if format.is_cube() { title } else { None },
                        lut_size,
                        has_shaper: false,
                    })
                } else {
                    Err("It is not a LUT.".to_string())
                },
                look_file: (valid && format.is_look_file()).then_some(LookFileFacts {
                    input: if format == LookFormat::Aml {
                        LookInput::LogC3ToRec709
                    } else {
                        LookInput::LogC4
                    },
                    display_transforms_ignored: format == LookFormat::Alf4c,
                }),
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x594d_4c4f_4f4b_5335),
        ..ProptestConfig::default()
    })]

    /// Whatever the facts, recognition returns one result per file, in order, with consistent
    /// counts, usable names, valid display targets, and package findings that add up. Run with
    /// the real package list, with and without fingerprints (including two fingerprints that
    /// share a hash).
    #[test]
    fn recognition_is_consistent_for_arbitrary_facts(
        files in proptest::collection::vec(facts_strategy(), 0..9),
        source_name in prop_oneof![
            Just("ARRI_Look_Library_LogC4.zip".to_string()),
            Just("ARRI_Look_Library_LogC3_to_Rec709.zip".to_string()),
            Just("ARRI_Display_LUTs_LogC4.zip".to_string()),
            "[ -~]{1,20}",
        ],
        kind in prop_oneof![Just(SourceKind::Zip), Just(SourceKind::Folder), Just(SourceKind::File)],
        fingerprinted in any::<bool>(),
    ) {
        let mut db = small_db(3);
        if fingerprinted {
            for (n, package) in db.packages.iter_mut().enumerate() {
                package.fingerprints = (0..3_u8)
                    .map(|i| Fingerprint {
                        path: format!("{} G{n}/Look_{i}.cube", 1000 + u32::from(i) * 1000),
                        size: 1000,
                        lut_size: 5,
                        // Hashes 0, 1 and 2 of every package: shared between packages, and file
                        // 2 repeats file 1's hash within the package.
                        sha256: format!("{:02x}", i.min(1)).repeat(32),
                    })
                    .collect();
            }
        }
        let source = SourceInfo { kind, name: source_name };
        let recognition = recognize(&db, &source, &files, &Ignored::default());

        prop_assert_eq!(recognition.files.len(), files.len());
        for (out, input) in recognition.files.iter().zip(&files) {
            prop_assert_eq!(&out.rel_path, &input.rel_path);
            match &out.outcome {
                Outcome::ArriLook { name, group_no, .. } => {
                    prop_assert!(!name.trim().is_empty());
                    prop_assert!(*group_no <= 9999);
                    prop_assert!(input.check.is_ok());
                }
                Outcome::ArriDisplay { .. } => {
                    prop_assert!(input.check.is_ok());
                }
                Outcome::OwnLookFile { .. } => {
                    prop_assert!(input.look_file.is_some());
                }
                Outcome::NeedsAnswer { .. } => {
                    prop_assert!(input.check.is_ok() && input.format.is_cube());
                }
                Outcome::Rejected { .. } => prop_assert!(input.check.is_err()),
            }
        }
        prop_assert_eq!(
            recognition.recognized_count()
                + recognition.needs_answer_count()
                + recognition.display_lut_count()
                + recognition.rejected_count(),
            files.len()
        );
        for finding in &recognition.packages {
            prop_assert!(finding.found <= files.len(), "{finding:?}");
            if fingerprinted {
                prop_assert_eq!(finding.expected, Some(3));
                prop_assert!(finding.found <= 3);
                prop_assert_eq!(finding.found + finding.missing.len(), 3, "{:?}", finding);
            }
        }
        // The verdict is Blocked exactly when nothing is usable.
        let usable = recognition.recognized_count() + recognition.needs_answer_count();
        prop_assert_eq!(recognition.verdict() == Verdict::Blocked, usable == 0);
    }
}
