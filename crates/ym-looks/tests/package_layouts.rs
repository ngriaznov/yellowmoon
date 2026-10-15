//! The Look Library zip as ARRI ships it, in every layout it might have: the package is recognised
//! from the name of the zip, the encoding, the LUT size and the number of files, and the folder
//! layout and the words in the look names only break ties. All LUTs are our own synthetic cubes;
//! only ARRI-like **names** are used (`ARRI_LogC4-to-LogC4_<Look>_v1_65.cube`, the zip name from
//! ARRI's page), so no ARRI data is involved.

mod support;

use std::path::{Path, PathBuf};

use support::{cube_text, no_progress, open_library, small_db, tree, write_zip};
use ym_color::look::LookInput;
use ym_looks::import::{ImportError, ImportOptions, import, import_path, inspect};
use ym_looks::library::LookGroup;
use ym_looks::recognize::{Evidence, Outcome, PackageStatus, Recognition, Verdict};

/// The name of the download Mykyta's Mac got, as the wizard showed it.
const REAL_ZIP: &str = "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip";

/// ARRI's file naming: `ARRI_LogC4-to-LogC4_<Look>_v1_65.cube`.
fn arri_name(look: &str) -> String {
    format!("ARRI_LogC4-to-LogC4_{look}_v1_65.cube")
}

fn look_names(n: usize) -> Vec<String> {
    const NAMES: [&str; 12] = [
        "Neutral",
        "Warm Film",
        "Cool Film",
        "Bleach Bypass",
        "Pale to Pqrs",
        "Warm P3 Glow",
        "PQ Noir",
        "HLG Haze",
        "sRGB Soft",
        "Rec709 Neutral",
        "Teal and Orange",
        "Kodak 2383",
    ];
    NAMES.iter().take(n).map(|s| (*s).to_string()).collect()
}

/// Paths inside the zip for each layout.
fn layout(kind: &str, looks: &[String]) -> Vec<String> {
    looks
        .iter()
        .enumerate()
        .map(|(i, look)| {
            let file = arri_name(look);
            match kind {
                "flat" => file,
                "one top-level folder" => format!("ARRI Look Library LogC4 (log-to-log)/{file}"),
                "one folder per look" => format!("{look}/{file}"),
                "named groups" => format!("{}/{file}", ["Colour", "Film", "Mono"][i % 3]),
                "wrapper and named groups" => {
                    format!("3D-LUTs/{}/{file}", ["Colour", "Film", "Mono"][i % 3])
                }
                "numbered groups" => {
                    format!("{}/{file}", ["1000 Colour", "3000 Film", "4000 Mono"][i % 3])
                }
                "wrapper and numbered groups" => format!(
                    "ARRI Look Library/{}/{file}",
                    ["1000 Colour", "3000 Film", "4000 Mono"][i % 3]
                ),
                "group number in the middle" => {
                    format!("Group 3000 Film/{file}")
                }
                other => panic!("{other}"),
            }
        })
        .collect()
}

const LAYOUTS: [&str; 8] = [
    "flat",
    "one top-level folder",
    "one folder per look",
    "named groups",
    "wrapper and named groups",
    "numbered groups",
    "wrapper and numbered groups",
    "group number in the middle",
];

/// Writes a zip of synthetic 5-point cubes at `paths`, with the junk a Mac and ARRI add: ALF4
/// files, a PDF, a `.DS_Store`, and a `__MACOSX` tree of AppleDouble files called `.cube`.
fn write_looks_zip(dir: &Path, zip_name: &str, paths: &[String]) -> PathBuf {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let title = format!("LogC4 look {i}");
        entries.push((
            path.clone(),
            cube_text(5, i as u32 + 1, &title).into_bytes(),
        ));
        entries.push((path.replace(".cube", ".alf4"), b"binary look file".to_vec()));
        // What macOS adds when it zips a folder: `__MACOSX/<same path with ._ before the name>`
        // and `._name` beside the real file. Both are called `.cube` and neither is a LUT.
        let (folder, file) = path.rsplit_once('/').unwrap_or(("", path));
        let apple_double = if folder.is_empty() {
            format!("__MACOSX/._{file}")
        } else {
            format!("__MACOSX/{folder}/._{file}")
        };
        entries.push((apple_double, b"\x00\x05\x16\x07 apple double".to_vec()));
        let beside = if folder.is_empty() {
            format!("._{file}")
        } else {
            format!("{folder}/._{file}")
        };
        entries.push((beside, b"\x00\x05\x16\x07 apple double".to_vec()));
    }
    entries.push((".DS_Store".to_string(), b"junk".to_vec()));
    entries.push(("ARRI Look Library notes.pdf".to_string(), b"%PDF-1.4".to_vec()));
    let path = dir.join(zip_name);
    write_zip(&path, &entries);
    path
}

fn looks_of(r: &Recognition) -> Vec<(&str, u32, &str, &str)> {
    r.files
        .iter()
        .map(|f| match &f.outcome {
            Outcome::ArriLook {
                package,
                group_no,
                group_name,
                name,
                input,
                evidence,
                ..
            } => {
                assert_eq!(*input, LookInput::LogC4);
                assert_eq!(*evidence, Evidence::Pattern);
                (package.as_str(), *group_no, group_name.as_str(), name.as_str())
            }
            other => panic!("{}: {other:?}", f.rel_path),
        })
        .collect()
}

#[test]
fn the_real_zip_name_is_recognized_in_every_layout() {
    let n = 9;
    let looks = look_names(n);
    for kind in LAYOUTS {
        let work = tempfile::tempdir().unwrap();
        let paths = layout(kind, &looks);
        let zip = write_looks_zip(work.path(), REAL_ZIP, &paths);
        let db = small_db(n);
        let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();

        assert_eq!(r.verdict(), Verdict::Ready, "{kind}: {:?}", r.issues);
        assert_eq!(r.packages.len(), 1, "{kind}");
        let finding = &r.packages[0];
        assert_eq!(finding.id, "look-library-logc4", "{kind}");
        assert_eq!((finding.found, finding.expected), (n, Some(n)), "{kind}");
        assert_eq!(finding.status, PackageStatus::Complete, "{kind}");
        assert_eq!(finding.lut_sizes, vec![5], "{kind}");
        // Only the cubes were read: no AppleDouble file, no ALF4, no PDF.
        assert_eq!(r.files.len(), n, "{kind}");
        assert_eq!(r.ignored.alf4, n, "{kind}");
        assert_eq!(r.ignored.macos_metadata, 2 * n + 1, "{kind}");
        assert_eq!(r.ignored.pdf, 1, "{kind}");
        assert_eq!(r.needs_answer_count(), 0, "{kind}");
        assert!(r.problems().next().is_none(), "{kind}");

        let found = looks_of(&r);
        assert!(found.iter().all(|f| f.0 == "look-library-logc4"), "{kind}");
        // The look names are the looks' own: no "ARRI", no "to LogC4", no version, no mesh size.
        let mut names: Vec<&str> = found.iter().map(|f| f.3).collect();
        names.sort_unstable();
        let mut expected: Vec<&str> = looks.iter().map(String::as_str).collect();
        expected.sort_unstable();
        assert_eq!(names, expected, "{kind}");
        // Four-digit group folders keep their numbers; every other layout is read as it is.
        match kind {
            "numbered groups" | "wrapper and numbered groups" => {
                assert!(found.iter().any(|f| f.1 == 3000 && f.2 == "Film"), "{kind}");
            }
            "named groups" | "wrapper and named groups" => {
                assert!(found.iter().all(|f| f.1 == 0), "{kind}");
                let mut groups: Vec<&str> = found.iter().map(|f| f.2).collect();
                groups.sort_unstable();
                groups.dedup();
                assert_eq!(groups, ["Colour", "Film", "Mono"], "{kind}");
            }
            _ => assert!(
                found.iter().all(|f| (f.1, f.2) == (0, "Ungrouped")),
                "{kind}: {found:?}"
            ),
        }

        // And it imports, as ARRI looks with LogC4 input, grouped, with the package recorded.
        let root = work.path().join("app");
        let mut library = open_library(&root);
        let report = import(
            &mut library,
            &db,
            &zip,
            &r,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), n, "{kind}");
        let imported: Vec<_> = library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .collect();
        assert_eq!(imported.len(), n, "{kind}");
        for look in &imported {
            assert_eq!(look.input, LookInput::LogC4);
            assert!(matches!(look.group, LookGroup::Arri { .. }));
            assert_eq!(
                look.source.as_ref().and_then(|s| s.package.as_deref()),
                Some("look-library-logc4")
            );
            assert!(look.is_enabled());
        }
        assert!(library.verify(&mut no_progress()).unwrap().is_clean());
        assert_eq!(open_library(&root).looks().len(), library.looks().len());
        // The same zip again changes nothing.
        let again = import_path(
            &mut library,
            &db,
            &zip,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap()
        .1;
        assert_eq!(again.unchanged.len(), n, "{kind}");
        assert!(again.added.is_empty() && again.updated.is_empty(), "{kind}");
    }
}

#[test]
fn spellings_of_the_zip_name_all_count() {
    let looks = look_names(4);
    let paths = layout("flat", &looks);
    for name in [
        "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
        "ARRI_Look_Library_LogC4.zip",
        "ARRI-Look-Library-Log-C4.zip",
        "arri look library logc4 (log-to-log) 3d-luts_data.zip",
        "ARRI LOOK LIBRARY LOGC4 (LOG-TO-LOG) 3D-LUTS_DATA.ZIP",
        "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data (1).zip",
    ] {
        let work = tempfile::tempdir().unwrap();
        let zip = write_looks_zip(work.path(), name, &paths);
        let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::Ready, "{name}: {:?}", r.issues);
        assert_eq!(r.packages[0].id, "look-library-logc4", "{name}");
    }
    // The same folder unpacked: the folder's name is the evidence.
    let work = tempfile::tempdir().unwrap();
    let zip = write_looks_zip(work.path(), REAL_ZIP, &paths);
    let unpacked = work.path().join("ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data");
    std::fs::create_dir_all(&unpacked).unwrap();
    for (i, path) in paths.iter().enumerate() {
        std::fs::write(unpacked.join(path), cube_text(5, i as u32 + 1, "x")).unwrap();
    }
    let r = inspect(&unpacked, &small_db(4), None, &mut no_progress()).unwrap();
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages[0].id, "look-library-logc4");
    drop(zip);
}

#[test]
fn macos_junk_is_never_counted_as_a_cube_or_a_group() {
    let n = 6;
    let looks = look_names(n);
    let work = tempfile::tempdir().unwrap();
    // A zip with ONLY the Mac tree for some looks, and a wrapper folder: `__MACOSX/ARRI…/._x.cube`.
    let mut paths = layout("wrapper and named groups", &looks);
    let zip = write_looks_zip(work.path(), REAL_ZIP, &paths);
    let r = inspect(&zip, &small_db(n), None, &mut no_progress()).unwrap();
    assert_eq!(r.files.len(), n);
    assert!(
        r.files
            .iter()
            .all(|f| !f.rel_path.contains("__MACOSX") && !f.name.starts_with("._")),
        "{:?}",
        r.files.iter().map(|f| &f.rel_path).collect::<Vec<_>>()
    );
    // The groups are the real ones: neither `__MACOSX` nor the `3D-LUTs` wrapper is a group.
    let groups: std::collections::BTreeSet<&str> =
        looks_of(&r).iter().map(|f| f.2).collect();
    assert_eq!(
        groups.into_iter().collect::<Vec<_>>(),
        ["Colour", "Film", "Mono"]
    );
    // A different case of the folder name, as some tools write it.
    paths.truncate(n);
    let loud = write_zip_with(work.path(), "loud.zip", &paths, "__macosx");
    let r = inspect(&loud, &small_db(n), None, &mut no_progress()).unwrap();
    assert_eq!(r.files.len(), n);
    assert_eq!(r.ignored.macos_metadata, n);
}

fn write_zip_with(dir: &Path, name: &str, paths: &[String], junk_folder: &str) -> PathBuf {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        entries.push((path.clone(), cube_text(5, i as u32 + 1, "x").into_bytes()));
        let (folder, file) = path.rsplit_once('/').unwrap_or(("", path));
        entries.push((
            format!("{junk_folder}/{folder}/._{file}"),
            b"apple double".to_vec(),
        ));
    }
    let path = dir.join(name);
    write_zip(&path, &entries);
    path
}

#[test]
fn look_names_with_p3_pq_hlg_srgb_and_709_in_them_are_looks_not_display_renderings() {
    // Names that hold the short display words as words of their own or inside other words, and a
    // zip that says "log-to-log": none of it is a display rendering, with or without group folders.
    let looks = look_names(12);
    for kind in ["flat", "numbered groups", "named groups"] {
        let work = tempfile::tempdir().unwrap();
        let zip = write_looks_zip(work.path(), REAL_ZIP, &layout(kind, &looks));
        let r = inspect(&zip, &small_db(12), None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::Ready, "{kind}: {:?}", r.issues);
        assert_eq!(r.packages.len(), 1, "{kind}");
        assert_eq!(r.packages[0].id, "look-library-logc4", "{kind}");
        let names: Vec<&str> = looks_of(&r).iter().map(|f| f.3).collect();
        for want in ["Pale to Pqrs", "Warm P3 Glow", "PQ Noir", "HLG Haze", "sRGB Soft", "Rec709 Neutral"] {
            assert!(names.contains(&want), "{kind}: {want} in {names:?}");
        }
        assert!(
            r.files
                .iter()
                .all(|f| matches!(f.outcome, Outcome::ArriLook { .. })),
            "{kind}"
        );
    }
}

#[test]
fn the_other_packages_and_the_wvo_package_are_told_apart_by_name() {
    let looks = look_names(4);
    let flat = layout("flat", &looks);
    let by_name = |zip_name: &str, paths: &[String]| -> Recognition {
        let work = tempfile::tempdir().unwrap();
        let zip = write_looks_zip(work.path(), zip_name, paths);
        inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap()
    };

    // The LogC3 look libraries, at their own LUT size (3 points in these tests), are theirs.
    let work = tempfile::tempdir().unwrap();
    let mut entries = Vec::new();
    for (i, path) in flat.iter().enumerate() {
        let path = path.replace("LogC4", "LogC3");
        entries.push((path, cube_text(3, i as u32 + 1, "x").into_bytes()));
    }
    let zip = work.path().join("ARRI Look Library LogC3 (log-to-log) 3D-LUTs.zip");
    write_zip(&zip, &entries);
    let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages[0].id, "look-library-logc3");
    let zip = work.path().join("ARRI Look Library LogC3 to Rec709 3D-LUTs.zip");
    write_zip(&zip, &entries);
    let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
    assert_eq!(r.packages[0].id, "look-library-logc3-rec709");

    // A LogC4 name over LogC3-sized cubes (3 points) is not the LogC4 package.
    let zip = work.path().join("ARRI Look Library LogC4 (log-to-log) 3D-LUTs.zip");
    write_zip(&zip, &entries);
    let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
    assert!(r.packages.is_empty(), "{:?}", r.packages);
    assert_eq!(r.verdict(), Verdict::NeedsAnswer);

    // The small WVO package (LogC4 to LogC4, for wireless video) is not a look package or the
    // display package, however many files it holds.
    for name in [
        "ARRI LogC4 WVO LUT Package.zip",
        "LogC4 WVO LUT Package.zip",
        "ARRI Look Library LogC4 WVO.zip",
    ] {
        let wvo: Vec<String> = ["33", "65"]
            .iter()
            .flat_map(|mesh| {
                (0..4).map(move |i| format!("ARRI_LogC4-to-LogC4_WVO{i}_v1_{mesh}.cube"))
            })
            .collect();
        let r = by_name(name, &wvo);
        assert!(r.packages.is_empty(), "{name}: {:?}", r.packages);
        assert_eq!(r.verdict(), Verdict::NeedsAnswer, "{name}");
    }

    // A name that says nothing: the generic question, with what the file names suggest.
    let r = by_name("holiday.zip", &flat);
    assert!(r.packages.is_empty());
    assert_eq!(r.verdict(), Verdict::NeedsAnswer);
    assert_eq!(r.needs_answer_count(), 4);
}

#[test]
fn display_luts_named_the_way_arri_names_them() {
    // `ARRI_LogC4-to-Gamma24_Rec709-D65_v1_33.cube` and `_65.cube` of one target are two files.
    let work = tempfile::tempdir().unwrap();
    let names = [
        "ARRI_LogC4-to-Gamma24_Rec709-D65_v1_33.cube",
        "ARRI_LogC4-to-Gamma24_Rec709-D65_v1_65.cube",
        "ARRI_LogC4-to-HLG_1K_Rec2020-D65_DW203_v1_33.cube",
        "ARRI_LogC4-to-HLG_1K_Rec2020-D65_DW203_v1_65.cube",
    ];
    let entries: Vec<(String, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.to_string(), cube_text(5, i as u32 + 1, "x").into_bytes()))
        .collect();
    for zip_name in ["ARRI LogC4 LUT Package.zip", "ARRI Display LUTs LogC4.zip"] {
        let zip = work.path().join(zip_name);
        write_zip(&zip, &entries);
        let db = small_db(4);
        let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::Ready, "{zip_name}: {:?}", r.issues);
        assert_eq!(r.packages[0].id, "display-luts-logc4", "{zip_name}");
        assert!(r
            .files
            .iter()
            .all(|f| matches!(f.outcome, Outcome::ArriDisplay { .. })));
        let mut library = open_library(&work.path().join(format!("app-{zip_name}")));
        let report = import(
            &mut library,
            &db,
            &zip,
            &r,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), 4, "{zip_name}: both meshes of a target are kept");
        assert_eq!(library.display_luts().len(), 4);
    }
}

#[test]
fn fewer_files_than_expected_is_an_incomplete_download_and_more_is_a_note() {
    let looks = look_names(8);
    let flat = layout("flat", &looks);
    let work = tempfile::tempdir().unwrap();
    // Expected 8: 6 is "incomplete" with the numbers (at least half are there)...
    let zip = write_looks_zip(work.path(), REAL_ZIP, &flat[..6]);
    let r = inspect(&zip, &small_db(8), None, &mut no_progress()).unwrap();
    assert_eq!(r.verdict(), Verdict::Incomplete, "{:?}", r.issues);
    assert_eq!(r.packages[0].status, PackageStatus::Incomplete);
    let message = &r.problems().next().unwrap().message;
    assert!(message.contains("should have 8 files, but this zip has 6"), "{message}");
    // ...3 of 8 is not read as the package at all...
    let zip = write_looks_zip(work.path(), "ARRI Look Library LogC4 small.zip", &flat[..3]);
    let r = inspect(&zip, &small_db(8), None, &mut no_progress()).unwrap();
    assert!(r.packages.is_empty());
    // ...and 8 files for a package that is now expected to have 6 are taken, with a note.
    let zip = write_looks_zip(work.path(), "ARRI Look Library LogC4 more.zip", &flat);
    let r = inspect(&zip, &small_db(6), None, &mut no_progress()).unwrap();
    assert_eq!(r.verdict(), Verdict::Ready);
    assert_eq!(r.packages[0].status, PackageStatus::MoreThanExpected);
    assert_eq!(r.packages[0].found, 8);
}

#[test]
fn an_unknown_set_with_an_answer_imports_as_the_users_own_luts_with_that_input() {
    // The fallback when nothing is recognised (a renamed zip, a layout nobody foresaw): the
    // generic question, and then every file is imported with the answer.
    let n = 5;
    let looks = look_names(n);
    for answer in [
        LookInput::LogC4,
        LookInput::LogC3Ei800,
        LookInput::LogC3ToRec709,
    ] {
        let work = tempfile::tempdir().unwrap();
        let zip = write_looks_zip(work.path(), "my picks.zip", &layout("flat", &looks));
        let db = small_db(n);
        let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::NeedsAnswer);
        assert_eq!(r.needs_answer_count(), n);
        assert!(r.packages.is_empty());

        let mut library = open_library(&work.path().join("app"));
        let err = import(
            &mut library,
            &db,
            &zip,
            &r,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap_err();
        assert!(matches!(err, ImportError::NeedsInput { .. }), "{err}");
        assert!(tree(&library.looks_dir()).is_empty(), "nothing written without an answer");

        let report = import(
            &mut library,
            &db,
            &zip,
            &r,
            &ImportOptions {
                input: Some(answer),
                ..ImportOptions::default()
            },
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), n);
        assert!(report.rejected.is_empty());
        let user: Vec<_> = library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "user")
            .collect();
        assert_eq!(user.len(), n);
        for look in user {
            assert_eq!(look.input, answer);
            assert!(matches!(look.group, LookGroup::User));
            assert!(library.transform(&look.id).is_ok());
        }
        assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    }
}
