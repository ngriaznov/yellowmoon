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
                    format!(
                        "{}/{file}",
                        ["1000 Colour", "3000 Film", "4000 Mono"][i % 3]
                    )
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

/// Writes a zip of synthetic 5-point cubes at `paths`, with the junk a Mac and ARRI add: `.xml`
/// files (ARRI's older look files, never read), a PDF, a `.DS_Store`, and a `__MACOSX` tree of
/// AppleDouble files called `.cube`.
fn write_looks_zip(dir: &Path, zip_name: &str, paths: &[String]) -> PathBuf {
    write_sized_looks_zip(dir, zip_name, paths, 5)
}

/// [`write_looks_zip`] with cubes of `size` points.
fn write_sized_looks_zip(dir: &Path, zip_name: &str, paths: &[String], size: usize) -> PathBuf {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let title = format!("synthetic look {i}");
        entries.push((
            path.clone(),
            cube_text(size, i as u32 + 1, &title).into_bytes(),
        ));
        entries.push((path.replace(".cube", ".xml"), b"<adicam/>".to_vec()));
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
    entries.push((
        "ARRI Look Library notes.pdf".to_string(),
        b"%PDF-1.4".to_vec(),
    ));
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
                (
                    package.as_str(),
                    *group_no,
                    group_name.as_str(),
                    name.as_str(),
                )
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
        // Only the cubes were read: no AppleDouble file, no .xml, no PDF.
        assert_eq!(r.files.len(), n, "{kind}");
        assert_eq!(r.ignored.xml, n, "{kind}");
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
    let unpacked = work
        .path()
        .join("ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data");
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
    let groups: std::collections::BTreeSet<&str> = looks_of(&r).iter().map(|f| f.2).collect();
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
        for want in [
            "Pale to Pqrs",
            "Warm P3 Glow",
            "PQ Noir",
            "HLG Haze",
            "sRGB Soft",
            "Rec709 Neutral",
        ] {
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
    let zip = work
        .path()
        .join("ARRI Look Library LogC3 (log-to-log) 3D-LUTs.zip");
    write_zip(&zip, &entries);
    let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages[0].id, "look-library-logc3");
    let zip = work
        .path()
        .join("ARRI Look Library LogC3 to Rec709 3D-LUTs.zip");
    write_zip(&zip, &entries);
    let r = inspect(&zip, &small_db(4), None, &mut no_progress()).unwrap();
    assert_eq!(r.packages[0].id, "look-library-logc3-rec709");

    // A LogC4 name over LogC3-sized cubes (3 points) is not the LogC4 package.
    let zip = work
        .path()
        .join("ARRI Look Library LogC4 (log-to-log) 3D-LUTs.zip");
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
        // Recognised as the display package by its name (all four files, both meshes of a
        // target), and said to hold nothing to import: no package is listed, no entry is made.
        assert_eq!(r.verdict(), Verdict::Blocked, "{zip_name}: {:?}", r.issues);
        assert!(r.packages.is_empty(), "{zip_name}: {:?}", r.packages);
        assert_eq!(r.display_lut_count(), 4, "{zip_name}");
        assert!(
            r.files
                .iter()
                .all(|f| matches!(&f.outcome, Outcome::ArriDisplay { package, .. } if package == "display-luts-logc4")),
            "{zip_name}"
        );
        let mut library = open_library(&work.path().join(format!("app-{zip_name}")));
        let err = import(
            &mut library,
            &db,
            &zip,
            &r,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("only display renderings"), "{err}");
        assert!(library.looks().iter().all(|l| l.id.prefix() == "builtin"));
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
    assert!(
        message.contains("should have 8 files, but this zip has 6"),
        "{message}"
    );
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
        assert!(
            tree(&library.looks_dir()).is_empty(),
            "nothing written without an answer"
        );

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

#[test]
fn the_downloads_are_known_by_the_names_on_arris_page() {
    let db = small_db(4);
    let ids = |name: &str| -> Vec<String> {
        db.packages_for_zip_name(name)
            .iter()
            .map(|p| p.id.clone())
            .collect()
    };
    assert_eq!(
        ids("ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip"),
        ["look-library-logc4"]
    );
    assert_eq!(
        ids("Look Library for LogC4 (3D-LUTs).zip"),
        ["look-library-logc4"]
    );
    assert_eq!(
        ids("ARRI Look Library LogC3 (3D-LUTs).zip"),
        ["look-library-logc3"]
    );
    assert_eq!(
        ids("ARRI Look Library LogC3 (to Rec709) 3D-LUTs.zip"),
        ["look-library-logc3-rec709"]
    );
    assert_eq!(ids("LogC4 LUT Package.zip"), ["display-luts-logc4"]);
    assert_eq!(ids("LogC3 LUT Package.zip"), ["display-luts-logc3"]);
    // The camera look files on the same page are the look-file packages (and never a 3D-LUT one),
    // and the small WVO package is none of them.
    assert_eq!(
        ids("Look Library for LogC4 (ALF4 Look Files).zip"),
        ["look-library-logc4-alf4"]
    );
    assert_eq!(
        ids("ARRI Look Library LogC3 (Look Files).zip"),
        ["look-library-logc3-rec709-alf2"]
    );
    assert!(ids("LogC4 WVO LUT Package.zip").is_empty());
    // The page gives no LUT size for the LogC3 log-to-log library, and no sizes for downloads.
    let real = ym_looks::recognize::PackageDb::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json"),
    )
    .unwrap();
    assert!(
        real.package("look-library-logc3")
            .unwrap()
            .expected
            .lut_sizes
            .is_empty()
    );
    assert_eq!(
        real.package("look-library-logc3-rec709")
            .unwrap()
            .expected
            .lut_sizes,
        [33]
    );
    assert_eq!(
        real.package("look-library-logc4")
            .unwrap()
            .expected
            .lut_sizes,
        [65]
    );
    assert_eq!(
        real.package("look-library-logc4").unwrap().download_title(),
        "Look Library for LogC4 (3D-LUTs)"
    );
}

/// The three real 3D-LUT shapes: each name, the encoding it expects, and the file names it holds.
///
/// The last number is the LUT size `small_db` gives that package (5 for LogC4, 3 for the LogC3 ones).
type Shape = (&'static str, LookInput, usize, fn(&str) -> String);

fn real_shapes() -> [Shape; 3] {
    [
        (REAL_ZIP, LookInput::LogC4, 5, |l| {
            format!("ARRI_LogC4-to-LogC4_{l}_v1_65.cube")
        }),
        (
            "ARRI Look Library LogC3 (3D-LUTs).zip",
            LookInput::LogC3Ei800,
            3,
            |l| format!("ARRI_LogC3-to-LogC3_{l}_v1_33.cube"),
        ),
        (
            "ARRI Look Library LogC3 (to Rec709) 3D-LUTs.zip",
            LookInput::LogC3ToRec709,
            3,
            |l| format!("ARRI_LogC3-to-Rec709_{l}_v1_33.cube"),
        ),
    ]
}

#[test]
fn pressing_import_works_for_each_real_zip_shape_even_when_nothing_is_recognized() {
    // The cubes are 9-point here, which none of the packages (5 in these tests) allows, so every
    // file asks the question. The wizard preselects what the names suggest; importing with that
    // answer must import every look, whatever the layout and whichever of the three it is.
    let looks = look_names(6);
    for (zip_name, answer, _, name_of) in real_shapes() {
        for kind in [
            "flat",
            "one top-level folder",
            "one folder per look",
            "named groups",
        ] {
            let paths: Vec<String> = looks
                .iter()
                .enumerate()
                .map(|(i, look)| {
                    let file = name_of(look);
                    match kind {
                        "flat" => file,
                        "one top-level folder" => format!("Library/{file}"),
                        "one folder per look" => format!("{look}/{file}"),
                        _ => format!("{}/{file}", ["Colour", "Film", "Mono"][i % 3]),
                    }
                })
                .collect();
            let work = tempfile::tempdir().unwrap();
            let zip = write_sized_looks_zip(work.path(), zip_name, &paths, 9);
            let db = small_db(6);
            let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();
            let label = format!("{zip_name} / {kind}");
            assert_eq!(r.verdict(), Verdict::NeedsAnswer, "{label}: {:?}", r.issues);
            assert_eq!(r.needs_answer_count(), 6, "{label}");
            // What the wizard preselects.
            let suggestions: std::collections::BTreeSet<String> = r
                .files
                .iter()
                .map(|f| match &f.outcome {
                    Outcome::NeedsAnswer {
                        suggested_input, ..
                    } => format!("{suggested_input:?}"),
                    other => panic!("{label}: {other:?}"),
                })
                .collect();
            assert_eq!(
                suggestions.into_iter().collect::<Vec<_>>(),
                [format!("{:?}", Some(answer))],
                "{label}"
            );

            let mut library = open_library(&work.path().join("app"));
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
            assert_eq!(report.added.len(), 6, "{label}");
            let user: Vec<_> = library
                .looks()
                .iter()
                .filter(|l| l.id.prefix() == "user")
                .collect();
            assert_eq!(user.len(), 6, "{label}");
            assert!(user.iter().all(|l| l.input == answer), "{label}");
            assert!(library.verify(&mut no_progress()).unwrap().is_clean());
        }
    }
}

#[test]
fn each_real_zip_shape_is_recognized_when_it_is_what_its_name_says() {
    // The same shapes, at the size the packages allow: recognised, no question at all.
    let looks = look_names(6);
    for (zip_name, answer, size, name_of) in real_shapes() {
        let paths: Vec<String> = looks.iter().map(|l| name_of(l)).collect();
        let work = tempfile::tempdir().unwrap();
        let zip = write_sized_looks_zip(work.path(), zip_name, &paths, size);
        let r = inspect(&zip, &small_db(6), None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::Ready, "{zip_name}: {:?}", r.issues);
        assert_eq!(r.packages.len(), 1, "{zip_name}");
        let input = match &r.files[0].outcome {
            Outcome::ArriLook { input, .. } => *input,
            other => panic!("{zip_name}: {other:?}"),
        };
        assert_eq!(input, answer, "{zip_name}");
    }
}

#[test]
fn the_real_count_of_87_looks_in_nine_groups_is_recognized_in_any_layout() {
    // 87 looks like the real package, at 5 points so the test stays small.
    let looks: Vec<String> = (1..=87).map(|i| format!("Look {i:02}")).collect();
    for kind in [
        "flat",
        "one top-level folder",
        "one folder per look",
        "nine named groups",
    ] {
        let paths: Vec<String> = looks
            .iter()
            .enumerate()
            .map(|(i, look)| {
                let file = arri_name(look);
                match kind {
                    "flat" => file,
                    "one top-level folder" => format!("ARRI Look Library LogC4/{file}"),
                    "one folder per look" => format!("{look}/{file}"),
                    _ => format!("Group {}/{file}", i % 9 + 1),
                }
            })
            .collect();
        let work = tempfile::tempdir().unwrap();
        let zip = write_looks_zip(work.path(), REAL_ZIP, &paths);
        let r = inspect(&zip, &small_db(87), None, &mut no_progress()).unwrap();
        assert_eq!(r.verdict(), Verdict::Ready, "{kind}: {:?}", r.issues);
        assert_eq!(r.packages[0].id, "look-library-logc4", "{kind}");
        assert_eq!(
            (r.packages[0].found, r.packages[0].expected),
            (87, Some(87))
        );
        assert_eq!(r.packages[0].status, PackageStatus::Complete);
        assert_eq!(r.needs_answer_count(), 0);
        let mut groups: Vec<&str> = looks_of(&r).iter().map(|f| f.2).collect();
        groups.sort_unstable();
        groups.dedup();
        match kind {
            "nine named groups" => assert_eq!(groups.len(), 9, "{groups:?}"),
            _ => assert_eq!(groups, ["Ungrouped"], "{kind}"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ARRI's real layouts, rebuilt from made-up looks (names and a few bytes only; see the header).
// ---------------------------------------------------------------------------------------------

/// Look codes and titles in the shape of ARRI's Look Library (a four-digit code, the first digit
/// the group, then a title), with titles of our own. Two looks share a title, as ARRI's do.
const CODED: [(u32, &str); 9] = [
    (1110, "Gray Test"),
    (1111, "Gray Test"),
    (1120, "Gray Test - Night"),
    (2110, "Studio Sample"),
    (3110, "Sample A"),
    (3111, "Sample B"),
    (4130, "Tinted - Sample"),
    (8120, "20s"),
    (9211, "Dusk Sample"),
];

/// A real download: the zip's name, the one folder inside it, the input its looks have, the LUT
/// size `small_db` allows, and whether file names use hyphens (the LogC3 to Rec709 package does).
struct RealShape {
    zip: &'static str,
    folder: &'static str,
    input: LookInput,
    size: usize,
    hyphens: bool,
    package: &'static str,
}

const SHAPES: [RealShape; 3] = [
    RealShape {
        zip: "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
        folder: "ARRI Look Library for LogC4 (3D-LUTs)",
        input: LookInput::LogC4,
        size: 5,
        hyphens: false,
        package: "look-library-logc4",
    },
    RealShape {
        zip: "ARRI Look Library LogC3 (log-to-log) 3D LUTs_data.zip",
        folder: "ARRI Look Library LogC3 (log-to-log)",
        input: LookInput::LogC3Ei800,
        size: 3,
        hyphens: false,
        package: "look-library-logc3",
    },
    RealShape {
        zip: "ARRI Look Library LogC3 (to Rec709) 3D-LUTs_data.zip",
        folder: "ARRI Look Library LogC3 to Rec709 3D-LUTs",
        input: LookInput::LogC3ToRec709,
        size: 3,
        hyphens: true,
        package: "look-library-logc3-rec709",
    },
];

/// The entries of a real-shaped zip: one top folder with flat `.cube` files, no group folders and
/// no encoding word in the file names, plus a `__MACOSX` tree of `._` files called `.cube`.
fn real_entries(shape: &RealShape) -> Vec<(String, Vec<u8>)> {
    let mut entries = vec![(format!("{}/", shape.folder), Vec::new())];
    for (i, (code, title)) in CODED.iter().enumerate() {
        let file = if shape.hyphens {
            format!(
                "{code}-{}.cube",
                title.replace(" - ", " ").replace(' ', "-")
            )
        } else {
            format!("{code} {title}.cube")
        };
        entries.push((
            format!("{}/{file}", shape.folder),
            cube_text(shape.size, i as u32 + 1, "made up").into_bytes(),
        ));
        entries.push((
            format!("__MACOSX/{}/._{file}", shape.folder),
            b"\x00\x05\x16\x07 apple double".to_vec(),
        ));
    }
    entries.push((
        format!("__MACOSX/._{}", shape.folder),
        b"apple double".to_vec(),
    ));
    entries
}

/// Writes the entries as a folder tree (what Finder makes from the zip).
fn write_tree(root: &Path, entries: &[(String, Vec<u8>)]) {
    for (path, bytes) in entries {
        let target = root.join(path.trim_end_matches('/'));
        if path.ends_with('/') {
            std::fs::create_dir_all(&target).unwrap();
        } else {
            std::fs::create_dir_all(target.parent().unwrap()).unwrap();
            std::fs::write(&target, bytes).unwrap();
        }
    }
}

#[test]
fn the_real_layouts_are_recognized_as_zips_and_as_unzipped_folders() {
    for shape in &SHAPES {
        let entries = real_entries(shape);
        let work = tempfile::tempdir().unwrap();
        let real_zip = work.path().join(shape.zip);
        write_zip(&real_zip, &entries);
        let renamed = work.path().join("download.zip");
        write_zip(&renamed, &entries);
        // Finder unzips into a folder named like the zip; the person may drop that or the one inside.
        let outer = work.path().join(shape.zip.trim_end_matches(".zip"));
        write_tree(&outer, &entries);
        let inner = outer.join(shape.folder);
        let db = small_db(CODED.len());

        for (form, source) in [
            ("zip", &real_zip),
            ("renamed zip", &renamed),
            ("unzipped folder", &outer),
            ("folder inside it", &inner),
        ] {
            let label = format!("{} / {form}", shape.zip);
            let r = inspect(source, &db, None, &mut no_progress()).unwrap();
            assert_eq!(r.verdict(), Verdict::Ready, "{label}: {:?}", r.issues);
            assert_eq!(r.packages.len(), 1, "{label}");
            assert_eq!(r.packages[0].id, shape.package, "{label}");
            assert_eq!(r.packages[0].found, CODED.len(), "{label}");
            assert_eq!(r.files.len(), CODED.len(), "junk is never a look: {label}");
            assert_eq!(r.needs_answer_count(), 0, "{label}");
            // A zip counts each of the Mac's files, a folder walk skips the `__MACOSX` folder whole.
            // The folder inside has no `__MACOSX` next to it.
            assert!(
                form == "folder inside it" || r.ignored.macos_metadata >= 1,
                "{label}: {:?}",
                r.ignored
            );

            // Groups from the first digit of the look code; names without the code; the two
            // looks that share a title keep their code.
            let found: Vec<(u32, &str, &str)> = r
                .files
                .iter()
                .map(|f| match &f.outcome {
                    Outcome::ArriLook {
                        group_no,
                        group_name,
                        name,
                        input,
                        ..
                    } => {
                        assert_eq!(*input, shape.input, "{label}");
                        (*group_no, group_name.as_str(), name.as_str())
                    }
                    other => panic!("{label}: {other:?}"),
                })
                .collect();
            let mut groups: Vec<(u32, &str)> = found.iter().map(|f| (f.0, f.1)).collect();
            groups.dedup();
            assert_eq!(
                groups,
                [
                    (1000, "Black and White"),
                    (2000, "Application"),
                    (3000, "Film"),
                    (4000, "Tinted"),
                    (8000, "Period"),
                    (9000, "Special")
                ],
                "{label}"
            );
            assert_eq!(r.packages[0].groups, 6, "{label}");
            let mut names: Vec<&str> = found.iter().map(|f| f.2).collect();
            names.sort_unstable();
            assert_eq!(
                names,
                [
                    "20s",
                    "Dusk Sample",
                    "Gray Test 1110",
                    "Gray Test 1111",
                    "Gray Test Night",
                    "Sample A",
                    "Sample B",
                    "Studio Sample",
                    "Tinted Sample"
                ],
                "{label}"
            );
        }

        // And the zip imports: every look, grouped, with this package's encoding.
        let mut library = open_library(&work.path().join("app"));
        let r = inspect(&real_zip, &db, None, &mut no_progress()).unwrap();
        let report = import(
            &mut library,
            &db,
            &real_zip,
            &r,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), CODED.len(), "{}", shape.zip);
        let imported: Vec<_> = library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .collect();
        assert_eq!(imported.len(), CODED.len());
        assert!(imported.iter().all(|l| l.input == shape.input));
        assert!(imported.iter().any(|l| matches!(&l.group, LookGroup::Arri { group_no: 3000, group_name, .. } if group_name == "Film")));
        assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    }
}

#[test]
fn the_real_layouts_are_told_apart_by_their_names_and_sizes() {
    // The LogC4 zip's looks at the LogC3 size, or the LogC3 zips' at the LogC4 size, are not that package.
    let work = tempfile::tempdir().unwrap();
    let db = small_db(CODED.len());
    let shape = &SHAPES[0];
    let entries: Vec<(String, Vec<u8>)> = real_entries(shape)
        .into_iter()
        .map(|(path, bytes)| {
            if path.ends_with(".cube") && !path.starts_with("__MACOSX") {
                (path, cube_text(3, 1, "made up").into_bytes())
            } else {
                (path, bytes)
            }
        })
        .collect();
    let zip = work.path().join(shape.zip);
    write_zip(&zip, &entries);
    let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();
    assert!(r.packages.is_empty(), "{:?}", r.packages);
    assert_eq!(r.verdict(), Verdict::NeedsAnswer);
    // What the wizard preselects for it: LogC4, from the zip and folder names.
    assert!(r.files.iter().all(|f| matches!(
        f.outcome,
        Outcome::NeedsAnswer {
            suggested_input: Some(LookInput::LogC4),
            ..
        }
    )));
}
